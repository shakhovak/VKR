import logging
import argparse
import os
import warnings
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import concatenate_datasets
from huggingface_hub import HfFolder
import torch
from utils import (
    preprocess_llama,
    generate_and_tokenize_prompt,
    Prompter,
    evaluate_llama,
)
import wandb
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("dataset_name", help="name of dataset")
parser.add_argument("dataset_split", help="split of dataset")
parser.add_argument("model_id_tokenizer", help="model id from hugging face")
parser.add_argument("model_id_model", help="model id from hugging face")
parser.add_argument("experiment_name", help="name of experience")
parser.add_argument("-i", "--input", required=True, help="Input file")

if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
    logger = logging.getLogger()

    # read files
    logger.info("Reading and augementing datasets...")
    raw_datasets = load_dataset(args.dataset_name, args.dataset_split)
    raw_datasets = concatenate_datasets(
        [raw_datasets["train"], raw_datasets["test"], raw_datasets["trial"]],
    )
    raw_datasets = raw_datasets.shuffle()
    if args.dataset_split == "restaurants":
        raw_datasets = raw_datasets.remove_columns(["aspectCategories"])

    dataset_train_test = raw_datasets.train_test_split(test_size=0.1)
    dataset_test_valid = dataset_train_test["test"].train_test_split(test_size=0.5)
    final_ds = DatasetDict(
        {
            "train": dataset_train_test["train"],
            "test": dataset_test_valid["train"],
            "val": dataset_test_valid["test"],
        }
    )
    instruction_dataset = final_ds.map(
        preprocess_llama, fn_kwargs={"prompts_file_path": args.input}
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id_tokenizer)
    tokenizer.pad_token_id = 0

    prompter = Prompter()

    train_data = (
        instruction_dataset["train"]
        .shuffle()
        .map(
            generate_and_tokenize_prompt,
            fn_kwargs={
                "prompter": prompter,
                "tokenizer": tokenizer,
            },
        )
    )
    test_data = (
        instruction_dataset["test"]
        .shuffle()
        .map(
            generate_and_tokenize_prompt,
            fn_kwargs={
                "prompter": prompter,
                "tokenizer": tokenizer,
            },
        )
    )
    val_data = instruction_dataset["val"].shuffle()

    logger.info("Loading model...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # load model from the hub
    model_base = AutoModelForCausalLM.from_pretrained(
        args.model_id_model, load_in_8bit=True, torch_dtype=torch.float16
    )
    # Define LoRA Config
    lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model_base)

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)

    HfFolder.save_token(os.environ["hugging_face_login"])

    os.environ["WANDB_PROJECT"] = "absa_fin"
    os.environ["WANDB_LOG_MODEL"] = "false"
    wandb.login(key=os.environ["wandb_login"])

    # Hugging Face repository id
    repository_id = f"{args.model_id_tokenizer.split('/')[1]}-absa-{args.dataset_split}"

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    # Define training args
    training_args = TrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        num_train_epochs=1,
        max_steps=400,
        learning_rate=3e-4,
        fp16=True,
        optim="adamw_torch",
        # logging & evaluation strategies
        evaluation_strategy="steps",
        eval_steps=40,
        report_to="wandb",
        logging_steps=40,
        save_strategy="no",
        save_total_limit=1,
        push_to_hub=False,
        hub_model_id=repository_id,
        hub_token=HfFolder.get_token(),
        run_name=args.experiment_name,
    )
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=test_data,
    )
    model.config.use_cache = False
    logger.info("Starting training...")
    trainer.train()

    wandb.finish()
    # =============================================================================
    # Evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.config.use_cache = True
    model.eval()

    logger.info("Starting evaluation...")
    evaluate_llama(val_data, prompter, tokenizer, device, model, args.experiment_name)

    # Save our tokenizer and create model card
    logger.info("Starting loading model to hub...")
    tokenizer.save_pretrained(repository_id)
    trainer.create_model_card()
    # Push the results to the hub
    trainer.push_to_hub()

    logger.info("All done.")
