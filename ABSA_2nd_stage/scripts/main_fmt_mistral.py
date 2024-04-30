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
    generate_and_tokenize_prompt,
    Prompter,
    evaluate_llama,
    preprocess_mistral_fmt,
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

    logger.info("Loading model...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # load model from the hub
    model_base = AutoModelForCausalLM.from_pretrained(
        args.model_id_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        token=os.environ["hf_readtoken"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id_tokenizer, token=os.environ["hf_readtoken"]
    )
    tokenizer.pad_token_id = 0
    # read files
    logger.info("Reading and augementing datasets...")
    raw_datasets = load_dataset(args.dataset_name, args.dataset_split)
    raw_datasets = concatenate_datasets(
        [raw_datasets["train"], raw_datasets["test"], raw_datasets["trial"]],
    )
    raw_datasets = raw_datasets.shuffle()
    if args.dataset_split == "restaurants":
        raw_datasets = raw_datasets.remove_columns(["aspectCategories"])

    # separete test dataset (not to be used in other sets creation!!!!)
    train_test_split = raw_datasets.train_test_split(test_size=0.04)
    ATE_dataset = train_test_split["train"].map(
        preprocess_mistral_fmt,
        fn_kwargs={"prompts_file_path": args.input, "dataset_task": "ATE"},
    )
    ASC_dataset = train_test_split["train"].map(
        preprocess_mistral_fmt,
        fn_kwargs={"prompts_file_path": args.input, "dataset_task": "ASC"},
    )
    ABSA_dataset = train_test_split["train"].map(
        preprocess_mistral_fmt,
        fn_kwargs={"prompts_file_path": args.input, "dataset_task": "ABSA"},
    )
    ABSA_test = train_test_split["test"].map(
        preprocess_mistral_fmt,
        fn_kwargs={"prompts_file_path": args.input, "dataset_task": "ABSA"},
    )
    combined_datasets = concatenate_datasets(
        [ATE_dataset, ASC_dataset, ABSA_dataset],
    )
    combined_datasets = combined_datasets.shuffle()
    dataset_train_test = combined_datasets.train_test_split(test_size=0.1)

    instruction_dataset = DatasetDict(
        {
            "train": dataset_train_test["train"],
            "test": dataset_train_test["test"],
            "val": ABSA_test,
        }
    )

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
    repository_id = (
        f"{args.model_id_tokenizer.split('/')[1]}-absa-MT-{args.dataset_split}"
    )

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
        max_steps=1200,
        learning_rate=3e-5,
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
