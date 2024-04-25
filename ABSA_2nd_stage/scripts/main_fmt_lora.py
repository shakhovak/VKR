import logging
import argparse
import os
import warnings
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from datasets import concatenate_datasets
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from utils import (
    evaluate_flan,
    preprocess_function,
    createABSA_dataset,
    createASC_dataset,
    createATE_dataset,
)
import wandb
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    TaskType,
    PeftModel,
    PeftConfig,
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

    # read files in s3
    logger.info("Reading and augementing datasets...")
    raw_datasets = load_dataset(args.dataset_name, args.dataset_split)
    raw_datasets = concatenate_datasets(
        [raw_datasets["train"], raw_datasets["test"], raw_datasets["trial"]],
    )
    raw_datasets = raw_datasets.shuffle()

    if args.dataset_split == "restaurants":
        raw_datasets = raw_datasets.remove_columns(["aspectCategories"])
    # separete test dataset (not to be used in other sets creation!!!!)
    train_test_split = raw_datasets.train_test_split(test_size=0.05)

    ATE_dataset = train_test_split["train"].map(
        createATE_dataset, fn_kwargs={"prompts_file_path": args.input}
    )
    ASC_dataset = train_test_split["train"].map(
        createASC_dataset, fn_kwargs={"prompts_file_path": args.input}
    )
    ABSA_dataset = train_test_split["train"].map(
        createABSA_dataset, fn_kwargs={"prompts_file_path": args.input}
    )
    ABSA_test = train_test_split["test"].map(
        createABSA_dataset, fn_kwargs={"prompts_file_path": args.input}
    )

    combined_datasets = concatenate_datasets(
        [ATE_dataset, ASC_dataset, ABSA_dataset],
    )
    dataset_train_test = combined_datasets.train_test_split(test_size=0.1)

    final_ds = DatasetDict(
        {
            "train": dataset_train_test["train"],
            "test": dataset_train_test["test"],
            "trial": ABSA_test,
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id_tokenizer)

    # The maximum total input sequence length after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets(
        [
            final_ds["train"],
            final_ds["test"],
            final_ds["trial"],
        ],
    ).map(
        lambda x: tokenizer(x["aspect_polarities_input"], truncation=True),
        batched=True,
        remove_columns=[
            "sentenceId",
            "text",
            "aspect_list",
            "polarities_list",
            "aspectTerms",
            "aspect_polarities_list",
            "aspect_polarities_output",
            "aspect_polarities_input",
        ],
    )
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])

    tokenized_targets = concatenate_datasets(
        [
            final_ds["train"],
            final_ds["test"],
            final_ds["trial"],
        ]
    ).map(
        lambda x: tokenizer(x["aspect_polarities_output"], truncation=True),
        batched=True,
        remove_columns=[
            "sentenceId",
            "text",
            "aspect_list",
            "polarities_list",
            "aspectTerms",
            "aspect_polarities_list",
            "aspect_polarities_output",
            "aspect_polarities_input",
        ],
    )
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])

    tokenized_dataset = final_ds.map(
        preprocess_function,
        batched=True,
        fn_kwargs={
            "max_source_length": max_source_length,
            "max_target_length": max_target_length,
            "tokenizer": tokenizer,
        },
        remove_columns=[
            "sentenceId",
            "text",
            "aspect_list",
            "polarities_list",
            "aspectTerms",
            "aspect_polarities_list",
            "aspect_polarities_output",
            "aspect_polarities_input",
        ],
    )
    logger.info("Loading model...")
    # load model from the hub
    model_base = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id_model, load_in_8bit=True, device_map="auto"
    )
    # Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model_base)

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )
    HfFolder.save_token(os.environ["hugging_face_login"])

    os.environ["WANDB_PROJECT"] = "absa_fin"
    os.environ["WANDB_LOG_MODEL"] = "false"
    wandb.login(key=os.environ["wandb_login"])

    # Hugging Face repository id
    repository_id = f"{args.model_id_tokenizer.split('/')[1]}-absa-multitask-laptops"

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        fp16=False,  # Overflows with fp16
        learning_rate=5e-5,
        num_train_epochs=10,
        warmup_ratio=0.1,
        weight_decay=0.1,
        optim="adamw_torch",
        # logging & evaluation strategies
        evaluation_strategy="steps",
        eval_steps=200,
        report_to="wandb",
        logging_steps=200,
        save_strategy="no",
        save_total_limit=1,
        push_to_hub=False,
        hub_model_id=repository_id,
        hub_token=HfFolder.get_token(),
        run_name=args.experiment_name,
    )
    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )
    logger.info("Starting training...")
    trainer.train()

    # Save our tokenizer and create model card
    logger.info("Starting loading model to hub...")
    tokenizer.save_pretrained(repository_id)
    trainer.create_model_card()
    # Push the results to the hub
    trainer.push_to_hub()

    wandb.finish()
    # =============================================================================
    # Evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    peft_model_id = repository_id

    config = PeftConfig.from_pretrained(peft_model_id)

    # Load the Lora model
    model_eval = PeftModel.from_pretrained(
        model_base, peft_model_id, device_map={"": 0}
    )
    model_eval.eval()

    logger.info("Starting evaluation...")
    evaluate_flan(
        task="multi",
        exp_name=args.experiment_name,
        trial_dataset=final_ds["trial"],
        model=model_eval,
        tokenizer=tokenizer,
        device=device,
        prompts_path=args.input,
    )
    logger.info("All done.")
