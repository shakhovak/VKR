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
from utils import preprocess_SemEval14, evaluate_flan, preprocess_function
import wandb

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("dataset_name", help="name of dataset")
parser.add_argument("dataset_split", help="split of dataset")
parser.add_argument("model_id", help="model id from hugging face")
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

    dataset_train_test = raw_datasets.train_test_split(test_size=0.1)
    dataset_test_valid = dataset_train_test["test"].train_test_split(test_size=0.5)
    final_ds = DatasetDict(
        {
            "train": dataset_train_test["train"],
            "test": dataset_test_valid["train"],
            "trial": dataset_test_valid["test"],
        }
    )
    augmented_dataset = final_ds.map(
        preprocess_SemEval14, fn_kwargs={"prompts_file_path": args.input}
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # The maximum total input sequence length after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets(
        [
            augmented_dataset["train"],
            augmented_dataset["test"],
            augmented_dataset["trial"],
        ],
    ).map(
        lambda x: tokenizer(x["aspect_polarities_input"], truncation=True),
        batched=True,
        remove_columns=[
            "sentenceId",
            "text",
            "aspectTerms",
            "aspect_polarities_list",
            "aspect_polarities_output",
            "aspect_polarities_input",
        ],
    )
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])

    tokenized_targets = concatenate_datasets(
        [
            augmented_dataset["train"],
            augmented_dataset["test"],
            augmented_dataset["trial"],
        ]
    ).map(
        lambda x: tokenizer(x["aspect_polarities_output"], truncation=True),
        batched=True,
        remove_columns=[
            "sentenceId",
            "text",
            "aspectTerms",
            "aspect_polarities_list",
            "aspect_polarities_output",
            "aspect_polarities_input",
        ],
    )
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])

    tokenized_dataset = augmented_dataset.map(
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
            "aspectTerms",
            "aspect_polarities_list",
            "aspect_polarities_output",
            "aspect_polarities_input",
        ],
    )
    logger.info("Loading model...")
    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)

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
    repository_id = f"{args.model_id.split('/')[1]}-absa-laptops"

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        fp16=False,  # Overflows with fp16
        learning_rate=5e-5,
        num_train_epochs=6,
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
    tokenizer_flan = AutoTokenizer.from_pretrained(repository_id)
    model_flan = AutoModelForSeq2SeqLM.from_pretrained(repository_id)
    model_flan.to(device)
    logger.info("Starting evaluation...")
    evaluate_flan(
        task="one_c",
        exp_name=args.experiment_name,
        trial_dataset=augmented_dataset["trial"],
        model=model_flan,
        tokenizer=tokenizer_flan,
        device=device,
        prompts_path=args.input,
    )
    logger.info("All done.")
