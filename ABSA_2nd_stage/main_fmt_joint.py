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

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("model_id", help="model id from hugging face")
parser.add_argument("experiment_name", help="name of experiment")
parser.add_argument("-i", "--input", required=True, help="Input file")

if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
    logger = logging.getLogger()

    # read files in s3
    logger.info("Reading and augementing datasets...")

    rest = load_dataset("alexcadillon/SemEval2014Task4", "restaurants")
    total_rest = concatenate_datasets(
        [rest["train"], rest["test"], rest["trial"]],
    )
    # separete test dataset (not to be used in other sets creation!!!!)
    train_test_rest = total_rest.train_test_split(test_size=0.05)

    laptops = load_dataset("alexcadillon/SemEval2014Task4", "laptops")
    total_laptops = concatenate_datasets(
        [laptops["train"], laptops["test"], laptops["trial"]],
    )
    # separete test dataset (not to be used in other sets creation!!!!)
    train_test_laptops = total_laptops.train_test_split(test_size=0.05)

    combined = concatenate_datasets([train_test_rest["train"],
                                     train_test_laptops["train"]])
    combined = combined.shuffle()

    ATE_dataset = combined.map(
        createATE_dataset, fn_kwargs={"prompts_file_path": args.input}
    )
    ASC_dataset = combined.map(
        createASC_dataset, fn_kwargs={"prompts_file_path": args.input}
    )
    ABSA_dataset = combined.map(
        createABSA_dataset, fn_kwargs={"prompts_file_path": args.input}
    )

    ABSA_test_rest = train_test_rest["test"].map(
        createABSA_dataset, fn_kwargs={"prompts_file_path": args.input}
    )
    ABSA_test_laptops = train_test_laptops["test"].map(
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
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # The maximum total input sequence length after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets(
        [final_ds["train"], final_ds["test"], ABSA_test_rest, ABSA_test_laptops],
    ).map(
        lambda x: tokenizer(x["aspect_polarities_input"], truncation=True),
        batched=True,
        remove_columns=[
            "sentenceId",
            "text",
            "aspect_list",
            "polarities_list",
            "aspectTerms",
            "aspectCategories",
            "aspect_polarities_list",
            "aspect_polarities_output",
            "aspect_polarities_input",
        ],
    )
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])

    tokenized_targets = concatenate_datasets(
        [final_ds["train"], final_ds["test"], ABSA_test_rest, ABSA_test_laptops]
    ).map(
        lambda x: tokenizer(x["aspect_polarities_output"], truncation=True),
        batched=True,
        remove_columns=[
            "sentenceId",
            "text",
            "aspect_list",
            "polarities_list",
            "aspectTerms",
            "aspectCategories",
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
            "aspectCategories",
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
    os.environ["WANDB_LOG_MODEL"] = "end"
    wandb.login(key=os.environ["wandb_login"])

    # Hugging Face repository id
    repository_id = f"{args.model_id.split('/')[1]}-absa-multitask-joint"

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        fp16=False,  # Overflows with fp16
        learning_rate=5e-5,
        num_train_epochs=4,
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
        task="multi",
        exp_name=args.experiment_name + "-" + "R",
        trial_dataset=ABSA_test_rest,
        model=model_flan,
        tokenizer=tokenizer_flan,
        device=device,
        prompts_path=args.input,
    )

    evaluate_flan(
        task="multi",
        exp_name=args.experiment_name + "-" + "L",
        trial_dataset=ABSA_test_laptops,
        model=model_flan,
        tokenizer=tokenizer_flan,
        device=device,
        prompts_path=args.input,
    )
    logger.info("All done.")
