import random
import json
import wandb
import pandas as pd
import logging
from typing import Union
import torch
from transformers import GenerationConfig
import re


def generate_response(
    task, model, tokenizer, question, top_p, temperature, prompts_path, device
):
    """ " function to generate response from FLAN models"""
    with open(prompts_path) as fp:
        template = json.load(fp)
    num = random.randint(0, len(template) - 1)
    if task == "one_c":
        instruction = template[str(num)]
        input = f"{instruction}\n{question}\n"
    else:
        instruction = template["ABSA"][str(num)]
        input = f"Task name: Aspect Term Extraction and Polarity Classification \n{instruction}\n{question}\n"
    input_ids = tokenizer.encode(input, return_tensors="pt")
    sample_output = model.generate(
        input_ids=input_ids.to(device),
        do_sample=True,
        max_length=1000,
        top_p=top_p,
        temperature=temperature,
        top_k=50,
        early_stopping=True,
    )
    out = tokenizer.decode(sample_output[0][1:], skip_special_tokens=True)
    if "</s>" in out:
        out = out[: out.find("</s>")].strip()
    return out


# ===============================================================================
def evaluate_flan(
    task,
    exp_name,
    trial_dataset,
    model,
    tokenizer,
    device,
    prompts_path,
    top_p=0.5,
    temperature=0.5,
):
    """function to evaluate generation results"""

    TP_aspect = 0
    FN_aspect = 0
    FP_aspect = 0
    TP_sent = 0
    FN_sent = 0
    FP_sent = 0
    answers = pd.DataFrame()

    run = wandb.init(
        project="absa_fin",
        name=exp_name,
    )

    wandb.define_metric("F1_A", summary="mean")
    wandb.define_metric("F1_S", summary="mean")
    wandb.define_metric("F1_macro_ABSA", summary="mean")
    wandb.define_metric("F1_micro_ABSA", summary="mean")

    logging.info(f"Evaluation for exp {exp_name} started...")

    for i in trial_dataset:
        answer = generate_response(
            task=task,
            model=model,
            tokenizer=tokenizer,
            question=i["text"],
            top_p=0.5,
            temperature=0.5,
            prompts_path=prompts_path,
            device=device,
        )
        answer = answer.split("Answer:")[1].strip().replace(": ", ":")
        new_row = {"y_pred": answer, "y_true": i["aspect_polarities_list"]}
        answers = pd.concat([answers, pd.DataFrame([new_row])])

        y_pred = answer.split(",")
        y_true = i["aspect_polarities_list"].split(",")

        aspects_true_lst = [item.split(":")[0] for item in y_true]
        aspects_pred_lst = [item.split(":")[0] for item in y_pred]

        for aspect in aspects_true_lst:
            if aspect in aspects_pred_lst:
                TP_aspect += 1
            else:
                FN_aspect += 1
        for aspect in aspects_pred_lst:
            if aspect not in aspects_true_lst:
                FP_aspect += 1
                FP_sent += 1

        for item in y_true:
            if item in y_pred:
                TP_sent += 1
            else:
                FN_sent += 1

    logging.info(f"Logging artefacts for experiement {exp_name} started...")
    data_to_log = wandb.Table(dataframe=answers)
    run.log({f"{exp_name}_preds": data_to_log})

    F1_aspect = 2 * TP_aspect / (2 * TP_aspect + FN_aspect + FP_aspect)
    F1_sent = 2 * TP_sent / (2 * TP_sent + FN_sent + FP_sent)
    F1_macro = (F1_aspect + F1_sent) / 2
    F1_micro = (
        2
        * (TP_aspect + TP_sent)
        / ((2 * (TP_aspect + TP_sent)) + (FN_aspect + FN_sent + FP_aspect + FP_sent))
    )

    log_dict = {
        "F1_A": F1_aspect,
        "F1_S": F1_sent,
        "F1_macro_ABSA": F1_macro,
        "F1_micro_ABSA": F1_micro,
    }
    wandb.log(log_dict)
    wandb.finish()
    logging.info("All done.")


# ===============================================================================


def preprocess_SemEval14(sample, prompts_file_path):
    with open(prompts_file_path) as fp:
        template = json.load(fp)

    num = random.randint(0, len(template) - 1)
    instruction = template[str(num)]

    sample["aspect_polarities_list"] = ",".join(
        [f"{item['term']}:{item['polarity']}" for item in sample["aspectTerms"]]
    )
    sample["aspect_polarities_output"] = f"Answer: \n{sample['aspect_polarities_list']}"
    sample["aspect_polarities_input"] = f"{instruction}\n{sample['text']}\n"
    return sample


# ===============================================================================


def preprocess_function(
    sample, tokenizer, max_source_length, max_target_length, padding="max_length"
):

    model_inputs = tokenizer(
        sample["aspect_polarities_input"],
        max_length=max_source_length,
        padding=padding,
        truncation=True,
    )
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=sample["aspect_polarities_output"],
        max_length=max_target_length,
        padding=padding,
        truncation=True,
    )
    # If we are padding here, replace all tokenizer.pad_token_id in the labels
    # by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ===============================================================================
def createASC_dataset(sample, prompts_file_path):
    with open(prompts_file_path) as fp:
        template = json.load(fp)

    num = random.randint(1, len(template) - 1)
    instruction = template["ASC"][str(num)]

    sample["aspect_list"] = ",".join([item["term"] for item in sample["aspectTerms"]])
    sample["polarities_list"] = ",".join(
        [item["polarity"] for item in sample["aspectTerms"]]
    )
    sample["aspect_polarities_list"] = ",".join(
        [f"{item['term']}:{item['polarity']}" for item in sample["aspectTerms"]]
    )
    sample["aspect_polarities_output"] = f"Answer: \n{sample['polarities_list']}"
    sample["aspect_polarities_input"] = (
        f"Task name: Sentiment Polarity Classification \n{instruction}\nSentence: \n{sample['text']}\nAspects mentioned: {sample['aspect_list']}\n"
    )
    return sample


# ===============================================================================


def createATE_dataset(sample, prompts_file_path):
    with open(prompts_file_path) as fp:
        template = json.load(fp)

    num = random.randint(1, len(template) - 1)
    instruction = template["ATE"][str(num)]

    sample["aspect_list"] = ",".join([item["term"] for item in sample["aspectTerms"]])
    sample["polarities_list"] = ",".join(
        [item["polarity"] for item in sample["aspectTerms"]]
    )
    sample["aspect_polarities_list"] = ",".join(
        [f"{item['term']}:{item['polarity']}" for item in sample["aspectTerms"]]
    )
    sample["aspect_polarities_output"] = f"Answer: \n{sample['aspect_list']}"
    sample["aspect_polarities_input"] = (
        f"Task name: Aspect Term Extraction \n{instruction}\n{sample['text']}\n"
    )
    return sample


# ===============================================================================
def createABSA_dataset(sample, prompts_file_path):
    with open(prompts_file_path) as fp:
        template = json.load(fp)

    num = random.randint(0, len(template) - 1)
    instruction = template["ABSA"][str(num)]

    sample["aspect_list"] = ",".join([item["term"] for item in sample["aspectTerms"]])
    sample["polarities_list"] = ",".join(
        [item["polarity"] for item in sample["aspectTerms"]]
    )
    sample["aspect_polarities_list"] = ",".join(
        [f"{item['term']}:{item['polarity']}" for item in sample["aspectTerms"]]
    )
    sample["aspect_polarities_output"] = f"Answer: \n{sample['aspect_polarities_list']}"
    sample["aspect_polarities_input"] = (
        f"Task name: Aspect Term Extraction and Polarity Classification \n{instruction}\n{sample['text']}\n"
    )
    return sample


# ===============================================================================


def preprocess_llama(sample, prompts_file_path):
    with open(prompts_file_path) as fp:
        template = json.load(fp)

    num = random.randint(0, len(template) - 1)
    instruction = template[str(num)]

    sample["instruction"] = f"""{instruction} {sample['text']} """

    sample["answer"] = ", ".join(
        [f"{item['term']}:{item['polarity']}" for item in sample["aspectTerms"]]
    )
    sample["output"] = (
        f"{sample['instruction']} Aspects and their polarity: {sample['answer']}"
    )
    sample["input"] = sample["text"]

    return sample


# ===============================================================================


def tokenize(prompt, tokenizer, cutoff_len=256, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


# ===============================================================================
def generate_and_tokenize_prompt(data_point, prompter, tokenizer):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt, tokenizer)
    user_prompt = prompter.generate_prompt(data_point["instruction"])
    tokenized_user_prompt = tokenize(user_prompt,
                                     tokenizer,
                                     add_eos_token=False)
    user_prompt_len = len(tokenized_user_prompt["input_ids"])

    tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt[
        "labels"
    ][
        user_prompt_len:
    ]  # could be sped up, probably
    return tokenized_full_prompt


# ===============================================================================


class Prompter(object):

    def generate_prompt(
        self,
        instruction: str,
        label: Union[None, str] = None,
    ) -> str:

        res = f"{instruction}\nAnswer: "

        if label:
            res = f"{res}{label}"

        return res

    def get_response(self, output: str) -> str:
        return output.split("Answer:")[1].strip()


# ===============================================================================


def generate_llama(
    text,
    prompter,
    tokenizer,
    device,
    model,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=2,
    max_new_tokens=256,
    **kwargs,
):
    prompt = prompter.generate_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True).strip()

    return prompter.get_response(output)


# ===============================================================================
def evaluate_llama(test_dataset, prompter, tokenizer, device, model, exp_name):
    TP_aspect = 0
    FN_aspect = 0
    FP_aspect = 0
    TP_sent = 0
    FN_sent = 0
    FP_sent = 0
    answers = pd.DataFrame()

    run = wandb.init(
        project="absa_fin",
        name=exp_name,
    )

    wandb.define_metric("F1_A", summary="mean")
    wandb.define_metric("F1_S", summary="mean")
    wandb.define_metric("F1_macro_ABSA", summary="mean")
    wandb.define_metric("F1_micro_ABSA", summary="mean")

    logging.info(f"Evaluation for exp {exp_name} started...")

    for item in test_dataset:
        answer = generate_llama(
            item["instruction"],
            prompter=prompter,
            tokenizer=tokenizer,
            device=device,
            model=model,
        )
        if "Aspects and their polarity:" in answer:
            answer = answer.split("Aspects and their polarity:")[1].strip()
            answer = re.sub("[^A-Za-z:, ]+", "", answer).strip()
        else:
            answer = answer.strip()
            answer = re.sub("[^a-zA-Z:, ]+", "", answer).strip()

        prediction = answer.lower()
        prediction = prediction.replace('aspect name:', "")
        prediction = (
            prediction.replace("aspects and their polarity:", "")
            .replace(".", "")
            .strip()
            .lower()
        )
        label = item["answer"].lower()
        label = re.sub("[^a-zA-Z:, ]+", "", label).strip()
        new_row = {"y_pred": prediction, "y_true": label}
        answers = pd.concat([answers, pd.DataFrame([new_row])])

        y_pred = prediction.split(",")
        y_true = label.split(",")

        aspects_true_lst = [item.split(":")[0] for item in y_true]
        aspects_pred_lst = [item.split(":")[0] for item in y_pred]

        for aspect in aspects_true_lst:
            if aspect in aspects_pred_lst:
                TP_aspect += 1
            else:
                FN_aspect += 1
        for aspect in aspects_pred_lst:
            if aspect not in aspects_true_lst:
                FP_aspect += 1
                FP_sent += 1

        for item in y_true:
            if item in y_pred:
                TP_sent += 1
            else:
                FN_sent += 1
    logging.info(f"Logging artefacts for experiement {exp_name} started...")
    data_to_log = wandb.Table(dataframe=answers)
    run.log({f"{exp_name}_preds": data_to_log})

    F1_aspect = 2 * TP_aspect / (2 * TP_aspect + FN_aspect + FP_aspect)
    F1_sent = 2 * TP_sent / (2 * TP_sent + FN_sent + FP_sent)
    F1_macro = (F1_aspect + F1_sent) / 2
    F1_micro = (
        2
        * (TP_aspect + TP_sent)
        / ((2 * (TP_aspect + TP_sent)) + (FN_aspect + FN_sent + FP_aspect + FP_sent))
    )

    log_dict = {
        "F1_A": F1_aspect,
        "F1_S": F1_sent,
        "F1_macro_ABSA": F1_macro,
        "F1_micro_ABSA": F1_micro,
    }
    wandb.log(log_dict)
    wandb.finish()
    logging.info("All logging is done.")

# ===============================================================================


def preprocess_llama_fmt(sample, prompts_file_path, dataset_task):
    with open(prompts_file_path) as fp:
        template = json.load(fp)

    num = random.randint(1, len(template) - 1)
    if dataset_task == "ATE":
        instruction = f"Task name: Aspect Term Extraction. {template[dataset_task][str(num)]}"
        sample["instruction"] = f"""{instruction}. Text: {sample['text']}"""
        sample["answer"] = ", ".join(
            [item["term"] for item in sample["aspectTerms"]]
        )
        sample["output"] = (
            f"{sample['instruction']} Aspects: {sample['answer']}"
        )
    elif dataset_task == "ACS":
        instruction = f"Task name: Sentiment Polarity Classification. {template[dataset_task][str(num)]}"
        sample["aspect_list"] = ", ".join([item["term"] for item in sample["aspectTerms"]])
        sample["answer"] = ",".join(
            [item["polarity"] for item in sample["aspectTerms"]]
        )
        sample["instruction"] = f"{instruction}. Sentence: {sample['text']} Aspects mentioned: {sample['aspect_list']}."
        sample["output"] = f"{sample['instruction']} Polarities of the aspects mentioned: {sample['answer']}."
    else:
        instruction = f"Task name: Aspect Term Extraction and Polarity Classification. {template[dataset_task][str(num)]}"
        sample["instruction"] = f"""{instruction} {sample['text']} """
        sample["answer"] = ", ".join(
            [f"{item['term']}:{item['polarity']}" for item in sample["aspectTerms"]]
        )
        sample["output"] = (
            f"{sample['instruction']} Aspects and their polarity: {sample['answer']}"
        )
    sample["input"] = sample["text"]

    return sample
# ===============================================================================


def preprocess_mistral(sample, prompts_file_path):
    with open(prompts_file_path) as fp:
        template = json.load(fp)

    num = random.randint(0, len(template) - 1)
    instruction = template[str(num)]

    sample["instruction"] = f"""<s>[INST] {instruction} {sample['text']} [/INST]"""

    sample["answer"] = ", ".join(
        [f"{item['term']}:{item['polarity']}" for item in sample["aspectTerms"]]
    )
    sample["output"] = (
        f"{sample['instruction']} Aspects and their polarity: {sample['answer']} </s>"
    )
    sample["input"] = sample["text"]

    return sample
# ===============================================================================


def preprocess_mistral_fmt(sample, prompts_file_path, dataset_task):
    with open(prompts_file_path) as fp:
        template = json.load(fp)

    num = random.randint(1, len(template) - 1)
    if dataset_task == "ATE":
        instruction = f"Task name: Aspect Term Extraction. {template[dataset_task][str(num)]}"
        sample["instruction"] = f"""[INST] {instruction}. Text: {sample['text']} [/INST]"""
        sample["answer"] = ", ".join(
            [item["term"] for item in sample["aspectTerms"]]
        )
        sample["output"] = (
            f"{sample['instruction']} Aspects: {sample['answer']}"
        )
    elif dataset_task == "ASC":
        instruction = f"Task name: Sentiment Polarity Classification. {template[dataset_task][str(num)]}"

        sample["aspect_list"] = ", ".join([item["term"] for item in sample["aspectTerms"]])
        sample["answer"] = ",".join(
            [item["polarity"] for item in sample["aspectTerms"]]
        )
        sample["instruction"] = f"[INST] {instruction}. Sentence: {sample['text']} Aspects mentioned: {sample['aspect_list']}. [/INST]"
        sample["output"] = f"{sample['instruction']} Polarities of the aspects mentioned: {sample['answer']}"                    
    else:
        instruction = f"Task name: Aspect Term Extraction and Polarity Classification. {template[dataset_task][str(num)]}"
        sample["instruction"] = f"""[INST] {instruction}. Text: {sample['text']} [/INST]"""
        sample["answer"] = ", ".join(
            [f"{item['term']}:{item['polarity']}" for item in sample["aspectTerms"]]
        )
        sample["output"] = (
            f"{sample['instruction']} Aspects and their polarity: {sample['answer']}"
        )
    sample["input"] = sample["text"]

    return sample
