import random
import json
from tqdm import tqdm_notebook
import wandb
import pandas as pd
import logging


def generate_response(
    model, tokenizer, question, top_p, temperature, prompts_path, device
        ):
    """" function to generate response from FLAN models"""
    with open(prompts_path) as fp:
        template = json.load(fp)

    num = random.randint(0, len(template)-1)
    instruction = template[str(num)]
    input = (
        f"{instruction}\n{question}\n"
    )
    input_ids = tokenizer.encode(input, return_tensors="pt")
    sample_output = model.generate(
        input_ids.to(device),
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
def evaluate_flan(exp_name, trial_dataset, model, tokenizer, device, top_p=0.5, temperature=0.5):
    """ function to evaluate generation results """
    TP_aspect = 0
    FN_aspect = 0
    FP_aspect = 0
    TP_sent = 0
    FN_sent = 0
    FP_sent = 0
    answers = pd.DataFrame()

    run = wandb.init(
        project="absa_research2",
        name=exp_name,
    )

    wandb.define_metric("f1_aspect_ABSA", summary="mean")
    wandb.define_metric("f1_sent_ABSA", summary="mean")
    wandb.define_metric("f1_macro_ABSA", summary="mean")
    wandb.define_metric("f1_micro_ABSA", summary="mean")

    logging.info(f"Evaluation for exp {exp_name} started...")
    for i in tqdm_notebook(trial_dataset):
        answer = generate_response(
                        model=model,
                        tokenizer=tokenizer,
                        question=i['text'],
                        top_p=0.5,
                        temperature=0.5,
                        prompts_path='prompts_absa.json',
                        device=device
                    )
        new_row = {
            'y_pred': answer,
            'y_true': i['aspect_polarities_list']
        }
        answers = pd.concat([answers, pd.DataFrame([new_row])])

        answer = answer.split("Answer:")[1].strip().replace(": ", ":")
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
    F1_macro = (F1_aspect + F1_sent)/2
    F1_micro = 2 * (TP_aspect + TP_sent) / ((2 * (TP_aspect + TP_sent)) + (FN_aspect + FN_sent + FP_aspect+FP_sent))

    log_dict = {
        "f1_aspect_ABSA": F1_aspect,
        "f1_sent_ABSA": F1_sent,
        "f1_macro_ABSA": F1_macro,
        "f1_micro_ABSA": F1_micro,
    }
    wandb.log(log_dict)
    wandb.finish()
    logging.info("All done.")


# ===============================================================================
def preprocess_SemEval14(sample):
    with open("prompts_absa.json") as fp:
        template = json.load(fp)

    num = random.randint(0, len(template) - 1)
    instruction = template[str(num)]
    
    sample["aspect_polarities_list"] = ",".join([f"{item['term']}:{item['polarity']}" for item in sample['aspectTerms']])
    sample["aspect_polarities_output"] = f"Answer: \n{sample['aspect_polarities_list']}"
    sample["aspect_polarity_input"] = f"{instruction}\n{sample['text']}\n"
    return sample
