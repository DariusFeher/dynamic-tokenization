from zett.tokenizer_converters import convert_to_byte_level
from transformers import AutoTokenizer
import torch
from transformers.modeling_flax_pytorch_utils import load_flax_weights_in_pytorch_model
from flax import serialization, traverse_util
import regex as re
from datasets import load_dataset
import pandas as pd
from adapters import AutoAdapterModel
from datasets.dataset_dict import DatasetDict
from typing import Dict, Union
from transformers import AutoModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import json
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    XLMRobertaForMaskedLM,
    AutoConfig,
    AutoModelForTokenClassification,
)
import evaluate
import numpy as np


def get_hypernet(device: str):
    hypernet = AutoModel.from_pretrained(
        "benjamin/zett-hypernetwork-xlm-roberta-base", trust_remote_code=True
    )
    return hypernet.to(device)


def get_model_from_mlm_with_adapter(
    exp_type: str,
    adapter_path: str,
    model_name: str = "xlm-roberta-base",
    mlm_model_path: str = "output_lora_peft/tokenizers/tokenizer_750k_samples_word_tk_hypernet_lora_alpha_512_drop_0.3_rank_256_seed_42_11062024/checkpoint-140000",
    best_adapter_criterion: str = "accuracy",
):
    config = AutoConfig.from_pretrained(mlm_model_path)
    model = XLMRobertaForMaskedLM.from_pretrained(mlm_model_path, config=config)

    config = PeftConfig.from_pretrained(adapter_path)
    model_classification = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )
    config.inference_mode = True
    if "checkpoint" not in adapter_path:
        adapter_path = get_best_checkpoint_adapter(
            adapter_path=adapter_path, criterion=best_adapter_criterion
        )
    print(f"NEW PATH {adapter_path}")

    model_classification.roberta.encoder = model.roberta.encoder

    model_classification = PeftModel.from_pretrained(model_classification, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if exp_type != "plain":
        tokenizer = convert_to_byte_level(tokenizer)[0]
    return model_classification, tokenizer


def get_merged_adapters_model_and_tokenizer(
    exp_type: str,
    model_name: str = "xlm-roberta-base",
    adapters: list = [],
    adapters_names: list = [],
    peft: bool = True,
    best_adapter_criterion: str = "accuracy",
):
    adapters_len = len(adapters)
    adapters_names_len = len(adapters_names)
    if adapters_len != adapters_names_len:
        raise Exception(
            f"The length of adapters and adapters names should be equal, but found {adapters_len} and {adapters_names_len}"
        )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    for idx, adapter_path in enumerate(adapters):
        if "checkpoint" not in adapter_path:
            adapter_path = get_best_checkpoint_adapter(
                adapter_path=adapter_path, criterion=best_adapter_criterion
            )
            print(f"NEW PATH {adapter_path}")
        if "tokenizers" not in adapter_path:
            model = PeftModel.from_pretrained(
                model, adapter_path, adapter_name=adapters_names[idx]
            )
        else:
            model.load_adapter(adapter_path, adapter_name=adapters_names[idx])
            del model.classifier.modules_to_save[adapters_names[idx]]
    weights = [0.4, 0.6]
    density = 0.8
    combination_type = "magnitude_prune"
    print(weights, combination_type, density)
    model.add_weighted_adapter(
        adapters=adapters_names,
        weights=weights,
        adapter_name="comb",
        combination_type=combination_type,
        density=density,
        # majority_sign_method="frequency"
    )

    model.set_adapter("comb")
    del model.classifier.modules_to_save["comb"]
    model.classifier.modules_to_save["comb"] = model.classifier.modules_to_save["xnli"]
    del model.classifier.modules_to_save["xnli"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if exp_type != "plain":
        tokenizer = convert_to_byte_level(tokenizer)[0]
    return model, tokenizer


def get_model_and_tokenizer(
    exp_type: str,
    model_name: str = "xlm-roberta-base",
    adapter_path: str = "artifacts/adapters/xnli_en/glue",
    peft: bool = False,
    best_adapter_criterion: str = "accuracy",
):
    if peft:
        config = PeftConfig.from_pretrained(adapter_path)
        config.inference_mode = True
        if "checkpoint" not in adapter_path:
            adapter_path = get_best_checkpoint_adapter(
                adapter_path=adapter_path, criterion=best_adapter_criterion
            )
        print(f"NEW PATH {adapter_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path, num_labels=3
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    else:
        model = AutoAdapterModel.from_pretrained(model_name)
        model.load_adapter(adapter_path, set_active=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if exp_type != "plain":
        tokenizer = convert_to_byte_level(tokenizer)[0]

    return model, tokenizer


def get_model_and_tokenizer_ner(
    exp_type: str,
    model_name: str = "xlm-roberta-base",
    adapter_path: str = "",
    peft: bool = False,
    best_adapter_criterion: str = "f1",
):
    if peft:
        config = PeftConfig.from_pretrained(adapter_path)
        config.inference_mode = True
        if "checkpoint" not in adapter_path:
            adapter_path = get_best_checkpoint_adapter(
                adapter_path=adapter_path, criterion=best_adapter_criterion
            )
        print(f"NEW PATH {adapter_path}")

        config2 = AutoConfig.from_pretrained(
            "xlm-roberta-base",
            num_labels=7,
            finetuning_task="ner",
        )
        model = AutoModelForTokenClassification.from_pretrained(
            config.base_model_name_or_path, config=config2
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    else:
        model = AutoAdapterModel.from_pretrained(model_name)
        model.load_adapter(adapter_path, set_active=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if exp_type != "plain":
        tokenizer = convert_to_byte_level(tokenizer)[0]

    return model, tokenizer


def get_merged_adapters_model_and_tokenizer_ner(
    exp_type: str,
    model_name: str = "xlm-roberta-base",
    adapters: list = [],
    adapters_names: list = [],
    peft: bool = True,
    best_adapter_criterion: str = "accuracy",
):
    adapters_len = len(adapters)
    adapters_names_len = len(adapters_names)
    if adapters_len != adapters_names_len:
        raise Exception(
            f"The length of adapters and adapters names should be equal, but found {adapters_len} and {adapters_names_len}"
        )
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=7)
    for idx, adapter_path in enumerate(adapters):
        if "checkpoint" not in adapter_path:
            adapter_path = get_best_checkpoint_adapter(
                adapter_path=adapter_path, criterion=best_adapter_criterion
            )
            print(f"NEW PATH {adapter_path}")
        if "tokenizers" not in adapter_path:
            print("HEREEE")
            model = PeftModel.from_pretrained(
                model, adapter_path, adapter_name=adapters_names[idx]
            )
        else:
            model.load_adapter(adapter_path, adapter_name=adapters_names[idx])
            del model.classifier.modules_to_save[adapters_names[idx]]

    weights = [1.5, 0.2]
    density = 0.8
    combination_type = "dare_linear"
    print(weights, combination_type, density)
    model.add_weighted_adapter(
        adapters=adapters_names,
        weights=weights,
        adapter_name="comb",
        combination_type=combination_type,
        density=density,
    )

    model.set_adapter("comb")
    del model.classifier.modules_to_save["comb"]
    model.classifier.modules_to_save["comb"] = model.classifier.modules_to_save["uner"]
    del model.classifier.modules_to_save["uner"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if exp_type != "plain":
        tokenizer = convert_to_byte_level(tokenizer)[0]
    return model, tokenizer


def get_dataset(name="xnli", language="en") -> DatasetDict:
    dataset = load_dataset(name, language)
    return dataset


def evaluate_xnli(
    encoded_dataset: DatasetDict, batch_size: int, model, device: str
) -> float:
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        data_loader = torch.utils.data.DataLoader(
            encoded_dataset, batch_size=batch_size
        )
        for batch in tqdm(data_loader, desc="Evaluating", total=len(data_loader)):
            inputs = {
                key: val.to(device) for key, val in batch.items() if key != "label"
            }
            outputs = model(**inputs)
            preds = torch.argmax(outputs[0], axis=-1)
            predictions.extend(preds.cpu().numpy())

    true_labels = encoded_dataset["label"]
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy


def evaluate_ner(
    encoded_dataset: DatasetDict, batch_size: int, model, device: str, label_list: list
) -> float:
    model.to(device)
    model.eval()
    predictions = []
    true_labels = []
    predictions = []

    with torch.no_grad():
        data_loader = torch.utils.data.DataLoader(
            encoded_dataset, batch_size=batch_size
        )
        for batch in tqdm(data_loader, desc="Evaluating"):
            # input_ids = batch['input_ids'].to(device)
            # attention_masks = batch['attention_mask'].to(device)
            label_ids = batch["labels"].to(device)

            inputs = {
                key: val.to(device) for key, val in batch.items() if key != "labels"
            }

            outputs = model(**inputs)
            logits = outputs.logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=2)

            # Remove ignored index (special tokens)
            batch_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(preds, label_ids)
            ]
            batch_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(preds, label_ids)
            ]
            predictions.extend(batch_predictions)
            true_labels.extend(batch_labels)

    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=predictions, references=true_labels)

    return (
        results["overall_precision"],
        results["overall_recall"],
        results["overall_f1"],
        results["overall_accuracy"],
    )


def get_best_checkpoint_adapter(
    adapter_path: str = "scripts/adapters/output_lora_peft/xnli_en_lora_alpha_32_drop_02_rank_16",
    criterion: str = "accuracy",
):
    state_file_path = os.path.join(adapter_path, "trainer_state.json")

    try:
        with open(state_file_path, "r") as file:
            data = json.load(file)
    except:
        raise Exception(f"File {state_file_path} not found!")

    best_score = None
    best_checkpoint_step = None

    for entry in data["log_history"]:
        if "eval_accuracy" in entry or "eval_loss" in entry or "eval_f1" in entry:
            current_score = entry.get(f"eval_{criterion}")
            # current_score = entry.get(
            #     'eval_accuracy' if criterion == 'accuracy' else 'eval_loss')

            if (
                best_score is None
                or (criterion == "accuracy" and current_score > best_score)
                or (criterion == "loss" and current_score < best_score)
                or (criterion == "f1" and current_score > best_score)
            ):
                best_score = current_score
                best_checkpoint_step = entry["step"]

    best_checkpoint_dir = os.path.join(
        adapter_path, f"checkpoint-{best_checkpoint_step}"
    )

    return best_checkpoint_dir
