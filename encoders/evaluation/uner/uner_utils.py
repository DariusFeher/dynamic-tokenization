"""
UNER Evaluation Utilities

This module provides helper functions for the UNER evaluation pipeline.

"""
import pandas as pd
import torch

def get_label_mappings(validation):
    label_list = validation.features["ner_tags"].feature.names
    label_to_id = {i: i for i in range(len(label_list))}
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)
    return label_list, label_to_id, b_to_i_label

def encode_examples_plain(examples, tokenizer, validation, label_all_tokens=False):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        padding="max_length",
        truncation=True,
        max_length=128,
        is_split_into_words=True,
        return_tensors="pt",
    )
    labels = []
    label_list = validation.features["ner_tags"].feature.names
    label_to_id = {i: i for i in range(len(label_list))}
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            else:
                if label_all_tokens:
                    label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs 