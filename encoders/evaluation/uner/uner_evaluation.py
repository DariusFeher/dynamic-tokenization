"""
UNER Evaluation Script

Functionality:
- Supports different tokenization strategies (original = plain, original_tk_hypernet = 
original tokenization with hypernetwork embeddings, word_tk_hypernet = word tokenization 
with hypernetwork embeddings, dynamic_bpe = dynamic tokenization with hypernetwork embeddings, etc.).
- Loads and evaluates models with or without adapters, including PEFT and merged adapters.
- Logs results and statistics to Weights & Biases (wandb) if enabled.

Usage:
    python3 encoders/evaluation/uner/uner_evaluation.py --exp_type $exp_type [...] - see below the full list or args

"""
import argparse
import os
import time
from collections import Counter
from typing import Dict

import pandas as pd
import torch
import wandb
from datasets import load_dataset
from encoders.evaluation.evaluation_utils import (evaluate_ner, get_dataset, get_hypernet,
                              get_merged_adapters_model_and_tokenizer_ner,
                              get_model_and_tokenizer_ner,
                              get_model_from_mlm_with_adapter)
from tokenizations.hypernet_cache import LRU_Cache
from tokenizations.tokenization_utils import DatasetEncoder
from tokenizations.tokenizers_utils import to_longest_prefix_tokenizer
from encoders.evaluation.uner.uner_utils import encode_examples_plain

parser = argparse.ArgumentParser(description="Running Dynamic Tokenization Experiments")
parser.add_argument(
    "--exp_type", type=str, default="plain", help="Type of tokenization used"
)
parser.add_argument(
    "--surface_form_maxlen",
    type=int,
    default=4,
    help="Maxlen used for surface forms before feeding them to the hypernet",
)
parser.add_argument("--cache_size", type=int, default=5000, help="Maximum cache size")
parser.add_argument("--no_wandb", action="store_true")
parser.add_argument(
    "--lng", type=str, default="en_ewt", help="Language on which we perform evaluation"
)
parser.add_argument(
    "--merges",
    type=int,
    default=500,
    help="Maximum nr of merges to perform during Dynamic BPE",
)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument(
    "--multiple_merges_exp",
    action="store_true",
    help="Used for experimentation with Dynamic BPE. Performs a sweep across multiple merge values",
)
parser.add_argument(
    "--bpe_token_boundary",
    type=str,
    default="pretokens",
    help="The boundary we want for merging subwords: pretokens, words, sentence",
)
parser.add_argument(
    "--adapter_path",
    type=str,
    default="output_lora_peft/UNER_subword_tk_en_lora_alpha_64_drop_0.3_rank_32_seed_42",
)  # /checkpoint-7448
parser.add_argument(
    "--best_adapter_criterion", type=str, default="f1"
)  # F1 for UNER, accuracy for XNLI
parser.add_argument("--peft", action="store_true", help="Experiment uses PEFT adapter")
parser.add_argument(
    "--exp_prefix",
    type=str,
    default="",
    help="Prefix to be added for the exp name in wandb",
)
parser.add_argument(
    "--adapters_to_merge", type=str, nargs="+", help="List of adapters to be merged"
)
parser.add_argument(
    "--adapters_names", type=str, nargs="+", help="List of adapters names to be merged"
)
parser.add_argument(
    "--mlm_model_path", type=str, default="", help="MLM model to use as base model"
)
parser.add_argument(
    "--use_lp_tokenizer", action="store_true", help="Use longest prefix tokenizer"
)

args = parser.parse_args()
exp_type = args.exp_type
cache_size = args.cache_size
max_nr_merges = args.merges
lng = args.lng
multiple_merges_exp = args.multiple_merges_exp
batch_size = args.batch_size
bpe_tokenizer_boundary = args.bpe_token_boundary
surface_form_maxlen = args.surface_form_maxlen
adapter_path = args.adapter_path
best_adapter_criterion = args.best_adapter_criterion
peft = args.peft
exp_prefix = args.exp_prefix
adapters_to_merge = args.adapters_to_merge or []
adapters_names = args.adapters_names or []
mlm_model_path = args.mlm_model_path

seq_lengths = []
tokens_processed_per_batch = []
unique_tokens_per_batch = []


dataset_split = ""
dataset = load_dataset("universalner/universal_ner", lng)

try:
    validation = dataset["validation"]  # .select(range(200))
    print("Using validation for evaluation")
    dataset_split = "validation"
    print(lng, "VALIDATION")
except:
    try:
        print(lng, "DEV")
        validation = dataset["dev"]
        print("Using dev for evaluation")
        dataset_split = "dev"
    except:
        print("TEST")
        validation = dataset["test"]
        print("Using test for evaluation")
        dataset_split = "test"
breakpoint()

if not args.no_wandb:
    adapter_name = os.path.basename(adapter_path)
    if exp_prefix:
        exp_name = f"{exp_prefix}_uner_ZeTT_{exp_type}_lng_{lng}_surflen_{surface_form_maxlen}_batch_{batch_size}_bpe_token_boundary_{bpe_tokenizer_boundary}_adapter_path_{adapter_name}"
    else:
        exp_name = f"uner_ZeTT_{exp_type}_lng_{lng}_surflen_{surface_form_maxlen}_batch_{batch_size}_bpe_token_boundary_{bpe_tokenizer_boundary}_adapter_path_{adapter_name}"
    wandb.init(
        project="dynamic-tokenization",
        # track hyperparameters and run metadata
        config={
            "dataset": "UNER",
            "exp_type": exp_type,
            "cache_size": cache_size,
            "surface_form_maxlen": surface_form_maxlen,
            "max_nr_merges": max_nr_merges,
            "adapter_path": adapter_path,
            "dataset": dataset_split,
        },
        name=exp_name,
    )

# PREPARE HYPERNET, MODEL and TOKENIZER
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
langs = [x.strip() for x in open("data/data/artifacts/26.txt")]
lang_index = torch.tensor(langs.index(lng.split("_")[0]), dtype=torch.int32).to(device)
hypernet = get_hypernet(device=device)

embeddings_cache = LRU_Cache(cache_size=cache_size, device=device)

if mlm_model_path != "":
    model, tokenizer = get_model_from_mlm_with_adapter(
        exp_type=exp_type,
        mlm_model_path=mlm_model_path,
        adapter_path=adapter_path,
        best_adapter_criterion=best_adapter_criterion,
    )
elif len(adapters_to_merge) == 0:
    model, tokenizer = get_model_and_tokenizer_ner(
        exp_type=exp_type,
        adapter_path=adapter_path,
        peft=peft,
        best_adapter_criterion=best_adapter_criterion,
    )
else:
    model, tokenizer = get_merged_adapters_model_and_tokenizer_ner(
        exp_type=exp_type,
        adapters=adapters_to_merge,
        adapters_names=adapters_names,
        best_adapter_criterion=best_adapter_criterion,
    )

if args.use_lp_tokenizer:
    tokenizer = to_longest_prefix_tokenizer(tokenizer)

source_embeddings = model.get_input_embeddings().weight.data.to(device)

datasetEncoder = DatasetEncoder(
    hypernet=hypernet,
    tokenizer=tokenizer,
    device=device,
    lang_index=lang_index,
    surface_form_maxlen=surface_form_maxlen,
    source_embeddings=source_embeddings,
    embeddings_cache=embeddings_cache,
    exp_type=exp_type,
    bpe_tokenizer_boundary=bpe_tokenizer_boundary,
    collect_extra_data=True,
    merges=max_nr_merges,
)

examples = pd.DataFrame(validation)
examples.head()

label_list = validation.features["ner_tags"].feature.names
# Required for encoding the dataset/batches
label_to_id = {i: i for i in range(len(label_list))}
b_to_i_label = []
for idx, label in enumerate(label_list):
    if label.startswith("B-") and label.replace("B-", "I-") in label_list:
        b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
    else:
        b_to_i_label.append(idx)

# EVALUATION
if exp_type == "plain":
    start_time = time.time()
    encoded_val = validation.map(
        encode_examples_plain, batched=True, batch_size=batch_size
    )
    end_time = time.time()
    encoding_time = end_time - start_time
    encoded_val.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
elif exp_type in ["original_tk_hypernet", "word_tk_hypernet", "fvt"] or (
    exp_type == "dynamic_bpe" and not multiple_merges_exp
):
    encoded_val, encoding_time = datasetEncoder.encode_dataset(
        dataset=validation,
        batch_size=32,
        merges=max_nr_merges,
        task="ner",
        label_all_tokens=False,
        label_to_id=label_to_id,
        b_to_i_label=b_to_i_label,
    )
    encoded_val.set_format(
        type="torch", columns=["inputs_embeds", "attention_mask", "labels"]
    )

if exp_type != "dynamic_bpe" or (exp_type == "dynamic_bpe" and not multiple_merges_exp):
    precision, recall, f1_score, overall_accuracy = evaluate_ner(
        encoded_dataset=encoded_val,
        batch_size=batch_size,
        model=model,
        device=device,
        label_list=label_list,
    )
else:
    datasetEncoder = DatasetEncoder(
        hypernet=hypernet,
        tokenizer=tokenizer,
        device=device,
        lang_index=lang_index,
        surface_form_maxlen=surface_form_maxlen,
        source_embeddings=source_embeddings,
        embeddings_cache=embeddings_cache,
        exp_type=exp_type,
        bpe_tokenizer_boundary=bpe_tokenizer_boundary,
        collect_extra_data=True,
    )
    if multiple_merges_exp:
        merges2f1_score = {}
        merges2seqLengths = {}
        merges, steps = 0, 1
        avg_length = 0

        while True:
            print(f"Encoding end evaluating with {merges} merges...")
            embeddings_cache = LRU_Cache(cache_size=cache_size, device=device)
            datasetEncoder.reset_state(embeddings_cache=embeddings_cache)

            encoded_val, encoding_time = datasetEncoder.encode_dataset(
                dataset=validation,
                batch_size=32,
                merges=merges,
                task="ner",
                label_all_tokens=False,
                label_to_id=label_to_id,
                b_to_i_label=b_to_i_label,
            )

            encoded_val.set_format(
                type="torch", columns=["inputs_embeds", "attention_mask", "labels"]
            )

            seq_lengths = datasetEncoder.seq_lengths
            curr_avg_length = sum(seq_lengths) / len(seq_lengths)
            print(curr_avg_length, avg_length)
            if curr_avg_length == avg_length:
                break

            merges2seqLengths[merges] = curr_avg_length
            avg_length = curr_avg_length

            precision, recall, f1_score, overall_accuracy = evaluate_ner(
                encoded_dataset=encoded_val,
                batch_size=batch_size,
                model=model,
                device=device,
                label_list=label_list,
            )
            merges2f1_score[merges] = f1_score
            merges += steps
            if merges == 10:
                steps = 5
            print(f"Avg. sequence lengths {avg_length}")
            print(f"F1-score {f1_score}")

        data_merges2seqLengths = [
            [nr_merges, length] for nr_merges, length in merges2seqLengths.items()
        ]
        table_merges2seqLengths = wandb.Table(
            data=data_merges2seqLengths,
            columns=["Number of Merges", "Avg Sequence Length"],
        )

        data_merges2f1_score = [
            [nr_merges, f1_score] for nr_merges, f1_score in merges2f1_score.items()
        ]
        table_merges2f1_score = wandb.Table(
            data=data_merges2f1_score, columns=["Number of Merges", "F1"]
        )

        data_f1_score_vs_seqLength = [
            [merges2seqLengths[merges], merges2f1_score[merges]]
            for merges in merges2f1_score.keys()
        ]
        table_f1_vs_seqLength = wandb.Table(
            data=data_f1_score_vs_seqLength, columns=["Avg Sequence Length", "F1"]
        )

        if not args.no_wandb:
            wandb.log(
                {
                    "F1 vs. Number of Merges": wandb.plot.line(
                        table_merges2f1_score,
                        "Number of Merges",
                        "F1",
                        title="F1 vs. Number of Merges",
                    ),
                    "Average Sequence Length vs. F1": wandb.plot.line(
                        table_merges2seqLengths,
                        "Number of Merges",
                        "Avg Sequence Length",
                        title="Average Sequence Length vs. Number of Merges",
                    ),
                    "F1 vs. Avg Sequence Length": wandb.plot.line(
                        table_f1_vs_seqLength,
                        "Avg Sequence Length",
                        "F1",
                        title="F1 vs. Avg Sequence Length",
                    ),
                }
            )
            wandb.finish()

if not args.no_wandb and (
    exp_type != "dynamic_bpe" or (exp_type == "dynamic_bpe" and not multiple_merges_exp)
):
    seq_lengths = datasetEncoder.seq_lengths
    tokens_processed_per_batch = datasetEncoder.tokens_processed_per_batch
    unique_tokens_per_batch = datasetEncoder.unique_tokens_per_batch
    length_counts = Counter(seq_lengths)
    avg_length = sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0
    print(avg_length)
    lengths = list(length_counts.keys())
    counts = list(length_counts.values())

    data = [[length, count] for length, count in sorted(length_counts.items())]
    table = wandb.Table(data=data, columns=["Sequence Length", "Count"])

    tokens_processed_counts = Counter(tokens_processed_per_batch)
    avg_tokens_processed = (
        sum(tokens_processed_per_batch) / len(tokens_processed_per_batch)
        if tokens_processed_per_batch
        else 0
    )

    nr_tokens_processed = list(tokens_processed_counts.keys())
    tokens_counts = list(tokens_processed_counts.values())
    table_tokens = wandb.Table(data=data, columns=["Tokens Processed", "Count"])

    avg_unique_tokens = (
        sum(unique_tokens_per_batch) / len(unique_tokens_per_batch)
        if unique_tokens_per_batch
        else 0
    )

    batches = list(range(1, len(tokens_processed_per_batch) + 1))
    data = list(zip(batches, tokens_processed_per_batch))
    table_tokens_per_batch = wandb.Table(
        data=data, columns=["Batch Number", "Tokens Processed"]
    )

    data_unique_tokens = list(zip(batches, unique_tokens_per_batch))
    table_unique_tokens_per_batch = wandb.Table(
        data=data_unique_tokens, columns=["Batch Number", "Unique Tokens"]
    )

    wandb.log(
        {
            "F1-score": f1_score,
            "Precision": precision,
            "Recall": recall,
            "Overall Accuracy": overall_accuracy,
            "Encoding_time": encoding_time,
            "Average Sequence Length": avg_length,
            "Sequence Length Distribution": wandb.plot.bar(
                table,
                "Sequence Length",
                "Count",
                title="Distribution of Sequence Lengths",
            ),
            "Average Unique Tokens per batch": avg_unique_tokens,
            "Average Tokens Processed per batch": avg_tokens_processed,
            "Frequency Distribution of Tokens Processed per batch": wandb.plot.bar(
                table_tokens,
                "Tokens Processed",
                "Count",
                title="Frequency Distribution of Tokens Processed per batch",
            ),
            "Unique Tokens Batch-by-Batch Evolution": wandb.plot.bar(
                table_unique_tokens_per_batch,
                "Batch Number",
                "Unique Tokens",
                title="Unique Tokens Batch-by-Batch Evolution",
            ),
            "Tokens Processed Batch-by-Batch Evolution": wandb.plot.bar(
                table_tokens_per_batch,
                "Batch Number",
                "Tokens Processed",
                title="Tokens Processed Batch-by-Batch Evolution",
            ),
        }
    )

    wandb.finish()

print(f"Adapter: {adapter_path}")
seq_lengths = datasetEncoder.seq_lengths
if seq_lengths:
    avg_length = sum(seq_lengths) / len(seq_lengths)
    print("Avg seq. len", avg_length)
print("Precision on the validation set:", precision)
print("Recall on the validation set:", recall)
print("F1-score on the validation set:", f1_score)
