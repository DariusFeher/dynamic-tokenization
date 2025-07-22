"""
XNLI Evaluation Script

Functionality:
- Supports different tokenization strategies (original = plain, original_tk_hypernet = 
original tokenization with hypernetwork embeddings, word_tk_hypernet = word tokenization 
with hypernetwork embeddings, dynamic_bpe = dynamic tokenization with hypernetwork embeddings, etc.).
- Loads and evaluates models with or without adapters, including PEFT and merged adapters.
- Logs results and statistics to Weights & Biases (wandb) if enabled.

Usage:
    python3 encoders/evaluation/xnli/xnli_evaluation.py --exp_type $exp_type [...] - see below the full list or args

"""
import argparse
import os
import time
from collections import Counter
from typing import Dict

import pandas as pd
import torch
import wandb
from encoders.evaluation.evaluation_utils import (evaluate_xnli, get_dataset, get_hypernet,
                              get_merged_adapters_model_and_tokenizer,
                              get_model_and_tokenizer,
                              get_model_from_mlm_with_adapter)
from tokenizations.hypernet_cache import LRU_Cache
from tokenizations.tokenization_utils import DatasetEncoder
from tokenizations.tokenizers_utils import to_longest_prefix_tokenizer

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
    "--lng", type=str, default="en", help="Language on which we perform evaluation"
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
    default="output_lora_peft/xnli_en_lora_alpha_64_drop_0.3_rank_32_seed_42/checkpoint-110448",
)
parser.add_argument("--best_adapter_criterion", type=str, default="accuracy")
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


if not args.no_wandb:
    adapter_name = os.path.basename(adapter_path)
    if exp_prefix:
        exp_name = f"{exp_prefix}_xnli_ZeTT_{exp_type}_lng_{lng}_surflen_{surface_form_maxlen}_batch_{batch_size}_bpe_token_boundary_{bpe_tokenizer_boundary}_adapter_path_{adapter_name}"
    else:
        exp_name = f"xnli_ZeTT_{exp_type}_lng_{lng}_surflen_{surface_form_maxlen}_batch_{batch_size}_bpe_token_boundary_{bpe_tokenizer_boundary}_adapter_path_{adapter_name}"
    wandb.init(
        project="dynamic-tokenization",
        # track hyperparameters and run metadata
        config={
            "dataset": "XNLI",
            "exp_type": exp_type,
            "cache_size": cache_size,
            "surface_form_maxlen": surface_form_maxlen,
            "max_nr_merges": max_nr_merges,
            "adapter_path": adapter_path,
        },
        name=exp_name,
    )

# PREPARE HYPERNET, MODEL and TOKENIZER
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
langs = [x.strip() for x in open("data/artifacts/26.txt")]
lang_index = torch.tensor(langs.index(lng), dtype=torch.int32).to(device)
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
    model, tokenizer = get_model_and_tokenizer(
        exp_type=exp_type,
        adapter_path=adapter_path,
        peft=peft,
        best_adapter_criterion=best_adapter_criterion,
    )
else:
    model, tokenizer = get_merged_adapters_model_and_tokenizer(
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
dataset = get_dataset(name="xnli", language=lng)
examples = pd.DataFrame(dataset["validation"])
examples.head()
validation = dataset["validation"]

import random

random.seed(42)


def encode_examples_plain(examples, max_length: int = 128) -> Dict[str, torch.Tensor]:
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

print("Check language:", validation[0])

# EVALUATION
if exp_type == "plain":
    start_time = time.time()
    encoded_val = validation.map(
        encode_examples_plain, batched=True, batch_size=batch_size
    )
    end_time = time.time()
    encoding_time = end_time - start_time
    encoded_val.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
elif exp_type in ["original_tk_hypernet", "word_tk_hypernet", "fvt"] or (
    exp_type == "dynamic_bpe" and not multiple_merges_exp
):
    encoded_val, encoding_time = datasetEncoder.encode_dataset(
        dataset=validation, batch_size=32, merges=max_nr_merges
    )
    encoded_val.set_format(
        type="torch", columns=["inputs_embeds", "attention_mask", "label"]
    )

if exp_type != "dynamic_bpe" or (exp_type == "dynamic_bpe" and not multiple_merges_exp):
    accuracy = evaluate_xnli(
        encoded_dataset=encoded_val, batch_size=batch_size, model=model, device=device
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
        merges2accuracy = {}
        merges2seqLengths = {}
        merges, steps = 0, 10
        avg_length = 0

        while True:
            print(f"Encoding end evaluating with {merges} merges...")
            embeddings_cache = LRU_Cache(cache_size=cache_size, device=device)
            datasetEncoder.reset_state(embeddings_cache=embeddings_cache)

            encoded_val, encoding_time = datasetEncoder.encode_dataset(
                dataset=validation, batch_size=batch_size, merges=merges
            )
            encoded_val.set_format(
                type="torch", columns=["inputs_embeds", "attention_mask", "label"]
            )

            seq_lengths = datasetEncoder.seq_lengths
            curr_avg_length = sum(seq_lengths) / len(seq_lengths)
            print(curr_avg_length, avg_length)
            if curr_avg_length == avg_length:
                break

            merges2seqLengths[merges] = curr_avg_length
            avg_length = curr_avg_length
            accuracy = evaluate_xnli(
                encoded_dataset=encoded_val,
                batch_size=batch_size,
                model=model,
                device=device,
            )
            merges2accuracy[merges] = accuracy
            merges += steps
            print(f"Avg. sequence lengths {avg_length}")
            print(f"Accuracy {accuracy}")

        data_merges2seqLengths = [
            [nr_merges, length] for nr_merges, length in merges2seqLengths.items()
        ]
        table_merges2seqLengths = wandb.Table(
            data=data_merges2seqLengths,
            columns=["Number of Merges", "Avg Sequence Length"],
        )

        data_merges2accuracy = [
            [nr_merges, accuracy] for nr_merges, accuracy in merges2accuracy.items()
        ]
        table_merges2accuracy = wandb.Table(
            data=data_merges2accuracy, columns=["Number of Merges", "Accuracy"]
        )

        data_accuracy_vs_seqLength = [
            [merges2seqLengths[merges], merges2accuracy[merges]]
            for merges in merges2accuracy.keys()
        ]
        table_accuracy_vs_seqLength = wandb.Table(
            data=data_accuracy_vs_seqLength, columns=["Avg Sequence Length", "Accuracy"]
        )

        if not args.no_wandb:
            wandb.log(
                {
                    "Accuracy vs. Number of Merges": wandb.plot.line(
                        table_merges2accuracy,
                        "Number of Merges",
                        "Accuracy",
                        title="Accuracy vs. Number of Merges",
                    ),
                    "Average Sequence Length vs. Number of Merges": wandb.plot.line(
                        table_merges2seqLengths,
                        "Number of Merges",
                        "Avg Sequence Length",
                        title="Average Sequence Length vs. Number of Merges",
                    ),
                    "Accuracy vs. Avg Sequence Length": wandb.plot.line(
                        table_accuracy_vs_seqLength,
                        "Avg Sequence Length",
                        "Accuracy",
                        title="Accuracy vs. Avg Sequence Length",
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
            "Accuracy": accuracy,
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
print("Accuracy on the validation set:", accuracy)
