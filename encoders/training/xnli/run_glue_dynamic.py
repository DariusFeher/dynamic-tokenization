#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Finetuning the library models for sequence classification on GLUE.

This script has been adapted to support dynamic tokenization with hypernetwork embeddings.
"""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
import transformers.utils
import wandb
from datasets import load_dataset
from peft import AdaLoraConfig, LoraConfig, PeftConfig, PeftModel, TaskType
from scipy.stats import t
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModel,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, DefaultDataCollator,
                          EvalPrediction, HfArgumentParser, PretrainedConfig,
                          Trainer, TrainingArguments, default_data_collator)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_datasets_available
from transformers.utils.versions import require_version
from zett.tokenizer_converters import convert_to_byte_level

HOME_PATH = "/mnt/nas_home/dmf45/dynamic_tokenization"
sys.path.insert(0, HOME_PATH)

from tokenizations.tokenization_utils import DatasetEncoder
from tokenizations.hypernet_cache import LRU_Cache
from encoders.evaluation.evaluation_utils import get_hypernet

# Number of merges to use for dynamic tokenization and the batch size
MERGES = 140
BATCH_SIZE = 32 

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(mode=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def seed_worker(_) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CustomTrainer(Trainer):
    def set_dataset_encoder_and_tokeniser_sampling(self, datasetEncoder):
        self.datasetEncoder = datasetEncoder
        self.do_tokeniser_sampling_per_sample = False
        self.do_tokeniser_sampling_per_batch = False
        self.do_tokeniser_sampling_per_batch_gaussian = False
        self.do_tokeniser_sampling_per_sample_gaussian = False
        self.do_tokeniser_sampling_per_batch_cauchy = False
        self.do_tokeniser_sampling_per_batch_student_t = False

    def init_tokeniser_sampling(
        self,
        device: str,
        file_path: str = "data/xnli/sequence_length_to_merges_dynamic_bpe.csv",
        do_tokeniser_sampling_per_sample: bool = False,
        do_tokeniser_sampling_per_batch: bool = False,
        max_bins: int = 30,
        do_tokeniser_sampling_per_batch_gaussian: bool = False,
        do_tokeniser_sampling_per_sample_gaussian: bool = False,
        do_tokeniser_sampling_per_batch_cauchy: bool = False,
        do_tokeniser_sampling_per_batch_student_t: bool = False,
    ) -> None:
        assert (
            do_tokeniser_sampling_per_sample
            or do_tokeniser_sampling_per_batch
            or do_tokeniser_sampling_per_batch_gaussian
            or do_tokeniser_sampling_per_sample_gaussian
            or do_tokeniser_sampling_per_batch_student_t
            or do_tokeniser_sampling_per_batch_cauchy
        )
        self.do_tokeniser_sampling_per_sample = do_tokeniser_sampling_per_sample
        self.do_tokeniser_sampling_per_batch = do_tokeniser_sampling_per_batch
        self.do_tokeniser_sampling_per_batch_gaussian = (
            do_tokeniser_sampling_per_batch_gaussian
        )
        self.do_tokeniser_sampling_per_sample_gaussian = (
            do_tokeniser_sampling_per_sample_gaussian
        )
        self.do_tokeniser_sampling_per_batch_cauchy = (
            do_tokeniser_sampling_per_batch_cauchy
        )
        self.do_tokeniser_sampling_per_batch_student_t = (
            do_tokeniser_sampling_per_batch_student_t
        )
        self.device = device
        self.tokeniser_sampled_percentages = []
        data = pd.read_csv(file_path)
        initial_sequence_length = data["Avg Sequence Length"].max()
        final_sequence_length = data["Avg Sequence Length"].min()
        data["Sequence Length Reduction %"] = (
            100
            * (initial_sequence_length - data["Avg Sequence Length"])
            / (initial_sequence_length - final_sequence_length)
        )

        self.max_bins = max_bins
        self.bin_counts, bin_edges = np.histogram(
            data["Sequence Length Reduction %"], bins=self.max_bins
        )
        values = data["Sequence Length Reduction %"].values

        bin_indices = np.digitize(values, bin_edges)
        self.binned_values = {i: [] for i in range(1, len(bin_edges))}

        for value, bin_index in zip(values, bin_indices):
            if bin_index < len(bin_edges):
                self.binned_values[bin_index].append(value)

        self.binned_values[self.max_bins].append(100.0)

        self.seqReduction2merges = dict(
            zip(data["Sequence Length Reduction %"], data["Number of Merges"])
        )
        self.seqReduction2merges[100.0] = 200

    def sample_tokenisers(self, batch_size: int) -> list:
        sampled_bins = np.random.choice(
            range(len(self.bin_counts)), size=batch_size, replace=True
        )

        sampled_percentages = [
            (
                np.random.choice(self.binned_values[bin_index + 1])
                if self.binned_values[bin_index + 1]
                else np.nan
            )
            for bin_index in sampled_bins
        ]
        merges = [
            self.seqReduction2merges[seqReduction]
            for seqReduction in sampled_percentages
        ]

        self.tokeniser_sampled_percentages.extend(sampled_percentages)
        return merges

    def sample_tokenisers_gaussian(self, batch_size: int) -> list:
        middle_bin_index = self.max_bins // 2
        mean_bin = middle_bin_index
        std_dev_bin = self.max_bins / 6

        sampled_bin_indices = np.random.normal(mean_bin, std_dev_bin, batch_size)
        sampled_bin_indices = np.clip(
            np.round(sampled_bin_indices), 0, self.max_bins - 1
        ).astype(int)

        sampled_percentages = [
            (
                np.random.choice(self.binned_values[bin_index + 1])
                if self.binned_values[bin_index + 1]
                else np.nan
            )
            for bin_index in sampled_bin_indices
        ]
        merges = [
            self.seqReduction2merges[seqReduction]
            for seqReduction in sampled_percentages
        ]

        self.tokeniser_sampled_percentages.extend(sampled_percentages)
        return merges

    def sample_tokeniser_gaussian(self) -> int:
        middle_bin_index = self.max_bins // 2
        mean_bin = middle_bin_index
        std_dev_bin = self.max_bins / 6

        sampled_bin = np.random.normal(mean_bin, std_dev_bin, 1)[0]
        sampled_bin = np.clip(np.round(sampled_bin), 0, self.max_bins - 1).astype(int)

        sampled_percentage_reduction = np.random.choice(
            self.binned_values[sampled_bin + 1]
        )
        merges = self.seqReduction2merges[sampled_percentage_reduction]
        self.tokeniser_sampled_percentages.append(sampled_percentage_reduction)

        return merges

    def sample_tokeniser(self) -> int:
        sampled_bin = np.random.choice(
            range(len(self.bin_counts)), size=1, replace=True
        )[0]

        sampled_percentage_reduction = np.random.choice(
            self.binned_values[sampled_bin + 1]
        )
        merges = self.seqReduction2merges[sampled_percentage_reduction]
        self.tokeniser_sampled_percentages.append(sampled_percentage_reduction)
        return merges

    def sample_tokeniser_cauchy(self) -> int:
        sampled_bin_idx = []
        while len(sampled_bin_idx) < 1:
            sample = t.rvs(1, loc=self.max_bins // 2, scale=6)
            if 0 <= sample <= 29:
                sampled_bin_idx.append(sample)

        sampled_bin_idx = np.array(sampled_bin_idx)[0]
        sampled_bin = np.clip(np.round(sampled_bin_idx), 0, self.max_bins - 1).astype(
            int
        )

        sampled_percentage_reduction = np.random.choice(
            self.binned_values[sampled_bin + 1]
        )
        merges = self.seqReduction2merges[sampled_percentage_reduction]
        self.tokeniser_sampled_percentages.append(sampled_percentage_reduction)
        return merges

    def sample_tokeniser_student_t(self) -> int:
        sampled_bin_idx = []
        while len(sampled_bin_idx) < 1:
            sample = t.rvs(10, loc=self.max_bins // 2, scale=6)
            if 0 <= sample <= 29:
                sampled_bin_idx.append(sample)

        sampled_bin_idx = np.array(sampled_bin_idx)[0]
        sampled_bin = np.clip(np.round(sampled_bin_idx), 0, self.max_bins - 1).astype(
            int
        )

        sampled_percentage_reduction = np.random.choice(
            self.binned_values[sampled_bin + 1]
        )
        merges = self.seqReduction2merges[sampled_percentage_reduction]
        self.tokeniser_sampled_percentages.append(sampled_percentage_reduction)
        return merges

    def _prepare_inputs(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if (
            "inputs_embeds" in inputs and "attention_mask" in inputs
        ):  # Used for evaluation which is already tokenised
            return inputs

        if (
            self.do_tokeniser_sampling_per_sample
            or self.do_tokeniser_sampling_per_sample_gaussian
        ):
            if self.do_tokeniser_sampling_per_sample:
                tokenisers_merges = self.sample_tokenisers(
                    batch_size=self._train_batch_size
                )
            else:
                tokenisers_merges = self.sample_tokenisers_gaussian(
                    batch_size=self._train_batch_size
                )
            sequence_batch_embeddings = []
            attention_masks = []

            for idx in range(len(inputs["labels"])):
                self.datasetEncoder.merges = tokenisers_merges[idx]
                result = self.datasetEncoder.encode_examples_unique_tokens_lru(
                    examples=inputs, tokenise_idx=idx
                )
                sequence_batch_embeddings.append(result["inputs_embeds"])
                attention_masks.append(result["attention_mask"])

            encoded_batch = {}
            encoded_batch["inputs_embeds"] = torch.stack(sequence_batch_embeddings).to(
                self.device
            )
            encoded_batch["attention_mask"] = torch.stack(attention_masks).to(
                self.device
            )
        elif (
            self.do_tokeniser_sampling_per_batch
            or self.do_tokeniser_sampling_per_batch_gaussian
            or self.do_tokeniser_sampling_per_batch_cauchy
            or self.do_tokeniser_sampling_per_batch_student_t
        ):
            if self.do_tokeniser_sampling_per_batch:  # Uniform distribution
                tokeniser_merges = self.sample_tokeniser()
            elif self.do_tokeniser_sampling_per_batch_gaussian:
                tokeniser_merges = self.sample_tokeniser_gaussian()
            elif self.do_tokeniser_sampling_per_batch_cauchy:
                tokeniser_merges = self.sample_tokeniser_cauchy()
            elif self.do_tokeniser_sampling_per_batch_student_t:
                tokeniser_merges = self.sample_tokeniser_student_t()
            self.datasetEncoder.merges = tokeniser_merges
            encoded_batch = self.datasetEncoder.encode_examples_unique_tokens_lru(
                examples=inputs
            )
        else:
            encoded_batch = self.datasetEncoder.encode_examples_unique_tokens_lru(
                examples=inputs
            )
        encoded_batch["labels"] = torch.tensor(inputs["labels"])
        return encoded_batch

    def get_train_dataloader(self):
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        g = torch.Generator()
        g.manual_seed(self.seed)
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            dataloader_params["generator"] = g

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


class DynamicDataCollator(DefaultDataCollator):
    def __init__(
        self,
        tokenizer,
        hypernet,
        device,
        lang_index,
        surface_form_maxlen,
        source_embeddings,
        embeddings_cache,
        exp_type,
        bpe_tokenizer_boundary,
        max_length=128,
        merges=0,
    ):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.datasetEncoder = DatasetEncoder(
            hypernet=hypernet,
            tokenizer=tokenizer,
            device=device,
            lang_index=lang_index,
            surface_form_maxlen=surface_form_maxlen,
            source_embeddings=source_embeddings,
            embeddings_cache=embeddings_cache,
            exp_type=exp_type,
            bpe_tokenizer_boundary=bpe_tokenizer_boundary,
            collect_extra_data=False,
            merges=merges,
        )
        self.merges = merges

    def __call__(self, examples):
        """
        Converts list of dictionaries to dictionaries of list to ease post-processing implementation of Trainer
        """
        if "inputs_embeds" in examples[0] and "attention_mask" in examples[0]:
            examples = {
                key if key != "label" else "labels": torch.stack(
                    [item[key] for item in examples]
                )
                for key in ["inputs_embeds", "attention_mask", "label"]
            }
        else:
            examples = {
                key if key != "label" else "labels": [dic[key] for dic in examples]
                for key in examples[0]
            }

        return examples


@dataclass
class CustomTrainingArguments(TrainingArguments):
    cache_size: int = field(
        default=10000, metadata={"help": "Size to use for hypernet embeddings cache."}
    )
    surface_form_maxlen: int = field(
        default=7,
        metadata={"help": "Maximum surface form length to be used for hypernet."},
    )
    exp_type: str = field(
        default="dynamic_bpe",
        metadata={
            "help": "The type of experiment which defines the tokenization method: plain, original_tk_hypernet, word_tk_hypernet, dynamic_bpe"
        },
    )
    bpe_tokenizer_boundary: str = field(
        default="pretokens",
        metadata={
            "help": "The boundary we want for merging subwords: pretokens, words, sentence"
        },
    )
    collect_extra_data: bool = field(
        default=True,
        metadata={
            "help": "Collect extra data such as average sequence lengths for a specific tokenization."
        },
    )
    do_tokeniser_sampling_per_sample: bool = field(
        default=False,
        metadata={
            "help": "Sample (Uniform) which sequence reduction tokeniser to use for each sample in the batch"
        },
    )
    do_tokeniser_sampling_per_batch: bool = field(
        default=False,
        metadata={
            "help": "Sample (Uniform) which sequence reduction tokeniser to use for each batch"
        },
    )
    do_tokeniser_sampling_per_sample_gaussian: bool = field(
        default=False,
        metadata={
            "help": "Sample (Gaussian) which sequence reduction tokeniser to use for each sample in the batch"
        },
    )
    do_tokeniser_sampling_per_batch_gaussian: bool = field(
        default=False,
        metadata={
            "help": "Sample (Gaussian) which sequence reduction tokeniser to use for each batch"
        },
    )
    do_tokeniser_sampling_per_batch_cauchy: bool = field(
        default=False,
        metadata={
            "help": "Sample (Cauchy) which sequence reduction tokeniser to use for each batch"
        },
    )
    do_tokeniser_sampling_per_batch_student_t: bool = field(
        default=False,
        metadata={
            "help": "Sample (Student-t) which sequence reduction tokeniser to use for each batch"
        },
    )


@dataclass
class AdapterArguments:
    lora: bool = field(default=False, metadata={"help": "Use PEFT-LoRA"})
    adalora: bool = field(default=False, metadata={"help": "Use PEFT-AdaLoRA"})
    train_adapter: bool = field(
        default=False, metadata={"help": "Choose whether to train LoRA adapter"}
    )
    lora_alpha: int = field(
        default=16, metadata={"help": "Alpha value for LoRA layers"}
    )
    lora_dropout: float = field(
        default=0.1, metadata={"help": "Dropout rate for LoRA layers"}
    )
    lora_rank: int = field(default=4, metadata={"help": "Rank for LoRA projection"})
    lora_bias: str = field(default="none", metadata={"help": "Bias to use for LoRA"})
    adalora_init_r: int = field(
        default=12, metadata={"help": "The initial rank for each incremental matrix"}
    )
    adalora_tinit: str = field(
        default=200, metadata={"help": "The steps of initial fine-tuning warmup"}
    )
    adalora_tfinal: str = field(
        default=1000, metadata={"help": "The step of final fine-tuning"}
    )
    adalora_deltaT: str = field(
        default=10,
        metadata={"help": "The time internval between two budget allocations"},
    )


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(task_to_keys.keys())
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in "
                    + ",".join(task_to_keys.keys())
                )
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError(
                "Need either a GLUE task, a training/validation file or a dataset name."
            )
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )
    tokenization_type: str = field(
        default="static",
        metadata={"help": "Will use either static or dynamic tokenization."},
    )
    dynamic_tokenization_merges: Optional[int] = field(
        default=0,
        metadata={
            "help": "Will perform the specified number of BPE merges during tokenization."
        },
    )
    further_training_adapter_path: str = field(
        default="",
        metadata={"help": "The adapter path that will be further fine-tuned."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            CustomTrainingArguments,
            AdapterArguments,
        )
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            adapter_args,
        ) = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print(f"Adapter Args: {adapter_args}")
    print(f"Model Args: {model_args}")
    print(f"Training Args: {training_args}")
    print(f"Data Args: {data_args}")

    training_args.remove_unused_columns = False
    training_args.data_seed = training_args.seed

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    setup_seed(training_args.seed)
    # seed_everything(training_args.seed)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {
            "train": data_args.train_file,
            "validation": data_args.validation_file,
        }

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError(
                    "Need either a GLUE task or a test file for `do_predict`."
                )

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in [
            "float32",
            "float64",
        ]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.further_training_adapter_path == "":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    else:
        peft_config = PeftConfig.from_pretrained(
            model_args.further_training_adapter_path
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path, num_labels=3
        )
        model = PeftModel.from_pretrained(
            model, model_args.further_training_adapter_path
        )
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

    # Convert the model into an adapter model
    if adapter_args.lora:
        logger.info(f"Using PEFT-LoRA")
        peft_config = LoraConfig(
            lora_alpha=adapter_args.lora_alpha,
            lora_dropout=adapter_args.lora_dropout,
            r=adapter_args.lora_rank,
            bias=adapter_args.lora_bias,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            modules_to_save=["classifier"],
        )
    elif adapter_args.adalora:
        logger.info(f"Using PEFT-AdaLoRA")
        peft_config = AdaLoraConfig(
            lora_alpha=adapter_args.lora_alpha,
            lora_dropout=adapter_args.lora_dropout,
            r=adapter_args.lora_rank,
            tinit=adapter_args.adalora_tinit,
            tfinal=adapter_args.adalora_tfinal,
            deltaT=adapter_args.adalora_deltaT,
            bias=adapter_args.lora_bias,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            modules_to_save=["classifier"],
            target_modules=["value", "query"],
        )
    if (
        adapter_args.lora or adapter_args.adalora
    ) and model_args.further_training_adapter_path == "":
        logger.info("Loading PEFT-adapter")
        model.add_adapter(peft_config)

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        sentence1_key, sentence2_key = "premise", "hypothesis"

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [
                (label_to_id[l] if l != -1 else -1) for l in examples["label"]
            ]
        return result

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name)
    else:
        metric = evaluate.load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    raw_datasets["train"] = raw_datasets["train"]
    raw_datasets["validation"] = raw_datasets["validation"]
    with training_args.main_process_first(desc="dataset map pre-processing"):
        if model_args.tokenization_type == "dynamic":
            tokenizer = convert_to_byte_level(tokenizer)[0]
            langs = [x.strip() for x in open("data/artifacts/26.txt")]
            lang_index = torch.tensor(
                langs.index(data_args.dataset_config_name), dtype=torch.int32
            ).to(training_args.device)
            hypernet = get_hypernet(device=training_args.device)
            source_embeddings = model.get_input_embeddings().weight.data.to(
                training_args.device
            )
            embeddings_cache = LRU_Cache(cache_size=10000, device=training_args.device)
            datasetEncoder = DatasetEncoder(
                hypernet=hypernet,
                tokenizer=tokenizer,
                device=training_args.device,
                lang_index=lang_index,
                surface_form_maxlen=training_args.surface_form_maxlen,
                source_embeddings=source_embeddings,
                embeddings_cache=embeddings_cache,
                exp_type=training_args.exp_type,
                bpe_tokenizer_boundary=training_args.bpe_tokenizer_boundary,
                merges=model_args.dynamic_tokenization_merges,
                collect_extra_data=training_args.collect_extra_data,
            )
        else:
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if model_args.tokenization_type == "dynamic":
        data_collator = DynamicDataCollator(
            hypernet=hypernet,
            tokenizer=tokenizer,
            device=training_args.device,
            lang_index=lang_index,
            surface_form_maxlen=training_args.surface_form_maxlen,
            source_embeddings=source_embeddings,
            embeddings_cache=embeddings_cache,
            exp_type=training_args.exp_type,
            bpe_tokenizer_boundary=training_args.bpe_tokenizer_boundary,
        )
    elif data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if (
            "validation" not in raw_datasets
            and "validation_matched" not in raw_datasets
        ):
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets[
            "validation_matched" if data_args.task_name == "mnli" else "validation"
        ]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if (
        training_args.do_predict
        or data_args.task_name is not None
        or data_args.test_file is not None
    ):
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets[
            "test_matched" if data_args.task_name == "mnli" else "test"
        ]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    trainer_class = (
        CustomTrainer if model_args.tokenization_type == "dynamic" else Trainer
    )

    if (
        training_args.do_tokeniser_sampling_per_batch
        or training_args.do_tokeniser_sampling_per_sample
        or training_args.do_tokeniser_sampling_per_sample_gaussian
        or training_args.do_tokeniser_sampling_per_batch_gaussian
        or training_args.do_tokeniser_sampling_per_batch_cauchy
        or training_args.do_tokeniser_sampling_per_batch_student_t
    ):
        datasetEncoderValidation = DatasetEncoder(
            hypernet=hypernet,
            tokenizer=tokenizer,
            device=training_args.device,
            lang_index=lang_index,
            surface_form_maxlen=training_args.surface_form_maxlen,
            source_embeddings=source_embeddings,
            embeddings_cache=embeddings_cache,
            exp_type="word_tk_hypernet",
            bpe_tokenizer_boundary=training_args.bpe_tokenizer_boundary,
            merges=MERGES,
            collect_extra_data=training_args.collect_extra_data,
        )
        eval_dataset, _ = datasetEncoderValidation.encode_dataset(
            dataset=eval_dataset, batch_size=BATCH_SIZE, merges=MERGES
        )
        eval_dataset.set_format(
            type="torch", columns=["inputs_embeds", "attention_mask", "label"]
        )

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=None if model_args.tokenization_type == "dynamic" else tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.seed = training_args.seed
    trainer.args.remove_unused_columns = False
    if model_args.tokenization_type == "dynamic":
        trainer.set_dataset_encoder_and_tokeniser_sampling(datasetEncoder)
        if (
            training_args.do_tokeniser_sampling_per_batch
            or training_args.do_tokeniser_sampling_per_sample
            or training_args.do_tokeniser_sampling_per_sample_gaussian
            or training_args.do_tokeniser_sampling_per_batch_gaussian
            or training_args.do_tokeniser_sampling_per_batch_cauchy
            or training_args.do_tokeniser_sampling_per_batch_student_t
        ):
            trainer.init_tokeniser_sampling(
                device=training_args.device,
                do_tokeniser_sampling_per_sample=training_args.do_tokeniser_sampling_per_sample,
                do_tokeniser_sampling_per_batch=training_args.do_tokeniser_sampling_per_batch,
                do_tokeniser_sampling_per_batch_gaussian=training_args.do_tokeniser_sampling_per_batch_gaussian,
                do_tokeniser_sampling_per_sample_gaussian=training_args.do_tokeniser_sampling_per_sample_gaussian,
                do_tokeniser_sampling_per_batch_cauchy=training_args.do_tokeniser_sampling_per_batch_cauchy,
                do_tokeniser_sampling_per_batch_student_t=training_args.do_tokeniser_sampling_per_batch_student_t,
            )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        print("Sampled percentages", trainer.tokeniser_sampled_percentages)
        sampled_percentages_as_lists = [
            [percentage] for percentage in trainer.tokeniser_sampled_percentages
        ]

        sampled_percentages_table = wandb.Table(
            data=sampled_percentages_as_lists, columns=["Sampled Percentages"]
        )
        wandb.log({"Sampled Percentages": sampled_percentages_table})

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = raw_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(
                    len(valid_mm_dataset), data_args.max_eval_samples
                )
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics(
                "eval", combined if task is not None and "mnli" in task else metrics
            )

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(
                predict_dataset, metric_key_prefix="predict"
            ).predictions
            predictions = (
                np.squeeze(predictions)
                if is_regression
                else np.argmax(predictions, axis=1)
            )

            output_predict_file = os.path.join(
                training_args.output_dir, f"predict_results_{task}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-classification",
    }
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
