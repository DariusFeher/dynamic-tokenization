import random
import time
from typing import Dict, Tuple

import numpy as np
import torch
from datasets import Dataset
from datasets.formatting.formatting import LazyBatch
from tokenizations.dynamic_bpe import Dynamic_BPE
from tokenizations.hypernet_cache import LRU_Cache
from tokenizations.tokenizers_utils import pretokenize, tokenize
from tqdm import tqdm
from zett.utils import get_surface_form_matrix


class DatasetEncoder:
    def __init__(
        self,
        tokenizer,
        hypernet=None,
        device: str = "cpu",
        lang_index: int = 0,
        surface_form_maxlen: int = 4,
        source_embeddings: torch.tensor = torch.tensor([]),
        embeddings_cache: LRU_Cache = None,
        exp_type: str = "",
        bpe_tokenizer_boundary: str = "",
        merges: int = None,
        collect_extra_data: bool = False,
    ) -> None:
        self.tokens_processed_per_batch = []
        self.unique_tokens_per_batch = []
        self.seq_lengths = []
        self.tokenizer = tokenizer
        self.hypernet = hypernet
        self.device = device
        self.lang_index = lang_index
        self.surface_form_maxlen = surface_form_maxlen
        self.source_embeddings = source_embeddings
        self.embeddings_cache = embeddings_cache
        self.exp_type = exp_type
        self.bpe_tokenizer_boundary = bpe_tokenizer_boundary
        self.collect_extra_data = collect_extra_data
        self.merges = merges
        self.dynamic_bpe = Dynamic_BPE(
            tokenizer=self.tokenizer, tokenizer_boundary=self.bpe_tokenizer_boundary
        )

    def reset_state(self, embeddings_cache: torch.tensor):
        self.seq_lengths = []
        self.tokens_processed_per_batch = []
        self.unique_tokens_per_batch = []
        self.embeddings_cache = embeddings_cache

    def compute_tokens_batch_embeddings(self, unique_tokens: set, task: str) -> None:
        tokens_to_process = [
            token
            for token in unique_tokens
            if token not in self.embeddings_cache.token2idx
        ]
        tokens_to_move_to_end = unique_tokens - set(tokens_to_process)

        nr_tokens_to_process = len(tokens_to_process)
        new_cache_size = self.embeddings_cache.size + nr_tokens_to_process

        if new_cache_size > self.embeddings_cache.capacity:
            self.embeddings_cache.evict_with_exceptions(
                unique_tokens, new_cache_size - self.embeddings_cache.capacity
            )

        if tokens_to_process:
            if self.collect_extra_data:
                self.tokens_processed_per_batch.append(nr_tokens_to_process)
                self.unique_tokens_per_batch.append(len(unique_tokens))

            surface_forms, _ = get_surface_form_matrix(
                tokens_to_process,
                self.surface_form_maxlen,
                tokenizer_to_use=self.tokenizer,
                verbose=False,
            )
            surface_forms = torch.from_numpy(surface_forms).to(self.device)
            special_tokens_mask = torch.isin(
                surface_forms[:, 0],
                torch.tensor(self.tokenizer.all_special_ids, device=self.device),
            )

            assert str(self.device) in str(
                surface_forms.device
            ), f"Device does not match: {surface_forms.device} different than {self.device}"
            assert str(self.device) in str(
                self.lang_index.device
            ), f"Device does not match: {self.lang_index.device} different than {self.device}"
            assert str(self.device) in str(
                self.source_embeddings.device
            ), f"Device does not match: {self.source_embeddings.device} different than {self.device}"

            hypernet_preds, _, bias = self.hypernet(
                surface_forms,
                lang_index=self.lang_index,
                source_embeddings=self.source_embeddings,
            )
            if task != "mmlu":
                hypernet_preds[special_tokens_mask] = self.source_embeddings[
                    surface_forms[special_tokens_mask, 0]
                ]
            else:
                hypernet_preds[special_tokens_mask] = self.source_embeddings[
                    surface_forms[special_tokens_mask, 0], : hypernet_preds.shape[1]
                ]  # Should be 4096 for Mistral. If this is not set, for special tokens we have embeddings of shape (1, 8192) and (1, 4096) for the rest of the tokens. This is because the source embeddings are concatenated
            self.embeddings_cache.put(tokens_to_process, hypernet_preds)
        self.embeddings_cache.move_tokens_to_end(tokens_to_move_to_end)

    def encode_examples_unique_tokens_lru(
        self,
        examples: LazyBatch,
        max_length: int = 128,
        merges: int = 0,
        tokenise_idx: int = -1,
        task: str = "nli",
        label_all_tokens: bool = False,
        label_to_id: list = [],
        b_to_i_label: list = [],
    ) -> Dict[str, torch.Tensor]:
        """
        Function for encoding a batch for either NLI or NER task.
        """
        batch_tokens = []
        unique_tokens = set()
        sequence_batch_embeddings = []
        attention_masks = []
        batch_word_ids = []
        merges = self.merges if self.merges != None else merges
        max_batch_length = 0

        with torch.no_grad():
            if self.exp_type == "dynamic_bpe":
                ner = task == "ner"
                batch_tokens, unique_tokens, batch_seq_lengths, batch_word_ids = (
                    self.dynamic_bpe.tokenize_batch(
                        batch_examples=examples,
                        max_nr_merges=merges,
                        mlm=False,
                        ner=ner,
                    )
                )
                if self.collect_extra_data:
                    self.seq_lengths.extend(batch_seq_lengths)
            else:
                if task == "nli":
                    if isinstance(examples, list):
                        for batch_example in examples:
                            if self.exp_type == "original_tk_hypernet":
                                tokens = (
                                    ["<s>"]
                                    + tokenize(batch_example["premise"], self.tokenizer)
                                    + ["</s>", "</s>"]
                                    + tokenize(
                                        batch_example["hypothesis"], self.tokenizer
                                    )
                                    + ["</s>"]
                                )
                            elif (
                                self.exp_type == "word_tk_hypernet"
                                or self.exp_type == "fvt"
                                or self.exp_type == "dynamic_bpe"
                            ):
                                tokens = (
                                    ["<s>"]
                                    + pretokenize(
                                        batch_example["premise"], self.tokenizer
                                    )
                                    + ["</s>", "</s>"]
                                    + pretokenize(
                                        batch_example["hypothesis"], self.tokenizer
                                    )
                                    + ["</s>"]
                                )
                            if self.collect_extra_data:
                                self.seq_lengths.append(len(tokens))
                            unique_tokens.update(tokens)
                            batch_tokens.append(tokens)
                    else:
                        for idx, _ in enumerate(examples["premise"]):
                            if self.exp_type == "original_tk_hypernet":
                                tokens = (
                                    ["<s>"]
                                    + tokenize(examples["premise"][idx], self.tokenizer)
                                    + ["</s>", "</s>"]
                                    + tokenize(
                                        examples["hypothesis"][idx], self.tokenizer
                                    )
                                    + ["</s>"]
                                )

                            elif (
                                self.exp_type == "word_tk_hypernet"
                                or self.exp_type == "fvt"
                                or self.exp_type == "dynamic_bpe"
                            ):
                                tokens = (
                                    ["<s>"]
                                    + pretokenize(
                                        examples["premise"][idx], self.tokenizer
                                    )
                                    + ["</s>", "</s>"]
                                    + pretokenize(
                                        examples["hypothesis"][idx], self.tokenizer
                                    )
                                    + ["</s>"]
                                )

                            if self.collect_extra_data:
                                self.seq_lengths.append(len(tokens))
                            unique_tokens.update(tokens)
                            batch_tokens.append(tokens)
                elif task == "ner":
                    if isinstance(examples, list):
                        for batch_example in examples:
                            tokens = ["<s>"]
                            word_ids = [None]
                            if self.exp_type == "original_tk_hypernet":
                                for word_index, word in enumerate(
                                    batch_example["tokens"]
                                ):
                                    subtokens = self.tokenizer.tokenize(
                                        word, max_length=max_length
                                    )
                                    tokens.extend(subtokens)
                                    word_ids.extend([word_index] * len(subtokens))
                            elif (
                                self.exp_type == "word_tk_hypernet"
                                or self.exp_type == "fvt"
                            ):
                                for word_index, word in enumerate(
                                    batch_example["tokens"]
                                ):
                                    pretokens = pretokenize(
                                        word, tokenizer=self.tokenizer
                                    )
                                    tokens.extend(pretokens)
                                    word_ids.extend([word_index] * len(pretokens))

                            if len(tokens) >= max_length:
                                tokens = tokens[: max_length - 1]
                                word_ids = word_ids[: max_length - 1]
                            tokens.append("</s>")
                            word_ids.append(None)

                            if self.collect_extra_data:
                                self.seq_lengths.append(len(tokens))
                            unique_tokens.update(tokens)
                            batch_tokens.append(tokens)
                            batch_word_ids.append(word_ids)
                    else:
                        for idx, _ in enumerate(examples["tokens"]):
                            tokens = ["<s>"]
                            word_ids = [None]
                            if self.exp_type == "original_tk_hypernet":
                                for word_index, word in enumerate(
                                    examples["tokens"][idx]
                                ):
                                    subtokens = self.tokenizer.tokenize(
                                        word, max_length=max_length
                                    )
                                    tokens.extend(subtokens)
                                    word_ids.extend([word_index] * len(subtokens))
                            elif (
                                self.exp_type == "word_tk_hypernet"
                                or self.exp_type == "fvt"
                            ):
                                for word_index, word in enumerate(
                                    examples["tokens"][idx]
                                ):
                                    pretokens = pretokenize(
                                        word, tokenizer=self.tokenizer
                                    )
                                    tokens.extend(pretokens)
                                    word_ids.extend([word_index] * len(pretokens))
                            if len(tokens) >= max_length:
                                tokens = tokens[: max_length - 1]
                                word_ids = word_ids[: max_length - 1]
                            tokens.append("</s>")
                            word_ids.append(None)

                            if self.collect_extra_data:
                                self.seq_lengths.append(len(tokens))
                            unique_tokens.update(tokens)
                            batch_tokens.append(tokens)
                            batch_word_ids.append(word_ids)
                else:
                    raise Exception("You must specify the type of task")

            unique_tokens.add(self.tokenizer.pad_token)
            if self.exp_type != "fvt":
                self.compute_tokens_batch_embeddings(unique_tokens, task=task)

                hypernet_preds = self.embeddings_cache.hypernet_preds
                token2idx = self.embeddings_cache.token2idx

            if tokenise_idx != -1:
                batch_tokens = [batch_tokens[tokenise_idx]]  # CHECK THIS

            # Pad and retrieve sequence tokens embeddings
            for idx, tokens_sequence in enumerate(batch_tokens):
                seq_len = len(tokens_sequence)

                if seq_len > max_length:
                    tokens_sequence = tokens_sequence[:max_length]

                if task != "mmlu":
                    attention_ones = len(tokens_sequence)
                    tokens_sequence += [self.tokenizer.pad_token] * (
                        max_length - attention_ones
                    )
                    attention_zeros = max_length - attention_ones
                    attention_mask = torch.tensor(
                        [1] * attention_ones + [0] * attention_zeros, device=self.device
                    )
                    attention_masks.append(attention_mask)
                else:
                    attention_ones = len(tokens_sequence)
                    pad_tokens_nr = max_batch_length - attention_ones
                    tokens_sequence = [self.tokenizer.pad_token] * (
                        pad_tokens_nr
                    ) + tokens_sequence
                    attention_zeros = pad_tokens_nr
                    attention_mask = torch.tensor(
                        [0] * attention_zeros + [1] * attention_ones, device=self.device
                    )
                    attention_masks.append(attention_mask)

                if self.exp_type != "fvt":
                    sequence_indices = [token2idx[token] for token in tokens_sequence]
                    embeddings = hypernet_preds[sequence_indices]
                else:
                    embeddings = torch.zeros(max_length, 768, device=self.device)
                    for i, token in enumerate(tokens_sequence):
                        decomposed = self.tokenizer._tokenizer.model.tokenize(token)
                        if not any(
                            x.id >= len(self.source_embeddings) for x in decomposed
                        ):
                            constituent_idx = np.array([x.id for x in decomposed])
                            if len(constituent_idx) > 0:
                                embeddings[i] = self.source_embeddings[
                                    constituent_idx
                                ].mean(0)
                            else:
                                embeddings[i] = self.source_embeddings[
                                    self.tokenizer.unk_token_id
                                ]
                        else:
                            embeddings[i] = self.source_embeddings[
                                self.tokenizer.unk_token_id
                            ]

                sequence_batch_embeddings.append(embeddings)
        if task == "nli" or task == "mmlu":
            if tokenise_idx != -1:
                return {
                    "inputs_embeds": sequence_batch_embeddings[0],
                    "attention_mask": attention_masks[0],
                }
            return {
                "inputs_embeds": torch.stack(sequence_batch_embeddings).to(self.device),
                "attention_mask": torch.stack(attention_masks).to(self.device),
            }

        elif task == "ner":
            labels = []
            for i, label in enumerate(examples["ner_tags"]):
                word_ids = batch_word_ids[i]
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(label_to_id[label[word_idx]])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        if label_all_tokens:
                            label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                        else:
                            label_ids.append(-100)
                    previous_word_idx = word_idx

                if len(label_ids) < max_length:
                    len_labeled_tokens = len(label_ids)
                    label_ids = torch.tensor(
                        label_ids + [-100] * (max_length - len_labeled_tokens),
                        device=self.device,
                    )
                else:
                    label_ids = torch.tensor(label_ids, device=self.device)

                labels.append(label_ids)
            return {
                "inputs_embeds": torch.stack(sequence_batch_embeddings).to(self.device),
                "attention_mask": torch.stack(attention_masks).to(self.device),
                "labels": torch.stack(labels).to(self.device),
            }

    def encode_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        max_length: int = 128,
        merges: int = 0,
        task: str = "nli",
        label_all_tokens: bool = False,
        label_to_id: list = [],
        b_to_i_label: list = [],
    ) -> Tuple[Dataset, float]:
        batched_results = []
        start_time = time.time()

        for i in tqdm(range(0, len(dataset), batch_size), desc="Encoding Dataset"):
            batch = dataset[i : i + batch_size]
            encoded_batch = self.encode_examples_unique_tokens_lru(
                batch,
                max_length=max_length,
                merges=merges,
                task=task,
                label_all_tokens=label_all_tokens,
                label_to_id=label_to_id,
                b_to_i_label=b_to_i_label,
            )
            if task == "nli":
                encoded_batch["label"] = torch.tensor(
                    batch["label"], device=self.device
                )
            batched_results.append(encoded_batch)
        end_time = time.time()
        encoding_time = end_time - start_time
        if task == "nli":
            combined_batches = {
                "inputs_embeds": torch.cat(
                    [batch["inputs_embeds"] for batch in batched_results]
                ),
                "attention_mask": torch.cat(
                    [batch["attention_mask"] for batch in batched_results]
                ),
                "label": torch.cat([batch["label"] for batch in batched_results]),
            }
        elif task == "ner":
            combined_batches = {
                "inputs_embeds": torch.cat(
                    [batch["inputs_embeds"] for batch in batched_results]
                ),
                "attention_mask": torch.cat(
                    [batch["attention_mask"] for batch in batched_results]
                ),
                "labels": torch.cat([batch["labels"] for batch in batched_results]),
            }
        return Dataset.from_dict(combined_batches), encoding_time

    def encode_examples_unique_tokens(self, examples, max_length=128, verbose=False):
        """
        Uses hypernet to predict embedding only for unique tokens rather than all the tokens in our batch sequences.
        """
        # return self.tokenizer(examples['premise'], examples['hypothesis'], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        batch_tokens = []
        unique_tokens = set()

        for idx, _ in enumerate(examples["premise"]):
            if self.exp_type == "original_tk_hypernet":
                tokens = (
                    ["<s>"]
                    + tokenize(examples["premise"][idx], tokenizer=self.tokenizer)
                    + ["</s>", "<s>"]
                    + tokenize(examples["hypothesis"][idx], tokenizer=self.tokenizer)
                    + ["</s>"]
                )
            elif self.exp_type == "word_tk_hypernet":
                tokens = (
                    ["<s>"]
                    + pretokenize(examples["premise"][idx], tokenizer=self.tokenizer)
                    + ["</s>", "<s>"]
                    + pretokenize(examples["hypothesis"][idx], tokenizer=self.tokenizer)
                    + ["</s>"]
                )

            # Chose this over np.unique as I think it's more efficient (O(n + m) compared to O(n log n))
            unique_tokens = unique_tokens.union(set(tokens))
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            batch_tokens.append(tokens)

        if verbose:
            for batch in batch_tokens:
                print(batch)
                print(len(batch))
            print(unique_tokens)
            total_length = sum(len(seq_tokens) for seq_tokens in batch_tokens)
            print(total_length)

        attention_masks = []  # attention masks for each sequence in batch

        for idx, seq_tokens in enumerate(batch_tokens):
            attention_ones = len(seq_tokens)
            seq_tokens += [self.tokenizer.pad_token] * (max_length - len(seq_tokens))
            attention_zeros = max_length - attention_ones
            batch_tokens[idx] = seq_tokens  # Updated padded sequence
            attention_mask = torch.tensor([1] * attention_ones + [0] * attention_zeros)
            attention_masks.append(torch.tensor(attention_mask))

        unique_tokens.add(self.tokenizer.pad_token)
        attention_masks_batch = attention_masks

        unique_tokens = list(unique_tokens)
        token2idx = {token: index for index, token in enumerate(unique_tokens)}

        surface_forms, _ = get_surface_form_matrix(
            unique_tokens,
            self.surface_form_maxlen,
            tokenizer_to_use=self.tokenizer,
            verbose=False,
        )
        surface_forms = torch.from_numpy(surface_forms).to(self.device)
        special_tokens_mask = torch.isin(
            surface_forms[:, 0],
            torch.tensor(self.tokenizer.all_special_ids).to(self.device),
        )

        assert str(self.device) in str(
            surface_forms.device
        ), f"Device does not match: {surface_forms.device} different than {self.device}"
        assert str(self.device) in str(
            self.lang_index.device
        ), f"Device does not match: {self.lang_index.device} different than {self.device}"
        assert str(self.device) in str(
            self.source_embeddings.device
        ), f"Device does not match: {self.source_embeddings.device} different than {self.device}"

        hypernet_preds, _, bias = self.hypernet(
            surface_forms,
            lang_index=torch.tensor(
                self.lang_index, dtype=torch.int32, device=self.device
            ),
            source_embeddings=self.source_embeddings,
        )
        hypernet_preds[special_tokens_mask] = self.source_embeddings[
            surface_forms[special_tokens_mask, 0]
        ]
        embeddings_batch = []

        for idx, tokens_sequence in enumerate(batch_tokens):
            sequence_indices = np.array([token2idx[token] for token in tokens_sequence])
            embeddings = hypernet_preds[sequence_indices]
            embeddings_batch.append(embeddings)

        return {
            "inputs_embeds": torch.stack(embeddings_batch),
            "attention_mask": torch.stack(attention_masks_batch),
        }