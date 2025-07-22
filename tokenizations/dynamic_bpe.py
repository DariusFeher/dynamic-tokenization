"""
Dynamic BPE Tokenizer

Implements a dynamic tokenization for a batch using byte pair encoding (BPE)

Public methods:
    - tokenize_batch: Tokenize a batch with a given number of BPE merges to be applied (i.e., dynamic tokenization)
    - get_merges2seqlen_for_dataset: Computes and saved # number of merges -> sequence reduction mapping for a given dataset (0% to 100% reduction)

"""
import pickle
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, Tuple, List, Set, Any

from datasets.formatting.formatting import LazyBatch
from tokenizations.tokenizers_utils import pretokenize, tokenize
from tokenizers import pre_tokenizers
from zett.utils import CHARS_TO_BYTES

class Dynamic_BPE:
    """
    Dynamic Byte Pair Encoding (BPE) tokenizer for dynamic tokenization experiments.

    Args:
        tokenizer: The base tokenizer object.
        tokenizer_boundary (str): Boundary for merging subwords ('pretokens', 'words', 'sentence', etc.).
    """
    def __init__(self, tokenizer: Any, tokenizer_boundary: str = "pretokens"):
        self.tokenizer = tokenizer
        self.tokenizer_boundary = tokenizer_boundary
        self.special_token_map = set(tokenizer.special_tokens_map.values())
        self.punctuation_tokens = {
            token
            for token in tokenizer.vocab
            if any(c in token for c in """.,!?;:()-"'\/`$%&*+<=>@[]^_{|}~""")
            and not any(char.isdigit() for char in token)
        }
        self.debug = False
        self.merges2seqLen = {}  # Used for sequence length analysis

    def tokenize_batch(
        self,
        batch_examples: LazyBatch,
        max_nr_merges: int = 1000,
        mlm: bool = False,
        max_length: int = 1280,
        ner: bool = False,
        nli: bool = False,
        mmlu: bool = False,
    ):
        """
        Tokenize a batch of examples using dynamic BPE with a specified number of merges.
        Returns tokenized sequences, unique tokens, sequence lengths, and (optionally) word ids.
        """
        unique_tokens_original, batch_tokens, batch_word_tokens, batch_word_ids = (
            self.tokenize_base_case(
                batch_examples=batch_examples,
                mlm=mlm,
                max_length=max_length,
                ner=ner,
                nli=nli,
                mmlu=mmlu,
            )
        )
        unique_tokens_bpe = set()
        batch_seq_lengths = []
        total_merges = 0
        while total_merges < max_nr_merges:
            best_pair = self.get_most_frequent_pair(batch_tokens=batch_tokens)
            if best_pair == "":
                print(f"Early exit, {total_merges} out of {max_nr_merges}")
                break
            total_merges += 1
            batch_tokens, batch_word_ids = self.merge_pair(
                a=best_pair[0],
                b=best_pair[1],
                batch_tokens=batch_tokens,
                ner=ner,
                batch_word_ids=batch_word_ids,
            )
        for tokenised_text in batch_tokens:
            unique_tokens_bpe.update(tokenised_text)
            batch_seq_lengths.append(len(tokenised_text))
        if self.debug:
            for i in range(32):
                if i < len(batch_tokens) and batch_tokens[i] != batch_word_tokens[i]:
                    print(i)
                    print(batch_tokens[i])
                    print(batch_word_tokens[i])
        return batch_tokens, unique_tokens_bpe, batch_seq_lengths, batch_word_ids

    def tokenize_batch_for_seq_len(
        self,
        batch_examples: LazyBatch,
        max_nr_merges: int = 20000,
        mlm: bool = False,
        max_length: int = 128,
        ner: bool = False,
        nli: bool = False,
        mmlu: bool = False,
    ):
        """
        Analyze the distribution of merges to average sequence lengths for a dataset batch.
        Populates self.merges2seqLen.
        """
        import copy
        _, batch_tokens, _, _ = self.tokenize_base_case(
            batch_examples=batch_examples,
            mlm=False,
            max_length=max_length,
            ner=False,
            nli=False,
            mmlu=True,
        )
        total_merges = 0
        if total_merges not in self.merges2seqLen:
            self.merges2seqLen[total_merges] = 0
        for tokenised_text in batch_tokens:
            self.merges2seqLen[total_merges] += len(tokenised_text)
        while total_merges < max_nr_merges:
            best_pair = self.get_most_frequent_pair(batch_tokens=batch_tokens)
            if best_pair == "":
                for i in range(total_merges + 1, max_nr_merges):
                    if i not in self.merges2seqLen:
                        self.merges2seqLen[i] = 0
                    for tokenised_text in batch_tokens:
                        self.merges2seqLen[i] += len(tokenised_text)
                break
            total_merges += 1
            batch_tokens, _ = self.merge_pair(
                a=best_pair[0], b=best_pair[1], batch_tokens=batch_tokens
            )
            if total_merges not in self.merges2seqLen:
                self.merges2seqLen[total_merges] = 0
            for tokenised_text in batch_tokens:
                self.merges2seqLen[total_merges] += len(tokenised_text)

    def get_merges2seqlen_for_dataset(self, dataset, batch_size: int = 32) -> None:
        """
        Compute and save the mapping from number of merges to average sequence length for a dataset.
        """
        from tqdm import tqdm
        self.merges2seqLen = {}
        max_length = 8192  # Can be parameterized
        for i in tqdm(range(0, len(dataset), batch_size), desc="Encoding Dataset"):
            batch = dataset[i : i + batch_size]
            self.tokenize_batch_for_seq_len(
                batch_examples=batch,
                max_nr_merges=100000,
                mlm=False,
                max_length=max_length,
                ner=False,
                nli=False,
                mmlu=True,
            )
        for merge in self.merges2seqLen:
            self.merges2seqLen[merge] = self.merges2seqLen[merge] / len(dataset)
        print(self.merges2seqLen)
        with open("MTBench100k_merges2SeqLen_v2_10k_128Batch_MADLAD.pkl", "wb") as f:
            pickle.dump(self.merges2seqLen, f)

    # === Helper methods ===

    @lru_cache(maxsize=None)
    def is_valid_pair(self, pair: Tuple[str, str]) -> bool:
        """
        Check if a pair of tokens can be merged, according to the current boundary and special token rules.
        """
        token1, token2 = pair[0], pair[1]
        # merging can be problematic if the pair is not a valid utf-8 string
        # in particular, if a full word is followed by something which is not valid utf-8
        # we would end up merging the two, even though we do not want to merge across words
        # in general, we do, however, want to allow merges across tokens which do not form a valid utf-8 string
        # so first apply a simple heuristic: if the resulting token would have a spacelike token ('\n', '\t', ' ')
        # somewhere in the middle, we do not merge them, except if all of them are spacelike (to allow compressing consecutive whitespace)
        # since in a peculiar edge case the Mistral pretokenizer also splits the whitespace in "x\n\ny" into two tokens (but not in "\n\n")
        # we also disallow merging "\n\n" (i.e. ĊĊ) for consistency with the pretokenizer
        spacelike_char_representations = "ĉĠĊ"
        if any(c in (token1 + token2)[1:] for c in spacelike_char_representations) and (
            token1 + token2 == "ĊĊ"
            or not all(c in spacelike_char_representations for c in (token1 + token2))
        ):
            return False

        try:
            if self.tokenizer_boundary == "sentence":
                return (
                    token1 not in self.special_token_map
                    and token2 not in self.special_token_map
                )

            b = [CHARS_TO_BYTES[c] for c in token1 + token2]
            string = bytes(b).decode("utf-8")

            if self.tokenizer_boundary == "pretokens":
                return (
                    len(
                        self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(string)
                    )
                    == 1
                )
            elif self.tokenizer_boundary == "word":
                return (
                    len(pre_tokenizers.WhitespaceSplit().pre_tokenize_str(string)) == 1
                    and token1 not in self.special_token_map
                    and token2 not in self.special_token_map
                )
            elif self.tokenizer_boundary == "word_hyphen":
                cond1 = (
                    len(pre_tokenizers.WhitespaceSplit().pre_tokenize_str(string)) == 1
                    and token1 not in self.special_token_map
                    and token2 not in self.special_token_map
                )
                if (
                    cond1
                    and token1 not in self.punctuation_tokens
                    and token2 in self.punctuation_tokens
                ):
                    return token2 == "-"
                return cond1
        except UnicodeDecodeError:
            # this chunk of bytes is not a valid string, so we can't test it
            return True

    def get_most_frequent_pair(self, batch_tokens: List[List[str]], check_valid: bool = True) -> Any:
        """
        Find the most frequent valid pair of tokens in the batch.
        """
        pair_freqs = Counter()
        for token_sequence in batch_tokens:
            pairs = zip(token_sequence, token_sequence[1:])
            if check_valid:
                pairs = (pair for pair in pairs if self.is_valid_pair(pair))
            pair_freqs.update(pairs)

        if pair_freqs:
            best_pair = max(pair_freqs, key=pair_freqs.get)
            return best_pair
        return ""

    def merge_pair(
        self, a: str, b: str, batch_tokens: List[List[str]], ner: bool = False, batch_word_ids: List[List[Any]] = []
    ):
        """
        Merge all occurrences of the pair (a, b) in the batch tokens.
        """
        for idx, token_seq in enumerate(batch_tokens):
            i = 0
            new_token_seq = []
            new_word_ids = []
            token_seq = batch_tokens[idx]
            while i < len(token_seq):
                if (
                    i < len(token_seq) - 1
                    and token_seq[i] == a
                    and token_seq[i + 1] == b
                ):
                    new_token_seq.append(a + b)
                    if ner:
                        new_word_ids.append(batch_word_ids[idx][i])
                    i += 2
                else:
                    new_token_seq.append(token_seq[i])
                    if ner:
                        new_word_ids.append(batch_word_ids[idx][i])
                    i += 1
            batch_tokens[idx] = new_token_seq
            if ner:
                batch_word_ids[idx] = new_word_ids
        return batch_tokens, batch_word_ids

    def tokenize_base_case(
        self,
        batch_examples: Any,
        mlm: bool = False,
        max_length: int = 128,
        ner: bool = False,
        nli: bool = False,
        mmlu: bool = False,
    ):
        """
        Tokenize a batch using the base tokenizer, before any merges.
        Returns unique tokens, batch tokens, batch word tokens, and batch word ids.
        """
        assert not (mlm and ner)
        batch_tokens = []
        batch_word_tokens = []
        unique_tokens_original = set()
        batch_word_ids = []
        if mmlu:
            if isinstance(batch_examples, list):
                for batch_example in batch_examples:
                    tokens = ["<s>"] + tokenize(
                        batch_example,
                        self.tokenizer,
                        max_length=max_length,
                        truncation=True,
                    )
                    if len(tokens) > max_length:
                        tokens = tokens[:max_length]
                    unique_tokens_original.update(tokens)
                    batch_tokens.append(tokens)
            else:
                for idx, _ in enumerate(batch_examples["prompt"]):
                    tokens = ["<s>"] + tokenize(
                        batch_examples["prompt"][idx],
                        self.tokenizer,
                        max_length=max_length,
                        truncation=True,
                    )
                    if len(tokens) > max_length:
                        tokens = tokens[:max_length]
                    unique_tokens_original.update(tokens)
                    batch_tokens.append(tokens)
        elif ner:
            if isinstance(batch_examples, list):
                for batch_example in batch_examples:
                    tokens = ["<s>"]
                    word_ids = [None]
                    for word_index, word in enumerate(batch_example["tokens"]):
                        subtokens = self.tokenizer.tokenize(word, max_length=max_length)
                        tokens.extend(subtokens)
                        word_ids.extend([word_index] * len(subtokens))
                    if len(tokens) >= max_length:
                        tokens = tokens[: max_length - 1]
                        word_ids = word_ids[: max_length - 1]
                    tokens.append("</s>")
                    word_ids.append(None)

                    batch_tokens.append(tokens)
                    unique_tokens_original.update(tokens)
                    batch_word_ids.append(word_ids)
            else:
                for idx, _ in enumerate(batch_examples["tokens"]):
                    tokens = ["<s>"]
                    word_ids = [None]
                    for word_index, word in enumerate(batch_examples["tokens"][idx]):
                        subtokens = self.tokenizer.tokenize(word, max_length=max_length)
                        tokens.extend(subtokens)
                        word_ids.extend([word_index] * len(subtokens))
                    if len(tokens) >= max_length:
                        tokens = tokens[: max_length - 1]
                        word_ids = word_ids[: max_length - 1]
                    tokens.append("</s>")
                    word_ids.append(None)

                    unique_tokens_original.update(tokens)
                    batch_tokens.append(tokens)
                    batch_word_ids.append(word_ids)

        elif nli:
            if isinstance(batch_examples, list):
                for batch_example in batch_examples:
                    tokens = (
                        ["<s>"]
                        + tokenize(batch_example["premise"], self.tokenizer)
                        + ["</s>", "</s>"]
                        + tokenize(batch_example["hypothesis"], self.tokenizer)
                        + ["</s>"]
                    )
                    batch_tokens.append(tokens)
                    unique_tokens_original.update(tokens)
            else:
                for idx, _ in enumerate(batch_examples["premise"]):
                    tokens = (
                        ["<s>"]
                        + tokenize(batch_examples["premise"][idx], self.tokenizer)
                        + ["</s>", "</s>"]
                        + tokenize(batch_examples["hypothesis"][idx], self.tokenizer)
                        + ["</s>"]
                    )
                    batch_tokens.append(tokens)
                    unique_tokens_original.update(tokens)
        elif mlm:
            if isinstance(batch_examples, list):
                for batch_example in batch_examples:
                    tokens = (
                        ["<s>"]
                        + tokenize(
                            batch_example["text"],
                            self.tokenizer,
                            max_length=max_length - 2,
                        )
                        + ["</s>"]
                    )
                    tokens = tokens[:max_length]
                    batch_tokens.append(tokens)
                    unique_tokens_original.update(tokens)
            else:
                for idx, _ in enumerate(batch_examples["text"]):
                    tokens = (
                        ["<s>"]
                        + tokenize(
                            batch_examples["text"][idx],
                            max_length=max_length - 2,
                            truncation=True,
                            tokenizer=self.tokenizer,
                        )
                        + ["</s>"]
                    )
        return unique_tokens_original, batch_tokens, batch_word_tokens, batch_word_ids

    def initialize_position_tracking(self, batch_tokens: List[List[str]]):
        """
        Build a mapping from token pairs to their positions in the batch (for ner).
        """
        token_positions = defaultdict(list)
        for idx, tokens in enumerate(batch_tokens):
            for pos in range(len(tokens) - 1):
                pair = (tokens[pos], tokens[pos + 1])
                token_positions[pair].append((idx, pos))
        return token_positions

    def merge_pair_with_tracking(
        self,
        best_pair: Tuple[str, str],
        batch_tokens: List[List[str]],
        token_positions: Dict[Tuple[str, str], List[Tuple[int, int]]],
        ner: bool = False,
        batch_word_ids: List[List[Any]] = [],
    ):
        """
        Merge a pair in the batch tokens, updating position tracking.
        """
        indices_to_merge = token_positions.pop(best_pair, [])
        new_pair = best_pair[0] + best_pair[1]

        for idx, pos in sorted(indices_to_merge, reverse=True):
            # Merge tokens in the batch_tokens
            batch_tokens[idx][pos] = new_pair
            # Remove the second part of the merged pair
            del batch_tokens[idx][pos + 1]

            # Update positions in the token_positions dictionary
            if pos > 0:  # Update the previous pair
                prev_pair = (batch_tokens[idx][pos - 1], best_pair[0])
                token_positions[prev_pair].remove((idx, pos - 1))
                new_prev_pair = (batch_tokens[idx][pos - 1], new_pair)
                token_positions[new_prev_pair].append((idx, pos - 1))

            if pos < len(batch_tokens[idx]) - 1:  # Update the next pair
                next_pair = (best_pair[1], batch_tokens[idx][pos + 1])
                token_positions[next_pair].remove((idx, pos))
                new_next_pair = (new_pair, batch_tokens[idx][pos])
                token_positions[new_next_pair].append((idx, pos))

            # Adjust batch_word_ids if ner is True
            if ner:
                batch_word_ids[idx][pos] = batch_word_ids[idx][pos]
                del batch_word_ids[idx][pos + 1]

        return batch_tokens, batch_word_ids, token_positions
