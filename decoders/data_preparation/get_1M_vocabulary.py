"""
Vocabulary Expansion Script for an Input Tokenizer's Vocabulary    

This script expands a pre-trained tokenizer vocabulary to 1M tokens by training
on additional text data. It takes a base Mistral 7B tokenizer and extends its
vocabulary using BPE (Byte Pair Encoding) training on a large text dataset.

The process requires multiple steps:
1. Loading an existing tokenizer (e.g., benjamin/zett-hypernetwork-Mistral-7B-v0.1)
2. Extracting its vocabulary and merge rules
3. Training a new BPE model on additional text data
4. Expanding the vocabulary to 1M tokens (while preserving special tokens)
5. Saving the expanded tokenizer

Inputs:
- Base tokenizer: e.g., benjamin/zett-hypernetwork-Mistral-7B-v0.1
- Training data: e.g., /mnt/nas_home/dmf45/disks/persist
- Language: e.g., en

Outputs:
- Expanded tokenizer path: e.g., decoders/data/large_tokenizer
"""

import os
import json
from typing import List, Tuple, Dict

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer
from datasets import load_dataset


def load_merges(file_path: str) -> List[Tuple[str, str]]:
    """
    Loads BPE merge rules from a text file.
    
    Args:
        file_path: Path to the merges.txt file
        
    Returns:
        List of tuples containing merge pairs
    """
    merges = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
    return merges


def load_vocab(file_path: str) -> Dict[str, int]:
    """
    Load vocabulary from a JSON file.
    
    Args:
        file_path: Path to the vocab.json file
        
    Returns:
        Dictionary mapping tokens to their IDs
    """
    with open(file_path, "r", encoding="utf-8") as file:
        vocab = json.load(file)
    return vocab


def load_training_data(data_path: str, language: str = "en") -> List[str]:
    """
    Load training text data from parquet files.
    
    Args:
        data_path: (Local) Base path to the training data
        language: (Optional) Language code for the dataset
        
    Returns:
        List of sentences (strings) for training
    """
    raw_datasets = {}
    # Assumes the dataset is stored in the following structure:
    # path/to/dataset/
    #   - {language}/
    #     - train/
    #       - {language}.parquet
    raw_datasets["train"] = load_dataset(
        "parquet",
        data_files=os.path.join(data_path, "train", f"{language}.parquet"),
        split="train",
    )
    return raw_datasets["train"]["text"]

def create_expanded_tokenizer(
    base_tokenizer_path: str,
    vocab_path: str,
    merges_path: str,
    training_texts: List[str],
    target_vocab_size: int = 1_000_000
) -> Tokenizer:
    """
    Create an expanded tokenizer by training on additional text data.
    
    Args:
        base_tokenizer_path: Path/name of the base tokenizer
        vocab_path: Path to the vocabulary file
        merges_path: Path to the merges file
        training_texts: List of text strings for training
        target_vocab_size: Target vocabulary size (default: 1M)
        
    Returns:
        Trained tokenizer with expanded vocabulary
    """
    # Load the base hypernetwork tokenizer
    print("Loading base tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)
    
    # Load existing vocabulary and merges
    print("Loading existing vocabulary and merges...")
    vocab, merges = BPE.read_file(vocab_path, merges_path)
    
    # Create new BPE model with existing vocabulary
    new_bpe_model = BPE(vocab=vocab, merges=merges)
    new_tokenizer = Tokenizer(new_bpe_model)
    
    # Configure trainer for vocabulary expansion
    print(f"Configuring trainer for {target_vocab_size} vocabulary size...")
    trainer = BpeTrainer(
        vocab_size=target_vocab_size,
        show_progress=True,
        special_tokens=list(base_tokenizer.special_tokens_map.values())
        + list(base_tokenizer.vocab.keys()),
    )
    
    # Copy pre-tokenizer and normalizer from base tokenizer
    new_tokenizer.pre_tokenizer = base_tokenizer.backend_tokenizer.pre_tokenizer
    new_tokenizer.normalizer = base_tokenizer.backend_tokenizer.normalizer
    
    # Expand the vocabulary to [target_vocab_size] by training on additional texts
    print("Training tokenizer on additional texts...")
    new_tokenizer.train_from_iterator(
        training_texts, trainer=trainer, length=len(training_texts)
    )
    
    return new_tokenizer


def main():
    """Main function to execute the vocabulary expansion process."""
    
    # Configuration
    BASE_TOKENIZER_PATH = "benjamin/zett-hypernetwork-Mistral-7B-v0.1"
    OLD_VOCAB_PATH = "decoders/data/tokenizer_hn_mistral/vocab.json"
    OLD_MERGES_PATH = "decoders/data/tokenizer_hn_mistral/merges.txt"
    OUTPUT_PATH = "decoders/data/large_tokenizer"
    TRAINING_DATA_PATH = "/mnt/nas_home/dmf45/disks/persist"
    LANGUAGE = "en"
    TARGET_VOCAB_SIZE = 1_000_000
    
    print("Starting vocabulary expansion process...")
    print(f"Target vocabulary size: {TARGET_VOCAB_SIZE:,}")
    print("Loading training data...")
    training_texts = load_training_data(TRAINING_DATA_PATH, LANGUAGE)
    print(f"Loaded {len(training_texts):,} training texts")
    
    new_tokenizer = create_expanded_tokenizer(
        base_tokenizer_path=BASE_TOKENIZER_PATH,
        vocab_path=OLD_VOCAB_PATH,
        merges_path=OLD_MERGES_PATH,
        training_texts=training_texts,
        target_vocab_size=TARGET_VOCAB_SIZE
    )
    
    # Save the expanded tokenizer
    print(f"Saving expanded tokenizer to {OUTPUT_PATH}...")
    new_tokenizer.model.save(OUTPUT_PATH)
    
    # Report final vocabulary size
    new_vocab_size = new_tokenizer.get_vocab_size()
    print(f"Final vocabulary size: {new_vocab_size:,}")

if __name__ == "__main__":
    main()
