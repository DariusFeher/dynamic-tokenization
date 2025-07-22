"""
Hypernetwork Embeddings Generation for the Expanded 1M Vocabulary

This script generates hypernetwork embeddings for tokens in an expanded vocabulary (1M tokens).
It takes a base Mistral 7B model and a hypernetwork model to predict embeddings for new
tokens that weren't in the original vocabulary.

The process involves:
1. Loading base model, hypernetwork, and tokenizers
2. Loading original and expanded vocabularies
3. Processing tokens in batches to generate embeddings
4. Handling special tokens by using base model embeddings
5. Saving the generated embeddings and token mappings

Inputs:
- Base model: e.g., mistralai/Mistral-7B-v0.1
- Hypernetwork: e.g., benjamin/zett-hypernetwork-Mistral-7B-v0.1
- Original vocab: e.g., decoders/data/tokenizer_hn_mistral/vocab.json
- Expanded vocab: e.g., decoders/data/large_tokenizer/vocab.json

Outputs:
- Embeddings: e.g., decoders/data/1M_vocab_embeddings/large_HN_embeddings.pt
- Token mappings: e.g., decoders/data/1M_vocab_embeddings/id2token.pkl, token2id.pkl
"""

import json
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from zett.utils import get_surface_form_matrix


def main():
    """Main function to execute the hypernetwork embeddings generation process."""

    # Config
    # Initial Hypernetwork Vocab
    VOCAB_INIT_PATH = "decoders/data/tokenizer_hn_mistral/vocab.json"
    # Expanded (1M) Vocab
    VOCAB_NEW_PATH = "decoders/data/large_tokenizer/vocab.json"
    # Output embeddings for 1M vocabulary
    OUTPUT_DIR = "decoders/data/1M_vocab_embeddings"
    # Languages file path
    LANGUAGES_FILE = "artifacts/26l.txt"
    TARGET_LANGUAGE = "en"
    BATCH_SIZE = 5000

    print("Starting hypernetwork embeddings generation...")

    base_model, hypernet, hn_tokenizer, device = load_models_and_tokenizers()
    source_embeddings = prepare_source_embeddings(base_model, device)

    # Load language index
    print("Loading language index...")
    langs = [x.strip() for x in open(LANGUAGES_FILE)]
    lang_index = torch.tensor(langs.index(TARGET_LANGUAGE), dtype=torch.int32).to(device)

    tokens_new_vocab = load_tokens_from_vocab(
        VOCAB_NEW_PATH
    )

    id2token, token2id = create_token_mappings(tokens_new_vocab)
    save_token_mappings(id2token, token2id, OUTPUT_DIR)

    target_surface_forms, special_tokens_mask = prepare_surface_forms(
        tokens_new_vocab, hn_tokenizer, device
    )

    # Generate embeddings in batches
    all_predicted_input_embeddings, all_predicted_output_embeddings = process_embeddings_in_batches(
        tokens_new_vocab=tokens_new_vocab,
        target_surface_forms=target_surface_forms,
        special_tokens_mask=special_tokens_mask,
        hypernet=hypernet,
        base_model=base_model,
        source_embeddings=source_embeddings,
        lang_index=lang_index,
        device=device,
        batch_size=BATCH_SIZE,
    )

    print(f"Saving embeddings to {OUTPUT_DIR}/large_HN_embeddings.pt...")
    torch.save(
        {
            "input_embeddings": all_predicted_input_embeddings,
            "output_embeddings": all_predicted_output_embeddings,
        },
        f"{OUTPUT_DIR}/large_HN_embeddings.pt",
    )

    print(f"Generated embeddings for {len(tokens_new_vocab):,} tokens")
    print(f"Input embeddings shape: {all_predicted_input_embeddings.shape}")
    print(f"Output embeddings shape: {all_predicted_output_embeddings.shape}")


def load_models_and_tokenizers(base_model_path: str = "mistralai/Mistral-7B-v0.1",
                               hypernet_path: str = "benjamin/zett-hypernetwork-Mistral-7B-v0.1",
                               hn_tokenizer_path: str = "benjamin/zett-hypernetwork-Mistral-7B-v0.1"):
    """
    Load the base model, hypernetwork, and tokenizers.

    Returns:
        Tuple containing (base_model, hypernet, hn_tokenizer, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

    print(f"Loading {hypernet_path} model...")
    hypernet_model = AutoModel.from_pretrained(hypernet_path, trust_remote_code=True
                                               ).to(device)

    print(f"Loading {hn_tokenizer_path} tokenizer...")
    hn_tokenizer = AutoTokenizer.from_pretrained(
        hn_tokenizer_path
    )

    return base_model, hypernet_model, hn_tokenizer, device


def prepare_source_embeddings(base_model: AutoModelForCausalLM, device: torch.device) -> torch.Tensor:
    """
    Prepare source embeddings by concatenating input and output embeddings.

    Args:
        base_model: The base Mistral model
        device: Target device for tensors

    Returns:
        Concatenated source embeddings
    """
    print("Preparing source embeddings...")
    source_embeddings = torch.concatenate(
        [
            base_model.get_input_embeddings().weight.data,
            base_model.get_output_embeddings().weight.data,
        ],
        axis=1,
    )
    return source_embeddings.to(device)


def load_tokens_from_vocab(vocab_new_path: str) -> List[str]:
    """
    Load expanded vocabularies.

    Args:
        vocab_init_path: Path to original HN vocabulary file
        vocab_new_path: Path to expanded e.g., 1M vocabulary file

    Returns:
        Returns list of tokens in the expanded vocabulary
    """
    with open(vocab_new_path, "r") as file:
        vocab_new = json.load(file)

    tokens_new_vocab = list(set(vocab_new.keys()))

    return tokens_new_vocab


def create_token_mappings(tokens_new_vocab: List[str]) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Create token-to-id and id-to-token mappings.

    Args:
        tokens_new_vocab: List of tokens in the expanded vocabulary

    Returns:
        Tuple containing (id2token, token2id) mappings
    """
    print("Creating token mappings...")
    id2token = {idx: token for idx, token in enumerate(tokens_new_vocab)}
    token2id = {token: idx for idx, token in enumerate(tokens_new_vocab)}

    return id2token, token2id


def save_token_mappings(id2token: Dict[int, str], token2id: Dict[str, int], output_dir: str):
    """
    Save token mappings to pickle files.

    Args:
        id2token: Mapping from token IDs to tokens
        token2id: Mapping from tokens to token IDs
        output_dir: Directory to save the mappings
    """
    print("Saving token mappings...")
    with open(f"{output_dir}/id2token.pkl", "wb") as id_file:
        pickle.dump(id2token, id_file)

    with open(f"{output_dir}/token2id.pkl", "wb") as token_file:
        pickle.dump(token2id, token_file)


def prepare_surface_forms(tokens_new_vocab: List[str], hn_tokenizer: AutoTokenizer,
                          device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare surface forms and special token masks for the expanded vocabulary.

    Args:
        tokens_new_vocab: List of tokens in the expanded vocabulary
        hn_tokenizer: Hypernetwork tokenizer
        device: Target device for tensors

    Returns:
        Tuple containing (target_surface_forms, special_tokens_mask)
    """
    print("Preparing surface forms...")
    target_surface_forms = get_surface_form_matrix(
        tokens_new_vocab,
        maxlen=hn_tokenizer.backend_tokenizer.model.config.hn_surface_maxlen,
        tokenizer_to_use=hn_tokenizer,
    )[0]
    target_surface_forms = torch.from_numpy(target_surface_forms).to(device)

    special_tokens_mask = torch.isin(
        target_surface_forms[:, 0],
        torch.tensor(hn_tokenizer.all_special_ids, device=device),
    )

    return target_surface_forms, special_tokens_mask


def generate_embeddings_batch(
    hypernet: AutoModel,
    batch_surface_forms: torch.Tensor,
    batch_special_tokens_mask: torch.Tensor,
    base_input_embeddings: torch.Tensor,
    base_output_embeddings: torch.Tensor,
    source_embeddings: torch.Tensor,
    lang_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate embeddings for a batch of tokens.

    Args:
        hypernet: Hypernetwork model
        batch_surface_forms: Surface forms for the current batch
        batch_special_tokens_mask: Special token mask for the current batch
        base_input_embeddings: Base model input embeddings
        base_output_embeddings: Base model output embeddings
        source_embeddings: Source embeddings for the hypernetwork
        lang_index: Language index tensor

    Returns:
        Tuple containing (predicted_input_embeddings, predicted_output_embeddings) each of shape (batch_size, hidden_size)
    """
    with torch.no_grad():
        predicted_input_embeddings, predicted_output_embeddings, _ = hypernet(
            batch_surface_forms,
            lang_index=lang_index,
            source_embeddings=source_embeddings,
        )

        # Use base model embeddings for special tokens
        predicted_input_embeddings[batch_special_tokens_mask] = base_input_embeddings[
            batch_surface_forms[batch_special_tokens_mask, 0]
        ]
        predicted_output_embeddings[batch_special_tokens_mask] = base_output_embeddings[
            batch_surface_forms[batch_special_tokens_mask, 0]
        ]

        return predicted_input_embeddings, predicted_output_embeddings


def process_embeddings_in_batches(
    tokens_new_vocab: List[str],
    target_surface_forms: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    hypernet: AutoModel,
    base_model: AutoModelForCausalLM,
    source_embeddings: torch.Tensor,
    lang_index: torch.Tensor,
    device: torch.device,
    batch_size: int = 5000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process all tokens in batches to generate embeddings.

    Args:
        tokens_new_vocab: List of tokens in the expanded vocabulary
        target_surface_forms: Surface forms for all tokens
        special_tokens_mask: Special token mask for all tokens
        hypernet: Hypernetwork model
        base_model: Base (Mistral) model
        source_embeddings: Source embeddings for the hypernetwork
        lang_index: Language index tensor
        device: Target device for tensors
        batch_size: Size of each processing batch

    Returns:
        Tuple containing (all_predicted_input_embeddings, all_predicted_output_embeddings)
    """
    print(f"Processing {len(tokens_new_vocab):,} tokens in batches of {batch_size}...")

    base_input_embeddings = base_model.get_input_embeddings().weight.data.to(device)
    base_output_embeddings = base_model.get_output_embeddings().weight.data.to(device)

    num_batches = len(tokens_new_vocab) // batch_size + (
        1 if len(tokens_new_vocab) % batch_size != 0 else 0
    )

    all_predicted_input_embeddings = []
    all_predicted_output_embeddings = []

    for i in tqdm(range(num_batches), desc="Generating embeddings"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(tokens_new_vocab))

        batch_surface_forms = target_surface_forms[start_idx:end_idx]
        batch_special_tokens_mask = special_tokens_mask[start_idx:end_idx]

        predicted_input_embeddings, predicted_output_embeddings = generate_embeddings_batch(
            hypernet=hypernet,
            batch_surface_forms=batch_surface_forms,
            batch_special_tokens_mask=batch_special_tokens_mask,
            base_input_embeddings=base_input_embeddings,
            base_output_embeddings=base_output_embeddings,
            source_embeddings=source_embeddings,
            lang_index=lang_index,
        )

        all_predicted_input_embeddings.append(predicted_input_embeddings.cpu())
        all_predicted_output_embeddings.append(predicted_output_embeddings.cpu())

    # Combine all embeddings
    all_predicted_input_embeddings = torch.cat(all_predicted_input_embeddings, dim=0)
    all_predicted_output_embeddings = torch.cat(all_predicted_output_embeddings, dim=0)

    return all_predicted_input_embeddings, all_predicted_output_embeddings


if __name__ == "__main__":
    main()
