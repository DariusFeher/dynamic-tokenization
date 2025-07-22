"""
LRU Cache for Hypernetwork Embeddings

This class implements a fixed-size LRU cache for storing token embeddings and biases, used in dynamic tokenization experiments
with hypernetworks.


Usage:
    cache = LRU_Cache(cache_size=5000, emb_size=768, device='cuda')
    idx = cache.get('token')
    cache.put(['token'], [embedding_tensor], biases)
    cache.evict_with_exceptions({'important_token'}, 10)
"""
import torch
from collections import OrderedDict
from typing import Union, List, Set, Optional

class LRU_Cache:
    def __init__(self, cache_size: int, emb_size: int = 768, device: str = "cpu"):
        """
        Initialize the LRU_Cache.

        Args:
            cache_size (int): Maximum number of tokens to cache.
            emb_size (int): Size of each embedding vector.
            device (str): Device to store tensors ('cpu' or 'cuda').
        """
        self.capacity = cache_size
        assert str(device) == "cuda" or "cuda" in str(device)
        self.emb_size = emb_size
        self.hypernet_preds = torch.zeros(cache_size, emb_size).to(device)
        self.biases = torch.zeros(cache_size, 1).to(device)
        self.token2idx = OrderedDict()
        self.free_indices = [i for i in range(cache_size - 1, -1, -1)]

    def get(self, key: str) -> Optional[int]:
        """
        Retrieve the cache index for a token, updating its LRU status.

        Args:
            key (str): The token to look up.
        Returns:
            Optional[int]: The cache index if present, else None.
        """
        if key not in self.token2idx:
            return None
        self.token2idx.move_to_end(key)
        return self.token2idx[key]

    def put(
        self, tokens: List[str], values: torch.Tensor, biases: torch.Tensor = torch.tensor([])
    ) -> None:
        """
        Insert tokens and their embeddings (and optional biases) into the cache.
        Evicts least recently used tokens if necessary.

        Args:
            tokens (List[str]): Tokens to insert.
            values (torch.Tensor): Embedding vectors for the tokens.
            biases (torch.Tensor, optional): Biases for the tokens.
        Raises:
            Exception: If there are not enough free indices for insertion.
        """
        indices = []
        for i, token in enumerate(tokens):
            if token in self.token2idx:
                self.token2idx.move_to_end(token)
                indices.append(self.token2idx[token])
            else:
                if not self.free_indices:
                    raise Exception("No free indices available in cache. Consider evicting tokens.")
                token_idx = self.free_indices.pop()
                self.token2idx[token] = token_idx
                indices.append(token_idx)
                
        self.hypernet_preds[indices] = values
        if biases.numel() > 0:
            self.biases[indices] = biases

    def evict_with_exceptions(
        self, tokens_eviction_exception: Set[str], nr_tokens_to_evict: int
    ) -> None:
        """
        Evict up to nr_tokens_to_evict tokens from the cache, skipping those in the exception set.

        Args:
            tokens_eviction_exception (Set[str]): Tokens that should not be evicted.
            nr_tokens_to_evict (int): Number of tokens to evict.
        """
        evicted_tokens = 0
        tokens_to_remove = []
        for token in list(self.token2idx.keys()):
            if token not in tokens_eviction_exception:
                self.free_indices.append(self.token2idx[token])
                tokens_to_remove.append(token)
                evicted_tokens += 1
                if evicted_tokens == nr_tokens_to_evict:
                    break
        for token in tokens_to_remove:
            del self.token2idx[token]

    def move_tokens_to_end(self, tokens: List[str]) -> None:
        """
        Mark tokens as recently used (move to end of LRU order).

        Args:
            tokens (List[str]): Tokens to update.
        Raises:
            Exception: If a token is not in the cache.
        """
        for token in tokens:
            if token in self.token2idx:
                self.token2idx.move_to_end(token)
            else:
                raise Exception(f"Token {token} not in cache.")

    @property
    def size(self) -> int:
        """
        Returns the current number of tokens in the cache.
        """
        return len(self.token2idx)
