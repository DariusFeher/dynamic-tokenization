"""
Script to create a ScaNN index for fast approximate nearest neighbor search on large embedding datasets.

Usage:
    python create_scann_index.py --embeddings-path <path> --output-dir <dir> [--neighbors 200] [--num-leaves 2000] [--leaves-to-search 250] [--reorder 200]

Default values are set for Mistral 1M HN embeddings.
"""

import numpy as np
import scann
import torch
import argparse
import os


def build_scann_index(
    dataset: np.ndarray,
    neighbors: int = 200,
    num_leaves: int = 2000,
    leaves_to_search: int = 250,
    reorder: int = 200,
) -> scann.scann_ops_pybind.ScannSearcher:
    """Build a ScaNN index with the given parameters."""
    return (
        scann.scann_ops_pybind.builder(dataset, neighbors, "dot_product")
        .tree(num_leaves=num_leaves, num_leaves_to_search=leaves_to_search, training_sample_size=dataset.shape[0])
        .score_ah(3, anisotropic_quantization_threshold=0.2)
        .reorder(reorder)
        .build()
    )


def main():
    parser = argparse.ArgumentParser(description="Create a ScaNN index for 1M HN embeddings.")
    parser.add_argument("--embeddings-path", type=str, default="decoders/data/1M_vocab_embeddings/large_HN_embeddings.pt",
                        help="Path to the .pt file containing the embeddings.")
    parser.add_argument("--output-dir", type=str, default="decoders/data/scann_index/scann_index_6_neighbours_200_reorder_1000",
                        help="Directory to save the ScaNN index.")
    parser.add_argument("--neighbors", type=int, default=200, help="Number of neighbors to search.")
    parser.add_argument("--num-leaves", type=int, default=2000, help="Number of leaves in the ScaNN tree.")
    parser.add_argument("--leaves-to-search", type=int, default=250, help="Number of leaves to search at query time.")
    parser.add_argument("--reorder", type=int, default=200, help="Number of candidates to reorder after search.")
    args = parser.parse_args()

    print(f"Loading embeddings from {args.embeddings_path} ...")
    embeddings = torch.load(args.embeddings_path)
    output_embeds = embeddings["output_embeddings"]
    if isinstance(output_embeds, torch.Tensor):
        output_embeds = output_embeds.cpu().numpy()
    print(f"Normalizing {output_embeds.shape[0]} embeddings ...")
    normalized_dataset = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    print("Building ScaNN index ...")
    searcher = build_scann_index(
        normalized_dataset,
        neighbors=args.neighbors,
        num_leaves=args.num_leaves,
        leaves_to_search=args.leaves_to_search,
        reorder=args.reorder,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Serializing ScaNN index to {args.output_dir} ...")
    searcher.serialize(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
