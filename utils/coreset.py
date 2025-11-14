from __future__ import annotations

import math

import torch


def farthest_point_sampling(
    embeddings: torch.Tensor,
    ratio: float,
    min_samples: int = 1,
) -> torch.Tensor:
    """Select a diverse coreset via greedy farthest-point sampling.

    Args:
        embeddings: Tensor of shape (N, D) containing patch descriptors.
        ratio: Fraction (0, 1] of descriptors to keep.
        min_samples: Lower bound on the coreset size.

    Returns:
        Tensor of shape (K, D) where K = ceil(N * ratio) clamped to [min_samples, N].
    """
    if embeddings.dim() != 2:
        raise ValueError("Embeddings for FPS must be 2D (num_vectors, dim).")
    if not 0 < ratio <= 1:
        raise ValueError("ratio must lie in (0, 1].")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive.")

    num_points = embeddings.size(0)
    target = int(math.ceil(num_points * ratio))
    target = max(min_samples, target)
    target = min(target, num_points)

    if target == num_points:
        return embeddings

    device = embeddings.device
    selected = torch.empty(target, dtype=torch.long, device=device)
    min_distances = torch.full((num_points,), float("inf"), device=device)

    # Start with the point farthest from the mean for stable behavior.
    centroid = embeddings.mean(dim=0, keepdim=True)
    initial_distances = torch.cdist(embeddings, centroid).squeeze(1)
    selected[0] = torch.argmax(initial_distances)
    anchor = embeddings[selected[0]].unsqueeze(0)
    min_distances = torch.minimum(min_distances, torch.cdist(embeddings, anchor).squeeze(1))

    for i in range(1, target):
        next_idx = torch.argmax(min_distances)
        selected[i] = next_idx
        anchor = embeddings[next_idx].unsqueeze(0)
        min_distances = torch.minimum(min_distances, torch.cdist(embeddings, anchor).squeeze(1))

    return embeddings[selected]


class CoresetSampler:
    """Wrapper that turns FPS into a callable coreset sampling strategy."""

    def __init__(self, ratio: float = 0.01, min_samples: int = 512) -> None:
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1].")
        self.ratio = ratio
        self.min_samples = min_samples

    def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
        return farthest_point_sampling(embeddings, self.ratio, self.min_samples)
