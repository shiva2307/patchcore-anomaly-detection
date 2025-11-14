from __future__ import annotations

import torch


def farthest_point_sampling(embeddings: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Naive Farthest Point Sampling (FPS) used by PatchCore for coreset selection."""
    if embeddings.dim() != 2:
        raise ValueError("Embeddings for FPS must be 2D.")

    total = embeddings.size(0)
    if num_samples >= total:
        return embeddings

    device = embeddings.device
    selected_indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    distances = torch.full((total,), float("inf"), device=device)

    # Randomly seed FPS to encourage coverage.
    seed = torch.randint(0, total, (1,), device=device)
    selected_indices[0] = seed

    centroid = embeddings[seed]
    distances = torch.minimum(distances, torch.norm(embeddings - centroid, dim=1))

    for i in range(1, num_samples):
        next_index = torch.argmax(distances)
        selected_indices[i] = next_index
        candidate = embeddings[next_index]
        distances = torch.minimum(distances, torch.norm(embeddings - candidate, dim=1))

    return embeddings[selected_indices]


class CoresetSampler:
    """Wrapper that turns FPS into a callable coreset sampling strategy."""

    def __init__(self, ratio: float = 0.01, min_samples: int = 512) -> None:
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1].")
        self.ratio = ratio
        self.min_samples = min_samples

    def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
        target = max(self.min_samples, int(len(embeddings) * self.ratio))
        target = min(target, len(embeddings))
        return farthest_point_sampling(embeddings, target)
