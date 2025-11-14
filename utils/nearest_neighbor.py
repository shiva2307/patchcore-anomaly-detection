from __future__ import annotations

from typing import Tuple

import torch


class NearestNeighborScorer:
    """Computes anomaly scores by measuring distances to a memory bank."""

    def __init__(self, k: int = 5, chunk_size: int = 2048) -> None:
        if k <= 0:
            raise ValueError("k must be positive.")
        self.k = k
        self.chunk_size = chunk_size

    def score(
        self,
        sample_embeddings: torch.Tensor,
        memory_embeddings: torch.Tensor,
        grid_size: Tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sample_embeddings.dim() != 2:
            raise ValueError("Sample embeddings must be 2D (patches, dim).")

        memory_embeddings = memory_embeddings.to(sample_embeddings.device)
        distances = self._chunked_cdist(sample_embeddings, memory_embeddings)
        k = min(self.k, memory_embeddings.size(0))
        nearest_dists, _ = torch.topk(distances, k=k, dim=1, largest=False)
        patch_scores = nearest_dists.mean(dim=1)
        anomaly_map = patch_scores.view(*grid_size)
        return patch_scores, anomaly_map

    def _chunked_cdist(
        self, samples: torch.Tensor, references: torch.Tensor
    ) -> torch.Tensor:
        if self.chunk_size is None or samples.size(0) <= self.chunk_size:
            return torch.cdist(samples, references)

        chunks = []
        for chunk in samples.split(self.chunk_size, dim=0):
            chunks.append(torch.cdist(chunk, references))
        return torch.cat(chunks, dim=0)
