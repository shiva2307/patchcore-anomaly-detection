from __future__ import annotations

from typing import Optional

import torch


class MemoryBank:
    """Simple tensor-based memory bank for PatchCore patch embeddings."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self._storage: Optional[torch.Tensor] = None

    def add(self, embeddings: torch.Tensor) -> None:
        self._validate_embeddings(embeddings)
        embeddings = embeddings.detach().to(self.device)
        if self._storage is None:
            self._storage = embeddings
        else:
            self._storage = torch.cat([self._storage, embeddings], dim=0)

    def replace(self, embeddings: torch.Tensor) -> None:
        self._validate_embeddings(embeddings)
        self._storage = embeddings.detach().to(self.device)

    def get(self, device: torch.device | None = None) -> torch.Tensor:
        if self._storage is None:
            raise RuntimeError("Memory bank is empty. Call PatchCore.fit() first.")
        if device is None or device == self.device:
            return self._storage
        return self._storage.to(device)

    def clear(self) -> None:
        self._storage = None

    def __len__(self) -> int:
        return 0 if self._storage is None else self._storage.size(0)

    @staticmethod
    def _validate_embeddings(embeddings: torch.Tensor) -> None:
        if embeddings.dim() != 2:
            raise ValueError(
                f"Embeddings must be 2D (num_patches, dim). Received shape {embeddings.shape}."
            )
