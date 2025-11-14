from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from utils.patch_extractor import PatchEmbeddingBatch


@dataclass
class InferenceResult:
    """Container for holding anomaly detection outputs for a single sample."""

    image_score: float
    anomaly_map: torch.Tensor
    patch_scores: torch.Tensor
    label: int
    mask: Optional[torch.Tensor]
    metadata: Dict[str, str]


class PatchCore(nn.Module):
    """Minimal PatchCore implementation that wires the building blocks together."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        patch_extractor,
        memory_bank,
        nearest_neighbor,
        coreset_sampler: Optional[callable] = None,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.patch_extractor = patch_extractor
        self.memory_bank = memory_bank
        self.nearest_neighbor = nearest_neighbor
        self.coreset_sampler = coreset_sampler

    def build_memory_bank(self, dataloader, device: torch.device) -> None:
        """Construct the reference memory bank from training images."""
        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(device, non_blocking=True)
                patch_batch = self._encode(images)
                flattened = self._flatten_batch(patch_batch.embeddings)
                self.memory_bank.add(flattened)

        if self.coreset_sampler is not None:
            condensed = self.coreset_sampler(self.memory_bank.get())
            self.memory_bank.replace(condensed)

    def evaluate(self, dataloader, device: torch.device) -> List[InferenceResult]:
        """Run anomaly scoring on a dataloader and return rich per-sample outputs."""
        self.eval()
        results: List[InferenceResult] = []
        with torch.no_grad():
            reference_bank = self.memory_bank.get(device=device)
            for batch in dataloader:
                images = batch["image"].to(device, non_blocking=True)
                patch_batch = self._encode(images)
                batch_metadata = batch.get("meta", [{}] * images.size(0))
                batch_labels = batch.get("label")
                if batch_labels is None:
                    batch_labels = torch.zeros(images.size(0), dtype=torch.long)
                if isinstance(batch_labels, torch.Tensor):
                    batch_labels = batch_labels.cpu()
                batch_masks = batch.get("mask")
                if isinstance(batch_masks, torch.Tensor):
                    batch_masks = batch_masks.cpu()
                for idx in range(images.size(0)):
                    sample_embedding = patch_batch.embeddings[idx]
                    patch_scores, anomaly_map = self.nearest_neighbor.score(
                        sample_embedding, reference_bank, patch_batch.grid_size
                    )
                    image_score = patch_scores.max().item()
                    results.append(
                        InferenceResult(
                            image_score=image_score,
                            anomaly_map=anomaly_map.cpu(),
                            patch_scores=patch_scores.cpu(),
                            label=int(batch_labels[idx]),
                            mask=batch_masks[idx] if isinstance(batch_masks, torch.Tensor) else None,
                            metadata=batch_metadata[idx],
                        )
                    )
        return results

    def _encode(self, images: torch.Tensor) -> PatchEmbeddingBatch:
        """Forward helper that extracts multi-scale features and patches."""
        feature_maps = self.feature_extractor(images)
        patch_batch = self.patch_extractor(feature_maps)
        return patch_batch

    @staticmethod
    def _flatten_batch(batch_embeddings: torch.Tensor) -> torch.Tensor:
        """Collapse the batch and spatial dimensions into a single patch dimension."""
        bsz, num_patches, dim = batch_embeddings.shape
        return batch_embeddings.reshape(bsz * num_patches, dim)
