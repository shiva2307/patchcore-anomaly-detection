from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from utils.coreset import CoresetSampler
from utils.feature_extractor import CNNFeatureExtractor
from utils.memory_bank import MemoryBank
from utils.nearest_neighbor import NearestNeighborScorer
from utils.patch_extractor import PatchEmbeddingBatch, PatchExtractor

BatchType = Union[torch.Tensor, Dict[str, torch.Tensor], Tuple, List]


@dataclass
class InferenceResult:
    """Container for holding anomaly detection outputs for a single sample."""

    image_score: float
    anomaly_map: torch.Tensor
    patch_scores: torch.Tensor
    label: Optional[int] = None
    mask: Optional[torch.Tensor] = None
    metadata: Dict[str, str] = field(default_factory=dict)


class PatchCore(nn.Module):
    """Minimal PatchCore implementation with fit/evaluate/predict helpers."""

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3"),
        device: Optional[torch.device | str] = None,
        patch_size: int = 3,
        stride: int = 1,
        coreset_ratio: float = 0.01,
        coreset_min_samples: int = 512,
        k_neighbors: int = 5,
        normalize_patches: bool = True,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.feature_extractor = CNNFeatureExtractor(
            backbone=backbone,
            layers=layers,
            pretrained=True,
            train_backbone=False,
        ).to(self.device)
        self.patch_extractor = PatchExtractor(
            kernel_size=patch_size,
            stride=stride,
            normalize=normalize_patches,
        )
        self.memory_bank = MemoryBank(device="cpu")
        self.nearest_neighbor = NearestNeighborScorer(k=k_neighbors)
        self.coreset_sampler = (
            CoresetSampler(ratio=coreset_ratio, min_samples=coreset_min_samples)
            if coreset_ratio and 0 < coreset_ratio < 1
            else None
        )

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        device = kwargs.get("device")
        if device is None and args:
            device = args[0]
        if device is not None:
            self.device = torch.device(device)
        return module

    def fit(self, training_images: Iterable[BatchType]) -> "PatchCore":
        """Build the reference memory bank from training images."""
        self.memory_bank.clear()
        self.eval()
        with torch.no_grad():
            for batch in self._batch_iterator(training_images):
                images, _, _, _ = self._prepare_batch(batch)
                images = images.to(self.device, non_blocking=True)
                patch_batch = self._encode(images)
                flattened = self._flatten_batch(patch_batch.embeddings)
                self.memory_bank.add(flattened.cpu())

        if len(self.memory_bank) == 0:
            raise RuntimeError("Memory bank is empty. Ensure training_images is not empty.")

        if self.coreset_sampler is not None:
            condensed = self.coreset_sampler(self.memory_bank.get())
            self.memory_bank.replace(condensed)
        return self

    def evaluate(self, test_images: Iterable[BatchType]) -> List[InferenceResult]:
        """Run anomaly scoring on a dataset and return per-sample results."""
        self._ensure_memory_bank()
        self.eval()
        results: List[InferenceResult] = []
        with torch.no_grad():
            reference_bank = self.memory_bank.get(device=self.device)
            for batch in self._batch_iterator(test_images):
                images, labels, masks, metadata = self._prepare_batch(batch)
                images = images.to(self.device, non_blocking=True)
                patch_batch = self._encode(images)
                for idx in range(images.size(0)):
                    patch_scores, anomaly_map = self.nearest_neighbor.score(
                        patch_batch.embeddings[idx],
                        reference_bank,
                        patch_batch.grid_size,
                    )
                    results.append(
                        InferenceResult(
                            image_score=patch_scores.max().item(),
                            anomaly_map=anomaly_map.cpu(),
                            patch_scores=patch_scores.cpu(),
                            label=labels[idx],
                            mask=masks[idx],
                            metadata=metadata[idx],
                        )
                    )
        return results

    def predict(self, image: BatchType) -> InferenceResult:
        """Score a single image and return its anomaly map and score."""
        self._ensure_memory_bank()
        self.eval()
        with torch.no_grad():
            images, labels, masks, metadata = self._prepare_batch(image)
            images = images.to(self.device, non_blocking=True)
            patch_batch = self._encode(images)
            reference_bank = self.memory_bank.get(device=self.device)
            patch_scores, anomaly_map = self.nearest_neighbor.score(
                patch_batch.embeddings[0],
                reference_bank,
                patch_batch.grid_size,
            )
            return InferenceResult(
                image_score=patch_scores.max().item(),
                anomaly_map=anomaly_map.cpu(),
                patch_scores=patch_scores.cpu(),
                label=labels[0],
                mask=masks[0],
                metadata=metadata[0],
            )

    def _encode(self, images: torch.Tensor) -> PatchEmbeddingBatch:
        """Forward helper that extracts multi-scale features and patches."""
        feature_maps = self.feature_extractor(images)
        patch_batch = self.patch_extractor(feature_maps)
        return patch_batch

    @staticmethod
    def _flatten_batch(batch_embeddings: torch.Tensor) -> torch.Tensor:
        """Collapse the batch and spatial dimensions into a single patch dimension."""
        bsz, num_patches, dim = batch_embeddings.shape
        return batch_embeddings.view(bsz * num_patches, dim)

    def _ensure_memory_bank(self) -> None:
        if len(self.memory_bank) == 0:
            raise RuntimeError("Memory bank is empty. Call fit() before inference.")

    @staticmethod
    def _batch_iterator(data: Iterable[BatchType]) -> Iterator[BatchType]:
        if isinstance(data, (torch.Tensor, dict)):
            yield data
            return
        if isinstance(data, (list, tuple)):
            for batch in data:
                yield batch
            return
        if hasattr(data, "__iter__"):
            for batch in data:
                yield batch
            return
        raise TypeError("training_images/test_images must be an iterable of batches.")

    def _prepare_batch(
        self, batch: BatchType
    ) -> Tuple[torch.Tensor, List[Optional[int]], List[Optional[torch.Tensor]], List[Dict[str, str]]]:
        images, labels, masks, metadata = self._extract_batch_components(batch)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        batch_size = images.size(0)
        label_list = self._expand_labels(labels, batch_size)
        mask_list = self._expand_masks(masks, batch_size)
        metadata_list = self._expand_metadata(metadata, batch_size)
        return images.float(), label_list, mask_list, metadata_list

    @staticmethod
    def _extract_batch_components(batch: BatchType):
        images = labels = masks = metadata = None
        if isinstance(batch, torch.Tensor):
            images = batch
        elif isinstance(batch, dict):
            images = batch.get("image")
            labels = batch.get("label")
            masks = batch.get("mask")
            metadata = batch.get("meta")
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 0:
                raise ValueError("Empty batch encountered.")
            images = batch[0]
            if len(batch) > 1:
                labels = batch[1]
            if len(batch) > 2:
                masks = batch[2]
            if len(batch) > 3:
                metadata = batch[3]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}.")

        if images is None:
            raise ValueError("Batch must contain image tensors.")

        return images, labels, masks, metadata

    @staticmethod
    def _expand_labels(labels, batch_size: int) -> List[Optional[int]]:
        if labels is None:
            return [None] * batch_size
        if isinstance(labels, torch.Tensor):
            labels_list = labels.detach().cpu().tolist()
        elif isinstance(labels, (list, tuple)):
            labels_list = list(labels)
        else:
            labels_list = [labels] * batch_size
        if len(labels_list) != batch_size:
            raise ValueError("Number of labels does not match batch size.")
        return [int(label) if label is not None else None for label in labels_list]

    @staticmethod
    def _expand_masks(masks, batch_size: int) -> List[Optional[torch.Tensor]]:
        if masks is None:
            return [None] * batch_size
        if isinstance(masks, torch.Tensor):
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            if masks.size(0) != batch_size:
                raise ValueError("Number of masks does not match batch size.")
            return [masks[idx].detach().cpu() for idx in range(masks.size(0))]
        if isinstance(masks, (list, tuple)):
            if len(masks) != batch_size:
                raise ValueError("Number of masks does not match batch size.")
            expanded = []
            for item in masks:
                if isinstance(item, torch.Tensor):
                    expanded.append(item.detach().cpu())
                else:
                    expanded.append(item)
            return expanded
        return [masks] * batch_size

    @staticmethod
    def _expand_metadata(metadata, batch_size: int) -> List[Dict[str, str]]:
        if metadata is None:
            return [{} for _ in range(batch_size)]
        if isinstance(metadata, list):
            if len(metadata) != batch_size:
                raise ValueError("Metadata list length must match batch size.")
            return [
                meta if isinstance(meta, dict) else {"value": str(meta)}
                for meta in metadata
            ]
        if isinstance(metadata, dict):
            return [metadata] * batch_size
        return [{"value": str(metadata)} for _ in range(batch_size)]
