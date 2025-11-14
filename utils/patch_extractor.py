from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


@dataclass
class PatchEmbeddingBatch:
    """Helper dataclass to keep patch embeddings and their spatial resolution together."""

    embeddings: torch.Tensor  # Shape: (B, L, D)
    grid_size: Tuple[int, int]


class PatchExtractor:
    """Extracts flattened patch descriptors from multi-scale feature maps."""

    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        padding: int | None = None,
        normalize: bool = True,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding if padding is not None else kernel_size // 2
        self.normalize = normalize

    def __call__(self, feature_maps: Sequence[torch.Tensor]) -> PatchEmbeddingBatch:
        if not feature_maps:
            raise ValueError("feature_maps cannot be empty.")

        patch_batches = []
        target_size = feature_maps[0].shape[-2:]
        grid_size = None
        for fmap in feature_maps:
            if fmap.shape[-2:] != target_size:
                fmap = F.interpolate(
                    fmap,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            patches, grid_size = self._extract_from_single_map(fmap)
            patch_batches.append(patches)

        embeddings = torch.cat(patch_batches, dim=-1)
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=-1)

        if grid_size is None:
            raise RuntimeError("Failed to infer grid size from feature maps.")

        return PatchEmbeddingBatch(embeddings=embeddings, grid_size=grid_size)

    def _extract_from_single_map(
        self, feature_map: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        bsz, _, height, width = feature_map.shape
        padding = _pair(self.padding)
        kernel = _pair(self.kernel_size)
        stride = _pair(self.stride)
        dilation = _pair(self.dilation)

        unfolded = F.unfold(
            feature_map,
            kernel_size=kernel,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )  # (B, C * K * K, L)
        unfolded = unfolded.transpose(1, 2)  # (B, L, C * K * K)

        h_out = self._compute_out_dim(height, kernel[0], padding[0], stride[0], dilation[0])
        w_out = self._compute_out_dim(width, kernel[1], padding[1], stride[1], dilation[1])
        grid_size = (h_out, w_out)

        return unfolded, grid_size

    @staticmethod
    def _compute_out_dim(
        size: int, kernel: int, padding: int, stride: int, dilation: int
    ) -> int:
        return math.floor(
            (size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
        )
