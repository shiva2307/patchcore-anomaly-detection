from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from .patch_extractor import extract_patches

__all__ = ["PatchEmbeddingBatch", "FeatureExtractor"]


@dataclass
class PatchEmbeddingBatch:
    embeddings: torch.Tensor  # Shape: (B, L, D)
    grid_size: Tuple[int, int]


class FeatureExtractor(nn.Module):
    """WideResNet50 backbone that outputs flattened patch embeddings."""

    def __init__(self, patch_size: int = 1) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.backbone = self._build_backbone()
        self.layer_names = ("layer2", "layer3")
        self._activations: dict[str, torch.Tensor] = {}
        self._register_hooks()
        self._freeze_backbone()

    def forward(self, x: torch.Tensor) -> PatchEmbeddingBatch:
        self._activations.clear()
        _ = self.backbone(x)
        layer2 = self._activations["layer2"]
        layer3 = self._activations["layer3"]

        if layer3.shape[-2:] != layer2.shape[-2:]:
            layer3 = F.interpolate(
                layer3,
                size=layer2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        batch_embeddings = []
        grid_h, grid_w = self._grid_dimensions(layer2.shape[-2:], self.patch_size)

        for idx in range(x.size(0)):
            l2_patches = extract_patches(layer2[idx], self.patch_size)
            l3_patches = extract_patches(layer3[idx], self.patch_size)
            flat_l2 = l2_patches.view(l2_patches.size(0), -1)
            flat_l3 = l3_patches.view(l3_patches.size(0), -1)
            batch_embeddings.append(torch.cat([flat_l2, flat_l3], dim=1))

        embeddings = torch.stack(batch_embeddings, dim=0)
        return PatchEmbeddingBatch(embeddings=embeddings, grid_size=(grid_h, grid_w))

    def _register_hooks(self) -> None:
        module_dict = dict(self.backbone.named_modules())
        missing_layers = [layer for layer in self.layer_names if layer not in module_dict]
        if missing_layers:
            raise ValueError(f"Invalid layer names: {missing_layers}")
        for name in self.layer_names:
            module_dict[name].register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, outputs):
            self._activations[name] = outputs.detach()

        return hook

    @staticmethod
    def _build_backbone() -> nn.Module:
        try:
            weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V2
            backbone = models.wide_resnet50_2(weights=weights)
        except AttributeError:
            backbone = models.wide_resnet50_2(pretrained=True)
        return backbone

    def _freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.backbone.eval()

    @staticmethod
    def _grid_dimensions(
        spatial_shape: Tuple[int, int], patch_size: int
    ) -> Tuple[int, int]:
        height, width = spatial_shape
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"Spatial dims {(height, width)} must be divisible by patch_size={patch_size}."
            )
        return height // patch_size, width // patch_size
