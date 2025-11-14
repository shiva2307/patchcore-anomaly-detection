from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn
from torchvision import models

__all__ = ["WideResNetFeatureExtractor"]


class WideResNetFeatureExtractor(nn.Module):
    """Feature extractor that exposes intermediate WideResNet activation maps."""

    def __init__(
        self,
        layers: Sequence[str] = ("layer2", "layer3"),
        pretrained: bool = True,
        train_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.layers = list(layers)
        self.backbone = self._build_backbone(pretrained)
        self._activations: dict[str, torch.Tensor] = {}
        self._register_hooks()

        if not train_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        self._activations.clear()
        _ = self.backbone(x)
        return [self._activations[name] for name in self.layers]

    def _make_hook(self, name: str):
        def hook(_module, _inputs, outputs):
            self._activations[name] = outputs.detach()

        return hook

    def _register_hooks(self) -> None:
        module_dict = dict(self.backbone.named_modules())
        missing_layers = [layer for layer in self.layers if layer not in module_dict]
        if missing_layers:
            raise ValueError(f"Invalid layer names for hook registration: {missing_layers}")

        for layer_name in self.layers:
            module_dict[layer_name].register_forward_hook(self._make_hook(layer_name))

    @staticmethod
    def _build_backbone(pretrained: bool) -> nn.Module:
        try:
            weight_enum = (
                models.Wide_ResNet50_2_Weights.IMAGENET1K_V2 if pretrained else None
            )
            return models.wide_resnet50_2(weights=weight_enum)
        except AttributeError:
            # Fallback for older torchvision versions.
            return models.wide_resnet50_2(pretrained=pretrained)
