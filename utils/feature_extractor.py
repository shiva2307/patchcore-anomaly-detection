from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn
from torchvision import models

__all__ = ["CNNFeatureExtractor", "WideResNetFeatureExtractor"]


class CNNFeatureExtractor(nn.Module):
    """Feature extractor that exposes intermediate activation maps from ResNet variants."""

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3"),
        pretrained: bool = True,
        train_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.layers = list(layers)
        self.backbone = self._build_backbone(backbone, pretrained)
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
    def _build_backbone(backbone: str, pretrained: bool) -> nn.Module:
        name = backbone.lower()
        if name in {"wide_resnet50_2", "wide_resnet50", "wideresnet50"}:
            try:
                weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V2 if pretrained else None
                return models.wide_resnet50_2(weights=weights)
            except AttributeError:
                return models.wide_resnet50_2(pretrained=pretrained)
        if name in {"resnet50", "rn50"}:
            try:
                weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
                return models.resnet50(weights=weights)
            except AttributeError:
                return models.resnet50(pretrained=pretrained)
        raise ValueError(f"Unsupported backbone '{backbone}'.")


# Backwards compatible alias.
class WideResNetFeatureExtractor(CNNFeatureExtractor):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("backbone", "wide_resnet50_2")
        super().__init__(*args, **kwargs)
