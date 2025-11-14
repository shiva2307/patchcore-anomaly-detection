from __future__ import annotations

import torch


def extract_patches(feature_map: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Return non-overlapping patches of size (patch_size, patch_size).

    Args:
        feature_map: Tensor with shape (C, H, W).
        patch_size: Size of the square patch (p).

    Returns:
        Tensor of shape (num_patches, C, patch_size, patch_size).
    """
    if feature_map.dim() != 3:
        raise ValueError("feature_map must have shape (C, H, W).")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")

    channels, height, width = feature_map.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Feature map size {(height, width)} must be divisible by patch_size={patch_size}."
        )

    reshaped = feature_map.view(
        channels,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    patches = (
        reshaped.permute(1, 3, 0, 2, 4)
        .contiguous()
        .view(-1, channels, patch_size, patch_size)
    )
    return patches
