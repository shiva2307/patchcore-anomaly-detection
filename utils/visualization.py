from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def _to_numpy_image(image: torch.Tensor | np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
        if array.ndim == 3:
            array = np.transpose(array, (1, 2, 0))
    elif isinstance(image, Image.Image):
        array = np.array(image)
    else:
        array = image

    array = array.astype(np.float32)
    array -= array.min()
    if array.max() > 0:
        array /= array.max()
    return array


def _to_numpy_map(anomaly_map: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(anomaly_map, torch.Tensor):
        array = anomaly_map.detach().cpu().numpy()
    else:
        array = anomaly_map
    array = array.astype(np.float32)
    array -= array.min()
    if array.max() > 0:
        array /= array.max()
    return array


def overlay_heatmap(
    image: torch.Tensor | np.ndarray | Image.Image,
    anomaly_map: torch.Tensor | np.ndarray,
    cmap: str = "jet",
    alpha: float = 0.5,
    save_path: Optional[str | Path] = None,
):
    """Overlay an anomaly heatmap on top of the source image."""
    rgb = _to_numpy_image(image)
    heatmap = _to_numpy_map(anomaly_map)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(rgb)
    ax.imshow(heatmap, cmap=cmap, alpha=alpha)
    ax.axis("off")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    return fig
