from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2

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


def overlay_heatmap(image, anomaly_map, save_path=None, return_image=False):
    """
    Overlay a colored anomaly heatmap on top of the original image.

    image:      PIL.Image (original)
    anomaly_map: numpy array (H, W) with anomaly scores
    """

    # anomaly_map: H x W (float) -> normalize 0–255
    anomaly_map = anomaly_map.astype(np.float32)
    h, w = anomaly_map.shape

    # ✅ resize original image to match heatmap size
    image = image.resize((w, h))
    original = np.array(image)          # H x W x 3

    # normalize anomaly map
    norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    heatmap_uint8 = np.uint8(norm * 255)

    # apply colormap and convert BGR -> RGB
    heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # overlay heatmap on original image
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    overlay_pil = Image.fromarray(overlay)

    # if caller wants the image back
    if return_image:
        return overlay_pil

    # otherwise just save to disk
    if save_path is not None:
        overlay_pil.save(save_path)

