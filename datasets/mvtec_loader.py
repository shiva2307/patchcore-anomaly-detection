from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGENET_STATS = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@dataclass
class SampleMetadata:
    path: str
    defect_type: str


class MVTecDataset(Dataset):
    """Thin PyTorch Dataset wrapper around the official MVTec AD folder layout."""

    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str = "train",
        resize: int = 256,
        crop: int = 224,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.category = category
        self.split = split
        self.crop_size = crop

        self.image_dir = self.root / category / split
        self.mask_dir = self.root / category / "ground_truth"
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing directory: {self.image_dir}")

        self.transform = self._build_image_transform(resize, crop, augment)
        self.mask_transform = self._build_mask_transform(resize, crop)
        self.samples = self._gather_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        image_tensor = self.transform(image)

        mask_available = sample["mask"] is not None and sample["mask"].exists()
        if mask_available:
            mask = Image.open(sample["mask"])
            mask_tensor = self.mask_transform(mask)
        else:
            mask_tensor = torch.zeros(1, self.crop_size, self.crop_size)

        return {
            "image": image_tensor,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "mask": mask_tensor,
            "path": str(sample["image"]),
            "meta": {
                "path": str(sample["image"]),
                "defect_type": sample["defect_type"],
                "has_mask": mask_available,
            },
        }

    def _build_image_transform(self, resize: int, crop: int, augment: bool):
        transforms: List = [T.Resize(resize)]
        if augment and self.split == "train":
            transforms.extend([T.RandomResizedCrop(crop), T.RandomHorizontalFlip()])
        else:
            transforms.append(T.CenterCrop(crop))
        transforms.extend([T.ToTensor(), T.Normalize(**IMAGENET_STATS)])
        return T.Compose(transforms)

    def _build_mask_transform(self, resize: int, crop: int):
        return T.Compose([T.Resize(resize, interpolation=T.InterpolationMode.NEAREST), T.CenterCrop(crop), T.ToTensor()])

    def _gather_samples(self):
        samples = []
        for defect_dir in sorted(p for p in self.image_dir.iterdir() if p.is_dir()):
            for image_path in sorted(defect_dir.glob("*.*")):
                if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                label = 0 if defect_dir.name == "good" else 1
                mask_path = None
                if self.split != "train" and label == 1:
                    mask_filename = f"{image_path.stem}_mask{image_path.suffix}"
                    mask_path = self.mask_dir / defect_dir.name / mask_filename

                samples.append(
                    {
                        "image": image_path,
                        "mask": mask_path,
                        "label": label,
                        "defect_type": defect_dir.name,
                    }
                )

        if not samples:
            raise RuntimeError(f"No samples found under {self.image_dir}")
        return samples
