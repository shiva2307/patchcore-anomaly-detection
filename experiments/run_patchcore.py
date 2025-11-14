from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from datasets.mvtec_loader import MVTecDataset
from models.patchcore import PatchCore
from utils.coreset import CoresetSampler
from utils.feature_extractor import WideResNetFeatureExtractor
from utils.memory_bank import MemoryBank
from utils.nearest_neighbor import NearestNeighborScorer
from utils.patch_extractor import PatchExtractor
from utils.visualization import overlay_heatmap


def parse_args():
    parser = argparse.ArgumentParser(description="PatchCore inference on MVTec AD.")
    parser.add_argument("--data-root", type=str, required=True, help="Path to the MVTec dataset root.")
    parser.add_argument("--category", type=str, required=True, help="Object category (e.g. screw, bottle).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--crop", type=int, default=224)
    parser.add_argument("--augment", action="store_true", help="Enable light augmentation on the train split.")
    parser.add_argument("--layers", type=str, default="layer2,layer3", help="Comma separated backbone layers.")
    parser.add_argument("--coreset-ratio", type=float, default=0.01)
    parser.add_argument("--coreset-min", type=int, default=512)
    parser.add_argument("--k", type=int, default=5, help="K for kNN anomaly scoring.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--save-dir", type=str, default=None, help="Optional directory to store visualizations.")
    parser.add_argument("--visualizations", type=int, default=4, help="Number of qualitative samples to save.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MVTecDataset(
        root=args.data_root,
        category=args.category,
        split="train",
        resize=args.resize,
        crop=args.crop,
        augment=args.augment,
    )
    test_dataset = MVTecDataset(
        root=args.data_root,
        category=args.category,
        split="test",
        resize=args.resize,
        crop=args.crop,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    feature_extractor = WideResNetFeatureExtractor(layers=args.layers.split(","))
    patch_extractor = PatchExtractor(kernel_size=3, stride=1, normalize=True)
    memory_bank = MemoryBank(device="cpu")
    coreset_sampler = CoresetSampler(ratio=args.coreset_ratio, min_samples=args.coreset_min)
    nn_scorer = NearestNeighborScorer(k=args.k)

    model = PatchCore(
        feature_extractor=feature_extractor,
        patch_extractor=patch_extractor,
        memory_bank=memory_bank,
        nearest_neighbor=nn_scorer,
        coreset_sampler=coreset_sampler,
    ).to(device)

    print(f"Building memory bank for category {args.category} on {device}...")
    model.build_memory_bank(train_loader, device=device)
    print(f"Memory bank size: {len(memory_bank)}")

    print("Evaluating...")
    results = model.evaluate(test_loader, device=device)

    image_scores = np.array([res.image_score for res in results])
    image_labels = np.array([res.label for res in results])
    image_auc = roc_auc_score(image_labels, image_scores)

    pixel_scores = []
    pixel_labels = []
    for res in results:
        if res.mask is None:
            continue
        anomaly_map = res.anomaly_map.unsqueeze(0).unsqueeze(0)
        upsampled = F.interpolate(
            anomaly_map,
            size=res.mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu()
        pixel_scores.append(upsampled.flatten().numpy())
        pixel_labels.append(res.mask.flatten().numpy())

    pixel_auc = (
        roc_auc_score(np.concatenate(pixel_labels), np.concatenate(pixel_scores))
        if pixel_scores
        else float("nan")
    )

    print(f"Image-level ROC-AUC: {image_auc:.4f}")
    print(f"Pixel-level ROC-AUC: {pixel_auc:.4f}")

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for idx, res in enumerate(results[: args.visualizations]):
            image_path = Path(res.metadata["path"])
            anomaly_map = res.anomaly_map.unsqueeze(0).unsqueeze(0)
            upsampled = F.interpolate(
                anomaly_map,
                size=(args.crop, args.crop),
                mode="bilinear",
                align_corners=False,
            ).squeeze().cpu().numpy()
            image = Image.open(image_path).convert("RGB")
            overlay_heatmap(
                image=image,
                anomaly_map=upsampled,
                save_path=save_dir / f"{idx:03d}_{image_path.stem}.png",
            )
        print(f"Saved qualitative visualizations to {save_dir}")


if __name__ == "__main__":
    main()
