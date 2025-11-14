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
from utils.coreset import CoresetSampler
from utils.feature_extractor import FeatureExtractor
from utils.memory_bank import MemoryBank
from utils.nearest_neighbor import NearestNeighborScorer
from utils.visualization import overlay_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PatchCore training and evaluation.")
    parser.add_argument("--data-root", type=str, default="mvtec", help="Path to the MVTec dataset.")
    parser.add_argument("--class", dest="category", type=str, default="bottle", help="MVTec class name.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--crop", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--coreset-ratio", type=float, default=0.01)
    parser.add_argument("--coreset-min", type=int, default=512)
    parser.add_argument("--k", type=int, default=5, help="k for kNN anomaly scoring.")
    parser.add_argument("--visualizations", type=int, default=8, help="Number of heatmaps to save.")
    parser.add_argument("--save-dir", type=str, default="outputs", help="Directory to store heatmaps.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser.parse_args()


def flatten_embeddings(batch_embeddings: torch.Tensor) -> torch.Tensor:
    bsz, patches, dim = batch_embeddings.shape
    return batch_embeddings.view(bsz * patches, dim)


def build_memory_bank(
    feature_extractor: FeatureExtractor,
    dataloader: DataLoader,
    memory_bank: MemoryBank,
    device: torch.device,
    coreset_sampler: CoresetSampler | None,
) -> None:
    feature_extractor.eval()
    memory_bank.clear()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            patch_batch = feature_extractor(images)
            memory_bank.add(flatten_embeddings(patch_batch.embeddings).cpu())

    if len(memory_bank) == 0:
        raise RuntimeError("Memory bank is empty. Check the training dataset.")

    if coreset_sampler is not None:
        memory_bank.replace(coreset_sampler(memory_bank.get()))


def evaluate(
    feature_extractor: FeatureExtractor,
    dataloader: DataLoader,
    memory_bank: MemoryBank,
    scorer: NearestNeighborScorer,
    device: torch.device,
) -> list[dict]:
    feature_extractor.eval()
    memory = memory_bank.get(device=device)
    results = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            patch_batch = feature_extractor(images)
            labels = batch["label"]
            masks = batch["mask"]
            metadata = batch["meta"]  # this is a dict of lists

            for idx in range(images.size(0)):
                patch_scores, anomaly_map = scorer.score(
                    patch_batch.embeddings[idx],
                    memory,
                    patch_batch.grid_size,
                )

                # 🔹 build per-sample meta dict from dict-of-lists
                if isinstance(metadata, dict):
                    sample_meta = {
                        key: (value[idx] if isinstance(value, (list, tuple)) else value)
                        for key, value in metadata.items()
                    }
                else:
                    # fallback if someone later changes collate_fn
                    sample_meta = metadata[idx]

                results.append(
                    {
                        "score": patch_scores.max().item(),
                        "patch_scores": patch_scores.cpu(),
                        "anomaly_map": anomaly_map.cpu(),
                        "label": int(labels[idx]),
                        "mask": masks[idx],
                        "meta": sample_meta,
                    }
                )
    return results



def compute_metrics(results: list[dict]) -> tuple[float, float]:
    image_scores = np.array([item["score"] for item in results])
    image_labels = np.array([item["label"] for item in results])
    image_auc = roc_auc_score(image_labels, image_scores)

    pixel_scores = []
    pixel_labels = []
    for item in results:
        mask = item["mask"]
        if mask is None:
            continue
        anomaly_map = item["anomaly_map"].unsqueeze(0).unsqueeze(0)
        upsampled = F.interpolate(
            anomaly_map,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()
        pixel_scores.append(upsampled.flatten())
        pixel_labels.append(mask.flatten().numpy())

    pixel_auc = (
        roc_auc_score(np.concatenate(pixel_labels), np.concatenate(pixel_scores))
        if pixel_scores
        else float("nan")
    )
    return image_auc, pixel_auc


def save_heatmaps(results: list[dict], crop_size: int, save_dir: Path, limit: int) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(results[:limit]):
        image_path = Path(item["meta"]["path"])  # original image path

        # Read original image
        image = Image.open(image_path).convert("RGB")

        # Upsample anomaly map to image size
        anomaly_map = item["anomaly_map"].unsqueeze(0).unsqueeze(0)
        upsampled = F.interpolate(
            anomaly_map,
            size=(crop_size, crop_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # Create heatmap overlay
        heatmap_image = overlay_heatmap(
            image=image,
            anomaly_map=upsampled,
            save_path=None,  # we'll handle saving manually
            return_image=True
        )

        # Create side-by-side output
        combined = Image.new("RGB", (image.width * 2, image.height))
        combined.paste(image, (0, 0))              # left
        combined.paste(heatmap_image, (image.width, 0))  # right

        # Save final combined image
        combined.save(save_dir / f"{idx:03d}_{image_path.stem}.png")


def main() -> None:
    args = parse_args()
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running PatchCore on class '{args.category}' with device {device}")

    train_dataset = MVTecDataset(
        root=args.data_root,
        category=args.category,
        split="train",
        resize=args.resize,
        crop=args.crop,
        augment=False,
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

    feature_extractor = FeatureExtractor(patch_size=args.patch_size).to(device)
    memory_bank = MemoryBank(device="cpu")
    coreset_sampler = CoresetSampler(ratio=args.coreset_ratio, min_samples=args.coreset_min)
    scorer = NearestNeighborScorer(k=args.k)

    print("1-5. Building memory bank with coreset subsampling...")
    build_memory_bank(feature_extractor, train_loader, memory_bank, device, coreset_sampler)
    print(f"Memory bank size after coreset: {len(memory_bank)}")

    print("6-9. Evaluating on test set and generating heatmaps...")
    results = evaluate(feature_extractor, test_loader, memory_bank, scorer, device)

    image_auc, pixel_auc = compute_metrics(results)
    print(f"10. Image-level ROC-AUC: {image_auc:.4f}")
    print(f"10. Pixel-level ROC-AUC: {pixel_auc:.4f}")

    if args.save_dir:
        save_dir = Path(args.save_dir) / args.category
        save_heatmaps(results, args.crop, save_dir, args.visualizations)
        print(f"Saved heatmaps to {save_dir}")


if __name__ == "__main__":
    main()
