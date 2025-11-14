# patchcore-anomaly-detection
Reproducing the PatchCore anomaly detection algorithm from the paper  "Towards Total Recall in Industrial Anomaly Detection".  Features: WideResNet backbone, patch extraction, memory bank construction,  coreset subsampling (Farthest Point Sampling), nearest-neighbor anomaly scoring,  and ROC-AUC evaluation on MVTec AD.

## Project Structure

- `datasets/` — data loaders (e.g., `mvtec_loader.py`) and related helpers.
- `models/` — model definitions (`patchcore.py` ties all building blocks together).
- `utils/` — reusable utilities for feature and patch extraction, coreset sampling, memory bank management, nearest-neighbor scoring, and visualization.
- `experiments/` — runnable scripts such as `run_patchcore.py` for end-to-end evaluation on MVTec AD.

## Quickstart

```bash
python experiments/run_patchcore.py \
  --data-root /path/to/mvtec \
  --category bottle \
  --batch-size 8 \
  --save-dir outputs/bottle
```
