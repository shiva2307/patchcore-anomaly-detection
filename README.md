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
  --backbone wide_resnet50_2 \
  --batch-size 8 \
  --save-dir outputs/bottle
```

### Minimal API Usage

```python
from torch.utils.data import DataLoader
from datasets.mvtec_loader import MVTecDataset
from models.patchcore import PatchCore

train_loader = DataLoader(MVTecDataset("/path/to/mvtec", "bottle", split="train"), batch_size=8)
test_loader = DataLoader(MVTecDataset("/path/to/mvtec", "bottle", split="test"), batch_size=8)

model = PatchCore(backbone="resnet50")
model.fit(train_loader)
results = model.evaluate(test_loader)
single_prediction = model.predict(test_loader.dataset[0])
```
