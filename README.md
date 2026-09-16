# AI-solar-filament-segmentation
AI computer vision pipeline for automated solar filament segmentation. Uses deep learning to train and evaluate segmentation models on GONG H-alpha observations, with optimized data processing, model architecture, inference, and post-processing designed to produce accurate filament masks while accounting for morphology and instance separation.

Built for the Solar Filament Segmentation Challenge 2026 (MAGFiLO dataset), which scores
submissions on Panoptic Quality with penalties for fragmentation, over-merging, and runtime.

## How it works

| File | Role |
|---|---|
| `data.py` | COCO-style MAGFiLO loader. Rasterizes polygons/spines once and caches them, splits train/val by underlying observation, filament-biased random tiling, augmentation. |
| `model.py` | `FilamentNet`: timm encoder + U-Net decoder with three heads (semantic, spine heatmap, instance embedding), plus the combined loss (Dice, focal, clDice, boundary, spine, discriminative push/pull). |
| `postprocess.py` | Turns predictions into instances: threshold, gap-bridge, label blobs, then split each blob by spine seeds clustered in embedding space. Encodes masks to COCO RLE. |
| `pipeline.py` | CLI: `train`, `infer`, `eval`, `sweep`. |

## Setup

```bash
python -m venv venv
venv/Scripts/activate        # Windows; use `source venv/bin/activate` elsewhere
# Install a CUDA build of torch first (see pytorch.org), otherwise training runs on CPU
pip install -r requirements-dev.txt
```

Place the competition data so the layout is:

```
train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
train/train_images/*.jpg
test/test_images/*.jpg
```

## Usage

```bash
python pipeline.py train --data_root . --out_dir runs/exp --tile_size 640
python pipeline.py sweep --data_root . --checkpoint runs/exp/best.pt
python pipeline.py infer --data_root . --checkpoint runs/exp/best.pt --sem_thresh 0.5 --spine_thresh 0.4
python pipeline.py eval  --pred_csv submission.csv --gt_json val_annotations.json
```

`sweep` tunes the post-processing thresholds on the held-out validation fold; pass the best values to
`infer`. Inference runs on the full 2048x2048 image by default (`--tile_size 0`); set `--tile_size 896`
if it runs out of GPU memory.

## Tests

```bash
pytest -q
ruff check .
```

The test suite uses a small synthetic dataset and a tiny backbone, so it runs on CPU in seconds and
needs neither the competition data nor pretrained weights.
