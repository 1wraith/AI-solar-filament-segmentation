AI Solar Filament Segmentation
A deep-learning computer vision pipeline for automated solar filament segmentation from GONG H-alpha observations.

This project was developed for the Solar Filament Segmentation Challenge 2026, using the MAGFiLO dataset. The pipeline is designed around the competition's emphasis on accurate pixel-level segmentation, instance separation, and computational efficiency.

Overview
The system combines semantic segmentation, filament-spine detection, and instance embeddings to identify individual solar filaments and produce segmentation masks suitable for evaluation.

The pipeline consists of four main stages:

Data preparation — loads MAGFiLO annotations, rasterizes filament polygons and spines, caches processed data, and generates filament-focused training crops.

Model inference — predicts filament probability, filament spine locations, and per-pixel instance embeddings.

Instance post-processing — converts model outputs into individual filament instances using connected components, gap bridging, spine seeds, and embedding-based clustering.

Evaluation and inference — evaluates predictions and provides threshold sweeping for post-processing parameters before generating final predictions.

Repository Structure
File	Responsibility
data.py	Dataset loading, annotation rasterization, caching, train/validation splitting, tiling, and augmentation.
model.py	FilamentNet architecture and training losses. Includes semantic, spine, and instance-embedding prediction heads.
postprocess.py	Converts network predictions into individual filament instances and provides mask encoding utilities.
pipeline.py	Command-line entry point for training, inference, evaluation, and parameter sweeps.
requirements.txt	Runtime Python dependencies.
requirements-dev.txt	Development and testing dependencies.

Key Features
Semantic filament segmentation

Filament spine prediction for instance identification

Per-pixel instance embeddings

TCP-style?

Instance-aware post-processing

Gap bridging for fragmented predictions

Embedding-based instance separation

COCO-compatible RLE mask encoding

Full-image inference for 2048×2048 observations

Optional tiled inference for lower-memory GPUs

Test-time augmentation

Configurable post-processing thresholds

Automated parameter sweeping

CPU-compatible test suite

GPU-accelerated training with CUDA-enabled PyTorch

Installation
Python 3.10+ is recommended.

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\Activate.ps1

On Linux/macOS:

source venv/bin/activate

Install the appropriate PyTorch build for your hardware first. For GPU training, install a CUDA-enabled version compatible with your NVIDIA driver.

Then install the project dependencies:

pip install -r requirements-dev.txt

Verify the installation:

python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"

For GPU training, CUDA: True should be reported.

Dataset
The project expects the competition data to be arranged approximately as follows:

.
├── train/
│   ├── MAGFiLO_1.0_Annotations_kaggle2026_train.json
│   └── train_images/
│       ├── image_001.jpg
│       └── ...
├── test/
│   └── test_images/
│       ├── image_001.jpg
│       └── ...
└── ...

The dataset itself is not included in this repository.

Obtain the competition dataset through the official competition distribution and ensure that its filenames and annotation structure match the expected layout before training.

Training
A typical training run is:

python pipeline.py train \
    --data_root . \
    --out_dir runs/exp \
    --tile_size 640

Training checkpoints and experiment outputs are written to the specified --out_dir.

For reproducible experiments, keep each training run in a separate directory rather than overwriting previous checkpoints.

Parameter Sweep
Post-processing parameters can significantly affect instance-level metrics.
