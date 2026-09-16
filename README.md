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

Run a sweep against a trained checkpoint:

python pipeline.py sweep \
    --data_root . \
    --checkpoint runs/exp/best.pt


The sweep evaluates different segmentation and instance-processing thresholds on the held-out validation data.

Record the selected parameters with the corresponding experiment rather than relying on undocumented defaults.

Inference

Generate predictions using a trained checkpoint:

python pipeline.py infer \
    --data_root . \
    --checkpoint runs/exp/best.pt \
    --sem_thresh 0.5 \
    --spine_thresh 0.4


By default, inference processes the complete 2048×2048 image.

If GPU memory is insufficient, tiled inference can be enabled:

python pipeline.py infer \
    --data_root . \
    --checkpoint runs/exp/best.pt \
    --tile_size 896


The optimal tile size depends on available GPU memory and should be validated on the target hardware.

Evaluation

Evaluate predictions against ground-truth annotations with:

python pipeline.py eval \
    --pred_csv submission.csv \
    --gt_json val_annotations.json


Evaluation should be performed on data that was not used to train the model.

Because the competition evaluates instance-level segmentation, validation should consider more than semantic pixel accuracy. Fragmentation and over-merging can affect the final score even when the overall foreground segmentation appears visually accurate.

Testing

Run the automated test suite:

pytest -q


Run static analysis:

ruff check .


The tests are designed to run on CPU and use synthetic data and lightweight model configurations. They do not require the competition dataset or trained checkpoints.

Before submitting changes, run both commands and ensure they complete successfully.

Performance Considerations

The project is designed to keep the inference path computationally efficient:

Full-image inference avoids unnecessary overlapping tiles when GPU memory permits.

Tiled inference remains available for constrained hardware.

Post-processing operates on connected regions rather than repeatedly scanning the complete image for every instance.

Dataset preprocessing and annotation rasterization can be cached.

The model uses a shared encoder with task-specific prediction heads.

Performance should be benchmarked on the hardware used for the final competition submission.

Reproducibility

For meaningful experiment comparisons, record:

Model configuration

PyTorch and dependency versions

Dataset version

Training/validation split

Random seed

Training hyperparameters

Checkpoint used

Post-processing parameters

Inference configuration

Hardware and CUDA version

Do not commit datasets, model checkpoints, generated submissions, or other large experiment artifacts unless explicitly required.

Development

Recommended workflow:

pytest -q
ruff check .


Keep changes focused and avoid committing generated files such as:

__pycache__/
.pytest_cache/
*.pyc
runs/
checkpoints/
*.pt
*.pth


These should be excluded through .gitignore.

Project Status

This is an active research and competition project. Model architecture, training strategy, post-processing, and evaluation methodology may continue to change as experiments are conducted.

Results should therefore be interpreted in the context of the specific model checkpoint, dataset split, and configuration used to generate them.

License

Add the project's chosen license here before distributing the repository publicly.

Acknowledgements

This project was developed for the Solar Filament Segmentation Challenge 2026 and uses GONG H-alpha observations and the MAGFiLO dataset provided through the competition.

Please refer to the competition's official documentation for dataset licensing, attribution requirements, evaluation rules, and submission requirements.
