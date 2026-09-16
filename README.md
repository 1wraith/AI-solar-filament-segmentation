AI Solar Filament Segmentation
A deep-learning pipeline that finds solar filaments in GONG H-alpha images of the Sun and outputs a separate mask for each filament.

Built for the Solar Filament Segmentation Challenge 2026 (MAGFiLO dataset). The challenge scores submissions on:

Panoptic Quality (PQ): how well each predicted filament matches a real one.
Fragmentation and over-merging penalties: one filament split into several predictions, or several filaments merged into one.
End-to-end runtime.
The design targets each of these: a topology-aware loss keeps thin structures connected, and an instance-embedding head separates filaments that touch.

Contents
How it works
Project layout
Installation
Data
Quick start
Command reference
Submission format
Testing
Troubleshooting
How it works
flowchart LR
    A[H-alpha image<br/>2048x2048] --> B[FilamentNet]
    B --> C[Semantic map<br/>filament vs background]
    B --> D[Spine heatmap<br/>filament centre lines]
    B --> E[Embedding map<br/>which filament is this pixel?]
    C --> F[Threshold +<br/>gap bridging]
    F --> G[Connected blobs]
    D --> H[Spine seeds]
    G --> I[Per-blob split]
    H --> I
    E --> I
    I --> J[Instance masks<br/>COCO RLE]
Model: FilamentNet
A U-Net with a pretrained timm encoder (default convnext_small) and three output heads:

Head	Output	Purpose
Semantic	1 channel	Is this pixel part of a filament?
Spine	1 channel	Heatmap of each filament's centre line, used to seed instances
Embedding	8 channels	Pixels of the same filament get similar vectors, so touching filaments can be separated
The encoder is swappable through --backbone. hrnet_w32 is a good candidate for thin structures because it keeps a high-resolution stream throughout the network.

Loss
The training loss sums several terms. Each one targets a specific way the predictions can go wrong:

Term	What it fixes
Dice + focal	Overall overlap between prediction and ground truth, and the heavy imbalance (filaments cover very few pixels)
clDice	Broken filaments: scores overlap along the skeleton so thin structures stay connected (Shit et al., CVPR 2021)
Boundary	Sloppy edges: extra weight near mask boundaries, where fine barbs are
Spine	Accuracy of the centre-line heatmap
Pull / push	Instance separation: discriminative embedding loss (De Brabandere et al., 2017)
By default the weight of each term is learned during training (uncertainty weighting, Kendall et al., 2018), clamped to a safe range. Pass --loss_weighting static to use fixed weights instead.

Post-processing
Threshold the semantic map, then close small gaps with a morphological closing so a filament isn't broken into pieces.
Label each connected blob. Every later step runs per blob, so pixels are never assigned to a filament elsewhere on the disk.
Skeletonize the spine heatmap into seeds.
Within a blob, merge seeds whose embeddings are close. This stops a filament with a broken spine being split in two.
A blob with zero or one seed becomes one instance. A blob with several seeds is split by assigning each pixel to the nearest seed in embedding space.
Drop instances smaller than --min_area, then encode each mask as COCO RLE.
Training details
Validation split: images are split by the underlying observation, so the same image annotated twice by different annotators can't land in both train and val.
Tiling: random tiles are sampled from each full-resolution image (--samples_per_image per image per epoch), biased toward tiles that contain filaments.
Augmentation:
flips and 90° rotations (filaments have no preferred orientation);
brightness/contrast changes and Gaussian noise;
a mild elastic distortion.
Learning rates: the pretrained backbone learns at a lower rate than the new decoder and heads, with linear warmup followed by cosine decay.
Mixed precision is used on GPU.
Checkpoints and stopping: checkpoints are written atomically, so a crash or power loss can't leave a half-written file. Training stops early when validation Dice stops improving.
Target caching: rasterized training targets are cached on disk (train/_raster_cache/) and in memory, so each image is rasterized only once.
Project layout
.
├── data.py            # MAGFiLO loader, rasterization + caching, augmentation, fold split
├── model.py           # FilamentNet and all loss functions
├── postprocess.py     # predictions -> instance masks -> RLE
├── pipeline.py        # CLI: train / infer / eval / sweep
├── tests/             # pytest suite (synthetic data, CPU-only)
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml     # pytest + ruff configuration
Installation
Requires Python 3.10+ (developed on 3.12).

git clone https://github.com/1wraith/AI-solar-filament-segmentation.git
cd AI-solar-filament-segmentation
python -m venv venv
Activate the environment:

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
Install PyTorch with GPU support first. The default pip install torch on Windows installs a CPU-only build, and the pipeline silently falls back to the CPU when no GPU is found, which makes training extremely slow. Pick the right command for your CUDA version from pytorch.org, for example:

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
Then install the remaining dependencies:

pip install -r requirements-dev.txt
Check that the GPU is visible (this should print True):

python -c "import torch; print(torch.cuda.is_available())"
Data
Download the competition data and arrange it like this. --data_root should point to the folder that contains train/ and test/:

<data_root>/
├── train/
│   ├── MAGFiLO_1.0_Annotations_kaggle2026_train.json
│   └── train_images/*.jpg
└── test/
    └── test_images/*.jpg
The data, checkpoints (runs/) and generated CSVs are excluded by .gitignore and are not part of this repository.

Quick start
# 1. Train (checkpoints go to runs/exp/best.pt and runs/exp/last.pt)
python pipeline.py train --data_root . --out_dir runs/exp

# 2. Tune post-processing thresholds on the held-out validation fold
python pipeline.py sweep --data_root . --checkpoint runs/exp/best.pt

# 3. Predict the test set using the best thresholds from the sweep
python pipeline.py infer --data_root . --checkpoint runs/exp/best.pt \
    --sem_thresh 0.5 --spine_thresh 0.4 --out_csv submission.csv
On an 8 GB GPU (e.g. RTX 3070), use smaller tiles and batches:

python pipeline.py train --data_root . --out_dir runs/exp --tile_size 640 --batch_size 4
Command reference
Run python pipeline.py <command> --help for the full list.

train
Option	Default	Notes
--data_root	required	Folder containing train/
--out_dir	runs/exp	Where best.pt and last.pt are written
--backbone	convnext_small	Any timm model supporting features_only
--tile_size	768	Training crop size
--batch_size	8	Lower this if you run out of GPU memory
--epochs	250	Upper bound; early stopping usually ends sooner
--patience	25	Epochs without val Dice improvement before stopping
--samples_per_image	6	Random tiles per image per epoch
--lr	3e-4	Decoder/head learning rate
--lr_backbone_mult	0.1	Backbone LR = lr × this
--warmup_epochs	3	Linear warmup before cosine decay
--loss_weighting	auto	auto (learned) or static
--cldice_iters	10	Lower (4–6) to speed up training
--embedding_dim	8	Size of the instance embedding
--val_fold	0	Which of 5 folds is held out
--num_workers	2 on Windows, up to 8 elsewhere	Data-loader processes; 0 disables multiprocessing
--mem_cache_size	64	Images cached in RAM per worker (~20–30 MB each)
--seed	42	Random seed
Each epoch prints training loss, validation Dice and the current loss weights. best.pt is chosen by validation Dice.

sweep
Runs the model once per validation image, then tries every combination of thresholds and reports PQ for each. Results are sorted and saved to --out_csv (default sweep_results.csv).

Option	Default
--sem_thresh_list	0.3,0.4,0.5,0.6,0.7
--spine_thresh_list	0.2,0.3,0.4,0.5
--iou_thresh	0.5
--val_fold	taken from the checkpoint
sweep also accepts the shared inference and post-processing options below.

infer
Writes predictions for every image in test/test_images/ to --out_csv (default submission.csv).

Option	Default	Notes
--sem_thresh	0.5	Semantic probability threshold
--spine_thresh	0.4	Spine heatmap threshold
--tta	flip	Test-time augmentation: comma list of flip, rot90 (empty for none)
Shared options (infer and sweep)
Option	Default	Notes
--tile_size	0	0 = whole image in one pass (faster, more consistent embeddings). Use e.g. 896 if you run out of GPU memory
--overlap	128	Tile overlap when tiling
--min_area	25	Drop instances smaller than this (pixels)
--gap_kernel	5	Size of the gap-closing step; 1 disables it
--merge_dist	1.0	Merge seeds whose embeddings are closer than this. Raise it if filaments get split; lower it if neighbours get merged
Unreadable images are skipped and listed at the end. The CSV is always written, even if the run is interrupted.

eval
Scores a submission CSV against a COCO-format ground-truth file:

python pipeline.py eval --pred_csv submission.csv --gt_json val_annotations.json
It reports:

TP / FP / FN counts;
Panoptic Quality;
mean Dice over matched filaments;
how many images had at least one missed filament and how many had at least one extra prediction.
A match needs IoU above --iou_thresh (default 0.5). Masks are decoded at 2048×2048. For validation scoring, sweep is usually simpler, since it needs no separate ground-truth file.

Submission format
One row per predicted filament:

Column	Content
filament_id	<image_id>_<n>, where image_id is the test file name without its extension and n counts from 1
segmentation_rle	COCO RLE counts string for a 2048×2048 mask
filament_id,segmentation_rle
<image_id>_1,<rle counts>
<image_id>_2,<rle counts>
Images with no detected filaments produce no rows. Check the competition's sample submission to confirm this is accepted.

Testing
pytest -q        # 53 tests
ruff check .     # lint
The tests build a small synthetic dataset and use a tiny backbone with random weights, so they run on CPU in about 10 seconds. They need neither the competition data nor any downloads. They cover:

Data: rasterization, the fold split, caching, augmentation and skipping corrupt images.
Model: forward-pass shapes, including the default backbone, and every loss term.
Post-processing:
a filament with a broken spine stays one instance;
far-apart filaments stay separate;
touching filaments are split;
filaments with no seed are kept whole.
Scoring: PQ and matching.
End to end: train, infer, sweep and eval all run to completion.
Troubleshooting
Symptom	Cause / fix
Training is extremely slow	Probably running on CPU. Check torch.cuda.is_available() and install a CUDA build of PyTorch
CUDA out of memory during training	Lower --batch_size or --tile_size
CUDA out of memory during inference	Use --tile_size 896
Windows machine runs out of RAM or crashes	Lower --num_workers (try 0) and/or --mem_cache_size
WARNING: ...% of the image was flagged as foreground	The checkpoint is undertrained or poorly calibrated. Train longer or raise --sem_thresh
skipping unreadable image	Truncated or corrupt JPEG on disk. Re-download the affected files
One filament predicted as several pieces	Raise --merge_dist or --gap_kernel, or lower --spine_thresh
Neighbouring filaments merged into one	Lower --merge_dist or --gap_kernel
Learned loss weights hit the clamp limits	Use --loss_weighting static
