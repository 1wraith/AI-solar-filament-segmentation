"""
Single CLI entrypoint for the full workflow:

    python pipeline.py train  --data_root ... --out_dir runs/exp
    python pipeline.py infer  --data_root ... --checkpoint runs/exp/best.pt --out_csv submission.csv
    python pipeline.py eval   --pred_csv submission.csv --gt_json val_annotations.json
    python pipeline.py sweep  --data_root ... --checkpoint runs/exp/best.pt

Kept as one file with subcommands rather than three scripts since they share
model loading, device setup, and the loss/metric definitions -- splitting
them made it easy for the checkpoint format or embedding_dim handling to
drift out of sync between train.py and infer.py.
"""
import argparse
import glob
import json
import math
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils
from tqdm import tqdm

from data import MagfiloDataset, collate_fn, worker_init_fn
from model import CombinedLoss, build_model
from postprocess import build_submission_rows, instances_from_predictions, rle_counts_to_mask

DEFAULT_BACKBONE = "convnext_small"


def set_seed(seed):
    """Seeds Python's random, numpy, and torch (CPU + all CUDA devices) for
    reproducibility. Full determinism with num_workers > 0 additionally
    needs data.worker_init_fn passed to the DataLoader, which cmd_train does."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint_atomic(ckpt, path):
    """torch.save() is not atomic -- if the process is killed (power loss,
    forced shutdown, OOM kill) mid-write, the target file can be left
    truncated/corrupted while still existing on disk, silently looking like
    a valid checkpoint until you try to load it. Write to a temp file in the
    same directory first, then os.replace() into place -- atomic on both
    Windows and POSIX, so a reader only ever sees either the old complete
    file or the new complete file, never a partial one. This is the same
    pattern already used for the raster cache in data.py, applied here after
    an actual overnight power-loss incident made the gap concrete rather
    than theoretical."""
    tmp_path = path + f".{os.getpid()}.tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)


def default_num_workers():
    """On Windows, multiprocessing uses `spawn`, which re-imports the whole
    script and rebuilds the dataset from scratch in every worker process --
    a real RAM multiplier that a default of 8 workers can push past what a
    machine has available (this is what caused a full system crash during
    early testing of this pipeline on Windows). Linux/Mac use `fork`, which
    is far cheaper, so a higher default is fine there. Either way this is
    just a default -- pass --num_workers explicitly to override."""
    if os.name == "nt":
        return 2
    return min(8, os.cpu_count() or 1)


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, scaler, train):
    model.train(train)
    running, n = {}, 0
    for batch in tqdm(loader, desc="train" if train else "val"):
        image = batch["image"].to(device)
        target = {k: batch[k].to(device) for k in ["semantic", "spine", "instance"]}

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", enabled=scaler is not None):
                out = model(image)
                loss, logs = criterion(out, target)

            if train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        for k, v in logs.items():
            running[k] = running.get(k, 0.0) + v
        n += 1
    return {k: v / n for k, v in running.items()}


def build_optimizer_and_scheduler(model, criterion, args):
    """Discriminative learning rates: the ImageNet-pretrained backbone gets
    a lower LR than the freshly-initialized decoder/heads. A single global
    LR applied equally to both is a common source of underperformance when
    fine-tuning a pretrained encoder on a small dataset (~700 training
    images here) -- the pretrained features can drift away faster than
    useful, or the fresh heads end up under-trained within the epoch/patience
    budget if the shared LR is tuned conservatively for the backbone's sake.

    Followed by linear warmup (protects the pretrained backbone from a
    jarring first few steps at full LR) into cosine decay, applied uniformly
    as a multiplier on top of each param group's own base LR.
    """
    backbone_params = list(model.encoder.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    param_groups = [
        {"params": backbone_params, "lr": args.lr * args.lr_backbone_mult, "name": "backbone"},
        {"params": other_params, "lr": args.lr, "name": "heads"},
    ]
    if args.loss_weighting == "auto":
        # The learnable per-term loss weights (see CombinedLoss) are
        # optimized too -- they're not part of `model`, so without this
        # they'd silently never update and auto_weight would do nothing.
        param_groups.append({"params": list(criterion.parameters()), "lr": args.lr, "name": "loss_weights"})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    warmup_epochs = min(args.warmup_epochs, max(1, args.epochs - 1))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def cmd_train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    train_ds = MagfiloDataset(args.data_root, split="train", tile_size=args.tile_size,
                                val_fold=args.val_fold, samples_per_image=args.samples_per_image,
                                mem_cache_size=args.mem_cache_size)
    val_ds = MagfiloDataset(args.data_root, split="val", tile_size=args.tile_size,
                              val_fold=args.val_fold, mem_cache_size=args.mem_cache_size)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_fn, drop_last=True,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None)

    print(f"train tiles/epoch: {len(train_ds)} ({len(train_ds) // args.batch_size} steps) "
          f"| val tiles: {len(val_ds)} | num_workers: {args.num_workers} | seed: {args.seed}")

    model = build_model(backbone=args.backbone, embedding_dim=args.embedding_dim).to(device)
    criterion = CombinedLoss(cldice_iters=args.cldice_iters,
                              auto_weight=(args.loss_weighting == "auto")).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, criterion, args)
    print(f"backbone lr: {args.lr * args.lr_backbone_mult:.2e} | heads lr: {args.lr:.2e} "
          f"| warmup epochs: {min(args.warmup_epochs, max(1, args.epochs - 1))} "
          f"| loss weighting: {args.loss_weighting}")

    scaler = None
    if device == "cuda":
        # torch.cuda.amp.GradScaler is the older API and is deprecated in
        # newer torch releases in favor of torch.amp.GradScaler("cuda").
        # Try the modern form first, fall back to the old one so this still
        # runs on whatever torch version is actually installed rather than
        # hard-failing on an API that may not exist yet/anymore.
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()

    # Checkpoint on val Dice specifically, not the blended total loss --
    # `total` mixes heterogeneous-scale terms (embedding pull/push especially)
    # that can drift independently of actual segmentation quality. Dice loss
    # (1 - dice score) is the most directly interpretable proxy available
    # from the per-epoch logs without paying for full post-processing.
    best_val_dice_loss = float("inf")
    epochs_since_improve = 0
    for epoch in range(args.epochs):
        train_logs = run_epoch(model, train_loader, criterion, optimizer, device, scaler, True)
        val_logs = run_epoch(model, val_loader, criterion, optimizer, device, scaler, False)
        scheduler.step()
        val_dice_score = 1 - val_logs["dice"]

        weight_str = " ".join(f"{n}={val_logs[f'weight_{n}']:.2f}" for n in CombinedLoss.LOSS_NAMES)
        print(f"[epoch {epoch}] train_total={train_logs['total']:.4f} "
              f"val_dice={val_dice_score:.4f} val_cldice_loss={val_logs['cldice']:.4f} "
              f"| weights: {weight_str}")

        # vars(args) includes `func`, argparse's internal reference to the
        # cmd_train function itself (set by set_defaults(func=cmd_train) in
        # build_parser()) -- a live code object, not data. Saving it caused
        # torch.load's secure-by-default weights_only=True to correctly
        # refuse to unpickle the checkpoint later. Strip it; everything else
        # in args is plain serializable data.
        save_args = {k: v for k, v in vars(args).items() if k != "func"}
        ckpt = {"model": model.state_dict(), "criterion": criterion.state_dict(), "args": save_args,
                "epoch": epoch, "val_dice": val_dice_score, "val_cldice_loss": val_logs["cldice"]}
        save_checkpoint_atomic(ckpt, os.path.join(args.out_dir, "last.pt"))

        if val_logs["dice"] < best_val_dice_loss:
            best_val_dice_loss = val_logs["dice"]
            epochs_since_improve = 0
            save_checkpoint_atomic(ckpt, os.path.join(args.out_dir, "best.pt"))
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= args.patience:
            print(f"no val Dice improvement in {args.patience} epochs, stopping early "
                  f"(best val dice = {1 - best_val_dice_loss:.4f})")
            break


# --------------------------------------------------------------------------
# infer
# --------------------------------------------------------------------------

def _tile_coords(size, tile, overlap):
    stride = tile - overlap
    coords = list(range(0, size - tile + 1, stride))
    if coords[-1] != size - tile:
        coords.append(size - tile)
    return coords


MODEL_STRIDE = 32  # convnext/hrnet features_only deepest stride


def _predict_tta(model, crop, device, tta_ops):
    """One (H, W) normalized crop -> averaged (sem, spine, emb) numpy arrays.
    Pads to a multiple of MODEL_STRIDE so any input size works, then crops
    the padding back off."""
    h, w = crop.shape
    ph, pw = (-h) % MODEL_STRIDE, (-w) % MODEL_STRIDE
    if ph or pw:
        crop = np.pad(crop, ((0, ph), (0, pw)), mode="reflect")
    crop_t = torch.from_numpy(np.ascontiguousarray(crop))[None, None].float().to(device)

    sem_sum = spine_sum = emb_sum = None
    for op in tta_ops:
        t = crop_t
        if op == "flip":
            t = torch.flip(t, dims=[-1])
        elif op == "rot90":
            t = torch.rot90(t, k=1, dims=[-2, -1])

        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device == "cuda"):
            out = model(t)
        sem, spine, emb = torch.sigmoid(out["semantic"]), torch.sigmoid(out["spine"]), out["embedding"]

        if op == "flip":
            sem, spine, emb = (torch.flip(v, dims=[-1]) for v in (sem, spine, emb))
        elif op == "rot90":
            sem, spine, emb = (torch.rot90(v, k=-1, dims=[-2, -1]) for v in (sem, spine, emb))

        sem, spine, emb = sem.float(), spine.float(), emb.float()
        sem_sum = sem if sem_sum is None else sem_sum + sem
        spine_sum = spine if spine_sum is None else spine_sum + spine
        emb_sum = emb if emb_sum is None else emb_sum + emb

    n = len(tta_ops)
    sem = (sem_sum / n)[0, :h, :w].cpu().numpy()
    spine = (spine_sum / n)[0, :h, :w].cpu().numpy()
    emb = (emb_sum / n)[0, :, :h, :w].cpu().numpy()
    return sem, spine, emb


def _predict_full_image(model, img, device, tile_size, overlap, tta_ops, embedding_dim):
    """tile_size <= 0 (the default) runs the whole image in one forward pass
    per TTA op. That's both faster (2 passes instead of 9 tiles x 2) and more
    correct for the embedding head: embeddings are only meaningful relative
    to other pixels in the SAME forward pass, so averaging embeddings across
    overlapping tiles blends values that were never trained to agree. Only
    fall back to tiling (tile_size > 0) if the full image doesn't fit in GPU
    memory."""
    H, W = img.shape
    norm = (img.astype(np.float32) / 255.0 - 0.5) / 0.25

    if tile_size <= 0 or (tile_size >= H and tile_size >= W):
        return _predict_tta(model, norm, device, tta_ops)

    sem_acc = np.zeros((H, W), dtype=np.float32)
    spine_acc = np.zeros((H, W), dtype=np.float32)
    emb_acc = np.zeros((embedding_dim, H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    for y in _tile_coords(H, tile_size, overlap):
        for x in _tile_coords(W, tile_size, overlap):
            crop = norm[y:y + tile_size, x:x + tile_size]
            sem, spine, emb = _predict_tta(model, crop, device, tta_ops)

            sem_acc[y:y + tile_size, x:x + tile_size] += sem
            spine_acc[y:y + tile_size, x:x + tile_size] += spine
            emb_acc[:, y:y + tile_size, x:x + tile_size] += emb
            count[y:y + tile_size, x:x + tile_size] += 1

    count = np.maximum(count, 1)
    return sem_acc / count, spine_acc / count, emb_acc / count[None]


def _save_submission(rows, out_csv):
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(out_csv, index=False)
    print(f"wrote {len(df)} rows to {out_csv}")


def cmd_infer(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tta_ops = ["identity"] + [o for o in args.tta.split(",") if o]

    # weights_only=False: we only ever load checkpoints this pipeline
    # generated itself, so the security tradeoff (protection against a
    # malicious checkpoint executing arbitrary code) doesn't apply here --
    # and older checkpoints saved before the `func`-stripping fix above
    # still need this to load at all.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt.get("args", {})
    embedding_dim = model_args.get("embedding_dim", 8)
    backbone = model_args.get("backbone", DEFAULT_BACKBONE)

    # Older checkpoints (saved before this field existed) won't have these --
    # report what's available rather than failing, since knowing "unknown"
    # is still more informative than silently saying nothing.
    epoch = ckpt.get("epoch", "unknown")
    val_dice = ckpt.get("val_dice", "unknown")
    print(f"loaded checkpoint from epoch {epoch}, val_dice={val_dice} "
          f"(low val_dice, e.g. well under ~0.1-0.2, usually means the model hasn't "
          f"learned to predict much positive area yet -- predictions may look sparse "
          f"or empty at default thresholds)")

    model = build_model(backbone=backbone, embedding_dim=embedding_dim, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_dir = os.path.join(args.data_root, "test", "test_images")
    image_paths = sorted(glob.glob(os.path.join(test_dir, "*.jpeg"))
                         + glob.glob(os.path.join(test_dir, "*.jpg")))

    all_rows = []
    skipped = []
    try:
        for path in tqdm(image_paths, desc="infer"):
            image_id = os.path.splitext(os.path.basename(path))[0]

            # Same corruption risk as the training images -- a truncated
            # test jpeg must not be allowed to kill the whole inference
            # pass after everything computed before it. Skip and keep going;
            # report what was skipped so it's visible rather than silent.
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[infer] unreadable image, skipping: {path}")
                skipped.append(path)
                continue

            try:
                sem_prob, spine_prob, emb = _predict_full_image(
                    model, img, device, args.tile_size, args.overlap, tta_ops, embedding_dim)
                instances = instances_from_predictions(
                    sem_prob, spine_prob, emb, sem_thresh=args.sem_thresh,
                    spine_thresh=args.spine_thresh, min_area=args.min_area,
                    gap_kernel=args.gap_kernel, merge_dist=args.merge_dist)
                all_rows.extend(build_submission_rows(image_id, instances))
            except Exception as e:
                print(f"[infer] error processing {image_id}, skipping: {e}")
                skipped.append(path)
                continue
    finally:
        # Always write out whatever was successfully predicted so far, even
        # if the loop above was interrupted (Ctrl+C, an unexpected crash) --
        # losing every prediction because of one bad image or an interrupt
        # partway through a long inference pass is worse than a partial CSV.
        _save_submission(all_rows, args.out_csv)

    if skipped:
        print(f"[infer] {len(skipped)} / {len(image_paths)} test images were skipped:")
        for p in skipped[:30]:
            print(f"  {p}")
        if len(skipped) > 30:
            print(f"  ... and {len(skipped) - 30} more")


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------

def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0.0


def _dice(a, b):
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return 2 * inter / denom if denom > 0 else 1.0


def _match_and_score(gt_masks, pred_masks, iou_thresh):
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    if n_gt == 0 and n_pred == 0:
        return dict(tp=0, fp=0, fn=0, iou_sum=0.0, dice_scores=[])

    iou_mat = np.zeros((n_gt, n_pred))
    for i, g in enumerate(gt_masks):
        for j, p in enumerate(pred_masks):
            iou_mat[i, j] = _iou(g, p)

    matched_gt, matched_pred = set(), set()
    tp, iou_sum, dice_scores = 0, 0.0, []
    while iou_mat.size:
        i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        best = iou_mat[i, j]
        if best <= iou_thresh:
            break
        matched_gt.add(i); matched_pred.add(j)
        tp += 1
        iou_sum += best
        dice_scores.append(_dice(gt_masks[i], pred_masks[j]))
        iou_mat[i, :] = -1
        iou_mat[:, j] = -1

    return dict(tp=tp, fp=n_pred - len(matched_pred), fn=n_gt - len(matched_gt),
                iou_sum=iou_sum, dice_scores=dice_scores)


def _compute_pq_metrics(gt_by_image, pred_by_image, iou_thresh):
    """Shared by cmd_eval and cmd_sweep so the two never compute PQ
    differently by accident."""
    total_tp = total_fp = total_fn = 0
    total_iou_sum = 0.0
    all_dice = []
    one_to_many = many_to_one = 0

    for image_id in set(gt_by_image) | set(pred_by_image):
        res = _match_and_score(gt_by_image.get(image_id, []), pred_by_image.get(image_id, []), iou_thresh)
        total_tp += res["tp"]; total_fp += res["fp"]; total_fn += res["fn"]
        total_iou_sum += res["iou_sum"]
        all_dice.extend(res["dice_scores"])
        if res["fn"] > 0 and res["tp"] > 0:
            one_to_many += 1
        if res["fp"] > 0 and res["tp"] > 0:
            many_to_one += 1

    pq = total_iou_sum / (total_tp + 0.5 * total_fp + 0.5 * total_fn + 1e-9)
    mean_dice = float(np.mean(all_dice)) if all_dice else 0.0
    return dict(pq=pq, mean_dice=mean_dice, tp=total_tp, fp=total_fp, fn=total_fn,
                one_to_many=one_to_many, many_to_one=many_to_one)


def cmd_eval(args):
    with open(args.gt_json) as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    gt_by_image = defaultdict(list)
    for a in coco["annotations"]:
        im = images[a["image_id"]]
        rle = mask_utils.frPyObjects(a["segmentation"], im["height"], im["width"])
        gt_by_image[a["image_id"]].append(mask_utils.decode(mask_utils.merge(rle)).astype(bool))

    df = pd.read_csv(args.pred_csv)
    pred_by_image = defaultdict(list)
    for _, row in df.iterrows():
        image_id = row["filament_id"].rsplit("_", 1)[0]
        pred_by_image[image_id].append(rle_counts_to_mask(row["segmentation_rle"]))

    m = _compute_pq_metrics(gt_by_image, pred_by_image, args.iou_thresh)

    print(f"Images evaluated:     {len(set(gt_by_image) | set(pred_by_image))}")
    print(f"TP / FP / FN:         {m['tp']} / {m['fp']} / {m['fn']}")
    print(f"Panoptic Quality:     {m['pq']:.4f}")
    print(f"Mean Dice (matched):  {m['mean_dice']:.4f}")
    print(f"Images w/ apparent fragmentation (1:many): {m['one_to_many']}")
    print(f"Images w/ apparent over-merging (many:1):  {m['many_to_one']}")


# --------------------------------------------------------------------------
# sweep -- threshold tuning against the held-out val split
# --------------------------------------------------------------------------

def cmd_sweep(args):
    """Runs inference ONCE per val image (sem/spine/embedding prediction is
    threshold-independent) then cheaply re-runs post-processing across a
    grid of sem_thresh x spine_thresh combinations, reporting PQ for each.

    This is the tool for the post-processing tuning step that can't be done
    before a real checkpoint exists -- point it at tomorrow's checkpoint and
    it turns "guess at thresholds" into "look at a table sorted by PQ."
    Evaluates against the same held-out val fold training used (never seen
    during training), not the actual competition test set.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # See the matching comment in cmd_infer -- always loading our own
    # self-generated checkpoints, so weights_only=False is safe here and
    # keeps older checkpoints (saved before the args-stripping fix) loadable.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt.get("args", {})
    embedding_dim = model_args.get("embedding_dim", 8)
    backbone = model_args.get("backbone", DEFAULT_BACKBONE)
    val_fold = model_args.get("val_fold", args.val_fold)

    epoch = ckpt.get("epoch", "unknown")
    val_dice = ckpt.get("val_dice", "unknown")
    print(f"loaded checkpoint from epoch {epoch}, val_dice={val_dice}")

    model = build_model(backbone=backbone, embedding_dim=embedding_dim, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_ds = MagfiloDataset(args.data_root, split="val", val_fold=val_fold, mem_cache_size=0)
    print(f"sweeping against {len(val_ds.samples)} held-out val images "
          f"(val_fold={val_fold}, matching training's split)")

    # Predict once per image -- sem_prob/spine_prob/embedding don't depend
    # on sem_thresh/spine_thresh, only the post-processing step does.
    cached_preds = []
    for im_info, anns in tqdm(val_ds.samples, desc="predicting"):
        img_path = os.path.join(args.data_root, "train", "train_images", im_info["file_name"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[sweep] unreadable image, skipping: {im_info['file_name']}")
            continue

        sem_prob, spine_prob, emb = _predict_full_image(
            model, img, device, args.tile_size, args.overlap, ["identity"], embedding_dim)

        gt_masks = []
        h, w = im_info["height"], im_info["width"]
        for a in anns:
            rle = mask_utils.frPyObjects(a["segmentation"], h, w)
            gt_masks.append(mask_utils.decode(mask_utils.merge(rle)).astype(bool))

        cached_preds.append((sem_prob, spine_prob, emb, gt_masks))

    sem_threshes = [float(x) for x in args.sem_thresh_list.split(",")]
    spine_threshes = [float(x) for x in args.spine_thresh_list.split(",")]

    results = []
    for sem_thresh in sem_threshes:
        for spine_thresh in spine_threshes:
            gt_by_image, pred_by_image = {}, {}
            for i, (sem_prob, spine_prob, emb, gt_masks) in enumerate(cached_preds):
                instances = instances_from_predictions(
                    sem_prob, spine_prob, emb, sem_thresh=sem_thresh,
                    spine_thresh=spine_thresh, min_area=args.min_area,
                    gap_kernel=args.gap_kernel, merge_dist=args.merge_dist)
                gt_by_image[i] = gt_masks
                pred_by_image[i] = instances

            m = _compute_pq_metrics(gt_by_image, pred_by_image, args.iou_thresh)
            results.append((sem_thresh, spine_thresh, m))
            print(f"  sem_thresh={sem_thresh:.2f} spine_thresh={spine_thresh:.2f} "
                  f"-> PQ={m['pq']:.4f} dice={m['mean_dice']:.4f} "
                  f"1:many={m['one_to_many']} many:1={m['many_to_one']}")

    results.sort(key=lambda r: r[2]["pq"], reverse=True)
    print("\nTop 5 by PQ:")
    for sem_thresh, spine_thresh, m in results[:5]:
        print(f"  sem_thresh={sem_thresh:.2f} spine_thresh={spine_thresh:.2f} -> PQ={m['pq']:.4f}")

    rows = [{"sem_thresh": s, "spine_thresh": sp, **m} for s, sp, m in results]
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\nfull results written to {args.out_csv}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

TILE_SIZE_HELP = ("0 (default) = predict the whole 2048x2048 image in one pass -- faster, and "
                  "keeps embeddings consistent across the image. Set e.g. 896 to tile "
                  "instead, only if the full image runs out of GPU memory.")


def add_postprocess_args(parser):
    """Post-processing knobs shared by infer and sweep, so a value tuned in
    sweep can be passed unchanged to infer."""
    parser.add_argument("--min_area", type=int, default=25)
    parser.add_argument("--gap_kernel", type=int, default=5,
                        help="Morphological closing kernel used to bridge small gaps "
                             "(<=1 disables).")
    parser.add_argument("--merge_dist", type=float, default=1.0,
                        help="Seeds in the same blob whose mean embeddings are closer than "
                             "this are merged into one filament (default = 2 * delta_pull). "
                             "Raise if filaments are being split; lower if touching "
                             "filaments are being merged.")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data_root", required=True)
    t.add_argument("--backbone", default=DEFAULT_BACKBONE,
                    help="convnext_small is the recommended default for ~880 images -- "
                         "convnext_base has meaningfully higher overfitting risk on a dataset "
                         "this size. Try hrnet_w32 once a convnext_small baseline is solid.")
    t.add_argument("--tile_size", type=int, default=768)
    t.add_argument("--batch_size", type=int, default=8,
                    help="8 fits comfortably in 16GB at tile_size=768; try 12 on 24GB.")
    t.add_argument("--epochs", type=int, default=250,
                    help="Upper bound only -- early stopping (--patience) will almost "
                         "certainly stop well before this on a dataset this size.")
    t.add_argument("--patience", type=int, default=25,
                    help="Stop after this many epochs with no val Dice improvement.")
    t.add_argument("--samples_per_image", type=int, default=6,
                    help="Random tile crops drawn per training image per epoch. >1 "
                         "meaningfully increases gradient steps/epoch on a small dataset "
                         "since each 2048x2048 image contains many distinct possible crops.")
    t.add_argument("--cldice_iters", type=int, default=10,
                    help="Soft-skeletonization iterations inside the clDice loss. This is "
                         "the single most expensive term per training step -- lower it "
                         "(e.g. 4-6) if training speed is a bottleneck on local hardware, "
                         "at some cost to how precisely it captures thin/branching topology.")
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--val_fold", type=int, default=0)
    t.add_argument("--embedding_dim", type=int, default=8)
    t.add_argument("--out_dir", default="runs/exp")
    t.add_argument("--num_workers", type=int, default=None,
                    help="Default is OS-aware (2 on Windows, up to 8 on Linux/Mac) -- see "
                         "default_num_workers(). Windows multiprocessing re-imports the "
                         "whole script per worker (no fork), which can exhaust system RAM "
                         "at high worker counts; pass 0 to disable multiprocessing entirely "
                         "if you hit instability.")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--mem_cache_size", type=int, default=64,
                    help="Images' rasterized targets kept in RAM per worker process (0 "
                         "disables). ~20-30MB each at 2048x2048, so this budget multiplies "
                         "by --num_workers. Cuts repeated disk decompression when the same "
                         "image is revisited within an epoch (samples_per_image > 1). Lower "
                         "it if you're RAM-constrained.")
    t.add_argument("--lr_backbone_mult", type=float, default=0.1,
                    help="Backbone LR = --lr * this multiplier; decoder/heads use --lr "
                         "directly. Keeps the ImageNet-pretrained encoder from drifting away "
                         "from useful features faster than the freshly-initialized heads "
                         "can catch up, which matters more on a small (~700 image) dataset.")
    t.add_argument("--warmup_epochs", type=int, default=3,
                    help="Linear LR warmup before cosine decay kicks in. Protects the "
                         "pretrained backbone from a jarring first few steps at full LR.")
    t.add_argument("--loss_weighting", choices=["auto", "static"], default="auto",
                    help="'auto' (default) learns a per-term loss weight during training "
                         "(uncertainty-weighted multi-task loss) instead of relying on the "
                         "static w_* defaults in CombinedLoss, which were never empirically "
                         "checked against this data's actual per-term gradient magnitudes. "
                         "Fall back to 'static' if the learned weights look unstable -- "
                         "check the 'weights:' line in the per-epoch log.")
    t.set_defaults(func=cmd_train)

    i = sub.add_parser("infer")
    i.add_argument("--data_root", required=True)
    i.add_argument("--checkpoint", required=True)
    i.add_argument("--out_csv", default="submission.csv")
    i.add_argument("--tile_size", type=int, default=0, help=TILE_SIZE_HELP)
    i.add_argument("--overlap", type=int, default=128)
    i.add_argument("--tta", default="flip", help="comma list: flip,rot90")
    i.add_argument("--sem_thresh", type=float, default=0.5)
    i.add_argument("--spine_thresh", type=float, default=0.4,
                    help="Use the best value found by `sweep`.")
    add_postprocess_args(i)
    i.set_defaults(func=cmd_infer)

    e = sub.add_parser("eval")
    e.add_argument("--pred_csv", required=True)
    e.add_argument("--gt_json", required=True)
    e.add_argument("--iou_thresh", type=float, default=0.5)
    e.set_defaults(func=cmd_eval)

    s = sub.add_parser("sweep")
    s.add_argument("--data_root", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--val_fold", type=int, default=0,
                    help="Only used as a fallback if the checkpoint's own args don't "
                         "record val_fold (older checkpoints) -- normally read automatically "
                         "from the checkpoint so the sweep matches what training held out.")
    s.add_argument("--tile_size", type=int, default=0, help=TILE_SIZE_HELP)
    s.add_argument("--overlap", type=int, default=128)
    s.add_argument("--sem_thresh_list", default="0.3,0.4,0.5,0.6,0.7",
                    help="Comma-separated sem_thresh values to try.")
    s.add_argument("--spine_thresh_list", default="0.2,0.3,0.4,0.5",
                    help="Comma-separated spine_thresh values to try.")
    add_postprocess_args(s)
    s.add_argument("--iou_thresh", type=float, default=0.5)
    s.add_argument("--out_csv", default="sweep_results.csv")
    s.set_defaults(func=cmd_sweep)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if getattr(args, "num_workers", None) is None and hasattr(args, "num_workers"):
        args.num_workers = default_num_workers()
    args.func(args)
