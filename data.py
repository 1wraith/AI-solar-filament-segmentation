"""
COCO-style loader for MAGFiLO, with on-disk caching of the rasterized
targets (semantic mask, instance id map, spine heatmap).

Rasterizing polygons + spine lines at 2048x2048 is not free, and the naive
approach of doing it inside __getitem__ means paying that cost on every
single epoch for every single image. Instead we rasterize once per image the
first time it's requested and cache the result as a compressed .npz next to
the annotation file; every subsequent epoch just loads three arrays.

Also handles the val split correctly: split by the UNDERLYING OBSERVATION,
not by annotator batch id, since e.g. 010401-<ts> and 010402-<ts> are the
same image annotated twice and must not straddle train/val.
"""
import collections
import hashlib
import json
import os
import random
import re

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

try:
    import albumentations as A
    _HAS_ALBUMENTATIONS = True
except ImportError:
    _HAS_ALBUMENTATIONS = False

BATCH_PREFIX_RE = re.compile(r"^\d{6}-")


def build_default_augmentation():
    """Default augmentation for small-dataset training (~880 images).

    With this little data, augmentation strength matters more than model
    capacity -- an under-augmented run will memorize the training set long
    before it generalizes. Flips/90-degree rotations are safe (filaments
    have no canonical orientation on the disk). Brightness/contrast and
    Gaussian noise mimic the real variation across GONG stations and
    exposure conditions. ElasticTransform is kept mild since it can distort
    thin barbs into implausible shapes if pushed too hard.
    """
    if not _HAS_ALBUMENTATIONS:
        return None

    base = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    ]

    # GaussNoise / ElasticTransform kwarg names have changed across
    # albumentations versions (e.g. var_limit -> std_range, alpha_affine
    # removed). Try the modern signature, then an older one, then skip
    # rather than let a version mismatch crash an overnight run before it
    # starts.
    try:
        base.append(A.GaussNoise(std_range=(0.02, 0.1), p=0.3))
    except TypeError:
        try:
            base.append(A.GaussNoise(var_limit=(5.0, 25.0), p=0.3))
        except TypeError:
            pass

    # mask_interpolation is pinned to nearest-neighbor explicitly rather than
    # relying on the library default. The `instance` target is integer
    # labels (0 = background, 1..N = filament id) -- any interpolation other
    # than nearest-neighbor would blur label boundaries into fractional,
    # meaningless values and corrupt the embedding loss's targets. Most
    # albumentations versions already default mask_interpolation to
    # cv2.INTER_NEAREST, but that default isn't guaranteed across versions,
    # so it's made explicit here and the transform is skipped entirely
    # (rather than left unpinned) if this signature isn't accepted.
    try:
        base.append(A.ElasticTransform(alpha=15, sigma=5, mask_interpolation=cv2.INTER_NEAREST, p=0.2))
    except TypeError:
        pass

    return A.Compose(base)


def group_key(image_id: str) -> str:
    """Strip the leading 6-digit annotator-batch prefix so duplicate
    annotations of the same observation hash to the same fold."""
    return BATCH_PREFIX_RE.sub("", image_id)


def fold_of(image_id: str, n_folds: int = 5) -> int:
    h = int(hashlib.md5(group_key(image_id).encode("utf-8")).hexdigest(), 16)
    return h % n_folds


def _rasterize(im_info, anns):
    h, w = im_info["height"], im_info["width"]
    semantic = np.zeros((h, w), dtype=np.uint8)
    instance = np.zeros((h, w), dtype=np.int32)
    spine_pts = np.zeros((h, w), dtype=np.uint8)

    for idx, a in enumerate(anns, start=1):
        poly = a["segmentation"]
        rle = mask_utils.frPyObjects(poly, h, w)
        m = mask_utils.decode(mask_utils.merge(rle)).astype(bool)
        semantic |= m.astype(np.uint8)

        # Known limitation: if two filament annotations genuinely overlap in
        # pixel space (rare, but not impossible given independent human
        # annotation), a pixel can only belong to one instance id. First-wins
        # here (only claim currently-unclaimed pixels) so the result is at
        # least deterministic and doesn't let a later annotation silently
        # erase an earlier one's already-assigned pixels; it does NOT
        # resolve the underlying ambiguity of which filament that pixel
        # should belong to for the embedding loss.
        unclaimed = m & (instance == 0)
        instance[unclaimed] = idx

        spine = a.get("spine") or []
        if len(spine) >= 4:
            pts = np.array(spine, dtype=np.int32).reshape(-1, 2)
            for i in range(len(pts) - 1):
                cv2.line(spine_pts, tuple(pts[i]), tuple(pts[i + 1]), 1, thickness=2)

    spine_heat = cv2.GaussianBlur(spine_pts.astype(np.float32), (0, 0), sigmaX=3)
    if spine_heat.max() > 0:
        spine_heat = spine_heat / spine_heat.max()

    return semantic, instance, spine_heat.astype(np.float16)


def _load_or_build_raster(cache_dir, im_info, anns):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{im_info['id']}.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["semantic"], data["instance"], data["spine"].astype(np.float32)

    semantic, instance, spine_heat = _rasterize(im_info, anns)

    # Write atomically: build the file under a unique temp name in the same
    # directory, then os.replace() it into place. With samples_per_image > 1
    # the same underlying image can be requested multiple times within one
    # epoch, and with num_workers > 0 those requests can land on different
    # worker processes concurrently -- two workers racing to
    # np.savez_compressed the SAME path at the same time can corrupt
    # whichever one loses, since the write isn't atomic. os.replace() is
    # atomic on both Windows and POSIX, so whichever worker finishes first
    # "wins" cleanly and no reader ever sees a partially-written file.
    tmp_path = os.path.join(cache_dir, f".{im_info['id']}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp_path, semantic=semantic, instance=instance, spine=spine_heat)
    try:
        os.replace(tmp_path, cache_path)
    except OSError:
        # Another process already finished writing this same file first --
        # that's fine, just clean up our temp file and use the winner's.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return semantic, instance, spine_heat.astype(np.float32)


class _LRUCache:
    """Small fixed-size in-memory LRU cache for rasterized targets, keyed by
    image id. With samples_per_image > 1, the same image is requested
    several times per epoch -- without this, every single one of those
    requests re-decompresses the on-disk .npz cache from scratch, which is
    real, avoidable CPU/I-O cost especially with num_workers=0 (no
    prefetch/overlap with GPU compute). Kept deliberately small and bounded
    by item count rather than caching the whole ~880-image dataset in RAM,
    since an unbounded cache is exactly the kind of thing that contributed
    to an earlier out-of-memory crash on this project."""

    def __init__(self, max_items=64):
        self.max_items = max_items
        self._store = collections.OrderedDict()

    def get(self, key):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key, value):
        if self.max_items <= 0:
            return
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.max_items:
            self._store.popitem(last=False)


class MagfiloDataset(Dataset):
    def __init__(self, data_root, split="train", tile_size=896,
                 val_fold=0, n_folds=5, augment="default", cache_dir=None,
                 samples_per_image=1, mem_cache_size=64):
        """
        augment: an albumentations Compose, None to disable, or "default" to
            use build_default_augmentation(). Only ever applied when
            split == "train" -- val must stay un-augmented to be a fair,
            stable comparison across epochs.
        samples_per_image: with only ~880 images, one random 768/896px crop
            per image per epoch massively under-uses each full-resolution
            image (a 2048x2048 image contains many non-overlapping possible
            crops). This inflates the virtual dataset length so each epoch
            draws `samples_per_image` independent random tiles per image
            instead of one, giving more gradient steps per epoch without
            touching the underlying data.
        mem_cache_size: number of images' rasterized targets to keep in RAM
            per worker process (0 disables). Each cached entry is roughly
            20-30MB uncompressed at 2048x2048, so the default of 64 is
            around 1.5-2GB per worker -- deliberately conservative given an
            earlier RAM-related crash on this project. Raise it if you have
            headroom and want to cut disk I/O further; note this budget
            multiplies by --num_workers, since each worker process keeps its
            own cache.
        """
        self.data_root = data_root
        self.tile_size = tile_size
        self.split = split
        if augment == "default":
            augment = build_default_augmentation() if split == "train" else None
        self.augment = augment
        self.samples_per_image = max(1, samples_per_image) if split == "train" else 1
        self.cache_dir = cache_dir or os.path.join(data_root, "train", "_raster_cache")
        self._mem_cache = _LRUCache(max_items=mem_cache_size)

        ann_path = os.path.join(
            data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
        with open(ann_path) as f:
            coco = json.load(f)

        images_by_id = {im["id"]: im for im in coco["images"]}
        anns_by_image = {}
        for a in coco["annotations"]:
            anns_by_image.setdefault(a["image_id"], []).append(a)

        ids = list(images_by_id.keys())
        keep = (lambda i: fold_of(i, n_folds) != val_fold) if split == "train" \
            else (lambda i: fold_of(i, n_folds) == val_fold)
        ids = [i for i in ids if keep(i)]

        self.samples = [(images_by_id[i], anns_by_image.get(i, [])) for i in ids]

    def __len__(self):
        return len(self.samples) * self.samples_per_image

    def _random_tile(self, img, semantic, instance, spine_heat):
        h, w = img.shape[:2]
        t = self.tile_size
        # bias sampling toward tiles that actually contain filament so we
        # don't waste most crops on empty sky background
        y, x = 0, 0
        for _ in range(10):
            y = random.randint(0, max(0, h - t))
            x = random.randint(0, max(0, w - t))
            if semantic[y:y + t, x:x + t].sum() > 50 or random.random() < 0.15:
                break
        sl = (slice(y, y + t), slice(x, x + t))
        # .copy() is required, not optional, now that img/semantic/instance/
        # spine_heat may be the SAME underlying array object returned
        # repeatedly from the in-memory LRU cache (see _get_full_data).
        # Plain numpy slicing returns a view sharing memory with the
        # original; without copying here, augmentation applied to that view
        # downstream could corrupt the cached array for every future access.
        return img[sl].copy(), semantic[sl].copy(), instance[sl].copy(), spine_heat[sl].copy()

    def _load_image(self, im_info):
        img_path = os.path.join(self.data_root, "train", "train_images", im_info["file_name"])
        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    def _get_full_data(self, im_info, anns):
        """Returns (img, semantic, instance, spine_heat) for one full
        (un-tiled) image, using the in-memory LRU cache to avoid repeated
        JPEG decode + .npz decompression when the same image is requested
        multiple times per epoch (inevitable with samples_per_image > 1).
        Returns None if the image file itself is unreadable."""
        cached = self._mem_cache.get(im_info["id"])
        if cached is not None:
            return cached

        img = self._load_image(im_info)
        if img is None:
            return None

        semantic, instance, spine_heat = _load_or_build_raster(self.cache_dir, im_info, anns)
        result = (img, semantic, instance, spine_heat)
        self._mem_cache.put(im_info["id"], result)
        return result

    def __getitem__(self, idx):
        # Some MAGFiLO jpegs are truncated/corrupted on disk (interrupted
        # download or extraction). cv2.imread silently returns None for
        # these rather than raising, which used to crash the whole
        # DataLoader worker mid-epoch. Instead, skip a bad image and fall
        # back to a different random sample so a handful of bad files
        # doesn't kill an overnight run.
        for _ in range(10):
            real_idx = idx % len(self.samples)
            im_info, anns = self.samples[real_idx]
            data = self._get_full_data(im_info, anns)
            if data is not None:
                break
            print(f"[data] skipping unreadable image: {im_info['file_name']}")
            idx = random.randrange(len(self.samples))
        else:
            raise RuntimeError(
                "10 consecutive unreadable images -- check train_images for "
                "widespread corruption (see the diagnostic scan in the README).")

        img, semantic, instance, spine_heat = data
        img, semantic, instance, spine_heat = self._random_tile(img, semantic, instance, spine_heat)

        if self.augment is not None:
            # instance map must ride along with the same transform as
            # semantic/spine, or the embedding loss ends up pulling/pushing
            # pixels that no longer correspond to the same filament post-flip.
            out = self.augment(image=img, masks=[semantic, spine_heat, instance])
            img, (semantic, spine_heat, instance) = out["image"], out["masks"]

        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.25

        return {
            "image": torch.from_numpy(img).unsqueeze(0).float(),
            "semantic": torch.from_numpy(semantic.astype(np.float32)),
            "spine": torch.from_numpy(spine_heat.astype(np.float32)),
            "instance": torch.from_numpy(instance.astype(np.int64)),
            "image_id": im_info["id"],
        }


def collate_fn(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in
           ["image", "semantic", "spine", "instance"]}
    out["image_id"] = [b["image_id"] for b in batch]
    return out


def worker_init_fn(worker_id):
    """Seed each DataLoader worker deterministically off the main process's
    seed, so --seed actually gives reproducible tiles/augmentation across
    workers, not just single-process runs. Pass to DataLoader(worker_init_fn=...)."""
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)
