import os

import numpy as np
import pytest
import torch

import data as data_mod
from data import MagfiloDataset, _LRUCache, _rasterize, collate_fn, fold_of, group_key


def test_group_key_strips_batch_prefix():
    assert group_key("010401-20240101000000") == group_key("010402-20240101000000")
    assert fold_of("010401-20240101000000") == fold_of("010402-20240101000000")


def test_folds_cover_range():
    folds = {fold_of(f"010401-2024{i:010d}") for i in range(200)}
    assert folds == set(range(5))


def test_rasterize_polygon_and_spine():
    im = {"id": "x", "height": 64, "width": 64}
    anns = [
        {"segmentation": [[10, 10, 50, 10, 50, 20, 10, 20]], "spine": [10, 15, 50, 15]},
        {"segmentation": [[10, 40, 50, 40, 50, 50, 10, 50]]},
    ]
    sem, inst, spine = _rasterize(im, anns)
    assert sem.dtype == np.uint8 and sem.max() == 1
    assert set(np.unique(inst)) == {0, 1, 2}
    assert spine.max() == pytest.approx(1.0)
    assert spine[15, 30] > spine[45, 30]


def test_rasterize_overlap_is_first_wins():
    im = {"id": "x", "height": 32, "width": 32}
    sq = [[5, 5, 20, 5, 20, 20, 5, 20]]
    _, inst, _ = _rasterize(im, [{"segmentation": sq}, {"segmentation": sq}])
    assert set(np.unique(inst)) == {0, 1}


def test_lru_cache_evicts_oldest():
    c = _LRUCache(max_items=2)
    c.put("a", 1); c.put("b", 2)
    c.get("a")
    c.put("c", 3)
    assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3
    off = _LRUCache(max_items=0)
    off.put("a", 1)
    assert off.get("a") is None


def test_train_val_split_is_disjoint_by_observation(data_root):
    tr = MagfiloDataset(data_root, split="train", tile_size=128, augment=None)
    va = MagfiloDataset(data_root, split="val", tile_size=128)
    tr_keys = {group_key(im["id"]) for im, _ in tr.samples}
    va_keys = {group_key(im["id"]) for im, _ in va.samples}
    assert tr_keys and va_keys and not tr_keys & va_keys
    assert va.augment is None


@pytest.mark.parametrize("augment", [None, "default"])
def test_getitem_shapes_and_types(data_root, augment):
    ds = MagfiloDataset(data_root, split="train", tile_size=128, augment=augment,
                        samples_per_image=3)
    assert len(ds) == 3 * len(ds.samples)
    for i in range(len(ds)):
        item = ds[i]
        assert item["image"].shape == (1, 128, 128)
        assert item["semantic"].shape == item["spine"].shape == item["instance"].shape == (128, 128)
        assert item["instance"].dtype == torch.int64
        assert set(item["semantic"].unique().tolist()) <= {0.0, 1.0}
        # instance labels must stay integer ids and line up with the semantic mask
        assert ((item["instance"] > 0).float() <= item["semantic"]).all()


def test_raster_cache_is_written_and_reused(data_root):
    ds = MagfiloDataset(data_root, split="train", tile_size=128, augment=None, mem_cache_size=0)
    ds[0]
    im_id = ds.samples[0][0]["id"]
    path = os.path.join(ds.cache_dir, f"{im_id}.npz")
    assert os.path.exists(path)
    mtime = os.path.getmtime(path)
    ds[0]
    assert os.path.getmtime(path) == mtime
    assert not [f for f in os.listdir(ds.cache_dir) if f.endswith(".tmp.npz")]


def test_unreadable_image_is_skipped(data_root, capsys):
    ds = MagfiloDataset(data_root, split="train", tile_size=128, augment=None)
    bad = ds.samples[0][0]["file_name"]
    with open(os.path.join(data_root, "train", "train_images", bad), "wb") as f:
        f.write(b"not a jpeg")
    item = ds[0]
    assert item["image"].shape == (1, 128, 128)
    assert "skipping unreadable image" in capsys.readouterr().out


def test_collate_fn(data_root):
    ds = MagfiloDataset(data_root, split="train", tile_size=64, augment=None)
    b = collate_fn([ds[0], ds[1]])
    assert b["image"].shape == (2, 1, 64, 64)
    assert len(b["image_id"]) == 2


def test_default_augmentation_builds():
    assert data_mod.build_default_augmentation() is not None
