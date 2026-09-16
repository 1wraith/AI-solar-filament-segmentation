"""Shared fixtures: a tiny synthetic MAGFiLO-style dataset on disk, so the
data/train/infer/sweep paths can be exercised end to end without the real
(~800MB) competition data or a GPU."""
import json
import os

import cv2
import numpy as np
import pytest
import torch

import data as data_mod

# Small, fast backbone for tests. The real default (convnext_small) is
# covered separately in test_model.py.
TEST_BACKBONE = "convnext_atto"
IMG_SIZE = 256


def _filament_polygon(rng, size):
    """A thin, slightly bent strip as a COCO polygon plus its spine."""
    x0, y0 = rng.integers(20, size // 2, size=2)
    x1, y1 = x0 + rng.integers(60, size // 2 - 10), y0 + rng.integers(-15, 15)
    xm, ym = (x0 + x1) // 2, (y0 + y1) // 2 + 6
    top = [(x0, y0 - 3), (xm, ym - 3), (x1, y1 - 3)]
    bottom = [(x1, y1 + 3), (xm, ym + 3), (x0, y0 + 3)]
    poly = [float(v) for pt in top + bottom for v in pt]
    spine = [int(v) for pt in [(x0, y0), (xm, ym), (x1, y1)] for v in pt]
    return poly, spine


def make_dataset(root, n_images=12, size=IMG_SIZE, seed=0):
    """Writes <root>/train/{train_images, annotations json} and
    <root>/test/test_images. Image ids use the real '<6-digit batch>-<ts>'
    shape so fold splitting behaves like on the real data."""
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(root, "train", "train_images")
    test_dir = os.path.join(root, "test", "test_images")
    os.makedirs(img_dir)
    os.makedirs(test_dir)

    images, annotations, ann_id = [], [], 1
    for k in range(n_images):
        image_id = f"0104{k % 3:02d}-20240101{k:06d}"
        fname = f"{image_id}.jpg"
        img = (rng.random((size, size)) * 60 + 100).astype(np.uint8)
        for _ in range(2):
            poly, spine = _filament_polygon(rng, size)
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(img, [pts], 30)
            annotations.append({"id": ann_id, "image_id": image_id, "category_id": 1,
                                "segmentation": [poly], "spine": spine, "iscrowd": 0})
            ann_id += 1
        cv2.imwrite(os.path.join(img_dir, fname), img)
        images.append({"id": image_id, "file_name": fname, "height": size, "width": size})

    for k in range(2):
        img = (rng.random((size, size)) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(test_dir, f"test-{k}.jpg"), img)

    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "filament"}]}
    with open(os.path.join(root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"), "w") as f:
        json.dump(coco, f)
    return coco


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "magfilo"
    coco = make_dataset(str(root))
    # The sweep/val paths need at least one image in the default val fold.
    assert any(data_mod.fold_of(im["id"]) == 0 for im in coco["images"])
    assert any(data_mod.fold_of(im["id"]) != 0 for im in coco["images"])
    return str(root)


@pytest.fixture
def tiny_checkpoint(tmp_path):
    """A randomly initialised checkpoint in the same format cmd_train writes."""
    import pipeline
    from model import CombinedLoss, build_model

    model = build_model(backbone=TEST_BACKBONE, embedding_dim=8, pretrained=False)
    ckpt = {"model": model.state_dict(), "criterion": CombinedLoss().state_dict(),
            "args": {"backbone": TEST_BACKBONE, "embedding_dim": 8, "val_fold": 0},
            "epoch": 0, "val_dice": 0.0, "val_cldice_loss": 1.0}
    path = str(tmp_path / "tiny.pt")
    pipeline.save_checkpoint_atomic(ckpt, path)
    return path


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)
