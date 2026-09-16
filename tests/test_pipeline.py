import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import pipeline
from model import CombinedLoss, build_model
from postprocess import mask_to_rle_counts
from tests.conftest import TEST_BACKBONE


def test_tile_coords_cover_image():
    assert pipeline._tile_coords(2048, 896, 128) == [0, 768, 1152]
    assert pipeline._tile_coords(896, 896, 128) == [0]


@pytest.mark.parametrize("tile_size", [0, 128])
@pytest.mark.parametrize("tta", [["identity"], ["identity", "flip", "rot90"]])
def test_predict_full_image_shapes(tile_size, tta):
    model = build_model(backbone=TEST_BACKBONE, pretrained=False).eval()
    img = (np.random.rand(200, 170) * 255).astype(np.uint8)   # not a multiple of 32
    sem, spine, emb = pipeline._predict_full_image(model, img, "cpu", tile_size, 32, tta, 8)
    assert sem.shape == spine.shape == img.shape
    assert emb.shape == (8, *img.shape)
    assert 0 <= sem.min() and sem.max() <= 1
    assert np.isfinite(emb).all()


def test_tta_of_symmetric_input_matches_identity():
    model = build_model(backbone=TEST_BACKBONE, pretrained=False).eval()
    img = np.full((64, 64), 128, np.uint8)
    a = pipeline._predict_full_image(model, img, "cpu", 0, 0, ["identity"], 8)[0]
    b = pipeline._predict_full_image(model, img, "cpu", 0, 0, ["identity", "flip"], 8)[0]
    assert np.allclose(a, b, atol=1e-4)


def test_save_checkpoint_atomic(tmp_path):
    path = str(tmp_path / "c.pt")
    pipeline.save_checkpoint_atomic({"x": torch.ones(2)}, path)
    assert torch.load(path)["x"].sum() == 2
    assert os.listdir(tmp_path) == ["c.pt"]


def test_lr_schedule_warmup_then_cosine():
    model = build_model(backbone=TEST_BACKBONE, pretrained=False)
    args = SimpleNamespace(lr=1e-3, lr_backbone_mult=0.1, loss_weighting="auto",
                           warmup_epochs=2, epochs=10)
    opt, sched = pipeline.build_optimizer_and_scheduler(model, CombinedLoss(), args)
    assert [g["name"] for g in opt.param_groups] == ["backbone", "heads", "loss_weights"]
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[1]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[1] == pytest.approx(1e-3)
    assert lrs[-1] < lrs[2]
    assert opt.param_groups[0]["lr"] == pytest.approx(opt.param_groups[1]["lr"] * 0.1)


def test_parser_defaults_are_shared_between_infer_and_sweep():
    p = pipeline.build_parser()
    inf = p.parse_args(["infer", "--data_root", "d", "--checkpoint", "c"])
    swp = p.parse_args(["sweep", "--data_root", "d", "--checkpoint", "c"])
    for k in ("tile_size", "min_area", "gap_kernel", "merge_dist"):
        assert getattr(inf, k) == getattr(swp, k)
    assert inf.tile_size == 0 and inf.spine_thresh == 0.4


# ---- metrics ---------------------------------------------------------------

def _sq(y, x, s=10, size=64):
    m = np.zeros((size, size), bool)
    m[y:y + s, x:x + s] = True
    return m


def test_pq_perfect_match():
    gt = {"a": [_sq(0, 0), _sq(30, 30)]}
    m = pipeline._compute_pq_metrics(gt, {"a": [_sq(30, 30), _sq(0, 0)]}, 0.5)
    assert m["pq"] == pytest.approx(1.0)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 0, 0)


def test_pq_counts_misses_and_false_positives():
    gt = {"a": [_sq(0, 0)], "b": [_sq(0, 0)]}
    pred = {"a": [_sq(0, 0), _sq(40, 40)]}   # image b predicted nothing
    m = pipeline._compute_pq_metrics(gt, pred, 0.5)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["pq"] == pytest.approx(1 / (1 + 0.5 + 0.5))


def test_pq_low_iou_is_not_a_match():
    m = pipeline._compute_pq_metrics({"a": [_sq(0, 0)]}, {"a": [_sq(0, 6)]}, 0.5)
    assert m["tp"] == 0 and m["pq"] == 0


def test_cmd_eval_end_to_end(tmp_path, capsys):
    gt = {"images": [{"id": "010401-1", "height": 2048, "width": 2048}],
          "annotations": [{"image_id": "010401-1",
                           "segmentation": [[100, 100, 200, 100, 200, 120, 100, 120]]}]}
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt))
    m = np.zeros((2048, 2048), bool)
    m[100:121, 100:201] = True
    pd.DataFrame([{"filament_id": "010401-1_1", "segmentation_rle": mask_to_rle_counts(m)}]) \
        .to_csv(tmp_path / "pred.csv", index=False)
    pipeline.cmd_eval(SimpleNamespace(gt_json=str(gt_path), pred_csv=str(tmp_path / "pred.csv"),
                                      iou_thresh=0.5))
    out = capsys.readouterr().out
    assert "TP / FP / FN:         1 / 0 / 0" in out


# ---- end-to-end subcommands on synthetic data ------------------------------

def _run_cli(argv):
    args = pipeline.build_parser().parse_args(argv)
    args.func(args)
    return args


def test_cmd_train_one_epoch(data_root, tmp_path, monkeypatch):
    real_build = pipeline.build_model
    monkeypatch.setattr(pipeline, "build_model",
                        lambda **kw: real_build(**{**kw, "pretrained": False}))
    out_dir = str(tmp_path / "run")
    _run_cli(["train", "--data_root", data_root, "--out_dir", out_dir,
              "--backbone", TEST_BACKBONE, "--tile_size", "64", "--batch_size", "2",
              "--epochs", "1", "--samples_per_image", "1", "--cldice_iters", "2",
              "--num_workers", "0"])
    for name in ("best.pt", "last.pt"):
        ckpt = torch.load(os.path.join(out_dir, name), weights_only=True)
        assert ckpt["epoch"] == 0 and "func" not in ckpt["args"]
        assert ckpt["args"]["backbone"] == TEST_BACKBONE


def test_cmd_infer_writes_submission(data_root, tiny_checkpoint, tmp_path):
    out_csv = str(tmp_path / "sub.csv")
    _run_cli(["infer", "--data_root", data_root, "--checkpoint", tiny_checkpoint,
              "--out_csv", out_csv, "--sem_thresh", "0.0", "--min_area", "1"])
    df = pd.read_csv(out_csv)
    assert list(df.columns) == ["filament_id", "segmentation_rle"]
    assert len(df) > 0
    assert df["filament_id"].str.match(r"^test-\d+_\d+$").all()


def test_cmd_infer_skips_corrupt_image(data_root, tiny_checkpoint, tmp_path, capsys):
    with open(os.path.join(data_root, "test", "test_images", "test-0.jpg"), "wb") as f:
        f.write(b"garbage")
    out_csv = str(tmp_path / "sub.csv")
    _run_cli(["infer", "--data_root", data_root, "--checkpoint", tiny_checkpoint,
              "--out_csv", out_csv, "--sem_thresh", "0.0", "--min_area", "1"])
    assert "1 / 2 test images were skipped" in capsys.readouterr().out
    assert os.path.exists(out_csv)


def test_cmd_sweep_writes_results(data_root, tiny_checkpoint, tmp_path):
    out_csv = str(tmp_path / "sweep.csv")
    _run_cli(["sweep", "--data_root", data_root, "--checkpoint", tiny_checkpoint,
              "--out_csv", out_csv, "--sem_thresh_list", "0.3,0.6",
              "--spine_thresh_list", "0.4"])
    df = pd.read_csv(out_csv)
    assert len(df) == 2
    assert {"sem_thresh", "spine_thresh", "pq", "tp", "fp", "fn"} <= set(df.columns)
