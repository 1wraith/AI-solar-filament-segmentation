import cv2
import numpy as np
import pytest

from postprocess import (
    _merge_close_seeds,
    build_submission_rows,
    gap_bridge,
    get_seed_components,
    instances_from_predictions,
    mask_to_rle_counts,
    rle_counts_to_mask,
)

H = W = 2048


def _n_components(mask):
    return cv2.connectedComponents(mask.astype(np.uint8))[0] - 1


@pytest.fixture(scope="module")
def scene():
    """Four synthetic situations on one 2048x2048 disk (see assertions)."""
    sem = np.zeros((H, W), np.float32)
    spine = np.zeros((H, W), np.float32)
    emb = np.zeros((8, H, W), np.float32)

    # A: long filament whose spine is broken in the middle -> 1 instance.
    cv2.line(sem, (100, 100), (900, 150), 1.0, 7)
    cv2.line(spine, (105, 100), (480, 124), 1.0, 1)
    cv2.line(spine, (520, 126), (895, 150), 1.0, 1)
    emb[0][sem > 0] = 1.0
    # The two halves differ slightly (0.5 apart): close enough to be merged
    # at the default merge_dist=1.0, far enough to split when merging is off.
    right_half = np.zeros((H, W), bool)
    right_half[:200, 500:1000] = True
    emb[2][right_half & (sem > 0)] = 0.5

    # B: far-away filament with the SAME embedding as A -> must stay separate.
    b = np.zeros((H, W), np.float32)
    cv2.line(b, (1500, 1800), (1900, 1700), 1.0, 7)
    sem = np.maximum(sem, b)
    emb[0][b > 0] = 1.0
    cv2.line(spine, (1505, 1799), (1895, 1701), 1.0, 1)

    # C: two touching filaments (a T) with different embeddings -> 2 instances.
    cv2.line(sem, (1000, 1000), (1400, 1000), 1.0, 7)
    cv2.line(sem, (1200, 1004), (1200, 1400), 1.0, 7)
    emb[1][990:1010, 995:1405] = 5.0
    cv2.line(spine, (1005, 1000), (1395, 1000), 1.0, 1)
    cv2.line(spine, (1200, 1020), (1200, 1395), 1.0, 1)

    # D: seedless thin filament -> exactly 1 instance (not chopped up).
    cv2.line(sem, (100, 1500), (800, 1600), 1.0, 7)

    return sem, spine, emb


def test_scene_instance_count(scene):
    masks = instances_from_predictions(*scene)
    assert len(masks) == 5


def test_every_instance_is_one_connected_piece(scene):
    for m in instances_from_predictions(*scene):
        assert _n_components(m) == 1


def test_instances_do_not_overlap(scene):
    masks = instances_from_predictions(*scene)
    assert np.stack(masks).sum(axis=0).max() == 1


def test_seedless_filament_is_not_fragmented(scene):
    sem, _, emb = scene
    masks = instances_from_predictions(sem, np.zeros_like(sem), emb)
    # One instance per connected blob: A, B, the T, and D.
    assert len(masks) == 4


def test_merge_dist_zero_splits_broken_spine(scene):
    sem, spine, emb = scene
    # With merging disabled, A's two spine pieces become two instances.
    assert len(instances_from_predictions(sem, spine, emb, merge_dist=0)) == 6


def test_empty_prediction_returns_no_instances():
    z = np.zeros((64, 64), np.float32)
    assert instances_from_predictions(z, z, np.zeros((8, 64, 64), np.float32)) == []


def test_min_area_filters_small_blobs():
    sem = np.zeros((128, 128), np.float32)
    sem[10:13, 10:13] = 1.0          # 9 px
    sem[50:70, 50:70] = 1.0          # 400 px
    z = np.zeros_like(sem)
    masks = instances_from_predictions(sem, z, np.zeros((8, 128, 128), np.float32), min_area=25)
    assert [int(m.sum()) for m in masks] == [400]


def test_sem_thresh_is_respected():
    sem = np.full((64, 64), 0.45, np.float32)
    z = np.zeros_like(sem)
    e = np.zeros((8, 64, 64), np.float32)
    assert instances_from_predictions(sem, z, e, sem_thresh=0.5) == []
    assert len(instances_from_predictions(sem, z, e, sem_thresh=0.4)) == 1


def test_gap_bridge_closes_small_gap_and_can_be_disabled():
    m = np.zeros((32, 64), np.uint8)
    m[14:18, 5:30] = 1
    m[14:18, 32:60] = 1               # 2px gap
    assert _n_components(gap_bridge(m, 5)) == 1
    assert _n_components(gap_bridge(m, 1)) == 2


def test_seed_components_drop_tiny_fragments():
    spine = np.zeros((64, 64), np.float32)
    spine[10, 5:40] = 1.0             # long seed
    spine[50, 50:53] = 1.0            # 3px fragment
    seeds, n = get_seed_components(spine, min_seed_size=8)
    assert n == 1
    assert set(np.unique(seeds)) == {0, 1}


def test_merge_close_seeds_groups_transitively():
    means = np.array([[0, 0], [0.8, 0], [1.6, 0], [10, 10]], np.float32)
    groups, n = _merge_close_seeds(means, merge_dist=1.0)
    assert n == 2
    assert groups[0] == groups[1] == groups[2] != groups[3]


def test_rle_round_trip():
    rng = np.random.default_rng(0)
    m = rng.random((H, W)) > 0.999
    assert np.array_equal(rle_counts_to_mask(mask_to_rle_counts(m)), m)


def test_submission_rows_format():
    m = np.zeros((H, W), bool)
    m[5:10, 5:10] = True
    rows = build_submission_rows("010401-20240101000000", [m, m])
    assert [r["filament_id"] for r in rows] == ["010401-20240101000000_1", "010401-20240101000000_2"]
    assert all(isinstance(r["segmentation_rle"], str) for r in rows)
