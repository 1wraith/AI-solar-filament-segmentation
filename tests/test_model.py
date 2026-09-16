import pytest
import torch

from model import (
    CombinedLoss,
    build_model,
    cl_dice_loss,
    embedding_push_pull_loss,
    soft_dice_loss,
    soft_skeletonize,
)
from tests.conftest import TEST_BACKBONE


@pytest.mark.parametrize("backbone,size", [
    (TEST_BACKBONE, 96),
    ("resnet18", 96),          # 5 feature levels (different decoder path)
    ("convnext_small", 64),    # the pipeline default
])
def test_forward_shapes(backbone, size):
    model = build_model(backbone=backbone, embedding_dim=8, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, size, size))
    assert out["semantic"].shape == (2, size, size)
    assert out["spine"].shape == (2, size, size)
    assert out["embedding"].shape == (2, 8, size, size)


def _batch(b=2, s=64):
    sem = torch.zeros(b, s, s)
    sem[:, 20:24, 5:60] = 1
    sem[:, 40:44, 5:60] = 1
    inst = torch.zeros(b, s, s, dtype=torch.long)
    inst[:, 20:24, 5:60] = 1
    inst[:, 40:44, 5:60] = 2
    spine = torch.zeros(b, s, s)
    spine[:, 22, 5:60] = 1
    return {"semantic": sem, "spine": spine, "instance": inst}


@pytest.mark.parametrize("auto_weight", [True, False])
def test_combined_loss_is_finite_and_backpropagates(auto_weight):
    model = build_model(backbone=TEST_BACKBONE, pretrained=False)
    crit = CombinedLoss(auto_weight=auto_weight, cldice_iters=3)
    out = model(torch.randn(2, 1, 64, 64))
    total, logs = crit(out, _batch())
    assert torch.isfinite(total)
    total.backward()
    assert all(p.grad is not None for p in model.semantic_head.parameters())
    assert set(CombinedLoss.LOSS_NAMES) <= set(logs)
    if auto_weight:
        assert crit.log_vars.grad is not None


def test_auto_weights_are_clamped():
    crit = CombinedLoss(auto_weight=True)
    with torch.no_grad():
        crit.log_vars.fill_(100.0)
    w = crit.effective_weights()
    assert min(w.values()) == pytest.approx(torch.exp(torch.tensor(-4.0)).item())


def test_dice_loss_extremes():
    t = _batch()["semantic"]
    assert soft_dice_loss(t * 40 - 20, t).item() < 1e-3
    assert soft_dice_loss(-(t * 40 - 20), t).item() > 0.99


def test_cldice_perfect_prediction_is_near_zero():
    t = _batch()["semantic"]
    assert cl_dice_loss(t * 40 - 20, t, iters=5).item() < 0.05


def test_soft_skeleton_is_thinner_than_input():
    x = _batch()["semantic"].unsqueeze(1)
    skel = soft_skeletonize(x, iters=5)
    assert 0 < skel.sum() < x.sum()


def test_embedding_loss_rewards_separated_instances():
    inst = _batch()["instance"]
    good = torch.zeros(2, 4, 64, 64)
    good[:, 0][inst == 1] = 0.0
    good[:, 0][inst == 2] = 5.0
    pull, push = embedding_push_pull_loss(good, inst)
    assert pull.item() == 0 and push.item() == 0

    bad = torch.zeros(2, 4, 64, 64)
    _, push_bad = embedding_push_pull_loss(bad, inst)
    assert push_bad.item() > 0


def test_embedding_loss_handles_no_instances():
    pull, push = embedding_push_pull_loss(torch.randn(1, 4, 16, 16),
                                          torch.zeros(1, 16, 16, dtype=torch.long))
    assert pull.item() == 0 and push.item() == 0
