"""
FilamentNet: encoder-decoder with three heads (semantic, spine, embedding),
plus the combined topology-aware loss used to train it.

Architecture and losses live in one file because they're two halves of the
same design decision -- each loss term exists to correct a specific failure
mode of a specific head, so it's easier to keep them next to each other than
to trace the correspondence across files.

Heads:
  semantic   1ch logits, filament vs background.
  spine      1ch logits -> sigmoid gives a skeleton heatmap, used at
             inference to seed connected components for instance separation.
  embedding  Cch dense embedding, used to cluster pixels into instances
             around spine seeds (see postprocess.py) instead of relying on
             box-shaped proposals, which is the wrong prior for thin,
             branching filaments.

Losses:
  dice + focal        standard region-overlap terms.
  cl_dice              topology-preserving term (Shit et al., CVPR 2021) --
                        scores overlap along the *skeleton* of prediction vs
                        target, directly rewarding connectivity rather than
                        raw area. This is the single highest-leverage term
                        for the fragmentation / barb-loss failure modes. It
                        is also the most expensive term per step (10 soft
                        erode/dilate iterations by default) -- cldice_iters
                        is exposed on CombinedLoss so it can be lowered on
                        slower local hardware if it turns out to be a
                        training-speed bottleneck.
  boundary              up-weights error near the mask edge, where thin barb
                        boundaries live and where Dice is most sensitive.
  spine                MSE against the spine heatmap target.
  pull / push          discriminative instance embedding loss (Neven et al. /
                        De Brabandere et al.) that trains the embedding head.

Swap backbone via timm; interface is backbone-agnostic through
`features_only=True`. Default is convnext_small (see pipeline.py's CLI help
for why, given a small ~880-image dataset) -- for best results on thin
curvilinear structures, try backbone="hrnet_w32" or "hrnet_w48", which keep
a high-resolution stream throughout the network instead of
downsample-then-upsample.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__()
        enc = list(reversed(encoder_channels))
        blocks, in_ch = [], enc[0]
        for i, out_ch in enumerate(decoder_channels):
            skip_ch = enc[i + 1] if i + 1 < len(enc) else 0
            blocks.append(ConvBlock(in_ch + skip_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, feats):
        feats = list(reversed(feats))
        x = feats[0]
        for i, block in enumerate(self.blocks):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            if i + 1 < len(feats):
                skip = feats[i + 1]
                if skip.shape[-2:] != x.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class FilamentNet(nn.Module):
    def __init__(self, backbone="convnext_small", embedding_dim=8, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained,
                                          in_chans=1, features_only=True)
        enc_channels = [f["num_chs"] for f in self.encoder.feature_info]
        self.decoder = UNetDecoder(enc_channels)
        final_ch = 16

        self.semantic_head = nn.Conv2d(final_ch, 1, 1)
        self.spine_head = nn.Conv2d(final_ch, 1, 1)
        self.embedding_head = nn.Conv2d(final_ch, embedding_dim, 1)

    def forward(self, x):
        in_size = x.shape[-2:]
        dec = self.decoder(self.encoder(x))
        if dec.shape[-2:] != in_size:
            dec = F.interpolate(dec, size=in_size, mode="bilinear", align_corners=False)
        return {
            "semantic": self.semantic_head(dec).squeeze(1),
            "spine": self.spine_head(dec).squeeze(1),
            "embedding": self.embedding_head(dec),
        }


def build_model(backbone="convnext_small", embedding_dim=8, pretrained=True):
    return FilamentNet(backbone=backbone, embedding_dim=embedding_dim, pretrained=pretrained)


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def soft_dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2 * (pred * target).sum(dim=(1, 2)) + eps
    den = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + eps
    return 1 - (num / den).mean()


def focal_loss(pred, target, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p = torch.sigmoid(pred)
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


def _soft_erode(x):
    p1 = -F.max_pool2d(-x, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def _soft_dilate(x):
    return F.max_pool2d(x, (3, 3), (1, 1), (1, 1))


def _soft_open(x):
    return _soft_dilate(_soft_erode(x))


def soft_skeletonize(x, iters=10):
    """Differentiable morphological skeletonization (Shit et al., 2021).
    `iters` trades topological fidelity for speed -- lower it (e.g. 4-6) if
    cl_dice_loss turns out to be a training-speed bottleneck on local
    hardware; the erode/dilate passes below are the expensive part of
    every forward/backward step this loss is included in."""
    x1 = _soft_open(x)
    skel = F.relu(x - x1)
    for _ in range(iters):
        x = _soft_erode(x)
        x1 = _soft_open(x)
        delta = F.relu(x - x1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cl_dice_loss(pred_logits, target, iters=10, eps=1e-6):
    pred = torch.sigmoid(pred_logits).unsqueeze(1)
    tgt = target.unsqueeze(1)

    skel_pred = soft_skeletonize(pred, iters)
    skel_tgt = soft_skeletonize(tgt, iters)

    t_prec = (skel_pred * tgt).sum(dim=(1, 2, 3)) + eps
    t_prec = t_prec / (skel_pred.sum(dim=(1, 2, 3)) + eps)
    t_sens = (skel_tgt * pred).sum(dim=(1, 2, 3)) + eps
    t_sens = t_sens / (skel_tgt.sum(dim=(1, 2, 3)) + eps)

    cl_dice = 2 * t_prec * t_sens / (t_prec + t_sens + eps)
    return 1 - cl_dice.mean()


def boundary_loss(pred_logits, target, kernel_size=5):
    """Up-weights error near the mask boundary, where barb edges live."""
    pred = torch.sigmoid(pred_logits).unsqueeze(1)
    tgt = target.unsqueeze(1)
    pad = kernel_size // 2
    tgt_dil = F.max_pool2d(tgt, kernel_size, 1, pad)
    tgt_ero = -F.max_pool2d(-tgt, kernel_size, 1, pad)
    boundary_band = (tgt_dil - tgt_ero).clamp(0, 1)
    return ((pred - tgt).abs() * (1 + 4 * boundary_band)).mean()


def spine_heatmap_loss(pred_logits, target):
    return F.mse_loss(torch.sigmoid(pred_logits), target)


def embedding_push_pull_loss(emb, instance_map, delta_pull=0.5, delta_push=1.5,
                               max_instances=24):
    """
    Discriminative instance embedding loss (Neven et al. / De Brabandere et
    al.): pulls same-instance pixel embeddings toward their mean, pushes
    different-instance means apart. Per-image instance loop is unavoidable
    (variable instance count per image), but the per-instance pull term is
    vectorized over pixels rather than looped.
    """
    B = emb.shape[0]
    total_pull, total_push, n_terms = emb.new_zeros(()), emb.new_zeros(()), 0

    for b in range(B):
        ids = torch.unique(instance_map[b])
        ids = ids[ids != 0][:max_instances]
        if len(ids) == 0:
            continue

        means = []
        for iid in ids:
            mask = instance_map[b] == iid
            if mask.sum() < 4:
                continue
            vecs = emb[b, :, mask].T
            mean = vecs.mean(dim=0)
            means.append(mean)
            total_pull = total_pull + F.relu((vecs - mean).norm(dim=1) - delta_pull).pow(2).mean()
            n_terms += 1

        if len(means) > 1:
            means = torch.stack(means, dim=0)
            dists = torch.cdist(means, means)
            iu = torch.triu_indices(len(means), len(means), offset=1, device=emb.device)
            d = dists[iu[0], iu[1]]
            total_push = total_push + F.relu(2 * delta_push - d).pow(2).mean()

    denom = max(n_terms, 1)
    return total_pull / denom, total_push / max(B, 1)


class CombinedLoss(nn.Module):
    LOSS_NAMES = ("dice", "focal", "cldice", "boundary", "spine", "pull", "push")

    def __init__(self, w_dice=1.0, w_focal=1.0, w_cldice=1.0, w_boundary=0.5,
                 w_spine=0.5, w_pull=1.0, w_push=1.0, cldice_iters=10, auto_weight=True):
        """
        auto_weight: if True (default), the static w_* arguments are only
            used as the *starting point* -- each term additionally gets a
            learnable weight that the optimizer adjusts during training
            based on that term's actual observed scale on this data
            (uncertainty-weighted multi-task loss, Kendall et al. 2018:
            https://arxiv.org/abs/1705.07115). This exists because the
            static w_* defaults were reasonable guesses, never empirically
            checked against this dataset's actual per-term gradient
            magnitudes -- it's common for one term (often the embedding
            push term, or focal) to end up dominating training even when
            the static weights look balanced on paper. With auto_weight,
            the optimizer corrects for that instead of you having to guess
            better numbers blind.

            The learnable parameters are log-variances s_i; the effective
            per-term weight is exp(-s_i), and total loss adds + s_i as a
            regularizer that prevents the trivial solution of driving every
            weight to zero (increasing s_i to shrink a term's contribution
            also grows the + s_i penalty). Initialized at s_i = 0, i.e.
            weight = 1.0 for every term at epoch 0 -- training starts from
            the same relative weighting as the static defaults and adapts
            from there, rather than starting somewhere untested.

            Set auto_weight=False to fall back to fixed static weights if
            the learned weights ever look unstable (e.g. exploding or
            collapsing to near-zero for a term that clearly still matters --
            check the logvar_* entries in the per-epoch training log).
        """
        super().__init__()
        self.static_w = dict(dice=w_dice, focal=w_focal, cldice=w_cldice, boundary=w_boundary,
                              spine=w_spine, pull=w_pull, push=w_push)
        self.cldice_iters = cldice_iters
        self.auto_weight = auto_weight
        if auto_weight:
            self.log_vars = nn.Parameter(torch.zeros(len(self.LOSS_NAMES)))
        else:
            self.log_vars = None

    def effective_weights(self):
        """Returns the current per-term weight actually applied -- static
        w_* if auto_weight=False, or exp(-log_var_i) if auto_weight=True.
        Useful for logging/diagnosing whether a term has drifted to
        dominate or been suppressed to near-zero."""
        if not self.auto_weight:
            return dict(self.static_w)
        with torch.no_grad():
            precisions = torch.exp(-self._clamped_log_vars())
        return {name: precisions[i].item() for i, name in enumerate(self.LOSS_NAMES)}

    def _clamped_log_vars(self):
        # Clamped to [-4, 4] and used for the ACTUAL loss computation in
        # forward(), not just for reporting. Without this, the optimizer can
        # exploit the uncertainty-weighting formula: driving a term's
        # log_var very negative shrinks `total` directly via the `+log_var`
        # regularizer, and if that term's raw loss is small enough, the
        # matching exp(-log_var) weight blow-up doesn't cancel the benefit.
        # This showed up for real during training -- train_total went
        # negative while val_dice stayed flat/got slightly worse, with
        # focal/boundary/spine/pull weights climbing every epoch while
        # dice/cldice stayed near 1.0. Clamping bounds the exploit: the
        # gradient through a saturated clamp is zero, so the optimizer gets
        # no further reward for pushing log_var past the boundary. Range of
        # [-4, 4] caps the weight between ~0.018x and ~54.6x the base scale,
        # generous enough for real rebalancing but not enough to let one
        # term's regularizer term swamp the others across the 7-term sum.
        return self.log_vars.clamp(-4, 4)

    def forward(self, out, batch):
        sem_logits, spine_logits, emb = out["semantic"], out["spine"], out["embedding"]
        target_sem, target_spine, instance = batch["semantic"], batch["spine"], batch["instance"]

        losses = dict(
            dice=soft_dice_loss(sem_logits, target_sem),
            focal=focal_loss(sem_logits, target_sem),
            cldice=cl_dice_loss(sem_logits, target_sem, iters=self.cldice_iters),
            boundary=boundary_loss(sem_logits, target_sem),
            spine=spine_heatmap_loss(spine_logits, target_spine),
        )
        losses["pull"], losses["push"] = embedding_push_pull_loss(emb, instance)

        if self.auto_weight:
            clamped = self._clamped_log_vars()
            total = sem_logits.new_zeros(())
            for i, name in enumerate(self.LOSS_NAMES):
                precision = torch.exp(-clamped[i])
                total = total + precision * losses[name] + clamped[i]
        else:
            total = sem_logits.new_zeros(())
            for name in self.LOSS_NAMES:
                total = total + self.static_w[name] * losses[name]

        logs = {name: v.item() for name, v in losses.items()}
        logs["total"] = total.item()
        for name, w in self.effective_weights().items():
            logs[f"weight_{name}"] = w
        return total, logs
