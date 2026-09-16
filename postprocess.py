"""
Turns (semantic prob map, spine heatmap, embedding map) into a list of
per-instance binary masks, encoded as RLE for submission. Kept as its own
file since it's a distinct stage from model/data -- pure array processing
with no torch dependency, run once per test image at inference time.

Pipeline:
  1. Threshold semantic map -> binary mask.
  2. Gap-bridge: close small breaks, directly targeting the fragmentation /
     one-to-many penalty.
  3. Label connected foreground blobs. Every step below runs PER BLOB, so a
     pixel can only ever be assigned to a seed inside its own blob. (An
     earlier version clustered against every seed in the whole image; the
     push loss only separates filaments that co-occur in a training tile,
     so distant filaments could share embeddings and pixels got assigned to
     seeds on the other side of the disk -- producing disconnected
     "instances" that hurt both IoU and the many-to-one penalty.)
  4. Threshold + skeletonize the spine heatmap -> seed components.
  5. Within each blob, merge seeds whose mean embeddings are close (a spine
     broken by a gap would otherwise split one filament in two).
  6. Blob with 0 or 1 (merged) seeds -> one instance. Blob with several ->
     assign each pixel to the nearest seed in embedding space, which
     separates touching/branching filaments. (Seedless blobs used to be
     split by distance-transform watershed, which chopped thin filaments
     into ~15px chunks along their ridge -- the worst case for the
     one-to-many penalty.)
  7. Drop instances below a minimum pixel-area floor (denoise).
  8. Encode each final instance mask to COCO RLE.
"""
import cv2
import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

EIGHT_CONN = np.ones((3, 3), dtype=bool)


def gap_bridge(binary_mask, close_kernel=5):
    if close_kernel <= 1:
        return binary_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    return cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


def get_seed_components(spine_prob, spine_thresh=0.4, min_seed_size=8):
    spine_bin = spine_prob > spine_thresh
    skel = skeletonize(spine_bin)
    seeds, _ = ndi.label(skel, structure=EIGHT_CONN)
    # Drop tiny seed fragments by hand rather than via remove_small_objects,
    # whose min_size/max_size kwarg changed meaning across skimage versions.
    sizes = np.bincount(seeds.ravel())
    keep = sizes >= min_seed_size
    keep[0] = False
    remap = np.zeros(len(sizes), dtype=np.int32)
    remap[keep] = np.arange(1, keep.sum() + 1)
    return remap[seeds], int(keep.sum())


def _merge_close_seeds(seed_means, merge_dist):
    """Union-find over seeds whose mean embeddings are within merge_dist.
    Returns a group index per seed (0..n_groups-1) and n_groups."""
    n = len(seed_means)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n > 1 and merge_dist > 0:
        d = np.linalg.norm(seed_means[:, None, :] - seed_means[None, :, :], axis=2)
        for i, j in np.argwhere(np.triu(d < merge_dist, k=1)):
            parent[find(i)] = find(j)

    roots = [find(i) for i in range(n)]
    uniq = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([uniq[r] for r in roots]), len(uniq)


def _split_blob(blob, seed_crop, emb_crop, merge_dist, chunk_size):
    """Split one blob (bool array, cropped) into sub-instances. Returns an
    int32 label array over the crop (0 outside the blob) and the count."""
    seed_ids = np.unique(seed_crop[blob])
    seed_ids = seed_ids[seed_ids != 0]
    if len(seed_ids) <= 1:
        return blob.astype(np.int32), 1

    # Mean embedding per seed, using only the seed pixels inside this blob.
    seed_means = np.stack([emb_crop[:, blob & (seed_crop == sid)].mean(axis=1)
                           for sid in seed_ids]).astype(np.float32)
    group_of_seed, n_groups = _merge_close_seeds(seed_means, merge_dist)
    if n_groups == 1:
        return blob.astype(np.int32), 1

    group_means = np.stack([seed_means[group_of_seed == g].mean(axis=0)
                            for g in range(n_groups)])

    # Chunked so an undertrained checkpoint that flags most of the image as
    # foreground (one giant blob) can't blow up memory.
    coords = np.nonzero(blob)
    n_px = len(coords[0])
    assign = np.empty(n_px, dtype=np.int32)
    for start in range(0, n_px, chunk_size):
        end = min(start + chunk_size, n_px)
        px_emb = emb_crop[:, coords[0][start:end], coords[1][start:end]].T  # (n, C)
        d = np.linalg.norm(px_emb[:, None, :] - group_means[None, :, :], axis=2)
        assign[start:end] = d.argmin(axis=1) + 1

    labels = np.zeros(blob.shape, dtype=np.int32)
    labels[coords] = assign
    return labels, n_groups


def instances_from_predictions(sem_prob, spine_prob, embedding, sem_thresh=0.5,
                                 spine_thresh=0.4, min_area=25, gap_kernel=5,
                                 merge_dist=1.0, chunk_size=200_000):
    """merge_dist: seeds in the same blob whose mean embeddings are closer
    than this are treated as one filament. Default 1.0 = 2 * delta_pull from
    the training loss (pixels of one instance are pulled to within 0.5 of
    their mean, so two seeds of the same instance should be within ~1.0)."""
    binary = gap_bridge(sem_prob > sem_thresh, close_kernel=gap_kernel)

    fg_fraction = binary.mean()
    if fg_fraction > 0.3:
        print(f"[postprocess] WARNING: {fg_fraction:.1%} of the image was flagged as "
              f"foreground at this sem_thresh -- this usually indicates an undertrained "
              f"or miscalibrated checkpoint rather than a real result. Treat this output "
              f"as unreliable.")

    blobs, n_blobs = ndi.label(binary, structure=EIGHT_CONN)
    if n_blobs == 0:
        return []
    seeds, _ = get_seed_components(spine_prob, spine_thresh=spine_thresh)

    out_masks = []
    H, W = binary.shape
    for bid, sl in enumerate(ndi.find_objects(blobs), start=1):
        if sl is None:
            continue
        blob = blobs[sl] == bid
        if blob.sum() < min_area:
            continue
        labels, n = _split_blob(blob, seeds[sl], embedding[(slice(None), *sl)],
                                merge_dist, chunk_size)
        for lid in range(1, n + 1):
            part = labels == lid
            if part.sum() < min_area:
                continue
            full = np.zeros((H, W), dtype=bool)
            full[sl] = part
            out_masks.append(full)
    return out_masks


def mask_to_rle_counts(mask_bool):
    """COCO RLE counts string per the competition's submission format
    (size is fixed 2048x2048, so we only need to store counts)."""
    rle = mask_utils.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    counts = rle["counts"]
    return counts.decode("utf-8") if isinstance(counts, bytes) else counts


def rle_counts_to_mask(counts, h=2048, w=2048):
    rle = {"size": [h, w], "counts": counts.encode("utf-8") if isinstance(counts, str) else counts}
    return mask_utils.decode(rle).astype(bool)


def build_submission_rows(image_id, instance_masks):
    return [{"filament_id": f"{image_id}_{i}", "segmentation_rle": mask_to_rle_counts(m)}
            for i, m in enumerate(instance_masks, start=1)]
