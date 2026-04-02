from __future__ import annotations

from pathlib import Path

import numpy as np


def _normalize_image(image_2d: np.ndarray) -> np.ndarray:
    img = image_2d.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-8:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def _axis_to_index(mode: str) -> int:
    if mode == "axial":
        return 0
    if mode == "coronal":
        return 1
    if mode == "sagittal":
        return 2
    raise ValueError(f"Unsupported slice mode: {mode}")


def _slice_2d(arr_3d: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return arr_3d[idx, :, :]
    if axis == 1:
        return arr_3d[:, idx, :]
    return arr_3d[:, :, idx]


def _consecutive_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = indices[0]
    prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        runs.append((start, prev))
        start = i
        prev = i
    runs.append((start, prev))
    return runs


def _best_run_center(
    runs: list[tuple[int, int]],
    slice_scores: np.ndarray,
    axis_len: int,
) -> int:
    """
    Choose best run by:
    1) larger total mask voxels in run
    2) closer run center to volume center
    """
    if not runs:
        return axis_len // 2

    vol_center = (axis_len - 1) / 2.0
    best_key = None
    best_run = runs[0]

    for start, end in runs:
        run_center = (start + end) / 2.0
        total_voxels = float(slice_scores[start : end + 1].sum())
        key = (total_voxels, -abs(run_center - vol_center))
        if best_key is None or key > best_key:
            best_key = key
            best_run = (start, end)

    return (best_run[0] + best_run[1]) // 2


def select_informative_slice_index(seg_arr: np.ndarray, axis: int, labels: list[int]) -> int:
    """
    Select most informative slice by mask co-occurrence priority:
    1) contains all 3 labels
    2) contains >=2 labels
    3) otherwise max total mask voxels

    For case (1) and (2), if multiple consecutive slices exist, select center of
    the best run. Best run is decided by highest total mask voxels; tie-breaker
    is closer to volume center.
    """
    axis_len = seg_arr.shape[axis]
    per_slice_presence = np.zeros(axis_len, dtype=np.int32)
    per_slice_total = np.zeros(axis_len, dtype=np.float64)

    for idx in range(axis_len):
        s2d = _slice_2d(seg_arr, axis, idx)
        counts = [(s2d == lb).sum() for lb in labels]
        per_slice_presence[idx] = int(sum(c > 0 for c in counts))
        per_slice_total[idx] = float(sum(counts))

    idx_all_three = [i for i in range(axis_len) if per_slice_presence[i] == 3]
    if idx_all_three:
        runs = _consecutive_runs(idx_all_three)
        return _best_run_center(runs, per_slice_total, axis_len)

    idx_any_two = [i for i in range(axis_len) if per_slice_presence[i] >= 2]
    if idx_any_two:
        runs = _consecutive_runs(idx_any_two)
        return _best_run_center(runs, per_slice_total, axis_len)

    # fallback: single-organ / any-mask slice with maximal coverage
    best_score = float(per_slice_total.max())
    cands = np.where(per_slice_total == best_score)[0]
    if len(cands) == 0:
        return axis_len // 2

    vol_center = (axis_len - 1) / 2.0
    cands = sorted(cands.tolist(), key=lambda x: abs(x - vol_center))
    return int(cands[0])


def save_overlay_preview(
    image_arr: np.ndarray,
    seg_arr: np.ndarray,
    organ_rgb: dict[str, tuple[int, int, int]],
    organ_orders: dict[str, int],
    out_path: Path,
    slice_mode: str = "axial",
    alpha: float = 0.4,
) -> None:
    """Save a single 2D overlay preview based on informative slice selection."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    axis = _axis_to_index(slice_mode)
    labels = list(organ_orders.values())
    idx = select_informative_slice_index(seg_arr, axis=axis, labels=labels)

    base = _normalize_image(_slice_2d(image_arr, axis, idx))
    seg2d = _slice_2d(seg_arr, axis, idx)

    rgb = np.stack([base, base, base], axis=-1)
    for organ, label in organ_orders.items():
        color = np.asarray(organ_rgb[organ], dtype=np.float32) / 255.0
        mask = seg2d == label
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title(f"{slice_mode} slice={idx}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_overlay_preview_three_plane(
    image_arr: np.ndarray,
    seg_arr: np.ndarray,
    organ_rgb: dict[str, tuple[int, int, int]],
    organ_orders: dict[str, int],
    out_path: Path,
    alpha: float = 0.4,
) -> None:
    """Save axial/coronal/sagittal overlays with informative slice per direction."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    modes = ["axial", "coronal", "sagittal"]

    import matplotlib.pyplot as plt

    labels = list(organ_orders.values())
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, mode in zip(axes, modes):
        axis = _axis_to_index(mode)
        idx = select_informative_slice_index(seg_arr, axis=axis, labels=labels)

        base = _normalize_image(_slice_2d(image_arr, axis, idx))
        seg2d = _slice_2d(seg_arr, axis, idx)

        rgb = np.stack([base, base, base], axis=-1)
        for organ, label in organ_orders.items():
            color = np.asarray(organ_rgb[organ], dtype=np.float32) / 255.0
            mask = seg2d == label
            rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color

        ax.imshow(rgb)
        ax.set_title(f"{mode} slice={idx}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
