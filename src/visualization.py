from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _normalize_image(image_2d: np.ndarray) -> np.ndarray:
    img = image_2d.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-8:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def _get_slice(arr_3d: np.ndarray, mode: str) -> np.ndarray:
    z, y, x = arr_3d.shape
    if mode == "axial":
        return arr_3d[z // 2, :, :]
    if mode == "coronal":
        return arr_3d[:, y // 2, :]
    if mode == "sagittal":
        return arr_3d[:, :, x // 2]
    raise ValueError(f"Unsupported slice mode: {mode}")


def save_overlay_preview(
    image_arr: np.ndarray,
    seg_arr: np.ndarray,
    organ_rgb: dict[str, tuple[int, int, int]],
    organ_orders: dict[str, int],
    out_path: Path,
    slice_mode: str = "axial",
    alpha: float = 0.4,
) -> None:
    """Save a single 2D overlay preview (middle slice) to PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = _normalize_image(_get_slice(image_arr, slice_mode))
    seg2d = _get_slice(seg_arr, slice_mode)

    rgb = np.stack([base, base, base], axis=-1)
    for organ, label in organ_orders.items():
        color = np.asarray(organ_rgb[organ], dtype=np.float32) / 255.0
        mask = seg2d == label
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
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
    """Save axial/coronal/sagittal overlay preview in one PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    modes = ["axial", "coronal", "sagittal"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, mode in zip(axes, modes):
        base = _normalize_image(_get_slice(image_arr, mode))
        seg2d = _get_slice(seg_arr, mode)
        rgb = np.stack([base, base, base], axis=-1)
        for organ, label in organ_orders.items():
            color = np.asarray(organ_rgb[organ], dtype=np.float32) / 255.0
            mask = seg2d == label
            rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color
        ax.imshow(rgb)
        ax.set_title(mode)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
