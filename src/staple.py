from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk


def _to_binary_sitk(img: sitk.Image) -> sitk.Image:
    """Cast image into UInt8 binary mask preserving geometry."""
    binary = sitk.Cast(img > 0, sitk.sitkUInt8)
    binary.CopyInformation(img)
    return binary


def generate_staple_image(images: Iterable[sitk.Image], threshold: float = 0.5) -> sitk.Image:
    """Generate a binary STAPLE consensus image from input masks."""
    imgs = [_to_binary_sitk(i) for i in images]
    if len(imgs) == 0:
        raise ValueError("STAPLE requires at least one input mask.")

    prob = sitk.STAPLE(imgs)
    consensus = sitk.Cast(prob >= threshold, sitk.sitkUInt8)
    consensus.CopyInformation(imgs[0])
    return consensus


def sitk_to_numpy(img: sitk.Image) -> np.ndarray:
    """Convert SimpleITK image to numpy array in z,y,x order."""
    return sitk.GetArrayFromImage(img)


def save_staple_image(img: sitk.Image, output_path: Path) -> None:
    """Persist STAPLE image to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(output_path))
