from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class OrganSpec:
    name: str
    order: int
    rgb: tuple[int, int, int]


def organ_specs_from_config(config: dict) -> list[OrganSpec]:
    """Read organ specs sorted by order from config."""
    organs_cfg = config["organs"]
    specs = [
        OrganSpec(name=name, order=int(info["order"]), rgb=tuple(info["RGB"]))
        for name, info in organs_cfg.items()
    ]
    return sorted(specs, key=lambda x: x.order)


def extract_organ_mask(multiclass_arr: np.ndarray, label_value: int) -> np.ndarray:
    """Extract binary mask for one label value."""
    return (multiclass_arr == label_value).astype(np.uint8)


def organ_present(multiclass_arr: np.ndarray, label_value: int) -> bool:
    """Check if organ label exists in segmentation."""
    return bool(np.any(multiclass_arr == label_value))


def compose_multiclass_from_binary(
    organ_binary_images: dict[str, sitk.Image],
    ordered_specs: list[OrganSpec],
) -> sitk.Image:
    """Combine binary organ masks into one multi-class image."""
    if not organ_binary_images:
        raise ValueError("No organ masks to compose")

    ref = next(iter(organ_binary_images.values()))
    out_arr = np.zeros(sitk.GetArrayFromImage(ref).shape, dtype=np.uint8)
    for spec in ordered_specs:
        if spec.name not in organ_binary_images:
            continue
        arr = sitk.GetArrayFromImage(organ_binary_images[spec.name]) > 0
        out_arr[arr] = np.uint8(spec.order)

    out = sitk.GetImageFromArray(out_arr)
    out.CopyInformation(ref)
    return out
