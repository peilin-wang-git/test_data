from __future__ import annotations

import SimpleITK as sitk


def to_binary_image(image: sitk.Image) -> sitk.Image:
    """Convert any label image to binary UInt8."""
    out = sitk.Cast(image > 0, sitk.sitkUInt8)
    out.CopyInformation(image)
    return out


def generate_staple(mask_images: list[sitk.Image], threshold: float = 0.5) -> sitk.Image:
    """Generate binary STAPLE consensus from binary masks."""
    if not mask_images:
        raise ValueError("Empty STAPLE input")
    binary = [to_binary_image(x) for x in mask_images]
    prob = sitk.STAPLE(binary)
    out = sitk.Cast(prob >= float(threshold), sitk.sitkUInt8)
    out.CopyInformation(binary[0])
    return out
