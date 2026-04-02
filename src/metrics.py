from __future__ import annotations

import numpy as np


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert mask to boolean binary mask."""
    return np.asarray(mask) > 0


def dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute Dice coefficient for binary masks.

    Rules:
    - both empty => 1.0
    - one empty, one non-empty => 0.0
    """
    a = binarize_mask(mask_a)
    b = binarize_mask(mask_b)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for Dice: {a.shape} vs {b.shape}")

    a_sum = int(a.sum())
    b_sum = int(b.sum())

    if a_sum == 0 and b_sum == 0:
        return 1.0
    if a_sum == 0 or b_sum == 0:
        return 0.0

    intersection = int(np.logical_and(a, b).sum())
    return float(2.0 * intersection / (a_sum + b_sum))
