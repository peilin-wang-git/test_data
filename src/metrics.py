from __future__ import annotations

import numpy as np


def dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute binary Dice coefficient with strict shape checking."""
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    a_sum = int(a.sum())
    b_sum = int(b.sum())
    if a_sum == 0 and b_sum == 0:
        return 1.0
    if a_sum == 0 or b_sum == 0:
        return 0.0

    inter = int(np.logical_and(a, b).sum())
    return float(2.0 * inter / (a_sum + b_sum))
