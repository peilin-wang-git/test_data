import numpy as np

from src.metrics import dice_coefficient


def test_dice_identical_masks_is_one() -> None:
    a = np.array([[0, 1], [1, 1]])
    b = np.array([[0, 1], [1, 1]])
    assert dice_coefficient(a, b) == 1.0


def test_dice_non_overlap_is_zero() -> None:
    a = np.array([[1, 0], [0, 0]])
    b = np.array([[0, 1], [0, 0]])
    assert dice_coefficient(a, b) == 0.0


def test_dice_both_empty_is_one() -> None:
    a = np.zeros((2, 2), dtype=int)
    b = np.zeros((2, 2), dtype=int)
    assert dice_coefficient(a, b) == 1.0


def test_dice_shape_mismatch_raises() -> None:
    a = np.zeros((2, 2), dtype=int)
    b = np.zeros((3, 3), dtype=int)
    try:
        _ = dice_coefficient(a, b)
        assert False, "Expected ValueError"
    except ValueError:
        assert True
