import numpy as np

from src.metrics import dice_coefficient


def test_dice_identical() -> None:
    a = np.array([[0, 1], [1, 0]])
    b = np.array([[0, 1], [1, 0]])
    assert dice_coefficient(a, b) == 1.0


def test_dice_disjoint() -> None:
    a = np.array([[1, 0], [0, 0]])
    b = np.array([[0, 1], [0, 0]])
    assert dice_coefficient(a, b) == 0.0


def test_dice_both_empty() -> None:
    a = np.zeros((2, 2), dtype=np.uint8)
    b = np.zeros((2, 2), dtype=np.uint8)
    assert dice_coefficient(a, b) == 1.0
