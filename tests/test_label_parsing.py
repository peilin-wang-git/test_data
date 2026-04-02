import numpy as np

from src.label_parser import extract_organ_mask, organ_present


def test_extract_organ_mask() -> None:
    seg = np.array([[0, 1, 2], [3, 1, 0]])
    prostate = extract_organ_mask(seg, 1)
    assert prostate.sum() == 2


def test_organ_present() -> None:
    seg = np.array([[0, 0], [0, 3]])
    assert organ_present(seg, 3) is True
    assert organ_present(seg, 2) is False
