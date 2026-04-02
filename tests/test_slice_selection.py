import numpy as np

from src.visualization import select_informative_slice_index


def test_select_prefers_three_masks_run_center() -> None:
    seg = np.zeros((10, 4, 4), dtype=np.uint8)
    # all three masks appear on slices 4,5,6 (continuous)
    for z in [4, 5, 6]:
        seg[z, 0, 0] = 1
        seg[z, 1, 1] = 2
        seg[z, 2, 2] = 3

    idx = select_informative_slice_index(seg, axis=0, labels=[1, 2, 3])
    assert idx == 5


def test_select_two_masks_when_no_three_masks() -> None:
    seg = np.zeros((8, 4, 4), dtype=np.uint8)
    # two-mask run on slices 1,2
    for z in [1, 2]:
        seg[z, 0, 0] = 1
        seg[z, 1, 1] = 2

    idx = select_informative_slice_index(seg, axis=0, labels=[1, 2, 3])
    assert idx in [1, 2]


def test_select_max_voxel_slice_when_only_single_mask() -> None:
    seg = np.zeros((6, 5, 5), dtype=np.uint8)
    seg[1, 0:2, 0:2] = 1   # 4 vox
    seg[4, 0:3, 0:3] = 1   # 9 vox

    idx = select_informative_slice_index(seg, axis=0, labels=[1, 2, 3])
    assert idx == 4
