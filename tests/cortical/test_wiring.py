"""Input groups, layout B, and the hard input mask."""

from itertools import combinations

import numpy as np

from cortical.sheet import square_patch
from cortical.wiring import INPUT_GROUPS, LAYOUT_B, N_INPUT, input_mask

L = 50


def _patch(group: str) -> np.ndarray:
    centre, size = LAYOUT_B[group]
    return square_patch(L, centre, size)


def test_groups_cover_all_inputs_once():
    covered = sorted(i for s in INPUT_GROUPS.values() for i in range(s.start, s.stop))
    assert covered == list(range(N_INPUT))
    assert N_INPUT == 115


def test_layout_b_patch_sizes():
    sizes = {g: len(_patch(g)) for g in LAYOUT_B}
    assert sizes == {"fixation": 36, "ring1": 144, "ring2": 144, "rule": 256}
    for a, b in combinations(LAYOUT_B, 2):
        assert not set(_patch(a)) & set(_patch(b))


def test_each_group_reaches_exactly_its_patch():
    mask = input_mask(L, LAYOUT_B)
    for group, rows in INPUT_GROUPS.items():
        indicator = np.zeros(L * L, dtype=bool)
        indicator[_patch(group)] = True
        assert (mask[rows] == indicator).all()


def test_patches_mask_nonzeros():
    assert input_mask(L, LAYOUT_B).sum() == 25_636


def test_dense_mask():
    mask = input_mask(L, None)
    assert mask.shape == (115, 2500)
    assert mask.all()
