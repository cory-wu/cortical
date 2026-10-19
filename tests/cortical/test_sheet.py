"""Torus geometry of the cortical sheet and its recurrent locality mask."""

import math

import numpy as np

from cortical.sheet import coords, locality_mask, square_patch, torus_distance


def test_coords_row_major():
    xy = coords(50)
    assert tuple(xy[51]) == (1, 1)
    assert tuple(xy[50]) == (0, 1)


def test_distance_wraps():
    dist = torus_distance(50)
    assert dist[0, 49] == 1.0  # (0, 0) and (49, 0) are neighbours across the edge
    assert math.isclose(dist[0, 25 * 50 + 25], math.hypot(25, 25))


def test_locality_mask_counts_match_khona():
    assert locality_mask(50, 6).sum() == 282_500
    assert locality_mask(50, 10).sum() == 792_500


def test_locality_mask_symmetric_with_self_connections():
    mask = locality_mask(50, 6)
    assert mask.dtype == bool
    assert (mask == mask.T).all()
    assert mask.diagonal().all()


def test_square_patch_ring1():
    units = square_patch(50, (12.5, 12.5), 12)
    xy = coords(50)[units]
    assert len(units) == 144
    assert xy.min() == 7 and xy.max() == 18
    assert (np.diff(units) > 0).all()  # sorted, unique


def test_square_patch_wraps_edge():
    units = square_patch(50, (0.5, 0.5), 4)
    xy = coords(50)[units]
    assert len(units) == 16
    assert set(xy[:, 0]) == set(xy[:, 1]) == {49, 0, 1, 2}
