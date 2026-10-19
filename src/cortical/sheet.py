"""Geometry of an LxL cortical sheet with periodic boundaries (a torus).

Unit ``i`` sits at lattice point ``(x, y) = (i % L, i // L)``.
"""

import numpy as np


def coords(L: int) -> np.ndarray:
    """(L², 2) integer (x, y) coordinates, row ``y * L + x``."""
    y, x = np.divmod(np.arange(L * L), L)
    return np.stack([x, y], axis=1)


def _torus_offsets(a: np.ndarray, b: np.ndarray, L: int) -> np.ndarray:
    """Per-axis shortest offsets |a - b| on a ring of length L."""
    delta = np.abs(a - b)
    return np.minimum(delta, L - delta)


def torus_distance(L: int) -> np.ndarray:
    """(L², L²) Euclidean distances on the torus."""
    xy = coords(L)
    offsets = _torus_offsets(xy[:, None, :], xy[None, :, :], L)
    return np.hypot(offsets[..., 0], offsets[..., 1])


def locality_mask(L: int, d: float) -> np.ndarray:
    """(L², L²) mask of unit pairs within distance ``d`` (inclusive, self included)."""
    return torus_distance(L) <= d


def square_patch(L: int, centre: tuple[float, float], size: int) -> np.ndarray:
    """Sorted indices of units with torus offset ``< size / 2`` from ``centre`` on both axes."""
    offsets = _torus_offsets(coords(L).astype(float), np.asarray(centre, dtype=float), L)
    return np.flatnonzero((offsets < size / 2).all(axis=1))
