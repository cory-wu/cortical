"""Which sheet units each input group connects to.

Observation layout (``cogtasks`` Mod-Cog multitask env): fixation, two
16-unit stimulus rings, then the 82-task one-hot rule input.
"""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from .sheet import square_patch

N_INPUT = 115

INPUT_GROUPS: Mapping[str, slice] = MappingProxyType(
    {
        "fixation": slice(0, 1),
        "ring1": slice(1, 17),
        "ring2": slice(17, 33),
        "rule": slice(33, 115),
    }
)

Layout = Mapping[str, tuple[tuple[float, float], int]]

# Patches on a 2x2 lattice 25 apart on the 50x50 torus, rings at opposite
# corners: group -> (centre (x, y), side length).
LAYOUT_B: Layout = MappingProxyType(
    {
        "fixation": ((12.5, 37.5), 6),
        "ring1": ((12.5, 12.5), 12),
        "ring2": ((37.5, 37.5), 12),
        "rule": ((37.5, 12.5), 16),
    }
)


# Layout B scaled up about 1.5x per side (sides kept even, since the centres are
# half-integers): 1,324 units, 53% of the sheet.
LAYOUT_B_LARGE: Layout = MappingProxyType(
    {
        "fixation": ((12.5, 37.5), 10),
        "ring1": ((12.5, 12.5), 18),
        "ring2": ((37.5, 37.5), 18),
        "rule": ((37.5, 12.5), 24),
    }
)


def input_mask(L: int, layout: Layout | None) -> np.ndarray:
    """(N_INPUT, L²) mask: each group's inputs reach only its patch; ``None`` is dense."""
    if layout is None:
        return np.ones((N_INPUT, L * L), dtype=bool)
    mask = np.zeros((N_INPUT, L * L), dtype=bool)
    for group, rows in INPUT_GROUPS.items():
        centre, size = layout[group]
        mask[rows, square_patch(L, centre, size)] = True
    return mask
