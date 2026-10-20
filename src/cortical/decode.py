"""Decode stimulus information from regions of the cortical sheet over time.

Built for one question: in the patch-wired network, can the ring-2 stimulus
angle be read out linearly from the ring-1 patch, which ring 2's inputs never
reach directly?
"""

import numpy as np

from cogtasks import list_tasks, make_task
from cogtasks.modcog._envs import _DelayMatch1DResponse, _Reach

N_RING = 16
_RING_ANGLES = 2 * np.pi * np.arange(N_RING) / N_RING


def _single_bump_tasks() -> tuple[str, ...]:
    """Tasks whose one stimulus sits on ring 1 or ring 2 (go/anti and match families)."""
    return tuple(
        name
        for name in list_tasks("modcog")
        if isinstance(make_task("modcog", name).unwrapped, _Reach | _DelayMatch1DResponse)
    )


SINGLE_BUMP_TASKS: tuple[str, ...] = _single_bump_tasks()


def stimulus_angle(ring: np.ndarray) -> float:
    """Circular mean of a 16-unit ring input, (16,) or summed over (T, 16); in [0, 2π)."""
    weights = np.asarray(ring, dtype=float).reshape(-1, N_RING).sum(axis=0)
    angle = float(np.mod(np.angle(np.sum(weights * np.exp(1j * _RING_ANGLES))), 2 * np.pi))
    return 0.0 if angle >= 2 * np.pi else angle  # mod of a tiny negative rounds to 2π
