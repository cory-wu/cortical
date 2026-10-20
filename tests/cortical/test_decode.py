"""Decoding ring-2 stimulus information from sheet regions."""

import numpy as np
import pytest

from cogtasks import make_task
from cogtasks.modcog._envs import _DelayMatch1DResponse, _gaussianbump, _Reach
from cortical.decode import SINGLE_BUMP_TASKS, stimulus_angle

THETA = 2 * np.pi * np.arange(16) / 16


def _circ_err(a, b):
    d = np.abs(np.mod(a - b, 2 * np.pi))
    return np.minimum(d, 2 * np.pi - d)


def test_single_bump_tasks():
    assert len(SINGLE_BUMP_TASKS) == 42
    for name in SINGLE_BUMP_TASKS:
        assert isinstance(make_task("modcog", name).unwrapped, _Reach | _DelayMatch1DResponse)
    assert "dm2" not in SINGLE_BUMP_TASKS and "multidm" not in SINGLE_BUMP_TASKS


@pytest.mark.parametrize("centre", [0.0, 0.3, np.pi, 5.9])
def test_stimulus_angle_recovers_bump_centre(centre):
    bump = _gaussianbump(centre, THETA, 1)
    assert _circ_err(stimulus_angle(bump), centre) < np.deg2rad(1)
    assert _circ_err(stimulus_angle(np.tile(bump, (5, 1))), centre) < np.deg2rad(1)
    assert 0 <= stimulus_angle(bump) < 2 * np.pi
