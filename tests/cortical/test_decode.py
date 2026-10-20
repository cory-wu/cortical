"""Decoding ring-2 stimulus information from sheet regions."""

import numpy as np
import pytest

from cogtasks import make_task
from cogtasks.modcog._envs import _DelayMatch1DResponse, _gaussianbump, _Reach
from cortical.decode import (
    SINGLE_BUMP_TASKS,
    circular_error,
    collect,
    decode_over_time,
    regions,
    stimulus_angle,
)
from cortical.evaluate import iter_trials
from cortical.model import MaskedRNN
from cortical.sheet import square_patch, torus_distance
from cortical.wiring import LAYOUT_B_LARGE, input_mask

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


SMALL_TASKS = ("go", "rtanti", "dms", "dlygointr")
N_PER_TASK = 8
RINGS = {1: slice(1, 17), 2: slice(17, 33)}


@pytest.fixture(scope="module")
def trialset():
    model = MaskedRNN(10, 2, input_mask(10, None))
    return collect(
        model, units=np.arange(0, 100, 7), n_trials=N_PER_TASK, seed=11, tasks=SMALL_TASKS
    )


def _stream_trials(seed=11):
    """Re-generate the same trials collect() saw, with their onsets, ends and records."""
    from cogtasks import list_tasks

    order = list_tasks("modcog")
    out = []
    for name in SMALL_TASKS:
        k = order.index(name)
        for ob, _gt, task, i_env in iter_trials(name, k, N_PER_TASK, seed + k):
            first = next(p for p in ("sample", "stimulus", "decision") if p in task.start_ind)
            onset, stop = task.start_ind[first], task.end_ind[first]
            out.append((name, ob, onset, stop, dict(task.trial), dict(task.timing), i_env))
    return out


def test_trialset_shapes(trialset):
    n = len(SMALL_TASKS) * N_PER_TASK
    assert trialset.activity.shape == (n, 23, 15)
    assert trialset.valid.shape == (n, 23)
    assert trialset.valid[:, 2].all()  # tau = 0 is always inside the trial
    assert len(trialset.task) == len(trialset.modality) == len(trialset.angle) == n


def test_onset_is_first_stimulus_step(trialset):
    for (_, ob, onset, _, _, _, i_env), modality in zip(
        _stream_trials(), trialset.modality, strict=True
    ):
        ring = ob[:, RINGS[modality]]
        assert ring[onset].sum() > 0
        assert ring[onset - 1].sum() == 0
        assert modality == i_env + 1


def test_modality_matches_stimulated_ring(trialset):
    for (_, ob, onset, *_), modality in zip(_stream_trials(), trialset.modality, strict=True):
        other = RINGS[3 - modality]
        assert ob[onset, RINGS[modality]].sum() > 0
        assert ob[onset, other].sum() == 0


def test_angle_matches_trial_records(trialset):
    for (name, _, _, _, trial, timing, _), angle in zip(
        _stream_trials(), trialset.angle, strict=True
    ):
        if name == "dms":
            expected = trial["sample_theta"]
        else:
            shift = int(trial["integrate"]) * 0.01 * timing["delay"] + int(trial["anti"]) * np.pi
            expected = THETA[trial["ground_truth"]] + shift
        assert _circ_err(angle, np.mod(expected, 2 * np.pi)) < np.deg2rad(1), name


def test_angle_agrees_with_input_circular_mean(trialset):
    """Independent check on the record lookup: noise-free bumps match exactly; the
    match tasks' sample carries input noise, so they only need to be close."""
    for (name, ob, onset, stop, *_), angle, modality in zip(
        _stream_trials(), trialset.angle, trialset.modality, strict=True
    ):
        tolerance = 10 if name == "dms" else 1
        observed = stimulus_angle(ob[onset:stop, RINGS[modality]])
        assert _circ_err(observed, angle) < np.deg2rad(tolerance), name


def test_both_modalities_occur(trialset):
    assert set(trialset.modality.tolist()) == {1, 2}


@pytest.fixture(scope="module")
def region_units():
    return regions(50, LAYOUT_B_LARGE)


def test_region_sizes(region_units):
    assert set(region_units) == {"ring1", "ring2", "ring1_self", "ring1_near", "matched"}
    for name in ("ring1", "ring2", "ring1_self"):
        assert len(region_units[name]) == 324
    assert len(region_units["matched"]) == len(region_units["ring1_near"]) >= 100


def test_patches_match_layout(region_units):
    for ring in ("ring1", "ring2"):
        np.testing.assert_array_equal(region_units[ring], square_patch(50, *LAYOUT_B_LARGE[ring]))
    np.testing.assert_array_equal(region_units["ring1_self"], region_units["ring1"])
    assert set(region_units["ring1_near"]) <= set(region_units["ring1"])


def test_matched_region_distances(region_units):
    """matched is input-free, unique, and pairs with ring1_near at equal distance from ring 2."""
    matched, near = region_units["matched"], region_units["ring1_near"]
    in_patches = np.concatenate([square_patch(50, c, s) for c, s in LAYOUT_B_LARGE.values()])
    assert not set(matched) & set(in_patches)
    assert len(set(matched)) == len(matched)
    to_ring2 = torus_distance(50)[:, region_units["ring2"]].min(axis=1)
    # Sorted distances pair up within the 0.5 matching tolerance.
    assert np.abs(np.sort(to_ring2[matched]) - np.sort(to_ring2[near])).max() <= 0.5


def _synthetic(n=400, units=50, steps=23, encoded_from=5, seed=0):
    """Activity that linearly encodes (cos θ, sin θ) from step `encoded_from` onward."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, n)
    weights = rng.normal(size=(2, units))
    activity = 0.3 * rng.normal(size=(n, steps, units))
    signal = np.stack([np.cos(angles), np.sin(angles)], axis=1) @ weights
    activity[:, encoded_from:] += signal[:, None, :]
    groups = np.repeat(np.arange(4), n // 4)
    return activity, angles, np.ones((n, steps), dtype=bool), groups


@pytest.fixture(scope="module")
def synthetic_result():
    activity, angles, valid, groups = _synthetic()
    return decode_over_time(activity, angles, valid, groups, n_shuffles=5, seed=1)


def test_recovers_encoded_angle(synthetic_result):
    assert (synthetic_result.accuracy[5:] >= 0.9).all()


def test_chance_without_encoding(synthetic_result):
    assert (synthetic_result.accuracy[:5] <= 0.25).all()


def test_null_stays_near_chance(synthetic_result):
    assert (synthetic_result.null95 <= 0.25).all()


def test_invalid_steps_excluded():
    activity, angles, valid, groups = _synthetic()
    valid[::2, 15:] = False
    activity[~valid] = 0.0
    result = decode_over_time(activity, angles, valid, groups, n_shuffles=2, seed=1)
    assert (result.n_trials[15:] == 200).all() and (result.n_trials[:15] == 400).all()
    assert (result.accuracy[15:] >= 0.9).all()
    assert (np.isnan(result.predicted) == ~valid).all()
    assert result.tau.tolist() == list(range(-2, 21))


def test_circular_error_wraps():
    assert np.isclose(circular_error(np.array([0.1]), np.array([2 * np.pi - 0.1]))[0], 0.2)
