"""Every task builds, produces well-formed trials and scores correct behaviour."""

import numpy as np
import pytest

from cogtasks import list_tasks, make_task

N_RING = 16
OB_DIM = 1 + 2 * N_RING  # fixation + two stimulus modalities
ACT_DIM = 1 + N_RING  # fixation + ring of choices
ALL_TASKS = list_tasks("modcog")  # a superset of cog20


@pytest.mark.parametrize("name", ALL_TASKS)
def test_task_builds_and_produces_valid_trials(name):
    env = make_task("modcog", name, seed=0)
    assert env.observation_space.shape == (OB_DIM,)
    assert env.action_space.n == ACT_DIM
    for _ in range(3):
        env.new_trial()
        ob, gt = env.unwrapped.ob, env.unwrapped.gt
        assert ob.shape[1] == OB_DIM
        assert len(ob) == len(gt) > 0
        assert np.isfinite(ob).all()
        assert ((gt >= 0) & (gt < ACT_DIM)).all()
        assert gt[0] == 0  # trials begin with fixation


@pytest.mark.parametrize("name", ALL_TASKS)
def test_following_ground_truth_earns_reward(name):
    """Stepping with the target action through a trial never fails the trial."""
    env = make_task("modcog", name, seed=0)
    env.reset()
    for _ in range(5):
        total, done = 0.0, False
        while not done:
            _, reward, terminated, truncated, info = env.step(env.unwrapped.gt_now)
            assert not (terminated or truncated)
            total += reward
            done = info["new_trial"]
        assert total >= 0  # no abort penalty while fixating on cue
        if info["gt"] != 0:  # trials with a response must be scored correct
            assert info["performance"] == 1


@pytest.mark.parametrize(("name", "step"), [("goseqr", 1), ("goseql", -1)])
def test_seq_target_rotates_one_unit_per_step(name, step):
    env = make_task("modcog", name, seed=0)
    env.new_trial()
    task = env.unwrapped
    gt = task.gt[task.start_ind["decision"] : task.end_ind["decision"]]
    choices = gt - 1  # choice units start after fixation
    np.testing.assert_array_equal(np.diff(choices) % N_RING, step % N_RING)
