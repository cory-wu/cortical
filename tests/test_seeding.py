"""A seed fixes construction and every trial, without touching global state."""

import numpy as np
import pytest

from cogtasks import make_dataset, make_multitask, make_task


@pytest.mark.parametrize("name", ["dlydm1", "dlygointr", "ctxdm2seql", "dmcseqr"])
def test_task_seed_is_reproducible(name):
    def sample(seed):
        env = make_task("modcog", name, seed=seed)
        trials = []
        for _ in range(5):
            env.new_trial()
            trials.append((env.unwrapped.ob.copy(), env.unwrapped.gt.copy()))
        return trials

    for (ob_a, gt_a), (ob_b, gt_b) in zip(sample(1), sample(1), strict=True):
        np.testing.assert_array_equal(ob_a, ob_b)
        np.testing.assert_array_equal(gt_a, gt_b)


def test_multitask_seed_is_reproducible():
    def sample():
        env = make_multitask("modcog", seed=2)
        env.reset()
        obs = []
        for _ in range(20):
            env.new_trial()
            obs.append(env.unwrapped.ob.copy())
        return obs

    for a, b in zip(sample(), sample(), strict=True):
        np.testing.assert_array_equal(a, b)


def test_multitask_tasks_get_independent_rng_streams():
    env = make_multitask("modcog", tasks=("go", "anti"), seed=0)
    draws = [e.unwrapped.rng.rand() for e in env.envs]
    assert draws[0] != draws[1]


def test_dataset_seed_is_reproducible():
    a = make_dataset("modcog", batch_size=4, seq_len=100, seed=3)
    b = make_dataset("modcog", batch_size=4, seq_len=100, seed=3)
    for _ in range(2):
        (xa, ya), (xb, yb) = a(), b()
        np.testing.assert_array_equal(xa, xb)
        np.testing.assert_array_equal(ya, yb)


def test_dataset_batch_elements_are_independent():
    inputs, _ = make_dataset("modcog", batch_size=4, seq_len=200, seed=0)()
    for i in range(1, 4):
        assert not np.array_equal(inputs[:, 0], inputs[:, i])


def test_seeding_leaves_global_numpy_state_untouched():
    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)
    make_task("modcog", "dms", seed=0)
    assert np.random.rand() == expected
