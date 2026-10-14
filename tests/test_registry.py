import numpy as np
import pytest

import cogtasks
from cogtasks import COLLECTIONS, list_tasks, make_dataset, make_multitask, make_task

YANG19 = (
    "go", "rtgo", "dlygo", "anti", "rtanti", "dlyanti",
    "dm1", "dm2", "ctxdm1", "ctxdm2", "multidm",
    "dlydm1", "dlydm2", "ctxdlydm1", "ctxdlydm2", "multidlydm",
    "dms", "dnms", "dmc", "dnmc",
)
N_RING = 16
OB_DIM = 1 + 2 * N_RING  # fixation + two stimulus modalities
ACT_DIM = 1 + N_RING  # fixation + ring of choices


def test_collections_are_registered():
    assert set(COLLECTIONS) >= {"yang19", "modcog"}


def test_yang19_has_the_twenty_cog_tasks_in_paper_order():
    assert list_tasks("yang19") == YANG19


def test_modcog_has_82_unique_tasks_starting_with_the_cog20_base():
    tasks = list_tasks("modcog")
    assert len(tasks) == 82
    assert len(set(tasks)) == 82
    assert tasks[:20] == YANG19
    assert sum(t.endswith(("intr", "intl")) for t in tasks) == 22
    assert sum(t.endswith(("seqr", "seql")) for t in tasks) == 40


def test_unknown_collection_raises():
    with pytest.raises(ValueError, match="Unknown collection"):
        list_tasks("nope")


def test_unknown_task_raises_with_collection_name():
    with pytest.raises(ValueError, match="modcog"):
        make_task("modcog", "nope")


@pytest.mark.parametrize(
    ("collection", "name"),
    [(c, n) for c in ("yang19", "modcog", "modcog_intseq") for n in list_tasks(c)],
)
def test_every_task_builds_and_produces_valid_trials(collection, name):
    env = make_task(collection, name, seed=0)
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


@pytest.mark.parametrize(
    ("collection", "name"),
    [(c, n) for c in ("yang19", "modcog") for n in list_tasks(c)],
)
def test_following_ground_truth_earns_reward(collection, name):
    """Stepping with the target action through a trial never fails the trial."""
    env = make_task(collection, name, seed=0)
    env.reset()
    for _ in range(5):
        total, done = 0.0, False
        while not done:
            gt = env.unwrapped.gt_now
            _, reward, terminated, truncated, info = env.step(gt)
            assert not (terminated or truncated)
            total += reward
            done = info["new_trial"]
        assert total >= 0  # no abort penalty while fixating on cue
        if info["gt"] != 0:  # trials with a response must be scored correct
            assert info["performance"] == 1


@pytest.mark.parametrize(
    ("collection", "name"),
    [("yang19", "dlydm1"), ("modcog", "dlygointr"), ("modcog", "ctxdm2seql"), ("modcog", "dmcseqr")],
)
def test_seed_makes_trials_reproducible(collection, name):
    def sample(seed):
        env = make_task(collection, name, seed=seed)
        trials = []
        for _ in range(5):
            env.new_trial()
            trials.append((env.unwrapped.ob.copy(), env.unwrapped.gt.copy()))
        return trials

    for (ob_a, gt_a), (ob_b, gt_b) in zip(sample(1), sample(1)):
        np.testing.assert_array_equal(ob_a, ob_b)
        np.testing.assert_array_equal(gt_a, gt_b)


@pytest.mark.parametrize(("name", "step"), [("goseqr", 1), ("goseql", -1)])
def test_seq_target_rotates_one_unit_per_step(name, step):
    env = make_task("modcog", name, seed=0)
    env.new_trial()
    task = env.unwrapped
    gt = task.gt[task.start_ind["decision"] : task.end_ind["decision"]]
    choices = gt - 1  # choice units start after fixation
    np.testing.assert_array_equal(np.diff(choices) % N_RING, step % N_RING)


def test_multitask_seed_is_reproducible():
    def sample():
        env = make_multitask("modcog", seed=2)
        env.reset()
        obs = []
        for _ in range(20):
            env.new_trial()
            obs.append(env.unwrapped.ob.copy())
        return obs

    for a, b in zip(sample(), sample()):
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


def test_multitask_appends_one_hot_rule_inputs():
    names = ("go", "dm1", "dms")
    env = make_multitask("modcog", tasks=names, seed=0)
    assert env.observation_space.shape == (OB_DIM + len(names),)
    env.reset()
    rule = env.unwrapped.ob[:, OB_DIM:]
    assert (rule.sum(axis=1) == 1).all()


def test_multitask_rule_input_survives_stepping():
    """Trials started by step() (not just new_trial) carry the rule input too."""
    n_tasks = len(list_tasks("modcog"))
    env = make_multitask("modcog", seed=0)
    ob, _ = env.reset()
    for _ in range(2000):
        assert ob.shape == (OB_DIM + n_tasks,)
        assert ob[OB_DIM:].sum() == 1
        ob, *_ = env.step(env.unwrapped.gt_now)


def test_multitask_defaults_to_the_whole_collection():
    env = make_multitask("yang19", seed=0)
    assert env.observation_space.shape == (OB_DIM + 20,)


def test_dataset_returns_time_major_batches():
    ds = make_dataset("yang19", batch_size=4, seq_len=50, seed=0)
    inputs, labels = ds()
    assert inputs.shape == (50, 4, OB_DIM + 20)
    assert labels.shape == (50, 4)
    assert ds.env.action_space.n == ACT_DIM


def test_package_exposes_version():
    assert isinstance(cogtasks.__version__, str)
