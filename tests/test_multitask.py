"""Interleaved multitask envs and supervised datasets."""

from cogtasks import list_tasks, make_dataset, make_multitask

OB_DIM = 33
ACT_DIM = 17


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
    env = make_multitask("cog20", seed=0)
    assert env.observation_space.shape == (OB_DIM + 20,)


def test_dataset_returns_time_major_batches():
    ds = make_dataset("cog20", batch_size=4, seq_len=50, seed=0)
    inputs, labels = ds()
    assert inputs.shape == (50, 4, OB_DIM + 20)
    assert labels.shape == (50, 4)
    assert ds.env.action_space.n == ACT_DIM
