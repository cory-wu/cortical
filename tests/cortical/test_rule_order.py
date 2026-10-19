"""Training streams put the rule one-hot at the generating task's list_tasks index.

The eval set (cortical.evaluate) assumes this order; if make_multitask ever
reordered its members, evaluation would silently test the wrong rule.
"""

from cogtasks import list_tasks, make_multitask, make_task
from cortical.evaluate import N_OBS

# Constructor settings that tell the Mod-Cog task variants apart.
_SIGNATURE = (
    "anti",
    "reaction",
    "integrate",
    "seq",
    "w_mod1",
    "w_mod2",
    "stim_mod1",
    "stim_mod2",
    "delaycomparison",
    "matchto",
    "matchgo",
)


def _signature(env):
    task = env.unwrapped
    return (type(task).__name__, *(getattr(task, a, None) for a in _SIGNATURE))


def _leaves(env):
    if hasattr(env, "envs"):
        return [leaf for member in env.envs for leaf in _leaves(member)]
    return [env.unwrapped]


def test_training_rule_index_names_the_generating_task():
    tasks = list_tasks("modcog")
    env = make_multitask("modcog", seed=0)
    env.reset()
    seen = set()
    for _ in range(200):
        env.new_trial()
        k = int(env.unwrapped.ob[0, N_OBS:].argmax())
        assert any(env.unwrapped is leaf for leaf in _leaves(env.envs[k]))
        assert _signature(env.envs[k]) == _signature(make_task("modcog", tasks[k]))
        seen.add(k)
    assert len(seen) > 40  # the check covered many different tasks
