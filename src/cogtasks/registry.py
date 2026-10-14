"""Look up and build tasks from the registered collections."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from types import MappingProxyType
from typing import TypeAlias

import neurogym as ngym
import numpy as np
from neurogym.core import TrialEnv, TrialWrapper
from neurogym.utils.scheduler import RandomSchedule
from neurogym.wrappers.block import ScheduleEnvs

from . import modcog, yang19

Env: TypeAlias = TrialEnv | TrialWrapper

COLLECTIONS: MappingProxyType[str, MappingProxyType[str, Callable[..., Env]]] = MappingProxyType(
    {
        "yang19": MappingProxyType(yang19.FACTORIES),
        "modcog": MappingProxyType({n: modcog.FACTORIES[n] for n in modcog.TASKS}),
        "modcog_intseq": MappingProxyType({n: modcog.FACTORIES[n] for n in modcog.INTSEQ_TASKS}),
    }
)


def _collection(collection: str) -> MappingProxyType[str, Callable[..., Env]]:
    try:
        return COLLECTIONS[collection]
    except KeyError:
        msg = f"Unknown collection {collection!r}; choose from {sorted(COLLECTIONS)}."
        raise ValueError(msg) from None


def _child_seeds(seed: int, n: int) -> list[int]:
    """``n`` independent integer seeds derived from ``seed``."""
    return [int(s) for s in np.random.SeedSequence(seed).generate_state(n)]


def _optional_child_seeds(seed: int | None, n: int) -> list[int | None]:
    """Like ``_child_seeds``, but ``n`` Nones when unseeded."""
    return [None] * n if seed is None else list(_child_seeds(seed, n))


def seed_env(env: Env, seed: int) -> None:
    """Seed ``env`` so that every sub-env and schedule gets its own RNG stream.

    ``ScheduleEnvs.seed`` would give all members and the schedule the same
    stream, correlating them, so nested schedules are seeded member by member.
    """
    if isinstance(env, ScheduleEnvs):
        *member_seeds, schedule_seed = _child_seeds(seed, len(env.envs) + 1)
        for member, member_seed in zip(env.envs, member_seeds):
            seed_env(member, member_seed)
        env.schedule.seed(schedule_seed)
    else:
        env.unwrapped.seed(seed)


def _leaves(env: Env) -> list[TrialEnv]:
    """The base TrialEnvs under ``env``, descending through nested schedules."""
    if isinstance(env, ScheduleEnvs):
        return [leaf for member in env.envs for leaf in _leaves(member)]
    return [env.unwrapped]


@contextmanager
def _global_numpy_seed(seed: int) -> Iterator[None]:
    """Temporarily seed ``np.random``; some Mod-Cog tasks draw from it at construction."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def list_tasks(collection: str) -> tuple[str, ...]:
    """Task names of ``collection``, in canonical order."""
    return tuple(_collection(collection))


def make_task(collection: str, name: str, seed: int | None = None, **kwargs) -> Env:
    """Build one task. ``kwargs`` (e.g. ``dt``, ``timing``) go to its factory.

    With ``seed``, construction and all trials are reproducible. Note that
    Mod-Cog ``*seq*`` factories overwrite any ``timing`` passed to them.
    """
    factories = _collection(collection)
    if name not in factories:
        msg = f"Unknown task {name!r} in collection {collection!r}; see list_tasks({collection!r})."
        raise ValueError(msg)
    if seed is None:
        return factories[name](**kwargs)
    construct_seed, trial_seed = _child_seeds(seed, 2)
    with _global_numpy_seed(construct_seed):
        env = factories[name](**kwargs)
    seed_env(env, trial_seed)
    return env


def make_multitask(
    collection: str,
    tasks: Sequence[str] | None = None,
    seed: int | None = None,
    env_input: bool = True,
    **kwargs,
) -> ScheduleEnvs:
    """Interleave tasks at random, trial by trial.

    With ``env_input`` a one-hot rule input (one unit per task, in ``tasks``
    order) is appended to the observation, as in Yang et al. 2019.
    """
    names = list_tasks(collection) if tasks is None else tuple(tasks)
    *task_seeds, schedule_seed = _optional_child_seeds(seed, len(names) + 1)
    envs = [make_task(collection, n, seed=s, **kwargs) for n, s in zip(names, task_seeds)]
    schedule = RandomSchedule(len(envs))
    schedule.seed(schedule_seed)
    env = ScheduleEnvs(envs, schedule=schedule, env_input=env_input)
    # ScheduleEnvs only re-points the first leaf of each member at itself, so
    # trials that other leaves of nested schedules (Mod-Cog's two-modality
    # tasks) start from step()/reset() would bypass the rule input.
    for leaf in _leaves(env):
        leaf.set_top(env)
    return env


def make_dataset(
    collection: str,
    batch_size: int,
    seq_len: int,
    tasks: Sequence[str] | None = None,
    seed: int | None = None,
    batch_first: bool = False,
    **kwargs,
) -> ngym.Dataset:
    """Supervised dataset over a multitask env; ``ds()`` returns ``(inputs, labels)``."""
    env_seeds = _optional_child_seeds(seed, batch_size)
    envs = [make_multitask(collection, tasks=tasks, seed=s, **kwargs) for s in env_seeds]
    # Dataset deep-copies the env it is given and resets the copies from OS
    # entropy. Deep copies keep drawing timings from the original's RNG via
    # ``lambda: self.rng.uniform(...)`` entries, so hand it a throwaway
    # template, then swap in the independent envs and refill the cache.
    template = make_multitask(collection, tasks=tasks, seed=seed, **kwargs)
    ds = ngym.Dataset(template, batch_size=batch_size, seq_len=seq_len, batch_first=batch_first)
    ds.envs = envs
    ds.env = envs[0]
    for env in envs:
        env.reset()
    ds._cache()
    return ds
