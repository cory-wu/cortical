"""Independent, reproducible RNG streams for (nested) neurogym envs."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeAlias

import numpy as np
from neurogym.core import TrialEnv, TrialWrapper
from neurogym.wrappers.block import ScheduleEnvs

Env: TypeAlias = TrialEnv | TrialWrapper


def child_seeds(seed: int, n: int) -> list[int]:
    """``n`` independent integer seeds derived from ``seed``."""
    return [int(s) for s in np.random.SeedSequence(seed).generate_state(n)]


def optional_child_seeds(seed: int | None, n: int) -> list[int | None]:
    """Like ``child_seeds``, but ``n`` Nones when unseeded."""
    return [None] * n if seed is None else list(child_seeds(seed, n))


def seed_env(env: Env, seed: int) -> None:
    """Seed ``env`` so that every sub-env and schedule gets its own RNG stream.

    ``ScheduleEnvs.seed`` would give all members and the schedule the same
    stream, correlating them, so nested schedules are seeded member by member.
    """
    if isinstance(env, ScheduleEnvs):
        *member_seeds, schedule_seed = child_seeds(seed, len(env.envs) + 1)
        for member, member_seed in zip(env.envs, member_seeds, strict=True):
            seed_env(member, member_seed)
        env.schedule.seed(schedule_seed)
    else:
        env.unwrapped.seed(seed)


@contextmanager
def global_numpy_seed(seed: int) -> Iterator[None]:
    """Temporarily seed ``np.random``; some Mod-Cog tasks draw from it at construction."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
