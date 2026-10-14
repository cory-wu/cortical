"""Mod-Cog task set (Khona, Chandra et al. 2023): Cog20 plus "int" and "seq" variants."""

from . import _tasks

BASE_TASKS: tuple[str, ...] = (
    "go", "rtgo", "dlygo", "anti", "rtanti", "dlyanti",
    "dm1", "dm2", "ctxdm1", "ctxdm2", "multidm",
    "dlydm1", "dlydm2", "ctxdlydm1", "ctxdlydm2", "multidlydm",
    "dms", "dnms", "dmc", "dnmc",
)  # fmt: skip

# Tasks with a delay period, which accept the "int" extension.
INT_TASKS: tuple[str, ...] = (
    "dlygo", "dlyanti", "dlydm1", "dlydm2", "ctxdlydm1", "ctxdlydm2",
    "multidlydm", "dms", "dnms", "dmc", "dnmc",
)  # fmt: skip

# The 82-task set of the Mod-Cog README, in its order: 20 base, 22 int, 40 seq.
TASKS: tuple[str, ...] = (
    BASE_TASKS
    + tuple(f"{t}int{d}" for t in INT_TASKS for d in "rl")
    + tuple(f"{t}seq{d}" for d in "rl" for t in BASE_TASKS)
)

# Upstream "int + seq" combinations, defined but not part of the 82-task set.
INTSEQ_TASKS: tuple[str, ...] = tuple(f"{t}intseq" for t in INT_TASKS)

FACTORIES = {name: getattr(_tasks, name) for name in TASKS + INTSEQ_TASKS}

__all__ = ["BASE_TASKS", "FACTORIES", "INTSEQ_TASKS", "INT_TASKS", "TASKS"]
