"""Mod-Cog (Khona, Chandra et al. 2023): its Cog20 implementation plus "int"/"seq" variants."""

from . import _tasks

# Cog20 (Yang et al. 2019), in paper order, as implemented by Mod-Cog.
COG20: tuple[str, ...] = (
    "go", "rtgo", "dlygo", "anti", "rtanti", "dlyanti",
    "dm1", "dm2", "ctxdm1", "ctxdm2", "multidm",
    "dlydm1", "dlydm2", "ctxdlydm1", "ctxdlydm2", "multidlydm",
    "dms", "dnms", "dmc", "dnmc",
)  # fmt: skip

# Tasks with a delay period, which accept the "int" extension.
INT_BASE: tuple[str, ...] = (
    "dlygo", "dlyanti", "dlydm1", "dlydm2", "ctxdlydm1", "ctxdlydm2",
    "multidlydm", "dms", "dnms", "dmc", "dnmc",
)  # fmt: skip

# The 82-task set of the Mod-Cog README, in its order: 20 base, 22 int, 40 seq.
MODCOG: tuple[str, ...] = (
    COG20
    + tuple(f"{t}int{d}" for t in INT_BASE for d in "rl")
    + tuple(f"{t}seq{d}" for d in "rl" for t in COG20)
)

FACTORIES = {name: getattr(_tasks, name) for name in MODCOG}

__all__ = ["COG20", "FACTORIES", "INT_BASE", "MODCOG"]
