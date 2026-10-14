"""Cog20 task set (Yang et al. 2019), served from neurogym's ``yang19`` collection."""

from neurogym.envs.collections import yang19 as _yang19

TASKS: tuple[str, ...] = (
    "go", "rtgo", "dlygo", "anti", "rtanti", "dlyanti",
    "dm1", "dm2", "ctxdm1", "ctxdm2", "multidm",
    "dlydm1", "dlydm2", "ctxdlydm1", "ctxdlydm2", "multidlydm",
    "dms", "dnms", "dmc", "dnmc",
)  # fmt: skip

FACTORIES = {name: getattr(_yang19, name) for name in TASKS}

__all__ = ["FACTORIES", "TASKS"]
