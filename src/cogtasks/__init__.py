"""Cog20 (Yang et al. 2019) and Mod-Cog (Khona et al. 2023), both as implemented by Mod-Cog."""

from .registry import COLLECTIONS, list_tasks, make_dataset, make_multitask, make_task

__version__ = "0.1.0"

__all__ = ["COLLECTIONS", "list_tasks", "make_dataset", "make_multitask", "make_task"]
