"""Cognitive task collections: Cog20 (Yang et al. 2019) and Mod-Cog (Khona et al. 2023)."""

from .registry import COLLECTIONS, list_tasks, make_dataset, make_multitask, make_task

__version__ = "0.1.0"

__all__ = ["COLLECTIONS", "list_tasks", "make_dataset", "make_multitask", "make_task"]
