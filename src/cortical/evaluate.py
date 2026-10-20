"""Strict per-trial accuracy on fixed, stream-like evaluation sets.

Training sequences are continuous streams of trials, so evaluation runs each
task as a stream too: ``n_trials`` back to back from a zero state, the hidden
state carrying over between trials, each trial scored on its own.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from neurogym.core import TrialEnv
from torch import nn

from cogtasks import list_tasks, make_task

from .wiring import N_INPUT

N_OBS = 33  # fixation + two stimulus rings; the rule one-hot follows


@dataclass(frozen=True)
class EvalSet:
    tasks: tuple[str, ...]
    inputs: np.ndarray  # (T_max, n_tasks, N_INPUT) float32, zero-padded at the end
    targets: np.ndarray  # (T_max, n_tasks) int64
    trials: tuple[tuple[tuple[int, int, int], ...], ...]  # per task: (start, decision_start, end)


def iter_trials(
    name: str, k: int, n_trials: int, seed: int
) -> Iterator[tuple[np.ndarray, np.ndarray, TrialEnv, int]]:
    """Yield ``(inputs with rule unit k, targets, task, i_env)`` for each trial of ``name``.

    ``task`` is the env that generated the trial; read its state before the next
    trial replaces it. ``i_env`` is the sub-env index (the stimulus modality for
    two-modality tasks), or 0 for tasks without one.
    """
    env = make_task("modcog", name, seed=seed)
    for _ in range(n_trials):
        env.new_trial()
        task = env.unwrapped
        ob = np.zeros((len(task.gt), N_INPUT), dtype=np.float32)
        ob[:, :N_OBS] = task.ob
        ob[:, N_OBS + k] = 1.0
        yield ob, task.gt.astype(np.int64), task, getattr(env, "i_env", 0)


def _task_stream(name: str, k: int, n_trials: int, seed: int):
    """Inputs, targets and trial bounds for ``n_trials`` of task ``name``, rule unit ``k``."""
    obs, gts, bounds, t = [], [], [], 0
    for ob, gt, task, _ in iter_trials(name, k, n_trials, seed):
        obs.append(ob)
        gts.append(gt)
        bounds.append((t, t + task.start_ind["decision"], t + len(gt)))
        t += len(gt)
    return np.concatenate(obs), np.concatenate(gts), tuple(bounds)


def build_eval_set(
    n_trials: int = 32, seed: int = 1234, tasks: Sequence[str] | None = None
) -> EvalSet:
    """Fixed streams for ``tasks`` (default: all of Mod-Cog).

    Each task keeps its index ``k`` in ``list_tasks("modcog")`` for its rule
    unit and its seed (``seed + k``), matching the training rule order.
    """
    order = list_tasks("modcog")
    names = order if tasks is None else tuple(tasks)
    streams = [_task_stream(n, order.index(n), n_trials, seed + order.index(n)) for n in names]
    t_max = max(len(gt) for _, gt, _ in streams)
    inputs = np.zeros((t_max, len(names), N_INPUT), dtype=np.float32)
    targets = np.zeros((t_max, len(names)), dtype=np.int64)
    for k, (ob, gt, _) in enumerate(streams):
        inputs[: len(gt), k] = ob
        targets[: len(gt), k] = gt
    return EvalSet(tuple(names), inputs, targets, tuple(b for _, _, b in streams))


def trial_correct(pred: np.ndarray, target: np.ndarray, decision_start: int) -> bool:
    """Fixation (0) before the response period, then the target at every response step."""
    holds_fixation = bool((pred[:decision_start] == 0).all())
    responds = bool((pred[decision_start:] == target[decision_start:]).all())
    return holds_fixation and responds


@torch.no_grad()
def score(model: nn.Module, eval_set: EvalSet, device: str = "cpu") -> dict[str, float]:
    """Accuracy per task: the fraction of its trials that are fully correct."""
    logits = model(torch.from_numpy(eval_set.inputs).to(device))
    pred = logits.argmax(dim=-1).cpu().numpy()
    acc = {}
    for k, (name, trials) in enumerate(zip(eval_set.tasks, eval_set.trials, strict=True)):
        correct = [
            trial_correct(pred[s:e, k], eval_set.targets[s:e, k], ds - s) for s, ds, e in trials
        ]
        acc[name] = float(np.mean(correct))
    return acc


def summarize(acc: Mapping[str, float]) -> dict[str, float]:
    """Mean accuracy over all tasks and per family (base, int, seq)."""

    def family(name: str) -> str:
        if name.endswith(("seqr", "seql")):
            return "seq"
        return "int" if name.endswith(("intr", "intl")) else "base"

    summary = {"all": float(np.mean(list(acc.values())))}
    for fam in ("base", "int", "seq"):
        values = [v for name, v in acc.items() if family(name) == fam]
        summary[fam] = float(np.mean(values)) if values else float("nan")
    return summary
