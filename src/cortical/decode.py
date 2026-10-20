"""Decode stimulus information from regions of the cortical sheet over time.

Built for one question: in the patch-wired network, can the ring-2 stimulus
angle be read out linearly from the ring-1 patch, which ring 2's inputs never
reach directly?
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from cogtasks import list_tasks, make_task
from cogtasks.modcog._envs import _DelayMatch1DResponse, _Reach

from .evaluate import iter_trials
from .model import MaskedRNN

N_RING = 16
_RING_ANGLES = 2 * np.pi * np.arange(N_RING) / N_RING


def _single_bump_tasks() -> tuple[str, ...]:
    """Tasks whose one stimulus sits on ring 1 or ring 2 (go/anti and match families)."""
    return tuple(
        name
        for name in list_tasks("modcog")
        if isinstance(make_task("modcog", name).unwrapped, _Reach | _DelayMatch1DResponse)
    )


SINGLE_BUMP_TASKS: tuple[str, ...] = _single_bump_tasks()


def stimulus_angle(ring: np.ndarray) -> float:
    """Circular mean of a 16-unit ring input, (16,) or summed over (T, 16); in [0, 2π)."""
    weights = np.asarray(ring, dtype=float).reshape(-1, N_RING).sum(axis=0)
    angle = float(np.mod(np.angle(np.sum(weights * np.exp(1j * _RING_ANGLES))), 2 * np.pi))
    return 0.0 if angle >= 2 * np.pi else angle  # mod of a tiny negative rounds to 2π


WINDOW = (-2, 20)  # steps relative to stimulus onset, inclusive
RING_INPUTS = {1: slice(1, 17), 2: slice(17, 33)}  # observation indices of each ring
# The first stimulus period: the sample in match tasks, the stimulus in go/anti
# tasks, and the decision period in reaction-time tasks (stimulus and go cue together).
_ONSET_PERIODS = ("sample", "stimulus", "decision")


def _true_angle(task) -> float:
    """The first stimulus's angle, from the trial record (exact, unlike the noisy input)."""
    if isinstance(task, _DelayMatch1DResponse):
        angle = task.trial["sample_theta"]
    else:  # _Reach: the stimulus sits at the target, shifted for int and anti variants
        trial = task.trial
        shift = int(trial["integrate"]) * task.omega * task.timing["delay"]
        angle = task.theta[trial["ground_truth"]] + shift + int(trial["anti"]) * np.pi
    return float(np.mod(angle, 2 * np.pi))


@dataclass(frozen=True)
class TrialSet:
    task: np.ndarray  # (n,) task names
    modality: np.ndarray  # (n,) 1 or 2: which ring holds the stimulus
    angle: np.ndarray  # (n,) first-stimulus angle in [0, 2π)
    activity: np.ndarray  # (n, window, len(units)) float32, zero where invalid
    valid: np.ndarray  # (n, window) bool: the step lies inside the trial
    units: np.ndarray  # recorded unit indices


def collect(
    model: MaskedRNN,
    units: np.ndarray,
    n_trials: int = 128,
    seed: int = 4321,
    tasks: Sequence[str] = SINGLE_BUMP_TASKS,
    device: str = "cpu",
) -> TrialSet:
    """Run each task as a stream of trials and keep activity windows around stimulus onset."""
    order = list_tasks("modcog")
    lo, hi = WINDOW
    offsets = np.arange(lo, hi + 1)
    names, modalities, angles, windows, valids = [], [], [], [], []
    for name in tasks:
        k = order.index(name)
        obs, onsets, ends, t = [], [], [], 0
        for ob, _, task, i_env in iter_trials(name, k, n_trials, seed + k):
            period = next(p for p in _ONSET_PERIODS if p in task.start_ind)
            start = task.start_ind[period]
            modality = i_env + 1
            names.append(name)
            modalities.append(modality)
            angles.append(_true_angle(task))
            onsets.append(t + start)
            ends.append(t + len(ob))
            obs.append(ob)
            t += len(ob)
        x = torch.from_numpy(np.concatenate(obs))[:, None].to(device)
        with torch.no_grad():
            hidden, _ = model.rnn(x)
        h = hidden[:, 0, torch.as_tensor(units, device=device)].cpu().numpy()
        for onset, end in zip(onsets, ends, strict=True):
            steps = onset + offsets
            inside = (steps >= 0) & (steps < end)
            window = np.zeros((len(offsets), len(units)), dtype=np.float32)
            window[inside] = h[steps[inside]]
            windows.append(window)
            valids.append(inside)
    return TrialSet(
        task=np.array(names),
        modality=np.array(modalities),
        angle=np.array(angles),
        activity=np.stack(windows),
        valid=np.stack(valids),
        units=np.asarray(units),
    )
