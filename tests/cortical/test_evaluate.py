"""Fixed evaluation streams and the strict per-trial score."""

import numpy as np
import pytest
import torch
from torch import nn

from cogtasks import list_tasks
from cortical.evaluate import EvalSet, build_eval_set, score, summarize, trial_correct

ALL_TASKS = list_tasks("modcog")
TARGET = np.array([0, 0, 0, 7, 8, 9])  # three fixation steps, then a response


@pytest.mark.parametrize(
    ("pred", "target", "expected"),
    [
        (TARGET, TARGET, True),  # perfect
        ([0, 0, 0, 0, 8, 9], TARGET, False),  # response one step late
        ([0, 5, 0, 7, 8, 9], TARGET, False),  # fixation broken
        ([0, 0, 0, 0, 3, 0], np.zeros(6), False),  # no-go trial, responds
        (np.zeros(6), np.zeros(6), True),  # no-go trial, fixates throughout
        ([0, 0, 0, 7, 2, 9], TARGET, False),  # one wrong response step
    ],
)
def test_trial_correct_cases(pred, target, expected):
    assert trial_correct(np.asarray(pred), np.asarray(target), decision_start=3) is expected


def test_eval_set_shapes_and_boundaries():
    es = build_eval_set(n_trials=4, tasks=("go", "dms"))
    t_max = es.inputs.shape[0]
    assert es.tasks == ("go", "dms")
    assert es.inputs.shape == (t_max, 2, 115) and es.inputs.dtype == np.float32
    assert es.targets.shape == (t_max, 2) and es.targets.dtype == np.int64
    for trials in es.trials:
        assert len(trials) == 4 and trials[0][0] == 0
        for start, decision_start, end in trials:
            assert start < decision_start < end <= t_max
        for (_, _, end), (next_start, _, _) in zip(trials, trials[1:], strict=False):
            assert end == next_start


def test_eval_rule_input_matches_training_order():
    es = build_eval_set(n_trials=2)
    assert es.tasks == ALL_TASKS
    for k, trials in enumerate(es.trials):
        stream = es.inputs[: trials[-1][2], k, 33:]
        assert (stream.argmax(axis=1) == k).all() and (stream.sum(axis=1) == 1).all()


def test_eval_set_is_reproducible():
    a, b = build_eval_set(n_trials=3, tasks=("dlygointr",)), build_eval_set(n_trials=3, tasks=("dlygointr",))
    np.testing.assert_array_equal(a.inputs, b.inputs)
    np.testing.assert_array_equal(a.targets, b.targets)
    assert a.trials == b.trials


class _Oracle(nn.Module):
    """Returns the eval set's targets as one-hot logits, and a wrong class on padding."""

    def __init__(self, es: EvalSet) -> None:
        super().__init__()
        labels = es.targets.copy()
        for k, trials in enumerate(es.trials):
            labels[trials[-1][2] :, k] = 5  # padding: would fail if it were scored
        self.logits = nn.functional.one_hot(torch.from_numpy(labels), 17).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits[: x.shape[0]]


def test_padding_does_not_change_score():
    padded = build_eval_set(4, tasks=("go", "dms", "dlygoseqr"))
    assert score(_Oracle(padded), padded) == {"go": 1.0, "dms": 1.0, "dlygoseqr": 1.0}
    alone = build_eval_set(4, tasks=("dms",))
    assert score(_Oracle(alone), alone) == {"dms": 1.0}


def test_summarize_families():
    acc = {t: 0.0 if "seq" in t else 0.5 if t.endswith(("intr", "intl")) else 1.0 for t in ALL_TASKS}
    summary = summarize(acc)
    assert summary["base"] == 1.0 and summary["int"] == 0.5 and summary["seq"] == 0.0
    assert summary["all"] == pytest.approx((20 * 1.0 + 22 * 0.5) / 82)
