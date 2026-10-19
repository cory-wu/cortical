"""Training CLI: outputs, checkpoint/resume, no retraining, device fallback."""

import json

import pytest
import torch

from cortical import train as train_mod
from cortical.train import main, pick_device, train

SMALL = {"L": 10, "d": 2, "eval_every": 2, "eval_trials": 1, "device": "cpu"}


def _metrics(run_dir):
    return json.loads((run_dir / "metrics.json").read_text())


def test_smoke_run_writes_outputs(tmp_path, small_layout):
    run_dir = train("patches", 0, tmp_path, steps=3, _layout=small_layout, **SMALL)
    assert run_dir == tmp_path / "patches_s0"
    metrics = _metrics(run_dir)
    assert metrics["done"] is True
    assert [e["step"] for e in metrics["evals"]] == [0, 2, 3]
    assert set(metrics["evals"][-1]["summary"]) == {"all", "base", "int", "seq"}
    assert len(metrics["evals"][-1]["acc"]) == 82
    assert (run_dir / "model.pt").exists() and (run_dir / "checkpoint.pt").exists()


def test_resume_continues_from_checkpoint(tmp_path, small_layout):
    first = _metrics(train("patches", 0, tmp_path, steps=2, _layout=small_layout, **SMALL))
    run_dir = train("patches", 0, tmp_path, steps=3, _layout=small_layout, **SMALL)
    metrics = _metrics(run_dir)
    assert [e["step"] for e in metrics["evals"]] == [0, 2, 3]
    assert [step for step, _ in metrics["loss"]] == [1, 2, 3]
    # Resumed after step 2: earlier entries are untouched, not replayed.
    assert metrics["loss"][:2] == first["loss"]
    assert metrics["evals"][:2] == first["evals"]
    assert torch.load(run_dir / "checkpoint.pt", weights_only=False)["step"] == 3
    assert metrics["done"] is True


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_resume_on_mps(tmp_path, small_layout):
    """Loading with map_location="mps" moves the RNG state off the CPU and crashes."""
    on_mps = {**SMALL, "device": "mps"}
    train("patches", 0, tmp_path, steps=2, _layout=small_layout, **on_mps)
    run_dir = train("patches", 0, tmp_path, steps=3, _layout=small_layout, **on_mps)
    assert [e["step"] for e in _metrics(run_dir)["evals"]] == [0, 2, 3]


def test_completed_run_is_not_retrained(tmp_path, small_layout):
    run_dir = train("patches", 0, tmp_path, steps=3, _layout=small_layout, **SMALL)
    stamp = (run_dir / "metrics.json").stat().st_mtime_ns
    train("patches", 0, tmp_path, steps=3, _layout=small_layout, **SMALL)
    assert (run_dir / "metrics.json").stat().st_mtime_ns == stamp


def test_patches_requires_full_sheet(tmp_path):
    with pytest.raises(ValueError, match="50"):
        train("patches", 0, tmp_path, steps=1, **SMALL)


def test_pick_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pick_device() == "cpu"


def test_cli_rejects_unknown_condition():
    with pytest.raises(SystemExit):
        main(["--condition", "nope"])


def test_conditions_are_baseline_and_patches():
    assert set(train_mod.CONDITIONS) == {"baseline", "patches"}
    assert train_mod.CONDITIONS["baseline"] is None
