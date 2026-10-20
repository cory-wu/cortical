"""Train a masked LM-RNN on Mod-Cog with Khona et al.'s setup.

    python -m cortical.train --condition patches --seed 0

Writes ``<out>/<condition>_s<seed>/``:
- ``metrics.json``: config, per-step loss, and periodic evaluations
- ``checkpoint.pt``: for resuming
- ``model.pt``: the final model

Re-running the same command resumes an interrupted run, and does nothing for
a finished one.
"""

import argparse
import json
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cogtasks import make_dataset

from .evaluate import build_eval_set, score, summarize
from .model import MaskedRNN
from .wiring import LAYOUT_B, LAYOUT_B_LARGE, Layout, input_mask

log = logging.getLogger(__name__)

CONDITIONS: Mapping[str, Layout | None] = {
    "baseline": None,
    "patches": LAYOUT_B,
    "patches_large": LAYOUT_B_LARGE,
}
BATCH_SIZE = 4
SEQ_LEN = 350
LR = 1e-3
N_OUTPUT = 17
LOG_EVERY = 10
DEFAULT_OUT = Path("results/02_input_patches")


def pick_device() -> str:
    """The fastest available device: an NVIDIA GPU, then Apple's GPU, then the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def _save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(".pt.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _resolve_layout(condition: str, L: int, override: Layout | None) -> Layout | None:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; choose from {sorted(CONDITIONS)}.")
    if override is not None:
        return override
    layout = CONDITIONS[condition]
    if layout is not None and L != 50:
        raise ValueError(f"Layout for {condition!r} is defined on a 50x50 sheet, got L={L}.")
    return layout


def _load_metrics(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _check_same_run(metrics: Mapping[str, Any], condition: str, L: int, d: float) -> None:
    """Refuse to continue a run directory that was trained with other settings."""
    requested = {"condition": condition, "L": L, "d": d}
    saved = {key: metrics["config"][key] for key in requested}
    if saved != requested:
        raise ValueError(
            f"Run directory holds a run with {saved}, not {requested}; use another --out."
        )


def _is_complete(metrics: Mapping[str, Any] | None, steps: int) -> bool:
    return bool(metrics and metrics["done"] and metrics["evals"][-1]["step"] >= steps)


def _evaluate(model: MaskedRNN, eval_set, device: str, step: int) -> dict[str, Any]:
    model.eval()
    acc = score(model, eval_set, device)
    model.train()
    summary = summarize(acc)
    log.info("step %d: accuracy %s", step, {k: round(v, 3) for k, v in summary.items()})
    return {"step": step, "acc": acc, "summary": summary}


def train(
    condition: str,
    seed: int,
    out: Path,
    steps: int = 10_000,
    L: int = 50,
    d: float = 10,
    eval_every: int = 500,
    eval_trials: int = 32,
    device: str | None = None,
    _layout: Layout | None = None,
) -> Path:
    """Train (or resume) one run; returns its directory."""
    layout = _resolve_layout(condition, L, _layout)
    device = device or pick_device()
    run_dir = Path(out) / f"{condition}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path, ckpt_path = run_dir / "metrics.json", run_dir / "checkpoint.pt"

    metrics = _load_metrics(metrics_path)
    if metrics is not None:
        _check_same_run(metrics, condition, L, d)
    if _is_complete(metrics, steps):
        log.info("%s is complete; nothing to do.", run_dir)
        return run_dir

    torch.manual_seed(seed)
    model = MaskedRNN(L, d, input_mask(L, layout), N_OUTPUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    eval_set = build_eval_set(n_trials=eval_trials)

    if metrics is not None and ckpt_path.exists():
        # Load onto the CPU: load_state_dict copies into the on-device tensors, and
        # the RNG state must stay a CPU ByteTensor.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        torch.set_rng_state(ckpt["rng"])
        step = ckpt["step"]
        metrics["loss"] = [entry for entry in metrics["loss"] if entry[0] <= step]
        metrics["evals"] = [e for e in metrics["evals"] if e["step"] <= step]
        metrics["config"]["steps"] = steps
        data_seed = seed * 1_000_003 + step
        log.info("Resuming %s from step %d.", run_dir, step)
    else:
        step, data_seed = 0, seed
        config = {
            "condition": condition, "seed": seed, "steps": steps, "L": L, "d": d,
            "eval_every": eval_every, "eval_trials": eval_trials, "device": device,
            "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN, "lr": LR,
            "n_input_weights": int(model.in_mask.sum()),
            "n_recurrent_weights": int(model.rec_mask.sum()),
        }  # fmt: skip
        metrics = {"config": config, "loss": [], "evals": [], "done": False}
        metrics["evals"].append(_evaluate(model, eval_set, device, step))

    dataset = make_dataset("modcog", batch_size=BATCH_SIZE, seq_len=SEQ_LEN, seed=data_seed)
    while step < steps:
        inputs, labels = dataset()
        x = torch.from_numpy(inputs).float().to(device)
        y = torch.from_numpy(labels).long().to(device)
        loss = F.cross_entropy(model(x).reshape(-1, N_OUTPUT), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.apply_masks()
        step += 1
        metrics["loss"].append([step, float(loss)])
        if step % LOG_EVERY == 0:
            log.info("step %d: loss %.4f", step, float(loss))
        if step % eval_every == 0 or step == steps:
            metrics["evals"].append(_evaluate(model, eval_set, device, step))
            _save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "rng": torch.get_rng_state(), "step": step},
                ckpt_path,
            )  # fmt: skip
            metrics["done"] = step == steps
            _write_json(metrics_path, metrics)

    _save(model.state_dict(), run_dir / "model.pt")
    log.info("Finished %s.", run_dir)
    return run_dir


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--d", type=float, default=10, help="recurrent locality radius")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--device", default=None, help="e.g. cuda, cuda:1, mps, cpu (default: auto-pick)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    train(args.condition, args.seed, args.out, steps=args.steps, d=args.d, device=args.device)


if __name__ == "__main__":
    main()
