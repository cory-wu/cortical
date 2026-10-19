# Input Patches (Notebook 02) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Khona's LM-RNN (50×50 torus, d = 6) on Mod-Cog in two conditions, dense input wiring and input wiring hard-masked to sheet patches (layout B), and compare them in notebook 02.

**Architecture:**
- A new `src/cortical/` package has five focused modules:
  - `sheet` gives the geometry and recurrent mask.
  - `wiring` gives the input mask.
  - `model` is a masked `nn.RNN`.
  - `evaluate` builds fixed streams and the strict per-trial score.
  - `train` is the CLI with checkpoint/resume.
- The training runs happen in the background via the CLI.
- The notebook only loads `results/02_input_patches/`.

**Tech Stack:** Python 3.12, PyTorch 2.x (MPS), `cogtasks` (this repo), neurogym 2.3.1, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-10-19-input-patches-design.md`

## Global Constraints

- **Sheet:** L = 50 and N = 2500 on a torus. Unit index = y·50 + x. d = 6, with distance ≤ d inclusive and self-connections included.
- **Model:** ReLU `nn.RNN` with both biases, a dense `nn.Linear(N, 17)` readout and PyTorch default init. Masks are applied at init and after every optimizer step.
- **Training:** Adam with lr 1e-3, no clipping, no weight decay. Unweighted cross-entropy averaged over all steps. Batch 4, seq_len 350, 10,000 steps.
- **Data:** `cogtasks.make_dataset("modcog", batch_size=4, seq_len=350, seed=...)`. Observation dim 115, 17 actions.
- **Input groups (observation indices):** fixation 0, ring1 1–16, ring2 17–32, rule 33–114.
- **Layout B (centre (x, y), side length):**
  - fixation: (12.5, 37.5), 6
  - ring1: (12.5, 12.5), 12
  - ring2: (37.5, 37.5), 12
  - rule: (37.5, 12.5), 16
- **Patch membership:** a patch is all lattice points with |x − cx| < size/2 and |y − cy| < size/2.
- **Evaluation:**
  - 32 trials per task, run as one stream that starts from a zero state.
  - A trial is correct iff the argmax is 0 at every step before `decision` and equals `gt` at every `decision` step.
  - Evaluate at step 0 and every 500 steps, always on the same fixed eval set.
- **Output:** results go to `results/02_input_patches/<condition>_s<seed>/`, and `*.pt` files are git-ignored.
- **Runs:** `baseline` and `patches`, with seed 0 only.

## Review Focus

1. **`weight_ih` orientation.** `nn.RNN.weight_ih_l0` is (N, I), the transpose of `M` (I, N). A wrongly oriented mask would still zero 91% of the weights, just the wrong ones. Pinned by Task 3's `test_ring1_drives_only_its_patch`.
2. **Padding.** The 82 eval streams have different lengths. Padded steps must never be scored, and they must not change the result. Pinned by Task 4's `test_padding_does_not_change_score`.
3. **Re-running a finished run.** A second `train` call on a completed output directory must exit without training or overwriting `metrics.json`. Pinned by Task 5's `test_completed_run_is_not_retrained`.
4. **No MPS.** On machines without MPS, training falls back to CPU instead of crashing. Pinned by Task 5's `test_pick_device_falls_back_to_cpu`.
5. **Rule-input order.** The one-hot rule index in the eval streams must match the order `make_dataset` uses (`list_tasks("modcog")`). Otherwise evaluation tests the wrong rule. Pinned by Task 4's `test_eval_rule_input_matches_training_order`.

---

### Task 1: Sheet geometry and recurrent mask

**Files:**
- Create: `src/cortical/__init__.py` (docstring only), `src/cortical/sheet.py`
- Test: `tests/cortical/test_sheet.py`
- Modify: `pyproject.toml`. Under `[tool.hatch.build.targets.wheel]`, change `packages` to `["src/cogtasks", "src/cortical"]`.

**Interfaces:**
- Produces:
  - `coords(L: int) -> np.ndarray`: (L², 2) int array of (x, y), row = y·L + x.
  - `torus_distance(L: int) -> np.ndarray`: (L², L²) float distances.
  - `locality_mask(L: int, d: float) -> np.ndarray`: (L², L²) bool.
  - `square_patch(L: int, centre: tuple[float, float], size: int) -> np.ndarray`: sorted int unit indices.

- [ ] **Step 1: Write the failing tests** in `tests/cortical/test_sheet.py`:
  - `test_coords_row_major`: `coords(50)[51] == (1, 1)` and `coords(50)[50] == (0, 1)`.
  - `test_distance_wraps`: units (0, 0) and (49, 0) are 1.0 apart; (0, 0) and (25, 25) are `hypot(25, 25)` apart.
  - `test_locality_mask_counts_match_khona`: `locality_mask(50, 6).sum() == 282_500` and `locality_mask(50, 10).sum() == 792_500`.
  - `test_locality_mask_symmetric_with_self_connections`: the mask is symmetric and its diagonal is all true.
  - `test_square_patch_ring1`: `square_patch(50, (12.5, 12.5), 12)` has 144 units whose x and y both lie in 7..18.
  - `test_square_patch_wraps_edge`: `square_patch(50, (0.5, 0.5), 4)` has 16 units, with x and y in {49, 0, 1, 2}.
- [ ] **Step 2: Run them and confirm they fail.** `.venv/bin/python -m pytest tests/cortical/test_sheet.py -q` fails with `ModuleNotFoundError: cortical`.
- [ ] **Step 3: Implement `sheet.py`.** Build the torus distance with `np.minimum(|Δ|, L − |Δ|)` per axis, and membership as the torus offset |x − cx| < size/2 per axis.
- [ ] **Step 4: Re-install and run the tests.** Run `.venv/bin/pip install -q -e ".[dev,notebook]"`, then pytest. Expected: 6 passed.
- [ ] **Step 5: Commit.** `feat: cortical sheet geometry and locality mask`

### Task 2: Input groups, layout B and the input mask

**Files:**
- Create: `src/cortical/wiring.py`
- Test: `tests/cortical/test_wiring.py`

**Interfaces:**
- Consumes: `square_patch` (Task 1).
- Produces:
  - `INPUT_GROUPS: Mapping[str, slice]`: `{"fixation": slice(0,1), "ring1": slice(1,17), "ring2": slice(17,33), "rule": slice(33,115)}`.
  - `LAYOUT_B: Mapping[str, tuple[tuple[float, float], int]]`, with the values from Global Constraints.
  - `N_INPUT = 115`.
  - `input_mask(L: int, layout: Mapping | None) -> np.ndarray`: (N_INPUT, L²) bool. `None` means all true.

- [ ] **Step 1: Write the failing tests:**
  - `test_groups_cover_all_inputs_once`: the slices tile 0..114 without gaps.
  - `test_layout_b_patch_sizes`: the sizes are fixation 36, ring1 144, ring2 144, rule 256, and the patches are pairwise disjoint.
  - `test_each_group_reaches_exactly_its_patch`: for each group, every row of `input_mask(50, LAYOUT_B)[slice]` equals that patch's indicator vector.
  - `test_patches_mask_nonzeros`: `input_mask(50, LAYOUT_B).sum() == 25_636`.
  - `test_dense_mask`: `input_mask(50, None)` is all true, with shape (115, 2500).
- [ ] **Step 2: Run the tests and confirm they fail** with an import error.
- [ ] **Step 3: Implement `wiring.py`.** Use read-only `MappingProxyType` for both constants.
- [ ] **Step 4: Run the tests.** Expected: 5 passed.
- [ ] **Step 5: Commit.** `feat: input groups, layout B and input mask`

### Task 3: Masked RNN model

**Files:**
- Create: `src/cortical/model.py`
- Test: `tests/cortical/test_model.py`

**Interfaces:**
- Consumes: `locality_mask` (Task 1), `N_INPUT` (Task 2).
- Produces: `class MaskedRNN(nn.Module)`, with `__init__(self, L: int, d: float, input_mask: np.ndarray, n_output: int = 17)`.
  - Attributes: `self.rnn = nn.RNN(N_INPUT, L*L, nonlinearity="relu")` and `self.readout = nn.Linear(L*L, n_output)`.
  - Buffers: `rec_mask` (N, N) float and `in_mask` (N, I) float, which is `input_mask.T`.
  - `apply_masks() -> None`, decorated `@torch.no_grad`. It multiplies the masks into `weight_hh_l0` and `weight_ih_l0` in place. It runs at the end of `__init__`.
  - `forward(x: Tensor[T, B, I]) -> Tensor[T, B, n_output]`, starting from a zero state.

- [ ] **Step 1: Write the failing tests.** Use L = 10, d = 2 and `input_mask(10, small_layout)`. The `small_layout` fixture is `{"fixation": ((2.5, 7.5), 2), "ring1": ((2.5, 2.5), 4), "ring2": ((7.5, 7.5), 4), "rule": ((7.5, 2.5), 4)}`.
  - `test_masked_entries_zero_after_init`: zero exactly where the masks are false.
  - `test_masks_survive_adam_step`: one Adam step on random data, then `apply_masks()`, leaves the masked entries zero.
  - `test_ring1_drives_only_its_patch`: set all biases and `weight_hh_l0` to 0. An input with only index 1 on produces nonzero pre-activation only on ring1's patch units at t = 0. Check this via `weight_ih_l0[:, 1] != 0` equalling the patch indicator.
  - `test_forward_shape`: (7, 3, 115) maps to (7, 3, 17).
  - `test_parameter_count_matches_nn_rnn`: the total parameter count is `N*N + N*115 + 2*N + N*17 + 17`.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement `model.py`.**
- [ ] **Step 4: Run the tests.** Expected: 5 passed.
- [ ] **Step 5: Commit.** `feat: masked LM-RNN model`

### Task 4: Fixed eval set and strict per-trial score

**Files:**
- Create: `src/cortical/evaluate.py`
- Test: `tests/cortical/test_evaluate.py`

**Interfaces:**
- Consumes: `cogtasks.make_task`, `cogtasks.list_tasks`, `N_INPUT` (Task 2).
- Produces:
  - `@dataclass(frozen=True) class EvalSet` with fields:
    - `tasks: tuple[str, ...]`
    - `inputs: np.ndarray`: (T_max, n_tasks, 115) float32
    - `targets: np.ndarray`: (T_max, n_tasks) int64
    - `trials: tuple[tuple[tuple[int, int, int], ...], ...]`: per task, one `(start, decision_start, end)` per trial, as stream step indices
  - `build_eval_set(n_trials: int = 32, seed: int = 1234, tasks: Sequence[str] | None = None) -> EvalSet`
    - Per task k (in `list_tasks("modcog")` order), call `make_task("modcog", name, seed=seed + k)` and draw `n_trials` with `new_trial()`.
    - For each trial, append the one-hot rule column `33 + k` to the 33-dim observation.
    - Concatenate the trials into one stream and pad with zeros at the end.
  - `trial_correct(pred: np.ndarray, target: np.ndarray, decision_start: int) -> bool`: `pred` and `target` are 1-D, and the trial is correct iff `pred[:decision_start] == 0` everywhere and `pred[decision_start:] == target[decision_start:]` everywhere.
  - `score(model: nn.Module, eval_set: EvalSet, device: str = "cpu") -> dict[str, float]`. `model` maps (T, B, 115) to (T, B, 17) logits.: maps task name to accuracy, under `torch.no_grad`. Argmax is taken over the output dim.
  - `summarize(acc: Mapping[str, float]) -> dict[str, float]`: keys `"all"`, `"base"`, `"int"` (name ends in intr/intl) and `"seq"` (ends in seqr/seql).

- [ ] **Step 1: Write the failing tests:**
  - `test_trial_correct_cases`, a table of hand-built cases with `decision_start = 3`:
    - perfect output → True
    - response one step late (`pred[3] = 0`) → False
    - fixation broken (`pred[1] = 5`) → False
    - no-go trial (target all 0) with a response at step 4 → False
    - no-go trial fixating throughout → True
    - one wrong response step → False
  - `test_eval_set_shapes_and_boundaries`: build with `n_trials=4, tasks=("go", "dms")`. For each trial, `start < decision_start < end ≤ T_max`, and consecutive trials abut (`end` equals the next trial's `start`).
  - `test_eval_rule_input_matches_training_order`: for the task at position k of `list_tasks("modcog")`, every non-padded step has `inputs[:, k, 33:]` one-hot at `33 + k`.
  - `test_eval_set_is_reproducible`: two builds with the same seed are array-equal.
  - `test_padding_does_not_change_score`: an oracle stub `nn.Module` built from an `EvalSet` returns the one-hot of that set's `targets` as logits, ignoring its input. On `build_eval_set(4, tasks=("go", "dms", "dlygoseqr"))` it scores 1.0 for every task. A stub built from a set with only `"dms"` (no padding) also scores 1.0.
  - `test_summarize_families`: the counts behind the means are 20/22/40 for the full `list_tasks("modcog")`.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement `evaluate.py`.** Read each trial's `decision_start` from `task.unwrapped.start_ind["decision"]`. `score` runs all tasks as one batch and slices out each trial.
- [ ] **Step 4: Run the tests.** Expected: 6 passed.
- [ ] **Step 5: Commit.** `feat: strict per-trial evaluation on fixed streams`

### Task 5: Training CLI with checkpoint/resume

**Files:**
- Create: `src/cortical/train.py`
- Test: `tests/cortical/test_train.py`
- Modify: `.gitignore`. Add `results/**/*.pt`.

**Interfaces:**
- Consumes: everything from Tasks 1–4, plus `cogtasks.make_dataset`.
- Produces:
  - `CONDITIONS = {"baseline": None, "patches": LAYOUT_B}`.
  - `pick_device() -> str`: `"mps"` if `torch.backends.mps.is_available()`, else `"cpu"`.
  - `train(condition: str, seed: int, out: Path, steps: int = 10_000, L: int = 50, d: float = 6, eval_every: int = 500, eval_trials: int = 32, device: str | None = None, _layout: Mapping | None = None) -> Path`. `_layout` overrides `CONDITIONS[condition]`, and only tests use it. It returns the run directory `out / f"{condition}_s{seed}"`.
  - `main(argv: Sequence[str] | None = None) -> None`, an argparse CLI: `--condition {baseline,patches}` (required), `--seed` (default 0), `--steps`, `--out` (default `results/02_input_patches`).
  - Files in the run dir:
    - `metrics.json`: `{"config": {...}, "loss": [[step, loss], ...], "evals": [{"step": s, "acc": {task: acc}, "summary": {...}}], "done": bool}`.
    - `checkpoint.pt`: model, optimizer and step.
    - `model.pt`: written at the end.
- Behaviour:
  - Seed torch with `seed`.
  - Build the model with `input_mask(L, CONDITIONS[condition])`.
  - Data comes from `make_dataset("modcog", batch_size=4, seq_len=350, seed=seed)`. On resume at step s, use seed `seed * 1_000_003 + s`.
  - Loss is `F.cross_entropy(logits.reshape(-1, 17), labels.reshape(-1))`.
  - After each optimizer step, call `model.apply_masks()`.
  - Log the loss every 10 steps. Evaluate at step 0, every `eval_every` steps, and at the final step. Write a checkpoint and `metrics.json` at every evaluation, atomically: write to a temp file, then `os.replace`.
  - If `metrics.json` says `"done": true`, log and return without training.
  - Module-level `logging` only; no `print`.
  - When the patch layout is used, `L` must be 50, since layout B is defined for 50. Otherwise raise `ValueError`. The smoke test passes a scaled layout through a private `_layout` override.

- [ ] **Step 1: Write the failing tests.** Use `L=10, d=2, steps=3, eval_every=2, eval_trials=1, device="cpu"`, with the Task 3 `small_layout` passed via `_layout`.
  - `test_smoke_run_writes_outputs`: `metrics.json` has `done` true, contains evals at steps 0, 2 and 3, and `model.pt` exists.
  - `test_resume_continues_from_checkpoint`: run with `steps=2`, then call again with `steps=3`. The evals cover steps [0, 2, 3], and the loss log starts where it left off (no duplicate steps).
  - `test_completed_run_is_not_retrained`: call a finished run again. The modification time of `metrics.json` is unchanged.
  - `test_pick_device_falls_back_to_cpu`: monkeypatch `torch.backends.mps.is_available` to return False, and `pick_device() == "cpu"`.
  - `test_cli_rejects_unknown_condition`: `main(["--condition", "nope"])` raises `SystemExit`.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement `train.py`.** End the module with `if __name__ == "__main__": main()`.
- [ ] **Step 4: Run the full suite.** `.venv/bin/python -m pytest -q -p no:warnings` passes everything. Also run `.venv/bin/ruff check src tests` and `.venv/bin/mypy src` cleanly.
- [ ] **Step 5: Run a 20-step real-size check.** `.venv/bin/python -m cortical.train --condition patches --steps 20 --out "$SCRATCH/check"`, where `$SCRATCH` is the session scratchpad directory. Expected: it completes, and the logged loss at step 20 is below the loss at step 0.
- [ ] **Step 6: Commit.** `feat: training CLI with checkpoint/resume`

### Task 6: Launch the two runs

**Files:** no code changes. The runs write to `results/02_input_patches/` (the `*.pt` files are ignored, and `metrics.json` is tracked).

- [ ] **Step 1: Start the two runs in the background, one after the other** (they share the GPU):
  `.venv/bin/python -m cortical.train --condition baseline --seed 0 && .venv/bin/python -m cortical.train --condition patches --seed 0`
- [ ] **Step 2: Monitor progress until both runs finish** by checking for `"done": true` in each `metrics.json`. The expected total is about 2.5 h. If a run dies, rerunning the same command resumes it.
- [ ] **Step 3: Commit the two `metrics.json` files.** `results: notebook 02 runs (baseline, patches; seed 0)`

### Task 7: Notebook 02

**Files:**
- Create: `notebooks/02_input_patches.ipynb`, built with `nbformat` and executed with kernel `cortical`
- Modify: `README.md`. Add the notebook to the Notebooks section and `src/cortical/` to Layout.

**Interfaces:**
- Consumes: `metrics.json` from both runs, plus `cortical.sheet`, `cortical.wiring` and `cortical.evaluate.summarize`.

- [ ] **Step 1: Build the notebook** with the five sections from the spec:
  - **Setup.** The settings table, with sources.
  - **Wiring.** A layout B sheet map, `M` grouped by input group, one unit's d = 6 neighbourhood, and the weight counts per condition (287,500 vs 25,636 input weights; 282,500 recurrent weights).
  - **Learning curves.** Loss, plus the all/base/int/seq accuracy against step, for both conditions.
  - **Final performance.** Per-task accuracy at 10k for both conditions, sorted by family, plus the per-family means.
  - **Caveat.** One seed per condition, against the paper's ≈ ±0.01 seed spread.
  Use the palette from notebook 01. Before choosing colours, follow the dataviz skill; the two conditions are categorical slots 1 and 2.
- [ ] **Step 2: Execute the notebook and check it.** It must run with no error outputs. Render every figure and inspect it for collisions and empty panels.
- [ ] **Step 3: Run ruff on the notebook.** `.venv/bin/ruff check notebooks` must be clean.
- [ ] **Step 4: Commit.** `feat: notebook 02, input patches vs dense input wiring`
