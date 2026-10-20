# Sensory Decoding (Notebook 03) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, over time, how well the ring-2 stimulus angle can be linearly decoded from the ring-1 patch of the trained `patched_wiring` network, against a ceiling, a floor and comparison regions, in notebook 03.

**Architecture:**
- A new module, `src/cortical/decode.py`, holds four pieces:
  - task selection and the angle helper
  - trial collection, which runs the model on streams and records activity windows plus per-trial metadata
  - region selection
  - a time-resolved, cross-validated ridge decoder
- `cortical.evaluate`'s per-trial stream builder is factored out so both modules share it.
- Notebook 03 loads the saved model and runs the analysis.

**Tech Stack:** Python 3.12, PyTorch, numpy, scikit-learn (`RidgeCV`, `StratifiedKFold`), cogtasks / neurogym 2.3.1, matplotlib.

**Spec:** `docs/superpowers/specs/2026-10-20-sensory-decoding-design.md`

## Global Constraints

- **Model:** `results/02_input_patches/patched_wiring_s0/model.pt`, loaded as `MaskedRNN(50, 10, input_mask(50, LAYOUT_B_LARGE))`.
- **Tasks:** the 42 Mod-Cog tasks whose `make_task(...).unwrapped` is a `_Reach` or a `_DelayMatch1DResponse`.
- **Streams:** 128 trials per task, as one stream from a zero state, with rule unit `33 + k` (k = index in `list_tasks("modcog")`), data seed 4321.
- **Modality:** `env.i_env` after `new_trial()`, plus 1, gives which ring holds the stimulus.
- **Onset:** the first step of `sample` (match tasks), `stimulus` (go/anti tasks with that period), or `decision` (`rt*` tasks).
- **Window:** τ = −2…20 relative to onset (23 steps), with steps past a trial's end masked invalid.
- **Angle:** `arg Σ_k u_k e^{2πik/16}` over the stimulated ring's 16 inputs, summed over the first stimulus period, in [0, 2π).
- **Regions:** 324 units each: `ring1`, `ring2`, `matched`, `ring1_self`.
- **Decoder:**
  - per τ, a `RidgeCV` with alphas `np.logspace(-2, 4, 13)` on per-unit z-scored activity, targets (cos θ, sin θ)
  - 5-fold `StratifiedKFold` by task, shuffled with `random_state=seed`
  - score: the fraction with circular error ≤ π/8; null: 20 label shuffles, reported at the 95th percentile
- **Output:** `notebooks/03_sensory_decoding.ipynb`, run with kernel `cortical`.
- **Dependency:** `scikit-learn` goes in a new `analysis` extra in `pyproject.toml`.
- **Style:** no `print` in `src/`; the code must pass ruff and mypy.

## Review Focus

1. **Off-by-one onset.** Activity at τ = 0 must be the first step the stimulus is on the input. An off-by-one shifts every curve by 100 ms and fakes or hides a lag. Pinned by Task 2's `test_onset_is_first_stimulus_step`.
2. **Information leaking across folds.** Z-scoring or alpha selection must use training folds only. Pinned by Task 4's `test_null_stays_near_chance`, which fails if fold statistics leak.
3. **Mislabelled modality.** If the modality flag were inverted, `ring1` would decode its own input and look like transmission. Pinned by Task 2's `test_modality_matches_stimulated_ring`.
4. **Invalid steps scored as zeros.** At large τ, short trials have ended. Scoring their zero-padding would drag the curves toward chance or bias them. Pinned by Task 4's `test_invalid_steps_excluded`.
5. **A distance-matched region that isn't matched**, or that overlaps a patch. The comparison would then be meaningless. Pinned by Task 3's `test_matched_region_distances`.

---

### Task 1: Single-bump tasks and the angle helper

**Files:**
- Create: `src/cortical/decode.py`
- Test: `tests/cortical/test_decode.py`

**Interfaces:**
- Produces:
  - `SINGLE_BUMP_TASKS: tuple[str, ...]` (42 entries, in `list_tasks("modcog")` order)
  - `stimulus_angle(ring: np.ndarray) -> float`. `ring` is (16,) or (T, 16); it is summed over time first, and the result is in [0, 2π).

- [ ] **Step 1: Write the failing tests:**
  - `test_single_bump_tasks`: there are 42; each `make_task("modcog", n).unwrapped` is a `_Reach` or a `_DelayMatch1DResponse`; `"dm2"` and `"multidm"` are not included.
  - `test_stimulus_angle_recovers_bump_centre`: for centres in `[0.0, 0.3, np.pi, 5.9]`, build the ring with `cogtasks.modcog._envs._gaussianbump(c, theta, 1)`, where `theta = 2π·arange(16)/16`. The circular error is < 1° each time. This also holds for a (5, 16) stack of the same bump.
- [ ] **Step 2: Run the tests and confirm they fail** with an ImportError.
- [ ] **Step 3: Implement both, then run the tests.** Expected: 2 passed.
- [ ] **Step 4: Commit.** `feat: single-bump task list and ring angle helper`

### Task 2: Trial collection

**Files:**
- Modify: `src/cortical/evaluate.py`. Extract the per-trial loop of `_task_stream` into `iter_trials(name: str, k: int, n_trials: int, seed: int) -> Iterator[tuple[np.ndarray, np.ndarray, TrialEnv, int]]`, which yields `(ob_with_rule (T, 115) float32, gt (T,) int64, unwrapped task, i_env)`. `i_env` is `getattr(env, "i_env", 0)`, because the dm-family tasks are not `ScheduleEnvs`. `_task_stream` keeps its behaviour by using it.
- Modify: `src/cortical/decode.py`
- Test: `tests/cortical/test_decode.py`. The existing `tests/cortical/test_evaluate.py` must stay green.

**Interfaces:**
- Consumes: `MaskedRNN` (forward on a (T, B, 115) tensor), `iter_trials`.
- Produces:
  - `WINDOW = (-2, 20)`
  - `@dataclass(frozen=True) class TrialSet` with fields:
    - `task: np.ndarray` (str, n)
    - `modality: np.ndarray` (int 1/2, n)
    - `angle: np.ndarray` (float, n)
    - `activity: np.ndarray` (n, 23, len(units), float32)
    - `valid: np.ndarray` (n, 23, bool)
    - `units: np.ndarray` (recorded unit indices)
  - `collect(model: nn.Module, units: np.ndarray, n_trials: int = 128, seed: int = 4321, tasks: Sequence[str] = SINGLE_BUMP_TASKS, device: str = "cpu") -> TrialSet`
- Behaviour: per task, build the stream with `iter_trials` and run the model once with `torch.no_grad`. The hidden state comes from `model.rnn(x)`, not the logits. Slice each trial's window `[onset − 2, onset + 21)` within the stream, and mark steps past the trial's end (or before the stream's start) invalid, zero-filled. The angle is `stimulus_angle` over the stimulated ring's 16 inputs (indices 1–16 for ring 1, 17–32 for ring 2) during the first stimulus period.

- [ ] **Step 1: Write the failing tests.** Use an untrained `MaskedRNN(10, 2, input_mask(10, None))`, a 10×10 sheet; its unit count doesn't matter for metadata, but the model must accept 115 inputs. Use `n_trials=8` and `tasks=("go", "rtanti", "dms", "dlygointr")`.
  - `test_trialset_shapes`: 32 trials; activity (32, 23, len(units)); `valid` has the same leading shape; `valid[:, 2]` (τ = 0) is all true.
  - `test_onset_is_first_stimulus_step`: re-generate the same streams with `iter_trials` (same seed). For each trial, the stimulated ring's input is nonzero at onset and zero at onset − 1.
  - `test_modality_matches_stimulated_ring`: at onset, ring-`modality`'s inputs carry the bump and the other ring's inputs are zero.
  - `test_angle_matches_trial_records`:
    - For `dms`, compare with `task.trial["sample_theta"]`.
    - For `go`/`rtanti`/`dlygointr`, compare with `(θ[gt] + integrate·0.01·delay + anti·π) mod 2π`, where `delay` is `task.timing["delay"]` at that trial.
    - The circular error is < 1°.
  - `test_both_modalities_occur`: both 1 and 2 appear.
  - The existing evaluate tests still pass after the refactor.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement the refactor and `collect`.** Read the onset from `task.start_ind` in this order: `"sample"`, then `"stimulus"`, then `"decision"`.
- [ ] **Step 4: Run `tests/cortical`.** Expected: all pass.
- [ ] **Step 5: Commit.** `feat: collect activity windows and stimulus metadata per trial`

### Task 3: Regions

**Files:**
- Modify: `src/cortical/decode.py`
- Test: `tests/cortical/test_decode.py`

**Interfaces:**
- Consumes: `square_patch`, `torus_distance`, `LAYOUT_B_LARGE`.
- Produces: `regions(L: int, layout: Layout) -> dict[str, np.ndarray]`, with keys `"ring1"`, `"ring2"`, `"matched"` and `"ring1_self"` (the same units as `"ring1"`), each a sorted int array.
- Matching: for each ring-1 unit in ascending order, take its distance to the nearest ring-2 unit, then pick the not-yet-used input-free unit (in no patch) with the closest such distance, breaking ties by lowest unit index.

- [ ] **Step 1: Write the failing tests:**
  - `test_region_sizes`: all four regions have 324 units under `LAYOUT_B_LARGE` at L = 50.
  - `test_patches_match_layout`: `ring1`/`ring2` equal `square_patch` of the layout.
  - `test_matched_region_distances`: `matched` has no overlap with any patch, no duplicates, and its distance-to-ring-2 quantiles (0, 0.1, …, 1) are each within 1.0 of `ring1`'s.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement `regions`, then run the tests.** Expected: 3 passed.
- [ ] **Step 4: Commit.** `feat: decoding regions with a distance-matched control`

### Task 4: Time-resolved decoder

**Files:**
- Modify: `src/cortical/decode.py`, and `pyproject.toml` (add `analysis = ["scikit-learn"]`; then install with `.venv/bin/pip install -q -e ".[dev,notebook,analysis]"`)
- Test: `tests/cortical/test_decode.py`

**Interfaces:**
- Produces:
  - `circular_error(a: np.ndarray, b: np.ndarray) -> np.ndarray` (radians, in [0, π])
  - `@dataclass(frozen=True) class DecodeResult` with fields:
    - `tau: np.ndarray` (23,)
    - `accuracy: np.ndarray` (23,): the fraction with error ≤ π/8
    - `median_error: np.ndarray` (23,)
    - `null95: np.ndarray` (23,)
    - `n_trials: np.ndarray` (23,): valid trials per τ
    - `predicted: np.ndarray` (n, 23): NaN where invalid
  - `decode_over_time(activity: np.ndarray, angles: np.ndarray, valid: np.ndarray, groups: np.ndarray, n_splits: int = 5, n_shuffles: int = 20, seed: int = 0) -> DecodeResult`. `groups` gives the stratification labels (the task names).
- Behaviour, per τ:
  - Use only valid trials.
  - For each fold, fit a `StandardScaler` and a `RidgeCV(alphas=np.logspace(-2, 4, 13))` on the training fold only, then predict the test fold.
  - The null repeats this with angles permuted (seeded) and takes the 95th percentile of accuracy over the shuffles.

- [ ] **Step 1: Write the failing tests** on synthetic data: 400 trials, 50 units, 23 steps, 4 groups. For τ ≥ 5, activity = `[cos θ, sin θ] @ W + 0.3·noise`, with a random fixed W (2×50). For τ < 5, pure noise.
  - `test_recovers_encoded_angle`: accuracy ≥ 0.9 at every τ ≥ 5.
  - `test_chance_without_encoding`: accuracy ≤ 0.25 at every τ < 5 (chance is 0.125).
  - `test_null_stays_near_chance`: `null95` ≤ 0.25 at every τ.
  - `test_invalid_steps_excluded`: mark half the trials invalid at τ ≥ 15 and set their activity to zero. Then `n_trials[15:] == 200`, accuracy at τ ≥ 15 is still ≥ 0.9, and `predicted` is NaN exactly where invalid.
  - `test_circular_error_wraps`: the error between 0.1 and 2π − 0.1 is 0.2.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement `circular_error` and `decode_over_time`.**
- [ ] **Step 4: Run the full suite, ruff and mypy.** `.venv/bin/python -m pytest -q -p no:warnings`, `.venv/bin/ruff check src tests` and `.venv/bin/mypy src` must all be clean.
- [ ] **Step 5: Commit.** `feat: time-resolved cross-validated angle decoder`

### Task 5: Notebook 03

**Files:**
- Create: `notebooks/03_sensory_decoding.ipynb` (built with nbformat; executed with kernel `cortical`)
- Modify: `README.md`. Add notebook 03, `decode.py`, and the `analysis` extra.

**Interfaces:**
- Consumes: `collect`, `regions`, `decode_over_time` and `circular_error` from Tasks 1–4, plus the trained model.

- [ ] **Step 1: Build the notebook:**
  1. **Question and setup.** The model, regions, trials, decoder, chance level and the no-MPS note.
  2. **Region map.** A 50×50 sheet showing the four patches plus the `matched` units, with ring 1 and ring 2 labelled and the d = 10 reach.
  3. **Main figure.** Accuracy (±22.5°) against τ × 100 ms for `ring1` (ring-2 trials), `ring2` (ceiling), `matched` and `ring1_self`. Each region's null95 is a band, chance is a line at 0.125, and stimulus onset is a vertical line at τ = 0.
  4. **Per family.** The same plot split into go/anti and match tasks.
  5. **Match-task timeline.** The `ring1` accuracy for the sample angle, with the median sample, delay and test spans shaded.
  6. **What we see.** Written after inspecting the numbers.
  7. **Caveats.** One seed; linear only; decodable is not the same as used; the delay varies by trial.
  Follow the dataviz skill. The four regions get categorical slots 1–4, and the validator must pass on them.
- [ ] **Step 2: Execute the notebook and check it.** It must run with no errors, and every rendered figure gets inspected.
- [ ] **Step 3: Run ruff on the notebook.** `.venv/bin/ruff check notebooks` must be clean.
- [ ] **Step 4: Commit.** `feat: notebook 03, decoding ring-2 angle from the ring-1 region`
