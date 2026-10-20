# Notebook 03: recovering ring-2 stimulus angle from the ring-1 region

Date: 2026-10-20
Status: approved in conversation; awaiting review of this written spec

## Question

In the trained `patched_wiring` network (notebook 02: 50×50 torus, d = 10, seed 0), ring 2's inputs reach only the ring-2 patch. **Can the ring-2 stimulus angle be linearly recovered from the ring-1 patch's activity, and when?** Any such information must arrive through recurrence: the two ring patches are about 11 units apart edge to edge, so it takes at least 2 recurrent steps.

**Success:** notebook 03 (`notebooks/03_sensory_decoding.ipynb`) shows time-resolved decoding of the ring-2 angle from the ring-1 patch, alongside a ceiling, a floor and two comparison regions. It shows this pooled across tasks and per family.

**Out of scope:**
- stimulus strength or coherence, and ring-2-derived decisions (possible follow-ups)
- nonlinear decoders
- causal tests (ablation)
- other models or seeds; this is `patched_wiring` only

## Trials

- **Tasks:** the 42 single-bump tasks, meaning every Mod-Cog task built on `_Reach` (the go/anti family, with its int/seq variants) or on `_DelayMatch1DResponse` (the dms/dnms/dmc/dnmc family, with its int/seq variants). These tasks put their one stimulus on ring 1 or ring 2 at random (`_MultiModalityStimulus`, modality 0 or 1). The dm-family tasks are excluded: they show two ring-2 bumps at once, or both rings.
- **Streams:** for each task, 128 trials run back to back as one stream from a zero hidden state, with the task's rule one-hot set (index = position in `list_tasks("modcog")`). This matches how `cortical.evaluate` builds its streams. The data seed is fixed, distinct from the evaluation seed.
- **Recorded per trial:**
  - `task`
  - `modality`: 1 or 2, the stimulated ring
  - `onset`: the stream index of the first step of the first stimulus period. That period is `stimulus` for go/anti tasks, `sample` for match tasks, and `decision` for `rt*` tasks, where the stimulus appears with the go cue.
  - `angle`: the circular mean of the stimulated ring's 16 input units, summed over that first stimulus period: `θ = arg Σ_k u_k e^{iθ_k}`, where θ_k = 2πk/16
  - the hidden activity of all 2,500 units from onset − 2 to onset + 20 (23 steps), zero-padded where a trial ends earlier, with a validity mask
- The decoded variable is this **first-stimulus angle** at every time step. For match tasks, the steps after the delay therefore test whether the **sample** angle is retained.

## Regions

Every region has 324 units, so decoders have equal capacity:

| region | units | trials | role |
|---|---|---|---|
| `ring1` | the ring-1 patch of `LAYOUT_B_LARGE` | modality 2 | the question |
| `ring2` | the ring-2 patch | modality 2 | ceiling (direct input) |
| `matched` | input-free units with the same distances to the ring-2 patch as `ring1`'s units | modality 2 | is ring 1 special, or just as far from ring 2? |
| `ring1_self` | the ring-1 patch | modality 1 | reference: the patch decoding its own input |

**Matching:** for each `ring1` unit, its distance to the nearest ring-2-patch unit on the torus. Then a greedy one-to-one match to the input-free unit (not in any patch) with the closest such distance, ties broken by unit index. A test checks that the two distance distributions agree within 1 unit at every quantile.

## Decoding

- **Fit:** at each relative time step τ ∈ [−2, 20], a ridge regression from the region's activity (z-scored per unit, with statistics from the training fold) to the targets (cos θ, sin θ). The decoded angle is `atan2` of the two predictions.
- **Validation:** 5-fold cross-validation over trials, stratified by task. The regularization strength is chosen by inner cross-validation over a log grid of 10⁻²…10⁴ (scikit-learn `RidgeCV`). Trials that have ended before τ are excluded at that step.
- **One decoder per τ pooled over all 42 tasks.** Per-family scores (go/anti, match) come from the pooled decoder's held-out predictions.
- **Score:** the fraction of held-out trials with circular error ≤ 22.5° (one ring unit), plus the median circular error. Chance level for the ±22.5° score is 0.125.
- **Null:** 20 refits per τ with the angles shuffled across trials. The report shows the null's 95th percentile.

## Code

- **`src/cortical/decode.py`** (new):
  - `SINGLE_BUMP_TASKS`: the 42 task names, derived from the task classes rather than hand-listed.
  - `collect(model, n_trials=128, seed=…) -> TrialSet`: a frozen dataclass of per-trial `task`, `modality`, `angle`, `onset`, plus activity (n_trials, 23, 2500) and a validity mask.
  - `stimulus_angle(ring_inputs) -> float`: the circular mean.
  - `regions(L, layout) -> dict[str, np.ndarray]`: unit indices, with `matched` built as above.
  - `decode_over_time(activity, angles, valid, n_splits=5, n_shuffles=20, seed=…) -> DecodeResult`: per τ, accuracy within 22.5°, median error, null 95th percentile, and held-out predictions.
- **`pyproject.toml`:** a new `analysis` extra holding `scikit-learn`.
- **Notebook 03:**
  - the region map
  - decoding against τ for the four regions, with the null band
  - the same split by family
  - for match tasks, an annotated sample → delay → test timeline
  - a short "what we see"
  - caveats: one seed, linear only, decodable is not the same as used

  It loads `results/02_input_patches/patched_wiring_s0/model.pt` and computes everything in the notebook in minutes on CPU or MPS.
- **Model weights:** they are git-ignored, so the notebook documents how to regenerate them (the training command from notebook 02).

## Testing (`tests/cortical/test_decode.py`)

- **`stimulus_angle`:** recovers the centre of a Gaussian ring bump to within 1°, including centres off the 16-point grid.
- **`SINGLE_BUMP_TASKS`:** has 42 entries, all built on `_Reach` or `_DelayMatch1DResponse`.
- **`collect`:** on a small untrained model,
  - shapes and masks are consistent
  - `angle` matches the task's own trial records to within 1°. For `_DelayMatch1DResponse` that's `trial["sample_theta"]`. `_Reach` doesn't store its stimulus angle, so the test rebuilds it from the trial record the way the task does: `θ[ground_truth] + integrate·ω·delay + anti·π`.
  - `onset` falls on the first stimulus step, with ring input nonzero at `onset` and zero just before it
  - both modalities occur
- **`regions`:** each region has 324 units. `ring1` and `ring2` equal the patches. `matched` is input-free, disjoint from the patches, and distance-matched as specified.
- **`decode_over_time`:**
  - on synthetic activity that linearly encodes (cos θ, sin θ) plus noise, the score is ≥ 0.9 at encoded steps and near chance at unencoded steps
  - the shuffle null stays near chance
  - invalid steps are excluded

## Compute

All of this is forward passes plus small ridge fits, and it runs locally. The rented GPU isn't needed for this analysis.

## Risks

- **The angle's source depends on the stimulus period.** Match-task stimuli carry input noise, and the circular mean over the whole period averages it out; the trial-record test guards this.
- **Which period counts as the stimulus differs by task.** The `rt*` tasks present the stimulus in `decision`; the onset test guards this.
- **One seed.** The notebook says so.
- **"Recoverable" isn't the same as "used".** Causal tests are out of scope, and the notebook says so.
