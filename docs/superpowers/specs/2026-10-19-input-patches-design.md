# Notebook 02: Khona LM-RNN with inputs wired to cortical patches

Date: 2026-10-19
Status: approved in conversation; awaiting review of this written spec

## Goal

Reproduce the Mod-Cog training setup of Khona, Chandra et al. 2023 (arXiv:2207.03523v2) with a locality-masked RNN (LM-RNN), then change exactly one thing. In Khona's setup each input group reaches every unit. In the variant, it reaches only units inside its own patch of the cortical sheet. The comparison is shown in notebook 02.

**Success:** both conditions train for 10k gradient steps with identical settings. Notebook 02 shows their wiring, learning curves and final per-task accuracy side by side.

**Out of scope:**
- other wiring rules (topographic maps, graded fall-off)
- a localized readout
- the scattered-input control
- more than one seed per condition

The code should make these easy to add later.

## Training setup

Khona released no training code. The table records the source of every setting.

| setting | value | source |
|---|---|---|
| tasks | `cogtasks.make_dataset("modcog", ...)`: 82 tasks, one-hot rule input, observation dim 115, 17 actions | paper; the released task list |
| dynamics | `h_{t+1} = φ(W h_t + W_in u_t + b_ih + b_hh)`, `o = W_out h + b_out`, discrete time | paper eq. 1; two biases inferred from the parameter counts |
| φ | ReLU | chosen (unstated in the paper) |
| sheet | 50×50 = 2500 units on a torus (periodic boundaries) | paper; periodicity inferred from the parameter counts |
| recurrent mask `G` | `G_ij = 1` iff torus distance ≤ d; self-connections included; d = 6 (282,500 nonzero entries, 4.5% density) | paper; counts match exactly |
| init | PyTorch `nn.RNN` / `nn.Linear` defaults: U(−1/√N, 1/√N) | inferred (nn.RNN) |
| masking | `W ← G ⊙ W`, `W_in ← M ⊙ W_in`, applied at init and after every optimizer step | paper (for `G`); same rule for `M` |
| loss | cross-entropy on softmax(o), unweighted, averaged over all time steps and batch | paper eq. 2 |
| optimizer | Adam, lr 1e-3, no gradient clipping, no weight decay | paper |
| batch × sequence | 4 × 350 steps (dt = 100 ms), from neurogym `Dataset` streams | Mod-Cog README example |
| budget | 10,000 gradient steps | paper (reports performance at 10k) |
| seeds | 1 per condition (seed 0) | chosen for compute; the paper used 15 |
| device | MPS, with a CPU fallback | about 0.4 s/step on an M2 Pro, so roughly 70 min per run |

## Conditions

Both conditions are a 50×50 torus with d = 6 and a dense readout.

1. **`baseline`**: `M` is all ones, so every input reaches all 2500 units (Khona).
2. **`patches`**: `M` restricts each input group to its patch (layout B):

| group | observation indices | patch | centre (x, y) | units |
|---|---|---|---|---|
| fixation | 0 | 6×6 | (12.5, 37.5) | 36 |
| ring 1 | 1–16 | 12×12 | (12.5, 12.5) | 144 |
| ring 2 | 17–32 | 12×12 | (37.5, 37.5) | 144 |
| rule | 33–114 | 16×16 | (37.5, 12.5) | 256 |

- Each patch is the set of lattice points (x, y) with |x − cx| < size/2 and |y − cy| < size/2. Lattice coordinates are integers 0..49, and unit index = y·50 + x. For example, ring 1 covers x, y ∈ 7..18.
- Within a patch the connection is dense: every input of the group connects to every unit of its patch, and to no unit outside it.
- The patches are disjoint and cover 580 units (23% of the sheet). The other 77% receives input only through recurrence.
- The patch centres sit on a 2×2 lattice 25 apart, with the rings at opposite corners (35.4 apart, the farthest pair on the torus). Rule and fixation each neighbour both rings (25 apart).
- The input-weight count falls from 115 × 2500 = 287,500 to 36 + 16·144 + 16·144 + 82·256 = 25,636.

## Evaluation

**Run trials as a stream.** Training uses continuous streams, so the hidden state carries over between trials. Evaluation does the same: for each task, n = 32 trials are concatenated into one sequence, with the task's rule input on. The network runs from a zero state, and every trial is scored separately. All 82 tasks are evaluated in one batch (the 82 streams are padded to equal length at the end; padding sits after the last trial, so it never affects a scored step).

**What counts as a correct trial** (argmax over the 17 outputs at every step of that trial):
- every step before the trial's `decision` period outputs fixation (0), and
- every step of the `decision` period outputs the target `gt` (for no-go trials the target is fixation throughout).

**What is reported:**
- per-task accuracy
- the mean over the 82 tasks, the headline number comparable to Khona's 10k-step table
- the mean per family: base (20), int (22), seq (40)

**When to evaluate:** at step 0 and every 500 steps, with a fixed evaluation seed so every evaluation uses the same trials.

## Code layout

A new package `src/cortical/` sits beside `cogtasks`, with one job per module:

- `sheet.py`
  - `coords(L)` gives the (N, 2) lattice coordinates.
  - `torus_distance(L)` gives the (N, N) pairwise distances.
  - `locality_mask(L, d)` gives the (N, N) bool mask `G`.
  - `square_patch(L, centre, size)` gives the unit indices of a patch.
- `wiring.py`
  - `INPUT_GROUPS` maps each group name to its observation slice.
  - `LAYOUT_B` maps each group name to (centre, size).
  - `input_mask(L, layout | None)` gives the (I, N) bool mask `M`; `None` means dense.
- `model.py`: `MaskedRNN(n_input, L, d, input_mask)` wraps `nn.RNN(nonlinearity="relu")` and `nn.Linear`.
  - It keeps `G` and `M` as buffers.
  - `apply_masks()` multiplies them into the weights in place, under `no_grad`.
  - `forward(x: (T, B, I)) -> logits (T, B, 17)`.
- `evaluate.py`
  - `build_eval_set(n_trials, seed)` makes fixed streams plus per-trial boundaries and `decision` spans, for all 82 tasks.
  - `score(model, eval_set)` returns per-task accuracy.
- `train.py`: a CLI, `python -m cortical.train --condition {baseline,patches} --seed 0 [--steps 10000] [--out DIR]`.
  - Every 500 steps it writes `metrics.json`: the loss curve, and each evaluation's per-task accuracy.
  - It saves a `checkpoint.pt` it can resume from (model, optimizer, step, torch RNG state) and a final `model.pt`. neurogym env state can't be checkpointed, so a resumed run rebuilds its data stream with seed `(seed, step)`. A resumed run is valid but not bit-identical to an uninterrupted one.
  - Output goes under `results/02_input_patches/<condition>_s<seed>/`, and `*.pt` files are git-ignored.

The training data is `cogtasks.make_dataset("modcog", batch_size=4, seq_len=350, seed=seed)`. Its inputs are float32 and its labels int64.

## Notebook 02

`notebooks/02_input_patches.ipynb` (kernel `cortical`) only loads saved results. It contains:
1. **Setup.** The training settings and their sources, as summarized in the table above.
2. **Wiring.** A sheet map of layout B, `M` as a 115×2500 matrix grouped by input group, the recurrent neighbourhood of one unit, and the weight counts per condition.
3. **Learning curves.** Training loss and evaluation accuracy (overall and per family) against gradient step, for both conditions.
4. **Final performance.** Per-task accuracy at 10k steps for both conditions, and per-family means.
5. **Caveat.** One seed per condition. The paper's seed-to-seed spread at 10k steps is about ±0.01.

## Testing

Unit tests (`tests/cortical/`):
- **Sheet.** Torus distances wrap correctly. `locality_mask(50, 6)` has 282,500 entries and `(50, 10)` has 792,500. Self-connections are included. The mask is symmetric.
- **Wiring.** Patch sizes are 36, 144, 144 and 256. Patches are disjoint. Each group's rows of `M` are true exactly on its patch. The dense layout is all true. There are 25,636 nonzeros in total.
- **Model.** Masked entries are zero after init and stay zero after an Adam step. The output shape is right. Parameter counts match the `nn.RNN` structure.
- **Evaluation.** On hand-built logits, perfect output scores 1. A late response, broken fixation, a response on a no-go trial, and a wrong response step each score 0.
- **Smoke test.** Training on a 10×10 sheet for 3 steps writes `metrics.json` and a checkpoint, and resuming continues from the saved step.

## Risks

- **One seed.** Differences are anecdotal. The notebook says so.
- **Unknown metric definition.** Khona's absolute numbers (≈0.80 at 10k) may not be comparable to the strict metric used here. The baseline is what we compare against.
- **Run time.** About 70 min per run. A run killed partway resumes from its checkpoint.
