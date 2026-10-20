# cortical

## `cogtasks`

`cogtasks` serves two standard cognitive task sets. Both come from one implementation, [Mod-Cog](https://github.com/mikailkhona/Mod_Cog) (Khona, Chandra et al. 2023, [arXiv:2207.03523](https://arxiv.org/abs/2207.03523)), vendored in `src/cogtasks/modcog/`:

| collection | tasks | contents |
|------------|-------|----------|
| `cog20`    | 20    | The Yang et al. 2019 tasks, in paper order, as Mod-Cog implements them |
| `modcog`   | 82    | `cog20`, then 22 `int` variants (delay tasks), then 40 `seq` variants (`r`/`l`) |

Every task uses a 16-unit ring and the default `dt` of 100 ms.
- Observations have 33 dims: fixation plus two stimulus modalities.
- Actions have 17 values: fixation plus 16 choices.
- In a multitask env, one rule unit per task is appended to the observation.

### Setup

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

### Usage

```python
from cogtasks import list_tasks, make_task, make_multitask, make_dataset

list_tasks("cog20")                       # ('go', 'rtgo', ..., 'dnmc')
env = make_task("modcog", "dlygointr", seed=0)

# Tasks interleaved at random, with a one-hot rule input (obs dim 33 + n_tasks)
env = make_multitask("modcog", seed=0)

# Supervised batches: inputs (seq_len, batch, 33 + n_tasks), labels (seq_len, batch)
ds = make_dataset("modcog", batch_size=64, seq_len=350, seed=0)
inputs, labels = ds()
```

- `tasks=[...]` picks a subset, in the order you give.
- Extra keyword arguments, such as `dt` or `timing`, are passed to each task factory.
- A `seed` makes construction and every trial reproducible.

### Notebooks

- `notebooks/01_task_gallery.ipynb` shows one example trial (inputs and targets) for each of the 82 tasks.
- `notebooks/02_input_patches.ipynb` trains Khona et al.'s locality-masked RNN (50×50 torus, d = 10) on Mod-Cog twice: once with dense input weights, and once with each input group strictly wired to its own patch of the sheet (`patched_wiring`, 53% of the sheet). It compares the two. Training runs separately (`python -m cortical.train --condition {baseline,patched_wiring} --seed 0`, about 6½ min per run on an RTX 4090, about 70 min on an M2 Pro). The notebook only loads `results/02_input_patches/`.

### Training on a GPU

`python -m cortical.train` picks a device automatically: an NVIDIA GPU (CUDA) first, then Apple's GPU (MPS), then the CPU. `--device` overrides the choice, e.g. `--device cuda:1` or `--device cpu`. Checkpoints resume across devices. To check speed on a new machine, run a short job and read the step timestamps in the log:

```bash
python -m cortical.train --condition baseline --steps 50 --out /tmp/speedcheck
```

For reference, an M2 Pro takes about 0.4 s per step (MPS) and 2.5 s per step (CPU). The workload is bound by memory bandwidth: with batch size 4, every time step re-reads the 2500×2500 recurrent matrix.


To run the notebooks, install the `notebook` extra (`pip install -e ".[notebook]"`) and select the `cortical` kernel.

### Layout

```
src/cogtasks/
  registry.py       collections and the make_* loaders
  _seeding.py       independent per-env RNG streams
  modcog/
    __init__.py     task lists (COG20, INT_BASE, MODCOG)
    _envs.py        Mod-Cog env classes (vendored)
    _tasks.py       Mod-Cog task factories (vendored)
tests/              collections, per-task behaviour, multitask/dataset, seeding
src/cortical/
  sheet.py          torus geometry and the recurrent locality mask
  wiring.py         input groups, patch layouts and the input mask
  model.py          masked LM-RNN (ReLU nn.RNN + dense readout)
  evaluate.py       fixed evaluation streams and strict per-trial accuracy
  train.py          training CLI with checkpoint/resume
notebooks/          task gallery, input-patch experiment
results/            run metrics (weights are git-ignored)
docs/superpowers/   design specs and implementation plans
```

### The vendored Mod-Cog code

The code comes from `mikailkhona/Mod_Cog` at commit `4110143`, which is CC0 (see `LICENSES/`). The task logic is unchanged apart from one bug fix. The edits are:
- **Bug fix: choice 16 wrapped to fixation.** Upstream applied the mod-16 wrap after mapping a choice to its action unit, so choice 16 became action 0 (fixation). As a result, no task ever targeted choice 16, about 1 in 16 trials that should end in a response became fixation-only, and `seq` targets passed through fixation mid-response. The wrap now happens in choice space, which makes these targets differ from Khona et al.'s published runs.
- **Port to neurogym 2.x.** `step` returns the 5-tuple, `TrialEnv` is imported from `neurogym.core`, and the per-trial `int` delay is drawn from the seeded `self.rng`.
- **Removed factories.** The 11 unused `*intseq` factories are gone.

Upstream quirks are kept as they are:
- **`dms`/`dnms`/`dmc`/`dnmc` delay.** It is drawn once per env instance, not once per trial, so each task has two delay values, one per modality. A `seed` fixes them.
- **`dlyantiseq*`.** These overwrite their `delay` timing override.
- **`dlygoseq*`.** These are identical to `goseq*`.
- **`*seq*` timing.** These factories overwrite any `timing` you pass.
- **`int` offsets are inconsistent across families:**
  - **`dlygo`/`dlyanti`:** about 2.5 units per 100 ms of delay (ω applied in radians), and *r* is counter-clockwise.
  - **`dms` family:** 1 unit per 100 ms (ω applied in ring units), and *r* is clockwise.
  - **`dlydm*` int tasks:** the delay is fixed at 500 ms, so the offset is a constant 5 units.

### neurogym pitfalls the loader works around

- **Nested schedules lose the rule input.** Mod-Cog's two-modality tasks are nested `ScheduleEnvs`. The outer wrapper only re-points the first leaf at itself, so trials that `step()`/`reset()` start from the other leaf skip the rule input. `make_multitask` re-points every leaf.
- **Seeds are shared or lost.** `ScheduleEnvs.seed` gives every member the same stream. `Dataset` reseeds its copies from OS entropy. Its deep copies also keep drawing timings from the original env's RNG through `lambda: self.rng...` timing entries. The loader derives independent seeds with `SeedSequence` and builds one env per batch element.

neurogym (Apache-2.0) is a dependency. Version 2.3.1 imports torch at load time.
