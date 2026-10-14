# cortical

## `cogtasks`

`cogtasks` serves two cognitive task sets through one loader:

| collection      | tasks | source |
|-----------------|-------|--------|
| `yang19`        | 20    | Cog20 from Yang et al. 2019, *Nat. Neurosci.* Loaded from neurogym's `yang19` collection. |
| `modcog`        | 82    | Mod-Cog from Khona, Chandra et al. 2023 ([arXiv:2207.03523](https://arxiv.org/abs/2207.03523)). The set is 20 base tasks, 22 `int` variants and 40 `seq` variants. It is vendored from [mikailkhona/Mod_Cog](https://github.com/mikailkhona/Mod_Cog). |
| `modcog_intseq` | 11    | Upstream `*intseq` combinations. They are not part of the 82-task set. |

Every task uses a 16-unit ring. Observations have 33 dims: fixation plus two stimulus modalities. Actions have 17 values: fixation plus 16 choices.

### Setup

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

### Usage

```python
from cogtasks import list_tasks, make_task, make_multitask, make_dataset

list_tasks("modcog")                      # ('go', 'rtgo', ..., 'dnmcseql')
env = make_task("modcog", "dlygointr", seed=0)

# All tasks interleaved at random, with a one-hot rule input appended (obs dim 33 + n_tasks)
env = make_multitask("modcog", seed=0)

# Supervised batches: inputs (seq_len, batch, 33 + n_tasks), labels (seq_len, batch)
ds = make_dataset("modcog", batch_size=64, seq_len=350, seed=0)
inputs, labels = ds()
```

`tasks=[...]` picks a subset in the order you give. Extra keyword arguments, such as `dt` or `timing`, are passed to each task factory.

### Notes on the Mod-Cog port

`src/cogtasks/modcog/_envs.py` and `_tasks.py` keep the upstream task logic verbatim. The only changes port it from gym-era neurogym to neurogym 2.3.1:
- the 5-tuple `step` API
- `TrialEnv` imported from `neurogym.core`
- the per-trial `int` delay drawn from the seeded `self.rng`

Upstream quirks are kept as they are:
- **`dms`/`dnms`/`dmc`/`dnmc` delay.** It is drawn **once per env instance** at construction from the global `np.random`, not once per trial. Each task wraps two modality instances, so it has only two delay values. Call `np.random.seed(...)` before building to fix them.
- **`dlyantiseqr`/`dlyantiseql`.** These overwrite their `delay` timing override.
- **`dlygoseq*`.** These are identical to `goseq*`.
- **Base tasks.** Mod-Cog's 20 base tasks are its own reimplementation, so they do not match neurogym's `yang19` trial for trial (for example, `go` lasts 20 steps vs 15).

### neurogym pitfalls the loader works around

- **Nested schedules lose the rule input.** Mod-Cog's two-modality tasks are nested `ScheduleEnvs`. The outer wrapper only re-points the first leaf at itself, so trials that `step()`/`reset()` start from the other leaf skip the one-hot rule input. `make_multitask` re-points every leaf. Driving the env only through `new_trial()`, as `Dataset` does, was never affected.
- **Seeds are shared or lost.** `ScheduleEnvs.seed` gives every member the same stream. `Dataset` reseeds its copies from OS entropy. Its deep copies also keep drawing timings from the original env's RNG through `lambda: self.rng...` timing entries. The loader therefore derives independent seeds with `SeedSequence` and builds one env per batch element.

Mod-Cog is CC0-1.0 (`LICENSES/Mod_Cog-CC0-1.0.txt`). neurogym is Apache-2.0.
