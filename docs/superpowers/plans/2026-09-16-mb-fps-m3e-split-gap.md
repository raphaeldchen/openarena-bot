# MB-FPS M3e: The Split Gap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the nine shipped M3c world models on the training episodes they learned from, beside the validation episodes every recorded number is on, and decide -- by a rule written before the run -- whether the open-loop failure is generalisation (memorisation), budget/capacity, or the objective.

**Architecture:** One new instrument, `scripts/split_gap.py`, that loads exactly as `trust_horizon.py` loads (imported by path, never copied), runs M3d's `reference_trajectories` pass on three strata of episodes (`val`, `train_held`, `train_probe`) under the one refit probe each record was scored with, and reduces every stratum through the pure functions of `mbfps.eval.trust`. The three-stratum rule lives in `mbfps.eval.probe.probe_episodes` (which `fit_probes` now routes through, so the probe and the diagnostic cannot disagree about which episodes are probe-seen), the pure readings in a new `mbfps.eval.split_gap`, and one new pooling primitive (`unpaired_contrast`) in `mbfps.eval.pooling`. The val stratum must reproduce the shipped record and diagnostic bitwise before anything is read.

**Tech Stack:** Python 3.12, torch (MPS for the run, CPU in tests), numpy, matplotlib (Agg) for one figure, pytest. No new dependency.

**Spec:** `docs/superpowers/specs/2026-09-16-mb-fps-m3e-split-gap-design.md`. Section numbers below are the spec's.

## Global Constraints

- Every task begins and ends with `.venv/bin/python -m pytest -q` fully green, **0 warnings**. Record the measured count at the end of each task; the suite is at `1437 passed` on `main` at `0550a82`.
- `runs/m3_study` and `runs/m3_study_v2` are read-only except for the files this diagnostic writes: `split_gap_<arm>_seed<n>.json`, `split_gap.txt`, `split_gap.log`, `split_gap.exit`, `split_gap.head`, `learning_curves.png`. **Never `rm` anything under `runs/`.** `ls -lt runs/m3_study_v2 | head` after the run must show nothing newer than those files.
- Exit statuses: `0`, and `11 / 12 / 14 / 30` reused from `trust_horizon.py` WITH ITS MEANINGS; `31` new (`EXIT_STRATA_NOT_A_PARTITION`). Never `1` (an uncaught traceback) or `2` (argparse). `test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own` gains `split_gap` in Task 6.
- Nothing under `src/mbfps/eval/{aggregate,summary,study,rollout,trust,trust_readings}.py`, `scripts/{report_study,diagnose_dynamics,run_study,pool_dynamics}.py` changes. `scripts/trust_horizon.py` changes ONLY by a behaviour-preserving extraction: its per-cell check-and-refit block becomes `prepare_cell()`, which its own `_run_cell` calls; every one of its existing tests passes unchanged and its printed output is byte-identical. `diagnostics.py` gains ONE additive field; `probe.py` a constant and a function whose introduction is behaviour-preserving; `pooling.py` one dataclass and one function.
- The M3c plan (`2026-09-11-...`) and the M3d plan (`2026-09-13-...`) are closed records. Results go in THIS plan's `## Task 7 results` only.
- Device: the run is `--device mps`; every test runs `device="cpu"`. A CPU run of the real cells exits 14 by design (spec 2.4).
- Commit after every task, message in the repo's voice (`feat:` / `test:` / `docs:`, a sentence, no period), ending with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- **HEAD does not move during the Task 7 run.** `split_gap.head` records it; every record's `git_sha` must equal it.
- Pre-registration holds: `DECISION_H = 15`, `FAMILY = 6`, `SEEDS_REQUIRED = 2`, `Q_REPORTED = (0.5, 0.75, 0.9)`, `R2_SENSITIVITY = 0.1`, `MIN_MOVE = 5.0` (trust's) are constants named in code and are not tuned after the run.

---

## Interface contract

### File map (one responsibility each)

| file | responsibility | task |
|---|---|---|
| `src/mbfps/eval/probe.py` | `PROBE_EPISODE_LIMIT`, `probe_episodes(paths, limit) -> (used, held)`; `fit_probes` takes `used` from it | 1 |
| `src/mbfps/eval/diagnostics.py` | `Trajectories.band: RolloutResult | None = None`; `reference_trajectories` sets it | 2 |
| `src/mbfps/eval/pooling.py` | `UnpairedContrast`, `unpaired_contrast(a, b)` | 2 |
| `src/mbfps/eval/split_gap.py` (new) | constants; `StrataNotAPartition`; `strata_partition`; `stratum_summary`; `StratumContrast`, `ArmInputs`, `GapInputs`, `Status`, `ArmReading`, `GapReading`, `reading_gap`, `format_reading_gap`; `learning_curve_summary`, `TERMS` | 3, 4 |
| `scripts/split_gap.py` (new) | load (via `trust_horizon` by path), strata, the five checks in order, three passes per cell, the record, then pooling glue, Reading G, sensitivity, learning curves + figure, `split_gap.txt` | 5, 6 |
| `tests/eval/conftest.py` | `_build_buffer(tmp_path, n)`; `small_buffer` (6) unchanged in behaviour; new `wide_buffer` (30) | 5 |
| `tests/eval/test_probe.py` | the `PROBE_EPISODE_LIMIT` coupling tests | 1 |
| `tests/eval/test_diagnostics.py` | `band` present and equal to the pass's curves | 2 |
| `tests/eval/test_pooling.py` | `unpaired_contrast` arithmetic and refusals | 2 |
| `tests/eval/test_split_gap.py` (new) | the pure functions, every rule mutated one at a time | 3, 4 |
| `tests/eval/test_split_gap_script.py` (new) | `main()` on `wide_buffer`; record fields; exits in order; verdict lines; figure | 5, 6 |
| `tests/eval/test_diagnose_dynamics_script.py` | the distinctness test gains `split_gap` | 6 |

### Shared signatures (every task's implementer sees only their own task; these are the names their neighbours use)

```python
# mbfps.eval.probe (Task 1)
PROBE_EPISODE_LIMIT: int = 20
def probe_episodes(paths, limit: int = PROBE_EPISODE_LIMIT) -> tuple[list, list]: ...
def fit_probes(model, paths, backbone, device, context=5, horizon=45, limit=PROBE_EPISODE_LIMIT, seed=0, select_episodes=4, ridge=None) -> tuple[dict, dict]: ...

# mbfps.eval.diagnostics (Task 2)
@dataclass(frozen=True)
class Trajectories:  # thirteen existing fields, then:
    band: "RolloutResult | None" = None

# mbfps.eval.pooling (Task 2)
@dataclass(frozen=True)
class UnpairedContrast:
    a: str; b: str; channel: str; windows_a: int; windows_b: int; clusters: int
    mean: float; se: float; z: float
def unpaired_contrast(a: PooledMean, b: PooledMean) -> UnpairedContrast: ...

# mbfps.eval.split_gap (Tasks 3, 4)
STRATA: tuple[str, ...] = ("val", "train_held", "train_probe")
DECISION: tuple[str, str] = ("train_held", "val")
CHANNELS: tuple[str, ...] = ("probe", "free")
CURVE_NAMES: tuple[str, ...] = ("rssm_position", "persistence_position", "floor_position",
                                "rssm_angle", "persistence_angle", "floor_angle")
DECISION_H: int = 15
REPORTED_H: tuple[int, ...] = (5, 15, 45)
FAMILY: int = 6
SEEDS_REQUIRED: int = 2
TERMS: tuple[str, ...] = ("embedding", "reward", "continue", "kl_dyn", "kl_rep")
class StrataNotAPartition(ValueError): ...
def strata_partition(all_paths, train, val, used) -> dict[str, list]: ...
def stratum_summary(traj: Trajectories, horizon: int) -> dict: ...
@dataclass(frozen=True)
class StratumContrast: estimate: float; se: float; z: float; clusters: int
@dataclass(frozen=True)
class ArmInputs:
    gap_free: StratumContrast; gap_probe: StratumContrast
    train_held_gap_final: dict[int, float]        # seed -> gap_closed(45) position on train_held
    per_seed: "dict[int, ArmInputs] | None"
@dataclass(frozen=True)
class GapInputs: arms: dict[str, ArmInputs]; z_fam: float; h: int
class Status(str, Enum): MEMORISATION, PARTIAL_GAP, NO_GAP, INVERTED_GAP, UNRESOLVED_PROBE
@dataclass(frozen=True)
class ArmReading:
    arm: str; status: Status; rule: str
    free_clears_up: bool; free_clears_down: bool; probe_clears_up: bool; probe_clears_down: bool
    seeds_clearing: int; seeds_total: int; train_held_unanimous: bool
@dataclass(frozen=True)
class GapReading: arms: dict[str, ArmReading]; h: int; z_fam: float
def reading_gap(inputs: GapInputs) -> GapReading: ...
def format_reading_gap(reading: GapReading, inputs: GapInputs) -> str: ...
def learning_curve_summary(history: dict, window: int = 100) -> dict: ...

# scripts/split_gap.py (Tasks 5, 6)
EXIT_OK = 0; EXIT_NO_CHECKPOINTS = 11; EXIT_SPLIT_MISMATCH = 12; EXIT_RECORD_MISMATCH = 14
EXIT_SELF_CHECK_FAILED = 30; EXIT_STRATA_NOT_A_PARTITION = 31
def split_gap_record_path(out_dir, arm, seed) -> Path   # split_gap_<arm>_seed<n>.json
def main(argv=None) -> int
```

### The per-cell record (Task 5 writes it, Task 6 pools it, Task 7 reads it)

```
arm, seed, context, horizon, split_seed, device, torch_version, git_sha,
checkpoint_git_sha,                       # the study record's git_sha
probe: {selection_r2, measurable},
episodes: {val: [...names], train_held: [...], train_probe: [...]},
self_check: {reference_position_max_delta, persistence_position_max_delta,
             windows_total_match, windows_episode_match, ok},
strata: { <stratum>: stratum_summary(...) }   # see Task 3 for the keys
```

### Repo facts every writer needs

- The eval fixture `small_buffer` (six 40-step episodes; `pos_x = 3t`, `pos_y = -2t`) is in `tests/eval/conftest.py`. `episode_split(val_fraction=0.2, seed=0)` holds out ONE of six; `window_starts(40, 2, 3) = range(0, 36, 5)` cuts EIGHT windows per episode. `run_job(StudyJob("random_vit", 1), buffer, out, steps=5, seq_len=4, context=2, horizon=3, device="cpu")` trains a cell in ~2 s (`random_vit` needs no downloaded weights).
- Scripts are loaded BY PATH in tests (`importlib.util.spec_from_file_location`), and `trust_horizon.py` loads `diagnose_dynamics.py` the same way via its `_sibling()`; `split_gap.py` does the same to `trust_horizon.py`.
- `study.write_record(path, record)` sanitises numpy arrays/scalars and NaN (`null` + a `nonfinite` map) -- a live record must NOT carry a top-level `nonfinite` key. `study.load_record` restores the floats. `study.git_sha()` is THE sha (asked of the package's checkout).
- `pooling.CellSeries` identity fields (`windows.episode`, `val`, `horizon`, `context`, `device`, `torch_version`) must agree across every cell pooled together; `rung` and `channel` name the reading.
- `trust.crossing_step(model_err, persist_err, moved)` returns `(n,)` float: first strict crossing at/after the first moved step, `horizon + 1` if never, NaN if never moved. `trust.survival(crossings, horizon)` is `(horizon + 1,)` over the finite crossings. `trust.trust_horizon(surv, q)` is the largest h with `surv[h] >= q` (0 if none, -1 if all NaN). `trust.persistence_margin(model_err, persist_err) = persist_err - model_err`.
- `summary.metric_summary(result: RolloutResult, metric)` returns `gap_final`, `steps_degenerate`, `steps_floor_above_persistence`, `gap_finite`, `final_*`, `band_*`, `relative_*`, `n_steps`.
- `KL_FREE_BITS = 0.20` lives in `mbfps.models.rssm`.
- A record's `history` is `{"loss": [20000 floats], "parts": [20000 dicts with keys embedding, reward, continue, kl_dyn, kl_rep]}`.

---

### Task 1: `probe.py` -- `PROBE_EPISODE_LIMIT`, `probe_episodes`, and `fit_probes` routes through it

**Files:**
- Modify: `src/mbfps/eval/probe.py` (`fit_probes`, currently `limit: int = 20` and `used = list(paths)[:limit]`)
- Test: `tests/eval/test_probe.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `PROBE_EPISODE_LIMIT: int = 20`; `probe_episodes(paths, limit=PROBE_EPISODE_LIMIT) -> tuple[list, list]`; `fit_probes(..., limit: int = PROBE_EPISODE_LIMIT, ...)` whose fit set is `probe_episodes(paths, limit)[0]`. Tasks 5 and 6 build the strata from `probe_episodes(train)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_probe.py`, after `test_fit_probes_rejects_an_empty_episode_list` (line ~747). Add `from pathlib import Path` and `import mbfps.eval.probe as probe_module` to the imports, and `PROBE_EPISODE_LIMIT`, `fit_probes`, `probe_episodes` to the `from mbfps.eval.probe import (...)` block.

```python
# ---------------------------------------------------------------------------
# probe_episodes -- the ONE rule for which training episodes the probe saw.
#
# `fit_probes` fits on the first `PROBE_EPISODE_LIMIT` training episodes and
# no others. The split-gap diagnostic evaluates the shipped checkpoints on
# the training episodes the probe did NOT see, so it needs the same rule;
# both call `probe_episodes`, and the default is pinned to the constant so
# re-hardcoding a `20` in either place fails here.
# ---------------------------------------------------------------------------


def test_the_probe_episode_limit_is_one_constant_that_fit_probes_and_probe_episodes_share():
    """Equality to 20 alone would pass again the moment someone re-hardcoded
    the literal in `fit_probes`; the SIGNATURE defaults are what couple the
    two, the way `VAL_FRACTION` couples the three split callers."""
    assert PROBE_EPISODE_LIMIT == 20
    assert inspect.signature(fit_probes).parameters["limit"].default == PROBE_EPISODE_LIMIT
    assert inspect.signature(probe_episodes).parameters["limit"].default == PROBE_EPISODE_LIMIT
    paths = [Path(f"ep_{i:06d}_len00526.npz") for i in range(25)]
    used, held = probe_episodes(paths)
    assert used == paths[:20] and held == paths[20:]
    assert probe_episodes(paths, limit=30) == (paths, [])
    assert probe_episodes(paths[:20]) == (paths[:20], []), "exactly the limit leaves nothing held"
    assert probe_episodes(paths, limit=3) == (paths[:3], paths[3:])
    with pytest.raises(ValueError, match="at least one"):
        probe_episodes(paths, limit=0)


def test_fit_probes_takes_its_fit_set_from_probe_episodes(tmp_path, monkeypatch):
    """The routing, pinned behaviourally: `fit_probes` calls `probe_episodes`
    with its own `paths` and `limit`, and gathers on EXACTLY the `used` list
    that comes back -- so a `fit_probes` that sliced for itself would gather
    the same episodes today and drift from the diagnostic's strata the day
    either rule changed."""
    paths = _write_episodes(tmp_path, [20, 20, 20, 20, 20])
    seen: list[tuple[list, int]] = []
    real_probe_episodes = probe_module.probe_episodes

    def spy(ps, limit=PROBE_EPISODE_LIMIT):
        seen.append((list(ps), limit))
        return real_probe_episodes(ps, limit)

    monkeypatch.setattr(probe_module, "probe_episodes", spy)
    gathered: list[list] = []
    real_gather = probe_module.gather_probe_data

    def gather_spy(model, ps, *args, **kwargs):
        gathered.append(list(ps))
        return real_gather(model, ps, *args, **kwargs)

    monkeypatch.setattr(probe_module, "gather_probe_data", gather_spy)

    _fit(paths, limit=3, select_episodes=0, ridge=1.0)

    assert seen == [(paths, 3)]
    assert gathered == [paths[:3]], "the fit set must be probe_episodes' `used`, nothing else"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -q -k "probe_episode" 2>&1 | tail -5`
Expected: `ImportError: cannot import name 'PROBE_EPISODE_LIMIT'` (collection error).

- [ ] **Step 3: Implement**

In `src/mbfps/eval/probe.py`, immediately above `def fit_probes(` add:

```python
PROBE_EPISODE_LIMIT: int = 20
"""How many TRAINING episodes the rollout's probe is fit on: the first
`PROBE_EPISODE_LIMIT` of the train split in the order `episode_split` returns
it, `select_episodes` of which are held back to select the ridge.

Named because a second consumer now depends on the SAME rule. The split-gap
diagnostic (`scripts/split_gap.py`) evaluates the shipped checkpoints on the
training episodes the probe never saw -- model-seen, probe-unseen -- and
which those are is exactly `probe_episodes(train)[1]`. A bare `20` here and
a bare `20` there would be two rules that agree today; `fit_probes` and the
diagnostic both call `probe_episodes`, and a test pins `fit_probes`'s
default to this constant, so moving it moves both or fails the suite.
"""


def probe_episodes(paths, limit: int = PROBE_EPISODE_LIMIT) -> tuple[list, list]:
    """`(used, held)`: the episodes the probe is fit on and every other one,
    both in the caller's order. `used` is the LEADING block, so a caller that
    hands the train split in `episode_split`'s order gets a deterministic set.
    `held` is empty when `paths` has at most `limit` entries -- legal for the
    probe, and what the split-gap diagnostic refuses as an empty stratum."""
    if limit < 1:
        raise ValueError(f"the probe needs at least one episode; limit={limit}")
    ordered = list(paths)
    return ordered[:limit], ordered[limit:]
```

Then in `fit_probes`: change the signature's `limit: int = 20,` to `limit: int = PROBE_EPISODE_LIMIT,`, and change the body's `used = list(paths)[:limit]` to `used, _ = probe_episodes(paths, limit)`. Add to the `limit:` line of its docstring's Args: `limit: how many episodes to fit on -- \`probe_episodes(paths, limit)[0]\`, the rule the split-gap strata share.`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -q 2>&1 | tail -3`
Expected: all pass, `0 warnings`.

- [ ] **Step 5: Full suite -- behaviour-preserving**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: `1439 passed` (1437 + 2), `0 warnings`. Every existing `fit_probes` test passes unchanged: the fit set is the same list it always was.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/probe.py tests/eval/test_probe.py
git commit -m "feat: probe_episodes is the one rule for which training episodes the probe saw, and fit_probes routes through it

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `Trajectories.band`, and `pooling.unpaired_contrast`

**Files:**
- Modify: `src/mbfps/eval/diagnostics.py` (class `Trajectories` ~line 766; `reference_trajectories` ~line 796)
- Modify: `src/mbfps/eval/pooling.py` (after `paired_contrast`, ~line 583)
- Test: `tests/eval/test_diagnostics.py`, `tests/eval/test_pooling.py`

**Interfaces:**
- Consumes: `RolloutResult` (already imported in `diagnostics.py`), `PooledMean`, `_z`, `IncompatibleCells` (in `pooling.py`).
- Produces: `Trajectories.band: RolloutResult | None = None`, set by `reference_trajectories` from `result.reference`; `UnpairedContrast` and `unpaired_contrast(a: PooledMean, b: PooledMean)`. Task 3's `stratum_summary` reads `traj.band`; Task 6 calls `unpaired_contrast`.

- [ ] **Step 1: Write the failing tests**

In `tests/eval/test_diagnostics.py`, add `from mbfps.eval.rollout import RolloutResult` to the imports (check it is not already there) and append after `test_reference_trajectories_raises_when_no_window_is_long_enough`:

```python
def test_reference_trajectories_carries_the_pass_band(tmp_path):
    """The pass already computes all six mean curves (`_Pass.reference` is a
    `RolloutResult`); `Trajectories` used to copy two of them out. `band` is
    the whole thing, so a consumer can take `gap_closed` on the stratum it
    just evaluated without a second `evaluate_rollout`. The two named curves
    stay and are the band's own -- the self-check reads them by name."""
    paths = [write(tmp_path, varied_action_episode())]
    model, probe = real_model_and_probe()
    result = reference_trajectories(
        model, paths, probe, context=CONTEXT, horizon=HORIZON, seed=7,
        device=torch.device("cpu"), feature_backbone=None,
    )
    band = result.band
    assert isinstance(band, RolloutResult)
    for name in ("rssm_position", "persistence_position", "floor_position",
                 "rssm_angle", "persistence_angle", "floor_angle"):
        assert getattr(band, name).shape == (HORIZON,), name
        assert np.isfinite(getattr(band, name)).all(), name
    np.testing.assert_array_equal(band.rssm_position, result.reference_position)
    np.testing.assert_array_equal(band.persistence_position, result.persistence_position)
    assert band.position_gap_closed().shape == (HORIZON,)
    # Additive: a fabricated Trajectories that reads no band need not build one.
    assert Trajectories.__dataclass_fields__["band"].default is None
```

In `tests/eval/test_pooling.py`, extend the `from mbfps.eval.pooling import (...)` block with `IncompatibleCells, PooledMean, UnpairedContrast, _z, unpaired_contrast` (keep whatever is already imported) and append:

```python
# ---------------------------------------------------------------------------
# unpaired_contrast -- two pools over DIFFERENT windows (two strata of one arm).
# ---------------------------------------------------------------------------


def _pooled(arm, rung, mean, se, *, clusters=24, windows=229, channel="free") -> PooledMean:
    """A `PooledMean` by hand: only the fields the contrast reads carry
    meaning; the rest are shaped correctly and otherwise inert."""
    return PooledMean(
        arm=arm, rung=rung, channel=channel, seeds=(0, 1, 2),
        windows=windows, windows_excluded=0, excluded_windows=(), clusters=clusters,
        series=np.full(windows, float(mean)), mean=float(mean), se=float(se),
        se_independent=float("nan"), z=_z(float(mean), float(se)),
    )


def test_unpaired_contrast_differences_the_means_adds_the_variances_and_takes_the_smaller_cluster_count():
    """0.62 - 0.22 = 0.40; se sqrt(0.03^2 + 0.04^2) = 0.05 EXACTLY (a 3-4-5
    triangle, so no rounding); z = 8. Clusters 24 against 78: the ruler is
    read against the smaller, conservative for the larger."""
    held = _pooled("frozen_ssl", "train_held", 0.62, 0.03, clusters=78, windows=740)
    val = _pooled("frozen_ssl", "val", 0.22, 0.04, clusters=24, windows=229)
    c = unpaired_contrast(held, val)
    assert isinstance(c, UnpairedContrast)
    assert (c.a, c.b, c.channel) == ("frozen_ssl/train_held", "frozen_ssl/val", "free")
    assert c.mean == pytest.approx(0.40)
    assert c.se == pytest.approx(0.05)
    assert c.z == pytest.approx(8.0)
    assert (c.windows_a, c.windows_b, c.clusters) == (740, 229, 24)
    flipped = unpaired_contrast(val, held)
    assert flipped.mean == pytest.approx(-0.40) and flipped.z == pytest.approx(-8.0)


def test_unpaired_contrast_identical_means_give_zero_and_a_lone_ruler_passes_through():
    c = unpaired_contrast(
        _pooled("pixel_ae", "train_held", 0.5, 0.02), _pooled("pixel_ae", "val", 0.5, 0.0)
    )
    assert c.mean == 0.0 and c.se == pytest.approx(0.02) and c.z == 0.0


def test_unpaired_contrast_nan_ruler_propagates_and_never_falls_back():
    """A clustered SE that could not be estimated (one cluster) is NaN, and so
    is the contrast's -- `_z`'s policy, never the naive standard error."""
    c = unpaired_contrast(
        _pooled("pixel_ae", "train_held", 0.5, float("nan"), clusters=1),
        _pooled("pixel_ae", "val", 0.2, 0.01),
    )
    assert c.mean == pytest.approx(0.3)
    assert np.isnan(c.se) and np.isnan(c.z)
    assert c.clusters == 1


def test_unpaired_contrast_refuses_a_different_reading_and_a_pool_against_itself():
    with pytest.raises(IncompatibleCells, match="not the same reading"):
        unpaired_contrast(
            _pooled("pixel_ae", "train_held", 0.5, 0.1, channel="free"),
            _pooled("pixel_ae", "val", 0.5, 0.1, channel="probe"),
        )
    with pytest.raises(IncompatibleCells, match="against itself"):
        unpaired_contrast(_pooled("pixel_ae", "val", 0.5, 0.1), _pooled("pixel_ae", "val", 0.4, 0.1))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_diagnostics.py -q -k carries_the_pass_band 2>&1 | tail -3 && .venv/bin/python -m pytest tests/eval/test_pooling.py -q -k unpaired 2>&1 | tail -3`
Expected: the first fails with `AttributeError: 'Trajectories' object has no attribute 'band'`; the second fails at collection with `ImportError: cannot import name 'UnpairedContrast'`.

- [ ] **Step 3: Implement `Trajectories.band`**

In `src/mbfps/eval/diagnostics.py`, class `Trajectories`: after the last existing field (`persistence_position: np.ndarray`) add:

```python
    band: "RolloutResult | None" = None
    """The pass's `_Pass.reference` -- all six mean curves (`rssm`,
    `persistence`, `floor` x position, angle) -- so a consumer can take
    `gap_closed` on the stratum it just evaluated without a second
    `evaluate_rollout` (M3e). `reference_trajectories` always sets it; the
    default exists so a fabricated `Trajectories` in a test that reads none
    of the band need not build one. `reference_position` and
    `persistence_position` above are `band.rssm_position` and
    `band.persistence_position`, kept under their own names because the
    self-check reads them by name."""
```

In `reference_trajectories`, the `return Trajectories(` call gains one keyword after `persistence_position=result.reference.persistence_position,`:

```python
        band=result.reference,
```

- [ ] **Step 4: Implement `unpaired_contrast`**

In `src/mbfps/eval/pooling.py`, directly after the `paired_contrast` function (before `def _normalised`):

```python
@dataclass(frozen=True)
class UnpairedContrast:
    """`a` minus `b` where the two pools score DIFFERENT windows -- two strata
    of one arm -- so nothing is paired: the standard error is the root sum of
    squares of the two clustered ones, and the cluster count the ruler is
    read against is the SMALLER of the two (conservative for the larger)."""

    a: str          # "<arm>/<rung>" of the first pool
    b: str          # "<arm>/<rung>" of the second
    channel: str
    windows_a: int
    windows_b: int
    clusters: int
    mean: float
    se: float
    z: float


def unpaired_contrast(a: PooledMean, b: PooledMean) -> UnpairedContrast:
    """`a.mean - b.mean`, `se = sqrt(a.se^2 + b.se^2)`, `z = mean / se` under
    `_z`'s zero-ruler policy; a NaN ruler on either side propagates and never
    becomes the naive one.

    The two pools must be the same READING (`channel`) and must not be the
    same pool (`arm` and `rung` both equal -- a stratum against itself). They
    are NOT required to share windows: that is the point. Compare
    `paired_contrast`, whose two arms must score identical windows so it can
    difference them window by window.
    """
    if a.channel != b.channel:
        raise IncompatibleCells(
            f"{a.arm}/{a.rung} ({a.channel}) and {b.arm}/{b.rung} ({b.channel}) are "
            "not the same reading"
        )
    if (a.arm, a.rung) == (b.arm, b.rung):
        raise IncompatibleCells(f"{a.arm}/{a.rung} would be contrasted against itself")
    mean = float(a.mean - b.mean)
    se = float(np.sqrt(a.se ** 2 + b.se ** 2))
    return UnpairedContrast(
        a=f"{a.arm}/{a.rung}", b=f"{b.arm}/{b.rung}", channel=a.channel,
        windows_a=int(a.windows), windows_b=int(b.windows),
        clusters=int(min(a.clusters, b.clusters)),
        mean=mean, se=se, z=_z(mean, se),
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_diagnostics.py tests/eval/test_pooling.py -q 2>&1 | tail -3`
Expected: all pass, `0 warnings`.

- [ ] **Step 6: Full suite -- M3d's pins still hold**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: `1444 passed` (1439 + 5), `0 warnings`. In particular every `test_trust_horizon_script.py` test that builds a `Trajectories(...)` by hand still passes: `band` defaulted.

- [ ] **Step 7: Commit**

```bash
git add src/mbfps/eval/diagnostics.py src/mbfps/eval/pooling.py tests/eval/test_diagnostics.py tests/eval/test_pooling.py
git commit -m "feat: Trajectories carries the pass's whole band, and pooling gains the unpaired contrast two strata need

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `split_gap.py` part 1 -- constants, `strata_partition`, `stratum_summary`

**Files:**
- Create: `src/mbfps/eval/split_gap.py`
- Test: `tests/eval/test_split_gap.py` (new)

**Interfaces:**
- Consumes: `Trajectories` (with `band`, Task 2), `trust.moved_mask / crossing_step / persistence_margin / survival / trust_horizon`, `summary.METRICS / metric_summary`, `trust_readings.Q_REPORTED`.
- Produces: `STRATA`, `DECISION`, `CHANNELS`, `CURVE_NAMES`, `DECISION_H`, `REPORTED_H`, `FAMILY`, `SEEDS_REQUIRED`, `TERMS`, `StrataNotAPartition`, `strata_partition(all_paths, train, val, used) -> dict[str, list]`, `stratum_summary(traj, horizon) -> dict`, `q_key(q) -> str`. Task 5 calls the first two per run / per stratum; Task 6 reads the summary's keys.

The summary's keys (Task 6 and the results section read these by name):

```
windows:       {total: int, episode: [int], clusters: int}
curves:        {<CURVE_NAMES>: (H,) float}
band:          {position: metric_summary(...), angle: metric_summary(...)}
moved:         (n, H) bool
first_moved:   (n,) float -- h0, 1-based; NaN if the window never moves
crossing:      {probe: (n,) float, free: (n,) float}
margin:        {probe: (n, H) float, free: (n, H) float}
survival:      {probe: (H+1,) float, free: (H+1,) float}
trust_horizon: {probe: {q50: int, q75: int, q90: int}, free: {...}}
counts:        {not_moved: (H,) int, never_moved: int}
```

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_split_gap.py`:

```python
"""mbfps.eval.split_gap: the pure functions of the M3e diagnostic, pinned on
fabricated inputs with hand-worked answers.

`strata_partition` on synthetic path lists, each of its refusals provoked one
at a time. `stratum_summary` on the SAME fabricated trajectories
`test_trust_horizon_script.py` uses (two windows, two steps, a biased probe),
so every crossing and margin below is a number that file already derives by
hand -- plus a `band` with curves chosen so `gap_closed` is a clean fraction.
"""

from pathlib import Path

import numpy as np
import pytest

from mbfps.eval.diagnostics import Trajectories
from mbfps.eval.rollout import RolloutResult
from mbfps.eval.split_gap import (
    CHANNELS,
    CURVE_NAMES,
    DECISION,
    DECISION_H,
    FAMILY,
    REPORTED_H,
    SEEDS_REQUIRED,
    STRATA,
    TERMS,
    StrataNotAPartition,
    q_key,
    strata_partition,
    stratum_summary,
)
from mbfps.eval.trust_readings import Q_REPORTED


def test_the_pre_registered_constants_are_the_specs():
    """Spec 2.1 / 3.1 / 3.2 / 2.3, as literals: these are not tuned after the run."""
    assert STRATA == ("val", "train_held", "train_probe")
    assert DECISION == ("train_held", "val")
    assert CHANNELS == ("probe", "free")
    assert CURVE_NAMES == ("rssm_position", "persistence_position", "floor_position",
                           "rssm_angle", "persistence_angle", "floor_angle")
    assert DECISION_H == 15 and REPORTED_H == (5, 15, 45) and DECISION_H in REPORTED_H
    assert FAMILY == 6 and SEEDS_REQUIRED == 2
    assert TERMS == ("embedding", "reward", "continue", "kl_dyn", "kl_rep")
    assert [q_key(q) for q in Q_REPORTED] == ["q50", "q75", "q90"], (
        "JSON keys may not contain '.'; write_record refuses them")


# ---------------------------------------------------------------------------
# strata_partition
# ---------------------------------------------------------------------------


def _paths(n: int) -> list[Path]:
    return [Path(f"ep_{i:06d}_len00526.npz") for i in range(n)]


def test_strata_partition_returns_the_three_lists_in_the_callers_order():
    """30 episodes: val = every fifth, train = the rest (24), used = train[:20]
    -> train_held = train[20:] (4). Order is preserved, never sorted."""
    everything = _paths(30)
    val = everything[::5]
    train = [p for p in everything if p not in val]
    used = train[:20]
    strata = strata_partition(everything, train, val, used)
    assert list(strata) == list(STRATA)
    assert strata["val"] == val
    assert strata["train_probe"] == used
    assert strata["train_held"] == train[20:] and len(strata["train_held"]) == 4
    assert sum(len(v) for v in strata.values()) == 30


def test_strata_partition_refuses_an_overlap_a_gap_a_non_prefix_and_an_empty_decision_stratum():
    everything = _paths(30)
    val = everything[::5]
    train = [p for p in everything if p not in val]
    used = train[:20]
    with pytest.raises(StrataNotAPartition, match="share"):
        strata_partition(everything, train, val + [train[0]], used)
    with pytest.raises(StrataNotAPartition, match="do not cover"):
        strata_partition(everything + [Path("ep_999999_len00526.npz")], train, val, used)
    with pytest.raises(StrataNotAPartition, match="do not cover"):
        strata_partition(everything, train[1:], val, train[1:21])
    with pytest.raises(StrataNotAPartition, match="leading block"):
        strata_partition(everything, train, val, train[1:21])
    # Six episodes, one held out: the probe takes all five training episodes.
    six = _paths(6)
    with pytest.raises(StrataNotAPartition, match="train_held is empty"):
        strata_partition(six, six[1:], six[:1], six[1:])


# ---------------------------------------------------------------------------
# stratum_summary, on fabricated trajectories.
# ---------------------------------------------------------------------------


BIAS = np.array([0.0, 9.0])


def _band() -> RolloutResult:
    """Means of the two rows below -- reference [2, 10.5], persistence
    [9.5, 31] -- with a floor of [1, 1]: gap_closed position = [7.5/8.5,
    20.5/30]. Angle curves chosen so gap_closed angle = [20/25, 20/35]."""
    return RolloutResult(
        horizon=np.array([1, 2]),
        rssm_position=np.array([2.0, 10.5]),
        persistence_position=np.array([9.5, 31.0]),
        floor_position=np.array([1.0, 1.0]),
        rssm_angle=np.array([10.0, 20.0]),
        persistence_angle=np.array([30.0, 40.0]),
        floor_angle=np.array([5.0, 5.0]),
    )


def _fabricated(*, stationary_third_window: bool = False) -> Trajectories:
    """Window 0 a perfect predictor, window 1 a persistence clone, the probe
    biased by BIAS at the anchor (the numbers of
    test_trust_horizon_script._fabricated). With the flag, a THIRD window that
    never moves: its every quantity must be NaN or counted, never pooled.

      persistence err  row 0 [15, 41]  row 1 [4, 21]
      model err        row 0 [0, 0]    row 1 [4, 21]
      -> crossing probe [3, 3] (never; ties never cross); margin probe
         row 0 [15, 41], row 1 [0, 0]
      D_hat row 0 [0, 0] row 1 [5, 5]; D_0 [3, 4] both rows
      -> crossing free [3, 1]; margin free row 0 [3, 4], row 1 [-2, -1]
    """
    true_at_context = np.array([[0.0, 0.0], [10.0, 10.0]])
    true_positions = np.array([[[12.0, 0.0], [40.0, 0.0]], [[10.0, 15.0], [10.0, 40.0]]])
    positions = np.array([[[12.0, 0.0], [40.0, 0.0]], [[10.0, 19.0], [10.0, 19.0]]])
    d_hat = np.array([[0.0, 0.0], [5.0, 5.0]])
    d_0 = np.array([[3.0, 4.0], [3.0, 4.0]])
    e_disp = np.array([[2.0, 4.0], [1.0, 2.0]])
    e_true = np.array([[2.0, 4.0], [2.0, 4.0]])
    episode = np.array([0, 1])
    if stationary_third_window:
        true_at_context = np.vstack([true_at_context, [[5.0, 5.0]]])
        true_positions = np.vstack([true_positions, [[[5.0, 5.0], [5.0, 5.0]]]])
        positions = np.vstack([positions, [[[5.0, 14.0], [5.0, 14.0]]]])
        d_hat = np.vstack([d_hat, [[1.0, 1.0]]])
        d_0 = np.vstack([d_0, [[1.0, 1.0]]])
        e_disp = np.vstack([e_disp, [[0.0, 0.0]]])
        e_true = np.vstack([e_true, [[0.0, 0.0]]])
        episode = np.array([0, 1, 1])
    positions_at_context = true_at_context + BIAS
    d = true_positions - true_at_context[:, None, :]
    n = true_positions.shape[0]
    return Trajectories(
        positions=positions,
        positions_at_context=positions_at_context,
        positions_real=positions_at_context[:, None, :] + 0.5 * d,
        true_positions=true_positions,
        true_at_context=true_at_context,
        embedding_distance_to_truth=d_hat,
        embedding_persistence_distance=d_0,
        embedding_displacement=e_disp,
        true_embedding_displacement=e_true,
        window_episode=episode,
        windows_total=n,
        reference_position=np.linalg.norm(positions - true_positions, axis=-1).mean(axis=0),
        persistence_position=np.linalg.norm(
            positions_at_context[:, None, :] - true_positions, axis=-1
        ).mean(axis=0),
        band=_band(),
    )


def test_stratum_summary_reduces_the_pass_through_trusts_functions():
    s = stratum_summary(_fabricated(), horizon=2)
    assert set(s) == {"windows", "curves", "band", "moved", "first_moved", "crossing",
                      "margin", "survival", "trust_horizon", "counts"}
    assert s["windows"] == {"total": 2, "episode": [0, 1], "clusters": 2}
    assert s["moved"].all() and s["moved"].shape == (2, 2)
    np.testing.assert_array_equal(s["first_moved"], [1.0, 1.0])
    np.testing.assert_array_equal(s["crossing"]["probe"], [3.0, 3.0])
    np.testing.assert_array_equal(s["crossing"]["free"], [3.0, 1.0])
    np.testing.assert_array_equal(s["margin"]["probe"], [[15.0, 41.0], [0.0, 0.0]])
    np.testing.assert_array_equal(s["margin"]["free"], [[3.0, 4.0], [-2.0, -1.0]])
    np.testing.assert_array_equal(s["survival"]["probe"], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(s["survival"]["free"], [1.0, 0.5, 0.5])
    assert s["trust_horizon"]["probe"] == {"q50": 2, "q75": 2, "q90": 2}
    assert s["trust_horizon"]["free"] == {"q50": 2, "q75": 0, "q90": 0}
    np.testing.assert_array_equal(s["counts"]["not_moved"], [0, 0])
    assert s["counts"]["never_moved"] == 0


def test_stratum_summary_carries_the_band_and_its_gap_closed():
    s = stratum_summary(_fabricated(), horizon=2)
    assert set(s["curves"]) == set(CURVE_NAMES)
    np.testing.assert_array_equal(s["curves"]["floor_position"], [1.0, 1.0])
    np.testing.assert_array_equal(s["curves"]["rssm_angle"], [10.0, 20.0])
    position, angle = s["band"]["position"], s["band"]["angle"]
    assert position["gap_final"] == pytest.approx(20.5 / 30.0)
    assert position["steps_degenerate"] == 0 and position["steps_floor_above_persistence"] == 0
    assert position["gap_finite"] == 2 and position["n_steps"] == 2
    assert angle["gap_final"] == pytest.approx(20.0 / 35.0)
    assert angle["final_persistence"] == 40.0


def test_stratum_summary_counts_a_window_that_never_moves_and_pools_nothing_from_it():
    """Row 2 stands still: NaN crossing in both channels, NaN h0, counted in
    `never_moved` and in `not_moved` at every step, and `survival` is
    unchanged because `trust.survival` drops NaN crossings."""
    s = stratum_summary(_fabricated(stationary_third_window=True), horizon=2)
    assert s["windows"] == {"total": 3, "episode": [0, 1, 1], "clusters": 2}
    assert not s["moved"][2].any()
    assert np.isnan(s["first_moved"][2]) and np.isfinite(s["first_moved"][:2]).all()
    assert np.isnan(s["crossing"]["probe"][2]) and np.isnan(s["crossing"]["free"][2])
    np.testing.assert_array_equal(s["survival"]["free"], [1.0, 0.5, 0.5])
    np.testing.assert_array_equal(s["counts"]["not_moved"], [1, 1])
    assert s["counts"]["never_moved"] == 1


def test_stratum_summary_refuses_a_pass_without_a_band_and_a_horizon_that_is_not_the_rows():
    import dataclasses

    with pytest.raises(ValueError, match="band"):
        stratum_summary(dataclasses.replace(_fabricated(), band=None), horizon=2)
    with pytest.raises(ValueError, match="horizon"):
        stratum_summary(_fabricated(), horizon=3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_split_gap.py -q 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'mbfps.eval.split_gap'`.

- [ ] **Step 3: Create the module**

Create `src/mbfps/eval/split_gap.py`:

```python
"""The split gap (M3e): does the world model roll out better on the episodes
it trained on?

Every rollout number the M3 study ever recorded is on the 24 held-out
episodes. The 20,000-step budget is ~430 passes over the 98 training episodes,
so "memorised, does not transfer" has been a live explanation for the h=45
failure since M3b and has never been put to the records. This module is the
pure half of the diagnostic that puts it there: the rule for WHICH training
episodes are model-seen but probe-unseen (`strata_partition`, over
`probe.probe_episodes`), the reduction of one stratum's reference pass to the
quantities the reading needs (`stratum_summary`, through `mbfps.eval.trust`),
the pre-registered reading over pooled stratum contrasts (`reading_gap`,
spec 3.3), and the per-term learning-curve summary read only when the reading
says the gap is not the cause (`learning_curve_summary`, spec 2.3).

Pure over numpy arrays and dicts: no torch, no files, no randomness. The
driver (`scripts/split_gap.py`) does the loading, the passes, the pooling and
the printing.
"""

from dataclasses import dataclass
from enum import Enum
from itertools import combinations

import numpy as np

from mbfps.eval.diagnostics import Trajectories
from mbfps.eval.summary import METRICS, metric_summary
from mbfps.eval.trust import (
    crossing_step,
    moved_mask,
    persistence_margin,
    survival,
    trust_horizon,
)
from mbfps.eval.trust_readings import Q_REPORTED

STRATA: tuple[str, ...] = ("val", "train_held", "train_probe")
"""Spec 2.1. `val` is the record's own and the self-check anchor; `train_held`
-- model-seen, probe-unseen -- is the decision stratum; `train_probe` is
confounded by construction (the probe saw it) and is information only."""
DECISION: tuple[str, str] = ("train_held", "val")
"""The two strata every contrast is between, in the orientation printed:
`train_held - val`, positive when the model does better on episodes it saw."""
CHANNELS: tuple[str, ...] = ("probe", "free")
"""`free` (embedding space, the training target's own units) decides; `probe`
(position, through the refit probe) is the control twin."""
CURVE_NAMES: tuple[str, ...] = (
    "rssm_position", "persistence_position", "floor_position",
    "rssm_angle", "persistence_angle", "floor_angle",
)
DECISION_H: int = 15
"""Spec 3.2: the governing spec's imagination horizon, pre-registered."""
REPORTED_H: tuple[int, ...] = (5, 15, 45)
"""Printed beside the decision horizon with their z; not decided on."""
FAMILY: int = 6
"""Spec 3.1: three arms x two channels at one horizon."""
SEEDS_REQUIRED: int = 2
"""Spec 3.3: a gap "clears ... pooled and within >= 2 of 3 seeds"."""
TERMS: tuple[str, ...] = ("embedding", "reward", "continue", "kl_dyn", "kl_rep")
"""The keys of every `history.parts[i]` a study record carries."""


def q_key(q: float) -> str:
    """`0.75 -> "q75"`: a record key may not contain '.', since `write_record`
    addresses non-finite fields by dotted path."""
    return f"q{int(round(q * 100))}"


# ---------------------------------------------------------------------------
# The strata.
# ---------------------------------------------------------------------------


class StrataNotAPartition(ValueError):
    """The three strata are not pairwise disjoint, do not cover the buffer,
    `train_probe` is not the leading block of `train`, or `train_held` is
    empty. A CODE (or fixture) defect, never a data one: `scripts/split_gap.py`
    maps it to exit 31 before any probe is refit."""


def strata_partition(all_paths, train, val, used) -> dict[str, list]:
    """`{"val": val, "train_held": train[len(used):], "train_probe": used}`,
    every list in the caller's order, after checking they partition
    `all_paths` (spec 2.4, exit 31).

    `used` must be the leading block of `train` -- what `probe_episodes(train)`
    returns -- so `train_held` is the rest of `train` and nothing else.
    """
    all_names = sorted(p.name for p in all_paths)
    train, val, used = list(train), list(val), list(used)
    if used != train[: len(used)]:
        raise StrataNotAPartition(
            "train_probe is not the leading block of train: probe_episodes and the "
            "split disagree on which episodes the probe was fit on"
        )
    held = train[len(used):]
    strata = {"val": val, "train_held": held, "train_probe": used}
    names = {name: [p.name for p in paths] for name, paths in strata.items()}
    for a, b in combinations(STRATA, 2):
        shared = sorted(set(names[a]) & set(names[b]))
        if shared:
            raise StrataNotAPartition(
                f"{a} and {b} share {len(shared)} episode(s), e.g. {shared[0]}"
            )
    union = sorted(n for stratum in names.values() for n in stratum)
    if union != all_names:
        missing = sorted(set(all_names) - set(union))
        extra = sorted(set(union) - set(all_names))
        raise StrataNotAPartition(
            f"the strata do not cover the buffer: {len(missing)} episode(s) missing"
            f"{f' (e.g. {missing[0]})' if missing else ''}, {len(extra)} not in the buffer"
            f"{f' (e.g. {extra[0]})' if extra else ''}"
        )
    if not held:
        raise StrataNotAPartition(
            f"train_held is empty: train has {len(train)} episode(s) and the probe used "
            f"{len(used)}, so there is no model-seen, probe-unseen stratum to decide on"
        )
    return strata


# ---------------------------------------------------------------------------
# One stratum's pass, reduced.
# ---------------------------------------------------------------------------


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2-D Euclidean distance over the last axis -- `position_error`'s norm."""
    return np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), axis=-1)


def stratum_summary(traj: Trajectories, horizon: int) -> dict:
    """Everything the reading and the results section read off one stratum
    of one cell, every quantity through the pure functions of
    `mbfps.eval.trust` or `mbfps.eval.summary` (spec 2.2).

    The moved mask is the TRUTH's (`moved_mask`), so both channels score the
    same rows. The probe channel's errors are the probe's positions against
    the true positions; the free channel's are the pass's two embedding
    series, `D_hat(h)` and `D_0(h)`, the training target's own units. The
    band is the pass's whole `RolloutResult`, summarised by `metric_summary`
    exactly as `run_job` summarises the val band, so `gap_final` here on
    `train_held` is spec 4.1's criterion on the train side.
    """
    if traj.band is None:
        raise ValueError(
            "Trajectories.band is None: the pass was not reference_trajectories, or a "
            "fabricated Trajectories carries no band"
        )
    moved = moved_mask(traj.true_positions, traj.true_at_context)
    model_err = _distance(traj.positions, traj.true_positions)
    persist_err = _distance(traj.positions_at_context[:, None, :], traj.true_positions)
    d_hat = np.asarray(traj.embedding_distance_to_truth, dtype=float)
    d_0 = np.asarray(traj.embedding_persistence_distance, dtype=float)
    expected = (int(traj.windows_total), int(horizon))
    if model_err.shape != expected or d_hat.shape != expected or d_0.shape != expected:
        raise ValueError(
            f"the rows are not (windows_total, horizon) = {expected}: positions give "
            f"{model_err.shape}, the embedding series {d_hat.shape} and {d_0.shape}"
        )
    crossing = {
        "probe": crossing_step(model_err, persist_err, moved),
        "free": crossing_step(d_hat, d_0, moved),
    }
    margin = {
        "probe": persistence_margin(model_err, persist_err),
        "free": persistence_margin(d_hat, d_0),
    }
    surv = {channel: survival(crossing[channel], horizon) for channel in CHANNELS}
    episode = np.asarray(traj.window_episode, dtype=int)
    return {
        "windows": {
            "total": int(traj.windows_total),
            "episode": [int(e) for e in episode],
            "clusters": int(np.unique(episode).size),
        },
        "curves": {name: np.asarray(getattr(traj.band, name), dtype=float) for name in CURVE_NAMES},
        "band": {metric: metric_summary(traj.band, metric) for metric in METRICS},
        "moved": moved,
        # h0 per window, 1-based: `crossing_step`'s own search start. NaN where
        # the window never moves within the horizon.
        "first_moved": np.where(moved.any(axis=1), moved.argmax(axis=1) + 1.0, np.nan),
        "crossing": crossing,
        "margin": margin,
        "survival": surv,
        "trust_horizon": {
            channel: {q_key(q): trust_horizon(surv[channel], q) for q in Q_REPORTED}
            for channel in CHANNELS
        },
        "counts": {
            "not_moved": (~moved).sum(axis=0),
            "never_moved": int((~moved.any(axis=1)).sum()),
        },
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_split_gap.py -q 2>&1 | tail -3`
Expected: 7 passed, `0 warnings`. If `METRICS` is not exported by `mbfps.eval.summary`, check its name there (`grep -n "^METRICS" src/mbfps/eval/summary.py`) and import what exists.

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: `1451 passed` (1444 + 7), `0 warnings`.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/split_gap.py tests/eval/test_split_gap.py
git commit -m "feat: split_gap.py part 1 -- the three strata as a checked partition, and one stratum's pass reduced through trust

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `split_gap.py` part 2 -- Reading G, its formatting, and the learning-curve summary

**Files:**
- Modify: `src/mbfps/eval/split_gap.py` (append)
- Test: `tests/eval/test_split_gap.py` (append)

**Interfaces:**
- Consumes: the constants of Task 3.
- Produces: `StratumContrast`, `ArmInputs`, `GapInputs`, `Status`, `ArmReading`, `GapReading`, `clears(z, z_fam)`, `train_held_passes_gate(gap_final)`, `reading_gap(inputs) -> GapReading`, `format_reading_gap(reading, inputs) -> str`, `learning_curve_summary(history, window=100) -> dict`. Task 6 builds `GapInputs` from pooled numbers and prints `format_reading_gap`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_split_gap.py`. Extend the `from mbfps.eval.split_gap import (...)` block with `ArmInputs, ArmReading, GapInputs, GapReading, Status, StratumContrast, clears, format_reading_gap, learning_curve_summary, reading_gap, train_held_passes_gate`.

```python
# ---------------------------------------------------------------------------
# Reading G, on fabricated pooled tables. z_fam is 3.0 throughout; every
# fabricated z is either clearly above (+4), clearly below (+1), clearly
# inverted (-4), or exactly the bar (3.0, which must NOT clear).
# ---------------------------------------------------------------------------

Z_FAM = 3.0


def _contrast(z: float) -> StratumContrast:
    return StratumContrast(estimate=0.1 * z, se=0.1, z=z, clusters=24)


def _arm(free_z, probe_z, *, seeds=(4.0, 4.0, 4.0), gate=(0.2, 0.3, 0.1)) -> ArmInputs:
    """Pooled contrasts plus three per-seed leaves (each with its own free z
    and the SAME probe z) and train_held's gap_closed(45) per seed."""
    per_seed = {
        seed: ArmInputs(
            gap_free=_contrast(z), gap_probe=_contrast(probe_z),
            train_held_gap_final={seed: gate[seed]}, per_seed=None,
        )
        for seed, z in enumerate(seeds)
    }
    return ArmInputs(
        gap_free=_contrast(free_z), gap_probe=_contrast(probe_z),
        train_held_gap_final={s: gate[s] for s in range(3)}, per_seed=per_seed,
    )


def _inputs(**arms) -> GapInputs:
    return GapInputs(arms=arms, z_fam=Z_FAM, h=DECISION_H)


def test_clears_is_strict_and_never_on_a_nan_or_infinite_z():
    assert clears(3.01, Z_FAM) and not clears(3.0, Z_FAM) and not clears(2.99, Z_FAM)
    assert not clears(float("nan"), Z_FAM) and not clears(float("inf"), Z_FAM)
    assert not clears(4.0, float("nan")), "a NaN bar cannot be cleared"


def test_train_held_passes_gate_is_spec_4_1_unanimity_with_nan_not_positive():
    assert train_held_passes_gate({0: 0.2, 1: 0.01, 2: 0.9})
    assert not train_held_passes_gate({0: 0.2, 1: 0.0, 2: 0.9}), "zero is not > 0"
    assert not train_held_passes_gate({0: 0.2, 1: -0.1, 2: 0.9})
    assert not train_held_passes_gate({0: 0.2, 1: float("nan"), 2: 0.9}), (
        "a NaN gap_closed is a non-positive band and is not > 0")
    assert not train_held_passes_gate({}), "no seed is not unanimity"


def test_memorisation_needs_the_pooled_gap_two_seeds_and_the_train_side_gate():
    reading = reading_gap(_inputs(frozen_ssl=_arm(4.0, 1.0)))
    r = reading.arms["frozen_ssl"]
    assert r.status is Status.MEMORISATION
    assert r.free_clears_up and not r.probe_clears_up and not r.probe_clears_down
    assert (r.seeds_clearing, r.seeds_total, r.train_held_unanimous) == (3, 3, True)
    assert "passes" in r.rule and "4.1" in r.rule
    assert (reading.h, reading.z_fam) == (DECISION_H, Z_FAM)


def test_partial_gap_when_one_train_held_seed_does_not_beat_persistence_at_45():
    r = reading_gap(_inputs(frozen_ssl=_arm(4.0, 1.0, gate=(0.2, -0.05, 0.1)))).arms["frozen_ssl"]
    assert r.status is Status.PARTIAL_GAP and not r.train_held_unanimous
    assert "FAILS" in r.rule
    nan = reading_gap(_inputs(a=_arm(4.0, 1.0, gate=(0.2, float("nan"), 0.1)))).arms["a"]
    assert nan.status is Status.PARTIAL_GAP, "a NaN band on one seed is not unanimity"


def test_a_pooled_gap_that_only_one_seed_shows_is_no_gap():
    r = reading_gap(_inputs(frozen_ssl=_arm(4.0, 1.0, seeds=(4.0, 1.0, 1.0)))).arms["frozen_ssl"]
    assert r.status is Status.NO_GAP and r.seeds_clearing == 1
    assert "only 1 of 3" in r.rule
    two = reading_gap(_inputs(frozen_ssl=_arm(4.0, 1.0, seeds=(4.0, 4.0, 1.0)))).arms["frozen_ssl"]
    assert two.status is Status.MEMORISATION and two.seeds_clearing == 2, "two of three suffice"


def test_no_gap_when_the_free_contrast_does_not_clear_even_if_the_probe_does():
    r = reading_gap(_inputs(random_vit=_arm(1.0, 4.0))).arms["random_vit"]
    assert r.status is Status.NO_GAP and r.probe_clears_up and not r.free_clears_up
    assert "does not clear" in r.rule
    bar = reading_gap(_inputs(random_vit=_arm(3.0, 1.0))).arms["random_vit"]
    assert bar.status is Status.NO_GAP, "z equal to the bar does not clear"


def test_inverted_gap_when_train_held_is_worse_than_val():
    r = reading_gap(_inputs(pixel_ae=_arm(-4.0, -1.0))).arms["pixel_ae"]
    assert r.status is Status.INVERTED_GAP and r.free_clears_down
    also_probe = reading_gap(_inputs(pixel_ae=_arm(-4.0, -4.0))).arms["pixel_ae"]
    assert also_probe.status is Status.INVERTED_GAP, "both channels inverted agree; not unresolved"


def test_unresolved_probe_when_the_two_channels_clear_with_opposite_signs_and_it_wins_precedence():
    up_down = reading_gap(_inputs(pixel_ae=_arm(4.0, -4.0))).arms["pixel_ae"]
    assert up_down.status is Status.UNRESOLVED_PROBE and "opposite signs" in up_down.rule
    down_up = reading_gap(_inputs(pixel_ae=_arm(-4.0, 4.0))).arms["pixel_ae"]
    assert down_up.status is Status.UNRESOLVED_PROBE
    # Precedence: the SAME inputs that would be MEMORISATION become unresolved
    # when the probe twin clears the other way.
    assert reading_gap(_inputs(a=_arm(4.0, 1.0))).arms["a"].status is Status.MEMORISATION
    assert reading_gap(_inputs(a=_arm(4.0, -4.0))).arms["a"].status is Status.UNRESOLVED_PROBE


def test_reading_gap_reads_every_arm_independently_in_the_callers_order():
    reading = reading_gap(_inputs(
        pixel_ae=_arm(-4.0, -1.0), frozen_ssl=_arm(4.0, 1.0), random_vit=_arm(1.0, 1.0),
    ))
    assert list(reading.arms) == ["pixel_ae", "frozen_ssl", "random_vit"]
    assert [r.status for r in reading.arms.values()] == [
        Status.INVERTED_GAP, Status.MEMORISATION, Status.NO_GAP,
    ]


def test_reading_gap_with_no_per_seed_leaves_never_reaches_memorisation():
    """Per-seed inputs absent (None): zero seeds clear, so the pooled gap can
    at most be NO_GAP with the seed count named -- never MEMORISATION."""
    arm = ArmInputs(gap_free=_contrast(4.0), gap_probe=_contrast(1.0),
                    train_held_gap_final={0: 0.2}, per_seed=None)
    r = reading_gap(_inputs(a=arm)).arms["a"]
    assert r.status is Status.NO_GAP and (r.seeds_clearing, r.seeds_total) == (0, 0)


def test_format_reading_gap_prints_each_arm_with_its_status_and_rule():
    inputs = _inputs(frozen_ssl=_arm(4.0, 1.0), random_vit=_arm(1.0, 1.0))
    text = format_reading_gap(reading_gap(inputs), inputs)
    assert f"h={DECISION_H}" in text and f"z_fam = {Z_FAM:.2f}" in text
    assert "frozen_ssl" in text and "MEMORISATION" in text
    assert "random_vit" in text and "NO GAP" in text
    assert "decided by:" in text
    assert "train_held - val" in text
    for column in ("free", "probe"):
        assert column in text


# ---------------------------------------------------------------------------
# learning_curve_summary, on a synthetic history.
# ---------------------------------------------------------------------------


def _history(n: int = 400) -> dict:
    """embedding descends 1 -> 0.5 linearly, reward ascends 0 -> 1, the other
    three terms are flat; loss is their sum."""
    i = np.arange(n, dtype=float)
    embedding = 1.0 - 0.5 * i / (n - 1)
    reward = i / (n - 1)
    parts = [
        {"embedding": float(embedding[k]), "reward": float(reward[k]), "continue": 0.7,
         "kl_dyn": 0.3, "kl_rep": 0.3}
        for k in range(n)
    ]
    loss = [sum(p.values()) for p in parts]
    return {"loss": loss, "parts": parts}


def test_learning_curve_summary_reads_quarters_and_the_smoothed_minimum():
    s = learning_curve_summary(_history(400), window=100)
    assert (s["steps"], s["window"], s["quarter"]) == (400, 100, 100)
    assert set(s["terms"]) == {"loss", *TERMS}
    emb = s["terms"]["embedding"]
    assert emb["last_quarter_mean"] == pytest.approx(1.0 - 0.5 * 349.5 / 399)
    assert emb["preceding_quarter_mean"] == pytest.approx(1.0 - 0.5 * 249.5 / 399)
    assert emb["descending"] and emb["change_pct"] < 0
    assert emb["smoothed_min_step"] == 400, "still descending: the minimum is the last window"
    rew = s["terms"]["reward"]
    assert not rew["descending"] and rew["change_pct"] > 0
    assert rew["smoothed_min_step"] == 100, "ascending: the minimum is the FIRST full window"
    flat = s["terms"]["kl_dyn"]
    assert flat["change_pct"] == pytest.approx(0.0) and not flat["descending"]
    assert s["terms"]["loss"]["last_quarter_mean"] == pytest.approx(
        emb["last_quarter_mean"] + rew["last_quarter_mean"] + 1.3
    )


def test_learning_curve_summary_shrinks_the_window_to_the_history_and_refuses_junk():
    short = learning_curve_summary(_history(8), window=100)
    assert short["window"] == 8 and short["quarter"] == 2
    with pytest.raises(ValueError, match="four"):
        learning_curve_summary(_history(3))
    with pytest.raises(ValueError, match="window"):
        learning_curve_summary(_history(8), window=0)
    broken = _history(8)
    broken["parts"] = broken["parts"][:-1]
    with pytest.raises(ValueError, match="disagree"):
        learning_curve_summary(broken)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_split_gap.py -q 2>&1 | tail -3`
Expected: `ImportError: cannot import name 'ArmInputs'`.

- [ ] **Step 3: Implement**

Append to `src/mbfps/eval/split_gap.py`:

```python
# ---------------------------------------------------------------------------
# Reading G -- the generalisation gap (spec 3.3), over pooled inputs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StratumContrast:
    """`pooling.unpaired_contrast` reduced to what the rules read."""

    estimate: float
    se: float
    z: float
    clusters: int


@dataclass(frozen=True)
class ArmInputs:
    """One arm's pooled numbers at the decision horizon: the two stratum
    contrasts (`train_held - val`), spec 4.1's criterion on `train_held` per
    seed, and the same inputs within each seed alone (leaves carry
    `per_seed=None`)."""

    gap_free: StratumContrast
    gap_probe: StratumContrast
    train_held_gap_final: dict[int, float]
    per_seed: "dict[int, ArmInputs] | None"


@dataclass(frozen=True)
class GapInputs:
    arms: dict[str, ArmInputs]
    z_fam: float
    h: int


class Status(str, Enum):
    """Spec 3.3's five outcomes, in the table's order of precedence."""

    UNRESOLVED_PROBE = "unresolved through the probe"
    MEMORISATION = "memorisation"
    PARTIAL_GAP = "partial gap"
    INVERTED_GAP = "inverted gap"
    NO_GAP = "no gap"


@dataclass(frozen=True)
class ArmReading:
    arm: str
    status: Status
    rule: str
    free_clears_up: bool
    free_clears_down: bool
    probe_clears_up: bool
    probe_clears_down: bool
    seeds_clearing: int
    seeds_total: int
    train_held_unanimous: bool


@dataclass(frozen=True)
class GapReading:
    arms: dict[str, ArmReading]
    h: int
    z_fam: float


def clears(z: float, z_fam: float) -> bool:
    """STRICTLY above the bar. NaN never clears (nothing to read); an infinite
    z never clears either -- `pooling._z` returns +-inf for a zero standard
    error, a degenerate ruler. `trust_readings._clears`'s policy, restated."""
    return bool(np.isfinite(z) and np.isfinite(z_fam) and z > z_fam)


def train_held_passes_gate(gap_final: dict[int, float]) -> bool:
    """Spec 3.3 / governing spec 4.1 on the train side: `gap_closed(45)` on
    position > 0 in ALL seeds of the arm, where a NaN (a non-positive
    persistence-to-floor band) is not > 0. No seed at all is not unanimity."""
    values = list(gap_final.values())
    return bool(values) and all(np.isfinite(v) and v > 0.0 for v in values)


def _fmt(value: float, spec: str = "+.2f") -> str:
    return format(value, spec) if np.isfinite(value) else str(value)


def _gate_detail(gap_final: dict[int, float]) -> str:
    return "gap_closed(45) per seed " + " / ".join(
        f"s{seed} {_fmt(gap_final[seed], '+.3f')}" for seed in sorted(gap_final)
    )


def _arm_reading(arm: str, a: ArmInputs, z_fam: float) -> ArmReading:
    """The rules of spec 3.3, in the table's precedence, each status carrying
    the sentence that decided it."""
    free_up, free_down = clears(a.gap_free.z, z_fam), clears(-a.gap_free.z, z_fam)
    probe_up, probe_down = clears(a.gap_probe.z, z_fam), clears(-a.gap_probe.z, z_fam)
    per_seed = a.per_seed or {}
    seeds_clearing = sum(1 for leaf in per_seed.values() if clears(leaf.gap_free.z, z_fam))
    unanimous = train_held_passes_gate(a.train_held_gap_final)
    fz, pz, bar = _fmt(a.gap_free.z), _fmt(a.gap_probe.z), f"{z_fam:.2f}"
    seeds = f"{seeds_clearing} of {len(per_seed)} seeds"
    if (free_up and probe_down) or (free_down and probe_up):
        status = Status.UNRESOLVED_PROBE
        rule = f"G_free z {fz} and G_probe z {pz} both clear +-{bar} with opposite signs"
    elif free_up and seeds_clearing >= SEEDS_REQUIRED and unanimous:
        status = Status.MEMORISATION
        rule = (f"G_free z {fz} > {bar} pooled and in {seeds}; train_held passes spec 4.1 "
                f"({_gate_detail(a.train_held_gap_final)})")
    elif free_up and seeds_clearing >= SEEDS_REQUIRED:
        status = Status.PARTIAL_GAP
        rule = (f"G_free z {fz} > {bar} pooled and in {seeds}; train_held FAILS spec 4.1 "
                f"({_gate_detail(a.train_held_gap_final)})")
    elif free_down:
        status = Status.INVERTED_GAP
        rule = f"G_free z {fz} < -{bar}: train_held is worse than val"
    elif free_up:
        status = Status.NO_GAP
        rule = (f"G_free z {fz} clears {bar} pooled but in only {seeds} "
                f"(>= {SEEDS_REQUIRED} required)")
    else:
        status = Status.NO_GAP
        rule = f"G_free z {fz} does not clear +-{bar}"
    return ArmReading(
        arm=arm, status=status, rule=rule,
        free_clears_up=free_up, free_clears_down=free_down,
        probe_clears_up=probe_up, probe_clears_down=probe_down,
        seeds_clearing=seeds_clearing, seeds_total=len(per_seed),
        train_held_unanimous=unanimous,
    )


def reading_gap(inputs: GapInputs) -> GapReading:
    """One `ArmReading` per arm, in the caller's order; arms never read each
    other (spec 4: no ranking)."""
    return GapReading(
        arms={arm: _arm_reading(arm, a, inputs.z_fam) for arm, a in inputs.arms.items()},
        h=inputs.h, z_fam=inputs.z_fam,
    )


def format_reading_gap(reading: GapReading, inputs: GapInputs) -> str:
    """The contrast table and the verdict lines, in `trust.txt`'s style."""
    lines = [
        f"--- Reading G: the generalisation gap at h={reading.h} (train_held - val); "
        f"z_fam = {reading.z_fam:.2f} ---",
        f"  {'arm':<12}{'channel':<9}{'estimate':>10}{'se':>9}{'z':>8}  clears",
    ]
    for arm, a in inputs.arms.items():
        for channel, c in (("free", a.gap_free), ("probe", a.gap_probe)):
            verdict = "yes" if clears(abs(c.z), reading.z_fam) else "no"
            lines.append(
                f"  {arm:<12}{channel:<9}{_fmt(c.estimate, '+.4f'):>10}{_fmt(c.se, '.4f'):>9}"
                f"{_fmt(c.z):>8}  {verdict}"
            )
    lines.append("  train_held, spec 4.1 (gap_closed(45) position > 0 in every seed):")
    for arm, a in inputs.arms.items():
        word = "passes" if train_held_passes_gate(a.train_held_gap_final) else "FAILS"
        lines.append(f"    {arm:<12}{_gate_detail(a.train_held_gap_final)} -> {word}")
    for arm, r in reading.arms.items():
        lines.append(
            f"  verdict: {arm:<12}{r.status.name.replace('_', ' ')} -- decided by: {r.rule}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Learning curves (spec 2.3): descriptive, no verdict.
# ---------------------------------------------------------------------------


def _smoothed(series: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(series, np.ones(window) / window, mode="valid")


def learning_curve_summary(history: dict, window: int = 100) -> dict:
    """Per term (and the summed loss): the mean over the last quarter of
    training against the mean over the quarter before it -- the statistic the
    M3b write-up used on the summed loss -- as sign and percentage, and the
    1-based step at which the `window`-step moving mean is lowest (the step
    at the END of the minimising window). `window` shrinks to the history's
    length when the history is shorter than it."""
    loss = np.asarray(history["loss"], dtype=float)
    parts = list(history["parts"])
    n = int(loss.size)
    if n < 4:
        raise ValueError(f"a learning curve needs at least four steps for quarters, got {n}")
    if len(parts) != n:
        raise ValueError(
            f"history.loss ({n}) and history.parts ({len(parts)}) disagree on the step count"
        )
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    window = min(int(window), n)
    series = {"loss": loss}
    for term in TERMS:
        series[term] = np.asarray([float(p[term]) for p in parts], dtype=float)
    quarter = n // 4
    terms = {}
    for name, s in series.items():
        last = float(s[n - quarter:].mean())
        previous = float(s[n - 2 * quarter: n - quarter].mean())
        smoothed = _smoothed(s, window)
        terms[name] = {
            "last_quarter_mean": last,
            "preceding_quarter_mean": previous,
            "change_pct": float(100.0 * (last - previous) / previous) if previous != 0.0 else float("nan"),
            "descending": bool(last < previous),
            "smoothed_min_step": int(np.argmin(smoothed)) + window,
            "smoothed_min": float(smoothed.min()),
            "smoothed_final": float(smoothed[-1]),
        }
    return {"steps": n, "window": window, "quarter": int(quarter), "terms": terms}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_split_gap.py -q 2>&1 | tail -3`
Expected: 20 passed, `0 warnings`.

- [ ] **Step 5: Mutation check, by hand, three ways**

Temporarily (do NOT commit) make each edit below, run `.venv/bin/python -m pytest tests/eval/test_split_gap.py -q -x`, confirm the named test fails, then revert with `git checkout src/mbfps/eval/split_gap.py`:

| mutation | test that must fail |
|---|---|
| in `_arm_reading`, swap the `MEMORISATION` and `PARTIAL_GAP` branches' `unanimous` conditions | `test_partial_gap_when_one_train_held_seed_...` |
| `clears`: `z > z_fam` -> `z >= z_fam` | `test_clears_is_strict...` and `test_no_gap_when_the_free_contrast_does_not_clear...` (the bar case) |
| `train_held_passes_gate`: drop `np.isfinite(v) and` | `test_train_held_passes_gate_...` and the NaN case of `test_partial_gap_...` |
| `_arm_reading`: check `UNRESOLVED_PROBE` AFTER `MEMORISATION` | `test_unresolved_probe_..._wins_precedence` |
| `learning_curve_summary`: `+ window` -> `+ window - 1` on `smoothed_min_step` | `test_learning_curve_summary_reads_quarters_...` |

- [ ] **Step 6: Full suite and commit**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: `1464 passed` (1451 + 13), `0 warnings`.

```bash
git add src/mbfps/eval/split_gap.py tests/eval/test_split_gap.py
git commit -m "feat: the pre-registered rules of Reading G over pooled stratum contrasts, and the per-term learning-curve summary

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `scripts/split_gap.py` part 1 -- load as `trust_horizon.py` does, the five checks in order, three passes per cell, one record per cell

**Files:**
- Create: `scripts/split_gap.py`
- Modify: `scripts/trust_horizon.py` (extract `Prepared` + `prepare_cell()` from `_run_cell`; behaviour-preserving)
- Modify: `tests/eval/conftest.py` (`small_buffer` -> `_build_buffer(tmp_path, n)` + `wide_buffer`)
- Test: `tests/eval/test_split_gap_script.py` (new); `tests/eval/test_trust_horizon_script.py` (one pin test)

**Interfaces:**
- Consumes: `trust_horizon.py`'s `Cell`, `CellMissing`, `load_cell`, `self_check`, `probe_is_measurable`, the new `prepare_cell`, and its four exit statuses (by path); `probe_episodes`; `reference_trajectories`; `strata_partition`, `stratum_summary`; `study.write_record`, `study.git_sha`, `study.SPLIT_SEED`.
- Produces: `EXIT_*` (0, 11, 12, 14, 30, 31), `split_gap_record_path(out_dir, arm, seed)`, `split_gap_record(...)`, `_run_cell(...)`, `main(argv) -> int` that -- in THIS task -- stops after the per-cell loop and prints the self-check and strata tables. Task 6 appends the pooling and readings to `main`.

- [ ] **Step 1: The fixture -- `_build_buffer(tmp_path, n)` and `wide_buffer`**

In `tests/eval/conftest.py`, replace the body of `small_buffer` with a call to a count-parametrised builder and add `wide_buffer`. The six-episode behaviour is byte-identical (same `fill` values 1..6, same seed, same feature RNG order):

```python
def _build_buffer(tmp_path, n_episodes: int) -> ReplayBuffer:
    """`n_episodes` synthetic 40-step episodes with cached features, one
    per study backbone at its own row width. `fill` runs 1..n so the
    six-episode fixture is exactly what it always was."""
    buf = ReplayBuffer(tmp_path / "data", capacity_transitions=100_000)
    for fill in range(1, n_episodes + 1):
        t = 40
        steps = np.arange(t)
        obs = np.zeros((t + 1, *OBS_SHAPE), dtype=np.uint8)
        obs[:, 0, 0, 0] = np.arange(t + 1, dtype=np.uint8)
        obs[:, 0, 0, 1] = fill
        buf.add(Episode(
            obs=obs,
            actions=(steps % 6).astype(np.int32),
            rewards=(steps % 3).astype(np.float32),
            terminated=(steps == t - 1),
            truncated=np.zeros(t, dtype=bool),
            privileged=np.stack([
                np.full(t + 1, 100.0),
                np.arange(t + 1, dtype=np.float32) * 3.0,
                np.arange(t + 1, dtype=np.float32) * -2.0,
                np.zeros(t + 1),
                (np.arange(t + 1) * 7.0) % 360.0,
            ], axis=1).astype(np.float32),
            privileged_keys=KEYS,
            policy_name="random",
            seed=fill,
            scenario="my_way_home",
        ))
    rng = np.random.default_rng(0)
    for path in buf.episode_paths():
        # One cache per study backbone, each at ITS OWN row width: the ViT
        # arms read (64, 384), pixel_ae reads (64, 32). A pixel_ae cache
        # written 384 wide would be refused by the loader's geometry check.
        for suffix, width in ((".features.npy", 384),
                              (".features_random_vit.npy", 384),
                              (".features_pixel_ae.npy", 32)):
            np.save(path.with_suffix(suffix),
                    rng.random((41, 64, width)).astype(np.float16))
    return buf


@pytest.fixture
def small_buffer(tmp_path):
    """Six episodes long enough for context+horizon, with cached features.

    Six, not four: `episode_split(val_fraction=0.2)` floors to zero validation
    episodes below five and raises.
    """
    return _build_buffer(tmp_path, 6)


@pytest.fixture
def wide_buffer(tmp_path):
    """Thirty episodes: `episode_split(0.2)` holds out six, leaving 24 to
    train on -- more than `PROBE_EPISODE_LIMIT`, so the split-gap strata are
    non-empty: train_probe 20, train_held 4, val 6."""
    return _build_buffer(tmp_path, 30)
```

Run: `.venv/bin/python -m pytest tests/eval -q -x 2>&1 | tail -2`
Expected: green -- every consumer of `small_buffer` sees the same six episodes.

- [ ] **Step 2: Write the failing script tests**

Create `tests/eval/test_split_gap_script.py`:

```python
"""scripts/split_gap.py: load as trust_horizon does, the five checks in order,
three passes per cell, one record per cell; then (Task 6) the pooling, Reading
G, the learning curves and split_gap.txt.

The script is loaded by path. The cell under test is a REAL tiny cell on
`wide_buffer` -- thirty 40-step episodes, `run_job(..., steps=5, seq_len=4,
context=2, horizon=3, device="cpu")` -- and `diagnose_dynamics.main` writes
the diagnostic the self-check needs, so `max |delta| == 0.0` is asserted
against a diagnostic the ladder really wrote on this machine.

THE FIXTURE'S FACTS: `episode_split(30, 0.2, seed=0)` holds out SIX; the 24
training episodes split as probe_episodes' 20 + 4; `window_starts(40, 2, 3)`
cuts EIGHT windows per episode, so the strata carry 48 / 32 / 160 windows over
6 / 4 / 20 clusters. `pos_x = 3t`, `pos_y = -2t`: every window moves
sqrt(13) = 3.61 units per step, under the 5-unit threshold at h=1, over it
from h=2, so `first_moved` is 2 everywhere and no window never moves.
"""

import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import mbfps.eval.study as study
from mbfps.eval.probe import PROBE_EPISODE_LIMIT
from mbfps.eval.split_gap import CURVE_NAMES, STRATA
from mbfps.eval.study import StudyJob, job_record_path, load_record, run_job
from mbfps.utils.config import ARMS

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script("split_gap")
diagnose = _load_script("diagnose_dynamics")
trust = _load_script("trust_horizon")

JOB = StudyJob("random_vit", 1)
JOB_KW = dict(steps=5, seq_len=4, context=2, horizon=3, device="cpu")
CONTEXT, HORIZON = JOB_KW["context"], JOB_KW["horizon"]
WINDOWS_PER_EPISODE = 8
N_EPISODES = {"val": 6, "train_held": 4, "train_probe": PROBE_EPISODE_LIMIT}
RECORD = "result_random_vit_seed1.json"
DIAGNOSTIC = "diagnostic_random_vit_seed1.json"
SPLIT_GAP = "split_gap_random_vit_seed1.json"

EXPECTED_KEYS = {
    "arm", "seed", "context", "horizon", "split_seed", "device", "torch_version",
    "git_sha", "checkpoint_git_sha", "probe", "episodes", "self_check", "strata",
    "nonfinite",
}
STRATUM_KEYS = {
    "windows", "curves", "band", "moved", "first_moved", "crossing", "margin",
    "survival", "trust_horizon", "counts",
}


@pytest.fixture
def cell(tmp_path, wide_buffer, capsys):
    """One real cell on thirty episodes and the diagnostic the ladder writes for it."""
    out = tmp_path / "out"
    run_job(JOB, wide_buffer, out, **JOB_KW)
    status = diagnose.main([
        "--out", str(out), "--data", str(wide_buffer.root), "--device", "cpu",
        "--arms", JOB.arm, "--seeds", str(JOB.seed),
        "--context", str(CONTEXT), "--horizon", str(HORIZON), "--ks", "1", str(HORIZON),
    ])
    assert status == diagnose.EXIT_OK, "the fixture's diagnostic was not written cleanly"
    capsys.readouterr()
    return types.SimpleNamespace(out=out, data=wide_buffer.root)


@pytest.fixture
def narrow_cell(tmp_path, small_buffer, capsys):
    """The SIX-episode cell: five training episodes, all of them the probe's.
    The strata cannot be built and the script must say so."""
    out = tmp_path / "out"
    run_job(JOB, small_buffer, out, **JOB_KW)
    status = diagnose.main([
        "--out", str(out), "--data", str(small_buffer.root), "--device", "cpu",
        "--arms", JOB.arm, "--seeds", str(JOB.seed),
        "--context", str(CONTEXT), "--horizon", str(HORIZON), "--ks", "1", str(HORIZON),
    ])
    assert status == diagnose.EXIT_OK
    capsys.readouterr()
    return types.SimpleNamespace(out=out, data=small_buffer.root)


def _argv(cell, *extra: str) -> list[str]:
    return [
        "--out", str(cell.out), "--data", str(cell.data), "--device", "cpu",
        "--arms", JOB.arm, "--seeds", str(JOB.seed), *extra,
    ]


def _doctor(path: Path, edit) -> None:
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


def _never_refit(*args, **kwargs):
    """Patched onto `script._trust`: the refit lives in trust_horizon's
    `prepare_cell`, which split_gap calls by path."""
    raise AssertionError("fit_probes ran; this refusal must come before the refit")


FAULTS = {
    "diagnostic": lambda cell: (cell.out / DIAGNOSTIC).unlink(),
    "split": lambda cell: _doctor(
        cell.out / RECORD,
        lambda r: r["episodes"].__setitem__("val", ["ep_000099_len00040.npz"]),
    ),
    "record": lambda cell: _doctor(
        cell.out / RECORD,
        lambda r: r["curves"]["rssm_position"].__setitem__(
            0, r["curves"]["rssm_position"][0] + 6.5
        ),
    ),
    "self_check": lambda cell: _doctor(
        cell.out / DIAGNOSTIC,
        lambda d: d["curves"]["reference_position"].__setitem__(
            0, d["curves"]["reference_position"][0] + 1e-3
        ),
    ),
}


# ---------------------------------------------------------------------------
# The parser and the statuses.
# ---------------------------------------------------------------------------


def test_the_parser_defaults_match_trust_horizons_and_add_the_figure_and_window():
    args = script._parser().parse_args([])
    assert args.out == Path("runs/m3_study_v2") and args.data == Path("data/my_way_home")
    assert args.device == "mps"
    assert args.context is None and args.horizon is None
    assert args.arms == list(ARMS) and args.seeds == [0, 1, 2]
    assert args.figure is None, "None -> <out>/learning_curves.png"
    assert args.window == 100
    with pytest.raises(SystemExit) as refused:
        script._parser().parse_args(["--arms", "cnn"])
    assert refused.value.code == 2


def test_the_reused_statuses_are_trust_horizons_and_thirty_one_is_new():
    assert script.EXIT_OK == 0
    assert script.EXIT_NO_CHECKPOINTS == trust.EXIT_NO_CHECKPOINTS == 11
    assert script.EXIT_SPLIT_MISMATCH == trust.EXIT_SPLIT_MISMATCH == 12
    assert script.EXIT_RECORD_MISMATCH == trust.EXIT_RECORD_MISMATCH == 14
    assert script.EXIT_SELF_CHECK_FAILED == trust.EXIT_SELF_CHECK_FAILED == 30
    assert script.EXIT_STRATA_NOT_A_PARTITION == 31
    mine = {k: v for k, v in vars(script).items() if k.startswith("EXIT_")}
    assert set(mine) == {
        "EXIT_OK", "EXIT_NO_CHECKPOINTS", "EXIT_SPLIT_MISMATCH", "EXIT_RECORD_MISMATCH",
        "EXIT_SELF_CHECK_FAILED", "EXIT_STRATA_NOT_A_PARTITION",
    }
    assert len(set(mine.values())) == len(mine)
    assert 31 not in {v for k, v in vars(trust).items() if k.startswith("EXIT_")}
    assert 31 not in {v for k, v in vars(diagnose).items() if k.startswith("EXIT_")}


# ---------------------------------------------------------------------------
# A clean cell.
# ---------------------------------------------------------------------------


def test_a_clean_cell_exits_ok_and_writes_a_record_with_three_strata(cell):
    assert script.main(_argv(cell)) == script.EXIT_OK
    path = cell.out / SPLIT_GAP
    assert path.exists()
    record = load_record(path)
    assert set(record) == EXPECTED_KEYS
    assert (record["arm"], record["seed"]) == (JOB.arm, JOB.seed)
    assert (record["context"], record["horizon"], record["split_seed"]) == (CONTEXT, HORIZON, 0)
    assert record["device"] == "cpu" and record["torch_version"] == torch.__version__
    assert record["git_sha"] == study.git_sha()
    study_record = load_record(job_record_path(cell.out, JOB))
    assert record["checkpoint_git_sha"] == study_record["git_sha"]
    assert record["self_check"]["ok"] is True
    assert record["self_check"]["reference_position_max_delta"] == 0.0
    assert record["self_check"]["persistence_position_max_delta"] == 0.0

    episodes = record["episodes"]
    assert list(episodes) == list(STRATA)
    assert {k: len(v) for k, v in episodes.items()} == N_EPISODES
    assert episodes["val"] == study_record["episodes"]["val"]
    names = [n for v in episodes.values() for n in v]
    assert len(names) == len(set(names)) == 30, "a partition: disjoint and complete"

    for stratum in STRATA:
        block = record["strata"][stratum]
        assert set(block) == STRATUM_KEYS, stratum
        n = N_EPISODES[stratum] * WINDOWS_PER_EPISODE
        assert block["windows"] == {
            "total": n,
            "episode": [e for e in range(N_EPISODES[stratum]) for _ in range(WINDOWS_PER_EPISODE)],
            "clusters": N_EPISODES[stratum],
        }
        for channel in ("probe", "free"):
            assert len(block["crossing"][channel]) == n
            assert np.asarray(block["margin"][channel], dtype=float).shape == (n, HORIZON)
            assert len(block["survival"][channel]) == HORIZON + 1
            assert set(block["trust_horizon"][channel]) == {"q50", "q75", "q90"}
        assert set(block["curves"]) == set(CURVE_NAMES)
        assert all(len(block["curves"][name]) == HORIZON for name in CURVE_NAMES)
        assert set(block["band"]) == {"position", "angle"}
        assert "gap_final" in block["band"]["position"]
        assert np.asarray(block["moved"], dtype=bool).shape == (n, HORIZON)
        assert block["first_moved"] == [2.0] * n, "sqrt(13) per step: moved from h=2"
        assert block["counts"]["never_moved"] == 0
        assert block["counts"]["not_moved"] == [n, 0, 0]
    # The val stratum's band IS the study record's curve, bitwise.
    assert record["strata"]["val"]["curves"]["rssm_position"] == study_record["curves"]["rssm_position"]
    assert record["strata"]["val"]["curves"]["persistence_position"] == study_record["curves"]["persistence_position"]


def test_the_strata_are_probe_episodes_split_of_the_train_list(cell):
    """train_probe is the first PROBE_EPISODE_LIMIT training episodes in the
    split's order, train_held the rest -- the rule `fit_probes` used."""
    from mbfps.data.buffer import ReplayBuffer
    from mbfps.data.split import VAL_FRACTION, episode_split

    assert script.main(_argv(cell)) == script.EXIT_OK
    record = load_record(cell.out / SPLIT_GAP)
    train, val = episode_split(
        ReplayBuffer(cell.data, capacity_transitions=10**9).episode_paths(),
        val_fraction=VAL_FRACTION, seed=0,
    )
    assert record["episodes"]["train_probe"] == [p.name for p in train[:PROBE_EPISODE_LIMIT]]
    assert record["episodes"]["train_held"] == [p.name for p in train[PROBE_EPISODE_LIMIT:]]
    assert record["episodes"]["val"] == [p.name for p in val]


def test_the_per_cell_line_names_the_strata_and_the_self_check(cell, capsys):
    assert script.main(_argv(cell)) == script.EXIT_OK
    out = capsys.readouterr().out
    assert "random_vit seed 1:" in out
    assert "val 48 windows / 6 episodes" in out
    assert "train_held 32 windows / 4 episodes" in out
    assert "train_probe 160 windows / 20 episodes" in out
    assert "self-check max|delta| reference 0.0e+00 persistence 0.0e+00" in out


# ---------------------------------------------------------------------------
# The refusals, in order.
# ---------------------------------------------------------------------------


def test_a_missing_diagnostic_is_exit_11(cell, capsys):
    FAULTS["diagnostic"](cell)
    assert script.main(_argv(cell)) == script.EXIT_NO_CHECKPOINTS
    assert DIAGNOSTIC in capsys.readouterr().out
    assert not (cell.out / SPLIT_GAP).exists()


def test_a_six_episode_buffer_is_exit_31_before_any_refit(narrow_cell, capsys, monkeypatch):
    """Five training episodes, all the probe's: no train_held. Judged once,
    up front, before the first probe refit."""
    monkeypatch.setattr(script._trust, "fit_probes", _never_refit)
    assert script.main(_argv(narrow_cell)) == script.EXIT_STRATA_NOT_A_PARTITION
    out = capsys.readouterr().out
    assert "STRATA NOT A PARTITION" in out and "train_held is empty" in out
    assert not (narrow_cell.out / SPLIT_GAP).exists()


def test_a_probe_rule_that_disagrees_with_the_split_is_exit_31(cell, capsys, monkeypatch):
    """`probe_episodes` returning a `used` that is not the leading block of
    train -- the drift the partition check exists for."""
    monkeypatch.setattr(script._trust, "fit_probes", _never_refit)
    real = script.probe_episodes

    def skewed(paths, limit=PROBE_EPISODE_LIMIT):
        used, held = real(paths, limit)
        return used[1:] + held[:1], held[1:] + used[:1]

    monkeypatch.setattr(script, "probe_episodes", skewed)
    assert script.main(_argv(cell)) == script.EXIT_STRATA_NOT_A_PARTITION
    assert "leading block" in capsys.readouterr().out
    assert not (cell.out / SPLIT_GAP).exists()


def test_a_split_that_is_not_the_records_is_exit_12_before_the_refit(cell, capsys, monkeypatch):
    monkeypatch.setattr(script._trust, "fit_probes", _never_refit)
    FAULTS["split"](cell)
    assert script.main(_argv(cell)) == script.EXIT_SPLIT_MISMATCH
    assert "SPLIT MISMATCH for random_vit seed 1" in capsys.readouterr().out
    assert not (cell.out / SPLIT_GAP).exists()


def test_a_protocol_that_disagrees_with_the_diagnostic_is_exit_14_before_the_refit(cell, capsys, monkeypatch):
    monkeypatch.setattr(script._trust, "fit_probes", _never_refit)
    assert script.main(_argv(cell, "--context", "3")) == script.EXIT_RECORD_MISMATCH
    out = capsys.readouterr().out
    assert "RECORD MISMATCH for random_vit seed 1" in out and "--context 3" in out


def test_a_study_record_the_rollout_no_longer_reproduces_is_exit_14(cell, capsys):
    FAULTS["record"](cell)
    assert script.main(_argv(cell)) == script.EXIT_RECORD_MISMATCH
    out = capsys.readouterr().out
    assert "RECORD MISMATCH for random_vit seed 1" in out and "device=cpu" in out
    assert not (cell.out / SPLIT_GAP).exists()


def test_a_doctored_reference_curve_is_exit_30_and_no_record_is_written(cell, capsys):
    FAULTS["self_check"](cell)
    assert script.main(_argv(cell)) == script.EXIT_SELF_CHECK_FAILED
    out = capsys.readouterr().out
    assert "SELF-CHECK FAILED for random_vit seed 1" in out and "reference_position" in out
    assert not (cell.out / SPLIT_GAP).exists()


@pytest.mark.parametrize(
    "earlier, later, status",
    [
        ("split", "self_check", "EXIT_SPLIT_MISMATCH"),
        ("record", "self_check", "EXIT_RECORD_MISMATCH"),
        ("diagnostic", "split", "EXIT_NO_CHECKPOINTS"),
    ],
)
def test_the_checks_are_judged_in_the_order_11_31_12_14_30(cell, earlier, later, status):
    FAULTS[earlier](cell)
    FAULTS[later](cell)
    assert script.main(_argv(cell)) == getattr(script, status)
    assert not (cell.out / SPLIT_GAP).exists()


def test_an_absent_requested_cell_is_exit_11(cell, capsys):
    base = ["--out", str(cell.out), "--data", str(cell.data), "--device", "cpu"]
    assert script.main(base) == script.EXIT_NO_CHECKPOINTS
    assert "pixel_ae seed 0" in capsys.readouterr().out
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_split_gap_script.py -q 2>&1 | tail -3`
Expected: collection error -- `FileNotFoundError` / `spec_from_file_location` on `scripts/split_gap.py`.

- [ ] **Step 3b: Extract `prepare_cell` in `trust_horizon.py` (behaviour-preserving)**

The block of `trust_horizon._run_cell` from the split-mismatch check through the `evaluate_rollout` reproduction check is exactly what `split_gap.py` must do before its own passes. Rather than a second copy, lift it into a function both scripts call. Its prints, their order and the statuses are unchanged.

In `scripts/trust_horizon.py`, add `RolloutResult` to the `from mbfps.eval.rollout import evaluate_rollout` line, and insert directly ABOVE `def _run_cell(`:

```python
@dataclass(frozen=True)
class Prepared:
    """One cell past its checks: the loaded model, the refit probe, the
    protocol kwargs every pass takes, the resolved protocol, and the val
    rollout that reproduced the record bitwise."""

    model: object
    embedding_probe: dict
    common: dict
    context: int
    horizon: int
    reference: RolloutResult


def prepare_cell(args, cell: Cell, device, train, val) -> tuple[int, Prepared | None]:
    """12, the protocol (14), the refit, the reproduction (14): everything a
    per-cell pass needs before it can start, and every refusal that comes
    before a pass. `(EXIT_OK, Prepared)` once the record reproduces;
    `(status, None)` with the refusal printed. Shared by this script's
    `_run_cell` and by `scripts/split_gap.py` (imported by path), so the two
    tools refuse a cell for the same reasons in the same words."""
    arm, seed = cell.arm, cell.seed
    names = [p.name for p in val]
    if cell.record["episodes"]["val"] != names:
        print(
            f"\nSPLIT MISMATCH for {arm} seed {seed}: the held-out episodes are not "
            "the ones the record was scored on.\n"
            f"  split:  {names}\n  record: {cell.record['episodes']['val']}"
        )
        return EXIT_SPLIT_MISMATCH, None

    mismatch = protocol_mismatch(args, cell.diagnostic)
    if mismatch is not None:
        print(
            f"\nRECORD MISMATCH for {arm} seed {seed}: {mismatch}. A rollout at "
            "another protocol cannot reproduce the record's curve, so nothing is refit."
        )
        return EXIT_RECORD_MISMATCH, None
    context = int(cell.diagnostic["context"]) if args.context is None else args.context
    horizon = int(cell.diagnostic["horizon"]) if args.horizon is None else args.horizon

    cfg = get_config(arm, seed=seed, device=args.device)
    model = load_checkpoint_model(args.out, arm, seed, cfg, device)
    backbone = encoder_backbone(cfg.encoder)
    # The probe is REFIT at the rollout's own context/horizon and at the
    # cell's seed, exactly as the ladder refit it: it is applied to latents
    # filtered from a zero state for exactly `context` real frames.
    _, embedding_probe = fit_probes(
        model, train, backbone, device, context=context, horizon=horizon, seed=seed,
    )
    common = dict(
        context=context, horizon=horizon, seed=seed, device=device, feature_backbone=backbone,
    )
    reference = evaluate_rollout(model, val, embedding_probe, **common)
    reproduction, step = _max_delta(
        reference.rssm_position, cell.record["curves"]["rssm_position"]
    )
    if reproduction != 0.0:
        print(
            f"\nRECORD MISMATCH for {arm} seed {seed}: evaluate_rollout no longer "
            f"reproduces the study record's curves.rssm_position (max abs "
            f"{reproduction:.3e} at step {step}). Measured, the records reproduce "
            f"bitwise on mps and miss on cpu by an arm-dependent 6-12 map units. "
            f"This run used device={device} torch={torch.__version__}."
        )
        return EXIT_RECORD_MISMATCH, None
    return EXIT_OK, Prepared(
        model=model, embedding_probe=embedding_probe, common=common,
        context=context, horizon=horizon, reference=reference,
    )
```

Then replace the body of `_run_cell` from its first statement through the `return EXIT_RECORD_MISMATCH, None` that follows the reproduction check with:

```python
    arm, seed = cell.arm, cell.seed
    status, prepared = prepare_cell(args, cell, device, train, val)
    if status != EXIT_OK:
        return status, None
    context, horizon = prepared.context, prepared.horizon
    model, embedding_probe, common = prepared.model, prepared.embedding_probe, prepared.common
```

so that the remainder of `_run_cell` (`traj = reference_trajectories(model, val, embedding_probe, **common)` onward) is untouched.

Append to `tests/eval/test_trust_horizon_script.py`, after `test_a_study_record_the_rollout_no_longer_reproduces_is_exit_14`:

```python
def test_prepare_cell_is_the_check_and_refit_block_and_run_cell_uses_it(cell, monkeypatch):
    """The extraction is behaviour-preserving: `prepare_cell` on the clean
    fixture returns EXIT_OK and a `Prepared` whose val rollout IS the
    record's curve, and `_run_cell` reaches its pass through it -- so the
    two scripts that share it refuse a cell for one set of reasons."""
    import argparse

    from mbfps.data.buffer import ReplayBuffer
    from mbfps.data.split import VAL_FRACTION, episode_split
    from mbfps.utils.device import get_device

    args = argparse.Namespace(out=cell.out, device="cpu", context=None, horizon=None)
    loaded = script.load_cell(cell.out, JOB.arm, JOB.seed)
    train, val = episode_split(
        ReplayBuffer(cell.data, capacity_transitions=10**9).episode_paths(),
        val_fraction=VAL_FRACTION, seed=0,
    )
    status, prepared = script.prepare_cell(args, loaded, get_device(prefer="cpu"), train, val)
    assert status == script.EXIT_OK and isinstance(prepared, script.Prepared)
    assert (prepared.context, prepared.horizon) == (CONTEXT, HORIZON)
    assert prepared.common["context"] == CONTEXT and prepared.common["seed"] == JOB.seed
    assert list(prepared.reference.rssm_position) == loaded.record["curves"]["rssm_position"]

    calls = []
    real = script.prepare_cell

    def spy(*a, **k):
        calls.append(a)
        return real(*a, **k)

    monkeypatch.setattr(script, "prepare_cell", spy)
    assert script.main(_argv(cell)) == script.EXIT_OK
    assert len(calls) == 1 and calls[0][1].arm == JOB.arm
```

Run: `.venv/bin/python -m pytest tests/eval/test_trust_horizon_script.py -q 2>&1 | tail -3`
Expected: every existing test passes unchanged plus the new one, `0 warnings`.

- [ ] **Step 4: Create the script**

Create `scripts/split_gap.py`:

```python
"""The split gap over the M3c cells: load, check, three strata per cell, one record per cell; then pool, read the gap, draw the learning curves.

Every rollout number the M3 study recorded is on the 24 held-out episodes. The
20,000-step budget is ~430 passes over the 98 training episodes, and whether
the h=45 failure is memorisation has never been measured. This script
evaluates each frozen checkpoint on THREE strata of episodes under the ONE
refit probe its record was scored with:

  val          the record's 24 held-out episodes -- the self-check anchor
  train_held   train[PROBE_EPISODE_LIMIT:] -- 78 episodes the model trained on
               and the probe never saw -- THE DECISION STRATUM
  train_probe  train[:PROBE_EPISODE_LIMIT] -- the probe's own 20; confounded
               by construction and labelled so wherever it is printed

and writes one `split_gap_<arm>_seed<n>.json` per cell holding, per stratum,
the band and its gap_closed, the crossing step in both channels, the
persistence margin in both channels, the survival curve and H*_q, and the
counts -- `mbfps.eval.split_gap.stratum_summary`. After the per-cell loop,
`main` pools the LIVE records, decides Reading G (spec 3.3), prints the
sensitivity lines and the learning-curve table, draws `learning_curves.png`
and writes `split_gap.txt`.

LOADING IS `trust_horizon.py`'S, NOT A SECOND COPY. `Cell`, `load_cell`,
`self_check`, `protocol_mismatch` and the checkpoint/diagnostic paths are
imported from that script by path -- the way every script test loads a
script, and the way trust_horizon itself imports diagnose_dynamics -- so the
three tools read one cell through one loader.

THE CHECKS RUN IN A FIXED ORDER and each has its own status:

  EXIT_NO_CHECKPOINTS (11)          a REQUESTED cell lacks its checkpoint,
                                    study record or diagnostic. Every cell is
                                    loaded before anything else.
  EXIT_STRATA_NOT_A_PARTITION (31)  NEW. The three strata are not a partition
                                    of the buffer, `train_probe` is not what
                                    `probe_episodes` hands `fit_probes`, or
                                    `train_held` is empty. Judged ONCE, up
                                    front, before any refit: a code or
                                    fixture defect, never a data one.
  EXIT_SPLIT_MISMATCH (12)          the split by name is not the record's.
  EXIT_RECORD_MISMATCH (14)         `evaluate_rollout` on val no longer
                                    reproduces the record's curves.rssm_position
                                    (an ENVIRONMENT difference: cpu misses by
                                    6-12 map units), or --context/--horizon
                                    disagree with the diagnostic's protocol.
  EXIT_SELF_CHECK_FAILED (30)       the val stratum's mean curves are not
                                    bitwise the diagnostic's, or its windows
                                    are not the diagnostic's. Same windows,
                                    same rollout, same refit probe -- or the
                                    train strata are not read against the
                                    ruler M3d validated.

11 / 12 / 14 / 30 carry `trust_horizon.py`'s meanings on purpose; 31 is in no
other tool's range (run_study 1/3-6/23, report_study 7-10, spike 10, diagnose
11-17, pool 18-22, trust 30, argparse 2, a traceback 1).
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

import mbfps.eval.pooling as pooling
from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.aggregate import SEEDS
from mbfps.eval.diagnostics import Trajectories, reference_trajectories
from mbfps.eval.probe import probe_episodes
from mbfps.eval.split_gap import (
    CHANNELS,
    DECISION,
    DECISION_H,
    FAMILY,
    REPORTED_H,
    STRATA,
    TERMS,
    ArmInputs,
    GapInputs,
    StrataNotAPartition,
    StratumContrast,
    format_reading_gap,
    learning_curve_summary,
    q_key,
    reading_gap,
    strata_partition,
    stratum_summary,
)
from mbfps.eval.study import SPLIT_SEED, git_sha, write_record
from mbfps.eval.trust import survival, trust_horizon
from mbfps.eval.trust_readings import ARMS_ORDER, Q_REPORTED
from mbfps.models.rssm import KL_FREE_BITS
from mbfps.utils.config import ARMS
from mbfps.utils.device import get_device


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_split_gap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_trust = _sibling("trust_horizon")
Cell = _trust.Cell
CellMissing = _trust.CellMissing
load_cell = _trust.load_cell
self_check = _trust.self_check
prepare_cell = _trust.prepare_cell
probe_is_measurable = _trust.probe_is_measurable

EXIT_OK = _trust.EXIT_OK
EXIT_NO_CHECKPOINTS = _trust.EXIT_NO_CHECKPOINTS
EXIT_SPLIT_MISMATCH = _trust.EXIT_SPLIT_MISMATCH
EXIT_RECORD_MISMATCH = _trust.EXIT_RECORD_MISMATCH
EXIT_SELF_CHECK_FAILED = _trust.EXIT_SELF_CHECK_FAILED
EXIT_STRATA_NOT_A_PARTITION = 31
"""0 / 11 / 12 / 14 / 30 are `trust_horizon.py`'s, imported so they cannot
drift; 31 is new and in no other tool's range."""


# ---------------------------------------------------------------------------
# One cell: the record.
# ---------------------------------------------------------------------------


def split_gap_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    """One file per cell, named by BOTH the arm and the seed."""
    return Path(out_dir) / f"split_gap_{arm}_seed{seed}.json"


def split_gap_record(
    cell: Cell, strata_traj: dict, strata_paths: dict, check, *, context, horizon, device
) -> dict:
    """The LIVE record for one cell: numpy arrays and real NaNs; `write_record`
    sanitises it on the way to disk, so no top-level `nonfinite` key here."""
    curves = cell.diagnostic["curves"]
    return {
        "arm": cell.arm,
        "seed": cell.seed,
        "context": int(context),
        "horizon": int(horizon),
        "split_seed": SPLIT_SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "git_sha": git_sha(),
        "checkpoint_git_sha": str(cell.record.get("git_sha", "unknown")),
        "probe": {
            "selection_r2": float(cell.diagnostic["probe"]["embedding_selection_r2"]),
            "measurable": bool(probe_is_measurable(
                {
                    "persistence": curves["persistence_position"][-1],
                    "floor": curves["floor_position"][-1],
                },
                widest_se=0.0,
            )),
        },
        "episodes": {name: [p.name for p in strata_paths[name]] for name in STRATA},
        "self_check": check.record(),
        "strata": {name: stratum_summary(strata_traj[name], horizon) for name in STRATA},
    }


def write_split_gap_record(out_dir: Path, record: dict) -> Path:
    path = split_gap_record_path(out_dir, record["arm"], record["seed"])
    write_record(path, record)
    return path


# ---------------------------------------------------------------------------
# The driver.
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/m3_study_v2"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    # None -> each cell's diagnostic says what it was written at; a value that
    # disagrees is refused (`protocol_mismatch`, 14).
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--figure", type=Path, default=None,
                        help="learning-curve figure; default <out>/learning_curves.png")
    parser.add_argument("--window", type=int, default=100,
                        help="moving-mean window (steps) for the learning curves")
    return parser


def _run_cell(args, cell: Cell, device, train, val, strata: dict) -> tuple[int, dict | None]:
    """One cell: trust_horizon's checks and refit (12, 14, 14) through
    `prepare_cell`, then the val pass and 30, the two train passes, the record."""
    arm, seed = cell.arm, cell.seed
    status, prepared = prepare_cell(args, cell, device, train, val)
    if status != EXIT_OK:
        return status, None
    model, embedding_probe, common = prepared.model, prepared.embedding_probe, prepared.common

    # val FIRST: it is the anchor, and a failed self-check costs no train pass.
    traj = {"val": reference_trajectories(model, val, embedding_probe, **common)}
    check = self_check(traj["val"], cell.diagnostic)
    if not check.ok:
        print(
            f"\nSELF-CHECK FAILED for {arm} seed {seed}: " + "; ".join(check.failures())
            + ". Same windows, same rollout, same refit probe -- or the train strata "
            "would be read against a ruler M3d did not validate. No record written."
        )
        return EXIT_SELF_CHECK_FAILED, None
    for name in ("train_held", "train_probe"):
        traj[name] = reference_trajectories(model, strata[name], embedding_probe, **common)

    record = split_gap_record(
        cell, traj, strata, check,
        context=prepared.context, horizon=prepared.horizon, device=device,
    )
    path = write_split_gap_record(args.out, record)
    counts = "; ".join(
        f"{name} {traj[name].windows_total} windows / {len(strata[name])} episodes"
        for name in STRATA
    )
    print(
        f"{arm} seed {seed}: {counts}; self-check max|delta| reference "
        f"{check.reference_position_max_delta:.1e} persistence "
        f"{check.persistence_position_max_delta:.1e}; wrote {path}"
    )
    return EXIT_OK, record


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = get_device(prefer=args.device)
    try:
        cells = [load_cell(args.out, arm, seed) for arm in args.arms for seed in args.seeds]
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS

    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    train, val = episode_split(
        buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED
    )
    # The strata are the SAME for every cell (one split, one probe rule), so
    # they are built and judged once, before the first refit.
    used, _ = probe_episodes(train)
    try:
        strata = {"val": val, **strata_partition(buffer.episode_paths(), train, val, used)}
    except StrataNotAPartition as error:
        print(f"STRATA NOT A PARTITION: {error}")
        return EXIT_STRATA_NOT_A_PARTITION

    records: dict[tuple[str, int], dict] = {}
    horizon = 0
    for cell in cells:
        status, record = _run_cell(args, cell, device, train, val, strata)
        if status != EXIT_OK:
            return status
        records[(cell.arm, cell.seed)] = record
        horizon = int(record["horizon"])
    # Task 6 appends the pooling, Reading G, the learning curves and split_gap.txt here.
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
```

(`strata_partition` already returns the `val` key; the `{"val": val, **...}` spread is harmless and keeps the dict's key order `val, train_held, train_probe`.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_split_gap_script.py -q 2>&1 | tail -3`
Expected: 16 passed (the fixture takes ~5-8 s per test that uses it), `0 warnings`. If `test_the_per_cell_line_...` fails on the exact window counts, check the fixture facts in the module docstring against the printed line before touching the format.

- [ ] **Step 6: Full suite and commit**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: `1481 passed` (1464 + 16 + 1 in `test_trust_horizon_script.py`), `0 warnings`; every pre-existing `test_trust_horizon_script.py` test passes unchanged.

```bash
git add scripts/split_gap.py scripts/trust_horizon.py tests/eval/conftest.py tests/eval/test_split_gap_script.py tests/eval/test_trust_horizon_script.py
git commit -m "feat: split_gap.py part 1 -- trust_horizon's check-and-refit block becomes prepare_cell, shared; five checks in order, three strata per cell, one record per cell

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: `scripts/split_gap.py` part 2 -- pooling glue, Reading G, sensitivity, learning curves, `split_gap.txt`, exit-status distinctness

**Files:**
- Modify: `scripts/split_gap.py` (insert the pooling glue and text before `_parser`; extend `main`)
- Modify: `tests/eval/test_diagnose_dynamics_script.py` (`test_every_exit_status_is_distinct_...`)
- Test: `tests/eval/test_split_gap_script.py` (append)

**Interfaces:**
- Consumes: the records of Task 5; `pooling.CellSeries / pool_arm / require_compatible / unpaired_contrast / cluster_threshold`; `split_gap.reading_gap / format_reading_gap / learning_curve_summary / GapInputs / ArmInputs / StratumContrast`; `trust.survival / trust_horizon`.
- Produces: `survival_series`, `margin_series`, `arm_inputs`, `gap_inputs`, `decision_horizon`, `survival_by_arm_and_stratum`, `_conditional_table` (over `trust_horizon.unmoved_fraction` / `conditional_survival` by path), `readings_text`, `write_readings`, `write_learning_curves`; `main` prints and writes everything.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_split_gap_script.py`:

```python
# ---------------------------------------------------------------------------
# Task 6: the pooling glue, Reading G, the learning curves, split_gap.txt.
# ---------------------------------------------------------------------------


def _fake_record(arm, seed, *, crossing_free, crossing_probe, moved_h, margin_free, episode, stratum="val"):
    """The slice of a record `survival_series` / `margin_series` read: one
    stratum with n windows, horizon 3."""
    n = len(episode)
    moved = np.zeros((n, 3), dtype=bool)
    moved[:, moved_h - 1:] = True
    margin = np.zeros((n, 3))
    margin[:, moved_h - 1] = margin_free
    return {
        "arm": arm, "seed": seed, "horizon": 3, "context": 2, "device": "cpu",
        "torch_version": torch.__version__,
        "episodes": {stratum: [f"ep{e}" for e in sorted(set(episode))]},
        "probe": {"selection_r2": 0.3, "measurable": True},
        "strata": {stratum: {
            "windows": {"total": n, "episode": list(episode), "clusters": len(set(episode))},
            "crossing": {"free": list(crossing_free), "probe": list(crossing_probe)},
            "margin": {"free": margin, "probe": margin},
            "moved": moved,
            "band": {"position": {"gap_final": 0.1}},
        }},
    }


def test_survival_series_is_the_indicator_over_the_windows_that_ever_moved():
    """Crossings [4, 2, NaN, 1] at h=2: alive [1, 0, -, 0]; the NaN window
    never moved and is `changed=False` -- M3d's S(h) denominator -- so the
    pooled mean over the kept three is S(2) = 1/3."""
    record = _fake_record("random_vit", 0, crossing_free=[4, 2, np.nan, 1],
                          crossing_probe=[4, 4, np.nan, 4], moved_h=1,
                          margin_free=[1, 1, 1, 1], episode=[0, 0, 1, 1])
    s = script.survival_series(record, "val", "free", h=2)
    assert isinstance(s, pooling.CellSeries)
    assert (s.arm, s.seed, s.rung, s.channel) == ("random_vit", 0, "val", "S/free")
    np.testing.assert_array_equal(s.delta, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(s.changed, [True, True, False, True])
    np.testing.assert_array_equal(s.episode, [0, 0, 1, 1])
    assert s.val == ("ep0", "ep1") and s.windows_total == 4
    pooled = pooling.pool_arm([s])
    assert pooled.mean == pytest.approx(1 / 3) and pooled.windows == 3


def test_margin_series_masks_by_moved_at_h_not_ever_moved():
    """moved from h=2 on: at h=1 every window is unmoved -> nothing kept; at
    h=2 all kept with the margin values."""
    record = _fake_record("random_vit", 0, crossing_free=[4, 2, 4, 1], crossing_probe=[4] * 4,
                          moved_h=2, margin_free=[3, -1, 2, 0], episode=[0, 0, 1, 1])
    at_one = script.margin_series(record, "val", "free", h=1)
    assert not at_one.changed.any()
    at_two = script.margin_series(record, "val", "free", h=2)
    assert at_two.changed.all() and at_two.channel == "margin/free"
    np.testing.assert_array_equal(at_two.delta, [3.0, -1.0, 2.0, 0.0])


def test_decision_horizon_is_fifteen_unless_the_run_is_shorter():
    assert script.decision_horizon(45) == (15, False)
    assert script.decision_horizon(15) == (15, False)
    assert script.decision_horizon(3) == (3, True), "clamped, and flagged so the text says so"


def test_gap_inputs_contrasts_train_held_against_val_per_arm_and_per_seed():
    """Two seeds, two strata, hand-built crossings. train_held: every window
    alive at h=2 in both seeds (S = 1); val: none alive (S = 0). The free
    contrast is +1.0 with a zero clustered SE on both sides (constant within
    every cluster) -> z = +inf, which the reading refuses to clear. The
    probe channel is identical here. Per-seed leaves carry one seed each,
    and the train_held gap_final per seed is what the record says."""
    def rec(seed, stratum, alive):
        crossing = [4, 4, 4, 4] if alive else [1, 1, 1, 1]
        r = _fake_record("random_vit", seed, crossing_free=crossing, crossing_probe=crossing,
                         moved_h=1, margin_free=[0] * 4, episode=[0, 0, 1, 1], stratum=stratum)
        r["strata"][stratum]["band"]["position"]["gap_final"] = 0.5 + seed
        return r

    records = {}
    for seed in (0, 1):
        merged = rec(seed, "val", alive=False)
        held = rec(seed, "train_held", alive=True)
        merged["strata"]["train_held"] = held["strata"]["train_held"]
        merged["episodes"]["train_held"] = held["episodes"]["train_held"]
        records[("random_vit", seed)] = merged
    inputs = script.gap_inputs(records, arms=["random_vit"], seeds=[0, 1], h=2)
    assert list(inputs.arms) == ["random_vit"] and inputs.h == 2
    a = inputs.arms["random_vit"]
    assert a.gap_free.estimate == pytest.approx(1.0) and a.gap_free.se == 0.0
    assert a.gap_free.z == float("inf") and a.gap_free.clusters == 2
    assert a.gap_probe.estimate == pytest.approx(1.0)
    assert a.train_held_gap_final == {0: 0.5, 1: 1.5}
    assert set(a.per_seed) == {0, 1}
    assert a.per_seed[1].train_held_gap_final == {1: 1.5} and a.per_seed[1].per_seed is None
    assert np.isfinite(inputs.z_fam), "cluster_threshold(6, 2) is finite"
    assert inputs.z_fam == pytest.approx(pooling.cluster_threshold(6, 2))


def test_an_unmeasurable_cell_leaves_the_probe_channel_only():
    def rec(seed, stratum):
        r = _fake_record("pixel_ae", seed, crossing_free=[4] * 4, crossing_probe=[4] * 4,
                         moved_h=1, margin_free=[0] * 4, episode=[0, 0, 1, 1], stratum=stratum)
        return r

    records = {}
    for seed in (0, 1):
        merged = rec(seed, "val")
        merged["strata"]["train_held"] = rec(seed, "train_held")["strata"]["train_held"]
        merged["episodes"]["train_held"] = ["ep0", "ep1"]
        records[("pixel_ae", seed)] = merged
    records[("pixel_ae", 0)]["probe"]["measurable"] = False
    records[("pixel_ae", 1)]["probe"]["measurable"] = False
    inputs = script.gap_inputs(records, arms=["pixel_ae"], seeds=[0, 1], h=2)
    a = inputs.arms["pixel_ae"]
    assert np.isfinite(a.gap_free.estimate), "the free channel pools every cell"
    assert np.isnan(a.gap_probe.estimate) and a.gap_probe.clusters == 0, (
        "no measurable cell: the probe contrast is NaN with no clusters, never a number")


@pytest.fixture
def gap_run(cell, capsys):
    assert script.main(_argv(cell)) == script.EXIT_OK
    out = capsys.readouterr().out
    return types.SimpleNamespace(cell=cell, stdout=out, text=(cell.out / "split_gap.txt").read_text())


def test_the_fixture_run_writes_split_gap_txt_identical_to_stdout_from_the_header_on(gap_run):
    header = "--- split gap: self-check"
    assert header in gap_run.stdout and gap_run.text.startswith(header)
    assert gap_run.stdout[gap_run.stdout.index(header):] == gap_run.text


def test_the_fixture_run_prints_every_block(gap_run):
    text = gap_run.text
    for block in (
        "--- split gap: self-check", "--- strata", "--- the band per stratum",
        "--- survival per stratum", "--- conditional survival per stratum",
        "--- Reading G", "--- G at the reported horizons",
        "--- sensitivity", "--- learning curves", "--- per seed",
    ):
        assert block in text, block
    assert "random_vit" in text
    for stratum in STRATA:
        assert stratum in text
    assert "train_probe" in text and "confounded" in text.lower()
    assert "verdict: random_vit" in text and "decided by:" in text
    assert "h=3" in text and "clamped" in text, "the fixture's horizon is 3 < 15"
    for term in TERMS:
        assert term in text
    assert "figure=" in text and (gap_run.cell.out / "learning_curves.png").stat().st_size > 0


def test_a_single_seed_run_cannot_read_memorisation_and_says_so(gap_run):
    """One seed: at most one seed can clear, below SEEDS_REQUIRED, so the
    status is one of the other four and the verdict line names the count."""
    verdict = [line for line in gap_run.text.splitlines() if "verdict: random_vit" in line]
    assert len(verdict) == 1
    assert "MEMORISATION" not in verdict[0]


def test_write_learning_curves_handles_a_missing_matplotlib_by_costing_the_figure_only(cell, monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def no_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    records = {("random_vit", 1): {"arm": "random_vit", "seed": 1,
                                   "history": {"loss": [1.0] * 8,
                                               "parts": [{t: 0.1 for t in TERMS}] * 8}}}
    line = script.write_learning_curves(records, tmp_path / "x.png", window=4)
    assert line.startswith("figure NOT written:") and not (tmp_path / "x.png").exists()
```

Also extend `test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own` in `tests/eval/test_diagnose_dynamics_script.py` by appending, after the trust block:

```python
    split_gap = statuses("split_gap")
    reused_by_split_gap = {**reused, "EXIT_SELF_CHECK_FAILED": 30}
    shared_with_trust = {
        name: value for name, value in split_gap.items() if value in set(trust.values()) - {0}
    }
    assert shared_with_trust == reused_by_split_gap, (
        f"split_gap shares {shared_with_trust} with trust_horizon; only "
        f"{reused_by_split_gap} is shared on purpose"
    )
    assert len(set(split_gap.values())) == len(split_gap), split_gap
    assert 1 not in split_gap.values() and 2 not in split_gap.values()
    assert split_gap["EXIT_STRATA_NOT_A_PARTITION"] == 31
    own = set(split_gap.values()) - set(reused_by_split_gap.values()) - {0}
    assert own == {31}
    for other in ("run_study", "report_study", "pool_dynamics", "diagnose_dynamics"):
        clash = own & set(statuses(other).values())
        assert not clash, f"split_gap collides with {other} on {clash}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_split_gap_script.py -q -k "survival_series or margin_series or decision_horizon or gap_inputs or gap_run or learning_curves or unmeasurable or single_seed" 2>&1 | tail -3`
Expected: `AttributeError: module has no attribute 'survival_series'` and friends.

- [ ] **Step 3: Implement the pooling glue and the text**

In `scripts/split_gap.py`, insert BEFORE `def _parser()`:

```python
# ---------------------------------------------------------------------------
# Pooling glue: per-cell records -> the inputs Reading G is decided on.
# ---------------------------------------------------------------------------
# Every estimator is `pooling.py`'s. What is decided here is only WHICH series
# goes in under WHICH mask (spec 3.1): the survival indicator over the windows
# that moved within the horizon (M3d's S(h) denominator, so the pooled mean IS
# S(h)); the margin over the windows moved AT h. The stratum contrast is the
# unpaired one -- two strata share no window.

R2_SENSITIVITY = 0.1
"""Cells whose probe selection R^2 is below this leave the probe channel on
the SENSITIVITY line only (spec 3.4); chosen knowing pixel_ae/s1 reads 0.018
and every other shipped cell >= 0.249. It changes no verdict."""
SURVIVAL_STEPS: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 30, 45)
"""The columns of the survival table, M3d's, filtered to <= the run's horizon."""


def _series(record: dict, stratum: str, channel: str, values, changed) -> pooling.CellSeries:
    """One per-window series of one cell's stratum, in the shape the pool
    reads. `rung` is the stratum and `channel` the reading ("S/free",
    "margin/probe"), so `require_compatible` pools seeds of one stratum only
    and `unpaired_contrast` refuses two different readings."""
    block = record["strata"][stratum]
    values = np.asarray(values, dtype=float)
    changed = np.asarray(changed, dtype=bool)
    episode = np.asarray(block["windows"]["episode"], dtype=int)
    if not (values.shape == changed.shape == episode.shape):
        raise ValueError(
            f"{record['arm']} seed {record['seed']} {stratum}/{channel}: values "
            f"{values.shape}, changed {changed.shape} and windows.episode {episode.shape} "
            "must all be (n_windows,)"
        )
    return pooling.CellSeries(
        arm=record["arm"], seed=int(record["seed"]), rung=stratum, channel=channel,
        delta=values, changed=changed, episode=episode, embedding=None, noise=None,
        windows_total=int(block["windows"]["total"]), val=tuple(record["episodes"][stratum]),
        horizon=int(record["horizon"]), context=int(record["context"]),
        device=str(record["device"]), torch_version=str(record["torch_version"]),
    )


def survival_series(record: dict, stratum: str, channel: str, h: int) -> pooling.CellSeries:
    """`1[h_x > h]` per window over the windows that moved within the horizon
    (finite crossing) -- so the pooled mean is S(h) exactly."""
    crossing = np.asarray(record["strata"][stratum]["crossing"][channel], dtype=float)
    return _series(record, stratum, f"S/{channel}", (crossing > h).astype(float), np.isfinite(crossing))


def margin_series(record: dict, stratum: str, channel: str, h: int) -> pooling.CellSeries:
    """`Delta(h)` per window over the windows moved AT h."""
    block = record["strata"][stratum]
    values = np.asarray(block["margin"][channel], dtype=float)[:, h - 1]
    moved = np.asarray(block["moved"], dtype=bool)[:, h - 1]
    return _series(record, stratum, f"margin/{channel}", values, moved & np.isfinite(values))


def _pooled(cells) -> pooling.PooledMean | None:
    """`pool_arm`, or None when there is nothing to pool -- no cell, or no
    window surviving every cell's mask -- decided before `np.mean`."""
    if not cells:
        return None
    pooling.require_compatible(cells)
    if not np.logical_and.reduce([c.changed for c in cells]).any():
        return None
    return pooling.pool_arm(cells)


_NO_CONTRAST = StratumContrast(estimate=float("nan"), se=float("nan"), z=float("nan"), clusters=0)


def _contrast(a: pooling.PooledMean | None, b: pooling.PooledMean | None) -> StratumContrast:
    if a is None or b is None:
        return _NO_CONTRAST
    c = pooling.unpaired_contrast(a, b)
    return StratumContrast(estimate=c.mean, se=c.se, z=c.z, clusters=c.clusters)


def _probe_ok(record: dict, min_r2: float | None) -> bool:
    """Spec 3.1: the probe channel pools the measurable cells; the sensitivity
    line additionally drops cells below `min_r2`."""
    if not record["probe"]["measurable"]:
        return False
    return min_r2 is None or float(record["probe"]["selection_r2"]) >= min_r2


def arm_inputs(records: dict, arm: str, seeds, *, h: int, series=survival_series,
               min_r2: float | None = None) -> ArmInputs:
    """One arm's `ArmInputs` at step `h`: the two stratum contrasts pooled over
    `seeds`, spec 4.1's `gap_final` on train_held per seed, and the same
    within each seed alone."""
    def pooled(stratum, channel, seed_list):
        cells = [
            series(records[(arm, s)], stratum, channel, h) for s in seed_list
            if channel == "free" or _probe_ok(records[(arm, s)], min_r2)
        ]
        return _pooled(cells)

    def contrast(channel, seed_list):
        return _contrast(pooled(DECISION[0], channel, seed_list), pooled(DECISION[1], channel, seed_list))

    gate = {
        int(s): float(records[(arm, s)]["strata"]["train_held"]["band"]["position"]["gap_final"])
        for s in seeds
    }
    per_seed = {
        int(s): ArmInputs(
            gap_free=contrast("free", [s]), gap_probe=contrast("probe", [s]),
            train_held_gap_final={int(s): gate[int(s)]}, per_seed=None,
        )
        for s in seeds
    }
    return ArmInputs(
        gap_free=contrast("free", list(seeds)), gap_probe=contrast("probe", list(seeds)),
        train_held_gap_final=gate, per_seed=per_seed,
    )


def _clusters(records: dict, stratum: str) -> int:
    return int(next(iter(records.values()))["strata"][stratum]["windows"]["clusters"])


def gap_inputs(records: dict, *, arms, seeds, h: int, series=survival_series,
               min_r2: float | None = None) -> GapInputs:
    """Every arm's inputs at `h`, and `z_fam` over the SMALLER of the two
    decision strata's cluster counts (spec 3.1)."""
    clusters = min(_clusters(records, DECISION[0]), _clusters(records, DECISION[1]))
    ordered = sorted(arms, key=lambda a: ARMS_ORDER.index(a) if a in ARMS_ORDER else len(ARMS_ORDER))
    return GapInputs(
        arms={arm: arm_inputs(records, arm, seeds, h=h, series=series, min_r2=min_r2) for arm in ordered},
        z_fam=pooling.cluster_threshold(FAMILY, clusters),
        h=h,
    )


def decision_horizon(horizon: int) -> tuple[int, bool]:
    """`DECISION_H`, clamped to the run's horizon; the flag says it was."""
    if DECISION_H <= horizon:
        return DECISION_H, False
    return int(horizon), True


# ---------------------------------------------------------------------------
# The printed tables.
# ---------------------------------------------------------------------------


def _num(value, spec: str = ".3f") -> str:
    value = float(value)
    return format(value, spec) if np.isfinite(value) else "n/a"


def _cells_in_order(records: dict) -> list[tuple[str, int]]:
    return sorted(records, key=lambda k: (ARMS_ORDER.index(k[0]) if k[0] in ARMS_ORDER else 9, k[1]))


def _self_check_table(records: dict) -> str:
    lines = [
        "--- split gap: self-check (val stratum against the diagnostic; exact) ---",
        f"  {'arm':<12}{'seed':>5}{'ref max|d|':>12}{'pers max|d|':>13}{'windows':>9}{'probe R2':>10}{'measurable':>12}  ok",
    ]
    for arm, seed in _cells_in_order(records):
        r = records[(arm, seed)]
        c = r["self_check"]
        lines.append(
            f"  {arm:<12}{seed:>5}{c['reference_position_max_delta']:>12.1e}"
            f"{c['persistence_position_max_delta']:>13.1e}"
            f"{str(c['windows_total_match'] and c['windows_episode_match']):>9}"
            f"{r['probe']['selection_r2']:>10.3f}{str(r['probe']['measurable']):>12}  {c['ok']}"
        )
    return "\n".join(lines) + "\n"


def _strata_table(records: dict, h: int) -> str:
    first = next(iter(records.values()))
    lines = [
        "--- strata (identical for every cell: one split, one probe rule) ---",
        f"  {'stratum':<12}{'episodes':>9}{'windows':>9}{'clusters':>9}{'never_moved':>12}"
        f"{'not_moved@1':>12}{f'not_moved@{h}':>13}  role",
    ]
    roles = {
        "val": "the record's; self-check anchor",
        "train_held": "model-seen, probe-unseen; THE DECISION STRATUM",
        "train_probe": "model-seen, probe-seen; CONFOUNDED, information only",
    }
    for stratum in STRATA:
        block = first["strata"][stratum]
        not_moved = block["counts"]["not_moved"]
        lines.append(
            f"  {stratum:<12}{len(first['episodes'][stratum]):>9}{block['windows']['total']:>9}"
            f"{block['windows']['clusters']:>9}{block['counts']['never_moved']:>12}"
            f"{int(not_moved[0]):>12}{int(not_moved[h - 1]):>13}  {roles[stratum]}"
        )
    return "\n".join(lines) + "\n"


def _band_table(records: dict, arms, seeds) -> str:
    horizon = int(next(iter(records.values()))["horizon"])
    lines = [
        f"--- the band per stratum: gap_closed({horizon}) on position (spec 4.1's metric; "
        "NaN = non-positive band) ---",
        f"  {'arm':<12}{'stratum':<12}" + "".join(f"{f's{s}':>10}" for s in seeds)
        + f"{'nanmean':>10}{'unanimous>0':>13}{'degenerate':>11}",
    ]
    for arm in arms:
        for stratum in STRATA:
            finals = [
                float(records[(arm, s)]["strata"][stratum]["band"]["position"]["gap_final"])
                for s in seeds
            ]
            degenerate = max(
                int(records[(arm, s)]["strata"][stratum]["band"]["position"]["steps_degenerate"])
                for s in seeds
            )
            unanimous = all(np.isfinite(f) and f > 0 for f in finals)
            mean = float(np.nanmean(finals)) if np.isfinite(finals).any() else float("nan")
            lines.append(
                f"  {arm:<12}{stratum:<12}" + "".join(f"{_num(f, '+.4f'):>10}" for f in finals)
                + f"{_num(mean, '+.4f'):>10}{str(unanimous):>13}{degenerate:>11}"
            )
    return "\n".join(lines) + "\n"


def survival_by_arm_and_stratum(records: dict, arms, seeds) -> dict:
    """`(arm, stratum, channel) -> S(h)` over the (window, seed) draws stacked,
    every cell in the free channel, the measurable cells in the probe one."""
    horizon = int(next(iter(records.values()))["horizon"])
    out = {}
    for arm in arms:
        for stratum in STRATA:
            for channel in CHANNELS:
                draws = [
                    np.asarray(records[(arm, s)]["strata"][stratum]["crossing"][channel], dtype=float)
                    for s in seeds if channel == "free" or _probe_ok(records[(arm, s)], None)
                ]
                stacked = np.concatenate(draws) if draws else np.zeros(0)
                out[(arm, stratum, channel)] = survival(stacked, horizon) if stacked.size else np.full(horizon + 1, np.nan)
    return out


def _survival_table(curves: dict, horizon: int) -> str:
    steps = [h for h in SURVIVAL_STEPS if h <= horizon]
    lines = [
        "--- survival per stratum: S(h) = fraction of moved draws with h_x > h "
        "(a window not yet moved at h survives vacuously, as in M3d) ---",
        f"  {'arm':<12}{'stratum':<12}{'channel':<8}" + "".join(f"{f'S({h})':>7}" for h in steps)
        + "".join(f"{'H*' + q_key(q)[1:]:>8}" for q in Q_REPORTED),
    ]
    for (arm, stratum, channel), s in curves.items():
        lines.append(
            f"  {arm:<12}{stratum:<12}{channel:<8}" + "".join(f"{_num(s[h], '.2f'):>7}" for h in steps)
            + "".join(f"{trust_horizon(s, q):>8d}" for q in Q_REPORTED)
        )
    return "\n".join(lines) + "\n"


def _conditional_table(records: dict, arms, seeds, horizon: int) -> str:
    """Beside S(h), M3d's two series (spec 2.2), computed by `trust_horizon.py`'s
    own functions imported by path: `u(h)`, the fraction of the counted draws
    whose window has not yet moved at h (and so survives vacuously), and
    `S_c(h) = (S(h) - u(h)) / (1 - u(h))`, the survival among the draws
    measured at h. Printed, never decided on."""
    steps = [h for h in SURVIVAL_STEPS if h <= horizon]
    lines = [
        "--- conditional survival per stratum: u(h) = unmoved fraction, "
        "S_c(h) = (S(h) - u(h)) / (1 - u(h)) (M3d's series; printed, not decided on) ---",
        f"  {'arm':<12}{'stratum':<12}{'channel':<8}" + "".join(f"{f'u({h})':>7}" for h in steps)
        + "".join(f"{f'Sc({h})':>8}" for h in steps),
    ]
    for arm in arms:
        for stratum in STRATA:
            for channel in CHANNELS:
                cells = [records[(arm, s)] for s in seeds
                         if channel == "free" or _probe_ok(records[(arm, s)], None)]
                if not cells:
                    continue
                crossings = np.concatenate([
                    np.asarray(c["strata"][stratum]["crossing"][channel], dtype=float) for c in cells
                ])
                first = np.concatenate([
                    np.asarray(c["strata"][stratum]["first_moved"], dtype=float) for c in cells
                ])
                s = (survival(crossings, horizon) if np.isfinite(crossings).any()
                     else np.full(horizon + 1, np.nan))
                u = _trust.unmoved_fraction(crossings, first, horizon)
                sc = _trust.conditional_survival(s, u)
                lines.append(
                    f"  {arm:<12}{stratum:<12}{channel:<8}"
                    + "".join(f"{_num(u[h], '.2f'):>7}" for h in steps)
                    + "".join(f"{_num(sc[h], '.2f'):>8}" for h in steps)
                )
    return "\n".join(lines) + "\n"


def _reported_table(records: dict, arms, seeds, horizon: int) -> str:
    steps = sorted({h for h in REPORTED_H if h <= horizon} | {horizon})
    lines = [
        "--- G at the reported horizons (train_held - val; z only; not decided on) ---",
        f"  {'arm':<12}{'channel':<8}" + "".join(f"{f'z@{h}':>9}" for h in steps)
        + f"{'dFree@dec':>11}",
    ]
    dec, _ = decision_horizon(horizon)
    for arm in arms:
        by_h = {h: arm_inputs(records, arm, seeds, h=h) for h in steps}
        margin = arm_inputs(records, arm, seeds, h=dec, series=margin_series).gap_free
        for channel in CHANNELS:
            zs = [getattr(by_h[h], f"gap_{channel}").z for h in steps]
            extra = f"{_num(margin.estimate, '+.4f')} (z {_num(margin.z, '+.2f')})" if channel == "free" else ""
            lines.append(
                f"  {arm:<12}{channel:<8}" + "".join(f"{_num(z, '+.2f'):>9}" for z in zs) + f"  {extra}"
            )
    lines.append("  dFree@dec: the stratum difference of the embedding-space margin Delta_free "
                 "at the decision horizon -- the continuous companion, in the loss's units.")
    return "\n".join(lines) + "\n"


def _sensitivity_text(records: dict, arms, seeds, h: int, z_fam: float) -> str:
    dropped = [f"{a} s{s}" for a, s in _cells_in_order(records)
               if float(records[(a, s)]["probe"]["selection_r2"]) < R2_SENSITIVITY]
    inputs = gap_inputs(records, arms=arms, seeds=seeds, h=h, min_r2=R2_SENSITIVITY)
    lines = [
        f"--- sensitivity (changes no verdict): probe channel with selection R2 < {R2_SENSITIVITY} "
        f"excluded -- dropped: {', '.join(dropped) if dropped else 'none'} ---",
    ]
    for arm, a in inputs.arms.items():
        lines.append(
            f"  {arm:<12}G_probe estimate {_num(a.gap_probe.estimate, '+.4f')} se "
            f"{_num(a.gap_probe.se, '.4f')} z {_num(a.gap_probe.z, '+.2f')} "
            f"(bar {z_fam:.2f}; clusters {a.gap_probe.clusters})"
        )
    return "\n".join(lines) + "\n"


def _per_seed_text(inputs: GapInputs) -> str:
    lines = ["--- per seed (the same contrasts within one seed alone; no seed averaging) ---"]
    for arm, a in inputs.arms.items():
        for seed, leaf in sorted((a.per_seed or {}).items()):
            leaf_reading = reading_gap(GapInputs(arms={arm: leaf}, z_fam=inputs.z_fam, h=inputs.h)).arms[arm]
            lines.append(
                f"  {arm:<12}s{seed}: G_free z {_num(leaf.gap_free.z, '+.2f')}, G_probe z "
                f"{_num(leaf.gap_probe.z, '+.2f')}, train_held gap_closed "
                f"{_num(leaf.train_held_gap_final[seed], '+.4f')} -> "
                f"{leaf_reading.status.name.replace('_', ' ')}"
            )
    return "\n".join(lines) + "\n"


def _learning_curve_table(records: dict, window: int) -> str:
    lines = [
        f"--- learning curves (from history.parts; {window}-step moving mean; descriptive, "
        "no verdict): last quarter vs preceding quarter, and the smoothed minimum's step ---",
        f"  {'arm':<12}{'seed':>5}{'term':<11}{'last_q':>11}{'prev_q':>11}{'change%':>9}"
        f"{'descending':>11}{'min_step':>9}",
    ]
    for arm, seed in _cells_in_order(records):
        summary = learning_curve_summary(records[(arm, seed)]["history"], window=window)
        for term in ("loss", *TERMS):
            t = summary["terms"][term]
            lines.append(
                f"  {arm:<12}{seed:>5}{term:<11}{_num(t['last_quarter_mean'], '.4f'):>11}"
                f"{_num(t['preceding_quarter_mean'], '.4f'):>11}{_num(t['change_pct'], '+.2f'):>9}"
                f"{str(t['descending']):>11}{t['smoothed_min_step']:>9}"
            )
    return "\n".join(lines) + "\n"


def write_learning_curves(records: dict, figure: Path, window: int = 100) -> str:
    """One panel per term plus the summed loss; arms coloured, seeds as thin
    lines, KL_FREE_BITS on the KL panels. A missing or broken matplotlib, or
    an unwritable path, costs the FIGURE and nothing else (report_study's
    guard)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, ValueError, OSError) as error:
        return f"figure NOT written: {error}"
    colours = {"pixel_ae": "tab:orange", "frozen_ssl": "tab:blue", "random_vit": "tab:gray"}
    panels = ("loss", *TERMS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    try:
        for ax, term in zip(axes.flat, panels):
            for (arm, seed), record in records.items():
                history = record["history"]
                s = (np.asarray(history["loss"], dtype=float) if term == "loss"
                     else np.asarray([float(p[term]) for p in history["parts"]], dtype=float))
                w = min(window, s.size)
                smoothed = np.convolve(s, np.ones(w) / w, mode="valid")
                ax.plot(np.arange(w, s.size + 1), smoothed, color=colours.get(arm, "black"),
                        linewidth=0.9, alpha=0.85, label=f"{arm} s{seed}")
            if term in ("kl_dyn", "kl_rep"):
                ax.axhline(KL_FREE_BITS, color="black", linestyle=":", linewidth=0.8, label="free bits")
            ax.set_title(term)
            ax.set_xlabel("training step")
            ax.set_ylabel(f"{window}-step mean")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=110)
    except (ValueError, OSError) as error:
        return f"figure NOT written: {error}"
    finally:
        plt.close(fig)
    return f"figure={figure}"


def readings_text(records: dict, *, arms, seeds, horizon: int, window: int, figure_line: str) -> str:
    """Everything below the per-cell lines, in `trust.txt`'s style; written to
    `split_gap.txt` byte-identical."""
    h, clamped = decision_horizon(horizon)
    inputs = gap_inputs(records, arms=arms, seeds=seeds, h=h)
    reading = reading_gap(inputs)
    note = (f"  NOTE: decision horizon clamped to the run's horizon h={h} "
            f"(pre-registered DECISION_H = {DECISION_H}).\n" if clamped else "")
    return "".join([
        _self_check_table(records),
        _strata_table(records, h),
        _band_table(records, arms, seeds),
        _survival_table(survival_by_arm_and_stratum(records, arms, seeds), horizon),
        _conditional_table(records, arms, seeds, horizon),
        note,
        f"  pooling: z_fam = cluster_threshold({FAMILY}, {min(_clusters(records, DECISION[0]), _clusters(records, DECISION[1]))}) = "
        f"{inputs.z_fam:.2f}; the survival indicator pools windows that moved within the horizon; "
        f"the probe channel pools measurable cells only.\n",
        format_reading_gap(reading, inputs),
        _reported_table(records, arms, seeds, horizon),
        _sensitivity_text(records, arms, seeds, h, inputs.z_fam),
        _per_seed_text(inputs),
        _learning_curve_table(records, window),
        f"  {figure_line}\n",
    ])


def write_readings(out_dir: Path, text: str) -> Path:
    path = Path(out_dir) / "split_gap.txt"
    path.write_text(text)
    return path
```

Then in `main`, replace the line `# Task 6 appends the pooling, Reading G, the learning curves and split_gap.txt here.` and the `return EXIT_OK` that follows it with:

```python
    # The learning curves come from the STUDY records' history, which the
    # split-gap record does not copy; carry them beside each live record.
    for cell in cells:
        records[(cell.arm, cell.seed)]["history"] = cell.record["history"]
    figure = args.figure if args.figure is not None else args.out / "learning_curves.png"
    figure_line = write_learning_curves(records, figure, window=args.window)
    text = readings_text(
        records, arms=list(args.arms), seeds=[int(s) for s in args.seeds],
        horizon=horizon, window=args.window, figure_line=figure_line,
    )
    print(text, end="")
    write_readings(args.out, text)
    return EXIT_OK
```

(The `history` is attached to the LIVE dict after the record was written, so the on-disk `split_gap_*.json` does not duplicate 20,000 x 6 floats already in the study record.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_split_gap_script.py tests/eval/test_diagnose_dynamics_script.py -q 2>&1 | tail -3`
Expected: all pass, `0 warnings`. The `gap_run` fixture runs `main` once per test that uses it (~8 s each).

- [ ] **Step 5: Mutation check, by hand, three ways**

Temporarily make each edit, run `.venv/bin/python -m pytest tests/eval/test_split_gap_script.py -q -x -k "survival_series or margin_series or gap_inputs or unmeasurable"`, confirm the named test fails, revert with `git checkout scripts/split_gap.py`:

| mutation | test that must fail |
|---|---|
| `survival_series`: `changed=np.isfinite(crossing)` -> `changed=np.ones_like(crossing, bool)` | `test_survival_series_is_the_indicator_...` (pooled mean 1/4, not 1/3) |
| `survival_series`: `crossing > h` -> `crossing >= h` | same test (alive [1, 1, -, 0]) |
| `margin_series`: mask `moved[:, h-1]` -> `moved.any(axis=1)` | `test_margin_series_masks_by_moved_at_h_...` |
| `arm_inputs`: swap `DECISION[0]` / `DECISION[1]` in `contrast` | `test_gap_inputs_contrasts_train_held_against_val_...` (estimate -1) |
| `_probe_ok`: drop the `measurable` check | `test_an_unmeasurable_cell_leaves_the_probe_channel_only` |
| `decision_horizon`: `<=` -> `<` | `test_decision_horizon_is_fifteen_unless_the_run_is_shorter` |

- [ ] **Step 6: Full suite and commit**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: `1490 passed` (1481 + 9), `0 warnings`. Record the number.

```bash
git add scripts/split_gap.py tests/eval/test_split_gap_script.py tests/eval/test_diagnose_dynamics_script.py
git commit -m "feat: split_gap pools the strata, decides Reading G, prints the sensitivity and the learning curves, and writes split_gap.txt

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Run on `runs/m3_study_v2`, record the reading

**Files:**
- Read: `runs/m3_study_v2/` (nine checkpoints, nine study records, nine diagnostics -- read-only)
- Write: `runs/m3_study_v2/split_gap_<arm>_seed<n>.json` x 9, `split_gap.txt`, `split_gap.log`, `split_gap.exit`, `split_gap.head`, `learning_curves.png`
- Modify: this plan, `## Task 7 results`

- [ ] **Step 1: Pre-flight**

```bash
cd /Users/raphaelchen/Desktop/csgo-bot && git status --short           # expect: clean (or only study.log untracked)
git branch --show-current                                                # expect: feat/m3e-split-gap
.venv/bin/python -m pytest -q 2>&1 | tail -2                             # expect: 1490 passed, 0 warnings
ls runs/m3_study_v2/world_model_*.pt | wc -l                             # expect: 9
ls runs/m3_study_v2/diagnostic_*.json | wc -l                            # expect: 9
ls runs/m3_study_v2/split_gap* 2>/dev/null                               # expect: nothing
.venv/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"   # expect: 2.13.0 True
```

If the worktree is not the main checkout, run from wherever `runs/` and `data/` live -- the shipped artefacts are in the main checkout at `/Users/raphaelchen/Desktop/csgo-bot`. Do NOT copy or move them.

- [ ] **Step 2: MPS smoke on one cell (the 14 device check, ~5 min)**

```bash
cd /Users/raphaelchen/Desktop/csgo-bot && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/split_gap.py --out runs/m3_study_v2 --device mps --arms random_vit --seeds 0 2>&1 | tail -30; echo "exit $?"
```

Expected: exit 0; the per-cell line reads `val 229 windows / 24 episodes; train_held ~740 windows / 78 episodes; train_probe ~190 windows / 20 episodes; self-check max|delta| reference 0.0e+00 persistence 0.0e+00`; a one-arm `split_gap.txt`. Then REMOVE ONLY the smoke's outputs so the full run's provenance is clean -- these are files this diagnostic created seconds ago, not study artefacts:

```bash
rm runs/m3_study_v2/split_gap_random_vit_seed0.json runs/m3_study_v2/split_gap.txt runs/m3_study_v2/learning_curves.png
```

If the smoke exits 14 with a non-zero reproduction: STOP. The environment is not the one the records were written on; nothing below is valid.

- [ ] **Step 3: Launch**

```bash
cd /Users/raphaelchen/Desktop/csgo-bot && git rev-parse HEAD > runs/m3_study_v2/split_gap.head && date -u +%Y-%m-%dT%H:%M:%SZ > runs/m3_study_v2/split_gap.started && find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null; PYTHONDONTWRITEBYTECODE=1 caffeinate -dimsu .venv/bin/python scripts/split_gap.py --out runs/m3_study_v2 --device mps 2>&1 | tee runs/m3_study_v2/split_gap.log; echo ${PIPESTATUS[0]} > runs/m3_study_v2/split_gap.exit; cat runs/m3_study_v2/split_gap.exit
```

Expected: ~40-50 minutes; `split_gap.exit` = `0`. **Do not commit, checkout, rebase or edit `src/` or `scripts/` while it runs.**

- [ ] **Step 4: Acceptance**

```bash
cd /Users/raphaelchen/Desktop/csgo-bot && cat runs/m3_study_v2/split_gap.exit && ls runs/m3_study_v2/split_gap_*.json | wc -l && .venv/bin/python - <<'EOF'
from pathlib import Path
from mbfps.eval.study import load_record
head = Path("runs/m3_study_v2/split_gap.head").read_text().strip()
recs = [load_record(p) for p in sorted(Path("runs/m3_study_v2").glob("split_gap_*_seed*.json"))]
assert len(recs) == 9, len(recs)
assert {r["git_sha"] for r in recs} == {head}, "one code state == HEAD"
assert {r["device"] for r in recs} == {"mps"}
assert {r["checkpoint_git_sha"] for r in recs} == {"ca3e140772d6bc741d4d04312763afe3dd754166"}
assert all(r["self_check"]["ok"] for r in recs)
assert all(r["self_check"]["reference_position_max_delta"] == 0.0 == r["self_check"]["persistence_position_max_delta"] for r in recs)
sizes = {tuple(len(r["episodes"][s]) for s in ("val", "train_held", "train_probe")) for r in recs}
assert sizes == {(24, 78, 20)}, sizes
print("OK: nine records, one git_sha == HEAD, one device == mps, self-check exactly 0.0 on 9/9, strata 24/78/20")
EOF
ls -lt runs/m3_study_v2 | head -16   # nothing newer than the run's own outputs
git status --short                    # clean under src/ and scripts/
```

- [ ] **Step 5: Fill `## Task 7 results`**

Copy numbers from `split_gap.txt`, not rounded. Use the template at the end of this plan. Then:

```bash
git add docs/superpowers/plans/2026-09-16-mb-fps-m3e-split-gap.md
git commit -m "docs: M3e split gap on runs/m3_study_v2 -- the self-check, the three strata, Reading G per arm, and the learning curves

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Exit criteria for this plan

- [ ] `pytest` fully green with zero warnings; the delta over 1437 is the sum of the tasks' stated deltas (record the measured number).
- [ ] `fit_probes`'s fit set is unchanged: exit 14 never fires on the real cells on mps (every record reproduces bitwise).
- [ ] `split_gap.py` on `runs/m3_study_v2` exits 0 with the self-check table reading `0.0e+00 / 0.0e+00 / True` on all nine rows and strata `24 / 78 / 20`.
- [ ] Nine `split_gap_<arm>_seed<n>.json` carrying every key of the contract; `split_gap.txt` carrying every block; `learning_curves.png` written.
- [ ] Reading G's status per arm is one of the five and its rule names the statistic and the bar; the decision horizon is 15, unclamped.
- [ ] `## Task 7 results` filled from `split_gap.txt`, numbers copied not rounded, closing paragraph stated against spec 4's non-claims. `NO_GAP` on every arm is a result.
- [ ] Every mutation-table row was run and caught.
- [ ] `runs/m3_study` and `runs/m3_study_v2`'s existing files untouched.

## Task 7 results (template -- replace every `…`)

**Provenance.** Nine records under one code state, `git_sha` = `…` (= `split_gap.head` = `git rev-parse HEAD`, tree clean under `src/` and `scripts/`), device `mps`, torch `…`; every record's `checkpoint_git_sha` = `ca3e140772d6bc741d4d04312763afe3dd754166`. Launched `…` (`split_gap.started`) under `caffeinate -dimsu`, `PYTHONDONTWRITEBYTECODE=1`, `__pycache__` cleared; `split_gap.exit` = `…`; wall `…`. `pytest`: `… passed`, 0 warnings, before launch.

**Self-check** (val stratum against the diagnostic): max|Δ| reference `…` / persistence `…` on 9/9; windows match 9/9; probe R² `…`–`…`; measurable 9/9.

**Strata.** val 24 episodes / 229 windows / 24 clusters; train_held 78 / `…` / 78; train_probe 20 / `…` / 20. never_moved `…` / `…` / `…`; not_moved@1 `…` / `…` / `…`; not_moved@15 `…` / `…` / `…`.

**The band per stratum, gap_closed(45) position** (spec 4.1's metric):

| arm | stratum | s0 | s1 | s2 | nanmean | unanimous > 0 |
|---|---|---|---|---|---|---|
| `pixel_ae` | val | … | … | … | … | … |
| `pixel_ae` | train_held | … | … | … | … | … |
| `pixel_ae` | train_probe (confounded) | … | … | … | … | … |
| `frozen_ssl` | val / train_held / train_probe | … |
| `random_vit` | val / train_held / train_probe | … |

**Survival per stratum** (S(h) at 1 / 5 / 15 / 45; H*_0.75), free channel:

| arm | val | train_held | train_probe |
|---|---|---|---|
| `pixel_ae` | … | … | … |
| `frozen_ssl` | … | … | … |
| `random_vit` | … | … | … |

(Probe channel: `…`.)

**Reading G at h = 15** (train_held − val; `z_fam` = `cluster_threshold(6, 24)` = `…`):

| arm | G_free estimate ± se (z) | G_probe estimate ± se (z) | train_held §4.1 | seeds clearing | **status** | decided by |
|---|---|---|---|---|---|---|
| `pixel_ae` | … | … | … | …/3 | **…** | … |
| `frozen_ssl` | … | … | … | …/3 | **…** | … |
| `random_vit` | … | … | … | …/3 | **…** | … |

G at h = 5 / 15 / 45, z, free / probe: `pixel_ae` … ; `frozen_ssl` … ; `random_vit` … . ΔFree(15) stratum difference: … .

**Sensitivity** (probe channel, R² < 0.1 excluded — dropped `…`): G_probe z per arm `…`; which changed: `…`; no verdict changed: `…`.

**Per seed:** `…`.

**Learning curves** (100-step mean; last quarter vs preceding; smoothed-min step), `embedding` and `kl_dyn` per cell:

| cell | embedding last_q / prev_q / change% / descending / min_step | kl_dyn last_q / prev_q / change% | loss change% |
|---|---|---|---|
| … | … | … | … |

Figure: `runs/m3_study_v2/learning_curves.png`.

**What this run establishes, and what it does not.** `…` — stated against spec 4: it does not change the M3 gate or any recorded verdict; it does not rank arms (every contrast is within-arm); a MEMORISATION reading says the fix is data, not how much or which policy or scenario; `train_probe` is confounded and decided nothing; probe-based numbers inherit the probe's R² and the decision channel is probe-free; everything is `my_way_home`, one split, the M3c checkpoints.
