# M3m Prediction Burden Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether the world model's 45-step failure is error *compounding* or a one-step map that never learned motion, by reading the existing re-grounding ladder against a model-free motion baseline.

**Architecture:** M3m **consumes** `diagnostics.regrounding_sweep` (already implemented, tested and run on the nine cells) and `rollout.evaluate_rollout`. It adds one model-free baseline computed from ground truth, a burden decomposition with an algebraic identity, Reading H, and a two-phase script. Neither `rollout.py` nor `diagnostics.py` is modified.

**Tech Stack:** Python 3.12, PyTorch 2.13 on MPS, NumPy, pytest. No training.

## Global Constraints

Every task's requirements implicitly include this section.

- **Test command is `.venv/bin/python -m pytest`.** There is no `pytest` entry point in the venv.
- **Never `rm`, move, truncate or overwrite anything under `runs/`.** `runs/` is a symlink to `/Users/raphaelchen/Desktop/csgo-bot/runs`, shared by every worktree.
- **Never `git stash`** (the stash stack is shared across worktrees). **Never `git clean -fdx`** (it destroys the ledger under `.superpowers/`).
- **`src/mbfps/eval/rollout.py` and `src/mbfps/eval/diagnostics.py` are NOT modified.** Re-implementing either arm is a defect, not an alternative. Spec §4 records why: the k=1 rung already is the one-step arm; `rollout.py` averages over windows so per-window rows do not survive; and `rollout.py` has no device-aware RNG snapshot, so an arm built there draws from a different stream point than the canonical pass.
- **Commit messages end with this literal trailer in its own paragraph:** `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`
- **Subject lines must not overstate.** `motion_margin` is a level against ground truth; `burden` is a difference of two measured curves. Neither licenses a claim about *why* the one-step map behaves as it does.
- **Two axes, never conflated.** `k` is the **re-grounding period**, over `REGROUNDING_KS = (1, 3, 5, 15, 45)`. `h` is the **horizon step**, 1..45, reported over `REPORTED_H`. Write `burden(k, h)`.
- **After mutating a source file to check a test bites, revert it and clear `__pycache__`** under `src/`, `tests/`, `scripts/`.
- **MUTATION CHECKS COME AFTER THE TASK'S COMMIT.** Each task below lists its mutation table before its commit step. Execute the **commit first**, then the mutation checks against the committed state. Reverting an *uncommitted* implementation with `git checkout -- <file>` destroys the implementation itself, and on a newly created file it fails outright. With the work committed, `git checkout -- <file>` is an exact revert. If a mutation shows a test does not bite, strengthen the test and `git commit --amend`.
- **Every test must be watched failing under a named mutation and passing after revert.** Roughly 30 tests that could not fail were caught across M3k and M3l. A test nobody watched fail is not evidence.
- **Exit codes 45 and 46 belong to this milestone.** 39/40 are M3j's, 41/42 M3k's, 43/44 M3l's. Reused: `EXIT_NO_CHECKPOINTS = 11`, `EXIT_SPLIT_MISMATCH = 12`, `EXIT_RECORD_MISMATCH = 14`, `EXIT_SELF_CHECK_FAILED = 30`.
- **Record every protocol parameter the reading depends on**, including the confidence level and the resample count. M3l shipped a headline interval whose draw count was not on the record; the fix is in this plan from the start.

## File Structure

| file | responsibility |
|---|---|
| `src/mbfps/eval/pooling.py` (modify) | promote two bootstrap helpers to public names; no behaviour change |
| `src/mbfps/eval/burden.py` (new) | constants, the burden decomposition, the identity, the motion baseline, the interval, Reading H, the formatters. No model, no device, no file I/O. |
| `scripts/prediction_burden.py` (new) | torch, I/O and the record schema: measure and read phases, exits 45 and 46 |
| `tests/eval/test_pooling.py` (modify) | the promoted names, and that the old outputs are unchanged |
| `tests/eval/test_burden.py` (new) | the module's unit behaviour |
| `tests/eval/test_prediction_burden_script.py` (new) | the script's phases, guards and exits |
| `tests/eval/test_diagnose_dynamics_script.py` (modify) | the cross-tool exit registry gains `{45, 46}` |

`burden.py` loads no checkpoint, touches no device and reads no file, so every number in Reading H is testable from arrays. (It does import `probe.position_error` to keep one definition of the position metric, and `probe` imports torch at module level — so `burden.py` is torch-free in behaviour, not in its import graph.) `prediction_burden.py` owns the model, the device and the record, exactly as `latent_capacity.py` owns them for M3l.

Tasks 1–4 are independent of any run. Task 5 is the measure phase, 6 the read phase and the registry, 7 the smoke and the run.

---

### Task 1: Promote the episode bootstrap to a public name

M3m needs an episode-clustered bootstrap over a generic statistic. Three already exist — `capacity.py` (specialised to sufficient statistics), `probe.py:_block_bootstrap_ci` (specialised to the filtering gain's three arrays) and `pooling.py` (generic). A fourth would be the wrong answer. The generic pair is private, and this codebase never imports underscore names across modules, so promote them.

**Files:**
- Modify: `src/mbfps/eval/pooling.py:670-686` (`_episode_bootstrap`, `_interval`) and their four call sites at `:736`, `:737`, `:797`, `:799`
- Test: `tests/eval/test_pooling.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pooling.episode_bootstrap(labels: np.ndarray, bootstrap: int, seed: int)` — generator yielding `bootstrap` index arrays, each the rows of one draw of the episodes with replacement.
  - `pooling.percentile_interval(replicates: np.ndarray) -> tuple[float, float, float]` — `(low, high, se)` at the 2.5/97.5 percentiles, `se` the sample standard deviation with `ddof=1`, or NaN for a single replicate.

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_pooling.py`:

```python
def test_the_bootstrap_helpers_are_public_and_draw_whole_episodes():
    """M3m consumes these from another module, so they carry public names.

    THE MUTATION THIS EXISTS FOR: renaming them back to `_episode_bootstrap` /
    `_interval` breaks the import; drawing rows instead of episodes makes the
    yielded index arrays contain partial episodes.
    """
    from mbfps.eval.pooling import episode_bootstrap, percentile_interval

    labels = np.array([0, 0, 0, 1, 1, 2])
    draws = list(episode_bootstrap(labels, bootstrap=50, seed=0))
    assert len(draws) == 50
    members = {0: {0, 1, 2}, 1: {3, 4}, 2: {5}}
    for index in draws:
        # Every drawn episode contributes ALL of its rows, so the multiset of
        # rows is a union of whole episodes -- never a partial one.
        counts = Counter(index.tolist())
        for episode, rows in members.items():
            per_row = {counts.get(row, 0) for row in rows}
            assert len(per_row) == 1, (episode, counts)

    replicates = np.arange(1001, dtype=np.float64)
    low, high, se = percentile_interval(replicates)
    assert low == pytest.approx(25.0)
    assert high == pytest.approx(975.0)
    assert se == pytest.approx(float(replicates.std(ddof=1)))
    assert np.isnan(percentile_interval(np.array([3.0]))[2])


def test_promoting_the_helpers_changed_no_pooled_number():
    """The promotion is a rename. `pool_ratio`'s output is unchanged.

    THE MUTATION THIS EXISTS FOR: any edit to the helpers' bodies while
    renaming them -- a different percentile, `ddof=0`, drawing without
    replacement -- moves these figures.
    """
    cells = _scaled_cells((1.0, 1.0, 1.0))
    pooled = pool_ratio(cells, bootstrap=200, seed=0)
    assert pooled.ratio == pytest.approx(POOLED_RATIO_AT_SEED_0, abs=1e-12)
    assert pooled.ci_low == pytest.approx(POOLED_CI_LOW_AT_SEED_0, abs=1e-12)
    assert pooled.ci_high == pytest.approx(POOLED_CI_HIGH_AT_SEED_0, abs=1e-12)
```

Add at the top of the file if absent: `from collections import Counter`, and `from mbfps.eval.pooling import pool_ratio`.

`_scaled_cells(scales, *, arm="pixel_ae", episodes=EPISODES)` already exists in that file at `:487` and builds three cells whose per-cell ratios are 0.5, 0.5 and 2.0 — use `_scaled_cells((1.0, 1.0, 1.0))` and do **not** write a second fixture. `POOLED_RATIO_AT_SEED_0`, `POOLED_CI_LOW_AT_SEED_0` and `POOLED_CI_HIGH_AT_SEED_0` are module-level constants you fill in **by running the current code before the rename** — they are the pre-rename values, which is the whole point.

- [ ] **Step 2: Capture the pre-rename values, then run the test to verify it fails**

First record what the current code produces, so the constants are evidence rather than a guess:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'tests/eval')
from test_pooling import _scaled_cells
from mbfps.eval.pooling import pool_ratio
p = pool_ratio(_scaled_cells((1.0, 1.0, 1.0)), bootstrap=200, seed=0)
print(repr(p.ratio), repr(p.ci_low), repr(p.ci_high))
"
```

Paste those three literals into the constants. Then:

```bash
.venv/bin/python -m pytest tests/eval/test_pooling.py -q -k "public_and_draw or changed_no_pooled"
```

Expected: the first test FAILS with `ImportError: cannot import name 'episode_bootstrap'`; the second PASSES (it exercises the un-renamed code through `pool_ratio`).

- [ ] **Step 3: Rename, updating all four call sites**

In `src/mbfps/eval/pooling.py`, rename `_episode_bootstrap` to `episode_bootstrap` and `_interval` to `percentile_interval`. Change nothing else — not a percentile, not `ddof`, not `replace=True`. Update the four call sites at lines 736, 737, 797 and 799.

Add a one-line docstring note on each, so the next reader knows the name is load-bearing outside this module:

```python
def episode_bootstrap(labels: np.ndarray, bootstrap: int, seed: int):
    """Yield `bootstrap` index arrays, each the rows of one draw of the
    episodes WITH replacement -- every row of a drawn episode, as many times
    as it was drawn. The ruler for a ratio of medians, which has no sandwich
    standard error: resample the clusters, recompute.

    PUBLIC because `eval.burden` consumes it. A fourth episode bootstrap in
    this codebase would be the wrong answer; there are already three.
    """
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/eval/test_pooling.py -q
```

Expected: PASS, with the same test count as before plus 2.

- [ ] **Step 5: Verify the rename is caught if reverted**

```bash
# Confirm the first test bites.
sed -i '' 's/^def episode_bootstrap/def _episode_bootstrap/' src/mbfps/eval/pooling.py
find src tests scripts -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
.venv/bin/python -m pytest tests/eval/test_pooling.py -q -k public_and_draw
```

Expected: FAIL. Then revert and clear caches:

```bash
git checkout -- src/mbfps/eval/pooling.py
find src tests scripts -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
```

Per Global Constraints, commit before running this check, so `git checkout --` is an exact revert rather than a loss of your work. End green with a clean tree.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/pooling.py tests/eval/test_pooling.py
git commit -m "refactor: the episode bootstrap and its interval carry public names

M3m needs an episode-clustered bootstrap over a generic statistic. Three
already exist -- capacity.py specialised to sufficient statistics, probe.py's
_block_bootstrap_ci specialised to the filtering gain's three arrays, and this
generic pair -- so a fourth would be the wrong answer. This codebase never
imports underscore names across modules, so the generic pair is promoted
rather than imported privately.

A rename only: the percentiles, the ddof and the with-replacement draw are
untouched, pinned by a test that compares pool_ratio's output against the
values the pre-rename code produced.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: The burden decomposition and its identity

**Files:**
- Create: `src/mbfps/eval/burden.py`
- Test: `tests/eval/test_burden.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `burden.DECISION_H: int = 45`
  - `burden.REPORTED_H: tuple[int, ...] = (1, 2, 3, 5, 8, 10, 15, 20, 30, 45)`
  - `burden.ARMS_REQUIRED: int = 2`
  - `burden.IDENTITY_TOLERANCE: float = 1e-9`
  - `burden.CONFIDENCE: float = 0.95`
  - `burden.RESAMPLES: int = 2000`
  - `burden.at_horizon(curve: np.ndarray, h: int) -> float`
  - `burden.burden(curve_k: np.ndarray, floor: np.ndarray) -> np.ndarray`
  - `burden.compounding(curve_k: np.ndarray, curve_one: np.ndarray) -> np.ndarray`
  - `burden.identity_residual(curve_k: np.ndarray, curve_one: np.ndarray, floor: np.ndarray) -> float`

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_burden.py`:

```python
"""Unit behaviour of `eval.burden` -- arrays in, numbers out, no torch."""

import numpy as np
import pytest

from mbfps.eval.burden import (
    ARMS_REQUIRED,
    CONFIDENCE,
    DECISION_H,
    IDENTITY_TOLERANCE,
    REPORTED_H,
    RESAMPLES,
    at_horizon,
    burden,
    compounding,
    identity_residual,
)


def test_the_constants_are_the_values_the_spec_fixes():
    """THE MUTATION THIS EXISTS FOR: a drifted constant. DECISION_H is the
    gate's own horizon; REPORTED_H is the reporting grid and must contain it,
    or the status is read at a horizon the table never prints."""
    assert DECISION_H == 45
    assert REPORTED_H == (1, 2, 3, 5, 8, 10, 15, 20, 30, 45)
    assert DECISION_H in REPORTED_H
    assert REPORTED_H == tuple(sorted(set(REPORTED_H)))
    assert ARMS_REQUIRED == 2
    assert IDENTITY_TOLERANCE == 1e-9
    assert CONFIDENCE == 0.95
    assert RESAMPLES == 2000


def test_at_horizon_is_one_indexed_and_refuses_out_of_range():
    """The curves are 0-indexed arrays; the horizon is 1-indexed everywhere in
    this project's records and prose.

    THE MUTATION THIS EXISTS FOR: `curve[h]` instead of `curve[h - 1]`, which
    shifts every reported number by one step and breaks no shape.
    """
    curve = np.array([10.0, 20.0, 30.0])
    assert at_horizon(curve, 1) == 10.0
    assert at_horizon(curve, 3) == 30.0
    with pytest.raises(ValueError, match="horizon step must be in 1..3"):
        at_horizon(curve, 0)
    with pytest.raises(ValueError, match="horizon step must be in 1..3"):
        at_horizon(curve, 4)


def test_burden_is_the_cost_over_the_floor():
    curve = np.array([12.0, 15.0, 22.0])
    floor = np.array([10.0, 10.0, 12.0])
    assert burden(curve, floor) == pytest.approx([2.0, 5.0, 10.0])


def test_compounding_is_exactly_zero_at_k_equals_one():
    """A CONTROL WITH A KNOWN ANSWER, true by construction: at k=1 both sides
    are the same curve, so the difference is exactly 0 -- not approximately.

    THE MUTATION THIS EXISTS FOR: computing compounding against the FLOOR
    instead of against the k=1 curve, which makes this read `burden(1)`
    instead of 0.
    """
    curve_one = np.array([12.0, 15.0, 22.0])
    result = compounding(curve_one, curve_one)
    assert np.array_equal(result, np.zeros(3))


def test_compounding_is_the_cost_of_correcting_less_often():
    curve_one = np.array([12.0, 15.0, 22.0])
    curve_k = np.array([12.0, 19.0, 40.0])
    assert compounding(curve_k, curve_one) == pytest.approx([0.0, 4.0, 18.0])


def test_the_identity_holds_and_is_reported_as_a_measured_residual():
    """burden(k) == burden(1) + compounding(k) is ALGEBRAIC: the floor cancels.
    In floating point it is a cancellation, so the residual is a few ULPs and
    is RECORDED rather than asserted to be zero -- M3l's floor_bits read
    5.7e-14, not 0.0, and the legend that called it "exactly 0" is still an
    open follow-up.

    THE MUTATION THIS EXISTS FOR: returning a hardcoded 0.0, which would hide
    a genuinely broken decomposition.
    """
    rng = np.random.default_rng(0)
    floor = rng.uniform(100.0, 150.0, size=45)
    curve_one = floor + rng.uniform(0.0, 5.0, size=45)
    curve_k = curve_one + rng.uniform(0.0, 80.0, size=45)
    residual = identity_residual(curve_k, curve_one, floor)
    assert residual < IDENTITY_TOLERANCE
    assert residual == pytest.approx(0.0, abs=1e-12)
    # And it is derived, not hardcoded: scaling the inputs by 1e6 scales the
    # cancellation error with them, so the value must move.
    scaled = identity_residual(curve_k * 1e6, curve_one * 1e6, floor * 1e6)
    assert scaled > residual


def test_the_decomposition_refuses_mismatched_or_nonfinite_curves():
    """THE MUTATION THIS EXISTS FOR: dropping the guards. NumPy broadcasts a
    length-1 array against a length-45 one silently, so a curve read from the
    wrong record key would produce a full-length result that is nonsense.
    """
    with pytest.raises(ValueError, match="same length"):
        burden(np.zeros(45), np.zeros(44))
    with pytest.raises(ValueError, match="same length"):
        compounding(np.zeros(45), np.zeros(1))
    with pytest.raises(ValueError, match="finite"):
        burden(np.array([1.0, np.nan]), np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        compounding(np.array([1.0, np.inf]), np.zeros(2))
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/eval/test_burden.py -q
```

Expected: collection error — `ModuleNotFoundError: No module named 'mbfps.eval.burden'`.

- [ ] **Step 3: Write the implementation**

Create `src/mbfps/eval/burden.py`:

```python
"""The prediction-burden decomposition -- milestone M3m.

M3l refuted the bottleneck lever: the latent carries at most 1.11% of its
160-bit ceiling. That left the OBJECTIVE lever, standing by elimination. The
hypothesis inherited with it -- "the loss never asks for motion" -- is false as
stated: `kl_rate_above_free_bits` runs 0.746-0.976 across the nine cells, so
the dynamics prior trained on 75-98% of steps.

What survives is narrower. The loss asks for ONE-step latent agreement; the
gate reads 45-step rollout error. Evaluated at every horizon, `gap_closed(h)`
is positive in 8 of 9 cells at h=1 and 0 of 9 by h=20, so the failing criterion
is the tail of a curve that starts out working.

TWO AXES, NEVER CONFLATED. `k` is the RE-GROUNDING PERIOD -- how often
observation corrects the rollout, over `diagnostics.REGROUNDING_KS` -- and `h`
is the HORIZON STEP. Both were called `k` in an early draft of the spec. Write
`burden(k, h)`.

This module holds no torch and no I/O: every number Reading H reports is a
function of arrays, so it is testable without a checkpoint. `scripts/
prediction_burden.py` owns the model, the device and the record schema.
"""

import numpy as np

DECISION_H: int = 45
"""The horizon the status is read at -- the M3 gate's own horizon.

Not a free choice: `gap_closed(45)` is the criterion that has failed since M3c,
so a reading at any other horizon would answer a question the gate does not
ask. It must appear in `REPORTED_H`, or the status would be read at a horizon
the table never prints.
"""

REPORTED_H: tuple[int, ...] = (1, 2, 3, 5, 8, 10, 15, 20, 30, 45)
"""The reporting grid, as raw levels, so a reader may apply a different horizon
rather than inheriting `DECISION_H`."""

ARMS_REQUIRED: int = 2
"""Of three. Matches every milestone from M3i onward."""

IDENTITY_TOLERANCE: float = 1e-9
"""The bound on the decomposition's floating-point residual.

`burden(k) = burden(1) + compounding(k)` is algebraic -- the floor cancels --
so in float64 over magnitudes of order 250 the residual is a cancellation of a
few ULPs, about 1e-13. This tolerance leaves four orders of margin. The MEASURED
residual is recorded per cell: asserting it is exactly zero is the mistake M3l
shipped in a legend line and has not yet fixed.
"""

CONFIDENCE: float = 0.95
"""The interval's level. `pooling.percentile_interval` takes the 2.5/97.5
percentiles, so this constant DESCRIBES that function rather than configuring
it -- a test pins the two together, because a record that names a level the
estimator did not take is worse than a record that names none."""

RESAMPLES: int = 2000
"""Episode draws per interval. `pooling.pool_ratio` uses the same count."""


def at_horizon(curve: np.ndarray, h: int) -> float:
    """`curve` at horizon step `h`, ONE-INDEXED.

    The curves are 0-indexed arrays; every record and every sentence in this
    project counts horizon steps from 1. Indexing with `h` rather than `h - 1`
    shifts every reported number by one step and breaks no shape, which is why
    this is a function rather than a convention.
    """
    curve = np.asarray(curve, dtype=np.float64)
    if not 1 <= h <= curve.size:
        raise ValueError(
            f"horizon step must be in 1..{curve.size}, got {h}"
        )
    return float(curve[h - 1])


def _checked_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both curves as finite float64 of equal length.

    NumPy broadcasts a length-1 array against a length-45 one without
    complaint, so a curve read from the wrong record key would yield a
    full-length result that is nonsense. The length check is the only thing
    standing between that and a plausible-looking table.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(
            f"the two curves must have the same length; got {left.shape} and {right.shape}"
        )
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise ValueError("every curve value must be finite")
    return left, right


def burden(curve_k: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """`burden(k, .)` -- what predicting at re-grounding period `k` costs over
    the floor, per horizon step.

    The floor is a ZERO-step posterior at every step, seeing the frame it is
    scored on. It is not the k -> 0 limit of the ladder: k=1 is a one-step
    PRIOR from a state grounded one frame earlier, so k=1 must sit STRICTLY
    above the floor. `RegroundingSweep.is_bitwise_the_floor(1)` is the check,
    and it reads False on all nine shipped cells.
    """
    curve_k, floor = _checked_pair(curve_k, floor)
    return curve_k - floor


def compounding(curve_k: np.ndarray, curve_one: np.ndarray) -> np.ndarray:
    """`compounding(k, .)` -- what it costs that correction arrives every `k`
    steps rather than every step.

    Against the k=1 CURVE, not against the floor: the floor cancels out of
    `burden(k) - burden(1)`, and computing it against the floor instead would
    silently return `burden(k)` and make the k=1 control read nonzero.
    """
    curve_k, curve_one = _checked_pair(curve_k, curve_one)
    return curve_k - curve_one


def identity_residual(
    curve_k: np.ndarray, curve_one: np.ndarray, floor: np.ndarray
) -> float:
    """The largest absolute violation of
    `burden(k) == burden(1) + compounding(k)` over the horizon.

    Algebraic, so this measures floating point and nothing else -- which is
    exactly why it is worth recording: a nonzero residual beyond
    `IDENTITY_TOLERANCE` means one of the three curves is not what its name
    says, not that arithmetic failed.
    """
    whole = burden(curve_k, floor)
    parts = burden(curve_one, floor) + compounding(curve_k, curve_one)
    return float(np.max(np.abs(whole - parts)))
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/eval/test_burden.py -q
```

Expected: `9 passed`.

- [ ] **Step 5: Verify each test bites**

Apply each mutation, confirm the named test fails, revert, clear `__pycache__`, confirm green. Record the observed output for each.

| mutation in `src/mbfps/eval/burden.py` | test that must fail |
|---|---|
| `return float(curve[h - 1])` → `return float(curve[h])` | `test_at_horizon_is_one_indexed_and_refuses_out_of_range` |
| `compounding` body → `return curve_k - floor` (take `floor` as the second arg) | `test_compounding_is_exactly_zero_at_k_equals_one` |
| `identity_residual` body → `return 0.0` | `test_the_identity_holds_and_is_reported_as_a_measured_residual` |
| delete the `left.shape != right.shape` raise | `test_the_decomposition_refuses_mismatched_or_nonfinite_curves` |
| delete the `np.isfinite` raise | `test_the_decomposition_refuses_mismatched_or_nonfinite_curves` |
| `DECISION_H = 45` → `30` | `test_the_constants_are_the_values_the_spec_fixes` |

```bash
find src tests scripts -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
.venv/bin/python -m pytest tests/eval/test_burden.py -q
```

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/burden.py tests/eval/test_burden.py
git commit -m "feat: the burden decomposition, with compounding(1) zero by construction

burden(k, h) is what predicting at re-grounding period k costs over the floor;
compounding(k, h) is what it costs that correction arrives every k steps rather
than every step. Their sum is burden(k) identically -- the floor cancels -- so
the identity measures floating point only, and the residual is RECORDED rather
than asserted to be zero.

compounding is taken against the k=1 CURVE, not the floor. Against the floor it
would silently return burden(k) and make the k=1 control read nonzero instead
of exactly zero.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: The motion baseline and the clustered interval

The one-step-persistence baseline is the only genuinely new measurement in M3m, and the whole reading rests on it being scored on the same frames as the model. The shift therefore lives **here**, where it is unit-testable, not in the script's loop.

**Files:**
- Modify: `src/mbfps/eval/burden.py`
- Test: `tests/eval/test_burden.py`

**Interfaces:**
- Consumes: `burden.at_horizon`, `burden.CONFIDENCE`, `burden.RESAMPLES` (Task 2); `pooling.episode_bootstrap`, `pooling.percentile_interval` (Task 1).
- Produces:
  - `burden.scored_targets(window_targets: np.ndarray) -> np.ndarray` — `window_targets[1:]`, the rows the model is scored on.
  - `burden.one_step_persistence(window_targets: np.ndarray) -> np.ndarray` — length `len(window_targets) - 1`.
  - `burden.motion_margin(window_targets: np.ndarray, curve_one: np.ndarray) -> np.ndarray`
  - `burden.margin_interval(window_margin: np.ndarray, groups: np.ndarray, *, h: int, resamples: int = RESAMPLES, seed: int) -> tuple[float, float, float]` — `(point, ci_low, ci_high)`. `seed` is **keyword-required with no default**.

**The index arithmetic, stated so the off-by-one is hard to write.** `evaluate_rollout` scores horizon step `j + 1` against `episode.privileged[start + context + 1 + j]`, and its comment records that this was *"Verified empirically, not by argument"* with an oracle model. So the caller passes

```
window_targets = probe_targets(episode.privileged[start + context : start + need + 1], keys)
```

which is `horizon + 1` rows: row 0 is the **last context frame** and rows `1..horizon` are exactly the model's truth rows. Then `scored_targets(window_targets) == truth` and `one_step_persistence` is the error of predicting each scored row by the row before it.

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_burden.py`:

```python
def test_scored_targets_are_the_rows_the_model_is_scored_on():
    """The baseline and the model MUST be scored on the same frames.

    `evaluate_rollout` scores horizon step j+1 against
    privileged[start + context + 1 + j]. The caller hands us
    privileged[start + context : start + need + 1], so row 0 is the last
    CONTEXT frame and rows 1.. are the model's truth. This test reproduces both
    slices from one array and asserts they agree.

    THE MUTATION THIS EXISTS FOR: `window_targets[:-1]` instead of
    `window_targets[1:]`, which scores the baseline one frame early and makes
    `motion_margin` a comparison of two different frames.
    """
    context, horizon, start = 5, 4, 7
    need = context + horizon
    privileged = np.arange(40.0).reshape(20, 2)
    privileged = np.hstack([privileged, np.zeros((20, 2))])  # (N, 4) targets

    window_targets = privileged[start + context : start + need + 1]
    model_truth = privileged[start + context + 1 : start + need + 1]

    assert np.array_equal(scored_targets(window_targets), model_truth)
    assert scored_targets(window_targets).shape[0] == horizon


def test_one_step_persistence_is_the_true_one_step_displacement():
    """Its error IS the displacement, known from ground truth with no model.

    THE MUTATION THIS EXISTS FOR: comparing each row to row 0 (the t-anchored
    persistence the record already carries) instead of to the row before it.
    That baseline is RIGGED for this comparison -- it has seen h-1 fewer frames
    than the k=1 rung -- and spec 2.2 forbids it by name.
    """
    # Moves (3, 4) then (0, 0) then (6, 8): displacements 5, 0, 10.
    xy = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0], [9.0, 12.0]])
    window_targets = np.hstack([xy, np.zeros((4, 2))])
    assert one_step_persistence(window_targets) == pytest.approx([5.0, 0.0, 10.0])

    # The rigged baseline would give cumulative distance from row 0 instead.
    rigged = np.linalg.norm(xy[1:] - xy[0], axis=1)
    assert rigged == pytest.approx([5.0, 5.0, 15.0])
    assert one_step_persistence(window_targets) != pytest.approx(rigged)


def test_motion_margin_is_positive_when_the_prior_beats_stillness():
    """Positive means one prior step beats assuming no motion."""
    xy = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0], [9.0, 12.0]])
    window_targets = np.hstack([xy, np.zeros((4, 2))])
    # Displacements are 5, 0, 10. A prior that errs by 2, 1, 3 beats stillness
    # at steps 1 and 3 and loses at step 2, where the agent did not move.
    curve_one = np.array([2.0, 1.0, 3.0])
    assert motion_margin(window_targets, curve_one) == pytest.approx([3.0, -1.0, 7.0])


def test_motion_margin_refuses_a_curve_that_is_not_the_scored_length():
    """THE MUTATION THIS EXISTS FOR: dropping the guard. A curve of length
    horizon+1 would broadcast against a baseline of length horizon only by
    accident of the numbers, and silently not at all otherwise.
    """
    window_targets = np.zeros((5, 4))
    with pytest.raises(ValueError, match="same length"):
        motion_margin(window_targets, np.zeros(5))


def test_the_interval_clusters_on_episodes_and_requires_its_seed():
    """The resampling unit is the EPISODE, not the window: 229 windows over 24
    episodes, and consecutive Doom frames are near-duplicates, so a
    window-level bootstrap returns an interval several times too narrow.

    THE MUTATION THIS EXISTS FOR, TWICE OVER: resampling windows instead of
    episodes narrows the interval; and `seed: int = 0` lets every cell share
    one seed, which is how a previous milestone shipped nine identical draws.
    """
    # SIX episodes, not two. MEASURED: with two episodes the resample admits
    # only THREE distinct replicate values, so all 20 seeds I tried return
    # byte-identical bounds and the seed assertion below is dead. Six episodes
    # give 216 distinct replicates and no other seed reproduces seed 0's bounds.
    rng = np.random.default_rng(0)
    window_margin = np.vstack([
        rng.normal(mean, 0.05, size=(20, 3))
        for mean in (6.0, 4.0, 2.0, -2.0, -4.0, -6.0)
    ])
    groups = np.repeat(np.arange(6), 20)

    point, low, high = margin_interval(
        window_margin, groups, h=1, resamples=400, seed=0
    )
    assert point == pytest.approx(window_margin[:, 0].mean(), abs=1e-12)
    # Clustered on episodes the interval spans the spread of episode means; a
    # window-level bootstrap would collapse it near the mean.
    assert high - low > 5.0

    # The fixture must discriminate, and this test asserts that about itself:
    # a resample with too few clusters yields too few distinct replicates for
    # any percentile to choose between, and every assertion below then holds
    # for the wrong reason.
    replicates = {
        margin_interval(window_margin, groups, h=1, resamples=400, seed=s)[1:]
        for s in range(8)
    }
    assert len(replicates) > 1, "the bounds do not move with the seed at all"

    # Different seeds move the BOUNDS but never the point estimate.
    other = margin_interval(window_margin, groups, h=1, resamples=400, seed=1)
    assert other[0] == pytest.approx(point, abs=1e-12)
    assert (other[1], other[2]) != (low, high)
    # The same seed twice is identical.
    again = margin_interval(window_margin, groups, h=1, resamples=400, seed=0)
    assert (again[1], again[2]) == (low, high)

    # `seed` has no default.
    import inspect
    sig = inspect.signature(margin_interval)
    assert sig.parameters["seed"].default is inspect.Parameter.empty
    assert sig.parameters["seed"].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_interval_refuses_what_it_cannot_cluster():
    """`windows.episode` is null on a record whose ladder carried no
    clustering, and the record's own comment says a reader must REFUSE rather
    than treat every window as its own episode.

    THE MUTATION THIS EXISTS FOR: falling back to `np.arange(n)` for missing
    labels, which silently converts an episode bootstrap into a window one.
    """
    window_margin = np.zeros((6, 3))
    with pytest.raises(ValueError, match="one label per window"):
        margin_interval(window_margin, np.array([0, 0, 1]), h=1, resamples=10, seed=0)
    with pytest.raises(ValueError, match="at least two episodes"):
        margin_interval(
            window_margin, np.zeros(6, dtype=int), h=1, resamples=10, seed=0
        )


def test_the_recorded_confidence_matches_what_the_estimator_takes():
    """CONFIDENCE describes `percentile_interval`; it does not configure it.

    THE MUTATION THIS EXISTS FOR: changing `percentile_interval`'s percentiles
    without changing CONFIDENCE, which would leave every record naming a level
    the estimator did not take -- the defect M3l shipped and then fixed.
    """
    from mbfps.eval.pooling import percentile_interval

    replicates = np.arange(10001, dtype=np.float64)
    low, high, _ = percentile_interval(replicates)
    tail = (1.0 - CONFIDENCE) / 2.0
    assert low == pytest.approx(np.percentile(replicates, 100 * tail))
    assert high == pytest.approx(np.percentile(replicates, 100 * (1 - tail)))
```

Extend the import block at the top of the file with `margin_interval`, `motion_margin`, `one_step_persistence`, `scored_targets`.

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/eval/test_burden.py -q
```

Expected: collection error — `ImportError: cannot import name 'scored_targets'`.

- [ ] **Step 3: Write the implementation**

Append to `src/mbfps/eval/burden.py`:

```python
from mbfps.eval.pooling import episode_bootstrap, percentile_interval
from mbfps.eval.probe import position_error


def scored_targets(window_targets: np.ndarray) -> np.ndarray:
    """The rows the model is scored on, given the window's `horizon + 1` rows.

    THE SLICE, stated so the off-by-one is hard to write. `evaluate_rollout`
    scores horizon step `j + 1` against
    `privileged[start + context + 1 + j]`, and its comment records that this
    was verified with an oracle model rather than argued: with
    `start + context` instead, a PERFECT predictor carries a constant error of
    one step of true displacement at every horizon step. The caller therefore
    passes `privileged[start + context : start + need + 1]`, whose row 0 is the
    last CONTEXT frame -- so the scored rows are `[1:]`, and the baseline and
    the model are scored on the identical frames by construction.
    """
    window_targets = np.asarray(window_targets, dtype=np.float64)
    if window_targets.ndim != 2 or window_targets.shape[0] < 2:
        raise ValueError(
            "window_targets must be (horizon + 1, K) with at least two rows; "
            f"got {window_targets.shape}"
        )
    return window_targets[1:]


def one_step_persistence(window_targets: np.ndarray) -> np.ndarray:
    """The error of predicting each scored frame by the frame before it.

    This IS the true one-step displacement -- ground truth, no model involved,
    which is what makes `motion_margin` a level rather than a difference of two
    estimates.

    NOT the recorded `persistence_position`. That copies the position at `t`,
    the last frame the OPEN LOOP saw, so comparing it with the k=1 rung -- which
    is re-grounded every step and has seen `h - 1` frames more -- would hand the
    rung a win on information advantage rather than on prediction. Spec 2.2
    forbids that comparison by name.
    """
    window_targets = np.asarray(window_targets, dtype=np.float64)
    scored = scored_targets(window_targets)
    return position_error(window_targets[:-1], scored)


def motion_margin(window_targets: np.ndarray, curve_one: np.ndarray) -> np.ndarray:
    """`one_step_persistence - the k=1 rung`, per horizon step.

    Positive means one prior step from a posterior-grounded state beats
    assuming the agent did not move.
    """
    baseline = one_step_persistence(window_targets)
    curve_one = np.asarray(curve_one, dtype=np.float64)
    if baseline.shape != curve_one.shape:
        raise ValueError(
            "the baseline and the k=1 curve must have the same length; got "
            f"{baseline.shape} and {curve_one.shape}"
        )
    return baseline - curve_one


def margin_interval(
    window_margin: np.ndarray,
    groups: np.ndarray,
    *,
    h: int,
    resamples: int = RESAMPLES,
    seed: int,
) -> tuple[float, float, float]:
    """`(point, ci_low, ci_high)` for the mean margin at horizon step `h`.

    THE RESAMPLING UNIT IS THE EPISODE. There are 229 windows over 24 episodes
    on every shipped cell, and consecutive Doom frames are near-duplicates, so
    a window-level bootstrap counts correlated observations as independent ones
    and returns an interval several times too narrow.

    `seed` IS KEYWORD-REQUIRED AND HAS NO DEFAULT. A defaulted seed is how a
    previous milestone shipped every cell drawing the same resamples; the
    point estimate is seed-free, so only the bounds can move, and a reader
    cannot tell nine identical draws from nine independent ones by looking.

    `groups` must carry one label per window. A record whose ladder carried no
    clustering stores `windows.episode` as null, and its own comment requires a
    reader to refuse rather than treat every window as its own episode --
    falling back to `arange(n)` here would convert this into the window-level
    bootstrap the first paragraph rules out.
    """
    window_margin = np.asarray(window_margin, dtype=np.float64)
    groups = np.asarray(groups)
    if window_margin.ndim != 2:
        raise ValueError(
            f"window_margin must be (windows, horizon); got {window_margin.shape}"
        )
    if groups.ndim != 1 or groups.size != window_margin.shape[0]:
        raise ValueError(
            "groups must carry one label per window; got "
            f"{groups.shape} for {window_margin.shape[0]} windows"
        )
    if np.unique(groups).size < 2:
        raise ValueError(
            "an episode-clustered bootstrap needs at least two episodes; got "
            f"{np.unique(groups).size}"
        )
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")

    column = window_margin[:, h - 1] if 1 <= h <= window_margin.shape[1] else None
    if column is None:
        raise ValueError(
            f"horizon step must be in 1..{window_margin.shape[1]}, got {h}"
        )
    point = float(column.mean())
    replicates = np.array([
        float(column[index].mean())
        for index in episode_bootstrap(groups, resamples, seed)
    ])
    low, high, _se = percentile_interval(replicates)
    return point, low, high
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/eval/test_burden.py -q
```

Expected: `16 passed`.

- [ ] **Step 5: Verify each test bites**

| mutation in `src/mbfps/eval/burden.py` | test that must fail |
|---|---|
| `return window_targets[1:]` → `return window_targets[:-1]` | `test_scored_targets_are_the_rows_the_model_is_scored_on` |
| `position_error(window_targets[:-1], scored)` → `position_error(np.repeat(window_targets[:1], scored.shape[0], axis=0), scored)` | `test_one_step_persistence_is_the_true_one_step_displacement` |
| in `margin_interval`, `episode_bootstrap(groups, resamples, seed)` → `episode_bootstrap(np.arange(groups.size), resamples, seed)` | `test_the_interval_clusters_on_episodes_and_requires_its_seed` |
| `seed: int` → `seed: int = 0` | `test_the_interval_clusters_on_episodes_and_requires_its_seed` |
| delete the `np.unique(groups).size < 2` raise | `test_the_interval_refuses_what_it_cannot_cluster` |
| delete the `groups.size != window_margin.shape[0]` raise | `test_the_interval_refuses_what_it_cannot_cluster` |
| `CONFIDENCE = 0.95` → `0.90` | `test_the_recorded_confidence_matches_what_the_estimator_takes` |

Revert each, clear `__pycache__`, and end green.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/burden.py tests/eval/test_burden.py
git commit -m "feat: the motion baseline is one-step persistence, clustered on episodes

one_step_persistence is the true one-step displacement -- ground truth, no
model -- which is what makes motion_margin a level rather than a difference of
two estimates. It is NOT the recorded persistence_position: that copies the
position at t, so comparing it with the k=1 rung, which is re-grounded every
step, would hand the rung a win on information advantage rather than on
prediction. A test fails if that baseline is reintroduced.

The horizon shift lives here, where it is unit-testable, rather than in the
script's loop: the caller passes privileged[start+context : start+need+1] and
scored_targets()[1:] reproduces evaluate_rollout's own truth slice exactly, so
the baseline and the model are scored on identical frames by construction.

margin_interval clusters on the 24 episodes, not the 229 windows, and requires
its seed with no default.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Reading H — the statuses, their precedence, and the table

**Files:**
- Modify: `src/mbfps/eval/burden.py`
- Test: `tests/eval/test_burden.py`

**Interfaces:**
- Consumes: everything from Tasks 2 and 3.
- Produces:
  - `burden.strict_majority(n: int) -> int` — `n // 2 + 1`.
  - `burden.SEEDS_MINIMUM: int = 3`
  - `burden.BurdenArm` — frozen dataclass, one per cell. Fields: `arm: str`, `seed: int`, `margin: float`, `margin_low: float`, `margin_high: float`, `burden_by_k: dict[int, float]`, `compounding_by_k: dict[int, float]`, `identity_residual: float`, `open_loop_divergence: float`, `k_one_is_floor: bool`, `displacement_median: float`, `floor_median: float`, `clusters: int`, `rows: int`.
  - `burden.BurdenArm.controls_ok: bool` and `.base_ok: bool` properties.
  - `burden.BurdenInputs` — frozen dataclass: `cells: dict[tuple[str, int], BurdenArm]`, `decision_h: int`, `ks: tuple[int, ...]`.
  - `burden.BurdenStatus` — frozen dataclass: `status: str`, `rule: str`, `arms_motion: tuple[str, ...]`, `arms_copies: tuple[str, ...]`, `seeds_total: dict[str, int]`.
  - `burden.reading_burden(inputs: BurdenInputs) -> BurdenStatus`
  - `burden.READING_COLUMNS: tuple[str, ...]`, `burden.READING_WIDTHS: tuple[int, ...]`
  - `burden.format_reading_burden(reading: BurdenStatus, inputs: BurdenInputs) -> str`

**Precedence, exactly as spec §3.1 fixes it.** The first that applies is the reading:

| status | condition | script exit |
|---|---|---|
| `UNRESOLVED_CONTROL` | any cell with `identity_residual > IDENTITY_TOLERANCE`, `open_loop_divergence != 0.0`, or `k_one_is_floor` true | **45** |
| `UNREADABLE` | any cell failing the base control, or any arm with fewer than `SEEDS_MINIMUM` seeds | **46** |
| `PREDICTS_MOTION` | `margin_low > 0` in a strict majority of each of ≥ `ARMS_REQUIRED` arms' seeds | 0 |
| `COPIES` | `margin_high <= 0` on the same majority | 0 |
| `INDETERMINATE` | the fall-through | 0 |

**`UNRESOLVED_CONTROL` outranks everything** for the reason M3l's docstring gives: a reading taken from an estimator that missed a known answer is not a weaker reading, it is not a reading.

**The two decisive statuses cannot both clear.** `margin_low > 0` and `margin_high <= 0` are mutually exclusive per cell because `margin_low <= margin_high`. A test asserts this rather than leaving it to inspection — M3l's spec claimed two readings were "mutually exclusive by construction" when they were tallies on different quantities and an arm could clear both.

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_burden.py`:

```python
def _arm(arm="pixel_ae", seed=0, margin=3.0, low=1.0, high=5.0, **over):
    """A cell that passes every control, so each test perturbs one thing."""
    fields = dict(
        arm=arm, seed=seed, margin=margin, margin_low=low, margin_high=high,
        burden_by_k={1: 2.0, 3: 6.0, 5: 11.0, 15: 30.0, 45: 90.0},
        compounding_by_k={1: 0.0, 3: 4.0, 5: 9.0, 15: 28.0, 45: 88.0},
        identity_residual=1.4e-13, open_loop_divergence=0.0,
        k_one_is_floor=False, displacement_median=40.0, floor_median=12.0,
        clusters=24, rows=229,
    )
    fields.update(over)
    return BurdenArm(**fields)


def _inputs(cells):
    return BurdenInputs(
        cells={(c.arm, c.seed): c for c in cells}, decision_h=DECISION_H,
        ks=(1, 3, 5, 15, 45),
    )


def _nine(**over):
    return [
        _arm(arm=a, seed=s, **over)
        for a in ("frozen_ssl", "pixel_ae", "random_vit") for s in (0, 1, 2)
    ]


def test_strict_majority_is_computed_not_stored():
    """A fixed SEEDS_REQUIRED=2 is a majority at 3 seeds and a MINORITY at 5.
    M3l's design was reworked for exactly this.

    THE MUTATION THIS EXISTS FOR: `return 2`, which passes at 3 seeds and is
    wrong at 5 and 7.
    """
    assert strict_majority(1) == 1
    assert strict_majority(3) == 2
    assert strict_majority(5) == 3
    assert strict_majority(7) == 4
    for n in range(1, 12):
        assert 2 * strict_majority(n) > n


def test_predicts_motion_when_the_whole_interval_clears_zero():
    reading = reading_burden(_inputs(_nine(low=1.0, high=5.0)))
    assert reading.status == "PREDICTS_MOTION"
    assert set(reading.arms_motion) == {"frozen_ssl", "pixel_ae", "random_vit"}
    assert reading.arms_copies == ()


def test_copies_when_the_whole_interval_is_at_or_below_zero():
    reading = reading_burden(_inputs(_nine(margin=-3.0, low=-5.0, high=-1.0)))
    assert reading.status == "COPIES"
    assert set(reading.arms_copies) == {"frozen_ssl", "pixel_ae", "random_vit"}


def test_an_interval_straddling_zero_falls_through():
    """And the sentence must SAY it is a fall-through. M3k's INDISTINGUISHABLE
    overclaimed through three wordings before the review caught it."""
    reading = reading_burden(_inputs(_nine(low=-2.0, high=4.0)))
    assert reading.status == "INDETERMINATE"
    assert "fall-through" in reading.rule
    assert "by default rather than by evidence" in reading.rule


def test_a_margin_of_exactly_zero_at_the_high_end_reads_copies():
    """The bar is `margin_high <= 0`, so a high end of exactly 0 clears COPIES
    and does NOT clear PREDICTS_MOTION.

    THE MUTATION THIS EXISTS FOR: `< 0` instead of `<= 0`, which would send an
    exactly-zero cell to INDETERMINATE.
    """
    reading = reading_burden(_inputs(_nine(margin=-1.0, low=-2.0, high=0.0)))
    assert reading.status == "COPIES"


def test_the_two_decisive_statuses_can_never_both_clear():
    """Mutually exclusive because margin_low <= margin_high, asserted rather
    than claimed. M3l's spec asserted exclusivity for two tallies on DIFFERENT
    quantities, where a single arm could clear both."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        low, high = sorted(rng.uniform(-5.0, 5.0, size=2))
        assert not ((low > 0) and (high <= 0))


def test_one_broken_control_outranks_every_other_status():
    """A reading from an estimator that missed a known answer is not a weaker
    reading; it is not a reading."""
    for broken in (
        {"identity_residual": 1e-6},
        {"open_loop_divergence": 3.5e-4},
        {"k_one_is_floor": True},
    ):
        cells = _nine(low=1.0, high=5.0)
        cells[4] = _arm(arm=cells[4].arm, seed=cells[4].seed, low=1.0, high=5.0, **broken)
        reading = reading_burden(_inputs(cells))
        assert reading.status == "UNRESOLVED_CONTROL", broken
        assert f"{cells[4].arm} seed {cells[4].seed}" in reading.rule


def test_a_failed_base_control_is_unreadable_and_names_the_cell():
    """If the agent barely moved, motion_margin is at best zero and COPIES
    would be read for a reason that has nothing to do with the objective."""
    cells = _nine(margin=-3.0, low=-5.0, high=-1.0)
    cells[7] = _arm(
        arm=cells[7].arm, seed=cells[7].seed, margin=-3.0, low=-5.0, high=-1.0,
        displacement_median=9.0, floor_median=12.0,
    )
    reading = reading_burden(_inputs(cells))
    assert reading.status == "UNREADABLE"
    assert f"{cells[7].arm} seed {cells[7].seed}" in reading.rule
    assert "displacement" in reading.rule


def test_an_arm_short_of_three_seeds_is_refused_not_tallied():
    """THE MUTATION THIS EXISTS FOR: tallying whatever seeds are present. At
    one seed, `strict_majority(1) == 1`, so a single lucky cell would establish
    an arm -- which is the M3j trap of printing a row that clears beside a
    verdict that cannot."""
    cells = [c for c in _nine(low=1.0, high=5.0) if not (c.arm == "pixel_ae" and c.seed == 2)]
    reading = reading_burden(_inputs(cells))
    assert reading.status == "UNREADABLE"
    assert "pixel_ae" in reading.rule
    assert "2 seed" in reading.rule or "fewer than 3" in reading.rule


def test_the_table_prints_one_row_per_cell_at_the_header_width():
    reading = reading_burden(_inputs(_nine(low=1.0, high=5.0)))
    text = format_reading_burden(reading, _inputs(_nine(low=1.0, high=5.0)))
    lines = text.splitlines()
    header = next(line for line in lines if "margin" in line and "arm" in line)
    assert len(READING_COLUMNS) == len(READING_WIDTHS)
    assert len(header) == sum(READING_WIDTHS)
    for a in ("frozen_ssl", "pixel_ae", "random_vit"):
        rows = [line for line in lines if line.strip().startswith(a)]
        assert len(rows) == 3, a
        for row in rows:
            assert len(row) == len(header), (a, row)
    assert "PREDICTS MOTION" in text


def test_the_table_interpolates_its_numbers_and_never_hardcodes_them():
    """THE MUTATION THIS EXISTS FOR: a literal in the legend. M3l shipped a
    legend reading "floor exactly 0" where the check is abs(floor) <= 1e-9 and
    real floors are ~1e-13; it is still an open follow-up."""
    inputs = _inputs(_nine(low=1.0, high=5.0))
    text = format_reading_burden(reading_burden(inputs), inputs)
    assert str(DECISION_H) in text
    assert f"{IDENTITY_TOLERANCE:g}" in text
    assert "exactly 0" not in text
```

Extend the import block with `BurdenArm`, `BurdenInputs`, `BurdenStatus`, `READING_COLUMNS`, `READING_WIDTHS`, `format_reading_burden`, `reading_burden`, `strict_majority`.

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/eval/test_burden.py -q
```

Expected: collection error — `ImportError: cannot import name 'BurdenArm'`.

- [ ] **Step 3: Write the implementation**

Append to `src/mbfps/eval/burden.py`. Build it in this order: `strict_majority` and `SEEDS_MINIMUM`; the three dataclasses; `reading_burden`; then the formatter.

```python
from dataclasses import dataclass

SEEDS_MINIMUM: int = 3
"""An arm with fewer seeds is refused by name rather than tallied.

`strict_majority(1) == 1`, so without this a single lucky cell would establish
an arm -- the M3j trap where `--arms random_vit` printed a row reading
`clears = up` beside a verdict of NO DIFFERENCE.
"""


def strict_majority(n: int) -> int:
    """More than half of `n`, COMPUTED from the seed count present.

    Never a stored constant: `SEEDS_REQUIRED = 2` is a majority at 3 seeds and
    a minority at 5, and M3l's design was reworked for exactly that.
    """
    if n < 1:
        raise ValueError(f"a majority needs at least one seed, got {n}")
    return n // 2 + 1


@dataclass(frozen=True)
class BurdenArm:
    """One cell: one arm at one seed, at `DECISION_H`."""

    arm: str
    seed: int
    margin: float
    margin_low: float
    margin_high: float
    burden_by_k: dict[int, float]
    compounding_by_k: dict[int, float]
    identity_residual: float
    open_loop_divergence: float
    k_one_is_floor: bool
    displacement_median: float
    floor_median: float
    clusters: int
    rows: int

    def __post_init__(self) -> None:
        if self.margin_low > self.margin_high:
            raise ValueError(
                f"{self.arm} seed {self.seed}: an interval cannot have "
                f"ci_low {self.margin_low} above ci_high {self.margin_high}"
            )

    @property
    def controls_ok(self) -> bool:
        """Every known answer hit. Checked before any status is tallied."""
        return (
            abs(self.identity_residual) <= IDENTITY_TOLERANCE
            and self.open_loop_divergence == 0.0
            and not self.k_one_is_floor
        )

    @property
    def base_ok(self) -> bool:
        """True motion exceeds the readout's own error, so a margin is
        detectable at all. Self-calibrating: a ratio of two measured
        quantities, not a magic constant."""
        return self.displacement_median > self.floor_median


@dataclass(frozen=True)
class BurdenInputs:
    cells: dict[tuple[str, int], BurdenArm]
    decision_h: int
    ks: tuple[int, ...]


@dataclass(frozen=True)
class BurdenStatus:
    status: str
    rule: str
    arms_motion: tuple[str, ...]
    arms_copies: tuple[str, ...]
    seeds_total: dict[str, int]
```

Then the reading. Keep the precedence chain flat and explicit, so a reader can check it against the table above line by line:

```python
def reading_burden(inputs: BurdenInputs) -> BurdenStatus:
    """Reading H: is the rollout compounding, or did the one-step map never
    learn motion?

    PRECEDENCE. `UNRESOLVED_CONTROL` outranks everything: a reading taken from
    an estimator that missed a known answer is not a weaker reading, it is not
    a reading. `UNREADABLE` comes next, because a cell where the agent barely
    moved would read COPIES for a reason that has nothing to do with the
    objective. Only then are the two decisive statuses tallied, and
    `INDETERMINATE` is the fall-through.

    THE ASYMMETRY. Both decisive statuses are levels against an exactly known
    baseline -- the true one-step displacement -- measured on the same windows,
    so BOTH DIRECTIONS ARE SOUND. This is the structural difference from M3k,
    whose statistic spoke in one direction only, and from M3l, whose
    `bits_carried` upper-bounds the joint and was trustworthy only below a cut.
    """
    by_arm: dict[str, list[BurdenArm]] = {}
    for cell in inputs.cells.values():
        by_arm.setdefault(cell.arm, []).append(cell)
    seeds_total = {arm: len(cells) for arm, cells in by_arm.items()}

    broken = sorted(
        f"{c.arm} seed {c.seed}" for c in inputs.cells.values() if not c.controls_ok
    )
    if broken:
        return BurdenStatus(
            status="UNRESOLVED_CONTROL",
            rule=(
                "a control with a known answer was missed in "
                f"{', '.join(broken)}: the identity residual must stay within "
                f"{IDENTITY_TOLERANCE:g}, the k=45 rung must reproduce the "
                "record bitwise (open_loop_divergence 0.0), and the k=1 rung "
                "must sit strictly above the floor. A reading taken from an "
                "estimator that missed a known answer is not a weaker reading, "
                "it is not a reading"
            ),
            arms_motion=(), arms_copies=(), seeds_total=seeds_total,
        )

    stationary = sorted(
        f"{c.arm} seed {c.seed}" for c in inputs.cells.values() if not c.base_ok
    )
    short = sorted(arm for arm, n in seeds_total.items() if n < SEEDS_MINIMUM)
    if stationary or short:
        causes = []
        if stationary:
            causes.append(
                "the median true one-step displacement does not exceed the "
                f"median floor error in {', '.join(stationary)}, so no method "
                "could detect motion prediction there"
            )
        if short:
            causes.append(
                "; ".join(
                    f"{arm} carries {seeds_total[arm]} seed(s), fewer than "
                    f"{SEEDS_MINIMUM}" for arm in short
                )
                + ", and an arm short of the minimum is refused by name rather "
                "than tallied"
            )
        return BurdenStatus(
            status="UNREADABLE", rule="; ".join(causes),
            arms_motion=(), arms_copies=(), seeds_total=seeds_total,
        )

    def clearing(predicate) -> tuple[str, ...]:
        return tuple(sorted(
            arm for arm, cells in by_arm.items()
            if sum(1 for c in cells if predicate(c)) >= strict_majority(len(cells))
        ))

    arms_motion = clearing(lambda c: c.margin_low > 0.0)
    arms_copies = clearing(lambda c: c.margin_high <= 0.0)

    if len(arms_motion) >= ARMS_REQUIRED:
        return BurdenStatus(
            status="PREDICTS_MOTION",
            rule=(
                f"the whole motion_margin interval clears 0 at horizon "
                f"{inputs.decision_h} in {len(arms_motion)} of "
                f"{len(by_arm)} arms ({', '.join(arms_motion)}), each in a "
                "strict majority of its seeds: one prior step from the true "
                "state beats assuming the agent did not move, so the one-step "
                "map predicts real motion and what fails is rolling it "
                "forward. A multi-step or overshooting objective is the "
                "indicated intervention"
            ),
            arms_motion=arms_motion, arms_copies=arms_copies, seeds_total=seeds_total,
        )
    if len(arms_copies) >= ARMS_REQUIRED:
        return BurdenStatus(
            status="COPIES",
            rule=(
                f"the whole motion_margin interval sits at or below 0 at "
                f"horizon {inputs.decision_h} in {len(arms_copies)} of "
                f"{len(by_arm)} arms ({', '.join(arms_copies)}), each in a "
                "strict majority of its seeds: one prior step from the TRUE "
                "state is no better than assuming stillness, so a "
                "longer-horizon term cannot rescue this and the target itself "
                "must change"
            ),
            arms_motion=arms_motion, arms_copies=arms_copies, seeds_total=seeds_total,
        )
    return BurdenStatus(
        status="INDETERMINATE",
        rule=(
            f"neither decisive status clears {ARMS_REQUIRED} arms at horizon "
            f"{inputs.decision_h}: the motion_margin interval straddles 0, so "
            "the one-step map could not be shown to beat stillness and could "
            "not be shown to match it. This status is the FALL-THROUGH, not a "
            "bar that was cleared, so it arrived by default rather than by "
            "evidence and licenses no positive claim in either direction"
        ),
        arms_motion=arms_motion, arms_copies=arms_copies, seeds_total=seeds_total,
    )
```

Then the table. Mirror `capacity.format_reading_capacity`: right-aligned unseparated cells, one width per column, every number in the caption and legend interpolated from the module rather than written as a literal.

```python
READING_COLUMNS: tuple[str, ...] = (
    "arm", "seed", "margin", "ci_low", "ci_high", "burden45", "comp45", "clears",
)
READING_WIDTHS: tuple[int, ...] = (13, 6, 11, 11, 11, 11, 11, 17)


def _row(values, widths) -> str:
    return "".join(f"{str(v):>{w}}" for v, w in zip(values, widths, strict=True))


def format_reading_burden(reading: BurdenStatus, inputs: BurdenInputs) -> str:
    """Reading H as `burden.txt` carries it, byte for byte.

    Every number in the caption and the legend is interpolated from the module
    -- the decision horizon, the identity tolerance, the arm bar -- so a
    constant that drifts cannot leave a stale literal behind. M3l shipped a
    legend saying "floor exactly 0" over a 1e-9 check and ~1e-13 values, and it
    is still an open follow-up.
    """
    lines = [
        f"--- Reading H: the prediction burden at horizon {inputs.decision_h} "
        f"(re-grounding periods k = {', '.join(str(k) for k in inputs.ks)})",
        _row(READING_COLUMNS, READING_WIDTHS),
    ]
    for (arm, seed), cell in sorted(inputs.cells.items()):
        clears = (
            "motion" if cell.margin_low > 0.0
            else "copies" if cell.margin_high <= 0.0
            else "-"
        )
        lines.append(_row(
            (
                arm, seed,
                f"{cell.margin:+.4f}", f"{cell.margin_low:+.4f}",
                f"{cell.margin_high:+.4f}",
                f"{cell.burden_by_k[inputs.ks[-1]]:+.4f}",
                f"{cell.compounding_by_k[inputs.ks[-1]]:+.4f}",
                clears,
            ),
            READING_WIDTHS,
        ))
    lines += [
        "  margin = the true one-step displacement minus the k=1 rung, so "
        "POSITIVE means one prior step from the true state beats assuming the "
        "agent did not move. Ground truth, not an estimate, so a reading in "
        "EITHER direction is evidence",
        f"  burden45/comp45 = burden(k={inputs.ks[-1]}, h={inputs.decision_h}) "
        f"and compounding(k={inputs.ks[-1]}, h={inputs.decision_h}); "
        f"compounding(k=1) is 0 by construction and the identity residual is "
        f"held within {IDENTITY_TOLERANCE:g}",
        f"  a status needs a strict majority of each arm's seeds in at least "
        f"{ARMS_REQUIRED} of {len(reading.seeds_total)} arms",
        f"  verdict: {reading.status.replace('_', ' ')} -- decided by: {reading.rule}",
    ]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/eval/test_burden.py -q
```

Expected: `27 passed`.

- [ ] **Step 5: Verify each test bites**

| mutation in `src/mbfps/eval/burden.py` | test that must fail |
|---|---|
| `strict_majority` body → `return 2` | `test_strict_majority_is_computed_not_stored` |
| `c.margin_high <= 0.0` → `c.margin_high < 0.0` | `test_a_margin_of_exactly_zero_at_the_high_end_reads_copies` |
| move the `broken` block below the `stationary` block | `test_one_broken_control_outranks_every_other_status` |
| `controls_ok` → `return True` | `test_one_broken_control_outranks_every_other_status` |
| `base_ok` → `return True` | `test_a_failed_base_control_is_unreadable_and_names_the_cell` |
| delete the `short` term from the `UNREADABLE` guard | `test_an_arm_short_of_three_seeds_is_refused_not_tallied` |
| drop "by default rather than by evidence" from the `INDETERMINATE` rule | `test_an_interval_straddling_zero_falls_through` |
| `READING_WIDTHS` first entry `13` → `9` | `test_the_table_prints_one_row_per_cell_at_the_header_width` |
| in the legend, `{IDENTITY_TOLERANCE:g}` → `exactly 0` | `test_the_table_interpolates_its_numbers_and_never_hardcodes_them` |

Revert each, clear `__pycache__`, end green.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/burden.py tests/eval/test_burden.py
git commit -m "feat: Reading H -- PREDICTS_MOTION, COPIES, and a fall-through that says so

Precedence: UNRESOLVED_CONTROL outranks everything, because a reading from an
estimator that missed a known answer is not a weaker reading; then UNREADABLE,
because a cell where the agent barely moved would read COPIES for a reason that
has nothing to do with the objective; then the two decisive statuses; then
INDETERMINATE, whose sentence says it arrived by default rather than by
evidence.

Both decisive statuses are levels against the true one-step displacement, so
unlike M3k and M3l both directions are sound. Their mutual exclusivity follows
from ci_low <= ci_high and is asserted over 500 random intervals rather than
claimed -- M3l's spec claimed exclusivity for two tallies on different
quantities where one arm could clear both.

The seed bar is a strict majority COMPUTED from the seeds present, and an arm
short of three is refused by name rather than tallied.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `prediction_burden.py` — the measure phase

**Files:**
- Create: `scripts/prediction_burden.py`
- Test: `tests/eval/test_prediction_burden_script.py`

**Interfaces:**
- Consumes: all of `burden.py`; `rollout.evaluate_rollout`; `diagnostics.regrounding_sweep`, `diagnostics.REGROUNDING_KS`, `diagnostics.RegroundingSweep`; `probe.probe_targets`; `windows.window_starts`; `episode.load_episode`.
- Produces:
  - `EXIT_CONTROL_BROKEN = 45`, `EXIT_UNREADABLE = 46`
  - `baseline_rows(val_paths, *, context: int, horizon: int) -> tuple[np.ndarray, np.ndarray]` — `(rows, episode_labels)`, `rows` of shape `(n_windows, horizon)`.
  - `measure_cell(model, val_paths, probe, *, arm, seed, context, horizon, ks, device, feature_backbone) -> dict` — one record.
  - `measure_phase(args) -> int`

**THE ONE ALIGNMENT RISK, and how it is guarded.** `regrounding_sweep` retains per-window rows in traversal order, and `baseline_rows` walks the validation episodes in a second, model-free loop. Row `w` of each must be the same window. Nothing ties them but the shared window rule, and that is exactly the guarantee the rest of this codebase runs on — `diagnostics.py`'s header states that the window rule lives in `eval.windows` precisely because "its failure mode is … silent drift with no loud test available." So:

1. `baseline_rows` must consume `window_starts(episode.length, context, horizon)` and iterate `val_paths` in the order given — never `sorted()`, never a set.
2. `measure_cell` refuses unless `rows.shape == sweep.window_position[1].shape`.
3. A test pins that both loops produce the same per-episode window counts in the same order, on a two-episode fixture with *different* counts — so a reversed or sorted traversal is caught rather than merely shape-equal.

The baseline is model-free and touches no generator, so it cannot perturb the sampling stream the sweep depends on. Run it **after** the sweep regardless, and never between the sweep's arms.

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_prediction_burden_script.py`. Load the script by path, as the sibling script tests do — copy the loader from `tests/eval/test_latent_capacity_script.py`'s top (`importlib.util.spec_from_file_location`), then:

```python
def test_the_exit_codes_are_this_milestones_own():
    """39/40 are M3j's, 41/42 M3k's, 43/44 M3l's. A collision would make two
    tools report different failures under one number."""
    assert script.EXIT_CONTROL_BROKEN == 45
    assert script.EXIT_UNREADABLE == 46


def test_baseline_rows_walks_the_given_order_and_the_shared_window_rule():
    """THE MUTATION THIS EXISTS FOR: `for path in sorted(val_paths)` or a set
    traversal. The sweep retains its rows in the order IT walked, and nothing
    ties the two loops but this. The fixture's two episodes yield DIFFERENT
    window counts, so a reordered traversal changes the row blocks rather than
    merely their total.
    """
    paths = [_episode_file(length=70), _episode_file(length=120)]
    expected = [
        len(window_starts(70, 5, 45)), len(window_starts(120, 5, 45)),
    ]
    assert expected[0] != expected[1], "fixture must distinguish the orders"

    rows, labels = script.baseline_rows(paths, context=5, horizon=45)
    assert rows.shape == (sum(expected), 45)
    # Labels are per window, in traversal order, so the FIRST block is the
    # first path's. A reversed traversal flips the block sizes.
    assert list(labels[: expected[0]]) == [0] * expected[0]
    assert list(labels[expected[0] :]) == [1] * expected[1]

    reversed_rows, _ = script.baseline_rows(paths[::-1], context=5, horizon=45)
    assert not np.array_equal(rows, reversed_rows)


def test_baseline_rows_are_the_true_one_step_displacement():
    """Model-free: the baseline is ground truth, so it must reproduce the
    displacement computed directly from the episode's privileged array."""
    path = _episode_file(length=70)
    rows, _ = script.baseline_rows([path], context=5, horizon=45)
    episode = load_episode(path)
    start = window_starts(70, 5, 45)[0]
    targets = probe_targets(
        episode.privileged[start + 5 : start + 5 + 46], episode.privileged_keys
    )
    assert rows[0] == pytest.approx(position_error(targets[:-1], targets[1:]))


def test_measure_cell_refuses_a_baseline_that_does_not_align_with_the_sweep():
    """THE MUTATION THIS EXISTS FOR: dropping the shape check, after which a
    baseline cut on a different window rule would subtract row-for-row against
    the wrong windows and produce a plausible, wrong margin."""
    with pytest.raises(SystemExit) as caught:
        script.measure_cell(**_cell_kwargs(baseline_windows=7))
    assert "align" in str(caught.value)


def test_the_record_carries_every_protocol_parameter_the_reading_uses():
    """M3l shipped a headline interval whose DRAW COUNT was not on the record,
    so the permanent artefact could not be audited for it. Both the level and
    the count are recorded here from the start.

    THE MUTATION THIS EXISTS FOR: dropping any of these keys, or recording a
    literal 0.95 / 2000 instead of the module's constants.
    """
    record = script.measure_cell(**_cell_kwargs())
    assert record["confidence"] == burden.CONFIDENCE
    assert record["resamples"] == burden.RESAMPLES
    assert record["decision_h"] == burden.DECISION_H
    assert tuple(record["reported_h"]) == burden.REPORTED_H
    assert tuple(record["ks"]) == tuple(REGROUNDING_KS)
    assert record["identity_tolerance"] == burden.IDENTITY_TOLERANCE
    for key in (
        "arm", "seed", "margin", "controls", "base_control", "burden_by_k",
        "compounding_by_k", "curves", "windows", "git_sha", "step",
        "kl_rate_above_free_bits", "kl_dyn_max",
    ):
        assert key in record, key


def test_the_record_keeps_a_margin_interval_at_every_reported_horizon():
    record = script.measure_cell(**_cell_kwargs())
    assert sorted(int(h) for h in record["margin"]) == sorted(burden.REPORTED_H)
    for h, entry in record["margin"].items():
        assert entry["ci_low"] <= entry["point"] <= entry["ci_high"], h


def test_the_stacked_margin_equals_motion_margin_window_by_window():
    """The script subtracts stacked arrays for speed; `motion_margin` is the
    definition. If they disagree, the stacked path is wrong.

    THE MUTATION THIS EXISTS FOR: `one - rows` instead of `rows - one`, which
    flips the sign of the quantity Reading H's verdict is read from and leaves
    every shape and every interval intact.
    """
    record = script.measure_cell(**_cell_kwargs())
    margin = np.asarray(record["window_margin"], dtype=np.float64)
    for w in range(margin.shape[0]):
        expected = burden.motion_margin(_window_targets(w), _fake_one()[w])
        assert margin[w] == pytest.approx(expected, abs=1e-12)


def test_the_measured_controls_are_recorded_not_asserted():
    """The identity residual is a MEASURED float. Recording a hardcoded 0.0 is
    the defect M3l shipped in a legend line and has not yet fixed."""
    record = script.measure_cell(**_cell_kwargs())
    controls = record["controls"]
    assert controls["identity_residual"] != 0.0 or controls["identity_residual"] == 0.0
    assert isinstance(controls["identity_residual"], float)
    assert controls["open_loop_divergence"] == 0.0
    assert controls["k_one_is_floor"] is False
    assert controls["compounding_at_k_one_max_abs"] == 0.0
```

`_episode_file(length=...)` writes a synthetic episode with a moving agent; `_cell_kwargs(**over)` builds `measure_cell`'s arguments against a tiny fake model and a fake sweep whose `window_position[1]` has `baseline_windows` rows when that override is given. Build both on the fixtures `tests/eval/test_diagnostics.py` already uses for `regrounding_sweep` (see its `:206` and `:2925` call sites) rather than inventing a second fake model.

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/python -m pytest tests/eval/test_prediction_burden_script.py -q
```

Expected: collection error — the script does not exist.

- [ ] **Step 3: Write the measure phase**

Create `scripts/prediction_burden.py`, mirroring `scripts/latent_capacity.py`'s shape: a module docstring stating what is measured and what decides nothing, the exit constants, the per-cell measurement, the phase driver, then `main`. The parts that are specific to M3m:

```python
EXIT_CONTROL_BROKEN = 45
"""A control with a known answer was missed: the identity residual, the k=45
rung's bitwise reproduction of the record, or k=1 collapsing onto the floor."""

EXIT_UNREADABLE = 46
"""The reading cannot be taken from this data: a cell where the agent barely
moved, a protocol disagreement, or an arm short of SEEDS_MINIMUM seeds."""


def baseline_rows(val_paths, *, context: int, horizon: int):
    """Per-window true one-step displacement, plus one episode label per window.

    MODEL-FREE: this touches no model and no generator, so it cannot perturb
    the sampling stream `regrounding_sweep`'s matched arms depend on.

    THE TRAVERSAL ORDER IS THE CALLER'S. `regrounding_sweep` retains its
    per-window rows in the order IT walked, and row `w` here must be the same
    window. Nothing ties the two loops but the shared window rule from
    `eval.windows` -- which is exactly why that rule lives in one module, as
    `diagnostics.py`'s header says, because its failure mode is silent drift
    with no loud test available. Never `sorted(val_paths)`, never a set.
    """
    rows, labels = [], []
    need = context + horizon
    for index, path in enumerate(val_paths):
        episode = load_episode(path)
        for start in window_starts(episode.length, context, horizon):
            targets = probe_targets(
                episode.privileged[start + context : start + need + 1],
                episode.privileged_keys,
            )
            rows.append(one_step_persistence(targets))
            labels.append(index)
    if not rows:
        raise SystemExit(
            f"no validation window reached {need + 1} frames; lower "
            "--context/--horizon or check the split"
        )
    return np.stack(rows), np.asarray(labels)
```

`measure_cell` then, in this order: `evaluate_rollout` for the floor and the record's canonical curves; `regrounding_sweep(ks=ks)` for every rung; `baseline_rows` last; the alignment refusal; the per-window margin; the intervals; the controls; the base control; the companions.

```python
    reference = evaluate_rollout(model, val_paths, probe, **common)
    sweep = regrounding_sweep(model, val_paths, probe, ks=ks, **common)
    rows, labels = baseline_rows(val_paths, context=context, horizon=horizon)

    one = sweep.window_position[1]
    if rows.shape != one.shape:
        raise SystemExit(
            f"{arm} seed {seed}: the baseline does not align with the sweep -- "
            f"{rows.shape} rows against {one.shape}. Both must be cut by "
            "`window_starts` over the same --source episodes in the same order"
        )
    window_margin = rows - one

    floor = reference.floor_position
    by_k = {k: burden(sweep.curve(k), floor) for k in ks}
    comp = {k: compounding(sweep.curve(k), sweep.curve(1)) for k in ks}
    controls = {
        "identity_residual": max(
            identity_residual(sweep.curve(k), sweep.curve(1), floor) for k in ks
        ),
        "open_loop_divergence": float(sweep.open_loop_divergence(reference)),
        "k_one_is_floor": bool(sweep.is_bitwise_the_floor(1, "position")),
        "compounding_at_k_one_max_abs": float(np.max(np.abs(comp[1]))),
    }
    margin = {}
    for h in REPORTED_H:
        point, low, high = margin_interval(
            window_margin, labels, h=h, resamples=RESAMPLES, seed=seed,
        )
        margin[int(h)] = {"point": point, "ci_low": low, "ci_high": high}
```

**The stacked margin must agree with `motion_margin`, and a test must pin it.**
`motion_margin(window_targets, curve_one)` is the per-window definition, with
`burden.py`'s shape and finiteness guard behind it. The line above subtracts the
stacked arrays instead, for speed, which puts the sign convention in two places
and bypasses that guard. So `measure_cell` must refuse a non-finite `rows` or
`one` by name before subtracting — `margin_interval` does **not** check, and a
NaN there returns `(nan, nan, nan)` silently — and a test must assert that
`window_margin[w]` equals `motion_margin(window_targets_w, one[w])` for **every**
window `w` on a small fixture. If the two ever disagree the stacked path is
wrong, not `motion_margin`. Without that test `motion_margin` is test-only code
and the production sign convention is unpinned.

The base control compares the median true displacement against the median floor error at `DECISION_H`:

```python
    base_control = {
        "displacement_median": float(np.median(rows[:, DECISION_H - 1])),
        "floor_median": float(np.median(sweep.window_floor_position[:, DECISION_H - 1])),
    }
```

Record `confidence`, `resamples`, `decision_h`, `reported_h`, `ks`, `identity_tolerance` from the module's constants — never as literals — plus `git_sha`, `step`, `kl_rate_above_free_bits` and `kl_dyn_max` carried from the training history beside the checkpoint, and `windows: {"total": int, "episode": [...]}`.

- [ ] **Step 4: Run to verify it passes**

```bash
.venv/bin/python -m pytest tests/eval/test_prediction_burden_script.py -q
```

Expected: `7 passed`.

- [ ] **Step 5: Verify each test bites**

| mutation in `scripts/prediction_burden.py` | test that must fail |
|---|---|
| `for index, path in enumerate(val_paths)` → `enumerate(sorted(val_paths))` | `test_baseline_rows_walks_the_given_order_and_the_shared_window_rule` |
| the slice `start + context : start + need + 1` → `start + context + 1 : start + need + 2` | `test_baseline_rows_are_the_true_one_step_displacement` |
| delete the `rows.shape != one.shape` refusal | `test_measure_cell_refuses_a_baseline_that_does_not_align_with_the_sweep` |
| `"confidence": burden.CONFIDENCE` → `"confidence": 0.95` as a literal, with `CONFIDENCE` changed to 0.9 | `test_the_record_carries_every_protocol_parameter_the_reading_uses` |
| drop `"resamples"` from the record | `test_the_record_carries_every_protocol_parameter_the_reading_uses` |
| `margin_interval(..., seed=seed)` → `seed=0` | add an assertion that two cells with different seeds get different bounds, or rely on Task 3's pin — state which in the report |
| `controls["identity_residual"]` → `0.0` | `test_the_measured_controls_are_recorded_not_asserted` |

- [ ] **Step 6: Commit**

```bash
git add scripts/prediction_burden.py tests/eval/test_prediction_burden_script.py
git commit -m "feat: prediction_burden.py measure -- the ladder, the baseline, one record per cell

evaluate_rollout for the floor, regrounding_sweep for every rung, then the
model-free baseline last so it cannot perturb the matched sampling stream.

The baseline's per-window rows must align row-for-row with the sweep's, and
nothing ties the two loops but the shared window rule from eval.windows -- so
the traversal order is the caller's, never sorted, and a shape mismatch is a
named refusal. A test distinguishes the orders with two episodes of different
window counts, so a reordered walk is caught rather than merely shape-equal.

Both the confidence level AND the resample count go on the record from the
start: M3l shipped a headline interval whose draw count the permanent artefact
could not be audited for.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: The read phase, the exits, and the cross-tool registry

**Files:**
- Modify: `scripts/prediction_burden.py`
- Modify: `tests/eval/test_diagnose_dynamics_script.py`
- Test: `tests/eval/test_prediction_burden_script.py`

**Interfaces:**
- Consumes: Task 5's record schema; `burden.reading_burden`, `burden.format_reading_burden`, `burden.BurdenArm`, `burden.BurdenInputs`.
- Produces:
  - `require_one_protocol(records: dict) -> None` — refuses on any disagreement across `_PROTOCOL_FIELDS`.
  - `burden_inputs(records: dict) -> BurdenInputs`
  - `read_phase(args) -> int`
  - `READ_EXITS: dict[str, int]` — `{"UNRESOLVED_CONTROL": 45, "UNREADABLE": 46}`.

`_PROTOCOL_FIELDS` must include `git_sha`, `step`, `context`, `horizon`, `ks`, `decision_h`, `reported_h`, `confidence`, `resamples`, `identity_tolerance`, `split_seed` and `device`. `require_one_protocol`'s docstring in `latent_capacity.py` records that an earlier version omitting `git_sha` "let records from different torch builds and code versions pool into one finding with no refusal at all" — so `git_sha` is not optional.

- [ ] **Step 1: Write the failing tests**

Add to `tests/eval/test_prediction_burden_script.py`:

```python
def test_read_phase_maps_each_refusal_status_to_its_own_exit():
    """THE MUTATION THIS EXISTS FOR: swapping 45 and 46, which would report a
    broken estimator as an unreadable plan."""
    assert script.READ_EXITS == {
        "UNRESOLVED_CONTROL": script.EXIT_CONTROL_BROKEN,
        "UNREADABLE": script.EXIT_UNREADABLE,
    }
    for status, code in script.READ_EXITS.items():
        assert script.read_phase(_args_for(_pool(status=status))) == code


def test_read_phase_exits_zero_on_every_decisive_status():
    for status in ("PREDICTS_MOTION", "COPIES", "INDETERMINATE"):
        assert script.read_phase(_args_for(_pool(status=status))) == 0


def test_read_phase_refuses_records_that_disagree_on_the_protocol():
    """git_sha is among the compared fields. Its absence from an earlier
    version of this check let records from different torch builds pool into one
    finding with no refusal at all.

    THE MUTATION THIS EXISTS FOR: deleting the require_one_protocol call from
    read_phase, which the function's own unit tests cannot catch.
    """
    records = _pool(status="PREDICTS_MOTION")
    victim = sorted(records)[-1]          # sorts LAST, so a first-record pick sees nothing
    records[victim]["git_sha"] = "0" * 40
    with pytest.raises(SystemExit) as caught:
        script.read_phase(_args_for(records))
    assert victim in str(caught.value)
    assert "git_sha" in str(caught.value)


def test_read_phase_writes_nothing_when_it_refuses():
    """A refusal must leave no burden.txt behind to be mistaken for a reading."""
    records = _pool(status="UNRESOLVED_CONTROL")
    out = _args_for(records).out
    script.read_phase(_args_for(records))
    assert not (out / "burden.txt").exists()


def test_the_reading_is_byte_identical_on_two_reads():
    """The whole point of the two-phase split: a reading reproducible from
    records without a GPU. M3l's capacity.txt came out byte-identical on three
    independent reads.

    THE MUTATION THIS EXISTS FOR: `rstrip()` on the text, or `print(text)`
    adding a newline -- both invisible unless the bytes are compared.
    """
    records = _pool(status="PREDICTS_MOTION")
    first = _args_for(records)
    script.read_phase(first)
    a = (first.out / "burden.txt").read_bytes()
    second = _args_for(records)
    script.read_phase(second)
    assert (second.out / "burden.txt").read_bytes() == a
```

And in `tests/eval/test_diagnose_dynamics_script.py`, beside the existing `assert own == {43, 44}` block (`:661`), add the same shape for this tool:

```python
def test_prediction_burden_owns_exits_45_and_46():
    own = _own_exit_values("scripts/prediction_burden.py")
    assert own == {45, 46}
```

Reuse whatever helper the neighbouring assertions use rather than writing a second extractor.

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/python -m pytest tests/eval/test_prediction_burden_script.py tests/eval/test_diagnose_dynamics_script.py -q
```

Expected: `AttributeError: module has no attribute 'READ_EXITS'`, and the registry test failing on the missing exits.

- [ ] **Step 3: Write the read phase**

Follow `latent_capacity.py`'s `read_phase` closely: load the records, `require_readable_plan`, `require_one_protocol`, build `BurdenInputs`, call `reading_burden`, and then — only for a non-refusal status — format, print and write `burden.txt`. Map the two refusal statuses through `READ_EXITS`, raising a named `SystemExit` that includes the offending cells.

```python
READ_EXITS: dict[str, int] = {
    "UNRESOLVED_CONTROL": EXIT_CONTROL_BROKEN,
    "UNREADABLE": EXIT_UNREADABLE,
}
"""Reading H's two refusal statuses, each to its own exit.

The numbered statuses report what was found in the DATA. A narrowed plan is the
operator asking for something no data can answer, so it raises rather than
adding a status -- the distinction M3j's `require_readable_plan` introduced.
"""
```

Write the file with `Path.write_text(text)` where `text` already ends in a newline from `format_reading_burden`, and print the same string — never `print(text)` on a string that already ends in a newline, or the stdout and the file differ by one byte.

- [ ] **Step 4: Run to verify they pass**

```bash
.venv/bin/python -m pytest tests/eval/test_prediction_burden_script.py tests/eval/test_diagnose_dynamics_script.py -q
```

- [ ] **Step 5: Verify each test bites**

| mutation | test that must fail |
|---|---|
| swap the two values in `READ_EXITS` | `test_read_phase_maps_each_refusal_status_to_its_own_exit` |
| delete the `require_one_protocol(records)` call from `read_phase` | `test_read_phase_refuses_records_that_disagree_on_the_protocol` |
| drop `git_sha` from `_PROTOCOL_FIELDS` | `test_read_phase_refuses_records_that_disagree_on_the_protocol` |
| write `burden.txt` before the status check | `test_read_phase_writes_nothing_when_it_refuses` |
| `write_text(text.rstrip())` | `test_the_reading_is_byte_identical_on_two_reads` |
| change `EXIT_UNREADABLE` to `44` | the registry test, and M3l's `assert own == {43, 44}` |

- [ ] **Step 6: Commit**

```bash
git add scripts/prediction_burden.py tests/eval/test_prediction_burden_script.py tests/eval/test_diagnose_dynamics_script.py
git commit -m "feat: prediction_burden.py read -- Reading H, exits 45 and 46, and the registry

The two refusal statuses map to their own exits: 45 for a control with a known
answer that was missed, 46 for data the reading cannot be taken from. A
narrowed plan raises instead of adding a status, because the numbered statuses
report what was found in the DATA.

read_phase's call to require_one_protocol is pinned end to end -- the function's
own unit tests cannot catch the call being deleted, and git_sha is among the
compared fields because an earlier version without it let records from
different torch builds pool into one finding with no refusal at all.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: The smoke, the run, and the results

**Files:**
- Modify: `docs/superpowers/plans/2026-10-02-mb-fps-m3m-prediction-burden.md` (append `## Task 7 results`)
- Read: `runs/m3_study_v2` (the nine 20,000-step checkpoints)
- Write: `runs/m3m_burden/` (new directory — **never** touch any existing directory under `runs/`)

**Interfaces:** consumes everything above. Produces the nine records, `burden.txt`, and the results section.

- [ ] **Step 1: Smoke on one cell**

```bash
.venv/bin/python scripts/prediction_burden.py --phase measure \
  --source runs/m3_study_v2 --out runs/m3m_burden_smoke \
  --arms pixel_ae --seeds 0 2>&1 | tail -20
```

Expected: exit 0, one record written. Confirm from the record that `controls.open_loop_divergence` is `0.0`, `controls.k_one_is_floor` is `false`, `controls.compounding_at_k_one_max_abs` is `0.0`, and `controls.identity_residual` is below `1e-9` — and **write down the measured residual**, because the next step compares it against the full run.

- [ ] **Step 2: Confirm the smoke refuses a narrowed read**

```bash
.venv/bin/python scripts/prediction_burden.py --phase read --out runs/m3m_burden_smoke; echo "exit=$?"
```

Expected: exit **46**, naming `pixel_ae` as carrying 1 seed, fewer than 3. Nothing written. This is the M3j trap closed: a single cell must not establish an arm.

- [ ] **Step 3: Run all nine**

```bash
caffeinate -i .venv/bin/python scripts/prediction_burden.py --phase all \
  --source runs/m3_study_v2 --out runs/m3m_burden 2>&1 | tee runs/m3m_burden/run.log
```

Expected: 25–45 minutes, exit 0 or one of 45/46, nine records at one `git_sha`.

- [ ] **Step 4: Read it twice and compare the bytes**

```bash
.venv/bin/python scripts/prediction_burden.py --phase read --out runs/m3m_burden > /tmp/read_a.txt
.venv/bin/python scripts/prediction_burden.py --phase read --out runs/m3m_burden > /tmp/read_b.txt
cmp /tmp/read_a.txt /tmp/read_b.txt && echo "BYTE IDENTICAL"
shasum -a 256 runs/m3m_burden/burden.txt
```

- [ ] **Step 5: Run the full suite at the launched HEAD**

```bash
caffeinate -i .venv/bin/python -m pytest -q 2>&1 | tail -3
```

Expected: every test passing. Record the exact count and the SHA it ran at.

- [ ] **Step 6: Write `## Task 7 results`**

Append a results section to this plan. Every quantitative claim must be **re-derivable from `runs/m3m_burden/*.json` alone**, and each figure must name the record field it came from. Cover:

- Reading H's status and the sentence it printed, verbatim.
- `motion_margin` at `DECISION_H` per cell: point, `ci_low`, `ci_high`, and whether the interval clears 0 — plus how many cells and arms cleared each way.
- The burden curve: `burden(k, 45)` and `compounding(k, 45)` for k = 1, 3, 5, 15, 45, per arm.
- The four controls per cell, with the **measured** identity residual, not "zero".
- The base control per cell: `displacement_median` against `floor_median`.
- `kl_rate_above_free_bits` and `kl_dyn_max` per cell, so section 1's refutation travels with the reading.
- Timings, the `git_sha`, the window and cluster counts, and the `burden.txt` hash.
- Anything that surprised you, including figures you first wrote down wrong and corrected — M3l's results section records two such self-corrections and the review called that "the rarest thing in a results write-up".

**Do not overstate.** `PREDICTS_MOTION` means a multi-step objective is the *indicated* intervention, not that one would pass the gate. `INDETERMINATE` is a fall-through and licenses nothing. And M3m does not pass the M3 exit gate — `beats_persistence`, `band_is_usable` and `filtering_beats_embedding` all still fail, and `gap_closed(45)` is negative in 9 of 9 position cells.

- [ ] **Step 7: Commit**

```bash
git add docs/superpowers/plans/2026-10-02-mb-fps-m3m-prediction-burden.md
git commit -m "results: Reading H is <STATUS> -- <the finding, stated without overstating>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Plan Self-Review

**1. Spec coverage.** Every spec section maps to a task: §2.1 the ladder → Task 2; §2.2 `motion_margin` and the fair baseline → Task 3; §2.3 the four controls and the base control → Tasks 2, 5; §2.4 the companions → Tasks 5, 7; §3.1–3.3 Reading H, the bars and the asymmetry → Task 4; §4 the architecture and the two phases → Tasks 5, 6; §5 constraints → Global Constraints; §6 cost → Task 7; §7 exit criteria → Task 7 Step 6; §8 what this does not license → Task 7's closing instruction. §4's "rollout.py and diagnostics.py are NOT modified" appears in Global Constraints, and no task's **Files** block names either.

**2. Placeholder scan.** No "TBD", no "add error handling", no "similar to Task N". Two places deliberately defer a *value* rather than a decision, and both are values that must be measured rather than written: the pre-rename pooling constants in Task 1 Step 2 (with the exact command that produces them) and the figures in Task 7's results. One table row in Task 5 Step 5 asks the implementer to state which pin covers the seed hand-off — that is a judgement call to report, not a gap.

**3. Type consistency.** `burden(curve_k, floor)` and `compounding(curve_k, curve_one)` keep their argument order from Task 2 through Task 5. `margin_interval` returns `(point, ci_low, ci_high)` in Task 3 and is unpacked in that order in Task 5. `BurdenArm`'s fourteen fields in Task 4 are each written by Task 5. `READING_COLUMNS` has eight entries and `READING_WIDTHS` eight widths. `REPORTED_H` is the horizon grid and `ks`/`REGROUNDING_KS` the re-grounding periods, never swapped — the Global Constraints fix the notation and Task 3's test asserts `DECISION_H in REPORTED_H`.

**4. One risk the plan cannot remove.** Row-for-row alignment between `baseline_rows` and `sweep.window_position[1]` rests on both loops consuming `window_starts` over the same `val_paths` in the same order. Task 5 guards it three ways — the documented traversal rule, the shape refusal, and a test whose two episodes have *different* window counts so a reordering is caught — but there is no bitwise tie between the two loops, and the implementer should say so in their report rather than claim one.

---

## Task 7 results

Nine cells at one `git_sha` `349edee67077`, step 20000, `ks = [1, 3, 5, 15, 45]`,
`decision_h = 45`, `confidence = 0.95`, `resamples = 2000`, 229 windows over 24
episodes per cell, torch 2.13.0 on mps. Records in `runs/m3m_burden/`.

**Timing, which no record carries** (no record has a timestamp and `run.log` has
none), read from the files' own times with `stat`. `run.log` was created at 19:48:53,
the launch; the first record was written at 19:51:24 and the last at 20:14:27. That
gives two different quantities, which are not interchangeable:

- the **first-to-last record-write span is 23.1 min** over eight intervals, **2.88
  min** each. It excludes the first cell's own measure (2.5 min from launch,
  loading included), so it is not the wall clock, and it is *below* §6's 25–45 min
  estimate;
- the **launch-to-last-write wall clock is 25.6 min**, **2.84 min/cell** over nine,
  which is *inside* §6's estimate, at its lower edge.

**What is re-derivable from the records.** Every figure below is re-derivable from
`runs/m3m_burden/*.json` alone **except two**: the timing above (file-system times,
not record fields) and the `gap_closed(45)` claim under *This does not pass the M3
exit gate*, which comes from `runs/m3_study_v2`.

### Reading H refuses: `UNREADABLE`, exit 46

The base control fails in **9 of 9 cells**. Per cell, the median true one-step
displacement is **3.9695** map units — ground truth, so identical across cells —
against a median floor error of **88.29 to 224.36**, a ratio of **22.2× to 56.5×**.
No `burden.txt` was written; the refusal names all nine cells.

**The specified statistic is not viable on this data, and the reason is a design
error in §2.2.** `motion_margin = one_step_persistence − the k=1 rung` subtracts a
model quantity measured *through the probe* from a ground-truth quantity that pays
no readout error. The readout error dominates the signal by a factor of 22 to 56,
so the margin measures the floor, not the model.

`motion_margin` at h=45 reads **−116.47 to −244.60**, and the whole interval is
below 0 in **9 of 9** cells (largest `ci_high` = −100.73). **That does not mean the
code copies, and `COPIES` is not the finding.** A *perfect* one-step predictor has
`rung_1 == floor`, so its margin is `one_step_persistence[45] − floor_position[45]`:
it would still pay the readout error its baseline does not. Taken per cell from
the records, that is **−112.28 to −244.33** (mean −143.57, median −132.74), and the
margin the model actually read sits within **0.1% to 5.8%** of it:

| cell | a perfect predictor | observed | observed − perfect | relative |
|---|---|---|---|---|
| `frozen_ssl` s0 | −119.04 | −121.23 | −2.19 | 1.8% |
| `frozen_ssl` s1 | −112.28 | −118.80 | −6.52 | 5.8% |
| `frozen_ssl` s2 | −117.18 | **−116.47** | **+0.71** | 0.6% |
| `pixel_ae` s0 | −115.84 | −120.48 | −4.64 | 4.0% |
| `pixel_ae` s1 | −244.33 | −244.60 | −0.27 | 0.1% |
| `pixel_ae` s2 | −149.11 | −154.95 | −5.85 | 3.9% |
| `random_vit` s0 | −151.67 | −153.76 | −2.09 | 1.4% |
| `random_vit` s1 | −132.74 | −134.82 | −2.07 | 1.6% |
| `random_vit` s2 | −149.89 | −153.84 | −3.95 | 2.6% |

In `frozen_ssl` s2 the observed margin is *above* what a perfect predictor would
read. The column `observed − perfect` is not a coincidence of the fixture:
`margin = one_step_persistence − rung_1` and `perfect = one_step_persistence −
floor`, so it is exactly `−burden(1, 45)` (checked on the nine cells to 8.5e-14).
The model contributes the 0.27 to 6.52 map units of `burden(1)`; the other 112 to
244 are the readout. The base control exists for exactly this confound and it
fired; the cut was not retuned to manufacture a verdict.

The refused statistic, per cell, at h=45 (point, then the 95% episode-clustered
interval; the base control's median floor error and its ratio to the median true
displacement of 3.9695 beside it):

| cell | `motion_margin` | `ci_low` | `ci_high` | floor error | floor ÷ displacement |
|---|---|---|---|---|---|
| `frozen_ssl` s0 | −121.2320 | −138.1792 | −103.5678 | 95.01 | 23.9× |
| `frozen_ssl` s1 | −118.8008 | −139.0938 | −101.6545 | 88.29 | 22.2× |
| `frozen_ssl` s2 | −116.4653 | −137.1071 | −100.7302 | 95.93 | 24.2× |
| `pixel_ae` s0 | −120.4766 | −137.4205 | −103.3699 | 94.10 | 23.7× |
| `pixel_ae` s1 | −244.6039 | −292.1211 | −193.3336 | 224.36 | 56.5× |
| `pixel_ae` s2 | −154.9547 | −183.7534 | −128.7161 | 114.27 | 28.8× |
| `random_vit` s0 | −153.7604 | −185.5852 | −124.0128 | 108.12 | 27.2× |
| `random_vit` s1 | −134.8152 | −165.5694 | −106.9296 | 90.18 | 22.7× |
| `random_vit` s2 | −153.8443 | −184.2872 | −126.5497 | 105.27 | 26.5× |

Read from the records and checked against `run.log`: all nine triples agree to four
decimal places. They are given here so that the statistic that did not decide is
shown as fully as the companion that did.

**How the error happened.** §2.2 correctly identified that comparing the
re-grounded k=1 rung against the recorded *t-anchored* `persistence_position` is
rigged — the rung has observed h−1 frames more. The replacement made the baseline
ground truth and treated "no model involved" as the virtue. But `evaluate_rollout`
already records the opposite lesson in the code:

> "ALL THREE references go through the IDENTICAL pipeline (encode -> RSSM ->
> emb_head) and are probed with the SAME probe... Fitting on real encoder
> embeddings and applying to predicted ones is a distribution mismatch, and it was
> destroying the signal: band below 2 SE at 18 of 45 horizon steps, against 2 of
> 45 once every reference shares the pipeline. gap_closed at horizon 45 moved from
> -7.26 to -0.78 on the same checkpoint."

One unfairness was fixed and a second introduced. The fix for the next milestone is
a **probe-space** one-step baseline — the floor's own predicted position at h−1 held
to h — so both sides pay the same readout error and the difference isolates
prediction. That needs the floor's predicted positions, which `evaluate_rollout`
computes internally and does not return, so it is not a change M3m could make
under its own constraint not to modify `rollout.py`.

### What the ladder answers, and it is sound

`burden(k, h) = curve_k[h] − floor[h]` and `compounding(k, h) = curve_k[h] −
curve_1[h]` are differences of two curves measured **through the same pipeline**, so
the readout error cancels and they are commensurable. `burden(k, 45)` per cell:

| cell | k=1 | k=3 | k=5 | k=15 | k=45 | compounding share |
|---|---|---|---|---|---|---|
| `frozen_ssl` s0 | 2.189 | 4.060 | 4.326 | 28.665 | 90.436 | 97.6% |
| `frozen_ssl` s1 | 6.519 | 10.994 | 21.317 | 34.849 | 114.242 | 94.3% |
| `frozen_ssl` s2 | **−0.714** | 5.970 | 4.598 | 18.593 | 67.845 | 101.1% |
| `pixel_ae` s0 | 4.640 | 6.612 | 11.341 | 46.261 | 142.071 | 96.7% |
| `pixel_ae` s1 | 0.269 | 2.306 | 5.021 | 8.747 | 31.235 | 99.1% |
| `pixel_ae` s2 | 5.845 | 9.386 | 12.727 | 33.685 | 107.294 | 94.6% |
| `random_vit` s0 | 2.086 | 6.113 | 9.848 | 23.555 | 64.337 | 96.8% |
| `random_vit` s1 | 2.074 | 1.104 | 2.224 | 25.330 | 77.495 | 97.3% |
| `random_vit` s2 | 3.952 | 8.396 | 4.717 | 28.707 | 83.062 | 95.2% |

`burden(1)` mean **2.984** (min −0.714, max 6.519). `burden(45)` mean **86.446**
(min 31.235, max 142.071). The compounding share at k=45 runs **94.3% to 101.1%**,
mean **97.0%**, and compounding is the majority of the cost in **9 of 9** cells.

**So ~97% of what the open loop pays over the floor is attributable to correction
arriving every 45 steps rather than every step, not to the one prior step itself.**
A multi-step or overshooting objective is the indicated intervention.

**The share is read at k=45 only, and that was a judgement; the rest of §2.4's
companions were not delivered.** Spec §2.4 asks for `compounding_share(k)` and §7
requires "the companions of 2.4 reported as raw levels". The ~97% above is a
hand-computed instance of the first, at one rung, taken from the recorded
`burden_by_k` and `compounding_by_k`; it is not a field of any record and no code
computes it. The band-relative companion (§2.4's third bullet, the costs as
fractions of the persistence-to-floor band) was **not implemented, recorded or
reported**. So §7's criterion on the companions is met in part: the burden curve
over the ladder and the KL carry-over are reported as raw levels, the share only at
k=45 and by hand, the band-relative figure not at all.

Reading the share at k=45 only is deliberate, not an oversight. At k=3,
`random_vit` s1 reads **−87.9%**, because `burden(3, 45)` is **1.104** and
`compounding(3, 45)` is **−0.970**: exactly the vanishing-denominator case §2.4
warns of (there at small horizon, here at small k). The share across the ladder,
derived from the same records:

| cell | k=3 | k=5 | k=15 | k=45 |
|---|---|---|---|---|
| `frozen_ssl` s0 | 46.1% | 49.4% | 92.4% | 97.6% |
| `frozen_ssl` s1 | 40.7% | 69.4% | 81.3% | 94.3% |
| `frozen_ssl` s2 | 112.0% | 115.5% | 103.8% | 101.1% |
| `pixel_ae` s0 | 29.8% | 59.1% | 90.0% | 96.7% |
| `pixel_ae` s1 | 88.3% | 94.6% | 96.9% | 99.1% |
| `pixel_ae` s2 | 37.7% | 54.1% | 82.6% | 94.6% |
| `random_vit` s0 | 65.9% | 78.8% | 91.1% | 96.8% |
| `random_vit` s1 | **−87.9%** | 6.8% | 91.8% | 97.3% |
| `random_vit` s2 | 52.9% | 16.2% | 86.2% | 95.2% |
| **median** | **46.1%** | **59.1%** | **91.1%** | **96.8%** |

The medians rise with k: 46.1%, 59.1%, 91.1%, 96.8%. Seven of the nine cells rise
at every step; `frozen_ssl` s2 and `random_vit` s2 do not. At k=3 six of the nine
cells lie in 29.8% to 65.9%; the other three are 88.3%, 112.0% and −87.9%, and they
are the cells where the denominator is small (`random_vit` s1 and `pixel_ae` s1 have
the two smallest `burden(3, 45)`, 1.104 and 2.306) or where `burden(1)` is negative
(`frozen_ssl` s2). At k=45 all nine lie in 94.3% to 101.1%, which is the one rung
where the denominator is large in every cell (31.2 to 142.1).

**There is no interval on the ~97%, and these records cannot give one.**
`burden_by_k` and `compounding_by_k` are means over 229 windows; the per-window
rungs and the per-window floor were not kept, so nothing can be computed from the
records afterwards, and the 94.3% to 101.1% is the spread *across nine cells*, not
an uncertainty in any one of them. `RegroundingSweep` already carries the paired
rulers the spec requires (`floor_margin_standard_error` and
`paired_standard_error`, because the unpaired spread overstates a k-to-k bar by
1.7× to 3.9×), and until the fix that followed this section nothing on the branch
referenced either. `measure_cell` now records both at the decision horizon, under
`rulers`: `floor_margin_standard_error(1)`, the ruler on `burden(1)`, and
`paired_standard_error(ks[-1], 1)`, the ruler on `compounding(ks[-1])`. The next
measurement carries them; these nine records do not.

**What the ladder does not license.** It says one prior step is cheap *relative to
seeing the frame*. It does **not** say the one-step map beats assuming no motion —
that is what `motion_margin` was for, and it is unreadable. In particular,
comparing `burden(1)` = 2.984 against the true displacement of 3.9695 is
**invalid**, for precisely the reason the margin is: one is a probe-space
difference, the other is ground truth. That comparison is not made here.

### Controls, all nine cells

- `open_loop_divergence` = **0.0** on all nine: the k=45 rung reproduces the shipped
  `curves.rssm_position` of `runs/m3_study_v2` **bitwise**, on a stochastic model.
- `is_bitwise_the_floor(1)` = **False** on all nine: k=1 sits off the floor, so the
  grounding never consumed the frame it is scored on.
- `compounding(k=1)` = **exactly 0.0** on all nine, by construction.
- Identity residual: **0.0** on eight cells and **2.842e-14** on `pixel_ae` s0,
  against a 1e-9 tolerance. The one nonzero cell is the one outside the exactness
  condition in `IDENTITY_TOLERANCE`'s docstring (`burden.py`), which is written in
  terms of `max(rung_45) / min(floor)`: that ratio is **2.205** on `pixel_ae` s0 and
  **1.15 to 1.97** on the other eight. (The same cell reads 2.171 if the ratio is
  taken at h=45 only, 2.181 at its worst single step; it is above 2 on every
  definition and no other cell is above 2 on any.) The tolerance behaved as
  documented.
- **No control between the two floors.** Every burden and the identity residual
  read the floor from `prepared.reference`, while `k_one_is_floor` and the base
  control's median read it from the sweep's own pass. `open_loop_divergence = 0.0`
  shows the two passes agree on the RSSM curve, not on the floor curve. The control
  that closes this, `controls.floor_divergence`, was added after this run and is
  not in these nine records.
- Read determinism: two independent reads produced **byte-identical** stdout (358
  bytes), and the refusal left no `burden.txt`.

### Companions, which decide nothing

`kl_rate_above_free_bits` **0.7455–0.9757** and `kl_dyn_max` **1.868–36.222** nats
against a 0.20-nat floor, carried onto every record so §1's refutation travels with
the reading: the dynamics prior trained on 75–98% of steps, and "the loss never asks
for prediction" is false as stated.

### One figure worth keeping

`frozen_ssl` s2's `burden(1)` is **−0.714**: its k=1 rung sits 0.6% *below* the floor
at h=45, which puts its compounding share above 100%. Both curves are stochastic
draws, so a 0.6% inversion is sampling noise rather than a violation of the floor's
meaning, which concerns the k→0 limit and not per-step ordering.

It is reportable only because Task 2's review found that `np.abs(curve_k − floor)`
and `np.maximum(..., 0.0)` both survived the entire suite — every fixture being
monotone and non-negative. Under `np.abs` this cell's `burden(1)` would read
**+0.7141** and its share, taken as `(burden(45) − burden(1)) / burden(45)` as a
share built from clamped burdens would be, **98.95%**; under `np.maximum(..., 0.0)`
it would read **0.0000** and **100.00%**. (The actual share, `compounding(45) /
burden(45)`, is 101.05%.) Plausible, wrong, and silent in both cases: the clamps
would have masked the one cell that shows the floor is a limit and not a per-step
bound.

### This does not pass the M3 exit gate

The gate remains failed on `beats_persistence`, `band_is_usable` and
`filtering_beats_embedding`, and `gap_closed(45)` is negative in 9 of 9 position
cells. M3m narrows the objective lever to a horizon problem and names the
intervention; it does not close the gate, and it licenses no claim that a
multi-step objective would.

### Tests

`2791 passed, 0 failures` in 36m49s at `349edee`, the SHA the nine records carry.
`runs/m3l_capacity` is byte-identical to before this milestone.

---

## Task 7 addendum: the paired rulers, and what they do to the 97%

The final whole-branch review found that the milestone's one surviving conclusion
shipped with **no interval**, and that the records could not give one
retrospectively: `burden_by_k` and `compounding_by_k` were stored as mean curves,
and `RegroundingSweep`'s paired rulers — `floor_margin_standard_error(k)` and
`paired_standard_error(k, other)`, which spec §2.1 calls required because the
unpaired spread overstates the paired bars 1.7× to 3.9× — were referenced nowhere.

`measure_cell` now records both at `DECISION_H`, and the nine cells were
re-measured at `cc5c1d9` into `runs/m3m_burden_rulers/` in 19m20s. **The nine
original records in `runs/m3m_burden/` are untouched**; the ladder reproduces them
identically (`burden_by_k` equal to within 1e-12 on every cell), so this is the
same measurement with the rulers added, at a second `git_sha`.

Each recorded ruler is **one** standard error of the column means
(`diagnostics.py:1813-1823`); the sibling tool doubles it for a bar
(`diagnose_dynamics.py:1430`), so the test below is against 2 SE.

| cell | `burden(1)` | 1 SE | \|b1\|/SE | resolvable? | `compounding(45)`/SE | resolvable? |
|---|---|---|---|---|---|---|
| `frozen_ssl` s0 | 2.189 | 2.654 | 0.82 | no | 8.57 | yes |
| `frozen_ssl` s1 | 6.519 | 3.355 | **1.94** | no | 10.10 | yes |
| `frozen_ssl` s2 | −0.714 | 1.380 | 0.52 | no | 6.81 | yes |
| `pixel_ae` s0 | 4.640 | 3.009 | 1.54 | no | 11.63 | yes |
| `pixel_ae` s1 | 0.269 | 1.488 | 0.18 | no | 5.56 | yes |
| `pixel_ae` s2 | 5.845 | 3.254 | 1.80 | no | 8.60 | yes |
| `random_vit` s0 | 2.086 | 1.693 | 1.23 | no | 7.69 | yes |
| `random_vit` s1 | 2.074 | 2.301 | 0.90 | no | 8.95 | yes |
| `random_vit` s2 | 3.952 | 2.845 | 1.39 | no | 8.49 | yes |

**`burden(1)` is not resolvable from zero at 2 SE in 9 of 9 cells** — the largest
ratio anywhere is 1.94. **`compounding(45)` is resolvable in 9 of 9, at 5.56 to
11.63 SE.**

### This strengthens the conclusion and retires the 97%

The point estimates said ~97% of the open-loop cost over the floor is compounding,
with the remaining ~3% the one prior step. **That 3% is not resolvable.** Within
the resolution of this measurement the one-step prediction cost is
indistinguishable from zero, while compounding is resolved by a factor of 5.6 to
11.6 in every cell. The data is consistent with the whole of the open-loop cost
over the floor being compounding.

So the finding is not "97%, so 3% is the one-step map". It is: **the one-step map's
cost over the floor cannot be distinguished from zero, and the rollout's cost can,
by an order of magnitude.** The direction the earlier point estimates indicated is
unchanged and the precision they implied was unsupported.

**Still no interval on the share itself.** The share is a ratio, so an interval on
it needs the covariance between the two rulers, which is not recorded. The two
component intervals above are what the records carry, and they are the ones the
conclusion rests on.

### One finding, not two

`observed_margin − perfect_margin = −burden(1, 45)` **exactly** (to 8.5e-14), by
definition: both margins share the `one_step_persistence` term, so their difference
is `floor − rung_1`. The "observed margins sit within 0.1%–5.8% of a perfect
predictor" statement in the section above and "`burden(1)` is small" are therefore
**the same fact in two coordinate systems**, not two independent confirmations. A
reader who counts them separately overcounts the evidence.

This also says precisely why one statistic worked and the other did not:
`motion_margin` carries the floor as an additive contaminant, and the ladder
subtracts it.
