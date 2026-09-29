# MB-FPS M3k — Width-Matched Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether `two_frame`'s k = 15 advantage over `deterministic` is information or feature count, so the project can pick between the bottleneck lever and the objective lever before spending ~13.5 hours on one of them.

**Architecture:** Three passes of M3j's ladder over one gather per cell — `shipped` (native widths), `down` (every block projected to 512, matching count **and** rank) and `up` (every block lifted to 2048, matching count only). `down` decides; `up` calibrates the width bias. Because `projection` returns the identity when `native == target`, each pass carries a free known-answer anchor: `deterministic` is untouched in `down` and `two_frame` is untouched in `up`, so each must reproduce its shipped gain exactly.

**Tech Stack:** Python 3.12, numpy, torch (MPS), pytest. No new dependencies.

## Global Constraints

Copied verbatim from `docs/superpowers/specs/2026-09-29-mb-fps-m3k-width-matched-ladder-design.md`. Every task's requirements implicitly include this section.

- **Evaluation only.** No training, no checkpoint written or altered, nothing under `runs/` removed.
- **`gain_from_blocks`'s output stays byte-identical.** It reports onto the research gate's path via `src/mbfps/eval/study.py` and `scripts/eval_rollout.py`, and its exact ten-key output is pinned by `test_gain_from_splits_output_is_byte_identical_after_the_generalisation`. **That pin is the test that proves the `_fit_and_score` extraction was safe — never re-record its values.**
- **`retention.Z_BEARING_RUNGS` is NOT edited.** M3j's records must keep reading under the rule they were taken under. The correction arrives as a default-valued `z_bearing` keyword.
- **`PASSES = ("shipped", "down", "up")`, `DOWN_WIDTH = 512`, `UP_WIDTH = 2048`, `CONTRAST_K = 15`.**
- **`CONTRAST_K`, never `DECISION_K`.** `retention.DECISION_K` is **4**; both modules are imported by the same script, and two names with one meaning and two values is a misreading waiting to happen.
- **`ANCHOR = {"down": "deterministic", "up": "two_frame"}`** — the rung untouched in each pass.
- **`projection(native, target)` returns the IDENTITY when `native == target`**, so the anchors are exact, not approximate.
- **One fixed Gaussian per `(native, target)`, scaled `1/sqrt(target)`, drawn once from `PROJECTION_SEED` and shared across all nine cells.** No cell gets its own draw.
- **The base `enc(t)` is untouched in every pass.** Matching applies to the rung's block only.
- **Reading F decides at k = 15 only**, not by disjunction, and is **two-sided**.
- **`SEEDS_REQUIRED = 2`, `ARMS_REQUIRED = 2`** — imported from `retention`, not re-spelled.
- **`BASE_R2_FLOOR` stays 0.10 and is not re-chosen**, gated on **position alone**.
- **Five statuses, this precedence:** `UNRESOLVED_BASE`, `UNRESOLVED_ANCHOR`, `PAST_FRAME_AHEAD`, `RECURRENT_AHEAD`, `INDISTINGUISHABLE`.
- **Only the `down` anchor gates Reading F.** An `up` anchor failure suppresses the width-bias companion and is reported; it does not suppress the reading.
- **Exit codes:** 41 `EXIT_BASE_UNRESOLVED`, 42 `EXIT_ANCHOR_BROKEN`. M3i holds 38, M3j holds 39–40.
- **Test command:** `.venv/bin/python -m pytest` from the repo root. There is no `pytest` entry point in the venv.
- **Repo conventions:** never `git stash` (shared stack across worktrees); never `rm` under `runs/`; commit trailer `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` in its own paragraph; write the message to a file and use `git commit -F` if it contains an apostrophe; after mutating a source file to check a test bites, clear `src/` and `tests/` `__pycache__`; never `git clean -fdx`.

## File Structure

| file | responsibility |
|---|---|
| `src/mbfps/eval/probe.py` | **modify.** Extract `_fit_and_score`; add `contrast_from_blocks`. |
| `src/mbfps/eval/retention.py` | **modify.** `reading_retention` gains `z_bearing`; the constant is untouched. |
| `src/mbfps/eval/width.py` | **create.** Pure: the pass constants, the projections, Reading F, the tables. No torch, no I/O, no record schema. |
| `scripts/latent_width.py` | **create.** Owns torch, I/O and the record schema: one gather per cell, three passes, `width.txt`. |
| `tests/eval/test_probe.py` | **modify.** The contrast; the golden pin re-proving the extraction. |
| `tests/eval/test_retention.py` | **modify.** `z_bearing` defaults to the module constant. |
| `tests/eval/test_width.py` | **create.** |
| `tests/eval/test_latent_width_script.py` | **create.** |
| `tests/eval/test_diagnose_dynamics_script.py` | **modify.** `latent_width` (41, 42) in the exit-status registry. |

`width.py` holds the statistics and `latent_width.py` everything impure, exactly as M3j split `retention.py` from `scripts/latent_retention.py`. If a function needs a `Path` or a `torch.device`, it belongs in the script.

---

### Task 1: `contrast_from_blocks`, and the extraction that makes it DRY

**Why:** Reading F compares two rungs. Because both share the identical base on the identical rows, the difference of *gains* is exactly the difference of *joint* R² — the base cancels outright. `_block_bootstrap_ci` already computes `r2(A) − r2(B)` per resample, so the contrast needs no new statistic, only a function that fits two joints and hands it their scored predictions.

**Files:**
- Modify: `src/mbfps/eval/probe.py`
- Test: `tests/eval/test_probe.py`

**Interfaces:**
- Consumes: `GainSplit(base, block, target)` with `.joint()` and `.rows()`; `fit_probe`, `apply_probe`, `_mean_r2`, `_block_bootstrap_ci(..., groups=...)` — all already on `main`.
- Produces, used by Task 5:

```python
def _fit_and_score(fit_x, fit_y, select_x, select_y, score_x, score_y):
    """(predicted, r2, ridge) for one arm. `select_x is None` -> default penalty."""

def contrast_from_blocks(a, b, *, groups, resamples=1000, confidence=0.95, seed=0) -> dict: ...
```

`a` and `b` are each a `(fit, select, score)` triple of `GainSplit`s. Returns exactly:
`"contrast"`, `"a_r2"`, `"b_r2"`, `"ci_low"`, `"ci_high"`, `"confidence"`, `"n_scored_windows"`, `"a_ridge"`, `"b_ridge"`, `"ridge_selected"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_probe.py`, in the `filtering_gain` region:

```python
from mbfps.eval.probe import contrast_from_blocks  # noqa: E402


def _triple(latent, embedding, targets, rows, h_dim=2, block=None):
    """A `(fit, select, score)` triple of `GainSplit`s over `rows`."""
    chosen = latent[rows, :h_dim] if block is None else block[rows]
    return GainSplit(base=embedding[rows], block=chosen, target=targets[rows])


def test_contrast_is_exactly_zero_when_both_arms_are_the_same_block():
    """The known-answer case. `contrast` is a difference of two joint R^2 values
    fit on the same rows from the same splits, so handing it the same block
    twice must give exactly 0.0 -- not approximately. `fit_probe` is
    deterministic, so any nonzero here means the two arms were not fit on the
    same thing, which is the defect this function is most exposed to."""
    latent, embedding, targets = _history_case(600, seed=0)
    rows = np.arange(600)
    sl = (rows[:200], rows[200:400], rows[400:600])
    same = tuple(_triple(latent, embedding, targets, s) for s in sl)
    out = contrast_from_blocks(same, same, groups=np.arange(200) // 5,
                               resamples=50, confidence=0.9, seed=1)
    assert out["contrast"] == 0.0
    assert out["a_r2"] == out["b_r2"]
    assert out["ci_low"] == 0.0 and out["ci_high"] == 0.0


def test_contrast_equals_the_difference_of_the_two_gains():
    """The base cancels: contrast(A, B) must equal gain(A) - gain(B) computed on
    the same splits with the same base. If it does not, the function is not
    measuring what Reading F claims it measures."""
    latent, embedding, targets = _history_case(600, seed=2)
    rows = np.arange(600)
    sl = (rows[:200], rows[200:400], rows[400:600])
    noise = np.random.default_rng(3).normal(size=(600, 2))
    a = tuple(_triple(latent, embedding, targets, s) for s in sl)
    b = tuple(_triple(latent, embedding, targets, s, block=noise) for s in sl)
    kw = dict(groups=np.arange(200) // 5, resamples=50, confidence=0.9, seed=1)
    out = contrast_from_blocks(a, b, **kw)
    gain_a = gain_from_blocks(a[0], a[1], a[2], **kw)
    gain_b = gain_from_blocks(b[0], b[1], b[2], **kw)
    assert out["contrast"] == pytest.approx(gain_a["gain"] - gain_b["gain"], abs=1e-12)
    assert out["a_r2"] == pytest.approx(gain_a["joint_r2"], abs=1e-12)
    assert out["b_r2"] == pytest.approx(gain_b["joint_r2"], abs=1e-12)


def test_contrast_is_signed_and_antisymmetric():
    """Reading F is TWO-sided, so the sign carries meaning and swapping the
    arms must negate it. A function that reported |contrast| would read
    RECURRENT_AHEAD as PAST_FRAME_AHEAD."""
    latent, embedding, targets = _history_case(600, seed=4)
    rows = np.arange(600)
    sl = (rows[:200], rows[200:400], rows[400:600])
    noise = np.random.default_rng(5).normal(size=(600, 2))
    a = tuple(_triple(latent, embedding, targets, s) for s in sl)
    b = tuple(_triple(latent, embedding, targets, s, block=noise) for s in sl)
    kw = dict(groups=np.arange(200) // 5, resamples=50, confidence=0.9, seed=1)
    forward = contrast_from_blocks(a, b, **kw)
    backward = contrast_from_blocks(b, a, **kw)
    assert forward["contrast"] > 0, "the lagged block must beat pure noise"
    assert backward["contrast"] == pytest.approx(-forward["contrast"], abs=1e-12)
    assert backward["ci_low"] == pytest.approx(-forward["ci_high"], abs=1e-9)


def test_contrast_rejects_arms_scored_on_different_rows():
    """Two arms scored on different rows is not a contrast, it is two unrelated
    numbers subtracted. The base cancels only if the rows are identical."""
    latent, embedding, targets = _history_case(400, seed=6)
    rows = np.arange(400)
    a = tuple(_triple(latent, embedding, targets, s)
              for s in (rows[:100], rows[100:200], rows[200:300]))
    b = tuple(_triple(latent, embedding, targets, s)
              for s in (rows[:100], rows[100:200], rows[300:400]))
    with pytest.raises(ValueError, match="same rows"):
        contrast_from_blocks(a, b, groups=np.arange(100) // 5, resamples=10)


def test_contrast_rejects_arms_with_different_bases():
    """The whole statistic rests on the base cancelling. Two arms with
    different bases would leave a base term in the difference, and the result
    would silently stop being a contrast between the blocks."""
    latent, embedding, targets = _history_case(400, seed=7)
    rows = np.arange(400)
    other = embedding + 1.0
    a = tuple(_triple(latent, embedding, targets, s)
              for s in (rows[:100], rows[100:200], rows[200:300]))
    b = tuple(GainSplit(base=other[s], block=latent[s, :2], target=targets[s])
              for s in (rows[:100], rows[100:200], rows[200:300]))
    with pytest.raises(ValueError, match="same base"):
        contrast_from_blocks(a, b, groups=np.arange(100) // 5, resamples=10)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -k contrast -v`
Expected: FAIL, `ImportError: cannot import name 'contrast_from_blocks'`.

- [ ] **Step 3: Extract `_fit_and_score`**

In `src/mbfps/eval/probe.py`, insert above `gain_from_blocks`:

```python
def _fit_and_score(fit_x, fit_y, select_x, select_y, score_x, score_y):
    """Fit one arm on `fit`, select its ridge on `select`, score it on `score`.

    The three-split discipline for a single feature set, factored out because
    `gain_from_blocks` needs it twice (joint and base) and
    `contrast_from_blocks` needs it twice again (two joints). `select_x is
    None` takes `fit_probe`'s default penalty, which is unbiased too, just
    weaker.

    Returns `(predicted, r2, ridge)` on the scored rows.
    """
    probe = (
        fit_probe(fit_x, fit_y)
        if select_x is None
        else fit_probe(fit_x, fit_y, select_x, select_y)
    )
    predicted = apply_probe(probe, score_x)
    return predicted, _mean_r2(predicted, score_y), probe["ridge"]
```

Then replace `gain_from_blocks`' fitting block — everything from `if select is None:` through the two `_mean_r2` lines — with:

```python
    select_target = (
        None if select is None else np.asarray(select.target, dtype=np.float64)
    )
    if select is not None:
        select.rows()
    joint_predicted, joint_r2, joint_ridge = _fit_and_score(
        joint_fit, target_fit,
        None if select is None else select.joint(), select_target,
        joint_score, target_score,
    )
    base_predicted, base_r2, base_ridge = _fit_and_score(
        base_fit, target_fit,
        None if select is None else np.asarray(select.base, dtype=np.float64),
        select_target,
        base_score, target_score,
    )
```

and change the return's last two entries to `"joint_ridge": joint_ridge, "embedding_ridge": base_ridge`.

- [ ] **Step 4: Prove the extraction was safe**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -k byte_identical -v`
Expected: **PASS**, with the same ten values it has always had. This is the whole point of that pin — do not touch its expected values. If it fails, the extraction changed behaviour and is wrong.

- [ ] **Step 5: Add `contrast_from_blocks`**

```python
def contrast_from_blocks(
    a, b, *, groups: np.ndarray, resamples: int = 1000,
    confidence: float = 0.95, seed: int = 0,
) -> dict:
    """`R2([base (+) A]) - R2([base (+) B])` -- the base cancels outright.

    `a` and `b` are each a `(fit, select, score)` triple of `GainSplit`s. Both
    arms MUST carry the same base on the same rows: that is what makes this a
    contrast between the two BLOCKS rather than two unrelated levels
    subtracted, and it is why the difference of gains equals the difference of
    joint R^2 with no base term surviving.

    `_block_bootstrap_ci` already computes `r2(A) - r2(B)` per resample, so the
    interval needs no new statistic -- only the two arms' scored predictions.

    SIGNED, and the sign is the reading. Reading F is two-sided: a negative
    contrast means B's block beats A's, which is a finding about B rather than
    an absence. Swapping the arms negates the result, pinned by test.
    """
    a_fit, a_select, a_score = a
    b_fit, b_select, b_score = b
    n = a_score.rows()
    if b_score.rows() != n:
        raise ValueError(
            f"both arms must be scored on the same rows; got {n} and "
            f"{b_score.rows()}"
        )
    a_base = np.asarray(a_score.base, dtype=np.float64)
    b_base = np.asarray(b_score.base, dtype=np.float64)
    if a_base.shape != b_base.shape or not np.array_equal(a_base, b_base):
        raise ValueError(
            "both arms must carry the same base; the contrast is only a "
            "comparison of the two BLOCKS because the base cancels"
        )
    groups_arr = np.asarray(groups)
    if groups_arr.shape != (n,):
        raise ValueError(
            f"groups must be one label per scored row; got {groups_arr.shape} "
            f"for {n} rows"
        )
    n_groups = int(np.unique(groups_arr).size)
    if n_groups < 2:
        raise ValueError(
            "a bootstrap interval needs at least two resampling units (distinct "
            f"`groups` labels); got {n_groups}"
        )
    target_score = np.asarray(a_score.target, dtype=np.float64)

    def arm(fit, select, score):
        select_target = (
            None if select is None else np.asarray(select.target, dtype=np.float64)
        )
        return _fit_and_score(
            fit.joint(), np.asarray(fit.target, dtype=np.float64),
            None if select is None else select.joint(), select_target,
            score.joint(), target_score,
        )

    a_predicted, a_r2, a_ridge = arm(a_fit, a_select, a_score)
    b_predicted, b_r2, b_ridge = arm(b_fit, b_select, b_score)
    low, high = _block_bootstrap_ci(
        a_predicted, b_predicted, target_score,
        groups=groups_arr, resamples=resamples, confidence=confidence, seed=seed,
    )
    return {
        "contrast": a_r2 - b_r2,
        "a_r2": a_r2,
        "b_r2": b_r2,
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
        "n_scored_windows": n_groups,
        "a_ridge": a_ridge,
        "b_ridge": b_ridge,
        "ridge_selected": a_select is not None,
    }
```

- [ ] **Step 6: Run the three suites that cover every `filtering_gain` caller**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py tests/eval/test_study.py tests/eval/test_eval_rollout_script.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/mbfps/eval/probe.py tests/eval/test_probe.py
git commit -m "feat: contrast_from_blocks, and the _fit_and_score extraction the golden pin proves safe"
```

---

### Task 2: `reading_retention` gains a `z_bearing` keyword

**Why:** M3k's companion re-runs Reading E under the corrected `("stochastic",)`, because `full = h ⊕ z` cannot attribute a clearance to `z`. **Editing the module constant would silently re-read M3j's records under a rule they were never taken under** — and M3j's own results are emphatic that a pre-registered rule is not rewritten after the numbers are seen. A default-valued parameter keeps both readings alive at once, each reproducible from its own artefacts.

**Files:**
- Modify: `src/mbfps/eval/retention.py`
- Test: `tests/eval/test_retention.py`

**Interfaces:**
- Produces, used by Task 6: `reading_retention(inputs, *, z_bearing=Z_BEARING_RUNGS) -> RetentionStatus`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_retention.py`:

```python
def test_reading_retention_defaults_to_the_module_z_bearing_set():
    """M3j's records must keep reading exactly as recorded. The default is the
    module constant, so every existing caller is unaffected by the keyword
    existing at all."""
    import inspect
    from mbfps.eval.retention import Z_BEARING_RUNGS as shipped
    sig = inspect.signature(reading_retention)
    assert sig.parameters["z_bearing"].default == shipped
    assert sig.parameters["z_bearing"].kind is inspect.Parameter.KEYWORD_ONLY
    assert shipped == ("stochastic", "full"), (
        "the module constant must NOT be edited -- M3k corrects it by passing "
        "the keyword, not by changing what M3j's records read under"
    )


def test_reading_retention_honours_a_corrected_z_bearing_set():
    """M3j's own results recorded that `full` is h + z and so cannot attribute a
    clearance to z, which made MOTION_RETAINED reachable on h alone. With only
    `stochastic` z-bearing, a state where `full` and `deterministic` clear but
    `stochastic` does not must read BOTTLENECK_LOSS, not MOTION_RETAINED."""
    inputs = _inputs({("translation", 4, "full"): 0.2,
                      ("translation", 4, "deterministic"): 0.2})
    assert reading_retention(inputs).status == "MOTION_RETAINED"
    corrected = reading_retention(inputs, z_bearing=("stochastic",))
    assert corrected.status == "BOTTLENECK_LOSS"
    assert corrected.surviving == "deterministic"


def test_reading_retention_rejects_a_z_bearing_rung_that_is_not_a_rung():
    """A typo in the corrected tuple would silently make nothing z-bearing and
    read BOTTLENECK_LOSS on every input."""
    inputs = _inputs({("translation", 4, "stochastic"): 0.2})
    with pytest.raises(ValueError, match="not a rung"):
        reading_retention(inputs, z_bearing=("stocastic",))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_retention.py -k z_bearing -v`
Expected: FAIL, `TypeError: reading_retention() got an unexpected keyword argument 'z_bearing'`.

- [ ] **Step 3: Add the keyword**

Change `reading_retention`'s signature to:

```python
def reading_retention(
    inputs: RetentionInputs, *, z_bearing: tuple[str, ...] = Z_BEARING_RUNGS,
) -> RetentionStatus:
```

Append to its docstring:

```
    `z_bearing` defaults to the module's `Z_BEARING_RUNGS` so every existing
    caller and every M3j record reads exactly as recorded. M3k passes
    `("stochastic",)` instead, because `full` is `h (+) z` and a clearing
    `full` cannot attribute anything to `z` -- which made MOTION_RETAINED
    reachable on `h` alone, as M3j's own results recorded. The CONSTANT IS NOT
    EDITED: a pre-registered rule is not rewritten after the numbers are seen,
    and a default-valued parameter keeps both readings alive at once. Do not
    "simplify" this by hardcoding the corrected tuple.
```

Add the guard at the top of the body, before `base_failed` is computed:

```python
    unknown = [rung for rung in z_bearing if rung not in RUNGS]
    if unknown:
        raise ValueError(
            f"z_bearing names {unknown}, which is not a rung; expected a subset "
            f"of {RUNGS}"
        )
```

and change the loop `for rung in Z_BEARING_RUNGS:` to `for rung in z_bearing:`.

- [ ] **Step 4: Run the retention suite**

Run: `.venv/bin/python -m pytest tests/eval/test_retention.py -q`
Expected: PASS. Every pre-existing test is unaffected, because the default is the constant.

- [ ] **Step 5: Prove M3j's records still read as recorded**

```bash
.venv/bin/python -c "
import json, glob, importlib.util
spec = importlib.util.spec_from_file_location('lr', 'scripts/latent_retention.py')
lr = importlib.util.module_from_spec(spec); spec.loader.exec_module(lr)
from mbfps.eval.retention import reading_retention
recs = {}
for f in glob.glob('runs/m3j_retention/retention_*.json'):
    d = json.load(open(f)); recs[(d['arm'], d['seed'])] = d
r = reading_retention(lr.retention_inputs(recs))
print('M3j records read as:', r.status, '| surviving:', r.surviving)
assert r.status == 'MOTION_RETAINED', r.status
print('unchanged.')
"
```

Expected: `M3j records read as: MOTION_RETAINED | surviving: stochastic` then `unchanged.`

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/retention.py tests/eval/test_retention.py
git commit -m "feat: reading_retention takes a z_bearing set, so M3k can correct M3j without rewriting what its records read under"
```

---

### Task 3: `width.py` — the pass constants, the projections, and the anchors

**Why:** the projections are what make the three passes comparable, and `projection` returning the identity when `native == target` is what makes the anchors exact rather than approximate. Everything here is pure: arrays in, arrays out.

**Files:**
- Create: `src/mbfps/eval/width.py`
- Test: `tests/eval/test_width.py`

**Interfaces:**
- Consumes: `mbfps.models.rssm.RSSMConfig` and `LATENT_DIM` for the widths; `mbfps.eval.retention.RUNGS`.
- Produces, used by Tasks 4, 5, 6:

```python
PASSES: tuple[str, ...] = ("shipped", "down", "up")
DOWN_WIDTH: int = 512
UP_WIDTH: int = 2048
CONTRAST_K: int = 15
PROJECTION_SEED: int = 0
RUNG_WIDTH: dict[str, int]          # derived from RSSMConfig, never re-spelled
TARGET_WIDTH: dict[str, int | None] # pass -> target, None for `shipped`
ANCHOR: dict[str, str]              # pass -> the rung it leaves untouched

def projection(native: int, target: int, *, seed: int = PROJECTION_SEED) -> np.ndarray | None: ...
def pass_block(block: np.ndarray, pass_name: str, *, seed: int = PROJECTION_SEED) -> np.ndarray: ...
```

`projection` returns **`None` for the identity** rather than an identity matrix, so the anchor path does no matmul at all and cannot drift by floating point.

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_width.py`:

```python
"""M3k: is the past frame's advantage information, or feature count?"""

import numpy as np
import pytest

from mbfps.eval.retention import RUNGS
from mbfps.eval.width import (
    ANCHOR, CONTRAST_K, DOWN_WIDTH, PASSES, PROJECTION_SEED, RUNG_WIDTH,
    TARGET_WIDTH, UP_WIDTH, pass_block, projection,
)


def test_constants_are_the_pre_registered_values():
    assert PASSES == ("shipped", "down", "up")
    assert DOWN_WIDTH == 512 and UP_WIDTH == 2048
    assert CONTRAST_K == 15
    assert TARGET_WIDTH == {"shipped": None, "down": 512, "up": 2048}


def test_contrast_k_is_not_named_decision_k():
    """`retention.DECISION_K` is 4 and means the horizon Reading E decides at;
    this module's is 15. Both are imported by the same script, so one name with
    two meanings is a misreading waiting to happen. Pinned so a later 'tidy-up'
    cannot unify them."""
    import mbfps.eval.width as width
    from mbfps.eval.retention import DECISION_K
    assert DECISION_K == 4 and CONTRAST_K == 15
    assert not hasattr(width, "DECISION_K")


def test_rung_widths_are_derived_from_the_rssm_config_not_re_spelled():
    """A hardcoded 512 here would silently disagree with the model if
    `RSSMConfig` ever changed, and every pass would be matching the wrong
    width."""
    from mbfps.models.rssm import LATENT_DIM, RSSMConfig
    z = RSSMConfig.z_cats * RSSMConfig.z_classes
    assert RUNG_WIDTH == {
        "two_frame": RSSMConfig.embed_dim,
        "deterministic": RSSMConfig.h_dim,
        "stochastic": z,
        "full": RSSMConfig.h_dim + z,
    }
    assert RUNG_WIDTH["full"] == LATENT_DIM
    assert set(RUNG_WIDTH) == set(RUNGS)


def test_each_anchor_is_the_rung_its_pass_leaves_untouched():
    """THE structural property the anchors rest on: a pass's anchor must be the
    rung whose native width already equals that pass's target, so the pass does
    not touch it and its gain must reproduce the shipped one exactly. Derived
    here rather than trusted, so ANCHOR cannot drift away from the widths."""
    for pass_name, rung in ANCHOR.items():
        target = TARGET_WIDTH[pass_name]
        assert RUNG_WIDTH[rung] == target, (
            f"{pass_name}'s anchor {rung!r} is {RUNG_WIDTH[rung]} wide but the "
            f"pass targets {target}; it would be projected, not untouched"
        )
        untouched = [r for r in RUNGS if RUNG_WIDTH[r] == target]
        assert untouched == [rung], f"{pass_name}: expected exactly one anchor"
    assert set(ANCHOR) == {"down", "up"}, "`shipped` projects nothing, so it has no anchor"


def test_projection_is_the_identity_when_no_projection_is_needed():
    """None, not an identity matrix: the anchor path must do no matmul at all,
    so it cannot drift by a floating-point ulp and the anchor test can demand
    exact equality."""
    assert projection(512, 512) is None
    assert projection(2048, 2048) is None
    assert projection(2048, 512) is not None


def test_projection_has_the_johnson_lindenstrauss_shape_and_scale():
    p = projection(2048, 512)
    assert p.shape == (2048, 512)
    # Gaussian / sqrt(target): column norms ~1, so inner products survive.
    assert np.allclose(p.std(), 1.0 / np.sqrt(512), rtol=0.05)


def test_projection_is_fixed_across_calls_and_shared_across_cells():
    """One matrix per (native, target), drawn once. If it were redrawn per call
    no two cells would be comparable, and an arm could win on a lucky draw."""
    a, b = projection(1024, 512), projection(1024, 512)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(projection(1024, 512), projection(2048, 512)[:1024])


def test_pass_block_leaves_the_shipped_pass_untouched():
    block = np.arange(40, dtype=np.float64).reshape(10, 4)
    np.testing.assert_array_equal(pass_block(block, "shipped"), block)


def test_the_down_pass_matches_count_and_rank():
    """Projected to 512 every block has 512 columns AND rank 512, so `down` is
    the pass that can ask who wins at equal capacity."""
    rng = np.random.default_rng(0)
    for native in (2048, 1536, 1024):
        block = rng.normal(size=(3000, native))
        out = pass_block(block, "down")
        assert out.shape == (3000, DOWN_WIDTH)
        assert np.linalg.matrix_rank(out) == DOWN_WIDTH
    already = rng.normal(size=(3000, 512))
    np.testing.assert_array_equal(pass_block(already, "down"), already)


def test_the_up_pass_matches_count_but_preserves_rank():
    """THE property that makes `up` a pure width control: lifting adds columns
    and no information, so the rank must stay at the block's native width. If
    the rank rose, the lift would be adding capacity and the calibration would
    measure the wrong thing."""
    rng = np.random.default_rng(1)
    block = rng.normal(size=(3000, 512))
    out = pass_block(block, "up")
    assert out.shape == (3000, UP_WIDTH)
    assert np.linalg.matrix_rank(out) == 512, (
        "the lift changed the rank; it is meant to add columns, not information"
    )
    already = rng.normal(size=(3000, 2048))
    np.testing.assert_array_equal(pass_block(already, "up"), already)


def test_pass_block_rejects_an_unknown_pass():
    with pytest.raises(ValueError, match="unknown pass"):
        pass_block(np.zeros((4, 8)), "sideways")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_width.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'mbfps.eval.width'`.

- [ ] **Step 3: Write `width.py`**

```python
"""M3k: is the past frame's advantage over the recurrent state information, or
feature count?

M3j left two candidate levers and one confounded comparison. Its objective-lever
argument rests on `two_frame` (+0.03489) beating `deterministic` (+0.02071) at
k = 15 in 8 of 9 cells -- but those blocks are 2048 features against 512.
Measured during M3k's design, a random lift of `h` from 512 to 2048 columns,
adding ZERO information and leaving the rank at 512, buys +0.01355 and +0.01377
of gain on two cells. That is larger than the +0.01418 pooled gap the argument
depends on.

Three passes of one ladder settle it. `down` projects every block to 512,
matching count AND rank, and decides. `up` lifts every block to 2048, matching
count only, and calibrates how much count alone was worth. Neither suffices
alone: a random projection DESTROYS information, so a rung losing in `down`
might have lost to the projection; the lift preserves information exactly but
leaves `enc(t-k)`'s richer rank intact.

Everything here is pure: arrays in, arrays and readings out. No torch, no I/O,
no record schema. `scripts/latent_width.py` owns all three.
"""

from __future__ import annotations

import numpy as np

from mbfps.eval.retention import RUNGS
from mbfps.models.rssm import RSSMConfig

PASSES: tuple[str, ...] = ("shipped", "down", "up")
DOWN_WIDTH: int = 512
UP_WIDTH: int = 2048

TARGET_WIDTH: dict[str, int | None] = {
    "shipped": None, "down": DOWN_WIDTH, "up": UP_WIDTH,
}

CONTRAST_K: int = 15
"""The horizon Reading F decides at.

DELIBERATELY NOT NAMED `DECISION_K`, which is 4 in `retention` and means the
horizon Reading E decides at. Both modules are imported by the same script, and
one name carrying two meanings is a misreading waiting to happen. 15 is where
M3j's objective-lever claim lives and nowhere else -- `two_frame` loses at k = 1
and k = 4 even before matching -- so testing the claim at its own horizon is the
honest test, and it avoids the incoherence a disjunction would create
(PAST_FRAME_AHEAD at one horizon and RECURRENT_AHEAD at another are
contradictory, not complementary).
"""

PROJECTION_SEED: int = 0

_Z_DIM = RSSMConfig.z_cats * RSSMConfig.z_classes
RUNG_WIDTH: dict[str, int] = {
    "two_frame": RSSMConfig.embed_dim,
    "deterministic": RSSMConfig.h_dim,
    "stochastic": _Z_DIM,
    "full": RSSMConfig.h_dim + _Z_DIM,
}
"""Each rung's native block width, DERIVED from the model's own config rather
than re-spelled. A hardcoded 512 here would silently disagree with the model if
`RSSMConfig` ever changed, and every pass would match the wrong width."""

ANCHOR: dict[str, str] = {"down": "deterministic", "up": "two_frame"}
"""Each pass's known-answer rung: the one whose native width already equals the
pass's target, so the pass leaves it untouched and its gain must reproduce the
shipped one EXACTLY. `shipped` projects nothing and so has no anchor. Pinned
against `RUNG_WIDTH` by test, so this cannot drift away from the widths."""


def projection(native: int, target: int, *, seed: int = PROJECTION_SEED):
    """The fixed matrix taking a `native`-wide block to `target`, or `None`.

    `None` means no projection is needed -- NOT an identity matrix. The anchor
    path must do no matmul at all, so it cannot drift by a floating-point ulp
    and the anchor test can demand exact equality rather than approximate.

    Gaussian scaled `1/sqrt(target)`, the standard Johnson-Lindenstrauss form,
    which approximately preserves inner products. Drawn from a fixed seed keyed
    on `(native, target)` and therefore SHARED ACROSS ALL NINE CELLS: a matrix
    redrawn per call would make no two cells comparable and would let an arm win
    on a lucky draw.
    """
    if native == target:
        return None
    rng = np.random.default_rng([seed, native, target])
    return rng.normal(size=(native, target)) / np.sqrt(target)


def pass_block(block: np.ndarray, pass_name: str, *, seed: int = PROJECTION_SEED):
    """`block` as `pass_name` sees it.

    `shipped` returns it unchanged. `down` projects to 512, matching count AND
    rank. `up` lifts to 2048, matching count while LEAVING THE RANK at the
    block's native width -- that is what makes `up` a pure width control rather
    than a capacity change, and it is pinned by a rank test.
    """
    if pass_name not in TARGET_WIDTH:
        raise ValueError(f"unknown pass {pass_name!r}; expected one of {PASSES}")
    target = TARGET_WIDTH[pass_name]
    block = np.asarray(block, dtype=np.float64)
    if target is None:
        return block
    matrix = projection(block.shape[1], target, seed=seed)
    return block if matrix is None else block @ matrix
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/eval/test_width.py -q`
Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/eval/width.py tests/eval/test_width.py
git commit -m "feat: width.py -- three passes, JL projections, and the two anchors that come free from the identity case"
```

---

### Task 4: `width.py` — Reading F and its table

**Why:** Reading F is the milestone. Its five statuses each pre-register a different consequence for the 13.5-hour retrain, so the reading and the lever choice are one act.

**Files:**
- Modify: `src/mbfps/eval/width.py` — append
- Test: `tests/eval/test_width.py` — append

**Interfaces:**
- Consumes: `retention.SEEDS_REQUIRED`, `retention.ARMS_REQUIRED`, `retention.BaseControl`, `retention.BASE_R2_FLOOR`.
- Produces, used by Task 6:

```python
@dataclass(frozen=True)
class ContrastArm:
    contrast: float; ci_low: float; ci_high: float
    seeds_up: int; seeds_down: int; seeds_total: int
    def clears_up(self) -> bool: ...
    def clears_down(self) -> bool: ...

@dataclass(frozen=True)
class ContrastInputs:
    arms: dict[str, ContrastArm]
    base: dict[str, BaseControl]
    anchors: dict[str, bool]      # pass -> did its anchor reproduce
    clusters: int
    rows: int

@dataclass(frozen=True)
class ContrastStatus:
    status: str; rule: str
    arms_up: tuple[str, ...]; arms_down: tuple[str, ...]
    base_failed: tuple[str, ...]; anchors_broken: tuple[str, ...]

def contrast_arm(contrasts: list[dict]) -> ContrastArm: ...
def reading_contrast(inputs: ContrastInputs) -> ContrastStatus: ...
READING_COLUMNS: tuple[str, ...]; READING_WIDTHS: tuple[int, ...]
def format_reading_contrast(reading, inputs) -> str: ...
```

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_width.py`:

```python
from mbfps.eval.retention import ARMS_REQUIRED, BASE_R2_FLOOR, BaseControl, SEEDS_REQUIRED
from mbfps.eval.width import (  # noqa: E402
    READING_COLUMNS, READING_WIDTHS, ContrastArm, ContrastInputs,
    contrast_arm, format_reading_contrast, reading_contrast,
)

ARMS = ("frozen_ssl", "pixel_ae", "random_vit")


def _arm(ci_low: float, ci_high: float) -> ContrastArm:
    up = 3 if ci_low > 0 else 0
    down = 3 if ci_high < 0 else 0
    return ContrastArm(contrast=(ci_low + ci_high) / 2, ci_low=ci_low,
                       ci_high=ci_high, seeds_up=up, seeds_down=down, seeds_total=3)


def _inputs(arms=None, *, base_r2=0.60, anchors=None) -> ContrastInputs:
    return ContrastInputs(
        arms=arms or {a: _arm(-0.01, 0.01) for a in ARMS},
        base={a: BaseControl(r2=base_r2, seeds_clear=3 if base_r2 > BASE_R2_FLOOR else 0,
                             seeds_total=3) for a in ARMS},
        anchors=anchors or {"down": True, "up": True},
        clusters=24, rows=8015,
    )


def test_contrast_arm_refuses_a_non_finite_value():
    """M3i's ledger note, still binding: every comparison against NaN is False,
    so a non-finite contrast would read 'not up' AND 'not down' at once, which
    lands on INDISTINGUISHABLE -- a pre-registered finding asserted from a
    broken number."""
    for field in ("contrast", "ci_low", "ci_high"):
        for bad in (float("nan"), float("inf"), float("-inf")):
            seed = {"contrast": 0.1, "ci_low": 0.05, "ci_high": 0.2}
            seed[field] = bad
            with pytest.raises(ValueError, match="non-finite"):
                contrast_arm([seed] * SEEDS_REQUIRED)


def test_contrast_arm_tallies_both_directions_separately():
    """Reading F is two-sided, so an arm needs BOTH tallies: M3h shipped a table
    that printed only the up tally beside a verdict read from the down one."""
    arm = contrast_arm([
        {"contrast": +0.30, "ci_low": +0.10, "ci_high": +0.50},
        {"contrast": -0.20, "ci_low": -0.40, "ci_high": -0.05},
        {"contrast": +0.05, "ci_low": -0.02, "ci_high": +0.12},
    ])
    assert arm.seeds_up == 1 and arm.seeds_down == 1 and arm.seeds_total == 3
    assert arm.contrast == pytest.approx(0.05)
    assert arm.ci_low == pytest.approx(-0.40) and arm.ci_high == pytest.approx(+0.50)
    assert arm.clears_up() is False and arm.clears_down() is False


def test_contrast_arm_needs_seeds_required_seeds():
    with pytest.raises(ValueError, match="at least"):
        contrast_arm([{"contrast": 0.1, "ci_low": 0.05, "ci_high": 0.2}])


def test_reading_is_unresolved_base_when_the_frame_cannot_locate_itself():
    reading = reading_contrast(_inputs({a: _arm(0.1, 0.3) for a in ARMS},
                                       base_r2=BASE_R2_FLOOR - 0.05))
    assert reading.status == "UNRESOLVED_BASE"
    assert reading.base_failed == ARMS
    assert reading.arms_up == () and reading.arms_down == ()


def test_reading_is_unresolved_anchor_when_the_down_pass_anchor_breaks():
    """`down` is the pass Reading F is taken on, so a broken anchor there means
    the projection plumbing is wrong and the reading is unreadable."""
    reading = reading_contrast(_inputs({a: _arm(0.1, 0.3) for a in ARMS},
                                       anchors={"down": False, "up": True}))
    assert reading.status == "UNRESOLVED_ANCHOR"
    assert reading.anchors_broken == ("down",)


def test_a_broken_up_anchor_does_not_suppress_the_reading():
    """Reading F is taken on `down` alone. `up` only calibrates the width-bias
    companion, so letting its failure block a verdict would let a companion that
    decides nothing veto one that does."""
    reading = reading_contrast(_inputs({a: _arm(0.1, 0.3) for a in ARMS},
                                       anchors={"down": True, "up": False}))
    assert reading.status == "PAST_FRAME_AHEAD"
    assert reading.anchors_broken == ("up",), "still reported, just not fatal"


def test_reading_is_past_frame_ahead_when_the_contrast_clears_positive():
    reading = reading_contrast(_inputs({
        "frozen_ssl": _arm(0.02, 0.06), "pixel_ae": _arm(0.03, 0.07),
        "random_vit": _arm(-0.01, 0.01),
    }))
    assert reading.status == "PAST_FRAME_AHEAD"
    assert reading.arms_up == ("frozen_ssl", "pixel_ae")


def test_reading_is_recurrent_ahead_when_the_contrast_clears_negative():
    """The direction M3j could not have reported: h retaining MORE than the past
    frame at equal width is a positive finding about h, not an absence."""
    reading = reading_contrast(_inputs({
        "frozen_ssl": _arm(-0.06, -0.02), "pixel_ae": _arm(-0.07, -0.03),
        "random_vit": _arm(-0.01, 0.01),
    }))
    assert reading.status == "RECURRENT_AHEAD"
    assert reading.arms_down == ("frozen_ssl", "pixel_ae")


def test_reading_is_indistinguishable_when_neither_direction_clears():
    reading = reading_contrast(_inputs())
    assert reading.status == "INDISTINGUISHABLE"
    assert "could not" in reading.rule or "by default" in reading.rule


def test_up_and_down_cannot_both_clear_with_three_arms():
    """ARMS_REQUIRED of 2 out of 3 arms means the two directions cannot both
    reach the bar -- 2 + 2 > 3. Pinned so a later change to ARMS_REQUIRED or the
    arm count surfaces the contradiction here rather than in a verdict."""
    assert 2 * ARMS_REQUIRED > len(ARMS)


def test_reading_columns_and_widths_stay_the_same_length():
    assert len(READING_COLUMNS) == len(READING_WIDTHS)


def test_reading_table_puts_each_value_under_its_own_caption():
    """Header and rows sliced at the same offsets, and the VALUE asserted --
    asserting non-emptiness alone is what let three wrong captions ship here."""
    inputs = _inputs({"frozen_ssl": _arm(0.02, 0.06), "pixel_ae": _arm(-0.07, -0.03),
                      "random_vit": _arm(-0.01, 0.01)})
    text = format_reading_contrast(reading_contrast(inputs), inputs)
    lines = [line for line in text.splitlines() if line.startswith("  ")]
    header = lines[0]
    offset = 2
    for name, width in zip(READING_COLUMNS, READING_WIDTHS, strict=True):
        assert header[offset:offset + width].strip() == name
        offset += width
    row = next(line for line in lines[1:] if "frozen_ssl" in line)
    offset = 2
    parsed = {}
    for name, width in zip(READING_COLUMNS, READING_WIDTHS, strict=True):
        parsed[name] = row[offset:offset + width].strip()
        offset += width
    arm = inputs.arms["frozen_ssl"]
    assert parsed["arm"] == "frozen_ssl"
    assert float(parsed["contrast"]) == pytest.approx(arm.contrast, abs=5e-5)
    assert float(parsed["ci_low"]) == pytest.approx(arm.ci_low, abs=5e-5)
    assert float(parsed["ci_high"]) == pytest.approx(arm.ci_high, abs=5e-5)
    assert parsed["up"] == f"{arm.seeds_up}/3" and parsed["dn"] == f"{arm.seeds_down}/3"


def test_reading_caption_names_the_horizon_the_rule_and_the_clusters():
    inputs = _inputs()
    caption = format_reading_contrast(reading_contrast(inputs), inputs).splitlines()[0]
    assert f"k = {CONTRAST_K}" in caption
    assert "two_frame" in caption and "deterministic" in caption
    assert "24 clusters" in caption
    assert "two-sided" in caption
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_width.py -q`
Expected: FAIL, `ImportError: cannot import name 'ContrastArm'`.

- [ ] **Step 3: Append Reading F**

```python
@dataclass(frozen=True)
class ContrastArm:
    """One arm's k = 15 contrast, summarised over its seeds, BOTH ways.

    `seeds_down` sits beside `seeds_up` because M3h shipped a Reading N table
    that printed only the up tally while its verdict was read from the down
    one: every row said `0/3` under a verdict asserting "3 of 3 arms", and the
    natural misreading was the opposite of the truth.

    `contrast` is the seed mean; `ci_low` the least lower bound and `ci_high`
    the greatest upper bound across seeds -- the conservative summary each way,
    so an arm is never credited with an interval only its luckiest seed reached.
    Neither decides: the tallies do.
    """

    contrast: float
    ci_low: float
    ci_high: float
    seeds_up: int
    seeds_down: int
    seeds_total: int

    def clears_up(self) -> bool:
        return self.seeds_up >= SEEDS_REQUIRED

    def clears_down(self) -> bool:
        return self.seeds_down >= SEEDS_REQUIRED


@dataclass(frozen=True)
class ContrastInputs:
    """Everything Reading F is decided on. `anchors` maps each projecting pass
    to whether its known-answer rung reproduced its shipped gain."""

    arms: dict[str, ContrastArm]
    base: dict[str, BaseControl]
    anchors: dict[str, bool]
    clusters: int
    rows: int


@dataclass(frozen=True)
class ContrastStatus:
    status: str
    rule: str
    arms_up: tuple[str, ...]
    arms_down: tuple[str, ...]
    base_failed: tuple[str, ...]
    anchors_broken: tuple[str, ...]


def contrast_arm(contrasts: list[dict]) -> ContrastArm:
    """One arm's `ContrastArm` from its per-seed `probe.contrast_from_blocks`
    dicts.

    THE FINITENESS GUARD RAISES, and this is M3i's ledger note still binding:
    every comparison against NaN is False, so a non-finite contrast would read
    "not up" AND "not down" at once. That lands on INDISTINGUISHABLE -- a
    pre-registered finding asserted from a broken number, and the direction of
    the silence is toward a status the project will act on.
    """
    if len(contrasts) < SEEDS_REQUIRED:
        raise ValueError(
            f"an arm needs at least SEEDS_REQUIRED={SEEDS_REQUIRED} seeds to "
            f"summarise, got {len(contrasts)}"
        )
    for index, seed in enumerate(contrasts):
        bad = {
            key: float(seed[key])
            for key in ("contrast", "ci_low", "ci_high")
            if not np.isfinite(float(seed[key]))
        }
        if bad:
            raise ValueError(
                f"non-finite {', '.join(sorted(bad))} in seed index {index}: "
                f"{bad}. A non-finite contrast is an error about the "
                "measurement, not an INDISTINGUISHABLE reading."
            )
    return ContrastArm(
        contrast=float(np.mean([c["contrast"] for c in contrasts])),
        ci_low=float(min(float(c["ci_low"]) for c in contrasts)),
        ci_high=float(max(float(c["ci_high"]) for c in contrasts)),
        seeds_up=sum(1 for c in contrasts if float(c["ci_low"]) > 0.0),
        seeds_down=sum(1 for c in contrasts if float(c["ci_high"]) < 0.0),
        seeds_total=len(contrasts),
    )


def reading_contrast(inputs: ContrastInputs) -> ContrastStatus:
    """Reading F: at equal block width, does the past frame still beat the
    recurrent state?

    Precedence (spec 3.2). `UNRESOLVED_BASE` outranks every result -- a
    comparison means nothing if the current frame cannot say where it is. Then
    `UNRESOLVED_ANCHOR`, but ONLY for the `down` pass: Reading F is taken on
    `down` alone, so a broken `up` anchor suppresses the width-bias companion
    and is reported, without suppressing the reading. Letting a companion that
    decides nothing veto a verdict that does would be the wrong trade.

    TWO-SIDED, and that departs from Reading E with a reason. M3j's gain was
    one-sided because a negative gain meant noise. Here `deterministic` beating
    `two_frame` at equal width is a positive finding about what `h` retains, not
    an absence. The two directions cannot both clear: 2 of 3 arms each way needs
    4 of 3 arms.
    """
    base_failed = tuple(
        sorted(a for a, control in inputs.base.items() if not control.clears())
    )
    holding = len(inputs.base) - len(base_failed)
    anchors_broken = tuple(sorted(p for p, ok in inputs.anchors.items() if not ok))

    if holding < ARMS_REQUIRED:
        return ContrastStatus(
            status="UNRESOLVED_BASE",
            rule=(
                f"enc(t) -> position cleared r2 {BASE_R2_FLOOR:.2f} in only {holding} "
                f"of {len(inputs.base)} arms ({', '.join(base_failed)} failed); the "
                f"current frame cannot linearly say where it is, so no comparison "
                f"between blocks means anything and no reading is taken"
            ),
            arms_up=(), arms_down=(), base_failed=base_failed,
            anchors_broken=anchors_broken,
        )
    if "down" in anchors_broken:
        return ContrastStatus(
            status="UNRESOLVED_ANCHOR",
            rule=(
                f"the down pass's anchor rung ({ANCHOR['down']}) did not reproduce its "
                f"shipped gain, so the projection plumbing is wrong and the pass "
                f"Reading F is taken on is unreadable; no reading is taken"
            ),
            arms_up=(), arms_down=(), base_failed=(),
            anchors_broken=anchors_broken,
        )

    up = tuple(sorted(a for a, arm in inputs.arms.items() if arm.clears_up()))
    down = tuple(sorted(a for a, arm in inputs.arms.items() if arm.clears_down()))

    if len(up) >= ARMS_REQUIRED:
        return ContrastStatus(
            status="PAST_FRAME_AHEAD",
            rule=(
                f"at equal block width the past frame still beats the recurrent state "
                f"at k = {CONTRAST_K} in {len(up)} of {len(inputs.arms)} arms "
                f"({', '.join(up)}), each in at least {SEEDS_REQUIRED} seeds; M3j's "
                f"objective-lever argument survives the width matching"
            ),
            arms_up=up, arms_down=down, base_failed=(), anchors_broken=anchors_broken,
        )
    if len(down) >= ARMS_REQUIRED:
        return ContrastStatus(
            status="RECURRENT_AHEAD",
            rule=(
                f"at equal block width the recurrent state beats the past frame at "
                f"k = {CONTRAST_K} in {len(down)} of {len(inputs.arms)} arms "
                f"({', '.join(down)}); M3j's objective-lever argument is refuted and "
                f"its k = 15 result was width"
            ),
            arms_up=up, arms_down=down, base_failed=(), anchors_broken=anchors_broken,
        )
    return ContrastStatus(
        status="INDISTINGUISHABLE",
        rule=(
            f"neither direction clears in {ARMS_REQUIRED} arms at k = {CONTRAST_K}; "
            f"M3j's k = 15 result was width, and the bottleneck lever stands alone BY "
            f"DEFAULT rather than by evidence -- we could not tell the two blocks "
            f"apart, which is not the same as ruling one out"
        ),
        arms_up=up, arms_down=down, base_failed=(), anchors_broken=anchors_broken,
    )


READING_COLUMNS: tuple[str, ...] = (
    "arm", "contrast", "ci_low", "ci_high", "up", "dn", "clears",
)
READING_WIDTHS: tuple[int, ...] = (13, 12, 11, 11, 7, 7, 9)


def format_reading_contrast(reading: ContrastStatus, inputs: ContrastInputs) -> str:
    """Reading F as it is printed and written to `width.txt`, byte for byte."""
    lines = [
        f"--- Reading F: at EQUAL block width (512), does two_frame still beat "
        f"deterministic on translation at k = {CONTRAST_K}? "
        f"(difference of joint R^2; the shared base cancels; two-sided, clears when "
        f"an interval excludes 0 in {SEEDS_REQUIRED} of 3 seeds and {ARMS_REQUIRED} "
        f"of 3 arms); {inputs.rows} rows over {inputs.clusters} clusters ---",
        "  " + "".join(
            f"{name:>{width}}"
            for name, width in zip(READING_COLUMNS, READING_WIDTHS, strict=True)
        ),
    ]
    for name in sorted(inputs.arms):
        arm = inputs.arms[name]
        clears = "up" if arm.clears_up() else ("down" if arm.clears_down() else "no")
        lines.append("  " + "".join(
            f"{value:>{width}}"
            for value, width in zip((
                name, f"{arm.contrast:+.4f}", f"{arm.ci_low:+.4f}",
                f"{arm.ci_high:+.4f}", f"{arm.seeds_up}/{arm.seeds_total}",
                f"{arm.seeds_down}/{arm.seeds_total}", clears,
            ), READING_WIDTHS, strict=True)
        ))
    lines += [
        "  base control (enc(t) -> position, must clear r2 "
        f"{BASE_R2_FLOOR:.2f}): " + ", ".join(
            f"{a} r2={inputs.base[a].r2:+.3f} "
            f"{inputs.base[a].seeds_clear}/{inputs.base[a].seeds_total}"
            for a in sorted(inputs.base)
        ),
        "  anchors (a pass's untouched rung must reproduce its shipped gain): "
        + ", ".join(
            f"{p}={'ok' if ok else 'BROKEN'}" for p, ok in sorted(inputs.anchors.items())
        ),
        f"  verdict: {reading.status.replace('_', ' ')} -- decided by: {reading.rule}",
    ]
    return "\n".join(lines)
```

Add to the imports at the top of `width.py`:

```python
from dataclasses import dataclass

from mbfps.eval.retention import (
    ARMS_REQUIRED, BASE_R2_FLOOR, BaseControl, RUNGS, SEEDS_REQUIRED,
)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/eval/test_width.py -q`
Expected: PASS, 24 tests.

- [ ] **Step 5: Run one mutation by hand**

Temporarily swap the `PAST_FRAME_AHEAD` and `RECURRENT_AHEAD` blocks in
`reading_contrast`, then run:

`.venv/bin/python -m pytest tests/eval/test_width.py -q`

Expected: FAIL on both direction tests. Revert, clear `src/` and `tests/`
`__pycache__`, and confirm the suite passes again. On this project a test that
cannot fail has shipped seven times; one hand-run mutation per reading task is
the cheapest guard.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/width.py tests/eval/test_width.py
git commit -m "feat: Reading F -- five statuses, two-sided, and only the down anchor gates it"
```

---

### Task 5: `scripts/latent_width.py` — three passes over one gather

**Why:** the impure half. One gather per cell serves all three passes — only the probe fits differ, which is what keeps this from costing three times M3j.

**Files:**
- Create: `scripts/latent_width.py`
- Test: `tests/eval/test_latent_width_script.py`

**Interfaces:**
- Consumes: from `probe` — `gather_probe_data`, `GainSplit`, `gain_from_blocks`, `contrast_from_blocks`, `fit_probe`, `probe_r2`. From `retention` — `RUNGS`, `TARGETS`, `K_REPORTED`, `RESAMPLES`, `CONFIDENCE`, `RETENTION_FAMILY`, `shifted_rows`, `backward_translation`, `backward_rotation`, `rung_block`. From `width` — `PASSES`, `ANCHOR`, `CONTRAST_K`, `PROJECTION_SEED`, `RUNG_WIDTH`, `TARGET_WIDTH`, `pass_block`. From `scripts/trust_horizon.py` loaded by path (copy `scripts/latent_retention.py`'s `_sibling` loader and its `_cell_args` shim verbatim) — `Cell`, `load_cell`, `self_check`, `prepare_cell`, `reference_trajectories`, and the inherited exit codes.
- Produces, used by Task 6:

```python
EXIT_BASE_UNRESOLVED = 41
EXIT_ANCHOR_BROKEN = 42
PHASES: tuple[str, ...] = ("measure", "read", "all")
FIT_EPISODES: int = 20
SELECT_EPISODES: int = 20

def width_record_path(out_dir: Path, arm: str, seed: int) -> Path: ...
def gather_three_splits(prepared, train, val, *, seed) -> tuple[dict, dict | None, dict]: ...
def cell_passes(fit, select, score, *, h_dim, ks=K_REPORTED) -> dict: ...
def anchor_check(passes: dict) -> dict[str, bool]: ...
def base_control(fit, select, score) -> dict: ...
def measure_cell(args, cell, device, train, val, ks=K_REPORTED) -> tuple[int, dict | None]: ...
def measure_phase(args, cells, device, train, val, ks=K_REPORTED) -> int: ...
```

`cell_passes` returns `{"passes": {pass: {target: {k_key: {rung: gain_dict}}}}, "contrast": {k_key: contrast_dict}, "rows": {k_key: int}}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_latent_width_script.py`:

```python
"""M3k's script: three passes over one gather, and the anchors that gate them."""

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

from mbfps.eval.retention import K_REPORTED, RUNGS, TARGETS
from mbfps.eval.width import ANCHOR, CONTRAST_K, PASSES, RUNG_WIDTH

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "latent_width.py"


def _load():
    spec = importlib.util.spec_from_file_location("latent_width_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load()


def _gathered(n_windows=8, steps=20, enc=2048, h_dim=512, z=1024, seed=0):
    """A `gather_probe_data`-shaped dict at the REAL widths, so the projections
    exercise their production shapes. The deterministic half of the latent holds
    the previous frame's position, so `deterministic` must gain and a noise rung
    must not."""
    rng = np.random.default_rng(seed)
    rows = n_windows * steps
    window = np.repeat(np.arange(n_windows), steps)
    step = np.tile(np.arange(steps), n_windows)
    episode = window // 2
    pos = np.column_stack([
        window * 100.0 + step * 3.0 + rng.normal(scale=5.0, size=rows),
        window * 50.0 - step * 2.0 + rng.normal(scale=5.0, size=rows),
    ])
    radians = np.deg2rad((step * 7.0 + window * 11.0) % 360.0)
    targets = np.column_stack([pos, np.sin(radians), np.cos(radians)])
    encoder = np.column_stack([pos, rng.normal(size=(rows, enc - 2))])
    lagged = np.roll(pos, 1, axis=0)
    latent = np.column_stack([
        lagged, rng.normal(size=(rows, h_dim - 2)), rng.normal(size=(rows, z)),
    ])
    return {
        "latent": latent, "encoder_embedding": encoder,
        "embedding": encoder + rng.normal(scale=20.0, size=encoder.shape),
        "targets": targets, "window": window, "step": step, "episode": episode,
    }


def test_exit_codes_are_41_and_42_and_do_not_collide():
    """M3i holds 38 and M3j holds 39-40. A reused number makes two different
    refusals indistinguishable to a caller reading `$?`."""
    assert script.EXIT_BASE_UNRESOLVED == 41
    assert script.EXIT_ANCHOR_BROKEN == 42


def test_cell_passes_covers_every_pass_target_rung_and_horizon():
    splits = [_gathered(seed=s) for s in (0, 1, 2)]
    out = script.cell_passes(*splits, h_dim=512, ks=(1, CONTRAST_K))
    assert set(out["passes"]) == set(PASSES)
    for pass_name in PASSES:
        assert set(out["passes"][pass_name]) == set(TARGETS)
        for target in TARGETS:
            assert set(out["passes"][pass_name][target]) == {"k1", f"k{CONTRAST_K}"}
            for k in out["passes"][pass_name][target]:
                assert set(out["passes"][pass_name][target][k]) == set(RUNGS)


def test_each_pass_anchor_reproduces_its_shipped_gain_exactly():
    """THE known-answer property. `projection` returns the identity when the
    native width already equals the target, so the anchor rung is not touched
    and its gain must be bit-identical to the shipped pass -- not approximately
    equal. Anything else means the projection machinery is running where it
    should not."""
    splits = [_gathered(seed=s) for s in (0, 1, 2)]
    out = script.cell_passes(*splits, h_dim=512, ks=(CONTRAST_K,))
    key = f"k{CONTRAST_K}"
    for pass_name, rung in ANCHOR.items():
        shipped = out["passes"]["shipped"]["translation"][key][rung]["gain"]
        projected = out["passes"][pass_name]["translation"][key][rung]["gain"]
        assert projected == shipped, (
            f"{pass_name}'s anchor {rung} moved: {projected} vs shipped {shipped}"
        )
    assert script.anchor_check(out["passes"]) == {"down": True, "up": True}


def test_anchor_check_reports_a_broken_anchor():
    splits = [_gathered(seed=s) for s in (0, 1, 2)]
    out = script.cell_passes(*splits, h_dim=512, ks=(CONTRAST_K,))
    key = f"k{CONTRAST_K}"
    out["passes"]["down"]["translation"][key][ANCHOR["down"]]["gain"] += 1e-9
    assert script.anchor_check(out["passes"])["down"] is False


def test_the_down_pass_projects_every_rung_to_the_same_width(monkeypatch):
    """Matching means every rung's BLOCK is the same width; the base is left
    alone so gains stay on M3j's scale. Only a recorded width can catch a rung
    that silently kept its native size."""
    seen = []
    real = script.gain_from_blocks

    def recording(fit, select, score, **kwargs):
        seen.append(np.asarray(fit.block).shape[1])
        return real(fit, select, score, **kwargs)

    monkeypatch.setattr(script, "gain_from_blocks", recording)
    splits = [_gathered(seed=s) for s in (0, 1, 2)]
    script.cell_passes(*splits, h_dim=512, ks=(CONTRAST_K,))
    # Order-independent, because the loop nests k > target > pass > rung and a
    # positional slice would silently assume otherwise. Per target: `shipped`
    # contributes each native width once, `down` contributes 512 four times and
    # `up` contributes 2048 four times. Over both targets that is:
    from collections import Counter
    assert Counter(seen) == {2048: 10, 512: 10, 1024: 2, 1536: 2}, Counter(seen)


def test_the_contrast_is_taken_on_the_down_pass_at_the_contrast_horizon():
    """Reading F is decided on `down` at k = CONTRAST_K. A contrast taken on the
    shipped pass would reproduce exactly the confound this milestone exists to
    remove."""
    splits = [_gathered(seed=s) for s in (0, 1, 2)]
    out = script.cell_passes(*splits, h_dim=512, ks=(1, CONTRAST_K))
    assert set(out["contrast"]) == {f"k{CONTRAST_K}"}
    c = out["contrast"][f"k{CONTRAST_K}"]
    down = out["passes"]["down"]["translation"][f"k{CONTRAST_K}"]
    assert c["a_r2"] == pytest.approx(down["two_frame"]["joint_r2"], abs=1e-12)
    assert c["b_r2"] == pytest.approx(down["deterministic"]["joint_r2"], abs=1e-12)
    assert c["contrast"] == pytest.approx(c["a_r2"] - c["b_r2"], abs=1e-12)


def test_base_control_records_position_and_the_four_column_mean():
    """M3k gates on position alone (spec 2.5), but the 4-column mean is kept so
    the record bridges to M3j's."""
    splits = [_gathered(seed=s) for s in (0, 1, 2)]
    control = script.base_control(*splits)
    assert control["position_r2"] > 0.5
    assert "r2" in control and "per_column_r2" in control
    assert "gain" not in control, "the base control is a level, not a gain"


def test_gather_three_splits_scores_every_validation_episode(monkeypatch):
    """M3j's smoke caught this: `filtering_gain`'s `limit` caps the fit split AND
    the scored split, so mirroring it discarded four of the 24 validation
    episodes. The scored split takes all of them."""
    calls = []

    def recorder(model, paths, backbone, device, *, context, horizon, limit, seed):
        calls.append({"paths": list(paths), "seed": seed, "limit": limit})
        return _gathered()

    monkeypatch.setattr(script, "gather_probe_data", recorder)
    prepared = types.SimpleNamespace(
        model="m", common={"feature_backbone": "b", "device": "cpu"},
        context=5, horizon=45,
    )
    train = [f"t{i}" for i in range(50)]
    val = [f"v{i}" for i in range(10)]
    script.gather_three_splits(prepared, train, val, seed=7)
    fit, select, score = calls
    assert fit["paths"] == train[:script.FIT_EPISODES] and fit["seed"] == 7
    assert select["seed"] == 9 and score["seed"] == 8
    assert score["limit"] == len(val) != script.FIT_EPISODES


def test_one_gather_serves_all_three_passes(monkeypatch):
    """The reason this is not 3x M3j's cost. Three gathers per CELL, not per
    pass -- nine would mean the passes were re-gathering identical rows."""
    calls = []
    monkeypatch.setattr(script, "gather_probe_data",
                        lambda *a, **k: calls.append(1) or _gathered())
    prepared = types.SimpleNamespace(
        model="m", common={"feature_backbone": "b", "device": "cpu"},
        context=5, horizon=45,
    )
    script.gather_three_splits(prepared, [f"t{i}" for i in range(50)],
                               [f"v{i}" for i in range(10)], seed=0)
    assert len(calls) == 3, "fit, select and score -- once each, shared by every pass"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_latent_width_script.py -q`
Expected: collection error — the script does not exist.

- [ ] **Step 3: Write the script's measure half**

Create `scripts/latent_width.py`. Copy the `_sibling` loader, the `trust_horizon`
re-exports, `_cell_args`, `gather_three_splits`, `base_control` and `write_record`
usage from `scripts/latent_retention.py` **verbatim** — including `_cell_args`'s
docstring, since `prepare_cell` loads the checkpoint from the `out` of the args it
is handed and that is the STUDY directory. Then:

```python
EXIT_BASE_UNRESOLVED = 41
"""`enc(t)` -> absolute POSITION did not read. Every gain and every contrast is
measured against that base, so if the current frame cannot linearly say where it
is, no comparison between blocks means anything."""

EXIT_ANCHOR_BROKEN = 42
"""The `down` pass's anchor rung did not reproduce its shipped gain. `projection`
returns the identity when the native width already equals the target, so that
rung is untouched and its gain MUST be bit-identical. Anything else means the
projection machinery ran where it should not, and the pass Reading F is taken on
is unreadable."""

PHASES: tuple[str, ...] = ("measure", "read", "all")


def width_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    return Path(out_dir) / f"width_{arm}_seed{seed}.json"


def _split_for(data, target, k, rung, pass_name, h_dim):
    """One rung's `GainSplit` for one pass, and the rows it uses.

    The base is `enc(t)` on the target's rows and is NOT projected: matching
    applies to the rung's block only, so every gain stays on M3j's scale and the
    base R^2 is identical across passes by construction.
    """
    values, rows = TARGET_BUILDERS[target](
        data["targets"], data["window"], data["step"], k,
    )
    if rows.size == 0:
        raise ValueError(
            f"k={k}: no window is long enough to carry a backward displacement; "
            "a gain computed on zero rows would describe nothing"
        )
    _, source = shifted_rows(data["window"], data["step"], k)
    block = rung_block(data, rung, rows=rows, source=source, h_dim=h_dim)
    return GainSplit(
        base=np.asarray(data["encoder_embedding"], dtype=np.float64)[rows],
        block=pass_block(block, pass_name),
        target=values,
    ), rows


def cell_passes(fit, select, score, *, h_dim: int, ks=K_REPORTED) -> dict:
    """Every `(pass, target, k, rung)` gain for one cell, plus the contrast.

    Three passes over ONE gather: only the probe fits differ, which is what
    keeps this from costing three times M3j. The bootstrap groups on `episode`,
    never `window`.
    """
    passes: dict = {p: {t: {} for t in TARGETS} for p in PASSES}
    contrast: dict = {}
    rows_by_k: dict[str, int] = {}
    for k in ks:
        row_sets = {}
        for target in TARGETS:
            for pass_name in PASSES:
                splits = {}
                for rung in RUNGS:
                    splits[rung] = {}
                    for name, data in (("fit", fit), ("select", select), ("score", score)):
                        if data is None:          # only ever the select split
                            splits[rung][name] = None
                            continue
                        split, rows = _split_for(data, target, k, rung, pass_name, h_dim)
                        splits[rung][name] = split
                        row_sets[name] = rows
                groups = np.asarray(score["episode"])[row_sets["score"]]
                passes[pass_name][target][k_key(k)] = {
                    rung: gain_from_blocks(
                        splits[rung]["fit"], splits[rung]["select"], splits[rung]["score"],
                        groups=groups, resamples=RESAMPLES, confidence=CONFIDENCE,
                        seed=0,
                    )
                    for rung in RUNGS
                }
        rows_by_k[k_key(k)] = int(row_sets["score"].size)
        if k == CONTRAST_K:
            # Reading F: taken on `down`, where the widths are matched. A
            # contrast on the shipped pass would reproduce exactly the confound
            # this milestone exists to remove.
            groups = np.asarray(score["episode"])[row_sets["score"]]
            def triple(rung):
                return tuple(
                    _split_for(d, "translation", k, rung, "down", h_dim)[0]
                    if d is not None else None
                    for d in (fit, select, score)
                )
            contrast[k_key(k)] = contrast_from_blocks(
                triple("two_frame"), triple("deterministic"),
                groups=groups, resamples=RESAMPLES, confidence=CONFIDENCE, seed=0,
            )
    return {"passes": passes, "contrast": contrast, "rows": rows_by_k}


def anchor_check(passes: dict) -> dict[str, bool]:
    """Did each projecting pass leave its anchor rung bit-identical?

    `ANCHOR[pass]` is the rung whose native width already equals that pass's
    target, so `projection` returns the identity and the rung is not touched.
    Exact equality, not `isclose`: an approximate match would mean a matmul ran.

    Checked at EVERY measured horizon rather than only at `CONTRAST_K`, so a
    smoke run whose `ks` omit the contrast horizon still checks its anchors
    instead of raising a KeyError -- and so a projection that leaks at one
    horizon and not another cannot hide.
    """
    shipped = passes["shipped"]
    horizons = sorted(set(shipped["translation"]))
    if not horizons:
        raise ValueError("no horizon was measured; there is no anchor to check")
    return {
        pass_name: all(
            passes[pass_name][target][key][rung]["gain"]
            == shipped[target][key][rung]["gain"]
            for target in TARGETS
            for key in horizons
        )
        for pass_name, rung in ANCHOR.items()
    }
```

`measure_cell` mirrors `scripts/latent_retention.py`'s exactly — `prepare_cell`
via `_cell_args(args, args.source)`, the `reference_trajectories` pass, the
`self_check(traj, cell.diagnostic)` gate, `cell.record["steps"]` — and records
`cell_passes`' three keys, `anchor_check`'s result, `base_control`, the
`projection_seed`, and `{rung: RUNG_WIDTH[rung]}` so every pass is reproducible.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/eval/test_latent_width_script.py -q`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/latent_width.py tests/eval/test_latent_width_script.py
git commit -m "feat: latent_width.py measure -- three passes over one gather, with the anchors recorded per cell"
```

---

### Task 6: `scripts/latent_width.py` — the read phase and the registry

**Why:** `read` pools the nine records into Reading F, the width-bias companion, and the corrected Reading E, then writes `width.txt`.

**Files:**
- Modify: `scripts/latent_width.py` — append
- Test: `tests/eval/test_latent_width_script.py`, `tests/eval/test_diagnose_dynamics_script.py`

**Interfaces:**
- Consumes: `width.contrast_arm`, `width.reading_contrast`, `width.format_reading_contrast`, and `retention.reading_retention(..., z_bearing=("stochastic",))` from Task 2.
- Produces: `contrast_inputs`, `width_bias_table`, `corrected_reading_e`, `width_text`, `read_phase`, `main`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_latent_width_script.py`:

```python
def _record(arm, seed, *, clears=0.0, base_r2=0.65, anchors=(True, True)):
    key = f"k{CONTRAST_K}"
    passes = {p: {t: {key: {r: {"gain": 0.01, "ci_low": -0.01, "ci_high": 0.03,
                                "joint_r2": 0.05, "embedding_r2": 0.04,
                                "confidence": 0.95, "n_scored_windows": 24,
                                "ridge_selected": True, "joint_ridge": 1e5,
                                "embedding_ridge": 1e5}
                          for r in RUNGS} for t in TARGETS} for p in PASSES}
    return {
        "arm": arm, "seed": seed, "step": 20000, "device": "cpu",
        "context": 5, "horizon": 45, "h_dim": 512,
        "passes": passes,
        "contrast": {key: {"contrast": clears, "ci_low": clears - 0.01,
                           "ci_high": clears + 0.01, "a_r2": 0.05, "b_r2": 0.04,
                           "confidence": 0.95, "n_scored_windows": 24,
                           "a_ridge": 1e5, "b_ridge": 1e5, "ridge_selected": True}},
        "rows": {key: 8015}, "clusters": 24,
        "anchors": {"down": anchors[0], "up": anchors[1]},
        "base_control": {"r2": 0.33, "position_r2": base_r2,
                         "per_column_r2": [0.7, 0.7, 0.0, -0.05],
                         "ridge": 1e3, "ridge_selected": True, "rows": 11450},
        "projection_seed": 0, "rung_width": dict(RUNG_WIDTH),
        "windows": {"episode": list(range(24)), "window": list(range(24))},
        "self_check": {"ok": True}, "git_sha": "abc", "ks": [CONTRAST_K],
        "torch_version": "2.13.0", "split_seed": 0, "record_git_sha": "ca3e140",
        "episodes": {"fit": [], "select": [], "val": [f"e{i}" for i in range(24)]},
    }


def _records(**kw):
    return {(a, s): _record(a, s, **kw)
            for a in ("frozen_ssl", "pixel_ae", "random_vit") for s in (0, 1, 2)}


@pytest.mark.parametrize("clears,expected", [
    (+0.05, "PAST FRAME AHEAD"),
    (-0.05, "RECURRENT AHEAD"),
    (0.0, "INDISTINGUISHABLE"),
])
def test_read_reaches_every_decided_status(tmp_path, capsys, clears, expected):
    import json
    for (arm, seed), rec in _records(clears=clears).items():
        script.width_record_path(tmp_path, arm, seed).write_text(json.dumps(rec))
    status = script.main(["--phase", "read", "--out", str(tmp_path)])
    printed = capsys.readouterr().out
    assert f"verdict: {expected}" in printed
    assert status == script.EXIT_OK
    assert (tmp_path / "width.txt").read_bytes() == printed.encode()


def test_read_exits_41_when_the_position_control_fails(tmp_path):
    import json
    from mbfps.eval.retention import BASE_R2_FLOOR
    for (arm, seed), rec in _records(base_r2=BASE_R2_FLOOR - 0.05).items():
        script.width_record_path(tmp_path, arm, seed).write_text(json.dumps(rec))
    assert script.main(["--phase", "read", "--out", str(tmp_path)]) == 41


def test_read_exits_42_when_the_down_anchor_is_broken(tmp_path):
    import json
    for (arm, seed), rec in _records(anchors=(False, True)).items():
        script.width_record_path(tmp_path, arm, seed).write_text(json.dumps(rec))
    assert script.main(["--phase", "read", "--out", str(tmp_path)]) == 42


def test_a_broken_up_anchor_is_reported_but_does_not_block_the_reading(tmp_path, capsys):
    import json
    for (arm, seed), rec in _records(clears=+0.05, anchors=(True, False)).items():
        script.width_record_path(tmp_path, arm, seed).write_text(json.dumps(rec))
    assert script.main(["--phase", "read", "--out", str(tmp_path)]) == script.EXIT_OK
    out = capsys.readouterr().out
    assert "verdict: PAST FRAME AHEAD" in out
    assert "up=BROKEN" in out
    assert "width bias" in out and "unreadable" in out


def test_the_corrected_reading_e_uses_stochastic_only(monkeypatch, tmp_path):
    """M3j's own results recorded that `full` is h + z and cannot attribute a
    clearance to z. The companion must pass the corrected tuple rather than rely
    on the module constant, which stays as M3j's records were taken under."""
    import json
    seen = {}
    real = script.reading_retention

    def recording(inputs, **kwargs):
        seen.update(kwargs)
        return real(inputs, **kwargs)

    monkeypatch.setattr(script, "reading_retention", recording)
    for (arm, seed), rec in _records().items():
        script.width_record_path(tmp_path, arm, seed).write_text(json.dumps(rec))
    script.main(["--phase", "read", "--out", str(tmp_path)])
    assert seen.get("z_bearing") == ("stochastic",)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_latent_width_script.py -q`
Expected: FAIL — `AttributeError: module has no attribute 'main'`.

- [ ] **Step 3: Write the read half**

Mirror `scripts/latent_retention.py`'s `read_phase` / `_parser` / `main` in shape
and refusal wording. The parts that differ:

```python
READ_EXITS = {
    "UNRESOLVED_BASE": EXIT_BASE_UNRESOLVED,
    "UNRESOLVED_ANCHOR": EXIT_ANCHOR_BROKEN,
}
"""Status -> exit code, for the two refusals. Every other status is a READING
and exits 0: a milestone that exited non-zero on a finding would make "the run
worked" and "the news was good" the same signal."""


def contrast_inputs(records: dict) -> ContrastInputs:
    """Pool the nine records into Reading F's input.

    The seed tally comes from the per-seed intervals in the records, not from a
    pooled estimate: R^2 is not a per-window quantity, so there is nothing to
    pool the way a paired contrast pools. The agreement requirement IS the rule.
    """
    arms = sorted({arm for arm, _ in records})
    key = k_key(CONTRAST_K)
    per_arm = {
        arm: contrast_arm([
            records[(arm, seed)]["contrast"][key]
            for _, seed in sorted(c for c in records if c[0] == arm)
        ])
        for arm in arms
    }
    base = {}
    for arm in arms:
        levels = [
            float(records[(arm, seed)]["base_control"]["position_r2"])
            for _, seed in sorted(c for c in records if c[0] == arm)
        ]
        base[arm] = BaseControl(
            r2=float(np.mean(levels)),
            seeds_clear=sum(1 for r in levels if r > BASE_R2_FLOOR),
            seeds_total=len(levels),
        )
    anchors = {
        pass_name: all(r["anchors"][pass_name] for r in records.values())
        for pass_name in ANCHOR
    }
    first = records[next(iter(sorted(records)))]
    return ContrastInputs(
        arms=per_arm, base=base, anchors=anchors,
        clusters=int(first["clusters"]), rows=int(first["rows"][key]),
    )
```

`width_bias_table(records)` prints `up` minus `shipped` per rung and target —
what count alone is worth — and **prints "unreadable" instead of numbers when the
`up` anchor is broken**, because a pass whose anchor moved cannot calibrate
anything. `corrected_reading_e(records)` builds M3j's `RetentionInputs` from the
`down` pass and calls `reading_retention(inputs, z_bearing=("stochastic",))`.
`width_text` composes the self-check table, the width-bias table, the corrected
Reading E, then `format_reading_contrast` — evidence before conclusion — and
`read_phase` writes `width.txt` **before** printing, so the artefact gates the log.

- [ ] **Step 4: Add `latent_width` to the exit-status registry**

`tests/eval/test_diagnose_dynamics_script.py` holds the registry test listing
`stage_decomposition`, `sharper_latent`, `latent_motion` and `latent_retention`.
Add `latent_width` with `{41, 42}`, following that file's existing shape exactly.

- [ ] **Step 5: Run every affected suite**

Run: `.venv/bin/python -m pytest tests/eval/test_latent_width_script.py tests/eval/test_width.py tests/eval/test_retention.py tests/eval/test_diagnose_dynamics_script.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/latent_width.py tests/eval/
git commit -m "feat: latent_width.py read -- Reading F, the width-bias table, and the corrected Reading E"
```

---

### Task 7: The run

**Why:** the deliverable is the reading on the nine shipped cells, not the code that computes it.

- [ ] **Step 1: Check the disk**

```bash
du -sh "${TMPDIR}/pytest-of-${USER}" ; df -h . | tail -1
```

- [ ] **Step 2: Run the whole suite**

```bash
caffeinate -dimsu .venv/bin/python -m pytest -q
```

Expected: 0 failures, 0 warnings. Record the count and duration. Do **not**
commit while a run is in progress.

- [ ] **Step 3: Smoke one cell, timed**

```bash
mkdir -p runs/m3k_smoke && git rev-parse HEAD > runs/m3k_smoke/.head
date -u +%Y-%m-%dT%H:%M:%SZ > runs/m3k_smoke/.started
caffeinate -dimsu .venv/bin/python scripts/latent_width.py \
  --phase measure --arms pixel_ae --seeds 0 \
  --source runs/m3_study_v2 --out runs/m3k_smoke 2>&1 | tee runs/m3k_smoke/smoke.log
```

**Read the record for format before the real run, and check three things by
hand:** `clusters == 24`, `rows == {"k1": 11221, "k4": 10534, "k15": 8015}`, and
`anchors == {"down": true, "up": true}`. M3j's smoke caught a bug that had
silently discarded 17% of the evaluation data and that the whole suite had
missed. Record the wall clock; the measure phase is sized from it.

- [ ] **Step 4: Measure all nine cells**

```bash
git rev-parse HEAD > runs/m3k_retention/.head
date -u +%Y-%m-%dT%H:%M:%SZ > runs/m3k_retention/.started
nohup caffeinate -dimsu .venv/bin/python scripts/latent_width.py \
  --phase measure --source runs/m3_study_v2 --out runs/m3k_retention \
  > runs/m3k_retention/measure.log 2>&1 &
```

**Wait on the process, not on a file** — M3i lost time to a waiter that fired on
a stale marker.

- [ ] **Step 5: Accept the run**

M3j's checks — nine records at one `git_sha` equal to `.head`, all `mps`,
`self_check.ok` 9/9, 229 windows over 24 episode clusters, rows
`{k1: 11221, k4: 10534, k15: 8015}`, every gain and bound finite, `nonfinite`
empty — **plus two**: both anchors reproduce on all nine cells, and
`projection_seed` and `rung_width` are recorded on every record.

- [ ] **Step 6: Read, and verify byte-identity**

```bash
.venv/bin/python scripts/latent_width.py --phase read --out runs/m3k_retention \
  2>&1 | tee runs/m3k_retention/read.log
echo "exit: $?"
diff <(.venv/bin/python scripts/latent_width.py --phase read --out runs/m3k_retention) \
     runs/m3k_retention/width.txt && echo "byte-identical"
```

- [ ] **Step 7: Write `## Task 7 results` into this plan**

Every number quoted from an artefact with its definition. Cover, in order: the
verdict and the lever it pre-registers; provenance; the suite; acceptance; the
anchors; the base and position controls; Reading F's table; **the width-bias
table, against §1.1's design-time measurement of ~+0.0137 — state plainly whether
the nine-cell figure agrees**; the corrected Reading E and whether it differs from
M3j's `MOTION_RETAINED`; and what the milestone does not support.

If the status is `INDISTINGUISHABLE`, say explicitly that the bottleneck lever
stands **by default rather than by evidence** — spec §3.3 requires it, and "we
could not tell them apart" is not "we ruled one out."

- [ ] **Step 8: Commit**

```bash
git add docs/superpowers/plans/2026-09-29-mb-fps-m3k-width-matched-ladder.md
git commit -F <message file>
```

`runs/` is gitignored, so the records live on disk and the results section is the
committed record. Use `-F` with a file: M3i lost a commit to shell quoting.

---

## Exit criteria

- Reading F is taken on nine records at one `git_sha`, or the run refused with a
  numbered status and the refusal is recorded.
- Both anchors reproduce **exactly** on all nine cells.
- `gain_from_blocks`' ten-key output is byte-identical, pinned by the golden test.
- `retention.Z_BEARING_RUNGS` is unedited and M3j's records still read
  `MOTION_RETAINED`.
- All five Reading F statuses are reachable and driven end-to-end by a test.
- The whole suite passes with 0 failures and 0 warnings.
- `width.txt` is byte-identical to a second `--phase read`.
- No checkpoint altered, nothing under `runs/` removed, no M3b–M3j verdict touched.

## Plan self-review (writing-plans)

**Spec coverage.** §1 → Task 7 Step 7. §2.1 → Tasks 3, 5. §2.2 → Task 3. §2.3 →
Tasks 3, 5. §2.4 → Task 3. §2.5 → Tasks 2, 6. §3.1 → Tasks 1, 5. §3.2 → Task 4.
§3.3 → Task 7 Step 7. §3.4 → Task 6. §4 → Task 7 Step 7. §5 → Tasks 1–6. §6 →
Task 7. §7 → the File Structure table. No gaps.

**Placeholder scan.** Clean. Three defects were found in this plan's own code and
fixed inline rather than left for a reviewer:

1. **`test_the_down_pass_projects_every_rung_to_the_same_width` sliced `seen`
   positionally**, assuming a loop order the code does not have — `cell_passes`
   nests `k > target > pass > rung`, so `seen[:8]` spans two passes, not one.
   Replaced with an order-independent `Counter` assertion over the full run.
2. **The splits loop wrote a literal `"select"` key where it meant `name`.**
   Correct by accident (the branch is only reachable when `name == "select"`)
   and obscure; now uses `name`.
3. **`anchor_check` keyed on `CONTRAST_K` alone**, so a smoke run whose `ks`
   omit the contrast horizon would raise `KeyError` instead of checking its
   anchors. Now checks every measured horizon, which also means a projection
   leaking at one horizon and not another cannot hide.

**Type consistency.** `GainSplit(base, block, target)` in Tasks 1 and 5.
`contrast_from_blocks(a, b, *, groups, ...)` takes `(fit, select, score)` triples
in Tasks 1, 5. `ContrastArm` carries `contrast, ci_low, ci_high, seeds_up,
seeds_down, seeds_total` in Tasks 4, 6. `pass_block(block, pass_name)` in Tasks
3, 5. `anchor_check(passes) -> dict[str, bool]` in Tasks 5, 6. `CONTRAST_K` never
`DECISION_K`, pinned by a test in Task 3.
