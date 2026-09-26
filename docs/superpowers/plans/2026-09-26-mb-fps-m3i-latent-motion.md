# MB-FPS M3i — Latent Motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the posterior latent of the nine shipped M3c checkpoints encodes step-to-step *displacement*, or only absolute position — evaluation only, no training, deciding in code whether M3h §8's three prior-side levers are still worth pulling.

**Architecture:** A new statistics-and-reading module `src/mbfps/eval/motion.py` and a new run script `scripts/latent_motion.py`, in the shape M3g and M3h established. The script loads each cell through `trust_horizon.prepare_cell` (inheriting the reviewed loader, the protocol checks and the reproduction bound), fits a ridge probe from the posterior latent to displacement on TRAINING episodes, scores it per validation window against the persistence baseline `Δ = 0`, and runs the same probe on a permuted pairing as a known-answer control that must refuse. Reading D is decided by the project's existing cluster-robust pooling.

**Tech Stack:** Python 3.12, PyTorch (MPS), NumPy, pytest. No new dependencies.

## Global Constraints

- **Pre-registered, never tuned after a run:** `MOTION_FAMILY = 3`, `z_fam = cluster_threshold(3, 24) = 2.582`, `SEEDS_REQUIRED = 2`, `ARMS_REQUIRED = 2`, `K_REPORTED = (1, 5, 15, 30, 45)`, `CONTROL_SEED = 0`. The decision horizon is `DECISION_H = 15`, **imported from `mbfps.eval.split_gap`**, never re-spelled.
- **Reading D statuses and precedence:** `UNRESOLVED_CONTROL` → `MOTION_ENCODED` → `NO_MOTION` → `NO_DIFFERENCE`. A failed control can never be overwritten by a result.
- **The control must refuse.** If the permuted probe clears ±`z_fam` anywhere, exit **38 `EXIT_CONTROL_LEAKED`** and take no reading.
- **Evaluation only.** No training, no checkpoint written or altered, nothing under `runs/` removed, no M3b–M3h verdict or record touched.
- **Leakage:** the probe is fit on **training** episodes only, at the rollout's own `context` and `horizon`.
- **Exit codes:** 0 / 11 / 12 / 14 / 30 imported from `trust_horizon`; **38** new. M3h holds 35–37.
- **The reproduction rule (M3h spec §2.4):** a fresh pass reproduces a **stored** artefact within `64 * ulp(m)` via `mbfps.eval.reproduction`; in-run and training comparisons stay exactly 0.0.
- **Style:** docstrings explain the *why* and match the neighbouring register; tests carry hand-typed expected values, one rule mutated per test; a test that cannot fail is a defect.
- Run from the worktree root with `.venv/bin/python -m pytest ...`; `runs/` and `data/` are symlinks to the main checkout. Never `git stash`. Do not commit while a run is in progress.
- **Before any full-suite run:** `du -sh $TMPDIR/pytest-of-$USER`. `tmp_path_retention_policy = "failed"` landed in PR #8, but the suite still writes heavily.
- Every commit message ends with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## File Structure

| file | responsibility |
|---|---|
| `src/mbfps/eval/probe.py` | **modified**: `gather_probe_data` gains row-aligned `"window"` and `"step"` indices so a caller can reconstruct per-window trajectories. Additive; the four existing keys are unchanged. |
| `src/mbfps/eval/motion.py` | **new**: the constants, `displacement`, `contrast_series`, the descriptive statistics, Reading D and its formatter. No I/O, no torch. |
| `scripts/latent_motion.py` | **new**: phases `measure \| read \| all`; per-cell records; `motion.txt`. |
| `tests/eval/test_motion.py` | **new**: the module's rules, hand-typed. |
| `tests/eval/test_latent_motion_script.py` | **new**: the script end to end on the shared tiny fixture. |
| `tests/eval/test_probe.py` | **modified**: the two new indices. |
| `tests/eval/test_diagnose_dynamics_script.py` | **modified**: the exit-status registry gains `latent_motion`. |

---

### Task 1: `gather_probe_data` keeps the window and step of every row

M3i must compute `Δ(t→t+k) = p(t+k) − p(t)` *within a window*. `gather_probe_data` currently returns four row-aligned `(N, …)` arrays with no way to tell which rows share a window or what order they are in. Re-deriving that by reshaping would depend on an undocumented regularity; re-implementing the gathering loop would duplicate the window protocol whose docstring says the filtering depth is load-bearing. So the gatherer returns the two indices it already knows.

**Files:**
- Modify: `src/mbfps/eval/probe.py` (`gather_probe_data`, around lines 153–295)
- Test: `tests/eval/test_probe.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `gather_probe_data(...)` returns, in addition to `"latent"`, `"embedding"`, `"encoder_embedding"`, `"targets"`, two row-aligned arrays — `"window"` `(N,)` int, a 0-based index identifying which window each row came from, and `"step"` `(N,)` int, the 0-based position of the row within its window (`0 .. context + horizon - 1`).

- [ ] **Step 1: Write the failing test**

Add to `tests/eval/test_probe.py`:

```python
def test_gather_probe_data_labels_every_row_with_its_window_and_step(small_buffer):
    """M3i reconstructs per-window trajectories from these rows, which needs
    the window each row came from and its order inside that window. Without
    them a caller has to reshape on an undocumented regularity."""
    from mbfps.eval.probe import gather_probe_data

    model, paths, backbone, device = _tiny_probe_setup(small_buffer)
    data = gather_probe_data(
        model, paths, backbone, device, context=2, horizon=3, limit=4, seed=0
    )
    n = data["latent"].shape[0]
    window, step = data["window"], data["step"]

    assert window.shape == (n,) and step.shape == (n,)
    assert window.dtype.kind == "i" and step.dtype.kind == "i"
    # Every window contributes exactly context + horizon rows, in order.
    assert set(np.unique(step).tolist()) == set(range(2 + 3))
    for w in np.unique(window):
        rows = step[window == w]
        assert rows.tolist() == list(range(2 + 3)), f"window {w} is not in step order"
    # The indices are row-aligned with the payload, not a separate ordering.
    assert data["targets"].shape[0] == n and data["encoder_embedding"].shape[0] == n


def test_gather_probe_data_leaves_the_four_original_arrays_unchanged(small_buffer):
    """The indices are ADDITIVE. `fit_probes` and every existing caller read
    the four original keys and must see byte-identical arrays."""
    from mbfps.eval.probe import gather_probe_data

    model, paths, backbone, device = _tiny_probe_setup(small_buffer)
    kwargs = dict(context=2, horizon=3, limit=4, seed=0)
    first = gather_probe_data(model, paths, backbone, device, **kwargs)
    second = gather_probe_data(model, paths, backbone, device, **kwargs)
    for key in ("latent", "embedding", "encoder_embedding", "targets"):
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)
    assert set(first) == {
        "latent", "embedding", "encoder_embedding", "targets", "window", "step",
    }
```

`_tiny_probe_setup` is a helper this file already has for the shared `small_buffer` fixture; if it is spelled differently, follow the file's existing spelling rather than adding a second helper.

- [ ] **Step 2: Run it to watch it fail**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -q -k "window_and_step or original_arrays"`
Expected: `KeyError: 'window'`.

- [ ] **Step 3: Return the two indices**

In `gather_probe_data`, accumulate alongside the existing lists. Where the function appends one window's rows, append its indices too:

```python
    windows: list[np.ndarray] = []
    steps: list[np.ndarray] = []
```

and, inside the per-window loop, immediately after the window's `latent` / `targets` rows are appended:

```python
        rows = int(latent.shape[1])
        windows.append(np.full(rows, window_index, dtype=np.int64))
        steps.append(np.arange(rows, dtype=np.int64))
        window_index += 1
```

with `window_index = 0` initialised beside the lists. Then extend the return:

```python
    return {
        "latent": np.concatenate(latents),
        "targets": np.concatenate(targets),
        "embedding": np.concatenate(embeddings),
        "encoder_embedding": np.concatenate(encoder_embeddings),
        # Row-aligned with the four arrays above: which window each row came
        # from and its order within that window. M3i reconstructs per-window
        # trajectories from these; every earlier caller ignores them.
        "window": np.concatenate(windows),
        "step": np.concatenate(steps),
    }
```

Read the existing loop before editing: use ITS variable names for the per-window latent and the accumulator lists rather than the placeholders above, and place the two appends where the window's rows are actually added so the three stay row-aligned.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -q`
Expected: all pass.

- [ ] **Step 5: Run every consumer of `gather_probe_data`**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py tests/eval/test_rollout.py tests/eval/test_diagnostics.py -q`
Expected: all pass. `fit_probes` and `evaluate_rollout` read the four original keys; a break here is this task's to fix.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/probe.py tests/eval/test_probe.py
git commit -m "feat: gather_probe_data labels every row with its window and step

M3i computes displacement WITHIN a window, which needs to know which rows
share a window and their order. Re-deriving that by reshape would depend on
an undocumented regularity, and re-implementing the gathering loop would
duplicate a window protocol whose filtering depth is load-bearing. Additive:
the four existing arrays are unchanged and every earlier caller ignores the
new keys.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `motion.py` — the constants, `displacement` and `contrast_series`

**Files:**
- Create: `src/mbfps/eval/motion.py`
- Test: `tests/eval/test_motion.py`

**Interfaces:**
- Consumes: `DECISION_H` from `mbfps.eval.split_gap`; `cluster_threshold` from `mbfps.eval.pooling`.
- Produces: `MOTION_FAMILY: int`, `SEEDS_REQUIRED: int`, `ARMS_REQUIRED: int`, `K_REPORTED: tuple[int, ...]`, `CONTROL_SEED: int`, `motion_threshold(clusters: int) -> float`; `displacement(positions: np.ndarray, k: int) -> np.ndarray`; `contrast_series(predicted: np.ndarray, true: np.ndarray) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_motion.py`:

```python
"""src/mbfps/eval/motion.py: the pre-registered constants, the displacement a
window actually travelled, and the paired contrast against staying put."""

import numpy as np
import pytest

from mbfps.eval.motion import (
    ARMS_REQUIRED,
    CONTROL_SEED,
    K_REPORTED,
    MOTION_FAMILY,
    SEEDS_REQUIRED,
    contrast_series,
    displacement,
    motion_threshold,
)
from mbfps.eval.split_gap import DECISION_H


def test_the_pre_registered_constants():
    """Spec 3.1 and 3.2, fixed before the run and never tuned after it."""
    assert MOTION_FAMILY == 3
    assert SEEDS_REQUIRED == 2
    assert ARMS_REQUIRED == 2
    assert K_REPORTED == (1, 5, 15, 30, 45)
    assert CONTROL_SEED == 0
    assert DECISION_H == 15, "the decision horizon is split_gap's, not a second spelling"
    assert DECISION_H in K_REPORTED, "the decided horizon must also be reported"


def test_the_family_threshold_is_the_projects_cluster_rule():
    """Three arms over the shipped 24 episode clusters. Hand-typed: a bar that
    drifts with a refactor is a bar that decided nothing."""
    assert motion_threshold(24) == pytest.approx(2.5820, abs=5e-5)


def test_displacement_is_the_vector_a_window_travelled_over_k_steps():
    """`p(t+k) - p(t)` per window, in map units, from the first step onward.
    Not a speed and not a distance: the probe predicts a VECTOR."""
    positions = np.array([
        [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]],     # window 0: +3,+4 a step
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],     # window 1: never moves
    ])
    np.testing.assert_allclose(displacement(positions, 1), [[3.0, 4.0], [0.0, 0.0]])
    np.testing.assert_allclose(displacement(positions, 2), [[6.0, 8.0], [0.0, 0.0]])


def test_displacement_refuses_a_k_the_window_is_too_short_for():
    positions = np.zeros((2, 3, 2))
    with pytest.raises(ValueError, match="k=5"):
        displacement(positions, 5)


def test_contrast_is_persistence_error_minus_model_error():
    """Positive means the latent beat staying put. `error_persist` is `||d||`
    because the persistence baseline predicts no displacement at all."""
    true = np.array([[3.0, 4.0], [0.0, 10.0]])        # norms 5 and 10
    predicted = np.array([[3.0, 0.0], [0.0, 10.0]])   # errors 4 and 0
    np.testing.assert_allclose(contrast_series(predicted, true), [1.0, 10.0])


def test_a_model_that_predicts_nothing_scores_exactly_zero():
    """The persistence baseline IS the zero prediction, so a probe that
    outputs zeros must contrast at exactly 0.0 -- not approximately. This is
    the fixed point the whole reading is read against."""
    true = np.array([[3.0, 4.0], [-6.0, 8.0], [0.0, 0.0]])
    np.testing.assert_array_equal(contrast_series(np.zeros_like(true), true), np.zeros(3))


def test_contrast_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        contrast_series(np.zeros((3, 2)), np.zeros((4, 2)))
```

- [ ] **Step 2: Run it to watch it fail**

Run: `.venv/bin/python -m pytest tests/eval/test_motion.py -q`
Expected: `ModuleNotFoundError: No module named 'mbfps.eval.motion'`.

- [ ] **Step 3: Write the module**

Create `src/mbfps/eval/motion.py`:

```python
"""M3i: does the posterior latent encode MOTION, or only absolute position?

M3g measured that the frame injects 0.290-0.489 nats beyond what the
teacher-forced prior already predicted, summed over all 32 groups, and that
copying the previous latent predicts the next one BETTER than the model's own
prior does (persist 0.824-0.866 against teacher 0.796-0.845). M3h then ruled
out the prior's sampling temperature. Between them they leave the three
remaining prior-side levers pulling on a stage with about four tenths of a nat
of headroom.

So this module asks a different question. The latent demonstrably carries
absolute position -- `latent_selection_r2` 0.18-0.34 across the M3c records --
and a latent that encodes "which corridor am I in" can score well on a
position probe while carrying nothing about step-to-step DISPLACEMENT. The M3
gate scores the imagined trajectory against PERSISTENCE, staying put, so a
latent without displacement cannot beat it whatever the prior does.

Everything here is pure: arrays in, arrays and readings out. No torch, no I/O,
no record schema. `scripts/latent_motion.py` owns all three.
"""

from __future__ import annotations

import numpy as np

from mbfps.eval.pooling import cluster_threshold

# Pre-registered (spec 3.1, 3.2). Three arms, each contrast within a cell
# against that cell's own persistence baseline -- no arm is ranked against
# another, so the family is the arms and not their pairs.
MOTION_FAMILY: int = 3
SEEDS_REQUIRED: int = 2
ARMS_REQUIRED: int = 2

# Reported at every one of these; DECIDED only at `split_gap.DECISION_H`, the
# horizon every milestone since M3e has decided at.
K_REPORTED: tuple[int, ...] = (1, 5, 15, 30, 45)

# The permutation control's seed. Fixed so the control is reproducible: a
# control that draws a fresh permutation each run is a control whose refusal
# cannot be repeated.
CONTROL_SEED: int = 0


def motion_threshold(clusters: int) -> float:
    """The z Reading D must clear: the project's cluster-robust Bonferroni
    bar over `MOTION_FAMILY`, read against t(G-1) for G episode clusters.
    2.582 on the shipped 24-episode split."""
    return cluster_threshold(MOTION_FAMILY, clusters)


def displacement(positions, k: int) -> np.ndarray:
    """`p(t+k) - p(t)` per window, as a VECTOR in map units. `(n, 2)`.

    Taken from the window's first scored step, so every window contributes
    exactly one displacement at each k and the rows stay alignable with the
    per-window masks the pooling clusters on. A window shorter than `k + 1`
    steps is refused rather than truncated: a short window silently scored at
    a smaller k would be a different horizon pooled as if it were this one.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError(f"positions must be (n, steps, 2), got {positions.shape}")
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if positions.shape[1] < k + 1:
        raise ValueError(
            f"k={k} needs {k + 1} steps per window, got {positions.shape[1]}"
        )
    return positions[:, k, :] - positions[:, 0, :]


def contrast_series(predicted, true) -> np.ndarray:
    """Per window, `||true|| - ||predicted - true||`: how much closer the
    probe's displacement is than predicting no displacement at all. `(n,)`.

    Positive means the latent beat staying put. The persistence baseline is
    the ZERO prediction, so its error is `||true||` exactly -- which makes a
    probe that outputs zeros score exactly 0.0, the fixed point the reading is
    read against. Deliberately the same shape as `gap_closed`: a paired,
    per-window, model-against-persistence contrast, so `pooling.paired_contrast`
    reads it unchanged and clusters it on the same episodes.
    """
    predicted = np.asarray(predicted, dtype=float)
    true = np.asarray(true, dtype=float)
    if predicted.shape != true.shape:
        raise ValueError(
            f"predicted {predicted.shape} and true {true.shape} must be the same shape"
        )
    error_persist = np.linalg.norm(true, axis=-1)
    error_model = np.linalg.norm(predicted - true, axis=-1)
    return error_persist - error_model
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_motion.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/eval/motion.py tests/eval/test_motion.py
git commit -m "feat: motion.py -- the pre-registered constants, per-window displacement, and the contrast against staying put

contrast_series is deliberately gap_closed's shape, so pooling.paired_contrast
reads it unchanged on the same episode clusters. A probe that predicts zeros
scores exactly 0.0, which is the fixed point Reading D is read against.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `motion.py` — Reading D

**Files:**
- Modify: `src/mbfps/eval/motion.py` (append)
- Test: `tests/eval/test_motion.py` (append)

**Interfaces:**
- Consumes: `motion_threshold`, `SEEDS_REQUIRED`, `ARMS_REQUIRED` from Task 2.
- Produces: `MotionArm(estimate: float, se: float, z: float, seeds_up: int, seeds_down: int, seeds_total: int)`; `MotionInputs(arms: dict[str, MotionArm], control: dict[str, MotionArm], z_fam: float, k: int, clusters: int)`; `MotionStatus(status: str, rule: str, arms_up: tuple[str, ...], arms_down: tuple[str, ...], leaked: tuple[str, ...])`; `reading_displacement(inputs: MotionInputs) -> MotionStatus`.

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_motion.py`:

```python
from mbfps.eval.motion import (
    MotionArm,
    MotionInputs,
    reading_displacement,
)

Z = 2.5820


def _arm(z, up=3, down=0, total=3):
    """An arm whose estimate carries the sign of its z, so a test that flips a
    z does not leave an estimate contradicting it."""
    return MotionArm(estimate=0.1 * z, se=0.1, z=z, seeds_up=up, seeds_down=down, seeds_total=total)


def _inputs(arms, control=None, k=15):
    clean = {a: _arm(0.4, up=0, down=0) for a in arms}
    return MotionInputs(
        arms=arms, control=control if control is not None else clean,
        z_fam=Z, k=k, clusters=24,
    )


def test_two_arms_clearing_up_in_two_seeds_is_motion_encoded():
    reading = reading_displacement(_inputs({
        "pixel_ae": _arm(3.0, up=2), "frozen_ssl": _arm(4.1, up=3), "random_vit": _arm(1.0, up=1),
    }))
    assert reading.status == "MOTION_ENCODED"
    assert reading.arms_up == ("frozen_ssl", "pixel_ae")
    assert "2 of 3 arms" in reading.rule


def test_one_arm_clearing_up_is_not_enough():
    """ARMS_REQUIRED = 2. A single arm clearing is one cell's worth of
    evidence wearing a family-corrected bar."""
    reading = reading_displacement(_inputs({
        "pixel_ae": _arm(5.0, up=3), "frozen_ssl": _arm(0.2), "random_vit": _arm(0.1),
    }))
    assert reading.status == "NO_DIFFERENCE"


def test_an_arm_clearing_up_on_one_seed_does_not_count():
    """SEEDS_REQUIRED = 2, so a pooled clear carried by a single seed is not
    an arm that cleared."""
    reading = reading_displacement(_inputs({
        "pixel_ae": _arm(4.0, up=1), "frozen_ssl": _arm(4.0, up=1), "random_vit": _arm(0.1),
    }))
    assert reading.status == "NO_DIFFERENCE"


def test_no_arm_up_and_some_arm_down_is_no_motion():
    """The latent is measurably WORSE than staying put -- the result that
    retires M3h section 8's three prior-side levers."""
    reading = reading_displacement(_inputs({
        "pixel_ae": _arm(-3.3, up=0, down=2), "frozen_ssl": _arm(-4.0, up=0, down=3),
        "random_vit": _arm(-0.5, up=0, down=0),
    }))
    assert reading.status == "NO_MOTION"
    assert reading.arms_down == ("frozen_ssl", "pixel_ae")


def test_nothing_clearing_either_way_is_no_difference():
    reading = reading_displacement(_inputs({
        "pixel_ae": _arm(1.0), "frozen_ssl": _arm(-1.2), "random_vit": _arm(0.3),
    }))
    assert reading.status == "NO_DIFFERENCE"


def test_a_leaking_control_suppresses_a_result_that_would_otherwise_pass():
    """Precedence: UNRESOLVED_CONTROL outranks every result. The permuted
    pairing cannot carry signal, so a control that clears means the instrument
    is reading structure that does not exist -- and the reading it would
    otherwise have printed is exactly the one not to trust."""
    arms = {"pixel_ae": _arm(3.0, up=2), "frozen_ssl": _arm(4.1, up=3), "random_vit": _arm(1.0)}
    control = {"pixel_ae": _arm(0.1, up=0), "frozen_ssl": _arm(3.9, up=0), "random_vit": _arm(0.2)}
    reading = reading_displacement(_inputs(arms, control=control))
    assert reading.status == "UNRESOLVED_CONTROL"
    assert reading.leaked == ("frozen_ssl",)
    assert reading.arms_up == (), "a suppressed reading reports no result"


def test_the_control_leaks_on_a_NEGATIVE_clear_too():
    """The control is two-sided: a permuted pairing that is reliably WORSE
    than chance is as much a broken instrument as one that is better."""
    arms = {"pixel_ae": _arm(0.2), "frozen_ssl": _arm(0.1), "random_vit": _arm(0.3)}
    control = {"pixel_ae": _arm(-4.5), "frozen_ssl": _arm(0.1), "random_vit": _arm(0.2)}
    assert reading_displacement(_inputs(arms, control=control)).status == "UNRESOLVED_CONTROL"


def test_the_rule_sentence_names_the_horizon_it_was_decided_at():
    reading = reading_displacement(_inputs({
        "pixel_ae": _arm(0.1), "frozen_ssl": _arm(0.1), "random_vit": _arm(0.1)}, k=15))
    assert "k = 15" in reading.rule
```

- [ ] **Step 2: Run it to watch it fail**

Run: `.venv/bin/python -m pytest tests/eval/test_motion.py -q -k "motion_encoded or no_motion or control or arms or seeds or difference or horizon"`
Expected: `ImportError: cannot import name 'MotionArm'`.

- [ ] **Step 3: Append the reading**

Append to `src/mbfps/eval/motion.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class MotionArm:
    """One arm's pooled contrast at one k, with the seed tallies BOTH ways.

    `seeds_down` is carried beside `seeds_up` because M3h shipped a Reading N
    table that printed only the up tally while its verdict was read from the
    down one: every row said `0/3` under a verdict asserting "3 of 3 arms",
    and the natural misreading was the opposite of the truth.
    """

    estimate: float
    se: float
    z: float
    seeds_up: int
    seeds_down: int
    seeds_total: int

    def clears_up(self, z_fam: float) -> bool:
        return self.z >= z_fam and self.seeds_up >= SEEDS_REQUIRED

    def clears_down(self, z_fam: float) -> bool:
        return self.z <= -z_fam and self.seeds_down >= SEEDS_REQUIRED

    def leaks(self, z_fam: float) -> bool:
        """For a CONTROL arm: cleared the bar in either direction. Two-sided
        on purpose -- a permuted pairing that is reliably worse than chance is
        as much a broken instrument as one that is better, and only the seed
        tallies are ignored here because a control has no result to replicate.
        """
        return abs(self.z) >= z_fam


@dataclass(frozen=True)
class MotionInputs:
    """Everything Reading D is decided on: the real arms, the permuted
    control, the bar, the horizon and the cluster count."""

    arms: dict[str, MotionArm]
    control: dict[str, MotionArm]
    z_fam: float
    k: int
    clusters: int


@dataclass(frozen=True)
class MotionStatus:
    status: str
    rule: str
    arms_up: tuple[str, ...]
    arms_down: tuple[str, ...]
    leaked: tuple[str, ...]


def reading_displacement(inputs: MotionInputs) -> MotionStatus:
    """Does the posterior latent encode displacement (spec 3.2)?

    Precedence, and it is the point: UNRESOLVED_CONTROL outranks every
    result. The control's pairing was permuted, so it cannot carry signal; a
    control that clears means the instrument is reading structure that does
    not exist, and the reading it would otherwise have printed is precisely
    the one not to trust. A suppressed reading reports no arms at all rather
    than reporting them beside a warning nobody reads.
    """
    z_fam = float(inputs.z_fam)
    leaked = tuple(sorted(a for a, arm in inputs.control.items() if arm.leaks(z_fam)))
    if leaked:
        return MotionStatus(
            status="UNRESOLVED_CONTROL",
            rule=(
                f"the permuted control cleared +/-{z_fam:.2f} in {', '.join(leaked)} at "
                f"k = {inputs.k}; the pairing it scores cannot carry signal, so the "
                f"instrument is reading structure that is not there and no reading is taken"
            ),
            arms_up=(), arms_down=(), leaked=leaked,
        )

    up = tuple(sorted(a for a, arm in inputs.arms.items() if arm.clears_up(z_fam)))
    down = tuple(sorted(a for a, arm in inputs.arms.items() if arm.clears_down(z_fam)))

    if len(up) >= ARMS_REQUIRED:
        return MotionStatus(
            status="MOTION_ENCODED",
            rule=(
                f"the latent beats staying put by more than +{z_fam:.2f} in "
                f"{len(up)} of {len(inputs.arms)} arms ({', '.join(up)}) at k = {inputs.k}, "
                f"each in at least {SEEDS_REQUIRED} of its seeds"
            ),
            arms_up=up, arms_down=down, leaked=(),
        )
    if not up and len(down) >= ARMS_REQUIRED:
        return MotionStatus(
            status="NO_MOTION",
            rule=(
                f"no arm beats staying put at k = {inputs.k}, and the latent is WORSE than "
                f"staying put by more than -{z_fam:.2f} in {len(down)} of {len(inputs.arms)} "
                f"arms ({', '.join(down)})"
            ),
            arms_up=(), arms_down=down, leaked=(),
        )
    return MotionStatus(
        status="NO_DIFFERENCE",
        rule=(
            f"no {ARMS_REQUIRED} arms clear +/-{z_fam:.2f} at k = {inputs.k} in at least "
            f"{SEEDS_REQUIRED} seeds each; the latent is indistinguishable from staying put"
        ),
        arms_up=up, arms_down=down, leaked=(),
    )
```

Move the `from dataclasses import dataclass` up into the module's existing import block rather than leaving it mid-file.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_motion.py -q`
Expected: `15 passed`.

- [ ] **Step 5: Mutation-check the precedence**

Temporarily move the `if leaked:` block below the `if len(up) >= ARMS_REQUIRED:` block and run:

Run: `.venv/bin/python -m pytest tests/eval/test_motion.py -q -k "leaking_control"`
Expected: FAIL. Restore the order and confirm it passes again. A precedence that no test can break is a precedence that is not enforced.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/motion.py tests/eval/test_motion.py
git commit -m "feat: Reading D -- MOTION_ENCODED / NO_MOTION / NO_DIFFERENCE, and a control that outranks all three

The permuted control cannot carry signal, so a control that clears means the
instrument is reading structure that is not there -- and the reading it would
otherwise print is exactly the one not to trust. UNRESOLVED_CONTROL therefore
outranks every result and reports no arms at all; mutation-verified by moving
the check below the result branches.

MotionArm carries seeds_down beside seeds_up because M3h shipped a table that
printed only the up tally while its verdict was read from the down one.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `motion.py` — the printed table, and a test that pins each caption to its column

Three defects of exactly one class have shipped on this project: `f24c2f3` (a noise column captioned as something else), `06249d8` (a self-check header saying "exact" over 1.7e-13 deltas) and the final review's C2 (a mean printed under a "medians" caption). Each was found by a human reading output, never by a test. This task adds the test the deferred-minor list has been asking for since M3h Task 9.

**Files:**
- Modify: `src/mbfps/eval/motion.py` (append)
- Test: `tests/eval/test_motion.py` (append)

**Interfaces:**
- Consumes: `MotionInputs`, `MotionStatus` from Task 3.
- Produces: `NOISE_FREE_COLUMNS: tuple[str, ...]` — the reading table's column headers in printed order; `format_reading_displacement(reading: MotionStatus, inputs: MotionInputs) -> str`, ending in a newline.

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_motion.py`:

```python
from mbfps.eval.motion import READING_COLUMNS, format_reading_displacement


def test_every_column_header_is_printed_in_the_declared_order():
    """The caption-column pairing, pinned. Three defects of this exact class
    have shipped on this project and every one was caught by a person reading
    output rather than by a test."""
    arms = {"pixel_ae": _arm(3.0, up=2), "frozen_ssl": _arm(4.1, up=3), "random_vit": _arm(1.0)}
    text = format_reading_displacement(reading_displacement(_inputs(arms)), _inputs(arms))
    header = next(line for line in text.splitlines() if "estimate" in line)
    assert header.split() == list(READING_COLUMNS)


def test_each_arms_row_carries_its_own_numbers_under_those_headers():
    """A swapped column is the defect this pins: the seed tallies differ
    between the arms here, so up/down transposed or an arm's row taking its
    neighbour's numbers both fail."""
    arms = {
        "pixel_ae": _arm(3.0, up=2, down=0),
        "frozen_ssl": _arm(-4.1, up=0, down=3),
        "random_vit": _arm(1.0, up=1, down=0),
    }
    inputs = _inputs(arms)
    text = format_reading_displacement(reading_displacement(inputs), inputs)
    rows = {line.split()[0]: line.split() for line in text.splitlines()
            if line.strip().startswith(("pixel_ae", "frozen_ssl", "random_vit"))}
    cols = list(READING_COLUMNS)
    assert rows["pixel_ae"][cols.index("z")] == "3.00"
    assert rows["pixel_ae"][cols.index("up")] == "2/3"
    assert rows["pixel_ae"][cols.index("dn")] == "0/3"
    assert rows["frozen_ssl"][cols.index("z")] == "-4.10"
    assert rows["frozen_ssl"][cols.index("up")] == "0/3"
    assert rows["frozen_ssl"][cols.index("dn")] == "3/3"


def test_the_table_prints_the_status_and_its_rule():
    arms = {"pixel_ae": _arm(-3.3, up=0, down=2), "frozen_ssl": _arm(-4.0, up=0, down=3),
            "random_vit": _arm(-0.5)}
    inputs = _inputs(arms)
    text = format_reading_displacement(reading_displacement(inputs), inputs)
    assert "NO MOTION" in text
    assert "decided by:" in text
    assert text.endswith("\n")


def test_a_suppressed_reading_prints_the_control_and_no_verdict_row():
    """UNRESOLVED_CONTROL must not print a table a reader could mistake for a
    result."""
    arms = {"pixel_ae": _arm(3.0, up=2), "frozen_ssl": _arm(4.1, up=3), "random_vit": _arm(1.0)}
    control = {"pixel_ae": _arm(0.1), "frozen_ssl": _arm(3.9), "random_vit": _arm(0.2)}
    inputs = _inputs(arms, control=control)
    text = format_reading_displacement(reading_displacement(inputs), inputs)
    assert "UNRESOLVED CONTROL" in text
    assert "MOTION ENCODED" not in text
```

- [ ] **Step 2: Run it to watch it fail**

Run: `.venv/bin/python -m pytest tests/eval/test_motion.py -q -k "column or row or status or suppressed"`
Expected: `ImportError: cannot import name 'READING_COLUMNS'`.

- [ ] **Step 3: Append the formatter**

Append to `src/mbfps/eval/motion.py`:

```python
# The reading table's columns, in printed order. Declared once so a test can
# assert the header against THIS and the rows against the same index -- the
# caption-column pairing that three shipped defects on this project all broke.
READING_COLUMNS: tuple[str, ...] = (
    "arm", "estimate", "se", "z", "up", "dn", "clears",
)


def format_reading_displacement(reading: MotionStatus, inputs: MotionInputs) -> str:
    """Reading D as it is printed and written to `motion.txt`, byte for byte.

    The caption names the reduction and the horizon, because a table whose
    header does not say what its columns hold is how this project has shipped
    a wrong number three times.
    """
    lines = [
        f"--- Reading D: does the latent encode displacement at k = {inputs.k} "
        f"(per-window ||true|| - ||predicted - true||, map units, against staying put); "
        f"z_fam = {inputs.z_fam:.2f} over {inputs.clusters} clusters ---",
        "  " + "".join(
            f"{name:>{width}}" for name, width in zip(
                READING_COLUMNS, (12, 11, 8, 8, 6, 6, 8), strict=True,
            )
        ),
    ]
    for arm in sorted(inputs.arms):
        cell = inputs.arms[arm]
        if cell.clears_up(inputs.z_fam):
            clears = "up"
        elif cell.clears_down(inputs.z_fam):
            clears = "down"
        else:
            clears = "no"
        lines.append(
            f"  {arm:>12}{cell.estimate:>11.4f}{cell.se:>8.4f}{cell.z:>8.2f}"
            f"{f'{cell.seeds_up}/{cell.seeds_total}':>6}"
            f"{f'{cell.seeds_down}/{cell.seeds_total}':>6}{clears:>8}"
        )
    lines.append(
        "  control (permuted pairing, cannot carry signal): "
        + ", ".join(
            f"{arm} z={inputs.control[arm].z:+.2f}" for arm in sorted(inputs.control)
        )
    )
    lines.append(f"  verdict: {reading.status.replace('_', ' ')} -- decided by: {reading.rule}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_motion.py -q`
Expected: `19 passed`.

- [ ] **Step 5: Mutation-check the caption test**

Swap the `up` and `dn` fields in the row f-string, run `-k "each_arms_row"`, and confirm it FAILS. Restore and confirm it passes. A caption test that survives a swapped column pins nothing.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/motion.py tests/eval/test_motion.py
git commit -m "feat: Reading D's table, with the caption-column pairing pinned by a test

READING_COLUMNS is declared once and the test asserts the header against it
and every row by the same index, so a swapped or mislabelled column fails.
Three defects of exactly this class have shipped on this project (f24c2f3,
06249d8, and the final review's C2) and every one was caught by a person
reading output rather than by a test. Mutation-verified by swapping up/dn.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `motion.py` — the descriptive block

Spec §2.3. These decide nothing and carry no threshold: inventing a bar for a quantity nobody has looked at yet is the error spec §1.1 records. They are written into every record so the next milestone reads them from an artefact rather than from a scratch measurement.

**Files:**
- Modify: `src/mbfps/eval/motion.py` (append)
- Test: `tests/eval/test_motion.py` (append)

**Interfaces:**
- Consumes: `entropy_by_group` from `mbfps.eval.stages`.
- Produces: `latent_description(post_logits: np.ndarray, prior_logits: np.ndarray) -> dict` returning keys `entropy_by_group` (list, length G), `entropy_mean` (float), `entropy_max` (float), `live_groups` (float), `top1_posterior` (float), `top1_prior` (float).

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_motion.py`:

```python
from mbfps.eval.motion import latent_description


def _logits(rows):
    """`(1, T, G, C)` from a list of per-step, per-group class indices, with a
    sharp one-hot at each. Sharp on purpose: a hand-typed entropy of 0 and a
    top-1 mass of 1 are values a reader can check without running anything."""
    rows = np.asarray(rows)
    t, g = rows.shape
    out = np.full((1, t, g, 4), -20.0)
    for i in range(t):
        for j in range(g):
            out[0, i, j, rows[i, j]] = 20.0
    return out


def test_a_sharp_latent_has_zero_entropy_and_full_top1_mass():
    post = _logits([[0, 1], [0, 1]])
    d = latent_description(post, post)
    assert d["entropy_mean"] == pytest.approx(0.0, abs=1e-6)
    assert d["top1_posterior"] == pytest.approx(1.0, abs=1e-6)
    assert d["top1_prior"] == pytest.approx(1.0, abs=1e-6)
    assert len(d["entropy_by_group"]) == 2


def test_a_uniform_latent_has_log_C_entropy():
    """Four classes, so the ceiling is log 4 = 1.386 nats -- the number a
    reader compares the real cells against."""
    post = np.zeros((1, 2, 3, 4))
    d = latent_description(post, post)
    assert d["entropy_mean"] == pytest.approx(np.log(4.0), abs=1e-6)
    assert d["entropy_max"] == pytest.approx(np.log(4.0), abs=1e-6)
    assert d["top1_posterior"] == pytest.approx(0.25, abs=1e-6)


def test_live_groups_counts_the_groups_whose_argmax_ever_changes():
    """Group 0 changes class between steps, group 1 never does. A latent whose
    groups are mostly constant is far smaller than G x C in effect, which is
    the whole reason this statistic is recorded."""
    post = _logits([[0, 1], [2, 1], [2, 1]])
    assert latent_description(post, post)["live_groups"] == pytest.approx(1.0)


def test_live_groups_is_zero_for_a_latent_that_never_moves():
    post = _logits([[3, 3], [3, 3]])
    assert latent_description(post, post)["live_groups"] == pytest.approx(0.0)


def test_the_description_reads_the_prior_separately_from_the_posterior():
    """A swapped argument is the defect this catches: the two are different
    distributions here, so transposing them changes both top-1 masses."""
    post = _logits([[0, 0]])
    prior = np.zeros((1, 1, 2, 4))
    d = latent_description(post, prior)
    assert d["top1_posterior"] == pytest.approx(1.0, abs=1e-6)
    assert d["top1_prior"] == pytest.approx(0.25, abs=1e-6)
```

- [ ] **Step 2: Run it to watch it fail**

Run: `.venv/bin/python -m pytest tests/eval/test_motion.py -q -k "sharp_latent or uniform or live_groups or reads_the_prior"`
Expected: `ImportError: cannot import name 'latent_description'`.

- [ ] **Step 3: Append the description**

Append to `src/mbfps/eval/motion.py` (and add `from mbfps.eval.stages import entropy_by_group` to the import block):

```python
def latent_description(post_logits, prior_logits) -> dict:
    """The descriptive block (spec 2.3): what the latent looks like, with no
    threshold attached to any of it.

    Reported beside Reading D and deciding nothing. Every one of these is
    written into the record so the next milestone quotes an artefact rather
    than a scratch measurement -- which is exactly how M3h came to ship a
    premise that was never measured as stated.

    `live_groups` is the mean over windows of how many groups change argmax at
    least once within the window: a latent whose groups are mostly constant is
    far smaller than G x C in effect, however much capacity it nominally has.
    """
    post = np.asarray(post_logits, dtype=float)
    prior = np.asarray(prior_logits, dtype=float)
    if post.ndim != 4 or prior.ndim != 4:
        raise ValueError(
            f"expected (n, steps, groups, classes); got post {post.shape}, prior {prior.shape}"
        )
    entropy = entropy_by_group(post)
    modes = post.argmax(axis=-1)                       # (n, steps, groups)
    changes = (modes[:, 1:, :] != modes[:, :-1, :]).any(axis=1)   # (n, groups)
    return {
        "entropy_by_group": [float(x) for x in entropy],
        "entropy_mean": float(np.mean(entropy)),
        "entropy_max": float(np.log(post.shape[-1])),
        "live_groups": float(changes.sum(axis=-1).mean()),
        "top1_posterior": float(_top1_mass(post)),
        "top1_prior": float(_top1_mass(prior)),
    }


def _top1_mass(logits: np.ndarray) -> float:
    """Mean probability the most likely class in each group carries. 1/C for a
    uniform group, 1.0 for a sharp one."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    return float(probs.max(axis=-1).mean())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_motion.py -q`
Expected: `24 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/eval/motion.py tests/eval/test_motion.py
git commit -m "feat: the descriptive block -- entropy, live groups and top-1 mass, with no threshold on any of it

Reported beside Reading D and deciding nothing; inventing a bar for a quantity
nobody has looked at yet is the error spec 1.1 records. Written into every
record so the next milestone quotes an artefact rather than a scratch
measurement.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: `scripts/latent_motion.py` — the measure phase

**Files:**
- Create: `scripts/latent_motion.py`
- Test: `tests/eval/test_latent_motion_script.py`

**Interfaces:**
- Consumes: everything from `mbfps.eval.motion`; `gather_probe_data`'s `"window"` / `"step"` (Task 1); `trust_horizon.prepare_cell`, `trust_horizon.cell_series`, `trust_horizon.load_cell` via `_sibling`; `probe.fit_probe`, `probe.gather_probe_data`; `pooling.paired_contrast`, `pooling.pool_arm`.
- Produces: `EXIT_CONTROL_LEAKED = 38`; `motion_record_path(out_dir, arm, seed) -> Path`; `fit_displacement_probe(data, k) -> dict`; `apply_displacement_probe(probe, latents) -> np.ndarray`; `permute_pairing(n, seed) -> np.ndarray`; `measure_cell(args, cell, device, train, val) -> tuple[int, dict | None]`; `measure_phase(args) -> int`.

- [ ] **Step 1: Write the failing test for the permutation**

Create `tests/eval/test_latent_motion_script.py` with the module loaded by path, the way `tests/eval/test_sharper_latent_script.py` loads its script — follow that file's existing spelling for the loader and the fixtures rather than inventing new ones. First tests:

```python
def test_the_control_permutation_pairs_no_window_with_its_own_displacement():
    """The control must destroy the latent->displacement pairing. A
    permutation that leaves any window paired with its own row is a control
    that still carries that window's signal."""
    order = script.permute_pairing(200, seed=script.CONTROL_SEED)
    assert sorted(order.tolist()) == list(range(200)), "must be a permutation, not a resample"
    assert not (order == np.arange(200)).any(), "no window may keep its own displacement"


def test_the_control_permutation_is_reproducible():
    """A control that draws a fresh permutation each run is a control whose
    refusal cannot be repeated."""
    a = script.permute_pairing(64, seed=script.CONTROL_SEED)
    b = script.permute_pairing(64, seed=script.CONTROL_SEED)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, script.permute_pairing(64, seed=script.CONTROL_SEED + 1))


def test_the_permutation_refuses_a_window_count_it_cannot_derange():
    """A single window cannot be paired with anything but itself."""
    with pytest.raises(ValueError, match="at least 2"):
        script.permute_pairing(1, seed=0)


def test_a_probe_fit_on_exact_displacement_recovers_it():
    """Known answer: latents that linearly encode the displacement must be
    read back at essentially zero error, so a positive contrast is possible at
    all. Without this the reading could only ever fail."""
    rng = np.random.default_rng(0)
    latents = rng.normal(size=(64, 8))
    true = latents[:, :2] * 3.0
    probe = script.fit_displacement_probe(
        {"latent": latents, "displacement": true}, k=1
    )
    predicted = script.apply_displacement_probe(probe, latents)
    assert np.abs(predicted - true).max() < 1e-6


def test_a_probe_fit_on_noise_does_not_beat_staying_put():
    """The other known answer. Latents independent of the displacement must
    not produce a positive contrast on held-out rows."""
    from mbfps.eval.motion import contrast_series

    rng = np.random.default_rng(1)
    latents = rng.normal(size=(256, 8))
    true = rng.normal(size=(256, 2)) * 10.0
    probe = script.fit_displacement_probe(
        {"latent": latents[:128], "displacement": true[:128]}, k=1
    )
    held_out = script.apply_displacement_probe(probe, latents[128:])
    assert contrast_series(held_out, true[128:]).mean() <= 0.0
```

- [ ] **Step 2: Run it to watch it fail**

Run: `.venv/bin/python -m pytest tests/eval/test_latent_motion_script.py -q`
Expected: collection error — `scripts/latent_motion.py` does not exist.

- [ ] **Step 3: Write the script's measure half**

Create `scripts/latent_motion.py`. Model its header, `_sibling` loader, argparse block and phase dispatch on `scripts/sharper_latent.py` — read that file first and follow it, including the `_sibling` trap it documents (each importer gets a DISTINCT class object, so never `except` another module's copy).

The pieces this task adds:

```python
EXIT_CONTROL_LEAKED = 38
"""The permuted control cleared the bar. The pairing it scores cannot carry
signal, so a control that clears means the instrument is reading structure
that is not there -- and no reading is taken."""


def motion_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    """One file per cell, named by BOTH the arm and the seed -- see
    `study.job_record_path` for what a colliding name costs."""
    return Path(out_dir) / f"motion_{arm}_seed{seed}.json"


def permute_pairing(n: int, seed: int) -> np.ndarray:
    """A DERANGEMENT of `n` rows: every window is paired with some other
    window's displacement, never its own.

    A plain shuffle would leave roughly one window in e paired with itself,
    each contributing real signal to a control whose whole claim is that it
    has none. Fixed seed so a refusal can be reproduced.
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"a derangement needs at least 2 rows, got {n}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fixed = np.flatnonzero(order == np.arange(n))
    for i in fixed:
        j = (i + 1) % n
        order[i], order[j] = order[j], order[i]
    return order


def fit_displacement_probe(data: dict, k: int) -> dict:
    """Ridge from the posterior latent to the displacement, in `fit_probe`'s
    own shape so the two probes are fit and applied the same way.

    `data` carries `"latent"` `(N, LATENT)` and `"displacement"` `(N, 2)`,
    row-aligned. `k` is recorded on the probe so a probe fit at one horizon
    cannot be applied at another in silence.
    """
    probe = fit_probe(np.asarray(data["latent"]), np.asarray(data["displacement"]))
    return {**probe, "k": int(k)}


def apply_displacement_probe(probe: dict, latents) -> np.ndarray:
    """The probe's predicted displacement, `(n, 2)`."""
    return apply_probe(probe, np.asarray(latents))
```

`fit_probe` and `apply_probe` are `mbfps.eval.probe`'s, verified: `fit_probe(latents, targets, val_latents=None, val_targets=None, ridge=None) -> dict` returns the weights beside the standardisation because both are needed to apply it, and `apply_probe(probe, latents) -> np.ndarray` consumes exactly that dict. `ARMS` is `mbfps.utils.config`'s and `SEEDS` is `mbfps.eval.aggregate`'s; re-export them on the script the way `sharper_latent.py` does rather than re-spelling either.

`measure_cell` then, per cell:

1. `status, prepared = _trust.prepare_cell(args, cell, device, train, val)`; return `(status, None)` on anything but `EXIT_OK`.
2. Gather TRAINING rows with `gather_probe_data(model, train, backbone, device, context=prepared.context, horizon=prepared.horizon, seed=cell.seed)`, and VALIDATION rows the same way on `val`.
3. For each `k` in `K_REPORTED`: reshape each split's rows by `"window"` / `"step"` into `(n_windows, steps, …)`, take `positions = targets[..., :2]`, compute `displacement(positions, k)`, fit on the training split, apply to the validation split, and build `contrast_series(predicted, true)`.
4. Build the control at the same `k`: `order = permute_pairing(n_val, CONTROL_SEED)` and contrast `apply_displacement_probe(probe, val_latents)` against `true[order]`.
5. Write the record: `arm`, `seed`, `git_sha`, `device`, `torch_version`, `context`, `horizon`, `step`, `split_seed`, `episodes`, `windows` (total and episode index, copied from the cell's record so `cell_series` can cluster), `self_check` from `prepared`, a `k` block per reported horizon holding `contrast` and `control` per-window series, and `description` from `latent_description`.

Write the record through `study.write_record` — the strict-JSON writer that turns a NaN into `null` plus its token rather than a bare `NaN` the pooling reader would choke on.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_latent_motion_script.py -q`
Expected: the five tests above pass.

- [ ] **Step 5: Mutation-check the derangement**

Replace the fixed-point repair loop with a plain `rng.permutation(n)` and run `-k "no_window_with_its_own"`. Expected: FAIL. Restore and confirm it passes.

- [ ] **Step 6: Commit**

```bash
git add scripts/latent_motion.py tests/eval/test_latent_motion_script.py
git commit -m "feat: latent_motion.py measure -- the displacement probe, and a control that is a derangement rather than a shuffle

A plain shuffle leaves roughly one window in e paired with its own
displacement, each contributing real signal to a control whose whole claim is
that it has none. Fixed seed so a refusal can be reproduced.

Two known-answer tests bound the probe from both sides: latents that encode
the displacement exactly are read back at <1e-6, and latents independent of it
do not beat staying put on held-out rows. Without the first the reading could
only ever fail.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: `scripts/latent_motion.py` — the read phase, exit 38, and the registry

**Files:**
- Modify: `scripts/latent_motion.py`
- Modify: `tests/eval/test_diagnose_dynamics_script.py` (the exit-status registry test it holds — it already lists `stage_decomposition` and `sharper_latent`)
- Test: `tests/eval/test_latent_motion_script.py` (append)

**Interfaces:**
- Consumes: `motion_record_path`, `EXIT_CONTROL_LEAKED` (Task 6); `reading_displacement`, `format_reading_displacement`, `motion_threshold`, `latent_description` (Tasks 3–5); `cell_series` from `trust_horizon` via `_sibling`.
- Produces: `motion_inputs(records, k) -> MotionInputs`; `read_phase(args) -> int`; `motion.txt` in `args.out`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_latent_motion_script.py`:

```python
def test_the_reading_is_built_from_the_records_treatment_and_control(measured):
    """The orientation, pinned by value. A swapped treatment and control would
    invert the verdict, and no shape check would notice."""
    records = script.load_motion(measured.out, script.ARMS, script.SEEDS)
    inputs = script.motion_inputs(records, k=15)
    assert set(inputs.arms) == set(script.ARMS)
    assert set(inputs.control) == set(script.ARMS)
    assert inputs.k == 15
    assert inputs.z_fam == pytest.approx(script.motion_threshold(inputs.clusters))
    # The control is the PERMUTED series, so it must differ from the treatment.
    for arm in inputs.arms:
        assert inputs.arms[arm].z != inputs.control[arm].z


def test_a_leaking_control_is_exit_38_and_writes_no_reading(measured, capsys):
    """The gate, in code. Doctoring the control to clear the bar must refuse
    before any verdict is printed."""
    for arm, seed in [(a, s) for a in script.ARMS for s in script.SEEDS]:
        path = script.motion_record_path(measured.out, arm, seed)
        record = load_record(path)
        entry = record["k"]["k15"]
        entry["control"] = [x + 500.0 for x in entry["contrast"]]
        write_record(path, record)
    assert script.main(_argv(measured, "--phase", "read")) == script.EXIT_CONTROL_LEAKED
    out = capsys.readouterr().out
    assert "UNRESOLVED CONTROL" in out
    assert "MOTION ENCODED" not in out and "NO MOTION" not in out
    assert not (measured.out / "motion.txt").exists()


def test_read_writes_motion_txt_byte_identical_to_what_it_printed(measured, capsys):
    assert script.main(_argv(measured, "--phase", "read")) == script.EXIT_OK
    printed = capsys.readouterr().out
    written = (measured.out / "motion.txt").read_text()
    assert written == printed, "motion.txt must be what the reader saw"


def test_the_reading_is_decided_at_DECISION_H_and_reports_the_others(measured, capsys):
    """Only k = 15 decides. The other horizons are printed and decide nothing,
    exactly as the M3 gate is."""
    script.main(_argv(measured, "--phase", "read"))
    out = capsys.readouterr().out
    assert "Reading D: does the latent encode displacement at k = 15" in out
    for k in (1, 5, 30, 45):
        assert f"k = {k}" in out or f"k{k}" in out
```

And in `tests/eval/test_diagnose_dynamics_script.py`, extend the registry test's tuples to include `"latent_motion"` beside `"stage_decomposition"` and `"sharper_latent"`, and assert its statuses:

```python
    motion = statuses("latent_motion")
    assert motion["EXIT_CONTROL_LEAKED"] == 38
    # Inherited from trust_horizon, and identical to it.
    for name in ("EXIT_OK", "EXIT_NO_CHECKPOINTS", "EXIT_SPLIT_MISMATCH",
                 "EXIT_RECORD_MISMATCH", "EXIT_SELF_CHECK_FAILED"):
        assert motion[name] == statuses("trust_horizon")[name]
    # 38 is new: no other tool may already use it.
    assert 38 not in {v for tool in OTHER_TOOLS for v in statuses(tool).values()}
```

Read the existing registry test before editing — use ITS helper names (`statuses`, the tool tuples) rather than the placeholders above.

- [ ] **Step 2: Run them to watch them fail**

Run: `.venv/bin/python -m pytest tests/eval/test_latent_motion_script.py tests/eval/test_diagnose_dynamics_script.py -q -k "reading or exit_38 or motion_txt or DECISION_H or registry"`
Expected: `AttributeError: module has no attribute 'motion_inputs'`.

- [ ] **Step 3: Write the read phase**

In `scripts/latent_motion.py`:

```python
def motion_inputs(records: dict, k: int) -> MotionInputs:
    """The per-cell records reduced to what Reading D is decided on.

    Treatment and control are built the SAME way through `cell_series` and
    `pool_arm`, differing only in which per-window series goes in -- so the
    control is a real control and not a differently-computed number that
    happens to be called one.
    """
    key = f"k{int(k)}"
    arms: dict[str, MotionArm] = {}
    control: dict[str, MotionArm] = {}
    clusters = 0
    for arm in sorted({a for a, _ in records}):
        treat_cells, ctrl_cells = [], []
        for (a, seed), record in sorted(records.items()):
            if a != arm:
                continue
            entry = record["k"][key]
            changed = np.ones(len(entry["contrast"]), dtype=bool)
            treat_cells.append(_trust.cell_series(
                arm, seed, "displacement", np.asarray(entry["contrast"], float), changed, record))
            ctrl_cells.append(_trust.cell_series(
                arm, seed, "control", np.asarray(entry["control"], float), changed, record))
        treat, ctrl = pooling.pool_arm(treat_cells), pooling.pool_arm(ctrl_cells)
        clusters = int(treat.clusters)
        arms[arm] = _arm_from_pool(treat, treat_cells)
        control[arm] = _arm_from_pool(ctrl, ctrl_cells)
    return MotionInputs(
        arms=arms, control=control, z_fam=motion_threshold(clusters), k=int(k), clusters=clusters,
    )
```

`_arm_from_pool(pooled, cells)` builds a `MotionArm` from the pooled mean and its clustered SE, and counts `seeds_up` / `seeds_down` as the number of that arm's cells whose own per-seed z clears `±z_fam`. Write it beside `motion_inputs`, and read `trust_horizon`'s `_pooled_mean` and `_contrast` first for the per-seed idiom this project already uses.

`read_phase` then:

1. Load every record; missing one is exit 11 naming the first.
2. `inputs = motion_inputs(records, DECISION_H)`.
3. `reading = reading_displacement(inputs)`.
4. Print the self-check table, the descriptive table, the per-k table, then `format_reading_displacement(reading, inputs)`.
5. **If `reading.status == "UNRESOLVED_CONTROL"`: print, return `EXIT_CONTROL_LEAKED`, and write no `motion.txt`.** A suppressed reading must not leave an artefact a later reader mistakes for a result.
6. Otherwise write `motion.txt` byte-identical to what was printed and return `EXIT_OK`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_latent_motion_script.py tests/eval/test_diagnose_dynamics_script.py -q`
Expected: all pass.

- [ ] **Step 5: Mutation-check the gate and the orientation**

Two mutants, each restored after:
- Return `EXIT_OK` instead of `EXIT_CONTROL_LEAKED` on a leak → `-k "exit_38"` must FAIL.
- Swap `arms=` and `control=` in the `MotionInputs(...)` construction → `-k "treatment_and_control"` must FAIL.

- [ ] **Step 6: Commit**

```bash
git add scripts/latent_motion.py tests/eval/test_latent_motion_script.py tests/eval/test_diagnose_dynamics_script.py
git commit -m "feat: latent_motion.py read -- Reading D at DECISION_H, the control gate in code, and the registry entry

Treatment and control are built the same way through cell_series and pool_arm,
differing only in which per-window series goes in, so the control is a real
control rather than a differently-computed number that happens to be called
one. A leaking control returns 38 and writes NO motion.txt: a suppressed
reading must not leave an artefact a later reader mistakes for a result.
Mutation-verified on both the gate and the treatment/control orientation.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: The smoke, the run, and the results

**Files:**
- Run artefacts (gitignored): `runs/m3i_smoke/`, then `runs/m3i_motion/`
- Modify: this plan (`## Task 8 results`)

- [ ] **Step 1: Check the disk before the suite**

```bash
du -sh "$TMPDIR/pytest-of-$USER" 2>/dev/null; df -h /System/Volumes/Data | tail -1
```

The suite writes heavily even with `tmp_path_retention_policy = "failed"`. If free space is under ~30 GB, clear stale roots with `rm -rf "$TMPDIR/pytest-of-$USER"/pytest-*` while no pytest is running. This is pytest scratch in the system temp area, NOT under `runs/`.

- [ ] **Step 2: Whole suite, caffeinated**

```bash
nohup caffeinate -dimsu .venv/bin/python -m pytest -q --tb=line > .superpowers/sdd/suite-m3i.txt 2>&1 &
```

Expected: 0 failures, 0 errors, 0 warnings. **Must be caffeinated and run with the lid open.** An uncaffeinated overnight suite and a disk-full suite have both failed on this machine; the disk is the cause that was proven, and the failure prints `OSError: <n> requested and 0 written` from inside a fixture with no summary line, because pytest then crashes in its own report formatter. `pytest -x` is what gets the real traceback out.

- [ ] **Step 3: Smoke one cell**

```bash
mkdir -p runs/m3i_smoke && PYTHONUNBUFFERED=1 caffeinate -dimsu .venv/bin/python scripts/latent_motion.py \
  --phase all --out runs/m3i_smoke --arms pixel_ae --seeds 0 2>&1 | tail -40
```

Then **read the output**, not just its exit code. M3h's smoke caught a reporting bug the whole suite had missed, and the final review caught two more of the same class. Check specifically: every column holds what its header says; the self-check deltas are inside the reproduction bound; the control's z is printed and is small; `live_groups` is plausible against 32; the per-k table's horizons are the ones asked for. Fix anything wrong and re-smoke before the real run.

- [ ] **Step 4: The run**

```bash
git status --porcelain && git log --oneline -1   # must be clean; nothing is committed while a run is in progress
mkdir -p runs/m3i_motion && git rev-parse HEAD > runs/m3i_motion/motion.head && \
date -u +%Y-%m-%dT%H:%M:%SZ > runs/m3i_motion/motion.started && \
nohup sh -c 'PYTHONUNBUFFERED=1 caffeinate -dimsu .venv/bin/python scripts/latent_motion.py --phase measure --out runs/m3i_motion > runs/m3i_motion/measure.log 2>&1; echo $? > runs/m3i_motion/motion.exit; date -u +%Y-%m-%dT%H:%M:%SZ > runs/m3i_motion/motion.finished' > /dev/null 2>&1 &
```

Nine cells, one `observe` pass and the probe fits each: **~1 h**. Wait on the PROCESS, not on `motion.exit` — that file may already exist from an earlier attempt, and `until [ -f ... ]` then returns instantly:

```bash
until ! pgrep -f "latent_motion[.]py" > /dev/null; do sleep 60; done
```

- [ ] **Step 5: Acceptance**

```bash
.venv/bin/python - <<'EOF'
import glob, json
from pathlib import Path
head = Path("runs/m3i_motion/motion.head").read_text().strip()
records = [json.load(open(p)) for p in sorted(glob.glob("runs/m3i_motion/motion_*.json"))]
assert len(records) == 9, len(records)
assert {r["git_sha"] for r in records} == {head}
assert {r["device"] for r in records} == {"mps"}
assert all(r["self_check"]["ok"] for r in records)
assert {(r["windows"]["total"], r["windows"]["clusters"]) for r in records} == {(229, 24)}
assert {r["step"] for r in records} == {20000}
assert all(set(r["k"]) == {"k1", "k5", "k15", "k30", "k45"} for r in records)
print("OK: nine records at one git_sha ==", head[:7], "on mps; 229/24; five horizons each")
EOF
.venv/bin/python scripts/latent_motion.py --phase read --out runs/m3i_motion 2>&1 | tail -40
```

- [ ] **Step 6: Write `## Task 8 results` into this plan**

Numbers copied from `motion.txt` and the records, not rounded and not paraphrased:

1. **Provenance** — `git_sha`, device, torch, `.started` / `.finished` and wall time, `motion.exit`, and that `motion.txt` is byte-identical to what `read` printed.
2. **Suite** — the whole-suite count at the launched HEAD, 0 skipped, 0 warnings.
3. **Acceptance** — Step 5's output.
4. **The self-check** — the reproduction bound's deltas on 9/9, in ULPs beside map units.
5. **The control** — every arm's z, and that it did not leak. If it did, that is the whole result: the instrument is untrustworthy and nothing else in this section may be read.
6. **Reading D** — the whole table: per arm the estimate ± se, z, both seed tallies, whether it clears; the status and its rule sentence.
7. **The other horizons** — k = 1, 5, 30, 45, stated as reported and deciding nothing.
8. **The descriptive block** — per-group entropy against the `log 32 = 3.466` ceiling, live groups against 32, top-1 mass for posterior and prior, and `information` beside M3g's recorded 0.290 / 0.467 / 0.489.
9. **`pixel_ae`/s1** — whether Reading D splits along the cell whose recorded `latent_selection_r2` is −0.008 (spec §3.4).
10. **Read against §3.3 and §4** — one paragraph per arm's status; what §3.3 pre-registered as following from it, applied without renegotiation; and the non-claims restated as measured, including that a linear probe bounds only what a *linear* read-out can exploit.

Before committing this section, check every sentence against the artefacts. M3h shipped five unsupported sentences in its results; four were caught by self-review and one by the final reviewer. The classes that slipped: a count that was arithmetically wrong, a monotonicity the table two lines below falsified, a cell credited with its arm's pooled figure, and a statistic quoted from the wrong run.

- [ ] **Step 7: Commit the results**

```bash
git add docs/superpowers/plans/2026-09-26-mb-fps-m3i-latent-motion.md
git commit -m "docs: M3i -- <Reading D's status and what it commits the project to>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Exit criteria

- [ ] Tasks 1–7 committed, each reviewed; the whole suite green with 0 warnings at the launched HEAD.
- [ ] The smoke run's output READ for format, any defect fixed before the real run.
- [ ] Nine records at one `git_sha` == HEAD on `mps`; 229 windows / 24 clusters; self-check within the reproduction bound on 9/9; five horizons in every record; `motion.exit` 0.
- [ ] The permuted control did not leak — or, if it did, exit 38 is the result and no reading is reported.
- [ ] Reading D read and recorded, and §3.3's pre-registered consequence applied as written rather than renegotiated after the numbers.
- [ ] `## Task 8 results` written, every number copied from an artefact, every sentence checked against it.

## Plan self-review (writing-plans)

- **Spec coverage.** §2 protocol → Tasks 6, 8. §2.1 the displacement probe → Tasks 1 (the indices it needs), 2 (the statistic), 6 (the fit). §2.2 the control → Tasks 3 (precedence), 6 (the derangement), 7 (the gate). §2.3 the descriptive family → Task 5. §3.1 pooling → Task 7's `motion_inputs`. §3.2 Reading D → Tasks 3, 4. §3.3 the pre-registered consequence → Task 8 Step 6 item 10. §3.4 `pixel_ae`/s1 → Task 8 Step 6 item 9. §4 non-claims → Task 8 Step 6 item 10. §5 code shape → Tasks 2–7 as named; the registry → Task 7. §6 the run → Task 8. §7 files → the File Structure table.
- **Placeholders.** None: every code step carries its code, and the three places that say "read the existing file first" name what to read and why (the `_sibling` class trap, the registry test's own helper names, `fit_probe`'s real signature) rather than deferring a decision.
- **Type consistency.** `MotionArm(estimate, se, z, seeds_up, seeds_down, seeds_total)` (Task 3) is what Task 7's `_arm_from_pool` builds and Task 4's formatter reads; `MotionInputs(arms, control, z_fam, k, clusters)` is built only in Task 7 and consumed in Tasks 3 and 4; `MotionStatus(status, rule, arms_up, arms_down, leaked)` is produced in Task 3 and read in Task 4; `contrast_series` and `displacement` (Task 2) are called only in Task 6; `READING_COLUMNS` (Task 4) is the single spelling of the table's headers; `EXIT_CONTROL_LEAKED = 38` (Task 6) is asserted in Task 7's registry test; `motion_record_path` (Task 6) is used by Task 7's tests; `"window"` / `"step"` (Task 1) are consumed only in Task 6 Step 3 item 3.
