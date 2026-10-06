# M3o Probe Ceiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the readout error that made M3n unreadable is a limit of the linear probe, of the representations, or of the observation itself.

**Architecture:** A 2x2 of {linear ridge, kernel ridge RBF} x {raw pixels, encoder embedding} plus a sequence row, every cell reported in both r2 and median map-unit error, read through episode-clustered bootstrap intervals. No training: every input is frozen on disk. A prerequisite fixes `probe._solve`, which cannot run at the raw-pixel feature dimension.

**Tech Stack:** Python 3, numpy, torch, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-05-mb-fps-m3o-probe-ceiling-design.md`

## Global Constraints

- Test command is `.venv/bin/python -m pytest`. There is no `pytest` entry point in the venv.
- Commit trailer is the literal `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` in its own paragraph, on every commit, regardless of which model writes it.
- **Never** `git stash` — the stash stack is shared across worktrees and other sessions use it. Use a temporary WIP commit instead.
- **Never** `rm` anything under `runs/`. It is a symlink to a directory outside every worktree, shared by all of them. `runs/m3n_motion/` (9 records) and `runs/m3m_burden*/` (18) must survive this milestone untouched.
- **Never** `git clean -fdx` — it destroys the session ledger at `.superpowers/sdd/`.
- After mutating a source file to check a test bites, clear `__pycache__` under `src/`, `tests/` and `scripts/` before re-running.
- **When you pipe pytest through `tail` or `head`, the pipeline's exit code is the pager's, not pytest's.** Always read the `N passed` / `N failed` summary line yourself. This milestone's predecessor lost a review cycle to a run that reported exit 0 over `1 failed, 474 passed`.
- `src/mbfps/eval/rollout.py`, `src/mbfps/eval/diagnostics.py`, `src/mbfps/eval/headroom.py`, `src/mbfps/eval/burden.py`, `scripts/trust_horizon.py`, `scripts/motion_headroom.py` are **unchanged by Tasks 1-6**. Only the conditional Task 7 may read them, and it modifies none. Verify with `git diff --stat 4956f59 HEAD -- <paths>` at the end of every task.
- Exact constants, copied verbatim: `FRAME_BUDGET = 12000`, `CONFIDENCE = 0.95`, `RESAMPLES = 2000`, `SEEDS_MINIMUM = 3`, `ARMS_REQUIRED = 2`, `ZERO_INFORMATION_MEDIAN = 283.05`, `DISPLACEMENT_TARGET = 3.9694722203504225`, `EXIT_HARNESS_DISAGREES = 49`, `EXIT_NO_MAJORITY = 50`. `RIDGES` is inherited from `probe.py`, not redefined.
- The strict-majority bar is **computed**, never stored: `burden.strict_majority(len(cells))`. It returns 5 at nine cells and 6 at eleven.
- Every difference is read from an episode-clustered bootstrap interval via `probe._block_bootstrap_ci`, at a per-cell seed derived as M3n derives it — `zlib.crc32` of base, arm and seed. **Never `hash()`**: it is salted per process, so records would not reproduce.

### Test pre-commitments (every task)

Each is a defect class this arc has shipped. They are requirements, not advice. In M3n, **eleven** of the plan's draft fixtures were wrong — two failed against correct code, five let real mutations survive, two would have broken the GPU run. Treat this plan's test code as a draft to verify, not as ground truth; if a fixture will not produce what its test expects, fix the fixture and report what you changed.

1. **Fixtures must be shown to discriminate** — assert the fixture produces what the assertion needs before asserting anything about the result.
2. **No self-referential expectations** — an expected value never comes from the thing under test.
3. **Stochastic fixtures where determinism would hide a bug** — a deterministic fake leaves `seed` unbound and a seed mutation reads 0.0.
4. **Every test calls the code under test.**
5. **Docstrings name only mutations the fixture can reach.**
6. **Exact equality where exactness is load-bearing** — a relative tolerance accepted a `sum`-for-`max` mutant in M3m.

---

## File Structure

| file | change | responsibility |
| --- | --- | --- |
| `src/mbfps/eval/probe.py` | modify | `_solve` gains the dual path; `_fit_and_score` gains an `estimator` parameter |
| `src/mbfps/eval/ceiling.py` | create | the two new estimators, `apply_readout`, the grid, the reading, the formatter |
| `scripts/probe_ceiling.py` | create | measure / read / all phases, record schema, exits 49 and 50 |
| `tests/eval/test_probe.py` | modify | the dual path, at every ridge |
| `tests/eval/test_ceiling.py` | create | the estimators, the grid, the reading |
| `tests/eval/test_probe_ceiling_script.py` | create | the phases, the schema, the exits |
| `tests/eval/test_diagnose_dynamics_script.py` | modify | exit registry gains `{49, 50}` |

---

## Task 1: `probe._solve` solves in whichever space is smaller

`_solve` materialises `ridge * np.eye(D + 1)`. At `D = 37632` — a 112x112x3 frame — that one array is 11.3 GB, `design.T @ design` is another, and their sum another; `fit_probe` calls `_solve` once per rung of `RIDGES`, five times. A first attempt at the raw-pixel cell took 18 GB of swap.

This is **not a latent bug**. Every existing caller probes at `D ~1536-2048`, where `D^2` is 33 MB. M3o's raw-pixel cell is the first where `D` exceeds `n`.

**Files:**
- Modify: `src/mbfps/eval/probe.py:59-63` (`_solve`), `:86-97` (`fit_probe`'s ridge loop)
- Test: `tests/eval/test_probe.py`

**Interfaces:**
- Consumes: nothing
- Produces: `_solve(x, y, ridge)` unchanged in signature and return value. `fit_probe` unchanged in signature and return dict.

- [ ] **Step 1: Write the failing test**

Add to `tests/eval/test_probe.py`:

```python
def test_the_dual_and_primal_solves_agree_at_every_rung_of_the_ladder():
    """The reformulation is exact, so it is pinned at EVERY ridge, not one.

    Agreement is worst at the smallest ridge -- measured 5.164e-09 at 1e-1
    against 3.553e-15 at 1e7 -- because that is where the Gram is worst
    conditioned. A test at a single representative rung would pass against a
    dual path that is wrong precisely where it is hardest.
    """
    rng = np.random.default_rng(0)
    n, d = 40, 300                      # n < d, so the dual path is the one taken
    x = rng.normal(size=(n, d))
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    y = rng.normal(size=(n, 4)) * 240.0
    assert n < d, "the fixture must exercise the dual branch"
    assert np.ptp(y) > 100.0, "targets must span a real range, as Doom coordinates do"

    query = rng.normal(size=(17, d))
    design = np.concatenate([query, np.ones((17, 1))], axis=1)
    for ridge in RIDGES:
        dual = design @ _solve(x, y, ridge)
        primal = design @ _primal_solve_for_test(x, y, ridge)
        assert np.max(np.abs(dual - primal)) < 1e-7, (
            f"ridge={ridge:g}: {np.max(np.abs(dual - primal)):.3e}"
        )


def test_the_solve_takes_the_primal_path_when_the_features_are_fewer_than_the_rows():
    """The switch is on `min(n, D)`, not "always dual".

    The dual's cost is n^2, so it is the WRONG form at 12,000 rows and 2,048
    features -- which is every existing caller in this repo. The branch is
    pinned by its result, not by inspecting which was taken: both forms give
    the same answer, so a wrong branch is a performance bug, and the guard is
    the shape assertion below plus the agreement test above.
    """
    rng = np.random.default_rng(1)
    n, d = 300, 40                      # n > d, so the primal path is the one taken
    x = (lambda a: (a - a.mean(0)) / (a.std(0) + 1e-8))(rng.normal(size=(n, d)))
    y = rng.normal(size=(n, 4)) * 240.0
    w = _solve(x, y, 1e3)
    assert w.shape == (d + 1, 4), "an intercept row is appended in both forms"
    assert np.all(np.isfinite(w))


def test_fit_probe_is_unchanged_on_the_shape_every_existing_caller_uses():
    """A golden check against the shipped behaviour at latent scale.

    The expected values come from the PRIMAL form computed inline here, not
    from `fit_probe`, so a change inside it cannot move both sides.
    """
    rng = np.random.default_rng(2)
    xf, yf = rng.normal(size=(120, 48)), rng.normal(size=(120, 4)) * 240.0
    xv, yv = rng.normal(size=(60, 48)), rng.normal(size=(60, 4)) * 240.0
    probe = fit_probe(xf, yf, xv, yv)

    mean, scale = xf.mean(0), xf.std(0) + 1e-8
    xs = (xf - mean) / scale
    expected = _primal_solve_for_test(xs, yf, probe["ridge"])
    assert np.max(np.abs(probe["w"] - expected)) < 1e-8
    assert probe["ridge"] in RIDGES
```

Add this helper to the same test module — it is the OLD implementation, kept as the oracle:

```python
def _primal_solve_for_test(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """`probe._solve` as it stood before Task 1, kept as the oracle.

    An independent implementation, not a call into the module under test, so
    the agreement assertions above cannot be satisfied by both sides moving
    together -- the defect class this project has hit twice.
    """
    design = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    penalty = ridge * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)
```

Extend the module's existing import from `mbfps.eval.probe` with `_solve` and `RIDGES`.

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/eval/test_probe.py -k "dual_and_primal or primal_path or unchanged_on_the_shape" -v
```

Expected: the agreement test fails on MEMORY or passes trivially (both call the same primal code). Record which. The real red signal is Step 5's mutation.

- [ ] **Step 3: Write the implementation**

Replace `_solve` in `src/mbfps/eval/probe.py`:

```python
def _solve(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Ridge weights with an UNREGULARISED intercept, solved in the smaller space.

    Two algebraically identical forms:

        primal (D x D):  w = (X'X + lambda I_D)^-1 X'y
        dual   (n x n):  w = X'(XX' + lambda I_n)^-1 y

    The primal is natural at latent scale, where D is 1536-2048 and D^2 is
    33 MB. It is unusable on raw frames: at D = 37632 the `np.eye` alone is
    11.3 GB, and `fit_probe` calls this once per rung of `RIDGES`.

    The dual's cost is n^2, so neither form is universally right and the
    branch is on `min(n, D)`. Verified identical across the whole ladder:
    5.164e-09 at ridge 1e-1, 3.553e-15 at 1e7. Agreement is worst at the
    smallest ridge, where the Gram is worst conditioned.

    THE INTERCEPT IS NEVER REGULARISED, in either form. `fit_probe`
    standardises `x` to zero column means on the fit split, so the dual is
    taken on already-centred features and the intercept is exactly `y.mean(0)`
    -- not an approximation.
    """
    n, d = x.shape
    if d < n:
        design = np.concatenate([x, np.ones((n, 1))], axis=1)
        gram = design.T @ design
        gram.flat[:: gram.shape[0] + 1] += ridge      # in place; no np.eye
        gram[-1, -1] -= ridge                         # the intercept stays free
        return np.linalg.solve(gram, design.T @ y)
    centre = y.mean(0)
    gram = x @ x.T
    gram.flat[:: gram.shape[0] + 1] += ridge
    weights = x.T @ np.linalg.solve(gram, y - centre)
    return np.concatenate([weights, centre[None, :]], axis=0)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/eval/test_probe.py -q
```

Expected: PASS, and report the `N passed` line.

- [ ] **Step 5: Verify the tests bite**

Commit first, so `git checkout` restores your work rather than destroying it. Then apply each mutation, clear `__pycache__`, run, confirm the named failure, restore:

| mutation | test that must fail |
| --- | --- |
| dual branch: drop `- centre` from `y - centre` | `test_the_dual_and_primal_solves_agree_at_every_rung_of_the_ladder` |
| dual branch: return `weights` without the intercept row | the agreement test (shape) |
| primal branch: drop `gram[-1, -1] -= ridge` | the agreement test, at the largest ridges |
| branch condition `d < n` -> `d > n` | neither — **report this honestly.** Both forms are correct; a flipped branch is a memory bug, not a value bug, and no value test can see it. |

Report every result, including the fourth.

- [ ] **Step 6: Measure the saving**

```bash
.venv/bin/python - <<'EOF'
import numpy as np, sys, time, tracemalloc; sys.path.insert(0, "src")
from mbfps.eval.probe import fit_probe
rng = np.random.default_rng(0)
x, y = rng.normal(size=(600, 8000)).astype(np.float64), rng.normal(size=(600, 4)) * 240.0
xv, yv = rng.normal(size=(200, 8000)), rng.normal(size=(200, 4)) * 240.0
tracemalloc.start(); t = time.time()
p = fit_probe(x, y, xv, yv)
peak = tracemalloc.get_traced_memory()[1] / 2**30
print(f"D=8000 n=600: peak {peak:.3f} GB, {time.time()-t:.1f}s, ridge {p['ridge']:.0e}")
print(f"the primal would need {(8001**2*8)/2**30:.2f} GB per matrix, x3 live, x5 ridges")
EOF
```

Report the peak. `D = 8000` rather than 37632 so the comparison number is computable without actually allocating it.

- [ ] **Step 7: Commit**

```bash
git add src/mbfps/eval/probe.py tests/eval/test_probe.py
git commit -F - <<'EOF'
perf: solve the ridge in whichever space is smaller

_solve materialised ridge * np.eye(D + 1). At D = 37632 -- a 112x112x3 frame
-- that one array is 11.3 GB, design.T @ design is another, and fit_probe
calls _solve once per rung of RIDGES. A first attempt at M3o's raw-pixel cell
took 18 GB of swap.

Not a latent bug: every existing caller probes at D ~1536-2048, where D^2 is
33 MB and the primal form is natural. The raw-pixel cell is simply the first
caller where D exceeds n.

The dual form is an exact algebraic reformulation, pinned at every rung
because agreement is worst at the smallest ridge -- 5.164e-09 at 1e-1 against
3.553e-15 at 1e7. The dual's own cost is n^2, so the branch is on min(n, D)
rather than always-dual. The test's oracle is the old implementation written
out independently, so a change inside _solve cannot move both sides.

np.eye is no longer materialised in either form: the ridge is added to the
diagonal in place, and the intercept is freed by subtracting it back from the
last entry.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

## Task 2: the two new estimators, and a dispatching `apply_readout`

**Files:**
- Create: `src/mbfps/eval/ceiling.py`
- Test: `tests/eval/test_ceiling.py`

**Interfaces:**
- Consumes: `probe.fit_probe(x, y, val_x, val_y)`, `probe.apply_probe(probe, x)`, `probe.RIDGES`, `probe._mean_r2(predicted, targets)`
- Produces:
  ```python
  BANDWIDTHS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
  NEIGHBOURS: tuple[int, ...] = (1, 4, 16, 64)

  def fit_kernel_probe(x, y, val_x, val_y) -> dict   # {"kind": "rbf", ...}
  def fit_knn_probe(x, y, val_x, val_y) -> dict      # {"kind": "knn", ...}
  def apply_readout(readout: dict, x: np.ndarray) -> np.ndarray
  ```
  `apply_readout` dispatches on `readout.get("kind", "linear")` and sends the linear case to `probe.apply_probe` unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_ceiling.py`:

```python
import numpy as np
import pytest

from mbfps.eval.ceiling import (
    BANDWIDTHS, NEIGHBOURS, apply_readout, fit_kernel_probe, fit_knn_probe,
)
from mbfps.eval.probe import apply_probe, fit_probe


def _linear_task(seed=0, n=240, d=12):
    """A task a LINEAR probe solves well. The RBF must match it, not beat it."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    w = rng.normal(size=(d, 4)) * 30.0
    return x, x @ w + rng.normal(size=(n, 4)) * 0.5


def _curved_task(seed=0, n=240, d=3):
    """A task a linear probe CANNOT solve: the target is a radial function.

    This is the fixture that makes "the RBF recovers more" a measurement
    rather than an artefact of capacity. Its discrimination is asserted in the
    test before anything is concluded from it.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    r = np.linalg.norm(x, axis=1, keepdims=True)
    return x, np.concatenate([np.sin(2.0 * r), np.cos(2.0 * r), r ** 2, -r], axis=1) * 40.0


def test_the_rbf_recovers_a_curved_target_the_linear_probe_cannot():
    """The fixture is shown to defeat the linear probe FIRST.

    Without that assertion the comparison could pass on a task both solve, and
    would then be measuring nothing.
    """
    xf, yf = _curved_task(0); xv, yv = _curved_task(1); xs, ys = _curved_task(2)
    lin = apply_readout(fit_probe(xf, yf, xv, yv), xs)
    from mbfps.eval.probe import _mean_r2
    lin_r2 = _mean_r2(lin, ys)
    assert lin_r2 < 0.5, f"fixture must defeat the linear probe; it scored {lin_r2:.3f}"

    rbf = apply_readout(fit_kernel_probe(xf, yf, xv, yv), xs)
    assert _mean_r2(rbf, ys) > lin_r2 + 0.2


def test_the_rbf_matches_the_linear_probe_on_a_linear_task():
    """Nesting: an RBF approaches the linear kernel as its bandwidth grows.

    So the RBF must not be WORSE on a task the linear probe solves. This is the
    control that stops "D >= C" being evidence of anything -- it is guaranteed
    by construction, and this test is what makes that guarantee checkable.
    """
    from mbfps.eval.probe import _mean_r2
    xf, yf = _linear_task(0); xv, yv = _linear_task(1); xs, ys = _linear_task(2)
    lin = _mean_r2(apply_readout(fit_probe(xf, yf, xv, yv), xs), ys)
    rbf = _mean_r2(apply_readout(fit_kernel_probe(xf, yf, xv, yv), xs), ys)
    assert lin > 0.9, f"fixture must be solvable linearly; linear scored {lin:.3f}"
    assert rbf > lin - 0.05, f"the RBF nests the linear probe; {rbf:.3f} vs {lin:.3f}"


def test_the_knn_recovers_a_curved_target_and_reports_its_k():
    from mbfps.eval.probe import _mean_r2
    xf, yf = _curved_task(0); xv, yv = _curved_task(1); xs, ys = _curved_task(2)
    readout = fit_knn_probe(xf, yf, xv, yv)
    assert readout["k"] in NEIGHBOURS
    assert _mean_r2(apply_readout(readout, xs), ys) > 0.5


def test_each_estimator_selects_its_capacity_on_the_held_out_split():
    """A capacity chosen on the FIT split selects itself in.

    Both estimators must move their chosen rung when the selection data moves,
    and the chosen rung must come from the declared ladder.
    """
    xf, yf = _curved_task(0)
    a = fit_kernel_probe(xf, yf, *_curved_task(1))
    b = fit_kernel_probe(xf, yf, *( _curved_task(1)[0] * 4.0, _curved_task(1)[1]))
    assert a["bandwidth"] in BANDWIDTHS and b["bandwidth"] in BANDWIDTHS
    assert a["bandwidth"] != b["bandwidth"], "selection must respond to the select split"


def test_apply_readout_sends_a_linear_probe_to_the_shipped_path_unchanged():
    """`apply_probe` is the shipped function and must not be reimplemented.

    Compared bitwise, because any divergence would silently change thirteen
    milestones' worth of recorded numbers.
    """
    xf, yf = _linear_task(0); xs, _ = _linear_task(2)
    probe = fit_probe(xf, yf, *_linear_task(1))
    np.testing.assert_array_equal(apply_readout(probe, xs), apply_probe(probe, xs))


def test_apply_readout_refuses_a_readout_whose_kind_it_does_not_know():
    with pytest.raises(KeyError, match="unknown readout kind"):
        apply_readout({"kind": "wavelet"}, np.zeros((3, 3)))
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/eval/test_ceiling.py -v
```

Expected: FAIL at collection, `ModuleNotFoundError: No module named 'mbfps.eval.ceiling'`.

- [ ] **Step 3: Write the module**

Create `src/mbfps/eval/ceiling.py`:

```python
"""M3o: what can the READOUT reach, independently of the model?

M3n returned NO_MAJORITY. Its gate failed in 4 of 9 cells because a perfect
one-step predictor beats a copying one by 0.576-11.824 map units against
readout errors of 131.88-259.06. Thirteen milestones have measured the world
model through that readout; this module measures the readout.

THREE EXPLANATIONS, and they call for different responses: the linear probe
family is too weak, the representations do not carry position, or position is
not recoverable from a single frame in this maze. The third is untested and
would explain why three different information sets -- embedding alone, latent
alone, both jointly -- all land within 0.03 of each other at r2 ~0.33.

WHY AN RBF AND NOT AN MLP. Kernel ridge is the SAME estimator as the shipped
probe, lifted into a nonlinear feature space, so the linear-versus-nonlinear
comparison is a capacity comparison within one family rather than across two
fitting procedures. It is closed form: no SGD, no seeds, no training-loop
variance entering the one measurement whose job is to bound variance. It is
also literally `probe._solve`'s dual branch with a kernel in place of the
Gram, so the memory fix and the nonlinear probe are one piece of machinery.

AND WHY "THE RBF WINS" IS NOT A FINDING. An RBF approaches the linear kernel
as its bandwidth grows, so the linear probe is nearly a special case and the
held-out selection will pick something like it if nothing better exists. The
finding is the MAGNITUDE of the difference and whether it is resolvable --
the trap M3n walked into from the other side, where `skill` had a positive
point estimate in 8 of 9 cells and meant nothing until the interval was drawn.
"""

from __future__ import annotations

import numpy as np

from mbfps.eval.probe import _mean_r2, apply_probe, fit_probe

BANDWIDTHS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
"""RBF length scales, as multiples of the median pairwise distance.

Expressed as multiples rather than absolutes so the ladder means the same
thing at D = 2048 and D = 37632, where raw distances differ by orders of
magnitude. The largest rung is where the kernel is closest to linear."""

NEIGHBOURS: tuple[int, ...] = (1, 4, 16, 64)
"""k for the k-NN readout. 1 is the aliasing question at its sharpest: if the
single nearest frame in feature space is at a distant true position, the
feature space aliases."""


def _median_scale(x: np.ndarray) -> float:
    """The median pairwise distance, the length unit `BANDWIDTHS` multiplies.

    Computed on a capped subsample: it is a scale, not a statistic, and the
    full pairwise matrix is n^2 for no gain.
    """
    rng = np.random.default_rng(0)
    take = x if x.shape[0] <= 512 else x[rng.choice(x.shape[0], 512, replace=False)]
    d = np.linalg.norm(take[:, None, :] - take[None, :, :], axis=-1)
    off = d[~np.eye(len(take), dtype=bool)]
    scale = float(np.median(off))
    if not scale > 0.0:
        raise ValueError("every pair of feature rows is identical; no scale to set")
    return scale


def _rbf(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    sq = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
    return np.exp(-gamma * np.maximum(sq, 0.0))


def fit_kernel_probe(x, y, val_x, val_y) -> dict:
    """Kernel ridge with an RBF kernel, bandwidth selected on held-out data.

    The same three-split discipline as `fit_probe`: fit here, select the
    bandwidth on `val_*`, and the caller scores on a third split. A bandwidth
    selected on the fit split selects itself in, which is exactly what makes a
    nonlinear probe untrustworthy as evidence.
    """
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    mean, scale = x.mean(0), x.std(0) + 1e-8
    xs = (x - mean) / scale
    xv = (np.asarray(val_x, dtype=np.float64) - mean) / scale
    unit = _median_scale(xs)
    centre = y.mean(0)
    best = None
    for multiple in BANDWIDTHS:
        gamma = 1.0 / (2.0 * (multiple * unit) ** 2)
        gram = _rbf(xs, xs, gamma)
        gram.flat[:: gram.shape[0] + 1] += 1e-3
        alpha = np.linalg.solve(gram, y - centre)
        score = _mean_r2(_rbf(xv, xs, gamma) @ alpha + centre, val_y)
        if best is None or score > best[0]:
            best = (score, multiple, gamma, alpha)
    return {
        "kind": "rbf", "mean": mean, "scale": scale, "anchors": xs,
        "alpha": best[3], "centre": centre, "gamma": best[2],
        "bandwidth": best[1], "r2": best[0],
    }


def fit_knn_probe(x, y, val_x, val_y) -> dict:
    """k-nearest-neighbour regression, k selected on held-out data.

    Assumption-free, and the most direct test of aliasing: it asks whether
    near-in-feature-space implies near-in-position, which is the question.
    """
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    mean, scale = x.mean(0), x.std(0) + 1e-8
    xs = (x - mean) / scale
    xv = (np.asarray(val_x, dtype=np.float64) - mean) / scale
    best = None
    for k in NEIGHBOURS:
        if k > xs.shape[0]:
            continue
        score = _mean_r2(_knn_predict(xs, y, xv, k), val_y)
        if best is None or score > best[0]:
            best = (score, k)
    if best is None:
        raise ValueError(
            f"no rung of NEIGHBOURS={NEIGHBOURS} fits {xs.shape[0]} fit rows"
        )
    return {
        "kind": "knn", "mean": mean, "scale": scale, "anchors": xs,
        "targets": y, "k": best[1], "r2": best[0],
    }


def _knn_predict(anchors, targets, query, k) -> np.ndarray:
    sq = (query * query).sum(1)[:, None] + (anchors * anchors).sum(1)[None, :] \
        - 2.0 * (query @ anchors.T)
    near = np.argpartition(sq, kth=k - 1, axis=1)[:, :k]
    return targets[near].mean(axis=1)


def apply_readout(readout: dict, x: np.ndarray) -> np.ndarray:
    """Apply any of the three readouts, dispatching on `kind`.

    A readout with no `kind` is the shipped linear probe and goes to
    `probe.apply_probe` UNCHANGED. Reimplementing that path would silently
    change thirteen milestones' worth of recorded numbers, so it is delegated
    rather than duplicated.
    """
    kind = readout.get("kind", "linear")
    if kind == "linear":
        return apply_probe(readout, x)
    xs = (np.asarray(x, dtype=np.float64) - readout["mean"]) / readout["scale"]
    if kind == "rbf":
        return _rbf(xs, readout["anchors"], readout["gamma"]) @ readout["alpha"] \
            + readout["centre"]
    if kind == "knn":
        return _knn_predict(readout["anchors"], readout["targets"], xs, readout["k"])
    raise KeyError(f"unknown readout kind {kind!r}; expected linear, rbf or knn")
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/eval/test_ceiling.py -q
```

Expected: PASS, 7 tests. Report the summary line.

- [ ] **Step 5: Verify the tests bite**

Commit first. Then, clearing `__pycache__` between runs:

| mutation | test that must fail |
| --- | --- |
| `fit_kernel_probe`: select on `(xs, y)` instead of `(xv, val_y)` | `test_each_estimator_selects_its_capacity_on_the_held_out_split` |
| `fit_kernel_probe`: fix `multiple` to `BANDWIDTHS[-1]` | `test_the_rbf_recovers_a_curved_target_the_linear_probe_cannot` |
| `_knn_predict`: `.mean(axis=1)` -> `[:, 0]` (ignore k) | `test_the_knn_recovers_a_curved_target_and_reports_its_k` |
| `apply_readout`: inline the linear path instead of delegating | `test_apply_readout_sends_a_linear_probe_to_the_shipped_path_unchanged` |
| `apply_readout`: drop the final `raise` | `test_apply_readout_refuses_a_readout_whose_kind_it_does_not_know` |

Report each with its failure message, and say plainly if any survives.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/ceiling.py tests/eval/test_ceiling.py
git commit -F - <<'EOF'
feat: an RBF kernel readout and a k-NN readout, beside the shipped linear one

Kernel ridge is the same estimator as the shipped probe lifted into a
nonlinear feature space, so linear-versus-nonlinear is a capacity comparison
within one family rather than across two fitting procedures. Closed form: no
SGD, no seeds, no training-loop variance entering the measurement whose job
is to bound variance. It is also _solve's dual branch with a kernel in place
of the Gram.

Both ladders are selected on the held-out split, never the fit split -- a
capacity chosen on the fit split selects itself in, which is what makes a
nonlinear probe untrustworthy as evidence. BANDWIDTHS are multiples of the
median pairwise distance so the ladder means the same thing at D = 2048 and
D = 37632.

The curved fixture is asserted to defeat the linear probe BEFORE anything is
concluded from the RBF beating it; without that the comparison could pass on
a task both solve. The nesting control is the other half: an RBF approaches
the linear kernel as its bandwidth grows, so it must not be WORSE on a task
the linear probe solves, and "the RBF wins" is therefore not by itself
evidence of anything.

apply_readout delegates the linear case to the shipped apply_probe rather
than reimplementing it, pinned bitwise.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

## Task 3: the grid, and the instrument's quality on the applied distribution

**Files:**
- Modify: `src/mbfps/eval/ceiling.py` (append)
- Test: `tests/eval/test_ceiling.py` (append)

**Interfaces:**
- Consumes: Task 2's estimators; `probe.probe_targets`, `probe._block_bootstrap_ci`, `probe._validate_groups`
- Produces:
  ```python
  FRAME_BUDGET: int = 12000
  ZERO_INFORMATION_MEDIAN: float = 283.05

  @dataclass(frozen=True)
  class Readout:
      name: str          # "linear" | "rbf" | "knn"
      r2: float
      median_error: float      # map units, the unit M3n's 131.88-259.06 is in
      capacity: float          # the selected ridge, bandwidth or k
      predicted: np.ndarray    # kept so a difference can be bootstrapped

  def score_readout(name, fit, select, score) -> Readout
  def zero_information_error(fit_targets, score_targets) -> float
  def difference_interval(a: Readout, b: Readout, score_targets, groups, *, seed) -> dict
  ```
  `fit`, `select` and `score` are each a `(features, targets)` pair. `difference_interval` returns `{"point", "ci_low", "ci_high"}` on `a.r2 - b.r2`.

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_ceiling.py`:

```python
def test_a_readout_reports_its_error_in_map_units_not_only_r2():
    """r2 is not the unit the decision is made in.

    M3n's readout error was 131.88-259.06 MAP UNITS against a signal of
    0.576-11.824, and that comparison is what made it unreadable. A result in
    r2 alone cannot be set against it.
    """
    xf, yf = _linear_task(0)
    out = score_readout("linear", (xf, yf), _linear_task(1), _linear_task(2))
    assert out.name == "linear" and 0.0 < out.r2 <= 1.0
    assert out.median_error > 0.0
    assert out.predicted.shape == _linear_task(2)[1].shape


def test_the_zero_information_error_is_the_distance_to_the_fit_mean():
    """The scale every map-unit figure is read against, derived not asserted.

    The expected value is computed here from the arrays, not taken from the
    function, so a change inside it cannot move both sides.
    """
    _, yf = _linear_task(0); _, ys = _linear_task(2)
    expected = float(np.median(np.linalg.norm(ys[:, :2] - yf[:, :2].mean(0), axis=1)))
    assert zero_information_error(yf, ys) == pytest.approx(expected, abs=1e-12)


def test_a_difference_interval_is_clustered_on_the_groups_it_is_given():
    """Two cluster labellings on the same rows must give different widths.

    Episodes, never frames: adjacent Doom frames are near-duplicates, and this
    project measured a 1.26x-2.17x penalty for pretending otherwise.
    """
    xf, yf = _curved_task(0); sel = _curved_task(1); sc = _curved_task(2)
    rbf = score_readout("rbf", (xf, yf), sel, sc)
    lin = score_readout("linear", (xf, yf), sel, sc)
    n = sc[1].shape[0]
    few = difference_interval(rbf, lin, sc[1], np.arange(n) // 60, seed=0)
    many = difference_interval(rbf, lin, sc[1], np.arange(n), seed=0)
    assert few["ci_low"] < few["ci_high"], "fixture must give a nonzero-width interval"
    assert many["ci_low"] < many["ci_high"]
    assert (few["ci_high"] - few["ci_low"]) > (many["ci_high"] - many["ci_low"])
    assert few["point"] == pytest.approx(rbf.r2 - lin.r2, abs=1e-12)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/eval/test_ceiling.py -k "map_units or zero_information or difference_interval" -v
```

Expected: FAIL at collection, `ImportError: cannot import name 'score_readout'`.

- [ ] **Step 3: Write the implementation**

Append to `src/mbfps/eval/ceiling.py`:

```python
FRAME_BUDGET: int = 12000
"""Frames drawn from the validation episodes, by a fixed stride.

STATED RATHER THAN DISCOVERED. `_solve`'s dual branch costs n^2: 12,000 frames
is 1.17 GB, 20,000 is 3.2 GB. M3o's predecessor found an 18 GB allocation at
run time, which is the failure this constant exists to prevent."""

ZERO_INFORMATION_MEDIAN: float = 283.05
"""Median error of predicting the training mean, on the shipped split.

Measured on 12,073 validation frames from 24 episodes. The scale every
map-unit figure in this milestone is read against: the shipped rollout probe's
floor error of 131.88-259.06 is 45% to 88% of it, so that probe carries real
information and is not broken. Recomputed as a control rather than trusted."""


@dataclass(frozen=True)
class Readout:
    """One estimator scored on one feature set, in both units."""

    name: str
    r2: float
    median_error: float
    """Median Euclidean position error in MAP UNITS, over the scored rows.

    Carried beside r2 because r2 is not the unit the decision is made in. M3n
    was unreadable because a 0.576-11.824 signal sat under a 131.88-259.06
    readout error, and a result in r2 alone cannot be set against that."""
    capacity: float
    predicted: np.ndarray
    """The scored-row predictions, kept so a difference between two readouts
    can be bootstrapped with both held fixed."""


_FITTERS = {"linear": fit_probe, "rbf": fit_kernel_probe, "knn": fit_knn_probe}
_CAPACITY_KEY = {"linear": "ridge", "rbf": "bandwidth", "knn": "k"}


def score_readout(name: str, fit, select, score) -> Readout:
    """Fit on `fit`, select capacity on `select`, score on `score`.

    Three splits, never two. A capacity selected on the fit split selects
    itself in, and that is the confound which makes a nonlinear probe
    untrustworthy as evidence rather than merely optimistic.
    """
    if name not in _FITTERS:
        raise KeyError(f"unknown readout {name!r}; expected {sorted(_FITTERS)}")
    readout = _FITTERS[name](fit[0], fit[1], select[0], select[1])
    predicted = apply_readout(readout, score[0])
    return Readout(
        name=name,
        r2=_mean_r2(predicted, score[1]),
        median_error=float(np.median(
            np.linalg.norm(predicted[:, :2] - score[1][:, :2], axis=1)
        )),
        capacity=float(readout[_CAPACITY_KEY[name]]),
        predicted=predicted,
    )


def zero_information_error(fit_targets, score_targets) -> float:
    """Median map-unit error of predicting the FIT split's mean position.

    The floor of informativeness. A readout that scores above this carries
    nothing, and one at 45% of it -- which is where the shipped rollout probe's
    best cell sits -- carries a great deal.
    """
    mu = np.asarray(fit_targets, dtype=np.float64)[:, :2].mean(0)
    return float(np.median(
        np.linalg.norm(np.asarray(score_targets, dtype=np.float64)[:, :2] - mu, axis=1)
    ))


def difference_interval(a: Readout, b: Readout, score_targets, groups, *, seed) -> dict:
    """`a.r2 - b.r2` with an episode-clustered percentile interval.

    `probe._block_bootstrap_ci` already does exactly this: it takes two
    prediction arrays and the targets, resamples `groups` with both readouts
    held FIXED, and returns the percentile interval on the difference. Its
    parameter names say joint and embedding, but the computation is generic in
    the two arrays, so no second resampler is written here.
    """
    low, high = _block_bootstrap_ci(
        a.predicted, b.predicted, np.asarray(score_targets, dtype=np.float64),
        resamples=RESAMPLES, confidence=CONFIDENCE, seed=seed, groups=groups,
    )
    return {"point": a.r2 - b.r2, "ci_low": low, "ci_high": high}
```

Add to the module's imports: `from dataclasses import dataclass` and extend the `probe` import with `_block_bootstrap_ci`. Add `CONFIDENCE: float = 0.95` and `RESAMPLES: int = 2000` beside the other constants.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/eval/test_ceiling.py -q
```

Expected: PASS, 10 tests.

- [ ] **Step 5: Verify the tests bite**

Commit first, then:

| mutation | test that must fail |
| --- | --- |
| `score_readout`: select on `fit` instead of `select` | `test_each_estimator_selects_its_capacity_on_the_held_out_split` |
| `median_error`: `np.median` -> `np.mean` | `test_the_zero_information_error_is_the_distance_to_the_fit_mean` — **check this; it may NOT fail, since that test does not go through `score_readout`.** If it survives, add an assertion that pins `median_error` against an independently computed median, and report that you did. |
| `zero_information_error`: use the SCORE split's mean | `test_the_zero_information_error_is_the_distance_to_the_fit_mean` |
| `difference_interval`: drop `groups=groups` | `test_a_difference_interval_is_clustered_on_the_groups_it_is_given` |

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/ceiling.py tests/eval/test_ceiling.py
git commit -F - <<'EOF'
feat: score a readout in both r2 and map units, with a clustered interval

r2 is not the unit the decision is made in. M3n was unreadable because a
0.576-11.824 map-unit signal sat under a 131.88-259.06 readout error, so
every cell carries its median map-unit error beside its r2.

Three splits, never two: a capacity selected on the fit split selects itself
in, which is the confound that makes a nonlinear probe untrustworthy as
evidence rather than merely optimistic.

No new resampler. probe._block_bootstrap_ci already takes two prediction
arrays and the targets, resamples groups with both readouts held fixed, and
returns the percentile interval on the r2 difference; its parameter names say
joint and embedding but the computation is generic in the two arrays.

FRAME_BUDGET is stated rather than discovered. The dual branch costs n^2:
12,000 frames is 1.17 GB and 20,000 is 3.2 GB, and this milestone's first
attempt found an 18 GB allocation at run time.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

## Task 4: the reading

**Files:**
- Modify: `src/mbfps/eval/ceiling.py` (append)
- Test: `tests/eval/test_ceiling.py` (append)

**Interfaces:**
- Consumes: Task 3's `Readout` and `difference_interval`; `burden.strict_majority`
- Produces:
  ```python
  VERDICTS = ("PROBE_LIMITED", "REPRESENTATION_LIMITED", "OBSERVATION_LIMITED")
  NO_MAJORITY = "NO_MAJORITY"

  @dataclass(frozen=True)
  class CeilingCell:
      arm: str
      seed: int
      probe_gain: dict      # D - C, the nonlinear-over-linear difference
      pixel_gain: dict      # B - D, the raw-pixel-over-representation difference
      sequence_gain: dict   # F - B, history over one frame
      def verdict(self) -> str

  @dataclass(frozen=True)
  class CeilingInputs:
      cells: dict[tuple[str, int], CeilingCell]
  @dataclass(frozen=True)
  class CeilingStatus:
      verdict: str
      rule: str
      tally: dict[str, tuple[tuple[str, int], ...]]

  def reading_ceiling(inputs: CeilingInputs) -> CeilingStatus
  READING_COLUMNS, READING_WIDTHS
  def format_reading_ceiling(reading, inputs) -> str
  ```

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_ceiling.py`:

```python
def _iv(point, low, high):
    return {"point": point, "ci_low": low, "ci_high": high}


def _cell(arm, seed, probe, pixel, seq=None):
    return CeilingCell(
        arm=arm, seed=seed, probe_gain=_iv(*probe), pixel_gain=_iv(*pixel),
        sequence_gain=_iv(*(seq or (0.0, -0.1, 0.1))),
    )


def _nine(probe, pixel, seq=None):
    return CeilingInputs(cells={
        (a, s): _cell(a, s, probe, pixel, seq)
        for a in ("frozen_ssl", "pixel_ae", "random_vit") for s in (0, 1, 2)
    })


def test_each_verdict_is_reachable_and_the_three_are_distinct():
    """Built as a table so no two rows share a discriminating field by accident."""
    table = {
        "PROBE_LIMITED":          ((0.20, 0.10, 0.30), (0.01, -0.05, 0.07)),
        "REPRESENTATION_LIMITED": ((0.01, -0.05, 0.07), (0.20, 0.10, 0.30)),
        "OBSERVATION_LIMITED":    ((0.01, -0.05, 0.07), (0.01, -0.05, 0.07)),
    }
    seen = {k: _cell("a", 0, *v).verdict() for k, v in table.items()}
    assert seen == {k: k for k in table}
    assert len(set(seen.values())) == 3


def test_probe_limited_wins_when_both_differences_resolve():
    """Precedence, stated as a value rather than left to branch order.

    If the linear family was the limit, that is the finding regardless of what
    the pixel comparison says, because the representation comparison was made
    with the weaker estimator on one side.
    """
    assert _cell("a", 0, (0.20, 0.10, 0.30), (0.20, 0.10, 0.30)).verdict() == "PROBE_LIMITED"


def test_a_strict_majority_is_computed_from_the_cells_present():
    """Five of nine carries; four does not. The bar is never stored."""
    probe_lim, flat = (0.20, 0.10, 0.30), (0.01, -0.05, 0.07)
    cells = {}
    for i, (a, s) in enumerate(
        (a, s) for a in ("frozen_ssl", "pixel_ae", "random_vit") for s in (0, 1, 2)
    ):
        cells[(a, s)] = _cell(a, s, probe_lim if i < 5 else flat, flat)
    assert reading_ceiling(CeilingInputs(cells=cells)).verdict == "PROBE_LIMITED"

    cells = {}
    for i, (a, s) in enumerate(
        (a, s) for a in ("frozen_ssl", "pixel_ae", "random_vit") for s in (0, 1, 2)
    ):
        cells[(a, s)] = _cell(a, s, probe_lim if i < 4 else flat, flat)
    got = reading_ceiling(CeilingInputs(cells=cells))
    assert got.verdict in (NO_MAJORITY, "OBSERVATION_LIMITED")
    assert "4" in got.rule and "9" in got.rule


def test_an_arm_short_of_seeds_minimum_is_refused_by_name():
    flat = (0.01, -0.05, 0.07)
    cells = {("frozen_ssl", s): _cell("frozen_ssl", s, flat, flat) for s in (0, 1, 2)}
    cells[("pixel_ae", 0)] = _cell("pixel_ae", 0, flat, flat)
    with pytest.raises(ValueError, match="SEEDS_MINIMUM"):
        reading_ceiling(CeilingInputs(cells=cells))


def test_the_formatter_aligns_and_names_the_sequence_rider_separately():
    """The sequence result is a rider, not a verdict, and must read as one."""
    inputs = _nine((0.20, 0.10, 0.30), (0.01, -0.05, 0.07), (0.15, 0.05, 0.25))
    text = format_reading_ceiling(reading_ceiling(inputs), inputs)
    header = next(l for l in text.splitlines() if l.startswith(READING_COLUMNS[0]))
    body = [l for l in text.splitlines() if l.startswith(("frozen_ssl", "pixel_ae", "random_vit"))]
    assert len(header) == sum(READING_WIDTHS)
    assert len(body) == 9 and all(len(l) == len(header) for l in body)
    assert "history resolves" in text.lower()
    assert text.endswith("\n")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/eval/test_ceiling.py -k "verdict or majority or seeds_minimum or formatter" -v
```

Expected: FAIL at collection, `ImportError: cannot import name 'CeilingCell'`.

- [ ] **Step 3: Write the implementation**

Append to `src/mbfps/eval/ceiling.py`:

```python
VERDICTS: tuple[str, ...] = (
    "PROBE_LIMITED", "REPRESENTATION_LIMITED", "OBSERVATION_LIMITED",
)
NO_MAJORITY: str = "NO_MAJORITY"
SEEDS_MINIMUM: int = 3
ARMS_REQUIRED: int = 2


def _resolvable(interval: dict) -> bool:
    """The whole interval lies above zero. `ci_low > 0.0`, strictly.

    An interval whose lower bound is exactly 0.0 does not exclude zero, and the
    reading is a question about exclusion.
    """
    return interval["ci_low"] > 0.0


@dataclass(frozen=True)
class CeilingCell:
    """One arm at one seed: the three differences the verdict reads."""

    arm: str
    seed: int
    probe_gain: dict
    """`D - C`: the nonlinear readout over the linear one, same features."""
    pixel_gain: dict
    """`B - D`: the best readout of raw pixels over the best of the
    representation. Formed against THIS cell's `D` from the one study-wide
    `B`, so the nine differences share a term and are correlated. The record
    says so; it does not make them invalid, since each asks about its own
    cell's representation."""
    sequence_gain: dict
    """`F - B`: a window of frames over one frame. A RIDER, not a verdict."""

    def verdict(self) -> str:
        """`PROBE_LIMITED` takes precedence when both differences resolve.

        If the linear family was the limit, that is the finding regardless of
        the pixel comparison, because that comparison was made with the weaker
        estimator on one side and is not yet interpretable.
        """
        if _resolvable(self.probe_gain):
            return "PROBE_LIMITED"
        if _resolvable(self.pixel_gain):
            return "REPRESENTATION_LIMITED"
        return "OBSERVATION_LIMITED"


@dataclass(frozen=True)
class CeilingInputs:
    cells: dict[tuple[str, int], CeilingCell]


@dataclass(frozen=True)
class CeilingStatus:
    verdict: str
    rule: str
    tally: dict[str, tuple[tuple[str, int], ...]]


def reading_ceiling(inputs: CeilingInputs) -> CeilingStatus:
    """Strict majority of the cells, bar computed from the cells present."""
    if not inputs.cells:
        raise ValueError("reading_ceiling needs at least one cell")
    by_arm: dict[str, set[int]] = {}
    for arm, seed in inputs.cells:
        by_arm.setdefault(arm, set()).add(seed)
    short = {a: sorted(s) for a, s in by_arm.items() if len(s) < SEEDS_MINIMUM}
    if short:
        raise ValueError(
            f"every arm needs SEEDS_MINIMUM={SEEDS_MINIMUM} seeds before a verdict "
            f"is tallied; got {short}"
        )
    if len(by_arm) < ARMS_REQUIRED:
        raise ValueError(
            f"a verdict needs ARMS_REQUIRED={ARMS_REQUIRED} arms; got {sorted(by_arm)}"
        )
    tally = {name: [] for name in VERDICTS}
    for key, cell in sorted(inputs.cells.items()):
        tally[cell.verdict()].append(key)
    frozen = {name: tuple(v) for name, v in tally.items()}
    needed = strict_majority(len(inputs.cells))
    winners = [n for n in VERDICTS if len(frozen[n]) >= needed]
    assert len(winners) <= 1, (
        f"two verdicts cannot both hold a strict majority of {len(inputs.cells)}; "
        f"got {winners} -- the partition is broken"
    )
    counts = ", ".join(f"{n}={len(frozen[n])}" for n in VERDICTS)
    if not winners:
        return CeilingStatus(
            verdict=NO_MAJORITY,
            rule=f"no verdict reached {needed} of {len(inputs.cells)} cells ({counts})",
            tally=frozen,
        )
    return CeilingStatus(
        verdict=winners[0],
        rule=(
            f"{winners[0]} in {len(frozen[winners[0]])} of {len(inputs.cells)} cells "
            f"(strict majority {needed}, computed from the cells present; {counts})"
        ),
        tally=frozen,
    )


READING_COLUMNS: tuple[str, ...] = (
    "cell", "D-C (probe)", "B-D (pixels)", "F-B (sequence)", "verdict",
)
READING_WIDTHS: tuple[int, ...] = (22, 30, 30, 30, 24)


def _row(values, widths) -> str:
    return "".join(str(v).ljust(w) for v, w in zip(values, widths, strict=True))


def _gap(iv: dict) -> str:
    return f"{iv['point']:+7.4f} [{iv['ci_low']:+7.4f},{iv['ci_high']:+7.4f}]"


def format_reading_ceiling(reading: CeilingStatus, inputs: CeilingInputs) -> str:
    """The table, the verdict, and a legend conditioned on what each test shows.

    EVERY DIRECTIONAL SENTENCE IS CONDITIONAL. The predecessor shipped an
    unconditional one in three places its own results refuted, and the
    milestone before that licensed "the map genuinely does not predict motion"
    from a test that only failed to reject.
    """
    lines = [
        "M3o probe ceiling -- what the readout can reach, independently of the model",
        "",
        _row(READING_COLUMNS, READING_WIDTHS),
    ]
    for (arm, seed), cell in sorted(inputs.cells.items()):
        lines.append(_row(
            (f"{arm}_seed{seed}", _gap(cell.probe_gain), _gap(cell.pixel_gain),
             _gap(cell.sequence_gain), cell.verdict()),
            READING_WIDTHS,
        ))
    riders = sum(1 for c in inputs.cells.values() if _resolvable(c.sequence_gain))
    lines += [
        "",
        f"verdict: {reading.verdict}",
        f"rule:    {reading.rule}",
        f"rider:   history resolves what one frame cannot in {riders} of "
        f"{len(inputs.cells)} cells (F - B resolvably above zero)",
        "",
        "legend",
        "  D-C  the nonlinear readout over the linear one, on the same features.",
        "       An RBF nests the linear kernel, so D >= C is near-guaranteed and",
        "       only a RESOLVABLE difference is evidence of anything.",
        "  B-D  the best readout of raw pixels over the best of the representation.",
        "       B is one study-wide number, so the nine differences share a term",
        "       and are correlated; each still asks about its own representation.",
        "  F-B  a window of frames over one frame. A RIDER on the verdict, not a",
        "       verdict: it answers gate criterion 4's question without criterion",
        "       4's 'A beats B' framing.",
        "  A difference that does not resolve is the ruler failing to find one,",
        "  not a measurement that there is none; each interval's upper bound is",
        "  how much is NOT excluded.",
        "",
    ]
    return "\n".join(lines)
```

Add `from mbfps.eval.burden import strict_majority` to the imports.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/eval/test_ceiling.py -q
```

Expected: PASS, 15 tests.

- [ ] **Step 5: Verify the tests bite**

Commit first, then:

| mutation | test that must fail |
| --- | --- |
| `verdict`: check `pixel_gain` before `probe_gain` | `test_probe_limited_wins_when_both_differences_resolve` |
| `_resolvable`: `ci_low > 0.0` -> `ci_low >= 0.0` | **check it.** No fixture above has a lower bound of exactly 0.0, so it may survive; if it does, add a boundary case and report that you did. |
| `reading_ceiling`: `strict_majority(len(...))` -> `5` | neither majority test (both have nine cells) — **add an eleven-cell case where five carry, which fails by value under a stored 5** |
| `reading_ceiling`: delete the `SEEDS_MINIMUM` refusal | `test_an_arm_short_of_seeds_minimum_is_refused_by_name` |
| `_row`: drop `strict=True` | `test_the_formatter_aligns_and_names_the_sequence_rider_separately`, after shortening `READING_WIDTHS` by one |

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/ceiling.py tests/eval/test_ceiling.py
git commit -F - <<'EOF'
feat: the three-way reading, with the sequence result as a rider

PROBE_LIMITED takes precedence when both differences resolve: if the linear
family was the limit, that is the finding regardless of the pixel comparison,
because that comparison was made with the weaker estimator on one side.

The sequence gain is a rider, not a verdict. It answers gate criterion 4's
question -- does history carry what one frame cannot -- without criterion 4's
"A beats B" framing, which the shipped gain measurement already shows is the
wrong shape: the latent ADDS a resolvable +0.024 to +0.076 r2 in 4 of 9 cells
while losing "latent beats embedding" in 8 of 9.

The legend says that a difference which does not resolve is the ruler failing
to find one, not a measurement that there is none, and that each upper bound
is how much is not excluded. The predecessor licensed the stronger reading
from exactly this shape of test and had to withdraw it.

The bar is computed, never stored: strict_majority returns 5 at nine cells
and 6 at eleven.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

## Task 5: `scripts/probe_ceiling.py`

**Files:**
- Create: `scripts/probe_ceiling.py`
- Test: `tests/eval/test_probe_ceiling_script.py`
- Modify: `tests/eval/test_diagnose_dynamics_script.py` (exit registry gains `{49, 50}`)

**Interfaces:**
- Consumes: everything from Tasks 1-4; `probe.probe_episodes`, `probe.gather_probe_data`, `data.split.episode_split`
- Produces: a CLI with `--phase {all,measure,read}`, `--out`, `--arms`, `--seeds`, `--frames`, `--bootstrap-seed`; `EXIT_HARNESS_DISAGREES = 49`, `EXIT_NO_MAJORITY = 50`; `ceiling_record_path(out, arm, seed) -> Path`; `cell_bootstrap_seed(base, arm, seed) -> int`

**`scripts/prediction_burden.py` and `scripts/motion_headroom.py` are the siblings this script mirrors.** Read `motion_headroom.py` in full before writing: same two-phase split, same protocol check, same `git_sha` / `record_git_sha` discipline, same `_Csv` list-flag parser that accepts both comma-joined and space-separated values, same record-writing pattern. Where this plan and that file disagree on a detail of form, follow the file and say so in your report.

**Record schema** (one JSON per cell; every constant stored, so none is a literal that survives the tests):

```python
{
  "arm": str, "seed": int, "step": int,
  "git_sha": str, "record_git_sha": str, "torch_version": str, "device": str,
  "frames": 12000, "frame_stride": int,
  "confidence": 0.95, "resamples": 2000,
  "bootstrap_seed": int, "cell_bootstrap_seed": int, "split_seed": int,
  "zero_information_median": 283.05,
  "episodes": {"fit": [...], "select": [...], "score": [...]},
  "groups": [...],
  "applied": {"r2": float, "median_error": float},
  "readouts": {
    "pixels_linear":     {"r2": f, "median_error": f, "capacity": f},
    "pixels_rbf":        {...}, "embedding_linear": {...},
    "embedding_rbf":     {...}, "embedding_knn":   {...},
    "sequence_linear":   {...}, "sequence_rbf":    {...},
  },
  "gains": {
    "probe":    {"point": f, "ci_low": f, "ci_high": f},
    "pixel":    {...}, "sequence": {...},
  },
  "controls": {
    "harness_r2_matches_study": bool,
    "study_embedding_selection_r2": float,
    "nesting_rbf_at_widest_vs_linear": float,
    "zero_information_recomputed": float,
  },
}
```

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_probe_ceiling_script.py`, mirroring `tests/eval/test_motion_headroom_script.py`'s fake-checkpoint helpers. Read the sibling for the fixture shapes.

**Four helpers this module must define**, since the tests below depend on them:

- `_write_nine(tmp_path, *, harness_matches)` — writes nine records across three arms and three seeds, with `controls.harness_r2_matches_study` set from the flag. Seeds must include values other than 0, and the function must assert its own discrimination (at least three distinct episode-group labels, and gains whose intervals have nonzero width) before returning.
- `_write_split(tmp_path)` — nine records whose verdicts split so that none reaches the bar of five.
- `_measure_one_fake_cell(tmp_path)` — runs `measure_cell` against the fake checkpoint and returns the written record as a dict.
- `_capture_read_stdout(tmp_path)` — runs the read phase and returns its stdout.

```python
def test_the_record_path_separates_seeds(tmp_path):
    """Three seeds of one arm must not collide on one filename.

    The predecessor's fixtures all used seed 0, which hid exactly this: the
    real run would have written three seeds of an arm to one file, destroying
    6 of 9 records and surfacing only at the read phase after the compute was
    spent.
    """
    paths = {ceiling_record_path(tmp_path, "pixel_ae", s) for s in (0, 1, 2)}
    assert len(paths) == 3


def test_the_measure_phase_propagates_a_failed_cell_as_a_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.probe_ceiling.measure_cell",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert main(["--phase", "measure", "--out", str(tmp_path)]) != 0


def test_the_read_returns_49_when_the_harness_does_not_reproduce_the_study(tmp_path):
    """Exit 49 says M3o is measuring something other than what the study
    measured, so no verdict about the model is available."""
    _write_nine(tmp_path, harness_matches=False)
    assert main(["--phase", "read", "--out", str(tmp_path)]) == EXIT_HARNESS_DISAGREES


def test_the_read_returns_50_when_no_verdict_holds_a_majority(tmp_path):
    _write_split(tmp_path)
    assert main(["--phase", "read", "--out", str(tmp_path)]) == EXIT_NO_MAJORITY


def test_every_recorded_constant_moves_when_the_module_constant_moves(tmp_path, monkeypatch):
    """Five record constants were literals surviving all 50 of a predecessor's
    tests. Each is patched one at a time, not together, so a literal cannot
    hide behind a correct neighbour."""
    for name, patched in (
        ("CONFIDENCE", 0.9), ("RESAMPLES", 7), ("FRAME_BUDGET", 64),
        ("ZERO_INFORMATION_MEDIAN", 1.5),
    ):
        monkeypatch.setattr(f"scripts.probe_ceiling.{name}", patched)
        record = _measure_one_fake_cell(tmp_path)
        assert record[_RECORD_KEY[name]] == patched


_RECORD_KEY = {
    "CONFIDENCE": "confidence",
    "RESAMPLES": "resamples",
    "FRAME_BUDGET": "frames",
    "ZERO_INFORMATION_MEDIAN": "zero_information_median",
}
"""Module constant -> the record key it is stored under. Defined here rather
than inlined so a constant added to the record without a test is visible as a
missing entry."""


def test_the_bootstrap_seed_is_derived_per_cell_and_is_not_hash(tmp_path):
    """One shared seed makes the nine cells resample the SAME draws, and the
    verdict is a majority over them, which treats them as independent.
    `hash()` is salted per process, so records would not reproduce."""
    seeds = {cell_bootstrap_seed(0, a, s)
             for a in ("frozen_ssl", "pixel_ae", "random_vit") for s in (0, 1, 2)}
    assert len(seeds) == 9
    assert cell_bootstrap_seed(0, "pixel_ae", 0) == cell_bootstrap_seed(0, "pixel_ae", 0)
    assert cell_bootstrap_seed(0, "pixel_ae", 0) != cell_bootstrap_seed(1, "pixel_ae", 0)


def test_the_read_is_byte_identical_across_two_reads(tmp_path):
    """Compared against text built from the SPEC path, not a second call to the
    same code: two reads of the same code strip the same bytes, so a
    byte-identity test against itself cannot catch an rstrip()."""
    _write_nine(tmp_path, harness_matches=True)
    first = _capture_read_stdout(tmp_path)
    assert first == _capture_read_stdout(tmp_path)
    assert first == (tmp_path / "ceiling.txt").read_text()
```

And in `tests/eval/test_diagnose_dynamics_script.py`, extend the exit registry exactly as `motion_headroom`'s 47 and 48 are checked:

```python
def test_probe_ceiling_exit_codes_are_new_and_do_not_collide():
    own = {v for k, v in vars(probe_ceiling).items()
           if k.startswith("EXIT_") and isinstance(v, int)}
    assert {49, 50} <= own
    assert probe_ceiling.EXIT_HARNESS_DISAGREES == 49
    assert probe_ceiling.EXIT_NO_MAJORITY == 50
    assert own.isdisjoint({43, 44, 45, 46, 47, 48})
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/eval/test_probe_ceiling_script.py -v
```

Expected: FAIL at collection, `ModuleNotFoundError: No module named 'scripts.probe_ceiling'`.

- [ ] **Step 3: Write the script**

Follow `scripts/motion_headroom.py` structurally. The pieces that are M3o's:

```python
EXIT_HARNESS_DISAGREES: int = 49
"""RETURNED BY THE READ PHASE when cell `C` fails to reproduce the study's own
`embedding_selection_r2` (0.320-0.345 across the nine cells).

M3o is then measuring something other than what the study measured, and no
verdict about the model is available. This is a statement about the HARNESS,
not about the representations, and it exits before any gain is read."""

EXIT_NO_MAJORITY: int = 50
"""RETURNED BY THE READ PHASE when no verdict held a strict majority."""

READ_EXITS: dict[str, int] = {NO_MAJORITY: EXIT_NO_MAJORITY}
"""The three model verdicts exit 0: each is an answer. 49 is raised before the
tally, from the controls, so it is not keyed here."""


def cell_bootstrap_seed(base_seed: int, arm: str, seed: int) -> int:
    """`crc32` of base, arm and seed. NEVER `hash()`, which is salted per
    process, so records would not reproduce across runs."""
    return zlib.crc32(f"{int(base_seed)}:{arm}:{int(seed)}".encode("utf-8"))


def ceiling_record_path(out: Path, arm: str, seed: int) -> Path:
    """One file per CELL, with the seed in the NAME, not only the payload."""
    return Path(out) / f"ceiling_{arm}_seed{seed}.json"
```

**`applied` is spec §3.1 and must be computed, not inferred from the schema.** It is the shipped rollout probe's quality on the distribution `evaluate_rollout` actually uses it on — predicted embeddings over an imagined rollout — which is recorded nowhere today. The rows exist: `diagnostics.reference_trajectories` returns `positions` (the probe's own output at each imagined step, `(n_windows, horizon, 2)`) and `true_positions` (the same shape). So:

```python
def applied_quality(trajectories) -> dict:
    """The rollout probe's r2 and median map-unit error on PREDICTED embeddings.

    Spec section 3.1, and the reason it is measured first: the 2x2 below is
    fit on REAL encoder embeddings at frame t, while the error that made M3n
    unreadable is this one. `embedding_selection_r2` spans 0.320-0.345 across
    the nine cells while the floor error spans 131.88-259.06 -- 8% against
    nearly 2x -- so the two distributions demonstrably differ, and a better
    probe on the first says little about the second until this is recorded.
    """
    predicted = trajectories.positions.reshape(-1, 2)
    truth = trajectories.true_positions.reshape(-1, 2)
    return {
        "r2": _mean_r2(predicted, truth),
        "median_error": float(np.median(np.linalg.norm(predicted - truth, axis=1))),
    }
```

It costs one `reference_trajectories` pass per cell, about 2.3 minutes at M3n's measured rate, and it is the first thing `measure_cell` does.

`measure_cell` then builds the three splits by **episode** from `episode_split`, subsamples to `FRAME_BUDGET` by a fixed stride recorded as `frame_stride`, scores the seven readouts through `ceiling.score_readout`, forms the three gains through `ceiling.difference_interval` at `cell_bootstrap_seed(...)`, and records the four controls. The raw-pixel readouts use no model, so compute them once and reuse across cells — assert they are identical across the nine records in the read phase, which is the free bitwise control.

`read_phase` loads every record, refuses on a protocol disagreement, checks `controls.harness_r2_matches_study` and exits 49 if any cell fails, builds `CeilingInputs`, calls `reading_ceiling`, writes `format_reading_ceiling`'s text to stdout **and** `<out>/ceiling.txt`, and exits `READ_EXITS.get(reading.verdict, 0)`. **Print `reading.rule`, not only `reading.verdict`.**

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/eval/test_probe_ceiling_script.py tests/eval/test_diagnose_dynamics_script.py -q
```

Expected: PASS. Report both summary lines.

- [ ] **Step 5: Verify the tests bite**

Mutations to run, each with the test that catches it reported:
- `ceiling_record_path`: drop the seed from the filename
- the measure phase: swallow a failed cell's status and return 0
- `cell_bootstrap_seed`: use `hash()` instead of `crc32`
- the read phase: skip the `harness_r2_matches_study` check
- each of the four record constants replaced by a literal

- [ ] **Step 6: Smoke the script**

```bash
.venv/bin/python scripts/probe_ceiling.py --phase all --out runs/m3o_smoke \
  --arms pixel_ae --seeds 0 --frames 400 --bootstrap-seed 0 2>&1 | tail -20
```

Expected: a nonzero exit refusing on the arms or seeds gate. **That refusal is the expected result** — a zero exit means the gate is not wired. Report the exit code and the message.

- [ ] **Step 7: Commit**

```bash
git add scripts/probe_ceiling.py tests/eval/test_probe_ceiling_script.py tests/eval/test_diagnose_dynamics_script.py
git commit -F - <<'EOF'
feat: scripts/probe_ceiling.py, the measure and read phases

Exit 49 is the harness's own verdict: cell C failed to reproduce the study's
embedding_selection_r2, so M3o is measuring something other than what the
study measured and no statement about the representations is available. It is
raised from the controls, before any gain is tallied.

The raw-pixel readouts use no model, so they are computed once and reused
across the nine cells -- and asserted identical across the nine records at
read time, which is a free bitwise control of the same kind the predecessor
got from its two reductions.

Each cell derives its own bootstrap seed from crc32 of base, arm and seed.
One shared seed would make the nine cells resample the same draws while the
verdict is a majority over them, which treats them as independent readings.
Never hash(), which is salted per process.

The record path carries the seed in the NAME. The predecessor's fixtures all
used seed 0, which hid a collision that would have destroyed 6 of 9 records
and surfaced only after the compute was spent.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

## Task 6: run the nine cells and write the results

**Files:**
- Create: `runs/m3o_ceiling/` (records; **never `rm` anything under `runs/`**)
- Modify: `docs/superpowers/plans/2026-10-06-mb-fps-m3o-probe-ceiling.md` (append `## Task 6 results`)

- [ ] **Step 1: Confirm the suite is green at the commit being run**

```bash
.venv/bin/python -m pytest -q > /tmp/m3o-suite.txt 2>&1; echo "PYTEST EXIT: $?"
grep -E "passed|failed" /tmp/m3o-suite.txt | tail -2
git rev-parse --short HEAD
```

Read the `N passed` line, not the exit code. Record both and the SHA. A run whose suite was not green at that SHA is not reportable.

- [ ] **Step 2: Measure**

```bash
mkdir -p runs/m3o_ceiling
.venv/bin/python scripts/probe_ceiling.py --phase measure --out runs/m3o_ceiling \
  --arms frozen_ssl,pixel_ae,random_vit --seeds 0,1,2 \
  --frames 12000 --bootstrap-seed 0 > runs/m3o_ceiling/measure.log 2>&1
echo "measure exit: $?"
tail -20 runs/m3o_ceiling/measure.log
ls runs/m3o_ceiling/ceiling_*.json | wc -l    # must print 9
```

The `mkdir` and the redirect are both load-bearing: `tee` opens its file before the script creates `--out`, and a pipe makes `$?` the pager's status.

- [ ] **Step 3: Check every control before reading any statistic**

```bash
.venv/bin/python - <<'EOF'
import glob, json
for p in sorted(glob.glob("runs/m3o_ceiling/ceiling_*.json")):
    c = json.load(open(p))["controls"]
    print(p.split("/")[-1], c)
EOF
```

Required: `harness_r2_matches_study` true in 9 of 9; `study_embedding_selection_r2` within 0.320-0.345; `zero_information_recomputed` within 1.0 of 283.05; `nesting_rbf_at_widest_vs_linear` not materially negative. Also assert the raw-pixel readouts are **identical across all nine records** — that row uses no model, so a difference means a model leaked into a model-free measurement.

- [ ] **Step 4: Read**

```bash
.venv/bin/python scripts/probe_ceiling.py --phase read --out runs/m3o_ceiling > runs/m3o_ceiling/read.log 2>&1
echo "read exit: $?"
cat runs/m3o_ceiling/read.log
```

Exit 0 is one of the three verdicts; 49 is `EXIT_HARNESS_DISAGREES`; 50 is `EXIT_NO_MAJORITY`. Record the code **and** the `rule:` line.

- [ ] **Step 5: Confirm the read is reproducible**

```bash
SECOND=$(mktemp)
.venv/bin/python scripts/probe_ceiling.py --phase read --out runs/m3o_ceiling > "$SECOND" 2>&1
diff runs/m3o_ceiling/read.log "$SECOND" && echo "byte-identical"
```

- [ ] **Step 6: Report the result in map units, against M3n's numbers**

The decision this feeds is whether a better readout rescues M3n. So report, per cell: the best readout's median map-unit error against the shipped probe's 131.88-259.06, and against the 283.05 zero-information baseline. An r2 improvement that does not move the map-unit error is not actionable.

- [ ] **Step 7: Write the results section**

Append `## Task 6 results`. Every number must be re-derivable from the records by the command that produced it. **Three of M3m's reported figures were not, and two of M3n's were not — including one that understated a prevalence by 7x.** State explicitly:

- the verdict and exit code, with the `rule:` line
- all four controls per cell, and the raw-pixel identity check
- the full seven-readout table in both r2 and map units
- the three gains with their intervals, and how many cells each resolves in
- the sequence rider: in how many cells `F - B` resolves
- **what the result licenses, in the spec's own §8 words, not new ones**
- the three things §9 says M3o does not answer, restated against what was found
- the wall clock, measured, not estimated

- [ ] **Step 8: Commit**

Replace the subject with what the run actually found before committing. A subject asserting more than the data supports is the defect M3l shipped and had to amend.

```bash
git add docs/superpowers/plans/2026-10-06-mb-fps-m3o-probe-ceiling.md
git commit -F - <<'EOF'
results: what the readout can reach

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

## Task 7 (CONDITIONAL): recompute M3n's decision cell with the better readout

**Run this task ONLY if Task 6 returned `PROBE_LIMITED`.** On the other two verdicts it answers nothing, and skipping it is the correct outcome rather than a shortfall. Record in the ledger which happened and why.

**Files:**
- Create: `runs/m3o_rescue/` (records)
- Modify: the plan's results section (append `## Task 7 results`)

- [ ] **Step 1: Confirm the gate**

```bash
grep -E "^verdict:" runs/m3o_ceiling/read.log
```

Proceed only on `PROBE_LIMITED`. If it says anything else, append one paragraph to the results saying the phase was gated out and by which verdict, and stop.

- [ ] **Step 2: Re-measure M3n's decision cell with the better readout**

`scripts/motion_headroom.py --phase measure` takes the probe it is handed. Re-run it with the RBF readout in place of the linear one, to a NEW directory — never over `runs/m3n_motion/`:

```bash
mkdir -p runs/m3o_rescue
.venv/bin/python scripts/motion_headroom.py --phase measure --out runs/m3o_rescue \
  --arms frozen_ssl,pixel_ae,random_vit --seeds 0,1,2 --ks 1,3,5,15,45 \
  --bootstrap-seed 0 --readout rbf > runs/m3o_rescue/measure.log 2>&1
echo "measure exit: $?"
```

`--readout` does not exist yet; adding it is this task's only change to `motion_headroom.py`, and it must default to `linear` so every existing invocation is unchanged. Pin that default with a test before running.

- [ ] **Step 3: Read, and compare against M3n**

```bash
.venv/bin/python scripts/motion_headroom.py --phase read --out runs/m3o_rescue > runs/m3o_rescue/read.log 2>&1
echo "read exit: $?"
```

Then report, against M3n's shipped reading:

| | M3n (linear) | M3o (rbf) |
| --- | --- | --- |
| `skill(1,1)` resolvably positive | 1 of 9 | ? |
| gate failures (`headroom` not resolvable) | 4 of 9 | ? |
| verdict / exit | `NO_MAJORITY` / 48 | ? |
| `headroom(1,1)` range | 0.576 to 11.824 | ? |

**The spec's falsifiable prediction:** as the readout improves, `headroom` converges toward the true one-step displacement, **3.9694722203504225**, from both sides, and its interval tightens. Report whether it did. A result that improves r2 without moving `headroom` toward 3.9695 refutes the prediction and is the more interesting outcome.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-10-06-mb-fps-m3o-probe-ceiling.md
git commit -F - <<'EOF'
results: whether a better readout rescues M3n's measurement

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```
