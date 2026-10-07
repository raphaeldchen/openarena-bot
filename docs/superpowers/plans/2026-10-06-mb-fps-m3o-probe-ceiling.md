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
- **`data/` IS ALSO A SYMLINK OUT OF THE WORKTREE**, to the main checkout's `data/` — 122 episodes of `my_way_home` (98 training, 24 validation), shared by every worktree and every session. Read it; never write, move or delete anything under it. The same rule as `runs/`, and the plan previously named only `runs/`.
- **Never** `git clean -fdx` — it destroys the session ledger at `.superpowers/sdd/`.
- After mutating a source file to check a test bites, clear `__pycache__` under `src/`, `tests/` and `scripts/` before re-running.
- **When you pipe pytest through `tail` or `head`, the pipeline's exit code is the pager's, not pytest's.** Always read the `N passed` / `N failed` summary line yourself. This milestone's predecessor lost a review cycle to a run that reported exit 0 over `1 failed, 474 passed`.
- `src/mbfps/eval/rollout.py`, `src/mbfps/eval/diagnostics.py`, `src/mbfps/eval/headroom.py`, `src/mbfps/eval/burden.py`, `scripts/trust_horizon.py`, `scripts/motion_headroom.py` are **unchanged by Tasks 1-6**. Verify with `git diff --stat 4956f59 HEAD -- <paths>` at the end of every task.
- **TASK 7 MAY MODIFY TWO OF THEM, ADDITIVELY. Decided by the human; this clause replaces the earlier "and it modifies none", which contradicted Task 7's own step.** `regrounding_sweep` takes the probe as a dict and calls `apply_probe`, which understands only the linear form, so routing an RBF readout through M3n's sweep needs:
  - `diagnostics.py`: the one call site `apply_probe(w, x)` becomes `apply_readout(w, x)`. **Provably behaviour-preserving** — `ceiling.apply_readout` dispatches on `kind` and `test_apply_readout_sends_a_linear_probe_to_the_shipped_path_unchanged` already pins that a linear probe reaches `apply_probe` untouched, so every existing caller stays bit-identical.
  - `motion_headroom.py`: an opt-in `--readout {linear,rbf}` defaulting to `linear`, with the default pinned by a test before anything runs.

  The rejected alternative was a second script reimplementing `regrounding_sweep`. That is a duplicate of a measurement path, which `burden.py`'s own module docstring records as the drift that already destroyed a signal once in this project.

  **The freeze's purpose is reproducibility, so Task 7 proves it rather than asserting it:** re-run M3n's own nine cells through the changed code and confirm `runs/m3n_motion/`'s records reproduce bitwise. No other frozen file may change. Verify with `git diff --stat 4956f59 HEAD -- <paths>` at the end of every task.
- **BOTH SIDES OF `D - C` SEARCH `ceiling.CEILING_RIDGES`, NOT `probe.RIDGES`.** Task 2 measured the ladder beating the linear reference by +0.012 to +0.020 **on a purely linear task**; Task 2b found the cause and removed it. `probe.RIDGES` steps by 100x and the optimum is interior, so the RBF rungs were reaching regularisation the linear reference could not — the floor and the gain an oracle ridge buys `C` are the SAME number. `CEILING_RIDGES` is 33 rungs at 10^0.25 over the same span; the floor then reads -0.0016 to +0.0000. Spec §3.3.1.
- **`probe.RIDGES` IS FROZEN** at `(1e-1, 1e1, 1e3, 1e5, 1e7)`. Thirteen milestones of recorded numbers came through it and `fit_probe` keeps it as the default for its new `ridges=` parameter. A task that changes its value is wrong.
- **`C` is the best LINEAR readout and is NOT the shipped probe.** `C` searches `CEILING_RIDGES` and is expected ABOVE the shipped `embedding_selection_r2` band (0.320-0.345) by up to ~0.02. The reconciliation and **exit 49 read `C_shipped`** — the same fit over `probe.RIDGES`, recorded as `shipped_grid_r2`. Gating on `C` would exit 49 on a correct run. Spec §3.3.1, §6.
- **`PROBE_LIMITED` is read against a MEASURED FLOOR, never against zero.** The floor is sharply shape-dependent — ~0 at D=37632 even on the coarse grid, largest at D=2048 — so Task 2b's three shapes do not license nine cells. Each cell measures `ceiling.null_floor(n, d)` at its own shape; `PROBE_LIMITED` requires the real `D - C` resolvably above it. A `null_floor` materially above zero is a finding about the instrument, not a number to subtract. Spec §6.1. **Task 2b implements it, Task 4 reads it, Task 5 records it.**
- Exact constants, copied verbatim: `FRAME_BUDGET = 12000`, `CONFIDENCE = 0.95`, `RESAMPLES = 2000`, `SEEDS_MINIMUM = 3`, `ARMS_REQUIRED = 2`, `ZERO_INFORMATION_MEDIAN = 283.05`, `DISPLACEMENT_TARGET = 3.9694722203504225`, `EXIT_HARNESS_DISAGREES = 49`, `EXIT_NO_MAJORITY = 50`. `RIDGES` is inherited from `probe.py`, not redefined. `CEILING_RIDGES` is `tuple(10.0 ** (-1.0 + 8.0 * i / 32) for i in range(33))` — 33 rungs, and it contains `RIDGES` bitwise as every 8th rung.
- **THE RUN IS `--frames 4000`, SETTLED BY THE HUMAN.** With the fit split moved to the study's 20 TRAINING episodes (§3.4), stride 1 gives fit 9,044 / select 5,713 / score 6,264 rows, and the kernel ladder's `n^3.04` makes that ~7 hours for nine cells with an unmeasured multi-GB raw-pixel ladder. `--frames 4000` gives stride 3 — fit 3,016 / select 1,905 / score 2,088 — for ~15 minutes total and 0.37 GiB per raw-pixel split.

  **The reason is statistical, not only cost.** Striding keeps every episode and thins only within-episode density, which is where the redundancy is: this project measured a 1.26x-2.17x clustering penalty for treating adjacent Doom frames as independent, so the thinned rows carry nearly the information the full set does. The run records `rows_per_split` and `frame_stride`, so the choice is visible in every record.

  State in Task 6's results what the reduced `n` costs row B as a ceiling claim.

- **EVERYTHING EXPENSIVE SCALES AS `n^3`, so the budget is not a detail.** An earlier revision of this line assumed `n ≈ 4,000` from dividing validation three ways; the fit split is the training episodes instead, so stride 1 is more than twice that. The measured scaling:

  | per split | one kernel ladder | `null_floor`, 3 draws | nine cells of floors | raw pixels, D=37632 |
  | --- | --- | --- | --- | --- |
  | n ≈ 4,000 (actual) | 36 s | 109 s | **17 min** | 1.12 GiB/split, 3.36 GiB for three |
  | n = 12,000 (the cap) | 1,027 s | 3,082 s | **7.7 hours** | 3.36 GiB/split, 10.1 GiB for three |

  Measured: 0.93 s at n=1200 and 63.2 s at n=4800, which is `n^3.04` — the seven eigendecompositions in `fit_kernel_probe`. **So the planned run is comfortable and needs no reduced budget.** Do not add one. But a change that lets `n` reach the cap turns 17 minutes into 7.7 hours, so `measure_cell` records the rows per split beside `frame_stride`, and the read phase prints them.

  Two corrections to what an earlier revision of this line claimed. The 18 GB allocation that stopped this milestone's first attempt was `_solve`'s PRIMAL path building `np.eye(37632)` at 11.3 GiB; **Task 1 fixed that** by branching on `min(n, D)`, and the dual Gram at n ≈ 4,000 is 128 MiB. And `FRAME_BUDGET`'s own docstring reasons from the Gram, which is right — it was the feature matrix that needed checking, and at the real `n` that is 1.12 GiB, not the 3.36 GiB the cap implies.
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
        assert np.max(np.abs(dual - primal)) < 1e-6, (
            f"ridge={ridge:g}: {np.max(np.abs(dual - primal)):.3e}"
        )


def test_the_dual_solve_centres_its_own_features_rather_than_assuming_a_caller_did():
    """UNCENTRED features, which the centred fixture above cannot catch.

    `fit_probe` passes standardised features, so a dual branch that assumes
    centring works in production and is silently wrong for any direct caller.
    Measured against the primal on features with a mean of 5.0, a dual that
    takes `y.mean(0)` as the intercept without centring `x` is off by 504.5
    map units at ridge 1e-1 and 130.3 at 1e3 -- plausible numbers, not a
    crash. This is the fixture that makes that mutation visible.
    """
    rng = np.random.default_rng(3)
    n, d = 40, 300
    x = rng.normal(size=(n, d)) + 5.0
    y = rng.normal(size=(n, 4)) * 240.0
    assert n < d, "the fixture must exercise the dual branch"
    assert np.min(np.abs(x.mean(0))) > 1.0, "the fixture must be genuinely uncentred"

    query = rng.normal(size=(17, d))
    design = np.concatenate([query, np.ones((17, 1))], axis=1)
    for ridge in RIDGES:
        gap = np.max(np.abs(
            design @ _solve(x, y, ridge) - design @ _primal_solve_for_test(x, y, ridge)
        ))
        assert gap < 1e-5, f"ridge={ridge:g}: {gap:.3e}"


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

    THE INTERCEPT IS NEVER REGULARISED, in either form, and the dual CENTRES
    `x` ITSELF rather than assuming a centred caller. `fit_probe` does pass
    standardised features, so assuming it would work in production -- and be
    wrong by up to 504 map units for anyone calling `_solve` directly with raw
    features, silently, since both forms return a plausible array. Ridge with a
    free intercept is exactly: centre both, solve without an intercept, then
    recover it as `y.mean(0) - x.mean(0) @ w`.
    """
    n, d = x.shape
    if d < n:
        design = np.concatenate([x, np.ones((n, 1))], axis=1)
        gram = design.T @ design
        gram.flat[:: gram.shape[0] + 1] += ridge      # in place; no np.eye
        gram[-1, -1] -= ridge                         # the intercept stays free
        return np.linalg.solve(gram, design.T @ y)
    x_centre, y_centre = x.mean(0), y.mean(0)
    xc = x - x_centre
    gram = xc @ xc.T
    gram.flat[:: gram.shape[0] + 1] += ridge
    weights = xc.T @ np.linalg.solve(gram, y - y_centre)
    return np.concatenate([weights, (y_centre - x_centre @ weights)[None, :]], axis=0)
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
| dual branch: drop `- y_centre` | `test_the_dual_and_primal_solves_agree_at_every_rung_of_the_ladder` |
| dual branch: return `weights` without the intercept row | the agreement test (shape) |
| dual branch: take `x` as already centred (`xc = x`, intercept `y_centre`) | `test_the_dual_solve_centres_its_own_features_rather_than_assuming_a_caller_did` ONLY — the centred fixture cannot see it |
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

## Task 2b: the ridge grid both sides search, and the floor that proves it

Task 2 made `D >= C` exact on the SELECTION split by putting the linear kernel in
the ladder. It did not make `D - C` mean "nonlinear capacity": both sides were
still searching `probe.RIDGES`, which steps by 100x, and the RBF rungs could
reach regularisation strengths the linear reference could not. This task gives
both sides one grid fine enough to resolve the optimum, and adds the per-cell
floor that proves it per shape. Spec §3.3.1, §6.1.

**Files:**
- Modify: `src/mbfps/eval/probe.py` — `fit_probe` gains `ridges=RIDGES`
- Modify: `src/mbfps/eval/ceiling.py` — append `CEILING_RIDGES`, `fit_linear_probe`, `null_floor`; **change `fit_kernel_probe`'s jitter loop at `:186`**
- Test: `tests/eval/test_probe.py` (append), `tests/eval/test_ceiling.py` (append **and three existing tests updated, below**)

**BOTH SIDES, OR THE BUG RETURNS MIRRORED.** `fit_kernel_probe` currently searches
jitters over `RIDGES` (`ceiling.py:186`). Giving `C` the finer grid while the
ladder keeps the coarse one would leave the ladder's LINEAR RUNG restricted to 5
jitters against `C`'s 33, so `C` could beat `D` — on the fixture below by about
-0.10, since `C` reaches 0.44 at ridge 178 where the coarse grid reaches 0.33.
That is Task 2's defect inverted: `D - C` negative for instrument reasons, and
`PROBE_LIMITED` unreachable. The nesting argument requires the two search spaces
to share the grid, so **`fit_kernel_probe` moves to `CEILING_RIDGES` in the same
commit.**

**Interfaces:**
- Consumes: Task 1's `probe._solve_ladder`; Task 2's `fit_kernel_probe`, `apply_readout`
- Produces:
  ```python
  # probe.py — the ONLY change is a new keyword with the shipped default
  def fit_probe(latents, targets, val_latents=None, val_targets=None,
                ridge=None, ridges: Sequence[float] = RIDGES) -> dict

  # ceiling.py
  CEILING_RIDGES: tuple[float, ...]                 # 33 rungs, 10^0.25 over [1e-1, 1e7]
  def fit_linear_probe(x, y, val_x, val_y) -> dict  # C: fit_probe over CEILING_RIDGES
  def null_floor(n: int, d: int, *, seed: int = 0) -> float
  ```
  Task 3 wires `fit_linear_probe` as the `"linear"` readout and plain `fit_probe`
  as `"linear_shipped"`. Do not change `fit_probe`'s default — `"linear_shipped"`
  IS the shipped probe.

**The measurements this task is built on**, all re-derivable from the fixture
below and none to be taken on trust:

| shape (n, D) | best ridge | `C` shipped grid | `C` fine grid | floor shipped | floor fine |
| --- | --- | --- | --- | --- | --- |
| 2000, 1024 | 3.16e3 | 0.4855 | 0.4964 | **+0.0109** | -0.0001 |
| 2000, 2048 | 5.62e3 | 0.4957 | 0.5164 | **+0.0201** | -0.0000 |
| 2000, 37632 | flat | 0.4643 | 0.4643 | +0.0000 | +0.0000 |
| 300, 150 (the fixture) | 177.8 | 0.3099-0.3459 | 0.4022-0.4510 | **+0.0894 to +0.1106** | -0.0016 to +0.0000 |

The floor and the oracle-ridge gain coincide at every shape where the gap
exists, which is why refining the step removes it rather than merely shrinking
it. The fixture shape is the one to test on: the effect is 0.09-0.11 and a draw
takes 0.4 s, where the n=2000 shapes take ~4 s each.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_probe.py`:

```python
def test_the_ridges_parameter_changes_no_arithmetic_at_the_shipped_rungs():
    """`ridges=` is a pure EXTENSION: it adds rungs and moves no existing number.

    Thirteen milestones of recorded numbers came through `RIDGES`, so this is
    checked bitwise rather than to a tolerance, and in BOTH branches of
    `_solve_ladder`. The features are UNCENTRED so the dual's own centring is
    exercised: on centred input the two branches agree for a reason that would
    hide a centring fault.
    """
    fine = tuple(10.0 ** (-1.0 + 8.0 * i / 32) for i in range(33))
    assert tuple(fine[i] for i in range(0, 33, 8)) == RIDGES, (
        "fixture must line the shipped rungs up with every 8th fine rung"
    )
    for n, d in ((400, 60), (60, 400)):          # d < n, then n <= d
        rng = np.random.default_rng(0)
        x = rng.normal(size=(n, d)) + 5.0
        y = rng.normal(size=(n, 4)) * 240.0
        wide, shipped = _solve_ladder(x, y, fine), _solve_ladder(x, y, RIDGES)
        for j, i in enumerate(range(0, 33, 8)):
            assert np.array_equal(wide[i], shipped[j]), (
                f"n={n} d={d} rung {RIDGES[j]:g} moved by "
                f"{np.abs(wide[i] - shipped[j]).max():.3g}"
            )


def test_fit_probe_defaults_to_the_shipped_grid():
    """The default path is byte-identical to the one without the parameter.

    Guards the frozen constant itself, and that `fit_probe` passes `ridges`
    through rather than reaching past it to the module global.
    """
    assert RIDGES == (1e-1, 1e1, 1e3, 1e5, 1e7), "the shipped grid is frozen"
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(120, 30)), rng.normal(size=(120, 4)) * 240.0
    xv, yv = rng.normal(size=(60, 30)), rng.normal(size=(60, 4)) * 240.0
    default, explicit = fit_probe(x, y, xv, yv), fit_probe(x, y, xv, yv, ridges=RIDGES)
    assert default["ridge"] == explicit["ridge"]
    assert np.array_equal(default["w"], explicit["w"])
    narrow = fit_probe(x, y, xv, yv, ridges=(RIDGES[0],))
    assert narrow["ridge"] == RIDGES[0], "the parameter must actually be used"
```

Append to `tests/eval/test_ceiling.py`:

```python
def _gapped_task(seed=0, n=300, d=150, rank=40, noise=120.0):
    """A linear task whose best ridge falls BETWEEN the shipped grid's rungs.

    Same construction as `_embedding_task`, at the shape where the grid's 100x
    step costs the most. The optimum is 177.8; `probe.RIDGES` offers 1e1 and 1e3
    and selects 1e3 in every draw, for a scored r2 0.09-0.11 below what the fine
    grid reaches. `_embedding_task` cannot show this -- at n=400 D=1024 the ridge
    response is flat enough that the shipped grid loses nothing.
    """
    g = np.random.default_rng(777)
    a = g.normal(size=(rank, d)); w = g.normal(size=(rank, 4))
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, rank))
    x = z @ a + rng.normal(size=(n, d)) * 3.0
    return x, (z @ w) * 20.0 + rng.normal(size=(n, 4)) * noise


def test_the_ceiling_grid_contains_the_shipped_grid_and_is_finer():
    """Containment by VALUE, and the rung count, pinned separately.

    Containment alone does not pin the step: a 17-rung 10^0.5 grid contains
    `RIDGES` too, and it left a +0.0004 floor at D=2048 where 33 rungs left
    -0.0000. So the count is asserted as well.
    """
    assert len(CEILING_RIDGES) == 33
    assert set(RIDGES) <= set(CEILING_RIDGES)
    assert CEILING_RIDGES[0] == RIDGES[0] and CEILING_RIDGES[-1] == RIDGES[-1]
    assert all(a < b for a, b in zip(CEILING_RIDGES, CEILING_RIDGES[1:]))


@pytest.mark.parametrize("seed", range(3))
def test_the_shipped_grid_straddles_the_best_ridge_and_the_fine_grid_resolves_it(seed):
    """The whole reason `C` does not use `probe.RIDGES`.

    The fixture is shown to discriminate first: the shipped grid must select an
    ENDPOINT of the interval the optimum lies in, so there is something to
    resolve. Then the fine grid must land strictly inside that interval and read
    materially higher on a third split.
    """
    train, select, score = (_gapped_task(3 * seed + i) for i in range(3))
    shipped = fit_probe(*train, *select)
    assert shipped["ridge"] == 1e3, f"fixture must straddle; shipped picked {shipped['ridge']:g}"

    fine = fit_linear_probe(*train, *select)
    assert 1e1 < fine["ridge"] < 1e3, f"fine grid picked {fine['ridge']:g}, not interior"
    gap = _score_on(fine, score) - _score_on(shipped, score)
    assert gap > 0.05, f"fine grid bought only {gap:+.4f}"


@pytest.mark.parametrize("seed", range(3))
def test_the_null_floor_is_near_zero_on_the_grid_the_ceiling_searches(seed):
    """`D - C` where there is nothing nonlinear to find.

    The fixture is shown to discriminate by measuring the SAME quantity over the
    shipped grid, which must come back far above zero -- otherwise a test that
    only checks `null_floor` is near zero would pass on a coarse grid too.
    """
    coarse = _null_floor_over(RIDGES, 300, 150, seed=seed)
    assert coarse > 0.05, f"fixture must show a coarse-grid floor; got {coarse:+.4f}"
    assert abs(null_floor(300, 150, seed=seed)) < 0.005, (
        f"floor is {null_floor(300, 150, seed=seed):+.4f} on CEILING_RIDGES"
    )
```

`_null_floor_over(ridges, n, d, *, seed)` is a TEST helper, not production code:
it is `null_floor`'s body with the grid as a parameter. Write it in the test
file by calling `null_floor` under `monkeypatch.setattr(ceiling, "CEILING_RIDGES",
ridges)` if that is cleaner than duplicating the body — **prefer the
monkeypatch**, so the helper cannot drift from the function it is validating.
Adjust the test signature to take `monkeypatch` accordingly.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/eval/test_probe.py tests/eval/test_ceiling.py -k "ridges_parameter or defaults_to_the_shipped or ceiling_grid_contains or straddles or null_floor" -v
```

Expected: FAIL — `ImportError: cannot import name 'CEILING_RIDGES'`, and
`fit_probe() got an unexpected keyword argument 'ridges'`.

- [ ] **Step 3: Write the implementation**

In `src/mbfps/eval/probe.py`, change `fit_probe`'s signature and its ladder loop
ONLY:

```python
def fit_probe(latents, targets, val_latents=None, val_targets=None, ridge=None,
              ridges: Sequence[float] = RIDGES) -> dict:
```

and

```python
    for ridge, w in zip(ridges, _solve_ladder(xs, y, ridges)):
```

Add to `fit_probe`'s docstring, after the existing paragraphs:

```
    `ridges` is the grid SELECTION searches, defaulting to the shipped `RIDGES`
    so no existing caller moves. It exists because `RIDGES` steps by 100x and
    the optimum is interior: measured, the shipped grid selects 1e3 where the
    best ridge is 177.8 and gives up 0.09-0.11 of scored R^2 for it (n=300,
    D=150), and +0.011 to +0.021 at n=2000, D=1024-2048. `ceiling.CEILING_RIDGES`
    is the finer grid, and it CONTAINS `RIDGES` bitwise, so passing it adds
    rungs and moves no existing number. `RIDGES` stays the default because
    thirteen milestones of recorded numbers came through it.
```

Append to `src/mbfps/eval/ceiling.py`:

```python
CEILING_RIDGES: tuple[float, ...] = tuple(
    10.0 ** (-1.0 + 8.0 * i / 32) for i in range(33)
)
"""The ridge grid BOTH sides of `D - C` search: 33 rungs at a 10^0.25 step over
`probe.RIDGES`'s span, [1e-1, 1e7].

`RIDGES` steps by 100x, and that is too coarse to resolve an interior optimum.
Measured on a purely linear task, the kernel ladder beat the linear reference by
+0.0109 at D=1024 and +0.0201 at D=2048 (n=2000) -- and the gain an ORACLE ridge
would have bought the linear reference at the same shapes is +0.0110 and +0.0208.
The same number. The RBF rungs were not finding nonlinear structure, they were
interpolating the ridge grid, and `PROBE_LIMITED` would have fired on it.

On this grid the floor reads -0.0016 to +0.0000, and the linear reference rises
to exactly the oracle's value. The SPAN is unchanged -- the optimum was interior
at every shape measured, 0 of 6 draws at an endpoint -- so only the step was
wrong. Where the ridge response has no sharp interior optimum, the raw-pixel row
at D >> n, there was no floor on either grid.

IT CONTAINS `RIDGES` BITWISE, as every 8th rung, and `probe._solve_ladder`
writes each rung's diagonal from a pristine copy rather than from the previous
rung's. So this grid ADDS rungs and moves no number any earlier milestone
recorded. `probe.RIDGES` itself is untouched."""


def fit_linear_probe(x, y, val_x, val_y) -> dict:
    """The linear reference `C`: the shipped probe over `CEILING_RIDGES`.

    NOT the shipped probe's own number -- that is `fit_probe` at its default,
    which M3o records separately as `shipped_grid_r2` and reconciles against the
    study. `C` is the best LINEAR readout, so that `D - C` isolates
    nonlinearity rather than ridge-grid resolution.
    """
    return fit_probe(x, y, val_x, val_y, ridges=CEILING_RIDGES)


_NULL_RANK: int = 40
_NULL_NOISE: float = 120.0


def _null_split(seed: int, n: int, d: int):
    """One split of a SYNTHETIC purely-linear task: a low-rank signal pushed
    through a fixed random map into `d` dimensions, plus isotropic nuisance.

    The map and the target weights come from a generator that does NOT depend on
    `seed`, so the three splits sample ONE task. Drawn from `seed`, each split
    would get its own map and the readouts would be fit on one task and scored
    on another.
    """
    g = np.random.default_rng(777)
    a = g.normal(size=(_NULL_RANK, d)); w = g.normal(size=(_NULL_RANK, 4))
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, _NULL_RANK))
    x = z @ a + rng.normal(size=(n, d)) * 3.0
    return x, (z @ w) * 20.0 + rng.normal(size=(n, 4)) * _NULL_NOISE


def null_floor(n: int, d: int, *, seed: int = 0) -> float:
    """`D - C` at this shape when the target is EXACTLY linear in the features.

    The honest answer is 0: nothing nonlinear is there to find. Anything
    materially above it is the ladder beating the linear reference for
    instrument reasons, which would read as `PROBE_LIMITED` and mean nothing
    about the representation.

    MEASURED PER CELL rather than taken from a figure, because the floor proved
    sharply shape-dependent: on the shipped grid it was +0.0201 at D=2048 and
    +0.0000 at D=37632. Three shapes do not license nine cells.
    """
    train, select, score = (_null_split(3 * seed + i, n, d) for i in range(3))
    linear = fit_linear_probe(*train, *select)
    kernel = fit_kernel_probe(*train, *select)
    return (_mean_r2(apply_readout(kernel, score[0]), score[1])
            - _mean_r2(apply_readout(linear, score[0]), score[1]))
```

Then change `fit_kernel_probe`'s jitter loop, `ceiling.py:186`:

```python
        for jitter in CEILING_RIDGES:
```

and in its docstring replace the two phrases that name the grid — "each over the
shipped `RIDGES`" and "the jitters are `fit_probe`'s own `RIDGES`" — with
`CEILING_RIDGES`, adding:

```
    THE LINEAR RUNG AND THE LINEAR REFERENCE SEARCH THE SAME GRID. Nesting is a
    claim about search spaces: with the ladder on `RIDGES` and `fit_linear_probe`
    on `CEILING_RIDGES` the ladder's linear rung would have 5 jitters against the
    reference's 33, and `D - C` would go NEGATIVE for grid reasons alone -- about
    -0.10 at n=300, D=150. Both are `CEILING_RIDGES`. The jitter ladder's length
    is nearly free here: each kernel is eigendecomposed once and every jitter is
    then O(n^2), where the decomposition is O(n^3).
```

**Six existing places in `tests/eval/test_ceiling.py` move with the grid.** The
first three fail if you do not change them; the last three PASS either way and
must still change, for the reason given.

Fails unless changed:

- `:122` — `for jitter in RIDGES:` in the brute-force ladder helper → `CEILING_RIDGES`
- `:258` — `assert len(entries) == (1 + len(BANDWIDTHS)) * len(RIDGES)` → `len(CEILING_RIDGES)`
- `:534` — the docstring phrase "over the same `RIDGES`" → `CEILING_RIDGES`

Do not weaken `:258`. It pins the ladder's exact rung count, and after this
change it reads `7 * 33 = 231`.

Passes either way, and still wrong — **the three nesting tests compare the ladder
against `fit_probe`, and after this change the ladder's search space contains
`fit_linear_probe`'s, not `fit_probe`'s.** Point all three at
`fit_linear_probe`:

- `:497` `test_the_kernel_readout_is_never_worse_than_the_linear_probe`
- `:531` `test_the_kernel_readout_cannot_select_below_the_linear_probe_on_its_selection_split`
- `:547` `test_the_linear_rung_is_the_shipped_probe_and_not_an_approximation_of_it`

`D >= C_shipped` does remain true, because `CEILING_RIDGES` CONTAINS `RIDGES` —
which is exactly why these keep passing and why that is not evidence. The claim
worth testing is the exact one: the ladder's linear rung reproduces the readout
that searches the SAME grid.

`:547` is the sharpest case. Measured on its fixture, `fit_probe` and
`fit_linear_probe` both select ridge 1000, so `readout["jitter"] == probe["ridge"]`
holds by coincidence — the optimum happens to land on a shipped rung at
n=400, D=1024. On `_gapped_task` it would not: 1e3 against 177.8. Rename it to
`..._is_the_linear_reference_and_not_an_approximation_of_it` and state the
grid in its docstring.

`Sequence` is already imported in `probe.py` (`_solve_ladder` annotates with it).
`ceiling.py` already imports `fit_probe`, `apply_probe`, `_mean_r2` and `RIDGES`
from `probe`; `RIDGES` stays imported — the containment test reads it.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/eval/test_probe.py tests/eval/test_ceiling.py -q
```

Read the `N passed` line yourself. Expected: all pass, 149 + 10 = 159.

Then confirm nothing else moved:

```bash
.venv/bin/python -m pytest -q
```

A changed number anywhere else means `ridges=` is not a pure extension. Report
the full count.

- [ ] **Step 5: Verify the tests bite**

Commit first, then:

| mutation | test that must fail |
| --- | --- |
| `fit_probe`: ladder loop reads the module `RIDGES` instead of `ridges` | `test_fit_probe_defaults_to_the_shipped_grid` (the `narrow` assertion) and `test_the_shipped_grid_straddles_the_best_ridge_and_the_fine_grid_resolves_it` |
| `CEILING_RIDGES`: `range(33)` -> `range(17)` with step `4.0` (a 10^0.5 grid) | `test_the_ceiling_grid_contains_the_shipped_grid_and_is_finer` on the length assertion. **Containment still holds for this mutant** — that is why the count is asserted separately. |
| `CEILING_RIDGES`: span `-1.0 ... 7.0` -> `-1.0 ... 6.0` | the containment assertion (1e7 leaves the grid) |
| `fit_linear_probe`: drop `ridges=CEILING_RIDGES` | `test_the_shipped_grid_straddles...` (fine grid picks 1e3) and `test_the_null_floor_is_near_zero...` |
| `fit_kernel_probe`: jitter loop back to `RIDGES` | `test_the_null_floor_is_near_zero...` — the floor goes to about -0.10, which the `abs(...) < 0.005` assertion catches. Also `test_the_kernel_readout_cannot_select_below_the_linear_probe_on_its_selection_split`, whichever of Task 2's nesting tests compares against `fit_probe`: **check which of those now compare the ladder against `fit_probe` (shipped grid) rather than `fit_linear_probe`, and report it.** A nesting test left on `fit_probe` is testing the wrong containment after this change — the ladder no longer contains the shipped grid's selection, it contains the finer one's. |
| `null_floor`: `_null_split(3 * seed + i, ...)` -> `_null_split(seed, ...)` | all three splits become identical, so `D - C` is measured in-sample. **Verify this bites**; if the floor stays under 0.005 in-sample the test cannot see it, so add an assertion that the three splits differ and report that you did. |
| `null_floor`: return `C - D` instead of `D - C` | `test_the_null_floor_is_near_zero...` only if the coarse-grid discrimination assertion runs first — it reads +0.1 and would read -0.1. Confirm the sign assertion is `> 0.05`, not `abs(...) > 0.05`. |

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/probe.py src/mbfps/eval/ceiling.py tests/eval/test_probe.py tests/eval/test_ceiling.py
git commit -F - <<'EOF'
feat: one ridge grid for both sides of D - C, fine enough to resolve the optimum

Task 2's linear rung made D >= C exact on the selection split. It did not make
D - C mean nonlinear capacity: probe.RIDGES steps by 100x and the optimum is
interior, so the RBF rungs reached regularisation the linear reference could
not. The floor on a purely linear task (+0.0109 at D=1024, +0.0201 at D=2048,
n=2000) EQUALS the gain an oracle ridge buys the linear reference (+0.0110,
+0.0208) -- the ladder was interpolating the grid, not finding structure.

CEILING_RIDGES is 33 rungs at 10^0.25 over the same span. The floor then reads
-0.0016 to +0.0000 and C rises to the oracle's value exactly. The span was
already right: the optimum was interior in 0 of 6 draws at an endpoint.

probe.RIDGES is untouched and stays fit_probe's default, so no earlier
milestone's number moves. CEILING_RIDGES contains it bitwise as every 8th rung,
and _solve_ladder writes each rung from a pristine diagonal, so passing the
finer grid adds rungs and changes no arithmetic -- asserted bitwise in both
branches on uncentred features.

null_floor measures D - C per cell at the cell's own shape. The floor proved
sharply shape-dependent (+0.0201 at D=2048, +0.0000 at D=37632), so three
shapes do not license nine cells.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

## Task 3: the grid, and the instrument's quality on the applied distribution

**Files:**
- Modify: `src/mbfps/eval/ceiling.py` (append)
- Test: `tests/eval/test_ceiling.py` (append)

**Interfaces:**
- Consumes: Task 2's estimators and Task 2b's `fit_linear_probe`; `probe.probe_targets`, `probe._block_bootstrap_ci`, `probe._validate_groups`
- Produces:
  ```python
  FRAME_BUDGET: int = 12000
  ZERO_INFORMATION_MEDIAN: float = 283.05

  @dataclass(frozen=True)
  class Readout:
      name: str          # "linear" | "linear_shipped" | "rbf" | "knn"
      r2: float
      mean_error: float        # map units. THE statistic M3n's 131.88-259.06 is
      median_error: float      # map units, robust; the unit 283.05 is in
      capacity: float          # the selected ridge, bandwidth or k
      predicted: np.ndarray    # kept so a difference can be bootstrapped

  def score_readout(name, fit, select, score) -> Readout
  def zero_information_error(fit_targets, score_targets) -> dict   # {"mean", "median"}
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
    mean_error: float
    """MEAN Euclidean position error in map units, over the scored rows.

    Carried beside r2 because r2 is not the unit the decision is made in. M3n
    was unreadable because a 0.576-11.824 signal sat under a 131.88-259.06
    readout error, and a result in r2 alone cannot be set against that.

    THE MEAN IS THE ONE THAT COMPARES TO 131.88-259.06. `rollout.py:237`
    averages the per-window position error across windows, so that figure is a
    mean; a median reported against it would be a different statistic in the
    same unit."""
    median_error: float
    """MEDIAN of the same errors, carried beside the mean.

    Robust to the tail a readout produces when it extrapolates off its support,
    and it is the statistic `ZERO_INFORMATION_MEDIAN` is expressed in. Every
    ratio must state which statistic is on each side: 131.88-259.06 against
    283.05 is a mean over a median and is not one."""
    capacity: float
    predicted: np.ndarray
    """The scored-row predictions, kept so a difference between two readouts
    can be bootstrapped with both held fixed."""


_FITTERS = {
    "linear": fit_linear_probe,      # C: the best linear readout, CEILING_RIDGES
    "linear_shipped": fit_probe,     # C_shipped: the study's own probe, RIDGES
    "rbf": fit_kernel_probe,
    "knn": fit_knn_probe,
}
"""Two linear entries, and the difference is the whole point of Task 2b.

`"linear"` searches `CEILING_RIDGES` and is what `D - C` is read against, so the
difference isolates nonlinearity rather than ridge-grid resolution.
`"linear_shipped"` is `fit_probe` at its default and is the ONLY one that
reconciles against the study's `embedding_selection_r2` -- it is what exit 49
reads. Gating on `"linear"` would exit 49 on a correct run, because the finer
grid is expected to read up to ~0.02 ABOVE the shipped band."""

_CAPACITY_KEY = {
    "linear": "ridge", "linear_shipped": "ridge",
    "rbf": "bandwidth", "knn": "k",
}


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
        mean_error=float(np.mean(position_error(predicted, score[1]))),
        median_error=float(np.median(position_error(predicted, score[1]))),
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

Add to the module's imports: `from dataclasses import dataclass` and extend the `probe` import with `_block_bootstrap_ci`, `_validate_groups` and **`position_error`** — the codebase's single definition of the map-unit metric. Do not re-spell `np.linalg.norm(...[:, :2] - ...[:, :2], axis=1)`; `burden.py`'s module docstring records that duplicating this metric is the drift that already destroyed a signal once in this project. `fit_linear_probe` is Task 2b's and is already in this module. Add `CONFIDENCE: float = 0.95` and `RESAMPLES: int = 2000` beside the other constants.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/eval/test_ceiling.py -q
```

Read the `N passed` line yourself. Expected: the three new tests pass on top of
Task 2b's total; report the count rather than matching a number from this plan.

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

        `probe_gain` here is already `D - C - null_floor`: the cell's own
        measured floor is subtracted before the interval is drawn, so
        `_resolvable` still asks whether the whole interval is above zero. A
        decade ridge grid alone buys +0.012 to +0.020 on a purely linear task,
        which would otherwise fire this verdict for a reason that has nothing
        to do with nonlinear structure.

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
  "frames": 12000, "frame_stride": int, "rows_per_split": {"fit": int, "select": int, "score": int},
  "confidence": 0.95, "resamples": 2000,
  "bootstrap_seed": int, "cell_bootstrap_seed": int, "split_seed": int,
  "zero_information_median": 283.05,
  "episodes": {"fit": [...], "select": [...], "score": [...]},
  "groups": [...],
  "applied": {"r2": float, "median_error": float},
  "ceiling_ridges": [...],        # the 33 rungs, stored; no literal survives the tests
  "readouts": {
    # every readout carries ALL of these; see Task 3's `Readout`
    "pixels_linear": {"r2": f, "mean_error": f, "median_error": f,
                      "capacity": f_or_str, "jitter": f_or_null},
    "pixels_rbf":        {...}, "embedding_linear":        {...},
    "embedding_rbf":     {...}, "embedding_knn":           {...},
    "sequence_linear":   {...}, "sequence_rbf":            {...},
    "embedding_linear_shipped": {...},   # C_shipped, the reconciliation anchor
  },
  "shipped_grid_r2": float,      # C_shipped: the "linear_shipped" readout's r2
  "fine_minus_shipped_on_score": float,   # C - C_shipped on the SCORE split; a
                                 # large systematic dip means the finer grid is
                                 # overfitting selection. Recorded, not gated.
  "null_floor": float,           # ceiling.null_floor(n, d) at THIS cell's shape
  "gains": {
    "probe":    {"point": f, "ci_low": f, "ci_high": f},
    "pixel":    {...}, "sequence": {...},
  },
  "controls": {
    "harness_r2_matches_study": bool,
    "study_embedding_selection_r2": float,
    "nesting_rbf_at_widest_vs_linear": float,
    "zero_information_recomputed": {"mean": float, "median": float},
  },
}
```

**`shipped_grid_r2`, not `embedding_linear`, is what `harness_r2_matches_study`
reads and what exit 49 fires on.** `embedding_linear` searches `CEILING_RIDGES`
and is expected up to ~0.02 ABOVE the study's 0.320-0.345 band, so reconciling on
it would exit 49 on a correct run. Spec §3.3.1.

**`capacity` ALSO DROPPED THE RBF'S JITTER — fixed in Task 3, not here.**
`fit_kernel_probe` selects bandwidth and jitter JOINTLY, and `score_readout`
discards the fitted dict, so nothing downstream could have recovered the jitter
without refitting the ladder. `Readout` therefore carries `jitter` beside
`capacity`: the RBF's selected jitter, the two linear readouts' selected ridge
(the same quantity in the dual form), and `None` for k-NN. Record it.

**`capacity` IS `inf` ON EVERY CELL WHERE NOTHING NONLINEAR HELPS, and `inf` is
not JSON.** The RBF ladder's linear rung reports `bandwidth = inf`, which is
exactly what wins when the representation carries no nonlinear structure — the
`PROBE_LIMITED`-is-false case, so the common one. `json.dumps` writes it as the
bare token `Infinity`, which no strict JSON reader accepts, and `allow_nan=False`
raises instead. **Serialise it as the string `"inf"`** — settled, because that needs nothing
added to `Readout`, where `None` would require a `kernel` field first. Apply the
same rule to every readout's `capacity`, not just the RBF's, and **assert in the
read phase that no record round-trips through
`json.loads(..., parse_constant=...)` with a non-finite capacity.** `jitter` is
already `None` for k-NN and serialises as JSON `null`, so it needs no rule.

**`null_floor` is measured at the cell's own `(n, d)`** — the `n` after the
stride subsample and the `d` of the embedding feature set, the same shape
`embedding_rbf` and `embedding_linear` are fit at. It costs one extra ladder fit
per cell. Record it even when it comes back at zero: a floor materially above
zero is a finding about the instrument, and Task 4 reads `probe_gain` against it.

```python
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

**THE PROBE GAIN GOES THROUGH `ceiling.probe_gain_over_floor`, NOT
`difference_interval`.** Task 4 added it because nothing else forms
`D - C - max(null_floor, 0)`: it takes `(d, c, floor, score_targets, groups, *,
seed)`, refuses `c.name != "linear"` and `d.name != "rbf"`, and clamps a negative
floor. If this phase forms `gains["probe"]` with `difference_interval` directly,
every cell reads against ZERO and nothing in `CeilingCell` can tell — which is
the exact defect Task 2b exists to have fixed. **Add a read-phase test that a
`D - C` interval lying below `null_floor` does not yield `PROBE_LIMITED`.**

**`B - D` AND `F - B` HAVE NO NAME GUARD, because B, D and F are all `"rbf"`.**
`probe_gain_over_floor` can refuse a swapped C, but the pixel and sequence gains
cannot be protected that way: exchanging the two arguments returns a wrong-sign
gain and nothing refuses it. Pin each by VALUE on a fixture where the expected
sign is known by construction — raw pixels beating the representation for the
pixel gain, and history beating one frame for the sequence gain.

**MAP UNITS ARE COMPARED HERE, NOT IN TASK 4.** Nothing in the reading touches a
map-unit statistic. This phase is where `mean_error` is set against the
zero-information MEAN and `median_error` against `ZERO_INFORMATION_MEDIAN =
283.05`, each ratio naming its statistic, and where M3n's 131.88-259.06 — a mean
— is compared only to `mean_error`.

**THE FIT SPLIT IS THE STUDY'S 20 TRAINING EPISODES.** Dividing the 24
validation episodes three ways starves the fit and `C_shipped` reads -0.026 to
0.166 against the 0.3195-0.3451 band, so exit 49 fires on all nine cells of a
correct run. Criterion 4 fits on TRAINING episodes —
`study.py` calls `filtering_report(model, train_paths, val_paths, ...)` — and fit
there the same rows read 0.300 to 0.342, worst gap 0.035. Fit on the 20 training
episodes strided to the frame budget; select on 12 validation episodes; score on
the other 12. Spec §3.4. `prepare_shared` already receives the training paths.

**THE RECONCILIATION FIELD IS `filtering.criterion_4.embedding_r2`, NOT
`probe.embedding_selection_r2`.** The band 0.3195-0.3451 is the former; the
latter spans 0.0180 to 0.3829 over the same nine cells, a factor of 21, and is a
different quantity. Gate `harness_r2_matches_study` on
`study_embedding_r2` (the criterion-4 field) and record
`study_embedding_selection_r2` beside it, ungated. Spec §2.

`measure_cell` then builds the remaining splits by **episode** from `episode_split`, subsamples to `FRAME_BUDGET` by a fixed stride recorded as `frame_stride`, scores the seven readouts through `ceiling.score_readout`, forms the three gains through `ceiling.difference_interval` at `cell_bootstrap_seed(...)`, and records the four controls. The raw-pixel readouts use no model, so compute them once and reuse across cells — assert they are identical across the nine records in the read phase, which is the free bitwise control.

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
  --frames 4000 --bootstrap-seed 0 > runs/m3o_ceiling/measure.log 2>&1
echo "measure exit: $?"
tail -20 runs/m3o_ceiling/measure.log
ls runs/m3o_ceiling/ceiling_*.json | wc -l    # must print 9
```

The `mkdir` and the redirect are both load-bearing: `tee` opens its file before the script creates `--out`, and a pipe makes `$?` the pager's status.

**`--frames 4000` is the settled budget, not a default.** At stride 1 the fit
split is 9,044 rows and the kernel ladder's `n^3.04` makes nine cells ~7 hours
with an unmeasured multi-GB raw-pixel ladder; `--frames 4000` gives stride 3 —
fit 3,016 / select 1,905 / score 2,088 — for ~15 minutes. Striding keeps every
episode and thins only within-episode density, where the redundancy is. Verify
`frame_stride` is 3 and `rows_per_split` matches in every record, and **state in
the results what the reduced `n` costs row B as a ceiling claim.**

- [ ] **Step 3: Check every control before reading any statistic**

```bash
.venv/bin/python - <<'EOF'
import glob, json
for p in sorted(glob.glob("runs/m3o_ceiling/ceiling_*.json")):
    c = json.load(open(p))["controls"]
    print(p.split("/")[-1], c)
EOF
```

Required, each corrected against what the quantity actually reads — all three of
the original checks would have misfired on a correct run:

- `harness_r2_matches_study` true in 9 of 9. **Measured: true in 9 of 9**, with
  `C_shipped` at 0.292-0.313 against the study's 0.3195-0.3451 and a worst gap of
  0.0398, comfortably inside `HARNESS_R2_TOLERANCE = 0.10`.
- **`study_embedding_r2` within 0.3195-0.3451** — the criterion-4 field. NOT
  `study_embedding_selection_r2`, which spans 0.0180-0.3829 and would fail; it is
  recorded ungated.
- **`zero_information_recomputed` is reported, not gated against 283.05.** The
  recomputed pair is 291.36 mean / 275.02 median over all 98 training episodes;
  the spec's 292.82 / 283.05 come from the leading 24 training episodes, a subset
  nothing in the study uses. Assert instead that **both** map-unit ratios use the
  recomputed pair — `mean_error` over the recomputed mean and `median_error` over
  the recomputed median — because dividing the median by the pinned 283.05 while
  dividing the mean by a recomputed figure is two different constants in one
  table.
- **`nesting_rbf_at_widest_vs_linear` is EXPECTED NEGATIVE** on good cells and is
  recorded ungated. It is the asymptotic nesting claim, which §3.3 measured to be
  false in practice. Nesting is structural — the linear kernel is a rung — and is
  pinned by test, not here. The original "not materially negative" check had the
  sign backwards.
- The gap against the study is **one-signed**: the study sits above `C_shipped` in
  9 of 9, because its ridge is selected on the rows it is scored on. Report that
  rather than treating it as noise.
- `fine_minus_shipped_on_score` ranges -0.0083 to +0.0364, mean +0.0071, negative
  in 4 of 9. All three `pixel_ae` seeds dip 0.006-0.008. Not strongly negative, so
  the finer grid is not overfitting selection, but report the per-arm pattern.

Also assert the raw-pixel readouts are **identical across all nine records** —
that row uses no model, so a difference means a model leaked into a model-free
measurement.

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
