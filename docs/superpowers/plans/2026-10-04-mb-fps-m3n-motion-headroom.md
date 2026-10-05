# M3n Motion Headroom Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Place the world model's one-step map on an axis whose ends are copying (predict no motion) and perfect (predict as well as the probe permits), with an episode-clustered ruler, over the `(k, h)` ladder.

**Architecture:** Three probe-space error differences per `(k, h)` — `headroom = hold - floor`, `skill = hold - rung`, `deficit = rung - floor` — satisfying `skill + deficit == headroom` exactly. `hold_k(h)` is the floor's probe position at the rung's last re-grounding step, held forward; it comes from `Trajectories` fields that have existed since M3d, so `rollout.py` is not touched. The verdict is a two-sided test of `skill > 0` and `deficit > 0` behind a readability gate on `headroom > 0`, each read from an episode-clustered bootstrap interval.

**Tech Stack:** Python 3, numpy, torch, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-04-mb-fps-m3n-motion-headroom-design.md`

## Global Constraints

- Test command is `.venv/bin/python -m pytest`. There is no `pytest` entry point in the venv.
- Commit trailer is the literal string `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` in its own paragraph, on every commit, regardless of which model writes it.
- **Never** `git stash` — the stash stack is shared across worktrees and other sessions use it. Use a temporary WIP commit instead.
- **Never** `rm` anything under `runs/`. `runs/` is a symlink to `/Users/raphaelchen/Desktop/csgo-bot/runs`, outside every worktree, shared by all of them: a delete here is a delete everywhere.
- **Never** `git clean -fdx` — it destroys the session ledger at `.superpowers/sdd/`.
- After mutating a source file to check that a test bites, clear `__pycache__` under `src/`, `tests/` and `scripts/` before re-running.
- `src/mbfps/eval/motion.py` and `scripts/latent_motion.py` are **M3i's**. Do not touch them. `src/mbfps/eval/ladder.py` is also taken. M3n's module is `src/mbfps/eval/headroom.py` and its script is `scripts/motion_headroom.py`.
- `src/mbfps/eval/rollout.py`, `src/mbfps/eval/probe.py` and `scripts/trust_horizon.py` are **unchanged by this milestone**. Verify with `git diff --stat` at the end of every task.
- Exact constants, copied verbatim: `REPORTED_H = (1, 2, 3, 5, 8, 10, 15, 20, 30, 45)`, `DECISION_K = 1`, `DECISION_H = 1`, `IDENTITY_TOLERANCE = 1e-9`, `CONFIDENCE = 0.95`, `RESAMPLES = 2000`, `SECONDARY_SIGMAS = 2`, `SEEDS_MINIMUM = 3`, `ARMS_REQUIRED = 2`, `EXIT_UNREADABLE_HEADROOM = 47`, `EXIT_NO_MAJORITY = 48`. Protocol: `context = 5`, `horizon = 45`. `REGROUNDING_KS = (1, 3, 5, 15, 45)` is inherited from `diagnostics.py`, not redefined.
- The strict-majority bar is **computed**, never stored. The spec's "CELLS_REQUIRED = 5" is the value `strict_majority(9)` returns; `burden.strict_majority`'s own docstring forbids storing it, because 5 is a majority at 9 cells and a minority at 11.
- `RegroundingSweep.standard_error` returns **one** standard error. Anything recorded as a 2-SE figure doubles it, as `scripts/diagnose_dynamics.py:1430` does. Mislabelling one SE as two in M3m produced the opposite conclusion from the correct one.
- The median true one-step displacement, `3.9694722203504225`, is **recorded and used by no statistic**. It is the quantity M3m's design mistook for a probe-space scale.
- **Read every sweep control with `!= 0.0`, never with `> 0`.** `persistence_divergence` and `k_invariance_at_h1` propagate NaN through `np.ptp` and `np.max` by design, so a bad row surfaces as NaN rather than as a large number — and `nan > 0` is **False**, so a `> 0` test reports a broken measurement as passing. Binds `measure_cell`'s assembly, the read phase, and Task 8's control check.

### Test pre-commitments (every task)

Each is a defect class this arc has shipped. They are requirements, not advice.

1. **Fixtures must be shown to discriminate.** Before asserting anything about a result, assert the fixture produces what the assertion needs — `clusters >= 3`, `ci_low < ci_high`, a nonzero standard error. M3m's first comparison fixture yielded 2 clusters and an interval of exactly zero width; two of three target mutations passed through it, and the tell was three different quantities printing identical digits.
2. **No self-referential expectations.** An expected value never comes from the thing under test. M3m hit this twice; both times the test caught one side of a two-sided mutation and missed the other, because both halves moved together.
3. **The fake model is stochastic.** With a deterministic fake, `seed`, `device` and `feature_backbone` go unbound and a seed mutation reads a difference of 0.0.
4. **Every test calls the code under test.** M3m shipped a test that asserted a tautology over random numbers and would have passed against an empty file.
5. **Docstrings name only mutations the fixture can reach.** Six instances in this arc named a mutation inexpressible at that scope or never reached.
6. **Exact equality where exactness is load-bearing.** M3m nearly shipped `pytest.approx(rel=1e-6)` against a `sum`-vs-`max` mutation it would have accepted. Measured on M3n's own k=45 column, 8 of 9 cells have a residual of exactly 0.0 under both reductions and the ninth separates them by only 4x (1.421e-14 against 5.684e-14) — so a *relative* tolerance is the wrong instrument regardless of the gap's size, and the fixture must be built to separate the two rather than relying on real data to do it.

---

## File Structure

| file | change | responsibility |
| --- | --- | --- |
| `src/mbfps/eval/pooling.py` | modify | gains `clustered_interval`, which is `burden.margin_interval` generalised and rehoused beside `episode_bootstrap` and `percentile_interval` |
| `src/mbfps/eval/diagnostics.py` | modify | gains `ground_step`; `RegroundingSweep` gains `hold_position` / `window_hold_position` and two self-check methods; `regrounding_sweep` passes `keep_trajectories=True` |
| `src/mbfps/eval/headroom.py` | create | the three differences, their identity, the reading, the formatter |
| `scripts/motion_headroom.py` | create | measure / read / all phases, protocol check, seeds gate, record schema, exits 47 and 48 |
| `src/mbfps/eval/burden.py` | modify | loses `motion_margin`, `margin_interval` and all of Reading H; keeps the ladder |
| `scripts/prediction_burden.py` | modify | stops computing the contaminated margin; read phase points at M3n |
| `tests/eval/test_pooling.py` | modify | `clustered_interval` |
| `tests/eval/test_diagnostics.py` | modify | `ground_step`, the hold curves, the two reductions, the flag control |
| `tests/eval/test_headroom.py` | create | the differences, the identity, the gate, the four-way placement, the formatter |
| `tests/eval/test_motion_headroom_script.py` | create | the phases, the record schema, the exits |
| `tests/eval/test_burden.py` | modify | Reading H tests removed with the code |
| `tests/eval/test_prediction_burden_script.py` | modify | the repointed read phase |
| `tests/eval/test_diagnose_dynamics_script.py` | modify | exit registry gains `{47, 48}` |

---

## Task 1: `pooling.clustered_interval`

`burden.margin_interval` is **already** an episode-clustered bootstrap over an arbitrary `(windows, horizon)` array — only its name and docstring are margin-specific. M3n needs exactly this function three times per `(k, h)`, so it moves to `pooling.py` beside the other two resampling helpers before anything consumes it.

**Files:**
- Modify: `src/mbfps/eval/pooling.py` (add `clustered_interval`)
- Modify: `src/mbfps/eval/burden.py:238-301` (remove `margin_interval`)
- Modify: `scripts/prediction_burden.py:131` (import) and `:390` (call site)
- Test: `tests/eval/test_pooling.py`, `tests/eval/test_burden.py`

**Interfaces:**
- Consumes: `pooling.episode_bootstrap(labels, bootstrap, seed)`, `pooling.percentile_interval(replicates) -> (low, high, se)`
- Produces:
  ```python
  def clustered_interval(
      window_rows: np.ndarray,      # (n_windows, horizon), finite everywhere
      groups: np.ndarray,           # (n_windows,), one episode label per window
      *,
      h: int,                       # 1-based horizon step
      resamples: int,
      seed: int,
  ) -> tuple[float, float, float]:  # (point, ci_low, ci_high)
  ```
  Both `resamples` and `seed` are keyword-required with **no defaults**.

- [ ] **Step 1: Write the failing test**

Add to `tests/eval/test_pooling.py`:

```python
def test_clustered_interval_resamples_episodes_not_windows():
    """A window-level bootstrap would return a visibly narrower interval.

    Six episodes of four windows each, with the episode mean carrying all the
    signal: resampling windows averages the between-episode variation away,
    resampling episodes keeps it. The fixture's own discrimination is asserted
    first, because an interval of zero width would pass any comparison.
    """
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(6), 4)
    offsets = np.array([-9.0, -5.0, -1.0, 1.0, 5.0, 9.0])
    rows = (offsets[groups][:, None] + rng.normal(0.0, 0.1, size=(24, 3)))

    assert np.unique(groups).size == 6
    assert rows.shape == (24, 3)

    point, low, high = clustered_interval(
        rows, groups, h=1, resamples=500, seed=0
    )
    assert low < high, "fixture must produce a nonzero-width interval"
    assert low < point < high

    window_groups = np.arange(24)
    _p, w_low, w_high = clustered_interval(
        rows, window_groups, h=1, resamples=500, seed=0
    )
    assert (high - low) > 2.0 * (w_high - w_low), (
        "clustering by episode must widen the interval against clustering by "
        f"window; got {high - low:.4f} against {w_high - w_low:.4f}"
    )


def test_clustered_interval_refuses_a_defaulted_seed_and_a_single_cluster():
    """`seed` is keyword-required with no default, and one episode is refused.

    A defaulted seed is how a previous milestone shipped every cell drawing
    the same resamples; the point estimate is seed-free, so only the bounds
    move and a reader cannot tell nine identical draws from nine independent
    ones. One cluster is not a bootstrap.
    """
    rows = np.arange(12.0).reshape(4, 3)
    with pytest.raises(TypeError):
        clustered_interval(rows, np.arange(4), h=1, resamples=10)
    with pytest.raises(ValueError, match="at least two episodes"):
        clustered_interval(
            rows, np.zeros(4, dtype=int), h=1, resamples=10, seed=0
        )


def test_clustered_interval_refuses_a_nonfinite_value_outside_the_read_column():
    """A NaN anywhere is refused, not only in the column `h` reads.

    A NaN that reaches the mean returns `(nan, nan, nan)` with no exception
    and no indication of which input was bad, and that triple is what a
    verdict is read from.
    """
    rows = np.ones((4, 3))
    rows[2, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        clustered_interval(rows, np.arange(4) // 2, h=1, resamples=10, seed=0)
```

Import at the top of the file: `from mbfps.eval.pooling import clustered_interval`.

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/eval/test_pooling.py -k clustered_interval -v
```

Expected: FAIL at collection with `ImportError: cannot import name 'clustered_interval' from 'mbfps.eval.pooling'`.

- [ ] **Step 3: Move the function**

Cut lines 238–301 of `src/mbfps/eval/burden.py` (the whole `margin_interval` definition and its docstring) and paste into `src/mbfps/eval/pooling.py` immediately after `percentile_interval`, renamed and with the parameter renamed. The body is unchanged except that every occurrence of `window_margin` becomes `window_rows`, and `resamples` loses its default:

```python
def clustered_interval(
    window_rows: np.ndarray,
    groups: np.ndarray,
    *,
    h: int,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    """`(point, ci_low, ci_high)` for the mean of column `h` across windows.

    THE RESAMPLING UNIT IS THE EPISODE. There are 229 windows over 24 episodes
    on every shipped cell, and consecutive Doom frames are near-duplicates, so
    a window-level bootstrap counts correlated observations as independent ones
    and returns an interval several times too narrow. Measured on M3m's
    `window_margin` rows, the episode-clustered standard error at h=1 is 1.26x
    to 2.17x the iid one across the nine cells.

    THE DIRECTION OF THAT ERROR MATTERS. An understated standard error makes a
    quantity look MORE resolvable, so an iid ruler is the conservative choice
    for a claim of non-resolvability and the dangerous one for a claim of
    resolvability. M3n's verdict rests on claims of resolvability.

    `seed` AND `resamples` ARE KEYWORD-REQUIRED WITH NO DEFAULTS. A defaulted
    seed is how a previous milestone shipped every cell drawing the same
    resamples; the point estimate is seed-free, so only the bounds can move,
    and a reader cannot tell nine identical draws from nine independent ones
    by looking.

    `groups` must carry one label per window. A record whose ladder carried no
    clustering stores `windows.episode` as null, and its own comment requires a
    reader to refuse rather than treat every window as its own episode --
    falling back to `arange(n)` here would convert this into the window-level
    bootstrap the first paragraph rules out.

    `window_rows` must be finite EVERYWHERE, not only in the column read at
    `h`. A NaN that reaches the mean returns `(nan, nan, nan)` -- no exception,
    no indication of which input was bad -- and that triple is the quantity a
    verdict is read from, so a silent one is a silent wrong verdict.

    GENERALISED FROM `burden.margin_interval` (M3m), which read only the
    motion margin. Nothing about the resampling is margin-specific, and M3n
    reads three different difference arrays through it per (k, h).
    """
    window_rows = np.asarray(window_rows, dtype=np.float64)
    groups = np.asarray(groups)
    if window_rows.ndim != 2:
        raise ValueError(
            f"window_rows must be (windows, horizon); got {window_rows.shape}"
        )
    if not np.isfinite(window_rows).all():
        raise ValueError("every window_rows value must be finite")
    if groups.ndim != 1 or groups.size != window_rows.shape[0]:
        raise ValueError(
            "groups must carry one label per window; got "
            f"{groups.shape} for {window_rows.shape[0]} windows"
        )
    if np.unique(groups).size < 2:
        raise ValueError(
            "an episode-clustered bootstrap needs at least two episodes; got "
            f"{np.unique(groups).size}"
        )
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")
    if not 1 <= h <= window_rows.shape[1]:
        raise ValueError(
            f"horizon step must be in 1..{window_rows.shape[1]}, got {h}"
        )

    column = window_rows[:, h - 1]
    point = float(column.mean())
    replicates = np.array([
        float(column[index].mean())
        for index in episode_bootstrap(groups, resamples, seed)
    ])
    low, high, _se = percentile_interval(replicates)
    return point, low, high
```

- [ ] **Step 4: Update the one existing call site**

In `scripts/prediction_burden.py`, remove `margin_interval` from the `mbfps.eval.burden` import at line 131, add `from mbfps.eval.pooling import clustered_interval` beside the other pooling import, and change the call at line 390 from `margin_interval(` to `clustered_interval(`, passing `resamples=RESAMPLES` explicitly (it was relying on the default) and renaming the first argument if it is passed by keyword.

- [ ] **Step 5: Prove the move is behaviour-free**

```bash
BEFORE=$(mktemp) && AFTER=$(mktemp)
git show HEAD:src/mbfps/eval/burden.py | sed -n '/^def margin_interval/,/return point, low, high/p' > "$BEFORE"
sed -n '/^def clustered_interval/,/return point, low, high/p' src/mbfps/eval/pooling.py > "$AFTER"
diff "$BEFORE" "$AFTER"
```

Expected: the only differences are the two names (`margin_interval` -> `clustered_interval`, `window_margin` -> `window_rows`), the docstring, the removed `resamples` default, and the hoisted range check. Confirm in the report that no arithmetic line changed. Write the diff into the report file.

- [ ] **Step 6: Remove `margin_interval`'s tests from `test_burden.py` and reinstate them against the new name**

Every test in `tests/eval/test_burden.py` naming `margin_interval` moves to `tests/eval/test_pooling.py` with the name and first-argument changes applied, and nothing else changed. Do not delete coverage.

- [ ] **Step 7: Run the suites**

```bash
.venv/bin/python -m pytest tests/eval/test_pooling.py tests/eval/test_burden.py tests/eval/test_prediction_burden_script.py -q
```

Expected: PASS, no failures.

- [ ] **Step 8: Commit**

```bash
git add src/mbfps/eval/pooling.py src/mbfps/eval/burden.py scripts/prediction_burden.py tests/eval/test_pooling.py tests/eval/test_burden.py
git commit -F - <<'EOF'
refactor: margin_interval becomes pooling.clustered_interval

Nothing about the resampling was margin-specific: the function reads one
column of an arbitrary (windows, horizon) array and bootstraps the episode
labels. M3n reads three difference arrays through it per (k, h), so it moves
beside episode_bootstrap and percentile_interval rather than being copied.

resamples loses its default alongside seed. A defaulted resample count is
the same hazard as a defaulted seed: the point estimate does not move, so
only the bounds can, and a reader cannot tell the draws apart by looking.

The arithmetic is unchanged; the diff against the moved text is in the task
report.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

## Task 2: `ground_step`, the index expression

One expression carries the whole `hold_k` construction, and the project's standing rule is that index arithmetic is pinned empirically rather than argued. This task lands the expression and the oracle tests that pin it; Task 3 wires it to real curves where the two bitwise reductions check it again.

**Files:**
- Modify: `src/mbfps/eval/diagnostics.py` (add `ground_step` immediately above `regrounding_sweep`)
- Test: `tests/eval/test_diagnostics.py`

**Interfaces:**
- Consumes: nothing
- Produces: `def ground_step(k: int, h: int) -> int` — the 0-based horizon step at which the rung that uses re-grounding period `k` was last re-grounded before scoring step `h`. `0` means the last context frame.

- [ ] **Step 1: Write the failing test**

Add to `tests/eval/test_diagnostics.py`:

```python
def test_ground_step_matches_the_sweeps_own_segment_rule():
    """The segment boundary, derived from the sweep's docstring, not from the code.

    `regrounding_sweep`'s docstring states that segment `s` covers horizon
    steps `[s*k, min((s+1)*k, H))` and is grounded on the posterior through
    frame `start + context + s*k`. The expected values below are built from
    that sentence with an independent loop, so a mutation to `ground_step`
    cannot move both sides.
    """
    horizon = 45
    for k in (1, 2, 3, 5, 15, 45):
        expected = {}
        for s in range(0, (horizon + k - 1) // k):
            for step0 in range(s * k, min((s + 1) * k, horizon)):
                expected[step0 + 1] = s * k        # 1-based h -> grounding step
        assert len(expected) == horizon
        for h in range(1, horizon + 1):
            assert ground_step(k, h) == expected[h], (
                f"k={k} h={h}: {ground_step(k, h)} != {expected[h]}"
            )


def test_ground_step_is_zero_for_every_h_when_k_is_the_horizon():
    """k=45 never re-grounds, so every step holds from the last context frame.

    This is the reduction that makes hold_45 identical to persistence; if it
    failed, `hold_45` would hold from inside the horizon and the far corner of
    the ladder would stop being the gate's `beats_persistence`.
    """
    assert {ground_step(45, h) for h in range(1, 46)} == {0}


def test_ground_step_is_zero_at_h_one_for_every_k():
    """The h=1 column is k-invariant, which makes the one-step verdict k-free."""
    assert {ground_step(k, 1) for k in (1, 3, 5, 15, 45)} == {0}


def test_ground_step_refuses_a_nonpositive_k_or_h():
    """`k * ((h - 1) // k)` returns a plausible number for h=0 and k<=0.

    At h=0 Python's floor division gives `k * -1`, a NEGATIVE grounding step,
    which would index the floor's positions from the end of the horizon and
    silently hold from the wrong frame.
    """
    for bad_k, bad_h in ((0, 1), (-1, 1), (1, 0), (1, -3)):
        with pytest.raises(ValueError):
            ground_step(bad_k, bad_h)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/eval/test_diagnostics.py -k ground_step -v
```

Expected: FAIL with `NameError: name 'ground_step' is not defined` (add the import to the test module's existing `from mbfps.eval.diagnostics import ...` line first, in which case it fails at collection with `ImportError`).

- [ ] **Step 3: Write the implementation**

Insert into `src/mbfps/eval/diagnostics.py` immediately above `def regrounding_sweep(`:

```python
def ground_step(k: int, h: int) -> int:
    """The 0-based horizon step the k-rung was last re-grounded at before `h`.

    `0` means the last CONTEXT frame -- the state every segment-0 step is
    imagined from -- not horizon step 1.

    DERIVED FROM `regrounding_sweep`'s OWN SEGMENT RULE, quoted from its
    docstring: segment `s` covers horizon steps `[s*k, min((s+1)*k, H))` and is
    grounded on the posterior through frame `start + context + s*k`. The 1-based
    step `h` sits in segment `(h - 1) // k`, so the grounding step is
    `k * ((h - 1) // k)`.

    Both arguments are CHECKED. At `h = 0` floor division gives `k * -1`, a
    negative grounding step, which would index the floor's positions from the
    END of the horizon and hold from the wrong frame -- a wrong number, not a
    crash, and one that no shape test would catch.
    """
    if k < 1:
        raise ValueError(f"re-grounding period must be >= 1, got {k}")
    if h < 1:
        raise ValueError(f"horizon step must be >= 1 (1-based), got {h}")
    return k * ((h - 1) // k)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/eval/test_diagnostics.py -k ground_step -v
```

Expected: PASS, 4 tests.

- [ ] **Step 5: Verify the tests bite**

Mutate `return k * ((h - 1) // k)` to `return k * (h // k)`, clear caches, re-run, confirm failure, restore:

```bash
.venv/bin/python - <<'EOF'
import pathlib, re
p = pathlib.Path("src/mbfps/eval/diagnostics.py")
t = p.read_text()
assert t.count("return k * ((h - 1) // k)") == 1
p.write_text(t.replace("return k * ((h - 1) // k)", "return k * (h // k)"))
EOF
find src tests scripts -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; true
.venv/bin/python -m pytest tests/eval/test_diagnostics.py -k ground_step -q
git checkout src/mbfps/eval/diagnostics.py
find src tests scripts -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; true
```

Expected: the mutated run FAILS `test_ground_step_matches_the_sweeps_own_segment_rule` and `test_ground_step_is_zero_at_h_one_for_every_k`. Record both failure messages in the report. Then re-apply the implementation (the `git checkout` above reverts it) — commit before mutating, or re-write the function.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/diagnostics.py tests/eval/test_diagnostics.py
git commit -F - <<'EOF'
feat: ground_step, the index expression hold_k is built from

The 0-based horizon step a k-rung was last re-grounded at before scoring
step h, which is k * ((h - 1) // k) by the sweep's own segment rule. 0 means
the last context frame.

The expected values in the test are built from the sweep's docstring with an
independent loop rather than from the function, so a mutation cannot move
both sides. Both arguments are checked: at h=0 floor division returns a
negative grounding step, which would index the floor's positions from the
end of the horizon and hold from the wrong frame without crashing.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

## Task 3: `RegroundingSweep` carries the hold baseline

`hold_k(h)` needs the floor's probe-space positions per window per step. `_diagnose(..., keep_trajectories=True)` has kept them since M3d as `positions_real`, `positions_at_context` and `true_positions`, all on the canonical pass — which is the right scope, because `RegroundingSweep.reference` is documented as "the ONE floor and persistence every k is read against."

**Files:**
- Modify: `src/mbfps/eval/diagnostics.py:1743-1761` (the `RegroundingSweep` dataclass), `:1776-1823` (add two self-check methods near `curve_standard_error` / `RegroundingSweep.standard_error`), `:1933-1948` (`regrounding_sweep`'s `_diagnose` call and the `RegroundingSweep` construction)
- Test: `tests/eval/test_diagnostics.py`

**Interfaces:**
- Consumes: `ground_step(k, h)` from Task 2; `probe.position_error(predicted, true)` which takes `(n, >=2)` arrays and returns `(n,)` Euclidean distances over the first two columns
- Produces, on `RegroundingSweep`:
  ```python
  hold_position: dict[int, np.ndarray]         # k -> (horizon,), mean over windows
  window_hold_position: dict[int, np.ndarray]  # k -> (n_windows, horizon)
  window_episode: np.ndarray                   # (n_windows,), `_Pass`'s own labels

  def persistence_divergence(self) -> float
      """max |hold_position[k_max] - reference.persistence_position|. MUST be 0.0."""
  def k_invariance_at_h1(self) -> float
      """max spread across k of the three h=1 values. MUST be 0.0."""
  ```

- [ ] **Step 1: Write the failing test**

Add to `tests/eval/test_diagnostics.py`. **Use the rig that file already has**, not a new one: `write(tmp_path, synthetic_episode())` for the path, `real_model_and_probe()` for a REAL stochastic model and probe, the module-level `sweep(model, paths, probe, ks=..., device=device)` helper, the `rollout(...)` helper for `evaluate_rollout`, the `CONTEXT, HORIZON = 3, 5` constants imported from `tests/eval/test_rollout.py`, and the existing `device` fixture. `reference_trajectories` is public and already imported at line 47.

`HORIZON` is **5**, not 45 — the synthetic episode is 20 frames. Every `k` below is expressed against `HORIZON` so nothing hardcodes a horizon the rig cannot reach.

```python
@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_hold_position_at_k_equals_the_horizon_is_bitwise_persistence(tmp_path, device):
    """The ladder's far corner IS the gate's beats_persistence criterion.

    k=HORIZON never re-grounds, so every step holds the probe of the last
    context frame -- which is what `persistence_position` already is. Bitwise,
    not approximately: both are `position_error` on the same two arrays.

    On the REAL stochastic model, because against an oracle rig no random
    number is drawn and every mutation that displaces the sampling stream is
    invisible -- the reason the sibling self-check tests in this file say the
    same thing.
    """
    path = write(tmp_path, synthetic_episode())
    model, probe = real_model_and_probe()
    model = model.to(device)

    result = sweep(model, [path], probe, ks=(1, HORIZON), device=device)
    assert result.windows_total >= 2, "fixture must yield more than one window"
    np.testing.assert_array_equal(
        result.hold_position[HORIZON], result.reference.persistence_position
    )
    assert result.persistence_divergence() == 0.0


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_the_hold_and_rung_curves_are_k_invariant_at_the_first_step(tmp_path, device):
    """At h=1 every k grounds at step 0 and imagines one step, so all k agree.

    Checked on hold and rung SEPARATELY: if only their difference agreed, a
    compensating error in both would pass. The rung half extends the property
    `test_every_k_is_grounded_identically_at_the_first_horizon_step` already
    pins, to the curve this task adds.
    """
    path = write(tmp_path, synthetic_episode())
    model, probe = real_model_and_probe()
    model = model.to(device)
    ks = (1, 2, 3, HORIZON)

    result = sweep(model, [path], probe, ks=ks, device=device)
    holds = [result.hold_position[k][0] for k in ks]
    rungs = [result.curve(k)[0] for k in ks]
    assert len(np.unique(holds)) == 1, holds
    assert len(np.unique(rungs)) == 1, rungs
    assert result.k_invariance_at_h1() == 0.0


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_hold_position_holds_the_floors_own_position_from_the_grounding_step(
    tmp_path, device
):
    """The expectation is rebuilt from the SEGMENT RULE, never from `ground_step`.

    Task 2's docstring-derived oracle is the ONLY independent guard on
    `ground_step`'s interior, because both bitwise reductions above sit at
    `g == 0` -- k=HORIZON grounds every step at the context frame, and h=1
    grounds there for every k. Neither ever exercises
    `positions_real[:, g - 1]` for `g > 0`.

    So if this test called `ground_step` to build its expectation, a wrong
    interior mapping would move BOTH sides and nothing in the milestone would
    catch it. That is the self-referential expectation M3m hit twice, and both
    times the test caught one side of a two-sided mutation and missed the
    other.

    The loop below therefore enumerates segments FORWARD from
    `regrounding_sweep`'s own docstring rule -- segment `s` covers horizon
    steps `[s*k, min((s+1)*k, H))` and grounds on the posterior through step
    `s*k` -- with no floor division anywhere, and asserts that steps grounding
    INSIDE the horizon were actually reached.

    `reference_trajectories` rather than `_diagnose`: it is the public entry
    point, and it DRAWS the noise reference where the sweep does not, so a
    bitwise match here also pins the two passes to the same floor.
    """
    k = 2                      # not a divisor of HORIZON=5: the last segment is ragged
    path = write(tmp_path, synthetic_episode())
    model, probe = real_model_and_probe()
    model = model.to(device)

    result = sweep(model, [path], probe, ks=(k, HORIZON), device=device)
    kept = reference_trajectories(
        model, [path], probe, context=CONTEXT, horizon=HORIZON, seed=0,
        device=device, feature_backbone=None,
    )

    expected = np.zeros((kept.windows_total, HORIZON))
    covered, grounded_inside = set(), 0
    for s in range(0, (HORIZON + k - 1) // k):
        for step0 in range(s * k, min((s + 1) * k, HORIZON)):
            held = (
                kept.positions_at_context if s == 0
                else kept.positions_real[:, s * k - 1]
            )
            expected[:, step0] = position_error(held, kept.true_positions[:, step0])
            covered.add(step0)
            grounded_inside += 0 if s == 0 else 1
    assert len(covered) == HORIZON, "the oracle must cover every horizon step"
    assert grounded_inside == HORIZON - k, (
        f"{HORIZON - k} of {HORIZON} steps must ground INSIDE the horizon, or "
        "this test only re-checks the g == 0 case the two reductions already pin"
    )
    assert expected.std() > 0.0, "fixture must produce a non-constant hold curve"
    np.testing.assert_array_equal(result.window_hold_position[k], expected)
```

**The `keep_trajectories=True` control needs no new test.** Two tests already in this file compare the sweep against `evaluate_rollout`, which never sets the flag: `test_the_sweep_at_k_equals_the_horizon_is_bitwise_the_open_loop_rollout` and `test_the_sweeps_floor_and_persistence_are_bitwise_the_rollouts`. Once `regrounding_sweep` passes the flag, those two exercise the flag-on path and will fail if it moved the arms or the reference. Confirm in your report that both still pass on both devices, and name them as the control.

**If the third test does not match bitwise**, report the magnitude and STOP — do not relax it to a tolerance. A divergence there means the sweep's floor and `reference_trajectories`' floor are different measurements, which would be a finding about existing code rather than about this task, and it is the controller's to adjudicate.

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/eval/test_diagnostics.py -k "hold_position or k_invariant" -v
```

Expected: FAIL with `AttributeError: 'RegroundingSweep' object has no attribute 'hold_position'`.

- [ ] **Step 3: Add the three fields to the dataclass**

In `src/mbfps/eval/diagnostics.py`, after `window_floor_position` and its docstring (around line 1760) and before `windows_total`, insert:

```python
    hold_position: dict[int, np.ndarray]
    """Per k, the error of the HOLD baseline: the floor's own probe position at
    the rung's last re-grounding step, held forward and scored on the same
    frames as the rung.

    The copying end of M3n's axis. AT `g == 0` -- every `h <= k`, which is the
    whole `k == horizon` column, the entire h=1 row, and M3n's decision cell --
    the rung and this baseline descend from the SAME posterior latent, so per
    window their difference is of order the model's predicted displacement,
    units, while either error alone is 130-260. AT `g > 0` the rung's own
    re-grounding `observe` draws fresh categorical samples and the floor's runs
    elsewhere in the same stream, so the difference there also carries a
    sampling-redraw term; measured on the rig at k=2, z differs in 28 of 32
    groups at g=2 and 32 of 32 at g=4, max |dh| 0.305.

    Whether the readout error also cancels at `g > 0` is NOT established. The
    `g == 0` case is, and it is the one the verdict rests on. This is why M3m's
    `motion_margin` was unfixable by better statistics: it subtracted a
    ground-truth displacement, so there was no common latent for anything to
    cancel against at any (k, h).

    At `k == horizon` this is bitwise `reference.persistence_position`
    (`persistence_divergence`), and at h=1 it is identical for every k
    (`k_invariance_at_h1`)."""
    window_hold_position: dict[int, np.ndarray]
    """The hold baseline's OWN per-window rows, so `hold - rung` has a paired
    ruler. Keeping only the mean leaves the margin with no correct standard
    error available at all -- the same reason `window_floor_position` is
    retained."""
    window_episode: np.ndarray
    """Per window, the index of the CONTRIBUTING episode it was cut from --
    `_Pass`'s own labels, copied through.

    REQUIRED, because every interval M3n reads is clustered on these. M3m had
    to recover them by walking the validation episodes A SECOND TIME in
    `baseline_rows`, and `scripts/prediction_burden.py:215-218` records the
    hazard that came with it: `diagnostics` numbers episodes AFTER skipping a
    too-short one, so the two walks disagree once such an episode precedes
    another. M3n needs no second walk -- every array it reads comes from this
    sweep -- so carrying the labels here removes the mismatch rather than
    documenting it."""
```

- [ ] **Step 4: Add the two self-check methods**

In `src/mbfps/eval/diagnostics.py`, beside the existing `open_loop_divergence` (around line 1825), add:

```python
    def persistence_divergence(self) -> float:
        """SELF-CHECK: max |hold at k=horizon - persistence|. MUST be 0.0.

        k=horizon never re-grounds, so `ground_step` returns 0 at every step and
        the hold baseline is the probe of the last context frame held forward --
        which is what `persistence_position` already is. A nonzero value means
        the grounding index is wrong, and the ladder's far corner has stopped
        being the M3 gate's `beats_persistence`.

        Returns 0.0 when the sweep did not run k=horizon, because there is then
        nothing to check and a NaN would print as a failure.
        """
        if self.horizon not in self.hold_position:
            return 0.0
        return float(np.max(np.abs(
            self.hold_position[self.horizon] - self.reference.persistence_position
        )))

    def k_invariance_at_h1(self) -> float:
        """SELF-CHECK: the largest across-k spread at h=1. MUST be 0.0.

        At h=1 every k grounds at step 0 and the rung is one prior step from the
        context posterior, so the hold baseline, the rung and the floor are all
        k-independent there. This is what makes M3n's one-step verdict k-free by
        construction rather than by a choice of k.

        All three are checked separately. Checking only their sum would pass a
        compensating error in two of them.
        """
        spreads = []
        for series in (
            [self.hold_position[k][0] for k in self.ks],
            [self.curve(k)[0] for k in self.ks],
        ):
            spreads.append(max(series) - min(series))
        return float(max(spreads))
```

- [ ] **Step 5: Compute the hold curves in `regrounding_sweep`**

Change the `_diagnose` call at `src/mbfps/eval/diagnostics.py:1933` to add `keep_trajectories=True`, then replace the `RegroundingSweep(...)` construction with:

```python
    result = _diagnose(
        model, val_paths, embedding_probe_weights,
        arms={k: segmented(k) for k in ks}, context=context, horizon=horizon,
        seed=seed, device=device, feature_backbone=feature_backbone,
        noise_reference=False,
        # The hold baseline needs the floor's own probe POSITIONS, not its
        # errors, and these are the fields that carry them (M3d). The flag's
        # docstring states the pass draws nothing from the stream and does only
        # numpy on rows already in hand; `test_keep_trajectories_leaves_the_
        # sweeps_curves_bitwise_unchanged` holds it to that. `keep_latents`
        # stays False: the stage decomposition is not read here, and it is the
        # flag that would actually cost memory.
        keep_trajectories=True,
    )
    window_hold = {}
    for k in ks:
        rows = np.empty_like(result.arms[k]["position"])
        for h in range(1, horizon + 1):
            g = ground_step(k, h)
            held = (
                result.positions_at_context if g == 0
                else result.positions_real[:, g - 1]
            )
            rows[:, h - 1] = position_error(held, result.true_positions[:, h - 1])
        window_hold[k] = rows
    return RegroundingSweep(
        ks=tuple(ks),
        horizon=horizon,
        reference=result.reference,
        position={k: result.arms[k]["position"].mean(axis=0) for k in ks},
        angle={k: result.arms[k]["angle"].mean(axis=0) for k in ks},
        window_position={k: result.arms[k]["position"] for k in ks},
        window_floor_position=result.reference_windows["floor_position"],
        hold_position={k: rows.mean(axis=0) for k, rows in window_hold.items()},
        window_hold_position=window_hold,
        window_episode=result.window_episode,
        windows_total=result.windows_total,
    )
```

`position_error` is already imported in this module; confirm with `grep -n "position_error" src/mbfps/eval/diagnostics.py` before adding an import.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/eval/test_diagnostics.py -q
```

Expected: PASS, no failures. Then the whole eval suite:

```bash
.venv/bin/python -m pytest tests/eval -q
```

Expected: PASS. If `test_prediction_burden_script.py` or `test_diagnose_dynamics_script.py` constructs a `RegroundingSweep` positionally, the three new required fields will break it — add them there with the same values the real sweep would produce, not with zeros.

- [ ] **Step 7: Verify the reductions bite**

Mutate `g = ground_step(k, h)` to `g = ground_step(k, h + 1)`, clear caches, re-run, confirm `persistence_divergence` and the reconstruction test both fail by value, restore. Record the observed divergences in the report.

- [ ] **Step 8: Commit**

```bash
git add src/mbfps/eval/diagnostics.py tests/eval/test_diagnostics.py
git commit -F - <<'EOF'
feat: RegroundingSweep carries the hold baseline and its per-window rows

hold_position[k] is the error of the floor's own probe position at the
rung's last re-grounding step, held forward and scored on the same frames as
the rung -- the copying end of M3n's axis. The rung and the baseline descend
from the same posterior latent, so per window their difference is of order
the model's predicted displacement while either error alone is 130-260. The
readout error cancels in the difference, not in either error.

rollout.py is untouched. positions_real, positions_at_context and
true_positions have been kept on the canonical pass since M3d, and the
canonical pass is the right scope because one floor serves every k by the
sweep's own documented construction.

Two self-checks: hold at k=horizon is bitwise persistence_position, and the
h=1 column is identical across every k -- which is what makes the one-step
verdict k-free rather than a choice of k. Both check hold and rung
separately; checking only their sum would pass a compensating error in two
of them.

keep_latents stays False.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

## Task 4: `headroom.py` — the three differences and their identity

**Files:**
- Create: `src/mbfps/eval/headroom.py`
- Test: `tests/eval/test_headroom.py`

**Interfaces:**
- Consumes: `burden.at_horizon(curve, h)`, `burden.checked_pair(left, right)` — the shape guard M3m's ladder uses to refuse mismatched curves. Re-export nothing; import what is needed.
- Produces:
  ```python
  REPORTED_H: tuple[int, ...] = (1, 2, 3, 5, 8, 10, 15, 20, 30, 45)
  DECISION_K: int = 1
  DECISION_H: int = 1
  IDENTITY_TOLERANCE: float = 1e-9
  CONFIDENCE: float = 0.95
  RESAMPLES: int = 2000
  SECONDARY_SIGMAS: int = 2
  SEEDS_MINIMUM: int = 3
  ARMS_REQUIRED: int = 2
  DISPLACEMENT_RECORDED_ONLY: float = 3.9694722203504225

  def headroom(hold: np.ndarray, floor: np.ndarray) -> np.ndarray
  def skill(hold: np.ndarray, rung: np.ndarray) -> np.ndarray
  def deficit(rung: np.ndarray, floor: np.ndarray) -> np.ndarray
  def triple_residual(hold, rung, floor) -> float
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_headroom.py`:

```python
import numpy as np
import pytest

from mbfps.eval.headroom import (
    DECISION_H, DECISION_K, DISPLACEMENT_RECORDED_ONLY, IDENTITY_TOLERANCE,
    REPORTED_H, deficit, headroom, skill, triple_residual,
)


def test_the_three_differences_satisfy_the_identity_exactly():
    """skill + deficit == headroom, with `==`, not approx.

    The hold term cancels algebraically, so the only residual is float
    rounding. `pytest.approx(rel=1e-6)` here would ACCEPT a `np.sum`-for-`np.max`
    mutation inside `triple_residual`. Measured on M3n's own k=45 column, 8 of 9 cells give exactly 0.0
    under both reductions and the ninth separates them by 4x (1.421e-14 against
    5.684e-14), so real data barely separates them: the FIXTURE has to. M3m nearly shipped
    exactly that tolerance against exactly that mutation.
    """
    rng = np.random.default_rng(0)
    floor = 100.0 + rng.uniform(0.0, 150.0, size=45)
    rung = floor + rng.uniform(0.0, 5.0, size=45)
    hold = rung + rng.uniform(-2.0, 8.0, size=45)
    assert np.ptp(floor) > 50.0, "fixture must span a real range of floors"

    whole = headroom(hold, floor)
    parts = skill(hold, rung) + deficit(rung, floor)
    assert np.max(np.abs(whole - parts)) < IDENTITY_TOLERANCE
    assert triple_residual(hold, rung, floor) < IDENTITY_TOLERANCE


def test_triple_residual_reports_the_worst_step_not_their_sum():
    """One bad step must not be averaged or summed away.

    The fixture breaks the identity at exactly one step by a known amount, so
    the expected value is that amount -- derived from the construction, not
    from the function.
    """
    floor = np.full(10, 100.0)
    rung = np.full(10, 103.0)
    hold = np.full(10, 110.0)
    broken = triple_residual(hold, rung, floor + np.eye(10)[3] * 0.5)
    assert broken == pytest.approx(0.5, abs=1e-12)


def test_skill_is_positive_when_the_rung_beats_the_hold_baseline():
    """Sign convention, stated as a value rather than as prose.

    A sign flip in `skill` would invert every verdict and break nothing else:
    the identity still holds, because `deficit` would absorb it.
    """
    assert skill(np.array([110.0]), np.array([103.0]))[0] == pytest.approx(7.0)
    assert deficit(np.array([103.0]), np.array([100.0]))[0] == pytest.approx(3.0)
    assert headroom(np.array([110.0]), np.array([100.0]))[0] == pytest.approx(10.0)


def test_mismatched_curve_lengths_are_refused():
    """A length mismatch must raise, not broadcast.

    numpy would broadcast a length-1 curve against a length-45 one and return
    45 plausible numbers.
    """
    with pytest.raises(ValueError):
        headroom(np.ones(45), np.ones(44))
    with pytest.raises(ValueError):
        skill(np.ones(45), np.ones(1))


def test_the_displacement_constant_is_recorded_and_unused():
    """3.9694722203504225 appears in no arithmetic in this module.

    It is the ground-truth quantity M3m's design mistook for a probe-space
    scale, kept so a reader of the record can see it plays no part. The guard
    is textual because an unused constant cannot be caught by a value.
    """
    import inspect

    from mbfps.eval import headroom as module

    source = inspect.getsource(module)
    body = source.split("DISPLACEMENT_RECORDED_ONLY", 1)[1]
    assert "DISPLACEMENT_RECORDED_ONLY" not in body.split('"""', 2)[-1], (
        "the displacement constant must not be referenced after its definition"
    )
    assert DISPLACEMENT_RECORDED_ONLY == 3.9694722203504225


def test_reported_h_and_the_decision_cell_are_the_spec_values():
    """Pinned literally. These select which numbers the milestone publishes."""
    assert REPORTED_H == (1, 2, 3, 5, 8, 10, 15, 20, 30, 45)
    assert (DECISION_K, DECISION_H) == (1, 1)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/eval/test_headroom.py -v
```

Expected: FAIL at collection, `ModuleNotFoundError: No module named 'mbfps.eval.headroom'`.

- [ ] **Step 3: Write the module**

Create `src/mbfps/eval/headroom.py`:

```python
"""M3n: is the one-step map closer to copying, or to perfect?

M3m established that the 45-step rollout's error is COMPOUNDING: `burden(1)`
is not resolvable from zero at 2 SE in 9 of 9 cells, while `compounding(45)`
is. It could not establish WHY the one-step map is cheap, and it said so: a
map that predicts motion well and a map that predicts almost no motion both
score a small `burden(1)`, because holding position is cheap relative to the
probe's readout error.

This module places the one-step map between those two ends.

WHY M3m's `motion_margin` COULD NOT DO IT. It subtracted the k=1 rung, a
PROBE-SPACE model error of 88-224 map units, from the median true one-step
displacement, a GROUND-TRUTH quantity of 3.97. Readout error dominated the
difference 22.2x-56.5x, the base control failed 9 of 9, and Reading H shipped
UNREADABLE. The error was attractive rather than careless: the statistic needs
a scale for "how much motion was there to predict," and the intuitive answer is
the true displacement. But the numerator lives in probe space, and the only
probe-space answer is what a PERFECT predictor would have won -- which is
measured, not constant, and varies 20.5x across the nine cells.

THE SAME MISTAKE IS AVAILABLE ONE LEVEL UP. Normalising this module's `skill`
by 3.9694722203504225 produces an apparent bimodal split across the nine cells
(three cells at 106-118%, five at 6.5-43.2%, and one at -1.8%
belonging to neither) that dissolves entirely under `headroom`
(-2.4% to 72.7%, unimodal). The pattern was the constant denominator.

THE THREE DIFFERENCES, all probe-space, all per (k, h):

    headroom(k, h) = hold_k(h) - floor(h)    what a PERFECT predictor wins
    skill(k, h)    = hold_k(h) - rung_k(h)   what the MODEL wins
    deficit(k, h)  = rung_k(h) - floor(h)    M3m's `burden(k, h)`, unchanged

`skill + deficit == headroom` exactly -- the hold term cancels. Measured on the
shipped M3m curves at the k=45 column, the max residual over nine cells and 45
steps is 1.421e-14.

THAT IDENTITY IS WHY THE READING CARRIES NO RATIO. The two-sided test against
the copying end and the perfect end is `skill > 0` and `deficit > 0`: two
differences, no denominator. A denominator would matter, because `headroom`
crosses zero inside the reported grid on a real cell -- `pixel_ae_seed1` reads
0.575649 at h=1 and -0.641206 at h=2, so the normalised share there prints
+339.5% at h=2 then -335.7% at h=3.

BOTH AXES ARE NAMED. `k` is the re-grounding period, `h` the horizon step.
Write `headroom(k, h)`, never `headroom(45)`.
"""

from __future__ import annotations

import numpy as np

from mbfps.eval.burden import checked_pair

REPORTED_H: tuple[int, ...] = (1, 2, 3, 5, 8, 10, 15, 20, 30, 45)
"""The horizon steps the record publishes. Inherited from M3m unchanged, so the
two milestones' tables are read on the same grid."""

DECISION_K: int = 1
DECISION_H: int = 1
"""The cell the verdict is read at. `DECISION_K` is a LABEL, not a choice: at
h=1 every k grounds at step 0 and imagines one prior step, so all three
differences are identical across k (`RegroundingSweep.k_invariance_at_h1`).
That makes the one-step verdict k-free by construction, which is a control
rather than an assumption."""

IDENTITY_TOLERANCE: float = 1e-9
"""`triple_residual` above this is a plumbing fault, not rounding. The observed
residual on the shipped curves is 1.421e-14, about five orders below."""

CONFIDENCE: float = 0.95
RESAMPLES: int = 2000
"""The verdict's episode-clustered bootstrap. 229 windows over 24 episodes on
every shipped cell; see `pooling.clustered_interval` for why the unit is the
episode and why the direction of that error matters here specifically."""

SECONDARY_SIGMAS: int = 2
"""Multiplier applied to `RegroundingSweep.standard_error` for the RECORDED secondary
figure, which decides nothing. `RegroundingSweep.standard_error` returns ONE standard error and
`scripts/diagnose_dynamics.py:1430` doubles it; mislabelling one as two in M3m
produced the opposite conclusion from the correct one, and the wrong one was
the more interesting-sounding."""

SEEDS_MINIMUM: int = 3
ARMS_REQUIRED: int = 2
"""An arm with fewer than `SEEDS_MINIMUM` seeds is refused by name rather than
tallied. `ARMS_REQUIRED` is satisfied automatically by the strict-majority bar
over nine cells -- no arm holds more than three -- and is stated anyway so the
protection M3m carried is visibly not dropped."""

DISPLACEMENT_RECORDED_ONLY: float = 3.9694722203504225
"""The median true one-step displacement on the shipped split. RECORDED AND
USED BY NO STATISTIC IN THIS MODULE.

It is the quantity M3m's design mistook for a probe-space scale, kept in the
record so a reader can see that it plays no part in the reading. Nothing below
this line references it."""


def headroom(hold: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """`hold - floor`: what a PERFECT predictor wins over copying, per step.

    The denominator M3m's design needed and did not have. Measured, not
    constant: 0.575649 to 11.824 at h=1 across the nine shipped cells.

    Also the readability gate. If this is not resolvably positive, a perfect
    predictor cannot be told from a copying one, and nothing between them can
    be placed either.
    """
    hold, floor = checked_pair(hold, floor)
    return hold - floor


def skill(hold: np.ndarray, rung: np.ndarray) -> np.ndarray:
    """`hold - rung`: what the MODEL wins over copying, per step.

    Positive means the rung beats holding the floor's own position from the
    rung's last re-grounding step. Both sides pass through the same encoder,
    RSSM and embedding head and are read by the same probe, so the readout
    error they share cancels in this difference -- which is the entire reason
    this statistic is readable where `motion_margin` was not.
    """
    hold, rung = checked_pair(hold, rung)
    return hold - rung


def deficit(rung: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """`rung - floor`: how far the rung sits above the floor, per step.

    IDENTICAL to M3m's `burden(k, h)`, which is kept and unchanged. Named again
    here because in M3n's framing it is the distance to the PERFECT end of the
    axis, and the two-sided test reads it as such: `deficit` not resolvable
    from zero means the rung is statistically indistinguishable from a perfect
    one-step predictor.
    """
    rung, floor = checked_pair(rung, floor)
    return rung - floor


def triple_residual(
    hold: np.ndarray, rung: np.ndarray, floor: np.ndarray
) -> float:
    """`max |headroom - (skill + deficit)|` over the horizon.

    `max`, NOT `sum` or `mean`: one bad step is a plumbing fault and summing
    would let 45 tiny roundings hide it while averaging would divide it away.
    A test asserting this with `pytest.approx(rel=1e-6)` would accept the
    `np.sum` mutant. Measured on M3n's own k=45 column, 8 of 9 cells give exactly 0.0
    under both reductions and the ninth separates them by 4x (1.421e-14 against
    5.684e-14), so real data barely separates them: the FIXTURE has to.
    """
    whole = headroom(hold, floor)
    parts = skill(hold, rung) + deficit(rung, floor)
    return float(np.max(np.abs(whole - parts)))
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/eval/test_headroom.py -v
```

Expected: PASS, 6 tests.

- [ ] **Step 5: Verify the tests bite**

Apply each mutation, clear caches, run, confirm the named test fails, restore:

| mutation | test that must fail |
| --- | --- |
| `triple_residual`: `np.max` -> `np.sum` | `test_triple_residual_reports_the_worst_step_not_their_sum` |
| `triple_residual`: `np.max` -> `np.mean` | `test_triple_residual_reports_the_worst_step_not_their_sum` |
| `skill`: `hold - rung` -> `rung - hold` | `test_skill_is_positive_when_the_rung_beats_the_hold_baseline` |
| `headroom`: drop the `checked_pair` call | `test_mismatched_curve_lengths_are_refused` |

Record each observed failure message in the report.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/headroom.py tests/eval/test_headroom.py
git commit -F - <<'EOF'
feat: the three probe-space differences and the identity that binds them

headroom = hold - floor, skill = hold - rung, deficit = rung - floor, with
skill + deficit == headroom exactly because the hold term cancels. deficit
IS M3m's burden, unchanged; it is named again because in M3n's framing it is
the distance to the perfect end of the axis.

That identity is why the reading carries no ratio. The two-sided test
against copying and perfect is skill > 0 and deficit > 0, two differences,
no denominator -- which matters because headroom crosses zero inside the
reported grid on pixel_ae_seed1 (0.575649 at h=1, -0.641206 at h=2), where
the normalised share prints +339.5% at h=2 then -335.7% at h=3.

triple_residual takes the max, not the sum: the gap between them on real
data barely separates the two -- 8 of 9 cells give exactly 0.0 under both and
the ninth by 4x -- so approx(rel=1e-6) would accept the np.sum mutant and the
fixture, not the data, has to separate them. The test asserts exact equality.

3.9694722203504225 is defined and referenced nowhere, with a textual guard,
because an unused constant cannot be caught by a value.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

## Task 5: `headroom.py` — the readability gate, the four-way placement, and the formatter

**Files:**
- Modify: `src/mbfps/eval/headroom.py` (append)
- Test: `tests/eval/test_headroom.py` (append)

**Interfaces:**
- Consumes: `burden.strict_majority(n)`, the constants from Task 4
- Produces:
  ```python
  PLACEMENTS: tuple[str, ...] = (
      "BETWEEN", "AT_PERFECT", "AT_COPYING", "AMBIGUOUS", "UNREADABLE",
  )

  @dataclass(frozen=True)
  class Interval:
      point: float
      ci_low: float
      ci_high: float
      def resolvably_positive(self) -> bool    # ci_low > 0.0

  @dataclass(frozen=True)
  class HeadroomCell:
      arm: str
      seed: int
      headroom: Interval
      skill: Interval
      deficit: Interval
      clusters: int
      def placement(self) -> str

  @dataclass(frozen=True)
  class HeadroomInputs:
      cells: dict[tuple[str, int], HeadroomCell]
      decision_k: int
      decision_h: int
      ks: tuple[int, ...]

  @dataclass(frozen=True)
  class HeadroomStatus:
      verdict: str          # a member of PLACEMENTS, or "NO_MAJORITY"
      rule: str
      tally: dict[str, tuple[tuple[str, int], ...]]

  def reading_headroom(inputs: HeadroomInputs) -> HeadroomStatus
  READING_COLUMNS: tuple[str, ...]
  READING_WIDTHS: tuple[int, ...]
  def format_reading_headroom(reading: HeadroomStatus, inputs: HeadroomInputs) -> str
  ```

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_headroom.py`:

```python
def _cell(arm, seed, hd, sk, df, clusters=24):
    return HeadroomCell(
        arm=arm, seed=seed,
        headroom=Interval(*hd), skill=Interval(*sk), deficit=Interval(*df),
        clusters=clusters,
    )


def _inputs(cells):
    return HeadroomInputs(
        cells={(c.arm, c.seed): c for c in cells},
        decision_k=DECISION_K, decision_h=DECISION_H, ks=(1, 3, 5, 15, 45),
    )


def test_the_gate_reads_unreadable_before_looking_at_skill_or_deficit():
    """A cell whose headroom straddles zero is UNREADABLE whatever skill says.

    The gate must run FIRST. This cell's skill is resolvably positive and its
    deficit is not, so without the gate it would read AT_PERFECT -- a verdict
    about a model, taken on a cell where a perfect predictor is
    indistinguishable from a copying one.
    """
    cell = _cell("a", 0, hd=(0.4, -0.9, 1.7), sk=(0.3, 0.1, 0.5), df=(0.1, -0.2, 0.4))
    assert cell.skill.resolvably_positive()
    assert not cell.deficit.resolvably_positive()
    assert cell.placement() == "UNREADABLE"


def test_the_four_placements_are_each_reachable_and_distinct():
    """Each row sets only what it needs to, and the four verdicts differ.

    Built as a table so no two rows share a discriminating field by accident;
    M3m shipped a precedence fixture in which nothing else was wrong, so the
    test could not have distinguished the branches.
    """
    table = {
        "BETWEEN":    (( 5.0,  3.0,  7.0), (2.0,  1.0, 3.0), (3.0,  1.5, 4.5)),
        "AT_PERFECT": (( 5.0,  3.0,  7.0), (2.0,  1.0, 3.0), (0.1, -0.9, 1.1)),
        "AT_COPYING": (( 5.0,  3.0,  7.0), (0.1, -0.9, 1.1), (3.0,  1.5, 4.5)),
        "AMBIGUOUS":  (( 5.0,  3.0,  7.0), (0.1, -0.9, 1.1), (0.1, -0.9, 1.1)),
    }
    seen = {}
    for expected, (hd, sk, df) in table.items():
        seen[expected] = _cell("a", 0, hd, sk, df).placement()
    assert seen == {k: k for k in table}
    assert len(set(seen.values())) == 4


def test_a_strict_majority_is_computed_from_the_cells_present():
    """Five of nine carries; four of nine does not. The bar is never stored.

    `strict_majority` returns 5 at nine cells and 6 at eleven, which is why
    storing 5 is a defect -- M3l's design was reworked for exactly this.
    """
    between = ((5.0, 3.0, 7.0), (2.0, 1.0, 3.0), (3.0, 1.5, 4.5))
    other = ((5.0, 3.0, 7.0), (0.1, -0.9, 1.1), (0.1, -0.9, 1.1))
    arms = ("frozen_ssl", "pixel_ae", "random_vit")

    cells, i = [], 0
    for arm in arms:
        for seed in (0, 1, 2):
            cells.append(_cell(arm, seed, *(between if i < 5 else other)))
            i += 1
    assert len(cells) == 9
    carried = reading_headroom(_inputs(cells))
    assert carried.verdict == "BETWEEN"
    assert len(carried.tally["BETWEEN"]) == 5

    cells, i = [], 0
    for arm in arms:
        for seed in (0, 1, 2):
            cells.append(_cell(arm, seed, *(between if i < 4 else other)))
            i += 1
    assert reading_headroom(_inputs(cells)).verdict == "NO_MAJORITY"


def test_an_arm_short_of_seeds_minimum_is_refused_by_name():
    """`strict_majority(1) == 1`, so one lucky cell must not establish an arm.

    This is the M3j trap: `--arms random_vit` printed a row reading
    `clears = up` beside a verdict of NO DIFFERENCE.
    """
    cells = [_cell("frozen_ssl", s, (5.0, 3.0, 7.0), (2.0, 1.0, 3.0), (3.0, 1.5, 4.5))
             for s in (0, 1, 2)]
    cells.append(_cell("pixel_ae", 0, (5.0, 3.0, 7.0), (2.0, 1.0, 3.0), (3.0, 1.5, 4.5)))
    with pytest.raises(ValueError, match="SEEDS_MINIMUM"):
        reading_headroom(_inputs(cells))


def test_the_formatter_prints_one_row_per_cell_and_names_the_decision_cell():
    """Width, row count, and the (k, h) label the table is read at.

    `str(DECISION_H) in text` would be true of any table containing "1", so
    the assertion is on the formatted label.
    """
    cells = [_cell(a, s, (5.0, 3.0, 7.0), (2.0, 1.0, 3.0), (3.0, 1.5, 4.5))
             for a in ("frozen_ssl", "pixel_ae", "random_vit") for s in (0, 1, 2)]
    inputs = _inputs(cells)
    text = format_reading_headroom(reading_headroom(inputs), inputs)
    lines = text.splitlines()
    assert f"k = {DECISION_K}, h = {DECISION_H}" in text
    body = [l for l in lines if l.startswith(("frozen_ssl", "pixel_ae", "random_vit"))]
    assert len(body) == 9
    header = next(l for l in lines if l.startswith(READING_COLUMNS[0]))
    assert len(header) == sum(READING_WIDTHS)
    assert all(len(l) == len(header) for l in body)
    assert text.endswith("\n")


def test_the_formatter_conditions_every_directional_sentence_on_the_gate():
    """No unconditional claim that a sign means a verdict.

    M3m shipped "both directions are sound" in three places the results
    refuted -- a docstring, the text written into burden.txt, and spec 3.3.
    The legend here must say what the gate makes the sign mean, so the
    sentence cannot be inherited as unconditional.
    """
    cells = [_cell(a, s, (0.4, -0.9, 1.7), (2.0, 1.0, 3.0), (3.0, 1.5, 4.5))
             for a in ("frozen_ssl", "pixel_ae", "random_vit") for s in (0, 1, 2)]
    inputs = _inputs(cells)
    text = format_reading_headroom(reading_headroom(inputs), inputs)
    assert "UNREADABLE" in text
    assert "headroom" in text.lower()
    assert "resolvably" in text.lower()
```

Extend the module's import list in the test file with `HeadroomCell`, `HeadroomInputs`, `Interval`, `PLACEMENTS`, `READING_COLUMNS`, `READING_WIDTHS`, `format_reading_headroom`, `reading_headroom`.

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/eval/test_headroom.py -v
```

Expected: FAIL at collection, `ImportError: cannot import name 'Interval'`.

- [ ] **Step 3: Write the reading**

Append to `src/mbfps/eval/headroom.py`:

```python
PLACEMENTS: tuple[str, ...] = (
    "BETWEEN", "AT_PERFECT", "AT_COPYING", "AMBIGUOUS", "UNREADABLE",
)
"""Where one cell's one-step map sits on the copying-to-perfect axis.

`UNREADABLE` is about the INSTRUMENT, the other four about the model. It is in
the same tuple because it is a per-cell outcome and a majority of it is a
verdict -- M3n's exit 47 -- rather than a missing value."""

NO_MAJORITY: str = "NO_MAJORITY"
"""No placement held a strict majority of the cells. M3n's exit 48."""


@dataclass(frozen=True)
class Interval:
    """A point estimate and an episode-clustered bootstrap interval."""

    point: float
    ci_low: float
    ci_high: float

    def resolvably_positive(self) -> bool:
        """The whole interval lies above zero.

        `ci_low > 0.0`, strictly. An interval whose lower bound is exactly 0.0
        does not exclude zero, and the whole two-sided reading is a question
        about exclusion.
        """
        return self.ci_low > 0.0


@dataclass(frozen=True)
class HeadroomCell:
    """One cell -- one arm at one seed -- at ONE (k, h)."""

    arm: str
    seed: int
    headroom: Interval
    skill: Interval
    deficit: Interval
    clusters: int
    """Distinct episodes behind the bootstrap. Carried because an interval
    drawn from one cluster is not an interval, and the script refuses on it
    rather than printing it."""

    def placement(self) -> str:
        """This cell's member of `PLACEMENTS`.

        THE GATE RUNS FIRST, and the order is load-bearing rather than tidy. A
        cell whose `headroom` straddles zero can still have a resolvably
        positive `skill` and an unresolvable `deficit`, which without the gate
        reads `AT_PERFECT` -- a verdict about a model, taken on a cell where a
        perfect predictor is indistinguishable from a copying one. One of the
        nine shipped cells looks likely to be in that regime, though the
        shipped records cannot settle it: `pixel_ae_seed1`'s headroom POINT
        ESTIMATE at h=1 is 0.575649 map units against a readout error of
        259.06, and at h=2 it is NEGATIVE at -0.641206. The h=2 sign is
        decisive on its own; whether the h=1 interval straddles zero is one of
        the things this milestone measures.

        `AMBIGUOUS` is reachable WITH a resolvable headroom. The gate
        establishes only that the two ends are separated, not that the ruler is
        fine enough to locate a point between them.
        """
        if not self.headroom.resolvably_positive():
            return "UNREADABLE"
        has_skill = self.skill.resolvably_positive()
        has_deficit = self.deficit.resolvably_positive()
        if has_skill and has_deficit:
            return "BETWEEN"
        if has_skill:
            return "AT_PERFECT"
        if has_deficit:
            return "AT_COPYING"
        return "AMBIGUOUS"


@dataclass(frozen=True)
class HeadroomInputs:
    """The pooled cells, with the (k, h) they were all read at."""

    cells: dict[tuple[str, int], HeadroomCell]
    decision_k: int
    decision_h: int
    ks: tuple[int, ...]


@dataclass(frozen=True)
class HeadroomStatus:
    """The verdict, the rule it was reached by, and the full tally."""

    verdict: str
    rule: str
    tally: dict[str, tuple[tuple[str, int], ...]]
    """Placement -> the (arm, seed) cells voting for it. Every placement
    present as a key, including those with no votes, so a reader can tell "no
    cell read AT_COPYING" from "AT_COPYING was not considered"."""


def reading_headroom(inputs: HeadroomInputs) -> HeadroomStatus:
    """Place the one-step map, by strict majority of the cells.

    REFUSES rather than falls through on an arm short of `SEEDS_MINIMUM`.
    `strict_majority(1) == 1`, so without that refusal a single lucky cell
    would establish an arm -- the M3j trap, where `--arms random_vit` printed a
    row reading `clears = up` beside a verdict of NO DIFFERENCE.

    THE BAR IS COMPUTED. `strict_majority(len(cells))` returns 5 at nine cells
    and 6 at eleven, so storing 5 would be a majority at one cell count and a
    minority at another; M3l's design was reworked for exactly that.

    `ARMS_REQUIRED` IS ENFORCED, by two checks rather than none. An earlier
    draft of this plan argued it needed no check, because at nine cells in
    three arms no arm holds more than three and so any five cells span at
    least two arms. That is true of the 3x3 shape and false of this function,
    which accepts any shape the CLI's free-form `--arms` and `--seeds` can
    build: one arm with three seeds returns a verdict from one arm, and arms
    of 3, 3 and 9 let the nine-seed arm carry the majority alone. The M3j trap
    was exactly `--arms random_vit`.

    So: refuse when fewer than `ARMS_REQUIRED` arms are present, and refuse a
    verdict whose winning cells do not span `ARMS_REQUIRED` arms. The first
    catches the degenerate plan; the second catches the unbalanced one, which
    the first does not.
    """
    if not inputs.cells:
        raise ValueError("reading_headroom needs at least one cell")
    by_arm: dict[str, set[int]] = {}
    for arm, seed in inputs.cells:
        by_arm.setdefault(arm, set()).add(seed)
    short = {arm: sorted(s) for arm, s in by_arm.items() if len(s) < SEEDS_MINIMUM}
    if short:
        raise ValueError(
            f"every arm needs SEEDS_MINIMUM={SEEDS_MINIMUM} seeds before a "
            f"placement is tallied; got {short}"
        )

    tally: dict[str, list[tuple[str, int]]] = {name: [] for name in PLACEMENTS}
    for key, cell in sorted(inputs.cells.items()):
        tally[cell.placement()].append(key)
    frozen = {name: tuple(votes) for name, votes in tally.items()}

    needed = strict_majority(len(inputs.cells))
    winners = [name for name in PLACEMENTS if len(frozen[name]) >= needed]
    at = f"k = {inputs.decision_k}, h = {inputs.decision_h}"
    if not winners:
        counts = ", ".join(f"{n}={len(frozen[n])}" for n in PLACEMENTS)
        return HeadroomStatus(
            verdict=NO_MAJORITY,
            rule=(
                f"no placement reached {needed} of {len(inputs.cells)} cells at "
                f"{at} ({counts})"
            ),
            tally=frozen,
        )
    verdict = winners[0]
    return HeadroomStatus(
        verdict=verdict,
        rule=(
            f"{verdict} in {len(frozen[verdict])} of {len(inputs.cells)} cells at "
            f"{at} (strict majority {needed}, computed from the cells present)"
        ),
        tally=frozen,
    )
```

Add `from dataclasses import dataclass` and `from mbfps.eval.burden import checked_pair, strict_majority` to the module's imports.

Note: `winners` can hold at most one name, because `PLACEMENTS` partitions the
cells and two disjoint sets cannot each exceed half. Do not add a tie branch;
add instead, immediately after the `winners` assignment:

```python
    assert len(winners) <= 1, (
        f"two placements cannot both hold a strict majority of {len(inputs.cells)} "
        f"cells; got {winners} -- the partition is broken"
    )
```

- [ ] **Step 4: Write the formatter**

Append to `src/mbfps/eval/headroom.py`:

```python
READING_COLUMNS: tuple[str, ...] = (
    "cell", "clusters", "headroom", "skill", "deficit", "share", "placement",
)
READING_WIDTHS: tuple[int, ...] = (22, 10, 24, 24, 24, 9, 12)


def _row(values, widths) -> str:
    return "".join(str(v).ljust(w) for v, w in zip(values, widths))


def _interval(i: Interval) -> str:
    return f"{i.point:+8.3f} [{i.ci_low:+8.3f},{i.ci_high:+8.3f}]"


def format_reading_headroom(
    reading: HeadroomStatus, inputs: HeadroomInputs
) -> str:
    """The table, the verdict, and a legend that is conditioned on the gate.

    EVERY DIRECTIONAL SENTENCE IS CONDITIONAL. M3m shipped "both directions are
    sound" unconditionally in three places its own results refuted -- a
    docstring, the legend written into `burden.txt`, and spec 3.3, which the
    next milestone would have inherited. A sign here means something only
    behind the headroom gate, and the legend says so in the same breath as the
    sign, so the sentence cannot be quoted without its condition.

    The `share` column is a POINT ESTIMATE WITH NO INTERVAL, printed only where
    the gate passed. It needs none: the verdict is two-sided on the two
    differences, so the share is presentation. Where the gate failed it prints
    `--`, never a number, because `headroom` crosses zero inside the reported
    grid on a real cell: `pixel_ae_seed1`'s headroom is 0.575649 at h=1 and
    -0.641206 at h=2, and the share reads +339.5% at h=2 then -335.7% at h=3.
    """
    at = f"k = {inputs.decision_k}, h = {inputs.decision_h}"
    lines = [
        f"M3n motion headroom -- the one-step map between copying and perfect, at {at}",
        "",
        _row(READING_COLUMNS, READING_WIDTHS),
    ]
    for (arm, seed), cell in sorted(inputs.cells.items()):
        placement = cell.placement()
        share = (
            f"{cell.skill.point / cell.headroom.point * 100:7.1f}%"
            if placement != "UNREADABLE" else "--"
        )
        lines.append(_row(
            (
                f"{arm}_seed{seed}", cell.clusters,
                _interval(cell.headroom), _interval(cell.skill),
                _interval(cell.deficit), share, placement,
            ),
            READING_WIDTHS,
        ))
    lines += [
        "",
        f"verdict: {reading.verdict}",
        f"rule:    {reading.rule}",
        "",
        "legend",
        "  headroom = hold - floor, what a PERFECT predictor wins over copying.",
        "  skill    = hold - rung,  what the model wins over copying.",
        "  deficit  = rung - floor, M3m's burden -- the distance to perfect.",
        "  skill + deficit == headroom exactly; the hold term cancels.",
        "  A cell is read ONLY IF headroom is resolvably above zero. Where it is",
        "  not, a perfect predictor is indistinguishable from a copying one, so",
        "  the sign of skill carries no claim about the model and the share is",
        "  printed as `--` rather than as a number.",
        "  Behind that gate: skill resolvably positive means the one-step map",
        "  beats holding; deficit resolvably positive means it is not yet",
        "  indistinguishable from perfect.",
        f"  Intervals are episode-clustered bootstraps at {CONFIDENCE:.2f} over "
        f"{RESAMPLES} resamples.",
        "",
    ]
    return "\n".join(lines)
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/eval/test_headroom.py -v
```

Expected: PASS, 12 tests.

- [ ] **Step 6: Verify the tests bite**

| mutation | test that must fail |
| --- | --- |
| `placement`: move the gate below the skill/deficit branches | `test_the_gate_reads_unreadable_before_looking_at_skill_or_deficit` |
| `placement`: swap the `AT_PERFECT` and `AT_COPYING` returns | `test_the_four_placements_are_each_reachable_and_distinct` |
| `resolvably_positive`: `ci_low > 0.0` -> `ci_low >= 0.0` | `test_the_four_placements_are_each_reachable_and_distinct` |
| `reading_headroom`: `strict_majority(len(...))` -> `5` | neither majority test (both have nine cells) — **so add a third case with eleven cells** before claiming this mutation is caught, or state in the report that it is not |
| `reading_headroom`: delete the `SEEDS_MINIMUM` refusal | `test_an_arm_short_of_seeds_minimum_is_refused_by_name` |

The fourth row is a real gap at nine cells. Add to `test_a_strict_majority_is_computed_from_the_cells_present` an eleven-cell case where six carry and five do not, so the stored-constant mutation fails by value. Do not report the mutation as caught without it.

- [ ] **Step 7: Commit**

```bash
git add src/mbfps/eval/headroom.py tests/eval/test_headroom.py
git commit -F - <<'EOF'
feat: the readability gate, the four-way placement, and the formatter

The gate runs first, and the order is load-bearing rather than tidy: a cell
whose headroom straddles zero can have a resolvably positive skill and an
unresolvable deficit, which without the gate reads AT_PERFECT -- a verdict
about a model, on a cell where a perfect predictor is indistinguishable from
a copying one. pixel_ae_seed1 is in that regime at h=1.

AMBIGUOUS is reachable with a resolvable headroom. The gate establishes that
the two ends are separated, not that the ruler can locate a point between
them.

The majority bar is computed, not stored: strict_majority returns 5 at nine
cells and 6 at eleven. The test carries an eleven-cell case so the
stored-constant mutation fails by value rather than passing unnoticed at
nine. ARMS_REQUIRED needs no separate check -- no arm holds more than three
of nine cells, so any five span two arms -- and the docstring says why there
is no `if` against it.

Every directional sentence in the legend is conditioned on the gate in the
same breath as the sign, so it cannot be quoted without its condition. M3m
shipped the unconditional form in three places its own results refuted.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

## Task 6: `scripts/motion_headroom.py`

**Files:**
- Create: `scripts/motion_headroom.py`
- Test: `tests/eval/test_motion_headroom_script.py`
- Modify: `tests/eval/test_diagnose_dynamics_script.py` (exit registry)

**Interfaces:**
- Consumes: everything from Tasks 1–5; `diagnostics.regrounding_sweep`, `diagnostics.REGROUNDING_KS`; `pooling.clustered_interval`
- Produces: a CLI with `--phase {all,measure,read}`, `--out`, `--arms`, `--seeds`, `--ks`, `--bootstrap-seed`; `EXIT_UNREADABLE_HEADROOM = 47`, `EXIT_NO_MAJORITY = 48`; `headroom_record_path(out, arm, seed) -> Path`

**Record schema** (one JSON per cell, every constant stored so none is a literal that survives the tests):

```python
{
  "arm": str, "seed": int, "step": int,
  "git_sha": str, "record_git_sha": str, "torch_version": str, "device": str,
  "context": 5, "horizon": 45,
  "ks": [1, 3, 5, 15, 45], "reported_h": [1, 2, 3, 5, 8, 10, 15, 20, 30, 45],
  "decision_k": 1, "decision_h": 1,
  "confidence": 0.95, "resamples": 2000,
  "identity_tolerance": 1e-09, "secondary_sigmas": 2,
  "bootstrap_seed": int, "split_seed": int,
  "displacement_median": 3.9694722203504225,
  "episodes": {"val": [...]},
  "windows": {"total": 229, "episode": [...]},
  "curves": {
    "floor_position": [...], "persistence_position": [...],
    "rssm_position": [...],
    "rungs": {"1": [...], ...}, "holds": {"1": [...], ...},
  },
  "intervals": {"1": {"1": {"headroom": {"point": f, "ci_low": f, "ci_high": f},
                            "skill": {...}, "deficit": {...}}, ...}, ...},
  "share": {"1": {"1": float_or_null, ...}, ...},
  "secondary": {"iid_2se": {"1": {"headroom": [...], "skill": [...],
                                  "deficit": [...]}, ...}},
  "controls": {
    "persistence_divergence": 0.0, "k_invariance_at_h1": 0.0,
    "open_loop_divergence": 0.0, "floor_divergence": 0.0,
    "identity_residual": 0.0, "negative_headroom_steps": [[k, h], ...],
  },
  "nonfinite": 0, "kl_dyn_max": float, "kl_rate_above_free_bits": float,
}
```

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_motion_headroom_script.py`. Mirror the structure of `tests/eval/test_prediction_burden_script.py` — read it first for the fake-checkpoint and fake-model helpers, and reuse them.

```python
def test_the_record_path_separates_seeds(tmp_path):
    """Three seeds of one arm must not collide on one filename.

    M3m's fixtures all used seed 0, which hid exactly this: in the real run
    `burden_record_path(out, arm, 0)` would have written all three seeds of an
    arm to one file, destroying 6 of 9 records and surfacing only at the read
    phase after the GPU time had been spent.
    """
    paths = {
        headroom_record_path(tmp_path, "pixel_ae", seed) for seed in (0, 1, 2)
    }
    assert len(paths) == 3
    assert all("pixel_ae" in p.name for p in paths)


def test_measure_propagates_a_failed_status_as_a_nonzero_exit(tmp_path, monkeypatch):
    """`--phase measure` must not discard a failure and exit 0.

    M3m found this path could return 0 after a failed measure, which is the
    worst available outcome of a run that costs GPU time: the operator reads
    success and the records are not there.
    """
    monkeypatch.setattr(
        "scripts.motion_headroom.measure_cell",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(SystemExit) as exit_info:
        main(["--phase", "measure", "--out", str(tmp_path)])
    assert exit_info.value.code != 0


def test_read_returns_47_when_a_majority_of_cells_is_unreadable(tmp_path):
    """Exit 47 is the instrument's verdict, not the model's."""
    _write_nine_records(tmp_path, headroom=(0.4, -0.9, 1.7),
                        skill=(2.0, 1.0, 3.0), deficit=(3.0, 1.5, 4.5))
    with pytest.raises(SystemExit) as exit_info:
        main(["--phase", "read", "--out", str(tmp_path)])
    assert exit_info.value.code == EXIT_UNREADABLE_HEADROOM


def test_read_returns_48_when_no_placement_holds_a_majority(tmp_path):
    """Exit 48 with a four-four-one split across nine cells."""
    _write_split_records(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        main(["--phase", "read", "--out", str(tmp_path)])
    assert exit_info.value.code == EXIT_NO_MAJORITY


def test_every_recorded_constant_moves_when_the_module_constant_moves(tmp_path, monkeypatch):
    """Five record constants were literals surviving all 50 tests in M3m.

    Each is monkeypatched to a distinguishable value and the written record
    must carry the patched value, not the original.
    """
    for name, patched in (
        ("CONFIDENCE", 0.9), ("RESAMPLES", 7), ("DECISION_H", 2),
        ("DECISION_K", 3), ("IDENTITY_TOLERANCE", 1e-5),
        ("SECONDARY_SIGMAS", 3),
    ):
        monkeypatch.setattr(f"scripts.motion_headroom.{name}", patched)
    record = _measure_one_fake_cell(tmp_path)
    assert record["confidence"] == 0.9
    assert record["resamples"] == 7
    assert record["decision_h"] == 2 and record["decision_k"] == 3
    assert record["identity_tolerance"] == 1e-5
    assert record["secondary_sigmas"] == 3


def test_the_read_phase_is_byte_identical_across_two_reads(tmp_path):
    """The reading must not depend on dict ordering or a fresh RNG draw.

    Compared against text built from the SPEC path, not from a second call to
    the same code: two reads of the same code strip the same bytes, so a
    byte-identity test against itself cannot catch an `rstrip()`.
    """
    _write_nine_records(tmp_path, headroom=(5.0, 3.0, 7.0),
                        skill=(2.0, 1.0, 3.0), deficit=(3.0, 1.5, 4.5))
    first = _capture_read_stdout(tmp_path)
    second = _capture_read_stdout(tmp_path)
    assert first == second
    assert first == (tmp_path / "headroom.txt").read_text()


def test_a_record_whose_episode_labels_are_null_is_refused(tmp_path):
    """No fallback to `arange(n)`.

    A record with no clustering must refuse, not silently become the
    window-level bootstrap `clustered_interval`'s first paragraph rules out.
    """
    _write_nine_records(tmp_path, headroom=(5.0, 3.0, 7.0),
                        skill=(2.0, 1.0, 3.0), deficit=(3.0, 1.5, 4.5))
    path = headroom_record_path(tmp_path, "pixel_ae", 1)
    record = json.loads(path.read_text())
    record["windows"]["episode"] = None
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit) as exit_info:
        main(["--phase", "read", "--out", str(tmp_path)])
    assert exit_info.value.code != 0
```

`_write_nine_records`, `_write_split_records`, `_measure_one_fake_cell` and `_capture_read_stdout` are helpers this test module defines. `_write_nine_records` must write **three arms x three seeds** with distinct per-cell window counts and at least three distinct episode labels, and must assert its own discrimination before returning.

And in `tests/eval/test_diagnose_dynamics_script.py`, extend the exit registry assertions so the new script's codes are checked exactly as `prediction_burden`'s 45 and 46 are, with the same no-collision rule:

```python
def test_motion_headroom_exit_codes_are_new_and_do_not_collide():
    """47 and 48 are M3n's. 43 and 44 are latent_capacity's, 45 and 46
    prediction_burden's, and no script used a higher number before this one."""
    own = {
        value for key, value in vars(motion_headroom).items()
        if key.startswith("EXIT_") and isinstance(value, int)
    }
    assert {47, 48} <= own
    assert motion_headroom.EXIT_UNREADABLE_HEADROOM == 47
    assert motion_headroom.EXIT_NO_MAJORITY == 48
    assert own.isdisjoint({43, 44, 45, 46})
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/eval/test_motion_headroom_script.py -v
```

Expected: FAIL at collection, `ModuleNotFoundError: No module named 'scripts.motion_headroom'`.

- [ ] **Step 3: Write the script**

Create `scripts/motion_headroom.py` following `scripts/prediction_burden.py`'s structure exactly — same argument names, same phase split, same protocol-check shape, same `git_sha` / `record_git_sha` discipline. The pieces that are new:

```python
EXIT_UNREADABLE_HEADROOM: int = 47
"""RETURNED BY THE READ PHASE when a strict majority of cells read UNREADABLE.

The INSTRUMENT's verdict, not the model's: a perfect one-step predictor is
indistinguishable from a copying one in most cells, so no statement about the
model is available at this (k, h). It is a finding rather than a failure --
every candidate change to the objective would be graded by the same probe --
and it redirects the next milestone to the readout rather than to training.
"""

EXIT_NO_MAJORITY: int = 48
"""RETURNED BY THE READ PHASE when no placement holds a strict majority.

Distinct from 47: there, most cells agreed that nothing is readable; here the
cells disagree about where the map sits.
"""

READ_EXITS: dict[str, int] = {
    "UNREADABLE": EXIT_UNREADABLE_HEADROOM,
    NO_MAJORITY: EXIT_NO_MAJORITY,
}
"""Keyed on exactly the statuses that are not a placement of the model. The
four model placements -- BETWEEN, AT_PERFECT, AT_COPYING, AMBIGUOUS -- exit 0:
each is an answer, and AMBIGUOUS is an answer about the ruler at a readable
cell, which is a different claim from 47's about the instrument.
"""


def headroom_record_path(out: Path, arm: str, seed: int) -> Path:
    """One file per CELL. The seed is in the name, not only in the payload.

    M3m's fixtures all used seed 0, which hid the collision this prevents: in
    the real run, three seeds of an arm would have been written to one
    filename, destroying 6 of 9 records and surfacing only at the read phase
    after the GPU time had been spent.
    """
    return Path(out) / f"headroom_{arm}_seed{seed}.json"


def measure_cell(
    model, val_paths, probe_weights, *, arm, seed, ks, context, horizon,
    bootstrap_seed, device, feature_backbone, shipped,
) -> dict:
    """One cell: one sweep, then every interval and control from its rows.

    ONE TRAVERSAL. The sweep computes the rungs, the shared floor and the hold
    baseline on the same windows in the same order and now carries the episode
    labels too, so no alignment between two passes is needed and none is
    assumed. M3m had to walk the validation episodes a second time for the
    labels, with a documented numbering mismatch; M3n does not.

    `shipped` is the cell's M3c diagnostic record, loaded the way
    `scripts/prediction_burden.py`'s measure phase loads it -- read that
    function for the record path and the curve field names. It is what
    `open_loop_divergence` and `floor_divergence` are read against, and
    together those two are ALSO the `keep_trajectories=True` control: the
    shipped curves were measured with the flag OFF, so a bitwise match proves
    the flag moved neither the arms nor the reference. A second traversal with
    the flag off would double the run to establish what these two establish
    for free, and the unit test in Task 3 pins the same claim directly.
    """
    sweep = regrounding_sweep(
        model, val_paths, probe_weights, ks=ks, context=context,
        horizon=horizon, seed=seed, device=device,
        feature_backbone=feature_backbone,
    )
    groups = sweep.window_episode
    intervals: dict = {}
    share: dict = {}
    secondary: dict = {}
    negative: list = []
    residuals: list = []
    for k in ks:
        hold_rows = sweep.window_hold_position[k]
        rung_rows = sweep.window_position[k]
        floor_rows = sweep.window_floor_position
        rows = {
            "headroom": hold_rows - floor_rows,
            "skill": hold_rows - rung_rows,
            "deficit": rung_rows - floor_rows,
        }
        # The identity on the PER-WINDOW rows, which is strictly stronger than
        # on the mean curves: a compensating pair of errors that cancels in the
        # mean does not cancel window by window.
        residuals.append(float(np.max(np.abs(
            rows["headroom"] - (rows["skill"] + rows["deficit"])
        ))))
        secondary[k] = {
            name: (SECONDARY_SIGMAS * RegroundingSweep.standard_error(array)).tolist()
            for name, array in rows.items()
        }
        intervals[k] = {}
        share[k] = {}
        for h in REPORTED_H:
            cell: dict = {}
            for name, array in rows.items():
                point, low, high = clustered_interval(
                    array, groups, h=h,
                    resamples=RESAMPLES, seed=bootstrap_seed,
                )
                cell[name] = {"point": point, "ci_low": low, "ci_high": high}
            intervals[k][h] = cell
            hd = cell["headroom"]
            # None, never a number, where the gate failed. `headroom` crosses
            # zero inside the reported grid on a real cell -- pixel_ae_seed1
            # reads 0.575649 at h=1 and -0.641206 at h=2 -- and the ratio there
            # prints +339.5% at h=2 then -335.7% at h=3.
            share[k][h] = (
                cell["skill"]["point"] / hd["point"]
                if hd["ci_low"] > 0.0 else None
            )
            if hd["point"] < 0.0:
                negative.append([k, h])

    reference = shipped_reference(shipped)   # a RolloutResult, see below
    return {
        "arm": arm,
        "seed": seed,
        "ks": list(ks),
        "reported_h": list(REPORTED_H),
        "decision_k": DECISION_K,
        "decision_h": DECISION_H,
        "confidence": CONFIDENCE,
        "resamples": RESAMPLES,
        "identity_tolerance": IDENTITY_TOLERANCE,
        "secondary_sigmas": SECONDARY_SIGMAS,
        "bootstrap_seed": bootstrap_seed,
        "context": context,
        "horizon": horizon,
        "displacement_median": DISPLACEMENT_RECORDED_ONLY,
        "windows": {"total": sweep.windows_total, "episode": groups.tolist()},
        "curves": {
            "floor_position": sweep.reference.floor_position.tolist(),
            "persistence_position": sweep.reference.persistence_position.tolist(),
            "rssm_position": sweep.reference.rssm_position.tolist(),
            "rungs": {str(k): sweep.curve(k).tolist() for k in ks},
            "holds": {str(k): sweep.hold_position[k].tolist() for k in ks},
        },
        "intervals": {
            str(k): {str(h): v for h, v in by_h.items()}
            for k, by_h in intervals.items()
        },
        "share": {
            str(k): {str(h): v for h, v in by_h.items()}
            for k, by_h in share.items()
        },
        "secondary": {"iid_2se": {str(k): v for k, v in secondary.items()}},
        "controls": {
            "persistence_divergence": sweep.persistence_divergence(),
            "k_invariance_at_h1": sweep.k_invariance_at_h1(),
            "open_loop_divergence": sweep.open_loop_divergence(reference),
            "floor_divergence": float(np.max(np.abs(
                sweep.reference.floor_position - reference.floor_position
            ))),
            "identity_residual": max(residuals),
            "negative_headroom_steps": negative,
        },
    }


def headroom_inputs(records: dict) -> HeadroomInputs:
    """The pooled records as `HeadroomInputs`, every cell at ONE (k, h).

    REFUSES a null `windows.episode` rather than falling back to `arange(n)`.
    A record with no clustering must stop the read; treating every window as
    its own episode converts the verdict's ruler into the window-level
    bootstrap `pooling.clustered_interval`'s first paragraph rules out, and it
    would do so silently, printing a narrower interval and a more confident
    verdict.

    REFUSES records that disagree on the decision cell or the ladder. A table
    whose rows were read at different (k, h) is not a table.
    """
    cells: dict = {}
    agreed: set = set()
    for (arm, seed), record in sorted(records.items()):
        labels = record["windows"]["episode"]
        if labels is None:
            raise ValueError(
                f"{arm} seed {seed} carries no episode labels, so no "
                "episode-clustered interval can be drawn from it; refusing "
                "rather than clustering by window"
            )
        k, h = record["decision_k"], record["decision_h"]
        agreed.add((k, h, tuple(record["ks"])))
        at = record["intervals"][str(k)][str(h)]
        cells[(arm, seed)] = HeadroomCell(
            arm=arm,
            seed=seed,
            headroom=Interval(**at["headroom"]),
            skill=Interval(**at["skill"]),
            deficit=Interval(**at["deficit"]),
            clusters=len(set(labels)),
        )
    if len(agreed) != 1:
        raise ValueError(
            f"records disagree on the decision cell or the ladder: {sorted(agreed)}"
        )
    ((k, h, ks),) = agreed
    return HeadroomInputs(cells=cells, decision_k=k, decision_h=h, ks=ks)
```

`shipped_reference(shipped)` rebuilds a `RolloutResult` from the M3c record's
stored curves; `scripts/prediction_burden.py` already does this for its own
`open_loop_divergence` check — reuse that function rather than writing a second
one, importing it if it is public and lifting it into a shared place if it is
not.

The `--phase read` path: load every record, call `headroom_inputs` (which
performs both refusals), call `reading_headroom`, write
`format_reading_headroom`'s text to both stdout and `<out>/headroom.txt`, and
exit with `READ_EXITS.get(reading.verdict, 0)`.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/eval/test_motion_headroom_script.py tests/eval/test_diagnose_dynamics_script.py -v
```

Expected: PASS.

- [ ] **Step 5: Smoke the script end to end on a tiny protocol**

```bash
.venv/bin/python scripts/motion_headroom.py --phase all --out runs/m3n_smoke \
  --arms pixel_ae --seeds 0 --ks 1,45 --bootstrap-seed 0 2>&1 | tail -30
```

Expected: the measure phase writes one record, the read phase refuses on `SEEDS_MINIMUM` with a nonzero exit naming the arm and its seed count. That refusal **is** the expected smoke result; a zero exit here would mean the seeds gate is not wired.

- [ ] **Step 6: Commit**

```bash
git add scripts/motion_headroom.py tests/eval/test_motion_headroom_script.py tests/eval/test_diagnose_dynamics_script.py
git commit -F - <<'EOF'
feat: scripts/motion_headroom.py, the measure and read phases

One traversal per cell: the sweep computes the rungs, the shared floor and
the hold baseline on the same windows in the same order, so no alignment
between two passes is needed and none is assumed.

Exit 47 is the instrument's verdict -- a majority of cells where a perfect
one-step predictor cannot be told from a copying one -- and 48 is the cells
disagreeing about where the map sits. The four model placements exit 0,
AMBIGUOUS included: it is an answer about the ruler at a readable cell,
which is a different claim from 47's about the instrument.

The record path separates seeds. M3m's fixtures all used seed 0, which hid a
collision that would have written three seeds of an arm to one filename and
destroyed 6 of 9 records, surfacing only at the read phase after the GPU
time was spent. The measure phase propagates a failed status as a nonzero
exit for the same reason.

Every constant the record stores is monkeypatched in a test, because five
record constants were literals surviving all 50 of M3m's tests.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

## Task 7: Remove M3m's contaminated statistic

The replacement now demonstrably exists, which is the spec's own argument for removing rather than deprecating.

**Files:**
- Modify: `src/mbfps/eval/burden.py` — remove `motion_margin`, `DECISION_H`, `ARMS_REQUIRED`, `CONFIDENCE`, `RESAMPLES`, `BurdenArm`, `BurdenInputs`, `BurdenStatus`, `_fall_through_rule`, `reading_burden`, `READING_COLUMNS`, `READING_WIDTHS`, `_row`, `format_reading_burden`. **Keep** `at_horizon`, `checked_pair`, `burden`, `compounding`, `identity_residual`, `scored_targets`, `one_step_persistence`, `strict_majority`, `SEEDS_MINIMUM`, `REPORTED_H`, `IDENTITY_TOLERANCE`.
- Modify: `scripts/prediction_burden.py` — stop computing `margin` in the measure phase; the read phase prints the stored `margin` from existing records under a superseded legend and points at M3n.
- Modify: `tests/eval/test_burden.py`, `tests/eval/test_prediction_burden_script.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `burden.py` reduced to the ladder. `headroom.py` imports `checked_pair` and `strict_majority` from it, so those two must survive — verify by running `tests/eval/test_headroom.py` after the removal.

- [ ] **Step 1: Write the failing test**

Add to `tests/eval/test_burden.py`:

```python
def test_the_contaminated_statistic_is_gone_from_the_module():
    """`motion_margin` and Reading H are removed, not deprecated in place.

    A live function whose legend says a reading of -116 means COPIES is the
    same hazard as the "both directions are sound" sentence, which shipped in
    three places M3m's own results refuted. Conditioning was the right fix
    while the records had to stay readable; leaving the function live now that
    a correct replacement exists converts a corrected error into a standing
    trap.
    """
    from mbfps.eval import burden as module

    for gone in (
        "motion_margin", "margin_interval", "reading_burden",
        "format_reading_burden", "BurdenInputs", "BurdenStatus", "BurdenArm",
        "_fall_through_rule", "READING_COLUMNS", "READING_WIDTHS",
    ):
        assert not hasattr(module, gone), f"{gone} must be removed"


def test_the_ladder_survives_the_removal():
    """`burden`, `compounding` and `identity_residual` were right and stay.

    M3n's `deficit` IS `burden`, and `headroom.py` imports `checked_pair` and
    `strict_majority` from here, so a removal that took them would break the
    replacement.
    """
    from mbfps.eval import burden as module

    for kept in (
        "at_horizon", "checked_pair", "burden", "compounding",
        "identity_residual", "scored_targets", "one_step_persistence",
        "strict_majority", "SEEDS_MINIMUM", "REPORTED_H",
        "IDENTITY_TOLERANCE",
    ):
        assert hasattr(module, kept), f"{kept} must survive"


def test_the_measure_phase_no_longer_computes_the_contaminated_margin():
    """New records must not carry a `margin` field.

    Old records keep theirs and stay readable; the point is to stop producing
    new contaminated numbers.
    """
    import inspect

    from scripts import prediction_burden

    source = inspect.getsource(prediction_burden.measure_cell)
    assert "margin" not in source.replace("motion_margin_superseded", "")
```

And to `tests/eval/test_prediction_burden_script.py`:

```python
def test_the_read_phase_points_at_m3n_instead_of_returning_a_verdict():
    """Reading H is retired. The exit is nonzero and the text names M3n.

    The stored margin from M3m's records is still printed, under a legend
    saying it is superseded and WHY -- a probe-space model error subtracted
    from a ground-truth displacement, dominated 22.2x-56.5x -- so a reader of
    an old record is not left to rediscover it.
    """
    _write_m3m_records(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        main(["--phase", "read", "--out", str(tmp_path)])
    assert exit_info.value.code != 0
    text = (tmp_path / "burden.txt").read_text()
    assert "motion_headroom" in text
    assert "superseded" in text.lower()
    assert "22.2" in text and "56.5" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/eval/test_burden.py tests/eval/test_prediction_burden_script.py -k "contaminated or ladder_survives or points_at_m3n" -v
```

Expected: FAIL — `motion_margin must be removed`.

- [ ] **Step 3: Remove the code**

Delete the named definitions from `src/mbfps/eval/burden.py`. Update its module docstring: the ladder is what the module is now, and the docstring must say that the motion statistic moved to `headroom.py` and why, in one paragraph, so the removal is discoverable from the file that used to hold it.

- [ ] **Step 4: Repoint the script**

In `scripts/prediction_burden.py`: drop the removed imports, delete the `margin` computation from `measure_cell`, and replace the read phase's verdict with a refusal that prints the stored `margin` under the superseded legend and names `scripts/motion_headroom.py`. Update the module docstring and the `EXIT_CONTROL_BROKEN` / `EXIT_UNREADABLE` docstrings to record that 45 and 46 are now historical.

- [ ] **Step 5: Delete the orphaned tests**

Remove every test in `tests/eval/test_burden.py` and `tests/eval/test_prediction_burden_script.py` that exercises a removed definition. Do **not** remove tests of `burden`, `compounding`, `identity_residual`, `scored_targets`, `one_step_persistence` or `strict_majority`.

- [ ] **Step 6: Run the whole suite**

```bash
.venv/bin/python -m pytest -q 2>&1 | tail -20
```

Expected: PASS, 0 failures. Then confirm the untouched files:

```bash
git diff --stat origin/main -- src/mbfps/eval/rollout.py src/mbfps/eval/probe.py scripts/trust_horizon.py src/mbfps/eval/motion.py scripts/latent_motion.py
```

Expected: empty output.

- [ ] **Step 7: Commit**

```bash
git add src/mbfps/eval/burden.py scripts/prediction_burden.py tests/eval/test_burden.py tests/eval/test_prediction_burden_script.py
git commit -F - <<'EOF'
refactor: remove motion_margin and Reading H, now that the replacement exists

motion_margin subtracted a probe-space model error of 88-224 map units from
a ground-truth displacement of 3.97, so readout error dominated 22.2x-56.5x,
the base control failed 9 of 9, and Reading H shipped UNREADABLE. Removed
rather than deprecated: a live function whose legend says a reading of -116
means COPIES is the same hazard as the "both directions are sound" sentence,
which shipped in three places M3m's own results refuted. Conditioning was
right while the records had to stay readable; leaving the function live now
that headroom.py exists converts a corrected error into a standing trap.

The ladder stays untouched -- burden, compounding, identity_residual -- and
is what the module now is. M3n's deficit IS burden; headroom.py imports
checked_pair and strict_majority from here.

prediction_burden stops computing margin for new records. Old records keep
theirs and still print, under a legend that says it is superseded and why,
so a reader of runs/m3m_burden is not left to rediscover the contamination.
45 and 46 are now historical exit codes.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

## Task 8: Run the nine cells and write the results

**Files:**
- Create: `runs/m3n_motion/` (records; **never** `rm` anything under `runs/`)
- Modify: `docs/superpowers/plans/2026-10-04-mb-fps-m3n-motion-headroom.md` (append `## Task 8 results`)

**Interfaces:**
- Consumes: `scripts/motion_headroom.py` from Task 6
- Produces: nine records, `runs/m3n_motion/headroom.txt`, and a results section

- [ ] **Step 1: Confirm the suite is green at the commit being run**

```bash
.venv/bin/python -m pytest -q 2>&1 | tail -5
git rev-parse --short HEAD
```

Record both the counts and the SHA. The records store `record_git_sha`; a run whose suite was not green at that SHA is not reportable.

- [ ] **Step 2: Measure**

```bash
.venv/bin/python scripts/motion_headroom.py --phase measure --out runs/m3n_motion \
  --arms frozen_ssl,pixel_ae,random_vit --seeds 0,1,2 --ks 1,3,5,15,45 \
  --bootstrap-seed 0 2>&1 | tee runs/m3n_motion/measure.log
```

Expected: nine records at `runs/m3n_motion/headroom_<arm>_seed<n>.json`, exit 0. M3m's comparable nine-cell re-measure took a measured 19m20s. Check the file count before reading:

```bash
ls runs/m3n_motion/headroom_*.json | wc -l    # must print 9
```

- [ ] **Step 3: Check every control before reading any statistic**

```bash
.venv/bin/python - <<'EOF'
import glob, json
for p in sorted(glob.glob("runs/m3n_motion/headroom_*.json")):
    c = json.load(open(p))["controls"]
    print(p.split("/")[-1], {k: v for k, v in c.items() if k != "negative_headroom_steps"},
          "neg:", len(c["negative_headroom_steps"]))
EOF
```

Expected: `persistence_divergence`, `k_invariance_at_h1`, `open_loop_divergence` and `floor_divergence` all exactly `0.0`; `identity_residual` below `1e-9`. **A nonzero value in any of the first four means the measurement is wrong and no statistic in the record may be reported.** Record the actual values.

`open_loop_divergence` and `floor_divergence` at `0.0` are also the
`keep_trajectories=True` control: the shipped M3c curves were measured with the
flag off, so a bitwise match on both proves the flag moved neither the arms nor
the reference.

- [ ] **Step 4: Read**

```bash
.venv/bin/python scripts/motion_headroom.py --phase read --out runs/m3n_motion 2>&1 | tee runs/m3n_motion/read.log
echo "exit: $?"
```

Record the exit code and the full table.

- [ ] **Step 5: Confirm the read is reproducible**

```bash
SECOND=$(mktemp)
.venv/bin/python scripts/motion_headroom.py --phase read --out runs/m3n_motion > "$SECOND"
diff runs/m3n_motion/headroom.txt "$SECOND" && echo "byte-identical"
```

- [ ] **Step 6: Re-report M3m's resolvability under the clustered ruler**

Spec §13. `deficit` is `burden`, so the records already carry it:

```bash
.venv/bin/python - <<'EOF'
import glob, json
print(f"{'cell':22s} {'deficit(1,1)':>26s} {'resolvable?':>12s}")
for p in sorted(glob.glob("runs/m3n_motion/headroom_*.json")):
    d = json.load(open(p))["intervals"]["1"]["1"]["deficit"]
    print(f"{p.split('/')[-1][9:-5]:22s} "
          f"{d['point']:+8.3f} [{d['ci_low']:+8.3f},{d['ci_high']:+8.3f}] "
          f"{str(d['ci_low'] > 0.0):>12s}")
EOF
```

Compare against M3m's claim — "`burden(1)` is not resolvable from zero at 2 SE in 9 of 9 cells, max 1.94 SE" — and report whether the clustered ruler agrees. It should, and more strongly.

- [ ] **Step 7: Write the results section**

Append `## Task 8 results` to this plan file. Every number must be re-derivable from the records by the command that produced it, and each claim must cite the file it came from. Three figures in M3m's results were not re-derivable from its own records; the rule exists because of that. State explicitly:

- the verdict and exit code
- all six controls per cell
- the `(k, h)` surface at the decision cell, and the `h` at which `skill` stops being resolvable in each cell
- how many cells read each placement
- the §13 re-reporting of `burden`'s resolvability
- which of §11's four outcomes obtained, and what it licenses — in the spec's own words, not new ones
- the three things §12 says M3n does **not** answer, restated against what the run found, so the next milestone cannot inherit a claim the data does not carry
- §14's by-product: the k=45 column across `REPORTED_H`, which is the `gap_closed` profile the M3 gate has only ever read at `h=45`, and the `h` at which it crosses zero in each cell
- the wall clock, measured, not estimated

- [ ] **Step 8: Commit**

```bash
git add docs/superpowers/plans/2026-10-04-mb-fps-m3n-motion-headroom.md
git commit -F - <<'EOF'
results: where the one-step map sits between copying and perfect

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

Replace the subject with what the run actually found before committing. A subject asserting more than the data supports is the defect M3l shipped and had to amend: its commit subject claimed equality where only `<=` held, and the commit body forbade the sentence the subject made.
