# MB-FPS M3j — Motion Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure where on the `enc(t) -> z -> h` path the information about motion the model has already observed is lost, so the project can choose between two training levers that cost the same 13.5-hour retrain.

**Architecture:** Every measurement is an *increment over the current frame*: both arms of every comparison are handed `enc(t)`, so the 32x32 categorical bottleneck and the position-constrains-motion confound both cancel. `probe.filtering_gain` already computes that shape of statistic with the three-split discipline it needs; this plan generalises its array-level half (`_gain_from_splits`) to take an arbitrary second block and target, adds label-based resampling so filtered rows are legal, and builds Reading E on top. Evaluation only: no training, no checkpoint altered, nothing under `runs/` removed.

**Tech Stack:** Python 3.12, numpy, torch (MPS), pytest. No new dependencies.

## Global Constraints

Copied verbatim from `docs/superpowers/specs/2026-09-27-mb-fps-m3j-motion-retention-design.md`. Every task's requirements implicitly include this section.

- **Evaluation only.** No training, no checkpoint written or altered, nothing under `runs/` removed.
- **`filtering_gain`'s output stays byte-identical.** It is on the gate's reporting path via `src/mbfps/eval/study.py:589` and `scripts/eval_rollout.py:201`. A test pins it.
- **No M3b–M3i verdict or record is touched.** No change to the M3 gate, `aggregate.py`, or `report_study.py`.
- **Protocol:** context 5, horizon 45, 229 windows over 24 validation episodes, device `mps`, the nine M3c cells in `runs/m3_study_v2` at `record_git_sha` `ca3e140`.
- **Pre-registered constants:** `RETENTION_FAMILY = 3`, `SEEDS_REQUIRED = 2`, `ARMS_REQUIRED = 2`, `K_REPORTED = (1, 4, 15)`, `DECISION_K = 4`, `RUNGS = ("two_frame", "deterministic", "stochastic", "full")`.
- **Clearing is one-sided:** a rung clears when `ci_low > 0`. Never two-sided.
- **A cleared gain at ANY k in `K_REPORTED` counts as motion retained.** The disjunction is deliberate.
- **`two_frame`'s block is `enc(t-k)`**, the frame the displacement is measured from — never `enc(t-1)`.
- **The interval clusters on episodes**, not windows.
- **A non-finite gain or interval bound is a refusal, not a non-clear.**
- **Six statuses, this precedence:** `UNRESOLVED_BASE`, `MOTION_RETAINED`, `BOTTLENECK_LOSS`, `MOTION_DISCARDED`, `TRANSLATION_UNRESOLVED`, `UNRESOLVED_MOTION`.
- **Exit codes:** 39 `EXIT_BASE_UNRESOLVED`, 40 `EXIT_MOTION_UNRESOLVED`. M3i holds 38; 0/11/12/14/30 are inherited from `trust_horizon`.
- **`select_episodes` stays at 20** (`filtering_gain`'s default). Its docstring records that a 4-episode split flips the sign of the gain.
- **Test command:** `.venv/bin/python -m pytest` from the repo root. There is no `pytest` entry point in the venv; `python -m pytest` is the only form that works.
- **Repo conventions:** never `git stash` (shared stack across worktrees); never `rm` under `runs/`; commit trailer `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`; do not commit while a run is in progress.

## File Structure

| file | responsibility |
|---|---|
| `src/mbfps/eval/probe.py` | **modify.** Three additive changes: an `episode` row label from `gather_probe_data`; optional `groups` on `_block_bootstrap_ci`; a new general `gain_from_blocks` + `GainSplit`, with `_gain_from_splits` reduced to a wrapper. |
| `src/mbfps/eval/retention.py` | **create.** Pure: arrays in, readings out. The pre-registered constants, the row-shift helper, the two target builders, the rung blocks, Reading E and its printed table. No torch, no I/O, no record schema. |
| `scripts/latent_retention.py` | **create.** Owns torch, I/O and the record schema. Phases `measure | read | all`, one record per cell, `retention.txt`. |
| `src/mbfps/eval/motion.py` | **modify.** The finiteness guard on `clears_up` / `clears_down` / `leaks`. |
| `tests/eval/test_probe.py` | **modify.** The episode label, the groups-equivalence pin, the `filtering_gain` byte-identical pin. |
| `tests/eval/test_retention.py` | **create.** |
| `tests/eval/test_latent_retention_script.py` | **create.** |
| `tests/eval/test_motion.py` | **modify.** The guard fires only on non-finite values. |
| `tests/eval/test_diagnose_dynamics_script.py` | **modify.** `latent_retention` in the exit-status registry. |

`retention.py` holds the statistics and `latent_retention.py` holds everything impure, exactly as M3i split `motion.py` from `scripts/latent_motion.py`. Follow that split: if a function needs a `Path` or a `torch.device`, it belongs in the script.

---

### Task 1: `gather_probe_data` labels every row with its episode

**Why:** the interval must resample the 24 episodes, not the 229 windows (spec 3.1). The dict returns `window` and `step` (added by M3i) but no episode, and windows within an episode are correlated, so grouping by window gives intervals narrower than the project's own standard.

**Files:**
- Modify: `src/mbfps/eval/probe.py` — `gather_probe_data`, roughly lines 195–312
- Test: `tests/eval/test_probe.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `gather_probe_data(...)` returns a **seventh** row-aligned array under the key `"episode"`, `(N,)` `int64`, the 0-based index of the contributing episode each row came from. Tasks 2, 3, 7 rely on it.

- [ ] **Step 1: Write the failing tests**

Add both to `tests/eval/test_probe.py`, directly after the existing
`test_gather_probe_data_labels_every_row_with_its_window_and_step`.

```python
def test_gather_probe_data_labels_every_row_with_its_episode(tmp_path):
    """M3j resamples EPISODES, not windows: 229 non-overlapping windows cut from
    24 trajectories are not 229 independent observations. The label has to come
    from the gather, because by the time a caller holds the arrays the episode
    boundary is gone."""
    paths = _write_episodes(tmp_path, [20, 20, 20])
    data = _gather(paths, context=2, horizon=3)
    n = data["latent"].shape[0]
    episode, window = data["episode"], data["window"]

    assert episode.shape == (n,) and episode.dtype.kind == "i"
    # Three contributing episodes, labelled 0..2 with no gaps.
    assert np.unique(episode).tolist() == [0, 1, 2]
    # Every window sits inside exactly one episode -- so grouping by episode is
    # strictly COARSER than grouping by window, which is the whole point.
    for w in np.unique(window):
        assert len(set(episode[window == w].tolist())) == 1, f"window {w} spans episodes"
    # And every episode contributes more than one window, or the two groupings
    # would coincide and this label would buy nothing.
    assert all(
        len(np.unique(window[episode == e])) > 1 for e in np.unique(episode)
    ), "each episode must contribute several windows or the label is pointless"


def test_gather_probe_data_episode_labels_skip_episodes_that_contribute_nothing(tmp_path):
    """An episode too short for one window is `continue`d before it appends any
    row. If the counter advanced anyway the labels would carry gaps, and a
    caller sizing its bootstrap from `episode.max() + 1` would resample empty
    groups."""
    paths = _write_episodes(tmp_path, [20, 3, 20])
    data = _gather(paths, context=2, horizon=3)
    assert np.unique(data["episode"]).tolist() == [0, 1], (
        "the short episode must not consume a label"
    )
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -k "labels_every_row_with_its_episode or episode_labels_skip" -v`
Expected: FAIL, both with `KeyError: 'episode'`.

- [ ] **Step 3: Add the label**

In `src/mbfps/eval/probe.py`, in `gather_probe_data`, extend the accumulators:

```python
    latents, embeddings, encoder_embeddings, targets = [], [], [], []
    windows: list[np.ndarray] = []
    steps: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    window_index = 0
    episode_index = 0
```

Inside `for start in starts:`, beside the existing `windows.append(...)` /
`steps.append(...)` pair:

```python
            windows.append(np.full(rows, window_index, dtype=np.int64))
            steps.append(np.arange(rows, dtype=np.int64))
            episodes.append(np.full(rows, episode_index, dtype=np.int64))
            window_index += 1
```

And immediately after the `for start in starts:` loop closes — still inside
`for path in list(paths)[:limit]:` — advance the episode counter:

```python
        # Advanced only here, AFTER the window loop, so an episode that reached
        # `continue` above (too short for one window) never consumes a label and
        # the labels stay gap-free.
        episode_index += 1
```

Extend the return:

```python
        "window": np.concatenate(windows),
        "step": np.concatenate(steps),
        # Which EPISODE each row came from, 0-based over the episodes that
        # actually contributed a window. The coarsest correlated unit the scored
        # array contains: windows are cut non-overlapping, but several windows
        # from one trajectory are not independent observations, and every
        # reading from M3e onward clusters on episodes rather than windows.
        "episode": np.concatenate(episodes),
```

- [ ] **Step 4: Update the docstring and the key-count test**

In `gather_probe_data`'s docstring, change `Returns six row-aligned arrays:` to
`Returns seven row-aligned arrays:` and add, after the `"step"` entry:

```
    - `"episode"` `(N,)` int -- the 0-based index of the episode each row came
      from, over the episodes that contributed at least one window. M3j
      resamples on this rather than on `"window"`; see `_block_bootstrap_ci`'s
      `groups`.
```

In `tests/eval/test_probe.py`, rename
`test_gather_probe_data_is_deterministic_and_returns_exactly_six_keys` to
`..._exactly_seven_keys`, change its docstring's `the two row indices` to
`the three row indices` and `The six-key set` to `The seven-key set`, and
extend the asserted set:

```python
    assert set(first) == {
        "latent", "embedding", "encoder_embedding", "targets",
        "window", "step", "episode",
    }
```

- [ ] **Step 5: Run the probe suite**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -q`
Expected: PASS, no failures. The new keys are additive, so every existing
assertion on the four payload arrays is unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/probe.py tests/eval/test_probe.py
git commit -m "feat: gather_probe_data labels every row with its episode, the unit M3j resamples"
```

---

### Task 2: `_block_bootstrap_ci` can resample by label

**Why:** backward displacement at k drops the first k rows of every window, and the positional path *raises* on a row count that is not a whole number of windows. Label-based grouping makes filtered rows legal and, separately, lets M3j cluster on episodes. The positional path must stay bit-identical because `filtering_gain` reports through it.

**Files:**
- Modify: `src/mbfps/eval/probe.py` — `_block_bootstrap_ci`, roughly lines 528–585
- Test: `tests/eval/test_probe.py`

**Interfaces:**
- Consumes: `data["episode"]` from Task 1 (only in later tasks; this task needs no gather).
- Produces: `_block_bootstrap_ci(joint_predicted, embedding_predicted, targets, *, window=None, groups=None, resamples, confidence, seed) -> tuple[float, float]`. Exactly one of `window` / `groups` must be given. Task 3 calls it with `groups`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/eval/test_probe.py`, in the `filtering_gain` region (after the
existing `_block_bootstrap_ci` tests). Import the private helper the way the
file already imports others in that region:

```python
from mbfps.eval.probe import _block_bootstrap_ci  # noqa: E402


def _bootstrap_arrays(n_rows: int, seed: int = 7):
    """Two predictions and a target with real structure, so the interval is not
    degenerate and a change in the resampling actually moves it."""
    rng = np.random.default_rng(seed)
    targets = rng.normal(size=(n_rows, 2)) * 10.0
    joint = targets + rng.normal(size=(n_rows, 2)) * 2.0
    embedding = targets + rng.normal(size=(n_rows, 2)) * 4.0
    return joint, embedding, targets


def test_block_bootstrap_by_label_reproduces_the_positional_path_exactly():
    """THE equivalence pin. Grouping by label must be a generalisation, not a
    replacement: given labels that describe the same blocks the positional path
    builds, the two must agree to the last bit. Without this, Task 3 silently
    re-bases every recorded `filtering_gain` interval."""
    joint, embedding, targets = _bootstrap_arrays(60)
    positional = _block_bootstrap_ci(
        joint, embedding, targets, window=10, resamples=200, confidence=0.9, seed=3,
    )
    labelled = _block_bootstrap_ci(
        joint, embedding, targets,
        groups=np.arange(60) // 10, resamples=200, confidence=0.9, seed=3,
    )
    assert labelled == positional, (
        f"labelled {labelled} != positional {positional}; the label path is not "
        "a generalisation of the positional one"
    )


def test_block_bootstrap_by_label_accepts_groups_of_unequal_size():
    """The reason the label path exists: backward displacement at k drops the
    first k rows of every window, so the groups are no longer equal-length and
    the positional path cannot express them."""
    joint, embedding, targets = _bootstrap_arrays(23)
    groups = np.array([0] * 5 + [1] * 11 + [2] * 7)
    low, high = _block_bootstrap_ci(
        joint, embedding, targets, groups=groups, resamples=100, confidence=0.9, seed=1,
    )
    assert low < high and np.isfinite([low, high]).all()


def test_block_bootstrap_rejects_both_or_neither_blocking():
    """Two ways to block is an ambiguity, not a convenience: a caller passing
    both would silently get one of them."""
    joint, embedding, targets = _bootstrap_arrays(20)
    with pytest.raises(ValueError, match="exactly one"):
        _block_bootstrap_ci(joint, embedding, targets, resamples=10,
                            confidence=0.9, seed=0)
    with pytest.raises(ValueError, match="exactly one"):
        _block_bootstrap_ci(joint, embedding, targets, window=10,
                            groups=np.arange(20) // 10, resamples=10,
                            confidence=0.9, seed=0)


def test_block_bootstrap_rejects_a_label_per_row_mismatch():
    """One label per scored row. A shorter `groups` would silently drop rows
    from every resample."""
    joint, embedding, targets = _bootstrap_arrays(20)
    with pytest.raises(ValueError, match="one label per scored row"):
        _block_bootstrap_ci(joint, embedding, targets, groups=np.arange(19),
                            resamples=10, confidence=0.9, seed=0)


def test_block_bootstrap_by_label_is_order_independent():
    """Labels identify groups; the row ORDER within the array must not change
    which rows travel together. A gather that emitted rows in a different order
    has to give the same interval."""
    joint, embedding, targets = _bootstrap_arrays(30)
    groups = np.arange(30) // 6
    straight = _block_bootstrap_ci(
        joint, embedding, targets, groups=groups, resamples=150,
        confidence=0.9, seed=5,
    )
    order = np.random.default_rng(0).permutation(30)
    shuffled = _block_bootstrap_ci(
        joint[order], embedding[order], targets[order], groups=groups[order],
        resamples=150, confidence=0.9, seed=5,
    )
    assert shuffled == pytest.approx(straight, abs=1e-12)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -k block_bootstrap -v`
Expected: the five new tests FAIL with `TypeError: _block_bootstrap_ci() got an
unexpected keyword argument 'groups'`; the pre-existing `_block_bootstrap_ci`
tests still PASS.

- [ ] **Step 3: Add the `groups` path**

Replace `_block_bootstrap_ci`'s signature and body in
`src/mbfps/eval/probe.py`. Keep the existing docstring and append the new
paragraphs; keep every existing error message verbatim.

```python
def _block_bootstrap_ci(
    joint_predicted: np.ndarray,
    embedding_predicted: np.ndarray,
    targets: np.ndarray,
    window: int | None = None,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> tuple[float, float]:
    """Percentile interval for the gain, resampling WHOLE WINDOWS.

    The resampling unit is the window, not the row. A window is `window`
    consecutive frames of one episode -- 50 in the production setting -- and
    consecutive Doom frames are near-duplicates, so a row-level bootstrap counts
    ~50 correlated observations as 50 independent ones and returns an interval
    several times too narrow. The windows are exactly the blocks
    `gather_probe_data` cuts, on a stride of their own length, so they are the
    coarsest unit the scored array actually contains.

    The probes are held FIXED across resamples. This is an interval on the
    scored sample -- how much the gain would move on a different draw of
    evaluation windows -- not on the whole fit/select/score pipeline.

    TWO WAYS TO BLOCK, EXACTLY ONE PER CALL. `window` blocks POSITIONALLY, in
    fixed strides, which is what `filtering_gain` has always reported through
    and is byte-for-byte unchanged. `groups` blocks by LABEL: rows sharing a
    label travel together. Two callers need the label form and the positional
    form cannot express either. Backward displacement at k drops the first k
    rows of every window, so the groups stop being equal-length and the stride
    arithmetic below would raise; and M3j clusters on the 24 EPISODES rather
    than the 229 windows, because several non-overlapping windows cut from one
    trajectory are not independent observations. Given labels that describe the
    same blocks the stride builds, the two paths agree to the last bit -- the
    same `picked` indices in the same order from the same generator -- and a
    test pins that.
    """
    if (window is None) == (groups is None):
        raise ValueError(
            "pass exactly one of `window` (positional blocks) or `groups` "
            "(labelled blocks); passing both or neither leaves the resampling "
            "unit ambiguous"
        )
    if resamples < 1:
        raise ValueError(f"resamples must be at least 1, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    n_rows = targets.shape[0]
    if groups is None:
        if window < 1:
            raise ValueError(f"window must be at least 1 row, got {window}")
        if n_rows % window:
            raise ValueError(
                f"{n_rows} scored rows is not a whole number of {window}-row windows; "
                "the block bootstrap would mix parts of two windows into one block"
            )
        strides = np.arange(n_rows).reshape(n_rows // window, window)
        n_blocks = strides.shape[0]

        def pick(indices: np.ndarray) -> np.ndarray:
            return strides[indices].reshape(-1)
    else:
        labels = np.asarray(groups)
        if labels.shape != (n_rows,):
            raise ValueError(
                f"groups must be one label per scored row; got {labels.shape} "
                f"for {n_rows} rows"
            )
        # Stable sort, so a group's rows keep their original relative order and
        # the labelled path reproduces the positional one row for row.
        order = np.argsort(labels, kind="stable")
        _, starts = np.unique(labels[order], return_index=True)
        parts = np.split(order, starts[1:])
        n_blocks = len(parts)

        def pick(indices: np.ndarray) -> np.ndarray:
            return np.concatenate([parts[j] for j in indices])

    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        picked = generator.integers(0, n_blocks, size=n_blocks)
        rows = pick(picked)
        draws[i] = _mean_r2(joint_predicted[rows], targets[rows]) - _mean_r2(
            embedding_predicted[rows], targets[rows]
        )
    tail = 100.0 * (1.0 - confidence) / 2.0
    return float(np.percentile(draws, tail)), float(np.percentile(draws, 100.0 - tail))
```

Note the two validations that moved **above** the blocking branch (`resamples`,
`confidence`): they are argument checks, not blocking checks, and their messages
are unchanged, so the existing tests that pin them still pass.

- [ ] **Step 4: Run the probe suite**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -q`
Expected: PASS. In particular `test_block_bootstrap_by_label_reproduces_the_positional_path_exactly`
must pass with exact equality, not approximately.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/eval/probe.py tests/eval/test_probe.py
git commit -m "feat: _block_bootstrap_ci can resample by label, pinned bit-identical to the positional path"
```

---

### Task 3: `gain_from_blocks` — the general core, with `filtering_gain` pinned unchanged

**Why:** M3j needs the same statistic with an arbitrary second block and an arbitrary target on a filtered row set. `_gain_from_splits` already does everything except those three things. Generalise it once rather than duplicating the three-split and bootstrap logic, and prove `filtering_gain` did not move.

**Files:**
- Modify: `src/mbfps/eval/probe.py` — add `GainSplit` and `gain_from_blocks`; reduce `_gain_from_splits` to a wrapper
- Test: `tests/eval/test_probe.py`

**Interfaces:**
- Consumes: `_block_bootstrap_ci(..., groups=...)` from Task 2.
- Produces, both used by Task 7:

```python
@dataclass(frozen=True)
class GainSplit:
    base: np.ndarray    # (n, p_base)   -- in BOTH arms
    block: np.ndarray   # (n, p_block)  -- appended to the joint arm only
    target: np.ndarray  # (n, c)

def gain_from_blocks(
    fit: GainSplit,
    select: GainSplit | None,
    score: GainSplit,
    *,
    groups: np.ndarray,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict: ...
```

Returns exactly the ten keys `_gain_from_splits` already returns: `"gain"`,
`"joint_r2"`, `"embedding_r2"`, `"ci_low"`, `"ci_high"`, `"confidence"`,
`"n_scored_windows"`, `"ridge_selected"`, `"joint_ridge"`, `"embedding_ridge"`.
`"n_scored_windows"` becomes the number of distinct `groups` labels.

- [ ] **Step 1: Write the byte-identity pin first, against the CURRENT code**

Add to `tests/eval/test_probe.py` in the `filtering_gain` region. These numbers
were captured from the unmodified `_gain_from_splits` on 2026-09-27 and are the
whole point of the test — do not recompute them after refactoring.

```python
def test_gain_from_splits_output_is_byte_identical_after_the_generalisation():
    """THE regression pin for Task 3. `filtering_gain` reports through
    `_gain_from_splits` onto the gate's own path (`study.py:589`,
    `scripts/eval_rollout.py:201`), so generalising its body is only safe if the
    numbers do not move. Every value below was captured from the code BEFORE
    `gain_from_blocks` existed. If a refactor changes one, the refactor is
    wrong -- do not re-record them."""
    latent, embedding, targets = _history_case(600, seed=0)
    parts = [
        _split(latent[s], embedding[s], targets[s])
        for s in (slice(0, 200), slice(200, 400), slice(400, 600))
    ]
    out = _gain_from_splits(parts[0], parts[1], parts[2], h_dim=2,
                            window=5, resamples=200, confidence=0.9, seed=11)
    assert out == {
        "gain": 0.9996786109619562,
        "joint_r2": 0.9996512622688909,
        "embedding_r2": -2.7348693065254448e-05,
        "ci_low": 0.9996553965287523,
        "ci_high": 1.0154949292104678,
        "confidence": 0.9,
        "n_scored_windows": 40,
        "ridge_selected": True,
        "joint_ridge": 0.1,
        "embedding_ridge": 10000000.0,
    }
```

- [ ] **Step 2: Run it to verify it PASSES on the unmodified code**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -k byte_identical -v`
Expected: **PASS.** This is the one test in the plan that must pass before the
implementation, not after — it is a regression pin, not a specification. If it
fails here, the golden values are wrong for this machine and must be
re-captured before going further (and the numbers above corrected).

- [ ] **Step 3: Commit the pin on its own**

```bash
git add tests/eval/test_probe.py
git commit -m "test: pin _gain_from_splits' exact output before generalising its body"
```

- [ ] **Step 4: Write the failing tests for the new function**

```python
from mbfps.eval.probe import GainSplit, gain_from_blocks  # noqa: E402


def _blocks(latent, embedding, targets, rows, h_dim=2):
    """A `GainSplit` over `rows`, with the deterministic half as the block --
    the same arrangement `_gain_from_splits` builds, so the two can be compared."""
    return GainSplit(
        base=embedding[rows], block=latent[rows, :h_dim], target=targets[rows],
    )


def test_gain_from_blocks_matches_gain_from_splits_on_the_same_arrangement():
    """The generalisation must be a generalisation. Handed the block and target
    `_gain_from_splits` builds internally, and labels describing its positional
    blocks, it must return the same dict."""
    latent, embedding, targets = _history_case(600, seed=0)
    sl = (slice(0, 200), slice(200, 400), slice(400, 600))
    parts = [_split(latent[s], embedding[s], targets[s]) for s in sl]
    reference = _gain_from_splits(parts[0], parts[1], parts[2], h_dim=2,
                                  window=5, resamples=200, confidence=0.9, seed=11)
    general = gain_from_blocks(
        _blocks(latent, embedding, targets, np.arange(600)[sl[0]]),
        _blocks(latent, embedding, targets, np.arange(600)[sl[1]]),
        _blocks(latent, embedding, targets, np.arange(600)[sl[2]]),
        groups=np.arange(200) // 5, resamples=200, confidence=0.9, seed=11,
    )
    assert general == reference


def test_gain_from_blocks_accepts_a_filtered_row_set():
    """The reason it exists. Backward displacement at k has no target for the
    first k rows of a window, so the scored set is not a whole number of
    windows and `_gain_from_splits` cannot express it."""
    latent, embedding, targets = _history_case(600, seed=1)
    keep = np.arange(600)[np.arange(600) % 5 != 0]   # drop row 0 of every block
    sl = (keep[keep < 200], keep[(keep >= 200) & (keep < 400)], keep[keep >= 400])
    out = gain_from_blocks(
        _blocks(latent, embedding, targets, sl[0]),
        _blocks(latent, embedding, targets, sl[1]),
        _blocks(latent, embedding, targets, sl[2]),
        groups=sl[2] // 5, resamples=100, confidence=0.9, seed=2,
    )
    assert out["n_scored_windows"] == 40, "one group per surviving window"
    assert out["gain"] > 0.5, (
        "the lagged block still carries the target after filtering; a near-zero "
        "gain here means the rows and the target came apart"
    )


def test_gain_from_blocks_counts_groups_not_rows_over_a_window():
    """`n_scored_windows` was `rows // window`. With labels there is no window
    length, so it has to be the number of distinct labels -- and unequal groups
    are exactly the case that separates the two."""
    latent, embedding, targets = _history_case(300, seed=3)
    rows = np.arange(300)
    groups = np.repeat(np.arange(7), (40, 40, 40, 40, 40, 40, 60))
    out = gain_from_blocks(
        _blocks(latent, embedding, targets, rows[:100]),
        _blocks(latent, embedding, targets, rows[100:200]),
        _blocks(latent, embedding, targets, rows[200:300]),
        groups=groups[200:300], resamples=50, confidence=0.9, seed=4,
    )
    assert out["n_scored_windows"] == len(np.unique(groups[200:300]))


def test_gain_from_blocks_puts_the_base_in_both_arms():
    """The cancellation the whole statistic rests on. If the base appeared only
    in one arm, the gain would measure the two feature sets' widths as much as
    the block's contribution -- and neither the bottleneck nor the
    position-constrains-motion confound would cancel."""
    latent, embedding, targets = _history_case(600, seed=5)
    rows = np.arange(600)
    # A block of pure noise must not show a gain: the base is identical in both
    # arms, so there is nothing for extra width alone to buy.
    noise = np.random.default_rng(9).normal(size=(600, 2))
    def split(sl):
        return GainSplit(base=embedding[sl], block=noise[sl], target=targets[sl])
    out = gain_from_blocks(
        split(rows[:200]), split(rows[200:400]), split(rows[400:600]),
        groups=np.arange(200) // 5, resamples=200, confidence=0.9, seed=6,
    )
    assert out["ci_low"] <= 0.0, (
        f"a noise block cleared zero (ci_low={out['ci_low']}); the base is not "
        "in both arms, or the selection is being taken on the scored rows"
    )


def test_gain_from_blocks_rejects_misaligned_rows():
    """Three arrays that do not describe the same rows is the defect this
    function is most exposed to, because its caller row-selects all three
    separately."""
    latent, embedding, targets = _history_case(300, seed=7)
    rows = np.arange(300)
    good = _blocks(latent, embedding, targets, rows[:100])
    bad = GainSplit(base=embedding[:100], block=latent[:99, :2], target=targets[:100])
    with pytest.raises(ValueError, match="same number of rows"):
        gain_from_blocks(bad, None, good, groups=np.arange(100) // 5, resamples=10)
    with pytest.raises(ValueError, match="one label per scored row"):
        gain_from_blocks(good, None, good, groups=np.arange(99) // 5, resamples=10)


def test_gain_from_blocks_without_a_selection_split_falls_back_unbiased():
    """`select=None` must take `fit_probe`'s default penalty for BOTH arms and
    say so, rather than selecting on the rows it scores."""
    latent, embedding, targets = _history_case(400, seed=8)
    rows = np.arange(400)
    out = gain_from_blocks(
        _blocks(latent, embedding, targets, rows[:200]), None,
        _blocks(latent, embedding, targets, rows[200:]),
        groups=np.arange(200) // 5, resamples=50, confidence=0.9, seed=1,
    )
    assert out["ridge_selected"] is False
    assert out["joint_ridge"] == 1e3 and out["embedding_ridge"] == 1e3
```

- [ ] **Step 5: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -k gain_from_blocks -v`
Expected: FAIL, all six with `ImportError: cannot import name 'GainSplit'`.

- [ ] **Step 6: Add `GainSplit` and `gain_from_blocks`**

In `src/mbfps/eval/probe.py`, add `from dataclasses import dataclass` to the
imports, then insert both directly **above** `_gain_from_splits`:

```python
@dataclass(frozen=True)
class GainSplit:
    """One split's three arrays for `gain_from_blocks`, already row-selected.

    `base` goes into BOTH arms and `block` into the joint arm only, which is
    what makes the statistic an increment rather than a level. The caller
    row-selects all three together: a target defined on a subset of the rows
    (backward displacement has none for the first k rows of a window) must be
    handed the SAME subset of features, and `gain_from_blocks` checks the three
    row counts rather than trusting it.
    """

    base: np.ndarray
    block: np.ndarray
    target: np.ndarray

    def joint(self) -> np.ndarray:
        return np.concatenate(
            [np.asarray(self.base, dtype=np.float64),
             np.asarray(self.block, dtype=np.float64)], axis=1,
        )

    def rows(self) -> int:
        counts = {
            np.asarray(self.base).shape[0],
            np.asarray(self.block).shape[0],
            np.asarray(self.target).shape[0],
        }
        if len(counts) != 1:
            raise ValueError(
                f"base, block and target must describe the same number of rows; "
                f"got {sorted(counts)}"
            )
        return counts.pop()


def gain_from_blocks(
    fit: GainSplit,
    select: GainSplit | None,
    score: GainSplit,
    *,
    groups: np.ndarray,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """`R2([base (+) block] -> target) - R2([base] -> target)` on three splits.

    The general form of `_gain_from_splits`, which is now a wrapper. Three
    things are the caller's here rather than hardcoded: the second feature
    block, the target, and the resampling groups. M3j needs all three -- a
    ladder of blocks (`enc(t-k)`, `h`, `z`, `h (+) z`), two motion targets, and
    episode-level rather than window-level clustering.

    The split discipline is unchanged and is the reason this is not
    `filtering_comparison`: weights from `fit`, ridge selected on `select`,
    reported R^2 and interval from `score`. A gain is a DIFFERENCE OF LEVELS
    between feature sets of different widths, so a ridge maximum taken on the
    scored rows favours the wider one and manufactures a positive gain out of
    the selection alone. `select=None` skips selection and takes `fit_probe`'s
    default penalty for both arms -- unbiased too, just weaker, and
    `ridge_selected` says which happened.

    `n_scored_windows` is the number of distinct `groups` labels. Under the
    positional wrapper that equals `rows // window`, which is what it always
    meant; under a filtered row set it is the number of surviving groups, which
    is what the interval actually resamples.
    """
    n_scored = score.rows()
    fit.rows()
    joint_fit, base_fit = fit.joint(), np.asarray(fit.base, dtype=np.float64)
    joint_score, base_score = score.joint(), np.asarray(score.base, dtype=np.float64)
    target_fit, target_score = fit.target, score.target

    if select is None:
        joint_probe = fit_probe(joint_fit, target_fit)
        base_probe = fit_probe(base_fit, target_fit)
    else:
        select.rows()
        joint_probe = fit_probe(joint_fit, target_fit, select.joint(), select.target)
        base_probe = fit_probe(
            base_fit, target_fit,
            np.asarray(select.base, dtype=np.float64), select.target,
        )

    joint_predicted = apply_probe(joint_probe, joint_score)
    base_predicted = apply_probe(base_probe, base_score)
    joint_r2 = _mean_r2(joint_predicted, target_score)
    base_r2 = _mean_r2(base_predicted, target_score)
    low, high = _block_bootstrap_ci(
        joint_predicted, base_predicted, target_score,
        groups=groups, resamples=resamples, confidence=confidence, seed=seed,
    )
    return {
        "gain": joint_r2 - base_r2,
        "joint_r2": joint_r2,
        "embedding_r2": base_r2,
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
        "n_scored_windows": int(np.unique(np.asarray(groups)).size),
        "ridge_selected": select is not None,
        "joint_ridge": joint_probe["ridge"],
        "embedding_ridge": base_probe["ridge"],
    }
```

The returned key stays `"embedding_r2"` rather than becoming `"base_r2"`:
`filtering_gain`'s callers in `study.py` and `scripts/eval_rollout.py` read it
by that name, and renaming it is a behaviour change this task is pinned against.

- [ ] **Step 7: Reduce `_gain_from_splits` to a wrapper**

Replace `_gain_from_splits`' body in `src/mbfps/eval/probe.py`. Keep its
existing docstring and append the last paragraph:

```python
def _gain_from_splits(
    fit: dict,
    select: dict | None,
    score: dict,
    h_dim: int,
    window: int,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """The array-level half of `filtering_gain`; see that docstring for the why.

    `fit`, `select` and `score` are three DISJOINT `gather_probe_data` dicts.
    The weights come from `fit`, the ridge is selected on `select`, and the
    reported R^2 and the interval come from `score` -- so, unlike
    `filtering_comparison`, no part of the number on the scored rows was tuned
    on those rows. `select=None` skips selection entirely and takes
    `fit_probe`'s default penalty; that is unbiased too, just weaker.

    This is now a thin wrapper over `gain_from_blocks`, which took over the
    split, selection and bootstrap logic so M3j could reuse it with a different
    block and target. What is fixed HERE is the block (`latent[:, :h_dim]`, the
    deterministic head), the target (the privileged state) and the blocking
    (positional, in `window`-row strides). Its output is pinned byte-for-byte by
    a test, because `filtering_gain` reports through it onto the gate's own path.
    """
    def split(data: dict) -> GainSplit:
        embedding = np.asarray(data["encoder_embedding"], dtype=np.float64)
        latent = np.asarray(data["latent"], dtype=np.float64)
        if not 1 <= h_dim <= latent.shape[1]:
            raise ValueError(
                f"h_dim={h_dim} does not index a {latent.shape[1]}-wide latent"
            )
        # `latent` is `cat([h, z])` -- h FIRST, per RSSM.observe. Slicing the
        # tail instead reads the stochastic state and answers a different
        # question with the same shapes.
        return GainSplit(base=embedding, block=latent[:, :h_dim], target=data["targets"])

    n_rows = np.asarray(score["targets"]).shape[0]
    if window < 1:
        raise ValueError(f"window must be at least 1 row, got {window}")
    if n_rows % window:
        raise ValueError(
            f"{n_rows} scored rows is not a whole number of {window}-row windows; "
            "the block bootstrap would mix parts of two windows into one block"
        )
    return gain_from_blocks(
        split(fit), None if select is None else split(select), split(score),
        # The same blocks the stride path builds, as labels: Task 2's
        # equivalence pin is what makes this substitution legal.
        groups=np.arange(n_rows) // window,
        resamples=resamples, confidence=confidence, seed=seed,
    )
```

The two `window` validations are duplicated here from `_block_bootstrap_ci`'s
positional branch **deliberately**: the existing tests raise them through
`_gain_from_splits`, and the labelled call below can no longer reach them.

- [ ] **Step 8: Run the probe suite**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -q`
Expected: PASS, including `test_gain_from_splits_output_is_byte_identical_after_the_generalisation`
with the same ten values it passed against in Step 2.

- [ ] **Step 9: Run every suite that reads `filtering_gain`**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py tests/eval/test_study.py tests/eval/test_eval_rollout_script.py -q`
Expected: PASS. These three cover every caller of `filtering_gain`.

- [ ] **Step 10: Commit**

```bash
git add src/mbfps/eval/probe.py tests/eval/test_probe.py
git commit -m "feat: gain_from_blocks generalises the gain over block, target and grouping; _gain_from_splits is now a wrapper"
```

---

### Task 4: `retention.py` — the constants, the row shift, the targets and the rungs

**Why:** everything Reading E is computed from, with no thresholds and no I/O. The row shift is the load-bearing piece: both the backward targets and the `two_frame` rung need "the row `k` steps earlier in the same window", and getting it from array contiguity rather than from the labels is the kind of assumption that survives every test until a gather changes order.

**Files:**
- Create: `src/mbfps/eval/retention.py`
- Test: `tests/eval/test_retention.py`

**Interfaces:**
- Consumes: `data["window"]`, `data["step"]`, `data["targets"]`, `data["encoder_embedding"]`, `data["latent"]` from `gather_probe_data` (Task 1 added `data["episode"]`, used by Task 7 rather than here).
- Produces, all used by Tasks 5, 6 and 7:

```python
RETENTION_FAMILY: int = 3
SEEDS_REQUIRED: int = 2
ARMS_REQUIRED: int = 2
K_REPORTED: tuple[int, ...] = (1, 4, 15)
DECISION_K: int = 4
RUNGS: tuple[str, ...] = ("two_frame", "deterministic", "stochastic", "full")
LATENT_RUNGS: tuple[str, ...] = ("deterministic", "stochastic", "full")
Z_BEARING_RUNGS: tuple[str, ...] = ("stochastic", "full")
TARGETS: tuple[str, ...] = ("translation", "rotation")
CONFIDENCE: float = 0.95
RESAMPLES: int = 1000

def shifted_rows(window, step, k: int) -> tuple[np.ndarray, np.ndarray]: ...
def backward_translation(targets, window, step, k: int) -> tuple[np.ndarray, np.ndarray]: ...
def backward_rotation(targets, window, step, k: int) -> tuple[np.ndarray, np.ndarray]: ...
def rung_block(data: dict, rung: str, *, rows, source, h_dim: int) -> np.ndarray: ...
```

Each target builder returns `(values, rows)`: the target values and the row
indices they are defined on. `rung_block` is handed those same `rows` plus the
`source` partners so all four rungs are scored on one identical row set.

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_retention.py`:

```python
"""M3j: where on the enc(t) -> z -> h path is observed motion lost?"""

import numpy as np
import pytest

from mbfps.eval.retention import (
    ARMS_REQUIRED,
    CONFIDENCE,
    DECISION_K,
    K_REPORTED,
    LATENT_RUNGS,
    RESAMPLES,
    RETENTION_FAMILY,
    RUNGS,
    SEEDS_REQUIRED,
    TARGETS,
    Z_BEARING_RUNGS,
    backward_rotation,
    backward_translation,
    rung_block,
    shifted_rows,
)

# Two windows of four steps. Positions are hand-chosen so every displacement
# below can be read off by eye, and the two windows differ so a target that
# silently paired rows across windows would change the numbers.
WINDOW = np.array([0, 0, 0, 0, 1, 1, 1, 1])
STEP = np.array([0, 1, 2, 3, 0, 1, 2, 3])
POS = np.array([
    [0.0, 0.0], [3.0, -2.0], [6.0, -4.0], [9.0, -6.0],
    [10.0, 10.0], [12.0, 13.0], [14.0, 16.0], [16.0, 19.0],
])
# Degrees. Window 0 WRAPS through 360 (350 -> 20 is +30, not -330), which is the
# case a plain subtraction gets wrong; window 1 turns a uniform +45.
ANGLE_DEG = np.array([350.0, 20.0, 50.0, 80.0, 100.0, 145.0, 190.0, 235.0])


def _targets() -> np.ndarray:
    """The `(N, 4)` array `probe.probe_targets` builds: pos_x, pos_y, sin, cos."""
    radians = np.deg2rad(ANGLE_DEG)
    return np.column_stack([POS[:, 0], POS[:, 1], np.sin(radians), np.cos(radians)])


def test_constants_are_the_pre_registered_values():
    """Spec 2.2, 2.3 and 3.1. These are pre-registered, so they are pinned by
    exact equality rather than by range -- the whole point of writing them down
    before the run is that they cannot drift afterwards."""
    assert RETENTION_FAMILY == 3
    assert SEEDS_REQUIRED == 2 and ARMS_REQUIRED == 2
    assert K_REPORTED == (1, 4, 15)
    assert DECISION_K == 4
    assert RUNGS == ("two_frame", "deterministic", "stochastic", "full")
    assert LATENT_RUNGS == ("deterministic", "stochastic", "full")
    assert Z_BEARING_RUNGS == ("stochastic", "full")
    assert TARGETS == ("translation", "rotation")
    assert CONFIDENCE == 0.95 and RESAMPLES == 1000


def test_decision_k_is_one_of_the_reported_horizons():
    """A decision horizon outside `K_REPORTED` would be decided on numbers the
    table never prints."""
    assert DECISION_K in K_REPORTED


def test_latent_rungs_and_z_bearing_rungs_are_subsets_of_rungs():
    """The status logic reads these three tuples; a name in one and not in
    `RUNGS` would be a rung nothing ever measures."""
    assert set(LATENT_RUNGS) < set(RUNGS)
    assert set(Z_BEARING_RUNGS) < set(LATENT_RUNGS)
    assert set(RUNGS) - set(LATENT_RUNGS) == {"two_frame"}


def test_shifted_rows_pairs_each_row_with_the_one_k_steps_back_in_its_window():
    rows, source = shifted_rows(WINDOW, STEP, 2)
    assert rows.tolist() == [2, 3, 6, 7]
    assert source.tolist() == [0, 1, 4, 5]


def test_shifted_rows_never_pairs_across_windows():
    """At k = 3 only the last row of each window qualifies. A pairing that
    walked off the front of window 1 into window 0 would still be shape-valid
    and would silently score a displacement between two different episodes."""
    rows, source = shifted_rows(WINDOW, STEP, 3)
    assert rows.tolist() == [3, 7]
    assert source.tolist() == [0, 4]
    assert np.array_equal(WINDOW[rows], WINDOW[source]), "a pair spans two windows"


def test_shifted_rows_is_order_independent():
    """The pairing comes from the LABELS, not from the array order. Deriving it
    from contiguity would pass on every gather that happens to emit rows in step
    order and break silently on one that does not."""
    straight_rows, straight_source = shifted_rows(WINDOW, STEP, 1)
    order = np.array([5, 0, 7, 2, 4, 1, 6, 3])
    rows, source = shifted_rows(WINDOW[order], STEP[order], 1)
    pairs = {
        (int(WINDOW[order][r]), int(STEP[order][r]), int(STEP[order][s]))
        for r, s in zip(rows, source)
    }
    expected = {
        (int(WINDOW[r]), int(STEP[r]), int(STEP[s]))
        for r, s in zip(straight_rows, straight_source)
    }
    assert pairs == expected


def test_shifted_rows_rejects_a_k_below_one():
    with pytest.raises(ValueError, match="k must be >= 1"):
        shifted_rows(WINDOW, STEP, 0)


def test_shifted_rows_returns_nothing_when_no_window_is_long_enough():
    """Not an error: the caller reports the row count and a horizon with no rows
    is a fact about the protocol, not a bug. Returning empty keeps the caller's
    refusal in one place."""
    rows, source = shifted_rows(WINDOW, STEP, 4)
    assert rows.size == 0 and source.size == 0


def test_backward_translation_is_the_displacement_already_observed():
    """`p(t) - p(t-k)`, hand-computed. Window 0 moves (3, -2) per step and
    window 1 moves (2, 3), so at k = 2 the two windows differ -- which is what
    catches a builder that paired every row with row 0 of its own window."""
    values, rows = backward_translation(_targets(), WINDOW, STEP, 2)
    assert rows.tolist() == [2, 3, 6, 7]
    np.testing.assert_allclose(values, [[6.0, -4.0], [6.0, -4.0], [4.0, 6.0], [4.0, 6.0]])


def test_backward_translation_at_k_one_is_the_single_step_move():
    values, rows = backward_translation(_targets(), WINDOW, STEP, 1)
    assert rows.tolist() == [1, 2, 3, 5, 6, 7]
    np.testing.assert_allclose(values, [
        [3.0, -2.0], [3.0, -2.0], [3.0, -2.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0],
    ])


def test_backward_rotation_wraps_through_360():
    """Window 0's first step is 350 deg -> 20 deg. That is +30, and a plain
    subtraction of the two angles gives -330. The target is (sin, cos) of the
    change, so the two differ: sin(-330) == sin(30) but the pair as a whole only
    matches if the wrap is handled -- which is why BOTH columns are asserted."""
    values, rows = backward_rotation(_targets(), WINDOW, STEP, 1)
    assert rows.tolist() == [1, 2, 3, 5, 6, 7]
    thirty = [np.sin(np.deg2rad(30.0)), np.cos(np.deg2rad(30.0))]
    forty_five = [np.sin(np.deg2rad(45.0)), np.cos(np.deg2rad(45.0))]
    np.testing.assert_allclose(
        values, [thirty, thirty, thirty, forty_five, forty_five, forty_five], atol=1e-12,
    )


def test_backward_rotation_and_translation_agree_on_their_rows():
    """The two targets are scored against the same feature rows at each k. If
    they disagreed on which rows qualify, the rotation control would be measured
    on a different sample than the reading it controls."""
    for k in (1, 2, 3):
        _, translation_rows = backward_translation(_targets(), WINDOW, STEP, k)
        _, rotation_rows = backward_rotation(_targets(), WINDOW, STEP, k)
        assert translation_rows.tolist() == rotation_rows.tolist(), f"k={k}"


def _data() -> dict:
    """The three feature arrays `rung_block` reads, with h_dim = 2 so `h` and
    `z` are DIFFERENT widths -- slicing the wrong end is then a shape error in
    some cases and a wrong number in others, and both must be caught."""
    enc = np.arange(8 * 2, dtype=np.float64).reshape(8, 2) * 1.0
    latent = np.arange(8 * 5, dtype=np.float64).reshape(8, 5) * 10.0
    return {"encoder_embedding": enc, "latent": latent}


def test_rung_block_two_frame_is_the_frame_the_displacement_is_measured_FROM():
    """Spec 2.2: `enc(t-k)`, not `enc(t-1)`. At k = 2 the two differ, and only
    `enc(t-k)` can carry a 2-step displacement together with `enc(t)`."""
    rows, source = shifted_rows(WINDOW, STEP, 2)
    block = rung_block(_data(), "two_frame", rows=rows, source=source, h_dim=2)
    np.testing.assert_array_equal(block, _data()["encoder_embedding"][source])
    # And it is NOT enc(t-1): row 3's partner is row 1, not row 2.
    assert not np.array_equal(block, _data()["encoder_embedding"][rows - 1])


def test_rung_block_splits_the_latent_at_h_dim():
    """`latent` is `cat([h, z])`, h FIRST. The two halves are different widths
    here so a reversed slice is caught by the shape as well as the values."""
    rows, source = shifted_rows(WINDOW, STEP, 1)
    latent = _data()["latent"]
    np.testing.assert_array_equal(
        rung_block(_data(), "deterministic", rows=rows, source=source, h_dim=2),
        latent[rows, :2],
    )
    np.testing.assert_array_equal(
        rung_block(_data(), "stochastic", rows=rows, source=source, h_dim=2),
        latent[rows, 2:],
    )
    np.testing.assert_array_equal(
        rung_block(_data(), "full", rows=rows, source=source, h_dim=2), latent[rows],
    )


def test_rung_block_gives_every_rung_the_same_number_of_rows():
    """All four rungs are differences against ONE shared base fit on ONE row
    set (spec 2.1). A rung with a different row count would be a gain against a
    different base, and the four would stop being comparable."""
    rows, source = shifted_rows(WINDOW, STEP, 2)
    widths = {
        rung: rung_block(_data(), rung, rows=rows, source=source, h_dim=2).shape[0]
        for rung in RUNGS
    }
    assert set(widths.values()) == {rows.size}, widths


def test_rung_block_rejects_an_unknown_rung():
    rows, source = shifted_rows(WINDOW, STEP, 1)
    with pytest.raises(ValueError, match="unknown rung"):
        rung_block(_data(), "recurrent", rows=rows, source=source, h_dim=2)


def test_rung_block_rejects_an_h_dim_that_does_not_index_the_latent():
    """A silently out-of-range slice returns an empty or full block instead of
    raising, and the gain would then be measured on the wrong half."""
    rows, source = shifted_rows(WINDOW, STEP, 1)
    for bad in (0, 5, 9):
        with pytest.raises(ValueError, match="does not index"):
            rung_block(_data(), "deterministic", rows=rows, source=source, h_dim=bad)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_retention.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'mbfps.eval.retention'`.

- [ ] **Step 3: Write `retention.py`**

Create `src/mbfps/eval/retention.py`:

```python
"""M3j: where on the enc(t) -> z -> h path is observed motion lost?

M3i read NO_MOTION -- at the decision horizon no seed of any arm beats
predicting no displacement at all -- and its section 3.3 retired all three of
M3h's remaining prior-side levers. But "upstream" names two levers that need
opposite changes and cost the same 13.5-hour retrain: the ENCODER, if the frozen
features do not resolve motion, or the OBJECTIVE, if they do and the RSSM
discards it. `world_model.py:81` is `embedding + reward + continue + kl`, where
the embedding target is the single frame just seen, so nothing in the loss ever
asks the latent to represent displacement.

Everything here is an INCREMENT over the current frame:

    gain = R2([enc(t) (+) B] -> y) - R2([enc(t)] -> y)

`enc(t)` sits in both arms, so the 32x32 categorical bottleneck cancels (the
argument `probe.filtering_gain` already makes) and so does the
position-constrains-motion confound, which is new here and is the reason M3i's
section 9 levels could not have answered this question: in a corridor your
location predicts your motion, and that credit belongs to the frame.

Targets are BACKWARD, so the model is scored only on motion it has already
observed -- the representation's best case, which is what makes a null decisive.

Everything here is pure: arrays in, arrays and readings out. No torch, no I/O,
no record schema. `scripts/latent_retention.py` owns all three.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Pre-registered (spec 2.2, 2.3, 3.1). Three arms, every gain within a cell
# against that cell's own base -- no arm is ranked against another, so the
# family is the arms and not their pairs.
RETENTION_FAMILY: int = 3
SEEDS_REQUIRED: int = 2
ARMS_REQUIRED: int = 2

# Reported at every one of these; a cleared gain at ANY of them counts as motion
# retained (spec 2.3). The disjunction is deliberate: the lever question is
# binary -- does the RSSM retain observed motion at all -- so being generous
# about WHERE makes a null across all three a stronger null.
K_REPORTED: tuple[int, ...] = (1, 4, 15)

# The horizon the reading is named for. k = 1 is degenerate on both targets
# (translation is 0.6% of the map extent; rotation is exactly zero 62.9% of the
# time), and of the two non-degenerate horizons k = 4 is the shorter, so it
# demands less integration from `h` and a null there is the stronger null.
DECISION_K: int = 4

# The ladder, in path order: what two real frames provide, then what each part
# of the latent adds. `two_frame` is a REFERENCE, not a ceiling -- `h`
# integrates the action sequence, which two frames do not contain, so a latent
# rung can legitimately exceed it.
RUNGS: tuple[str, ...] = ("two_frame", "deterministic", "stochastic", "full")
LATENT_RUNGS: tuple[str, ...] = ("deterministic", "stochastic", "full")
Z_BEARING_RUNGS: tuple[str, ...] = ("stochastic", "full")

# `translation` decides; `rotation` is the positive control. A 20-unit
# translation is ~2% of the map extent through a 112x112 frozen backbone, while
# a 32-degree turn rewrites the frame -- so reading rotation but not translation
# means spatial resolution, and reading neither means the RSSM discards motion
# as such.
TARGETS: tuple[str, ...] = ("translation", "rotation")

CONFIDENCE: float = 0.95
RESAMPLES: int = 1000
"""Interval mass and bootstrap draws. 0.95 per comparison rather than a
Bonferroni over 4 rungs x 3 horizons: that would need the 0.2nd percentile,
which 1000 draws cannot resolve. The multiple-comparison burden is carried by
the SEEDS_REQUIRED x ARMS_REQUIRED agreement instead -- the project's existing
instrument, and one this sample size can actually resolve."""


def shifted_rows(window, step, k: int) -> tuple[np.ndarray, np.ndarray]:
    """For every row with a partner `k` steps earlier IN THE SAME WINDOW, that
    row and its partner. `(rows, source)`, both `(m,)` int64.

    Built from the `window` and `step` LABELS rather than from array order.
    `gather_probe_data` does currently emit each window's rows contiguously in
    step order, and `rows - k` would therefore work today -- but that is a
    regularity of the gather, not a documented guarantee, and a pairing that
    walked off the front of one window into the previous one would stay
    shape-valid while scoring a displacement between two different episodes.

    A `k` no window is long enough for returns two empty arrays rather than
    raising: a horizon with no rows is a fact about the protocol that the
    caller reports beside its row counts, so the refusal lives in one place.
    """
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    window = np.asarray(window).astype(np.int64, copy=False)
    step = np.asarray(step).astype(np.int64, copy=False)
    if window.shape != step.shape or window.ndim != 1:
        raise ValueError(
            f"window {window.shape} and step {step.shape} must be the same 1-D shape"
        )
    index = {(int(w), int(s)): i for i, (w, s) in enumerate(zip(window, step))}
    rows, source = [], []
    for i, (w, s) in enumerate(zip(window, step)):
        partner = index.get((int(w), int(s) - k))
        if partner is not None:
            rows.append(i)
            source.append(partner)
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(source, dtype=np.int64),
    )


def backward_translation(targets, window, step, k: int) -> tuple[np.ndarray, np.ndarray]:
    """`p(t) - p(t-k)` as a VECTOR in map units, and the rows it is defined on.

    `targets` is the `(N, 4)` array `probe.probe_targets` builds -- pos_x,
    pos_y, sin(angle), cos(angle) -- so position is its first two columns.

    BACKWARD, which is the point: every frame in the pair has already been
    observed by the posterior, so this is the representation's best case. M3i
    scored FORWARD displacement, where the target also depends on the policy;
    a null here says the latent cannot report motion it has already seen, which
    is a sharper statement and a necessary condition for the forward one.
    """
    rows, source = shifted_rows(window, step, k)
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"targets must be (N, >=2), got {values.shape}")
    return values[rows, :2] - values[source, :2], rows


def backward_rotation(targets, window, step, k: int) -> tuple[np.ndarray, np.ndarray]:
    """`(sin, cos)` of the angle change over the last `k` steps, and its rows.

    The POSITIVE CONTROL for the motion question, not a competing reading. A
    20-unit translation is ~2% of the map extent seen through 112x112 frames
    from a frozen single-frame backbone; a 32-degree turn rewrites the frame. So
    a ladder that reads rotation but not translation is telling us about spatial
    RESOLUTION, while a ladder that reads neither is telling us the RSSM
    discards motion as such. Without this the translation null would be
    confounded with "the encoder cannot see a 20-unit move".

    `(sin, cos)` rather than degrees for the reason `probe.probe_targets` uses
    it on absolute angle: 359 and 1 must be near each other. The wrap needs no
    explicit handling HERE because sin and cos are already periodic -- taking
    them of the raw difference is identical to taking them of the wrapped one,
    and cheaper than wrapping first and then discarding the wrap.
    """
    rows, source = shifted_rows(window, step, k)
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError(f"targets must be (N, >=4), got {values.shape}")
    now = np.arctan2(values[rows, 2], values[rows, 3])
    then = np.arctan2(values[source, 2], values[source, 3])
    delta = now - then
    return np.stack([np.sin(delta), np.cos(delta)], axis=1), rows


def rung_block(data: dict, rung: str, *, rows, source, h_dim: int) -> np.ndarray:
    """The second feature block for `rung`, on `rows`.

    Handed the SAME `rows` every other rung gets, so all four are gains against
    one shared base fit on one row set (spec 2.1) and are therefore comparable
    to each other rather than each only to its own fit.

    `two_frame` reads `source` -- `enc(t-k)`, the frame the displacement is
    measured FROM. Not `enc(t-1)`: the two frames that determine
    `p(t) - p(t-k)` are `t` and `t-k`, so a frame one step back would leave the
    reference near-uninformative at every k > 1 and would also hand the latent
    rungs an advantage they did not earn, since `h(t)` sees frame `t-k` and the
    reference would not. At k = 1 the two coincide, which is what makes the
    mistake invisible.
    """
    rows = np.asarray(rows, dtype=np.int64)
    source = np.asarray(source, dtype=np.int64)
    if rung == "two_frame":
        return np.asarray(data["encoder_embedding"], dtype=np.float64)[source]
    if rung not in LATENT_RUNGS:
        raise ValueError(f"unknown rung {rung!r}; expected one of {RUNGS}")
    latent = np.asarray(data["latent"], dtype=np.float64)
    h_dim = int(h_dim)
    if not 1 <= h_dim < latent.shape[1]:
        raise ValueError(
            f"h_dim={h_dim} does not index a {latent.shape[1]}-wide latent into a "
            "non-empty (h, z) pair"
        )
    # `latent` is `cat([h, z])` -- h FIRST, per RSSM.observe. Slicing the tail
    # for `deterministic` reads the stochastic state instead and answers a
    # different question with a shape that is often still valid.
    if rung == "deterministic":
        return latent[rows, :h_dim]
    if rung == "stochastic":
        return latent[rows, h_dim:]
    return latent[rows]
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/eval/test_retention.py -q`
Expected: PASS, 17 tests.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/eval/retention.py tests/eval/test_retention.py
git commit -m "feat: retention.py -- the pre-registered constants, the label-driven row shift, the two backward targets and the four rungs"
```

---

### Task 5: `retention.py` — Reading E, its six statuses and the finiteness guard

**Why:** the reading is the milestone. It reports the last rung on the path at which observed motion is still linearly present, and each status pre-registers a different lever (spec 3.3), so the reading and the lever choice are one act.

**Files:**
- Modify: `src/mbfps/eval/retention.py` — append
- Test: `tests/eval/test_retention.py` — append

**Interfaces:**
- Consumes: `RUNGS`, `LATENT_RUNGS`, `Z_BEARING_RUNGS`, `SEEDS_REQUIRED`, `ARMS_REQUIRED`, `K_REPORTED`, `DECISION_K` from Task 4.
- Produces, used by Tasks 6 and 7:

```python
BASE_R2_FLOOR: float = 0.10

@dataclass(frozen=True)
class RungArm:
    gain: float          # mean over this arm's seeds
    ci_low: float        # the LEAST lower bound across its seeds
    ci_high: float       # the GREATEST upper bound; reported, decides nothing
    seeds_clear: int
    seeds_total: int
    def clears(self) -> bool: ...

@dataclass(frozen=True)
class BaseControl:
    r2: float
    seeds_clear: int
    seeds_total: int
    def clears(self) -> bool: ...

@dataclass(frozen=True)
class RetentionInputs:
    ladder: dict[str, dict[int, dict[str, dict[str, RungArm]]]]  # target -> k -> rung -> arm
    base: dict[str, BaseControl]                                 # arm -> control
    clusters: int
    rows: dict[int, int]                                         # k -> scored rows

@dataclass(frozen=True)
class RetentionStatus:
    status: str
    rule: str
    surviving: str | None
    translation_rungs: tuple[str, ...]
    rotation_rungs: tuple[str, ...]
    base_failed: tuple[str, ...]

def rung_arm(gains: list[dict]) -> RungArm: ...
def rung_clears_at(inputs: RetentionInputs, target: str, rung: str) -> tuple[int, ...]: ...
def reading_retention(inputs: RetentionInputs) -> RetentionStatus: ...
```

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_retention.py`:

```python
from mbfps.eval.retention import (  # noqa: E402
    BASE_R2_FLOOR,
    BaseControl,
    RetentionInputs,
    RungArm,
    reading_retention,
    rung_arm,
    rung_clears_at,
)

ARMS = ("frozen_ssl", "pixel_ae", "random_vit")


def _arm(ci_low: float, seeds_clear: int = 3) -> RungArm:
    """A `RungArm` whose clearing is set by `ci_low` and the seed tally alone."""
    return RungArm(gain=ci_low + 0.05, ci_low=ci_low, ci_high=ci_low + 0.10,
                   seeds_clear=seeds_clear, seeds_total=3)


def _ladder(clearing: dict) -> dict:
    """`{(target, k, rung): ci_low}` -> the full nested ladder, every unnamed
    cell at ci_low -0.01 (a rung that did not clear)."""
    out = {}
    for target in TARGETS:
        out[target] = {}
        for k in K_REPORTED:
            out[target][k] = {}
            for rung in RUNGS:
                ci_low = clearing.get((target, k, rung), -0.01)
                out[target][k][rung] = {a: _arm(ci_low) for a in ARMS}
    return out


def _inputs(clearing: dict, *, base_r2: float = 0.30, base_seeds: int = 3) -> RetentionInputs:
    return RetentionInputs(
        ladder=_ladder(clearing),
        base={a: BaseControl(r2=base_r2, seeds_clear=base_seeds, seeds_total=3)
              for a in ARMS},
        clusters=24,
        rows={1: 11221, 4: 10534, 15: 8015},
    )


# --- the finiteness guard: M3i's ledger left this as a note for its successor --


def test_rung_arm_refuses_a_non_finite_gain():
    """M3i's `clears_up`, `clears_down` and `leaks` ALL evaluate False on NaN, so
    one NaN would read "no clear" and "no leak" at once -- moving a verdict
    toward the wrong status while looking like a clean null. A refusal here is
    the whole point: a non-finite gain is an error, never a non-clear."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-finite"):
            rung_arm([{"gain": bad, "ci_low": 0.1, "ci_high": 0.2}])


def test_rung_arm_refuses_a_non_finite_interval_bound():
    with pytest.raises(ValueError, match="non-finite"):
        rung_arm([{"gain": 0.1, "ci_low": float("nan"), "ci_high": 0.2}])
    with pytest.raises(ValueError, match="non-finite"):
        rung_arm([{"gain": 0.1, "ci_low": 0.0, "ci_high": float("inf")}])


def test_rung_arm_refuses_an_empty_seed_list():
    with pytest.raises(ValueError, match="at least one seed"):
        rung_arm([])


def test_rung_arm_reports_the_seed_mean_and_the_least_lower_bound():
    """The gain is a MEAN so the printed number describes the arm; `ci_low` is
    the LEAST of the seeds' bounds, which is the conservative summary -- a rung
    is not credited for an interval only its luckiest seed achieved. The tally
    is what decides, and it counts seeds whose OWN bound cleared zero."""
    arm = rung_arm([
        {"gain": 0.30, "ci_low": 0.10, "ci_high": 0.50},
        {"gain": 0.20, "ci_low": -0.05, "ci_high": 0.45},
        {"gain": 0.10, "ci_low": 0.02, "ci_high": 0.18},
    ])
    assert arm.gain == pytest.approx(0.20)
    assert arm.ci_low == pytest.approx(-0.05)
    assert arm.ci_high == pytest.approx(0.50), "the GREATEST upper bound, not the mean"
    assert arm.seeds_clear == 2 and arm.seeds_total == 3
    assert arm.clears() is True, "2 of 3 seeds is SEEDS_REQUIRED"


def test_rung_arm_clearing_is_one_sided():
    """A NEGATIVE gain means appending the block made held-out R^2 worse: noise
    or selection slack, not a finding. M3i's control was two-sided on an
    argument its own results refuted, and the lesson is applied here rather than
    re-derived. A seed whose whole interval sits below zero must not clear."""
    arm = rung_arm([
        {"gain": -0.40, "ci_low": -0.60, "ci_high": -0.20},
        {"gain": -0.35, "ci_low": -0.55, "ci_high": -0.15},
        {"gain": -0.30, "ci_low": -0.50, "ci_high": -0.10},
    ])
    assert arm.seeds_clear == 0 and arm.clears() is False


def test_rung_arm_does_not_clear_on_an_interval_that_only_touches_zero():
    """`ci_low > 0`, strictly. An interval whose lower bound is exactly zero has
    not excluded zero."""
    arm = rung_arm([{"gain": 0.1, "ci_low": 0.0, "ci_high": 0.2}] * 3)
    assert arm.seeds_clear == 0 and arm.clears() is False


# --- the ladder ------------------------------------------------------------


def test_rung_clears_at_reports_every_horizon_the_rung_cleared():
    """The rule is a DISJUNCTION over `K_REPORTED` (spec 2.3), so the reading
    needs the horizons, not just a boolean -- they go into the rule text."""
    inputs = _inputs({("translation", 1, "full"): 0.2,
                      ("translation", 15, "full"): 0.3})
    assert rung_clears_at(inputs, "translation", "full") == (1, 15)
    assert rung_clears_at(inputs, "translation", "two_frame") == ()


def test_rung_clears_at_needs_arms_required_arms():
    """One arm clearing is not the rule. Two of three is."""
    inputs = _inputs({})
    inputs.ladder["translation"][4]["full"]["pixel_ae"] = _arm(0.5)
    assert rung_clears_at(inputs, "translation", "full") == ()
    inputs.ladder["translation"][4]["full"]["frozen_ssl"] = _arm(0.5)
    assert rung_clears_at(inputs, "translation", "full") == (4,)


def test_rung_clears_at_needs_seeds_required_seeds_in_each_arm():
    """An arm whose pooled bound clears on ONE seed has not replicated."""
    inputs = _inputs({})
    for arm in ("pixel_ae", "frozen_ssl"):
        inputs.ladder["translation"][4]["full"][arm] = _arm(0.5, seeds_clear=1)
    assert rung_clears_at(inputs, "translation", "full") == ()


# --- the six statuses, one test each ---------------------------------------


def test_reading_is_unresolved_base_when_the_current_frame_cannot_locate_itself():
    """The one true control failure, and it outranks every result. If `enc(t)`
    cannot linearly say where it is, the instrument is broken and a null on
    displacement means nothing."""
    inputs = _inputs({("translation", 4, "full"): 0.5}, base_r2=0.01)
    reading = reading_retention(inputs)
    assert reading.status == "UNRESOLVED_BASE"
    assert reading.base_failed == ARMS
    assert reading.surviving is None
    assert reading.translation_rungs == (), (
        "a suppressed reading must report no rungs at all, not report them "
        "beside a warning nobody reads"
    )


def test_reading_is_motion_retained_when_a_z_bearing_rung_clears():
    inputs = _inputs({("translation", 15, "stochastic"): 0.2})
    reading = reading_retention(inputs)
    assert reading.status == "MOTION_RETAINED"
    assert reading.surviving == "stochastic"
    assert "15" in reading.rule


def test_reading_is_motion_retained_even_when_two_frame_is_silent():
    """`h` integrates the ACTION sequence, which two frames do not contain, so a
    latent rung can clear where `two_frame` does not. An earlier draft of the
    spec refused in exactly this case by putting UNRESOLVED_MOTION second; this
    is the test that would have caught it."""
    inputs = _inputs({("translation", 4, "full"): 0.3})
    reading = reading_retention(inputs)
    assert reading.status == "MOTION_RETAINED"
    assert reading.translation_rungs == ("full",)


def test_reading_is_bottleneck_loss_when_only_the_deterministic_rung_clears():
    """`h` carries it and `z` destroys it: the lever is the 32x32 bottleneck and
    `rep_scale`, not the objective."""
    inputs = _inputs({("translation", 4, "deterministic"): 0.2,
                      ("translation", 4, "two_frame"): 0.4})
    reading = reading_retention(inputs)
    assert reading.status == "BOTTLENECK_LOSS"
    assert reading.surviving == "deterministic"


def test_reading_is_motion_discarded_when_only_two_frame_clears():
    """The information is available and the RSSM throws it away: the lever is
    the objective, which never asks for motion."""
    inputs = _inputs({("translation", 4, "two_frame"): 0.4,
                      ("rotation", 4, "two_frame"): 0.5})
    reading = reading_retention(inputs)
    assert reading.status == "MOTION_DISCARDED"
    assert reading.surviving == "two_frame"


def test_reading_is_translation_unresolved_when_only_rotation_reads():
    """Nothing reads translation but something reads rotation: the lever is the
    encoder's INPUT -- resolution, stride, frame stacking -- not the loss."""
    inputs = _inputs({("rotation", 4, "two_frame"): 0.5,
                      ("rotation", 15, "full"): 0.3})
    reading = reading_retention(inputs)
    assert reading.status == "TRANSLATION_UNRESOLVED"
    assert reading.surviving is None
    assert reading.rotation_rungs == ("two_frame", "full")


def test_reading_is_unresolved_motion_when_nothing_clears_anything():
    """No rung reads either target: the measurement detects no motion anywhere,
    so no lever is chosen and the next milestone is about the instrument. This
    requires the WHOLE ladder to be silent on BOTH targets, not `two_frame`
    alone."""
    reading = reading_retention(_inputs({}))
    assert reading.status == "UNRESOLVED_MOTION"
    assert reading.surviving is None
    assert reading.translation_rungs == () and reading.rotation_rungs == ()


def test_the_six_statuses_are_exhaustive_and_mutually_exclusive():
    """Every combination of (which rungs clear translation, which clear
    rotation, does the base hold) must land on exactly one status, and all six
    must be reachable. M3i shipped a status nothing exercised until the final
    review; this is the cheap version of that check."""
    import itertools

    seen = set()
    for base_ok in (True, False):
        for t_rungs in itertools.chain.from_iterable(
            itertools.combinations(RUNGS, n) for n in range(len(RUNGS) + 1)
        ):
            for r_rungs in ((), ("two_frame",), ("full",)):
                clearing = {("translation", 4, r): 0.2 for r in t_rungs}
                clearing.update({("rotation", 4, r): 0.2 for r in r_rungs})
                reading = reading_retention(
                    _inputs(clearing, base_r2=0.30 if base_ok else 0.0)
                )
                seen.add(reading.status)
    expected = {
        "UNRESOLVED_BASE", "MOTION_RETAINED", "BOTTLENECK_LOSS",
        "MOTION_DISCARDED", "TRANSLATION_UNRESOLVED", "UNRESOLVED_MOTION",
    }
    assert seen == expected, (
        f"unreachable: {sorted(expected - seen)}; unexpected: {sorted(seen - expected)}"
    )


def test_base_control_floor_is_below_every_recorded_latent_selection_r2():
    """`BASE_R2_FLOOR` must fail only when the instrument is broken, not when a
    cell is merely weak. The M3c records' `latent_selection_r2` runs 0.18-0.34
    (one outlier at -0.008 on the LATENT, not on the 2048-d encoder output), and
    M3i measured `enc(t)` -> position at +0.291 and +0.165 from 132 rows. The
    floor sits below all of those and well above zero."""
    assert 0.0 < BASE_R2_FLOOR < 0.165
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_retention.py -q`
Expected: collection error — `ImportError: cannot import name 'BASE_R2_FLOOR'`.

- [ ] **Step 3: Append the reading to `retention.py`**

```python
BASE_R2_FLOOR: float = 0.10
"""The r2 `enc(t)` -> absolute position must exceed for the reading to be taken.

A POSITIVE CONTROL promoted to a gate. M3i's equivalent lived only in its final
review and in no record; its own section 9 provenance note asked a successor to
build it into the measure phase, which is what this is.

The value has to fail only when the instrument is broken, never when a cell is
merely weak. The M3c records' `latent_selection_r2` runs 0.18-0.34, with one
outlier at -0.008 -- and that outlier is on the LATENT, a 32x32 bottleneck,
not on the 2048 continuous floats this control probes. M3i measured `enc(t)` ->
position at +0.291 and +0.165 from 132 rows; this milestone fits ~10,500. So
0.10 sits below every recorded figure and far above zero.
"""


@dataclass(frozen=True)
class RungArm:
    """One arm's rung at one target and horizon, summarised over its seeds.

    `gain` is the seed MEAN, so the printed number describes the arm. `ci_low`
    is the LEAST lower bound across the seeds -- the conservative summary, so a
    rung is never credited with an interval only its luckiest seed achieved, and
    `ci_high` is the GREATEST upper bound so the printed pair is a real interval
    rather than a one-sided stub. None of the three decides anything:
    `seeds_clear` does, and it counts seeds whose OWN lower bound excluded zero.
    `ci_high` is reported only -- the rule is one-sided, so nothing reads it.
    """

    gain: float
    ci_low: float
    ci_high: float
    seeds_clear: int
    seeds_total: int

    def clears(self) -> bool:
        """ONE-SIDED, and that is deliberate.

        A negative gain means appending the block made held-out R^2 WORSE, which
        is noise or selection slack rather than a finding about the
        representation. M3i shipped a two-sided control on the argument that a
        result reliably worse than chance is as broken an instrument as one
        reliably better; its own run refuted that (`## Task 8 results` section 5
        of `docs/superpowers/plans/2026-09-26-mb-fps-m3i-latent-motion.md`), and
        the lesson is applied here rather than re-derived.
        """
        return self.seeds_clear >= SEEDS_REQUIRED


@dataclass(frozen=True)
class BaseControl:
    """One arm's base control: `enc(t)` -> absolute position, as an r2 LEVEL.

    Not a gain. There is nothing to take an increment over -- this is the arm
    the increments are measured against, and the question is only whether it
    reads at all.
    """

    r2: float
    seeds_clear: int
    seeds_total: int

    def clears(self) -> bool:
        return self.seeds_clear >= SEEDS_REQUIRED


@dataclass(frozen=True)
class RetentionInputs:
    """Everything Reading E is decided on.

    `ladder` is `target -> k -> rung -> arm -> RungArm`, four levels because
    every one of them is quantified over in the rule: a rung clears a target if
    it clears at ANY k, in ARMS_REQUIRED arms, each in SEEDS_REQUIRED seeds.
    `rows` carries the scored row count per k so the reading prints the
    conditioning it was taken at rather than leaving it to be recomputed.
    """

    ladder: dict[str, dict[int, dict[str, dict[str, RungArm]]]]
    base: dict[str, BaseControl]
    clusters: int
    rows: dict[int, int]


@dataclass(frozen=True)
class RetentionStatus:
    status: str
    rule: str
    surviving: str | None
    translation_rungs: tuple[str, ...]
    rotation_rungs: tuple[str, ...]
    base_failed: tuple[str, ...]


def rung_arm(gains: list[dict]) -> RungArm:
    """One arm's `RungArm` from its per-seed `probe.gain_from_blocks` dicts.

    THE FINITENESS GUARD LIVES HERE, and it raises. M3i's ledger left this as
    the one note for its successor: `clears_up`, `clears_down` and `leaks` all
    evaluate False on NaN, so a single non-finite cell would read "no clear" and
    "no leak" at once -- moving a verdict toward the wrong status while looking
    like a clean null. A non-finite gain or interval bound is an ERROR about the
    measurement, never a statement about the representation, so it is refused
    at the point where the number first becomes a reading.
    """
    if not gains:
        raise ValueError("a rung needs at least one seed to summarise")
    for index, seed in enumerate(gains):
        values = {k: float(seed[k]) for k in ("gain", "ci_low", "ci_high")}
        bad = {k: v for k, v in values.items() if not np.isfinite(v)}
        if bad:
            raise ValueError(
                f"non-finite {', '.join(sorted(bad))} in seed index {index}: {bad}. "
                "A non-finite gain is an error about the measurement, not a "
                "non-clear -- see this function's docstring."
            )
    return RungArm(
        gain=float(np.mean([g["gain"] for g in gains])),
        ci_low=float(min(float(g["ci_low"]) for g in gains)),
        ci_high=float(max(float(g["ci_high"]) for g in gains)),
        seeds_clear=sum(1 for g in gains if float(g["ci_low"]) > 0.0),
        seeds_total=len(gains),
    )


def rung_clears_at(inputs: RetentionInputs, target: str, rung: str) -> tuple[int, ...]:
    """The horizons at which `rung` clears `target`, in `K_REPORTED` order.

    A tuple rather than a boolean because the rule is a DISJUNCTION over
    `K_REPORTED` (spec 2.3) and the reading's rule text names the horizons it
    cleared at. Empty means the rung did not clear anywhere.
    """
    cleared = []
    for k in K_REPORTED:
        arms = inputs.ladder[target][k][rung]
        if sum(1 for arm in arms.values() if arm.clears()) >= ARMS_REQUIRED:
            cleared.append(k)
    return tuple(cleared)


def _cleared_rungs(inputs: RetentionInputs, target: str) -> dict[str, tuple[int, ...]]:
    """Every rung that cleared `target`, mapped to its horizons, in RUNGS order."""
    found = {}
    for rung in RUNGS:
        at = rung_clears_at(inputs, target, rung)
        if at:
            found[rung] = at
    return found


def reading_retention(inputs: RetentionInputs) -> RetentionStatus:
    """Reading E: the last rung on the path at which observed motion survives.

    Precedence, and it is the point (spec 3.2). `UNRESOLVED_BASE` is the one
    true control failure and outranks every result -- a null on displacement
    means nothing if the current frame cannot say where it is. Then the ladder,
    highest surviving rung first, so the status IS the lever: a z-bearing rung
    means the bottleneck kept it and M3i is partially overturned; `h` alone
    means the bottleneck destroyed it; `two_frame` alone means the RSSM
    discarded available information; nothing on translation but something on
    rotation means translation is below the encoder's spatial resolution.

    THE LATENT RUNGS ARE CHECKED BEFORE `two_frame`, and `UNRESOLVED_MOTION` IS
    LAST. `h` integrates the action sequence, which two frames do not contain,
    so a latent rung can legitimately clear where `two_frame` does not. An
    earlier draft of the spec put `UNRESOLVED_MOTION` second, refusing whenever
    `two_frame` was silent on both targets -- which would have refused precisely
    that case. `UNRESOLVED_MOTION` therefore requires the WHOLE ladder to be
    silent on BOTH targets, and it is not a control failure but an empty
    measurement, which is why it sorts with the readings rather than ahead of
    them.

    A suppressed reading reports no rungs at all rather than reporting them
    beside a warning nobody reads.
    """
    base_failed = tuple(
        sorted(arm for arm, control in inputs.base.items() if not control.clears())
    )
    holding = len(inputs.base) - len(base_failed)
    if holding < ARMS_REQUIRED:
        return RetentionStatus(
            status="UNRESOLVED_BASE",
            rule=(
                f"enc(t) -> absolute position cleared r2 {BASE_R2_FLOOR:.2f} in only "
                f"{holding} of {len(inputs.base)} arms ({', '.join(base_failed)} failed); "
                f"the current frame cannot linearly say where it is, so the instrument "
                f"is broken and no reading is taken"
            ),
            surviving=None, translation_rungs=(), rotation_rungs=(), base_failed=base_failed,
        )

    translation = _cleared_rungs(inputs, "translation")
    rotation = _cleared_rungs(inputs, "rotation")
    t_rungs = tuple(translation)
    r_rungs = tuple(rotation)

    def at(rung: str) -> str:
        return ", ".join(f"k = {k}" for k in translation[rung])

    for rung in Z_BEARING_RUNGS:
        if rung in translation:
            return RetentionStatus(
                status="MOTION_RETAINED",
                rule=(
                    f"the {rung} rung adds displacement the current frame lacks at "
                    f"{at(rung)}, in at least {ARMS_REQUIRED} arms and {SEEDS_REQUIRED} "
                    f"seeds each; the bottlenecked latent retains motion it has observed, "
                    f"so M3i's NO_MOTION was about forward prediction rather than about "
                    f"the representation's content"
                ),
                surviving=rung, translation_rungs=t_rungs, rotation_rungs=r_rungs,
                base_failed=(),
            )
    if "deterministic" in translation:
        return RetentionStatus(
            status="BOTTLENECK_LOSS",
            rule=(
                f"the deterministic rung adds displacement at {at('deterministic')} but "
                f"no z-bearing rung does; h carries motion and the 32x32 categorical "
                f"bottleneck destroys it"
            ),
            surviving="deterministic", translation_rungs=t_rungs, rotation_rungs=r_rungs,
            base_failed=(),
        )
    if "two_frame" in translation:
        return RetentionStatus(
            status="MOTION_DISCARDED",
            rule=(
                f"two real frames add displacement at {at('two_frame')} and no part of "
                f"the latent does; the information is available and the RSSM discards it, "
                f"which is what a loss whose reconstruction target is the frame just seen "
                f"would predict"
            ),
            surviving="two_frame", translation_rungs=t_rungs, rotation_rungs=r_rungs,
            base_failed=(),
        )
    if rotation:
        return RetentionStatus(
            status="TRANSLATION_UNRESOLVED",
            rule=(
                f"no rung adds translation at any of k = "
                f"{', '.join(str(k) for k in K_REPORTED)}, but "
                f"{', '.join(r_rungs)} adds ROTATION; translation is below what "
                f"112x112 frozen single-frame features resolve, so the lever is the "
                f"encoder's input rather than the loss"
            ),
            surviving=None, translation_rungs=t_rungs, rotation_rungs=r_rungs,
            base_failed=(),
        )
    return RetentionStatus(
        status="UNRESOLVED_MOTION",
        rule=(
            f"no rung adds either translation or rotation at any of k = "
            f"{', '.join(str(k) for k in K_REPORTED)}; a linear read detects no motion "
            f"anywhere on the path, so the measurement is empty and no lever is chosen"
        ),
        surviving=None, translation_rungs=(), rotation_rungs=(), base_failed=(),
    )
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/eval/test_retention.py -q`
Expected: PASS.

- [ ] **Step 5: Run one mutation by hand to confirm the tests bite**

Temporarily change `Z_BEARING_RUNGS` to `("full",)` in
`src/mbfps/eval/retention.py` and run:

`.venv/bin/python -m pytest tests/eval/test_retention.py -q`

Expected: FAIL — `test_reading_is_motion_retained_when_a_z_bearing_rung_clears`
(it reads `MOTION_RETAINED` off `stochastic`) and
`test_latent_rungs_and_z_bearing_rungs_are_subsets_of_rungs`. Revert the change
and confirm the suite passes again. M3i had two tasks whose tests could not fail
because the fixtures were symmetric under exactly the mutation that mattered;
one hand-run mutation per reading task is the cheapest guard against that.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/retention.py tests/eval/test_retention.py
git commit -m "feat: Reading E -- six statuses whose precedence IS the lever choice, and the finiteness guard M3i left as a note"
```

---

### Task 6: `retention.py` — the printed tables, captions pinned to their columns

**Why:** this project has shipped a table whose caption did not match its columns **three times** (M3h's Reading N printed the up tally under a verdict read from the down one; M3i fixed two more). The columns are declared once as data so a test can assert the header and every row against the same tuple.

**Files:**
- Modify: `src/mbfps/eval/retention.py` — append
- Test: `tests/eval/test_retention.py` — append

**Interfaces:**
- Consumes: `RetentionInputs`, `RetentionStatus`, `RungArm` from Task 5.
- Produces, used by Task 8:

```python
READING_COLUMNS: tuple[str, ...] = ("rung", "arm", "gain", "ci_low", "ci_high", "seeds", "clears")
READING_WIDTHS: tuple[int, ...] = (15, 13, 10, 10, 10, 8, 8)
LADDER_COLUMNS: tuple[str, ...] = ("target", "rung", "k", "mean gain", "arms", "seeds", "clears")
LADDER_WIDTHS: tuple[int, ...] = (13, 15, 6, 12, 8, 8, 8)

def format_reading_retention(reading: RetentionStatus, inputs: RetentionInputs) -> str: ...
def format_ladder(inputs: RetentionInputs) -> str: ...
```

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_retention.py`:

```python
from mbfps.eval.retention import (  # noqa: E402
    LADDER_COLUMNS,
    LADDER_WIDTHS,
    READING_COLUMNS,
    READING_WIDTHS,
    format_ladder,
    format_reading_retention,
)


def test_reading_columns_and_widths_stay_the_same_length():
    """Header and rows are two separate f-strings built from these tuples. A
    length mismatch means one column's caption sits over another's values --
    the defect class this project has shipped three times."""
    assert len(READING_COLUMNS) == len(READING_WIDTHS)
    assert len(LADDER_COLUMNS) == len(LADDER_WIDTHS)


def test_every_reading_width_admits_its_widest_realistic_value():
    """A value as wide as its field glues onto the previous column with no
    separator. `seeds` holds "3/3" (3 chars), `clears` holds "yes"/"no", `arm`
    holds "frozen_ssl" (10) and "random_vit" (10), and a gain prints as
    "+0.1234" or "-12.3456" (8). Every width must EXCEED, not equal."""
    widest = {
        "rung": len("deterministic"), "arm": len("frozen_ssl"),
        "gain": len("-12.3456"), "ci_low": len("-12.3456"),
        "ci_high": len("-12.3456"), "seeds": len("3/3"), "clears": len("yes"),
    }
    for name, width in zip(READING_COLUMNS, READING_WIDTHS, strict=True):
        assert width > widest[name], (
            f"column {name!r} is {width} wide but holds up to {widest[name]} "
            "characters; a full-width value glues onto its left neighbour"
        )


def test_reading_header_captions_the_columns_it_prints():
    """The header and the rows must agree COLUMN BY COLUMN, sliced at the same
    offsets. Asserting the two strings look plausible is what let three wrong
    captions ship."""
    inputs = _inputs({("translation", DECISION_K, "full"): 0.2})
    text = format_reading_retention(reading_retention(inputs), inputs)
    lines = [line for line in text.splitlines() if line.startswith("  ")]
    header, first = lines[0], lines[1]
    offset = 2
    for name, width in zip(READING_COLUMNS, READING_WIDTHS, strict=True):
        assert header[offset:offset + width].strip() == name, (
            f"header column at offset {offset} is not {name!r}"
        )
        assert first[offset:offset + width].strip() != "", (
            f"the first row has nothing under the {name!r} caption"
        )
        offset += width


def test_reading_names_the_target_the_horizon_and_the_clusters_in_its_caption():
    """A table whose caption does not say what its numbers are is how this
    project shipped a wrong number three times. The caption has to name the
    statistic (a gain over enc(t), not a level), the target, the horizon, the
    one-sided rule and the cluster count."""
    inputs = _inputs({})
    caption = format_reading_retention(reading_retention(inputs), inputs).splitlines()[0]
    assert "gain over enc(t)" in caption
    assert "translation" in caption
    assert f"k = {DECISION_K}" in caption
    assert "ci_low > 0" in caption
    assert "24 clusters" in caption


def test_reading_prints_the_verdict_and_its_rule():
    inputs = _inputs({("translation", 4, "deterministic"): 0.2})
    text = format_reading_retention(reading_retention(inputs), inputs)
    assert "verdict: BOTTLENECK LOSS" in text
    assert "decided by:" in text


def test_reading_prints_the_base_control_beside_the_verdict():
    """The control is a gate, so its numbers belong next to the reading it
    licensed rather than in a companion nobody reads."""
    inputs = _inputs({})
    text = format_reading_retention(reading_retention(inputs), inputs)
    assert "base control (enc(t) -> absolute position" in text
    assert "frozen_ssl r2=+0.300" in text


def test_reading_prints_the_rotation_control_beside_the_verdict():
    """Reading rotation but not translation is a DIFFERENT lever from reading
    neither, so which rungs read rotation has to be on the face of the verdict."""
    inputs = _inputs({("rotation", 4, "two_frame"): 0.5})
    text = format_reading_retention(reading_retention(inputs), inputs)
    assert "rotation control" in text
    assert "two_frame" in text.split("rotation control")[1].splitlines()[0]


def test_ladder_prints_every_rung_at_every_horizon_for_both_targets():
    """`K_REPORTED` x `RUNGS` x `TARGETS` rows, all of them, because the rule is
    a disjunction over k and a reader has to be able to check it."""
    inputs = _inputs({("translation", 15, "full"): 0.3})
    rows = [
        line for line in format_ladder(inputs).splitlines()
        if line.startswith("  ") and "target" not in line
    ]
    assert len(rows) == len(TARGETS) * len(RUNGS) * len(K_REPORTED)


def test_ladder_header_captions_the_columns_it_prints():
    inputs = _inputs({})
    lines = [line for line in format_ladder(inputs).splitlines() if line.startswith("  ")]
    header, first = lines[0], lines[1]
    offset = 2
    for name, width in zip(LADDER_COLUMNS, LADDER_WIDTHS, strict=True):
        assert header[offset:offset + width].strip() == name
        assert first[offset:offset + width].strip() != ""
        offset += width


def test_ladder_says_it_decides_nothing_on_its_own():
    """Every horizon contributes to the disjunction, so no row "decides" alone.
    M3i printed `decides: no` beside every companion row for the opposite reason
    -- one horizon decided and the rest did not -- and a reader carrying that
    habit across would misread this table without the caption."""
    caption = format_ladder(_inputs({})).splitlines()[0]
    assert "any horizon counts" in caption
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_retention.py -k "reading_columns or reading_header or reading_names or reading_prints or ladder or widest" -q`
Expected: FAIL, `ImportError: cannot import name 'READING_COLUMNS'`.

- [ ] **Step 3: Append the formatters**

```python
# The reading table's columns, in printed order. Declared once so a test can
# assert the header against THIS and the rows against the same offsets -- the
# caption-column pairing that three shipped defects on this project all broke.
READING_COLUMNS: tuple[str, ...] = (
    "rung", "arm", "gain", "ci_low", "ci_high", "seeds", "clears",
)

# Field widths for READING_COLUMNS, in the same order. Kept beside the names so
# the header and the rows -- two separate f-strings -- cannot drift apart, and
# sized so a realistic value cannot EQUAL its width and glue onto the previous
# column with no separator (the arm-name-overflow defect this project shipped
# once already: "random_vit" is 10 characters, so its field is 13).
READING_WIDTHS: tuple[int, ...] = (15, 13, 10, 10, 10, 8, 8)

LADDER_COLUMNS: tuple[str, ...] = (
    "target", "rung", "k", "mean gain", "arms", "seeds", "clears",
)
LADDER_WIDTHS: tuple[int, ...] = (13, 15, 6, 12, 8, 8, 8)


def _row(values, widths) -> str:
    return "  " + "".join(
        f"{value:>{width}}" for value, width in zip(values, widths, strict=True)
    )


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def _seed_arms(arms: dict[str, RungArm]) -> tuple[int, int]:
    """`(arms clearing, seeds clearing across all of them)`."""
    return (
        sum(1 for arm in arms.values() if arm.clears()),
        sum(arm.seeds_clear for arm in arms.values()),
    )


def format_reading_retention(reading: RetentionStatus, inputs: RetentionInputs) -> str:
    """Reading E as it is printed and written to `retention.txt`, byte for byte.

    The caption names the statistic, the target, the horizon, the one-sided rule
    and the cluster count, because a table whose header does not say what its
    columns hold is how this project has shipped a wrong number three times.
    The two controls print beside the verdict rather than in a companion table:
    the base control is a GATE, and whether rotation read is a different LEVER
    from whether nothing read, so both belong on the face of the reading.
    """
    k = DECISION_K
    rows = inputs.ladder["translation"][k]
    lines = [
        f"--- Reading E: what each rung adds over enc(t) about translation "
        f"already observed, at k = {k} "
        f"(gain over enc(t) in mean R^2, one-sided: clears when ci_low > 0 in "
        f"{SEEDS_REQUIRED} of 3 seeds and {ARMS_REQUIRED} of 3 arms); "
        f"{inputs.rows.get(k, 0)} rows over {inputs.clusters} clusters ---",
        _row(READING_COLUMNS, READING_WIDTHS),
    ]
    for rung in RUNGS:
        for arm_name in sorted(rows[rung]):
            arm = rows[rung][arm_name]
            lines.append(_row((
                rung, arm_name,
                f"{arm.gain:+.4f}", f"{arm.ci_low:+.4f}", f"{arm.ci_high:+.4f}",
                f"{arm.seeds_clear}/{arm.seeds_total}", _yes(arm.clears()),
            ), READING_WIDTHS))
    base = "  base control (enc(t) -> absolute position, must clear r2 " + (
        f"{BASE_R2_FLOOR:.2f}): "
    ) + ", ".join(
        f"{name} r2={inputs.base[name].r2:+.3f} "
        f"{inputs.base[name].seeds_clear}/{inputs.base[name].seeds_total}"
        for name in sorted(inputs.base)
    )
    rotation = "  rotation control (positive; reads where translation cannot): " + (
        ", ".join(reading.rotation_rungs) if reading.rotation_rungs
        else "no rung cleared rotation at any horizon"
    )
    lines += [
        base,
        rotation,
        f"  verdict: {reading.status.replace('_', ' ')} -- decided by: {reading.rule}",
    ]
    return "\n".join(lines)


def format_ladder(inputs: RetentionInputs) -> str:
    """Every rung at every horizon for both targets -- the disjunction, in full.

    Reported so a reader can check the rule rather than take it on trust. No row
    decides on its own and the caption says so: M3i's companion table printed
    `decides: no` beside every row because exactly one horizon DID decide there,
    and a reader carrying that habit across would misread this one.
    """
    lines = [
        f"--- The ladder: gain over enc(t) at every horizon, both targets "
        f"(translation decides, rotation controls; any horizon counts -- the "
        f"rule is a disjunction over k = "
        f"{', '.join(str(k) for k in K_REPORTED)}) ---",
        _row(LADDER_COLUMNS, LADDER_WIDTHS),
    ]
    for target in TARGETS:
        for rung in RUNGS:
            for k in K_REPORTED:
                arms = inputs.ladder[target][k][rung]
                clearing, seeds = _seed_arms(arms)
                lines.append(_row((
                    target, rung, str(k),
                    f"{np.mean([a.gain for a in arms.values()]):+.4f}",
                    f"{clearing}/{len(arms)}",
                    f"{seeds}/{sum(a.seeds_total for a in arms.values())}",
                    _yes(clearing >= ARMS_REQUIRED),
                ), LADDER_WIDTHS))
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/eval/test_retention.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/eval/retention.py tests/eval/test_retention.py
git commit -m "feat: Reading E's tables, with both captions pinned to the columns they caption"
```

---

### Task 7: `scripts/latent_retention.py` — the measure phase

**Why:** the only impure half. It owns torch, the three gathers per cell, the shared bases, the record schema and every refusal that comes before a pass.

**Files:**
- Create: `scripts/latent_retention.py`
- Test: `tests/eval/test_latent_retention_script.py`

**Interfaces:**
- Consumes: `probe.gather_probe_data` (with `"episode"`, Task 1), `probe.GainSplit` / `probe.gain_from_blocks` (Task 3), and every name Tasks 4–6 produced.
- Consumes from `scripts/trust_horizon.py`, loaded by path exactly as `scripts/latent_motion.py` does: `Cell`, `CellMissing`, `load_cell`, `self_check`, `prepare_cell`, `EXIT_OK`, `EXIT_NO_CHECKPOINTS`, `EXIT_SPLIT_MISMATCH`, `EXIT_RECORD_MISMATCH`, `EXIT_SELF_CHECK_FAILED`.
- Produces, used by Task 8:

```python
EXIT_BASE_UNRESOLVED = 39
EXIT_MOTION_UNRESOLVED = 40
PHASES: tuple[str, ...] = ("measure", "read", "all")
FIT_EPISODES: int = 20
SELECT_EPISODES: int = 20

def retention_record_path(out_dir: Path, arm: str, seed: int) -> Path: ...
def k_key(k: int) -> str: ...
def gather_three_splits(prepared, train, val, *, seed) -> tuple[dict, dict | None, dict]: ...
def cell_ladder(fit, select, score, *, h_dim, ks=K_REPORTED) -> dict: ...
def base_control(fit, select, score) -> dict: ...
def measure_cell(args, cell, device, train, val, ks=K_REPORTED) -> tuple[int, dict | None]: ...
def measure_phase(args, cells, device, train, val, ks=K_REPORTED) -> int: ...
```

`cell_ladder` returns `{target: {k_key(k): {rung: gain_dict}}}` and a
`"rows"` map; `base_control` returns one `gain_from_blocks`-shaped dict whose
`joint_r2` is unused and whose `embedding_r2` is the base level.

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_latent_retention_script.py`. The module is loaded by
path, the way every script test in this suite loads one:

```python
"""M3j's script: the three gathers, the shared bases, and the record schema."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mbfps.eval.retention import K_REPORTED, RUNGS, TARGETS

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "latent_retention.py"


def _load():
    spec = importlib.util.spec_from_file_location("latent_retention_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load()


def _gathered(n_windows: int = 8, steps: int = 20, width: int = 6, h_dim: int = 3,
              seed: int = 0) -> dict:
    """A `gather_probe_data`-shaped dict with REAL structure: the deterministic
    half of the latent remembers the previous frame's position, so the
    deterministic rung must show a gain and a flat rung must not."""
    rng = np.random.default_rng(seed)
    rows = n_windows * steps
    window = np.repeat(np.arange(n_windows), steps)
    step = np.tile(np.arange(steps), n_windows)
    episode = window // 2
    pos = np.column_stack([
        window * 100.0 + step * 3.0, window * 50.0 - step * 2.0,
    ])
    radians = np.deg2rad((step * 7.0 + window * 11.0) % 360.0)
    targets = np.column_stack([pos, np.sin(radians), np.cos(radians)])
    enc = np.column_stack([pos, rng.normal(size=(rows, width - 2))])
    lagged = np.roll(pos, 1, axis=0)
    latent = np.column_stack([
        lagged, rng.normal(size=(rows, h_dim - 2)), rng.normal(size=(rows, 4)),
    ])
    return {
        "latent": latent, "embedding": enc.copy(), "encoder_embedding": enc,
        "targets": targets, "window": window, "step": step, "episode": episode,
    }


def test_exit_codes_are_39_and_40_and_do_not_collide_with_m3i():
    """M3i holds 38 (`EXIT_CONTROL_LEAKED`). A reused number makes two different
    refusals indistinguishable to a caller reading `$?`."""
    assert script.EXIT_BASE_UNRESOLVED == 39
    assert script.EXIT_MOTION_UNRESOLVED == 40
    assert script.EXIT_OK == 0


def test_phases_are_measure_read_all():
    assert script.PHASES == ("measure", "read", "all")


def test_episode_budget_is_the_filtering_gain_defaults():
    """`select_episodes` at 20, not 4. `probe.filtering_gain`'s docstring records
    that a 4-episode selection split picks the wrong ridge and FLIPS THE SIGN of
    the gain (+0.0325 against -0.0208 on the same checkpoint). Shrinking this to
    save a gather does not make the number noisier, it makes it wrong."""
    assert script.FIT_EPISODES == 20
    assert script.SELECT_EPISODES == 20


def test_record_path_names_both_the_arm_and_the_seed(tmp_path):
    """M3i's task reports collided because two milestones wrote the same
    filename; a record named by only one of (arm, seed) collides nine ways."""
    path = script.retention_record_path(tmp_path, "pixel_ae", 2)
    assert path.name == "retention_pixel_ae_seed2.json"


def test_k_key_carries_no_dot():
    """`write_record` addresses non-finite fields by dotted path, so a key with
    a '.' in it would be unaddressable."""
    assert script.k_key(15) == "k15"
    assert "." not in script.k_key(4)


def test_cell_ladder_covers_every_target_rung_and_horizon():
    """The rule is a disjunction over k across four rungs and two targets. A
    ladder missing a cell would silently make the disjunction narrower."""
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    ladder = script.cell_ladder(fit, select, score, h_dim=3, ks=(1, 2))
    assert set(ladder["ladder"]) == set(TARGETS)
    for target in TARGETS:
        assert set(ladder["ladder"][target]) == {"k1", "k2"}
        for k in ("k1", "k2"):
            assert set(ladder["ladder"][target][k]) == set(RUNGS)


def test_cell_ladder_hands_every_rung_a_byte_identical_base(monkeypatch):
    """Spec 2.1: the four gains at one `(target, k)` must be differences against
    the SAME base, or they are not comparable to each other -- only each to its
    own fit. `fit_probe` is deterministic, so identical base arrays and an
    identical target give an identical base level; the property to pin is
    therefore that the ARRAYS are identical, which four independent row
    selections would silently break while still producing four plausible gains.

    Both targets share a row set at a given k (Task 4 pins that), so the base is
    byte-identical across targets too and one distinct array is the correct
    count, not two."""
    calls = []
    real = script.gain_from_blocks

    def counting(fit, select, score, **kwargs):
        calls.append(np.asarray(fit.base).tobytes())
        return real(fit, select, score, **kwargs)

    monkeypatch.setattr(script, "gain_from_blocks", counting)
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    script.cell_ladder(fit, select, score, h_dim=3, ks=(1,))
    assert len(calls) == len(TARGETS) * len(RUNGS), "one call per (target, rung)"
    assert len(set(calls)) == 1, (
        "the rungs were not handed one identical base array; their gains are "
        "against different bases and cannot be compared to each other"
    )


def test_cell_ladder_groups_the_bootstrap_on_episodes_not_windows(monkeypatch):
    """Spec 3.1. 229 non-overlapping windows cut from 24 trajectories are not 229
    independent observations, and every reading from M3e onward clusters on the
    episodes. Passing `window` here would silently narrow every interval."""
    seen = []
    real = script.gain_from_blocks

    def recording(fit, select, score, **kwargs):
        seen.append(np.asarray(kwargs["groups"]).copy())
        return real(fit, select, score, **kwargs)

    monkeypatch.setattr(script, "gain_from_blocks", recording)
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    script.cell_ladder(fit, select, score, h_dim=3, ks=(1,))
    for groups in seen:
        assert np.unique(groups).size == np.unique(score["episode"]).size, (
            "the bootstrap is grouped on windows, not episodes"
        )


def test_cell_ladder_scores_every_rung_on_the_same_rows():
    """All four rungs are gains against one shared base, which is only true if
    they were scored on one row set. The row count per k is recorded so the
    claim is checkable from the artefact."""
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    ladder = script.cell_ladder(fit, select, score, h_dim=3, ks=(1, 4))
    # 8 windows of 20 steps: k=1 drops 8 rows, k=4 drops 32.
    assert ladder["rows"] == {"k1": 152, "k4": 128}
    for target in TARGETS:
        counts = {
            ladder["ladder"][target]["k4"][rung]["n_scored_windows"] for rung in RUNGS
        }
        assert len(counts) == 1, f"{target}: rungs scored on different groups: {counts}"


def test_cell_ladder_finds_the_gain_a_lagged_latent_carries():
    """A known-answer case. The deterministic half of `latent` is the PREVIOUS
    frame's position, so `enc(t) (+) h(t)` determines `p(t) - p(t-1)` exactly
    while `enc(t)` alone cannot. If the deterministic rung shows no gain here,
    the rows, the block and the target have come apart and no null this script
    produces would mean anything."""
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    ladder = script.cell_ladder(fit, select, score, h_dim=3, ks=(1,))
    deterministic = ladder["ladder"]["translation"]["k1"]["deterministic"]
    assert deterministic["gain"] > 0.1, deterministic
    assert deterministic["ci_low"] > 0.0, deterministic


def test_cell_ladder_shows_no_gain_for_a_rung_that_adds_only_noise():
    """The other half of the known-answer pair. `z` here is pure noise, so its
    rung must not clear -- otherwise the statistic is rewarding width and the
    base is not really in both arms."""
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    ladder = script.cell_ladder(fit, select, score, h_dim=3, ks=(1,))
    stochastic = ladder["ladder"]["translation"]["k1"]["stochastic"]
    assert stochastic["ci_low"] <= 0.0, stochastic


def test_base_control_reports_the_level_not_a_gain():
    """`enc(t)` -> absolute position. There is nothing to take an increment
    over: this IS the arm the increments are measured against."""
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    control = script.base_control(fit, select, score)
    assert control["r2"] > 0.5, (
        "enc(t) carries position exactly in this fixture; a low r2 means the "
        "control is reading the wrong array"
    )
    assert "gain" not in control, "the base control is a level, not a gain"


def test_cell_ladder_refuses_a_horizon_no_window_is_long_enough_for():
    """A k with no rows is a protocol refusal, not an empty table. Reporting a
    horizon whose gain was computed on zero rows would put a number in the
    record that describes nothing."""
    with pytest.raises(ValueError, match="no window is long enough"):
        script.cell_ladder(
            _gathered(steps=6), _gathered(steps=6, seed=1), _gathered(steps=6, seed=2),
            h_dim=3, ks=(15,),
        )
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_latent_retention_script.py -q`
Expected: collection error — the script does not exist.

- [ ] **Step 3: Write the script's module header and measure half**

Create `scripts/latent_retention.py`. Copy the `_sibling` loader and the
`trust_horizon` re-exports from `scripts/latent_motion.py` lines 176–200
verbatim, changing only the module name in `spec_from_file_location`. Then:

```python
EXIT_BASE_UNRESOLVED = 39
"""`enc(t)` -> absolute position did not read. The base arm is the thing every
gain is measured against, so if the current frame cannot linearly say where it
is, a null on displacement says nothing about the representation and no reading
is taken."""

EXIT_MOTION_UNRESOLVED = 40
"""No rung read either target at any horizon. The measurement is empty: it
detected no motion anywhere on the path, so it cannot localise a loss and no
lever is chosen."""

PHASES: tuple[str, ...] = ("measure", "read", "all")

FIT_EPISODES: int = 20
"""Training episodes the probe weights are fit on -- `probe.PROBE_EPISODE_LIMIT`,
the same set gate criterion 4 fits on, so the levels stay comparable."""

SELECT_EPISODES: int = 20
"""Training episodes AFTER the first `FIT_EPISODES` that select the ridge.

`probe.filtering_gain`'s default, and NOT `scripts/latent_motion.py`'s 4. That
docstring records the measurement: a 4-episode selection split picks 1e5 for
both arms and reports +0.0325 where a 20-episode split picks 1e3 and recovers
-0.0208 -- THE SIGN FLIPS on a one-step selection error, because a gain is a
difference of levels and the grid moves each level by ~0.10 per decade.
Shrinking this to save a gather does not make the number noisier, it makes it
wrong."""


def retention_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    """One file per cell, named by BOTH the arm and the seed -- see
    `study.job_record_path` for what a colliding name costs, and M3i's own
    task reports for what it cost there."""
    return Path(out_dir) / f"retention_{arm}_seed{seed}.json"


def k_key(k: int) -> str:
    """`15 -> "k15"`. A record key may not contain '.', since `write_record`
    addresses non-finite fields by dotted path."""
    return f"k{int(k)}"


TARGET_BUILDERS = {
    "translation": backward_translation,
    "rotation": backward_rotation,
}
"""`TARGETS` -> the builder that makes it. A dict rather than an if-chain so a
target named in `retention.TARGETS` and missing here fails at import-time
lookup rather than silently producing a ladder with a hole in it."""


def gather_three_splits(prepared, train, val, *, seed: int):
    """The three DISJOINT gathers `probe.gain_from_blocks` needs.

    The same structure `probe.filtering_gain` uses, and for the reason its
    docstring records: the weights come from the first `FIT_EPISODES` training
    episodes, the ridge is selected on the NEXT `SELECT_EPISODES` (from beyond
    the fit set, so the weights train on the same episodes criterion 4 uses),
    and the reported R^2 and interval come from the validation windows that
    neither the weights nor the selection ever saw. Seeds are `seed`, `seed + 2`
    and `seed + 1` in that order, matching `filtering_gain` exactly so the two
    diagnostics describe the same rows.
    """
    used = list(train)
    fit_paths = used[:FIT_EPISODES]
    if not fit_paths:
        raise ValueError("no training episodes to fit the retention probes on")
    select_paths = used[FIT_EPISODES:FIT_EPISODES + SELECT_EPISODES]

    def gather(paths, draw: int, limit: int | None = None):
        return gather_probe_data(
            prepared.model, paths,
            prepared.common["feature_backbone"], prepared.common["device"],
            context=prepared.context, horizon=prepared.horizon,
            limit=len(paths) if limit is None else limit, seed=draw,
        )

    return (
        gather(fit_paths, seed),
        gather(select_paths, seed + 2) if select_paths else None,
        gather(val, seed + 1, limit=FIT_EPISODES),
    )


def _split_for(data: dict, target: str, k: int, rung: str, h_dim: int):
    """One rung's `GainSplit` at one target and horizon, plus the rows it uses.

    `base` is `enc(t)` on exactly the rows the target is defined on, so every
    rung at this `(target, k)` is handed a BYTE-IDENTICAL base array. The base
    control is not built here: it is a level on every row, not a gain on a
    shifted subset, so `base_control` fits it directly.
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
    return GainSplit(
        base=np.asarray(data["encoder_embedding"], dtype=np.float64)[rows],
        block=rung_block(data, rung, rows=rows, source=source, h_dim=h_dim),
        target=values,
    ), rows


def cell_ladder(fit: dict, select: dict | None, score: dict, *, h_dim: int,
                ks=K_REPORTED) -> dict:
    """Every `(target, k, rung)` gain for one cell, plus the row count per k.

EVERY RUNG AT ONE (target, k) IS HANDED A BYTE-IDENTICAL BASE ARRAY, which is
    what makes the four gains differences against the same base level and so
    comparable to each other rather than each only to its own fit (spec 2.1).
    `fit_probe` is deterministic, so identical inputs give an identical base
    level and the redundant re-fit inside each `gain_from_blocks` call costs
    accuracy nothing -- the base is ~1/8 of a joint solve, so sharing the fitted
    probe would save ~11% of solve time and is not worth the extra interface.
    Four INDEPENDENT row selections would break the property while still
    producing four plausible gains, which is why a test pins the arrays.

    The bootstrap groups on `episode`, not `window` (spec 3.1).
    """
    ladder: dict = {target: {} for target in TARGETS}
    rows_by_k: dict[str, int] = {}
    for k in ks:
        for target in TARGETS:
            splits, row_sets = {}, {}
            for rung in RUNGS:
                splits[rung] = {}
                for name, data in (("fit", fit), ("select", select), ("score", score)):
                    if data is None:
                        splits[rung][name] = None
                        continue
                    split, rows = _split_for(data, target, k, rung, h_dim)
                    splits[rung][name] = split
                    row_sets[name] = rows
            groups = np.asarray(score["episode"])[row_sets["score"]]
            rows_by_k[k_key(k)] = int(row_sets["score"].size)
            ladder[target][k_key(k)] = {
                rung: gain_from_blocks(
                    splits[rung]["fit"], splits[rung]["select"], splits[rung]["score"],
                    groups=groups, resamples=RESAMPLES, confidence=CONFIDENCE,
                )
                for rung in RUNGS
            }
    return {"ladder": ladder, "rows": rows_by_k}


def base_control(fit: dict, select: dict | None, score: dict) -> dict:
    """`enc(t)` -> absolute position, as an r2 LEVEL on the validation windows.

    The positive control M3i measured only during its final review and recorded
    nowhere; its section 9 provenance note asked a successor to build it into the
    measure phase, and this is that. Every row is used -- there is no backward
    shift, because the target is the frame's own state.

    A level, not a gain: this is the arm every gain in `cell_ladder` is measured
    against, so the only question is whether it reads at all.
    """
    def probe_for(data):
        return (
            np.asarray(data["encoder_embedding"], dtype=np.float64),
            np.asarray(data["targets"], dtype=np.float64),
        )

    fit_x, fit_y = probe_for(fit)
    score_x, score_y = probe_for(score)
    if select is None:
        probe = fit_probe(fit_x, fit_y)
    else:
        select_x, select_y = probe_for(select)
        probe = fit_probe(fit_x, fit_y, select_x, select_y)
    return {
        "r2": probe_r2(probe, score_x, score_y),
        "ridge": probe["ridge"],
        "ridge_selected": select is not None,
        "rows": int(score_y.shape[0]),
    }
```

The imports this needs, at the top of the script beside the ones
`scripts/latent_motion.py` already uses:

```python
from mbfps.eval.probe import (
    GainSplit, fit_probe, gain_from_blocks, gather_probe_data, probe_r2,
)
from mbfps.eval.retention import (
    CONFIDENCE, DECISION_K, K_REPORTED, RESAMPLES, RUNGS, TARGETS,
    backward_rotation, backward_translation, rung_block, shifted_rows,
)
```

- [ ] **Step 4: Add `measure_cell` and `measure_phase`**

These mirror `scripts/latent_motion.py`'s `measure_cell` / `measure_phase`
exactly in shape — read that file's lines 513–656 and keep its refusal wording,
its `self_check` call and its `write_record` provenance block. The only
differences are the payload and the record name:

```python
def measure_cell(args, cell, device, train, val, ks=K_REPORTED) -> tuple[int, dict | None]:
    """One cell: the checks, the three gathers, the ladder, the base control.

    Every refusal that comes before a pass is `prepare_cell`'s, so this script
    and `scripts/split_gap.py` and `scripts/latent_motion.py` all refuse a cell
    for the same reasons in the same words -- and the reproduction bound of
    M3h spec 2.4 is checked here for free.
    """
    status, prepared = prepare_cell(args, cell, device, train, val)
    if prepared is None:
        return status, None
    fit, select, score = gather_three_splits(prepared, train, val, seed=cell.seed)
    h_dim = int(prepared.model.rssm.cfg.h_dim)
    ladder = cell_ladder(fit, select, score, h_dim=h_dim, ks=ks)
    record = {
        "arm": cell.arm, "seed": cell.seed, "step": cell.record["step"],
        "device": str(device), "context": prepared.context, "horizon": prepared.horizon,
        "h_dim": h_dim,
        "ladder": ladder["ladder"],
        "rows": ladder["rows"],
        "base_control": base_control(fit, select, score),
        "clusters": int(np.unique(score["episode"]).size),
        "windows": {
            "episode": np.asarray(score["episode"]).tolist(),
            "window": np.asarray(score["window"]).tolist(),
        },
        "self_check": self_check(cell, prepared.reference),
        "episodes": {"fit": [p.name for p in train[:FIT_EPISODES]],
                     "select": [p.name for p in
                                train[FIT_EPISODES:FIT_EPISODES + SELECT_EPISODES]],
                     "val": [p.name for p in val]},
    }
    return EXIT_OK, record
```

`measure_phase` iterates the cells, writes each record through the same
`write_record` helper `scripts/latent_motion.py` uses (so the non-finite scan
and the `git_sha` provenance are shared), prints one line per cell, and returns
the first non-`EXIT_OK` status or `EXIT_OK`.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/eval/test_latent_retention_script.py -q`
Expected: PASS, 14 tests.

- [ ] **Step 6: Commit**

```bash
git add scripts/latent_retention.py tests/eval/test_latent_retention_script.py
git commit -m "feat: latent_retention.py measure -- three disjoint gathers, one shared base per (target, k), episode-clustered intervals"
```

---

### Task 8: `scripts/latent_retention.py` — the read phase, the exit registry, and M3i's finiteness note

**Why:** `read` pools the nine records into Reading E and writes `retention.txt`. It also carries the three refusals that only a pooled view can see, and closes M3i's one open note.

**Files:**
- Modify: `scripts/latent_retention.py` — append
- Modify: `src/mbfps/eval/motion.py` — the finiteness guard
- Test: `tests/eval/test_latent_retention_script.py`, `tests/eval/test_motion.py`, `tests/eval/test_diagnose_dynamics_script.py`

**Interfaces:**
- Consumes: everything Tasks 4–7 produced.
- Produces:

```python
def load_retention(out_dir: Path, arms, seeds) -> dict: ...
def require_one_protocol(records: dict) -> None: ...
def require_readable_plan(arms, seeds) -> None: ...
def retention_inputs(records: dict) -> RetentionInputs: ...
def retention_text(records, inputs, reading) -> str: ...
def write_text(path: Path, text: str) -> Path: ...
def read_phase(args) -> int: ...
def main(argv: list[str] | None = None, *, ks=K_REPORTED) -> int: ...
```

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_latent_retention_script.py`:

```python
def _record(arm: str, seed: int, *, clearing=(), base_r2: float = 0.30,
            episodes: int = 24) -> dict:
    """A minimal record with the schema `read` addresses. `clearing` is a set of
    `(target, k, rung)` triples whose interval excludes zero."""
    ladder = {}
    for target in TARGETS:
        ladder[target] = {}
        for k in K_REPORTED:
            ladder[target][f"k{k}"] = {}
            for rung in RUNGS:
                hit = (target, k, rung) in clearing
                ladder[target][f"k{k}"][rung] = {
                    "gain": 0.25 if hit else -0.01,
                    "ci_low": 0.10 if hit else -0.08,
                    "ci_high": 0.40 if hit else 0.03,
                    "joint_r2": 0.5, "embedding_r2": 0.25,
                    "confidence": 0.95, "n_scored_windows": episodes,
                    "ridge_selected": True, "joint_ridge": 1e3,
                    "embedding_ridge": 1e3,
                }
    return {
        "arm": arm, "seed": seed, "step": 20000, "device": "cpu",
        "context": 5, "horizon": 45, "h_dim": 512,
        "ladder": ladder,
        "rows": {f"k{k}": 11221 - k * 229 for k in K_REPORTED},
        "base_control": {"r2": base_r2, "ridge": 1e3, "ridge_selected": True,
                         "rows": 11450},
        "clusters": episodes,
        "windows": {"episode": list(range(episodes)), "window": list(range(episodes))},
        "self_check": {"ok": True},
        "episodes": {"fit": [], "select": [], "val": [f"e{i}" for i in range(episodes)]},
    }


def _records(clearing=(), base_r2: float = 0.30) -> dict:
    arms = ("frozen_ssl", "pixel_ae", "random_vit")
    return {
        (arm, seed): _record(arm, seed, clearing=clearing, base_r2=base_r2)
        for arm in arms for seed in (0, 1, 2)
    }


def test_require_readable_plan_refuses_a_plan_too_narrow_to_reach_a_verdict():
    """The rule needs ARMS_REQUIRED arms of SEEDS_REQUIRED seeds. A narrower
    plan cannot reach any status, so reading it would print a verdict the data
    could not have supported -- M3i shipped exactly that defect and caught it in
    review, where a narrowed plan printed `z +996.13, 3/3 up` beside NO
    DIFFERENCE."""
    with pytest.raises(SystemExit):
        script.require_readable_plan(("pixel_ae",), (0, 1, 2))
    with pytest.raises(SystemExit):
        script.require_readable_plan(("pixel_ae", "frozen_ssl"), (0,))
    script.require_readable_plan(("pixel_ae", "frozen_ssl"), (0, 1))


def test_require_one_protocol_refuses_records_scored_on_different_windows():
    """Nine cells pooled into one reading must describe the same rows. Two
    protocols pooled as one would be a reading over a union nothing measured."""
    records = _records()
    records[("pixel_ae", 1)]["windows"]["episode"] = list(range(20))
    with pytest.raises(SystemExit):
        script.require_one_protocol(records)


def test_retention_inputs_tallies_seeds_and_arms_from_the_records():
    """The tally IS the rule, so it has to come from the per-seed intervals in
    the records rather than from a pooled number."""
    clearing = {("translation", 4, "full")}
    inputs = script.retention_inputs(_records(clearing))
    arm = inputs.ladder["translation"][4]["full"]["pixel_ae"]
    assert arm.seeds_clear == 3 and arm.seeds_total == 3
    other = inputs.ladder["translation"][1]["full"]["pixel_ae"]
    assert other.seeds_clear == 0


def test_retention_inputs_refuses_a_non_finite_gain_in_any_record():
    """M3i's ledger note, closed. A NaN must stop the read, not read as a
    non-clear."""
    records = _records()
    records[("pixel_ae", 1)]["ladder"]["translation"]["k4"]["full"]["gain"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        script.retention_inputs(records)


def test_retention_inputs_reads_the_base_control_against_the_floor():
    from mbfps.eval.retention import BASE_R2_FLOOR
    holding = script.retention_inputs(_records(base_r2=BASE_R2_FLOOR + 0.05))
    assert all(c.clears() for c in holding.base.values())
    failing = script.retention_inputs(_records(base_r2=BASE_R2_FLOOR - 0.05))
    assert not any(c.clears() for c in failing.base.values())


def test_retention_text_is_what_read_prints_byte_for_byte():
    """`retention.txt` and stdout must be the same bytes, or the artefact and
    the log disagree about what the run said."""
    from mbfps.eval.retention import reading_retention
    records = _records({("translation", 4, "two_frame")})
    inputs = script.retention_inputs(records)
    text = script.retention_text(records, inputs, reading_retention(inputs))
    assert "Reading E" in text and "verdict: MOTION DISCARDED" in text
    assert "The ladder" in text
    # The ladder is printed BEFORE the verdict, so a reader meets the evidence
    # before the conclusion -- the order M3i's motion.txt uses.
    assert text.index("The ladder") < text.index("Reading E")
```

Six end-to-end tests, one per status — the direct lesson from M3i, which shipped
a `MOTION_ENCODED` path nothing exercised until the final review:

```python
@pytest.mark.parametrize("clearing,expected", [
    ({("translation", 4, "stochastic")}, "MOTION RETAINED"),
    ({("translation", 15, "full")}, "MOTION RETAINED"),
    ({("translation", 4, "deterministic")}, "BOTTLENECK LOSS"),
    ({("translation", 4, "two_frame")}, "MOTION DISCARDED"),
    ({("rotation", 4, "two_frame")}, "TRANSLATION UNRESOLVED"),
    (set(), "UNRESOLVED MOTION"),
])
def test_read_reaches_every_status_end_to_end(tmp_path, capsys, clearing, expected):
    """Every branch reachable, driven through the real `read` phase to a printed
    verdict. M3i's spec promised a MOTION_ENCODED path and nothing tested it;
    the final whole-branch review found it, and a fixture per status is the
    cheap version of that check."""
    for (arm, seed), record in _records(clearing).items():
        path = script.retention_record_path(tmp_path, arm, seed)
        path.write_text(__import__("json").dumps(record))
    status = script.main(["--phase", "read", "--out", str(tmp_path)])
    printed = capsys.readouterr().out
    assert f"verdict: {expected}" in printed
    assert (tmp_path / "retention.txt").read_text() in printed
    assert status == script.EXIT_OK


def test_read_exits_39_when_the_base_control_fails(tmp_path, capsys):
    """A broken instrument is a refusal with its own exit code, not a status
    printed beside a reading."""
    from mbfps.eval.retention import BASE_R2_FLOOR
    for (arm, seed), record in _records(base_r2=BASE_R2_FLOOR - 0.05).items():
        script.retention_record_path(tmp_path, arm, seed).write_text(
            __import__("json").dumps(record)
        )
    assert script.main(["--phase", "read", "--out", str(tmp_path)]) == 39
    assert "UNRESOLVED BASE" in capsys.readouterr().out


def test_read_exits_40_when_nothing_reads_either_target(tmp_path):
    for (arm, seed), record in _records().items():
        script.retention_record_path(tmp_path, arm, seed).write_text(
            __import__("json").dumps(record)
        )
    assert script.main(["--phase", "read", "--out", str(tmp_path)]) == 40
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_latent_retention_script.py -q`
Expected: FAIL — `AttributeError: module has no attribute 'require_readable_plan'`.

- [ ] **Step 3: Write the read half**

Append to `scripts/latent_retention.py`. Mirror
`scripts/latent_motion.py`'s `load_motion` / `require_one_protocol` /
`require_readable_plan` / `motion_text` / `write_text` / `read_phase` /
`_parser` / `main` (lines 657–1131) in shape and refusal wording; the pieces
that differ:

```python
def require_readable_plan(arms, seeds) -> None:
    """Refuse a plan too narrow for the rule to reach any status.

    The rule needs `ARMS_REQUIRED` arms, each clearing in `SEEDS_REQUIRED`
    seeds. A plan with fewer cannot reach MOTION_RETAINED, BOTTLENECK_LOSS,
    MOTION_DISCARDED or TRANSLATION_UNRESOLVED at all -- it can only ever print
    UNRESOLVED_MOTION, which would look like a finding. M3i shipped this defect
    and caught it in review: a narrowed plan printed `z +996.13, 3/3 up` beside
    a NO DIFFERENCE verdict.
    """
    if len(arms) < ARMS_REQUIRED or len(seeds) < SEEDS_REQUIRED:
        raise SystemExit(
            f"a plan of {len(arms)} arm(s) x {len(seeds)} seed(s) cannot reach any "
            f"status: the rule needs {ARMS_REQUIRED} arms clearing in "
            f"{SEEDS_REQUIRED} seeds each, so this plan can only ever print "
            f"UNRESOLVED_MOTION -- which would read as a finding"
        )


def retention_inputs(records: dict) -> RetentionInputs:
    """Pool the nine records into Reading E's input.

    The seed tally is built from the per-seed intervals in the records, not from
    a pooled estimate: r2 is not a per-window quantity, so there is nothing to
    pool the way `pooling.paired_contrast` pools a contrast. The agreement
    requirement IS the rule here (spec 3.1), which is also what carries the
    multiple-comparison burden across 4 rungs x 3 horizons.

    `retention.rung_arm` raises on a non-finite gain or bound, which is where
    M3i's ledger note is closed: a NaN must stop the read rather than evaluate
    False in every predicate at once.
    """
    arms = sorted({arm for arm, _ in records})
    ladder: dict = {}
    for target in TARGETS:
        ladder[target] = {}
        for k in K_REPORTED:
            ladder[target][k] = {}
            for rung in RUNGS:
                ladder[target][k][rung] = {
                    arm: rung_arm([
                        records[(arm, seed)]["ladder"][target][k_key(k)][rung]
                        for _, seed in sorted(cell for cell in records if cell[0] == arm)
                    ])
                    for arm in arms
                }
    base = {}
    for arm in arms:
        levels = [
            float(records[(arm, seed)]["base_control"]["r2"])
            for _, seed in sorted(cell for cell in records if cell[0] == arm)
        ]
        if not all(np.isfinite(levels)):
            raise ValueError(f"non-finite base control r2 in arm {arm}: {levels}")
        base[arm] = BaseControl(
            r2=float(np.mean(levels)),
            seeds_clear=sum(1 for r in levels if r > BASE_R2_FLOOR),
            seeds_total=len(levels),
        )
    first = records[next(iter(sorted(records)))]
    return RetentionInputs(
        ladder=ladder, base=base,
        clusters=int(first["clusters"]),
        rows={k: int(first["rows"][k_key(k)]) for k in K_REPORTED},
    )


READ_EXITS = {
    "UNRESOLVED_BASE": EXIT_BASE_UNRESOLVED,
    "UNRESOLVED_MOTION": EXIT_MOTION_UNRESOLVED,
}
"""Status -> exit code, for the two statuses that are refusals. Every other
status is a reading and exits 0: a milestone that exited non-zero on a finding
would make "the run worked" and "the news was good" the same signal."""
```

`read_phase` loads the records, runs `require_one_protocol`, builds the inputs,
calls `reading_retention`, composes `retention_text` from the self-check table,
`format_ladder` and `format_reading_retention` in that order, writes
`retention.txt` through `write_text`, prints the same string, and returns
`READ_EXITS.get(reading.status, EXIT_OK)`.

- [ ] **Step 4: Add `latent_retention` to the exit-status registry**

`tests/eval/test_diagnose_dynamics_script.py` holds the registry test that
already lists `stage_decomposition`, `sharper_latent` and `latent_motion`. Add
`latent_retention` with its two codes, following that file's existing shape
exactly. Run:

`.venv/bin/python -m pytest tests/eval/test_diagnose_dynamics_script.py -q`
Expected: PASS.

- [ ] **Step 5: Close M3i's finiteness note in `motion.py`**

M3i's ledger left exactly one note for its successor: `clears_up`,
`clears_down` and `leaks` all evaluate `False` on NaN, so one NaN window would
read "no clear" and "no leak" at once. Add to `src/mbfps/eval/motion.py`, above
`MotionArm`:

```python
def _finite(value: float, what: str) -> float:
    """Refuse a non-finite statistic rather than letting it read as a non-clear.

    `clears_up`, `clears_down` and `leaks` are all comparisons, and every
    comparison against NaN is False -- so a single non-finite z would read "no
    clear" AND "no leak" at once, moving a verdict toward NO_DIFFERENCE or
    masking a leaking control, in both cases looking like a clean null.

    No M3i verdict moves. All 90 of its recorded series were verified finite, so
    this guard cannot fire on the data it was written for; it exists so the next
    milestone to reuse this reading is not the one that finds out.
    """
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite {what}: {value!r}")
    return value
```

Add `import math` at the top, and call it at the head of each predicate:

```python
    def clears_up(self, z_fam: float) -> bool:
        return (_finite(self.z, "z") >= _finite(z_fam, "z_fam")
                and self.seeds_up >= SEEDS_REQUIRED)

    def clears_down(self, z_fam: float) -> bool:
        return (_finite(self.z, "z") <= -_finite(z_fam, "z_fam")
                and self.seeds_down >= SEEDS_REQUIRED)
```

and in `leaks`, `return abs(_finite(self.z, "z")) >= _finite(z_fam, "z_fam")`.

Add to `tests/eval/test_motion.py`:

```python
def test_motion_arm_predicates_refuse_a_non_finite_z():
    """The one note M3i's ledger left its successor. A NaN z made all three
    predicates return False at once -- "no clear" and "no leak" together --
    which reads as a clean null. M3i's own 90 recorded series were verified
    finite, so no recorded verdict moves; the guard exists for the next reader."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        arm = MotionArm(estimate=0.0, se=1.0, z=bad, seeds_up=3, seeds_down=0,
                        seeds_total=3)
        for predicate in (arm.clears_up, arm.clears_down, arm.leaks):
            with pytest.raises(ValueError, match="non-finite z"):
                predicate(2.582)


def test_motion_arm_predicates_are_unchanged_on_finite_input():
    """The guard must be a guard, not a behaviour change. These are the same
    assertions the shipped predicates already satisfy."""
    up = MotionArm(estimate=1.0, se=0.3, z=3.0, seeds_up=2, seeds_down=0, seeds_total=3)
    assert up.clears_up(2.582) and not up.clears_down(2.582) and up.leaks(2.582)
    down = MotionArm(estimate=-1.0, se=0.3, z=-3.0, seeds_up=0, seeds_down=2,
                     seeds_total=3)
    assert down.clears_down(2.582) and not down.clears_up(2.582) and down.leaks(2.582)
```

- [ ] **Step 6: Run every affected suite**

Run: `.venv/bin/python -m pytest tests/eval/test_latent_retention_script.py tests/eval/test_motion.py tests/eval/test_latent_motion_script.py tests/eval/test_diagnose_dynamics_script.py -q`
Expected: PASS. `test_latent_motion_script.py` is included because the guard is
in a module that script reads through; a refusal on non-finite input cannot
change a result that had no non-finite input, and this is where that is checked.

- [ ] **Step 7: Commit**

```bash
git add scripts/latent_retention.py src/mbfps/eval/motion.py tests/eval/
git commit -m "feat: latent_retention.py read -- Reading E, the two refusal exits, and M3i's finiteness note closed"
```

---

### Task 9: The run

**Why:** the milestone's deliverable is the reading on the nine shipped cells, not the code that computes it.

**Files:**
- Create: `runs/m3j_smoke/`, `runs/m3j_retention/` (run artefacts; never `rm` anything under `runs/`)
- Modify: `docs/superpowers/plans/2026-09-27-mb-fps-m3j-motion-retention.md` — add `## Task 9 results`

- [ ] **Step 1: Check the disk before the suite**

```bash
du -sh "${TMPDIR}/pytest-of-${USER}" ; df -h . | tail -1
```

`tmp_path_retention_policy = "failed"` landed in PR #8 and held M3i's whole-suite
run to 1.0 GB against the ~24 GB it took before. The check is cheap and on
2026-09-26 its absence killed three consecutive suite runs.

- [ ] **Step 2: Run the whole suite**

```bash
caffeinate -dimsu .venv/bin/python -m pytest -q
```

Expected: 0 failures, 0 warnings. Record the exact count and duration; the
results section quotes them. Do **not** commit anything while a run is in
progress.

- [ ] **Step 3: Smoke one cell, and time it**

```bash
caffeinate -dimsu .venv/bin/python scripts/latent_retention.py \
  --phase all --arms pixel_ae --seeds 0 \
  --source runs/m3_study_v2 --out runs/m3j_smoke 2>&1 | tee runs/m3j_smoke/smoke.log
```

Expected: exit 0 or 40, `runs/m3j_smoke/retention.txt` written. `read` will
refuse a one-cell plan via `require_readable_plan`, so run `--phase measure`
alone if the combined phase refuses — that refusal is correct and Task 8 tests
it. **Read `retention.txt` for format before the real run** (M3h's smoke caught
a reporting bug the whole suite had missed) and **record the wall clock**, which
is how the measure phase is sized rather than by an estimate.

- [ ] **Step 4: Measure all nine cells**

```bash
git rev-parse HEAD > runs/m3j_retention/.head
date -u +%Y-%m-%dT%H:%M:%SZ > runs/m3j_retention/.started
nohup caffeinate -dimsu .venv/bin/python scripts/latent_retention.py \
  --phase measure --source runs/m3_study_v2 --out runs/m3j_retention \
  > runs/m3j_retention/measure.log 2>&1 &
echo $!
```

When it exits, write `.finished` and `.exit`. **Wait on the process, not on a
file** — M3i lost time to a waiter that fired instantly on a stale `.exit` from
an earlier run.

- [ ] **Step 5: Accept the run**

Check, and refuse to read if any fails:

- nine `retention_<arm>_seed<n>.json`, one `git_sha` equal to `.head`, all `mps`
- `self_check.ok` on 9/9, within M3h spec 2.4's reproduction bound
- 229 scored windows over 24 episode clusters on every cell: assert
  `len(set(windows.episode)) == 24` and `len(windows.window) == 229`
- `rows` equal to `{"k1": 11221, "k4": 10534, "k15": 8015}` on every cell
- every gain, `ci_low` and `ci_high` finite throughout
- `base_control.r2` above `BASE_R2_FLOOR` in at least 2 of 3 seeds in at least 2
  of 3 arms

- [ ] **Step 6: Read**

```bash
.venv/bin/python scripts/latent_retention.py --phase read \
  --out runs/m3j_retention 2>&1 | tee runs/m3j_retention/read.log
echo "exit: $?"
```

Then verify the artefact and the log agree:

```bash
diff <(.venv/bin/python scripts/latent_retention.py --phase read \
  --out runs/m3j_retention) runs/m3j_retention/retention.txt && echo "byte-identical"
```

- [ ] **Step 7: Write `## Task 9 results` into this plan**

Every number quoted from an artefact with its definition, never from a scratch
measurement — M3h shipped a motivating premise that was never measured as
stated, and M3i's spec 1.1 records why. Cover, in this order: the verdict and
the lever it pre-registers; provenance (git_sha, device, protocol, timings, exit
code); the suite; acceptance; the self-check table; the base control; Reading E's
table; the full ladder; the rotation control; and **what the milestone does not
support**, written as plainly as what it does.

State explicitly whether the reading **overturns M3i**: `MOTION_RETAINED` means
it does, and spec 3.3 pre-registered that this reverses M3h section 8's
retirement. That sentence is owed whichever way the numbers fall.

- [ ] **Step 8: Commit**

```bash
git add runs/m3j_retention docs/superpowers/plans/2026-09-27-mb-fps-m3j-motion-retention.md
git commit -F <commit message file>
```

Use `-F` with a message file: M3i lost a commit to shell quoting on a message
containing an apostrophe.

---

## Exit criteria

- Reading E is taken on nine records at one `git_sha`, or the run refused with a
  numbered status and the refusal is recorded.
- `filtering_gain`'s output is byte-identical, pinned by
  `test_gain_from_splits_output_is_byte_identical_after_the_generalisation`.
- All six statuses are reachable and each is driven end-to-end to a printed
  verdict by a test.
- The whole suite passes with 0 failures and 0 warnings.
- `retention.txt` is byte-identical to a second `--phase read`.
- No checkpoint altered, nothing under `runs/` removed, no M3b–M3i verdict or
  record touched.

## Plan self-review (writing-plans)

**Spec coverage.** 2.1 -> Tasks 3, 7. 2.2 -> Task 4. 2.3 -> Tasks 4, 7. 3.1 ->
Tasks 2, 5, 7. 3.2 -> Task 5. 3.3 -> Task 9 Step 7. 3.4 -> Tasks 7, 9. 3.5 ->
Task 6. 4 -> Task 9 Step 7. 5 -> Tasks 1–8. 6 -> Task 9. 7 -> the File
Structure table. No gaps.

**Placeholder scan.** Clean. Five defects were found and fixed inline rather than
left for a reviewer:

1. **Task 6 carried a deliberate placeholder** — a dead `if False else` expression
   printing the gain twice in the `ci_high` column, with a later step to "resolve"
   it. Staging a defect on purpose is worse than writing it right. Removed, and
   `ci_high` threaded properly through `RungArm` and `rung_arm` in Task 5.
2. **A false claim in spec 2.1**, propagated into Task 7. The spec said the base
   is "fit once per (target, k) and shared across all four rungs", and the code
   does not do that — each `gain_from_blocks` call refits it. The property that
   actually matters and is actually implemented is that every rung is handed a
   byte-identical base ARRAY, so the deterministic fit gives an identical base
   level. Spec and plan both corrected, and the test now pins the arrays.
3. **That test's assertion was also wrong** — it expected two distinct base
   arrays, one per target, but Task 4 pins that both targets share a row set at a
   given k, so the correct count is one.
4. **Two assertions that could not fail.** `f"unreachable statuses: {set(seen) ^ seen}"`
   is always empty, and `assert ... or True` in the `retention_text` test is a
   tautology. Both replaced with assertions that bite — the second now pins that
   the ladder prints before the verdict.
5. **A misnamed test** (`test_measure_cell_refuses_...` calling `cell_ladder`) and
   a dead `rung is None` branch in `_split_for`, reachable from nothing.

**Type consistency.** `shifted_rows` returns `(rows, source)` in Tasks 4, 6, 7.
`backward_translation` / `backward_rotation` return `(values, rows)` throughout.
`RungArm` carries `gain, ci_low, ci_high, seeds_clear, seeds_total` in Tasks 5,
6, 7, 8. `GainSplit` is `(base, block, target)` in Tasks 3 and 7.
`gain_from_blocks` returns `"embedding_r2"` (not `"base_r2"`) in Tasks 3, 7, 8,
because `study.py` and `scripts/eval_rollout.py` read it by that name.
`_block_bootstrap_ci` takes `window` XOR `groups` in Tasks 2, 3, 7.

## Task 9 results

**Reading E is `MOTION RETAINED`** — and the sentence that must sit beside it is that the
pre-registered rule cleared on a margin roughly **63 times smaller than the neighbouring rung's
gain**, while every magnitude bearing on `z` says the bottleneck is destroying the motion. Both
statements are true, both are reported, and the rule is not rewritten after the fact.

The verdict was decided by the `stochastic` rung at k = 15, whose mean gain over the nine cells is
**+0.00033**. At the same horizon the `deterministic` rung's mean gain is **+0.02071** — 63x larger
— and `full` minus `deterministic`, which is exactly what `z` adds over `h`, is **+0.00064** on
translation and **negative** on rotation at every horizon. So spec 3.3's pre-registered consequence
for `MOTION_RETAINED` (M3i partially overturned, M3h section 8's prior-side levers return) is owed
by the rule as written, and the measured magnitudes point somewhere else: at the 32x32 categorical
bottleneck, which is `BOTTLENECK_LOSS`'s lever — **and, section 9 below also shows, at the objective**,
because `h` itself recovers less displacement than two raw frames at the longest horizon. Section 9
sets out what the project should actually do with both.

### 1. Provenance

| | |
|---|---|
| `git_sha` (all nine records) | `8bd6f93` |
| `record_git_sha` (the M3c cells read) | `ca3e140` |
| device / torch | `mps` / 2.13.0 |
| protocol | context 5, horizon 45, `split_seed` 0, 24 val episodes, `K_REPORTED` (1, 4, 15), `DECISION_K` 4 |
| smoke (one cell, timed) | 2026-09-28T17:32:45Z -> 17:35:38Z (**2 m 53 s**) |
| measure (nine cells) | 2026-09-28T17:35:54Z -> 18:01:52Z (**25 m 58 s**) |
| `read` exit | **0** |
| artefacts | nine `retention_<arm>_seed<n>.json`, `retention.txt` (4,643 bytes) |
| `nonfinite` | empty on all nine |

`retention.txt` is byte-identical to what a second `--phase read` prints, verified with `cmp`.
No checkpoint was written or altered and nothing under `runs/` was removed; the phase is
evaluation only.

### 2. The smoke run earned its place, and the plan was wrong

The first smoke (`runs/m3j_smoke/`, kept) came back with **20 clusters and 9,261 rows at k = 1**
against the **24 and 11,221** spec sections 3.4 and 6 state and accept on. The cause:
`probe.filtering_gain`'s `limit` caps the fit split **and** the scored split at the same number —
its own docstring says "episodes for the fit split, and for the scored split" — and Task 7 was told
to mirror that function exactly. Mirroring it literally scored 20 of the 24 validation episodes and
silently discarded four, 17% of the evaluation data, on a protocol every milestone since M3d reads
as 229 windows over 24 episodes.

Fixed, and the second smoke (`runs/m3j_smoke2/`) reproduces the spec's numbers exactly. The test
that was supposed to pin this asserted `limit == FIT_EPISODES` on a fixture where `len(val)` was
**also** 20, so the two numbers could not be told apart; it now uses a fixture where they differ.

*This is what a smoke run is for, and it is the second milestone running where the smoke caught
something the whole suite had missed.*

### 3. Acceptance

Nine records; one `git_sha` equal to `.head`; all on `mps`; `self_check.ok` on 9/9 within M3h spec
2.4's reproduction bound; **11,450 scored rows over 229 distinct windows and 24 episode clusters on
every cell**; `rows` exactly `{k1: 11221, k4: 10534, k15: 8015}` on every cell; `step` 20000; every
gain and interval bound finite; `nonfinite` empty throughout; the base control holding in 3 of 3
arms. Passed on every check.

### 4. The base control, and its companion

```
  base control (enc(t) -> absolute position, must clear r2 0.10):
      frozen_ssl r2=+0.334 3/3,  pixel_ae r2=+0.337 3/3,  random_vit r2=+0.318 3/3
  position control (x/y only, same fitted probe):
      frozen_ssl +0.682,  pixel_ae +0.704,  random_vit +0.660
```

The gate is on the 4-column mean over `pos_x`, `pos_y`, `sin(angle)`, `cos(angle)`, because
`BASE_R2_FLOOR = 0.10` was calibrated against `latent_selection_r2`, which is also a 4-column mean.
A review raised that three docstrings call that number "absolute position" while it does not
isolate position, and that a cell with position r2 ~ 0 and heading r2 ~ 0.25 would clear the floor
on heading alone. The companion was recorded so the record could not hide that.

**Measured, it does not happen, and it fails in the safe direction.** Position alone reads
**0.66–0.70** in every cell, roughly double the gated 4-column mean of 0.32–0.34 — the heading
columns are what drag the mean down. No cell diverges, and the gate is if anything conservative.

### 5. Reading E

```
             rung          arm      gain    ci_low   ci_high   seeds  clears
        two_frame   frozen_ssl   +0.0101   -0.0021   +0.0227     1/3      no
        two_frame     pixel_ae   +0.0062   -0.0082   +0.0232     0/3      no
        two_frame   random_vit   +0.0098   -0.0033   +0.0197     2/3     yes
    deterministic   frozen_ssl   +0.0119   -0.0042   +0.0224     2/3     yes
    deterministic     pixel_ae   +0.0154   +0.0035   +0.0390     3/3     yes
    deterministic   random_vit   +0.0126   +0.0052   +0.0236     3/3     yes
       stochastic   frozen_ssl   +0.0005   -0.0002   +0.0012     2/3     yes
       stochastic     pixel_ae   +0.0020   -0.0035   +0.0132     1/3      no
       stochastic   random_vit   +0.0002   -0.0004   +0.0011     1/3      no
             full   frozen_ssl   +0.0121   -0.0041   +0.0228     2/3     yes
             full     pixel_ae   +0.0156   +0.0037   +0.0396     3/3     yes
             full   random_vit   +0.0127   +0.0053   +0.0240     3/3     yes
  verdict: MOTION RETAINED
```

At `DECISION_K` itself `stochastic` clears in only **1 of 3 arms**. The rule is a disjunction over
`K_REPORTED`, and what carried it is k = 15, where `stochastic` clears in 2 of 3 arms.

### 6. The ladder — the whole disjunction, and where the story actually is

Mean gain over all nine cells:

| rung | tr k=1 | tr k=4 | tr k=15 | rot k=1 | rot k=4 | rot k=15 |
|---|---|---|---|---|---|---|
| `two_frame` | +0.00216 | +0.00869 | **+0.03489** | +0.00012 | +0.00251 | +0.00051 |
| `deterministic` | +0.00751 | +0.01327 | +0.02071 | **+0.49380** | **+0.25648** | **+0.32484** |
| `stochastic` | +0.00066 | +0.00091 | +0.00033 | +0.00002 | −0.00002 | +0.00007 |
| `full` | +0.00765 | +0.01347 | +0.02135 | +0.48585 | +0.24928 | +0.31942 |

**`z` adds nothing over `h`, and on rotation it subtracts.** `full` minus `deterministic` — the
increment the bottleneck contributes on top of the recurrent state — is:

| | k = 1 | k = 4 | k = 15 |
|---|---|---|---|
| translation | +0.000144 | +0.000209 | +0.000642 |
| rotation | **−0.007953** | **−0.007203** | **−0.005421** |

### 7. The rotation control says the action channel is doing the work

`deterministic` reads rotation at **+0.494 / +0.256 / +0.325**, in **9 of 9 seeds at every
horizon** — one to two orders above anything on translation. `two_frame` reads rotation at
**+0.0001 / +0.0025 / +0.0005**, essentially nothing.

Two real frames give almost no rotation; the recurrent state gives half an R². The difference
between them is the **action sequence**, which `h` integrates and two frames do not contain —
exactly the asymmetry spec 2.2 predicted when it insisted `two_frame` is a reference and not a
ceiling. Turning in `my_way_home` is commanded directly by the action, so `h` can integrate it
almost perfectly; translation is mediated by geometry and collision, and `h` recovers far less of
it.

It also means the rotation control did its job in the direction that matters: the instrument can
read motion — hugely — so the small translation numbers are a fact about translation, not about a
dead probe.

### 8. `two_frame` grows with the horizon, as the sizing predicted

`two_frame` on translation runs **+0.00216 -> +0.00869 -> +0.03489** across k = 1, 4, 15, and only
clears at k = 15. Spec 2.3 sized a 1-step move at 5.77 map units — 0.6% of the map extent, against
a recorded probe position error of 222–247 units — and a 15-step move at 57.49. Two real frames
resolve a 57-unit displacement and not a 6-unit one, which is what the table shows.

### 9. A flaw in this milestone's own status set, disclosed rather than passed over

**`Z_BEARING_RUNGS = ("stochastic", "full")` cannot distinguish the two cases it exists to
separate.** `full` is `h (+) z`, so a `full` that clears says only that the *concatenation* beats
`enc(t)` — it cannot tell "motion survived into `z`" from "`h` carries it and `z` is dead weight".
Including it in the z-bearing set makes `MOTION_RETAINED` reachable on `h` alone, which is
precisely what `BOTTLENECK_LOSS` was written to name.

This is not hypothetical here. Re-running the shipped `reading_retention` with `stochastic` forced
to clear **nowhere**, at any horizon, in any arm:

```
as measured                  -> MOTION_RETAINED  | surviving: stochastic
if z alone cleared nowhere   -> MOTION_RETAINED  | surviving: full
```

The status does not move. So the verdict never rested on `z` carrying anything, and
`full - deterministic = +0.0002` on translation says `z` does not.

**The verdict stands as taken.** It was read under the rule exactly as pre-registered — one-sided,
`ci_low > 0`, 2 of 3 seeds in 2 of 3 arms, disjunction over `K_REPORTED` — and this project does
not rewrite a rule after seeing the numbers. Spec 3.3's consequence for `MOTION_RETAINED` follows
as written: M3i's `NO_MOTION` was a statement about *forward* prediction rather than about the
representation's content, and M3h section 8's blanket retirement of the prior-side levers is
reversed.

**Two more facts strengthen that stand.** `BOTTLENECK_LOSS` is **near-unreachable by construction**:
`full` is `h (+) z`, a strict superset of `deterministic`'s information, so `BOTTLENECK_LOSS`
requires `deterministic` to clear translation while `full` does not — a status this ladder was never
on equal footing to reach regardless of what `z` carries. And under the corrected
`Z_BEARING_RUNGS = ("stochastic",)` this reading is **still** `MOTION_RETAINED`: as section 5 above
already shows, `stochastic` itself clears in 2 of 3 arms at k = 15, so the corrected rule reaches the
same status the shipped one did without leaning on `full` at all.

**And the rule is recorded as miscalibrated, for the same reason M3i recorded its control.** A
milestone reusing this status set should put **only `stochastic`** in `Z_BEARING_RUNGS`, because
only `stochastic` isolates the bottleneck's output. `full` belongs beside `deterministic` as a
descriptive rung: useful for reading how much the concatenation carries, useless for attributing
it.

**What the measured magnitudes say, stated separately from the status.** Three facts, none of which
the status encodes:

- `z` alone carries **+0.00033** of translation at k = 15 — the horizon that decided this reading —
  against `h`'s **+0.02071**, a factor of 63.
- `z` adds **+0.0006 or less** over `h` on translation, and **−0.005 to −0.008** on rotation: on the
  target the instrument reads best, appending `z` makes the fit *worse*.
- `z` carries **+0.00007** of rotation where `h` carries **+0.325**, a factor of ~4,600.

A latent whose stochastic half contributes three ten-thousandths of an R² where its deterministic
half contributes two hundredths is not one that retains motion in any sense a training decision
should rest on. That is the reading `BOTTLENECK_LOSS` names: **`h` carries translation and the 32x32
categorical bottleneck destroys it.** It is not, however, the only lever the magnitudes support.

**A fourth fact: two raw frames beat `h` outright at the longest horizon.** At k = 15 on translation,
`two_frame` (`enc(t-15)`, width 2048) linearly recovers *more* displacement than `deterministic`
(`h(t)`, width 512), in 8 of the 9 cells:

```
  frozen_ssl s0  two_frame +0.02243   deterministic +0.02109   two_frame wins
  frozen_ssl s1  two_frame +0.03036   deterministic +0.02470   two_frame wins
  frozen_ssl s2  two_frame +0.02123   deterministic +0.00602   two_frame wins
  pixel_ae   s0  two_frame +0.03329   deterministic +0.03411   deterministic wins by 0.0008
  pixel_ae   s1  two_frame +0.03313   deterministic +0.02131   two_frame wins
  pixel_ae   s2  two_frame +0.03414   deterministic +0.01225   two_frame wins
  random_vit s0  two_frame +0.05066   deterministic +0.01704   two_frame wins
  random_vit s1  two_frame +0.04924   deterministic +0.02833   two_frame wins
  random_vit s2  two_frame +0.03950   deterministic +0.02151   two_frame wins

  two_frame beats deterministic in 8 of 9 cells at k=15
  means: k=1  tf +0.00216 / det +0.00751   k=4  tf +0.00869 / det +0.01327   k=15  tf +0.03489 / det +0.02071
```

`h` has every frame between `t-15` and `t`, *and* the whole 15-step action sequence, and still
recovers less translation than the two endpoint frames alone give a plain linear probe. That is not a
statement about the bottleneck — `two_frame` never touches `z` — it is evidence that the RSSM
**discards translation information the encoder demonstrably provided**, which is
`MOTION_DISCARDED`'s story, and the lever it names is the **objective**: `world_model.py:81`'s
embedding loss reconstructs only the frame just seen and never asks `h` to retain displacement (§1).

**The honest synthesis is that the magnitudes support two upstream levers, not one.** `z` adds
nothing over `h` (the bottleneck argument, three bullets above), *and* `h` recovers less displacement
than two raw frames at k = 15 (the objective argument here). The two are not in tension — they are
about different parts of the path — and read together they say the bottleneck destroys what little
the RSSM was already failing to retain in `h`.

**The caveat, in the same breath as the finding.** `two_frame`'s joint probe (`enc(t)` plus its
2048-wide block) is 4096 features wide against `deterministic`'s 2560, and n/p at k = 15 is **~2.0**
at that widest rung — the low end of spec 3.4's own 2.0–2.7 range (2.6 sits at `DECISION_K`, not
k = 15) — which is not n ≫ p. A wider block can win a comparably-sized ridge fit on selection slack
alone, the same mechanism spec 2.1 and `SELECT_EPISODES`'s own docstring warn flips a gain's *sign*
on a one-episode selection error. If width explains some or all of the `two_frame`-over-`deterministic`
margin, this comparison is uninformative about the objective specifically — and either way the prose
cannot say "every magnitude" points at the bottleneck alone, because this comparison exists and has
not been ruled out.

**So the magnitudes support two candidate levers, not a single one** — the bottleneck (`z_cats` x
`z_classes`, 32x32, at most ~160 bits, and `rep_scale = 0.1`, five times below `dyn_scale`) and the
objective (`world_model.py:81`'s embedding loss, which never asks `h` to retain displacement) — even
though the status word is `MOTION_RETAINED`. Whether to follow the pre-registered status or the
magnitudes, and which upstream lever to spend the retrain on, is the human's decision, and this
section exists so it is made on all three readings rather than on one word. All three agree on one
thing: M3i's `NO_MOTION` cannot stand as a claim about the representation, because `h` is part of the
latent M3i probed and `h` demonstrably carries motion.

### 10. What this milestone does not claim

- A **linear** probe is a lower bound. Every null here says no linear read-out of that block beats
  the current frame; not that the information is absent under every decoder.
- `two_frame` is **not** an upper bound, and section 7 is the measured proof: `h` exceeds it by two
  orders of magnitude on rotation, because it has the action channel.
- **Backward is not forward.** Everything measured here is motion the posterior has already
  observed. Nothing in this milestone says the M3 gate can pass, and no status here is a statement
  about `gap_closed`.
- The intervals hold the probes fixed across resamples — an interval on the scored sample, not on
  the fit/select/score pipeline. `_block_bootstrap_ci`'s own caveat, inherited.
- The gated base r2 is a 4-column mean, not position alone (section 4).
- It changes no M3 gate, no `aggregate.py`, no `filtering_gain` recorded number, and no M3b–M3i
  verdict.

### 11. Spec divergence, noted rather than resolved silently

Spec 3.5 originally said the per-rung `joint_r2` / `embedding_r2` levels and the selected ridges are
"printed beside the reading" together with the row count at each k and the two control readings.
`retention.txt` prints the two control readings and the row count at `DECISION_K` only; the per-rung
levels and ridges are in every JSON record but nowhere in the printed reading. Spec 3.5 has been
amended, during the final whole-branch review, to say so and to give the reason: the two shipped
tables (`format_ladder`, `format_reading_retention`) are pinned column-for-column by offset tests,
and adding a third table for five numbers per rung that carry no threshold was judged not worth the
risk to those pins. The bridge to M3i's recorded numbers and to `latent_selection_r2` that spec 3.5
names is therefore in the records, not on the face of `retention.txt` — a fact a reader of the
artefact alone would not otherwise know. No number, table, or verdict in this milestone moved because
of this.
