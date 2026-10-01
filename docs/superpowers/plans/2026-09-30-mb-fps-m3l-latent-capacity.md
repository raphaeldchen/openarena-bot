# MB-FPS M3l — Latent Capacity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether the stochastic latent `z` is out of room or was never asked to carry motion, so the project can pick between the bottleneck lever and the objective lever on evidence rather than on a default.

**Architecture:** Two statistics per cell from **one** gather over the nine existing checkpoints — `bits_carried`, the posterior code's per-categorical informations summed (an upper bound on its joint information) against an exact `z_cats × log₂(z_classes)` = 160-bit ceiling, and `frame_share`, `R²(enc(t) → post_probs)` under the three-split discipline. Two substitution controls with answers known by construction, plus a fixture that pins the log base. `RSSM.observe` already returns `post_logits` and `prior_logits`, so the gather change adds no computation and consumes no randomness. No training.

**Tech Stack:** Python 3.12, numpy, torch (MPS), pytest. No new dependencies.

## Global Constraints

Copied verbatim from `docs/superpowers/specs/2026-09-30-mb-fps-m3l-latent-capacity-design.md`. Every task's requirements implicitly include this section.

- **Evaluation only.** No training, no checkpoint written or altered, nothing under `runs/` removed.
- **`redundancy_bits` is reported beside `bits_carried`**, against a `redundancy_floor`, an empirical null measured from the data by rolling each categorical's rows circularly by its own random offset (a permutation would also destroy the autocorrelation of this gather's clustered rows). Both are measured on the DISTRIBUTIONS, not the argmaxes. `bits_carried` sums the PER-CATEGORICAL informations, so it upper-bounds the joint: 32 categoricals copying one 5-bit variable read the full 160 while carrying 5. A reading below a cut is conservative; a reading above one does not establish "capacity in use" unless the redundancy sits near its floor. The ratio is `redundancy_ratio(redundancy, floor)`, which is None for a collapsed code (floor below `RATIO_MIN_FLOOR`). Both companions gate nothing.
- **The ceiling is derived from `RSSMConfig`**, never hardcoded. `CEILING_BITS = RSSMConfig.z_cats * log2(RSSMConfig.z_classes)` = 160.0. A hardcoded 160 would silently disagree with the model if the latent shape changed.
- **`gather_probe_data`'s change must be provably additive.** It feeds the research gate via `src/mbfps/eval/study.py` and `scripts/eval_rollout.py`. Every pre-existing key must be byte-identical, pinned by test. `gain_from_blocks`' ten-key golden output is pinned by `test_gain_from_splits_output_is_byte_identical_after_the_generalisation` — **that test's values are never re-recorded.**
- **`BASE_R2_FLOOR` stays 0.10** and is not re-chosen, gated on **position alone**.
- **`SEEDS_REQUIRED` and `ARMS_REQUIRED` are imported from `retention`**, never re-spelled.
- **`retention.Z_BEARING_RUNGS` is not edited**, and no M3b–M3k verdict is touched. M3l adds a reading; it revises none.
- **Five statuses, this precedence:** `UNRESOLVED_ESTIMATOR`, `UNRESOLVED_BASE`, `SPARE_CAPACITY`, `FRAME_REENCODING`, `CAPACITY_BOUND`.
- **Both cuts are one half**, from one principle: half the derived 160-bit ceiling (**not** half the measured `ceiling_bits`), and half the variance for `frame_share`.
- **`CAPACITY_BOUND` is the fall-through** and the results must report it as arriving by default rather than by evidence.
- **A status requires the bootstrap INTERVAL to clear its cut**, never the point estimate.
- **Exit codes:** 43 `EXIT_ESTIMATOR_BROKEN`, 44 `EXIT_BASE_UNRESOLVED`. 38 is M3i, 39–40 M3j, 41–42 M3k. Highest existing is 42; verified no collision.
- **`capacity.txt` must be byte-identical to a second `--phase read`.**
- **Test command:** `.venv/bin/python -m pytest` from the repo root. There is **no** `pytest` entry point in the venv.
- **Repo conventions:** never `git stash` (shared stack across worktrees); never `rm` under `runs/`; never `git clean -fdx` (it destroys the session ledger at `.superpowers/sdd/`); commit trailer `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` in its own paragraph; write the message to a file and use `git commit -F` when it contains an apostrophe; after mutating a source file to check a test bites, restore it **and** run `find src tests scripts -name __pycache__ -type d -prune -exec rm -rf {} +`.

## File Structure

| file | responsibility |
|---|---|
| `src/mbfps/eval/probe.py` | **modify, additively.** `gather_probe_data` gains `"post_probs"` and `"prior_probs"`, each `(N, z_cats, z_classes)`. Nothing else changes. |
| `src/mbfps/eval/capacity.py` | **create.** Pure: the estimator, `CEILING_BITS`, the controls, the sufficient-statistic bootstrap, Reading G, the table. No torch, no `Path`, no I/O, no record schema. |
| `scripts/latent_capacity.py` | **create.** Owns torch, I/O and the record schema: one gather per cell, both statistics, both controls, the prior companion, measure and read phases. |
| `tests/eval/test_probe.py` | **modify.** The additivity pin. |
| `tests/eval/test_capacity.py` | **create.** |
| `tests/eval/test_latent_capacity_script.py` | **create.** |
| `tests/eval/test_diagnose_dynamics_script.py` | **modify.** 43 and 44 in the exit-status registry. |

The split mirrors `retention.py`/`latent_retention.py` and `width.py`/`latent_width.py`: one pure module per reading, one script owning torch and the schema. If a function needs a `Path` or a `torch.device`, it belongs in the script.

---

### Task 1: `gather_probe_data` yields the posterior distributions

**Files:**
- Modify: `src/mbfps/eval/probe.py` (the window loop around `:270-300`, the return dict at `:315-331`)
- Test: `tests/eval/test_probe.py`

**Interfaces:**
- Consumes: `RSSM.observe`, which already returns `"post_logits"` and `"prior_logits"` at `(B, T, z_cats, z_classes)` — see `_pack` in `src/mbfps/models/rssm.py`.
- Produces, used by Tasks 2 and 4: `gather_probe_data(...)["post_probs"]` and `["prior_probs"]`, each `(N, z_cats, z_classes)` float32, row-aligned with `"latent"`, `"encoder_embedding"`, `"targets"`, `"window"`, `"step"` and `"episode"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_probe.py`:

```python
def test_the_gather_yields_the_posterior_and_prior_distributions():
    """`bits_carried` needs the DISTRIBUTIONS, not the samples. `"latent"`
    carries the one-hot `z` the model drew; the information content of the code
    is a property of the distribution it was drawn from, so M3l needs
    `post_logits` as probabilities.

    Row-aligned with every existing array, and shaped `(N, z_cats, z_classes)`
    rather than flattened, because the estimator sums an entropy per categorical
    and a flattened array cannot tell the 32 groups apart."""
    model, paths, backbone, device = _probe_fixture()
    data = gather_probe_data(model, paths, backbone, device, context=3, horizon=5)

    n = data["latent"].shape[0]
    for key in ("post_probs", "prior_probs"):
        assert key in data, f"the gather does not yield {key}"
        assert data[key].shape == (n, RSSMConfig.z_cats, RSSMConfig.z_classes), (
            f"{key} is {data[key].shape}, expected "
            f"{(n, RSSMConfig.z_cats, RSSMConfig.z_classes)}"
        )
        # Distributions, not logits: each categorical must sum to 1.
        sums = data[key].sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)
        assert (data[key] >= 0.0).all(), f"{key} carries a negative probability"


def test_the_new_gather_keys_change_nothing_that_was_there_before():
    """`gather_probe_data` reports onto the research gate through
    `study.py` and `scripts/eval_rollout.py`, so M3l's change has to be
    provably additive rather than argued to be.

    It is additive structurally: `RSSM.observe` ALREADY computes and returns
    both logit tensors, so collecting them consumes no randomness and adds no
    operation. This pins that claim. Two gathers under one seed, and every
    pre-existing key must be byte-identical -- `assert_array_equal`, not
    `allclose`, because a perturbed RNG draw would change `z` and therefore
    `latent` and `embedding` outright, not slightly."""
    model, paths, backbone, device = _probe_fixture()
    kw = dict(context=3, horizon=5, seed=7)
    first = gather_probe_data(model, paths, backbone, device, **kw)
    second = gather_probe_data(model, paths, backbone, device, **kw)

    inherited = ("latent", "embedding", "encoder_embedding", "targets",
                 "window", "step", "episode")
    assert set(first) == set(inherited) | {"post_probs", "prior_probs"}, (
        "the gather's key set moved beyond the two keys M3l adds"
    )
    for key in inherited:
        np.testing.assert_array_equal(
            first[key], second[key],
            err_msg=f"{key} is not reproducible under one seed; M3l's change "
                    f"may have consumed a random draw",
        )
```

Add to that file's imports if absent: `from mbfps.models.rssm import RSSMConfig`.

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/bin/python -m pytest tests/eval/test_probe.py -q -k "posterior_and_prior or change_nothing"
```
Expected: `test_the_gather_yields_the_posterior_and_prior_distributions` FAILS with `the gather does not yield post_probs`. The second test FAILS on the key-set assertion.

- [ ] **Step 3: Collect the distributions in the window loop**

In `src/mbfps/eval/probe.py`, add to the accumulator list initialisation beside `latents`, `embeddings`, `encoder_embeddings`:

```python
    post_probs, prior_probs = [], []
```

Add this module-level helper just above `gather_probe_data`:

```python
def _sampling_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """The distribution `RSSM._sample` actually draws from, as probabilities.

    The temperature matters because the information content of the code is a
    property of the distribution `z` came from, not of the raw logits. Written
    as a guarded division rather than an unconditional one so the shipped
    `sample_temperature = 1.0` path is bitwise the plain softmax, exactly as
    `RSSM._sample` does it -- M3h added the temperature and took the same care,
    and an unconditional `logits / 1.0` is not guaranteed bitwise identical.
    """
    if temperature != 1.0:
        logits = logits / temperature
    return torch.softmax(logits, dim=-1)
```

Inside the window loop, immediately after `latent = torch.cat([observed["latent"], future["latent"]], dim=1)`:

```python
            # Row-aligned with `latent` by construction: the same two `observe`
            # calls, concatenated on the same axis in the same order. Deriving
            # these from a second pass would reintroduce the row-alignment
            # hazard M3k needed two layered guards for.
            temperature = model.rssm.cfg.sample_temperature
            for key, sink in (("post_logits", post_probs),
                              ("prior_logits", prior_probs)):
                joined = torch.cat([observed[key], future[key]], dim=1)
                sink.append(
                    _sampling_probs(joined, temperature)[0].float().cpu().numpy()
                )
```

In the return dict, after `"episode": np.concatenate(episodes),`:

```python
        # M3l. `(N, z_cats, z_classes)`, NOT flattened: the estimator sums an
        # entropy per categorical and a flattened array cannot tell the groups
        # apart. Additive -- `observe` already computed both, so collecting
        # them consumes no randomness, which a test pins.
        "post_probs": np.concatenate(post_probs),
        "prior_probs": np.concatenate(prior_probs),
```

- [ ] **Step 4: Run them to verify they pass**

```bash
.venv/bin/python -m pytest tests/eval/test_probe.py -q
```
Expected: PASS, including `test_gain_from_splits_output_is_byte_identical_after_the_generalisation` — whose ten values you must not touch.

- [ ] **Step 5: Confirm the gate path is unmoved**

```bash
.venv/bin/python -m pytest tests/eval/test_study.py tests/eval/test_eval_rollout_script.py -q
```
Expected: PASS. These are `gain_from_blocks`' two downstream consumers.

- [ ] **Step 6: Prove the additivity test bites**

Temporarily make the collection consume a draw — insert `torch.rand(1)` immediately before the `for key, sink in ...` loop — and run:

```bash
find src tests scripts -name __pycache__ -type d -prune -exec rm -rf {} +
.venv/bin/python -m pytest tests/eval/test_probe.py -q -k change_nothing
```
Expected: FAIL on a `latent` mismatch. Then restore, clear `__pycache__` again, confirm `git diff` is empty, and re-run to green.

- [ ] **Step 7: Commit**

```bash
git add src/mbfps/eval/probe.py tests/eval/test_probe.py
git commit -m "feat: the gather yields the posterior and prior distributions, additively"
```

---

### Task 2: `capacity.py` — the estimator, the ceiling, the controls

**Files:**
- Create: `src/mbfps/eval/capacity.py`
- Test: `tests/eval/test_capacity.py`

**Interfaces:**
- Consumes: `RSSMConfig.z_cats`, `RSSMConfig.z_classes`; `retention.CONFIDENCE`, `retention.RESAMPLES`.
- Produces, used by Tasks 3, 4 and 5: `CEILING_BITS`, `SPARE_CUT`, `FRAME_CUT`, `entropy_bits(p)`, `bits_carried(probs)`, `floor_bits(probs)`, `ceiling_bits(probs)`, `argmax_marginal_bits(probs)`, `live_classes(probs)`, `PAIR_CEILING_BITS`, `RATIO_MIN_FLOOR`, `redundancy_bits(probs)`, `redundancy_floor(probs, *, seed)` (seed REQUIRED), `redundancy_ratio(redundancy, floor)` (None where the floor is float noise), `EpisodeStats`, `episode_stats(probs, groups)`, `bits_interval(stats, *, resamples, confidence, seed)`.

**The redundancy companion post-dates the code blocks below.** `PAIR_CEILING_BITS`, `RATIO_MIN_FLOOR`, `redundancy_bits`, `redundancy_floor` and `redundancy_ratio`, with their tests, were added afterwards and live in `src/mbfps/eval/capacity.py` and `tests/eval/test_capacity.py`; the blocks below do not carry them.

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_capacity.py`:

```python
"""M3l: is `z` out of room, or was it never asked?"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mbfps.eval.capacity import (
    CEILING_BITS, FRAME_CUT, SPARE_CUT, argmax_marginal_bits, bits_carried,
    bits_interval, ceiling_bits, entropy_bits, episode_stats, floor_bits,
    live_classes,
)
from mbfps.eval.retention import CONFIDENCE, RESAMPLES
from mbfps.models.rssm import RSSMConfig

CATS, CLASSES = RSSMConfig.z_cats, RSSMConfig.z_classes


def _dirichlet(n: int, *, seed: int = 0, alpha: float = 1.0) -> np.ndarray:
    """A plausible posterior: `n` rows of `CATS` independent distributions."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.full(CLASSES, alpha), size=(n, CATS))


def _deterministic_uniform(n: int, *, seed: int = 0) -> np.ndarray:
    """A code that is a DETERMINISTIC function of the row, whose argmax is
    uniform over the classes. Its information content is exactly the ceiling."""
    rng = np.random.default_rng(seed)
    probs = np.zeros((n, CATS, CLASSES))
    for j in range(CATS):
        # Every class used exactly n/CLASSES times, so the marginal is uniform.
        classes = np.repeat(np.arange(CLASSES), n // CLASSES)
        rng.shuffle(classes)
        probs[np.arange(n), j, classes] = 1.0
    return probs


def test_the_ceiling_is_derived_from_the_model_config_not_re_spelled():
    """A hardcoded 160.0 would silently disagree with the model if the latent
    shape changed, and every cut in Reading G is a fraction of it."""
    assert CEILING_BITS == RSSMConfig.z_cats * math.log2(RSSMConfig.z_classes)
    assert CEILING_BITS == pytest.approx(160.0)
    assert SPARE_CUT == CEILING_BITS / 2, "the spare cut is HALF THE DERIVED ceiling"
    assert FRAME_CUT == 0.5


def test_a_deterministic_uniform_code_reads_exactly_log2_of_the_classes():
    """THE LOG-BASE CHECK, and the only one that catches nats.

    Measured: in nats the floor control still reads 0.000000 and both ceiling
    routes still agree exactly, and nats reads BELOW the bits ceiling so no
    inequality catches it either. Only a case with a known ABSOLUTE value does.
    A deterministic code with a uniform marginal carries exactly
    `log2(z_classes)` bits per categorical -- 5.000, where nats reads 3.466."""
    probs = _deterministic_uniform(CLASSES * 20)
    per_categorical = bits_carried(probs) / CATS
    assert per_categorical == pytest.approx(math.log2(CLASSES), abs=1e-9), (
        f"read {per_categorical:.4f} bits per categorical; "
        f"log2({CLASSES}) = {math.log2(CLASSES):.4f}, "
        f"ln({CLASSES}) = {math.log(CLASSES):.4f}"
    )
    assert bits_carried(probs) == pytest.approx(CEILING_BITS, abs=1e-9)


def test_the_floor_control_reads_zero_to_summation_order():
    """Every row replaced by the marginal: `H(m) - mean_n H(m)` is identically
    zero in exact arithmetic, so any reading away from zero is an arithmetic
    defect -- a missing normalisation, or a mean over the wrong axis. Computed
    through the real estimator on real-shaped data, not a fixture.

    NOT bit-exact, and the tolerance is the point. Measured: +5.68e-14 at 500
    rows, -1.42e-13 at 2,000, +5.40e-13 at the ~11,000 a real cell carries --
    float summation order, not a defect. This project has been bitten by the
    other reading of this: an M3h sweep refused on an 8.527e-14 mismatch that
    was summation order, costing a debugging cycle. 1e-9 is four orders above
    the observed drift and eleven below anything a real defect would produce."""
    for n in (500, 2000, 11000):
        assert floor_bits(_dirichlet(n, seed=1)) == pytest.approx(0.0, abs=1e-9)


def test_the_two_ceiling_routes_agree():
    """`ceiling_bits` substitutes a one-hot at each row's argmax and runs the
    real estimator; `argmax_marginal_bits` histograms the argmax INDICES. Two
    routes, one answer. This pins the routing -- a mean over the wrong axis, or
    a substitution that leaks a row's own distribution, breaks the agreement --
    but NOT the log base, which the test above owns."""
    probs = _dirichlet(500, seed=2)
    assert ceiling_bits(probs) == pytest.approx(argmax_marginal_bits(probs), abs=1e-9)


def test_the_only_bound_on_bits_carried_is_the_derived_ceiling():
    """THE TWO inequalities a refusal may rest on, both theorems:

        -1e-9 <= bits_carried <= CEILING_BITS + 1e-9
         0    <= ceiling_bits <= CEILING_BITS + 1e-9

    `bits <= ceiling_bits` IS NOT ONE OF THEM. `ceiling_bits` is the information
    content of the ARGMAX PATTERN, `bits_carried` that of the DISTRIBUTION, and a
    code whose argmax never moves while its tail varies reads 7.5207 against
    0.0000. Write the companion test that documents that counterexample, so the
    refusal cannot be re-added."""
    for n in (500, 2000, 11000):
        probs = _dirichlet(n, seed=3)
        assert -1e-9 <= bits_carried(probs) <= CEILING_BITS + 1e-9
        assert 0.0 <= ceiling_bits(probs) <= CEILING_BITS + 1e-9


def test_a_code_that_ignores_its_input_carries_no_bits():
    """The other end of the scale, and the posterior-collapse case: every row
    the SAME distribution means the code says nothing about which row it is."""
    one = _dirichlet(1, seed=4)[0]
    probs = np.broadcast_to(one, (300, CATS, CLASSES)).copy()
    assert bits_carried(probs) == pytest.approx(0.0, abs=1e-12)


def test_live_classes_counts_the_columns_that_vary():
    """A class that never varies across the scored rows is capacity the code is
    not using, and R^2 is undefined for it, so `frame_share` must exclude it.
    The count is reported beside the verdict: a CAPACITY_BOUND read printed
    next to a low `live_classes` would be self-contradicting."""
    probs = _dirichlet(300, seed=8)
    assert live_classes(probs) == CATS * CLASSES, "every class varies here"

    # A DIRICHLET base, not the one-hot one: zeroing a class in a one-hot code
    # empties every row whose one-hot WAS that class, and renormalising then
    # divides by zero. Measured on the Dirichlet base, the smallest row sum
    # after zeroing is 0.7242, so no row is emptied and no warning is raised.
    dead = probs.copy()
    dead[:, :, 7] = 0.0  # class 7 never varies anywhere: CATS dead columns
    dead /= dead.sum(axis=-1, keepdims=True)
    assert live_classes(dead) == CATS * CLASSES - CATS


def test_the_interval_from_sufficient_statistics_matches_a_direct_resample():
    """`bits_carried` is a function of per-episode sufficient statistics --
    the summed probabilities, the summed row entropies and the row count -- so
    the bootstrap resamples 24 small arrays instead of re-slicing an
    (11000, 32, 32) one. Measured: 0.02 ms against 54.89 ms per resample, a
    2601x speedup, and that is what makes the whole run ~30 minutes.

    It is EXACT, not an approximation -- but exact up to summation ORDER, which
    is why this compares with `approx` at 1e-9 rather than `==`. This project
    has been bitten by claiming bit-identity where only near-identity holds: an
    M3h sweep refused on a 8.527e-14 mismatch that was summation order, not a
    defect."""
    probs = _dirichlet(600, seed=5)
    rng = np.random.default_rng(11)
    groups = rng.integers(0, 12, size=600)
    stats = episode_stats(probs, groups)

    picked = rng.integers(0, 12, size=12)
    rows = np.concatenate([np.flatnonzero(groups == g) for g in picked])
    direct = bits_carried(probs[rows])
    from mbfps.eval.capacity import _bits_from_stats
    assert _bits_from_stats(stats, picked) == pytest.approx(direct, abs=1e-9)


def test_the_interval_brackets_the_point_estimate_and_needs_two_episodes():
    probs = _dirichlet(400, seed=6)
    groups = np.random.default_rng(12).integers(0, 10, size=400)
    stats = episode_stats(probs, groups)
    out = bits_interval(stats, resamples=200, confidence=CONFIDENCE, seed=0)
    assert out["bits"] == pytest.approx(bits_carried(probs), abs=1e-9)
    assert out["ci_low"] <= out["bits"] <= out["ci_high"]
    assert out["n_episodes"] == 10
    assert out["confidence"] == CONFIDENCE

    with pytest.raises(ValueError, match="at least two"):
        bits_interval(episode_stats(probs, np.zeros(400, dtype=int)),
                      resamples=10, confidence=CONFIDENCE, seed=0)


def test_the_estimator_refuses_rows_that_are_not_distributions():
    """A caller handing logits instead of probabilities would get a number, not
    an error, and it would look plausible. Refused at the point the array first
    becomes a reading."""
    probs = _dirichlet(50, seed=7)
    bad = probs.copy()
    bad[3, 5, :] *= 2.0
    with pytest.raises(ValueError, match="sum to 1"):
        bits_carried(bad)

    negative = probs.copy()
    negative[1, 2, 0] = -0.1
    negative[1, 2, 1] += 0.1
    with pytest.raises(ValueError, match="negative"):
        bits_carried(negative)

    with pytest.raises(ValueError, match="z_cats"):
        bits_carried(probs.reshape(50, CATS * CLASSES))
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/bin/python -m pytest tests/eval/test_capacity.py -q
```
Expected: collection error, `ModuleNotFoundError: No module named 'mbfps.eval.capacity'`.

- [ ] **Step 3: Write the module**

Create `src/mbfps/eval/capacity.py`:

```python
"""M3l: how many bits does the posterior code carry, and about what?

M3j established the study's largest, most unanimous effect -- `h` carries
observed motion and `z` does not, 9/9 in both targets. But that observation is
COMMON TO BOTH LEVERS: "`z` lacks the capacity" and "`z` was never asked"
predict it equally. M3k was built to separate them and returned
INDISTINGUISHABLE at effect/noise 0.22, and more seeds make that worse rather
than better -- a majority bar recedes as n grows whenever the per-seed rate is
below one half.

So this module asks the question that separates them directly, and asks it as a
LEVEL AGAINST AN EXACT CEILING rather than as a difference of two fitted
quantities. That is the property M3k lacked. Every win this project has had has
been a level (the anchors, compared with `==` against a known value); every
stall has been a difference.

Pure: numpy only. No torch, no `Path`, no I/O, no record schema -- those live in
`scripts/latent_capacity.py`.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from mbfps.eval.retention import CONFIDENCE, RESAMPLES
from mbfps.models.rssm import RSSMConfig

CEILING_BITS: float = RSSMConfig.z_cats * math.log2(RSSMConfig.z_classes)
"""The most the posterior code can carry: one `log2(z_classes)` per categorical.

DERIVED from the model's own config rather than re-spelled. A hardcoded 160.0
would silently disagree with the model if the latent shape changed, and both of
Reading G's cuts are fractions of this."""

SPARE_CUT: float = CEILING_BITS / 2
"""Below this, most of the capacity is idle.

HALF THE DERIVED CEILING, not half the measured `ceiling_bits` -- otherwise a
code that collapsed its own ceiling could satisfy the threshold by degenerating,
which is the opposite of what the status detects. The half is one principle's
third instance: M3k's seed bar is the weakest MAJORITY of seeds, its derivation
at five seeds gives three, and here a majority of the capacity. The claim
"capacity is the binding constraint" requires that most of the capacity is in
use."""

FRAME_CUT: float = 0.5
"""Above this, most of the code's variance is explained by the current frame,
which is what the embedding loss asks for. The same half, the same principle."""


def entropy_bits(p: np.ndarray) -> np.ndarray:
    """Shannon entropy in BITS over the last axis, with `0 log 0 = 0`.

    `np.where` guards the log twice on purpose: once to select, once to keep the
    argument positive, because `np.where` evaluates both branches and
    `log2(0)` would warn before being discarded. Test output must be pristine.
    """
    safe = np.where(p > 0.0, p, 1.0)
    return -np.where(p > 0.0, p * np.log2(safe), 0.0).sum(axis=-1)


def _require_distributions(probs: np.ndarray) -> np.ndarray:
    """`probs` as `(N, z_cats, z_classes)` of proper distributions, or a refusal.

    A caller handing logits instead of probabilities would otherwise get a
    number, and it would look plausible against a 160-bit ceiling rather than
    obviously wrong.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 3 or probs.shape[1:] != (RSSMConfig.z_cats, RSSMConfig.z_classes):
        raise ValueError(
            f"probs must be (N, z_cats, z_classes) = "
            f"(N, {RSSMConfig.z_cats}, {RSSMConfig.z_classes}), got {probs.shape}: "
            "a flattened array cannot tell the categorical groups apart, and the "
            "estimator sums one entropy per group"
        )
    if (probs < 0.0).any():
        raise ValueError(
            "probs carries a negative value, so it is not a distribution; "
            "logits were probably handed in place of probabilities"
        )
    sums = probs.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-5):
        raise ValueError(
            f"each categorical must sum to 1, got {sums.min():.6f}..{sums.max():.6f}; "
            "logits were probably handed in place of probabilities"
        )
    return probs


def bits_carried(probs: np.ndarray) -> float:
    """`Sum_j I(z_j ; h, enc(t))` in bits: the per-categorical informations summed.

    AN UPPER BOUND on the code's joint information, not the joint itself:
    redundancy across categoricals is counted once per categorical.

    `H(marginal) - E_n H(row)`, summed over the categoricals. Exact from the
    distributions -- no sampling.

    NOT "information about the frame": the posterior conditions on `h` AND
    `enc(t)`, and this measures how much the code varies across rows for any
    reason. `frame_share` is what separates the two sources.
    """
    probs = _require_distributions(probs)
    marginal = probs.mean(axis=0)
    return float(entropy_bits(marginal).sum() - entropy_bits(probs).sum(axis=1).mean())


def floor_bits(probs: np.ndarray) -> float:
    """Every row replaced by the marginal. EXACTLY 0.0 by construction.

    A substitution into the real estimator on the real data, not a fixture --
    the same shape of control as M3k's anchors, whose answer is known in
    advance. `H(m) - mean_n H(m)` is identically zero, so any nonzero reading is
    an arithmetic defect. It does NOT pin the log base: measured, nats also
    reads 0.000000.
    """
    probs = _require_distributions(probs)
    marginal = probs.mean(axis=0)
    return bits_carried(np.broadcast_to(marginal, probs.shape).copy())


def ceiling_bits(probs: np.ndarray) -> float:
    """Every row replaced by a one-hot at its argmax: the most this code's
    argmax pattern could carry. `argmax_marginal_bits` is the second route."""
    probs = _require_distributions(probs)
    onehot = np.zeros_like(probs)
    np.put_along_axis(onehot, probs.argmax(axis=-1)[..., None], 1.0, axis=-1)
    return bits_carried(onehot)


def argmax_marginal_bits(probs: np.ndarray) -> float:
    """`ceiling_bits` again, from the argmax INDICES via a histogram.

    Two independent routes to one answer pins the routing -- a mean over the
    wrong axis, or a substitution that leaks a row's own distribution, breaks
    the agreement. It does not pin the log base; both routes agree in nats too.
    """
    probs = _require_distributions(probs)
    idx = probs.argmax(axis=-1)
    n = idx.shape[0]
    total = 0.0
    for j in range(idx.shape[1]):
        share = np.bincount(idx[:, j], minlength=probs.shape[-1]) / n
        total += float(entropy_bits(share))
    return total


def live_classes(probs: np.ndarray) -> int:
    """How many of the `z_cats * z_classes` columns vary across the rows.

    A class that never varies is capacity the code is not using, and `R^2` is
    undefined for it, so `frame_share` excludes it. Reported beside the verdict:
    a CAPACITY_BOUND read printed next to a low count would be
    self-contradicting.
    """
    probs = _require_distributions(probs)
    flat = probs.reshape(probs.shape[0], -1)
    return int((flat.std(axis=0) > 0.0).sum())


@dataclasses.dataclass(frozen=True)
class EpisodeStats:
    """Per-episode sufficient statistics for `bits_carried`.

    `bits_carried` is a function of exactly three per-episode quantities: the
    summed probabilities, the summed row entropies and the row count. So the
    bootstrap resamples these small arrays instead of re-slicing an
    (11000, 32, 32) one -- measured, 0.02 ms against 54.89 ms per resample, a
    2601x speedup, and the reason the whole run is ~30 minutes rather than ~9.

    Exact, not an approximation -- but exact up to summation ORDER, so a test
    compares with `approx` rather than `==`. An M3h sweep once refused on an
    8.527e-14 mismatch that was summation order and not a defect.
    """

    prob_sum: np.ndarray   # (n_episodes, z_cats, z_classes)
    entropy_sum: np.ndarray  # (n_episodes,)
    rows: np.ndarray       # (n_episodes,)
    labels: np.ndarray     # (n_episodes,)


def episode_stats(probs: np.ndarray, groups: np.ndarray) -> EpisodeStats:
    """`EpisodeStats` for `probs` clustered by `groups`, one entry per label."""
    probs = _require_distributions(probs)
    groups = np.asarray(groups)
    if groups.shape != (probs.shape[0],):
        raise ValueError(
            f"groups must be one label per row; got {groups.shape} for "
            f"{probs.shape[0]} rows"
        )
    row_entropy = entropy_bits(probs).sum(axis=1)
    labels = np.unique(groups)
    masks = [groups == g for g in labels]
    return EpisodeStats(
        prob_sum=np.stack([probs[m].sum(axis=0) for m in masks]),
        entropy_sum=np.array([row_entropy[m].sum() for m in masks]),
        rows=np.array([int(m.sum()) for m in masks], dtype=np.float64),
        labels=labels,
    )


def _bits_from_stats(stats: EpisodeStats, picked: np.ndarray) -> float:
    """`bits_carried` over the episodes `picked` names, from the statistics."""
    total = stats.rows[picked].sum()
    marginal = stats.prob_sum[picked].sum(axis=0) / total
    return float(entropy_bits(marginal).sum() - stats.entropy_sum[picked].sum() / total)


def bits_interval(
    stats: EpisodeStats, *, resamples: int = RESAMPLES,
    confidence: float = CONFIDENCE, seed: int = 0,
) -> dict:
    """`bits_carried` with an episode-clustered percentile interval.

    Episodes, not windows: windows are cut non-overlapping but several windows
    from one trajectory are not independent observations, and every reading from
    M3e onward clusters on episodes.
    """
    n_episodes = stats.labels.size
    if n_episodes < 2:
        raise ValueError(
            "a bootstrap interval needs at least two resampling units (distinct "
            f"episode labels); got {n_episodes}"
        )
    index = np.arange(n_episodes)
    rng = np.random.default_rng(seed)
    draws = np.array([
        _bits_from_stats(stats, rng.choice(index, size=n_episodes, replace=True))
        for _ in range(resamples)
    ])
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return {
        "bits": _bits_from_stats(stats, index),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": confidence,
        "n_episodes": int(n_episodes),
    }
```

- [ ] **Step 4: Run them to verify they pass**

```bash
.venv/bin/python -m pytest tests/eval/test_capacity.py -q
```
Expected: `11 passed`, output pristine — no runtime warnings from `log2`.

- [ ] **Step 5: Prove the log-base test bites**

Change `np.log2` to `np.log` in `entropy_bits`, clear `__pycache__`, and run:

```bash
find src tests scripts -name __pycache__ -type d -prune -exec rm -rf {} +
.venv/bin/python -m pytest tests/eval/test_capacity.py -q
```
Expected: `test_a_deterministic_uniform_code_reads_exactly_log2_of_the_classes` FAILS reporting 3.4657 against 5.0000. Confirm the floor and two-route tests still **pass** under the mutant — that is the point of having a third check. Restore, clear `__pycache__`, verify `git diff` is empty, re-run to green.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/capacity.py tests/eval/test_capacity.py
git commit -m "feat: the bits estimator, its exact ceiling, and three checks that catch different things"
```

---

### Task 3: `capacity.py` — Reading G and its table

**Files:**
- Modify: `src/mbfps/eval/capacity.py` (append)
- Test: `tests/eval/test_capacity.py` (append)

**Interfaces:**
- Consumes: Task 2's `CEILING_BITS`, `SPARE_CUT`, `FRAME_CUT`; `retention.ARMS_REQUIRED`, `SEEDS_REQUIRED`, `BASE_R2_FLOOR`, `BaseControl`.
- Produces, used by Tasks 4 and 5: `CapacityArm` (fields `bits, bits_low, bits_high, frame_share, frame_low, frame_high, live, redundancy_bits, redundancy_floor, seeds_spare, seeds_frame, seeds_total`; the two redundancy fields are seed means like `bits`, decide nothing, and the ratio is derived from them with `redundancy_ratio`), `CapacityInputs` (`arms, base, controls, clusters, rows`), `CapacityStatus` (`status, rule, arms_spare, arms_frame, base_failed, controls_failed`), `capacity_arm(seeds)`, `reading_capacity(inputs)`, `READING_COLUMNS`, `READING_WIDTHS`, `format_reading_capacity(reading, inputs)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_capacity.py`:

```python
# ---------------------------------------------------------------------------
# Reading G
# ---------------------------------------------------------------------------

from mbfps.eval.capacity import (  # noqa: E402
    READING_COLUMNS, READING_WIDTHS, CapacityArm, CapacityInputs, capacity_arm,
    format_reading_capacity, reading_capacity,
)
from mbfps.eval.retention import (  # noqa: E402
    ARMS_REQUIRED, BASE_R2_FLOOR, BaseControl, SEEDS_REQUIRED,
)

ARMS = ("frozen_ssl", "pixel_ae", "random_vit")
SEEDS_TOTAL = 3
BASE_HOLDS, BASE_FAILS = 0.65, 0.05

SPARE = {"bits": 40.0, "ci_low": 30.0, "ci_high": 50.0}
FULL = {"bits": 150.0, "ci_low": 140.0, "ci_high": 158.0}


def _arm(bits: dict, frame: tuple[float, float, float], *, live: int = 1024,
         redundancy: tuple[float, float] = (0.065, 0.064),
         seeds_spare: int | None = None, seeds_frame: int | None = None,
         seeds_total: int = SEEDS_TOTAL) -> CapacityArm:
    """An arm whose bits interval is `bits` and frame interval is `frame`.

    `seeds_*` default to ALL seeds when the corresponding interval clears and
    NONE when it does not -- the 0-or-3 shape most fixtures want. Pass them to
    sit a fixture ON a threshold, which is the only way a bar gets pinned:
    M3k shipped four off-by-one mutations because every fixture was 0 or 3.
    """
    share, low, high = frame
    spare = seeds_total if bits["ci_high"] < SPARE_CUT else 0
    framey = seeds_total if low > FRAME_CUT else 0
    return CapacityArm(
        bits=bits["bits"], bits_low=bits["ci_low"], bits_high=bits["ci_high"],
        frame_share=share, frame_low=low, frame_high=high, live=live,
        redundancy_bits=redundancy[0], redundancy_floor=redundancy[1],
        seeds_spare=spare if seeds_spare is None else seeds_spare,
        seeds_frame=framey if seeds_frame is None else seeds_frame,
        seeds_total=seeds_total,
    )


def _base(r2: float, seeds_total: int = SEEDS_TOTAL) -> BaseControl:
    return BaseControl(r2=r2, seeds_clear=seeds_total if r2 > BASE_R2_FLOOR else 0,
                       seeds_total=seeds_total)


def _inputs(arms=None, *, base_r2=BASE_HOLDS, controls=None) -> CapacityInputs:
    arms = arms if arms is not None else {
        a: _arm(FULL, (0.20, 0.10, 0.30)) for a in ARMS
    }
    per_arm = base_r2 if isinstance(base_r2, dict) else dict.fromkeys(arms, base_r2)
    return CapacityInputs(
        arms=arms,
        base={a: _base(per_arm.get(a, BASE_HOLDS)) for a in arms},
        controls=controls if controls is not None else dict.fromkeys(arms, True),
        clusters=24, rows=11221,
    )


def test_most_of_the_capacity_idle_reads_spare_and_names_the_objective_lever():
    reading = reading_capacity(_inputs({a: _arm(SPARE, (0.20, 0.10, 0.30)) for a in ARMS}))
    assert reading.status == "SPARE_CAPACITY"
    assert reading.arms_spare == ARMS
    assert "objective" in reading.rule


def test_a_code_that_re_encodes_the_frame_reads_frame_reencoding():
    reading = reading_capacity(_inputs({a: _arm(FULL, (0.80, 0.70, 0.90)) for a in ARMS}))
    assert reading.status == "FRAME_REENCODING"
    assert reading.arms_frame == ARMS
    assert "objective" in reading.rule


def test_full_capacity_carrying_something_else_falls_through_to_capacity_bound():
    """CAPACITY_BOUND is the FALL-THROUGH and its rule must say so. The two
    objective-lever statuses each have to clear an interval; this one is what
    remains, so it is the EASIEST status to reach -- which is backwards, since
    it is the lever M3j's h-vs-z evidence already favours."""
    reading = reading_capacity(_inputs({a: _arm(FULL, (0.20, 0.10, 0.30)) for a in ARMS}))
    assert reading.status == "CAPACITY_BOUND"
    assert reading.arms_spare == () and reading.arms_frame == ()
    assert "BY DEFAULT rather than by evidence" in reading.rule
    assert "bottleneck" in reading.rule


def test_a_broken_control_outranks_every_reading():
    """The estimator's own checks gate everything: a reading taken from an
    estimator that missed a known answer is not a weaker reading, it is not a
    reading. Both-true fixture -- the base fails at the same time -- so
    swapping the two checks in the source is caught."""
    reading = reading_capacity(_inputs(
        {a: _arm(SPARE, (0.20, 0.10, 0.30)) for a in ARMS},
        base_r2=BASE_FAILS, controls={**dict.fromkeys(ARMS, True), "pixel_ae": False},
    ))
    assert reading.status == "UNRESOLVED_ESTIMATOR"
    assert reading.controls_failed == ("pixel_ae",)
    assert reading.base_failed == ARMS, "the base failure is REPORTED, not hidden"
    assert reading.arms_spare == () and reading.arms_frame == ()


def test_a_failed_base_outranks_a_reading_but_not_a_broken_control():
    reading = reading_capacity(_inputs(
        {a: _arm(SPARE, (0.20, 0.10, 0.30)) for a in ARMS}, base_r2=BASE_FAILS,
    ))
    assert reading.status == "UNRESOLVED_BASE"
    assert reading.base_failed == ARMS
    assert reading.controls_failed == ()


@pytest.mark.parametrize("status, arms, controls, base_r2, spare, framey", [
    ("UNRESOLVED_ESTIMATOR", {a: _arm(SPARE, (0.8, 0.7, 0.9)) for a in ARMS},
     {**dict.fromkeys(ARMS, True), "pixel_ae": False}, {"pixel_ae": BASE_FAILS}, (), ()),
    ("UNRESOLVED_BASE", {a: _arm(SPARE, (0.8, 0.7, 0.9)) for a in ARMS},
     None, dict.fromkeys(ARMS, BASE_FAILS), (), ()),
    ("SPARE_CAPACITY", {a: _arm(SPARE, (0.8, 0.7, 0.9)) for a in ARMS},
     None, {"pixel_ae": BASE_FAILS}, ARMS, ARMS),
    ("FRAME_REENCODING", {a: _arm(FULL, (0.8, 0.7, 0.9)) for a in ARMS},
     None, {"pixel_ae": BASE_FAILS}, (), ARMS),
    ("CAPACITY_BOUND", {a: _arm(FULL, (0.2, 0.1, 0.3)) for a in ARMS},
     None, {"pixel_ae": BASE_FAILS}, (), ()),
])
def test_every_branch_reports_all_four_of_its_tuples(
    status, arms, controls, base_r2, spare, framey,
):
    """M3k's final review found `arms_down=()` hardcoded in the deciding branch
    surviving 82 tests, because the parametrize that pinned `base_failed` used
    one-directional arms. So every case here carries a partial base failure AND
    arms that clear in both senses where the status allows, and all four
    reported tuples are asserted in all five branches."""
    reading = reading_capacity(_inputs(arms, base_r2=base_r2, controls=controls))
    assert reading.status == status
    assert reading.arms_spare == spare
    assert reading.arms_frame == framey
    expected_base = tuple(sorted(a for a, r in base_r2.items() if r <= BASE_R2_FLOOR))
    assert reading.base_failed == expected_base
    assert reading.controls_failed == (
        () if controls is None else tuple(sorted(a for a, ok in controls.items() if not ok))
    )


def test_exactly_arms_required_arms_read_and_one_short_does_not():
    """The bar IS Reading G. M3k shipped four off-by-one mutations on exactly
    this because no fixture sat on a threshold."""
    at_bar = {"frozen_ssl": _arm(SPARE, (0.2, 0.1, 0.3)),
              "pixel_ae": _arm(SPARE, (0.2, 0.1, 0.3)),
              "random_vit": _arm(FULL, (0.2, 0.1, 0.3))}
    reading = reading_capacity(_inputs(at_bar))
    assert len(reading.arms_spare) == ARMS_REQUIRED
    assert reading.status == "SPARE_CAPACITY"

    one_short = {**at_bar, "pixel_ae": _arm(FULL, (0.2, 0.1, 0.3))}
    assert len(reading_capacity(_inputs(one_short)).arms_spare) == ARMS_REQUIRED - 1
    assert reading_capacity(_inputs(one_short)).status == "CAPACITY_BOUND"


def test_an_arm_at_exactly_seeds_required_clears_and_one_short_does_not():
    assert _arm(SPARE, (0.2, 0.1, 0.3), seeds_spare=SEEDS_REQUIRED).clears_spare()
    assert not _arm(SPARE, (0.2, 0.1, 0.3),
                    seeds_spare=SEEDS_REQUIRED - 1).clears_spare()


def test_the_cut_is_on_the_interval_not_the_point_estimate():
    """A point estimate below the cut whose interval straddles it must NOT
    clear. Half the capacity idle is a claim, and a claim needs an interval."""
    straddling = {"bits": 70.0, "ci_low": 60.0, "ci_high": 95.0}
    assert straddling["bits"] < SPARE_CUT < straddling["ci_high"]
    arm = _arm(straddling, (0.2, 0.1, 0.3))
    assert arm.seeds_spare == 0, "the interval straddles the cut, so nothing clears"


def test_capacity_arm_refuses_a_measurement_it_cannot_read():
    """Non-finite, inverted, and a point estimate outside its own interval --
    each would otherwise pass silently and read as a null, because every
    comparison against NaN is False. `retention.rung_arm` refuses the same
    three for the same reason."""
    ok = {"bits": 40.0, "ci_low": 30.0, "ci_high": 50.0,
          "frame_share": 0.2, "frame_low": 0.1, "frame_high": 0.3, "live": 1024,
          "redundancy_bits": 0.065, "redundancy_floor": 0.064}
    assert capacity_arm([ok] * SEEDS_REQUIRED).seeds_total == SEEDS_REQUIRED

    for field, value, match in (
        ("bits", float("nan"), "non-finite"),
        ("ci_low", 60.0, "inverted"),
        ("bits", 90.0, "outside its own interval"),
        ("redundancy_floor", float("nan"), "non-finite"),
    ):
        with pytest.raises(ValueError, match=match):
            capacity_arm([ok, {**ok, field: value}, ok])

    with pytest.raises(ValueError, match="at least"):
        capacity_arm([ok])

    missing = {k: v for k, v in ok.items() if k != "redundancy_bits"}
    with pytest.raises(ValueError, match="missing redundancy_bits"):
        capacity_arm([ok, missing, ok])


def test_the_arm_carries_the_redundancy_companion_and_it_gates_nothing():
    """Task 6 must report the ratio beside `CAPACITY_BOUND`, derived from the
    records, so the arm has to carry both numbers -- as the seed MEANS, like
    `bits`. And they are a companion: wild values change no status."""
    import dataclasses

    ok = {"bits": 40.0, "ci_low": 30.0, "ci_high": 50.0, "frame_share": 0.2,
          "frame_low": 0.1, "frame_high": 0.3, "live": 1024}
    arm = capacity_arm([
        {**ok, "redundancy_bits": 0.10, "redundancy_floor": 0.05},
        {**ok, "redundancy_bits": 0.30, "redundancy_floor": 0.07},
        {**ok, "redundancy_bits": 0.20, "redundancy_floor": 0.06},
    ])
    assert arm.redundancy_bits == pytest.approx(0.20)
    assert arm.redundancy_floor == pytest.approx(0.06)

    base = _inputs()
    wild = _inputs({a: dataclasses.replace(x, redundancy_bits=9.0, redundancy_floor=0.0)
                    for a, x in base.arms.items()})
    assert reading_capacity(wild) == reading_capacity(base)


def test_the_table_pins_every_column_to_the_arm_it_came_from():
    """This project has shipped a table whose caption disagreed with its
    columns THREE times; the worst printed an `up` tally beside a verdict read
    from the `down` tally, so the natural misreading was the exact opposite of
    the truth. Every column's VALUE is pinned to its field, on two rows whose
    numbers all differ, so a neighbour swap is caught."""
    arms = {"frozen_ssl": _arm(SPARE, (0.21, 0.11, 0.31), live=1000),
            "pixel_ae": _arm(FULL, (0.82, 0.72, 0.92), live=1024),
            "random_vit": _arm(FULL, (0.43, 0.33, 0.53), live=900)}
    inputs = _inputs(arms)
    text = format_reading_capacity(reading_capacity(inputs), inputs)

    for name, arm, clears in (("frozen_ssl", arms["frozen_ssl"], "spare"),
                              ("pixel_ae", arms["pixel_ae"], "frame")):
        row = _parse_row(_table_line(text, name))
        assert row["arm"] == name
        assert float(row["bits"]) == pytest.approx(arm.bits, abs=5e-4)
        assert float(row["ci_low"]) == pytest.approx(arm.bits_low, abs=5e-4)
        assert float(row["ci_high"]) == pytest.approx(arm.bits_high, abs=5e-4)
        assert float(row["frame"]) == pytest.approx(arm.frame_share, abs=5e-5)
        assert int(row["live"]) == arm.live
        assert row["clears"] == clears
    assert _parse_row(_table_line(text, "random_vit"))["clears"] == "no"


def test_the_caption_uses_the_inputs_and_the_derived_ceiling_not_literals():
    """`_inputs()` always builds clusters=24, rows=11221, three arms, three
    seeds -- so a caption printing those as literals would pass every other
    test. Built directly, with every number different and both cuts patched."""
    import mbfps.eval.capacity as capacity

    names = ("frozen_ssl", "pixel_ae", "random_vit", "clip")
    arms = {a: _arm(FULL, (0.2, 0.1, 0.3), seeds_total=5) for a in names}
    inputs = CapacityInputs(
        arms=arms, base={a: _base(BASE_HOLDS, 5) for a in names},
        controls=dict.fromkeys(names, True), clusters=7, rows=123,
    )
    with patch.object(capacity, "CEILING_BITS", 320.0), \
         patch.object(capacity, "SPARE_CUT", 160.0):
        caption = format_reading_capacity(
            reading_capacity(inputs), inputs
        ).splitlines()[0]
    assert "7 clusters" in caption and "123 rows" in caption
    assert "24 clusters" not in caption and "11221 rows" not in caption
    assert "320" in caption and "160" in caption
    assert "of 4 arms" in caption and "of 5 seeds" in caption
    assert "of 3 arms" not in caption and "of 3 seeds" not in caption


def test_reading_columns_and_widths_stay_the_same_length():
    assert len(READING_COLUMNS) == len(READING_WIDTHS)
    assert READING_COLUMNS == ("arm", "bits", "ci_low", "ci_high", "frame",
                               "live", "clears")
```

Add to the file's imports: `from unittest.mock import patch`, and the two row helpers:

```python
def _table_line(text: str, first_token: str) -> str:
    """The one table row whose first token is `first_token`.

    Selected by token and asserted unique, NOT by indentation: the base-control
    and verdict lines are also indented and also contain every arm name, so an
    indentation filter would find the wrong line whenever the row order changed.
    """
    hits = [ln for ln in text.splitlines() if ln.split()[:1] == [first_token]]
    assert len(hits) == 1, f"expected exactly one row for {first_token}, got {len(hits)}"
    return hits[0]


def _parse_row(line: str) -> dict[str, str]:
    """A table row split back into `READING_COLUMNS` by the shipped widths."""
    out, at = {}, 2
    for name, width in zip(READING_COLUMNS, READING_WIDTHS, strict=True):
        out[name] = line[at:at + width].strip()
        at += width
    return out
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/bin/python -m pytest tests/eval/test_capacity.py -q -k "spare or reencoding or capacity_bound or branch or bar or table or caption or columns_and_widths or refuses_a_measurement or interval_not_the_point"
```
Expected: `ImportError: cannot import name 'CapacityArm'`.

- [ ] **Step 3: Append Reading G to the module**

Append to `src/mbfps/eval/capacity.py`:

```python
from mbfps.eval.retention import (
    ARMS_REQUIRED, BASE_R2_FLOOR, BaseControl, SEEDS_REQUIRED,
)

READING_COLUMNS: tuple[str, ...] = (
    "arm", "bits", "ci_low", "ci_high", "frame", "live", "clears",
)
READING_WIDTHS: tuple[int, ...] = (13, 10, 10, 10, 8, 7, 9)


@dataclasses.dataclass(frozen=True)
class CapacityArm:
    """One arm's reading, summarised across its seeds.

    The intervals are the CONSERVATIVE hull -- the lowest low and the highest
    high across the arm's seeds -- so an arm is never credited with a bound only
    its luckiest seed reached. `bits` and `frame_share` are the seed means.

    Neither point estimate decides anything: the two `seeds_*` tallies do, and
    they are counted per seed against the cuts before ever being averaged.
    """

    bits: float
    bits_low: float
    bits_high: float
    frame_share: float
    frame_low: float
    frame_high: float
    live: int
    redundancy_bits: float   # seed mean; a companion that decides nothing
    redundancy_floor: float  # seed mean of the circular-shift null
    seeds_spare: int
    seeds_frame: int
    seeds_total: int

    def clears_spare(self) -> bool:
        """`SEEDS_REQUIRED` seeds whose whole bits interval sits below the cut."""
        return self.seeds_spare >= SEEDS_REQUIRED

    def clears_frame(self) -> bool:
        """`SEEDS_REQUIRED` seeds whose whole frame interval sits above the cut."""
        return self.seeds_frame >= SEEDS_REQUIRED


@dataclasses.dataclass(frozen=True)
class CapacityInputs:
    """Reading G's inputs: the arms, their base controls, their estimator checks.

    `controls` is per arm and True when every one of that arm's seeds passed the
    floor, two-route and inequality checks. `clusters` and `rows` are caption
    figures and decide nothing.
    """

    arms: dict[str, CapacityArm]
    base: dict[str, BaseControl]
    controls: dict[str, bool]
    clusters: int
    rows: int


@dataclasses.dataclass(frozen=True)
class CapacityStatus:
    status: str
    rule: str
    arms_spare: tuple[str, ...]
    arms_frame: tuple[str, ...]
    base_failed: tuple[str, ...]
    controls_failed: tuple[str, ...]


def capacity_arm(seeds: list[dict]) -> CapacityArm:
    """One arm's `CapacityArm` from its per-seed measurement dicts.

    THE REFUSALS LIVE HERE, and they raise. A non-finite bound makes every
    comparison False, so one bad cell would read "does not clear" in both
    senses at once -- a verdict moved toward the wrong status while looking like
    a clean null. An inverted interval, a point estimate outside its own
    interval, and fewer than `SEEDS_REQUIRED` seeds are refused for the same
    reason: each would otherwise pass silently and read as a null.
    `retention.rung_arm` refuses the same three.
    """
    if len(seeds) < SEEDS_REQUIRED:
        raise ValueError(
            f"an arm needs at least SEEDS_REQUIRED={SEEDS_REQUIRED} seeds to "
            f"read, got {len(seeds)}: an arm with fewer can never satisfy the "
            "bar, so it would read as a null rather than as the refusal it is"
        )
    fields = ("bits", "ci_low", "ci_high", "frame_share", "frame_low", "frame_high",
              "redundancy_bits", "redundancy_floor")
    for i, seed in enumerate(seeds):
        for key in (*fields, "live"):
            if key not in seed:
                raise ValueError(f"seed index {i} is missing {key}")
        values = {k: float(seed[k]) for k in fields}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError(
                f"seed index {i} carries a non-finite value {values}: that is an "
                "error about the measurement, never a statement about the code"
            )
        for lo, mid, hi in (("ci_low", "bits", "ci_high"),
                            ("frame_low", "frame_share", "frame_high")):
            if values[lo] > values[hi]:
                raise ValueError(
                    f"seed index {i} has an inverted interval "
                    f"[{values[lo]}, {values[hi]}]"
                )
            if not values[lo] <= values[mid] <= values[hi]:
                raise ValueError(
                    f"seed index {i} has {mid}={values[mid]} outside its own "
                    f"interval [{values[lo]}, {values[hi]}]"
                )
    return CapacityArm(
        bits=float(np.mean([s["bits"] for s in seeds])),
        bits_low=float(min(s["ci_low"] for s in seeds)),
        bits_high=float(max(s["ci_high"] for s in seeds)),
        frame_share=float(np.mean([s["frame_share"] for s in seeds])),
        frame_low=float(min(s["frame_low"] for s in seeds)),
        frame_high=float(max(s["frame_high"] for s in seeds)),
        live=int(min(s["live"] for s in seeds)),
        redundancy_bits=float(np.mean([s["redundancy_bits"] for s in seeds])),
        redundancy_floor=float(np.mean([s["redundancy_floor"] for s in seeds])),
        seeds_spare=sum(1 for s in seeds if float(s["ci_high"]) < SPARE_CUT),
        seeds_frame=sum(1 for s in seeds if float(s["frame_low"]) > FRAME_CUT),
        seeds_total=len(seeds),
    )


def reading_capacity(inputs: CapacityInputs) -> CapacityStatus:
    """Reading G: is `z` out of room, or was it never asked?

    Precedence. `UNRESOLVED_ESTIMATOR` outranks everything: a reading taken from
    an estimator that missed a known answer is not a weaker reading, it is not a
    reading. Then `UNRESOLVED_BASE`, as in M3j and M3k -- a claim about the code
    means nothing if the current frame cannot say where it is.

    The three readings are MUTUALLY EXCLUSIVE BY CONSTRUCTION: the bits cut
    separates `SPARE_CAPACITY` from the other two, and the frame cut separates
    those two. So no arm can clear in two senses, and the whole class of
    ambiguity refusal M3k needed cannot arise here.

    `CAPACITY_BOUND` IS THE FALL-THROUGH. Both objective-lever statuses must
    clear an interval; this one is what remains. So the rule makes the lever
    M3j's h-vs-z evidence already favours the EASIEST status to reach, which is
    backwards, and the rule text says so in the words M3k used for
    INDISTINGUISHABLE: by default rather than by evidence.

    All four tuples are reported in every branch, including the branches that go
    on to take a reading. `ARMS_REQUIRED` arms holding is enough to proceed, so
    one arm can fail its base control while its measurement still votes -- the
    record must say so rather than read "nothing failed". M3k shipped exactly
    that hole and it survived 82 tests.
    """
    if len(inputs.arms) < ARMS_REQUIRED:
        raise ValueError(
            f"Reading G needs at least ARMS_REQUIRED={ARMS_REQUIRED} arms to "
            f"read, got {len(inputs.arms)}"
        )
    if set(inputs.base) != set(inputs.arms) or set(inputs.controls) != set(inputs.arms):
        raise ValueError(
            "inputs.base and inputs.controls must name exactly inputs.arms; got "
            f"arms {sorted(inputs.arms)}, base {sorted(inputs.base)}, "
            f"controls {sorted(inputs.controls)}: an arm with no base control or "
            "no estimator check would vote with nothing gating it"
        )
    controls_failed = tuple(sorted(a for a, ok in inputs.controls.items() if not ok))
    base_failed = tuple(
        sorted(a for a, control in inputs.base.items() if not control.clears())
    )
    spare = tuple(sorted(a for a, arm in inputs.arms.items() if arm.clears_spare()))
    frame = tuple(sorted(a for a, arm in inputs.arms.items() if arm.clears_frame()))
    n_arms = len(inputs.arms)

    if controls_failed:
        return CapacityStatus(
            status="UNRESOLVED_ESTIMATOR",
            rule=(
                f"the estimator missed a known answer in {', '.join(controls_failed)}; "
                "a reading taken from an estimator that failed its own floor, "
                "two-route or inequality check is not a weaker reading, it is not "
                "a reading"
            ),
            arms_spare=(), arms_frame=(), base_failed=base_failed,
            controls_failed=controls_failed,
        )
    if n_arms - len(base_failed) < ARMS_REQUIRED:
        return CapacityStatus(
            status="UNRESOLVED_BASE",
            rule=(
                f"only {n_arms - len(base_failed)} of {n_arms} arms clear "
                f"enc(t) -> position at r2 {BASE_R2_FLOOR:.2f} "
                f"({', '.join(base_failed)} failed); a claim about what the code "
                "carries means nothing where the frame cannot say where it is"
            ),
            arms_spare=(), arms_frame=(), base_failed=base_failed,
            controls_failed=controls_failed,
        )
    if len(spare) >= ARMS_REQUIRED:
        return CapacityStatus(
            status="SPARE_CAPACITY",
            rule=(
                f"the code's information content sits below half the "
                f"{CEILING_BITS:.0f}-bit ceiling in {len(spare)} of {n_arms} arms "
                f"({', '.join(spare)}), each in at least {SEEDS_REQUIRED} seeds; "
                "most of the capacity is idle, so adding capacity cannot be what "
                "limits the model -- the OBJECTIVE lever is where the next "
                "milestone goes"
            ),
            arms_spare=spare, arms_frame=frame, base_failed=base_failed,
            controls_failed=controls_failed,
        )
    if len(frame) >= ARMS_REQUIRED:
        return CapacityStatus(
            status="FRAME_REENCODING",
            rule=(
                f"most of the code's variance is explained by enc(t) alone in "
                f"{len(frame)} of {n_arms} arms ({', '.join(frame)}), each in at "
                f"least {SEEDS_REQUIRED} seeds; the capacity is in use and it is "
                "spent re-encoding the current frame, which is exactly what the "
                "embedding loss asks for -- the OBJECTIVE lever is where the next "
                "milestone goes"
            ),
            arms_spare=spare, arms_frame=frame, base_failed=base_failed,
            controls_failed=controls_failed,
        )
    return CapacityStatus(
        status="CAPACITY_BOUND",
        rule=(
            f"neither objective-lever status clears in {ARMS_REQUIRED} arms: the "
            "code is not mostly idle and is not mostly a re-encoding of the "
            "frame, so what limits it is consistent with capacity -- but this "
            "status is the FALL-THROUGH, not a bar that was cleared, so the "
            "bottleneck lever stands BY DEFAULT rather than by evidence. How much "
            "capacity would be enough is a magnitude and belongs in the run's own "
            "report, not here"
        ),
        arms_spare=spare, arms_frame=frame, base_failed=base_failed,
        controls_failed=controls_failed,
    )


def _row(values, widths) -> str:
    return "  " + "".join(f"{v:>{w}}" for v, w in zip(values, widths, strict=True))


def format_reading_capacity(reading: CapacityStatus, inputs: CapacityInputs) -> str:
    """Reading G as `capacity.txt` carries it, byte for byte.

    Every number in the caption is interpolated, never a literal: the ceiling
    and both cuts from the module, the arm count from `inputs.arms`, the seed
    count from the arms' common `seeds_total`, the shape from `inputs`. This
    project has shipped a caption disagreeing with its own columns three times.
    """
    counts = {arm.seeds_total for arm in inputs.arms.values()}
    if len(counts) != 1:
        raise ValueError(
            "the arms must share one seeds_total for the caption's "
            f"'{SEEDS_REQUIRED} of N seeds' to be true of every row, got "
            f"{ {a: arm.seeds_total for a, arm in sorted(inputs.arms.items())} }"
        )
    (seeds_total,) = counts
    lines = [
        f"--- Reading G: how many of the {CEILING_BITS:.0f} bits does the posterior "
        f"code carry, and is it a re-encoding of enc(t)? (spare when the whole bits "
        f"interval is below {SPARE_CUT:.0f}; frame when the whole share interval is "
        f"above {FRAME_CUT:.2f}; both in {SEEDS_REQUIRED} of {seeds_total} seeds and "
        f"{ARMS_REQUIRED} of {len(inputs.arms)} arms); "
        f"{inputs.rows} rows over {inputs.clusters} clusters ---",
        _row(READING_COLUMNS, READING_WIDTHS),
    ]
    for name, arm in sorted(inputs.arms.items()):
        clears = ("spare" if arm.clears_spare()
                  else ("frame" if arm.clears_frame() else "no"))
        lines.append(_row(
            (name, f"{arm.bits:.4f}", f"{arm.bits_low:.4f}", f"{arm.bits_high:.4f}",
             f"{arm.frame_share:.4f}", str(arm.live), clears),
            READING_WIDTHS,
        ))
    lines.append(
        "  base control (enc(t) -> position, must clear r2 "
        f"{BASE_R2_FLOOR:.2f}): " + ", ".join(
            f"{a} r2={inputs.base[a].r2:+.3f} "
            f"{inputs.base[a].seeds_clear}/{inputs.base[a].seeds_total}"
            for a in sorted(inputs.base)
        )
    )
    lines.append(
        "  estimator checks (floor exactly 0, two ceiling routes agreeing, the "
        "two theorem inequalities): " + ", ".join(
            f"{a}={'ok' if inputs.controls[a] else 'BROKEN'}"
            for a in sorted(inputs.controls)
        )
    )
    lines.append(
        f"  verdict: {reading.status.replace('_', ' ')} -- decided by: {reading.rule}"
    )
    return "\n".join(lines)
```

- [ ] **Step 4: Run them to verify they pass**

```bash
.venv/bin/python -m pytest tests/eval/test_capacity.py -q
```
Expected: all pass, output pristine.

- [ ] **Step 5: Prove the precedence and the bar bite**

Apply each mutation, confirm the named test fails, restore and clear `__pycache__`:

| mutation | must fail |
|---|---|
| swap the `controls_failed` and base-gate blocks | `test_a_broken_control_outranks_every_reading` |
| `len(spare) >= ARMS_REQUIRED` → `>= 1` | `test_exactly_arms_required_arms_read_and_one_short_does_not` |
| `seeds_spare >= SEEDS_REQUIRED` → `>= 1` | `test_an_arm_at_exactly_seeds_required_clears_and_one_short_does_not` |
| `ci_high < SPARE_CUT` → `bits < SPARE_CUT` | `test_the_cut_is_on_the_interval_not_the_point_estimate` |
| hardcode `arms_frame=()` in `SPARE_CAPACITY` | `test_every_branch_reports_all_four_of_its_tuples[SPARE_CAPACITY...]` |
| swap the `frame` and `live` columns in the row | `test_the_table_pins_every_column_to_the_arm_it_came_from` |
| write `of 3 arms` as a literal in the caption | `test_the_caption_uses_the_inputs_and_the_derived_ceiling_not_literals` |

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/capacity.py tests/eval/test_capacity.py
git commit -m "feat: Reading G -- five statuses, and CAPACITY_BOUND is the fall-through"
```

---

### Task 4: `scripts/latent_capacity.py` — the measure phase

**Files:**
- Create: `scripts/latent_capacity.py`
- Test: `tests/eval/test_latent_capacity_script.py`

**Interfaces:**
- Consumes: Task 1's `"post_probs"` / `"prior_probs"`; Task 2's `bits_carried`, `floor_bits`, `ceiling_bits`, `argmax_marginal_bits`, `live_classes`, `redundancy_bits`, `redundancy_floor`, `episode_stats`, `bits_interval`, `CEILING_BITS`; `probe.filtering_gain`-style splits, `probe.GainSplit`, `probe.fit_probe`; `retention.TARGETS`, `CONFIDENCE`, `RESAMPLES`.
- Produces, used by Task 5: `EXIT_ESTIMATOR_BROKEN = 43`, `EXIT_BASE_UNRESOLVED = 44`, `PHASES`, `capacity_record_path`, `write_record`, `load_record`, `_cell_args`, `gather_once`, `cell_capacity`, `base_control`, `estimator_checks`, `measure_cell`, `measure_phase`, `require_readable_plan`, `require_one_protocol`, `_PROTOCOL_FIELDS`.

**Read `scripts/latent_width.py` first.** It is the direct precedent for every part of this task, and four recorded blockers were paid for there.

- [ ] **Step 1: Write the failing tests for the four recorded blockers**

Create `tests/eval/test_latent_capacity_script.py` with these, which are the ones that cost real debugging cycles on the two previous milestones:

```python
def test_prepare_cell_is_handed_the_study_directory_not_the_record_directory():
    """`prepare_cell` loads the checkpoint from the `out` of the args IT is
    handed, and that must be the STUDY directory. Passing `args` through
    unchanged makes it hunt for the nine checkpoints among our own output
    records and refuse the first cell -- a real run blocker on M3j, fixed with
    this same shim. NOTHING ELSE COVERS THIS LINE: every other test either
    stubs `prepare_cell` or feeds `cell_capacity` directly."""
    seen = {}
    monkeypatch.setattr(script, "prepare_cell",
                        lambda a, c, d: seen.update(out=str(a.out)) or _prepared())
    script.measure_cell(_args(out="runs/m3l_capacity", source="runs/m3_study_v2"),
                        _cell(), "cpu", [], [])
    assert seen["out"] == "runs/m3_study_v2"
    assert seen["out"] != "runs/m3l_capacity"


def test_every_validation_episode_is_scored():
    """`filtering_gain`'s `limit` caps the FIT split and the SCORED split at the
    same number. M3j mirrored it and silently scored 20 of 24 validation
    episodes -- 17% of the evaluation data discarded. The test meant to pin it
    could not, because `len(val)` was ALSO 20 in its fixture. So this fixture's
    two numbers MUST differ."""
    val = [f"e{i}.npz" for i in range(24)]
    assert len(val) != script.FIT_EPISODES, "the fixture cannot fail if these agree"
    seen = []
    monkeypatch.setattr(script, "gather_probe_data",
                        lambda *a, **k: seen.append(k["limit"]) or _gathered())
    script.gather_once(_prepared(), [f"t{i}.npz" for i in range(20)], val, seed=0)
    assert seen[-1] == len(val) == 24


def test_require_one_protocol_compares_the_measured_shape_not_only_constants():
    """M3k's review found `projection_seed` and `rung_width` were CONSTANTS that
    cannot vary between two records of one code version, while the measured
    `h_dim` went unchecked. `z_cats`/`z_classes` are the same trap here: they
    set the ceiling every cut is a fraction of."""
    fields = {f for f, _ in script._PROTOCOL_FIELDS}
    assert {"git_sha", "torch_version", "episodes.val", "windows.episode",
            "z_cats", "z_classes"} <= fields


def test_the_plan_is_refused_before_the_probe_not_after():
    """`reading_capacity` raises on too few arms INSIDE the reading, i.e. after
    all the gather work. `scripts/latent_retention.py` refuses a too-narrow plan
    up front for exactly that reason -- without it, `--arms frozen_ssl` does the
    whole measure and then crashes."""
    with pytest.raises(SystemExit, match="ARMS_REQUIRED"):
        script.require_readable_plan(["frozen_ssl"], (0, 1, 2))
    with pytest.raises(SystemExit, match="SEEDS_REQUIRED"):
        script.require_readable_plan(list(script.ARMS), (0,))
    script.require_readable_plan(list(script.ARMS), (0, 1, 2))  # does not raise
```

Also write, in the same file: a test that `estimator_checks` returns False when `floor_bits` is nonzero, when the two ceiling routes disagree, and when either of the **two theorem** inequalities is violated (`-1e-9 <= bits <= CEILING_BITS + 1e-9` and `0 <= ceiling <= CEILING_BITS + 1e-9`) — each driven separately, each with a fixture that clears the others. **There is no third inequality:** `bits <= ceiling_bits` is not a theorem, and `tests/eval/test_capacity.py::test_ceiling_bits_is_not_an_upper_bound_on_bits_carried` carries the counterexample. Drive that same fixture through `estimator_checks` and assert it is NOT refused; a test that the record's key set is pinned exactly -- `redundancy_bits` and `redundancy_floor` among its keys, beside `bits` -- and round-trips through the real `write_record`/`load_record`; and a test that every probe call gets the cell's own bootstrap seed (`Counter` over **all** calls, not the first — M3j shipped every cell sharing seed 0, which correlates the interval noise the agreement rule treats as independent), and the same `Counter` over every `redundancy_floor` call, which must receive the cell's own seed.

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/bin/python -m pytest tests/eval/test_latent_capacity_script.py -q
```
Expected: collection error — the script does not exist.

- [ ] **Step 3: Write the measure phase**

Create `scripts/latent_capacity.py`, mirroring `scripts/latent_width.py`'s structure section for section. The pieces, with the decisions that matter:

```python
EXIT_ESTIMATOR_BROKEN = 43
"""The estimator missed a known answer. 38 is M3i, 39-40 M3j, 41-42 M3k."""

EXIT_BASE_UNRESOLVED = 44
"""`enc(t) -> position` did not clear BASE_R2_FLOOR in ARMS_REQUIRED arms."""

FIT_EPISODES = 20
SELECT_EPISODES = 20
PHASES = ("all", "measure", "read")


def gather_once(prepared, train, val, *, seed: int):
    """ONE gather per cell, scoring EVERY validation episode.

    `limit=len(val)` on the scored split, never `FIT_EPISODES`: see the test.
    One gather means one row set, so `bits_carried` and `frame_share` describe
    the same rows. A second gather would reintroduce the row-alignment hazard
    M3k needed two layered guards for.
    """


def estimator_checks(probs) -> dict:
    """The three checks, each reported separately so a failure names itself.

    Tolerances are 1e-9, not equality: the floor is zero in exact
    arithmetic but drifts to ~5e-13 by summation order at run scale, and an
    M3h sweep once refused on an 8.527e-14 mismatch that was exactly that.

    `floor` must be within 1e-9 of 0; `routes` compares `ceiling_bits` against
    `argmax_marginal_bits`; `bracket` is the TWO THEOREMS -- `-1e-9 <= bits <= CEILING_BITS + 1e-9` and
    `0 <= ceiling <= CEILING_BITS + 1e-9`. **NOT** `bits <= ceiling`, which is not
    a theorem: `ceiling_bits` is the ARGMAX PATTERN's information content, and a
    code whose argmax never moves while its tail varies reads 7.52 bits against a
    `ceiling_bits` of 0.00. Refusing on it would reject valid readings in exactly
    the diffuse regime this milestone investigates.
    Returns the three booleans AND the four numbers, because a check that
    reports only a boolean cannot be audited from the record.
    """


def cell_capacity(gathered, *, seed: int) -> dict:
    """One cell's `bits`, interval, `frame_share`, `live`, `redundancy_bits`,
    `redundancy_floor` and the three checks.

    `redundancy_floor(probs, seed=seed)` takes the CELL'S OWN seed, never a
    default: a defaulted seed is how M3j shipped every cell sharing one. Both
    redundancy numbers ship on the record as plain floats -- the ratio is derived
    at read time by `redundancy_ratio`, which is None for a collapsed code -- and
    they gate nothing: no check, status or refusal depends on them.

    `frame_share` is `R2(enc(t) -> post_probs)` on the flattened 1024 columns,
    ZERO-VARIANCE COLUMNS EXCLUDED -- R^2 is undefined for them and this project
    has already refused a fixture for exactly that. Their count ships as
    `live`.
    """
```

Follow `latent_width.py` for `_cell_args`, `base_control` (recording `r2`, `position_r2` and `per_column_r2`, gated on `position_r2`), `require_readable_plan`, `require_one_protocol` over a module-level `_PROTOCOL_FIELDS` the function **iterates**, `measure_cell` threading `seed=cell.seed`, `measure_phase`, `write_record`/`load_record`, and `_cell_line`.

- [ ] **Step 4: Run the file's tests**

```bash
.venv/bin/python -m pytest tests/eval/test_latent_capacity_script.py -q
```
Expected: all pass. Keep fixtures cheap — the read path needs no ridge solve, and `test_latent_width_script.py` already costs 75s.

- [ ] **Step 5: Prove each blocker's test bites**

Mutate each in turn, confirm the named test fails, restore and clear `__pycache__`: pass `args` straight to `prepare_cell`; set the scored `limit` to `FIT_EPISODES`; drop `z_cats` from `_PROTOCOL_FIELDS`; move `require_readable_plan` after the gather; make every cell use bootstrap seed 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/latent_capacity.py tests/eval/test_latent_capacity_script.py
git commit -m "feat: latent_capacity.py measure -- one gather per cell, three checks recorded per cell"
```

---

### Task 5: `scripts/latent_capacity.py` — the read phase and the registry

**Files:**
- Modify: `scripts/latent_capacity.py` (append)
- Modify: `tests/eval/test_diagnose_dynamics_script.py`
- Test: `tests/eval/test_latent_capacity_script.py` (append)

**Interfaces:**
- Consumes: Task 3's `CapacityInputs`, `capacity_arm`, `reading_capacity`, `format_reading_capacity`; Task 4's record schema and `require_one_protocol`.
- Produces: `capacity_inputs`, `base_controls`, `control_flags`, `capacity_text`, `write_text`, `read_phase`, `READ_EXITS`, `main`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_latent_capacity_script.py`. The ones that must be there, each with the defect it exists for named in its docstring:

1. **Nine distinct payloads.** The fixture threads a rank per `(arm, seed)` through every number, and a test drops the labels and asserts nine distinct payloads. M3j shipped a fixture where all nine records serialised to ONE payload with the labels dropped, so an implementation pairing `frozen_ssl` with `pixel_ae`'s seeds passed.
2. **Byte-identity between `capacity.txt` and stdout**, pinned with a sentinel carrying trailing spaces, a tab and three trailing newlines substituted for `capacity_text`. Both of this project's previous byte-identity tests were satisfied by `write_text(text.rstrip())`; only a sentinel like this catches it.
3. **A seed collapse in the BASE-CONTROL loop specifically.** The dangerous variant is every arm reporting its own seed 2 three times: the arms' r2 values still all differ, so a cross-arm *distinctness* check misses it. Assert the **tally**, not distinctness — the base control is a gate, so it is the worst place for a silent collapse.
4. **41/42/43/44 all distinct, and 43/44 mapped to the right statuses**, driven end-to-end through `main --phase read`.
5. **All five statuses driven end-to-end**, each asserting the printed `verdict:` line names that status and the estimator-checks line's `ok`/`BROKEN` matches the record. M3k's formatter was exercised on one status of five and its verdict line could be deleted with all 65 tests passing.
6. **`clusters`/`rows` disagreement refused, not sampled**, with the victim sorting **last** so a first-record pick sees nothing.

In `tests/eval/test_diagnose_dynamics_script.py`, extend the exit-status registry with `latent_capacity` owning `{43, 44}` and assert no collision with the eleven sibling scripts.

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/bin/python -m pytest tests/eval/test_latent_capacity_script.py tests/eval/test_diagnose_dynamics_script.py -q
```

- [ ] **Step 3: Write the read phase**

Append to `scripts/latent_capacity.py`:

```python
READ_EXITS = {
    "UNRESOLVED_ESTIMATOR": EXIT_ESTIMATOR_BROKEN,
    "UNRESOLVED_BASE": EXIT_BASE_UNRESOLVED,
}
"""Keyed on exactly the two refusal statuses `reading_capacity` returns. Every
other status falls through to EXIT_OK."""


def capacity_text(records, inputs, reading) -> str:
    """Everything `read` prints, in the order a reader should meet it: the
    self-check, the per-cell table, then Reading G -- evidence before the
    conclusion -- written to `capacity.txt` and printed as the SAME string."""
```

Every field the reading needs is read through a `_get(record, cell, *path)` that names the cell and the dotted path on a miss, as `latent_width.py` does -- including `redundancy_bits` and `redundancy_floor`, which `capacity_inputs` threads into each per-seed dict for `capacity_arm`. `capacity_text` prints, per arm, both numbers and `redundancy_ratio(arm.redundancy_bits, arm.redundancy_floor)` beside the verdict in EVERY status (`undefined` where it is None), and a test drives a collapsed cell through it. `clusters` and `rows` go through a `_one_value` that **refuses a disagreement naming the cells** rather than picking the first record's.

- [ ] **Step 4: Run them to verify they pass**

```bash
.venv/bin/python -m pytest tests/eval/test_latent_capacity_script.py tests/eval/test_diagnose_dynamics_script.py tests/eval/test_capacity.py -q
```

- [ ] **Step 5: Prove the byte-identity and formatter tests bite**

Replace `write_text`'s body with `path.write_text(text.rstrip())`; confirm the sentinel test fails. Delete the `verdict:` line; confirm the five-status test fails for all five. Invert the estimator-checks `'ok' if ok else 'BROKEN'`; confirm the same test fails. Restore each, clearing `__pycache__`.

- [ ] **Step 6: Commit**

```bash
git add scripts/latent_capacity.py tests/eval/test_latent_capacity_script.py tests/eval/test_diagnose_dynamics_script.py
git commit -m "feat: latent_capacity.py read -- Reading G, exits 43 and 44, and the registry"
```

---

### Task 6: The run

**Files:**
- Modify: `docs/superpowers/plans/2026-09-30-mb-fps-m3l-latent-capacity.md` (append `## Task 6 results`)

- [ ] **Step 1: Check the disk and the suite**

```bash
df -h . | tail -1
.venv/bin/python -m pytest -q 2>&1 | tail -3
```
Expected: ≥ 20Gi free; 0 failures and **0 warnings**. The suite takes ~35 minutes. M3k's run needed 16Gi and that was tight.

- [ ] **Step 2: Smoke one cell, timed**

```bash
mkdir -p runs/m3l_smoke && git rev-parse HEAD > runs/m3l_smoke/.head
date -u +%Y-%m-%dT%H:%M:%SZ > runs/m3l_smoke/.started
caffeinate -dimsu .venv/bin/python scripts/latent_capacity.py \
  --phase measure --arms pixel_ae --seeds 0 \
  --source runs/m3_study_v2 --out runs/m3l_smoke 2>&1 | tee runs/m3l_smoke/smoke.log
echo "python exit: ${PIPESTATUS[0]}"
```

**Read the printed lines and the record's `checks` field, never `$?`** — `measure_phase` returns `EXIT_OK` even when a check fails, because 43 lives in the read phase. Expected in the record: `clusters` 24, 229 windows, 11,450 rows, `floor` within 1e-9 of 0.0, the two ceiling routes agreeing to 1e-9, `-1e-9 <= bits <= 160 + 1e-9` and `0 <= ceiling <= 160 + 1e-9`.

**If any of those disagree, STOP and report** rather than committing an hour to nine cells. M3j's smoke caught a bug that had silently discarded 17% of the evaluation data.

- [ ] **Step 3: Measure all nine cells**

```bash
mkdir -p runs/m3l_capacity && git rev-parse HEAD > runs/m3l_capacity/.head
date -u +%Y-%m-%dT%H:%M:%SZ > runs/m3l_capacity/.started
nohup caffeinate -dimsu .venv/bin/python scripts/latent_capacity.py \
  --phase measure --source runs/m3_study_v2 --out runs/m3l_capacity \
  > runs/m3l_capacity/measure.log 2>&1 &
```

Budget **~30–45 minutes**, and do not conclude it has hung before that. M3k's nine cells took 81m31s because one cell took 28m against 5m42s–8m10s for the other eight when the host swapped. Poll the log; do not hold a foreground shell.

- [ ] **Step 4: Accept the run**

Assert from the records, not from the log: nine records at **one** `git_sha` equal to `.head`; `abs(checks.floor) < 1e-9` on all nine; `checks.routes` agreeing on all nine; both theorem inequalities holding on all nine; one `clusters`/`rows`/`z_cats`/`z_classes` across all nine; `nonfinite` empty.

- [ ] **Step 5: Read, and verify byte-identity**

```bash
.venv/bin/python scripts/latent_capacity.py --phase read --out runs/m3l_capacity \
  2>&1 | tee runs/m3l_capacity/read.log
echo "python exit: ${PIPESTATUS[0]}"
diff <(.venv/bin/python scripts/latent_capacity.py --phase read --out runs/m3l_capacity) \
     runs/m3l_capacity/capacity.txt && echo "byte-identical"
```

- [ ] **Step 6: Write `## Task 6 results` into this plan**

**Every quantitative claim must be derived from `runs/m3l_capacity/*.json` and say which field it came from.** This project has overclaimed in its results section three times; the worst wrote "every magnitude says the bottleneck" when the ladder showed the opposite lever's story in 8 of 9 cells.

Required, each with its cell count out of 9 and its exception named with its margin:

- `bits_carried` per arm and per cell, against the 160-bit ceiling, as a **share** as well as a level — described as **the per-categorical informations summed, an upper bound on the code's joint information**, never as "the code carries N of 160 bits". Measured: 32 categoricals all copying one 5-bit variable read 160.0000 while carrying 5 bits jointly, a 32× overstatement. So a reading BELOW a cut is sound and conservative; a reading ABOVE one does **not** establish that the capacity is in use, and if the status rests on a high reading the write-up must say so in the same breath
- `frame_share` per arm and per cell, and `live` out of 1024
- **`redundancy_bits` against `redundancy_floor`** per cell, the ratio taken from the records' two fields with `redundancy_ratio` ("not defined" where it is None, a collapsed code), and what the ratio licenses: near the floor means the categoricals are independent, so the summed `bits_carried` approximates the joint information and a high reading does mean the capacity is in use; far above it means the reading is inflated and a high value establishes nothing. **A `CAPACITY_BOUND` verdict must be reported beside this ratio**, because it means nothing unless the redundancy sits near its floor
- `prior_bits` as the companion that decides nothing, with one sentence on what it says about M3g's finding that the prior is the failing stage
- all three estimator checks on all nine cells
- **Reading G's status under the rule as written** — and if the magnitudes suggest another, report both, say plainly that they disagree, and leave the lever to the human
- **If the status is `CAPACITY_BOUND`, say in the same breath that it arrived by fall-through rather than by clearing a bar**, and that a low `live` beside it would be self-contradicting — report `live` in that sentence
- what the reading licenses and what it does not, per spec §8

- [ ] **Step 7: Commit**

Write the message to a file (the results contain apostrophes) and `git commit -F`.

---

## Exit criteria

- Reading G taken on nine records at one `git_sha`, or the run refused with a numbered status and the refusal recorded.
- `floor_bits` reads **within 1e-9 of 0.0** on all nine cells — zero up to float
  summation order, which is ~5e-13 at run scale, not bit-exact.
- The two `ceiling_bits` routes agree on all nine cells.
- `−1e-9 ≤ bits_carried ≤ 160 + 1e-9` and `0 ≤ ceiling_bits ≤ 160 + 1e-9` on all
  nine cells. **Not** `bits_carried ≤ ceiling_bits`, which is not a theorem.
- Every pre-existing `gather_probe_data` key byte-identical, pinned by test; `gain_from_blocks`' ten-key golden output unchanged.
- All five Reading G statuses reachable and driven end-to-end by a test.
- The whole suite passes with 0 failures and 0 warnings.
- `capacity.txt` byte-identical to a second `--phase read`.
- No checkpoint altered, nothing under `runs/` removed, no M3b–M3k verdict touched.

## Plan self-review (writing-plans)

**Spec coverage.** §1 → Task 6 Step 6. §2.1 → Task 2. §2.2 → Task 4 (`cell_capacity`). §2.3 → Task 2 Steps 1, 3, 5. §2.4 → Tasks 2 and 3 (`redundancy_*`, `CapacityArm`), Task 4 (`prior_bits`, `redundancy_bits` and `redundancy_floor` on the record) and Task 6 Step 6. §3.1 → Task 3. §3.2 → Task 2 (`SPARE_CUT`'s docstring) and Task 3. §3.3 → Task 3's `CAPACITY_BOUND` rule text and Task 6 Step 6. §4 → the File Structure table. §4.1 → Task 1. §4.2 → Task 4 (`gather_once`). §5 → Global Constraints. §6 → Task 6 Steps 2–3. §7 → Exit criteria. §8 → Task 6 Step 6. No gaps.

**Placeholder scan.** Tasks 1–3 carry complete code. Tasks 4 and 5 give the signatures, the docstrings that state each decision, and the tests by the defect each exists for, and name `scripts/latent_width.py` as the precedent to mirror section for section — deliberately, because reproducing 800 lines of a sibling script here would be a copy for the implementer to diverge from rather than a file to follow. **If that is too thin for a fresh implementer, those two tasks should be split before execution.**

**Type consistency.** `CapacityArm`'s twelve fields are spelled identically in Task 3's `_arm` helper, its dataclass, `capacity_arm`'s constructor and `format_reading_capacity`'s row. `CapacityInputs(arms, base, controls, clusters, rows)` in Tasks 3 and 5. `episode_stats(probs, groups) -> EpisodeStats` and `bits_interval(stats, *, resamples, confidence, seed)` in Tasks 2 and 4. `estimator_checks(probs) -> dict` with keys `floor`, `routes`, `bracket` in Tasks 4 and 5. `SPARE_CUT` is half `CEILING_BITS` in one place only.

**Five defects found in this plan's own code and fixed inline.** The last two were
found by *executing* the plan's estimator and fixtures before writing them down,
which is how M3k's three fixture defects should have been caught:

1. **`_arm`'s `seeds_spare` default read the point estimate.** It computed `seeds_total if bits["bits"] < SPARE_CUT` — which would have made `test_the_cut_is_on_the_interval_not_the_point_estimate` pass against an implementation that also used the point estimate, i.e. the test could not fail. Now `bits["ci_high"] < SPARE_CUT`, matching `capacity_arm`.
2. **`entropy_bits` warned on `log2(0)`.** The first form was `-(p * np.log2(np.where(p > 0, p, 1.0))).sum(-1)`, which is numerically right but still evaluates `p * log2(...)` where `p == 0` — fine — while an earlier draft passed `p` directly and emitted a `divide by zero` RuntimeWarning. The suite requires 0 warnings, so the guard is doubled and the docstring says why.
3. **`test_live_classes_counts_the_columns_that_vary` did not renormalise.** Zeroing class 7 left rows summing to 31/32, which `_require_distributions` refuses — the test would have failed on the guard rather than on the count. It now renormalises, and the expected count is `CATS * CLASSES - CATS` rather than a bare number, so it tracks the config.
4. **"`floor_bits` reads exactly 0.0" was false**, in the test, the exit criteria, the acceptance step and two docstrings. Measured: +5.68e-14 at 500 rows, −1.42e-13 at 2,000, +5.40e-13 at the ~11,000 a real cell carries — float summation order. Every tolerance is now 1e-9, with the M3h precedent cited where it is stated: a sweep once refused on an 8.527e-14 mismatch that was summation order and not a defect. Claiming bit-exactness where only near-exactness holds is a failure this project has already paid for once.
5. **The renormalisation in that same test divided by zero.** The fixture was the *one-hot* `_deterministic_uniform`, so zeroing class 7 emptied every row whose one-hot was class 7, and `dead /= dead.sum(...)` then raised `RuntimeWarning: invalid value encountered in divide` — which the suite's zero-warning requirement turns into a failure. The fixture is now the Dirichlet base, whose smallest row sum after zeroing is a measured 0.7242.
