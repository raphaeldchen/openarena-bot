# M3n: The Motion Ladder — Design

**Date:** 2026-10-04
**Branch:** `feat/m3n-motion-ladder`, off `1c81c19`
**Predecessor:** M3m, `docs/superpowers/specs/2026-10-02-mb-fps-m3m-prediction-burden-design.md`

## 1. The question

M3m established that the world model's 45-step rollout error is **compounding**:
`burden(1)` is not resolvable from zero at 2 SE in 9 of 9 cells (max 1.94 SE),
while `compounding(45)` is resolvable at 5.56–11.63 SE.

It could not establish *why* the one-step map is cheap. A map that predicts
motion well and a map that predicts almost no motion both score a small
`burden(1)`, because holding position is cheap relative to the probe's readout
error. M3m's own results section says so.

M3n answers that. It places the one-step map on an axis whose ends are
**copying** (predict no motion) and **perfect** (predict as well as the readout
permits), and reports where each cell sits, with a ruler.

## 2. What M3m got wrong, and why the error was attractive

M3m's designated statistic was

    motion_margin(h) = one_step_persistence(h) - rung_1(h)

`one_step_persistence` is the median true one-step displacement — a
**ground-truth** quantity, 3.9694722203504225 map units on the shipped split.
`rung_1` is the model's one-step error in **probe space**, 88.29–224.36. The
subtraction crosses coordinate systems, so readout error dominates by
22.2×–56.5×. The base control (median true displacement must exceed median
floor error) failed 9 of 9, and M3m shipped `UNREADABLE`, exit 46.

The error is attractive, not careless. `motion_margin` needs a scale for "how
much motion was there to predict," and the intuitive answer is the true
displacement. But the numerator lives in probe space, and the only probe-space
answer is **what a perfect predictor would have won**. That is a measured
quantity, not a constant, and it varies **20.5×** across the nine cells
(0.575649 to 11.824 map units at h=1).

The same mistake is available one level up: normalising M3n's margin by
3.9694722203504225 produces an apparent bimodal split across cells (four near
100%, five at 6–43%) that dissolves entirely under the correct denominator
(−2.4% to 72.7%, unimodal, mean 35.6%). The pattern was the constant denominator,
not the model.

## 3. The statistic

### 3.1 Three differences, one identity

All three are probe-space error differences, defined per re-grounding period
`k` and horizon step `h`. Both axes are named explicitly and never conflated;
write `headroom(k, h)`, never `headroom(45)`.

| name | definition | meaning |
| --- | --- | --- |
| `headroom(k,h)` | `hold_k(h) - floor(h)` | what a perfect predictor wins over copying |
| `skill(k,h)` | `hold_k(h) - rung_k(h)` | what the model wins over copying |
| `deficit(k,h)` | `rung_k(h) - floor(h)` | M3m's `burden(k,h)`, unchanged |

**Identity:** `skill(k,h) + deficit(k,h) == headroom(k,h)`, exactly — the
`hold_k(h)` term cancels. Measured on the shipped M3m records at the k=45
column, the max absolute residual over all nine cells and all 45 steps is
**1.421e-14**. It ships as `identity_residual` with `IDENTITY_TOLERANCE = 1e-9`,
the same role it has in M3m.

The identity is also why the reading carries **no ratio**. The two-sided test
against the copying end and the perfect end is `skill > 0` and `deficit > 0` —
two differences, no denominator.

### 3.2 `hold_k` and its index expression

`hold_k(h)` is the floor's probe-space position at the rung's **last
re-grounding step**, held forward to `h`. One expression:

    ground_step(k, h) = k * ((h - 1) // k)

    hold_k(h) = positions_at_context          when ground_step(k, h) == 0
                positions_real[:, g - 1]      otherwise, g = ground_step(k, h)

`positions_real[:, j]` is the floor's probe position at horizon step `j + 1`;
`positions_at_context` is the probe position at the last context frame, i.e.
horizon step 0. Both are per-window arrays already kept by
`_diagnose(..., keep_trajectories=True)` since M3d.

The index arithmetic is **not** accepted by argument. It is pinned by two
reductions, both measured at exactly `0.000e+00` on the nine shipped M3m
records:

1. **k=45 collapses to persistence.** Every `h` in `[1, 45]` gives
   `ground_step(45, h) == 0`, so `hold_45(h) == positions_at_context` for all
   `h`, and `||positions_at_context - true_positions(h)||` is bitwise
   `persistence_position[h]`. The ladder's far corner *is* the M3 gate's
   `beats_persistence` criterion.
2. **The h=1 column is k-invariant.** For every `k >= 1`,
   `ground_step(k, 1) == 0`, and the rung's prediction at `h=1` is one dynamics
   step from the context posterior regardless of `k`. So all three differences
   at `h=1` must be identical across all five `k`.

Reduction 2 makes the one-step verdict **k-free by construction**, which is why
the decision cell below names `k` only as a label.

### 3.3 The two-sided reading

Per cell, at the decision cell, each difference against the ruler of §3.5:

**Step 1 — readability gate.** `headroom` must be resolvably `> 0`. If it is
not, that cell reads `UNREADABLE`: a perfect predictor cannot be told from a
copying one, so nothing between them can be placed either.

**Step 2 — place the model.**

| `skill > 0` | `deficit > 0` | verdict | meaning |
| --- | --- | --- | --- |
| yes | yes | `BETWEEN` | real one-step skill *and* a real one-step deficit |
| yes | no | `AT_PERFECT` | as good as the readout permits |
| no | yes | `AT_COPYING` | not resolvably better than predicting no motion |
| no | no | `AMBIGUOUS` | ruler too coarse to place it |

`AMBIGUOUS` is reachable with a resolvable `headroom`: the gate only
establishes that the two ends are separated, not that the ruler is fine enough
to locate a point between them.

The gate is not a formality. `pixel_ae_seed1` has `headroom(45, 1) = 0.575649`
and `headroom(45, 2) = -0.641206` — the only negative step in all 45 — so the
normalised share at that cell reads **+339.5%** at `h=2` and **−335.7%** at
`h=3` as the denominator crosses zero. A tuned fractional bar would have been
applied to that ratio, with two absurd percentages printed beside eight
plausible ones as the only warning.

### 3.5 The ruler

The verdict uses an **episode-clustered bootstrap interval on the paired
per-window differences**. A difference is "resolvably > 0" when that interval
lies entirely above 0.

Both properties are load-bearing and neither alone is enough.

**Paired**, because the readout error is what the fix exists to cancel. The
hold baseline and the rung are both downstream of the same posterior latent, so
per window their difference is of order the predicted displacement — units —
while either error alone is 130–260. Differencing per window first, then
averaging, keeps that cancellation; averaging first and differencing the means
throws it away.

**Episode-clustered**, because the 229 windows come from 24 episodes at 3 to 10
windows each, and `_spread` divides by `sqrt(229)` as though they were
independent draws. Measured on the one per-window array M3m shipped
(`window_margin`, 229 x 45, all nine cells), the episode-clustered standard
error at `h=1` is **1.26x to 2.17x** the iid one, mean 1.66x:

| cell | iid SE | clustered SE | inflation |
| --- | --- | --- | --- |
| frozen_ssl_seed0 | 7.1989 | 9.7620 | 1.36x |
| frozen_ssl_seed1 | 8.0264 | 12.2222 | 1.52x |
| frozen_ssl_seed2 | 7.0110 | 11.4698 | 1.64x |
| pixel_ae_seed0 | 8.0209 | 10.1323 | 1.26x |
| pixel_ae_seed1 | 10.6598 | 23.1692 | 2.17x |
| pixel_ae_seed2 | 9.1684 | 14.0234 | 1.53x |
| random_vit_seed0 | 9.0088 | 18.5382 | 2.06x |
| random_vit_seed1 | 8.7380 | 17.4103 | 1.99x |
| random_vit_seed2 | 8.7132 | 12.2143 | 1.40x |

(`sqrt(229/24) = 3.09x` is the ceiling, reached only if windows within an
episode were perfectly correlated. For equal-sized episodes the SE ratio is at
most `sqrt(m)` in `m` windows per episode, so a fixture with four windows per
episode cannot exceed 2x by design.)

Re-derivable, and identical on both of M3m's record sets:

```python
rows = np.array(record["window_margin"]); lab = np.array(record["windows"]["episode"])
iid  = rows[:, 0].std(ddof=1) / np.sqrt(rows.shape[0])
reps = np.array([rows[i, 0].mean() for i in episode_bootstrap(lab, 2000, 0)])
inflation = percentile_interval(reps)[2] / iid
```

The direction matters. An understated standard error makes a quantity look MORE
resolvable, so it is the conservative choice for a claim of *non*-resolvability
and the dangerous one for a claim of resolvability. M3n's verdict rests on
claims of resolvability, which is why it cannot use `_spread`.

`paired_standard_error` from `_spread` is still **recorded**, as a cheaper
secondary figure with its understatement documented in the record, so the two
rulers can be compared. It does not decide anything.

### 3.4 The reported share

`skill / headroom` is reported as a **point estimate with no interval**, and
only where `headroom` is resolvably positive. It needs no interval: the verdict
is two-sided on the two differences, so the share is presentation. This closes
M3m's recorded follow-up (no interval on the compounding share, for want of the
covariance between two rulers) by making it unnecessary rather than by building
it.

## 4. Measurement path

**`rollout.py` is not modified.** M3m's results note said the positions
`evaluate_rollout` computes at `rollout.py:160-162` are not returned. That is
true of `evaluate_rollout`, but `_diagnose(..., keep_trajectories=True)` has
kept them since M3d:

- `Trajectories.positions_real` — `(n_windows, horizon, 2)`, the probe of the
  floor's embedding at each step
- `Trajectories.positions_at_context` — `(n_windows, 2)`, the probe at the last
  context frame
- `Trajectories.true_positions` — `(n_windows, horizon, 2)`, the truth slice
  the curves are scored against

One floor serves every `k` by construction: `RegroundingSweep.reference` is
documented as "the ONE floor and persistence every k is read against," and the
trajectory fields are filled on the canonical pass only. That is exactly the
scope `hold_k` needs.

**`regrounding_sweep` passes `keep_trajectories=True`.** The flag's own
docstring states the pass "draws nothing from the stream" and does only numpy
on rows already in hand, so the control is sharp: the sweep's curves with the
flag on must be **bitwise** the shipped ones.

**`keep_latents` stays `False`.** The stage decomposition is not read here, and
it is the flag that costs memory — `post_logits` is
`(n_windows, context + horizon, groups, classes)` against the trajectory
fields' `(n_windows, horizon, 2)`.

**`RegroundingSweep` gains two fields**, parallel to the existing pair:

    hold_position: dict[int, np.ndarray]          # mean curve per k
    window_hold_position: dict[int, np.ndarray]   # per-window rows per k

Positions stay internal to the sweep; it exports errors, which is the shape it
already has.

**Rulers become per-h, and are doubled.** `_spread`
(`src/mbfps/eval/diagnostics.py:1813`) already reduces over windows and returns
a `(horizon,)` array. M3m's record carried two scalars
(`floor_margin_standard_error = 2.654`, `paired_standard_error = 10.293` on
`frozen_ssl_seed0`) only because `scripts/prediction_burden.py` stored the
decision horizon alone. M3n records the full array for each of the three
differences.

`_spread` returns **one** SE. Where it is recorded as a secondary figure it is
doubled, as `scripts/diagnose_dynamics.py:1430` does. This is written down
because mislabelling one SE as two in M3m produced the opposite conclusion from
the correct one, and the wrong one was the more interesting-sounding.

## 5. Constants

    REGROUNDING_KS   = (1, 3, 5, 15, 45)        # inherited, unchanged
    REPORTED_H       = (1, 2, 3, 5, 8, 10, 15, 20, 30, 45)   # inherited
    DECISION_K       = 1        # a label; the h=1 column is k-invariant
    DECISION_H       = 1        # the one-step question
    IDENTITY_TOLERANCE = 1e-9
    CONFIDENCE       = 0.95     # the verdict's clustered bootstrap interval
    RESAMPLES        = 2000
    SECONDARY_SIGMAS = 2        # the recorded _spread figure only; decides nothing
    SEEDS_MINIMUM    = 3
    CELLS_REQUIRED   = 5        # strict majority of nine; implies >= 2 arms,
                                # since no arm holds more than 3 cells
    EXIT_UNREADABLE_HEADROOM = 47
    EXIT_NO_MAJORITY         = 48

Protocol: `context = 5`, `horizon = 45`, 229 windows from 24 validation
episodes, `split_seed` as shipped. Every constant above is recorded in each
cell's record, so none of them is a literal that survives the tests.

The median true one-step displacement, 3.9694722203504225, is **recorded and
not used by any statistic**. It is kept as the quantity M3m's design mistook
for a probe-space scale, so a reader of the record can see that it plays no
part in the reading.

## 6. Controls

The first four are required to be exactly `0.0`; `identity_residual` is held to
`IDENTITY_TOLERANCE`; `negative_headroom_steps` is a recorded list that the
readability gate reads rather than a value checked against a constant.

| control | claim it pins |
| --- | --- |
| `persistence_divergence` | `max abs(hold_45 - persistence_position)` |
| `k_invariance_at_h1` | the three differences at `h=1` are identical across all five `k` |
| `open_loop_divergence` | inherited from M3m — the k=45 rung reproduces `rssm_position` |
| `identity_residual` | `skill + deficit == headroom`, tolerance 1e-9, observed 1.421e-14 |
| `negative_headroom_steps` | the `(k, h)` cells with `headroom < 0`, recorded and judged by the gate |

`floor_divergence` is carried forward from M3m's record schema unchanged.

**The `keep_trajectories=True` claim is pinned without a second traversal.**
`open_loop_divergence` and `floor_divergence` read the sweep's arms and its
reference against the shipped M3c curves, which were measured with the flag
OFF — so a bitwise match on both proves the flag moved neither. A dedicated
`trajectory_flag_divergence` would require running every cell twice, doubling
the run to establish what those two establish for free, and a unit test pins
the same claim directly on a rig.

## 7. The run

Nine cells — `{frozen_ssl, pixel_ae, random_vit} x {seed 0, 1, 2}` — same
protocol, two phases (`measure`, `read`), records to `runs/m3n_motion/`. No
training and no new model forward passes: `hold_k` is numpy on rows already
retained. M3m's rulers re-measure of the same nine cells took a measured
**19m20s**; this run should be comparable.

Record path per cell must include the **seed**. M3m's test fixtures all used
seed 0, which hid a path collision that would have written all three seeds of
an arm to one filename, destroying 6 of 9 records and surfacing only at the
read phase after the GPU time had been spent.

The `measure` phase must propagate a failed status as a nonzero exit. M3m found
that `--phase measure` could discard a failed measure's status and exit 0,
which is the worst available outcome of a long run.

## 8. Testing requirements

Each item below is a defect class this arc has shipped, written as a
pre-commitment rather than as advice.

1. **Fixtures must be shown to discriminate.** Before asserting anything about
   a result, assert the fixture produces `clusters >= 3`, `ci_low < ci_high`,
   and `bootstrap_se > 0`. M3m's first comparison fixture yielded 2 clusters
   and an interval of exactly zero width, and two of three target mutations
   passed through it; the tell was three different quantities printing
   identical digits.
2. **No self-referential expectations.** `hold_k`'s expected value comes from a
   closed form over the test rig's own probe, never from the sweep. M3m hit
   this twice; both times the test caught one side of a two-sided mutation and
   missed the other, because both halves moved together.
3. **The fake model is stochastic.** With a deterministic fake, `seed`,
   `device` and `feature_backbone` go unbound and a seed mutation reads a
   difference of 0.0.
4. **Every test calls the code under test.** M3m shipped a test that asserted a
   tautology over random numbers and would have passed against an empty file —
   and the property it claimed to verify was false at four arms.
5. **Docstrings name only mutations the fixture can reach.** Six instances in
   this arc named a mutation inexpressible at that scope or never reached.
6. **Clear `src/`, `tests/`, `scripts/` `__pycache__`** after mutating a source
   file to check that a test bites.
7. Exact equality where exact equality is load-bearing. M3m's identity test
   nearly shipped `pytest.approx(rel=1e-6)` against a `sum`-vs-`max` mutation
   whose gap is in the 7th significant figure — which the tolerance would have
   accepted.

Test command: `.venv/bin/python -m pytest`. The exit-code registry test in
`tests/eval/test_diagnose_dynamics_script.py` gains `{47, 48}`.

## 9. Removing M3m's contaminated statistic

`src/mbfps/eval/burden.py` loses `motion_margin` and `reading_burden`'s motion
verdict, along with their tests. Deprecation in place is not enough: a live
function whose legend says a reading of −116 means `COPIES` is the same hazard
as the "both directions are sound" sentence, and that one shipped in three
places — a docstring, the legend written into `burden.txt`, and M3m spec §3.3,
which M3n would otherwise inherit. Conditioning was the right fix while the
records had to stay readable; leaving the function live now that a correct
replacement exists converts a corrected error into a standing trap.

What stays untouched: `burden()`, `compounding()`, `identity_residual()` and
the whole ladder. Those were right. M3n's `deficit` *is* `burden`, and §3.1's
identity is the same algebra one level up.

`scripts/prediction_burden.py` keeps parsing M3m's records — they are on disk
at `runs/m3m_burden/` and `runs/m3m_burden_rulers/`, and the M3m spec cites
them — but its Reading H path returns a pointer to M3n instead of a verdict.

## 10. New code

    src/mbfps/eval/headroom.py        statistic, reading, formatter
    scripts/motion_headroom.py        measure / read / all phases, protocol check, seeds gate

Named for the quantity M3n introduces, following the house pattern
(`burden.py`, `capacity.py`, `retention.py`, `width.py` are each named for
their statistic).

**`src/mbfps/eval/motion.py` and `scripts/latent_motion.py` are M3i's and are
not touched.** An earlier draft of this spec named `src/mbfps/eval/motion.py`
for M3n, which would have clobbered a shipped module — the same hazard class as
M3i overwriting M3h's brief directory. `src/mbfps/eval/ladder.py` is also
already taken.

Reuses `pooling.episode_bootstrap` and `pooling.percentile_interval`, which M3m
made public for exactly this, and `pooling.clustered_interval` — M3m's
`burden.margin_interval`, which is *already* an episode-clustered bootstrap over
an arbitrary `(windows, horizon)` array. §3.5's ruler is that function
generalised and rehoused, not new machinery.

Exit codes 47 and 48 are confirmed free: 43 and 44 are
`scripts/latent_capacity.py`'s, 45 and 46 are `scripts/prediction_burden.py`'s,
and nothing in `scripts/` uses a higher number.

## 11. What each outcome licenses

| majority verdict | what it licenses |
| --- | --- |
| `AT_PERFECT` | the one-step map is as good as the readout permits; all remaining error is compounding, and a multi-step or overshooting objective is the indicated intervention |
| `BETWEEN` | a real one-step deficit; fix one-step quality before paying ~13h of training for a multi-step objective |
| `AT_COPYING` | the map genuinely does not predict motion — M3m's original hypothesis, now on a fair ruler |
| `UNREADABLE`, exit 47 | the probe, not the model, is the limit; M3o redirects to the readout rather than to training |

`UNREADABLE` is a real possibility, not a formality. `pixel_ae_seed1`'s
`headroom(45, 1)` is 0.575649 map units: a perfect one-step predictor beats a
copying one by about half a map unit there, against a readout error of 259.06.
If several cells land in that regime, M3n's answer is that this instrument
cannot see one step — still worth knowing before committing thirteen hours,
because every candidate change to the objective would be graded by the same
probe.

## 12. What M3n does not answer

- **Whether a multi-step objective helps.** M3n places the one-step map on the
  copying-to-perfect axis and nothing more.
- **The M3 exit gate.** `beats_persistence`, `band_is_usable` and
  `filtering_beats_embedding` all still fail, and `gap_closed(45)` is negative
  in 9 of 9 cells. M3n gives the k=45 corner of that surface its first ruler;
  it does not move the gate.
- **Why the one-step skill varies across seeds.** The recovered share at `h=1`
  spans −2.4% to 72.7% across the nine cells with no clean backbone pattern —
  it crosses all three arms. M3n reports it; explaining it is not in scope.

## 13. Re-reporting M3m's resolvability under the clustered ruler

M3m's SE multiples were computed with `_spread`, which §3.5 shows understates
the standard error by 1.26x to 2.17x on the one per-window array it shipped.
M3n measures `deficit`, which *is* `burden`, with per-window rows — so it
re-reports M3m's two headline resolvability claims under the clustered ruler at
no extra cost:

- **"`burden(1)` is not resolvable from zero at 2 SE in 9 of 9 cells, max 1.94
  SE."** A claim of non-resolvability made with an understated SE is
  conservative, so this stands and strengthens.
- **"`compounding(45)` is resolvable at 5.56-11.63 SE."** A claim of
  resolvability made with an understated SE is overstated. The direction is
  safe — even dividing by the largest inflation measured, 5.56/2.17 = 2.56,
  still clears 2 — but the headline multiples are too large and M3n should
  publish the corrected ones.

The inflation factors in §3.5 are measured on `window_margin`, a different
quantity with its own clustering structure. They are cited to establish that
the effect is real and of this order, not to rescale M3m's numbers by analogy.
M3n measures the factors it reports.

## 14. By-product worth recording

The k=45 column of the ladder is computable from the shipped M3m records
without any new measurement, and it is the `gap_closed` profile the M3 gate has
only ever read at its last point. It crosses zero between `h=3` and `h=15` in
every cell — positive early, −38.8% to −116.3% at `h=45`. The gate reads the
one horizon at which the answer is most negative. M3n's record carries the
whole profile so that this is visible without re-deriving it.
