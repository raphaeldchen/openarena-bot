# M3o: the probe's own ceiling — design

**Date:** 2026-10-05
**Branch:** `feat/m3o-probe-ceiling`, off `4956f59`
**Predecessor:** M3n, `docs/superpowers/specs/2026-10-04-mb-fps-m3n-motion-headroom-design.md`

## 1. The question

M3n returned `NO_MAJORITY` (exit 48). Its readability gate failed in 4 of 9
cells because a perfect one-step predictor beats a copying one by only
**0.576 to 11.824 map units** against readout errors of **131.88 to 259.06**.
`skill` at the decision cell was resolvably positive in 1 of 9 cells where its
point estimate was positive in 8.

Thirteen milestones have measured the world model through that readout. M3o
asks what the readout itself can do.

Three explanations are open, and they call for different responses:

1. **The linear probe family is too weak.** A stronger readout would recover
   more, and the arc's negative findings need re-reading.
2. **The representations do not carry position.** The encoders discard what the
   frames contain — an M2 problem, not an M3 one.
3. **Position is not recoverable from a single frame in this scenario.**
   `my_way_home` is a maze; if distinct positions produce near-identical views,
   no readout of any kind separates them.

The third is the one no milestone has tested, and it would explain the
observation that prompted this: three different information sets — the encoder
embedding alone, the posterior latent alone, and both jointly — all land within
0.03 of each other at r2 ~0.33. That is the signature of a ceiling, not of a
difference between representations.

## 2. The two probes, and which one swamped M3n

This codebase has **two** probes, and conflating them is the first way M3o can
go wrong.

- **The criterion-4 probe** is fit on REAL encoder embeddings at frame `t`,
  **fit on the 20 TRAINING episodes and scored on validation**
  (`study.py` calls `filtering_report(model, train_paths, val_paths, ...)`). It
  reports **`filtering.criterion_4.embedding_r2`**, measured at **0.3195 to
  0.3451** across the nine cells of `runs/m3_study_v2`.

  **NOT `probe.embedding_selection_r2`**, which an earlier revision of this very
  section named — the conflation this section exists to prevent, made inside it.
  That field spans **0.0180 to 0.3829** over the same nine cells, a factor of 21,
  and is a different quantity. Every reconciliation in M3o reads
  `filtering.criterion_4.embedding_r2`.
- **The rollout probe** is the one `evaluate_rollout` takes. It is fit on the
  model's OWN PREDICTED embeddings (`rollout.py:195`), because all three
  rollout references pass through the same embedding head, and fitting on real
  embeddings and applying to predicted ones was "destroying the signal."

**The r2 of the second probe, on the distribution it is actually applied to,
is recorded nowhere.** What is recorded is its error in map units: the floor at
h=1 spans **131.88 to 259.06** across the nine cells — nearly 2x — against a
`criterion_4.embedding_r2` that spans 8% (0.3195 to 0.3451). Something differs
between the two distributions, and no milestone has named it.

For scale: a readout with ZERO information, predicting the training mean,
scores **292.82 mean / 283.05 median** map units on the validation set (12,073
frames, 24 episodes). So the shipped probe's floor error is 45% to 88% of the
zero-information baseline. It carries real information. It is not broken. But
its quality varies by a factor of two across the very cells whose statistics
get compared.

## 3. What M3o measures

### 3.1 The instrument's quality on the applied distribution (first)

Per cell: the r2 and the median map-unit error of the shipped rollout probe
where `evaluate_rollout` actually uses it — on predicted embeddings — against
the 283.05 zero-information baseline.

**This is measured FIRST and the rest is read in its light.** The 2x2 below is
measured on real encoder embeddings at frame `t`; the number that swamped M3n
is the error on predicted embeddings over an imagined rollout. If those differ
substantially, a better probe on the first distribution says little about the
second, and the 2x2 would be answering an adjacent question.

### 3.2 The grid

Every cell reports **both** r2 and map-unit error, so results are comparable to
M3n's 131-259 rather than living in a different unit.

**And the map-unit error is reported as BOTH a mean and a median, because M3n's
131.88-259.06 is a MEAN.** `rollout.py:237` averages the per-window position
error across windows at each horizon step, so a median reported beside that
figure would be a different statistic wearing the same unit. The comparison M3o
exists to make runs through the mean; the median is carried beside it because it
is robust to the tail that an extrapolating readout produces, and because
`ZERO_INFORMATION_MEDIAN` is a median. Every ratio states which statistic sits
on each side.

| feature set | linear ridge | kernel ridge (RBF) | k-NN |
| --- | --- | --- | --- |
| raw pixels at `t` | A | B | — |
| encoder embedding at `t` | C (best linear; `C_shipped` reconciles, §3.3.1) | D | yes |
| window of frames `t-k..t` | E | F | — |

- **A vs C** — does the encoder lose anything a linear probe could have used?
- **C vs D** — was the linear family the limit?
- **B** — the ceiling for any readout of one frame.
- **E, F** — does history resolve what one frame cannot?

**The raw-pixel row is model-independent.** It uses no checkpoint, so it is one
number for the whole study rather than nine — which also makes it a free
control (§6).

### 3.3 The estimators

**Linear ridge** is the shipped `fit_probe`, unchanged.

**Kernel ridge (RBF)** is the same estimator lifted into a nonlinear feature
space, so `C` vs `D` is a capacity comparison WITHIN one family rather than
across two fitting procedures. Closed form, no SGD, no seeds. Its bandwidth
ladder is selected on held-out episodes exactly as `RIDGES` is. It is also
literally the dual solve of §4, so the memory fix and the nonlinear probe are
one piece of machinery.

**k-NN** has a `k` ladder and no fitting. It asks whether near-in-feature-space
implies near-in-position, which is the aliasing question directly.

**The kernel ladder CONTAINS the linear probe as a rung**, so `D >= C` holds by
construction up to selection noise.

An earlier draft argued this asymptotically — an RBF approaches the linear
kernel as its bandwidth grows — and Task 2 measured that the argument fails in
practice. With a fixed diagonal jitter the widest RBF rung is effectively
under-regularised relative to what the linear probe selects, and the RBF came
out **worse** than linear once linear r2 exceeded about 0.37, by 0.002 to 0.004
on one fixture and 0.05 to 0.07 on another. A negative `D - C` for instrument
reasons would make `PROBE_LIMITED` unreachable even where the probe family IS
the limit.

So the ladder is `{linear kernel} u {RBF at each bandwidth}`, each over
`CEILING_RIDGES` (§3.3.1), selected together on held-out data. The search space
then strictly contains the linear reference's and the nesting is exact rather
than asymptotic: measured, `D - C` is +-0.0000 where the fixed-jitter version
read -0.0021 to -0.0040, and the ladder selects the linear rung when nothing
nonlinear helps. Each kernel is eigendecomposed once, so the jitter ladder
costs O(n^2) per rung instead of another O(n^3) solve. **"D > C" is therefore not a finding.** The finding is the
magnitude, and whether it is resolvable. This is the trap M3n walked into from
the other side: `skill`'s point estimate was positive in 8 of 9 cells and meant
nothing until the interval was drawn.

#### 3.3.1 Both sides search the same ridge grid, and it is finer than the shipped one

Nesting the linear kernel made `D >= C` exact ON THE SELECTION SPLIT. It did not
make `D - C` mean "nonlinear capacity", because both sides were still searching
ridges on a grid too coarse to resolve the optimum, and the RBF rungs could
reach regularisation strengths the linear reference could not.

`probe.RIDGES` is `(1e-1, 1e1, 1e3, 1e5, 1e7)` — a 100x step, not a decade. On a
purely linear task at n=2000 the best ridge sits at 3.16e3 (D=1024) and 5.62e3
(D=2048): between the grid's 1e3 and 1e5 rungs, a factor of 3 to 6 from either.
Measured on the embedding-shaped linear fixture, scored on a third split:

| n=2000 | floor `D - C`, 100x step | floor, 10^0.25 step | gain an oracle ridge buys `C` |
| --- | --- | --- | --- |
| D=1024 | **+0.0109** | -0.0001 | +0.0110 |
| D=2048 | **+0.0201** | -0.0000 | +0.0208 |
| D=37632 | +0.0000 | +0.0000 | — |

**The floor and the oracle gain are the same number.** That is the proof of
mechanism: the RBF ladder was not finding nonlinear structure, it was
interpolating the ridge grid. Where the ridge response has no sharp interior
optimum — the raw-pixel row, D >> n — there is no gap to interpolate and no
floor even on the coarse grid. Refining the step removes it at every shape
measured, and `C` rises to the oracle's value exactly: 0.4964 at D=1024 (from
0.4855), 0.5164 at D=2048 (from 0.4957).

So `CEILING_RIDGES` is **33 rungs at a 10^0.25 step over the shipped span
[1e-1, 1e7]**, and BOTH `C` and the ladder's jitters use it. The span is
unchanged: the optimum was interior at every shape measured, 0 of 6 draws at an
endpoint. Only the step was wrong.

**`probe.RIDGES` IS NOT TOUCHED.** Thirteen milestones of recorded numbers came
through that constant and `fit_probe` keeps it as its default; the finer grid is
M3o's, passed in. Restating every prior result is not in M3o's scope.

`CEILING_RIDGES` **contains `probe.RIDGES` bitwise**, as every 8th rung — a
property of the span and step, verified rather than assumed. Because
`_solve_ladder` writes each rung's diagonal from a pristine copy rather than
from the previous rung, selecting over only those five rungs of the fine ladder
reproduces `fit_probe` to 0.0 in both the primal and the dual branch. So the
reconciliation anchor is free:

- **`C`** — best over all 33 rungs. The best LINEAR readout, and the thing
  `D - C` is read against, so that the difference isolates nonlinearity.
- **`C_shipped`** — best over the 5 shipped rungs, recorded as
  `shipped_grid_r2`. The reconciliation against the study (§6), and what exit 49
  reads.

`C` is therefore **not** the shipped probe's r2 and is expected to exceed it by
up to ~0.02 at representation scale. That is the intent, and it is why the §6
reconciliation runs through `C_shipped`.

A side finding, recorded and out of scope: the shipped grid leaves +0.011 to
+0.021 of r2 unclaimed at representation scale, and `aggregate.ridge_groups`
and `study.py` already document gate criterion 4's gain SIGN flipping across
that grid (-0.0706, -0.0208, +0.0325) because the effect is ~0.02. Those are the
same order. M3o does not re-measure the gate.

### 3.4 Split discipline

Three splits — fit / select / score — as `probe.gain_from_blocks` already
enforces. Capacity is selected on a split that is neither the fit nor the score
split, or a high-capacity probe selects itself in. That is exactly the confound
that makes a nonlinear probe untrustworthy as evidence.

**THE FIT SPLIT IS THE STUDY'S 20 TRAINING EPISODES, NOT A THIRD OF VALIDATION.**
An earlier revision divided the 24 validation episodes three ways. That starves
the fit: criterion 4 fits on the 20 TRAINING episodes, so an 8-episode
validation fit is both a smaller and a different sample, and `C_shipped` then
reads **-0.026 to 0.166** against the 0.3195-0.3451 band it is supposed to
reconcile with — a gap of 0.17 to 0.35 in every one of the nine cells. Exit 49
would have fired on all nine of a correct run and M3o would have returned no
verdict at all.

Measured: fit on the study's own 20 training episodes, the same rows and targets
read **0.300 to 0.342**, worst gap 0.035 against the band. So the split is:

| split | episodes | role |
| --- | --- | --- |
| fit | the 20 TRAINING episodes, strided to the frame budget | where the weights come from |
| select | 12 validation episodes | where ridge, bandwidth and k are chosen |
| score | the other 12 validation episodes | where every reported number is read |

Selection and scoring stay entirely within validation and share no episode, so
the three-split discipline is unchanged — it is only the fit set that moves, to
the one criterion 4 actually uses. Validation splits in halves rather than
thirds, so the intervals are drawn on 12 episodes instead of 8.

The effective sample size here is the EPISODE count, not the frame count: this
project measured a 1.26x-2.17x clustering penalty for treating adjacent Doom
frames as independent, and 8 episodes of near-duplicates is a far smaller sample
than its ~4,000 rows suggest.

**Episodes, never frames.** Adjacent Doom frames are near-duplicates, and this
project has a measured 1.26x-2.17x penalty for pretending otherwise
(M3n §3.5).

## 4. Prerequisite: `probe._solve` cannot run at this scale

`_solve` materialises `ridge * np.eye(design.shape[1])`. At D = 37,632 — a
112x112x3 frame — that single array is **11.3 GB**, `design.T @ design` is
another 11.3 GB, and their sum another. `fit_probe` calls `_solve` once per
candidate in `RIDGES`, five times. A first attempt at the raw-pixel cell took
18 GB of swap and was stopped.

**This is not a latent bug.** Every existing caller probes latents or
embeddings at D ~1536-2048, where D^2 is 33 MB and the primal form is natural.
Thirteen milestones of numbers came through `_solve` without trouble. M3o's
raw-pixel cell is simply the first caller where D exceeds n.

The fix is an exact algebraic reformulation, not an approximation:

```
primal (D x D):  w = (X'X + lambda I_D)^-1 X'y     <- today
dual   (n x n):  w = X'(XX' + lambda I_n)^-1 y
```

Verified identical across the whole `RIDGES` ladder: max absolute prediction
difference 5.164e-09 at ridge 1e-1, 6.182e-11 at 1e1, 1.236e-12 at 1e3,
3.197e-14 at 1e5, 3.553e-15 at 1e7. Agreement is WORST at the smallest ridge,
so a test must cover every rung rather than a representative one.

Three changes, in order of impact:

1. **Solve in whichever space is smaller**, on `min(n, D)`. The primal path
   stays for the D < n callers; this is the same discipline as M3n's
   `_spread` -> `standard_error` rename, and must be proven behaviour-free.
2. **Never materialise `np.eye`.** `g.flat[::g.shape[0] + 1] += ridge` adds to
   the diagonal in place, saving one full allocation in either form.
3. **Hoist the Gram out of the ridge loop.** `fit_probe` recomputes
   `design.T @ design` five times; compute it once and add each ridge to a
   copy of the diagonal.

Because the intercept is unregularised, the dual is taken on centred data and
the intercept recovered as `y.mean(0)`. `fit_probe` already standardises `x` to
zero column means on the fit split, so this is exact rather than an
approximation.

**The cost flips, and the spec states the budget rather than discovering it.**
The dual costs n^2: at 12,073 validation frames that is 1.17 GB, at 20,000 it
is 3.2 GB. **M3o's frame budget is 12,000 frames**, subsampled by a fixed
stride from the validation episodes, recorded in every record. Only the
raw-pixel row needs the dual at all.

## 5. The reading

Per the resolvability of each difference against an episode-clustered interval:

| verdict | condition | meaning |
| --- | --- | --- |
| `PROBE_LIMITED` | `D - C` resolvably > 0 | the linear family was the limit |
| `REPRESENTATION_LIMITED` | `B - D` resolvably > 0 | the observation carries more than the representation preserves |
| `OBSERVATION_LIMITED` | none of B, C, D resolvably separated | the scenario aliases; no readout recovers more |

Crosscutting rider: **`F - B` resolvably > 0** means history resolves what one
frame cannot. That is gate criterion 4's question asked without criterion 4's
"A beats B" framing, which the shipped `gain` measurement already shows is the
wrong shape — the latent ADDS a resolvable +0.024 to +0.076 r2 in 4 of 9 cells
while losing "latent beats embedding" in 8 of 9.

**How a model-free row combines with nine model cells.** `A` and `B` are
single numbers for the whole study (§3.2), while `C`, `D`, `E` and `F` are nine
each. A difference involving a model-free cell is formed **per model cell** —
`B - D_i` for each `i` — giving nine differences from one `B`. The verdict is
then a strict majority over those nine, `strict_majority(len(cells))` computed
and never stored, as in M3n. The shared `B` makes the nine differences
correlated, which the record states; it does not make them invalid, because
each asks a question about its own cell's representation.

**The interval instrument.** Every difference is read from an
**episode-clustered bootstrap percentile interval** over the score split's
episodes, at `CONFIDENCE = 0.95` and `RESAMPLES = 2000`, drawn at a per-cell
seed derived as M3n derives it (`crc32` of base, arm and seed — never `hash()`,
which is salted per process). `pooling.clustered_interval` reads one column of
a `(windows, horizon)` array and so does not apply to an r2 difference, which
has no horizon axis. **M3o needs no new resampler**: `probe._block_bootstrap_ci`
already takes TWO prediction arrays and the targets, resamples `groups` with
the probes held fixed, and returns a percentile interval on the r2 difference.
Its parameter names say `joint_predicted` and `embedding_predicted`, but the
computation is generic in the two arrays, so the estimator-versus-estimator
comparison passes the RBF's and the linear's predictions on the same scored
rows. The sequence column uses `gain_from_blocks`, which wraps the same
helper.

### 5.1 The falsifiable target

`headroom = hold - floor`, and BOTH terms shrink as the probe improves. With a
perfect readout, `hold` is the true position at `h-1` and `floor` is the true
position at `h`, so headroom converges to the **true one-step displacement,
3.9694722203504225 map units** — the constant M3n recorded and deliberately
used in no statistic.

M3n measured headroom at 0.576 to 11.824, which BRACKETS 3.9695. So the readout
noise is not shifting the signal, it is dispersing it. A better probe therefore
helps M3n by collapsing the spread rather than by enlarging the signal, and
M3o has a prediction it can be wrong about: **as the readout improves, headroom
converges toward 3.9695 from both sides and its interval tightens.**

### 5.2 Optional final phase, gated

**Only if §5 returns `PROBE_LIMITED`:** recompute M3n's decision cell with the
better readout on the same nine checkpoints and report whether `headroom`
becomes resolvable in a strict majority. No training; about 20 minutes at
M3n's measured 2.28 min/cell. It turns "the probe was the limit" into "and here
is the measurement it rescues." Conditional, so it costs nothing on the other
two verdicts.

## 6. Controls

### 6.1 The null control: what `D - C` reads when there is nothing to find

Task 2 measured the kernel ladder beating the linear reference **on a purely
linear task** by +0.012 at D=1024 and +0.020 at D=2048, n=2000 — a **false
`PROBE_LIMITED`**, having nothing to do with nonlinear structure in the
representation, which is what the verdict is supposed to mean.

§3.3.1 found the cause and removed it: both sides were searching ridges on a
grid too coarse to resolve the optimum. With both on `CEILING_RIDGES` the floor
measures -0.0001 to +0.0000 at those same shapes.

**The control stays anyway, and is measured per cell** — not because the floor
is expected to be nonzero, but because it is the only thing that catches this
class of fault at a shape §3.3.1 did not measure. The floor proved sharply
shape-dependent: ~0 at D=37632 even on the coarse grid, largest at D=2048. A
figure measured at three shapes does not license the other six cells.

Each cell fits the whole ladder on a SYNTHETIC purely-linear task at its own
`n` and `D` and records that `D - C` as `null_floor`. `PROBE_LIMITED` requires
the real `D - C` resolvably above **that**, not above zero.

**A NEGATIVE FLOOR CLAMPS TO ZERO.** The floor is a bar to clear, not a
correction to apply. Subtracting a measured `-0.0016` would ADD to the gain and
make `PROBE_LIMITED` easier to reach than reading against zero — the opposite of
what the control is for. The bar is `max(null_floor, 0)`; a positive floor is
subtracted exactly.

**AND IT IS AVERAGED OVER THREE DRAWS.** A single draw of the floor is itself
noisy, and the floor is a nuisance parameter the verdict is read against, so it
is worth estimating more precisely than the quantity it bounds. Three draws at
the cell's own shape, averaged.

**AVERAGE FIRST, THEN CLAMP — the two do not commute.** Draws of -0.002, -0.001
and +0.003 average to +0.0000 and clamp to 0.0; clamped first they average to
+0.001, a bar the instrument never showed. `null_floor` returns the unclamped
mean and the clamp lives downstream, in the one place the bar is formed, so the
order cannot be got wrong by a caller.

One thing the floor cannot fix: the measured `D - C` has its own draw noise. On
purely linear data at n=300, D=150, `D - C` exceeded +0.003 in 4 of 40 draws, and
one draw read +0.0085 with an interval of [+0.0043, +0.0133] against a correctly
zero floor — a resolvably positive gain with nothing nonlinear present. At
n=1200, D=512 the same spread is -0.0001 to +0.0011. The real cells sit at the
larger end, and the strict majority of five is what makes one such cell
harmless, but **a reduced-frame smoke run can legitimately show a spurious
`PROBE_LIMITED` cell** and that is not a bug.

A `null_floor` that comes back materially above zero is a finding about the
instrument, not a number to subtract and move on from: it means §3.3.1's fix is
incomplete at that shape.

| control | claim it pins |
| --- | --- |
| dual equals primal at every rung of `RIDGES` | the prerequisite changed no arithmetic |
| `D - C` on a synthetic linear task at this cell's `n` and `D` | the floor `PROBE_LIMITED` must clear. ~0 once both sides share `CEILING_RIDGES`; it was +0.012 to +0.020 on the shipped grid, so this is what keeps that fix honest per cell |
| `C_shipped` reproduces the shipped `filtering.criterion_4.embedding_r2` (0.3195-0.3451), within 0.10 | M3o's harness measures what the study measured, not something adjacent. `C` searches a finer grid and is expected ABOVE that band (§3.3.1). The tolerance is set against a measured worst gap of 0.035 once the fit split is the study's training episodes (§3.4) |
| `C >= C_shipped` on the SELECTION split, exactly | containment: `CEILING_RIDGES` contains `RIDGES`. On the SCORE split it may dip by selection noise, but a large systematic dip means the finer grid is overfitting selection, so the score-split difference is recorded per cell |
| `_solve_ladder` over `CEILING_RIDGES`, at the 5 shipped rungs, is bitwise `_solve_ladder` over `RIDGES` | `CEILING_RIDGES` ADDED rungs and changed no arithmetic; `ridges=` is a pure extension |
| RBF and k-NN at their most-regularised ladder end approach the linear result | the kernel nests the linear probe, as claimed |
| **the raw-pixel row is identical across all nine cells** | it uses no model; a difference means a model leaked into a model-free measurement |
| the zero-information baseline recomputes to 283.05 median, **and its mean is recorded beside it** | the scale every map-unit figure is read against. Two statistics because the readout errors have two; a mean error over a median baseline is not a ratio |

Two are free and bitwise, of the same kind as M3n's two reductions: the
raw-pixel row's identity across the nine cells, and `C_shipped`'s agreement with
the fine ladder's shipped rungs.

## 7. Code

```
src/mbfps/eval/ceiling.py      the estimators, the grid, the reading, the formatter
scripts/probe_ceiling.py       measure / read / all phases, record schema
```

Named for the quantity, following `burden.py`, `capacity.py`, `retention.py`,
`width.py`, `headroom.py`.

**Exit codes.** 43-44 are `latent_capacity`'s, 45-46 `prediction_burden`'s,
47-48 `motion_headroom`'s, and nothing uses a higher number, so:

- **49 `EXIT_HARNESS_DISAGREES`** — `C_shipped` failed to reproduce the shipped
  `embedding_selection_r2` (§6). M3o is then measuring something other than
  what the study measured, and no verdict is available. Read on `C_shipped`,
  never on `C`: `C` searches the finer `CEILING_RIDGES` and is expected above
  the shipped band, so gating on it would exit 49 on a correct run.
- **50 `EXIT_NO_MAJORITY`** — no verdict held a strict majority of the nine
  cells.

The three model verdicts of §5 all exit 0: each is an answer.

Reused: `fit_probe` and `RIDGES`, `probe_targets`, `probe_r2`, `episode_split`,
`gain_from_blocks` for the sequence column, `_block_bootstrap_ci` for every
interval, and `_fit_and_score` — which already factors out the three-split
discipline for ONE feature set and is the seam the new estimators plug into.
New: the dual path, the RBF kernel, the k-NN readout, and a dispatching
`apply_readout`.

## 8. What each outcome licenses

- **`PROBE_LIMITED`** — the arc's negative readings need re-reading through the
  better probe, gate criteria 1 and 4 included. §5.2 makes that concrete for
  M3n.
- **`REPRESENTATION_LIMITED`** — the encoders discard position the frames
  carry. That is an M2 representation problem, and it would mean thirteen
  milestones have been debugging the wrong stage.
- **`OBSERVATION_LIMITED`** — position is not recoverable from one frame here.
  The gate is asking for something the observation cannot support, and either
  the gate changes or the observation does: more context, or a different
  scenario.

## 9. What M3o does not answer

- **Whether a better probe would help a POLICY.** A nonlinear readout can
  recover information a downstream controller could not use. M3o measures what
  is recoverable, not what is usable. The same confound class as M2's
  feature-scale issue and M3's single-pipeline rule.
- **Whether the world model is good.** Every cell here is a statement about the
  instrument or the representation at a single frame. The dynamics are not
  measured.
- **The M3 exit gate.** It stands at: criteria 2 and 5 pass, 1 and 4 fail, 3 is
  reported and degenerate (2 reward events in 9,949 steps). M3o may change how
  1 and 4 are READ; it does not re-run them.
