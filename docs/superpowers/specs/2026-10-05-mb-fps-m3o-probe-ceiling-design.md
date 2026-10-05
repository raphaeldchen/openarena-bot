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

- **The criterion-4 probe** is fit on REAL encoder embeddings at frame `t` and
  reports `embedding_selection_r2`, measured at **0.320 to 0.345** across the
  nine cells.
- **The rollout probe** is the one `evaluate_rollout` takes. It is fit on the
  model's OWN PREDICTED embeddings (`rollout.py:195`), because all three
  rollout references pass through the same embedding head, and fitting on real
  embeddings and applying to predicted ones was "destroying the signal."

**The r2 of the second probe, on the distribution it is actually applied to,
is recorded nowhere.** What is recorded is its error in map units: the floor at
h=1 spans **131.88 to 259.06** across the nine cells — nearly 2x — against a
`embedding_selection_r2` that spans 8%. Something differs between the two
distributions, and no milestone has named it.

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

Every cell reports **both** r2 and median map-unit error, so results are
directly comparable to M3n's 131-259 rather than living in a different unit.

| feature set | linear ridge | kernel ridge (RBF) | k-NN |
| --- | --- | --- | --- |
| raw pixels at `t` | A | B | — |
| encoder embedding at `t` | C (the criterion-4 probe, 0.320-0.345) | D | yes |
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

**An RBF kernel approaches the linear kernel as its bandwidth grows**, so the
linear probe is nearly a special case and `D >= C` is close to guaranteed by
construction. **"D > C" is therefore not a finding.** The finding is the
magnitude, and whether it is resolvable. This is the trap M3n walked into from
the other side: `skill`'s point estimate was positive in 8 of 9 cells and meant
nothing until the interval was drawn.

### 3.4 Split discipline

Three splits — fit / select / score — as `probe.gain_from_blocks` already
enforces. Capacity is selected on a split that is neither the fit nor the score
split, or a high-capacity probe selects itself in. That is exactly the confound
that makes a nonlinear probe untrustworthy as evidence.

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
a `(windows, horizon)` array and so does not apply directly to an r2
difference, which has no horizon axis; the sequence column uses
`gain_from_blocks`'s own bootstrap, and the estimator-versus-estimator
comparisons need the one new resampler M3o adds. It resamples episodes and
recomputes BOTH estimators' scores on each draw, so the two are paired within a
resample rather than compared across independent ones.

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

| control | claim it pins |
| --- | --- |
| dual equals primal at every rung of `RIDGES` | the prerequisite changed no arithmetic |
| cell `C` reproduces the shipped `embedding_selection_r2` (0.320-0.345) | M3o's harness measures what the study measured, not something adjacent |
| RBF and k-NN at their most-regularised ladder end approach the linear result | the kernel nests the linear probe, as claimed |
| **the raw-pixel row is identical across all nine cells** | it uses no model; a difference means a model leaked into a model-free measurement |
| the zero-information baseline recomputes to 283.05 median | the scale every map-unit figure is read against |

The fourth is free and bitwise, of the same kind as M3n's two reductions.

## 7. Code

```
src/mbfps/eval/ceiling.py      the estimators, the grid, the reading, the formatter
scripts/probe_ceiling.py       measure / read / all phases, record schema
```

Named for the quantity, following `burden.py`, `capacity.py`, `retention.py`,
`width.py`, `headroom.py`.

**Exit codes.** 43-44 are `latent_capacity`'s, 45-46 `prediction_burden`'s,
47-48 `motion_headroom`'s, and nothing uses a higher number, so:

- **49 `EXIT_HARNESS_DISAGREES`** — cell `C` failed to reproduce the shipped
  `embedding_selection_r2` (§6). M3o is then measuring something other than
  what the study measured, and no verdict is available.
- **50 `EXIT_NO_MAJORITY`** — no verdict held a strict majority of the nine
  cells.

The three model verdicts of §5 all exit 0: each is an answer.

Reused: `fit_probe` and `RIDGES`, `probe_targets`, `probe_r2`, `episode_split`,
`gain_from_blocks` for the sequence column, `pooling.clustered_interval` for
intervals. New: the dual path, the RBF kernel, the k-NN readout, and an
estimator-versus-estimator comparison — `gain_from_blocks` compares two feature
sets under one estimator, where M3o needs one feature set under two estimators.

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
