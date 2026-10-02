# MB-FPS M3m — is the rollout compounding, or did the one-step map never learn motion?

## 1. The question, and what M3l left

M3l refuted the **bottleneck** lever on evidence: `bits_carried` reads at most
**1.11%** of the exact 160-bit ceiling, with the whole interval below the 80-bit
cut in **9 of 9** cells and the largest `ci_high` anywhere 28× clear of it. A
code using ~1% of its capacity is not capacity-limited. That left the
**objective** lever — but standing by elimination, not on its own bar:
`FRAME_REENCODING` did not clear either (`frame_low > 0.5` in 0 of 9).

The hypothesis inherited from that milestone was "the loss targets the current
frame and never asks for motion". **As stated, it is false, and the data to show
that already existed.** `train_world_model` records
`kl_rate_above_free_bits` — the fraction of steps on which the dynamics KL
cleared the 0.20-nat free-bits floor and therefore contributed gradient — with a
comment anticipating this exact question: *"a run where it never happens trained
no dynamics prior at all … Record it rather than assume it."* Across the nine
20,000-step cells of `runs/m3_study_v2`:

| arm | `kl_rate_above_free_bits` | `kl_dyn_max` (nats) |
|---|---|---|
| `frozen_ssl` s0/s1/s2 | 0.900 / 0.781 / 0.746 | 36.22 / 15.71 / 1.87 |
| `pixel_ae` s0/s1/s2 | 0.913 / 0.832 / 0.842 | 2.95 / 11.41 / 14.22 |
| `random_vit` s0/s1/s2 | 0.971 / 0.976 / 0.956 | 27.57 / 6.19 / 10.83 |

**0 of 9** cells fall below the 0.5 warning threshold and **0 of 9** failed to
clear the 0.20-nat floor. The dynamics prior trained, on 75–98% of steps. The
loss does ask for one-step latent prediction, and it was answered.

What survives is narrower. The loss asks for **one-step** latent agreement; the
gate reads **45-step** rollout position error. Evaluating the gate's own metric
at every horizon, from the curves already stored in those same records:

| | k=1 | k=2 | k=3 | k=5 | k=8 | k=10 | k=15 | k=20 | k=30 | k=45 |
|---|---|---|---|---|---|---|---|---|---|---|
| cells with `gap_closed(k) > 0` | **8/9** | 8/9 | 6/9 | 5/9 | 4/9 | 2/9 | 1/9 | **0/9** | 0/9 | 0/9 |
| median persistence-to-floor band | 3.9 | 9.9 | 12.8 | 18.6 | 17.0 | 24.1 | 29.2 | 39.5 | 48.4 | 53.8 |

**The gate criterion that has failed for ten milestones passes at short
horizon.** At k=1, eight of nine cells are positive, spanning +0.08 to +0.73, with four of
the nine above +0.38.
`gap_closed(45)` is not measuring a model that learned nothing; it is measuring
the far end of a curve that starts out working. It also explains why
`random_vit` is "least bad" at k=45: once every cell is worse than persistence,
what is being ranked is how fast each degrades, not how well any predicts.

Note too that the two failing gate criteria fail at **opposite ends of the same
curve**. `band_is_usable` fails at short horizon, where the signal is real but
the denominator is 3.9 wide; `gap_closed` fails at long horizon, where the
denominator is ample but the signal has inverted. `pixel_ae` seed 1 shows what a
vanishing denominator does to the ratio: `+3.39` at k=2 and `−3.36` at k=3.

So M3m asks one question with two answers that demand **opposite**
interventions:

- The one-step map predicts real motion, and rolling it forward destroys that.
  → error **compounding**; a multi-step or overshooting objective is indicated.
- The one-step map never learned motion, and looks adequate at k=1 only because
  one step of displacement is small. → a longer-horizon term cannot help; the
  **target** must change.

Choosing an intervention before distinguishing these risks spending ~13 hours of
training (one cell at 20,000 steps took 1.47 h) on the wrong one.

## 2. What is measured

### 2.1 The prediction-burden ladder

Three curves at each horizon k, differing in exactly one variable — how much
prediction is demanded. Every one routes its final latent through the **same**
embedding head and the same position readout.

| curve | observes real frames through | final draw comes from | status |
|---|---|---|---|
| `floor(k)` | t+k | the posterior, which saw frame t+k | already recorded |
| `teacher_forced(k)` | t+k−1 | the prior, one step from the true state | **the one new arm** |
| `open_loop(k)` | t only | the prior, k steps from its own draws | already recorded as `rssm_position` |

The decomposition is **exact by construction**:

```
open_loop(k) − floor(k) = [teacher_forced(k) − floor(k)] + [open_loop(k) − teacher_forced(k)]
                        =       one_step_cost(k)         +     compounding_cost(k)
```

`one_step_cost(k)` is what one step of prediction costs at time t+k.
`compounding_cost(k)` is what it costs that the preceding k−1 draws were
uncorrected by observation rather than corrected by it. Both are **paired**
differences over identical windows with identical weights, not two independent
estimates — the distinction that stalled M3k, whose statistic had an
effect-to-noise ratio of 0.22.

### 2.2 `motion_margin` — and the baseline that makes it fair

The natural-looking comparison is `teacher_forced(k)` against the recorded
`persistence_position(k)`. **That comparison is rigged and must not be used.**
Recorded persistence copies the position at t, the last frame the *open-loop*
rollout saw, while `teacher_forced(k)` has observed through t+k−1. It would win
on information advantage rather than by predicting anything.

The fair baseline is **one-step persistence**: predict that the agent did not
move between t+k−1 and t+k. Its error is exactly the true one-step displacement,
known from ground truth with no model involved.

```
motion_margin(k) = one_step_persistence(k) − teacher_forced(k)
```

Positive means one prior step beats assuming stillness. This is the inherited
hypothesis — does the objective ever ask for motion — reduced to a level against
an exactly known quantity.

### 2.3 Controls with known answers, and the base control

Three controls come from construction rather than estimation:

1. **`compounding_cost(1) = 0` exactly.** At k=1 there are no preceding
   uncorrected draws, so `teacher_forced(1)` and `open_loop(1)` are the same
   computation.
2. **The identity.** `one_step_cost(k) + compounding_cost(k)` must equal
   `open_loop(k) − floor(k)`. Computed in float64 over magnitudes of order 250,
   a few ULPs is ~1e-13, so the tolerance is **1e-9** — about four orders of
   margin. The measured residual is recorded per cell rather than asserted to be
   zero: M3l's `floor_bits` read 5.7e-14, not 0.0, and the legend that called it
   "exactly 0" is still an open follow-up.
3. **The index pin.** `teacher_forced` at k=0 degenerates to `open_loop`'s first
   step, so their **prior logits at k=0 must be bit-identical**. This is a
   deterministic check on the alignment that `evaluate_rollout` records as
   *"Verified empirically, not by argument"*: with an oracle model, using
   `start + context` rather than `start + context + 1` gives a perfect predictor
   a constant nonzero error at every horizon.

The **base control** guards the one confound that would silently decide the
reading. If the agent is largely stationary, the true one-step displacement is
near zero, `motion_margin` is at best zero, and the milestone would read
`COPIES` for a reason that has nothing to do with the objective. So: per cell,
the **median one-step displacement must exceed the median `floor` error** at the
decision horizon. If true motion is smaller than the readout's own error, no
method could detect motion prediction, and the cell is unreadable. The floor is
self-calibrating — a ratio of two measured quantities, not a magic constant.
Cells that fail are named and the run refuses (exit 46).

### 2.4 Companions that decide nothing

Reported per cell as raw levels, gating nothing:

- **`compounding_share(k) = compounding_cost(k) / (open_loop(k) − floor(k))`.**
  Interpretable where the denominator is large, and exactly 0 at k=1 by control
  1. It is **not** read at small k, where the denominator is the same vanishing
  quantity that produced `pixel_ae` seed 1's `+3.39 / −3.36`.
- **The band-relative figure**, expressing the costs as fractions of the
  persistence-to-floor band so they can be set beside `gap_closed`. Reported for
  continuity with the gate's own units; decides nothing, for the same reason.
- **`kl_rate_above_free_bits` and `kl_dyn_max`** per cell, carried onto the
  record from the training history, so the refutation in section 1 travels with
  the reading rather than living only in this spec.

## 3. Reading H

### 3.1 The statuses and their precedence

Evaluated in order. The first that applies is the reading.

1. **Refusals** — no status, a named `SystemExit`, nothing written:
   - the identity residual exceeding 1e-9, or the k=0 logits not bit-identical
     → **exit 45**;
   - the base control failing in any cell, a protocol disagreement across
     records, or an arm/seed plan narrowed such that no data could reach a
     positive verdict → **exit 46**. The last is the trap
     `require_readable_plan` was built for in M3j, where `--arms random_vit`
     printed a row reading `clears = up` beside a verdict of `NO DIFFERENCE`.
2. **`PREDICTS_MOTION`** — `motion_margin` whole interval above 0 at the
   decision horizon, in a strict majority of the seeds present within each of
   at least `ARMS_REQUIRED` arms. The one-step map predicts real motion even
   from late states; what fails is rolling it forward. **A multi-step or
   overshooting objective is the indicated intervention.**
3. **`COPIES`** — `motion_margin` whole interval at or below 0, on the same
   majority. One prior step from the *true* state is no better than assuming
   stillness. **A longer-horizon term cannot rescue this; the target must
   change.**
4. **`INDETERMINATE`** — the fall-through. The results must say plainly that it
   arrived by fall-through rather than by clearing a bar, in the same words M3k
   used for `INDISTINGUISHABLE`: by default rather than by evidence.

### 3.2 Where the bars come from

`DECISION_H = 45`, because that is the horizon the M3 gate reads. The full
`REPORTED_K = (1, 2, 3, 5, 8, 10, 15, 20, 30, 45)` is reported as raw levels so
a reader may apply a different horizon rather than inheriting this one.

`ARMS_REQUIRED = 2` of 3, matching every milestone from M3i onward.

The seed bar is **a strict majority of the seeds actually present**, computed
from the record set, never a stored constant. M3l's design was reworked for
exactly this: a fixed `SEEDS_REQUIRED = 2` is a majority at 3 seeds and a
minority at 5, and at 5 seeds a rule written as a constant can be cleared both
ways at once. A run with fewer than 3 seeds in an arm cannot establish that
arm and is refused by name, not silently tallied.

### 3.3 The asymmetry, stated before the numbers

Requiring the **whole interval** clear of 0 makes `PREDICTS_MOTION` harder to
reach. Requiring the whole interval at or below 0 makes `COPIES` harder to
reach. `INDETERMINATE` is the fall-through and establishes nothing.

**Both decisive statuses are levels against an exactly known baseline measured
on the same windows, so both directions are sound.** This is the structural
difference from M3k, whose statistic could speak in only one direction and whose
`INDISTINGUISHABLE` therefore licensed no positive claim, and from M3l, whose
`bits_carried` upper-bounds the joint and so was trustworthy only *below* a cut.
Here the true one-step displacement is ground truth, not an estimate, and
`motion_margin` is a paired difference against it. A reading in either direction
is evidence.

What neither status licenses: a claim about **why** the one-step map behaves as
it does. `PREDICTS_MOTION` says a multi-step objective is the indicated next
intervention; it does not say that one would pass the gate. `COPIES` says a
longer horizon cannot help; it does not name what target would.

## 4. Architecture

| file | change |
|---|---|
| `src/mbfps/eval/rollout.py` (255 lines) | **modify** — the `teacher_forced` arm and the one-step-persistence baseline |
| `src/mbfps/eval/burden.py` | **new** — `DECISION_H`, `REPORTED_K`, `ARMS_REQUIRED`, the decomposition, `motion_margin`, Reading H with its statuses and sentences, the identity check, the formatters |
| `scripts/prediction_burden.py` | **new** — measure and read phases, exits 45 and 46 |
| `tests/eval/test_burden.py` | **new** |
| `tests/eval/test_prediction_burden_script.py` | **new** |
| `tests/eval/test_rollout.py` | **modify** — the new arm and the index pin |
| `tests/eval/test_diagnose_dynamics_script.py` | **modify** — the exit registry gains `{45, 46}` |

### 4.1 Why the new arm lives in `rollout.py` and nowhere else

The floor already runs `observe` over the real future, so the posterior `h` and
`z` at every future step — exactly the states `teacher_forced` needs — are
already in hand; the new arm is one batched `_step` plus `prior_net`, not a new
rollout loop. More importantly, the truth-index convention lives there, and so
does the window rule in `window_starts`, which exists precisely so that
`gather_probe_data` and both diagnostics cut identical windows. Computing the
new arm anywhere else means re-deriving an off-by-one that was settled
empirically.

`src/mbfps/eval/diagnostics.py` is 1,948 lines. It is not touched by this
milestone, and consolidating it is out of scope.

There are already three episode-clustered bootstraps — `capacity.py`,
`probe.py` and `pooling.py`. M3m **reuses** one rather than adding a fourth, and
must resample the same episode labels over the same window set that
`evaluate_rollout` cuts. Consolidating the three is out of scope.

### 4.2 Two phases, one record per cell

`--phase measure` writes one JSON record per cell carrying the four curves, the
per-window costs, the intervals, the three controls' measured values, the base
control, the training-history companions, and the protocol fields.
`--phase read` pools the nine, refuses on protocol disagreement, and prints
Reading H. A reading is therefore reproducible from records without a GPU, as
in M3l, where `capacity.txt` came out byte-identical on three independent reads.

## 5. Constraints

- No training. This milestone reads the nine existing 20,000-step checkpoints in
  `runs/m3_study_v2`.
- Nothing under `runs/` is removed or overwritten. `runs/` is a symlink to
  storage shared by every worktree.
- The test command is `.venv/bin/python -m pytest`; there is no `pytest` entry
  point in the venv.
- Rollout sampling stays at the model's own temperature. M3h swept it and
  Reading N was `SHARPER WORSE` — all twelve (arm, τ) estimates negative, 0 of 3
  seeds clearing positively anywhere — and `RSSM._sample` records that taking
  the mode collapses the trajectory to 3 distinct latents of 45 and roughly
  quadruples position error. Sharpening is not a free way to make rollouts
  reproducible, and reducing rollout noise is already known to hurt.
- Every test must be watched failing under a named mutation and passing after
  revert. Roughly 30 tests that could not fail were caught across M3k and M3l.

## 6. Cost

One pass over nine cells, no training. The new arm adds one batched prior step
per horizon on top of an existing 45-step `imagine` and 45-step `observe`, so
**20–35 minutes**, bracketed by M3l's 18-minute measure and M3h's 33-minute
sweep. The read phase is seconds.

## 7. Exit criteria

M3m is complete when Reading H is read from nine cells at one `git_sha`, with:

- the identity residual recorded per cell and within 1e-9;
- the k=0 logit pin holding on all nine;
- the base control recorded per cell, with the cells that fail it named;
- `motion_margin` with an episode-clustered interval at every
  `REPORTED_K`, and the status read at `DECISION_H`;
- the companions of 2.4 reported as raw levels;
- the results section re-derivable from the records alone.

## 8. What this does and does not license

M3m **does not pass the M3 exit gate** and is not intended to. The gate remains
failed on `beats_persistence`, `band_is_usable` and `filtering_beats_embedding`,
and `gap_closed(45)` is negative in 9 of 9 position cells. What M3m produces is
the evidence that selects the next intervention, and a horizon curve showing
that the failing criterion is the tail of a curve that starts out positive.

It licenses no claim that any intervention will pass the gate. It does not
revisit the bottleneck lever, which M3l refuted. It does not measure in latent
space, so it cannot say whether position error is inherited from latent error or
introduced by the readout — that is a separate question and a separate spec.

## 9. Spec self-review

- **Placeholders:** none. `DECISION_H = 45`, `REPORTED_K` has ten entries,
  `ARMS_REQUIRED = 2`, the identity tolerance is 1e-9 with its ULP derivation,
  the exit codes are 45 and 46, and the base control is a measured ratio rather
  than a constant.
- **Internal consistency:** the asymmetry in 3.3 matches the statuses in 3.1;
  the companions in 2.4 are excluded from 3.1's tally; the architecture in 4
  lists exactly the files the exit criteria in 7 require.
- **Two corrections made while drafting.** First, a claim that the model
  "closes 38–73% of the band in most cells" at k=1: the measured spread is
  +0.08 to +0.73 with four of nine above +0.38, so "most" was wrong and the
  figures are now stated exactly. Second, section 2.2's baseline. The first
  version of this design compared `teacher_forced` to the recorded
  `persistence_position`, which is anchored at t and would have handed the new
  arm a k-step information advantage. Section 2.2 now states the rigged
  comparison explicitly so it is not reintroduced, and section 5 requires a test
  that fails if it is.
- **Ambiguity:** "a strict majority of the seeds present" is stated as computed
  from the record set rather than stored, and an arm with fewer than 3 seeds is
  refused rather than tallied.
- **Scope:** one measurement, one reading, no training. Latent-space curves,
  bootstrap consolidation and `diagnostics.py` are all explicitly out.
