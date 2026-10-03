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

`diagnostics.regrounding_sweep` already imagines k steps, re-observes the real
frames, and repeats across the horizon, for `REGROUNDING_KS = (1, 3, 5, 15, 45)`.
M3m reads that existing ladder rather than building an arm of its own.

| rung | re-grounds | what each scored latent is | role |
|---|---|---|---|
| `floor` | every step | a ZERO-step posterior, seeing the frame it is scored on | the known lower anchor |
| `k = 1` | every step, after the prediction | a ONE-step prior from a posterior-grounded state | **the one-step map, measured** |
| `k = 3, 5, 15` | every 3 / 5 / 15 steps | a prior 1..k steps past its last correction | the burden curve |
| `k = 45` | never | the open loop | **bitwise self-check** against the shipped `rssm_position` |

The floor is **not** the k→0 limit: it sees the frame it is scored on, while k=1
is a prior step from a state grounded one frame earlier. So k=1 must sit
STRICTLY above the floor, and `is_bitwise_the_floor(1)` reads `False` on all
nine shipped cells — the control already holds.

Define, per horizon step and per window:

```
burden(k)      = sweep.curve(k) − floor
compounding(k) = burden(k) − burden(1)
```

`burden(1)` is what one step of prediction costs. `compounding(k)` is what it
costs that correction arrives every k steps rather than every step, and
`compounding(1) = 0` **exactly**, by construction rather than by estimate.
`burden(45)` is the full open-loop cost the gate reads.

Every rung is a mean over the same windows with the same weights, and
`RegroundingSweep` retains the per-window rows (`window_position` per k, and
`window_floor_position`) so every comparison is **paired**. The class's own
comment records that the unpaired spread overstates the paired bars by 1.7× to
3.9× on the shipped cells — enough to hide the k=1 versus k=3 separation — which
is why the paired form is required and the unpaired one is not used.

### 2.2 `motion_margin` — and the baseline that makes it fair

**Notation, because two different axes were both called `k` in the first draft.**
`k` is the **re-grounding period** — how often observation corrects the rollout —
and ranges over `REGROUNDING_KS = (1, 3, 5, 15, 45)`. `h` is the **horizon
step**, 1..45, and is reported over `REPORTED_H`. So `burden(k, h)` is the cost
at horizon step `h` of correcting only every `k` steps, and the quantity Reading
H decides on is at `k = 1`, `h = DECISION_H`.

The natural-looking comparison is the `k = 1` rung against the recorded
`persistence_position`. **That comparison is rigged and must not be used.**
Recorded persistence copies the position at t, the last frame the *open-loop*
rollout saw, while the `k = 1` rung is re-grounded every step and has therefore
observed through h−1. It would win on information advantage rather than by
predicting anything.

The fair baseline is **one-step persistence**: predict that the agent did not
move between h−1 and h. Its error is exactly the true one-step displacement,
known from ground truth with no model involved.

```
motion_margin(h) = one_step_persistence(h) − sweep.curve(1)[h]
```

Positive means one prior step beats assuming stillness. This is the inherited
hypothesis — does the objective ever ask for motion — reduced to a level against
an exactly known quantity.

### 2.3 Controls with known answers, and the base control

Four controls, three of them already implemented and already passing on the nine
shipped cells:

1. **`compounding(1) = 0` exactly** — the same computation on both sides.
2. **The k=45 rung reproduces the record bitwise.** `open_loop_divergence`
   reads 0.0 on all nine shipped cells. `regrounding_sweep` refuses a `ks` that
   omits the horizon, because that pass *is* this self-check: "a sweep without
   it reports curves nothing has checked".
3. **k=1 is strictly above the floor.** `is_bitwise_the_floor(1)` reads `False`
   on all nine. Equality would mean the grounding consumed the frame it is
   scored on, and the whole ladder would be meaningless.
4. **The burden identity.** `burden(1) + compounding(k)` must equal `burden(k)`.
   Computed in float64 over magnitudes of order 250, a few ULPs is ~1e-13, so
   the tolerance is **1e-9** — about four orders of margin. The measured
   residual is recorded per cell rather than asserted to be zero: M3l's
   `floor_bits` read 5.7e-14, not 0.0, and the legend that called it "exactly 0"
   is still an open follow-up.

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

- **The burden curve over the ladder** — `burden(k, DECISION_H)` for
  k = 1, 3, 5, 15, 45, which shows how fast the cost grows as correction becomes
  rarer, and whether it grows smoothly or jumps.
- **`compounding_share(k) = compounding(k, DECISION_H) / burden(k, DECISION_H)`.** Interpretable where
  the denominator is large, and exactly 0 at k=1 by control 1. It is **not**
  read at small horizon, where the denominator is the same vanishing quantity
  that produced `pixel_ae` seed 1's `+3.39 / −3.36`.
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
   - the burden identity residual exceeding 1e-9;
     `open_loop_divergence` not 0.0, so the k=45 rung does not reproduce the
     record bitwise; or `is_bitwise_the_floor(1)` returning `True`, so the
     grounding consumed the frame it is scored on → **exit 45**;
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
`REPORTED_H = (1, 2, 3, 5, 8, 10, 15, 20, 30, 45)` is reported as raw levels so
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

> **Errata, added after Task 7. The paragraph above is left as it was
> designed, because this spec is the record of what was designed and M3n
> inherits it.** Its bolded sentence and "A reading in either direction is
> evidence" are **not true unconditionally.** `motion_margin` subtracts a
> probe-space quantity (the k=1 rung) from a ground-truth one that pays no
> readout error, so the margin carries the readout error with it. Both
> directions are sound **only in a cell the base control of 2.3 admits** — where
> the median true one-step displacement exceeds the median floor error. Outside
> it a negative margin *is* the readout error, and a perfect one-step predictor
> would read negative too. On the nine shipped cells the displacement is 3.9695
> against a floor of 88.29 to 224.36 (22.2× to 56.5×), the base control refused
> all nine, and the margins sat within 0.1% to 5.8% of what a perfect predictor
> would have read. Reading H returned `UNREADABLE` and the question was answered
> by the ladder instead. See `## Task 7 results` in the plan, which also names the
> probe-space baseline that would make the margin readable.

What neither status licenses: a claim about **why** the one-step map behaves as
it does. `PREDICTS_MOTION` says a multi-step objective is the indicated next
intervention; it does not say that one would pass the gate. `COPIES` says a
longer horizon cannot help; it does not name what target would.

## 4. Architecture

| file | change |
|---|---|
| `src/mbfps/eval/burden.py` | **new** — `DECISION_H`, `REPORTED_H`, `ARMS_REQUIRED`, `burden`, `compounding`, `motion_margin`, the identity check, Reading H with its statuses and sentences, the formatters |
| `scripts/prediction_burden.py` | **new** — measure and read phases, exits 45 and 46 |
| `tests/eval/test_burden.py` | **new** |
| `tests/eval/test_prediction_burden_script.py` | **new** |
| `tests/eval/test_diagnose_dynamics_script.py` | **modify** — the exit registry gains `{45, 46}` |

**`src/mbfps/eval/rollout.py` and `src/mbfps/eval/diagnostics.py` are NOT
modified.** The first version of this spec put a new `teacher_forced` arm in
`rollout.py`. That was wrong three times over, and the correction is the main
thing this section records:

- The arm already exists. `regrounding_sweep`'s k=1 rung *is* a one-step prior
  from a posterior-grounded state, and it is implemented, tested, wired into
  `scripts/diagnose_dynamics.py --ks`, and already run on the nine cells.
- `rollout.py` averages over windows, so the per-window rows a clustered
  bootstrap needs do not survive there. `RegroundingSweep` retains them.
- `rollout.py` has no RNG-snapshot machinery. `_diagnose`'s header states that
  arms drawing from different stream points "differ by sampling noise, and the
  measured effect is uninterpretable", and that the snapshot must be
  device-aware because on MPS a CPU-only snapshot "inflates the apparent action
  effect by ~100x … while every CPU test still passes". A hand-built arm in
  `rollout.py` would manufacture exactly that artefact.

So M3m **consumes** `regrounding_sweep` and `evaluate_rollout` and adds only
what does not exist: the one-step-persistence baseline, `motion_margin`, the
burden decomposition, Reading H, and the two-phase script.

`diagnostics.py` is 1,948 lines and `rollout.py` duplicates its rollout body on
purpose — the header explains that unifying them would make the k=horizon
degeneracy true by construction and turn the bitwise self-check into
decoration. Neither is refactored here. There are also three
episode-clustered bootstraps already (`capacity.py`, `probe.py`, `pooling.py`);
M3m reuses one rather than adding a fourth, resampling the episode labels the
record already carries under `windows.episode`. Consolidating the three is out
of scope.

### 4.1 Why the nine shipped records are not enough

The sweep ran on all nine cells with `ks = [1, 3, 5, 15, 45]`, and
`self_checks.smallest_k = 1` with `smallest_k_is_bitwise_the_floor = False` is
recorded. But the keys carrying the sweep's **numbers** — `k` and
`floor_margin` — were added to `scripts/diagnose_dynamics.py` after those
records were written, and are absent from all nine. The curves exist in the code
path and not on disk, so M3m must re-run the sweep. It is an existing, tested
code path, not new behaviour.

### 4.2 Two phases, one record per cell

`--phase measure` calls `evaluate_rollout` and `regrounding_sweep` per cell and
writes one JSON record carrying: every rung's curve, the per-window rows, the
one-step-persistence baseline, `motion_margin` with its episode-clustered
interval at every `REPORTED_H`, the four controls' measured values, the base
control, the training-history companions, and the protocol fields.
`--phase read` pools the nine, refuses on protocol disagreement, and prints
Reading H. A reading is therefore reproducible from records without a GPU, as in
M3l, where `capacity.txt` came out byte-identical on three independent reads.

## 5. Constraints

- No training. This milestone reads the nine existing 20,000-step checkpoints in
  `runs/m3_study_v2`.
- `src/mbfps/eval/rollout.py` and `src/mbfps/eval/diagnostics.py` are not
  modified. M3m consumes `evaluate_rollout` and `regrounding_sweep`; a
  re-implementation of either arm is a defect, not an alternative (section 4).
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

One pass over nine cells, no training. Per cell: one `evaluate_rollout` and one
`regrounding_sweep` over five rungs. The action-intervention ladder and the
interventions that make `diagnose_dynamics` expensive are **not** run. M3h's
full nine-cell sweep took 33m25s, so expect **25–45 minutes**. The read phase is
seconds.

## 7. Exit criteria

M3m is complete when Reading H is read from nine cells at one `git_sha`, with:

- the burden identity residual recorded per cell and within 1e-9;
- `open_loop_divergence` 0.0 and `is_bitwise_the_floor(1)` `False` on all nine,
  both recorded;
- `burden(k)` and `compounding(k)` recorded for every rung of
  `REGROUNDING_KS`, with `compounding(1)` exactly 0;
- the base control recorded per cell, with the cells that fail it named;
- `motion_margin` with an episode-clustered interval at every
  `REPORTED_H`, and the status read at `DECISION_H`;
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

- **Placeholders:** none. `DECISION_H = 45`, `REPORTED_H` has ten entries,
  `ARMS_REQUIRED = 2`, the identity tolerance is 1e-9 with its ULP derivation,
  the exit codes are 45 and 46, and the base control is a measured ratio rather
  than a constant.
- **Internal consistency:** the asymmetry in 3.3 matches the statuses in 3.1;
  the companions in 2.4 are excluded from 3.1's tally; the architecture in 4
  lists exactly the files the exit criteria in 7 require.
- **Three corrections made while drafting.** The third was found only by reading
  the code the plan would have to call, and is the largest: the first version of
  section 4 specified a new `teacher_forced` arm in `rollout.py`. That arm
  already exists as `regrounding_sweep`'s k=1 rung — tested, wired, and already
  run on the nine cells — and `rollout.py` is additionally the wrong home,
  because it averages over windows and has no device-aware RNG snapshot. The
  spec review that preceded this checked the spec against itself and could not
  catch it. Second, a claim that the model
  "closes 38–73% of the band in most cells" at k=1: the measured spread is
  +0.08 to +0.73 with four of nine above +0.38, so "most" was wrong and the
  figures are now stated exactly. Second, section 2.2's baseline. The first
  version of this design compared `teacher_forced` to the recorded
  `persistence_position`, which is anchored at t and would have handed the new
  arm a k-step information advantage. Section 2.2 now states the rigged
  comparison explicitly so it is not reintroduced, and section 5 requires a test
  that fails if it is. First, the "38–73% in most cells" claim above.
- **Ambiguity:** "a strict majority of the seeds present" is stated as computed
  from the record set rather than stored, and an arm with fewer than 3 seeds is
  refused rather than tallied.
- **Scope:** one measurement, one reading, no training. Latent-space curves,
  bootstrap consolidation and `diagnostics.py` are all explicitly out.
