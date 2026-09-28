# MB-FPS M3j — Where is observed motion lost? (Design)

**Status:** design approved 2026-09-27. Evaluation only; no training, no checkpoint altered.

---

## 1. Why

M3i read `NO_MOTION`: at the decision horizon no seed of any arm beats predicting no displacement
at all. Its §3.3 pre-registered the consequence, and the consequence has been taken — all three of
M3h §8's prior-side levers are retired and the project is pointed upstream.

But "upstream" names two very different levers, and M3i cannot tell them apart:

- **The encoder.** If the frozen 2048-d features do not resolve motion, nothing the RSSM does can
  recover it, and the fix is the encoder's input — frame stacking, resolution, stride.
- **The objective.** If the features resolve motion and the RSSM discards it, the fix is the loss.
  `world_model.py:81` is `loss = embedding_loss + reward_loss + continue_loss + kl`, where
  `embedding_loss` is `F.mse_loss(predictions["embedding"], embeddings.detach())` — the model's
  reconstruction target is **the single frame it just saw**. Nothing in that objective ever asks the
  latent to represent displacement.

Picking wrong costs a nine-cell retrain. M3c's recorded `seconds` are 88–95 minutes per cell
(`runs/m3_study_v2/result_frozen_ssl_seed{0,1,2}.json`), so the wrong lever is ~13.5 hours.

### 1.1 What M3i actually established, and where it is thin

M3i's §9 positive controls are the sharpest thing in the milestone and the least well supported.
From the same 132 fit rows the instrument recovered absolute position at r² **+0.291 / +0.165**
while recovering backward displacement at r² **−0.010 / −0.021**. Those four numbers:

- were measured during the final whole-branch review, **not** by `scripts/latent_motion.py`, and are
  in no record — M3i's own §9 provenance note says a successor should build them into the measure
  phase;
- rest on **132 rows against 1536 features** (n/p = 0.086), one row per window at t0;
- have **no control for what the current frame already provides.** In a corridor your location
  constrains your motion. A probe on the current frame alone can predict backward displacement from
  position-conditional motion statistics, with no filtering involved at all. Any r² credited to the
  latent that the frame alone also achieves is not evidence the model retains anything.

M3j promotes those four numbers to a measured, recorded, controlled reading, and uses it to choose
the lever.

### 1.2 Why displacement is the right target for this statistic

`probe.filtering_gain` already asks "does the latent add anything over the current frame", and its
docstring records the measured answer on the privileged state at 20k: gain **−0.0208**. `h` buys
nothing about absolute position — but it could not have, because `e_t` is very nearly sufficient for
position. A single frame says where you are.

A single frame **cannot** say where you came from. So backward displacement is a target on which the
current frame is insufficient by construction, and a nonzero gain is the only way the model can
demonstrate that it filters at all. That is the measurement M3j makes.

### 1.3 Method note, carried forward

Every number in §1 is quoted from a committed artefact or from source, with its definition read out
rather than recalled. The magnitudes in §2.3 were measured on the 122-episode dataset during design
and are labelled as design sizing, not as findings. M3i's §1 inherited M3h's habit of quoting a
motivating statistic that was never measured as stated; §1.1 of M3i's spec records why.

---

## 2. What it does

Evaluation only, on the nine shipped M3c checkpoints (`runs/m3_study_v2`, `record_git_sha`
`ca3e140`), at the protocol every milestone since M3d has used: **context 5, horizon 45, 229 windows
over 24 validation episodes**, device `mps`.

### 2.1 The statistic — an increment over the current frame

```
gain = R2([enc(t) (+) B] -> y)  -  R2([enc(t)] -> y)
```

`enc(t)` is the raw encoder embedding, `B` a second feature block, `y` a target. Both arms are handed
the same `enc(t)`, so two things cancel: the 32x32 categorical bottleneck (as
`filtering_gain` already argues) and **the position-constrains-motion confound**, which is new here
and is the reason §1.1's levels could not have answered this question.

`gain` is computed on **three disjoint splits**, exactly as `filtering_gain` does and for the reason
its docstring records: weights from the fit windows, ridge selected on **further** training episodes
held out from those, R² and interval reported on validation windows that neither the weights nor the
selection ever saw. A gain is a difference of levels and the two feature sets have different widths,
so a ridge maximum taken on the scored rows favours the wider one and manufactures a positive gain
out of the selection alone. `filtering_gain`'s measured example has **the sign flipping on a
one-step selection error**, which is why `select_episodes` stays at 20.

**Every rung at one (target, k) is handed a byte-identical base array.** That is what makes the four
gains differences against the *same* base level, so they are comparable to each other and not merely
each to its own fit. Four independent row selections would break the property while still producing
four plausible gains, so it is pinned by a test on the arrays rather than left to inspection. The
base probe is refit inside each rung's call and that costs accuracy nothing, because the fit is
deterministic on identical inputs; sharing the fitted probe instead would save about 11% of solve
time (a 2048-wide base solve is roughly an eighth of a 4096-wide joint one) and is not worth the
extra interface.

### 2.2 The ladder of second blocks

`RSSM.observe` advances `h` with `actions[:, i]` FIRST and forms the posterior from that
already-advanced `h`, so `h(t)` sees embeddings `0..t-1` and actions `0..t` but **not** `enc(t)`,
while `z(t)` sees `h(t)` and `enc(t)`. `_pack` already returns `h` and `z` separately.

| rung | `B` | width | asks |
|---|---|---|---|
| `two_frame` | `enc(t-k)` | 2048 | what two real frames linearly provide |
| `deterministic` | `h(t)` | 512 | what the recurrent state adds |
| `stochastic` | `z(t)` | 1024 | what the bottlenecked latent adds |
| `full` | `h(t) (+) z(t)` | 1536 | M3i's input — the bridge to `NO_MOTION` |

`enc(t) (+) h(t)` is therefore exactly "this frame plus everything strictly before it", which is
what backward displacement requires and what `enc(t)` alone provably lacks.

**`two_frame`'s block is `enc(t-k)`, not `enc(t-1)`, and the distinction is not cosmetic.** The two
frames that determine `p(t) - p(t-k)` are `t` and `t-k`. Pairing `enc(t)` with `enc(t-1)` would
hand the reference a frame one step back and ask it about a 15-step displacement, making the rung
near-uninformative at every k > 1 and the phrase "what two real frames linearly provide" false. It
would also break the comparison the design rests on: `h(t)` sees every frame `0..t-1`, `t-k`
included, so the reference must see `t-k` as well or the latent rungs gain an advantage they did not
earn. At k = 1 the two definitions coincide, which is what made the error invisible in the first
draft of this section.

**`two_frame` is a reference, not a ceiling, and must not be described as one.** `h` integrates the
**action** sequence, which two frames do not contain, so a latent rung can legitimately exceed it.
Calling it an upper bound would be the same class of overclaim as M3i's "the machinery the M3 gate
uses", corrected post-run in that spec's §4.

### 2.3 The targets

All **backward**, so the model is scored only on motion it has already observed — the
representation's best case, which is what makes a null decisive. Measured on the 122-episode dataset
(design sizing; position std is x 254.6 / y 189.0 over a map extent of ~930 x 800 units):

| k | mean ‖Δpos‖ | median | frac exactly 0 | mean abs Δangle | frac exactly 0 |
|---|---|---|---|---|---|
| 1 | 5.77 | 4.74 | 0.086 | 3.45° | **0.629** |
| 4 | 20.52 | 16.29 | 0.022 | 11.15° | 0.193 |
| 15 | 57.49 | 46.31 | 0.002 | 32.72° | 0.039 |

- **Decision target — translation.** The `(pos_x, pos_y)` components of `p(t) - p(t-k)`.
  `K_REPORTED = (1, 4, 15)`, `DECISION_K = 4`. **k = 1 is degenerate on both targets** — translation
  is 0.6% of the map extent against a recorded probe position error of 222–247 units
  (`gather_probe_data`'s docstring), and rotation has median 0.00° and is exactly zero 62.9% of the
  time. k = 4 and k = 15 are both non-degenerate; **k = 4 is chosen as the shorter of the two**,
  because it demands less integration from `h`, so a null there is the stronger null. That choice
  gates nothing on the positive side: **a cleared gain at ANY reported k counts as motion
  retained.** The lever question is binary — does the RSSM retain observed motion at all — so a
  disjunction is strictly more generous to the model, and a null across all three horizons is a
  stronger null than a null at any one of them.
- **Motion control (positive) — rotation.** The wrapped `Δangle` over the same k, as `(sin, cos)` of
  the change so that 359° and 1° are near each other — the representation `probe.probe_targets`
  already uses for absolute angle, for the same reason. Rotation is the natural positive
  control for the motion question: a 20-unit translation is ~2% of the map through a 112x112 frozen
  backbone, while a 32° turn rewrites the frame. **Reading rotation but not translation means
  spatial resolution; reading neither means the RSSM discards motion as such.** Without this
  control a translation null is confounded with "the encoder cannot see a 20-unit move", and the
  milestone would repeat M3i's weakness.
- **Base control (positive) — absolute position.** `enc(t)` -> the privileged state, i.e.
  `filtering_gain`'s own base arm. If the current frame cannot linearly report where it is, the
  instrument is broken and nothing is read.

---

## 3. How it decides

### 3.1 Reading E

**Reading E reports the last rung on the path at which observed motion is still linearly present.**
The statuses are names for rungs, so the reading and the lever choice are one act.

**Clearing is one-sided: a rung clears when `ci_low > 0`.** A negative gain means appending a block
made held-out R² worse — noise or selection slack, not a finding. M3i made its control two-sided on
an argument its own results refuted (`## Task 8 results` §5: against a randomly-paired displacement
any nonzero prediction can only increase the error, so the negative sign is correct behaviour). This
applies that lesson rather than re-deriving it.

**Agreement carries the multiple-comparison burden, not a Bonferroni.** `SEEDS_REQUIRED = 2` and
`ARMS_REQUIRED = 2`, as in M3h and M3i: a rung clears at k only if it clears in >= 2 of 3 seeds in
>= 2 of 3 arms. A Bonferroni over 4 rungs x 3 horizons would need the 0.2nd bootstrap percentile,
which 1000 resamples cannot resolve; the 2-of-3 x 2-of-3 conjunction is the project's existing
instrument and is resolvable at `confidence = 0.95`, `resamples = 1000`.

**The interval clusters on episodes, not windows.** `_block_bootstrap_ci` today blocks positionally
into 229 window-sized blocks. 229 non-overlapping chunks cut from 24 trajectories are not
independent, and every reading from M3e onward clusters on the 24 episodes. M3j's reading resamples
**episodes**. `filtering_gain`'s recorded behaviour is not changed; see §5.

### 3.2 Statuses

In precedence order. Exhaustive and mutually exclusive.

| status | rule |
|---|---|
| `UNRESOLVED_BASE` | the base control fails: `enc(t)` -> absolute position does not read in the required seeds and arms. The instrument is broken; **nothing is read.** Exit **39**. |
| `MOTION_RETAINED` | `stochastic` or `full` clears on translation at any reported k |
| `BOTTLENECK_LOSS` | `deterministic` clears on translation; neither `stochastic` nor `full` does |
| `MOTION_DISCARDED` | no latent rung clears on translation; `two_frame` does |
| `TRANSLATION_UNRESOLVED` | **no** rung clears on translation at any k, but **some** rung clears on rotation |
| `UNRESOLVED_MOTION` | no rung clears on **either** target at any k. The measurement detects no motion anywhere; **nothing is read.** Exit **40**. |

The latent rungs are checked **before** `two_frame` because `h` integrates actions, which two frames
do not contain, so a latent rung can clear where `two_frame` does not (§2.2).

**`UNRESOLVED_MOTION` is last, not second, and that ordering is load-bearing.** An earlier draft of
this spec refused whenever `two_frame` was silent on both targets — which, by the action argument in
§2.2, would have refused precisely the case where a latent rung clears translation on information
two frames do not contain. `UNRESOLVED_MOTION` therefore requires the *whole ladder* to be silent on
*both* targets, not `two_frame` alone.

The one true control failure is `UNRESOLVED_BASE`, and it stays first, so a failed control can never
be overwritten by a result — the ordering rule M3i established. `UNRESOLVED_MOTION` is not a control
failure but an empty measurement, which is why it sorts with the readings rather than ahead of
them.

A non-finite gain or interval bound is a **refusal**, not a non-clear. M3i's ledger left this as a
note for its successor: `clears_up`, `clears_down` and `leaks` all evaluate `False` on NaN, so one
NaN window would read "no clear" and "no leak" at once. It is built in here rather than noted again.

### 3.3 What each status commits the project to, pre-registered

| status | what the project does next |
|---|---|
| `MOTION_RETAINED` | **M3i is partially overturned.** The information is in the latent; the failure is the prior's ability to propagate it, not the representation's content. M3h §8's retirement is reversed and the prior-side levers return. The next milestone asks why the rollout does not exploit what the filter retains. |
| `BOTTLENECK_LOSS` | **Lever: the bottleneck.** `h` carries it and `z` destroys it. `z_cats` x `z_classes` is 32x32, at most ~160 bits, and `rep_scale = 0.1` is five times below `dyn_scale`. One training run, on the bottleneck. |
| `MOTION_DISCARDED` | **Lever: the objective.** The information is available and the RSSM discards it, because the loss never asks for motion — the embedding target is the single frame just seen (§1). One training run, adding a motion term. |
| `TRANSLATION_UNRESOLVED` | **Lever: the encoder's input.** 112x112 frozen single-frame features do not resolve a 20-unit translation. Frame stacking, resolution, or stride. Not a loss change. |
| `UNRESOLVED_BASE` or `UNRESOLVED_MOTION` | **No lever is chosen.** The next milestone is about the instrument, not the model. |

This is written down now so the consequence cannot be argued after the numbers are seen.

### 3.4 Conditioning, stated rather than claimed

24 validation episodes give 229 windows of 50 rows = 11,450 scored rows. Backward displacement at k
drops the first k rows of every window: 11,221 at k = 1, 10,534 at k = 4, 8,015 at k = 15. The
widest rung is `enc(t) (+) enc(t-1)` at 4096 features.

So **n/p is 2.0–2.7 at the widest rung** and 3.1–4.4 at `deterministic` (2.6 and 4.1 at
`DECISION_K`). That is *not* n >> p, and the
spec says so rather than implying otherwise. Against M3i's 132 rows on 1536 features (n/p = 0.086)
it is a 23–32x improvement, and the ridge is selected on a held-out split rather than on the scored
rows. The row counts are written into every record so the claim is checkable.

### 3.5 Companions — change no verdict

Printed beside the reading and carried in the records: every rung's `joint_r2` and `embedding_r2`
level, the ridge selected for each arm, the row count at each k, and the two control readings. The
levels are what bridge to M3i's recorded numbers and to the `latent_selection_r2` 0.18–0.34 the
earlier milestones quote. None of them carries a threshold.

---

## 4. What it does not claim

- A linear probe is a **lower bound** on the information present. A null says no *linear* read-out of
  that block beats the current frame; not that the information is absent under every decoder.
- `two_frame` is **not an upper bound** on the latent rungs — it lacks the action channel (§2.2).
- **Backward is not forward.** Retaining motion already observed is necessary, not sufficient, for
  predicting future motion. `MOTION_RETAINED` does not say the M3 gate can pass, and no status here
  is a statement about `gap_closed`.
- The interval holds the probes fixed across resamples. It is an interval on the scored sample — how
  much the gain would move on a different draw of evaluation windows — not on the whole
  fit/select/score pipeline. That is `_block_bootstrap_ci`'s own documented caveat, inherited and
  restated rather than quietly dropped.
- It does not change the M3 gate, `aggregate.py`, `report_study.py`, `filtering_gain`'s recorded
  numbers, any shipped checkpoint, or any verdict M3b–M3i recorded.
- It does not rank arms. Every gain is within a cell, against that cell's own base on the same rows.
- A reading on `my_way_home`'s 24 validation episodes at context 5 / horizon 45, at 20,000 steps, is
  not a statement that the world model works.

---

## 5. Shape of the code

**`src/mbfps/eval/retention.py`** — new. The pre-registered constants: `RETENTION_FAMILY = 3`,
`SEEDS_REQUIRED = 2`, `ARMS_REQUIRED = 2`, `K_REPORTED = (1, 4, 15)`, `DECISION_K = 4`, and `RUNGS`
as named second-block selectors. Target builders `backward_translation(positions, k)` and
`backward_rotation(angles, k)`. `RetentionArm`, `RetentionInputs`, `RetentionStatus`,
`reading_retention`, `READING_COLUMNS` / `READING_WIDTHS` / `format_reading_retention`. Holds the
finiteness guard of §3.2. Pinned by test: the six statuses and their precedence; that a failed
control suppresses any reading; the seed and arm tallies; that clearing is one-sided; and that the
disjunction over `K_REPORTED` is a disjunction.

**`src/mbfps/eval/probe.py`** — three additive changes. `filtering_gain`'s output stays
byte-identical and a test pins it, because it is on the gate's reporting path via `study.py:589` and
`scripts/eval_rollout.py:201`.

1. `gather_probe_data` returns a seventh row-aligned array, `"episode"` — the episode index each row
   came from, needed to cluster on episodes (§3.1). Added the way M3i added `"window"` and `"step"`.
2. `_block_bootstrap_ci` gains an optional `groups`. With `groups=None` the positional path is
   untouched; with labels it resamples by group. **A test pins that the label path reproduces the
   positional path bit-for-bit on unfiltered data** — which is what makes row filtering free, since
   backward displacement at k drops rows and the positional path raises on a row count that is not a
   whole number of windows.
3. A new general `gain_from_blocks(fit, select, score, *, block, target, groups, ...)`.
   `_gain_from_splits` becomes a thin wrapper that calls it with the `latent[:, :h_dim]` block and
   the privileged target, so there is no duplicated split, selection or bootstrap logic and no
   behaviour change on the existing path.

**`src/mbfps/eval/motion.py`** — the §3.2 finiteness guard added to `clears_up`, `clears_down` and
`leaks`, with a test that it fires only on non-finite values. A refusal on non-finite input cannot
change a result that had no non-finite input, and M3i's 90 series were verified all-finite, so no
M3i verdict moves and no re-run is needed.

**`scripts/latent_retention.py`** — new. Phases `measure | read | all`, mirroring
`scripts/latent_motion.py`. Cells load through `trust_horizon.prepare_cell`, inheriting the reviewed
loader, the protocol checks (12, 14) and M3h §2.4's reproduction bound. One record per cell,
`retention_<arm>_seed<n>.json`, carrying every (rung, k, target) gain with its interval, the ridges
selected, §3.5's companions, the two positive controls and the row counts. `read` writes
`retention.txt`, byte-identical to what it prints.

**Exit codes.** 0 / 11 / 12 / 14 / 30 inherited from `trust_horizon`; **39 `EXIT_BASE_UNRESOLVED`**
and **40 `EXIT_MOTION_UNRESOLVED`** new. M3i holds 38, so there is no clash. The exit-status registry
test lives in `tests/eval/test_diagnose_dynamics_script.py` and gains `latent_retention`.

**Tests.** `tests/eval/test_retention.py` and `tests/eval/test_latent_retention_script.py`, in the
established idiom — hand-typed expected values, one rule mutated per test, and a test pinning the
printed table's captions against the columns they caption, which is the defect class that has now
shipped three times in this project.

**Six end-to-end tests, one per status.** M3i shipped a `MOTION_ENCODED` path that nothing exercised
until the final whole-branch review found it; every branch here is reachable by a fixture that drives
the pipeline to that printed verdict.

---

## 6. The run

- **Smoke:** one cell (`pixel_ae`/s0) into `runs/m3j_smoke/`, a directory the real run never reads.
  Its `retention.txt` is read for format **and its wall clock is measured**, so the measure phase is
  sized from a timed cell rather than from an estimate in this spec.
- **Measure:** nine cells. Three gathers per cell (20 fit / 20 select / 24 score episodes, from a
  training pool of ~98) instead of M3i's one, then 4 rungs x 3 horizons x 2 targets of gain per
  cell with the shared bases of §2.1. Under `caffeinate -dimsu` and `nohup`, with `.head` /
  `.started` / `.finished` / `.exit` beside the log.
- **Acceptance:** nine records at one `git_sha` equal to HEAD, all on `mps`; `self_check.ok` on 9/9
  within M3h's reproduction bound; 229 scored windows over 24 episode clusters on every cell; row
  counts equal to §3.4's 11,221 / 10,534 / 8,015 at k = 1 / 4 / 15; every gain and interval bound
  finite; `retention.txt` byte-identical to a second `--phase read`.
- **Before any full-suite run:** check `du -sh $TMPDIR/pytest-of-$USER`. `tmp_path_retention_policy
  = "failed"` landed in PR #8 and held M3i's suite to 1.0 GB, but the check is cheap.

---

## 7. Files

| file | change |
|---|---|
| `src/mbfps/eval/retention.py` | new — statistics and Reading E |
| `scripts/latent_retention.py` | new — the run, phases, records, `retention.txt` |
| `src/mbfps/eval/probe.py` | `episode` label; `groups` on the bootstrap; `gain_from_blocks` |
| `src/mbfps/eval/motion.py` | the finiteness guard |
| `tests/eval/test_retention.py` | new |
| `tests/eval/test_latent_retention_script.py` | new |
| `tests/eval/test_probe.py` | the byte-identical pin, the groups-equivalence pin, the episode label |
| `tests/eval/test_motion.py` | the finiteness guard fires only on non-finite values |
| `tests/eval/test_diagnose_dynamics_script.py` | add `latent_retention` (39, 40) to the exit-status registry |
| `docs/superpowers/plans/2026-09-27-mb-fps-m3j-motion-retention.md` | the implementation plan |

Nothing else is modified. No checkpoint is altered, nothing under `runs/` is removed, and no
M3b–M3i verdict or record is touched.
