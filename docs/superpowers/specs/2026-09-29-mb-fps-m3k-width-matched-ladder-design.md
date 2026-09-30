# MB-FPS M3k — Is the past frame's advantage information, or feature count? (Design)

**Status:** design approved 2026-09-29. Evaluation only; no training, no checkpoint altered.

---

## 1. Why

M3j left the project with **two candidate levers and one confounded comparison**, and the next
milestone spends ~13.5 hours on whichever it picks (M3c's recorded `seconds` are 88–95 minutes per
cell, nine cells).

- **The bottleneck lever** rests on `z` carrying +0.00091 of translation at k = 4 against `h`'s
  +0.01327, adding at most +0.00064 over `h`, and **subtracting** 0.005–0.008 on rotation.
- **The objective lever** rests on `two_frame` (+0.03489) beating `deterministic` (+0.02071) at
  k = 15, in 8 of 9 cells — two raw frames beating the recurrent state, which has those frames,
  every frame between them, *and* the action sequence.

M3j flagged the second as confounded: `two_frame`'s block is **2048** features against
`deterministic`'s **512**. It could not say whether that mattered. **Measured during this design, on
two cells, it does.**

### 1.1 The confound is real, and it is the size of the effect

A random lift of `h` from 512 to 2048 columns — a fixed Gaussian matrix, so **zero information
added and the rank still 512** — buys:

| cell | `h` as shipped (512) | `h` lifted to 2048 | inflation |
|---|---|---|---|
| `pixel_ae`/s0 | +0.03411 | +0.04766 | **+0.01355** |
| `random_vit`/s0 | +0.01704 | +0.03081 | **+0.01377** |

Consistent across two cells, and **larger than the +0.01418 pooled `two_frame` − `deterministic`
gap the objective argument depends on.** Feature count alone buys about as much gain as the effect
M3j attributed to information.

### 1.2 And the advantage may survive matching anyway

On `random_vit`/s0, the largest-gap cell, at matched rank **and** count (512 against 512):

```
  h  512  (shipped)                               +0.01704
  enc(t-k) -> 512  (rank + count matched to h)    +0.03902
```

Still more than double. On `pixel_ae`/s0 — the one cell of nine where the shipped ordering is
already reversed — the matched comparison reverses too (+0.02487 against +0.03411). **Two cells
disagree, which is exactly why this needs all nine.**

### 1.3 The confound cuts the other way on M3j's other conclusion, and strengthens it

Every rung in M3j's ladder had a different block width: `two_frame` 2048, `stochastic` 1024, `full`
1536, `deterministic` 512. If ~1,500 extra columns buy ~+0.0137, then `stochastic` competed against
`deterministic` **with twice the columns — a handicap in `z`'s favour** — and still lost by a factor
of 15 (+0.00091 against +0.01327 at k = 4).

So the bottleneck finding was measured *through* a bias that flattered it, and the real gap is
wider than M3j reported. The confound M3j flagged as a threat to one conclusion turns out to have
been protecting the other. **A width-matched ladder therefore re-reads every cross-rung comparison
in M3j, not only `two_frame` against `deterministic`.**

### 1.4 Method note

Every number in §1 is measured, not recalled: §1.1 and §1.2 from a scratch pass run during this
design on the shipped code path and reported here as design evidence rather than as a finding;
§1.3's ladder figures from `runs/m3j_retention`'s nine records. M3h shipped a motivating premise
that was never measured as stated, and M3i's spec §1.1 records why that is the failure this note
exists to prevent.

---

## 2. What it does

Evaluation only, on the nine shipped M3c checkpoints (`runs/m3_study_v2`, `record_git_sha`
`ca3e140`), at the protocol every milestone since M3d has used: context 5, horizon 45, **229 windows
over 24 validation episodes**, device `mps`.

### 2.1 Three passes of one ladder

| pass | every rung's block | matches | role |
|---|---|---|---|
| `shipped` | native width | nothing | reproduces M3j; the baseline |
| `down` | **512** | count **and** rank | **decides** |
| `up` | **2048** | count only | **calibrates** the width bias |

**One gather per cell serves all three.** Only the probe fits differ, so this is not three times
M3j's cost.

**The base `enc(t)` is untouched in every pass.** Matching applies to the rung's *block* only, so
every gain stays on M3j's scale and remains directly comparable to its records, and the base R^2 is
identical across passes by construction.

### 2.2 The projections

One fixed Gaussian matrix per `(native width -> target width)`, drawn once from `PROJECTION_SEED`
and **shared across all nine cells**, so no cell gets its own draw and no arm can be favoured by a
lucky matrix. Scaled `1 / sqrt(target)`, the standard Johnson–Lindenstrauss form.

`projection(native, target)` returns **the identity when `native == target`**. That is what makes
§2.3's anchors exact rather than approximate, and it is the single most important line in the
module.

### 2.3 Two known-answer anchors, free

`deterministic` is already 512 wide, so the `down` pass must reproduce its shipped gain **exactly**.
`two_frame` is already 2048 wide, so the `up` pass must reproduce its shipped gain **exactly**.

Either failing means the projection plumbing is wrong and **that pass is unreadable** (§3.2). The
anchors cost nothing — they are rungs the pass computes anyway — and they are the known-answer
discipline M3g established.

### 2.4 What `down` matches that `up` does not

After projection to 512 every block has rank 512 and count 512 — **fully matched**. Under the lift a
block keeps its native rank (`h` lifted to 2048 columns is still rank 512), so `up` isolates count
from content but leaves `enc(t-k)`'s richer rank intact.

That asymmetry is the reason the roles are not interchangeable: `down` can decide who wins at equal
capacity; `up` can only say how much count alone was worth. And `down` alone would not be enough
either — a random projection **destroys** information, so a rung that loses there might have lost to
the projection. The two together bracket the answer; neither does on its own.

### 2.5 M3j's two recorded corrections, adopted

M3j's `## Task 9 results` §9 recorded both for a successor. This is a new pre-registration, not a
rewrite of M3j's verdict:

- **`Z_BEARING_RUNGS = ("stochastic",)`.** `full` is `h (+) z`, so a clearing `full` cannot
  attribute anything to `z`; including it made `MOTION_RETAINED` reachable on `h` alone. Verified in
  M3j: forcing `stochastic` to clear nowhere still returned `MOTION_RETAINED`, via `full`.
- **The base control gates on position alone**, not the 4-column mean over `pos_x`, `pos_y`,
  `sin(angle)`, `cos(angle)`.

**`BASE_R2_FLOOR` stays at 0.10 and is deliberately not re-chosen.** It was calibrated against
`latent_selection_r2`, itself a 4-column mean of 0.18–0.34. Gated on position alone it is very
loose: M3j measured position r^2 at **0.650–0.710** across the nine cells (arm means 0.660 / 0.682 /
0.704), clearing it more than sixfold. Raising it now would be
choosing a threshold *after seeing the numbers it gates*, which is how M3h's spec §1 went wrong and
what this project refuses. The looseness is recorded rather than corrected, and a milestone wanting
a tighter position floor must calibrate it against something other than these nine cells.

---

## 3. How it decides

### 3.1 Reading F — the statistic

```
contrast = R2([enc(t) (+) two_frame])  -  R2([enc(t) (+) deterministic])
```

Both rungs share the identical base on the identical rows, so the difference of *gains* is exactly
the difference of *joint* R^2 — the base cancels outright and no base term survives. Per cell, with
the episode-clustered block bootstrap every reading since M3e has used.

**Decided at k = 15, not by a disjunction over horizons.** M3j's objective-lever claim is
horizon-specific: `two_frame` loses at k = 1 and k = 4 even before matching, and only wins at
k = 15. Testing the claim at its own horizon is the honest test, and it avoids an incoherence a
disjunction would create — `PAST_FRAME_AHEAD` at one horizon and `RECURRENT_AHEAD` at another are
contradictory rather than complementary. k = 1 and k = 4 are reported and decide nothing.

**Two-sided, and this departs from M3j with a reason.** M3j's gain was one-sided because a negative
gain meant noise. Here both directions carry meaning: `deterministic` beating `two_frame` at equal
width is a positive finding about what `h` retains, not an absence. `SEEDS_REQUIRED = 2`,
`ARMS_REQUIRED = 2` as always, and the agreement requirement carries the multiple-comparison burden
as it has since M3h.

### 3.2 Reading F — the statuses

In precedence order. Exhaustive and mutually exclusive.

| status | rule |
|---|---|
| `UNRESOLVED_BASE` | `enc(t)` -> **position** does not clear `BASE_R2_FLOOR` in the required seeds and arms. Nothing is read. Exit **41**. |
| `UNRESOLVED_ANCHOR` | the **`down`** pass's §2.3 anchor does not reproduce its shipped gain. The pass Reading F is taken on is unreadable, so nothing is read. Exit **42**. |
| `PAST_FRAME_AHEAD` | the contrast clears **+** at k = 15, in >= 2 arms and >= 2 seeds each |
| `RECURRENT_AHEAD` | the contrast clears **-** at k = 15, same agreement |
| `INDISTINGUISHABLE` | neither clears |

**An `up` anchor failure is not `UNRESOLVED_ANCHOR`.** Reading F is taken on the `down` pass alone,
so a broken `up` anchor suppresses §3.4's width-bias companion and is reported as such -- it does
not suppress the reading. Only the `down` anchor gates Reading F. Conflating the two would let a
failure in a companion that decides nothing block a verdict that does.

### 3.3 What each status commits the project to, pre-registered

- **`PAST_FRAME_AHEAD`** — the objective lever is supported **at equal capacity**. M3j's two-lever
  ambiguity resolves toward the loss: the RSSM discards translation the encoder demonstrably
  provided. The next milestone changes the objective (`world_model.py:81`'s embedding loss
  reconstructs only the frame just seen).
- **`RECURRENT_AHEAD`** — `h` retains *more* than the past frame at equal width, so M3j's objective
  argument is refuted and it **was** width. The bottleneck lever stands alone, on evidence.
- **`INDISTINGUISHABLE`** — we could not tell the two blocks apart at equal width. **How much of
  M3j's k = 15 gap width accounts for is a magnitude, and the status licenses none**: it belongs in
  the run's own report, derived from the records it describes. The bottleneck lever stands alone **by
  default rather than by evidence**, and the results must say so: "we could not tell them apart" is
  not "we ruled one out."

  *Revised TWICE at the final whole-branch review, after the run: **only this English gloss changed;
  the decision rule did not** — the five statuses, their precedence, `SEEDS_REQUIRED`, `ARMS_REQUIRED`,
  `BASE_R2_FLOOR`, `CONTRAST_K` and the thresholds are as pre-registered. As first written this
  bullet said M3j's result "was width", which §4 forbids (`INDISTINGUISHABLE` "licenses no positive
  claim") and which contradicted the bullet's own last sentence. The two statements were inconsistent
  as written, before any data existed; §4 states the governing principle and this resolves the
  inconsistency in its favour. "Was width" is `RECURRENT_AHEAD`'s clause, where it is licensed.*
- **Either `UNRESOLVED_*`** — no lever is chosen and the next milestone is about the instrument.

This is written down now so the consequence cannot be argued after the numbers are seen.

### 3.4 Two companions that decide nothing

- **The width-bias table** — `up` minus `shipped` per rung, i.e. what count alone is worth. §1.1
  measured ~+0.0137 for `h` on two cells; this puts it on all nine and on every rung.
- **Reading E re-run on the `down` pass** under the corrected `z_bearing = ("stochastic",)`.

**The second companion is owed to M3j and may contradict it.** Remove `full` from the z-bearing set
and M3j's `MOTION_RETAINED` hangs entirely on `stochastic`, which cleared 2 of 3 arms at k = 15 with
a mean gain of **+0.00033**. Match the widths and `stochastic` also loses the 1024-against-512
handicap §1.3 identifies. So the corrected, width-matched Reading E may read `BOTTLENECK_LOSS` where
M3j read `MOTION_RETAINED`.

**That would not overturn M3j.** M3j's verdict was taken under its own pre-registered rule and
stands as recorded. It would mean the two milestones' statuses differ for reasons documented in
advance — which is why this is pre-registered here as a companion rather than discovered afterwards
and presented as a correction.

---

## 4. What it does not claim

- A **linear** probe is a lower bound. Every null says no *linear* read-out of that block at that
  width beats the reference; not that the information is absent under every decoder.
- **A random projection destroys information.** A rung that loses in the `down` pass may have lost
  to the projection rather than to the comparison, which is why `up` is run beside it (§2.4) and why
  `INDISTINGUISHABLE` licenses no positive claim.
- **`up` does not equalise rank**, only count (§2.4).
- **Backward is not forward.** Everything measured is motion the posterior has already observed.
  Nothing here says the M3 gate can pass, and no status is a statement about `gap_closed`.
- The intervals hold the probes fixed across resamples — an interval on the scored sample, not on
  the fit/select/score pipeline. `_block_bootstrap_ci`'s own caveat, inherited.
- It does not change the M3 gate, `aggregate.py`, `filtering_gain`'s recorded numbers,
  `retention.Z_BEARING_RUNGS`, or any verdict or record M3b–M3j produced.

---

## 5. Shape of the code

**`src/mbfps/eval/width.py`** — new, pure: arrays in, readings out. No torch, no I/O, no record
schema. `PASSES = ("shipped", "down", "up")`, `DOWN_WIDTH = 512`, `UP_WIDTH = 2048`,
`PROJECTION_SEED`, `ANCHOR = {"down": "deterministic", "up": "two_frame"}`, and
**`CONTRAST_K = 15`** -- deliberately NOT named `DECISION_K`, which already means **4** in
`retention.py`. Two modules exporting the same name with different values is a misreading waiting
to happen, and both are imported by the same script.
`projection(native, target)` — fixed Gaussian scaled `1/sqrt(target)`, **identity when
`native == target`**. `ContrastArm`, `ContrastInputs`, `ContrastStatus`, `reading_contrast`, and the
printed tables with their columns declared once as data.

**`src/mbfps/eval/probe.py`** — two additive changes. Extract
`_fit_and_score(fit, select, score) -> (predicted, r2, ridge)` from `gain_from_blocks`, used by
both. Add `contrast_from_blocks(a, b, *, groups, ...)`, which fits two joints on the same splits and
hands their scored predictions to `_block_bootstrap_ci` — that function already computes
`r2(A) - r2(B)` per resample, so the contrast needs no new statistic. **`gain_from_blocks`'s output
stays byte-identical, and the existing golden pin is exactly the test that proves the extraction was
safe.**

**`src/mbfps/eval/retention.py`** — one additive change: `reading_retention(inputs, *,
z_bearing=Z_BEARING_RUNGS)`. **The module constant is not edited.** Editing it would silently
re-read M3j's records under a rule they were never taken under; a default-valued parameter keeps
both readings alive at once, each reproducible from its own artefacts. A later reader must not
"simplify" this by hardcoding the corrected tuple.

**`scripts/latent_width.py`** — new. Phases `measure | read | all`. Cells load through
`trust_horizon.prepare_cell` via the `_cell_args` shim, inheriting the reviewed loader, the protocol
checks (12, 14) and M3h §2.4's reproduction bound. **One gather per cell, shared across all three
passes.** One record per cell carrying all three, plus the projection seed and every matrix's shape.
`read` writes `width.txt`, byte-identical to what it prints.

**Exit codes.** 0 / 11 / 12 / 14 / 30 inherited; **41 `EXIT_BASE_UNRESOLVED`** and **42
`EXIT_ANCHOR_BROKEN`** new. M3i holds 38 and M3j holds 39–40, so there is no clash. The registry
test in `tests/eval/test_diagnose_dynamics_script.py` gains `latent_width`.

**Tests** — `tests/eval/test_width.py` and `tests/eval/test_latent_width_script.py` in the
established idiom: hand-typed expected values, one rule mutated per test, and the captions pinned
against the columns they caption. Three carry the most weight:

- **The anchors as known-answer tests:** a `down` pass reproduces `deterministic`'s shipped gain and
  an `up` pass `two_frame`'s, to floating-point equality — because `projection` returns the identity
  there, not an approximation.
- **The lift adds no information:** a lifted block's **rank** is pinned equal to its native width, so
  the pass's claim is checked rather than asserted.
- **One end-to-end test per Reading F status**, both refusals included.

---

## 6. The run

- **Smoke:** one cell into `runs/m3k_smoke/`, timed, its `width.txt` read for format before the real
  run. M3j's smoke caught a bug that had silently discarded 17% of the evaluation data and that the
  whole suite had missed; it is not optional.
- **Measure:** nine cells, one gather each, three passes. Under `caffeinate -dimsu` and `nohup`,
  with `.head` / `.started` / `.finished` / `.exit` beside the log. **Wait on the process, not on a
  file** — M3i lost time to a waiter that fired on a stale marker.
- **Acceptance:** M3j's list — nine records at one `git_sha` equal to HEAD, all `mps`,
  `self_check.ok` on 9/9 within the reproduction bound, 229 windows over 24 episode clusters, rows
  `{k1: 11221, k4: 10534, k15: 8015}`, every gain and bound finite, `nonfinite` empty — **plus two**:
  both §2.3 anchors reproduce on all nine cells, and the projection seed and every matrix shape are
  recorded so the passes are reproducible.
- **Before any full-suite run:** `du -sh $TMPDIR/pytest-of-$USER`.

---

## 7. Files

| file | change |
|---|---|
| `src/mbfps/eval/width.py` | new — projections, Reading F, the tables |
| `scripts/latent_width.py` | new — three passes, records, `width.txt` |
| `src/mbfps/eval/probe.py` | `_fit_and_score` extracted; `contrast_from_blocks` added |
| `src/mbfps/eval/retention.py` | `reading_retention` gains a `z_bearing` keyword; the constant is untouched |
| `tests/eval/test_width.py` | new |
| `tests/eval/test_latent_width_script.py` | new |
| `tests/eval/test_probe.py` | the contrast, and the golden pin re-proving the extraction |
| `tests/eval/test_retention.py` | the `z_bearing` keyword defaults to the module constant |
| `tests/eval/test_diagnose_dynamics_script.py` | add `latent_width` (41, 42) to the exit-status registry |
| `docs/superpowers/plans/2026-09-29-mb-fps-m3k-width-matched-ladder.md` | the implementation plan |

Nothing else is modified. No checkpoint is altered, nothing under `runs/` is removed, and no
M3b–M3j verdict or record is touched.
