# MB-FPS M3l — is `z` out of room, or was it never asked?

## 1. The question, and why M3k could not answer it

M3j established the largest and most unanimous effect in the study: the recurrent
state `h` carries observed motion and the stochastic latent `z` does not.

| effect | size | between-seed sd | effect / noise | unanimity |
|---|---|---|---|---|
| `h` − `z`, translation, k = 15 | +0.02038 | 0.00814 | 2.50 | **9/9** |
| `h` − `z`, rotation, k = 15 | +0.32477 | 0.07681 | 4.23 | **9/9** |

The rollout runs on `z`, because the prior is what predicts forward. So motion
reaches `h` and does not reach the thing the rollout uses.

**But that observation is common to both candidate levers and discriminates
neither.**

- **Bottleneck story** — `z` lacks the capacity. A 32 × 32 categorical latent
  cannot represent the displacement `h` holds.
- **Objective story** — `z` has capacity to spare, but the embedding loss
  targets the *current frame*, which is position-like, so nothing ever pushes
  motion into it.

M3k was built to discriminate them by asking whether `two_frame`'s advantage over
`h` survives width matching. It returned `INDISTINGUISHABLE`, and the residual is
not resolvable at reasonable cost:

| quantity | value |
|---|---|
| effect (mean contrast, `down` pass, k = 15) | +0.00247 |
| within-cell noise (median bootstrap half-width) | 0.00941 |
| between-seed spread (sd of the nine contrasts) | 0.01110 |
| effect / between-seed noise | **0.22** |
| evaluation episodes needed for one cell to clear | ~348 clusters, ≈ **15×** the data |

More seeds do not help. The per-seed clearing rate is 0.33, and a majority bar
recedes as *n* grows whenever the rate is below one half: P(an arm clears) falls
from 0.259 at three seeds to 0.210 at five. Seeds estimate the between-seed
spread; they never reduce it. And with tighter intervals the seeds would still
disagree — they would merely disagree *cleanly*, which `reading_contrast` refuses
as incoherent.

**So M3l does not re-ask M3k's question.** It asks the one question that
separates the two stories directly, and it asks it as a **level against an exact
ceiling** rather than as a difference of two fitted quantities. That is the
property M3k lacked, and the reason its statistic had effect/noise 0.22.

## 2. What is measured

Per cell, over the same validation rows M3j and M3k used, from the nine
checkpoints already on disk. **No training. Evaluation only.**

### 2.1 `bits_carried` — the code's information content

```
bits_carried = Σⱼ [ H(marginalⱼ) − E_n H(postⱼ(n)) ]        j = 1 .. z_cats
```

`postⱼ(n)` is row `n`'s posterior distribution over categorical `j`'s classes;
`marginalⱼ` is those distributions averaged over rows. Both terms are computed in
**bits**, exactly from the distributions — no sampling.

**Ceiling: `z_cats × log₂(z_classes)` = 32 × 5 = 160 bits**, derived from
`RSSMConfig`, never re-spelled.

The statistic is properly `I(z ; h, enc(t))` — the code's *total* information
content, not its information about the current frame. The posterior conditions on
both `h` and `enc(t)`, and this measures how much the code varies across rows for
any reason. §2.2 is what separates the two sources.

The estimator uses the softmax the model **actually samples from**, i.e. with
`cfg.sample_temperature` applied, because that is the distribution `z` is drawn
from and therefore the one whose information content matters. At the shipped
`sample_temperature = 1.0` no division is taken, so this is bitwise the plain
softmax today; the division is written so a future temperature change is
reflected rather than silently ignored.

### 2.2 `frame_share` — what those bits are carrying

`R²(enc(t) → post_probs)` under the project's three-split discipline: weights fit
on the fit split, ridge selected on further held-out training episodes, R² scored
on validation. A generalisation number, not a fit statistic. Ceiling 1.0.

High `frame_share` means `z` is largely a re-encoding of the current frame —
which is what the embedding loss asks for.

The target is `post_probs` flattened to `z_cats × z_classes` = 1024 columns, and
`R²` is the mean over columns. **Columns with zero variance across the scored rows
are excluded from that mean**, because R² is undefined for them — this project has
already refused a fixture for exactly that reason. Their count is reported as
`live_classes` out of 1024 and is itself a saturation signal: a class that never
wins and never varies is capacity the code is not using. It decides nothing, but a
`CAPACITY_BOUND` verdict printed beside a low `live_classes` would be
self-contradicting, so both appear in the table.

### 2.3 Two controls with exact known answers

Both are substitutions into the **real pipeline on the real data**, not fixtures.
This is the same construction as M3k's anchors: a known answer computed through
the code path under test.

| control | substitution | must read |
|---|---|---|
| `floor_bits` | every row's posterior replaced by `marginalⱼ` | **exactly 0.000 bits** |
| `ceiling_bits` | every row's posterior replaced by a one-hot at its argmax | **exactly `Σⱼ H(marginal of argmaxⱼ)`** |

`floor_bits` is identically zero by construction — `H(m) − mean_n H(m)` — so any
nonzero reading is an arithmetic defect: a wrong log base, a missing
normalisation, a mean over the wrong axis.

`ceiling_bits` is computed **twice by different routes** — once as
`bits_carried` of the one-hot substitution, once as a histogram over the argmax
*indices* — and the two must agree. That pins the routing: a mean taken over the
wrong axis, or a substitution that leaks a row's own distribution, breaks the
agreement.

**Neither control pins the log base, and a third check is needed for it.**
Measured: in nats the floor still reads 0.000000 and the two ceiling routes still
agree exactly (109.8100 both ways, against 158.4223 in bits). Nor does the
`≤ 160` inequality catch it, because nats reads *lower* than the bits ceiling, not
higher. The base is pinned only by a case with a known **absolute** value: a
fixture whose posterior is a deterministic function of the row and whose argmax is
uniform across the dataset must read exactly **`log₂(z_classes)` = 5.000 bits per
categorical**, where nats would read 3.466. That fixture is the base check; the
two controls above are the arithmetic and routing checks.

`bits_carried` must satisfy `0 ≤ bits_carried ≤ ceiling_bits ≤ 160`. A violation
of any of those three inequalities is an error about the measurement and refuses
the reading. Note what this does **not** catch: see the log-base paragraph above.

### 2.4 A companion that decides nothing

`prior_bits` — the same estimator on `prior_logits`. The prior does not see
`enc(t)`, so this is the information the prior's distribution carries through `h`
alone. It is **not** a known-answer control: the prior is a function of `h`, `h`
encodes past frames, so its distribution genuinely varies across rows and the
reading is positive. It is free from the same forward pass, and it speaks to
M3g's finding that the prior is the failing stage, so it is reported. It gates
nothing.

## 3. Reading G

### 3.1 The statuses and their precedence

| status | condition | consequence |
|---|---|---|
| `UNRESOLVED_ESTIMATOR` | a control missed its known answer, or an inequality in §2.3 is violated | no reading; exit 43 |
| `UNRESOLVED_BASE` | `enc(t) → position` below `BASE_R2_FLOOR` | no reading; exit 44 |
| `SPARE_CAPACITY` | `bits_carried` interval entirely below half the ceiling | **objective lever** |
| `FRAME_REENCODING` | not spare, and `frame_share` interval entirely above 0.5 | **objective lever** |
| `CAPACITY_BOUND` | not spare, and `frame_share` not above 0.5 | **bottleneck lever** |

Precedence is the order of the table. The three readings are **mutually exclusive
by construction** — the `bits_carried` cut partitions `SPARE_CAPACITY` from the
other two, and `frame_share` partitions those two — so no arm can clear in two
directions and the entire class of ambiguity refusal M3k required cannot arise
here.

An arm clears a status in `SEEDS_REQUIRED` of its seeds; a status is read when
`ARMS_REQUIRED` arms clear it. Both are imported from `retention`, not re-spelled.

A status requires the **bootstrap interval** to clear the cut, never the point
estimate. Intervals are episode-clustered block bootstrap at `CONFIDENCE`, as
everywhere else in the project.

### 3.2 Where the two cuts come from

Both cuts are the same number from the same principle: **"most of" means more than
half.** Half the 160-bit ceiling for capacity; half the variance for
`frame_share`.

That principle is already load-bearing in this project, and M3l is its third
instance:

1. **M3j / M3k's seed bar** — `SEEDS_REQUIRED = 2` of 3 is the weakest *majority*
   of seeds that also makes both-directions-clearing impossible
   (`2 · SEEDS_REQUIRED > seeds_total`).
2. **The same rule at five seeds** yields 3, derived rather than chosen — which
   is how M3l establishes the principle predicts the number already
   pre-registered at a count already run, not merely the number wanted next.
3. **M3l's two cuts** — a majority of the capacity, a majority of the variance.

A reader can check the derivation against instances 1 and 2 rather than taking a
chosen constant on trust.

The claim "capacity is the binding constraint" *requires* that most of the
capacity is in use. If more than half the code is idle, adding capacity cannot be
what is limiting the model, whatever else is. That is the content of the first
cut, and it is why it is a half rather than a tuned value.

### 3.3 The asymmetry, stated before the numbers

Requiring the whole interval below half the ceiling makes `SPARE_CAPACITY`
**harder** to reach. Requiring `frame_share` entirely above 0.5 makes
`FRAME_REENCODING` **harder** to reach. `CAPACITY_BOUND` is the **fall-through**:
it is what remains when neither objective-lever status clears.

So the rule makes the bottleneck lever — the one the `h` − `z` evidence already
leans toward — the *easiest* status to reach. That is the wrong way round, and it
is written down here so the consequence cannot be argued after the numbers are
seen:

**If Reading G returns `CAPACITY_BOUND`, the results must say plainly that it
arrived by fall-through rather than by clearing a bar**, in the same words M3k
used for `INDISTINGUISHABLE`: by default rather than by evidence. A
`CAPACITY_BOUND` verdict is weaker evidence for the bottleneck lever than either
objective-lever status would be against it.

`bits_carried`, `frame_share`, `prior_bits` and both controls are reported per
cell as raw levels regardless of status, so a reader may apply a different cut to
the numbers rather than inheriting this one.

## 4. Architecture

| file | responsibility |
|---|---|
| `src/mbfps/eval/probe.py` | **modify, additively.** `gather_probe_data` gains `"post_probs"` and `"prior_probs"`, each `(N, z_cats, z_classes)`. No existing key changes. |
| `src/mbfps/eval/capacity.py` | **create.** Pure: the estimator, the ceiling, the controls, Reading G, its table. No torch, no `Path`, no I/O, no record schema. |
| `scripts/latent_capacity.py` | **create.** Owns torch, I/O and the record schema: one gather per cell, both statistics, both controls, the prior companion, measure and read phases. |
| `tests/eval/test_probe.py` | **modify.** The additivity pin. |
| `tests/eval/test_capacity.py` | **create.** |
| `tests/eval/test_latent_capacity_script.py` | **create.** |
| `tests/eval/test_diagnose_dynamics_script.py` | **modify.** 43 and 44 in the exit-status registry. |

The split mirrors `retention.py` / `latent_retention.py` and `width.py` /
`latent_width.py` exactly: one pure module per reading, one script owning torch
and the schema. If a function needs a `Path` or a `torch.device` it belongs in the
script.

### 4.1 Touching the gate path

`gather_probe_data` feeds the research gate through `src/mbfps/eval/study.py` and
`scripts/eval_rollout.py`, so the change must be **provably additive**.

It is, structurally: `post_logits` and `prior_logits` are **already computed**
inside `RSSM.observe` — it returns them today — so collecting them consumes no
randomness and adds no operation. The change cannot perturb an RNG draw.

That argument is pinned by test rather than asserted: gather twice under one
seed, once with the new keys suppressed, and assert **every pre-existing key is
byte-identical**. `gain_from_blocks`' golden pin
(`test_gain_from_splits_output_is_byte_identical_after_the_generalisation`) then
re-proves the downstream path, as it did in M3k. **That pin's ten values are
never re-recorded.**

Memory: one probs array is ~45 MB at ~11,000 rows, two arrays ~90 MB, against the
1.3–1.6 GB the existing measure already peaks at.

### 4.2 One gather per cell

Both statistics and both controls come from **one** gather, so every number
describes one row set. Deriving `frame_share` from a second gather would
reintroduce the row-alignment hazard M3k needed two layered guards for: a builder
returning the same count of rows in a different order silently misaligns a block
against its own target, with no refusal anywhere.

## 5. Constraints

- **Evaluation only.** No training, no checkpoint written or altered, nothing
  under `runs/` removed.
- **The ceiling is derived from `RSSMConfig`**, never hardcoded. A hardcoded 160
  would silently disagree with the model if the latent shape changed.
- **`BASE_R2_FLOOR` stays 0.10** and is not re-chosen, gated on **position
  alone** — as corrected in M3k.
- **`SEEDS_REQUIRED` and `ARMS_REQUIRED` are imported from `retention`**, not
  re-spelled.
- **`retention.Z_BEARING_RUNGS` is not edited**, and no M3b–M3k verdict is
  touched. M3l adds a reading; it revises none.
- **Exit codes: 43 `EXIT_ESTIMATOR_BROKEN`, 44 `EXIT_BASE_UNRESOLVED`.** 38 is
  M3i, 39–40 M3j, 41–42 M3k.
- **`capacity.txt` must be byte-identical to a second `--phase read`.**
- **Test command:** `.venv/bin/python -m pytest` from the repo root. There is no
  `pytest` entry point in the venv.

## 6. Cost

| step | estimate |
|---|---|
| one gather per cell (forward pass, no ridge ladder) | ~1–2 min/cell |
| one ridge probe per cell for `frame_share` | ~1.5 min/cell |
| nine cells | **~30 min** |
| read phase | seconds |

Against M3k's 81 minutes, because M3l runs one probe per cell rather than
forty-eight, and the bits estimator is a pass over distributions with no linear
solve. The dominant cost in M3k was `np.linalg.solve` at ~96% of runtime, which
M3l largely does not pay.

Budget generously: the M3k run had one cell take 28 minutes against 5m42s–8m10s
for the other eight when the host swapped. Watch free disk.

## 7. Exit criteria

- Reading G taken on nine records at one `git_sha`, or the run refused with a
  numbered status and the refusal recorded.
- `floor_bits` reads **exactly 0.000** on all nine cells.
- `ceiling_bits` matches its independently computed argmax-marginal entropy on
  all nine cells, in bits.
- `0 ≤ bits_carried ≤ ceiling_bits ≤ 160` on all nine cells.
- Every pre-existing `gather_probe_data` key byte-identical to before the change,
  pinned by test; `gain_from_blocks`' ten-key golden output unchanged.
- All five Reading G statuses reachable and driven end-to-end by a test.
- The whole suite passes with 0 failures and 0 warnings.
- `capacity.txt` byte-identical to a second `--phase read`.
- No checkpoint altered, nothing under `runs/` removed, no M3b–M3k verdict
  touched.

## 8. What this does and does not license

**It can refute the bottleneck lever.** `SPARE_CAPACITY` or `FRAME_REENCODING`
means adding capacity is not what is limiting the model, and the ~13.5 hours
should go to the objective instead.

**It cannot prove the bottleneck lever**, only fail to refute it — see §3.3.
`CAPACITY_BOUND` is a fall-through, and the results must say so.

**It does not reopen M3k.** Reading F stays `INDISTINGUISHABLE`. M3l's finding is
about a different quantity measured a different way, and the two are companions
rather than a revision.

**It does not move the M3 gate.** The gate is `beats_persistence` and eight
criteria; M3l is instrumentation, like M3d–M3k. It chooses which lever the next
milestone pulls.

## 9. Spec self-review

**Placeholders.** None. Every threshold is derived in §3.2 or imported per §5.

**Internal consistency.** §2.1's statistic is named `I(z ; h, enc(t))` and §2.4
explains why the prior is a companion rather than a control — the earlier
formulation of this design claimed the prior was a zero-answer control, which is
false because the prior is a function of `h` and `h` encodes past frames. §2.3
carries the two controls that *are* exact.

An earlier formulation also claimed the two controls "together pin the log base."
They do not — measured, nats gives floor 0.000000 and identical ceiling routes,
and reads *below* the 160-bit ceiling so no inequality catches it. §2.3 now
separates the three checks: floor for arithmetic, the two ceiling routes for
routing, and a known-absolute-value fixture for the base.

The ceiling is 160 in §2.1 and derived from `RSSMConfig` in §5; no section
hardcodes it.

**Scope.** One reading, two statistics, two controls, one companion, one script,
one pure module. Comparable to M3j and smaller than M3k, which needed three
passes and two anchors.

**Ambiguity.** Two readings of "half the ceiling" were possible — half of 160
bits, or half of the *measured* `ceiling_bits`. §3.1 means **half of the derived
160-bit ceiling**, so the cut does not move with the data. Stated explicitly here
because the alternative is a threshold that a degenerate code could satisfy by
collapsing its own ceiling.
