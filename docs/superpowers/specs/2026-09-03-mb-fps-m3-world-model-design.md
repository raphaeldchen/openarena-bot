# MB-FPS M3: World Model on Frozen Offline Data — Design

**Date:** 2026-09-03
**Governing spec:** `docs/superpowers/specs/2026-09-01-mb-fps-design.md` §6 M3
**Predecessors:** M0–M1 (`feat/m0-m1-env-and-data`, plan `2026-09-01-...`), M2 (plan `2026-09-02-...`)
**Status:** approved, pending implementation plans

---

## 1. Purpose

M3 adds the RSSM and latent dynamics, and is the first milestone where the three arms are
compared on something that matters. M2 compared them on pixel reconstruction and showed
that metric is, if anything, *anti-correlated* with semantic abstraction — DINOv2 came last
and got worse when its scale advantage was removed. M3 asks the question the study is
actually about: **does a representation support predicting where the agent will be?**

This is the portfolio centrepiece named in the governing spec, and it arrives at milestone
three rather than milestone nine.

---

## 2. The central design decision: which space the arms are compared in

Each arm's RSSM predicts its **own** encoder's embedding. `cnn` learns a 2048-d space from
scratch; the two feature arms learn different 2048-d spaces over different backbones.
**Rollout error in embedding space is therefore not comparable across arms** — an arm whose
embeddings happen to occupy a smaller scale would post a lower number for a reason with no
scientific content.

This is the same defect as M2's feature-scale confound, one level up, and standardisation
would not fix it: the spaces differ in geometry, not merely in scale.

Cross-arm comparison must therefore happen in a space all three arms share:

| candidate space | shared across arms? | verdict |
|---|---|---|
| encoder embedding | no — one per arm | training diagnostic only, never a ranking |
| pixels, via the decoder | yes | usable, but M2 showed pixel error misleads on abstraction |
| **`privileged_state`** | yes | **primary** — and physically meaningful |

**Decision: the primary cross-arm metric is position and angle error, in world units,
obtained by probing the imagined latents at each horizon step.**

Consequences, all of them good:

- Comparable by construction: metres and degrees, not embedding distance.
- It is the quantity a policy actually needs, so it predicts M4 success better than
  reconstruction does.
- The persistence baseline becomes physically interpretable — "the agent never moved", so
  its error *is* the true displacement.
- It is legible without a background in representation learning: "0.4 m of position error
  after 45 imagined steps" communicates; a latent MSE does not.

Embedding-space error is still logged — it is what the loss optimises and it is the primary
training diagnostic — but it is labelled per-arm and never used to rank arms.

`privileged_state` remains **evaluation-only**. It is read by the probe, which is fit
post-hoc on frozen latents with no gradient path to the world model, on held-out episodes.
The existing isolation test is extended rather than relaxed.

---

## 3. Evaluation design

### 3.1 Rollout protocol

Per the governing spec: condition on **5 real frames**, then imagine **45 steps** driven by
the recorded action stream alone, with no further observations.

### 3.2 The interpretive band

A rollout error is meaningless in isolation. Every reported error is bracketed:

| reference | how it is computed | meaning |
|---|---|---|
| **persistence** (upper) | hold the last context frame's embedding constant for all 45 steps | what "learned nothing" scores |
| **RSSM** | probe the imagined prior latents | the model under test |
| **encoder floor** (lower) | probe the encoder embedding of the *real* frame at each step | the best any dynamics model could reach, given this encoder |

The floor matters because it separates two failures M2's pixel metric could not: *the
dynamics model is weak* versus *the encoder already discarded this information*. Without
it, a low-scoring arm is uninterpretable.

### 3.2a Measured design constants (spike, 2026-09-04)

The evaluation design in §3.2 was specified on paper and then measured. Three of its
implicit choices were wrong, and are corrected here. All figures come from a 2,000-step
`random_vit` run on the real dataset, 229 validation windows.

**All three references must pass through the identical pipeline.** The band is computed on
`emb_head(latent)` for the model, for persistence, and for the floor — never on raw encoder
embeddings. Fitting the probe on real embeddings and applying it to the model's *predicted*
embeddings is a distribution mismatch, and it was destroying most of the signal:

| | probe on real embeddings | probe in-distribution |
|---|---|---|
| horizon steps with band below 2 SE | 18/45 | **2/45** |
| horizon steps with band <= 0 | 9/45 | **2/45** |
| median band | +21.15 | **+53.23** |
| `gap_closed` at horizon 45 | -7.26 | **-0.78** |

**`free_bits` is 0.20, not the 1.0 nat of governing spec §3.5.** At 1.0 the dynamics prior
receives gradient on 1 of 9 sampled steps, because the measured dyn KL rarely clears the
floor; at 0.20 it receives gradient on 8 of 9. A floor of exactly 0 is also wrong — it
penalises every nat and collapses the posterior (KL 0.0100 -> 0.0001 over 3,000 steps).

| free_bits | prior_net gets gradient | kl_dyn at end |
|---|---|---|
| 0.00 | — | 0.0001 (collapsed) |
| 0.05 | 7/9 steps | 0.354 |
| **0.20** | **8/9 steps** | **0.333** |
| 1.00 (governing spec) | 1/9 steps | 0.717 |

**Rollouts stay stochastic; determinism comes from seeding.** Taking the categorical mode is
not a neutral way to remove sampling noise — it collapses the imagined trajectory and
roughly triples the error. It also gets *worse* with training, as sharpening logits make the
mode more dominant, so an untrained smoke test will not reveal it.

| rollout mode | distinct latents / 45 | position error at 45 |
|---|---|---|
| **stochastic** | **45** | **356** |
| argmax | 3 | 1406 |
| mean | 7 | 363 |

**The probe needs ridge selection over standardised inputs.** A fixed `ridge=1.0` on
unstandardised features underfits badly against targets of std ~240 (Doom map units). The
measured optimum is 10^3–10^5, and the difference is R^2 0.16 against a ceiling of 0.42.

**Decodability ceiling.** A linear probe on raw cached features reaches held-out R^2 of
**0.422** (DINOv2) and **0.369** (random_vit). Position is therefore only moderately
linearly decodable here, which bounds what any arm can score — and is exactly why §3.2's
floor is needed to interpret the numbers.

Note the ordering: **DINOv2 leads on position decoding while it came last on M2's pixel
reconstruction.** That is the first direct evidence for this study's hypothesis, and it
independently vindicates §2 — had M3 compared arms in pixel space it would have ranked them
backwards. One seed, one scenario, frozen features: suggestive, not a result.

### 3.3 The headline comparable number

```
gap_closed = (persistence_error - rssm_error) / (persistence_error - floor_error)
```

Bounded in [0, 1] under normal conditions: **0** means no better than assuming the agent
never moved; **1** means as good as this encoder permits. Measured behaviour, with the §3.2a
corrections applied: the band carries statistically meaningful width at 43 of 45 horizon
steps, and a 2,000-step model scores -0.78. **Report the raw persistence/model/floor curves
alongside the ratio**, since the band is roughly 25% of the error magnitude and a ratio over
a narrow denominator deserves its inputs shown. It is dimensionless and therefore
comparable across arms even though the arms' encoders have different floors — which is
precisely the property embedding MSE lacks.

Negative values are possible and are reported, not clipped: they mean the model is worse
than persistence.

### 3.4 The probe

A **linear** probe, deliberately — a nonlinear probe measures the probe's capacity as much
as the representation's content. Fit on held-out episodes only.

Two probes, serving different questions:

- **Imagination probe (primary).** Prior latents at each of the 45 imagined steps → position
  and angle. This is what feeds §3.2 and §3.3.
- **Filtering probe (gate criterion §4.4, and a diagnostic).** Posterior latent at step
  *t* → state at *t*, compared
  against a probe on the encoder embedding at *t*. The posterior has seen frame *t*, so it
  cannot add information about *t* by the data-processing inequality; what it can add is
  **history**. If it does not beat the embedding probe, the deterministic state `h` is
  carrying nothing, which is a specific, actionable bug.

Note that the "encoder-embedding reference" and the persistence baseline coincide in the
imagination setting — probing the last context embedding *is* the persistence prediction.
The two gate criteria chosen for this milestone therefore reduce to one measurement, which
is a sign the design is coherent rather than two loosely related checks.

---

## 4. Exit gate

M3 passes when, **across 3 arms × 3 seeds**:

1. **Beats persistence.** For every arm, **all three seeds individually** post
   `gap_closed > 0` at horizon 45. An arm that cannot beat "the agent never moved" has not
   learned dynamics.

   Stated as a unanimity requirement rather than a t-test on purpose: with n = 3 a t-test
   carries 2 degrees of freedom and would dress up a weak result in strong-looking notation.
   Unanimity across three seeds is a sign test at p = 0.125 under a coin-flip null — modest,
   honestly so, and the most three seeds can support. Seed count is a compute decision, and
   if the arms turn out to separate narrowly, the correct response is more seeds, not a
   better-sounding test.
2. **Per-step error-vs-horizon curve** produced for each arm, with persistence and floor
   drawn on the same axes.
3. **Reward-prediction accuracy** reported per arm.
4. **Filtering probe beats the encoder-embedding probe**, showing `h` carries history.
5. **All invariant tests green** (§6) — overfit-one-batch, gradient-flow,
   privileged-state isolation, determinism, golden rollout, split stability. The
   imagination-isolation test is *not* an M3 criterion: it asserts that an actor-critic step
   leaves world-model gradients untouched, and there is no actor until M4. M3 builds its
   harness; M4 gates on it.

No absolute threshold is set on `gap_closed`. M2 demonstrated the cost of arbitrary
thresholds — its 4,000-step budget was chosen without measurement and produced a reading
that was an optimisation artifact. The gate is relative to baselines computed on the same
data; the *value* of `gap_closed` is the finding, not a hurdle invented in advance.

**The gate does not require the arms to separate.** If all three post similar `gap_closed`,
that is a legitimate and reportable result: it would say pixel-space representation choice
does not much affect short-horizon latent dynamics in this scenario. Designing a gate that
can only pass if the hypothesis is confirmed would make the study unfalsifiable.

---

## 5. Architecture

### 5.1 Modules

```
models/rssm.py            RSSM: GRU deterministic path, categorical stochastic
                          state, prior/posterior, KL balancing, free bits
models/heads.py           embedding predictor, reward head, continue head
training/world_model.py   training loop
eval/rollout.py           open-loop rollout, persistence, encoder floor
eval/probe.py             linear probe, held-out fitting
scripts/train_world_model.py
scripts/eval_rollout.py
```

Hyperparameters follow governing spec §3.5: `h` = 512, `z` = 32 categoricals × 32 classes,
head MLPs 2 × 512, batch/sequence 16/64, KL free bits 1.0 nat, KL scales dyn/rep 0.5/0.1,
world-model LR 1e-4, fp32.

### 5.2 Arm parity survives

The RSSM consumes the 2048-d embedding, which is already equalised across arms, so
`rssm.py`, `heads.py` and `world_model.py` are all arm-invariant. **`encoders.py` remains
the only module that differs between arms.** That the invariant holds unchanged through a
milestone this much larger is evidence the M2 boundary was drawn in the right place.

### 5.3 Generalising the M2 decoder-parity bug

M2 found that the shared decoder initialised differently for `cnn` because `CNNEncoder`
consumed 26,382,304 global-RNG draws before the decoder was built. The fix was local, but
the *defect* was not about decoders: **any module constructed after the encoder inherits an
RNG the encoder has advanced.** M3 adds four more such modules and therefore four new places
for the same bug.

Rather than four more `fork_rng` blocks, add:

```python
seeding.fork_seed(base_seed: int, name: str) -> int
```

deriving a stable per-module seed from a hash of `name`. Every module's initialisation then
depends only on the run seed and the module's own identity — not on construction order, and
not on how large some other module happens to be. The existing decoder fix folds into it.

### 5.4 Held-out episode split

M2 had no held-out split, and its numbers are explicitly recorded as training-set
reconstructions. A probe R² fit and reported on training data measures memorisation, so M3
requires a split.

- Split at **episode** level, not window level — windows drawn from one episode share frames,
  so a window-level split leaks across the boundary.
- Fixed by seed and **pinned by a test**. A split that silently differed between arms would
  invalidate every cross-arm comparison, and would do so invisibly.

### 5.5 `terminated` versus `truncated`

The continue head predicts non-termination. A time-limit truncation must not train it toward
"the episode ended" — the episode did not end, the clock ran out. Conflating them corrupts
bootstrapping, which M4 depends on directly.

This distinction had **zero test coverage** until 2026-09-03: every loader fixture filled
both arrays with zeros and asserted only shape and dtype, so slicing `truncated` from
`ep.terminated` passed the entire suite. It is now pinned by value-level tests. M3 makes it
load-bearing.

---

## 6. Testing

Carries the governing spec's §7 invariants, all of which apply for the first time here:

| test | catches |
|---|---|
| **Overfit-one-batch** — loss → ~0 on 4 fixed sequences | a failure here is a bug, not a hyperparameter |
| **Gradient-flow** — every trainable parameter has non-`None`, non-zero grad after one step | detached tensors, the classic RSSM bug |
| **Imagination-isolation** — after an actor-critic step every world-model param has `grad is None` | unit-tests the project's central claim (lands in M4, harness built here) |
| **Privileged-state isolation** — privileged keys never appear in a training tensor | silent invalidation of the whole experiment |
| **Determinism** — same seed twice yields identical trajectories | nondeterminism poisoning arm comparisons |
| **Golden rollout** — fixed seed + weights → hash of imagined latents | silent regressions during refactor |
| **Split stability** — the train/val split is identical across arms and runs | invisible invalidation of every comparison |

Standing discipline, carried from M0–M2: **every guard is mutation-tested.** A test that
cannot fail is worse than no test, because it is counted as coverage. M2's whole-branch
review found eleven mutations the committed suite could not catch.

**Mutation harnesses must self-check.** M2's first harness reported every mutation caught,
including deliberately fatal ones: the package is installed editable, so `import mbfps`
resolved to the real working tree and the mutated export was never loaded. Run mutations
with `PYTHONPATH` pointed at the export, and prove the harness works with a known-fatal
mutation before trusting any result from it.

---

## 7. Compute plan

M3 does not fit on the development Mac. Estimating from M2's measurements — `cnn` 4.66
ms/frame, feature arms ~2.35 ms/frame at `seq_len=1` — a `seq_len=64` step processes 1,040
frames, and the RSSM adds 64 sequential recurrent steps whose small kernels pay MPS launch
overhead. 3 arms × 3 seeds at 20k steps plausibly exceeds 200 hours. The host also logged a
`Thermal Emergency Sleep` during a 50-minute M2 run.

**That estimate is not trusted.** The equivalent M2 estimate predicted a 9.6× arm speed ratio
and the measured value was 1.96×. The first task of implementation therefore **measures**
real per-step cost with a minimal RSSM, and the study is scoped from that number.

Unattended runs require `caffeinate -dimsu`; `caffeinate -i` asserts only
`PreventUserIdleSystemSleep` and was insufficient — wall-clock ran 2.3× active compute until
the stronger assertion was applied. Note that `time.perf_counter()` on macOS halts during
system sleep, so recorded `steps_per_second` is trustworthy but **cannot self-detect
suspension**.

### Work split

The milestone splits at the venue boundary, which is also where the work changes character:

- **Plan 3 — world model and evaluation harness.** Local, single arm, small scale. Gate: the
  pipeline is proven — invariant tests green, overfit-one-batch converges, a rollout runs
  end-to-end and produces the band of §3.2.
- **Plan 4 — the 3 × 3 study.** Rented GPU. Gate: M3's real exit criteria (§4).

Plan 4 is written **after** Plan 3's cost measurement, so the study's scale comes from a
measurement rather than an estimate. M2's 4,000-step budget was specified in advance without
measurement and produced a result that was an optimisation artifact rather than a
representation finding; this ordering exists to avoid repeating that.

---

## 8. Carried constraints

1. `encoders.py` is the only module that may differ between arms.
2. `privileged_state` is evaluation-only and must never reach a training tensor.
3. `terminated` and `truncated` stay distinct everywhere.
4. All arms consume byte-identical 112×112×3 uint8 frames and emit 2048-d embeddings.
5. Feature caches are namespaced per backbone; no arm may read another's.
6. Every guard is mutation-tested, with a self-checked harness.
7. Cross-arm claims require multiple seeds. Single-seed differences are descriptive only —
   with one checkpoint per arm, paired SEM shrinks as `1/sqrt(draws)` without bound and the
   resulting t-statistic carries no effect-size meaning.

---

## 9. Open questions

- ~~**Does `gap_closed` behave well when an arm's floor is very close to persistence?**~~
  **ANSWERED (§3.2a).** It does not, when the probe is fit on a different distribution from
  the one it is applied to: the band was dead at 18 of 45 horizon steps. Routing every
  reference through the identical pipeline fixes it — 2 of 45 — and the raw curves are now
  reported alongside the ratio regardless.
- **Is 45 imagined steps the right horizon for `my_way_home`?** Episodes are ≥65 steps, so 5
  + 45 fits. Whether position error saturates well before 45 is an empirical question the
  error-vs-horizon curve answers directly.
- **Which GPU instance type.** Deferred to Plan 4, when the measured per-step cost is known.
