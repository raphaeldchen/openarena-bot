# MB-FPS M3h — The sharper latent: is the rollout noise-limited, and does training it sharper help? (Design)

**Status:** approved in conversation 2026-09-23 (intervene on the latent's stochasticity, not on the prior's learning signal; a rollout-temperature sweep as a pre-registered gate on the retrain; one spec covering both phases with the strength a rule of the sweep's outcome; sections 1–3 approved in turn). The first M3 study that changes training. The M3 gate and every recorded M3b–M3g verdict are untouched; no shipped checkpoint is altered.
**Reads:** `runs/m3_study_v2` — the nine M3c cells (checkpoints, study records, diagnostics; code state `ca3e140`, device `mps`); `data/my_way_home`.
**Writes:** `runs/m3h_sweep/` — one sweep record per cell, `sweep.txt`, `sweep_curves.png`; `runs/m3h_sharper/` — nine retrained cells in the study's layout, their study records, diagnostics and trust records, `sharper.txt`, `sharper_curves.png`; one results section in this study's own plan.
**Base:** `main` at `e3370e2` (PR #6, M3g, merged); branch `feat/m3h-sharper-latent`.

## 1. Why

Four hypotheses for the M3 gate failure are closed by pre-registered readings. M3e: no gap between the episodes the model trained on and the held-out ones — not memorisation. M3f: evaluated at its own one-step-loss minimum each ViT arm rolls out *worse* than at step 20,000 — not checkpoint timing. M3g: the posterior carries the frame (encode passes 3 of 3 seeds in both ViT arms) and the prior is the stage that fails at h = 15.

But M3g's own disclosure, and the measurements made designing this study, say the prior's failure is a near-tie rather than an absence (scratch measurements over the same 229 validation windows; not recorded results):

- KL(posterior ‖ teacher-forced prior) is **0.0125–0.019 nats per group** at 20k: the prior already matches the posterior's *distribution* closely. Its argmax differs because both are diffuse — the prior's top-1 mass is 0.51–0.57 and the posterior's mode is the prior's top-2 in 99 % of cases — so mode accuracy reads a near-tie, which is how the prior can lose to a mode-copy by 0.03 while beating the like-for-like sample-copy bar by +0.25 to +0.31 (M3g's results section).
- The posterior's entropy is 0.71–0.87 nats per group, and there are 32 groups: **each imagination step draws ~23–28 nats of fresh entropy**, against the ~0.4–0.6 nats of dynamics information the frame injects.
- In embedding space the consequence is direct. Median over the 229 windows, through the same head: the imagined displacement ‖ê(h) − ê(0)‖ against the **noise reference** — the distance between two draws of the *same* model from the *same* state, which every M3c diagnostic already carries — and the truth's displacement:

| cell | h = 1 | h = 5 | h = 15 | h = 45 |
|---|---|---|---|---|
| `pixel_ae` s2 | 1.39 / **1.45** / 16.50 | 2.74 / **3.42** / 20.98 | 3.59 / **4.97** / 23.83 | 4.45 / **6.63** / 24.60 |
| `frozen_ssl` s0 | 2.39 / **2.11** / 26.11 | 5.18 / **5.86** / 32.84 | 7.33 / **9.04** / 36.41 | 10.94 / **12.50** / 37.93 |
| `random_vit` s1 | 1.79 / **1.70** / 13.14 | 4.05 / **4.43** / 19.71 | 5.06 / **6.51** / 24.39 | 5.36 / **9.74** / 27.92 |

  (imagined / noise / true; every one of the nine cells has this shape.) **From h = 5 onward the model's own resampling noise exceeds its entire imagined motion**, and both are 3–7× smaller than the truth's.

So the dynamics signal is not missing, it is swamped. M3h asks whether the rollout is noise-limited, and if it is, trains the latent sharper.

> ***Corrected after the run (2026-09-26). The bolded claim above, and "every one of the nine cells has this shape", are not supported and were never measured the way they are stated.*** *The figures came from scratch measurements made while designing M3h, in which the noise column was a **mean** over windows and the imagined column a **median**. Embedding distances are strongly right-skewed, so that comparison is not like-for-like and it inflates the noise side. The shipped code carried the same mismatch under a caption reading "medians over windows" until `a80e0ed` (final review, finding C2); both reductions are now recorded for both quantities.*
>
> *Measured like-for-like at h = 15, τ = 1.0, over the nine cells: **median against median, the noise exceeds the imagined displacement in 4 of 9 cells; mean against mean, in 6 of 9.** Never nine. `random_vit`/s0 reverses hardest — 2.392 against 3.705, the displacement half again larger than the noise. The claim that survives, and the one that motivated the study fairly, is the weaker one: the two-draw noise is **comparable in size to** the whole imagined displacement, which is still a striking thing to be true of a dynamics model at h = 15.*
>
> *This does not rescue the hypothesis, and the correction cuts against the study rather than for it: M3h's result is `SHARPER_WORSE`, so a premise that was weaker than stated was pointing at a lever that does not work anyway. See `## Task 10 results` §8 in the plan.*

## 2. What it does

### 2.1 Phase 1 — the sweep (evaluation only)

On the nine shipped M3c checkpoints, re-run the canonical diagnostic pass sampling the **prior inside `imagine` only** at temperature τ ∈ `TAU_GRID = (1.0, 0.7, 0.5, 0.3, 0.0)` — `Categorical(logits / τ)`, argmax at τ = 0. **Every temperature draws from the generator exactly once per step, including τ = 0, where the draw is taken and discarded before the argmax**: a sampler that skipped the draw would leave the stream at a different point and the floor's `observe`, which runs after the rollout, would no longer be the one the gate scored. The context filter and the floor's `observe` are untouched, so the persistence anchor, the floor, the band and the probe are bitwise what the gate scored; the noise reference is drawn at the same τ as the canonical pass, because "two draws of the same model" must mean two draws of the same sampler. One record per cell holding every τ's trust pass.

`prepare_cell` (exit 12, 14) loads each cell as M3e–M3g load it. **τ = 1.0 is the anchor:** the pass at 1.0 must reproduce that cell's shipped diagnostic at max abs delta 0.0 (`trust_horizon.self_check`) *— amended 2026-09-24 to the reproduction bound of §2.4, measured at 6 ULPs on macOS 27.0 where it used to be 0.0*, judged before any other τ of that cell runs; otherwise exit 30 and no record. `_sample` skips the division entirely at 1.0, so the default path is bitwise today's by construction rather than by relying on `x / 1.0`.

### 2.2 Phase 2 — the retrain

One constant, `SAMPLE_TEMPERATURE`, divides the logits inside `RSSM._sample` — the one place both the posterior and the prior draw — so the model is trained at the sharpness it is evaluated at and the mapping from the sweep is the identity: **Phase 2 runs at `SAMPLE_TEMPERATURE = τ*`**, with no second calibration and no new dial. Nothing else changes: the same nine cells (3 arms × seeds 0, 1, 2), the same M1 dataset and split, `STEPS = 20000`, batch 16 × 64, lr 1e-4, `KL_FREE_BITS = 0.20`, `dyn_scale/rep_scale = 0.5/0.1`, the same architecture and `LATENT_DIM` — so every instrument from M3b through M3g reads the new cells unchanged and the M3c nine remain the baseline on the same ruler.

**The τ = 1 identity check, in place of an anchor.** M3f measured MPS training to be bitwise deterministic on this box (the anchor exactly 0.0 at every one of 5,000 steps in 9/9 cells), so a retrain at τ = 1.0 *is* the M3c run — which makes the shipped cells the control arm without spending fifteen hours on one. It is verified, not assumed: before the real run one cell retrains `IDENTITY_STEPS = 500` steps at `SAMPLE_TEMPERATURE = 1.0`, and its per-step losses must equal the M3c record's prefix exactly (exit 37). *Amended 2026-09-24: this check keeps the exact rule and therefore refuses on macOS 27.0, because a retrain no longer reproduces a record written on the previous build at all (§2.4). Phase 2's control arm is an open question that only Reading N can make live*. That check also pins that the temperature edit is a no-op at 1.0 — the one way this intervention could silently change something it should not.

### 2.3 The checkpoint payload

`_save_checkpoint`'s payload gains `sample_temperature`. This deliberately extends an invariant the tests pin (the three-key payload, which M3f already extended with `step` for rungs), and it exists because a checkpoint trained at τ* loaded into a default configuration would be evaluated at τ = 1 in silence — the class of mis-evaluation every check in this project exists to prevent. `load_checkpoint_model` rebuilds the configuration with the payload's value and refuses a mismatch against the temperature it was asked for (exit 36); a checkpoint without the key — every M3b–M3f artefact — reads as 1.0.

### 2.4 The reproduction rule — amended mid-run (2026-09-24)

*Written after Task 9's suite was green and before Phase 1 was relaunched, because the environment moved under the study. Every clause in this spec that demands "max abs delta 0.0" of a comparison against a **stored** artefact is superseded by this subsection. In-run comparisons and training comparisons are not.*

macOS 27.0 was installed on this box at 15:00:46 on 2026-09-24 and booted at 16:39:50 — between the smoke run that reproduced bitwise at 03:01 and the sweep launched at 19:54, which refused at its first cell with exit 14 and `max abs 8.526513e-14`. torch is the same 2.13.0 wheel throughout. What was measured, before anything was changed:

**A forward pass drifts by at most 6 ULPs.** Over the nine M3c cells, the same code on the same checkpoints reproduces the shipped `curves.rssm_position` to at most **1.705303e-13** (`frozen_ssl`/s1, 6.000002 ULPs of float64; largest relative error 7.656e-16). Two cells — `pixel_ae`/s1 and `frozen_ssl`/s0 — still reproduce **exactly**. Every cell's `rssm_position(45)` is identical to ten printed decimals.

**The code did not move.** `ca3e140`, the commit the M3c records were written at, and this branch's `f24c2f3` compute **bit-identical** `rssm_position` on macOS 27.0 in 9/9 cells, and bit-identical 500-step training losses on `pixel_ae`/s0. Nothing between M3c and M3h changed either computation — which also verifies, by a cleaner experiment than the one designed for it, that the τ = 1.0 sampling edit is the no-op §2.2 claims.

**Within the new build the computation is deterministic.** Two forward passes in one process agree at exactly 0.0; two 500-step training runs of `random_vit`/s0 — the most divergence-prone arm — agree at exactly 0.0.

**A wrong device is still 6–12 map units away**, the discrepancy these gates exist to refuse, about thirteen orders of magnitude above the platform band. The whole suite (1714 tests) is green under macOS 27.0, so no test compared against a shipped record bitwise.

So `max |delta| == 0.0` no longer separates a code regression from the Metal kernels it happens to run on. **The rule for a stored artefact becomes `max |delta| <= REPRODUCTION_ULPS * ulp(m)`**, where `m` is the stored value's magnitude at the disagreeing step and `REPRODUCTION_ULPS = 64` — about ten times the measured worst case, and far tighter than the `~N * ulp` a reduction over hundreds of windows could produce. The measured delta **and the magnitude that scales it** are written into every record, so `trustworthy` recomputes the bound instead of trusting a recorded verdict, and a record claiming a nonzero delta without its scale is refused. A delta above the bound means re-characterise the platform, not raise the bound. `mbfps.eval.reproduction` holds the rule; it replaces the exact test in `trust_horizon.prepare_cell` (14), `trust_horizon.SelfCheck` (30) and `trust_horizon.trustworthy`.

**Two places keep the exact rule, and the distinction is the point.**

*Inside one run.* `diagnose_dynamics`'s `open_loop` and `stream` compare two code paths in a single process, where determinism is measured, so nothing is granted. (`diagnose_dynamics`'s own comparison against the stored record does need the bound; it is not reached by `sharper_latent.py` and is recorded as a follow-up rather than changed here.)

*Training.* A retrain does **not** reproduce a record written on the other build, by a margin no ULP band covers — measured over 500 steps at τ = 1.0:

| cell | max \|Δ\| on the per-step loss | at step | magnitude there | ULPs | first differs |
|---|---|---|---|---|---|
| `pixel_ae`/s0 | 3.021955e-05 | 465 | 0.427468 | 5.4e+11 | 18 |
| `frozen_ssl`/s0 | 3.874302e-07 | 426 | 0.343362 | 7.0e+09 | 8 |
| `random_vit`/s0 | 6.833319e-01 | 492 | 0.595513 | 6.2e+15 | 3 |

The mechanism is discreteness, not rounding: the straight-through categorical sampler draws from distributions whose top-1 mass is 0.51–0.57 (§1), so a last-bit change in `probs` flips **which class is drawn**, and a flipped one-hot moves the loss by O(0.1–1) rather than O(1e-16). Training feeds that back through the gradients; a forward pass has no feedback, which is why the two bands differ by eleven orders of magnitude. M3f's bitwise-training measurement was correct and is a property of a fixed platform, not a portable one.

**What this costs §2.2.** The identity check (37) and `checkpoint_ladder`'s hard anchor therefore keep `!= 0.0` and will correctly refuse on this box — a refusal is the right answer, not a tolerance to widen. The consequence is that **the shipped M3c cells can no longer serve as Phase 2's control arm on this machine**: τ\* cells trained under macOS 27.0 against control cells trained under macOS 26.x would confound temperature with platform, which is exactly the confound the identity check was written to prevent. Phase 2 remains *possible* — training is reproducible here — but only by retraining the τ = 1.0 control on this build too, nine more cells, taking Phase 2 from ~15 h 30 m to ~31 h. **That decision is deferred and is not taken by this amendment:** Phase 1 is untouched by any of it, and if Reading N returns `NOT_NOISE_LIMITED` or `SHARPER_WORSE` then Phase 2 never runs and the question does not arise.

## 3. How it decides

Stated here, before either phase runs.

### 3.1 Pooling

Both readings use the machinery M3d–M3g use: the per-window probe-free crossing survival indicator `1[h× > h]` over the windows that moved within the horizon, `pooling.paired_contrast` with seeds averaged per window and the standard error clustered over the 24 validation episodes, and `pooling.cluster_threshold(family, clusters)` for the bar. `DECISION_H = 15`, the governing spec's imagination horizon, as M3e, M3f and M3g pre-registered it; h = 5 and h = 45 are printed and not decided on.

### 3.2 Reading N — is the rollout noise-limited? (Phase 1)

Per arm and per candidate τ, the paired contrast on the same 229 windows

`N(τ) = S_free(15)[τ] − S_free(15)[1.0]`,

positive when sampling the prior more sharply keeps the rollout ahead of embedding-space persistence for 15 imagined steps on more of the moved draws. `TAU_CANDIDATES = (0.7, 0.5, 0.3)`; **τ = 0.0 is reported as an endpoint and is not a candidate** — M3a measured argmax collapsing the trajectory to 3 distinct latents in 45 steps at roughly four times the position error, and a τ = 0 sampler is not something to train toward. Family = 3 arms × 3 candidates = 9, `z_fam = cluster_threshold(9, 24) = 3.06`.

| status | rule | what follows |
|---|---|---|
| `NOISE_LIMITED` | some candidate τ has `N(τ) > 0` clearing `z_fam` pooled **and** in ≥ `SEEDS_REQUIRED = 2` of 3 seeds, in **at least two of the three arms** | Phase 2 runs at `τ*` = the τ with the largest pooled contrast among those clearing, ties to the **larger** τ (the milder intervention) |
| `SHARPER_WORSE` | no candidate clears positively in two arms, and some candidate's contrast clears *negatively* in two or more arms | Phase 2 does not run; the sampling noise is load-bearing, which is a finding about the representation |
| `NOT_NOISE_LIMITED` | neither of the above — including a τ that clears positively in one arm only, which the sentence names | Phase 2 does not run. The entropy hypothesis is refuted on the shipped models at this grid; M3h closes with the sweep as its result and the next study is the shrink or KL-balance option, with this evidence |

The gate (`gap_closed(45) > 0` in every seed) is computed at every τ and **reported, not decided on**; an arm × τ that passes is printed as `GATE PASSES at tau=…` and stands as the gate's own reading beside Reading N. The retrain is gated in code, not by a human reading a table: the `train` phase refuses (exit 35) unless the sweep record reads `NOISE_LIMITED`, and takes τ* from that record.

### 3.3 Reading M — does training it sharper help? (Phase 2)

Per arm, the paired contrast on the same 229 windows

`M = S_free(15)[M3h at τ*] − S_free(15)[M3c at τ = 1]`,

with its probe twin as the control. Family = 3 arms × 2 channels = 6, `z_fam = cluster_threshold(6, 24) = 2.89`. Statuses in precedence, each printed with the rule that produced it:

| status | rule |
|---|---|
| `UNRESOLVED_PROBE` | the free and probe contrasts both clear with **opposite** signs; nothing below is read for that arm |
| `SHARPER_BETTER` | `M` clears `z_fam` positively pooled **and** in ≥ 2 of 3 seeds |
| `SHARPER_WORSE` | `M` clears `z_fam` negatively pooled **and** in ≥ 2 of 3 seeds |
| `NO_DIFFERENCE` | none of the above, including a pooled clear that does not replicate, which the sentence names |

A single-seed leaf has no replication clause and reads "this seed alone", as M3e's correction requires. **The M3 gate is reported per cell and per arm and is not decided on**; if it passes, that is the gate's own reading, printed as `GATE PASSES`.

### 3.4 Companions — change no verdict

- The KL regime of the new cells (`kl_dyn`, `kl_rate_above_free_bits`) beside M3c's: a sharper sampler changes the KL the 0.20 floor was calibrated against, and that confound is measured and reported rather than assumed away.
- The validation objective (`val_objective`) of the new cells beside M3c's.
- The two-draw noise reference at each horizon beside the imagined displacement and the truth's — the statistic that motivated the study, measured on the new models and at every τ of the sweep.
- `S(h)`, `S_c(h)`, `u(h)` and `H*_{0.5,0.75,0.9}` on both channels, per arm, at every τ and for the retrain.
- The M3g stage decomposition re-run on the new cells (`scripts/stage_decomposition.py`, about a minute a cell): does the stage that failed move? A table with no status attached — M3g's reading was pre-registered on the M3c cells.

## 4. What it does not claim

- It does not change the M3 gate, `aggregate.py`, `report_study.py`, any shipped checkpoint, or any verdict M3b–M3g recorded.
- *Added 2026-09-24:* the reproduction bound of §2.4 does not revisit any earlier reading. M3b–M3g were measured and validated under the previous macOS build and nothing rewrites their records; the bound governs runs made from now on. It is also not a claim that the two builds compute the same thing — they demonstrably do not — only that the difference is 6 ULPs where a wrong device is 6–12 map units, and that no reported digit moves.
- It does not rank arms. Every contrast is within an arm — τ against τ = 1, or the retrain against its own M3c cell.
- `τ*` is the best of a three-value grid read at one horizon on the free channel. It is not an optimum, and the sweep does not say what a *trained* sharper model's best sampling temperature would be.
- The temperature changes the KL regime the free-bits floor was calibrated against. `SHARPER_BETTER` therefore does not attribute the gain to entropy alone; the KL companion is printed so a reader can see how far the regime moved.
- `NOT_NOISE_LIMITED` refutes the hypothesis on the shipped models at this grid. It does not say a model trained sharper would fail — it says the evidence does not justify fifteen hours of training on that premise.
- `SHARPER_WORSE` in either reading says the sharper model keeps ahead of persistence on fewer draws. It does not say why, and it does not distinguish "the noise was doing work" from "this model, trained at this temperature, is worse for another reason".
- A gate pass, if it comes, is a pass for these nine cells on `my_way_home`'s 24 validation episodes at context 5 / horizon 45, at 20,000 steps, at one temperature — not a statement that the world model works.

## 5. Shape of the code

Five units; two extend what exists, three are new.

**`src/mbfps/models/rssm.py`** — `SAMPLE_TEMPERATURE = 1.0` (the shipped default, and the value every M3b–M3g artefact was trained and evaluated at); `RSSMConfig.sample_temperature: float = SAMPLE_TEMPERATURE`, threaded from `TrainConfig.sample_temperature` so `get_config(arm, sample_temperature=τ)` reaches it through the existing override path; `_sample(logits, temperature=None)` divides by the temperature **only when it is not 1.0**, so the default path is bitwise unchanged by construction; `imagine(actions, state, temperature=None)` takes a rollout override (None = the model's own), `observe` always uses the model's own. Pinned: at 1.0 every sampled latent is bitwise what today's code produces on the same stream; at τ < 1 the sampled distribution is measurably sharper (the sampled class equals the argmax more often); a temperature of 0 takes the argmax; `imagine`'s override does not change `observe`; the config default is 1.0 and an M3c config is unchanged.

**`src/mbfps/training/world_model.py`** — `_save_checkpoint`'s payload gains `sample_temperature` (always written, from `cfg.train.sample_temperature`); `train_world_model` is otherwise unchanged. Pinned: the payload's keys; a temperature of 1.0 trains bitwise what M3c trained (the identity check's premise, asserted on a tiny CPU config).

**`scripts/diagnose_dynamics.py`** — `load_checkpoint_model` reads the payload's `sample_temperature` (absent = 1.0), rebuilds the configuration with it, and refuses a mismatch against the temperature asked for (exit 36 in the new script; the existing scripts pass no temperature and therefore accept only 1.0 checkpoints, which is every artefact they read today). Pinned: an M3c checkpoint loads unchanged; a τ* checkpoint loaded without asking for τ* is refused by name.

**`src/mbfps/eval/reproduction.py`** *(added 2026-09-24)* — `REPRODUCTION_ULPS = 64`, `reproduction_bound(magnitude, *, ulps)` and `reproduces(delta, magnitude, *, ulps)`: §2.4's rule, in one place, so the three gates that judge a stored artefact cannot drift apart. Pinned: the bound at three magnitudes; that it scales with the magnitude and not the delta; that the measured 6-ULP worst case passes and a 6–12 map-unit device miss does not; that a zero or non-finite magnitude and `ulps=0` all demand the exact rule; and that a non-finite delta never reproduces.

**`src/mbfps/eval/diagnostics.py`** — `rollout_temperature: float | None = None` on `_diagnose` and `reference_trajectories`, used for the canonical `imagine` and for the noise reference and nowhere else. *Corrected while implementing (2026-09-23): the default is None — the model's own `cfg.sample_temperature` — and the keyword is then not passed to `imagine` at all. A default of 1.0 would have rolled a retrained cell out at 1.0 during Phase 2's evaluation, evaluating it with a sampler its training never saw; that is the mis-evaluation §2.3's payload check exists to catch, and it would have reached it through the one path that does not go through the payload.* Pinned: with no override every field of the pass is bitwise what it is today, the generator ends at the same state and no temperature keyword reaches `imagine`; at an override below 1 the canonical rollout differs and the floor, the persistence anchor and the probe do not; a model whose own temperature is 0 rolls out deterministically when nothing is asked for.

**`src/mbfps/eval/sharper.py`** (new) — pure over floats, dicts and arrays; no torch, no files: `TAU_GRID`, `TAU_CANDIDATES`, `REFERENCE_TAU = 1.0`, `STEPS = 20000`, `IDENTITY_STEPS = 500`, `DECISION_H` (imported from `split_gap`), `SWEEP_FAMILY = 9`, `RETRAIN_FAMILY = 6`, `SEEDS_REQUIRED = 2`; `tau_star(per_tau_contrasts, per_seed, z_fam) -> float | None` (§3.2's rule — the largest pooled contrast among the candidates clearing pooled and in ≥ 2 seeds in ≥ 2 arms, ties to the larger τ; None when the status is not `NOISE_LIMITED`); `SweepStatus`, `RetrainStatus`, their input and reading dataclasses, `reading_noise`, `reading_sharper`, and the two formatters through `split_gap.fmt_z`. Pinned on fabricated contrasts, one rule mutated at a time: two arms clearing at one τ → `NOISE_LIMITED` with that τ; one arm only → `NOT_NOISE_LIMITED` naming the arm; two arms clearing negatively → `SHARPER_WORSE`; a tie between two clearing τ → the larger; a pooled clear replicating in one seed → not counted; z exactly at the bar does not clear; NaN never clears; the four retrain statuses and the single-seed leaf.

**`scripts/sharper_latent.py`** (new) — loads `trust_horizon`, `split_gap` and `stage_decomposition` by path as the others load their siblings. `--phase sweep|train|evaluate|read|all`; `--out runs/m3h_sharper`, `--sweep-out runs/m3h_sweep`, `--reference runs/m3_study_v2`, `--data`, `--device`, `--context`, `--horizon`, `--arms`, `--seeds`, `--steps`, `--figure`. `--out` equal to `--reference` or to `--sweep-out` is argparse's 2.
- `sweep`: per cell, `prepare_cell`, the τ = 1.0 pass and its self-check (30) before any other τ, then each τ's pass, its trust reduction, its gate; one `sweep_<arm>_seed<n>.json`.
- `train`: the sweep records (11), Reading N, **exit 35 unless `NOISE_LIMITED`**; the identity check (37); then nine retrains at τ* through `train_world_model`, saving into `--out` in the study's layout.
- `evaluate`: the study's own evaluation half (`evaluate_job`) per cell at τ*, then the trust pass over a rung-style cell (the checkpoint and its study record, the protocol in the diagnostic's slot) exactly as M3f evaluated its rungs — a retrained cell has no prior diagnostic to self-check against, and neither reading consumes the diagnostic's intervention ladder. The temperature is asked for and checked (36).
- `read`: both readings, the tables (the sweep by τ; the retrain against M3c; the gate; survival and conditional survival; the KL regime; the objective; the noise reference; the stage companion), `sweep.txt` / `sharper.txt` and the two figures.
- Exit 0 when every phase asked for ran and the reading printed; 11 / 12 / 14 / 30 with their existing meanings; **35 `EXIT_NOT_NOISE_LIMITED`**, **36 `EXIT_TEMPERATURE_MISMATCH`**, **37 `EXIT_IDENTITY_CHECK_FAILED`** new, in no other tool's range (run_study 1/3–6/23, report_study 7–10, spike 10, diagnose 11–17, pool 18–22, trust 30, split_gap 31, ladder 32–33, stages 34, argparse 2, a traceback 1).

**Tests.** `tests/models/test_rssm.py`, `tests/training/test_world_model.py`, `tests/eval/test_diagnostics.py`, `tests/eval/test_diagnose_dynamics_script.py` for the four extensions and the status registry; `tests/eval/test_sharper.py` for the pure rules; `tests/eval/test_sharper_latent_script.py` on the ladder tests' fixtures (a tiny reference study on `wide_buffer`, `steps=5, seq_len=4, context=2, horizon=3, device="cpu"`, a two-value τ grid): every exit code, the gate between the phases, the identity check, the records, `sweep.txt` and `sharper.txt`.

**Record.** Results go in the plan written from this spec, in its own results section, numbers copied not rounded, the closing paragraph stated against §4. `NOT_NOISE_LIMITED` — Phase 2 never running — is a complete result for this milestone. The M3c–M3g plans are closed records and are not amended.

## 6. The run

- **Smoke:** one cell (`pixel_ae`/s0), the full τ grid plus the identity check, into `runs/m3h_smoke` — a directory the real run never reads — about ten minutes; its `sweep.txt` read for format.
- **Phase 1:** nine cells × five τ, about a minute a pass: **~1 h**. Its record decides, and the `train` phase reads that record rather than a person.
- **Phase 2, if `NOISE_LIMITED`:** nine retrains at 20,000 steps (~1.6 h a cell) plus evaluation and diagnostics: **~15 h 30 m**, under `caffeinate -dimsu` and `nohup` with `PYTHONUNBUFFERED=1`, `.head` / `.started` / `.finished` / `.exit` beside the log. ~350 MB of checkpoints. Phases are resumable: an evaluation failure re-runs on the saved checkpoints without retraining.
- **Acceptance:** the τ = 1.0 self-check within §2.4's reproduction bound on 9/9 cells (measured 0.0 on two cells and at most 6 ULPs on the rest); 229 windows over 24 clusters on every τ of every cell; the identity check exact over its 500 steps — which on macOS 27.0 it cannot be, see §2.4; and, if Phase 2 runs, nine records at one `git_sha` == HEAD on `mps`, every checkpoint carrying `sample_temperature = τ*`, and the τ* the sweep record named.

## 7. Files

- `src/mbfps/models/rssm.py`, `src/mbfps/training/world_model.py`, `src/mbfps/eval/diagnostics.py`, `scripts/diagnose_dynamics.py` — modified (§5).
- `src/mbfps/eval/sharper.py`, `scripts/sharper_latent.py` — new (§5).
- `tests/models/test_rssm.py`, `tests/training/test_world_model.py`, `tests/eval/test_diagnostics.py`, `tests/eval/test_diagnose_dynamics_script.py` — modified; `tests/eval/test_sharper.py`, `tests/eval/test_sharper_latent_script.py` — new.
- `docs/superpowers/specs/2026-09-23-mb-fps-m3h-sharper-latent-design.md` — this document.
- `docs/superpowers/plans/2026-09-23-mb-fps-m3h-sharper-latent.md` — the plan, with its results section.
