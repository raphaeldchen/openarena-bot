# MB-FPS M3g — The stage decomposition: which stage of imagination fails? (Design)

**Status:** approved in conversation 2026-09-19 (decompose before intervening; a "first failing stage" reading; the stages read in the latent with the posterior as the truth; a known-answer control that refuses; approach A of three; sections 1–3 approved in turn). Separate diagnostic on the shipped checkpoints; nothing retrained; the M3 gate and every recorded verdict untouched.
**Reads:** `runs/m3_study_v2` — the nine M3c cells (checkpoints, study records, diagnostics; one code state `ca3e140`, device `mps`); `runs/m3f_ladder/step4000` — the two control cells (`frozen_ssl` seeds 1 and 2: checkpoints and rung records); `data/my_way_home`.
**Writes:** `runs/m3g_stages/` — `stages_<arm>_seed<n>.json` for nine cells and two controls, `stages.txt`, `stages_curves.png`; one results section in this diagnostic's own plan.
**Base:** `main` at `28bdabc` (PR #5, M3f, merged); branch `feat/m3g-stage-decomposition`.

## 1. Why

Two hypotheses for the M3 gate failure are closed by pre-registered readings. M3e: the models are no better on the episodes they trained on than on held-out ones (`NO_GAP` on all three arms) — not memorisation. M3f: evaluated at its own one-step-loss minimum, each ViT arm rolls out *worse* than at step 20,000 (`EARLIER_WORSE`, z −6.88 and −5.30), and the control shows no difference — not checkpoint timing. Both were built around the absolute `embedding` loss, and the design session for this study measured that quantity to be confounded by the embedding's own scale: the loss's target is `embeddings.detach()`, the trainable bottleneck's own output with no normalisation, so the encoder sets the scale of what is reconstructed. Over the shipped checkpoints and the M3f rungs (a scratch measurement over the same 229 validation windows; not a recorded result):

- `pixel_ae` s1's loss falls 0.230 → 0.086 from rung 1000 to 20,000 while the embedding's per-dimension std falls 0.52 → 0.32 and the fraction of its variance the posterior reconstructs (R²) stays at 0.15. `frozen_ssl` s1's loss *rises* after rung 4000 while its std grows 0.40 → 0.55 and R² goes from −0.01 to 0.28. In relative terms the ViT arms' reconstruction improves with training; M3f's verdict stands and its stated reason does not.
- `frozen_ssl` s1 and s2 are blind through rung 4000: R² ≈ 0 (the head predicts the mean embedding), the posterior equal to the prior (val `kl_dyn` 0.03–0.07, below the 0.20 free-bits floor). At 20,000 the posterior explains 15–28 % of the embedding variance in `pixel_ae` and `frozen_ssl`, and a wrong frame from the same window explains most of that.
- The evaluation channels are noise-bound at short horizons. The encoder's embedding moves 26 units in one frame and 38 in forty-five (`frozen_ssl` s0 medians; every arm the same shape): the space is dominated by per-frame variance no dynamics model can predict, so "ahead of persistence" in the free channel is a coin flip at step 1 (`S_c(1)` 0.53–0.56 in every arm at 20k) even for a model with the dynamics right. In position, the M3d records show the imagined displacement's direction uncorrelated with the truth (cos ≈ 0) at h = 1, 5, 15 and 45 alike, and the probe's residual error is of the order of the 45-step displacement. The imagination, meanwhile, moves 4–10× less in embedding space than the truth does.

So what is not known is *which stage* of imagination fails at 20k: whether the posterior carries the frame (**encode**), whether one prior step from the true state predicts the next posterior (**predict**), whether that survives fifteen open-loop steps on the prior's own samples (**carry**), or whether a latent that tracks is lost in the rendering the gate scores (**decode**). M3g measures the four in the model's own currency — the 32 × 32 categorical latent, where the posterior on the validation windows is exactly known and neither the jitter nor the probe limits the measurement — and names the first stage that fails per arm. Each status commits the next study, M3h, to one intervention, written in §3.4 before the run.

## 2. What it measures

### 2.1 Cells and controls

- **Cells.** The nine M3c cells (`pixel_ae`, `frozen_ssl`, `random_vit` × seeds 0, 1, 2) in `runs/m3_study_v2`: the 20,000-step checkpoint, the study record and the diagnostic of each, loaded through `trust_horizon.load_cell` and `prepare_cell` exactly as M3e and M3f load them — exit 12 for a missing artefact, 14 for a protocol mismatch or a rollout that no longer reproduces the record's `rssm_position` curve. The 229 validation windows over 24 episode-clusters at context 5 / horizon 45 are the trust pass's own; every quantity below is over them.
- **Controls.** `frozen_ssl` seeds 1 and 2 at rung 4000 of `runs/m3f_ladder`, loaded through `checkpoint_ladder.rung_cell` (a rung directory is a study directory: the rung's checkpoint and its `result_<arm>_seed<n>.json`) and the same `prepare_cell`, which reproduces the rung record's curve (14). Their known state is §1's: blind, posterior equal to prior. `CONTROL_ARM = "frozen_ssl"`, `CONTROL_SEEDS = (1, 2)`, `--control runs/m3f_ladder/step4000`; a control is a fourth arm to the pooling and to nothing else.

### 2.2 The pass

`diagnostics._diagnose` gains `keep_latents: bool = False` beside `keep_trajectories`, and `reference_trajectories` passes it through. With it False the pass is byte-identical to today's. With it True, on the canonical pass only — the context filter, the `imagine` over the real horizon actions, the floor's `observe` over the real future frames from the context state, as the pass already runs them — `Trajectories` additionally carries per window:

| field | shape | what |
|---|---|---|
| `post_logits` | `(n, context + horizon, 32, 32)` | the posterior over the whole window: the context filter's five steps then the floor's forty-five. Index `context − 1` is t₀, the last context frame; index `context − 1 + h` is horizon step h. |
| `prior_teacher_logits` | `(n, horizon, 32, 32)` | the floor pass's `prior_logits`: at horizon step h, `prior_net` on h advanced by the *posterior's* sample at h − 1 and the true action — the teacher-forced one-step prediction, which `observe` computes already. |
| `prior_open_logits` | `(n, horizon, 32, 32)` | the canonical `imagine`'s `prior_logits`: the prior fed its own samples from t₀. |
| `posterior_rendering_distance` | `(n, horizon)` | ‖ê_post(h) − e(h)‖: the head on the floor's latent against the encoder's frame (companion). |
| `true_step_displacement` | `(n, horizon)` | ‖e(h) − e(h − 1)‖, with e(0) the last context frame: the one-frame jitter (companion). |

The five live in their own `_LATENT_FIELDS` tuple; `_TRAJECTORY_FIELDS` and the tests that pin it stand. Nothing else in the pass changes, and nothing new is drawn from the stream: the imagination the stages read is the rollout the gate scored. Roughly 131 MB of float32 logits per cell live in memory and are never written.

### 2.3 The statistics

Pure numpy in `mbfps/eval/stages.py`, float64. `mode` is argmax over the 32 classes; h runs 1..45; g over the 32 groups; w over windows.

- **Encode.** `information[w]` = mean over h of Σ_g KL(post_{h,g} ‖ prior_teacher_{h,g}), each KL from log-softmax — the per-element form of `rssm._categorical_kl` before its batch mean: the information the frame injects beyond what the prior predicted. **Contrast:** `information − KL_FREE_BITS`, the floor (0.20 nats) below which the training objective itself treats the posterior as carrying nothing worth optimising.
- **Predict.** `accuracy_teacher[w]` = mean over (h, g) of 1[mode(prior_teacher_h)_g = mode(post_h)_g]. `accuracy_persistence[w]` = mean of 1[mode(post_{h−1})_g = mode(post_h)_g], with post_0 the posterior at t₀ — latent persistence, what a model that predicts "nothing changes" scores exactly. `accuracy_marginal[w]` = mean of 1[m_g = mode(post_h)_g], where m_g is the most frequent posterior mode of group g over that cell's own (window, h) pairs (in-sample by design — it makes the baseline stronger, not weaker; ties to the smallest class index). **Contrasts:** `teacher − persistence` and `teacher − marginal`.
- **Carry.** `open_accuracy[w, k]` = mean over g of 1[mode(prior_open_k)_g = mode(post_k)_g], k = 1..45. `open_persistence[w, k]` = mean of 1[mode(post_0)_g = mode(post_k)_g]; `open_marginal[w, k]` likewise with m_g. **Contrast:** `open_accuracy[:, 15] − open_persistence[:, 15]` at `DECISION_H = 15` (M3e/M3f's horizon, the governing spec's imagination horizon). The full k-curve and `open − marginal` are companions.
- **Decode.** `margin15[w]` = `embedding_persistence_distance[w, 14] − embedding_distance_to_truth[w, 14]` — ‖ê(0) − e(15)‖ − ‖ê(15) − e(15)‖, both already kept by the trust pass — on the windows the truth has moved at h = 15 (`trust.moved_mask`; `u(15)` = 0.03 at 20k, about seven of the 229 windows excluded). **Contrast:** the margin itself against zero.

Companions per cell: mean posterior entropy per group `(32,)` against ln 32 = 3.466; the marginal classes `(32,)`; the prior's mean negative log-probability of the posterior's mode, teacher-forced; the median rendering distance and the median one-frame jitter per h; the probe-channel margin at h = 15 and 45, ‖p̂(0) − p(h)‖ − ‖p̂(h) − p(h)‖ from the kept positions.

### 2.4 Self-checks

- **Reproduction (14)** inside `prepare_cell`: the pass's `rssm_position` curve equals the cell's study record's at max |Δ| == 0.0. On cells and controls alike.
- **Self-check (30)** on the nine cells: `trust_horizon.self_check` against the cell's diagnostic — the same windows, the same rollout, the same refit probe — exactly as M3d–M3f.
- **The step-1 identity (30)**, on cells and controls: `prior_open_logits[:, 0]` equals `prior_teacher_logits[:, 0]` bitwise. Both are `prior_net` on the same h after the same context state and the same action; a cell where they differ has a plumbing fault, and no record is written for it.

## 3. How it decides

Stated here, before the run.

### 3.1 Pooling

- **Series.** Each contrast is a per-window series with every window changed (the latent changes every step) except decode's, whose `changed` is the moved mask at 15.
- **Pooled and per seed.** `pooling.pool_arm` over an arm's three cells: seeds averaged per window, the mean, the episode-clustered SE over 24 clusters, z. `pool_arm([cell])` is each seed's leaf.
- **Family.** `FAMILY = 5`, the five contrasts one arm's reading may consult; **`z_fam = pooling.cluster_threshold(5, 24) = 2.81`**. Not 15: each arm's reading is its own fixed-sequence procedure — the stages are tested in a pre-registered order and reading stops at the first not passed — and claims nothing across arms. The control is a fourth arm under the same bar with its two seeds.

### 3.2 Rules and statuses

A stage **passes** when its pooled estimate is > 0 with z > `z_fam` and its per-seed z clears in at least `SEEDS_REQUIRED = 2` of the arm's seeds; *predict* passes only when both its contrasts do, and a seed holds predict only when both clear within it. `clears` is strict and never true on NaN or ±inf. The arm's status is the first stage in the order that does not pass:

| status | rule | read |
|---|---|---|
| `ENCODE_FAILS` | encode does not pass | the frame does not change the belief beyond the objective's own floor |
| `PREDICT_FAILS` | encode passes; predict does not | the posterior knows the frame and one prior step from the true state does not predict the next |
| `CARRY_FAILS` | encode and predict pass; carry does not | one step is right and it does not survive fifteen on the prior's own samples |
| `DECODE_FAILS` | the three latent stages pass; decode does not | the latent still tracks at 15 and the rendering the gate scores loses to persistence |
| `NO_STAGE_FAILS` | all four pass | every stage passes at h = 15 while the gate fails at h = 45 |

Every status carries the sentence that decided it: the contrast and its z; **`failed`** when the estimate is ≤ 0 or z < −`z_fam`, **`not shown`** when |z| is below the bar; the seeds that held; predict's sentence names the first of its two contrasts that does not clear, persistence before marginal — e.g. `predict not shown: teacher − marginal z +1.20 does not clear +2.81 (teacher − persistence z +4.10 clears; seeds holding 2 of 3)`. The reading names the first stage not passed; the sentence keeps absence of evidence apart from evidence of absence. Per-seed statuses are printed under the single-seed rule of M3e/M3f — a seed's own leaves, "this seed alone", no replication clause — for information; the arm's status is the pooled one.

### 3.3 The known-answer control

The control arm (`frozen_ssl` × seeds 1, 2 at rung 4000) is read by the identical rule and must return `ENCODE_FAILS`. Any other status is **exit 34 `EXIT_CONTROL_MISREAD`**: the control table is printed with the sentence that decided it, no arm's reading is printed, and `stages.txt` is not written. A control that reads `ENCODE_FAILS` is expected, not assumed — §1's measurement is a scratch number and the instrument is what is being validated. The control is reported in its own table and never pooled into any arm.

### 3.4 What each status commits M3h to

| status | M3h |
|---|---|
| `ENCODE_FAILS` | the objective's target: a fixed-scale embedding target (LayerNorm on the bottleneck output, or a frozen / EMA target) and DreamerV3's 1 % unimix on the categoricals — the posterior is not being taught to use the frame |
| `PREDICT_FAILS` | the prior's signal: the dyn-KL weight and the free-bits floor (per element, recalibrated), the prior's action conditioning — the posterior knows the frame and the prior does not learn its dynamics |
| `CARRY_FAILS` | multi-step consistency (latent overshooting) and the sampling at imagination — one step is right and it compounds |
| `DECODE_FAILS` | the evaluation space and the head: a temporally denoised target, a stronger probe, or a latent-space gate — the model tracks and the ruler cannot see it |
| `NO_STAGE_FAILS` | the horizon: carry at 45 |

Three arms may read three statuses; M3h is brainstormed from the table, not from the run.

### 3.5 Companions — change no verdict

- The carry k-curve at k = 1, 2, 3, 5, 10, 15, 30, 45 with both baselines, per arm; `open − marginal` at 15 with its z.
- The decode table's probe-channel margins at h = 15 and 45 with their z.
- Per arm: the number of groups whose mean posterior entropy is below ½ ln 32; the prior's NLL of the posterior mode; the median rendering distance beside the median one-frame jitter at h = 1, 5, 15, 45.
- The control's every table, beside the arms'.

## 4. What it does not claim

- It does not change the M3 gate, `report_study.py`, `aggregate.py`, any checkpoint, or any verdict M3b–M3f recorded. No checkpoint is retrained.
- It does not rank arms. Every contrast is within an arm against a baseline in the same space; between-arm differences are printed and not tested.
- A stage *passing* at h = 15 in the latent says nothing about h = 45, and nothing about position: the gate's own reading stands beside this one.
- A status says which stage is the first not passed. It does not say *why* — §3.4 names the intervention M3h tests, not the mechanism.
- `not shown` is not `failed`: a stage whose contrast does not clear the bar in either direction reads as not passed, and the sentence says so.
- The marginal baseline is in-sample and therefore the stronger of the two; a prior that clears it clears a bar the marginal itself could not have set higher.
- Everything is `my_way_home`'s 24 validation episodes at context 5 / horizon 45, the M3c seeds and configuration, the shipped 20,000-step checkpoints and two rung-4000 checkpoints of one M3f run. A different horizon, environment or split is a different measurement.

## 5. Shape of the code

Four units, one job each; one extends what exists, three are new.

**`src/mbfps/eval/diagnostics.py`** — `keep_latents: bool = False` on `_diagnose` and `reference_trajectories`; the five fields of §2.2 on `Trajectories`, listed in a new `_LATENT_FIELDS` tuple and None unless kept; the step-1 identity asserted where the fields are kept, as a `ValueError` naming the window. Pinned: with the flag False every existing field is bitwise what it was; with it True every existing field is still bitwise the same and the new ones have §2.2's shapes; the identity holds on the fixture model; `post_logits[:, :context]` equals the context filter's own `post_logits`.

**`src/mbfps/eval/stages.py`** (new) — pure over arrays and floats; no torch, no files:
- `DECISION_H = 15` (imported from `split_gap`), `FAMILY = 5`, `SEEDS_REQUIRED = 2`, `REPORTED_K = (1, 2, 3, 5, 10, 15, 30, 45)`, `KL_FREE_BITS` (imported from `rssm`), `STAGES = ("encode", "predict", "carry", "decode")`.
- `categorical_kl(logits_q, logits_p) -> (…)` summed over groups; `mode(logits) -> (…, 32)`; `entropy_by_group(post_logits) -> (32,)`.
- `information(post_logits, prior_teacher_logits, context) -> (n,)`; `teacher_accuracy(post_logits, prior_teacher_logits, context) -> (n,)`; `persistence_accuracy(post_logits, context) -> (n,)`; `marginal_classes(post_logits, context) -> (32,)`, `marginal_accuracy(post_logits, classes, context) -> (n,)`; `open_accuracy(post_logits, prior_open_logits, context) -> (n, horizon)`, `open_persistence(post_logits, context) -> (n, horizon)`, `open_marginal(post_logits, classes, context) -> (n, horizon)`; `teacher_nll(post_logits, prior_teacher_logits, context) -> (n,)`; `decode_margin(persistence_distance, distance_to_truth, h) -> (n,)`.
- `Status`, `StageResult` (estimate, se, z, per-seed z, `passes`, `wording` ∈ {`passes`, `failed`, `not shown`}), `ArmInputs` (per stage its pooled contrast(s) and per-seed leaves), `StagesInputs` (`arms`, `z_fam`, `h`), `ArmReading` (`status`, `reason`, `stages`), `StagesReading`, `clears(z, bar)`, `reading_stages(inputs) -> StagesReading` — §3.2's rules in order, the reason sentence built from the first stage not passed; `format_reading_stages` through `split_gap.fmt_z`.
- Pinned on fabricated logits with hand-typed answers, one rule mutated at a time: a *blind* prior (posterior equal to prior, uniform) → information 0, every accuracy at chance, `ENCODE_FAILS`; a *perfect teacher* (prior mode equal to the next posterior mode) → `teacher_accuracy` 1 while persistence and marginal are below it; a *persistence clone* (prior mode equal to the current posterior mode) → `teacher − persistence` exactly 0; a *marginal clone* → `teacher − marginal` exactly 0; an *open-loop prior that decays* (right at k = 1, chance at k = 15) → `PREDICT` passes and `CARRY_FAILS`; a fabricated margin → `DECODE_FAILS` and `NO_STAGE_FAILS`; +4 z pooled in 1 of 3 seeds → not passed with "seeds holding 1 of 3" in the sentence; z exactly at the bar does not clear; NaN never clears; on logits whose `prior_open_logits[:, 0]` equals `prior_teacher_logits[:, 0]`, `open_accuracy[:, 0]` equals the teacher accuracy at h = 1; the entropy of a one-hot posterior is 0 and of a uniform one ln 32.

**`scripts/stage_decomposition.py`** (new) — loads `trust_horizon` and `checkpoint_ladder` by path as the others load their siblings. Flags: `--out` (default `runs/m3g_stages`), `--reference` (default `runs/m3_study_v2`), `--control` (default `runs/m3f_ladder/step4000`), `--data` (default `data/my_way_home`), `--device`, `--context 5`, `--horizon 45`, `--arms`, `--seeds`, `--phase evaluate|read|all`, `--figure` (default `<out>/stages_curves.png`). `--out` equal to `--reference` or to `--control` is argparse's 2 (nothing is trained here, but a record written into a study directory is a study directory changed).
- `evaluate`: per cell — `load_cell` (12), `prepare_cell` (14), `reference_trajectories(keep_latents=True)`, `self_check` (30), the identity (30), the statistics, one `stages_<arm>_seed<n>.json` (provenance: `git_sha`, device, torch, `context`, `horizon`, `split_seed`, the checkpoint directory and its `step`, the record's `git_sha`; the self-check; `windows {total, episode}`; the contrast series and their components; the k-curves; the moved mask at 15; the companions); then each control through `rung_cell`, `prepare_cell` and the identity, written under `stages_control_<arm>_seed<n>.json`. Every cell requested is loaded before any pass runs, M3f's pattern.
- `read`: the nine records and the two controls (11 on a missing one; a record whose self-check is not `ok` refused as 11; records disagreeing on `context`/`horizon`/windows raise), the series, `pool_arm`, `reading_stages`, the control rule (34), and `stages.txt` in `ladder.txt`'s style: the self-check table; one table per stage; the k-curve table; the decode table with the probe margins; the companions; the control tables; the readings with their sentences; the per-seed statuses; the figure — open-loop accuracy against k per arm with both baselines, per-group posterior entropy per arm, `information` per arm with the 0.20 line.
- Exit **0** when every phase asked for ran and the reading printed; **11 / 12 / 14 / 30** with their existing meanings; **34** new. `EXIT_CONTROL_MISREAD = 34` joins the distinctness test.

**Tests.** `tests/eval/test_diagnostics.py`: the `keep_latents` pins. `tests/eval/test_stages.py`: the pure functions and the reading. `tests/eval/test_stage_decomposition_script.py`, on the ladder tests' fixtures (`wide_buffer`, `JOB_KW = dict(steps=5, seq_len=4, context=2, horizon=3, device="cpu")`): a tiny reference study and a tiny control rung; `--phase all` writes one record per fixture cell and per fixture control and a reading line per arm; a doctored `rssm_position` → 14 before any pass; a doctored diagnostic → 30; a control whose record is doctored to read anything but `ENCODE_FAILS` → 34 with no `stages.txt`; a missing record → 11 on `read`; `--out == --reference` → 2. The control refusal is tested by monkeypatching `reading_stages`' inputs for the control, not by training a non-blind rung. A mutation table with the harness self-checked, as every M3 task's.

**Record.** Results go in the plan written from this spec, in its own results section, numbers copied not rounded, the closing paragraph stated against §4. Any status on any arm is a result. The M3c–M3f plans are closed records and are not amended.

## 6. The run

- **Smoke:** `--arms pixel_ae --seeds 0 --phase all` with the control, into `runs/m3g_smoke` — a directory the real run never reads — to see the self-check, the identity and the control path exercised end to end (the control reading `ENCODE_FAILS` at 4000 is expected, not assumed). Its `stages.txt` is read for format. About six minutes.
- **Real run:** nine cells and two controls, `--phase all`, under `caffeinate -dimsu` and `nohup` with `PYTHONUNBUFFERED=1`, `stages.started` / `stages.head` / `stages.finished` / `stages.exit` beside `stages.log`. By M3f's rate about two minutes a pass: under half an hour.
- **Acceptance:** eleven records at one `git_sha` == HEAD (tree clean under `src/` and `scripts/`), device `mps`; the trust self-check exactly 0.0 on 9/9 and the identity on 11/11; 229 windows over 24 clusters on every record; the control reading `ENCODE_FAILS`.

## 7. Files

- `src/mbfps/eval/diagnostics.py` — modified (§2.2, §5).
- `src/mbfps/eval/stages.py` — new (§2.3, §3, §5).
- `scripts/stage_decomposition.py` — new (§5).
- `tests/eval/test_diagnostics.py` — modified; `tests/eval/test_stages.py`, `tests/eval/test_stage_decomposition_script.py` — new; `tests/eval/test_diagnose_dynamics_script.py` — the distinctness test gains `stage_decomposition`.
- `docs/superpowers/specs/2026-09-19-mb-fps-m3g-stage-decomposition-design.md` — this document.
- `docs/superpowers/plans/2026-09-19-mb-fps-m3g-stage-decomposition.md` — the plan, with its results section.
