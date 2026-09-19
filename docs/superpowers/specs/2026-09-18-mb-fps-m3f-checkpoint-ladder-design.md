# MB-FPS M3f — The checkpoint ladder: is the M3 failure an artefact of when the model is evaluated? (Design)

**Status:** approved in conversation 2026-09-18 (scope, cells, schedule, reading and anchor policy chosen by the author; approach A of three; sections 1–3 approved in turn). Separate diagnostic; the M3 gate and every recorded verdict untouched.
**Reads:** `runs/m3_study_v2` — the nine M3c records (their `history.loss` and `history.parts`, one code state `ca3e140`, device `mps`), checkpoints and `diagnostic_<arm>_seed<n>.json`; `data/my_way_home`.
**Writes:** `runs/m3f_ladder/` — five rung directories in the study's layout, `train_<arm>_seed<n>.json`, `ladder_<arm>_seed<n>.json`, `ladder.txt`, `ladder_curves.png`; one results section in this diagnostic's own plan.
**Base:** `main` at `c4946b4` (PR #4, M3e, merged); branch `feat/m3f-checkpoint-ladder`.

## 1. Why

M3e ruled memorisation out: on the 78 training episodes the probe never saw, every cell is below persistence at h=45 exactly as it is on validation (`gap_closed(45)` −0.43 to −0.91; Reading G `NO_GAP` on all three arms). The failure is a fitting failure. M3e's learning curves then said where: the `embedding` term — the one-step prediction loss that rollouts depend on — reaches its minimum at step 1,610–4,535 of 20,000 in every ViT-arm cell and **rises** over the last quarter in five of the six (+10.8 % to +23.0 %), and the summed loss rises in two of them. All three `pixel_ae` cells are still descending at 20k. Every rollout number M3b–M3e recorded was measured on the step-20,000 checkpoint, which for the ViT arms is well past the model's own best on the term that matters.

So the cheapest live hypothesis is not data and not budget: the ViT arms were evaluated at the wrong time. If a model at its own loss minimum rolls out no better, the problem is the objective or the optimisation regime and the next step is a loss-balance study; if it rolls out better, checkpoint selection is a free, arm-neutral fix and the M3 gate is re-read at the right step. Either answer is worth one overnight run, and both come from the same run.

The instruments already exist. Training is seeded with no schedule (plain Adam, fixed learning rate, a seeded loader), so retraining a cell to 5,000 steps follows the original run's first 5,000 steps; `run_job`'s evaluation half writes a study record from any loaded model; `trust_horizon.py`'s pass reads any study directory; M3e's pooling glue turns a pass into per-window survival series. What is new is a checkpoint ladder during training, a rung-versus-rung reading, and one measurement M3e could not make — the training objective on validation episodes — so the training minimum can be checked against the validation minimum.

## 2. What it measures

### 2.1 Cells and rungs

All nine cells (3 arms × 3 seeds), retrained with the study's configuration — `get_config(arm, steps=5000, seq_len=64, seed=<seed>, device=mps)`, the same `episode_split(..., seed=SPLIT_SEED)`, the same `SequenceLoader` seed — to `STEPS = 5000`, saving a checkpoint at each rung of `RUNGS = (1000, 2000, 3000, 4000, 5000)` into `runs/m3f_ladder/step{N}/world_model_<arm>_seed<n>.pt`, labelled `{"arm", "seed", "step", "state_dict"}`. Each `step{N}/` is a study directory: `evaluate_job` writes its `result_<arm>_seed<n>.json` there and every existing instrument can read it.

**Rung 20000 is `runs/m3_study_v2` itself.** It is read through `prepare_cell` and self-checked bitwise against its own diagnostic record (§2.4, exit 30) before any rung of the cell is evaluated — the ruler M3d and M3e validated, unchanged.

`pixel_ae` is the negative control. Its `embedding` minima are at 8,359 / 19,096 / 16,873, beyond every rung, so all of its rungs lie before its minimum and should read no better than 20000. If they do, "earlier is better" is a property of undertrained priors — the slow-drift concern M3d's Reading 1 raised — and not of the loss minimum.

### 2.2 The primary rung

Per cell, the primary rung is the rung nearest the step at which the 100-step moving mean of the cell's `embedding` term reached its minimum in the M3c record — M3e's `learning_curve_summary(history, window=100)["terms"]["embedding"]["smoothed_min_step"]`, recomputed from the reference record at run time — with ties to the earlier rung and a minimum beyond the ladder mapped to the last rung: `primary_rung(min_step, rungs)`. The rule is code; the values it yields on the shipped records are listed here so they are on record before the run:

| cell | `embedding` min step (M3e) | primary rung |
|---|---|---|
| `frozen_ssl` s0 / s1 / s2 | 2,045 / 3,555 / 4,510 | **2000 / 4000 / 5000** |
| `random_vit` s0 / s1 / s2 | 1,610 / 4,535 / 3,275 | **2000 / 5000 / 3000** |
| `pixel_ae` s0 / s1 / s2 | 8,359 / 19,096 / 16,873 | **5000 / 5000 / 5000** (control) |

The other rungs are evaluated for the shape of `H*(step)`, `gap_closed(step)` and the validation objective by step — descriptive, no verdict attached.

### 2.3 Per rung, per cell

- **The study record** — `evaluate_job` on the rung's checkpoint: `fit_probes` on `train` (the probe is refit per rung; each checkpoint has its own latent space), `reward_accuracy`, `evaluate_rollout`, `filtering_report`, `filtering_gain`, the record's provenance fields, and a `history` block that is the retrain's history sliced to the rung (`history_at(history, N)`: `loss[:N]`, `parts[:N]`, `steps = N`, `seconds` = the elapsed time the trainer recorded at that step, `steps_per_second`, `loss_last20`, `kl_dyn_max` and `kl_rate_above_free_bits` recomputed over the prefix). From it, the gate's own metric: `gap_closed(45)` on position per seed and its unanimity across the arm's three seeds — `beats_persistence` at that rung.
- **The trust pass** — `prepare_cell` on the rung directory (its exit-14 reproduction now proves `evaluate_job`'s rollout and the pass agree), one `reference_trajectories` band on `val`, and M3e's `stratum_summary(traj, horizon)`: the six curves, `moved`, `h×` and `Δ(h)` on both channels, `S(h)`, `H*_q` for q ∈ {0.5, 0.75, 0.9}, `u(h)`, `S_c(h)`, the counts. The 229 windows over 24 episode-clusters are the same at every rung, by construction and checked.
- **The validation objective** — new: `val_objective(model, buffer, val_paths, cfg, device, batches=50, seed=0)`, the model's own training loss and its `parts` (`embedding`, `reward`, `continue`, `kl_dyn`, `kl_rep`) averaged over 50 seeded `SequenceLoader` draws from the validation episodes at the training `batch_size` and `seq_len`. Printed beside the smoothed training `embedding` at the same step. Descriptive: it says whether the training minimum is also the validation minimum, which M3e could not read.

### 2.4 The anchor and the self-checks

**Anchor.** After each cell trains, `anchor_delta(retrain.loss, reference.history.loss, STEPS)` = max |Δ| over the first 5,000 per-step losses. The smoke run (one cell, the full 5,000 steps, `--anchor report`) measures it once. If it is exactly 0.0, MPS training is deterministic on this box and the real run pins `--anchor hard`: a non-zero delta is **exit 32 `EXIT_ANCHOR_MISMATCH`**, naming the cell and the first differing step, and every rung is then a state the original run passed through. If it is not 0.0, the real run pins `--anchor report`: the delta is printed in the anchor table and the results say the rungs are "a run with this seed", not the original run's states. The policy is written into this spec as one sentence after the smoke, marked as such, and not changed after the real run.

*Pinned after the smoke run (random_vit/s0, 5,000 steps, MPS, torch 2.13.0, code state `3d6c562`, 2026-09-19): the anchor delta measured exactly 0.0 at every one of the 5,000 steps, so the real run is `--anchor hard` and every rung is a state the original run passed through.*

**Self-checks**, per cell, by phase — `train`: 11, 32; `evaluate`: 11, then rung 20000's 12, 14, 30 **before any rung of that cell is evaluated**, then per rung 33, 12, 14 — with their exit statuses:

| exit | check | meaning |
|---|---|---|
| 11 | the reference cell's checkpoint, record and diagnostic exist, and (in `evaluate`) the rung checkpoint and `train_<arm>_seed<n>.json` exist | `EXIT_NO_CHECKPOINTS`, reused |
| **32** | the retrain's first 5,000 losses equal the reference record's (`--anchor hard` only) | new, `EXIT_ANCHOR_MISMATCH` |
| **33** | the rung checkpoint's `arm`, `seed` **and** `step` are the rung's | new, `EXIT_RUNG_MISLABELLED`; a `step2000` file under `step3000/` is reported under a step it was never trained to |
| 12 | the split reproduces the record's `episodes` block by name | `EXIT_SPLIT_MISMATCH`, through `prepare_cell` |
| 14 | `evaluate_rollout` on `val` reproduces the record's `curves.rssm_position` bitwise — on a rung, `evaluate_job`'s; on 20000, the M3c record's | `EXIT_RECORD_MISMATCH`, through `prepare_cell` |
| 30 | rung 20000's band reproduces the M3c diagnostic's `reference_position` / `persistence_position` at max \|Δ\| == 0.0 and its window counts | `EXIT_SELF_CHECK_FAILED`; judged before any rung of the cell is evaluated, and `read` refuses a ladder record whose recorded self-check is not 0.0 |

Pooling's own refusals keep their existing exits. 32 and 33 are distinct from every other tool's exit status and from argparse's 2, and join `test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own`.

## 3. How it decides

Stated here, before the run.

### 3.1 Pooling

- **Series.** M3e's `survival_series(record, stratum="val", channel, h)` — the indicator `1[h× > h]` with `changed = isfinite(h×)` — and `margin_series` — `Δ(h)` with `changed = moved[:, h−1] & isfinite` — built from each rung's summary exactly as `scripts/split_gap.py` builds them.
- **Treatment and control.** For arm A, the treatment group is the three cells `{seed k at primary_rung_k}` and the control group the three cells `{seed k at 20000}`, labelled `A@primary` and `A@20000` so `pooling.paired_contrast` — which refuses a contrast of an arm label against itself — reads them as two groups on the same windows. `paired_contrast(treatment, control)`: seeds averaged per window within each group, the per-window difference, its mean, the episode-clustered SE over the windows every cell of both groups changed, and z. The pairing is on window identity, which `_require_same_windows` enforces.
- **Family.** Reading T makes six clustered contrasts — three arms × two channels at one horizon; `z_fam = pooling.cluster_threshold(family=6, clusters=24)`, the val stratum's cluster count. Every contrast in the family is read against it.
- **Per seed as well as pooled.** The same contrast within each seed alone — that seed's primary rung against that seed's 20000 — clustered over the same episodes, printed with its own status under the single-seed rule (§3.3).
- **Measurability.** The probe channel pools the cells whose persistence-to-floor band at h=45 is positive on `val` at **both** rungs of the pair; the free channel pools every cell.

### 3.2 Horizon

**h = 15**, as M3e pre-registered it: the governing spec's imagination horizon, where `S(15)` on val is 0.14–0.20 at 20000 and has room to move in either direction. h = 5 and h = 45 are printed with their z and not decided on.

### 3.3 Reading T — the timing hypothesis

Per arm, the primary statistic is the paired rung difference of the survival fraction on the free channel,

`T_free(15) = S_free^primary(15) − S_free^20000(15)`,

positive when the model at its loss minimum keeps ahead of persistence in embedding space for 15 imagined steps on more of the moved (window, seed) draws than the step-20,000 model does. Its twin is `T_probe(15)`. The paired difference of `Δ_free(15)` is the continuous companion and is not decided on.

Verdict per arm, in this precedence, each printed with the rule that produced it:

| status | rule | read |
|---|---|---|
| `UNRESOLVED_PROBE` | `T_free(15)` and `T_probe(15)` both clear `z_fam` with **opposite** signs | the channels contradict; nothing below is read for this arm |
| `EARLIER_BETTER` | `T_free(15)` z > `z_fam` pooled **and** within ≥ 2 of 3 seeds | evaluating at the loss minimum helps; checkpoint selection is the next M3 change, and the gate is re-read at the primary rung |
| `EARLIER_WORSE` | `T_free(15)` z < −`z_fam` pooled **and** within ≥ 2 of 3 seeds | the later model rolls out better despite its higher one-step loss; the loss is not the rollout's ruler — the objective question, sharpened |
| `NO_DIFFERENCE` | none of the above — including a pooled clear that does not replicate, which the rule sentence names | the checkpoint is not the cause; the next study is loss balance |

A single-seed leaf has no replication clause: a clearing seed reads `EARLIER_BETTER` / `EARLIER_WORSE` on its own, "this seed alone" — the M3e correction, built in. One pooled status per arm and one per seed; the arm's status is the pooled one. `pixel_ae` reading `EARLIER_BETTER` is the control failing and is written as such, not as a fix.

**The gate is reported, not decided.** For every arm × rung, `gap_closed(45)` per seed and the unanimity flag, in one table; an arm × rung that passes is printed as `GATE PASSES at step N` and stands as the gate's own reading, alongside Reading T, never folded into it.

### 3.4 Companions — change no verdict

- The probe channel recomputed with cells of probe selection R² < 0.1 at either rung excluded, M3d's pattern.
- `T_free` and `T_probe` at h = 5 and h = 45 with their z.
- `H*_{0.5, 0.75, 0.9}` on both channels, `S(5) / S(15) / S(45)`, `u(h)` and `S_c(h)` at every rung, per arm.
- The validation objective (`loss` and each part) at every rung beside the smoothed training `embedding` at the same step; the rung at which each is smallest, per cell.

## 4. What it does not claim

- It does not change the M3 gate, `report_study.py`, `aggregate.py`, or any verdict M3b–M3e recorded. A gate pass at an earlier rung is a new reading of the same gate at a different checkpoint, and the plan says so in those words.
- It does not rank arms. Every contrast is within-arm, rung against rung; between-arm differences are printed and not tested.
- `EARLIER_BETTER` says checkpoint timing matters. It does not say the loss minimum is the *best* checkpoint — the ladder is 1,000 steps coarse and the minimum is of a smoothed noisy curve — and it does not say why the loss rises afterwards.
- `NO_DIFFERENCE` says the checkpoint is not the cause. It does not identify which of loss balance, learning rate, or the KL regime is.
- With `--anchor report` the rungs are a fresh run with the study's seed; the reading holds for that run and the anchor delta is printed with it.
- Probe-based numbers inherit the probe's R² and its floor; they are comparable within a cell between rungs because both rungs are read through a probe fit the same way, and the decision channel is probe-free.
- Everything is `my_way_home`'s 122 episodes, context 5 / horizon 45, one split, the M3c seeds and configuration. A different schedule, horizon or environment is a different measurement.

## 5. Shape of the code

Five units, one job each; three extend what exists, two are new.

**`src/mbfps/training/world_model.py`** — `train_world_model(cfg, buffer, out_dir, log_every=100, checkpoint_steps: tuple[int, ...] = ())`, and `history_at(history, step) -> dict`, which slices a training history to a prefix and recomputes the derived fields (`seconds` from `checkpoint_seconds`, the two KL summaries over the prefix) beside the `_kl_rate` it needs. When `step + 1` is in `checkpoint_steps` and `out_dir` is set: `torch.save({"arm", "seed", "step", "state_dict"}, out_dir / f"step{N}" / f"world_model_{arm}_seed{seed}.pt")` and `history["checkpoint_seconds"][N] = elapsed`. The default `()` leaves the final save and the history byte-identical; a step beyond `cfg.train.steps` is a `ValueError` before training starts. Pinned: default unchanged; `(2, 4)` on a tiny config writes two labelled files and two elapsed times, and the step-4 `state_dict` equals the final when `steps=4`; a shorter run's losses are the longer run's prefix on CPU (the anchor's premise, where it can be asserted in a test); `history_at` on a synthetic history.

**`src/mbfps/eval/study.py`** — `evaluate_job(job, buffer, out_dir, *, history, steps, seq_len, context, horizon, device) -> dict`: the block of `run_job` from the checkpoint load to `write_record`, moved, unchanged; `run_job` = `train_world_model` + `evaluate_job`. Pinned: `run_job`'s record on the fixture equals `train_world_model` + `evaluate_job`'s in the sanitised projection.

**`src/mbfps/eval/objective.py`** (new) — `val_objective(model, buffer, paths, cfg, device, *, batches=50, seed=0) -> dict[str, float]`: mean `loss` and mean of each `parts` entry over `batches` draws from `SequenceLoader(buffer, batch_size=cfg.train.batch_size, seq_len=cfg.train.seq_len, seed=seed, paths=paths, ...)` with the model in eval mode and no gradient. Pinned: finite, seed-deterministic, different for two differently initialised models, and equal to `model(batch)`'s own `parts` on one batch.

**`src/mbfps/eval/ladder.py`** (new) — pure over floats, dicts and arrays; no torch, no files:
- `RUNGS = (1000, 2000, 3000, 4000, 5000)`, `STEPS = 5000`, `REFERENCE_RUNG = 20000`, `DECISION_H = 15`, `REPORTED_H = (5, 15, 45)`, `Q_REPORTED = (0.5, 0.75, 0.9)`, `FAMILY = 6`, `SEEDS_REQUIRED = 2`, `R2_SENSITIVITY = 0.1`, `CURVE_WINDOW = 100`, `OBJECTIVE_BATCHES = 50`.
- `primary_rung(min_step: int, rungs) -> int` — nearest; ties to the earlier; beyond the last → the last; before the first → the first.
- `anchor_delta(retrain_loss, reference_loss, steps) -> tuple[float, int | None]` — max |Δ| over the first `steps` and the first differing step (None when 0.0); a reference shorter than `steps` raises.
- `Status` (`UNRESOLVED_PROBE`, `EARLIER_BETTER`, `EARLIER_WORSE`, `NO_DIFFERENCE`), `ArmInputs` (`t_free`, `t_probe`, `per_seed`, `primary_rungs`), `TimingInputs` (`arms`, `z_fam`, `h`), `ArmReading`, `TimingReading`; `clears(z, bar)` strict and never on NaN or ±inf; `reading_timing(inputs) -> TimingReading` — §3.3's rules with the single-seed vacuous clause; `format_reading_timing`.
- `gate_passes = split_gap.train_held_passes_gate` — the same predicate (all finite, all > 0), imported under the name this reading uses.
- Pinned on fabricated inputs, one rule mutated at a time: opposite-sign clears → `UNRESOLVED_PROBE`; +4 z in 3 of 3 → `EARLIER_BETTER`; −4 z in 2 of 3 → `EARLIER_WORSE`; +4 z pooled in 1 of 3 → `NO_DIFFERENCE` with "only 1 of 3 seeds" in the sentence; z exactly at the bar does not clear; a NaN never clears; a single-seed leaf at +4 z reads `EARLIER_BETTER` with "this seed alone".

**`scripts/checkpoint_ladder.py`** (new) — loads `trust_horizon` and `split_gap` by path as the others load their siblings. Flags: `--out` (default `runs/m3f_ladder`), `--reference` (default `runs/m3_study_v2`), `--data` (default `data/my_way_home`), `--device`, `--context 5`, `--horizon 45`, `--steps 5000` (≥ the last rung, else argparse's 2), `--arms`, `--seeds`, `--phase train|evaluate|read|all`, `--anchor hard|report` (default pinned by §2.4 after the smoke), `--objective-batches 50`, `--figure` (default `<out>/ladder_curves.png`). The phase functions take `rungs` as a parameter with `RUNGS` as its default so tests run a tiny ladder without patching the pre-registration.
- `train`: per cell, `get_config` + `train_world_model(checkpoint_steps=rungs)` into `--out`; the anchor against the reference record; `train_<arm>_seed<n>.json` (the full history, `checkpoint_seconds`, the anchor delta and first differing step, `git_sha`, device, torch version, the reference record's `git_sha`).
- `evaluate`: per cell, rung 20000 first — `prepare_cell` on `--reference`, the exit-30 self-check, the band, `stratum_summary`, `val_objective` on the 20k model — then each rung: 33, `evaluate_job(history_at(history, N))`, `prepare_cell` on the rung directory, band, `stratum_summary`, `val_objective`. One `ladder_<arm>_seed<n>.json` per cell: `rungs: {N: {gate, summary, objective, probe_r2, record_git_sha}}` for all six, the primary rung and its `min_step`, provenance.
- `read`: nine ladder records (one `git_sha`, one window identity), the series, the contrasts, `reading_timing`, and `ladder.txt` in `split_gap.txt`'s style: the anchor table; the primary-rung table; the gate table (arm × rung, per-seed `gap_closed(45)`, `GATE PASSES` flags); `H*_q` by rung; `S(h)`, `u(h)`, `S_c(h)` by rung; the objective table; Reading T with z, `z_fam` and the rule sentences; the sensitivity block; the per-seed block; the figure. Exit **0** when every phase asked for ran and the reading printed; **11 / 12 / 14 / 30** with their existing meanings; **32 / 33** new.

**Tests.** `tests/eval/test_ladder.py`: the pure functions above. `tests/training/test_world_model.py`: the `checkpoint_steps` pins. `tests/eval/test_run_study.py`: the `evaluate_job` / `history_at` pins. `tests/eval/test_objective.py`: the `val_objective` pins. `tests/eval/test_checkpoint_ladder_script.py`, on the existing real-`run_job` fixture cell as the reference (`steps=5`, `seq_len=4`, `context=2`, `horizon=3`, `device="cpu"`) with a tiny ladder (`rungs=(2, 4)`, `--steps 4`): `--phase all` produces the layout, six-rung records and a Reading T line per arm; `--anchor hard` on a doctored reference history → 32 naming the step, `--anchor report` prints it; a checkpoint relabelled to another step → 33; a missing rung checkpoint → 11; a doctored `reference_position` → 30 **before any rung is evaluated** (a `_never_evaluate` guard on `evaluate_job`, M3e's `_never_refit` pattern), and `read` on a ladder record whose self-check is doctored non-zero → 30; the exit-status distinctness test gains `checkpoint_ladder`. A mutation table with the harness self-checked, as every M3 task's.

**Record.** Results go in the plan written from this spec, in its own results section, numbers copied not rounded, the closing paragraph stated against §4's non-claims. `NO_DIFFERENCE` on every arm is a result. The M3c, M3d and M3e plans are closed records and are not amended.

## 6. The run

- **Smoke:** `random_vit`/s0, `--phase all --anchor report`, into `runs/m3f_smoke` — a directory the real run never reads, since nothing under `runs/` is ever removed — the full 5,000 steps (~22 min) plus six evaluations (~18 min). Its anchor delta pins `--anchor` (§2.4); its `ladder.txt` is read for format.
- **Real run:** all nine cells, `--phase all`, under `caffeinate -dimsu` and `nohup`, with `ladder.started` / `ladder.head` / `ladder.finished` and the log. Cost: 9 × ~22 min training + 45 × ~3.5 min evaluation + 9 reference self-checks ≈ **6.5 h**; ~1.8 GB of checkpoints. Phases are resumable: a failure in `evaluate` is re-run on the saved checkpoints without retraining.
- **Acceptance:** nine `ladder_*.json` at one `git_sha` == HEAD (tree clean under `src/` and `scripts/`), device `mps`; rung 20000's self-check exactly 0.0 on 9/9; the anchor 0.0 on 9/9 under `hard`; 229 windows over 24 clusters on every rung of every cell; the primary rungs equal to §2.2's table.

## 7. Files

| file | change |
|---|---|
| `src/mbfps/training/world_model.py` | `checkpoint_steps`, the rung saves, `checkpoint_seconds`; `history_at` |
| `src/mbfps/eval/study.py` | `evaluate_job` extracted from `run_job` |
| `src/mbfps/eval/objective.py` | new: `val_objective` |
| `src/mbfps/eval/ladder.py` | new: the constants, `primary_rung`, `anchor_delta`, Reading T |
| `scripts/checkpoint_ladder.py` | new: the three phases, records, pooling, Reading T, figure, `ladder.txt` |
| `tests/training/test_world_model.py`, `tests/eval/test_run_study.py` | the pins above |
| `tests/eval/test_ladder.py`, `tests/eval/test_objective.py`, `tests/eval/test_checkpoint_ladder_script.py` | new |
| `tests/eval/test_diagnose_dynamics_script.py` | the exit-status distinctness test gains `checkpoint_ladder` |
| `runs/m3f_ladder/**` | outputs; gitignored like everything under `runs/` |
