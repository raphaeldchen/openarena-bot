# MB-FPS M3i — Does the latent encode motion at all? (Design)

**Status:** design approved 2026-09-26. Evaluation only; no training, no checkpoint altered.

---

## 1. Why

M3 has never passed its exit gate: `gap_closed(45) > 0` in every seed on position. Five milestones have narrowed where the loss is, and M3h narrowed it to the point where the remaining prior-side levers look unpayable.

What the **recorded** artefacts say, with their definitions rather than their headlines:

- **The prior already knows nearly everything the posterior knows.** M3g's `information` is `KL(posterior ‖ teacher-forced prior)` *summed over all 32 groups*, measured at **0.290 / 0.467 / 0.489 nats** (`pixel_ae` / `frozen_ssl` / `random_vit`, `runs/m3g_stages/stages.txt`). Roughly 0.01–0.015 nats per group. Whatever the frame injects beyond the prior's prediction, there is about four tenths of a nat of it in the entire latent.
- **Persistence predicts the latent better than the model's own prior does.** Same table: `persist` 0.824–0.866 against `teacher` 0.796–0.845, over (step, group) argmax agreement. Copying `z(t−1)` beats the one-step prior in all three arms.
- **The marginal alone gets two thirds.** `marginal` 0.605–0.655 — guessing each group's most common class.
- **The latent does carry absolute position**: `latent_selection_r2` 0.18–0.34 across the M3c records, with one outlier at **−0.008** (`pixel_ae`/s1).

M3h ruled out the prior's *sampling temperature* and named three remaining levers in its §8 — the dyn-KL weight, the free-bits floor's batch-and-time-mean clamp, and action conditioning. All three are prior-side. Against 0.4 nats of headroom and a prior that already loses to persistence, they are levers on a stage with almost nothing left to learn.

**The hypothesis M3i tests.** *Absolute position and motion are different things.* A latent that encodes "which corridor am I in" scores well on a position probe while carrying nothing about step-to-step displacement. The M3 gate compares the imagined trajectory against **persistence** — staying put. If the latent carries position but not displacement, the model cannot beat persistence by construction, whatever the prior does. That one hypothesis would explain the `persist > teacher` inversion, the gate never passing, and M3h's null together.

**What M3i is not.** It is not another prior-side experiment, and it spends no training time. It measures the representation's **best case**: the posterior latent, on real frames. If a latent that has seen the frame cannot predict displacement, nothing downstream can — which is what makes a negative result strong enough to retire three levers rather than merely discourage them.

### 1.1 Method note, carried forward deliberately

M3h shipped a motivating premise in its spec §1 that was never measured as stated — a mean compared against a median, corrected post-run to hold in 4 of 9 cells rather than 9. Every number in §1 above is quoted from a committed artefact with its definition read out of the source, not from a scratch measurement. M3i's own descriptive statistics are written into records for the same reason.

---

## 2. What it does

Evaluation only, on the nine shipped M3c checkpoints (`runs/m3_study_v2`, `record_git_sha` `ca3e140`), at the protocol every milestone since M3d has used: **context 5, horizon 45, 229 windows over 24 validation episodes, `decision_h = 15`**, device `mps`.

One `observe` pass per cell yields posterior logits, teacher-forced prior logits and the sampled latents; `diagnostics._diagnose`'s `keep_latents=True` flag (added in M3g) already retains exactly the five `_Pass` latent fields needed. From that pass, three families of measurement.

### 2.1 The rule — a displacement probe

A ridge probe maps the posterior latent at step *t* to the **displacement** `Δ(t→t+k) = p(t+k) − p(t)`. Fit on **training** episodes only, at the rollout's own context and horizon — `fit_probes`' docstring records that fitting at a different filtering depth measurably moves the result, and that validation paths would leak.

Scored per window, in the project's existing idiom:

```
error_model   = || Δ̂(t→t+k) − Δ(t→t+k) ||
error_persist = || 0 − Δ(t→t+k) ||  =  || Δ(t→t+k) ||
contrast      = error_persist − error_model        (positive ⇒ the latent beats "stay put")
```

This is deliberately the same shape as `gap_closed`: a paired, per-window, model-against-persistence contrast. It reuses `pooling.paired_contrast` unchanged and is clustered on the same 24 episodes.

### 2.2 The control — a permutation, and it must refuse

The same probe, with **which window's latent is paired with which window's Δ** permuted under a fixed seed. This destroys the latent→displacement pairing while leaving both marginal distributions untouched. Shuffling *within* a window would not do this: it would leave each latent paired with its own Δ.

If the permuted probe clears ±`z_fam` anywhere, the instrument is reading structure that cannot exist. The run exits **38 `EXIT_CONTROL_LEAKED`** and **no reading is taken** — a known-answer control in M3g's style, which refuses rather than reports.

### 2.3 The descriptive family — reports, decides nothing

Per cell, written into the record and printed beside the reading:

- **Per-group entropy** of the posterior, via `stages.entropy_by_group` (exists, tested): `(G,)` in nats, 0 for a sharp group and `log 32 = 3.466` for a uniform one.
- **Live groups** — how many of the 32 groups change argmax at least once within a window, median over windows. A latent whose groups are mostly constant is far smaller than 32×32 in effect.
- **Top-1 mass** of the posterior and of the teacher-forced prior.
- **`information`** recomputed here beside them, so this milestone's headline number and M3g's are read off the same pass.

None of these carry a threshold. Inventing one for a quantity nobody has looked at yet is precisely the error §1.1 records.

---

## 3. How it decides

### 3.1 Pooling

`pooling.pool_arm` / `paired_contrast`, seeds averaged per window, episode-clustered over the 24 validation episodes — the same machinery M3e–M3h read. Family = **3 arms**, so `z_fam = cluster_threshold(3, 24) = 2.582`. `SEEDS_REQUIRED = 2`, `ARMS_REQUIRED = 2`, matching M3h.

### 3.2 Reading D — does the latent encode displacement?

Decided at `k = DECISION_H = 15`, imported from `split_gap` rather than re-spelled. Statuses, in precedence order:

| status | rule |
|---|---|
| `UNRESOLVED_CONTROL` | the permuted control cleared ±2.582 anywhere — **nothing is read** |
| `MOTION_ENCODED` | the contrast clears **+2.582** pooled **and** in ≥ 2 of 3 seeds, in ≥ 2 of 3 arms |
| `NO_MOTION` | no arm clears positively, and some arm clears **−2.582** |
| `NO_DIFFERENCE` | neither — indistinguishable from "stay put" at this horizon |

Precedence is that order, so a failed control can never be overwritten by a result.

`K_REPORTED = (1, 5, 15, 30, 45)`. Only k = 15 decides; the others are printed and decide nothing, exactly as the M3 gate is.

### 3.3 What each status commits the project to, pre-registered

- **`NO_MOTION` or `NO_DIFFERENCE`** — all three of M3h §8's levers are refuted in advance. They are prior-side, and the prior has ~0.4 nats of headroom over a latent that carries no displacement. The project redirects upstream, to the encoder and the representation loss (`rep_scale = 0.1`, five times below `dyn_scale`).
- **`MOTION_ENCODED`** — the latent does carry displacement the rollout fails to exploit. The §8 levers stay live and §2.3's tables rank them.

This is written down now so the consequence cannot be argued after the numbers are seen.

### 3.4 `pixel_ae`/s1

That cell's recorded `latent_selection_r2` is **−0.008**: its latent predicts absolute position essentially not at all. It **stays in the nine**. Dropping a cell after seeing its numbers is the freedom this project refuses. If Reading D splits along that cell, the results section says so.

---

## 4. What it does not claim

- It does not change the M3 gate, `aggregate.py`, `report_study.py`, any shipped checkpoint, or any verdict M3b–M3h recorded.
- It does not rank arms. Every contrast is within a cell, against that cell's own persistence baseline on the same windows.
- A linear ridge probe is a **lower bound** on the information present: `NO_MOTION` says no *linear* read-out of the posterior latent beats staying put, not that the information is absent under every decoder. The claim it licenses is about what the rollout's own linear machinery can exploit — which is the machinery the M3 gate uses.
- `MOTION_ENCODED` does not say the prior can learn the displacement, only that it is there to be learned.
- It says nothing about a model trained differently. It reads the nine cells that exist.
- A reading at `k = 15` on `my_way_home`'s 24 validation episodes at context 5 / horizon 45, at 20,000 steps, is not a statement that the world model works.

---

## 5. Shape of the code

**`src/mbfps/eval/motion.py`** — `displacement(positions, k)`; the paired contrast against `Δ = 0`; `MotionArm`, `MotionInputs`, `MotionStatus`; `reading_displacement`; `format_reading_displacement`. Holds the pre-registered constants: `MOTION_FAMILY = 3`, `SEEDS_REQUIRED = 2`, `ARMS_REQUIRED = 2`, `K_REPORTED`, `CONTROL_SEED`. Imports `DECISION_H` from `split_gap` and `entropy_by_group` from `stages` rather than re-spelling either. Pinned: the four statuses and their precedence; that a failed control suppresses any reading; the seed and arm tallies; and the family threshold's exact value.

**`scripts/latent_motion.py`** — phases `measure | read | all`. Loads each cell through `trust_horizon.prepare_cell`, inheriting the reviewed loader, the protocol checks (12, 14) and the reproduction bound (spec M3h §2.4). One record per cell, `motion_<arm>_seed<n>.json`, carrying the contrast at every `K_REPORTED`, the control's contrast, and §2.3's descriptive block. `read` writes `motion.txt`, byte-identical to what it prints.

**Exit codes.** 0 / 11 / 12 / 14 / 30 inherited from `trust_horizon`; **38 `EXIT_CONTROL_LEAKED`** new. M3h holds 35–37, so there is no clash. The exit-status registry test lives in `tests/eval/test_diagnose_dynamics_script.py` (it already lists `stage_decomposition` and `sharper_latent`) and gains `latent_motion`.

**Tests.** `tests/eval/test_motion.py` and `tests/eval/test_latent_motion_script.py`, in the established idiom — hand-typed expected values, one rule mutated per test — plus two **known-answer** cases: a fabricated latent that encodes displacement exactly must read `MOTION_ENCODED`, and pure noise must read `NO_MOTION` with the control not firing. A test that pins the printed table's captions against the columns they caption, which is the defect class that has now shipped three times in this project.

---

## 6. The run

- **Smoke:** one cell (`pixel_ae`/s0) into `runs/m3i_smoke/`, a directory the real run never reads; its `motion.txt` read for format before the real run. M3h's smoke caught a reporting bug the whole suite had missed.
- **Measure:** nine cells, one `observe` pass and two probe fits each: **~1 h**, under `caffeinate -dimsu` and `nohup`, `.head` / `.started` / `.finished` / `.exit` beside the log.
- **Acceptance:** nine records at one `git_sha` == HEAD on `mps`; 229 windows over 24 clusters on every cell; the self-check within the reproduction bound on 9/9; `sweep`-style provenance.
- **Before any full-suite run:** check `du -sh $TMPDIR/pytest-of-$USER`. The suite's fixtures leave ~20 GB per run; `tmp_path_retention_policy = "failed"` was merged in PR #8, but the habit is cheap.

---

## 7. Files

| file | change |
|---|---|
| `src/mbfps/eval/motion.py` | new — statistics and Reading D |
| `scripts/latent_motion.py` | new — the run, phases, records, `motion.txt` |
| `tests/eval/test_motion.py` | new |
| `tests/eval/test_latent_motion_script.py` | new |
| `tests/eval/test_diagnose_dynamics_script.py` | add `latent_motion` (38) to the exit-status registry test it holds |
| `docs/superpowers/plans/2026-09-26-mb-fps-m3i-latent-motion.md` | the implementation plan |

Nothing else is modified. No checkpoint is altered, nothing under `runs/` is removed, and no M3b–M3h verdict or record is touched.
