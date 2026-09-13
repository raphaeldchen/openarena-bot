# MB-FPS M3d — The trust horizon: does the h=45 gate reward slow drift, and how far can imagination be trusted? (Design)

**Status:** approved in conversation 2026-09-13 (sections 1–4 in turn; the decision rules revised after three independent critics — code facts, consistency, and a hostile-statistician pass — and re-approved). Separate diagnostic; the M3 gate untouched.
**Reads:** `runs/m3_study_v2` — the nine M3c checkpoints and records (one code state `ca3e140`, device `mps`) and the nine `diagnostic_<arm>_seed<n>.json` the ladder wrote.
**Writes:** `runs/m3_study_v2/trust_<arm>_seed<n>.json`, `trust.txt`, and one results section in this diagnostic's own plan.

## 1. Why

M3c's gate reads `beats_persistence` at horizon step 45: does a 45-step open-loop imagination land closer to the true position than "assume the agent never moved"? Every one of the nine cells fails it. But the arms fail it in an order that inverts the mechanism readings:

| arm | gap closed at h=45 (mean of 3 seeds) | held-action position response (pooled z) |
|---|---|---|
| `random_vit` | −0.46 (least bad) | +3.33 (below the 3.47 family bar: null) |
| `frozen_ssl` | −0.72 | +10.59 |
| `pixel_ae` | −0.81 | +4.64 |

The arm whose prior responds least to actions loses least at h=45. Two readings are possible and the records cannot separate them: (a) `random_vit`'s prior *drifts slowly* — stays near persistence, which the metric rewards — while `frozen_ssl` moves in the right direction but too far; or (b) `frozen_ssl` moves in the wrong direction. The ladder stores per-step *mean error curves*, not per-window predicted positions, so magnitude and direction cannot be told apart from what is on disk.

The same records do settle a second thing, computed 2026-09-13 from the stored `k45_position` and `persistence_position` curves:

| cell | step at which the mean open-loop error first exceeds persistence | gap@5 | gap@15 | gap@45 |
|---|---|---|---|---|
| `frozen_ssl`/s0 · s1 · s2 | 10 · 5 · 2 | +0.36 · −0.11 · −1.33 | −0.48 · −0.56 · −1.47 | −0.85 · −0.65 · −0.65 |
| `pixel_ae`/s0 · s1 · s2 | 9 · 2 · 6 | +0.19 · −0.79 · +0.01 | −0.32 · −0.48 · −0.41 | −1.16 · −0.59 · −0.67 |
| `random_vit`/s0 · s1 · s2 | 1 · 11 · 17 | −0.49 · +0.26 · +0.48 | −0.50 · −0.21 · +0.10 | −0.44 · −0.39 · −0.55 |

The mean curve first loses to persistence at a median step of 6 (range 1–17): open-loop imagination beats "you did not move" for about five steps (0–16; `random_vit`/s0 never does), then never again. (Not the k-step *re-grounding* sweep, where re-feeding a real frame every 15 steps beats persistence at step 45 in every cell.) M4 trains an actor-critic on imagined trajectories; whatever horizon it imagines over has to be one where the model is better than standing still. That number is currently ~5, it depends on the arm, and the gate ranks the arms one way while the action response ranks them another.

Two deliverables, one diagnostic: (1) a pre-registered test of whether the h=45 metric rewards slow drift, so the research comparison (design objective 3) is not answered by "the prediction space that moves least"; (2) the per-arm horizon M4 designs around.

**Scale of the instrument, measured on the ladder's 229 windows:** the true 2-D displacement `|p(h) − p(0)|` has median 4.0 map units at h=1, 21.5 at h=5, 56.5 at h=15, 115.6 at h=45 (`pos_z` is constant on `my_way_home`; position is `(pos_x, pos_y)`). The position probe's own error on a *real* frame (the floor) is 120–250 units; its selection R² across the nine cells is 0.018–0.383 — `pixel_ae`/s1 reads 0.018, every other cell ≥ 0.249. The persistence-to-floor band at h=45 is 20–70 units. Every probe-based number below is a difference of that order on top of a floor larger than the displacement being predicted; Section 4 says what that permits, and every probe-based reading has a probe-free twin.

## 2. What it measures

### 2.1 Notation

Per cell (9 = 3 arms × 3 seeds), per window w (the ladder's 229, cut from the 24 validation episodes of `episode_split(seed=0)` by `window_starts(context=5, horizon=45)`, labelled by contributing episode exactly as `_Pass.window_episode` labels them), per open-loop step h = 1..45. The reference pass is the canonical pass `_diagnose` already runs — context filter shared, RNG snapshot matched, `imagine` over the real horizon actions, then the floor's posterior — with `arms={}` and `noise_reference=True` so the stream sequence is the ladder's.

Frames: `p(0)` is the true `(pos_x, pos_y)` at frame `start + context` (the last context frame); `p(h)` at frame `start + context + h` — the frame the h-th horizon action *produced*, i.e. `start + context + 1 + k` for 0-based k = h − 1, the slice `_diagnose` already takes.

Probe channel (every quantity through the one pipeline `rollout.py` mandates — embedding head, then the cell's probe):
- `p̂(0)`: the probe of the embedding-head output of the context filter's last posterior latent (`observed["latent"][:, -1]`). Held for the horizon this is exactly `rollout.py`'s persistence (`pers_pred`). Nothing new is anchored.
- `p̂(h)`: the probe of the embedding-head output of the imagined latent at step h.
- `p̂_real(h)`: the probe of the floor pipeline's embedding at step h — the posterior run on the *real* future frames, through the same head (`floor_embeddings[h − 1]`). The same probe reading the trajectory that actually happened.

Embedding channel (probe-free):
- `e(h)`: the encoder's embedding of frame `start + context + h` — `handle.embeddings[0, context + h − 1]` — which is the embedding head's own training target (`world_model.py`: `mse_loss(predictions["embedding"], embeddings.detach())`). `e(0)` = `handle.embeddings[0, context − 1]`.
- `ê(h)`: the embedding-head output of the imagined latent at step h; `ê(0)`: the embedding-head output of the last posterior latent (`rollout.py`'s `last_context_embedding`).
- The ladder's embedding-space channels are imagination-vs-imagination (intervened vs canonical, canonical vs a second draw). Nothing today measures imagination-vs-truth; the two series below are **new** and the hook computes them.

Displacements: `d = p(h) − p(0)`, `d̂ = p̂(h) − p̂(0)`, `d̂_real = p̂_real(h) − p̂(0)`.

**Moved mask.** `moved(w, h)` ⇔ `|p(h) − p(0)| ≥ 5` map units — ground truth, so the same mask serves both channels. It excludes ~58 % of windows at h=1, ~16 % at h=5, ~1 % at h=45; the excluded count is reported at every h.

### 2.2 Per-window quantities

Each is computed per (window, seed) — 687 draws per arm on the shipped split — and pooled as Section 3 says. Nothing about a crossing step is ever seed-averaged.

**1. Crossing step `h×`.** Let `h₀(w)` be the first h with `moved(w, h)`. `h×` is the first h ≥ `h₀` at which model error *strictly* exceeds persistence error; `horizon + 1` (= 46) if never; NaN if the window never moves within the horizon (counted, not pooled). Ties never cross: a persistence clone is not worse than persistence.
- Probe-based: `|p̂(h) − p(h)| > |p̂(0) − p(h)|`.
- Probe-free: `D̂(h) > D₀(h)` with `D̂(h) = ‖ê(h) − e(h)‖` and `D₀(h) = ‖ê(0) − e(h)‖` (the embedding-space persistence error).

**2. Displacement decomposition**, at each h, on `moved` windows only:
- probe-normalised magnitude ratio `R_probe(h) = |d̂| / |d̂_real|` — the probe's reading of the imagined displacement over the *same probe's* reading of the real displacement, so a flatter probe cancels (the raw `|d̂| / |d|` is stored beside it as a secondary column, not decided on);
- probe-free magnitude ratio `R_free(h) = ‖ê(h) − ê(0)‖ / ‖e(h) − e(0)‖`;
- direction agreement `cos(h) = cos∠(d̂, d)`, NaN where `|d̂| = 0` (counted alongside the `moved` exclusions).
Gloss: ratio 0 = the prior sits still, 1 = the right distance, > 1 = overshoot; cosine 1 = the right direction.

**3. Persistence margin** `Δ(h) = |p̂(0) − p(h)| − |p̂(h) − p(h)|`, map units, positive when the model beats persistence at that step. It is the numerator of `gap_closed` per window, and it is the decision statistic in place of `gap_closed` itself: `gap_closed` is a ratio of *mean* curves whose per-window band is frequently non-positive (`rollout.py`'s own warning), so it has no per-window estimator to cluster.

**4. Scale-corrected error**, at each h, on `moved` windows: with α on the grid `np.linspace(0, 2, 201)` and the two folds A = windows from even `window_episode` labels, B = odd, `α_A(h) = argmin_α median_{w∈A} |p̂(0) + α·d̂ − p(h)|`, and every window in B is scored with `α_A`: `c_w(h) = |p̂(0) + α_A(h)·d̂ − p(h)|`; symmetrically for B → A. Per h the record holds `α_A, α_B`, the scoring-fold medians, and the per-window held-out `c_w(h)`. `α` on a grid endpoint (0 or 2) on either fold at the h a reading uses is a **boundary**, flagged in the record — at α = 0 the "corrected" error is persistence itself, and a boundary is not a correction. With fewer than two validation episodes both folds are NaN and the record says `folds_available = false`.

### 2.3 Self-check

Run before any reading is printed, per cell, in this order: `diagnose_dynamics.py`'s own preconditions first — a checkpoint and record per cell (else **11**), the split by name (else **12**), and `evaluate_rollout` reproducing the study record's `curves.rssm_position` exactly (else **14**, an environment difference: a cpu run misses by 6–12 map units). Then the trust pass's own: `np.stack(rows).mean(axis=0)` over windows of `|p̂(h) − p(h)|` and of `|p̂(0) − p(h)|` must equal the diagnostic's `curves.reference_position` (bitwise equal to `k45_position` in all nine shipped records) and `curves.persistence_position` with **max |Δ| == 0.0** at every h, the same rule as the ladder's `record_reproduction`; and `windows.total` and `windows.episode` must equal the diagnostic's. Any of these failing is **30** `EXIT_SELF_CHECK_FAILED`, naming the cell, the curve and the step, and no reading is printed: same windows, same rollout, same refit probe — or this is not measuring what the ladder measured.

## 3. How it decides

Both readings are stated here, before the run.

### 3.1 Pooling

One inferential procedure for means, one for ratios, both already in `pooling.py`:
- **Means and contrasts** (`Δ`, `cos`, held-out `c`, `h×` differences): `CellSeries` built directly from the per-window arrays (`changed` = the `moved` mask at that h; NaN windows excluded and counted), `pool_arm` (seeds averaged per window, mean over windows, episode-clustered SE over 24 clusters) and `paired_contrast` (the same, on the per-window difference between two arms), z read against t(23). Except `h×`, which is never seed-averaged: its contrast is the paired difference per (window, seed) draw.
- **Ratios** (`R_probe`, `R_free`): ratio of medians — `median` of numerators over `median` of denominators, seeds stacked as draws, `moved` windows only — with the 95 % episode-bootstrap interval `pool_ratio` implements (2000 draws, seed 0), uncorrected as in the ladder.
- **Family.** Reading 1 makes eight clustered contrasts (three pairwise `Δ(45)`, one `cos(45)`, one held-out `c(45)` per fold, and the two-channel `h×` control); `cluster_threshold(family=8, clusters=24)` is the bar `z_fam` every one of them is read against.
- **Per seed as well as pooled.** Every condition below is also evaluated within each seed alone (no seed averaging; the clustered SE over that seed's windows), and printed.
- **Measurability.** A cell enters the probe-based pooling iff its persistence-to-floor band at h=45 is positive — the ladder's `probe_is_measurable` rule; all nine shipped cells pass it. A second line recomputes every probe-based statistic with cells of probe selection R² < 0.1 excluded — a threshold chosen here knowing `pixel_ae`/s1 reads 0.018 and every other cell ≥ 0.249 — as a **sensitivity report that changes no verdict**.

### 3.2 Reading 1 — does the h=45 gate reward slow drift?

*Supported* only if (i), (ii) and (iii) all hold pooled **and** each holds within at least two of the three seeds; any other outcome is *not supported*, *not testable* or *unresolved*, written as such.

- **(i) The arm that moves least is the arm the gate likes best.** The *best-Δ arm* is the arm whose pooled `Δ(45)` contrast against *each* of the other two has `z > z_fam`; if no arm clears both, (i) is *undecidable at this precision* and Reading 1 is *not testable*, never *supported*. The *least-moving arm* is the arm with the smallest pooled `R_probe(45)` — and it must be the same arm under `R_free(45)`, or (i) is *unresolved through the probe*. (i) holds iff the least-moving arm is the best-Δ arm.
- **(ii) The treatment moves the right way.** `paired_contrast(frozen_ssl − random_vit)` on per-window seed-mean `cos(45)` has `z > z_fam`.
- **(iii) Take the magnitude out and the gate ranking goes away.** On each fold, `paired_contrast(frozen_ssl − random_vit)` on the held-out `c_w(45)`: holds iff `z < −z_fam` (the treatment's corrected error is smaller); fails iff `z > z_fam`; *undecided* otherwise. (iii) holds only if it holds on both folds, and is *unreadable* — Reading 1 *unresolved* — if either arm's α is on the grid boundary on either fold.
- **Probe control.** The paired per-draw difference of `h×` (frozen_ssl − random_vit) in each channel: if both channels clear `z_fam` with *opposite* signs, Reading 1 is *unresolved through the probe* whatever (i)–(iii) said. A tie, or an interval covering 0 in either channel, contradicts nothing. `pixel_ae`'s pair is printed for information.

(ii) failing alone is the informative failure: the prior does not drift slowly, it moves wrong.

### 3.3 Reading 2 — the horizon M4 designs around

Per arm and per channel, the **survival curve** `S(h)` = the fraction of (window, seed) draws with `h× > h`, h = 0..45, over draws that moved. `H*_q` = the largest h with `S(h) ≥ q`; **q = 0.75 pre-registered**, with q = 0.5 and 0.9 printed beside it (`H*_0.5` is the old median reading). The table M4 cites is `S(h)` itself at every h; `H*_q` is its summary, an integer by construction. One derived number: **`H*_min` = min over the three arms of the probe-free `H*_0.75`** — probe-independent, so a dead probe cannot set it — with the probe-based `H*_0.75` printed beside it so M4 can see whether the channels agree. No verdict is attached.

Neither reading touches `report_study.py`, `aggregate.py`, or the recorded M3 verdicts.

## 4. What it does not claim

- It does not change the M3 gate or either study's verdict; `beats_persistence` at h=45 stays the recorded 45-step stress test.
- It does not say which arm is best for control. That is M4's live-deployment criterion; this diagnostic is one of its inputs.
- Probe-based numbers inherit the probe's R² (0.018–0.383) and a floor larger than the displacement it reads. They are comparable *between arms* because every arm is read through the same pipeline and `R_probe` cancels the probe's slope; they are not trustworthy in absolute units, which is why `H*_min` is probe-free and every probe-based reading has a probe-free twin or control.
- Everything is on `my_way_home`'s 24 validation episodes at context 5 / horizon 45, one split, the M3c checkpoints. A different environment, split or horizon is a different measurement.

## 5. Shape of the code

Three units, one job each.

**`src/mbfps/eval/trust.py`** (new) — pure functions over numpy arrays, no torch, no files:
- `crossing_step(model_err, persist_err, moved) -> np.ndarray` — `(n, horizon)` in, per-row first strict crossing at or after the first `moved` step (1-based); `horizon + 1` for never; NaN for never moved.
- `displacement_decomposition(p_hat, p_true, p_hat0, p_true0, p_hat_real, moved) -> Decomposition` — per row and step: `ratio_probe`, `ratio_raw`, `cosine`, NaN where not `moved` or where `|d̂| = 0`, with the counts.
- `embedding_ratio(e_hat, e_true, e_hat0, e_true0, moved) -> np.ndarray` — `R_free` per row and step.
- `persistence_margin(model_err, persist_err) -> np.ndarray` — `Δ`.
- `scale_corrected_error(p_hat, p_true, p_hat0, moved, episode, alphas) -> ScaleCorrection` — per step: `alpha_a`, `alpha_b`, the scoring-fold medians, the per-row held-out error, `boundary` flags; all NaN with `folds_available = False` when fewer than two episode labels are present.
- `survival(crossings, horizon) -> np.ndarray` and `trust_horizon(survival, q) -> int`.
Each is pinned by synthetic priors with known answers, and each synthetic prior is a mutation the others cannot catch: a prior returning `p̂(0)` forever gives `ratio_probe` 0, cosine NaN, `Δ` 0 and `h× = horizon + 1` everywhere (told from the perfect predictor by the ratio, not by `h×`); a perfect predictor gives `h× = horizon + 1`, ratio 1, cosine 1, α = 1 on both folds and held-out error equal to its raw error; a 2× overshoot gives ratio 2, cosine 1, α = 0.5 and a held-out error equal to the perfect predictor's; a pure-noise `d̂` gives α = 0 on both folds **reported as a boundary**; a two-seed prior whose per-seed `d̂` point opposite ways gives a seed-mean cosine of 0; an even-count `h×` multiset gives an integer `H*_q` with no rounding.

**One hook in `src/mbfps/eval/diagnostics.py`** — `keep_trajectories: bool = False` on `_diagnose`. With it True the reference pass (and only it) additionally carries on `_Pass`: `positions (n_windows, horizon, 2)`, `positions_at_context (n_windows, 2)`, `positions_real (n_windows, horizon, 2)`, `true_positions (n_windows, horizon, 2)`, `true_at_context (n_windows, 2)`, `embedding_distance_to_truth (n_windows, horizon)` = `D̂`, `embedding_persistence_distance (n_windows, horizon)` = `D₀`, `embedding_displacement (n_windows, horizon)` = `‖ê(h) − ê(0)‖`, `true_embedding_displacement (n_windows, horizon)` = `‖e(h) − e(0)‖`. The episode labels are the existing `_Pass.window_episode`; no new field. With the flag False the pass is byte-identical to today's (the ladder's tests pin that); with it True its mean curves equal the stored diagnostic's (the Section 2.3 self-check pins that).

**`scripts/trust_horizon.py`** — `--out`, `--device`, `--data` (default `data/my_way_home`), `--context` / `--horizon` (defaults read from each cell's diagnostic; a mismatch with the record is refused as `diagnose_dynamics.py` refuses it). Loading is `diagnose_dynamics.py`'s: checkpoint + record per cell, the record's split, and the probe **refit** exactly as that script refits it — `fit_probes` on the train split at the cell's seed and protocol; no probe weights exist on disk, the diagnostic carries only the probe's R², and the refit is part of what the self-check pins. Per cell: the checks of Section 2.3 in order; one reference pass with `keep_trajectories=True`; the quantities of Section 2.2; `trust_<arm>_seed<n>.json` holding, per h, the per-window `h×` (both channels), `Δ`, `R_probe`, `R_raw`, `R_free`, `cos`, `c_w` with `α_A`, `α_B`, `boundary`, the excluded and NaN counts, and the self-check deltas. Then pooling and the two readings, printed to stdout and `trust.txt` in the ladder's style (per-arm block, then contrasts, then the verdict lines with the rule each was decided by). Exit **0** when every cell's checks hold and both readings printed; **11 / 12 / 14** with `diagnose_dynamics.py`'s meanings; **30** `EXIT_SELF_CHECK_FAILED` — distinct from the other tools' 1–23 and argparse's 2, and added to `test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own`.

**Tests.** `tests/eval/test_trust.py` for the pure functions (the six synthetic priors above; the `moved` mask; both folds; the boundary flag). `tests/eval/test_trust_horizon_script.py` drives `main()` on the shared 6-episode fixture the way `test_aggregate.py` and `test_run_study.py` build their tiny cell — `run_job(..., steps=5, seq_len=4, context=2, horizon=3, device="cpu")`, then `diagnose_dynamics.main` on it to write the diagnostic the self-check needs — asserting the record's fields, the exit statuses in order (a missing diagnostic, a doctored `reference_position`, a wrong device), and `folds_available = false` and NaN pooled SEs on the fixture's single validation episode. The two readings and every rule of Section 3 are tested on **fabricated pooled tables**, each rule mutated one at a time (a best-Δ arm that clears one contrast but not the other; a boundary α; opposite-sign `h×` channels; a seed that disagrees; an even-count survival curve). A mutation table with the harness self-checked three ways, as every M3 task's.

**Record.** Results go in the plan written from this spec, `docs/superpowers/plans/2026-09-13-mb-fps-m3d-trust-horizon.md`, in its own results section; the M3c plan is a closed record and is not amended.

## 6. Files

| file | change |
|---|---|
| `src/mbfps/eval/trust.py` | new: the pure functions of Section 5 and the `Decomposition` / `ScaleCorrection` dataclasses |
| `src/mbfps/eval/diagnostics.py` | `keep_trajectories` flag on `_diagnose`; nine optional fields on `_Pass`, reference arm only |
| `src/mbfps/eval/pooling.py` | unchanged — `CellSeries`, `pool_arm`, `paired_contrast`, `pool_ratio`, `cluster_threshold` are used as they are |
| `scripts/trust_horizon.py` | new: load, checks, reference pass, records, pooling, readings, exit statuses |
| `tests/eval/test_trust.py`, `tests/eval/test_trust_horizon_script.py` | new |
| `tests/eval/test_diagnose_dynamics_script.py` | the exit-status distinctness test gains `trust_horizon` |
| `runs/m3_study_v2/trust_*.json`, `trust.txt` | outputs; gitignored like everything under `runs/` |
