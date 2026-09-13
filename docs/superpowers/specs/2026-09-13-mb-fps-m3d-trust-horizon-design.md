# MB-FPS M3d — The trust horizon: does the h=45 gate reward slow drift, and how far can imagination be trusted? (Design)

**Status:** approved in conversation 2026-09-13; separate diagnostic, the M3 gate untouched (option 1 of three).
**Reads:** `runs/m3_study_v2` (nine M3c checkpoints and records, one code state `ca3e140`, device `mps`) and the nine `diagnostic_<arm>_seed<n>.json` the ladder wrote.
**Writes:** `runs/m3_study_v2/trust_<arm>_seed<n>.json`, `trust.txt`, and one results section in this diagnostic's own plan.

## 1. Why

M3c's gate reads `beats_persistence` at horizon step 45: does a 45-step open-loop imagination land closer to the true position than "assume the agent never moved"? Every one of the nine cells fails it. But the arms fail it in an order that inverts the mechanism readings:

| arm | gap closed at h=45 (mean of 3 seeds) | held-action position response (pooled z) |
|---|---|---|
| `random_vit` | −0.46 (least bad) | +3.33 (below the 3.47 family bar: null) |
| `frozen_ssl` | −0.72 | +10.59 |
| `pixel_ae` | −0.81 | +4.64 |

The arm whose prior responds least to actions loses least at h=45. Two readings of that are possible and the records cannot separate them: (a) `random_vit`'s prior *drifts slowly* — stays near persistence, which the metric rewards — while `frozen_ssl` moves in the right direction but too far; or (b) `frozen_ssl` moves in the wrong direction. The ladder stores per-step *mean error curves*, not per-window predicted positions, so magnitude and direction cannot be told apart from what is on disk.

The same records do settle a second thing, computed 2026-09-13 from the stored `k45_position` and `persistence_position` curves:

| cell | step where open-loop error first exceeds persistence | gap@5 | gap@15 | gap@45 |
|---|---|---|---|---|
| `frozen_ssl`/s0 · s1 · s2 | 10 · 5 · 2 | +0.36 · −0.11 · −1.33 | −0.48 · −0.56 · −1.47 | −0.85 · −0.65 · −0.65 |
| `pixel_ae`/s0 · s1 · s2 | 9 · 2 · 6 | +0.19 · −0.79 · +0.01 | −0.32 · −0.48 · −0.41 | −1.16 · −0.59 · −0.67 |
| `random_vit`/s0 · s1 · s2 | 1 · 11 · 17 | −0.49 · +0.26 · +0.48 | −0.50 · −0.21 · +0.10 | −0.44 · −0.39 · −0.55 |

Open-loop imagination beats persistence for a median of ~6 steps (range 1–17), then never again. (Not to be confused with the k-step *re-grounding* sweep, where re-feeding a real frame every 15 steps beats persistence at step 45 in every cell.) M4 trains an actor-critic on imagined trajectories; whatever horizon it imagines over has to be one where the model is better than standing still. That number is currently ~6 and depends on the arm, and the gate ranks the arms one way while the action response ranks them another.

Two deliverables, one diagnostic: (1) a pre-registered test of whether the h=45 metric rewards slow drift, so the research comparison (design objective 3) is not answered by "the prediction space that moves least"; (2) the per-arm horizon M4 designs around.

**Scale of the instrument, measured on the ladder's 229 windows:** the true 2-D displacement `|p(h) − p(0)|` has median 4.0 map units at h=1, 21.5 at h=5, 56.5 at h=15, 115.6 at h=45 (`pos_z` is constant on `my_way_home`; position is `(pos_x, pos_y)`). The position probe's own error on a *real* frame (the floor) is 120–250 units. The persistence-to-floor band at h=45 is 20–70 units. Every probe-based number below is a difference of that order on top of a floor larger than the displacement being predicted; Section 4 says what that permits.

## 2. What it measures

Per cell (9), per window (the ladder's 229, cut from 24 validation episodes of `episode_split(seed=0)` by `window_starts(context=5, horizon=45)`), per open-loop step h = 1..45, the reference pass — `imagine` over the real horizon actions, context filter shared, RNG snapshot matched, exactly the canonical pass `_diagnose` already runs — keeps four things the ladder discards:

- `p̂(h)`: the probe reading of the imagined latent at step h (through the embedding head and the cell's own probe, the single-pipeline rule of `rollout.py`); `p̂(0)`: the same reading at the last context step. `p̂(0)` held is what `rollout.py` calls persistence, so nothing new is anchored.
- `p(h)`: true `(pos_x, pos_y)` from `privileged_state` at the frame the action produced (`start + context + 1 + k`, the off-by-one `rollout.py` pins); `p(0)`: at the last context step.
- `D̂(h) = ‖ê(h) − e(h)‖`: embedding-space distance from the imagined embedding to the real imagination's, as the ladder's probe-free channel computes it; `D₀(h) = ‖ê(0) − e(h)‖`: the embedding-space *persistence* distance. The ladder's noise reference (the two-draw distance) is carried alongside as the scale.

Per window, three quantities:

1. **Crossing step** `h×`: the first h at which model error exceeds persistence error. Probe-based: `|p̂(h) − p(h)| > |p̂(0) − p(h)|`. Probe-free: `D̂(h) > D₀(h)`. A window that never crosses records `h× = 46`. Reported as a distribution over windows — median and interquartile range — never only a mean curve.
2. **Displacement decomposition** at each h, over the windows where the agent moved: `|p(h) − p(0)| ≥ 5` map units (excludes ~58 % of windows at h=1, ~16 % at h=5, ~1 % at h=45; the excluded count is reported at every h). With `d̂ = p̂(h) − p̂(0)` and `d = p(h) − p(0)`:
   - magnitude ratio `|d̂| / |d|` (0 = the prior sits still, 1 = right distance, > 1 = overshoot);
   - direction agreement `cos∠(d̂, d)`;
   - **scale-corrected error** `min over α ∈ [0, 2] of median_w |p̂(0) + α·d̂ − p(h)|`, with α fit on 12 of the 24 episodes and scored on the other 12, both folds, both reported. This is what the error *would* be if only the magnitude were wrong.
3. **Trust horizon** `H*`: the largest h at which the median window still beats persistence — `median(h×) − 1` — probe-based and probe-free.

**Self-check, run before any reading is printed.** The mean over windows of `|p̂(h) − p(h)|` and of `|p̂(0) − p(h)|` must equal the stored `curves.k45_position` and `curves.persistence_position` of `diagnostic_<arm>_seed<n>.json` to float precision at every h, for every cell; and the window count and per-window episode labels must equal the diagnostic's. Same windows, same rollout, same probe — or this is not measuring what the ladder measured, and the run exits non-zero without a reading.

## 3. How it decides

Both readings are stated here, before the run.

**Reading 1 — does the h=45 gate reward slow drift?** Pooled over seeds per arm (seeds averaged per window, as `pool_arm` does), with the episode-clustered standard error of `pooling.py` (24 clusters; z read against t(23)). *Supported* only if all three hold:

- (i) the arm with the smallest pooled magnitude ratio `|d̂| / |d|` at h=45 is the arm with the best pooled `gap@45` — the ranking by "how little the prior moves" matches the ranking by the gate;
- (ii) `frozen_ssl`'s direction agreement `cos∠(d̂, d)` at h=45 exceeds `random_vit`'s, paired per window, with the 95 % episode-bootstrap CI excluding 0 — the treatment moves the right way;
- (iii) after scale correction, `frozen_ssl`'s held-out error is no worse than `random_vit`'s on both folds — the gate ranking flips or ties once magnitude is taken out.

Any of the three failing → *not supported*, written as such. (ii) failing alone is the informative failure: the prior does not drift slowly, it moves wrong. The probe is controlled by the probe-free crossing: if the rank order of the arms by median `h×` differs between the probe-based and probe-free readings, Reading 1 is reported as *unresolved through the probe* rather than as supported or not.

**Reading 2 — the horizon M4 designs around.** Per arm: median `h×` (probe-based and probe-free), `H*`, and the fraction of windows still beating persistence at h = 5, 10 and 15 (Dreamer's default imagination horizon). No verdict; a table M4's spec cites. One derived number: `H*_min`, the smallest `H*` across the three arms — the horizon any of these world models can be trusted to.

Neither reading touches `report_study.py`, `aggregate.py`, or the recorded M3 verdicts.

## 4. What it does not claim

- It does not change the M3 gate or either study's verdict; `beats_persistence` at h=45 stays the recorded 45-step stress test.
- It does not say which arm is best for control. That is M4's live-deployment criterion, and this diagnostic is one of its inputs.
- Probe-based numbers inherit the probe's ~0.3 R² and a floor larger than the displacement it reads; they are comparable *between arms* because every arm is read through the same pipeline, not trustworthy in absolute units. The probe-free reading is the check on that, not a replacement for it.
- Everything is on `my_way_home`'s 24 validation episodes at context 5 / horizon 45, one split, the M3c checkpoints. A different environment, split or horizon is a different measurement.

## 5. Shape of the code

Three units, one job each.

**`src/mbfps/eval/trust.py`** (new) — pure functions over numpy arrays, no torch, no files:
- `crossing_step(model_err, persist_err) -> np.ndarray` — `(n_windows, horizon)` in, per-window first crossing (1-based) out, `horizon + 1` for never.
- `displacement_decomposition(p_hat, p_true, p_hat0, p_true0, min_move=5.0) -> Decomposition` — per window and step: magnitude ratio, cosine, and a boolean `moved` mask; NaN where `moved` is False.
- `scale_corrected_error(p_hat, p_true, p_hat0, episode, fit_episodes, alphas) -> float` — α chosen on `fit_episodes`, scored on the rest; the caller passes both folds.
- `trust_horizon(crossings) -> int` — `median − 1`, integer.
Each is pinned by synthetic priors with known answers, and each synthetic prior is a mutation the others cannot catch: a prior returning `p̂(0)` forever gives ratio 0, `h× = 1` wherever the agent moved; a perfect predictor gives `h× = horizon + 1`, ratio 1, cosine 1, scale-corrected error equal to the raw error; a predictor that overshoots by exactly 2× gives ratio 2, cosine 1, and a scale-corrected error at α = 0.5 equal to the perfect predictor's.

**One hook in `src/mbfps/eval/diagnostics.py`** — `_diagnose` already computes `p̂(h)` and `ê(h)` per window in the canonical pass and keeps only the errors. An optional `keep_trajectories: bool = False` makes `_Pass` carry, for the reference arm only: `positions (n_windows, horizon, 2)`, `positions_at_context (n_windows, 2)`, `true_positions (n_windows, horizon, 2)`, `true_at_context (n_windows, 2)`, `embedding_distance (n_windows, horizon)`, `embedding_persistence_distance (n_windows, horizon)`. With the flag False the pass is byte-identical to today's; the ladder's 108 tests and the Section 2 self-check pin that.

**`scripts/trust_horizon.py`** — `--out runs/m3_study_v2 --device mps`, the same loading path as `diagnose_dynamics.py` (checkpoint + record per cell, the record's split and protocol, the diagnostic's probe weights). Per cell: one reference pass with `keep_trajectories=True`; the self-check against `diagnostic_<arm>_seed<n>.json`; the three quantities; `trust_<arm>_seed<n>.json` with per-window `h×` (both), per-step medians of the decomposition with excluded counts, both folds' scale-corrected error, and the self-check deltas. Then pooling and the two readings, printed to stdout and `trust.txt`. Exit 0 when every cell's self-check holds and both readings printed; **30** `EXIT_SELF_CHECK_FAILED` (named, distinct from the other tools' 7–23) with the offending cell and step; the missing-checkpoint / wrong-device / record-mismatch statuses reuse `diagnose_dynamics.py`'s numbers with the same meanings.

**Tests.** `tests/eval/test_trust.py` for the pure functions (the three synthetic priors, the `moved` mask, both folds); `tests/eval/test_trust_horizon_script.py` driving `main()` on the shared 6-episode fixture with the tiny model, asserting the record's fields, the self-check's refusal on a doctored diagnostic curve, and that the printed readings follow the Section 3 rules on a fabricated pooled table (each rule mutated one at a time). A mutation table with the harness self-checked three ways, as every M3 task's.

**Record.** Results go in `docs/superpowers/plans/<date>-mb-fps-m3d-trust-horizon.md`'s own results section — the M3c plan is a closed record and is not amended.

## 6. Files

| file | change |
|---|---|
| `src/mbfps/eval/trust.py` | new: the four pure functions and `Decomposition` |
| `src/mbfps/eval/diagnostics.py` | `keep_trajectories` flag on `_diagnose`; six optional fields on `_Pass` |
| `scripts/trust_horizon.py` | new: load, reference pass, self-check, records, pooling, readings, exit statuses |
| `tests/eval/test_trust.py`, `tests/eval/test_trust_horizon_script.py` | new |
| `runs/m3_study_v2/trust_*.json`, `trust.txt` | outputs; gitignored like everything under `runs/` |
