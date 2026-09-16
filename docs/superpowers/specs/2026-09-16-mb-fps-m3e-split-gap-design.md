# MB-FPS M3e — The split gap: does the world model roll out better on the episodes it trained on? (Design)

**Status:** approved in conversation 2026-09-16 (channels, base branch and scope chosen by the author; approach A of three; sections 1–3 approved in turn). Separate diagnostic; the M3 gate and every recorded verdict untouched.
**Reads:** `runs/m3_study_v2` — the nine M3c checkpoints and records (one code state `ca3e140`, device `mps`), the nine `diagnostic_<arm>_seed<n>.json` the ladder wrote, and each record's `history.parts`.
**Writes:** `runs/m3_study_v2/split_gap_<arm>_seed<n>.json`, `split_gap.txt`, `learning_curves.png`, and one results section in this diagnostic's own plan.
**Base:** `main` at `0550a82` (PR #3, M3d, merged); branch `feat/m3e-split-gap`.

## 1. Why

Every one of the nine M3c cells fails `beats_persistence` at h=45, and M3d put the trust horizon at `H*_min` = 1: open-loop imagination is better than "the agent never moved" for about one step. M4 cannot be designed around that number, and the M3b write-up's open questions still list three causes for it that call for three different fixes:

| cause | what the records say today | fix if true |
|---|---|---|
| **generalisation** — the model memorised its 98 training episodes and does not transfer | nothing; every rollout number ever recorded is on the 24 held-out episodes | more data |
| **budget / capacity** — 20,000 steps or a 9.7M-parameter RSSM is not enough to fit 45-step dynamics | the summed training loss was higher over the last 5,000 steps than the first 15,000 in 7 of 7 M3b cells; no per-term or validation curve was read | longer, or bigger, one pilot cell first |
| **objective** — a one-step embedding loss does not penalise multi-step drift | h=1 error sits at the encoder floor in all nine cells and compounds from there; the prior is blind to action order and counts | a multi-step loss: an arm-parity change, nine cells re-run |

The first cause has never been tested, and it is the one the replay arithmetic points at: 20,000 steps × 16 sequences × 64 frames = 20.5M frame-samples over ~48,000 training transitions, about **430 passes** over a fixed dataset. In online Dreamer that ratio is ordinary because the buffer grows and shifts with the policy; here the dataset was frozen on purpose (governing spec §5.1), which turns the ratio into 430 passes over the same 98 episodes.

The test is one the existing instrument already permits. `fit_probes` fits the rollout's probe on the FIRST 20 training episodes (16 for the weights, 4 to select the ridge; `limit=20`) and no others — so 78 training episodes are ones the model saw and the probe never did. Evaluating the shipped checkpoints on those 78 with the very probe each record was scored with is a train-side rollout with no probe refit and no in-sample-probe confound. And M3d's probe-free channel — the embedding-space distance to the true embedding against the embedding-space persistence distance — is the training loss's own target, so on it the question "does the model predict its own target better on episodes it trained on?" is asked in the loss's own units.

One diagnostic, two blocks: (1) a pre-registered reading of the train-versus-validation gap, so the next world-model change is chosen by evidence rather than by which fix is cheapest; (2) the per-term learning curves the records already carry and nobody has plotted, read only if (1) says the gap is not the cause.

## 2. What it measures

### 2.1 Strata

Per cell (9 = 3 arms × 3 seeds), the split is `episode_split(buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED)` — the one every record, `eval_rollout.py`, `diagnose_dynamics.py` and `trust_horizon.py` use — with `train` in the order `episode_split` returns it (sorted by name, validation removed). Three strata:

| stratum | episodes | the model trained on it | the probe was fit on it | role |
|---|---|---|---|---|
| `val` | 24 | no | no | the record's; the self-check anchor (§2.4) |
| `train_held` | `train[PROBE_EPISODE_LIMIT:]` = 78 | **yes** | no | **the decision stratum** |
| `train_probe` | `train[:PROBE_EPISODE_LIMIT]` = 20 | yes | yes — 16 fit, 4 ridge-selection | information only; shows the in-sample-probe effect size and is labelled confounded wherever it is printed |

`PROBE_EPISODE_LIMIT = 20` is today's bare `limit: int = 20` default on `fit_probes`, lifted into a named constant in `mbfps.eval.probe` beside a pure `probe_episodes(paths, limit=PROBE_EPISODE_LIMIT) -> (used, held)`. `fit_probes` routes its `used = list(paths)[:limit]` through that function and this script's strata come from the same call, so the two cannot disagree about which episodes are probe-seen. The change is behaviour-preserving, and the val stratum's bitwise reproduction of every record (§2.4, exit 14) is what proves it on the real checkpoints. The strata are a partition of the 122 episodes — pairwise disjoint, union complete — checked, not assumed (§2.4, exit 31).

Windows are cut per episode by `window_starts(length, context=5, horizon=45)` exactly as `evaluate_rollout`, the ladder and the trust pass cut them. `val` gives the shipped 229 windows over 24 episode-clusters; `train_held` about 740 over 78; `train_probe` about 190 over 20. Each stratum's window and cluster counts are reported.

### 2.2 Per stratum, per cell

One `reference_trajectories` pass (M3d's canonical pass with `keep_trajectories=True`: context filter, RNG snapshot, `imagine` over the real horizon actions, the floor's posterior, the noise reference drawn so the stream is walked as the ladder walks it) with the cell's refit probe, at the record's `context`, `horizon` and `seed`. The probe is fit ONCE per cell, on `train`, exactly as `run_job` fit it; the three strata are three evaluations under that one probe.

`Trajectories` gains one additive field, `band: RolloutResult` — the pass's `_Pass.reference`, which already holds all six mean curves (`rssm / persistence / floor` × position, angle) and which `reference_trajectories` today copies only two curves out of. The thirteen existing fields are unchanged; M3d's tests pin them and the val self-check pins the two curves bitwise.

From the pass, per stratum:

- **The band.** The six mean curves and `gap_closed(h)` on position and angle at every h, via `RolloutResult.position_gap_closed()` / `angle_gap_closed()`; `steps_degenerate` (non-positive band) per metric, as `aggregate.py` counts it. This is the gate's own metric on the gate's own scale, so §4.1 can be read on the train side directly.
- **Probe-free, per window and step** (M3d's two series): `D̂(h)` = `embedding_distance_to_truth`, `D₀(h)` = `embedding_persistence_distance`. From them: `h×_free` = `trust.crossing_step(D̂, D₀, moved)`; the survival curve `S_free(h)` = `trust.survival(h×_free, horizon)` over the (window, seed) draws that moved; `H*_free_q` = `trust.trust_horizon(S_free, q)` for q ∈ {0.5, 0.75, 0.9}; and one **new** per-window statistic, the embedding-space persistence margin `Δ_free(h) = D₀(h) − D̂(h)`, positive when the model beats persistence in embedding space at step h — `trust.persistence_margin(D̂, D₀)`, the same function on the free channel.
- **Probe-based, per window and step** (M3d's, as the control twin): `h×_probe`, `S_probe(h)`, `H*_probe_q`, `Δ(h)` — from `positions`, `positions_at_context`, `true_positions`, through `trust.persistence_margin` and `trust.crossing_step` as `trust_horizon.py` computes them.
- **Moved mask.** `trust.moved_mask(true_positions, true_at_context)`, ground truth at 5 map units, the same rule as M3d; `never_moved` and `not_moved(h)` counted per stratum. `S(h)` inherits M3d's reading: a draw whose window has not yet moved at h survives h vacuously, and the unmoved fraction `u(h)` and conditional survival `S_c(h)` are printed beside it as M3d prints them.

### 2.3 Learning curves

From each record's `history` — `loss` and `parts` (`embedding`, `reward`, `continue`, `kl_dyn`, `kl_rep`), 20,000 entries each — a 100-step moving mean per term. One figure, `learning_curves.png`: one panel per term plus the summed loss, the three arms coloured, the three seeds of an arm as thin lines, `KL_FREE_BITS` drawn on the two KL panels. Per cell, per term, two numbers: the mean over the last quarter (steps 15,001–20,000) against the mean over the preceding quarter (10,001–15,000), as sign and percentage — the statistic the M3b write-up already used on the summed loss — and the step at which the smoothed `embedding` term reached its minimum. **No verdict is attached to this block.** It is the discriminator between budget and objective, read only after Reading G says the gap is not the cause.

### 2.4 Self-check

Run before any reading is printed, per cell, in this order, each with the exit status it fails with:

| exit | check | meaning |
|---|---|---|
| 11 | a checkpoint, a study record and a diagnostic record exist for the cell | `diagnose_dynamics.py`'s `EXIT_NO_CHECKPOINTS` |
| 12 | the split reproduces the record's `episodes` block by name | its `EXIT_SPLIT_MISMATCH` |
| 14 | `evaluate_rollout` on `val` reproduces the record's `curves.rssm_position` **bitwise** | its `EXIT_RECORD_MISMATCH`; a CPU run misses by 6–12 map units, so this is the device check |
| 30 | the `val` stratum's `np.stack(rows).mean(axis=0)` of the model and persistence errors equals the diagnostic's `reference_position` / `persistence_position` at **max \|Δ\| == 0.0** at every h, and its `windows_total` and `window_episode` equal the diagnostic's | `trust_horizon.py`'s `EXIT_SELF_CHECK_FAILED`: the instrument is the ladder's and the trust pass's, or nothing is read |
| **31** | the three strata are pairwise disjoint, their union is `buffer.episode_paths()`, and `train_probe` equals the `used` list `probe_episodes` handed `fit_probes` | new, `EXIT_STRATA_NOT_A_PARTITION`; a CODE defect, not a data one |

Any failure names the cell, the check and (for 14 and 30) the curve and the step, and no reading is printed. 31 is distinct from every other tool's exit status and from argparse's 2, and joins `test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own`.

## 3. How it decides

Stated here, before the run.

### 3.1 Pooling

- **Within a stratum.** Per arm, `pooling.pool_arm` over the three seeds' `CellSeries` built from the per-window arrays: seeds averaged per window, mean over windows, episode-clustered SE over that stratum's clusters. Two masks, one per kind of series. For the survival indicator `1[h× > h]` the `changed` mask is `isfinite(h×)` — the window moved at some step within the horizon, M3d's `S(h)` denominator — so the pooled mean over windows IS `S(h)` (every window carries all three seeds) and the clustered SE is `S(h)`'s ruler. For `Δ(h)` the `changed` mask is the `moved` mask at h; windows not moved at h are excluded and counted, exactly as M3d pools its margin.
- **Between strata.** The strata share no windows, so the contrast is a **new** `pooling.unpaired_contrast(a: PooledMean, b: PooledMean) -> UnpairedContrast`: `estimate = a.mean − b.mean`, `se = sqrt(a.se² + b.se²)`, `z = estimate / se`, `clusters = min(a.clusters, b.clusters)`. z is read against t(23) — the smaller stratum's cluster count, conservative for the larger one.
- **Family.** Reading G makes six clustered contrasts — three arms × two channels (free primary, probe twin) at one horizon; `z_fam = pooling.cluster_threshold(family=6, clusters)` with `clusters` the smaller of the two decision strata's cluster counts — 24 on the shipped split (`val` 24, `train_held` 78). Every contrast in the family is read against it.
- **Per seed as well as pooled.** Every condition is also evaluated within each seed alone (that seed's windows, clustered) and printed.
- **Measurability.** The probe channel pools the cells whose persistence-to-floor band at h=45 is positive on `val` — M3d's rule; all nine shipped cells pass it. The free channel pools every cell.

### 3.2 Horizon

**h = 15** is pre-registered as the decision horizon: it is the governing spec's imagination horizon (§3.5), and on `val` survival is already low there (`S(15)` 0.14–0.27 across arms and channels, M3d), so a memorised train side has room to separate. h = 5 and h = 45 are printed beside it with their z and are not decided on.

### 3.3 Reading G — the generalisation gap

Per arm, the primary statistic is the stratum difference of the survival fraction on the free channel,

`G_free(15) = S_free^train_held(15) − S_free^val(15)`,

the difference in the fraction of moved (window, seed) draws still ahead of persistence in embedding space at 15 imagined steps, pooled as §3.1. Its twin is `G_probe(15)`, the same on the probe channel. The stratum difference of `Δ_free(15)` is printed as the continuous companion in the loss's units and is not decided on.

Verdict per arm, in this precedence, each printed with the rule that produced it:

| status | rule | read |
|---|---|---|
| `UNRESOLVED_PROBE` | `G_free(15)` and `G_probe(15)` both clear `z_fam` with **opposite** signs | the two channels contradict; nothing below is read for this arm |
| `MEMORISATION` | `G_free(15) > 0` with z > `z_fam` pooled **and** within ≥ 2 of 3 seeds, **and** `train_held` passes spec §4.1 — `gap_closed(45)` on position > 0 in all three seeds of the arm, where a NaN (non-positive band) is not > 0 | the model learned dynamics for the episodes it saw and they do not transfer; the fix is data |
| `PARTIAL_GAP` | `G_free(15)` clears as above, but `train_held` fails §4.1 | some memorisation, yet it cannot beat persistence at 45 steps even on its own training data; data **and** budget/capacity/objective |
| `INVERTED_GAP` | `G_free(15)` z < −`z_fam` | train worse than val; reported, not interpreted |
| `NO_GAP` | none of the above | train and val indistinguishable at 15 steps; the failure is not generalisation — read §2.3 |

One pooled status per arm and one per seed; the arm's status is the pooled one. Three arms disagreeing is a result, not a failure, and is written as such.

### 3.4 Sensitivity — changes no verdict

Two lines, M3d's pattern: every probe-channel statistic recomputed with cells of probe selection R² < 0.1 excluded (`pixel_ae`/s1 at 0.018 and no other shipped cell); and `G_free` and `G_probe` at h = 5 and h = 45 with their z, so a reader can see whether the horizon choice carried the verdict.

## 4. What it does not claim

- It does not change the M3 gate, `report_study.py`, `aggregate.py`, or any verdict M3b, M3c or M3d recorded. `beats_persistence` at h=45 on `val` stays the recorded 45-step stress test.
- It does not rank arms. Every contrast is within-arm, train against val; a between-arm difference in `G` is printed and not tested.
- A `MEMORISATION` reading says the next fix is data. It does not say how much, which collection policy, or on which scenario — and the M4 scenario decision (governing spec §6 M5's `basic` → `defend_the_center` → `deadly_corridor` ladder) is a separate one that this reading feeds.
- `train_probe` is confounded by construction — the probe saw those episodes — and is never a decision input.
- Probe-based numbers inherit the probe's R² (0.018–0.383) and a floor larger than the displacement they read; they are comparable within an arm between strata because both strata pass through one probe, and they are not trustworthy in absolute units, which is why the decision channel is probe-free.
- Everything is `my_way_home`'s 122 episodes at context 5 / horizon 45, one split, the M3c checkpoints. A different environment, split or horizon is a different measurement.

## 5. Shape of the code

Four units, one job each, and one additive field.

**`src/mbfps/eval/probe.py`** — `PROBE_EPISODE_LIMIT = 20`; `probe_episodes(paths, limit=PROBE_EPISODE_LIMIT) -> tuple[list, list]` returning `(list(paths)[:limit], list(paths)[limit:])`; `fit_probes` takes its `used` from it. Pinned by a test that moves the constant and requires `fit_probes`'s fit set and `probe_episodes`'s `used` to follow together — equality to 20 would pass again the moment someone re-hardcodes the literal, so the test asserts the coupling, as `VAL_FRACTION`'s does.

**`src/mbfps/eval/diagnostics.py`** — `Trajectories.band: RolloutResult`, set by `reference_trajectories` from `result.reference`. No other change; the pass's numbers do not move.

**`src/mbfps/eval/pooling.py`** — `UnpairedContrast` dataclass (`estimate`, `se`, `z`, `clusters`, `windows_a`, `windows_b`) and `unpaired_contrast(a, b)` as §3.1 defines them, beside `paired_contrast`. Pinned by arithmetic on fabricated `PooledMean`s: two identical means give 0 with the combined SE; a zero SE on one side gives the other's SE; `clusters` is the smaller.

**`src/mbfps/eval/split_gap.py`** (new) — pure over numpy arrays and dicts, no torch, no files:
- `STRATA = ("val", "train_held", "train_probe")`, `DECISION_H = 15`, `REPORTED_H = (5, 15, 45)`, `Q_REPORTED = (0.5, 0.75, 0.9)`, `FAMILY = 6`.
- `strata_partition(all_paths, train, val, used) -> dict[str, list]` — the three lists; raises `StrataNotAPartition` (the script maps it to 31) if not disjoint, not complete, or `train_probe != used`.
- `stratum_summary(traj: Trajectories, horizon) -> dict` — per stratum: the six curves, `gap_closed` per metric, `steps_degenerate`, `moved`, `h×` both channels, `Δ` both channels, `S(h)`, `H*_q`, `u(h)`, `S_c(h)`, the counts. Calls `trust.moved_mask`, `trust.crossing_step`, `trust.persistence_margin`, `trust.survival`, `trust.trust_horizon`.
- `reading_gap(inputs: GapInputs) -> GapReading` — §3.3's rules over pooled tables: per arm, `G_free`, `G_probe`, the §4.1 unanimity on `train_held`, the per-seed agreement, the status and the rule that produced it. Each rule is a mutation the others cannot catch: a free contrast that clears with the probe twin clearing opposite → `UNRESOLVED_PROBE`; clears in 3 of 3 seeds with `train_held` unanimous → `MEMORISATION`; the same with one `train_held` seed at `gap_closed(45) ≤ 0` → `PARTIAL_GAP`; clears pooled but in 1 of 3 seeds → `NO_GAP`; clears negative → `INVERTED_GAP`; an interval covering 0 → `NO_GAP`.
- `learning_curve_summary(history: dict, window: int = 100) -> dict` — §2.3's per-term quarter means, sign and percentage, and the smoothed embedding minimum's step. Pinned on a synthetic history with known quarter means and a known minimum.

**`scripts/split_gap.py`** (new) — `--out` (default `runs/m3_study_v2`), `--data` (default `data/my_way_home`), `--device`, `--arms`, `--seeds`, `--context` / `--horizon` (defaults read from each cell's diagnostic, as `trust_horizon.py` reads them; a mismatch with the record refused as `diagnose_dynamics.py` refuses it), `--figure` (default `<out>/learning_curves.png`), `--window` (the learning curves' moving-mean width, default 100). Loading is `trust_horizon.py`'s (`load_cell`: checkpoint, record, diagnostic, split by name, `fit_probes` on `train` at the cell's seed and protocol). Per cell: the §2.4 checks in order; three `reference_trajectories` passes, `val` first; `stratum_summary` on each; `split_gap_<arm>_seed<n>.json` with `git_sha` (= HEAD, tree clean under `src/` and `scripts/`), `device`, torch version, the checkpoint's `git_sha`, `context`, `horizon`, `split_seed`, the strata's episode names, and the self-check deltas. Then pooling, Reading G, the sensitivity lines, the learning-curve summary and figure, printed to stdout and `split_gap.txt` in `trust.txt`'s style: the self-check table, the strata table, the per-arm block per stratum, the contrasts with z and `z_fam`, the verdict lines with their rules, the sensitivity block, the learning-curve table. Exit **0** when every cell's checks hold and the reading printed; **11 / 12 / 14 / 30** with their existing meanings; **31** new.

**Tests.** `tests/eval/test_split_gap.py`: `strata_partition` (the three lists on a synthetic split; each of the three failure modes raises); the `PROBE_EPISODE_LIMIT` coupling test; `unpaired_contrast` arithmetic; every §3.3 rule on fabricated pooled tables, one mutation at a time; `learning_curve_summary` on a synthetic history. `tests/eval/test_split_gap_script.py`: `main()` on a **30-episode** variant of the eval fixture (`conftest.small_buffer`'s construction parametrised by count: train 24 → `train_probe` 20, `train_held` 4, val 6), built the way `test_trust_horizon_script.py` builds its cell — `run_job(..., steps=5, seq_len=4, context=2, horizon=3, device="cpu")`, then `diagnose_dynamics.main` to write the diagnostic the self-check needs — asserting the record's fields, `band` present on `Trajectories`, the exit statuses in order (a missing diagnostic, a doctored `reference_position`, a stratum list doctored to overlap), the figure written, and the verdict lines present. A mutation table with the harness self-checked three ways, as every M3 task's.

**Record.** Results go in the plan written from this spec, `docs/superpowers/plans/2026-09-16-mb-fps-m3e-split-gap.md`, in its own results section, numbers copied not rounded, the closing paragraph stated against §4's non-claims. `NO_GAP` on every arm is a result. The M3c and M3d plans are closed records and are not amended.

## 6. Files

| file | change |
|---|---|
| `src/mbfps/eval/probe.py` | `PROBE_EPISODE_LIMIT`, `probe_episodes`; `fit_probes` routes through it |
| `src/mbfps/eval/diagnostics.py` | `Trajectories.band`; `reference_trajectories` sets it |
| `src/mbfps/eval/pooling.py` | `UnpairedContrast`, `unpaired_contrast` |
| `src/mbfps/eval/split_gap.py` | new: the pure functions and constants of §5 |
| `scripts/split_gap.py` | new: load, checks, three passes, records, pooling, Reading G, figure, `split_gap.txt` |
| `tests/eval/conftest.py` | the fixture's construction takes a count; `small_buffer` stays six |
| `tests/eval/test_split_gap.py`, `tests/eval/test_split_gap_script.py` | new |
| `tests/eval/test_diagnose_dynamics_script.py` | the exit-status distinctness test gains `split_gap` |
| `tests/eval/test_probe.py` | the `PROBE_EPISODE_LIMIT` coupling test |
| `runs/m3_study_v2/split_gap_*.json`, `split_gap.txt`, `learning_curves.png`, `split_gap.log`, `split_gap.exit` | outputs; gitignored like everything under `runs/` |
