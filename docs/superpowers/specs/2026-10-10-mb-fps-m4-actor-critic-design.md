# MB-FPS M4 — Actor-critic in imagination, on `basic`, at H = 15 (Design)

**Date:** 2026-10-10
**Status:** design approved in conversation 2026-10-10 (task `basic`; staged one cell then 3×3;
approach A of three — one spec, three gated stages; sections 1–3 approved in turn, including the
symlog amendment to Stage A). Awaiting review of this written spec.
**Base:** `main` at `4956f59` (PR #14, M3n, merged).
**Governs:** milestone M4 of `docs/superpowers/specs/2026-09-01-mb-fps-design.md` §6, with the
changes §2 lists.
**Reads:** the ViZDoom `basic` scenario (new data, collected here); for the known-negative control
only, `runs/m3_study_v2` (`frozen_ssl` seed 0) and `data/my_way_home`.
**Writes:** `data/basic/`; `runs/m4a_entry/`, `runs/m4b_agent/`, `runs/m4c_study/`; one results
section per stage in this milestone's plan.

---

## 1. Why M4 now, and why not after the M3 gate

M3's exit gate has never passed. Thirteen milestones (M3a–M3n) have narrowed why without moving it.
Three recorded facts say the gate is the wrong prerequisite for M4:

1. **The horizon is not M4's.** The gate reads `gap_closed` at h = 45 (about 5.1 s). The actor-critic
   imagines **H = 15** (governing spec §3.5; about 1.7 s at `frame_skip = 4`), which is also
   DreamerV1–V3's horizon. At h = 15 the gate's quantity is positive in only **1 of 9** cells
   (M3n's k = 45 table), so shortening the horizon does not pass it either.
2. **The quantity is not M4's.** The actor optimises imagined *reward*, discounted by the imagined
   *continue* probability. It never reads position. On `my_way_home` the reward head is a constant:
   R² between −0.0007 and +0.0015 in all nine cells, against a validation split holding **2** goal
   events (`runs/m3_study_v2/report.txt`, "reward prediction"). Only **16 of 122** collected episodes
   ever reach the goal. A policy trained in that imagination has no signal to follow, however well
   the model tracks position.
3. **The ruler cannot resolve the quantity at the horizon that matters.** The position probe's
   readout error is 131–259 map units (M3n). The agent's true median displacement over 15 steps
   is 34–72 units (measured over all 122 `my_way_home` episodes, both policies).

M4 therefore moves to a task whose reward the world model can learn, and replaces the position gate
(as M4's entry condition) with a check on the quantities the actor actually consumes.

M3's gate, verdicts and diagnostic results are untouched. Its investigation continues as research,
not as a blocker.

---

## 2. Changes to the governing spec

| governing spec | this spec | reason |
|---|---|---|
| M4 trains on M3's frozen `my_way_home` world model | M4 collects `basic` data and trains new world models on it | §1 item 2 |
| M4's entry condition: M3's exit gate | Stage A's control-relevant entry check (§4.5) | §1 items 1–3 |
| `basic` first appears at M5 | `basic` is M4's task; M5 keeps its ladder (`basic` → `defend_the_center` → `deadly_corridor`) | unchanged in order, earlier in time |
| Reward head: MSE on raw reward | MSE on `symlog(reward)` | §4.3 |
| Loader: episodes shorter than `seq_len` are never sampled | padded and masked (§4.2) | §3 |

Unchanged: imagination horizon 15, γ 0.997, λ 0.95, actor/critic learning rates 3e-5, entropy
3e-4, critic EMA 0.98, head MLPs 2 × 512, fp32 on MPS, and the architectural invariant that
`encoders.py` is the only module that differs between arms.

---

## 3. Measured facts about `basic`

The scenario config (`vizdoom/scenarios/basic.cfg`): buttons `MOVE_LEFT`, `MOVE_RIGHT`, `ATTACK`
(4 actions with our no-op); `living_reward = -1` per tic; `episode_timeout = 300` tics, so at most
**75 steps (about 8.6 s)** at `frame_skip = 4`. One step is 4 tics of a 35 Hz engine, about 114 ms.

Measured over 200 episodes per policy (seeds 0–199, the existing `RandomPolicy` and
`ScriptedPolicy`, `frame_skip = 4`; a scratch measurement, not a recorded result):

| | `random` | `scripted` (shipped) |
|---|---|---|
| episodes ending in a kill | **55%** | 27% |
| median episode length | 59 | 75 (timeout) |
| median length of kill episodes | **20** | 6 |
| episodes shorter than 64 steps | **52%** | 27% |
| episodes shorter than 20 steps | 27% | 25% |
| mean return (sd) | −170.7 (186.2) | −233.6 (183.1) |

Per-step rewards observed under `random` (200 episodes): −4 (a step), −9 (a step with a missed
shot), and **+97 to +99** on the step that kills. **The kill and the termination are not always the
same step:** in 19 of 109 kills the kill step reports `terminated = False` at +97, and the next step
reports `terminated = True` at −1. The continue head and every reading below must handle a kill
followed one step later by the episode's end.

Two consequences shape the design:

- **The shipped loader would discard most kills.** `SequenceLoader` samples only episodes at least
  `seq_len = 64` long (`src/mbfps/data/loader.py:174`). Kill episodes have a median length of 20.
- **The shipped `scripted` policy does not fit `basic`.** Its sweep reverses every
  `_SWEEP_LEN = 100` steps, longer than any `basic` episode, so it only ever moves left. It has no
  `MOVE_FORWARD` on this button set, so it also never "advances".

---

## 4. Stage A — the world model on `basic` (one cell: `frozen_ssl`, seed 0)

### 4.1 Data

- Collect `data/basic/` with the existing collector, mixing `random` and `scripted` as M1 did, with
  the policy tag recorded per episode.
- `ScriptedPolicy` gains a `sweep_len` constructor argument (default 100, the shipped behaviour).
  `scripts/collect.py` gains `--sweep-len`; `basic` is collected with **10**.
- Capacity: the collector default, 60,000 transitions. At `random`'s median length of 59 that is
  about 1,000 episodes and, at a 55% kill rate, several hundred kill events, against 16 goal events
  in `my_way_home`. The actual counts are recorded, not assumed.
- The held-out episode split and the frozen-feature cache follow M1: the cache is generated once,
  on one device, and reused by every stage.
- The dataset is **fixed** for all of M4. Nothing the agent does is written back to it.

### 4.2 Short episodes: pad and mask

An episode shorter than `seq_len` is sampled from start 0 and padded to `seq_len` with zeros. The
batch carries a boolean `mask` of shape `(B, T)`, true on real transitions. Episodes at least
`seq_len` long are sampled exactly as before, with `mask` all true.

Every world-model loss term (embedding, reward, continue, KL) becomes a masked mean over real
transitions. A padded transition contributes nothing to any loss or gradient.

Each sequence stays inside one episode, so the RSSM needs no state-reset logic. `seq_len` stays 64.
Dreamer's alternative, concatenating episodes with an `is_first` reset, changes the RSSM. It is out
of scope unless masking is shown to be insufficient.

**Sampling weight.** As shipped, the loader picks among eligible episodes uniformly. With every
episode now eligible, that stays uniform per episode, so a 6-step episode is drawn as often as a
75-step one. That is a deliberate over-weighting of short (mostly kill) episodes, and it is recorded
in the plan's results rather than corrected silently.

### 4.3 Reward head on `symlog`

The reward head is trained on `symlog(r) = sign(r) · ln(1 + |r|)` and read through
`symexp(y) = sign(y) · (exp(|y|) − 1)`. On `basic`, −4 maps to −1.61 and +97 to +4.58, so the kill no
longer dominates the squared error by three orders of magnitude.

This applies to every arm, so the architectural invariant holds. A checkpoint records which reward
transform its head was trained with (`reward_transform: "identity" | "symlog"`). Every consumer reads
the head through that field, never by assumption: the M3 checkpoints are `identity`.

### 4.4 Training

The existing M3 training code, extended only by §4.2 and §4.3: `frozen_ssl`, seed 0, 20,000 steps,
batch 16 × 64, `free_bits = 0.20` (M3 §3.2a). M3c's measured 88–95 minutes per cell is the
estimate. The actual time is recorded.

### 4.5 The entry check: does imagination predict reward over 15 steps?

The check reproduces the start states M4 will use.

**Procedure, per held-out episode** of `T` transitions (`obs` has `T + 1` rows):

1. Filter the whole episode with `observe`, pairing `encoder(obs[1:T+1])` with `actions[0:T]` (the
   RSSM's action-time convention, `src/mbfps/models/rssm.py:141`). Posterior state `s_i` has seen
   frames up to `obs[i+1]`.
2. From **every** `s_i` with `i < T − 1`, imagine `H_i = min(15, T − 1 − i)` steps with `imagine`,
   driven by the recorded actions `actions[i+1 : i+1+H_i]`.
3. The imagined step `j` (0-based) predicts the reward of `actions[i+1+j]`, i.e. `rewards[i+1+j]`,
   and the continue target of that same transition.

**Reading R (return).** For each start `i`:

- imagined return `Ŕ_i = Σ_j symexp(reward_head(ẑ_j))` over the `H_i` imagined steps
- real return `R_i = Σ_j rewards[i+1+j]` over the same steps

The rewards are undiscounted, and continue is deliberately left out, so this reading isolates the
reward head under imagination. Then:

    R2_return = 1 − Σ (R_i − Ŕ_i)² / Σ (R_i − c_R)²

where `c_R` is the mean of `R_i` over the **training** episodes' starts. That makes the denominator
the error of the best constant known before seeing validation data, so a constant predictor scores at
or below zero.

**Reading C (continue).** Over every imagined step:

- `p` is the predicted continue probability (sigmoid of the continue logit)
- `c` is the continue target (`continue_target`, `src/mbfps/models/heads.py:25`): 1 until the
  terminating transition, 0 on it

Then:

    brier_skill = 1 − mean (p − c)² / mean (c_C − c)²

where `c_C` is the training-set continue rate.

**Intervals.** Both readings take an **episode-clustered bootstrap**: held-out episodes are
resampled with replacement, 2,000 resamples at a fixed seed, using `pooling.episode_bootstrap` and
`pooling.percentile_interval` (`src/mbfps/eval/pooling.py:670`, `:687`), with a 95% percentile
interval.

**Gate.** Stage A passes if **both** `R2_return` and `brier_skill` have `ci_low > 0`.

**Reported, not gated:**

- **The posterior reference.** The same two readings with the heads applied to the *posterior*
  states `s_{i+1..}` instead of imagined ones, i.e. teacher-forced. This attributes a failure:
  - posterior fails → the head or the data is the problem
  - posterior passes and imagination fails → the problem is imagination at H = 15
- The kill and termination counts in the held-out split, and how many starts each reading used.
- Position `gap_closed` at h = 15, for continuity with M3. It does not gate anything.

### 4.6 Known-negative control

The identical check, run on `runs/m3_study_v2`'s `frozen_ssl` seed 0 checkpoint over
`data/my_way_home`'s held-out split (reading its reward head as `identity`), **must fail Reading R**:
`ci_low ≤ 0`. Its reward head is recorded at R² ≈ 0.

If that control passes, the check is broken, and **no Stage A result may be reported**. Reading C on
the control is reported, not required, because its held-out split may hold too few terminations to
read at all.

### 4.7 What each Stage A outcome licenses

| outcome | licenses |
|---|---|
| both readings pass, control fails | Stage B |
| either reading fails; posterior reference also fails | a head or data fix (more kills, loss weighting) before any actor-critic. Stage B does not run |
| either reading fails; posterior reference passes | imagination at H = 15 is the obstacle on `basic`. A shorter horizon is a **new decision for the author**, recorded in a spec amendment, never an automatic fallback. Stage B does not run |
| control passes | the instrument is broken; nothing is read |

---

## 5. Stage B — actor-critic in imagination (one cell)

### 5.1 Components

`Actor` and `Critic` in `src/mbfps/models/heads.py`, where the governing spec §4 places them. Each is
a 2 × 512 SiLU MLP over the latent (`h` and `z` together, `LATENT_DIM`):

- **Actor:** outputs logits of a categorical over the scenario's actions.
- **Critic:** outputs a scalar value. A slow copy, updated by EMA with rate 0.98, serves as the
  bootstrap target.

Both are seeded through `seeded_init` like every other module.

### 5.2 Imagination with the actor in the loop

`RSSM.imagine_policy(actor, state, horizon)` advances one step at a time. At each step it takes the
current latent, samples `a ~ actor(latent)`, steps the cell, and samples `z` from the prior. It
returns the latents, the actions and the actions' log-probabilities and entropies. `imagine` is left
exactly as it is: M3's diagnostics depend on it.

**Start states:** every *real* (masked-true) posterior state of a replay batch drawn from the
training split, flattened (up to 16 × 64 = 1,024). Padded positions are never start states.

**The world model is frozen.** It is loaded from Stage A and its parameters are set to
`requires_grad = False`. The actor-critic optimiser holds only actor and critic parameters.

### 5.3 Returns and losses

Over the H = 15 imagined steps:

- `r_t` is `symexp(reward_head(latent_t))`, the head read through its recorded transform.
- `γ_t = 0.997 · σ(continue_logit_t)`, so a predicted episode end cuts the return.
- The bootstrap at step H is `V_slow(latent_H)`.

λ-returns, with λ = 0.95:

    G_t = r_t + γ_t · [(1 − λ) · V_slow(latent_{t+1}) + λ · G_{t+1}],   G_H = V_slow(latent_H)

**Critic loss:** squared error of `V(latent_t)` against `stopgrad(G_t)`, weighted by the cumulative
continue product, so imagined steps past a predicted end count less.

**Actor loss:** REINFORCE, DreamerV3's estimator for discrete actions:

    −stopgrad((G_t − V(latent_t)) / S) · log π(a_t | latent_t) − 3e-4 · entropy(π(· | latent_t))

where `S = max(1, P95(G) − P5(G))`. The percentiles are tracked as an EMA over batches with rate 0.99,
DreamerV3's return normalisation. The same continue weighting applies.

The **imagination-isolation test** (governing spec §7) gates Stage B: after an actor-critic update,
every world-model parameter's `.grad` is `None`.

### 5.4 Live evaluation

The live game is used **only** to evaluate. No real-environment transition ever reaches a gradient
or the dataset.

**Deployment:**

1. At reset, the state starts at `initial_state`.
2. The first frame is filtered as if a no-op (action index 0) had produced it. That matches how every
   training sequence begins: an initial state, then an action and the frame it led to.
3. At each step the actor **samples** an action from the posterior latent, as it was trained.
4. The resulting frame is encoded and filtered with that action.

The deterministic mode (argmax) is reported alongside, not gated.

**Seeds:** 100 evaluation episodes on fixed seeds **1,000,000–1,000,099**, disjoint from every
collection seed. The random baseline is `RandomPolicy` on the **same 100 seeds**, reseeded per
episode with that episode's seed, as the collector does. The agent's action sampling is seeded the
same way, so a rerun reproduces every episode.

**Per-episode record:** return, length, whether it ended in a kill, and the imagined λ-return at the
episode's first state.

### 5.5 The Stage B gate (pre-registered)

`d_s = return_agent(s) − return_random(s)` for each of the 100 seeds. Resample the seeds, 2,000
resamples at a fixed seed, and take the 95% percentile interval of `mean(d)`.

**Stage B passes if `ci_low > 0`, and the imagination-isolation test is green.**

The governing spec's M4 gate is "beats a random policy when deployed live". This makes "beats"
precise. Random's measured spread (sd 186 per episode) gives a standard error of about 19 over 100
episodes. A policy that kills within about 3–5 steps would score roughly +80 to +90 (an estimate
from §3's per-step rewards, not a measurement). So the gate can only pass for a real effect, and it
is not set unreachably high.

**Reported, not gated:**

- kill rate
- median episode length
- the argmax-mode return
- **the imagined-vs-real gap**: mean imagined λ-return at episode start against mean real return. A
  large positive gap is the signature of an actor exploiting the model's errors (governing spec §5.2)

### 5.6 Known-answer control

An **untrained** actor (seeded initialisation, zero updates) is run through §5.4 and §5.5 unchanged
and **must fail** the gate. If it passes, the evaluation is broken and no Stage B result may be
reported.

### 5.7 Budget and schedule

- **Default:** 10,000 actor-critic updates. Live evaluation (§5.4) runs at updates 2,000, 4,000,
  6,000, 8,000 and 10,000.
- **The gate is read at update 10,000 only**, fixed now, not at whichever checkpoint looks best. The
  intermediate evaluations are a learning curve, not a selection set.
- The wall-clock time per update and per live episode is **measured in the plan before the run**. If
  10,000 updates would exceed one overnight run, the budget is revised in the plan *before* the run
  and recorded with its reason. It is never revised after results are seen.

---

## 6. Stage C — the 3 × 3 study

### 6.1 Cells

Stage A and Stage B code, unchanged, on `frozen_ssl`, `pixel_ae` and `random_vit` × seeds 0, 1, 2,
all on the one `data/basic/` dataset.

- `pixel_ae` first needs its autoencoder trained on `basic` (the M2 step, existing code).
- Each arm's feature cache is generated once, on one device.
- Stage A's seed-0 `frozen_ssl` cell is reused, not retrained, if the code has not changed since. Its
  `git_sha` is checked.

**The per-cell entry check is reported, not gating.** How well an encoder supports reward and
continue prediction is part of the research result. A cell that fails §4.5 still gets an
actor-critic run, and its entry readings are printed beside its live return.

### 6.2 The cross-arm reading (pre-registered)

A cell's score is its mean live return over the 100 evaluation seeds at update 10,000.

**Arm X ranks above arm Y** only if every seed of X scores above every seed of Y: 9 of 9 pairwise
wins. That is the extreme outcome of a 3 vs 3 Mann–Whitney test, one-sided p = 1/20 = 0.05, the most
three seeds can support (the same reasoning as M3 §4's unanimity rule).

Any other pattern reads **`NOT_SEPARATED`** for that pair, which is a legitimate, reportable result.
The reading prints all three pairs.

### 6.3 The Stage C gate

All nine cells present, no duplicates, every cell's Stage B readings (§5.5) and entry readings (§4.5)
reported, and the §6.2 reading printed.

Stage C does **not** require any arm to pass Stage B or to separate. A gate that only passes if the
hypothesis is confirmed would make the study unfalsifiable (M3 §4).

---

## 7. Module and file changes

| file | change |
|---|---|
| `src/mbfps/data/policies.py` | `ScriptedPolicy(sweep_len=100)` |
| `scripts/collect.py` | `--sweep-len` |
| `src/mbfps/data/loader.py` | pad and mask (§4.2); batches carry `mask` |
| `src/mbfps/models/transforms.py` (new) | `symlog`, `symexp` |
| `src/mbfps/training/world_model.py` | masked losses; reward head on `symlog`; checkpoint records `reward_transform` |
| `src/mbfps/models/rssm.py` | `imagine_policy` (new method; `imagine` unchanged) |
| `src/mbfps/models/heads.py` | `Actor`, `Critic` |
| `src/mbfps/training/ac_trainer.py` (new) | λ-returns, return scale, losses, training loop, checkpoint ladder |
| `src/mbfps/eval/entry.py` (new) | §4.5 readings and §4.6 control |
| `src/mbfps/eval/live.py` (new) | §5.4 deployment, §5.5 gate, §5.6 control |
| `scripts/entry_check.py`, `scripts/train_actor_critic.py`, `scripts/eval_live.py`, `scripts/m4_study.py` (new) | stage drivers; each `read` phase exits non-zero on a failed gate with a distinct code per outcome, as M3n's `motion_headroom.py` does |

Only `encoders.py` differs between arms. Every file above is arm-invariant.

---

## 8. Testing

Each new piece gets a known-answer test:

- **Pad and mask:** the masked world-model loss on a short episode padded to `seq_len` equals the
  loss on the same episode unpadded. Padded positions receive zero gradient.
- **Loader eligibility:** an episode of length 6 is sampled; before the change it was not.
- **`ScriptedPolicy(sweep_len=10)`:** reverses direction every 10 steps on `basic`'s button set. The
  default reproduces the shipped action stream exactly.
- **`symlog`/`symexp`:** round-trip to float tolerance across −9 … +101, and are odd functions.
- **`reward_transform`:** a checkpoint without the field is read as `identity`, and an M3 checkpoint
  loads unchanged.
- **Entry readings:** a head returning the true rewards scores `R2_return = 1`; a constant equal to
  `c_R` scores `≤ 0`; a kill followed by a separate termination step is scored on the right step for
  both readings.
- **`imagine_policy`:** with an actor fixed to a given action sequence, its latents match `imagine`'s
  exactly under the same seed.
- **λ-returns:** match a hand-computed 4-step example, including a predicted end at step 2.
- **Return scale:** `S` is 1 when `P95 − P5 < 1`, and the EMA matches a hand-computed sequence.
- **Imagination isolation:** after an actor-critic step, every world-model `.grad` is `None`, and the
  world model's parameters are bitwise unchanged.
- **Live gate:** passes on a synthetic policy that is clearly better per seed, and fails on one
  identical to the baseline.
- **Untrained-actor control:** fails the gate (§5.6), checked in the test suite with a short run.
- **Determinism:** a fixed seed and fixed weights reproduce the same imagined trajectory hash (the
  golden-rollout pattern from M3).

The suite must be green at the commit each stage's run is taken from, recorded with its counts and
the SHA, as in every M3 milestone.

---

## 9. Compute

| step | estimate |
|---|---|
| `basic` collection, 60,000 transitions | minutes (200 episodes took 6 s, without the feature cache) |
| world model, one cell | ~90 min (M3c's 88–95 min per cell) |
| actor-critic, one cell | **measured in the plan first**, then budgeted (§5.7) |
| live evaluation, 100 + 100 episodes × 5 checkpoints | ≤ 75 steps each, one encoder pass per frame; **measured** |
| Stage C world models | ~13.5 h for nine cells, plus the `pixel_ae` autoencoder |

Stage C fits in about two overnight runs on the M4. The cloud budget is not needed for M4.

---

## 10. What M4 does not answer

- **Whether the world model tracks a shifting, policy-induced distribution.** That is M5. The
  dataset here is fixed by design.
- **Whether the `my_way_home` world models support control.** M4 does not use them, except as §4.6's
  known-negative control.
- **The M3 exit gate.** Unmoved, and still failing. M4 neither re-reads it nor depends on it.
- **Whether the arm ranking transfers** to harder scenarios or to the online loop. §6.2's reading is
  about `basic`, a fixed dataset, and 10,000 updates.
- **Model-free comparison** at equal environment steps. That is M5 (governing spec §6).

---

## 11. Open questions

1. **Collection mix.** M1 used an equal `random` / `scripted` split. On `basic`, `scripted` (even
   with `sweep_len = 10`) may kill less often than `random`. The plan measures both policies' kill
   rates with the new sweep before collecting and records the mix chosen. The mix is decided before
   Stage A trains and is not tuned afterwards.
2. **Sampling weight** (§4.2). If Stage A's posterior reference fails on Reading R, length-weighted
   sampling is the first thing to examine. It is not changed pre-emptively.
