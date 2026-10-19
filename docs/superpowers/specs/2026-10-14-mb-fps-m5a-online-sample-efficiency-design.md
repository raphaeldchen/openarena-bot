# MB-FPS M5a — Close the loop on `basic`: is model-based learning more sample-efficient than model-free? (Design)

**Date:** 2026-10-14
**Status:** design approved in conversation 2026-10-14 (goal: sample efficiency on `basic` first; baseline
reads the same frozen DINOv2 features; DQN-family baseline; 100,000 environment steps × 5 seeds per arm;
primary metric the area under the evaluation curve; approach A of three — one synchronous interleaved
loop; sections 1–4 approved in turn). Awaiting review of this written spec.
**Base:** `main` at `fceccb9` (PR #16, M4, merged); branch `feat/m5a-online-loop`.
**Governs:** the first part of milestone M5 of `docs/superpowers/specs/2026-09-01-mb-fps-design.md` §6.
The scenario ladder (`defend_the_center`, `deadly_corridor`) and the encoder comparison are M5b.
**Reads:** the ViZDoom `basic` scenario, live; nothing from M4's datasets or checkpoints.
**Writes:** `runs/m5a/<arm>_seed<n>/` (replay, checkpoints, evaluation records), `runs/m5a/study.txt`;
one results section in this milestone's plan.

---

## 1. Why, and what M5a decides

The governing spec's stated motivation is reduced sample inefficiency, and its M5 rationale names the gap:
"Without a model-free comparator at equal environment steps, that is an assertion rather than a result."
M4 showed a policy trained purely in imagination plays `basic` — 9 of 9 cells beat a random policy live
(M4 Stage C) — but every M4 world model was trained on a fixed 53,966-transition dataset, so M4 says
nothing about how many environment steps the method needs.

M5a asks one question: **at equal environment steps, does an online Dreamer agent learn `basic` faster
than a model-free agent reading the same representation?**

Held fixed so the answer is attributable:

- **One representation.** Both arms read the same frozen DINOv2 patch features through the same trainable
  `BottleneckEncoder` class (M4's `frozen_ssl` arm). Only "learn in a world model's imagination" versus
  "learn from real steps" differs.
- **One budget.** 100,000 environment steps per run, counted from the first step, including the random
  prefill. Nothing is warm-started from M4: reusing M4's dataset would give one arm 53,966 free steps.
- **One gradient-step budget.** One update per 5 environment steps in both arms.
- **One evaluation.** The same 100 live seeds, at the same checkpoints, for both arms.

M4's §10 deferred two questions to M5; M5a answers the first ("whether the world model tracks a shifting,
policy-induced distribution") as a by-product and the second ("model-free comparison at equal environment
steps") directly.

---

## 2. Changes to earlier specs

| earlier spec | this spec | reason |
|---|---|---|
| Governing §6 M5: loop + baseline + three-scenario ladder in one milestone | M5a: loop + baseline on `basic`; the ladder and the encoder comparison are M5b | scope; §1 |
| Governing §6 M5: "a model-free baseline (PPO or DQN)" | Double DQN with 3-step returns, on the same features | the stronger sample-efficiency comparator; §5 |
| M4 trainer: reward readout unbounded; terminal start states imagined at full weight | two opt-in amendments, on in M5a, off by default (§4.3) | M4 final review; M4's records stay reproducible |
| M4: world model frozen during actor-critic training | the world model trains online, interleaved with the actor-critic | the loop; §3 |

Unchanged from M4: `frame_skip = 4`; `basic`'s 4 actions; the world model's architecture, losses, learning
rate (1e-4), `free_bits` 0.20, symlog reward head, padded-and-masked short episodes, batch 16 × 64;
the actor-critic's H = 15, γ 0.997, λ 0.95, learning rates 3e-5, entropy 3e-4, critic EMA 0.98, return-scale
EMA 0.99; the live evaluation seeds 1,000,000–1,000,099; `encoders.py` the only arm-varying module.

---

## 3. The shared online loop

### 3.1 Structure

Both arms run one driver, `train_online(arm, seed, budget=100_000)`: one process, seeded, deterministic.

1. **Prefill.** 5,000 environment steps with `RandomPolicy`, reseeded per episode from the run seed. Both
   arms' prefills for the same seed are identical (same seeds, same policy, same engine).
2. **Main loop**, until the budget is spent:
   - **act** for 100 environment steps with the arm's current behaviour policy (§4.2, §5.3), encoding each
     frame live with frozen DINOv2 and appending each finished episode to the replay;
   - **train**: 20 update rounds (§4.1, §5.2).
3. **Evaluate** (§6) at step 0 (after the prefill, before any update — the control checkpoint) and at
   every 10,000 budget steps: 10k, 20k, …, 100k.

**Budget accounting.** A budget step is one `env.step` on the training environment, prefill included.
Evaluation runs on a separate environment instance; its steps are not counted and never enter the replay.
An episode in progress when an act phase ends continues into the next act phase. The run ends at exactly
100,000 budget steps; an episode cut off by the budget is stored truncated.

### 3.2 Train ratio

One update round per 5 budget steps in both arms: 20 rounds per 100 steps, about 19,000 rounds after the
5,000-step prefill. For Dreamer a round is one world-model step plus one actor-critic step; for DQN, one
Q-network step. Both arms therefore take the same number of gradient steps per real step.

Measured M4 costs (one world-model step plus one actor-critic step ≈ 0.66 s on this machine) put a Dreamer
run at about 3.7 h of updates, ~15 min of collection (DINOv2 at ~130 frames/s) and ~5 min of evaluation —
about 4 h per run, 20 h for five seeds. In DreamerV3's units this is a replay ratio of about 205 replayed
timesteps per environment step (1,024 per round ÷ 5).

**The two arms replay different amounts of data per update** (Dreamer 16 × 64 = 1,024 timesteps through
the world model; DQN 256 transitions), because the algorithms consume data differently. The comparison
equalises environment steps and gradient steps, not replayed samples; the results report both.

### 3.3 Replay

- Every finished episode is written to `runs/m5a/<arm>_seed<n>/replay/` in M1's episode format with its
  DINOv2 feature cache alongside — about 4.9 GB of features at 100,000 steps. Nothing is evicted.
- Features are computed once, at collection, on the training device, and reused for every update; a frame
  is never re-encoded.
- The replay keeps an in-memory index to which new episodes are appended; the directory is never
  re-scanned during a run.
- Dreamer samples sequences with M4's loader semantics: episode chosen uniformly, short episodes padded and
  masked (M4 §4.2). DQN samples transitions uniformly over all stored transitions (§5.2).
- The two arms' replays are separate directories: each arm trains only on data its own policy collected
  (plus the shared-identical prefill).

### 3.4 Resumable checkpoints

At every evaluation checkpoint, after evaluation, the driver writes a resumable state:
model and optimiser state dicts, Dreamer's slow critic and return-scale statistics, DQN's target network
and ε schedule position, the replay index, the budget step count, the in-progress episode, and every RNG
state (Python, NumPy, torch CPU, torch MPS, the environment's). A rerun with the same arguments resumes
from the latest state. **A resumed run must be bit-identical to an uninterrupted one** (§8).

### 3.5 Isolation

In the Dreamer arm the world model now trains, but an actor-critic step must still leave every world-model
parameter's `.grad` as `None` (M4 §5.2's test, unchanged in meaning). The driver checks this after the
first actor-critic step of every run and records `isolation_ok`.

---

## 4. The Dreamer arm

### 4.1 Components and one update round

The `frozen_ssl` world model (4 actions, `pad_short`, `reward_transform = "symlog"`), `Actor`, `Critic`,
the slow critic and `ReturnScale` — all from M4, all initialised from scratch with the run seed.

One update round:

1. sample one replay batch (16 × 64, masked);
2. **one world-model step** on it with M4's masked losses (Adam, 1e-4, gradient clip 100);
3. **one actor-critic step**: start states are that batch's posterior states, recomputed with the updated
   world model under `no_grad` (§4.3 drops terminal ones); imagine H = 15 with `imagine_policy`; M4's
   λ-returns, critic loss, REINFORCE actor loss, return scale and slow-critic EMA.

M4's `train_actor_critic` loop body is refactored into two functions, `world_model_step(model, opt, batch)`
and `actor_critic_step(state, batch)`. M4's fixed-dataset trainer calls the same functions, so the two
code paths cannot drift. With the amendment flags off the refactored M4 trainer must reproduce M4's
recorded Stage B training series exactly (§8).

### 4.2 Acting

The behaviour policy is `AgentPolicy`'s posterior filter with the actor **sampling**, as in M4's gated
evaluation, using the policy weights current at the start of each 100-step act phase. No ε: DreamerV3
explores through the actor's entropy bonus. Each frame's DINOv2 features are computed once and serve both
the filter and the replay.

### 4.3 The two amendments (M4 final review)

Both are flags on the actor-critic configuration, **off by default** so M4's code path and records are
unchanged; M5a's driver turns both on.

**`bound_reward_readout`.** Before the inverse symlog, the reward head's output is clipped to
`[symlog(r_min), symlog(r_max)]`, where `r_min`/`r_max` are the running minimum and maximum of rewards
actually stored in the replay at the time of the update — data-derived, no scenario constant. On `basic`
the bound reaches [−9, +99] once both a missed shot and a kill have been observed. This removes the
extrapolation route behind M4's `random_vit` seed 0 imagined return of +885.58; it does not prevent the
actor exploiting model errors within the range.

**`drop_terminal_starts`.** A start state at a transition whose `terminated` is true is excluded, so
imagination never begins after an episode's end. Truncated (time-limit) transitions remain start states:
the episode did not end there. In M4 about 1.4% of start states were terminal.

### 4.4 Recorded at each checkpoint (reported, not gated)

World-model loss terms, actor and critic losses, replay size in episodes and transitions, the share of
start states dropped, the share of reward readouts clipped, `r_min`/`r_max`, and — from evaluation — the
imagined-vs-real gap at episode start, the exploitation signature M4 §5.5 names.

---

## 5. The DQN arm

### 5.1 Network

Per frame, the cached DINOv2 features pass through a `BottleneckEncoder` — the same class, width
(`bottleneck_dim` 32, `embed_dim` 2048) and seeded initialisation as Dreamer's — giving a 2048-d embedding.
DQN has no recurrent state, so it receives the **last 4 frames' embeddings concatenated** (the standard
frame stack; at an episode's start the first frame is repeated). A 2 × 512 SiLU MLP maps the 8,192-d stack
to 4 Q-values. The bottleneck trains with the Q-network.

### 5.2 Algorithm

Double DQN with 3-step returns, at standard settings chosen before any run and not tuned against Dreamer:

| setting | value |
|---|---|
| γ | 0.99 |
| n-step | 3 |
| target network | hard copy every 2,000 updates |
| loss | Huber (δ = 1) |
| optimiser | Adam, lr 1e-4, gradient clip 10 |
| batch | 256 transitions, uniform over all stored transitions |
| reward scale | × 0.01 for the targets |

- **Terminal handling matches M4's `continue_target`:** a terminated transition does not bootstrap; a
  truncated one does. An n-step return stops at a termination and bootstraps from the last real state
  before a truncation or the end of stored data.
- **Reward scale.** A fixed 0.01 multiplier is monotone, so the optimal policy is unchanged, and it keeps
  `basic`'s +97..+99 kill rewards inside the Huber loss's quadratic region. Rewards are not clipped to ±1
  as on Atari: that would erase the difference between a kill and a step penalty.

### 5.3 Exploration

ε-greedy, ε decaying linearly from 1.0 at budget step 0 to 0.05 at budget step 50,000, then held at 0.05.
The prefill is uniformly random regardless (it is shared with Dreamer, §3.1) but counts toward the decay.

### 5.4 Known-answer test

On a small deterministic chain environment with known optimal Q-values (including a terminating and a
truncating branch), repeated DQN updates converge to those values within a stated tolerance. This checks
the n-step targets, the terminated/truncated handling, Double-DQN target selection and the target-network
copy before any ViZDoom run.

---

## 6. Evaluation

At each checkpoint (§3.1) each arm plays the 100 seeds 1,000,000–1,000,099 on a separate `basic` instance,
in its pre-registered mode:

| arm | gated mode | reported only |
|---|---|---|
| Dreamer | actor **samples** (M4's gated mode) | actor argmax |
| DQN | **greedy**, ε = 0 | ε = 0.05 |

Per episode: return, length, kill, and (Dreamer) the imagined λ-return at the first state. Episodes are
reseeded per seed, as in M4, so a rerun reproduces them. `RandomPolicy` on the same seeds is the random
baseline, measured once.

**The step-0 control.** Both arms are evaluated at budget step 0 — after the prefill, before any update.
Each arm's step-0 policy run through M4's seed-paired gate against random **must fail it**. If either
passes, the evaluation harness is broken and no reading is printed (§7.3). Step 0 is not part of the AUC.

---

## 7. The pre-registered reading

### 7.1 Primary metric

For each run, **AUC** = the mean, over the 10 checkpoints (10k … 100k), of the mean gated-mode return
over the 100 evaluation episodes. Five AUCs per arm.

### 7.2 The reading

Compare every Dreamer seed's AUC with every DQN seed's: 25 pairs. Ties count as half a win (they cannot
reach the bar either way).

| verdict | rule | one-sided exact p |
|---|---|---|
| `DREAMER_MORE_EFFICIENT` | Dreamer's AUC higher in **at least 21 of 25** pairs (Mann–Whitney U ≤ 4) | 12/252 = 0.048 |
| `DQN_MORE_EFFICIENT` | DQN's AUC higher in at least 21 of 25 pairs | 12/252 = 0.048 |
| `NOT_SEPARATED` | anything else | — |

The two-sided false-positive rate is 24/252 ≈ 0.095. `NOT_SEPARATED` is a reportable result, as in M4
§6.2: it licenses no ranking in either direction and does not establish that the arms are equal.

### 7.3 Validity, enforced by `read`

A run is valid only if: it spent exactly 100,000 budget steps; its evaluation seeds equal the fixed list
in order at every checkpoint and for the random baseline; its step-0 control fails; and (Dreamer)
`isolation_ok` is true. `read` exits:

| code | meaning |
|---|---|
| 0 | all ten runs present and valid; the reading is printed (any verdict) |
| 90 | `MISSING_RUNS` — a run's records are absent |
| 91 | `INVALID_RUNS` — a run fails §7.3; the reading is **not** printed |

Missing takes precedence over invalid. A run's AUC never enters the reading unless the run is valid.

### 7.4 Reported, not gated

- the median AUC difference (Dreamer − DQN);
- each arm's learning curve: mean and range across seeds of the gated-mode return at every checkpoint;
- budget steps to a mean gated-mode return of +50 at a checkpoint, per run; a run that never reaches it is
  reported as ">100k";
- the gated-mode return at 100k;
- the reported-only evaluation modes (§6);
- Dreamer's imagined-vs-real gap at every checkpoint (§4.4);
- **wall-clock time and gradient steps per run.** M5a measures environment-sample efficiency, not compute
  efficiency: Dreamer spends far more computation per real step, and the results say so.

---

## 8. Testing

- **Resume.** A short run interrupted after a checkpoint and resumed is bit-identical to an uninterrupted
  run of the same length (weights, replay, evaluation records).
- **M4 reproduction.** With both amendment flags off, the refactored fixed-dataset trainer reproduces a
  recorded M4 training series exactly on a fixture, and the M4 test suite passes unchanged.
- **Amendments.** `bound_reward_readout` clips exactly at the running range and is the identity inside it;
  `drop_terminal_starts` removes terminated starts and keeps truncated ones.
- **Isolation.** After an online actor-critic step, every world-model `.grad` is `None`, while the
  world-model step does update the world model.
- **Budget accounting.** A run spends exactly the budget; evaluation steps are not counted; an episode
  spanning act phases is stored once and whole.
- **DQN known-answer** (§5.4); frame-stack padding at episode start; n-step targets at termination and
  truncation.
- **Reading.** On synthetic AUCs: 21 of 25 is `DREAMER_MORE_EFFICIENT`, 20 of 25 is `NOT_SEPARATED`,
  the reverse holds, ties count half.
- **`read`.** Exits 90 on a missing run and 91 on each invalidity in §7.3 without printing the reading;
  exits 0 on ten valid synthetic runs.

The suite must be green at the commit each run is taken from, as in every earlier milestone.

---

## 9. Compute

| item | estimate |
|---|---|
| Dreamer run (100,000 steps) | ~4 h (3.7 h updates, ~15 min collection, ~5 min evaluation) |
| DQN run | ~1 h (cheap updates; collection and evaluation as Dreamer) |
| all ten runs | ~25 h — about two overnight runs on this machine |
| disk | ~4.9 GB of features per run × 10 runs ≈ 49 GB if all are kept |

**Disk is the binding constraint.** About 20 GB is free. Each run's replay features are deleted after the
run's records are written and its read-relevant files are verified — the replay is reproducible from the
seed by the determinism in §3.4, and the evaluation records, not the replay, carry every reported number.
The plan states the exact retention rule; nothing under `runs/m4*` or `data/` is touched.

---

## 10. What M5a does not answer

- **Harder scenarios** (`defend_the_center`, `deadly_corridor`) and **the encoder comparison** — M5b.
- **Compute efficiency.** M5a equalises environment steps and gradient steps, not wall-clock time or FLOPs.
- **Whether the result transfers** to other train ratios, budgets or model-free algorithms. One ratio,
  one budget and one baseline are fixed in advance.
- **Raw-pixel model-free performance.** The baseline reads frozen DINOv2 features by design (§1); a
  pixel-CNN DQN would answer a different question.
- **M3's tools.** `eval/study.py`'s routing through `load_world_model` / `read_reward` remains a follow-up;
  M5a never calls it.

---

## 11. Open questions

1. **Replay retention under the disk limit.** §9 deletes each run's feature cache after verification.
   If a run's replay is wanted for later analysis (e.g. diagnosing an exploitation episode), the plan
   names which runs to keep before the run starts, not after its result is seen.
2. **Steps-to-threshold bar.** +50 is reported, not gated. It sits below every M4 Stage C score except
   `random_vit` seed 0's (the other eight scored +62.26 to +78.45 at 10,000 updates, the exploited cell
   −94.98); a different bar changes no verdict.
