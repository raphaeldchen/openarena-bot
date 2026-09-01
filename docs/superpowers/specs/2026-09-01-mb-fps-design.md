# MB-FPS: Model-Based First-Person Shooter Agent — Design (M0–M5)

**Date:** 2026-09-01
**Status:** Approved for implementation planning
**Scope:** Milestones M0–M5. M6–M8 (ablation study, OpenArena, writeup) get a separate spec cycle.

---

## 1. Purpose

Build a DreamerV3-style model-based reinforcement learning pipeline that learns a latent world model
from pixels and trains an actor-critic policy **exclusively inside that model's imagined rollouts**.

The project has three concurrent objectives, all of which constrain the design:

1. **A working agent** — a policy that demonstrably plays ViZDoom, learned in imagination.
2. **A portfolio artifact** — legible visual output: dream rollouts, learning curves, a clean repo.
3. **A research contribution** — a controlled study of *prediction space for control*.

Objective 3 is the binding constraint. A controlled comparison cannot be retrofitted; the codebase is
built as an A/B/C harness from M0 onward.

### Relationship to the MIRA paper

This project is inspired by MIRA (Hu, Volhejn, Ramanana Rahary, Mulder et al., 2026, arXiv:2607.05352),
but **is not a reimplementation of it**. MIRA is a 5B-parameter latent-diffusion *generative video world
model* — a neural game engine whose output is playable frames. It trains no agent. It sits in Section 2.2
of its own related-work taxonomy.

This project sits in Section 2.1 — *world models for control* — the Dreamer/RSSM lineage MIRA explicitly
contrasts itself against. What transfers from MIRA is not its architecture but its central empirical
finding: that building the prediction space on a **frozen self-supervised encoder plus a learned
bottleneck** outperforms a learned-from-scratch autoencoder. MIRA established this for video generation
quality. Whether it holds for *control* is the question this project asks.

MIRA scale for reference: 10,000 match-hours, 82,983 matches, codec trained on 8xH100, world models on
16–32xH100, inference on a B200. This project runs on one Apple M4 with 16GB unified memory, with an
optional 100–200 USD cloud-GPU burst for final runs. The scale gap is roughly three orders of magnitude
and is the reason the Dreamer lineage — deliberately sample- and compute-efficient — is the correct choice.

### Closest prior work

- **DreamerV3** (Hafner et al., 2025a) — the architecture being adapted.
- **DIAMOND** (Alonso et al., 2024) — trains agents inside diffusion world models; released a CS:GO world
  model. Nearest published analogue to the original project intent.
- **DINO-WM** (Zhou et al., 2025) — predicts future frozen DINO features for *planning* (MPC) on simple
  manipulation/navigation tasks. Nearest work to this project's research question.

**Novelty position (to be verified before writeup):** frozen SSL features as the prediction space for
**actor-critic-in-imagination** (not MPC), in **visually complex 3D game environments** (not manipulation),
at **small scale**. This slice appears unclaimed. If the novelty check narrows it further, the work still
stands as a clean small-scale empirical study with a properly controlled design.

---

## 2. Constraints

| Constraint | Value | Consequence |
|---|---|---|
| Compute | Apple M4, 16GB unified, MPS backend, no CUDA | Small models; fp32; CPU escape hatch |
| Budget | 100–200 USD optional cloud burst | ~50–100 A100-hours; one or two final runs, not iteration |
| Phase 1 env | ViZDoom 1.3.0 (macOS arm64 wheel confirmed for cp310–cp314) | Native, no emulation |
| Phase 2 env | OpenArena / ioquake3 (native arm64 Universal 2, open source) | Deferred to M7 |
| Python | 3.12 in a project venv | Broadest wheel coverage; system 3.14 left untouched |

### Explicitly ruled out

- **CS2 / CS:GO live engine** — no macOS build exists. Not a driver problem; there is no binary.
- **`pydirectinput`** — Win32 `SendInput` wrapper, Windows-only.
- **Halo CE / MCC** — MCC guidance indicates 36GB+ for stable Apple Silicon play; no native multiplayer
  bots; anti-cheat. Optional stretch demo only, never a dependency.
- **Anything MIRA-scale** — diffusion world models, 5B parameters, video-quality objectives.

---

## 3. Architecture

DreamerV3-shaped. Six components, one of which is frozen in the treatment arms.

```
                    +---------------- World Model ----------------+
  obs_t --> Encoder --> e_t --+
                              +--> RSSM --> h_t --> z_hat_{t+1}    (dynamics)
  a_t ------------------------+            |
                                           +--> Decoder  --> obs_hat   (training signal only)
                                           +--> Reward   --> r_hat
                                           +--> Continue --> c_hat
                    +---------------------------------------------+
                                           |
                    +--- trained ONLY on imagined (h,z) rollouts ---+
                                           +--> Actor  --> a
                                           +--> Critic --> V
```

The actor-critic never receives a real frame and never produces a gradient on world-model parameters.
This is enforced by a unit test (Section 7), not by convention.

### 3.1 The three arms

The research variable is **the space the RSSM predicts in**. Everything downstream of the embedding —
RSSM sizing, heads, actor, critic, losses, optimizer, seeds, environment-step budget — is held identical.

| | **Arm 1: `cnn`** (baseline) | **Arm 2: `frozen_ssl`** (treatment) | **Arm 3: `random_vit`** (control) |
|---|---|---|---|
| Encoder | Learned CNN, trained end-to-end | DINOv2-S/14, **frozen** | Randomly-initialised ViT-S/14, **frozen** |
| Bottleneck | Linear 12544 -> 2048 | Linear 384 -> 32 per patch, flattened to 2048 | identical to Arm 2 |
| RSSM predicts | Pixel-space reconstruction target | SSL feature target | Random-projection feature target |
| Trained params in encoder path | ~4M (encoder + decoder) | ~0.2M (bottleneck) | ~0.2M (bottleneck) |
| Encoder runs during WM training | Yes, forward + backward | No — features precomputed | No — features precomputed |

**Arm 3 is what makes this a study rather than a demo.** If Arm 2 beats Arm 1, the obvious objection is
that a frozen encoder simply gives the RSSM a *stationary* prediction target instead of a moving one.
Arm 3 isolates that: same frozen-ness, same bottleneck, same target stability, zero pretrained knowledge.

- Arm 2 > Arm 3 -> the benefit is the pretraining.
- Arm 2 ~= Arm 3 -> the benefit is target stability, which is a more interesting finding.

Arm 3 costs almost nothing to implement: it is Arm 2 with a re-initialised backbone.

### 3.2 Input equalisation

To avoid a resolution confound, **all arms consume byte-identical 112x112 RGB frames.**

- 112 = 8 x 14, so DINOv2's patch-14 backbone yields an 8x8 patch grid (64 patches x 384 dims).
- The CNN arm applies four stride-2 convolutions -> 7x7x256 = 12544, then a linear projection to 2048.
- The SSL arms apply a per-patch linear 384 -> 32 -> 8x8x32 = 2048.

Both arms therefore emit a **2048-dimensional embedding from identical pixels**. Resolution and embedding
width are both controlled; only the representation differs.

### 3.3 Why frozen encoders are cheaper, not more expensive

Because the SSL backbone never updates, each frame is encoded **once** at collection time and the features
are cached in the replay buffer. Arms 2 and 3 then train with no ViT in the loop at all — no forward pass,
no backward pass, no encoder gradients. On MPS, where small sequential kernel launches dominate, the
treatment arms are expected to train *faster* than the baseline. This is what makes a three-arm study
feasible on a laptop.

**What gets cached.** The bottleneck is *learned* and therefore changes during training, so the cache must
sit upstream of it: **store the frozen backbone's raw 64 x 384 patch features and apply the learned
bottleneck at train time.**

This costs disk. A 112x112x3 uint8 frame is 37,632 bytes; 64 x 384 float32 patch features are 98,304 bytes,
about 2.6x larger. Storing the cache as float16 halves that to ~1.3x the raw frame, which is the default.
Buffer capacity is sized accordingly at M1.

### 3.4 Visualisation decoder

In Arms 2 and 3 the RSSM predicts features, which are not viewable. A **feature-to-pixel decoder is trained
separately, after the fact, purely for visualisation, and never contributes gradients to the RL loop.**
Keeping it out of the training path ensures the portfolio visuals cannot become an experimental confound.

### 3.5 Hyperparameters (initial, DreamerV3-small)

| Parameter | Value |
|---|---|
| Deterministic state `h` | 512 |
| Stochastic state `z` | 32 categoricals x 32 classes |
| Head MLPs | 2 x 512 |
| Batch / sequence length | 16 / 64 |
| Imagination horizon | 15 |
| Discount gamma / lambda | 0.997 / 0.95 |
| LR: world model / actor / critic | 1e-4 / 3e-5 / 3e-5 |
| KL free bits | 1.0 nat |
| KL scales (dyn / rep) | 0.5 / 0.1 |
| Actor entropy | 3e-4 |
| Critic EMA | 0.98 |
| Precision | fp32 (MPS) |

---

## 4. Module structure

```
src/mbfps/
  envs/
    protocol.py        EnvProtocol -- the modular engine hook (M0 defines, M7 implements)
    vizdoom_env.py     ViZDoom implementation
    wrappers.py        preprocessing, action repeat, episode limits
    registry.py        make_env(name, cfg)
  data/
    collector.py       policy -> episodes, with crash recovery
    policies.py        collection policies: random | scripted | epsilon-greedy agent
    buffer.py          fixed-capacity episode storage + eviction
    loader.py          sequence sampling -> (B, T, ...) batches
    features.py        frozen-encoder feature cache
  models/
    encoders.py        cnn | frozen_ssl | random_vit   <-- ONLY file differing between arms
    rssm.py            recurrent state-space core
    decoders.py        pixel decoder; separate visualisation decoder
    heads.py           reward, continue, actor, critic
    world_model.py     assembly
  training/
    wm_trainer.py      M2 / M3
    ac_trainer.py      M4 -- imagination only
    dreamer.py         M5 -- the online loop
    losses.py
  utils/
    config.py  seeding.py  device.py  logging.py
configs/
  base.yaml  arm_cnn.yaml  arm_frozen_ssl.yaml  arm_random_vit.yaml
tests/
scripts/
```

**Architectural invariant:** `encoders.py` is the only module that differs across arms. If an arm requires
editing `rssm.py`, `heads.py`, or `ac_trainer.py`, the comparison is no longer controlled. Any such change
must be applied to all three arms or not at all.

### 4.1 The engine-hook interface

```python
class EnvProtocol(Protocol):
    observation_space: gym.Space          # Box(112, 112, 3), uint8
    action_space: gym.Space               # Discrete(n)

    def reset(self, *, seed: int | None = None) -> tuple[Obs, Info]: ...
    def step(self, action: int) -> tuple[Obs, float, bool, bool, Info]: ...
    def close(self) -> None: ...

    @property
    def privileged_state(self) -> dict | None:
        """Ground-truth engine state. EVALUATION PROBES ONLY -- never a training input."""
```

`privileged_state` is the analogue of MIRA's BakkesMod physics-state channel. ViZDoom provides it via
`game_variables` and the labels buffer; ioquake3 will provide it from source at M7. It is consumed only by
linear-probe evaluation (does the latent encode position, health, ammo?) — a real result and a strong
portfolio figure. Leaking it into observations would silently invalidate the entire project, hence the
docstring contract and a dedicated test.

---

## 5. Data flow

### 5.1 Provenance

**All training data is generated locally by running a policy in a locally-running game process.** Nothing is
downloaded, scraped, or parsed from external sources. There is no public dataset dependency at any milestone.

| Milestone | Data generated by | Distribution |
|---|---|---|
| M1–M4 | Mixed collection policies (Section 5.2) in local ViZDoom | **Fixed** — frozen deliberately |
| M5 | The agent under training, improving over time | Shifting |
| M7 | Same collector, local OpenArena instance | (separate spec) |

### 5.2 Collection policies

A single random policy has poor state coverage, and that failure propagates dangerously: the world model
learns dynamics only for visited states and hallucinates elsewhere, after which the actor-critic in M4
optimises directly into those hallucinated regions. The result is an agent with high imagined return and
near-zero real return — a data-coverage bug that presents as a policy bug.

MIRA documents this failure at 10,000 hours of data: a resting ball drifts because "a still ball is a rare
event in the data", and the model reproduces the stereotyped kickoff boost even when the player holds back.
They name single-policy collection as an explicit limitation on behavioural diversity. A random policy on a
laptop will hit this considerably harder.

M1 therefore collects from a **mixture** of policies, with the mixture weights recorded in the dataset
manifest:

| Policy | Purpose | Available from |
|---|---|---|
| `random` | Unbiased local coverage around the start distribution | M1 |
| `scripted` | Forward-biased movement with sweeping aim — reaches states random play rarely visits | M1 |
| `epsilon_greedy` | A partially-trained agent with high epsilon — covers on-policy-relevant states | M4 onward |

Each episode records which policy generated it, so coverage can be attributed and the mixture adjusted.

### 5.3 Pipeline

**Collection:** `env -> (obs, action, reward, terminated, truncated, privileged) -> episode -> disk (one
npz per episode)`. For SSL arms, `obs -> frozen backbone -> 64x384 patch features` cached alongside.

**Training:** `disk -> sequence sampler -> (B=16, T=64, ...) -> world model`.

**Imagination:** every (h, z) position in the B x T batch is flattened into a batch of 1024 start states,
then rolled forward H=15 steps by the RSSM under the actor. Actor-critic losses are computed on these
imagined trajectories only.

---

## 6. Milestones

Each milestone's success criterion de-risks the next. The ordering is deliberate: freeze one variable at a
time so that a failure has exactly one plausible cause.

### M0 — Environment spine
Gymnasium-conformant ViZDoom env behind `EnvProtocol`; preprocessing wrappers; seeding; headless mode.

**Done when:** the same seed produces a bit-identical episode across two runs; headless throughput is
measured and recorded; a saved episode replays from its action sequence alone and matches frame-for-frame.

### M1 — Data pipeline
Mixed-policy collection (`random` + `scripted`, per Section 5.2), episode storage with eviction, sequence
loader, frozen-feature cache. Each episode is tagged with the policy that produced it.

**Done when:** a target transition count is collected within a bounded wall-clock time; the loader's
batches/sec is benchmarked; a round-trip test proves stored data equals collected data; the feature cache
produces byte-identical features on repeat encoding of the same frame; and a **state-visitation histogram**
over `privileged_state` is produced per policy, showing that `scripted` reaches regions `random` does not.

*The visitation histogram is a diagnostic, not a gate — its purpose is to make coverage gaps visible before
M3 rather than inferring them from a degraded rollout at M4.*

### M2 — Representation, standalone
Train encoder + decoder alone on M1 data. **No recurrence yet.**

**Done when:** reconstructions are visually recognisable, reconstruction loss has plateaued, and a
side-by-side comparison grid is saved for all three arms.

*Rationale: most world-model bugs are representation bugs, and a broken autoencoder is far easier to see in
a reconstruction grid than to infer from a degraded rollout.*

### M3 — World model on frozen offline data  [Project Goal 1: Latent Imagination Engine]
Add RSSM, KL balancing, free bits, reward and continue heads. Train on M1's **fixed** dataset.

**Done when:** conditioning on 5 real frames and then imagining 45 steps from the action stream alone
produces a coherent rollout; a per-step-error-vs-horizon curve is produced; reward-prediction accuracy is
reported; a linear probe from latent to `privileged_state` is fit and its R^2 reported.

*This is the portfolio centerpiece and it arrives at milestone three, not nine.*

### M4 — Actor-critic in imagination  [Project Goal 2: Latent Space Planning]
Lambda-returns, entropy bonus, critic EMA target. **World model loaded from M3 and frozen.**

**Done when:** a policy trained with zero real-environment gradients beats a random policy when deployed
live; the imagination-isolation test passes.

*Rationale: freezing the world model separates "can a policy be learned from imagination?" from "can the
world model track a shifting policy-induced data distribution?" Debugging both at once is how projects die.*

### M5 — Close the loop
Full online Dreamer loop: interleaved collection, world-model training, and actor-critic training, with a
configurable train ratio. Add a model-free baseline (PPO or DQN) at equal environment steps.

**Done when:** learning curves are produced on `basic`, then `defend_the_center`, then `deadly_corridor`;
the agent's return exceeds the model-free baseline at equal environment steps, or the gap is characterised.

*The project's stated motivation is reduced sample inefficiency. Without a model-free comparator at equal
environment steps, that is an assertion rather than a result.*

---

## 7. Testing

| Test | Catches |
|---|---|
| **Overfit-one-batch** — world model drives loss to ~0 on 4 fixed sequences | A failure here is a bug, not a hyperparameter problem |
| **Gradient-flow** — every trainable parameter has a non-`None`, non-zero grad after one step | Detached tensors, the classic RSSM bug |
| **Imagination-isolation** — after an actor-critic step, all world-model params have `grad is None` | Directly unit-tests the project's central claim |
| **Privileged-state isolation** — assert `privileged_state` keys never appear in any training tensor | Silent invalidation of the whole experiment |
| **Determinism** — same seed twice yields identical trajectories | Nondeterminism poisoning arm comparisons |
| **Round-trip** — write episode, read back, assert equal | Data-pipeline corruption |
| **Shape/dtype contracts** on every module | Integration bugs, cheaply |
| **Golden rollout** — fixed seed + weights -> hash of imagined latents | Silent regressions during refactors |

Where a research claim can be expressed as an invariant over gradients or shapes, it is written as a test
rather than a comment.

---

## 8. Failure modes and mitigations

| Risk | Mitigation |
|---|---|
| MPS operator gaps / numerical differences | fp32 throughout; `PYTORCH_ENABLE_MPS_FALLBACK=1`; `--device cpu` escape hatch (models are small enough for CPU debugging) |
| KL collapse or NaN loss | Free bits, KL balancing, gradient clipping; assert-on-NaN dumps a checkpoint rather than dying silently |
| ViZDoom segfault mid-episode | Collector catches, restarts the env, discards the partial episode, logs the incident |
| Unbounded replay disk growth | Fixed-capacity buffer with episode eviction |
| Sequential RSSM unroll is slow on MPS | Prefer batch size over sequence length; benchmark early at M1; consider a Transformer world-model variant if the GRU dominates wall-clock |
| Research novelty narrower than assumed | Verify before writeup; design degrades gracefully to a clean small-scale empirical study |
| Arm divergence via well-meaning edits | `encoders.py`-only invariant, enforced in review |

---

## 9. Out of scope

- Anti-cheat circumvention, live multiplayer matchmaking, or any online CS service.
- Multi-agent or self-play world models (MIRA's actual contribution).
- Audio processing.
- **Replay-file parsing.** The original project definition allowed aggregating data by parsing replay files.
  That path existed for CS:GO demo files; with CS:GO out of scope it has no subject. ViZDoom has no replay
  corpus worth parsing, and while OpenArena records demos, no large public collection exists. All data is
  generated locally by the collector.
- Diffusion or flow-matching world models.
- Real-time interactive play of the world model as a standalone engine.
- M6–M8 (ablation study execution, OpenArena environment, writeup) — separate spec cycle.

---

## 10. Open questions

1. **DINOv3 vs DINOv2 weights.** DINOv3 is gated; DINOv2-S/14 is the working default. Resolve at M2.
2. **Action space.** Discrete combinations to start. Whether mouse-delta requires continuous or
   finer-grained discretisation is deferred until M5 results on `deadly_corridor`.
3. **Cloud burst timing.** Whether the 100–200 USD is spent on M5 tuning or held for the M6 study is
   decided once M5 wall-clock per run is measured.
4. **Repository name.** Project directory is currently `csgo-bot`; the Python package is `mbfps`. Renaming
   the directory is optional and left to the author.
