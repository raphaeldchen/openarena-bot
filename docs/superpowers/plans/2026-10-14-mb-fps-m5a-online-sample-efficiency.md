# MB-FPS M5a — Online Dreamer vs DQN on `basic` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an online Dreamer agent and a Double-DQN agent, both reading the same frozen DINOv2 features, for 100,000 environment steps × 5 seeds each on ViZDoom `basic`, and read a pre-registered 21-of-25 rank test on the area under the evaluation curve.

**Architecture:** One synchronous driver (`mbfps.training.online.train_online`) alternates 100-step act phases (`OnlineRollout`) with 20 update rounds, evaluates every 10,000 budget steps, and writes a resumable checkpoint after each evaluation. Two arms implement one interface (`mbfps.training.arms`): `DreamerArm` reuses M4's world model and a refactored `ActorCriticLearner`; `DQNArm` uses `mbfps.training.dqn`. `mbfps.eval.efficiency` and `scripts/m5a_study.py` compute the reading with validity checks and exit codes.

**Tech Stack:** Python 3.12, PyTorch (MPS, fp32), NumPy, ViZDoom 1.3.0, pytest. Run from the worktree with `PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python` (the venv's editable install otherwise imports the main checkout).

**Spec:** `docs/superpowers/specs/2026-10-14-mb-fps-m5a-online-sample-efficiency-design.md` (section numbers below — §3.1 etc. — refer to it).

## Global Constraints

- Budget **100,000** environment steps per run, prefill included; prefill **5,000** random steps; act phases of **100** steps; **20** update rounds per act phase (one per 5 steps); evaluation at budget step **0** and every **10,000**.
- Evaluation seeds **1,000,000–1,000,099** (`mbfps.eval.live.EVAL_SEEDS`), on a separate env instance; evaluation steps are never counted and never stored.
- Training-episode seeds: `2_000_000 + 100_000 * run_seed + k` for the run's k-th episode (disjoint from evaluation seeds).
- Dreamer: `frozen_ssl` world model, `n_actions=4`, `pad_short=True`, `reward_transform="symlog"`, Adam 1e-4, clip 100, batch 16 × 64; actor-critic exactly M4's `ACConfig` defaults with `bound_reward_readout=True`, `drop_terminal_starts=True`.
- DQN: Double DQN, n-step **3**, γ **0.99**, target hard copy every **2,000** updates, Huber δ 1, Adam **1e-4**, clip **10**, batch **256** transitions, reward scale **0.01**, frame stack **4**, ε linear **1.0 → 0.05** over budget steps 0–50,000 then **0.05**; MLP 2 × 512 SiLU.
- Gated evaluation modes: Dreamer **`sample`**, DQN **`greedy`** (ε = 0). Reported-only: Dreamer `argmax`, DQN `eps0.05` (implemented as `eps0_05`; strict JSON keys, ruling in Task 8).
- Reading: AUC = mean over the 10 checkpoints 10k…100k of the mean gated return; 25 pairs; ties count ½; **≥ 21** wins → a `*_MORE_EFFICIENT` verdict; else `NOT_SEPARATED`.
- `read` exits **0** (reading printed), **90** `MISSING_RUNS`, **91** `INVALID_RUNS`; missing beats invalid.
- Amendment flags default **off**; M4's code paths and recorded results must stay bit-for-bit reproducible.
- Each run's replay feature caches (`*.features.npy`) are **deleted after the run's record is verified complete**; episode `.npz` files and records are kept (M5a: frames were deleted after verification by an ops watcher with user approval, see M5a results). Never touch `runs/m4*`, `runs/m3*`, `data/`.
- Run outputs go to `runs/m5a/<arm>_seed<n>/` (`runs` is a symlink into the main checkout).
- Run long commands with a redirect and read `$?`, never through a pipe.
- Commit trailers: the Co-Authored-By line the committing agent's environment specifies.
- Tests run with `PYTHONPATH=src:.` (the loop tests import `tests.training.fake_env`).

## Review Focus

1. **An episode in progress at a checkpoint** must resume exactly: the engine is replayed from the episode's seed and recorded actions, frames are re-encoded, and the policy's internal state is restored. Pinned in Task 8 (`test_resume_is_bit_identical`).
2. **The prefill boundary** cuts the in-progress random episode as truncated at exactly 5,000 steps, and the arm's first episode starts fresh. Pinned in Task 8 (`test_prefill_cuts_at_exactly_the_prefill_step`).
3. **The budget end** stores the in-progress episode truncated and spends exactly the budget. Pinned in Task 6 (`test_close_budget_stores_the_tail_truncated`) and Task 8 (`test_run_spends_exactly_the_budget`).
4. **Reward bounds before any kill is observed** are the observed range only (e.g. [−9, −4]); they widen as episodes arrive. Pinned in Task 7 (`test_reward_bounds_track_observed_rewards`).
5. **Deleting replay features** must refuse an incomplete or unverifiable record. Pinned in Task 9 (`test_delete_refuses_incomplete_record`).

---

## File Structure

| file | responsibility |
|---|---|
| `src/mbfps/models/transforms.py` (modify) | `read_reward(..., bounds=None)` |
| `src/mbfps/training/ac_trainer.py` (modify) | amendment flags; `start_states(..., drop_terminal=)`; `ActorCriticLearner` (one AC step, state dict); M4 trainer uses it |
| `src/mbfps/training/world_model.py` (modify) | `world_model_step`; `train_world_model` uses it |
| `src/mbfps/data/loader.py` (modify) | `SequenceLoader.add(path)` |
| `src/mbfps/data/collector.py` (modify) | module-level `privileged_row` (was `Collector._row`) |
| `src/mbfps/eval/live.py` (modify) | `AgentPolicy.act_from_features`, `get_state`/`set_state` |
| `src/mbfps/training/dqn.py` (create) | `DQNConfig`, `epsilon`, `QNetwork`, `TransitionSampler`, `dqn_step`, `EpsilonGreedyQPolicy` |
| `src/mbfps/training/online_rollout.py` (create) | `OnlineRollout`: budget accounting, episodes across act phases, replay writes, resume |
| `src/mbfps/training/arms.py` (create) | `DreamerArm`, `DQNArm` |
| `src/mbfps/training/online.py` (create) | `OnlineConfig`, `train_online`, checkpoints, records, `verify_record`, `delete_replay_features` |
| `scripts/train_online.py` (create) | one run on real ViZDoom + DINOv2; verify and delete features |
| `src/mbfps/eval/efficiency.py` (create) | AUC, pairwise wins, verdict, validity, secondary metrics |
| `scripts/m5a_study.py` (create) | run the ten runs; read with exit codes |
| `tests/training/fake_env.py` (create) | a deterministic fake environment for loop tests |

---

## Task 1: The two amendments, behind flags

**Files:**
- Modify: `src/mbfps/models/transforms.py` (`read_reward`)
- Modify: `src/mbfps/training/ac_trainer.py` (`ACConfig`, `start_states`)
- Test: `tests/models/test_transforms.py`, `tests/training/test_ac_trainer.py`

**Interfaces:**
- Produces: `read_reward(head_output, transform, bounds: tuple[float, float] | None = None)`; `ACConfig.bound_reward_readout: bool = False`, `ACConfig.drop_terminal_starts: bool = False`; `start_states(world_model, batch, drop_terminal: bool = False)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/models/test_transforms.py`:

```python
def test_read_reward_bounds_clip_in_target_space():
    """Bounds are reward units; the clip happens on the head's output, in the
    transform's space, so symexp never sees a value beyond symlog(r_max)."""
    head = torch.tensor([symlog(torch.tensor(500.0)).item(), 0.0, -50.0])
    out = read_reward(head, "symlog", bounds=(-9.0, 99.0))
    torch.testing.assert_close(out, torch.tensor([99.0, 0.0, -9.0]), rtol=1e-5, atol=1e-4)


def test_read_reward_bounds_are_identity_inside():
    head = symlog(torch.tensor([-4.0, 50.0]))
    torch.testing.assert_close(read_reward(head, "symlog", bounds=(-9.0, 99.0)),
                               torch.tensor([-4.0, 50.0]), rtol=1e-5, atol=1e-4)


def test_read_reward_without_bounds_is_unchanged():
    head = torch.tensor([7.0])
    assert torch.equal(read_reward(head, "symlog"), symexp(head))
```

Append to `tests/training/test_ac_trainer.py`:

```python
def test_amendment_flags_default_off():
    cfg = ACConfig()
    assert cfg.bound_reward_readout is False and cfg.drop_terminal_starts is False


def test_drop_terminal_removes_terminated_keeps_truncated(world):
    buf, model = world
    batch = SequenceLoader(buf, batch_size=4, seq_len=16, seed=0, pad_short=True,
                           load_obs=True).sample()
    batch["mask"][:] = True
    batch["terminated"][:] = False
    batch["truncated"][:] = False
    batch["terminated"][0, 3] = True
    batch["truncated"][1, 5] = True
    dev = to_device(batch, torch.device("cpu"))
    h_all, _ = start_states(model, dev)
    h_drop, _ = start_states(model, dev, drop_terminal=True)
    assert h_all.shape[0] == 4 * 16
    assert h_drop.shape[0] == 4 * 16 - 1          # only the terminated one
```

- [ ] **Step 2: Run them to see them fail**

Run: `PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/models/test_transforms.py tests/training/test_ac_trainer.py -q -k "bounds or amendment or drop_terminal"`
Expected: FAIL — `unexpected keyword argument 'bounds'`, `no attribute 'bound_reward_readout'`, `unexpected keyword argument 'drop_terminal'`.

- [ ] **Step 3: Implement**

In `transforms.py`, replace `read_reward`:

```python
def read_reward(
    head_output: torch.Tensor, transform: str, bounds: tuple[float, float] | None = None
) -> torch.Tensor:
    """The reward head's output in reward units.

    `bounds` (M5a §4.3): `(r_min, r_max)` in reward units. The head's output is
    clipped to `[reward_target(r_min), reward_target(r_max)]` BEFORE the inverse
    transform, so an off-distribution head output cannot extrapolate through
    `symexp`. None leaves the readout exactly as M4 had it."""
    _require_known(transform)
    if bounds is not None:
        lo, hi = (reward_target(torch.tensor(float(b)), transform).item() for b in bounds)
        head_output = head_output.clamp(min=lo, max=hi)
    return symexp(head_output) if transform == "symlog" else head_output
```

In `ac_trainer.py`, add to `ACConfig` (after `grad_clip`):

```python
    bound_reward_readout: bool = False
    """M5a §4.3: clip the reward head to the observed reward range before the
    inverse transform. Off reproduces M4."""
    drop_terminal_starts: bool = False
    """M5a §4.3: never start imagination at a terminated transition. Off
    reproduces M4."""
```

Replace `start_states`:

```python
@torch.no_grad()
def start_states(world_model, batch: dict, drop_terminal: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior `(h, z)` at every masked-in position of `batch`.

    `drop_terminal` (M5a §4.3) also excludes positions whose transition
    terminated the episode; truncated (time-limit) positions are kept."""
    out = world_model.rssm.observe(world_model.embed(batch), batch["actions"])
    mask = batch["mask"].bool()
    if drop_terminal:
        mask = mask & ~batch["terminated"].bool()
    return out["h"][mask], out["z"][mask]
```

- [ ] **Step 4: Run the transforms and trainer suites**

Run: `PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/models/test_transforms.py tests/training/test_ac_trainer.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/models/transforms.py src/mbfps/training/ac_trainer.py tests/models/test_transforms.py tests/training/test_ac_trainer.py
git commit -m "feat: a bounded reward readout and terminal-start dropping, both off by default"
```

---

## Task 2: `ActorCriticLearner` — one actor-critic step, reusable; M4 reproduces

**Files:**
- Modify: `src/mbfps/training/ac_trainer.py`
- Create: `tests/training/fixtures/ac_golden_m4.json`
- Test: `tests/training/test_ac_trainer.py`

**Interfaces:**
- Consumes: Task 1's flags, `read_reward(..., bounds)`, `start_states(..., drop_terminal)`.
- Produces:
  - `ActorCriticLearner(n_actions: int, cfg: ACConfig, device: torch.device)` with attributes `actor`, `critic`, `critic_slow`, `scale`; methods `step(world_model, batch, reward_bounds: tuple[float, float] | None = None) -> dict` returning floats `actor_loss, critic_loss, scale, imagined_return, entropy, n_starts, n_dropped, clipped_share`; `state_dict() -> dict`; `load_state_dict(sd)`.
  - `train_actor_critic` (M4) unchanged in signature and output; it now drives an `ActorCriticLearner`.

- [ ] **Step 1: Record the golden series from the CURRENT code, before changing anything**

```bash
PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python - <<'EOF'
import json, tempfile
from pathlib import Path
import numpy as np, torch
from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.envs.protocol import OBS_SHAPE
from mbfps.training.ac_trainer import ACConfig, train_actor_critic
from mbfps.training.world_model import load_world_model, train_world_model
from mbfps.utils.config import get_config
KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")
def ep(t, fill):
    obs = np.zeros((t + 1, *OBS_SHAPE), dtype=np.uint8); obs[:, 0, 0, 1] = fill
    term = np.zeros(t, dtype=bool); term[-1] = fill % 2 == 0
    return Episode(obs=obs, actions=(np.arange(t) % 4).astype(np.int32),
                   rewards=np.full(t, -4.0, np.float32), terminated=term, truncated=np.zeros(t, bool),
                   privileged=np.zeros((t + 1, len(KEYS)), np.float32), privileged_keys=KEYS,
                   policy_name="random", seed=fill, scenario="basic")
root = Path(tempfile.mkdtemp())
buf = ReplayBuffer(root / "data", capacity_transitions=100_000)
for fill in range(1, 8):
    buf.add(ep(30 if fill % 2 else 6, fill))
cfg = get_config("cnn", steps=2, seq_len=16, batch_size=4, device="cpu",
                 n_actions=4, pad_short=True, reward_transform="symlog")
train_world_model(cfg, buf, out_dir=root / "wm")
model = load_world_model(root / "wm" / "world_model_cnn_seed0.pt", torch.device("cpu"))
h = train_actor_critic(model, buf, ACConfig(updates=4, checkpoint_every=4, batch_size=4,
                                            seq_len=16, device="cpu"), out_dir=None, log_every=0)
keys = ("actor_loss", "critic_loss", "scale", "imagined_return", "entropy")
Path("tests/training/fixtures").mkdir(parents=True, exist_ok=True)
Path("tests/training/fixtures/ac_golden_m4.json").write_text(json.dumps({k: h[k] for k in keys}, indent=2))
print({k: h[k][:2] for k in keys})
EOF
```

This fixture is the M4 code's own output. Commit it in Step 6 with the refactor; it must not be regenerated after the refactor.

- [ ] **Step 2: Write the failing tests** (append to `tests/training/test_ac_trainer.py`)

```python
import json

from mbfps.training.ac_trainer import ActorCriticLearner

GOLDEN = Path(__file__).parent / "fixtures" / "ac_golden_m4.json"


def test_m4_trainer_reproduces_golden_series(world):
    """The refactor must not move M4: with the flags off, the series the M4
    code produced before the refactor is reproduced exactly."""
    buf, model = world
    h = train_actor_critic(model, buf, ACConfig(updates=4, checkpoint_every=4, batch_size=4,
                                                seq_len=16, device="cpu"),
                           out_dir=None, log_every=0)
    golden = json.loads(GOLDEN.read_text())
    for key, values in golden.items():
        assert h[key] == values, key


def test_learner_step_returns_metrics_and_leaves_world_model_untouched(world):
    buf, model = world
    for p in model.parameters():
        p.grad = None
    learner = ActorCriticLearner(4, ACConfig(batch_size=4, seq_len=16, device="cpu",
                                             drop_terminal_starts=True,
                                             bound_reward_readout=True),
                                 torch.device("cpu"))
    batch = to_device(SequenceLoader(buf, 4, 16, seed=0, pad_short=True, load_obs=True).sample(),
                      torch.device("cpu"))
    out = learner.step(model, batch, reward_bounds=(-9.0, 99.0))
    for key in ("actor_loss", "critic_loss", "scale", "imagined_return", "entropy",
                "n_starts", "n_dropped", "clipped_share"):
        assert isinstance(out[key], float), key
    assert out["n_starts"] + out["n_dropped"] == float(batch["mask"].sum())
    assert all(p.grad is None for p in model.parameters())


def test_learner_state_dict_round_trips(world):
    _, model = world
    a = ActorCriticLearner(4, ACConfig(device="cpu"), torch.device("cpu"))
    a.scale.low, a.scale.high = -3.0, 7.0
    b = ActorCriticLearner(4, ACConfig(seed=9, device="cpu"), torch.device("cpu"))
    b.load_state_dict(a.state_dict())
    for x, y in zip(a.actor.parameters(), b.actor.parameters()):
        assert torch.equal(x, y)
    for x, y in zip(a.critic_slow.parameters(), b.critic_slow.parameters()):
        assert torch.equal(x, y)
    assert (b.scale.low, b.scale.high) == (-3.0, 7.0)
```

(`Path` is imported at the top of the file already for the script test; if not, add `from pathlib import Path`.)

- [ ] **Step 3: Run to see the new tests fail and the golden test pass on the OLD code**

Run: `PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/training/test_ac_trainer.py -q -k "golden or learner"`
Expected: `test_m4_trainer_reproduces_golden_series` PASSES (the fixture is the current code's output); the two learner tests FAIL with `cannot import name 'ActorCriticLearner'`.

- [ ] **Step 4: Implement the learner and drive M4's trainer through it**

Add to `ac_trainer.py`, before `train_actor_critic`:

```python
class ActorCriticLearner:
    """One DreamerV3 actor-critic update on a world model's imagination (M4
    §5.3), reusable by the fixed-dataset trainer (M4) and the online loop (M5a).

    The world model is read under `no_grad` only: no gradient from this step
    can reach it, whether or not its parameters require grad."""

    def __init__(self, n_actions: int, cfg: ACConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.actor = Actor(n_actions, seed=cfg.seed).to(device)
        self.critic = Critic(seed=cfg.seed).to(device)
        self.critic_slow = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.scale = ReturnScale(cfg.scale_decay)

    def step(self, world_model, batch: dict,
             reward_bounds: tuple[float, float] | None = None) -> dict[str, float]:
        cfg = self.cfg
        transform = world_model.cfg.train.reward_transform
        h, z = start_states(world_model, batch, drop_terminal=cfg.drop_terminal_starts)
        n_real = float(batch["mask"].bool().sum())
        bounds = reward_bounds if cfg.bound_reward_readout else None
        sample_actor = lambda latent: self.actor.dist(latent).sample()
        with torch.no_grad():
            traj = world_model.rssm.imagine_policy(sample_actor, (h, z), cfg.horizon)
            heads = world_model.heads(traj["latent"][:, 1:])
            clipped_share = 0.0
            if bounds is not None:
                lo, hi = (reward_target_value(b, transform) for b in bounds)
                raw = heads["reward"]
                clipped_share = float(((raw < lo) | (raw > hi)).float().mean())
            rewards = read_reward(heads["reward"], transform, bounds=bounds)
            cont = torch.sigmoid(heads["continue_logit"])
            values_slow = self.critic_slow(traj["latent"])
            returns = lambda_returns(rewards, cfg.gamma * cont, values_slow, cfg.lam)
            weights = continue_weights(cont)
        latents = traj["latent"][:, :-1]

        values = self.critic(latents)
        critic_loss = (0.5 * weights * (values - returns).pow(2)).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.critic_opt.step()
        _ema(self.critic_slow, self.critic, cfg.critic_ema)

        s = self.scale.update(returns)
        dist = self.actor.dist(latents)
        advantage = ((returns - values.detach()) / s).detach()
        entropy = dist.entropy()
        actor_loss = (weights * (-advantage * dist.log_prob(traj["action"])
                                 - cfg.entropy * entropy)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
        self.actor_opt.step()

        out = {"actor_loss": float(actor_loss.detach()),
               "critic_loss": float(critic_loss.detach()),
               "scale": s,
               "imagined_return": float(returns[:, 0].mean()),
               "entropy": float(entropy.mean().detach()),
               "n_starts": float(h.shape[0]),
               "n_dropped": n_real - float(h.shape[0]),
               "clipped_share": clipped_share}
        if h.device.type == "mps":
            # N (start states) differs every update, so every update allocates
            # tensors of new shapes and the MPS caching allocator never reuses
            # its blocks. Driver memory grew ~60 MB/update and hit OOM near
            # update 350 (M4); releasing the cache each update holds it flat.
            torch.mps.empty_cache()
        return out

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "critic_slow": self.critic_slow.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "scale": {"low": self.scale.low, "high": self.scale.high}}

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_slow.load_state_dict(sd["critic_slow"])
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        self.scale.low, self.scale.high = sd["scale"]["low"], sd["scale"]["high"]
```

Add the helper next to the imports, and import `reward_target`:

```python
from mbfps.models.transforms import read_reward, reward_target


def reward_target_value(reward: float, transform: str) -> float:
    """A reward bound in the head's output space (M5a §4.3)."""
    return float(reward_target(torch.tensor(float(reward)), transform))
```

Rewrite `train_actor_critic`'s body to use the learner. Keep construction order and every RNG draw as before: `seed_everything(cfg.seed)`, `freeze`, then construct the learner (its `Actor`/`Critic` are seeded through `seeded_init`, so construction order does not affect their weights), then the loader:

```python
def train_actor_critic(world_model, buffer: ReplayBuffer, cfg: ACConfig,
                       out_dir: Path | None, log_every: int = 100) -> dict[str, Any]:
    seed_everything(cfg.seed)
    device = next(world_model.parameters()).device
    freeze(world_model)
    arm, n_actions = world_model.cfg.arm, world_model.cfg.train.n_actions
    learner = ActorCriticLearner(n_actions, cfg, device)

    train_paths, _ = episode_split(buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=0)
    kind = world_model.input_kind
    loader = SequenceLoader(
        buffer, batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed=cfg.seed,
        load_obs=kind == "obs", load_features=kind == "features",
        feature_backbone=encoder_backbone(world_model.cfg.encoder) or "dinov2",
        paths=train_paths, pad_short=True,
    )
    bounds = None
    if cfg.bound_reward_readout:
        rewards = np.concatenate([np.load(p)["rewards"] for p in train_paths])
        bounds = (float(rewards.min()), float(rewards.max()))

    history: dict[str, Any] = {"actor_loss": [], "critic_loss": [], "scale": [],
                               "imagined_return": [], "entropy": [], "checkpoints": []}
    isolation_ok = True
    start = time.perf_counter()
    for update in range(1, cfg.updates + 1):
        batch = to_device(loader.sample(), device)
        out = learner.step(world_model, batch, reward_bounds=bounds)
        if update == 1:
            isolation_ok = all(p.grad is None for p in world_model.parameters())
            if not isolation_ok:
                raise RuntimeError("a world-model parameter received a gradient (spec §5.2)")
        for key in ("actor_loss", "critic_loss", "scale", "imagined_return", "entropy"):
            history[key].append(out[key])
        if out_dir is not None and update % cfg.checkpoint_every == 0:
            path = _save(Path(out_dir), arm, cfg.seed, update, n_actions,
                         learner.actor, learner.critic, learner.critic_slow)
            history["checkpoints"].append(str(path))
        if log_every and update % log_every == 0:
            print(f"[ac {arm}] update {update}/{cfg.updates} actor={history['actor_loss'][-1]:+.4f} "
                  f"critic={history['critic_loss'][-1]:.4f} G0={history['imagined_return'][-1]:+.2f}")

    seconds = time.perf_counter() - start
    history.update({"arm": arm, "seed": cfg.seed, "updates": cfg.updates,
                    "seconds": seconds, "seconds_per_update": seconds / cfg.updates,
                    "isolation_ok": isolation_ok})
    return history
```

Add `import numpy as np` to the module's imports. The existing MPS empty-cache test (`test_mps_cache_released_every_update`) must still count exactly one call per update: the call now lives in `step`, and the loop no longer makes it.

- [ ] **Step 5: Run the trainer suite**

Run: `PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/training/test_ac_trainer.py tests/eval/test_live.py tests/eval/test_eval_live_script.py -q`
Expected: all PASS, including the golden test and both MPS-cache tests.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/training/ac_trainer.py tests/training/test_ac_trainer.py tests/training/fixtures/ac_golden_m4.json
git commit -m "refactor: ActorCriticLearner holds one actor-critic step; M4's trainer reproduces its golden series"
```

---

## Task 3: `world_model_step`

**Files:**
- Modify: `src/mbfps/training/world_model.py`
- Create: `tests/training/fixtures/wm_golden_m3.json`
- Test: `tests/training/test_world_model.py`

**Interfaces:**
- Produces: `world_model_step(model, optimiser, batch, grad_clip: float = 100.0) -> tuple[float, dict]` (loss, parts); after it returns, every world-model parameter's `.grad` is `None`.

- [ ] **Step 1: Record the golden loss series from the CURRENT code**

```bash
PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python - <<'EOF'
import json, tempfile
from pathlib import Path
import numpy as np
from mbfps.data.buffer import ReplayBuffer
import importlib.util, sys
spec = importlib.util.spec_from_file_location("twm", "tests/training/test_world_model.py")
twm = importlib.util.module_from_spec(spec); spec.loader.exec_module(twm)
from mbfps.training.world_model import train_world_model
from mbfps.utils.config import get_config
root = Path(tempfile.mkdtemp())
buf = ReplayBuffer(root, capacity_transitions=10_000)
for fill in (1, 2, 3, 4, 5):
    buf.add(twm.make_episode(t=40, fill=fill))
h = train_world_model(get_config("cnn", steps=4, seq_len=16, batch_size=4, device="cpu"),
                      buf, out_dir=None, log_every=0)
Path("tests/training/fixtures").mkdir(parents=True, exist_ok=True)
Path("tests/training/fixtures/wm_golden_m3.json").write_text(json.dumps({"loss": h["loss"]}, indent=2))
print(h["loss"])
EOF
```

- [ ] **Step 2: Write the failing tests** (append to `tests/training/test_world_model.py`)

```python
import json

from mbfps.training.world_model import world_model_step

WM_GOLDEN = Path(__file__).parent / "fixtures" / "wm_golden_m3.json"


def test_train_world_model_reproduces_golden_losses(buffer):
    h = train_world_model(get_config("cnn", steps=4, seq_len=16, batch_size=4, device="cpu"),
                          buffer, out_dir=None, log_every=0)
    assert h["loss"] == json.loads(WM_GOLDEN.read_text())["loss"]


def test_world_model_step_updates_and_clears_grads(buffer):
    seed_everything(0)
    model = WorldModel(get_config("cnn", seq_len=16, batch_size=4, device="cpu"))
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = [p.detach().clone() for p in model.parameters()]
    batch = to_device(SequenceLoader(buffer, 4, 16, seed=0).sample(), torch.device("cpu"))
    loss, parts = world_model_step(model, opt, batch)
    assert isinstance(loss, float) and "embedding" in parts
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
    assert all(p.grad is None for p in model.parameters())
```

(`Path`, `SequenceLoader`, `to_device`, `seed_everything`, `WorldModel` are already imported in this test file; add any that are not.)

- [ ] **Step 3: Run to see the step test fail and the golden pass**

Run: `PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/training/test_world_model.py -q -k "golden or world_model_step"`
Expected: golden PASSES; `test_world_model_step_updates_and_clears_grads` FAILS with `cannot import name 'world_model_step'`.

- [ ] **Step 4: Implement**

Add to `world_model.py`, above `train_world_model`:

```python
def world_model_step(model, optimiser, batch: dict[str, Any],
                     grad_clip: float = 100.0) -> tuple[float, dict[str, float]]:
    """One world-model update (M3/M4 losses, masked when the batch carries a
    mask). Gradients are released after the step, so a following actor-critic
    step can verify it left every world-model `.grad` as None (M5a §3.5)."""
    loss, parts = model(batch)
    optimiser.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimiser.step()
    optimiser.zero_grad(set_to_none=True)
    return float(loss.detach()), parts
```

In `train_world_model`, replace the five lines from `loss, parts = model(...)` through `losses.append(...)` with:

```python
        loss_value, parts = world_model_step(model, optimiser, to_device(loader.sample(), device))
        losses.append(loss_value)
```

- [ ] **Step 5: Run the world-model suite and the eval suites that train world models**

Run: `PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/training -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/training/world_model.py tests/training/test_world_model.py tests/training/fixtures/wm_golden_m3.json
git commit -m "refactor: world_model_step, reused by train_world_model, which reproduces its golden losses"
```

---

## Task 4: `SequenceLoader.add` and `privileged_row`

**Files:**
- Modify: `src/mbfps/data/loader.py` (`SequenceLoader`)
- Modify: `src/mbfps/data/collector.py` (`Collector._row` → module `privileged_row`)
- Test: `tests/data/test_loader.py`, `tests/data/test_collector.py`

**Interfaces:**
- Produces: `SequenceLoader.add(path: Path) -> None`; `privileged_row(state, keys, previous) -> np.ndarray` in `mbfps.data.collector`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_loader.py`:

```python
def test_add_makes_a_new_episode_sampleable(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    first = buf.add(make_episode(t=20, fill=1))
    loader = SequenceLoader(buf, batch_size=16, seq_len=8, seed=0, paths=[first])
    assert set(np.unique(loader.sample()["obs"])) == {1}
    second = buf.add(make_episode(t=20, fill=2))
    loader.add(second)
    seen = set()
    for _ in range(10):
        seen |= set(np.unique(loader.sample()["obs"]).tolist())
    assert seen == {1, 2}
    assert loader.episode_path(1) == second


def test_add_checks_feature_geometry(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    path = buf.add(make_episode(t=10, fill=1))
    np.save(path.with_suffix(feature_suffix("dinov2")), np.zeros((11, 64, 32), np.float16))
    loader = SequenceLoader(buf, 4, 8, seed=0, load_features=True, paths=[])
    with pytest.raises(ValueError):
        loader.add(path)
```

Append to `tests/data/test_collector.py`:

```python
def test_privileged_row_is_module_level():
    from mbfps.data.collector import privileged_row
    row = privileged_row({"a": 1.0, "b": 2.0}, ("a", "b"), previous=None)
    assert row.tolist() == [1.0, 2.0]
    assert privileged_row(None, ("a", "b"), previous=row).tolist() == [1.0, 2.0]
```

- [ ] **Step 2: Run to see them fail**

Run: `PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/data/test_loader.py tests/data/test_collector.py -q -k "add_ or privileged_row"`
Expected: FAIL — `no attribute 'add'`, `cannot import name 'privileged_row'`.

- [ ] **Step 3: Implement**

In `SequenceLoader`, factor the per-path feature loading out of `__init__` into a method and add `add`:

```python
    def _load_features_for(self, path: Path) -> np.ndarray:
        feature_path = path.with_suffix(feature_suffix(self.feature_backbone))
        if not feature_path.is_file():
            raise FileNotFoundError(
                f"no cached features for {path.name}; expected {feature_path.name}. "
                f"Run scripts/cache_features.py --backbone {self.feature_backbone} first."
            )
        features = np.load(feature_path, mmap_mode="r")
        _check_feature_geometry(features, self.feature_backbone,
                                BACKBONE_GEOMETRY[self.feature_backbone], feature_path)
        return features

    def add(self, path: Path) -> None:
        """Append one episode (M5a: the online replay grows during a run).
        The directory is never re-scanned; the new episode becomes eligible
        for the next `sample()`."""
        path = Path(path)
        episode = self._read(path)
        features = self._load_features_for(path) if self.load_features else None
        self._paths.append(path)
        self._episodes.append(episode)
        if self.load_features:
            self._features.append(features)
```

In `__init__`, replace the body of `for path in self._paths:` (inside `if load_features:`) with `self._features.append(self._load_features_for(path))`, keeping the registry `KeyError` check before the loop. Behaviour and messages are unchanged.

In `collector.py`, move `Collector._row`'s body to a module-level function with the same docstring, and make the static method delegate:

```python
def privileged_row(
    state: dict[str, float] | None, keys: tuple[str, ...], previous: np.ndarray | None
) -> np.ndarray:
    """<the existing `_row` docstring, unchanged>"""
    if state:
        missing = [k for k in keys if k not in state]
        if missing:
            raise DataIntegrityError(f"privileged state lost keys mid-episode: {missing}")
        return np.asarray([state[k] for k in keys], dtype=np.float32)
    if previous is not None:
        return previous.copy()
    return np.zeros(len(keys), dtype=np.float32)
```

and in the class: `_row = staticmethod(privileged_row)`.

- [ ] **Step 4: Run the data suites**

Run: `PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/data -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/data/loader.py src/mbfps/data/collector.py tests/data/test_loader.py tests/data/test_collector.py
git commit -m "feat: the loader grows one episode at a time, and privileged_row is shared"
```

---

## Task 5: The DQN core

**Files:**
- Create: `src/mbfps/training/dqn.py`
- Test: `tests/training/test_dqn.py`

**Interfaces:**
- Produces:
  - `DQNConfig` (frozen dataclass): `gamma=0.99, n_step=3, target_every=2000, lr=1e-4, grad_clip=10.0, batch_size=256, reward_scale=0.01, eps_start=1.0, eps_end=0.05, eps_decay_steps=50_000, frame_stack=4, hidden=512`
  - `epsilon(budget_step: int, cfg: DQNConfig) -> float`
  - `QNetwork(n_actions: int, seed: int, frame_stack: int = 4, hidden: int = 512)`; `forward(stack (B, k, 64, 384)) -> (B, n_actions)`; attribute `encoder` (a `BottleneckEncoder`)
  - `stack_indices(t: int, k: int) -> list[int]`
  - `TransitionSampler(seed: int, cfg: DQNConfig)`: `.add(features, actions, rewards, terminated, truncated)` (features `(T+1, ...)` array or memmap); `.n_transitions`; `.sample(batch_size) -> dict` of numpy arrays `obs (B,k,...)`, `action (B,)`, `ret (B,)`, `discount (B,)`, `next_obs (B,k,...)`; `.rng` (a `numpy.random.Generator`)
  - `dqn_step(q, q_target, optimiser, batch: dict, update_index: int, cfg: DQNConfig, device) -> dict` (`loss`, `q_mean`); copies `q` into `q_target` when `update_index % cfg.target_every == 0`
  - `EpsilonGreedyQPolicy(q, encode, device, epsilon_fn: Callable[[], float], frame_stack=4, name="dqn")`: `reset(seed)`, `act(obs)`, `act_from_features(features)`, `get_state()`, `set_state(state)`

- [ ] **Step 1: Write the failing tests** (`tests/training/test_dqn.py`)

```python
import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from mbfps.training.dqn import (
    DQNConfig, EpsilonGreedyQPolicy, QNetwork, TransitionSampler, dqn_step, epsilon,
    stack_indices,
)
from mbfps.training.world_model import WorldModel
from mbfps.utils.config import get_config
from mbfps.utils.seeding import seed_everything


def test_epsilon_schedule():
    cfg = DQNConfig()
    assert epsilon(0, cfg) == 1.0
    assert epsilon(25_000, cfg) == pytest.approx(0.525)
    assert epsilon(50_000, cfg) == pytest.approx(0.05)
    assert epsilon(90_000, cfg) == pytest.approx(0.05)


def test_stack_indices_repeat_the_first_frame():
    assert stack_indices(0, 4) == [0, 0, 0, 0]
    assert stack_indices(2, 4) == [0, 0, 1, 2]
    assert stack_indices(9, 4) == [6, 7, 8, 9]


def test_q_encoder_matches_dreamers_encoder_init():
    """Same class, same width, same seeded initialisation (M5a §5.1)."""
    seed_everything(3)
    wm = WorldModel(get_config("frozen_ssl", seed=3, device="cpu", n_actions=4))
    q = QNetwork(4, seed=3)
    for (ka, a), (kb, b) in zip(wm.encoder.state_dict().items(), q.encoder.state_dict().items()):
        assert ka == kb and torch.equal(a, b)


def test_q_network_shapes():
    q = QNetwork(4, seed=0)
    out = q(torch.zeros(5, 4, 64, 384, dtype=torch.float16))
    assert out.shape == (5, 4)


def _episode(rewards, terminated, truncated=None, width=3):
    t = len(rewards)
    feats = np.arange(t + 1, dtype=np.float32)[:, None, None] * np.ones((1, 1, width), np.float32)
    return (feats, np.zeros(t, np.int64), np.asarray(rewards, np.float32),
            np.asarray(terminated, bool),
            np.zeros(t, bool) if truncated is None else np.asarray(truncated, bool))


def _all_rows(sampler, n=2000):
    b = sampler.sample(n)
    return {(int(b["obs"][i, -1, 0, 0]), ): (float(b["ret"][i]), float(b["discount"][i]),
                                            int(b["next_obs"][i, -1, 0, 0])) for i in range(n)}


def test_n_step_targets_stop_at_termination():
    cfg = DQNConfig(gamma=0.5, n_step=3, reward_scale=1.0)
    s = TransitionSampler(seed=0, cfg=cfg)
    s.add(*_episode([1, 2, 4, 8], [0, 0, 0, 1]))
    rows = _all_rows(s)
    # t=0: 1 + .5*2 + .25*4 = 3, bootstraps from obs 3 with .5**3
    assert rows[(0,)] == (3.0, 0.125, 3)
    # t=2: 4 + .5*8, terminated at t=3 -> no bootstrap
    assert rows[(2,)] == (8.0, 0.0, 4)
    # t=3: 8, terminated
    assert rows[(3,)] == (8.0, 0.0, 4)


def test_n_step_truncation_bootstraps_from_the_last_stored_obs():
    cfg = DQNConfig(gamma=0.5, n_step=3, reward_scale=1.0)
    s = TransitionSampler(seed=0, cfg=cfg)
    s.add(*_episode([1, 1], [0, 0], truncated=[0, 1]))
    rows = _all_rows(s)
    assert rows[(0,)] == (1.5, 0.25, 2)   # 2 steps to the end, bootstrap obs 2
    assert rows[(1,)] == (1.0, 0.5, 2)


def test_reward_scale_applies_to_targets():
    s = TransitionSampler(seed=0, cfg=DQNConfig(n_step=1))
    s.add(*_episode([97.0], [1]))
    assert float(s.sample(1)["ret"][0]) == pytest.approx(0.97)


def test_double_dqn_learns_a_chain():
    """Known answer (M5a §5.4). States 0,1,2 encoded one-hot; action 1 moves
    right, action 0 stays; moving right from 2 pays 1 and terminates. With
    gamma .9: Q*(s,1) = .9**(2-s), Q*(s,0) = .9 * Q*(s,1)."""
    torch.manual_seed(0)
    cfg = DQNConfig(gamma=0.9, n_step=1, reward_scale=1.0, target_every=50, lr=1e-2,
                    batch_size=64, frame_stack=1)
    rng = np.random.default_rng(0)
    s = TransitionSampler(seed=0, cfg=cfg)
    for _ in range(200):
        state, feats, acts, rews, term = 0, [np.eye(3)[0]], [], [], []
        while True:
            a = int(rng.integers(0, 2))
            done = state == 2 and a == 1
            nxt = state if a == 0 else min(state + 1, 2)
            acts.append(a); rews.append(1.0 if done else 0.0); term.append(done)
            feats.append(np.eye(3)[nxt]); state = nxt
            if done or len(acts) >= 12:
                break
        s.add(np.asarray(feats, np.float32)[:, None, :], np.asarray(acts), np.asarray(rews, np.float32),
              np.asarray(term), np.asarray([False] * (len(acts) - 1) + [not term[-1]]))
    q = nn.Sequential(nn.Flatten(), nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 2))
    q_target = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    for i in range(1, 3001):
        dqn_step(q, q_target, opt, s.sample(cfg.batch_size), i, cfg, torch.device("cpu"))
    with torch.no_grad():
        got = q(torch.eye(3)[:, None, None, :].float()).numpy()
    want = np.array([[0.9 * 0.81, 0.81], [0.9 * 0.9, 0.9], [0.9 * 1.0, 1.0]])
    np.testing.assert_allclose(got, want, atol=0.05)


def test_dqn_step_copies_target_on_schedule():
    cfg = DQNConfig(target_every=2, n_step=1, frame_stack=1)
    s = TransitionSampler(seed=0, cfg=cfg)
    s.add(*_episode([1.0, 0.0], [0, 1]))
    q = nn.Sequential(nn.Flatten(), nn.Linear(3, 2))
    q_target = copy.deepcopy(q)
    opt = torch.optim.SGD(q.parameters(), lr=0.1)
    dqn_step(q, q_target, opt, s.sample(4), 1, cfg, torch.device("cpu"))
    assert not torch.equal(q[1].weight, q_target[1].weight)
    dqn_step(q, q_target, opt, s.sample(4), 2, cfg, torch.device("cpu"))
    assert torch.equal(q[1].weight, q_target[1].weight)


def test_policy_repeats_first_frame_and_is_seeded():
    q = QNetwork(4, seed=0)
    feats = lambda v: np.full((64, 384), v, np.float16)
    p = EpsilonGreedyQPolicy(q, encode=None, device=torch.device("cpu"), epsilon_fn=lambda: 0.5)
    p.reset(seed=7)
    a = [p.act_from_features(feats(i)) for i in range(6)]
    p.reset(seed=7)
    b = [p.act_from_features(feats(i)) for i in range(6)]
    assert a == b
    p.reset(seed=7)
    p.act_from_features(feats(3))
    stack = p.get_state()["stack"]
    assert len(stack) == 4 and all(np.array_equal(f, feats(3)) for f in stack)
```

- [ ] **Step 2: Run to see them fail**

Run: `PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/training/test_dqn.py -q`
Expected: FAIL — `ModuleNotFoundError: mbfps.training.dqn`.

- [ ] **Step 3: Implement `src/mbfps/training/dqn.py`**

```python
"""The model-free baseline (M5a §5): Double DQN with n-step returns.

Reads the same frozen DINOv2 features as the Dreamer arm, through the same
`BottleneckEncoder` class at the same seeded initialisation, with a 4-frame
stack in place of Dreamer's recurrent state. Settings are standard (Rainbow-
lite) and were fixed before any run (§5.2)."""

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mbfps.models.encoders import build_encoder
from mbfps.utils.config import EncoderConfig
from mbfps.utils.seeding import fork_seed, seeded_init


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 0.99
    n_step: int = 3
    target_every: int = 2000
    lr: float = 1e-4
    grad_clip: float = 10.0
    batch_size: int = 256
    reward_scale: float = 0.01
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    frame_stack: int = 4
    hidden: int = 512


def epsilon(budget_step: int, cfg: DQNConfig) -> float:
    """Linear from `eps_start` at step 0 to `eps_end` at `eps_decay_steps`, then flat."""
    frac = min(1.0, budget_step / cfg.eps_decay_steps)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def stack_indices(t: int, k: int) -> list[int]:
    """Frame indices of the k-stack ending at `t`; the first frame repeats at an episode's start."""
    return [max(0, t - k + 1 + i) for i in range(k)]


class QNetwork(nn.Module):
    """Bottleneck per frame, the k embeddings concatenated, a 2 x 512 SiLU MLP."""

    def __init__(self, n_actions: int, seed: int, frame_stack: int = 4, hidden: int = 512) -> None:
        super().__init__()
        enc_cfg = EncoderConfig(kind="frozen_ssl")
        # The world model builds its encoder as the first module after
        # seed_everything(seed); drawing from a fresh generator at `seed`
        # reproduces that initialisation exactly (§5.1).
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = build_encoder(enc_cfg)
        self.frame_stack = frame_stack
        with seeded_init(seed, "q_head"):
            self.head = nn.Sequential(
                nn.Linear(frame_stack * enc_cfg.embed_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_actions))

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        b, k = stack.shape[:2]
        emb = self.encoder(stack.reshape(b * k, *stack.shape[2:]))
        return self.head(emb.reshape(b, -1))


class TransitionSampler:
    """Uniform transitions over every stored episode, with n-step targets.

    Slot `t` of an episode is the transition from obs[t] by actions[t]. The
    n-step return stops at a termination (no bootstrap) and otherwise
    bootstraps from obs[t + m], where m = min(n, T - t): truncation and the end
    of stored data both bootstrap from the last stored obs (§5.2)."""

    def __init__(self, seed: int, cfg: DQNConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._eps: list[tuple] = []
        self._ends: list[int] = []

    @property
    def n_transitions(self) -> int:
        return self._ends[-1] if self._ends else 0

    def add(self, features, actions, rewards, terminated, truncated) -> None:
        self._eps.append((features, np.asarray(actions), np.asarray(rewards, np.float32),
                          np.asarray(terminated, bool), np.asarray(truncated, bool)))
        self._ends.append(self.n_transitions + len(actions))

    def _target(self, e: int, t: int) -> tuple[float, float, int]:
        _, _, rewards, terminated, _ = self._eps[e]
        cfg, T = self.cfg, len(rewards)
        m = min(cfg.n_step, T - t)
        ret, done = 0.0, False
        for k in range(m):
            ret += (cfg.gamma ** k) * float(rewards[t + k]) * cfg.reward_scale
            if terminated[t + k]:
                done, m = True, k + 1
                break
        return ret, (0.0 if done else cfg.gamma ** m), t + m

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if not self._ends:
            raise ValueError("no transitions stored")
        k = self.cfg.frame_stack
        idx = self.rng.integers(0, self.n_transitions, batch_size)
        obs, nxt, act, ret, disc = [], [], [], [], []
        for i in idx:
            e = int(np.searchsorted(self._ends, i, side="right"))
            t = int(i - (self._ends[e - 1] if e else 0))
            feats, actions = self._eps[e][0], self._eps[e][1]
            r, d, n = self._target(e, t)
            obs.append(np.asarray(feats[stack_indices(t, k)]))
            nxt.append(np.asarray(feats[stack_indices(n, k)]))
            act.append(int(actions[t])); ret.append(r); disc.append(d)
        return {"obs": np.stack(obs), "next_obs": np.stack(nxt),
                "action": np.asarray(act, np.int64),
                "ret": np.asarray(ret, np.float32), "discount": np.asarray(disc, np.float32)}


def dqn_step(q, q_target, optimiser, batch: dict, update_index: int, cfg: DQNConfig,
             device: torch.device) -> dict[str, float]:
    """One Double-DQN update: the online net picks a*, the target net values it."""
    obs = torch.as_tensor(batch["obs"]).to(device)
    nxt = torch.as_tensor(batch["next_obs"]).to(device)
    act = torch.as_tensor(batch["action"]).to(device)
    ret = torch.as_tensor(batch["ret"]).to(device)
    disc = torch.as_tensor(batch["discount"]).to(device)
    with torch.no_grad():
        a_star = q(nxt).argmax(dim=-1, keepdim=True)
        target = ret + disc * q_target(nxt).gather(1, a_star).squeeze(1)
    q_sa = q(obs).gather(1, act[:, None]).squeeze(1)
    loss = F.smooth_l1_loss(q_sa, target, beta=1.0)
    optimiser.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip)
    optimiser.step()
    if update_index % cfg.target_every == 0:
        q_target.load_state_dict(q.state_dict())
    return {"loss": float(loss.detach()), "q_mean": float(q_sa.detach().mean())}


class EpsilonGreedyQPolicy:
    """ε-greedy over a frame stack. `epsilon_fn` is read at every action, so
    the behaviour policy follows the budget-step schedule; evaluation passes a
    constant. The ε draw uses its own generator, reseeded per episode."""

    def __init__(self, q, encode: Callable | None, device: torch.device,
                 epsilon_fn: Callable[[], float], frame_stack: int = 4, name: str = "dqn") -> None:
        self.q, self.encode, self.device = q, encode, device
        self.epsilon_fn, self.k, self.name = epsilon_fn, frame_stack, name
        self.reset(seed=0)

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(fork_seed(int(seed or 0), f"eps:{self.name}"))
        self._stack: deque = deque(maxlen=self.k)

    def act(self, obs: np.ndarray) -> int:
        return self.act_from_features(self.encode(obs[None])[0])

    @torch.no_grad()
    def act_from_features(self, features: np.ndarray) -> int:
        if not self._stack:
            self._stack.extend([features] * self.k)
        else:
            self._stack.append(features)
        explore = self._rng.random() < self.epsilon_fn()
        random_action = int(self._rng.integers(0, 4))
        if explore:
            return random_action
        stack = torch.as_tensor(np.stack(self._stack)[None]).to(self.device)
        return int(self.q(stack).argmax(-1))

    def get_state(self) -> dict:
        return {"stack": [np.asarray(f) for f in self._stack],
                "rng": self._rng.bit_generator.state}

    def set_state(self, state: dict) -> None:
        self._stack = deque(state["stack"], maxlen=self.k)
        self._rng.bit_generator.state = state["rng"]
```

Both random draws are taken on every action (the second discarded when not exploring), so the generator's stream does not depend on ε — the same discipline M3h's `_sample` documents. The action count 4 is `basic`'s; `random_action` reads it from the Q-network's output width instead if you prefer — but keep one draw per call either way.

- [ ] **Step 4: Run the DQN tests**

Run: `PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/training/test_dqn.py -q`
Expected: all PASS. If the chain test's tolerance is missed, report the learned table rather than loosening the tolerance.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/training/dqn.py tests/training/test_dqn.py
git commit -m "feat: Double DQN with n-step targets on the shared features, and its known-answer chain"
```

---

## Task 6: `OnlineRollout` — budget accounting, episodes across act phases, replay writes

**Files:**
- Create: `src/mbfps/training/online_rollout.py`
- Create: `tests/training/fake_env.py`
- Test: `tests/training/test_online_rollout.py`

**Interfaces:**
- Consumes: Task 4 `privileged_row`; `ReplayBuffer`, `Episode`, `feature_suffix`.
- Produces:
  - `train_episode_seed(run_seed: int, k: int) -> int` (= `2_000_000 + 100_000 * run_seed + k`, `k < 100_000` asserted)
  - `OnlineRollout(env, encode, replay: ReplayBuffer, run_seed: int, on_episode: Callable[[Path], None], backbone: str = "dinov2", policy_name: str = "")`; attributes `budget_step: int`, `episodes_done: int`, `in_progress: bool`; methods `step(policy) -> None`, `close_episode_truncated() -> None`, `state() -> dict`, `restore(state: dict, policy) -> None`
  - `act_with(policy, obs, features) -> int` (uses `act_from_features` when the policy has it, else `act(obs)`)
  - `tests/training/fake_env.py`: `FakeEnv` (deterministic, seedable, `privileged_state`, `scenario`, `action_space.n == 4`) and `fake_encode(frames) -> (N, 64, 384) float16`

- [ ] **Step 1: Write the fake environment** (`tests/training/fake_env.py`)

```python
"""A deterministic stand-in for ViZDoom `basic` (M5a loop tests).

reset(seed) fixes a target position in 0..7; action 1 moves left, 2 moves
right, 3 shoots. Shooting on target terminates with +97; otherwise -4 a step
(-9 for a miss); after `limit` steps the episode truncates. Frames encode the
position and step, so features differ step to step."""

import numpy as np
from gymnasium import spaces

from mbfps.envs.protocol import OBS_SHAPE


class FakeEnv:
    def __init__(self, limit: int = 12) -> None:
        self.scenario = "fake_basic"
        self.action_space = spaces.Discrete(4)
        self.limit = limit
        self._pos = self._target = self._t = 0

    def reset(self, *, seed=None):
        rng = np.random.default_rng(seed)
        self._target, self._pos, self._t = int(rng.integers(0, 8)), 4, 0
        return self._obs(), {}

    def step(self, action: int):
        self._t += 1
        reward, terminated = -4.0, False
        if action == 1:
            self._pos = max(0, self._pos - 1)
        elif action == 2:
            self._pos = min(7, self._pos + 1)
        elif action == 3:
            if self._pos == self._target:
                reward, terminated = 97.0, True
            else:
                reward = -9.0
        truncated = (not terminated) and self._t >= self.limit
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self):
        obs = np.zeros(OBS_SHAPE, dtype=np.uint8)
        obs[0, 0, 0], obs[0, 0, 1], obs[0, 0, 2] = self._pos, self._target, self._t
        return obs

    @property
    def privileged_state(self):
        return {"pos_x": float(self._pos), "pos_y": 0.0}

    def close(self) -> None:
        pass


def fake_encode(frames: np.ndarray) -> np.ndarray:
    out = np.zeros((len(frames), 64, 384), np.float16)
    out[:, 0, :3] = frames[:, 0, 0, :3]
    return out
```

- [ ] **Step 2: Write the failing tests** (`tests/training/test_online_rollout.py`)

```python
import numpy as np
import pytest

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import load_episode
from mbfps.data.loader import feature_suffix
from mbfps.data.policies import RandomPolicy
from mbfps.training.online_rollout import OnlineRollout, act_with, train_episode_seed
from tests.training.fake_env import FakeEnv, fake_encode


def _rollout(tmp_path, seen):
    replay = ReplayBuffer(tmp_path / "replay", capacity_transitions=10**9)
    return OnlineRollout(FakeEnv(), fake_encode, replay, run_seed=1,
                         on_episode=seen.append, policy_name="random")


def test_train_episode_seeds_are_disjoint_from_eval_seeds():
    assert train_episode_seed(0, 0) == 2_000_000
    assert train_episode_seed(4, 99_999) == 2_499_999
    with pytest.raises(AssertionError):
        train_episode_seed(0, 100_000)


def test_steps_count_the_budget_and_write_whole_episodes(tmp_path):
    seen = []
    r = _rollout(tmp_path, seen)
    policy = RandomPolicy(4, seed=0)
    for _ in range(50):
        r.step(policy)
    assert r.budget_step == 50
    total = sum(load_episode(p).length for p in seen)
    assert total + (r.current_length if r.in_progress else 0) == 50
    for p in seen:
        ep = load_episode(p)
        feats = np.load(p.with_suffix(feature_suffix("dinov2")))
        assert feats.shape == (ep.length + 1, 64, 384)
        assert ep.terminated[-1] or ep.truncated[-1]
        np.testing.assert_array_equal(feats, fake_encode(ep.obs))


def test_episode_seeds_follow_the_run_seed(tmp_path):
    seen = []
    r = _rollout(tmp_path, seen)
    policy = RandomPolicy(4, seed=0)
    while len(seen) < 3:
        r.step(policy)
    assert [load_episode(p).seed for p in seen] == [train_episode_seed(1, k) for k in range(3)]


def test_close_budget_stores_the_tail_truncated(tmp_path):
    seen = []
    r = _rollout(tmp_path, seen)
    policy = RandomPolicy(4, seed=0)
    for _ in range(5):
        r.step(policy)
    assert r.in_progress
    r.close_episode_truncated()
    ep = load_episode(seen[-1])
    assert ep.length == 5 and ep.truncated[-1] and not ep.terminated.any()
    assert not r.in_progress and r.budget_step == 5


def test_restore_replays_the_engine_to_the_same_point(tmp_path):
    seen_a, seen_b = [], []
    a = _rollout(tmp_path / "a", seen_a)
    pa = RandomPolicy(4, seed=0)
    for _ in range(7):
        a.step(pa)
    state = a.state()
    b = _rollout(tmp_path / "b", seen_b)
    pb = RandomPolicy(4, seed=0)
    b.restore(state, pb)
    pb._rng.bit_generator.state = pa._rng.bit_generator.state
    for _ in range(20):
        a.step(pa)
        b.step(pb)
    assert seen_b, "no episode finished after the restore"
    assert load_episode(seen_a[-1]).actions.tolist() == load_episode(seen_b[-1]).actions.tolist()
    assert a.budget_step == b.budget_step


def test_act_with_prefers_features():
    class P:
        def act_from_features(self, f): return 3
        def act(self, obs): return 1
    class Q:
        def act(self, obs): return 1
    assert act_with(P(), None, None) == 3 and act_with(Q(), None, None) == 1
```

`tests/training/__init__.py` exists, so `tests.training.fake_env` imports when pytest runs from the repo root. If the import fails, add `tests/__init__.py` (empty) and note it in the report.

- [ ] **Step 3: Run to see them fail**

Run: `PYTHONPATH=src:. /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/training/test_online_rollout.py -q`
Expected: FAIL — `ModuleNotFoundError: mbfps.training.online_rollout`.

- [ ] **Step 4: Implement `src/mbfps/training/online_rollout.py`**

```python
"""Experience collection for the online loop (M5a §3.1, §3.3, §3.4).

Every `env.step` here is one budget step. An episode in progress when an act
phase ends simply continues on the next `step` call. A finished episode is
written once, whole, with its features alongside, and announced through
`on_episode`. The engine cannot be snapshotted mid-episode, so `state()`
records the episode's seed and actions and `restore()` replays them (M0
proved an episode replays exactly from its actions)."""

from pathlib import Path
from typing import Callable

import numpy as np

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.collector import privileged_row
from mbfps.data.episode import Episode
from mbfps.data.loader import feature_suffix

_EPISODE_SEED_BASE = 2_000_000
_EPISODES_PER_RUN = 100_000


def train_episode_seed(run_seed: int, k: int) -> int:
    """Seed of the run's k-th training episode, disjoint from EVAL_SEEDS."""
    assert 0 <= k < _EPISODES_PER_RUN, k
    return _EPISODE_SEED_BASE + _EPISODES_PER_RUN * int(run_seed) + int(k)


def act_with(policy, obs, features) -> int:
    if hasattr(policy, "act_from_features"):
        return int(policy.act_from_features(features))
    return int(policy.act(obs))


class OnlineRollout:
    def __init__(self, env, encode: Callable[[np.ndarray], np.ndarray], replay: ReplayBuffer,
                 run_seed: int, on_episode: Callable[[Path], None],
                 backbone: str = "dinov2", policy_name: str = "") -> None:
        self.env, self.encode, self.replay = env, encode, replay
        self.run_seed, self.on_episode = run_seed, on_episode
        self.backbone, self.policy_name = backbone, policy_name
        self.budget_step = 0
        self.episodes_done = 0
        self._ep: dict | None = None

    @property
    def in_progress(self) -> bool:
        return self._ep is not None

    @property
    def current_length(self) -> int:
        return len(self._ep["actions"]) if self._ep else 0

    def _start(self, policy) -> None:
        seed = train_episode_seed(self.run_seed, self.episodes_done)
        obs, _ = self.env.reset(seed=seed)
        policy.reset(seed=seed)
        state = self.env.privileged_state
        keys = tuple(state) if state else ()
        self._ep = {"seed": seed, "keys": keys, "obs": [obs.copy()],
                    "features": [self.encode(obs[None])[0]],
                    "privileged": [privileged_row(state, keys, previous=None)],
                    "actions": [], "rewards": [], "terminated": [], "truncated": []}

    def step(self, policy) -> None:
        if self._ep is None:
            self._start(policy)
        ep = self._ep
        action = act_with(policy, ep["obs"][-1], ep["features"][-1])
        obs, reward, terminated, truncated, _ = self.env.step(action)
        self.budget_step += 1
        self._record(action, obs, reward, terminated, truncated)
        if terminated or truncated:
            self._finish()

    def _record(self, action, obs, reward, terminated, truncated) -> None:
        ep = self._ep
        ep["actions"].append(int(action)); ep["rewards"].append(float(reward))
        ep["terminated"].append(bool(terminated)); ep["truncated"].append(bool(truncated))
        ep["obs"].append(obs.copy())
        ep["features"].append(self.encode(obs[None])[0])
        ep["privileged"].append(privileged_row(self.env.privileged_state, ep["keys"],
                                               previous=ep["privileged"][-1]))

    def close_episode_truncated(self) -> None:
        """End the in-progress episode as truncated (prefill end, budget end)."""
        if self._ep is not None and self._ep["actions"]:
            self._ep["truncated"][-1] = True
            self._finish()
        self._ep = None

    def _finish(self) -> None:
        ep = self._ep
        episode = Episode(
            obs=np.stack(ep["obs"]), actions=np.asarray(ep["actions"], np.int32),
            rewards=np.asarray(ep["rewards"], np.float32),
            terminated=np.asarray(ep["terminated"], bool),
            truncated=np.asarray(ep["truncated"], bool),
            privileged=np.stack(ep["privileged"]).astype(np.float32),
            privileged_keys=ep["keys"], policy_name=self.policy_name,
            seed=ep["seed"], scenario=self.env.scenario)
        path = self.replay.add(episode)
        np.save(path.with_suffix(feature_suffix(self.backbone)), np.stack(ep["features"]))
        self.episodes_done += 1
        self._ep = None
        self.on_episode(path)

    def state(self) -> dict:
        ep = self._ep
        return {"budget_step": self.budget_step, "episodes_done": self.episodes_done,
                "episode": None if ep is None else {
                    "seed": ep["seed"], "actions": list(ep["actions"])}}

    def restore(self, state: dict, policy) -> None:
        """Replay the in-progress episode's actions into the engine, re-encoding
        its frames. The policy's internal state is restored by the caller
        (its own `set_state`); `policy.reset` is called here only to mirror
        `_start`, and the caller overwrites the state afterwards."""
        self.budget_step = int(state["budget_step"])
        self.episodes_done = int(state["episodes_done"])
        self._ep = None
        saved = state["episode"]
        if saved is None:
            return
        self._start(policy)
        assert self._ep["seed"] == saved["seed"], (self._ep["seed"], saved["seed"])
        for action in saved["actions"]:
            obs, reward, terminated, truncated, _ = self.env.step(int(action))
            self._record(action, obs, reward, terminated, truncated)
```

- [ ] **Step 5: Run the rollout tests**

Run: `PYTHONPATH=src:. /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/training/test_online_rollout.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/training/online_rollout.py tests/training/fake_env.py tests/training/test_online_rollout.py
git commit -m "feat: OnlineRollout counts budget steps, carries episodes across act phases, and resumes by replay"
```

---

## Task 7: The two arms

**Files:**
- Modify: `src/mbfps/eval/live.py` (`AgentPolicy`: `act_from_features`, `get_state`, `set_state`)
- Create: `src/mbfps/training/arms.py`
- Test: `tests/training/test_arms.py`, `tests/eval/test_live.py`

**Interfaces:**
- Consumes: Task 2 `ActorCriticLearner`; Task 3 `world_model_step`; Task 4 `SequenceLoader.add`; Task 5 DQN pieces.
- Produces (both arms):
  - attributes `name: str` (`"dreamer"` / `"dqn"`), `gated_mode: str` (`"sample"` / `"greedy"`), `isolation_ok: bool | None`
  - `behaviour_policy(budget_step_fn: Callable[[], int])` → a persistent policy object with `reset`, `act`, `act_from_features`, `get_state`, `set_state`
  - `add_episode(path: Path) -> None`
  - `update_round() -> dict[str, float]`
  - `eval_policies() -> dict[str, policy]` (gated mode first)
  - `diagnostics() -> dict` (means since the last call, then reset)
  - `state_dict() -> dict`; `load_state_dict(sd: dict) -> None` (after the same episodes were re-added)
  - constructors `DreamerArm(seed, device, encode, replay, batch_size=16, seq_len=64)`, `DQNArm(seed, device, encode, cfg: DQNConfig = DQNConfig())`
- `AgentPolicy.act_from_features(features (64, 384)) -> int`; `act(obs)` = `act_from_features(encode(obs[None])[0])`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_live.py`:

```python
def test_act_from_features_matches_act():
    model, agent = _agent()
    frame = np.zeros((112, 112, 3), dtype=np.uint8)
    agent.reset(seed=3)
    a = [agent.act(frame) for _ in range(4)]
    agent.reset(seed=3)
    b = [agent.act_from_features(frame) for _ in range(4)]   # cnn kind: features are pixels
    assert a == b


def test_agent_state_round_trips():
    model, agent = _agent()
    frame = np.zeros((112, 112, 3), dtype=np.uint8)
    agent.reset(seed=1)
    agent.act(frame)
    s = agent.get_state()
    agent.act(frame)
    agent.set_state(s)
    assert agent.get_state()["last_action"] == s["last_action"]
    assert torch.equal(agent.get_state()["h"], s["h"])
```

Create `tests/training/test_arms.py`:

```python
import numpy as np
import pytest
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.policies import RandomPolicy
from mbfps.training.arms import DQNArm, DreamerArm
from mbfps.training.dqn import DQNConfig
from mbfps.training.online_rollout import OnlineRollout
from tests.training.fake_env import FakeEnv, fake_encode

CPU = torch.device("cpu")


def _fill(tmp_path, arm, steps=60):
    replay = ReplayBuffer(tmp_path / "replay", capacity_transitions=10**9)
    r = OnlineRollout(FakeEnv(), fake_encode, replay, run_seed=0, on_episode=arm.add_episode)
    p = RandomPolicy(4, seed=0)
    for _ in range(steps):
        r.step(p)
    r.close_episode_truncated()
    return replay


def _dreamer(tmp_path):
    replay = ReplayBuffer(tmp_path / "replay", capacity_transitions=10**9)
    return DreamerArm(seed=0, device=CPU, encode=fake_encode, replay=replay,
                      batch_size=2, seq_len=8)


def test_reward_bounds_track_observed_rewards(tmp_path):
    arm = _dreamer(tmp_path)
    assert arm.reward_bounds is None
    _fill(tmp_path, arm, steps=40)
    lo, hi = arm.reward_bounds
    assert lo <= -4.0 and hi >= -4.0


def test_dreamer_round_trains_world_model_and_keeps_isolation(tmp_path):
    arm = _dreamer(tmp_path)
    _fill(tmp_path, arm)
    before = [p.detach().clone() for p in arm.world_model.parameters()]
    out = arm.update_round()
    assert arm.isolation_ok is True
    assert any(not torch.equal(b, p) for b, p in zip(before, arm.world_model.parameters()))
    for key in ("wm_loss", "actor_loss", "critic_loss", "n_dropped", "clipped_share"):
        assert key in out


def test_dreamer_state_round_trips(tmp_path):
    a = _dreamer(tmp_path / "a")
    _fill(tmp_path / "a", a)
    a.update_round()
    b = _dreamer(tmp_path / "b")
    _fill(tmp_path / "b", b)
    b.load_state_dict(a.state_dict())
    torch.manual_seed(5)
    out_a = a.update_round()
    torch.manual_seed(5)
    out_b = b.update_round()
    assert out_a == out_b


def test_dqn_round_and_state_round_trip(tmp_path):
    cfg = DQNConfig(batch_size=8)
    a = DQNArm(seed=0, device=CPU, encode=fake_encode, cfg=cfg)
    _fill(tmp_path / "a", a)
    a.update_round()
    b = DQNArm(seed=0, device=CPU, encode=fake_encode, cfg=cfg)
    _fill(tmp_path / "b", b)
    b.load_state_dict(a.state_dict())
    assert a.update_round() == b.update_round()


def test_eval_policies_put_the_gated_mode_first(tmp_path):
    d = _dreamer(tmp_path / "d")
    q = DQNArm(seed=0, device=CPU, encode=fake_encode)
    assert list(d.eval_policies())[0] == d.gated_mode == "sample"
    assert list(q.eval_policies())[0] == q.gated_mode == "greedy"
    assert set(d.eval_policies()) == {"sample", "argmax"}
    assert set(q.eval_policies()) == {"greedy", "eps0.05"}


def test_dqn_behaviour_policy_follows_the_schedule(tmp_path):
    q = DQNArm(seed=0, device=CPU, encode=fake_encode)
    step = {"n": 0}
    p = q.behaviour_policy(lambda: step["n"])
    assert p.epsilon_fn() == 1.0
    step["n"] = 50_000
    assert p.epsilon_fn() == pytest.approx(0.05)
```

The `DreamerArm` here uses the `frozen_ssl` world model on `fake_encode` features (64 × 384), so it exercises the real encoder path. `fake_encode` writes features with the dinov2 suffix, which is what the loader reads for `frozen_ssl`.

- [ ] **Step 2: Run to see them fail**

Run: `PYTHONPATH=src:. /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/training/test_arms.py tests/eval/test_live.py -q`
Expected: FAIL — `ModuleNotFoundError: mbfps.training.arms`, `no attribute 'act_from_features'`.

- [ ] **Step 3: Add to `AgentPolicy`** (`src/mbfps/eval/live.py`)

Replace `act` with:

```python
    def act(self, obs: np.ndarray) -> int:
        return self.act_from_features(self.encode(obs[None])[0])

    @torch.no_grad()
    def act_from_features(self, features: np.ndarray) -> int:
        """`act` for a frame whose backbone features are already computed
        (M5a: the online loop encodes each frame once, for the replay and the
        policy). Identical arithmetic to M4's `act`."""
        inputs = torch.as_tensor(np.asarray(features)[None]).to(self.device)
        embedding = self.world_model.encoder(inputs)[:, None]           # (1, 1, E)
        action_in = torch.tensor([[self._last_action]], device=self.device)
        out = self.world_model.rssm.observe(embedding, action_in, state=self._state)
        self._state = (out["h"][:, -1], out["z"][:, -1])
        latent = out["latent"][:, -1]
        if self._first and self.critic_slow is not None:
            self.imagined_start_return = self._imagined_return()
        self._first = False
        dist = self.actor.dist(latent)
        action = int(dist.sample() if self.mode == "sample" else dist.probs.argmax(-1))
        self._last_action = action
        return action

    def get_state(self) -> dict:
        return {"h": self._state[0].detach().cpu(), "z": self._state[1].detach().cpu(),
                "last_action": self._last_action, "first": self._first,
                "imagined_start_return": self.imagined_start_return}

    def set_state(self, state: dict) -> None:
        self._state = (state["h"].to(self.device), state["z"].to(self.device))
        self._last_action = int(state["last_action"])
        self._first = bool(state["first"])
        self.imagined_start_return = state["imagined_start_return"]
```

Check: M4's `tests/eval/test_live.py` and `test_eval_live_script.py` must pass unchanged — the arithmetic is the same.

- [ ] **Step 4: Implement `src/mbfps/training/arms.py`**

```python
"""The two arms of M5a behind one interface (spec §4, §5).

Both read frozen DINOv2 features (the `frozen_ssl` cache suffix) and are built
from the run seed. `update_round()` is one update round of §3.2; the driver
calls it 20 times per 100 budget steps."""

import copy
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.loader import SequenceLoader, feature_suffix
from mbfps.eval.live import AgentPolicy
from mbfps.training.ac_trainer import ACConfig, ActorCriticLearner
from mbfps.training.dqn import (DQNConfig, EpsilonGreedyQPolicy, QNetwork,
                                TransitionSampler, dqn_step, epsilon)
from mbfps.training.world_model import WorldModel, world_model_step
from mbfps.utils.config import get_config
from mbfps.utils.device import to_device
from mbfps.utils.seeding import seed_everything

BACKBONE = "dinov2"
N_ACTIONS = 4


class _Means:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.n = 0

    def add(self, d: dict[str, float]) -> None:
        for k, v in d.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
        self.n += 1

    def pop(self) -> dict[str, float]:
        out = {k: v / self.n for k, v in self.sums.items()} if self.n else {}
        self.sums, self.n = {}, 0
        return out


class DreamerArm:
    name, gated_mode = "dreamer", "sample"

    def __init__(self, seed: int, device: torch.device, encode, replay: ReplayBuffer,
                 batch_size: int = 16, seq_len: int = 64) -> None:
        seed_everything(seed)   # the world model is built first after this, as in M3/M4
        self.device, self.encode = device, encode
        self.world_model = WorldModel(get_config(
            "frozen_ssl", seed=seed, device=device.type, n_actions=N_ACTIONS,
            pad_short=True, reward_transform="symlog")).to(device)
        self.wm_opt = torch.optim.Adam(self.world_model.parameters(),
                                       lr=self.world_model.cfg.train.lr)
        self.learner = ActorCriticLearner(
            N_ACTIONS, ACConfig(seed=seed, device=device.type, batch_size=batch_size,
                                seq_len=seq_len, bound_reward_readout=True,
                                drop_terminal_starts=True), device)
        self.loader = SequenceLoader(replay, batch_size=batch_size, seq_len=seq_len, seed=seed,
                                     load_obs=False, load_features=True,
                                     feature_backbone=BACKBONE, paths=[], pad_short=True)
        self.reward_bounds: tuple[float, float] | None = None
        self.isolation_ok: bool | None = None
        self._means = _Means()
        self._replay_transitions = 0

    def add_episode(self, path: Path) -> None:
        self.loader.add(path)
        rewards = np.load(path)["rewards"]
        self._replay_transitions += len(rewards)
        lo, hi = float(rewards.min()), float(rewards.max())
        if self.reward_bounds is not None:
            lo, hi = min(lo, self.reward_bounds[0]), max(hi, self.reward_bounds[1])
        self.reward_bounds = (lo, hi)

    def update_round(self) -> dict[str, float]:
        batch = to_device(self.loader.sample(), self.device)
        wm_loss, parts = world_model_step(self.world_model, self.wm_opt, batch)
        ac = self.learner.step(self.world_model, batch, reward_bounds=self.reward_bounds)
        if self.isolation_ok is None:
            self.isolation_ok = all(p.grad is None for p in self.world_model.parameters())
            if not self.isolation_ok:
                raise RuntimeError("an actor-critic step put a gradient on the world model (§3.5)")
        out = {"wm_loss": wm_loss, **{f"wm_{k}": v for k, v in parts.items()}, **ac}
        self._means.add(out)
        return out

    def behaviour_policy(self, budget_step_fn: Callable[[], int]) -> AgentPolicy:
        return AgentPolicy(self.world_model, self.learner.actor, self.encode, self.device,
                           mode="sample")

    def eval_policies(self) -> dict:
        kw = {"critic_slow": self.learner.critic_slow}
        return {"sample": AgentPolicy(self.world_model, self.learner.actor, self.encode,
                                      self.device, mode="sample", **kw),
                "argmax": AgentPolicy(self.world_model, self.learner.actor, self.encode,
                                      self.device, mode="argmax", **kw)}

    def diagnostics(self) -> dict:
        return {**self._means.pop(), "reward_bounds": self.reward_bounds,
                "replay_transitions": self._replay_transitions,
                "replay_episodes": len(self.loader._paths)}

    def state_dict(self) -> dict:
        return {"world_model": self.world_model.state_dict(), "wm_opt": self.wm_opt.state_dict(),
                "learner": self.learner.state_dict(),
                "loader_rng": self.loader._rng.bit_generator.state,
                "isolation_ok": self.isolation_ok}

    def load_state_dict(self, sd: dict) -> None:
        self.world_model.load_state_dict(sd["world_model"])
        self.wm_opt.load_state_dict(sd["wm_opt"])
        self.learner.load_state_dict(sd["learner"])
        self.loader._rng.bit_generator.state = sd["loader_rng"]
        self.isolation_ok = sd["isolation_ok"]


class DQNArm:
    name, gated_mode = "dqn", "greedy"

    def __init__(self, seed: int, device: torch.device, encode,
                 cfg: DQNConfig = DQNConfig()) -> None:
        seed_everything(seed)
        self.device, self.encode, self.cfg = device, encode, cfg
        self.q = QNetwork(N_ACTIONS, seed=seed, frame_stack=cfg.frame_stack,
                          hidden=cfg.hidden).to(device)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.sampler = TransitionSampler(seed=seed, cfg=cfg)
        self.updates = 0
        self.isolation_ok = None   # not applicable to DQN
        self._means = _Means()
        self._budget_step_fn: Callable[[], int] = lambda: 0

    def add_episode(self, path: Path) -> None:
        with np.load(path) as d:
            arrays = (d["actions"], d["rewards"], d["terminated"], d["truncated"])
        features = np.load(path.with_suffix(feature_suffix(BACKBONE)), mmap_mode="r")
        self.sampler.add(features, *arrays)

    def update_round(self) -> dict[str, float]:
        self.updates += 1
        out = dqn_step(self.q, self.q_target, self.opt, self.sampler.sample(self.cfg.batch_size),
                       self.updates, self.cfg, self.device)
        self._means.add(out)
        return out

    def behaviour_policy(self, budget_step_fn: Callable[[], int]) -> EpsilonGreedyQPolicy:
        self._budget_step_fn = budget_step_fn
        return EpsilonGreedyQPolicy(self.q, self.encode, self.device,
                                    epsilon_fn=lambda: epsilon(budget_step_fn(), self.cfg),
                                    frame_stack=self.cfg.frame_stack, name="dqn")

    def eval_policies(self) -> dict:
        k = self.cfg.frame_stack
        return {"greedy": EpsilonGreedyQPolicy(self.q, self.encode, self.device, lambda: 0.0,
                                               frame_stack=k, name="dqn_greedy"),
                "eps0.05": EpsilonGreedyQPolicy(self.q, self.encode, self.device, lambda: 0.05,
                                                frame_stack=k, name="dqn_eps")}

    def diagnostics(self) -> dict:
        return {**self._means.pop(), "updates": self.updates,
                "epsilon": epsilon(self._budget_step_fn(), self.cfg),
                "replay_transitions": self.sampler.n_transitions}

    def state_dict(self) -> dict:
        return {"q": self.q.state_dict(), "q_target": self.q_target.state_dict(),
                "opt": self.opt.state_dict(), "updates": self.updates,
                "sampler_rng": self.sampler.rng.bit_generator.state}

    def load_state_dict(self, sd: dict) -> None:
        self.q.load_state_dict(sd["q"])
        self.q_target.load_state_dict(sd["q_target"])
        self.opt.load_state_dict(sd["opt"])
        self.updates = int(sd["updates"])
        self.sampler.rng.bit_generator.state = sd["sampler_rng"]
```

Add a public accessor instead of reaching into the loader's privates if you prefer (`SequenceLoader.n_episodes`, `SequenceLoader.rng`); if so, add it in this task with a one-line test.

- [ ] **Step 5: Run the arm and live suites**

Run: `PYTHONPATH=src:. /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/training/test_arms.py tests/eval/test_live.py tests/eval/test_eval_live_script.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/live.py src/mbfps/training/arms.py tests/training/test_arms.py tests/eval/test_live.py
git commit -m "feat: DreamerArm and DQNArm behind one interface, each resumable"
```

---

## Task 8: The driver — `train_online`, evaluation, resumable checkpoints, records

**Files:**
- Create: `src/mbfps/training/online.py`
- Test: `tests/training/test_online.py`

**Interfaces:**
- Consumes: Tasks 6–7; `mbfps.eval.live.run_episodes`, `EVAL_SEEDS`; `mbfps.eval.study.write_record`, `load_record`, `git_sha`; `RandomPolicy`.
- Produces:
  - `OnlineConfig` (frozen dataclass): `budget=100_000, prefill=5_000, act_steps=100, rounds_per_act=20, eval_every=10_000, eval_seeds=EVAL_SEEDS`
  - `train_online(arm_name: str, seed: int, out_dir: Path, env_factory, encode, device, cfg: OnlineConfig = OnlineConfig(), arm_kwargs: dict | None = None, stop_after_step: int | None = None) -> dict` — returns the record; resumes from `out_dir/state.pt` when present
  - record file `out_dir/record.json` keys: `arm, seed, gated_mode, seeds, random, evals` (list of `{"step", "modes": {mode: episodes}, "diagnostics", "update_rounds"}`), `budget_steps, update_rounds, isolation_ok, complete, seconds, timing, git_sha, config`
  - `checkpoint_steps(cfg) -> list[int]` = `[0, eval_every, …, budget]`

- [ ] **Step 1: Write the failing tests** (`tests/training/test_online.py`)

```python
import json

import numpy as np
import pytest
import torch

from mbfps.data.episode import load_episode
from mbfps.eval.study import load_record
from mbfps.training.dqn import DQNConfig
from mbfps.training.online import OnlineConfig, train_online
from tests.training.fake_env import FakeEnv, fake_encode

CPU = torch.device("cpu")
TINY = OnlineConfig(budget=60, prefill=10, act_steps=10, rounds_per_act=2, eval_every=20,
                    eval_seeds=(1_000_000, 1_000_001, 1_000_002))
DREAMER_KW = {"batch_size": 2, "seq_len": 8}


def _run(tmp_path, arm="dreamer", stop=None, kw=None):
    kw = DREAMER_KW if (arm == "dreamer" and kw is None) else (kw or {})
    return train_online(arm, seed=0, out_dir=tmp_path, env_factory=FakeEnv, encode=fake_encode,
                        device=CPU, cfg=TINY, arm_kwargs=kw, stop_after_step=stop)


@pytest.mark.parametrize("arm", ["dreamer", "dqn"])
def test_run_spends_exactly_the_budget(tmp_path, arm):
    kw = {"cfg": DQNConfig(batch_size=8)} if arm == "dqn" else None
    rec = _run(tmp_path, arm, kw=kw)
    assert rec["complete"] is True
    assert rec["budget_steps"] == 60
    assert [e["step"] for e in rec["evals"]] == [0, 20, 40, 60]
    assert rec["update_rounds"] == (60 - 10) // 10 * 2
    stored = sum(load_episode(p).length for p in (tmp_path / "replay").glob("ep_*.npz"))
    assert stored == 60
    for e in rec["evals"]:
        for mode, eps in e["modes"].items():
            assert [x["seed"] for x in eps] == list(TINY.eval_seeds)
    assert [x["seed"] for x in rec["random"]] == list(TINY.eval_seeds)


def test_prefill_cuts_at_exactly_the_prefill_step(tmp_path):
    _run(tmp_path)
    eps = sorted((tmp_path / "replay").glob("ep_*.npz"))
    lengths = np.cumsum([load_episode(p).length for p in eps])
    assert 10 in lengths.tolist()          # an episode boundary at the prefill step
    first_arm = eps[lengths.tolist().index(10) + 1]
    assert load_episode(first_arm).policy_name == "dreamer"


def test_resume_is_bit_identical(tmp_path):
    full = _run(tmp_path / "full")
    part = _run(tmp_path / "resumed", stop=20)
    assert part["complete"] is False
    resumed = _run(tmp_path / "resumed")
    strip = lambda r: {k: v for k, v in r.items() if k not in ("seconds", "timing", "git_sha")}
    assert json.dumps(strip(full), sort_keys=True) == json.dumps(strip(resumed), sort_keys=True)
    a = torch.load(tmp_path / "full" / "state.pt", weights_only=False)["arm"]["world_model"]
    b = torch.load(tmp_path / "resumed" / "state.pt", weights_only=False)["arm"]["world_model"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_evaluation_steps_never_enter_the_replay(tmp_path):
    rec = _run(tmp_path)
    seeds = {load_episode(p).seed for p in (tmp_path / "replay").glob("ep_*.npz")}
    assert not seeds & set(TINY.eval_seeds)


def test_isolation_recorded_for_dreamer(tmp_path):
    assert _run(tmp_path)["isolation_ok"] is True
```

- [ ] **Step 2: Run to see them fail**

Run: `PYTHONPATH=src:. /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/training/test_online.py -q`
Expected: FAIL — `ModuleNotFoundError: mbfps.training.online`.

- [ ] **Step 3: Implement `src/mbfps/training/online.py`**

```python
"""The M5a driver (spec §3): prefill, then act 100 / train 20 until the budget
is spent, evaluating at step 0 and every 10,000 budget steps. After each
evaluation the full state is written; a rerun resumes from it and is
bit-identical to an uninterrupted run (§3.4)."""

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.policies import RandomPolicy
from mbfps.eval.live import EVAL_SEEDS, run_episodes
from mbfps.eval.study import git_sha, load_record, write_record
from mbfps.training.arms import DQNArm, DreamerArm
from mbfps.training.online_rollout import OnlineRollout


@dataclass(frozen=True)
class OnlineConfig:
    budget: int = 100_000
    prefill: int = 5_000
    act_steps: int = 100
    rounds_per_act: int = 20
    eval_every: int = 10_000
    eval_seeds: tuple[int, ...] = EVAL_SEEDS

    def __post_init__(self) -> None:
        if (self.budget - self.prefill) % self.act_steps or self.eval_every % self.act_steps \
                or self.prefill % self.act_steps or self.budget % self.eval_every:
            raise ValueError("prefill, eval_every and budget must align with act_steps "
                             "and budget with eval_every, so every checkpoint is reached exactly")


def checkpoint_steps(cfg: OnlineConfig) -> list[int]:
    return list(range(0, cfg.budget + 1, cfg.eval_every))


def _make_arm(name: str, seed: int, device, encode, replay, kwargs):
    if name == "dreamer":
        return DreamerArm(seed, device, encode, replay, **kwargs)
    if name == "dqn":
        return DQNArm(seed, device, encode, **kwargs)
    raise ValueError(f"unknown arm {name!r}")


def _rng_state(device) -> dict:
    out = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch": torch.get_rng_state()}
    if device.type == "mps":
        out["mps"] = torch.mps.get_rng_state()
    return out


def _set_rng_state(state: dict, device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "mps":
        torch.mps.set_rng_state(state["mps"])


def _evaluate(arm, env, seeds) -> dict:
    return {mode: run_episodes(env, policy, seeds) for mode, policy in arm.eval_policies().items()}


def train_online(arm_name: str, seed: int, out_dir: Path, env_factory, encode, device,
                 cfg: OnlineConfig = OnlineConfig(), arm_kwargs: dict | None = None,
                 stop_after_step: int | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path, record_path = out_dir / "state.pt", out_dir / "record.json"
    replay = ReplayBuffer(out_dir / "replay", capacity_transitions=10**9)
    arm = _make_arm(arm_name, seed, device, encode, replay, arm_kwargs or {})
    train_env, eval_env = env_factory(), env_factory()
    rollout = OnlineRollout(train_env, encode, replay, run_seed=seed,
                            on_episode=arm.add_episode, policy_name=arm.name)
    policy = arm.behaviour_policy(lambda: rollout.budget_step)
    timing = {"act": 0.0, "train": 0.0, "eval": 0.0}
    seeds = list(cfg.eval_seeds)

    if state_path.exists():
        # weights_only=False: the state holds NumPy RNG states and frame stacks,
        # which the restricted unpickler refuses. The file is written only by
        # `_checkpoint` below, into this run's own directory, and never shipped.
        saved = torch.load(state_path, weights_only=False)
        for path in saved["replay_paths"]:
            arm.add_episode(Path(path))
        arm.load_state_dict(saved["arm"])
        rollout.restore(saved["rollout"], policy)
        if saved["policy"] is not None:
            policy.set_state(saved["policy"])
        _set_rng_state(saved["rng"], device)
        record = load_record(record_path)
        update_rounds = record["update_rounds"]
        timing = record["timing"]
    else:
        prefill_policy = RandomPolicy(4, seed=seed)
        rollout.policy_name = "random"
        t0 = time.perf_counter()
        while rollout.budget_step < cfg.prefill:
            rollout.step(prefill_policy)
        rollout.close_episode_truncated()
        rollout.policy_name = arm.name
        timing["act"] += time.perf_counter() - t0
        record = {"arm": arm.name, "seed": seed, "gated_mode": arm.gated_mode, "seeds": seeds,
                  "config": asdict(cfg), "evals": [], "complete": False,
                  "random": run_episodes(eval_env, RandomPolicy(4, seed=0), seeds)}
        update_rounds = 0
        t0 = time.perf_counter()
        record["evals"].append({"step": 0, "modes": _evaluate(arm, eval_env, seeds),
                                "diagnostics": arm.diagnostics(), "update_rounds": 0})
        timing["eval"] += time.perf_counter() - t0
        _checkpoint(arm, rollout, policy, replay, record, update_rounds, timing, device,
                    state_path, record_path)

    while rollout.budget_step < cfg.budget:
        if stop_after_step is not None and rollout.budget_step >= stop_after_step:
            return record
        t0 = time.perf_counter()
        for _ in range(cfg.act_steps):
            rollout.step(policy)
        if rollout.budget_step == cfg.budget:
            rollout.close_episode_truncated()
        timing["act"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(cfg.rounds_per_act):
            arm.update_round()
            update_rounds += 1
        timing["train"] += time.perf_counter() - t0
        if rollout.budget_step % cfg.eval_every == 0:
            t0 = time.perf_counter()
            record["evals"].append({"step": rollout.budget_step,
                                    "modes": _evaluate(arm, eval_env, seeds),
                                    "diagnostics": arm.diagnostics(),
                                    "update_rounds": update_rounds})
            timing["eval"] += time.perf_counter() - t0
            record["complete"] = rollout.budget_step == cfg.budget
            _checkpoint(arm, rollout, policy, replay, record, update_rounds, timing, device,
                        state_path, record_path)
    return record


def _checkpoint(arm, rollout, policy, replay, record, update_rounds, timing, device,
                state_path: Path, record_path: Path) -> None:
    record.update({"budget_steps": rollout.budget_step, "update_rounds": update_rounds,
                   "isolation_ok": arm.isolation_ok, "timing": dict(timing),
                   "seconds": sum(timing.values()), "git_sha": git_sha()})
    tmp = state_path.with_suffix(".tmp")
    torch.save({"arm": arm.state_dict(), "rollout": rollout.state(),
                "policy": policy.get_state() if rollout.in_progress else None,
                "replay_paths": [str(p) for p in replay.episode_paths()],
                "rng": _rng_state(device)}, tmp)
    tmp.replace(state_path)
    write_record(record_path, record)
```

Note on resume: `RandomPolicy` and arm policies have no state between episodes, and the prefill always ends with `close_episode_truncated()` before the step-0 checkpoint, so `policy` is only saved for an arm episode in progress. On resume, `arm.add_episode` re-adds every stored episode in order before `load_state_dict` restores the samplers' RNG states, so later samples are identical. The `RandomPolicy` used at evaluation is reseeded per episode by `run_episodes`.

- [ ] **Step 4: Run the driver tests**

Run: `PYTHONPATH=src:. /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/training/test_online.py -q`
Expected: all PASS. If `test_resume_is_bit_identical` fails, diff the two records' first differing field and find the RNG or state that was not saved; do not weaken the assertion.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/training/online.py tests/training/test_online.py
git commit -m "feat: train_online -- prefill, act/train, evaluation, and resumes that are bit-identical"
```

---

## Task 9: `scripts/train_online.py`, record verification, and verified feature deletion

**Files:**
- Modify: `src/mbfps/training/online.py` (add `verify_record`, `delete_replay_features`)
- Create: `scripts/train_online.py`
- Test: `tests/training/test_online.py`

**Interfaces:**
- Produces:
  - `verify_record(record: dict, cfg: OnlineConfig) -> list[str]` — empty when complete: `complete` true, `budget_steps == cfg.budget`, eval steps == `checkpoint_steps(cfg)`, every mode's and `random`'s seeds equal `cfg.eval_seeds` in order, `gated_mode` present in every eval
  - `delete_replay_features(out_dir: Path, cfg: OnlineConfig) -> int` — refuses (raises `ValueError` listing the problems) unless `verify_record(load_record(record.json))` is empty; deletes `replay/*.features*.npy` and `state.pt`, sets `record["replay_features_deleted"] = True`, rewrites the record, returns the number of files deleted
  - `python scripts/train_online.py --arm {dreamer,dqn} --seed N --out DIR [--device mps] [--budget/--prefill/--eval-every for smoke runs] [--keep-features]`

- [ ] **Step 1: Write the failing tests** (append to `tests/training/test_online.py`)

```python
from mbfps.training.online import delete_replay_features, verify_record


def test_verify_accepts_a_complete_run(tmp_path):
    rec = _run(tmp_path)
    assert verify_record(rec, TINY) == []


def test_delete_refuses_incomplete_record(tmp_path):
    _run(tmp_path, stop=20)
    with pytest.raises(ValueError, match="complete"):
        delete_replay_features(tmp_path, TINY)
    assert list((tmp_path / "replay").glob("*.features*.npy"))


def test_delete_removes_features_and_state_only(tmp_path):
    _run(tmp_path)
    n_npz = len(list((tmp_path / "replay").glob("ep_*.npz")))
    deleted = delete_replay_features(tmp_path, TINY)
    assert deleted == n_npz + 1                       # every feature file, plus state.pt
    assert not list((tmp_path / "replay").glob("*.features*.npy"))
    assert len(list((tmp_path / "replay").glob("ep_*.npz"))) == n_npz
    assert load_record(tmp_path / "record.json")["replay_features_deleted"] is True


def test_verify_flags_a_reordered_eval(tmp_path):
    rec = _run(tmp_path)
    rec["evals"][1]["modes"]["sample"] = rec["evals"][1]["modes"]["sample"][::-1]
    assert any("seeds" in p for p in verify_record(rec, TINY))
```

- [ ] **Step 2: Run to see them fail**

Run: `PYTHONPATH=src:. /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/training/test_online.py -q -k "verify or delete"`
Expected: FAIL — `cannot import name 'delete_replay_features'`.

- [ ] **Step 3: Implement** (append to `online.py`)

```python
def verify_record(record: dict, cfg: OnlineConfig) -> list[str]:
    """Why a run's record is not complete and well-formed (§3.4, §7.3); [] if it is."""
    problems = []
    seeds = list(cfg.eval_seeds)
    if record.get("complete") is not True:
        problems.append("record is not complete")
    if record.get("budget_steps") != cfg.budget:
        problems.append(f"budget_steps {record.get('budget_steps')} != {cfg.budget}")
    steps = [e["step"] for e in record.get("evals", [])]
    if steps != checkpoint_steps(cfg):
        problems.append(f"eval steps {steps} != {checkpoint_steps(cfg)}")
    if [e["seed"] for e in record.get("random", [])] != seeds:
        problems.append("random baseline seeds differ from the evaluation seeds")
    for e in record.get("evals", []):
        if record.get("gated_mode") not in e["modes"]:
            problems.append(f"step {e['step']}: gated mode missing")
        for mode, episodes in e["modes"].items():
            if [x["seed"] for x in episodes] != seeds:
                problems.append(f"step {e['step']} {mode}: seeds differ from the evaluation seeds")
    return problems


def delete_replay_features(out_dir: Path, cfg: OnlineConfig) -> int:
    """§9: delete a finished run's feature caches and resumable state, only
    after its record verifies. Episode files and the record are kept."""
    out_dir = Path(out_dir)
    record_path = out_dir / "record.json"
    record = load_record(record_path)
    problems = verify_record(record, cfg)
    if problems:
        raise ValueError("refusing to delete replay features: " + "; ".join(problems))
    victims = sorted((out_dir / "replay").glob("*.features*.npy"))
    state = out_dir / "state.pt"
    if state.exists():
        victims.append(state)
    for path in victims:
        path.unlink()
    record["replay_features_deleted"] = True
    write_record(record_path, record)
    return len(victims)
```

Create `scripts/train_online.py`:

```python
"""One M5a run on ViZDoom `basic` with live DINOv2 features (spec §3).

Resumes from OUT/state.pt if present. After a complete run, verifies the
record and deletes the replay's feature caches (spec §9) unless --keep-features."""

import argparse
import sys
from pathlib import Path

from mbfps.envs.registry import make_env
from mbfps.training.online import (OnlineConfig, delete_replay_features, train_online,
                                   verify_record)
from mbfps.utils.device import get_device


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("dreamer", "dqn"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--budget", type=int, default=100_000)
    p.add_argument("--prefill", type=int, default=5_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--keep-features", action="store_true")
    args = p.parse_args(argv)

    from mbfps.data.features import FeatureExtractor
    device = get_device(prefer=args.device)
    cfg = OnlineConfig(budget=args.budget, prefill=args.prefill, eval_every=args.eval_every)
    encode = FeatureExtractor(backbone="dinov2", device=device.type).encode
    env_factory = lambda: make_env("vizdoom", scenario="basic", frame_skip=4)
    record = train_online(args.arm, args.seed, args.out, env_factory, encode, device, cfg)
    problems = verify_record(record, cfg)
    if problems:
        print("INCOMPLETE: " + "; ".join(problems))
        return 1
    print(f"complete: {args.arm} seed {args.seed}, {record['update_rounds']} update rounds, "
          f"{record['seconds']:.0f} s")
    if not args.keep_features:
        n = delete_replay_features(args.out, cfg)
        print(f"deleted {n} feature/state files after verification")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the driver tests**

Run: `PYTHONPATH=src:. /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/training/test_online.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/training/online.py scripts/train_online.py tests/training/test_online.py
git commit -m "feat: scripts/train_online.py, and feature deletion only after the record verifies"
```

---

## Task 10: The reading — `efficiency.py` and `scripts/m5a_study.py`

**Files:**
- Create: `src/mbfps/eval/efficiency.py`
- Create: `scripts/m5a_study.py`
- Test: `tests/eval/test_efficiency.py`, `tests/eval/test_m5a_study_script.py`

**Interfaces:**
- Produces (`mbfps.eval.efficiency`):
  - `ARMS = ("dreamer", "dqn")`, `SEEDS = (0, 1, 2, 3, 4)`, `WIN_BAR = 21`, `THRESHOLD = 50.0`
  - `gated_returns(record, step) -> list[float]`; `auc(record, cfg) -> float` (mean over `checkpoint_steps(cfg)[1:]` of mean gated return)
  - `pairwise_wins(a: list[float], b: list[float]) -> float` (ties ½)
  - `verdict(dreamer_aucs, dqn_aucs) -> str` in `{"DREAMER_MORE_EFFICIENT", "DQN_MORE_EFFICIENT", "NOT_SEPARATED"}`
  - `run_problems(record, cfg) -> list[str]` = `verify_record(...)` + the step-0 control (`paired_gate(step-0 gated, random)["passed"]` must be False) + Dreamer `isolation_ok is True`
  - `steps_to_threshold(record, cfg, threshold=THRESHOLD) -> int | None`
- `python scripts/m5a_study.py --phase read [--runs runs/m5a]` → `study.txt`, exit 0 / 90 / 91; `--phase run [--runs runs/m5a] [--arms dreamer,dqn] [--seeds 0,1,2,3,4]` → runs `scripts/train_online.py` for each missing or incomplete run, seed-major (dreamer s0, dqn s0, dreamer s1, …), stopping at the first failure.

- [ ] **Step 1: Write the failing tests**

`tests/eval/test_efficiency.py`:

```python
import pytest

from mbfps.eval.efficiency import auc, pairwise_wins, run_problems, steps_to_threshold, verdict
from mbfps.training.online import OnlineConfig

CFG = OnlineConfig(budget=30, prefill=10, act_steps=10, eval_every=10, eval_seeds=(1, 2))


def _rec(arm="dreamer", curve=(-160.0, 10.0, 60.0, 70.0), step0=-170.0, random=-165.0,
         isolation=True):
    def eps(r):
        return [{"seed": s, "return": r} for s in (1, 2)]
    mode = "sample" if arm == "dreamer" else "greedy"
    evals = [{"step": 10 * i, "modes": {mode: eps(step0 if i == 0 else curve[i])}}
             for i in range(4)]
    return {"arm": arm, "seed": 0, "gated_mode": mode, "complete": True, "budget_steps": 30,
            "random": eps(random), "evals": evals,
            "isolation_ok": isolation if arm == "dreamer" else None}


def test_auc_is_the_mean_over_non_zero_checkpoints():
    assert auc(_rec(), CFG) == pytest.approx((10 + 60 + 70) / 3)


def test_pairwise_wins_count_ties_half():
    assert pairwise_wins([3, 2], [1, 2]) == 3.5


@pytest.mark.parametrize("d, expected_wins, expected", [
    ([9, 9, 9, 9, 9], 25, "DREAMER_MORE_EFFICIENT"),
    ([9, 9, 9, 9, 1], 21, "DREAMER_MORE_EFFICIENT"),
    ([9, 9, 9, 9, -1], 20, "NOT_SEPARATED"),
    ([1, 1, 1, 1, -1], 4, "DQN_MORE_EFFICIENT"),
    ([1, 1, 1, 1, 1], 5, "NOT_SEPARATED"),
])
def test_verdict_bar(d, expected_wins, expected):
    q = [0, 2, 4, 6, 8]
    assert pairwise_wins(d, q) == expected_wins
    assert verdict(d, q) == expected


def test_step0_control_must_fail():
    assert run_problems(_rec(), CFG) == []
    bad = _rec(step0=95.0)
    assert any("step-0 control" in p for p in run_problems(bad, CFG))


def test_isolation_required_for_dreamer_only():
    assert any("isolation" in p for p in run_problems(_rec(isolation=False), CFG))
    assert run_problems(_rec(arm="dqn"), CFG) == []


def test_steps_to_threshold():
    assert steps_to_threshold(_rec(), CFG) == 20
    assert steps_to_threshold(_rec(curve=(-160, 1, 2, 3)), CFG) is None
```

`tests/eval/test_m5a_study_script.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mbfps.eval.study import write_record

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "m5a_study.py"
SEEDS = list(range(1_000_000, 1_000_100))
STEPS = list(range(0, 100_001, 10_000))


def _write(runs, arm, seed, level, step0=-160.0, isolation=True, budget=100_000):
    mode = "sample" if arm == "dreamer" else "greedy"
    eps = lambda r: [{"seed": s, "return": r + (s % 5)} for s in SEEDS]
    rec = {"arm": arm, "seed": seed, "gated_mode": mode, "complete": True,
           "budget_steps": budget, "seeds": SEEDS, "random": eps(-155.0),
           "isolation_ok": isolation if arm == "dreamer" else None,
           "update_rounds": 19_000, "seconds": 3600.0,
           "evals": [{"step": s, "modes": {mode: eps(step0 if s == 0 else level)}} for s in STEPS]}
    write_record(runs / f"{arm}_seed{seed}" / "record.json", rec)


def _run(*args):
    p = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def test_missing_runs_exit_90(tmp_path):
    _write(tmp_path, "dreamer", 0, 60.0)
    assert _run("--phase", "read", "--runs", tmp_path)[0] == 90


def test_ten_valid_runs_print_the_reading(tmp_path):
    for s in range(5):
        _write(tmp_path, "dreamer", s, 60.0 + s)
        _write(tmp_path, "dqn", s, 10.0 + s)
    code, out = _run("--phase", "read", "--runs", tmp_path)
    assert code == 0, out
    assert "DREAMER_MORE_EFFICIENT" in (tmp_path / "study.txt").read_text()


@pytest.mark.parametrize("bad", [{"step0": 95.0}, {"isolation": False}, {"budget": 99_900}])
def test_invalid_run_exits_91_without_the_reading(tmp_path, bad):
    for s in range(5):
        _write(tmp_path, "dreamer", s, 60.0 + s, **(bad if s == 2 else {}))
        _write(tmp_path, "dqn", s, 10.0 + s)
    code, out = _run("--phase", "read", "--runs", tmp_path)
    assert code == 91, out
    text = (tmp_path / "study.txt").read_text()
    assert "INVALID_RUNS" in text and "dreamer_seed2" in text
    assert "MORE_EFFICIENT" not in text and "NOT_SEPARATED" not in text
```

- [ ] **Step 2: Run to see them fail**

Run: `PYTHONPATH=src:. /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/eval/test_efficiency.py tests/eval/test_m5a_study_script.py -q`
Expected: FAIL — `ModuleNotFoundError: mbfps.eval.efficiency`; the script does not exist.

- [ ] **Step 3: Implement `src/mbfps/eval/efficiency.py`**

```python
"""M5a's pre-registered reading (spec §7)."""

import numpy as np

from mbfps.eval.live import paired_gate
from mbfps.training.online import OnlineConfig, checkpoint_steps, verify_record

ARMS: tuple[str, ...] = ("dreamer", "dqn")
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
WIN_BAR: float = 21.0
THRESHOLD: float = 50.0
VERDICTS = ("DREAMER_MORE_EFFICIENT", "DQN_MORE_EFFICIENT", "NOT_SEPARATED")


def _eval_at(record: dict, step: int) -> dict:
    return next(e for e in record["evals"] if e["step"] == step)


def gated_returns(record: dict, step: int) -> list[float]:
    return [float(e["return"]) for e in _eval_at(record, step)["modes"][record["gated_mode"]]]


def auc(record: dict, cfg: OnlineConfig) -> float:
    """§7.1: mean over the non-zero checkpoints of the mean gated return."""
    return float(np.mean([np.mean(gated_returns(record, s)) for s in checkpoint_steps(cfg)[1:]]))


def pairwise_wins(a, b) -> float:
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None, :]
    return float((a > b).sum() + 0.5 * (a == b).sum())


def verdict(dreamer_aucs, dqn_aucs) -> str:
    if pairwise_wins(dreamer_aucs, dqn_aucs) >= WIN_BAR:
        return "DREAMER_MORE_EFFICIENT"
    if pairwise_wins(dqn_aucs, dreamer_aucs) >= WIN_BAR:
        return "DQN_MORE_EFFICIENT"
    return "NOT_SEPARATED"


def run_problems(record: dict, cfg: OnlineConfig) -> list[str]:
    """§7.3: everything that makes a run invalid; [] if it is valid."""
    problems = verify_record(record, cfg)
    if problems:
        return problems
    random_returns = [float(e["return"]) for e in record["random"]]
    if paired_gate(gated_returns(record, 0), random_returns)["passed"]:
        problems.append("step-0 control passes the paired gate against random")
    if record["arm"] == "dreamer" and record.get("isolation_ok") is not True:
        problems.append("isolation_ok is not true")
    return problems


def steps_to_threshold(record: dict, cfg: OnlineConfig, threshold: float = THRESHOLD):
    for s in checkpoint_steps(cfg)[1:]:
        if np.mean(gated_returns(record, s)) >= threshold:
            return s
    return None
```

`verify_record` checks every mode's seeds against `cfg.eval_seeds`; the script passes the production `OnlineConfig()` (seeds 1,000,000–1,000,099), and the unit tests pass their own small config.

- [ ] **Step 4: Implement `scripts/m5a_study.py`**

```python
"""M5a study: run the ten runs; read the pre-registered result (spec §7).

Exit codes (read):
  0   all ten runs present and valid; the reading is printed (any verdict)
  90  MISSING_RUNS   a run's record is absent or incomplete
  91  INVALID_RUNS   a run fails §7.3; the reading is not printed
Missing takes precedence over invalid."""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from mbfps.eval.efficiency import (ARMS, SEEDS, auc, gated_returns, pairwise_wins,
                                   run_problems, steps_to_threshold, verdict)
from mbfps.eval.study import load_record
from mbfps.training.online import OnlineConfig, checkpoint_steps

ROOT = Path(__file__).resolve().parent
EXIT_OK, EXIT_MISSING_RUNS, EXIT_INVALID_RUNS = 0, 90, 91


def _record_path(runs: Path, arm: str, seed: int) -> Path:
    return runs / f"{arm}_seed{seed}" / "record.json"


def run(args) -> int:
    for seed in args.seeds:
        for arm in args.arms:
            path = _record_path(args.runs, arm, seed)
            if path.exists() and load_record(path).get("replay_features_deleted"):
                print(f"skip (done): {arm} seed {seed}")
                continue
            cmd = [sys.executable, str(ROOT / "train_online.py"), "--arm", arm,
                   "--seed", str(seed), "--out", str(path.parent)]
            print("run:", " ".join(cmd), flush=True)
            code = subprocess.call(cmd)
            if code != 0:
                print(f"FAILED ({code}): {arm} seed {seed}")
                return code
    return EXIT_OK


def read(args) -> int:
    cfg = OnlineConfig()
    records, missing, invalid = {}, [], []
    for arm in ARMS:
        for seed in SEEDS:
            path = _record_path(args.runs, arm, seed)
            if not path.exists():
                missing.append(f"{arm}_seed{seed}")
                continue
            rec = load_record(path)
            if rec.get("complete") is not True:
                missing.append(f"{arm}_seed{seed} (incomplete)")
                continue
            problems = run_problems(rec, cfg)
            if problems:
                invalid.append(f"{arm}_seed{seed}: " + "; ".join(problems))
            records[(arm, seed)] = rec

    lines = ["M5a -- online Dreamer vs DQN on basic, 100,000 steps x 5 seeds"]
    for (arm, seed), rec in sorted(records.items()):
        curve = " ".join(f"{np.mean(gated_returns(rec, s)):+7.1f}" for s in checkpoint_steps(cfg))
        lines.append(f"  {arm:<7s} s{seed}  AUC {auc(rec, cfg):+8.2f}  "
                     f"steps-to-+50 {steps_to_threshold(rec, cfg) or '>100k'}  "
                     f"rounds {rec.get('update_rounds')}  seconds {rec.get('seconds', 0):.0f}")
        lines.append(f"      curve(0..100k): {curve}")
    if missing:
        lines.append("MISSING_RUNS: " + "; ".join(missing))
        code = EXIT_MISSING_RUNS
    elif invalid:
        lines.append("INVALID_RUNS: " + "; ".join(invalid))
        code = EXIT_INVALID_RUNS
    else:
        d = [auc(records[("dreamer", s)], cfg) for s in SEEDS]
        q = [auc(records[("dqn", s)], cfg) for s in SEEDS]
        lines += [f"Dreamer AUCs: {', '.join(f'{x:+.2f}' for x in d)}",
                  f"DQN AUCs:     {', '.join(f'{x:+.2f}' for x in q)}",
                  f"Dreamer wins {pairwise_wins(d, q):g} of 25 (bar 21; ties count 1/2)",
                  f"median AUC difference (Dreamer - DQN): {np.median(d) - np.median(q):+.2f}",
                  f"VERDICT: {verdict(d, q)}"]
        code = EXIT_OK
    args.runs.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (args.runs / "study.txt").write_text(text)
    print(text, end="")
    return code


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("run", "read"), required=True)
    p.add_argument("--runs", type=Path, default=Path("runs/m5a"))
    p.add_argument("--arms", type=lambda s: s.split(","), default=list(ARMS))
    p.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=list(SEEDS))
    args = p.parse_args(argv)
    return run(args) if args.phase == "run" else read(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the reading tests**

Run: `PYTHONPATH=src:. /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/eval/test_efficiency.py tests/eval/test_m5a_study_script.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/efficiency.py scripts/m5a_study.py tests/eval/test_efficiency.py tests/eval/test_m5a_study_script.py
git commit -m "feat: the M5a reading -- AUC, the 21-of-25 bar, and a read that refuses invalid runs"
```

---

## Task 11: Smoke runs on real ViZDoom, timing, and the suite

**Files:**
- Create: `runs/m5a_smoke/` (smoke outputs; may be deleted afterwards with `rm -rf runs/m5a_smoke` — it holds nothing reported)
- Modify: this plan (append `## Smoke results`)

- [ ] **Step 1: Full suite green at the commit being run**

```bash
PYTHONPATH=src:. /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest -q > "$TMPDIR/m5a_suite.log" 2>&1; echo "exit: $?"; tail -3 "$TMPDIR/m5a_suite.log"
git rev-parse --short HEAD
```

- [ ] **Step 2: A short real run per arm** (background; each ~10 minutes)

```bash
for ARM in dreamer dqn; do
  PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python scripts/train_online.py \
    --arm $ARM --seed 0 --out runs/m5a_smoke/${ARM}_seed0 --budget 3000 --prefill 1000 --eval-every 1000 \
    > runs/m5a_smoke/${ARM}.log 2>&1; echo "$ARM exit: $?"
done
```

Expected: both exit 0, print `complete`, and delete their feature caches. Read `record.json` for each: `timing` (act/train/eval seconds), `update_rounds` (400), the step-0 and step-3000 gated returns, and (Dreamer) `isolation_ok`.

- [ ] **Step 3: Project the full study's wall clock**

From the Dreamer smoke: `train_seconds / update_rounds` × 19,000 + `act_seconds / 3000` × 100,000 + `eval_seconds / 4` × 11; the same for DQN. If the projected total for ten runs exceeds **40 hours**, stop and report to the author before Task 12; the budget and train ratio are fixed by the spec and are not changed here.

- [ ] **Step 4: Write `## Smoke results` and commit**

Record the suite counts and SHA, both smoke runs' timings and projections, and anything that looked wrong. Commit only the plan file: `results: M5a smoke -- <what the smoke showed>`.

---

## Task 12: The ten runs, the reading, and the results

**Files:**
- Create: `runs/m5a/<arm>_seed<n>/` × 10, `runs/m5a/study.txt`, `runs/m5a/read.log`
- Modify: this plan (append `## M5a results`)

- [ ] **Step 1: Suite green at the commit being run** (as Task 11 Step 1, unless no code changed since; record which).

- [ ] **Step 2: Disk check and retention** — `df -h . | tail -1` must show ≥ 8 GB free (one run's features peak at ~5 GB before deletion). Spec §11.1: no run's replay features are retained — decided here, before any run; record it in the results.

- [ ] **Step 3: Run** (background; ~25 h; resumable — re-issue the same command after any interruption)

```bash
PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python scripts/m5a_study.py --phase run \
  > runs/m5a_run.log 2>&1; echo "run exit: $?"
```

After any non-zero exit: read the log's last `FAILED` line, diagnose (systematic-debugging), fix through review if it is code, and re-issue; completed runs are skipped and an interrupted run resumes from its last checkpoint.

- [ ] **Step 4: Read**

```bash
PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python scripts/m5a_study.py --phase read \
  > runs/m5a/read.log 2>&1; echo "read exit: $?"; cat runs/m5a/read.log
```

Then re-run the read into a scratch copy of the ten `record.json` files and confirm the output is byte-identical.

- [ ] **Step 5: Write `## M5a results` and commit**

Every number from a named record with the command that derives it, as in M4's results sections: the suite chain; each run's AUC, curve, steps-to-+50, final return, update rounds, wall clock (act/train/eval); every step-0 control; Dreamer's isolation and imagined-vs-real gap across checkpoints; the reported-only modes; the 25 pairwise comparisons; the verdict, exit code and what §7.2 licenses in the spec's words; the §10 limits; and that the feature caches were deleted after verification (§9). Commit only the plan file: `results: M5a -- <the finding, no more>`.

---

## Smoke results

**Suite.** At `dd1bb2a`: 3814 passed, 0 failed, 1464.63 s. The one later code commit, `ba0f977`, adds only `OnlineConfig` validation; at `ba0f977` every test that touches `OnlineConfig`, `train_online` or `m5a_study` was re-run: 43 passed, 37.4 s.

**Attempt 1 (budget 3000, prefill 1000, eval_every 1000), refused.** Both arms trained and evaluated, but the eval steps came out [0, 2000, 3000]. When `prefill == eval_every`, the step-0 evaluation sits at budget step 1000, so checkpoint 1000 is never evaluated. `verify_record` refused both runs and correctly left their feature caches in place. The fix is `ba0f977`: `OnlineConfig` now raises if `prefill >= eval_every` or `prefill >= budget`. Production uses prefill 5000 and eval_every 10000, so it was never affected.

**Attempt 2 (budget 3000, prefill 500, eval_every 1000, at `ba0f977`).** Both runs exited 0 and printed `complete`:
- eval steps [0, 1000, 2000, 3000];
- 500 update rounds;
- `nonfinite` {};
- feature caches and `state.pt` deleted after verification (70 files each);
- Dreamer `isolation_ok` True.

| | train s/round | act ms/step | eval s/checkpoint | projected 100k run |
|---|---|---|---|---|
| Dreamer | 339.4 / 500 = 0.679 | 54.8 / 3000 = 18.3 | 1093.8 / 4 = 273 | 12,896 + 1,826 + 3,008 s = **4.93 h** |
| DQN | 23.2 / 500 = 0.046 | 35.8 / 3000 = 11.9 | 703.1 / 4 = 176 | 882 + 1,192 + 1,934 s = **1.11 h** |

Ten runs: 5 × (4.93 + 1.11) = **30.2 h**. This is under the 40 h stop and above the spec's ~25 h estimate. Evaluation is about 40% of the total, and its cost is highest while policies are poor, because random-like episodes run to the timeout.

Gated returns at steps 0 / 1000 / 2000 / 3000 (random −164.2):
- Dreamer `sample`: −171.3 / −170.2 / −158.8 / −166.0.
- DQN `greedy`: −300.5 / −165.3 / −218.7 / −226.9.

Three thousand steps measure nothing about efficiency. They only show that the loop runs and that both step-0 controls sit at or below random. The reported-only modes are Dreamer `argmax`, which is −300 (timeout) until −236 at step 3000, and DQN `eps0_05`, which is −289 to −127. Neither looks wrong for untrained policies.

**Real-stack crash-resume.** This closes Task 8's deferred minor.
- A Dreamer seed-0 run with the attempt-2 configuration was stopped at budget step 1500 via `train_online(..., stop_after_step=1500)`. No checkpoint is written at the stop, so this simulates a crash after checkpoint 1000.
- It was then resumed by `scripts/train_online.py` to 3000.
- Its `evals`, `random`, `update_rounds`, `budget_steps`, `isolation_ok` and `complete` are **identical** to the uninterrupted attempt-2 record.
- So on ViZDoom and DINOv2 on MPS, a resume still reproduces the run exactly: orphan pruning, re-encoding the replayed episode, and restoring the recurrent policy state.

**Nothing else looked wrong.** `runs/m5a_smoke*` are scratch. Copies of the smoke records are in the SDD workspace's `runlogs/`.

---

## M5a results

Run on 2026-10-14 to 2026-10-16: five seeds (0-4) of two arms, online Dreamer (`frozen_ssl` world model, actor sampling) and Double DQN (greedy), scenario `basic`, 100,000 environment steps per run (5,000 random prefill steps included), one update round per 5 steps after the prefill, evaluation on seeds 1,000,000-1,000,099 at steps 0, 10,000, ..., 100,000. Branch `feat/m5a-online-loop`. Paths are worktree-relative; `runs` is a symlink into the main checkout. The ledger is `.superpowers/sdd/2026-10-14-mb-fps-m5a-online-sample-efficiency/progress.md` (lines beginning `Task 11:` and `Task 12:`); the run log, read log, read output and frame-reaper script and log are committed in `docs/superpowers/evidence/m5a/` (`m5a_run.log`, `read.log`, `study.txt`, `frame_reaper.py`, `frame_reaper.log`), copied unchanged from the originals (`runs/m5a_run.log`, `runs/m5a/read.log`, `runs/m5a/study.txt`, the SDD workspace's `runlogs/`) except that `m5a_run.log` has the HuggingFace Hub warning and weight-loading progress-bar lines removed (`grep -v`), so its sha256 (`898aec909cff3efc32b1bca4c332410985f19ba95a446c96d419f8fad51ed2f1`) differs from the original's (`7051989c0a39fe6ac7e83164b65309ad8c8b1ef2d8b8510fbdd52dcdc29a9407`); the ledger, `m5a_suite.log` and the smoke-record copies remain only in the untracked SDD workspace (`.superpowers/` is gitignored). The `record.json` files stay under `runs/m5a/`. Python below means `PYTHONPATH=src:. /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python`. The ten runs and the read were executed by the controller; this section was written afterwards from the ten `runs/m5a/<arm>_seed<n>/record.json` files and the logs, and every number below was re-derived from those records by the snippets shown (the numbers in `study.txt` agree with them wherever both print the same quantity).

All snippets in this section share this prelude, run from the worktree root (pure reads; nothing is written):

```python
import json, itertools, numpy as np
from mbfps.eval.efficiency import auc, gated_returns, pairwise_wins, run_problems, steps_to_threshold, verdict
from mbfps.eval.live import paired_gate
from mbfps.training.online import OnlineConfig, checkpoint_steps
cfg = OnlineConfig(); CK = checkpoint_steps(cfg)
R = {(a, s): json.load(open(f"runs/m5a/{a}_seed{s}/record.json")) for a in ("dreamer", "dqn") for s in range(5)}
ev = lambda r, st: next(e for e in r["evals"] if e["step"] == st)
mean = lambda r, st, mode=None: float(np.mean([e["return"] for e in ev(r, st)["modes"][mode or r["gated_mode"]]]))
```

### Verdict

**The §7.2 reading is `DQN_MORE_EFFICIENT`. Dreamer's AUC is higher in 1 of the 25 pairs; DQN's is higher in 24. Exit code 0.** All ten runs are present, complete and valid under §7.3, so the reading is printed.

```
Dreamer AUCs: +13.22, -17.25, +36.83, -13.23, +24.60
DQN AUCs:     +33.86, +46.52, +49.70, +51.23, +43.88
Dreamer wins 1 of 25 (bar 21; ties count 1/2)
median AUC difference (Dreamer - DQN): -33.31
VERDICT: DQN_MORE_EFFICIENT
```

- Record: `docs/superpowers/evidence/m5a/read.log` (byte-identical to `docs/superpowers/evidence/m5a/study.txt`: `cmp docs/superpowers/evidence/m5a/study.txt docs/superpowers/evidence/m5a/read.log` prints nothing; both are byte-identical to the originals `runs/m5a/study.txt` and `runs/m5a/read.log`). The exit code is the ledger line `Task 12: step 4: read exit 0`. `shasum -a 256 docs/superpowers/evidence/m5a/study.txt` prints `4cb14b0657bdd83b1d64170be9ccd6f16860ab500be59fbfdb6f87d11c62251e`.
- Re-derive, into a scratch directory, because the read writes `study.txt` under its `--runs` directory:

```bash
D=$(mktemp -d); for d in runs/m5a/*_seed*; do mkdir -p "$D/$(basename $d)"; cp $d/record.json "$D/$(basename $d)/"; done
Python scripts/m5a_study.py --phase read --runs "$D" > "$D/second.txt"; echo $?
cmp docs/superpowers/evidence/m5a/study.txt "$D/study.txt" && cmp docs/superpowers/evidence/m5a/read.log "$D/second.txt" && echo byte-identical
```

  At write time this printed exit `0` and `byte-identical`, and the `sha256` above. The read is a pure function of the ten records; it does not run any episode.
- The "median AUC difference" is the difference of the two arm medians (13.22 - 46.52 = -33.31 before rounding the inputs); the median of the 25 pairwise differences is also -33.31 (-33.306 in both cases).

**What §7.2 licenses.** The spec states the rule for this verdict and its p-value, in its table row: "DQN's AUC higher in at least 21 of 25 pairs" with one-sided exact p "12/252 = 0.048". Its Dreamer row carries the parenthetical "(Mann–Whitney U ≤ 4)" for the same bar, and the spec adds "The two-sided false-positive rate is 24/252 ≈ 0.095." The only sentence in §7.2 that speaks of what a verdict licenses is the one for the third outcome: "`NOT_SEPARATED` is a reportable result, as in M4 §6.2: it licenses no ranking in either direction and does not establish that the arms are equal." The spec gives no wider licence in words for a directional verdict than the rule itself; what follows is this section's reading of it, not the spec's wording. The bar was met with margin: 24 of 25 is U = 1, which has 2 of the 252 equally likely rankings at or below it (the snippet under "The 25 pairs" enumerates them; this exact p of 2/252 = 0.0079 is arithmetic on the observed U, not part of the pre-registered reading, which is the 21-of-25 bar). The question the spec puts (§1) is "at equal environment steps, does an online Dreamer agent learn `basic` faster than a model-free agent reading the same representation?" This section's gloss: on `basic`, with five seeds per arm, 100,000 environment steps, one update per 5 steps, and AUC defined as the mean over the ten checkpoints 10k...100k of the mean gated return over 100 evaluation episodes, the DQN arm is the more sample-efficient one by the pre-registered rule, and nothing wider is claimed.

**What it does not license** (spec §10, restated): it is one scenario (`basic`), one train ratio (1 update per 5 steps), one budget (100,000 steps) and one model-free baseline (Double DQN on the same frozen DINOv2 features), all fixed in advance; it says nothing about harder scenarios or the encoder comparison (M5b), nothing about raw-pixel model-free agents, and nothing about transfer to other ratios, budgets or algorithms. It measures environment-sample efficiency, not compute efficiency: Dreamer spent about 3x DQN's wall-clock per run (see "Compute"), and the verdict is silent on which is cheaper for a given return. It does not say why Dreamer was slower to learn; the diagnostics below describe what the records show during that phase and do not establish a cause. It also does not say the arms differ at the end of training: both reach about +77 to +80 (see "The learning curves").

### Commits, suite, and what ran where

| item | value | source |
|---|---|---|
| suite | 3814 passed, 0 failed, 1464.63 s (0:24:24), at `dd1bb2a` | `tail -1 .superpowers/sdd/2026-10-14-mb-fps-m5a-online-sample-efficiency/runlogs/m5a_suite.log` prints `3814 passed in 1464.63s (0:24:24)`; the commit is from the ledger (`Task 11: step 1`) |
| later code commit `ba0f977` | adds only `OnlineConfig` validation (`prefill >= eval_every` or `>= budget` refused) | `git diff --stat dd1bb2a ba0f977 -- src scripts tests` lists `src/mbfps/training/online.py` (5 lines), `tests/eval/test_efficiency.py` (2) and `tests/training/test_online.py` (14) |
| covering tests at `ba0f977` | every test touching `OnlineConfig`, `train_online` or `m5a_study`: 43 passed, 37.4 s | ledger `Task 11: Ruling (Task 12 step 1)` (ledger only; no full-suite re-run at `ba0f977`, a ruling recorded there) |
| run commit `c38a33a` | docs-only on top of `ba0f977` | `git diff --stat ba0f977 c38a33a -- src scripts tests` prints nothing (exit 0); `git diff --stat ba0f977 c38a33a` lists only this plan file (36 insertions) |
| commit the ten records carry | `c38a33a` | snippet A below |

The chain is: the full suite was green at `dd1bb2a`; `ba0f977` changed only the `OnlineConfig` constructor checks and its tests, which were re-run; `c38a33a` changed no source, script or test. The suite was not re-run on `c38a33a` itself.

**After the records.** The final review's fix commit changed `src/mbfps/training/online.py` (and `tests/training/test_online.py`) after the ten records were made: a resume's identity now includes the commit and the device type, a record's `git_sha` is set once at its creation instead of at every checkpoint, and the CUDA RNG state is saved. None of this touches training, evaluation or reading arithmetic, and the ten records are unchanged (no run resumed and HEAD did not move). `tests/training` (148 passed), `tests/eval/test_efficiency.py` and `tests/eval/test_m5a_study_script.py` were re-run on that commit.

**Snippet A (validity and identity).**

```python
print({r["git_sha"][:7] for r in R.values()}, {r["complete"] for r in R.values()}, {r["budget_steps"] for r in R.values()},
      {r["update_rounds"] for r in R.values()}, [r["nonfinite"] for r in R.values()])
print(sum(bool(run_problems(r, cfg, *k)) for k, r in R.items()), [R[("dreamer", s)]["isolation_ok"] for s in range(5)],
      {r["replay_features_deleted"] for r in R.values()})
```

At write time this printed `{'c38a33a'} {True} {100000} {19000} [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]` and `0 [True, True, True, True, True] {True}`. In words: all ten records carry `c38a33a`, are `complete`, spent exactly 100,000 budget steps and 19,000 update rounds (95,000 post-prefill steps / 5), and have an empty `nonfinite`; `run_problems` (the same validity function the read uses: §7.3 plus `verify_record`) returns `[]` for all ten; Dreamer's `isolation_ok` is True in all five seeds (no actor-critic step ever put a gradient on a world-model parameter); `replay_features_deleted` is True in all ten.

**What ran.** The study was launched on 2026-10-14 at about 20:10 (-0500; `c38a33a` is dated 20:10:46) under `caffeinate -i` with `Python scripts/m5a_study.py --phase run > runs/m5a_run.log 2>&1`, and finished on 2026-10-16 at about 21:20 (the last `complete:` line is in `docs/superpowers/evidence/m5a/m5a_run.log`, whose original `runs/m5a_run.log` was last written 21:23). It ran as one process, in the order dreamer s0, dqn s0, dreamer s1, dqn s1, ... dqn s4; `grep -c 'skip (done)\|FAILED\|Traceback\|resum' docs/superpowers/evidence/m5a/m5a_run.log` prints `0` and there are ten `complete:` lines, so no run failed, was skipped or needed a resume. The run exit was 0 (ledger `Task 12: step 3`). The wall-clock span is about 49 hours; the sum of the ten runs' own `seconds` is 33.83 hours (snippet H). The difference is the laptop sleeping (lid or battery): each run's `seconds` and `timing` are `time.perf_counter` sums and exclude periods the machine was asleep. This section says nothing further about the cause of any timing figure.

### The ten runs

Records: `runs/m5a/<arm>_seed<n>/record.json` (Dreamer gated mode `sample`, DQN `greedy`). AUC is `mbfps.eval.efficiency.auc`; steps-to-+50 is `steps_to_threshold` (first checkpoint whose mean gated return is at least +50; no persistence is required); the return at 100k is the mean over the 100 evaluation episodes; seconds are the record's `seconds` and `timing` (act, train, eval).

**Snippet B (per run).**

```python
for (a, s), r in R.items():
    print(a, s, f"{auc(r, cfg):+.2f}", steps_to_threshold(r, cfg) or ">100k", f"{mean(r, 100000):+.2f}", r["update_rounds"],
          round(r["seconds"]), *(round(r["timing"][k]) for k in ("act", "train", "eval")),
          " ".join(f"{mean(r, c):+.1f}" for c in CK))
```

| arm | seed | AUC | steps to +50 | return at 100k | update rounds | seconds | act | train | eval |
|---|---|---|---|---|---|---|---|---|---|
| dreamer | 0 | +13.22 | 40000 | +77.37 | 19,000 | 15,793 | 2,638 | 12,001 | 1,154 |
| dreamer | 1 | -17.25 | 50000 | +77.39 | 19,000 | 15,953 | 2,468 | 11,977 | 1,507 |
| dreamer | 2 | +36.83 | 40000 | +77.22 | 19,000 | 17,220 | 2,835 | 13,412 | 973 |
| dreamer | 3 | -13.23 | 40000 | +78.08 | 19,000 | 21,927 | 3,265 | 17,379 | 1,283 |
| dreamer | 4 | +24.60 | 40000 | +76.54 | 19,000 | 20,362 | 2,907 | 16,416 | 1,038 |
| dqn | 0 | +33.86 | 40000 | +79.44 | 19,000 | 5,136 | 2,164 | 2,091 | 882 |
| dqn | 1 | +46.52 | 30000 | +79.76 | 19,000 | 5,229 | 2,146 | 2,445 | 638 |
| dqn | 2 | +49.70 | 20000 | +79.04 | 19,000 | 6,228 | 3,012 | 2,565 | 651 |
| dqn | 3 | +51.23 | 20000 | +79.74 | 19,000 | 6,344 | 3,056 | 2,506 | 782 |
| dqn | 4 | +43.88 | 40000 | +79.46 | 19,000 | 7,600 | 3,056 | 3,766 | 778 |

No run failed to reach +50 within the budget (no `>100k`). Steps-to-+50: Dreamer 40k in four seeds and 50k in seed 1; DQN 20k (seeds 2, 3), 30k (seed 1) and 40k (seeds 0, 4). The statistic is a first crossing on a 10k grid: DQN seed 2 is at +64.2 at 20k and +43.7 at 30k, and Dreamer seed 2 is at +49.6 at 30k, just under the bar, and +74.9 at 40k; Dreamer seed 4 is at +50.02 at 40k, just over the bar.

Per-run curves (mean gated return over the 100 evaluation episodes at each checkpoint; the same snippet, last column group):

| arm | seed | 0k | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 100k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dreamer | 0 | -171.3 | -93.9 | -223.0 | -90.1 | +72.1 | +77.2 | +77.7 | +78.1 | +78.4 | +78.2 | +77.4 |
| dreamer | 1 | -167.0 | -96.4 | -160.2 | -224.8 | -149.0 | +75.1 | +76.0 | +76.2 | +75.5 | +77.7 | +77.4 |
| dreamer | 2 | -164.8 | -83.1 | -135.8 | +49.6 | +74.9 | +77.4 | +77.4 | +76.9 | +76.7 | +77.1 | +77.2 |
| dreamer | 3 | -176.5 | -270.4 | -270.6 | -119.0 | +68.1 | +74.6 | +75.7 | +74.7 | +78.4 | +78.0 | +78.1 |
| dreamer | 4 | -165.8 | -92.5 | -120.5 | -53.6 | +50.0 | +75.8 | +77.2 | +77.4 | +78.1 | +77.6 | +76.5 |
| dqn | 0 | -300.5 | -167.5 | -67.2 | +32.9 | +74.8 | +77.2 | +75.3 | +75.7 | +78.6 | +79.4 | +79.4 |
| dqn | 1 | -273.4 | -137.9 | +2.2 | +69.2 | +70.0 | +72.7 | +74.7 | +77.4 | +78.6 | +78.6 | +79.8 |
| dqn | 2 | -206.4 | -150.1 | +64.2 | +43.7 | +70.7 | +74.8 | +76.5 | +78.7 | +80.0 | +79.6 | +79.0 |
| dqn | 3 | -300.0 | -149.6 | +63.5 | +56.6 | +73.7 | +74.4 | +77.9 | +77.6 | +79.1 | +79.3 | +79.7 |
| dqn | 4 | -313.0 | -107.4 | -3.5 | +18.3 | +69.4 | +72.3 | +76.8 | +75.7 | +79.1 | +78.6 | +79.5 |

These are the same values as the `curve(0..100k)` lines of `docs/superpowers/evidence/m5a/study.txt`.

### The learning curves (§7.4)

Mean and range across the five seeds of the gated-mode return at every checkpoint (snippet D):

```python
for c in CK:
    print(c, *(f"{np.mean(m):+.1f} [{min(m):+.1f}, {max(m):+.1f}]" for m in ([mean(R[(a, s)], c) for s in range(5)] for a in ("dreamer", "dqn"))))
print("early 10-40k", *(round(np.mean([[mean(R[(a, s)], c) for c in CK[1:5]] for s in range(5)]), 1) for a in ("dreamer", "dqn")),
      "late 50-100k", *(round(np.mean([[mean(R[(a, s)], c) for c in CK[5:]] for s in range(5)]), 1) for a in ("dreamer", "dqn")))
```

| step | Dreamer mean | Dreamer range | DQN mean | DQN range |
|---|---|---|---|---|
| 0k | -169.1 | -176.5 to -164.8 | -278.7 | -313.0 to -206.4 |
| 10k | -127.2 | -270.4 to -83.1 | -142.5 | -167.5 to -107.4 |
| 20k | -182.0 | -270.6 to -120.5 | +11.9 | -67.2 to +64.2 |
| 30k | -87.6 | -224.8 to +49.6 | +44.1 | +18.3 to +69.2 |
| 40k | +23.2 | -149.0 to +74.9 | +71.7 | +69.4 to +74.8 |
| 50k | +76.0 | +74.6 to +77.4 | +74.3 | +72.3 to +77.2 |
| 60k | +76.8 | +75.7 to +77.7 | +76.2 | +74.7 to +77.9 |
| 70k | +76.6 | +74.7 to +78.1 | +77.0 | +75.7 to +78.7 |
| 80k | +77.4 | +75.5 to +78.4 | +79.1 | +78.6 to +80.0 |
| 90k | +77.7 | +77.1 to +78.2 | +79.1 | +78.6 to +79.6 |
| 100k | +77.3 | +76.5 to +78.1 | +79.5 | +79.0 to +79.8 |

Step 0 is not part of the AUC. What the table shows, descriptively:

- **Where the AUC difference comes from.** Over checkpoints 10k-40k the arms' mean gated returns average -93.4 (Dreamer) and -3.7 (DQN); over 50k-100k they average +77.0 and +77.5. The AUC gap (Dreamer mean AUC +8.83, DQN +45.04) is made in the first 40k steps. At 50k and 60k Dreamer's seed-mean is slightly above DQN's (+76.0 against +74.3; +76.8 against +76.2), and at 70k they are +76.6 against +77.0.
- **Both arms plateau at about +77 to +80.** From 50k, every run of both arms holds between +72.3 and +80.0. At 80k, 90k and 100k the two arms' per-seed ranges do not overlap, DQN's being higher (at 100k, Dreamer +76.5 to +78.1, DQN +79.0 to +79.8, a mean difference of 2.2). This is an observation about the last three checkpoints, not part of the pre-registered reading. At 100k both arms' evaluation episodes are all kills (kill rate 1.0) with mean length 6.3 (Dreamer) and 5.9 (DQN) steps.
- **DQN's curve is closer to monotone.** After 20k its largest single fall is DQN seed 2 from +64.2 to +43.7. Dreamer is not monotone before 50k: in all five seeds the mean return falls between two consecutive checkpoints, by 28 to 129 points in the largest fall of each of seeds 0, 1, 2, 3 and 4 (seed 0 from -93.9 to -223.0, a fall of 129.1; seed 1 from -96.4 to -160.2 and then to -224.8, falls of 63.8 and 64.6; seed 2 from -83.1 to -135.8, 52.7; seed 3 from -176.5 at step 0 to -270.4 at 10k, 93.9, and then by 0.2 more to -270.6 at 20k; seed 4 from -92.5 to -120.5, 28.0). All five are above +74 at 50k (seed 1, the last to recover, is at -149.0 at 40k and +75.1 at 50k).

### Step-0 controls (§6, §7.3)

Both arms are evaluated at budget step 0, after the prefill and before any update; each arm's step-0 policy through M4's seed-paired gate against the random policy on the same 100 seeds must **fail** it (the gate passes only if the paired-difference interval's `ci_low` is above 0). Each record carries its own random-policy episodes on the same 100 seeds (the spec has the baseline measured once); they are identical in all ten records.

```python
rnd = [e["return"] for e in R[("dreamer", 0)]["random"]]
print("random mean", round(np.mean(rnd), 2), "len", round(np.mean([e["length"] for e in R[("dreamer", 0)]["random"]]), 1),
      "kill", np.mean([e["killed"] for e in R[("dreamer", 0)]["random"]]), "identical in all ten:", len({tuple(e["return"] for e in r["random"]) for r in R.values()}) == 1)
for k, r in R.items():
    g = paired_gate(gated_returns(r, 0), [e["return"] for e in r["random"]])
    print(k, round(mean(r, 0), 2), {a: b for a, b in g.items()})
```

Random baseline: mean return -164.21, mean length 47.1, kill rate 0.57; identical in all ten records (`True`).

| run | step-0 mean return | point | ci_low | ci_high | passed |
|---|---|---|---|---|---|
| dreamer s0 | -171.34 | -7.13 | -49.48 | +35.56 | False |
| dreamer s1 | -167.02 | -2.81 | -45.12 | +41.80 | False |
| dreamer s2 | -164.81 | -0.60 | -44.60 | +45.07 | False |
| dreamer s3 | -176.47 | -12.26 | -54.04 | +32.39 | False |
| dreamer s4 | -165.77 | -1.56 | -45.43 | +43.52 | False |
| dqn s0 | -300.50 | -136.29 | -172.91 | -98.84 | False |
| dqn s1 | -273.44 | -109.23 | -148.41 | -70.52 | False |
| dqn s2 | -206.45 | -42.24 | -95.87 | +7.78 | False |
| dqn s3 | -300.00 | -135.79 | -172.40 | -98.59 | False |
| dqn s4 | -313.00 | -148.79 | -186.06 | -113.83 | False |

All ten fail the gate, as §6 requires (`passed` is False in every row; `run_problems` raises the step-0 problem otherwise and returns `[]` for all ten). They fail in different ways: Dreamer's untrained actor is indistinguishable from random (every interval straddles 0), while DQN's untrained greedy policy is worse than random (four of the five intervals lie entirely below 0); its step-0 episodes time out in 70% to 100% of cases (mean length 55.9 to 75.0, against random's 47.1; `truncated` and `length` of `evals[0]["modes"]["greedy"]`).

### The 25 pairs (§7.2)

Dreamer seed rows against DQN seed columns; each cell is the AUC difference (Dreamer - DQN), and the winner is the arm with the higher AUC. There are no ties.

```python
d = [auc(R[("dreamer", s)], cfg) for s in range(5)]; q = [auc(R[("dqn", s)], cfg) for s in range(5)]
for i, x in enumerate(d): print(i, " ".join(f"{x - y:+.2f}" for y in q))
print(pairwise_wins(d, q), pairwise_wins(q, d), verdict(d, q), np.median(d) - np.median(q), np.median([x - y for x in d for y in q]))
U = [sum(a > b for a, b in itertools.product(c, set(range(10)) - set(c))) for c in itertools.combinations(range(10), 5)]
print("exact one-sided p, DQN-wins>=24 (U<=1):", sum(u <= 1 for u in U), "/", len(U), "; U<=4:", sum(u <= 4 for u in U), "/", len(U))
```

| Dreamer \ DQN | s0 (+33.86) | s1 (+46.52) | s2 (+49.70) | s3 (+51.23) | s4 (+43.88) |
|---|---|---|---|---|---|
| s0 (+13.22) | DQN (-20.64) | DQN (-33.31) | DQN (-36.48) | DQN (-38.01) | DQN (-30.66) |
| s1 (-17.25) | DQN (-51.11) | DQN (-63.78) | DQN (-66.95) | DQN (-68.48) | DQN (-61.13) |
| s2 (+36.83) | **Dreamer (+2.97)** | DQN (-9.69) | DQN (-12.87) | DQN (-14.39) | DQN (-7.05) |
| s3 (-13.23) | DQN (-47.09) | DQN (-59.75) | DQN (-62.93) | DQN (-64.46) | DQN (-57.11) |
| s4 (+24.60) | DQN (-9.26) | DQN (-21.93) | DQN (-25.10) | DQN (-26.63) | DQN (-19.28) |

The snippet prints `1.0 24.0 DQN_MORE_EFFICIENT -33.306 -33.306` and `exact one-sided p, DQN-wins>=24 (U<=1): 2 / 252 ; U<=4: 12 / 252` (the 12/252 is the spec's table entry for the 21-of-25 bar, reproduced). The single Dreamer win is seed 2 over DQN seed 0, by 2.97 AUC points. Dreamer's best AUC (+36.83, seed 2) is below DQN's median (+49.70) and below four of the five DQN AUCs; Dreamer seeds 1 and 3 have negative AUCs. The two Dreamer seeds with negative AUC are two of the three that fall below random (see "Dreamer diagnostics").

### The reported-only modes (§6, §7.4)

Dreamer `argmax` and DQN `eps0_05` (epsilon 0.05) are recorded at every checkpoint on the same 100 seeds; they are not part of the reading. Seed-means (snippet F):

```python
for c in CK:
    print(c, *(f"{np.mean([mean(R[(a, s)], c, m) for s in range(5)]):+.1f}" for a, m in (("dreamer", "sample"), ("dreamer", "argmax"), ("dqn", "greedy"), ("dqn", "eps0_05"))))
A = {(a, m): [np.mean([mean(R[(a, s)], c, m) for c in CK[1:]]) for s in range(5)] for a, m in (("dreamer", "sample"), ("dreamer", "argmax"), ("dqn", "greedy"), ("dqn", "eps0_05"))}
print({k: round(np.mean(v), 2) for k, v in A.items()}, pairwise_wins(A[("dreamer", "argmax")], A[("dqn", "eps0_05")]))
```

| step | Dreamer `sample` (gated) | Dreamer `argmax` | DQN `greedy` (gated) | DQN `eps0_05` |
|---|---|---|---|---|
| 0k | -169.1 | -268.5 | -278.7 | -261.3 |
| 10k | -127.2 | -149.8 | -142.5 | -111.2 |
| 20k | -182.0 | -190.9 | +11.9 | +19.2 |
| 30k | -87.6 | -90.4 | +44.1 | +44.4 |
| 40k | +23.2 | +18.0 | +71.7 | +70.5 |
| 50k | +76.0 | +76.4 | +74.3 | +72.3 |
| 60k | +76.8 | +77.2 | +76.2 | +74.5 |
| 70k | +76.6 | +77.8 | +77.0 | +74.3 |
| 80k | +77.4 | +77.2 | +79.1 | +76.8 |
| 90k | +77.7 | +77.6 | +79.1 | +77.0 |
| 100k | +77.3 | +77.5 | +79.5 | +77.3 |

Mean AUC by mode: Dreamer `sample` +8.83, Dreamer `argmax` +5.06; DQN `greedy` +45.04, DQN `eps0_05` +47.51. Applying the same 25-pair comparison to the reported-only modes (descriptive only; it is not a pre-registered reading) gives 0.0 Dreamer wins of 25 for Dreamer `argmax` against DQN `eps0_05`: Dreamer's best `argmax` AUC (+34.43) is below DQN's lowest `eps0_05` AUC (+38.77). So the ordering does not depend on which evaluation mode is used for either arm. After 50k, Dreamer's `argmax` and `sample` seed-means differ by at most 1.2, and DQN's `eps0_05` seed-mean is 1.7 to 2.7 below its greedy seed-mean. The training epsilon recorded in the DQN diagnostics is 0.81 at 10k and 0.05 at 50k and 100k in all five seeds, so from 50k the `eps0_05` evaluation mode matches the behaviour policy's floor.

### Dreamer diagnostics (§4.4, reported not gated)

Source: `evals[i]["diagnostics"]` (training diagnostics: means over the update rounds since the previous checkpoint) and `evals[i]["modes"]["sample"][j]["imagined_return"]` (evaluation: the imagined lambda-return at each evaluation episode's first filtered state, read with `critic_slow`, `gamma`, `lam` and the unbounded reward readout; `mbfps.eval.live.AgentPolicy._imagined_return`). The imagined-vs-real gap at episode start (§4.4) is the evaluation imagined return minus the episode's real return, averaged over the 100 evaluation episodes. It is the spec's diagnostic, not a calibrated error: the imagined value is a discounted lambda-return over a 15-step horizon, bootstrapped by the slow critic, and the real return is an undiscounted episode sum.

**Snippet G.**

```python
D = [R[("dreamer", s)] for s in range(5)]
imag = lambda r, c: float(np.mean([e["imagined_return"] for e in ev(r, c)["modes"]["sample"]]))
rows = [(s, c, mean(D[s], c, "sample"), imag(D[s], c)) for s in range(5) for c in CK[1:]]
print("real < random:", [(s, c, round(re, 1)) for s, c, re, im in rows if re < np.mean(rnd)])
print("gap > +100:", [(s, c, round(re), round(im), round(im - re)) for s, c, re, im in rows if im - re > 100])
print("gap < -100:", [(s, c, round(re), round(im), round(im - re)) for s, c, re, im in rows if im - re < -100])
late = [im - re for s, c, re, im in rows if c >= 60000]
print("60k+ gap: mean", round(np.mean(late), 2), "mean abs", round(np.mean(np.abs(late)), 2), "range", round(min(late), 1), round(max(late), 1), len(late))
for s, c in ((0, 20000), (1, 30000), (3, 10000), (3, 20000), (1, 20000), (4, 20000)):
    m = ev(D[s], c)["modes"]["sample"]
    print(s, c, "len", round(np.mean([x["length"] for x in m]), 1), "kill", np.mean([x["killed"] for x in m]), "truncated", np.mean([x["truncated"] for x in m]))
print("random truncated", np.mean([e["truncated"] for e in R[("dreamer", 0)]["random"]]))
dg = lambda s, c, k: ev(D[s], c)["diagnostics"][k]
for k in ("imagined_return", "clipped_share", "n_dropped", "n_starts", "entropy", "scale", "critic_loss", "wm_reward"):
    print(k, *(f"{np.mean([dg(s, c, k) for s in range(5)]):.4g}" for c in CK[1:]))
```

Imagined return at episode start, real return, and the gap (imagined - real), per seed and checkpoint:

| seed | | 0k | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 100k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | real | -171.3 | -93.9 | -223.0 | -90.1 | +72.1 | +77.2 | +77.7 | +78.1 | +78.4 | +78.2 | +77.4 |
| 0 | imagined | +0.1 | -116.7 | -77.7 | -154.3 | +43.0 | +76.5 | +81.8 | +78.3 | +78.1 | +79.7 | +76.9 |
| 0 | gap | +171.4 | -22.8 | +145.3 | -64.2 | -29.1 | -0.7 | +4.1 | +0.2 | -0.2 | +1.6 | -0.5 |
| 1 | real | -167.0 | -96.4 | -160.2 | -224.8 | -149.0 | +75.1 | +76.0 | +76.2 | +75.5 | +77.7 | +77.4 |
| 1 | imagined | +0.0 | -132.3 | -367.7 | -75.7 | -163.6 | +73.6 | +72.5 | +79.1 | +74.7 | +79.0 | +85.4 |
| 1 | gap | +167.0 | -35.9 | -207.5 | +149.2 | -14.6 | -1.5 | -3.5 | +3.0 | -0.8 | +1.3 | +8.0 |
| 2 | real | -164.8 | -83.1 | -135.8 | +49.6 | +74.9 | +77.4 | +77.4 | +76.9 | +76.7 | +77.1 | +77.2 |
| 2 | imagined | -0.1 | -112.9 | -92.1 | +3.7 | +64.8 | +78.7 | +79.5 | +75.4 | +81.3 | +74.9 | +82.0 |
| 2 | gap | +164.7 | -29.8 | +43.7 | -46.0 | -10.1 | +1.3 | +2.1 | -1.5 | +4.6 | -2.2 | +4.8 |
| 3 | real | -176.5 | -270.4 | -270.6 | -119.0 | +68.1 | +74.6 | +75.7 | +74.7 | +78.4 | +78.0 | +78.1 |
| 3 | imagined | -0.0 | -163.8 | -120.1 | -117.8 | +36.6 | +71.2 | +79.5 | +71.1 | +75.9 | +76.5 | +76.7 |
| 3 | gap | +176.4 | +106.5 | +150.5 | +1.2 | -31.5 | -3.4 | +3.8 | -3.6 | -2.5 | -1.6 | -1.4 |
| 4 | real | -165.8 | -92.5 | -120.5 | -53.6 | +50.0 | +75.8 | +77.2 | +77.4 | +78.1 | +77.6 | +76.5 |
| 4 | imagined | -0.0 | -144.4 | -392.2 | -62.4 | +23.7 | +64.7 | +79.1 | +77.5 | +82.5 | +73.9 | +83.5 |
| 4 | gap | +165.8 | -51.8 | -271.7 | -8.7 | -26.3 | -11.1 | +1.9 | +0.2 | +4.4 | -3.8 | +7.0 |

Training diagnostics (seed-mean over the five Dreamer seeds of the checkpoint's window mean):

| step | imagined_return | clipped_share | dropped share of start states | entropy | return scale | critic_loss | wm_reward |
|---|---|---|---|---|---|---|---|
| 10k | -93.78 | 0.0026 | 1.3% | 1.102 | 91.4 | 292 | 0.403 |
| 20k | -398.8 | 0.0051 | 1.3% | 0.232 | 547.6 | 1,714 | 0.0737 |
| 30k | -403.6 | 0.0299 | 1.3% | 0.323 | 790.4 | 3,871 | 0.0206 |
| 40k | -125.3 | 0.0388 | 1.8% | 0.234 | 380.5 | 1,091 | 0.0136 |
| 50k | +37.74 | 0.0572 | 3.8% | 0.105 | 126.5 | 192 | 0.0125 |
| 60k | +66.06 | 0.0736 | 6.0% | 0.081 | 91.8 | 21.7 | 0.00977 |
| 70k | +70.46 | 0.0737 | 7.6% | 0.067 | 89.7 | 13.9 | 0.00886 |
| 80k | +73.31 | 0.0726 | 8.8% | 0.064 | 86.5 | 11.0 | 0.00876 |
| 90k | +75.68 | 0.0739 | 9.7% | 0.065 | 86.5 | 9.10 | 0.00823 |
| 100k | +77.00 | 0.0708 | 10.3% | 0.075 | 77.4 | 8.08 | 0.00662 |

(The table's columns follow `CK[1:]`; the dropped share is `n_dropped / (n_dropped + n_starts)` per seed, then averaged; `clipped_share` is the share of imagined reward readouts outside the bound, per §4.3.) At step 0 (before any update) the diagnostics hold only `reward_bounds`, `replay_transitions` (5,000) and `replay_episodes` (101 to 118 across seeds, per `evals[0]`).

What the diagnostics show:

- **Isolation held.** `isolation_ok` is True in all five Dreamer runs (snippet A).
- **Reward bounds.** `reward_bounds` is [-9.0, 99.0] at every checkpoint, including step 0, in all five seeds (`{tuple(ev(r, c)["diagnostics"]["reward_bounds"]) for r in D for c in CK}` is `{(-9.0, 99.0)}`), so both a missed shot and a kill were in the prefill replay and the bound never widened afterwards.
- **Clipping.** `clipped_share` rises from 0.26% (10k) to about 7% from 60k; the largest value in any seed at any checkpoint is 0.0867. The bound is active throughout and never dominant. The dropped share of start states (terminal starts, §4.3) is 1.3% through 30k (about the 1.4% the spec cites for M4) and rises to 10.3% at 100k, as episodes shorten to about 6 steps and every one ends in a kill.
- **Imagined against real, after the transient.** From 60k to 100k (25 seed-checkpoints) the gap has mean +1.0, mean absolute value 2.7, range -3.8 to +8.0; at 50k it is -11.1 to +1.3. Imagination and reality agree to within a few points once the policy is at its plateau. There is no sign of the exploitation signature (§4.4: imagined return far above the real one) in the plateau phase.
- **Imagined against real, during the dips.** This is the observation worth recording. Four seed-checkpoints have a real mean return below random (-164.21): seed 0 at 20k (-223.0), seed 1 at 30k (-224.8), seed 3 at 10k (-270.4) and seed 3 at 20k (-270.6) (the `real < random` line of snippet G). Three of the five seeds (0, 1, 3) have at least one; seeds 2 and 4 never fall below random after step 0 (their minima are -135.8 and -120.5). The same four seed-checkpoints are exactly the four whose gap exceeds +100 (imagined return at episode start minus real: +145, +149, +107, +150, with imagined returns of -78, -76, -164 and -120): the world model's start-of-episode estimate was 107 to 150 points above what the policy then earned. At those four checkpoints the evaluation episodes also show lower kill rates (0.21, 0.36, 0.08 and 0.31, against 0.57 for random) and longer episodes with more time-outs (truncated share 0.79, 0.64, 0.92 and 0.69, against 0.43 for random).
- **But the gap is not always positive when the policy is poor.** Two other checkpoints have a gap below -100, in the other direction: seed 1 at 20k (imagined -368, real -160, gap -208) and seed 4 at 20k (imagined -392, real -120, gap -272). So a large gap of either sign occurs at 20k, and a policy near or somewhat better than random can have a very pessimistic imagined value. Also, the training-window `imagined_return` (the lambda-return at the first imagined step over start states drawn from the replay, bounded readout) is strongly negative at 20k and 30k in every seed (all five below -190) and at 40k in four of five seeds (seed 2 is +34.0; the seed-mean -125 at 40k hides this); the seed-means over 20k-40k are -399, -404 and -125, far below the real evaluation returns of the same checkpoints (-182.0, -87.6, +23.2), while the return scale `scale` and `critic_loss` peak at 20k-30k (seed-mean scale 548 and 790, against 91 at 10k and 77-92 from 60k; critic loss 1,714 and 3,871). Since the reward readout is bounded below by -9, 15 imagined steps contribute at least -135 undiscounted; training-window returns near -400 therefore come from the slow critic's bootstrap values (an inference from the bound, not a separate measurement).
- **What this does and does not show.** The records show that in every Dreamer seed that fell below random, the fall coincides with an evaluation-time imagined return over 100 points above the real return, the pattern M4 §5.5 names as an exploitation signature. They do not show that the policy was exploiting model error: the pattern could also be produced by a value estimate that has not yet caught up with a changing policy (a second explanation, equally untested here), two other checkpoints show a large gap of the opposite sign, the training-window imagined returns are the opposite sign to the evaluation-time ones, and the evaluation-time quantity is a single-state estimate averaged over 100 episodes. **Hypothesis only, not tested here:** that the actor drifted into regions where the world model's reward or continue head over-predicted return and was corrected as the replay grew. It is untestable from what survives: no replay, world model or actor from any checkpoint was retained (state.pt, feature caches and episode frames were deleted after each record verified; see "Disk and retention"), and the plan named no run to keep before the study began (spec §11.1, open question 1). The records alone cannot tell this apart from the alternatives.

### Compute (§7.4)

M5a equalises environment steps (100,000) and gradient steps, not wall-clock time. Every run took 19,000 update rounds. A Dreamer round is one world-model step plus one actor-critic step (so 19,000 of each); a DQN round is one Q-network step (19,000 in total).

**Snippet H.**

```python
sec = {a: [R[(a, s)]["seconds"] for s in range(5)] for a in ("dreamer", "dqn")}
tim = {a: {k: np.mean([R[(a, s)]["timing"][k] for s in range(5)]) for k in ("act", "train", "eval")} for a in ("dreamer", "dqn")}
print({a: (round(np.mean(v)), round(min(v)), round(max(v)), round(sum(v) / 3600, 2)) for a, v in sec.items()})
print("ratio", round(np.mean(sec["dreamer"]) / np.mean(sec["dqn"]), 2), "median ratio", round(np.median(sec["dreamer"]) / np.median(sec["dqn"]), 2),
      [round(x / y, 2) for x, y in zip(sec["dreamer"], sec["dqn"])], "total h", round((sum(sec["dreamer"]) + sum(sec["dqn"])) / 3600, 2))
print({a: {k: round(v) for k, v in t.items()} for a, t in tim.items()}, {k: round(tim["dreamer"][k] / tim["dqn"][k], 2) for k in tim["dreamer"]})
print("train s/round", {a: round(tim[a]["train"] / 19000, 3) for a in tim}, "eval s/ckpt", {a: round(tim[a]["eval"] / 11) for a in tim},
      "eval share", {a: round(tim[a]["eval"] / np.mean(sec[a]), 3) for a in tim})
```

| | Dreamer | DQN | ratio |
|---|---|---|---|
| seconds per run, mean (range) | 18,251 (15,793 to 21,927) | 6,107 (5,136 to 7,600) | 2.99 (median-to-median 2.76) |
| hours, five runs | 25.35 | 8.48 | |
| act, mean seconds | 2,823 | 2,687 | 1.05 |
| train, mean seconds | 14,237 | 2,674 | 5.32 |
| eval, mean seconds | 1,191 | 746 | 1.60 |
| train seconds per update round | 0.749 | 0.141 | |
| eval seconds per evaluation (11 per run) | 108 | 68 | |
| eval share of a run | 6.5% | 12.2% | |

Per seed (Dreamer / DQN) the ratio is 3.07, 3.05, 2.76, 3.46, 2.68. Dreamer used about three times DQN's per-run time; its update phase alone is 5.3x DQN's. The ten runs sum to 33.83 hours (Dreamer 25.35, DQN 8.48). Against the Task 11 smoke projection (`Task 11: step 3`: Dreamer 4.93 h, DQN 1.11 h per run, 30.2 h for ten), the measured mean runs are 5.07 h (Dreamer) and 1.70 h (DQN); the total is 3.6 h above the projection and under the 40 h stop, and above the spec's ~25 h estimate. The measured DQN train time per round (0.141 s) is three times the smoke figure (0.046 s) and Dreamer's (0.749 s) is 10% above it (0.679 s); the evaluation share of a run (6.5% and 12.2%) is below the smoke projection's ~40%, which the smoke results themselves attributed to random-like policies running to time-out. The records do not explain these differences and this section does not. (Added after the final review, which is the only place an explanation was offered: its finding M6 names a plausible but untested contributor to the rise in DQN's train time, a per-sample cost that grows with the replay size, in `TransitionSampler.sample` at `src/mbfps/training/dqn.py:108-121` and `SequenceLoader.sample` at `src/mbfps/data/loader.py:207-231`. No run or measurement here tests it; see Follow-ups before M5b.)

Dreamer seeds 3 and 4 measured slower than the others: 21,927 s and 20,362 s, against 15,793 s, 15,953 s and 17,220 s for seeds 0-2, with the difference in the `train` component (17,379 s and 16,416 s, against 11,977 s to 13,412 s). This is reported without explanation. It does not enter the reading, which is on returns and not on time; using seeds 0-2 only, the ratio is 2.95 (Dreamer mean 16,322 s against DQN 5,531 s), so the 2.99 headline is not an artefact of the two slower seeds. Per-run `seconds` and `timing` are `time.perf_counter` sums and exclude periods the machine was asleep.

**Amount of data replayed per update (spec §3.2).** §3.2 says the arms replay different amounts per update (Dreamer 16 × 64 = 1,024 timesteps, DQN 256 transitions) and that "the results report both". The Dreamer figure that counts is not the padded 1,024. `diagnostics["n_starts"] + diagnostics["n_dropped"]` is the number of real (mask-true) timesteps in the batch the round used: `ActorCriticLearner.step` sets `n_real = batch["mask"].sum()`, `n_starts` is the number of start states kept and `n_dropped = n_real - n_starts` (`src/mbfps/training/ac_trainer.py:178` and `:220-221`), and the diagnostics are means over the update rounds since the previous checkpoint. The DQN records carry no such field because the batch is always 256 transitions (`DQNConfig.batch_size`, `src/mbfps/training/dqn.py:29`).

**Snippet I.**

```python
D = [R[("dreamer", s)] for s in range(5)]
real = lambda s, c: ev(D[s], c)["diagnostics"]["n_starts"] + ev(D[s], c)["diagnostics"]["n_dropped"]
for s in range(5):
    print(s, *(f"{real(s, c):.0f}" for c in CK[1:]))
print("mean", *(f"{np.mean([real(s, c) for s in range(5)]):.0f}" for c in CK[1:]))
print("seeds below 256:", {c: sum(real(s, c) < 256 for s in range(5)) for c in CK[1:]})
print("max over seeds, 70k..100k:", round(max(real(s, c) for s in range(5) for c in CK if c >= 70000)))
```

Real (unpadded) timesteps per Dreamer update round, window mean at each checkpoint, per seed:

| seed | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 100k |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 612 | 610 | 620 | 532 | 305 | 219 | 183 | 162 | 151 | 142 |
| 1 | 647 | 666 | 663 | 652 | 519 | 303 | 231 | 194 | 173 | 163 |
| 2 | 647 | 595 | 601 | 414 | 244 | 193 | 167 | 151 | 142 | 137 |
| 3 | 693 | 678 | 651 | 559 | 340 | 238 | 198 | 173 | 158 | 150 |
| 4 | 664 | 652 | 642 | 595 | 397 | 262 | 205 | 182 | 165 | 155 |
| **mean** | 652 | 640 | 635 | 550 | 361 | 243 | 197 | 172 | 158 | 150 |
| seeds below 256 | 0 | 0 | 0 | 0 | 1 | 3 | 5 | 5 | 5 | 5 |

| per update | amount |
|---|---|
| Dreamer, padded batch (16 × 64) | 1,024 timesteps |
| Dreamer, real timesteps, 10k window | 612 to 693 (mean 652) |
| Dreamer, real timesteps, 100k window | 137 to 163 (mean 150) |
| DQN | 256 transitions, constant |

Dreamer replays 1,024 timesteps per update only if padding counts. Counting real timesteps, it replays 2.4 to 2.7 times DQN's 256 in the first window and fewer than DQN from 70k on: at 70k, 80k, 90k and 100k every Dreamer seed is below 256 (the largest value in any seed from 70k on is 231), and at 60k three of five seeds are. The likely cause (the final review's reading; not tested here) is in the data and not in the loop: uniform episode sampling over a replay that fills with roughly 6-step kill episodes (see the dropped-share discussion in the diagnostics above) leaves most of each 64-step window as padding. The verdict is unaffected, since it is read on returns at equal environment steps and equal update rounds; this table only qualifies the §3.2 framing that Dreamer always replays about four times as much data per gradient step. It is a count of timesteps in the batch; it does not weight them by how informative they are, and this section does not claim either arm learned more per replayed sample.

### Disk and retention (§9, §11.1)

**Decided before any run (Task 12, Step 2):** no run's replay is retained. Disk at the start was 14 GiB free (`Task 12: step 2`), above the 8 GB bar. Each run's feature caches (`*.features.npy`) and `state.pt` were deleted by `scripts/train_online.py` only after `verify_record` passed on the run's record; the record then carries `replay_features_deleted` True (snippet A: True in all ten).

**Deviation from the §9 text, a gap in the spec and not a change of reading.** On 2026-10-15 the disk reached 5.8 GiB free and the user decided at about 16:28 (the controller's ruling; the ledger line `Task 12: user decision` says "16:4x", which the controller has corrected, and the watcher's first deletion is logged at 16:29). Spec §9 estimated disk as "~4.9 GB of features per run" and counted only the feature caches. It did not count the roughly 2 GB of episode frame files (`replay/*.npz`) each run keeps, and the plan's Global Constraints said "episode `.npz` files and records are kept". With ten of them, the study would have run out of disk during DQN seed 2. The user chose to delete each run's episode frames after its record verifies. This was done by an ops watcher outside the study code, `docs/superpowers/evidence/m5a/frame_reaper.py` (log `docs/superpowers/evidence/m5a/frame_reaper.log`), so the code did not change and all ten runs keep one `git_sha`. The watcher deletes a run's frames only when `verify_record` passes, `replay_features_deleted` is True, `state.pt` is gone and no feature file remains; it never touches `record.json`. The log has ten run lines, one per run (plus a closing `all ten runs reaped; exiting` line at 2026-10-16 21:26), each `verified; deleted 9,673 to 11,538 episode frame files (1.95 to 1.98 GiB); leftovers []`; dreamer s0, s1 and dqn s0, s1 were reaped in the first pass on 2026-10-15 at 16:29-16:30 and the last, dqn s4, on 2026-10-16 at 21:24. Evidence: `ls runs/m5a/*/` lists exactly one file, `record.json`, in each of the ten run directories, and `du -sh runs/m5a` prints `4.7M`; the disk had 15 GiB free after the last run (`df` at 2026-10-16 21:26, per the controller).

**Consequence.** Nothing but the ten records, the read outputs and the logs survives from the study: no replay, no world-model or policy checkpoints. Every number in this section comes from the records, which carry every evaluation episode (return, length, kill, truncation, imagined return) at every checkpoint, so nothing reported here is lost. Anything that needs the replay or a checkpoint, such as examining the below-random Dreamer checkpoints, needs a rerun.

### Limits (§10)

Restated from spec §10 (the first four bullets below; the last two are this section's own additions, not in §10):

- **One scenario, one ratio, one budget, one baseline.** `basic` only, one update round per 5 environment steps, 100,000 steps per run, Double DQN as the only model-free comparator, all fixed in advance. `defend_the_center`, `deadly_corridor` and the encoder comparison are M5b.
- **Environment-step efficiency, not compute efficiency.** Equal environment steps and equal gradient steps; Dreamer's wall-clock was about 3x DQN's and its update phase 5.3x (above).
- **Shared representation by design.** Both arms read frozen DINOv2 features through the same trainable `BottleneckEncoder` class, so nothing here speaks to raw-pixel model-free performance.
- **M3's tools** (`eval/study.py`'s routing through `load_world_model` / `read_reward`) remain a follow-up; M5a never called it.
- **Addition (not in §10): five seeds per arm.** The reading is the pre-registered 21-of-25 bar; 24 of 25 clears it, but five seeds per arm cannot support a claim about the spread of either arm's behaviour beyond what the table shows.
- **Addition (not in §10): unexplained by the records:** why Dreamer's curve is slower to rise and has deep transient dips; why its seeds 3 and 4 measured slower; why the smoke compute projection was off. The diagnostics above are descriptions, and the one hypothesis is labelled as such.

### Follow-ups before M5b

From the final review (`.superpowers/sdd/2026-10-14-mb-fps-m5a-online-sample-efficiency/final-review.md`, untracked) and its deferred-minor triage. None changes a recorded number. Line numbers are at the commit that added this list.

- **M3.** `verify_record` (`src/mbfps/training/online.py:222`) and `run_problems` (`src/mbfps/eval/efficiency.py:43-67`) do not check `record["config"] == asdict(cfg)` or `update_rounds == (budget - prefill) / act_steps * rounds_per_act`, and the record does not carry `arm_kwargs` (written only to `state.pt`, `online.py:168-171` and `_checkpoint`), so a finished record cannot show which DQN settings it ran with. Write `arm_kwargs` into the record, check both in `verify_record`, add a test.
- **M4.** A fresh start prunes every `ep_*` file under `<out>/replay` without proof that it owns them (`src/mbfps/training/online.py:91-100` and the call at `:138`). Write a marker holding `meta` before the first episode; prune only when it is absent or matches, otherwise refuse a non-empty `replay/`.
- **M5.** A NaN AUC gives `NOT_SEPARATED` and exit 0 instead of 91 (`src/mbfps/eval/efficiency.py:24-40` and `run_problems`), and a malformed eval entry raises `KeyError` so the read exits 1 instead of 91 (`scripts/m5a_study.py:69`). Add an `np.isfinite(auc)` check to `run_problems` and catch `KeyError`/`StopIteration` there.
- **M6.** Per-sample cost grows with the replay size: `SequenceLoader._usable()` is rebuilt over every episode and `rng.choice` is called on that list (`src/mbfps/data/loader.py:199-205`, `:218`, `:231`); `TransitionSampler.sample` calls `np.searchsorted` on a Python list once per transition (`src/mbfps/training/dqn.py:112-121`; `_ends` is a list, `:85`). Keep `_ends` as a NumPy array and cache the usable list, keeping `rng.choice` over an equal-valued array so the draws and future numbers are unchanged. A plausible, untested contributor to DQN's train time rising from 0.046 s to 0.141 s per round.
- **M7.** `AgentPolicy.reset` reseeds the global torch generator at every training-episode start and every eval-policy construction (`src/mbfps/eval/live.py:42`), coupling Dreamer's training noise to acting and evaluation. Document it in the M5b spec, or give the policy its own `torch.Generator` (that changes future numbers, so it belongs in M5b).
- **Must land before M5b: `QNetwork` hard-codes the encoder.** `EncoderConfig(kind="frozen_ssl")` at `src/mbfps/training/dqn.py:54`; needed before M5b's encoder comparison.
- **Must land before M5b: the random action hard-codes 4 actions.** `self._rng.integers(0, 4)` at `src/mbfps/training/dqn.py:174`; needed before any scenario with another action count.
- **Must land before M5b: the chain test cannot tell Double DQN from vanilla.** `test_double_dqn_learns_a_chain` (`tests/training/test_dqn.py:88`) covers only n=1 and passes for either; spec §5.4 asks the known-answer test to check both.
- **Fold frame deletion into code.** Extend `delete_replay_features` (`src/mbfps/training/online.py:244`) to: verify the record, flag it (`replay_features_deleted`, plus a new `replay_frames_deleted`), then delete `state.pt`, the features, then the episode frames, leaving `replay/` empty; with tests, a corrected docstring ("Episode files and the record are kept" is now false for M5a), and an update to spec §9's disk arithmetic (about 2 GB of frames per run). This replaces the untracked-then-committed ops script `docs/superpowers/evidence/m5a/frame_reaper.py` with tested code.
