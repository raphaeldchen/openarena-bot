# MB-FPS M4 — Actor-Critic in Imagination on `basic` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a Dreamer-style actor-critic purely inside a frozen world model's 15-step imagination on ViZDoom `basic`, gated first on whether imagination predicts reward and episode end, then on live play against a random policy, then across the 3 arms × 3 seeds.

**Architecture:** Stage A extends the existing M3 world-model code (padded short episodes with a loss mask, a symlog reward head, a configurable action count) and adds an entry check (`mbfps.eval.entry`). Stage B adds `Actor`/`Critic` heads, `RSSM.imagine_policy`, the actor-critic trainer (`mbfps.training.ac_trainer`) and live evaluation (`mbfps.eval.live`). Stage C is a driver (`scripts/m4_study.py`) running the same code over nine cells plus the pre-registered cross-arm reading. Every gate is a script `read` phase with a distinct exit code per outcome.

**Tech Stack:** Python 3.12, PyTorch (MPS, fp32), NumPy, ViZDoom 1.3.0, gymnasium, pytest. Run everything with `.venv/bin/python` from the main checkout's venv (`/Users/raphaelchen/Desktop/csgo-bot/.venv`).

**Spec:** `docs/superpowers/specs/2026-10-10-mb-fps-m4-actor-critic-design.md` (read it alongside this plan; section numbers below — §4.5 etc. — refer to it).

## Global Constraints

- Imagination horizon **15**; γ **0.997**; λ **0.95**; actor and critic LR **3e-5**; entropy **3e-4**; critic EMA **0.98**; return-scale percentile EMA **0.99**; head MLPs **2 × 512** SiLU.
- `frame_skip = 4`; `basic` has **4** actions (no-op + `MOVE_LEFT`, `MOVE_RIGHT`, `ATTACK`); `my_way_home` has **6**.
- World model: 20,000 steps, batch 16 × `seq_len` 64, `free_bits` 0.20 (unchanged constant `KL_FREE_BITS`), fp32.
- Reward head on `symlog(r) = sign(r)·ln(1+|r|)` for every M4 world model; read through `symexp`. M3 checkpoints are `identity`.
- Every interval: episode-clustered (Stage A) or seed-paired (Stage B) bootstrap, **2,000** resamples, 95% percentile interval via `mbfps.eval.pooling.percentile_interval`. A gate passes only on `ci_low > 0`; compare with `>`, so NaN fails.
- Live evaluation seeds **1,000,000–1,000,099** (100 episodes), disjoint from collection seeds. Random baseline on the same seeds.
- Actor-critic: **10,000** updates; live evaluation at updates **2,000, 4,000, 6,000, 8,000, 10,000**; gate read at **10,000 only**.
- The world model is **frozen** during actor-critic training: `requires_grad=False` on every world-model parameter; the optimiser holds only actor and critic parameters.
- `privileged` is evaluation-only and never reaches a training gradient (existing `to_device` whitelist; keep it that way).
- `encoders.py` stays the only module that differs between arms.
- Never `rm` anything under `runs/` or `data/my_way_home/`. Feature caches under `data/basic/` are cleared only with `scripts/cache_features.py --clear`.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Run scripts with a redirect, then read `$?`, never through a pipe: `cmd > log 2>&1; echo "exit: $?"`. A pipe reports the pager's status (M3n Task 8).

## Review Focus

1. **A kill one step before termination.** In 19 of 109 measured kills the +97 arrives with `terminated=False` and the next step is `terminated=True` at −1. Every slot alignment (entry readings, λ-returns, continue targets) must put each on its own step. Pinned in Task 6 (`test_kill_then_terminal_on_separate_steps`).
2. **An episode shorter than `seq_len` with a terminal on its last real step.** The mask must include that step and exclude every pad after it, in every loss term. Pinned in Task 4 (`test_short_episode_terminal_step_is_masked_in`) and Task 5 (`test_padded_loss_equals_unpadded_loss`).
3. **Loading an M3 checkpoint into M4 code.** No `n_actions`, `reward_transform` or `pad_short` in its payload; it must load as 6 actions, `identity`, unpadded — that is §4.6's control. Pinned in Task 5 (`test_m3_payload_loads_with_defaults`).
4. **Starting imagination from a padded position.** A start state taken from a pad would train the actor on garbage. Pinned in Task 10 (`test_start_states_exclude_padding`).
5. **The first live frame.** It must be filtered as if a no-op produced it, and the agent's RNG must be reseeded per episode so a rerun reproduces it. Pinned in Task 11 (`test_first_frame_is_filtered_with_noop` and `test_agent_episode_is_reproducible`).

---

## File Structure

| file | responsibility |
|---|---|
| `src/mbfps/data/policies.py` (modify) | `ScriptedPolicy(sweep_len=...)` |
| `scripts/collect.py` (modify) | `--sweep-len`, `--policies` |
| `src/mbfps/models/transforms.py` (create) | `symlog`, `symexp`, `read_reward` |
| `src/mbfps/utils/config.py` (modify) | `TrainConfig.n_actions`, `.pad_short`, `.reward_transform` |
| `src/mbfps/data/loader.py` (modify) | `pad_short`: sample short episodes, pad, emit `mask` |
| `src/mbfps/utils/device.py` (modify) | `mask` joins the device whitelist |
| `src/mbfps/models/rssm.py` (modify) | masked `kl_loss`; `imagine_policy` |
| `src/mbfps/training/world_model.py` (modify) | masked losses, symlog target, payload fields, `load_world_model` |
| `scripts/train_world_model.py` (modify) | `--n-actions`, `--pad-short`, `--reward-transform`, `--record` |
| `src/mbfps/eval/entry.py` (create) | §4.5 arrays, baselines, readings, verdict |
| `scripts/entry_check.py` (create) | Stage A measure/read, §4.6 control, exit codes |
| `src/mbfps/models/heads.py` (modify) | `Actor`, `Critic` |
| `src/mbfps/training/ac_trainer.py` (create) | λ-returns, `ReturnScale`, losses, start states, training loop |
| `scripts/train_actor_critic.py` (create) | Stage B training driver |
| `src/mbfps/eval/live.py` (create) | `AgentPolicy`, `run_episodes`, `paired_gate` |
| `scripts/eval_live.py` (create) | Stage B measure/read, untrained control, exit codes |
| `scripts/m4_study.py` (create) | Stage C driver and the §6.2 reading |

---

## Task 1: `ScriptedPolicy` sweep length and the collector's policy mix

**Files:**
- Modify: `src/mbfps/data/policies.py` (class `ScriptedPolicy`)
- Modify: `scripts/collect.py`
- Test: `tests/data/test_policies.py`

**Interfaces:**
- Produces: `ScriptedPolicy(button_names, seed=0, sweep_len=100)`; `collect.py --sweep-len N --policies random,scripted`.

- [ ] **Step 1: Write the failing tests** (append to `tests/data/test_policies.py`)

```python
def test_sweep_len_default_reproduces_the_shipped_stream(obs):
    """sweep_len=100 is the shipped _SWEEP_LEN; the action stream must not move."""
    shipped = _run(ScriptedPolicy(HOME, seed=3), obs, n=500)
    explicit = _run(ScriptedPolicy(HOME, seed=3, sweep_len=100), obs, n=500)
    assert shipped == explicit


def test_sweep_len_10_reverses_every_10_steps_on_basic(obs):
    """On `basic` (no MOVE_FORWARD, no TURN_*) the sweep strafes. With
    sweep_len=10 the strafe direction flips at t = 10, 20, ... -- the shipped
    100 never flips inside a 75-step episode.

    Counted, not asserted per step: 1 step in 8 falls through to a uniform
    random action, which can be either strafe in either phase. The sweep's own
    choice must dominate each phase."""
    policy = ScriptedPolicy(BASIC, seed=0, sweep_len=10)
    left, right = 1, 2  # MOVE_LEFT, MOVE_RIGHT action indices
    counts = {0: {left: 0, right: 0}, 1: {left: 0, right: 0}}
    for t in range(1, 401):          # `act` increments t before reading it
        a = policy.act(obs)
        if a in (left, right):
            counts[(t // 10) % 2][a] += 1
    assert counts[0][left] > 4 * counts[0][right]
    assert counts[1][right] > 4 * counts[1][left]


def test_shipped_sweep_never_reverses_inside_a_basic_episode(obs):
    policy = ScriptedPolicy(BASIC, seed=0)
    actions = [policy.act(obs) for _ in range(75)]
    assert actions.count(1) > 4 * actions.count(2)


def test_sweep_len_must_be_positive():
    with pytest.raises(ValueError, match="sweep_len"):
        ScriptedPolicy(BASIC, seed=0, sweep_len=0)
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/data/test_policies.py -k sweep_len -q`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'sweep_len'`.

- [ ] **Step 3: Implement**

In `ScriptedPolicy`, keep `_SWEEP_LEN = 100` as the default and add the argument:

```python
    def __init__(
        self, button_names: Sequence[str], seed: int = 0, sweep_len: int = _SWEEP_LEN
    ) -> None:
        if sweep_len < 1:
            raise ValueError(f"sweep_len must be >= 1, got {sweep_len}")
        self.name = "scripted"
        self._seed = seed
        self._sweep_len = int(sweep_len)
        # ... the rest of __init__ unchanged ...
```

`_SWEEP_LEN` is referenced as a default before the class body finishes, so define the default as a module constant just above the class:

```python
_DEFAULT_SWEEP_LEN = 100
"""The shipped sweep length. On `basic` (episodes <= 75 steps) it never
reverses, so M4 collects `basic` with 10 (spec §3)."""
```

and use `sweep_len: int = _DEFAULT_SWEEP_LEN` in the signature (keep `_SWEEP_LEN = _DEFAULT_SWEEP_LEN` as a class attribute for anything that reads it). In `act`, replace `self._SWEEP_LEN` with `self._sweep_len`:

```python
            sweeping_left = (self._t // self._sweep_len) % 2 == 0
```

- [ ] **Step 4: Run the policy tests**

Run: `.venv/bin/python -m pytest tests/data/test_policies.py -q`
Expected: all PASS.

- [ ] **Step 5: Add the collector flags**

In `scripts/collect.py`, add after `--frame-skip`:

```python
    parser.add_argument(
        "--sweep-len", type=int, default=100,
        help="ScriptedPolicy sweep length; basic is collected with 10 (M4 spec §3)",
    )
    parser.add_argument(
        "--policies", default="random,scripted",
        help="comma-separated policies, alternated in this order per episode",
    )
```

Build `ScriptedPolicy(button_names, seed=args.seed, sweep_len=args.sweep_len)`, and replace the alternation line with:

```python
    order = [name.strip() for name in args.policies.split(",") if name.strip()]
    unknown = sorted(set(order) - set(collectors))
    if not order or unknown:
        raise SystemExit(f"--policies must name some of {sorted(collectors)}; got {args.policies!r}")
    # ... inside the loop:
        name = order[i % len(order)]
```

- [ ] **Step 6: Check the script still parses and runs a tiny collection**

```bash
.venv/bin/python scripts/collect.py --scenario basic --episodes 4 --sweep-len 10 \
  --policies random,scripted --out "$TMPDIR/m4_collect_smoke" > "$TMPDIR/m4_collect_smoke.log" 2>&1
echo "exit: $?"; tail -4 "$TMPDIR/m4_collect_smoke.log"
```

Expected: exit 0, `episodes_kept=4`.

- [ ] **Step 7: Commit**

```bash
git add src/mbfps/data/policies.py scripts/collect.py tests/data/test_policies.py
git commit -F - <<'EOF'
feat: ScriptedPolicy sweep length, and a policy mix the collector takes

On basic the shipped 100-step sweep never reverses inside a 75-step
episode. sweep_len defaults to the shipped value, so existing streams
are unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

## Task 2: `symlog` / `symexp` and `read_reward`

**Files:**
- Create: `src/mbfps/models/transforms.py`
- Test: `tests/models/test_transforms.py`

**Interfaces:**
- Produces: `symlog(x: Tensor) -> Tensor`, `symexp(y: Tensor) -> Tensor`, `REWARD_TRANSFORMS = ("identity", "symlog")`, `read_reward(raw_head_output: Tensor, transform: str) -> Tensor` (the head output mapped back to reward units).

- [ ] **Step 1: Write the failing tests**

```python
import pytest
import torch

from mbfps.models.transforms import REWARD_TRANSFORMS, read_reward, symexp, symlog


def test_round_trip_over_the_basic_reward_range():
    x = torch.linspace(-9.0, 101.0, 1101, dtype=torch.float64)
    torch.testing.assert_close(symexp(symlog(x)), x, rtol=0, atol=1e-9)


def test_both_are_odd():
    x = torch.tensor([0.5, 4.0, 97.0], dtype=torch.float64)
    torch.testing.assert_close(symlog(-x), -symlog(x))
    torch.testing.assert_close(symexp(-x), -symexp(x))


def test_known_values():
    """The spec's own figures: -4 -> -1.61, +97 -> +4.58."""
    out = symlog(torch.tensor([-4.0, 97.0]))
    torch.testing.assert_close(out, torch.tensor([-1.6094, 4.5850]), atol=1e-4, rtol=0)


def test_read_reward_identity_is_the_raw_output():
    y = torch.tensor([1.0, -2.0])
    assert torch.equal(read_reward(y, "identity"), y)


def test_read_reward_symlog_inverts():
    y = symlog(torch.tensor([97.0, -4.0]))
    torch.testing.assert_close(read_reward(y, "symlog"), torch.tensor([97.0, -4.0]))


def test_unknown_transform_is_refused():
    assert REWARD_TRANSFORMS == ("identity", "symlog")
    with pytest.raises(ValueError, match="reward_transform"):
        read_reward(torch.zeros(1), "log")
```

- [ ] **Step 2: Run to see it fail**

Run: `.venv/bin/python -m pytest tests/models/test_transforms.py -q`
Expected: FAIL — `ModuleNotFoundError: mbfps.models.transforms`.

- [ ] **Step 3: Implement**

```python
"""Reward transforms (milestone M4).

`basic`'s per-step rewards span -9 to +99: a squared error on the raw scale is
dominated by the kill by three orders of magnitude. DreamerV3 trains the reward
head on `symlog` and reads it through `symexp`. A checkpoint records which
transform its head was trained with (`reward_transform`), and every consumer
reads the head through `read_reward` -- never by assumption, because the M3
checkpoints are `identity`.
"""

import torch

REWARD_TRANSFORMS: tuple[str, ...] = ("identity", "symlog")


def symlog(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * ln(1 + |x|)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(y: torch.Tensor) -> torch.Tensor:
    """sign(y) * (exp(|y|) - 1), the inverse of `symlog`."""
    return torch.sign(y) * torch.expm1(torch.abs(y))


def _require_known(transform: str) -> None:
    if transform not in REWARD_TRANSFORMS:
        raise ValueError(
            f"unknown reward_transform {transform!r}; available: {list(REWARD_TRANSFORMS)}"
        )


def reward_target(rewards: torch.Tensor, transform: str) -> torch.Tensor:
    """What the reward head is trained to output for `rewards`."""
    _require_known(transform)
    return symlog(rewards) if transform == "symlog" else rewards


def read_reward(head_output: torch.Tensor, transform: str) -> torch.Tensor:
    """The reward head's output in reward units."""
    _require_known(transform)
    return symexp(head_output) if transform == "symlog" else head_output
```

Add to the test file:

```python
from mbfps.models.transforms import reward_target


def test_reward_target_matches_read_reward():
    r = torch.tensor([-9.0, -4.0, 97.0])
    for transform in REWARD_TRANSFORMS:
        torch.testing.assert_close(read_reward(reward_target(r, transform), transform), r)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/models/test_transforms.py -q`
Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/models/transforms.py tests/models/test_transforms.py
git commit -F - <<'EOF'
feat: symlog reward transform, read through the transform the head was trained with

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

## Task 3: `TrainConfig` carries the action count, padding and reward transform

**Files:**
- Modify: `src/mbfps/utils/config.py` (`TrainConfig`)
- Test: `tests/utils/test_config.py`

**Interfaces:**
- Produces: `TrainConfig.n_actions: int = 6`, `TrainConfig.pad_short: bool = False`, `TrainConfig.reward_transform: str = "identity"`; `get_config(arm, n_actions=4, pad_short=True, reward_transform="symlog")` works through the existing `**overrides`.

The defaults are M3's values, so every M3 config, checkpoint and test is unchanged.

- [ ] **Step 1: Write the failing tests** (append to `tests/utils/test_config.py`)

```python
from mbfps.utils.config import TrainConfig, get_config


def test_m4_fields_default_to_m3_behaviour():
    train = TrainConfig()
    assert (train.n_actions, train.pad_short, train.reward_transform) == (6, False, "identity")


def test_m4_fields_override_through_get_config():
    cfg = get_config("frozen_ssl", n_actions=4, pad_short=True, reward_transform="symlog")
    assert (cfg.train.n_actions, cfg.train.pad_short, cfg.train.reward_transform) == (4, True, "symlog")


def test_unknown_reward_transform_is_refused_at_construction():
    import pytest
    with pytest.raises(ValueError, match="reward_transform"):
        get_config("frozen_ssl", reward_transform="log")


def test_n_actions_must_be_positive():
    import pytest
    with pytest.raises(ValueError, match="n_actions"):
        get_config("frozen_ssl", n_actions=0)
```

- [ ] **Step 2: Run to see it fail**

Run: `.venv/bin/python -m pytest tests/utils/test_config.py -q -k "m4_fields or reward_transform or n_actions"`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'n_actions'`.

- [ ] **Step 3: Implement**

`config.py` must not import `mbfps.models.transforms` at module top if that would create a cycle; `transforms.py` imports only torch, so it is safe. Add to `TrainConfig`:

```python
    n_actions: int = 6
    """Width of the RSSM's one-hot action input. 6 is `my_way_home` and every
    M3 checkpoint; `basic` is 4 (M4 spec §3)."""
    pad_short: bool = False
    """Sample episodes shorter than `seq_len`, padded and masked (M4 spec §4.2).
    False is M3's loader exactly."""
    reward_transform: str = "identity"
    """`identity` (M3) or `symlog` (M4 spec §4.3)."""

    def __post_init__(self) -> None:
        from mbfps.models.transforms import REWARD_TRANSFORMS

        if self.reward_transform not in REWARD_TRANSFORMS:
            raise ValueError(
                f"unknown reward_transform {self.reward_transform!r}; "
                f"available: {list(REWARD_TRANSFORMS)}"
            )
        if self.n_actions < 1:
            raise ValueError(f"n_actions must be >= 1, got {self.n_actions}")
```

- [ ] **Step 4: Run the config tests and the M3 suite that builds configs**

Run: `.venv/bin/python -m pytest tests/utils/test_config.py tests/training/test_world_model.py -q`
Expected: all PASS (M3 behaviour unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/utils/config.py tests/utils/test_config.py
git commit -F - <<'EOF'
feat: TrainConfig carries n_actions, pad_short and reward_transform, at M3's values

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

## Task 4: The loader pads short episodes and emits a mask

**Files:**
- Modify: `src/mbfps/data/loader.py` (`SequenceLoader.__init__`, `_usable`, `sample`)
- Modify: `src/mbfps/utils/device.py` (`_TRAINING_KEYS`)
- Test: `tests/data/test_loader.py`, `tests/utils/test_device.py`

**Interfaces:**
- Produces: `SequenceLoader(..., pad_short: bool = False)`. With `pad_short=True`, `sample()` adds `batch["mask"]`: `(B, T)` bool, True on real transitions. Without it, the batch has no `mask` key and is byte-identical to before.
- `to_device` moves `mask`.

- [ ] **Step 1: Write the failing tests** (append to `tests/data/test_loader.py`)

```python
@pytest.fixture
def mixed_buffer(tmp_path):
    """Two long episodes (80) and one short one (6) whose last step terminates."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=80, fill=1))
    buf.add(make_episode(t=80, fill=2))
    short = make_episode(t=6, fill=3)
    short.rewards[:] = 1.0
    short.terminated[-1] = True
    buf.add(short)
    return buf


def test_default_loader_never_samples_a_short_episode(mixed_buffer):
    loader = SequenceLoader(mixed_buffer, batch_size=32, seq_len=16, seed=0)
    batch = loader.sample()
    assert "mask" not in batch
    assert not np.any(batch["obs"][:, 0, 0, 0, 0] == 3)


def test_pad_short_samples_a_short_episode(mixed_buffer):
    loader = SequenceLoader(mixed_buffer, batch_size=32, seq_len=16, seed=0, pad_short=True)
    batch = loader.sample()
    short_rows = np.flatnonzero(batch["obs"][:, 0, 0, 0, 0] == 3)
    assert short_rows.size > 0


def test_short_episode_terminal_step_is_masked_in(mixed_buffer):
    loader = SequenceLoader(mixed_buffer, batch_size=32, seq_len=16, seed=0, pad_short=True)
    batch = loader.sample()
    row = int(np.flatnonzero(batch["obs"][:, 0, 0, 0, 0] == 3)[0])
    assert batch["mask"][row].tolist() == [True] * 6 + [False] * 10
    assert batch["terminated"][row, 5]
    assert not batch["terminated"][row, 6:].any()
    assert np.all(batch["rewards"][row, :6] == 1.0)
    assert np.all(batch["rewards"][row, 6:] == 0.0)
    assert batch["window_start"][row] == 0
    # obs carries T+1 rows: 7 real frames, then zero padding.
    assert np.all(batch["obs"][row, :7] == 3)
    assert np.all(batch["obs"][row, 7:] == 0)


def test_pad_short_masks_long_windows_all_true(mixed_buffer):
    loader = SequenceLoader(mixed_buffer, batch_size=32, seq_len=16, seed=0, pad_short=True)
    batch = loader.sample()
    long_rows = np.flatnonzero(batch["obs"][:, 0, 0, 0, 0] != 3)
    assert batch["mask"][long_rows].all()


def test_pad_short_pads_features_with_zeros(mixed_buffer):
    rng = np.random.default_rng(0)
    for path in mixed_buffer.episode_paths():
        n = len(np.load(path)["actions"]) + 1
        np.save(path.with_suffix(feature_suffix("dinov2")),
                (rng.random((n, 64, 384)) + 1.0).astype(np.float16))
    loader = SequenceLoader(mixed_buffer, batch_size=32, seq_len=16, seed=0,
                            pad_short=True, load_obs=True, load_features=True)
    batch = loader.sample()
    row = int(np.flatnonzero(batch["obs"][:, 0, 0, 0, 0] == 3)[0])
    assert batch["features"].shape[1] == 17
    assert np.all(batch["features"][row, :7] > 0)
    assert np.all(batch["features"][row, 7:] == 0)
```

Append to `tests/utils/test_device.py`:

```python
def test_mask_is_moved_to_the_device():
    import numpy as np
    import torch
    from mbfps.utils.device import to_device

    moved = to_device({"mask": np.ones((2, 3), dtype=bool)}, torch.device("cpu"))
    assert moved["mask"].dtype == torch.bool
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/bin/python -m pytest tests/data/test_loader.py tests/utils/test_device.py -q -k "pad_short or short or mask"`
Expected: FAIL — `unexpected keyword argument 'pad_short'`, and `KeyError: 'mask'`.

- [ ] **Step 3: Implement the loader change**

Add `pad_short: bool = False` as the last `__init__` parameter and store `self.pad_short = pad_short`. Replace `_usable`:

```python
    def _usable(self) -> list[int]:
        """Episodes a window can be drawn from. With `pad_short` every non-empty
        episode is eligible and a short one is padded (M4 spec §4.2): `basic`'s
        kill episodes have a median length of 20, so a 64-step floor would
        discard most of the reward events the world model has to learn."""
        floor = 1 if self.pad_short else self.seq_len
        return [i for i, ep in enumerate(self._episodes) if ep.length >= floor]
```

In `sample`, replace the per-row body with a version that pads. Keep the RNG draw count identical for long episodes:

```python
        masks = []
        for _ in range(self.batch_size):
            idx = int(self._rng.choice(usable))
            ep = self._episodes[idx]
            start = int(self._rng.integers(0, max(ep.length - self.seq_len, 0) + 1))
            end = min(start + self.seq_len, ep.length)
            real = end - start
            pad = self.seq_len - real
            actions.append(_pad(ep.actions[start:end], pad))
            rewards.append(_pad(ep.rewards[start:end], pad))
            terminated.append(_pad(ep.terminated[start:end], pad))
            truncated.append(_pad(ep.truncated[start:end], pad))
            masks.append(np.arange(self.seq_len) < real)
            indices.append(idx)
            starts.append(start)
            if self.load_obs:
                obs.append(_pad(ep.obs[start : end + 1], pad))
            if self.load_features:
                features.append(_pad(np.asarray(self._features[idx][start : end + 1]), pad))
            if include_privileged:
                privileged.append(_pad(ep.privileged[start : end + 1], pad))
```

and after building `batch`:

```python
        if self.pad_short:
            batch["mask"] = np.stack(masks)
```

For a long episode `max(ep.length - self.seq_len, 0) + 1` equals the old `ep.length - self.seq_len + 1`, `pad` is 0, and `_pad` returns its input unchanged, so the default path is byte-identical. Add the helper at module level:

```python
def _pad(rows: np.ndarray, pad: int) -> np.ndarray:
    """`rows` followed by `pad` zero rows of the same trailing shape and dtype."""
    if pad == 0:
        return rows
    filler = np.zeros((pad, *rows.shape[1:]), dtype=rows.dtype)
    return np.concatenate([rows, filler], axis=0)
```

Update the `ValueError` message in `sample` to name the floor: `f"no episodes long enough for seq_len={self.seq_len} (pad_short={self.pad_short}); ..."`.

- [ ] **Step 4: Add `mask` to the device whitelist**

In `src/mbfps/utils/device.py`:

```python
_TRAINING_KEYS = ("obs", "features", "actions", "rewards", "terminated", "truncated", "mask")
```

and add one sentence to its docstring: "`mask` (M4) marks real transitions in a padded window."

- [ ] **Step 5: Run the loader and device suites in full**

Run: `.venv/bin/python -m pytest tests/data/test_loader.py tests/utils/test_device.py -q`
Expected: all PASS, including every pre-existing test (the default path is unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/data/loader.py src/mbfps/utils/device.py tests/data/test_loader.py tests/utils/test_device.py
git commit -F - <<'EOF'
feat: the loader samples short episodes padded, with a mask of real transitions

Off by default, and byte-identical when off.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

## Task 5: Masked world-model losses, the symlog reward head, and `load_world_model`

**Files:**
- Modify: `src/mbfps/models/rssm.py` (`_categorical_kl`, `kl_loss`)
- Modify: `src/mbfps/training/world_model.py` (`WorldModel.__init__`, `forward`, `_save_checkpoint`, `train_world_model`; add `load_world_model`)
- Modify: `scripts/train_world_model.py`
- Test: `tests/training/test_world_model.py`, `tests/models/test_rssm.py`

**Interfaces:**
- Consumes: Task 2 `reward_target`; Task 3 `TrainConfig` fields; Task 4 `pad_short`, `batch["mask"]`.
- Produces:
  - `kl_loss(post_logits, prior_logits, free_bits=..., dyn_scale=..., rep_scale=..., mask: Tensor | None = None)`
  - checkpoint payload keys: `arm`, `seed`, `sample_temperature`, `state_dict`, **`n_actions`, `reward_transform`, `pad_short`** (+ `step` on a rung)
  - `load_world_model(path: Path, device: torch.device) -> WorldModel` (eval mode; config rebuilt from the payload; missing M4 keys default to 6 / `identity` / False)
  - `scripts/train_world_model.py --data D --out O --arm A --seed S --n-actions N --pad-short --reward-transform symlog --record PATH` (the record is JSON with `git_sha`, `seconds`, `kl_rate_above_free_bits`, `kl_dyn_max`, `steps`, the three M4 fields)

- [ ] **Step 1: Write the failing RSSM test** (append to `tests/models/test_rssm.py`)

```python
def test_kl_loss_mask_all_true_matches_unmasked():
    post, prior = torch.randn(B, T, 32, 32), torch.randn(B, T, 32, 32)
    unmasked, _ = kl_loss(post, prior)
    masked, _ = kl_loss(post, prior, mask=torch.ones(B, T, dtype=torch.bool))
    torch.testing.assert_close(masked, unmasked)


def test_kl_loss_mask_ignores_padded_steps():
    post, prior = torch.randn(B, T, 32, 32), torch.randn(B, T, 32, 32)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, :3] = True
    garbage_post = post.clone()
    garbage_post[:, 3:] = 1e3 * torch.randn(B, T - 3, 32, 32)
    a, _ = kl_loss(post, prior, mask=mask)
    b, _ = kl_loss(garbage_post, prior, mask=mask)
    torch.testing.assert_close(a, b)
```

- [ ] **Step 2: Write the failing world-model tests** (append to `tests/training/test_world_model.py`)

```python
from mbfps.data.loader import SequenceLoader
from mbfps.models.transforms import symlog
from mbfps.training.world_model import load_world_model
from mbfps.utils.device import to_device


def _m4_cfg(**overrides):
    base = dict(steps=2, seq_len=16, batch_size=4, device="cpu", n_actions=6,
                pad_short=True, reward_transform="symlog")
    base.update(overrides)
    return get_config("cnn", **base)


def test_padded_loss_equals_unpadded_loss(tmp_path):
    """A 6-step episode padded to 16 must score exactly what the same 6 steps
    score unpadded: the pad contributes nothing to any term."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=6, fill=1))
    seed_everything(0)
    model = WorldModel(_m4_cfg())
    padded = SequenceLoader(buf, batch_size=1, seq_len=16, seed=0, pad_short=True).sample()
    exact = SequenceLoader(buf, batch_size=1, seq_len=6, seed=0, pad_short=True).sample()
    assert padded["mask"].sum() == 6 and exact["mask"].all()
    torch.manual_seed(1)
    loss_padded, parts_padded = model(to_device(padded, torch.device("cpu")))
    torch.manual_seed(1)
    loss_exact, parts_exact = model(to_device(exact, torch.device("cpu")))
    torch.testing.assert_close(loss_padded, loss_exact, rtol=1e-5, atol=1e-6)
    for key in ("embedding", "reward", "continue"):
        assert parts_padded[key] == pytest.approx(parts_exact[key], rel=1e-5, abs=1e-7)


def test_padded_positions_get_no_gradient(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=6, fill=1))
    model = WorldModel(_m4_cfg())
    batch = to_device(
        SequenceLoader(buf, batch_size=1, seq_len=16, seed=0, pad_short=True).sample(),
        torch.device("cpu"),
    )
    torch.manual_seed(2)
    loss_clean, parts_clean = model(batch)
    batch["rewards"][0, 6:] = 1e6          # garbage in the pad
    batch["terminated"][0, 6:] = True
    torch.manual_seed(2)
    loss_dirty, parts_dirty = model(batch)
    torch.testing.assert_close(loss_dirty, loss_clean)
    assert parts_dirty["reward"] == parts_clean["reward"]
    assert parts_dirty["continue"] == parts_clean["continue"]


def test_reward_head_trains_on_symlog(tmp_path):
    """With reward_transform='symlog' the reward term is the MSE against
    symlog(r), not r: a head output of symlog(r) scores zero."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    ep = make_episode(t=20, fill=1)
    ep.rewards[:] = 97.0
    buf.add(ep)
    model = WorldModel(_m4_cfg())
    with torch.no_grad():
        model.heads.reward[-1].weight.zero_()
        model.heads.reward[-1].bias.fill_(float(symlog(torch.tensor(97.0))))
    batch = to_device(SequenceLoader(buf, 1, 16, seed=0, pad_short=True).sample(),
                      torch.device("cpu"))
    _, parts = model(batch)
    assert parts["reward"] == pytest.approx(0.0, abs=1e-10)


def test_n_actions_sizes_the_rssm():
    assert WorldModel(_m4_cfg(n_actions=4)).rssm.cfg.n_actions == 4


def test_m4_payload_records_its_fields(tmp_path, buffer):
    cfg = _m4_cfg(n_actions=6)
    train_world_model(cfg, buffer, out_dir=tmp_path / "run")
    payload = torch.load(tmp_path / "run" / "world_model_cnn_seed0.pt", weights_only=True)
    assert payload["n_actions"] == 6
    assert payload["reward_transform"] == "symlog"
    assert payload["pad_short"] is True


def test_load_world_model_round_trips(tmp_path, buffer):
    cfg = _m4_cfg()
    train_world_model(cfg, buffer, out_dir=tmp_path / "run")
    model = load_world_model(tmp_path / "run" / "world_model_cnn_seed0.pt", torch.device("cpu"))
    assert not model.training
    assert model.cfg.train.reward_transform == "symlog"
    assert model.cfg.train.pad_short is True


def test_m3_payload_loads_with_defaults(tmp_path, buffer):
    """An M3 checkpoint has none of the M4 keys: it must load as 6 actions,
    identity, unpadded -- the configuration it was trained with (spec §4.6)."""
    cfg = get_config("cnn", steps=2, seq_len=16, batch_size=4, device="cpu")
    train_world_model(cfg, buffer, out_dir=tmp_path / "run")
    path = tmp_path / "run" / "world_model_cnn_seed0.pt"
    payload = torch.load(path, weights_only=True)
    for key in ("n_actions", "reward_transform", "pad_short"):
        payload.pop(key)
    torch.save(payload, path)
    model = load_world_model(path, torch.device("cpu"))
    assert (model.cfg.train.n_actions, model.cfg.train.reward_transform,
            model.cfg.train.pad_short) == (6, "identity", False)
```

- [ ] **Step 3: Run to see them fail**

Run: `.venv/bin/python -m pytest tests/models/test_rssm.py tests/training/test_world_model.py -q -k "mask or padded or symlog or n_actions or payload or load_world_model"`
Expected: FAIL — `unexpected keyword argument 'mask'`, `cannot import name 'load_world_model'`.

- [ ] **Step 4: Implement the masked KL** (`src/mbfps/models/rssm.py`)

```python
def _categorical_kl(
    logits_q: torch.Tensor, logits_p: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """KL(q || p) summed over the 32 categorical groups, averaged over B and T.

    With `mask` (M4), the average is over the masked-in (B, T) positions only,
    so padding contributes nothing. With every position masked in this is the
    unmasked mean up to float rounding; with `mask=None` it is that mean
    exactly, which is what every M3 caller gets."""
    log_q = F.log_softmax(logits_q, dim=-1)
    log_p = F.log_softmax(logits_p, dim=-1)
    per_step = (log_q.exp() * (log_q - log_p)).sum(-1).sum(-1)
    if mask is None:
        return per_step.mean()
    weight = mask.to(per_step.dtype)
    return (per_step * weight).sum() / weight.sum()
```

Add `mask: torch.Tensor | None = None` as `kl_loss`'s last parameter and pass it to both `_categorical_kl` calls. The free-bits clamp still applies to the (masked) batch-and-time mean, so the `KL_FREE_BITS` calibration semantics are unchanged.

- [ ] **Step 5: Implement the world-model changes** (`src/mbfps/training/world_model.py`)

Imports:

```python
from mbfps.models.transforms import reward_target
```

In `WorldModel.__init__`, pass the action count:

```python
            RSSMConfig(
                embed_dim=cfg.encoder.embed_dim,
                n_actions=cfg.train.n_actions,
                sample_temperature=cfg.train.sample_temperature,
            ),
```

Replace the loss block in `forward`:

```python
        mask = batch.get("mask")
        target_reward = reward_target(batch["rewards"], self.cfg.train.reward_transform)
        target_continue = continue_target(batch["terminated"], batch["truncated"])
        if mask is None:
            embedding_loss = F.mse_loss(predictions["embedding"], embeddings.detach())
            reward_loss = F.mse_loss(predictions["reward"], target_reward)
            continue_loss = F.binary_cross_entropy_with_logits(
                predictions["continue_logit"], target_continue
            )
        else:
            # M4 §4.2: every term is a mean over REAL transitions only.
            weight = mask.to(embeddings.dtype)
            count = weight.sum()
            embedding_loss = (
                (predictions["embedding"] - embeddings.detach()).pow(2).mean(-1) * weight
            ).sum() / count
            reward_loss = ((predictions["reward"] - target_reward).pow(2) * weight).sum() / count
            continue_loss = (
                F.binary_cross_entropy_with_logits(
                    predictions["continue_logit"], target_continue, reduction="none"
                )
                * weight
            ).sum() / count
        kl, kl_parts = kl_loss(out["post_logits"], out["prior_logits"], mask=mask)
```

The `mask is None` branch is the M3 code verbatim, so M3's loss is bitwise unchanged.

In `_save_checkpoint`, add to `payload`:

```python
        "n_actions": int(cfg.train.n_actions),
        "reward_transform": str(cfg.train.reward_transform),
        "pad_short": bool(cfg.train.pad_short),
```

In `train_world_model`, pass `pad_short=cfg.train.pad_short` to `SequenceLoader(...)`.

Add the loader function after `_save_checkpoint`:

```python
def load_world_model(path: Path, device: torch.device) -> "WorldModel":
    """Rebuild a world model from its checkpoint, in eval mode.

    The configuration comes from the PAYLOAD, never from a caller's
    assumption: an M3 checkpoint has no `n_actions`, `reward_transform` or
    `pad_short`, and loads as 6 actions, `identity`, unpadded -- what it was
    trained with. That is what lets M4 §4.6's control read an M3 model through
    M4 code."""
    from mbfps.utils.config import get_config

    payload = torch.load(path, map_location=device, weights_only=True)
    cfg = get_config(
        payload["arm"],
        seed=int(payload["seed"]),
        device=device.type,
        sample_temperature=float(payload.get("sample_temperature", 1.0)),
        n_actions=int(payload.get("n_actions", 6)),
        reward_transform=str(payload.get("reward_transform", "identity")),
        pad_short=bool(payload.get("pad_short", False)),
    )
    model = WorldModel(cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
```

- [ ] **Step 6: Update the two tests that pin the payload key set**

`tests/training/test_world_model.py:754` and `:768` assert the exact M3 key set. Add the three keys to both expected sets:

```python
    assert set(payload) == {"arm", "seed", "state_dict", "sample_temperature",
                            "n_actions", "reward_transform", "pad_short"}
```

and the same plus `"step"` at `:768`. Change nothing else in those tests: the rung layout is unchanged.

- [ ] **Step 7: Add the script flags** (`scripts/train_world_model.py`)

```python
    parser.add_argument("--n-actions", type=int, default=6)
    parser.add_argument("--pad-short", action="store_true")
    parser.add_argument("--reward-transform", choices=("identity", "symlog"), default="identity")
    parser.add_argument("--record", type=Path, default=None,
                        help="write a JSON summary of the run here (M4)")
```

Pass `n_actions=args.n_actions, pad_short=args.pad_short, reward_transform=args.reward_transform` into `get_config`. After the prints:

```python
    if args.record is not None:
        from mbfps.eval.study import git_sha, write_record

        write_record(args.record, {
            "arm": cfg.arm, "seed": cfg.train.seed, "steps": history["steps"],
            "seconds": history["seconds"], "data": str(args.data),
            "kl_dyn_max": history["kl_dyn_max"],
            "kl_rate_above_free_bits": history["kl_rate_above_free_bits"],
            "n_actions": cfg.train.n_actions, "pad_short": cfg.train.pad_short,
            "reward_transform": cfg.train.reward_transform, "git_sha": git_sha(),
        })
```

- [ ] **Step 8: Run the world-model, RSSM and full eval suites**

Run: `.venv/bin/python -m pytest tests/models tests/training -q`
Expected: all PASS.

Run: `.venv/bin/python -m pytest tests/eval -q -x`
Expected: all PASS — M3's diagnostics still load and score the shipped-format checkpoints.

- [ ] **Step 9: Commit**

```bash
git add src/mbfps/models/rssm.py src/mbfps/training/world_model.py scripts/train_world_model.py \
  tests/models/test_rssm.py tests/training/test_world_model.py
git commit -F - <<'EOF'
feat: masked world-model losses, a symlog reward head, and checkpoints that say so

Without a mask every loss is M3's code verbatim. load_world_model reads
the configuration from the payload, so an M3 checkpoint loads as the
6-action identity model it is.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

## Task 6: The entry check — arrays, baselines, readings, verdict

**Files:**
- Create: `src/mbfps/eval/entry.py`
- Test: `tests/eval/test_entry.py`

**Interfaces:**
- Consumes: Task 2 `read_reward`; Task 5 `WorldModel` (`.embed`-style encoding via `.encoder`, `.rssm.observe`, `.rssm.imagine`, `.heads`, `.cfg.train.reward_transform`); `mbfps.eval.pooling.episode_bootstrap`, `percentile_interval`; `mbfps.models.heads.continue_target`.
- Produces:
  - `HORIZON = 15`, `RESAMPLES = 2000`
  - `EpisodeArrays` dataclass: `imagined_reward, imagined_continue, posterior_reward, posterior_continue, real_reward, real_continue, valid` — all `np.ndarray (N, HORIZON)` float64/bool, `N = T - 1` starts.
  - `real_arrays(rewards, terminated, truncated, horizon) -> (real_reward, real_continue, valid)`
  - `episode_arrays(model, source, actions, rewards, terminated, truncated, device, seed, horizon=HORIZON) -> EpisodeArrays`
  - `constant_baselines(episodes: list[tuple[rewards, terminated, truncated]], horizon=HORIZON) -> tuple[float, float]` returns `(c_R, c_C)`
  - `return_reading(pred, real, valid, labels, c_R, resamples, seed) -> dict` and `continue_reading(p, c, valid, labels, c_C, resamples, seed) -> dict`; each dict is `{"point", "ci_low", "ci_high", "n_rows", "n_episodes"}`
  - `entry_verdict(imagined: dict, posterior: dict) -> str` in `{"ENTRY_PASS", "HEAD_OR_DATA", "IMAGINATION"}`
  - `control_broken(control_return: dict) -> bool`

Slot convention (spec §4.5): posterior state `s_i` is the state after action `i` (it has seen `obs[i+1]`). Start `i` runs over `0 .. T-2`; imagined step `j` predicts `rewards[i+1+j]` and `continue_target[i+1+j]`; `valid[i, j]` is `i + 1 + j < T`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
import torch

from mbfps.eval.entry import (
    HORIZON, constant_baselines, continue_reading, control_broken, entry_verdict,
    real_arrays, return_reading,
)


def _episode(rewards, terminated):
    r = np.asarray(rewards, dtype=np.float64)
    t = np.asarray(terminated, dtype=bool)
    return r, t, np.zeros_like(t)


def test_real_arrays_slot_alignment():
    r, t, tr = _episode([10, 11, 12, 13, 14], [0, 0, 0, 0, 1])
    real_r, real_c, valid = real_arrays(r, t, tr, horizon=3)
    assert real_r.shape == (4, 3)
    assert real_r[0].tolist() == [11, 12, 13]
    assert real_r[2].tolist()[:2] == [13, 14]
    assert valid[2].tolist() == [True, True, False]
    assert valid[3].tolist() == [True, False, False]
    assert real_c[3, 0] == 0.0          # the terminating transition
    assert real_c[0].tolist() == [1.0, 1.0, 1.0]


def test_kill_then_terminal_on_separate_steps():
    """basic: +97 arrives with terminated=False, then -1 with terminated=True
    on the next step (19 of 109 measured kills). Each lands on its own slot."""
    r, t, tr = _episode([-4, -4, 97, -1], [0, 0, 0, 1])
    real_r, real_c, valid = real_arrays(r, t, tr, horizon=15)
    assert real_r[0, :3].tolist() == [-4, 97, -1]
    assert real_c[0, :3].tolist() == [1.0, 1.0, 0.0]
    assert valid[0].sum() == 3


def test_truncation_is_not_termination():
    r, t, _ = _episode([-4, -4, -4], [0, 0, 0])
    tr = np.array([False, False, True])
    _, real_c, _ = real_arrays(r, t, tr, horizon=3)
    assert real_c[0].tolist()[:2] == [1.0, 1.0]


def test_constant_baselines_are_means_over_valid_slots():
    eps = [_episode([1, 2, 3], [0, 0, 1])]
    c_r, c_c = constant_baselines(eps, horizon=15)
    # starts 0 and 1: returns 2+3=5 and 3 -> mean 4
    assert c_r == pytest.approx(4.0)
    # valid slots: (0,0)=1,(0,1)=0,(1,0)=0 -> mean 1/3
    assert c_c == pytest.approx(1 / 3)


def _two_episode_rows(n=40, seed=0):
    rng = np.random.default_rng(seed)
    real = rng.normal(0, 10, size=(n, HORIZON))
    valid = np.ones((n, HORIZON), dtype=bool)
    labels = np.repeat([0, 1, 2, 3], n // 4)
    return real, valid, labels


def test_return_reading_true_rewards_score_one():
    real, valid, labels = _two_episode_rows()
    out = return_reading(real, real, valid, labels, c_R=0.0, resamples=200, seed=0)
    assert out["point"] == pytest.approx(1.0)
    assert out["ci_low"] > 0


def test_return_reading_constant_scores_at_most_zero():
    real, valid, labels = _two_episode_rows()
    c_r = float(real.sum(1).mean())
    pred = np.zeros_like(real)
    pred[:, 0] = c_r
    out = return_reading(pred, real, valid, labels, c_R=c_r, resamples=200, seed=0)
    assert out["point"] == pytest.approx(0.0, abs=1e-12)
    assert not out["ci_low"] > 0


def test_continue_reading_perfect_and_constant():
    rng = np.random.default_rng(1)
    c = (rng.random((40, HORIZON)) > 0.1).astype(float)
    valid = np.ones_like(c, dtype=bool)
    labels = np.repeat([0, 1, 2, 3], 10)
    rate = float(c.mean())
    perfect = continue_reading(c, c, valid, labels, c_C=rate, resamples=200, seed=0)
    constant = continue_reading(np.full_like(c, rate), c, valid, labels, c_C=rate,
                                resamples=200, seed=0)
    assert perfect["point"] == pytest.approx(1.0)
    assert constant["point"] == pytest.approx(0.0, abs=1e-12)
    assert perfect["ci_low"] > 0 and not constant["ci_low"] > 0


def test_invalid_slots_are_ignored():
    real, valid, labels = _two_episode_rows()
    valid[:, 10:] = False
    garbage = real.copy()
    garbage[:, 10:] = 1e9
    out = return_reading(garbage, real, valid, labels, c_R=0.0, resamples=50, seed=0)
    assert out["point"] == pytest.approx(1.0)


def _iv(low):
    return {"point": low, "ci_low": low, "ci_high": low + 1}


def test_verdicts():
    good, bad = _iv(0.1), _iv(-0.1)
    assert entry_verdict({"return": good, "continue": good},
                         {"return": good, "continue": good}) == "ENTRY_PASS"
    assert entry_verdict({"return": bad, "continue": good},
                         {"return": bad, "continue": good}) == "HEAD_OR_DATA"
    assert entry_verdict({"return": bad, "continue": good},
                         {"return": good, "continue": good}) == "IMAGINATION"
    nan = {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    assert entry_verdict({"return": nan, "continue": good},
                         {"return": good, "continue": good}) == "IMAGINATION"


def test_control_broken_compares_with_greater_than():
    assert control_broken(_iv(0.01))
    assert not control_broken(_iv(0.0))
    assert not control_broken({"ci_low": float("nan")})
```

Add a model-level alignment test that runs a real (tiny, untrained) world model end to end:

```python
def test_episode_arrays_on_a_tiny_model():
    from mbfps.eval.entry import episode_arrays
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    cfg = get_config("cnn", device="cpu", n_actions=4, reward_transform="symlog")
    model = WorldModel(cfg).eval()
    t = 8
    obs = np.zeros((t + 1, 112, 112, 3), dtype=np.uint8)
    r, term, trunc = _episode([-4] * 6 + [97, -1], [0] * 7 + [1])
    actions = np.arange(t) % 4
    a = episode_arrays(model, obs, actions, r, term, trunc, torch.device("cpu"), seed=0)
    assert a.imagined_reward.shape == (t - 1, HORIZON)
    # transition 6 is the kill (+97, continues); transition 7 terminates (-1).
    # Start i reads transition i + 1 + j at slot j.
    assert a.real_reward[0, 5] == 97 and a.real_continue[0, 5] == 1.0
    assert a.real_reward[0, 6] == -1 and a.real_continue[0, 6] == 0.0
    assert a.real_continue[1, 5] == 0.0
    assert a.posterior_reward.shape == a.imagined_reward.shape
    assert a.valid.sum() == sum(min(HORIZON, t - 1 - i) for i in range(t - 1))
    b = episode_arrays(model, obs, actions, r, term, trunc, torch.device("cpu"), seed=0)
    assert np.array_equal(a.imagined_reward, b.imagined_reward)  # seeded
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/bin/python -m pytest tests/eval/test_entry.py -q`
Expected: FAIL — `ModuleNotFoundError: mbfps.eval.entry`.

- [ ] **Step 3: Implement `src/mbfps/eval/entry.py`**

```python
"""Stage A's entry check (M4 spec §4.5): does imagination predict reward and
episode end over 15 steps?

The start states are the ones the actor-critic will use: EVERY posterior state
of an episode. From state `s_i` (after action `i`, having seen `obs[i+1]`) the
model imagines up to 15 steps driven by the RECORDED actions
`actions[i+1 : i+1+H_i]`; imagined step `j` predicts `rewards[i+1+j]` and the
continue target of that same transition. All starts of an episode go through
ONE `imagine` call of HORIZON steps with the actions zero-padded past the
episode's end; the steps past the end are dropped by `valid`, and imagining
them cannot change an earlier step.

Two readings, each an episode-clustered bootstrap interval:
  return    R^2 of the imagined H_i-step return against the real one, against
            the best constant known from TRAINING data (c_R).
  continue  Brier skill of the continue head against the training-set
            continue rate (c_C).
The same readings on the POSTERIOR states (teacher-forced) attribute a failure:
the head/data, or imagination. They are reported, not gated.
"""

from dataclasses import dataclass

import numpy as np
import torch

from mbfps.eval.pooling import episode_bootstrap, percentile_interval
from mbfps.models.heads import continue_target
from mbfps.models.transforms import read_reward

HORIZON: int = 15
RESAMPLES: int = 2000
VERDICTS: tuple[str, ...] = ("ENTRY_PASS", "HEAD_OR_DATA", "IMAGINATION")


@dataclass
class EpisodeArrays:
    imagined_reward: np.ndarray
    imagined_continue: np.ndarray
    posterior_reward: np.ndarray
    posterior_continue: np.ndarray
    real_reward: np.ndarray
    real_continue: np.ndarray
    valid: np.ndarray


def real_arrays(rewards, terminated, truncated, horizon: int = HORIZON):
    """`(real_reward, real_continue, valid)`, each `(T - 1, horizon)`: slot
    `(i, j)` is transition `i + 1 + j`, valid while that index is `< T`."""
    rewards = np.asarray(rewards, dtype=np.float64)
    cont = continue_target(
        torch.as_tensor(np.asarray(terminated)), torch.as_tensor(np.asarray(truncated))
    ).numpy().astype(np.float64)
    t = rewards.shape[0]
    index = np.arange(t - 1)[:, None] + 1 + np.arange(horizon)[None, :]
    valid = index < t
    safe = np.where(valid, index, 0)
    return (
        np.where(valid, rewards[safe], 0.0),
        np.where(valid, cont[safe], 0.0),
        valid,
    )


@torch.no_grad()
def episode_arrays(
    model, source, actions, rewards, terminated, truncated, device, seed: int,
    horizon: int = HORIZON,
) -> EpisodeArrays:
    """Imagined, posterior and real arrays for one episode.

    `source` is the episode's `(T+1, ...)` model input -- pixels for the `cnn`
    kind, the arm's cached feature rows otherwise (`rollout.source_for`).
    Seeded with `seed` immediately before the filter, so an episode's arrays do
    not depend on which episodes were read before it."""
    t = len(actions)
    torch.manual_seed(int(seed))
    src = torch.as_tensor(np.asarray(source)).to(device)
    embeddings = model.encoder(src).unsqueeze(0)[:, 1:]          # (1, T, E)
    acts = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=device)
    post = model.rssm.observe(embeddings, acts.unsqueeze(0))
    h, z = post["h"][0, : t - 1], post["z"][0, : t - 1]          # starts 0..T-2

    index = torch.arange(t - 1, device=device)[:, None] + 1 + torch.arange(horizon, device=device)
    valid_t = index < t
    plan = torch.where(valid_t, acts[torch.where(valid_t, index, 0)], torch.zeros_like(index))
    imagined = model.rssm.imagine(plan, (h, z))
    transform = model.cfg.train.reward_transform
    heads_im = model.heads(imagined["latent"])
    heads_post = model.heads(post["latent"][0])                   # (T, D) -> per transition
    post_index = torch.where(valid_t, index, 0)

    real_r, real_c, valid = real_arrays(rewards, terminated, truncated, horizon)
    as_np = lambda x: x.detach().double().cpu().numpy()
    return EpisodeArrays(
        imagined_reward=as_np(read_reward(heads_im["reward"], transform)),
        imagined_continue=as_np(torch.sigmoid(heads_im["continue_logit"])),
        posterior_reward=as_np(read_reward(heads_post["reward"][post_index], transform)),
        posterior_continue=as_np(torch.sigmoid(heads_post["continue_logit"][post_index])),
        real_reward=real_r,
        real_continue=real_c,
        valid=valid,
    )


def constant_baselines(episodes, horizon: int = HORIZON) -> tuple[float, float]:
    """`(c_R, c_C)` from TRAINING episodes' real data only, weighted exactly as
    the readings weight validation rows: one return per start, one continue
    target per valid slot."""
    returns, conts = [], []
    for rewards, terminated, truncated in episodes:
        real_r, real_c, valid = real_arrays(rewards, terminated, truncated, horizon)
        returns.append((real_r * valid).sum(1))
        conts.append(real_c[valid])
    return float(np.concatenate(returns).mean()), float(np.concatenate(conts).mean())


def _interval(stat, n_rows: int, labels: np.ndarray, resamples: int, seed: int) -> dict:
    point = stat(np.arange(n_rows))
    reps = np.array([stat(rows) for rows in episode_bootstrap(labels, resamples, seed)])
    low, high, _ = percentile_interval(reps)
    return {"point": float(point), "ci_low": low, "ci_high": high,
            "n_rows": int(n_rows), "n_episodes": int(np.unique(labels).size)}


def return_reading(pred, real, valid, labels, c_R: float, resamples: int, seed: int) -> dict:
    """R^2 of per-start returns. `labels` is one episode label per start."""
    pred_ret = (np.asarray(pred) * valid).sum(1)
    real_ret = (np.asarray(real) * valid).sum(1)

    def stat(rows):
        sse = np.sum((real_ret[rows] - pred_ret[rows]) ** 2)
        sst = np.sum((real_ret[rows] - c_R) ** 2)
        return 1.0 - sse / sst if sst > 0 else float("nan")

    return _interval(stat, real_ret.size, np.asarray(labels), resamples, seed)


def continue_reading(p, c, valid, labels, c_C: float, resamples: int, seed: int) -> dict:
    """Brier skill over valid slots. `labels` is one episode label per START;
    it is expanded to the slots here."""
    valid = np.asarray(valid, dtype=bool)
    slot_labels = np.repeat(np.asarray(labels), valid.sum(1))
    p_rows, c_rows = np.asarray(p)[valid], np.asarray(c)[valid]

    def stat(rows):
        model = np.mean((p_rows[rows] - c_rows[rows]) ** 2)
        const = np.mean((c_C - c_rows[rows]) ** 2)
        return 1.0 - model / const if const > 0 else float("nan")

    return _interval(stat, c_rows.size, slot_labels, resamples, seed)


def _passes(interval: dict) -> bool:
    return bool(interval.get("ci_low", float("nan")) > 0.0)


def entry_verdict(imagined: dict, posterior: dict) -> str:
    """§4.7. A reading whose interval is NaN fails, by the `>`."""
    failing = [k for k in ("return", "continue") if not _passes(imagined[k])]
    if not failing:
        return "ENTRY_PASS"
    if any(not _passes(posterior[k]) for k in failing):
        return "HEAD_OR_DATA"
    return "IMAGINATION"


def control_broken(control_return: dict) -> bool:
    """§4.6: the M3 model's return reading passing means the check is broken."""
    return _passes(control_return)
```

`model.heads(post["latent"][0])` returns `(T,)` tensors; indexing them with the `(T-1, horizon)` `post_index` gives the slot layout.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/eval/test_entry.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/eval/entry.py tests/eval/test_entry.py
git commit -F - <<'EOF'
feat: Stage A's entry readings -- imagined 15-step return and continue, by episode

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

## Task 7: `scripts/entry_check.py` — measure, control, read

**Files:**
- Create: `scripts/entry_check.py`
- Test: `tests/eval/test_entry_check_script.py`

**Interfaces:**
- Consumes: Task 5 `load_world_model`; Task 6 everything; `mbfps.data.split.episode_split`, `VAL_FRACTION`; `mbfps.eval.study.write_record`, `load_record`, `git_sha`; `mbfps.eval.rollout.source_for`; `mbfps.models.encoders.encoder_backbone`; `mbfps.eval.probe.fit_probes`, `mbfps.eval.rollout.evaluate_rollout`.
- Produces:
  - `python scripts/entry_check.py --phase measure --checkpoint CKPT --data DIR --out OUT [--role cell|control] [--resamples 2000] [--no-position] [--device mps]` → writes `OUT/entry_<arm>_seed<n>.json` (role cell) or `OUT/entry_control.json` (role control)
  - `python scripts/entry_check.py --phase read --out OUT --arm A --seed S` → prints `OUT/entry.txt`, returns `EXIT_OK=0` (ENTRY_PASS), `EXIT_HEAD_OR_DATA=60`, `EXIT_IMAGINATION=61`, `EXIT_CONTROL_BROKEN=62`, `EXIT_MISSING=11`
  - Record fields: `arm, seed, role, checkpoint, data, reward_transform, n_actions, c_R, c_C, imagined{return,continue}, posterior{return,continue}, counts{episodes, starts, slots, kills, terminations}, position_gap_closed_h15 (float or null), verdict (role cell only), git_sha, seconds`.

- [ ] **Step 1: Write the failing test**

```python
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.envs.protocol import OBS_SHAPE
from mbfps.eval.study import load_record
from mbfps.training.world_model import train_world_model
from mbfps.utils.config import get_config

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "entry_check.py"
KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def _basic_like(t, fill, kill_at=None):
    rewards = np.full(t, -4.0, dtype=np.float32)
    terminated = np.zeros(t, dtype=bool)
    if kill_at is not None:
        rewards[kill_at] = 97.0
        terminated[kill_at] = True
    obs = np.zeros((t + 1, *OBS_SHAPE), dtype=np.uint8)
    obs[:, 0, 0, 1] = fill
    return Episode(
        obs=obs, actions=(np.arange(t) % 4).astype(np.int32), rewards=rewards,
        terminated=terminated, truncated=np.zeros(t, dtype=bool),
        privileged=np.zeros((t + 1, len(KEYS)), dtype=np.float32),
        privileged_keys=KEYS, policy_name="random", seed=fill, scenario="basic",
    )


@pytest.fixture
def trained(tmp_path):
    buf = ReplayBuffer(tmp_path / "data", capacity_transitions=100_000)
    for fill in range(1, 11):
        t = 30 if fill % 2 else 12
        buf.add(_basic_like(t, fill, kill_at=t - 1 if fill % 3 == 0 else None))
    cfg = get_config("cnn", steps=3, seq_len=16, batch_size=4, device="cpu",
                     n_actions=4, pad_short=True, reward_transform="symlog")
    train_world_model(cfg, buf, out_dir=tmp_path / "wm")
    return tmp_path, tmp_path / "wm" / "world_model_cnn_seed0.pt"


def _run(*args):
    proc = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def test_measure_writes_a_record_and_read_maps_the_verdict(trained):
    root, ckpt = trained
    out = root / "entry"
    code, log = _run("--phase", "measure", "--checkpoint", ckpt, "--data", root / "data",
                     "--out", out, "--resamples", 50, "--no-position", "--device", "cpu")
    assert code == 0, log
    record = load_record(out / "entry_cnn_seed0.json")
    for key in ("c_R", "c_C", "imagined", "posterior", "counts", "verdict", "git_sha"):
        assert key in record
    assert record["reward_transform"] == "symlog" and record["n_actions"] == 4
    assert record["counts"]["episodes"] >= 1

    # The read refuses without a control record.
    code, log = _run("--phase", "read", "--out", out, "--arm", "cnn", "--seed", 0)
    assert code == 11, log

    code, log = _run("--phase", "measure", "--checkpoint", ckpt, "--data", root / "data",
                     "--out", out, "--role", "control", "--resamples", 50,
                     "--no-position", "--device", "cpu")
    assert code == 0, log
    code, log = _run("--phase", "read", "--out", out, "--arm", "cnn", "--seed", 0)
    control = load_record(out / "entry_control.json")
    if control["imagined"]["return"]["ci_low"] > 0:   # NaN compares False, as the script does
        assert code == 62
    else:
        assert code in (0, 60, 61)
    assert (out / "entry.txt").is_file()
```

- [ ] **Step 2: Run to see it fail**

Run: `.venv/bin/python -m pytest tests/eval/test_entry_check_script.py -q`
Expected: FAIL — the script does not exist (non-zero exit, "No such file").

- [ ] **Step 3: Implement `scripts/entry_check.py`**

```python
"""Stage A's entry check (M4 spec §4.5-§4.7).

    measure --role cell     one world model over its data's held-out split
    measure --role control  §4.6: the M3 my_way_home model; must FAIL Reading R
    read                    the verdict, refusing without the control

Exit codes (read):
  0   ENTRY_PASS        both readings' intervals lie above zero
  60  HEAD_OR_DATA      a reading fails and its posterior reference fails too
  61  IMAGINATION       a reading fails and its posterior reference passes
  62  CONTROL_BROKEN    the control's return reading passed: nothing is read
  11  MISSING           a record the read needs is absent
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import load_episode
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.entry import (
    HORIZON, RESAMPLES, constant_baselines, continue_reading, control_broken,
    entry_verdict, episode_arrays, return_reading,
)
from mbfps.eval.rollout import source_for
from mbfps.eval.study import git_sha, load_record, write_record
from mbfps.models.encoders import encoder_backbone
from mbfps.training.world_model import load_world_model
from mbfps.utils.device import get_device
from mbfps.utils.seeding import fork_seed

EXIT_OK = 0
EXIT_MISSING = 11
EXIT_HEAD_OR_DATA = 60
EXIT_IMAGINATION = 61
EXIT_CONTROL_BROKEN = 62
READ_EXITS = {"ENTRY_PASS": EXIT_OK, "HEAD_OR_DATA": EXIT_HEAD_OR_DATA,
              "IMAGINATION": EXIT_IMAGINATION}
SPLIT_SEED = 0
BOOTSTRAP_SEED = 0


def record_path(out: Path, role: str, arm: str, seed: int) -> Path:
    return out / ("entry_control.json" if role == "control" else f"entry_{arm}_seed{seed}.json")


def _readings(stack: dict, labels, c_r, c_c, resamples) -> tuple[dict, dict]:
    imagined = {
        "return": return_reading(stack["imagined_reward"], stack["real_reward"], stack["valid"],
                                 labels, c_r, resamples, BOOTSTRAP_SEED),
        "continue": continue_reading(stack["imagined_continue"], stack["real_continue"],
                                     stack["valid"], labels, c_c, resamples, BOOTSTRAP_SEED),
    }
    posterior = {
        "return": return_reading(stack["posterior_reward"], stack["real_reward"], stack["valid"],
                                 labels, c_r, resamples, BOOTSTRAP_SEED),
        "continue": continue_reading(stack["posterior_continue"], stack["real_continue"],
                                     stack["valid"], labels, c_c, resamples, BOOTSTRAP_SEED),
    }
    return imagined, posterior


def measure(args) -> int:
    start = time.perf_counter()
    device = get_device(prefer=args.device)
    model = load_world_model(args.checkpoint, device)
    arm, seed = model.cfg.arm, model.cfg.train.seed
    backbone = encoder_backbone(model.cfg.encoder) or "dinov2"
    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    train_paths, val_paths = episode_split(buffer.episode_paths(),
                                           val_fraction=VAL_FRACTION, seed=SPLIT_SEED)

    train_eps = []
    for path in train_paths:
        with np.load(path) as d:
            train_eps.append((d["rewards"], d["terminated"], d["truncated"]))
    c_r, c_c = constant_baselines(train_eps, HORIZON)

    fields = ("imagined_reward", "imagined_continue", "posterior_reward",
              "posterior_continue", "real_reward", "real_continue", "valid")
    rows = {k: [] for k in fields}
    labels, kills, terminations = [], 0, 0
    for label, path in enumerate(val_paths):
        ep = load_episode(path)
        if ep.length < 2:
            continue
        a = episode_arrays(
            model, source_for(model, path, ep, backbone), ep.actions, ep.rewards,
            ep.terminated, ep.truncated, device,
            seed=fork_seed(seed, f"entry:{path.name}"),
        )
        for k in fields:
            rows[k].append(getattr(a, k))
        labels.append(np.full(ep.length - 1, label))
        kills += int(np.sum(ep.rewards > 50.0))
        terminations += int(ep.terminated.any())
    stack = {k: np.concatenate(v) for k, v in rows.items()}
    labels = np.concatenate(labels)
    imagined, posterior = _readings(stack, labels, c_r, c_c, args.resamples)

    position = None
    if not args.no_position:
        from mbfps.eval.probe import fit_probes
        from mbfps.eval.rollout import evaluate_rollout

        _, embedding_probe = fit_probes(model, train_paths, backbone, device,
                                        context=5, horizon=HORIZON, seed=seed)
        rollout = evaluate_rollout(model, val_paths, embedding_probe, context=5,
                                   horizon=HORIZON, seed=seed, device=device,
                                   feature_backbone=backbone)
        position = float(rollout.position_gap_closed()[-1])

    record = {
        "arm": arm, "seed": seed, "role": args.role,
        "checkpoint": str(args.checkpoint), "data": str(args.data),
        "reward_transform": model.cfg.train.reward_transform,
        "n_actions": model.cfg.train.n_actions,
        "horizon": HORIZON, "resamples": args.resamples,
        "c_R": c_r, "c_C": c_c, "imagined": imagined, "posterior": posterior,
        "counts": {"episodes": int(np.unique(labels).size), "starts": int(labels.size),
                   "slots": int(stack["valid"].sum()), "kills": kills,
                   "terminations": terminations},
        "position_gap_closed_h15": position,
        "git_sha": git_sha(), "seconds": time.perf_counter() - start,
    }
    if args.role == "cell":
        record["verdict"] = entry_verdict(imagined, posterior)
    write_record(record_path(args.out, args.role, arm, seed), record)
    print(f"wrote {record_path(args.out, args.role, arm, seed)}")
    return EXIT_OK


def _line(name: str, iv: dict) -> str:
    return (f"  {name:<20s} {iv['point']:+.4f} [{iv['ci_low']:+.4f}, {iv['ci_high']:+.4f}]"
            f"  rows={iv['n_rows']} episodes={iv['n_episodes']}")


def read(args) -> int:
    cell_path = record_path(args.out, "cell", args.arm, args.seed)
    control_path = record_path(args.out, "control", args.arm, args.seed)
    missing = [p for p in (cell_path, control_path) if not p.is_file()]
    if missing:
        print(f"MISSING: {[str(p) for p in missing]}")
        return EXIT_MISSING
    cell, control = load_record(cell_path), load_record(control_path)
    broken = control_broken(control["imagined"]["return"])
    lines = [
        f"M4 Stage A entry check -- {args.arm} seed {args.seed}",
        f"horizon {cell['horizon']}, resamples {cell['resamples']}, "
        f"c_R {cell['c_R']:+.3f}, c_C {cell['c_C']:.4f}",
        f"counts {cell['counts']}",
        "imagined (GATED):",
        _line("return R^2", cell["imagined"]["return"]),
        _line("continue Brier skill", cell["imagined"]["continue"]),
        "posterior reference (reported):",
        _line("return R^2", cell["posterior"]["return"]),
        _line("continue Brier skill", cell["posterior"]["continue"]),
        f"position gap_closed at h=15 (reported): {cell['position_gap_closed_h15']}",
        "control (M3 my_way_home model; Reading R must FAIL):",
        _line("return R^2", control["imagined"]["return"]),
        _line("continue Brier skill", control["imagined"]["continue"]),
    ]
    verdict = "CONTROL_BROKEN" if broken else cell["verdict"]
    lines.append(f"VERDICT: {verdict}")
    text = "\n".join(lines) + "\n"
    (args.out / "entry.txt").write_text(text)
    print(text, end="")
    return EXIT_CONTROL_BROKEN if broken else READ_EXITS[cell["verdict"]]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("measure", "read"), required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--out", type=Path, default=Path("runs/m4a_entry"))
    p.add_argument("--role", choices=("cell", "control"), default="cell")
    p.add_argument("--resamples", type=int, default=RESAMPLES)
    p.add_argument("--no-position", action="store_true")
    p.add_argument("--device", default="mps")
    p.add_argument("--arm")
    p.add_argument("--seed", type=int)
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.phase == "measure":
        if args.checkpoint is None or args.data is None:
            raise SystemExit("--phase measure needs --checkpoint and --data")
        return measure(args)
    if args.arm is None or args.seed is None:
        raise SystemExit("--phase read needs --arm and --seed")
    return read(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test**

Run: `.venv/bin/python -m pytest tests/eval/test_entry_check_script.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/entry_check.py tests/eval/test_entry_check_script.py
git commit -F - <<'EOF'
feat: scripts/entry_check.py, which refuses to read without its known-negative control

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

## Task 8: Run Stage A and write its results

**Files:**
- Create: `data/basic/` (episodes and the dinov2 feature cache), `runs/m4a_world/`, `runs/m4a_entry/`
- Modify: this plan (append `## Stage A results`)

**Interfaces:**
- Consumes: Tasks 1–7.
- Produces: `runs/m4a_world/world_model_frozen_ssl_seed0.pt` (Stage B's world model), `runs/m4a_world/train_frozen_ssl_seed0.json`, `runs/m4a_entry/entry_frozen_ssl_seed0.json`, `runs/m4a_entry/entry_control.json`, `runs/m4a_entry/entry.txt`.

- [ ] **Step 1: Suite green at the commit being run**

```bash
.venv/bin/python -m pytest -q > "$TMPDIR/m4a_suite.log" 2>&1; echo "exit: $?"; tail -3 "$TMPDIR/m4a_suite.log"
git rev-parse --short HEAD
```

Record the counts and the SHA. A run whose suite was not green at its SHA is not reportable.

- [ ] **Step 2: Decide the collection mix (spec §11), before collecting**

```bash
.venv/bin/python - > runs_m4a_mix.log 2>&1 <<'EOF'
import numpy as np
from mbfps.envs.registry import make_env
from mbfps.data.policies import RandomPolicy, ScriptedPolicy
env = make_env("vizdoom", scenario="basic", frame_skip=4)
for name, pol in (("random", RandomPolicy(env.action_space.n)),
                  ("scripted_sweep10", ScriptedPolicy(env.button_names, sweep_len=10))):
    lens, kills = [], 0
    for ep in range(200):
        obs, _ = env.reset(seed=ep); pol.reset(seed=ep); n = 0
        while True:
            obs, r, term, trunc, _ = env.step(pol.act(obs)); n += 1
            if term or trunc: break
        lens.append(n); kills += int(term)
    print(f"{name}: kill_rate={kills/200:.3f} median_len={np.median(lens):.0f}")
EOF
echo "exit: $?"; cat runs_m4a_mix.log; mv runs_m4a_mix.log "$TMPDIR/"
```

**The pre-registered rule:** collect `--policies random,scripted` (M1's equal alternation) if `scripted_sweep10`'s kill rate is at least 0.10; otherwise collect `--policies random` alone. Record both kill rates and the choice in the results. The mix is not revisited after training.

- [ ] **Step 3: Collect**

Check disk first: the dinov2 cache for ~60,000 transitions is ~3 GB.

```bash
df -h . | tail -1      # need >= 5 GB free
.venv/bin/python scripts/collect.py --scenario basic --episodes 1200 --sweep-len 10 \
  --policies <CHOICE FROM STEP 2> --capacity 60000 --max-steps 1000 --out data/basic \
  > "$TMPDIR/m4a_collect.log" 2>&1
echo "exit: $?"; tail -5 "$TMPDIR/m4a_collect.log"
```

Expected: exit 0, `transitions` near 60,000 (the buffer evicts the oldest beyond capacity).

- [ ] **Step 4: Cache dinov2 features**

```bash
.venv/bin/python scripts/cache_features.py --data data/basic --backbone dinov2 --device mps \
  > "$TMPDIR/m4a_cache.log" 2>&1
echo "exit: $?"; tail -3 "$TMPDIR/m4a_cache.log"
```

- [ ] **Step 5: Train the world model**

```bash
mkdir -p runs/m4a_world
.venv/bin/python scripts/train_world_model.py --arm frozen_ssl --seed 0 --steps 20000 \
  --data data/basic --out runs/m4a_world --n-actions 4 --pad-short --reward-transform symlog \
  --record runs/m4a_world/train_frozen_ssl_seed0.json > runs/m4a_world/train.log 2>&1
echo "exit: $?"; tail -8 runs/m4a_world/train.log
```

Expected: exit 0; `kl_rate_above_free_bits` printed; about 90 minutes.

- [ ] **Step 6: Measure the cell and the control**

```bash
mkdir -p runs/m4a_entry
.venv/bin/python scripts/entry_check.py --phase measure --role cell \
  --checkpoint runs/m4a_world/world_model_frozen_ssl_seed0.pt --data data/basic \
  --out runs/m4a_entry > runs/m4a_entry/measure_cell.log 2>&1
echo "cell exit: $?"
.venv/bin/python scripts/entry_check.py --phase measure --role control \
  --checkpoint runs/m3_study_v2/world_model_frozen_ssl_seed0.pt --data data/my_way_home \
  --no-position --out runs/m4a_entry > runs/m4a_entry/measure_control.log 2>&1
echo "control exit: $?"
```

Both must exit 0. The control reads `data/my_way_home`'s existing dinov2 cache; confirm `data/my_way_home/*.features.npy` exist before running (`ls data/my_way_home/*.features.npy | wc -l` must print the episode count).

The cell's position reading (`fit_probes` + `evaluate_rollout` at h=15) is reported, not gated. If it raises on `basic` data, do not patch around it inside this task: rerun the cell measure with `--no-position`, record the traceback and the rerun in the results, and leave the fix to a follow-up. The gated readings do not depend on it.

- [ ] **Step 7: Read**

```bash
.venv/bin/python scripts/entry_check.py --phase read --out runs/m4a_entry --arm frozen_ssl --seed 0 \
  > runs/m4a_entry/read.log 2>&1
echo "read exit: $?"; cat runs/m4a_entry/read.log
```

Exit 0 is ENTRY_PASS; 60/61/62 are §4.7's other rows. Record the code and the whole table.

- [ ] **Step 8: Write the results and commit**

Append `## Stage A results` to this plan. State, each re-derivable from a named record:

- the suite counts and SHA; the collection mix decision with both kill rates; episode, transition, kill and termination counts in `data/basic` and in the held-out split
- the world model's `seconds`, `kl_rate_above_free_bits`, `kl_dyn_max`
- both gated readings with intervals; both posterior references; position `gap_closed` at h=15
- the control's two readings; whether it failed Reading R as required
- the verdict, the exit code, and the §4.7 row it licenses, in the spec's words

```bash
git add docs/superpowers/plans/2026-10-10-mb-fps-m4-actor-critic.md
git commit -F - <<'EOF'
results: Stage A -- <what the run found, no more>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

Replace the subject's placeholder with the finding before committing. **If the verdict is not ENTRY_PASS, stop here.** Stage B does not run (§4.7); report to the author.

---

## Task 9: `Actor`, `Critic` and `RSSM.imagine_policy`

**Files:**
- Modify: `src/mbfps/models/heads.py` (add `Actor`, `Critic`)
- Modify: `src/mbfps/models/rssm.py` (add `imagine_policy`)
- Test: `tests/models/test_heads.py`, `tests/models/test_rssm.py`

**Interfaces:**
- Produces:
  - `Actor(n_actions: int, hidden: int = 512, seed: int = 0)`; `.forward(latent) -> logits (N, n_actions)`; `.dist(latent) -> torch.distributions.Categorical`
  - `Critic(hidden: int = 512, seed: int = 0)`; `.forward(latent) -> (N,)`
  - `RSSM.imagine_policy(policy: Callable[[Tensor], Tensor], state: tuple[Tensor, Tensor], horizon: int) -> dict` with `"latent": (N, horizon+1, LATENT_DIM)` (index 0 is the start state) and `"action": (N, horizon)` long. `policy` maps a `(N, LATENT_DIM)` latent to `(N,)` action indices. Runs under `torch.no_grad()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/models/test_heads.py`:

```python
import torch

from mbfps.models.heads import Actor, Critic
from mbfps.models.rssm import LATENT_DIM


def test_actor_and_critic_shapes():
    latent = torch.randn(5, LATENT_DIM)
    actor, critic = Actor(n_actions=4), Critic()
    assert actor(latent).shape == (5, 4)
    assert actor.dist(latent).sample().shape == (5,)
    assert critic(latent).shape == (5,)


def test_actor_and_critic_are_seeded():
    a, b = Actor(4, seed=3), Actor(4, seed=3)
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.equal(p, q)
    assert not torch.equal(next(Actor(4, seed=4).parameters()), next(a.parameters()))
```

Append to `tests/models/test_rssm.py`:

```python
def test_imagine_policy_with_fixed_actions_matches_imagine(rssm):
    """A policy that replays a fixed action sequence consumes no RNG, so the
    latents must equal `imagine` on the same actions from the same seed."""
    n, horizon = 4, 6
    torch.manual_seed(0)
    start = (torch.randn(n, 512), torch.zeros(n, 1024))
    plan = torch.randint(0, 6, (n, horizon))
    steps = iter(range(horizon))
    policy = lambda latent: plan[:, next(steps)]
    torch.manual_seed(7)
    out = rssm.imagine_policy(policy, start, horizon)
    torch.manual_seed(7)
    ref = rssm.imagine(plan, start)
    assert out["latent"].shape == (n, horizon + 1, LATENT_DIM)
    torch.testing.assert_close(out["latent"][:, 1:], ref["latent"])
    torch.testing.assert_close(out["latent"][:, 0], torch.cat(start, dim=-1))
    assert torch.equal(out["action"], plan)


def test_imagine_policy_builds_no_graph(rssm):
    start = (torch.randn(2, 512), torch.zeros(2, 1024))
    out = rssm.imagine_policy(lambda latent: torch.zeros(2, dtype=torch.long), start, 3)
    assert not out["latent"].requires_grad
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/bin/python -m pytest tests/models/test_heads.py tests/models/test_rssm.py -q -k "actor or critic or imagine_policy"`
Expected: FAIL — `cannot import name 'Actor'`, `no attribute 'imagine_policy'`.

- [ ] **Step 3: Implement the heads** (append to `src/mbfps/models/heads.py`)

```python
class Actor(nn.Module):
    """Policy over the latent: logits of a categorical over the scenario's actions (M4)."""

    def __init__(self, n_actions: int, hidden: int = 512, seed: int = 0) -> None:
        super().__init__()
        with seeded_init(seed, "actor"):
            self.net = _mlp(LATENT_DIM, hidden, n_actions)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)

    def dist(self, latent: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.forward(latent))


class Critic(nn.Module):
    """Value of the latent (M4). The slow target is a deep copy kept by the trainer."""

    def __init__(self, hidden: int = 512, seed: int = 0) -> None:
        super().__init__()
        with seeded_init(seed, "critic"):
            self.net = _mlp(LATENT_DIM, hidden, 1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)
```

Update the module docstring's first paragraph to say the module also holds M4's actor and critic, as the governing spec §4 places them.

- [ ] **Step 4: Implement `imagine_policy`** (in `RSSM`, after `imagine`)

```python
    @torch.no_grad()
    def imagine_policy(self, policy, state, horizon: int) -> dict[str, torch.Tensor]:
        """Roll forward on the prior with `policy` choosing each action (M4).

        `policy` maps the current `(N, LATENT_DIM)` latent to `(N,)` action
        indices. Returns `latent` `(N, horizon + 1, LATENT_DIM)` -- index 0 the
        start state, index t the state after the t-th action -- and `action`
        `(N, horizon)`. No graph is built: the world model is frozen in M4, and
        the actor's log-probabilities are recomputed by the trainer on these
        detached latents. `imagine` is untouched: M3's diagnostics depend on it.
        """
        h, z = state
        latents, actions = [torch.cat([h, z], dim=-1)], []
        for _ in range(horizon):
            action = policy(latents[-1]).long()
            h = self._step(h, z, self._onehot_actions(action))
            z = self._sample(self.prior_net(h))
            latents.append(torch.cat([h, z], dim=-1))
            actions.append(action)
        return {"latent": torch.stack(latents, dim=1), "action": torch.stack(actions, dim=1)}
```

- [ ] **Step 5: Run the model suites**

Run: `.venv/bin/python -m pytest tests/models -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/models/heads.py src/mbfps/models/rssm.py tests/models/test_heads.py tests/models/test_rssm.py
git commit -F - <<'EOF'
feat: Actor, Critic and imagine_policy -- the prior rolled forward under a policy

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

## Task 10: The actor-critic trainer

**Files:**
- Create: `src/mbfps/training/ac_trainer.py`
- Create: `scripts/train_actor_critic.py`
- Test: `tests/training/test_ac_trainer.py`

**Interfaces:**
- Consumes: Task 4 loader (`pad_short`, `mask`); Task 5 `load_world_model`, `WorldModel.embed`; Task 2 `read_reward`; Task 9 `Actor`, `Critic`, `imagine_policy`.
- Produces:
  - `ACConfig` (frozen dataclass): `horizon=15, gamma=0.997, lam=0.95, actor_lr=3e-5, critic_lr=3e-5, entropy=3e-4, critic_ema=0.98, scale_decay=0.99, updates=10_000, checkpoint_every=2_000, batch_size=16, seq_len=64, seed=0, device="mps", grad_clip=100.0`
  - `lambda_returns(rewards (N,H), discounts (N,H), values (N,H+1), lam) -> (N,H)`
  - `ReturnScale(decay)`: `.update(returns) -> float` (the scale S after the update), `.low`, `.high`
  - `continue_weights(cont (N,H)) -> (N,H)` — `w_0 = 1`, `w_t = prod_{k<t} cont_k`
  - `start_states(world_model, batch) -> (h, z)` — only masked-in positions
  - `freeze(world_model) -> None`
  - `train_actor_critic(world_model, buffer, cfg: ACConfig, out_dir: Path | None, log_every=100) -> dict` history with `actor_loss, critic_loss, scale, imagined_return, entropy` lists, `seconds`, `seconds_per_update`, `isolation_ok`; writes `actor_critic_<arm>_seed<n>_step<u>.pt` (keys `actor`, `critic`, `critic_slow`, `step`, `arm`, `seed`, `n_actions`) at every `checkpoint_every` and at `updates`
  - `load_actor(path, device) -> Actor`
  - `scripts/train_actor_critic.py --world-model CKPT --data DIR --out OUT [--updates N] [--checkpoint-every N] [--device mps] [--seed 0]` → also writes `OUT/train_<arm>_seed<n>.json`

λ-return convention (spec §5.3): `rewards[:, t]` and `discounts[:, t]` come from latent `t+1` (the state the t-th action led to); `values[:, t]` is `V_slow(latent_t)`; `G_t = r_t + d_t·((1−λ)·V(t+1) + λ·G_{t+1})`, `G_H = V(H)`.

- [ ] **Step 1: Write the failing tests**

```python
import copy

import numpy as np
import pytest
import torch

from mbfps.training.ac_trainer import (
    ACConfig, ReturnScale, continue_weights, freeze, lambda_returns, start_states,
    train_actor_critic,
)


def test_lambda_returns_hand_computed():
    """4 steps, a predicted end at step 2 (discount 0 there)."""
    r = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    d = torch.tensor([[0.9, 0.9, 0.0, 0.9]])
    v = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]])
    lam = 0.5
    g4 = 50.0
    g3 = 4.0 + 0.9 * (0.5 * 50.0 + 0.5 * g4)
    g2 = 3.0 + 0.0 * (0.5 * 40.0 + 0.5 * g3)
    g1 = 2.0 + 0.9 * (0.5 * 30.0 + 0.5 * g2)
    g0 = 1.0 + 0.9 * (0.5 * 20.0 + 0.5 * g1)
    out = lambda_returns(r, d, v, lam)
    torch.testing.assert_close(out, torch.tensor([[g0, g1, g2, g3]]))


def test_continue_weights():
    c = torch.tensor([[0.5, 0.5, 0.0, 1.0]])
    torch.testing.assert_close(continue_weights(c), torch.tensor([[1.0, 0.5, 0.25, 0.0]]))


def test_return_scale_floor_and_ema():
    s = ReturnScale(decay=0.5)
    assert s.update(torch.zeros(100)) == 1.0          # P95 - P5 = 0 -> floor 1
    x = torch.linspace(0, 100, 101)
    s2 = ReturnScale(decay=0.5)
    first = s2.update(x)                               # first update sets directly
    assert first == pytest.approx(90.0)
    second = s2.update(x * 2)                          # 0.5*90 + 0.5*180
    assert second == pytest.approx(135.0)
```

For the trainer-level tests, reuse the Task 7 fixture builder. Put it in `tests/training/test_ac_trainer.py`:

```python
from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.data.loader import SequenceLoader
from mbfps.envs.protocol import OBS_SHAPE
from mbfps.training.world_model import WorldModel, load_world_model, train_world_model
from mbfps.utils.config import get_config
from mbfps.utils.device import to_device

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def _ep(t, fill):
    obs = np.zeros((t + 1, *OBS_SHAPE), dtype=np.uint8)
    obs[:, 0, 0, 1] = fill
    term = np.zeros(t, dtype=bool); term[-1] = fill % 2 == 0
    return Episode(obs=obs, actions=(np.arange(t) % 4).astype(np.int32),
                   rewards=np.full(t, -4.0, np.float32), terminated=term,
                   truncated=np.zeros(t, bool),
                   privileged=np.zeros((t + 1, len(KEYS)), np.float32),
                   privileged_keys=KEYS, policy_name="random", seed=fill, scenario="basic")


@pytest.fixture
def world(tmp_path):
    buf = ReplayBuffer(tmp_path / "data", capacity_transitions=100_000)
    for fill in range(1, 8):
        buf.add(_ep(30 if fill % 2 else 6, fill))
    cfg = get_config("cnn", steps=2, seq_len=16, batch_size=4, device="cpu",
                     n_actions=4, pad_short=True, reward_transform="symlog")
    train_world_model(cfg, buf, out_dir=tmp_path / "wm")
    return buf, load_world_model(tmp_path / "wm" / "world_model_cnn_seed0.pt", torch.device("cpu"))


def test_start_states_exclude_padding(world):
    buf, model = world
    batch = SequenceLoader(buf, batch_size=8, seq_len=16, seed=0, pad_short=True,
                           load_obs=True).sample()
    batch["mask"][0, 5:] = False       # a known pad, whatever the draw was
    h, z = start_states(model, to_device(batch, torch.device("cpu")))
    assert h.shape[0] == int(batch["mask"].sum())
    assert h.shape[0] <= 8 * 16 - 11
    assert z.shape == (h.shape[0], 1024)


def test_isolation_world_model_untouched(world, tmp_path):
    buf, model = world
    before = copy.deepcopy(model.state_dict())
    cfg = ACConfig(updates=3, checkpoint_every=3, batch_size=4, seq_len=16, device="cpu")
    history = train_actor_critic(model, buf, cfg, out_dir=tmp_path / "ac", log_every=0)
    assert history["isolation_ok"] is True
    for p in model.parameters():
        assert p.grad is None and not p.requires_grad
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key
    assert (tmp_path / "ac" / "actor_critic_cnn_seed0_step3.pt").is_file()


def test_training_is_reproducible(world, tmp_path):
    buf, model = world
    cfg = ACConfig(updates=2, checkpoint_every=2, batch_size=4, seq_len=16, device="cpu")
    a = train_actor_critic(model, buf, cfg, out_dir=None, log_every=0)
    b = train_actor_critic(model, buf, cfg, out_dir=None, log_every=0)
    assert a["actor_loss"] == b["actor_loss"]
    assert a["critic_loss"] == b["critic_loss"]


def test_checkpoint_every_must_divide_updates():
    with pytest.raises(ValueError, match="checkpoint_every"):
        ACConfig(updates=10, checkpoint_every=3)
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/bin/python -m pytest tests/training/test_ac_trainer.py -q`
Expected: FAIL — `ModuleNotFoundError: mbfps.training.ac_trainer`.

- [ ] **Step 3: Implement `src/mbfps/training/ac_trainer.py`**

```python
"""Actor-critic in imagination -- milestone M4, Stage B.

The world model is loaded from Stage A and FROZEN: every parameter has
`requires_grad=False`, and the optimisers hold only the actor's and critic's
parameters. Start states are every REAL posterior state of a replay batch from
the training split (padding excluded). From each, `imagine_policy` rolls the
prior 15 steps with the actor sampling each action. Returns, losses and the
return scale follow DreamerV3 (spec §5.3).

No real-environment transition ever reaches a gradient here: the trainer
never constructs an environment.
"""

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.loader import SequenceLoader
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.models.encoders import encoder_backbone
from mbfps.models.heads import Actor, Critic
from mbfps.models.transforms import read_reward
from mbfps.utils.device import to_device
from mbfps.utils.seeding import seed_everything


@dataclass(frozen=True)
class ACConfig:
    horizon: int = 15
    gamma: float = 0.997
    lam: float = 0.95
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    entropy: float = 3e-4
    critic_ema: float = 0.98
    scale_decay: float = 0.99
    updates: int = 10_000
    checkpoint_every: int = 2_000
    batch_size: int = 16
    seq_len: int = 64
    seed: int = 0
    device: str = "mps"
    grad_clip: float = 100.0

    def __post_init__(self) -> None:
        if self.checkpoint_every < 1 or self.updates % self.checkpoint_every:
            raise ValueError(
                f"checkpoint_every ({self.checkpoint_every}) must divide updates "
                f"({self.updates}): the gate is read at the final update, which must be a rung"
            )


def lambda_returns(rewards, discounts, values, lam: float) -> torch.Tensor:
    """`G_t = r_t + d_t * ((1 - lam) * V_{t+1} + lam * G_{t+1})`, `G_H = V_H`.
    `rewards`, `discounts`: `(N, H)`; `values`: `(N, H + 1)`."""
    horizon = rewards.shape[1]
    out = [None] * horizon
    nxt = values[:, horizon]
    for t in reversed(range(horizon)):
        nxt = rewards[:, t] + discounts[:, t] * ((1 - lam) * values[:, t + 1] + lam * nxt)
        out[t] = nxt
    return torch.stack(out, dim=1)


def continue_weights(cont: torch.Tensor) -> torch.Tensor:
    """`w_0 = 1`, `w_t = prod_{k<t} cont_k`: steps past a predicted end count less."""
    ones = torch.ones_like(cont[:, :1])
    return torch.cumprod(torch.cat([ones, cont[:, :-1]], dim=1), dim=1)


class ReturnScale:
    """DreamerV3's return normalisation: `S = max(1, P95 - P5)`, the
    percentiles tracked as an EMA. The first update sets them directly."""

    def __init__(self, decay: float) -> None:
        self.decay = decay
        self.low: float | None = None
        self.high: float | None = None

    def update(self, returns: torch.Tensor) -> float:
        flat = returns.detach().flatten().float()
        low = float(torch.quantile(flat, 0.05))
        high = float(torch.quantile(flat, 0.95))
        if self.low is None:
            self.low, self.high = low, high
        else:
            self.low = self.decay * self.low + (1 - self.decay) * low
            self.high = self.decay * self.high + (1 - self.decay) * high
        return max(1.0, self.high - self.low)


def freeze(world_model: nn.Module) -> None:
    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad_(False)
        p.grad = None


@torch.no_grad()
def start_states(world_model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior `(h, z)` at every masked-in position of `batch`."""
    out = world_model.rssm.observe(world_model.embed(batch), batch["actions"])
    mask = batch["mask"].bool()
    return out["h"][mask], out["z"][mask]


def _ema(slow: nn.Module, fast: nn.Module, rate: float) -> None:
    with torch.no_grad():
        for s, f in zip(slow.parameters(), fast.parameters()):
            s.mul_(rate).add_(f, alpha=1 - rate)


def _save(out_dir: Path, arm: str, seed: int, step: int, n_actions: int,
          actor, critic, critic_slow) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"actor_critic_{arm}_seed{seed}_step{step}.pt"
    torch.save({"arm": arm, "seed": seed, "step": step, "n_actions": n_actions,
                "actor": actor.state_dict(), "critic": critic.state_dict(),
                "critic_slow": critic_slow.state_dict()}, path)
    return path


def load_actor(path: Path, device: torch.device) -> Actor:
    payload = torch.load(path, map_location=device, weights_only=True)
    actor = Actor(int(payload["n_actions"]), seed=int(payload["seed"])).to(device)
    actor.load_state_dict(payload["actor"])
    return actor.eval()


def train_actor_critic(world_model, buffer: ReplayBuffer, cfg: ACConfig,
                       out_dir: Path | None, log_every: int = 100) -> dict[str, Any]:
    seed_everything(cfg.seed)
    device = next(world_model.parameters()).device
    freeze(world_model)
    arm, n_actions = world_model.cfg.arm, world_model.cfg.train.n_actions
    transform = world_model.cfg.train.reward_transform

    actor = Actor(n_actions, seed=cfg.seed).to(device)
    critic = Critic(seed=cfg.seed).to(device)
    critic_slow = copy.deepcopy(critic).requires_grad_(False)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    scale = ReturnScale(cfg.scale_decay)

    train_paths, _ = episode_split(buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=0)
    kind = world_model.input_kind
    loader = SequenceLoader(
        buffer, batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed=cfg.seed,
        load_obs=kind == "obs", load_features=kind == "features",
        feature_backbone=encoder_backbone(world_model.cfg.encoder) or "dinov2",
        paths=train_paths, pad_short=True,
    )
    sample_actor = lambda latent: actor.dist(latent).sample()

    history: dict[str, Any] = {"actor_loss": [], "critic_loss": [], "scale": [],
                               "imagined_return": [], "entropy": [], "checkpoints": []}
    isolation_ok = True
    start = time.perf_counter()
    for update in range(1, cfg.updates + 1):
        batch = to_device(loader.sample(), device)
        h, z = start_states(world_model, batch)
        with torch.no_grad():
            traj = world_model.rssm.imagine_policy(sample_actor, (h, z), cfg.horizon)
            heads = world_model.heads(traj["latent"][:, 1:])
            rewards = read_reward(heads["reward"], transform)
            cont = torch.sigmoid(heads["continue_logit"])
            values_slow = critic_slow(traj["latent"])
            returns = lambda_returns(rewards, cfg.gamma * cont, values_slow, cfg.lam)
            weights = continue_weights(cont)
        latents = traj["latent"][:, :-1]

        values = critic(latents)
        critic_loss = (0.5 * weights * (values - returns).pow(2)).mean()
        critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), cfg.grad_clip)
        critic_opt.step()
        _ema(critic_slow, critic, cfg.critic_ema)

        s = scale.update(returns)
        dist = actor.dist(latents)
        advantage = ((returns - values.detach()) / s).detach()
        entropy = dist.entropy()
        actor_loss = (weights * (-advantage * dist.log_prob(traj["action"])
                                 - cfg.entropy * entropy)).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
        actor_opt.step()

        if update == 1:
            isolation_ok = all(p.grad is None for p in world_model.parameters())
            if not isolation_ok:
                raise RuntimeError("a world-model parameter received a gradient (spec §5.2)")

        history["actor_loss"].append(float(actor_loss.detach()))
        history["critic_loss"].append(float(critic_loss.detach()))
        history["scale"].append(s)
        history["imagined_return"].append(float(returns[:, 0].mean()))
        history["entropy"].append(float(entropy.mean().detach()))
        if out_dir is not None and update % cfg.checkpoint_every == 0:
            path = _save(Path(out_dir), arm, cfg.seed, update, n_actions,
                         actor, critic, critic_slow)
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

`world_model.embed(batch)` reads `batch["features"]` with `T+1` rows and returns the last `T` — the same pairing training used. `cfg.seed` seeds the actor; the world model's own seed is its checkpoint's.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/training/test_ac_trainer.py -q`
Expected: all PASS.

- [ ] **Step 5: Write `scripts/train_actor_critic.py`**

```python
"""Train the actor-critic in a frozen world model's imagination (M4 Stage B)."""

import argparse
import sys
from pathlib import Path

from mbfps.data.buffer import ReplayBuffer
from mbfps.eval.study import git_sha, write_record
from mbfps.training.ac_trainer import ACConfig, train_actor_critic
from mbfps.training.world_model import load_world_model
from mbfps.utils.device import get_device


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--world-model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--updates", type=int, default=10_000)
    p.add_argument("--checkpoint-every", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps")
    args = p.parse_args(argv)

    device = get_device(prefer=args.device)
    model = load_world_model(args.world_model, device)
    cfg = ACConfig(updates=args.updates, checkpoint_every=args.checkpoint_every,
                   seed=args.seed, device=args.device)
    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    history = train_actor_critic(model, buffer, cfg, out_dir=args.out)
    record = {k: v for k, v in history.items()}
    record.update({"world_model": str(args.world_model), "data": str(args.data),
                   "config": cfg.__dict__, "git_sha": git_sha()})
    path = args.out / f"train_{history['arm']}_seed{args.seed}.json"
    write_record(path, record)
    print(f"seconds_per_update={history['seconds_per_update']:.4f} isolation_ok={history['isolation_ok']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Add a script smoke test to `tests/training/test_ac_trainer.py`:

```python
def test_script_writes_record_and_checkpoints(world, tmp_path):
    import json, subprocess, sys
    from pathlib import Path
    buf, _ = world
    script = Path(__file__).resolve().parents[2] / "scripts" / "train_actor_critic.py"
    out = tmp_path / "ac_script"
    proc = subprocess.run([sys.executable, str(script),
                           "--world-model", str(tmp_path / "wm" / "world_model_cnn_seed0.pt"),
                           "--data", str(tmp_path / "data"), "--out", str(out),
                           "--updates", "2", "--checkpoint-every", "1", "--device", "cpu"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    record = json.loads((out / "train_cnn_seed0.json").read_text())
    assert record["isolation_ok"] is True and record["updates"] == 2
    assert (out / "actor_critic_cnn_seed0_step1.pt").is_file()
    assert (out / "actor_critic_cnn_seed0_step2.pt").is_file()
```

Run: `.venv/bin/python -m pytest tests/training/test_ac_trainer.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/training/ac_trainer.py scripts/train_actor_critic.py tests/training/test_ac_trainer.py
git commit -F - <<'EOF'
feat: the actor-critic trainer, on a frozen world model it checks it never touches

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

## Task 11: Live evaluation — `AgentPolicy`, episodes, the paired gate

**Files:**
- Create: `src/mbfps/eval/live.py`
- Test: `tests/eval/test_live.py`

**Interfaces:**
- Consumes: Task 5 `WorldModel`; Task 9 `Actor`, `imagine_policy`; Task 10 `lambda_returns`, `Critic`; `mbfps.data.policies.RandomPolicy`.
- Produces:
  - `EVAL_SEEDS = tuple(range(1_000_000, 1_000_100))`, `RESAMPLES = 2000`
  - `AgentPolicy(world_model, actor, encode: Callable[[np.ndarray], np.ndarray], device, mode="sample", critic_slow=None, gamma=0.997, lam=0.95, horizon=15)`: `.name = "agent"`, `.reset(seed)`, `.act(obs) -> int`, `.imagined_start_return: float | None` (set on the first `act` of an episode when `critic_slow` is given)
  - `run_episodes(env, policy, seeds) -> list[dict]` with keys `seed, return, length, killed, truncated, imagined_return`
  - `paired_gate(agent_returns, random_returns, resamples=RESAMPLES, seed=0) -> dict` with `point, ci_low, ci_high, passed`

`encode` maps one `(1, 112, 112, 3)` uint8 frame to the world model's encoder input: the cached-feature geometry (e.g. `(1, 64, 384)` float16) for a feature arm. In production it is `FeatureExtractor(backbone, device).encode`; tests pass a stub.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
import torch

from mbfps.eval.live import EVAL_SEEDS, AgentPolicy, paired_gate, run_episodes
from mbfps.models.heads import Actor, Critic
from mbfps.training.world_model import WorldModel
from mbfps.utils.config import get_config


def test_eval_seeds():
    assert EVAL_SEEDS[0] == 1_000_000 and len(EVAL_SEEDS) == 100


def test_paired_gate_passes_a_clearly_better_policy():
    rng = np.random.default_rng(0)
    base = rng.normal(-170, 186, 100)
    out = paired_gate(base + 120.0, base, resamples=500, seed=0)
    assert out["passed"] and out["ci_low"] > 0


def test_paired_gate_fails_an_identical_policy():
    base = np.random.default_rng(0).normal(-170, 186, 100)
    out = paired_gate(base, base, resamples=500, seed=0)
    assert not out["passed"] and out["point"] == 0.0


def _agent(seed=0):
    cfg = get_config("cnn", device="cpu", n_actions=4)
    model = WorldModel(cfg).eval()
    stub = lambda frame: frame          # cnn kind consumes pixels directly
    return model, AgentPolicy(model, Actor(4, seed=seed), stub, torch.device("cpu"),
                              critic_slow=Critic(seed=seed))


def test_first_frame_is_filtered_with_noop(monkeypatch):
    model, agent = _agent()
    seen = []
    real_observe = model.rssm.observe
    def spy(embeddings, actions, state=None):
        seen.append(actions.clone())
        return real_observe(embeddings, actions, state)
    monkeypatch.setattr(model.rssm, "observe", spy)
    agent.reset(seed=5)
    frame = np.zeros((112, 112, 3), dtype=np.uint8)
    first = agent.act(frame)
    agent.act(frame)
    assert seen[0].tolist() == [[0]]            # the no-op
    assert seen[1].tolist() == [[first]]        # then the agent's own last action
    assert agent.imagined_start_return is not None


class _FakeEnv:
    """Ten-step episodes; reward = +1 for action 3, else -1."""
    def __init__(self): self.t = 0
    def reset(self, *, seed=None):
        self.t = 0
        return np.zeros((112, 112, 3), dtype=np.uint8), {}
    def step(self, action):
        self.t += 1
        done = self.t >= 10
        return (np.full((112, 112, 3), self.t, dtype=np.uint8),
                1.0 if action == 3 else -1.0, False, done, {})


def test_agent_episode_is_reproducible():
    _, agent = _agent()
    a = run_episodes(_FakeEnv(), agent, seeds=[7, 8])
    b = run_episodes(_FakeEnv(), agent, seeds=[7, 8])
    assert [e["return"] for e in a] == [e["return"] for e in b]
    assert [e["imagined_return"] for e in a] == [e["imagined_return"] for e in b]
    assert all(e["length"] == 10 and e["truncated"] and not e["killed"] for e in a)


def test_run_episodes_with_random_policy_on_real_basic():
    from mbfps.data.policies import RandomPolicy
    from mbfps.envs.registry import make_env
    env = make_env("vizdoom", scenario="basic", frame_skip=4)
    try:
        eps = run_episodes(env, RandomPolicy(env.action_space.n), seeds=EVAL_SEEDS[:3])
    finally:
        env.close()
    assert len(eps) == 3 and all(1 <= e["length"] <= 75 for e in eps)
    assert all(e["imagined_return"] is None for e in eps)
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/bin/python -m pytest tests/eval/test_live.py -q`
Expected: FAIL — `ModuleNotFoundError: mbfps.eval.live`.

- [ ] **Step 3: Implement `src/mbfps/eval/live.py`**

```python
"""Live evaluation (M4 spec §5.4-§5.6).

The live game is used ONLY to evaluate: nothing here writes a dataset or takes
a gradient. Each frame goes through the arm's frozen backbone (`encode`), the
world model's encoder and the posterior filter; the actor then SAMPLES an action
from the posterior latent, as it was trained. The first frame of an episode is
filtered as if a no-op (action 0) had produced it, which is how every training
sequence begins: an initial state, then an action and the frame it led to.

Every episode reseeds torch from the episode seed, so a rerun reproduces it
(the posterior's z draw and the actor's action draw both use that stream).
"""

from typing import Callable

import numpy as np
import torch

from mbfps.models.transforms import read_reward
from mbfps.training.ac_trainer import lambda_returns
from mbfps.utils.seeding import fork_seed

EVAL_SEEDS: tuple[int, ...] = tuple(range(1_000_000, 1_000_100))
RESAMPLES: int = 2000
NOOP: int = 0


class AgentPolicy:
    def __init__(self, world_model, actor, encode: Callable[[np.ndarray], np.ndarray],
                 device: torch.device, mode: str = "sample", critic_slow=None,
                 gamma: float = 0.997, lam: float = 0.95, horizon: int = 15) -> None:
        if mode not in ("sample", "argmax"):
            raise ValueError(f"mode must be 'sample' or 'argmax', got {mode!r}")
        self.name = "agent" if mode == "sample" else "agent_argmax"
        self.world_model, self.actor, self.encode = world_model, actor, encode
        self.device, self.mode = device, mode
        self.critic_slow, self.gamma, self.lam, self.horizon = critic_slow, gamma, lam, horizon
        self.imagined_start_return: float | None = None
        self.reset(seed=0)

    def reset(self, seed: int | None = None) -> None:
        torch.manual_seed(fork_seed(int(seed or 0), f"live:{self.name}"))
        self._state = self.world_model.rssm.initial_state(1, self.device)
        self._last_action = NOOP
        self._first = True
        self.imagined_start_return = None

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        inputs = torch.as_tensor(self.encode(obs[None])).to(self.device)
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

    def _imagined_return(self) -> float:
        wm = self.world_model
        traj = wm.rssm.imagine_policy(lambda l: self.actor.dist(l).sample(),
                                      self._state, self.horizon)
        heads = wm.heads(traj["latent"][:, 1:])
        rewards = read_reward(heads["reward"], wm.cfg.train.reward_transform)
        discounts = self.gamma * torch.sigmoid(heads["continue_logit"])
        returns = lambda_returns(rewards, discounts, self.critic_slow(traj["latent"]), self.lam)
        return float(returns[0, 0])


def run_episodes(env, policy, seeds) -> list[dict]:
    episodes = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        policy.reset(seed=int(seed))
        total, length = 0.0, 0
        while True:
            obs, reward, terminated, truncated, _ = env.step(policy.act(obs))
            total += float(reward)
            length += 1
            if terminated or truncated:
                break
        episodes.append({"seed": int(seed), "return": total, "length": length,
                         "killed": bool(terminated), "truncated": bool(truncated),
                         "imagined_return": getattr(policy, "imagined_start_return", None)})
    return episodes


def paired_gate(agent_returns, random_returns, resamples: int = RESAMPLES, seed: int = 0) -> dict:
    """§5.5: the mean per-seed difference, bootstrapped over seeds."""
    d = np.asarray(agent_returns, dtype=np.float64) - np.asarray(random_returns, dtype=np.float64)
    rng = np.random.default_rng(seed)
    reps = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(resamples)])
    low, high = np.percentile(reps, [2.5, 97.5])
    return {"point": float(d.mean()), "ci_low": float(low), "ci_high": float(high),
            "passed": bool(low > 0.0), "n": int(d.size)}
```

`RandomPolicy.reset(seed)` already reseeds per episode, so `run_episodes` gives the random baseline the same seed discipline.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/eval/test_live.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/eval/live.py tests/eval/test_live.py
git commit -F - <<'EOF'
feat: live evaluation -- the agent through the posterior filter, and a seed-paired gate

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

## Task 12: `scripts/eval_live.py` — measure, untrained control, read

**Files:**
- Create: `scripts/eval_live.py`
- Test: `tests/eval/test_eval_live_script.py`

**Interfaces:**
- Consumes: Task 5 `load_world_model`; Task 10 `load_actor`, `ACConfig`; Task 11 everything; `mbfps.data.features.FeatureExtractor`; `mbfps.models.encoders.encoder_backbone`; `mbfps.envs.registry.make_env`; `mbfps.models.heads.Actor`, `Critic`.
- Produces:
  - `python scripts/eval_live.py --phase measure --world-model WM --agent-dir AC --out OUT [--steps 2000,4000,6000,8000,10000] [--episodes 100] [--device mps] [--encode-stub]` → `OUT/live_<arm>_seed<n>.json` with `random` (episodes), `untrained` (episodes), `rungs` (`{step: {"sample": episodes, "argmax": episodes}}`), `git_sha`, `seconds`, `seconds_per_episode`
  - `python scripts/eval_live.py --phase read --out OUT --arm A --seed S --final-step 10000` → `OUT/live_<arm>_seed<n>.txt`; exit `0` PASS, `70` AGENT_NOT_BETTER, `71` CONTROL_BROKEN (the untrained actor passes), `72` ISOLATION_BROKEN (the training record's `isolation_ok` is not true), `11` MISSING
  - `--encode-stub` (tests only) encodes with zeros of the backbone geometry instead of loading DINOv2.
  - `--pixel-ae-checkpoint PATH` is passed to `FeatureExtractor(checkpoint=...)`; for `pixel_ae` it must be the autoencoder the arm's cache was written with.

- [ ] **Step 1: Write the failing test**

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eval_live.py"
AC_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_actor_critic.py"


def _run(script, *args):
    p = subprocess.run([sys.executable, str(script), *map(str, args)],
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


@pytest.fixture
def agent(tmp_path):
    """A frozen_ssl world model on 4 actions over a tiny basic-like buffer with
    dinov2-geometry caches, and 2 actor-critic updates."""
    import numpy as np
    from mbfps.data.buffer import ReplayBuffer
    from mbfps.data.episode import Episode
    from mbfps.envs.protocol import OBS_SHAPE
    from mbfps.training.world_model import train_world_model
    from mbfps.utils.config import get_config

    keys = ("health", "pos_x", "pos_y", "pos_z", "angle")
    buf = ReplayBuffer(tmp_path / "data", capacity_transitions=100_000)
    for fill in range(1, 8):
        t = 30
        buf.add(Episode(obs=np.zeros((t + 1, *OBS_SHAPE), np.uint8),
                        actions=(np.arange(t) % 4).astype(np.int32),
                        rewards=np.full(t, -4.0, np.float32), terminated=np.zeros(t, bool),
                        truncated=np.zeros(t, bool),
                        privileged=np.zeros((t + 1, len(keys)), np.float32),
                        privileged_keys=keys, policy_name="random", seed=fill, scenario="basic"))
    rng = np.random.default_rng(0)
    for path in buf.episode_paths():
        np.save(path.with_suffix(".features.npy"), rng.random((31, 64, 384)).astype(np.float16))
    cfg = get_config("frozen_ssl", steps=2, seq_len=16, batch_size=4, device="cpu",
                     n_actions=4, pad_short=True, reward_transform="symlog")
    train_world_model(cfg, buf, out_dir=tmp_path / "wm")
    wm = tmp_path / "wm" / "world_model_frozen_ssl_seed0.pt"
    code, log = _run(AC_SCRIPT, "--world-model", wm, "--data", tmp_path / "data",
                     "--out", tmp_path / "ac", "--updates", 2, "--checkpoint-every", 2,
                     "--device", "cpu")
    assert code == 0, log
    return tmp_path, wm


def test_measure_and_read(agent):
    root, wm = agent
    out = root / "live"
    code, log = _run(SCRIPT, "--phase", "measure", "--world-model", wm,
                     "--agent-dir", root / "ac", "--out", out, "--steps", "2",
                     "--episodes", 3, "--device", "cpu", "--encode-stub")
    assert code == 0, log
    record = json.loads((out / "live_frozen_ssl_seed0.json").read_text())
    assert len(record["random"]) == 3 and len(record["untrained"]) == 3
    assert set(record["rungs"]) == {"2"}
    assert [e["seed"] for e in record["random"]] == [e["seed"] for e in record["rungs"]["2"]["sample"]]

    code, log = _run(SCRIPT, "--phase", "read", "--out", out, "--arm", "frozen_ssl",
                     "--seed", 0, "--final-step", 2, "--agent-dir", root / "ac")
    assert code in (0, 70, 71), log
    assert (out / "live_frozen_ssl_seed0.txt").is_file()


def test_read_refuses_a_missing_record(tmp_path):
    code, _ = _run(SCRIPT, "--phase", "read", "--out", tmp_path, "--arm", "frozen_ssl",
                   "--seed", 0, "--final-step", 2, "--agent-dir", tmp_path)
    assert code == 11


def _write_live(out, untrained_return, agent_return, isolation=True):
    from mbfps.eval.study import write_record
    seeds = list(range(100))
    ep = lambda s, r: {"seed": s, "return": r + (s % 7), "length": 10, "killed": False,
                       "truncated": True, "imagined_return": r}
    write_record(out / "live_frozen_ssl_seed0.json", {
        "arm": "frozen_ssl", "seed": 0, "seeds": seeds,
        "random": [ep(s, -170.0) for s in seeds],
        "untrained": [ep(s, untrained_return) for s in seeds],
        "rungs": {"10000": {m: [ep(s, agent_return) for s in seeds]
                            for m in ("sample", "argmax")}}})
    write_record(out / "train_frozen_ssl_seed0.json", {"isolation_ok": isolation})


@pytest.mark.parametrize("untrained, agent, isolation, expected", [
    (-170.0, 80.0, True, 0),      # PASS
    (-170.0, -170.0, True, 70),   # AGENT_NOT_BETTER
    (80.0, 80.0, True, 71),       # CONTROL_BROKEN: the untrained actor passes
    (-170.0, 80.0, False, 72),    # ISOLATION_BROKEN
])
def test_read_maps_each_outcome(tmp_path, untrained, agent, isolation, expected):
    _write_live(tmp_path, untrained, agent, isolation)
    code, log = _run(SCRIPT, "--phase", "read", "--out", tmp_path, "--agent-dir", tmp_path,
                     "--arm", "frozen_ssl", "--seed", 0, "--final-step", 10000)
    assert code == expected, log


def test_pixel_ae_checkpoint_reaches_the_extractor(monkeypatch, tmp_path):
    """Live pixel_ae features must come from the autoencoder the cache was
    written with, not FeatureExtractor's my_way_home default."""
    import torch
    from mbfps.data import features
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    sys.path.insert(0, str(SCRIPT.parent))
    import eval_live

    seen = {}

    class Fake:
        def __init__(self, backbone, device, checkpoint=None):
            seen.update(backbone=backbone, checkpoint=checkpoint)
        def encode(self, frames):
            return frames

    monkeypatch.setattr(features, "FeatureExtractor", Fake)
    model = WorldModel(get_config("pixel_ae", device="cpu", n_actions=4))
    ckpt = tmp_path / "autoencoder_cnn.pt"
    eval_live._encoder_fn(model, torch.device("cpu"), stub=False, pixel_ae_checkpoint=ckpt)
    assert seen == {"backbone": "pixel_ae", "checkpoint": ckpt}
```

- [ ] **Step 2: Run to see it fail**

Run: `.venv/bin/python -m pytest tests/eval/test_eval_live_script.py -q`
Expected: FAIL — the script does not exist.

- [ ] **Step 3: Implement `scripts/eval_live.py`**

```python
"""Stage B live evaluation (M4 spec §5.4-§5.7).

measure: for one cell, the random baseline, the UNTRAINED actor (the §5.6
control) and every requested rung, each over the same 100 seeds; each rung in
both modes ("sample" is the gated one, "argmax" is reported).

read: the §5.5 gate at --final-step, refusing to report if the control passes
or the training record does not certify isolation.

Exit codes (read):
  0   PASS              the trained agent's paired interval lies above zero
  70  AGENT_NOT_BETTER  it does not
  71  CONTROL_BROKEN    the untrained actor passes the same gate
  72  ISOLATION_BROKEN  the training record's isolation_ok is not true
  11  MISSING           a record the read needs is absent
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mbfps.data.policies import RandomPolicy
from mbfps.envs.registry import make_env
from mbfps.eval.live import EVAL_SEEDS, AgentPolicy, paired_gate, run_episodes
from mbfps.eval.study import git_sha, load_record, write_record
from mbfps.models.encoders import encoder_backbone
from mbfps.models.heads import Actor, Critic
from mbfps.training.ac_trainer import load_actor
from mbfps.training.world_model import load_world_model
from mbfps.utils.device import get_device

EXIT_OK, EXIT_MISSING = 0, 11
EXIT_AGENT_NOT_BETTER, EXIT_CONTROL_BROKEN, EXIT_ISOLATION_BROKEN = 70, 71, 72
SCENARIO = "basic"


def _encoder_fn(model, device, stub: bool, pixel_ae_checkpoint: Path | None = None):
    """The frozen backbone that produced this arm's training cache, run live.

    `pixel_ae_checkpoint` MUST be the autoencoder the arm's cache was written
    with: `FeatureExtractor` otherwise loads `PIXEL_AE_CHECKPOINT`, the
    my_way_home autoencoder, and the agent would act on features from a
    different encoder than it was trained on. Ignored by the ViT backbones."""
    backbone = encoder_backbone(model.cfg.encoder)
    if backbone is None:
        return lambda frames: frames
    from mbfps.data.geometry import BACKBONE_GEOMETRY
    if stub:
        shape = BACKBONE_GEOMETRY[backbone]
        return lambda frames: np.zeros((len(frames), *shape), dtype=np.float16)
    from mbfps.data import features
    return features.FeatureExtractor(backbone=backbone, device=device.type,
                                     checkpoint=pixel_ae_checkpoint).encode


def _critic_slow(path: Path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    critic = Critic(seed=int(payload["seed"])).to(device)
    critic.load_state_dict(payload["critic_slow"])
    return critic.eval()


def live_path(out: Path, arm: str, seed: int) -> Path:
    return out / f"live_{arm}_seed{seed}.json"


def measure(args) -> int:
    start = time.perf_counter()
    device = get_device(prefer=args.device)
    model = load_world_model(args.world_model, device)
    arm, seed, n_actions = model.cfg.arm, model.cfg.train.seed, model.cfg.train.n_actions
    encode = _encoder_fn(model, device, args.encode_stub, args.pixel_ae_checkpoint)
    seeds = EVAL_SEEDS[: args.episodes]
    env = make_env("vizdoom", scenario=SCENARIO, frame_skip=4)
    n_episodes = 0
    try:
        record = {"arm": arm, "seed": seed, "seeds": list(seeds), "rungs": {}}
        record["random"] = run_episodes(env, RandomPolicy(env.action_space.n), seeds)
        untrained = AgentPolicy(model, Actor(n_actions, seed=seed).to(device).eval(),
                                encode, device)
        record["untrained"] = run_episodes(env, untrained, seeds)
        n_episodes += 2 * len(seeds)
        for step in args.steps:
            path = args.agent_dir / f"actor_critic_{arm}_seed{seed}_step{step}.pt"
            actor, critic_slow = load_actor(path, device), _critic_slow(path, device)
            record["rungs"][str(step)] = {
                mode: run_episodes(env, AgentPolicy(model, actor, encode, device, mode=mode,
                                                    critic_slow=critic_slow), seeds)
                for mode in ("sample", "argmax")
            }
            n_episodes += 2 * len(seeds)
            print(f"[live {arm} s{seed}] step {step} done")
    finally:
        env.close()
    seconds = time.perf_counter() - start
    record.update({"world_model": str(args.world_model), "agent_dir": str(args.agent_dir),
                   "git_sha": git_sha(), "seconds": seconds,
                   "seconds_per_episode": seconds / max(n_episodes, 1)})
    write_record(live_path(args.out, arm, seed), record)
    print(f"wrote {live_path(args.out, arm, seed)}")
    return EXIT_OK


def _summary(episodes) -> str:
    r = np.array([e["return"] for e in episodes])
    kills = np.mean([e["killed"] for e in episodes])
    length = np.median([e["length"] for e in episodes])
    return f"return {r.mean():+8.2f} (sd {r.std():6.2f})  kill {kills:.2f}  median_len {length:.0f}"


def read(args) -> int:
    path = live_path(args.out, args.arm, args.seed)
    train_path = args.agent_dir / f"train_{args.arm}_seed{args.seed}.json"
    if not path.is_file() or not train_path.is_file():
        print(f"MISSING: {path if not path.is_file() else train_path}")
        return EXIT_MISSING
    record, train = load_record(path), load_record(train_path)
    key = str(args.final_step)
    if key not in record["rungs"]:
        print(f"MISSING: rung {key} not in {sorted(record['rungs'])}")
        return EXIT_MISSING
    rand = [e["return"] for e in record["random"]]
    control = paired_gate([e["return"] for e in record["untrained"]], rand)
    final = record["rungs"][key]
    gate = paired_gate([e["return"] for e in final["sample"]], rand)
    imagined = [e["imagined_return"] for e in final["sample"]]
    real = [e["return"] for e in final["sample"]]

    lines = [f"M4 Stage B live evaluation -- {args.arm} seed {args.seed}, {len(rand)} seeds",
             f"  random      {_summary(record['random'])}",
             f"  untrained   {_summary(record['untrained'])}",
             f"    control gate: {control['point']:+.2f} [{control['ci_low']:+.2f}, {control['ci_high']:+.2f}]"
             f"  (must NOT pass)"]
    for step in sorted(record["rungs"], key=int):
        lines.append(f"  step {step:>6s} sample {_summary(record['rungs'][step]['sample'])}")
        lines.append(f"  step {step:>6s} argmax {_summary(record['rungs'][step]['argmax'])}")
    lines += [f"GATE at step {key}: {gate['point']:+.2f} [{gate['ci_low']:+.2f}, {gate['ci_high']:+.2f}]",
              f"imagined-vs-real gap at step {key}: {np.mean(imagined) - np.mean(real):+.2f} "
              f"(imagined {np.mean(imagined):+.2f}, real {np.mean(real):+.2f})",
              f"isolation_ok: {train.get('isolation_ok')}"]
    if train.get("isolation_ok") is not True:
        verdict, code = "ISOLATION_BROKEN", EXIT_ISOLATION_BROKEN
    elif control["passed"]:
        verdict, code = "CONTROL_BROKEN", EXIT_CONTROL_BROKEN
    elif gate["passed"]:
        verdict, code = "PASS", EXIT_OK
    else:
        verdict, code = "AGENT_NOT_BETTER", EXIT_AGENT_NOT_BETTER
    lines.append(f"VERDICT: {verdict}")
    text = "\n".join(lines) + "\n"
    path.with_suffix(".txt").write_text(text)
    print(text, end="")
    return code


def _steps(text: str) -> list[int]:
    return [int(s) for s in text.split(",") if s.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("measure", "read"), required=True)
    p.add_argument("--world-model", type=Path)
    p.add_argument("--agent-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/m4b_agent"))
    p.add_argument("--steps", type=_steps, default=[2000, 4000, 6000, 8000, 10000])
    p.add_argument("--episodes", type=int, default=len(EVAL_SEEDS))
    p.add_argument("--device", default="mps")
    p.add_argument("--encode-stub", action="store_true", help="tests only: zero features")
    p.add_argument("--pixel-ae-checkpoint", type=Path, default=None,
                   help="the autoencoder the pixel_ae cache was written with (Stage C: "
                        "runs/m2_basic/autoencoder_cnn.pt)")
    p.add_argument("--arm")
    p.add_argument("--seed", type=int)
    p.add_argument("--final-step", type=int, default=10_000)
    args = p.parse_args(argv)
    if args.phase == "measure":
        if args.world_model is None:
            raise SystemExit("--phase measure needs --world-model")
        return measure(args)
    if args.arm is None or args.seed is None:
        raise SystemExit("--phase read needs --arm and --seed")
    return read(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test**

Run: `.venv/bin/python -m pytest tests/eval/test_eval_live_script.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_live.py tests/eval/test_eval_live_script.py
git commit -F - <<'EOF'
feat: scripts/eval_live.py -- the gate, refusing when the untrained actor passes it

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

## Task 13: Run Stage B and write its results

**Files:**
- Create: `runs/m4b_agent/`
- Modify: this plan (append `## Stage B results`)

**Interfaces:**
- Consumes: Stage A's `runs/m4a_world/world_model_frozen_ssl_seed0.pt`, `data/basic` with its dinov2 cache.
- Produces: `runs/m4b_agent/actor_critic_frozen_ssl_seed0_step{2000..10000}.pt`, `train_frozen_ssl_seed0.json`, `live_frozen_ssl_seed0.json`, `live_frozen_ssl_seed0.txt`.

**Precondition:** Stage A's verdict was ENTRY_PASS. If not, this task does not run.

- [ ] **Step 1: Suite green at the commit being run** — as Task 8 Step 1; record counts and SHA.

- [ ] **Step 2: Measure the cost before committing to the budget (§5.7)**

```bash
mkdir -p runs/m4b_agent/timing
.venv/bin/python scripts/train_actor_critic.py --world-model runs/m4a_world/world_model_frozen_ssl_seed0.pt \
  --data data/basic --out runs/m4b_agent/timing --updates 200 --checkpoint-every 200 \
  > runs/m4b_agent/timing/train.log 2>&1
echo "exit: $?"; tail -2 runs/m4b_agent/timing/train.log
.venv/bin/python scripts/eval_live.py --phase measure --world-model runs/m4a_world/world_model_frozen_ssl_seed0.pt \
  --agent-dir runs/m4b_agent/timing --out runs/m4b_agent/timing --steps 200 --episodes 10 \
  > runs/m4b_agent/timing/live.log 2>&1
echo "exit: $?"; grep seconds_per_episode runs/m4b_agent/timing/live_frozen_ssl_seed0.json
```

Compute: `train_hours = seconds_per_update × 10,000 / 3600` and `eval_hours = seconds_per_episode × (200 + 5 × 200) / 3600` (random + untrained, then five rungs × two modes, 100 seeds each). **If `train_hours + eval_hours` exceeds 12 hours**, write the revised `--updates`/`--checkpoint-every` (keeping five evenly spaced rungs and the gate at the last) into this task *now*, with the measured numbers as the reason, and commit that edit before Step 3. Never revise after results are seen.

- [ ] **Step 3: Train**

```bash
.venv/bin/python scripts/train_actor_critic.py --world-model runs/m4a_world/world_model_frozen_ssl_seed0.pt \
  --data data/basic --out runs/m4b_agent --updates 10000 --checkpoint-every 2000 \
  > runs/m4b_agent/train.log 2>&1
echo "exit: $?"; tail -3 runs/m4b_agent/train.log
ls runs/m4b_agent/actor_critic_frozen_ssl_seed0_step*.pt | wc -l    # must print 5
```

- [ ] **Step 4: Evaluate live**

```bash
.venv/bin/python scripts/eval_live.py --phase measure --world-model runs/m4a_world/world_model_frozen_ssl_seed0.pt \
  --agent-dir runs/m4b_agent --out runs/m4b_agent > runs/m4b_agent/live.log 2>&1
echo "exit: $?"
```

- [ ] **Step 5: Read**

```bash
.venv/bin/python scripts/eval_live.py --phase read --out runs/m4b_agent --agent-dir runs/m4b_agent \
  --arm frozen_ssl --seed 0 --final-step 10000 > runs/m4b_agent/read.log 2>&1
echo "read exit: $?"; cat runs/m4b_agent/read.log
```

Exit 0 is PASS; 70/71/72 as the script documents. Record the code and the full table.

- [ ] **Step 6: Reproducibility check**

```bash
SECOND=$(mktemp)
.venv/bin/python scripts/eval_live.py --phase read --out runs/m4b_agent --agent-dir runs/m4b_agent \
  --arm frozen_ssl --seed 0 --final-step 10000 > "$SECOND" 2>&1
diff runs/m4b_agent/read.log "$SECOND" && echo "byte-identical"
```

- [ ] **Step 7: Write the results and commit**

Append `## Stage B results` with, each from a named record: suite counts and SHA; measured `seconds_per_update` and `seconds_per_episode` and the budget decision; the learning curve across rungs (sample and argmax); the gate at 10,000 with its interval; the untrained control's interval; the imagined-vs-real gap; `isolation_ok`; the verdict and exit code.

```bash
git add docs/superpowers/plans/2026-10-10-mb-fps-m4-actor-critic.md
git commit -F - <<'EOF'
results: Stage B -- <what the run found, no more>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

**If the verdict is not PASS, stop and report to the author before Stage C.** The spec does not make Stage C conditional on Stage B passing (§6.3), but a failed or broken single cell is a decision point the author should see first.

---

## Task 14: The Stage C driver and the cross-arm reading

**Files:**
- Create: `scripts/m4_study.py`
- Test: `tests/eval/test_m4_study_script.py`

**Interfaces:**
- Consumes: Tasks 5, 7, 10, 12 scripts; `mbfps.eval.study.load_record`; `mbfps.eval.live.paired_gate`; `mbfps.eval.entry` verdict fields in entry records.
- Produces:
  - `separation(scores: dict[str, list[float]]) -> dict[tuple[str, str], str]` — for each ordered arm pair, `"ABOVE"` if every seed of the first beats every seed of the second (9 of 9), `"BELOW"` for the reverse, else `"NOT_SEPARATED"`; keys are sorted pairs `(a, b)` with `a < b`.
  - `python scripts/m4_study.py --phase plan` prints the cell order and the commands it would run, without running them.
  - `python scripts/m4_study.py --phase run --arm A [--seeds 0,1,2]` runs, per seed and skipping any step whose output already exists: world model → entry measure → actor-critic → live measure. One arm per invocation, so a feature cache can be cleared between arms.
  - `python scripts/m4_study.py --phase read` → `runs/m4c_study/study.txt`; exit `0` if all nine cells are present, else `80` (MISSING_CELLS).
  - Directories: `runs/m4c_world/`, `runs/m4c_entry/`, `runs/m4c_agent/<arm>_seed<n>/`, `runs/m4c_study/`.

- [ ] **Step 1: Write the failing tests**

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "m4_study.py"
sys.path.insert(0, str(SCRIPT.parent))
from m4_study import separation  # noqa: E402


def test_separation_requires_nine_of_nine():
    scores = {"a": [10, 11, 12], "b": [1, 2, 3], "c": [2, 11.5, 0]}
    out = separation(scores)
    assert out[("a", "b")] == "ABOVE"
    assert out[("a", "c")] == "NOT_SEPARATED"
    assert out[("b", "c")] == "NOT_SEPARATED"


def test_separation_below_and_ties():
    out = separation({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert out[("a", "b")] == "BELOW"
    tie = separation({"a": [1, 2, 4], "b": [4, 5, 6]})
    assert tie[("a", "b")] == "NOT_SEPARATED"     # 4 vs 4 is not a win


def _fake_cell(root: Path, arm: str, seed: int, ret: float):
    entry = root / "m4c_entry"; entry.mkdir(parents=True, exist_ok=True)
    iv = {"point": 0.1, "ci_low": 0.05, "ci_high": 0.2, "n_rows": 1, "n_episodes": 1}
    (entry / f"entry_{arm}_seed{seed}.json").write_text(json.dumps({
        "arm": arm, "seed": seed, "verdict": "ENTRY_PASS",
        "imagined": {"return": iv, "continue": iv}, "posterior": {"return": iv, "continue": iv},
        "_nonfinite": {}}))
    agent = root / "m4c_agent" / f"{arm}_seed{seed}"; agent.mkdir(parents=True, exist_ok=True)
    ep = lambda r: {"seed": 1, "return": r, "length": 10, "killed": True,
                    "truncated": False, "imagined_return": r}
    (agent / f"live_{arm}_seed{seed}.json").write_text(json.dumps({
        "arm": arm, "seed": seed, "random": [ep(-100.0)] * 4, "untrained": [ep(-100.0)] * 4,
        "rungs": {"10000": {"sample": [ep(ret)] * 4, "argmax": [ep(ret)] * 4}},
        "_nonfinite": {}}))
    (agent / f"train_{arm}_seed{seed}.json").write_text(json.dumps(
        {"isolation_ok": True, "_nonfinite": {}}))


def test_pixel_ae_live_command_names_the_basic_autoencoder(tmp_path):
    from m4_study import PIXEL_AE_BASIC, _commands
    live = _commands(tmp_path, tmp_path / "data", "pixel_ae", 0)[-1][1]
    assert live[-2:] == ["--pixel-ae-checkpoint", str(PIXEL_AE_BASIC)]
    vit = _commands(tmp_path, tmp_path / "data", "random_vit", 0)[-1][1]
    assert "--pixel-ae-checkpoint" not in vit


def test_read_reports_missing_cells(tmp_path):
    _fake_cell(tmp_path, "frozen_ssl", 0, 50.0)
    p = subprocess.run([sys.executable, str(SCRIPT), "--phase", "read", "--runs", str(tmp_path)],
                       capture_output=True, text=True)
    assert p.returncode == 80, p.stdout + p.stderr


def test_read_with_nine_cells(tmp_path):
    for arm, base in (("frozen_ssl", 50.0), ("pixel_ae", 0.0), ("random_vit", 0.5)):
        for seed in (0, 1, 2):
            _fake_cell(tmp_path, arm, seed, base + seed)
    p = subprocess.run([sys.executable, str(SCRIPT), "--phase", "read", "--runs", str(tmp_path)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    text = (tmp_path / "m4c_study" / "study.txt").read_text()
    assert "frozen_ssl vs pixel_ae: ABOVE" in text
    assert "pixel_ae vs random_vit: NOT_SEPARATED" in text
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/bin/python -m pytest tests/eval/test_m4_study_script.py -q`
Expected: FAIL — `ModuleNotFoundError: m4_study`.

- [ ] **Step 3: Implement `scripts/m4_study.py`**

```python
"""Stage C: the 3 x 3 study (M4 spec §6).

run    one arm at a time, so its feature cache can be cleared before the next
       arm's is written (one ~3 GB ViT cache fits; two may not). Every step is
       skipped when its output already exists, so an interrupted run resumes.
read   every cell's entry and live readings, and the §6.2 cross-arm reading:
       arm X ranks above arm Y only if every seed of X beats every seed of Y.

Exit codes (read): 0 all nine cells present and read; 80 MISSING_CELLS.
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ARMS = ("frozen_ssl", "pixel_ae", "random_vit")
SEEDS = (0, 1, 2)
FINAL_STEP = 10_000
PIXEL_AE_BASIC = Path("runs/m2_basic/autoencoder_cnn.pt")
"""The autoencoder Stage C trains on `basic` and writes the pixel_ae cache with.
Live evaluation must encode with the same one (Task 12's --pixel-ae-checkpoint)."""
EXIT_OK, EXIT_MISSING_CELLS = 0, 80


def separation(scores: dict) -> dict:
    out = {}
    for a, b in itertools.combinations(sorted(scores), 2):
        sa, sb = np.asarray(scores[a], float), np.asarray(scores[b], float)
        if np.all(sa[:, None] > sb[None, :]):
            out[(a, b)] = "ABOVE"
        elif np.all(sb[:, None] > sa[None, :]):
            out[(a, b)] = "BELOW"
        else:
            out[(a, b)] = "NOT_SEPARATED"
    return out


def _paths(runs: Path, arm: str, seed: int) -> dict:
    agent = runs / "m4c_agent" / f"{arm}_seed{seed}"
    return {
        "world": runs / "m4c_world" / f"world_model_{arm}_seed{seed}.pt",
        "entry": runs / "m4c_entry" / f"entry_{arm}_seed{seed}.json",
        "agent": agent,
        "final": agent / f"actor_critic_{arm}_seed{seed}_step{FINAL_STEP}.pt",
        "train": agent / f"train_{arm}_seed{seed}.json",
        "live": agent / f"live_{arm}_seed{seed}.json",
    }


def _commands(runs: Path, data: Path, arm: str, seed: int) -> list[tuple[Path, list[str]]]:
    py, p = sys.executable, _paths(runs, arm, seed)
    live_extra = ["--pixel-ae-checkpoint", str(PIXEL_AE_BASIC)] if arm == "pixel_ae" else []
    return [
        (p["world"], [py, str(ROOT / "train_world_model.py"), "--arm", arm, "--seed", str(seed),
                      "--data", str(data), "--out", str(p["world"].parent), "--n-actions", "4",
                      "--pad-short", "--reward-transform", "symlog",
                      "--record", str(p["world"].parent / f"train_{arm}_seed{seed}.json")]),
        (p["entry"], [py, str(ROOT / "entry_check.py"), "--phase", "measure", "--role", "cell",
                      "--checkpoint", str(p["world"]), "--data", str(data),
                      "--out", str(p["entry"].parent)]),
        (p["final"], [py, str(ROOT / "train_actor_critic.py"), "--world-model", str(p["world"]),
                      "--data", str(data), "--out", str(p["agent"]), "--seed", str(seed)]),
        (p["live"], [py, str(ROOT / "eval_live.py"), "--phase", "measure",
                     "--world-model", str(p["world"]), "--agent-dir", str(p["agent"]),
                     "--out", str(p["agent"]), *live_extra]),
    ]


def run(args) -> int:
    for seed in args.seeds:
        for output, cmd in _commands(args.runs, args.data, args.arm, seed):
            if output.exists():
                print(f"skip (exists): {output}")
                continue
            print("run:", " ".join(cmd), flush=True)
            if args.phase == "plan":
                continue
            code = subprocess.call(cmd)
            if code != 0:
                print(f"FAILED ({code}): {' '.join(cmd)}")
                return code
    return EXIT_OK


def read(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mbfps.eval.live import paired_gate
    from mbfps.eval.study import load_record

    missing, scores, lines = [], {arm: [] for arm in ARMS}, ["M4 Stage C -- 3 arms x 3 seeds"]
    for arm in ARMS:
        for seed in SEEDS:
            p = _paths(args.runs, arm, seed)
            absent = [k for k in ("entry", "live", "train") if not p[k].is_file()]
            if absent:
                missing.append(f"{arm}/seed{seed}: {absent}")
                continue
            entry, live, train = load_record(p["entry"]), load_record(p["live"]), load_record(p["train"])
            final = live["rungs"][str(FINAL_STEP)]["sample"]
            rand = [e["return"] for e in live["random"]]
            gate = paired_gate([e["return"] for e in final], rand)
            control = paired_gate([e["return"] for e in live["untrained"]], rand)
            score = float(np.mean([e["return"] for e in final]))
            scores[arm].append(score)
            iv = entry["imagined"]
            lines.append(
                f"  {arm:<11s} s{seed}  score {score:+8.2f}  gate {gate['point']:+7.2f} "
                f"[{gate['ci_low']:+7.2f},{gate['ci_high']:+7.2f}] {'PASS' if gate['passed'] else 'fail'}"
                f"  control {'BROKEN' if control['passed'] else 'ok'}"
                f"  isolation {train.get('isolation_ok')}"
                f"  entry {entry['verdict']} (R {iv['return']['ci_low']:+.3f}, C {iv['continue']['ci_low']:+.3f})")
    if missing:
        lines.append("MISSING_CELLS: " + "; ".join(missing))
    else:
        lines.append("cross-arm reading (9 of 9 pairwise wins to rank):")
        for (a, b), verdict in separation(scores).items():
            lines.append(f"  {a} vs {b}: {verdict}")
    out = args.runs / "m4c_study"
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (out / "study.txt").write_text(text)
    print(text, end="")
    return EXIT_MISSING_CELLS if missing else EXIT_OK


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("plan", "run", "read"), required=True)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--data", type=Path, default=Path("data/basic"))
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=list(SEEDS))
    args = p.parse_args(argv)
    if args.phase in ("plan", "run"):
        if args.arm is None:
            raise SystemExit("--phase plan/run needs --arm (one arm per invocation)")
        return run(args)
    return read(args)


if __name__ == "__main__":
    sys.exit(main())
```

`load_record` reads its non-finite map with `.get(NONFINITE_KEY, {})`, so the fake records need no such key; the `"_nonfinite": {}` entries in the fixture are inert and may be dropped.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/eval/test_m4_study_script.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/m4_study.py tests/eval/test_m4_study_script.py
git commit -F - <<'EOF'
feat: the Stage C driver, and the nine-of-nine cross-arm reading

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

## Task 15: Run Stage C and write its results

**Files:**
- Create: `runs/m2_basic/` (the pixel autoencoder on `basic`), `runs/m4c_world/`, `runs/m4c_entry/`, `runs/m4c_agent/`, `runs/m4c_study/`
- Modify: this plan (append `## Stage C results`)

**Interfaces:**
- Consumes: Task 14 driver; `scripts/train_autoencoder.py`; `scripts/cache_features.py`.

- [ ] **Step 1: Suite green at the commit being run** — as Task 8 Step 1.

- [ ] **Step 2: Reuse Stage A's cell only if the code it depends on is unchanged (§6.1)**

```bash
SHA=$(.venv/bin/python -c "import json;print(json.load(open('runs/m4a_world/train_frozen_ssl_seed0.json'))['git_sha'])")
git diff --quiet "$SHA" HEAD -- src/mbfps/models src/mbfps/training/world_model.py src/mbfps/data src/mbfps/utils \
  && echo "UNCHANGED" || echo "CHANGED"
```

If UNCHANGED: `mkdir -p runs/m4c_world && cp runs/m4a_world/world_model_frozen_ssl_seed0.pt runs/m4a_world/train_frozen_ssl_seed0.json runs/m4c_world/` (copy, never move). If CHANGED: retrain it with the others. Record which.

- [ ] **Step 3: `frozen_ssl` (the dinov2 cache from Stage A is still on disk)**

```bash
.venv/bin/python scripts/m4_study.py --phase plan --arm frozen_ssl     # read the commands
.venv/bin/python scripts/m4_study.py --phase run --arm frozen_ssl > runs/m4c_frozen_ssl.log 2>&1
echo "exit: $?"
```

Expected: exit 0; three cells complete. Then free the cache:

```bash
.venv/bin/python scripts/cache_features.py --data data/basic --backbone dinov2 --clear
```

- [ ] **Step 4: `random_vit`**

```bash
df -h . | tail -1                       # need >= 5 GB free
.venv/bin/python scripts/cache_features.py --data data/basic --backbone random_vit --device mps \
  > "$TMPDIR/m4c_cache_rv.log" 2>&1; echo "exit: $?"
.venv/bin/python scripts/m4_study.py --phase run --arm random_vit > runs/m4c_random_vit.log 2>&1
echo "exit: $?"
.venv/bin/python scripts/cache_features.py --data data/basic --backbone random_vit --clear
```

- [ ] **Step 5: `pixel_ae` — the autoencoder on `basic` first (the M2 step)**

M2's canonical autoencoder (`runs/m2_fixed`) was trained for 20,000 steps at `--seq-len 1`; the `basic` one matches. `--seq-len 1` also means no `basic` episode is too short for the M2 loader.

```bash
.venv/bin/python scripts/train_autoencoder.py --arm cnn --data data/basic --out runs/m2_basic \
  --steps 20000 --seq-len 1 > runs/m2_basic.log 2>&1; echo "exit: $?"
ls runs/m2_basic/autoencoder_cnn.pt
.venv/bin/python scripts/cache_features.py --data data/basic --backbone pixel_ae \
  --checkpoint runs/m2_basic/autoencoder_cnn.pt --device mps > "$TMPDIR/m4c_cache_pa.log" 2>&1
echo "exit: $?"
.venv/bin/python scripts/m4_study.py --phase plan --arm pixel_ae | grep -c -- "--pixel-ae-checkpoint runs/m2_basic/autoencoder_cnn.pt"
```

The last line must print a nonzero count: live evaluation for `pixel_ae` encodes with this same autoencoder (Task 12's option, threaded by Task 14), not `FeatureExtractor`'s `my_way_home` default. Then:

```bash
.venv/bin/python scripts/m4_study.py --phase run --arm pixel_ae > runs/m4c_pixel_ae.log 2>&1
echo "exit: $?"
```

- [ ] **Step 6: Read**

```bash
.venv/bin/python scripts/m4_study.py --phase read > runs/m4c_study/read.log 2>&1
echo "read exit: $?"; cat runs/m4c_study/read.log
```

Expected: exit 0 (all nine present). Exit 80 lists the missing cells; resume with `--phase run` for that arm (its cache must be re-created first).

- [ ] **Step 7: Write the results and commit**

Append `## Stage C results` with, each from a named record: suite counts and SHA; whether Stage A's cell was reused; per cell the entry verdict and readings, the gate and its interval, the control, `isolation_ok`; the per-arm scores; the three pairwise readings with what §6.2 licenses for each; wall clock per arm, measured. State plainly that `NOT_SEPARATED` is a result, not a failure (§6.3).

```bash
git add docs/superpowers/plans/2026-10-10-mb-fps-m4-actor-critic.md
git commit -F - <<'EOF'
results: Stage C -- <what the run found, no more>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```
