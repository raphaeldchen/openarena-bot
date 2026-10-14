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

## Stage A results

Run on 2026-10-10 and 2026-10-11, one cell (`frozen_ssl`, seed 0, scenario `basic`, 4 actions, symlog reward head) plus the known-negative control. Branch `claude/project-recap-ba6e4a`. Paths below are worktree-relative; `runs` and `data` are symlinks into the main checkout (ledger ruling R2). The ledger is `.superpowers/sdd/2026-10-10-mb-fps-m4-actor-critic/progress.md` (lines beginning `Task 8:`); run logs are in `.superpowers/sdd/2026-10-10-mb-fps-m4-actor-critic/runlogs/`. Python below means `PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python` (ruling R1).

### Verdict

**ENTRY_PASS, exit code 0.** Both gated readings have `ci_low > 0` and the control fails Reading R as required. Spec §4.7 row, in the spec's words: "both readings pass, control fails | Stage B". Stage B is licensed to run.

- Record: `runs/m4a_entry/read.log` (last line `VERDICT: ENTRY_PASS`); exit 0 is the ledger line `Task 8: step 7: read exit 0 ENTRY_PASS`. Re-derive: copy `runs/m4a_entry/entry_*.json` into an empty directory `D` and run `python scripts/entry_check.py --phase read --out D --arm frozen_ssl --seed 0; echo $?`. The read recomputes the verdict from the two JSONs; it printed `VERDICT: ENTRY_PASS` and exited 0 when re-run at write time.
- The result is one cell and one seed. It licenses Stage B for this cell only.

### Commits, suite, and what ran where

| item | value | source |
|---|---|---|
| suite | 3077 passed, in 6701.94 s (1:51:41); the log's last line is `3077 passed in 6701.94s (1:51:41)` | `tail -1 .superpowers/sdd/2026-10-10-mb-fps-m4-actor-critic/runlogs/m4a_suite.log`; ledger `Task 8: step 1` |
| suite commit | `d674ab7`, the commit the world model trained at | ledger `Task 8: step 1`; `python3 -c "import json;print(json.load(open('runs/m4a_world/train_frozen_ssl_seed0.json'))['git_sha'])"` prints `d674ab7306689fc580e90cf58e9b88bc3e0913c9` |
| entry check commit | `03be77b` | `python3 -c "import json;[print(json.load(open(f'runs/m4a_entry/{n}.json'))['git_sha']) for n in ('entry_frozen_ssl_seed0','entry_control')]"` prints `03be77b1f70ca925a5e42d62ab1cfcfd8798fbb0` twice |

The suite was green at `d674ab7`, not at `03be77b`. The entry check ran at `03be77b` because the first measure attempts exposed a run-time bug (see "Failed measure attempts"). The diff between the two commits is confined to the entry-check module and its test: `git diff --stat d674ab7..03be77b` prints

```
 src/mbfps/eval/entry.py  |  3 ++-
 tests/eval/test_entry.py | 21 +++++++++++++++++++++
 2 files changed, 23 insertions(+), 1 deletion(-)
```

so "suite green at the commit being run" is treated as met under ruling R7 (the full suite at `d674ab7` plus the covering tests at `03be77b`). The full suite was not re-run at `03be77b`.

### Collection mix decision (before collecting)

Pre-registered rule (Task 8 Step 2): collect `--policies random,scripted` if `scripted_sweep10`'s kill rate is at least 0.10, otherwise `random` alone. Both rates are from 200 episodes each (`cat .superpowers/sdd/2026-10-10-mb-fps-m4-actor-critic/runlogs/m4a_mix.log`):

| policy | kill_rate | median_len |
|---|---|---|
| random | 0.545 | 59 |
| scripted_sweep10 | 0.610 | 48 |

0.610 >= 0.10, so the rule selects `--policies random,scripted` (ledger `Task 8: step 2`). The mix was not revisited after training.

### Dataset: `data/basic`

Collection: 1200 episodes kept, 0 crashes, 53,966 transitions (below the 60,000 capacity, so nothing was evicted), 259.0 s (`tail -5 .superpowers/sdd/2026-10-10-mb-fps-m4-actor-critic/runlogs/m4a_collect.log`). dinov2 features cached for 1200 episodes in 426 s on mps, exit 0 (`tail -3 .superpowers/sdd/2026-10-10-mb-fps-m4-actor-critic/runlogs/m4a_cache.log`).

Counts below come from loading every `data/basic/ep_*_len*.npz` (`ls data/basic/ep_*_len*.npz | wc -l` prints 1200). A "kill" is a reward above 50 (`rewards > 50`), the same threshold `scripts/entry_check.py` uses; the reward values present in the dataset are exactly {-9, -4, -1, 97, 98, 99}, and 709 are above 50. A "termination" is an episode with `terminated.any()`; "truncated" is an episode with `truncated.any()`. Script:

```python
from pathlib import Path
import numpy as np
from mbfps.data.split import episode_split, VAL_FRACTION
paths = sorted(Path("data/basic").glob("ep_*_len*.npz"))
train, val = episode_split(paths, val_fraction=VAL_FRACTION, seed=0)   # VAL_FRACTION = 0.2
def stats(ps, policy=None):
    eps = [np.load(p) for p in ps]
    eps = [d for d in eps if policy is None or str(d["policy_name"]) == policy]
    return dict(episodes=len(eps), transitions=sum(len(d["actions"]) for d in eps),
                kills=sum(int((d["rewards"] > 50).sum()) for d in eps),
                terminated=sum(int(d["terminated"].any()) for d in eps),
                truncated=sum(int(d["truncated"].any()) for d in eps))
```

| set | policy | episodes | transitions | kills | terminated episodes | truncated episodes |
|---|---|---|---|---|---|---|
| full | random | 600 | 27,531 | 340 | 340 | 260 |
| full | scripted | 600 | 26,435 | 369 | 369 | 231 |
| full | all | 1200 | 53,966 | 709 | 709 | 491 |
| held-out (240) | random | 126 | 5,593 | 73 | 73 | 53 |
| held-out (240) | scripted | 114 | 4,867 | 71 | 71 | 43 |
| held-out (240) | all | 240 | 10,460 | 144 | 144 | 96 |
| training (960) | all | 960 | 43,506 | 565 | 565 | 395 |

Every terminated episode holds exactly one kill (kills equal terminations in every row). The held-out split is `episode_split(paths, val_fraction=0.2, seed=0)`: 240 held-out and 960 training episodes. The held-out totals (240 episodes, 144 kills, 144 terminations) match the entry record: `python3 -c "import json;print(json.load(open('runs/m4a_entry/entry_frozen_ssl_seed0.json'))['counts'])"` prints `{'episodes': 240, 'starts': 10220, 'slots': 131553, 'kills': 144, 'terminations': 144}`. The 10,220 starts equal the held-out transitions minus one per episode (10,460 - 240), which follows from the spec's `i < T - 1` start rule.

### World model

Record `runs/m4a_world/train_frozen_ssl_seed0.json`, log `runs/m4a_world/train.log` (`python3 -c "import json;d=json.load(open('runs/m4a_world/train_frozen_ssl_seed0.json'));print({k:d[k] for k in ('steps','seconds','kl_rate_above_free_bits','kl_dyn_max','n_actions','pad_short','reward_transform')})"`):

| quantity | value |
|---|---|
| steps | 20,000 |
| `seconds` | 12,820.69 (the log rounds to 12,821 and prints `steps_per_second=1.56`) |
| `kl_rate_above_free_bits` | 0.9901 |
| `kl_dyn_max` | 1.92509 |
| `loss_first20` / `loss_last20` | 3.70965 / 0.55078 (`tail -8 runs/m4a_world/train.log`) |
| `nonfinite` | `{}` |

The wall clock was inflated by the full test suite running concurrently on the same machine (ledger `Task 8: step 5`: "slowed by the concurrent suite"). The plan's Step 5 expected about 90 minutes; the run took 12,821 s (3 h 34 min). The records do not separate how much of the difference is the suite, so 12,821 s is not a measure of this arm's training cost.

### Gated readings (imagined, horizon 15)

Record `runs/m4a_entry/entry_frozen_ssl_seed0.json`; table printed in `runs/m4a_entry/entry.txt` (identical to `runs/m4a_entry/read.log`). Re-derive: `python3 -c "import json;d=json.load(open('runs/m4a_entry/entry_frozen_ssl_seed0.json'));print(d['imagined']);print(d['posterior']);print(d['c_R'],d['c_C'],d['seconds'])"`. Intervals are 95% percentile intervals from 2,000 episode-clustered bootstrap resamples over the 240 held-out episodes (spec §4.5). Constants from the 960 training episodes: `c_R` = -44.510, `c_C` = 0.9885.

| reading | point | 95% interval | rows | gate |
|---|---|---|---|---|
| Reading R, `R2_return` | +0.4334 | [+0.4007, +0.4680] | 10,220 starts | `ci_low` > 0: pass |
| Reading C, `brier_skill` | +0.1190 | [+0.0836, +0.1555] | 131,553 imagined steps | `ci_low` > 0: pass |

Both lower bounds are above zero. The continue reading clears zero by a smaller margin than the return reading (point 0.119 against a base rate `c_C` of 0.9885, so most imagined steps are non-terminating).

### Posterior references (reported, not gated)

Same two readings with the heads applied to the posterior states (teacher-forced), same rows (`d['posterior']` in the same record):

| reading | point | 95% interval |
|---|---|---|
| `R2_return` | +0.7162 | [+0.6735, +0.7593] |
| `brier_skill` | +0.6816 | [+0.5999, +0.7621] |

Both posterior readings are above their imagined counterparts (return +0.7162 against +0.4334; continue +0.6816 against +0.1190), and the intervals do not overlap for either reading. The imagined continue skill is the larger relative drop. These references attribute a failure under §4.7; since neither gated reading failed, they are context for Stage B and attribute nothing here.

### Position `gap_closed` at h = 15 (reported, not gated)

`python3 -c "import json;d=json.load(open('runs/m4a_entry/entry_frozen_ssl_seed0.json'));print(d['position_gap_closed_h15'],d['position_note'])"` prints `0.7961103530158837 None`: the position step ran (`position_note` is null, so it was not skipped) and the value is +0.796.

What it is: `position_gap_closed()[-1]` from `mbfps.eval.rollout.evaluate_rollout`, with probes fit by `fit_probes` on the training episodes, context 5, horizon 15, the dimensionless fraction of the persistence-to-floor band the model closes at the last horizon step (`src/mbfps/eval/rollout.py` module docstring). What it is not: it gates nothing in §4.5 and does not enter the verdict. It is one cell and one seed, with no interval. The record stores the ratio only, not the band width, so the band-width caveat in the `rollout.py` docstring cannot be checked from this record. It is on a different scenario (`basic`) and a different dataset from M3's `my_way_home` numbers, and is not compared to them here.

### Known-negative control (spec §4.6)

Record `runs/m4a_entry/entry_control.json`: M3's `runs/m3_study_v2/world_model_frozen_ssl_seed0.pt` over `data/my_way_home`'s held-out split, reward head read as `identity`, 6 actions, position skipped (`--no-position`). Re-derive: `python3 -c "import json;d=json.load(open('runs/m4a_entry/entry_control.json'));print(d['imagined']);print(d['counts'],d['reward_transform'],d['position_note'])"`.

| reading | point | 95% interval | rows | required |
|---|---|---|---|---|
| Reading R, `R2_return` (imagined) | -0.0049 | [-0.1017, +0.1382] | 12,025 starts, 24 episodes | `ci_low` <= 0 |
| Reading C, `brier_skill` (imagined) | +0.0104 | [+0.0004, +0.8038] | 177,855 steps, 24 episodes | reported only |

**The control failed Reading R as required** (`ci_low` = -0.1017 <= 0), so the check does not find skill in this head, as §4.6 requires. On 24 episodes its interval still reaches +0.138, so this is a failure to find skill, not a measurement that there is none. Reading C on the control is reported only: its held-out split holds 0 kills and 2 terminations in 24 episodes (`counts` in the record), and the interval is correspondingly wide ([+0.0004, +0.8038]); the lower bound sits just above zero and nothing is concluded from it.

### Failed measure attempts (no records)

The Step 6 measure commands ran three times. Only the third produced the records above.

1. Attempt 1: both measures failed before running with exit 127. Cause: a controller shell-quoting error (zsh did not word-split a `$PY` variable). Not a code defect (ledger `Task 8: step 6 attempt 1`).
2. Attempt 2: both measures crashed in `entry.py`'s `episode_arrays`, which cast with `x.detach().double().cpu()`; the MPS backend has no float64. A Task 6 defect visible only on mps, because the unit tests run on cpu (ledger `Task 8: step 6 attempt 2`). Fixed at `03be77b` ("episode_arrays moves to the CPU before the float64 cast"), which added an mps regression test that skips when mps is absent.
3. Attempt 3, at `03be77b`: cell and control both exit 0 (ledger `Task 8: step 6`). Cell 74.6 s, control 23.9 s (`seconds` in each record; `runs/m4a_entry/measure_cell.log` and `measure_control.log` each contain only the `wrote ...` line).

Neither failed attempt wrote a record. The only files in `runs/m4a_entry/` are the final `entry_frozen_ssl_seed0.json`, `entry_control.json`, `entry.txt`, `read.log` and the two measure logs, and both JSONs carry `git_sha` `03be77b`. The failed attempts' own console output is not retained in the repository.

### Not established by this run

- One cell (`frozen_ssl`) and one seed (0): no between-arm or between-seed statement is made.
- The entry check reads reward and continue under 15-step imagination from held-out posterior starts. It does not read policy quality; that is Stage B's job.
- Stage A's world model is the model Stage B loads (`runs/m4a_world/world_model_frozen_ssl_seed0.pt`).

## Stage B results

Run on 2026-10-11, one cell (`frozen_ssl`, seed 0, scenario `basic`, 4 actions), loading Stage A's world model `runs/m4a_world/world_model_frozen_ssl_seed0.pt` and the fixed dataset `data/basic`. Branch `claude/project-recap-ba6e4a`. Paths are worktree-relative; `runs` and `data` are symlinks into the main checkout (ledger ruling R2). The ledger is `.superpowers/sdd/2026-10-10-mb-fps-m4-actor-critic/progress.md` (lines beginning `Task 13:` and `Task 10:`); run logs are in `.superpowers/sdd/2026-10-10-mb-fps-m4-actor-critic/runlogs/`. Python below means `PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python` (ruling R1). Training and live runs were executed by the controller (ruling R5).

### Verdict

**PASS, exit code 0.** The paired gate at update 10,000 has `ci_low` = +191.86 > 0, the untrained control's interval includes zero (it fails the gate, as §5.6 requires), and `isolation_ok` is True. The last line of `runs/m4b_agent/read.log` is `VERDICT: PASS`; the exit code is the ledger line `Task 13: step 5-6: read exit 0 PASS`.

Re-derive (into a scratch directory so the records are not rewritten; the read writes `live_frozen_ssl_seed0.txt` beside its input):

```bash
D=$(mktemp -d); cp runs/m4b_agent/live_frozen_ssl_seed0.json runs/m4b_agent/train_frozen_ssl_seed0.json "$D"
python scripts/eval_live.py --phase read --out "$D" --agent-dir "$D" --arm frozen_ssl --seed 0 --final-step 10000 > "$D/second.txt"; echo $?
diff runs/m4b_agent/read.log "$D/second.txt" && echo byte-identical
```

At write time this printed exit `0` and `byte-identical`. That is the Step 6 reproducibility check: the read is a pure function of the two JSONs. It does not re-run the live episodes.

This is a statement about one cell, one scenario and a fixed dataset; see "Not established by this run".

### Commits, suite, and what ran where

| item | value | source |
|---|---|---|
| suite | 3113 passed, 0 failed, 2582.48 s (0:43:02); the log's last line is `3113 passed in 2582.48s (0:43:02)` | `tail -1 .superpowers/sdd/2026-10-10-mb-fps-m4-actor-critic/runlogs/m4b_suite.log`; ledger `Task 13: step 1` |
| suite commit | `2535828` (the log itself records no SHA; the commit is from the ledger line) | ledger `Task 13: step 1` |
| timing run commit | `2535828` | `python3 -c "import json;print(json.load(open('runs/m4b_agent/timing/train_frozen_ssl_seed0.json'))['git_sha'])"` |
| training and live run commit | `0a9506d` | `python3 -c "import json;[print(json.load(open(f'runs/m4b_agent/{n}_frozen_ssl_seed0.json'))['git_sha']) for n in ('train','live')]"` prints `0a9506df437701ed115f0f24939eaf78650e73a1` twice |

The suite was green at `2535828`, not at `0a9506d`. The run is at `0a9506d` because the first training attempt crashed (below) and the fix landed after the suite ran. The diff between the two commits is confined to the trainer and its test: `git diff --stat 2535828..0a9506d` prints

```
 src/mbfps/training/ac_trainer.py  |  6 ++++++
 tests/training/test_ac_trainer.py | 26 ++++++++++++++++++++++++++
 2 files changed, 32 insertions(+)
```

Ruling R9 treats "suite green at the commit being run" as met by the full suite at `2535828` (3113 passed) plus the covering tests at `0a9506d` (`tests/training/test_ac_trainer.py` and `tests/models`, 124 passed per R9; that second run is recorded in the ledger only, not in a run log). The full suite was not re-run at `0a9506d`. A regression outside the two changed files is impossible from this diff; a regression inside `ac_trainer.py` that its test does not cover is not excluded by the suite, only by the run itself completing.

### Budget decision (§5.7), made before the training run

Measured at `2535828` by the Step 2 commands, 200 updates and 10 seeds, records in `runs/m4b_agent/timing/`: `python3 -c "import json;a=json.load(open('runs/m4b_agent/timing/train_frozen_ssl_seed0.json'));b=json.load(open('runs/m4b_agent/timing/live_frozen_ssl_seed0.json'));print(a['seconds'],a['seconds_per_update'],a['isolation_ok']);print(b['seconds'],b['seconds_per_episode'])"`.

| quantity | value |
|---|---|
| `seconds_per_update` | 0.48766 (97.53 s over 200 updates) |
| `seconds_per_episode` | 0.52265 (20.91 s over 40 episodes: 10 seeds x random, untrained, sample, argmax) |
| `train_hours` = 0.48766 x 10,000 / 3600 | 1.355 h |
| `eval_hours` = 0.52265 x (200 + 5 x 200) / 3600 | 0.174 h |
| sum | 1.529 h, against the 12 h limit |

1.529 h < 12 h, so the budget was left at the default: 10,000 updates, `--checkpoint-every 2000`, gate at 10,000 (ledger `Task 13: step 2`). No revision was made, and none was made after any result was seen. The first timing attempt was interrupted by a session restart before writing anything (empty `train.log`, no checkpoint, no live process) and was rerun (ledger `Task 13: step 2 attempt 1`).

Realised cost, for the record, from the final runs: training took 3,967.13 s (66.1 min, `seconds` in `runs/m4b_agent/train_frozen_ssl_seed0.json`), 0.39671 s per update; the live measure took 338.28 s (`seconds` in `runs/m4b_agent/live_frozen_ssl_seed0.json`), 0.28190 s per episode over 1,200 episodes. The timing run was made on the code before the memory fix and ran only 200 updates, fewer than the ~350 at which the first attempt crashed; why the final per-update and per-episode costs are lower than the measured ones is not attributed by these records.

### The first training attempt: MPS out-of-memory crash, diagnosis, fix

Attempt 1 of Step 3, at `2535828`, crashed with an MPS out-of-memory error at about update 350 (MPS allocated 2.10 GiB, other allocations 17.98 GiB; ledger `Task 13: step 3 attempt 1`). The first rung is at update 2,000, so **no checkpoint was written**. The console output of that attempt is not retained (`runs/m4b_agent/train.log` was overwritten by the successful attempt), so the figures in this paragraph are the ledger's.

Root cause, measured by a diagnostic script that changes no repository code (`runlogs/mps_mem_probe.py`, ledger `Task 13: root cause`): the number of start states N per update varies with the mask, so every update allocates tensors of new shapes and MPS's caching allocator never reuses blocks. Allocated memory stays flat while driver memory grows. From `runlogs/probe_plain.log` (250 updates, no release):

| update | N | allocated (GiB) | driver (GiB) |
|---|---|---|---|
| 25 | 541 | 0.24 | 3.93 |
| 100 | 781 | 0.32 | 8.69 |
| 200 | 777 | 0.33 | 14.40 |
| 250 | 544 | 0.24 | 16.52 |

N ranged over 541..781 in the logged updates; allocated stayed in 0.24..0.33 GiB; driver memory grew by 12.59 GiB between updates 25 and 250, about 57 MiB per update. With `torch.mps.empty_cache()` once per update (`runlogs/probe_empty.log`, same 250 updates) the driver figure is 1.10 GiB at every logged update except update 200, where it is 1.11 GiB, and the 250 updates took 91.22 s of wall time (`real` line of that log). Re-derive: `cat .superpowers/sdd/2026-10-10-mb-fps-m4-actor-critic/runlogs/probe_plain.log .superpowers/sdd/2026-10-10-mb-fps-m4-actor-critic/runlogs/probe_empty.log`. The probes ran 250 updates and did not themselves reach the out-of-memory point.

Fix: commit `0a9506d` ("fix: release the MPS cache each actor-critic update; the varying start-state count defeats block reuse"), six lines in `src/mbfps/training/ac_trainer.py` calling `torch.mps.empty_cache()` once per update when the device is mps, plus a test (mps test RED to GREEN; a cpu test asserts no call; ledger `Task 10: fix round 1/5`). The fix changes only memory release; it adds no operation to the computation.

No checkpoint from the failed attempt exists or was used: `runs/m4b_agent/` holds exactly five checkpoints, `actor_critic_frozen_ssl_seed0_step{2000,4000,6000,8000,10000}.pt`, with file times 14:02 to 14:56 on 2026-10-11, after the fix commit (13:49:01, `git log -1 --format=%ad --date=iso 0a9506d`); `train_frozen_ssl_seed0.json` lists exactly these five in `checkpoints`, and carries `git_sha` `0a9506d`. The only other checkpoint is `runs/m4b_agent/timing/actor_critic_frozen_ssl_seed0_step200.pt`, from the timing run, in its own directory; the live evaluation read `--agent-dir runs/m4b_agent`. The successful attempt trained 10,000 updates with `isolation_ok` True and `nonfinite` empty (`train_frozen_ssl_seed0.json`).

### Training

Record `runs/m4b_agent/train_frozen_ssl_seed0.json`, log `runs/m4b_agent/train.log`. Config as recorded: horizon 15, gamma 0.997, lambda 0.95, actor and critic learning rate 3e-5, entropy 3e-4, critic EMA 0.98, batch 16 sequences of 64, seed 0, device mps. The imagined start-state lambda-return `G0` printed every 100 updates (a mean over that update's start states, so a noisy per-batch figure): -17.75 at 1,000; +11.11 at 2,000; +20.34 at 4,000; +17.54 at 6,000; +17.75 at 8,000; +32.44 at 10,000 (`grep -E 'update (1000|2000|4000|6000|8000|10000)/' runs/m4b_agent/train.log`; ledger `Task 13: step 3`). `G0` was -45.18 at update 100 and first positive at update 1,300 (+9.64). The last five logged values (9,600 to 10,000) are +34.86, +12.98, +19.55, +21.07, +32.44, so the single-batch figure moves by more than 20 between prints late in training and the 10,000 value should not be read as a trend. The critic loss fell from 363.96 at update 100 to 72.50 at update 10,000 and stayed between 46.16 and 74.45 at the printed updates from 2,000 on.

### Live evaluation: the gate, the control, the learning curve

Record `runs/m4b_agent/live_frozen_ssl_seed0.json` (printed table: `runs/m4b_agent/read.log`, identical to `live_frozen_ssl_seed0.txt`). 100 evaluation episodes on seeds 1,000,000 to 1,000,099 (`d['seeds']`), the same seeds for random, untrained and every rung and mode; the read refuses with `SEED_MISMATCH` otherwise. Each entry below is the mean of `return` over the 100 episodes of that list; the "gate" column is the mean of the per-seed difference from `random`, with a 95% percentile interval from 2,000 seed resamples at a fixed seed (`paired_gate` in `src/mbfps/eval/live.py`). Re-derive:

```python
import json, numpy as np
from mbfps.eval.live import paired_gate
d = json.load(open("runs/m4b_agent/live_frozen_ssl_seed0.json"))
r = [e["return"] for e in d["random"]]
for name, l in [("untrained", d["untrained"])] + [(f"{s} {m}", d["rungs"][s][m]) for s in d["rungs"] for m in ("sample", "argmax")]:
    a = [e["return"] for e in l]; g = paired_gate(a, r)
    print(name, np.mean(a), np.std(a), np.mean([e["killed"] for e in l]), np.median([e["length"] for e in l]),
          g["point"], g["ci_low"], g["ci_high"], g["passed"])
```

| list | mean return | sd | kill rate | median length | gate: point [95% interval] |
|---|---|---|---|---|---|
| random (`RandomPolicy`) | -164.21 | 186.19 | 0.57 | 59 | (baseline) |
| untrained actor (control) | -159.86 | 181.09 | 0.59 | 52.5 | +4.35 [-41.91, +50.70] |
| step 2,000, sample | +65.06 | 20.81 | 1.00 | 9 | +229.27 [+193.47, +264.82] |
| step 2,000, argmax | +66.55 | 19.60 | 1.00 | 8 | +230.76 [+195.03, +266.57] |
| step 4,000, sample | +61.96 | 21.38 | 1.00 | 9.5 | +226.17 [+191.01, +261.73] |
| step 4,000, argmax | +66.60 | 17.66 | 1.00 | 8.5 | +230.81 [+195.11, +266.08] |
| step 6,000, sample | +62.36 | 22.36 | 1.00 | 9 | +226.57 [+190.91, +261.76] |
| step 6,000, argmax | +64.84 | 20.02 | 1.00 | 9 | +229.05 [+194.00, +264.34] |
| step 8,000, sample | +63.62 | 22.89 | 1.00 | 9 | +227.83 [+192.01, +262.63] |
| step 8,000, argmax | +65.18 | 19.10 | 1.00 | 9 | +229.39 [+193.49, +264.85] |
| **step 10,000, sample (the gate)** | **+63.18** | 22.45 | 1.00 | 9 | **+227.39 [+191.86, +262.69]** |
| step 10,000, argmax | +65.26 | 18.85 | 1.00 | 9 | +229.47 [+193.87, +264.74] |

The standard deviations are population sd (`numpy` default), as `scripts/eval_live.py` prints them. The medians in the table are the records' exact values (`np.median` of `length`); `read.log` formats them with `:.0f` and so prints 52 (untrained), 10 (step 4,000 sample) and 8 (step 4,000 argmax) where the records give 52.5, 9.5 and 8.5. All other medians are whole numbers. The `random` and `untrained` rows of the table above agree with `read.log`; the intervals for the non-gate rows were computed with the same `paired_gate` call and are not printed by `read.log` (the script prints only the control's and the 10,000-sample gate's).

**The gate.** Read at update 10,000, sample mode, as fixed in §5.7: +227.39, 95% interval [+191.86, +262.69], `ci_low` > 0. The gate was read at 10,000 only; the other rungs are a learning curve and were not used to select a checkpoint.

**The control (§5.6).** The untrained actor (seeded initialisation, zero updates) goes through the same evaluation: +4.35 [-41.91, +50.70]. `ci_low` is below zero, so it fails the gate as §5.6 requires and the evaluation is not shown broken. The interval is wide (it reaches +50.70); this is a failure to find an effect, not a measurement that the untrained actor equals random. Its kill rate (0.59) and median length (52.5) are close to random's (0.57, 59).

**The learning curve.** Every rung from 2,000 on has sample-mode mean return between +61.96 and +65.06 and a kill rate of 1.00; argmax-mode between +64.84 and +66.60. Every rung's paired-gate interval has `ci_low` above +190. The curve is flat from the first rung: the first evaluation, at update 2,000, already passes the gate by the same margin as update 10,000, so these records show no improvement in live return between 2,000 and 10,000 updates. They say nothing about the first 2,000 updates, which were not evaluated live (the imagined `G0` in the training log was negative until update ~1,300). The rung-to-rung differences in mean return (range 3.1 in sample mode) are about the size of the standard error of one rung mean (sd roughly 21 to 23 over 100 episodes, so about 2.2); no test of rung against rung was run, and none is claimed. Argmax mean exceeds sample mean at every rung (by 1.5 to 4.6); no interval was computed for that difference and it is not gated. Only 33 of the 100 episodes at 10,000 have identical returns in the two modes.

### Per-seed facts

All derived from the live record; the script that produced them is in the box below (needs `numpy` only).

```python
import json, numpy as np
d = json.load(open("runs/m4b_agent/live_frozen_ssl_seed0.json"))
r = np.array([e["return"] for e in d["random"]]); k = np.array([e["killed"] for e in d["random"]])
l = d["rungs"]["10000"]["sample"]; a = np.array([e["return"] for e in l])
print((a > r).sum(), (a == r).sum(), (a < r).sum())
for m in (k, ~k): print(m.sum(), (a > r)[m].sum(), (a == r)[m].sum(), (a < r)[m].sum(), r[m].mean(), a[m].mean(), (a - r)[m].mean())
im = np.array([e["imagined_return"] for e in l]); print(im.mean(), a.mean(), (im > a).sum(), (im < a).sum(), np.corrcoef(im, a)[0, 1])
```

At the gate (update 10,000, sample mode), against random on the same seed:

- The agent's return is higher on **90 of 100 seeds**, equal on 3, lower on 7. By rung (sample mode) the "higher" counts are 89 (2,000), 90 (4,000), 89 (6,000), 90 (8,000), 90 (10,000); in argmax mode 87, 89, 87, 86, 87. The untrained control beats random on 44 seeds, ties on 7, loses on 49 (`d['untrained']` against `d['random']`).
- All three ties are 91 against 91. The seven losses are seeds 1000004 (agent 71, random 79), 1000014 (67, 87), 1000015 (91, 95), 1000021 (24, 55), 1000027 (91, 95), 1000044 (91, 95), 1000098 (91, 95): the random policy killed on each of these, quickly.
- Kill rate: the agent killed on 100 of 100 episodes at every rung and in both modes (`killed` is the environment's `terminated`), with no truncated episode; the maximum agent episode length is 26 at 10,000 sample (18 at argmax) against the 75-step maximum for random. Random killed on 57 of 100 (43 truncated); the untrained actor on 59 of 100 (41 truncated).
- Where the +227.39 comes from. On the 57 seeds where random killed, random's mean return is -18.96 and the agent's is +69.79, a mean difference of +88.75 (agent higher on 47, equal on 3, lower on 7). On the 43 seeds where random was truncated without a kill, random's mean return is -356.74 and the agent's is +54.42, a mean difference of +411.16 (agent higher on all 43). Weighted by 57 and 43 these give 50.59 + 176.80 = 227.39. So about three quarters of the gate's point estimate comes from seeds on which random never killed. The gate is a mean over seeds and does not separate these two groups; random's long, penalty-accumulating truncated episodes set much of the baseline's low mean.
- The agent's best episode (+95) equals random's best (+95); the agent's lowest at 10,000 sample is -21, random's is -375. The agent's returns span -21 to +95 (sd 22.45); a kill ends the episode, and the agent kills in a median of 9 steps.

### Imagined-vs-real gap (reported, not gated)

At update 10,000, sample mode: mean imagined start-state lambda-return +38.33, mean real return +63.18, **gap -24.85** (`read.log`: `imagined-vs-real gap at step 10000: -24.85 (imagined +38.33, real +63.18)`). The imagined value is `imagined_return` in each episode of the rung, computed at that episode's first state; the random and untrained lists carry `None` for it. Per rung (sample / argmax): 2,000: -38.34 / -38.07; 4,000: -30.24 / -32.64; 6,000: -27.77 / -25.09; 8,000: -26.73 / -26.52; 10,000: -24.85 / -26.65 (imagined minus real, from the same loop over `d["rungs"]`). Per episode at 10,000 sample, imagined is below real on 88 of 100 and above on 12, and the correlation between the two is +0.18.

What the sign does and does not mean. §5.5 names a large **positive** gap as the signature of an actor exploiting the model's errors. The gap here is negative at every rung and in both modes, so the §5.5 symptom, a large positive gap, is not observed in this cell. Because the two quantities are not like-for-like (below), a negative gap does not rule out a positive exploitation error hidden by the difference between them. It does not show the world model is accurate, and the two quantities are not the same quantity: the imagined number is a 15-step lambda-return in the model's latent space, discounted by `gamma * sigmoid(continue)` at each step and bootstrapped at the horizon from the slow critic's value (`AgentPolicy._imagined_return` in `src/mbfps/eval/live.py`), while the real number is an undiscounted sum of environment rewards to the end of a real episode (median length 9). Over a median episode of about 9 steps with gamma = 0.997 the discount itself is small, so the main differences are the critic bootstrap and the 15-step horizon extending past a short episode. `AgentPolicy._imagined_return` always SAMPLES actions in imagination (the `dist.sample()` inside `_imagined_return` in `src/mbfps/eval/live.py`), whatever the evaluation mode, so the "argmax" gap column compares a sampled-policy imagined return with an argmax-policy real return. A discounted, critic-bootstrapped estimate being below an undiscounted real return does not by itself establish model error, and a negative gap of -24.85 should not be read as "the model is pessimistic" or "the critic is under-confident" without a like-for-like comparison, which was not run. The per-episode correlation of +0.18 is low, so the imagined figure predicts which seeds the agent scores well on only weakly. The training-log `G0` at update 10,000 (+32.44) is a different sample (one training batch of replay start states) from the +38.33 (the 100 live first states) and the two should not be compared as a check.

### Imagination isolation

`isolation_ok: True` (`runs/m4b_agent/train_frozen_ssl_seed0.json`, also `read.log`; the timing run's record also says True). The flag is set in `train_actor_critic` at update 1: every world-model parameter's `.grad` is `None` after the first actor-critic step, and the trainer raises if not. It is therefore checked once, at the first update, in this run; the unit test for the same property (`tests/training/test_ac_trainer.py::test_isolation_world_model_untouched`) is part of the suite counted above. The trainer draws its start states only from the replay buffer over `data/basic`; the live episodes are run by `scripts/eval_live.py`, a separate script, and the records do not show any live transition used in training.

### Not established by this run

- One cell (`frozen_ssl`) and one seed (0), one scenario (`basic`), and the fixed dataset of Stage A (1,200 scripted-and-random episodes). No between-arm or between-seed statement is made; nothing here compares arms, which is Stage C.
- The 100 evaluation seeds are one fixed set (1,000,000 to 1,000,099); the intervals resample these seeds and do not cover variation in the agent's training seed or in the world model.
- The gate shows the agent's mean return exceeds a uniform-random policy's on `basic`. It does not show the agent is a strong policy: no scripted or other non-random baseline was evaluated here, and the return difference is largely random's truncated episodes (see "Per-seed facts").
- Whether the flat learning curve reflects a quickly saturated task, a ceiling set by the environment, or the learning rate was not tested; no ablation was run.
- No claim is made about the world model's accuracy from the imagined-vs-real gap.

## Stage C results

Run on 2026-10-11 to 2026-10-13: the 3 x 3 study of spec §6, arms `frozen_ssl`, `pixel_ae`, `random_vit` x seeds 0, 1, 2, scenario `basic`, 4 actions, symlog reward head, the fixed dataset `data/basic` (1,200 episodes, Stage A). Branch `claude/project-recap-ba6e4a`. Paths are worktree-relative; `runs` and `data` are symlinks into the main checkout (ledger ruling R2). The ledger is `.superpowers/sdd/2026-10-10-mb-fps-m4-actor-critic/progress.md` (lines beginning `Task 15:`); run logs are in `.superpowers/sdd/2026-10-10-mb-fps-m4-actor-critic/runlogs/` and `runs/`. Python below means `PYTHONPATH=src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python` (ruling R1); `python3` with no `PYTHONPATH` is used only where a command needs no repository import. Everything was executed by the controller (ruling R5).

### Verdict

**Stage C gate (§6.3) met; the §6.2 reading is `NOT_SEPARATED` for all three pairs. Exit code 0.** All nine cells are present, once each (`ls runs/m4c_entry runs/m4c_world/*.pt runs/m4c_agent` list nine entry records, nine world models, nine agent directories); every cell's live gate, untrained control, `isolation_ok` and entry readings are reported below; the cross-arm reading is printed. Every cell passes the §5.5 live gate (`ci_low` > 0), every control fails it as §5.6 requires, every cell has `isolation_ok` True, and every cell has entry verdict `ENTRY_PASS`.

```
frozen_ssl vs pixel_ae: NOT_SEPARATED
frozen_ssl vs random_vit: NOT_SEPARATED
pixel_ae vs random_vit: NOT_SEPARATED
```

- Record: `runs/m4c_study/read.log` (identical to `runs/m4c_study/study.txt`); the exit code is the ledger line `Task 15: step 6: read exit 0`.
- Re-derive, into a scratch directory so the records are not rewritten (the read writes `study.txt` under its `--runs` directory):

```bash
D=$(mktemp -d); mkdir -p "$D/m4c_entry" "$D/m4c_agent"; cp runs/m4c_entry/*.json "$D/m4c_entry/"
for d in runs/m4c_agent/*; do mkdir -p "$D/m4c_agent/$(basename $d)"; cp $d/*.json "$D/m4c_agent/$(basename $d)/"; done
python scripts/m4_study.py --phase read --runs "$D" > "$D/second.txt"; echo $?
diff runs/m4c_study/read.log "$D/second.txt" && echo byte-identical
```

At write time this printed exit `0` and `byte-identical`. The read is a pure function of the 9 entry, 9 train and 9 live JSONs; it does not re-run the live episodes.

**`NOT_SEPARATED` is a result, not a failure** (§6.3: "Stage C does **not** require any arm to pass Stage B or to separate"; §6.2 calls it "a legitimate, reportable result"). What it licenses: on `basic`, with 3 seeds per arm, 100 evaluation seeds, and the score defined as mean live return at update 10,000, **no arm ranks above another** under the pre-registered 9-of-9 rule. What it does not license: it does not establish that the arms are equal or equivalent. No equivalence test or margin was pre-registered or run; the rule can only detect complete separation, and the absence of complete separation at 3 vs 3 is what is reported. The pairwise win counts below are descriptive and carry no p-value.

### Commits, suite, and what ran where

| item | value | source |
|---|---|---|
| suite | 3113 passed, 0 failed, 2582.48 s (0:43:02), at `2535828` | `tail -1 .superpowers/sdd/2026-10-10-mb-fps-m4-actor-critic/runlogs/m4b_suite.log` prints `3113 passed in 2582.48s (0:43:02)`; the commit is from the ledger (`Task 13: step 1`) and ruling R9 |
| covering tests for the MPS fix | `tests/training/test_ac_trainer.py` + `tests/models`, 124 passed, at `0a9506d` | ruling R9 (ledger only; not in a run log) |
| commit the 36 Stage C records carry | `e235d07` | `python3 -c "import json,glob;print({json.load(open(f))['git_sha'][:7] for f in glob.glob('runs/m4c_world/train_*.json')+glob.glob('runs/m4c_entry/entry_*.json')+glob.glob('runs/m4c_agent/*/*.json')})"` prints `{'e235d07'}` (9 world + 9 entry + 18 agent records; `nonfinite` is `{}` in all 36) |
| code identity with `0a9506d` | `src/`, `scripts/` and `tests/` byte-identical | `git diff --stat 0a9506d HEAD -- src scripts` prints nothing (exit 0); so does `git diff --stat 0a9506d e235d07 -- src scripts tests` |

R9's chain: the suite ran green at `2535828` (which already contains `scripts/m4_study.py`, the Stage C driver, and its test `tests/eval/test_m4_study_script.py`; `git show --stat 2535828`); `0a9506d` added only the MPS cache release in `ac_trainer.py` and its test, covered by the 124 tests; the Stage C runs executed at `e235d07`, whose `src/`, `scripts/` and `tests/` equal `0a9506d`'s. Commits `fd19eb5` and `e235d07` changed only this plan file (`git diff --stat 0a9506d HEAD` lists one file, the plan). The full suite was **not** re-run at `0a9506d` or `e235d07` for Stage C (ledger `Task 15: step 1`: "suite per R9"). The records carry `e235d07`, not `0a9506d`, because that was HEAD when the arms ran; the trees are the same. As in Stage B, a regression that the 124 covering tests do not cover is excluded by the Stage C runs completing, not by the suite.

No run failed: `grep -c 'FAILED\|Traceback\|Error' runs/m4c_frozen_ssl.log runs/m4c_random_vit.log runs/m4c_pixel_ae.log` prints `0` for each, and `grep -c 'skip (exists)'` prints `0` for each (no cell was skipped as already present). Each arm log holds 12 `run:` lines (3 seeds x world model, entry check, actor-critic, live). The `pixel_ae` log's three live commands carry `--pixel-ae-checkpoint runs/m2_basic/autoencoder_cnn.pt` (`grep '^run:' runs/m4c_pixel_ae.log | grep -c -- '--pixel-ae-checkpoint runs/m2_basic/autoencoder_cnn.pt'` prints `3`), and the three live records say so (`live_pixel_ae_seed{0,1,2}.json['pixel_ae_checkpoint']` is that path; for the other six cells it is `None`).

### Stage A's cell was retrained, not reused

The pre-registered reuse check (plan Task 15 Step 2, spec §6.1) compares the code Stage A's world model was trained at against HEAD:

```bash
SHA=$(python3 -c "import json;print(json.load(open('runs/m4a_world/train_frozen_ssl_seed0.json'))['git_sha'])")   # d674ab7306689fc580e90cf58e9b88bc3e0913c9
git diff --quiet "$SHA" HEAD -- src/mbfps/models src/mbfps/training/world_model.py src/mbfps/data src/mbfps/utils && echo UNCHANGED || echo CHANGED
git diff --stat "$SHA" HEAD -- src/mbfps/models src/mbfps/training/world_model.py src/mbfps/data src/mbfps/utils
```

It prints **`CHANGED`** (ledger `Task 15: step 2`), with

```
 src/mbfps/models/heads.py | 37 +++++++++++++++++++++++++++++++++----
 src/mbfps/models/rssm.py  | 27 +++++++++++++++++++++++++++
 2 files changed, 60 insertions(+), 4 deletions(-)
```

The ledger attributes these to Task 9's `Actor`/`Critic`/`imagine_policy` additions. Under the pre-registered rule the cell was therefore **retrained with the other eight, not copied**: `runs/m4c_world/world_model_frozen_ssl_seed0.pt` carries `git_sha` `e235d07` and its own `seconds` (6,017.18; Stage A's was 12,820.69, a run that overlapped a concurrent test suite; Stage A says the records do not separate how much of the difference that accounts for, and neither does this section). Stage A's files in `runs/m4a_world/` were not touched.

**Does the retrained cell reproduce Stage B's? Yes, exactly, on every record compared.** Stage B's live record is `runs/m4b_agent/live_frozen_ssl_seed0.json` (code `0a9506d`); Stage C's is `runs/m4c_agent/frozen_ssl_seed0/live_frozen_ssl_seed0.json` (code `e235d07`, same tree). The two world models were trained separately (Stage A's, Stage C's) and the two actor-critic runs were separate 10,000-update runs.

```python
import json, numpy as np, torch
A = json.load(open("runs/m4b_agent/live_frozen_ssl_seed0.json")); B = json.load(open("runs/m4c_agent/frozen_ssl_seed0/live_frozen_ssl_seed0.json"))
key = lambda l: [(e["seed"], e["return"], e["length"], e["killed"], e["truncated"], e["imagined_return"]) for e in l]
print(A["seeds"] == B["seeds"], key(A["random"]) == key(B["random"]), key(A["untrained"]) == key(B["untrained"]))
print(all(key(A["rungs"][s][m]) == key(B["rungs"][s][m]) for s in A["rungs"] for m in ("sample", "argmax")))
ta = json.load(open("runs/m4b_agent/train_frozen_ssl_seed0.json")); tb = json.load(open("runs/m4c_agent/frozen_ssl_seed0/train_frozen_ssl_seed0.json"))
print({k: ta[k] == tb[k] for k in ("actor_loss", "critic_loss", "scale", "imagined_return", "entropy")})
wa = torch.load("runs/m4a_world/world_model_frozen_ssl_seed0.pt", map_location="cpu", weights_only=True)["state_dict"]
wb = torch.load("runs/m4c_world/world_model_frozen_ssl_seed0.pt", map_location="cpu", weights_only=True)["state_dict"]
print(len(wa), set(wa) == set(wb), all(torch.equal(wa[k], wb[k]) for k in wa))
ea = json.load(open("runs/m4a_entry/entry_frozen_ssl_seed0.json")); ec = json.load(open("runs/m4c_entry/entry_frozen_ssl_seed0.json"))
print({k: ea[k] == ec[k] for k in ("imagined", "posterior", "counts", "c_R", "c_C", "position_gap_closed_h15", "verdict")})
for st in (2000, 4000, 6000, 8000, 10000):
    ca = torch.load(f"runs/m4b_agent/actor_critic_frozen_ssl_seed0_step{st}.pt", map_location="cpu", weights_only=True)
    cb = torch.load(f"runs/m4c_agent/frozen_ssl_seed0/actor_critic_frozen_ssl_seed0_step{st}.pt", map_location="cpu", weights_only=True)
    print(st, {k: set(ca[k]) == set(cb[k]) and all(torch.equal(ca[k][n], cb[k][n]) for n in ca[k]) for k in ("actor", "critic", "critic_slow")})
```

At write time this printed `True True True`; `True`; all five training series `True` (each 10,000 entries); `36 True True`; all seven entry fields `True`; and, for each of the five rungs (2,000 to 10,000), `{'actor': True, 'critic': True, 'critic_slow': True}` (6 tensors in each of the three networks). In words:

- **Matches:** the 100 evaluation seeds; all 1,200 episode records in the random list, the untrained list and the ten rung-mode lists (seed by seed: `return`, `length`, `killed`, `truncated` and `imagined_return`, with a maximum absolute `imagined_return` difference of 0.0); the 10,000-entry `actor_loss`, `critic_loss`, `scale`, `imagined_return` and `entropy` series of the training record; all 36 tensors of the world model's `state_dict`; the actor, critic and slow-critic tensors of all five actor-critic checkpoints (`torch.equal`); and the entry record's `imagined`, `posterior`, `counts`, `c_R`, `c_C`, `position_gap_closed_h15` and `verdict`. Seed-0 gate point +227.39 and every rung's mean are therefore Stage B's, to the printed precision.
- **Differs, only in provenance fields:** `git_sha` (`d674ab7`/`0a9506d`/`03be77b` against `e235d07`); `seconds` and its derived rates (`seconds_per_update`, `seconds_per_episode`): world 12,820.69 against 6,017.18; actor-critic 3,967.13 (Stage B) against 3,743.40; live 338.28 against 362.96; and file paths (`world_model`, `checkpoints`, `agent_dir` in the Stage B and Stage C agent records; `checkpoint` in the entry records). Re-derive: `[k for k in A if A[k] != B[k]]` and the same for the training records print only those keys (`world_model, agent_dir, git_sha, seconds, seconds_per_episode` for the live records; `checkpoints, seconds, seconds_per_update, world_model, git_sha` for the training records).
- **What this shows:** on this machine, retraining with the Task 9 code in place reproduces Stage A's world model and Stage B's agent bit for bit, so reusing Stage A's cell would have given the same cell. It does not show that training is deterministic in general (one cell, one device, one pair of runs), and it says nothing about the other eight cells, which have no earlier counterpart.

### Prerequisites and ruling R10

| step | record | value |
|---|---|---|
| `random_vit` cache | `runlogs/m4c_cache_rv.log` | `features_cached=1200 elapsed_s=140`, 395 frames/s, mps |
| `pixel_ae` autoencoder on `basic` | `runs/m2_basic.log`, `runs/m2_basic/history_cnn.json` | 20,000 steps, `--seq-len 1` (command in the plan, Step 5), 2,861.74 s (`seconds`), mean of first 20 losses 0.08894, of last 20 0.00072 (`python3 -c "import json,statistics as s;l=json.load(open('runs/m2_basic/history_cnn.json'))['loss'];print(len(l),round(s.mean(l[:20]),5),round(s.mean(l[-20:]),5))"` prints `20000 0.08894 0.00072`) |
| `pixel_ae` cache | `runlogs/m4c_cache_pa.log` | `features_cached=1200 elapsed_s=18`, 3,022 frames/s, mps |

**R10: the dinov2 cache was kept, not cleared.** Plan Step 3 says to run `cache_features.py --backbone dinov2 --clear` after the `frozen_ssl` arm "to free the cache". The controller did not (ruling R10): 20 GB was free, a second ~2.7 GB ViT cache fit with margin, and nothing was deleted that was not needed; R10 says the Step 4 check of at least 5 GB free would be made first. Evidence at write time: `ls data/basic | sed 's/ep_.*_len[0-9]*\.//' | sort | uniq -c` prints 1200 each of `features.npy` (dinov2), `features_pixel_ae.npy`, `features_random_vit.npy` and `npz`; the dinov2 files are 2.53 GiB in total and their modification times run 2026-10-10 23:10:21 to 2026-10-11 00:08:28, i.e. Stage A's write, untouched; the `random_vit` files date 2026-10-12 02:30:10 to 02:32:30 (2.53 GiB) and the `pixel_ae` files 2026-10-12 20:38:09 to 20:38:27 (0.21 GiB). `df -h .` prints 15 GiB available at write time. The risk R10 named, disk running short during the `random_vit` cache, did not occur (1,200 episodes cached, exit 0: ledger `Task 15: step 4`). Each arm's cache was written once, on mps, as §6.1 requires, and the three arms read different suffix files, so keeping the dinov2 files cannot have changed any other arm's inputs. The cost is 2.53 GiB left on disk.

### The nine cells

Records: entry `runs/m4c_entry/entry_<arm>_seed<n>.json`; agent `runs/m4c_agent/<arm>_seed<n>/train_<arm>_seed<n>.json` and `live_<arm>_seed<n>.json`; world model `runs/m4c_world/train_<arm>_seed<n>.json`. Each live record holds 100 evaluation episodes on seeds 1,000,000 to 1,000,099, the same seeds and the same `random` list (identical returns) in all nine files. The score is the mean `return` of the 100 episodes at update 10,000, sample mode. The gate and control columns are `paired_gate` against `random` (mean of the per-seed difference, 95% percentile interval from 2,000 resamples at a fixed seed; `src/mbfps/eval/live.py`). Entry intervals are 95% percentile intervals from 2,000 episode-clustered resamples of the 240 held-out episodes, imagined horizon 15 (spec §4.5); `ENTRY_PASS` requires `ci_low` > 0 for both. All nine records carry the same `counts` (240 episodes, 10,220 starts, 131,553 slots, 144 kills, 144 terminations) and `c_R` = -44.510, `c_C` = 0.9885. Re-derive (this script prints the table below row by row; its last block prints the §6.2 comparisons):

```python
import json, itertools, numpy as np
from mbfps.eval.live import paired_gate
ARMS = ("frozen_ssl", "pixel_ae", "random_vit"); S = {}
for arm in ARMS:
    for s in range(3):
        d = f"runs/m4c_agent/{arm}_seed{s}"
        L = json.load(open(f"{d}/live_{arm}_seed{s}.json")); T = json.load(open(f"{d}/train_{arm}_seed{s}.json"))
        E = json.load(open(f"runs/m4c_entry/entry_{arm}_seed{s}.json")); i = E["imagined"]
        r = [e["return"] for e in L["random"]]; a = [e["return"] for e in L["rungs"]["10000"]["sample"]]
        g = paired_gate(a, r); c = paired_gate([e["return"] for e in L["untrained"]], r); S[(arm, s)] = np.mean(a)
        print(arm, s, E["verdict"], i["return"]["point"], i["return"]["ci_low"], i["return"]["ci_high"],
              i["continue"]["point"], i["continue"]["ci_low"], i["continue"]["ci_high"], np.mean(a),
              g["point"], g["ci_low"], g["ci_high"], g["passed"], c["point"], c["ci_low"], c["ci_high"], c["passed"],
              T["isolation_ok"], T["nonfinite"], L["nonfinite"], E["nonfinite"])
for x, y in itertools.combinations(ARMS, 2):
    xs, ys = [S[(x, s)] for s in range(3)], [S[(y, s)] for s in range(3)]
    print(x, y, [(i, j, "X" if xs[i] > ys[j] else "Y") for i in range(3) for j in range(3)])
```

| cell | entry verdict | Reading R, `R2_return` imagined: point [95%] | Reading C, `brier_skill` imagined: point [95%] | score | gate: point [95%] | control: point [95%] | isolation_ok |
|---|---|---|---|---|---|---|---|
| frozen_ssl s0 | ENTRY_PASS | +0.4334 [+0.4007, +0.4680] | +0.1190 [+0.0836, +0.1555] | +63.18 | +227.39 [+191.86, +262.69] | +4.35 [-41.91, +50.70] | True |
| frozen_ssl s1 | ENTRY_PASS | +0.7315 [+0.6674, +0.7890] | +0.5116 [+0.3721, +0.6255] | +76.18 | +240.39 [+204.23, +276.14] | -5.59 [-50.80, +40.60] | True |
| frozen_ssl s2 | ENTRY_PASS | +0.7413 [+0.6908, +0.7909] | +0.4595 [+0.3225, +0.5776] | +78.45 | +242.66 [+206.54, +278.19] | -4.66 [-53.06, +43.25] | True |
| pixel_ae s0 | ENTRY_PASS | +0.7192 [+0.6432, +0.7856] | +0.4676 [+0.3055, +0.6042] | +67.73 | +231.94 [+195.08, +267.38] | +8.79 [-35.30, +52.85] | True |
| pixel_ae s1 | ENTRY_PASS | +0.7157 [+0.6170, +0.7945] | +0.4788 [+0.3203, +0.6204] | +64.96 | +229.17 [+193.92, +263.56] | -8.12 [-54.76, +38.36] | True |
| pixel_ae s2 | ENTRY_PASS | +0.7165 [+0.6500, +0.7774] | +0.4637 [+0.3144, +0.5873] | +62.26 | +226.47 [+190.45, +261.35] | -3.65 [-52.35, +45.08] | True |
| random_vit s0 | ENTRY_PASS | +0.6978 [+0.6261, +0.7630] | +0.2800 [+0.0544, +0.4583] | **-94.98** | +69.23 [+29.94, +106.85] | +7.53 [-37.42, +53.53] | True |
| random_vit s1 | ENTRY_PASS | +0.7208 [+0.6608, +0.7718] | +0.4323 [+0.2816, +0.5532] | +73.93 | +238.14 [+201.40, +274.24] | -9.19 [-55.73, +37.29] | True |
| random_vit s2 | ENTRY_PASS | +0.6751 [+0.5930, +0.7479] | +0.3133 [+0.1193, +0.4774] | +76.34 | +240.55 [+205.13, +276.33] | +5.85 [-40.72, +53.58] | True |

The `read.log` columns "R" and "C" are the two `ci_low` values above (rounded to three places, e.g. `R +0.401, C +0.084` for `frozen_ssl` s0). `nonfinite` is `{}` in all 36 records. In every row the gate's `ci_low` is above zero (smallest: +29.94, `random_vit` s0) and the control's `ci_low` is below zero (largest: -35.30, `pixel_ae` s0). Each control interval reaches +37 to +54 at its upper end, so each is a failure to find an effect, not a measurement that the untrained actor equals random (the Stage B reading of its own control carries over). `frozen_ssl` s0's gate, control and every number in its row are the Stage B values (see above). The gate's baseline is the same `random` list in every cell (mean -164.21, 57 of 100 episodes ending in a kill), so the gate says each agent beat uniform random; it does not rank the arms.

**Per arm, the three scores (§6.2's input):**

| arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `frozen_ssl` | +63.18 | +76.18 | +78.45 |
| `pixel_ae` | +67.73 | +64.96 | +62.26 |
| `random_vit` | -94.98 | +73.93 | +76.34 |

Learning curve, sample-mode mean return at each rung, with the 10,000-update kill rate, median length, argmax score and imagined-vs-real gap (reported, not gated; same definitions as Stage B; gap = mean `imagined_return` minus mean `return` of the sample-mode list). Re-derive:

```python
import json, numpy as np
for arm in ("frozen_ssl", "pixel_ae", "random_vit"):
    for s in range(3):
        L = json.load(open(f"runs/m4c_agent/{arm}_seed{s}/live_{arm}_seed{s}.json"))
        sm = {st: L["rungs"][st]["sample"] for st in L["rungs"]}; f = sm["10000"]
        print(arm, s, [np.mean([e["return"] for e in sm[st]]) for st in sm], np.mean([e["killed"] for e in f]),
              np.median([e["length"] for e in f]), np.mean([e["return"] for e in L["rungs"]["10000"]["argmax"]]),
              np.mean([e["imagined_return"] for e in f]) - np.mean([e["return"] for e in f]), np.mean([e["imagined_return"] for e in f]))
```

The table:

| cell | 2,000 | 4,000 | 6,000 | 8,000 | 10,000 | kill rate | median length | argmax at 10,000 | gap at 10,000 (imagined) |
|---|---|---|---|---|---|---|---|---|---|
| frozen_ssl s0 | +65.06 | +61.96 | +62.36 | +63.62 | +63.18 | 1.00 | 9 | +65.26 | -24.85 (+38.33) |
| frozen_ssl s1 | +73.35 | +75.55 | +76.14 | +76.15 | +76.18 | 1.00 | 7 | +75.94 | -0.05 (+76.13) |
| frozen_ssl s2 | +74.83 | +77.44 | +77.77 | +77.96 | +78.45 | 1.00 | 6 | +78.56 | +0.69 (+79.14) |
| pixel_ae s0 | +72.38 | +68.00 | +63.62 | +65.50 | +67.73 | 1.00 | 6 | +63.40 | +6.62 (+74.35) |
| pixel_ae s1 | +66.24 | +60.21 | +67.95 | +69.84 | +64.96 | 0.99 | 7 | +55.77 | +9.08 (+74.04) |
| pixel_ae s2 | +67.01 | +70.27 | +67.42 | +63.40 | +62.26 | 1.00 | 7 | +54.59 | +11.16 (+73.42) |
| random_vit s0 | +70.71 | +72.80 | +43.35 | -100.36 | -94.98 | 0.81 | 37 | -106.32 | **+980.56** (+885.58) |
| random_vit s1 | +73.39 | +73.61 | +74.68 | +75.21 | +73.93 | 1.00 | 7 | +73.57 | -1.73 (+72.20) |
| random_vit s2 | +71.09 | +73.05 | +74.98 | +75.33 | +76.34 | 1.00 | 7 | +77.34 | -1.29 (+75.05) |

(The `random` list kills on 57 of 100 episodes with median length 59 in all nine files.) Eight of the nine curves have a sample-mode mean between +60.21 and +78.45 at every rung from 2,000 on; the ninth is `random_vit` s0 (next section). Five cells (`frozen_ssl` s0, s1, s2; `random_vit` s1, s2) have an argmax score within 2.1 of their sample score at 10,000 (differences +2.08, -0.24, +0.11, -0.36, +1.00); the three `pixel_ae` cells have an argmax score below their sample score by 4.33, 9.19 and 7.67, and `random_vit` s0 by 11.34 (no interval computed, not gated, no explanation tested). The `frozen_ssl` s0 gap (-24.85) differs in size from the other `frozen_ssl` seeds (-0.05, +0.69); that, like the signs elsewhere, is reported, not gated, and subject to the like-for-like caveat Stage B gives for the gap.

### The §6.2 reading, pair by pair

§6.2, in the spec's words: "**Arm X ranks above arm Y** only if every seed of X scores above every seed of Y: 9 of 9 pairwise wins." "Any other pattern reads **`NOT_SEPARATED`** for that pair, which is a legitimate, reportable result." A "win" below is a cell-to-cell comparison of the score (mean live return at update 10,000). The code (`separation` in `scripts/m4_study.py`) also reads `BELOW` when every seed of Y beats every seed of X; no pair shows that either (the smallest win count for either side is 2).

| pair | reading | wins X : Y of 9 | the comparisons that broke 9 of 9 for X above Y |
|---|---|---|---|
| `frozen_ssl` vs `pixel_ae` | **NOT_SEPARATED** | 7 : 2 | `frozen_ssl` s0 (+63.18) is below `pixel_ae` s0 (+67.73) and below `pixel_ae` s1 (+64.96). It is above `pixel_ae` s2 (+62.26) by 0.92. |
| `frozen_ssl` vs `random_vit` | **NOT_SEPARATED** | 6 : 3 | `frozen_ssl` s0 (+63.18) is below `random_vit` s1 (+73.93) and below `random_vit` s2 (+76.34); `frozen_ssl` s1 (+76.18) is below `random_vit` s2 (+76.34) by 0.16. `random_vit` s0 (-94.98) is below all three `frozen_ssl` seeds. |
| `pixel_ae` vs `random_vit` | **NOT_SEPARATED** | 3 : 6 | every `pixel_ae` seed (+67.73, +64.96, +62.26) is below `random_vit` s1 (+73.93) and below `random_vit` s2 (+76.34): six comparisons. The three `pixel_ae` wins are all against `random_vit` s0 (-94.98). |

What §6.2 licenses for each pair: `frozen_ssl` vs `pixel_ae`, `NOT_SEPARATED` licenses no ranking of `frozen_ssl` over `pixel_ae` or of `pixel_ae` over `frozen_ssl`; `frozen_ssl` vs `random_vit`, `NOT_SEPARATED` licenses no ranking of `frozen_ssl` over `random_vit` or of `random_vit` over `frozen_ssl`; `pixel_ae` vs `random_vit`, `NOT_SEPARATED` licenses no ranking of `pixel_ae` over `random_vit` or of `random_vit` over `pixel_ae`. None of the three licenses a statement that the two arms are equal.

Facts about the table that the records support. In the first pair, the two comparisons that break the rule both involve `frozen_ssl` s0; the other seven comparisons, including all six involving `frozen_ssl` s1 and s2, are `frozen_ssl` wins. In the second pair, three of the six `frozen_ssl` wins are against `random_vit` s0, the degraded cell; the three comparisons that break the rule are `frozen_ssl` s0 against `random_vit` s1 and s2, and `frozen_ssl` s1 against `random_vit` s2 (a margin of 0.16). In the third pair, `random_vit` s0 is the only `random_vit` seed that `pixel_ae` beats, so the pair's reading depends on `random_vit` s0 in the sense that without it every comparison would go `random_vit`'s way (6 of 6); that arithmetic is a description of the records, **not a reading**: §6.2 fixes three seeds per arm, and dropping a cell after seeing it is not a pre-registered reading and is not licensed here. The within-arm score ranges (maximum minus minimum) are 15.27 (`frozen_ssl`), 5.47 (`pixel_ae`) and 171.32 (`random_vit`, set by seed 0).

Re-derive: the second print in the script above lists all 27 comparisons as `(i, j, X|Y)` (X = first-named arm's seed `i` score above second-named arm's seed `j` score, Y = the reverse); `grep -A3 'cross-arm' runs/m4c_study/read.log` prints the three readings.

### `random_vit` seed 0 (score -94.98, gate still PASS)

This cell is reported as the records show it. Its score of -94.98 is 157.24 below the next-lowest cell (`pixel_ae` s2, +62.26); the gate is +69.23 [+29.94, +106.85], so it still passes (`ci_low` > 0), because the gate's baseline is `random`'s mean of -164.21 (Stage B notes that random's long truncated episodes set much of that low mean). Re-derive:

```python
import json, numpy as np
from mbfps.eval.live import paired_gate
L = json.load(open("runs/m4c_agent/random_vit_seed0/live_random_vit_seed0.json")); r = [e["return"] for e in L["random"]]
for st in L["rungs"]:
    for m in ("sample", "argmax"):
        l = L["rungs"][st][m]; a = [e["return"] for e in l]; g = paired_gate(a, r); im = np.mean([e["imagined_return"] for e in l])
        print(st, m, np.mean(a), np.std(a), np.mean([e["killed"] for e in l]), np.median([e["length"] for e in l]),
              max(e["length"] for e in l), g["point"], g["ci_low"], g["ci_high"], im, im - np.mean(a))
l = L["rungs"]["10000"]["sample"]; a = np.array([e["return"] for e in l]); le = np.array([e["length"] for e in l]); k = np.array([e["killed"] for e in l])
print(k.sum(), (~k).sum(), sorted(set(a[~k])), a[k].mean(), np.median(le[k]), (a > np.array(r)).sum(), (a < np.array(r)).sum())
T = json.load(open("runs/m4c_agent/random_vit_seed0/train_random_vit_seed0.json")); sc, g0 = np.array(T["scale"]), np.array(T["imagined_return"])
print(np.argmax(sc > 150) + 1, np.argmax(g0 > 150) + 1, np.argmax(g0 > 300) + 1, sc.max(), g0.max())
for lo in range(0, 10000, 500): print(lo + 1, np.median(sc[lo:lo+500]), np.median(T["critic_loss"][lo:lo+500]), np.median(g0[lo:lo+500]), np.median(T["entropy"][lo:lo+500]))
```

**Live record, rung by rung** (`live_random_vit_seed0.json`; sample mode unless stated; `random` for reference: mean -164.21, kill rate 0.57, median length 59):

| update | mean return | kill rate | median length | max length | gate: point [95%] | mean imagined `G0` | imagined minus real |
|---|---|---|---|---|---|---|---|
| 2,000 | +70.71 | 1.00 | 8 | 33 | +234.92 [+197.91, +270.83] | +65.75 | -4.96 |
| 4,000 | +72.80 | 1.00 | 7 | 46 | +237.01 [+200.84, +272.88] | +68.25 | -4.55 |
| 6,000 | +43.35 | 1.00 | 11 | 49 | +207.56 [+173.73, +240.91] | +78.55 | +35.20 |
| 8,000 | -100.36 | 0.81 | 42.5 | 75 | +63.85 [+27.36, +100.05] | +505.02 | +605.38 |
| **10,000 (the gate)** | **-94.98** | 0.81 | 37 | 75 | **+69.23 [+29.94, +106.85]** | +885.58 | +980.56 |
| 10,000, argmax | -106.32 | 0.78 | 42 | 75 | +57.89 [+21.34, +95.46] | | |

Argmax-mode means at 2,000 to 8,000: +74.13, +75.48, +50.79, -99.75 (kill rates 1.00, 1.00, 1.00, 0.81). The study reads update 10,000 only (§6.2); the 2,000 and 4,000 rungs (+70.71, +72.80) are in the record and were not used to pick a checkpoint.

- **Episodes at the gate (10,000, sample):** 81 killed, 19 truncated at the 75-step cap (all 19 have `killed` False). The 19 truncated episodes return -335, -330, -310, -305 or -300 (14 of them exactly -300). The 81 killed episodes have mean return -45.84 and median length 33 (the other eight cells' medians are 6 to 9 over all 100 episodes). Against `random` on the same seed the agent is higher on 54 seeds, lower on 46, equal on none (`frozen_ssl` s0 in Stage B and here: higher on 90, lower on 7, equal on 3). The sample-mode and argmax-mode returns coincide on 20 of 100 episodes.
- **Entry readings** (table above and `entry_random_vit_seed0.json`): `ENTRY_PASS`; Reading R +0.6978 [+0.6261, +0.7630]; Reading C +0.2800 [+0.0544, +0.4583]. Reading C's `ci_low` (+0.0544) is the smallest of the nine cells; its point (+0.2800) is the second smallest, after `frozen_ssl` s0's +0.1190, which scored +63.18 with a stable curve. The posterior references are in the range of the other cells (R +0.9308 against +0.7162 to +0.9493; C +0.8646 against +0.6816 to +0.9258), and `position_gap_closed_h15` is +0.9882, the largest of the nine (+0.7961 to +0.9375 in the other eight). Its world-model `kl_dyn_max` is 3.528, below `random_vit` s1's 4.524 and s2's 9.772. So the entry record sets this cell apart only on Reading C's lower bound (smallest of nine) and on the position ratio (largest of nine), and gates nothing about either.
- **Training record** (`train_random_vit_seed0.json`, 10,000 per-update values for each series; `scale` is the recorded return-scale series and `imagined_return` the per-update start-state lambda-return `G0` of that batch; windows are medians over 500 updates):

| updates | `scale` | critic loss | `G0` | entropy |
|---|---|---|---|---|
| 4,001-4,500 | 102.0 | 12.4 | +65.1 | 0.242 |
| 5,001-5,500 | 145.6 | 1,422.2 | +68.2 | 0.166 |
| 6,001-6,500 | 239.8 | 7,471.3 | +97.2 | 0.083 |
| 7,001-7,500 | 418.5 | 11,413.9 | +278.8 | 0.048 |
| 8,001-8,500 | 394.6 | 6,772.2 | +493.4 | 0.033 |
| 9,501-10,000 | 381.9 | 4,703.8 | +773.9 | 0.030 |

`scale` first exceeds 150 at update 5,281; `G0` first exceeds +150 at update 6,176 and +300 at update 6,997; the run's maxima are `scale` 459.2 and `G0` +888.6 (at update 10,000). The largest real return in any of the 90 rung lists (9 cells x 5 rungs x 2 modes) is +95 and the largest reward in the dataset is 99 with a kill ending the episode (Stage A), so the `G0` values above +100 in this record, and the live mean imagined return of +885.58, are above any return the environment gave in these records. In the other eight cells the maximum `scale` is 103.3 to 133.0 and the maximum `G0` is +44.5 to +80.3 (`max(T['scale'])`, `max(T['imagined_return'])` per cell). The critic loss has single-update spikes well before the live degradation (maximum per-update value 15,200 in updates 1,501-2,000 and 36,161 in 2,001-2,500, while the window medians there are 16.6 and 15.1), so a large single-update critic loss is not by itself a marker of the later change. The other eight cells' per-update critic-loss maxima are 525.6 to 1,268.6; this cell's is 148,042.3.
- **Imagined against real.** The sign and size of the gap move from -4.96 (2,000) to +980.56 (10,000), against -24.85 to +11.16 at 10,000 in the other eight cells. §5.5 names "a large positive gap" as "the signature of an actor exploiting the model's errors"; this cell shows a large positive gap, and its live return fell over the same updates. Per episode at 10,000 the imagined return ranges 633.17 to 1,056.84 (median 901.79) and correlates +0.37 with the real return.

**What the records do not show.** They do not say why this cell degraded: no ablation, no rerun of seed 0, no policy diagnostic at the checkpoints (action frequencies, state visitation), and no comparison of the world model's imagined rollouts against real ones for this cell beyond the entry check were run. The gap is not like-for-like with the real return (Stage B: discounted, critic-bootstrapped, 15-step horizon against an undiscounted episode sum, and sample-mode imagination for both modes), so its size is not a measure of model error. The co-occurrence of the rising imagined return, the rising `scale`, and the falling real return after update ~5,000 is what the records show; the exploitation reading is §5.5's description of that pattern, not a result of this run. One seed of three degraded in one arm; the records do not separate an effect of the arm from an effect of the seed (the agent seed and the world-model seed are the same number in each cell).

### Wall clock per arm

Two numbers per arm, because the machine slept during the run. **Recorded** is the sum of the `seconds` fields (world model, entry check, actor-critic training, live evaluation) of the nine records; in the cells where checkpoint times show no gap it matches the elapsed time between file times (e.g. `random_vit` s0 training: 3,524.48 recorded, 3,526 between the entry record and the training record), and where they show a gap it is smaller than that span by about the gap, so it is read as compute time with sleep excluded (consistent with a clock that does not advance during sleep; the world-model, actor-critic, entry-check, live-evaluation and autoencoder code all time with `time.perf_counter`). **Elapsed** is file modification time to file modification time (`stat`), so it includes any sleep. Re-derive:

```python
import json, os
for arm in ("frozen_ssl", "random_vit", "pixel_ae"):
    tot = {"world": 0, "entry": 0, "train": 0, "live": 0}
    for s in range(3):
        d = f"runs/m4c_agent/{arm}_seed{s}"
        tot["world"] += json.load(open(f"runs/m4c_world/train_{arm}_seed{s}.json"))["seconds"]
        tot["entry"] += json.load(open(f"runs/m4c_entry/entry_{arm}_seed{s}.json"))["seconds"]
        tot["train"] += json.load(open(f"{d}/train_{arm}_seed{s}.json"))["seconds"]
        tot["live"] += json.load(open(f"{d}/live_{arm}_seed{s}.json"))["seconds"]
    print(arm, {k: round(v) for k, v in tot.items()}, round(sum(tot.values())), os.stat(f"{d}/live_{arm}_seed2.json").st_mtime)
```

| arm | world models (3) | entry checks (3) | actor-critic (3 x 10,000 updates) | live evaluation (3 x 1,200 episodes) | recorded total | prerequisite | elapsed |
|---|---|---|---|---|---|---|---|
| `frozen_ssl` | 18,628 s | 107 s | 12,547 s | 1,001 s | 32,283 s (8.97 h) | none (dinov2 cache from Stage A) | at least 40,878 s (11.35 h), 2026-10-11 15:08:19 to 2026-10-12 02:29:36 |
| `random_vit` | 18,549 s | 104 s | 11,679 s | 1,272 s | 31,604 s (8.78 h) | cache 140 s | 62,245 s (17.29 h), 2026-10-12 02:32:30 to 19:49:55 |
| `pixel_ae` | 15,643 s | 111 s | 11,412 s | 387 s | 27,553 s (7.65 h) | autoencoder 2,861.74 s + cache 18 s | 74,968 s (20.82 h), 2026-10-12 20:38:27 to 2026-10-13 17:27:55; 77,855 s (21.63 h) counting from the autoencoder's start, 19:50:20 |

The start of `frozen_ssl` is not recorded (no log has timestamps): it is the seed-0 world model's file time minus its recorded `seconds` (the world-model file is written when training ends), so it is a lower bound on the arm's elapsed time. The `random_vit` and `pixel_ae` starts are the modification times of their cache logs, the step before each arm; the `pixel_ae` autoencoder's start (its log's file time minus its `seconds`) is 25 s after `random_vit`'s last file, so the arms ran back to back. Per cell (recorded seconds): world model `frozen_ssl` 6,017 / 6,063 / 6,548, `random_vit` 5,520 / 7,163 / 5,866, `pixel_ae` 5,087 / 5,228 / 5,328; actor-critic 3,743 / 4,755 / 4,048, 3,524 / 4,136 / 4,018 and 3,809 / 4,032 / 3,570; live 363 / 323 / 315, 669 / 338 / 265 and 125 / 130 / 132 (seeds 0 / 1 / 2). Live evaluation of `pixel_ae` takes about a fifth to a half of the others' time per cell; the records do not say why.

**Machine-sleep gaps.** Checkpoint files are written every 2,000 updates, so their modification times show where a training run stalled (`stat -f '%Sm %N' -t '%m-%d %H:%M:%S' runs/m4c_agent/<arm>_seed<n>/actor_critic_*.pt`). Of the 45 intervals (5 per cell; the first is measured from the entry record), 39 are 668 to 1,160 s; the six that are not:

| cell | interval | file times | gap |
|---|---|---|---|
| `frozen_ssl` s0 | step 8,000 to 10,000 | 10-11 17:36:09 to 18:18:15 | 2,526 s (typical 697 to 713 s) |
| `frozen_ssl` s2 | step 2,000 to 4,000 | 10-12 01:14:39 to 01:46:22 | 1,903 s (typical 751 to 933 s) |
| `random_vit` s2 | step 4,000 to 6,000 | 10-12 17:51:07 to 19:15:54 | 5,087 s, 84.8 min (its other intervals: 769, 796, 680 and 1,094 s); the ledger's "one apparent machine sleep" |
| `pixel_ae` s2 | entry record to step 2,000 | 10-13 16:14:22 to 16:38:10 | 1,428 s (typical 677 to 769 s) |
| `pixel_ae` s1 | entry record to step 2,000 | 10-13 01:43:15 to 09:04:38 | 26,483 s |
| `pixel_ae` s1 | step 2,000 to 4,000 | 10-13 09:04:38 to 12:54:04 | 13,766 s; together about 11 h, the ledger's "09:04 -> 12:54" |

Training records agree: `pixel_ae` s1's `seconds` is 4,032 for a span of 42,268 s from entry record to training record. A longer gap in a stage with no intermediate file cannot be placed inside the stage: taking each stage's mtime span minus its recorded `seconds` (previous file's time to this file's time), the spans exceed their recorded seconds by more than 600 s in `frozen_ssl` s0 training (1,601 s), s0 live (840 s), s1 world (4,407 s), s1 live (670 s), s2 training (1,066 s); `random_vit` s1 world (25,225 s, 7.0 h, between seed 0's live record at 10-12 05:15:04 and seed 1's world model at 14:14:52), s2 world (613 s), s2 training (4,407 s); `pixel_ae` s0 world (1,753 s), s1 world (2,203 s), s1 training (38,236 s), s2 world (4,503 s), s2 training (709 s). These excesses are consistent with sleep but the records do not distinguish sleep from other stalls or from process start-up (the `pixel_ae` s0 world figure includes the time from the cache log's file time to the world-model start). Elapsed minus recorded is 8,595 s for `frozen_ssl`, 30,641 s for `random_vit` and 47,415 s for `pixel_ae`. The elapsed ordering of the arms (11.4, 17.3, 20.8 h) differs from the recorded-seconds ordering (8.97, 8.78 and 7.65 h) by amounts that lie in gaps the records do not attribute; the ledger notes apparent machine sleep, and these gaps are consistent with it, but the records do not show that sleep accounts for all of the excess. Whether `pixel_ae`'s lower recorded total (smaller autoencoder features, 0.21 GiB cache against 2.53 GiB) is the cause was not tested.

### Not established by this run

- **Which arm is better.** `NOT_SEPARATED` for all three pairs; no ranking. It is also not evidence of equality: three seeds per arm, a 9-of-9 rule, no equivalence margin. The descriptive win counts (7:2, 6:3, 3:6) license no ordering.
- **Why `random_vit` s0 degraded**, or whether `random_vit` as an arm is prone to it. One seed of three. The `pixel_ae` vs `random_vit` win count (3:6) depends on its score (the three `pixel_ae` wins are all against it); the other two `random_vit` seeds score +73.93 and +76.34. No diagnosis beyond the records quoted above.
- **That the score is a measure of policy quality beyond `basic`.** All cells pass the gate against uniform random; no scripted or other non-random baseline was evaluated (as in Stage B). The score is a mean over 100 fixed evaluation seeds (1,000,000 to 1,000,099); the gate intervals resample those seeds and do not cover variation of the training seed, which the three-seed rule is the only handle on.
- **The learning curve between 0 and 2,000 updates**, and any claim from rungs other than 10,000 (read by §6.2 only at 10,000); the argmax scores and imagined-vs-real gaps are reported, not gated, and not like-for-like.
- **Spec §10, restated.** M4 does not answer whether the world model tracks a shifting, policy-induced distribution (the dataset is fixed; that is M5); whether the `my_way_home` world models support control (M4 uses them only as §4.6's known-negative control); the M3 exit gate (unmoved and still failing; M4 neither re-reads it nor depends on it); whether the arm ranking transfers to harder scenarios or the online loop (§6.2's reading is about `basic`, a fixed dataset, and 10,000 updates; here it is no ranking at all); or model-free comparison at equal environment steps (M5, governing spec §6).
