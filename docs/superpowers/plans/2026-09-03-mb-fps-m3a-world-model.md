# MB-FPS M3a: World Model and Evaluation Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and prove an RSSM world model and its evaluation harness locally on a single arm, so the 3 arms × 3 seeds study (Plan 4) can be scoped from a measurement rather than an estimate.

**Architecture:** A DreamerV3-style RSSM consumes the 2048-d encoder embedding that M2 already equalises across arms, so every module added here is arm-invariant and `encoders.py` remains the only module that differs between arms. Evaluation probes imagined latents into `privileged_state` space — the only space all three arms share — and brackets every error between a persistence baseline above and an encoder floor below.

**Tech Stack:** Python 3.12, PyTorch (MPS), NumPy. Existing `mbfps` package.

**Spec:** `docs/superpowers/specs/2026-09-03-mb-fps-m3-world-model-design.md`

---

## Global Constraints

- `src/mbfps/models/encoders.py` is the ONLY module that may differ in behaviour between arms.
- `privileged_state` is EVALUATION-ONLY. It must never reach a training tensor or receive gradient.
- `terminated` and `truncated` stay distinct. The continue head targets `terminated` only.
- All arms consume byte-identical `(112, 112, 3)` uint8 frames and emit 2048-d embeddings.
- Feature caches are namespaced per backbone; no arm may read another's.
- Every guard is mutation-tested. A test that cannot fail is worse than no test.
- Mutation harnesses MUST self-check: run with `PYTHONPATH` pointed at the export AND prove the harness works with a known-fatal mutation first. The package is installed editable, so a naive `git archive` export is silently shadowed by the working tree.
- Unattended runs require `caffeinate -dimsu`. `caffeinate -i` is insufficient.
- fp32 throughout (MPS).

**Hyperparameters** (governing spec §3.5, do not vary in this plan):

| Parameter | Value |
|---|---|
| Deterministic state `h` | 512 |
| Stochastic state `z` | 32 categoricals × 32 classes |
| Head MLPs | 2 × 512 |
| Batch / sequence | 16 / 64 |
| KL free bits | 1.0 nat |
| KL scales (dyn / rep) | 0.5 / 0.1 |
| World-model LR | 1e-4 |
| Precision | fp32 |

**Measured facts about the dataset** (verified 2026-09-03, do not re-derive):

- 122 episodes, 59,511 transitions, `data/my_way_home`.
- `n_actions = 6` (observed action ids 0–5).
- `privileged_keys = ('health', 'pos_x', 'pos_y', 'pos_z', 'angle')`.
- **`health` and `pos_z` have EXACTLY ZERO variance** (1 unique value across 40 episodes). They must be excluded from every probe target — R² on a zero-variance target is 0/0.
- **`angle` is in degrees and wraps**: 191 wrap events in 19,463 steps, max single-step delta 352.97°. Probe it as `(sin, cos)`, never as raw degrees.
- Positions are Doom map units, not metres. `pos_x` std ≈ 253, `pos_y` std ≈ 188.

**Latent dimensions** (derived, used everywhere below):

```
H_DIM      = 512                    # deterministic
Z_CATS     = 32; Z_CLASSES = 32     # stochastic
Z_DIM      = 32 * 32 = 1024         # flattened one-hot
LATENT_DIM = H_DIM + Z_DIM = 1536   # what heads and probes consume
EMBED_DIM  = 2048                   # encoder output, the prediction target
```

---

## File Structure

| File | Responsibility |
|---|---|
| `src/mbfps/utils/seeding.py` (modify) | add `fork_seed` — per-module seeds independent of construction order |
| `src/mbfps/data/split.py` (create) | deterministic episode-level train/val split |
| `src/mbfps/models/rssm.py` (create) | GRU deterministic path, categorical stochastic state, prior/posterior, KL |
| `src/mbfps/models/heads.py` (create) | embedding predictor, reward head, continue head |
| `src/mbfps/training/world_model.py` (create) | training loop and loss assembly |
| `src/mbfps/eval/probe.py` (create) | linear probe latent → privileged state, angular-aware |
| `src/mbfps/eval/rollout.py` (create) | open-loop rollout, persistence, encoder floor, `gap_closed` |
| `scripts/measure_rssm_cost.py` (create) | Task 1 timing probe |
| `scripts/train_world_model.py` (create) | CLI |
| `scripts/eval_rollout.py` (create) | CLI, error-vs-horizon curve |

---

## Task 1: Per-step cost probe

Measures real cost before building on an estimate. M2's carried constraint predicted a 9.6× arm ratio; measured was 1.96×. The stand-in here is shape- and structure-accurate (a GRU stepped `seq_len` times plus MLPs at the real widths), which is what determines cost — weights do not.

**Files:**
- Create: `scripts/measure_rssm_cost.py`
- Test: none (a measurement script, not library code; its output is recorded in the plan)

**Interfaces:**
- Consumes: `mbfps.utils.device.get_device`
- Produces: printed `ms/step` per input kind; recorded in "## Task 1 results" below

- [ ] **Step 1: Write the measurement script**

```python
# scripts/measure_rssm_cost.py
"""Measure real RSSM per-step cost before the model is built.

Plan 4's study is scoped from this number. M2's equivalent estimate predicted a
9.6x arm speed ratio where the measured value was 1.96x, so this exists to keep
the same mistake from setting a multi-hour compute budget.

The stand-in is shape- and structure-accurate: a GRUCell stepped `seq_len`
times at the real widths, plus heads at the real widths. Cost is determined by
tensor shapes and the sequential structure, not by trained weights.
"""

import argparse
import time

import torch
import torch.nn as nn

from mbfps.utils.device import get_device

H_DIM, Z_CATS, Z_CLASSES = 512, 32, 32
Z_DIM = Z_CATS * Z_CLASSES
LATENT_DIM = H_DIM + Z_DIM
EMBED_DIM = 2048
N_ACTIONS = 6


class _Standin(nn.Module):
    """Same shapes and same sequential structure as the real RSSM."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.GRUCell(Z_DIM + N_ACTIONS, H_DIM)
        self.prior = nn.Sequential(nn.Linear(H_DIM, 512), nn.SiLU(), nn.Linear(512, Z_DIM))
        self.post = nn.Sequential(
            nn.Linear(H_DIM + EMBED_DIM, 512), nn.SiLU(), nn.Linear(512, Z_DIM)
        )
        self.emb_head = nn.Sequential(
            nn.Linear(LATENT_DIM, 512), nn.SiLU(), nn.Linear(512, EMBED_DIM)
        )
        self.reward_head = nn.Sequential(
            nn.Linear(LATENT_DIM, 512), nn.SiLU(), nn.Linear(512, 1)
        )
        self.cont_head = nn.Sequential(
            nn.Linear(LATENT_DIM, 512), nn.SiLU(), nn.Linear(512, 1)
        )

    def forward(self, embeddings, actions):
        b, t, _ = embeddings.shape
        h = torch.zeros(b, H_DIM, device=embeddings.device)
        z = torch.zeros(b, Z_DIM, device=embeddings.device)
        outs = []
        for i in range(t):
            h = self.cell(torch.cat([z, actions[:, i]], dim=-1), h)
            _ = self.prior(h)
            logits = self.post(torch.cat([h, embeddings[:, i]], dim=-1))
            probs = torch.softmax(logits.view(b, Z_CATS, Z_CLASSES), dim=-1)
            z = probs.reshape(b, Z_DIM)
            outs.append(torch.cat([h, z], dim=-1))
        latents = torch.stack(outs, dim=1)
        return self.emb_head(latents), self.reward_head(latents), self.cont_head(latents)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    device = get_device(prefer=args.device)
    model = _Standin().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)

    embeddings = torch.randn(args.batch_size, args.seq_len, EMBED_DIM, device=device)
    actions = torch.zeros(args.batch_size, args.seq_len, N_ACTIONS, device=device)
    actions[:, :, 0] = 1.0

    for i in range(3):  # warm up: first steps pay kernel compilation
        loss = sum(o.square().mean() for o in model(embeddings, actions))
        optimiser.zero_grad(); loss.backward(); optimiser.step()
    if device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    for i in range(args.steps):
        loss = sum(o.square().mean() for o in model(embeddings, actions))
        optimiser.zero_grad(); loss.backward(); optimiser.step()
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    ms = 1000 * elapsed / args.steps
    print(f"device={device} batch={args.batch_size} seq_len={args.seq_len}")
    print(f"ms_per_step={ms:.1f}")
    print(f"steps_per_second={1000 / ms:.3f}")
    print(f"hours_for_20k_steps={20_000 * ms / 3_600_000:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and record the number**

```bash
.venv/bin/python scripts/measure_rssm_cost.py --seq-len 64
```

Expected: prints `ms_per_step`, `steps_per_second`, `hours_for_20k_steps`.

- [ ] **Step 3: Run the seq_len sweep**

```bash
for s in 16 32 64; do .venv/bin/python scripts/measure_rssm_cost.py --seq-len $s | grep -E "seq_len|ms_per_step"; done
```

Record all three in "## Task 1 results" at the bottom of this file.

- [ ] **Step 4: Decision gate**

If `hours_for_20k_steps` exceeds **3 hours** for the stand-in, STOP and report before
building further: the 3 × 3 study would exceed 27 GPU-hours even before per-arm encoder
cost, and either `seq_len`, `steps`, or the instance type needs revisiting in Plan 4.
Record the decision either way — a "no action needed" is also a result.

- [ ] **Step 5: Commit**

```bash
git add scripts/measure_rssm_cost.py docs/superpowers/plans/2026-09-03-mb-fps-m3a-world-model.md
git commit -m "feat: measure real RSSM per-step cost before scoping the study"
```

---

## Task 2: `fork_seed` — module init independent of construction order

Generalises the M2 decoder-parity bug. The defect was never about decoders: any module built after the encoder inherits an RNG the encoder has advanced, by 26,382,304 draws for `cnn` against 12,320 for the bottleneck arms. This plan adds four more such modules.

**Files:**
- Modify: `src/mbfps/utils/seeding.py`
- Modify: `src/mbfps/training/autoencoder.py` (fold the existing local fix into the helper)
- Test: `tests/utils/test_seeding.py`

**Interfaces:**
- Produces: `fork_seed(base_seed: int, name: str) -> int`, and `seeded_init(base_seed: int, name: str)` — a context manager that forks the RNG and seeds it from `fork_seed`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/utils/test_seeding.py  (append)
import torch
import torch.nn as nn

from mbfps.utils.seeding import fork_seed, seed_everything, seeded_init


def test_fork_seed_is_deterministic():
    assert fork_seed(0, "decoder") == fork_seed(0, "decoder")


def test_fork_seed_differs_by_name():
    assert fork_seed(0, "decoder") != fork_seed(0, "rssm")


def test_fork_seed_differs_by_base_seed():
    assert fork_seed(0, "decoder") != fork_seed(1, "decoder")


def test_seeded_init_is_independent_of_prior_rng_consumption():
    """The whole point: a module's weights must not depend on what came before.

    This is the M2 decoder-parity bug in miniature -- the CNN arm drew 26,382,304
    values before the decoder was built and the bottleneck arms drew 12,320, so
    the "shared" decoder started from different weights per arm.
    """
    def build(waste: int) -> torch.Tensor:
        seed_everything(0)
        torch.randn(waste)  # simulate a differently-sized encoder
        with seeded_init(0, "decoder"):
            layer = nn.Linear(8, 8)
        return layer.weight.detach().clone()

    assert torch.equal(build(0), build(1_000_000))


def test_seeded_init_restores_the_outer_rng():
    """It must not disturb the stream its caller is using."""
    seed_everything(0)
    before = torch.randn(4)
    seed_everything(0)
    with seeded_init(0, "whatever"):
        nn.Linear(64, 64)
    after = torch.randn(4)
    assert torch.equal(before, after)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/utils/test_seeding.py -q`
Expected: FAIL with `ImportError: cannot import name 'fork_seed'`

- [ ] **Step 3: Implement**

```python
# src/mbfps/utils/seeding.py  (append)
import hashlib
from contextlib import contextmanager

_SEED_SPACE = 2**31


def fork_seed(base_seed: int, name: str) -> int:
    """A stable seed for `name`, derived from `base_seed`.

    Uses blake2b rather than `hash()`, whose string hashing is randomised per
    process by PYTHONHASHSEED -- that would make module init non-reproducible
    across runs, which is the opposite of what this is for.
    """
    digest = hashlib.blake2b(
        name.encode("utf-8"), digest_size=8, key=str(base_seed).encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big") % _SEED_SPACE


@contextmanager
def seeded_init(base_seed: int, name: str):
    """Build a module under an RNG seeded only by `base_seed` and `name`.

    Module construction otherwise inherits whatever the global RNG has already
    produced, so a module's weights depend on how many draws every module built
    before it consumed. That made M2's shared decoder initialise differently for
    the pixel arm than for the feature arms -- an arm-parity violation in a
    module that was supposed to be identical across all three.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fork_seed(base_seed, name))
        yield
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/utils/test_seeding.py -q`
Expected: PASS

- [ ] **Step 5: Fold the M2 decoder fix into the helper**

In `src/mbfps/training/autoencoder.py`, replace the local fork with the helper and delete
`_DECODER_SEED_OFFSET`:

```python
        self.encoder = build_encoder(cfg.encoder)
        # The decoder is shared across arms and must initialise identically for
        # all of them. See seeding.seeded_init for why construction order would
        # otherwise decide its weights.
        with seeded_init(cfg.train.seed, "pixel_decoder"):
            self.decoder = PixelDecoder(
                in_dim=cfg.encoder.embed_dim, depth=cfg.encoder.cnn_depth
            )
```

Add `from mbfps.utils.seeding import seed_everything, seeded_init` to the imports.

- [ ] **Step 6: Verify the M2 parity test still passes**

Run: `.venv/bin/python -m pytest tests/training/test_autoencoder.py -q`
Expected: PASS — in particular `test_decoder_initialises_identically_for_every_arm`
and `test_decoder_init_tracks_the_train_seed`.

> Note: this changes the decoder's *values* (a different seed than
> `seed + 1_000_003`), so M2's checkpoints in `runs/m2_fixed/` are not
> reproducible after this task. That is acceptable — M2's recorded conclusions
> depend on the arms being *mutually* identical, which still holds, and the
> re-run showed decoder init has no measurable effect on outcome.

- [ ] **Step 7: Mutation-test the guard**

This block is the harness every later task's mutation table depends on. Three things it
must get right, each of which silently produced a meaningless result during M2:

1. `git archive HEAD` exports the **last commit**, not the working tree — copy in the files
   this task just wrote, one `cp` per file (`cp a b c d` means "copy a, b, c into directory d").
2. The venv path must be **absolute**; `cd "$WORK"` makes `.venv/bin/python` unresolvable.
3. `PYTHONPATH` must point at the export, or the editable install shadows it and the
   mutated code is never imported.

```bash
REPO=/Users/raphaelchen/Desktop/csgo-bot
PY="$REPO/.venv/bin/python"
WORK=$(mktemp -d)
git -C "$REPO" archive HEAD | tar -x -C "$WORK"
cp "$REPO/src/mbfps/utils/seeding.py"      "$WORK/src/mbfps/utils/seeding.py"
cp "$REPO/tests/utils/test_seeding.py"     "$WORK/tests/utils/test_seeding.py"

# SELF-CHECK: a known-fatal mutation MUST fail. If it passes, the harness is
# shadowed and every mutation result in this plan is worthless.
cp "$WORK/src/mbfps/utils/seeding.py" "$WORK/seeding.orig"
python3 - "$WORK/src/mbfps/utils/seeding.py" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
old = "torch.manual_seed(fork_seed(base_seed, name))"
assert s.count(old) == 1, "anchor missing -- fix the harness, not the test"
open(p, "w").write(s.replace(old, "pass"))
EOF
( cd "$WORK" && PYTHONPATH="$WORK/src" "$PY" -m pytest tests/utils/test_seeding.py -q )
```

Expected: **FAIL**, specifically on `test_seeded_init_is_independent_of_prior_rng_consumption`.

A PASS, or a failure with `ImportError` / `no tests ran` / `file not found`, means the
harness is broken rather than the mutation being caught. Fix it before trusting any
mutation table in this plan. Restore with
`cp "$WORK/seeding.orig" "$WORK/src/mbfps/utils/seeding.py"` between mutations.

- [ ] **Step 8: Commit**

```bash
git add src/mbfps/utils/seeding.py src/mbfps/training/autoencoder.py tests/utils/test_seeding.py
git commit -m "feat: fork_seed makes module init independent of construction order"
```

---

## Task 3: Deterministic episode-level train/val split

M2 had no held-out split and its numbers are recorded as training-set reconstructions. A probe R² fit and reported on training data measures memorisation.

**Files:**
- Create: `src/mbfps/data/split.py`
- Test: `tests/data/test_split.py`

**Interfaces:**
- Consumes: `ReplayBuffer.episode_paths() -> list[Path]`
- Produces: `episode_split(paths: list[Path], val_fraction: float = 0.2, seed: int = 0) -> tuple[list[Path], list[Path]]` returning `(train_paths, val_paths)`.
- Produces: `SequenceLoader(..., paths: list[Path] | None = None)` — restricts sampling to
  `paths`. **Without this the split is inert**: `SequenceLoader` reads
  `buffer.episode_paths()` and has no way to exclude the validation episodes, so the world
  model would train on every episode the probe is later evaluated on.

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_split.py
from pathlib import Path

import pytest

from mbfps.data.split import episode_split

PATHS = [Path(f"ep_{i:06d}_len00100.npz") for i in range(20)]


def test_split_is_deterministic():
    assert episode_split(PATHS, seed=0) == episode_split(PATHS, seed=0)


def test_split_differs_by_seed():
    assert episode_split(PATHS, seed=0) != episode_split(PATHS, seed=1)


def test_split_is_a_partition():
    train, val = episode_split(PATHS, val_fraction=0.25)
    assert sorted(train + val) == sorted(PATHS)
    assert set(train).isdisjoint(val)


def test_val_fraction_is_respected():
    train, val = episode_split(PATHS, val_fraction=0.25)
    assert len(val) == 5


def test_split_does_not_depend_on_input_order():
    """Episode ordering must not silently change which episodes are held out."""
    a = episode_split(PATHS, seed=0)
    b = episode_split(list(reversed(PATHS)), seed=0)
    assert sorted(a[1]) == sorted(b[1])


def test_split_is_identical_across_arms():
    """Arms must be compared on the same held-out episodes, or the comparison
    measures which episodes each arm happened to get."""
    splits = {arm: episode_split(PATHS, seed=0) for arm in ("cnn", "frozen_ssl", "random_vit")}
    assert len({tuple(sorted(v)) for _, v in splits.values()}) == 1


def test_empty_val_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        episode_split(PATHS[:2], val_fraction=0.01)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/data/test_split.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.data.split'`

- [ ] **Step 3: Implement**

```python
# src/mbfps/data/split.py
"""Deterministic held-out split, at episode granularity.

Splitting at window granularity would leak: windows drawn from one episode
overlap in frames, so a window in validation can share pixels with a window in
training and the held-out error would be optimistic.

The split is a pure function of the episode *names* and the seed, not of the
order `episode_paths()` happens to return -- an ordering change would otherwise
silently move episodes across the boundary and invalidate comparisons against
previously recorded numbers.
"""

from pathlib import Path

import numpy as np


def episode_split(
    paths: list[Path], val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[Path], list[Path]]:
    """Partition `paths` into (train, val).

    Args:
        paths: episode files, in any order.
        val_fraction: share held out, rounded down but forced to at least one.
        seed: fixes the partition. The SAME seed must be used for every arm.

    Raises:
        ValueError: if the split cannot yield at least one episode on each side.
    """
    ordered = sorted(paths, key=lambda p: p.name)
    n_val = int(len(ordered) * val_fraction)
    if n_val < 1 or len(ordered) - n_val < 1:
        raise ValueError(
            f"need at least one episode on each side; {len(ordered)} episodes at "
            f"val_fraction={val_fraction} gives {n_val} validation episodes"
        )
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    val_index = set(permutation[:n_val].tolist())
    train = [p for i, p in enumerate(ordered) if i not in val_index]
    val = [p for i, p in enumerate(ordered) if i in val_index]
    return train, val
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_split.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Make the split reachable from the loader**

`episode_split` is inert unless training can honour it. `SequenceLoader` reads
`buffer.episode_paths()` directly, so add an override. In `SequenceLoader.__init__`, add
`paths: list[Path] | None = None` as the final keyword argument and replace

```python
        self._paths: list[Path] = buffer.episode_paths()
```

with

```python
        self._paths: list[Path] = (
            list(paths) if paths is not None else buffer.episode_paths()
        )
```

documenting it as: *restrict sampling to these episodes; used to hold the validation split
out of training -- without it the loader draws from every episode in the buffer and any
held-out evaluation is contaminated.*

```python
# tests/data/test_split.py  (append)
from mbfps.data.buffer import ReplayBuffer
from mbfps.data.loader import SequenceLoader
from tests.data.test_loader import make_episode


def test_loader_restricted_to_paths_never_samples_outside_them(tmp_path):
    """If this fails the split is decorative: the world model would train on
    the very episodes the probe is later evaluated on."""
    buffer = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    for fill in (1, 2, 3, 4):
        buffer.add(make_episode(t=80, fill=fill))
    train, _ = episode_split(buffer.episode_paths(), val_fraction=0.5, seed=0)

    loader = SequenceLoader(buffer, batch_size=8, seq_len=16, seed=0, paths=train)
    allowed = {p.name for p in train}
    for _ in range(20):
        batch = loader.sample()
        for i in range(8):
            name = loader.episode_path(int(batch["episode_index"][i])).name
            assert name in allowed, f"sampled {name}, which is held out"


def test_loader_without_paths_uses_the_whole_buffer(tmp_path):
    buffer = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    for fill in (1, 2, 3, 4):
        buffer.add(make_episode(t=80, fill=fill))
    assert len(SequenceLoader(buffer, seq_len=16)._paths) == 4
```

Run: `.venv/bin/python -m pytest tests/data/test_split.py -q` — expect PASS (9 tests).

- [ ] **Step 6: Mutation-test**

Apply each mutation to a `PYTHONPATH`-isolated export (self-check first, per Task 2 Step 7):

| mutation | must be caught by |
|---|---|
| `ordered = list(paths)` (drop the sort) | `test_split_does_not_depend_on_input_order` |
| `permutation = np.random.permutation(len(ordered))` (unseeded) | `test_split_is_deterministic` |
| `val_index = set(permutation[:n_val + 1].tolist())` | `test_val_fraction_is_respected` |
| drop the `n_val < 1` guard | `test_empty_val_is_rejected` |
| ignore the `paths` argument in `SequenceLoader` | `test_loader_restricted_to_paths_never_samples_outside_them` |

- [ ] **Step 7: Commit**

```bash
git add src/mbfps/data/split.py src/mbfps/data/loader.py tests/data/test_split.py
git commit -m "feat: deterministic episode-level split, honoured by the loader"
```

---

## Task 4: RSSM core — deterministic path and categorical stochastic state

**Files:**
- Create: `src/mbfps/models/rssm.py`
- Test: `tests/models/test_rssm.py`

**Interfaces:**
- Consumes: `mbfps.utils.seeding.seeded_init`
- Produces:
  - `RSSMConfig(h_dim=512, z_cats=32, z_classes=32, embed_dim=2048, n_actions=6, hidden=512)`
  - `RSSM(cfg: RSSMConfig)` with:
    - `initial_state(batch_size, device) -> tuple[Tensor, Tensor]` returning `(h, z)` of shapes `(B, 512)` and `(B, 1024)`
    - `observe(embeddings: (B,T,2048), actions: (B,T) int64, state=None) -> dict` with keys `"h" (B,T,512)`, `"z" (B,T,1024)`, `"prior_logits" (B,T,32,32)`, `"post_logits" (B,T,32,32)`, `"latent" (B,T,1536)`
    - `imagine(actions: (B,T) int64, state) -> dict` with `"h"`, `"z"`, `"prior_logits"`, `"latent"` — no embeddings consumed
  - `LATENT_DIM = 1536`

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_rssm.py
import pytest
import torch

from mbfps.models.rssm import LATENT_DIM, RSSM, RSSMConfig

B, T = 3, 7


@pytest.fixture
def rssm():
    return RSSM(RSSMConfig())


def _batch():
    return (
        torch.randn(B, T, 2048),
        torch.randint(0, 6, (B, T)),
    )


def test_observe_shapes(rssm):
    embeddings, actions = _batch()
    out = rssm.observe(embeddings, actions)
    assert out["h"].shape == (B, T, 512)
    assert out["z"].shape == (B, T, 1024)
    assert out["prior_logits"].shape == (B, T, 32, 32)
    assert out["post_logits"].shape == (B, T, 32, 32)
    assert out["latent"].shape == (B, T, LATENT_DIM)


def test_latent_is_h_concatenated_with_z(rssm):
    embeddings, actions = _batch()
    out = rssm.observe(embeddings, actions)
    torch.testing.assert_close(out["latent"], torch.cat([out["h"], out["z"]], dim=-1))


def test_z_is_a_valid_categorical_sample(rssm):
    """32 one-hot vectors of 32 classes: each group sums to 1 with a single 1."""
    embeddings, actions = _batch()
    z = rssm.observe(embeddings, actions)["z"].view(B, T, 32, 32)
    torch.testing.assert_close(z.sum(-1), torch.ones(B, T, 32))
    assert torch.equal(z.max(-1).values, torch.ones(B, T, 32))


def test_straight_through_gradient_reaches_the_posterior(rssm):
    """A hard one-hot sample has zero gradient; the straight-through estimator
    is what lets the encoder train at all. Without it this grad is None."""
    embeddings, actions = _batch()
    embeddings.requires_grad_(True)
    rssm.observe(embeddings, actions)["z"].sum().backward()
    assert embeddings.grad is not None
    assert embeddings.grad.abs().sum() > 0


def test_imagine_consumes_no_embeddings(rssm):
    """Imagination must run from actions alone -- this is the project's claim."""
    embeddings, actions = _batch()
    state = rssm.initial_state(B, embeddings.device)
    out = rssm.imagine(actions, state)
    assert out["latent"].shape == (B, T, LATENT_DIM)
    assert "post_logits" not in out


def test_imagined_trajectory_depends_on_actions(rssm):
    """A model that ignores actions cannot be used for planning.

    `deterministic=True` is essential here. With sampling, two calls differ by
    RNG alone, so the assertion holds even when the action input is zeroed out
    and the test cannot fail.
    """
    state = rssm.initial_state(B, torch.device("cpu"))
    a = torch.zeros(B, T, dtype=torch.long)
    b = torch.full((B, T), 3, dtype=torch.long)
    ha = rssm.imagine(a, state, deterministic=True)["h"]
    hb = rssm.imagine(b, state, deterministic=True)["h"]
    assert not torch.allclose(ha, hb)


def test_identical_actions_give_identical_deterministic_rollouts(rssm):
    """Guards the guard: if this fails, the test above passes on noise."""
    state = rssm.initial_state(B, torch.device("cpu"))
    a = torch.zeros(B, T, dtype=torch.long)
    first = rssm.imagine(a, state, deterministic=True)["h"]
    second = rssm.imagine(a, state, deterministic=True)["h"]
    torch.testing.assert_close(first, second)


def test_deterministic_state_carries_history(rssm):
    """h at step t must differ when the actions BEFORE t differed."""
    state = rssm.initial_state(B, torch.device("cpu"))
    a = torch.zeros(B, T, dtype=torch.long)
    b = a.clone()
    b[:, 0] = 5  # differ only at the first step
    ha = rssm.imagine(a, state, deterministic=True)["h"][:, -1]
    hb = rssm.imagine(b, state, deterministic=True)["h"][:, -1]
    assert not torch.allclose(ha, hb), "h does not propagate history"


def test_initial_state_shapes(rssm):
    h, z = rssm.initial_state(B, torch.device("cpu"))
    assert h.shape == (B, 512) and z.shape == (B, 1024)
    assert h.abs().sum() == 0 and z.abs().sum() == 0


def test_init_is_independent_of_prior_rng_consumption():
    """Same guarantee as the shared decoder: construction order must not decide
    weights, or arms with different encoder sizes get different RSSMs."""
    from mbfps.utils.seeding import seed_everything

    def build(waste: int) -> torch.Tensor:
        seed_everything(0)
        torch.randn(waste)
        return next(RSSM(RSSMConfig(), seed=0).parameters()).detach().clone()

    assert torch.equal(build(0), build(500_000))
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/models/test_rssm.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.models.rssm'`

- [ ] **Step 3: Implement**

```python
# src/mbfps/models/rssm.py
"""Recurrent state-space model -- the latent dynamics core (milestone M3).

State is a pair: a deterministic GRU state `h` carrying history, and a
stochastic state `z` of 32 categorical variables over 32 classes each.
Categoricals rather than a Gaussian follow DreamerV3; a hard one-hot sample has
zero gradient, so a straight-through estimator passes the softmax gradient
through, which is what lets the encoder train at all.

`observe` runs the posterior, which sees the encoder embedding. `imagine` runs
the prior only, driven by actions alone -- that asymmetry is the whole point of
the architecture, and the reason the actor in M4 never touches a real frame.

This module is arm-invariant: it consumes the 2048-d embedding that M2
equalises across arms, so `encoders.py` remains the only module that differs.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mbfps.utils.seeding import seeded_init


@dataclass(frozen=True)
class RSSMConfig:
    """Shared byte-for-byte across arms."""

    h_dim: int = 512
    z_cats: int = 32
    z_classes: int = 32
    embed_dim: int = 2048
    n_actions: int = 6
    hidden: int = 512


LATENT_DIM = 512 + 32 * 32
"""What the heads and probes consume: `h` concatenated with flattened `z`."""


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class RSSM(nn.Module):
    """Deterministic GRU path plus a categorical stochastic state."""

    def __init__(self, cfg: RSSMConfig, seed: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.z_dim = cfg.z_cats * cfg.z_classes
        with seeded_init(seed, "rssm"):
            self.cell = nn.GRUCell(self.z_dim + cfg.n_actions, cfg.h_dim)
            self.prior_net = _mlp(cfg.h_dim, cfg.hidden, self.z_dim)
            self.post_net = _mlp(cfg.h_dim + cfg.embed_dim, cfg.hidden, self.z_dim)

    def initial_state(self, batch_size: int, device: torch.device):
        """Zero `(h, z)`."""
        return (
            torch.zeros(batch_size, self.cfg.h_dim, device=device),
            torch.zeros(batch_size, self.z_dim, device=device),
        )

    def _sample(self, logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """One-hot sample with a straight-through gradient.

        `argmax` has zero gradient everywhere, so the backward pass uses the
        softmax probabilities instead: `probs + (onehot - probs).detach()` is
        numerically the one-hot in forward and the softmax in backward.

        `deterministic=True` takes the mode instead of sampling. Evaluation MUST
        use it: with sampling, the same checkpoint and probe produced
        gap_closed@45 of -0.121, +0.319 and -0.428 on three consecutive runs --
        the M3 gate criterion is "gap_closed > 0", so its sign was being decided
        by RNG rather than by the model.
        """
        shaped = logits.view(*logits.shape[:-1], self.cfg.z_cats, self.cfg.z_classes)
        probs = F.softmax(shaped, dim=-1)
        if deterministic:
            index = probs.argmax(dim=-1)
        else:
            index = torch.distributions.Categorical(probs=probs).sample()
        onehot = F.one_hot(index, self.cfg.z_classes).to(probs.dtype)
        return (probs + (onehot - probs).detach()).flatten(-2)

    def _step(self, h: torch.Tensor, z: torch.Tensor, action_onehot: torch.Tensor):
        return self.cell(torch.cat([z, action_onehot], dim=-1), h)

    def _onehot_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return F.one_hot(actions.long(), self.cfg.n_actions).float()

    def observe(self, embeddings, actions, state=None) -> dict[str, torch.Tensor]:
        """Filter: the posterior sees each embedding."""
        b, t, _ = embeddings.shape
        h, z = state if state is not None else self.initial_state(b, embeddings.device)
        actions_onehot = self._onehot_actions(actions)
        hs, zs, priors, posts = [], [], [], []
        for i in range(t):
            h = self._step(h, z, actions_onehot[:, i])
            prior_logits = self.prior_net(h)
            post_logits = self.post_net(torch.cat([h, embeddings[:, i]], dim=-1))
            z = self._sample(post_logits)
            hs.append(h); zs.append(z)
            priors.append(prior_logits); posts.append(post_logits)
        return self._pack(hs, zs, priors, posts)

    def imagine(self, actions, state, deterministic: bool = False) -> dict[str, torch.Tensor]:
        """Roll forward on the prior alone -- no embeddings consumed.

        Pass `deterministic=True` for evaluation; see `_sample`.
        """
        h, z = state
        actions_onehot = self._onehot_actions(actions)
        hs, zs, priors = [], [], []
        for i in range(actions.shape[1]):
            h = self._step(h, z, actions_onehot[:, i])
            prior_logits = self.prior_net(h)
            z = self._sample(prior_logits, deterministic=deterministic)
            hs.append(h); zs.append(z); priors.append(prior_logits)
        return self._pack(hs, zs, priors, None)

    def _pack(self, hs, zs, priors, posts) -> dict[str, torch.Tensor]:
        cats, classes = self.cfg.z_cats, self.cfg.z_classes
        h = torch.stack(hs, dim=1)
        z = torch.stack(zs, dim=1)
        out = {
            "h": h,
            "z": z,
            "latent": torch.cat([h, z], dim=-1),
            "prior_logits": torch.stack(priors, dim=1).unflatten(-1, (cats, classes)),
        }
        if posts is not None:
            out["post_logits"] = torch.stack(posts, dim=1).unflatten(-1, (cats, classes))
        return out
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/models/test_rssm.py -q`
Expected: PASS (9 tests)

- [ ] **Step 5: Mutation-test**

| mutation | must be caught by |
|---|---|
| `return onehot` (drop straight-through) | `test_straight_through_gradient_reaches_the_posterior` |
| `self.cell(torch.cat([z, action_onehot * 0], ...), h)` | `test_imagined_trajectory_depends_on_actions` |
| `h = self._step(torch.zeros_like(h), z, ...)` (drop recurrence) | `test_deterministic_state_carries_history` |
| `z = self._sample(prior_logits)` inside `observe` | `test_straight_through_...` still passes — ADD a test if uncaught |

Record any mutation that survives; an uncaught mutation is a missing test, not an
acceptable result.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/models/rssm.py tests/models/test_rssm.py
git commit -m "feat: RSSM with categorical latents and straight-through gradients"
```

---

## Task 5: KL loss with balancing and free bits

**Files:**
- Modify: `src/mbfps/models/rssm.py`
- Test: `tests/models/test_rssm.py` (append)

**Interfaces:**
- Produces: `kl_loss(post_logits, prior_logits, free_bits=1.0, dyn_scale=0.5, rep_scale=0.1) -> tuple[Tensor, dict[str, float]]` returning `(loss, parts)` where `parts` has `"dyn"` and `"rep"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_rssm.py  (append)
from mbfps.models.rssm import kl_loss


def _logits(b=2, t=3, peak=0.0):
    x = torch.zeros(b, t, 32, 32)
    x[..., 0] = peak
    return x


def test_kl_is_zero_when_distributions_match_and_free_bits_off():
    same = _logits(peak=3.0)
    loss, parts = kl_loss(same, same.clone(), free_bits=0.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert parts["dyn"] == pytest.approx(0.0, abs=1e-6)


def test_kl_is_positive_when_distributions_differ():
    loss, _ = kl_loss(_logits(peak=5.0), _logits(peak=0.0), free_bits=0.0)
    assert loss.item() > 0


def test_free_bits_clamps_small_divergences():
    """Below the floor the KL must not be optimised away -- that is the point:
    it stops posterior collapse to the prior."""
    near = _logits(peak=0.01)
    clamped, _ = kl_loss(near, _logits(peak=0.0), free_bits=1.0)
    unclamped, _ = kl_loss(near, _logits(peak=0.0), free_bits=0.0)
    assert unclamped.item() < clamped.item()


def test_dyn_and_rep_are_weighted_differently():
    """0.5 / 0.1 per the spec: the prior is pulled toward the posterior five
    times as hard as the reverse."""
    post, prior = _logits(peak=4.0), _logits(peak=0.0)
    only_dyn, _ = kl_loss(post, prior, free_bits=0.0, dyn_scale=1.0, rep_scale=0.0)
    only_rep, _ = kl_loss(post, prior, free_bits=0.0, dyn_scale=0.0, rep_scale=1.0)
    both, _ = kl_loss(post, prior, free_bits=0.0, dyn_scale=0.5, rep_scale=0.1)
    assert both.item() == pytest.approx(0.5 * only_dyn.item() + 0.1 * only_rep.item(), rel=1e-5)


def test_kl_balancing_stops_gradients_on_the_right_side():
    """dyn trains the PRIOR only; rep trains the POSTERIOR only. Getting this
    backwards trains the posterior to be uninformative."""
    post = _logits(peak=4.0).requires_grad_(True)
    prior = _logits(peak=0.0).requires_grad_(True)
    loss, _ = kl_loss(post, prior, free_bits=0.0, dyn_scale=1.0, rep_scale=0.0)
    loss.backward()
    assert post.grad.abs().sum() == 0, "dyn term must not train the posterior"
    assert prior.grad.abs().sum() > 0, "dyn term must train the prior"
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/models/test_rssm.py -q -k kl`
Expected: FAIL with `ImportError: cannot import name 'kl_loss'`

- [ ] **Step 3: Implement**

```python
# src/mbfps/models/rssm.py  (append)
def _categorical_kl(logits_q: torch.Tensor, logits_p: torch.Tensor) -> torch.Tensor:
    """KL(q || p) summed over the 32 categorical groups, averaged over B and T."""
    log_q = F.log_softmax(logits_q, dim=-1)
    log_p = F.log_softmax(logits_p, dim=-1)
    per_group = (log_q.exp() * (log_q - log_p)).sum(-1)
    return per_group.sum(-1).mean()


KL_FREE_BITS = 1.0
"""Governing spec 3.5. Below this the KL is not optimised at all.

Measured on this dataset the dyn KL at initialisation is ~0.31 nat, so the prior
is clamped -- and therefore frozen -- until the posterior becomes informative.
That warm-up is intentional, but whether it ENDS here is empirical, which is why
`train_world_model` records `kl_cleared_free_bits`. Clamping per (batch, time)
element does not change this: measured per-element KL spans 0.266-0.352 nat.
"""


def kl_loss(
    post_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    free_bits: float = KL_FREE_BITS,
    dyn_scale: float = 0.5,
    rep_scale: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """KL with DreamerV3 balancing and free bits.

    Two terms with stop-gradients on opposite sides. `dyn` pulls the PRIOR
    toward the posterior; `rep` pulls the POSTERIOR toward the prior, and is
    weighted five times lower so the representation is not dragged toward an
    uninformative dynamics prediction.

    Free bits clamp each term at `free_bits` nats. Below that floor the KL is
    not optimised at all, which is what prevents posterior collapse -- without
    it the cheapest way to cut the loss is to make the posterior carry nothing.
    """
    dyn = _categorical_kl(post_logits.detach(), prior_logits)
    rep = _categorical_kl(post_logits, prior_logits.detach())
    floor = torch.tensor(free_bits, dtype=dyn.dtype, device=dyn.device)
    loss = dyn_scale * torch.maximum(dyn, floor) + rep_scale * torch.maximum(rep, floor)
    return loss, {"dyn": float(dyn.detach()), "rep": float(rep.detach())}
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/models/test_rssm.py -q`
Expected: PASS (14 tests)

- [ ] **Step 5: Mutation-test**

| mutation | must be caught by |
|---|---|
| swap `.detach()` between `dyn` and `rep` | `test_kl_balancing_stops_gradients_on_the_right_side` |
| drop both `.detach()` calls | same test |
| `torch.minimum` instead of `torch.maximum` | `test_free_bits_clamps_small_divergences` |
| `dyn_scale * dyn + rep_scale * rep` (drop free bits) | `test_free_bits_clamps_small_divergences` |

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/models/rssm.py tests/models/test_rssm.py
git commit -m "feat: KL balancing and free bits for the RSSM"
```

---

## Task 6: Prediction heads

The continue head targets `terminated` ONLY. A time-limit truncation is not a terminal state, and conflating them corrupts the bootstrapping M4 depends on. This distinction had zero test coverage until 2026-09-03.

**Files:**
- Create: `src/mbfps/models/heads.py`
- Test: `tests/models/test_heads.py`

**Interfaces:**
- Consumes: `LATENT_DIM` from `mbfps.models.rssm`
- Produces: `WorldModelHeads(embed_dim=2048, hidden=512, seed=0)` with `forward(latent) -> dict` keys `"embedding" (B,T,2048)`, `"reward" (B,T)`, `"continue_logit" (B,T)`; and `continue_target(terminated, truncated) -> Tensor`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_heads.py
import pytest
import torch

from mbfps.models.heads import WorldModelHeads, continue_target
from mbfps.models.rssm import LATENT_DIM

B, T = 3, 5


def test_head_shapes():
    out = WorldModelHeads()(torch.randn(B, T, LATENT_DIM))
    assert out["embedding"].shape == (B, T, 2048)
    assert out["reward"].shape == (B, T)
    assert out["continue_logit"].shape == (B, T)


def test_continue_target_is_one_when_nothing_ended():
    t = torch.zeros(B, T, dtype=torch.bool)
    torch.testing.assert_close(continue_target(t, t), torch.ones(B, T))


def test_continue_target_is_zero_only_on_termination():
    terminated = torch.zeros(B, T, dtype=torch.bool)
    terminated[0, 2] = True
    expected = torch.ones(B, T)
    expected[0, 2] = 0.0
    torch.testing.assert_close(
        continue_target(terminated, torch.zeros_like(terminated)), expected
    )


def test_truncation_does_not_end_the_episode():
    """THE invariant. A time-limit cutoff is not a terminal state: the episode
    did not end, the clock ran out. Training the continue head to say otherwise
    corrupts bootstrapping, which M4 depends on directly."""
    truncated = torch.zeros(B, T, dtype=torch.bool)
    truncated[1, 3] = True
    torch.testing.assert_close(
        continue_target(torch.zeros_like(truncated), truncated), torch.ones(B, T)
    )


def test_terminated_and_truncated_together_still_terminate():
    flag = torch.zeros(B, T, dtype=torch.bool)
    flag[0, 0] = True
    expected = torch.ones(B, T)
    expected[0, 0] = 0.0
    torch.testing.assert_close(continue_target(flag, flag), expected)


def test_heads_init_is_independent_of_prior_rng_consumption():
    from mbfps.utils.seeding import seed_everything

    def build(waste: int) -> torch.Tensor:
        seed_everything(0)
        torch.randn(waste)
        return next(WorldModelHeads(seed=0).parameters()).detach().clone()

    assert torch.equal(build(0), build(250_000))
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/models/test_heads.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.models.heads'`

- [ ] **Step 3: Implement**

```python
# src/mbfps/models/heads.py
"""Prediction heads over the RSSM latent (milestone M3).

Three heads share the latent: the embedding predictor supplies the world
model's main reconstruction signal, and the reward and continue heads supply
what M4's actor-critic will bootstrap from.

Arm-invariant, like the RSSM: these consume the latent, which derives from the
2048-d embedding M2 equalises across arms.
"""

import torch
import torch.nn as nn

from mbfps.models.rssm import LATENT_DIM
from mbfps.utils.seeding import seeded_init


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


def continue_target(terminated: torch.Tensor, truncated: torch.Tensor) -> torch.Tensor:
    """1.0 while the episode continues, 0.0 where it genuinely terminated.

    `truncated` is deliberately unused for the value and is accepted only to
    make the asymmetry explicit at every call site. A time-limit cutoff is not
    a terminal state: the episode did not end, the clock ran out. Treating it
    as terminal teaches the critic that running out of time is as bad as dying,
    which is the classic time-limit bootstrapping bug.
    """
    del truncated  # intentionally not part of the target -- see docstring
    return (~terminated.bool()).to(torch.float32)


class WorldModelHeads(nn.Module):
    """Embedding, reward and continue predictors over the latent."""

    def __init__(self, embed_dim: int = 2048, hidden: int = 512, seed: int = 0) -> None:
        super().__init__()
        with seeded_init(seed, "world_model_heads"):
            self.embedding = _mlp(LATENT_DIM, hidden, embed_dim)
            self.reward = _mlp(LATENT_DIM, hidden, 1)
            self.continue_ = _mlp(LATENT_DIM, hidden, 1)

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "embedding": self.embedding(latent),
            "reward": self.reward(latent).squeeze(-1),
            "continue_logit": self.continue_(latent).squeeze(-1),
        }
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/models/test_heads.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Mutation-test**

| mutation | must be caught by |
|---|---|
| `return (~(terminated.bool() \| truncated.bool())).float()` | `test_truncation_does_not_end_the_episode` |
| `return (~truncated.bool()).float()` | `test_continue_target_is_zero_only_on_termination` |
| `return terminated.float()` (inverted) | `test_continue_target_is_one_when_nothing_ended` |

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/models/heads.py tests/models/test_heads.py
git commit -m "feat: world-model heads; continue targets terminated, never truncated"
```

---

## Task 7: World-model training loop

**Files:**
- Create: `src/mbfps/training/world_model.py`
- Create: `scripts/train_world_model.py`
- Test: `tests/training/test_world_model.py`

**Interfaces:**
- Consumes: `RSSM`, `kl_loss`, `WorldModelHeads`, `continue_target`, `build_encoder`, `encoder_backbone`, `encoder_input_kind`, `SequenceLoader`, `ReplayBuffer`, `episode_split`
- Produces:
  - `WorldModel(cfg: Config)` — `nn.Module` with `.encoder`, `.rssm`, `.heads`, and `forward(batch) -> tuple[Tensor, dict[str, float]]` returning `(loss, parts)`
  - `train_world_model(cfg, buffer, out_dir, log_every=100) -> dict` with `"arm"`, `"steps"`, `"loss"`, `"parts"`, `"seconds"`

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_world_model.py
import numpy as np
import pytest
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.envs.protocol import OBS_SHAPE
from mbfps.training.world_model import WorldModel, train_world_model
from mbfps.utils.config import get_config

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def make_episode(t: int, fill: int) -> Episode:
    steps = np.arange(t)
    obs = np.zeros((t + 1, *OBS_SHAPE), dtype=np.uint8)
    obs[:, 0, 0, 0] = np.arange(t + 1, dtype=np.uint8)
    obs[:, 0, 0, 1] = fill
    return Episode(
        obs=obs,
        actions=(steps % 6).astype(np.int32),
        rewards=(steps % 3).astype(np.float32),
        terminated=(steps == t - 1),
        truncated=np.zeros(t, dtype=bool),
        privileged=np.stack(
            [
                np.full(t + 1, 100.0),
                np.arange(t + 1, dtype=np.float32) * 3.0,
                np.arange(t + 1, dtype=np.float32) * -2.0,
                np.zeros(t + 1),
                (np.arange(t + 1) * 7.0) % 360.0,
            ],
            axis=1,
        ).astype(np.float32),
        privileged_keys=KEYS,
        policy_name="random",
        seed=fill,
        scenario="my_way_home",
    )


@pytest.fixture
def buffer(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    for fill in (1, 2, 3, 4):
        buf.add(make_episode(t=40, fill=fill))
    return buf


def tiny(arm: str = "cnn", **kw):
    base = dict(steps=3, batch_size=2, seq_len=6, device="cpu")
    base.update(kw)
    return get_config(arm, **base)


def test_forward_returns_a_scalar_loss_and_named_parts(buffer):
    from mbfps.data.loader import SequenceLoader
    from mbfps.utils.device import to_device

    cfg = tiny()
    model = WorldModel(cfg)
    batch = SequenceLoader(buffer, batch_size=2, seq_len=6, seed=0).sample()
    loss, parts = model(to_device(batch, torch.device("cpu")))
    assert loss.ndim == 0 and torch.isfinite(loss)
    for key in ("embedding", "reward", "continue", "kl_dyn", "kl_rep"):
        assert key in parts


def test_every_trainable_parameter_receives_gradient(buffer):
    """The classic RSSM bug is a detached tensor silently freezing a submodule.

    `prior_net` is excluded and handled by the next two tests. Free bits clamp
    the dyn KL below 1.0 nat, and the measured KL at initialisation on this
    dataset is ~0.31 nat, so `torch.maximum(dyn, 1.0)` is constant and the prior
    legitimately receives no gradient yet. That is the intended DreamerV3
    warm-up -- the prior must not be allowed to collapse the posterior before
    the posterior carries anything -- not a detached tensor.

    Note this is NOT fixed by clamping per (batch, time) element: measured
    per-element KL spans 0.266-0.352 nat, so every element is below the floor.
    """
    from mbfps.data.loader import SequenceLoader
    from mbfps.utils.device import to_device

    model = WorldModel(tiny())
    batch = SequenceLoader(buffer, batch_size=2, seq_len=6, seed=0).sample()
    loss, _ = model(to_device(batch, torch.device("cpu")))
    loss.backward()
    dead = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad
        and not name.startswith("rssm.prior_net")
        and (p.grad is None or p.grad.abs().sum() == 0)
    ]
    assert not dead, f"parameters receiving no gradient: {dead}"


def test_prior_net_trains_once_the_kl_exceeds_free_bits(buffer):
    """The other half: the warm-up must actually end.

    If the prior only ever sees a clamped constant it is never trained, and
    every `imagine()` rollout -- the whole M3 gate and all of M4 -- runs on
    random weights. This constructs the post-warm-up condition directly.
    """
    from mbfps.data.loader import SequenceLoader
    from mbfps.utils.device import to_device

    model = WorldModel(tiny())
    # Drive the posterior far from the prior so the KL clears the floor.
    with torch.no_grad():
        for p in model.rssm.post_net[-1].parameters():
            p.mul_(50.0)
    batch = SequenceLoader(buffer, batch_size=2, seq_len=6, seed=0).sample()
    loss, parts = model(to_device(batch, torch.device("cpu")))
    assert parts["kl_dyn"] > 1.0, (
        f"fixture failed to clear the free-bits floor (kl_dyn={parts['kl_dyn']:.3f}); "
        "this test cannot check what it claims to"
    )
    loss.backward()
    grad = sum(
        p.grad.abs().sum() for p in model.rssm.prior_net.parameters() if p.grad is not None
    )
    assert grad > 0, "prior_net receives no gradient even above the free-bits floor"


def test_history_reports_whether_the_kl_ever_cleared_the_floor(buffer):
    """Makes the warm-up an observable, not an assumption.

    Whether the KL crosses 1.0 nat on THIS dataset at THIS scale is an empirical
    question. If it never does, the prior is frozen for the whole run and the
    result is meaningless -- so the history must carry the evidence either way.
    """
    history = train_world_model(tiny(), buffer, out_dir=None)
    assert "kl_dyn_max" in history
    assert "kl_cleared_free_bits" in history
    assert history["kl_dyn_max"] == pytest.approx(
        max(p["kl_dyn"] for p in history["parts"])
    )


def test_overfits_a_single_fixed_batch(buffer):
    """A failure here is a bug, not a hyperparameter problem."""
    from mbfps.data.loader import SequenceLoader
    from mbfps.utils.device import to_device

    torch.manual_seed(0)
    model = WorldModel(tiny(lr=3e-4))
    batch = to_device(
        SequenceLoader(buffer, batch_size=2, seq_len=6, seed=0).sample(),
        torch.device("cpu"),
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-4)
    first = float(model(batch)[0])
    for _ in range(120):
        loss, _ = model(batch)
        optimiser.zero_grad(); loss.backward(); optimiser.step()
    assert float(loss) < 0.5 * first, f"did not overfit: {first:.4f} -> {float(loss):.4f}"


def test_privileged_state_never_enters_the_loss(buffer):
    """privileged_state is evaluation-only. If it reaches a training tensor the
    whole experiment is silently invalid."""
    from mbfps.data.loader import SequenceLoader
    from mbfps.utils.device import to_device

    model = WorldModel(tiny())
    batch = SequenceLoader(buffer, batch_size=2, seq_len=6, seed=0).sample(
        include_privileged=True
    )
    batch = to_device(batch, torch.device("cpu"))
    batch["privileged"].requires_grad_(True)
    loss, _ = model(batch)
    loss.backward()
    assert batch["privileged"].grad is None or batch["privileged"].grad.abs().sum() == 0


def test_same_seed_reproduces_the_loss_curve(buffer):
    a = train_world_model(tiny(), buffer, out_dir=None)
    b = train_world_model(tiny(), buffer, out_dir=None)
    assert a["loss"] == b["loss"]


def test_different_seeds_diverge(buffer):
    a = train_world_model(tiny(seed=0), buffer, out_dir=None)
    b = train_world_model(tiny(seed=1), buffer, out_dir=None)
    assert a["loss"] != b["loss"]


@pytest.mark.parametrize("arm", ["cnn", "frozen_ssl", "random_vit"])
def test_all_three_arms_train(buffer, arm, tmp_path):
    if arm != "cnn":
        for path in buffer.episode_paths():
            suffix = ".features.npy" if arm == "frozen_ssl" else ".features_random_vit.npy"
            np.save(path.with_suffix(suffix), np.zeros((41, 64, 384), dtype=np.float16))
    history = train_world_model(tiny(arm), buffer, out_dir=None)
    assert history["steps"] == 3
    assert all(np.isfinite(history["loss"]))
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/training/test_world_model.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.training.world_model'`

- [ ] **Step 3: Move `to_device` and widen its whitelist — carefully**

Read the existing `to_device` in `src/mbfps/training/autoencoder.py` before changing it:

```python
        if isinstance(value, np.ndarray) and key in ("obs", "features"):
```

**It is a whitelist, not a converter.** Only `obs` and `features` reach the device. That is
a *structural* barrier for invariant #2 — `privileged` cannot enter a training tensor
because it never leaves the CPU dict. Two consequences:

1. The world model needs `actions`, `rewards`, `terminated` and `truncated`, none of which
   currently survive. Without widening the whitelist, `WorldModel.forward` raises `KeyError`
   on its first line.
2. Replacing the whitelist with a permissive `hasattr(value, "dtype")` converter would move
   `privileged` onto the device and silently dissolve the barrier. **Do not do that.**

Widen the whitelist to exactly what the world model needs, and name the exclusion:

```python
# src/mbfps/utils/device.py  (append)
from typing import Any

import numpy as np

_TRAINING_KEYS = ("obs", "features", "actions", "rewards", "terminated", "truncated")
"""Keys allowed onto the compute device.

A whitelist, deliberately. `privileged` is absent and must stay absent: it is
evaluation-only, and keeping it off the device makes that a structural property
rather than a rule someone has to remember. `episode_index` and `window_start`
are bookkeeping and stay on the host.
"""


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move the whitelisted arrays of a loader batch onto `device`."""
    return {
        key: torch.from_numpy(value).to(device)
        for key, value in batch.items()
        if isinstance(value, np.ndarray) and key in _TRAINING_KEYS
    }
```

In `src/mbfps/training/autoencoder.py`, delete the local definition and re-export:

```python
from mbfps.utils.device import get_device, to_device  # noqa: F401 -- re-exported
```

Add the guard test:

```python
# tests/utils/test_device.py
import numpy as np
import torch

from mbfps.utils.device import to_device


def test_privileged_is_never_moved_to_the_device():
    """Invariant #2 held structurally: privileged cannot reach a training
    tensor because it never leaves the host dict."""
    batch = {
        "obs": np.zeros((1, 2), dtype=np.uint8),
        "privileged": np.zeros((1, 5), dtype=np.float32),
    }
    moved = to_device(batch, torch.device("cpu"))
    assert "privileged" not in moved
    assert "obs" in moved


def test_world_model_inputs_survive():
    batch = {
        k: np.zeros((1, 2), dtype=np.float32)
        for k in ("obs", "features", "actions", "rewards", "terminated", "truncated")
    }
    assert set(to_device(batch, torch.device("cpu"))) == set(batch)
```

Run `.venv/bin/python -m pytest tests/training/ tests/utils/ -q` and confirm M2 still
passes. Use `from mbfps.utils.device import to_device` everywhere below.

- [ ] **Step 4: Implement the model**

```python
# src/mbfps/training/world_model.py
"""World-model training -- milestone M3.

Encoder, RSSM and heads trained jointly on the frozen M1 dataset. The encoder
is the only arm-varying part; the RSSM and heads consume the 2048-d embedding
M2 equalises, so they are byte-identical across arms.

`privileged_state` is never read here. It is evaluation-only and appears only
in `mbfps.eval`.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.loader import SequenceLoader
from mbfps.data.split import episode_split
from mbfps.models.encoders import build_encoder, encoder_backbone, encoder_input_kind
from mbfps.models.heads import WorldModelHeads, continue_target
from mbfps.models.rssm import KL_FREE_BITS, RSSM, RSSMConfig, kl_loss
from mbfps.utils.config import Config
from mbfps.utils.device import get_device
from mbfps.utils.seeding import seed_everything


class WorldModel(nn.Module):
    """Encoder + RSSM + heads."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_kind = encoder_input_kind(cfg.encoder)
        self.encoder = build_encoder(cfg.encoder)
        self.rssm = RSSM(
            RSSMConfig(embed_dim=cfg.encoder.embed_dim), seed=cfg.train.seed
        )
        self.heads = WorldModelHeads(
            embed_dim=cfg.encoder.embed_dim, seed=cfg.train.seed
        )

    def embed(self, batch: dict[str, Any]) -> torch.Tensor:
        """Encode a `(B, T+1, ...)` window down to `(B, T, 2048)`.

        The window carries one more frame than transitions; the RSSM consumes
        the first T, which are the states the T actions were taken from.
        """
        source = batch["obs"] if self.input_kind == "obs" else batch["features"]
        b, t_plus_one = source.shape[0], source.shape[1]
        flat = source.reshape(b * t_plus_one, *source.shape[2:])
        embeddings = self.encoder(flat).view(b, t_plus_one, -1)
        return embeddings[:, :-1]

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        embeddings = self.embed(batch)
        out = self.rssm.observe(embeddings, batch["actions"])
        predictions = self.heads(out["latent"])

        embedding_loss = F.mse_loss(predictions["embedding"], embeddings.detach())
        reward_loss = F.mse_loss(predictions["reward"], batch["rewards"])
        continue_loss = F.binary_cross_entropy_with_logits(
            predictions["continue_logit"],
            continue_target(batch["terminated"], batch["truncated"]),
        )
        kl, kl_parts = kl_loss(out["post_logits"], out["prior_logits"])

        loss = embedding_loss + reward_loss + continue_loss + kl
        parts = {
            "embedding": float(embedding_loss.detach()),
            "reward": float(reward_loss.detach()),
            "continue": float(continue_loss.detach()),
            "kl_dyn": kl_parts["dyn"],
            "kl_rep": kl_parts["rep"],
        }
        return loss, parts
```

- [ ] **Step 5: Implement the training loop**

```python
# src/mbfps/training/world_model.py  (append)
def train_world_model(
    cfg: Config,
    buffer: ReplayBuffer,
    out_dir: Path | None,
    log_every: int = 100,
) -> dict[str, Any]:
    """Train one arm's world model and return its history."""
    from mbfps.utils.device import to_device

    seed_everything(cfg.train.seed)
    device = get_device(prefer=cfg.train.device)
    model = WorldModel(cfg).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    backbone = encoder_backbone(cfg.encoder)
    # Train on the TRAINING episodes only. The split seed is fixed at 0 and is
    # deliberately NOT cfg.train.seed: every arm and every seed must be held out
    # on the same episodes, or the comparison measures which episodes each run
    # happened to get rather than which representation is better.
    train_paths, _ = episode_split(buffer.episode_paths(), val_fraction=0.2, seed=0)
    loader = SequenceLoader(
        buffer,
        batch_size=cfg.train.batch_size,
        seq_len=cfg.train.seq_len,
        seed=cfg.train.seed,
        load_obs=model.input_kind == "obs",
        load_features=model.input_kind == "features",
        feature_backbone=backbone or "dinov2",
        paths=train_paths,
    )

    losses: list[float] = []
    history_parts: list[dict[str, float]] = []
    start = time.perf_counter()
    for step in range(cfg.train.steps):
        loss, parts = model(to_device(loader.sample(), device))
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimiser.step()
        losses.append(float(loss.detach()))
        history_parts.append(parts)
        if log_every and (step + 1) % log_every == 0:
            print(f"[{cfg.arm}] step {step + 1}/{cfg.train.steps} loss={losses[-1]:.5f}")

    kl_dyn_max = max((p["kl_dyn"] for p in history_parts), default=0.0)
    history = {
        "arm": cfg.arm,
        "steps": cfg.train.steps,
        "loss": losses,
        "parts": history_parts,
        "seconds": time.perf_counter() - start,
        # Free bits clamp the dyn KL below KL_FREE_BITS nats, so prior_net is
        # frozen until the posterior becomes informative enough to clear the
        # floor. Whether that ever happens on this dataset is empirical, and a
        # run where it never happens trained no dynamics prior at all -- every
        # imagined rollout would come from random weights. Record it rather
        # than assume it.
        "kl_dyn_max": kl_dyn_max,
        "kl_cleared_free_bits": bool(kl_dyn_max > KL_FREE_BITS),
    }
    if not history["kl_cleared_free_bits"]:
        print(
            f"[{cfg.arm}] WARNING: kl_dyn peaked at {kl_dyn_max:.4f}, never clearing "
            f"the {KL_FREE_BITS} nat free-bits floor -- prior_net was never trained, "
            "so imagine() runs on its initialisation. Lower free_bits or train longer."
        )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"arm": cfg.arm, "seed": cfg.train.seed, "state_dict": model.state_dict()},
            out_dir / f"world_model_{cfg.arm}_seed{cfg.train.seed}.pt",
        )
    return history
```

- [ ] **Step 6: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/training/test_world_model.py -q`
Expected: PASS (9 tests)

- [ ] **Step 7: Write the CLI**

```python
# scripts/train_world_model.py
"""Train one arm's world model (milestone M3)."""

import argparse
from pathlib import Path

from mbfps.data.buffer import ReplayBuffer
from mbfps.training.world_model import train_world_model
from mbfps.utils.config import ARMS, get_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--out", type=Path, default=Path("runs/m3"))
    args = parser.parse_args()

    cfg = get_config(
        args.arm,
        steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    history = train_world_model(cfg, buffer, out_dir=args.out)
    n = min(20, len(history["loss"]))
    print(f"arm={cfg.arm} seed={cfg.train.seed} steps={history['steps']}")
    print(f"seconds={history['seconds']:.0f}")
    print(f"steps_per_second={history['steps'] / history['seconds']:.2f}")
    print(f"loss_first{n}={sum(history['loss'][:n]) / n:.5f}")
    print(f"loss_last{n}={sum(history['loss'][-n:]) / n:.5f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Smoke-test the CLI on real data**

```bash
.venv/bin/python -u scripts/train_world_model.py --arm frozen_ssl --steps 20 --seq-len 16 --out runs/m3_smoke
```

Expected: prints `steps_per_second` and both loss summaries; no exception.

- [ ] **Step 9: Mutation-test**

| mutation | must be caught by |
|---|---|
| `continue_target(batch["truncated"], batch["terminated"])` (swapped) | `test_all_three_arms_train` will NOT catch it — verify `tests/models/test_heads.py` covers it, else add an assertion here |
| `embeddings.detach()` → `embeddings` in `embedding_loss` | none yet — ADD a test that the embedding target is detached |
| `loss = embedding_loss + reward_loss + continue_loss` (drop KL) | `test_every_trainable_parameter_receives_gradient` (prior_net goes dead) |
| `out["latent"].detach()` before the heads | `test_every_trainable_parameter_receives_gradient` |

Any mutation that survives is a missing test. Add it before committing.

- [ ] **Step 10: Commit**

```bash
git add src/mbfps/utils/device.py src/mbfps/training/autoencoder.py src/mbfps/training/world_model.py scripts/train_world_model.py tests/training/test_world_model.py
git commit -m "feat: world-model training loop with gradient-flow and overfit guards"
```

---

## Task 8: Linear probe into privileged-state space

The probe is the ONLY place `privileged_state` is read. It is fit post-hoc on frozen latents with no gradient path to the world model.

**Two measured facts constrain the target set** (verified 2026-09-03, do not re-derive):
`health` and `pos_z` have EXACTLY ZERO variance in `my_way_home`, so R² on them is 0/0.
`angle` wraps (191 wrap events / 19,463 steps), so it is probed as `(sin, cos)`.

**Files:**
- Create: `src/mbfps/eval/__init__.py`, `src/mbfps/eval/probe.py`
- Test: `tests/eval/__init__.py`, `tests/eval/test_probe.py`

**Interfaces:**
- Produces:
  - `PROBE_KEYS = ("pos_x", "pos_y", "angle")`
  - `probe_targets(privileged, keys) -> np.ndarray` of shape `(N, 4)` — `pos_x`, `pos_y`, `sin(angle)`, `cos(angle)`
  - `fit_probe(latents: np.ndarray, targets: np.ndarray, ridge: float = 1.0) -> np.ndarray` returning weights `(LATENT_DIM + 1, 4)`
  - `apply_probe(weights, latents) -> np.ndarray` `(N, 4)`
  - `position_error(pred, true) -> np.ndarray` `(N,)` euclidean, Doom map units
  - `angle_error_degrees(pred, true) -> np.ndarray` `(N,)` in `[0, 180]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_probe.py
import numpy as np
import pytest

from mbfps.eval.probe import (
    PROBE_KEYS,
    angle_error_degrees,
    apply_probe,
    fit_probe,
    position_error,
    probe_targets,
)

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def test_probe_keys_exclude_the_zero_variance_channels():
    """health and pos_z have exactly one unique value in my_way_home, so R^2 on
    them is 0/0. Including them would emit NaN or inflate an averaged score."""
    assert "health" not in PROBE_KEYS
    assert "pos_z" not in PROBE_KEYS
    assert PROBE_KEYS == ("pos_x", "pos_y", "angle")


def test_targets_encode_angle_as_sin_cos():
    privileged = np.array([[100.0, 5.0, -3.0, 0.0, 90.0]], dtype=np.float32)
    targets = probe_targets(privileged, KEYS)
    assert targets.shape == (1, 4)
    np.testing.assert_allclose(targets[0, :2], [5.0, -3.0], atol=1e-6)
    np.testing.assert_allclose(targets[0, 2:], [1.0, 0.0], atol=1e-6)


def test_targets_are_continuous_across_the_wrap():
    """359 deg and 1 deg are two degrees apart. In raw degrees they are 358
    apart, which is what makes a linear probe on degrees wrong."""
    a = probe_targets(np.array([[0, 0, 0, 0, 359.0]], dtype=np.float32), KEYS)
    b = probe_targets(np.array([[0, 0, 0, 0, 1.0]], dtype=np.float32), KEYS)
    assert np.linalg.norm(a[0, 2:] - b[0, 2:]) < 0.05


def test_probe_recovers_an_exact_linear_map():
    rng = np.random.default_rng(0)
    latents = rng.normal(size=(200, 16))
    true_w = rng.normal(size=(17, 4))
    targets = latents @ true_w[:-1] + true_w[-1]
    weights = fit_probe(latents, targets, ridge=1e-8)
    np.testing.assert_allclose(apply_probe(weights, latents), targets, atol=1e-4)


def test_probe_has_an_intercept():
    """Without a bias term a probe cannot represent a constant offset, and Doom
    coordinates are nowhere near zero-centred."""
    latents = np.zeros((10, 4))
    targets = np.full((10, 4), 7.0)
    np.testing.assert_allclose(
        apply_probe(fit_probe(latents, targets, ridge=1e-8), latents), targets, atol=1e-3
    )


def test_position_error_is_euclidean():
    pred = np.array([[3.0, 4.0, 0.0, 1.0]])
    true = np.array([[0.0, 0.0, 0.0, 1.0]])
    np.testing.assert_allclose(position_error(pred, true), [5.0])


def test_angle_error_wraps_the_short_way():
    """1 deg vs 359 deg is an error of 2 deg, not 358."""
    def at(deg):
        r = np.deg2rad(deg)
        return np.array([[0.0, 0.0, np.sin(r), np.cos(r)]])

    np.testing.assert_allclose(angle_error_degrees(at(1.0), at(359.0)), [2.0], atol=1e-4)


def test_angle_error_is_bounded_at_180():
    def at(deg):
        r = np.deg2rad(deg)
        return np.array([[0.0, 0.0, np.sin(r), np.cos(r)]])

    assert angle_error_degrees(at(0.0), at(180.0))[0] == pytest.approx(180.0, abs=1e-4)
    for deg in (0.0, 90.0, 200.0, 350.0):
        assert 0.0 <= angle_error_degrees(at(deg), at(37.0))[0] <= 180.0


def test_unnormalised_sin_cos_predictions_still_give_a_valid_angle():
    """A linear probe's (sin, cos) output does not lie on the unit circle.
    atan2 must be used, not arcsin, or the error is garbage off-circle."""
    pred = np.array([[0.0, 0.0, 0.6, 0.6]])   # 45 deg, norm 0.85
    true = np.array([[0.0, 0.0, 0.0, 1.0]])   # 0 deg
    np.testing.assert_allclose(angle_error_degrees(pred, true), [45.0], atol=1e-4)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.eval'`

- [ ] **Step 3: Implement**

```python
# src/mbfps/eval/probe.py
"""Linear probe from latent state into privileged space -- EVALUATION ONLY.

This is the only module that reads `privileged_state`. It is fit post-hoc on
frozen latents with no gradient path to the world model, on held-out episodes.

Deliberately linear: a nonlinear probe measures the probe's capacity as much as
the representation's content, and the question here is what the latent already
encodes.

Two properties of `my_way_home`, measured rather than assumed:

- `health` and `pos_z` have exactly one unique value across the dataset, so R^2
  on them is 0/0. They are excluded; see PROBE_KEYS.
- `angle` is in degrees and wraps, with 191 wrap events in 19,463 steps. It is
  represented as (sin, cos) so that 359 deg and 1 deg are near each other, and
  errors are recovered with atan2 -- a linear probe's raw (sin, cos) output does
  not lie on the unit circle, so arcsin would be wrong.

Positions are Doom map units, not metres.
"""

import numpy as np

PROBE_KEYS: tuple[str, ...] = ("pos_x", "pos_y", "angle")
"""Privileged channels that actually vary in this scenario."""

TARGET_DIM = 4
"""pos_x, pos_y, sin(angle), cos(angle)."""


def probe_targets(privileged: np.ndarray, keys: tuple[str, ...]) -> np.ndarray:
    """Build `(N, 4)` targets from a `(N, K)` privileged array."""
    index = {k: i for i, k in enumerate(keys)}
    missing = [k for k in PROBE_KEYS if k not in index]
    if missing:
        raise KeyError(f"privileged_keys lacks {missing}; got {keys}")
    radians = np.deg2rad(privileged[:, index["angle"]].astype(np.float64))
    return np.stack(
        [
            privileged[:, index["pos_x"]].astype(np.float64),
            privileged[:, index["pos_y"]].astype(np.float64),
            np.sin(radians),
            np.cos(radians),
        ],
        axis=1,
    )


def fit_probe(latents: np.ndarray, targets: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """Closed-form ridge regression with an intercept.

    Returns weights of shape `(D + 1, 4)`; the last row is the intercept. The
    intercept is not regularised -- penalising it would bias predictions toward
    zero, and Doom coordinates are nowhere near zero-centred.
    """
    x = np.asarray(latents, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    design = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    penalty = ridge * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    gram = design.T @ design + penalty
    return np.linalg.solve(gram, design.T @ y)


def apply_probe(weights: np.ndarray, latents: np.ndarray) -> np.ndarray:
    x = np.asarray(latents, dtype=np.float64)
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1) @ weights


def position_error(predicted: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Euclidean distance in Doom map units."""
    return np.linalg.norm(predicted[:, :2] - true[:, :2], axis=1)


def angle_error_degrees(predicted: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Absolute angular error in [0, 180].

    `atan2` recovers the angle from an arbitrary (sin, cos) pair, which matters
    because a linear probe's output does not lie on the unit circle.
    """
    pred = np.arctan2(predicted[:, 2], predicted[:, 3])
    real = np.arctan2(true[:, 2], true[:, 3])
    difference = np.abs(np.rad2deg(np.arctan2(np.sin(pred - real), np.cos(pred - real))))
    return difference
```

- [ ] **Step 4: Create the package markers and run**

```bash
touch src/mbfps/eval/__init__.py tests/eval/__init__.py
.venv/bin/python -m pytest tests/eval/test_probe.py -q
```

Expected: PASS (9 tests)

- [ ] **Step 5: Mutation-test**

| mutation | must be caught by |
|---|---|
| `PROBE_KEYS = ("pos_x", "pos_y", "pos_z", "angle")` | `test_probe_keys_exclude_the_zero_variance_channels` |
| return raw degrees instead of `(sin, cos)` | `test_targets_are_continuous_across_the_wrap` |
| `np.arcsin(predicted[:, 2])` instead of `atan2` | `test_unnormalised_sin_cos_predictions_still_give_a_valid_angle` |
| drop the intercept column | `test_probe_has_an_intercept` |
| `penalty[-1, -1] = ridge` (regularise the intercept) | `test_probe_has_an_intercept` at larger ridge — verify, and add a test if not caught |
| drop the `np.arctan2(sin, cos)` wrap in the error | `test_angle_error_wraps_the_short_way` |

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval tests/eval
git commit -m "feat: linear probe into privileged space, angle-aware and variance-checked"
```

---

## Task 9: Rollout harness — persistence, RSSM, encoder floor

**Files:**
- Create: `src/mbfps/eval/rollout.py`
- Create: `scripts/eval_rollout.py`
- Test: `tests/eval/test_rollout.py`

**Interfaces:**
- Produces:
  - `RolloutResult` dataclass with `horizon: np.ndarray (H,)`, `rssm_position: (H,)`, `persistence_position: (H,)`, `floor_position: (H,)`, and the three `*_angle` counterparts
  - `gap_closed(persistence, model, floor) -> np.ndarray` — elementwise, NOT clipped
  - `evaluate_rollout(model, val_paths, probe_weights, embedding_probe_weights, context=5, horizon=45, device=None, feature_backbone=None) -> RolloutResult`
    — `probe_weights` maps the 1536-d latent; `embedding_probe_weights` maps the 2048-d
    encoder embedding and supplies the floor (spec §3.2). They are different shapes and
    must be fit separately.
  - `source_for(model, path, episode, feature_backbone) -> np.ndarray` — pixels for the pixel arm, the arm's own cached features otherwise

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_rollout.py
import numpy as np
import pytest

from mbfps.eval.rollout import gap_closed


def test_gap_closed_is_one_when_the_model_reaches_the_floor():
    np.testing.assert_allclose(gap_closed(np.array([10.0]), np.array([2.0]), np.array([2.0])), [1.0])


def test_gap_closed_is_zero_when_the_model_only_matches_persistence():
    np.testing.assert_allclose(gap_closed(np.array([10.0]), np.array([10.0]), np.array([2.0])), [0.0])


def test_gap_closed_is_negative_when_worse_than_persistence():
    """Reported, never clipped: worse-than-persistence is a real finding and
    hiding it behind a floor of zero would make a broken model look adequate."""
    assert gap_closed(np.array([10.0]), np.array([12.0]), np.array([2.0]))[0] < 0


def test_gap_closed_is_nan_when_the_band_collapses():
    """If persistence and the floor coincide there is no headroom to close, and
    the ratio is 0/0. NaN is correct; a silent 0 or 1 would be a lie."""
    result = gap_closed(np.array([5.0]), np.array([5.0]), np.array([5.0]))
    assert np.isnan(result[0])


def test_gap_closed_is_nan_when_the_floor_exceeds_persistence():
    """A negative band inverts the ratio's sign, so a model WORSE than
    persistence would score positive -- and "gap_closed > 0" is the M3 gate.
    Measured on real data the floor does exceed persistence, so this is the
    common case, not a corner."""
    result = gap_closed(np.array([10.0]), np.array([12.0]), np.array([14.0]))
    assert np.isnan(result[0]), "a negative band must not produce a signed score"


def test_gap_closed_is_elementwise_over_the_horizon():
    persistence = np.array([10.0, 20.0])
    model = np.array([6.0, 20.0])
    floor = np.array([2.0, 0.0])
    np.testing.assert_allclose(gap_closed(persistence, model, floor), [0.5, 0.0])
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_rollout.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.eval.rollout'`

- [ ] **Step 3: Implement `gap_closed`**

```python
# src/mbfps/eval/rollout.py
"""Open-loop rollout evaluation -- milestone M3.

A rollout error is meaningless in isolation, so every error is bracketed:

  persistence (upper) -- hold the last context state for the whole horizon.
                         This is what "learned nothing" scores, and it is
                         physically interpretable: the agent never moved, so
                         the error IS the true displacement.
  RSSM                -- the model under test, imagining from actions alone.
  encoder floor (lower) -- probe the encoder embedding of the REAL frame at
                         each step. The best any dynamics model could reach
                         given this encoder, and what separates "the dynamics
                         model is weak" from "the encoder already discarded
                         this information".

The cross-arm number is `gap_closed`, the dimensionless fraction of that band
the model closes. Arms have different encoders and therefore different floors,
so a raw error is not comparable between them and this ratio is.
"""

from dataclasses import dataclass

import numpy as np


def gap_closed(
    persistence: np.ndarray, model: np.ndarray, floor: np.ndarray
) -> np.ndarray:
    """Fraction of the persistence-to-floor band the model closes.

    1.0 means as good as the encoder permits; 0.0 means no better than assuming
    the agent never moved. Negative values are returned, not clipped -- worse
    than persistence is a real result. NaN where the band has zero width, since
    the ratio is genuinely undefined there rather than zero.
    """
    band = persistence - floor
    with np.errstate(divide="ignore", invalid="ignore"):
        # NaN for a non-positive band, not merely a zero one. A negative band
        # means the floor sits ABOVE persistence, which flips the sign of the
        # ratio -- an arm worse than persistence would then score positive, and
        # "gap_closed > 0" is the M3 gate criterion. Measured on real data the
        # band is ~0.6% of the error magnitude and does go negative, so this is
        # the common case rather than a corner.
        result = np.where(band <= 0, np.nan, (persistence - model) / band)
    return result
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_rollout.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Implement the rollout evaluator**

```python
# src/mbfps/eval/rollout.py  (append)
import torch

from mbfps.data.episode import load_episode
from mbfps.eval.probe import (
    angle_error_degrees,
    apply_probe,
    position_error,
    probe_targets,
)


@dataclass
class RolloutResult:
    """Per-horizon-step errors for all three references."""

    horizon: np.ndarray
    rssm_position: np.ndarray
    persistence_position: np.ndarray
    floor_position: np.ndarray
    rssm_angle: np.ndarray
    persistence_angle: np.ndarray
    floor_angle: np.ndarray

    def position_gap_closed(self) -> np.ndarray:
        return gap_closed(self.persistence_position, self.rssm_position, self.floor_position)

    def angle_gap_closed(self) -> np.ndarray:
        return gap_closed(self.persistence_angle, self.rssm_angle, self.floor_angle)


@torch.no_grad()
def evaluate_rollout(
    model,
    val_paths,
    probe_weights: np.ndarray,
    embedding_probe_weights: np.ndarray,
    context: int = 5,
    horizon: int = 45,
    device: torch.device | None = None,
    feature_backbone: str | None = None,
) -> RolloutResult:
    """Condition on `context` real frames, then imagine `horizon` steps.

    Every window in every validation episode that is long enough contributes.
    """
    model.eval()
    device = device or next(model.parameters()).device
    need = context + horizon

    rssm_pos, pers_pos, floor_pos = [], [], []
    rssm_ang, pers_ang, floor_ang = [], [], []

    for path in val_paths:
        episode = load_episode(path)
        if episode.length < need + 1:
            continue
        source = source_for(model, path, episode, feature_backbone)
        for start in range(0, episode.length - need, need):
            window = slice(start, start + need + 1)
            embeddings = model.encoder(
                torch.as_tensor(source[window][:-1]).to(device)
            ).unsqueeze(0)
            actions = torch.as_tensor(
                episode.actions[start : start + need].astype(np.int64)
            ).unsqueeze(0).to(device)

            observed = model.rssm.observe(
                embeddings[:, :context], actions[:, :context]
            )
            state = (observed["h"][:, -1], observed["z"][:, -1])
            imagined = model.rssm.imagine(
                actions[:, context:], state, deterministic=True
            )

            # Floor: probe the ENCODER EMBEDDING of the real future frame, per
            # spec 3.2. The earlier version probed the posterior LATENT here,
            # which is not a lower bound on anything -- it carries per-step
            # sampling noise, and measured on real data it exceeded persistence
            # at 8 of 10 horizon steps, inverting gap_closed's denominator.
            future_embeddings = embeddings[0, context:].cpu().numpy()

            truth = probe_targets(
                episode.privileged[start + context : start + need], episode.privileged_keys
            )
            last_context = observed["latent"][0, -1].cpu().numpy()

            model_pred = apply_probe(probe_weights, imagined["latent"][0].cpu().numpy())
            floor_pred = apply_probe(embedding_probe_weights, future_embeddings)
            pers_pred = apply_probe(
                probe_weights, np.repeat(last_context[None, :], horizon, axis=0)
            )

            rssm_pos.append(position_error(model_pred, truth))
            floor_pos.append(position_error(floor_pred, truth))
            pers_pos.append(position_error(pers_pred, truth))
            rssm_ang.append(angle_error_degrees(model_pred, truth))
            floor_ang.append(angle_error_degrees(floor_pred, truth))
            pers_ang.append(angle_error_degrees(pers_pred, truth))

    if not rssm_pos:
        raise ValueError(
            f"no validation window reached {need + 1} frames; "
            "lower --context/--horizon or check the split"
        )

    stack = lambda xs: np.stack(xs).mean(axis=0)  # noqa: E731
    return RolloutResult(
        horizon=np.arange(1, horizon + 1),
        rssm_position=stack(rssm_pos),
        persistence_position=stack(pers_pos),
        floor_position=stack(floor_pos),
        rssm_angle=stack(rssm_ang),
        persistence_angle=stack(pers_ang),
        floor_angle=stack(floor_ang),
    )


def source_for(model, path, episode, feature_backbone):
    """Pixels for the pixel arm, the arm's own cached features otherwise."""
    if model.input_kind == "obs":
        return episode.obs
    from mbfps.data.loader import feature_suffix

    return np.load(path.with_suffix(feature_suffix(feature_backbone)))
```

- [ ] **Step 6: Add an end-to-end test on real data**

```python
# tests/eval/test_rollout.py  (append)
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import episode_split
from mbfps.eval.probe import fit_probe, probe_targets
from mbfps.eval.rollout import evaluate_rollout
from mbfps.models.encoders import encoder_backbone
from mbfps.training.world_model import WorldModel
from mbfps.utils.config import get_config


@pytest.mark.slow
def test_rollout_produces_the_full_band_on_real_data():
    """The gate artifact: three curves over the horizon, no NaNs, floor below
    persistence. An untrained model need not beat persistence -- this checks the
    harness produces a readable band, not that the model is good."""
    cfg = get_config("random_vit", device="cpu", seed=0)
    model = WorldModel(cfg)
    buffer = ReplayBuffer("data/my_way_home", capacity_transitions=10**9)
    _, val = episode_split(buffer.episode_paths(), val_fraction=0.2, seed=0)

    latents = np.random.default_rng(0).normal(size=(64, 1536))
    targets = np.random.default_rng(1).normal(size=(64, 4))
    weights = fit_probe(latents, targets)

    result = evaluate_rollout(
        model, val[:2], weights, context=5, horizon=10,
        device=torch.device("cpu"), feature_backbone=encoder_backbone(cfg.encoder),
    )
    assert result.rssm_position.shape == (10,)
    assert np.isfinite(result.rssm_position).all()
    assert np.isfinite(result.persistence_position).all()
    assert np.isfinite(result.floor_position).all()

    # The band is REPORTED, not asserted. With the earlier posterior-latent
    # floor it was measured inverting at 8 of 10 horizon steps; the floor is now
    # the encoder-embedding probe per spec 3.2, which should behave better, but
    # "should" is not evidence and this plan does not assert unverified
    # relationships. Task 12 Step 5 records the real numbers.
    band = result.persistence_position - result.floor_position
    inverted = int((band <= 0).sum())
    print(
        f"\nband width: min={band.min():.3f} median={np.median(band):.3f} "
        f"max={band.max():.3f}; floor above persistence at {inverted}/{len(band)} steps"
    )
    if inverted:
        print(
            "NOTE: gap_closed is NaN at those steps by design -- a negative band "
            "would otherwise invert the sign of the M3 gate criterion."
        )
```

- [ ] **Step 7: Run**

Run: `.venv/bin/python -m pytest tests/eval/ -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/mbfps/eval/rollout.py tests/eval/test_rollout.py
git commit -m "feat: open-loop rollout with persistence and encoder-floor references"
```

---

## Task 10: Filtering probe — does `h` carry history?

Spec §3.4 and gate criterion §4.4. The posterior latent has already seen frame *t*, so by
the data-processing inequality it cannot add information *about t* over the encoder
embedding of *t*. What it can add is **history**. If a probe on the posterior latent does
not beat a probe on the raw embedding, the deterministic state is carrying nothing — a
specific, actionable bug that no other test in this plan would surface.

**Files:**
- Modify: `src/mbfps/eval/probe.py`
- Test: `tests/eval/test_probe.py` (append)

**Interfaces:**
- Consumes: `fit_probe`, `apply_probe`, `probe_targets`, `position_error`
- Produces: `probe_r2(weights, latents, targets) -> float` — R² over the 4 target columns,
  and `filtering_comparison(latent_train, embed_train, latent_val, embed_val, targets_train, targets_val) -> dict`
  with keys `"latent_r2"`, `"embedding_r2"`, `"latent_beats_embedding"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_probe.py  (append)
from mbfps.eval.probe import filtering_comparison, probe_r2


def test_r2_is_one_for_a_perfect_linear_fit():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 8))
    w = rng.normal(size=(9, 4))
    y = x @ w[:-1] + w[-1]
    assert probe_r2(fit_probe(x, y, ridge=1e-10), x, y) == pytest.approx(1.0, abs=1e-4)


def test_r2_is_near_zero_for_unrelated_features():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 8))
    y = rng.normal(size=(400, 4))
    assert probe_r2(fit_probe(x, y, ridge=1.0), rng.normal(size=(400, 8)), y) < 0.2


def test_r2_uses_per_column_variance_not_pooled():
    """pos_x has std ~253 and sin(angle) ~0.7. Pooling the variance would let
    the position columns dominate and hide a useless angle prediction."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 6))
    y = np.column_stack([
        x[:, 0] * 250.0,            # large scale, perfectly predictable
        x[:, 1] * 250.0,
        rng.normal(size=200) * 0.7,  # small scale, pure noise
        rng.normal(size=200) * 0.7,
    ])
    r2 = probe_r2(fit_probe(x, y, ridge=1e-8), x, y)
    assert r2 < 0.9, f"pooled variance hid two unpredictable columns (r2={r2:.3f})"


def test_filtering_comparison_detects_history_in_the_latent():
    """Synthetic: the latent carries a lagged signal the embedding lacks, so a
    probe on it must score higher."""
    rng = np.random.default_rng(0)
    n = 400
    signal = rng.normal(size=n)
    lagged = np.roll(signal, 1)
    embed = signal[:, None] * np.ones((1, 4))
    latent = np.column_stack([embed, lagged[:, None] * np.ones((1, 4))])
    targets = np.column_stack([lagged, lagged, lagged, lagged]) + 0.01 * rng.normal(size=(n, 4))

    half = n // 2
    result = filtering_comparison(
        latent[:half], embed[:half], latent[half:], embed[half:],
        targets[:half], targets[half:],
    )
    assert result["latent_r2"] > result["embedding_r2"]
    assert result["latent_beats_embedding"] is True


def test_filtering_comparison_reports_failure_when_the_latent_adds_nothing():
    """If h is dead the latent is just the embedding, and this must say so
    rather than passing quietly."""
    rng = np.random.default_rng(0)
    n = 400
    embed = rng.normal(size=(n, 4))
    latent = np.concatenate([embed, np.zeros((n, 4))], axis=1)  # dead h
    targets = embed + 0.01 * rng.normal(size=(n, 4))
    half = n // 2
    result = filtering_comparison(
        latent[:half], embed[:half], latent[half:], embed[half:],
        targets[:half], targets[half:],
    )
    assert result["latent_beats_embedding"] is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -q -k "r2 or filtering"`
Expected: FAIL with `ImportError: cannot import name 'probe_r2'`

- [ ] **Step 3: Implement**

```python
# src/mbfps/eval/probe.py  (append)
def probe_r2(weights: np.ndarray, latents: np.ndarray, targets: np.ndarray) -> float:
    """Mean R^2 across the four target columns.

    Averaged per column rather than pooled: `pos_x` has std ~253 while
    `sin(angle)` has std ~0.7, so a pooled variance would be dominated by
    position and a completely useless angle prediction would not show up.

    A column with zero variance contributes nothing rather than a NaN -- which
    is why PROBE_KEYS excludes `health` and `pos_z`, but the guard stays in case
    a validation slice happens to be degenerate.
    """
    predicted = apply_probe(weights, latents)
    scores = []
    for column in range(targets.shape[1]):
        truth = targets[:, column]
        denominator = float(((truth - truth.mean()) ** 2).sum())
        if denominator == 0.0:
            continue
        residual = float(((truth - predicted[:, column]) ** 2).sum())
        scores.append(1.0 - residual / denominator)
    if not scores:
        raise ValueError("every target column has zero variance; nothing to score")
    return float(np.mean(scores))


def filtering_comparison(
    latent_train: np.ndarray,
    embedding_train: np.ndarray,
    latent_val: np.ndarray,
    embedding_val: np.ndarray,
    targets_train: np.ndarray,
    targets_val: np.ndarray,
    ridge: float = 1.0,
) -> dict:
    """Does the posterior latent beat the raw embedding at the SAME timestep?

    The posterior has already seen frame t, so it cannot add information about t
    over the embedding of t. What it can add is history, carried in the
    deterministic state `h`. If it does not win here, `h` is inert.
    """
    latent_weights = fit_probe(latent_train, targets_train, ridge=ridge)
    embedding_weights = fit_probe(embedding_train, targets_train, ridge=ridge)
    latent_r2 = probe_r2(latent_weights, latent_val, targets_val)
    embedding_r2 = probe_r2(embedding_weights, embedding_val, targets_val)
    return {
        "latent_r2": latent_r2,
        "embedding_r2": embedding_r2,
        "latent_beats_embedding": bool(latent_r2 > embedding_r2),
    }
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -q`
Expected: PASS (14 tests)

- [ ] **Step 5: Mutation-test**

| mutation | must be caught by |
|---|---|
| pool the variance across columns instead of averaging per column | `test_r2_uses_per_column_variance_not_pooled` |
| `latent_r2 >= embedding_r2` (ties count as a win) | `test_filtering_comparison_reports_failure_when_the_latent_adds_nothing` |
| fit both probes on the validation set | `test_r2_is_near_zero_for_unrelated_features` |
| `return 1.0 - residual / len(truth)` (MSE, not R²) | `test_r2_is_one_for_a_perfect_linear_fit` |

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/probe.py tests/eval/test_probe.py
git commit -m "feat: filtering probe tests whether the deterministic state carries history"
```

---

## Task 11: Golden-rollout regression test

**Files:**
- Test: `tests/eval/test_golden_rollout.py`

**Interfaces:**
- Consumes: `RSSM`, `seeded_init`

- [ ] **Step 1: Write the test**

```python
# tests/eval/test_golden_rollout.py
"""Pins imagined latents against silent regression during refactors.

Not a correctness test -- a change-detector, deliberately. When it fails,
either the change was intended (update the hash, say why in the commit) or a
refactor silently altered the dynamics.
"""

import hashlib

import numpy as np
import torch

from mbfps.models.rssm import RSSM, RSSMConfig
from mbfps.utils.seeding import seed_everything


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def test_imagined_latents_are_stable():
    seed_everything(0)
    rssm = RSSM(RSSMConfig(), seed=0)
    torch.manual_seed(0)
    actions = torch.arange(12).remainder(6).view(1, 12)
    state = rssm.initial_state(1, torch.device("cpu"))
    with torch.no_grad():
        latent = rssm.imagine(actions, state)["latent"].numpy()
    digest = _hash(np.round(latent, 5))
    assert digest == "REPLACE_ON_FIRST_RUN", (
        f"imagined latents changed; got {digest}. If the change was intended, "
        "update this constant and explain why in the commit message."
    )
```

- [ ] **Step 2: Run once to obtain the digest**

Run: `.venv/bin/python -m pytest tests/eval/test_golden_rollout.py -q`
Expected: FAIL, printing the actual digest. Copy it into the assertion, replacing
`REPLACE_ON_FIRST_RUN`.

- [ ] **Step 3: Re-run to confirm it passes and is stable**

```bash
.venv/bin/python -m pytest tests/eval/test_golden_rollout.py -q
.venv/bin/python -m pytest tests/eval/test_golden_rollout.py -q
```

Expected: PASS both times. If the digest differs between runs, sampling is not seeded
deterministically — fix that before proceeding; a flaky golden test is worse than none.

- [ ] **Step 4: Commit**

```bash
git add tests/eval/test_golden_rollout.py
git commit -m "test: golden rollout pins imagined latents against silent regression"
```

---

## Task 12: End-to-end local validation on one arm

**Files:**
- Create: `scripts/eval_rollout.py`
- Modify: this plan (record results)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the CLI**

```python
# scripts/eval_rollout.py
"""Fit the probe on train episodes, evaluate rollouts on held-out ones."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import load_episode
from mbfps.data.split import episode_split
from mbfps.eval.probe import fit_probe, probe_targets
from mbfps.eval.rollout import evaluate_rollout
from mbfps.models.encoders import encoder_backbone
from mbfps.training.world_model import WorldModel
from mbfps.utils.config import ARMS, get_config
from mbfps.utils.device import get_device


@torch.no_grad()
def fit_probes(model, paths, backbone, device, limit=20):
    """Fit BOTH probes on training episodes only.

    Returns `(latent_weights, embedding_weights)`. Two probes, because they map
    different spaces: the 1536-d latent for the model and persistence, and the
    2048-d encoder embedding for the floor (spec 3.2). Fitting one and reusing
    it for the other is a shape error at best and a meaningless floor at worst.
    """
    from mbfps.eval.rollout import source_for

    latents, embeds, targets = [], [], []
    for path in paths[:limit]:
        episode = load_episode(path)
        source = source_for(model, path, episode, backbone)
        embeddings = model.encoder(torch.as_tensor(source[:-1]).to(device)).unsqueeze(0)
        actions = torch.as_tensor(episode.actions.astype(np.int64)).unsqueeze(0).to(device)
        out = model.rssm.observe(embeddings, actions)
        latents.append(out["latent"][0].cpu().numpy())
        embeds.append(embeddings[0].cpu().numpy())
        targets.append(probe_targets(episode.privileged[:-1], episode.privileged_keys))
    y = np.concatenate(targets)
    return (
        fit_probe(np.concatenate(latents), y),
        fit_probe(np.concatenate(embeds), y),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = get_device(prefer=args.device)
    cfg = get_config(args.arm, seed=args.seed, device=args.device)
    model = WorldModel(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("arm") != args.arm:
        raise SystemExit(
            f"checkpoint is for arm {checkpoint.get('arm')!r}, not {args.arm!r}"
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    train, val = episode_split(buffer.episode_paths(), val_fraction=0.2, seed=0)
    backbone = encoder_backbone(cfg.encoder)

    latent_weights, embedding_weights = fit_probes(model, train, backbone, device)
    result = evaluate_rollout(
        model, val, latent_weights, embedding_weights,
        context=args.context, horizon=args.horizon,
        device=device, feature_backbone=backbone,
    )

    pos_gap = result.position_gap_closed()
    print(f"arm={args.arm} seed={args.seed} horizon={args.horizon}")
    print(f"position_error_final   rssm={result.rssm_position[-1]:9.2f} "
          f"persistence={result.persistence_position[-1]:9.2f} "
          f"floor={result.floor_position[-1]:9.2f}  (Doom map units)")
    print(f"angle_error_final_deg  rssm={result.rssm_angle[-1]:6.2f} "
          f"persistence={result.persistence_angle[-1]:6.2f} "
          f"floor={result.floor_angle[-1]:6.2f}")
    print(f"position_gap_closed_final={pos_gap[-1]:.4f}")
    print(f"angle_gap_closed_final={result.angle_gap_closed()[-1]:.4f}")

    # Report the WHOLE curve, not just its last point. The band is narrow and can
    # invert at some horizons and not others, so a single final value hides both
    # the shape and the NaNs.
    band = result.persistence_position - result.floor_position
    print(f"band_width  min={band.min():.3f} median={np.median(band):.3f} max={band.max():.3f}")
    print(f"steps_with_floor_above_persistence={int((band <= 0).sum())}/{len(band)}")
    print(f"gap_closed  finite={int(np.isfinite(pos_gap).sum())}/{len(pos_gap)} "
          f"mean={np.nanmean(pos_gap):+.4f}")
    if not np.isfinite(pos_gap).any():
        print("WARNING: gap_closed is NaN at every horizon step -- the band is "
              "non-positive throughout, so this metric says nothing here. Report "
              "raw errors instead (spec 9, open question 1).")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({k: v.tolist() for k, v in vars(result).items()}))
        print(f"curves={args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Train one arm at small scale**

```bash
caffeinate -dimsu .venv/bin/python -u scripts/train_world_model.py \
  --arm random_vit --steps 2000 --seq-len 32 --seed 0 --out runs/m3_local
```

Record `steps_per_second` — this is the real measurement Plan 4 is scoped from.

- [ ] **Step 3: Evaluate the rollout**

```bash
.venv/bin/python scripts/eval_rollout.py --arm random_vit \
  --checkpoint runs/m3_local/world_model_random_vit_seed0.pt \
  --out runs/m3_local/curves_random_vit.json
```

- [ ] **Step 4: Run the untrained control — the gate that can actually fail**

`gap_closed` against persistence and the floor does NOT establish that training did
anything: an untrained model was measured outscoring a trained one, because at short
horizons both are dominated by probe error. So compare the trained checkpoint against the
SAME architecture at initialisation, evaluated identically:

```bash
.venv/bin/python - <<'EOF'
import numpy as np, torch
from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import episode_split
from mbfps.eval.rollout import evaluate_rollout
from mbfps.models.encoders import encoder_backbone
from mbfps.training.world_model import WorldModel
from mbfps.utils.config import get_config
from mbfps.utils.device import get_device
import scripts.eval_rollout as E

cfg = get_config("random_vit", device="cpu", seed=0)
device = get_device(prefer="cpu")
buffer = ReplayBuffer("data/my_way_home", capacity_transitions=10**9)
train, val = episode_split(buffer.episode_paths(), val_fraction=0.2, seed=0)
backbone = encoder_backbone(cfg.encoder)

scores = {}
for label, load in (("untrained", False), ("trained", True)):
    model = WorldModel(cfg).to(device)
    if load:
        ck = torch.load("runs/m3_local/world_model_random_vit_seed0.pt",
                        map_location=device, weights_only=True)
        model.load_state_dict(ck["state_dict"])
    model.eval()
    lw, ew = E.fit_probes(model, train, backbone, device)
    r = evaluate_rollout(model, val, lw, ew, context=5, horizon=45,
                         device=device, feature_backbone=backbone)
    scores[label] = (r.rssm_position[-1], np.nanmean(r.position_gap_closed()))
    print(f"{label:<10} position_error@45={scores[label][0]:8.2f}  "
          f"mean gap_closed={scores[label][1]:+.4f}")

print()
print("trained beats untrained on raw error:",
      scores["trained"][0] < scores["untrained"][0])
EOF
```

**This is the real gate for Task 12.** The trained model must beat its own initialisation
on raw position error. If it does not, training accomplished nothing and no amount of
baseline arithmetic will disguise that — check `kl_cleared_free_bits` in the training
history first, since a frozen `prior_net` produces exactly this symptom.

A negative `gap_closed` is still an acceptable outcome at 2000 steps; this task gates the
harness and the training signal, not final model quality. What must not happen: an
exception, an all-NaN `gap_closed`, or trained scoring worse than untrained.

- [ ] **Step 5: Record results**

Fill in "## Task 12 results" below with the measured `steps_per_second`, the three final
errors, and `gap_closed`. Note explicitly whether the `gap_closed` denominator
(`persistence - floor`) was large enough for the ratio to be stable — spec §9 open
question 1.

- [ ] **Step 6: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS. Record the count.

- [ ] **Step 7: Commit**

```bash
git add scripts/eval_rollout.py docs/superpowers/plans/2026-09-03-mb-fps-m3a-world-model.md
git commit -m "feat: end-to-end rollout evaluation, local single-arm validation"
```

---

## Exit criteria for this plan

- [ ] `pytest` fully green, with the counts each task states.
- [ ] Task 1's measured per-step cost recorded, and Task 12's real measurement recorded.
- [ ] Every trainable parameter receives gradient (`test_every_trainable_parameter_receives_gradient`).
- [ ] The world model overfits a single fixed batch.
- [ ] `privileged_state` proven absent from every training tensor.
- [ ] The rollout harness emits all three curves with no NaN and `floor <= persistence`.
- [ ] Filtering probe implemented and its comparison reported (spec §4.4).
- [ ] Training provably uses ONLY the training split (`test_loader_restricted_to_paths_never_samples_outside_them`).
- [ ] `kl_cleared_free_bits` recorded; if False, the prior never trained and the run is void.
- [ ] Rollout evaluation is deterministic — two runs of Task 12 give identical `gap_closed`.
- [ ] The trained model beats its own initialisation on raw position error.
- [ ] Band width recorded, with the count of horizon steps where the floor exceeds persistence.
- [ ] Golden-rollout digest pinned and stable across runs.
- [ ] Every mutation listed in each task's mutation table is caught, using a **self-checked** harness.
- [ ] `encoders.py` still the only arm-varying module (grep for `cfg.arm` / `kind` outside it).

---

## Task 1 results

*(fill in during execution)*

| seq_len | ms/step | steps/s | hours for 20k |
|---|---|---|---|
| 16 | | | |
| 32 | | | |
| 64 | | | |

Decision: *(record whether the >3h gate fired and what was decided)*

---

## Task 12 results

*(fill in during execution)*

| quantity | value |
|---|---|
| arm / seed / steps | random_vit / 0 / 2000 |
| steps_per_second (measured) | |
| position error — rssm / persistence / floor | |
| angle error — rssm / persistence / floor | |
| position_gap_closed (final) | |
| band width `persistence - floor` (min/median/max) | |
| horizon steps with floor above persistence | |
| gap_closed finite at how many of 45 steps | |
| ratio stable? (spec §9 Q1) | |
| `kl_cleared_free_bits` | |
| trained beats untrained on raw error? | |
| two eval runs identical? (determinism) | |
| test count | |
