# MB-FPS M2: Representation and the Shared Data Path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three arms' encoders and decoders, train each as a standalone autoencoder on the M1 dataset, and produce the side-by-side reconstruction grids that gate M2 — plus the obs-optional, feature-capable data path that M3 will also depend on.

**Architecture:** A config system selects one of three arms, which differ **only** in `encoders.py`. Arm 1 learns a CNN over pixels; Arms 2 and 3 apply a small learned bottleneck to cached features from a frozen backbone (pretrained DINOv2, or a randomly-initialised ViT of identical architecture). All three emit a 2048-dimensional embedding from byte-identical 112x112 frames. A decoder reconstructs pixels for Arm 1 and provides visualisation-only reconstructions for Arms 2 and 3.

**Tech Stack:** Python 3.12, PyTorch 2.13 (MPS), HuggingFace Transformers 5.16, NumPy 2.x, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-mb-fps-design.md` (sections 3.1-3.5, milestone M2)
**Predecessor:** `docs/superpowers/plans/2026-09-01-mb-fps-m0-m1-env-and-data.md` (complete; 155 tests passing)

---

## Measured facts this plan is designed around

Every number below was measured on this machine against the committed code and dataset. Do not re-derive them, and do not write code that contradicts them.

### Dataset (frozen, produced by M1)

| | |
|---|---|
| Episodes | 122 (`data/my_way_home/`), 119 usable at `seq_len=64` |
| Transitions | 59,511 total; 51,901 distinct T=64 window offsets |
| Obs per episode | 526 frames of 112x112x3 uint8 (~19.8 MB) |
| DINOv2 features per episode | `(526, 64, 384)` float16 (~25.9 MB) |
| Feature cache total | 2.93 GB |

### Training throughput on MPS (batch 16, seq 64)

| Operation | MPS | CPU | Note |
|---|---|---|---|
| CNN encoder fwd+bwd, 1024 frames | 649 ms | 3233 ms | MPS 5.0x faster |
| GRU unroll, T=64, H=512 | 42.3 ms | 46.6 ms | **MPS gives ~nothing** — 64 sequential small kernels |
| MLP head, 1024 x 1024 | 3.2 ms | 5.1 ms | |
| **Arm 1 full step** (CNN + pixel decoder + RSSM) | **1546 ms** | | 0.65 steps/s |
| **Arm 2/3 full step** (bottleneck + RSSM, no CNN) | **161 ms** | | 6.22 steps/s |

**The arms differ 9.6x in cost, and Arm 1 is the budget.** A 3-arm x 3-seed study costs 31 h at 20k steps, 78 h at 50k, 156 h at 100k. That decision belongs to the study milestone, not here — M2 and M3 need one model per arm, not nine.

### Data-path costs

| | |
|---|---|
| `load_all()` with obs | 2.24 GB resident |
| Metadata-only load (skip obs) | 0.07 s, 0.04 GB |
| Memmap all feature files | 0.04 s, 0.04 GB |
| Feature batch 16x65x64x384 from memmap | 101 ms (51 MB) |

Loading costs ~6% of an Arm 1 step but ~63% of an Arm 2/3 step, so **prefetching matters only for the fast arms** — Task 3.

### Platform constraint (relaxed 2026-09-02)

The disk previously had 2.3 GB free against a 2.93 GB feature cache, which would have forced one cache at a time. **Space has since been freed: 20 GB available.** Both caches (5.86 GB total) now fit simultaneously, so arms 2 and 3 can be cached once and kept.

The `--backbone` flag and the `require_free_bytes` guard in Task 4 are still built: caching is the one operation here that can fill a disk, and failing before writing beats dying halfway and leaving a partial cache that loads as a `FileNotFoundError` on some episodes but not others. The guard stays; the one-at-a-time *workflow* is no longer needed.

---

## Global Constraints

- **Python 3.12** in `.venv/`. Every Python and pytest invocation uses `.venv/bin/python`.
- **fp32 for compute; float16 only for cached features on disk.** No autocast on MPS.
- **All arms consume byte-identical 112x112x3 uint8 frames.** 112 = 8 x 14, giving DINOv2 an exact 8x8 patch grid.
- **All arms emit a 2048-dimensional embedding.** Resolution and embedding width are controlled; only the representation differs.
- **`encoders.py` is the ONLY module that differs between arms.** If an arm requires editing `decoders.py`, `rssm.py`, or a trainer, the comparison is no longer controlled — apply the change to all three arms or not at all.
- **`privileged_state` is evaluation-only** and must never reach a training tensor. `SequenceLoader.sample()` defaults `include_privileged=False`; keep it that way.
- **The feature cache must be generated once on one device.** CPU and MPS DINOv2 outputs differ in ~3% of float16 elements, so regenerating per arm would silently break input equality.
- **`my_way_home` coverage is near-saturated at 61 episodes per policy.** More episodes of the same two policies will not add state coverage; do not "improve" results by collecting more.
- Package is `mbfps`; source under `src/`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/mbfps/utils/config.py` | Frozen dataclass configs; one field selects the arm |
| `configs/base.yaml` | Shared hyperparameters |
| `configs/arm_cnn.yaml`, `arm_frozen_ssl.yaml`, `arm_random_vit.yaml` | Per-arm overrides (encoder only) |
| `src/mbfps/data/loader.py` (modify) | `load_obs` / `load_features` flags; memmapped feature windows |
| `src/mbfps/data/prefetch.py` | Background batch prefetching |
| `src/mbfps/data/features.py` (modify) | `--backbone` selection, random-ViT support, disk guard |
| `src/mbfps/models/encoders.py` | `CNNEncoder`, `BottleneckEncoder` — the only arm-varying module |
| `src/mbfps/models/decoders.py` | `PixelDecoder` (Arm 1 training + all-arm visualisation) |
| `src/mbfps/training/autoencoder.py` | M2 standalone autoencoder training loop |
| `scripts/cache_features.py` | Cache features for a chosen backbone over an existing buffer |
| `scripts/train_autoencoder.py` | M2 training entry point |
| `scripts/reconstruction_grid.py` | M2 exit-gate artifact |

### Deliberate deviation from the spec: Python configs, not YAML

Spec section 4 lists `configs/base.yaml` plus three per-arm YAML files. This plan uses frozen dataclasses in `src/mbfps/utils/config.py` instead. (`pyyaml` 6.0.3 is installed, so this is a choice, not a limitation.)

Reason: the study's validity rests on every arm sharing identical settings except the encoder. With YAML, a typo in one arm's file — `batch_size: 61` instead of `16`, or a key that silently does not override anything — produces a broken comparison that still runs and still reports numbers. With a shared base dataclass that arms override **only** in their `encoder` field, accidental divergence is structurally impossible and a typo is an `AttributeError` at import.

This serves the spec's stated intent (a controlled comparison) better than its letter. If a reviewer disagrees, the change is contained to one module.

---

## Task 1: Config system and arm registry

**Files:**
- Create: `src/mbfps/utils/config.py`
- Create: `tests/utils/__init__.py`
- Create: `tests/utils/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `EncoderConfig(kind: str, embed_dim: int = 2048, cnn_depth: int = 32, patch_dim: int = 384, bottleneck_dim: int = 32)`; `TrainConfig(batch_size: int = 16, seq_len: int = 64, lr: float = 1e-4, steps: int = 20_000, seed: int = 0, device: str = "mps")`; `Config(arm: str, encoder: EncoderConfig, train: TrainConfig, data_root: str = "data/my_way_home")`; `ARMS: tuple[str, ...] = ("cnn", "frozen_ssl", "random_vit")`; `get_config(arm: str, **overrides) -> Config`.

- [ ] **Step 1: Write the failing test**

```python
# tests/utils/test_config.py
import dataclasses

import pytest

from mbfps.utils.config import ARMS, Config, EncoderConfig, TrainConfig, get_config


def test_three_arms_are_registered():
    assert ARMS == ("cnn", "frozen_ssl", "random_vit")


def test_each_arm_builds_a_config():
    for arm in ARMS:
        cfg = get_config(arm)
        assert isinstance(cfg, Config)
        assert cfg.arm == arm
        assert cfg.encoder.kind == arm


def test_unknown_arm_rejected():
    with pytest.raises(KeyError, match="unknown arm 'nope'"):
        get_config("nope")


def test_unknown_arm_error_lists_valid_arms():
    with pytest.raises(KeyError, match="cnn"):
        get_config("nope")


def test_all_arms_share_identical_training_settings():
    """The study's validity depends on this. Arms may differ ONLY in encoder."""
    trains = {arm: get_config(arm).train for arm in ARMS}
    first = trains["cnn"]
    for arm, train in trains.items():
        assert train == first, f"arm {arm!r} has different training settings"


def test_all_arms_share_identical_embed_dim():
    """Embedding width is controlled; only the representation differs."""
    dims = {get_config(arm).encoder.embed_dim for arm in ARMS}
    assert dims == {2048}


def test_arms_differ_only_in_encoder_kind():
    configs = {arm: get_config(arm) for arm in ARMS}
    encoders = {arm: dataclasses.asdict(c.encoder) for arm, c in configs.items()}
    baseline = dict(encoders["cnn"])
    for arm, enc in encoders.items():
        differing = {k for k in enc if enc[k] != baseline[k]}
        assert differing <= {"kind"}, f"arm {arm!r} differs beyond kind: {differing}"


def test_configs_are_frozen():
    cfg = get_config("cnn")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.train.batch_size = 32


def test_overrides_apply_to_train_settings():
    cfg = get_config("cnn", steps=5, batch_size=2)
    assert cfg.train.steps == 5
    assert cfg.train.batch_size == 2


def test_override_of_unknown_field_rejected():
    with pytest.raises(TypeError):
        get_config("cnn", not_a_field=1)


def test_seq_len_default_matches_the_dataset():
    """119 of 122 episodes support a 64-step window; a larger default would
    silently discard usable episodes."""
    assert get_config("cnn").train.seq_len == 64
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/utils/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.utils.config'`

- [ ] **Step 3: Write the implementation**

```python
# tests/utils/__init__.py
```

```python
# src/mbfps/utils/config.py
"""Experiment configuration.

The study compares three arms that must differ in exactly one respect: the
space the model predicts in. Everything else -- batch size, sequence length,
learning rate, seed, embedding width -- is held identical.

That invariant is enforced structurally rather than by discipline: every arm is
built from the same `TrainConfig` instance and the same `EncoderConfig`
defaults, overriding only `kind`. A configuration file format would let one
arm's settings drift silently; here a stray field is a TypeError at call time.
"""

import dataclasses
from dataclasses import dataclass, replace

ARMS: tuple[str, ...] = ("cnn", "frozen_ssl", "random_vit")
"""The three arms. `cnn` is the baseline, `frozen_ssl` the treatment,
`random_vit` the control that separates pretraining from target stability."""


@dataclass(frozen=True)
class EncoderConfig:
    """Encoder settings. `kind` is the ONLY field that may differ across arms."""

    kind: str
    embed_dim: int = 2048
    cnn_depth: int = 32
    patch_dim: int = 384
    bottleneck_dim: int = 32


@dataclass(frozen=True)
class TrainConfig:
    """Training settings, shared byte-for-byte across all arms."""

    batch_size: int = 16
    seq_len: int = 64
    lr: float = 1e-4
    steps: int = 20_000
    seed: int = 0
    device: str = "mps"


@dataclass(frozen=True)
class Config:
    """A complete experiment configuration."""

    arm: str
    encoder: EncoderConfig
    train: TrainConfig
    data_root: str = "data/my_way_home"


def get_config(arm: str, **overrides) -> Config:
    """Build the configuration for `arm`, applying `overrides` to TrainConfig.

    Args:
        arm: one of `ARMS`.
        **overrides: field names of `TrainConfig`.

    Raises:
        KeyError: if `arm` is not registered.
        TypeError: if an override names a field `TrainConfig` does not have.
    """
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; available: {list(ARMS)}")
    train = replace(TrainConfig(), **overrides) if overrides else TrainConfig()
    return Config(arm=arm, encoder=EncoderConfig(kind=arm), train=train)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/utils/test_config.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/utils/config.py tests/utils
git commit -m "feat: arm configuration with structurally-enforced parity"
```

---

## Task 2: Obs-optional, feature-capable sequence loading

**Files:**
- Modify: `src/mbfps/data/loader.py`
- Modify: `tests/data/test_loader.py`

**Interfaces:**
- Consumes: `ReplayBuffer` from `mbfps.data.buffer`; `load_episode` from `mbfps.data.episode`.
- Produces: `SequenceLoader(buffer, batch_size=16, seq_len=64, seed=0, load_obs=True, load_features=False, feature_backbone="dinov2")`. When `load_features=True`, `sample()` adds `features (B, T+1, 64, 384) float16` read from the cache for `feature_backbone`. When `load_obs=False`, `sample()` omits `obs` and the loader never reads pixel data from disk. Also `feature_suffix(backbone: str) -> str`.

**Per-backbone suffixes.** Arms 2 and 3 use different frozen backbones, and both caches now live on disk at once. A single shared filename would let the second caching run silently overwrite the first, leaving both arms training on identical inputs — which would destroy the control without any error. `dinov2` keeps `.features.npy` (the 122 files M1 already wrote); every other backbone gets `.features_<backbone>.npy`.

**Why:** Arms 2 and 3 never touch pixels, yet the current loader eagerly loads every episode's obs — 2.24 GB resident (measured). Metadata-only loading costs 0.07 s and 0.04 GB, and memmapping all 122 feature files costs 0.04 s and 0.04 GB. `.npz` is lazily decompressed per key, so simply not reading `obs` avoids the cost entirely.

- [ ] **Step 1: Write the failing test**

Append to `tests/data/test_loader.py`:

```python
def test_load_obs_false_omits_obs(buffer):
    loader = SequenceLoader(buffer, batch_size=4, seq_len=16, seed=0, load_obs=False)
    batch = loader.sample()
    assert "obs" not in batch
    assert batch["actions"].shape == (4, 16)
    assert batch["episode_index"].shape == (4,)
    assert batch["window_start"].shape == (4,)


def test_load_obs_false_still_supports_privileged(buffer):
    loader = SequenceLoader(buffer, batch_size=4, seq_len=16, seed=0, load_obs=False)
    batch = loader.sample(include_privileged=True)
    assert batch["privileged"].shape[0] == 4


def test_load_features_requires_cached_files(tmp_path):
    """A missing cache must fail loudly, not silently train on nothing."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=80, fill=1))
    with pytest.raises(FileNotFoundError, match="no cached features"):
        SequenceLoader(buf, batch_size=2, seq_len=16, seed=0, load_features=True)


def test_features_window_matches_manual_slice(tmp_path):
    """The batch must slice the cache at exactly the reported window_start."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    for fill in (1, 2):
        buf.add(make_episode(t=80, fill=fill))
    rng = np.random.default_rng(0)
    for path in buf.episode_paths():
        feats = rng.random((81, 4, 8)).astype(np.float16)
        np.save(path.with_suffix(".features.npy"), feats)

    loader = SequenceLoader(
        buf, batch_size=4, seq_len=16, seed=0, load_obs=False, load_features=True
    )
    batch = loader.sample()
    assert batch["features"].shape == (4, 17, 4, 8)
    assert batch["features"].dtype == np.float16
    for i in range(4):
        idx, start = int(batch["episode_index"][i]), int(batch["window_start"][i])
        on_disk = np.load(loader.episode_path(idx).with_suffix(".features.npy"))
        assert np.array_equal(on_disk[start : start + 17], batch["features"][i])


def test_features_and_obs_windows_are_aligned(tmp_path):
    """Both must come from the same episode and the same offset."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=80, fill=3))
    feats = np.arange(81 * 4 * 8, dtype=np.float16).reshape(81, 4, 8)
    np.save(buf.episode_paths()[0].with_suffix(".features.npy"), feats)

    loader = SequenceLoader(
        buf, batch_size=2, seq_len=16, seed=0, load_obs=True, load_features=True
    )
    batch = loader.sample()
    for i in range(2):
        start = int(batch["window_start"][i])
        assert np.array_equal(batch["features"][i], feats[start : start + 17])
        assert batch["obs"][i].shape == (17, *OBS_SHAPE)


def test_feature_suffix_namespaces_non_default_backbones():
    from mbfps.data.loader import feature_suffix

    assert feature_suffix("dinov2") == ".features.npy"
    assert feature_suffix("random_vit") == ".features_random_vit.npy"


def test_loader_reads_the_requested_backbones_cache(tmp_path):
    """Two caches coexist; picking the wrong one would silently destroy the
    control by training arms 2 and 3 on identical inputs."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=80, fill=1))
    path = buf.episode_paths()[0]
    np.save(path.with_suffix(".features.npy"), np.zeros((81, 4, 8), np.float16))
    np.save(path.with_suffix(".features_random_vit.npy"), np.ones((81, 4, 8), np.float16))

    for backbone, expected in (("dinov2", 0.0), ("random_vit", 1.0)):
        loader = SequenceLoader(
            buf, 2, 16, seed=0, load_obs=False,
            load_features=True, feature_backbone=backbone,
        )
        assert (loader.sample()["features"] == expected).all(), backbone


def test_obs_free_loader_does_not_read_pixels(tmp_path, monkeypatch):
    """Guards the 2.24 GB regression: obs must never be decompressed."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=80, fill=1))

    import mbfps.data.loader as loader_module

    def _boom(*args, **kwargs):
        raise AssertionError("load_obs=False must not call load_episode")

    monkeypatch.setattr(loader_module, "load_episode", _boom)
    loader = SequenceLoader(buf, batch_size=2, seq_len=16, seed=0, load_obs=False)
    assert loader.sample()["actions"].shape == (2, 16)
```

You will also need `from mbfps.envs.protocol import OBS_SHAPE` and `from mbfps.data.episode import load_episode` available in the test module; add them if absent.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/data/test_loader.py -v -k "load_obs or features"`
Expected: FAIL — `SequenceLoader() got an unexpected keyword argument 'load_obs'`

- [ ] **Step 3: Rewrite `src/mbfps/data/loader.py`**

```python
# src/mbfps/data/loader.py
"""Sequence sampling for world-model training.

Windows never span an episode boundary: a window that stitched the end of one
episode to the start of another would teach the RSSM a transition the engine
can never produce.

Two arms of the study never read pixels at all -- they train on cached features
from a frozen backbone. Loading obs for them costs 2.24 GB of resident memory
for nothing (measured), so `load_obs=False` skips it. `.npz` decompresses
lazily per key, so not reading `obs` genuinely avoids the cost.

Feature files are memmapped rather than loaded: mapping all 122 costs 0.04 s
and 0.04 GB, versus 2.93 GB to read them.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import load_episode

logger = logging.getLogger(__name__)

_WARN_BYTES = 4_000_000_000
"""Warn above this resident footprint.

One episode's obs is ~19.8 MB, so the default capacity of 60,000 transitions
(~122 episodes) sits near 2.24 GB. The threshold is set above that so the
warning fires only when a run is genuinely at risk on the 16 GB shared budget,
rather than on every ordinary run.
"""

_DEFAULT_BACKBONE = "dinov2"


def feature_suffix(backbone: str) -> str:
    """Sibling filename suffix for `backbone`'s cached features.

    `dinov2` keeps the bare `.features.npy` that M1 already wrote, so the
    existing 122 files stay valid. Every other backbone is namespaced, because
    two caches now coexist and a shared name would let one silently overwrite
    the other -- leaving arms 2 and 3 training on identical inputs, with no
    error to notice.
    """
    if backbone == _DEFAULT_BACKBONE:
        return ".features.npy"
    return f".features_{backbone}.npy"


class SequenceLoader:
    """Samples `(B, T)` windows from episodes held in a `ReplayBuffer`."""

    def __init__(
        self,
        buffer: ReplayBuffer,
        batch_size: int = 16,
        seq_len: int = 64,
        seed: int = 0,
        load_obs: bool = True,
        load_features: bool = False,
        feature_backbone: str = _DEFAULT_BACKBONE,
    ) -> None:
        self.buffer = buffer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.load_obs = load_obs
        self.load_features = load_features
        self.feature_backbone = feature_backbone
        self._rng = np.random.default_rng(seed)

        # Keep paths and episodes index-aligned by construction rather than by
        # relying on load_all() happening to preserve order.
        self._paths: list[Path] = buffer.episode_paths()
        self._episodes = [self._read(p) for p in self._paths]

        self._features: list[np.ndarray | None] = []
        if load_features:
            suffix = feature_suffix(feature_backbone)
            for path in self._paths:
                feature_path = path.with_suffix(suffix)
                if not feature_path.is_file():
                    raise FileNotFoundError(
                        f"no cached features for {path.name}; expected "
                        f"{feature_path.name}. Run scripts/cache_features.py "
                        f"--backbone {feature_backbone} first."
                    )
                self._features.append(np.load(feature_path, mmap_mode="r"))

        if load_obs:
            total = sum(ep.obs.nbytes for ep in self._episodes)
            if total > _WARN_BYTES:
                logger.warning(
                    "SequenceLoader holds %d episodes (%.2f GB) resident in RAM. "
                    "Episodes are loaded eagerly at construction. Pass "
                    "load_obs=False for arms that train on cached features.",
                    len(self._episodes),
                    total / 1e9,
                )

    def _read(self, path: Path):
        """Load one episode, skipping its obs array when pixels are not needed."""
        if self.load_obs:
            return load_episode(path)
        return _EpisodeMeta.from_npz(path)

    def episode_path(self, index: int) -> Path:
        """File backing the episode a batch's `episode_index` refers to.

        Use it to locate the sibling feature cache for a window.
        """
        return self._paths[index]

    def _usable(self) -> list[int]:
        return [i for i, ep in enumerate(self._episodes) if ep.length >= self.seq_len]

    def sample(self, include_privileged: bool = False) -> dict[str, Any]:
        """Sample one batch.

        Args:
            include_privileged: include ground-truth engine state. EVALUATION
                PROBES ONLY -- passing True in a training loop invalidates the
                study. Defaults to False so the safe path needs no thought.

        Raises:
            ValueError: if no episode is at least `seq_len` transitions long.
        """
        usable = self._usable()
        if not usable:
            raise ValueError(
                f"no episodes long enough for seq_len={self.seq_len}; "
                f"buffer holds {len(self._episodes)} episodes"
            )

        obs, actions, rewards = [], [], []
        terminated, truncated, privileged, features = [], [], [], []
        indices, starts = [], []

        for _ in range(self.batch_size):
            idx = int(self._rng.choice(usable))
            ep = self._episodes[idx]
            start = int(self._rng.integers(0, ep.length - self.seq_len + 1))
            end = start + self.seq_len
            actions.append(ep.actions[start:end])
            rewards.append(ep.rewards[start:end])
            terminated.append(ep.terminated[start:end])
            truncated.append(ep.truncated[start:end])
            indices.append(idx)
            starts.append(start)
            if self.load_obs:
                obs.append(ep.obs[start : end + 1])
            if self.load_features:
                features.append(np.asarray(self._features[idx][start : end + 1]))
            if include_privileged:
                privileged.append(ep.privileged[start : end + 1])

        batch: dict[str, Any] = {
            "actions": np.stack(actions).astype(np.int32),
            "rewards": np.stack(rewards).astype(np.float32),
            "terminated": np.stack(terminated).astype(bool),
            "truncated": np.stack(truncated).astype(bool),
            "episode_index": np.asarray(indices, dtype=np.int32),
            "window_start": np.asarray(starts, dtype=np.int32),
        }
        if self.load_obs:
            batch["obs"] = np.stack(obs).astype(np.uint8)
        if self.load_features:
            batch["features"] = np.stack(features)
        if include_privileged:
            batch["privileged"] = np.stack(privileged).astype(np.float32)
        return batch


class _EpisodeMeta:
    """An episode with everything except its obs array.

    `np.load` on an npz returns a lazy handle, so the pixel data is never
    decompressed when only these fields are read.
    """

    __slots__ = (
        "actions", "rewards", "terminated", "truncated",
        "privileged", "privileged_keys", "policy_name", "seed", "scenario",
    )

    @classmethod
    def from_npz(cls, path: Path) -> "_EpisodeMeta":
        meta = cls()
        with np.load(path) as data:
            meta.actions = data["actions"]
            meta.rewards = data["rewards"]
            meta.terminated = data["terminated"]
            meta.truncated = data["truncated"]
            meta.privileged = data["privileged"]
            meta.privileged_keys = tuple(data["privileged_keys"].tolist())
            meta.policy_name = str(data["policy_name"])
            meta.seed = int(data["seed"])
            meta.scenario = str(data["scenario"])
        return meta

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_loader.py -v`
Expected: 20 passed (12 existing plus the 8 added here).

- [ ] **Step 5: Verify the memory claim on the real dataset**

```bash
.venv/bin/python -c "
import resource
from mbfps.data.buffer import ReplayBuffer
from mbfps.data.loader import SequenceLoader
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9
buf = ReplayBuffer('data/my_way_home', capacity_transitions=10**9)
ld = SequenceLoader(buf, 16, 64, 0, load_obs=False, load_features=True)
print(f'obs-free + features loader RSS: {rss():.2f} GB')
b = ld.sample()
print('keys:', sorted(b)); print('features:', b['features'].shape, b['features'].dtype)
print(f'after one batch: {rss():.2f} GB   (eager-obs baseline was 2.24 GB)'
)"
```

Expected: RSS well under 1 GB, `obs` absent from the keys, `features` of shape `(16, 65, 64, 384)` float16. Record the number in the commit message.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/data/loader.py tests/data/test_loader.py
git commit -m "feat: obs-optional and feature-capable sequence loading"
```

---

## Task 3: Background prefetching

**Files:**
- Create: `src/mbfps/data/prefetch.py`
- Create: `tests/data/test_prefetch.py`

**Interfaces:**
- Consumes: `SequenceLoader` from `mbfps.data.loader`.
- Produces: `Prefetcher(loader, depth: int = 2)` — an iterable yielding batches produced on a background thread; `close() -> None` (idempotent); supports the context-manager protocol.

**Why:** A feature batch costs 101 ms to read (51 MB from memmap) while an Arm 2/3 training step costs 161 ms — so unoverlapped loading adds ~63% to the fast arms. Arm 1's step is 1546 ms, where the same 101 ms is ~6%, so this exists for the SSL arms. NumPy releases the GIL during the copy, so a thread suffices; no process pool is needed.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_prefetch.py
import numpy as np
import pytest

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.data.loader import SequenceLoader
from mbfps.data.prefetch import Prefetcher
from mbfps.envs.protocol import OBS_SHAPE

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def make_episode(t: int, fill: int) -> Episode:
    return Episode(
        obs=np.full((t + 1, *OBS_SHAPE), fill, dtype=np.uint8),
        actions=np.zeros(t, dtype=np.int32),
        rewards=np.zeros(t, dtype=np.float32),
        terminated=np.zeros(t, dtype=bool),
        truncated=np.zeros(t, dtype=bool),
        privileged=np.zeros((t + 1, len(KEYS)), dtype=np.float32),
        privileged_keys=KEYS,
        policy_name="random",
        seed=fill,
        scenario="my_way_home",
    )


@pytest.fixture
def buffer(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    for fill in (1, 2, 3):
        buf.add(make_episode(t=80, fill=fill))
    return buf


def test_yields_the_requested_number_of_batches(buffer):
    loader = SequenceLoader(buffer, batch_size=2, seq_len=16, seed=0)
    with Prefetcher(loader, depth=2) as pf:
        batches = [next(iter(pf)) for _ in range(3)]
    assert len(batches) == 3
    assert all(b["actions"].shape == (2, 16) for b in batches)


def test_batches_match_direct_sampling_in_order(buffer):
    """Prefetching must not change what the loader would have produced."""
    direct_loader = SequenceLoader(buffer, batch_size=2, seq_len=16, seed=7)
    expected = [direct_loader.sample() for _ in range(4)]

    pf_loader = SequenceLoader(buffer, batch_size=2, seq_len=16, seed=7)
    with Prefetcher(pf_loader, depth=2) as pf:
        it = iter(pf)
        got = [next(it) for _ in range(4)]

    for i, (a, b) in enumerate(zip(expected, got)):
        assert np.array_equal(a["obs"], b["obs"]), f"batch {i} differs"
        assert np.array_equal(a["episode_index"], b["episode_index"])
        assert np.array_equal(a["window_start"], b["window_start"])


def test_close_is_idempotent(buffer):
    pf = Prefetcher(SequenceLoader(buffer, 2, 16, seed=0), depth=1)
    next(iter(pf))
    pf.close()
    pf.close()


def test_context_manager_closes_the_thread(buffer):
    import threading

    before = threading.active_count()
    with Prefetcher(SequenceLoader(buffer, 2, 16, seed=0), depth=1) as pf:
        next(iter(pf))
    assert threading.active_count() == before, "worker thread outlived the context"


def test_worker_exception_propagates_to_the_consumer(buffer):
    """A failing loader must raise here, not hang the consumer forever."""

    class _Broken(SequenceLoader):
        def sample(self, include_privileged: bool = False):
            raise RuntimeError("loader exploded")

    with Prefetcher(_Broken(buffer, 2, 16, seed=0), depth=1) as pf:
        with pytest.raises(RuntimeError, match="loader exploded"):
            next(iter(pf))


def test_depth_must_be_positive(buffer):
    with pytest.raises(ValueError, match="depth must be at least 1"):
        Prefetcher(SequenceLoader(buffer, 2, 16, seed=0), depth=0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_prefetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.data.prefetch'`

- [ ] **Step 3: Write the implementation**

```python
# src/mbfps/data/prefetch.py
"""Overlap batch loading with computation.

Reading a feature batch costs ~101 ms (51 MB from memmap) while an SSL-arm
training step costs ~161 ms, so serial loading adds ~63% to the arms that are
otherwise fast. A single background thread is enough: NumPy releases the GIL
while copying, so the read genuinely overlaps the forward/backward pass.

The pixel arm sees ~6% overhead from the same read and does not need this, but
using one path for every arm keeps the arms comparable.
"""

import queue
import threading
from typing import Any, Iterator

from mbfps.data.loader import SequenceLoader

_SENTINEL = object()


class Prefetcher:
    """Yields batches from `loader`, produced on a background thread."""

    def __init__(self, loader: SequenceLoader, depth: int = 2) -> None:
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self._loader = loader
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def _work(self) -> None:
        try:
            while not self._stop.is_set():
                batch = self._loader.sample()
                while not self._stop.is_set():
                    try:
                        self._queue.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
            # Deliver the failure rather than dying silently and leaving the
            # consumer blocked on an empty queue forever.
            self._queue.put(exc)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def close(self) -> None:
        """Stop the worker thread. Safe to call more than once."""
        if self._stop.is_set():
            return
        self._stop.set()
        # Drain so a worker blocked on put() can observe the stop flag.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=2.0)

    def __enter__(self) -> "Prefetcher":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_prefetch.py -v`
Expected: 6 passed.

- [ ] **Step 5: Measure the overlap on the real dataset**

```bash
.venv/bin/python -c "
import time
from mbfps.data.buffer import ReplayBuffer
from mbfps.data.loader import SequenceLoader
from mbfps.data.prefetch import Prefetcher
buf = ReplayBuffer('data/my_way_home', capacity_transitions=10**9)
mk = lambda: SequenceLoader(buf, 16, 64, 0, load_obs=False, load_features=True)
ld = mk(); ld.sample()
t0 = time.perf_counter()
for _ in range(10): ld.sample()
serial = (time.perf_counter()-t0)/10
with Prefetcher(mk(), depth=2) as pf:
    it = iter(pf); next(it)
    t0 = time.perf_counter()
    for _ in range(10): next(it)
    pre = (time.perf_counter()-t0)/10
print(f'serial   : {serial*1000:6.1f} ms/batch')
print(f'prefetch : {pre*1000:6.1f} ms/batch')
"
```

Expected: the prefetched figure is substantially lower than serial when a consumer is doing work between batches. Note that this microbenchmark has no compute to hide behind, so the gain here will be modest — record both numbers; the real benefit shows up in Task 7's training loop.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/data/prefetch.py tests/data/test_prefetch.py
git commit -m "feat: background batch prefetching to overlap I/O with compute"
```

---

## Task 4: Backbone-parameterised feature caching

**Files:**
- Modify: `src/mbfps/data/features.py`
- Modify: `src/mbfps/data/buffer.py` (eviction must clear every backbone's cache)
- Modify: `tests/data/test_features.py`
- Modify: `tests/data/test_buffer.py`
- Create: `scripts/cache_features.py`

**Interfaces:**
- Consumes: `get_device` from `mbfps.utils.device`; `OBS_SHAPE`; `ReplayBuffer`.
- Produces: `BACKBONES: tuple[str, ...] = ("dinov2", "random_vit")`; `build_backbone(kind: str, seed: int = 0) -> torch.nn.Module`; `FeatureExtractor(backbone: str = "dinov2", device: str = "mps", seed: int = 0)` retaining `encode` and the `N_PATCHES` / `FEATURE_DIM` constants; `require_free_bytes(path: Path, needed: int) -> None`; `cache_episode_features(ep_path, extractor, batch_size=32) -> Path` — now writes to the suffix for `extractor.backbone`, so two caches coexist without colliding.

**Two constraints this task exists to satisfy:**

1. **Arm 3 needs its own cached features.** It is Arm 2 with a randomly-initialised backbone of identical architecture. Verified: `AutoModel.from_config(AutoConfig.from_pretrained("facebook/dinov2-small"))` under a fixed seed yields the same `(N, 65, 384)` shape, 22.1M frozen parameters, and bit-identical weights on repeat.
2. **The disk has 2.3 GB free and one cache is 2.93 GB.** Two caches will not fit. The workflow is therefore: cache one backbone, train that arm, delete the cache, cache the next. `require_free_bytes` must fail *before* writing rather than filling the disk and dying partway.

- [ ] **Step 1: Write the failing test**

Append to `tests/data/test_features.py`:

```python
def test_backbones_registered():
    from mbfps.data.features import BACKBONES

    assert BACKBONES == ("dinov2", "random_vit")


def test_unknown_backbone_rejected():
    from mbfps.data.features import build_backbone

    with pytest.raises(KeyError, match="unknown backbone 'nope'"):
        build_backbone("nope")


@pytest.mark.slow
def test_random_vit_has_the_same_output_shape_as_dinov2():
    """Arm 3 must be Arm 2 with different weights, not a different shape."""
    ext = FeatureExtractor(backbone="random_vit", device="cpu", seed=0)
    frames = np.random.default_rng(0).integers(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    assert ext.encode(frames).shape == (2, N_PATCHES, FEATURE_DIM)


@pytest.mark.slow
def test_random_vit_is_reproducible_from_its_seed():
    a = FeatureExtractor(backbone="random_vit", device="cpu", seed=3)
    b = FeatureExtractor(backbone="random_vit", device="cpu", seed=3)
    frames = np.random.default_rng(1).integers(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    assert np.array_equal(a.encode(frames), b.encode(frames))


@pytest.mark.slow
def test_random_vit_differs_from_dinov2():
    """If these matched, Arm 3 would not be a control at all."""
    rnd = FeatureExtractor(backbone="random_vit", device="cpu", seed=0)
    pre = FeatureExtractor(backbone="dinov2", device="cpu")
    frames = np.random.default_rng(2).integers(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    assert not np.allclose(
        rnd.encode(frames).astype(np.float32),
        pre.encode(frames).astype(np.float32),
        atol=1e-2,
    )


@pytest.mark.slow
def test_different_seeds_give_different_random_backbones():
    a = FeatureExtractor(backbone="random_vit", device="cpu", seed=0)
    b = FeatureExtractor(backbone="random_vit", device="cpu", seed=1)
    frames = np.random.default_rng(3).integers(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    assert not np.array_equal(a.encode(frames), b.encode(frames))


def test_require_free_bytes_passes_when_space_available(tmp_path):
    from mbfps.data.features import require_free_bytes

    require_free_bytes(tmp_path, 1)


def test_require_free_bytes_raises_when_space_insufficient(tmp_path):
    from mbfps.data.features import require_free_bytes

    with pytest.raises(OSError, match="needs .* free"):
        require_free_bytes(tmp_path, 10**15)


def test_require_free_bytes_message_names_both_numbers(tmp_path):
    from mbfps.data.features import require_free_bytes

    with pytest.raises(OSError) as excinfo:
        require_free_bytes(tmp_path, 10**15)
    message = str(excinfo.value)
    assert "GB" in message and "available" in message
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/data/test_features.py -v -k "backbone or random_vit or free_bytes"`
Expected: FAIL with `ImportError: cannot import name 'BACKBONES'`

- [ ] **Step 3: Modify `src/mbfps/data/features.py`**

Replace the `FeatureExtractor.__init__` and add the new module-level helpers. Keep `encode`, `cache_episode_features`, `N_PATCHES`, `FEATURE_DIM`, and the ImageNet constants exactly as they are.

```python
import shutil
from pathlib import Path

from transformers import AutoConfig, AutoModel

BACKBONES: tuple[str, ...] = ("dinov2", "random_vit")
"""Frozen backbones. `dinov2` is the treatment arm's pretrained encoder;
`random_vit` is the control -- identical architecture, random weights, which is
what separates "pretraining helps" from "a stationary target helps"."""

_MODEL_NAME = "facebook/dinov2-small"


def build_backbone(kind: str, seed: int = 0) -> "torch.nn.Module":
    """Construct a frozen backbone.

    Args:
        kind: one of `BACKBONES`.
        seed: RNG seed for `random_vit`. Ignored for `dinov2`, whose weights
            are fixed. Verified: the same seed reproduces bit-identical weights.

    Raises:
        KeyError: if `kind` is not registered.
    """
    if kind not in BACKBONES:
        raise KeyError(f"unknown backbone {kind!r}; available: {list(BACKBONES)}")
    if kind == "dinov2":
        return AutoModel.from_pretrained(_MODEL_NAME)
    torch.manual_seed(seed)
    return AutoModel.from_config(AutoConfig.from_pretrained(_MODEL_NAME))


def require_free_bytes(path: Path, needed: int) -> None:
    """Fail before writing if `path`'s filesystem cannot hold `needed` bytes.

    The DINOv2 cache for the M1 dataset is 2.93 GB and this machine has had as
    little as 2.3 GB free, so a caching run can plausibly fill the disk. Failing
    up front beats dying halfway and leaving a partial cache that later loads
    as a FileNotFoundError on some episodes but not others.

    Raises:
        OSError: if free space is less than `needed`.
    """
    free = shutil.disk_usage(Path(path)).free
    if free < needed:
        raise OSError(
            f"caching needs {needed / 1e9:.2f} GB free at {path}, "
            f"but only {free / 1e9:.2f} GB is available"
        )
```

Change `FeatureExtractor.__init__` to:

```python
    def __init__(
        self, backbone: str = "dinov2", device: str = "mps", seed: int = 0
    ) -> None:
        self.backbone = backbone
        self.device = get_device(prefer=device)
        self.model = build_backbone(backbone, seed=seed).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
```

- [ ] **Step 3b: Route writes and eviction through the per-backbone suffix**

Two places in the committed code hardcode `.features.npy` and must now use `feature_suffix`, or the second cache will overwrite the first and eviction will orphan it.

In `src/mbfps/data/features.py`, `cache_episode_features` currently ends with:

```python
    out_path = ep_path.with_suffix(".features.npy")
```

Replace it with:

```python
    from mbfps.data.loader import feature_suffix

    out_path = ep_path.with_suffix(feature_suffix(extractor.backbone))
```

and update its docstring to say the filename is namespaced by backbone.

In `src/mbfps/data/buffer.py`, `_evict` currently unlinks one fixed name:

```python
            path.with_suffix(".features.npy").unlink(missing_ok=True)
```

Replace it with a glob over every cache variant, so a second backbone's features cannot outlive their episode:

```python
            for cache in path.parent.glob(f"{path.stem}.features*.npy"):
                cache.unlink(missing_ok=True)
```

Add to `tests/data/test_buffer.py`:

```python
def test_eviction_removes_every_backbones_feature_cache(tmp_path):
    """Two caches coexist; evicting only one orphans the other forever."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=25)
    first = buf.add(make_episode(10, seed=1))
    caches = [
        first.with_suffix(".features.npy"),
        first.with_suffix(".features_random_vit.npy"),
    ]
    for cache in caches:
        np.save(cache, np.zeros((11, 4, 8), dtype=np.float16))
    buf.add(make_episode(10, seed=2))
    buf.add(make_episode(10, seed=3))
    assert not first.is_file()
    for cache in caches:
        assert not cache.is_file(), f"{cache.name} outlived its episode"
```

Run: `.venv/bin/python -m pytest tests/data/test_buffer.py -v`
Expected: 16 passed (15 existing plus this one).

- [ ] **Step 4: Write the caching script**

```python
# scripts/cache_features.py
"""Cache frozen-backbone features for every episode in a buffer.

Run once per backbone. Because one cache is ~2.93 GB and this machine has had
as little as 2.3 GB free, the workflow is one cache at a time:

    cache_features.py --backbone dinov2      # train arm 2
    cache_features.py --backbone dinov2 --clear
    cache_features.py --backbone random_vit  # train arm 3

Features must be generated once on one device: CPU and MPS outputs differ in
~3% of float16 elements, so mixing them would silently break the input
equality the study depends on.
"""

import argparse
import time
from pathlib import Path

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.features import (
    BACKBONES,
    FEATURE_DIM,
    N_PATCHES,
    FeatureExtractor,
    cache_episode_features,
    require_free_bytes,
)
from mbfps.data.loader import feature_suffix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--backbone", choices=BACKBONES, default="dinov2")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--clear", action="store_true", help="delete existing caches and exit"
    )
    args = parser.parse_args()

    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    paths = buffer.episode_paths()
    if not paths:
        raise SystemExit(f"no episodes found in {args.data}")

    if args.clear:
        removed = 0
        for path in paths:
            feature_path = path.with_suffix(feature_suffix(args.backbone))
            if feature_path.is_file():
                feature_path.unlink()
                removed += 1
        print(f"removed={removed} {args.backbone} feature files from {args.data}")
        return

    frames = sum(ep.length + 1 for ep in buffer.load_all())
    needed = frames * N_PATCHES * FEATURE_DIM * 2  # float16
    print(f"episodes={len(paths)} frames={frames} needs={needed / 1e9:.2f} GB")
    require_free_bytes(args.data, needed)

    extractor = FeatureExtractor(
        backbone=args.backbone, device=args.device, seed=args.seed
    )
    start = time.perf_counter()
    for i, path in enumerate(paths, start=1):
        cache_episode_features(path, extractor)
        if i % 20 == 0 or i == len(paths):
            print(f"[{i}/{len(paths)}] elapsed={time.perf_counter() - start:.0f}s")

    elapsed = time.perf_counter() - start
    print(f"backbone={args.backbone} device={args.device} seed={args.seed}")
    print(f"features_cached={len(paths)} elapsed_s={elapsed:.0f}")
    print(f"frames_per_second={frames / elapsed:.0f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_features.py -v`
Expected: 17 passed (9 existing plus the 8 added here).

Also confirm the existing M1 cache still loads: `.venv/bin/python -c "from mbfps.data.loader import feature_suffix; print(feature_suffix('dinov2'))"` must print `.features.npy`, matching the 122 files already on disk.

- [ ] **Step 6: Verify the disk guard on the real filesystem**

```bash
.venv/bin/python -c "
from pathlib import Path
from mbfps.data.features import require_free_bytes
import shutil
free = shutil.disk_usage('.').free
print(f'free: {free/1e9:.2f} GB')
require_free_bytes(Path('.'), 1000)
print('small request: OK')
try:
    require_free_bytes(Path('.'), free * 2)
    print('FAIL: oversized request was allowed')
except OSError as e:
    print('oversized request correctly refused:', e)
"
```

Expected: the small request passes and the oversized one raises with both numbers named.

- [ ] **Step 7: Commit**

```bash
git add src/mbfps/data/features.py tests/data/test_features.py scripts/cache_features.py
git commit -m "feat: backbone-parameterised feature caching with a disk guard"
```

---

## Task 5: Encoders for all three arms

**Files:**
- Create: `src/mbfps/models/__init__.py`
- Create: `src/mbfps/models/encoders.py`
- Create: `tests/models/__init__.py`
- Create: `tests/models/test_encoders.py`

**Interfaces:**
- Consumes: `EncoderConfig` from `mbfps.utils.config`; `OBS_SHAPE`.
- Produces: `CNNEncoder(cfg: EncoderConfig)` mapping `(N, 112, 112, 3)` uint8 to `(N, 2048)` float32; `BottleneckEncoder(cfg: EncoderConfig)` mapping `(N, 64, 384)` float32 to `(N, 2048)` float32; `build_encoder(cfg: EncoderConfig) -> nn.Module`; `encoder_input_kind(cfg: EncoderConfig) -> str` returning `"obs"` or `"features"`.

**This is the only module that differs between arms.** If a later task needs to special-case an arm anywhere else, the comparison has stopped being controlled.

**Recorded parameter counts.** The spec's section 3.1 estimates "~4M (encoder + decoder)" for Arm 1 and "~0.2M (bottleneck)" for Arms 2/3. Both estimates are wrong; measured values for the architecture the spec *describes* are:

| | spec estimate | measured |
|---|---|---|
| Arm 1 conv stack | | 0.690 M |
| Arm 1 `Linear(12544 -> 2048)` | | 25.692 M |
| Arm 1 encoder (conv + projection) | | **26.382 M** |
| Arm 1 decoder | | **26.393 M** |
| Arm 1 encoder + decoder | ~4 M | **52.775 M** |
| Arm 2/3 bottleneck `Linear(384 -> 32)` | ~0.2 M | **0.012 M** |

The `~4M` figure came from DreamerV3's native 64x64 configuration, whose projection is `Linear(4096, 1024)` = 4.195 M, and was not recomputed for 112x112 with a 2048-wide embedding. **Implement the architecture the spec describes, not the parameter count it estimates.** Shrinking the baseline to be cheaper would be tuning it down, which biases the study toward the treatment — the opposite of what a baseline is for.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_encoders.py
import pytest
import torch

from mbfps.envs.protocol import OBS_SHAPE
from mbfps.models.encoders import (
    BottleneckEncoder,
    CNNEncoder,
    build_encoder,
    encoder_input_kind,
)
from mbfps.utils.config import ARMS, EncoderConfig


def cfg(kind: str) -> EncoderConfig:
    return EncoderConfig(kind=kind)


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def test_cnn_encoder_output_shape():
    enc = CNNEncoder(cfg("cnn"))
    obs = torch.randint(0, 256, (4, *OBS_SHAPE), dtype=torch.uint8)
    assert enc(obs).shape == (4, 2048)


def test_cnn_encoder_output_is_float32():
    enc = CNNEncoder(cfg("cnn"))
    obs = torch.randint(0, 256, (2, *OBS_SHAPE), dtype=torch.uint8)
    assert enc(obs).dtype == torch.float32


def test_cnn_encoder_accepts_uint8_without_manual_conversion():
    """The loader hands out uint8; the encoder owns normalisation."""
    enc = CNNEncoder(cfg("cnn"))
    obs = torch.full((2, *OBS_SHAPE), 255, dtype=torch.uint8)
    assert torch.isfinite(enc(obs)).all()


def test_bottleneck_encoder_output_shape():
    enc = BottleneckEncoder(cfg("frozen_ssl"))
    feats = torch.randn(4, 64, 384)
    assert enc(feats).shape == (4, 2048)


def test_bottleneck_flattens_patch_grid_to_embed_dim():
    """64 patches x 32 bottleneck dims = 2048, matching the CNN arm exactly."""
    c = cfg("frozen_ssl")
    assert 64 * c.bottleneck_dim == c.embed_dim


@pytest.mark.parametrize("arm", ARMS)
def test_build_encoder_returns_something_for_every_arm(arm):
    assert build_encoder(cfg(arm)) is not None


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_emits_the_same_embedding_width(arm):
    c = cfg(arm)
    enc = build_encoder(c)
    if encoder_input_kind(c) == "obs":
        x = torch.randint(0, 256, (3, *OBS_SHAPE), dtype=torch.uint8)
    else:
        x = torch.randn(3, 64, 384)
    assert enc(x).shape == (3, 2048)


def test_input_kind_is_obs_for_cnn_and_features_for_ssl_arms():
    assert encoder_input_kind(cfg("cnn")) == "obs"
    assert encoder_input_kind(cfg("frozen_ssl")) == "features"
    assert encoder_input_kind(cfg("random_vit")) == "features"


def test_unknown_arm_rejected():
    with pytest.raises(KeyError, match="unknown encoder kind 'nope'"):
        build_encoder(EncoderConfig(kind="nope"))


def test_ssl_arms_build_identical_architectures():
    """Arm 3 is Arm 2 with different cached inputs -- the trainable module is
    the same, or the control is not a control."""
    a = BottleneckEncoder(cfg("frozen_ssl"))
    b = BottleneckEncoder(cfg("random_vit"))
    assert n_params(a) == n_params(b)
    assert [tuple(p.shape) for p in a.parameters()] == [
        tuple(p.shape) for p in b.parameters()
    ]


def test_recorded_parameter_counts():
    """Pins the measured counts so a silent architecture change is visible.

    The spec's own estimates (~4M encoder+decoder, ~0.2M bottleneck) are wrong;
    these are the measured values for the architecture it describes.
    """
    assert n_params(CNNEncoder(cfg("cnn"))) == 26_382_304
    assert n_params(BottleneckEncoder(cfg("frozen_ssl"))) == 12_320


def test_cnn_gradients_flow_to_every_parameter():
    enc = CNNEncoder(cfg("cnn"))
    obs = torch.randint(0, 256, (2, *OBS_SHAPE), dtype=torch.uint8)
    enc(obs).square().mean().backward()
    missing = [n for n, p in enc.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_bottleneck_gradients_flow_to_every_parameter():
    enc = BottleneckEncoder(cfg("frozen_ssl"))
    enc(torch.randn(2, 64, 384)).square().mean().backward()
    missing = [n for n, p in enc.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/models/test_encoders.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.models'`

- [ ] **Step 3: Write the implementation**

```python
# src/mbfps/models/__init__.py
```

```python
# tests/models/__init__.py
```

```python
# src/mbfps/models/encoders.py
"""Encoders -- the ONLY module that differs between the study's three arms.

Arm 1 learns a CNN over pixels. Arms 2 and 3 apply a small learned bottleneck
to features from a frozen backbone, cached at collection time so no vision
transformer runs during training.

All three emit a 2048-dimensional embedding from byte-identical 112x112 frames,
so resolution and embedding width are controlled and only the representation
differs. If any other module ever needs to know which arm is running, the
comparison has stopped being controlled.
"""

import torch
import torch.nn as nn

from mbfps.utils.config import EncoderConfig

_SPATIAL = 7
"""112 / 2^4 = 7, the spatial size after four stride-2 convolutions."""


class CNNEncoder(nn.Module):
    """Learned convolutional encoder over raw pixels (Arm 1).

    Takes uint8 straight from the loader and owns its own normalisation, so no
    caller has to remember to scale. Four stride-2 convolutions reduce
    112x112 to 7x7x256 = 12544, then a linear projection gives the shared
    2048-dimensional embedding.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        channels, layers = 3, []
        for multiple in (1, 2, 4, 8):
            out = cfg.cnn_depth * multiple
            layers += [nn.Conv2d(channels, out, 4, stride=2, padding=1), nn.SiLU()]
            channels = out
        self.conv = nn.Sequential(*layers)
        self.project = nn.Linear(channels * _SPATIAL * _SPATIAL, cfg.embed_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Map `(N, 112, 112, 3)` uint8 to `(N, embed_dim)` float32."""
        x = obs.to(torch.float32).div(255.0).sub(0.5).permute(0, 3, 1, 2)
        return self.project(self.conv(x).flatten(1))


class BottleneckEncoder(nn.Module):
    """Learned per-patch bottleneck over frozen-backbone features (Arms 2, 3).

    The backbone never updates, so its features are cached once at collection
    time and this module is the only trained part of the encoder path. A ViT's
    patch grid is far wider than the shared embedding (64 x 384 = 24576), so a
    per-patch linear reduces each patch to `bottleneck_dim` and the grid is
    flattened to exactly `embed_dim`.

    Arms 2 and 3 build the identical module; only the cached inputs differ.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        if 64 * cfg.bottleneck_dim != cfg.embed_dim:
            raise ValueError(
                f"64 patches x bottleneck_dim {cfg.bottleneck_dim} must equal "
                f"embed_dim {cfg.embed_dim}"
            )
        self.bottleneck = nn.Linear(cfg.patch_dim, cfg.bottleneck_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map `(N, 64, 384)` to `(N, embed_dim)` float32."""
        return self.bottleneck(features.to(torch.float32)).flatten(1)


def encoder_input_kind(cfg: EncoderConfig) -> str:
    """Whether this arm's encoder consumes `"obs"` or cached `"features"`."""
    return "obs" if cfg.kind == "cnn" else "features"


def build_encoder(cfg: EncoderConfig) -> nn.Module:
    """Construct the encoder for `cfg.kind`.

    Raises:
        KeyError: if `cfg.kind` is not a registered arm.
    """
    if cfg.kind == "cnn":
        return CNNEncoder(cfg)
    if cfg.kind in ("frozen_ssl", "random_vit"):
        return BottleneckEncoder(cfg)
    raise KeyError(f"unknown encoder kind {cfg.kind!r}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/models/test_encoders.py -v`
Expected: 17 passed (several are parametrised over the three arms).

If `test_recorded_parameter_counts` fails, do **not** adjust the constant to match — the architecture has drifted from the spec's description. Compare your conv channel progression and projection width against section 3.2 before changing anything.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/models tests/models
git commit -m "feat: CNN and bottleneck encoders emitting a shared 2048-d embedding"
```

---

## Task 6: Pixel decoder

**Files:**
- Create: `src/mbfps/models/decoders.py`
- Create: `tests/models/test_decoders.py`

**Interfaces:**
- Consumes: `EncoderConfig`; `OBS_SHAPE`.
- Produces: `PixelDecoder(in_dim: int, depth: int = 32)` mapping `(N, in_dim)` float32 to `(N, 112, 112, 3)` float32 in [0, 1]; `reconstruction_loss(pred: Tensor, target_obs: Tensor) -> Tensor`.

**One decoder serves two purposes.** For Arm 1 it is part of the trained model — reconstruction is the objective that shapes the latent. For Arms 2 and 3 it is a *visualisation* decoder: it makes the feature-space embedding viewable so M2's comparison grid can be drawn, and in a later milestone it will be trained separately and kept out of the RL path so pretty pictures cannot become an experimental confound. `in_dim` is a parameter because M3 will feed it a concatenated recurrent state rather than a bare embedding.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_decoders.py
import pytest
import torch

from mbfps.envs.protocol import OBS_SHAPE
from mbfps.models.decoders import PixelDecoder, reconstruction_loss


def test_output_shape_matches_observation_shape():
    dec = PixelDecoder(in_dim=2048)
    assert dec(torch.randn(4, 2048)).shape == (4, *OBS_SHAPE)


def test_output_is_float32_in_unit_range():
    dec = PixelDecoder(in_dim=2048)
    out = dec(torch.randn(4, 2048) * 10.0)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_in_dim_is_configurable_for_later_recurrent_use():
    """M3 feeds a concatenated recurrent state, not a bare embedding."""
    dec = PixelDecoder(in_dim=1536)
    assert dec(torch.randn(2, 1536)).shape == (2, *OBS_SHAPE)


def test_wrong_input_width_raises():
    dec = PixelDecoder(in_dim=2048)
    with pytest.raises(RuntimeError):
        dec(torch.randn(2, 1024))


def test_gradients_flow_to_every_parameter():
    dec = PixelDecoder(in_dim=2048)
    dec(torch.randn(2, 2048)).square().mean().backward()
    missing = [n for n, p in dec.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_reconstruction_loss_is_zero_for_a_perfect_match():
    target = torch.randint(0, 256, (2, *OBS_SHAPE), dtype=torch.uint8)
    perfect = target.to(torch.float32) / 255.0
    assert reconstruction_loss(perfect, target).item() == pytest.approx(0.0, abs=1e-6)


def test_reconstruction_loss_is_positive_for_a_mismatch():
    target = torch.zeros((2, *OBS_SHAPE), dtype=torch.uint8)
    pred = torch.ones((2, *OBS_SHAPE), dtype=torch.float32)
    assert reconstruction_loss(pred, target).item() == pytest.approx(1.0, abs=1e-6)


def test_reconstruction_loss_accepts_uint8_targets_directly():
    """The loader hands out uint8; the loss owns the conversion."""
    target = torch.randint(0, 256, (2, *OBS_SHAPE), dtype=torch.uint8)
    pred = torch.rand((2, *OBS_SHAPE))
    assert torch.isfinite(reconstruction_loss(pred, target))


def test_recorded_parameter_count():
    """Pins the measured count so an architecture drift is visible."""
    n = sum(p.numel() for p in PixelDecoder(in_dim=2048).parameters())
    assert n == 26_392_547
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/models/test_decoders.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.models.decoders'`

- [ ] **Step 3: Write the implementation**

```python
# src/mbfps/models/decoders.py
"""Pixel decoder.

For the pixel arm this is part of the trained model: reconstruction is the
objective that shapes the latent. For the feature arms it is a visualisation
decoder that makes a feature-space embedding viewable, so the arms can be
compared side by side at all.

`in_dim` is a parameter rather than a constant because a later milestone feeds
this a concatenated recurrent state instead of a bare embedding.
"""

import torch
import torch.nn as nn

from mbfps.envs.protocol import OBS_SHAPE

_SPATIAL = 7
"""Matches the encoder: four stride-2 steps between 7x7 and 112x112."""


class PixelDecoder(nn.Module):
    """Maps a latent vector back to a 112x112x3 image in [0, 1]."""

    def __init__(self, in_dim: int, depth: int = 32) -> None:
        super().__init__()
        channels = depth * 8
        self.project = nn.Linear(in_dim, channels * _SPATIAL * _SPATIAL)
        self._channels = channels
        layers = []
        for multiple in (4, 2, 1):
            out = depth * multiple
            layers += [
                nn.ConvTranspose2d(channels, out, 4, stride=2, padding=1),
                nn.SiLU(),
            ]
            channels = out
        layers.append(nn.ConvTranspose2d(channels, 3, 4, stride=2, padding=1))
        self.deconv = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Map `(N, in_dim)` to `(N, 112, 112, 3)` float32 in [0, 1]."""
        x = self.project(latent).view(-1, self._channels, _SPATIAL, _SPATIAL)
        image = torch.sigmoid(self.deconv(x))
        return image.permute(0, 2, 3, 1)


def reconstruction_loss(pred: torch.Tensor, target_obs: torch.Tensor) -> torch.Tensor:
    """Mean squared error between a prediction in [0, 1] and a uint8 target.

    The loss owns the uint8 conversion so no caller has to remember to scale;
    a forgotten division by 255 would train against a 255x-larger target and
    look like a diverging model rather than a units bug.
    """
    target = target_obs.to(pred.dtype)
    if target_obs.dtype == torch.uint8:
        target = target / 255.0
    return (pred - target).square().mean()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/models/test_decoders.py -v`
Expected: 9 passed.

If `test_recorded_parameter_count` fails, print the count and check the channel progression against the encoder's before editing the constant — the two are deliberate mirrors.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/models/decoders.py tests/models/test_decoders.py
git commit -m "feat: pixel decoder shared by the trained and visualisation paths"
```

---

## Task 7: Standalone autoencoder trainer

**Files:**
- Create: `src/mbfps/training/__init__.py`
- Create: `src/mbfps/training/autoencoder.py`
- Create: `tests/training/__init__.py`
- Create: `tests/training/test_autoencoder.py`
- Create: `scripts/train_autoencoder.py`

**Interfaces:**
- Consumes: `Config`, `get_config`; `build_encoder`, `encoder_input_kind`; `PixelDecoder`, `reconstruction_loss`; `SequenceLoader`; `Prefetcher`; `ReplayBuffer`; `get_device`; `seed_everything`.
- Produces: `AutoencoderModel(cfg: Config)` with `.encoder`, `.decoder`, `.input_kind`, and `forward(batch: dict) -> tuple[Tensor, Tensor]` returning `(reconstruction, target_obs)`; `to_device(batch: dict, device: torch.device) -> dict` (public: Task 8's grid script uses it too); `train_autoencoder(cfg: Config, buffer: ReplayBuffer, out_dir: Path, log_every: int = 100) -> dict[str, Any]` returning a history dict with `"loss"`, `"steps"`, `"seconds"`, `"arm"`.

**No recurrence yet.** This is M2's whole point: most world-model bugs are representation bugs, and a broken autoencoder is far easier to see in a reconstruction grid than to infer from a degraded rollout. Adding the RSSM before this passes would mean debugging two things at once.

All three arms train encoder-then-decoder against the same pixel target, so their reconstruction grids are directly comparable. Arm 1 trains its CNN; Arms 2 and 3 train only their bottleneck, because the backbone is frozen and its features are cached.

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_autoencoder.py
import numpy as np
import pytest
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.envs.protocol import OBS_SHAPE
from mbfps.training.autoencoder import AutoencoderModel, train_autoencoder
from mbfps.utils.config import get_config

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def make_episode(t: int, fill: int) -> Episode:
    return Episode(
        obs=np.full((t + 1, *OBS_SHAPE), fill, dtype=np.uint8),
        actions=np.zeros(t, dtype=np.int32),
        rewards=np.zeros(t, dtype=np.float32),
        terminated=np.zeros(t, dtype=bool),
        truncated=np.zeros(t, dtype=bool),
        privileged=np.zeros((t + 1, len(KEYS)), dtype=np.float32),
        privileged_keys=KEYS,
        policy_name="random",
        seed=fill,
        scenario="my_way_home",
    )


@pytest.fixture
def buffer(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    rng = np.random.default_rng(0)
    for fill in (10, 60, 110):
        buf.add(make_episode(t=40, fill=fill))
    for path in buf.episode_paths():
        feats = rng.random((41, 64, 384)).astype(np.float16)
        np.save(path.with_suffix(".features.npy"), feats)
    return buf


def tiny(arm: str):
    return get_config(arm, steps=3, batch_size=2, seq_len=8, device="cpu")


def test_cnn_model_reconstructs_to_observation_shape(buffer):
    cfg = tiny("cnn")
    model = AutoencoderModel(cfg)
    batch = {"obs": torch.randint(0, 256, (2, 9, *OBS_SHAPE), dtype=torch.uint8)}
    recon, target = model(batch)
    assert recon.shape == target.shape
    assert recon.shape[-3:] == OBS_SHAPE


def test_ssl_model_consumes_features_and_reconstructs_pixels(buffer):
    cfg = tiny("frozen_ssl")
    model = AutoencoderModel(cfg)
    batch = {
        "obs": torch.randint(0, 256, (2, 9, *OBS_SHAPE), dtype=torch.uint8),
        "features": torch.randn(2, 9, 64, 384, dtype=torch.float16),
    }
    recon, target = model(batch)
    assert recon.shape == target.shape


def test_ssl_model_ignores_obs_as_input(buffer):
    """Feature arms must not sneak pixels into the encoder path."""
    cfg = tiny("frozen_ssl")
    model = AutoencoderModel(cfg)
    feats = torch.randn(2, 9, 64, 384, dtype=torch.float16)
    a = model({"obs": torch.zeros(2, 9, *OBS_SHAPE, dtype=torch.uint8),
               "features": feats})[0]
    b = model({"obs": torch.full((2, 9, *OBS_SHAPE), 255, dtype=torch.uint8),
               "features": feats})[0]
    assert torch.allclose(a, b), "reconstruction changed with obs; features arm leaked pixels"


@pytest.mark.parametrize("arm", ["cnn", "frozen_ssl", "random_vit"])
def test_training_runs_and_returns_history(buffer, arm):
    history = train_autoencoder(tiny(arm), buffer, out_dir=None)
    assert history["arm"] == arm
    assert history["steps"] == 3
    assert len(history["loss"]) == 3
    assert all(np.isfinite(history["loss"]))


def test_training_reduces_loss_on_a_trivial_dataset(buffer):
    """Every episode is a constant colour, so a working model must fit it fast."""
    history = train_autoencoder(
        get_config("cnn", steps=60, batch_size=2, seq_len=4, device="cpu", lr=1e-3),
        buffer,
        out_dir=None,
    )
    first, last = np.mean(history["loss"][:5]), np.mean(history["loss"][-5:])
    assert last < first * 0.6, f"loss barely moved: {first:.4f} -> {last:.4f}"


def test_ssl_arm_trains_only_its_bottleneck(buffer):
    """The frozen backbone must contribute no trainable parameters."""
    model = AutoencoderModel(tiny("frozen_ssl"))
    trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    assert trainable == 12_320


def test_checkpoint_written_when_out_dir_given(buffer, tmp_path):
    out = tmp_path / "run"
    train_autoencoder(tiny("cnn"), buffer, out_dir=out)
    assert (out / "autoencoder_cnn.pt").is_file()


def test_same_seed_reproduces_the_loss_curve(buffer):
    a = train_autoencoder(tiny("cnn"), buffer, out_dir=None)
    b = train_autoencoder(tiny("cnn"), buffer, out_dir=None)
    assert np.allclose(a["loss"], b["loss"]), "training is not reproducible from its seed"


def test_different_seeds_give_different_curves(buffer):
    a = train_autoencoder(get_config("cnn", steps=3, batch_size=2, seq_len=8,
                                     device="cpu", seed=0), buffer, out_dir=None)
    b = train_autoencoder(get_config("cnn", steps=3, batch_size=2, seq_len=8,
                                     device="cpu", seed=1), buffer, out_dir=None)
    assert not np.allclose(a["loss"], b["loss"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/training/test_autoencoder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.training'`

- [ ] **Step 3: Write the implementation**

```python
# src/mbfps/training/__init__.py
```

```python
# tests/training/__init__.py
```

```python
# src/mbfps/training/autoencoder.py
"""Standalone autoencoder training -- milestone M2.

No recurrence. Most world-model bugs are representation bugs, and a broken
autoencoder is obvious in a reconstruction grid but hard to infer from a
degraded rollout, so this stage is validated before dynamics are added.

All three arms reconstruct the same pixel target, which is what makes their
grids comparable. The pixel arm trains its CNN; the feature arms train only
their bottleneck, because the backbone is frozen and already cached.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.loader import SequenceLoader
from mbfps.data.prefetch import Prefetcher
from mbfps.models.decoders import PixelDecoder, reconstruction_loss
from mbfps.models.encoders import build_encoder, encoder_input_kind
from mbfps.utils.config import Config
from mbfps.utils.device import get_device
from mbfps.utils.seeding import seed_everything


class AutoencoderModel(nn.Module):
    """Encoder plus pixel decoder, with no recurrent state."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_kind = encoder_input_kind(cfg.encoder)
        self.encoder = build_encoder(cfg.encoder)
        self.decoder = PixelDecoder(
            in_dim=cfg.encoder.embed_dim, depth=cfg.encoder.cnn_depth
        )

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(reconstruction, pixel_target)`, both `(N, 112, 112, 3)`.

        The pixel target is always obs. The encoder *input* is obs only for the
        pixel arm; the feature arms read cached features and never see pixels.
        """
        target = batch["obs"]
        target = target.reshape(-1, *target.shape[-3:])
        if self.input_kind == "obs":
            source = target
        else:
            features = batch["features"]
            source = features.reshape(-1, *features.shape[-2:])
        return self.decoder(self.encoder(source)), target


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray) and key in ("obs", "features"):
            out[key] = torch.from_numpy(value).to(device)
    return out


def train_autoencoder(
    cfg: Config,
    buffer: ReplayBuffer,
    out_dir: Path | None,
    log_every: int = 100,
) -> dict[str, Any]:
    """Train one arm's autoencoder and return its loss history.

    Args:
        cfg: the arm's configuration.
        buffer: the frozen M1 dataset.
        out_dir: where to write a checkpoint, or None to skip writing.
        log_every: print a progress line this often.

    Returns:
        A history dict with `"arm"`, `"steps"`, `"loss"` (per-step floats),
        and `"seconds"`.
    """
    seed_everything(cfg.train.seed)
    device = get_device(prefer=cfg.train.device)
    model = AutoencoderModel(cfg).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    needs_features = model.input_kind == "features"
    loader = SequenceLoader(
        buffer,
        batch_size=cfg.train.batch_size,
        seq_len=cfg.train.seq_len,
        seed=cfg.train.seed,
        load_obs=True,  # always: obs is the reconstruction target for every arm
        load_features=needs_features,
    )

    losses: list[float] = []
    start = time.perf_counter()
    with Prefetcher(loader, depth=2) as prefetcher:
        stream = iter(prefetcher)
        for step in range(cfg.train.steps):
            batch = to_device(next(stream), device)
            optimiser.zero_grad()
            reconstruction, target = model(batch)
            loss = reconstruction_loss(reconstruction, target)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
            if log_every and (step + 1) % log_every == 0:
                recent = float(np.mean(losses[-log_every:]))
                print(f"[{cfg.arm}] step {step + 1}/{cfg.train.steps} loss={recent:.5f}")

    elapsed = time.perf_counter() - start
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"arm": cfg.arm, "state_dict": model.state_dict()},
            out_dir / f"autoencoder_{cfg.arm}.pt",
        )
    return {
        "arm": cfg.arm,
        "steps": cfg.train.steps,
        "loss": losses,
        "seconds": elapsed,
    }
```

- [ ] **Step 4: Write the training entry point**

```python
# scripts/train_autoencoder.py
"""Train one arm's standalone autoencoder (milestone M2)."""

import argparse
import json
from pathlib import Path

from mbfps.data.buffer import ReplayBuffer
from mbfps.training.autoencoder import train_autoencoder
from mbfps.utils.config import ARMS, get_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--out", type=Path, default=Path("runs/m2"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    cfg = get_config(
        args.arm,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
        device=args.device,
    )
    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    history = train_autoencoder(cfg, buffer, out_dir=args.out)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"history_{args.arm}.json").write_text(json.dumps(history))
    first = sum(history["loss"][:20]) / min(20, len(history["loss"]))
    last = sum(history["loss"][-20:]) / min(20, len(history["loss"]))
    print(f"arm={args.arm} steps={history['steps']} elapsed_s={history['seconds']:.0f}")
    print(f"steps_per_second={history['steps'] / history['seconds']:.2f}")
    print(f"loss_first20={first:.5f} loss_last20={last:.5f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/training/test_autoencoder.py -v`
Expected: 11 passed (one test is parametrised over the three arms).

`test_training_reduces_loss_on_a_trivial_dataset` is the one that matters: every episode is a single constant colour, so any working autoencoder fits it quickly. If it fails, the model is broken — do not raise the threshold to make it pass.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/training tests/training scripts/train_autoencoder.py
git commit -m "feat: standalone autoencoder training for all three arms"
```

---

## Task 8: M2 exit gate — train all three arms and produce the comparison grid

**Files:**
- Create: `scripts/reconstruction_grid.py`
- Modify: `docs/superpowers/plans/2026-09-02-mb-fps-m2-representation.md` (record measured numbers)

**Interfaces:**
- Consumes: everything from Tasks 1-7.
- Produces: `runs/m2/reconstruction_<arm>.png` and `runs/m2/reconstruction_all_arms.png`. No new library API.

This is the M2 exit gate: **reconstructions are visually recognisable, loss has plateaued, and a side-by-side grid exists for all three arms.**

**Disk workflow.** With 20 GB free, both caches coexist (5.86 GB total) and nothing needs clearing:

```
arm 1: needs no cache          -> train -> grid
arm 2: dinov2 (already cached) -> train -> grid
arm 3: cache random_vit        -> train -> grid
```

Check free space before the `random_vit` run. `cache_features.py` refuses if the result will not fit, which is intended behaviour, not a bug to work around.

- [ ] **Step 1: Write the grid script**

```python
# scripts/reconstruction_grid.py
"""Render original-vs-reconstruction pairs for a trained arm (milestone M2)."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mbfps.data.buffer import ReplayBuffer  # noqa: E402
from mbfps.data.loader import SequenceLoader  # noqa: E402
from mbfps.training.autoencoder import AutoencoderModel, to_device  # noqa: E402
from mbfps.utils.config import ARMS, get_config  # noqa: E402
from mbfps.utils.device import get_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--run", type=Path, default=Path("runs/m2"))
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    cfg = get_config(args.arm, batch_size=args.samples, seq_len=1, device=args.device)
    device = get_device(prefer=args.device)
    model = AutoencoderModel(cfg).to(device)
    checkpoint = torch.load(
        args.run / f"autoencoder_{args.arm}.pt", map_location=device
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    loader = SequenceLoader(
        buffer,
        batch_size=args.samples,
        seq_len=1,
        seed=0,
        load_obs=True,
        load_features=model.input_kind == "features",
    )
    with torch.no_grad():
        reconstruction, target = model(to_device(loader.sample(), device))

    recon = reconstruction.cpu().numpy()
    orig = target.cpu().numpy().astype(np.float32) / 255.0
    mse = float(((recon - orig) ** 2).mean())

    n = args.samples
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.8), squeeze=False)
    for i in range(n):
        axes[0][i].imshow(np.clip(orig[i], 0, 1))
        axes[1][i].imshow(np.clip(recon[i], 0, 1))
        for row in (0, 1):
            axes[row][i].set_xticks([])
            axes[row][i].set_yticks([])
    axes[0][0].set_ylabel("original", fontsize=11)
    axes[1][0].set_ylabel("reconstruction", fontsize=11)
    fig.suptitle(f"arm={args.arm}   pixel MSE={mse:.5f}", fontsize=13)
    fig.tight_layout()

    args.run.mkdir(parents=True, exist_ok=True)
    out_path = args.run / f"reconstruction_{args.arm}.png"
    fig.savefig(out_path, dpi=110)
    print(f"figure={out_path}")
    print(f"arm={args.arm} pixel_mse={mse:.5f} samples={n}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Train and render Arm 1 (no feature cache needed)**

```bash
.venv/bin/python scripts/train_autoencoder.py --arm cnn --steps 2000
.venv/bin/python scripts/reconstruction_grid.py --arm cnn
```

Record `steps_per_second`, `loss_first20`, `loss_last20`, and `pixel_mse`. Expected from the measured CNN cost (649 ms fwd+bwd for 1024 frames, plus the decoder): roughly 0.7-1.0 steps/s, so 2000 steps is 35-50 minutes.

- [ ] **Step 3: Cache DINOv2 features, train and render Arm 2, then clear**

```bash
ls data/my_way_home/*.features.npy | wc -l    # expect 122 already present
.venv/bin/python scripts/train_autoencoder.py --arm frozen_ssl --steps 2000
.venv/bin/python scripts/reconstruction_grid.py --arm frozen_ssl
```

The DINOv2 cache from M1 is already on disk, so no caching run is needed here. Record the same four numbers. Expected far faster than Arm 1 — no CNN in the loop.

- [ ] **Step 4: Cache the random-ViT features and do Arm 3**

The `random_vit` cache must live alongside the DINOv2 one, so it needs its own filename suffix — otherwise caching it would silently overwrite Arm 2's features and both arms would train on the same inputs, quietly destroying the control. `cache_features.py` writes `.features.npy` for `dinov2` and `.features_random_vit.npy` for `random_vit`; `SequenceLoader` selects by the same rule.

```bash
df -h . | tail -1                                    # confirm space before caching
.venv/bin/python scripts/cache_features.py --backbone random_vit --seed 0
ls data/my_way_home/*.features.npy | wc -l           # 122, dinov2, untouched
ls data/my_way_home/*.features_random_vit.npy | wc -l  # 122, new
.venv/bin/python scripts/train_autoencoder.py --arm random_vit --steps 2000
.venv/bin/python scripts/reconstruction_grid.py --arm random_vit
```

If `cache_features.py` refuses on free space, free more or reduce the dataset; do not disable the guard.

- [ ] **Step 5: Combine into the side-by-side grid**

```bash
.venv/bin/python -c "
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from pathlib import Path
arms = ['cnn','frozen_ssl','random_vit']
run = Path('runs/m2')
fig, axes = plt.subplots(len(arms), 1, figsize=(14, 5*len(arms)), squeeze=False)
for ax, arm in zip(axes[:,0], arms):
    ax.imshow(mpimg.imread(run/f'reconstruction_{arm}.png')); ax.axis('off')
fig.tight_layout(); out = run/'reconstruction_all_arms.png'; fig.savefig(out, dpi=110)
print('figure=', out)
"
```

- [ ] **Step 6: Evaluate the M2 gate**

The gate is qualitative by design — the spec asks for *visually recognisable* reconstructions. Open `runs/m2/reconstruction_all_arms.png` and check, for each arm:

1. Reconstructions show recognisable maze structure (walls, corridors, floor/ceiling boundaries), not uniform mush.
2. `loss_last20` is well below `loss_first20` and the curve has flattened. If it is still falling steeply at 2000 steps, train longer before judging.
3. All three arms rendered.

Record the three `pixel_mse` values. **Expect Arm 1 to reconstruct best** — it optimises pixels through a 26.4M-parameter encoder, while the feature arms reconstruct from a 12,320-parameter bottleneck over frozen features. A worse pixel MSE for the SSL arms is *not* a failure of this gate; the study's question is whether that representation predicts dynamics better, which M3 answers. Note the numbers and move on.

- [ ] **Step 7: Record the measured numbers in this plan**

Add a "## M2 results" section to this file containing the three arms' `steps_per_second`, `loss_first20`, `loss_last20`, and `pixel_mse`, plus the wall-clock for each run. The next plan sizes its training runs from these.

- [ ] **Step 8: Run the full suite and commit**

```bash
.venv/bin/python -m pytest -q
git add scripts/reconstruction_grid.py docs/superpowers/plans/2026-09-02-mb-fps-m2-representation.md
git commit -m "feat: M2 exit gate -- reconstruction grids for all three arms"
```

`runs/` and `data/` are gitignored; do not commit figures or the dataset.

---

## Exit criteria for this plan

- [ ] `pytest` fully green, with the counts each task states.
- [ ] All three arms train without error and their loss curves flatten.
- [ ] `runs/m2/reconstruction_all_arms.png` exists and shows recognisable structure for each arm.
- [ ] The three `pixel_mse` values and per-arm `steps_per_second` are recorded in this file.
- [ ] `SequenceLoader(load_obs=False)` verified to keep resident memory well under the 2.24 GB eager baseline.
- [ ] One feature cache on disk at the end, not two.

## Deferred to the next plan (M3)

The RSSM, KL balancing with free bits, reward and continue heads, the world-model trainer, open-loop rollout evaluation with an error-versus-horizon curve, and the linear probe from latent to `privileged_state`.

## Carried constraints

- **The study, not this plan, is the compute decision.** Measured: Arm 1 costs 1546 ms/step against 161 ms for the SSL arms — 9.6x. A 3-arm x 3-seed study is 31 h at 20k steps and 156 h at 100k, which is what the cloud budget was reserved for.
- **Both feature caches coexist** (5.86 GB of the 20 GB free), under distinct suffixes so neither can overwrite the other. The next plan inherits both and needs no re-caching.
- **Generate features once, on one device.** CPU and MPS differ in ~3% of float16 elements.
- **`my_way_home` coverage is near-saturated** at 61 episodes per policy; more of the same data will not add coverage.
