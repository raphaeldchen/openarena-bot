# MB-FPS M3c: Retire the End-to-End Pixel Arm, Add `pixel_ae`, Re-run the Study — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the pixel arm an informative embedding from step 0 by making M2's trained pixel autoencoder a third frozen backbone, then re-run 3 arms × 3 seeds under one code state and evaluate the M3 gate.

**Architecture:** Every arm becomes the same pipeline — frozen backbone → cached `(n_patches, patch_dim)` rows → the shared `BottleneckEncoder` → 2048 — differing only in what the backbone was pretrained on. Patch geometry moves from `EncoderConfig` to a backbone registry so the shared bottleneck can serve a `(64, 32)` cache beside the two `(64, 384)` ones. `cnn` is retired from `ARMS` but stays buildable under `KINDS` so M2 keeps working. Tasks 1–6 are code; Task 7 is the spike that replaces the free-bits calibration; Task 8 is the run.

**Tech Stack:** Python 3.12, PyTorch 2.13 (MPS), NumPy. Existing `mbfps` package. Study runs locally on MPS.

**Spec:** `docs/superpowers/specs/2026-09-11-mb-fps-m3c-pixel-ae-arm-design.md`
**Predecessor:** `docs/superpowers/plans/2026-09-04-mb-fps-m3b-study.md` (1,120 tests, complete; gate NOT PASSED)

---

## Global Constraints

- `src/mbfps/models/encoders.py` is the ONLY module that may differ in behaviour between arms. After this plan, no arm reads pixels at training time: `encoder_input_kind` is `"features"` for every arm in `ARMS`.
- `ARMS = ("pixel_ae", "frozen_ssl", "random_vit")` is what every M3 tool takes as `choices`. `KINDS = ("cnn",) + ARMS` is what `get_config` and `build_encoder` accept. `cnn` cannot be selected by any M3 tool and must keep building for M2.
- `bottleneck_dim = 32` and `embed_dim = 2048` are shared. `patch_dim` and `n_patches` are properties of the BACKBONE (`BACKBONE_GEOMETRY`), never of the arm config. The guard `n_patches × bottleneck_dim == embed_dim` runs per backbone.
- `KL_FREE_BITS = 0.20`, KL scales dyn/rep 0.5/0.1, h=512, z=32×32, head MLPs 2×512, batch/seq 16/64, LR 1e-4, fp32, 20,000 steps, split seed 0. **None of these change.** The spike (Task 7) validates 0.20 on the new arm; it does not search.
- The pixel-arm cache is the M2 encoder's 2048-d output partitioned into `(64, 32)` rows, float16, NO ImageNet normalisation (M2 trained on raw `uint8 / 255`). It invents nothing.
- Every guard is mutation-tested. **Harnesses must self-check three ways:** `PYTHONPATH` at the export (the package is installed editable), a known-fatal mutation first, `__pycache__` cleared with `PYTHONDONTWRITEBYTECODE=1`.
- Tests must not depend on `runs/m2_fixed/autoencoder_cnn.pt`. Build a tiny fake M2 checkpoint in `tmp_path` from a real `CNNEncoder` state_dict with `{"arm": "cnn", "state_dict": ...}`.
- Nothing under `runs/` or `data/` is committed. Unattended runs need `caffeinate -dimsu`.
- The M3b records in `runs/m3_study` are NOT compared against the new study (spec §5): different `encoders.py`, different code state, one arm renamed.

**Measured facts this plan is built on** (verified 2026-09-11 on this Mac, MPS, batch 16 × seq 64 — do not re-derive):

The embedding loss each arm minimises, at step 0 on real data, and the dyn KL at step 0:

| arm | `embedding_loss` at step 0 | dyn KL at step 0 |
|---|---|---|
| `cnn` (end-to-end, M3b) | **0.0008** | 0.022 |
| `frozen_ssl` | 0.3165 | 0.032 |
| `random_vit` | 0.4104 | 0.038 |

The end-to-end pixel arm's target is born collapsed — ~400× easier than the feature arms' — so nothing forces its posterior away from the prior, the KL never clears 0.20, and the `rep` term (the encoder's only non-degenerate gradient, since the embedding loss is detached) stays clamped at exactly `0.00e+00`. No free-bits value fixes it. Task 7's check (1) asserts the new arm lands in the feature-arm band.

Per-term gradient norm reaching each arm's encoder at step 0:

| term | `cnn` | `frozen_ssl` | `random_vit` |
|---|---|---|---|
| embedding (detached) | 1.0e-07 | 1.1e-05 | 2.6e-05 |
| reward | 7.4e-06 | 5.8e-05 | 6.5e-05 |
| continue | 4.1e-05 | 4.0e-04 | 4.5e-04 |
| KL clamped at 0.20 | **0.0** | **0.0** | **0.0** |
| KL `rep` unclamped | 2.1e-04 | 5.1e-03 | 1.6e-02 |

M2's pixel autoencoder (`runs/m2_fixed/autoencoder_cnn.pt`, keys `arm`, `state_dict`): encoder 26.38M params, four stride-2 convs to `(256, 7, 7)`, then `project: Linear(12544 → 2048)`. Its `project` layer is already a trained bottleneck to the shared space.

Bottleneck asymmetry, stated so it is not discovered: `pixel_ae`'s `Linear(32 → 32)` is 1,056 params; the ViT arms' `Linear(384 → 32)` is 12,320. Same class, same per-row treatment. Recorded as `encoder_params` in every record.

Costs (from M3b's nine records): feature arms 3.66–3.88 steps/s, **1.45–1.54 h per cell**. `pixel_ae` is a feature arm, so nine cells ≈ **13.5 h**, against the 38 h a free-bits re-run would have cost with the pixel arm at 9.3–10.1 h per cell.

Test sweep size (grepped 2026-09-11): 322 references to `"cnn"` across 14 test files. ~280 are study-arm labels in fixtures and become `"pixel_ae"`; 14 in `test_autoencoder.py` and the `CNNEncoder` tests in `test_encoders.py` mean the encoder class and stay.

---

## File Structure

| file | responsibility after this plan |
|---|---|
| `src/mbfps/data/features.py` | The backbone registry: `BACKBONES`, `BACKBONE_GEOMETRY`, `PIXEL_AE_CHECKPOINT`, `build_backbone`, `FeatureExtractor`, `cache_episode_features`. Owns "what a frozen backbone emits." |
| `src/mbfps/models/encoders.py` | The arm registry: `_ARM_BACKBONE`, `encoder_input_kind`, `encoder_backbone`, `build_encoder`; `CNNEncoder` (M2), `BottleneckEncoder` (every study arm). Owns "which encoder an arm builds." |
| `src/mbfps/utils/config.py` | `ARMS`, `KINDS`, `EncoderConfig` (no `patch_dim`), `TrainConfig`, `Config`, `get_config`. |
| `src/mbfps/data/loader.py` | `SequenceLoader` validates a cache's row shape against its backbone's geometry on first read. |
| `src/mbfps/eval/study.py` | `run_job` writes `git_sha`, `device`, `encoder_params`, `history`. |
| `scripts/cache_features.py` | `--backbone pixel_ae --checkpoint PATH`. |
| `scripts/run_study.py` | `REQUIRED_RECORD_KEYS` grows by four; `_SLOW_ARMS` is empty. |
| `scripts/spike_pixel_ae.py` | Task 7's three checks, exit 0 / 10. |
| `scripts/train_autoencoder.py`, `reconstruction_grid.py`, `eval_reconstruction.py` | `choices=KINDS`. |
| tests mirroring each | `tests/data/test_features.py`, `tests/data/test_loader.py`, `tests/models/test_encoders.py`, `tests/utils/test_config.py`, `tests/eval/test_study.py`, `tests/eval/test_run_study.py`, the sweep in Task 4, `tests/scripts/test_spike_pixel_ae.py`. |

---

> **Line numbers** in every Files block are as of commit `087e04a`. Each task shifts them for the next; every edit is also named by the symbol or test it touches — locate by that, treat the ranges as hints. Tasks are strictly sequential: Task N assumes Tasks 1..N-1 have landed and been committed.
>
> **Test counts** stated as `Expected: N passed` were computed against the same commit and are stale by every earlier task's additions. The binding expectation is the *delta* the step names (`17 more than…`) on top of what the previous task's last step measured; write the measured absolute number in the task report. Pre-flight (2026-09-11) corrected the absolute numbers where the cross-task check could compute them.

---

### Task 1: Backbone geometry registry; BottleneckEncoder reads it; patch_dim leaves EncoderConfig

The same two numbers -- 64 patches, 384 wide -- live in three places today: `N_PATCHES` / `FEATURE_DIM` in `features.py`, `_N_PATCHES` in `encoders.py`, and `patch_dim = 384` on `EncoderConfig`. All three describe the ViT backbones' output, none is a choice the study makes per arm, and every reader copies one of them. A third backbone whose rows are not 384 wide (Task 2's `pixel_ae` is `(64, 32)`) would be built against `Linear(384, 32)` with no error until the first matmul, and its cache-size estimate would be 12x too high. This task makes geometry a property of the backbone: one registry, `BACKBONE_GEOMETRY`, next to `BACKBONES`, and every reader goes through it. The encoder also gains the check it never had -- that the rows it is handed are the rows its backbone is registered for -- because `Linear(384, 32)` consumes `(N, 32, 384)` without complaint and emits the wrong embedding width.

Nothing about the arms changes here: `ARMS` still holds `cnn`, no arm is added, and both registered backbones stay `(64, 384)`, so every existing cache and fixture is valid. This task is the seam Task 2 cuts along.

**Files:**
- Modify: `src/mbfps/data/features.py` -- module docstring (lines 1-7); delete `N_PATCHES` / `FEATURE_DIM` (lines 19-23); add `BACKBONE_GEOMETRY` after `BACKBONES` (after line 31); `FeatureExtractor.encode` (lines 86-107)
- Modify: `src/mbfps/models/encoders.py` -- imports and delete `_N_PATCHES` (lines 13-22); `BottleneckEncoder` and the routing helpers it now calls (lines 50-112)
- Modify: `src/mbfps/utils/config.py` -- `EncoderConfig` (lines 21-29)
- Modify: `scripts/cache_features.py` -- imports (lines 20-27) and the disk estimate (line 58)
- Test: `tests/data/test_features.py` (modify: imports lines 6-11, `test_patch_grid_is_8x8` lines 22-24, and every `N_PATCHES` / `FEATURE_DIM` reader at lines 29, 129, 149, 169, 177, 198, 273, 306)
- Test: `tests/models/test_encoders.py` (modify: imports lines 6-15, `test_bottleneck_flattens_patch_grid_to_embed_dim` lines 85-88; insert a new block before line 204)
- Test: `tests/utils/test_config.py` (append)
- Test: `tests/data/test_cache_features_script.py` (create)

> `encoders.py` now imports `mbfps.data.features`, which imports `transformers` at module level. Measured: `import mbfps.models.encoders` goes from 0.5 s to ~1.5 s. It is on every training path, `transformers` is already a hard dependency, and the spec puts the registry in `features.py` beside `BACKBONES` -- so this is accepted and stated rather than discovered. There is no import cycle: `features.py` imports only `mbfps.envs.protocol` and `mbfps.utils.device` at the top, and `mbfps.data.loader` lazily inside `cache_episode_features`.

**Interfaces:**
- Consumes: `BACKBONES: tuple[str, ...] = ("dinov2", "random_vit")` and `build_backbone(kind, seed=0)` from `mbfps.data.features` (both unchanged here; Task 2 extends them); `EncoderConfig` fields `kind`, `embed_dim = 2048`, `cnn_depth = 32`, `bottleneck_dim = 32`, `standardise_features = True` from `mbfps.utils.config`; `_ARM_BACKBONE`, `encoder_backbone(cfg) -> str | None`, `encoder_input_kind(cfg) -> str`, `build_encoder(cfg) -> nn.Module` from `mbfps.models.encoders` (unchanged here; Task 2 adds `pixel_ae`); `ARMS = ("cnn", "frozen_ssl", "random_vit")` (unchanged here); `OBS_SHAPE` from `mbfps.envs.protocol`.
- Produces: `mbfps.data.features.BACKBONE_GEOMETRY: dict[str, tuple[int, int]] = {"dinov2": (64, 384), "random_vit": (64, 384)}` -- `(n_patches, patch_dim)` per backbone, defined directly after `BACKBONES`; `N_PATCHES` and `FEATURE_DIM` REMOVED from `mbfps.data.features`; `FeatureExtractor.encode(frames: np.ndarray) -> np.ndarray` now returns `(N, *BACKBONE_GEOMETRY[self.backbone])` float16 and raises `ValueError` naming the backbone, the registered geometry and the emitted one if they differ; `mbfps.models.encoders.BottleneckEncoder(cfg: EncoderConfig)` reads `n_patches, patch_dim = BACKBONE_GEOMETRY[encoder_backbone(cfg)]`, keeps the guard `n_patches * cfg.bottleneck_dim == cfg.embed_dim`, builds `self.bottleneck = nn.Linear(patch_dim, cfg.bottleneck_dim)` and `self.norm = nn.LayerNorm(patch_dim, elementwise_affine=False)` or `nn.Identity()`, exposes plain attributes `backbone: str`, `n_patches: int`, `patch_dim: int`, raises `ValueError` for a kind whose backbone is `None`, and its `forward(features)` raises `ValueError` naming backbone / expected / got when `features.shape[1:] != (n_patches, patch_dim)`; `_N_PATCHES` REMOVED from `mbfps.models.encoders`; `EncoderConfig.patch_dim` REMOVED (`EncoderConfig(kind, embed_dim=2048, cnn_depth=32, bottleneck_dim=32, standardise_features=True)`); `scripts/cache_features.py: cache_bytes(frames: int, backbone: str) -> int` = `frames * n_patches * patch_dim * 2`.

- [ ] **Step 1: Write the failing registry tests in `tests/data/test_features.py`**

This file is module-marked `slow` (line 14) because its fixture loads DINOv2; the three new tests inherit the mark and run with the rest of the 1120. Replace the import block (lines 6-11) and `test_patch_grid_is_8x8` (lines 22-24):

```python
# tests/data/test_features.py, lines 6-11 -- replace the import block
import mbfps.data.features as features
from mbfps.data.features import (
    BACKBONE_GEOMETRY,
    BACKBONES,
    FeatureExtractor,
    cache_episode_features,
)
```

```python
# tests/data/test_features.py -- replace test_patch_grid_is_8x8 (lines 22-24) with this block
# --- Backbone geometry ------------------------------------------------------
# `(n_patches, patch_dim)` belongs to the backbone, not to the study. The two
# ViT backbones share 112 / 14 = 8, an 8x8 = 64 patch grid, at DINOv2-small's
# hidden size of 384. Both numbers used to be module constants that every
# reader -- the encoder, the cache-size estimate, these tests -- copied; a
# third backbone with a different width would have been built against 384
# with no error until the first matmul. There is now one registry.


def test_backbone_geometry_registry():
    """Pins the literal geometry AND that every backbone has exactly one entry.

    The literal is deliberate: a registry that silently grows or loses a key
    is the failure mode, so a new backbone must edit this assertion by hand.
    """
    assert BACKBONE_GEOMETRY == {"dinov2": (64, 384), "random_vit": (64, 384)}
    assert set(BACKBONE_GEOMETRY) == set(BACKBONES), (
        "every registered backbone needs a geometry and vice versa"
    )


def test_the_old_module_constants_are_gone():
    """Every reader goes through the dict. A leftover `N_PATCHES` or
    `FEATURE_DIM` is a second source of truth that a future reader will copy."""
    assert not hasattr(features, "N_PATCHES")
    assert not hasattr(features, "FEATURE_DIM")


def test_encode_checks_its_output_against_the_registry(monkeypatch):
    """`encode`'s patch-count check must read the registry, not a literal 64.

    DINOv2 emits 64 patches, so on the real registry a hardcoded 64 and a
    registry read are indistinguishable. Rebinding the entry to a count the
    backbone cannot produce forces the difference: only a registry read
    raises. Built fresh rather than via the module fixture so the patched
    registry is what the extractor sees at encode time.
    """
    extractor = FeatureExtractor(device="cpu")
    frames = np.zeros((1, *OBS_SHAPE), dtype=np.uint8)
    assert extractor.encode(frames).shape == (1, 64, 384)  # sanity: unpatched passes
    monkeypatch.setitem(BACKBONE_GEOMETRY, "dinov2", (63, 384))
    with pytest.raises(ValueError, match=r"dinov2.*\(63, 384\).*\(64, 384\)"):
        extractor.encode(frames)
```

Then port every remaining reader of the deleted constants. Each is a one-line change; the left column is the current text, the right the replacement:

| line | current | replacement |
|---|---|---|
| 29 | `assert extractor.encode(frames).shape == (3, N_PATCHES, FEATURE_DIM)` | `assert extractor.encode(frames).shape == (3, *BACKBONE_GEOMETRY[extractor.backbone])` |
| 129 | `assert saved.shape == (4, N_PATCHES, FEATURE_DIM)` | `assert saved.shape == (4, *BACKBONE_GEOMETRY[extractor.backbone])` |
| 149 | `return np.zeros((len(frames), N_PATCHES, FEATURE_DIM), dtype=np.float16)` | `return np.zeros((len(frames), *BACKBONE_GEOMETRY[self.backbone]), dtype=np.float16)` |
| 169 | `np.save(dinov2_cache, np.full((4, N_PATCHES, FEATURE_DIM), 7, dtype=np.float16))` | `np.save(dinov2_cache, np.full((4, *BACKBONE_GEOMETRY["dinov2"]), 7, dtype=np.float16))` |
| 177 | `assert np.load(out_path).shape == (4, N_PATCHES, FEATURE_DIM)` | `assert np.load(out_path).shape == (4, *BACKBONE_GEOMETRY["random_vit"])` |
| 180-183 | `def test_backbones_registered():` with its local `from mbfps.data.features import BACKBONES` | drop the local import (it is now at module top); body stays `assert BACKBONES == ("dinov2", "random_vit")` |
| 198 | `assert ext.encode(frames).shape == (2, N_PATCHES, FEATURE_DIM)` | `assert ext.encode(frames).shape == (2, *BACKBONE_GEOMETRY["random_vit"])` |
| 272-274 | `markers[:, None, None], (len(frames), N_PATCHES, FEATURE_DIM)` | `markers[:, None, None], (len(frames), *BACKBONE_GEOMETRY[self.backbone])` |
| 306 | `assert written.shape == (n, N_PATCHES, FEATURE_DIM)` | `assert written.shape == (n, *BACKBONE_GEOMETRY["random_vit"])` |

The stubs at lines 149 and 273 read `self.backbone` (both are `"random_vit"`) rather than a literal so a stub that claims a backbone reports that backbone's geometry. When done, `grep -n "N_PATCHES\|FEATURE_DIM" tests/data/test_features.py` must hit only the `hasattr` lines and the docstring of `test_the_old_module_constants_are_gone`.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/data/test_features.py -q`
Expected: collection ERROR -- `ImportError: cannot import name 'BACKBONE_GEOMETRY' from 'mbfps.data.features'`.

- [ ] **Step 3: Implement the registry in `src/mbfps/data/features.py`**

Replace the module docstring (lines 1-7):

```python
"""Frozen-backbone patch-feature cache.

The backbone never updates, so each frame is encoded once at collection time and
the features are reused for every training step. The learned bottleneck lives
downstream, so this cache holds the raw `(n_patches, patch_dim)` grid rather
than a bottlenecked vector. The grid's shape is the BACKBONE's property, read
from `BACKBONE_GEOMETRY`, never a constant a reader copies.
"""
```

Delete lines 19-23 entirely (`N_PATCHES = 64`, its docstring, `FEATURE_DIM = 384`, its docstring, and the blank line after). Then insert the registry directly after `BACKBONES`'s docstring (after line 31, before `_MODEL_NAME`):

```python
BACKBONE_GEOMETRY: dict[str, tuple[int, int]] = {
    "dinov2": (64, 384),
    "random_vit": (64, 384),
}
"""`(n_patches, patch_dim)` of each backbone's cached rows, per frame.

Both ViT backbones tile 112x112 at patch 14 into an 8x8 = 64 grid and emit
DINOv2-small's 384-wide hidden state. That is a fact about the backbone, not
a choice the study makes per arm, so it lives here beside `BACKBONES` and
every reader -- `FeatureExtractor.encode`'s output check, the encoder's
bottleneck width and its `n_patches * bottleneck_dim == embed_dim` guard, the
cache-size estimate in scripts/cache_features.py -- takes it from this dict.
Before this registry the same two numbers were a module constant here, a
module constant in encoders.py and a field on `EncoderConfig`; a backbone
whose rows are not 384 wide would have been built against 384 with no error
until the first matmul. Every key of this dict is in `BACKBONES` and vice
versa; tests/data/test_features.py pins both the literal and that identity.
"""
```

Replace `FeatureExtractor.encode` (lines 86-107) so its output check reads the registry and names all three things the message needs -- the backbone, what it is registered as, and what it emitted:

```python
    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> np.ndarray:
        """Encode `(N, 112, 112, 3)` uint8 frames to `(N, n_patches, patch_dim)`
        float16, with the geometry taken from `BACKBONE_GEOMETRY[self.backbone]`.

        Raises:
            ValueError: if `frames` does not match the expected shape, or the
                backbone emits a grid other than the one registered for it.
        """
        if frames.ndim != 4 or frames.shape[1:] != OBS_SHAPE:
            raise ValueError(
                f"expected frames of shape (N, {OBS_SHAPE}), got {frames.shape}"
            )
        x = frames.astype(np.float32) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
        out = self.model(pixel_values=tensor).last_hidden_state
        patches = out[:, 1:, :]  # drop the CLS token
        # The registry is the contract every downstream reader builds against,
        # so a backbone that emits anything else is refused HERE, before a
        # single cache file is written with the wrong rows in it.
        expected = BACKBONE_GEOMETRY[self.backbone]
        if tuple(patches.shape[1:]) != expected:
            raise ValueError(
                f"backbone {self.backbone!r} is registered as {expected} per frame "
                f"but emitted {tuple(patches.shape[1:])}; check that the input is "
                f"112x112 and the patch size is 14"
            )
        return patches.to(torch.float16).cpu().numpy()
```

`build_backbone`, `require_free_bytes`, `FeatureExtractor.__init__` and `cache_episode_features` are untouched.

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/data/test_features.py -q`
Expected: PASS -- `22 passed` (20 before, minus `test_patch_grid_is_8x8`, plus 3).

`scripts/cache_features.py` is now broken (it still imports the deleted constants) and nothing in the suite imports it yet; Step 13 makes that a red test before Step 15 fixes it. Do not run `tests/models/test_encoders.py` yet -- it is unchanged and still green, because `encoders.py` has not started reading the registry.

- [ ] **Step 5: Write the failing encoder tests in `tests/models/test_encoders.py`**

Replace the import block (lines 6-15) -- it gains the module alias, the registry, and merges the duplicated `mbfps.utils.config` import:

```python
# tests/models/test_encoders.py, lines 6-15 -- replace
import mbfps.models.encoders as encoders
from mbfps.data.features import BACKBONE_GEOMETRY
from mbfps.envs.protocol import OBS_SHAPE
from mbfps.models.encoders import (
    BottleneckEncoder,
    CNNEncoder,
    build_encoder,
    encoder_backbone,
    encoder_input_kind,
)
from mbfps.utils.config import ARMS, EncoderConfig, get_config
```

Replace `test_bottleneck_flattens_patch_grid_to_embed_dim` (lines 85-88), which hardcodes the 64:

```python
def test_bottleneck_flattens_patch_grid_to_embed_dim():
    """64 patches x 32 bottleneck dims = 2048, matching the CNN arm exactly.

    The patch count is the BACKBONE's, read from the registry, not a number
    this test knows: a backbone with a different grid must satisfy the same
    identity with its own row count.
    """
    c = cfg("frozen_ssl")
    n_patches, _ = BACKBONE_GEOMETRY[encoder_backbone(c)]
    assert n_patches == 64
    assert n_patches * c.bottleneck_dim == c.embed_dim
```

Insert this block immediately before the `# --- Feature standardisation ---` comment (line 204):

```python
# --- Backbone geometry ------------------------------------------------------
# `(n_patches, patch_dim)` is a property of the frozen backbone whose cache the
# arm reads, not a choice the study makes per arm. Before this block the
# encoder carried a module constant `_N_PATCHES = 64` and `EncoderConfig` a
# shared `patch_dim = 384`, both describing the ViT backbones' output. A third
# backbone with a different row width would have been built against 384 with
# no error until the first matmul. The geometry now lives in ONE registry,
# `mbfps.data.features.BACKBONE_GEOMETRY`, and the encoder reads it.


def test_bottleneck_takes_its_geometry_from_the_backbone_registry(monkeypatch):
    """Both numbers must come from the registry -- neither may be hardcoded.

    Both registered backbones today are (64, 384), so building the real arms
    cannot tell a registry read from a literal 64 or 384 (fixture coincidence).
    Rebinding one backbone's entry to a geometry that matches NEITHER constant
    is what makes a hardcoded value fail: 128 rows of 16 still multiply out to
    2048 with `bottleneck_dim=16`, so the only way this build and forward pass
    succeed is if the encoder read the registry for both values.
    """
    monkeypatch.setitem(BACKBONE_GEOMETRY, "random_vit", (128, 16))
    enc = BottleneckEncoder(EncoderConfig(kind="random_vit", bottleneck_dim=16))
    assert enc.bottleneck.in_features == 16, "patch_dim must come from the registry"
    assert enc.bottleneck.out_features == 16
    assert tuple(enc.norm.normalized_shape) == (16,), (
        "the LayerNorm must be sized by the registry's patch_dim too"
    )
    assert enc(torch.randn(2, 128, 16)).shape == (2, 2048), (
        "n_patches must come from the registry"
    )


def test_bottleneck_guard_uses_the_registry_patch_count(monkeypatch):
    """`n_patches * bottleneck_dim == embed_dim` runs per backbone.

    With 128 rows the default `bottleneck_dim=32` gives 4096, not 2048, and
    the guard must say so. A guard still multiplying a literal 64 would accept
    this config and build an encoder that emits the wrong width.
    """
    monkeypatch.setitem(BACKBONE_GEOMETRY, "random_vit", (128, 16))
    with pytest.raises(ValueError, match=r"128 patches x bottleneck_dim 32"):
        BottleneckEncoder(cfg("random_vit"))


def test_bottleneck_guard_names_the_backbone_geometry():
    """The unchanged real-arm case: 64 x 33 != 2048, and the message says so."""
    with pytest.raises(ValueError, match=r"64 patches x bottleneck_dim 33.*embed_dim 2048"):
        BottleneckEncoder(EncoderConfig(kind="frozen_ssl", bottleneck_dim=33))


@pytest.mark.parametrize("arm", ["frozen_ssl", "random_vit"])
def test_bottleneck_forward_rejects_the_wrong_row_count(arm):
    """A cache with the wrong number of rows must not silently reshape.

    This is the dangerous case: `Linear(384, 32)` happily consumes
    `(N, 32, 384)` and emits `(N, 1024)`, which is the wrong embedding width
    and would only surface as a shape error deep inside the RSSM. The
    encoder must refuse at its own boundary and name what it expected.
    """
    enc = build_encoder(cfg(arm))
    backbone = encoder_backbone(cfg(arm))
    with pytest.raises(ValueError) as excinfo:
        enc(torch.randn(2, 32, 384))
    message = str(excinfo.value)
    assert backbone in message, "the error must name the backbone"
    assert "(64, 384)" in message, "the error must name the expected geometry"
    assert "(32, 384)" in message, "the error must name what it got"


def test_bottleneck_forward_rejects_the_wrong_row_width():
    """Wrong width would be a matmul error anyway; it must be OUR error."""
    enc = BottleneckEncoder(cfg("frozen_ssl"))
    with pytest.raises(ValueError, match=r"dinov2.*\(64, 384\).*\(64, 32\)"):
        enc(torch.randn(2, 64, 32))


def test_bottleneck_forward_rejects_a_flat_input():
    """`(N, 24576)` is the right number of values in the wrong shape."""
    enc = BottleneckEncoder(cfg("frozen_ssl"))
    with pytest.raises(ValueError, match=r"dinov2"):
        enc(torch.randn(2, 64 * 384))


def test_bottleneck_refuses_a_kind_that_reads_pixels():
    """`cnn` has no backbone and therefore no geometry to read."""
    with pytest.raises(ValueError, match=r"'cnn'.*pixels"):
        BottleneckEncoder(cfg("cnn"))


def test_no_second_source_of_truth_for_the_patch_count():
    """Every reader goes through the registry, so the module constant is gone.

    A leftover `_N_PATCHES` is how a future reader bypasses the dict. (The
    matching removal of `EncoderConfig.patch_dim` is guarded in
    tests/utils/test_config.py.)
    """
    assert not hasattr(encoders, "_N_PATCHES")
```

`test_bottleneck_guard_names_the_backbone_geometry` passes before Step 7 -- it pins the existing message format on a real arm so the registry rewrite cannot drop the numbers from it. Every other new test is red.

- [ ] **Step 6: Run to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/models/test_encoders.py -q`
Expected: `8 failed, 28 passed`. The eight, with the reason each fails:

| test | reason |
|---|---|
| `test_bottleneck_takes_its_geometry_from_the_backbone_registry` | `ValueError: 64 patches x bottleneck_dim 16 must equal embed_dim 2048` -- the guard multiplies `_N_PATCHES`, not the registry |
| `test_bottleneck_guard_uses_the_registry_patch_count` | `DID NOT RAISE` -- 64 x 32 = 2048 passes the literal guard |
| `test_bottleneck_forward_rejects_the_wrong_row_count[frozen_ssl]`, `[random_vit]` | `DID NOT RAISE` -- `(2, 32, 384)` flows through `Linear(384, 32)` and emits `(2, 1024)` |
| `test_bottleneck_forward_rejects_the_wrong_row_width` | `RuntimeError: Given normalized_shape=[384], expected input with shape [*, 384], but got input of size[2, 64, 32]` -- torch's error, not ours |
| `test_bottleneck_forward_rejects_a_flat_input` | same `RuntimeError`, from `[2, 24576]` |
| `test_bottleneck_refuses_a_kind_that_reads_pixels` | `DID NOT RAISE` -- `kind` is never read by the constructor |
| `test_no_second_source_of_truth_for_the_patch_count` | `assert not True` -- `_N_PATCHES` exists |

- [ ] **Step 7: Implement the registry-driven `BottleneckEncoder` in `src/mbfps/models/encoders.py`**

Replace lines 13-22 (the imports and the two module constants) -- `_SPATIAL` stays, `_N_PATCHES` goes:

```python
import torch
import torch.nn as nn

from mbfps.data.features import BACKBONE_GEOMETRY
from mbfps.utils.config import EncoderConfig

_SPATIAL = 7
"""112 / 2^4 = 7, the spatial size after four stride-2 convolutions."""
```

Replace lines 50-112 -- the whole of `BottleneckEncoder`, `encoder_input_kind`, `_ARM_BACKBONE` and `encoder_backbone` -- with the block below. The routing helpers move ABOVE the class unchanged, because the constructor now calls `encoder_backbone` and the module reads top to bottom. `CNNEncoder` (lines 25-47) and `build_encoder` (lines 115-125) are untouched.

```python
def encoder_input_kind(cfg: EncoderConfig) -> str:
    """Whether this arm's encoder consumes `"obs"` or cached `"features"`."""
    return "obs" if cfg.kind == "cnn" else "features"


_ARM_BACKBONE: dict[str, str | None] = {
    "cnn": None,
    "frozen_ssl": "dinov2",
    "random_vit": "random_vit",
}
"""Which cached feature set each arm reads. None means the arm reads pixels.

The arm name and the backbone name are deliberately not assumed equal:
`frozen_ssl` reads the `dinov2` cache. Deriving one from the other by string
identity would silently send the treatment arm to a cache that does not exist.
"""


def encoder_backbone(cfg: EncoderConfig) -> str | None:
    """Backbone whose cached features this arm consumes, or None for pixels.

    Raises:
        KeyError: if `cfg.kind` is not a registered arm.
    """
    if cfg.kind not in _ARM_BACKBONE:
        raise KeyError(f"unknown encoder kind {cfg.kind!r}")
    return _ARM_BACKBONE[cfg.kind]


class BottleneckEncoder(nn.Module):
    """Learned per-patch bottleneck over frozen-backbone features (Arms 2, 3).

    The backbone never updates, so its features are cached once at collection
    time and this module is the only trained part of the encoder path. A
    backbone's grid is far wider than the shared embedding (a ViT's is
    64 x 384 = 24576), so a per-row linear reduces each row to `bottleneck_dim`
    and the grid is flattened to exactly `embed_dim`.

    The grid's `(n_patches, patch_dim)` is the BACKBONE's property, read from
    `BACKBONE_GEOMETRY` for the backbone `cfg.kind` maps to. `bottleneck_dim`
    and `embed_dim` stay shared in `EncoderConfig`, so the identity
    `n_patches * bottleneck_dim == embed_dim` is checked per backbone.

    Every feature arm builds this same class; only the cached inputs differ.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        backbone = encoder_backbone(cfg)
        if backbone is None:
            raise ValueError(
                f"encoder kind {cfg.kind!r} reads pixels; BottleneckEncoder needs "
                "a kind that reads a frozen backbone's cached features"
            )
        n_patches, patch_dim = BACKBONE_GEOMETRY[backbone]
        if n_patches * cfg.bottleneck_dim != cfg.embed_dim:
            raise ValueError(
                f"{n_patches} patches x bottleneck_dim {cfg.bottleneck_dim} "
                f"must equal embed_dim {cfg.embed_dim} (backbone {backbone!r})"
            )
        # Plain attributes, not buffers: they must not enter the state_dict,
        # which the arm-parity tests compare key-for-key across feature arms.
        self.backbone = backbone
        self.n_patches = n_patches
        self.patch_dim = patch_dim
        self.bottleneck = nn.Linear(patch_dim, cfg.bottleneck_dim)
        # Non-learnable on purpose: it must add no parameters and no arm-varying
        # behaviour, only equalise the input scale the caches arrive at.
        # DINOv2's cache has std 2.3559 and random_vit's 1.0000, so without this
        # the treatment and control arms train at effectively different learning
        # rates and the contrast is confounded with optimisation conditioning.
        self.norm: nn.Module = (
            nn.LayerNorm(patch_dim, elementwise_affine=False)
            if cfg.standardise_features
            else nn.Identity()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map `(N, n_patches, patch_dim)` to `(N, embed_dim)` float32.

        Raises:
            ValueError: if the rows are not the registered geometry. The
                dangerous case is the wrong ROW COUNT with the right width:
                `Linear(384, 32)` consumes `(N, 32, 384)` without complaint
                and emits `(N, 1024)`, the wrong embedding width, which only
                surfaces as a shape error somewhere inside the RSSM. So the
                check is at this boundary and names the backbone it is for.
        """
        expected = (self.n_patches, self.patch_dim)
        if tuple(features.shape[1:]) != expected:
            raise ValueError(
                f"backbone {self.backbone!r} features must be (N, {self.n_patches}, "
                f"{self.patch_dim}), i.e. rows of {expected}, got rows of "
                f"{tuple(features.shape[1:])} from input shape {tuple(features.shape)}"
            )
        return self.bottleneck(self.norm(features.to(torch.float32))).flatten(1)
```

Every caller already hands the encoder a 3-D `(N, n_patches, patch_dim)` tensor -- `WorldModel.embed` flattens `(B, T+1, ...)` to `(B*(T+1), ...)` first, and `rollout.py`, `probe.py`, `study.py`, `diagnostics.py` index one episode's cache -- so the forward check changes nothing on the happy path. A 2-D or 4-D input has `shape[1:]` of length 1 or 3, which never equals a 2-tuple, so no separate `ndim` check is needed.

- [ ] **Step 8: Run to verify it passes**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/models/test_encoders.py -q`
Expected: PASS -- `36 passed`. The arm-parity tests (`test_ssl_arms_build_the_identical_module`, `test_ssl_arms_build_value_identical_modules`) and the pinned `12_320` parameter count stay green: the new attributes are plain ints, not buffers or parameters.

- [ ] **Step 9: Write the failing config tests -- append to `tests/utils/test_config.py`**

```python
# tests/utils/test_config.py -- append at end of file

# --- patch_dim is not an arm-level setting ---------------------------------
# It described the ViT backbones' 384-wide rows and was consumed in exactly two
# lines of BottleneckEncoder. A third backbone with a different width would
# have been built against 384 with no error. The width now lives with the
# backbone (`mbfps.data.features.BACKBONE_GEOMETRY`), and a dead field left
# here is how a future arm gets built against the wrong number.


def test_encoder_config_has_no_patch_dim_field():
    names = {f.name for f in dataclasses.fields(EncoderConfig)}
    assert "patch_dim" not in names, names


def test_encoder_config_rejects_patch_dim_at_construction():
    """The field must be GONE, not merely unused: a config that silently
    accepts `patch_dim=512` is the exact failure the removal exists to stop."""
    with pytest.raises(TypeError, match="patch_dim"):
        EncoderConfig(kind="frozen_ssl", patch_dim=512)


def test_encoder_config_keeps_the_shared_bottleneck_fields():
    """The removal must take ONLY patch_dim; the two shared numbers the
    per-backbone guard `n_patches * bottleneck_dim == embed_dim` reads stay."""
    c = EncoderConfig(kind="frozen_ssl")
    assert c.embed_dim == 2048
    assert c.bottleneck_dim == 32
    assert c.cnn_depth == 32
    assert c.standardise_features is True
```

`dataclasses`, `pytest` and `EncoderConfig` are already imported at the top of the file (lines 1-5).

- [ ] **Step 10: Run to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/utils/test_config.py -q`
Expected: `2 failed, 14 passed` -- `test_encoder_config_has_no_patch_dim_field` with `assert 'patch_dim' not in {'bottleneck_dim', 'cnn_depth', 'embed_dim', 'kind', 'patch_dim', 'standardise_features'}`, and `test_encoder_config_rejects_patch_dim_at_construction` with `DID NOT RAISE TypeError`.

- [ ] **Step 11: Remove `patch_dim` from `EncoderConfig` in `src/mbfps/utils/config.py`**

Replace lines 21-29 (the class header through `bottleneck_dim`; the `standardise_features` field and its docstring at lines 30-38 stay exactly as they are):

```python
@dataclass(frozen=True)
class EncoderConfig:
    """Encoder settings. `kind` is the ONLY field that may differ across arms.

    There is deliberately no `patch_dim` here. The width of a cached feature
    row is the frozen BACKBONE's property, not an arm-level setting, and lives
    in `mbfps.data.features.BACKBONE_GEOMETRY` beside its patch count. A
    shared 384 used to sit here and was read by exactly two lines of
    `BottleneckEncoder`; a backbone with narrower rows would have been built
    against 384 with no error until the first matmul. `bottleneck_dim` and
    `embed_dim` stay: they are the study's choices, held identical across
    arms, and the encoder checks `n_patches * bottleneck_dim == embed_dim`
    against each backbone's own patch count.
    """

    kind: str
    embed_dim: int = 2048
    cnn_depth: int = 32
    bottleneck_dim: int = 32
```

`ARMS`, `TrainConfig`, `Config` and `get_config` are untouched in this task. The existing `test_arms_differ_only_in_encoder_kind` (test_config.py lines 44-50) compares `dataclasses.asdict` of every arm's encoder against `cnn`'s; with the field gone it simply has one fewer key to compare and stays green.

- [ ] **Step 12: Run to verify it passes**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/utils/test_config.py tests/models/test_encoders.py -q`
Expected: PASS -- `52 passed` (16 + 36). Step 7's encoder no longer reads `cfg.patch_dim`, which is why this removal is safe to make after it and not before.

- [ ] **Step 13: Write the failing script test -- create `tests/data/test_cache_features_script.py`**

```python
# tests/data/test_cache_features_script.py
"""The cache-size estimate reads the backbone's geometry, not a constant.

`scripts/cache_features.py` refuses to start when the disk cannot hold the
cache it is about to write. That estimate used to multiply two module
constants, `N_PATCHES * FEATURE_DIM`, that described the ViT backbones only.
A backbone with narrower rows would have been sized as if it were 384 wide
and refused on a disk with room for it. The arithmetic is now a function that
reads the registry, so it can be tested without a buffer or a backbone.
"""

import importlib.util
from pathlib import Path

import pytest

from mbfps.data.features import BACKBONE_GEOMETRY

_SPEC = importlib.util.spec_from_file_location(
    "cache_features_script",
    Path(__file__).resolve().parents[2] / "scripts" / "cache_features.py",
)
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)


def test_cache_bytes_is_frames_times_geometry_times_float16():
    assert script.cache_bytes(10, "dinov2") == 10 * 64 * 384 * 2
    assert script.cache_bytes(10, "random_vit") == 10 * 64 * 384 * 2


def test_cache_bytes_reads_the_registry_not_a_constant(monkeypatch):
    """Both real backbones are (64, 384), so the test above cannot tell a
    registry read from the old constants. A geometry that matches neither
    constant can."""
    monkeypatch.setitem(BACKBONE_GEOMETRY, "random_vit", (64, 32))
    assert script.cache_bytes(10, "random_vit") == 10 * 64 * 32 * 2


def test_cache_bytes_rejects_an_unknown_backbone():
    with pytest.raises(KeyError):
        script.cache_bytes(10, "nope")
```

The script is loaded by path, exactly as `tests/eval/test_eval_rollout_script.py` does, because `scripts/` is not a package.

- [ ] **Step 14: Run to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/data/test_cache_features_script.py -q`
Expected: collection ERROR -- `ImportError: cannot import name 'FEATURE_DIM' from 'mbfps.data.features'` raised from the script's own import block. (This is the breakage Step 3 introduced, now caught by a test instead of by the next operator to run the script.)

- [ ] **Step 15: Port the disk estimate in `scripts/cache_features.py`**

Replace the import block (lines 20-27) and add `cache_bytes` directly above `main`:

```python
from mbfps.data.features import (
    BACKBONE_GEOMETRY,
    BACKBONES,
    FeatureExtractor,
    cache_episode_features,
    require_free_bytes,
)
from mbfps.data.loader import feature_suffix


def cache_bytes(frames: int, backbone: str) -> int:
    """Bytes the float16 cache of `frames` frames will occupy for `backbone`.

    Reads the backbone's own `(n_patches, patch_dim)`: a backbone with rows
    narrower than the ViTs' 384 must not be sized as if it were 384 wide, or
    a disk with room for its cache refuses to start the run.

    Raises:
        KeyError: if `backbone` has no registered geometry.
    """
    n_patches, patch_dim = BACKBONE_GEOMETRY[backbone]
    return frames * n_patches * patch_dim * 2  # float16
```

Then replace line 58 inside `main`:

```python
    needed = cache_bytes(frames, args.backbone)
```

`--backbone` already takes `choices=BACKBONES` (line 34), so `args.backbone` is always a registry key by the time `cache_bytes` runs. The `--checkpoint` flag is Task 2's.

- [ ] **Step 16: Run to verify it passes, then the whole suite**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/data/test_cache_features_script.py -q`
Expected: PASS -- `3 passed`.

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q`
Expected: PASS -- `1137 passed, 1 warning` (1120 before, minus `test_patch_grid_is_8x8`, plus 3 + 9 + 3 + 3). The suite includes real-data tests that read `data/my_way_home` relative to the cwd, so run it from the repo root. Verified: `grep -rn "N_PATCHES\|FEATURE_DIM\|patch_dim" src scripts` hits only docstrings, the `n_patches, patch_dim = BACKBONE_GEOMETRY[...]` reads in `encoders.py` and `cache_features.py`, and `BottleneckEncoder`'s `self.patch_dim` -- no constant and no config field.

- [ ] **Step 17: Mutation-test**

Set up a self-checked harness. The package is installed editable, so an export that is not on `PYTHONPATH` is silently shadowed by the working tree; the work is not yet committed, so export by copy rather than `git archive`:

```bash
# from the repo root
PY=$PWD/.venv/bin/python
MUT=/tmp/m3c_task1_mut && rm -rf $MUT && mkdir -p $MUT
rsync -a --exclude __pycache__ src scripts tests pyproject.toml $MUT/
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$MUT/src
# self-check 1: the harness imports the EXPORT, not the editable install
(cd $MUT && $PY -c "import mbfps; print(mbfps.__file__)")   # must print $MUT/src/mbfps/__init__.py
# self-check 2: a known-fatal mutation FIRST -- an unproven harness proves nothing
sed -i '' 's/^        backbone = encoder_backbone(cfg)$/        raise RuntimeError("harness self-check")/' $MUT/src/mbfps/models/encoders.py
(cd $MUT && $PY -m pytest tests/models/test_encoders.py -q -p no:cacheprovider)   # expected: 23 failed, 13 passed
rsync -a --exclude __pycache__ src/ $MUT/src/                                    # restore
# self-check 3: no stale bytecode can survive a same-length mutation
find $MUT -name __pycache__ -type d -exec rm -rf {} +
```

Apply each mutation to the export, run the named test(s) from `$MUT` with `-p no:cacheprovider`, restore with the `rsync` line, and clear `__pycache__` between runs. Each row was run and caught while writing this plan; the counts are what the harness printed.

| mutation | must be caught by |
|---|---|
| `encoders.py`: `n_patches, patch_dim = 64, 384` instead of the registry read | `test_bottleneck_takes_its_geometry_from_the_backbone_registry` (1 failed) |
| `encoders.py`: `nn.LayerNorm(384, ...)` instead of `patch_dim` | `test_bottleneck_takes_its_geometry_from_the_backbone_registry` (1 failed) |
| `encoders.py`: guard is `64 * cfg.bottleneck_dim != cfg.embed_dim` | `test_bottleneck_guard_uses_the_registry_patch_count` (1 failed) |
| `encoders.py`: forward check replaced by `if False:` | `test_bottleneck_forward_rejects_the_wrong_row_count[frozen_ssl]`, `[random_vit]`, `..._wrong_row_width`, `..._a_flat_input` (4 failed) |
| `encoders.py`: forward checks only `features.ndim != 3 or features.shape[1] != self.n_patches` (L3: half the compound) | `test_bottleneck_forward_rejects_the_wrong_row_width` (1 failed -- torch's `RuntimeError`, not ours) |
| `encoders.py`: forward checks only `features.shape[-1] != self.patch_dim` (L3: the other half) | `test_bottleneck_forward_rejects_the_wrong_row_count[frozen_ssl]`, `[random_vit]` (2 failed) |
| `encoders.py`: forward checks only `features[0].numel() != self.n_patches * self.patch_dim` (L2: what else satisfies the shape test) | `test_bottleneck_forward_rejects_a_flat_input` (1 failed; the row-count and row-width tests both pass under this mutation, which is why the flat test exists) |
| `encoders.py`: `if backbone is None:` replaced by `if False:` | `test_bottleneck_refuses_a_kind_that_reads_pixels` (1 failed -- `KeyError: None` instead of `ValueError`) |
| `encoders.py`: `_N_PATCHES = 64` restored, unused | `test_no_second_source_of_truth_for_the_patch_count` (1 failed) |
| `config.py`: `patch_dim: int = 384` restored | `test_encoder_config_has_no_patch_dim_field`, `test_encoder_config_rejects_patch_dim_at_construction` (2 failed) |
| `config.py`: `bottleneck_dim` removed along with it | `test_encoder_config_keeps_the_shared_bottleneck_fields` (1 failed) |
| `features.py`: registry gains `"pixel_ae": (64, 32)` without `BACKBONES` gaining it | `test_backbone_geometry_registry` (1 failed) |
| `features.py`: `N_PATCHES = 64` restored, unused | `test_the_old_module_constants_are_gone` (1 failed) |
| `features.py`: `encode` compares against a literal `(64, 384)` | `test_encode_checks_its_output_against_the_registry` (1 failed) |
| `features.py`: `encode`'s output check replaced by `if False:` | `test_encode_checks_its_output_against_the_registry` (1 failed) |
| `cache_features.py`: `return frames * 64 * 384 * 2` | `test_cache_bytes_reads_the_registry_not_a_constant` (1 failed) |
| `cache_features.py`: `* 4` (float32) instead of `* 2` | `test_cache_bytes_is_frames_times_geometry_times_float16` (1 failed) |

Any mutation that survives is a missing test. Add it before committing.

- [ ] **Step 18: Commit**

```bash
git add src/mbfps/data/features.py src/mbfps/models/encoders.py src/mbfps/utils/config.py scripts/cache_features.py tests/data/test_features.py tests/data/test_cache_features_script.py tests/models/test_encoders.py tests/utils/test_config.py
git commit -m "feat: patch geometry belongs to the backbone, not the arm -- one registry, and the bottleneck refuses rows that are not its backbone's

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: The pixel_ae backbone: build_backbone loads and freezes the M2 encoder; FeatureExtractor emits (64, 32)

The M3b pixel arm never entered the comparison because its target was born collapsed: a random-init CNN through the bottleneck emits a near-constant embedding (spec §1: `embedding_loss` 0.0008 at step 0 against 0.32–0.41 for the feature arms), so nothing ever forced the posterior to encode anything. Spec §2.1's fix is to give the pixel arm what the other two already have — a frozen backbone with an informative output from step 0 — and the backbone is the encoder half of the M2 pixel autoencoder (`runs/m2_fixed/autoencoder_cnn.pt`: four stride-2 convs to `(256, 7, 7)`, then `project: Linear(12544 → 2048)`, 26.38M params), which was trained on this very data.

This task registers that backbone in `features.py` and teaches `FeatureExtractor` to cache it as `(64, 32)` rows: the 2048-d vector partitioned row-major into 64 rows of 32, so that `64 × bottleneck_dim(32) == embed_dim(2048)` holds exactly as it does for the ViTs (spec §2.1: "invents nothing — it is a reshape"). It extends `BACKBONE_GEOMETRY`, the registry Task 1 introduced as the single source of truth for `(n_patches, patch_dim)`, with the third backbone's `(64, 32)`. **No arm changes here** — `pixel_ae` becomes a *backbone* in this task; `encoders.py` and `config.py` make it an *arm* in later tasks.

**Files:**
- Modify: `src/mbfps/data/features.py` — the file as Task 1 left it: `BACKBONES` gains `"pixel_ae"`; `BACKBONE_GEOMETRY` (Task 1's two-backbone registry) gains `"pixel_ae": (64, 32)`; new `PIXEL_AE_CHECKPOINT`, 36–52 (`build_backbone` gains `checkpoint` and the `pixel_ae` branch, backed by a new `_load_pixel_ae`), 74–107 (`FeatureExtractor.__init__` gains `checkpoint`; `encode` branches on backbone and validates against the registry). Lines 55–71 (`require_free_bytes`) and 110–141 (`cache_episode_features`) are untouched.
- Test: `tests/data/test_features.py` — the import block and `test_backbone_geometry_registry` from Task 1 (updated to three backbones), 180–190 (`test_backbones_registered`, `test_unknown_backbone_rejected`), then a new `pixel_ae` block appended after line 310.

> The M2 checkpoint is gitignored and absent on a fresh checkout, so no test may
> open `runs/m2_fixed/autoencoder_cnn.pt`. The checkpoint under test is built in
> `tmp_path` from a real `CNNEncoder`, in the exact `{"arm", "state_dict"}` layout
> `mbfps.training.autoencoder.train_autoencoder` writes (verified against the real
> file: keys `['arm', 'state_dict']`, ten `encoder.*` tensors from `encoder.conv.0.weight`
> to `encoder.project.bias`, `decoder.*` tensors alongside). It cannot be tiny in bytes:
> the contract fixes the architecture to `CNNEncoder(EncoderConfig(kind="cnn"))`, whose
> `project` layer alone is 12544 × 2048, so the file is ~105 MB and is written ONCE per
> module by a `scope="module"` fixture. Measured: the whole file runs in ~6 s.

> `tests/data/test_features.py` carries a module-level `pytestmark = pytest.mark.slow`
> (the dinov2 fixture downloads weights). The new tests inherit the mark although they
> download nothing; the suite runs with `-ra` only, so they are collected. Do not move
> them to a new file — the shape-parity tests against the ViT path need this module's
> `extractor` fixture.

**Interfaces:**
- Consumes (from Task 1): `mbfps.data.features.BACKBONE_GEOMETRY` as the two-backbone dict `{"dinov2": (64, 384), "random_vit": (64, 384)}` with `N_PATCHES`/`FEATURE_DIM` already removed; `scripts/cache_features.py::cache_bytes(frames: int, backbone: str) -> int`, which reads the dict — so adding `pixel_ae` to the dict is the ONLY change the cache script needs, and this task makes none to it.
- Consumes: `mbfps.models.encoders.CNNEncoder(cfg: EncoderConfig)` — forward maps `(N, 112, 112, 3)` uint8 to `(N, 2048)` float32 and owns its own `/255 - 0.5` and NHWC→NCHW; state_dict keys `conv.{0,2,4,6}.{weight,bias}`, `project.{weight,bias}`. `mbfps.utils.config.EncoderConfig(kind="cnn")` (defaults `embed_dim=2048`, `cnn_depth=32`). The M2 checkpoint layout `{"arm": str, "state_dict": {"encoder.*", "decoder.*"}}` from `mbfps.training.autoencoder.train_autoencoder`. `mbfps.data.loader.feature_suffix(backbone)` (already namespaces `pixel_ae` as `.features_pixel_ae.npy`). `mbfps.envs.protocol.OBS_SHAPE`, `mbfps.utils.device.get_device`.
- Produces, in `src/mbfps/data/features.py`:
  - `BACKBONES: tuple[str, ...] = ("dinov2", "random_vit", "pixel_ae")`
  - `BACKBONE_GEOMETRY: dict[str, tuple[int, int]] = {"dinov2": (64, 384), "random_vit": (64, 384), "pixel_ae": (64, 32)}` — `(n_patches, patch_dim)`, defined next to `BACKBONES`. `N_PATCHES` and `FEATURE_DIM` are REMOVED; every reader goes through the dict.
  - `PIXEL_AE_CHECKPOINT: Path = Path("runs/m2_fixed/autoencoder_cnn.pt")`
  - `build_backbone(kind: str, seed: int = 0, checkpoint: Path | None = None) -> torch.nn.Module` — for `pixel_ae`: loads `checkpoint` (default `PIXEL_AE_CHECKPOINT`), requires `ck["arm"] == "cnn"`, builds `CNNEncoder(EncoderConfig(kind="cnn"))`, loads the `encoder.*` subset, returns it frozen and in eval mode. `FileNotFoundError` naming the path if missing; `ValueError` if `arm != "cnn"`; `KeyError` for an unregistered kind (unchanged).
  - `FeatureExtractor.__init__(self, backbone: str = "dinov2", device: str = "mps", seed: int = 0, checkpoint: Path | None = None)`
  - `FeatureExtractor.encode(self, frames: np.ndarray) -> np.ndarray` — `(N, n_patches, patch_dim)` float16 per `BACKBONE_GEOMETRY[self.backbone]`. The `pixel_ae` path hands raw uint8 to the CNN (NO ImageNet normalisation) and reshapes `(N, 2048)` → `(N, 64, 32)` row-major.
  - `cache_episode_features(ep_path, extractor, batch_size=32) -> Path` — unchanged.

- [ ] **Step 1: Write the failing tests**

Three edits to `tests/data/test_features.py`. First, replace the import block and the `test_backbone_geometry_registry` test that Task 1 wrote at the top of the file (Task 1 already removed `test_patch_grid_is_8x8`) with:

```python
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from mbfps.data.episode import Episode, save_episode
import mbfps.data.features as features  # Task 1's `test_the_old_module_constants_are_gone` reads it
from mbfps.data.features import (
    BACKBONE_GEOMETRY,
    BACKBONES,
    PIXEL_AE_CHECKPOINT,
    FeatureExtractor,
    build_backbone,
    cache_episode_features,
)
from mbfps.envs.protocol import OBS_SHAPE

pytestmark = pytest.mark.slow  # downloads ~88MB on first run


@pytest.fixture(scope="module")
def extractor():
    return FeatureExtractor(device="cpu")


def test_backbone_geometry_is_the_literal_registry():
    """Literal, not derived: geometry is what the loader and the bottleneck
    size themselves by, so a typo here is a matmul error at step 0 (or, worse,
    a cache that loads and trains on the wrong width)."""
    assert BACKBONE_GEOMETRY == {
        "dinov2": (64, 384),
        "random_vit": (64, 384),
        "pixel_ae": (64, 32),
    }


def test_every_backbone_has_a_geometry_and_nothing_else_does():
    assert set(BACKBONE_GEOMETRY) == set(BACKBONES)
```

Task 1 already rewrote every shape assertion in the file (the dinov2 and random_vit shape tests, the stub extractors, the batching test) to read `BACKBONE_GEOMETRY[...]`, and `test_backbone_geometry_is_the_literal_registry` above pins that dict to literals — so the emitted shapes are checked against literals transitively. Do **not** add test-local `N_PATCHES`/`FEATURE_DIM` constants; nothing would read them.

Second, replace lines 180–190 (`test_backbones_registered` and `test_unknown_backbone_rejected`, which did function-local imports) with:

```python
def test_backbones_registered():
    assert BACKBONES == ("dinov2", "random_vit", "pixel_ae")


def test_unknown_backbone_rejected():
    with pytest.raises(KeyError, match="unknown backbone 'nope'"):
        build_backbone("nope")
```

Third, append after the last line of the file (line 310, the end of `test_cache_episode_features_batches_and_preserves_frame_order`):

```python


# --- pixel_ae: the M2 autoencoder's encoder as a frozen backbone --------------
# None of these tests touch `runs/m2_fixed/autoencoder_cnn.pt`: it is gitignored
# and absent on a fresh checkout. The checkpoint under test is built here from a
# real `CNNEncoder` in the exact `{"arm", "state_dict"}` layout that
# `mbfps.training.autoencoder.train_autoencoder` writes, with `encoder.*` keys
# next to `decoder.*` keys so the loader has to select the subset. It is a full
# 26.4M-parameter encoder (the contract fixes the architecture to
# `CNNEncoder(EncoderConfig(kind="cnn"))`, whose `project` layer is 12544 x
# 2048), so it is written ONCE per module rather than per test.


def _frames(seed: int, n: int = 2) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n, *OBS_SHAPE), dtype=np.uint8)


@pytest.fixture(scope="module")
def fake_m2(tmp_path_factory):
    """A fake M2 `cnn` checkpoint and the state_dict it was written from."""
    from mbfps.models.encoders import CNNEncoder
    from mbfps.utils.config import EncoderConfig

    torch.manual_seed(1234)
    encoder = CNNEncoder(EncoderConfig(kind="cnn"))
    state = {f"encoder.{k}": v.clone() for k, v in encoder.state_dict().items()}
    # Decoder keys, as in the real file. The loader must ignore them rather
    # than fail a strict load or, worse, try to fit them into the encoder.
    state["decoder.project.weight"] = torch.zeros(3, 2048)
    state["decoder.project.bias"] = torch.zeros(3)
    path = tmp_path_factory.mktemp("m2") / "autoencoder_cnn.pt"
    torch.save({"arm": "cnn", "state_dict": state}, path)
    return path, state


@pytest.fixture(scope="module")
def pixel_ae(fake_m2):
    path, _ = fake_m2
    return FeatureExtractor(backbone="pixel_ae", device="cpu", checkpoint=path)


def test_pixel_ae_checkpoint_default_is_the_m2_cnn_autoencoder():
    assert PIXEL_AE_CHECKPOINT == Path("runs/m2_fixed/autoencoder_cnn.pt")


def test_build_backbone_pixel_ae_loads_the_checkpoint_weights_not_a_fresh_init(fake_m2):
    """Every encoder tensor must equal the checkpoint's, so the features are
    the TRAINED autoencoder's and not a random CNN's -- which is the exact
    collapsed target the design retires."""
    from mbfps.models.encoders import CNNEncoder
    from mbfps.utils.config import EncoderConfig

    path, state = fake_m2
    loaded = build_backbone("pixel_ae", checkpoint=path).state_dict()
    expected = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    assert set(loaded) == set(expected)
    for key, tensor in expected.items():
        assert torch.equal(loaded[key], tensor), key
    # Self-check: a fresh init must NOT match, or the equality above is vacuous.
    fresh = CNNEncoder(EncoderConfig(kind="cnn")).state_dict()
    assert not torch.equal(fresh["project.weight"], expected["project.weight"])


def test_build_backbone_pixel_ae_is_frozen(fake_m2):
    path, _ = fake_m2
    model = build_backbone("pixel_ae", checkpoint=path)
    params = list(model.parameters())
    assert len(params) == 10  # 4 convs + project, weight and bias each
    assert all(not p.requires_grad for p in params)


def test_build_backbone_pixel_ae_is_in_eval_mode(fake_m2):
    path, _ = fake_m2
    assert build_backbone("pixel_ae", checkpoint=path).training is False


def test_build_backbone_pixel_ae_missing_checkpoint_names_the_path(tmp_path):
    """Matched on the guard's own wording, not just the exception type:
    `torch.load` raises its own FileNotFoundError with the path in it, so a
    type-only assertion passes with the guard deleted. The guard exists to
    say WHICH backbone wanted the file and how M2 produces it."""
    missing = tmp_path / "nowhere" / "autoencoder_cnn.pt"
    with pytest.raises(
        FileNotFoundError,
        match="pixel_ae checkpoint not found at " + re.escape(str(missing)),
    ):
        build_backbone("pixel_ae", checkpoint=missing)


def test_build_backbone_pixel_ae_rejects_a_non_cnn_checkpoint(fake_m2, tmp_path):
    """Only the arm tag differs from a loadable checkpoint. A loader that
    skipped the tag and relied on `load_state_dict` failing would pass a
    mismatched-weights test and still accept this file."""
    _, state = fake_m2
    wrong = tmp_path / "autoencoder_frozen_ssl.pt"
    torch.save({"arm": "frozen_ssl", "state_dict": state}, wrong)
    with pytest.raises(ValueError, match="frozen_ssl"):
        build_backbone("pixel_ae", checkpoint=wrong)


def test_build_backbone_pixel_ae_reads_the_default_path_when_none_is_given(
    fake_m2, monkeypatch
):
    """`checkpoint=None` means the module default, not "no checkpoint"."""
    path, _ = fake_m2
    monkeypatch.setattr(features, "PIXEL_AE_CHECKPOINT", path)
    assert build_backbone("pixel_ae").training is False
    monkeypatch.setattr(features, "PIXEL_AE_CHECKPOINT", path.with_name("absent.pt"))
    with pytest.raises(FileNotFoundError, match="absent.pt"):
        build_backbone("pixel_ae")


def test_pixel_ae_encode_output_shape_and_dtype(pixel_ae):
    out = pixel_ae.encode(_frames(0, n=3))
    assert out.shape == (3, 64, 32)
    assert out.dtype == np.float16


def test_pixel_ae_encode_is_byte_identical_on_repeat(pixel_ae):
    frames = _frames(1)
    assert np.array_equal(pixel_ae.encode(frames), pixel_ae.encode(frames))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS device"
)
def test_pixel_ae_encode_is_byte_identical_on_repeat_on_mps(fake_m2):
    """`cache_features.py` runs on MPS by default; CPU repeatability alone
    would not license the cache the study trains on."""
    path, _ = fake_m2
    ext = FeatureExtractor(backbone="pixel_ae", device="mps", checkpoint=path)
    frames = _frames(6)
    assert np.array_equal(ext.encode(frames), ext.encode(frames))


def test_pixel_ae_encode_is_the_2048_vector_partitioned_row_major(pixel_ae, fake_m2):
    """Row r of the (64, 32) grid is dims [32r, 32r+32) of the encoder's
    output. Any other partition -- transposed, or a (32, 64) grid swapped
    into place -- is a permutation of the same numbers, so shape cannot catch
    it; flattening back must reproduce the encoder's output bit-for-bit."""
    from mbfps.models.encoders import CNNEncoder
    from mbfps.utils.config import EncoderConfig

    _, state = fake_m2
    reference = CNNEncoder(EncoderConfig(kind="cnn")).eval()
    reference.load_state_dict(
        {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    )
    frames = _frames(2)
    with torch.no_grad():
        expected = reference(torch.from_numpy(frames)).to(torch.float16).numpy()
    assert expected.shape == (2, 2048)
    # Self-check: the values must be distinct enough that a permutation is
    # detectable, or this test could not tell row-major from any other order.
    assert np.unique(expected[0]).size > 1500

    flat = pixel_ae.encode(frames).reshape(2, 2048)
    assert np.array_equal(flat, expected)


def test_pixel_ae_encode_feeds_raw_uint8_with_no_imagenet_normalisation(pixel_ae, fake_m2):
    """M2 trained the CNN on `uint8 / 255 - 0.5`, which `CNNEncoder.forward`
    applies itself. ImageNet mean/std on top would shift every input off the
    distribution the weights were fit on. Asserts closeness to the raw path
    AND clear separation from the normalised one, so the check is not vacuous."""
    from mbfps.models.encoders import CNNEncoder
    from mbfps.utils.config import EncoderConfig

    _, state = fake_m2
    reference = CNNEncoder(EncoderConfig(kind="cnn")).eval()
    reference.load_state_dict(
        {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    )
    frames = _frames(3)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # What a copy-pasted ViT preprocessing branch would hand the CNN: floats
    # already scaled, which the CNN then divides by 255 again.
    normalised_input = (frames.astype(np.float32) / 255.0 - mean) / std
    with torch.no_grad():
        raw = reference(torch.from_numpy(frames)).numpy()
        normalised = reference(torch.from_numpy(normalised_input)).numpy()

    actual = pixel_ae.encode(frames).reshape(2, 2048).astype(np.float32)
    # Tolerances are RELATIVE to the output scale. A random-init CNN emits
    # values of order 5e-3 (the near-constant embedding of design §1), so an
    # absolute atol=1e-2 would accept the normalised path too and the test
    # would pass against either implementation. float16 keeps ~3 significant
    # digits; 5e-3 covers its round-trip and nothing more.
    scale = float(np.abs(raw).max())
    tol = dict(rtol=5e-3, atol=5e-3 * scale)
    assert np.allclose(actual, raw, **tol)
    assert not np.allclose(normalised, raw, **tol), (
        "raw and normalised encodings agree within tolerance; this test "
        "cannot detect whether normalisation is applied"
    )


def test_pixel_ae_encode_rejects_wrong_frame_shape(pixel_ae):
    with pytest.raises(ValueError, match="expected frames of shape"):
        pixel_ae.encode(np.zeros((2, 64, 64, 3), dtype=np.uint8))


def test_pixel_ae_encode_refuses_an_encoder_of_the_wrong_width(pixel_ae, monkeypatch):
    """A checkpoint with a different `embed_dim` would otherwise die inside
    `reshape` with a message about tensor sizes, not about the backbone."""
    monkeypatch.setattr(pixel_ae, "model", lambda x: torch.zeros(len(x), 2047))
    with pytest.raises(ValueError, match=r"pixel_ae.*2047.*64 x 32"):
        pixel_ae.encode(_frames(4))


def test_vit_encode_refuses_the_wrong_patch_count(extractor, monkeypatch):
    """The shared post-branch guard, exercised on the ViT path: 1 + 63 tokens
    (a wrong patch size) must be named, not silently cached as 63 rows."""

    class _Out:
        last_hidden_state = torch.zeros(2, 1 + 63, 384)

    monkeypatch.setattr(extractor, "model", lambda pixel_values: _Out())
    # registered (64, 384) first, emitted (63, 384) second -- Task 1's field order
    with pytest.raises(ValueError, match=r"dinov2.*\(64, 384\).*\(63, 384\)"):
        extractor.encode(_frames(5))


def test_pixel_ae_cache_writes_64_by_32_rows_to_its_own_suffix(tmp_path, pixel_ae):
    """End to end through `cache_episode_features`: the pixel_ae cache must
    land in `.features_pixel_ae.npy` with the registry's geometry, so the
    loader's shape validation and `feature_suffix` agree with what is on disk."""
    keys = ("health", "pos_x", "pos_y", "pos_z", "angle")
    episode = Episode(
        obs=np.zeros((4, *OBS_SHAPE), dtype=np.uint8),
        actions=np.zeros(3, dtype=np.int32),
        rewards=np.zeros(3, dtype=np.float32),
        terminated=np.zeros(3, dtype=bool),
        truncated=np.zeros(3, dtype=bool),
        privileged=np.zeros((4, len(keys)), dtype=np.float32),
        privileged_keys=keys,
        policy_name="random",
        seed=0,
        scenario="my_way_home",
    )
    path = tmp_path / "ep_000000_len00003.npz"
    save_episode(episode, path)
    out = cache_episode_features(path, pixel_ae)
    assert out.name == "ep_000000_len00003.features_pixel_ae.npy"
    saved = np.load(out)
    assert saved.shape == (4, 64, 32)  # literal: this is the on-disk contract
    assert saved.dtype == np.float16
```

Why these particular tests, by the species they defend against:
- **M8 (nothing requires the evaluated model to be the loaded one):** `..._loads_the_checkpoint_weights_not_a_fresh_init` compares every tensor to the file's and self-checks that a fresh init differs; `..._partitioned_row_major` and `..._no_imagenet_normalisation` compare `encode` against a reference built from the same state_dict, so a fresh-init or partial load fails three tests, not one.
- **L2 (what else satisfies this):** the missing-checkpoint test matches the guard's wording because `torch.load` raises its own `FileNotFoundError` with the path in it; a type-only assertion survives deleting the guard (measured — see Step 5).
- **L3 (compound conditions):** the wrong-arm checkpoint carries a *loadable* `cnn` state_dict with only the tag changed, so a loader that relied on `load_state_dict` failing cannot pass it.
- **L1 (fixture coincidence):** the row-major test asserts > 1500 distinct values in the 2048, and the normalisation test asserts the normalised alternative is *not* within tolerance, so neither can pass by accident of a degenerate fixture. The tolerance is relative to the output scale because a random-init CNN emits ~5e-3 — an absolute `atol=1e-2` accepted both implementations when first tried.
- **L5 (a variant never exercised):** the ViT path's guard is exercised on `dinov2`; the pixel path's on `pixel_ae`; MPS repeatability on both (the existing dinov2 test plus the new one).
- **L7 (parametrising over the collection under test):** `BACKBONES` and `BACKBONE_GEOMETRY` are asserted as literals; the ViT geometry the old tests pin is a test-local literal, not read from the registry.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/data/test_features.py -q`
Expected: FAIL at collection — `ImportError: cannot import name 'PIXEL_AE_CHECKPOINT' from 'mbfps.data.features'` (1 error; `BACKBONE_GEOMETRY` and `BACKBONES` already exist from Task 1, so the first name Python cannot resolve is the one this task adds).

- [ ] **Step 3: Implement**

Replace `src/mbfps/data/features.py` in full. `require_free_bytes` and `cache_episode_features` are byte-identical to before; everything above `require_free_bytes` and the whole of `FeatureExtractor` change.

```python
"""Frozen-backbone patch-feature cache.

A backbone never updates, so each frame is encoded once at collection time and
the features are reused for every training step. The learned bottleneck lives
downstream, so this cache holds the raw `(n_patches, patch_dim)` grid rather
than a bottlenecked vector. The grid's geometry is a property of the backbone,
recorded in `BACKBONE_GEOMETRY`: the two ViTs emit an 8x8 grid of 384-d patch
tokens; the M2 pixel autoencoder emits one 2048-d vector, partitioned into 64
rows of 32 so the same per-row bottleneck downstream can consume it.
"""

import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel

from mbfps.envs.protocol import OBS_SHAPE
from mbfps.utils.device import get_device

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BACKBONES: tuple[str, ...] = ("dinov2", "random_vit", "pixel_ae")
"""Frozen backbones. `dinov2` is the treatment arm's pretrained encoder;
`random_vit` is the control -- identical architecture, random weights, which is
what separates "pretraining helps" from "a stationary target helps"; `pixel_ae`
is the encoder of the M2 pixel autoencoder trained on this very data, which
makes the pixel arm a frozen-target arm like the other two instead of the
end-to-end arm that M3b measured never leaving its collapsed initial state."""

BACKBONE_GEOMETRY: dict[str, tuple[int, int]] = {
    "dinov2": (64, 384),
    "random_vit": (64, 384),
    "pixel_ae": (64, 32),
}
"""`(n_patches, patch_dim)` of each backbone's cached rows.

112 / 14 = 8, so a patch-14 ViT yields an 8x8 = 64 grid of hidden-size-384
tokens. The pixel autoencoder's encoder ends in `Linear(12544 -> 2048)`, one
vector with no spatial meaning; it is reshaped row-major into 64 rows of 32 so
that `64 * bottleneck_dim(32) == embed_dim(2048)` holds for it exactly as for
the ViTs. Every consumer of a cache's shape -- the loader's validation, the
bottleneck's `Linear` width, the cache-size estimate -- reads this dict; there
is deliberately no module constant for "the" patch count or width any more."""

PIXEL_AE_CHECKPOINT: Path = Path("runs/m2_fixed/autoencoder_cnn.pt")
"""Where M2 left the pixel autoencoder (`scripts/train_autoencoder.py --arm cnn`).
Gitignored, so a fresh checkout has to retrain it or copy it in."""

_MODEL_NAME = "facebook/dinov2-small"


def _load_pixel_ae(path: Path) -> torch.nn.Module:
    """Load the encoder half of an M2 `cnn` autoencoder checkpoint, frozen."""
    # Local import: `mbfps.models.encoders` reads `BACKBONE_GEOMETRY` from this
    # module at import time, so a top-level import here would be circular.
    from mbfps.models.encoders import CNNEncoder
    from mbfps.utils.config import EncoderConfig

    if not path.is_file():
        raise FileNotFoundError(
            f"pixel_ae checkpoint not found at {path}; M2 writes it with "
            "scripts/train_autoencoder.py --arm cnn (runs/ is gitignored)"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    arm = checkpoint.get("arm")
    if arm != "cnn":
        # The tag is checked, not just the tensors: the feature arms' M2
        # checkpoints hold a BottleneckEncoder whose keys would fail the strict
        # load below anyway, but with a message about tensor names rather than
        # about which file was handed in.
        raise ValueError(
            f"pixel_ae needs the M2 'cnn' autoencoder, but {path} has arm={arm!r}"
        )
    prefix = "encoder."
    encoder_state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    encoder = CNNEncoder(EncoderConfig(kind="cnn"))
    encoder.load_state_dict(encoder_state)  # strict: a missing tensor raises
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


def build_backbone(
    kind: str, seed: int = 0, checkpoint: Path | None = None
) -> "torch.nn.Module":
    """Construct a frozen backbone.

    Args:
        kind: one of `BACKBONES`.
        seed: RNG seed for `random_vit`. Ignored for `dinov2` and `pixel_ae`,
            whose weights are fixed. Verified: the same seed reproduces
            bit-identical weights.
        checkpoint: for `pixel_ae`, the M2 autoencoder file; None means
            `PIXEL_AE_CHECKPOINT`. Ignored for the ViT backbones.

    Raises:
        KeyError: if `kind` is not registered.
        FileNotFoundError: `pixel_ae` with no checkpoint at the path.
        ValueError: `pixel_ae` with a checkpoint whose `arm` is not `"cnn"`.
    """
    if kind not in BACKBONES:
        raise KeyError(f"unknown backbone {kind!r}; available: {list(BACKBONES)}")
    if kind == "dinov2":
        return AutoModel.from_pretrained(_MODEL_NAME)
    if kind == "pixel_ae":
        return _load_pixel_ae(
            PIXEL_AE_CHECKPOINT if checkpoint is None else Path(checkpoint)
        )
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


class FeatureExtractor:
    """Encodes frames with a frozen backbone."""

    def __init__(
        self,
        backbone: str = "dinov2",
        device: str = "mps",
        seed: int = 0,
        checkpoint: Path | None = None,
    ) -> None:
        self.backbone = backbone
        self.device = get_device(prefer=device)
        self.model = (
            build_backbone(backbone, seed=seed, checkpoint=checkpoint)
            .to(self.device)
            .eval()
        )
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> np.ndarray:
        """Encode `(N, 112, 112, 3)` uint8 frames to `(N, n_patches, patch_dim)`
        float16, with the geometry from `BACKBONE_GEOMETRY[self.backbone]`.

        Raises:
            ValueError: if `frames` does not match the expected shape, or the
                backbone emits something other than its registered geometry.
        """
        if frames.ndim != 4 or frames.shape[1:] != OBS_SHAPE:
            raise ValueError(
                f"expected frames of shape (N, {OBS_SHAPE}), got {frames.shape}"
            )
        n_patches, patch_dim = BACKBONE_GEOMETRY[self.backbone]
        if self.backbone == "pixel_ae":
            # Raw uint8 straight in, NO ImageNet statistics: `CNNEncoder.forward`
            # does its own `/255 - 0.5` and NHWC->NCHW, and M2 trained it on
            # exactly that. Normalising here would hand the frozen weights
            # inputs from a distribution they were never fit on.
            out = self.model(torch.from_numpy(frames).to(self.device))  # (N, 2048)
            if out.shape[1] != n_patches * patch_dim:
                raise ValueError(
                    f"pixel_ae encoder emitted {out.shape[1]} dims per frame, "
                    f"expected {n_patches} x {patch_dim} = {n_patches * patch_dim}"
                )
            patches = out.reshape(len(frames), n_patches, patch_dim)
        else:
            x = frames.astype(np.float32) / 255.0
            x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
            tensor = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
            out = self.model(pixel_values=tensor).last_hidden_state
            patches = out[:, 1:, :]  # drop the CLS token
        if tuple(patches.shape[1:]) != (n_patches, patch_dim):
            raise ValueError(
                f"backbone {self.backbone!r} is registered as {(n_patches, patch_dim)} per frame "
                f"but emitted {tuple(patches.shape[1:])}; check that the input is "
                f"112x112 and the patch size is 14"
            )  # Task 1's wording, which its `test_encode_checks_its_output_against_the_registry` pins (registered first, emitted second)
        return patches.to(torch.float16).cpu().numpy()


def cache_episode_features(
    ep_path: Path, extractor: FeatureExtractor, batch_size: int = 32
) -> Path:
    """Encode an episode's frames and write a sibling feature file.

    The output filename is namespaced by `extractor.backbone` (see
    `mbfps.data.loader.feature_suffix`), so caches for different backbones
    coexist instead of one overwriting the other.

    Called by `scripts/collect.py --cache-features` as each episode is written.
    Frames are encoded in batches: an episode can run to a thousand frames, and
    a single forward pass over all of them would exhaust unified memory.

    Returns:
        Path to the written feature file.
    """
    ep_path = Path(ep_path)
    # allow_pickle is left at its default of False, matching load_episode's
    # rationale in episode.py: obs is a plain uint8 array and round-trips
    # through npz without pickle, so there is no need to enable arbitrary
    # code execution on load.
    with np.load(ep_path) as data:
        obs = data["obs"]
    chunks = [
        extractor.encode(obs[i : i + batch_size]) for i in range(0, len(obs), batch_size)
    ]
    features = np.concatenate(chunks, axis=0)
    from mbfps.data.loader import feature_suffix

    out_path = ep_path.with_suffix(feature_suffix(extractor.backbone))
    np.save(out_path, features)
    return out_path
```

Three things in there that are deliberate and easy to "tidy" away:
- `_load_pixel_ae` imports `CNNEncoder` and `EncoderConfig` **inside the function**. The next task makes `mbfps.models.encoders` read `BACKBONE_GEOMETRY` from this module at import time; a top-level import here would then be a circular import in whichever direction is loaded first. This mirrors the existing function-local `feature_suffix` import in `cache_episode_features`, which exists for the same reason against `loader.py`.
- `FeatureExtractor.__init__` still calls `.eval()` and freezes parameters itself. That is redundant for `pixel_ae` (already frozen by `_load_pixel_ae`) but is what freezes the two ViTs; the contract puts the `pixel_ae` guarantee on `build_backbone` so a caller that never goes through `FeatureExtractor` gets a frozen model too.
- The `pixel_ae` branch has its own width check *before* `reshape`, and both branches share the geometry check after. Without the first, a checkpoint with a different `embed_dim` dies inside `reshape` with a message about tensor sizes; without the second, a ViT with the wrong patch size would cache 63 rows and the loader would refuse the file with no hint of why.

`scripts/cache_features.py` needs no edit: Task 1's `cache_bytes(frames, backbone)` already sizes the cache from `BACKBONE_GEOMETRY`, and `--backbone` already takes `choices=BACKBONES`, so `--backbone pixel_ae` parses and is sized correctly the moment the tuple and the dict gain it. The `--checkpoint` flag is Task 5's.

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/data/test_features.py -q`
Expected: PASS — 17 more than Task 1's Step 16 measured for this file (22 → **39**; on a box without MPS one of the new tests skips). Measured ~6 s.

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/cache_features.py --help`
Expected: the usage line, with `--backbone {dinov2,random_vit,pixel_ae}`. This is the only check on the script edit — `scripts/` is not a package and the size estimate runs after a real buffer load — so the import must be seen to succeed here rather than discovered on the first cache run.

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q`
Expected: 17 more than Task 1's Step 16 measured for the whole suite (1137 → **1154**). Measured: the full suite takes ~16 min on this Mac; the slow real-data tests read `data/my_way_home` relative to the cwd, so run it from the repo root.

- [ ] **Step 5: Mutation-test**

Set up a self-checked harness. The three self-checks all matter here: the package is installed editable, so a harness that copies the tree and forgets `PYTHONPATH` tests the *repo's* `features.py` and reports every mutation caught; a same-length mutation can leave a stale `.pyc`; and an unproven harness proves nothing. Save this as a scratch file (never commit it), run it, and read the first three lines before the table:

```python
# scratch/mutate_task2.py -- do not commit
import os, shutil, subprocess, sys
from pathlib import Path

REPO = Path("/Users/raphaelchen/Desktop/csgo-bot")
S = Path(__file__).resolve().parent / "tree"            # a copy of src/ tests/ scripts/ pyproject.toml
SRC = S / "src" / "mbfps" / "data" / "features.py"
PY = str(REPO / ".venv" / "bin" / "python")
ENV = dict(os.environ, PYTHONPATH=str(S / "src"), PYTHONDONTWRITEBYTECODE="1")

def clear_pyc():
    for d in S.rglob("__pycache__"):
        shutil.rmtree(d)

def run():
    clear_pyc()
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "tests/data/test_features.py"], cwd=S, env=ENV,
                       capture_output=True, text=True)
    failed = sorted({l.split("::")[-1].split(" ")[0] for l in r.stdout.splitlines()
                     if l.startswith(("FAILED", "ERROR"))})
    return r.returncode, failed, r.stdout.splitlines()[-1]

# self-check 1: the copy shadows the editable install
where = subprocess.run([PY, "-c", "import mbfps; print(mbfps.__file__)"], cwd=S, env=ENV,
                       capture_output=True, text=True).stdout.strip()
assert where.startswith(str(S)), where
print("shadowing OK:", where)

original = SRC.read_text()
MUTATIONS = {
 # self-check 2: a known-fatal mutation FIRST; the harness must see it
 "FATAL: BACKBONES emptied": ('BACKBONES: tuple[str, ...] = ("dinov2", "random_vit", "pixel_ae")', 'BACKBONES: tuple[str, ...] = ()'),
 "drop .eval() in _load_pixel_ae": ("    encoder.eval()\n    for param in encoder.parameters():", "    for param in encoder.parameters():"),
 "drop requires_grad loop": ("    for param in encoder.parameters():\n        param.requires_grad_(False)\n    return encoder", "    return encoder"),
 "skip load_state_dict (fresh init)": ("    encoder.load_state_dict(encoder_state)  # strict: a missing tensor raises\n", ""),
 "load conv.* only": ('        if key.startswith(prefix)\n    }', '        if key.startswith(prefix + "conv.")\n    }'),
 "drop arm check": ('    if arm != "cnn":', '    if False:'),
 "drop is_file guard": ("    if not path.is_file():", "    if False:"),
 "ignore checkpoint arg, always default": ("PIXEL_AE_CHECKPOINT if checkpoint is None else Path(checkpoint)", "PIXEL_AE_CHECKPOINT"),
 "ignore default when None": ("PIXEL_AE_CHECKPOINT if checkpoint is None else Path(checkpoint)", "Path(checkpoint)"),
 "extractor does not forward checkpoint": ("build_backbone(backbone, seed=seed, checkpoint=checkpoint)", "build_backbone(backbone, seed=seed)"),
 "imagenet-normalise pixel_ae": ("            out = self.model(torch.from_numpy(frames).to(self.device))  # (N, 2048)", "            x = (frames.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD\n            out = self.model(torch.from_numpy(x).to(self.device))"),
 "column-major partition": ("            patches = out.reshape(len(frames), n_patches, patch_dim)", "            patches = out.reshape(len(frames), patch_dim, n_patches).transpose(1, 2)"),
 "float32 output": ("        return patches.to(torch.float16).cpu().numpy()", "        return patches.to(torch.float32).cpu().numpy()"),
 "drop pixel_ae width guard": ("            if out.shape[1] != n_patches * patch_dim:", "            if False:"),
 "drop post-branch geometry guard": ("        if tuple(patches.shape[1:]) != (n_patches, patch_dim):", "        if False:"),
 "geometry pixel_ae (32, 64)": ('    "pixel_ae": (64, 32),', '    "pixel_ae": (32, 64),'),
 "geometry pixel_ae (64, 384)": ('    "pixel_ae": (64, 32),', '    "pixel_ae": (64, 384),'),
 "BACKBONES without pixel_ae": ('("dinov2", "random_vit", "pixel_ae")', '("dinov2", "random_vit")'),
 "default checkpoint path wrong": ('Path("runs/m2_fixed/autoencoder_cnn.pt")', 'Path("runs/m2_fixed/autoencoder_frozen_ssl.pt")'),
 "ViT path for every backbone": ('        if self.backbone == "pixel_ae":', '        if False:'),
}
try:
    rc, failed, last = run()
    assert rc == 0, ("baseline not green", last)
    print("baseline:", last)
    for name, (old, new) in MUTATIONS.items():
        assert original.count(old) == 1, (name, original.count(old))
        SRC.write_text(original.replace(old, new))
        rc, failed, last = run()
        SRC.write_text(original)
        print(f"{'CAUGHT' if rc else 'SURVIVED':8} | {name} | {failed[:3]}")
        if name.startswith("FATAL") and rc == 0:
            sys.exit("harness cannot detect a fatal mutation")
finally:
    SRC.write_text(original)
    clear_pyc()   # self-check 3: never leave bytecode from a mutated source behind
```

Build the copy with `mkdir -p scratch/tree && cp -R src tests scripts pyproject.toml scratch/tree/` and run `.venv/bin/python scratch/mutate_task2.py`. Every row must read `CAUGHT`:

| mutation | must be caught by |
|---|---|
| `BACKBONES = ()` (harness self-check, fatal) | 28 tests, starting with `test_backbones_registered` |
| `_load_pixel_ae` omits `.eval()` | `test_build_backbone_pixel_ae_is_in_eval_mode` |
| `_load_pixel_ae` omits the `requires_grad_(False)` loop | `test_build_backbone_pixel_ae_is_frozen` |
| `load_state_dict` skipped (fresh-init CNN returned) | `..._loads_the_checkpoint_weights_not_a_fresh_init`, `..._partitioned_row_major`, `..._no_imagenet_normalisation` |
| only `encoder.conv.*` loaded | `..._loads_the_checkpoint_weights_not_a_fresh_init` (and 10 others, via the strict-load RuntimeError) |
| `arm != "cnn"` check dropped | `test_build_backbone_pixel_ae_rejects_a_non_cnn_checkpoint` |
| `is_file` guard dropped (torch.load raises its own FileNotFoundError) | `test_build_backbone_pixel_ae_missing_checkpoint_names_the_path` — **this row survived a type-only `pytest.raises(FileNotFoundError, match=path)`**; the test matches the guard's wording for exactly this reason |
| `checkpoint` argument ignored, default always used | every `pixel_ae` test — on this Mac the real `runs/m2_fixed` file loads and `..._not_a_fresh_init` fails on weight inequality; on a fresh checkout they fail with FileNotFoundError |
| `checkpoint=None` no longer means the default (`Path(None)`) | `test_build_backbone_pixel_ae_reads_the_default_path_when_none_is_given` |
| `FeatureExtractor` does not forward `checkpoint` to `build_backbone` | the seven `pixel_ae` extractor tests, from `test_pixel_ae_encode_output_shape_and_dtype` |
| ImageNet mean/std applied on the `pixel_ae` path | `..._no_imagenet_normalisation`, `..._partitioned_row_major` |
| partition `reshape(N, 32, 64).transpose(1, 2)` instead of row-major `(N, 64, 32)` | `..._partitioned_row_major`, `..._no_imagenet_normalisation` |
| float32 returned instead of float16 | `test_pixel_ae_encode_output_shape_and_dtype`, `test_encode_output_dtype_is_float16`, both cache tests |
| `pixel_ae` width guard dropped | `test_pixel_ae_encode_refuses_an_encoder_of_the_wrong_width` (reshape's RuntimeError is not a ValueError) |
| post-branch geometry guard dropped | `test_vit_encode_refuses_the_wrong_patch_count` |
| `BACKBONE_GEOMETRY["pixel_ae"] = (32, 64)` | `test_backbone_geometry_is_the_literal_registry`, `test_pixel_ae_encode_output_shape_and_dtype` |
| `BACKBONE_GEOMETRY["pixel_ae"] = (64, 384)` | `test_backbone_geometry_is_the_literal_registry` and every `pixel_ae` encode test (width guard fires) |
| `BACKBONES` without `"pixel_ae"` | `test_backbones_registered`, `test_every_backbone_has_a_geometry_and_nothing_else_does`, every `pixel_ae` test (KeyError) |
| `PIXEL_AE_CHECKPOINT` points at the `frozen_ssl` file | `test_pixel_ae_checkpoint_default_is_the_m2_cnn_autoencoder` |
| `pixel_ae` routed through the ViT branch | every `pixel_ae` encode test (`pixel_values=` is not a `CNNEncoder` argument) |

All twenty rows were run and caught on 2026-09-11. Any mutation that survives is a missing test. Add it before committing.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/data/features.py tests/data/test_features.py
git commit -m "feat: the M2 pixel autoencoder is a frozen backbone, cached as (64, 32) rows through the backbone geometry registry

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: The loader refuses a feature cache whose rows do not match its backbone geometry

A feature cache is a bare `.npy`; nothing in it records which backbone wrote it. The filename suffix says which backbone the loader *thinks* it is reading, and the row geometry is the only property of the bytes that can contradict it. Until this task there was one geometry, `(64, 384)`, so the question never arose. With `pixel_ae` there are two -- `(64, 384)` for the ViTs, `(64, 32)` for the M2 pixel autoencoder -- and a cache written under the wrong suffix (or the flat `(T, 2048)` vector a future writer produces by forgetting the reshape in spec §2.1) would surface as a matmul shape error inside `BottleneckEncoder` at step 0, after every episode was loaded and the model built. Spec §2.2: "a cache whose rows do not match its backbone's geometry is refused at load, not discovered as a matmul error at step 0."

The loader already memmaps every feature file (`np.load(feature_path, mmap_mode="r")`, `loader.py:94`), so `.shape` comes from the 128-byte `.npy` header and the guard costs no frame read. It is checked on the **first** file only: one `cache_features.py` invocation writes one backbone's cache for every episode, so geometry is a property of the cache, not of a file, and one error at one place is what a reader wants.

**Depends on Task 1** having landed `BACKBONE_GEOMETRY` in `src/mbfps/data/features.py`. Step 4 cannot go green without it (the loader's import of the registry raises `ImportError`); Step 2 goes red with or without it.

**Files:**
- Modify: `src/mbfps/data/loader.py` -- module docstring (lines 8-14); a new `_check_feature_geometry` helper inserted after `feature_suffix` (after line 51); the `if load_features:` block of `SequenceLoader.__init__` (lines 83-94)
- Test: `tests/data/test_loader.py` -- the import (line 6); the three existing cache writers at lines 223 / 230, 242, 268-269, which write `(81, 4, 8)` rows that the guard now refuses under the `dinov2` suffix; `test_feature_suffix_namespaces_non_default_backbones` (lines 258-259); a new "Feature geometry" block appended after `test_obs_loading_loader_does_read_pixels` (after line 326, before the "Value-level window alignment" section)
- Test: `tests/data/test_split.py` -- line 97, the fourth `(81, 4, 8)` writer that reaches `SequenceLoader`

> Every other cache writer in the suite (`tests/training/test_world_model.py:238`, `tests/training/test_autoencoder.py:40/173/187/294`, `tests/eval/conftest.py:46`) already writes `(41, 64, 384)`, and the diagnostics/rollout tests that write `(N, 1, 1, 1)` or `(3, 2)` caches read them through `np.load` in `rollout.source_for`, never through `SequenceLoader`. Verified by running the whole suite with this task applied: 1131 collected = 1120 + the 11 tests added here, no other file moves. `tests/data/test_buffer.py`'s `(11, 4, 8)` files exercise eviction only and never reach a loader.

**Interfaces:**
- Consumes: `mbfps.data.features.BACKBONE_GEOMETRY: dict[str, tuple[int, int]]` (Task 1) -- `{"dinov2": (64, 384), "random_vit": (64, 384), "pixel_ae": (64, 32)}`, `(n_patches, patch_dim)`; `mbfps.data.loader.feature_suffix(backbone: str) -> str` (existing).
- Produces: `SequenceLoader.__init__(...)` -- when `load_features` is True, raises `KeyError` naming `feature_backbone` and listing `BACKBONE_GEOMETRY`'s keys if the backbone is unregistered (before any file lookup), and on the first feature file it reads raises `ValueError` if `features.shape[1:] != BACKBONE_GEOMETRY[feature_backbone]`, naming backbone, expected, got and the path. Also `mbfps.data.loader._check_feature_geometry(features: np.ndarray, backbone: str, expected: tuple[int, int], path: Path) -> None` (private helper the constructor calls). No signature changes to `SequenceLoader`; no new parameters.

- [ ] **Step 1: Write the failing tests**

Three edits to `tests/data/test_loader.py` first, all in existing code, then the new block.

(a) The import at line 6 gains `feature_suffix`, which the new fixtures use to name each backbone's cache:

```python
from mbfps.data.loader import SequenceLoader, feature_suffix
```

(b) The three existing cache writers become real `dinov2` geometry. They wrote `(81, 4, 8)` because nothing cared about the width; after this task the loader refuses that shape under the bare `.features.npy` suffix, so these tests would fail for the wrong reason at Step 4. In `test_features_window_matches_manual_slice` (lines 223 and 230):

```python
    for path in buf.episode_paths():
        # (64, 384) is dinov2's registered geometry; the loader now refuses
        # anything else under this suffix (see the geometry tests below).
        feats = rng.random((81, 64, 384)).astype(np.float16)
        np.save(path.with_suffix(".features.npy"), feats)

    loader = SequenceLoader(
        buf, batch_size=4, seq_len=16, seed=0, load_obs=False, load_features=True
    )
    batch = loader.sample()
    assert batch["features"].shape == (4, 17, 64, 384)
```

In `test_features_and_obs_windows_are_aligned` (line 242) -- note the dtype becomes `float32`: `np.arange(81 * 64 * 384)` runs to 1,990,655, and float16 tops out at 65,504, so a float16 arange would be `inf` for 97% of its cells and `inf == inf` would make the alignment assertion vacuous. float32 is exact for integers below 2^24:

```python
    feats = np.arange(81 * 64 * 384, dtype=np.float32).reshape(81, 64, 384)
    np.save(buf.episode_paths()[0].with_suffix(".features.npy"), feats)
```

In `test_loader_reads_the_requested_backbones_cache` (lines 268-269):

```python
    np.save(path.with_suffix(".features.npy"), np.zeros((81, 64, 384), np.float16))
    np.save(path.with_suffix(".features_random_vit.npy"), np.ones((81, 64, 384), np.float16))
```

And in `tests/data/test_split.py` line 97, the same substitution (this test constructs a `SequenceLoader` with `load_features=True` at line 103):

```python
        marker = float(path.stem.split("_len")[0].split("_")[1])
        # (64, 384) is dinov2's registered geometry; the loader refuses any
        # other row shape under the bare `.features.npy` suffix.
        feats = np.full((81, 64, 384), marker, dtype=np.float16)
        np.save(path.with_suffix(".features.npy"), feats)
```

(c) `test_feature_suffix_namespaces_non_default_backbones` (line 258) gains the third backbone, so the suffix the new tests write under is pinned:

```python
    assert feature_suffix("dinov2") == ".features.npy"
    assert feature_suffix("random_vit") == ".features_random_vit.npy"
    assert feature_suffix("pixel_ae") == ".features_pixel_ae.npy"
```

(d) Append this block after `test_obs_loading_loader_does_read_pixels` (after line 326), immediately before the `# --- Value-level window alignment` comment:

```python
# --- Feature geometry ---------------------------------------------------------
# A cache is a bare `.npy`; nothing in it records which backbone wrote it. The
# suffix says which backbone the loader THINKS it is reading, and the row
# geometry is the only property of the bytes that can contradict it. With
# three backbones and two geometries -- (64, 384) for the ViTs, (64, 32) for
# the M2 pixel autoencoder -- a cache written under the wrong suffix used to
# surface as a matmul shape error inside `BottleneckEncoder` at step 0, after
# the episode loading and model construction time was paid. It is refused at
# construction now, naming what was found and what the suffix promised.

# Literal, deliberately NOT `BACKBONE_GEOMETRY.items()`: parametrising over
# the registry would make these tests shrink silently if an entry were
# dropped, and would agree with whatever the registry said even if a width
# were edited to the wrong number.
_GEOMETRY_CASES = [
    ("dinov2", (64, 384)),
    ("random_vit", (64, 384)),
    ("pixel_ae", (64, 32)),
]


def _buffer_with_cache(tmp_path, backbone, rows, n_episodes=1, t=80):
    """`n_episodes` episodes, each with a `(t + 1, *rows)` cache for `backbone`."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    for i in range(n_episodes):
        buf.add(make_episode(t=t, fill=i + 1))
    for path in buf.episode_paths():
        np.save(path.with_suffix(feature_suffix(backbone)),
                np.zeros((t + 1, *rows), dtype=np.float16))
    return buf


def _feature_loader(buf, backbone):
    return SequenceLoader(
        buf, batch_size=2, seq_len=16, seed=0, load_obs=False,
        load_features=True, feature_backbone=backbone,
    )


def test_a_vit_geometry_cache_under_the_pixel_ae_suffix_is_refused(tmp_path):
    """The rows are 64 in both, so this is the width mismatch alone."""
    buf = _buffer_with_cache(tmp_path, "pixel_ae", (64, 384))
    with pytest.raises(ValueError, match="does not match backbone"):
        _feature_loader(buf, "pixel_ae")


def test_a_pixel_ae_geometry_cache_under_the_dinov2_suffix_is_refused(tmp_path):
    """The reverse direction: same row count, narrower rows than promised."""
    buf = _buffer_with_cache(tmp_path, "dinov2", (64, 32))
    with pytest.raises(ValueError, match="does not match backbone"):
        _feature_loader(buf, "dinov2")


def test_a_cache_with_the_right_width_but_wrong_row_count_is_refused(tmp_path):
    """The two tests above both carry 64 rows, so a guard comparing only the
    width passes them. This one has the right width and 16 rows, so a guard
    comparing only the row count is the one it catches -- together they pin
    the comparison to the whole `(n_patches, patch_dim)` pair."""
    buf = _buffer_with_cache(tmp_path, "dinov2", (16, 384))
    with pytest.raises(ValueError, match="does not match backbone"):
        _feature_loader(buf, "dinov2")


def test_an_unpartitioned_pixel_ae_cache_is_refused(tmp_path):
    """The M2 encoder emits a flat 2048-vector; the cache must hold it as
    64 rows of 32. A `(T, 2048)` file is exactly the mistake a future writer
    makes by forgetting the reshape, and its `shape[1:]` is `(2048,)`."""
    buf = _buffer_with_cache(tmp_path, "pixel_ae", (2048,))
    with pytest.raises(ValueError, match=r"got \(2048,\)"):
        _feature_loader(buf, "pixel_ae")


def test_the_refusal_names_backbone_expected_got_and_path(tmp_path):
    """Each of the four is asserted on its own, in labelled form: a message
    that carried both shapes but swapped `expected` and `got` would still
    contain both substrings, and would send the user to re-cache the wrong
    backbone."""
    buf = _buffer_with_cache(tmp_path, "pixel_ae", (64, 384))
    feature_path = buf.episode_paths()[0].with_suffix(".features_pixel_ae.npy")
    with pytest.raises(ValueError) as excinfo:
        _feature_loader(buf, "pixel_ae")
    message = str(excinfo.value)
    assert "'pixel_ae'" in message
    assert "expected rows of shape (64, 32)" in message
    assert "got (64, 384)" in message
    assert str(feature_path) in message


@pytest.mark.parametrize("backbone,rows", _GEOMETRY_CASES)
def test_every_registered_backbone_loads_a_cache_of_its_own_geometry(
    tmp_path, backbone, rows
):
    """The complement of the refusals: the guard must admit the shape the
    backbone actually writes, for EVERY backbone, or `pixel_ae` -- the one
    with the unusual width -- is the one a hardcoded `(64, 384)` rejects."""
    buf = _buffer_with_cache(tmp_path, backbone, rows)
    batch = _feature_loader(buf, backbone).sample()
    assert batch["features"].shape == (2, 17, *rows)


def test_the_guard_fires_on_the_first_cache_it_reads(tmp_path):
    """Every fixture above holds one episode, so "first", "last" and "any"
    are indistinguishable there. Two episodes, of which only the first is
    malformed, pin the guard to the first file it opens."""
    buf = _buffer_with_cache(tmp_path, "pixel_ae", (64, 32), n_episodes=2)
    first = buf.episode_paths()[0].with_suffix(".features_pixel_ae.npy")
    np.save(first, np.zeros((81, 64, 384), dtype=np.float16))
    with pytest.raises(ValueError, match="does not match backbone"):
        _feature_loader(buf, "pixel_ae")


def test_the_geometry_check_reads_the_header_not_the_frames(tmp_path):
    """The module docstring promises memmapped caches (0.04 GB to map all
    122 against 2.93 GB to read them), and nothing pinned it before the
    guard existed. A guard written as `np.asarray(np.load(...)).shape`, or a
    `np.load` that lost `mmap_mode`, would turn every construction into a
    full read of the cache. `.shape` on a memmap comes from the header."""
    buf = _buffer_with_cache(tmp_path, "pixel_ae", (64, 32))
    loader = _feature_loader(buf, "pixel_ae")
    assert isinstance(loader._features[0], np.memmap)


def test_an_unregistered_backbone_is_a_keyerror_listing_the_registry(tmp_path):
    """No cache exists for a name that is not a backbone, so this also pins
    the ORDER of the checks: the name is validated before the file lookup.
    Otherwise a typo raises FileNotFoundError and tells the user to run
    `cache_features.py --backbone <typo>`, which rejects the same name one
    step later."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=80, fill=1))
    with pytest.raises(KeyError) as excinfo:
        _feature_loader(buf, "vit_huge")
    message = str(excinfo.value)
    assert "'vit_huge'" in message
    for registered in ("dinov2", "random_vit", "pixel_ae"):
        assert registered in message


```

Why these and not fewer. The first two refusals both carry 64 rows, so a guard comparing only the width passes both, and the `(16, 384)` case exists to catch the guard comparing only the row count -- the pair is what pins `shape[1:]` as a whole (L3). The loads test runs every registered backbone with a literal geometry rather than iterating the dict (L5, L7): `pixel_ae` is the only backbone whose width is not 384, so a hardcoded `(64, 384)` survives every test that does not load a valid `pixel_ae` cache. The message test asserts four labelled substrings separately (L2): both shapes are present in a swapped message. The two-episode test exists because every other fixture has one episode, where "first", "last" and "any" coincide (L1). The `KeyError` test uses a name with no cache on disk, so the ordering of the two checks is observable.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/data/test_loader.py -q`
Expected: **7 failed, 31 passed** -- six `Failed: DID NOT RAISE ValueError` (the four refusals, the message test, the two-episode test) and `test_an_unregistered_backbone_is_a_keyerror_listing_the_registry` failing with `FileNotFoundError: no cached features for ep_000000_len00080.npz; expected ep_000000_len00080.features_vit_huge.npy. Run scripts/cache_features.py --backbone vit_huge first.` -- the misdirection the ordering test is written against. The loads test, the memmap test and the reshaped existing tests already pass: they are the complement, and must keep passing after Step 3.

- [ ] **Step 3: Implement**

Three edits to `src/mbfps/data/loader.py`.

(a) The module docstring (lines 8-14). The "Two arms" sentence is stale once all three study arms are feature arms, and the memmap paragraph is where the guard's cost argument belongs:

```python
The study's arms never read pixels at all -- they train on cached features
from a frozen backbone; only M2's end-to-end `cnn` encoder reads obs. Loading
obs for a feature arm costs 2.24 GB of resident memory for nothing (measured),
so `load_obs=False` skips it. `.npz` decompresses lazily per key, so not
reading `obs` genuinely avoids the cost.

Feature files are memmapped rather than loaded: mapping all 122 costs 0.04 s
and 0.04 GB, versus 2.93 GB to read them. The first file's row geometry is
checked against the backbone registry at construction, from the header alone:
a cache is a bare `.npy` that does not record which backbone wrote it, and
with two geometries in play -- (64, 384) for the ViTs, (64, 32) for the M2
pixel autoencoder -- a wrong-suffix cache used to surface as a matmul error
in the encoder at step 0, after the loading time was paid.
"""
```

(b) The helper, inserted after `feature_suffix` (after line 51, before `class SequenceLoader`):

```python
def _check_feature_geometry(
    features: np.ndarray, backbone: str, expected: tuple[int, int], path: Path
) -> None:
    """Refuse a cache whose rows are not `backbone`'s registered geometry.

    `features` is the memmap `np.load(..., mmap_mode="r")` returned, so
    `.shape` comes from the `.npy` header and no frame is read. The message
    names all four things a reader needs -- which backbone the suffix
    promised, what that backbone writes, what the file holds, and which file
    -- because the fix is always "re-cache this backbone" and the user must
    not have to reopen the file to know that.

    Raises:
        ValueError: if `features.shape[1:] != expected`.
    """
    got = tuple(features.shape[1:])
    if got != tuple(expected):
        raise ValueError(
            f"feature cache {path} does not match backbone {backbone!r}: "
            f"expected rows of shape {tuple(expected)}, got {got}. "
            f"Re-run scripts/cache_features.py --backbone {backbone}."
        )
```

`shape[1:]`, not `shape[-2:]`: a flat `(T, 2048)` cache has `shape[-2:] == (T, 2048)`, which is refused either way, but `shape[1:] == (2048,)` is what the message should say the file holds.

(c) The `if load_features:` block of `SequenceLoader.__init__` (lines 83-94) becomes:

```python
        self._features: list[np.ndarray | None] = []
        if load_features:
            # Imported here rather than at module top: `mbfps.data.features`
            # pulls in `transformers` (measured 1.53 s against this module's
            # 0.09 s), and the pixel `cnn` kind and M2's autoencoder tools
            # construct loaders that never read a cache. The registry is the
            # one source of truth for a backbone's row geometry --
            # `BottleneckEncoder` reads the same dict -- so what the loader
            # admits is exactly what the encoder's `Linear` can consume.
            from mbfps.data.features import BACKBONE_GEOMETRY

            # Before the file loop: an unregistered name has no cache either,
            # and the FileNotFoundError below would send the user to
            # `cache_features.py --backbone <typo>`, which refuses the same
            # name one step later.
            if feature_backbone not in BACKBONE_GEOMETRY:
                raise KeyError(
                    f"unknown feature backbone {feature_backbone!r}; "
                    f"available: {list(BACKBONE_GEOMETRY)}"
                )
            expected = BACKBONE_GEOMETRY[feature_backbone]
            suffix = feature_suffix(feature_backbone)
            for index, path in enumerate(self._paths):
                feature_path = path.with_suffix(suffix)
                if not feature_path.is_file():
                    raise FileNotFoundError(
                        f"no cached features for {path.name}; expected "
                        f"{feature_path.name}. Run scripts/cache_features.py "
                        f"--backbone {feature_backbone} first."
                    )
                features = np.load(feature_path, mmap_mode="r")
                # The first file only. One `cache_features.py` invocation
                # writes one backbone's cache for every episode, so the
                # geometry is a property of the cache, not of a file, and
                # one error at one place is what a reader wants.
                if index == 0:
                    _check_feature_geometry(
                        features, feature_backbone, expected, feature_path
                    )
                self._features.append(features)
```

The lazy import is safe with respect to the circular dependency that already exists: `features.py` imports `feature_suffix` from this module inside `cache_episode_features` (line 137), also lazily, so neither module-level import touches the other. The `KeyError` is raised explicitly rather than letting `BACKBONE_GEOMETRY[feature_backbone]` raise, because the bare dict `KeyError` carries only the missing key and not the list of valid names. `sample()` is untouched: `np.asarray(self._features[idx][start : end + 1])` still slices the memmap per window.

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/data/test_loader.py tests/data/test_split.py -q`
Expected: **PASS (48 tests)** -- 38 in `test_loader.py` (27 existing + 11 new: 9 functions, the loads test x3), 10 in `test_split.py`.

Then the whole suite, because the guard is on the path of every feature-arm training and evaluation test:

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q`
Expected: **1131 passed** (1120 + 11), 3 skipped, same as before this task. If any test outside `tests/data/` now fails with `does not match backbone`, it writes a feature cache in a shape its backbone never produces and reads it through `SequenceLoader`; fix the fixture's shape, not the guard. (Verified with this task applied: none does.)

- [ ] **Step 5: Mutation-test**

Set up a self-checked harness. All three checks below have silently lied in this project (M3b plan, Global Constraints):

```bash
# 1. The package is installed EDITABLE, so a naive export is shadowed by
#    src/. Run with PYTHONPATH at the mutated copy and confirm it wins.
S=/tmp/m3c-task3 && rm -rf $S && mkdir -p $S && cp -R src tests scripts pyproject.toml $S/
cd $S && PYTHONPATH=$S/src PYTHONDONTWRITEBYTECODE=1 \
  /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -c "import mbfps; print(mbfps.__file__)"
# must print /tmp/m3c-task3/src/mbfps/__init__.py, NOT .../csgo-bot/src/...

# 2. An unproven harness proves nothing: run the known-fatal mutation FIRST
#    (M0 below, delete the `_check_feature_geometry` call) and confirm SIX
#    tests fail before trusting any other row.

# 3. A same-byte-length mutation can leave a stale .pyc that survives a
#    correct restore: before EVERY run,
find $S/src -name __pycache__ -exec rm -rf {} + ; export PYTHONDONTWRITEBYTECODE=1
```

Apply each mutation to `$S/src/mbfps/data/loader.py`, run `PYTHONPATH=$S/src .venv/bin/python -m pytest tests/data/test_loader.py -q -p no:cacheprovider` from `$S`, restore from the repo copy, and confirm the restored file passes 38 again at the end.

| mutation | must be caught by |
|---|---|
| M0 (known-fatal, run first): delete the `if index == 0: _check_feature_geometry(...)` call | all six refusal/message/two-episode tests: `test_a_vit_geometry_cache_under_the_pixel_ae_suffix_is_refused`, `test_a_pixel_ae_geometry_cache_under_the_dinov2_suffix_is_refused`, `test_a_cache_with_the_right_width_but_wrong_row_count_is_refused`, `test_an_unpartitioned_pixel_ae_cache_is_refused`, `test_the_refusal_names_backbone_expected_got_and_path`, `test_the_guard_fires_on_the_first_cache_it_reads` |
| M1: compare the width only -- `if got[-1] != expected[-1]:` | `test_a_cache_with_the_right_width_but_wrong_row_count_is_refused` (the ONLY test that catches it -- both suffix-swap tests carry 64 rows) |
| M2: compare the row count only -- `if got[0] != expected[0]:` | `test_a_vit_geometry_cache_under_the_pixel_ae_suffix_is_refused`, `test_a_pixel_ae_geometry_cache_under_the_dinov2_suffix_is_refused` |
| M3: `expected = (64, 384)` hardcoded instead of the registry lookup | `test_every_registered_backbone_loads_a_cache_of_its_own_geometry[pixel_ae-rows2]` (a valid `pixel_ae` cache is refused) and `test_a_vit_geometry_cache_under_the_pixel_ae_suffix_is_refused` (an invalid one is admitted) |
| M4: swap `expected` and `got` in the message | `test_the_refusal_names_backbone_expected_got_and_path` (labelled substrings), `test_an_unpartitioned_pixel_ae_cache_is_refused` (`got (2048,)` no longer appears) |
| M5: drop `{path}` from the message | `test_the_refusal_names_backbone_expected_got_and_path` |
| M6: delete the explicit `KeyError` guard and let `BACKBONE_GEOMETRY[feature_backbone]` raise bare | `test_an_unregistered_backbone_is_a_keyerror_listing_the_registry` (the bare `KeyError('vit_huge')` names no registered backbone) |
| M7: `if index == len(self._paths) - 1:` (validate the last file, not the first) | `test_the_guard_fires_on_the_first_cache_it_reads` (the ONLY test that catches it -- every other fixture has one episode) |
| M8: `features = np.asarray(np.load(feature_path, mmap_mode="r"))` (a full read) | `test_the_geometry_check_reads_the_header_not_the_frames` |
| M9: `got = tuple(features.shape)` (compare including `T`) | every loads test, `test_features_window_matches_manual_slice`, `test_features_and_obs_windows_are_aligned`, `test_loader_reads_the_requested_backbones_cache` |
| M10: move the `KeyError` guard to after the file loop (`expected = BACKBONE_GEOMETRY.get(feature_backbone, (64, 384))` before it) | `test_an_unregistered_backbone_is_a_keyerror_listing_the_registry` (raises `FileNotFoundError` instead) |

All eleven were run against this exact test file and this exact implementation; each row's catcher is what actually fired, and M1 and M7 are each caught by exactly one test. Any mutation that survives is a missing test. Add it before committing.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/data/loader.py tests/data/test_loader.py tests/data/test_split.py
git commit -m "feat: the loader refuses a feature cache whose rows do not match its backbone geometry

A cache is a bare .npy that does not record which backbone wrote it. With
pixel_ae's (64, 32) rows beside the ViTs' (64, 384), a cache under the
wrong suffix surfaced as a matmul error in BottleneckEncoder at step 0.
SequenceLoader now checks the first file's shape[1:] against
BACKBONE_GEOMETRY from the memmap header, naming backbone, expected, got
and path, and rejects an unregistered backbone name before looking for
its (nonexistent) cache.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

---

### Task 4: Retire `cnn` from the study: `ARMS`, `KINDS`, `get_config`, the arm registry, `_SLOW_ARMS`, and the test sweep

The end-to-end pixel arm never entered the M3b comparison (spec §1: `embedding_loss` 0.0008 at step 0, dynamics KL above the free-bits floor on 15–18 of 20,000 steps, prior never trained). Spec §2.3 retires it from the *study* without deleting it from the *codebase*: `ARMS` becomes `("pixel_ae", "frozen_ssl", "random_vit")`, a second tuple `KINDS = ("cnn",) + ARMS` names everything `build_encoder` can construct, and `get_config` validates against `KINDS` so M2's autoencoder scripts and the shipped M3b checkpoints keep building `cnn`. Every M3 tool already takes `choices=ARMS`, so the retirement cascades through the driver, report, aggregation, rollout evaluation, diagnostics and pooling by changing one tuple — and through ~430 test references that use `"cnn"` as an arm label.

This task assumes Tasks 1–3 have landed: `BACKBONE_GEOMETRY` exists in `mbfps.data.features` with a `"pixel_ae": (64, 32)` entry, `BottleneckEncoder.__init__` reads `BACKBONE_GEOMETRY[encoder_backbone(cfg)]`, and `SequenceLoader` validates feature shape against it. Until this task adds `"pixel_ae"` to `_ARM_BACKBONE`, `encoder_backbone(EncoderConfig(kind="pixel_ae"))` raises `KeyError`, so no test before this task can have exercised the `pixel_ae` bottleneck; the `Linear(32, 32)` assertion in Step 5 is the first end-to-end proof that the registry, the geometry and the routing agree.

**Files:**
- Modify: `src/mbfps/utils/config.py` — lines 16–18 (`ARMS`), 63–77 (`get_config`)
- Modify: `src/mbfps/models/encoders.py` — lines 1–11 (module docstring), 86–125 (`encoder_input_kind`, `_ARM_BACKBONE`, `encoder_backbone`, `build_encoder`); line numbers are HEAD's, Task 2 may have shifted them
- Modify: `scripts/run_study.py` — lines 1–8 (module docstring), 81–87 (`_SLOW_ARMS`), 333 (comment), 411–417 (`quarantine_record` docstring)
- Modify: `scripts/train_autoencoder.py` — lines 9, 14
- Modify: `scripts/reconstruction_grid.py` — lines 18, 24
- Modify: `scripts/eval_reconstruction.py` — lines 30, 66–67, 78
- Test (rewrite in place): `tests/utils/test_config.py` — lines 1–27, 30–33, 44–47
- Test (modify): `tests/models/test_encoders.py` — lines 6–15, 91–110, 124–131, 185–186
- Test (modify): `tests/training/test_world_model.py` — line 10, 233–241
- Test (modify): `tests/eval/test_run_study.py` — lines 700–711, 1038–1077, 1483–1514, 1728–1733, plus the mechanical sweep
- Test (modify): `tests/eval/test_study.py` — lines 45–53, 161–162, 186–188, 196–197, 717, 1606, 1717–1718, 1734–1760
- Test (modify): `tests/eval/conftest.py` — lines 43–48
- Test (sweep + 2 hand edits): `tests/eval/test_aggregate.py` — lines 2511–2512, 2908
- Test (sweep only): `tests/eval/test_diagnose_dynamics_script.py`, `tests/eval/test_pooling.py` (its local `ARMS` at line 49 included), `tests/eval/test_pool_dynamics_script.py`
- Test (comment only): `tests/eval/test_diagnostics.py` — line 2815
- Test (create): `tests/training/test_m2_scripts_take_kinds.py`
- NOT touched, and why: `tests/training/test_autoencoder.py` (14 refs) and the `CNNEncoder` tests in `tests/models/test_encoders.py` mean the encoder CLASS, which still exists; `tests/eval/test_rollout.py:631` (`real_model_and_probe(arm="cnn")`) is the same species as `tiny()` — the real-RSSM rollout tests need an encoder that reads no feature file, and `get_config("cnn")` is permitted by `KINDS`; `tests/eval/fixtures/shuffled_cnn_seed0_pre_ladder.json` and `SHIPPED_ARM = "cnn"` in `test_diagnostics.py` pin an M3b artefact (`runs/m3_study/world_model_cnn_seed0.pt`) bitwise and must keep naming the arm that produced it; `tests/data/test_split.py:39` iterates a literal tuple that never reaches the registry; prose in `scripts/diagnose_dynamics.py`, `pool_dynamics.py`, `src/mbfps/eval/diagnostics.py`, `pooling.py` quoting `cnn`'s measured −0.036 R² describes M3b's measurements and stays.

> Two tuples, not one, because two milestones read them. `ARMS` is what the study
> can *select*; `KINDS` is what the code can *build*. A study record, checkpoint or
> diagnostic that says `arm="cnn"` continues to mean the end-to-end pixel arm of
> M3b, and nothing can read a `pixel_ae` artefact under the old name or vice
> versa, because no M3 tool offers `cnn` and `job_record_path` puts the arm in the
> filename.

**Interfaces:**
- Consumes:
  - `mbfps.data.features.BACKBONE_GEOMETRY: dict[str, tuple[int, int]]` (Task 1) — read by the tests to shape feature inputs; `{"dinov2": (64, 384), "random_vit": (64, 384), "pixel_ae": (64, 32)}`
  - `mbfps.models.encoders.BottleneckEncoder(cfg: EncoderConfig)` (Task 2) — reads `BACKBONE_GEOMETRY[encoder_backbone(cfg)]`, so with this task's registry entry `pixel_ae` builds `Linear(32, 32)`
  - `mbfps.models.encoders.CNNEncoder(cfg: EncoderConfig)` — unchanged
  - `mbfps.utils.config.EncoderConfig(kind: str, embed_dim: int = 2048, cnn_depth: int = 32, bottleneck_dim: int = 32, standardise_features: bool = True)` — `patch_dim` already removed by Task 1
  - `mbfps.eval.study.StudyJob`, `run_job`, `job_record_path` — unchanged
- Produces:
  - `mbfps.utils.config.ARMS: tuple[str, ...] = ("pixel_ae", "frozen_ssl", "random_vit")`
  - `mbfps.utils.config.KINDS: tuple[str, ...] = ("cnn",) + ARMS`
  - `mbfps.utils.config.get_config(arm: str, **overrides) -> Config` — raises `KeyError` if `arm not in KINDS`; the message lists `KINDS`
  - `mbfps.models.encoders._ARM_BACKBONE = {"cnn": None, "pixel_ae": "pixel_ae", "frozen_ssl": "dinov2", "random_vit": "random_vit"}`
  - `mbfps.models.encoders.encoder_input_kind(cfg) -> str` — `"obs"` iff `cfg.kind == "cnn"`, else `"features"`
  - `mbfps.models.encoders.encoder_backbone(cfg) -> str | None` — `"pixel_ae"` for `pixel_ae`
  - `mbfps.models.encoders.build_encoder(cfg) -> nn.Module` — `cnn -> CNNEncoder`; `pixel_ae`/`frozen_ssl`/`random_vit -> BottleneckEncoder`; else `KeyError`
  - `scripts/run_study.py: _SLOW_ARMS: tuple[str, ...] = ()`
  - `scripts/train_autoencoder.py`, `scripts/reconstruction_grid.py`: `--arm choices=KINDS`
  - `scripts/eval_reconstruction.py`: new `--arms nargs="+" choices=KINDS default=["cnn", "frozen_ssl", "random_vit"]`
  - `tests/eval/conftest.py::small_buffer` also writes `<episode>.features_pixel_ae.npy` of shape `(41, 64, 32)` float16

> Step-by-step order matters here. Step 3 changes `ARMS`, and from that moment
> until Step 14 lands the sweep, `tests/eval/test_aggregate.py` and its siblings
> are red — they build fixtures labelled `"cnn"` and the gate now expects
> `"pixel_ae"`. Each step's run is scoped to the file it just changed; the full
> suite is run once, in Step 15, and must be green there.

- [ ] **Step 1: Write the failing config tests**

Replace lines 1–27 of `tests/utils/test_config.py` (the imports through `test_unknown_arm_error_lists_valid_arms`) with:

```python
import dataclasses

import pytest

from mbfps.utils.config import (
    ARMS,
    KINDS,
    Config,
    EncoderConfig,
    TrainConfig,
    get_config,
)


def test_the_studys_three_arms_are_registered_and_cnn_is_not_one_of_them():
    """M3c retired the end-to-end pixel arm from the study (its embedding
    target was born collapsed, `embedding_loss` 0.0008 at step 0, and its
    dynamics prior never trained) and put the frozen M2 autoencoder in its
    place. Asserted as the whole tuple, in order: the arm order is the row
    order of every report table."""
    assert ARMS == ("pixel_ae", "frozen_ssl", "random_vit")
    assert "cnn" not in ARMS


def test_kinds_is_the_arms_plus_the_cnn_encoder_m2_still_builds():
    """Two tuples because two milestones read them: M2's autoencoder scripts
    still construct `CNNEncoder` under the name `cnn`, and M3's tools must not
    be able to select it. `cnn` first, so a reader sees at once which member
    is the odd one out."""
    assert KINDS == ("cnn",) + ARMS
    assert KINDS == ("cnn", "pixel_ae", "frozen_ssl", "random_vit")


@pytest.mark.parametrize("kind", KINDS)
def test_each_kind_builds_a_config(kind):
    """Over KINDS, not ARMS: `get_config("cnn")` must keep working for M2 and
    for the M3b artefacts that pin it, and `get_config("pixel_ae")` must work
    for the new study. Parametrised so the name of the kind that stops
    building is in the failure, not lost in a loop."""
    cfg = get_config(kind)
    assert isinstance(cfg, Config)
    assert cfg.arm == kind
    assert cfg.encoder.kind == kind


def test_get_config_still_builds_cnn_even_though_the_study_does_not_run_it():
    """The retired arm is buildable, not selectable. This is the property the
    shipped M3b checkpoints (`runs/m3_study/world_model_cnn_seed0.pt`) and
    M2's autoencoder tests depend on. Named explicitly rather than left to
    the KINDS parametrisation above: a KINDS that lost `cnn` would shrink
    that test, not fail it."""
    assert "cnn" not in ARMS
    assert get_config("cnn").encoder.kind == "cnn"


def test_unknown_arm_rejected():
    with pytest.raises(KeyError, match="unknown arm 'nope'"):
        get_config("nope")


def test_unknown_arm_error_lists_every_kind():
    """The message lists KINDS -- all four -- so an operator who typed the
    retired name sees it is still a kind, and one who typed a study arm
    wrong sees the study's three."""
    with pytest.raises(KeyError) as excinfo:
        get_config("nope")
    message = str(excinfo.value)
    for kind in KINDS:
        assert kind in message, f"{kind!r} missing from {message}"
    assert str(list(KINDS)) in message
```

Then, in the two tests that pick a baseline arm out of a dict keyed by `ARMS` (they would `KeyError` once `cnn` leaves the tuple), change line 33 and line 47:

```python
    first = trains["pixel_ae"]          # was trains["cnn"]
```
```python
    baseline = dict(encoders["pixel_ae"])   # was encoders["cnn"]
```

Leave `get_config("cnn")` at lines 72, 76, 83 and 89 exactly as they are: after this task they are four more places that prove the retired kind still builds.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/utils/test_config.py -q`
Expected: FAIL at collection — `ImportError: cannot import name 'KINDS' from 'mbfps.utils.config'`.

- [ ] **Step 3: Implement `ARMS`, `KINDS`, `get_config`**

Replace lines 16–18 of `src/mbfps/utils/config.py`:

```python
ARMS: tuple[str, ...] = ("pixel_ae", "frozen_ssl", "random_vit")
"""The study's arms. Every M3 tool -- the driver, report, aggregation, rollout
evaluation, diagnostics, pooling -- takes `choices=ARMS`.

`pixel_ae` is the frozen M2 pixel autoencoder's encoder, `frozen_ssl` the
DINOv2 treatment, `random_vit` the control that separates pretraining from
target stability. M3b's end-to-end `cnn` arm is not here: measured, its
embedding target was born collapsed (`embedding_loss` 0.0008 at step 0 against
0.32-0.41 for the feature arms), its dynamics KL never cleared the free-bits
floor, and its prior never received a gradient. The order here is the row
order of every report table."""

KINDS: tuple[str, ...] = ("cnn",) + ARMS
"""Every encoder `build_encoder` can construct. `get_config` validates against
this, not `ARMS`, because M2's autoencoder scripts and tests still build the
end-to-end `CNNEncoder` under the name `cnn` and must keep doing so -- and the
shipped M3b checkpoints (`runs/m3_study/world_model_cnn_seed*.pt`) load through
`get_config("cnn")`. Buildable is not selectable: no M3 tool offers `cnn`, so
no M3c artefact can be written under the old name."""
```

Replace lines 63–77 (`get_config`):

```python
def get_config(arm: str, **overrides) -> Config:
    """Build the configuration for `arm`, applying `overrides` to TrainConfig.

    Args:
        arm: one of `KINDS` -- a study arm, or `cnn` for M2's autoencoder and
            the shipped M3b artefacts.
        **overrides: field names of `TrainConfig`.

    Raises:
        KeyError: if `arm` is not a registered kind. The message lists `KINDS`,
            so a reader sees both that `cnn` is still a kind and which three
            the study runs.
        TypeError: if an override names a field `TrainConfig` does not have.
    """
    if arm not in KINDS:
        raise KeyError(f"unknown arm {arm!r}; available: {list(KINDS)}")
    train = replace(TrainConfig(), **overrides) if overrides else TrainConfig()
    return Config(arm=arm, encoder=EncoderConfig(kind=arm), train=train)
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/utils/test_config.py -q`
Expected: PASS — 5 more than Task 1 left in the file (16 → **21**).

- [ ] **Step 5: Write the failing encoder tests**

In `tests/models/test_encoders.py`, replace the imports at lines 6–15 with:

```python
import mbfps.models.encoders as encoders  # Task 1's `test_no_second_source_of_truth_for_the_patch_count` reads it
from mbfps.data.features import BACKBONE_GEOMETRY
from mbfps.envs.protocol import OBS_SHAPE
from mbfps.utils.config import get_config
from mbfps.models.encoders import (
    BottleneckEncoder,
    CNNEncoder,
    build_encoder,
    encoder_backbone,
    encoder_input_kind,
)
from mbfps.utils.config import ARMS, KINDS, EncoderConfig
```

Replace lines 91–110 (`test_build_encoder_returns_something_for_every_arm`, `test_every_arm_emits_the_same_embedding_width`, `test_input_kind_is_obs_for_cnn_and_features_for_ssl_arms`) with:

```python
def test_the_kinds_under_test_are_the_four_the_registry_knows():
    """L7 guard for every test below that parametrises over KINDS or ARMS: a
    tuple that silently lost a member would shrink those tests rather than
    fail them. The names are spelled out once, here."""
    assert KINDS == ("cnn", "pixel_ae", "frozen_ssl", "random_vit")
    assert ARMS == ("pixel_ae", "frozen_ssl", "random_vit")


@pytest.mark.parametrize("kind", KINDS)
def test_build_encoder_routes_every_kind_to_its_class(kind):
    """Over KINDS, not ARMS: `cnn` must stay constructible for M2 and for the
    shipped M3b checkpoints, and `pixel_ae` must route to the SAME bottleneck
    class the ViT arms use -- the study's arms are one pipeline."""
    built = build_encoder(cfg(kind))
    expected = CNNEncoder if kind == "cnn" else BottleneckEncoder
    assert type(built) is expected, (kind, type(built))


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_emits_the_same_embedding_width(kind):
    """The feature input is shaped from the backbone registry, not a literal
    (64, 384): `pixel_ae`'s rows are 32 wide, and a hardcoded width would
    make this test the one place the study's own geometry is not read."""
    c = cfg(kind)
    enc = build_encoder(c)
    if encoder_input_kind(c) == "obs":
        x = torch.randint(0, 256, (3, *OBS_SHAPE), dtype=torch.uint8)
    else:
        n_patches, patch_dim = BACKBONE_GEOMETRY[encoder_backbone(c)]
        x = torch.randn(3, n_patches, patch_dim)
    assert enc(x).shape == (3, 2048)


def test_input_kind_is_obs_for_cnn_and_features_for_every_study_arm():
    """Every study arm reads cached features. Written out by name AND over
    the tuple: the tuple form is what the study relies on (`run_job` picks
    the cache from this), the named form is what stops a shrunken ARMS from
    passing it vacuously."""
    assert encoder_input_kind(cfg("cnn")) == "obs"
    assert encoder_input_kind(cfg("pixel_ae")) == "features"
    assert encoder_input_kind(cfg("frozen_ssl")) == "features"
    assert encoder_input_kind(cfg("random_vit")) == "features"
    assert {encoder_input_kind(cfg(arm)) for arm in ARMS} == {"features"}
    assert "cnn" not in ARMS, "the one obs-kind encoder is not a study arm"
```

Replace lines 124–131 (the body of `test_each_arm_maps_to_its_own_backbone` and all of `test_feature_arms_map_to_distinct_backbones`) with:

```python
    assert encoder_backbone(cfg("cnn")) is None
    assert encoder_backbone(cfg("pixel_ae")) == "pixel_ae"
    assert encoder_backbone(cfg("frozen_ssl")) == "dinov2"
    assert encoder_backbone(cfg("random_vit")) == "random_vit"


def test_study_arms_map_to_three_distinct_backbones():
    """If any two collided, two arms would be the same experiment."""
    backbones = [encoder_backbone(cfg(arm)) for arm in ARMS]
    assert len(set(backbones)) == 3 == len(ARMS), backbones
    assert None not in backbones, "a study arm that reads pixels has no cache"
```

Replace lines 185–186 (the two assertions inside `test_recorded_parameter_counts`) with the three below and the new test that follows:

```python
    assert n_params(CNNEncoder(cfg("cnn"))) == 26_382_304
    assert n_params(BottleneckEncoder(cfg("frozen_ssl"))) == 12_320
    assert n_params(BottleneckEncoder(cfg("pixel_ae"))) == 1_056


def test_pixel_ae_builds_the_same_bottleneck_class_over_32_wide_rows():
    """Spec section 2.2's one stated asymmetry: same class, same per-row
    treatment, same state-dict keys as the ViT arms, but `Linear(32 -> 32)`
    (1,056 parameters) against `Linear(384 -> 32)` (12,320), because the M2
    autoencoder's `project` layer already reduced each row to 32. Pinned so
    the asymmetry stays the one the design recorded and not a second, silent
    one -- and driven through `build_encoder`, so a routing table that sent
    `pixel_ae` to the ViT geometry fails here rather than at step 0."""
    built = build_encoder(cfg("pixel_ae"))
    reference = build_encoder(cfg("frozen_ssl"))
    assert type(built) is BottleneckEncoder
    assert built.state_dict().keys() == reference.state_dict().keys()
    assert isinstance(built.bottleneck, torch.nn.Linear)
    assert (built.bottleneck.in_features, built.bottleneck.out_features) == (32, 32)
    assert (reference.bottleneck.in_features, reference.bottleneck.out_features) == (384, 32)
    assert isinstance(built.norm, torch.nn.LayerNorm)
    assert tuple(built.norm.normalized_shape) == (32,)
    assert n_params(built.norm) == 0
    assert built(torch.randn(3, 64, 32)).shape == (3, 2048)
```

- [ ] **Step 6: Run to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/models/test_encoders.py -q`
Expected: FAIL — `test_build_encoder_routes_every_kind_to_its_class[pixel_ae]`, `test_every_kind_emits_the_same_embedding_width[pixel_ae]`, `test_each_arm_maps_to_its_own_backbone`, `test_study_arms_map_to_three_distinct_backbones`, `test_recorded_parameter_counts` and `test_pixel_ae_builds_the_same_bottleneck_class_over_32_wide_rows` all raise `KeyError: "unknown encoder kind 'pixel_ae'"`; `test_input_kind_is_obs_for_cnn_and_features_for_every_study_arm` passes already (the `!= "cnn"` branch happens to be right) — its guard is the `encoder_input_kind` mutation in Step 16.

- [ ] **Step 7: Implement the registry entries**

Replace lines 1–11 of `src/mbfps/models/encoders.py` (the module docstring):

```python
"""Encoders -- the ONLY module that differs between the study's three arms.

Every study arm applies the same small learned bottleneck to rows cached from
a frozen backbone: `pixel_ae` from the M2 pixel autoencoder's encoder,
`frozen_ssl` from DINOv2, `random_vit` from an untrained ViT. The arms differ
in exactly one thing -- what the backbone was pretrained on -- and the row
geometry each reads comes from `BACKBONE_GEOMETRY`.

`CNNEncoder`, the end-to-end pixel encoder, is M3b's retired `cnn` arm and
M2's autoencoder encoder. It stays constructible under `kind="cnn"` (M2's
scripts and tests build it, and the shipped M3b checkpoints load through it)
but `cnn` is not in `ARMS`, so no M3 tool can select it.

All four emit a 2048-dimensional embedding from byte-identical 112x112 frames,
so resolution and embedding width are controlled and only the representation
differs. If any other module ever needs to know which arm is running, the
comparison has stopped being controlled.
"""
```

Two replacements, not one span — Task 1 moved the routing helpers ABOVE `BottleneckEncoder`, so "`encoder_input_kind` through the end of the file" would now swallow the class. (i) Replace the contiguous block `encoder_input_kind`, `_ARM_BACKBONE`, `encoder_backbone` (it sits between `CNNEncoder` and `class BottleneckEncoder`) with the first three definitions below. (ii) Replace `build_encoder` (the last definition in the file, after the class) with the last definition below. `BottleneckEncoder` stays exactly as Task 1 left it:

```python
def encoder_input_kind(cfg: EncoderConfig) -> str:
    """Whether this encoder consumes `"obs"` or cached `"features"`.

    `"obs"` iff `cfg.kind == "cnn"`. Every study arm -- `pixel_ae` included,
    its M2 encoder having been run over the frames at cache time -- reads
    features; the one pixel-consuming encoder is the retired kind.
    """
    return "obs" if cfg.kind == "cnn" else "features"


_ARM_BACKBONE: dict[str, str | None] = {
    "cnn": None,
    "pixel_ae": "pixel_ae",
    "frozen_ssl": "dinov2",
    "random_vit": "random_vit",
}
"""Which cached feature set each kind reads. None means the encoder reads pixels.

The arm name and the backbone name are deliberately not assumed equal:
`frozen_ssl` reads the `dinov2` cache. Deriving one from the other by string
identity would silently send the treatment arm to a cache that does not exist.
`pixel_ae` and `random_vit` happen to share their backbone's name; that is a
coincidence the tests name, not a rule this table relies on.
"""


def encoder_backbone(cfg: EncoderConfig) -> str | None:
    """Backbone whose cached features this kind consumes, or None for pixels.

    Raises:
        KeyError: if `cfg.kind` is not a registered kind.
    """
    if cfg.kind not in _ARM_BACKBONE:
        raise KeyError(f"unknown encoder kind {cfg.kind!r}")
    return _ARM_BACKBONE[cfg.kind]


def build_encoder(cfg: EncoderConfig) -> nn.Module:
    """Construct the encoder for `cfg.kind`.

    `cnn` -> `CNNEncoder`; every study arm -> the one `BottleneckEncoder`
    class, which reads its row geometry from `BACKBONE_GEOMETRY` for the
    kind's backbone. The routing is by explicit name, not `kind != "cnn"`,
    so an unregistered kind is a KeyError here rather than a KeyError from
    inside the registry lookup.

    Raises:
        KeyError: if `cfg.kind` is not a registered kind.
    """
    if cfg.kind == "cnn":
        return CNNEncoder(cfg)
    if cfg.kind in ("pixel_ae", "frozen_ssl", "random_vit"):
        return BottleneckEncoder(cfg)
    raise KeyError(f"unknown encoder kind {cfg.kind!r}")
```

- [ ] **Step 8: Run to verify it passes**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/models/test_encoders.py -q`
Expected: PASS — 4 more than Task 1 left in the file (36 → **40**). If `test_pixel_ae_builds_the_same_bottleneck_class_over_32_wide_rows` fails with `in_features == 384`, Task 2's `BottleneckEncoder` is not reading `BACKBONE_GEOMETRY` — that is the dependency, not a bug in this step.

- [ ] **Step 9: Write the failing driver and M2-script tests**

In `tests/eval/test_run_study.py`, replace lines 705–706 (inside `test_the_driver_uses_the_studys_own_arms_and_seeds`):

```python
    assert run_study._SLOW_ARMS == (), (
        "M3c retired the end-to-end pixel arm; every study arm now trains at "
        "feature-arm speed (~1.5 h per cell), so no arm is deferred. An "
        "arm listed here would be scheduled last for a cost it does not have")
    assert "cnn" not in ARMS
```

Replace lines 1038–1044 (`test_cheap_arms_run_first`):

```python
def test_no_arm_is_deferred_and_the_order_is_arm_then_seed(tmp_path):
    """With `_SLOW_ARMS` empty the whole schedule is `(arm, seed)`, arms in
    string order. Under M3b the 8.3-hour pixel arm ran last so an interruption
    still left the treatment/control contrast complete; every M3c arm costs
    the same ~1.5 h, so there is nothing to defer and the order is pinned
    here IN FULL rather than as "the slow one is last" -- the arm names are
    spelled out so a shrunken ARMS cannot pass this by iterating less."""
    jobs = run_study.pending_jobs(tmp_path, ARMS, SEEDS)
    assert sorted(ARMS) == ["frozen_ssl", "pixel_ae", "random_vit"]
    assert jobs == [
        StudyJob("frozen_ssl", 0), StudyJob("frozen_ssl", 1),
        StudyJob("frozen_ssl", 2),
        StudyJob("pixel_ae", 0), StudyJob("pixel_ae", 1),
        StudyJob("pixel_ae", 2),
        StudyJob("random_vit", 0), StudyJob("random_vit", 1),
        StudyJob("random_vit", 2),
    ]
    assert len(jobs) == 9
```

Replace lines 1055–1060 (the call and expected list in `test_the_order_is_fully_determined_and_seeds_ascend`; the arms are handed over in the wrong order too, so the sort is doing the work):

```python
    jobs = run_study.pending_jobs(tmp_path, ("random_vit", "pixel_ae"), (2, 0, 1))
    assert jobs == [
        StudyJob("pixel_ae", 0), StudyJob("pixel_ae", 1), StudyJob("pixel_ae", 2),
        StudyJob("random_vit", 0), StudyJob("random_vit", 1),
        StudyJob("random_vit", 2),
    ]
```

Replace lines 1064–1074 (`test_the_order_holds_after_a_partial_resume`'s docstring, setup and expected list):

```python
    """Resume and ordering interact: what is left must still be in
    `(arm, seed)` order, with the completed cells simply absent."""
    _write_complete(tmp_path, StudyJob("frozen_ssl", 0))
    _write_complete(tmp_path, StudyJob("pixel_ae", 0))
    jobs = run_study.pending_jobs(tmp_path, ARMS, SEEDS)
    assert jobs == [
        StudyJob("frozen_ssl", 1), StudyJob("frozen_ssl", 2),
        StudyJob("pixel_ae", 1), StudyJob("pixel_ae", 2),
        StudyJob("random_vit", 0), StudyJob("random_vit", 1),
        StudyJob("random_vit", 2),
    ]
```

Replace lines 1513–1514 (inside `test_a_failing_cell_does_not_abort_the_other_eight`; the victim is `random_vit/1` and the cell after it in `(arm, seed)` order is `random_vit/2`):

```python
    assert [c["job"] for c in calls][-1] == StudyJob("random_vit", 2), (
        "the cell after the failing one was never reached")
```

The test below goes after `test_main_refuses_an_arm_the_study_does_not_have` — **but write it in Step 14, after Step 13's sweep has run, not now.** Its body holds `"cnn"` as the retired name (`assert "cnn" in KINDS`, `--arms cnn`, `result_cnn_seed0.json`), and the sweep's `ANY` regex would rewrite every one of those to `"pixel_ae"` and turn it into a test that `pixel_ae` is refused. In Steps 10 and 12, count it as not yet present (4 driver tests, not 5); Step 14 adds it and runs it.

```python
def test_main_refuses_the_retired_pixel_arm_by_name(tmp_path, episode_dir,
                                                    monkeypatch):
    """`cnn` is a KIND `get_config` still builds, so the only thing that
    keeps the study from training a cell of the retired arm is `--arms`
    taking `choices=ARMS` rather than `choices=KINDS`. A driver that accepted
    it would write `result_cnn_seed0.json` into the M3c directory and the
    aggregation would read a fourth arm. `run_job` is stubbed so that if the
    refusal is missing this fails on the exit status, not by training."""
    from mbfps.utils.config import KINDS, get_config

    assert "cnn" in KINDS and get_config("cnn").arm == "cnn", (
        "the precondition: the name IS buildable, so only argparse refuses it")
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    with pytest.raises(SystemExit) as excinfo:
        run_study.main(["--data", str(episode_dir), "--out", str(tmp_path),
                        "--arms", "cnn", "--seeds", "0"])
    assert excinfo.value.code == ARGPARSE_USAGE_STATUS
    assert calls == [], "a cell of the retired arm was trained"
    assert tuple(run_study._parser().parse_args([]).arms) == ARMS
```

Create `tests/training/test_m2_scripts_take_kinds.py`:

```python
"""M2's three autoencoder scripts take `choices=KINDS`, not `choices=ARMS`.

M3c retired `cnn` from `ARMS`. These scripts are how M2 trained, rendered and
scored the end-to-end `CNNEncoder`, and `runs/m2_fixed/autoencoder_cnn.pt` --
which is now the `pixel_ae` backbone -- was produced by the first of them. If
they took `ARMS` they would refuse the one kind they exist for.

The parser is captured rather than run: each script's `main` calls
`parse_args()` inline and then trains, so `parse_args` is replaced with a
function that hands the parser back and stops. What is asserted is the
parser's real configuration, not the source text.
"""

import argparse
import importlib.util
from pathlib import Path

import pytest

from mbfps.utils.config import ARMS, KINDS

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_for_kinds_test", _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Captured(Exception):
    """Raised from the stand-in `parse_args` so `main` never reaches training."""


def _capture_parser(monkeypatch, script) -> argparse.ArgumentParser:
    holder = {}

    def parse_args(self, args=None, namespace=None):
        holder["parser"] = self
        raise _Captured

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_args)
    with pytest.raises(_Captured):
        script.main()
    return holder["parser"]


def _choices(parser: argparse.ArgumentParser, dest: str):
    action = next(a for a in parser._actions if a.dest == dest)
    return tuple(action.choices)


def test_kinds_is_arms_plus_cnn():
    """L7 guard for the parametrisations below."""
    assert KINDS == ("cnn", "pixel_ae", "frozen_ssl", "random_vit")
    assert "cnn" in KINDS and "cnn" not in ARMS


@pytest.mark.parametrize("name", ["train_autoencoder", "reconstruction_grid"])
def test_single_arm_m2_scripts_offer_every_kind(monkeypatch, name):
    parser = _capture_parser(monkeypatch, _load(name))
    assert _choices(parser, "arm") == KINDS, (
        f"scripts/{name}.py must take choices=KINDS: `cnn` is the encoder it "
        "exists to build, and `ARMS` no longer contains it")


def test_eval_reconstruction_offers_every_kind_and_defaults_to_the_m2_arms(
    monkeypatch,
):
    """`--arms` is new: the script used to loop over `ARMS`, which after M3c
    would silently drop `cnn` from M2's paired comparison. The default is the
    three kinds M2 actually trained -- `pixel_ae` is constructible but has no
    `autoencoder_pixel_ae.pt`, so it is opt-in rather than a guaranteed
    FileNotFoundError on the default command line."""
    parser = _capture_parser(monkeypatch, _load("eval_reconstruction"))
    assert _choices(parser, "arms") == KINDS
    action = next(a for a in parser._actions if a.dest == "arms")
    assert list(action.default) == ["cnn", "frozen_ssl", "random_vit"]
    assert action.nargs == "+"
```

- [ ] **Step 10: Run to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_run_study.py tests/training/test_m2_scripts_take_kinds.py -q -k "driver_uses or no_arm_is_deferred or fully_determined or partial_resume or retired_pixel_arm or offer_every_kind or offers_every_kind"`
Expected: FAIL —
- `test_the_driver_uses_the_studys_own_arms_and_seeds`: `assert ('cnn',) == ()`
- `test_no_arm_is_deferred_and_the_order_is_arm_then_seed`, `test_the_order_is_fully_determined_and_seeds_ascend`, `test_the_order_holds_after_a_partial_resume`: `_cost_key` still keys on `_SLOW_ARMS == ("cnn",)`; with `cnn` no longer in `ARMS` nothing is slow and these three pass already — their guard is the `_SLOW_ARMS = ("pixel_ae",)` mutation in Step 16
- `test_main_refuses_the_retired_pixel_arm_by_name`: passes already, because `--arms` already takes `choices=ARMS` and Step 3 removed `cnn`; its guard is the `choices=KINDS` mutation in Step 16
- `test_single_arm_m2_scripts_offer_every_kind[train_autoencoder]` and `[reconstruction_grid]`: `assert ('pixel_ae', 'frozen_ssl', 'random_vit') == ('cnn', 'pixel_ae', 'frozen_ssl', 'random_vit')` — the scripts still take `choices=ARMS`, which is exactly the M2 regression this task exists to prevent
- `test_eval_reconstruction_offers_every_kind_and_defaults_to_the_m2_arms`: `StopIteration` — there is no `--arms` action yet

- [ ] **Step 11: Implement `_SLOW_ARMS = ()` and `choices=KINDS`**

In `scripts/run_study.py`, replace lines 1–8 (the docstring's first paragraph):

```python
"""Run the 3 arms x 3 seeds study, resumably.

Every arm reads cached frozen-backbone features and trains at the same speed:
measured 3.98 steps/s at seq_len=64 on the development Mac, ~1.4 h per
20k-step cell, ~13.5 h for the nine. Under M3b the end-to-end pixel arm ran
at 0.67 steps/s (8.3 h per cell) and was scheduled last so an interruption
still left the treatment/control contrast complete; M3c retired that arm, so
`_SLOW_ARMS` is empty and cells run in plain `(arm, seed)` order.
```

Replace lines 81–87 (`_SLOW_ARMS` and its docstring):

```python
_SLOW_ARMS: tuple[str, ...] = ()
"""Arms scheduled after every other arm. EMPTY since M3c: the 0.67 steps/s
end-to-end pixel arm was retired from the study and `pixel_ae` reads a cache
like the other two, so nothing costs more than anything else.

Membership of this tuple is the whole scheduling policy: everything not in it
runs first, and with it empty `_cost_key` reduces to `(arm, seed)`. Kept as a
tuple rather than deleted so an arm that does cost more can be put on the
correct side of the split by editing one line -- and so the driver's test can
assert that today nothing is deferred.
"""
```

Line 333, the comment inside `pending_jobs`:

```python
                continue  # `--arms pixel_ae pixel_ae` must not buy the same 1.4 h twice
```

Lines 411–417, inside `quarantine_record`'s docstring, replace the two sentences that price the pixel arm:

```python
    RENAMED RATHER THAN DELETED, AND THE RUN CONTINUES. Deleting is wrong: the
    file is the output of a real ~1.4-hour training run and may be the only
    copy of whichever cell it actually describes. Refusing to continue is also
    wrong: it contradicts the failure policy at the top of this file -- one bad
    cell must not throw away the eight that would still run. So the file is
    kept, under a name the aggregation cannot read as a record, and the cell
    goes back to pending.
```

`scripts/train_autoencoder.py`, lines 9 and 14:

```python
from mbfps.utils.config import KINDS, get_config
```
```python
    # KINDS, not ARMS: M3c retired `cnn` from the study, but this script is
    # how the end-to-end CNNEncoder gets trained, and its checkpoint is the
    # `pixel_ae` backbone.
    parser.add_argument("--arm", choices=KINDS, required=True)
```

`scripts/reconstruction_grid.py`, lines 18 and 24:

```python
from mbfps.utils.config import KINDS, get_config  # noqa: E402
```
```python
    parser.add_argument("--arm", choices=KINDS, required=True)  # KINDS: M2 renders cnn
```

`scripts/eval_reconstruction.py`, line 30:

```python
from mbfps.utils.config import KINDS, get_config
```

After line 66 (`parser.add_argument("--run", ...)`), add:

```python
    # KINDS, not ARMS: M3c retired `cnn` from the study, but `cnn` is the
    # encoder this comparison was built around and its checkpoint is the one
    # that became the `pixel_ae` backbone. The default is what M2 trained;
    # `pixel_ae` is constructible (an autoencoder over its own cached rows)
    # but no `autoencoder_pixel_ae.pt` exists, so it is opt-in.
    parser.add_argument("--arms", nargs="+", choices=KINDS,
                        default=["cnn", "frozen_ssl", "random_vit"])
```

Line 78, the loop in `main`:

```python
    for arm in args.arms:
```

- [ ] **Step 12: Run to verify it passes**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/training/test_m2_scripts_take_kinds.py -q`
Expected: PASS (4 tests)

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_run_study.py -q -k "driver_uses or no_arm_is_deferred or fully_determined or partial_resume or retired_pixel_arm"`
Expected: PASS (5 tests). The rest of `test_run_study.py` is still red — it labels fixtures `"cnn"` — until Step 13.

- [ ] **Step 13: The mechanical sweep — `"cnn"` as an arm label becomes `"pixel_ae"`**

Five files use `cnn` purely as a study-arm label: in fixtures, in expected record filenames (`result_cnn_seed0.json`, `diagnostic_cnn_seed0.json`, `world_model_cnn_seed0.pt`), in expected report rows, and in `test_pooling.py`'s own local `ARMS` tuple at line 49 (which becomes `("pixel_ae", "frozen_ssl", "random_vit")`, matching the config's). A plain `sed s/cnn/pixel_ae/` is wrong twice over: `report_study` renders `{arm:<12}` and `{label:<16}` columns, so the five extra characters of `pixel_ae` must come out of the padding that follows or 32 fixed-width row assertions break; and `\b` treats `_` as a word character, so `result_cnn_seed0` would be skipped. Run this from the repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'EOF'
import re
from pathlib import Path

# A table cell: `cnn` or `cnn/sN` followed by at least five spaces of column
# padding. `pixel_ae` is five characters longer, so five spaces come out.
PAD = re.compile(r"(?<![A-Za-z0-9_])cnn(/s\d)?( {5,})")
# Everything else: `cnn` not embedded in a longer alphanumeric run. `_` is
# NOT a boundary here, so `result_cnn_seed0.json` is renamed; `cnn2` (an
# invalid-name example in test_run_study.py) is not.
ANY = re.compile(r"(?<![A-Za-z0-9])cnn(?![A-Za-z0-9])")

for name in [
    "tests/eval/test_aggregate.py",
    "tests/eval/test_diagnose_dynamics_script.py",
    "tests/eval/test_pooling.py",
    "tests/eval/test_run_study.py",
    "tests/eval/test_pool_dynamics_script.py",
]:
    path = Path(name)
    text = path.read_text()
    before = len(ANY.findall(text))
    text, n_pad = PAD.subn(
        lambda m: "pixel_ae" + (m.group(1) or "") + m.group(2)[5:], text)
    text, n_any = ANY.subn("pixel_ae", text)
    path.write_text(text)
    print(f"{name}: {before} references; {n_pad} padded table cells, "
          f"{n_any} plain; remaining: {len(ANY.findall(text))}")
EOF
```

Expected output, exactly:

```
tests/eval/test_aggregate.py: 270 references; 32 padded table cells, 238 plain; remaining: 0
tests/eval/test_diagnose_dynamics_script.py: 78 references; 0 padded table cells, 78 plain; remaining: 0
tests/eval/test_pooling.py: 31 references; 0 padded table cells, 31 plain; remaining: 0
tests/eval/test_run_study.py: N references; 0 padded table cells, N plain; remaining: 0   # N = 38 minus the six Step 9 removed; record the measured N
tests/eval/test_pool_dynamics_script.py: 10 references; 0 padded table cells, 10 plain; remaining: 0
```

(The task brief's counts — 164, 68, 24, 21, 3 — were of the quoted form `"cnn"` only; the totals above also cover filenames, rendered rows and prose.) Then confirm what is left:

Run: `grep -n "cnn" tests/eval/test_aggregate.py tests/eval/test_diagnose_dynamics_script.py tests/eval/test_pooling.py tests/eval/test_run_study.py tests/eval/test_pool_dynamics_script.py`
Expected: exactly two lines, both in `test_run_study.py`, both the invalid name `cnn2` (their line numbers moved with Step 9's edits).

- [ ] **Step 14: The hand edits the sweep cannot make**

**First**, now that the sweep has run: append `test_main_refuses_the_retired_pixel_arm_by_name` from Step 9, verbatim, after `test_main_refuses_an_arm_the_study_does_not_have` in `tests/eval/test_run_study.py`, and run it — `-k retired_pixel_arm`, expected PASS (1 test; its guard is Step 16's `choices=KINDS` mutation). Then:

Four things do not survive a rename: `evaluate_gate` sorts cell listings alphabetically and `pixel_ae` sorts after `frozen_ssl` where `cnn` sorted before it; `test_study.py` reasons about which arm's name coincides with its backbone's, and `pixel_ae` reads a cache where `cnn` read pixels; the shared fixture has no `pixel_ae` cache; and `test_world_model.py`'s arm sweep is a literal list that never grows.

**`tests/eval/test_aggregate.py`** — two `sorted()` orderings. Lines 2511–2512:

```python
        "  CURVES INCOMPLETE (3 of 9): frozen_ssl/s1 (floor_angle), "
        "pixel_ae/s0 (rssm_position), arm='pixel_ae' seed='7' (floor_position)"), (
```

Line 2908:

```python
        "  MISSING CELLS (3 of 9): frozen_ssl/s2, pixel_ae/s2, random_vit/s2")
```

**`tests/eval/conftest.py`** — lines 43–48, so every `run_job` on `pixel_ae` finds a cache of its own geometry:

```python
    rng = np.random.default_rng(0)
    for path in buf.episode_paths():
        # One cache per study backbone, each at ITS OWN row width: the ViT
        # arms read (64, 384), pixel_ae reads (64, 32). A pixel_ae cache
        # written 384 wide would be refused by the loader's geometry check.
        for suffix, width in ((".features.npy", 384),
                              (".features_random_vit.npy", 384),
                              (".features_pixel_ae.npy", 32)):
            np.save(path.with_suffix(suffix),
                    rng.random((41, 64, width)).astype(np.float16))
    return buf
```

**`tests/eval/test_study.py`** — lines 45–53:

```python
OTHER_ARM = "pixel_ae"      # a second arm run below, so `"arm": job.arm` is
                            # not satisfied by hardcoding the first one
THIRD_ARM = "frozen_ssl"    # THE ONLY ARM WHOSE NAME IS NOT ITS BACKBONE'S.
                            # `random_vit` reads the `random_vit` cache and
                            # `pixel_ae` reads the `pixel_ae` cache, so on
                            # those two arms alone `backbone =
                            # encoder_backbone(cfg.encoder) -> backbone =
                            # job.arm` is a numerical no-op. This arm reads
                            # `dinov2`, and running it here is what makes
                            # that mutation visible at all.
```

Lines 161–162:

```python
    # `random_vit` and `pixel_ae`'s is called `pixel_ae` -- so `backbone =
    # job.arm` was a numerical no-op and survived the whole suite.
```

Lines 186–188:

```python
    assert backbones[OTHER_ARM] == OTHER_ARM, (
        f"{OTHER_ARM!r}'s arm name and backbone name are the same string too, "
        "so it cannot discriminate either")
```

Lines 196–197:

```python
    a = job_record_path(tmp_path, StudyJob("pixel_ae", 0))
    b = job_record_path(tmp_path, StudyJob("pixel_ae", 1))
```

Line 717: `` (`pixel_ae`/99 against this `` — the docstring's example checkpoint label.

Line 1606: `` of the MSE on this fixture and were bit-identical for M3b's pixel arm. So `` — the measurement was made on the retired arm and is quoted as history.

Lines 1717–1718:

```python
        `random_vit`'s arm name IS its backbone name and so is `pixel_ae`'s,
        and those were the only two arms this file ever ran. `frozen_ssl`, the
```

Lines 1734–1760 (from `assert expected == {...}` through the end of the "part two" check). With every arm a feature arm, the `if expected is not None` gate is gone and the wrong-cache check for `pixel_ae` is a *shape* difference, which is stronger than a value difference:

```python
    assert expected == {JOB_ARM: JOB_ARM, OTHER_ARM: OTHER_ARM,
                        THIRD_ARM: THIRD_ARM_BACKBONE}[arm], (
        f"{arm!r} no longer reads the cache this test was written against")

    # Check-it-can-fail, part one: a hardcoded "dinov2" is the WRONG answer
    # for two of the three arms, and `job.arm` is the wrong answer for the
    # third -- so between them the parametrisation moves.
    if arm != THIRD_ARM:
        assert expected != THIRD_ARM_BACKBONE, (
            "a hardcoded dinov2 backbone is indistinguishable from the truth "
            f"on {arm!r}")
    else:
        assert expected != arm, (
            f"the arm name and the backbone name coincide on {arm!r}, so "
            "`backbone = job.arm` is invisible here")

    # Check-it-can-fail, part two: every arm is a feature arm now, and
    # reading the OTHER cache must be a real change, not a relabelling. For
    # the ViT arms that is different numbers of the same shape; for
    # `pixel_ae` it is a different SHAPE -- (64, 32) rows against (64, 384)
    # -- which the loader's geometry check refuses outright.
    other_backbone = JOB_ARM if expected == THIRD_ARM_BACKBONE \
        else THIRD_ARM_BACKBONE
    episode = small_buffer.episode_paths()[0]
    mine = np.load(episode.with_suffix(feature_suffix(expected)))
    theirs = np.load(episode.with_suffix(feature_suffix(other_backbone)))
    assert mine.shape != theirs.shape or not np.array_equal(mine, theirs), (
        "the two feature caches hold identical data in this fixture, so "
        "reading the wrong one is a numerical no-op and nothing below "
        "guards anything")
```

**`tests/training/test_world_model.py`** — line 10 becomes `from mbfps.utils.config import ARMS, get_config`. `tiny()`'s default at line 61 stays `"cnn"`: the RSSM and loss tests are indifferent to which encoder feeds them, a pixel encoder needs no feature file, and `KINDS` permits it. Replace lines 233–241 (`test_all_three_arms_train`):

```python
def test_the_arms_this_file_trains_are_the_studys_three():
    """L7 guard for the parametrisation below: iterating `ARMS` would let a
    tuple that lost a member shrink the test instead of failing it, so the
    three names are written out here, once. `cnn` is not among them -- it is
    `tiny()`'s default above because the RSSM and loss tests are indifferent
    to which encoder feeds them and a pixel encoder needs no feature file,
    but it is not an arm the study trains."""
    assert ARMS == ("pixel_ae", "frozen_ssl", "random_vit")
    assert "cnn" not in ARMS


#: The cache each study arm reads, by suffix and row width. `pixel_ae` rows
#: are 32 wide (the M2 autoencoder's 2048-d projection cut into 64 rows);
#: the ViT arms' are 384 (DINOv2-small's hidden size). Written here rather
#: than derived from `BACKBONE_GEOMETRY`, so a registry that drifted would
#: fail this test rather than be copied into it.
_ARM_CACHE = {
    "pixel_ae": (".features_pixel_ae.npy", 32),
    "frozen_ssl": (".features.npy", 384),
    "random_vit": (".features_random_vit.npy", 384),
}


@pytest.mark.parametrize("arm", ARMS)
def test_all_three_arms_train(buffer, arm, tmp_path):
    """Every study arm trains end to end through `train_world_model`, each on
    a cache of ITS OWN geometry. Guarded against L7 by the test above."""
    assert set(_ARM_CACHE) == set(ARMS)
    suffix, width = _ARM_CACHE[arm]
    for path in buffer.episode_paths():
        np.save(path.with_suffix(suffix), np.zeros((41, 64, width), dtype=np.float16))
    history = train_world_model(tiny(arm), buffer, out_dir=None)
    assert history["steps"] == 3
    assert history["arm"] == arm
    assert all(np.isfinite(history["loss"]))
```

**`tests/eval/test_diagnostics.py`** — line 2815 keeps `"cnn"`; add the reason above it so the next sweep does not "fix" it:

```python
# The RETIRED end-to-end pixel arm, on purpose. These self-checks pin an M3b
# artefact (`runs/m3_study/world_model_cnn_seed0.pt`, its record, and
# `fixtures/shuffled_cnn_seed0_pre_ladder.json`) bitwise, so they must keep
# naming the arm that produced it. `get_config("cnn")` below still builds it
# because `cnn` is in KINDS; it is not in ARMS, so nothing in the M3c study
# can write a `cnn` artefact this could be confused with.
SHIPPED_ARM, SHIPPED_SEED = "cnn", 0
```

- [ ] **Step 15: Run the full suite**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests -q`
Expected: PASS — 15 more than Task 3 left the suite at: +5 `test_config.py` (16 → 21), +4 `test_encoders.py` (36 → 40), +1 `test_world_model.py` (32 → 33), +1 `test_run_study.py` (141 → 142), +4 `test_m2_scripts_take_kinds.py`; `test_study.py` (62), `test_aggregate.py` (170) and the other swept files keep their counts. The shipped-checkpoint tests in `test_diagnostics.py` run on this box (they need `runs/m3_study`, `data/my_way_home` and MPS) and still pass: they build their config with `get_config("cnn")`, which `KINDS` permits, and load `world_model_cnn_seed0.pt` through `CNNEncoder`, which still exists.

If `tests/data/test_features.py::test_backbones_registered` fails, that is Task 1's tuple, not this task.

- [ ] **Step 16: Mutation-test**

Set up the harness with all three self-checks. Nothing is committed yet, so restores are `cp` from a backup, not `git checkout`:

```bash
export PYTHONDONTWRITEBYTECODE=1
find src scripts tests -name __pycache__ -type d -exec rm -rf {} +      # self-check 3: no stale .pyc can survive a same-length mutation
.venv/bin/python -c "import mbfps; print(mbfps.__file__)"                # self-check 1: must print <repo>/src/mbfps/__init__.py -- the editable install IS this tree
cp src/mbfps/utils/config.py /tmp/config.bak
sed -i '' 's/^ARMS: tuple\[str, ...\] = ("pixel_ae", "frozen_ssl", "random_vit")/ARMS: tuple[str, ...] = ()/' src/mbfps/utils/config.py
.venv/bin/python -m pytest tests/utils/test_config.py -q                 # self-check 2: a known-fatal mutation FIRST -- must FAIL; a green run here means the harness is not testing this tree
cp /tmp/config.bak src/mbfps/utils/config.py
```

Then each row, applied with `sed -i ''` (or by hand), run, and restored from its backup:

| mutation | must be caught by |
|---|---|
| `ARMS` reordered to `("frozen_ssl", "pixel_ae", "random_vit")` | `test_config::test_the_studys_three_arms_are_registered_and_cnn_is_not_one_of_them`, `test_kinds_is_the_arms_plus_the_cnn_encoder_m2_still_builds` |
| `KINDS = ARMS` (drop `cnn`) | `test_config::test_kinds_is_the_arms_plus_the_cnn_encoder_m2_still_builds`, `test_get_config_still_builds_cnn_even_though_the_study_does_not_run_it`, and the four `get_config("cnn")` tests at lines 72–89 |
| `get_config` validates `arm not in ARMS` | `test_config::test_each_kind_builds_a_config[cnn]`, `test_get_config_still_builds_cnn_...`; every `tests/training/test_autoencoder.py` test that calls `tiny("cnn")` |
| `_ARM_BACKBONE["pixel_ae"] = "dinov2"` | `test_encoders::test_each_arm_maps_to_its_own_backbone`, `test_study_arms_map_to_three_distinct_backbones`, `test_recorded_parameter_counts` (12,320 ≠ 1,056), `test_pixel_ae_builds_the_same_bottleneck_class_over_32_wide_rows` (`in_features == 384`); `test_study::test_every_evaluation_reads_the_cache_the_arms_encoder_actually_uses[pixel_ae]` |
| `build_encoder`: `if cfg.kind in ("cnn", "pixel_ae"): return CNNEncoder(cfg)` | `test_encoders::test_build_encoder_routes_every_kind_to_its_class[pixel_ae]`, `test_every_kind_emits_the_same_embedding_width[pixel_ae]`, `test_pixel_ae_builds_...` |
| `encoder_input_kind`: `"obs" if cfg.kind in ("cnn", "pixel_ae")` | `test_encoders::test_input_kind_is_obs_for_cnn_and_features_for_every_study_arm`, `test_every_kind_emits_the_same_embedding_width[pixel_ae]` |
| `_SLOW_ARMS = ("pixel_ae",)` | `test_run_study::test_the_driver_uses_the_studys_own_arms_and_seeds`, `test_no_arm_is_deferred_and_the_order_is_arm_then_seed`, `test_the_order_is_fully_determined_and_seeds_ascend`, `test_the_order_holds_after_a_partial_resume`, `test_a_failing_cell_does_not_abort_the_other_eight[*]` |
| `run_study.py`: `--arms ... choices=KINDS` (import `KINDS` too) | `test_run_study::test_main_refuses_the_retired_pixel_arm_by_name` — and ONLY that test; measured, the other 141 stay green, which is why it exists |
| `train_autoencoder.py`: `choices=ARMS` | `test_m2_scripts_take_kinds::test_single_arm_m2_scripts_offer_every_kind[train_autoencoder]` |
| `eval_reconstruction.py`: `--arms` default `list(KINDS)` | `test_m2_scripts_take_kinds::test_eval_reconstruction_offers_every_kind_and_defaults_to_the_m2_arms` |
| `conftest.py`: `pixel_ae` cache written `(41, 64, 384)` | `test_run_study::test_a_real_record_from_run_job_is_recognised_as_complete[pixel_ae]`, `test_study::test_every_evaluation_reads_the_cache_...[pixel_ae]` and `test_the_record_is_labelled_with_the_arm_the_job_asked_for` — `ValueError: expected (64, 32), got (64, 384)` from the geometry check |
| `test_world_model.py`: `_ARM_CACHE["pixel_ae"]` width `384` (fixture self-check) | `test_all_three_arms_train[pixel_ae]` |

Any mutation that survives is a missing test. Add it before committing. After the last row: `find src scripts tests -name __pycache__ -type d -exec rm -rf {} +` and `git diff --stat` must show only the files in the Files block.

- [ ] **Step 17: Commit**

```bash
git add src/mbfps/utils/config.py src/mbfps/models/encoders.py \
        scripts/run_study.py scripts/train_autoencoder.py scripts/reconstruction_grid.py scripts/eval_reconstruction.py \
        tests/utils/test_config.py tests/models/test_encoders.py tests/training/test_world_model.py \
        tests/training/test_m2_scripts_take_kinds.py tests/eval/conftest.py tests/eval/test_study.py \
        tests/eval/test_run_study.py tests/eval/test_aggregate.py tests/eval/test_diagnose_dynamics_script.py \
        tests/eval/test_pooling.py tests/eval/test_pool_dynamics_script.py tests/eval/test_diagnostics.py
git commit -m "feat: cnn is retired from the study but stays buildable -- ARMS gains pixel_ae, KINDS keeps cnn for M2 and the shipped M3b artefacts

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

`git status` must show nothing under `runs/` or `data/` staged.

---

### Task 5: `cache_features.py --backbone pixel_ae`, and the cache itself

The script is the only thing that writes a feature cache, and a cache is then read by every training step of two-thirds of the study with nothing checking how it was made. `--backbone` already takes `choices=BACKBONES`, so `pixel_ae` is *accepted* the moment Task 1 registers it — but three things in the script still assume every backbone is a ViT: the disk guard asks for `frames × 64 × 384 × 2` bytes (2.93 GB for a cache that is 0.24 GB), `FeatureExtractor` is built without the checkpoint the `pixel_ae` backbone is loaded from, and nothing exercises the write path for `(64, 32)` rows. Then the cache is actually produced, and checked for the one property spec §1 says the pixel arm lacked: a target that is not born collapsed.

**Files:**
- Modify: `scripts/cache_features.py` — the file as Task 1 left it (registry import, `cache_bytes(frames, backbone)`, disk estimate already ported). This task factors the inline parser into `_parser()` / `main(argv)` and adds `--checkpoint`. `cache_bytes`'s body is kept identical (its docstring is reworded).
- Test: `tests/data/test_cache_features_script.py` (modify — Task 1 created it with three `cache_bytes` tests and the `importlib` loader; keep all three verbatim and append)
- Modify: this plan (record the measured cache size and per-dim std under "## Task 5 results")

**Interfaces:**
- Consumes: `BACKBONES`, `BACKBONE_GEOMETRY`, `PIXEL_AE_CHECKPOINT`, `FeatureExtractor(backbone: str = "dinov2", device: str = "mps", seed: int = 0, checkpoint: Path | None = None)`, `cache_episode_features(ep_path, extractor, batch_size=32) -> Path`, `require_free_bytes(path, needed)` from `mbfps.data.features`; `feature_suffix(backbone: str) -> str` from `mbfps.data.loader`; `ReplayBuffer` from `mbfps.data.buffer`. Tests only: `CNNEncoder` from `mbfps.models.encoders` and `EncoderConfig` from `mbfps.utils.config` to build a fake M2 checkpoint; `Episode`, `OBS_SHAPE`.
- Consumes (from Task 1): `scripts/cache_features.py::cache_bytes(frames: int, backbone: str) -> int` and the three tests that pin it (`test_cache_bytes_is_frames_times_geometry_times_float16`, `test_cache_bytes_reads_the_registry_not_a_constant`, `test_cache_bytes_rejects_an_unknown_backbone`) together with the file's `importlib` loader header; from Task 2: `PIXEL_AE_CHECKPOINT`, `FeatureExtractor(..., checkpoint=)`, `BACKBONES` containing `pixel_ae`.
- Produces, in `scripts/cache_features.py`: `_parser() -> argparse.ArgumentParser` with the new `--checkpoint PATH` flag (`type=Path`, `default=PIXEL_AE_CHECKPOINT`, forwarded to `FeatureExtractor` as `checkpoint=` for every backbone); `main(argv: list[str] | None = None) -> None`. On disk: 122 files `data/my_way_home/ep_*.features_pixel_ae.npy`, each `(T+1, 64, 32)` float16.

> The parser and the byte estimate become module-level functions for the same
> reason `scripts/run_study.py` has `_parser()` and `main(argv)`: a script whose
> only entry point reads `sys.argv` cannot be driven from a test. The tests load
> the script by path, as `tests/eval/test_eval_rollout_script.py` does, because
> `scripts/` is not a package.

> **The tests must not depend on `runs/m2_fixed/autoencoder_cnn.pt`.** Nothing
> under `runs/` is committed, so a test that reads it passes on this machine and
> errors on every other. Each test that needs a checkpoint builds an M2-shaped
> one in `tmp_path` from a real `CNNEncoder` — `train_autoencoder` saves
> `{"arm": cfg.arm, "state_dict": model.state_dict()}` with the encoder under
> `encoder.*`, and that is exactly what is written. It is 105 MB and takes 0.2 s.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_cache_features_script.py
# EXTENDS the file Task 1 created. The docstring below REPLACES Task 1's. Task 1's
# import block, its `_SPEC` / `script` / `exec_module` trio and its three cache_bytes
# tests stay exactly as they are. The header repeated below is context, not text to
# paste again: MERGE the import lines (add only the ones Task 1's file lacks -- numpy,
# torch, ReplayBuffer, ...), keep ONE `_SPEC`/`script` trio, and append everything
# from the first new fixture onward after Task 1's third test.
"""`scripts/cache_features.py` with a third backbone whose rows are not (64, 384).

The script is the only thing that writes a feature cache, and a cache is read
by every training step of two-thirds of the study without anything checking
how it was made. So the tests are arranged around the ways a `pixel_ae` cache
can be wrong while looking right: a file written under the ViT suffix (so the
`frozen_ssl` arm silently trains on autoencoder features), a file whose rows
are (64, 384) or a flat 2048 (refused at load, but only after five minutes of
encoding), a disk guard that demands the ViT cache's 2.93 GB for a 0.24 GB
write, and -- the defect this project keeps meeting -- an encoder that is not
the one the checkpoint holds. Every fixture builds its own M2-shaped
checkpoint from a real `CNNEncoder`, so nothing here depends on
`runs/m2_fixed/autoencoder_cnn.pt` existing.

The script is loaded by path, as `tests/eval/test_eval_rollout_script.py`
does, because `scripts/` is not a package.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.data.features import BACKBONE_GEOMETRY, BACKBONES, PIXEL_AE_CHECKPOINT
from mbfps.envs.protocol import OBS_SHAPE
from mbfps.models.encoders import CNNEncoder
from mbfps.utils.config import EncoderConfig

_SPEC = importlib.util.spec_from_file_location(
    "cache_features_script",
    Path(__file__).resolve().parents[2] / "scripts" / "cache_features.py",
)
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")
T = 5
"""Transitions per fixture episode: T + 1 = 6 frames, one batch of 32."""
N_FRAMES = T + 1

# Hand-written, NOT derived from BACKBONE_GEOMETRY (L7: parametrising the
# expectation over the registry under test would let a wrong registry entry
# agree with itself). The literal rows are what the disk guard must be told
# for one six-frame episode: 64 x 384 float16 rows for the ViTs, 64 x 32 for
# the autoencoder. `test_expectations_cover_every_backbone` forces a new row
# the day a fourth backbone is registered.
EXPECTED_BYTES = {
    "dinov2": N_FRAMES * 64 * 384 * 2,
    "random_vit": N_FRAMES * 64 * 384 * 2,
    "pixel_ae": N_FRAMES * 64 * 32 * 2,
}
EXPECTED_SUFFIX = {
    "dinov2": ".features.npy",
    "random_vit": ".features_random_vit.npy",
    "pixel_ae": ".features_pixel_ae.npy",
}


def _episode(seed: int) -> Episode:
    """Six random frames: random so that no two encode to the same row."""
    rng = np.random.default_rng(seed)
    return Episode(
        obs=rng.integers(0, 256, (N_FRAMES, *OBS_SHAPE), dtype=np.uint8),
        actions=np.zeros(T, dtype=np.int32),
        rewards=np.zeros(T, dtype=np.float32),
        terminated=np.zeros(T, dtype=bool),
        truncated=np.zeros(T, dtype=bool),
        privileged=np.zeros((N_FRAMES, len(KEYS)), dtype=np.float32),
        privileged_keys=KEYS,
        policy_name="random",
        seed=seed,
        scenario="my_way_home",
    )


@pytest.fixture
def data_dir(tmp_path) -> Path:
    root = tmp_path / "data"
    ReplayBuffer(root, capacity_transitions=10_000).add(_episode(seed=0))
    return root


def _fake_m2_checkpoint(path: Path, seed: int) -> Path:
    """An M2-shaped checkpoint from a real CNNEncoder, without runs/m2_fixed.

    `mbfps.training.autoencoder.train_autoencoder` saves
    `{"arm": cfg.arm, "state_dict": model.state_dict()}` where the model holds
    its encoder under `encoder.` and its decoder under `decoder.`. Only the
    encoder subset matters to the backbone loader, and a random-init encoder is
    enough to prove that THIS file's weights are the ones that reached the
    cache -- two seeds give two different encoders.
    """
    torch.manual_seed(seed)
    encoder = CNNEncoder(EncoderConfig(kind="cnn"))
    torch.save(
        {
            "arm": "cnn",
            "state_dict": {f"encoder.{k}": v for k, v in encoder.state_dict().items()},
        },
        path,
    )
    return path


@pytest.fixture
def spy_extractor(monkeypatch) -> list[dict]:
    """Replace the script's FeatureExtractor with one that records its kwargs.

    It still encodes -- zeros of the backbone's own geometry -- so `main` runs
    to completion and the recorded call can be asserted (L8: a spy whose calls
    are never asserted is not a test).
    """
    calls: list[dict] = []

    class _Spy:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.backbone = kwargs["backbone"]

        def encode(self, frames):
            n_patches, patch_dim = BACKBONE_GEOMETRY[self.backbone]
            return np.zeros((len(frames), n_patches, patch_dim), dtype=np.float16)

    monkeypatch.setattr(script, "FeatureExtractor", _Spy)
    return calls


# --- the command line -------------------------------------------------------


@pytest.mark.parametrize("backbone", BACKBONES)
def test_every_registered_backbone_is_accepted(backbone):
    assert script._parser().parse_args(["--backbone", backbone]).backbone == backbone


def test_pixel_ae_is_accepted_and_the_arm_name_cnn_is_not():
    """The backbone is `pixel_ae`; `cnn` is an encoder kind, not a cache.

    Stated with literals rather than over BACKBONES: if the registry lost
    `pixel_ae`, the parametrised test above would shrink and stay green.
    """
    assert script._parser().parse_args(["--backbone", "pixel_ae"]).backbone == "pixel_ae"
    with pytest.raises(SystemExit):
        script._parser().parse_args(["--backbone", "cnn"])


def test_checkpoint_defaults_to_the_m2_encoder_and_parses_as_a_path():
    parser = script._parser()
    default = parser.parse_args([]).checkpoint
    assert default == PIXEL_AE_CHECKPOINT
    assert isinstance(default, Path)
    explicit = parser.parse_args(["--checkpoint", "runs/other/autoencoder_cnn.pt"]).checkpoint
    assert explicit == Path("runs/other/autoencoder_cnn.pt")
    assert isinstance(explicit, Path), "a str here reaches FeatureExtractor as a str"


# --- the disk guard ---------------------------------------------------------


def test_expectations_cover_every_backbone():
    assert set(EXPECTED_BYTES) == set(BACKBONES), "add a hand-written row"
    assert set(EXPECTED_SUFFIX) == set(BACKBONES), "add a hand-written row"


def test_cache_bytes_matches_the_m1_dataset_numbers():
    """The M1 buffer is 122 episodes, 59,633 frames (summed from the filenames).

    Its ViT caches are 2.93 GB on disk, so `dinov2` must reproduce that; the
    `pixel_ae` cache is 12x smaller. Stated in bytes so a 4-byte (float32)
    slip or a ViT-width slip for `pixel_ae` is a different number, not a
    rounding.
    """
    assert script.cache_bytes(59_633, "dinov2") == 2_931_081_216
    assert script.cache_bytes(59_633, "random_vit") == 2_931_081_216
    assert script.cache_bytes(59_633, "pixel_ae") == 244_256_768


@pytest.mark.parametrize("backbone", BACKBONES)
def test_main_asks_the_disk_for_the_backbones_own_byte_count(
    backbone, data_dir, spy_extractor, monkeypatch
):
    """The old estimate was `frames * 64 * 384 * 2` for every backbone.

    For `pixel_ae` that demands 2.93 GB free to write 0.24 GB -- on the
    2.3-GB-free machine this script was written for, the one cache that fits
    would be the one refused.
    """
    asked: list[tuple[Path, int]] = []
    monkeypatch.setattr(
        script, "require_free_bytes", lambda path, needed: asked.append((path, needed))
    )
    script.main(
        ["--data", str(data_dir), "--backbone", backbone, "--device", "cpu",
         "--checkpoint", str(data_dir / "unused.pt")]
    )
    assert asked == [(data_dir, EXPECTED_BYTES[backbone])]


# --- the checkpoint reaches the extractor -----------------------------------


@pytest.mark.parametrize("backbone", BACKBONES)
def test_main_forwards_checkpoint_to_the_extractor(backbone, data_dir, spy_extractor):
    """Every backbone, not only `pixel_ae`: the flag is forwarded unconditionally
    and `build_backbone` decides what to do with it. A conditional here would
    be a second place that knows which backbone reads a checkpoint."""
    checkpoint = data_dir / "some_autoencoder.pt"
    script.main(
        ["--data", str(data_dir), "--backbone", backbone, "--device", "cpu",
         "--seed", "3", "--checkpoint", str(checkpoint)]
    )
    assert spy_extractor == [
        {"backbone": backbone, "device": "cpu", "seed": 3, "checkpoint": checkpoint}
    ]


def test_main_forwards_the_default_checkpoint_when_none_is_given(data_dir, spy_extractor):
    script.main(["--data", str(data_dir), "--backbone", "pixel_ae", "--device", "cpu"])
    [call] = spy_extractor
    assert call["checkpoint"] == PIXEL_AE_CHECKPOINT


# --- the cache itself -------------------------------------------------------


def test_pixel_ae_cache_is_the_loaded_encoders_output(data_dir, tmp_path):
    """End to end through the real FeatureExtractor on CPU.

    The file must sit under the `pixel_ae` suffix, hold (T+1, 64, 32) float16,
    and be bit-for-bit the forward pass of the encoder in `--checkpoint` (M8:
    shape and dtype alone are satisfied by a random-init encoder, or by
    whatever `PIXEL_AE_CHECKPOINT` happens to hold on this machine). Both
    sides run one CPU batch of the same six frames, so bit-equality is the
    right bar -- `test_features.py` holds the DINOv2 path to the same one.
    """
    checkpoint = _fake_m2_checkpoint(tmp_path / "autoencoder_cnn.pt", seed=0)
    script.main(
        ["--data", str(data_dir), "--backbone", "pixel_ae", "--device", "cpu",
         "--checkpoint", str(checkpoint)]
    )

    [ep_path] = ReplayBuffer(data_dir, capacity_transitions=10_000).episode_paths()
    out = ep_path.with_suffix(".features_pixel_ae.npy")  # literal: the name IS the contract
    assert out.is_file(), sorted(p.name for p in data_dir.iterdir())
    assert not ep_path.with_suffix(".features.npy").exists(), (
        "written under the dinov2 suffix -- frozen_ssl would train on these"
    )
    saved = np.load(out)
    with np.load(ep_path) as data:
        obs = data["obs"]
    assert saved.shape == (obs.shape[0], 64, 32)
    assert saved.dtype == np.float16

    encoder = CNNEncoder(EncoderConfig(kind="cnn"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    encoder.load_state_dict({k.removeprefix("encoder."): v for k, v in state.items()})
    encoder.eval()
    with torch.no_grad():
        reference = encoder(torch.from_numpy(obs)).reshape(N_FRAMES, 64, 32)
    np.testing.assert_array_equal(saved, reference.to(torch.float16).numpy())
    # L1: if every frame encoded to one row, the equality above could not
    # distinguish a cache in frame order from one that is not.
    assert not np.array_equal(saved[0], saved[1])

    # The contents follow --checkpoint: a second encoder, a second cache.
    other = _fake_m2_checkpoint(tmp_path / "autoencoder_cnn_other.pt", seed=1)
    script.main(
        ["--data", str(data_dir), "--backbone", "pixel_ae", "--device", "cpu",
         "--checkpoint", str(other)]
    )
    assert not np.array_equal(np.load(out), saved)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_clear_removes_only_the_named_backbones_cache(backbone, data_dir, spy_extractor):
    """Three caches coexist now; `--clear` must take exactly the one named.

    It must also build no extractor: clearing the `pixel_ae` cache cannot be
    made to depend on the checkpoint that produced it still existing.
    """
    [ep_path] = ReplayBuffer(data_dir, capacity_transitions=10_000).episode_paths()
    for suffix in EXPECTED_SUFFIX.values():
        np.save(ep_path.with_suffix(suffix), np.zeros((N_FRAMES, 1, 1), dtype=np.float16))

    script.main(["--data", str(data_dir), "--backbone", backbone, "--clear"])

    survivors = {
        name for name, suffix in EXPECTED_SUFFIX.items()
        if ep_path.with_suffix(suffix).is_file()
    }
    assert survivors == set(BACKBONES) - {backbone}
    assert spy_extractor == [], "--clear constructed a backbone"
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/data/test_cache_features_script.py -q`

Expected: FAIL, at collection or at the first test depending on what the script
imports when you get here. If `scripts/cache_features.py` still has its original
lines 20–27 (`from mbfps.data.features import (BACKBONES, FEATURE_DIM, N_PATCHES, ...)`),
the module load at `_SPEC.loader.exec_module(script)` fails with
`ImportError: cannot import name 'FEATURE_DIM' from 'mbfps.data.features'`, because
Task 1 removed that constant — one collection error, zero tests run. If an earlier
task already trimmed that import, collection succeeds and every test fails with
`AttributeError: module 'cache_features_script' has no attribute '_parser'` (or
`'cache_bytes'`). Either is the red you want. What must not happen is a pass.

- [ ] **Step 3: Implement**

Replace `scripts/cache_features.py` in full. `cache_bytes` below has the same two-line body as (its docstring is reworded)
Task 1's (Task 1's three tests must stay green through this replacement); the docstring
now says which cache is the exception and why; `_parser` and `main(argv)` exist so the tests can drive it; and
`--checkpoint` is forwarded for every backbone — `build_backbone` is the one place
that knows `pixel_ae` is the backbone that reads it, and a conditional here would
be a second copy of that knowledge.

```python
# scripts/cache_features.py
"""Cache frozen-backbone features for every episode in a buffer.

Run once per backbone. Because a ViT cache is ~2.93 GB and this machine has
had as little as 2.3 GB free, the workflow is one cache at a time:

    cache_features.py --backbone dinov2      # train arm 2
    cache_features.py --backbone dinov2 --clear
    cache_features.py --backbone random_vit  # train arm 3

The `pixel_ae` cache is the exception. Its backbone is the M2 pixel
autoencoder's encoder, read from `--checkpoint` (default
`PIXEL_AE_CHECKPOINT`), and its rows are (64, 32) rather than (64, 384), so the
whole cache is ~0.24 GB and coexists with either ViT cache:

    cache_features.py --backbone pixel_ae --checkpoint runs/m2_fixed/autoencoder_cnn.pt

Features must be generated once on one device: CPU and MPS outputs differ in
~3% of float16 elements, so mixing them would silently break the input
equality the study depends on.
"""

import argparse
import time
from pathlib import Path

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.features import (
    BACKBONE_GEOMETRY,
    BACKBONES,
    PIXEL_AE_CHECKPOINT,
    FeatureExtractor,
    cache_episode_features,
    require_free_bytes,
)
from mbfps.data.loader import feature_suffix


def cache_bytes(frames: int, backbone: str) -> int:
    """Bytes a float16 cache of `frames` rows of `backbone`'s geometry occupies.

    Read from the registry, not from a module constant: the estimate used to
    be `frames * 64 * 384 * 2` for every backbone, which for `pixel_ae` would
    demand 2.93 GB of free disk to write 0.24 GB. On the 2.3-GB-free machine
    this script was written for, that refusal would block the one cache that
    fits.
    """
    n_patches, patch_dim = BACKBONE_GEOMETRY[backbone]
    return frames * n_patches * patch_dim * 2  # float16


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--backbone", choices=BACKBONES, default="dinov2")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    # Forwarded to FeatureExtractor for EVERY backbone; build_backbone is the
    # one place that knows pixel_ae is the backbone that reads it. A
    # conditional here would be a second copy of that knowledge.
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PIXEL_AE_CHECKPOINT,
        help="M2 autoencoder checkpoint whose encoder is the pixel_ae backbone "
        "(ignored by the ViT backbones)",
    )
    parser.add_argument(
        "--clear", action="store_true", help="delete existing caches and exit"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)

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
    needed = cache_bytes(frames, args.backbone)
    print(f"episodes={len(paths)} frames={frames} needs={needed / 1e9:.2f} GB")
    require_free_bytes(args.data, needed)

    extractor = FeatureExtractor(
        backbone=args.backbone,
        device=args.device,
        seed=args.seed,
        checkpoint=args.checkpoint,
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

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/data/test_cache_features_script.py -q`
Expected: PASS (21 tests — Task 1's 3 `cache_bytes` tests plus this task's 18; ~3 s; the two end-to-end runs build 105 MB checkpoints in `tmp_path`)

Then the whole suite: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q`
Expected: green, the prior count plus 18.

- [ ] **Step 5: Mutation-test**

The script is loaded by path from the working tree, so the harness mutates
`scripts/cache_features.py` in place and restores it from a copy. The three
self-checks are built in: a known-fatal mutation runs FIRST (a harness that
shows green on it is not running the mutated file); the mutated file is
confirmed to be the file under test (`grep` counts the mutation in the very
path the test's `_SPEC` loads — no export is involved, and `mbfps` itself is
imported from the editable install at `src/`, which is the code being tested);
and `__pycache__` is cleared before every run under `PYTHONDONTWRITEBYTECODE=1`
with `-p no:cacheprovider`, so a same-byte-length mutation cannot leave a stale
`.pyc` behind.

```bash
cp scripts/cache_features.py /tmp/cache_features.py.orig
run() {
  find . -name __pycache__ -exec rm -rf {} + 2>/dev/null
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/data/test_cache_features_script.py \
    -q -p no:cacheprovider 2>&1 | grep -E "^(FAILED|ERROR|[0-9]+ (passed|failed))" | sed 's/ - .*//'
}
mut() {  # mut "<label>" "<exact text to replace>" "<replacement>"
  echo "=== $1"
  cp /tmp/cache_features.py.orig scripts/cache_features.py
  .venv/bin/python - "$2" "$3" <<'EOF'
import sys
from pathlib import Path
p = Path("scripts/cache_features.py"); s = p.read_text()
assert s.count(sys.argv[1]) == 1, f"mutation target not unique: {sys.argv[1]!r}"
p.write_text(s.replace(sys.argv[1], sys.argv[2]))
EOF
  echo "mutated lines in the file under test: $(grep -c "$3" scripts/cache_features.py)"
  run
}

mut "M0 known-fatal: no --checkpoint flag"   '        "--checkpoint",' '        "--checkpoint-x",'
mut "M1 geometry always read for dinov2"     'BACKBONE_GEOMETRY[backbone]' 'BACKBONE_GEOMETRY["dinov2"]'
mut "M2 float32 bytes"                       'patch_dim * 2  # float16' 'patch_dim * 4  # float16'
mut "M3 main asks for the ViT byte count"    'needed = cache_bytes(frames, args.backbone)' 'needed = cache_bytes(frames, "dinov2")'
mut "M4 --checkpoint parsed, not forwarded"  '        checkpoint=args.checkpoint,' ''
mut "M5 default None"                        '        default=PIXEL_AE_CHECKPOINT,' '        default=None,'
mut "M6 --checkpoint as str"                 '        type=Path,
        default=PIXEL_AE_CHECKPOINT,' '        type=str,
        default=str(PIXEL_AE_CHECKPOINT),'
mut "M7 choices hardcoded to the ViTs"       'choices=BACKBONES' 'choices=("dinov2", "random_vit")'
mut "M8 --clear always takes the dinov2 suffix" 'feature_suffix(args.backbone)' 'feature_suffix("dinov2")'
mut "M9 --clear falls through into caching"  '        print(f"removed={removed} {args.backbone} feature files from {args.data}")
        return' '        print(f"removed={removed} {args.backbone} feature files from {args.data}")'
mut "M10 forwarded under another kwarg name" 'checkpoint=args.checkpoint,' 'ckpt=args.checkpoint,'

cp /tmp/cache_features.py.orig scripts/cache_features.py && rm /tmp/cache_features.py.orig
git diff --stat scripts/cache_features.py   # must print nothing
echo "=== restored"; run                    # must print: 18 passed
```

| mutation | must be caught by |
|---|---|
| M0 known-fatal: the flag is `--checkpoint-x` | 9 failures — every test that parses or forwards `--checkpoint`. A green run here means the harness is not running the mutated file; stop and fix the harness. |
| M1 `cache_bytes` reads `BACKBONE_GEOMETRY["dinov2"]` for every backbone | `test_cache_bytes_matches_the_m1_dataset_numbers`, `test_main_asks_the_disk_for_the_backbones_own_byte_count[pixel_ae]` |
| M2 4 bytes per element | `test_cache_bytes_matches_the_m1_dataset_numbers`, `test_main_asks_the_disk_...` (all three) |
| M3 `main` asks the disk for the ViT byte count regardless of `--backbone` | `test_main_asks_the_disk_for_the_backbones_own_byte_count[pixel_ae]` |
| M4 `--checkpoint` parsed but never passed to `FeatureExtractor` | `test_main_forwards_checkpoint_to_the_extractor` (all three), `test_main_forwards_the_default_checkpoint_when_none_is_given`, and `test_pixel_ae_cache_is_the_loaded_encoders_output` — on a machine with `runs/m2_fixed/autoencoder_cnn.pt` the extractor silently loads THAT encoder and the bit-equality against the fake checkpoint's forward pass fails (M8); on a machine without it, `FileNotFoundError` |
| M5 `default=None` | `test_checkpoint_defaults_to_the_m2_encoder_and_parses_as_a_path`, `test_main_forwards_the_default_checkpoint_when_none_is_given` |
| M6 `type=str` | `test_checkpoint_defaults_...` (`isinstance(..., Path)`) and every forwarding test (`Path("x") != "x"`) |
| M7 `choices=("dinov2", "random_vit")` | `test_every_registered_backbone_is_accepted[pixel_ae]`, `test_pixel_ae_is_accepted_and_the_arm_name_cnn_is_not`, and every `pixel_ae` run (argparse `SystemExit`) |
| M8 `--clear` deletes the `dinov2` suffix whatever `--backbone` says | `test_clear_removes_only_the_named_backbones_cache[random_vit]`, `[pixel_ae]` |
| M9 `--clear` does not `return` | `test_clear_removes_only_the_named_backbones_cache` (all three: the spy records a construction) |
| M10 forwarded as `ckpt=` | every forwarding test (`TypeError` from the `**kwargs` spy; the end-to-end run from the real `FeatureExtractor`) |

Verified on 2026-09-11 against a scratch copy carrying Task 1's `features.py`
contract: M0 kills 9, M1 2, M2 4, M3 1, M4 5, M5 2, M6 5, M7 7, M8 2, M9 3,
M10 5; restore → 18 passed. Any mutation that survives on your run is a missing
test. Add it before committing.

- [ ] **Step 6: Commit**

```bash
git add scripts/cache_features.py tests/data/test_cache_features_script.py
git commit -m "feat: cache_features.py takes --checkpoint and sizes the disk guard per backbone

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Produce the `pixel_ae` cache**

Both ViT caches (2.93 GB each) are already on disk, and this one is 0.24 GB, so no
`--clear` is needed first. `caffeinate -dimsu`, not `-i`: `-i` has been measured
insufficient on this host.

```bash
mkdir -p runs/m3_study_v2
caffeinate -dimsu .venv/bin/python -u scripts/cache_features.py \
  --backbone pixel_ae --checkpoint runs/m2_fixed/autoencoder_cnn.pt --device mps \
  2>&1 | tee runs/m3_study_v2/cache_pixel_ae.log
```

Expected output (elapsed figures are what to expect, not what to match):

```
episodes=122 frames=59633 needs=0.24 GB
[20/122] elapsed=...s
[40/122] elapsed=...s
[60/122] elapsed=...s
[80/122] elapsed=...s
[100/122] elapsed=...s
[120/122] elapsed=...s
[122/122] elapsed=...s
backbone=pixel_ae device=mps seed=0
features_cached=122 elapsed_s=...
frames_per_second=...
```

`needs=0.24 GB`, not the spec §2.1's "~0.5 GB": 59,633 × 64 × 32 × 2 bytes =
244,256,768. The spec's figure is a float32 count. Expect one to five minutes
end to end: the encoder itself is ~20 s on MPS (measured 3,406 frames/s at batch
32 on the real checkpoint, 526 frames in 0.15 s), and the rest is decompressing
each episode's `obs` twice — once in `load_all()` for the frame count, once to
encode. If the first line reads `needs=2.93 GB`, the disk guard is still sized
for the ViTs and Step 3 did not land.

- [ ] **Step 8: Verify the cache and record what it is**

This is the check spec §1 is about. The end-to-end pixel arm's target was born
collapsed — a randomly-initialised `CNNEncoder` on real frames emits per-dim std
of ~0.0007 (median over 2048 dims, measured on 256 frames of episode 0) — and a
`pixel_ae` cache with per-dim std ~0 would reproduce that failure under a new
name. A cache is only fit to enter the study if every one of its 2048 dims moves
across frames. The floors below sit three decades above the collapsed value and
one below the trained encoder's: on the first three episodes (1,578 frames) the
real M2 encoder measured per-dim std min 0.37 / median 1.06 / max 5.25 and global
std 1.34.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - pixel_ae <<'EOF' 2>&1 | tee -a runs/m3_study_v2/cache_pixel_ae.log
import sys
import numpy as np
from pathlib import Path
from mbfps.data.buffer import ReplayBuffer
from mbfps.data.features import BACKBONE_GEOMETRY
from mbfps.data.loader import feature_suffix

backbone = sys.argv[1]
n_patches, patch_dim = BACKBONE_GEOMETRY[backbone]
root = Path("data/my_way_home")
paths = ReplayBuffer(root, capacity_transitions=10**9).episode_paths()
assert len(paths) == 122, len(paths)

d = n_patches * patch_dim
count, total_bytes = 0, 0
sum1, sum2 = np.zeros(d), np.zeros(d)   # float64 accumulators over all frames
for path in paths:
    feats = np.load(path.with_suffix(feature_suffix(backbone)), mmap_mode="r")
    with np.load(path) as data:
        n_obs = data["obs"].shape[0]   # the episode's own frame count, not the filename's
    assert feats.shape == (n_obs, n_patches, patch_dim), (path.name, feats.shape, n_obs)
    assert feats.dtype == np.float16, (path.name, feats.dtype)
    x = np.asarray(feats, dtype=np.float64).reshape(n_obs, d)
    assert np.isfinite(x).all(), f"{path.name}: non-finite values"
    count += n_obs
    sum1 += x.sum(0)
    sum2 += (x * x).sum(0)
    total_bytes += feats.nbytes
mean = sum1 / count
std = np.sqrt(np.maximum(sum2 / count - mean**2, 0.0))
all_mean = sum1.sum() / (count * d)
global_std = np.sqrt(sum2.sum() / (count * d) - all_mean**2)
print(f"backbone={backbone} files={len(paths)} frames={count} "
      f"rows=({n_patches}, {patch_dim}) size_gb={total_bytes / 1e9:.3f}")
print(f"per_dim_std min={std.min():.4f} p05={np.percentile(std, 5):.4f} "
      f"median={np.median(std):.4f} max={std.max():.4f} dims_below_0.01={(std < 0.01).sum()}")
print(f"global_std={global_std:.4f}")
assert std.min() > 0.0, "a dimension with zero variance: the cache carries a constant"
assert std.min() > 0.01 and np.median(std) > 0.1, "collapsed target (spec section 1)"
print("OK")
EOF
```

Expected (~20 s; the same script run on the `random_vit` cache prints
`files=122 frames=59633 rows=(64, 384) size_gb=2.931`, `per_dim_std min=0.2492
p05=0.3372 median=0.5849 max=1.6683 dims_below_0.01=0`, `global_std=1.0000` —
the 1.0000 is the number `EncoderConfig.standardise_features` already cites,
which is the accumulator's cross-check):

```
backbone=pixel_ae files=122 frames=59633 rows=(64, 32) size_gb=0.244
per_dim_std min=0.3... p05=... median=1.0... max=5... dims_below_0.01=0
global_std=1.3...
OK
```

An `AssertionError` on shape names the file and both counts; on the collapsed
check, STOP — do not proceed to the spike or the study with this cache. The
likeliest causes are the checkpoint's `decoder.*`-only weights being loaded
(the encoder left at init) or `encode` normalising the input a second time;
both are Task 1's `build_backbone` / `FeatureExtractor.encode`, not this script.

Then fill "## Task 5 results" at the bottom of this plan:

| quantity | value |
|---|---|
| cache size on disk (`du -ch data/my_way_home/*.features_pixel_ae.npy \| tail -1`) | — |
| files / frames / rows | 122 / 59,633 / (64, 32) |
| per-dim std min / p05 / median / max, dims below 0.01 | — |
| global std (DINOv2 cache 2.3559, random_vit 1.0000, for the LayerNorm rationale) | — |
| encode wall time and frames/s from the script's last two lines | — |
| `git rev-parse HEAD` the cache was written at | — |

The per-dim std row is the one that licenses the spike in §3: it is the direct
measurement of "the target is informative from step 0" that §1 could only infer
from `embedding_loss`.

- [ ] **Step 9: Commit the record**

Nothing under `data/` or `runs/` is committed; only the plan changes.

```bash
git add docs/superpowers/plans/2026-09-11-mb-fps-m3c-pixel-ae-arm.md
git commit -m "docs: the pixel_ae cache is written and its target is not collapsed

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

---

### Task 6: The record carries git_sha, device, encoder_params and the full history

The M3b write-up had to reconstruct three facts after the fact: which code state produced the nine cells (from file mtimes), where each cell ran (from the log), and whether the pixel arm's dynamics KL ever cleared the free-bits floor (from a scalar rate, because the per-step history was discarded — its open item 6). Spec §2.2 adds a fourth: the one place the arms are not byte-identical is the bottleneck's input width (`pixel_ae` 1,056 parameters against the ViT arms' 12,320), and it is to be "recorded as an `encoder_params` integer in every study record rather than hidden". Each becomes a top-level field of the record, and the driver refuses to call a cell finished without them.

**Files:**
- Modify: `src/mbfps/eval/study.py` — imports (lines 23–29); new helpers inserted between `load_record` (ends line 314) and `_probe_summary` (line 317); `run_job` docstring (lines 350–373), the post-load block (lines 398–399), and the record dict (lines 431–462)
- Modify: `scripts/run_study.py` — `REQUIRED_RECORD_KEYS` and its docstring (lines 97–122)
- Test: `tests/eval/test_study.py` (append after line 2200)
- Test: `tests/eval/test_run_study.py` — `EXPECTED_RECORD_KEYS`/`OPTIONAL_RECORD_KEYS` (lines 272–286), `_record_body` (lines 325–388), two docstring counts (lines 893–896 and 900–918), two new tests inserted before `test_a_record_holding_exactly_the_required_keys_is_complete` (line 900)

> READ THIS FIRST — the dotted-path scheme already handles list indices. `_sanitise`
> (`study.py` lines 199–203) walks lists with `f"{path}.{i}"`, and `load_record` (lines
> 301–313) indexes a list container with `int(segment)`. Verified before writing this
> task: `{"history": {"parts": [{"kl_dyn": 0.1}, {"kl_dyn": nan}]}}` is written as
> `null` under the map key `history.parts.1.kl_dyn` and read back as a float NaN. So
> `history` needs NO change to the non-finite policy; this task only pins that the
> scheme keeps working four segments deep, because the record now depends on it.
>
> One existing test assumes two-segment paths: `test_an_undefined_gap_is_never_read_back_as_zero`
> does `section, field = dotted.split(".", 1)` then `written[section][field]` over the
> shared `record` fixture's map. It is unaffected because that fixture trains five steps
> to finite losses (the only non-finite entries are `gap_final`/`gap_mean`). Do not
> "fix" it in this task; the deep path is covered by the two new tests below.

**Interfaces:**
- Consumes: `train_world_model(cfg, buffer, out_dir, log_every=100) -> dict` from `mbfps.training.world_model`, whose returned history carries `"loss": list[float]` (one per step) and `"parts": list[dict[str, float]]` (one per step, keys `embedding`, `reward`, `continue`, `kl_dyn`, `kl_rep`, all Python floats — `WorldModel.forward` builds them with `float(...)`). `WorldModel(cfg).encoder`, the module `build_encoder` returns. `get_device(prefer) -> torch.device` from `mbfps.utils.device`. `ARMS == ("pixel_ae", "frozen_ssl", "random_vit")` from `mbfps.utils.config`. `to_json_record`, `write_record`, `load_record`, `NONFINITE_KEY`, `job_record_path`, `StudyJob` (already in `study.py`). The `small_buffer` fixture in `tests/eval/conftest.py`, which after the `ARMS` change writes a `(41, 64, 32)` `.features_pixel_ae.npy` beside the two `(41, 64, 384)` caches for every episode — the parametrised `encoder_params` test runs `run_job` on every arm and reads it.
- Produces:
  - `run_job(...)` record gains four top-level keys, in this exact shape: `"git_sha": str` (the 40-hex output of `git rev-parse HEAD` asked of `study.py`'s own directory, or `"unknown"` — never raises), `"device": str` (`str(torch_device)`), `"encoder_params": int` (sum of `p.numel()` over `model.encoder.parameters()` with `requires_grad`), `"history": {"loss": list[float], "parts": list[dict[str, float]]}` (the full per-step history, every step).
  - `mbfps.eval.study.UNKNOWN_GIT_SHA: str = "unknown"`
  - `mbfps.eval.study._git_sha() -> str` — private helper behind the key, so its four failure modes can be tested without four training runs.
  - `mbfps.eval.study._trainable_parameters(module: torch.nn.Module) -> int` — private helper behind the key, so the `requires_grad` filter can be tested on a module that has a frozen parameter (no study arm does).
  - `scripts/run_study.py`: `REQUIRED_RECORD_KEYS` gains `"git_sha"`, `"device"`, `"encoder_params"`, `"history"`.
  - `tests/eval/test_run_study.py`: `M3C_RECORD_KEYS: frozenset[str]` literal; `EXPECTED_RECORD_KEYS` gains the four; `_record_body` writes the four.
  - `job_summary` is deliberately NOT changed. Its block is pinned character-for-character by `EXPECTED_SUMMARY`, and the four fields are for the aggregation across nine files (the report is where "all nine share one sha" is checked), not for the per-cell log line, which already prints the requested device.

- [ ] **Step 1: Write the failing tests for the record (`tests/eval/test_study.py`)**

Append at the end of the file (after `test_the_out_dir_and_the_record_path_accept_the_strings_argparse_gives`, line 2200). The module already imports `math`, `np`, `pytest`, `Path`, `study`, `run_job`, `StudyJob`, `to_json_record`, `write_record`, `load_record`, `NONFINITE_KEY`, `job_record_path`, and defines `JOB`, `JOB_KW`, `JOB_SEED`, `PAIRWISE_DISTINCT_PARAMETERS`, `strict_loads`, and the `record`/`small_buffer` fixtures — every name below resolves.

```python
# tests/eval/test_study.py  (append)


# --------------------------------------------------------------------------
# provenance: git_sha, device, encoder_params, and the full history (M3c)
# --------------------------------------------------------------------------
#
# The M3b write-up had to RECONSTRUCT three things after the fact: which code
# state produced the nine cells (from file mtimes), where each cell ran (from
# the log), and whether the pixel arm's KL ever cleared the free-bits floor
# (from a scalar rate, because the per-step history was thrown away). Each is
# now a field. Every guard below pins the field to a value the test computes
# INDEPENDENTLY -- the test's own `git rev-parse`, its own `numel` sum, the
# history it handed `train_world_model` -- because "is a string" and "is a
# list" are satisfied by a hardcoded literal.

#: What the record must hold under `history`, and the five loss terms every
#: `parts[i]` carries, spelled out as literals so a term dropped from
#: `WorldModel.forward`'s dict fails here by name.
HISTORY_KEYS = frozenset({"loss", "parts"})
PART_KEYS = frozenset({"embedding", "reward", "continue", "kl_dyn", "kl_rep"})

#: Spec 2.2's one stated asymmetry, per arm: `pixel_ae`'s bottleneck is
#: `Linear(32 -> 32)` = 32*32 + 32; the ViT arms' is `Linear(384 -> 32)` =
#: 384*32 + 32. A LITERAL, not a computation, so that a geometry change that
#: silently alters the bottleneck width shows up as a number, not as two
#: computations agreeing with each other.
EXPECTED_ENCODER_PARAMS = {
    "pixel_ae": 1_056,
    "frozen_ssl": 12_320,
    "random_vit": 12_320,
}


def _git_head(cwd: Path) -> str | None:
    """`git rev-parse HEAD` at `cwd`, run by the TEST, or None if git cannot."""
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd,
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def test_the_record_carries_the_sha_git_reports_for_the_code_that_ran(record):
    """`git_sha` must equal what `git rev-parse HEAD` says HERE, in this
    checkout -- the test runs git itself and compares. "Is a 40-character
    string" is satisfied by a literal pasted into `run_job`, and a literal is
    precisely the thing that would make nine records from two code states
    look like one.

    The sha is asked of the directory `study.py` lives in, not the process's
    cwd; the two are the same repository here, and this test pins the value,
    not the cwd (that is `test_git_sha_is_asked_of_the_code_not_the_cwd`).
    """
    repo_root = Path(__file__).resolve().parents[2]
    expected = _git_head(repo_root)
    if expected is None:
        pytest.skip("git cannot report HEAD for this checkout, so the value "
                    "cannot be pinned here; the failure path is tested below")
    assert len(expected) == 40 and set(expected) <= set("0123456789abcdef"), (
        f"the test's own git call returned {expected!r}, not a sha; the "
        "comparison below would be against garbage")
    assert record["git_sha"] == expected
    assert isinstance(record["git_sha"], str)
    assert record["git_sha"] != study.UNKNOWN_GIT_SHA


def test_git_sha_is_asked_of_the_code_not_the_cwd(monkeypatch):
    """`cwd=` must be `study.py`'s own directory. A study launched from
    `/scratch` must report the code's commit, not `/scratch`'s (or fail
    because `/scratch` is not a repository)."""
    import subprocess

    seen: dict = {}
    real_run = subprocess.run

    def spy(args, **kwargs):
        seen["args"] = list(args)
        seen["cwd"] = kwargs.get("cwd")
        return real_run(args, **kwargs)

    monkeypatch.setattr(study.subprocess, "run", spy)
    study._git_sha()
    assert seen["args"] == ["git", "rev-parse", "HEAD"]
    assert Path(seen["cwd"]).resolve() == Path(study.__file__).resolve().parent, (
        f"git was asked at {seen['cwd']!r}, not where the code lives")


@pytest.mark.parametrize("failure", [
    "no_git_executable", "git_hangs", "not_a_repository", "empty_stdout",
])
def test_git_sha_is_unknown_rather_than_an_exception_when_git_cannot_answer(
    monkeypatch, failure
):
    """A rented box without git must not lose a cell to a provenance field.

    Four ways the subprocess can fail to produce a sha, EACH EXERCISED ALONE
    so that no single guard covers for another: the executable is missing
    (`OSError`), it hangs (`TimeoutExpired`, a `SubprocessError`), it runs but
    the directory is not a repository (non-zero status, and git's usual
    empty stdout), and the degenerate zero-status-empty-stdout -- which a
    guard on the status alone would pass through as `""`.

    `not_a_repository` deliberately puts a non-sha ON stdout with the
    non-zero status, so the returncode guard is what has to catch it; the
    `or UNKNOWN` fallback alone would let "abc" through.
    """
    import subprocess
    from types import SimpleNamespace

    def fake_run(args, **kwargs):
        if failure == "no_git_executable":
            raise FileNotFoundError(2, "No such file or directory", "git")
        if failure == "git_hangs":
            raise subprocess.TimeoutExpired(args, kwargs.get("timeout", 0))
        if failure == "not_a_repository":
            return SimpleNamespace(returncode=128, stdout="abc\n", stderr="fatal")
        return SimpleNamespace(returncode=0, stdout="\n", stderr="")

    monkeypatch.setattr(study.subprocess, "run", fake_run)
    assert study._git_sha() == study.UNKNOWN_GIT_SHA == "unknown"


def test_the_record_says_where_the_cell_ran_not_what_was_asked_for(
    tmp_path, small_buffer, monkeypatch
):
    """`device` is `str(torch_device)` -- the device `get_device` RESOLVED --
    not the `device` string the job was launched with. On a box without CUDA,
    `--device cuda` runs on the CPU, and a record that says "cuda" is the lie
    the field exists to prevent.

    `get_device` is pinned to CPU here so the guard is machine-independent:
    without the pin, `get_device(prefer="cuda")` already IS cpu on this Mac
    and would be cuda on the box, so `"device": device` would pass on the
    box for the wrong reason. With it, the request ("cuda") and the answer
    ("cpu") differ everywhere."""
    import torch

    requested = "cuda"
    resolved = torch.device("cpu")
    seen: dict = {}

    def pinned_get_device(prefer="mps"):
        seen["prefer"] = prefer
        return resolved

    monkeypatch.setattr(study, "get_device", pinned_get_device)
    result = run_job(JOB, small_buffer, tmp_path,
                     **dict(JOB_KW, device=requested))

    assert seen["prefer"] == requested, "run_job did not ask for the job's device"
    assert str(resolved) != requested, (
        "the pinned device spells the same as the request, so the assertion "
        "below cannot tell the resolved device from the requested string")
    assert result["device"] == str(resolved) == "cpu"
    assert isinstance(result["device"], str)
    assert result["device"] != requested


@pytest.mark.parametrize("arm", sorted(EXPECTED_ENCODER_PARAMS))
def test_encoder_params_counts_the_trainable_encoder_parameters_of_each_arm(
    tmp_path, small_buffer, arm
):
    """`encoder_params` is the sum of `numel` over the ENCODER's trainable
    parameters, computed here independently of `run_job` -- and equal to the
    spec's own stated number for the arm.

    Run for every arm because the value is the one place the arms differ and
    a literal (12,320) would pass on two of the three. The whole-model count
    is asserted different so that `model.parameters()` in place of
    `model.encoder.parameters()` is a visible mutation and not a coincidence.
    """
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import ARMS, get_config

    assert set(EXPECTED_ENCODER_PARAMS) == set(ARMS), (
        "the expected counts must name exactly the study's arms; an arm "
        "without a stated count is an arm whose asymmetry is unrecorded")

    job = StudyJob(arm, JOB_SEED)
    result = run_job(job, small_buffer, tmp_path, **JOB_KW)

    model = WorldModel(get_config(arm, device="cpu", seed=JOB_SEED))
    expected = sum(p.numel() for p in model.encoder.parameters()
                   if p.requires_grad)
    whole_model = sum(p.numel() for p in model.parameters())
    assert expected != whole_model, (
        "the encoder is the whole model, so counting everything would pass")
    assert result["encoder_params"] == expected == EXPECTED_ENCODER_PARAMS[arm]
    assert isinstance(result["encoder_params"], int)
    assert not isinstance(result["encoder_params"], bool)


def test_encoder_params_excludes_frozen_parameters():
    """The `requires_grad` filter, alone. On every study arm all encoder
    parameters train, so `run_job` cannot show the filter; a module with one
    frozen tensor can. Without it a future encoder wrapping a frozen backbone
    would report the backbone's millions as "trained"."""
    import torch

    module = torch.nn.Module()
    module.trained = torch.nn.Parameter(torch.zeros(3, 4))          # 12
    module.frozen = torch.nn.Parameter(torch.zeros(100), requires_grad=False)
    module.register_buffer("stat", torch.zeros(1000))               # never
    assert study._trainable_parameters(module) == 12


def test_the_record_carries_the_full_history_at_the_jobs_own_length(record):
    """One `loss` and one five-term `parts` entry PER STEP, for every step
    the job trained -- `steps` entries, not the last twenty and not a mean.
    The M3b write-up's section 1 had to reconstruct the pixel arm's KL
    trajectory because this list was discarded."""
    history = record["history"]
    assert set(history) == HISTORY_KEYS
    assert isinstance(history["loss"], list)
    assert isinstance(history["parts"], list)
    assert len(history["loss"]) == JOB_KW["steps"]
    assert len(history["parts"]) == JOB_KW["steps"]
    assert JOB_KW["steps"] != 20, "a [-20:] slice would be invisible at 20 steps"
    for value in history["loss"]:
        assert isinstance(value, float)
    for part in history["parts"]:
        assert set(part) == PART_KEYS, part
        assert all(isinstance(v, float) for v in part.values())
    # The scalar convergence number is derived from the SAME curve.
    assert record["loss_last20"] == pytest.approx(
        float(np.mean(history["loss"][-20:])))


def test_the_history_is_the_one_training_returned_and_not_a_summary_of_it(
    tmp_path, small_buffer, monkeypatch
):
    """Exact equality against a history the test built: 23 distinct losses
    and 23 distinct `parts` rows, longer than 20 (so a `[-20:]` slice
    shortens it), not equal to `steps` (so a `[:steps]` slice shortens it),
    with a NaN in one row so the non-finite policy is exercised on a value
    FOUR path segments deep -- `history.parts.<k>.kl_dyn` -- through the
    file and back."""
    real_train = study.train_world_model
    n = 23
    assert n not in PAIRWISE_DISTINCT_PARAMETERS.values() and n > 20
    losses = [100.0 + i for i in range(n)]
    nan_row = 7
    parts = [
        {"embedding": 1.0 + i, "reward": 2.0 + i, "continue": 3.0 + i,
         "kl_dyn": float("nan") if i == nan_row else 4.0 + i,
         "kl_rep": 5.0 + i}
        for i in range(n)
    ]

    def doctored(cfg, buffer, out_dir, log_every=100):
        history = real_train(cfg, buffer, out_dir=out_dir, log_every=log_every)
        history["loss"] = list(losses)
        history["parts"] = [dict(p) for p in parts]
        return history

    monkeypatch.setattr(study, "train_world_model", doctored)
    result = run_job(JOB, small_buffer, tmp_path, **JOB_KW)

    assert result["history"]["loss"] == losses
    assert len(result["history"]["parts"]) == n
    for i, (got, want) in enumerate(zip(result["history"]["parts"], parts)):
        assert set(got) == set(want) == PART_KEYS
        for key in PART_KEYS:
            if i == nan_row and key == "kl_dyn":
                assert math.isnan(got[key])
            else:
                assert got[key] == want[key]

    # Through the file and back: the NaN is a `null` plus a four-segment
    # dotted path, and `load_record` restores a float NaN, not 0.0 or None.
    written = to_json_record(result)
    dotted = f"history.parts.{nan_row}.kl_dyn"
    assert written["history"]["parts"][nan_row]["kl_dyn"] is None
    assert written[NONFINITE_KEY][dotted] == "nan"
    restored = load_record(job_record_path(tmp_path, JOB))
    assert math.isnan(restored["history"]["parts"][nan_row]["kl_dyn"])
    assert restored["history"]["parts"][nan_row]["kl_rep"] == 5.0 + nan_row
    assert restored["history"]["loss"] == losses


def test_a_nan_inside_parts_round_trips_through_the_nonfinite_map(tmp_path):
    """The dotted-path scheme at unit scale, on the record's new shape:
    `parts` is a LIST of dicts, so the path carries an integer segment
    (`history.parts.1.kl_dyn`), and `load_record` must index the list rather
    than look the string "1" up in a dict."""
    original = {
        "history": {
            "loss": [0.5, float("inf")],
            "parts": [{"kl_dyn": 0.1}, {"kl_dyn": float("nan"), "kl_rep": 0.2}],
        },
    }
    written = write_record(tmp_path / "r.json", original)
    assert written["history"]["parts"][1]["kl_dyn"] is None
    assert written["history"]["parts"][1]["kl_rep"] == 0.2
    assert written["history"]["loss"][1] is None
    assert written[NONFINITE_KEY] == {
        "history.loss.1": "inf",
        "history.parts.1.kl_dyn": "nan",
    }
    text = (tmp_path / "r.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert strict_loads(text) == written

    back = load_record(tmp_path / "r.json")
    assert math.isnan(back["history"]["parts"][1]["kl_dyn"])
    assert back["history"]["parts"][1]["kl_rep"] == 0.2
    assert back["history"]["parts"][0]["kl_dyn"] == 0.1
    assert back["history"]["loss"][1] == float("inf")
    # The caller's dict still holds the real NaN; only the copy is lossy.
    assert math.isnan(original["history"]["parts"][1]["kl_dyn"])
```

> Why the device test pins `get_device` instead of reading a tensor's device: the
> existing `test_the_jobs_device_reaches_the_training_config_and_the_evaluation`
> explains that `get_device(prefer="cuda")` already IS cpu on this Mac, so any guard
> that compares against the machine's real answer passes on the box for the wrong
> reason. Pinning makes the request and the answer differ on every machine.
>
> Why `n = 23` in the doctored history: it must exceed 20 (a `[-20:]` slice shortens
> it), differ from `steps` (a `[:steps]` slice shortens it), and collide with none of
> `PAIRWISE_DISTINCT_PARAMETERS` — the file's own guard against fixture coincidence.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_study.py -q -k "sha or where_the_cell_ran or encoder_params or history or nan_inside_parts"`
Expected: **13 failed, 2 passed**. The failures are `KeyError: 'git_sha'`, `KeyError: 'device'` (1), `KeyError: 'encoder_params'` (3, one per arm), `KeyError: 'history'` (2), `AttributeError: module 'mbfps.eval.study' has no attribute 'subprocess'` (5: the cwd test and the four failure modes) and `... has no attribute '_trainable_parameters'` (1). The two passes are `test_the_record_reports_the_training_history_it_was_given` (an existing test matched by `-k history`) and `test_a_nan_inside_parts_round_trips_through_the_nonfinite_map` — the latter is green before Step 5 BY DESIGN: it guards the existing list-index branches of `_sanitise`/`load_record` that the new `history` field now depends on, and its red is the Step 7 mutation that removes those branches, not the missing key.

- [ ] **Step 3: Write the failing tests for the driver (`tests/eval/test_run_study.py`)**

Four edits. First, the two literals at lines 272–286 become:

```python
# tests/eval/test_run_study.py  (replace lines 272-286)
EXPECTED_RECORD_KEYS = frozenset({
    "arm", "seed", "steps", "seq_len", "context", "horizon", "split_seed",
    "seconds", "steps_per_second", "kl_rate_above_free_bits", "episodes",
    "probe", "position", "angle", "filtering", "reward", "curves",
    "git_sha", "device", "encoder_params", "history",
    "nonfinite",
})
"""Spelled out, `NONFINITE_KEY` included, so a RENAME is caught as well."""

M3C_RECORD_KEYS = frozenset({"git_sha", "device", "encoder_params", "history"})
"""The four provenance keys the M3c re-run added, as their own literal.

REQUIRED, NOT OPTIONAL. `runs/m3_study_v2` is a fresh directory and the M3b
cells are not reused (spec 2.4), so no record the driver will ever resume
off legitimately lacks them -- and requiring them is what makes "one code
state produced all nine cells" a field every finished cell must carry rather
than one some of them happen to. A second literal beside
`EXPECTED_RECORD_KEYS` rather than a slice of it, so that the four can be
named in a parametrisation without deriving the cases from the collection
under test; `test_the_m3c_keys_are_in_both_literals` ties the two together.
"""

OPTIONAL_RECORD_KEYS = frozenset({"kl_dyn_max", "loss_last20"})
"""The top-level keys `run_job` writes that a record is allowed to lack.

`run_job` writes twenty-four top-level keys; twenty-two of them are required.
These two are training diagnostics printed in the log, and a record without
them is still a finished cell. Naming them here is what lets the drift guard
below be an EQUALITY against a real record instead of a subset test.
"""
```

Second, `_record_body` (line 325) gains the four keys. Insert between `"loss_last20": 0.75,` and `"episodes": {...}`:

```python
        "kl_dyn_max": 1.25,
        "loss_last20": 0.75,
        # The M3c provenance fields. Synthetic values no implementation would
        # produce, in the spirit of CLI_DEVICE: nothing here is compared
        # against a real `git rev-parse` or a real parameter count (that is
        # `tests/eval/test_study.py`'s job); the driver only needs them PRESENT.
        "git_sha": "0123456789abcdef0123456789abcdef01234567",
        "device": "device-from-the-record",
        "encoder_params": 4242,
        "history": {
            "loss": [0.9, 0.8],
            "parts": [
                {"embedding": 0.5, "reward": 0.1, "continue": 0.2,
                 "kl_dyn": 0.05, "kl_rep": 0.05},
                {"embedding": 0.4, "reward": 0.1, "continue": 0.2,
                 "kl_dyn": 0.05, "kl_rep": 0.05},
            ],
        },
        "episodes": {"train": ["ep_a.npz", "ep_b.npz"], "val": ["ep_c.npz"]},
```

Third, two counts in existing docstrings go stale and are corrected, because a docstring that says "eighteen" beside a set of twenty-two is the kind of thing that stops the next reader trusting either. In `test_a_record_missing_any_field_is_treated_as_pending` (line 893–894): `# incomplete would pass all eighteen parametrisations above.` becomes `# incomplete would pass all twenty-two parametrisations above.` In `test_a_record_holding_exactly_the_required_keys_is_complete`'s docstring (lines 903–907), replace

```
    `_record_body` always writes all twenty top-level keys and a proper-subset
    relation holds for every fixture in this file. The one record shape that
    tells `<` from `<=` is the one with exactly the eighteen required keys and
    neither optional diagnostic -- a finished cell that legitimately lacks
```

with

```
    `_record_body` always writes all twenty-four top-level keys and a
    proper-subset relation holds for every fixture in this file. The one
    record shape that tells `<` from `<=` is the one with exactly the
    twenty-two required keys and neither optional diagnostic -- a finished
    cell that legitimately lacks
```

Fourth, insert the two new tests immediately BEFORE `def test_a_record_holding_exactly_the_required_keys_is_complete(tmp_path):` (line 900):

```python
def test_the_m3c_keys_are_in_both_literals():
    """The four are a slice of `EXPECTED_RECORD_KEYS` AND of the driver's set.

    `M3C_RECORD_KEYS` is what the parametrisation below reads. If it drifted
    from `EXPECTED_RECORD_KEYS` -- a key renamed in one literal and not the
    other -- the parametrised guard would be exercising a key the driver was
    never asked to require, and pass because deleting an absent key from a
    record changes nothing. Pinned against the driver's own set too, so the
    four cannot quietly become optional.
    """
    assert M3C_RECORD_KEYS == {"git_sha", "device", "encoder_params", "history"}
    assert M3C_RECORD_KEYS <= EXPECTED_RECORD_KEYS
    assert M3C_RECORD_KEYS <= run_study.REQUIRED_RECORD_KEYS
    assert not (M3C_RECORD_KEYS & OPTIONAL_RECORD_KEYS)


@pytest.mark.parametrize("missing", sorted(M3C_RECORD_KEYS))
def test_a_record_missing_a_provenance_field_is_treated_as_pending(
    tmp_path, missing
):
    """Each of the four M3c keys, ALONE. A record from the M3b code state --
    every field the gate reads, none of the provenance -- is exactly what a
    stale checkout on the box would write, and it must be re-run, not
    reported: the whole point of `git_sha` is that a record without it
    cannot be shown to come from the same code as the other eight.

    Parametrised over the LITERAL `M3C_RECORD_KEYS`, never over the driver's
    set: derive the cases from `REQUIRED_RECORD_KEYS` and dropping `history`
    from it deletes the case that would have caught the drop.
    """
    assert missing in _record_body(INCOMPLETE_JOB), (
        f"the fixture record never carried {missing!r}, so deleting it is a "
        "no-op and this case cannot fail")
    path = job_record_path(tmp_path, INCOMPLETE_JOB)
    damaged = _sanitised(INCOMPLETE_JOB)
    del damaged[missing]
    path.write_text(json.dumps(damaged))
    assert not run_study.record_is_complete(path, INCOMPLETE_JOB)
    assert INCOMPLETE_JOB in run_study.pending_jobs(tmp_path, ARMS, SEEDS)

    path.write_text(json.dumps(_sanitised(INCOMPLETE_JOB)))
    assert run_study.record_is_complete(path, INCOMPLETE_JOB)


```

> `test_a_record_missing_any_field_is_treated_as_pending` is already parametrised over
> `sorted(EXPECTED_RECORD_KEYS)`, so it gains four cases from the first edit alone.
> The second parametrisation over `M3C_RECORD_KEYS` is deliberately redundant with
> it: the L7 rule is that the keys under test are named in a literal of their own,
> and `test_the_m3c_keys_are_in_both_literals` is what stops that literal drifting
> from the set the first test reads.

- [ ] **Step 4: Run to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_run_study.py -q -k "hand_written_set or m3c or provenance or exactly_the_required or missing_any_field"`
Expected: **10 failed, 19 passed**. `test_required_record_keys_is_exactly_this_hand_written_set` fails with `Extra items in the right set: 'git_sha' 'device' 'history' 'encoder_params'`; `test_the_m3c_keys_are_in_both_literals` fails on `M3C_RECORD_KEYS <= run_study.REQUIRED_RECORD_KEYS`; the four `missing_any_field[...]` cases and the four `provenance_field[...]` cases for `device`, `encoder_params`, `git_sha`, `history` each fail with `assert not True` on `record_is_complete` — the driver still calls a record without the key finished, which is the defect.

- [ ] **Step 5: Implement the record (`src/mbfps/eval/study.py`)**

Add `subprocess` to the imports (line 23–29 block):

```python
import json
import math
import os
import subprocess
import time
```

Insert the two helpers and the constant between `load_record` (ends at line 314) and `_probe_summary` (line 317):

```python
UNKNOWN_GIT_SHA = "unknown"
"""What `_git_sha` reports when git cannot answer. Never an exception."""


def _git_sha() -> str:
    """The commit the code that is running was checked out from, or "unknown".

    Asked of the directory THIS FILE lives in, not of the process's working
    directory. The record exists to say "one code state produced all nine
    cells", and the code state is wherever `mbfps` was imported from -- the
    editable install on the laptop, or a checkout on the rented box. A run
    launched from somewhere else (`cd /scratch && python .../run_study.py`)
    must not report the sha of /scratch.

    NEVER RAISES. A rented box without git installed, a checkout unpacked from
    a tarball with no `.git`, a `git` that hangs on a stale lock: every one of
    those is a provenance gap, not a reason to lose a 1.5-hour cell. The
    provenance field says "unknown" and the cell is still written. Only the
    two failures a subprocess call can actually produce are caught -- `OSError`
    (no such executable, permissions) and `subprocess.SubprocessError`
    (`TimeoutExpired`) -- so a programming error in this function is still a
    traceback and not a quiet "unknown" in all nine records.

    The sha alone does not say the tree was clean. That is deliberate: this is
    a label for grouping records by code state, and a `-dirty` suffix would
    make two cells from one uncommitted tree look like two code states.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_GIT_SHA
    if completed.returncode != 0:
        return UNKNOWN_GIT_SHA
    # `or`: a zero status with nothing on stdout is not a sha either. Written
    # as a second guard rather than folded into the one above so each half
    # can be mutation-tested alone.
    return completed.stdout.strip() or UNKNOWN_GIT_SHA


def _trainable_parameters(module: torch.nn.Module) -> int:
    """How many parameters `module` trains: `requires_grad` ones only.

    Every study arm's encoder is a `BottleneckEncoder` whose parameters are all
    trainable, so on the study's own models the `requires_grad` filter is a
    numerical no-op -- which is exactly why it is a separate function with its
    own test: the day an encoder carries a frozen sub-module, `encoder_params`
    must still report what the arm TRAINS, because that is the asymmetry the
    spec records it for (1,056 for `pixel_ae` against 12,320 for the ViT arms).
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
```

Extend `run_job`'s docstring: after the paragraph ending `mismatch worth ~25 map units of position error.` (line 372) and before the closing `"""`, add:

```python
    FOUR PROVENANCE FIELDS, added for the M3c re-run and each answering a
    question the M3b write-up had to reconstruct after the fact:

      * `git_sha`        -- "one code state produced all nine cells" is a
                            field to compare, not an inference from mtimes;
      * `device`         -- where the cell actually RAN (`str(torch_device)`),
                            which on a box without CUDA is "cpu" however the
                            command line spelled it. The request string is
                            already in the log; the record holds the answer;
      * `encoder_params` -- the one place the arms are not byte-identical:
                            `pixel_ae`'s bottleneck is `Linear(32 -> 32)`,
                            the ViT arms' `Linear(384 -> 32)`. Recorded so it
                            is visible in every record rather than hidden;
      * `history`        -- the FULL per-step `loss` and `parts`, so the next
                            "the pixel arm's KL never cleared the floor" is a
                            measurement read off nine files instead of a
                            reconstruction. ~3 MB per record at 20,000 steps
                            with `indent=2`; the aggregation reads the scalar
                            fields and never touches it, and `_sanitise` maps
                            a non-finite loss at step k to `history.loss.k`
                            like any other nested field.
```

After `model.eval()` (line 399), count the encoder:

```python
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    # Counted on the EVALUATED model's encoder, after the load: the same
    # object every number below is measured on.
    encoder_params = _trainable_parameters(model.encoder)
```

And in the record dict (lines 431–462), add the three scalars after `"split_seed"` and the history after `"loss_last20"`:

```python
    record = {
        "arm": job.arm,
        "seed": job.seed,
        "steps": steps,
        "seq_len": seq_len,
        "context": context,
        "horizon": horizon,
        "split_seed": SPLIT_SEED,
        "git_sha": _git_sha(),
        "device": str(torch_device),
        "encoder_params": int(encoder_params),
        "seconds": float(time.perf_counter() - started),
        "steps_per_second": float(history["steps"] / history["seconds"]),
        "kl_rate_above_free_bits": float(history["kl_rate_above_free_bits"]),
        "kl_dyn_max": float(history["kl_dyn_max"]),
        "loss_last20": float(np.mean(history["loss"][-20:])),
        # The whole curve, every step, both the total and its five terms.
        # Coerced element by element so the record holds plain floats whatever
        # `train_world_model` appended (a non-finite one survives the coercion
        # and is written through the `nonfinite` map, never dropped).
        "history": {
            "loss": [float(v) for v in history["loss"]],
            "parts": [
                {str(k): float(v) for k, v in part.items()}
                for part in history["parts"]
            ],
        },
        # The held-out episodes by name, so "all nine cells were scored on the
        # same episodes" is checkable after the fact rather than assumed.
        "episodes": {
```

The rest of the dict (`episodes`, `probe`, `position`, `angle`, `filtering`, `reward`, `curves`) is unchanged.

- [ ] **Step 6: Run to verify the record tests pass**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_study.py -q`
Expected: PASS — every test in the file, 14 more than Step 2 collected (13 newly green plus the one that was green by design). On the tree as Task 5 left it that is 62 + 14 = **76** passed; `-rs` must list NO skip from `test_the_record_carries_the_sha_git_reports_for_the_code_that_ran` (a skip there means the test could not run git and pinned nothing).

- [ ] **Step 7: Implement the driver (`scripts/run_study.py`)**

Replace `REQUIRED_RECORD_KEYS` and the first paragraph of its docstring (lines 97–108) with:

```python
REQUIRED_RECORD_KEYS = frozenset({
    "arm", "seed", "steps", "seq_len", "context", "horizon", "split_seed",
    "seconds", "steps_per_second", "kl_rate_above_free_bits", "episodes",
    "probe", "position", "angle", "filtering", "reward", "curves",
    "git_sha", "device", "encoder_params", "history",
    NONFINITE_KEY,
})
"""Every top-level key `run_job` writes. A record missing one is not done.

This is the completeness test that "the file parses" is not. `NONFINITE_KEY` is
in the set on purpose: only `write_record` adds it, so a hand-made or
half-converted JSON blob at a record path is treated as pending rather than
mistaken for a finished 8.3-hour cell.

THE FOUR M3c PROVENANCE KEYS ARE REQUIRED, NOT OPTIONAL. `git_sha`, `device`,
`encoder_params` and `history` were added so that "one code state produced
all nine cells" is a field rather than a reconstruction from file mtimes. A
record without them is what a stale checkout on the box writes, and it is
precisely the record that cannot be shown to belong with the other eight --
so it is re-run. Nothing legitimate lacks them: the M3c study runs into a
fresh `--out`, and the M3b records are not reused (spec 2.4).
```

The remaining paragraphs of the docstring ("DRIFT IN EITHER DIRECTION ..." and "The guards are deliberately not derived ...") stay as they are.

- [ ] **Step 8: Run to verify everything passes**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_study.py tests/eval/test_run_study.py -q`
Expected: PASS, **23 more** than the two files collected before this task (14 in `test_study.py`; in `test_run_study.py` the four new `missing_any_field` cases, `test_the_m3c_keys_are_in_both_literals`, and the four `provenance_field` cases). On the tree as Task 5 left it: **227 passed** (was 62 + 142 = 204). The drift guard `test_a_real_record_from_run_job_is_recognised_as_complete[<arm>]` runs the real `run_job` for every arm and asserts `REQUIRED_RECORD_KEYS == written - OPTIONAL_RECORD_KEYS`, so this step is also where "the driver demands exactly what `run_job` writes" is proved for `pixel_ae`.

Then the whole suite: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests -q`
Expected: PASS, 23 more than the count Task 5 left (23 more than Task 5's full-suite count), 3 pre-existing skips in `tests/eval/test_diagnostics.py` (they need the shipped M3b artefacts and MPS), exit 0.

- [ ] **Step 9: Mutation-test**

The harness for this task has a fourth self-check on top of the three standard ones, because `git_sha` is a fact about the checkout the code runs from:

```bash
# 1. Export to a scratch copy and run against IT, not the editable install.
S=/tmp/task6-mut && rm -rf $S && mkdir -p $S && git archive HEAD | tar -x -C $S
# 4. THE EXTRA CHECK. A bare `git archive` export has no repository, so
#    `_git_sha()` returns "unknown" there AND the test's own git call fails,
#    so test_the_record_carries_the_sha_... SKIPS instead of running. A skip
#    is not a catch. Give the export a repository of its own:
( cd $S && git init -q && git add -A && git -c user.email=m@m -c user.name=m commit -qm export )
export PYTHONPATH=$S/src PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -c "import mbfps.eval.study as s; print(s.__file__)"   # must print $S/src/...
# 3. No stale bytecode can survive a same-length mutation.
find $S -name __pycache__ -exec rm -rf {} +
# Confirm the sha test RUNS here (0 skipped in the output):
.venv/bin/python -m pytest $S/tests/eval/test_study.py -q -rs -k sha_git_reports
# 2. Prove the harness with a known-fatal mutation FIRST: delete the line
#    `"git_sha": _git_sha(),` from $S/src/mbfps/eval/study.py and run
#    `-k sha_git_reports` -- it must FAIL. Restore with `git -C $S checkout -- .`
#    and clear __pycache__ again before every row below.
```

Every row below was run against this task's implementation with that harness; every one was caught.

| mutation | must be caught by |
|---|---|
| `"git_sha": _git_sha()` → `"git_sha": "unknown"` | `test_the_record_carries_the_sha_git_reports_for_the_code_that_ran` |
| `cwd=Path(__file__).resolve().parent` → `cwd=None` (the process's cwd) | `test_git_sha_is_asked_of_the_code_not_the_cwd` |
| the `except (OSError, subprocess.SubprocessError)` branch `raise`s instead of returning | `test_git_sha_is_unknown_rather_than_an_exception_when_git_cannot_answer[no_git_executable]` AND `[git_hangs]` (both, one per exception class) |
| the `if completed.returncode != 0` guard deleted | `...[not_a_repository]` — stdout carries `"abc"` on purpose so the `or` fallback cannot cover for it |
| `.strip() or UNKNOWN_GIT_SHA` → `.strip()` | `...[empty_stdout]` |
| `.strip()` dropped (`completed.stdout or UNKNOWN_GIT_SHA`) | `test_the_record_carries_the_sha_git_reports_for_the_code_that_ran` (trailing newline) |
| `"device": str(torch_device)` → `"device": device` (the request string) | `test_the_record_says_where_the_cell_ran_not_what_was_asked_for` |
| `_trainable_parameters(model.encoder)` → `_trainable_parameters(model)` | `test_encoder_params_counts_the_trainable_encoder_parameters_of_each_arm` (all three arms) |
| `if p.requires_grad` dropped from `_trainable_parameters` | `test_encoder_params_excludes_frozen_parameters` |
| `"encoder_params": int(encoder_params)` → `12320` | `test_encoder_params_counts_the_trainable_encoder_parameters_of_each_arm[pixel_ae]` (1,056 ≠ 12,320) |
| `"loss": [... history["loss"]]` → `history["loss"][-20:]` | `test_the_history_is_the_one_training_returned_and_not_a_summary_of_it` (23 entries) — NOT the fixture-length test, whose 5 steps are inside any 20-window |
| `"loss": [... history["loss"]]` → `history["loss"][:steps]` | `test_the_history_is_the_one_training_returned_and_not_a_summary_of_it` (23 ≠ 5) |
| `"parts"` emitted empty (`history["parts"][:0]`) | `test_the_record_carries_the_full_history_at_the_jobs_own_length` |
| the `"history"` key renamed / the dict replaced by the bare loss list | `test_the_record_carries_the_full_history_at_the_jobs_own_length` (`set(history) == HISTORY_KEYS`) |
| one term dropped from each `parts` row (`if k != "kl_rep"`) | `test_the_record_carries_the_full_history_at_the_jobs_own_length` (`set(part) == PART_KEYS`) |
| `load_record` indexes every container as a dict (`container = container[segment]`) | `test_a_nan_inside_parts_round_trips_through_the_nonfinite_map` (`TypeError: list indices must be integers`) |
| `_sanitise`'s list branch drops the index from the path (`f"{path}.{i}"` → `path`) | `test_a_nan_inside_parts_round_trips_through_the_nonfinite_map` (map keys differ) |
| `"history"` removed from `REQUIRED_RECORD_KEYS` | `test_required_record_keys_is_exactly_this_hand_written_set` AND `test_a_record_missing_a_provenance_field_is_treated_as_pending[history]` |
| `"git_sha"` removed from `REQUIRED_RECORD_KEYS` | the same pair, `[git_sha]` |

Any mutation that survives is a missing test. Add it before committing.

- [ ] **Step 10: Commit**

```bash
git add src/mbfps/eval/study.py scripts/run_study.py tests/eval/test_study.py tests/eval/test_run_study.py
git commit -m "feat: every study record carries its git sha, its device, its encoder size and the full loss history

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: The spike: 2,000 steps of `pixel_ae`, and the three checks that license the study

Spec §3. Before the nine-cell run buys 13.5 h of compute, one 2,000-step `pixel_ae` run at seed 0 must show all three of: an informative embedding target at step 0 (loss in the feature-arm band 0.1–1.0, against the retired pixel arm's 0.0008), a dynamics prior that actually trains (`kl_rate_above_free_bits > 0.5`), and cached features that carry position at all (held-out linear-probe R² > 0). M3b's pixel arm failed the first and second silently, 9 h per cell, three cells deep, and the failure was reconstructed from checkpoint diffs afterwards because no run kept its per-step history. This spike keeps it — `history["loss"]` and `history["parts"]` in full, M3b write-up open item 6 — and turns the three criteria into code with a PASS/FAIL verdict and an exit status, so the decision to run the study is a measurement, not a judgement call over a log.

This is a run-and-record task like M3b Task 7, with one small script in front of it. The script is tested the way `scripts/run_study.py` is: loaded by path with `importlib`, its pure verdict functions driven from fabricated histories, and its one end-to-end path run on the tiny `small_buffer` fixture on CPU.

**Files:**
- Create: `scripts/spike_pixel_ae.py`
- Create: `tests/eval/test_spike_pixel_ae_script.py`
- Modify: this plan — the `## Task 7 results` section at the end (create it; template in Step 9)

> Read before writing, so the pieces this script reuses are reused and not re-implemented:
> `src/mbfps/training/world_model.py` lines 102–181 (`train_world_model`: what `history` carries; the side-by-side calls it at `steps=1`, so step 0's batch is produced by that code, not by a copy of it);
> `src/mbfps/eval/study.py` lines 44–52 (`SPLIT_SEED`) and 245–286 (`write_record`: atomic, strict-JSON, non-finite-safe — `history["parts"]` can carry a NaN and this is the only writer that survives it);
> `src/mbfps/eval/probe.py` lines 35–50 (`probe_targets`), 53–95 (`RIDGES`, `fit_probe` with ridge selection), 138–149 (`probe_r2`), and 298–383 (`fit_probes`, whose 16-fit / 4-select episode split this task mirrors);
> `src/mbfps/data/loader.py` lines 40–51 (`feature_suffix`) and 54–99 (`SequenceLoader`'s constructor);
> `src/mbfps/data/split.py` lines 35–60 (`episode_split`);
> `tests/eval/conftest.py` lines 11–48 (`small_buffer`: six 40-step episodes with `(41, 64, 384)` caches for `dinov2` and `random_vit` — this task adds the `(41, 64, 32)` `pixel_ae` cache in a local fixture).

**Which probe helper check 3 reuses, and why neither `fit_probes` nor `gather_probe_data`.** Both take a `model` and probe what the *model* produces — `gather_probe_data` runs `model.encoder`, `model.rssm.observe` and `model.heads`, and `fit_probes` fits on its `"embedding"` (the predicted embedding) and `"latent"` arrays. Check 3 asks about the **cached features themselves**, before any trainable layer, which is what §3.2a's reference numbers (DINOv2 0.422, `random_vit` 0.369) were measured on. So the script uses the array-level functions from the same module — `probe_targets` for the `(T+1, 4)` targets, `fit_probe(x_fit, y_fit, x_select, y_select)` for ridge selection on held-out episodes, `probe_r2` to score — on the `(T+1, 64, 32)` `pixel_ae` cache flattened to `(T+1, 2048)`. Flattening is the honest input: spec §2.1 says the `(64, 32)` partition "invents nothing — it is a reshape of a vector whose 2048 dimensions have no spatial meaning", so a patch-mean (§3.2a's reduction for the ViT caches, where 64 rows are 64 real patches) would average unrelated dimensions here. The split mirrors `fit_probes(limit=20, select_episodes=4)`: weights on the first 16 training episodes, ridge selected on the next 4, scored on the first 20 validation episodes of the study's own `episode_split(seed=SPLIT_SEED)`.

**M8 (nothing requires the evaluated model to be the loaded one) does not apply here in its usual form** — the spike loads no checkpoint; checks 1 and 2 are read off the training history and check 3 off the cache. Its analogue is "the number printed is not the number judged", and it is guarded twice: the side-by-side helper is asserted equal to `train_world_model`'s own step-0 value on every arm, and the record on disk is asserted to carry each check's `value` equal to the field it was computed from.

**Interfaces:**
- Consumes: `train_world_model(cfg, buffer, out_dir) -> dict` and `WorldModel(cfg)` from `mbfps.training.world_model`; `get_config(arm, **overrides)` and `ARMS` from `mbfps.utils.config`; `encoder_backbone(cfg)` from `mbfps.models.encoders`; `BACKBONE_GEOMETRY` from `mbfps.data.features`; `feature_suffix(backbone)` and `SequenceLoader` from `mbfps.data.loader`; `episode_split`, `VAL_FRACTION` from `mbfps.data.split`; `SPLIT_SEED`, `write_record`, `load_record` from `mbfps.eval.study`; `fit_probe`, `probe_r2`, `probe_targets` from `mbfps.eval.probe`; `load_episode`; `ReplayBuffer`; `get_device`, `to_device`; `seed_everything`; `KL_FREE_BITS` from `mbfps.models.rssm`.
- Produces, all in `scripts/spike_pixel_ae.py`:
  - constants `SPIKE_ARM = "pixel_ae"`, `SPIKE_SEED = 0`, `SPIKE_STEPS = 2_000`, `EMBEDDING_LOSS_BAND = (0.1, 1.0)`, `MIN_KL_RATE = 0.5`, `MIN_PROBE_R2 = 0.0`, `PROBE_LIMIT = 20`, `PROBE_SELECT_EPISODES = 4`, `CHECK_KEYS = ("embedding_loss_step0", "kl_rate_above_free_bits", "probe_r2_cached_features")`, `RUN_STUDY`, `STOP_RECALIBRATE_FREE_BITS`, `STOP_BACKBONE_UNINFORMATIVE`, `EXIT_OK = 0`, `EXIT_SPIKE_FAILED = 10`
  - `check_embedding_loss(parts: list[dict]) -> dict`, `check_kl_rate(history: dict) -> dict`, `check_probe_r2(r2: float) -> dict` — each `{"value", ..., "passed"}`
  - `evaluate_checks(history: dict, feature_probe_r2: float) -> dict[str, dict]` keyed by `CHECK_KEYS`
  - `all_passed(checks: dict) -> bool`, `decision(checks: dict) -> str`, `exit_status(checks: dict) -> int`
  - `step0_embedding_loss(cfg, buffer) -> float` — `train_world_model` itself, at `steps=1` and `out_dir=None`; returns `history["parts"][0]["embedding"]` (pre-flight 2026-09-12: the plan's original hand copy of the setup block was verbatim duplication; the user chose this form)
  - `load_cached_features(path: Path, backbone: str) -> np.ndarray` — `(T+1, n_patches * patch_dim)` float64, validated against `BACKBONE_GEOMETRY`
  - `cached_feature_probe(train_paths, val_paths, backbone, limit=PROBE_LIMIT, select_episodes=PROBE_SELECT_EPISODES) -> dict` with keys `"backbone"`, `"r2"`, `"ridge"`, `"selection_r2"`, `"n_fit_episodes"`, `"n_select_episodes"`, `"n_scored_episodes"`, `"n_scored_rows"`, `"n_columns"`
  - `spike_record_path(out_dir) -> Path`, `report(record: dict) -> str`, `git_sha() -> str`, `main(argv=None) -> int`

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_spike_pixel_ae_script.py
"""The spike that licenses the M3c study: three checks, one verdict, one exit status.

Spec section 3 says the nine-cell run happens only if a 2,000-step `pixel_ae`
run shows (1) an embedding loss at step 0 in the feature-arm band, (2) a
dynamics prior that trained on more than half its steps, and (3) cached
features a linear probe can read position from. The M3b pixel arm failed (1)
and (2) silently and cost 27 h before anyone looked. These tests are arranged
around the ways a verdict can be wrong without looking wrong: a check that
reads the last step instead of the first, a threshold that is inclusive where
the spec is strict, a check whose failure the others can mask, a probe scored
on the rows it was fit on, a side-by-side number that is not the number the
check judged, and an exit status that says 0 whatever the checks said.

The script is loaded by path, like `scripts/run_study.py` in its own tests:
`scripts/` is not a package.
"""

import importlib.util
import math
import re
from pathlib import Path

import numpy as np
import pytest

from mbfps.data.episode import load_episode
from mbfps.data.loader import feature_suffix
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.probe import probe_targets
from mbfps.eval.study import SPLIT_SEED, load_record
from mbfps.training.world_model import train_world_model
from mbfps.utils.config import ARMS, get_config

_SPEC = importlib.util.spec_from_file_location(
    "spike_pixel_ae_script",
    Path(__file__).resolve().parents[2] / "scripts" / "spike_pixel_ae.py",
)
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)

# The three study arms, AS A LITERAL. Parametrising over `ARMS` itself would
# let a fourth arm -- or a dropped one -- shrink this file's coverage without
# a failure; the equality assertion in the first test is what notices.
SPIKE_ARMS = ("pixel_ae", "frozen_ssl", "random_vit")

# The numbers the M3b study measured on its dead pixel arm, so the failing
# cases below are the failure that actually happened and not round numbers.
M3B_CNN_EMBEDDING_LOSS_STEP0 = 0.0008    # M3c design section 1, table 1
M3B_CNN_KL_RATE = 0.0009                 # 18 of 20,000 steps above the floor
M3B_CNN_KL_DYN_MAX = 0.800               # cnn/s1's peak, ABOVE the floor
M3B_CNN_PROBE_R2 = -0.036                # Task 2 results: a constant predictor

TINY = dict(steps=3, seq_len=6, batch_size=2, device="cpu")


def _history(embedding_step0=0.35, embedding_last=0.20, kl_rate=0.9,
             kl_dyn_max=0.7) -> dict:
    """A fabricated `train_world_model` history with the fields the checks read.

    Step 0 and the last step carry DIFFERENT embedding losses on purpose: a
    check that read `parts[-1]`, or the mean, would agree with one that read
    `parts[0]` on a history where they coincide.
    """
    return {
        "parts": [{"embedding": embedding_step0, "kl_dyn": 0.1},
                  {"embedding": embedding_last, "kl_dyn": 0.3}],
        "kl_rate_above_free_bits": kl_rate,
        "kl_dyn_max": kl_dyn_max,
        "loss": [1.0, 0.9],
    }


def _checks(embedding=0.35, kl_rate=0.9, r2=0.3) -> dict:
    return script.evaluate_checks(_history(embedding_step0=embedding,
                                           kl_rate=kl_rate), r2)


def _passed(checks: dict) -> dict:
    return {key: checks[key]["passed"] for key in script.CHECK_KEYS}


@pytest.fixture
def spike_buffer(small_buffer):
    """`small_buffer` plus a `pixel_ae` cache at the registry's (64, 32) rows.

    `tests/eval/conftest.py` writes the two ViT caches at (41, 64, 384). The
    loader validates rows against `BACKBONE_GEOMETRY`, so `pixel_ae` needs its
    own file at its own geometry or the arm cannot even build a batch.
    """
    rng = np.random.default_rng(1)
    for path in small_buffer.episode_paths():
        np.save(path.with_suffix(feature_suffix("pixel_ae")),
                rng.random((41, 64, 32)).astype(np.float16))
    return small_buffer


def _plant_position(path: Path) -> None:
    """Overwrite `path`'s pixel_ae cache so its first four columns ARE the targets.

    Everything else is zero, so the only way a probe on this cache scores
    R^2 > 0.9 is by reading THIS file and aligning its rows with the frames.
    """
    episode = load_episode(path)
    targets = probe_targets(episode.privileged, episode.privileged_keys)
    features = np.zeros((targets.shape[0], 64, 32), dtype=np.float16)
    features[:, 0, :4] = targets
    np.save(path.with_suffix(feature_suffix("pixel_ae")), features)


# --------------------------------------------------------------------------
# what the spike is
# --------------------------------------------------------------------------

def test_the_spike_is_pinned_to_pixel_ae_seed_0_and_2000_steps():
    """Spec section 3: ONE arm, ONE seed, 2,000 steps. The arm and seed are
    constants, not flags -- a `--arm frozen_ssl` spike would pass its own
    checks trivially and license nothing about the arm the study is for."""
    assert script.SPIKE_ARM == "pixel_ae"
    assert script.SPIKE_SEED == 0
    assert script.SPIKE_STEPS == 2_000
    assert set(SPIKE_ARMS) == set(ARMS), (
        "the study's arms changed; the side-by-side and the step-0 test below "
        "must cover every one of them")
    parser = script._parser()
    defaults = parser.parse_args([])
    assert defaults.steps == 2_000
    assert defaults.out == "runs/m3c_spike"
    for flag in ("--arm", "--seed"):
        with pytest.raises(SystemExit):
            parser.parse_args([flag, "0"])


# --------------------------------------------------------------------------
# check 1: embedding loss at step 0, in [0.1, 1.0]
# --------------------------------------------------------------------------

def test_embedding_check_flips_on_each_edge_alone():
    """Both edges of the band, each tested where the other cannot help."""
    assert script.check_embedding_loss([{"embedding": 0.35}])["passed"] is True
    # below the band: the M3b pixel arm's own number
    assert script.check_embedding_loss(
        [{"embedding": M3B_CNN_EMBEDDING_LOSS_STEP0}])["passed"] is False
    # above the band: a target so wide the RSSM would be fitting noise
    assert script.check_embedding_loss([{"embedding": 1.5}])["passed"] is False
    # the band is CLOSED at both ends ("in [0.1, 1.0]")
    assert script.check_embedding_loss([{"embedding": 0.1}])["passed"] is True
    assert script.check_embedding_loss([{"embedding": 1.0}])["passed"] is True
    assert script.check_embedding_loss([{"embedding": 0.0999}])["passed"] is False
    assert script.check_embedding_loss([{"embedding": 1.0001}])["passed"] is False


def test_embedding_check_reads_step_zero_not_the_last_step():
    """The failure section 1 documents is a target that is born collapsed. A
    check that reads the end of training would miss it: the loss can only
    fall from there, and 0.0008 -> 0.0003 is still 'in the band' on nothing."""
    born_dead = [{"embedding": M3B_CNN_EMBEDDING_LOSS_STEP0}, {"embedding": 0.5}]
    assert script.check_embedding_loss(born_dead)["passed"] is False
    born_alive = [{"embedding": 0.5}, {"embedding": M3B_CNN_EMBEDDING_LOSS_STEP0}]
    assert script.check_embedding_loss(born_alive)["passed"] is True
    assert script.check_embedding_loss(born_alive)["value"] == 0.5


def test_embedding_check_refuses_an_empty_history():
    with pytest.raises(ValueError, match="no steps"):
        script.check_embedding_loss([])


# --------------------------------------------------------------------------
# check 2: kl_rate_above_free_bits > 0.5
# --------------------------------------------------------------------------

def test_kl_check_flips_at_one_half():
    """Strict: `_kl_rate`'s own warning fires at `< 0.5`, and exactly one half
    is the prior having trained on precisely as many steps as it did not."""
    assert script.check_kl_rate(_history(kl_rate=0.51))["passed"] is True
    assert script.check_kl_rate(_history(kl_rate=0.5))["passed"] is False
    assert script.check_kl_rate(_history(kl_rate=0.49))["passed"] is False
    assert script.check_kl_rate(_history(kl_rate=M3B_CNN_KL_RATE))["passed"] is False


def test_kl_check_reads_the_rate_not_the_max():
    """`kl_dyn_max` is the number a max-based check would read, and it is the
    number that hid M3b's dead arm: cnn/s1 peaked at 0.800 -- four times the
    floor -- while clearing it on 16 of 20,000 steps."""
    history = _history(kl_rate=M3B_CNN_KL_RATE, kl_dyn_max=M3B_CNN_KL_DYN_MAX)
    assert history["kl_dyn_max"] > 0.5, "fixture: the max must be the misleading one"
    assert script.check_kl_rate(history)["passed"] is False
    assert script.check_kl_rate(history)["value"] == M3B_CNN_KL_RATE


# --------------------------------------------------------------------------
# check 3: held-out probe R^2 on the cached features > 0
# --------------------------------------------------------------------------

def test_probe_check_flips_at_zero():
    """Strict: R^2 of exactly 0 is the constant predictor at the mean, which is
    what the M3b pixel arm's probe collapsed to (-0.036 after ridge 1e7)."""
    assert script.check_probe_r2(0.01)["passed"] is True
    assert script.check_probe_r2(0.0)["passed"] is False
    assert script.check_probe_r2(M3B_CNN_PROBE_R2)["passed"] is False


def test_non_finite_values_fail_every_check():
    """NaN compares False against everything, and a check written as
    `not (value < low or value > high)` would pass it. Each check alone."""
    nan = float("nan")
    assert script.check_embedding_loss([{"embedding": nan}])["passed"] is False
    assert script.check_kl_rate(_history(kl_rate=nan))["passed"] is False
    assert script.check_probe_r2(nan)["passed"] is False
    assert script.all_passed(_checks(embedding=nan)) is False


# --------------------------------------------------------------------------
# the three together: independence, conjunction, decision, exit status
# --------------------------------------------------------------------------

def test_each_check_flips_independently_of_the_other_two():
    """Flip one input at a time; exactly that check must fail. A check that
    read another's field, or the verdict of another, would flip two."""
    assert _passed(_checks()) == {
        "embedding_loss_step0": True,
        "kl_rate_above_free_bits": True,
        "probe_r2_cached_features": True,
    }
    assert _passed(_checks(embedding=M3B_CNN_EMBEDDING_LOSS_STEP0)) == {
        "embedding_loss_step0": False,
        "kl_rate_above_free_bits": True,
        "probe_r2_cached_features": True,
    }
    assert _passed(_checks(kl_rate=M3B_CNN_KL_RATE)) == {
        "embedding_loss_step0": True,
        "kl_rate_above_free_bits": False,
        "probe_r2_cached_features": True,
    }
    assert _passed(_checks(r2=M3B_CNN_PROBE_R2)) == {
        "embedding_loss_step0": True,
        "kl_rate_above_free_bits": True,
        "probe_r2_cached_features": False,
    }


def test_all_passed_requires_all_three():
    """A conjunction, tested one failure at a time so `any` cannot pass."""
    assert script.all_passed(_checks()) is True
    assert script.all_passed(_checks(embedding=M3B_CNN_EMBEDDING_LOSS_STEP0)) is False
    assert script.all_passed(_checks(kl_rate=M3B_CNN_KL_RATE)) is False
    assert script.all_passed(_checks(r2=M3B_CNN_PROBE_R2)) is False


def test_all_passed_refuses_a_checks_dict_missing_a_check():
    checks = _checks()
    del checks["probe_r2_cached_features"]
    with pytest.raises(ValueError, match="probe_r2_cached_features"):
        script.all_passed(checks)


def test_decision_runs_the_study_only_when_all_three_pass():
    assert script.decision(_checks()) == script.RUN_STUDY


def test_decision_sends_a_lone_kl_failure_to_a_free_bits_recalibration():
    """Spec section 3: 'Only if the second fails does KL_FREE_BITS get
    revisited' -- the target is informative, the prior just did not clear the
    floor, so the floor is the thing to measure across all three arms."""
    assert script.decision(_checks(kl_rate=M3B_CNN_KL_RATE)) == \
        script.STOP_RECALIBRATE_FREE_BITS


def test_decision_calls_the_backbone_uninformative_on_check_1_or_3():
    """Each alone: an uninformative target (1) or features with no position
    in them (3) is a property of the backbone, and no floor can fix it."""
    assert script.decision(_checks(embedding=M3B_CNN_EMBEDDING_LOSS_STEP0)) == \
        script.STOP_BACKBONE_UNINFORMATIVE
    assert script.decision(_checks(r2=M3B_CNN_PROBE_R2)) == \
        script.STOP_BACKBONE_UNINFORMATIVE


def test_decision_backbone_verdict_dominates_a_kl_failure():
    """Checks 1 and 2 failing together is exactly the M3b pixel arm, and its
    lesson (section 1) is that lowering the floor cannot help a collapsed
    target. The recalibration route must not be offered."""
    both = _checks(embedding=M3B_CNN_EMBEDDING_LOSS_STEP0, kl_rate=M3B_CNN_KL_RATE)
    assert script.decision(both) == script.STOP_BACKBONE_UNINFORMATIVE


def test_exit_status_is_zero_iff_all_pass():
    assert script.exit_status(_checks()) == 0
    for failing in (_checks(embedding=M3B_CNN_EMBEDDING_LOSS_STEP0),
                    _checks(kl_rate=M3B_CNN_KL_RATE),
                    _checks(r2=M3B_CNN_PROBE_R2)):
        assert script.exit_status(failing) == 10
    assert script.EXIT_OK == 0 and script.EXIT_SPIKE_FAILED == 10


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

def _fake_record(checks: dict) -> dict:
    return {
        "arm": "pixel_ae", "seed": 0, "steps": 2000, "device": "cpu",
        "git_sha": "0123abcd", "steps_per_second": 3.9, "kl_dyn_max": 0.7,
        "embedding_loss_step0": {"pixel_ae": 0.35, "frozen_ssl": 0.3165,
                                 "random_vit": 0.4104},
        "probe": {"backbone": "pixel_ae", "r2": 0.3, "ridge": 1e3,
                  "selection_r2": 0.31, "n_fit_episodes": 16,
                  "n_select_episodes": 4, "n_scored_episodes": 20,
                  "n_scored_rows": 10520, "n_columns": 2048},
        "checks": checks,
        "decision": script.decision(checks),
    }


def test_report_marks_each_check_with_its_own_verdict():
    """Three PASS when all pass; exactly one FAIL, on the failing check's own
    line, when one fails. Captured AND asserted, per line."""
    verdicts = lambda text: re.findall(r"\b(PASS|FAIL)\b", text)  # noqa: E731

    text = script.report(_fake_record(_checks()))
    assert verdicts(text) == ["PASS", "PASS", "PASS"]
    assert script.RUN_STUDY in text
    for arm in SPIKE_ARMS:
        assert f"{arm}=" in text, "the side-by-side must name every arm"

    text = script.report(_fake_record(_checks(kl_rate=M3B_CNN_KL_RATE)))
    assert verdicts(text) == ["PASS", "FAIL", "PASS"]
    line = next(l for l in text.splitlines() if l.startswith("check 2"))
    assert "FAIL" in line
    assert script.STOP_RECALIBRATE_FREE_BITS in text


# --------------------------------------------------------------------------
# the side-by-side: the number printed is the number train_world_model judges
# --------------------------------------------------------------------------

def test_step0_embedding_loss_matches_train_world_model_step_zero_on_every_arm(
        spike_buffer):
    """The side-by-side prints the step-0 embedding loss for all three arms
    from one forward pass each. It is only a comparison if that pass is the
    SAME pass `train_world_model` scores at step 0 -- same seed, same split,
    same loader draw, same posterior sample -- so the helper is held to
    `history["parts"][0]["embedding"]` on every arm, not just the spike's.

    The three values are then required to be pairwise distinct: a helper that
    ignored `cfg` and built `SPIKE_ARM` three times would return one number
    thrice, and on a fixture where the arms happened to coincide the equality
    assertions above could not tell.
    """
    seen = {}
    for arm in SPIKE_ARMS:
        cfg = get_config(arm, seed=0, **TINY)
        expected = train_world_model(cfg, spike_buffer, out_dir=None)
        got = script.step0_embedding_loss(cfg, spike_buffer)
        assert got == pytest.approx(expected["parts"][0]["embedding"], rel=1e-6), arm
        seen[arm] = round(got, 6)
    assert len(set(seen.values())) == 3, (
        f"fixture coincidence: two arms share a step-0 embedding loss {seen}; "
        "the per-arm equality assertions cannot distinguish them")


def test_step0_embedding_loss_runs_one_step_saves_nothing_and_reads_step_zero(
        monkeypatch):
    """The equality test above cannot see three things, because on the TINY
    fixture they change nothing: a helper that forgot `steps=1` (20,000 steps
    per arm on the real run, three times), one that passed an `out_dir` (a
    checkpoint written under the spike directory), and one that read
    `parts[-1]` (equal to `parts[0]` at one step). A spy in place of
    `train_world_model` pins all three."""
    calls = []

    def spy(cfg, buffer, out_dir, log_every=100):
        calls.append((cfg, buffer, out_dir, log_every))
        return {"parts": [{"embedding": 0.31}, {"embedding": 0.17}]}

    monkeypatch.setattr(script, "train_world_model", spy)
    cfg = get_config("frozen_ssl", seed=0, **TINY)
    assert script.step0_embedding_loss(cfg, "the buffer") == 0.31
    (seen_cfg, seen_buffer, seen_out_dir, seen_log_every), = calls
    assert seen_cfg.train.steps == 1
    assert seen_cfg.arm == "frozen_ssl" and seen_cfg.train.seed == 0
    assert replace(seen_cfg.train, steps=cfg.train.steps) == cfg.train, (
        "every TrainConfig field but steps must reach train_world_model unchanged")
    assert seen_buffer == "the buffer"
    assert seen_out_dir is None
    assert seen_log_every == 0


# --------------------------------------------------------------------------
# check 3's probe on the cache itself
# --------------------------------------------------------------------------

def test_cached_feature_probe_recovers_position_planted_in_the_pixel_ae_cache(
        spike_buffer):
    """Position planted in the pixel_ae cache -- and ONLY there; the dinov2
    and random_vit caches in this fixture are noise at a different geometry --
    must come back through the probe. Reading any other cache raises on
    geometry; reading this one with misaligned rows scores far lower."""
    train, val = episode_split(spike_buffer.episode_paths(),
                               val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
    for path in train + val:
        _plant_position(path)
    result = script.cached_feature_probe(train, val, "pixel_ae")
    assert result["backbone"] == "pixel_ae"
    assert result["n_columns"] == 64 * 32
    assert result["r2"] > 0.9, result


def test_cached_feature_probe_scores_the_validation_episodes_not_the_fit_ones(
        spike_buffer):
    """Signal on the training episodes, noise on the held-out one: a probe
    scored on its own fit rows reads ~1.0 here; scored held out it reads
    nothing. 'Held-out' is the word in the spec's check."""
    train, val = episode_split(spike_buffer.episode_paths(),
                               val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
    for path in train:
        _plant_position(path)          # val keeps the fixture's noise cache
    result = script.cached_feature_probe(train, val, "pixel_ae")
    assert result["r2"] < 0.5, result


def test_cached_feature_probe_refuses_a_cache_of_the_wrong_geometry(spike_buffer):
    """A stale cache from another geometry must be refused by name, not
    flattened into a probe of the wrong width that scores something."""
    train, val = episode_split(spike_buffer.episode_paths(),
                               val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
    bad = train[0].with_suffix(feature_suffix("pixel_ae"))
    np.save(bad, np.zeros((41, 64, 33), dtype=np.float16))
    with pytest.raises(ValueError) as error:
        script.cached_feature_probe(train, val, "pixel_ae")
    message = str(error.value)
    assert "pixel_ae" in message and "(64, 32)" in message and "(64, 33)" in message
    assert bad.name in message


def test_cached_feature_probe_refuses_a_cache_that_is_not_one_row_per_frame(
        spike_buffer):
    """T rows against T+1 privileged frames: `probe_targets` would still
    return T+1 rows and a silent truncation would fit on shifted frames."""
    train, val = episode_split(spike_buffer.episode_paths(),
                               val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
    short = train[0].with_suffix(feature_suffix("pixel_ae"))
    np.save(short, np.zeros((40, 64, 32), dtype=np.float16))
    with pytest.raises(ValueError, match="40 rows.*41 frames"):
        script.cached_feature_probe(train, val, "pixel_ae")


def test_cached_feature_probe_reports_the_split_it_used(spike_buffer):
    """Five training episodes with a 4-episode selection set leaves ONE to
    fit on; the record must say so rather than let a reader assume 16/4/20."""
    train, val = episode_split(spike_buffer.episode_paths(),
                               val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
    assert (len(train), len(val)) == (5, 1)
    result = script.cached_feature_probe(train, val, "pixel_ae")
    assert result["n_fit_episodes"] == 1
    assert result["n_select_episodes"] == 4
    assert result["n_scored_episodes"] == 1
    assert result["n_scored_rows"] == 41
    assert result["ridge"] in (1e-1, 1e1, 1e3, 1e5, 1e7)
    assert math.isfinite(result["selection_r2"])


def test_cached_feature_probe_needs_an_episode_to_fit_on_beyond_the_selection_set(
        spike_buffer):
    train, val = episode_split(spike_buffer.episode_paths(),
                               val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
    with pytest.raises(ValueError, match="fit"):
        script.cached_feature_probe(train[:4], val, "pixel_ae")
    with pytest.raises(ValueError, match="validation"):
        script.cached_feature_probe(train, [], "pixel_ae")


# --------------------------------------------------------------------------
# end to end, on CPU, three steps
# --------------------------------------------------------------------------

def test_main_writes_the_full_history_the_checks_and_the_verdict(
        spike_buffer, tmp_path, capsys):
    """One tiny run through `main`. What the record must carry: the FULL
    per-step history (open item 6), every check with its value equal to the
    field it was judged from, the side-by-side for every arm, and a decision.
    And the exit status must be the checks' verdict, not a constant 0."""
    out = tmp_path / "spike"
    status = script.main([
        "--data", str(spike_buffer.root), "--out", str(out),
        "--steps", "3", "--seq-len", "6", "--batch-size", "2", "--device", "cpu",
    ])
    record = load_record(script.spike_record_path(out))

    assert record["arm"] == "pixel_ae" and record["seed"] == 0
    assert record["steps"] == 3 and record["seq_len"] == 6
    assert record["split_seed"] == SPLIT_SEED
    assert record["device"] == "cpu"
    assert isinstance(record["git_sha"], str) and record["git_sha"]

    assert len(record["history"]["loss"]) == 3
    assert len(record["history"]["parts"]) == 3
    for part in record["history"]["parts"]:
        assert set(part) == {"embedding", "reward", "continue", "kl_dyn", "kl_rep"}

    assert set(record["checks"]) == set(script.CHECK_KEYS)
    assert record["checks"]["embedding_loss_step0"]["value"] == \
        record["history"]["parts"][0]["embedding"]
    assert record["checks"]["kl_rate_above_free_bits"]["value"] == \
        record["kl_rate_above_free_bits"]
    assert record["checks"]["probe_r2_cached_features"]["value"] == \
        record["probe"]["r2"]
    assert record["probe"]["backbone"] == "pixel_ae"

    assert set(record["embedding_loss_step0"]) == set(ARMS)
    assert record["embedding_loss_step0"]["pixel_ae"] == pytest.approx(
        record["history"]["parts"][0]["embedding"], rel=1e-6)
    assert record["decision"] == script.decision(record["checks"])
    assert (out / "world_model_pixel_ae_seed0.pt").is_file()

    # Fixture sanity, so the exit-10 path is genuinely exercised: three steps
    # at lr 1e-4 cannot lift a 0.03-nat dyn KL past the 0.20 floor.
    assert record["checks"]["kl_rate_above_free_bits"]["passed"] is False, (
        "the fixture now passes every check, so a `return 0` in main is invisible")
    assert status == script.EXIT_SPIKE_FAILED

    text = capsys.readouterr().out
    for label in ("check 1", "check 2", "check 3", "decision: "):
        assert label in text
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_spike_pixel_ae_script.py -q`
Expected: FAIL at collection — `FileNotFoundError: [Errno 2] No such file or directory: '.../scripts/spike_pixel_ae.py'` raised by `_SPEC.loader.exec_module`. (One error, zero tests collected.)

- [ ] **Step 3: Implement the script**

```python
# scripts/spike_pixel_ae.py
"""The M3c spike: 2,000 steps of `pixel_ae`, and the three checks that license the study.

Spec section 3. The nine-cell study (13.5 h) runs only if one 2,000-step
`pixel_ae` run at seed 0 shows all three of:

  1. `embedding_loss` at step 0 in [0.1, 1.0]      -- the target is informative.
     The retired end-to-end pixel arm read 0.0008: a randomly initialised CNN
     through the bottleneck emits a near-constant embedding, predicting it is
     trivial, nothing forces the posterior to encode anything, and the run is
     at a fixed point from step 0. The feature arms read 0.3165 (frozen_ssl)
     and 0.4104 (random_vit) -- that is the band.
  2. `kl_rate_above_free_bits` > 0.5               -- the dynamics prior trains.
     `train_world_model` already warns below 0.5 on `_kl_rate`'s own
     criterion; here it is a verdict. M3b's pixel arm cleared the floor on 15
     to 18 of 20,000 steps, so `prior_net` was never trained and `imagine()`
     ran on its initialisation.
  3. held-out linear-probe R^2 on the CACHED pixel_ae features > 0
                                                   -- the features carry position.
     Measured on the ViT caches at spec 3.2a: DINOv2 0.422, random_vit 0.369.
     The M3b pixel arm's probe collapsed to a constant at -0.036.

WHY THIS IS A SCRIPT AND NOT A SHELL NOTE. M3b calibrated `KL_FREE_BITS` on a
2,000-step `random_vit` run and then judged the pixel arm's failure by eye
from checkpoint diffs, because nothing had kept the per-step history. The
three criteria are functions here, the verdict is an exit status, and the
record on disk carries `history["loss"]` and `history["parts"]` IN FULL
(M3b write-up open item 6) beside the git SHA and device that produced it.

WHICH PROBE. Check 3 asks about the cache itself, before any trainable layer.
`fit_probes` and `gather_probe_data` take a model and probe what the MODEL
produces (its encoder output, its latent, its predicted embedding), so they
answer a different question; this script uses the array-level `fit_probe`,
`probe_r2` and `probe_targets` from the same module directly. The (64, 32)
rows are flattened to the encoder's own 2048-vector: spec 2.1 says the
partition is a reshape that invents nothing, so a patch-mean -- 3.2a's
reduction for the ViT caches, where the 64 rows are 64 real patches -- would
average unrelated dimensions here. The split mirrors `fit_probes(limit=20,
select_episodes=4)`: weights on 16 training episodes, ridge selected on the
next 4, scored on the first 20 validation episodes of the study's own split.

THE SIDE-BY-SIDE. Check 1's band is only meaningful next to the arms it was
taken from, so the script also prints the step-0 embedding loss for
`frozen_ssl` and `random_vit`, from one forward pass each with no training.
`step0_embedding_loss` is `train_world_model` run for exactly one step with
nothing saved: the same seed, split, loader draw and posterior sample by
construction, and tested equal to `history["parts"][0]["embedding"]` on every
arm -- a copy of the setup block would have to be kept in step by hand.

EXIT STATUS: 0 iff all three checks pass, 10 otherwise. 10 is
`report_study.py`'s `EXIT_GATE_NOT_PASSED` and means the same thing here -- a
verdict that was computed and printed in full, not an error. Anything else
(no data, a missing cache) is an uncaught traceback and exit 1; this spike is
attended, unlike the study driver, and a traceback is the right report.
"""

import argparse
import contextlib
import io
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import load_episode
from mbfps.data.features import BACKBONE_GEOMETRY
from mbfps.data.loader import feature_suffix
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.probe import fit_probe, probe_r2, probe_targets
from mbfps.eval.study import SPLIT_SEED, write_record
from mbfps.models.encoders import encoder_backbone
from mbfps.models.rssm import KL_FREE_BITS
from mbfps.training.world_model import train_world_model
from mbfps.utils.config import ARMS, get_config

SPIKE_ARM = "pixel_ae"
SPIKE_SEED = 0
SPIKE_STEPS = 2_000
"""One arm, one seed, 2,000 steps -- spec section 3. Constants, not flags: a
spike on any other arm passes its own checks and licenses nothing about this
one, and the seed is the study's seed 0 so the run is one of its nine cells
cut short, not a tenth configuration."""

EMBEDDING_LOSS_BAND: tuple[float, float] = (0.1, 1.0)
"""Closed band for check 1. The feature arms measured 0.3165 and 0.4104 at
step 0; the retired pixel arm 0.0008 (M3c design section 1)."""

MIN_KL_RATE = 0.5
"""Strict floor for check 2 -- the same 0.5 `train_world_model` warns below."""

MIN_PROBE_R2 = 0.0
"""Strict floor for check 3. Exactly zero is the constant predictor."""

PROBE_LIMIT = 20
PROBE_SELECT_EPISODES = 4
"""`fit_probes`' own split: the first `PROBE_LIMIT` training episodes, of
which the last `PROBE_SELECT_EPISODES` select the ridge; scored on the first
`PROBE_LIMIT` validation episodes."""

CHECK_KEYS: tuple[str, ...] = (
    "embedding_loss_step0", "kl_rate_above_free_bits", "probe_r2_cached_features",
)
"""The three checks, in spec order. `all_passed` and `report` read this."""

RUN_STUDY = "RUN_STUDY"
STOP_RECALIBRATE_FREE_BITS = "STOP_RECALIBRATE_FREE_BITS"
STOP_BACKBONE_UNINFORMATIVE = "STOP_BACKBONE_UNINFORMATIVE"
DECISION_TEXT = {
    RUN_STUDY: "all three checks pass; run the nine cells (Task 8)",
    STOP_RECALIBRATE_FREE_BITS: (
        "checks 1 and 3 pass but the prior did not train; STOP and re-plan "
        "KL_FREE_BITS as a three-arm calibration with the KL trajectory "
        "recorded per step (spec section 3)"),
    STOP_BACKBONE_UNINFORMATIVE: (
        "check 1 or 3 failed: the frozen backbone is not informative, and no "
        "free-bits value creates a signal it does not have (spec section 1); "
        "STOP, do not run the study"),
}

EXIT_OK = 0
EXIT_SPIKE_FAILED = 10


# --------------------------------------------------------------------------
# the three checks
# --------------------------------------------------------------------------

def check_embedding_loss(parts: list[dict]) -> dict:
    """Check 1 on `history["parts"]`: the FIRST step's embedding loss.

    The first, not the last or the mean: section 1's failure is a target that
    is born collapsed, and from 0.0008 the loss can only fall further.
    """
    if not parts:
        raise ValueError("history has no steps; the step-0 embedding loss does not exist")
    value = float(parts[0]["embedding"])
    low, high = EMBEDDING_LOSS_BAND
    # `low <= value <= high` is False for NaN, which is the right answer.
    return {"value": value, "low": low, "high": high, "passed": bool(low <= value <= high)}


def check_kl_rate(history: dict) -> dict:
    """Check 2 on the RATE, never `kl_dyn_max`: cnn/s1 peaked at 0.800 nat
    while clearing the floor on 16 of 20,000 steps."""
    value = float(history["kl_rate_above_free_bits"])
    return {"value": value, "threshold": MIN_KL_RATE, "passed": bool(value > MIN_KL_RATE)}


def check_probe_r2(r2: float) -> dict:
    """Check 3: held-out R^2 on the cached features, strictly positive."""
    value = float(r2)
    return {"value": value, "threshold": MIN_PROBE_R2, "passed": bool(value > MIN_PROBE_R2)}


def evaluate_checks(history: dict, feature_probe_r2: float) -> dict:
    """All three, keyed by `CHECK_KEYS`, each judged from its own input only."""
    return {
        "embedding_loss_step0": check_embedding_loss(history["parts"]),
        "kl_rate_above_free_bits": check_kl_rate(history),
        "probe_r2_cached_features": check_probe_r2(feature_probe_r2),
    }


def all_passed(checks: dict) -> bool:
    """A conjunction over `CHECK_KEYS`; a missing check is an error, not a pass."""
    missing = [key for key in CHECK_KEYS if key not in checks]
    if missing:
        raise ValueError(f"checks are missing {missing}; a check that was never run cannot pass")
    return all(bool(checks[key]["passed"]) for key in CHECK_KEYS)


def decision(checks: dict) -> str:
    """Spec section 3's decision rule.

    The backbone verdict DOMINATES: if check 1 or 3 fails, the target is not
    informative and section 1 shows that no `KL_FREE_BITS` value can create a
    learning signal the arm does not have -- so the recalibration route is not
    offered even when check 2 failed too. Only a lone check-2 failure sends
    the floor back for a genuine three-arm calibration.
    """
    if not (checks["embedding_loss_step0"]["passed"]
            and checks["probe_r2_cached_features"]["passed"]):
        return STOP_BACKBONE_UNINFORMATIVE
    if not checks["kl_rate_above_free_bits"]["passed"]:
        return STOP_RECALIBRATE_FREE_BITS
    return RUN_STUDY


def exit_status(checks: dict) -> int:
    return EXIT_OK if all_passed(checks) else EXIT_SPIKE_FAILED


# --------------------------------------------------------------------------
# the side-by-side
# --------------------------------------------------------------------------

def step0_embedding_loss(cfg, buffer: ReplayBuffer) -> float:
    """`train_world_model`'s step-0 embedding loss for `cfg`, from `train_world_model`.

    One step of the real training loop with nothing saved. The seed, the split,
    the loader draw and the posterior sample are the ones the check judges
    BECAUSE THIS IS THAT CODE PATH -- there is no copy of the setup block to
    keep in step with `world_model.py`. The price is one backward pass and one
    Adam step per arm, which is seconds. The one-step run's own "kl_dyn
    exceeded the floor on only x% of steps" warning is meaningless at a single
    step and is swallowed; the 2,000-step run's warning is not.
    """
    one_step = replace(cfg, train=replace(cfg.train, steps=1))
    with contextlib.redirect_stdout(io.StringIO()):
        history = train_world_model(one_step, buffer, out_dir=None, log_every=0)
    return float(history["parts"][0]["embedding"])


# --------------------------------------------------------------------------
# check 3's probe on the cache
# --------------------------------------------------------------------------

def load_cached_features(path: Path, backbone: str) -> np.ndarray:
    """`(T+1, n_patches * patch_dim)` float64 rows of `backbone`'s cache for one episode.

    Validated against `BACKBONE_GEOMETRY` by name, like the loader: a cache
    at another geometry is refused here rather than flattened into a probe of
    the wrong width that scores something.
    """
    n_patches, patch_dim = BACKBONE_GEOMETRY[backbone]
    feature_path = path.with_suffix(feature_suffix(backbone))
    features = np.load(feature_path)
    if tuple(features.shape[1:]) != (n_patches, patch_dim):
        raise ValueError(
            f"{feature_path.name} holds rows of shape {tuple(features.shape[1:])}, "
            f"but backbone {backbone!r} caches {(n_patches, patch_dim)}"
        )
    return features.reshape(features.shape[0], n_patches * patch_dim).astype(np.float64)


def _rows(paths, backbone: str) -> tuple[np.ndarray, np.ndarray]:
    """Row-aligned `(features, targets)` over `paths`, one row per FRAME."""
    xs, ys = [], []
    for path in paths:
        episode = load_episode(path)
        x = load_cached_features(path, backbone)
        y = probe_targets(episode.privileged, episode.privileged_keys)
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"{path.name}: the {backbone} cache has {x.shape[0]} rows against "
                f"{y.shape[0]} frames; a silent truncation would fit on shifted frames"
            )
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def cached_feature_probe(
    train_paths, val_paths, backbone: str,
    limit: int = PROBE_LIMIT, select_episodes: int = PROBE_SELECT_EPISODES,
) -> dict:
    """Held-out linear-probe R^2 on `backbone`'s cache -- check 3.

    Split at EPISODE granularity, like `fit_probes`: the first `limit`
    training episodes, of which the last `select_episodes` select the ridge
    (consecutive frames are near-duplicates, so a row-wise split would put the
    same scene on both sides and every ridge would look equal); scored on the
    first `limit` VALIDATION episodes, which neither the weights nor the
    selection saw. The sizes actually used are returned, because on a small
    buffer they are not 16 / 4 / 20 and the record must say what they were.
    """
    used = list(train_paths)[:limit]
    scored = list(val_paths)[:limit]
    spare = len(used) - select_episodes
    if spare < 1:
        raise ValueError(
            f"{len(used)} training episodes leave none to fit on after holding "
            f"{select_episodes} out for ridge selection"
        )
    if not scored:
        raise ValueError("no validation episodes to score the probe on")

    x_fit, y_fit = _rows(used[:spare], backbone)
    x_select, y_select = _rows(used[spare:], backbone)
    x_val, y_val = _rows(scored, backbone)
    probe = fit_probe(x_fit, y_fit, x_select, y_select)
    return {
        "backbone": backbone,
        "r2": float(probe_r2(probe, x_val, y_val)),
        "ridge": float(probe["ridge"]),
        "selection_r2": float(probe["r2"]),
        "n_fit_episodes": spare,
        "n_select_episodes": len(used) - spare,
        "n_scored_episodes": len(scored),
        "n_scored_rows": int(y_val.shape[0]),
        "n_columns": int(x_fit.shape[1]),
    }


# --------------------------------------------------------------------------
# record, report, main
# --------------------------------------------------------------------------

def git_sha() -> str:
    """`git rev-parse HEAD`, or "unknown". Never raises: a missing `git` on the
    box must not be the thing that loses a 10-minute run's record."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out or "unknown"


def spike_record_path(out_dir) -> Path:
    return Path(out_dir) / f"spike_{SPIKE_ARM}_seed{SPIKE_SEED}.json"


def _verdict(check: dict) -> str:
    return "PASS" if check["passed"] else "FAIL"


def report(record: dict) -> str:
    """The block to paste into the plan's results section, verbatim."""
    checks = record["checks"]
    c1, c2, c3 = (checks[key] for key in CHECK_KEYS)
    side = record["embedding_loss_step0"]
    probe = record["probe"]
    lines = [
        f"=== M3c spike: {record['arm']} seed {record['seed']}, {record['steps']} steps "
        f"on {record['device']} at {record['git_sha']} ===",
        "embedding_loss_step0  "
        + "  ".join(f"{arm}={side[arm]:.4f}" for arm in ARMS)
        + "   (M3c design section 1 measured cnn=0.0008 frozen_ssl=0.3165 random_vit=0.4104)",
        f"training              steps_per_second={record['steps_per_second']:.2f} "
        f"kl_dyn_max={record['kl_dyn_max']:.4f}",
        f"check 1  embedding_loss_step0       {c1['value']:.4f}  "
        f"in [{c1['low']}, {c1['high']}]  {_verdict(c1)}",
        f"check 2  kl_rate_above_free_bits    {c2['value']:.4f}  "
        f"> {c2['threshold']}  {_verdict(c2)}   (floor KL_FREE_BITS={KL_FREE_BITS})",
        f"check 3  probe_r2_cached_features   {c3['value']:+.4f}  "
        f"> {c3['threshold']}  {_verdict(c3)}   "
        f"(ridge {probe['ridge']:g}, fit {probe['n_fit_episodes']} / select "
        f"{probe['n_select_episodes']} / scored {probe['n_scored_episodes']} val "
        f"episodes, {probe['n_scored_rows']} rows x {probe['n_columns']} columns)",
        f"decision: {record['decision']} -- {DECISION_TEXT[record['decision']]}",
    ]
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # No --arm and no --seed, on purpose; see SPIKE_ARM.
    parser.add_argument("--data", default="data/my_way_home")
    parser.add_argument("--out", default="runs/m3c_spike")
    parser.add_argument("--steps", type=int, default=SPIKE_STEPS)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="mps")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    out_dir = Path(args.out)
    overrides = dict(steps=args.steps, seq_len=args.seq_len,
                     batch_size=args.batch_size, device=args.device)

    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    train_paths, val_paths = episode_split(
        buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED
    )
    cfg = get_config(SPIKE_ARM, seed=SPIKE_SEED, **overrides)
    backbone = encoder_backbone(cfg.encoder)

    history = train_world_model(cfg, buffer, out_dir=out_dir)
    probe = cached_feature_probe(train_paths, val_paths, backbone)
    # Every arm, the spike's included: the pixel_ae value here is tested equal
    # to history["parts"][0]["embedding"] on CPU, and printing both on MPS is
    # what would show a device-level discrepancy if one ever appeared.
    side_by_side = {
        arm: step0_embedding_loss(get_config(arm, seed=SPIKE_SEED, **overrides), buffer)
        for arm in ARMS
    }
    checks = evaluate_checks(history, probe["r2"])

    record = {
        "arm": SPIKE_ARM,
        "seed": SPIKE_SEED,
        "steps": args.steps,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "split_seed": SPLIT_SEED,
        "git_sha": git_sha(),
        "device": str(get_device(prefer=args.device)),
        "seconds": float(history["seconds"]),
        "steps_per_second": float(history["steps"] / history["seconds"]),
        "kl_rate_above_free_bits": float(history["kl_rate_above_free_bits"]),
        "kl_dyn_max": float(history["kl_dyn_max"]),
        # IN FULL. Open item 6 of the M3b write-up: section 1 of the M3c design
        # had to be reconstructed because no run kept this.
        "history": {
            "loss": [float(v) for v in history["loss"]],
            "parts": [{k: float(v) for k, v in p.items()} for p in history["parts"]],
        },
        "embedding_loss_step0": side_by_side,
        "probe": probe,
        "checks": checks,
        "decision": decision(checks),
        "episodes": {
            "train": [p.name for p in train_paths],
            "val": [p.name for p in val_paths],
        },
    }
    # `write_record`, not `json.dump`: a `kl_rep` of NaN in one of 2,000 parts
    # is a bare `NaN` token no strict parser reads, and the write is atomic.
    write_record(spike_record_path(out_dir), record)
    print(report(record), flush=True)
    print(f"record: {spike_record_path(out_dir)}", flush=True)
    return exit_status(checks)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_spike_pixel_ae_script.py -q`
Expected: PASS (25 tests). Then the whole suite:

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q`
Expected: green — the count is the count Task 6 left plus 25, exit 0. Record the number.

- [ ] **Step 5: Mutation-test**

Set up the harness with its three self-checks before trusting a single row. All three have silently lied in this project.

```bash
export PYTHONDONTWRITEBYTECODE=1
T="tests/eval/test_spike_pixel_ae_script.py"
S="scripts/spike_pixel_ae.py"
run() { find . -name __pycache__ -prune -exec rm -rf {} + ; .venv/bin/python -m pytest "$T" -q -x 2>&1 | tail -1; }
restore() { git checkout -- "$S"; git diff --stat -- "$S" | grep -q . && echo "RESTORE FAILED" && exit 1; }

# self-check 1: the tests exercise the file at scripts/, not a stale copy.
#   `scripts/` is not a package and the test loads it BY PATH, so the editable
#   install cannot shadow it -- but `mbfps` IS editable, so any row that
#   mutates a library helper must be run from the working tree, never from a
#   `git archive` export. Prove the path: an unparseable script must break every test.
echo "this is not python" >> "$S"; run   # expected: ERROR at collection (SyntaxError)
restore

# self-check 2: an unproven harness proves nothing. Run a known-fatal mutation
#   FIRST and watch it fail before reading any other row.
sed -i '' 's/return EXIT_OK if all_passed(checks) else EXIT_SPIKE_FAILED/return EXIT_OK/' "$S"; run
#   expected: FAILED test_exit_status_is_zero_iff_all_pass (and test_main_...)
restore; run                              # expected: 25 passed

# self-check 3: a same-byte-length mutation can leave a stale .pyc that survives
#   a correct restore. `run` clears __pycache__ and PYTHONDONTWRITEBYTECODE is
#   set, so after every restore the green run above is the proof.
```

Then, one row at a time (mutate, `run`, confirm the named test is in the failures, `restore`, `run` green):

| mutation | must be caught by |
|---|---|
| `check_embedding_loss` reads `parts[-1]` | `test_embedding_check_reads_step_zero_not_the_last_step` |
| `check_embedding_loss` drops the lower edge (`value <= high` only) | `test_embedding_check_flips_on_each_edge_alone` (the 0.0008 case) |
| `check_embedding_loss` drops the upper edge | `test_embedding_check_flips_on_each_edge_alone` (the 1.5 case) |
| `check_embedding_loss` makes the band open at 0.1 (`low < value`) | `test_embedding_check_flips_on_each_edge_alone` (the 0.1 case) |
| `check_embedding_loss` written as `not (value < low or value > high)` | `test_non_finite_values_fail_every_check` |
| `check_kl_rate` uses `>=` | `test_kl_check_flips_at_one_half` (the 0.5 case) |
| `check_kl_rate` reads `history["kl_dyn_max"]` | `test_kl_check_reads_the_rate_not_the_max` |
| `check_probe_r2` uses `>=` | `test_probe_check_flips_at_zero` (the 0.0 case) |
| `evaluate_checks` hands check 3 `history["kl_rate_above_free_bits"]` instead of the probe R² | `test_each_check_flips_independently_of_the_other_two` (the `r2=` case flips nothing; the `kl_rate=` case flips two) |
| `all_passed` uses `any` | `test_all_passed_requires_all_three` |
| `all_passed` drops the missing-key guard | `test_all_passed_refuses_a_checks_dict_missing_a_check` |
| `decision` returns `RUN_STUDY` whenever check 1 passes | `test_decision_sends_a_lone_kl_failure_to_a_free_bits_recalibration`, `test_decision_calls_the_backbone_uninformative_on_check_1_or_3` (the `r2=` case) |
| `decision` tests check 2 first (offers recalibration on a dead backbone) | `test_decision_backbone_verdict_dominates_a_kl_failure` |
| `exit_status` returns `EXIT_OK` | `test_exit_status_is_zero_iff_all_pass`, `test_main_writes_the_full_history_the_checks_and_the_verdict` |
| `report` prints `PASS` for every check | `test_report_marks_each_check_with_its_own_verdict` |
| `report` prints the pixel_ae value under every arm's name | `test_report_marks_each_check_with_its_own_verdict` cannot see it — **add** an assertion that `frozen_ssl=0.3165` and `random_vit=0.4104` appear verbatim before committing |
| `step0_embedding_loss` builds `get_config(SPIKE_ARM, ...)` regardless of `cfg` | `test_step0_embedding_loss_matches_train_world_model_step_zero_on_every_arm` (the `frozen_ssl` and `random_vit` rows) |
| `step0_embedding_loss` passes `cfg` unchanged (no `steps=1`) | `test_step0_embedding_loss_runs_one_step_saves_nothing_and_reads_step_zero` (`steps == 1`); the equality test cannot -- at TINY's 3 steps `parts[0]` is the same number |
| `step0_embedding_loss` passes `out_dir=Path(".")` | `test_step0_embedding_loss_runs_one_step_saves_nothing_and_reads_step_zero` (`out_dir is None`) |
| `step0_embedding_loss` returns `parts[-1]["embedding"]` | `test_step0_embedding_loss_runs_one_step_saves_nothing_and_reads_step_zero` (the spy's two parts differ) |
| `step0_embedding_loss` builds `replace(cfg.train, steps=1, seed=0)` | `test_step0_embedding_loss_runs_one_step_saves_nothing_and_reads_step_zero` (`replace(seen.train, steps=...) == cfg.train`) -- **run this row with the fixture's seed changed to 1 first**, or the mutation is invisible |
| `cached_feature_probe` scores on `x_fit, y_fit` | `test_cached_feature_probe_scores_the_validation_episodes_not_the_fit_ones` |
| `load_cached_features` reads `feature_suffix("dinov2")` | `test_cached_feature_probe_recovers_position_planted_in_the_pixel_ae_cache` (geometry `(64, 384)` is refused) |
| `load_cached_features` drops the geometry guard | `test_cached_feature_probe_refuses_a_cache_of_the_wrong_geometry` |
| `_rows` drops the row-count guard | `test_cached_feature_probe_refuses_a_cache_that_is_not_one_row_per_frame` |
| `cached_feature_probe` reports `n_fit_episodes = limit - select_episodes` (16) instead of what it used | `test_cached_feature_probe_reports_the_split_it_used` |
| `main` writes `history["loss"][-20:]` instead of the full list | `test_main_writes_the_full_history_the_checks_and_the_verdict` (length 3 ≠ 3 only if steps > 20 — **change the test to assert `len == record["steps"]` AND run it once with `--steps 25`** — a 3-step run cannot tell `[-20:]` from the full list, so "caught by inspection" is not an option for this row) |
| `main` judges the checks from `side_by_side["pixel_ae"]` instead of `history` | NOT caught on CPU — the two are equal by construction there; this is why `report` prints both and Step 9 asks you to compare them on MPS |
| `_parser` gains `--seed` | `test_the_spike_is_pinned_to_pixel_ae_seed_0_and_2000_steps` |

Any row that survives is a missing test. Add it before committing; the two rows marked "add"/"change" above are the ones known to need it.

- [ ] **Step 6: Commit the script**

```bash
git add scripts/spike_pixel_ae.py tests/eval/test_spike_pixel_ae_script.py
git commit -m "feat: the M3c spike -- three checks with a verdict, and the full history on disk

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Preconditions for the run**

The spike needs the `pixel_ae` cache for all 122 episodes (produced by this plan's `cache_features --backbone pixel_ae` task) beside the two ViT caches that already exist, and a green suite at the code state that will run it.

```bash
ls data/my_way_home/*.features_pixel_ae.npy | wc -l      # expected: 122
ls data/my_way_home/*.features.npy | wc -l               # expected: 122
ls data/my_way_home/*.features_random_vit.npy | wc -l    # expected: 122
.venv/bin/python -c "import numpy as np, glob; f = sorted(glob.glob('data/my_way_home/*.features_pixel_ae.npy'))[0]; a = np.load(f); print(a.shape, a.dtype)"
# expected: (<T+1>, 64, 32) float16
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q   # expected: the Step 4 count, exit 0
git status --short                                        # expected: clean (nothing under runs/ or data/ is tracked)
git rev-parse HEAD                                        # note it; the record must carry this SHA
```

If the `pixel_ae` count is not 122, STOP and run the cache task first; `SequenceLoader` raises `FileNotFoundError` on the first missing file and the spike is not a spike without all of them.

- [ ] **Step 8: Run the spike**

```bash
mkdir -p runs/m3c_spike
caffeinate -dimsu .venv/bin/python -u scripts/spike_pixel_ae.py \
  --data data/my_way_home --out runs/m3c_spike --device mps 2>&1 | tee runs/m3c_spike/spike.log
echo "exit=${PIPESTATUS[0]}"
```

Expected duration **~10 min** on this Mac: 2,000 steps at the feature-arm rate (3.8–3.98 steps/s measured in M3b, ~8.5 min), then the probe (40 episodes read once, one 2,049×2,049 solve per ridge, under a minute), then three single forward passes. `caffeinate -dimsu` is required for an unattended run on this host; `-i` alone is insufficient.

Expected output, in order: `[pixel_ae] step 100/2000 loss=...` every 100 steps; possibly `train_world_model`'s `WARNING: kl_dyn exceeded the 0.2 nat floor on only ...` (that IS check 2 failing — the verdict below will say so); then the block:

```
=== M3c spike: pixel_ae seed 0, 2000 steps on mps at <sha> ===
embedding_loss_step0  pixel_ae=0.____  frozen_ssl=0.3165  random_vit=0.4104   (M3c design section 1 measured cnn=0.0008 frozen_ssl=0.3165 random_vit=0.4104)
training              steps_per_second=3.__ kl_dyn_max=0.____
check 1  embedding_loss_step0       0.____  in [0.1, 1.0]  PASS|FAIL
check 2  kl_rate_above_free_bits    0.____  > 0.5  PASS|FAIL   (floor KL_FREE_BITS=0.2)
check 3  probe_r2_cached_features   +0.____  > 0.0  PASS|FAIL   (ridge 1000, fit 16 / select 4 / scored 20 val episodes, 10520 rows x 2048 columns)
decision: RUN_STUDY|STOP_RECALIBRATE_FREE_BITS|STOP_BACKBONE_UNINFORMATIVE -- ...
record: runs/m3c_spike/spike_pixel_ae_seed0.json
exit=0|10
```

The `frozen_ssl` and `random_vit` side-by-side values should reproduce section 1's 0.3165 and 0.4104 to a few decimals — they are the same computation on the same seed and the same caches. If they do not, the caches, the split or the seeding moved, and the spike measured something else: STOP and find out what before reading the checks.

Sanity checks on the record before reading the verdict:

```bash
.venv/bin/python - <<'EOF'
from mbfps.eval.study import load_record
r = load_record("runs/m3c_spike/spike_pixel_ae_seed0.json")
print("git_sha", r["git_sha"], "device", r["device"], "steps", r["steps"])
print("history lengths", len(r["history"]["loss"]), len(r["history"]["parts"]))   # both 2000
print("step0 from history", r["history"]["parts"][0]["embedding"])
print("step0 from side-by-side", r["embedding_loss_step0"]["pixel_ae"])          # same number on MPS?
print("checks", {k: (v["value"], v["passed"]) for k, v in r["checks"].items()})
print("decision", r["decision"])
print("nonfinite", r["nonfinite"])                                                # expected {}
EOF
```

`git_sha` must equal the SHA from Step 7; the two step-0 numbers must agree to float precision (they are the one mutation the CPU tests cannot see — if they differ on MPS, record both and which one the check used: `history`).

- [ ] **Step 9: Record the result and take the decision**

Create `## Task 7 results` at the end of this plan and fill it. **A FAIL is a result, not a failure to hide** — the spike exists to be allowed to say no before 13.5 h are spent.

```markdown

## Task 7 results

**Run:** `runs/m3c_spike/spike_pixel_ae_seed0.json`, log `runs/m3c_spike/spike.log`.
git SHA ______ · device ______ · 2,000 steps · seq_len 64 · batch 16 · ____ steps/s · ____ s wall.
Probe protocol: `fit_probe` ridge selection on the flattened `(T+1, 2048)` cache; fit 16 / select 4 training
episodes, scored on 20 validation episodes of `episode_split(seed=0)`, ____ rows; selected ridge ____;
selection R² ____.

| check | value | must read | verdict |
|---|---|---|---|
| 1 `embedding_loss` at step 0 (`history["parts"][0]["embedding"]`) | ______ | in [0.1, 1.0] | PASS / FAIL |
| 2 `kl_rate_above_free_bits` over 2,000 steps (`kl_dyn_max` ______) | ______ | > 0.5 | PASS / FAIL |
| 3 held-out probe R² on the cached `pixel_ae` features | ______ | > 0 | PASS / FAIL |

**Side-by-side, embedding loss at step 0** (one forward pass each, seed 0, same split and loader draw):

| arm | this run | M3c design §1 |
|---|---|---|
| `pixel_ae` | ______ (side-by-side) / ______ (history) | — (new) |
| `frozen_ssl` | ______ | 0.3165 |
| `random_vit` | ______ | 0.4104 |
| `cnn` (retired) | not run | 0.0008 |

**Decision:** `______` (the script's own verdict; exit status ____).

Decision rule, spec §3:
- all three PASS → **Task 8**: run the nine cells into `runs/m3_study_v2`.
- check 2 FAILS alone → **STOP**. Do not run the study. Re-plan `KL_FREE_BITS` as a genuine
  three-arm calibration with the KL trajectory recorded per step; `history["parts"][i]["kl_dyn"]`
  from this record is the `pixel_ae` trajectory for it.
- check 1 or 3 FAILS (with or without 2) → **STOP**. The frozen `pixel_ae` backbone is not
  informative under this objective; §1's argument says no floor can fix that. Do not run the study;
  the next step is a different backbone (the `pixel_ae_conv` alternative in §2.1 is the named
  candidate), not a different floor.
```

If check 2 alone failed, also paste the first 20 and last 20 entries of `[p["kl_dyn"] for p in r["history"]["parts"]]` under the table — the trajectory is the thing the recalibration will be planned from, and it is the first time this project has had it.

- [ ] **Step 10: Commit the result**

```bash
git status --short          # must show ONLY the plan; runs/ is gitignored
git add docs/superpowers/plans/2026-09-11-mb-fps-m3c-pixel-ae-arm.md
git commit -m "docs: M3c spike -- pixel_ae at 2,000 steps, three checks, and the decision

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

If the verdict was `RUN_STUDY`, proceed to Task 8. Otherwise this plan stops here, and the commit message's second line should say which check failed.

---

### Task 8: The nine-cell run, the report, the diagnostics, the pooled statistic

Everything before this task changed code. This task spends ~14 laptop-hours producing the nine records the spec's §4 asks for, and then reads them in the order the spec fixes: `report_study.py`, `diagnose_dynamics.py`, `pool_dynamics.py`. No module under `src/` or `scripts/` changes here. What this task adds to the repository is one section of this plan, filled from the run's own artefacts — which live under `runs/` and are gitignored, so the numbers in the plan are the only copy that survives a clone.

**The M3b records in `runs/m3_study` are NOT compared against, anywhere in this task** (spec §5). They were produced by a different `encoders.py` — the `_N_PATCHES` constant and `cfg.patch_dim` are gone, the geometry now comes from `BACKBONE_GEOMETRY` — under a different code state, with one arm (`cnn`) that no longer exists as a study arm and a new one (`pixel_ae`) that did not. `encoders.py` is on every arm's path (§2.4), so even the six `frozen_ssl`/`random_vit` cells are not the same experiment. The tools refuse the comparison mechanically too: with `ARMS = ("pixel_ae", "frozen_ssl", "random_vit")`, `report_study.py --out runs/m3_study` now fails `every_record_is_a_study_cell` on the three `cnn` records and `all_nine_cells_present` on the three missing `pixel_ae` cells. The M3b write-up stands as the record of what M3b measured; nothing here revises it. **Every script in this task defaults `--out` to `runs/m3_study`, the M3b directory** — pass `--out runs/m3_study_v2` to every one of them, every time. Omitting it does not error; it silently reports the wrong study.

**Files:**
- Create: nothing under version control. `runs/m3_study_v2/` (records, checkpoints, diagnostics, figure, `study.log`) and `runs/m3c_check_records.py` (the acceptance check below) are both under `runs/`, which `.gitignore` excludes.
- Modify: `docs/superpowers/plans/2026-09-11-mb-fps-m3c-pixel-ae-arm.md` (this plan — the file this task section lives in) — append "## Task 8 results".
- Test: the acceptance check `runs/m3c_check_records.py`, written in Step 1 and run in Steps 2, 3 and 7. It is a check on the run's artefacts, not on code, so it is not a pytest file; see Step 9 for why that is still test-first.

> Read before starting: `scripts/run_study.py` lines 78–87 (`SEEDS`, `_SLOW_ARMS`), 97–122 (`REQUIRED_RECORD_KEYS`), 143–198 (lock, exit statuses), 312–337 (job ordering), 352–398 (`acquire_lock`), 607–694 (`main`); `scripts/report_study.py` lines 57–84 (exit statuses), 214–242 (`training_table`); `scripts/diagnose_dynamics.py` lines 100–140 and 177–187 (exit statuses), 1667–1808 (`main`); `scripts/pool_dynamics.py` lines 73–79 and 257–301; `src/mbfps/eval/study.py` `run_job`; `src/mbfps/training/world_model.py` lines 149–176 (the KL rate and its WARNING).

**Interfaces:**
- Consumes: `scripts/run_study.py` CLI — `--out`, `--device`, `--arms` (`choices=ARMS`), `--seeds` (`choices=(0, 1, 2)`), exit statuses `EXIT_OK = 0`, `EXIT_JOB_FAILED = 1`, `EXIT_CONFIG_MISMATCH = 3`, `EXIT_LOCKED = 4`, `EXIT_NO_DATA = 5`, `EXIT_OUT_UNUSABLE = 6`, and `_SLOW_ARMS: tuple[str, ...] = ()`; the record keys `run_job` writes, in particular the four this milestone added — `"git_sha": str`, `"device": str`, `"encoder_params": int`, `"history": {"loss": list[float], "parts": list[dict[str, float]]}` — beside `"kl_rate_above_free_bits"`, `"steps_per_second"`, `"seconds"`, `"probe"["embedding_selection_r2"]`, `"position"["gap_final"]`; `scripts/report_study.py` CLI — `--out`, `--figure`, exit `EXIT_OK = 0` (gate PASSED), `EXIT_GATE_NOT_PASSED = 10`, `EXIT_NO_RECORDS = 7`, `EXIT_MISLABELLED = 8`, `EXIT_UNREADABLE = 9`; `scripts/diagnose_dynamics.py` CLI — `--out`, `--device`, exit `0` and `11`–`17`; `scripts/pool_dynamics.py` CLI — `--out`, `--treatment`, `--control` (both `choices=ARMS`), exit `0` and `18`–`22`; `mbfps.eval.aggregate.load_records(out_dir) -> list[dict]`, `mbfps.eval.aggregate.record_cell(record) -> tuple[str, int] | None`; `mbfps.utils.config.ARMS`; `mbfps.models.rssm.KL_FREE_BITS = 0.20`.
- Produces: `runs/m3_study_v2/result_<arm>_seed<n>.json` ×9, `world_model_<arm>_seed<n>.pt` ×9, `diagnostic_<arm>_seed<n>.json` ×9, `curves.png`, `study.log`, `report.txt`, `diagnostics.txt`, `pool_vs_random_vit.txt`, `pool_vs_pixel_ae.txt` (none committed); the "## Task 8 results" section of this plan (committed).

- [ ] **Step 1: Write the acceptance check — the failing test**

This is the test the run has to pass, written before the run so that "the run worked" is a check the run can fail rather than an impression from the log. It asserts the things no table in `report_study.py` prints: that all nine cells came from **one git SHA which is HEAD**, on **one device which is `mps`**, with the **encoder parameter counts the spec states** (§2.2: `pixel_ae` 1,056; the ViT arms 12,320), a **full 20,000-step history**, and the two conditions spec §3 licenses the pixel arm on — `kl_rate_above_free_bits > 0.5` and a step-0 embedding loss in the feature-arm band. It then prints the provenance rows the results template in Step 10 is filled from.

```python
# runs/m3c_check_records.py
"""Acceptance check for the M3c nine-cell run. Exits 1 on the first failure.

Not a pytest file and not under version control (`runs/` is gitignored):
it checks ARTEFACTS, not code, and it is only meaningful against the one
directory the study wrote. Usage:

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python runs/m3c_check_records.py [OUT]

OUT defaults to runs/m3_study_v2. Passing runs/m3_study (the M3b directory)
is Step 3's proof that the check can fail -- it must, on the missing
`pixel_ae` cells and the three `cnn` strangers, BEFORE it is trusted on v2.
"""

import subprocess
import sys

from mbfps.eval.aggregate import load_records, record_cell
from mbfps.models.rssm import KL_FREE_BITS
from mbfps.utils.config import ARMS

OUT = sys.argv[1] if len(sys.argv) > 1 else "runs/m3_study_v2"

# HAND-WRITTEN LITERALS, deliberately not derived from `ARMS` or `SEEDS`.
# A check parametrised over the collection under test shrinks with it: drop
# an arm from ARMS and the expected set drops the same cells, and an
# eight-cell study passes as complete (the L7 species). The literal is
# compared for EQUALITY against the package's tuple so that a drift between
# the two is itself a failure.
EXPECTED_ARMS = ("pixel_ae", "frozen_ssl", "random_vit")
EXPECTED_CELLS = {(arm, seed) for arm in EXPECTED_ARMS for seed in (0, 1, 2)}
# Spec section 2.2: Linear(32 -> 32) = 32*32 + 32; Linear(384 -> 32) =
# 384*32 + 32. The LayerNorm is parameter-free, so this is the whole encoder.
EXPECTED_PARAMS = {"pixel_ae": 1056, "frozen_ssl": 12320, "random_vit": 12320}
STEPS = 20_000
# Spec section 3, first check: the embedding target is informative from step
# 0. The M3b pixel arm read 0.0008 there; the feature arms 0.32 and 0.41.
EMBED0_BAND = (0.1, 1.0)

assert tuple(ARMS) == EXPECTED_ARMS, f"ARMS is {ARMS}, not {EXPECTED_ARMS}"

head = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
).stdout.strip()
dirty = subprocess.run(
    ["git", "status", "--porcelain", "--", "src", "scripts"],
    capture_output=True, text=True, check=True,
).stdout.strip()
# A SHA only names a code state if the tree at that SHA is the tree that ran.
# `git_sha` in the record is `git rev-parse HEAD` at run time, which says
# nothing about uncommitted edits to src/ or scripts/; this is the only place
# that difference is checked.
assert not dirty, f"src/ or scripts/ has uncommitted changes:\n{dirty}"

records = load_records(OUT)
cells = {record_cell(r): r for r in records}
missing = sorted(EXPECTED_CELLS - set(cells))
assert not missing, (
    f"expected 9 records in {OUT}, found {len(cells)}; missing {missing}")
strangers = sorted(set(cells) - EXPECTED_CELLS)
assert not strangers, f"records in {OUT} that are not study cells: {strangers}"
assert len(records) == 9, f"{len(records)} records for 9 cells: a duplicate"

shas = {r["git_sha"] for r in records}
assert shas == {head}, (
    f"git_sha across the nine records is {sorted(shas)}; HEAD is {head}. "
    "Either two code states produced these cells, or HEAD moved after the run.")
devices = {r["device"] for r in records}
# `get_device` falls back to cpu SILENTLY when MPS is unavailable, and a cpu
# cell is both ~6x slower and a different RNG stream (the M3b write-up
# measured cpu moving a gain by 0.009 and flipping two ridges). One device,
# and it is the one asked for.
assert devices == {"mps"}, f"device across the nine records is {sorted(devices)}"

rows = []
for (arm, seed), r in sorted(cells.items()):
    assert r["encoder_params"] == EXPECTED_PARAMS[arm], (
        f"{arm}/s{seed}: encoder_params={r['encoder_params']}, "
        f"expected {EXPECTED_PARAMS[arm]}")
    loss, parts = r["history"]["loss"], r["history"]["parts"]
    assert r["steps"] == STEPS, f"{arm}/s{seed}: steps={r['steps']}"
    assert len(loss) == len(parts) == STEPS, (
        f"{arm}/s{seed}: history has {len(loss)} losses and {len(parts)} parts, "
        f"not {STEPS} -- the full per-step history is the point of the field")
    kl_rate = r["kl_rate_above_free_bits"]
    # `_kl_rate`'s own comparability criterion: arms whose prior trained on
    # different fractions of steps cannot be compared. Every M3b feature
    # cell read 0.75-0.98; the M3b pixel arm read 0.0008.
    assert kl_rate > 0.5, (
        f"{arm}/s{seed}: kl_rate_above_free_bits={kl_rate:.4f} <= 0.5; the "
        f"dynamics prior cleared the {KL_FREE_BITS} nat floor on too few steps")
    embed0 = parts[0]["embedding"]
    if arm == "pixel_ae":
        assert EMBED0_BAND[0] <= embed0 <= EMBED0_BAND[1], (
            f"pixel_ae/s{seed}: embedding loss at step 0 is {embed0:.4f}, outside "
            f"{EMBED0_BAND}; the target is not informative (spec section 3)")
    rows.append((
        arm, seed, r["git_sha"][:9], r["device"], r["encoder_params"],
        r["steps_per_second"], kl_rate, embed0,
        # Printed, NOT asserted: the ladder's verdict in diagnose_dynamics.py
        # is what decides whether this probe can register an action effect.
        r["probe"]["embedding_selection_r2"],
        r["position"]["gap_final"], r["seconds"] / 3600.0,
    ))

print(f"{'cell':<16}{'git_sha':>10}{'device':>7}{'enc_params':>11}{'steps/s':>9}"
      f"{'kl_rate':>9}{'embed0':>9}{'probe_r2':>10}{'gap_final':>11}{'hours':>7}")
for arm, seed, sha, dev, n, sps, kl, e0, r2, gap, h in rows:
    print(f"{f'{arm}/s{seed}':<16}{sha:>10}{dev:>7}{n:>11}{sps:>9.3f}{kl:>9.4f}"
          f"{e0:>9.4f}{r2:>+10.4f}{gap:>+11.4f}{h:>7.3f}")
print(f"total hours {sum(r[-1] for r in rows):.3f}; HEAD {head}")
print("OK: nine cells, one git_sha == HEAD, one device == mps")
```

- [ ] **Step 2: Run it against the empty study directory — it must fail**

Run:
```bash
cd /Users/raphaelchen/Desktop/csgo-bot
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python runs/m3c_check_records.py
```
Expected: FAIL —
```
AssertionError: expected 9 records in runs/m3_study_v2, found 0; missing [('frozen_ssl', 0), ('frozen_ssl', 1), ('frozen_ssl', 2), ('pixel_ae', 0), ('pixel_ae', 1), ('pixel_ae', 2), ('random_vit', 0), ('random_vit', 1), ('random_vit', 2)]
```
(`load_records` globs a directory that does not exist yet and returns an empty list; that is the correct "nothing has run" reading.) If it instead fails on the `dirty` assertion, commit the preceding tasks' work first — the run must be made from a committed tree or `git_sha` names a state nobody can check out.

- [ ] **Step 3: Prove the check can fail on a real study — run it against M3b's directory**

The M3b directory has nine complete records; if the check passed on it, the check is not checking. Run:
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python runs/m3c_check_records.py runs/m3_study
```
Expected: FAIL —
```
AssertionError: expected 9 records in runs/m3_study, found 9; missing [('pixel_ae', 0), ('pixel_ae', 1), ('pixel_ae', 2)]
```
Nine records, none of them `pixel_ae`. (The three `cnn` strangers would trip the next assertion, and the absent `git_sha` key the one after; the first failure is enough, but note that all three would fire — this directory fails the check three independent ways.)

- [ ] **Step 4: Pre-flight — everything the driver will not check for you**

Each of these is a way the 14 hours get spent on the wrong run, in the order they should be looked at. All must hold before Step 5.

```bash
cd /Users/raphaelchen/Desktop/csgo-bot

# 1. The suite is green on the tree that will run. Expected: "N passed", exit 0,
#    with N the count the preceding task's Step 4 states -- never below 1120.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q

# 2. The tree that will run is committed. Expected: no output.
git status --porcelain -- src scripts

# 3. No stale bytecode: the nine cells are produced by the committed source.
find src scripts -name __pycache__ -type d -exec rm -rf {} +

# 4. MPS is what the cells will run on. `get_device` falls back to cpu silently
#    and a cpu cell takes ~6x longer. Expected: True
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"

# 5. All three feature caches are present, one per episode, and the pixel one
#    has the geometry BACKBONE_GEOMETRY["pixel_ae"] promises. Expected:
#    122 / 122 / 122, then "(526, 64, 32) float16".
ls data/my_way_home/*.features.npy | wc -l
ls data/my_way_home/*.features_random_vit.npy | wc -l
ls data/my_way_home/*.features_pixel_ae.npy | wc -l
.venv/bin/python -c "
import glob, numpy as np
a = np.load(sorted(glob.glob('data/my_way_home/*.features_pixel_ae.npy'))[0])
print(a.shape, a.dtype)"

# 6. The M2 checkpoint the pixel_ae backbone was cached from is still there
#    (cache_features.py needed it; the study itself does not). Expected: the path.
ls runs/m2_fixed/autoencoder_cnn.pt

# 7. The study directory does not exist yet, so nothing can be "already done".
#    Expected: "No such file or directory".
ls runs/m3_study_v2
```

A missing `pixel_ae` cache does not stop the driver — the failure policy in `run_study.py` catches the `FileNotFoundError`, prints a traceback, and moves on. You would find out at hour 4.5, when the first `pixel_ae` cell fails in its first second and the driver proceeds to `random_vit`. Check 5 is what makes that a ten-second finding instead.

**Precondition from spec §3:** the 2,000-step `pixel_ae` spike has been run and all three of its checks read as the spec requires — `embedding_loss` at step 0 in 0.1–1.0, `kl_rate_above_free_bits > 0.5`, held-out probe R² on the cached `pixel_ae` features > 0. Have the three numbers in hand; they go into the results section. If any of them failed, the spec says `KL_FREE_BITS` is revisited as a three-arm calibration first, and this task does not start.

- [ ] **Step 5: Run the nine cells**

One invocation, all arms, all seeds. This is now sane where it was not in M3b: `_SLOW_ARMS` is empty, every cell trains at feature-arm speed (M3b measured 3.66–3.89 steps/s → 1.43–1.52 h training + ~100 s evaluation per cell), so the whole study is **~13.5 h** — 9 × 1.5 h — against the 38 h the free-bits re-run would have cost with the end-to-end pixel arm at 9 h per cell (spec §4). One `caffeinate` covers it; it is one night plus a morning.

```bash
cd /Users/raphaelchen/Desktop/csgo-bot
mkdir -p runs/m3_study_v2        # tee needs the directory before the driver creates it
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 caffeinate -dimsu .venv/bin/python -u scripts/run_study.py \
  --out runs/m3_study_v2 --device mps 2>&1 | tee runs/m3_study_v2/study.log
echo "driver exit: $?"
```

`caffeinate -dimsu`, not `-i`: the Global Constraints record that `-i` was insufficient on this host. `-u` tells the driver to log into `runs/m3_study_v2/` rather than the repo root — the M3b run's `tee study.log` left an untracked file at the root that still shows in `git status`; under `runs/` it is gitignored and beside the records it describes. `set -o pipefail` makes `$?` the driver's status rather than `tee`'s (works in zsh and bash).

Expected first line:
```
2026-09-1xTHH:MM:SS 9 job(s) pending: frozen_ssl/s0, frozen_ssl/s1, frozen_ssl/s2, pixel_ae/s0, pixel_ae/s1, pixel_ae/s2, random_vit/s0, random_vit/s1, random_vit/s2
```
Nine, in that order. With `_SLOW_ARMS = ()` the sort key is `(False, arm, seed)`, so the arms run alphabetically and seeds 0..2 within each. Then, per cell, `===== [k/9] <arm> seed <n> <timestamp> =====`, a `[<arm>] step 100/20000 loss=...` line every 100 steps, and the `job_summary` block. The run ends with:
```
9/9 job(s) completed; records in runs/m3_study_v2
driver exit: 0
```

**The per-seed alternative.** If you would rather have a complete three-arm comparison at one seed after 4.5 h than three `frozen_ssl` cells, run the seeds one at a time; the driver resumes off the records, so the three invocations and the single one produce the identical nine files:
```bash
PYTHONDONTWRITEBYTECODE=1 caffeinate -dimsu .venv/bin/python -u scripts/run_study.py \
  --out runs/m3_study_v2 --device mps --seeds 0 2>&1 | tee -a runs/m3_study_v2/study.log
# then --seeds 1, then --seeds 2 (or a bare invocation, which runs whatever is left)
```
`tee -a` on the later invocations, or each one truncates the log of the last. `--seeds 0` orders `frozen_ssl/s0, pixel_ae/s0, random_vit/s0` — the pixel arm's first cell finishes at ~3 h instead of ~6 h, which is the argument for this form.

**Three things to check while it runs, and when.**

1. *At ~1.5 h, when `result_frozen_ssl_seed0.json` appears:* the device.
   ```bash
   .venv/bin/python -c "import json; r=json.load(open('runs/m3_study_v2/result_frozen_ssl_seed0.json')); print(r['device'], r['git_sha'], r['encoder_params'], r['steps_per_second'])"
   ```
   Expected: `mps <HEAD sha> 12320 3.6..3.9`. `cpu` here means MPS was not available and the remaining eight cells will take ~6× longer on a different RNG stream — Ctrl-C, fix, delete the record, restart. The record is ~5 MB (the full 20,000-step history, 20,000 × 5 loss parts, is in it), so read it with Python, not `less`.

2. *At the end of the first `pixel_ae` cell* (~6 h in the single invocation, ~3 h in the per-seed form) — **the first number to read in this whole task:** its `job_summary` block's `training` line:
   ```
     training  kl_rate_above_free_bits=0.xxx kl_dyn_max=x.xxx loss_last20=x.xxxx
   ```
   `kl_rate_above_free_bits` **must read > 0.5**. M3b's pixel arm read 0.0008 — 15–18 steps of 20,000 — and that single number is the whole reason this milestone exists (spec §1). If it is below 0.5 the line before the summary is `[pixel_ae] WARNING: kl_dyn exceeded the 0.2 nat floor on only x.x% of steps`, and the spec's answer is not to run the other two pixel cells: stop (Ctrl-C), record the number, and go back to §3 — `KL_FREE_BITS` is revisited as a genuine three-arm calibration with the KL trajectory read per step out of `history["parts"]`. At the end of the run:
   ```bash
   grep -c "WARNING: kl_dyn" runs/m3_study_v2/study.log
   ```
   Expected: `0`.

3. *Any cell's summary:* `cell arm=<arm> seed=<n> (requested <arm>/s<n>)` — the two must agree; a `MISLABELLED` line means the record was quarantined to `*.mislabelled` and the cell counts as failed.

**If the run is interrupted.** Nothing is lost but the in-flight cell: no record is written for an unfinished cell, and re-running the same command picks up exactly the cells without one (`k job(s) pending`, k < 9). A checkpoint `world_model_<arm>_seed<n>.pt` left by a cell killed during evaluation is overwritten by the retrain. Two cases for the claim file:

- *Ctrl-C:* `main`'s `finally` removes `runs/m3_study_v2/study.lock`. Just re-run.
- *A hard kill — power, `kill -9`, the machine going down:* the lock survives, by design (it names the pid, host and start time so a live run can be told from a dead one), and the re-run exits **4** (`EXIT_LOCKED`) with `another driver already holds runs/m3_study_v2/study.lock: {"pid": ..., "host": ..., "started": ...}`. Confirm the pid is dead, then remove the claim and re-run:
  ```bash
  cat runs/m3_study_v2/study.lock          # {"pid": 12345, "host": "...", "started": "..."}
  ps -p 12345                              # expected: no such process
  rm runs/m3_study_v2/study.lock
  ```
  Never remove it while `ps` shows the pid alive: two drivers on one `--out` run every pending cell twice.

The other non-zero statuses: **1** — some cells failed, the log names them after `FAILED:` with a traceback each, and a re-run retries exactly those; **3** — a record in `--out` was made at a different `--steps`/`--seq-len` (you pointed at a smoke directory; use a fresh `--out`); **5** — no episodes under `--data`; **6** — `--out` cannot be created. None of them is 2, which is argparse's own and means the command line was wrong.

- [ ] **Step 6: The report**

```bash
cd /Users/raphaelchen/Desktop/csgo-bot
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/report_study.py \
  --out runs/m3_study_v2 --figure runs/m3_study_v2/curves.png | tee runs/m3_study_v2/report.txt
echo "report exit: ${pipestatus[1]:-${PIPESTATUS[0]}}"
```

Expected exit **0 or 10, and both are results.** `0` is `EXIT_OK`: the gate PASSED — every one of the eight computed criteria holds. `10` is `EXIT_GATE_NOT_PASSED`: the report printed in full and the verdict line reads `GATE: NOT PASSED`; the spec is explicit that a null result is legitimate (§5 says outright that the feature arms' M3b losses to persistence are not expected to be fixed by this change). Anything else is not a result: `7` — no `result_*_seed*.json` in `--out` (wrong path); `8` — a record's filename disagrees with its `arm`/`seed`; `9` — a record could not be read back. The last line before the exit is `figure=runs/m3_study_v2/curves.png`; a `figure NOT written: ...` line costs the figure only and does not change the verdict.

Read the `--- training ---` table first:
```
cell               steps/s   kl_rate  kl_dyn_max  loss_last20   wall_h
frozen_ssl/s0         3.xx     0.xxx       x.xxx       x.xxxx     1.xx
...
pixel_ae/s0           3.xx     0.xxx       x.xxx       x.xxxx     1.xx
```
**Every `pixel_ae` row's `kl_rate` reads > 0.5, or the arm is not in the comparison and the rest of this report is read as a two-arm study.** Then the gate block: eight `[PASS]`/`[FAIL]` lines, one `[ n/a] invariant_tests_green`, and the verdict.

- [ ] **Step 7: Run the acceptance check — it must pass now**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python runs/m3c_check_records.py | tee runs/m3_study_v2/provenance.txt
```
Expected: PASS — the nine provenance rows, then
```
total hours 13.xxx; HEAD <sha>
OK: nine cells, one git_sha == HEAD, one device == mps
```
The rows are the raw material for the first table of Step 10. A failure here after a completed run is a finding, not a nuisance: `git_sha` split across two values means two code states, and the cells from the older one are re-run, not explained away; `device` reading `cpu` anywhere means that cell is re-run on MPS; `kl_rate` ≤ 0.5 on a `pixel_ae` cell is the §3 stop.

- [ ] **Step 8: The diagnostics and the pooled statistic, in the spec's order**

The ladder and the k-step sweep over the nine new checkpoints (spec §4). No retraining; ~30 min for nine cells on MPS (each cell refits the probe, ~20 s, then runs the three rungs and the sweep over the 229 windows). `--device mps` is the default but is passed explicitly: on cpu the shipped curves do not reproduce and the run ends **14** (`EXIT_RECORD_MISMATCH`) — an environment difference, not a code defect, and one this study does not need to buy.

```bash
cd /Users/raphaelchen/Desktop/csgo-bot
PYTHONDONTWRITEBYTECODE=1 caffeinate -dimsu .venv/bin/python -u scripts/diagnose_dynamics.py \
  --out runs/m3_study_v2 --device mps 2>&1 | tee runs/m3_study_v2/diagnostics.txt
echo "diagnose exit: ${pipestatus[1]:-${PIPESTATUS[0]}}"
```

Expected exit **0**, and the self-check table's three numeric columns **exactly `0`** on all nine rows, `noise_restored True`, `noise_same 0`:
```
arm           seed   open_loop_k  record_repro  stream_drift  k1_is_floor  probe_r2  noise_restored  noise_same
frozen_ssl       0       0.0e+00       0.0e+00       0.0e+00        False    +0.xxx            True           0
...
```
Non-zero in `record_repro` alone with the other two at zero is the environment (exit 14); non-zero in `open_loop_k` (13) or `stream_drift` (15) is a code defect and stops the task. `11` means a cell has a checkpoint but no record or vice versa — the run in Step 5 did not finish. `16` is a checkpoint whose `arm`/`seed` are not its file's. `12` means the split the diagnostic computed is not the one the record was scored on — impossible with `SPLIT_SEED = 0` on the same data, so it means `--data` points elsewhere.

`probe_r2` for the three `pixel_ae` rows is the number M3b's pixel arm read −0.036 on. Positive here is what lets the ladder read the pixel arm through its position probe at all; if it is not, the verdict block for that cell says `UNMEASURABLE THROUGH THIS PROBE` and quotes the embedding-space ratio instead, and the results section records exactly that.

Then the pooled statistic, twice — once against each control the spec's design makes available. `pool_dynamics.py` reads the per-window series the diagnostic records carry and never loads a model; each run takes under a minute (2,000-resample cluster bootstrap).

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/pool_dynamics.py --out runs/m3_study_v2 \
  --treatment frozen_ssl --control random_vit | tee runs/m3_study_v2/pool_vs_random_vit.txt
echo "pool exit: ${pipestatus[1]:-${PIPESTATUS[0]}}"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/pool_dynamics.py --out runs/m3_study_v2 \
  --treatment frozen_ssl --control pixel_ae | tee runs/m3_study_v2/pool_vs_pixel_ae.txt
echo "pool exit: ${pipestatus[1]:-${PIPESTATUS[0]}}"
```

Expected exit **0** both times, each beginning
```
--- pooled ladder: n_seeds=3 seeds per arm, n_windows=229, n_episodes=24; seeds averaged per window and clustered by episode -- ... device=mps torch=2.13.0 ---
family: 24 comparisons (3 arms x 3 rungs x 2 channels, plus 3 x 2 contrasts); family-wise threshold z=3.47, read against t(23) ...
```
`n_windows=229` and `n_episodes=24` because the split is `SPLIT_SEED = 0` on the same 122 episodes M3b used — a different count means a different `--data`. The per-arm block is identical between the two runs (it does not depend on the control); only the contrast and pathway blocks differ. `18` means a `diagnostic_<arm>_seed<n>.json` is missing (Step 8's first command did not finish); `20` means a diagnostic record predates the per-window series (you pointed at `runs/m3_study` — the M3b directory's `cnn` records are not `ARMS` members either, and would have failed the diagnostic's `--arms` choices before that); `19` means two cells do not share windows or device; `22` means a cell's noise reference measured no spread.

- [ ] **Step 9: Mutation-test — what a wrong run looks like, and which check catches it**

There is no module under test in this task, so the harness's three self-checks apply to the *run* rather than to a mutated function, and each has a real meaning here:

1. **Run at the export, not the working tree** ↔ the nine cells are produced by the *committed* tree: Step 4's `git status --porcelain -- src scripts` is empty, `__pycache__` is cleared, `PYTHONDONTWRITEBYTECODE=1` is on every command. `git_sha` is only evidence of a code state under those three conditions, and the check's `dirty` assertion enforces the first.
2. **Prove the harness on a known-fatal mutation first** ↔ Step 3: the check fails on `runs/m3_study`, a real nine-record study, before it is trusted on `runs/m3_study_v2`.
3. **No stale bytecode** ↔ Step 4 item 3 and `PYTHONDONTWRITEBYTECODE=1` throughout.

Every row below is a way the study could be wrong while `run_study.py` exits 0 and the tables look complete. Each names the check that refuses it; a row with no check would be a missing check, and there is none.

| corruption of the run | caught by |
|---|---|
| a cell run after a further commit (two code states) | `runs/m3c_check_records.py`: `git_sha across the nine records is [...]; HEAD is ...` |
| the tree edited after the run, or before it without committing | the `dirty` assertion; and the `shas == {head}` assertion once HEAD moves |
| MPS unavailable, `get_device` fell back silently | `device across the nine records is ['cpu', 'mps']`; visible at 1.5 h via Step 5 check 1 |
| a `pixel_ae` cell whose prior never trained (the M3b failure) | `kl_rate_above_free_bits=0.0008 <= 0.5`; the driver's own `WARNING: kl_dyn` line; the report's `training` table |
| a `pixel_ae` cell whose embedding target is collapsed (spec §1's 0.0008) | `embedding loss at step 0 is 0.0008, outside (0.1, 1.0)` |
| an M3b `result_frozen_ssl_seed0.json` copied into `runs/m3_study_v2` to save 1.5 h | the driver: it lacks the four new `REQUIRED_RECORD_KEYS`, so the cell is pending and is re-run over it; the check: no `git_sha` key, and `history` absent |
| an M3b `result_cnn_seed*.json` copied in | `records in runs/m3_study_v2 that are not study cells: [('cnn', 0)]`; `report_study.py`'s `every_record_is_a_study_cell` FAIL |
| a `pixel_ae` record with the wrong bottleneck width (a `random_vit` cache mislabelled) | `encoder_params=12320, expected 1056` — and before that, the loader's `BACKBONE_GEOMETRY` shape check refuses the cache at step 0 |
| a record from a `--steps 2000` spike directory reused as `--out` | the driver exits 3 (`EXIT_CONFIG_MISMATCH`); the check's `steps=2000` assertion |
| `--out` omitted from any of the four scripts | `report_study.py --out runs/m3_study`: `every_record_is_a_study_cell` FAIL and `all_nine_cells_present` FAIL; `diagnose_dynamics.py`: exit 11 (no `pixel_ae` checkpoints) ; `pool_dynamics.py`: exit 18 |
| diagnostics run on cpu | exit 14, `RECORD MISMATCH ... This run used device=cpu` |
| `ARMS` silently shrunk to two arms by a preceding task | `ARMS is (...), not ('pixel_ae', 'frozen_ssl', 'random_vit')` — the literal, not the tuple, defines "nine" |

Any corruption you can think of that no row catches is a missing assertion in `runs/m3c_check_records.py`. Add it before Step 10.

- [ ] **Step 10: Record the results honestly**

Append the section below to this plan as `## Task 8 results`, after the exit criteria, and fill every `…` from the named artefact — `runs/m3_study_v2/provenance.txt`, `report.txt`, `diagnostics.txt`, `pool_vs_random_vit.txt`, `pool_vs_pixel_ae.txt`. Numbers are copied, not rounded further than the tool printed them. **A NOT PASSED verdict is a result, not a failure to hide**; so is an `UNMEASURABLE` pixel cell, and so is a `pixel_ae` arm that beats the ViT arms or loses to them. The spec's §5 lists what this design does not claim, and the results must not claim it either.

````markdown

## Task 8 results

**Provenance.** All nine cells of `runs/m3_study_v2` were produced by **one code state** and
**one device**, as fields rather than as a reconstruction from mtimes: `git_sha` = `…`
(= `git rev-parse HEAD`, tree clean under `src/` and `scripts/`) and `device` = `mps` in
all nine records (`runs/m3c_check_records.py`: `OK: nine cells, one git_sha == HEAD, one
device == mps`). torch `2.13.0`. Driver: `…` invocation(s), `study.log` from
`…` to `…` (calendar span `…` h); total compute `…` h summed over the nine `seconds`
fields. `report_study.py` exited `…` (`0` = PASSED / `10` = `EXIT_GATE_NOT_PASSED`).
Figure: `runs/m3_study_v2/curves.png`. The spike that licensed the run (spec §3): step-0
`embedding_loss` `…`, `kl_rate_above_free_bits` `…`, cached-feature probe R² `…`.

**The M3b records in `runs/m3_study` are not compared against** (spec §5): different
`encoders.py`, different code state, the `cnn` arm retired and `pixel_ae` new. The M3b
write-up stands as the record of what M3b measured.

**The first number.** `kl_rate_above_free_bits` per cell, `pixel_ae` first because it is
the reason for the re-run — M3b's pixel arm read 0.0008 here:

| cell | steps/s | kl_rate | kl_dyn_max | loss_last20 | wall_h | embed_loss step 0 | probe_r2 |
|---|---|---|---|---|---|---|---|
| `pixel_ae`/s0 | … | **…** | … | … | … | … | … |
| `pixel_ae`/s1 | … | **…** | … | … | … | … | … |
| `pixel_ae`/s2 | … | **…** | … | … | … | … | … |
| `frozen_ssl`/s0 | … | … | … | … | … | … | … |
| `frozen_ssl`/s1 | … | … | … | … | … | … | … |
| `frozen_ssl`/s2 | … | … | … | … | … | … | … |
| `random_vit`/s0 | … | … | … | … | … | … | … |
| `random_vit`/s1 | … | … | … | … | … | … | … |
| `random_vit`/s2 | … | … | … | … | … | … | … |

`kl_rate` > 0.5 in `…` of 9 cells; `grep -c "WARNING: kl_dyn" study.log` = `…`. Read: the
pixel arm's prior `…` (trained on `…`% of steps, against the M3b arm's 0.08%), so the
three arms `…` comparable on `_kl_rate`'s own criterion.

**`gap_closed` at h=45, position (`GATE_METRIC`),** with the three provenance columns that
say one code state and one device produced every row. `+nan` is the contract for a
non-positive persistence-to-floor band. Means are `np.nanmean` over the `finite` cells.

| arm | encoder_params | git_sha | device | steps/s | seed 0 | seed 1 | seed 2 | mean (finite n/3) | unanimous > 0 |
|---|---|---|---|---|---|---|---|---|---|
| `pixel_ae` | 1056 | `…` | mps | … | … | … | … | … (…/3) | … |
| `frozen_ssl` | 12320 | `…` | mps | … | … | … | … | … (…/3) | … |
| `random_vit` | 12320 | `…` | mps | … | … | … | … | … (…/3) | … |

The `encoder_params` asymmetry (1,056 against 12,320, spec §2.2) is recorded, not hidden:
against the 9.73M-parameter RSSM and heads every arm shares byte-for-byte it is noise.

**The gate**, every criterion `evaluate_gate` reports, in its order:

| criterion | verdict | kind |
|---|---|---|
| `all_nine_cells_present` | … | accounting (harness) |
| `no_duplicate_cells` | … | accounting (harness) |
| `every_record_is_a_study_cell` | … | accounting (harness) |
| `beats_persistence` | … | spec §4.1 |
| `band_is_usable` | … | harness guard (`MAX_DEGENERATE_STEPS = 0`, both metrics) |
| `filtering_beats_embedding` | … | spec §4.4 |
| `reward_reported` | … | spec §4.3 (reported, deliberately not gated) |
| `curves_produced` | … | spec §4.2 |
| `invariant_tests_green` | NOT EVALUATED by the gate; externally: `… passed`, exit 0 | spec §4.5 |
| **GATE** | **…** (exit `…`) | |

Per-cell detail for the failing criteria (`band_is_usable`: degenerate steps per cell and
metric; `filtering_beats_embedding`: which cells read True, with margin): `…`

**Throughput and cost** (`steps_per_second` is training only; `seconds` is the whole cell):

| arm | steps/s per seed | mean steps/s | hours (3 seeds) | share |
|---|---|---|---|---|
| `pixel_ae` | … / … / … | … | … | …% |
| `frozen_ssl` | … / … / … | … | … | …% |
| `random_vit` | … / … / … | … | … | …% |
| **total** | — | — | **…** | vs the spec's ~13.5 h |

**Diagnostics** (`diagnose_dynamics.py`, exit `…`). Self-checks: `open_loop_k`,
`record_repro`, `stream_drift` all exactly `0.0` on 9/9 cells; `noise_restored` True and
`noise_same` 0 on 9/9: `…` (yes/no — if no, which cell and which column). The ladder, position,
horizon-mean delta (intervened − real) ± 2 SE, episode-clustered; family-wise threshold
z = `…` over `…` comparisons:

| cell | shuffled (order) | resampled (counts) | held FWD − held NOOP | z | verdict |
|---|---|---|---|---|---|
| `pixel_ae`/s0 | … | … | … | … | … |
| `pixel_ae`/s1 | … | … | … | … | … |
| `pixel_ae`/s2 | … | … | … | … | … |
| `frozen_ssl`/s0 | … | … | … | … | … |
| `frozen_ssl`/s1 | … | … | … | … | … |
| `frozen_ssl`/s2 | … | … | … | … | … |
| `random_vit`/s0 | … | … | … | … | … |
| `random_vit`/s1 | … | … | … | … | … |
| `random_vit`/s2 | … | … | … | … | … |

The pixel arm's verdict is `…` — M3b's read `UNMEASURABLE THROUGH THIS PROBE` on all three
cells because its probe was a constant (`probe_r2` −0.036); with `probe_r2` = `…` here it
`…`. The k-step re-grounding sweep, position error at h=45:

| cell | floor | k=1 | k=3 | k=5 | k=15 | k=45 | persistence |
|---|---|---|---|---|---|---|---|
| `pixel_ae`/s0 | … | … | … | … | … | … | … |
| … (all nine) | | | | | | | |

**The pooled statistic** (`pool_dynamics.py`; `n_windows=229`, `n_episodes=24`, seeds
averaged per window, episode-clustered; z read against t(23); family threshold z = `…`).
Per arm × rung, position delta ± 2 SE (z), and the embedding ratio [95% cluster bootstrap]:

| arm | shuffled | resampled | held FWD − NOOP | embedding ratio: shuffled / resampled / contrast |
|---|---|---|---|---|
| `pixel_ae` | … (z …) | … (z …) | … (z …) | … / … / … [… , …] |
| `frozen_ssl` | … (z …) | … (z …) | … (z …) | … / … / … [… , …] |
| `random_vit` | … (z …) | … (z …) | … (z …) | … / … / … [… , …] |

The two between-arm contrasts, paired per window, held contrast row; the position line is
the pathway as read through each arm's own probe, the embedding line is probe-free:

| contrast | position delta ± 2 SE (z) | embedding ratio contrast [95% CI] | pathway reading (own-probe / probe-free) |
|---|---|---|---|
| `frozen_ssl` − `random_vit` | … | … | … / … |
| `frozen_ssl` − `pixel_ae` | … | … | … / … |

**What this run establishes, and what it does not.** `…` — stated against spec §5's three
non-claims: it does not show `pixel_ae` is the right pixel baseline in the abstract, only a
fair one under this objective; it does not claim the feature arms' M3b failures are fixed;
it does not compare against M3b's records.
````

- [ ] **Step 11: Commit**

Only the plan. `runs/` is gitignored and stays so — the records (~5 MB each with the full history), the checkpoints, the diagnostics, the figure and the logs are not committed, which is why every number the section quotes is in the section.

```bash
cd /Users/raphaelchen/Desktop/csgo-bot
git status --porcelain            # expected: only the plan file modified (and M3b's untracked study.log, untouched)
git add docs/superpowers/plans/2026-09-11-mb-fps-m3c-pixel-ae-arm.md
git commit -m "docs: M3c nine-cell run under one code state -- gate verdict, diagnostics and pooled statistic

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Exit criteria for this plan

- [ ] `pytest` fully green, with the counts each task states.
- [ ] `patch_dim` exists nowhere outside `BACKBONE_GEOMETRY` and `BottleneckEncoder`'s attributes (`grep -rn "patch_dim" src scripts`).
- [ ] `encoder_input_kind` is `"features"` for every arm in `ARMS`; `cnn` still builds via `KINDS` and M2's three scripts accept it.
- [ ] Every mutation table row is caught, using a harness self-checked all three ways.
- [ ] `data/my_way_home/*.features_pixel_ae.npy`: 122 files, `(T+1, 64, 32)` float16, per-dim std > 0.
- [ ] The spike's three checks PASS (Task 7 results filled in). **If any fails, the study does not run.**
- [ ] Nine records in `runs/m3_study_v2`, one per (arm, seed), every one carrying `git_sha`, `device`, `encoder_params`, `history`; all nine share one `git_sha` and one `device`.
- [ ] `kl_rate_above_free_bits > 0.5` for all three `pixel_ae` cells — the first thing read from the new study.
- [ ] The gate verdict is recorded per criterion, whatever it is.
- [ ] `encoders.py` still the only arm-varying module (`grep -rn "cfg.arm\|\.kind" src/ --include='*.py' | grep -v encoders.py` hits only log prefixes, filenames and record labels).
- [ ] The M3b records in `runs/m3_study` are untouched and uncompared.

---

## Task 5 results

Written 2026-09-12 by `scripts/cache_features.py --backbone pixel_ae --checkpoint
runs/m2_fixed/autoencoder_cnn.pt --device mps` under `caffeinate -dimsu`; log
`runs/m3_study_v2/cache_pixel_ae.log`. Disk guard printed `needs=0.24 GB`, not the ViTs'
2.93 GB. The check is Step 8's script, run over all 122 files: every file `(T+1, 64, 32)`
float16, finite, and its `T+1` equal to the episode's own `obs` count.

| quantity | value |
|---|---|
| cache size on disk (`du -ch data/my_way_home/*.features_pixel_ae.npy \| tail -1`) | 233M (`size_gb=0.244` from `nbytes`; 244,256,768 B = 59,633 × 64 × 32 × 2) |
| files / frames / rows | 122 / 59,633 / (64, 32) |
| per-dim std min / p05 / median / max, dims below 0.01 | 0.4534 / 0.8496 / 1.2044 / 4.4148, 0 |
| global std (DINOv2 cache 2.3559, random_vit 1.0000, for the LayerNorm rationale) | 1.4840 |
| encode wall time and frames/s from the script's last two lines | `elapsed_s=14`, `frames_per_second=4309` (MPS, batch 32) |
| `git rev-parse HEAD` the cache was written at | `306d2c4a961eab02f55cda9583d0d5182090cc0f` |

The target is not born collapsed: the smallest per-dim std over the whole buffer is 0.4534,
three decades above the ~0.0007 a random-init `CNNEncoder` emits (§1), and the median 1.2044
sits beside the 1.06 measured on the first three episodes before the cache existed. Both
floors of the Step 8 check (`min > 0.01`, `median > 0.1`) clear with two decades to spare.
This is the direct measurement that licenses the Task 7 spike.

## Task 7 results

**Run:** `runs/m3c_spike/spike_pixel_ae_seed0.json`, log `runs/m3c_spike/spike.log`.
git SHA `b4e903aa6808b6e583a54c4ad78ac244e5483e6e` · device `mps` · 2,000 steps · seq_len 64 · batch 16 · 4.00 steps/s · 500.5 s wall.
Run 2026-09-12 under `caffeinate -dimsu`; exit status 0; `nonfinite` `{}`; `history["loss"]` and
`history["parts"]` both 2,000 entries long, on disk in full.
Probe protocol: `fit_probe` ridge selection on the flattened `(T+1, 2048)` cache; fit 16 / select 4 training
episodes, scored on 20 validation episodes of `episode_split(seed=0)`, 9,969 rows; selected ridge 1e3;
selection R² 0.3458.

| check | value | must read | verdict |
|---|---|---|---|
| 1 `embedding_loss` at step 0 (`history["parts"][0]["embedding"]`) | 0.3602 | in [0.1, 1.0] | PASS |
| 2 `kl_rate_above_free_bits` over 2,000 steps (`kl_dyn_max` 1.3457) | 0.5865 | > 0.5 | PASS |
| 3 held-out probe R² on the cached `pixel_ae` features | +0.3052 | > 0 | PASS |

**Side-by-side, embedding loss at step 0** (one forward pass each, seed 0, same split and loader draw):

| arm | this run | M3c design §1 |
|---|---|---|
| `pixel_ae` | 0.360220730304718 (side-by-side) / 0.360220730304718 (history) | — (new) |
| `frozen_ssl` | 0.3165 | 0.3165 |
| `random_vit` | 0.4104 | 0.4104 |
| `cnn` (retired) | not run | 0.0008 |

The two `pixel_ae` step-0 numbers are bit-identical on MPS, so the check judged the number the
side-by-side printed (the one mutation the CPU tests cannot see). `frozen_ssl` and `random_vit`
reproduce §1 to every printed decimal: the caches, the split and the seeding are the ones §1 measured.

Read from the full history, for the record: the dyn KL starts at 0.0338 (step 0), first clears the
0.20 floor at step 11, and sits at 0.40–0.55 over the last ten steps; the embedding loss falls from
0.3602 to 0.1938 and the total loss from 1.155 to 0.436. The training log printed no free-bits
warning, and the check-2 rate of 0.5865 is a margin of 0.0865 over the floor of 0.5, not a wide one:
it says the prior trained on 1,173 of 2,000 steps, and Task 8's first read of the new study
(`kl_rate_above_free_bits > 0.5` for all three `pixel_ae` cells, at 20,000 steps) is where that
margin gets measured properly.

**Decision:** `RUN_STUDY` (the script's own verdict; exit status 0).

Decision rule, spec §3:
- all three PASS → **Task 8**: run the nine cells into `runs/m3_study_v2`.
- check 2 FAILS alone → **STOP**. Do not run the study. Re-plan `KL_FREE_BITS` as a genuine
  three-arm calibration with the KL trajectory recorded per step; `history["parts"][i]["kl_dyn"]`
  from this record is the `pixel_ae` trajectory for it.
- check 1 or 3 FAILS (with or without 2) → **STOP**. The frozen `pixel_ae` backbone is not
  informative under this objective; §1's argument says no floor can fix that. Do not run the study;
  the next step is a different backbone (the `pixel_ae_conv` alternative in §2.1 is the named
  candidate), not a different floor.

All three passed: Task 8 runs.
