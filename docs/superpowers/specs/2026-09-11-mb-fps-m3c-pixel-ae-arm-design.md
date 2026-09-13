# MB-FPS M3c: Retire the end-to-end pixel arm, add `pixel_ae` — Design

**Status:** approved design, 2026-09-11
**Governs:** the M3 re-run (`runs/m3_study_v2`)
**Predecessor:** `docs/superpowers/plans/2026-09-04-mb-fps-m3b-study.md`, sections
"## M3 results" and "### Addendum (2026-09-11)"
**Supersedes:** the `cnn` arm of `docs/superpowers/specs/2026-09-03-mb-fps-m3-world-model-design.md`

---

## 1. The measured fact this design exists to fix

The M3b study's pixel arm never entered the comparison. Its dynamics KL cleared the 0.20-nat
free-bits floor on 15–18 of 20,000 steps in every seed, its dynamics prior never received a
gradient, and its position probe collapsed to a constant (R² −0.036, identical to five decimals
across three seeds). The obvious remedy — recalibrate `KL_FREE_BITS`, which was measured on a
2,000-step `random_vit` run alone — was scoped and then measured before being run. It cannot work.

At initialisation, on real data, the embedding loss each arm is asked to minimise:

| arm | `embedding_loss` at step 0 | dyn KL at step 0 |
|---|---|---|
| `cnn` | **0.0008** | 0.022 |
| `frozen_ssl` | 0.3165 | 0.032 |
| `random_vit` | 0.4104 | 0.038 |

The pixel arm's target is born collapsed. A randomly-initialised CNN through the bottleneck emits
a near-constant embedding, so predicting it is already trivial — ~400× easier than predicting
frozen DINOv2 or ViT features. Nothing then forces the posterior to encode anything, so it stays
at the prior, the KL stays at its init value below the floor, and the `rep` term — the encoder's
only non-degenerate gradient source, since the embedding loss is `mse(pred, embeddings.detach())`
and the reward/continue targets are near-constant — stays clamped at exactly `0.00e+00`. The
encoder never moves; the target stays collapsed. A fixed point from the first step.

Measured per-term gradient norm reaching each arm's encoder at step 0 (MPS, batch 16 × 64):

| term | `cnn` | `frozen_ssl` | `random_vit` |
|---|---|---|---|
| embedding (detached) | 1.0e-07 | 1.1e-05 | 2.6e-05 |
| reward | 7.4e-06 | 5.8e-05 | 6.5e-05 |
| continue | 4.1e-05 | 4.0e-04 | 4.5e-04 |
| KL, clamped at 0.20 | **0.0** | **0.0** | **0.0** |
| KL `rep`, unclamped | 2.1e-04 | 5.1e-03 | 1.6e-02 |

Lowering the floor unclamps `rep`, which pulls the posterior *toward* the prior — a regulariser,
pushing further into collapse. No value of `KL_FREE_BITS` creates a learning signal the pixel arm
does not have. The feature arms escape because their target is informative from step 0: the
posterior has to encode it, diverges from the prior, the KL climbs past 0.20 within the first few
hundred steps, and `rep`/`dyn` unclamp.

**The end-to-end pixel arm was structurally doomed under this objective, not a fair baseline.**
It needs what the other two arms already have: an informative embedding from step 0, from a frozen
backbone.

## 2. The design

### 2.1 A third frozen backbone: the M2 pixel autoencoder

M2 trained a pixel autoencoder per arm; the `cnn` one is `runs/m2_fixed/autoencoder_cnn.pt`
(encoder 26.38M params: four stride-2 convs to `(256, 7, 7)`, then `project: Linear(12544 → 2048)`).
Its `project` layer already *is* a trained bottleneck to the shared 2048-d space. That encoder
becomes the third frozen backbone, exactly as DINOv2 and the random ViT are for the other two arms.

`scripts/cache_features.py --backbone pixel_ae` loads the checkpoint, freezes the full encoder
(conv + project), runs it over every frame of every episode, and writes
`<episode>.features_pixel_ae.npy` of shape `(T+1, 64, 32)` — the 2048-d output partitioned into
64 rows of 32. `feature_suffix` already namespaces any non-default backbone, so this cache cannot
collide with the two that exist. Cost: 122 episodes × 526 frames through a 26M-param conv on MPS,
~5 min; 0.244 GB on disk (float16, measured: 122 x (T+1, 64, 32)).

The partition into `(64, 32)` invents nothing — it is a reshape of a vector whose 2048 dimensions
have no spatial meaning, into the row count the shared bottleneck expects. The alternative,
caching the `(256, 7, 7)` conv map interpolated to an 8×8 grid as 64 spatial patches of 256, gives
more "honest" patches at the cost of inventing spatial positions the network never computed. It
is not taken. If a reviewer wants it, it is a second backbone name (`pixel_ae_conv`), not a change
to this one.

### 2.2 Patch geometry is a property of the backbone, not the arm

Today `EncoderConfig.patch_dim = 384` is a shared field consumed in exactly two lines of
`BottleneckEncoder` — the `Linear(patch_dim, bottleneck_dim)` and the optional
`LayerNorm(patch_dim)` — and `_N_PATCHES = 64` is a module constant. Both describe the ViT
backbones' output, not a design choice of the study. They move to a backbone registry:

```python
BACKBONE_GEOMETRY: dict[str, tuple[int, int]] = {
    "dinov2":     (64, 384),
    "random_vit": (64, 384),
    "pixel_ae":   (64, 32),
}
```

`BottleneckEncoder` takes its `(n_patches, patch_dim)` from the registry for the arm's backbone;
`bottleneck_dim = 32` and `embed_dim = 2048` stay shared in `EncoderConfig`; the guard
`n_patches * bottleneck_dim == embed_dim` stays and now runs per backbone. `patch_dim` is removed
from `EncoderConfig` — it was never an arm-level choice, and leaving a dead field is how a future
arm gets built against the wrong width. The loader's feature-shape validation reads the same
registry, so a cache whose rows do not match its backbone's geometry is refused at load, not
discovered as a matmul error at step 0.

Every arm is then literally the same pipeline: frozen backbone → cached `(n_patches, patch_dim)`
rows → per-row `LayerNorm` (no parameters) → the same `BottleneckEncoder` class → 2048. The arms
differ in exactly one thing: what the backbone was pretrained on — pixel reconstruction on this
data, DINOv2 self-supervision, or nothing.

**One asymmetry, stated so it is not discovered:** `pixel_ae`'s bottleneck is
`Linear(32 → 32)` = 1,056 parameters; the ViT arms' is `Linear(384 → 32)` = 12,320. Same class,
same per-row treatment, different input width. Against the 9.73M-parameter RSSM and heads that
all arms share byte-for-byte it is noise, and it is recorded here and as an `encoder_params`
integer in every study record rather than hidden.

### 2.3 `cnn` is retired from the study, not deleted from the codebase

```python
ARMS: tuple[str, ...] = ("pixel_ae", "frozen_ssl", "random_vit")
"""The study's arms. Every M3 tool -- the driver, report, aggregation, rollout
evaluation, diagnostics, pooling -- takes `choices=ARMS`."""

KINDS: tuple[str, ...] = ("cnn",) + ARMS
"""Every encoder `build_encoder` can construct. `get_config` validates against
this, not `ARMS`, because M2's autoencoder scripts and tests still build the
end-to-end `CNNEncoder` under the name `cnn` and must keep doing so."""
```

Two tuples, not one, because two milestones read them. `cnn` stays buildable: `build_encoder`,
`encoder_input_kind` (`"obs"`) and `_ARM_BACKBONE` (`None`) keep their `cnn` branches, and M2's
`train_autoencoder.py`, `reconstruction_grid.py` and `eval_reconstruction.py` take
`choices=KINDS`. It cannot be *selected* by any M3 tool, because those take `choices=ARMS`. A study
record, checkpoint or diagnostic that says `arm="cnn"` continues to mean the end-to-end pixel arm
of M3b, and nothing can read a `pixel_ae` artefact under the old name or vice versa.

The `_SLOW_ARMS` cost ordering in `scripts/run_study.py` becomes empty: every study arm trains at
feature-arm speed.

Test impact, so it is planned rather than discovered: ~280 references to `"cnn"` in the aggregate,
driver, diagnostics and pooling tests use it as an arm *label* in fixtures and become
`"pixel_ae"`; the `CNNEncoder` tests in `tests/models/test_encoders.py` and all of
`tests/training/test_autoencoder.py` mean the encoder class and keep it; `tests/training/
test_world_model.py`'s `tiny()` fixture keeps `cnn` as its default because the RSSM and loss tests
are indifferent to which encoder feeds them and a pixel encoder needs no feature file, while its
`test_all_three_arms_train` is parametrised over `ARMS` and writes `(41, 64, 32)` features for
`pixel_ae`.

### 2.4 What does not change

The RSSM, the heads, `TrainConfig` (batch 16, seq 64, lr 1e-4, 20,000 steps), `KL_FREE_BITS =
0.20`, the KL scales, the split seed, the evaluation protocol (context 5, horizon 45), and every
gate criterion. The record schema changes only additively (§4). The six finished `frozen_ssl` and `random_vit` cells are
retrained in the new study, not reused — the arms must be produced by one code state, and this
change touches `encoders.py`, which is on every arm's path.

## 3. The spike that replaces the calibration

Before the nine-cell run: 2,000 steps of `pixel_ae`, seed 0, into a scratch `--out`, ~10 min at
feature-arm speed. It is a calibration on an informative target, which the original 2,000-step
`random_vit` spot check (spec §3.2a) was; the search for a floor that would unlock a dead arm was
not. A result that licenses the nine-cell run must show all three of:

| check | must read | why |
|---|---|---|
| `embedding_loss` at step 0 | in the feature-arm band, 0.1–1.0 | the target is informative; §1's 0.0008 is the failure |
| `kl_rate_above_free_bits` over 2,000 steps | > 0.5 | the prior trains; the arms are comparable on `_kl_rate`'s own criterion |
| held-out probe R² on the cached `pixel_ae` features | > 0 | the frozen features carry position at all (DINOv2 0.422, random_vit 0.369 at §3.2a) |

Only if the second fails does `KL_FREE_BITS` get revisited, and then as a genuine three-arm
calibration with the KL trajectory recorded per step. The spike persists `history["loss"]` and
`history["parts"]` in full — the M3b write-up's open item 6, which costs one line and would have
made §1 a measurement instead of a reconstruction.

## 4. The run

Nine cells into `runs/m3_study_v2`: three arms × seeds {0, 1, 2}, 20,000 steps, MPS, all at
~1.5 h per cell — **~13.5 h**, against the 38 h the free-bits re-run would have cost with the
pixel arm at 9 h per cell. Then, in order: `scripts/report_study.py` on the new `--out`;
`scripts/diagnose_dynamics.py` over the nine new checkpoints; `scripts/pool_dynamics.py`. The
record schema gains `git_sha` and `device` (the M3b write-up's other half of open item 6), so
"one code state produced all nine cells" is a field, not a reconstruction from file mtimes.

## 5. What this design does not claim

- It does not claim `pixel_ae` is the "right" pixel baseline in the abstract. It claims it is a
  *fair* one under this objective, which the end-to-end arm was measured not to be.
- It does not claim the feature arms' M3b failures are fixed. They lose to persistence for a
  reason that is not the encoder (M3b addendum: a coarse, physically-signed action pathway that
  cannot resolve sequence). The re-run reproduces that under one code state with a live pixel
  arm; it does not change the shared RSSM or its training.
- It does not compare against the M3b records. Different `encoders.py`, different code state, one
  arm renamed. The M3b write-up stands as the record of what M3b measured.

## 6. Files

| file | change |
|---|---|
| `src/mbfps/data/features.py` | `BACKBONES` gains `"pixel_ae"`; `build_backbone` loads and freezes the M2 encoder; `FeatureExtractor` emits `(64, 32)` rows for it; `BACKBONE_GEOMETRY` is defined here, next to `BACKBONES` |
| `src/mbfps/models/encoders.py` | `BottleneckEncoder` reads geometry from the registry; `_N_PATCHES` and `cfg.patch_dim` removed; `_KIND_BACKBONE` (was `_ARM_BACKBONE`), `build_encoder`, `encoder_input_kind` gain `pixel_ae`; keep `cnn` (buildable under `KINDS`, not selectable by any M3 tool -- section 2.3) |
| `src/mbfps/utils/config.py` | `ARMS`, `KINDS`; `get_config` validates against `KINDS`; `EncoderConfig.patch_dim` removed; `cnn_depth` stays (M2) |
| `scripts/train_autoencoder.py`, `reconstruction_grid.py`, `eval_reconstruction.py` | `choices=KINDS` (M2 keeps building `cnn`) |
| `src/mbfps/data/loader.py` | feature-shape validation against the registry |
| `src/mbfps/eval/study.py` | record gains `git_sha`, `device`, `encoder_params`, and the full loss/parts history |
| `scripts/cache_features.py` | `--backbone pixel_ae` |
| `scripts/run_study.py` | `_SLOW_ARMS` |
| tests mirroring each of the above | every guard mutation-tested; every arm exercised |
