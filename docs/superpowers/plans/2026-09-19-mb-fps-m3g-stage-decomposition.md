# MB-FPS M3g — The Stage Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the nine shipped M3c checkpoints, measure four stages of imagination in the model's own 32 × 32 categorical latent — *encode*, *predict*, *carry*, *decode* — and name, per arm, the first stage that fails, with two known-blind rung-4000 cells as a control the instrument must read `ENCODE_FAILS` on or refuse.

**Architecture:** The canonical diagnostic pass (`diagnostics._diagnose`) gains a `keep_latents` flag that keeps the posterior logits over the whole window, the teacher-forced prior logits (which `observe` already computes), the open-loop prior logits, and two embedding-space companions — on the canonical pass only, drawing nothing new from the stream, so the trust self-check covers the rollout the stages read. A pure module (`mbfps/eval/stages.py`) turns those arrays into per-window statistics against latent-persistence and marginal baselines, and decides the four rules in order, each status carrying its sentence. One script (`scripts/stage_decomposition.py`) loads cells through `trust_horizon.prepare_cell` and controls through `checkpoint_ladder.rung_cell`, writes one record per cell, pools through `pooling.pool_arm`, refuses on the control, prints the tables and writes `stages.txt` and a figure. Then one smoke run, one real run, one results section.

**Tech Stack:** Python 3.12, torch 2.13 (MPS), numpy, matplotlib (figure only), pytest. Spec: `docs/superpowers/specs/2026-09-19-mb-fps-m3g-stage-decomposition-design.md`.

## Global Constraints

- **Pre-registered constants, never tuned after the run:** `DECISION_H = 15` (imported from `split_gap`); `FAMILY = 5`; `z_fam = pooling.cluster_threshold(5, 24) = 2.81` on the shipped split; `SEEDS_REQUIRED = 2`; the encode floor is `rssm.KL_FREE_BITS = 0.20` (imported, not restated); `REPORTED_K = (1, 2, 3, 5, 10, 15, 30, 45)`; `STAGES = ("encode", "predict", "carry", "decode")` in that order; statuses `ENCODE_FAILS → PREDICT_FAILS → CARRY_FAILS → DECODE_FAILS → NO_STAGE_FAILS`.
- **The control:** `CONTROL_ARM = "frozen_ssl"`, `CONTROL_SEEDS = (1, 2)`, `--control runs/m3f_ladder/step4000`; it must read `ENCODE_FAILS` or the run exits **34 `EXIT_CONTROL_MISREAD`** with no arm's reading printed and no `stages.txt` written. A control with fewer than `SEEDS_REQUIRED` seeds is a usage error (argparse 2): with one seed nothing can pass, so `ENCODE_FAILS` would be vacuous.
- **A stage passes** when its pooled contrast(s) clear `z_fam` (strictly; NaN and ±inf never clear) and its per-seed contrast(s) clear in ≥ `SEEDS_REQUIRED` seeds; *predict* needs both its contrasts, and a seed holds predict only when both clear within it. The status is the first stage in `STAGES` not passed. Sentences say `passes` / `failed` (estimate ≤ 0 or z < −`z_fam`) / `not shown` (otherwise), name the first contrast not clearing (predict: persistence before marginal), and list the seeds holding. A single-seed leaf (`per_seed=None`) has a vacuous replication clause ("this seed alone").
- **The pass:** `keep_latents` defaults to False and every existing field of the pass and of `Trajectories` stays bitwise what it was with the flag on or off; the flag draws nothing from the stream; `keep_latents=True` requires `keep_trajectories=True`; the step-1 identity `prior_open_logits[:, 0] == prior_teacher_logits[:, 0]` (bitwise) is asserted where the fields are kept, as `LatentIdentityError` (a `ValueError`) naming the window start.
- **Exit codes:** 0; 11 / 12 / 14 / 30 with `trust_horizon.py`'s meanings, imported; 34 new; argparse 2 for `--out` equal to `--reference` or `--control`, or fewer than two distinct control seeds.
- **Nothing is retrained; nothing under `runs/` is ever removed; the M3c, M3d, M3e and M3f records and every recorded verdict are untouched.** Results live in this plan's `## Task 7 results`; the earlier plans are closed and never amended. Never `git stash`. Do not commit while a run is in progress (`git_sha()` is stamped into records at write time).
- **Style:** match the neighbouring modules' docstring register (the *why*, not the *what*); tests carry hand-typed expected values, one rule mutated per test; no test asserts nothing; every commit ends with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- **Run from the worktree root** with `.venv/bin/python -m pytest ...`. `runs/` and `data/` are symlinks to the main checkout.

---

## File Structure

| file | responsibility |
|---|---|
| `src/mbfps/eval/diagnostics.py` (modify) | `keep_latents` on `_diagnose` / `reference_trajectories`; five new `_Pass` and `Trajectories` fields in `_LATENT_FIELDS`; `LatentIdentityError`; the step-1 identity |
| `src/mbfps/eval/stages.py` (new) | pure statistics (KL, modes, accuracies, baselines, NLL, entropy, decode margin) and the reading (`ArmInputs`, `StagesInputs`, `Status`, `StageResult`, `ArmReading`, `StagesReading`, `reading_stages`, `format_reading_stages`) |
| `scripts/stage_decomposition.py` (new) | `evaluate`: cells + controls → `stages_*.json`; `read`: pooling glue, the control rule, the tables, the figure, `stages.txt`; `main` |
| `tests/eval/test_diagnostics.py` (modify) | the `keep_latents` pins |
| `tests/eval/test_stages.py` (new) | the pure functions and the reading, on fabricated logits and contrasts |
| `tests/eval/test_stage_decomposition_script.py` (new) | the script on the ladder tests' fixtures: every exit code, the records, `stages.txt` |
| `tests/eval/test_diagnose_dynamics_script.py` (modify) | the distinctness test gains `stage_decomposition` (34) |

Tasks 1–3 are pure and independent of the run; Task 4 is `evaluate`; Task 5 is `read`; Task 6 is the distinctness test and the smoke; Task 7 is the real run and the results.

---

### Task 1: `keep_latents` on the canonical pass

**Files:**
- Modify: `src/mbfps/eval/diagnostics.py` — `_Pass` (after `true_embedding_displacement`, ~line 474), `_TRAJECTORY_FIELDS` (~line 476, add `_LATENT_FIELDS` after it), `_diagnose` (signature ~line 500, body ~lines 574–576 and 686–717, return ~line 750), `Trajectories` (~line 767), `reference_trajectories` (~line 820)
- Test: `tests/eval/test_diagnostics.py` (append at the end)

**Interfaces:**
- Consumes: `_diagnose`, `_Pass`, `Trajectories`, `reference_trajectories`, `_embedding_distance`, `_TRAJECTORY_FIELDS` as they are.
- Produces: `_diagnose(..., keep_trajectories=False, keep_latents=False)`; `_LATENT_FIELDS = ("post_logits", "prior_teacher_logits", "prior_open_logits", "posterior_rendering_distance", "true_step_displacement")`; `Trajectories.post_logits` `(n, context + horizon, G, C)` float32, `.prior_teacher_logits` and `.prior_open_logits` `(n, horizon, G, C)` float32, `.posterior_rendering_distance` and `.true_step_displacement` `(n, horizon)` float64, all `None` unless kept; `reference_trajectories(..., keep_latents: bool = False)`; `class LatentIdentityError(ValueError)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_diagnostics.py`:

```python


# ---------------------------------------------------------------------------
# The latent fields (M3g): `keep_latents` on `_diagnose` and
# `reference_trajectories`. The stage decomposition reads the posterior over
# the whole window, the teacher-forced prior (`observe` computes it already)
# and the open-loop prior, in the model's own categorical latent. Two things
# can go wrong without breaking a shape: the flag can move a field the trust
# pass pins bitwise, and the open-loop prior at step 1 can stop being the
# teacher-forced prior at step 1 -- which would mean the two passes no longer
# start from the same context state. The real sampling rig pins both.
# ---------------------------------------------------------------------------

LATENT_FIELDS = (
    "post_logits", "prior_teacher_logits", "prior_open_logits",
    "posterior_rendering_distance", "true_step_displacement",
)
"""Literal, deliberately NOT `diagnostics_module._LATENT_FIELDS`: a field
dropped from the module's tuple must be a missing attribute here, not a
shorter loop."""


def latent_pass(model, paths, probe, *, keep, device=None, seed=0, trajectories=True):
    """`_diagnose` as `reference_trajectories` calls it, with the latent flag chosen."""
    with torch.no_grad():
        return diagnostics_module._diagnose(
            model, paths, probe, arms={}, context=CONTEXT, horizon=HORIZON,
            seed=seed, device=device or torch.device("cpu"), feature_backbone=None,
            noise_reference=True, keep_trajectories=trajectories, keep_latents=keep,
        )


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_keep_latents_leaves_every_existing_field_bitwise_and_fills_the_five(tmp_path, device):
    """The trust pass's pin, from the other side: the latent flag is the ONLY
    difference between two passes on the real sampling rig, and every field
    the trust pass reads -- the nine trajectory arrays, the six curves, the
    six per-window matrices, the noise reference and its two self-checks,
    the counts and the labels -- must be bitwise the same across them, and
    the generator must end at the same state, so the flag drew nothing from
    the stream. Without the flag all five latent fields are None; with it
    none of them is."""
    model, paths, probe = _two_episode_rig(tmp_path, device)

    plain = latent_pass(model, paths, probe, keep=False, device=device)
    left_by_plain = diagnostics_module._rng_snapshot(device)
    kept = latent_pass(model, paths, probe, keep=True, device=device)
    left_by_kept = diagnostics_module._rng_snapshot(device)

    for name in LATENT_FIELDS:
        assert getattr(plain, name) is None, name
        assert getattr(kept, name) is not None, name
    for name in TRAJECTORY_FIELDS:
        np.testing.assert_array_equal(getattr(kept, name), getattr(plain, name), err_msg=name)
    for name in ("horizon", "rssm_position", "persistence_position", "floor_position",
                 "rssm_angle", "persistence_angle", "floor_angle"):
        np.testing.assert_array_equal(
            getattr(kept.reference, name), getattr(plain.reference, name), err_msg=name
        )
    for name, rows in plain.reference_windows.items():
        np.testing.assert_array_equal(kept.reference_windows[name], rows, err_msg=name)
    np.testing.assert_array_equal(kept.noise_embedding, plain.noise_embedding)
    np.testing.assert_array_equal(kept.noise_bitwise_real, plain.noise_bitwise_real)
    np.testing.assert_array_equal(kept.noise_stream_restored, plain.noise_stream_restored)
    assert kept.windows_total == plain.windows_total == 4
    np.testing.assert_array_equal(kept.window_episode, plain.window_episode)
    assert set(left_by_kept) == set(left_by_plain)
    for key in left_by_plain:
        assert torch.equal(left_by_kept[key], left_by_plain[key]), key


def test_the_latent_flag_is_keyword_only_off_by_default_and_needs_the_trajectories(tmp_path):
    """Off by default so the ladder, the sweep and the trust pass keep running
    the pass they pin bitwise. The latent fields are read beside the
    trajectory rows (the decode stage is the free-channel margin), so asking
    for them without the rows is refused before any window is cut."""
    parameter = inspect.signature(diagnostics_module._diagnose).parameters["keep_latents"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False
    parameter = inspect.signature(reference_trajectories).parameters["keep_latents"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False
    model, paths, probe = _two_episode_rig(tmp_path, torch.device("cpu"))
    with pytest.raises(ValueError, match="keep_trajectories"):
        latent_pass(model, paths, probe, keep=True, trajectories=False)


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_the_latent_fields_have_their_shapes_and_the_step_one_identity_holds(tmp_path, device):
    """Four windows, three context frames, five horizon steps, the real
    32 x 32 latent: the posterior spans the whole window `(4, 8, 32, 32)`,
    the two priors the horizon `(4, 5, 32, 32)`, both float32 as the model
    emits them; the two distances `(4, 5)` float64 like the trajectory rows.
    The open-loop prior at step 1 is the teacher-forced prior at step 1
    BITWISE -- both are `prior_net` on the same `h` after the same context
    state and the same action -- and `Trajectories` carries the same five
    arrays bitwise when asked, None when not."""
    model, paths, probe = _two_episode_rig(tmp_path, device)
    groups, classes = model.rssm.cfg.z_cats, model.rssm.cfg.z_classes
    kept = latent_pass(model, paths, probe, keep=True, device=device)

    assert kept.post_logits.shape == (4, CONTEXT + HORIZON, groups, classes)
    assert kept.prior_teacher_logits.shape == (4, HORIZON, groups, classes)
    assert kept.prior_open_logits.shape == (4, HORIZON, groups, classes)
    for name in ("post_logits", "prior_teacher_logits", "prior_open_logits"):
        assert getattr(kept, name).dtype == np.float32, name
        assert np.isfinite(getattr(kept, name)).all(), name
    for name in ("posterior_rendering_distance", "true_step_displacement"):
        assert getattr(kept, name).shape == (4, HORIZON), name
        assert getattr(kept, name).dtype == np.float64, name
        assert (getattr(kept, name) >= 0).all() and np.isfinite(getattr(kept, name)).all(), name
    np.testing.assert_array_equal(kept.prior_open_logits[:, 0], kept.prior_teacher_logits[:, 0])
    # The posterior is not the prior: the posterior saw the frame.
    assert not np.array_equal(kept.post_logits[:, CONTEXT:], kept.prior_teacher_logits)

    with torch.no_grad():
        with_latents = reference_trajectories(
            model, paths, probe, context=CONTEXT, horizon=HORIZON, seed=0, device=device,
            feature_backbone=None, keep_latents=True,
        )
        without = reference_trajectories(
            model, paths, probe, context=CONTEXT, horizon=HORIZON, seed=0, device=device,
            feature_backbone=None,
        )
    for name in LATENT_FIELDS:
        np.testing.assert_array_equal(getattr(with_latents, name), getattr(kept, name), err_msg=name)
        assert getattr(without, name) is None, name
    for name in TRAJECTORY_FIELDS:
        np.testing.assert_array_equal(getattr(with_latents, name), getattr(without, name), err_msg=name)


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_the_step_displacement_is_the_one_frame_jitter_of_the_true_embedding(tmp_path, device):
    """`true_step_displacement[:, h-1]` is `||e(h) - e(h-1)||` with e(0) the
    last context frame, so at h = 1 it is `true_embedding_displacement`'s
    own first column (`||e(1) - e(0)||`) bitwise, and the cumulative path
    it traces bounds the straight-line displacement at every h (the
    triangle inequality) -- a row taken one frame off fails the first;
    a row taken against the wrong anchor fails the second."""
    model, paths, probe = _two_episode_rig(tmp_path, device)
    kept = latent_pass(model, paths, probe, keep=True, device=device)
    np.testing.assert_array_equal(
        kept.true_step_displacement[:, 0], kept.true_embedding_displacement[:, 0]
    )
    path = np.cumsum(kept.true_step_displacement, axis=1)
    assert (kept.true_embedding_displacement <= path + 1e-9).all()
    assert (kept.true_step_displacement[:, 1:] > 0).all()


def test_a_broken_step_one_identity_is_refused_by_name(tmp_path, monkeypatch):
    """A pass whose open-loop prior at step 1 is not the teacher-forced prior
    at step 1 did not start both from the same context state; it is refused
    as `LatentIdentityError` (a ValueError) naming the window, and nothing
    is returned. Forced by making `imagine` advance from a perturbed state."""
    model, paths, probe = _two_episode_rig(tmp_path, torch.device("cpu"))
    real_imagine = model.rssm.imagine

    def perturbed_imagine(actions, state):
        h, z = state
        return real_imagine(actions, (h + 1.0, z))

    monkeypatch.setattr(model.rssm, "imagine", perturbed_imagine)
    with pytest.raises(diagnostics_module.LatentIdentityError, match="window at start"):
        latent_pass(model, paths, probe, keep=True)
    assert issubclass(diagnostics_module.LatentIdentityError, ValueError)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_diagnostics.py -k "latent or step_displacement or step_one_identity" -q`
Expected: FAIL — `TypeError: _diagnose() got an unexpected keyword argument 'keep_latents'` and `AttributeError: ... has no attribute 'LatentIdentityError'`.

- [ ] **Step 3: Add the fields, the tuple and the error to `diagnostics.py`**

After `true_embedding_displacement`'s docstring in `_Pass` (the last field before `_TRAJECTORY_FIELDS`), add:

```python
    # The latent-space fields (M3g). None unless the traversal ran with
    # `keep_latents=True`; then every one is set, for the CANONICAL pass only.
    post_logits: np.ndarray | None = None
    """`(n_windows, context + horizon, groups, classes)`: the posterior's
    logits over the whole window -- the context filter's steps, then the
    floor's over the real future frames. Index `context - 1` is the last
    context frame t0; index `context - 1 + h` is horizon step h. float32, as
    the model emits them; the consumer reduces in float64."""
    prior_teacher_logits: np.ndarray | None = None
    """`(n_windows, horizon, groups, classes)`: the floor pass's
    `prior_logits` -- at horizon step h, `prior_net` on the `h` advanced by
    the POSTERIOR's sample at h - 1 and the true action: the teacher-forced
    one-step prediction, which `observe` computes already."""
    prior_open_logits: np.ndarray | None = None
    """`(n_windows, horizon, groups, classes)`: the canonical `imagine`'s
    `prior_logits` -- the prior fed its own samples from t0."""
    posterior_rendering_distance: np.ndarray | None = None
    """`(n_windows, horizon)`: `||e_post(h) - e(h)||`, the head on the floor's
    latent against the encoder's embedding of the same frame."""
    true_step_displacement: np.ndarray | None = None
    """`(n_windows, horizon)`: `||e(h) - e(h - 1)||`, with e(0) the last
    context frame -- the one-frame jitter of the encoder's embedding."""
```

After the `_TRAJECTORY_FIELDS` tuple and its docstring, add:

```python
_LATENT_FIELDS: tuple[str, ...] = (
    "post_logits", "prior_teacher_logits", "prior_open_logits",
    "posterior_rendering_distance", "true_step_displacement",
)
"""The five `_Pass` fields `keep_latents` fills (M3g), in `_Pass` order;
`reference_trajectories` copies them onto `Trajectories` by this name.
Separate from `_TRAJECTORY_FIELDS` so the M3d list, and the tests that pin
it literally, stand as they are."""


class LatentIdentityError(ValueError):
    """The open-loop prior at step 1 is not the teacher-forced prior at step
    1 bitwise. Both are `prior_net` on the same `h` after the same context
    state and the same action, so a difference means the two passes did not
    start from the same state -- a plumbing fault, refused by name rather
    than left to surface as one more ValueError among the window guards."""
```

- [ ] **Step 4: Thread the flag through `_diagnose`**

Change the signature:

```python
    noise_reference: bool = True,
    keep_trajectories: bool = False,
    keep_latents: bool = False,
) -> _Pass:
```

Append to the docstring, after the `keep_trajectories` paragraph (before `Deliberately NOT separately decorated`):

```
    `keep_latents` (M3g) additionally carries, on the CANONICAL pass only,
    the posterior's logits over the whole window (the context filter's then
    the floor's), the floor pass's `prior_logits` -- the teacher-forced
    one-step prediction `observe` computes anyway -- and the canonical
    `imagine`'s `prior_logits`, plus the head's rendering error on the
    floor's latent and the one-frame jitter of the true embedding. Nothing
    is run that the pass did not already run and nothing is drawn from the
    stream. It needs `keep_trajectories`: the stage decomposition reads the
    latents beside the rows. The open-loop and teacher-forced priors at
    step 1 are asserted bitwise equal per window (`LatentIdentityError`).
```

After the `kept = (...)` assignment (the line building `{name: [] for name in _TRAJECTORY_FIELDS}`), add:

```python
    if keep_latents and not keep_trajectories:
        raise ValueError(
            "keep_latents needs keep_trajectories: the latent fields are read beside the "
            "trajectory rows, and a pass that keeps one without the other reads nothing"
        )
    kept_latents: dict[str, list[np.ndarray]] | None = (
        {name: [] for name in _LATENT_FIELDS} if keep_latents else None
    )
```

Inside the `if kept is not None:` block, after the last `kept[...].append(...)` (the `true_embedding_displacement` append), add — at the same indentation as those appends:

```python
                if kept_latents is not None:
                    # Both priors below were computed above: `real` by the
                    # floor's observe, `imagined` by the canonical imagine,
                    # each from `state` with `actions[:, context]` first. So
                    # their step-1 logits are `prior_net` on the same `h`,
                    # and a difference is a fault, not a measurement.
                    teacher = real["prior_logits"][0]
                    opened = imagined["prior_logits"][0]
                    if not torch.equal(opened[0], teacher[0]):
                        raise LatentIdentityError(
                            f"window at start {start}: the open-loop prior at step 1 is not "
                            "the teacher-forced prior at step 1 bitwise; the two passes did "
                            "not start from the same context state"
                        )
                    kept_latents["post_logits"].append(
                        torch.cat([observed["post_logits"], real["post_logits"]], dim=1)[0]
                        .cpu().numpy()
                    )
                    kept_latents["prior_teacher_logits"].append(teacher.cpu().numpy())
                    kept_latents["prior_open_logits"].append(opened.cpu().numpy())
                    kept_latents["posterior_rendering_distance"].append(
                        _embedding_distance(floor_embeddings, true_embedding[1:])
                    )
                    kept_latents["true_step_displacement"].append(
                        _embedding_distance(true_embedding[1:], true_embedding[:-1])
                    )
```

In the `return _Pass(...)` at the end, after the `**({} if kept is None else ...)` line, add:

```python
        **({} if kept_latents is None else {name: np.stack(rows) for name, rows in kept_latents.items()}),
```

- [ ] **Step 5: Extend `Trajectories` and `reference_trajectories`**

In `Trajectories`, after the `band` field and its docstring, add:

```python
    post_logits: np.ndarray | None = None
    prior_teacher_logits: np.ndarray | None = None
    prior_open_logits: np.ndarray | None = None
    posterior_rendering_distance: np.ndarray | None = None
    true_step_displacement: np.ndarray | None = None
    """The five `_LATENT_FIELDS` (M3g), under `_Pass`'s names and shapes;
    None unless `reference_trajectories` was asked for them. The nine
    trajectory arrays above are unchanged by the request."""
```

Change `reference_trajectories`'s signature and body:

```python
    device,
    feature_backbone,
    keep_latents: bool = False,
) -> Trajectories:
```

Append to its docstring: `` `keep_latents` (M3g) asks the pass for the five latent fields as well; the nine trajectory arrays and the band are bitwise the same either way. ``

```python
    result = _diagnose(
        model, val_paths, embedding_probe_weights,
        arms={}, context=context, horizon=horizon, seed=seed, device=device,
        feature_backbone=feature_backbone, noise_reference=True,
        keep_trajectories=True, keep_latents=keep_latents,
    )
    return Trajectories(
        **{name: getattr(result, name) for name in _TRAJECTORY_FIELDS},
        **({name: getattr(result, name) for name in _LATENT_FIELDS} if keep_latents else {}),
        window_episode=result.window_episode,
        windows_total=result.windows_total,
        reference_position=result.reference.rssm_position,
        persistence_position=result.reference.persistence_position,
        band=result.reference,
    )
```

- [ ] **Step 6: Run the new tests and the whole diagnostics file**

Run: `.venv/bin/python -m pytest tests/eval/test_diagnostics.py -q`
Expected: all PASS, including the five new tests on every device present (cpu; and mps on the laptop). The pre-existing M3d tests are unchanged and green — the default path is byte-identical.

- [ ] **Step 7: Run the consumers of `Trajectories`**

Run: `.venv/bin/python -m pytest tests/eval/test_trust_horizon_script.py tests/eval/test_split_gap.py tests/eval/test_checkpoint_ladder_script.py -q`
Expected: PASS — the new fields default to None, so every fabricated `Trajectories` still constructs.

- [ ] **Step 8: Commit**

```bash
git add src/mbfps/eval/diagnostics.py tests/eval/test_diagnostics.py
git commit -m "feat: keep_latents -- the canonical pass keeps the posterior over the window, the teacher-forced and open-loop priors and two embedding companions, asserting the step-1 identity; the default path bitwise unchanged

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `mbfps/eval/stages.py` — the statistics

**Files:**
- Create: `src/mbfps/eval/stages.py`
- Test: `tests/eval/test_stages.py`

**Interfaces:**
- Consumes: `mbfps.eval.split_gap.DECISION_H`, `mbfps.models.rssm.KL_FREE_BITS`.
- Produces (all pure numpy, float64, no torch, no files): `FAMILY = 5`, `SEEDS_REQUIRED = 2`, `REPORTED_K`, `STAGES`, `DECISION_H`, `KL_FREE_BITS` (re-exported); `log_softmax(logits)`; `categorical_kl(logits_q, logits_p) -> (...)`; `mode(logits) -> (..., G)` int; `entropy_by_group(logits) -> (G,)`; `information(post_logits, prior_teacher_logits, context) -> (n,)`; `teacher_accuracy(post_logits, prior_teacher_logits, context) -> (n,)`; `persistence_accuracy(post_logits, context) -> (n,)`; `marginal_classes(post_logits, context) -> (G,)`; `marginal_accuracy(post_logits, classes, context) -> (n,)`; `open_accuracy(post_logits, prior_open_logits, context) -> (n, H)`; `open_persistence(post_logits, context) -> (n, H)`; `open_marginal(post_logits, classes, context) -> (n, H)`; `teacher_nll(post_logits, prior_teacher_logits, context) -> (n,)`; `decode_margin(persistence_distance, distance_to_truth, h) -> (n,)`. Task 3 appends the reading to the same module.

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_stages.py`:

```python
"""mbfps.eval.stages, the pure half of M3g: the latent-space statistics of
spec 2.3 on fabricated logits with hand-typed answers, and (Task 3) the
reading of spec 3.2 on fabricated contrasts.

The rig is deliberately small -- 2 windows, 3 window steps (1 context, 2
horizon), 2 groups of 3 classes -- so every expected number is typed by hand
in the test and every function is pinned against a value, not against
itself. Logits are built from class indices with `sharp` (1000 at the class:
after the softmax the other classes underflow to exactly 0, so entropies and
NLLs come out exact) or `uniform` (all zero).
"""

import numpy as np
import pytest

import mbfps.eval.stages as stages
from mbfps.eval.stages import (
    DECISION_H,
    FAMILY,
    KL_FREE_BITS,
    REPORTED_K,
    SEEDS_REQUIRED,
    STAGES,
    categorical_kl,
    decode_margin,
    entropy_by_group,
    information,
    log_softmax,
    marginal_accuracy,
    marginal_classes,
    mode,
    open_accuracy,
    open_marginal,
    open_persistence,
    persistence_accuracy,
    teacher_accuracy,
    teacher_nll,
)

GROUPS, CLASSES = 2, 3
CONTEXT = 1


def sharp(index) -> np.ndarray:
    """`(..., G)` class indices -> `(..., G, C)` logits with 1000 at the class."""
    index = np.asarray(index)
    return (np.eye(CLASSES)[index] * 1000.0).astype(np.float64)


def uniform(*leading) -> np.ndarray:
    return np.zeros((*leading, GROUPS, CLASSES), dtype=np.float64)


# The posterior over 3 window steps (step 0 is t0, the last context frame):
#   window 0: t0 [0, 1] -> h=1 [1, 1] -> h=2 [2, 0]
#   window 1: t0 [2, 2] -> h=1 [2, 2] -> h=2 [2, 2]
POST = sharp([[[0, 1], [1, 1], [2, 0]], [[2, 2], [2, 2], [2, 2]]])
# Teacher-forced prior over the 2 horizon steps: perfect on window 0, wrong
# everywhere on window 1.
TEACHER = sharp([[[1, 1], [2, 0]], [[0, 0], [0, 0]]])
# Open-loop prior: right at k=1 in both windows, wrong in both groups at k=2.
OPEN = sharp([[[1, 1], [1, 1]], [[2, 2], [1, 1]]])


def test_the_pre_registered_constants_are_the_specs():
    assert DECISION_H == 15 and FAMILY == 5 and SEEDS_REQUIRED == 2
    assert KL_FREE_BITS == 0.20
    assert REPORTED_K == (1, 2, 3, 5, 10, 15, 30, 45)
    assert STAGES == ("encode", "predict", "carry", "decode")


def test_log_softmax_normalises_and_is_shift_invariant():
    logits = np.array([[1.0, 2.0, 3.0]])
    log_p = log_softmax(logits)
    assert np.exp(log_p).sum() == pytest.approx(1.0)
    np.testing.assert_allclose(log_softmax(logits + 100.0), log_p)


def test_categorical_kl_is_zero_between_identical_distributions_and_sums_over_groups():
    """Identical logits give exactly 0.0 (the difference of two identical
    log-softmaxes is exactly zero), whatever the sharpness. Against a uniform
    prior the KL is `log C - H(q)` per group, summed over groups: a q of
    (1/2, 1/4, 1/4) in both groups gives 2 * (log 3 - 1.5 log 2)."""
    assert categorical_kl(POST, POST).shape == (2, 3)
    assert (categorical_kl(POST, POST) == 0.0).all()
    assert (categorical_kl(uniform(4), uniform(4)) == 0.0).all()
    q = np.log(np.array([[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]]))[None]
    expected = 2 * (np.log(3) - 1.5 * np.log(2))
    assert categorical_kl(q, uniform(1)) == pytest.approx(np.array([expected]))
    with pytest.raises(ValueError, match="shape"):
        categorical_kl(POST, TEACHER)


def test_mode_is_the_argmax_over_classes_with_ties_to_the_smallest():
    np.testing.assert_array_equal(mode(POST), [[[0, 1], [1, 1], [2, 0]], [[2, 2], [2, 2], [2, 2]]])
    np.testing.assert_array_equal(mode(uniform(1)), [[0, 0]])
    assert mode(POST).dtype.kind == "i"


def test_entropy_by_group_is_zero_when_sharp_and_log_c_when_uniform():
    np.testing.assert_array_equal(entropy_by_group(POST), [0.0, 0.0])
    np.testing.assert_allclose(entropy_by_group(uniform(2, 3)), [np.log(3), np.log(3)])
    mixed = np.concatenate([sharp([[[0, 0]]]), uniform(1, 1)], axis=1)  # one sharp, one uniform step
    np.testing.assert_allclose(entropy_by_group(mixed), [np.log(3) / 2, np.log(3) / 2])


def test_information_is_the_horizon_mean_kl_of_posterior_against_the_teacher():
    """Window 0: the teacher equals the posterior at both horizon steps
    (KL 0). Window 1: the posterior is sharp at class 2, the teacher sharp
    at class 0, in both groups at both steps: KL per group is
    `1 * (0 - (-1000))` = 1000, two groups, so 2000 at each step."""
    np.testing.assert_allclose(information(POST, TEACHER, CONTEXT), [0.0, 2000.0])
    assert (information(POST, POST[:, CONTEXT:], CONTEXT) == 0.0).all()


def test_teacher_accuracy_persistence_and_marginal_are_the_hand_counts():
    """Window 0 teacher: both steps, both groups right -> 1.0; window 1 -> 0.0.
    Persistence (the class at h-1 predicts h): window 0 step 0->1 keeps g1
    only (1/2), step 1->2 keeps neither (0) -> 0.25; window 1 keeps all
    -> 1.0. Marginal classes over the four horizon (window, step) pairs:
    g0 sees [1, 2, 2, 2] -> 2, g1 sees [1, 0, 2, 2] -> 2; window 0 matches
    (0 + 1)/4 = 0.25, window 1 matches everything."""
    np.testing.assert_array_equal(teacher_accuracy(POST, TEACHER, CONTEXT), [1.0, 0.0])
    np.testing.assert_array_equal(persistence_accuracy(POST, CONTEXT), [0.25, 1.0])
    classes = marginal_classes(POST, CONTEXT)
    np.testing.assert_array_equal(classes, [2, 2])
    np.testing.assert_array_equal(marginal_accuracy(POST, classes, CONTEXT), [0.25, 1.0])


def test_marginal_classes_break_ties_toward_the_smallest_class():
    tied = sharp([[[0, 2], [1, 1]]])  # context 1: one horizon step, g0 sees [1], g1 sees [1]
    np.testing.assert_array_equal(marginal_classes(tied, CONTEXT), [1, 1])
    two_steps = sharp([[[0, 0], [2, 1], [1, 2]]])  # g0 sees [2, 1], g1 sees [1, 2]: ties -> 1
    np.testing.assert_array_equal(marginal_classes(two_steps, CONTEXT), [1, 1])


def test_open_loop_accuracy_and_its_two_baselines_per_step():
    """Open: right at k=1, wrong at k=2 in both windows. Persistence from t0:
    window 0 anchor [0, 1] against [1, 1] then [2, 0] -> 0.5, 0; window 1
    anchor [2, 2] against [2, 2] twice -> 1, 1. Marginal [2, 2]: window 0
    [1, 1] -> 0, [2, 0] -> 0.5; window 1 -> 1, 1."""
    np.testing.assert_array_equal(open_accuracy(POST, OPEN, CONTEXT), [[1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_array_equal(open_persistence(POST, CONTEXT), [[0.5, 0.0], [1.0, 1.0]])
    classes = marginal_classes(POST, CONTEXT)
    np.testing.assert_array_equal(open_marginal(POST, classes, CONTEXT), [[0.0, 0.5], [1.0, 1.0]])


def test_open_accuracy_at_step_one_equals_the_teacher_accuracy_at_step_one_on_equal_priors():
    """When the open-loop prior at step 1 is the teacher-forced prior at step
    1 (the pass's identity), the two accuracies agree at h = 1 exactly."""
    opened = OPEN.copy()
    opened[:, 0] = TEACHER[:, 0]
    per_step_teacher = (mode(TEACHER) == mode(POST[:, CONTEXT:])).mean(axis=2)
    np.testing.assert_array_equal(open_accuracy(POST, opened, CONTEXT)[:, 0], per_step_teacher[:, 0])


def test_teacher_nll_is_zero_when_right_and_the_gap_in_logits_when_wrong():
    """A sharp prior right about the posterior's mode assigns it log p = 0;
    wrong by a 1000-logit margin it assigns -1000 -- window 0 averages 0,
    window 1 averages 1000."""
    np.testing.assert_allclose(teacher_nll(POST, TEACHER, CONTEXT), [0.0, 1000.0])
    assert teacher_nll(POST, uniform(2, 2), CONTEXT) == pytest.approx([np.log(3), np.log(3)])


def test_decode_margin_is_persistence_minus_model_at_h():
    persistence = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    model = np.array([[0.5, 2.5, 1.0], [4.0, 1.0, 9.0]])
    np.testing.assert_array_equal(decode_margin(persistence, model, 1), [0.5, 0.0])
    np.testing.assert_array_equal(decode_margin(persistence, model, 3), [2.0, -3.0])
    with pytest.raises(ValueError, match="h"):
        decode_margin(persistence, model, 4)
    with pytest.raises(ValueError, match="shape"):
        decode_margin(persistence, model[:, :2], 1)


def test_the_shapes_and_the_context_are_checked_before_anything_is_reduced():
    with pytest.raises(ValueError, match="context"):
        information(POST, TEACHER, 0)
    with pytest.raises(ValueError, match="context"):
        persistence_accuracy(POST, 3)
    with pytest.raises(ValueError, match="horizon posterior"):
        teacher_accuracy(POST, TEACHER[:, :1], CONTEXT)
    with pytest.raises(ValueError, match="horizon posterior"):
        open_accuracy(POST, OPEN[:, :, :1], CONTEXT)
    with pytest.raises(ValueError, match="classes"):
        marginal_accuracy(POST, np.array([2]), CONTEXT)
    with pytest.raises(ValueError, match="groups, classes"):
        mode(np.zeros(3))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_stages.py -q`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'mbfps.eval.stages'`.

- [ ] **Step 3: Write the module**

Create `src/mbfps/eval/stages.py`:

```python
"""The stage decomposition's pure rules -- milestone M3g.

Two halves, neither of which touches torch or a file:

  * spec 2.3 -- the statistics: on the posterior over a window, the
    teacher-forced prior and the open-loop prior (the three logit arrays
    `diagnostics.keep_latents` keeps), the information the frame injects,
    the mode accuracies of the two priors against the posterior's next mode,
    the two baselines every accuracy is read against (latent persistence:
    the class does not change; the marginal: the most frequent class), the
    prior's NLL of the posterior's mode, the per-group entropy, and the
    free-channel margin at one horizon step;
  * spec 3.2 -- the reading: per arm, four stages in a fixed order, each a
    pooled contrast against `z_fam` replicated across seeds, and one status
    with the sentence that decided it (Task 3).

Everything reduces in float64 from whatever it is handed. The window layout
is the pass's: `post_logits[:, context - 1]` is t0, the last context frame,
and `post_logits[:, context - 1 + h]` is horizon step h; the two prior
arrays are indexed by h - 1.
"""

import numpy as np

from mbfps.eval.split_gap import DECISION_H
from mbfps.models.rssm import KL_FREE_BITS

FAMILY: int = 5
"""Spec 3.1: the five contrasts one arm's reading may consult -- encode,
predict (two), carry, decode. Each arm is its own fixed-sequence procedure."""
SEEDS_REQUIRED: int = 2
"""Spec 3.2: a stage passes pooled AND within >= 2 seeds."""
REPORTED_K: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 30, 45)
"""The carry k-curve's printed columns, filtered to <= the run's horizon."""
STAGES: tuple[str, ...] = ("encode", "predict", "carry", "decode")
"""The order the stages are read in; the status is the first not passed."""

__all__ = [
    "DECISION_H", "FAMILY", "KL_FREE_BITS", "REPORTED_K", "SEEDS_REQUIRED", "STAGES",
    "categorical_kl", "decode_margin", "entropy_by_group", "information", "log_softmax",
    "marginal_accuracy", "marginal_classes", "mode", "open_accuracy", "open_marginal",
    "open_persistence", "persistence_accuracy", "teacher_accuracy", "teacher_nll",
]


# ---------------------------------------------------------------------------
# Logits.
# ---------------------------------------------------------------------------


def _logits(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError(f"{name} needs (..., groups, classes), got shape {array.shape}")
    return array


def log_softmax(logits) -> np.ndarray:
    """Over the last axis, shifted by the max so a 1000-logit class does not
    overflow: the other classes then underflow to exactly 0 probability,
    which is what makes a sharp posterior's entropy exactly 0."""
    array = _logits(logits, "logits")
    shifted = array - array.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def categorical_kl(logits_q, logits_p) -> np.ndarray:
    """KL(q || p) summed over the groups, per leading index: `(..., G, C)` ->
    `(...)`. The per-element form of `rssm._categorical_kl` before its
    batch-and-time mean, in float64."""
    log_q, log_p = log_softmax(logits_q), log_softmax(logits_p)
    if log_q.shape != log_p.shape:
        raise ValueError(f"logits_q shape {log_q.shape} and logits_p shape {log_p.shape} differ")
    per_group = (np.exp(log_q) * (log_q - log_p)).sum(axis=-1)
    return per_group.sum(axis=-1)


def mode(logits) -> np.ndarray:
    """The argmax over the classes: `(..., G, C)` -> `(..., G)` int; a tie
    goes to the smallest class, as `np.argmax` does."""
    return np.argmax(_logits(logits, "logits"), axis=-1)


def entropy_by_group(logits) -> np.ndarray:
    """Per-group entropy in nats, averaged over every leading axis:
    `(..., G, C)` -> `(G,)`. 0 for a sharp group, log C for a uniform one."""
    log_p = log_softmax(logits)
    p = np.exp(log_p)
    entropy = -(p * np.where(p > 0, log_p, 0.0)).sum(axis=-1)
    return entropy.reshape(-1, entropy.shape[-1]).mean(axis=0)


# ---------------------------------------------------------------------------
# The window layout.
# ---------------------------------------------------------------------------


def _split(post_logits, context: int) -> tuple[np.ndarray, np.ndarray]:
    """`(anchor (n, G, C), horizon (n, H, G, C))`: the posterior at t0 and at
    horizon steps 1..H."""
    post = _logits(post_logits, "post_logits")
    if post.ndim != 4:
        raise ValueError(f"post_logits needs (windows, steps, groups, classes), got shape {post.shape}")
    context = int(context)
    if not 1 <= context < post.shape[1]:
        raise ValueError(
            f"context {context} must be >= 1 and leave at least one horizon step of the "
            f"{post.shape[1]} window steps"
        )
    return post[:, context - 1], post[:, context:]


def _horizon_prior(value, name: str, horizon: np.ndarray) -> np.ndarray:
    prior = _logits(value, name)
    if prior.shape != horizon.shape:
        raise ValueError(f"{name} shape {prior.shape} must match the horizon posterior {horizon.shape}")
    return prior


def _classes(value, horizon: np.ndarray) -> np.ndarray:
    classes = np.asarray(value, dtype=int)
    if classes.shape != (horizon.shape[2],):
        raise ValueError(f"classes must be one class per group, shape ({horizon.shape[2]},), got {classes.shape}")
    return classes


# ---------------------------------------------------------------------------
# Encode.
# ---------------------------------------------------------------------------


def information(post_logits, prior_teacher_logits, context: int) -> np.ndarray:
    """Per window, the mean over horizon steps of KL(posterior || teacher-forced
    prior) summed over groups -- the information the frame injects beyond
    what the prior predicted (spec 2.3, encode). `(n,)`."""
    _, horizon = _split(post_logits, context)
    teacher = _horizon_prior(prior_teacher_logits, "prior_teacher_logits", horizon)
    return categorical_kl(horizon, teacher).mean(axis=1)


# ---------------------------------------------------------------------------
# Predict: teacher-forced, one step.
# ---------------------------------------------------------------------------


def teacher_accuracy(post_logits, prior_teacher_logits, context: int) -> np.ndarray:
    """Per window, over (horizon step, group): does the teacher-forced prior's
    mode equal the posterior's mode at the same step? `(n,)`."""
    _, horizon = _split(post_logits, context)
    teacher = _horizon_prior(prior_teacher_logits, "prior_teacher_logits", horizon)
    return (mode(teacher) == mode(horizon)).mean(axis=(1, 2))


def persistence_accuracy(post_logits, context: int) -> np.ndarray:
    """Latent persistence: the posterior's mode at h - 1 predicts h (with t0
    before h = 1) -- what a model that says "nothing changes" scores exactly.
    `(n,)`."""
    post = _logits(post_logits, "post_logits")
    _split(post, context)
    modes = mode(post[:, int(context) - 1:])
    return (modes[:, :-1] == modes[:, 1:]).mean(axis=(1, 2))


def marginal_classes(post_logits, context: int) -> np.ndarray:
    """Per group, the most frequent posterior mode over every (window,
    horizon step) pair; ties to the smallest class. `(G,)`. In-sample by
    design: the stronger baseline."""
    _, horizon = _split(post_logits, context)
    modes = mode(horizon)
    classes = horizon.shape[-1]
    counts = np.stack([
        np.bincount(modes[:, :, g].ravel(), minlength=classes) for g in range(modes.shape[-1])
    ])
    return counts.argmax(axis=1)


def marginal_accuracy(post_logits, classes, context: int) -> np.ndarray:
    """Per window, over (horizon step, group): does the marginal class equal
    the posterior's mode? `(n,)`."""
    _, horizon = _split(post_logits, context)
    marginal = _classes(classes, horizon)
    return (mode(horizon) == marginal[None, None, :]).mean(axis=(1, 2))


# ---------------------------------------------------------------------------
# Carry: open-loop from t0, per step.
# ---------------------------------------------------------------------------


def open_accuracy(post_logits, prior_open_logits, context: int) -> np.ndarray:
    """Per window and step k, over groups: does the open-loop prior's mode at
    k equal the posterior's mode at k? `(n, H)`."""
    _, horizon = _split(post_logits, context)
    opened = _horizon_prior(prior_open_logits, "prior_open_logits", horizon)
    return (mode(opened) == mode(horizon)).mean(axis=2)


def open_persistence(post_logits, context: int) -> np.ndarray:
    """Latent persistence from t0: the posterior's mode at t0 predicts every
    k. `(n, H)`."""
    anchor, horizon = _split(post_logits, context)
    return (mode(anchor)[:, None, :] == mode(horizon)).mean(axis=2)


def open_marginal(post_logits, classes, context: int) -> np.ndarray:
    """The marginal class against the posterior's mode at every k. `(n, H)`."""
    _, horizon = _split(post_logits, context)
    marginal = _classes(classes, horizon)
    return (mode(horizon) == marginal[None, None, :]).mean(axis=2)


# ---------------------------------------------------------------------------
# Companions and decode.
# ---------------------------------------------------------------------------


def teacher_nll(post_logits, prior_teacher_logits, context: int) -> np.ndarray:
    """Per window, the mean over (horizon step, group) of the teacher-forced
    prior's negative log-probability of the posterior's mode. `(n,)`."""
    _, horizon = _split(post_logits, context)
    teacher = _horizon_prior(prior_teacher_logits, "prior_teacher_logits", horizon)
    picked = np.take_along_axis(log_softmax(teacher), mode(horizon)[..., None], axis=-1)[..., 0]
    return -picked.mean(axis=(1, 2))


def decode_margin(persistence_distance, distance_to_truth, h: int) -> np.ndarray:
    """`||e_hat(0) - e(h)|| - ||e_hat(h) - e(h)||` per window at one horizon
    step: positive where the imagination is closer to the truth than the held
    context rendering is. `(n,)`."""
    persistence = np.asarray(persistence_distance, dtype=np.float64)
    model = np.asarray(distance_to_truth, dtype=np.float64)
    if persistence.ndim != 2 or persistence.shape != model.shape:
        raise ValueError(
            f"persistence_distance shape {persistence.shape} and distance_to_truth shape "
            f"{model.shape} must both be (windows, horizon)"
        )
    h = int(h)
    if not 1 <= h <= persistence.shape[1]:
        raise ValueError(f"h must be in 1..{persistence.shape[1]}, got {h}")
    return persistence[:, h - 1] - model[:, h - 1]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_stages.py -q`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/eval/stages.py tests/eval/test_stages.py
git commit -m "feat: the stage decomposition's statistics -- the frame's information, the teacher-forced and open-loop mode accuracies against latent persistence and the marginal, the prior's NLL, the per-group entropy, the decode margin

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `stages.py` — the reading

**Files:**
- Modify: `src/mbfps/eval/stages.py` (append)
- Test: `tests/eval/test_stages.py` (append)

**Interfaces:**
- Consumes: `split_gap.StratumContrast` (`estimate`, `se`, `z`, `clusters`), `split_gap.clears`, `split_gap.fmt_z`; Task 2's constants.
- Produces: `CONTRAST_LABELS: dict[str, tuple[str, ...]]`; `ArmInputs(encode, predict_persistence, predict_marginal, carry, decode, per_seed)` with `.contrasts(stage) -> tuple[tuple[str, StratumContrast], ...]`; `StagesInputs(arms, z_fam, h)`; `Status` (`ENCODE_FAILS`, `PREDICT_FAILS`, `CARRY_FAILS`, `DECODE_FAILS`, `NO_STAGE_FAILS`); `FAILING_STATUS: dict[str, Status]`; `StageResult(stage, passes, wording, rule, seeds_holding, seeds_total)`; `ArmReading(arm, status, rule, stages)`; `StagesReading(arms, h, z_fam)`; `reading_stages(inputs) -> StagesReading`; `format_reading_stages(reading, inputs) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_stages.py`:

```python


# ---------------------------------------------------------------------------
# The reading (spec 3.2): four stages in order, one status per arm, the
# sentence that decided it. Fabricated contrasts; one rule mutated per test.
# ---------------------------------------------------------------------------

from mbfps.eval.split_gap import StratumContrast  # noqa: E402
from mbfps.eval.stages import (  # noqa: E402
    CONTRAST_LABELS,
    FAILING_STATUS,
    ArmInputs,
    ArmReading,
    StageResult,
    StagesInputs,
    StagesReading,
    Status,
    format_reading_stages,
    reading_stages,
)

Z_FAM = 2.81


def c(z: float, estimate: float | None = None) -> StratumContrast:
    """A contrast with the given z; the estimate follows its sign unless given."""
    if estimate is None:
        estimate = float("nan") if not np.isfinite(z) else 0.01 * z
    return StratumContrast(estimate=estimate, se=0.01, z=z, clusters=24)


def leaf(encode=5.0, pp=5.0, pm=5.0, carry=5.0, decode=5.0) -> ArmInputs:
    return ArmInputs(
        encode=c(encode), predict_persistence=c(pp), predict_marginal=c(pm),
        carry=c(carry), decode=c(decode), per_seed=None,
    )


def arm(encode=5.0, pp=5.0, pm=5.0, carry=5.0, decode=5.0, seeds=None) -> ArmInputs:
    """Pooled inputs whose three leaves equal the pooled contrasts unless
    `seeds` gives each leaf its own `(encode, pp, pm, carry, decode)`."""
    pooled = leaf(encode, pp, pm, carry, decode)
    leaves = (
        {0: pooled, 1: pooled, 2: pooled} if seeds is None
        else {i: leaf(*values) for i, values in enumerate(seeds)}
    )
    return ArmInputs(
        encode=pooled.encode, predict_persistence=pooled.predict_persistence,
        predict_marginal=pooled.predict_marginal, carry=pooled.carry, decode=pooled.decode,
        per_seed=leaves,
    )


def read(**arms) -> StagesReading:
    return reading_stages(StagesInputs(arms=arms, z_fam=Z_FAM, h=15))


def test_the_status_order_and_the_contrast_labels_are_the_specs():
    assert [s.name for s in Status] == [
        "ENCODE_FAILS", "PREDICT_FAILS", "CARRY_FAILS", "DECODE_FAILS", "NO_STAGE_FAILS",
    ]
    assert FAILING_STATUS == {
        "encode": Status.ENCODE_FAILS, "predict": Status.PREDICT_FAILS,
        "carry": Status.CARRY_FAILS, "decode": Status.DECODE_FAILS,
    }
    assert CONTRAST_LABELS == {
        "encode": ("information - 0.20",),
        "predict": ("teacher - persistence", "teacher - marginal"),
        "carry": ("open - persistence",),
        "decode": ("free margin",),
    }
    a = leaf()
    assert [label for label, _ in a.contrasts("predict")] == ["teacher - persistence", "teacher - marginal"]
    assert a.contrasts("encode")[0][1] is a.encode
    with pytest.raises(KeyError):
        a.contrasts("render")


def test_every_stage_clearing_in_every_seed_reads_no_stage_fails():
    reading = read(pixel_ae=arm())
    r = reading.arms["pixel_ae"]
    assert isinstance(r, ArmReading) and r.status is Status.NO_STAGE_FAILS
    assert all(isinstance(s, StageResult) and s.passes and s.wording == "passes" for s in r.stages.values())
    assert list(r.stages) == list(STAGES)
    assert r.rule.startswith("every stage passes")
    assert "seeds holding 3 of 3" in r.stages["encode"].rule
    assert reading.h == 15 and reading.z_fam == Z_FAM


def test_a_negative_encode_contrast_reads_encode_fails_as_failed():
    r = read(a=arm(encode=-5.0)).arms["a"]
    assert r.status is Status.ENCODE_FAILS
    assert r.rule == r.stages["encode"].rule
    assert r.stages["encode"].wording == "failed"
    assert r.rule.startswith("encode failed: information - 0.20 z -5.00 < -2.81")
    assert "seeds holding 0 of 3" in r.rule


def test_an_encode_contrast_below_the_bar_reads_encode_fails_as_not_shown():
    r = read(a=arm(encode=1.0)).arms["a"]
    assert r.status is Status.ENCODE_FAILS
    assert r.stages["encode"].wording == "not shown"
    assert r.rule.startswith("encode not shown: information - 0.20 z +1.00 does not clear +2.81")


def test_a_non_positive_estimate_with_a_small_z_reads_failed_not_not_shown():
    inputs = arm()
    inputs = ArmInputs(
        encode=c(-0.5, estimate=-0.002), predict_persistence=inputs.predict_persistence,
        predict_marginal=inputs.predict_marginal, carry=inputs.carry, decode=inputs.decode,
        per_seed=inputs.per_seed,
    )
    r = read(a=inputs).arms["a"]
    assert r.status is Status.ENCODE_FAILS and r.stages["encode"].wording == "failed"
    assert "estimate -0.0020 <= 0" in r.rule


def test_predict_needs_both_contrasts_and_names_the_first_not_clearing():
    marginal_short = read(a=arm(pm=1.2)).arms["a"]
    assert marginal_short.status is Status.PREDICT_FAILS
    assert marginal_short.rule.startswith("predict not shown: teacher - marginal z +1.20 does not clear +2.81")
    assert "teacher - persistence z +5.00 clears +2.81" in marginal_short.rule
    assert "seeds holding 0 of 3" in marginal_short.rule
    persistence_short = read(a=arm(pp=1.0, pm=1.0)).arms["a"]
    assert persistence_short.status is Status.PREDICT_FAILS
    assert "teacher - persistence z +1.00 does not clear" in persistence_short.rule
    assert "teacher - marginal" not in persistence_short.rule.split("(")[0]


def test_carry_and_decode_fail_in_their_turn():
    assert read(a=arm(carry=-4.0)).arms["a"].status is Status.CARRY_FAILS
    assert read(a=arm(carry=-4.0)).arms["a"].rule.startswith("carry failed: open - persistence z -4.00 < -2.81")
    assert read(a=arm(decode=0.5)).arms["a"].status is Status.DECODE_FAILS
    assert read(a=arm(decode=0.5)).arms["a"].rule.startswith("decode not shown: free margin z +0.50 does not clear")


def test_the_first_stage_not_passed_decides_even_when_later_stages_fail_too():
    r = read(a=arm(encode=1.0, carry=-9.0, decode=-9.0)).arms["a"]
    assert r.status is Status.ENCODE_FAILS
    assert not r.stages["carry"].passes and r.stages["carry"].wording == "failed"


def test_a_pooled_clear_that_replicates_in_one_seed_only_is_not_passed():
    r = read(a=arm(seeds=[(5.0, 5, 5, 5, 5), (1.0, 5, 5, 5, 5), (1.0, 5, 5, 5, 5)])).arms["a"]
    assert r.status is Status.ENCODE_FAILS
    assert r.stages["encode"].wording == "not shown"
    assert r.stages["encode"].seeds_holding == 1 and r.stages["encode"].seeds_total == 3
    assert "seeds holding 1 of 3 (>= 2 required)" in r.rule
    assert r.rule.startswith("encode not shown: information - 0.20 z +5.00 clears +2.81 pooled")


def test_a_seed_holds_predict_only_when_both_its_contrasts_clear():
    r = read(a=arm(seeds=[(5, 5.0, 5.0, 5, 5), (5, 5.0, 1.0, 5, 5), (5, 1.0, 5.0, 5, 5)])).arms["a"]
    assert r.status is Status.PREDICT_FAILS
    assert r.stages["predict"].seeds_holding == 1
    two = read(a=arm(seeds=[(5, 5.0, 5.0, 5, 5), (5, 5.0, 5.0, 5, 5), (5, 1.0, 5.0, 5, 5)])).arms["a"]
    assert two.status is Status.NO_STAGE_FAILS and two.stages["predict"].seeds_holding == 2


def test_a_single_seed_leaf_read_alone_has_a_vacuous_replication_clause():
    r = read(a=leaf()).arms["a"]
    assert r.status is Status.NO_STAGE_FAILS
    assert "this seed alone" in r.stages["encode"].rule
    assert r.stages["encode"].seeds_holding == 1 and r.stages["encode"].seeds_total == 1
    assert read(a=leaf(carry=1.0)).arms["a"].status is Status.CARRY_FAILS


def test_z_exactly_at_the_bar_does_not_clear_and_a_nan_cannot_be_read():
    assert read(a=arm(encode=Z_FAM)).arms["a"].status is Status.ENCODE_FAILS
    nan = read(a=arm(carry=float("nan"))).arms["a"]
    assert nan.status is Status.CARRY_FAILS and nan.stages["carry"].wording == "not shown"
    assert "open - persistence z nan cannot be read" in nan.rule
    assert read(a=arm(decode=float("inf"))).arms["a"].status is Status.DECODE_FAILS


def test_arms_are_read_independently_and_in_the_callers_order():
    reading = read(b=arm(encode=-5.0), a=arm())
    assert list(reading.arms) == ["b", "a"]
    assert reading.arms["b"].status is Status.ENCODE_FAILS
    assert reading.arms["a"].status is Status.NO_STAGE_FAILS


def test_format_reading_stages_prints_every_contrast_and_the_verdicts():
    inputs = StagesInputs(arms={"pixel_ae": arm(pm=1.2), "frozen_ssl": arm()}, z_fam=Z_FAM, h=15)
    text = format_reading_stages(reading_stages(inputs), inputs)
    assert text.startswith("--- Reading S: the first failing stage at h=15")
    assert "z_fam = 2.81" in text
    for label in ("information - 0.20", "teacher - persistence", "teacher - marginal",
                  "open - persistence", "free margin"):
        assert text.count(label) >= 2, label
    assert "  verdict: pixel_ae    PREDICT FAILS -- decided by: predict not shown" in text
    assert "  verdict: frozen_ssl  NO STAGE FAILS -- decided by: every stage passes" in text
    assert "+1.20" in text and "yes" in text and "no" in text
    assert text.endswith("\n")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_stages.py -q`
Expected: FAIL at the second import block — `ImportError: cannot import name 'CONTRAST_LABELS'`.

- [ ] **Step 3: Append the reading to `stages.py`**

Add to the imports at the top of `src/mbfps/eval/stages.py`:

```python
from dataclasses import dataclass
from enum import Enum

from mbfps.eval.split_gap import DECISION_H, StratumContrast, clears, fmt_z
```

(replace the existing `from mbfps.eval.split_gap import DECISION_H` line) and extend `__all__` with `"CONTRAST_LABELS", "FAILING_STATUS", "ArmInputs", "ArmReading", "StageResult", "StagesInputs", "StagesReading", "Status", "format_reading_stages", "reading_stages"`. Then append:

```python


# ---------------------------------------------------------------------------
# The reading (spec 3.2), over pooled inputs.
# ---------------------------------------------------------------------------

CONTRAST_LABELS: dict[str, tuple[str, ...]] = {
    "encode": (f"information - {KL_FREE_BITS:.2f}",),
    "predict": ("teacher - persistence", "teacher - marginal"),
    "carry": ("open - persistence",),
    "decode": ("free margin",),
}
"""Each stage's contrasts in the order its sentence names them: predict's
persistence contrast before its marginal one."""


@dataclass(frozen=True)
class ArmInputs:
    """One arm's five pooled contrasts at the decision horizon, and the same
    within each seed alone (leaves carry `per_seed=None`). `contrasts(stage)`
    is the stage's contrasts under `CONTRAST_LABELS`' names and order."""

    encode: StratumContrast
    predict_persistence: StratumContrast
    predict_marginal: StratumContrast
    carry: StratumContrast
    decode: StratumContrast
    per_seed: "dict[int, ArmInputs] | None"

    def contrasts(self, stage: str) -> tuple[tuple[str, StratumContrast], ...]:
        labels = CONTRAST_LABELS[stage]
        values = {
            "encode": (self.encode,),
            "predict": (self.predict_persistence, self.predict_marginal),
            "carry": (self.carry,),
            "decode": (self.decode,),
        }[stage]
        return tuple(zip(labels, values))


@dataclass(frozen=True)
class StagesInputs:
    arms: dict[str, ArmInputs]
    z_fam: float
    h: int


class Status(str, Enum):
    """Spec 3.2's five outcomes: the first stage not passed, or none."""

    ENCODE_FAILS = "encode fails"
    PREDICT_FAILS = "predict fails"
    CARRY_FAILS = "carry fails"
    DECODE_FAILS = "decode fails"
    NO_STAGE_FAILS = "no stage fails"


FAILING_STATUS: dict[str, Status] = {
    "encode": Status.ENCODE_FAILS,
    "predict": Status.PREDICT_FAILS,
    "carry": Status.CARRY_FAILS,
    "decode": Status.DECODE_FAILS,
}


@dataclass(frozen=True)
class StageResult:
    """One stage of one arm: whether it passed, the word the sentence uses
    (`passes` / `failed` / `not shown`), the sentence, the replication."""

    stage: str
    passes: bool
    wording: str
    rule: str
    seeds_holding: int
    seeds_total: int


@dataclass(frozen=True)
class ArmReading:
    """`stages` carries every stage's result, in `STAGES` order, so the
    tables can print the ones after the first not passed; `status` and
    `rule` are the first not passed's, or `NO_STAGE_FAILS`."""

    arm: str
    status: Status
    rule: str
    stages: dict[str, StageResult]


@dataclass(frozen=True)
class StagesReading:
    arms: dict[str, ArmReading]
    h: int
    z_fam: float


def _holds(a: ArmInputs, stage: str, z_fam: float) -> bool:
    """Every contrast of the stage clears within these inputs."""
    return all(clears(contrast.z, z_fam) for _, contrast in a.contrasts(stage))


def _stage_result(stage: str, a: ArmInputs, z_fam: float) -> StageResult:
    """Spec 3.2's rule for one stage: pooled clears on every contrast AND
    replicated in >= SEEDS_REQUIRED seeds. The sentence names the first
    contrast not clearing (`failed` when its estimate is <= 0 or it clears
    the wrong way, `not shown` otherwise), then the contrasts that do clear,
    then the seeds holding."""
    pooled = a.contrasts(stage)
    bar = f"{z_fam:.2f}"
    if a.per_seed is None:
        # A single-seed leaf, read alone: there is nothing to replicate across.
        seeds_holding, seeds_total, replicated = int(_holds(a, stage, z_fam)), 1, True
        seeds_words = "this seed alone"
    else:
        seeds_holding = sum(1 for one in a.per_seed.values() if _holds(one, stage, z_fam))
        seeds_total = len(a.per_seed)
        replicated = seeds_holding >= SEEDS_REQUIRED
        seeds_words = f"seeds holding {seeds_holding} of {seeds_total}"
    clearing = [
        f"{label} z {fmt_z(contrast.z)} clears +{bar}"
        for label, contrast in pooled if clears(contrast.z, z_fam)
    ]
    blocking = [(label, contrast) for label, contrast in pooled if not clears(contrast.z, z_fam)]
    if blocking:
        label, contrast = blocking[0]
        passes = False
        if not np.isfinite(contrast.z):
            wording, verdict = "not shown", f"{label} z {fmt_z(contrast.z)} cannot be read"
        elif clears(-contrast.z, z_fam):
            wording, verdict = "failed", f"{label} z {fmt_z(contrast.z)} < -{bar}"
        elif not contrast.estimate > 0:
            wording = "failed"
            verdict = f"{label} estimate {fmt_z(contrast.estimate, '+.4f')} <= 0 (z {fmt_z(contrast.z)})"
        else:
            wording, verdict = "not shown", f"{label} z {fmt_z(contrast.z)} does not clear +{bar}"
        rest = clearing + [seeds_words]
    elif not replicated:
        passes, wording = False, "not shown"
        verdict = f"{clearing[0]} pooled"
        rest = clearing[1:] + [f"{seeds_words} (>= {SEEDS_REQUIRED} required)"]
    else:
        passes, wording = True, "passes"
        verdict = clearing[0]
        rest = clearing[1:] + [seeds_words]
    return StageResult(
        stage=stage, passes=passes, wording=wording,
        rule=f"{stage} {wording}: {verdict} ({'; '.join(rest)})",
        seeds_holding=seeds_holding, seeds_total=seeds_total,
    )


def _arm_reading(arm: str, a: ArmInputs, z_fam: float, h: int) -> ArmReading:
    """The stages in order; the status is the first not passed."""
    results = {stage: _stage_result(stage, a, z_fam) for stage in STAGES}
    for stage in STAGES:
        if not results[stage].passes:
            return ArmReading(arm=arm, status=FAILING_STATUS[stage], rule=results[stage].rule, stages=results)
    summary = "; ".join(results[stage].rule for stage in STAGES)
    return ArmReading(
        arm=arm, status=Status.NO_STAGE_FAILS,
        rule=f"every stage passes at h={h}: {summary}", stages=results,
    )


def reading_stages(inputs: StagesInputs) -> StagesReading:
    """One `ArmReading` per arm, in the caller's order; arms never read each
    other (spec 4: no ranking)."""
    return StagesReading(
        arms={arm: _arm_reading(arm, a, inputs.z_fam, inputs.h) for arm, a in inputs.arms.items()},
        h=inputs.h, z_fam=inputs.z_fam,
    )


def format_reading_stages(reading: StagesReading, inputs: StagesInputs) -> str:
    """The contrast table (every contrast of every arm, pooled), each stage's
    sentence, and the verdict lines, in `ladder.txt`'s style."""
    lines = [
        f"--- Reading S: the first failing stage at h={reading.h} (each contrast pooled over "
        f"the val windows, seeds averaged per window, episode-clustered); z_fam = {reading.z_fam:.2f} ---",
        f"  {'arm':<12}{'stage':<9}{'contrast':<24}{'estimate':>10}{'se':>9}{'z':>8}  clears",
    ]
    for arm, a in inputs.arms.items():
        for stage in STAGES:
            for label, contrast in a.contrasts(stage):
                verdict = "yes" if clears(contrast.z, reading.z_fam) else "no"
                lines.append(
                    f"  {arm:<12}{stage:<9}{label:<24}{fmt_z(contrast.estimate, '+.4f'):>10}"
                    f"{fmt_z(contrast.se, '.4f'):>9}{fmt_z(contrast.z):>8}  {verdict}"
                )
    lines.append("  stage by stage:")
    for arm, r in reading.arms.items():
        for stage in STAGES:
            lines.append(f"    {arm:<12}{r.stages[stage].rule}")
    for arm, r in reading.arms.items():
        lines.append(f"  verdict: {arm:<12}{r.status.name.replace('_', ' ')} -- decided by: {r.rule}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_stages.py -q`
Expected: 27 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/eval/stages.py tests/eval/test_stages.py
git commit -m "feat: the stage reading -- four stages in order, each a pooled contrast replicated across seeds, the first not passed naming the status with the sentence that decided it

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `scripts/stage_decomposition.py` — `evaluate`

**Files:**
- Create: `scripts/stage_decomposition.py`
- Test: `tests/eval/test_stage_decomposition_script.py`

**Interfaces:**
- Consumes: `trust_horizon.{Cell, CellMissing, load_cell, self_check, prepare_cell, EXIT_*}` and `checkpoint_ladder.rung_cell` by path; `diagnostics.reference_trajectories(..., keep_latents=True)` and `LatentIdentityError` (Task 1); the statistics of Task 2; `split_gap.decision_horizon`; `study.{SPLIT_SEED, git_sha, load_record, write_record}`; `trust.moved_mask`.
- Produces: `EXIT_CONTROL_MISREAD = 34`; `PHASES = ("evaluate",)` (Task 5 extends it); `CONTROL_ARM`, `CONTROL_SEEDS`, `CONTROL_DIR`, `CONTROL_LABEL = "control"`; `stages_record_path(out, arm, seed)`, `control_record_path(out, arm, seed)`, `record_path(out, kind, arm, seed)`; `cell_statistics(traj, *, context) -> dict`; `stages_record(...) -> dict`; `evaluate_cell(args, cell, device, train, val, *, kind, source) -> (status, record | None)`; `load_requested(args, cells, controls)`; `evaluate_phase(args, cells, controls, device, train, val) -> status`; `_parser()`, `main(argv=None) -> int`.

The record, one per cell and per control (`stages_<arm>_seed<n>.json` / `stages_control_<arm>_seed<n>.json`), top-level keys exactly: `arm, seed, kind ("cell" | "control"), label (the arm, or "control"), source, step, record_git_sha, context, horizon, decision_h, split_seed, device, torch_version, git_sha, episodes {val}, windows {total, episode, clusters}, latent {groups, classes}, self_check (the trust `SelfCheck.record()` for a cell, null for a control), identity (true), information [n], accuracy {teacher [n], persistence [n], marginal [n]}, open {accuracy [n][H], persistence [n][H], marginal [n][H]}, decode {persistence_distance [n][H], distance_to_truth [n][H], probe_persistence [n][H], probe_model [n][H], moved [n][H]}, companions {entropy [G], marginal_classes [G], nll [n], rendering_median [H], jitter_median [H]}, nonfinite`.

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_stage_decomposition_script.py`:

```python
"""scripts/stage_decomposition.py: per cell the trust pass with its latents,
the self-check, the identity, the statistics and one record (Task 4); then
(Task 5) the pooling, the control rule, Reading S, the tables and stages.txt.

The script is loaded by path. The REFERENCE is a real tiny cell on
`wide_buffer` -- thirty 40-step episodes, `run_job(..., steps=5, seq_len=4,
context=2, horizon=3, device="cpu")` -- with the diagnostic
`diagnose_dynamics.main` writes, exactly as the ladder tests build theirs.
The CONTROL is a second study directory holding the same arm at seeds 1 and
2 trained for 2 steps: a rung directory is a study directory, and
`rung_cell` reads a checkpoint and a record and nothing else. Every refusal
below is exercised by doctoring one file or monkeypatching one name.

THE FIXTURE'S FACTS: six validation episodes, `window_starts(40, 2, 3)` cuts
eight windows per episode -- 48 windows over 6 clusters on every record;
the decision horizon clamps from 15 to the run's 3.
"""

import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pytest

from mbfps.eval.diagnostics import LatentIdentityError
from mbfps.eval.stages import KL_FREE_BITS, SEEDS_REQUIRED
from mbfps.eval.study import StudyJob, load_record, run_job

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script("stage_decomposition")
diagnose = _load_script("diagnose_dynamics")
trust = _load_script("trust_horizon")

JOB = StudyJob("random_vit", 1)
JOB_KW = dict(steps=5, seq_len=4, context=2, horizon=3, device="cpu")
CONTEXT, HORIZON = JOB_KW["context"], JOB_KW["horizon"]
CONTROL_JOBS = (StudyJob("random_vit", 1), StudyJob("random_vit", 2))
CONTROL_KW = dict(steps=2, seq_len=4, context=2, horizon=3, device="cpu")
WINDOWS, CLUSTERS = 48, 6
GROUPS, CLASSES = 32, 32
RECORD = "result_random_vit_seed1.json"
DIAGNOSTIC = "diagnostic_random_vit_seed1.json"
STAGES_RECORD = "stages_random_vit_seed1.json"
CONTROL_RECORDS = ("stages_control_random_vit_seed1.json", "stages_control_random_vit_seed2.json")

RECORD_KEYS = {
    "arm", "seed", "kind", "label", "source", "step", "record_git_sha", "context", "horizon",
    "decision_h", "split_seed", "device", "torch_version", "git_sha", "episodes", "windows",
    "latent", "self_check", "identity", "information", "accuracy", "open", "decode",
    "companions", "nonfinite",
}


@pytest.fixture
def reference(tmp_path, wide_buffer, capsys):
    """One real cell with its diagnostic, and a control directory holding the
    same arm at two seeds trained for two steps."""
    out = tmp_path / "reference"
    run_job(JOB, wide_buffer, out, **JOB_KW)
    status = diagnose.main([
        "--out", str(out), "--data", str(wide_buffer.root), "--device", "cpu",
        "--arms", JOB.arm, "--seeds", str(JOB.seed),
        "--context", str(CONTEXT), "--horizon", str(HORIZON), "--ks", "1", str(HORIZON),
    ])
    assert status == diagnose.EXIT_OK, "the fixture's diagnostic was not written cleanly"
    control = tmp_path / "control"
    for job in CONTROL_JOBS:
        run_job(job, wide_buffer, control, **CONTROL_KW)
    capsys.readouterr()
    return types.SimpleNamespace(
        out=out, control=control, data=wide_buffer.root, stages=tmp_path / "stages",
    )


def _argv(ref, *extra: str) -> list[str]:
    return [
        "--out", str(ref.stages), "--reference", str(ref.out), "--control", str(ref.control),
        "--data", str(ref.data), "--device", "cpu", "--arms", JOB.arm, "--seeds", str(JOB.seed),
        "--control-arm", CONTROL_JOBS[0].arm,
        "--control-seeds", *[str(job.seed) for job in CONTROL_JOBS], *extra,
    ]


def _run(ref, *extra: str) -> int:
    return script.main(_argv(ref, *extra))


@pytest.fixture
def evaluated(reference, capsys):
    assert _run(reference, "--phase", "evaluate") == script.EXIT_OK
    capsys.readouterr()
    return reference


def _doctor(path: Path, edit) -> None:
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


def _never_pass(*args, **kwargs):
    raise AssertionError("reference_trajectories ran; this refusal must come before any pass")


# ---------------------------------------------------------------------------
# The parser and the statuses.
# ---------------------------------------------------------------------------


def test_the_parser_defaults_and_the_pinned_control_are_the_specs():
    args = script._parser().parse_args([])
    assert args.out == Path("runs/m3g_stages") and args.reference == Path("runs/m3_study_v2")
    assert args.control == script.CONTROL_DIR == Path("runs/m3f_ladder/step4000")
    assert args.control_arm == script.CONTROL_ARM == "frozen_ssl"
    assert tuple(args.control_seeds) == script.CONTROL_SEEDS == (1, 2)
    assert args.data == Path("data/my_way_home") and args.device == "mps"
    assert args.context is None and args.horizon is None
    assert args.arms == ["pixel_ae", "frozen_ssl", "random_vit"] and args.seeds == [0, 1, 2]
    assert args.phase == script.PHASES[-1] and args.figure is None
    assert script.CONTROL_LABEL == "control"


def test_out_equal_to_reference_or_control_is_argparses_own_usage_error(tmp_path):
    """A record written into a study directory is a study directory changed:
    judged at the parser, before any directory is read."""
    same = str(tmp_path / "study")
    with pytest.raises(SystemExit) as raised:
        script.main(["--out", same, "--reference", same])
    assert raised.value.code == 2
    with pytest.raises(SystemExit) as raised:
        script.main(["--out", same, "--control", same])
    assert raised.value.code == 2


def test_fewer_than_two_distinct_control_seeds_is_a_usage_error():
    """With one seed no stage can pass (SEEDS_REQUIRED = 2), so the control
    would read ENCODE_FAILS vacuously and validate nothing."""
    for seeds in (["1"], ["1", "1"]):
        with pytest.raises(SystemExit) as raised:
            script.main(["--control-seeds", *seeds])
        assert raised.value.code == 2
    assert SEEDS_REQUIRED == 2


def test_the_reused_statuses_are_trust_horizons_and_34_is_new():
    assert script.EXIT_OK == trust.EXIT_OK == 0
    assert script.EXIT_NO_CHECKPOINTS == trust.EXIT_NO_CHECKPOINTS == 11
    assert script.EXIT_SPLIT_MISMATCH == trust.EXIT_SPLIT_MISMATCH == 12
    assert script.EXIT_RECORD_MISMATCH == trust.EXIT_RECORD_MISMATCH == 14
    assert script.EXIT_SELF_CHECK_FAILED == trust.EXIT_SELF_CHECK_FAILED == 30
    assert script.EXIT_CONTROL_MISREAD == 34


# ---------------------------------------------------------------------------
# evaluate.
# ---------------------------------------------------------------------------


def test_a_missing_cell_or_control_is_exit_11_before_any_pass_runs(reference, monkeypatch, capsys):
    monkeypatch.setattr(script, "reference_trajectories", _never_pass)
    assert _run(reference, "--phase", "evaluate", "--seeds", "0") == script.EXIT_NO_CHECKPOINTS
    assert "NO CELL" in capsys.readouterr().out
    assert _run(reference, "--phase", "evaluate", "--control-seeds", "1", "3") == script.EXIT_NO_CHECKPOINTS
    assert "NO CELL" in capsys.readouterr().out
    assert not reference.stages.exists()


def test_evaluate_writes_one_record_per_cell_and_per_control_with_the_spec_shapes(evaluated):
    cell = load_record(evaluated.stages / STAGES_RECORD)
    controls = [load_record(evaluated.stages / name) for name in CONTROL_RECORDS]
    for record in (cell, *controls):
        assert set(record) == RECORD_KEYS
        assert record["context"] == CONTEXT and record["horizon"] == HORIZON
        assert record["decision_h"] == HORIZON  # 15 clamped to the run's horizon
        assert record["windows"]["total"] == WINDOWS and record["windows"]["clusters"] == CLUSTERS
        assert len(record["windows"]["episode"]) == WINDOWS
        assert record["latent"] == {"groups": GROUPS, "classes": CLASSES}
        assert record["identity"] is True
        assert record["git_sha"] == cell["git_sha"]
        assert np.asarray(record["information"]).shape == (WINDOWS,)
        assert (np.asarray(record["information"]) >= 0).all()
        for name in ("teacher", "persistence", "marginal"):
            values = np.asarray(record["accuracy"][name], dtype=float)
            assert values.shape == (WINDOWS,) and (0 <= values).all() and (values <= 1).all()
        for name in ("accuracy", "persistence", "marginal"):
            values = np.asarray(record["open"][name], dtype=float)
            assert values.shape == (WINDOWS, HORIZON) and (0 <= values).all() and (values <= 1).all()
        for name in ("persistence_distance", "distance_to_truth", "probe_persistence", "probe_model"):
            values = np.asarray(record["decode"][name], dtype=float)
            assert values.shape == (WINDOWS, HORIZON) and (values >= 0).all()
        assert np.asarray(record["decode"]["moved"]).shape == (WINDOWS, HORIZON)
        assert np.asarray(record["decode"]["moved"]).dtype == bool
        assert np.asarray(record["companions"]["entropy"]).shape == (GROUPS,)
        assert np.asarray(record["companions"]["marginal_classes"]).shape == (GROUPS,)
        assert np.asarray(record["companions"]["nll"]).shape == (WINDOWS,)
        assert np.asarray(record["companions"]["rendering_median"]).shape == (HORIZON,)
        assert np.asarray(record["companions"]["jitter_median"]).shape == (HORIZON,)
    assert cell["kind"] == "cell" and cell["label"] == JOB.arm and cell["step"] == JOB_KW["steps"]
    assert cell["self_check"]["ok"] is True
    assert cell["self_check"]["reference_position_max_delta"] == 0.0
    assert cell["source"] == str(evaluated.out)
    for record, job in zip(controls, CONTROL_JOBS):
        assert record["kind"] == "control" and record["label"] == script.CONTROL_LABEL
        assert record["arm"] == job.arm and record["seed"] == job.seed
        assert record["step"] == CONTROL_KW["steps"] and record["self_check"] is None
        assert record["source"] == str(evaluated.control)


def test_evaluate_is_deterministic_on_cpu(evaluated, capsys):
    """The same pass twice writes the same bytes: the latents are the rollout
    the gate scored, seeded, and nothing in the statistics draws."""
    before = {name: (evaluated.stages / name).read_text() for name in (STAGES_RECORD, *CONTROL_RECORDS)}
    assert _run(evaluated, "--phase", "evaluate") == script.EXIT_OK
    capsys.readouterr()
    for name, text in before.items():
        assert (evaluated.stages / name).read_text() == text, name


def test_the_step_and_the_source_tell_a_control_from_its_cell(evaluated):
    """The control at seed 1 is the SAME arm and seed as the cell, two steps
    old: what tells them apart in the records is the kind, the label, the
    step and the source -- never the file name alone."""
    cell = load_record(evaluated.stages / STAGES_RECORD)
    control = load_record(evaluated.stages / CONTROL_RECORDS[0])
    assert (cell["arm"], cell["seed"]) == (control["arm"], control["seed"])
    assert cell["step"] != control["step"] and cell["source"] != control["source"]
    assert cell["label"] != control["label"]


def test_a_split_that_is_not_the_records_is_exit_12_and_a_protocol_flag_is_exit_14(reference, monkeypatch, capsys):
    monkeypatch.setattr(script, "reference_trajectories", _never_pass)
    assert _run(reference, "--phase", "evaluate", "--context", "3") == script.EXIT_RECORD_MISMATCH
    assert "RECORD MISMATCH" in capsys.readouterr().out
    _doctor(reference.out / RECORD, lambda r: r["episodes"].__setitem__("val", ["ep_000099_len00040.npz"]))
    assert _run(reference, "--phase", "evaluate") == script.EXIT_SPLIT_MISMATCH
    assert "SPLIT MISMATCH" in capsys.readouterr().out


def test_a_record_whose_curve_no_longer_reproduces_is_exit_14_before_the_trust_pass(reference, monkeypatch, capsys):
    _doctor(reference.out / RECORD, lambda r: r["curves"]["rssm_position"].__setitem__(
        0, r["curves"]["rssm_position"][0] + 1e-3))
    monkeypatch.setattr(script, "reference_trajectories", _never_pass)
    assert _run(reference, "--phase", "evaluate") == script.EXIT_RECORD_MISMATCH
    assert "no longer reproduces" in capsys.readouterr().out


def test_a_doctored_diagnostic_is_exit_30_and_writes_no_record(reference, capsys):
    _doctor(reference.out / DIAGNOSTIC, lambda d: d["curves"]["reference_position"].__setitem__(
        0, d["curves"]["reference_position"][0] + 1e-3))
    assert _run(reference, "--phase", "evaluate") == script.EXIT_SELF_CHECK_FAILED
    assert "SELF-CHECK FAILED" in capsys.readouterr().out
    assert not (reference.stages / STAGES_RECORD).exists()


def test_a_broken_step_one_identity_is_exit_30_by_name_and_writes_no_record(reference, monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise LatentIdentityError("window at start 0: the open-loop prior at step 1 is not the teacher-forced prior")

    monkeypatch.setattr(script, "reference_trajectories", broken)
    assert _run(reference, "--phase", "evaluate") == script.EXIT_SELF_CHECK_FAILED
    out = capsys.readouterr().out
    assert "SELF-CHECK FAILED" in out and "window at start 0" in out
    assert not (reference.stages / STAGES_RECORD).exists()


def test_a_control_whose_checkpoint_is_missing_is_exit_11_before_any_pass(reference, monkeypatch, capsys):
    (reference.control / "world_model_random_vit_seed2.pt").unlink()
    monkeypatch.setattr(script, "reference_trajectories", _never_pass)
    assert _run(reference, "--phase", "evaluate") == script.EXIT_NO_CHECKPOINTS
    assert "seed 2: no checkpoint" in capsys.readouterr().out


def test_the_decode_inputs_and_the_marginal_classes_are_what_read_will_consume(evaluated):
    """The logits are not written, so the record is pinned on what it does
    carry for `read`: the two distances of each decode channel are finite
    `(n, H)` arrays whose difference is the margin `read` pools, and the
    marginal classes are valid class indices."""
    record = load_record(evaluated.stages / STAGES_RECORD)
    decode = {k: np.asarray(v, dtype=float) for k, v in record["decode"].items() if k != "moved"}
    free = decode["persistence_distance"] - decode["distance_to_truth"]
    assert free.shape == (WINDOWS, HORIZON)
    probe = decode["probe_persistence"] - decode["probe_model"]
    assert np.isfinite(probe).all()
    assert record["companions"]["marginal_classes"] == [int(c) for c in record["companions"]["marginal_classes"]]
    assert all(0 <= c < CLASSES for c in record["companions"]["marginal_classes"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_stage_decomposition_script.py -q`
Expected: FAIL at collection — `FileNotFoundError` loading `scripts/stage_decomposition.py`.

- [ ] **Step 3: Write the script**

Create `scripts/stage_decomposition.py`:

```python
"""The stage decomposition over the M3c cells: which stage of imagination fails?

M3e ruled memorisation out and M3f ruled checkpoint timing out, and the
quantity both were built around -- the absolute `embedding` loss -- is
confounded by the embedding's own scale (spec 2026-09-19 M3g, section 1).
What is not known is which STAGE of imagination fails at step 20,000:
whether the posterior carries the frame (encode), whether one prior step
from the true state predicts the next posterior (predict), whether that
survives fifteen open-loop steps on the prior's own samples (carry), or
whether a latent that tracks is lost in the rendering the gate scores
(decode). This script measures the four in the model's own 32 x 32
categorical latent, where the posterior on the validation windows is
exactly known:

  evaluate  per cell: `prepare_cell` on the reference study (12, 14), the
            trust pass with `keep_latents` -- the rollout the gate scored,
            plus the posterior over the window, the teacher-forced and the
            open-loop prior -- the bitwise self-check against the cell's
            diagnostic (30), the step-1 identity (30), the statistics of
            spec 2.3, one stages_<arm>_seed<n>.json. Then the CONTROL cells
            -- known-blind rung-4000 checkpoints from the M3f ladder, loaded
            as rung cells -- through the same pass, into
            stages_control_<arm>_seed<n>.json.
  read      pool the records, read the control (34), decide Reading S,
            print the tables, write stages.txt and stages_curves.png.

LOADING IS `trust_horizon.py`'S and `checkpoint_ladder.py`'S: `Cell`,
`load_cell`, `self_check`, `prepare_cell` and `rung_cell` are imported by
path, so a cell is refused here for the reasons and in the words the other
tools refuse it.

THE CHECKS, BY PHASE, each with its own status:

  evaluate: EXIT_NO_CHECKPOINTS (11)     a requested cell or control lacks its
                                          checkpoint or record (a cell, also its
                                          diagnostic); judged for every one
                                          before any pass runs.
            EXIT_SPLIT_MISMATCH (12)      the split by name is not the record's.
            EXIT_RECORD_MISMATCH (14)     --context/--horizon disagree with the
                                          protocol, or evaluate_rollout no longer
                                          reproduces the record's curve.
            EXIT_SELF_CHECK_FAILED (30)   a cell's trust pass is not bitwise its
                                          diagnostic, or (cell or control) the
                                          open-loop prior at step 1 is not the
                                          teacher-forced prior at step 1.
  read:     EXIT_NO_CHECKPOINTS (11)     a requested record is missing.
            EXIT_SELF_CHECK_FAILED (30)   a record's recorded self-check is not
                                          ok, or its identity flag is not set.
            EXIT_CONTROL_MISREAD (34)     NEW. the known-blind control reads
                                          anything but ENCODE_FAILS: no arm's
                                          reading is printed, no stages.txt.

11 / 12 / 14 / 30 carry `trust_horizon.py`'s meanings on purpose; 34 is in
no other tool's range (run_study 1/3-6/23, report_study 7-10, spike 10,
diagnose 11-17, pool 18-22, trust 30, split_gap 31, ladder 32-33, argparse
2, a traceback 1).
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.aggregate import SEEDS
from mbfps.eval.diagnostics import LatentIdentityError, reference_trajectories
from mbfps.eval.split_gap import decision_horizon
from mbfps.eval.stages import (
    SEEDS_REQUIRED,
    entropy_by_group,
    information,
    marginal_accuracy,
    marginal_classes,
    open_accuracy,
    open_marginal,
    open_persistence,
    persistence_accuracy,
    teacher_accuracy,
    teacher_nll,
)
from mbfps.eval.study import SPLIT_SEED, git_sha, load_record, write_record
from mbfps.eval.trust import moved_mask
from mbfps.utils.config import ARMS
from mbfps.utils.device import get_device


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_stage_decomposition", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_trust = _sibling("trust_horizon")
Cell = _trust.Cell
CellMissing = _trust.CellMissing
load_cell = _trust.load_cell
self_check = _trust.self_check
prepare_cell = _trust.prepare_cell
_ladder = _sibling("checkpoint_ladder")
rung_cell = _ladder.rung_cell

EXIT_OK = _trust.EXIT_OK
EXIT_NO_CHECKPOINTS = _trust.EXIT_NO_CHECKPOINTS
EXIT_SPLIT_MISMATCH = _trust.EXIT_SPLIT_MISMATCH
EXIT_RECORD_MISMATCH = _trust.EXIT_RECORD_MISMATCH
EXIT_SELF_CHECK_FAILED = _trust.EXIT_SELF_CHECK_FAILED
EXIT_CONTROL_MISREAD = 34
"""0 / 11 / 12 / 14 / 30 are `trust_horizon.py`'s, imported so they cannot
drift; 34 is new and in no other tool's range."""

PHASES: tuple[str, ...] = ("evaluate",)
CONTROL_ARM: str = "frozen_ssl"
CONTROL_SEEDS: tuple[int, ...] = (1, 2)
CONTROL_DIR: Path = Path("runs/m3f_ladder/step4000")
"""Spec 2.1: the known-blind control -- frozen_ssl seeds 1 and 2 at rung 4000
of the M3f ladder (reconstruction R^2 ~ 0, posterior equal to prior)."""
CONTROL_LABEL: str = "control"
"""The arm label the control's records pool under: a fourth arm to the
pooling, never one of the three."""
KINDS: tuple[str, ...] = ("cell", "control")


# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------


def stages_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    return Path(out_dir) / f"stages_{arm}_seed{seed}.json"


def control_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    return Path(out_dir) / f"stages_control_{arm}_seed{seed}.json"


def record_path(out_dir: Path, kind: str, arm: str, seed: int) -> Path:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return (stages_record_path if kind == "cell" else control_record_path)(out_dir, arm, seed)


# ---------------------------------------------------------------------------
# evaluate: one cell's pass, statistics and record.
# ---------------------------------------------------------------------------


_cell_args = _ladder._cell_args
"""`prepare_cell` reads `out`, `device`, `context` and `horizon` off its
args; a cell is read with `out` pointed at the directory it lives in -- the
ladder's helper, imported rather than copied."""


def cell_statistics(traj, *, context: int) -> dict:
    """Spec 2.3 over one cell's kept latents and rows: every per-window
    series `read` pools, the components the tables print, the companions.
    The decode inputs are stored as the two distances of each channel so
    `read` takes the margin at its own h through `stages.decode_margin`."""
    post, teacher, opened = traj.post_logits, traj.prior_teacher_logits, traj.prior_open_logits
    classes = marginal_classes(post, context)
    true_positions = np.asarray(traj.true_positions, dtype=np.float64)
    persistence_position = np.linalg.norm(
        np.asarray(traj.positions_at_context, dtype=np.float64)[:, None, :] - true_positions, axis=-1
    )
    model_position = np.linalg.norm(np.asarray(traj.positions, dtype=np.float64) - true_positions, axis=-1)
    return {
        "latent": {"groups": int(post.shape[-2]), "classes": int(post.shape[-1])},
        "information": information(post, teacher, context),
        "accuracy": {
            "teacher": teacher_accuracy(post, teacher, context),
            "persistence": persistence_accuracy(post, context),
            "marginal": marginal_accuracy(post, classes, context),
        },
        "open": {
            "accuracy": open_accuracy(post, opened, context),
            "persistence": open_persistence(post, context),
            "marginal": open_marginal(post, classes, context),
        },
        "decode": {
            "persistence_distance": np.asarray(traj.embedding_persistence_distance, dtype=np.float64),
            "distance_to_truth": np.asarray(traj.embedding_distance_to_truth, dtype=np.float64),
            "probe_persistence": persistence_position,
            "probe_model": model_position,
            "moved": moved_mask(traj.true_positions, traj.true_at_context),
        },
        "companions": {
            "entropy": entropy_by_group(post[:, context:]),
            "marginal_classes": classes,
            "nll": teacher_nll(post, teacher, context),
            "rendering_median": np.median(traj.posterior_rendering_distance, axis=0),
            "jitter_median": np.median(traj.true_step_displacement, axis=0),
        },
    }


def stages_record(cell: Cell, traj, stats: dict, *, kind: str, source: Path, context: int,
                  horizon: int, h: int, device, check) -> dict:
    """The LIVE record for one cell or control: numpy arrays and real NaNs;
    `write_record` sanitises it. `label` is what the pooling groups by -- the
    arm for a cell, `CONTROL_LABEL` for a control -- so a control at the same
    arm and seed as a cell never pools with it."""
    return {
        "arm": cell.arm,
        "seed": int(cell.seed),
        "kind": kind,
        "label": cell.arm if kind == "cell" else CONTROL_LABEL,
        "source": str(source),
        "step": int(cell.record["steps"]),
        "record_git_sha": cell.record["git_sha"],
        "context": int(context),
        "horizon": int(horizon),
        "decision_h": int(h),
        "split_seed": SPLIT_SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "git_sha": git_sha(),
        "episodes": {"val": list(cell.record["episodes"]["val"])},
        "windows": {
            "total": int(traj.windows_total),
            "episode": [int(e) for e in traj.window_episode],
            "clusters": int(np.unique(np.asarray(traj.window_episode)).size),
        },
        "self_check": None if check is None else check.record(),
        "identity": True,
        **stats,
    }


def evaluate_cell(args, cell: Cell, device, train, val, *, kind: str, source: Path) -> tuple[int, dict | None]:
    """One cell or control: 12, the protocol (14), the refit, the
    reproduction (14), the trust pass with its latents, the identity (30), a
    cell's self-check (30), the statistics, the record."""
    status, prepared = prepare_cell(_cell_args(args, source), cell, device, train, val)
    if status != EXIT_OK:
        return status, None
    try:
        traj = reference_trajectories(
            prepared.model, val, prepared.embedding_probe, keep_latents=True, **prepared.common,
        )
    except LatentIdentityError as error:
        print(
            f"\nSELF-CHECK FAILED for {cell.arm} seed {cell.seed} ({kind}): {error}. The "
            "open-loop and teacher-forced passes did not start from the same state, so nothing "
            "read off them is one rollout. No record written."
        )
        return EXIT_SELF_CHECK_FAILED, None
    check = None
    if kind == "cell":
        check = self_check(traj, cell.diagnostic)
        if not check.ok:
            print(
                f"\nSELF-CHECK FAILED for {cell.arm} seed {cell.seed}: " + "; ".join(check.failures())
                + ". Same windows, same rollout, same refit probe -- or this is not the rollout "
                "the gate scored. No record written."
            )
            return EXIT_SELF_CHECK_FAILED, None
    h, _ = decision_horizon(prepared.horizon)
    stats = cell_statistics(traj, context=prepared.context)
    record = stages_record(
        cell, traj, stats, kind=kind, source=source, context=prepared.context,
        horizon=prepared.horizon, h=h, device=device, check=check,
    )
    path = record_path(args.out, kind, cell.arm, cell.seed)
    write_record(path, record)
    accuracy = stats["accuracy"]
    print(
        f"{cell.arm} seed {cell.seed} ({kind}, step {record['step']}): information "
        f"{float(np.mean(stats['information'])):.3f} nats; teacher {float(np.mean(accuracy['teacher'])):.3f} / "
        f"persistence {float(np.mean(accuracy['persistence'])):.3f} / marginal "
        f"{float(np.mean(accuracy['marginal'])):.3f}; open({h}) "
        f"{float(np.mean(stats['open']['accuracy'][:, h - 1])):.3f}; wrote {path}"
    )
    return EXIT_OK, record


def load_requested(args, cells, controls) -> list[tuple[str, Path, Cell]]:
    """Every requested cell and control, loaded BEFORE any pass runs, so a
    missing one is reported (11) before minutes are spent."""
    loaded = []
    for arm, seed in cells:
        loaded.append(("cell", args.reference, load_cell(args.reference, arm, seed)))
    for arm, seed in controls:
        loaded.append(("control", args.control, rung_cell(args.control, arm, seed)))
    return loaded


def evaluate_phase(args, cells, controls, device, train, val) -> int:
    try:
        loaded = load_requested(args, cells, controls)
    # Two classes, one name: `_sibling` executes each script afresh, so the
    # ladder's `rung_cell` raises the ladder's own `CellMissing`, a distinct
    # class from the one this module took from trust_horizon.
    except (CellMissing, _ladder.CellMissing) as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    args.out.mkdir(parents=True, exist_ok=True)
    for kind, source, cell in loaded:
        status, _ = evaluate_cell(args, cell, device, train, val, kind=kind, source=source)
        if status != EXIT_OK:
            return status
    return EXIT_OK


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/m3g_stages"))
    parser.add_argument("--reference", type=Path, default=Path("runs/m3_study_v2"),
                        help="the M3c study directory: the nine 20,000-step cells")
    parser.add_argument("--control", type=Path, default=CONTROL_DIR,
                        help="the rung directory holding the known-blind control cells")
    parser.add_argument("--control-arm", default=CONTROL_ARM, choices=list(ARMS))
    parser.add_argument("--control-seeds", nargs="+", type=int, default=list(CONTROL_SEEDS))
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--device", default="mps")
    # None -> the cell's diagnostic (a control: its record) says what it was
    # written at; a value that disagrees is refused (`protocol_mismatch`, 14).
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--phase", choices=PHASES, default=PHASES[-1])
    parser.add_argument("--figure", type=Path, default=None,
                        help="the curves figure; default <out>/stages_curves.png")
    return parser


def _check_usage(parser: argparse.ArgumentParser, args) -> None:
    for name, other in (("--reference", args.reference), ("--control", args.control)):
        if args.out.resolve() == Path(other).resolve():
            parser.error(
                f"--out and {name} are the same directory; a record written into a study "
                "directory is a study directory changed"
            )
    if len(set(args.control_seeds)) < SEEDS_REQUIRED:
        parser.error(
            f"--control-seeds needs at least {SEEDS_REQUIRED} distinct seeds; with fewer no stage "
            "can pass, so the control would read ENCODE_FAILS vacuously"
        )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _check_usage(parser, args)
    device = get_device(prefer=args.device)
    cells = [(arm, int(seed)) for arm in args.arms for seed in args.seeds]
    controls = [(args.control_arm, int(seed)) for seed in args.control_seeds]
    if args.phase in ("evaluate", "all"):
        buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
        train, val = episode_split(buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
        status = evaluate_phase(args, cells, controls, device, train, val)
        if status != EXIT_OK:
            return status
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_stage_decomposition_script.py -q`
Expected: 14 passed. (The fixture trains three tiny cells on CPU and runs the diagnostic; expect a minute or two.)

- [ ] **Step 5: Commit**

```bash
git add scripts/stage_decomposition.py tests/eval/test_stage_decomposition_script.py
git commit -m "feat: stage_decomposition.py evaluate -- each cell through prepare_cell, the trust pass with its latents, the self-check and the step-1 identity, the statistics of spec 2.3 and one record; the known-blind controls through the same pass as rung cells

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `scripts/stage_decomposition.py` — `read`

**Files:**
- Modify: `scripts/stage_decomposition.py` (imports, `PHASES`, append the read half, replace `main`)
- Test: `tests/eval/test_stage_decomposition_script.py` (append)

**Interfaces:**
- Consumes: Task 3's `ArmInputs`, `StagesInputs`, `Status`, `StagesReading`, `reading_stages`, `format_reading_stages`, `FAMILY`, `REPORTED_K`, `STAGES`, `KL_FREE_BITS`; `split_gap.{StratumContrast, cell_series, fmt_z}`; `pooling.{pool_arm, cluster_threshold, CellSeries}`; `trust_readings.ARMS_ORDER`; `stages.decode_margin`.
- Produces: `PHASES = ("evaluate", "read", "all")`; `CONTRASTS`; `contrast_values(record, name, h) -> (values, changed)`; `pooled(cells) -> StratumContrast`; `arm_inputs(records, label, seeds, *, h) -> ArmInputs`; `stages_inputs(records, *, arms, seeds, control_seeds, h) -> (StagesInputs, StagesInputs)`; the tables; `write_curves`; `readings_text`; `write_readings`; `load_records`; `read_phase(args, cells, controls) -> status`; `main` handling `read` and `all`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_stage_decomposition_script.py`:

```python


# ---------------------------------------------------------------------------
# read: the pooling glue, the control rule, Reading S, stages.txt.
# ---------------------------------------------------------------------------

from mbfps.eval.pooling import cluster_threshold  # noqa: E402
from mbfps.eval.split_gap import StratumContrast  # noqa: E402
from mbfps.eval.stages import (  # noqa: E402
    FAMILY,
    STAGES,
    ArmReading,
    StageResult,
    StagesReading,
    Status,
)

SECTIONS = (
    "--- self-check per record",
    "--- stage components per arm",
    "--- carry: open-loop accuracy against k",
    "--- decode: the paired margin",
    "--- companions (change no verdict)",
    "--- the known-answer control",
    "--- Reading S: the first failing stage at h=3",
    "--- per seed",
)


@pytest.fixture
def stages_run(evaluated, capsys):
    assert _run(evaluated, "--phase", "read") == script.EXIT_OK
    text = capsys.readouterr().out
    return types.SimpleNamespace(ref=evaluated, text=text, path=evaluated.stages / "stages.txt")


def _fabricated_reading(status: Status):
    """A `reading_stages` stand-in giving every arm one status, with a full
    stage table so the control text can still be formatted."""
    def fake(inputs):
        arms = {}
        for arm in inputs.arms:
            stages = {
                stage: StageResult(stage=stage, passes=True, wording="passes",
                                   rule=f"{stage} passes: fabricated", seeds_holding=2, seeds_total=2)
                for stage in STAGES
            }
            arms[arm] = ArmReading(arm=arm, status=status, rule="fabricated", stages=stages)
        return StagesReading(arms=arms, h=inputs.h, z_fam=inputs.z_fam)
    return fake


def _record(n=4, horizon=3, information=(0.5, 0.1, 0.3, 0.9)):
    """A fabricated stages record: four windows, three steps, hand-typed series."""
    return {
        "label": "arm", "seed": 0, "arm": "arm", "kind": "cell",
        "episodes": {"val": ["a", "b"]}, "horizon": horizon, "context": 2, "decision_h": horizon,
        "device": "cpu", "torch_version": "x",
        "windows": {"total": n, "episode": [0, 0, 1, 1], "clusters": 2},
        "information": list(information),
        "accuracy": {"teacher": [0.9, 0.8, 0.7, 0.6], "persistence": [0.5, 0.5, 0.5, 0.5],
                     "marginal": [0.4, 0.3, 0.2, 0.1]},
        "open": {"accuracy": [[1.0, 0.8, 0.6]] * n, "persistence": [[0.5, 0.4, 0.3]] * n,
                 "marginal": [[0.2, 0.2, 0.2]] * n},
        "decode": {"persistence_distance": [[3.0, 4.0, 5.0]] * n, "distance_to_truth": [[1.0, 2.0, 6.0]] * n,
                   "probe_persistence": [[30.0, 40.0, 50.0]] * n, "probe_model": [[10.0, 20.0, 60.0]] * n,
                   "moved": [[True, True, True], [True, True, False], [True, True, True], [False, False, False]]},
    }


def test_contrast_values_are_the_pre_registered_differences_under_their_masks():
    r = _record()
    values, changed = script.contrast_values(r, "encode", 3)
    np.testing.assert_allclose(values, np.array([0.5, 0.1, 0.3, 0.9]) - KL_FREE_BITS)
    assert changed.all() and changed.shape == (4,)
    np.testing.assert_allclose(script.contrast_values(r, "predict_persistence", 3)[0], [0.4, 0.3, 0.2, 0.1])
    np.testing.assert_allclose(script.contrast_values(r, "predict_marginal", 3)[0], [0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(script.contrast_values(r, "carry", 2)[0], [0.4] * 4)
    np.testing.assert_allclose(script.contrast_values(r, "carry_marginal", 3)[0], [0.4] * 4)
    values, changed = script.contrast_values(r, "decode", 3)
    np.testing.assert_allclose(values, [-1.0] * 4)
    np.testing.assert_array_equal(changed, [True, False, True, False])
    values, changed = script.contrast_values(r, "decode_probe", 1)
    np.testing.assert_allclose(values, [20.0] * 4)
    np.testing.assert_array_equal(changed, [True, True, True, False])
    assert script.CONTRASTS == ("encode", "predict_persistence", "predict_marginal", "carry", "decode")
    with pytest.raises(KeyError):
        script.contrast_values(r, "render", 1)


def test_pooled_is_pool_arm_reduced_to_a_contrast_and_nan_with_nothing_to_pool():
    r = _record()
    series = script._series(r, *script.contrast_values(r, "encode", 3), channel="encode")
    p = script.pooled([series])
    assert isinstance(p, StratumContrast)
    assert p.estimate == pytest.approx(np.mean([0.5, 0.1, 0.3, 0.9]) - KL_FREE_BITS)
    assert p.clusters == 2 and np.isfinite(p.se)
    unmoved = dict(r, decode=dict(r["decode"], moved=[[False] * 3] * 4))
    empty = script.pooled([script._series(unmoved, *script.contrast_values(unmoved, "decode", 3), channel="decode")])
    assert np.isnan(empty.z) and empty.clusters == 0
    assert np.isnan(script.pooled([]).z)


def test_stages_inputs_pool_each_arm_and_the_control_under_the_family_bar(stages_run):
    status, records = script.load_records(
        types.SimpleNamespace(out=stages_run.ref.stages),
        [(JOB.arm, JOB.seed)], [(job.arm, job.seed) for job in CONTROL_JOBS],
    )
    assert status == script.EXIT_OK
    assert set(records) == {(JOB.arm, JOB.seed), (script.CONTROL_LABEL, 1), (script.CONTROL_LABEL, 2)}
    inputs, control = script.stages_inputs(
        records, arms=[JOB.arm], seeds=[JOB.seed], control_seeds=[1, 2], h=HORIZON,
    )
    assert inputs.z_fam == control.z_fam == cluster_threshold(FAMILY, CLUSTERS)
    assert inputs.h == control.h == HORIZON
    assert list(inputs.arms) == [JOB.arm] and list(control.arms) == [script.CONTROL_LABEL]
    a = inputs.arms[JOB.arm]
    assert set(a.per_seed) == {JOB.seed} and a.per_seed[JOB.seed].per_seed is None
    assert set(control.arms[script.CONTROL_LABEL].per_seed) == {1, 2}
    for stage in STAGES:
        for _, contrast in a.contrasts(stage):
            assert contrast.clusters == CLUSTERS and np.isfinite(contrast.estimate)


def test_read_without_records_is_exit_11(reference, capsys):
    assert _run(reference, "--phase", "read") == script.EXIT_NO_CHECKPOINTS
    assert "run --phase evaluate first" in capsys.readouterr().out


def test_read_refuses_a_record_whose_self_check_is_not_ok_or_whose_identity_is_unset(evaluated, capsys):
    path = evaluated.stages / STAGES_RECORD
    original = path.read_text()
    _doctor(path, lambda r: r["self_check"].update({"ok": False}))
    assert _run(evaluated, "--phase", "read") == script.EXIT_SELF_CHECK_FAILED
    assert "SELF-CHECK FAILED" in capsys.readouterr().out
    path.write_text(original)
    _doctor(path, lambda r: r["self_check"].update({"reference_position_max_delta": 1e-6}))
    assert _run(evaluated, "--phase", "read") == script.EXIT_SELF_CHECK_FAILED
    capsys.readouterr()
    path.write_text(original)
    _doctor(evaluated.stages / CONTROL_RECORDS[1], lambda r: r.__setitem__("identity", False))
    assert _run(evaluated, "--phase", "read") == script.EXIT_SELF_CHECK_FAILED
    assert "identity" in capsys.readouterr().out
    assert not (evaluated.stages / "stages.txt").exists()


def test_read_raises_on_records_at_different_protocols(evaluated):
    _doctor(evaluated.stages / CONTROL_RECORDS[0], lambda r: r.__setitem__("horizon", 2))
    with pytest.raises(ValueError, match="horizon"):
        _run(evaluated, "--phase", "read")


def test_a_control_that_does_not_read_encode_fails_is_exit_34_with_no_reading_and_no_file(evaluated, monkeypatch, capsys):
    """The known-blind control is the instrument's validation: read as
    anything but ENCODE_FAILS, the control tables are printed with the
    sentence, no arm's reading is, and stages.txt is not written."""
    monkeypatch.setattr(script, "reading_stages", _fabricated_reading(Status.NO_STAGE_FAILS))
    assert _run(evaluated, "--phase", "read") == script.EXIT_CONTROL_MISREAD
    out = capsys.readouterr().out
    assert "CONTROL MISREAD" in out and "NO STAGE FAILS" in out and "fabricated" in out
    assert "--- the known-answer control" in out
    assert "--- Reading S" not in out.split("CONTROL MISREAD")[1]
    assert f"verdict: {JOB.arm}" not in out
    assert not (evaluated.stages / "stages.txt").exists()


def test_a_control_reading_encode_fails_lets_the_reading_through(evaluated, monkeypatch, capsys):
    monkeypatch.setattr(script, "reading_stages", _fabricated_reading(Status.ENCODE_FAILS))
    assert _run(evaluated, "--phase", "read") == script.EXIT_OK
    out = capsys.readouterr().out
    assert "CONTROL MISREAD" not in out and f"verdict: {JOB.arm}" in out
    assert (evaluated.stages / "stages.txt").exists()


def test_read_prints_every_section_writes_it_byte_identical_and_draws_the_figure(stages_run):
    text = stages_run.text
    for section in SECTIONS:
        assert section in text, section
    assert stages_run.path.read_text() == text
    assert "z_fam = cluster_threshold(5, 6)" in text
    assert f"verdict: {JOB.arm:<12}ENCODE FAILS" in text  # one seed: nothing can replicate
    assert "seeds holding" in text and "this seed alone" in text
    assert "control" in text and "must read ENCODE FAILS" in text
    assert "NOTE: decision horizon clamped to the run's horizon h=3" in text
    figure = stages_run.ref.stages / "stages_curves.png"
    assert (f"figure={figure}" in text) == figure.exists()
    assert "figure NOT written" in text or figure.stat().st_size > 0


def test_read_is_idempotent_and_all_runs_both_phases(stages_run, capsys):
    assert _run(stages_run.ref, "--phase", "read") == script.EXIT_OK
    assert capsys.readouterr().out == stages_run.text
    assert _run(stages_run.ref, "--phase", "all") == script.EXIT_OK
    out = capsys.readouterr().out
    assert out.endswith(stages_run.text) and "wrote" in out


def test_the_groups_order_the_arms_as_the_readings_do_and_put_the_control_last():
    records = {("random_vit", 0): {}, ("pixel_ae", 0): {}, ("control", 1): {}, ("control", 2): {}}
    groups = script._groups(records, ["random_vit", "pixel_ae"], [0], [1, 2])
    assert [label for label, _ in groups] == ["pixel_ae", "random_vit", "control"]
    assert groups[-1][1] == [("control", 1), ("control", 2)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_stage_decomposition_script.py -q -k "read or contrast_values or pooled or stages_inputs or groups or control"`
Expected: FAIL — `AttributeError: module has no attribute 'contrast_values'`, and the `--phase read` runs are argparse errors (`SystemExit: 2`) until `PHASES` grows.

- [ ] **Step 3: Extend the imports and `PHASES`**

Replace the `from mbfps.eval.split_gap import decision_horizon` line and the `from mbfps.eval.stages import (...)` block with:

```python
import mbfps.eval.pooling as pooling
from mbfps.eval.split_gap import StratumContrast, cell_series, decision_horizon, fmt_z
from mbfps.eval.stages import (
    DECISION_H,
    FAMILY,
    KL_FREE_BITS,
    REPORTED_K,
    SEEDS_REQUIRED,
    STAGES,
    ArmInputs,
    StagesInputs,
    StagesReading,
    Status,
    decode_margin,
    entropy_by_group,
    format_reading_stages,
    information,
    marginal_accuracy,
    marginal_classes,
    open_accuracy,
    open_marginal,
    open_persistence,
    persistence_accuracy,
    reading_stages,
    teacher_accuracy,
    teacher_nll,
)
from mbfps.eval.trust_readings import ARMS_ORDER
```

and change `PHASES` to:

```python
PHASES: tuple[str, ...] = ("evaluate", "read", "all")
```

- [ ] **Step 4: Append the read half**

Insert before the `# main` banner:

```python
# ---------------------------------------------------------------------------
# read: pooling glue -- stages records -> the inputs Reading S is decided on.
# ---------------------------------------------------------------------------
# Every estimator is `pooling.py`'s. What is decided here is only WHICH
# per-window series goes in under WHICH mask (spec 3.1): every latent
# contrast over every window; the decode margin over the windows the truth
# moved at h. A record's `label` is the group -- the arm, or "control".

CONTRASTS: tuple[str, ...] = ("encode", "predict_persistence", "predict_marginal", "carry", "decode")
"""The five pre-registered contrasts, `ArmInputs`' fields in order."""


def _series(record: dict, values, changed, *, channel: str) -> pooling.CellSeries:
    """One per-window series of one record through `split_gap.cell_series`;
    `arm` is the record's label, `rung` the stratum (val)."""
    return cell_series(
        record, values, changed,
        arm=record["label"], seed=record["seed"], rung="val", channel=channel,
        val=record["episodes"]["val"], horizon=record["horizon"], context=record["context"],
        device=record["device"], torch_version=record["torch_version"],
    )


def contrast_values(record: dict, name: str, h: int) -> tuple[np.ndarray, np.ndarray]:
    """`(values, changed)` of one contrast of one record at horizon step `h`:
    the five of `CONTRASTS`, plus the companions `carry_marginal` and
    `decode_probe`. The latent contrasts change on every window; the decode
    margins on the windows moved at h."""
    every = np.ones(len(record["windows"]["episode"]), dtype=bool)
    accuracy = {k: np.asarray(v, dtype=np.float64) for k, v in record["accuracy"].items()}
    opened = {k: np.asarray(v, dtype=np.float64) for k, v in record["open"].items()}
    decode = record["decode"]
    if name == "encode":
        return np.asarray(record["information"], dtype=np.float64) - KL_FREE_BITS, every
    if name == "predict_persistence":
        return accuracy["teacher"] - accuracy["persistence"], every
    if name == "predict_marginal":
        return accuracy["teacher"] - accuracy["marginal"], every
    if name == "carry":
        return opened["accuracy"][:, h - 1] - opened["persistence"][:, h - 1], every
    if name == "carry_marginal":
        return opened["accuracy"][:, h - 1] - opened["marginal"][:, h - 1], every
    moved = np.asarray(decode["moved"], dtype=bool)[:, h - 1]
    if name == "decode":
        return decode_margin(decode["persistence_distance"], decode["distance_to_truth"], h), moved
    if name == "decode_probe":
        return decode_margin(decode["probe_persistence"], decode["probe_model"], h), moved
    raise KeyError(name)


_NO_CONTRAST = StratumContrast(estimate=float("nan"), se=float("nan"), z=float("nan"), clusters=0)


def pooled(cells) -> StratumContrast:
    """`pooling.pool_arm` reduced to what the rules read, or the NaN contrast
    when there is nothing to pool -- no cell, or no window every cell
    changed."""
    cells = list(cells)
    if not cells or not np.logical_and.reduce([c.changed for c in cells]).any():
        return _NO_CONTRAST
    p = pooling.pool_arm(cells)
    return StratumContrast(estimate=p.mean, se=p.se, z=p.z, clusters=p.clusters)


def _contrast(records: dict, keys, name: str, h: int) -> StratumContrast:
    return pooled([_series(records[k], *contrast_values(records[k], name, h), channel=name) for k in keys])


def arm_inputs(records: dict, label: str, seeds, *, h: int) -> ArmInputs:
    """One group's `ArmInputs` at `h`: the five contrasts pooled over its
    seeds, and the same within each seed alone."""
    keys = [(label, int(s)) for s in seeds]

    def inputs(subset, per_seed):
        return ArmInputs(
            encode=_contrast(records, subset, "encode", h),
            predict_persistence=_contrast(records, subset, "predict_persistence", h),
            predict_marginal=_contrast(records, subset, "predict_marginal", h),
            carry=_contrast(records, subset, "carry", h),
            decode=_contrast(records, subset, "decode", h),
            per_seed=per_seed,
        )

    return inputs(keys, {key[1]: inputs([key], None) for key in keys})


def _clusters(records: dict) -> int:
    return int(next(iter(records.values()))["windows"]["clusters"])


def _ordered(arms) -> list[str]:
    return sorted(arms, key=lambda a: ARMS_ORDER.index(a) if a in ARMS_ORDER else len(ARMS_ORDER))


def stages_inputs(records: dict, *, arms, seeds, control_seeds, h: int) -> tuple[StagesInputs, StagesInputs]:
    """The arms' inputs and the control's, under one `z_fam` over the val
    stratum's cluster count (spec 3.1)."""
    z_fam = pooling.cluster_threshold(FAMILY, _clusters(records))
    return (
        StagesInputs(arms={arm: arm_inputs(records, arm, seeds, h=h) for arm in _ordered(arms)}, z_fam=z_fam, h=h),
        StagesInputs(arms={CONTROL_LABEL: arm_inputs(records, CONTROL_LABEL, control_seeds, h=h)}, z_fam=z_fam, h=h),
    )


def _groups(records: dict, arms, seeds, control_seeds) -> list[tuple[str, list[tuple[str, int]]]]:
    """`[(label, record keys)]`: the arms in the readings' order, the control last."""
    groups = [(arm, [(arm, int(s)) for s in seeds]) for arm in _ordered(arms)]
    groups.append((CONTROL_LABEL, [(CONTROL_LABEL, int(s)) for s in control_seeds]))
    return groups


# ---------------------------------------------------------------------------
# read: the tables.
# ---------------------------------------------------------------------------


def _num(value, spec: str = ".3f") -> str:
    return fmt_z(float(value), spec)


def _mean_over(records: dict, keys, pick) -> float:
    """The mean over windows of `pick(record)`, then over the group's records."""
    return float(np.mean([np.mean(np.asarray(pick(records[k]), dtype=np.float64)) for k in keys]))


def _self_check_table(records: dict) -> str:
    lines = [
        "--- self-check per record: the pass against the cell's diagnostic (exact; a control has none) "
        "and the step-1 prior identity ---",
        f"  {'kind':<9}{'arm':<12}{'seed':>5}{'step':>7}{'ref max|d|':>12}{'pers max|d|':>13}{'windows':>9}  identity",
    ]
    for (label, seed), r in records.items():
        check = r["self_check"]
        ref = _num(check["reference_position_max_delta"], ".1e") if check else "n/a"
        pers = _num(check["persistence_position_max_delta"], ".1e") if check else "n/a"
        lines.append(
            f"  {r['kind']:<9}{r['arm']:<12}{int(seed):>5}{int(r['step']):>7}{ref:>12}{pers:>13}"
            f"{int(r['windows']['total']):>9}  {'yes' if r['identity'] else 'NO'}"
        )
    return "\n".join(lines) + "\n"


def _components_table(records: dict, groups, h: int) -> str:
    """Every stage's components per group: means over windows and seeds."""
    lines = [
        f"--- stage components per arm and for the control: means over windows and seeds (information in "
        f"nats; accuracies over (step, group); open/o-pers/o-marg at k={h}; the free margin in embedding "
        f"units and the probe margin in map units at h={h}, over the windows moved at h) ---",
        f"  {'arm':<12}{'info':>7}{'teacher':>9}{'persist':>9}{'marginal':>9}{'open':>7}{'o-pers':>8}"
        f"{'o-marg':>8}{'free m':>9}{'probe m':>9}",
    ]
    for label, keys in groups:
        def margin(which):
            means = []
            for k in keys:
                values, changed = contrast_values(records[k], which, h)
                means.append(float(values[changed].mean()) if changed.any() else float("nan"))
            return float(np.nanmean(means)) if np.isfinite(means).any() else float("nan")

        lines.append(
            f"  {label:<12}"
            f"{_num(_mean_over(records, keys, lambda r: r['information'])):>7}"
            f"{_num(_mean_over(records, keys, lambda r: r['accuracy']['teacher'])):>9}"
            f"{_num(_mean_over(records, keys, lambda r: r['accuracy']['persistence'])):>9}"
            f"{_num(_mean_over(records, keys, lambda r: r['accuracy']['marginal'])):>9}"
            f"{_num(_mean_over(records, keys, lambda r: np.asarray(r['open']['accuracy'])[:, h - 1])):>7}"
            f"{_num(_mean_over(records, keys, lambda r: np.asarray(r['open']['persistence'])[:, h - 1])):>8}"
            f"{_num(_mean_over(records, keys, lambda r: np.asarray(r['open']['marginal'])[:, h - 1])):>8}"
            f"{_num(margin('decode')):>9}{_num(margin('decode_probe')):>9}"
        )
    return "\n".join(lines) + "\n"


def _curve_table(records: dict, groups, horizon: int) -> str:
    ks = [k for k in REPORTED_K if k <= horizon]
    lines = [
        "--- carry: open-loop accuracy against k (means over windows and seeds), beside latent persistence "
        "from t0 and the marginal class ---",
        f"  {'arm':<12}{'series':<12}" + "".join(f"{f'k={k}':>8}" for k in ks),
    ]
    for label, keys in groups:
        for series in ("accuracy", "persistence", "marginal"):
            values = [_mean_over(records, keys, lambda r, k=k: np.asarray(r["open"][series])[:, k - 1]) for k in ks]
            lines.append(f"  {label:<12}{series:<12}" + "".join(f"{v:8.3f}" for v in values))
    return "\n".join(lines) + "\n"


def _decode_table(records: dict, groups, h: int, horizon: int) -> str:
    hs = [h] if h == horizon else [h, horizon]
    lines = [
        "--- decode: the paired margin persistence - model over the windows moved at h (free: embedding units, "
        "decides; probe: map units, companion); estimate +- clustered se, z ---",
        f"  {'arm':<12}{'channel':<8}{'h':>4}{'estimate':>10}{'se':>9}{'z':>8}{'windows':>9}",
    ]
    for label, keys in groups:
        for channel, name in (("free", "decode"), ("probe", "decode_probe")):
            for hh in hs:
                cells = [_series(records[k], *contrast_values(records[k], name, hh), channel=name) for k in keys]
                p = pooled(cells)
                windows = int(np.logical_and.reduce([c.changed for c in cells]).sum())
                lines.append(
                    f"  {label:<12}{channel:<8}{hh:>4}{fmt_z(p.estimate, '+.4f'):>10}{fmt_z(p.se, '.4f'):>9}"
                    f"{fmt_z(p.z):>8}{windows:>9}"
                )
    return "\n".join(lines) + "\n"


def _companion_table(records: dict, groups, horizon: int) -> str:
    ks = [k for k in (1, 5, 15, 45) if k <= horizon]
    lines = [
        "--- companions (change no verdict): groups whose mean posterior entropy is below half of ln(classes) "
        "(mean over seeds); the teacher-forced prior's NLL of the posterior mode; the median rendering distance "
        "||e_post(h) - e(h)|| beside the median one-frame jitter ||e(h) - e(h-1)|| ---",
        f"  {'arm':<12}{'groups<lnC/2':>13}{'nll':>8}" + "".join(f"{f'render({k})':>12}{f'jitter({k})':>12}" for k in ks),
    ]
    for label, keys in groups:
        rows = [records[k] for k in keys]
        below = float(np.mean([
            (np.asarray(r["companions"]["entropy"], dtype=np.float64) < 0.5 * np.log(r["latent"]["classes"])).sum()
            for r in rows
        ]))
        nll = _mean_over(records, keys, lambda r: r["companions"]["nll"])
        pairs = "".join(
            f"{_mean_over(records, keys, lambda r, k=k: np.asarray(r['companions']['rendering_median'])[k - 1]):12.3f}"
            f"{_mean_over(records, keys, lambda r, k=k: np.asarray(r['companions']['jitter_median'])[k - 1]):12.3f}"
            for k in ks
        )
        lines.append(f"  {label:<12}{below:>13.1f}{nll:>8.3f}{pairs}")
    return "\n".join(lines) + "\n"


def _control_text(reading: StagesReading, inputs: StagesInputs, records: dict, control_seeds) -> str:
    r = next(records[k] for k in records if k[0] == CONTROL_LABEL)
    head = (
        f"--- the known-answer control: {r['arm']} seeds {sorted(int(s) for s in control_seeds)} at step "
        f"{r['step']} from {r['source']} -- blind by measurement, must read ENCODE FAILS; reported here, "
        "pooled into no arm ---\n"
    )
    return head + format_reading_stages(reading, inputs)


def _per_seed_text(inputs: StagesInputs) -> str:
    lines = ["--- per seed (each seed's own leaves read alone; no replication clause) ---"]
    for arm, a in inputs.arms.items():
        for seed, one in sorted(a.per_seed.items()):
            r = reading_stages(StagesInputs(arms={arm: one}, z_fam=inputs.z_fam, h=inputs.h)).arms[arm]
            lines.append(f"  {arm:<12}s{seed}  {r.status.name.replace('_', ' ')} -- {r.rule}")
    return "\n".join(lines) + "\n"


def write_curves(records: dict, groups, figure: Path, horizon: int, h: int) -> str:
    """Three panels: open-loop accuracy against k with both baselines, the
    sorted per-group posterior entropy, and `information` per record against
    the free-bits line. A missing or broken matplotlib, or an unwritable
    path, costs the FIGURE and nothing else (report_study's guard)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, ValueError, OSError) as error:
        return f"figure NOT written: {error}"
    colours = {"pixel_ae": "tab:orange", "frozen_ssl": "tab:blue", "random_vit": "tab:gray", CONTROL_LABEL: "tab:red"}
    ks = np.arange(1, horizon + 1)
    classes = int(next(iter(records.values()))["latent"]["classes"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    try:
        carry, entropy, info = axes.flat
        for label, keys in groups:
            rows = [records[k] for k in keys]
            colour = colours.get(label, "black")
            for series, style in (("accuracy", "-"), ("persistence", ":"), ("marginal", "--")):
                y = np.mean([np.asarray(r["open"][series], dtype=np.float64).mean(axis=0) for r in rows], axis=0)
                carry.plot(ks, y, style, color=colour, linewidth=1.0,
                           label=f"{label} {series}" if series == "accuracy" else None)
            sorted_entropy = np.sort(np.mean([np.asarray(r["companions"]["entropy"], dtype=np.float64) for r in rows], axis=0))
            entropy.plot(np.arange(sorted_entropy.size), sorted_entropy, marker="o", markersize=3,
                         color=colour, linewidth=0.9, label=label)
            info.scatter([label] * len(rows), [float(np.mean(r["information"])) for r in rows], color=colour, s=18)
        carry.axvline(h, color="black", linestyle=":", linewidth=0.8)
        carry.set_title("open-loop accuracy vs k (solid); persistence (dotted); marginal (dashed)")
        carry.set_xlabel("k")
        entropy.axhline(np.log(classes), color="black", linestyle=":", linewidth=0.8)
        entropy.set_title("mean posterior entropy per group, sorted (nats)")
        entropy.set_xlabel("group (sorted)")
        info.axhline(KL_FREE_BITS, color="black", linestyle=":", linewidth=0.8)
        info.set_title("information KL(post || teacher prior) per record (nats)")
        handles, labels = carry.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), fontsize=8)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=110)
    except (ValueError, OSError) as error:
        return f"figure NOT written: {error}"
    finally:
        plt.close(fig)
    return f"figure={figure}"


def readings_text(records: dict, *, groups, inputs: StagesInputs, reading: StagesReading, control_text: str,
                  horizon: int, clamped: bool, figure_line: str) -> str:
    """Everything `read` prints, in `ladder.txt`'s style; written to
    `stages.txt` byte-identical."""
    h = inputs.h
    note = (f"  NOTE: decision horizon clamped to the run's horizon h={h} "
            f"(pre-registered DECISION_H = {DECISION_H}).\n" if clamped else "")
    return "".join([
        _self_check_table(records),
        _components_table(records, groups, h),
        _curve_table(records, groups, horizon),
        _decode_table(records, groups, h, horizon),
        _companion_table(records, groups, horizon),
        note,
        f"  pooling: z_fam = cluster_threshold({FAMILY}, {_clusters(records)}) = {inputs.z_fam:.2f}; each "
        "contrast is a per-window series pooled over the group's seeds (seeds averaged per window, "
        f"episode-clustered); a stage passes pooled and in >= {SEEDS_REQUIRED} seeds; the decode margin "
        "pools the windows moved at h.\n",
        control_text,
        format_reading_stages(reading, inputs),
        _per_seed_text(inputs),
        f"  {figure_line}\n",
    ])


def write_readings(out_dir: Path, text: str) -> Path:
    path = Path(out_dir) / "stages.txt"
    path.write_text(text)
    return path


def load_records(args, cells, controls) -> tuple[int, dict]:
    """Every requested record, keyed by `(label, seed)`; 11 names the first missing."""
    records: dict[tuple[str, int], dict] = {}
    for kind, pairs in (("cell", cells), ("control", controls)):
        for arm, seed in pairs:
            path = record_path(args.out, kind, arm, seed)
            if not path.exists():
                print(f"NO CELL: {arm} seed {seed} ({kind}): no stages record at {path}; run --phase evaluate first")
                return EXIT_NO_CHECKPOINTS, {}
            record = load_record(path)
            records[(str(record["label"]), int(seed))] = record
    return EXIT_OK, records


def _trustworthy(check) -> bool:
    return bool(check) and (
        check.get("ok") is True
        and check.get("reference_position_max_delta") == 0.0
        and check.get("persistence_position_max_delta") == 0.0
        and bool(check.get("windows_total_match"))
        and bool(check.get("windows_episode_match"))
    )


def read_phase(args, cells, controls) -> int:
    """Load every record (11), refuse one whose self-check is not ok or
    whose identity is unset (30), refuse records at different protocols,
    read the control FIRST (34), then the figure, the text, `stages.txt`."""
    status, records = load_records(args, cells, controls)
    if status != EXIT_OK:
        return status
    for (label, seed), r in records.items():
        if r["kind"] == "cell" and not _trustworthy(r["self_check"]):
            print(
                f"\nSELF-CHECK FAILED for {r['arm']} seed {seed}: the record's self-check is "
                f"{r['self_check']!r}; it is not read against a ruler that reproduced."
            )
            return EXIT_SELF_CHECK_FAILED
        if r["identity"] is not True:
            print(
                f"\nSELF-CHECK FAILED for {r['arm']} seed {seed} ({r['kind']}): the record's step-1 "
                "identity flag is not set; its two priors are not one rollout."
            )
            return EXIT_SELF_CHECK_FAILED
    first = next(iter(records.values()))
    for (label, seed), r in records.items():
        for key in ("context", "horizon", "decision_h"):
            if r[key] != first[key]:
                raise ValueError(
                    f"{label} seed {seed}: {key} {r[key]} is not the first record's {first[key]}; "
                    "records at different protocols are different runs"
                )
        if r["windows"]["episode"] != first["windows"]["episode"]:
            raise ValueError(f"{label} seed {seed}: its windows are not the first record's")
    horizon, h = int(first["horizon"]), int(first["decision_h"])
    _, clamped = decision_horizon(horizon)
    arms, seeds = list(args.arms), [int(s) for s in args.seeds]
    control_seeds = [int(s) for s in args.control_seeds]
    inputs, control_inputs = stages_inputs(records, arms=arms, seeds=seeds, control_seeds=control_seeds, h=h)
    reading, control_reading = reading_stages(inputs), reading_stages(control_inputs)
    control_text = _control_text(control_reading, control_inputs, records, control_seeds)
    control = control_reading.arms[CONTROL_LABEL]
    if control.status is not Status.ENCODE_FAILS:
        print(control_text, end="")
        print(
            f"\nCONTROL MISREAD: the known-blind control reads {control.status.name.replace('_', ' ')} -- "
            f"decided by: {control.rule}. An instrument that does not read ENCODE FAILS on a posterior "
            "equal to its prior is not reading encode; no arm's reading is printed and stages.txt is not "
            "written."
        )
        return EXIT_CONTROL_MISREAD
    groups = _groups(records, arms, seeds, control_seeds)
    figure = args.figure if args.figure is not None else args.out / "stages_curves.png"
    figure_line = write_curves(records, groups, figure, horizon, h)
    text = readings_text(
        records, groups=groups, inputs=inputs, reading=reading, control_text=control_text,
        horizon=horizon, clamped=clamped, figure_line=figure_line,
    )
    print(text, end="")
    write_readings(args.out, text)
    return EXIT_OK
```

- [ ] **Step 5: Replace `main`**

```python
def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _check_usage(parser, args)
    device = get_device(prefer=args.device)
    cells = [(arm, int(seed)) for arm in args.arms for seed in args.seeds]
    controls = [(args.control_arm, int(seed)) for seed in args.control_seeds]
    if args.phase in ("evaluate", "all"):
        buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
        train, val = episode_split(buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
        status = evaluate_phase(args, cells, controls, device, train, val)
        if status != EXIT_OK:
            return status
    if args.phase in ("read", "all"):
        return read_phase(args, cells, controls)
    return EXIT_OK
```

One correction to Task 4's test: `test_the_parser_defaults_and_the_pinned_control_are_the_specs` asserted `args.phase == script.PHASES[-1]`; that now reads `"all"`, as the spec says, with no edit to the test.

- [ ] **Step 6: Run the script tests**

Run: `.venv/bin/python -m pytest tests/eval/test_stage_decomposition_script.py -q`
Expected: 25 passed.

- [ ] **Step 7: Commit**

```bash
git add scripts/stage_decomposition.py tests/eval/test_stage_decomposition_script.py
git commit -m "feat: stage_decomposition.py read -- the five contrasts pooled per arm and for the control, the control read first and refused on anything but ENCODE_FAILS, Reading S, the component, carry, decode and companion tables, stages.txt and the figure

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: The exit-status registry, the whole suite, the smoke run

**Files:**
- Modify: `tests/eval/test_diagnose_dynamics_script.py` (`test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own`, ~line 463–545)
- Run artefacts (gitignored): `runs/m3g_smoke/`

**Interfaces:**
- Consumes: `scripts/stage_decomposition.py` (Tasks 4–5), its `EXIT_*` names.
- Produces: the registry clause for `stage_decomposition`; a green whole suite at the commit the smoke runs from; `runs/m3g_smoke/stages.txt` read for format.

- [ ] **Step 1: Extend the distinctness test**

In `tests/eval/test_diagnose_dynamics_script.py`, inside `test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own`, after the `checkpoint_ladder` clause (the loop ending `assert not clash, f"checkpoint_ladder collides with {other} on {clash}"`), add:

```python
    stages = statuses("stage_decomposition")
    reused_by_stages = {**reused, "EXIT_SELF_CHECK_FAILED": 30}
    shared_with_trust = {
        name: value for name, value in stages.items() if value in set(trust.values()) - {0}
    }
    assert shared_with_trust == reused_by_stages, (
        f"stage_decomposition shares {shared_with_trust} with trust_horizon; only "
        f"{reused_by_stages} is shared on purpose"
    )
    assert len(set(stages.values())) == len(stages), stages
    assert 1 not in stages.values() and 2 not in stages.values()
    assert stages["EXIT_CONTROL_MISREAD"] == 34
    own = set(stages.values()) - set(reused_by_stages.values()) - {0}
    assert own == {34}
    for other in ("run_study", "report_study", "pool_dynamics", "diagnose_dynamics", "split_gap",
                  "checkpoint_ladder"):
        clash = own & set(statuses(other).values())
        assert not clash, f"stage_decomposition collides with {other} on {clash}"
```

and add `"stage_decomposition"` to the `for other in (...)` tuple of the `checkpoint_ladder` clause, so the two new tools are checked against each other from both sides.

- [ ] **Step 2: Run the registry test**

Run: `.venv/bin/python -m pytest tests/eval/test_diagnose_dynamics_script.py -k distinct -q`
Expected: 1 passed.

- [ ] **Step 3: Run the whole suite**

Run: `.venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -4`
Expected: every test passes, 0 skipped with `runs/` reachable, 0 warnings. Record the count in the ledger; it is written into `## Task 7 results` from this output, not predicted here.

- [ ] **Step 4: Commit**

```bash
git add tests/eval/test_diagnose_dynamics_script.py
git commit -m "test: the exit-status registry gains stage_decomposition -- 34 new, 11/12/14/30 trust_horizon's, no clash with any other tool

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 5: The smoke run**

One cell and the two controls, into a directory the real run never reads:

```bash
mkdir -p runs/m3g_smoke && git rev-parse HEAD > runs/m3g_smoke/stages.head && \
PYTHONUNBUFFERED=1 .venv/bin/python scripts/stage_decomposition.py \
  --out runs/m3g_smoke --arms pixel_ae --seeds 0 --phase all > runs/m3g_smoke/stages.log 2>&1; \
echo $? | tee runs/m3g_smoke/stages.exit
```

Expected: exit **0** in about six minutes — the cell self-checks at 0.0, the identity holds on all three passes, `pixel_ae` s0 reads `ENCODE_FAILS` (one seed: nothing can replicate, and the per-seed line carries the seed's own status), and the control reads `ENCODE_FAILS` on its own merits (its encode contrast negative). Read `runs/m3g_smoke/stages.log` and `runs/m3g_smoke/stages.txt` in full.

**If the exit is 34**, the control did not read `ENCODE_FAILS`: STOP. That is a finding about the instrument or about the control's premise (§1's scratch measurement), and either is the user's decision before the real run — report the control table and its sentence verbatim and ask.

- [ ] **Step 6: Read the smoke's `stages.txt` for format**

Check, and fix in the script with a test where wrong: every table's header columns align with its rows (M3f shipped a missing space between two header columns; look for it); the control section precedes Reading S; the per-seed block reads "this seed alone"; the figure line names `runs/m3g_smoke/stages_curves.png` and the file opens; the components table's margins are finite over the moved windows; no `nan` where a number is expected on a cell that self-checked. Commit any fix:

```bash
git add scripts/stage_decomposition.py tests/eval/test_stage_decomposition_script.py
git commit -m "fix: <what the smoke's stages.txt showed>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Then re-run the whole suite if the script changed (Step 3's command) and record the count.

---

### Task 7: The real run and the results

**Files:**
- Run artefacts (gitignored): `runs/m3g_stages/` — nine `stages_<arm>_seed<n>.json`, two `stages_control_frozen_ssl_seed{1,2}.json`, `stages.txt`, `stages_curves.png`, `stages.log`, `stages.head`, `stages.started`, `stages.finished`, `stages.exit`
- Modify: this plan (`## Task 7 results`)

**Interfaces:**
- Consumes: the script at a clean, green HEAD; `runs/m3_study_v2`; `runs/m3f_ladder/step4000`.
- Produces: the run and its reading; the results section.

- [ ] **Step 1: Confirm the state the run is stamped with**

```bash
git status --porcelain && git log --oneline -1
```

Expected: empty status (the tree is clean; `git_sha()` is stamped into every record at write time) and the HEAD the suite was green at in Task 6. Do not commit anything until the run has finished.

- [ ] **Step 2: Launch**

```bash
mkdir -p runs/m3g_stages && git rev-parse HEAD > runs/m3g_stages/stages.head && \
date -u +%Y-%m-%dT%H:%M:%SZ > runs/m3g_stages/stages.started && \
nohup sh -c 'PYTHONUNBUFFERED=1 caffeinate -dimsu .venv/bin/python scripts/stage_decomposition.py --out runs/m3g_stages --phase all > runs/m3g_stages/stages.log 2>&1; echo $? > runs/m3g_stages/stages.exit; date -u +%Y-%m-%dT%H:%M:%SZ > runs/m3g_stages/stages.finished' > /dev/null 2>&1 &
```

Eleven passes at about two minutes each on MPS: under half an hour. Poll `runs/m3g_stages/stages.exit` (absent until the run ends); read `stages.log` as it grows (it is unbuffered).

- [ ] **Step 3: Acceptance**

```bash
cat runs/m3g_stages/stages.exit runs/m3g_stages/stages.head runs/m3g_stages/stages.started runs/m3g_stages/stages.finished
.venv/bin/python - <<'EOF'
import json, glob
from pathlib import Path
head = Path("runs/m3g_stages/stages.head").read_text().strip()
paths = sorted(glob.glob("runs/m3g_stages/stages_*.json"))
records = [json.load(open(p)) for p in paths]
assert len(records) == 11, len(records)
assert {r["git_sha"] for r in records} == {head}, ({r["git_sha"] for r in records}, head)
assert {r["device"] for r in records} == {"mps"}
cells = [r for r in records if r["kind"] == "cell"]; controls = [r for r in records if r["kind"] == "control"]
assert len(cells) == 9 and len(controls) == 2
assert all(r["self_check"]["ok"] and r["self_check"]["reference_position_max_delta"] == 0.0
           and r["self_check"]["persistence_position_max_delta"] == 0.0 for r in cells)
assert all(r["identity"] is True for r in records)
assert {(r["windows"]["total"], r["windows"]["clusters"]) for r in records} == {(229, 24)}
assert {r["step"] for r in cells} == {20000} and {r["step"] for r in controls} == {4000}
assert {(r["arm"], r["seed"]) for r in controls} == {("frozen_ssl", 1), ("frozen_ssl", 2)}
assert {r["decision_h"] for r in records} == {15}
print("OK: eleven records at one git_sha == HEAD", head[:7], "on mps; self-check 0.0 on 9/9; identity 11/11; 229/24 everywhere")
EOF
grep -n "verdict:" runs/m3g_stages/stages.txt
```

Expected: exit `0`; the assertions print `OK`; the control's verdict line reads `ENCODE FAILS`; one verdict line per arm. `stages.txt` is byte-identical to the read phase's stdout in `stages.log`.

- [ ] **Step 4: Write `## Task 7 results` into this plan**

Under a new `## Task 7 results` heading at the end of this plan, numbers copied from `stages.txt` and the records, not rounded:

1. **Provenance.** `git_sha` of the eleven records (= `stages.head`), device, torch version, `stages.started` / `stages.finished` and the wall time, `stages.exit`, the per-cell seconds if the log shows them; `stages.txt` byte-identical to the read's stdout.
2. **Suite.** The whole-suite count at the launched HEAD (Task 6 Step 3, re-run if the script changed after the smoke), 0 skipped, 0 warnings.
3. **Acceptance.** Step 3's `OK` line.
4. **The self-check table** and the identity, 11/11.
5. **The control**, first: its component numbers, its five contrasts with z, its status and sentence. `ENCODE FAILS` is the instrument validating; say what its information and accuracies were beside the arms'.
6. **Reading S**, per arm: the five contrasts (estimate ± se, z, clears), the four stage sentences, the status, the seeds holding; the per-seed statuses.
7. **The components table, the carry k-curve, the decode table, the companions** — copied.
8. **The figure** — path and size.
9. **Read against §4 and §3.4.** One bullet per arm: its status, what §3.4 commits M3h to for that status, and what the companions say beside it (which stage's numbers moved, how far the open-loop accuracy sits above its baselines at k = 1, 5, 15, how many groups are informative, the rendering distance beside the jitter). Then the non-claims restated as measured: no arm ranked, no checkpoint changed, `not shown` kept apart from `failed`, h = 15 in the latent says nothing about h = 45 in position.

Any status on any arm is a result. Three arms may read three statuses; M3h is brainstormed from the table.

- [ ] **Step 5: Commit the results**

```bash
git add docs/superpowers/plans/2026-09-19-mb-fps-m3g-stage-decomposition.md
git commit -m "docs: M3g stage decomposition on runs/m3g_stages -- <the three statuses>; the control ENCODE FAILS; the self-check 0.0 and the identity on every record

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Exit criteria

- [ ] Tasks 1–5 committed, each reviewed; Task 6's registry clause committed; the whole suite green with 0 warnings at the launched HEAD.
- [ ] The smoke run exit 0 with the control reading `ENCODE_FAILS`; its `stages.txt` read for format and any defect fixed before the real run.
- [ ] Eleven records at one `git_sha` == HEAD on `mps`; the trust self-check exactly 0.0 on 9/9; the identity on 11/11; 229 windows / 24 clusters on every record; the control `ENCODE_FAILS`; `stages.exit` 0.
- [ ] `## Task 7 results` written, numbers copied, each arm's status stated against §3.4 and the closing bullets against §4.

## Plan self-review (writing-plans)

- **Spec coverage.** §2.1 cells and controls → Tasks 4 (`load_cell`, `rung_cell`, `prepare_cell`), 7. §2.2 the pass → Task 1. §2.3 statistics → Task 2; the record → Task 4. §2.4 self-checks → Tasks 1 (identity), 4 (14, 30). §3.1 pooling → Task 5 (`pooled`, `arm_inputs`, `stages_inputs`, FAMILY 5). §3.2 rules and statuses → Task 3. §3.3 the control → Task 5 (34) and Task 4 (the usage error on seeds). §3.4 the intervention map → Task 7's results section reads each status against it. §3.5 companions → Tasks 4 (record), 5 (tables, figure). §4 → Task 7. §5 code shape → Tasks 1–5 as named; the exit registry → Task 6. §6 the run → Tasks 6–7. §7 files → the file structure table.
- **Placeholders.** None: every code step carries its code; the results section lists what is copied, not what it will say.
- **Type consistency.** `ArmInputs` fields `encode, predict_persistence, predict_marginal, carry, decode, per_seed` (Task 3) are what Task 5's `arm_inputs` builds; `StratumContrast(estimate, se, z, clusters)` is `split_gap`'s; `contrast_values` names match `CONTRASTS` plus `carry_marginal` / `decode_probe`; `Trajectories`' five new fields (Task 1) are the ones `cell_statistics` (Task 4) reads; `stages_record`'s keys are `RECORD_KEYS` in the Task 4 test and what `contrast_values`, the tables and `load_records` read in Task 5; `LatentIdentityError` is raised in Task 1 and caught in Task 4; `PHASES[-1]` is the parser default in both Task 4 (`evaluate`) and Task 5 (`all`).
