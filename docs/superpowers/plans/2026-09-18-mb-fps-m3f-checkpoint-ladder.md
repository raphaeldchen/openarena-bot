# MB-FPS M3f — Checkpoint Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain the nine M3c cells to 5,000 steps saving a checkpoint every 1,000, evaluate every rung and the 20,000-step reference through the study's own evaluation half and the trust pass, and decide — by rules fixed before the run — whether the model at its one-step-loss minimum rolls out better than the model at step 20,000.

**Architecture:** Three extensions of what exists (`train_world_model` saves rungs; `run_job`'s evaluation half becomes `evaluate_job`; a new `val_objective` measures the training loss on validation episodes), one pure rules module (`mbfps.eval.ladder`: the primary-rung rule, the anchor, Reading T), and one script (`scripts/checkpoint_ladder.py`) with three resumable phases — `train`, `evaluate`, `read` — that reuses `trust_horizon.py`'s `prepare_cell` and `split_gap.py`'s `stratum_summary` and pooling glue by path. Each rung directory is a study directory, so every existing instrument reads it unchanged.

**Tech Stack:** Python 3.12, torch 2.13 (MPS), numpy, pytest; matplotlib for the one figure (guarded). Spec: `docs/superpowers/specs/2026-09-18-mb-fps-m3f-checkpoint-ladder-design.md`.

## Global Constraints

- Pre-registration, fixed before the run and never tuned after it: `RUNGS = (1000, 2000, 3000, 4000, 5000)`, `STEPS = 5000`, `REFERENCE_RUNG = 20000`, `DECISION_H = 15`, `REPORTED_H = (5, 15, 45)`, `FAMILY = 6`, `SEEDS_REQUIRED = 2`, `R2_SENSITIVITY = 0.1`, `CURVE_WINDOW = 100`, `OBJECTIVE_BATCHES = 50`.
- The primary rung per cell is `primary_rung(min_step, rungs)`: the rung nearest the reference record's smoothed `embedding` minimum (`learning_curve_summary(history, window=100)["terms"]["embedding"]["smoothed_min_step"]`), ties to the earlier rung. On the shipped records it yields `frozen_ssl` 2000 / 4000 / 5000, `random_vit` 2000 / 5000 / 3000, `pixel_ae` 5000 / 5000 / 5000 (spec §2.2).
- Reading T (spec §3.3), in this precedence: `UNRESOLVED_PROBE` (free and probe clear with opposite signs) → `EARLIER_BETTER` (free clears **positive** pooled and in ≥ `SEEDS_REQUIRED` seeds) → `EARLIER_WORSE` (free clears **negative** pooled and in ≥ `SEEDS_REQUIRED` seeds) → `NO_DIFFERENCE`. A single-seed leaf (`per_seed=None`) has a vacuous replication clause and reads "this seed alone". `clears` is strict and never on NaN or ±inf.
- Pooling (spec §3.1): series are `split_gap.py`'s survival indicator `1[h× > h]` under `changed = isfinite(h×)` and margin `Δ[:, h−1]` under `moved[:, h−1] & isfinite`; the contrast is `pooling.paired_contrast(treatment, control)` with treatment cells labelled `<arm>@primary` and control cells `<arm>@20000`; `z_fam = pooling.cluster_threshold(FAMILY, clusters)` with `clusters` the val stratum's cluster count (24 on the shipped split). The probe channel pools cells measurable at **both** rungs of the pair.
- Rung 20000 is the reference study directory (`--reference`, default `runs/m3_study_v2`), read through `prepare_cell` and self-checked bitwise against its diagnostic (exit 30) **before any rung of that cell is evaluated**; `read` refuses a ladder record whose recorded self-check is not `ok`.
- Exit statuses: 0 / 11 / 12 / 14 / 30 keep `trust_horizon.py`'s meanings; **32 `EXIT_ANCHOR_MISMATCH`** and **33 `EXIT_RUNG_MISLABELLED`** are new and in no other tool's range; argparse's 2 and a traceback's 1 are never reused. `test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own` in `tests/eval/test_diagnose_dynamics_script.py` gains `checkpoint_ladder`.
- The anchor policy (`--anchor hard|report`) is pinned by the smoke run's measurement and written into spec §2.4 as one marked sentence before the real run (Task 7).
- Scripts load siblings by path (`_sibling`); `checkpoint_ladder.py → trust_horizon.py, split_gap.py`.
- Every rung directory has the study's layout: `step{N}/world_model_<arm>_seed<n>.pt` (payload `{"arm", "seed", "step", "state_dict"}`) and `step{N}/result_<arm>_seed<n>.json` written by `evaluate_job`.
- `train_world_model(checkpoint_steps=())` leaves the final checkpoint's payload byte-identical (`{"arm", "seed", "state_dict"}`, no `step` key) and adds only `history["checkpoint_seconds"]`.
- Results live in this plan's `## Task 7 results`; the M3c, M3d and M3e plans are closed and never amended. Never `rm` anything under `runs/`. Never `git stash`.
- Commit messages: a lowercase type prefix and a descriptive subject; every commit ends with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Python for this tree: `.venv/bin/python`; run tests as `.venv/bin/python -m pytest ...`.

---

## File map

| file | responsibility |
|---|---|
| `src/mbfps/training/world_model.py` | `_save_checkpoint`; `train_world_model(..., checkpoint_steps)` saves rungs and records `checkpoint_seconds`; `history_at(history, step)` |
| `src/mbfps/eval/study.py` | `evaluate_job` (the evaluation half of `run_job`, moved); `run_job` = train + evaluate |
| `src/mbfps/eval/objective.py` | new: `val_objective` |
| `src/mbfps/eval/ladder.py` | new: constants, `primary_rung`, `anchor_delta`, Reading T |
| `src/mbfps/eval/split_gap.py`, `scripts/split_gap.py` | Task 4b: `survival_indicator`, `margin_at`, `cell_series`, `decision_horizon` extracted; the script's wrappers over them |
| `scripts/checkpoint_ladder.py` | new: `train` / `evaluate` / `read` phases, records, pooling glue, tables, figure, `ladder.txt` |
| `tests/training/test_world_model.py` | rung and `history_at` pins |
| `tests/eval/test_run_study.py` | `evaluate_job` pin |
| `tests/eval/test_objective.py`, `tests/eval/test_ladder.py`, `tests/eval/test_checkpoint_ladder_script.py` | new |
| `tests/eval/test_diagnose_dynamics_script.py` | distinctness test gains `checkpoint_ladder` |

---

### Task 1: `train_world_model` saves a ladder of rungs, and `history_at` slices a history to one

**Files:**
- Modify: `src/mbfps/training/world_model.py:100-181` (`train_world_model`), plus a `_save_checkpoint` helper above it and `history_at` below it
- Test: `tests/training/test_world_model.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `train_world_model(cfg, buffer, out_dir, log_every=100, checkpoint_steps: tuple[int, ...] = ()) -> dict` whose history gains `"checkpoint_seconds": dict[int, float]`; rung files at `out_dir / f"step{N}" / f"world_model_{arm}_seed{seed}.pt"` with payload `{"arm", "seed", "step", "state_dict"}`; `history_at(history: dict, step: int) -> dict` (Task 2's `evaluate_job` and Task 5's script consume both).

- [ ] **Step 1: Write the failing tests**

Append to `tests/training/test_world_model.py` (after `test_history_records_every_step`, before the "Behavioural guards" banner):

```python
# ---------------------------------------------------------------------------
# M3f: a ladder of rungs during training, and a history sliced to one.
# ---------------------------------------------------------------------------


def test_checkpoint_steps_default_leaves_the_final_save_and_the_history_as_they_were(buffer, tmp_path):
    """The M3c study and every reader of its checkpoints must not notice this
    change: no rung directories, the same three payload keys, and the history
    gains only an empty `checkpoint_seconds`."""
    out = tmp_path / "ckpt"
    history = train_world_model(tiny(seed=3), buffer, out_dir=out)
    assert sorted(p.name for p in out.iterdir()) == ["world_model_cnn_seed3.pt"]
    payload = torch.load(out / "world_model_cnn_seed3.pt", weights_only=True)
    assert set(payload) == {"arm", "seed", "state_dict"}
    assert history["checkpoint_seconds"] == {}


def test_checkpoint_steps_write_labelled_rungs_in_the_studys_layout(buffer, tmp_path):
    """Each rung is a study directory: the final checkpoint's own file name
    under `step{N}/`, labelled with the step it was saved at, and the elapsed
    seconds recorded so a rung's record can carry an honest `seconds`."""
    out = tmp_path / "ckpt"
    history = train_world_model(
        tiny(steps=4, seed=3), buffer, out_dir=out, checkpoint_steps=(2, 4)
    )
    for step in (2, 4):
        payload = torch.load(out / f"step{step}" / "world_model_cnn_seed3.pt", weights_only=True)
        assert (payload["arm"], payload["seed"], payload["step"]) == ("cnn", 3, step)
        assert any(k.startswith("rssm.") for k in payload["state_dict"])
    assert sorted(history["checkpoint_seconds"]) == [2, 4]
    assert 0.0 < history["checkpoint_seconds"][2] <= history["checkpoint_seconds"][4] <= history["seconds"]
    # The final save is still written beside the rungs, unchanged.
    assert set(torch.load(out / "world_model_cnn_seed3.pt", weights_only=True)) == {"arm", "seed", "state_dict"}


def test_the_last_rung_holds_the_same_weights_as_the_final_checkpoint(buffer, tmp_path):
    out = tmp_path / "ckpt"
    train_world_model(tiny(steps=4, seed=3), buffer, out_dir=out, checkpoint_steps=(4,))
    final = torch.load(out / "world_model_cnn_seed3.pt", weights_only=True)["state_dict"]
    rung = torch.load(out / "step4" / "world_model_cnn_seed3.pt", weights_only=True)["state_dict"]
    assert final.keys() == rung.keys()
    assert all(torch.equal(final[k], rung[k]) for k in final)


def test_a_rung_is_a_state_the_longer_run_passed_through(buffer, tmp_path):
    """THE ANCHOR'S PREMISE (spec 2.4): the first N steps of a run do not depend
    on how many steps follow -- the loader's draws and the optimiser's updates
    are a function of the seed alone. Asserted on the loss curve on CPU, where
    a test can assert it; the smoke run measures it on MPS."""
    longer = train_world_model(tiny(steps=4, seed=3), buffer, out_dir=None)
    shorter = train_world_model(tiny(steps=2, seed=3), buffer, out_dir=None)
    assert shorter["loss"] == longer["loss"][:2]


def test_a_rung_outside_the_run_or_without_an_out_dir_is_refused_before_training(buffer, tmp_path):
    out = tmp_path / "ckpt"
    with pytest.raises(ValueError, match="outside 1..3"):
        train_world_model(tiny(steps=3), buffer, out_dir=out, checkpoint_steps=(5,))
    with pytest.raises(ValueError, match="outside 1..3"):
        train_world_model(tiny(steps=3), buffer, out_dir=out, checkpoint_steps=(0,))
    with pytest.raises(ValueError, match="out_dir"):
        train_world_model(tiny(steps=3), buffer, out_dir=None, checkpoint_steps=(2,))
    assert not out.exists(), "a refused run must write nothing"


def _four_step_history() -> dict:
    return {
        "arm": "cnn", "steps": 4,
        "loss": [4.0, 3.0, 2.0, 1.0],
        "parts": [{"embedding": 1.0, "kl_dyn": v, "kl_rep": v} for v in (0.1, 0.3, 0.3, 0.3)],
        "seconds": 8.0, "kl_dyn_max": 0.3, "kl_rate_above_free_bits": 0.75,
        "checkpoint_seconds": {2: 3.0, 4: 8.0},
    }


def test_history_at_cuts_the_prefix_and_recomputes_what_a_shorter_run_would_have_recorded():
    """`kl_dyn` 0.1 / 0.3 / 0.3 / 0.3 against KL_FREE_BITS = 0.20: the rate over
    the first two steps is 0.5, over all four 0.75 -- the prefix's own number,
    not the full run's copied down."""
    assert KL_FREE_BITS == 0.20
    at2 = history_at(_four_step_history(), 2)
    assert at2["steps"] == 2
    assert at2["loss"] == [4.0, 3.0]
    assert [p["kl_dyn"] for p in at2["parts"]] == [0.1, 0.3]
    assert at2["seconds"] == 3.0
    assert at2["kl_dyn_max"] == 0.3
    assert at2["kl_rate_above_free_bits"] == 0.5
    assert at2["checkpoint_seconds"] == {2: 3.0}
    assert at2["arm"] == "cnn"
    at4 = history_at(_four_step_history(), 4)
    assert at4["kl_rate_above_free_bits"] == 0.75 and at4["seconds"] == 8.0
    assert at4["checkpoint_seconds"] == {2: 3.0, 4: 8.0}


def test_history_at_accepts_the_string_keys_a_json_round_trip_leaves():
    history = _four_step_history()
    history["checkpoint_seconds"] = {"2": 3.0, "4": 8.0}
    assert history_at(history, 2)["seconds"] == 3.0


def test_history_at_refuses_a_step_outside_the_history_or_one_no_rung_was_saved_at():
    with pytest.raises(ValueError, match="outside"):
        history_at(_four_step_history(), 5)
    with pytest.raises(ValueError, match="outside"):
        history_at(_four_step_history(), 0)
    with pytest.raises(ValueError, match="checkpoint_seconds"):
        history_at(_four_step_history(), 3)
```

And extend the module's imports at the top of the test file:

```python
from mbfps.training.world_model import WorldModel, history_at, train_world_model
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/training/test_world_model.py -q -k "checkpoint_steps or rung or history_at"`
Expected: FAIL — `ImportError: cannot import name 'history_at'` (the whole module fails to import until Step 3).

- [ ] **Step 3: Implement**

In `src/mbfps/training/world_model.py`, add above `train_world_model`:

```python
def _save_checkpoint(model: nn.Module, cfg: Config, directory: Path, step: int | None = None) -> Path:
    """The checkpoint file every reader of a study directory opens:
    `world_model_{arm}_seed{seed}.pt` under `directory`, with the arm and the
    seed in the payload so a mislabelled file is caught at load. A rung adds
    its `step`; the final save carries exactly the three keys it always has."""
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "arm": cfg.arm, "seed": cfg.train.seed, "state_dict": model.state_dict(),
    }
    if step is not None:
        payload["step"] = int(step)
    path = directory / f"world_model_{cfg.arm}_seed{cfg.train.seed}.pt"
    torch.save(payload, path)
    return path
```

Replace `train_world_model` with:

```python
def train_world_model(
    cfg: Config,
    buffer: ReplayBuffer,
    out_dir: Path | None,
    log_every: int = 100,
    checkpoint_steps: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Train one arm's world model and return its history.

    `checkpoint_steps` (M3f): 1-based steps at which the weights are ALSO
    saved, each into `out_dir / f"step{N}"` under the final checkpoint's own
    file name and payload plus a `"step"` key -- a rung directory in the
    study's layout, so every reader of a study directory reads a rung
    unchanged. The elapsed seconds at each rung go to
    `history["checkpoint_seconds"]`. Empty (the default) leaves the final save
    as it was and adds only that empty dict to the history. A rung outside
    `1..cfg.train.steps` is refused before anything is built: a rung the run
    never reaches would be a checkpoint that is never written.
    """
    from mbfps.utils.device import to_device

    rungs = tuple(sorted({int(s) for s in checkpoint_steps}))
    outside = [s for s in rungs if s < 1 or s > cfg.train.steps]
    if outside:
        raise ValueError(
            f"checkpoint_steps {outside} lie outside 1..{cfg.train.steps}: a rung the run "
            "never reaches would be a checkpoint that is never written"
        )
    if rungs and out_dir is None:
        raise ValueError("checkpoint_steps need an out_dir to write the rungs into")

    seed_everything(cfg.train.seed)
    device = get_device(prefer=cfg.train.device)
    model = WorldModel(cfg).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    backbone = encoder_backbone(cfg.encoder)
    # Train on the TRAINING episodes only. The split seed is fixed at 0 and is
    # deliberately NOT cfg.train.seed: every arm and every seed must be held out
    # on the same episodes, or the comparison measures which episodes each run
    # happened to get rather than which representation is better.
    train_paths, _ = episode_split(
        buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=0
    )
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
    checkpoint_seconds: dict[int, float] = {}
    start = time.perf_counter()
    for step in range(cfg.train.steps):
        loss, parts = model(to_device(loader.sample(), device))
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimiser.step()
        losses.append(float(loss.detach()))
        history_parts.append(parts)
        if (step + 1) in rungs:
            checkpoint_seconds[step + 1] = time.perf_counter() - start
            _save_checkpoint(model, cfg, Path(out_dir) / f"step{step + 1}", step=step + 1)
        if log_every and (step + 1) % log_every == 0:
            print(f"[{cfg.arm}] step {step + 1}/{cfg.train.steps} loss={losses[-1]:.5f}")

    kl_values = [p["kl_dyn"] for p in history_parts]
    kl_dyn_max = max(kl_values, default=0.0)
    kl_rate = _kl_rate(kl_values)
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
        "kl_rate_above_free_bits": kl_rate,
        # Elapsed seconds at each rung of `checkpoint_steps` (M3f).
        "checkpoint_seconds": checkpoint_seconds,
    }
    if kl_rate < 0.5:
        print(
            f"[{cfg.arm}] WARNING: kl_dyn exceeded the {KL_FREE_BITS} nat floor on only "
            f"{100 * kl_rate:.1f}% of steps (peak {kl_dyn_max:.4f}). The dynamics prior "
            "is largely untrained, so imagine() runs close to its initialisation. Arms "
            "whose rates differ cannot be compared: the comparison would measure how "
            "much each prior trained, not the representations."
        )
    if out_dir is not None:
        _save_checkpoint(model, cfg, Path(out_dir))
    return history


def history_at(history: dict[str, Any], step: int) -> dict[str, Any]:
    """The training history as it stood at 1-based `step` (M3f): `loss` and
    `parts` cut to the prefix, `steps` = `step`, `seconds` the elapsed time
    the trainer recorded at that rung (`checkpoint_seconds[step]`, or the
    run's total when `step` is its last), the two KL summaries recomputed over
    the prefix, and `checkpoint_seconds` cut to the rungs at or before it --
    the numbers a run stopped at that step would have recorded, so a rung's
    study record carries the same fields as the study's with the same
    meaning. `checkpoint_seconds` keys may be int or str (a JSON round trip
    makes them str)."""
    step = int(step)
    n = len(history["loss"])
    if not 1 <= step <= n:
        raise ValueError(f"step {step} is outside the history's 1..{n}")
    recorded = {int(k): float(v) for k, v in dict(history.get("checkpoint_seconds", {})).items()}
    if step in recorded:
        seconds = recorded[step]
    elif step == n:
        seconds = float(history["seconds"])
    else:
        raise ValueError(
            f"no checkpoint_seconds entry for step {step}; the trainer records one per rung "
            f"and this history has {sorted(recorded)}"
        )
    parts = [dict(p) for p in list(history["parts"])[:step]]
    kl_values = [float(p["kl_dyn"]) for p in parts]
    return {
        "arm": history["arm"],
        "steps": step,
        "loss": [float(v) for v in list(history["loss"])[:step]],
        "parts": parts,
        "seconds": seconds,
        "kl_dyn_max": max(kl_values, default=0.0),
        "kl_rate_above_free_bits": _kl_rate(kl_values),
        "checkpoint_seconds": {k: v for k, v in recorded.items() if k <= step},
    }
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/training/test_world_model.py -q -k "checkpoint_steps or rung or history_at"`
Expected: 8 passed.

- [ ] **Step 5: Run the whole training test file and the study tests that train**

Run: `.venv/bin/python -m pytest tests/training/test_world_model.py tests/eval/test_run_study.py -q`
Expected: all pass (the existing `test_checkpoint_is_written_with_its_arm_and_seed` and `test_history_records_every_step` unchanged; `run_job`'s record unchanged because `history_record` copies only `loss` and `parts`).

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/training/world_model.py tests/training/test_world_model.py
git commit -m "feat: train_world_model saves a ladder of labelled rungs in the study's layout, and history_at slices a history to one

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `run_job`'s evaluation half becomes `evaluate_job`, shared

**Files:**
- Modify: `src/mbfps/eval/study.py:430-590` (`run_job`) — the block from the checkpoint load to `write_record` moves into `evaluate_job`, unchanged
- Test: `tests/eval/test_run_study.py`

**Interfaces:**
- Consumes: `train_world_model`, `history_at` (Task 1).
- Produces: `evaluate_job(job: StudyJob, buffer: ReplayBuffer, out_dir: Path, *, history: dict, steps: int = 20_000, seq_len: int = 64, context: int = 5, horizon: int = 45, device: str = "mps", started: float | None = None) -> dict` — loads `out_dir / world_model_{arm}_seed{seed}.pt` (arm/seed check), fits the probes, scores the rollout and the filtering, writes `result_{arm}_seed{seed}.json` into `out_dir`, returns the LIVE record. `run_job` is unchanged in behaviour and record. Task 5 calls `evaluate_job` on every rung directory.

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_run_study.py`, at the end of the "a real record" section (after `test_a_real_record_from_run_job_is_recognised_as_complete`):

```python
def test_run_job_is_train_world_model_followed_by_evaluate_job(tmp_path, small_buffer):
    """M3f evaluates a rung by calling the evaluation half alone on a rung
    directory. This pins that half as `evaluate_job`: the same cell trained
    and evaluated through `run_job` and through the two halves writes the same
    record, field for field, in the sanitised projection -- `seconds` and
    `steps_per_second` aside, which are wall-clock."""
    job = StudyJob("random_vit", INCOMPLETE_JOB.seed)
    whole = study.run_job(job, small_buffer, tmp_path / "whole", **RUN_KW)

    cfg = get_config(job.arm, steps=RUN_KW["steps"], seq_len=RUN_KW["seq_len"],
                     seed=job.seed, device=RUN_KW["device"])
    history = train_world_model(cfg, small_buffer, out_dir=tmp_path / "halves")
    halves = study.evaluate_job(job, small_buffer, tmp_path / "halves", history=history, **RUN_KW)

    timing = {"seconds", "steps_per_second"}
    assert {k: v for k, v in study.to_json_record(whole).items() if k not in timing} == \
        {k: v for k, v in study.to_json_record(halves).items() if k not in timing}
    assert job_record_path(tmp_path / "halves", job).is_file()
    assert halves["steps"] == RUN_KW["steps"] and halves["history"]["loss"] == history["loss"]
    assert halves["seconds"] > 0.0
```

Add to the file's imports:

```python
from mbfps.training.world_model import train_world_model
from mbfps.utils.config import ARMS, Config, get_config
```

(`Config` and `ARMS` are already imported on that line; extend it.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/eval/test_run_study.py -q -k evaluate_job`
Expected: FAIL — `AttributeError: module 'mbfps.eval.study' has no attribute 'evaluate_job'`.

- [ ] **Step 3: Implement**

In `src/mbfps/eval/study.py`, replace `run_job` (its whole definition, docstring included) with the two functions below. The docstring of `run_job` is kept word for word; the moved block is `evaluate_job`'s body and is unchanged except for the four lines marked `# M3f`.

```python
def run_job(
    job: StudyJob,
    buffer: ReplayBuffer,
    out_dir: Path,
    steps: int = 20_000,
    seq_len: int = 64,
    context: int = 5,
    horizon: int = 45,
    device: str = "mps",
) -> dict:
    """Train, evaluate, and write one result record. Returns the LIVE record.

    (The existing docstring body -- from "THE RETURNED DICT IS NOT THE FILE'S
    CONTENT." through the FOUR PROVENANCE FIELDS list -- stays here word for
    word; it is in the file today. Only the paragraph below is added.)

    M3f: the evaluation half is `evaluate_job`, so a rung's checkpoint can be
    scored into a rung directory with no retraining. This function is
    `train_world_model` followed by it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = get_config(job.arm, steps=steps, seq_len=seq_len, seed=job.seed, device=device)

    started = time.perf_counter()
    history = train_world_model(cfg, buffer, out_dir=out_dir)
    return evaluate_job(
        job, buffer, out_dir, history=history, steps=steps, seq_len=seq_len,
        context=context, horizon=horizon, device=device, started=started,
    )


def evaluate_job(
    job: StudyJob,
    buffer: ReplayBuffer,
    out_dir: Path,
    *,
    history: dict,
    steps: int = 20_000,
    seq_len: int = 64,
    context: int = 5,
    horizon: int = 45,
    device: str = "mps",
    started: float | None = None,
) -> dict:
    """The evaluation half of `run_job` (M3f): load the checkpoint `out_dir`
    holds for `job`, fit the probes, score the rollout and the filtering, and
    write `result_<arm>_seed<n>.json` into `out_dir`. Returns the LIVE record
    (see `run_job` for what that means).

    `history` is the training history the record carries -- the run's own from
    `run_job`, or a rung's prefix (`training.world_model.history_at`) when the
    checkpoint is a rung's; `steps` should agree with `history["steps"]`.
    `started` is the `perf_counter` the record's `seconds` is measured from;
    None measures the evaluation alone, which is what a rung's record means
    by it (its training time is `history["seconds"]`, as `steps_per_second`
    reads it).
    """
    out_dir = Path(out_dir)                                       # M3f
    started = time.perf_counter() if started is None else started  # M3f
    cfg = get_config(job.arm, steps=steps, seq_len=seq_len, seed=job.seed, device=device)  # M3f
    torch_device = get_device(prefer=device)                      # M3f
    train_paths, val_paths = episode_split(
        buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED
    )
    backbone = encoder_backbone(cfg.encoder)

    model = WorldModel(cfg).to(torch_device)
    checkpoint = torch.load(
        out_dir / f"world_model_{job.arm}_seed{job.seed}.pt",
        map_location=torch_device,
        weights_only=True,
    )
    if checkpoint.get("arm") != job.arm or checkpoint.get("seed") != job.seed:
        raise ValueError(
            f"checkpoint in {out_dir} is arm={checkpoint.get('arm')!r} "
            f"seed={checkpoint.get('seed')!r}, not this job's "
            f"arm={job.arm!r} seed={job.seed!r}"
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    # Counted on the EVALUATED model's encoder, after the load: the same
    # object every number below is measured on.
    encoder_params = _trainable_parameters(model.encoder)

    latent_probe, embedding_probe = fit_probes(
        model, train_paths, backbone, torch_device,
        context=context, horizon=horizon, seed=job.seed,
    )
    reward = reward_accuracy(
        model, val_paths, backbone, torch_device, seed=job.seed
    )
    result = evaluate_rollout(
        model, val_paths, embedding_probe,
        context=context, horizon=horizon, seed=job.seed,
        device=torch_device, feature_backbone=backbone,
    )
    # The criterion_4 / gain block and the `record = {...}` literal follow
    # here EXACTLY as they are in today's run_job (moved, not retyped), through
    # `assert set(METRICS) <= set(record)`; the only difference is that
    # `steps`, `seq_len`, `context`, `horizon`, `device`, `history` and
    # `started` are now this function's parameters.
    write_record(job_record_path(out_dir, job), record)
    return record
```

The implementer moves the text rather than retyping it: cut from `torch_device = get_device(prefer=device)` (the first line after `cfg = ...` in today's `run_job`) through `return record`, paste it as `evaluate_job`'s body after the `# M3f` lines, and delete the `started = time.perf_counter()` / `history = train_world_model(...)` pair from what was pasted (they stay in `run_job`). Nothing else in the block changes.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/eval/test_run_study.py -q -k "evaluate_job or real_record"`
Expected: 4 passed (the new test and the three-arm drift guard).

- [ ] **Step 5: Run the study and script test files that call `run_job`**

Run: `.venv/bin/python -m pytest tests/eval/test_run_study.py tests/eval/test_trust_horizon_script.py tests/eval/test_split_gap_script.py -q`
Expected: all pass, no warnings.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/study.py tests/eval/test_run_study.py
git commit -m "refactor: run_job's evaluation half is evaluate_job, so a rung directory can be scored without retraining; run_job's record unchanged and pinned

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `val_objective` — the training loss on validation episodes

**Files:**
- Create: `src/mbfps/eval/objective.py`
- Test: `tests/eval/test_objective.py`

**Interfaces:**
- Consumes: `WorldModel.forward` (returns `(loss, parts)` with parts `embedding`, `reward`, `continue`, `kl_dyn`, `kl_rep`); `SequenceLoader(buffer, batch_size, seq_len, seed, load_obs, load_features, feature_backbone, paths)`; `utils.device.to_device`.
- Produces: `val_objective(model, buffer, paths, cfg, device, *, batches: int = 50, seed: int = 0) -> dict[str, float]` with keys `loss`, `embedding`, `reward`, `continue`, `kl_dyn`, `kl_rep` — each the mean over `batches` draws. Task 5 calls it on every rung.

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_objective.py`:

```python
"""`mbfps.eval.objective.val_objective`: the training objective on the
validation episodes -- the one number M3e could not read.

The model is a freshly initialised `random_vit` world model on the six-episode
fixture at `batch_size=2, seq_len=4`; nothing here trains. What is pinned is
the arithmetic (the mean of the model's own parts over seeded draws from the
paths it is given), the seeding, and that the call leaves the model as it
found it.
"""

import pytest
import torch

import mbfps.eval.objective as objective
from mbfps.data.loader import SequenceLoader
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.objective import val_objective
from mbfps.eval.study import SPLIT_SEED
from mbfps.models.encoders import encoder_backbone
from mbfps.training.world_model import WorldModel
from mbfps.utils.config import get_config
from mbfps.utils.device import to_device
from mbfps.utils.seeding import seed_everything

PARTS = ("embedding", "reward", "continue", "kl_dyn", "kl_rep")
DEVICE = torch.device("cpu")


@pytest.fixture
def fresh(small_buffer):
    cfg = get_config("random_vit", steps=1, batch_size=2, seq_len=4, device="cpu")
    seed_everything(0)
    model = WorldModel(cfg).to(DEVICE)
    _, val = episode_split(small_buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
    return model, cfg, val


def test_one_batch_is_the_models_own_loss_and_parts_on_that_batch(small_buffer, fresh):
    """`batches=1`, seed 7: the same draw the loader makes at seed 7 and the
    same categorical sample `torch.manual_seed(7)` pins, in eval mode."""
    model, cfg, val = fresh
    out = val_objective(model, small_buffer, val, cfg, DEVICE, batches=1, seed=7)

    loader = SequenceLoader(
        small_buffer, batch_size=cfg.train.batch_size, seq_len=cfg.train.seq_len, seed=7,
        load_obs=model.input_kind == "obs", load_features=model.input_kind == "features",
        feature_backbone=encoder_backbone(cfg.encoder) or "dinov2", paths=list(val),
    )
    model.eval()
    torch.manual_seed(7)
    with torch.no_grad():
        loss, parts = model(to_device(loader.sample(), DEVICE))
    assert set(out) == {"loss", *PARTS}
    assert out["loss"] == pytest.approx(float(loss))
    for name in PARTS:
        assert out[name] == pytest.approx(parts[name])


def test_the_mean_over_batches_is_the_mean_of_single_batch_calls_in_the_same_draw_order(small_buffer, fresh):
    """Three batches at seed 3 average the same three draws a three-batch call
    makes -- which a loader re-seeded per call could not reproduce, so this
    also pins that the loader is built ONCE per call."""
    model, cfg, val = fresh
    three = val_objective(model, small_buffer, val, cfg, DEVICE, batches=3, seed=3)
    loader = SequenceLoader(
        small_buffer, batch_size=cfg.train.batch_size, seq_len=cfg.train.seq_len, seed=3,
        load_obs=model.input_kind == "obs", load_features=model.input_kind == "features",
        feature_backbone=encoder_backbone(cfg.encoder) or "dinov2", paths=list(val),
    )
    model.eval()
    torch.manual_seed(3)
    losses = []
    with torch.no_grad():
        for _ in range(3):
            loss, _ = model(to_device(loader.sample(), DEVICE))
            losses.append(float(loss))
    assert three["loss"] == pytest.approx(sum(losses) / 3)


def test_the_seed_pins_the_draws_and_the_sampler(small_buffer, fresh):
    model, cfg, val = fresh
    a = val_objective(model, small_buffer, val, cfg, DEVICE, batches=2, seed=1)
    b = val_objective(model, small_buffer, val, cfg, DEVICE, batches=2, seed=1)
    c = val_objective(model, small_buffer, val, cfg, DEVICE, batches=2, seed=2)
    assert a == b
    assert a != c


def test_it_reads_only_the_paths_it_is_given(small_buffer, fresh, monkeypatch):
    model, cfg, val = fresh
    seen = {}
    real = objective.SequenceLoader

    def spy(buffer, **kw):
        seen["paths"] = kw["paths"]
        seen["batch_size"], seen["seq_len"] = kw["batch_size"], kw["seq_len"]
        return real(buffer, **kw)

    monkeypatch.setattr(objective, "SequenceLoader", spy)
    val_objective(model, small_buffer, val, cfg, DEVICE, batches=1, seed=0)
    assert seen["paths"] == list(val)
    assert (seen["batch_size"], seen["seq_len"]) == (cfg.train.batch_size, cfg.train.seq_len)


def test_it_leaves_the_model_in_the_mode_it_found_it_and_builds_no_graph(small_buffer, fresh):
    model, cfg, val = fresh
    model.train()
    val_objective(model, small_buffer, val, cfg, DEVICE, batches=1, seed=0)
    assert model.training
    model.eval()
    val_objective(model, small_buffer, val, cfg, DEVICE, batches=1, seed=0)
    assert not model.training
    assert all(p.grad is None for p in model.parameters())


def test_two_different_models_read_differently(small_buffer, fresh):
    model, cfg, val = fresh
    seed_everything(1)
    other = WorldModel(cfg).to(DEVICE)
    a = val_objective(model, small_buffer, val, cfg, DEVICE, batches=1, seed=0)
    b = val_objective(other, small_buffer, val, cfg, DEVICE, batches=1, seed=0)
    assert a["loss"] != b["loss"]


def test_zero_batches_is_refused(small_buffer, fresh):
    model, cfg, val = fresh
    with pytest.raises(ValueError, match="batches"):
        val_objective(model, small_buffer, val, cfg, DEVICE, batches=0, seed=0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_objective.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mbfps.eval.objective'`.

- [ ] **Step 3: Implement**

Create `src/mbfps/eval/objective.py`:

```python
"""The training objective on held-out episodes -- milestone M3f.

`train_world_model` records the loss on every TRAINING batch and nothing on
validation, so every "minimum" M3e read off a learning curve is a training
minimum. This measures the same objective -- `WorldModel.forward`'s loss and
its five parts -- on sequences drawn from the validation episodes, so a
training minimum can be set beside a validation one. Descriptive: nothing is
decided on it (spec 2.3).
"""

import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.loader import SequenceLoader
from mbfps.models.encoders import encoder_backbone
from mbfps.utils.config import Config
from mbfps.utils.device import to_device


def val_objective(
    model, buffer: ReplayBuffer, paths, cfg: Config, device, *, batches: int = 50, seed: int = 0
) -> dict[str, float]:
    """Mean `loss` and mean of each part over `batches` draws of the training
    loader's own shape (`cfg.train.batch_size` x `cfg.train.seq_len`) from
    `paths`, with the model in eval mode, no gradient, and the categorical
    sampler pinned by `torch.manual_seed(seed)` as `evaluate_rollout` pins it.
    The loader is built once, so `seed` fixes the whole sequence of draws.
    Returns `{"loss", "embedding", "reward", "continue", "kl_dyn", "kl_rep"}`.
    The model is returned to the mode it was found in."""
    batches = int(batches)
    if batches < 1:
        raise ValueError(f"batches must be >= 1, got {batches}")
    loader = SequenceLoader(
        buffer,
        batch_size=cfg.train.batch_size,
        seq_len=cfg.train.seq_len,
        seed=seed,
        load_obs=model.input_kind == "obs",
        load_features=model.input_kind == "features",
        feature_backbone=encoder_backbone(cfg.encoder) or "dinov2",
        paths=list(paths),
    )
    was_training = model.training
    model.eval()
    torch.manual_seed(seed)
    totals: dict[str, float] = {}
    try:
        with torch.no_grad():
            for _ in range(batches):
                loss, parts = model(to_device(loader.sample(), device))
                totals["loss"] = totals.get("loss", 0.0) + float(loss)
                for name, value in parts.items():
                    totals[name] = totals.get(name, 0.0) + float(value)
    finally:
        model.train(was_training)
    return {name: value / batches for name, value in totals.items()}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_objective.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/eval/objective.py tests/eval/test_objective.py
git commit -m "feat: val_objective -- the training loss and its parts on the validation episodes, seeded, in eval mode, the model left as found

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `mbfps.eval.ladder` — the primary rung, the anchor, and Reading T; and the shared pooling helpers

**Files:**
- Create: `src/mbfps/eval/ladder.py`
- Modify (part b): `src/mbfps/eval/split_gap.py` (four pure helpers), `scripts/split_gap.py:191-227,296-300` (its wrappers rewritten over them, behaviour-preserving)
- Test: `tests/eval/test_ladder.py`; part (b) in `tests/eval/test_split_gap.py`

**Interfaces:**
- Consumes: `mbfps.eval.split_gap.StratumContrast` (`estimate, se, z, clusters`), `clears(z, bar)`, `train_held_passes_gate(gap_final: dict[int, float]) -> bool`; `mbfps.eval.trust_readings.Q_REPORTED`.
- Produces (Task 5/6 consume): the constants `RUNGS, STEPS, REFERENCE_RUNG, DECISION_H` (re-exported from `split_gap` -- one decision horizon, M3e's), `REPORTED_H, FAMILY, SEEDS_REQUIRED, R2_SENSITIVITY, CURVE_WINDOW, OBJECTIVE_BATCHES`; part (b), in `mbfps.eval.split_gap`: `survival_indicator(summary, channel, h) -> (values, changed)`, `margin_at(summary, channel, h) -> (values, changed)`, `cell_series(summary, values, changed, *, arm, seed, rung, channel, val, horizon, context, device, torch_version) -> pooling.CellSeries`, `decision_horizon(horizon) -> (h, clamped)`; `primary_rung(min_step: int, rungs=RUNGS) -> int`; `anchor_delta(retrain_loss, reference_loss, steps: int) -> tuple[float, int | None]`; `gate_passes` (= `train_held_passes_gate`); `ArmInputs(t_free, t_probe, primary_rungs: dict[int, int], per_seed: dict[int, ArmInputs] | None)`; `TimingInputs(arms: dict[str, ArmInputs], z_fam: float, h: int)`; `Status`; `ArmReading`; `TimingReading(arms, h, z_fam)`; `reading_timing(inputs) -> TimingReading`; `format_reading_timing(reading, inputs) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_ladder.py`:

```python
"""`mbfps.eval.ladder`: the pure rules of M3f, one mutation at a time.

`primary_rung` is spec 2.2; `anchor_delta` is spec 2.4; `reading_timing` is
spec 3.3 in the table's precedence. Every rule is exercised by an input that
only that rule can decide, so a rule dropped or reordered fails a named test.
"""

import math

import numpy as np
import pytest

from mbfps.eval.ladder import (
    CURVE_WINDOW,
    DECISION_H,
    FAMILY,
    OBJECTIVE_BATCHES,
    R2_SENSITIVITY,
    REFERENCE_RUNG,
    REPORTED_H,
    RUNGS,
    SEEDS_REQUIRED,
    STEPS,
    ArmInputs,
    Status,
    TimingInputs,
    anchor_delta,
    format_reading_timing,
    gate_passes,
    primary_rung,
    reading_timing,
)
from mbfps.eval.split_gap import StratumContrast, train_held_passes_gate

Z_FAM = 2.89
"""`cluster_threshold(6, 24)` on the shipped split, to two decimals; the
tests only need a bar that 3.0 clears and 2.0 does not."""


def test_the_pre_registered_constants_are_the_specs():
    assert RUNGS == (1000, 2000, 3000, 4000, 5000)
    assert STEPS == 5000 and STEPS >= max(RUNGS)
    assert REFERENCE_RUNG == 20000
    assert DECISION_H == 15 and REPORTED_H == (5, 15, 45)
    assert FAMILY == 6 and SEEDS_REQUIRED == 2
    assert R2_SENSITIVITY == 0.1 and CURVE_WINDOW == 100 and OBJECTIVE_BATCHES == 50
    assert gate_passes is train_held_passes_gate


# ---------------------------------------------------------------------------
# primary_rung
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("min_step, rung", [
    (2045, 2000), (3555, 4000), (4510, 5000),        # frozen_ssl s0 / s1 / s2
    (1610, 2000), (4535, 5000), (3275, 3000),        # random_vit s0 / s1 / s2
    (8359, 5000), (19096, 5000), (16873, 5000),      # pixel_ae, beyond the ladder
])
def test_primary_rung_on_the_shipped_minima_is_the_specs_table(min_step, rung):
    assert primary_rung(min_step) == rung


def test_primary_rung_ties_go_to_the_earlier_rung_and_the_ends_clamp():
    assert primary_rung(1500) == 1000
    assert primary_rung(2500) == 2000
    assert primary_rung(1) == 1000
    assert primary_rung(10**9) == 5000
    assert primary_rung(3, rungs=(2, 4)) == 2
    assert primary_rung(5, rungs=(4, 2)) == 4, "the rungs need not be sorted"


def test_primary_rung_refuses_no_rungs_or_a_step_before_the_first_step():
    with pytest.raises(ValueError, match="rung"):
        primary_rung(100, rungs=())
    with pytest.raises(ValueError, match="1-based"):
        primary_rung(0)


# ---------------------------------------------------------------------------
# anchor_delta
# ---------------------------------------------------------------------------


def test_anchor_delta_is_zero_with_no_step_when_the_prefixes_are_identical():
    assert anchor_delta([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 9.0], 3) == (0.0, None)


def test_anchor_delta_is_the_largest_difference_and_names_the_first_differing_step():
    delta, step = anchor_delta([1.0, 2.5, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0], 4)
    assert delta == 1.0 and step == 2
    assert anchor_delta([1.0, 2.5, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0], 1) == (0.0, None)


def test_anchor_delta_treats_a_nan_as_a_difference_and_reports_it_as_infinite():
    delta, step = anchor_delta([1.0, float("nan")], [1.0, 2.0], 2)
    assert math.isinf(delta) and step == 2
    delta, step = anchor_delta([1.0, float("nan")], [1.0, float("nan")], 2)
    assert math.isinf(delta) and step == 2, "NaN is not equal to NaN; the anchor is exact equality"


def test_anchor_delta_refuses_a_history_shorter_than_the_steps_it_is_asked_about():
    with pytest.raises(ValueError, match="shorter"):
        anchor_delta([1.0, 2.0], [1.0, 2.0, 3.0], 3)
    with pytest.raises(ValueError, match="shorter"):
        anchor_delta([1.0, 2.0, 3.0], [1.0, 2.0], 3)
    with pytest.raises(ValueError, match="steps"):
        anchor_delta([1.0], [1.0], 0)


# ---------------------------------------------------------------------------
# Reading T
# ---------------------------------------------------------------------------


def _contrast(z: float, clusters: int = 24) -> StratumContrast:
    return StratumContrast(estimate=0.01 * z, se=0.01, z=z, clusters=clusters)


def _leaf(seed: int, free_z: float, probe_z: float = 0.0, rung: int = 2000) -> ArmInputs:
    return ArmInputs(
        t_free=_contrast(free_z), t_probe=_contrast(probe_z),
        primary_rungs={seed: rung}, per_seed=None,
    )


def _arm(free_z: float, probe_z: float = 0.0, seed_free=(4.0, 4.0, 4.0)) -> ArmInputs:
    leaves = {s: _leaf(s, z) for s, z in enumerate(seed_free)}
    return ArmInputs(
        t_free=_contrast(free_z), t_probe=_contrast(probe_z),
        primary_rungs={s: 2000 for s in leaves}, per_seed=leaves,
    )


def _inputs(**arms) -> TimingInputs:
    return TimingInputs(arms=arms, z_fam=Z_FAM, h=DECISION_H)


def _read(arm: ArmInputs):
    return reading_timing(_inputs(a=arm)).arms["a"]


def test_opposite_sign_clears_are_unresolved_and_nothing_below_is_read():
    r = _read(_arm(free_z=4.0, probe_z=-4.0))
    assert r.status is Status.UNRESOLVED_PROBE
    assert "opposite signs" in r.rule
    assert _read(_arm(free_z=-4.0, probe_z=4.0)).status is Status.UNRESOLVED_PROBE
    # Same-sign clears are NOT unresolved.
    assert _read(_arm(free_z=4.0, probe_z=4.0)).status is Status.EARLIER_BETTER


def test_a_positive_free_clear_replicated_in_two_seeds_is_earlier_better():
    r = _read(_arm(free_z=4.0, seed_free=(4.0, 4.0, 0.5)))
    assert r.status is Status.EARLIER_BETTER
    assert (r.seeds_up, r.seeds_total) == (2, 3)
    assert "2 of 3 seeds" in r.rule


def test_a_negative_free_clear_replicated_in_two_seeds_is_earlier_worse():
    r = _read(_arm(free_z=-4.0, seed_free=(-4.0, 0.5, -4.0)))
    assert r.status is Status.EARLIER_WORSE
    assert (r.seeds_down, r.seeds_total) == (2, 3)
    assert "2 of 3 seeds" in r.rule


def test_a_pooled_clear_in_one_seed_only_is_no_difference_and_the_sentence_says_so():
    r = _read(_arm(free_z=4.0, seed_free=(4.0, 0.5, 0.5)))
    assert r.status is Status.NO_DIFFERENCE
    assert "only 1 of 3 seeds" in r.rule and f">= {SEEDS_REQUIRED} required" in r.rule
    r = _read(_arm(free_z=-4.0, seed_free=(-4.0, 0.5, 0.5)))
    assert r.status is Status.NO_DIFFERENCE and "only 1 of 3 seeds" in r.rule


def test_an_interval_covering_zero_is_no_difference():
    r = _read(_arm(free_z=1.0, seed_free=(4.0, 4.0, 4.0)))
    assert r.status is Status.NO_DIFFERENCE
    assert "does not clear" in r.rule
    assert r.seeds_up == 3, "the seeds still counted; the pooled bar decided"


def test_the_bar_is_strict_and_a_nan_or_infinite_z_never_clears():
    assert _read(_arm(free_z=Z_FAM)).status is Status.NO_DIFFERENCE
    assert _read(_arm(free_z=3.0)).status is Status.EARLIER_BETTER
    assert _read(_arm(free_z=float("nan"))).status is Status.NO_DIFFERENCE
    assert _read(_arm(free_z=float("inf"))).status is Status.NO_DIFFERENCE
    assert _read(_arm(free_z=4.0, probe_z=float("-inf"))).status is Status.EARLIER_BETTER, \
        "an infinite probe z is not a clear, so it cannot make the arm unresolved"


def test_a_single_seed_leaf_has_a_vacuous_replication_clause():
    """The M3e correction, built in: a leaf read alone (per_seed=None) is not
    held to SEEDS_REQUIRED -- there is nothing to replicate across."""
    up = _read(_leaf(0, free_z=4.0))
    assert up.status is Status.EARLIER_BETTER and (up.seeds_up, up.seeds_total) == (1, 1)
    assert "this seed alone" in up.rule
    down = _read(_leaf(0, free_z=-4.0))
    assert down.status is Status.EARLIER_WORSE and (down.seeds_down, down.seeds_total) == (1, 1)
    flat = _read(_leaf(0, free_z=1.0))
    assert flat.status is Status.NO_DIFFERENCE and (flat.seeds_up, flat.seeds_total) == (0, 1)


def test_an_empty_per_seed_dict_is_zero_of_zero_and_never_replicates():
    arm = ArmInputs(t_free=_contrast(4.0), t_probe=_contrast(0.0), primary_rungs={}, per_seed={})
    r = _read(arm)
    assert r.status is Status.NO_DIFFERENCE and (r.seeds_up, r.seeds_total) == (0, 0)
    assert "only 0 of 0 seeds" in r.rule


def test_arms_are_read_independently_and_in_the_callers_order():
    reading = reading_timing(_inputs(
        random_vit=_arm(4.0), pixel_ae=_arm(-4.0, seed_free=(-4.0, -4.0, -4.0)), frozen_ssl=_arm(0.5),
    ))
    assert list(reading.arms) == ["random_vit", "pixel_ae", "frozen_ssl"]
    assert [r.status for r in reading.arms.values()] == [
        Status.EARLIER_BETTER, Status.EARLIER_WORSE, Status.NO_DIFFERENCE,
    ]
    assert reading.h == DECISION_H and reading.z_fam == Z_FAM


def test_format_prints_the_contrasts_the_primary_rungs_and_a_verdict_per_arm():
    inputs = _inputs(frozen_ssl=_arm(4.0), random_vit=_arm(0.5))
    text = format_reading_timing(reading_timing(inputs), inputs)
    assert f"Reading T" in text and f"h={DECISION_H}" in text and f"z_fam = {Z_FAM:.2f}" in text
    assert "frozen_ssl" in text and "EARLIER BETTER" in text
    assert "random_vit" in text and "NO DIFFERENCE" in text
    assert "s0 2000" in text, "the primary rung per seed is printed beside the verdict"
    assert "decided by:" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_ladder.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mbfps.eval.ladder'`.

- [ ] **Step 3: Implement**

Create `src/mbfps/eval/ladder.py`:

```python
"""The checkpoint ladder's pure rules -- milestone M3f.

Three things, none of which touches torch or a file:

  * spec 2.2 -- `primary_rung`: the rung nearest a cell's smoothed
    `embedding` minimum, ties to the earlier;
  * spec 2.4 -- `anchor_delta`: does the retrain's loss curve equal the
    original run's, step for step, and where does it first differ;
  * spec 3.3 -- Reading T: per arm, the paired rung contrast on the free
    channel against `z_fam`, its probe twin as the control, the replication
    across seeds, and one status with the sentence that decided it.

The contrasts arrive as `split_gap.StratumContrast` (estimate, se, z,
clusters) -- the same reduction of a pooled contrast Reading G reads -- built
by `scripts/checkpoint_ladder.py` from `pooling.paired_contrast`.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from mbfps.eval.split_gap import DECISION_H, StratumContrast, clears, train_held_passes_gate

RUNGS: tuple[int, ...] = (1000, 2000, 3000, 4000, 5000)
"""The checkpoint steps saved during the retrain (spec 2.1)."""
STEPS: int = 5000
"""The retrain's length; every rung lies at or before it."""
REFERENCE_RUNG: int = 20000
"""The M3c study's checkpoint, read from `--reference`, never retrained."""
# DECISION_H is M3e's, imported above: one decision horizon (15) for both
# readings, so `split_gap.decision_horizon` serves this script unchanged.
REPORTED_H: tuple[int, ...] = (5, 15, 45)
FAMILY: int = 6
"""Reading T's family: three arms x two channels at one horizon (spec 3.1)."""
SEEDS_REQUIRED: int = 2
R2_SENSITIVITY: float = 0.1
CURVE_WINDOW: int = 100
"""The moving-mean width the primary rung's minimum is read at -- M3e's."""
OBJECTIVE_BATCHES: int = 50
"""Validation draws per rung for `val_objective` (spec 2.3)."""

gate_passes = train_held_passes_gate
"""Spec 3.3's gate line: `gap_closed(45)` on position > 0 in EVERY seed, a
NaN never > 0 -- the same predicate Reading G applied to `train_held`, under
the name this reading uses."""


def primary_rung(min_step: int, rungs=RUNGS) -> int:
    """The rung nearest `min_step`; ties go to the EARLIER rung; a minimum
    before the first rung reads the first and one beyond the last reads the
    last (spec 2.2). `rungs` need not be sorted."""
    ordered = sorted(int(r) for r in rungs)
    if not ordered:
        raise ValueError("primary_rung needs at least one rung")
    min_step = int(min_step)
    if min_step < 1:
        raise ValueError(f"min_step is a 1-based training step, got {min_step}")
    best = ordered[0]
    for rung in ordered[1:]:
        if abs(rung - min_step) < abs(best - min_step):
            best = rung
    return best


def anchor_delta(retrain_loss, reference_loss, steps: int) -> tuple[float, int | None]:
    """`max |retrain - reference|` over the first `steps` per-step losses and
    the 1-based step at which they first differ, or `(0.0, None)` when the
    two prefixes are equal element for element. Equality is exact: a NaN on
    either side is a difference (NaN != NaN) and is reported as an infinite
    delta. Either history shorter than `steps` is refused."""
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    a = np.asarray(list(retrain_loss)[:steps], dtype=float)
    b = np.asarray(list(reference_loss)[:steps], dtype=float)
    if a.size < steps or b.size < steps:
        raise ValueError(
            f"a history is shorter than the {steps} steps the anchor covers: retrain "
            f"{a.size}, reference {b.size}"
        )
    differing = ~(a == b)
    if not differing.any():
        return 0.0, None
    first = int(np.argmax(differing)) + 1
    diff = np.abs(a - b)
    if not np.isfinite(diff[differing]).all():
        return float("inf"), first
    return float(diff.max()), first


# ---------------------------------------------------------------------------
# Reading T -- the timing hypothesis (spec 3.3), over pooled inputs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmInputs:
    """One arm's paired rung contrasts at the decision horizon
    (`primary - reference`), the primary rung each seed was read at, and the
    same inputs within each seed alone (leaves carry `per_seed=None`)."""

    t_free: StratumContrast
    t_probe: StratumContrast
    primary_rungs: dict[int, int]
    per_seed: "dict[int, ArmInputs] | None"


@dataclass(frozen=True)
class TimingInputs:
    arms: dict[str, ArmInputs]
    z_fam: float
    h: int


class Status(str, Enum):
    """Spec 3.3's four outcomes, in the table's order of precedence."""

    UNRESOLVED_PROBE = "unresolved through the probe"
    EARLIER_BETTER = "earlier better"
    EARLIER_WORSE = "earlier worse"
    NO_DIFFERENCE = "no difference"


@dataclass(frozen=True)
class ArmReading:
    """`seeds_up` / `seeds_down` / `seeds_total`: the replication the status
    was decided on. A pooled reading (`per_seed` a dict, possibly empty)
    counts that dict's clearing leaves; a single-seed leaf read alone
    (`per_seed=None`) has a vacuous replication clause -- `seeds_total = 1`,
    `seeds_up`/`seeds_down` 1 where its own contrast clears -- so a clearing
    leaf reads EARLIER_BETTER or EARLIER_WORSE on its own."""

    arm: str
    status: Status
    rule: str
    free_clears_up: bool
    free_clears_down: bool
    probe_clears_up: bool
    probe_clears_down: bool
    seeds_up: int
    seeds_down: int
    seeds_total: int


@dataclass(frozen=True)
class TimingReading:
    arms: dict[str, ArmReading]
    h: int
    z_fam: float


def _fmt(value: float, spec: str = "+.2f") -> str:
    return format(value, spec) if np.isfinite(value) else str(value)


def _arm_reading(arm: str, a: ArmInputs, z_fam: float) -> ArmReading:
    """The rules of spec 3.3, in the table's precedence, each status carrying
    the sentence that decided it."""
    free_up, free_down = clears(a.t_free.z, z_fam), clears(-a.t_free.z, z_fam)
    probe_up, probe_down = clears(a.t_probe.z, z_fam), clears(-a.t_probe.z, z_fam)
    if a.per_seed is None:
        # A single-seed leaf, read alone: there is nothing to replicate across.
        seeds_up, seeds_down, seeds_total = int(free_up), int(free_down), 1
        replicated_up = replicated_down = True
        up_words = down_words = "this seed alone"
    else:
        seeds_up = sum(1 for leaf in a.per_seed.values() if clears(leaf.t_free.z, z_fam))
        seeds_down = sum(1 for leaf in a.per_seed.values() if clears(-leaf.t_free.z, z_fam))
        seeds_total = len(a.per_seed)
        replicated_up = seeds_up >= SEEDS_REQUIRED
        replicated_down = seeds_down >= SEEDS_REQUIRED
        up_words = f"{seeds_up} of {seeds_total} seeds"
        down_words = f"{seeds_down} of {seeds_total} seeds"
    fz, pz, bar = _fmt(a.t_free.z), _fmt(a.t_probe.z), f"{z_fam:.2f}"
    if (free_up and probe_down) or (free_down and probe_up):
        status = Status.UNRESOLVED_PROBE
        rule = f"T_free z {fz} and T_probe z {pz} both clear +-{bar} with opposite signs"
    elif free_up and replicated_up:
        status = Status.EARLIER_BETTER
        rule = f"T_free z {fz} > {bar} pooled and in {up_words}"
    elif free_down and replicated_down:
        status = Status.EARLIER_WORSE
        rule = f"T_free z {fz} < -{bar} pooled and in {down_words}"
    elif free_up:
        status = Status.NO_DIFFERENCE
        rule = (f"T_free z {fz} clears {bar} pooled but in only {up_words} "
                f"(>= {SEEDS_REQUIRED} required)")
    elif free_down:
        status = Status.NO_DIFFERENCE
        rule = (f"T_free z {fz} clears -{bar} pooled but in only {down_words} "
                f"(>= {SEEDS_REQUIRED} required)")
    else:
        status = Status.NO_DIFFERENCE
        rule = f"T_free z {fz} does not clear +-{bar}"
    return ArmReading(
        arm=arm, status=status, rule=rule,
        free_clears_up=free_up, free_clears_down=free_down,
        probe_clears_up=probe_up, probe_clears_down=probe_down,
        seeds_up=seeds_up, seeds_down=seeds_down, seeds_total=seeds_total,
    )


def reading_timing(inputs: TimingInputs) -> TimingReading:
    """One `ArmReading` per arm, in the caller's order; arms never read each
    other (spec 4: no ranking)."""
    return TimingReading(
        arms={arm: _arm_reading(arm, a, inputs.z_fam) for arm, a in inputs.arms.items()},
        h=inputs.h, z_fam=inputs.z_fam,
    )


def format_reading_timing(reading: TimingReading, inputs: TimingInputs) -> str:
    """The contrast table, the primary rungs, and the verdict lines, in
    `split_gap.txt`'s style."""
    lines = [
        f"--- Reading T: the timing hypothesis at h={reading.h} (primary rung - "
        f"{REFERENCE_RUNG}, paired on the val windows); z_fam = {reading.z_fam:.2f} ---",
        f"  {'arm':<12}{'channel':<9}{'estimate':>10}{'se':>9}{'z':>8}  clears",
    ]
    for arm, a in inputs.arms.items():
        for channel, c in (("free", a.t_free), ("probe", a.t_probe)):
            verdict = "yes" if clears(abs(c.z), reading.z_fam) else "no"
            lines.append(
                f"  {arm:<12}{channel:<9}{_fmt(c.estimate, '+.4f'):>10}{_fmt(c.se, '.4f'):>9}"
                f"{_fmt(c.z):>8}  {verdict}"
            )
    lines.append("  primary rung per seed (the rung nearest the reference run's smoothed embedding minimum):")
    for arm, a in inputs.arms.items():
        rungs = " / ".join(f"s{seed} {rung}" for seed, rung in sorted(a.primary_rungs.items()))
        lines.append(f"    {arm:<12}{rungs}")
    for arm, r in reading.arms.items():
        lines.append(
            f"  verdict: {arm:<12}{r.status.name.replace('_', ' ')} -- decided by: {r.rule}"
        )
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_ladder.py -q`
Expected: 26 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/eval/ladder.py tests/eval/test_ladder.py
git commit -m "feat: the pre-registered rules of M3f -- the primary rung, the anchor delta, and Reading T over paired rung contrasts with the single-seed clause built in

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

#### Part (b): the shared pooling helpers

`scripts/split_gap.py` builds `pooling.CellSeries` from a stratum summary in `_series`, `survival_series` and `margin_series`, and clamps the decision horizon in `decision_horizon`. Task 6 needs exactly the same four operations on a rung's summary. The pure parts move into `mbfps.eval.split_gap`; the script's functions become wrappers; its existing tests (`test_survival_series_is_the_indicator_over_the_windows_that_ever_moved`, `test_margin_series_masks_by_moved_at_h_not_ever_moved`, `test_decision_horizon_is_fifteen_unless_the_run_is_shorter` in `tests/eval/test_split_gap_script.py`) pin that nothing moved.

- [ ] **Step 6: Write the failing tests**

Append to `tests/eval/test_split_gap.py`:

```python
# ---------------------------------------------------------------------------
# The shared pooling helpers (M3f Task 4b): from a summary to a CellSeries.
# ---------------------------------------------------------------------------

import mbfps.eval.pooling as pooling  # noqa: E402
from mbfps.eval.split_gap import cell_series, decision_horizon, margin_at, survival_indicator  # noqa: E402


def _summary(crossing, margin, moved, episode=(0, 0, 1, 1)) -> dict:
    return {
        "windows": {"total": len(episode), "episode": list(episode), "clusters": len(set(episode))},
        "crossing": {"free": list(crossing), "probe": list(crossing)},
        "margin": {"free": [list(row) for row in margin], "probe": [list(row) for row in margin]},
        "moved": [list(row) for row in moved],
    }


IDENTITY = dict(arm="random_vit", seed=0, rung="val", channel="S/free", val=("ep0", "ep1"),
                horizon=3, context=2, device="cpu", torch_version="2.13.0")


def test_survival_indicator_is_alive_at_h_over_the_windows_that_ever_moved():
    values, changed = survival_indicator(_summary([4, 2, np.nan, 1], [[0] * 3] * 4, [[True] * 3] * 4), "free", 2)
    np.testing.assert_array_equal(values, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(changed, [True, True, False, True])


def test_margin_at_reads_column_h_minus_one_under_moved_at_h():
    margin = [[3, 30], [-1, -10], [2, 20], [0, np.nan]]
    moved = [[False, True], [False, True], [False, True], [False, True]]
    values, changed = margin_at(_summary([4] * 4, margin, moved), "free", 2)
    np.testing.assert_array_equal(values[:3], [30.0, -10.0, 20.0])
    assert changed.tolist() == [True, True, True, False], "a NaN margin is not a measurement"
    _, at_one = margin_at(_summary([4] * 4, margin, moved), "free", 1)
    assert not at_one.any(), "nothing has moved at h=1"


def test_cell_series_carries_the_summarys_windows_and_the_identity_it_is_given():
    summary = _summary([4, 2, 3, 1], [[0] * 3] * 4, [[True] * 3] * 4)
    s = cell_series(summary, [1.0, 0.0, 1.0, 0.0], [True, True, False, True], **IDENTITY)
    assert isinstance(s, pooling.CellSeries)
    assert (s.arm, s.seed, s.rung, s.channel) == ("random_vit", 0, "val", "S/free")
    np.testing.assert_array_equal(s.episode, [0, 0, 1, 1])
    assert s.windows_total == 4 and s.val == ("ep0", "ep1")
    assert (s.horizon, s.context, s.device, s.torch_version) == (3, 2, "cpu", "2.13.0")
    assert s.embedding is None and s.noise is None
    # Windows 0, 1 and 3 are kept (window 2 is `changed=False`): the mean
    # over [1.0, 0.0, 0.0] is 1/3, whatever window 2 holds.
    pooled = pooling.pool_arm([s])
    assert pooled.mean == pytest.approx(1 / 3) and pooled.windows == 3


def test_cell_series_refuses_values_that_are_not_one_per_window():
    summary = _summary([4, 2, 3, 1], [[0] * 3] * 4, [[True] * 3] * 4)
    with pytest.raises(ValueError, match="n_windows"):
        cell_series(summary, [1.0, 0.0], [True, True], **IDENTITY)
    with pytest.raises(ValueError, match="n_windows"):
        cell_series(summary, [1.0, 0.0, 1.0, 0.0], [True, True], **IDENTITY)


def test_decision_horizon_is_the_pre_registered_one_unless_the_run_is_shorter():
    assert decision_horizon(45) == (DECISION_H, False)
    assert decision_horizon(DECISION_H) == (DECISION_H, False)
    assert decision_horizon(3) == (3, True)
```

(`DECISION_H` and `np`/`pytest` are already imported at the top of that file.)

- [ ] **Step 7: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_split_gap.py -q -k "indicator or margin_at or cell_series or decision_horizon"`
Expected: FAIL — `ImportError: cannot import name 'cell_series'`.

- [ ] **Step 8: Implement**

In `src/mbfps/eval/split_gap.py`, add `import mbfps.eval.pooling as pooling` to the imports and append, after `learning_curve_summary`:

```python
# ---------------------------------------------------------------------------
# From a stratum summary to the series the pool reads (spec 3.1; shared with
# the M3f ladder, which pools a rung's summary the same way).
# ---------------------------------------------------------------------------


def survival_indicator(summary: dict, channel: str, h: int) -> tuple[np.ndarray, np.ndarray]:
    """`1[h_x > h]` per window, and the mask of the windows that moved within
    the horizon (finite crossing) -- so the pooled mean is S(h) exactly."""
    crossing = np.asarray(summary["crossing"][channel], dtype=float)
    return (crossing > h).astype(float), np.isfinite(crossing)


def margin_at(summary: dict, channel: str, h: int) -> tuple[np.ndarray, np.ndarray]:
    """`Delta(h)` per window, and the mask of the windows moved AT h whose
    margin is a number."""
    values = np.asarray(summary["margin"][channel], dtype=float)[:, h - 1]
    moved = np.asarray(summary["moved"], dtype=bool)[:, h - 1]
    return values, moved & np.isfinite(values)


def cell_series(summary: dict, values, changed, *, arm: str, seed: int, rung: str, channel: str,
                val, horizon: int, context: int, device: str, torch_version: str) -> pooling.CellSeries:
    """One per-window series of one cell, in the shape the pool reads: the
    summary's window index as the cluster labels, `values` under `changed`,
    and the identity `require_compatible` checks. `arm` and `rung` are
    labels: the split gap passes the arm and the stratum, the ladder a rung
    group and `val`."""
    values = np.asarray(values, dtype=float)
    changed = np.asarray(changed, dtype=bool)
    episode = np.asarray(summary["windows"]["episode"], dtype=int)
    if not (values.shape == changed.shape == episode.shape):
        raise ValueError(
            f"{arm} seed {seed} {rung}/{channel}: values {values.shape}, changed {changed.shape} "
            f"and windows.episode {episode.shape} must all be (n_windows,)"
        )
    return pooling.CellSeries(
        arm=arm, seed=int(seed), rung=rung, channel=channel,
        delta=values, changed=changed, episode=episode, embedding=None, noise=None,
        windows_total=int(summary["windows"]["total"]), val=tuple(val),
        horizon=int(horizon), context=int(context), device=str(device), torch_version=str(torch_version),
    )


def decision_horizon(horizon: int) -> tuple[int, bool]:
    """`DECISION_H`, clamped to the run's horizon; the flag says it was."""
    if DECISION_H <= horizon:
        return DECISION_H, False
    return int(horizon), True
```

In `scripts/split_gap.py`, extend the `from mbfps.eval.split_gap import (...)` list with `cell_series`, `decision_horizon`, `margin_at`, `survival_indicator`; replace the bodies of `_series`, `survival_series` and `margin_series` with:

```python
def _series(record: dict, stratum: str, channel: str, values, changed) -> pooling.CellSeries:
    """One per-window series of one cell's stratum, in the shape the pool
    reads. `rung` is the stratum and `channel` the reading ("S/free",
    "margin/probe"), so `require_compatible` pools seeds of one stratum only
    and `unpaired_contrast` refuses two different readings."""
    return cell_series(
        record["strata"][stratum], values, changed,
        arm=record["arm"], seed=record["seed"], rung=stratum, channel=channel,
        val=record["episodes"][stratum], horizon=record["horizon"], context=record["context"],
        device=record["device"], torch_version=record["torch_version"],
    )


def survival_series(record: dict, stratum: str, channel: str, h: int) -> pooling.CellSeries:
    """`1[h_x > h]` per window over the windows that moved within the horizon
    (finite crossing) -- so the pooled mean is S(h) exactly."""
    return _series(record, stratum, f"S/{channel}", *survival_indicator(record["strata"][stratum], channel, h))


def margin_series(record: dict, stratum: str, channel: str, h: int) -> pooling.CellSeries:
    """`Delta(h)` per window over the windows moved AT h."""
    return _series(record, stratum, f"margin/{channel}", *margin_at(record["strata"][stratum], channel, h))
```

and delete the script's own `def decision_horizon(...)` (the imported name serves every caller, including the tests' `script.decision_horizon`).

- [ ] **Step 9: Run the helper tests and the split-gap script tests that pin the wrappers**

Run: `.venv/bin/python -m pytest tests/eval/test_split_gap.py tests/eval/test_split_gap_script.py -q`
Expected: all pass (20 + 5 new in the pure file; the 25 script tests unchanged, including the three that pin `survival_series`, `margin_series` and `decision_horizon`).

- [ ] **Step 10: Commit**

```bash
git add src/mbfps/eval/split_gap.py scripts/split_gap.py tests/eval/test_split_gap.py
git commit -m "refactor: the split gap's series builders and decision_horizon become pure helpers in mbfps.eval.split_gap, shared with the ladder; the script's wrappers pinned unchanged

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `scripts/checkpoint_ladder.py` part 1 — the `train` and `evaluate` phases, one train record and one ladder record per cell

**Files:**
- Create: `scripts/checkpoint_ladder.py`
- Test: `tests/eval/test_checkpoint_ladder_script.py`

**Interfaces:**
- Consumes: `train_world_model(..., checkpoint_steps)`, `history_at` (Task 1); `evaluate_job` (Task 2); `val_objective` (Task 3); `ladder.*` (Task 4); by path from `trust_horizon.py`: `Cell`, `CellMissing`, `load_cell`, `self_check`, `prepare_cell`, `probe_is_measurable`, `checkpoint_path`, the exit constants; `split_gap.stratum_summary`, `split_gap.learning_curve_summary`.
- Produces (Task 6 consumes): `rung_dir(out, step)`, `train_record_path(out, arm, seed)`, `ladder_record_path(out, arm, seed)`; the ladder record's shape — `entries: {str(step): {step, record_git_sha, gate: {gap_final, steps_degenerate}, probe: {selection_r2, measurable}, objective: {...}, train_embedding, self_check (reference only), summary: <stratum_summary>}}` for every rung and `REFERENCE_RUNG`, plus `primary_rung`, `embedding_min_step`, `rungs`, `episodes.val`, provenance; `main(argv=None, *, rungs=RUNGS) -> int` with `--phase train|evaluate|all` (Task 6 adds `read`).

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_checkpoint_ladder_script.py`:

```python
"""scripts/checkpoint_ladder.py: retrain a cell saving rungs, anchor it to the
reference run, evaluate every rung and the reference through the study's own
evaluation half and the trust pass, one ladder record per cell; then (Task 6)
the pooling, Reading T, the tables and ladder.txt.

The script is loaded by path. The REFERENCE is a real tiny cell on
`wide_buffer` -- thirty 40-step episodes, `run_job(..., steps=5, seq_len=4,
context=2, horizon=3, device="cpu")` -- with the diagnostic
`diagnose_dynamics.main` writes, exactly as the split-gap tests build theirs.
The LADDER retrains the same cell to `--steps 4` saving rungs (2, 4): on CPU
the first four losses reproduce the reference's first four bitwise, so
`--anchor hard` passes and every refusal below is exercised by doctoring one
file.

THE FIXTURE'S FACTS: six validation episodes, `window_starts(40, 2, 3)` cuts
eight windows per episode -- 48 windows over 6 clusters on every rung.
"""

import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from mbfps.eval.ladder import CURVE_WINDOW, OBJECTIVE_BATCHES, REFERENCE_RUNG, STEPS
from mbfps.eval.split_gap import CURVE_NAMES
from mbfps.eval.study import StudyJob, job_record_path, load_record, run_job
from mbfps.training.world_model import history_at

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script("checkpoint_ladder")
diagnose = _load_script("diagnose_dynamics")
trust = _load_script("trust_horizon")

JOB = StudyJob("random_vit", 1)
JOB_KW = dict(steps=5, seq_len=4, context=2, horizon=3, device="cpu")
CONTEXT, HORIZON = JOB_KW["context"], JOB_KW["horizon"]
LADDER_RUNGS = (2, 4)
LADDER_STEPS = 4
WINDOWS, CLUSTERS = 48, 6
RECORD = "result_random_vit_seed1.json"
DIAGNOSTIC = "diagnostic_random_vit_seed1.json"
TRAIN = "train_random_vit_seed1.json"
LADDER = "ladder_random_vit_seed1.json"
CHECKPOINT = "world_model_random_vit_seed1.pt"

LADDER_KEYS = {
    "arm", "seed", "context", "horizon", "split_seed", "device", "torch_version", "git_sha",
    "reference_git_sha", "train_git_sha", "anchor", "rungs", "reference_rung", "primary_rung",
    "embedding_min_step", "curve_window", "objective_batches", "episodes", "entries", "nonfinite",
}
ENTRY_KEYS = {
    "step", "record_git_sha", "gate", "probe", "objective", "train_embedding", "self_check", "summary",
}
OBJECTIVE_KEYS = {"loss", "embedding", "reward", "continue", "kl_dyn", "kl_rep"}


@pytest.fixture
def reference(tmp_path, wide_buffer, capsys):
    """One real cell on thirty episodes and the diagnostic the ladder writes for it."""
    out = tmp_path / "reference"
    run_job(JOB, wide_buffer, out, **JOB_KW)
    status = diagnose.main([
        "--out", str(out), "--data", str(wide_buffer.root), "--device", "cpu",
        "--arms", JOB.arm, "--seeds", str(JOB.seed),
        "--context", str(CONTEXT), "--horizon", str(HORIZON), "--ks", "1", str(HORIZON),
    ])
    assert status == diagnose.EXIT_OK, "the fixture's diagnostic was not written cleanly"
    capsys.readouterr()
    return types.SimpleNamespace(out=out, data=wide_buffer.root, ladder=tmp_path / "ladder")


def _argv(ref, *extra: str) -> list[str]:
    return [
        "--out", str(ref.ladder), "--reference", str(ref.out), "--data", str(ref.data),
        "--device", "cpu", "--arms", JOB.arm, "--seeds", str(JOB.seed),
        "--steps", str(LADDER_STEPS), "--objective-batches", "2", "--window", "2", *extra,
    ]


def _run(ref, *extra: str) -> int:
    return script.main(_argv(ref, *extra), rungs=LADDER_RUNGS)


@pytest.fixture
def trained(reference, capsys):
    assert _run(reference, "--phase", "train", "--anchor", "hard") == script.EXIT_OK
    capsys.readouterr()
    return reference


@pytest.fixture
def evaluated(trained, capsys):
    assert _run(trained, "--phase", "evaluate", "--anchor", "hard") == script.EXIT_OK
    capsys.readouterr()
    return trained


def _doctor(path: Path, edit) -> None:
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


def _never_train(*args, **kwargs):
    raise AssertionError("train_world_model ran; this refusal must come before any training")


def _never_evaluate(*args, **kwargs):
    raise AssertionError("evaluate_job ran; this refusal must come before any rung is evaluated")


# ---------------------------------------------------------------------------
# The parser and the statuses.
# ---------------------------------------------------------------------------


def test_the_parser_defaults_are_the_specs():
    args = script._parser().parse_args([])
    assert args.out == Path("runs/m3f_ladder") and args.reference == Path("runs/m3_study_v2")
    assert args.data == Path("data/my_way_home") and args.device == "mps"
    assert args.context is None and args.horizon is None
    assert args.steps == STEPS and args.phase == "all"
    assert args.anchor in script.ANCHOR_POLICIES == ("hard", "report")
    assert args.objective_batches == OBJECTIVE_BATCHES and args.window == CURVE_WINDOW
    assert args.figure is None


def test_steps_below_the_last_rung_is_argparses_own_usage_error():
    """Judged at the parser, before any directory is read."""
    with pytest.raises(SystemExit) as raised:
        script.main(["--steps", "1"], rungs=LADDER_RUNGS)
    assert raised.value.code == 2


def test_the_reused_statuses_are_trust_horizons_and_32_and_33_are_new():
    assert (script.EXIT_OK, script.EXIT_NO_CHECKPOINTS, script.EXIT_SPLIT_MISMATCH,
            script.EXIT_RECORD_MISMATCH, script.EXIT_SELF_CHECK_FAILED) == (0, 11, 12, 14, 30)
    assert script.EXIT_NO_CHECKPOINTS is trust.EXIT_NO_CHECKPOINTS
    assert script.EXIT_SELF_CHECK_FAILED is trust.EXIT_SELF_CHECK_FAILED
    assert (script.EXIT_ANCHOR_MISMATCH, script.EXIT_RUNG_MISLABELLED) == (32, 33)
    own = {v for k, v in vars(script).items() if k.startswith("EXIT_")}
    assert own == {0, 11, 12, 14, 30, 32, 33}


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def test_train_writes_the_rungs_the_final_checkpoint_and_a_train_record_anchored_at_zero(trained):
    for step in LADDER_RUNGS:
        payload = torch.load(trained.ladder / f"step{step}" / CHECKPOINT, weights_only=True)
        assert (payload["arm"], payload["seed"], payload["step"]) == (JOB.arm, JOB.seed, step)
    assert (trained.ladder / CHECKPOINT).is_file()

    record = load_record(trained.ladder / TRAIN)
    assert record["arm"] == JOB.arm and record["seed"] == JOB.seed
    assert record["steps"] == LADDER_STEPS and record["rungs"] == list(LADDER_RUNGS)
    assert record["seq_len"] == JOB_KW["seq_len"]
    assert len(record["history"]["loss"]) == LADDER_STEPS == len(record["history"]["parts"])
    assert sorted(record["checkpoint_seconds"]) == ["2", "4"]
    assert record["anchor"] == {
        "policy": "hard", "steps": LADDER_STEPS, "max_delta": 0.0, "first_step": None,
        "reference_git_sha": load_record(trained.out / RECORD)["git_sha"],
    }
    # THE ANCHOR'S CONTENT: the retrain's losses ARE the reference run's first four.
    reference = load_record(trained.out / RECORD)
    assert record["history"]["loss"] == reference["history"]["loss"][:LADDER_STEPS]
    assert record["git_sha"] and record["device"] == "cpu" and record["torch_version"] == torch.__version__


def test_a_missing_reference_cell_is_exit_11_before_any_training(reference, monkeypatch, capsys):
    (reference.out / DIAGNOSTIC).unlink()
    monkeypatch.setattr(script, "train_world_model", _never_train)
    assert _run(reference, "--phase", "train") == script.EXIT_NO_CHECKPOINTS
    assert "NO CELL" in capsys.readouterr().out
    assert not reference.ladder.exists()


def test_a_hard_anchor_that_does_not_match_is_exit_32_naming_the_first_differing_step(reference, capsys):
    _doctor(reference.out / RECORD, lambda r: r["history"]["loss"].__setitem__(1, r["history"]["loss"][1] + 1.0))
    assert _run(reference, "--phase", "train", "--anchor", "hard") == script.EXIT_ANCHOR_MISMATCH
    out = capsys.readouterr().out
    assert "ANCHOR MISMATCH" in out and "first at step 2" in out
    # The train record IS written -- it is the evidence -- and says what it measured.
    record = load_record(reference.ladder / TRAIN)
    assert record["anchor"]["max_delta"] == pytest.approx(1.0) and record["anchor"]["first_step"] == 2
    assert record["anchor"]["policy"] == "hard"


def test_a_report_anchor_prints_the_delta_and_continues(reference, capsys):
    _doctor(reference.out / RECORD, lambda r: r["history"]["loss"].__setitem__(1, r["history"]["loss"][1] + 1.0))
    assert _run(reference, "--phase", "train", "--anchor", "report") == script.EXIT_OK
    out = capsys.readouterr().out
    assert "anchor max|delta| 1.0e+00 (first at step 2)" in out and "ANCHOR MISMATCH" not in out
    assert load_record(reference.ladder / TRAIN)["anchor"]["policy"] == "report"


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_writes_a_study_record_per_rung_and_one_ladder_record(evaluated):
    reference = load_record(evaluated.out / RECORD)
    for step in LADDER_RUNGS:
        rung = load_record(job_record_path(evaluated.ladder / f"step{step}", JOB))
        assert rung["steps"] == step and len(rung["history"]["loss"]) == step
        assert rung["history"]["loss"] == reference["history"]["loss"][:step]
        assert (rung["context"], rung["horizon"], rung["seq_len"]) == (CONTEXT, HORIZON, JOB_KW["seq_len"])
        assert rung["episodes"] == reference["episodes"]

    ladder = load_record(evaluated.ladder / LADDER)
    assert set(ladder) == LADDER_KEYS
    assert ladder["rungs"] == list(LADDER_RUNGS) and ladder["reference_rung"] == REFERENCE_RUNG
    assert set(ladder["entries"]) == {"2", "4", str(REFERENCE_RUNG)}
    assert ladder["primary_rung"] in LADDER_RUNGS and 1 <= ladder["embedding_min_step"] <= JOB_KW["steps"]
    assert ladder["episodes"]["val"] == reference["episodes"]["val"]
    assert ladder["anchor"]["max_delta"] == 0.0 and ladder["objective_batches"] == 2
    assert ladder["reference_git_sha"] == reference["git_sha"]
    for key, entry in ladder["entries"].items():
        assert set(entry) == ENTRY_KEYS and entry["step"] == int(key)
        assert set(entry["objective"]) == OBJECTIVE_KEYS
        assert np.isfinite(entry["train_embedding"])
        assert set(entry["gate"]) == {"gap_final", "steps_degenerate"}
        assert set(entry["probe"]) == {"selection_r2", "measurable"}
        summary = entry["summary"]
        assert summary["windows"]["total"] == WINDOWS and summary["windows"]["clusters"] == CLUSTERS
        assert set(summary["curves"]) == set(CURVE_NAMES)
        assert np.asarray(summary["crossing"]["free"]).shape == (WINDOWS,)
        assert np.asarray(summary["margin"]["probe"]).shape == (WINDOWS, HORIZON)
    assert ladder["entries"][str(REFERENCE_RUNG)]["self_check"]["ok"] is True
    assert ladder["entries"][str(REFERENCE_RUNG)]["self_check"]["reference_position_max_delta"] == 0.0
    assert ladder["entries"]["2"]["self_check"] is None
    assert ladder["entries"][str(REFERENCE_RUNG)]["gate"]["gap_final"] == reference["position"]["gap_final"] or (
        np.isnan(ladder["entries"][str(REFERENCE_RUNG)]["gate"]["gap_final"]) and np.isnan(reference["position"]["gap_final"])
    )


def test_the_reference_entrys_train_embedding_is_the_reference_runs_own_smoothed_term(evaluated):
    ladder = load_record(evaluated.ladder / LADDER)
    reference = load_record(evaluated.out / RECORD)
    parts = reference["history"]["parts"]
    expected = float(np.mean([p["embedding"] for p in parts[-2:]]))  # --window 2 at the last step
    assert ladder["entries"][str(REFERENCE_RUNG)]["train_embedding"] == pytest.approx(expected)
    train = load_record(evaluated.ladder / TRAIN)
    expected_2 = float(np.mean([p["embedding"] for p in train["history"]["parts"][:2]]))
    assert ladder["entries"]["2"]["train_embedding"] == pytest.approx(expected_2)


def test_evaluate_without_a_train_record_is_exit_11(reference, capsys):
    assert _run(reference, "--phase", "evaluate") == script.EXIT_NO_CHECKPOINTS
    assert "run --phase train first" in capsys.readouterr().out


def test_a_doctored_reference_curve_is_exit_30_before_any_rung_is_evaluated(trained, monkeypatch, capsys):
    _doctor(trained.out / DIAGNOSTIC, lambda d: d["curves"]["reference_position"].__setitem__(
        0, d["curves"]["reference_position"][0] + 1e-3))
    monkeypatch.setattr(script, "evaluate_job", _never_evaluate)
    assert _run(trained, "--phase", "evaluate") == script.EXIT_SELF_CHECK_FAILED
    assert "SELF-CHECK FAILED" in capsys.readouterr().out
    assert not (trained.ladder / LADDER).exists()


def test_a_split_that_is_not_the_references_is_exit_12_and_a_protocol_flag_is_exit_14(trained, monkeypatch, capsys):
    monkeypatch.setattr(script, "evaluate_job", _never_evaluate)
    assert _run(trained, "--phase", "evaluate", "--context", "3") == script.EXIT_RECORD_MISMATCH
    assert "RECORD MISMATCH" in capsys.readouterr().out
    _doctor(trained.out / RECORD, lambda r: r["episodes"].__setitem__("val", ["ep_000099_len00040.npz"]))
    assert _run(trained, "--phase", "evaluate") == script.EXIT_SPLIT_MISMATCH
    assert "SPLIT MISMATCH" in capsys.readouterr().out


def test_a_rung_checkpoint_labelled_with_another_step_is_exit_33(trained, capsys):
    path = trained.ladder / "step2" / CHECKPOINT
    payload = torch.load(path, weights_only=True)
    payload["step"] = 4
    torch.save(payload, path)
    assert _run(trained, "--phase", "evaluate") == script.EXIT_RUNG_MISLABELLED
    out = capsys.readouterr().out
    assert "RUNG MISLABELLED" in out and "step=4" in out and "step=2" in out
    assert not (trained.ladder / LADDER).exists()


def test_a_missing_rung_checkpoint_is_exit_11(trained, capsys):
    (trained.ladder / "step4" / CHECKPOINT).unlink()
    assert _run(trained, "--phase", "evaluate") == script.EXIT_NO_CHECKPOINTS
    assert "NO CELL" in capsys.readouterr().out


def test_a_hard_evaluate_refuses_a_train_record_whose_anchor_is_not_zero_before_anything_runs(trained, monkeypatch, capsys):
    _doctor(trained.ladder / TRAIN, lambda t: t["anchor"].update({"max_delta": 0.5, "first_step": 3}))
    monkeypatch.setattr(script, "evaluate_job", _never_evaluate)
    monkeypatch.setattr(script, "prepare_cell", _never_evaluate)
    assert _run(trained, "--phase", "evaluate", "--anchor", "hard") == script.EXIT_ANCHOR_MISMATCH
    assert "first at step 3" in capsys.readouterr().out


def test_all_trains_then_evaluates_each_cell(reference, capsys):
    assert _run(reference, "--phase", "all", "--anchor", "hard") == script.EXIT_OK
    assert (reference.ladder / TRAIN).is_file() and (reference.ladder / LADDER).is_file()


# ---------------------------------------------------------------------------
# The helpers, pure.
# ---------------------------------------------------------------------------


def test_history_from_train_record_round_trips_what_history_at_needs():
    history = {
        "arm": "random_vit", "steps": 4, "loss": [4.0, 3.0, 2.0, 1.0],
        "parts": [{"embedding": e, "reward": 0.0, "continue": 0.0, "kl_dyn": 0.3, "kl_rep": 0.3}
                  for e in (0.4, 0.3, 0.2, 0.1)],
        "seconds": 8.0, "kl_dyn_max": 0.3, "kl_rate_above_free_bits": 1.0,
        "checkpoint_seconds": {2: 3.0, 4: 8.0},
    }
    reference = types.SimpleNamespace(record={"seq_len": 4, "git_sha": "abc"})
    record = script.train_record("random_vit", 1, history, (2, 4), 0.0, None,
                                 steps=4, policy="hard", reference=reference, device="cpu")
    assert record["checkpoint_seconds"] == {"2": 3.0, "4": 8.0} and record["seq_len"] == 4
    back = script.history_from_train_record(json.loads(json.dumps(record)))
    assert history_at(back, 2) == history_at(history, 2)
    assert history_at(back, 4)["seconds"] == 8.0


def test_smoothed_at_is_the_window_mean_ending_at_the_step():
    parts = [{"embedding": v} for v in (1.0, 2.0, 4.0, 8.0)]
    assert script.smoothed_at(parts, "embedding", 4, 2) == 6.0
    assert script.smoothed_at(parts, "embedding", 3, 2) == 3.0
    assert script.smoothed_at(parts, "embedding", 1, 100) == 1.0, "the window shrinks to the prefix"
    with pytest.raises(ValueError):
        script.smoothed_at(parts, "embedding", 0, 2)


def test_rung_cell_carries_the_rung_records_own_protocol_in_the_diagnostics_slot(evaluated):
    cell = script.rung_cell(evaluated.ladder / "step2", JOB.arm, JOB.seed)
    assert cell.diagnostic == {"context": CONTEXT, "horizon": HORIZON}
    assert cell.record["steps"] == 2 and cell.checkpoint.is_file()
    with pytest.raises(script.CellMissing, match="no checkpoint"):
        script.rung_cell(evaluated.ladder / "step3", JOB.arm, JOB.seed)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_checkpoint_ladder_script.py -q`
Expected: FAIL — `FileNotFoundError` from `_load_script("checkpoint_ladder")` (every test errors until the script exists).

- [ ] **Step 3: Implement**

Create `scripts/checkpoint_ladder.py`:

```python
"""The checkpoint ladder over the M3c cells: retrain each cell to 5,000 steps saving five rungs, evaluate every rung and the 20,000-step reference, then pool and decide Reading T.

M3e ruled memorisation out and read, off the study records' learning curves,
that the ViT arms' one-step `embedding` loss reaches its minimum at step
1,610-4,535 of 20,000 and rises afterwards. Every rollout number M3b-M3e
recorded was measured at step 20,000. This script asks whether the model at
its own loss minimum rolls out better (spec 2026-09-18 M3f):

  train     retrain each requested cell from the study's seed and config to
            --steps (5,000), saving a labelled checkpoint into step{N}/ at
            each rung -- a study directory in the study's layout -- and
            anchor the retrain to the reference run: its per-step losses
            against the reference record's history.loss, exact.
  evaluate  per cell: the REFERENCE first (prepare_cell on --reference, the
            bitwise self-check against its diagnostic, the trust pass, the
            validation objective), then every rung: the study's own
            evaluation half (`evaluate_job`) writes step{N}/result_*.json,
            prepare_cell reads it back, the trust pass, the objective. One
            ladder_<arm>_seed<n>.json per cell holding all six rungs.
  read      pool the ladder records, decide Reading T, print the tables,
            write ladder.txt and ladder_curves.png (Task 6).

LOADING IS `trust_horizon.py`'S: `Cell`, `load_cell`, `self_check`,
`prepare_cell` and the checkpoint path are imported by path, and the trust
pass is `split_gap.py`'s `stratum_summary`. A rung's `Cell` carries the rung
record's own protocol in the diagnostic's slot -- a rung has no diagnostic,
and that is all `prepare_cell` reads there.

THE CHECKS, BY PHASE, each with its own status:

  train:    EXIT_NO_CHECKPOINTS (11)     a requested reference cell lacks its
                                          checkpoint, record or diagnostic;
                                          judged for every cell before any trains.
            EXIT_ANCHOR_MISMATCH (32)     NEW. --anchor hard and the retrain's
                                          losses are not the reference's.
  evaluate: EXIT_NO_CHECKPOINTS (11)     the reference cell, the train record
                                          or a rung checkpoint is missing.
            EXIT_ANCHOR_MISMATCH (32)     --anchor hard and the train record's
                                          anchor is not 0.0 (a `train` run under
                                          `report` cannot be read under `hard`).
            EXIT_SPLIT_MISMATCH (12)      the split by name is not the record's.
            EXIT_RECORD_MISMATCH (14)     --context/--horizon disagree with the
                                          protocol, or evaluate_rollout no longer
                                          reproduces the record's curve.
            EXIT_SELF_CHECK_FAILED (30)   the reference's trust pass is not
                                          bitwise its diagnostic -- judged BEFORE
                                          any rung of the cell is evaluated.
            EXIT_RUNG_MISLABELLED (33)    NEW. a rung checkpoint's arm, seed or
                                          step is not the rung's.

11 / 12 / 14 / 30 carry `trust_horizon.py`'s meanings on purpose; 32 and 33
are in no other tool's range (run_study 1/3-6/23, report_study 7-10, spike
10, diagnose 11-17, pool 18-22, trust 30, split_gap 31, argparse 2, a
traceback 1).
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.aggregate import SEEDS
from mbfps.eval.diagnostics import reference_trajectories
from mbfps.eval.ladder import (
    CURVE_WINDOW,
    OBJECTIVE_BATCHES,
    REFERENCE_RUNG,
    RUNGS,
    STEPS,
    anchor_delta,
    primary_rung,
)
from mbfps.eval.objective import val_objective
from mbfps.eval.split_gap import learning_curve_summary, stratum_summary
from mbfps.eval.study import (
    SPLIT_SEED,
    StudyJob,
    evaluate_job,
    git_sha,
    history_record,
    job_record_path,
    load_record,
    write_record,
)
from mbfps.training.world_model import history_at, train_world_model
from mbfps.utils.config import ARMS, get_config
from mbfps.utils.device import get_device


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_checkpoint_ladder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_trust = _sibling("trust_horizon")
Cell = _trust.Cell
CellMissing = _trust.CellMissing
load_cell = _trust.load_cell
self_check = _trust.self_check
prepare_cell = _trust.prepare_cell
probe_is_measurable = _trust.probe_is_measurable
checkpoint_path = _trust.checkpoint_path

EXIT_OK = _trust.EXIT_OK
EXIT_NO_CHECKPOINTS = _trust.EXIT_NO_CHECKPOINTS
EXIT_SPLIT_MISMATCH = _trust.EXIT_SPLIT_MISMATCH
EXIT_RECORD_MISMATCH = _trust.EXIT_RECORD_MISMATCH
EXIT_SELF_CHECK_FAILED = _trust.EXIT_SELF_CHECK_FAILED
EXIT_ANCHOR_MISMATCH = 32
EXIT_RUNG_MISLABELLED = 33
"""0 / 11 / 12 / 14 / 30 are `trust_horizon.py`'s, imported so they cannot
drift; 32 and 33 are new and in no other tool's range."""

PHASES: tuple[str, ...] = ("train", "evaluate", "all")
ANCHOR_POLICIES: tuple[str, ...] = ("hard", "report")
ANCHOR_DEFAULT: str = "report"
"""The smoke run's measurement pins this (spec 2.4, Task 7); until then the
default is the policy that cannot stop a run."""


# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------


def rung_dir(out_dir: Path, step: int) -> Path:
    """`step{N}/` under `--out`: a study directory, holding this rung's
    checkpoint and its study record under the study's own file names."""
    return Path(out_dir) / f"step{int(step)}"


def train_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    return Path(out_dir) / f"train_{arm}_seed{seed}.json"


def ladder_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    return Path(out_dir) / f"ladder_{arm}_seed{seed}.json"


# ---------------------------------------------------------------------------
# train: one cell retrained, its rungs saved, its anchor measured.
# ---------------------------------------------------------------------------


def train_record(arm: str, seed: int, history: dict, rungs, delta: float, first: int | None, *,
                 steps: int, policy: str, reference: Cell, device) -> dict:
    """The LIVE train record: the whole retrain history (so `evaluate` can
    slice it per rung without retraining), the elapsed seconds at each rung,
    the anchor, and provenance."""
    return {
        "arm": arm,
        "seed": int(seed),
        "steps": int(steps),
        "rungs": [int(r) for r in rungs],
        "seq_len": int(reference.record["seq_len"]),
        "history": history_record(history),
        "checkpoint_seconds": {str(k): float(v) for k, v in history["checkpoint_seconds"].items()},
        "seconds": float(history["seconds"]),
        "kl_dyn_max": float(history["kl_dyn_max"]),
        "kl_rate_above_free_bits": float(history["kl_rate_above_free_bits"]),
        "anchor": {
            "policy": policy,
            "steps": int(steps),
            "max_delta": float(delta),
            "first_step": None if first is None else int(first),
            "reference_git_sha": str(reference.record.get("git_sha", "unknown")),
        },
        "git_sha": git_sha(),
        "device": str(device),
        "torch_version": torch.__version__,
    }


def history_from_train_record(record: dict) -> dict:
    """The trainer's history dict, rebuilt from a train record, in the shape
    `history_at` reads (JSON turned the `checkpoint_seconds` keys into str)."""
    return {
        "arm": record["arm"],
        "steps": int(record["steps"]),
        "loss": [float(v) for v in record["history"]["loss"]],
        "parts": [dict(p) for p in record["history"]["parts"]],
        "seconds": float(record["seconds"]),
        "kl_dyn_max": float(record["kl_dyn_max"]),
        "kl_rate_above_free_bits": float(record["kl_rate_above_free_bits"]),
        "checkpoint_seconds": {int(k): float(v) for k, v in record["checkpoint_seconds"].items()},
    }


def anchor_failure(arm: str, seed: int, delta: float, first, steps: int) -> str:
    return (
        f"\nANCHOR MISMATCH for {arm} seed {seed}: the retrain's per-step loss is not the "
        f"reference record's (max abs {delta:.3e}, first at step {first} of {steps}). Under "
        "--anchor hard the rungs must be states the original run passed through, and they "
        "are not. No rung is evaluated."
    )


def train_cell(args, buffer, arm: str, seed: int, reference: Cell, device, rungs) -> tuple[int, dict | None]:
    """Retrain one cell from the study's seed and configuration -- `seq_len`
    the reference record's, `steps` the ladder's -- saving a rung at each of
    `rungs`, then anchor it. The train record is written whatever the anchor
    says: it is the evidence."""
    cfg = get_config(
        arm, steps=args.steps, seq_len=int(reference.record["seq_len"]), seed=seed, device=args.device,
    )
    history = train_world_model(cfg, buffer, out_dir=args.out, checkpoint_steps=tuple(rungs))
    delta, first = anchor_delta(history["loss"], reference.record["history"]["loss"], args.steps)
    record = train_record(
        arm, seed, history, rungs, delta, first,
        steps=args.steps, policy=args.anchor, reference=reference, device=device,
    )
    path = train_record_path(args.out, arm, seed)
    write_record(path, record)
    where = f" (first at step {first})" if first is not None else ""
    print(
        f"{arm} seed {seed}: trained {args.steps} steps in {history['seconds']:.0f}s, rungs "
        f"{list(rungs)}; anchor max|delta| {delta:.1e}{where}; wrote {path}"
    )
    if args.anchor == "hard" and delta != 0.0:
        print(anchor_failure(arm, seed, delta, first, args.steps))
        return EXIT_ANCHOR_MISMATCH, None
    return EXIT_OK, record


# ---------------------------------------------------------------------------
# evaluate: the reference first, then every rung.
# ---------------------------------------------------------------------------


def rung_checkpoint_label(path: Path) -> tuple:
    """`(arm, seed, step)` as the checkpoint labels itself."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload.get("arm"), payload.get("seed"), payload.get("step")


def rung_cell(directory: Path, arm: str, seed: int) -> Cell:
    """A rung's `Cell`: its checkpoint and study record, and in the
    diagnostic's slot -- a rung has no diagnostic -- the record's own
    protocol, which is all `prepare_cell` reads there (`protocol_mismatch`).
    `self_check` is never run on a rung."""
    directory = Path(directory)
    checkpoint = checkpoint_path(directory, arm, seed)
    record_path = job_record_path(directory, StudyJob(arm=arm, seed=seed))
    for kind, path in (("checkpoint", checkpoint), ("study record", record_path)):
        if not path.exists():
            raise CellMissing(f"{arm} seed {seed}: no {kind} at {path}")
    record = load_record(record_path)
    return Cell(
        arm=arm, seed=seed, checkpoint=checkpoint, record=record,
        diagnostic={"context": int(record["context"]), "horizon": int(record["horizon"])},
    )


def _cell_args(args, out_dir: Path) -> types.SimpleNamespace:
    """`prepare_cell` reads `out`, `device`, `context` and `horizon` off its
    args; each rung is read with `out` pointed at its own directory."""
    return types.SimpleNamespace(
        out=Path(out_dir), device=args.device, context=args.context, horizon=args.horizon,
    )


def smoothed_at(parts, term: str, step: int, window: int) -> float:
    """The `window`-step moving mean of `term` ending at 1-based `step` (the
    window shrinks to the prefix) -- the training-side number printed beside
    the validation objective at that rung."""
    values = np.asarray([float(p[term]) for p in list(parts)[:int(step)]], dtype=float)
    if values.size == 0:
        raise ValueError(f"no training steps at or before step {step}")
    w = min(int(window), values.size)
    return float(values[-w:].mean())


def rung_entry(step: int, record: dict, traj, prepared, objective: dict, train_parts, window: int,
               check=None) -> dict:
    """One rung of one cell: the gate's own metric off the study record, the
    probe's measurability, the validation objective, the smoothed training
    embedding at that step, the reference's self-check (None on a rung), and
    the trust pass reduced by `stratum_summary`."""
    curves = record["curves"]
    r2 = record["probe"]["embedding_selection_r2"]
    return {
        "step": int(step),
        "record_git_sha": str(record.get("git_sha", "unknown")),
        "gate": {
            "gap_final": float(record["position"]["gap_final"]),
            "steps_degenerate": int(record["position"]["steps_degenerate"]),
        },
        "probe": {
            "selection_r2": float(r2) if r2 is not None else float("nan"),
            "measurable": bool(probe_is_measurable(
                {"persistence": curves["persistence_position"][-1], "floor": curves["floor_position"][-1]},
                widest_se=0.0,
            )),
        },
        "objective": {name: float(value) for name, value in objective.items()},
        "train_embedding": smoothed_at(train_parts, "embedding", step, window),
        "self_check": None if check is None else check.record(),
        "summary": stratum_summary(traj, prepared.horizon),
    }


def evaluate_reference(args, reference: Cell, device, train, val, buffer) -> tuple[int, dict | None, tuple]:
    """Rung 20000: `prepare_cell` on `--reference` (12, 14), the trust pass,
    the bitwise self-check against the diagnostic (30), the objective.
    Returns the entry and the resolved `(context, horizon)` every rung is
    evaluated at."""
    status, prepared = prepare_cell(_cell_args(args, args.reference), reference, device, train, val)
    if status != EXIT_OK:
        return status, None, ()
    traj = reference_trajectories(prepared.model, val, prepared.embedding_probe, **prepared.common)
    check = self_check(traj, reference.diagnostic)
    if not check.ok:
        print(
            f"\nSELF-CHECK FAILED for {reference.arm} seed {reference.seed} at the reference "
            f"(step {REFERENCE_RUNG}): " + "; ".join(check.failures())
            + ". The ruler M3d validated does not reproduce, so no rung is evaluated against it."
        )
        return EXIT_SELF_CHECK_FAILED, None, ()
    cfg = get_config(
        reference.arm, seq_len=int(reference.record["seq_len"]), seed=reference.seed, device=args.device,
    )
    objective = val_objective(
        prepared.model, buffer, val, cfg, device, batches=args.objective_batches, seed=reference.seed,
    )
    entry = rung_entry(
        REFERENCE_RUNG, reference.record, traj, prepared, objective,
        reference.record["history"]["parts"], args.window, check=check,
    )
    return EXIT_OK, entry, (prepared.context, prepared.horizon)


def evaluate_rung(args, buffer, arm: str, seed: int, step: int, history: dict, device, train, val, *,
                  context: int, horizon: int, seq_len: int) -> tuple[int, dict | None]:
    """One rung: the checkpoint's label (11, 33), `evaluate_job` into the rung
    directory, `prepare_cell` on it (12, 14 -- proving the two rollouts
    agree), the trust pass, the objective."""
    directory = rung_dir(args.out, step)
    checkpoint = checkpoint_path(directory, arm, seed)
    if not checkpoint.exists():
        print(f"NO CELL: {arm} seed {seed}: no checkpoint at {checkpoint}")
        return EXIT_NO_CHECKPOINTS, None
    label = rung_checkpoint_label(checkpoint)
    if label != (arm, int(seed), int(step)):
        print(
            f"\nRUNG MISLABELLED: {checkpoint} is arm={label[0]!r} seed={label[1]!r} "
            f"step={label[2]!r}, not this rung's arm={arm!r} seed={seed!r} step={step!r}; it "
            "would be reported under a step it was never trained to."
        )
        return EXIT_RUNG_MISLABELLED, None
    evaluate_job(
        StudyJob(arm=arm, seed=seed), buffer, directory, history=history_at(history, step),
        steps=int(step), seq_len=seq_len, context=context, horizon=horizon, device=args.device,
    )
    cell = rung_cell(directory, arm, seed)
    status, prepared = prepare_cell(_cell_args(args, directory), cell, device, train, val)
    if status != EXIT_OK:
        return status, None
    traj = reference_trajectories(prepared.model, val, prepared.embedding_probe, **prepared.common)
    cfg = get_config(arm, seq_len=seq_len, seed=seed, device=args.device)
    objective = val_objective(
        prepared.model, buffer, val, cfg, device, batches=args.objective_batches, seed=seed,
    )
    return EXIT_OK, rung_entry(step, cell.record, traj, prepared, objective, history["parts"], args.window)


def ladder_record(arm: str, seed: int, entries: dict, *, primary: int, min_step: int, train_rec: dict,
                  args, device, val, rungs, context: int, horizon: int) -> dict:
    """The LIVE ladder record for one cell: every rung's entry under its step,
    the primary rung and the minimum it was read from, and provenance."""
    return {
        "arm": arm,
        "seed": int(seed),
        "context": int(context),
        "horizon": int(horizon),
        "split_seed": SPLIT_SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "git_sha": git_sha(),
        "reference_git_sha": entries[REFERENCE_RUNG]["record_git_sha"],
        "train_git_sha": str(train_rec["git_sha"]),
        "anchor": dict(train_rec["anchor"]),
        "rungs": [int(r) for r in rungs],
        "reference_rung": REFERENCE_RUNG,
        "primary_rung": int(primary),
        "embedding_min_step": int(min_step),
        "curve_window": int(args.window),
        "objective_batches": int(args.objective_batches),
        "episodes": {"val": [p.name for p in val]},
        "entries": {str(step): entries[step] for step in sorted(entries)},
    }


def evaluate_cell(args, buffer, arm: str, seed: int, device, train, val, rungs) -> tuple[int, dict | None]:
    """One cell: 11 (reference, train record), 32 under `hard`, the reference
    (12, 14, 30) BEFORE any rung, then each rung (11, 33, 12, 14), then the
    primary rung off the reference's history, and the ladder record."""
    try:
        reference = load_cell(args.reference, arm, seed)
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS, None
    train_path = train_record_path(args.out, arm, seed)
    if not train_path.exists():
        print(f"NO CELL: {arm} seed {seed}: no train record at {train_path}; run --phase train first")
        return EXIT_NO_CHECKPOINTS, None
    train_rec = load_record(train_path)
    anchor = train_rec["anchor"]
    if args.anchor == "hard" and float(anchor["max_delta"]) != 0.0:
        print(anchor_failure(arm, seed, float(anchor["max_delta"]), anchor["first_step"], int(anchor["steps"])))
        return EXIT_ANCHOR_MISMATCH, None

    # The reference FIRST: its self-check is the ruler every rung is read against.
    status, ref_entry, protocol = evaluate_reference(args, reference, device, train, val, buffer)
    if status != EXIT_OK:
        return status, None
    context, horizon = protocol
    seq_len = int(reference.record["seq_len"])
    history = history_from_train_record(train_rec)
    entries = {REFERENCE_RUNG: ref_entry}
    for step in rungs:
        status, entry = evaluate_rung(
            args, buffer, arm, seed, int(step), history, device, train, val,
            context=context, horizon=horizon, seq_len=seq_len,
        )
        if status != EXIT_OK:
            return status, None
        entries[int(step)] = entry

    min_step = learning_curve_summary(reference.record["history"], window=args.window)["terms"]["embedding"]["smoothed_min_step"]
    primary = primary_rung(min_step, rungs)
    record = ladder_record(
        arm, seed, entries, primary=primary, min_step=min_step, train_rec=train_rec, args=args,
        device=device, val=val, rungs=rungs, context=context, horizon=horizon,
    )
    path = ladder_record_path(args.out, arm, seed)
    write_record(path, record)
    check = ref_entry["self_check"]
    print(
        f"{arm} seed {seed}: reference self-check max|delta| reference "
        f"{check['reference_position_max_delta']:.1e} persistence "
        f"{check['persistence_position_max_delta']:.1e}; {len(rungs)} rungs evaluated; "
        f"embedding minimum at step {min_step} -> primary rung {primary}; wrote {path}"
    )
    return EXIT_OK, record


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/m3f_ladder"))
    parser.add_argument("--reference", type=Path, default=Path("runs/m3_study_v2"),
                        help="the M3c study directory: rung 20000, and the anchor's history")
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--device", default="mps")
    # None -> the reference diagnostic says what it was written at; a value
    # that disagrees is refused (`protocol_mismatch`, 14).
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--steps", type=int, default=STEPS, help="the retrain's length")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--anchor", choices=ANCHOR_POLICIES, default=ANCHOR_DEFAULT,
                        help="hard: refuse a retrain whose losses are not the reference's (32); "
                             "report: print the delta and continue")
    parser.add_argument("--objective-batches", type=int, default=OBJECTIVE_BATCHES,
                        help="validation draws per rung for the objective")
    parser.add_argument("--window", type=int, default=CURVE_WINDOW,
                        help="moving-mean window (steps) the embedding minimum is read at")
    parser.add_argument("--figure", type=Path, default=None,
                        help="the curves figure; default <out>/ladder_curves.png")
    return parser


def main(argv: list[str] | None = None, *, rungs=RUNGS) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    rungs = tuple(sorted({int(r) for r in rungs}))
    if args.steps < max(rungs):
        parser.error(f"--steps {args.steps} is below the last rung {max(rungs)}")
    device = get_device(prefer=args.device)
    cells = [(arm, int(seed)) for arm in args.arms for seed in args.seeds]

    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    train, val = episode_split(buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED)

    references: dict[tuple[str, int], Cell] = {}
    if args.phase in ("train", "all"):
        # Every requested reference cell is loaded before anything trains.
        try:
            references = {(arm, seed): load_cell(args.reference, arm, seed) for arm, seed in cells}
        except CellMissing as error:
            print(f"NO CELL: {error}")
            return EXIT_NO_CHECKPOINTS
        args.out.mkdir(parents=True, exist_ok=True)

    for arm, seed in cells:
        if args.phase in ("train", "all"):
            status, _ = train_cell(args, buffer, arm, seed, references[(arm, seed)], device, rungs)
            if status != EXIT_OK:
                return status
        if args.phase in ("evaluate", "all"):
            status, _ = evaluate_cell(args, buffer, arm, seed, device, train, val, rungs)
            if status != EXIT_OK:
                return status
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_checkpoint_ladder_script.py -q`
Expected: 19 passed (about 2–3 minutes: the reference fixture trains and diagnoses a real cell per test).

If `test_train_writes_the_rungs_..._anchored_at_zero` fails on the loss equality alone, the retrain is not reproducing the reference on CPU: report it as BLOCKED with the two loss lists — that is the anchor's premise (Task 1's `test_a_rung_is_a_state_the_longer_run_passed_through` pins it at the trainer) and the plan needs the human, not a workaround.

- [ ] **Step 5: Run the sibling script tests, which load `trust_horizon` and `split_gap` the same way**

Run: `.venv/bin/python -m pytest tests/eval/test_trust_horizon_script.py tests/eval/test_split_gap_script.py -q`
Expected: all pass (nothing in either script changed).

- [ ] **Step 6: Commit**

```bash
git add scripts/checkpoint_ladder.py tests/eval/test_checkpoint_ladder_script.py
git commit -m "feat: checkpoint_ladder.py part 1 -- retrain a cell saving labelled rungs, anchor it to the reference run, evaluate the reference (self-checked first) and every rung through evaluate_job and the trust pass, one ladder record per cell

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: `scripts/checkpoint_ladder.py` part 2 — the `read` phase: pooling, Reading T, the tables, the figure, `ladder.txt`

**Files:**
- Modify: `scripts/checkpoint_ladder.py` (add the `read` phase; `PHASES` gains `"read"`; `main` builds the buffer only for `train`/`evaluate`)
- Modify: `tests/eval/test_checkpoint_ladder_script.py` (append), `tests/eval/test_diagnose_dynamics_script.py` (the distinctness test gains `checkpoint_ladder`)

**Interfaces:**
- Consumes: the ladder record (Task 5); `pooling.paired_contrast`, `pooling.cluster_threshold`; `ladder.ArmInputs/TimingInputs/reading_timing/format_reading_timing/gate_passes`, `DECISION_H`, `REPORTED_H`, `FAMILY`, `R2_SENSITIVITY`; `split_gap.CHANNELS`, `TERMS`, `StratumContrast`, `q_key`, and Task 4b's `cell_series`, `survival_indicator`, `margin_at`, `decision_horizon`; `trust.survival/trust_horizon`; `trust_horizon.py`'s `unmoved_fraction`, `conditional_survival` (by path); `split_gap.py`'s `SURVIVAL_STEPS` (by path).
- Produces: `survival_series(record, step, channel, h, label)`, `margin_series(...)` (thin wrappers over Task 4b's helpers), `arm_inputs(records, arm, seeds, *, h, series, min_r2) -> ArmInputs`, `timing_inputs(records, *, arms, seeds, h, series, min_r2) -> TimingInputs`, `readings_text(...) -> str`, `write_curves(records, figure, rungs) -> str`, `write_readings(out_dir, text) -> Path` (`ladder.txt`), `read_phase(args, cells, rungs) -> int`; `--phase read`. `decision_horizon` is imported from `split_gap`, not redefined.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_checkpoint_ladder_script.py`:

```python
# ---------------------------------------------------------------------------
# read (Task 6): pooling glue on fabricated records, then the fixture run.
# ---------------------------------------------------------------------------

from mbfps.eval.ladder import DECISION_H  # noqa: E402


def _fake_entry(step: int, crossing, *, windows: int, horizon: int, measurable: bool, r2: float) -> dict:
    crossing = np.asarray(crossing, dtype=float)
    moved = np.ones((windows, horizon), dtype=bool)
    margin = np.tile(np.arange(1, horizon + 1, dtype=float), (windows, 1))
    return {
        "step": step, "record_git_sha": "ref",
        "gate": {"gap_final": 0.1, "steps_degenerate": 0},
        "probe": {"selection_r2": r2, "measurable": measurable},
        "objective": {k: 0.1 for k in OBJECTIVE_KEYS},
        "train_embedding": 0.1,
        "self_check": ({"reference_position_max_delta": 0.0, "persistence_position_max_delta": 0.0,
                        "windows_total_match": True, "windows_episode_match": True, "ok": True}
                       if step == REFERENCE_RUNG else None),
        "summary": {
            "windows": {"total": windows, "episode": [0, 0, 1, 1][:windows], "clusters": 2},
            "crossing": {"free": crossing.tolist(), "probe": crossing.tolist()},
            "margin": {"free": margin.tolist(), "probe": margin.tolist()},
            "moved": moved.tolist(),
            "first_moved": [1.0] * windows,
            "survival": {"free": [1.0] * (horizon + 1), "probe": [1.0] * (horizon + 1)},
            "trust_horizon": {"free": {"q50": 1, "q75": 1, "q90": 0}, "probe": {"q50": 1, "q75": 1, "q90": 0}},
            "counts": {"not_moved": [0] * horizon, "never_moved": 0},
        },
    }


def _fake_record(seed: int, *, primary: int, crossing_primary, crossing_reference,
                 measurable_reference: bool = True, r2: float = 0.5) -> dict:
    kw = dict(windows=4, horizon=3, r2=r2)
    entries = {
        "2": _fake_entry(2, crossing_primary if primary == 2 else [1, 1, 1, 1], measurable=True, **kw),
        "4": _fake_entry(4, crossing_primary if primary == 4 else [1, 1, 1, 1], measurable=True, **kw),
        str(REFERENCE_RUNG): _fake_entry(REFERENCE_RUNG, crossing_reference, measurable=measurable_reference, **kw),
    }
    return {
        "arm": "random_vit", "seed": seed, "context": 2, "horizon": 3, "device": "cpu",
        "torch_version": "t", "episodes": {"val": ["a", "b"]}, "primary_rung": primary,
        "rungs": [2, 4], "entries": entries,
    }


def test_survival_series_is_the_indicator_over_the_windows_that_ever_moved():
    r = _fake_record(0, primary=2, crossing_primary=[3, 3, float("nan"), 1], crossing_reference=[1, 1, 1, 1])
    s = script.survival_series(r, 2, "free", 2, "random_vit@primary")
    assert s.arm == "random_vit@primary" and s.rung == "val" and s.channel == "S/free"
    assert s.delta.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert s.changed.tolist() == [True, True, False, True]
    assert s.episode.tolist() == [0, 0, 1, 1] and s.windows_total == 4


def test_margin_series_masks_by_moved_at_h_not_ever_moved():
    r = _fake_record(0, primary=2, crossing_primary=[3, 3, 3, 3], crossing_reference=[1, 1, 1, 1])
    r["entries"]["2"]["summary"]["moved"][1][1] = False
    s = script.margin_series(r, 2, "probe", 2, "random_vit@primary")
    assert s.channel == "margin/probe" and s.delta.tolist() == [2.0, 2.0, 2.0, 2.0]
    assert s.changed.tolist() == [True, False, True, True]


def test_arm_inputs_pairs_each_seeds_primary_rung_against_its_reference_on_the_same_windows():
    """Seed 0's primary is rung 2, seed 1's is rung 4; at h=2 the indicator is
    `crossing > 2`. Treatment seed-mean [1, .5, .5, 0] against control [0, 0,
    0, 0]: the paired mean is 0.5, over 2 episode clusters."""
    records = {
        ("random_vit", 0): _fake_record(0, primary=2, crossing_primary=[3, 3, 1, 1], crossing_reference=[1, 1, 1, 1]),
        ("random_vit", 1): _fake_record(1, primary=4, crossing_primary=[3, 1, 3, 1], crossing_reference=[1, 1, 1, 1]),
    }
    a = script.arm_inputs(records, "random_vit", [0, 1], h=2)
    assert a.primary_rungs == {0: 2, 1: 4}
    assert a.t_free.estimate == pytest.approx(0.5) and a.t_free.clusters == 2
    assert a.t_probe.estimate == pytest.approx(0.5)
    assert a.per_seed[0].t_free.estimate == pytest.approx(0.5) and a.per_seed[0].per_seed is None
    assert a.per_seed[1].t_free.estimate == pytest.approx(0.5) and a.per_seed[1].primary_rungs == {1: 4}


def test_a_cell_unmeasurable_at_either_rung_leaves_the_probe_channel_only():
    records = {
        ("random_vit", 0): _fake_record(0, primary=2, crossing_primary=[3, 3, 3, 3], crossing_reference=[1, 1, 1, 1],
                                        measurable_reference=False),
    }
    a = script.arm_inputs(records, "random_vit", [0], h=2)
    assert a.t_free.estimate == pytest.approx(1.0)
    assert np.isnan(a.t_probe.estimate) and a.t_probe.clusters == 0
    # The sensitivity line drops a low-R2 cell the same way.
    low = _fake_record(0, primary=2, crossing_primary=[3, 3, 3, 3], crossing_reference=[1, 1, 1, 1], r2=0.05)
    b = script.arm_inputs({("random_vit", 0): low}, "random_vit", [0], h=2, min_r2=0.1)
    assert np.isnan(b.t_probe.estimate) and b.t_free.estimate == pytest.approx(1.0)


def test_timing_inputs_reads_z_fam_off_the_val_clusters_and_orders_the_arms():
    records = {
        ("random_vit", 0): _fake_record(0, primary=2, crossing_primary=[3, 3, 3, 3], crossing_reference=[1, 1, 1, 1]),
    }
    inputs = script.timing_inputs(records, arms=["random_vit"], seeds=[0], h=2)
    assert inputs.h == 2 and list(inputs.arms) == ["random_vit"]
    assert inputs.z_fam == pytest.approx(script.pooling.cluster_threshold(6, 2))


def test_decision_horizon_is_fifteen_unless_the_run_is_shorter():
    assert script.decision_horizon(45) == (DECISION_H, False)
    assert script.decision_horizon(3) == (3, True)


@pytest.fixture
def ladder_run(evaluated, capsys):
    status = _run(evaluated, "--phase", "read")
    out = capsys.readouterr().out
    assert status == script.EXIT_OK, out
    return types.SimpleNamespace(cell=evaluated, out=out, text=(evaluated.ladder / "ladder.txt").read_text())


def test_read_writes_ladder_txt_identical_to_stdout(ladder_run):
    assert ladder_run.text == ladder_run.out
    assert (ladder_run.cell.ladder / "ladder_curves.png").is_file()


def test_read_prints_every_block_in_order(ladder_run):
    text = ladder_run.text
    headers = [
        "--- anchor:", "--- primary rung per cell:", "--- reference self-check",
        "--- the gate at every rung:", "--- survival per rung:", "--- conditional survival per rung:",
        "--- the validation objective per rung", "  pooling: z_fam = cluster_threshold(6, 6)",
        "--- Reading T:", "--- T at the reported horizons", "--- sensitivity", "--- per seed",
        "figure=",
    ]
    positions = [text.index(h) for h in headers]
    assert positions == sorted(positions), "the blocks are printed in the spec's order"
    assert "verdict: random_vit" in text and "decided by:" in text
    assert f"h={HORIZON}" in text and "(pre-registered DECISION_H = 15)" in text, \
        "the fixture's horizon is 3, so the decision horizon is clamped and says so"
    assert "GATE PASSES" in text or "gap_closed(3)" in text
    assert "20000" in text


def test_a_single_seed_run_reads_this_seed_alone_per_seed_and_zero_of_one_pooled(ladder_run):
    text = ladder_run.text
    assert "  random_vit  s1:" in text
    # With one seed the pooled reading can never replicate: whatever the
    # fixture's contrast is, the status is NO DIFFERENCE and the rule says why
    # if it cleared.
    line = next(l for l in text.splitlines() if l.startswith("  verdict: random_vit"))
    assert "NO DIFFERENCE" in line
    assert ("does not clear" in line) or ("only 1 of 1 seeds" in line)


def test_read_refuses_a_ladder_record_whose_self_check_is_not_ok(evaluated, capsys):
    _doctor(evaluated.ladder / LADDER, lambda r: r["entries"][str(REFERENCE_RUNG)]["self_check"].update({"ok": False}))
    assert _run(evaluated, "--phase", "read") == script.EXIT_SELF_CHECK_FAILED
    assert "SELF-CHECK FAILED" in capsys.readouterr().out
    assert not (evaluated.ladder / "ladder.txt").exists()


def test_read_without_a_ladder_record_is_exit_11(trained, capsys):
    assert _run(trained, "--phase", "read") == script.EXIT_NO_CHECKPOINTS
    assert "NO CELL" in capsys.readouterr().out


def test_write_curves_handles_a_missing_matplotlib_by_costing_the_figure_only(evaluated, monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def no_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    records = {(JOB.arm, JOB.seed): load_record(evaluated.ladder / LADDER)}
    line = script.write_curves(records, tmp_path / "curves.png", LADDER_RUNGS)
    assert line.startswith("figure NOT written:") and not (tmp_path / "curves.png").exists()
```

Then in `tests/eval/test_diagnose_dynamics_script.py`, append to the end of `test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own` (after the `split_gap` block):

```python
    ladder = statuses("checkpoint_ladder")
    reused_by_ladder = {**reused, "EXIT_SELF_CHECK_FAILED": 30}
    shared_with_trust = {
        name: value for name, value in ladder.items() if value in set(trust.values()) - {0}
    }
    assert shared_with_trust == reused_by_ladder, (
        f"checkpoint_ladder shares {shared_with_trust} with trust_horizon; only "
        f"{reused_by_ladder} is shared on purpose"
    )
    assert len(set(ladder.values())) == len(ladder), ladder
    assert 1 not in ladder.values() and 2 not in ladder.values()
    assert ladder["EXIT_ANCHOR_MISMATCH"] == 32 and ladder["EXIT_RUNG_MISLABELLED"] == 33
    own = set(ladder.values()) - set(reused_by_ladder.values()) - {0}
    assert own == {32, 33}
    for other in ("run_study", "report_study", "pool_dynamics", "diagnose_dynamics", "split_gap"):
        clash = own & set(statuses(other).values())
        assert not clash, f"checkpoint_ladder collides with {other} on {clash}"
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_checkpoint_ladder_script.py -q -k "series or arm_inputs or unmeasurable or timing_inputs or decision_horizon or read or curves"`
Expected: FAIL — `AttributeError: module has no attribute 'survival_series'` (and `--phase read` refused by argparse in the fixture-driven tests).

- [ ] **Step 3: Implement**

In `scripts/checkpoint_ladder.py`:

(a) Extend the imports:

```python
import mbfps.eval.pooling as pooling
from mbfps.eval.ladder import (
    CURVE_WINDOW,
    DECISION_H,
    FAMILY,
    OBJECTIVE_BATCHES,
    R2_SENSITIVITY,
    REFERENCE_RUNG,
    REPORTED_H,
    RUNGS,
    STEPS,
    ArmInputs,
    TimingInputs,
    anchor_delta,
    format_reading_timing,
    gate_passes,
    primary_rung,
    reading_timing,
)
from mbfps.eval.split_gap import (
    CHANNELS,
    TERMS,
    StratumContrast,
    cell_series,
    decision_horizon,
    learning_curve_summary,
    margin_at,
    q_key,
    stratum_summary,
    survival_indicator,
)
from mbfps.eval.trust import survival, trust_horizon
from mbfps.eval.trust_readings import ARMS_ORDER, Q_REPORTED
```

and, after the `_trust` block:

```python
_split = _sibling("split_gap")
SURVIVAL_STEPS = _split.SURVIVAL_STEPS
"""The survival table's columns, M3d's and M3e's, filtered to <= the run's horizon."""
```

(b) `PHASES = ("train", "evaluate", "read", "all")`.

(c) Append the `read` phase before `_parser`:

```python
# ---------------------------------------------------------------------------
# read: pooling glue -- ladder records -> the inputs Reading T is decided on.
# ---------------------------------------------------------------------------
# Every estimator is `pooling.py`'s. What is decided here is only WHICH
# series goes in under WHICH mask (spec 3.1): the survival indicator over the
# windows that moved within the horizon; the margin over the windows moved AT
# h. The rung contrast is the PAIRED one -- both rungs score the same val
# windows -- with the two rung groups labelled as two arms so
# `paired_contrast` reads them as such.


def _entry(record: dict, step: int) -> dict:
    return record["entries"][str(int(step))]


def _series(record: dict, step: int, channel: str, values, changed, label: str) -> pooling.CellSeries:
    """One per-window series of one cell at one rung, through
    `split_gap.cell_series`. `arm` is the GROUP label (`<arm>@primary` /
    `<arm>@20000`): the two rung groups of one arm are two "arms" on the same
    windows to `paired_contrast`, which refuses an arm against itself. `rung`
    is the stratum (val); `channel` the reading."""
    return cell_series(
        _entry(record, step)["summary"], values, changed,
        arm=label, seed=record["seed"], rung="val", channel=channel,
        val=record["episodes"]["val"], horizon=record["horizon"], context=record["context"],
        device=record["device"], torch_version=record["torch_version"],
    )


def survival_series(record: dict, step: int, channel: str, h: int, label: str) -> pooling.CellSeries:
    """`split_gap.survival_indicator` on this rung's summary: `1[h_x > h]`
    over the windows that moved within the horizon."""
    return _series(record, step, f"S/{channel}", *survival_indicator(_entry(record, step)["summary"], channel, h), label)


def margin_series(record: dict, step: int, channel: str, h: int, label: str) -> pooling.CellSeries:
    """`split_gap.margin_at` on this rung's summary: `Delta(h)` over the
    windows moved AT h."""
    return _series(record, step, f"margin/{channel}", *margin_at(_entry(record, step)["summary"], channel, h), label)


_NO_CONTRAST = StratumContrast(estimate=float("nan"), se=float("nan"), z=float("nan"), clusters=0)


def _paired(treatment, control) -> StratumContrast:
    """`pooling.paired_contrast`, or the NaN contrast when there is nothing to
    pair -- no cell, or no window every cell of both groups changed."""
    if not treatment or not control:
        return _NO_CONTRAST
    if not np.logical_and.reduce([c.changed for c in list(treatment) + list(control)]).any():
        return _NO_CONTRAST
    c = pooling.paired_contrast(treatment, control)
    return StratumContrast(estimate=c.mean, se=c.se, z=c.z, clusters=c.clusters)


def _probe_ok(record: dict, step: int, min_r2: float | None) -> bool:
    entry = _entry(record, step)
    if not entry["probe"]["measurable"]:
        return False
    if min_r2 is None:
        return True
    r2 = float(entry["probe"]["selection_r2"])
    return bool(np.isfinite(r2) and r2 >= min_r2)


def _included(record: dict, channel: str, min_r2: float | None) -> bool:
    """Spec 3.1: the free channel pools every cell; the probe channel the
    cells measurable (and, on the sensitivity line, at or above `min_r2`) at
    BOTH rungs of the pair."""
    if channel == "free":
        return True
    return _probe_ok(record, record["primary_rung"], min_r2) and _probe_ok(record, REFERENCE_RUNG, min_r2)


def arm_inputs(records: dict, arm: str, seeds, *, h: int, series=survival_series,
               min_r2: float | None = None) -> ArmInputs:
    """One arm's `ArmInputs` at step `h`: each seed's primary rung paired
    against its reference, pooled over `seeds`, and the same within each seed
    alone."""
    def contrast(channel, seed_list):
        kept = [records[(arm, s)] for s in seed_list if _included(records[(arm, s)], channel, min_r2)]
        treatment = [series(r, int(r["primary_rung"]), channel, h, f"{arm}@primary") for r in kept]
        control = [series(r, REFERENCE_RUNG, channel, h, f"{arm}@{REFERENCE_RUNG}") for r in kept]
        return _paired(treatment, control)

    primary = {int(s): int(records[(arm, s)]["primary_rung"]) for s in seeds}
    per_seed = {
        int(s): ArmInputs(
            t_free=contrast("free", [s]), t_probe=contrast("probe", [s]),
            primary_rungs={int(s): primary[int(s)]}, per_seed=None,
        )
        for s in seeds
    }
    return ArmInputs(
        t_free=contrast("free", list(seeds)), t_probe=contrast("probe", list(seeds)),
        primary_rungs=primary, per_seed=per_seed,
    )


def _clusters(records: dict) -> int:
    return int(_entry(next(iter(records.values())), REFERENCE_RUNG)["summary"]["windows"]["clusters"])


def timing_inputs(records: dict, *, arms, seeds, h: int, series=survival_series,
                  min_r2: float | None = None) -> TimingInputs:
    """Every arm's inputs at `h`, and `z_fam` over the val stratum's cluster
    count (spec 3.1)."""
    ordered = sorted(arms, key=lambda a: ARMS_ORDER.index(a) if a in ARMS_ORDER else len(ARMS_ORDER))
    return TimingInputs(
        arms={arm: arm_inputs(records, arm, seeds, h=h, series=series, min_r2=min_r2) for arm in ordered},
        z_fam=pooling.cluster_threshold(FAMILY, _clusters(records)),
        h=h,
    )


# `decision_horizon` is `split_gap`'s, imported: one decision horizon (M3e's
# DECISION_H) for both readings.


# ---------------------------------------------------------------------------
# The printed tables.
# ---------------------------------------------------------------------------


def _num(value, spec: str = ".3f") -> str:
    value = float(value)
    return format(value, spec) if np.isfinite(value) else "n/a"


def _cells_in_order(records: dict) -> list[tuple[str, int]]:
    return sorted(records, key=lambda k: (ARMS_ORDER.index(k[0]) if k[0] in ARMS_ORDER else 9, k[1]))


def _steps_in_order(rungs) -> list[int]:
    return [*sorted(int(r) for r in rungs), REFERENCE_RUNG]


def _anchor_table(records: dict) -> str:
    lines = [
        "--- anchor: the retrain's per-step loss against the reference run's (spec 2.4; exact equality) ---",
        f"  {'arm':<12}{'seed':>5}{'steps':>7}{'policy':>8}{'max|delta|':>12}{'first_step':>11}  train git_sha",
    ]
    for arm, seed in _cells_in_order(records):
        r = records[(arm, seed)]
        a = r["anchor"]
        first = "-" if a["first_step"] is None else str(a["first_step"])
        lines.append(
            f"  {arm:<12}{seed:>5}{a['steps']:>7}{a['policy']:>8}{float(a['max_delta']):>12.1e}"
            f"{first:>11}  {r['train_git_sha'][:12]}"
        )
    return "\n".join(lines) + "\n"


def _primary_table(records: dict, rungs) -> str:
    first = next(iter(records.values()))
    lines = [
        f"--- primary rung per cell: the rung nearest the reference run's smoothed embedding minimum "
        f"({first['curve_window']}-step window; ties to the earlier; rungs {list(rungs)}) ---",
        f"  {'arm':<12}{'seed':>5}{'min_step':>10}{'primary':>9}",
    ]
    for arm, seed in _cells_in_order(records):
        r = records[(arm, seed)]
        lines.append(f"  {arm:<12}{seed:>5}{r['embedding_min_step']:>10}{r['primary_rung']:>9}")
    return "\n".join(lines) + "\n"


def _self_check_table(records: dict) -> str:
    lines = [
        f"--- reference self-check (step {REFERENCE_RUNG}'s trust pass against its diagnostic; exact) ---",
        f"  {'arm':<12}{'seed':>5}{'ref max|d|':>12}{'pers max|d|':>13}{'windows':>9}{'probe R2':>10}{'measurable':>12}  ok",
    ]
    for arm, seed in _cells_in_order(records):
        e = _entry(records[(arm, seed)], REFERENCE_RUNG)
        c = e["self_check"]
        lines.append(
            f"  {arm:<12}{seed:>5}{c['reference_position_max_delta']:>12.1e}"
            f"{c['persistence_position_max_delta']:>13.1e}"
            f"{str(c['windows_total_match'] and c['windows_episode_match']):>9}"
            f"{_num(e['probe']['selection_r2']):>10}{str(e['probe']['measurable']):>12}  {c['ok']}"
        )
    return "\n".join(lines) + "\n"


def _gate_table(records: dict, arms, seeds, rungs, horizon: int) -> str:
    lines = [
        f"--- the gate at every rung: gap_closed({horizon}) on position per seed (spec 4.1's metric; "
        "NaN = non-positive band); GATE PASSES = > 0 in every seed. Reported, not decided on. ---",
        f"  {'arm':<12}{'step':>7}" + "".join(f"{f's{s}':>10}" for s in seeds)
        + f"{'nanmean':>10}{'degen(max)':>11}  gate",
    ]
    for arm in arms:
        for step in _steps_in_order(rungs):
            finals = {int(s): float(_entry(records[(arm, s)], step)["gate"]["gap_final"]) for s in seeds}
            degenerate = max(int(_entry(records[(arm, s)], step)["gate"]["steps_degenerate"]) for s in seeds)
            values = list(finals.values())
            mean = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            flag = f"GATE PASSES at step {step}" if gate_passes(finals) else ""
            lines.append(
                f"  {arm:<12}{step:>7}" + "".join(f"{_num(finals[int(s)], '+.4f'):>10}" for s in seeds)
                + f"{_num(mean, '+.4f'):>10}{degenerate:>11}  {flag}"
            )
    return "\n".join(lines) + "\n"


def _stacked(records: dict, arm: str, step: int, channel: str, seeds, key: str) -> np.ndarray:
    """One per-window array of `key` -- `crossing` (per channel) or
    `first_moved` -- concatenated over the seeds included in `channel` at this
    rung: every cell in the free channel, the measurable cells in the probe
    one."""
    rows = []
    for s in seeds:
        if channel != "free" and not _probe_ok(records[(arm, s)], step, None):
            continue
        block = _entry(records[(arm, s)], step)["summary"]
        rows.append(np.asarray(block["crossing"][channel] if key == "crossing" else block[key], dtype=float))
    return np.concatenate(rows) if rows else np.zeros(0)


def _survival_table(records: dict, arms, seeds, rungs, horizon: int) -> str:
    steps = [h for h in SURVIVAL_STEPS if h <= horizon]
    lines = [
        "--- survival per rung: S(h) = fraction of moved draws with h_x > h "
        "(a window not yet moved at h survives vacuously, as in M3d); seeds stacked ---",
        f"  {'arm':<12}{'step':>7} {'channel':<7}" + "".join(f"{f'S({h})':>7}" for h in steps)
        + "".join(f"{'H*' + q_key(q)[1:]:>8}" for q in Q_REPORTED),
    ]
    for arm in arms:
        for step in _steps_in_order(rungs):
            for channel in CHANNELS:
                crossings = _stacked(records, arm, step, channel, seeds, "crossing")
                s = survival(crossings, horizon) if crossings.size else np.full(horizon + 1, np.nan)
                lines.append(
                    f"  {arm:<12}{step:>7} {channel:<7}" + "".join(f"{_num(s[h], '.2f'):>7}" for h in steps)
                    + "".join(f"{trust_horizon(s, q):>8d}" if crossings.size else f"{'n/a':>8}" for q in Q_REPORTED)
                )
    return "\n".join(lines) + "\n"


def _conditional_table(records: dict, arms, seeds, rungs, horizon: int) -> str:
    """M3d's `u(h)` and `S_c(h)` beside S(h), through `trust_horizon.py`'s own
    functions imported by path; printed, never decided on."""
    steps = [h for h in SURVIVAL_STEPS if h <= horizon]
    lines = [
        "--- conditional survival per rung: u(h) = unmoved fraction, S_c(h) = (S(h) - u(h)) / (1 - u(h)) "
        "(M3d's series; printed, not decided on) ---",
        f"  {'arm':<12}{'step':>7} {'channel':<7}" + "".join(f"{f'u({h})':>7}" for h in steps)
        + "".join(f"{f'Sc({h})':>8}" for h in steps),
    ]
    for arm in arms:
        for step in _steps_in_order(rungs):
            for channel in CHANNELS:
                crossings = _stacked(records, arm, step, channel, seeds, "crossing")
                first = _stacked(records, arm, step, channel, seeds, "first_moved")
                if not crossings.size:
                    lines.append(f"  {arm:<12}{step:>7} {channel:<7}" + "".join(f"{'n/a':>7}" for _ in steps)
                                 + "".join(f"{'n/a':>8}" for _ in steps))
                    continue
                s = (survival(crossings, horizon) if np.isfinite(crossings).any()
                     else np.full(horizon + 1, np.nan))
                u = _trust.unmoved_fraction(crossings, first, horizon)
                sc = _trust.conditional_survival(s, u)
                lines.append(
                    f"  {arm:<12}{step:>7} {channel:<7}"
                    + "".join(f"{_num(u[h], '.2f'):>7}" for h in steps)
                    + "".join(f"{_num(sc[h], '.2f'):>8}" for h in steps)
                )
    return "\n".join(lines) + "\n"


def _objective_table(records: dict, rungs) -> str:
    first = next(iter(records.values()))
    lines = [
        f"--- the validation objective per rung (val_objective over {first['objective_batches']} draws; "
        "descriptive, no verdict) beside the smoothed training embedding at that step ---",
        f"  {'arm':<12}{'seed':>5}{'step':>7}{'val loss':>10}{'val emb':>9}{'train emb':>10}"
        + "".join(f"{term:>10}" for term in TERMS if term != "embedding"),
    ]
    for arm, seed in _cells_in_order(records):
        r = records[(arm, seed)]
        best_emb = best_loss = None
        for step in _steps_in_order(rungs):
            e = _entry(r, step)
            o = e["objective"]
            lines.append(
                f"  {arm:<12}{seed:>5}{step:>7}{_num(o['loss'], '.4f'):>10}{_num(o['embedding'], '.4f'):>9}"
                f"{_num(e['train_embedding'], '.4f'):>10}"
                + "".join(f"{_num(o[term], '.4f'):>10}" for term in TERMS if term != "embedding")
            )
            if np.isfinite(o["embedding"]) and (best_emb is None or o["embedding"] < best_emb[1]):
                best_emb = (step, o["embedding"])
            if np.isfinite(o["loss"]) and (best_loss is None or o["loss"] < best_loss[1]):
                best_loss = (step, o["loss"])
        smallest_at = lambda best: "n/a" if best is None else str(best[0])
        lines.append(
            f"  {arm:<12}{seed:>5}  val embedding smallest at step {smallest_at(best_emb)}; val loss smallest at "
            f"step {smallest_at(best_loss)}; primary rung {r['primary_rung']}"
        )
    return "\n".join(lines) + "\n"


def _reported_table(records: dict, arms, seeds, horizon: int) -> str:
    steps = sorted({h for h in REPORTED_H if h <= horizon} | {horizon})
    lines = [
        f"--- T at the reported horizons (primary - {REFERENCE_RUNG}; z only; not decided on) ---",
        f"  {'arm':<12}{'channel':<8}" + "".join(f"{f'z@{h}':>9}" for h in steps) + f"{'dFree@dec':>11}",
    ]
    dec, _ = decision_horizon(horizon)
    for arm in arms:
        by_h = {h: arm_inputs(records, arm, seeds, h=h) for h in steps}
        margin = arm_inputs(records, arm, seeds, h=dec, series=margin_series).t_free
        for channel in CHANNELS:
            zs = [getattr(by_h[h], f"t_{channel}").z for h in steps]
            extra = (f"{_num(margin.estimate, '+.4f')} (z {_num(margin.z, '+.2f')})"
                     if channel == "free" else "")
            lines.append(
                f"  {arm:<12}{channel:<8}" + "".join(f"{_num(z, '+.2f'):>9}" for z in zs) + f"  {extra}"
            )
    lines.append("  dFree@dec: the paired rung difference of the embedding-space margin Delta_free "
                 "at the decision horizon -- the continuous companion, in the loss's units.")
    return "\n".join(lines) + "\n"


def _sensitivity_text(records: dict, arms, seeds, h: int, z_fam: float) -> str:
    dropped = [
        f"{a} s{s}" for a, s in _cells_in_order(records)
        if any(not _probe_ok(records[(a, s)], step, R2_SENSITIVITY)
               and _probe_ok(records[(a, s)], step, None)
               for step in (records[(a, s)]["primary_rung"], REFERENCE_RUNG))
    ]
    inputs = timing_inputs(records, arms=arms, seeds=seeds, h=h, min_r2=R2_SENSITIVITY)
    lines = [
        f"--- sensitivity (changes no verdict): probe channel with selection R2 < {R2_SENSITIVITY} at "
        f"either rung excluded -- dropped: {', '.join(dropped) if dropped else 'none'} ---",
    ]
    for arm, a in inputs.arms.items():
        lines.append(
            f"  {arm:<12}T_probe estimate {_num(a.t_probe.estimate, '+.4f')} se "
            f"{_num(a.t_probe.se, '.4f')} z {_num(a.t_probe.z, '+.2f')} "
            f"(bar {z_fam:.2f}; clusters {a.t_probe.clusters})"
        )
    return "\n".join(lines) + "\n"


def _per_seed_text(inputs: TimingInputs) -> str:
    lines = ["--- per seed (the same contrast within one seed alone; no seed averaging) ---"]
    for arm, a in inputs.arms.items():
        for seed, leaf in sorted((a.per_seed or {}).items()):
            leaf_reading = reading_timing(TimingInputs(arms={arm: leaf}, z_fam=inputs.z_fam, h=inputs.h)).arms[arm]
            lines.append(
                f"  {arm:<12}s{seed}: primary rung {leaf.primary_rungs[seed]}; T_free z "
                f"{_num(leaf.t_free.z, '+.2f')}, T_probe z {_num(leaf.t_probe.z, '+.2f')} -> "
                f"{leaf_reading.status.name.replace('_', ' ')}"
            )
    return "\n".join(lines) + "\n"


def write_curves(records: dict, figure: Path, rungs) -> str:
    """Three panels against the rung (the reference at the right): H*_0.75 on
    the free channel, gap_closed at the horizon, and the validation
    embedding; arms coloured, seeds as thin lines. A missing or broken
    matplotlib, or an unwritable path, costs the FIGURE and nothing else
    (report_study's guard)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, ValueError, OSError) as error:
        return f"figure NOT written: {error}"
    colours = {"pixel_ae": "tab:orange", "frozen_ssl": "tab:blue", "random_vit": "tab:gray"}
    steps = _steps_in_order(rungs)
    x = np.arange(len(steps))
    panels = (
        ("H*_0.75, free channel", lambda e: e["summary"]["trust_horizon"]["free"][q_key(0.75)]),
        ("gap_closed at the horizon (position)", lambda e: e["gate"]["gap_final"]),
        ("validation embedding loss", lambda e: e["objective"]["embedding"]),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    try:
        for ax, (title, pick) in zip(axes.flat, panels):
            for (arm, seed), record in records.items():
                y = [float(pick(_entry(record, step))) for step in steps]
                ax.plot(x, y, marker="o", markersize=3, color=colours.get(arm, "black"),
                        linewidth=0.9, alpha=0.85, label=f"{arm} s{seed}")
            if title.startswith("gap_closed"):
                ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in steps], rotation=45)
            ax.set_xlabel("training step (rung)")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), fontsize=8)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=110)
    except (ValueError, OSError) as error:
        return f"figure NOT written: {error}"
    finally:
        plt.close(fig)
    return f"figure={figure}"


def readings_text(records: dict, *, arms, seeds, rungs, horizon: int, figure_line: str) -> str:
    """Everything `read` prints, in `split_gap.txt`'s style; written to
    `ladder.txt` byte-identical."""
    h, clamped = decision_horizon(horizon)
    inputs = timing_inputs(records, arms=arms, seeds=seeds, h=h)
    reading = reading_timing(inputs)
    note = (f"  NOTE: decision horizon clamped to the run's horizon h={h} "
            f"(pre-registered DECISION_H = {DECISION_H}).\n" if clamped else "")
    return "".join([
        _anchor_table(records),
        _primary_table(records, rungs),
        _self_check_table(records),
        _gate_table(records, arms, seeds, rungs, horizon),
        _survival_table(records, arms, seeds, rungs, horizon),
        _conditional_table(records, arms, seeds, rungs, horizon),
        _objective_table(records, rungs),
        note,
        f"  pooling: z_fam = cluster_threshold({FAMILY}, {_clusters(records)}) = {inputs.z_fam:.2f}; "
        "each seed's primary rung is paired against its reference on the same val windows; the "
        "survival indicator pools windows that moved within the horizon; the probe channel pools "
        "cells measurable at both rungs.\n",
        format_reading_timing(reading, inputs),
        _reported_table(records, arms, seeds, horizon),
        _sensitivity_text(records, arms, seeds, h, inputs.z_fam),
        _per_seed_text(inputs),
        f"  {figure_line}\n",
    ])


def write_readings(out_dir: Path, text: str) -> Path:
    path = Path(out_dir) / "ladder.txt"
    path.write_text(text)
    return path


def read_phase(args, cells, rungs) -> int:
    """Load every requested ladder record (11), refuse one whose reference
    self-check is not ok (30), then the figure, the text, `ladder.txt`."""
    records: dict[tuple[str, int], dict] = {}
    for arm, seed in cells:
        path = ladder_record_path(args.out, arm, seed)
        if not path.exists():
            print(f"NO CELL: {arm} seed {seed}: no ladder record at {path}; run --phase evaluate first")
            return EXIT_NO_CHECKPOINTS
        records[(arm, seed)] = load_record(path)
    for (arm, seed), record in records.items():
        check = _entry(record, REFERENCE_RUNG)["self_check"]
        if not check or not check.get("ok"):
            print(
                f"\nSELF-CHECK FAILED for {arm} seed {seed}: the ladder record's reference "
                f"self-check is {check!r}; the record is not read against a ruler that reproduced."
            )
            return EXIT_SELF_CHECK_FAILED
        if record["rungs"] != [int(r) for r in rungs]:
            raise ValueError(
                f"{arm} seed {seed}: the ladder record holds rungs {record['rungs']}, not the "
                f"requested {list(rungs)}; a different ladder is a different --out"
            )
    horizon = int(next(iter(records.values()))["horizon"])
    figure = args.figure if args.figure is not None else args.out / "ladder_curves.png"
    figure_line = write_curves(records, figure, rungs)
    text = readings_text(
        records, arms=list(args.arms), seeds=[int(s) for s in args.seeds],
        rungs=rungs, horizon=horizon, figure_line=figure_line,
    )
    print(text, end="")
    write_readings(args.out, text)
    return EXIT_OK
```

(d) Replace `main` with:

```python
def main(argv: list[str] | None = None, *, rungs=RUNGS) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    rungs = tuple(sorted({int(r) for r in rungs}))
    if args.steps < max(rungs):
        parser.error(f"--steps {args.steps} is below the last rung {max(rungs)}")
    device = get_device(prefer=args.device)
    cells = [(arm, int(seed)) for arm in args.arms for seed in args.seeds]

    if args.phase in ("train", "evaluate", "all"):
        buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
        train, val = episode_split(buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
        references: dict[tuple[str, int], Cell] = {}
        if args.phase in ("train", "all"):
            # Every requested reference cell is loaded before anything trains.
            try:
                references = {(arm, seed): load_cell(args.reference, arm, seed) for arm, seed in cells}
            except CellMissing as error:
                print(f"NO CELL: {error}")
                return EXIT_NO_CHECKPOINTS
            args.out.mkdir(parents=True, exist_ok=True)
        for arm, seed in cells:
            if args.phase in ("train", "all"):
                status, _ = train_cell(args, buffer, arm, seed, references[(arm, seed)], device, rungs)
                if status != EXIT_OK:
                    return status
            if args.phase in ("evaluate", "all"):
                status, _ = evaluate_cell(args, buffer, arm, seed, device, train, val, rungs)
                if status != EXIT_OK:
                    return status
    if args.phase in ("read", "all"):
        return read_phase(args, cells, rungs)
    return EXIT_OK
```

(e) Extend the module docstring's `read` line to say what it now does ("pool the ladder records, decide Reading T, print the tables, write ladder.txt and ladder_curves.png") and add to the checks block:

```
  read:     EXIT_NO_CHECKPOINTS (11)     a requested ladder record is missing.
            EXIT_SELF_CHECK_FAILED (30)   a ladder record's recorded reference
                                          self-check is not ok.
```

- [ ] **Step 4: Run the script tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_checkpoint_ladder_script.py tests/eval/test_diagnose_dynamics_script.py -q`
Expected: all pass — 19 + 12 new in the ladder file (31), and the distinctness test green with `checkpoint_ladder`.

- [ ] **Step 5: Run the eval test directory**

Run: `.venv/bin/python -m pytest tests/eval -q`
Expected: all pass, 0 warnings.

- [ ] **Step 6: Commit**

```bash
git add scripts/checkpoint_ladder.py tests/eval/test_checkpoint_ladder_script.py tests/eval/test_diagnose_dynamics_script.py
git commit -m "feat: checkpoint_ladder read -- each seed's primary rung paired against its reference, Reading T, the gate and H* at every rung, the validation objective, and ladder.txt

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: the smoke, the anchor pin, the real run, the results

**Files:**
- Modify: `scripts/checkpoint_ladder.py` (`ANCHOR_DEFAULT`, pinned by the smoke), `docs/superpowers/specs/2026-09-18-mb-fps-m3f-checkpoint-ladder-design.md` (§2.4: one marked sentence), this plan (`## Task 7 results`)
- Outputs (gitignored): `runs/m3f_smoke/**`, `runs/m3f_ladder/**`

**Interfaces:**
- Consumes: everything above; `runs/m3_study_v2` (the nine M3c cells, their records and diagnostics); `data/my_way_home`.
- Produces: the results section and the pinned anchor policy.

This task is run by the controller, not a subagent: it takes ~7 hours of wall-clock and touches the real artefacts. Never `rm` anything under `runs/`; the smoke writes to its own directory so nothing needs removing.

- [ ] **Step 1: The full suite at the code state that will run**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: 1490 (the M3e count) + 8 (Task 1) + 1 (Task 2) + 7 (Task 3) + 26 + 5 (Task 4 and 4b) + 19 (Task 5) + 12 + 1 (Task 6 and its NaN fix) = **1569 passed**, 0 skipped with `runs/` reachable, 0 warnings. Record the count.

- [ ] **Step 2: The smoke — one cell, the full ladder, `--anchor report`**

```bash
mkdir -p runs/m3f_smoke && git rev-parse HEAD > runs/m3f_smoke/ladder.head && date -u +%Y-%m-%dT%H:%M:%SZ > runs/m3f_smoke/ladder.started
nohup bash -c 'caffeinate -dimsu env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/checkpoint_ladder.py --out runs/m3f_smoke --arms random_vit --seeds 0 --phase all --anchor report > runs/m3f_smoke/ladder.log 2>&1; echo $? > runs/m3f_smoke/ladder.exit; date -u +%Y-%m-%dT%H:%M:%SZ > runs/m3f_smoke/ladder.finished' > /dev/null 2>&1 &
```

(The exit status is written by the SAME shell that ran the script — the M3e run's wrapper waited on a PID from another shell and recorded a bogus 127.) Expected: ~22 min training + ~6 × 3 min evaluation ≈ 45 min. Watch `runs/m3f_smoke/ladder.exit` appear (the `Monitor` tool, not a sleep loop). Then:

```bash
cat runs/m3f_smoke/ladder.exit; grep -E 'anchor max\|delta\||SELF-CHECK|MISMATCH|MISLABELLED|NO CELL|Traceback' runs/m3f_smoke/ladder.log; tail -60 runs/m3f_smoke/ladder.txt
```

Expected: exit `0`; the anchor line for `random_vit` seed 0 with its `max|delta|`; `ladder.txt` with every block and `verdict: random_vit` (a single-seed run reads `NO DIFFERENCE` or a clear "in only 1 of 1 seeds" — the smoke is for format and the anchor, not a reading). Read `ladder.txt` in full once for column alignment and any `n/a` that should be a number.

- [ ] **Step 3: Pin the anchor policy (spec §2.4) — decided by the measurement, not by taste**

If the smoke's `anchor max|delta|` is exactly `0.0e+00`: set `ANCHOR_DEFAULT = "hard"` in `scripts/checkpoint_ladder.py`, and add to spec §2.4, directly after the **Anchor.** paragraph:

> *Pinned after the smoke run (random_vit/s0, 5,000 steps, MPS, torch 2.13.0, code state `<sha>`): the anchor delta measured exactly 0.0, so the real run is `--anchor hard` and every rung is a state the original run passed through.*

Otherwise set `ANCHOR_DEFAULT = "report"` and add:

> *Pinned after the smoke run (random_vit/s0, 5,000 steps, MPS, torch 2.13.0, code state `<sha>`): the anchor delta measured `<value>` (first at step `<n>`), so MPS training is not bitwise on this box; the real run is `--anchor report`, the rungs are a run with the study's seed, and the anchor claim is dropped.*

Update `test_the_parser_defaults_are_the_specs` if it pins the default (it accepts either value). Commit:

```bash
git add scripts/checkpoint_ladder.py docs/superpowers/specs/2026-09-18-mb-fps-m3f-checkpoint-ladder-design.md
git commit -m "docs: pin the M3f anchor policy from the smoke run's measurement

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Also amend spec §6's smoke sentence to say the smoke writes to `runs/m3f_smoke`, a directory the real run never reads (the repository rule is that nothing under `runs/` is ever removed).

- [ ] **Step 4: The real run — nine cells, all phases, the pinned policy**

Confirm `git status` is clean under `src/`, `scripts/` and `docs/`, then:

```bash
mkdir -p runs/m3f_ladder && git rev-parse HEAD > runs/m3f_ladder/ladder.head && date -u +%Y-%m-%dT%H:%M:%SZ > runs/m3f_ladder/ladder.started
nohup bash -c 'caffeinate -dimsu env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/checkpoint_ladder.py --out runs/m3f_ladder --phase all > runs/m3f_ladder/ladder.log 2>&1; echo $? > runs/m3f_ladder/ladder.exit; date -u +%Y-%m-%dT%H:%M:%SZ > runs/m3f_ladder/ladder.finished' > /dev/null 2>&1 &
```

Expected: ~6.5 h. Monitor `ladder_*.json` appearing (one per cell, ~40 min apart) and `ladder.exit`. If a cell refuses, the exit status names why; `--phase evaluate` re-runs on the saved checkpoints without retraining, `--phase read` on the saved ladder records.

- [ ] **Step 5: Acceptance**

```bash
.venv/bin/python - <<'EOF'
import json, subprocess
from pathlib import Path
from mbfps.eval.study import load_record
from mbfps.eval.ladder import RUNGS, REFERENCE_RUNG
out = Path("runs/m3f_ladder"); head = (out / "ladder.head").read_text().strip()
expected_primary = {("frozen_ssl", 0): 2000, ("frozen_ssl", 1): 4000, ("frozen_ssl", 2): 5000,
                    ("random_vit", 0): 2000, ("random_vit", 1): 5000, ("random_vit", 2): 3000,
                    ("pixel_ae", 0): 5000, ("pixel_ae", 1): 5000, ("pixel_ae", 2): 5000}
records = {}
for (arm, seed), primary in expected_primary.items():
    r = load_record(out / f"ladder_{arm}_seed{seed}.json"); records[(arm, seed)] = r
    assert r["git_sha"] == head == r["train_git_sha"], (arm, seed, r["git_sha"], head)
    assert r["device"] == "mps", (arm, seed, r["device"])
    c = r["entries"][str(REFERENCE_RUNG)]["self_check"]
    assert c["ok"] and c["reference_position_max_delta"] == 0.0 == c["persistence_position_max_delta"]
    assert r["primary_rung"] == primary, (arm, seed, r["primary_rung"], r["embedding_min_step"])
    assert r["rungs"] == list(RUNGS)
    for step in (*RUNGS, REFERENCE_RUNG):
        w = r["entries"][str(step)]["summary"]["windows"]
        assert (w["total"], w["clusters"]) == (229, 24), (arm, seed, step, w)
    a = r["anchor"]; print(f"{arm:<11} s{seed} anchor {a['policy']} max|delta| {a['max_delta']:.1e} first {a['first_step']} primary {r['primary_rung']} (min {r['embedding_min_step']})")
print("exit", (out / "ladder.exit").read_text().strip(), "started", (out / "ladder.started").read_text().strip(), "finished", (out / "ladder.finished").read_text().strip())
print("OK: nine records, one git_sha == HEAD, device mps, self-check 0.0 on 9/9, 229/24 on every rung, primary rungs == spec 2.2")
EOF
ls -lt runs/m3f_ladder | head -5
```

Expected: `OK ...`, exit `0`, and under `hard` every anchor `0.0e+00`.

- [ ] **Step 6: Write `## Task 7 results` into this plan**

Copy, do not round. The section carries, in this order:

1. **Provenance** — `git_sha` (= `ladder.head`), device, torch, start/finish, wall time, `ladder.exit`, the anchor policy and every cell's anchor delta, the suite count from Step 1.
2. **Primary rungs** — the nine `(embedding_min_step, primary_rung)` pairs beside spec §2.2's table.
3. **Reference self-check** — the table from `ladder.txt`.
4. **The gate at every rung** — arm × step, `gap_closed(45)` per seed, unanimity; any `GATE PASSES` line quoted verbatim.
5. **Survival and H\*** — `S(5)/S(15)/S(45)` and `H*_{0.5,0.75,0.9}` per arm × rung × channel, free channel first; the conditional `S_c(15)` beside.
6. **The validation objective** — per cell × rung: val `loss`, val `embedding`, train `embedding`; the rung at which each is smallest against the primary rung.
7. **Reading T at h = 15** — the contrast table (estimate ± se, z, clears), the per-seed primary rungs, the status per arm with its rule sentence; `pixel_ae`'s line read as the control.
8. **Companions** — z at h = 5, 15, 45; `Δ_free(15)`; the sensitivity line; the per-seed statuses.
9. **Figure** — `runs/m3f_ladder/ladder_curves.png`.
10. **Closing paragraph** — the verdict per arm stated against §4's non-claims, and what it says the next study is (loss balance / KL regime if `NO_DIFFERENCE`; checkpoint selection and a gate re-read if `EARLIER_BETTER`; the objective question if `EARLIER_WORSE`).

Then tick the plan's exit criteria and commit:

```bash
git add docs/superpowers/plans/2026-09-18-mb-fps-m3f-checkpoint-ladder.md
git commit -m "docs: M3f checkpoint ladder on runs/m3f_ladder -- Reading T per arm, the gate and H* at every rung, the validation objective

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 7 results

**Provenance.** Nine ladder records under one code state, `git_sha` = `train_git_sha` =
`7a6c228a2acf3d08c5951694dc167b275edc4401` (= `ladder.head` = `git rev-parse HEAD`, tree clean),
device `mps`, torch `2.13.0`, in all nine `ladder_<arm>_seed<n>.json`; every reference entry's
`record_git_sha` = `ca3e140772d6bc741d4d04312763afe3dd754166`, the M3c study's. Launched
`2026-09-19T08:27:13Z` (`ladder.started`) under `caffeinate -dimsu` with `PYTHONUNBUFFERED=1`,
`--out runs/m3f_ladder --phase all` and the pinned `--anchor hard` default, from the
`feat/m3f-checkpoint-ladder` worktree with `runs/` and `data/` symlinked to the main checkout's;
finished `2026-09-19T14:07:22Z` (`ladder.finished`), wall **5 h 40 min 09 s**; `ladder.exit` = `0`,
captured by the shell that ran the script. Per cell: 5,000 training steps in 1,415–1,468 s
(3.41–3.53 steps/s; `pixel_ae` the fastest arm) and six evaluations in about 13 min. `ladder.txt`
is byte-identical to the `read` phase's stdout. `ls -lt runs/m3f_ladder | head` shows nothing newer
than `ladder.finished`, `ladder.exit` and `ladder.txt`; the directory holds 2.0 GB (45 rung
checkpoints of 39 MB, 45 rung study records, 9 train records, 9 ladder records, the figure), and
nine top-level `world_model_<arm>_seed<n>.pt` — the trainer's final save, holding the same weights as `step5000/` (bitwise-equal `state_dict`s; the final payload lacks the `step` key the rung save carries)
because `STEPS == max(RUNGS)`; the top level holds no study record, so no instrument reads it as a
cell.

**The anchor** (`--anchor hard`, spec 2.4): the retrain's per-step loss equals the M3c record's
`history.loss` at **every one of the 5,000 steps in every one of the nine cells** — max |Δ| `0.0e+00`,
first differing step `-` on nine lines of the anchor table. MPS training is bitwise deterministic on
this box under torch 2.13.0, and every rung below is a state the original run passed through. (The
smoke run that pinned the policy — `random_vit`/s0 into `runs/m3f_smoke`, 07:48:41Z–08:25:50Z, exit
0, code state `3d6c562` — measured the same 0.0.)

**Suite.** `pytest` at `3d6c562` (the last code commit before the smoke): `1569 passed` in 13 min 44 s,
0 skipped with `runs/` reachable, 0 warnings, exit 0 — 1490 + 8 + 1 + 7 + 26 + 5 + 19 + 12 + 1, as
the plan's header now reads. At the launched HEAD `7a6c228` (two docs commits, `ANCHOR_DEFAULT =
"hard"`, and a header space in two tables later): `1569 passed` in 13 min 42 s, 0 skipped, 0 warnings, exit 0 — the same count.

**Acceptance** (`runs/m3f_ladder`, run inline): `OK: nine records, one git_sha == HEAD 7a6c228,
device mps, self-check 0.0 on 9/9, 229/24 on every rung, primary rungs == spec 2.2`.

**Primary rungs** — `learning_curve_summary(reference.history, window=100)["terms"]["embedding"]["smoothed_min_step"]`
→ `primary_rung`, computed at run time, against spec §2.2's pre-registered table:

| cell | `embedding` min step | primary rung | spec §2.2 |
|---|---|---|---|
| `pixel_ae` s0 / s1 / s2 | 8,359 / 19,096 / 16,873 | **5000 / 5000 / 5000** | 5000 / 5000 / 5000 ✓ |
| `frozen_ssl` s0 / s1 / s2 | 2,045 / 3,555 / 4,510 | **2000 / 4000 / 5000** | 2000 / 4000 / 5000 ✓ |
| `random_vit` s0 / s1 / s2 | 1,610 / 4,535 / 3,275 | **2000 / 5000 / 3000** | 2000 / 5000 / 3000 ✓ |

**Reference self-check** (step 20000's trust pass against its M3c diagnostic; exact): `ref max|d|`
`0.0e+00` and `pers max|d|` `0.0e+00` on 9/9 cells, windows `True` on 9/9, `measurable` True on 9/9;
probe selection R² `0.366 / 0.018 / 0.272` (`pixel_ae`), `0.360 / 0.334 / 0.383` (`frozen_ssl`),
`0.297 / 0.330 / 0.249` (`random_vit`) — the M3d/M3e numbers. Every rung of every cell scores the
same 229 val windows over 24 episode-clusters; `u(h)` is identical on every line of the conditional
table (0.58 / 0.35 / 0.21 / 0.13 / 0.05 / 0.03 / 0.00 / 0.00 at h = 1 / 2 / 3 / 5 / 10 / 15 / 30 / 45),
as it must be — the moved mask is the truth's.

**The gate at every rung** (`gap_closed(45)` on position per seed; NaN = non-positive band;
`GATE PASSES` = > 0 in every seed; reported, not decided on). **No arm passes at any rung.**

| arm | step | s0 | s1 | s2 | nanmean | degenerate (max over seeds) |
|---|---|---|---|---|---|---|
| `pixel_ae` | 1000 | −1.7644 | −3.7309 | n/a | −2.7477 | 1 |
| `pixel_ae` | 2000 | −3.6633 | −3.2706 | −2.6703 | −3.2014 | 1 |
| `pixel_ae` | 3000 | −1.0851 | −63.1216 | −4.1054 | −22.7707 | 2 |
| `pixel_ae` | 4000 | −0.7270 | −2.8571 | −0.7349 | −1.4396 | 0 |
| `pixel_ae` | 5000 | −0.1831 | −2.0846 | −5.5826 | −2.6168 | 0 |
| `pixel_ae` | 20000 | −1.1630 | −0.5878 | −0.6739 | −0.8082 | 0 |
| `frozen_ssl` | 1000 | n/a | n/a | +1.3845 | +1.3845 | 44 |
| `frozen_ssl` | 2000 | n/a | +1.9942 | +0.9828 | +1.4885 | 45 |
| `frozen_ssl` | 3000 | −2.3192 | n/a | +1.2144 | −0.5524 | 11 |
| `frozen_ssl` | 4000 | −0.9903 | n/a | n/a | −0.9903 | 2 |
| `frozen_ssl` | 5000 | −1.5876 | −1.7580 | n/a | −1.6728 | 1 |
| `frozen_ssl` | 20000 | −0.8523 | −0.6470 | −0.6491 | −0.7161 | 0 |
| `random_vit` | 1000 | −2.9910 | −1.8782 | −15.1137 | −6.6610 | 1 |
| `random_vit` | 2000 | −1.1611 | −1.7363 | −0.5918 | −1.1631 | 1 |
| `random_vit` | 3000 | −1.1065 | −2.8457 | −2.9254 | −2.2925 | 0 |
| `random_vit` | 4000 | −1.3049 | −1.7059 | −1.2567 | −1.4225 | 1 |
| `random_vit` | 5000 | −0.9090 | −1.7879 | −1.8623 | −1.5197 | 0 |
| `random_vit` | 20000 | −0.4419 | −0.3875 | −0.5452 | −0.4582 | 0 |

The 20000 row is the M3c records' own `gap_final` (`rung_entry` reads it off the reference record;
what reproduces bitwise is the trust pass behind it, per the self-check above). The positive
`frozen_ssl` entries at 1000–3000 (+0.98 to +1.99) sit beside `degenerate` 44, 45 and 11 — the
persistence-to-floor band is non-positive or vanishing over most of the horizon at those rungs (the
posterior's own probe-read position is no better than persistence), so the metric there reads the
band's collapse, not a rollout that beats persistence; the NaN seeds beside them are the same
collapse, and unanimity fails. Among the rows whose three seeds all carry a finite `gap_closed`,
the 20000 row is the least negative for both ViT arms; for `pixel_ae` s0 the least negative seed
value is at 5000 (−0.1831), for s1 and s2 at 20000.

**Survival and H\*** (free channel; `S(h)` over the moved (window, seed) draws of the arm's three
seeds stacked, 687 per line; `H*_0.75 = 1` on every line of every arm, rung and channel but
`random_vit` 4000 probe = 0; `H*_0.9 = 0` everywhere):

| arm | step | `S(5)` | `S(15)` | `S(45)` | `H*_0.5` | `S_c(15)` | probe `S(15)` |
|---|---|---|---|---|---|---|---|
| `pixel_ae` | 1000 / 2000 / 3000 / 4000 / 5000 | 0.27 / 0.28 / 0.27 / 0.35 / 0.41 | 0.08 / 0.07 / 0.08 / 0.13 / 0.16 | 0.02 / 0.00 / 0.00 / 0.03 / 0.02 | 2 / 2 / 2 / 2 / 3 | 0.05 / 0.04 / 0.05 / 0.10 / 0.13 | 0.14 / 0.13 / 0.13 / 0.20 / 0.20 |
| `pixel_ae` | **20000** | 0.40 | **0.17** | 0.04 | 3 | 0.14 | 0.23 |
| `frozen_ssl` | 1000 / 2000 / 3000 / 4000 / 5000 | 0.35 / 0.27 / 0.27 / 0.32 / 0.28 | 0.11 / 0.06 / 0.07 / 0.09 / 0.07 | 0.01 / 0.00 / 0.00 / 0.01 / 0.01 | 2 / 2 / 2 / 2 / 2 | 0.07 / 0.02 / 0.04 / 0.06 / 0.04 | 0.18 / 0.16 / 0.10 / 0.22 / 0.12 |
| `frozen_ssl` | **20000** | 0.39 | **0.14** | 0.02 | 3 | 0.11 | 0.22 |
| `random_vit` | 1000 / 2000 / 3000 / 4000 / 5000 | 0.29 / 0.27 / 0.30 / 0.26 / 0.34 | 0.11 / 0.08 / 0.11 / 0.07 / 0.11 | 0.02 / 0.01 / 0.02 / 0.00 / 0.02 | 2 / 2 / 2 / 2 / 2 | 0.08 / 0.05 / 0.07 / 0.04 / 0.08 | 0.13 / 0.15 / 0.14 / 0.13 / 0.15 |
| `random_vit` | **20000** | 0.39 | **0.18** | 0.05 | 3 | 0.15 | 0.27 |

In every arm the 20000 row has the highest free-channel `S(15)` and `S(45)` of the six rungs and
the only `H*_0.5 = 3` (`pixel_ae` reaches 3 at 5000 too). No rung anywhere lifts `H*_0.75` off 1.

**The validation objective** (`val_objective` over 50 seeded draws per rung; `val emb` = the
one-step `embedding` term on validation sequences, `train emb` = the 100-step moving mean of the
training term at that step; the `reward` term is 0.0000 and `continue` ≤ 0.0007 on every line):

| cell | rung | val loss | val emb | train emb | val `kl_dyn` | val emb smallest at | val loss smallest at |
|---|---|---|---|---|---|---|---|
| `pixel_ae` s0 | 1000 / 2000 / 3000 / 4000 / 5000 / **20000** | 0.7541 / 0.5059 / 0.3551 / 0.3315 / 0.3862 / 0.4406 | 0.2300 / 0.1955 / 0.1802 / 0.1766 / 0.1732 / **0.1562** | 0.2337 / 0.1892 / 0.1769 / 0.1692 / 0.1678 / 0.1473 | 0.8725 / 0.5164 / 0.2905 / 0.2562 / 0.3541 / 0.4733 | 20000 | 4000 |
| `pixel_ae` s1 | — | 0.7156 / 0.5105 / 0.3218 / 0.5400 / 0.4358 / 0.2249 | 0.2289 / 0.1815 / 0.1746 / 0.1624 / 0.1598 / **0.0847** | 0.2238 / 0.1737 / 0.1714 / 0.1637 / 0.1524 / 0.0790 | 0.8111 / 0.5485 / 0.2420 / 0.6293 / 0.4600 / 0.2193 | 20000 | 20000 |
| `pixel_ae` s2 | — | 0.7525 / 0.6542 / 0.3425 / 0.4788 / 0.4109 / 0.3248 | 0.2220 / 0.1770 / 0.1745 / 0.1768 / 0.1631 / **0.1282** | 0.2196 / 0.1725 / 0.1697 / 0.1713 / 0.1625 / 0.1241 | 0.8837 / 0.7949 / 0.2788 / 0.5030 / 0.4125 / 0.3273 | 20000 | 20000 |
| `frozen_ssl` s0 | — | 0.3230 / 0.3118 / 0.6762 / 0.6584 / 0.4515 / 0.7358 | 0.2022 / **0.1914** / 0.2190 / 0.2294 / 0.2415 / 0.3146 | 0.1999 / 0.1966 / 0.2117 / 0.2170 / 0.2324 / 0.2941 | 0.0628 / 0.1478 / 0.7612 / 0.7142 / 0.3492 / 0.7015 | 2000 | 2000 |
| `frozen_ssl` s1 | — | 0.3056 / 0.2909 / 0.2849 / 0.2814 / 0.3971 / 0.4840 | 0.1855 / 0.1709 / 0.1649 / **0.1614** / 0.1877 / 0.2136 | 0.1867 / 0.1691 / 0.1672 / 0.1595 / 0.1830 / 0.2065 | 0.0655 / 0.0789 / 0.0709 / 0.0734 / 0.3490 / 0.4506 | 4000 | 4000 |
| `frozen_ssl` s2 | — | 0.3017 / 0.2987 / 0.2894 / 0.2813 / 0.6046 / 0.6809 | 0.1812 / 0.1784 / 0.1691 / **0.1610** / 0.2282 / 0.3654 | 0.1818 / 0.1772 / 0.1671 / 0.1591 / 0.1659 / 0.3454 | 0.0980 / 0.0644 / 0.0418 / 0.0305 / 0.6269 / 0.5255 | 4000 | 4000 |
| `random_vit` s0 | — | 0.4351 / 0.4539 / 0.2550 / 0.2966 / 0.3010 / 0.5131 | **0.0915** / 0.0961 / 0.0928 / 0.0933 / 0.0954 / 0.1945 | 0.1076 / 0.0945 / 0.0955 / 0.0921 / 0.0960 / 0.1904 | 0.5718 / 0.5956 / 0.2661 / 0.3376 / 0.3418 / 0.5304 | 1000 | 3000 |
| `random_vit` s1 | — | 0.3774 / 0.4028 / 0.3637 / 0.6346 / 0.3285 / 0.5281 | **0.0715** / 0.0880 / 0.0833 / 0.1049 / 0.0895 / 0.1952 | 0.0793 / 0.0988 / 0.0835 / 0.0988 / 0.0886 / 0.1846 | 0.5098 / 0.5246 / 0.4673 / 0.8829 / 0.3984 / 0.5547 | 1000 | 5000 |
| `random_vit` s2 | — | 0.8819 / 0.3685 / 0.4422 / 0.5096 / 0.4163 / 0.4737 | 0.1457 / **0.0780** / 0.0792 / 0.0804 / 0.0908 / 0.1298 | 0.2275 / 0.0887 / 0.0865 / 0.0885 / 0.0953 / 0.1380 | 1.2264 / 0.4837 / 0.6045 / 0.7150 / 0.5420 / 0.5729 | 2000 | 2000 |

The training minimum is also the validation minimum, arm by arm: `val emb` is smallest at 20000
in all three `pixel_ae` cells (the arm still descending at 20k) and at 1000–4000 in all six ViT
cells, where it is 1.3–2.7× smaller than at 20000 (`frozen_ssl` 0.19 → 0.31, 0.16 → 0.21,
0.16 → 0.37; `random_vit` 0.09 → 0.19, 0.07 → 0.20, 0.08 → 0.13); at the primary rung itself it is
1.3–2.2× smaller than at 20000. `train emb` tracks `val emb` to
within 0.01–0.02 on every line but `frozen_ssl` s2 at 5000 (0.1659 vs 0.2282) and `random_vit` s2 at
1000 (0.2275 vs 0.1457, the smoothing window straddling a fast descent): the rise of the ViT arms'
one-step loss over training is a rise on validation too, not overfitting-in-time. The ladder's
1,000-step coarseness (spec §4) bites visibly in one cell: `frozen_ssl` s2's rule-chosen primary
rung is 5000 (nearest to 4,510), where val `embedding` is 0.2282 against 0.1610 at 4000 — its
contrast is read at a rung already past its minimum, and it is the arm's most negative seed
(z −5.84).

**The KL regime by rung** (from the rung study records' `kl_rate_above_free_bits`, the share of
training steps up to that rung whose `kl_dyn` cleared `KL_FREE_BITS = 0.20` — the only steps on
which `prior_net` receives gradient — and the val `kl_dyn` above; read for §4's "does not say why"):

| cell | clearance rate over the first 1000 / 2000 / 3000 / 4000 / 5000 steps | at 20000 |
|---|---|---|
| `pixel_ae` s0 / s1 / s2 | 0.17 / 0.59 / 0.72 / 0.77 / 0.81 — 0.22 / 0.61 / 0.67 / 0.75 / 0.80 — 0.23 / 0.62 / 0.74 / 0.80 / 0.82 | 0.91 / 0.83 / 0.84 |
| `frozen_ssl` s0 / s1 / s2 | 0.01 / 0.01 / 0.33 / 0.50 / 0.60 — **0.01 / 0.01 / 0.01 / 0.01 / 0.14** — **0.01 / 0.00 / 0.00 / 0.01 / 0.01** | 0.90 / 0.78 / 0.75 |
| `random_vit` s0 / s1 / s2 | 0.60 / 0.80 / 0.84 / 0.87 / 0.89 — 0.52 / 0.76 / 0.84 / 0.88 / 0.90 — 0.25 / 0.62 / 0.74 / 0.81 / 0.85 | 0.97 / 0.98 / 0.96 |

In `frozen_ssl` s1 and s2 the dynamics KL cleared the floor on about 1 % of the first 4,000–5,000
steps, so the prior was essentially untrained through rung 4000 in both cells (val `kl_dyn` 0.073
and 0.030 there, below the floor) — s1's primary rung. s2's primary rung is 5000, where the
transition has just begun: val `kl_dyn` is 0.627 and val loss jumped 0.28 → 0.60 between 4000 and
5000, while the cumulative clearance rate is still 0.01 because the clearing steps are the last few
dozen. Those are the rungs
at which the one-step `embedding` loss is smallest. `frozen_ssl` s0's prior begins training between
2000 and 3000 (rate 0.01 → 0.33); its primary rung, 2000, is before that. The `random_vit` and
`pixel_ae` priors clear the floor from the first 1,000–2,000 steps on.

**Reading T at h = 15** (`primary − 20000`, paired on the 229 val windows, seeds averaged per
window, episode-clustered; `z_fam = cluster_threshold(6, 24) = 2.89`):

| arm | `T_free(15)` estimate ± se (z) | `T_probe(15)` estimate ± se (z) | seeds clearing | **status** | decided by |
|---|---|---|---|---|---|
| `pixel_ae` (control) | −0.0116 ± 0.0193 (−0.60) | −0.0277 ± 0.0199 (−1.39) | 0 / 3 | **NO DIFFERENCE** | `T_free z -0.60 does not clear +-2.89` |
| `frozen_ssl` | **−0.0844 ± 0.0123 (−6.88)** | n/a (no cell measurable at both rungs) | 3 / 3 down | **EARLIER WORSE** | `T_free z -6.88 < -2.89 pooled and in 3 of 3 seeds` |
| `random_vit` | **−0.0771 ± 0.0146 (−5.30)** | −0.1485 ± 0.0273 (−5.43) | 2 / 3 down | **EARLIER WORSE** | `T_free z -5.30 < -2.89 pooled and in 2 of 3 seeds` |

The control reads as a control: `pixel_ae`, whose loss minimum lies beyond the ladder and whose
primary rung is the last one, shows no difference between 5000 and 20000 (per seed z +0.56 /
−0.58 / −2.01, none clearing). Both ViT arms read `EARLIER_WORSE`: the model at its one-step-loss
minimum keeps ahead of embedding-space persistence at 15 imagined steps on 7.7–8.4 percentage
points FEWER of the moved draws than the step-20,000 model does, with z at −5.3 and −6.9 against a
bar of 2.89, replicated in 3 of 3 (`frozen_ssl`: per-seed z −3.24 / −3.04 / −5.84) and 2 of 3
(`random_vit`: −3.14 / −0.88 / −6.42) seeds. `random_vit`'s probe channel agrees (z −5.43, per seed
−3.91 / −3.20 / −4.61). `frozen_ssl`'s probe channel is unmeasurable at its primary rungs — the
band at h=45 is non-positive there (the gate table's NaNs and `degenerate` 44/45/11) — so the free
channel decides alone, as spec 3.1 provides; no `UNRESOLVED_PROBE` was reachable.

**Companions** (change no verdict). z at the reported horizons, `primary − 20000`:

| arm | channel | z@5 | z@15 | z@45 | `Δ_free(15)` paired difference |
|---|---|---|---|---|---|
| `pixel_ae` | free | +0.23 | −0.60 | −2.23 | −0.2091 (z −1.91) |
| `pixel_ae` | probe | −0.46 | −1.39 | −1.58 | |
| `frozen_ssl` | free | −5.54 | −6.88 | −1.89 | +0.2323 (z +1.81) |
| `frozen_ssl` | probe | n/a | n/a | n/a | |
| `random_vit` | free | −4.36 | −5.30 | −3.44 | −0.9410 (z −5.18) |
| `random_vit` | probe | −4.26 | −5.43 | −4.05 | |

The ViT arms' deficit at the primary rung holds at h = 5 (z −5.54, −4.36) and, for `random_vit`, at
h = 45 too (−3.44 free, −4.05 probe); `frozen_ssl`'s h = 45 contrast is −1.89 (both rungs near
zero survival there). The margin companion agrees in sign for `random_vit` (−0.94 embedding units,
z −5.18) and has the opposite sign for `frozen_ssl` (+0.23, z +1.81, not clearing): at h = 15 the
early `frozen_ssl` model's embedding-space margin is slightly larger on the windows still moving
while far fewer of its draws are still ahead — the two statistics measure different things and the
pre-registered one is the survival. Sensitivity: dropping the two `pixel_ae` cells with probe R²
< 0.1 at either rung of their pair (s1: 0.018 at 20000; s2: 0.060 at its primary rung 5000) moves `pixel_ae`'s `T_probe` from −0.0277 (z
−1.39) to +0.0131 (z +0.31) — still not clearing — and nothing else. Per seed: `NO DIFFERENCE` on
all three `pixel_ae` seeds and on `random_vit` s1 (z −0.88); `EARLIER WORSE` on the other five.

**Figure.** `runs/m3f_ladder/ladder_curves.png` (106 KB): `H*_0.75` (free), `gap_closed(45)` and the
validation `embedding` against the rung, the reference at the right; arms coloured, seeds thin.

**Read against §4.** The M3 gate is unchanged: no arm passes `beats_persistence` at any rung, and
every recorded M3b–M3e verdict stands. No arm is ranked against another. What this run says is the
pre-registered thing and one descriptive thing beside it:

- The timing hypothesis is **refuted for both ViT arms, in the opposite direction**: evaluated at
  its own one-step-loss minimum — a minimum that is a validation minimum too — the model rolls out
  WORSE than at step 20,000 — the paired survival contrast is negative at every reported horizon on
  every measurable channel, clearing `z_fam` at h = 5 and 15 for both arms and at h = 45 for
  `random_vit` (`frozen_ssl`'s h = 45 contrast is −1.89, both rungs near zero survival there) —
  while its one-step loss is 1.3–2.2× lower. The control arm shows no difference. Checkpoint selection on the
  `embedding` loss would make M3 worse, not better; the M3 gate is not an artefact of when the
  model was evaluated.
- The one-step embedding loss is therefore not the rollout's ruler (spec 3.3's `EARLIER_WORSE`
  reading): the training that raises it from step ~4,000 to 20,000 is the training that lengthens
  imagination. The KL-regime table names the candidate mechanism without deciding it: in
  `frozen_ssl` the loss minimum falls where the dynamics prior has received gradient on ~1 % of
  steps — a low one-step loss with an untrained prior — and the loss rises exactly as the KL clears
  the free-bits floor and the prior begins to train; in `random_vit` the prior trains from the
  start and the same rise-with-better-rollouts appears, so the trade is not only the floor. Either
  way the next M3 study is the objective — the balance between the one-step reconstruction, the
  KL terms and what a 15-step rollout needs — and not the schedule, the data (M3e) or the budget.
- Everything is `my_way_home`'s 24 validation episodes at context 5 / horizon 45, the M3c seeds
  and configuration, a 1,000-step ladder to 5,000 and the shipped 20,000-step checkpoints; the
  rungs are bitwise the original runs' states. A finer ladder, a longer one, or another environment
  is a different measurement.

---

## Exit criteria

- [x] Tasks 1–6 committed, each reviewed; the full suite green with 0 warnings at the launched HEAD (`1569 passed` at `7a6c228`).
- [x] The anchor policy pinned in spec §2.4 and `ANCHOR_DEFAULT` from the smoke's measurement, before the real run (`hard`, commit `219b4c6`).
- [x] Nine `ladder_*.json` at one `git_sha` == HEAD on `mps`; reference self-check 0.0 on 9/9; 229 windows / 24 clusters on every rung; primary rungs equal to spec §2.2.
- [x] `## Task 7 results` written, numbers copied, verdicts stated against spec §4.
