# MB-FPS M3b: Close the Harness, Run the Study — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three findings that block M3's evaluation harness, then run 3 arms × 3 seeds and evaluate M3's exit gate.

**Architecture:** Plan 3 built and proved the world model and its harness locally on one arm. Three review findings remain, all in the evaluation path, and all must be fixed before the study runs — a study executed against a biased harness costs its whole compute budget twice. Tasks 1–3 close them; Tasks 4–7 run the study and judge the gate.

**Tech Stack:** Python 3.12, PyTorch, NumPy. Existing `mbfps` package. Study executes on a rented GPU.

**Spec:** `docs/superpowers/specs/2026-09-03-mb-fps-m3-world-model-design.md`
**Predecessor:** `docs/superpowers/plans/2026-09-03-mb-fps-m3a-world-model.md` (423 tests, complete)

---

## Global Constraints

- `src/mbfps/models/encoders.py` is the ONLY module that may differ in behaviour between arms.
- `privileged_state` is EVALUATION-ONLY and must never reach a training tensor or receive gradient.
- `terminated` and `truncated` stay distinct; the continue head targets `terminated` only.
- All three rollout references (model, persistence, floor) go through the IDENTICAL pipeline and are probed by the SAME probe, fit on the model's own PREDICTED embeddings.
- Sampling is ALWAYS stochastic. Reproducibility comes from seeding, never from taking the categorical mode — the mode was measured collapsing the imagined trajectory to 3 distinct latents of 45 and roughly quadrupling error.
- `KL_FREE_BITS = 0.20`, KL scales dyn/rep 0.5/0.1, h=512, z=32×32, head MLPs 2×512, batch/seq 16/64, LR 1e-4, fp32.
- The train/val split seed is fixed at **0** for every arm and every seed. Varying it would mean the comparison measures which episodes each run happened to get.
- Every guard is mutation-tested. **Harnesses must self-check three ways**, all of which have silently lied in this project: the package is installed EDITABLE so a naive `git archive` export is shadowed (run with `PYTHONPATH` at the export); an unproven harness proves nothing (run a known-fatal mutation FIRST); and a same-byte-length mutation can leave a stale `.pyc` that survives a correct restore (clear `__pycache__`, set `PYTHONDONTWRITEBYTECODE=1`).
- Nothing under `runs/` or `data/` may be committed (both gitignored).
- Unattended local runs need `caffeinate -dimsu`; `caffeinate -i` is insufficient on this host.

**Measured costs** (verified 2026-09-04, `seq_len=64`, batch 16, this Mac — do not re-derive):

| arm | steps/s | 20,000 steps |
|---|---|---|
| `frozen_ssl`, `random_vit` | 3.98 | 1.4 h |
| `cnn` | 0.67 | 8.3 h |

3 arms × 3 seeds ≈ **33 laptop-hours**, of which `cnn` is 25 h (75%). The arm ratio is **5.9×** at `seq_len=64`. Note that M2 measured 1.96× at `seq_len=1`, where fixed per-step overhead dominates; that figure does NOT transfer to this config and must not be used for sizing.

**Measured reward distribution** (verified 2026-09-05 over 40 episodes — do not re-derive):

`my_way_home` reward is **near-degenerate**: 19,417 of 19,424 steps carry the same living
penalty of -0.0004, six steps carry the goal reward of ~0.9996, and std is 0.0176. Only
0.03% of steps are terminal. Spec §4.3 asks for reward-prediction accuracy per arm, and it
must be reported — but a scalar R² or MSE over this distribution is dominated by the
constant and says almost nothing. `reward_accuracy` therefore reports the constant-predictor
baseline and the event count beside the error, so the number is read for what it is. This is
the same situation as `health` and `pos_z` having exactly zero variance, which is why
`PROBE_KEYS` excludes them.

**Measured evaluation facts** (from Plan 3, do not re-derive):

- At 20,000 steps the posterior latent reaches held-out R² **+0.2466** for position and the band is healthy: median width 55.2, **0/45** horizon steps with floor above persistence, **45/45** finite `gap_closed`. At 2,000 steps the latent was at −0.0058 and the band was noise. **Never judge the harness on an undertrained model.**
- `kl_rate_above_free_bits = 0.9725` at 20k, confirming `KL_FREE_BITS = 0.20`.
- `gap_closed` at horizon 45 was **−0.887** (mean −0.383, max +0.0131) for `random_vit` at 20k, `seq_len=32`. The gate's question is whether `seq_len=64` across three seeds clears zero.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/mbfps/eval/summary.py` (create) | ONE band/degeneracy summary, shared by the script and the study |
| `scripts/eval_rollout.py` (modify) | report degeneracy guards for angle as well as position |
| `src/mbfps/eval/probe.py` (modify) | extract `gather_probe_data`; fit probes under the ROLLOUT protocol; add `filtering_report` |
| `src/mbfps/eval/study.py` (create) | one `(arm, seed)` job: train, evaluate, emit a result record |
| `scripts/run_study.py` (create) | resumable driver over the 9 jobs |
| `src/mbfps/eval/aggregate.py` (create) | NaN-aware aggregation across seeds; gate evaluation |
| `scripts/report_study.py` (create) | error-vs-horizon curves and the gate verdict |

---

## Task 1: Degeneracy guards for angle, not just position

Every guard in `band_report` covers position. `angle_gap_closed_final` is printed bare, and recomputed from a real run it ranges **+8.68 to −23.04** with 1/45 steps NaN. The module docstring's own rule — "must be NaN-aware AND must report the band width alongside the ratio" — is violated by the harness that enforces it for the sibling metric.

**Files:**
- Create: `src/mbfps/eval/summary.py`
- Modify: `scripts/eval_rollout.py`
- Test: `tests/eval/test_eval_rollout_script.py` (create)

> The computation lives in the LIBRARY, not the script. Task 4's study job needs
> exactly the same summary and the same degeneracy threshold, and a script cannot
> be imported (`scripts/` is not a package — verified: `import scripts.eval_rollout`
> raises ModuleNotFoundError). Two copies is how the threshold drifts between what
> a single run reports and what the nine-cell study records.

**Interfaces:**
- Consumes: `RolloutResult` from `mbfps.eval.rollout`, with fields `horizon`, `rssm_position`, `persistence_position`, `floor_position`, `rssm_angle`, `persistence_angle`, `floor_angle`, and methods `position_gap_closed()`, `angle_gap_closed()`.
- Produces: `mbfps.eval.summary.metric_summary(result, metric: str) -> dict` where `metric` is `"position"` or `"angle"`, returning keys; `scripts/eval_rollout.py` imports it and aliases it as `band_report`. Also exports `METRICS` and `DEGENERATE`. `"band_min"`, `"band_median"`, `"band_max"`, `"relative_min"`, `"relative_median"`, `"relative_max"`, `"steps_floor_above_persistence"`, `"steps_degenerate"`, `"gap_finite"`, `"gap_mean"`, `"gap_min"`, `"gap_max"`, `"n_steps"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_eval_rollout_script.py
"""The reporting layer must guard BOTH metrics, not just position.

Every degeneracy guard was written for position and none for angle, so
`angle_gap_closed_final` was printed as a bare number while its curve ranged
+8.68 to -23.04 with one NaN step.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "eval_rollout_script",
    Path(__file__).resolve().parents[2] / "scripts" / "eval_rollout.py",
)
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)

from mbfps.eval.rollout import RolloutResult


def _result(pos, pers_pos, floor_pos, ang, pers_ang, floor_ang) -> RolloutResult:
    n = len(pos)
    return RolloutResult(
        horizon=np.arange(1, n + 1),
        rssm_position=np.array(pos, dtype=float),
        persistence_position=np.array(pers_pos, dtype=float),
        floor_position=np.array(floor_pos, dtype=float),
        rssm_angle=np.array(ang, dtype=float),
        persistence_angle=np.array(pers_ang, dtype=float),
        floor_angle=np.array(floor_ang, dtype=float),
    )


def test_band_report_accepts_a_metric_and_reads_the_matching_curves():
    """Asking for angle must read the angle curves, not the position ones."""
    result = _result(
        pos=[10.0, 10.0], pers_pos=[20.0, 20.0], floor_pos=[0.0, 0.0],
        ang=[5.0, 5.0], pers_ang=[9.0, 9.0], floor_ang=[1.0, 1.0],
    )
    position = script.band_report(result, "position")
    angle = script.band_report(result, "angle")
    assert position["band_median"] == pytest.approx(20.0)
    assert angle["band_median"] == pytest.approx(8.0)
    assert position["gap_mean"] == pytest.approx(0.5)
    assert angle["gap_mean"] == pytest.approx(0.5)


def test_band_report_counts_floor_above_persistence_for_angle():
    result = _result(
        pos=[1.0], pers_pos=[2.0], floor_pos=[0.0],
        ang=[1.0], pers_ang=[2.0], floor_ang=[3.0],   # floor ABOVE persistence
    )
    assert script.band_report(result, "position")["steps_floor_above_persistence"] == 0
    assert script.band_report(result, "angle")["steps_floor_above_persistence"] == 1


def test_band_report_counts_nan_gap_steps_for_angle():
    """A non-positive band yields NaN; the count must be reported, not hidden."""
    result = _result(
        pos=[1.0, 1.0], pers_pos=[2.0, 2.0], floor_pos=[0.0, 0.0],
        ang=[1.0, 1.0], pers_ang=[2.0, 2.0], floor_ang=[2.0, 0.0],  # first band == 0
    )
    angle = script.band_report(result, "angle")
    assert angle["gap_finite"] == 1
    assert angle["n_steps"] == 2


def test_band_report_flags_a_degenerate_positive_band_for_angle():
    """A band that is positive but numerically negligible makes the ratio
    meaningless -- measured, it produced values like 125.2 and 9.06."""
    result = _result(
        pos=[1.0], pers_pos=[2.0], floor_pos=[0.0],
        ang=[1.0], pers_ang=[2.0], floor_ang=[2.0 - 1e-9],
    )
    assert script.band_report(result, "angle")["steps_degenerate"] == 1


def test_band_report_rejects_an_unknown_metric():
    result = _result([1.0], [2.0], [0.0], [1.0], [2.0], [0.0])
    with pytest.raises(ValueError, match="position.*angle"):
        script.band_report(result, "velocity")


def test_print_report_emits_both_metrics(capsys):
    """The angle block must carry the same guards as the position block."""
    result = _result(
        pos=[1.0, 1.0], pers_pos=[2.0, 2.0], floor_pos=[0.0, 0.0],
        ang=[1.0, 1.0], pers_ang=[2.0, 2.0], floor_ang=[0.0, 0.0],
    )
    script.print_report(
        result,
        {"position": script.band_report(result, "position"),
         "angle": script.band_report(result, "angle")},
    )
    out = capsys.readouterr().out
    for metric in ("position", "angle"):
        assert f"{metric}_band_width" in out
        assert f"{metric}_gap_closed" in out
        assert f"{metric}_steps_with_floor_above_persistence" in out
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/eval/test_eval_rollout_script.py -q`
Expected: FAIL — `band_report()` takes 1 positional argument but 2 were given.

- [ ] **Step 3: Implement**

Put the computation in `src/mbfps/eval/summary.py`. Its module docstring should say
that both the eval script and the study job consume it, and that keeping two copies
is how the degeneracy threshold drifts between a single run and the nine-cell study.
The module defines `METRICS = ("position", "angle")`, `DEGENERATE = 1e-3`, and
`metric_summary` exactly as below.

`DEGENERATE`'s docstring must record why it exists: measured on an untrained model
with a poor probe, a band ~1e-4 wide produced `gap_closed` values of 125.2, 9.06 and
-0.68. A non-positive band already returns NaN; a numerically degenerate POSITIVE one
does not, and is counted here so no consumer reports the ratio as if it meant
something.

Then in `scripts/eval_rollout.py`, import it and replace `band_report`/`print_report`:

```python
from mbfps.eval.summary import DEGENERATE, METRICS, metric_summary

# Thin alias so the script reads naturally; the maths lives in the library
# because Task 4's study job needs the identical computation.
band_report = metric_summary


# --- the body of metric_summary, for reference; it goes in summary.py ---
def metric_summary(result, metric: str) -> dict:
    """Band statistics and degeneracy counts for one metric.

    `metric` is "position" or "angle". Both get the same guards: an earlier
    version computed all of this for position only and printed the angle ratio
    bare, while that curve ranged +8.68 to -23.04 with a NaN step in it.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    model = getattr(result, f"rssm_{metric}")
    persistence = getattr(result, f"persistence_{metric}")
    floor = getattr(result, f"floor_{metric}")
    gap = getattr(result, f"{metric}_gap_closed")()

    band = persistence - floor
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(persistence == 0, np.nan, band / persistence)
    finite = np.isfinite(gap)
    return {
        "final_model": float(model[-1]),
        "final_persistence": float(persistence[-1]),
        "final_floor": float(floor[-1]),
        "band_min": float(band.min()),
        "band_median": float(np.median(band)),
        "band_max": float(band.max()),
        "relative_min": float(np.nanmin(relative)),
        "relative_median": float(np.nanmedian(relative)),
        "relative_max": float(np.nanmax(relative)),
        "steps_floor_above_persistence": int((band <= 0).sum()),
        "steps_degenerate": int(((band > 0) & (relative < DEGENERATE)).sum()),
        "gap_finite": int(finite.sum()),
        "gap_mean": float(np.nanmean(gap)) if finite.any() else float("nan"),
        "gap_min": float(np.nanmin(gap)) if finite.any() else float("nan"),
        "gap_max": float(np.nanmax(gap)) if finite.any() else float("nan"),
        "gap_final": float(gap[-1]),
        "n_steps": int(len(gap)),
    }


def print_report(result, reports: dict) -> None:
    """Print both metrics with identical guards."""
    units = {"position": "Doom map units", "angle": "degrees"}
    for metric in METRICS:
        r = reports[metric]
        print(f"\n--- {metric} ({units[metric]}) ---")
        print(f"{metric}_final          rssm={r['final_model']:9.2f} "
              f"persistence={r['final_persistence']:9.2f} floor={r['final_floor']:9.2f}")
        print(f"{metric}_band_width     min={r['band_min']:9.3f} "
              f"median={r['band_median']:9.3f} max={r['band_max']:9.3f}")
        print(f"{metric}_band_relative  min={r['relative_min']:.5f} "
              f"median={r['relative_median']:.5f} max={r['relative_max']:.5f}")
        print(f"{metric}_steps_with_floor_above_persistence="
              f"{r['steps_floor_above_persistence']}/{r['n_steps']}")
        print(f"{metric}_steps_with_degenerate_positive_band="
              f"{r['steps_degenerate']}/{r['n_steps']}  (band < {DEGENERATE} x persistence)")
        print(f"{metric}_gap_closed     finite={r['gap_finite']}/{r['n_steps']} "
              f"mean={r['gap_mean']:+.4f} min={r['gap_min']:+.4f} max={r['gap_max']:+.4f} "
              f"final={r['gap_final']:+.4f}")
        if r["gap_finite"] == 0:
            print(f"WARNING: {metric} gap_closed is NaN at every horizon step -- the band "
                  "is non-positive throughout, so this metric says nothing here. Report "
                  "the raw curves instead.")
        elif r["steps_degenerate"]:
            print(f"WARNING: {metric} band is numerically degenerate at "
                  f"{r['steps_degenerate']}/{r['n_steps']} steps; the ratio is unstable "
                  "there and the raw curves should be read directly.")
```

Update the call site at the bottom of `main()`:

```python
    reports = {m: metric_summary(result, m) for m in METRICS}
    print_report(result, reports)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/eval/test_eval_rollout_script.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Mutation-test**

Set up a self-checked harness (see Global Constraints), then:

| mutation | must be caught by |
|---|---|
| `band_report` ignores `metric` and always reads position | `test_band_report_accepts_a_metric_and_reads_the_matching_curves` |
| `steps_degenerate` computed as `(band < DEGENERATE)` (absolute, not relative) | `test_band_report_flags_a_degenerate_positive_band_for_angle` |
| `print_report` loops over `("position",)` only | `test_print_report_emits_both_metrics` |
| drop the `metric not in METRICS` guard | `test_band_report_rejects_an_unknown_metric` |

Any mutation that survives is a missing test. Add it before committing.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_rollout.py tests/eval/test_eval_rollout_script.py
git commit -m "fix: angle gets the same degeneracy guards as position"
```

---

## Task 2: Fit the probes under the rollout's own protocol

`fit_probes` filters each training episode **whole**, so `h` carries ~500 steps of history. `evaluate_rollout` then applies that probe to latents produced from **5** context steps out of a zero state. Measured on the shipped checkpoint, same frames and same probe: position error **222.4** with long context against **247.5** with short.

The bias is not common-mode across the three references. Persistence is frozen at context 5 forever while the floor's context grows to 50, and probe error by context length is 317.3 / 223.4 / 213.5 / 215.3 at contexts 1 / 5 / 20 / 50 — so the floor gains roughly 8 units from filtering history alone. This is the same defect class as the real-versus-predicted embedding confound, on the context-length axis.

**Files:**
- Modify: `src/mbfps/eval/probe.py`
- Test: `tests/eval/test_probe.py` (append)

**Interfaces:**
- Consumes: `fit_probe`, `probe_r2`, `probe_targets` (already in this module); `source_for` from `mbfps.eval.rollout`; `load_episode` from `mbfps.data.episode`.
- Produces:
  - `gather_probe_data(model, paths, backbone, device, context, horizon, limit=20, seed=0) -> dict` with keys `"latent"`, `"embedding"`, `"targets"`, each an `np.ndarray` whose rows are aligned. Latents come from the SAME observe-then-imagine protocol `evaluate_rollout` uses.
  - `fit_probes(model, paths, backbone, device, context=5, horizon=45, limit=20, seed=0, select_episodes=4, ridge=None) -> tuple[dict, dict]` — unchanged return, new `context`/`horizon` arguments.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_probe.py  (append)
from mbfps.eval.probe import gather_probe_data


class _CountingModel:
    """Records the context length every `observe` call receives."""

    input_kind = "features"

    def __init__(self):
        self.observed_lengths = []
        self.imagined_lengths = []

    class _RSSM:
        def __init__(self, outer):
            self.outer = outer

        def observe(self, embeddings, actions, state=None):
            self.outer.observed_lengths.append(int(embeddings.shape[1]))
            b, t, _ = embeddings.shape
            latent = torch.arange(b * t * 4, dtype=torch.float32).view(b, t, 4)
            return {"latent": latent, "h": latent[..., :2], "z": latent[..., 2:]}

        def imagine(self, actions, state):
            self.outer.imagined_lengths.append(int(actions.shape[1]))
            b, t = actions.shape
            latent = torch.zeros(b, t, 4)
            return {"latent": latent, "h": latent[..., :2], "z": latent[..., 2:]}

    def __getattr__(self, name):
        if name == "rssm":
            return _CountingModel._RSSM(self)
        raise AttributeError(name)


def test_probe_data_is_gathered_under_the_rollout_context_length(monkeypatch, tmp_path):
    """The probe must be fit on latents produced the way the rollout produces them.

    Filtering a whole episode gives `h` ~500 steps of history; the rollout gives
    it 5. Measured, that is a 222.4-vs-247.5 position-error difference on the
    same frames with the same probe -- and it is NOT common-mode, because
    persistence stays at context 5 while the floor's context grows to 50.
    """
    from mbfps.eval import probe as probe_module

    model = _CountingModel()
    monkeypatch.setattr(probe_module, "_gather_episode", _fake_gather)
    gather_probe_data(model, [tmp_path / "ep.npz"], None, torch.device("cpu"),
                      context=5, horizon=45, limit=1)
    assert model.observed_lengths, "observe was never called"
    assert set(model.observed_lengths) == {5}, (
        f"probe data was filtered with context lengths {set(model.observed_lengths)}, "
        "not the rollout's 5 -- the probe would be fit on a different distribution "
        "from the one it is applied to"
    )
```

> The helper `_fake_gather` and the fixture episode are supplied in Step 3 alongside
> the implementation, because the test needs the exact loader shape the module uses.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -q -k probe_data`
Expected: FAIL with `ImportError: cannot import name 'gather_probe_data'`

- [ ] **Step 3: Implement**

Extract the gathering out of `fit_probes` and make it follow the rollout protocol:

```python
# src/mbfps/eval/probe.py  (replace the body of fit_probes with the two below)

@torch.no_grad()
def gather_probe_data(
    model,
    paths,
    backbone,
    device,
    context: int = 5,
    horizon: int = 45,
    limit: int = 20,
    seed: int = 0,
) -> dict:
    """Collect probe training data using the ROLLOUT's own window protocol.

    Every window is `context` real frames filtered from a ZERO state, then
    `horizon` imagined steps -- exactly what `evaluate_rollout` does. Filtering
    a whole episode instead gives `h` hundreds of steps of history that the
    rollout never has, and the probe is then fit on a distribution it is never
    applied to. Measured on one checkpoint: position error 222.4 under
    whole-episode filtering against 247.5 under the rollout's 5-step context,
    and the bias is not common-mode -- persistence stays at context 5 while the
    floor's context grows to `context + horizon`.

    Returns `{"latent": (N, LATENT), "embedding": (N, EMBED), "targets": (N, 4)}`
    with rows aligned. Seeds the global RNG, because the posterior samples.
    """
    from mbfps.data.episode import load_episode
    from mbfps.eval.rollout import source_for

    torch.manual_seed(seed)
    need = context + horizon
    latents, embeddings, targets = [], [], []

    for path in list(paths)[:limit]:
        episode = load_episode(path)
        if episode.length < need + 1:
            continue
        source = source_for(model, path, episode, backbone)
        all_actions = torch.as_tensor(
            episode.actions.astype(np.int64)
        ).unsqueeze(0).to(device)

        for start in range(0, episode.length - need + 1, need):
            frames = torch.as_tensor(source[start : start + need + 1]).to(device)
            window_embeddings = model.encoder(frames).unsqueeze(0)[:, 1:]
            actions = all_actions[:, start : start + need]

            observed = model.rssm.observe(
                window_embeddings[:, :context], actions[:, :context]
            )
            state = (observed["h"][:, -1], observed["z"][:, -1])
            future = model.rssm.observe(
                window_embeddings[:, context:], actions[:, context:], state=state
            )
            latent = torch.cat([observed["latent"], future["latent"]], dim=1)
            predicted = model.heads(latent)["embedding"]

            latents.append(latent[0].cpu().numpy())
            embeddings.append(predicted[0].cpu().numpy())
            targets.append(
                probe_targets(
                    episode.privileged[start + 1 : start + need + 1],
                    episode.privileged_keys,
                )
            )

    if not latents:
        raise ValueError(
            f"no training window reached {need + 1} frames; lower context/horizon"
        )
    return {
        "latent": np.concatenate(latents),
        "embedding": np.concatenate(embeddings),
        "targets": np.concatenate(targets),
    }


def fit_probes(
    model,
    paths,
    backbone,
    device,
    context: int = 5,
    horizon: int = 45,
    limit: int = 20,
    seed: int = 0,
    select_episodes: int = 4,
    ridge: float | None = None,
) -> tuple[dict, dict]:
    """Fit both probes on TRAINING episodes, under the rollout's protocol.

    Returns `(latent_probe, embedding_probe)`. `embedding_probe` is THE probe
    `evaluate_rollout` takes, fit on the model's OWN PREDICTED embeddings;
    `latent_probe` is a diagnostic on the 1536-d posterior latent, reported
    alongside and never used for the band.
    """
    paths = list(paths)
    n_select = min(select_episodes, max(1, len(paths[:limit]) // 4))
    fit_paths, select_paths = paths[: limit - n_select], paths[limit - n_select : limit]

    train = gather_probe_data(model, fit_paths, backbone, device, context, horizon,
                              limit=limit, seed=seed)
    if ridge is not None or not select_paths:
        return (
            fit_probe(train["latent"], train["targets"], ridge=ridge),
            fit_probe(train["embedding"], train["targets"], ridge=ridge),
        )
    select = gather_probe_data(model, select_paths, backbone, device, context, horizon,
                               limit=len(select_paths), seed=seed + 1)
    return (
        fit_probe(train["latent"], train["targets"],
                  select["latent"], select["targets"]),
        fit_probe(train["embedding"], train["targets"],
                  select["embedding"], select["targets"]),
    )
```

Add the test helper the Step 1 test needs, at the top of the appended test block:

```python
# tests/eval/test_probe.py  (add above the new test)
def _fake_gather(*args, **kwargs):
    raise AssertionError("unused; present so monkeypatch has a target")
```

and replace the monkeypatch line in the test with a real single-episode fixture:

```python
    from mbfps.data.buffer import ReplayBuffer
    from tests.data.test_loader import make_episode

    buffer = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buffer.add(make_episode(t=80, fill=1))
    path = buffer.episode_paths()[0]
    model.encoder = lambda frames: torch.zeros(len(frames), 2048)
    model.heads = lambda latent: {"embedding": torch.zeros(*latent.shape[:2], 2048)}
    gather_probe_data(model, [path], None, torch.device("cpu"),
                      context=5, horizon=45, limit=1)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -q`
Expected: PASS

- [ ] **Step 5: Verify the fix empirically on the real checkpoint**

```bash
caffeinate -dimsu .venv/bin/python - <<'EOF'
import numpy as np, torch
from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import episode_split
from mbfps.eval.probe import fit_probes, probe_r2, gather_probe_data
from mbfps.models.encoders import encoder_backbone
from mbfps.training.world_model import WorldModel
from mbfps.utils.config import get_config
from mbfps.utils.device import get_device

cfg = get_config("random_vit", device="cpu", seed=0)
device = get_device(prefer="cpu")
model = WorldModel(cfg).to(device)
model.load_state_dict(torch.load("runs/m3_20k/world_model_random_vit_seed0.pt",
                                 map_location=device, weights_only=True)["state_dict"])
model.eval()
buf = ReplayBuffer("data/my_way_home", capacity_transitions=10**9)
train, val = episode_split(buf.episode_paths(), val_fraction=0.2, seed=0)
bk = encoder_backbone(cfg.encoder)

_, emb = fit_probes(model, train, bk, device)
held = gather_probe_data(model, val, bk, device, limit=6, seed=99)
print(f"embedding probe held-out R^2 under matched protocol: "
      f"{probe_r2(emb, held['embedding'], held['targets']):+.4f}")
EOF
```

Record the number in "## Task 2 results" at the bottom of this file. The
pre-fix comparison point is that the probe was fit on whole-episode filtering
and scored ~0.29 on its own mismatched selection set.

- [ ] **Step 6: Mutation-test**

| mutation | must be caught by |
|---|---|
| `gather_probe_data` filters the whole episode (`observe(window_embeddings, actions)`) | `test_probe_data_is_gathered_under_the_rollout_context_length` |
| the `start + 1 : start + need + 1` target slice loses its `+1` | add a test asserting targets align with the latents' frames |
| `fit_probes` passes `select` data as the FIT data | `test_ridge_selection_scores_on_held_out_data_not_training_data` — verify, and add one if it does not fire |

- [ ] **Step 7: Commit**

```bash
git add src/mbfps/eval/probe.py tests/eval/test_probe.py
git commit -m "fix: fit probes under the rollout's own context protocol"
```

---

## Task 3: Wire the filtering comparison so the gate criterion can be produced

`filtering_comparison` is implemented and unit-tested but called from nothing in `src/` or `scripts/` — so spec §4 gate criterion 4 ("filtering probe beats the encoder-embedding probe, showing `h` carries history") cannot be produced at all. Task 2's `gather_probe_data` now returns exactly the arrays it needs.

**Files:**
- Modify: `src/mbfps/eval/probe.py`, `scripts/eval_rollout.py`
- Test: `tests/eval/test_probe.py` (append)

**Interfaces:**
- Consumes: `gather_probe_data` (Task 2), `filtering_comparison` (existing).
- Produces: `filtering_report(model, train_paths, val_paths, backbone, device, context=5, horizon=45, limit=20, seed=0) -> dict` with keys `"latent_r2"`, `"embedding_r2"`, `"latent_beats_embedding"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_probe.py  (append)
from mbfps.eval.probe import filtering_report


def test_filtering_report_returns_the_gate_criterion_keys(monkeypatch):
    """Spec section 4 criterion 4 cannot be reported unless something calls this."""
    from mbfps.eval import probe as probe_module

    rng = np.random.default_rng(0)
    n = 300
    signal = rng.normal(size=n)
    lagged = np.roll(signal, 1)
    embedding = signal[:, None] * np.ones((1, 4))
    latent = np.column_stack([embedding, lagged[:, None] * np.ones((1, 4))])
    targets = np.column_stack([lagged] * 4) + 0.01 * rng.normal(size=(n, 4))

    calls = []

    def fake_gather(model, paths, backbone, device, context=5, horizon=45,
                    limit=20, seed=0):
        calls.append(list(paths))
        half = n // 2
        sl = slice(0, half) if len(calls) == 1 else slice(half, n)
        return {"latent": latent[sl], "embedding": embedding[sl], "targets": targets[sl]}

    monkeypatch.setattr(probe_module, "gather_probe_data", fake_gather)
    out = filtering_report(object(), ["train"], ["val"], None, torch.device("cpu"))

    assert set(out) == {"latent_r2", "embedding_r2", "latent_beats_embedding"}
    assert out["latent_beats_embedding"] is True
    assert calls == [["train"], ["val"]], "train and val must be gathered separately"


def test_filtering_report_does_not_gather_train_and_val_together(monkeypatch):
    """Fitting and scoring on the same rows would make the criterion vacuous."""
    from mbfps.eval import probe as probe_module

    seen = []

    def fake_gather(model, paths, backbone, device, **kwargs):
        seen.append(tuple(paths))
        k = 200
        rng = np.random.default_rng(len(seen))
        return {"latent": rng.normal(size=(k, 6)),
                "embedding": rng.normal(size=(k, 4)),
                "targets": rng.normal(size=(k, 4))}

    monkeypatch.setattr(probe_module, "gather_probe_data", fake_gather)
    filtering_report(object(), ["a", "b"], ["c"], None, torch.device("cpu"))
    assert seen == [("a", "b"), ("c",)]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -q -k filtering_report`
Expected: FAIL with `ImportError: cannot import name 'filtering_report'`

- [ ] **Step 3: Implement**

```python
# src/mbfps/eval/probe.py  (append)
def filtering_report(
    model,
    train_paths,
    val_paths,
    backbone,
    device,
    context: int = 5,
    horizon: int = 45,
    limit: int = 20,
    seed: int = 0,
) -> dict:
    """Spec section 4, gate criterion 4: does the deterministic state carry history?

    The posterior has already seen frame t, so by the data-processing inequality
    it cannot add information about t over the embedding of t. What it can add
    is memory. If the latent probe does not beat the embedding probe here, `h`
    is inert -- a specific, actionable bug nothing else in the harness surfaces.

    Train and validation windows are gathered SEPARATELY and under the rollout's
    own context protocol, so neither probe is fit on rows it is scored on.
    """
    train = gather_probe_data(model, train_paths, backbone, device, context, horizon,
                              limit=limit, seed=seed)
    val = gather_probe_data(model, val_paths, backbone, device, context, horizon,
                            limit=limit, seed=seed + 1)
    return filtering_comparison(
        train["latent"], train["embedding"],
        val["latent"], val["embedding"],
        train["targets"], val["targets"],
    )
```

Call it from `scripts/eval_rollout.py`, after the rollout report:

```python
    from mbfps.eval.probe import filtering_report

    filtering = filtering_report(model, train, val, backbone, device,
                                 context=args.context, horizon=args.horizon)
    print("\n--- filtering probe (spec section 4, criterion 4) ---")
    print(f"latent_r2={filtering['latent_r2']:+.4f} "
          f"embedding_r2={filtering['embedding_r2']:+.4f}")
    print(f"latent_beats_embedding={filtering['latent_beats_embedding']}")
    if not filtering["latent_beats_embedding"]:
        print("WARNING: the deterministic state h adds nothing over the embedding at "
              "the same timestep -- it is carrying no history.")
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/eval/test_probe.py -q`
Expected: PASS

- [ ] **Step 5: Run it on the real 20k checkpoint**

```bash
caffeinate -dimsu .venv/bin/python scripts/eval_rollout.py --arm random_vit \
  --checkpoint runs/m3_20k/world_model_random_vit_seed0.pt --device cpu
```

Record `latent_r2`, `embedding_r2` and the verdict in "## Task 3 results" below.
A `False` verdict is a REAL RESULT to record, not something to tune away.

- [ ] **Step 6: Mutation-test**

| mutation | must be caught by |
|---|---|
| `filtering_report` gathers train and val in one call and splits the rows | `test_filtering_report_does_not_gather_train_and_val_together` |
| `filtering_report` passes `val` arrays as the train arguments | `test_filtering_report_returns_the_gate_criterion_keys` |
| the script never calls `filtering_report` | add an assertion in `tests/eval/test_eval_rollout_script.py` that `main`'s output contains `latent_beats_embedding` |

- [ ] **Step 7: Commit**

```bash
git add src/mbfps/eval/probe.py scripts/eval_rollout.py tests/eval/
git commit -m "feat: wire the filtering comparison so gate criterion 4 is produced"
```

---

## Task 4: One study job — train, evaluate, emit a record

**Files:**
- Create: `src/mbfps/eval/study.py`
- Test: `tests/eval/test_study.py`

**Interfaces:**
- Consumes: `train_world_model`, `WorldModel`, `fit_probes`, `filtering_report`, `evaluate_rollout`, `episode_split`, `encoder_backbone`, `band_report` logic (re-implemented here so the library does not import a script).
- Produces:
  - `StudyJob(arm: str, seed: int)` dataclass
  - `reward_accuracy(model, val_paths, backbone, device, limit=20, seed=0) -> dict` with keys `"mse"`, `"baseline_mse"`, `"r2"`, `"n_steps"`, `"n_reward_events"`, `"is_degenerate"`
  - `run_job(job, buffer, out_dir, steps=20_000, seq_len=64, context=5, horizon=45, device="mps") -> dict` — the result record
  - `job_record_path(out_dir, job) -> Path`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_study.py
import json
from pathlib import Path

import numpy as np
import pytest

from mbfps.eval.study import StudyJob, job_record_path, run_job


def test_job_record_path_is_unique_per_arm_and_seed(tmp_path):
    a = job_record_path(tmp_path, StudyJob("cnn", 0))
    b = job_record_path(tmp_path, StudyJob("cnn", 1))
    c = job_record_path(tmp_path, StudyJob("frozen_ssl", 0))
    assert len({a, b, c}) == 3
    assert a.suffix == ".json"


def test_record_carries_everything_the_gate_needs(tmp_path, small_buffer):
    """A record missing a field silently drops one cell of the 3x3 table."""
    record = run_job(
        StudyJob("random_vit", 0), small_buffer, tmp_path,
        steps=3, seq_len=4, context=2, horizon=3, device="cpu",
    )
    for key in ("arm", "seed", "steps", "seconds", "steps_per_second",
                "kl_rate_above_free_bits", "position", "angle", "filtering",
                "reward", "curves"):
        assert key in record, f"record is missing {key!r}"
    for metric in ("position", "angle"):
        for key in ("gap_final", "gap_mean", "band_median",
                    "steps_floor_above_persistence", "steps_degenerate", "gap_finite"):
            assert key in record[metric], f"{metric} is missing {key!r}"
    assert set(record["filtering"]) == {
        "latent_r2", "embedding_r2", "latent_beats_embedding"}


def test_reward_accuracy_reports_its_own_degeneracy(small_buffer):
    """Spec 4.3 asks for reward accuracy, but this scenario's reward is
    near-constant: 19,417 of 19,424 real steps share one value and six carry the
    goal. A bare MSE would look precise and mean nothing, so the report must
    carry the baseline and the event count and flag the degeneracy itself."""
    import torch

    from mbfps.data.split import episode_split
    from mbfps.eval.study import reward_accuracy
    from mbfps.models.encoders import encoder_backbone
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    cfg = get_config("random_vit", device="cpu", seed=0)
    model = WorldModel(cfg)
    model.eval()
    _, val = episode_split(small_buffer.episode_paths(), val_fraction=0.2, seed=0)
    out = reward_accuracy(model, val, encoder_backbone(cfg.encoder),
                          torch.device("cpu"), limit=2)

    assert set(out) == {"mse", "baseline_mse", "r2", "n_steps",
                        "n_reward_events", "is_degenerate"}
    assert out["n_steps"] > 0
    assert out["baseline_mse"] >= 0.0
    assert isinstance(out["is_degenerate"], bool)


def test_reward_accuracy_flags_a_constant_reward_as_degenerate():
    """A constant target makes R^2 undefined; the flag is what stops a caller
    reporting a confident-looking number over it."""
    import numpy as np

    from mbfps.eval.study import _summarise_reward

    constant = _summarise_reward(np.zeros(50), np.zeros(50))
    assert constant["is_degenerate"] is True
    assert constant["n_reward_events"] == 0

    varied = _summarise_reward(np.arange(50, dtype=float), np.arange(50, dtype=float))
    assert varied["is_degenerate"] is False
    assert varied["r2"] > 0.99


def test_record_round_trips_through_json(tmp_path, small_buffer):
    """The driver writes these to disk; a numpy float would not survive."""
    record = run_job(
        StudyJob("random_vit", 0), small_buffer, tmp_path,
        steps=3, seq_len=4, context=2, horizon=3, device="cpu",
    )
    path = job_record_path(tmp_path, StudyJob("random_vit", 0))
    assert path.is_file()
    assert json.loads(path.read_text())["arm"] == "random_vit"


def test_two_runs_of_the_same_job_agree(tmp_path, small_buffer):
    """The whole study rests on this: same arm, same seed, same numbers."""
    kw = dict(steps=3, seq_len=4, context=2, horizon=3, device="cpu")
    first = run_job(StudyJob("random_vit", 0), small_buffer, tmp_path / "a", **kw)
    second = run_job(StudyJob("random_vit", 0), small_buffer, tmp_path / "b", **kw)
    assert first["position"]["gap_final"] == second["position"]["gap_final"]
    assert first["curves"]["rssm_position"] == second["curves"]["rssm_position"]


def test_different_seeds_give_different_results(tmp_path, small_buffer):
    kw = dict(steps=3, seq_len=4, context=2, horizon=3, device="cpu")
    a = run_job(StudyJob("random_vit", 0), small_buffer, tmp_path / "a", **kw)
    b = run_job(StudyJob("random_vit", 1), small_buffer, tmp_path / "b", **kw)
    assert a["curves"]["rssm_position"] != b["curves"]["rssm_position"]
```

Add the shared fixture:

```python
# tests/eval/conftest.py
import numpy as np
import pytest

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.envs.protocol import OBS_SHAPE

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


@pytest.fixture
def small_buffer(tmp_path):
    """Six episodes long enough for context+horizon, with cached features.

    Six, not four: `episode_split(val_fraction=0.2)` floors to zero validation
    episodes below five and raises.
    """
    buf = ReplayBuffer(tmp_path / "data", capacity_transitions=100_000)
    for fill in range(1, 7):
        t = 40
        steps = np.arange(t)
        obs = np.zeros((t + 1, *OBS_SHAPE), dtype=np.uint8)
        obs[:, 0, 0, 0] = np.arange(t + 1, dtype=np.uint8)
        obs[:, 0, 0, 1] = fill
        buf.add(Episode(
            obs=obs,
            actions=(steps % 6).astype(np.int32),
            rewards=(steps % 3).astype(np.float32),
            terminated=(steps == t - 1),
            truncated=np.zeros(t, dtype=bool),
            privileged=np.stack([
                np.full(t + 1, 100.0),
                np.arange(t + 1, dtype=np.float32) * 3.0,
                np.arange(t + 1, dtype=np.float32) * -2.0,
                np.zeros(t + 1),
                (np.arange(t + 1) * 7.0) % 360.0,
            ], axis=1).astype(np.float32),
            privileged_keys=KEYS,
            policy_name="random",
            seed=fill,
            scenario="my_way_home",
        ))
    rng = np.random.default_rng(0)
    for path in buf.episode_paths():
        for suffix in (".features.npy", ".features_random_vit.npy"):
            np.save(path.with_suffix(suffix),
                    rng.random((41, 64, 384)).astype(np.float16))
    return buf
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/eval/test_study.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.eval.study'`

- [ ] **Step 3: Implement**

```python
# src/mbfps/eval/study.py
"""One cell of the 3 arms x 3 seeds study.

A job trains one arm at one seed, evaluates it, and emits a JSON-serialisable
record. The driver in `scripts/run_study.py` is resumable off these records, so
a job that has already produced one is never re-run -- which matters when the
pixel arm costs 8.3 h per seed against the feature arms' 1.4 h.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import episode_split
from mbfps.eval.probe import filtering_report, fit_probes
from mbfps.eval.rollout import evaluate_rollout
from mbfps.eval.summary import metric_summary
from mbfps.models.encoders import encoder_backbone
from mbfps.training.world_model import WorldModel, train_world_model
from mbfps.utils.config import get_config
from mbfps.utils.device import get_device

@dataclass(frozen=True)
class StudyJob:
    arm: str
    seed: int


def job_record_path(out_dir: Path, job: StudyJob) -> Path:
    return Path(out_dir) / f"result_{job.arm}_seed{job.seed}.json"


DEGENERATE_REWARD_FRACTION = 0.99
"""If this share of steps carry the modal reward, the target is degenerate.

Measured on `my_way_home`: 19,417 of 19,424 steps share the living penalty and
six carry the goal, so the modal share is 0.9996. A scalar accuracy over that
is dominated by the constant.
"""


def _summarise_reward(predicted: np.ndarray, true: np.ndarray) -> dict:
    """Reward error beside the baseline that makes it readable.

    `r2` is NaN when the target has no variance -- reporting 0.0 there would
    read as "explains nothing" when the truth is "there was nothing to explain".
    """
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    denominator = float(((true - true.mean()) ** 2).sum())
    mse = float(((predicted - true) ** 2).mean())
    baseline = float(((true - true.mean()) ** 2).mean())
    modal_share = float(np.bincount(
        np.unique(np.round(true, 6), return_inverse=True)[1]).max() / true.size)
    return {
        "mse": mse,
        "baseline_mse": baseline,
        "r2": float("nan") if denominator == 0.0 else 1.0 - (mse * true.size) / denominator,
        "n_steps": int(true.size),
        "n_reward_events": int((np.round(true, 6) != np.round(np.median(true), 6)).sum()),
        "is_degenerate": bool(modal_share >= DEGENERATE_REWARD_FRACTION),
    }


@torch.no_grad()
def reward_accuracy(model, val_paths, backbone, device, limit: int = 20,
                    seed: int = 0) -> dict:
    """Spec section 4, criterion 3: reward-prediction accuracy per arm.

    Filters each held-out episode and compares the reward head's output against
    the recorded reward. Reported with its baseline and event count because
    `my_way_home`'s reward is near-constant -- see DEGENERATE_REWARD_FRACTION.
    """
    from mbfps.data.episode import load_episode
    from mbfps.eval.rollout import source_for

    torch.manual_seed(seed)
    predicted, true = [], []
    for path in list(val_paths)[:limit]:
        episode = load_episode(path)
        source = source_for(model, path, episode, backbone)
        embeddings = model.encoder(torch.as_tensor(source).to(device)).unsqueeze(0)[:, 1:]
        actions = torch.as_tensor(
            episode.actions.astype(np.int64)).unsqueeze(0).to(device)
        out = model.rssm.observe(embeddings, actions)
        predicted.append(model.heads(out["latent"])["reward"][0].cpu().numpy())
        true.append(episode.rewards)
    if not predicted:
        raise ValueError("no validation episode produced reward predictions")
    return _summarise_reward(np.concatenate(predicted), np.concatenate(true))


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
    """Train, evaluate, and write one result record. Returns the record."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = get_config(job.arm, steps=steps, seq_len=seq_len, seed=job.seed, device=device)
    torch_device = get_device(prefer=device)

    started = time.perf_counter()
    history = train_world_model(cfg, buffer, out_dir=out_dir)
    train_paths, val_paths = episode_split(buffer.episode_paths(),
                                           val_fraction=0.2, seed=0)
    backbone = encoder_backbone(cfg.encoder)

    model = WorldModel(cfg).to(torch_device)
    checkpoint = torch.load(
        out_dir / f"world_model_{job.arm}_seed{job.seed}.pt",
        map_location=torch_device, weights_only=True,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    _, embedding_probe = fit_probes(model, train_paths, backbone, torch_device,
                                    context=context, horizon=horizon, seed=job.seed)
    reward = reward_accuracy(model, val_paths, backbone, torch_device, seed=job.seed)
    result = evaluate_rollout(model, val_paths, embedding_probe, context=context,
                              horizon=horizon, seed=job.seed, device=torch_device,
                              feature_backbone=backbone)
    filtering = filtering_report(model, train_paths, val_paths, backbone, torch_device,
                                 context=context, horizon=horizon, seed=job.seed)

    record = {
        "arm": job.arm,
        "seed": job.seed,
        "steps": steps,
        "seq_len": seq_len,
        "context": context,
        "horizon": horizon,
        "seconds": float(time.perf_counter() - started),
        "steps_per_second": float(history["steps"] / history["seconds"]),
        "kl_rate_above_free_bits": float(history["kl_rate_above_free_bits"]),
        "loss_last20": float(np.mean(history["loss"][-20:])),
        "position": metric_summary(result, "position"),
        "angle": metric_summary(result, "angle"),
        "filtering": filtering,
        "reward": reward,
        "curves": {
            name: [float(v) for v in getattr(result, name)]
            for name in ("rssm_position", "persistence_position", "floor_position",
                         "rssm_angle", "persistence_angle", "floor_angle")
        },
    }
    job_record_path(out_dir, job).write_text(json.dumps(record, indent=2))
    return record
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/eval/test_study.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Mutation-test**

| mutation | must be caught by |
|---|---|
| `job_record_path` ignores `seed` | `test_job_record_path_is_unique_per_arm_and_seed` |
| `run_job` passes `seed=0` to `evaluate_rollout` regardless of the job | `test_different_seeds_give_different_results` |
| `metric_summary` reads position curves regardless of `metric` | Task 1's `test_band_report_accepts_a_metric_and_reads_the_matching_curves` |
| the record omits `filtering` | `test_record_carries_everything_the_gate_needs` |
| `_summarise_reward` returns `r2=0.0` instead of NaN for a constant target | `test_reward_accuracy_flags_a_constant_reward_as_degenerate` |
| `is_degenerate` hardcoded to False | `test_reward_accuracy_flags_a_constant_reward_as_degenerate` |

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/eval/study.py tests/eval/test_study.py tests/eval/conftest.py
git commit -m "feat: one study job emits a complete, JSON-round-tripping record"
```

---

## Task 5: Resumable driver over the nine jobs

The pixel arm costs 8.3 h per seed against the feature arms' 1.4 h, so a driver that restarts from zero after an interruption can lose most of a day. It must skip jobs whose record already exists.

**Files:**
- Create: `scripts/run_study.py`
- Test: `tests/eval/test_run_study.py`

**Interfaces:**
- Consumes: `StudyJob`, `run_job`, `job_record_path` from `mbfps.eval.study`.
- Produces: `pending_jobs(out_dir, arms, seeds) -> list[StudyJob]`, ordered cheapest-arm-first.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_run_study.py
import importlib.util
import json
from pathlib import Path

from mbfps.eval.study import StudyJob, job_record_path

_SPEC = importlib.util.spec_from_file_location(
    "run_study", Path(__file__).resolve().parents[2] / "scripts" / "run_study.py")
run_study = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_study)

ARMS = ("cnn", "frozen_ssl", "random_vit")
SEEDS = (0, 1, 2)


def test_all_nine_jobs_are_pending_on_a_fresh_directory(tmp_path):
    jobs = run_study.pending_jobs(tmp_path, ARMS, SEEDS)
    assert len(jobs) == 9
    assert len({(j.arm, j.seed) for j in jobs}) == 9


def test_a_job_with_an_existing_record_is_skipped(tmp_path):
    """Resumability. The pixel arm is 8.3 h per seed; re-running it is a lost day."""
    done = StudyJob("cnn", 1)
    job_record_path(tmp_path, done).write_text(json.dumps({"arm": "cnn", "seed": 1}))
    jobs = run_study.pending_jobs(tmp_path, ARMS, SEEDS)
    assert len(jobs) == 8
    assert done not in jobs


def test_cheap_arms_run_first(tmp_path):
    """Feature arms are 5.9x faster; running them first means an interruption
    still leaves the treatment/control contrast complete."""
    jobs = run_study.pending_jobs(tmp_path, ARMS, SEEDS)
    cnn_positions = [i for i, j in enumerate(jobs) if j.arm == "cnn"]
    other_positions = [i for i, j in enumerate(jobs) if j.arm != "cnn"]
    assert min(cnn_positions) > max(other_positions)


def test_a_corrupt_record_is_treated_as_pending(tmp_path):
    """A truncated JSON file from a killed process must not mark a job done."""
    job_record_path(tmp_path, StudyJob("cnn", 0)).write_text("{not json")
    jobs = run_study.pending_jobs(tmp_path, ARMS, SEEDS)
    assert StudyJob("cnn", 0) in jobs
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/eval/test_run_study.py -q`
Expected: FAIL — `run_study.py` does not exist.

- [ ] **Step 3: Implement**

```python
# scripts/run_study.py
"""Run the 3 arms x 3 seeds study, resumably.

Measured costs at seq_len=64 on the development Mac: feature arms 3.98 steps/s
(1.4 h per 20k-step job), pixel arm 0.67 steps/s (8.3 h). The whole study is
about 33 laptop-hours, of which the pixel arm is 25. Feature arms are therefore
scheduled first: if the run is interrupted, the treatment/control contrast
(frozen_ssl against random_vit) is already complete.

Resumable off the per-job JSON records, so an interrupted run costs at most one
job rather than the whole study.
"""

import argparse
import json
from pathlib import Path

from mbfps.data.buffer import ReplayBuffer
from mbfps.eval.study import StudyJob, job_record_path, run_job
from mbfps.utils.config import ARMS

SEEDS = (0, 1, 2)
_SLOW_ARMS = ("cnn",)


def pending_jobs(out_dir, arms=ARMS, seeds=SEEDS) -> list[StudyJob]:
    """Jobs without a readable record, cheapest arms first."""
    jobs = []
    for arm in arms:
        for seed in seeds:
            job = StudyJob(arm, seed)
            path = job_record_path(out_dir, job)
            if path.is_file():
                try:
                    json.loads(path.read_text())
                    continue
                except json.JSONDecodeError:
                    pass  # truncated by a killed process -- re-run it
            jobs.append(job)
    return sorted(jobs, key=lambda j: (j.arm in _SLOW_ARMS, j.arm, j.seed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--out", type=Path, default=Path("runs/m3_study"))
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    args = parser.parse_args()

    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    jobs = pending_jobs(args.out, tuple(args.arms), tuple(args.seeds))
    print(f"{len(jobs)} job(s) pending: "
          f"{', '.join(f'{j.arm}/s{j.seed}' for j in jobs)}")

    for i, job in enumerate(jobs, 1):
        print(f"\n===== [{i}/{len(jobs)}] {job.arm} seed {job.seed} =====", flush=True)
        record = run_job(job, buffer, args.out, steps=args.steps,
                         seq_len=args.seq_len, device=args.device)
        print(f"  steps/s={record['steps_per_second']:.2f} "
              f"kl_rate={record['kl_rate_above_free_bits']:.3f} "
              f"position_gap_final={record['position']['gap_final']:+.4f} "
              f"band_median={record['position']['band_median']:.2f}", flush=True)

    print(f"\nall records in {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/eval/test_run_study.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Mutation-test**

| mutation | must be caught by |
|---|---|
| `pending_jobs` returns everything, ignoring existing records | `test_a_job_with_an_existing_record_is_skipped` |
| the sort key drops `j.arm in _SLOW_ARMS` | `test_cheap_arms_run_first` |
| the `JSONDecodeError` branch `continue`s instead of falling through | `test_a_corrupt_record_is_treated_as_pending` |

- [ ] **Step 6: Commit**

```bash
git add scripts/run_study.py tests/eval/test_run_study.py
git commit -m "feat: resumable driver, cheap arms first"
```

---

## Task 6: Aggregate across seeds and evaluate the gate

**Files:**
- Create: `src/mbfps/eval/aggregate.py`, `scripts/report_study.py`
- Test: `tests/eval/test_aggregate.py`

**Interfaces:**
- Consumes: the JSON records written by Task 4.
- Produces:
  - `load_records(out_dir) -> list[dict]`
  - `per_arm(records) -> dict[str, dict]` with `"n_seeds"`, `"gap_final_by_seed"`, `"gap_final_mean"`, `"all_seeds_positive"`, `"any_degenerate"`, `"filtering_all_pass"`
  - `evaluate_gate(records) -> dict` with `"criteria"` (a dict of named booleans) and `"passed"`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_aggregate.py
import json

import numpy as np
import pytest

from mbfps.eval.aggregate import evaluate_gate, load_records, per_arm

ARMS = ("cnn", "frozen_ssl", "random_vit")


def _record(arm, seed, gap, *, degenerate=0, latent_wins=True, finite=45):
    return {
        "arm": arm, "seed": seed, "steps": 20_000,
        "steps_per_second": 4.0, "kl_rate_above_free_bits": 0.97,
        "position": {"gap_final": gap, "gap_mean": gap, "band_median": 50.0,
                     "steps_floor_above_persistence": 0,
                     "steps_degenerate": degenerate, "gap_finite": finite,
                     "n_steps": 45, "final_model": 100.0,
                     "final_persistence": 200.0, "final_floor": 50.0},
        "angle": {"gap_final": gap, "gap_mean": gap, "band_median": 5.0,
                  "steps_floor_above_persistence": 0, "steps_degenerate": 0,
                  "gap_finite": finite, "n_steps": 45, "final_model": 10.0,
                  "final_persistence": 20.0, "final_floor": 5.0},
        "filtering": {"latent_r2": 0.3, "embedding_r2": 0.2,
                      "latent_beats_embedding": latent_wins},
        "curves": {},
    }


def _write(tmp_path, records):
    for r in records:
        (tmp_path / f"result_{r['arm']}_seed{r['seed']}.json").write_text(json.dumps(r))
    return tmp_path


def test_load_records_reads_every_json_in_the_directory(tmp_path):
    _write(tmp_path, [_record("cnn", s, 0.1) for s in (0, 1, 2)])
    assert len(load_records(tmp_path)) == 3


def test_gate_passes_when_all_three_seeds_are_positive_for_every_arm(tmp_path):
    records = [_record(a, s, 0.2) for a in ARMS for s in (0, 1, 2)]
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["beats_persistence"] is True
    assert verdict["passed"] is True


def test_gate_fails_when_one_seed_of_one_arm_is_negative(tmp_path):
    """Unanimity, deliberately -- with n=3 a t-test has 2 degrees of freedom and
    would dress a weak result in strong-looking notation."""
    records = [_record(a, s, 0.2) for a in ARMS for s in (0, 1, 2)]
    records[4]["position"]["gap_final"] = -0.01
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["beats_persistence"] is False
    assert verdict["passed"] is False


def test_gate_fails_when_the_filtering_probe_loses_for_any_arm(tmp_path):
    records = [_record(a, s, 0.2) for a in ARMS for s in (0, 1, 2)]
    records[7]["filtering"]["latent_beats_embedding"] = False
    assert evaluate_gate(records)["criteria"]["filtering_beats_embedding"] is False


def test_gate_fails_on_a_degenerate_band(tmp_path):
    """A ratio over a numerically degenerate denominator is not evidence."""
    records = [_record(a, s, 0.2) for a in ARMS for s in (0, 1, 2)]
    records[2]["position"]["steps_degenerate"] = 7
    assert evaluate_gate(records)["criteria"]["band_is_usable"] is False


def test_gate_fails_when_a_seed_is_missing(tmp_path):
    """Nine cells or no verdict -- averaging over a silently varying subset is
    how a 3x3 study becomes a 3x2 study nobody notices."""
    records = [_record(a, s, 0.2) for a in ARMS for s in (0, 1)]
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["all_nine_cells_present"] is False
    assert verdict["passed"] is False


def test_per_arm_aggregation_is_nan_aware(tmp_path):
    """A NaN gap must not silently poison an arm's mean."""
    records = [_record("cnn", s, 0.2) for s in (0, 1, 2)]
    records[1]["position"]["gap_final"] = float("nan")
    summary = per_arm(records)["cnn"]
    assert summary["gap_final_mean"] == pytest.approx(0.2)
    assert summary["n_seeds_with_finite_gap"] == 2
    assert summary["all_seeds_positive"] is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/eval/test_aggregate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.eval.aggregate'`

- [ ] **Step 3: Implement**

```python
# src/mbfps/eval/aggregate.py
"""Aggregate the study's records and judge M3's exit gate.

Every reduction here is NaN-aware. `gap_closed` returns NaN wherever the band
is non-positive, so a naive mean would either poison an arm's score or, worse,
average over a silently varying subset of horizon steps.
"""

import json
from pathlib import Path

import numpy as np

from mbfps.utils.config import ARMS

SEEDS = (0, 1, 2)
MAX_DEGENERATE_STEPS = 0
"""How many horizon steps may have a numerically degenerate band.

Zero. At 20,000 steps the measured value was 0/45; a nonzero count means the
ratio is unstable and the raw curves should be read instead.
"""


def load_records(out_dir) -> list[dict]:
    return [json.loads(p.read_text())
            for p in sorted(Path(out_dir).glob("result_*_seed*.json"))]


def per_arm(records: list[dict]) -> dict:
    """Per-arm summary across seeds, NaN-aware throughout."""
    summary = {}
    for arm in {r["arm"] for r in records}:
        rows = sorted((r for r in records if r["arm"] == arm), key=lambda r: r["seed"])
        gaps = np.array([r["position"]["gap_final"] for r in rows], dtype=float)
        finite = np.isfinite(gaps)
        summary[arm] = {
            "n_seeds": len(rows),
            "seeds": [r["seed"] for r in rows],
            "gap_final_by_seed": [float(g) for g in gaps],
            "n_seeds_with_finite_gap": int(finite.sum()),
            "gap_final_mean": float(np.nanmean(gaps)) if finite.any() else float("nan"),
            # Unanimity, not a mean: a NaN seed is NOT positive.
            "all_seeds_positive": bool(finite.all() and (gaps > 0).all()),
            "max_degenerate_steps": max(r["position"]["steps_degenerate"] for r in rows),
            "filtering_all_pass": all(
                r["filtering"]["latent_beats_embedding"] for r in rows),
            "steps_per_second": float(np.mean([r["steps_per_second"] for r in rows])),
            "kl_rate_min": float(min(r["kl_rate_above_free_bits"] for r in rows)),
        }
    return summary


def evaluate_gate(records: list[dict], arms=ARMS, seeds=SEEDS) -> dict:
    """Spec section 4. Every criterion is reported; `passed` is their conjunction."""
    arms_summary = per_arm(records)
    present = {(r["arm"], r["seed"]) for r in records}
    expected = {(a, s) for a in arms for s in seeds}

    criteria = {
        "all_nine_cells_present": present >= expected,
        "beats_persistence": bool(
            arms_summary and all(
                arms_summary.get(a, {}).get("all_seeds_positive", False) for a in arms)),
        "band_is_usable": bool(
            arms_summary and all(
                arms_summary.get(a, {}).get("max_degenerate_steps", 1)
                <= MAX_DEGENERATE_STEPS for a in arms)),
        "filtering_beats_embedding": bool(
            arms_summary and all(
                arms_summary.get(a, {}).get("filtering_all_pass", False) for a in arms)),
    }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "per_arm": arms_summary,
        "missing_cells": sorted(expected - present),
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/eval/test_aggregate.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Write the report script**

```python
# scripts/report_study.py
"""Print the 3x3 table, the gate verdict, and the error-vs-horizon curves."""

import argparse
from pathlib import Path

import numpy as np

from mbfps.eval.aggregate import evaluate_gate, load_records, per_arm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/m3_study"))
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()

    records = load_records(args.out)
    if not records:
        raise SystemExit(f"no result_*.json under {args.out}")
    summary = per_arm(records)

    print(f"{'arm':<12}{'seeds':>7}{'steps/s':>9}{'kl_rate':>9}"
          f"{'gap_final by seed':>34}{'mean':>9}{'unanimous':>11}")
    for arm in sorted(summary):
        s = summary[arm]
        by_seed = "  ".join(f"{g:+.4f}" for g in s["gap_final_by_seed"])
        print(f"{arm:<12}{s['n_seeds']:>7}{s['steps_per_second']:>9.2f}"
              f"{s['kl_rate_min']:>9.3f}{by_seed:>34}{s['gap_final_mean']:>+9.4f}"
              f"{str(s['all_seeds_positive']):>11}")

    verdict = evaluate_gate(records)
    print("\n--- M3 exit gate ---")
    for name, ok in verdict["criteria"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if verdict["missing_cells"]:
        print(f"  missing cells: {verdict['missing_cells']}")
    print(f"\nGATE: {'PASSED' if verdict['passed'] else 'NOT PASSED'}")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        arms = sorted(summary)
        fig, axes = plt.subplots(1, len(arms), figsize=(5 * len(arms), 4), squeeze=False)
        for ax, arm in zip(axes[0], arms):
            rows = [r for r in records if r["arm"] == arm]
            for name, style in (("persistence_position", "--"),
                                ("rssm_position", "-"),
                                ("floor_position", ":")):
                curve = np.mean([r["curves"][name] for r in rows], axis=0)
                ax.plot(np.arange(1, len(curve) + 1), curve, style,
                        label=name.replace("_position", ""))
            ax.set_title(arm); ax.set_xlabel("imagined step")
            ax.set_ylabel("position error (Doom map units)"); ax.legend()
        fig.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=110)
        print(f"figure={args.figure}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Mutation-test**

| mutation | must be caught by |
|---|---|
| `all_seeds_positive` uses `np.nanmean(gaps) > 0` instead of unanimity | `test_gate_fails_when_one_seed_of_one_arm_is_negative` |
| `all_seeds_positive` ignores NaN seeds | `test_per_arm_aggregation_is_nan_aware` |
| `evaluate_gate` drops `all_nine_cells_present` | `test_gate_fails_when_a_seed_is_missing` |
| `MAX_DEGENERATE_STEPS` raised to 45 | `test_gate_fails_on_a_degenerate_band` |

- [ ] **Step 7: Commit**

```bash
git add src/mbfps/eval/aggregate.py scripts/report_study.py tests/eval/test_aggregate.py
git commit -m "feat: NaN-aware aggregation and the M3 gate verdict"
```

---

## Task 7: Run the study and record the result

**Files:**
- Modify: this plan (record results)

- [ ] **Step 1: Full suite and a local smoke run**

```bash
.venv/bin/python -m pytest -q
caffeinate -dimsu .venv/bin/python scripts/run_study.py --out runs/m3_smoke \
  --steps 20 --seq-len 8 --device cpu --arms random_vit --seeds 0 --allow-short
.venv/bin/python scripts/report_study.py --out runs/m3_smoke
```

Expected: the suite green, one record written, the report printing a table and a
gate verdict of NOT PASSED (one cell of nine).

`--allow-short` is required here and only here. `--steps`/`--seq-len` below the
study's own floors (5000 / 32) are refused at the command line, because
`--steps 200` for `--steps 20000` otherwise writes nine complete records at a
smoke configuration and exits 0, and the corrective re-run is then refused as a
configuration mismatch until all nine are deleted by hand. Step 3 below is the
real run and passes neither flag.

- [ ] **Step 2: Provision the GPU box**

The study is ~33 laptop-hours; on a rented GPU expect roughly 4–8. Any provider
works — the driver only needs Python 3.12, the repo, and the dataset. Record
which provider, instance type and hourly rate you used in "## Task 7 results".

Copy the repo and `data/my_way_home` to the box, install the package, and
confirm the environment reproduces local numbers before spending hours on it:

```bash
python -m pytest -q                      # must match the local count
python scripts/train_world_model.py --arm random_vit --steps 60 --seq-len 64 \
  --out /tmp/probe --device cuda         # record steps/s
```

**Parity gate:** if the test count differs from local, STOP and resolve it before
running the study. A different environment silently producing different numbers
is exactly the failure this project has spent two milestones learning to catch.

- [ ] **Step 3: Run the study**

```bash
python -u scripts/run_study.py --out runs/m3_study --device cuda 2>&1 | tee study.log
```

The driver is resumable: re-running it skips jobs with a record. Feature arms
run first, so an interruption still leaves the treatment/control contrast whole.

- [ ] **Step 4: Report**

```bash
python scripts/report_study.py --out runs/m3_study --figure runs/m3_study/curves.png
```

- [ ] **Step 5: Record the results honestly**

Fill "## Task 7 results" with the 3×3 table, the gate verdict per criterion, the
measured throughput per arm, and the figure path. **A NOT PASSED verdict is a
result, not a failure to hide.** If the arms do not separate, record that — the
spec states explicitly that a null result is legitimate and that a gate which
can only pass when the hypothesis holds would make the study unfalsifiable.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/plans/2026-09-04-mb-fps-m3b-study.md
git commit -m "docs: M3 study results and gate verdict"
```

---

## Exit criteria for this plan

- [x] `pytest` fully green, with the counts each task states — **851 passed, 1 warning,
      exit 0** (851 collected).
- [x] Angle carries the same degeneracy guards as position (Task 1) — `band_is_usable` is
      judged over both metrics, and `frozen_ssl`'s only failure of it is a single angle step.
- [x] Probes are fit under the rollout's own context protocol, verified on a real checkpoint
      (Task 2) — see Task 2 results above.
- [x] `filtering_report` is called by the eval script, so spec §4 criterion 4 is produced
      (Task 3) — nine cells reported; see Task 3 results above.
- [ ] Every mutation table row is caught, using a harness self-checked all three ways —
      **satisfied per task at the time each task landed; NOT re-run against the code state
      that produced these nine records.** The trust claim in the results section rests on
      Tasks 1–6's mutation tables plus the 851 tests, and says so explicitly.
- [x] Nine records present, one per (arm, seed) — verified, no duplicates, no strangers.
- [x] Reward-prediction accuracy reported per arm, with its baseline and degeneracy flag
      (spec §4.3) — nine cells, `is_degenerate=True` in all nine.
- [x] The gate verdict is recorded per criterion, whatever it is — recorded above, **NOT
      PASSED**.
- [x] `encoders.py` still the only arm-varying module — grep re-run; the only hits outside
      `encoders.py` are log prefixes, checkpoint filenames and record labels in
      `training/world_model.py` and `training/autoencoder.py`.

---

## Task 2 results

Held-out embedding-probe R² under the matched protocol is the record's
`probe.embedding_selection_r2` — the model's *predicted* embedding, fit on 16 train episodes
and selected on a 4-episode held-out train split, under the rollout's own 5-frame context.

| quantity | value |
|---|---|
| embedding probe held-out R² under matched protocol | `cnn` −0.0357 / −0.0361 / −0.0366 (ridge 1e7, all three); `frozen_ssl` +0.3597 / +0.3336 / +0.3829 (ridge 1e5 / 1e3 / 1e5); `random_vit` +0.2974 / +0.3300 / +0.2486 (ridge 1e3, all three) — seeds 0/1/2 |
| comparison to the pre-fix mismatched figure (~0.29) | Feature arms land **at or above** it — six of six in 0.249–0.383, median 0.315 — so the matched protocol did not cost signal where the encoder carries any. The pixel arm lands **far below**, at −0.036, because its encoder is collapsed rather than because the protocol changed; the pre-fix ~0.29 was a whole-episode-filtered probe scored on its own mismatched selection set and is not a like-for-like comparison for any arm. Independent confirmation that the protocol fix worked as designed is §3.2a's own measurement, reproduced by these records: steps with band ≤ 0 fell 9/45 → 2/45 and the median band rose +21.15 → +53.23. |

## Task 3 results

`filtering_report`, gate criterion 4 (spec §4.4). Both R² are **optimistically biased levels
by construction** — the ridge is selected on the rows the probe is then scored on — so the
comparison is fair but the levels must never be quoted as clean held-out numbers. Criterion 4's
selected ridge is not written to any record; re-derived it is 1e3 for both sides in
`frozen_ssl`/s2 and 1e7 for both sides in all three `cnn` cells.

| quantity | value |
|---|---|
| `latent_r2` | `cnn` −0.08533 / −0.08533 / −0.08533; `frozen_ssl` +0.2892 / +0.2781 / +0.3376; `random_vit` +0.2290 / +0.2743 / +0.1575 (seeds 0/1/2) |
| `embedding_r2` | `cnn` +0.2703 / +0.2736 / +0.2369; `frozen_ssl` +0.3321 / +0.3451 / +0.3353; `random_vit` +0.3195 / +0.3241 / +0.3256 (seeds 0/1/2) |
| `latent_beats_embedding` | **True in 1 of 9 cells** (`frozen_ssl`/s2, margin +0.00225). False in the other eight, by −0.0429 to −0.3589. The criterion is a conjunction over all nine cells, so it **FAILS**. The one True cell is a sign flip in noise: its 95% window-level block bootstrap (2,000 resamples, probes held fixed, 189 windows) is [−0.0291, +0.0331] with P(margin > 0) = 0.55; it is True at 1 of 5 ridge decades; and `frozen_ssl`'s own seed-to-seed SD on this margin is 0.0352, 15.6× the effect. |

## Task 7 results

**Provider / instance / rate:** none — the study did **not** run on a rented GPU box. Step 2's
provisioning and parity gate were skipped and no provider, instance type, hourly rate, or
environment-parity test count was recorded. Measured throughput (`cnn` 0.551–0.598 steps/s,
feature arms 3.659–3.885) is at or below this plan's own local-laptop baseline of 0.67 / 3.98,
so the run was local, on MPS. `study.log` covers 7 of 9 cells (both earlier `frozen_ssl` cells
predate it) and spans 2026-09-08 to 2026-09-10 across five driver invocations, one of which
(`cnn`/s0, started 11:54:29) produced no step output and was relaunched 8.6 h later. **One code
state produced all nine cells**: the last commit touching `src/` or `scripts/` is c1d1d4f
(2026-09-06 23:06), the earliest record mtime is `result_frozen_ssl_seed0.json` (2026-09-07
16:13) and the latest is `result_cnn_seed2.json` (2026-09-10 10:17); the record schema carries
no git SHA or device, so this had to be reconstructed from mtimes. Total compute 37.975 h summed
over the nine `seconds` fields — **not** a calendar span, which the records cannot recover.
Figure: `runs/m3_study/curves.png`.

Table values are `gap_closed` at the final horizon step (h=45) for the **position** metric,
`GATE_METRIC = "position"`. `steps/s` is the arm mean of the three cells' training-only rate.
`+nan` is the contract for a non-positive persistence-to-floor band: that cell measured
nothing. Means are `np.nanmean`, so `cnn`'s is a single seed wearing an arm's label.

| arm | steps/s | seed 0 | seed 1 | seed 2 | mean | unanimous > 0 |
|---|---|---|---|---|---|---|
| `cnn` | 0.575 | +nan | +nan | +1.3317 | +1.3317 (finite 1/3) | False |
| `frozen_ssl` | 3.782 | −0.8523 | −0.6470 | −0.6491 | −0.7161 (finite 3/3) | False |
| `random_vit` | 3.857 | −0.4419 | −0.3875 | −0.5452 | −0.4582 (finite 3/3) | False |

The template's six gate rows, filled:

| gate criterion | verdict |
|---|---|
| all nine cells present | **PASS** — 9 of 9 expected (arm, seed) cells, no duplicates, no strangers |
| reward-prediction accuracy reported per arm (spec §4.3) | **PASS** — reported and deliberately not gated; all nine cells `is_degenerate=True`, 2 reward events in 9,949 scored steps, `r2` ∈ [−0.00144, +0.00149], `baseline_mse` 2.010250e-04 bit-identical across arms |
| beats persistence (unanimous across seeds) | **FAIL** — all three arms independently: `frozen_ssl` and `random_vit` negative in 3/3 seeds, `cnn` undefined in 2/3 (`floor ≥ persistence` on 45/45 steps in seed 1) |
| band is usable (no degenerate steps) | **FAIL** — `MAX_DEGENERATE_STEPS = 0` over **both** metrics: `cnn` 18 (angle s2; also 16 position steps in s2), `frozen_ssl` 1 (a single angle step in s0), `random_vit` 0 |
| filtering probe beats the embedding probe | **FAIL** — True in 1 of 9 cells (`frozen_ssl`/s2, margin +0.0023, 95% window bootstrap [−0.0291, +0.0331] straddling zero); the criterion is a conjunction over all nine |
| **GATE** | **NOT PASSED** (`report_study.py` exit 10 = `EXIT_GATE_NOT_PASSED`) |

`evaluate_gate` computes **four more criteria than this template's six**, and reports a ninth
as not evaluated. They are listed separately so a reader diffing against the plan is not
surprised; none of them changes the verdict, which is already NOT PASSED on three of the eight
computed criteria.

| criterion the harness adds beyond the template | verdict |
|---|---|
| `no_duplicate_cells` (implemented, not in spec §4) | **PASS** |
| `every_record_is_a_study_cell` (implemented, not in spec §4) | **PASS** |
| `curves_produced` (spec §4.2, present in the spec but not in this template) | **PASS** — all six curves in all nine records; figure at `runs/m3_study/curves.png` |
| `invariant_tests_green` (spec §4.5) | **NOT EVALUATED by the gate** — printed `[ n/a]`; resolved externally and **green**: `.venv/bin/python -m pytest -q` → **851 passed, 1 warning, exit 0** (851 collected) |

Note on the spec accounting: spec §4 lists exactly five criteria. **Two of them fail** (§4.1
beats persistence, §4.4 filtering); §4.2 and §4.3 pass; §4.5 is green in pytest but the gate
does not read it. The third failing criterion, `band_is_usable`, is the harness's own guard on
§4.1's denominator (`MAX_DEGENERATE_STEPS = 0` in `aggregate.py`, introduced with
`evaluate_gate` in the aggregation task; Task 1 extended its *input* to cover angle as well as
position).

**Measured throughput and cost per arm** (`steps_per_second` is training only; `seconds` is
the whole cell, and the difference — 85.9–110.2 s per cell, 14.2 min / 0.6% of the study — is
evaluation):

| arm | steps/s per seed | mean steps/s | hours (3 seeds) | share |
|---|---|---|---|---|
| `cnn` | 0.5768 / 0.5513 / 0.5975 | 0.575 | 29.093 | 76.6% |
| `frozen_ssl` | 3.6592 / 3.8020 / 3.8850 | 3.782 | 4.489 | 11.8% |
| `random_vit` | 3.8504 / 3.8561 / 3.8648 | 3.857 | 4.393 | 11.6% |
| **total** | — | — | **37.975** | +15.1% vs the plan's 33 h |

Arm ratio measured at **6.1–7.0×** across all cell pairs (6.3–7.0× same-seed on steps/s,
6.26–6.90× same-seed on wall seconds, 6.48–6.62× on arm mean hours), against this plan's
carried 5.9×. The miss is on the pixel arm: 86% of its predicted 0.67 steps/s, versus 96% of
the predicted 3.98 for the feature arms.

**Reading note required with this table.** The `cnn` row is not a weak result, it is an
unmeasured one. Its dynamics KL cleared the 0.20-nat free-bits floor on 18, 16 and 15 of
20,000 steps (0.075–0.090%), peaking at 0.400 / 0.800 / 0.368 nat, against 14,910–19,513 steps
for the six feature cells — so `prior_net` received a gradient 15–18 times in the whole run and
`imagine()` runs close to its initialisation; `train_world_model` printed that warning three
times, all `[cnn]`. Its band probe selected the grid's maximum ridge 1e7 at a negative held-out
selection R² (−0.0357 / −0.0361 / −0.0366), and its model, persistence and floor curves all sit
within 291.83–299.29 map units at every horizon step. The `+1.3317` is a ratio of sub-map-unit
noise (numerator 0.528, denominator 0.397) with 16 of 45 horizon steps degenerate; it must
never be quoted as the pixel baseline beating anything.

---

## M3 results

All nine cells of the 3-arm × 3-seed study are in `runs/m3_study/` as
`result_<arm>_seed<n>.json`, with checkpoints beside them and the horizon figure at
`runs/m3_study/curves.png`. Every cell ran 20,000 steps at `seq_len=64`, `context=5`,
`horizon=45`, `split_seed=0`, on a byte-identical 98-train / 24-val episode split
(verified by hashing the `episodes` block of all nine records, not assumed).

```
.venv/bin/python scripts/report_study.py --out runs/m3_study --figure runs/m3_study/curves.png
```

**exits 10 (`EXIT_GATE_NOT_PASSED`). GATE: NOT PASSED.** Three of the eight computed criteria
fail; five pass; the ninth is printed `[ n/a]`.

| criterion | verdict | kind |
|---|---|---|
| `all_nine_cells_present` | PASS | accounting (harness) |
| `no_duplicate_cells` | PASS | accounting (harness) |
| `every_record_is_a_study_cell` | PASS | accounting (harness) |
| `beats_persistence` | **FAIL** | spec §4.1 |
| `band_is_usable` | **FAIL** | harness guard (`MAX_DEGENERATE_STEPS = 0`, `aggregate.py`); judged over both metrics since Task 1 |
| `filtering_beats_embedding` | **FAIL** | spec §4.4 |
| `reward_reported` | PASS | spec §4.3 |
| `curves_produced` | PASS | spec §4.2 |
| `invariant_tests_green` | NOT EVALUATED (green externally) | spec §4.5 |

Note on wording: "three of nine criteria failed" flattens a bookkeeping check and a spec
criterion into the same unit. The precise statement is **two of the spec's five criteria
fail** (§4.1 rollout, §4.4 filtering), and a third failure comes from the harness's own
`band_is_usable` guard on §4.1's denominator. §4.2 and §4.3 pass; §4.5 is green in pytest but
is not read by the gate. The spec's §4 lists exactly five criteria and `band_is_usable` is
none of them — it is introduced with `evaluate_gate` and `MAX_DEGENERATE_STEPS` in the
aggregation task, and Task 1 only extended its *input* to cover angle as well as position.

**The failure is not a harness failure, and here is exactly how far that claim reaches.**
`.venv/bin/python -m pytest -q` is **851 passed, 1 warning, exit 0** (851 collected). Criterion
5's six named invariants — overfit-one-batch, gradient-flow, privileged-state isolation,
determinism, golden rollout, split stability — all have live test homes inside that 851.
Separately, every one of the 108 summary fields in the nine records (`gap_final`,
`gap_finite`, `steps_degenerate`, `steps_floor_above_persistence`, `final_model`, `n_steps`,
across 9 cells × 2 metrics) was recomputed from the raw per-step curves stored in the same
record and matched exactly, and criterion 4's `latent_r2` / `embedding_r2` and the whole gain
estimator were re-derived from the shipped checkpoints and reproduced **bit-for-bit in all
nine cells** — same gains, same selected ridges, same CIs.

Two limits on that, stated rather than left implicit. First, **no mutation run was repeated
for this study.** The plan requires every guard to be mutation-tested and each of Tasks 1–6
ships its own mutation table; the trust claim here rests on those tables plus the 851 tests,
not on a fresh mutation pass over the code that produced these nine records. M2's own results
section is the reason this matters — it recorded 11 mutations that passed the committed suite,
including a `terminated`/`truncated` swap the suite could not see because every fixture was
zeros. Second, **the recomputation cannot catch an error shared by both paths it compares.**
Re-deriving the summary fields from curves in the same record, and criterion 4 from the same
checkpoints, would reproduce a truth-slice off-by-one exactly — and `rollout.py`'s own comment
says such an error "would shift every reported position error without failing any shape or
smoke test". The instrument computes what it claims to compute. Whether what it claims is the
right quantity is carried by the tests, not by the reproduction.

Third, a reproduction detail that matters for anyone re-deriving these numbers: the study ran
on **MPS**. A CPU re-derivation of the gain block does *not* reproduce the shipped records —
the posterior samples are a different RNG stream, and on this checkout a CPU run moved
`frozen_ssl`/s1's gain from +0.0407 to +0.0315 and flipped two of its selected ridges. Every
re-derived number below was produced on MPS and checked against the shipped record first.

### The primary metric: no arm beats persistence

`gap_closed` at horizon 45, position, the gate metric (`GATE_METRIC = "position"`; angle is
computed and printed but does not gate criterion 1):

| arm | seed 0 | seed 1 | seed 2 | mean | finite | unanimous > 0 | degenerate | floor≥pers |
|---|---|---|---|---|---|---|---|---|
| `cnn` | NaN | NaN | +1.3317 | +1.3317 | 1/3 | False | 16 | 45 |
| `frozen_ssl` | −0.8523 | −0.6470 | −0.6491 | −0.7161 | 3/3 | False | 0 | 0 |
| `random_vit` | −0.4419 | −0.3875 | −0.5452 | −0.4582 | 3/3 | False | 0 | 0 |

The `degenerate` and `floor≥pers` columns are per-arm **maxima over seeds**, not sums:
`cnn`'s per-seed position counts are 8/0/16 degenerate and 37/45/1 floor-above-persistence.

`gap_closed = 1` means the imagined rollout is as accurate as the encoder itself permits;
`0` means exactly as accurate as holding the last context frame's decoded position for the
whole horizon. **Negative means the model is worse than assuming the agent never moved.**
In raw map units at h=45:

| cell | model | persistence | floor | band | band/persistence |
|---|---|---|---|---|---|
| `frozen_ssl`/s0 | 214.92 | 173.31 | 124.49 | 48.82 | +0.2817 |
| `frozen_ssl`/s1 | 231.97 | 187.09 | 117.72 | 69.36 | +0.3707 |
| `frozen_ssl`/s2 | 190.47 | 163.76 | 122.62 | 41.14 | +0.2512 |
| `random_vit`/s0 | 221.45 | 201.74 | 157.12 | 44.62 | +0.2212 |
| `random_vit`/s1 | 215.68 | 194.04 | 138.18 | 55.85 | +0.2878 |
| `random_vit`/s2 | 238.40 | 209.09 | 155.34 | 53.76 | +0.2571 |
| `cnn`/s0 | 298.82 | 298.73 | 298.82 | −0.09 | −0.0003 |
| `cnn`/s1 | 299.14 | 298.20 | 298.99 | −0.79 | −0.0026 |
| `cnn`/s2 | 298.76 | 299.29 | 298.89 | +0.40 | +0.0013 |

The six feature cells add **26.7–44.9** (`frozen_ssl`) and **19.7–29.3** (`random_vit`) map
units of error on top of a do-nothing baseline that already carries the full 126.22-unit mean
displacement at h=45. `beats_persistence` fails for all three arms independently: the two
feature arms on three negative gaps each, `cnn` on two undefined ones.

**The failure is horizon-dependent, and the shape matters more than the endpoint.** Counting
the six feature cells with `gap_closed > 0` at each horizon step, from the stored curves:

| horizon step | cells with `gap_closed > 0` (of 6) |
|---|---|
| h = 1 | 5 |
| h = 2 | 5 |
| h = 3–4 | 4 |
| h = 5–9 | 3 |
| h = 10 | 2 |
| h = 11–16 | 1 |
| h ≥ 17 | 0 |

**There is no horizon at which all six feature cells beat persistence**, and the best case is
h ≤ 2 at 5 of 6. At h=1 the per-cell gaps are `frozen_ssl` +0.541/+0.479/+0.084 and
`random_vit` −0.024/+0.727/+0.201 (s0/s1/s2), and the model's error first exceeds persistence
at h = 10/5/2 (`frozen_ssl`) and h = 1/11/17 (`random_vit`) — ranges that overlap completely
between the arms.

Those h=1 ratios deserve the same scrutiny this write-up applies to `cnn` at h=45, because
they are built the same way. The h=1 bands are **3.17 / 9.81 / 3.07** (`frozen_ssl`) and
**3.05 / 5.81 / 3.92** (`random_vit`) map units — 1.7–6.1% of the persistence error at that
step — and the raw model-minus-persistence advantages the ratios divide are
**+1.72 / +4.69 / +0.26** and **−0.07 / +4.22 / +0.79** map units. `frozen_ssl`/s2's "+0.084"
is a 0.26-map-unit edge. Three of the six one-step advantages are under one map unit, and no
interval is attached to any of them. So: at h=1 the model is ahead of persistence by 0.26–4.7
map units in 5 of 6 feature cells, on bands 3–10 units wide and with no interval. That is
consistent with the model having learned some one-step dynamics; it does not establish it.

Doubling `seq_len` from 32 to 64 roughly halved the deficit relative to the pilot
(`random_vit` −0.887 → −0.44 / −0.39 / −0.55, seeds 0/1/2) and changed the sign in no cell.

### Is this about the representations, or about the shared setup?

This is the question the milestone turns on, so it gets answered directly rather than left
to the reader.

**It is very likely about the shared setup.** `frozen_ssl` and `random_vit` differ in
exactly one respect — DINOv2 features versus randomly-initialised ViT features through a
byte-identical 12,320-parameter `BottleneckEncoder`. Their encoder floors differ by 28.6 map
units (121.61 ± 3.49 vs 150.21 ± 10.45): the two representations are genuinely not equally
good at supporting a position readout. Both arms nevertheless lose to persistence in **6 of 6
cells**, on clean bands (0 degenerate steps, 0 floor-above-persistence steps, 45/45 finite
gaps), with the same crossover shape. Two representations of materially different quality
failing the same way points at what they share: the RSSM, `imagine()`, the 20,000-step budget,
the 45-step horizon — not at the representation contrast the study was built to run.

The pixel arm cannot be added to that argument (below), so this is a **two-arm** conclusion
reported in a three-arm table. And it is not an equivalence claim: three seeds per arm has
no power to establish "representation choice does not matter". What is established is that
neither of two very different feature spaces was sufficient, under this recipe, to make the
learned dynamics beat a null baseline at 45 steps.

**Attribution is not mechanism, and this section stops one step short of one.** Four candidate
causes are named above and none is ranked, because nothing in the shipped data discriminates
among them. Two re-scores of the existing checkpoints would, at roughly two minutes per cell
and no retraining:

- **Action-shuffled imagination.** Re-run `evaluate_rollout` with the action sequence
  permuted. If the position error is unchanged, the dynamics prior is not action-conditioned
  at all — a shared-setup fact that would dominate everything else in this write-up and block
  M4 outright.
- **Teacher-forced (posterior) rollout at the same horizons.** The floor already runs the
  posterior on real frames; scoring a posterior-forced trajectory at each horizon step
  separates "error accumulates through imagination" from "the encoder/probe is the ceiling".

Neither was run. They are the two cheapest things that would turn a two-arm inference into
a mechanism, and they head the ranked next-actions list at the end.

### Addendum (2026-09-11): the action pathway, measured

The two re-scores the previous section stops short of were run (commits `3688311` and
`0f03915`; `scripts/diagnose_dynamics.py`, `src/mbfps/eval/diagnostics.py`) on the nine
shipped checkpoints, no retraining, MPS, torch 2.13.0. Both are paired: every intervened
imagination is replayed from the same per-window RNG snapshot as the real one, so a delta is
a difference between action sequences and not a reading of sampling noise. The pairing is
proven rather than assumed — `stream_drift`, `open_loop_divergence` and `record_reproduction`
read exactly `0.0` on all nine cells, i.e. the real arm is bitwise the shipped
`curves["rssm_position"]`.

**The first reading was wrong and is retracted.** The action-shuffle result alone (`3688311`)
was read as "the prior ignores actions; M4 is blocked." A permutation preserves the action
multiset, so it tests order sensitivity only, and the follow-up ladder shows the prior is
action-conditioned — just not to order. The retraction is recorded here so the shape of the
error survives: one null rung, read as a null pathway.

**The ladder.** Three interventions of strictly increasing perturbation on the 45 horizon
actions of each of the 229 scored windows. Every rung changed all 229 windows in every cell
(steps moved per window, mean / min: shuffled 30.6 / 14, resampled 35.2 / 24, held 45 / 45),
so no null below is a no-op. Position, horizon-mean delta (intervened − real) ± 2 SE
(episode-clustered, 24 clusters); family-wise threshold z = 3.31 over 54 comparisons:

| cell | shuffled (order) | resampled (counts) | held FWD − held NOOP | z | |
|---|---|---|---|---|---|
| cnn/s0 | −0.01 ± 0.01 | +0.04 ± 0.30 | +1.31 ± 1.43 | 1.82 | unmeasurable |
| cnn/s1 | −0.01 ± 0.03 | +0.04 ± 0.59 | +2.79 ± 2.95 | 1.89 | unmeasurable |
| cnn/s2 | −0.01 ± 0.02 | +0.09 ± 0.46 | +1.79 ± 2.07 | 1.73 | unmeasurable |
| frozen_ssl/s0 | +0.99 ± 3.95 | +5.21 ± 7.90 | **+42.59 ± 11.47** | **7.42** | responds |
| frozen_ssl/s1 | +0.03 ± 2.23 | −3.33 ± 4.81 | **+26.70 ± 12.53** | **4.26** | responds |
| frozen_ssl/s2 | +1.93 ± 2.73 | −5.72 ± 6.34 | **+17.66 ± 8.40** | **4.20** | responds |
| random_vit/s0 | −0.08 ± 0.54 | +2.30 ± 2.75 | +10.12 ± 7.78 | 2.60 | nominal |
| random_vit/s1 | +2.03 ± 2.87 | −0.48 ± 2.05 | +12.89 ± 8.90 | 2.90 | nominal |
| random_vit/s2 | −0.11 ± 1.67 | +1.09 ± 4.46 | +10.43 ± 15.16 | 1.38 | null |

*Shuffled* (same multiset, permuted order) and *resampled* (i.i.d. from the scored windows'
own action marginal — 959 / 2134 / 1802 / 3559 / 941 / 910 over NOOP, TURN_LEFT, TURN_RIGHT,
MOVE_FORWARD, MOVE_LEFT, MOVE_RIGHT) are null in 9 of 9. The equivalence bounds those nulls
license, as a fraction of the persistence-to-floor range at h=45: shuffled 1.2–8.1%,
resampled 3.7–16.2% across the six feature cells. An order or count effect larger than that
would have been seen; none was.

*Held* replaces the whole horizon with one action, for each of the six, and the reported
statistic is the pre-registered contrast between the two most physically distinct: hold
MOVE_FORWARD minus hold NOOP. `frozen_ssl` clears the family-wise threshold in 3 of 3;
`random_vit` is nominal (> 2 SE) in 2 of 3 and below in 1 — under-powered per cell but
positive in every seed; the sign is positive in **9 of 9** cells (two-sided sign test
p = 0.004; the cells share windows, so this is a pattern, not nine independent tests). The
pixel arm is null and unmeasurable for the reason already given — its probe reads a constant.

**The sign is physical.** Per held action on `frozen_ssl`/s0, delta ± 2 SE:

| held | delta |
|---|---|
| NOOP | −11.60 ± 7.91 |
| TURN_LEFT | +22.93 ± 16.55 |
| TURN_RIGHT | +11.78 ± 9.10 |
| MOVE_FORWARD | **+30.99 ± 12.04** |
| MOVE_LEFT | +1.31 ± 8.78 |
| MOVE_RIGHT | −0.66 ± 6.18 |

Holding FORWARD runs the imagined position away from the truth; holding NOOP holds it
nearer. The prior knows that forward moves and noop does not.

**Why the first ladder produced a false null, so the next reader does not repeat it.** The
first held rung held only the globally rarest action, chosen to maximise distance from the
real sequence. That action is MOVE_RIGHT, and it is the one the model barely responds to
(−0.66 ± 6.18 above). A null under it was the fixture answering the question by accident —
the L1 species in a new costume. Holding every action and contrasting FORWARD against NOOP
is what turned a false null into a z = 7.4 response on the same checkpoints, windows and seed.

**What the three rungs say together.** The prior conditions on a coarse horizon-average of
the actions — which a permutation preserves exactly, a resample from the same marginal
preserves in expectation, and only a held action moves far. It responds to the gist of the
plan and not the plan. That is why the open loop still loses to persistence *under the real
actions*: a mixed 45-step sequence averages to something the prior tracks too crudely to
follow step by step.

**The k-step re-grounding sweep** (imagine k steps, re-observe the real frame, repeat)
localises where the crude pathway stops being enough. Position error at h=45, the six
feature cells:

| cell | floor | k=1 | k=3 | k=5 | k=15 | k=45 | persistence |
|---|---|---|---|---|---|---|---|
| frozen_ssl/s0 | 124.5 | 126.7 | 128.5 | 128.8 | 153.2 | 214.9 | 173.3 |
| frozen_ssl/s1 | 117.7 | 124.2 | 128.7 | 139.0 | 152.6 | 232.0 | 187.1 |
| frozen_ssl/s2 | 122.6 | 121.9 | 128.6 | 127.2 | 141.2 | 190.5 | 163.8 |
| random_vit/s0 | 157.1 | 159.2 | 163.2 | 167.0 | 180.7 | 221.5 | 201.7 |
| random_vit/s1 | 138.2 | 140.3 | 139.3 | 140.4 | 163.5 | 215.7 | 194.0 |
| random_vit/s2 | 155.3 | 159.3 | 163.7 | 160.1 | 184.0 | 238.4 | 209.1 |

Re-grounding every 1–5 steps sits at the floor; by k=15 roughly half the recoverable error
is gone; k=45 is the open loop and is worse than persistence. k=45 reproduces
`curves["rssm_position"]` bitwise, and k=1 is never bitwise the floor (it is one prior step
above it, not the posterior of the scored step), so the sweep's two endpoints are checked
against numbers this write-up already carries. The pixel arm sits at 298.7–299.4 for every
k, floor and persistence: nothing.

**What this changes for M4.** The blocker is not action-blindness. It is the one the gate
already found — compounding error past k ≈ 5 under every action sequence tested, including
the real one. That moves the remedy from "add an action pathway" (a design problem) to
"stop the compounding" (a modelling problem: RSSM capacity, the free-bits schedule that
starved the prior on the pixel arm and left it at 75–90% on the feature arms, the 20,000-step
budget on which the loss was still rising in 7 of 7 logged cells). The pixel arm's
checkpoints remain unusable for imagination regardless.

**What the ladder does NOT establish.** It does not show that the pathway is *sufficient*
for control — a prior that resolves the plan's gist may still be too coarse for an actor to
plan through. It does not rank the feature arms: `random_vit`'s per-cell nominal readings
are consistent with the same pathway at lower power, not with its absence. And every
interval here is the same within-cell, evaluation-window interval as elsewhere in this
write-up — it covers no seed, fit or sampling variation. Two follow-ups would tighten it:
a probe-free embedding-space reading of each rung (removing the ridge probe's R² of
0.30–0.38 from the effect size), and a pooled cross-cell statistic in place of nine per-cell
verdicts that disagree across `random_vit` seeds.

### The treatment is below the control, and the study cannot say why

On the gate metric the **treatment loses to the control**: `frozen_ssl` −0.7161 against
`random_vit` −0.4582, with complete 3-vs-3 separation — `frozen_ssl`'s best (−0.6470) sits
below `random_vit`'s worst (−0.5452). The direction survives band-free renormalisation
(relative excess over persistence at h=45: 24.0/24.0/16.3% vs 9.8/11.2/14.0%; horizon-mean
and trapezoidal-area ratios likewise, all with complete separation).

Four things stop this from being a finding about pretraining.

1. **The design's statistical ceiling, and the tail is not free to choose.** At 3 vs 3 there
   are only C(6,3) = 20 permutation assignments. The exact two-sided test returns its floor,
   p = 2/20 = **0.100**, and that is the number this comparison gets. A one-sided 1/20 = 0.050
   is available only by taking the tail in the observed direction — and the observed direction
   is one the spec does not pre-register. Spec §3.1 registers two branches and only two:
   "Arm 2 > Arm 3 → the benefit is the pretraining. Arm 2 ≈ Arm 3 → the benefit is target
   stability." There is no branch for Arm 2 < Arm 3. Run in the **pre-registered** direction
   the exact one-sided test returns **p = 20/20 = 1.00**. Quoting 0.050 here would launder a
   post-hoc tail into a pre-specified one. The point that survives is that *complete separation
   is the strongest ordering this design can express*, and its exact two-sided floor is 0.100.
   Note also that the spec's between-arm decision rule is unexecutable at n = 3: its two
   branches are `>` and `≈`, and this design can separate neither from the other. It has no
   test for its own primary comparison.
2. **The ordering is not robust to which number you read.** `frozen_ssl` has the *lower*
   raw final-step position error (212.45 ± 20.86 vs 225.18 ± 11.81, exact two-sided
   p = 8/20 = 0.400) and the *worse* `gap_closed`, because `gap_closed`'s denominator and
   its persistence reference are both arm-specific. On the horizon-averaged `gap_mean` the
   ranges overlap outright (`frozen_ssl` −0.4106/−0.5201/−0.9366; `random_vit`
   −0.4493/−0.1619/−0.1447), and `random_vit`/s0 crosses persistence at h=1, earlier than
   any `frozen_ssl` seed.
3. **The confound is separated 3-vs-3 in the same direction as the outcome.**
   `random_vit`'s dynamics prior cleared the free-bits floor on 95.6–97.6% of steps;
   `frozen_ssl`'s on 74.6–90.0%. min(`random_vit`) = 0.9564 > max(`frozen_ssl`) = 0.8999 —
   the same complete separation as the outcome. `world_model.py` carries exactly this
   reasoning in the warning it raises below a 0.5 rate — *"Arms whose rates differ cannot be
   compared: the comparison would measure how much each prior trained, not the
   representations."* Both feature arms sit above that threshold, so the warning never fired
   for them (it fired three times, all `[cnn]`), but the argument applies unchanged to a
   3-vs-3 separation.

   The free within-arm check runs *against* the alternative it is usually offered as. Taking
   `kl_rate` against position `gap_final` across the six feature cells: within `frozen_ssl`
   r = **−0.97** (0.8999 → −0.8523, 0.7814 → −0.6470, 0.7455 → −0.6491 — more prior training,
   *worse* rollout); within `random_vit` r = **+0.99**; pooled r = +0.53, driven entirely by
   the between-arm split. Three points per arm settles nothing in either direction. What it
   does mean is that "the arm whose prior trained more rolls out better" is not a mechanism
   this data supports — it is an alternative the design cannot rule out, because `kl_rate` is
   separated 3-vs-3 with the outcome and cannot be disentangled at n = 3. Equalising the KL
   rate is the experiment that would, and it was not run.
4. **The ordering is not separated from imagination-sampling noise.** `run_job` passes
   `seed=job.seed` into `evaluate_rollout`, `fit_probes`, `reward_accuracy`,
   `filtering_report` and `filtering_gain` (`src/mbfps/eval/study.py:403-430`), and
   `evaluate_rollout` calls `torch.manual_seed(seed)` over a deliberately stochastic
   imagination (the plan's global constraint is that sampling is *always* stochastic and
   determinism comes from seeding). **The training seed and the imagination sampling seed are
   the same number, and no cell was ever re-evaluated at a second sampling seed.** So the
   three "seeds" per arm vary training noise and imagination noise together, one realisation
   each. There is no evidence that the complete separation survives resampling the imagination
   noise alone — and getting that evidence requires no retraining at all.

**Not supported:** "frozen DINOv2 pretraining hurts world-model rollout"; "pretraining buys
nothing over a stable prediction target"; any use of the word *significant* for a between-arm
difference in this study.

### The pixel arm is void: KL starvation, not a pixel result

| cell | steps/s | wall h | kl_rate | steps above floor | kl_dyn_max | loss_last20 |
|---|---|---|---|---|---|---|
| `cnn`/s0 | 0.5768 | 9.660 | 0.00090 | **18** / 20000 | 0.3999 | 0.12083 |
| `cnn`/s1 | 0.5513 | 10.107 | 0.00080 | **16** / 20000 | 0.7999 | 0.12006 |
| `cnn`/s2 | 0.5975 | 9.326 | 0.00075 | **15** / 20000 | 0.3678 | 0.12021 |
| `frozen_ssl`/s0 | 3.6592 | 1.543 | 0.89990 | 17998 | 36.2216 | 0.64148 |
| `frozen_ssl`/s1 | 3.8020 | 1.492 | 0.78135 | 15627 | 15.7070 | 0.46395 |
| `frozen_ssl`/s2 | 3.8850 | 1.454 | 0.74550 | 14910 | 1.8684 | 0.56451 |
| `random_vit`/s0 | 3.8504 | 1.467 | 0.97100 | 19420 | 27.5747 | 0.46219 |
| `random_vit`/s1 | 3.8561 | 1.465 | 0.97565 | 19513 | 6.1859 | 0.45893 |
| `random_vit`/s2 | 3.8648 | 1.461 | 0.95635 | 19127 | 10.8280 | 0.48863 |

`kl_rate_above_free_bits` is a **fraction of training steps**, not a KL magnitude:
`_kl_rate` counts steps whose `kl_dyn` exceeded `KL_FREE_BITS = 0.20` nat. Multiplied out,
the pixel arm's dynamics prior received a gradient on **15–18 steps out of 20,000** — a
ratio of about 1:1000 against the feature arms.

The mechanism is exact. In `kl_loss`, `dyn = KL(post.detach ‖ prior)` is `prior_net`'s
**only** gradient source in the entire objective (`observe` samples `z` from `post_logits`,
so `prior_logits` reach neither the latent nor any head), and `torch.maximum(dyn, floor)`
returns the constant whenever `dyn ≤ 0.20`. `dyn` and `rep` are numerically identical every
step — they differ only in which side is detached — so below the floor the posterior gets no
KL gradient either, and the whole KL contribution is the constant
0.5 × 0.20 + 0.1 × 0.20 = **0.12000**. The pixel arm's `loss_last20` is that constant plus
8.3e-4, 6e-5 and 2.1e-4 of everything else; its minimum logged loss over 20,000 steps is
0.12000–0.12001. **99.9% of the pixel arm's loss is a constant with zero gradient**, and it
got there by step 500–600 — 14–18 minutes into a 9.3–10.1 hour run. Roughly 19,400 of
20,000 steps per cell (**≈28.1 h** across three seeds, from the measured per-cell rates) were
spent after the arm was already dead.

What that produced, measured on the checkpoints. All figures below are computed on the
**9,450 probe rows** each cell's evaluation actually uses (189 windows × 50 frames, 20
validation episodes, gathered under the rollout's own context protocol) so the row set is
stated rather than implied:

- **Collapsed reconstruction target.** The embedding loss regresses the head onto the
  encoder's own detached output; a constant encoder is a zero-loss optimum of it. The pixel
  encoder's mean per-dimension std across those rows is **0.0051–0.0092** against an embedding
  RMS of 0.094–0.130; its total across-row variance is **0.21–1.4%** of the mean vector's
  squared norm (2.1e-3, 1.3e-2, 1.4e-2). The head's scale-free fit to that target is
  **negative** (1 − mse/var = −0.108 / −0.128 / −0.124) — worse than predicting the constant
  mean. The feature arms explain **+0.222 to +0.482** of theirs on the same statistic, with
  across-row variance 130–580% of the mean vector's squared norm.
- **Uniform posterior.** Categorical entropy 3.46523–3.46540 nat against ln 32 = 3.46574 —
  under 0.0006 nat of information per group out of 3.47. The feature arms carry 2.46–2.69.
- **Untrained prior, behaviourally.** The trained prior sits 0.00040–0.00281 nat below
  uniform; an *untrained* RSSM driven by the **same trained encoder** sits 0.00034–0.00036
  below. `prior_net` relative L2 displacement 0.198–0.217 with cosine-to-init 0.977–0.981,
  against 1.14–1.88 and 0.46–0.65 for the feature arms; the residual drift is ~50–55
  full-size Adam steps against a ~150–180 ceiling implied by 15–18 gradient events with a
  10-step momentum tail. "`prior_net` was frozen" is too strong; *`imagine()` for this arm
  samples `z` from an essentially uniform 32×32 categorical* is exact.

**"Pixels are worse than features" is not what was measured.** The pixel *encoder* is not
information-free: standardised, its raw embedding still supports a position/angle ridge probe
at R² **+0.2369 to +0.2736**. What died is everything downstream — posterior latent −0.0853,
predicted embedding −0.086. And `cnn` differs from the feature arms on three axes at once
(raw pixels vs cached features; 26,382,304 vs 12,320 trainable encoder parameters;
end-to-end vs frozen backbone), so even a healthy `cnn` cell would not isolate the
representation. `loss_last20` licenses no cross-arm comparison either, for three independently
sufficient reasons: the columns are different fractions of a differently-clamped objective;
the target variances differ by a factor of **1,400–13,100** on those same rows (total
across-row variance 0.075–0.279 for `cnn` against 395–977 for the feature arms); and the arm
with the smallest loss has the only negative scale-free fit.

**The direction of causation between the collapse and the floor is not established, and the
reason it is unmeasurable is a record projection, not physics.** The collapse is visible by
step 500–600, `dyn` starts below the floor in *every* arm, and the two feed each other: a
collapsed target leaves the posterior nothing to encode, and a clamped KL removes the only
term that could reshape either network. `train_world_model` **does** return the trace that
would settle it — `history["loss"]` (20,000 values) and `history["parts"]` (per-step embedding,
reward, continue, `kl_dyn`, `kl_rep`; `src/mbfps/training/world_model.py:78-84,153-157`) — and
`run_job` keeps only `loss_last20`, `kl_rate_above_free_bits` and `kl_dyn_max`
(`src/mbfps/eval/study.py:440-443`). The per-step embedding loss that was discarded is
precisely the collapse trace. The records store **no per-step `kl_dyn` at all**, only the rate
and the max.

`KL_FREE_BITS = 0.20` was calibrated on a 2,000-step `random_vit` run (spec §3.2a) and never
re-measured on the pixel arm, whose dyn KL cleared 0.20 on **15–18 of 20,000 steps** and
peaked at 0.368–0.800 nat. The floor was cleared — on about 0.08% of steps. That, not "never
above the floor", is the mechanism.

**What fixing it costs, measured, and why it is not simply "retune the floor".**

| option | cost |
|---|---|
| re-run the pixel arm at 3 seeds | **29.09 h** (76.6% of the study's 37.98 h) |
| one free-bits calibration value, 2,000 steps, pixel arm | 0.93–1.01 h |
| the docstring's own four-value calibration table | 3.7–4.0 h |
| pixel-only retune + confirmation run | ≈33 h |
| arm-parity-preserving fix (one new shared floor, re-run everywhere) | 37.98 h + calibration |

And `KL_FREE_BITS` **cannot be varied per arm without breaking the study's invariant**. It is
a module constant in `models/rssm.py` consumed as `kl_loss`'s default at one call site, with
no field in `Config`, `TrainConfig` or `EncoderConfig` and no route through `get_config`'s
`**overrides`. `utils/config.py`'s docstring is explicit — the arms *"must differ in exactly
one respect"* and `kind` is *"the ONLY field that may differ"*. Per-arm tuning means adding an
arm-varying field, after which the treatment/control contrast is confounded with optimisation.
Nor is retuning obviously the fix: the constant's own docstring records `free_bits 0.00`
producing `kl_dyn` 0.0001, annotated *(collapsed)*, and lowering the floor switches the `rep`
term back on, pulling the posterior toward the prior. The failure is in the **target** — a
self-predicted embedding a constant satisfies — so the next attempt at a pixel arm has to
address that (stop-gradient/EMA target, a variance or covariance regulariser, or a pixel or
reward decoder a constant cannot satisfy). The floor is a second question underneath it.

**The one number nobody may quote:** `cnn`/s2's `gap_closed = +1.3317`. It is one seed of
three (the report prints `finite 1/3` beside it precisely so this cannot be read as an arm
result), on a band of **0.397 map units** inside a 299-unit error, with 16 of its 45 horizon
steps flagged degenerate and a per-step `gap_closed` swinging from +0.26 to +11.04. Its
sibling seed had the floor at or above persistence on **45 of 45** steps and produced zero
usable measurements. Read carelessly, that row makes the pixel baseline look like the only
arm to exceed its encoder floor; `band_is_usable` exists to stop exactly that.

### The pixel arm's band is measured with a probe that reads a constant

Two different probes appear in this section and they must not be merged. The **band probe** is
`fit_probes`' embedding probe on the model's *predicted* embedding (2048-d), which
`evaluate_rollout` scores the band through: for all three `cnn` cells it selected ridge
**1e7 — the maximum of the five-point grid** — at a *negative* held-out selection R²
(−0.0357 / −0.0361 / −0.0366). **Criterion 4's latent probe** is a different fit on the
1536-d posterior latent (512 deterministic `h` + a 32×32 categorical `z`), scored on val rows
at R² −0.08533, also at ridge 1e7. The bullets below analyse the latter, where the collapse is
provable rather than suggestive:

- Coefficients at 1e7 are ~1/1500th of their unpenalised norm (‖w‖₂ 255.60 → 0.16433 for
  s0), and the intercept equals the train column mean to machine precision.
- The exact ridge→∞ limit, computed model-free from the targets, is **−0.0852610703**. The
  three seeds report −0.085327072 / −0.085331293 / −0.085329496 — 6.6e-5 to 7.0e-5 below it.
  They also agree with each other to 4.2e-6, but that agreement carries no information about
  the encoders and is not offered as evidence: once three models all sit on the ridge
  boundary their scores are pinned to a property of the split, and they would agree to five
  figures whether the encoders were identical or wildly different. The collapse is established
  by the non-circular evidence above — across-row variance of 0.21–1.4%, a negative scale-free
  head fit, and a posterior entropy within 0.0006 nat of ln 32.
- R² rises monotonically across all five decades and an off-grid 1e15 lands exactly on the
  limit (−0.0852610703). The boundary binds; the selector is choosing "throw all 1536 latent
  features away" because every less-shrunk alternative is worse. **Extending `RIDGES` would
  move the number by 6.6e-5 against a 0.356 deficit and change no verdict anywhere.**
- Physically: at 1e7 the probe's prediction varies by 0.15 and 0.23 map units in x and y
  against targets varying by 221 and 193, and the constant it collapsed to has a median
  position error of 269.7 units on a region spanning ~930 × 800.

That negative intercept-only level is a property of the **split**, not the probe: train and
val column means differ (e.g. cos a: +0.1269 vs −0.1771), so predicting the train mean on val
rows scores −0.08526; predicting the val mean scores exactly 0.0.

Consequence for the write-up's arithmetic: `cnn`'s `gap_closed` column is not a *weak* result,
it is an **unmeasured** one — a band whose three references all land at the no-information
level of the target (its model/persistence/floor curves span **291.83–299.29** across the
three cells at every horizon step, flat, at or slightly worse than a constant-centroid
predictor). Two chance-level estimates appear here and they differ because the row sets
differ: **286.72** map units is the mean distance to the centroid over all **10,305** rollout
rows (229 windows × 45 steps; on the h=45 rows alone it is 289.56, median 263.34), and
**269.7 / 273.7** are the median / mean intercept-only error over the 9,450 probe rows. Both
make the same point.

One caution against over-reading even the collapse: at the least-penalised ridge the pixel
latent probe *does* fit its training rows (‖w‖₂ ≈ 255) and scores −0.486 on val. That is
overfitting, not emptiness. The supported claim is that **the linear probe the spec mandates
finds nothing in the pixel latent that transfers**.

### Criterion 4: 1 of 9 cells, and that one is noise

`filtering_beats_embedding` asks whether the posterior latent beats the raw encoder embedding
of the same frame. It is a conjunction over all nine cells; **eight fail.**

| cell | `latent_r2` | `embedding_r2` | margin | pass |
|---|---|---|---|---|
| `cnn`/s0 | −0.0853 | +0.2703 | −0.3556 | False |
| `cnn`/s1 | −0.0853 | +0.2736 | −0.3589 | False |
| `cnn`/s2 | −0.0853 | +0.2369 | −0.3223 | False |
| `frozen_ssl`/s0 | +0.2892 | +0.3321 | −0.0429 | False |
| `frozen_ssl`/s1 | +0.2781 | +0.3451 | −0.0671 | False |
| `frozen_ssl`/s2 | +0.3376 | +0.3353 | **+0.0023** | **True** |
| `random_vit`/s0 | +0.2290 | +0.3195 | −0.0905 | False |
| `random_vit`/s1 | +0.2743 | +0.3241 | −0.0498 | False |
| `random_vit`/s2 | +0.1575 | +0.3256 | −0.1681 | False |

The single True cell is a sign flip in noise, and three independent arguments each close a
different escape route.

**(a) Sampling.** A 95% window-level block bootstrap of that margin — `_block_bootstrap_ci`,
probes held fixed, 189 windows, run at 2,000 resamples — is **[−0.0291, +0.0331]**. It
straddles zero, the point estimate is ~1/14th of the half-width, and P(margin > 0) = 0.55.
This interval is *not* valid nominal coverage: criterion 4's ridge is selected on the very
rows the bootstrap then resamples (`filtering_comparison` fits with val as the selection set),
and both probes and the selection are held fixed across resamples. For this one True cell that
bias runs *toward* the alternative, so "straddles zero" is conservative and the argument
holds. **The eight negative cells are not defended by their intervals** — in that direction
the bias is not conservative, and eight intervals excluding zero would read as eight
independent confirmations when it is eight applications of an interval that covers only
evaluation-window sampling. They rest instead on the conjunction and on the magnitudes:
−0.032 to −0.359, one to two orders above the True cell's +0.0023.

**(b) Estimator.** The margin is True at 1 of 5 ridge decades (1e3, where it is +0.0023);
at 1e−1 / 1e1 / 1e5 / 1e7 the same model reads −0.0458 / −0.0182 / −0.0904 / −0.0035. And 1e3
is the decade the five-way maximum picked on the very rows the number is scored on.

**(c) Seeds.** `frozen_ssl`'s own seed-to-seed SD on this margin is 0.0352, **15.6× the
effect.** And it is moot: eight Falses fail the conjunction eight times over. Reporting it as
a partial success would be the single most misleading sentence available.

Two disclosures the field names invite readers to miss. First, both criterion-4 R² are
**optimistically biased levels by construction** — `filtering_comparison`'s docstring says so
— because the ridge is selected on the exact rows the probe is then scored on. The
*comparison* is fair (both sides get the same five-way maximum on the same rows); the
*levels* are not, and must never be quoted as clean held-out numbers. Second, the record's
`probe.*` block is a **different fit** and does not describe criterion 4 at all: it uses the
model's *predicted* embedding, fit on 16 train episodes and scored on a 4-episode train
selection split, while criterion 4 uses the *raw encoder* embedding fit on 20 and scored on
20 val episodes. Criterion 4's own selected ridge is never written to any record. The
apparent `+0.2703` vs `−0.0357` inconsistency is two similarly-named fields, and both
reproduce bit-exactly. The genuinely interesting fact underneath: on the *same* val rows the
model's predicted embedding also collapses (−0.0857, ridge 1e7) while the raw encoder
embedding does not (+0.2703, ridge 1e1) — **at identical width**. Both are 2048-d
(`EncoderConfig.embed_dim = 2048` for every arm, and `WorldModelHeads.embedding` is
`_mlp(LATENT_DIM, 512, embed_dim)`), so this is not a feature-count effect.
**The pixel arm's position information survives its encoder and is destroyed by everything
after it.**

Finally, failing criterion 4 is **not** by itself evidence that `h` is inert, and
`filtering_gain`'s docstring says so: the criterion pits a 1536-d latent (512 deterministic +
a 32×32 categorical worth at most 160 bits about frame *t*) against 2048 raw encoder floats,
so a model can carry real history and lose on the bottleneck alone.

### The filtering gain, and the ridge selection that makes its sign undecidable

The bottleneck-free companion — R²([e ⊕ h] → s) minus R²([e] → s), with weights from the fit
split, the ridge selected on a **third, held-out** selection split, and the score on val — is
the diagnostic that answers the question criterion 4 cannot. It is **not a gate criterion**;
nothing in `evaluate_gate` reads it. All values below were re-derived from the shipped
checkpoints and reproduce the records exactly.

| cell | gain (as run) | 95% CI | joint / embed ridge | matched @1e3 | matched @1e5 |
|---|---|---|---|---|---|
| `cnn`/s0 | −0.1165 | [−0.1333, −0.1015] | 1e1 / 1e1 | −0.1176 | −0.0070 |
| `cnn`/s1 | −0.1790 | [−0.2037, −0.1586] | 1e1 / 1e1 | −0.1707 | −0.0088 |
| `cnn`/s2 | −0.1197 | [−0.1347, −0.1066] | 1e1 / 1e1 | −0.1145 | −0.0067 |
| `frozen_ssl`/s0 | +0.0316 | [+0.0123, +0.0506] | 1e3 / 1e3 | +0.0316 | +0.0259 |
| `frozen_ssl`/s1 | +0.0407 | [+0.0169, +0.0632] | 1e3 / **1e5** | +0.0279 | +0.0270 |
| `frozen_ssl`/s2 | +0.0763 | [+0.0512, +0.1023] | 1e3 / **1e5** | +0.0504 | +0.0495 |
| `random_vit`/s0 | +0.0239 | [+0.0075, +0.0401] | 1e3 / 1e3 | +0.0239 | +0.0476 |
| `random_vit`/s1 | −0.0337 | [−0.0527, −0.0146] | **1e5** / 1e3 | **+0.0296** | +0.0543 |
| `random_vit`/s2 | −0.0442 | [−0.0635, −0.0248] | **1e5** / 1e3 | −0.0241 | +0.0344 |

As run this reads as a clean treatment effect: `frozen_ssl` positive 3/3, `random_vit`
positive 1/3, arm means +0.0495 vs −0.0180. **It does not survive the estimator.**

- **Two decades are live, and neither is pre-registered.** On the scored rows themselves both
  feature sets peak at the same ridge in all nine cells (1e1 for `cnn`, 1e3 for all six
  feature cells; 18 of 18 side-argmaxes agree within a cell). On the *held-out selection
  split* — which is the unbiased selector by design, and whose whole point is that
  "no part of the number on the scored rows was tuned on those rows" — four of those eighteen
  sides prefer 1e5 instead. So there are two defensible decades. 1e3 is the scored-row
  optimum, which is exactly the optimistic procedure this write-up condemns for criterion 4
  and which `filtering_gain`'s own docstring warns "would manufacture a positive gain out of
  the selection alone". 1e5 is what the honest selector picks on four of the eighteen sides.
  The held-out choices are **not errors** — the inter-decade margins on the selection split
  are 0.0037, 0.0089, 0.0104 and 0.0127, and disagreement at that scale is expected sampling
  variation, not a defect. Calling 1e3 "what the scored rows support" asserts an in-sample
  optimum as ground truth on 189 windows.
- **The disagreements are nonetheless arm-correlated, in the arm-flattering direction.** All
  four picked the heavier penalty. Both `frozen_ssl` disagreements are on the baseline
  (embedding) side; both `random_vit` disagreements are on the treatment (joint) side. Their
  costs on the scored rows: +0.0127, +0.0259, −0.0633, −0.0200. Two of the four are
  independently confirmed by a field already in the shipped records: criterion 4's
  `embedding_r2` is the same fit on the same rows with the ridge chosen in-sample, and it
  equals the gain's `embedding_r2` to the printed digit in the seven cells that selected the
  scored-row argmax, exceeding it by exactly +0.0127 and +0.0259 in the two `frozen_ssl` cells
  that did not.
- **The arm ordering of the matched gain is a property of the decade, not of the
  representation.** Matched at 1e3 the arm means are **+0.0366** (`frozen_ssl`) vs **+0.0098**
  (`random_vit`), a difference of **+0.027** (per-seed ssl−vit: +0.0077, −0.0016, +0.0745;
  SE 0.0240, t(2) = 1.12, resting entirely on seed 2). Matched at 1e5 the arm means are
  **+0.0341** vs **+0.0454** — a difference of **−0.011**, with `random_vit` positive **3/3**
  and ahead. The matched gain does not order the arms; its sign flips with the decade. There
  is therefore no fraction of the +0.0675 as-run arm gap that can be called "real": the
  residual is not a share of anything, it is unidentified.
- **Individual cells flip too.** `random_vit`/s1 **changes sign** and is "significant" in both
  directions: −0.0337 [−0.0527, −0.0146] as run, +0.0296 [+0.0092, +0.0508] matched at 1e3 —
  same checkpoint, same rows, same bootstrap seed.
- **The asymmetry does not average out with more seeds**, because which side sits on the
  knife-edge is a property of the arm's ridge-response curve. Between 1e3 and 1e5 on the
  scored rows, `frozen_ssl`'s embedding side moves only 0.0127–0.0259 and its joint side
  0.0137–0.0268 — both flat — while `random_vit`'s embedding side moves 0.0786–0.0928. And
  the decisions are coin flips: 11 of 18 side-decisions have |ΔR²| < 0.015 on the selection
  split.

The report already prints a WARNING that the nine cells did not all select the same decade
and that the group **mean**'s sign is not interpretable. That warning is correct and does not
go far enough on four counts: the per-cell gains and their "CI excl 0 = True" column are
equally compromised; the one pair of cells sharing both ridges (`frozen_ssl`/s0 +
`random_vit`/s0 at 1e3/1e3) is the study's **only** within-estimator cross-arm contrast and
the report averages it into a single pooled number (+0.0278), hiding the contrast itself
(+0.0077, paired window bootstrap [−0.0118, +0.0268], 3.5× the point estimate wide); the
wording implies noise that seeds would average out; and it offers no remedy though a
common-ridge score exists and is unambiguous.

**Conclusion: the apparent treatment effect in the gain is not separable from estimator
selection, and no matched decade rescues a treatment claim — because the two defensible
decades order the arms in opposite directions.** A note on the codebase's carried
"~0.10 per decade" constant: measured across all 72 adjacent-decade steps in all nine cells,
the median move is 0.0888 (range 0.0011–0.3666), but the step actually being decided
(1e3 vs 1e5) moves the feature arms' levels by only 0.0127–0.0928 — the same order as the
gain, not five times it.

The one gain result robust near the level optimum is `cnn`'s: −0.1165/−0.1790/−0.1197 as run
and −0.1176/−0.1707/−0.1145 matched at 1e3, an order of magnitude outside any CI. Appending
`h` to the pixel embedding measurably **destroys** the readout at every decade where the
readout works. (At 1e5 and above the gain collapses to ~0 for a degenerate reason — both
probes fall to R² ≤ 0.03, so it is the difference of two useless probes.)

**What every CI in this study covers.** `_block_bootstrap_ci` holds both fitted probes fixed
and resamples only the 189 scored windows. It covers evaluation-window sampling and **none**
of: the training seed, the probe fit, the ridge selection, the imagination/posterior sampling,
or the episode split. The sampling term is worth naming twice, because in this study it is not
even separable: the imagination seed *is* the training seed, so "seed variation" and "sampling
variation" arrive as one number and neither can be read alone. Two demonstrations from the
table above: `random_vit`'s three per-cell CIs all exclude zero and **disagree in sign**;
`frozen_ssl`'s three all exclude zero while a 2-df interval on the arm mean, [−0.0091,
+0.1082], does **not**. For `cnn`, the only arm whose three cells share an estimator, the
between-seed SD is **0.0352** against a mean within-cell bootstrap SE of **0.0089** —
**3.9× in SE, ~15× in variance.** The reported interval measures roughly a fifteenth of the
variance that actually moves this number. The gate metric `gap_closed` carries **no interval
of any kind**, and criterion 4 is a bare boolean over two point estimates. Every uncertainty
statement in the report attaches to the one diagnostic that does not gate.

### Criterion 3, reward: reported, deliberately not gated

| quantity | value |
|---|---|
| scored steps (20 val episodes) | 9,949 |
| reward events in them | **2** |
| `baseline_mse` (constant predictor), bit-identical in all nine cells | 2.010250e-04 |
| `r2` range across all nine cells | −0.00144 … +0.00149 |
| mse / baseline_mse | 0.99851 … 1.00144 |
| `is_degenerate` | True, all nine |

`reward_reported` passes: it checks that `mse` and `baseline_mse` are finite and `n_steps > 0`
in every cell. It is a **bookkeeping check wearing a substantive name**. Eight of nine cells
are strictly *worse* than the constant-mean predictor and the best (`frozen_ssl`/s2) beats it
by 3.0e-7 in MSE. Nothing in this table ranks anything; it must be read as
uninformative-by-construction, which is exactly what `is_degenerate` flags and what
`aggregate.py`'s own comment argues — gating on a non-degenerate target *"would fail the
milestone for a property of the scenario that was known before the study started."*

What the table *does* establish is a real consistency check: every arm was scored against a
bit-identical target on a bit-identical split, and the head produced shape-aligned finite
predictions on 9,949 held-out steps in all nine cells. Note also that "2 events" is a property
of the 20-episode eval slice, not of the scenario — the plan's separately measured figure is 6
reward events in 19,424 steps, and the full 122-episode dataset carries 21 goal events across
16 episodes and 59,511 steps.

**The `continue` head is trained on every step and measured nowhere.**
`WorldModelHeads.continue_` carries a BCE term inside the summed loss
(`src/mbfps/training/world_model.py:71-77`), and no record field, no report column and no gate
criterion touches it — `loss_last20` is the *summed* loss and the per-term split is discarded
with the rest of `history["parts"]`. Terminals are ~0.03% of steps. Given that the reward head
on a comparably sparse target is indistinguishable from a constant predictor in 9/9 cells, the
prior on the continue head is that it is equally degenerate, and M3 never looked. This is the
component M4's actor terminates and discounts imagined trajectories with.

### The cost actually paid

| arm | cells | steps/s (range) | hours | share |
|---|---|---|---|---|
| `cnn` | 3 | 0.5513–0.5975 | **29.093** | 76.6% |
| `frozen_ssl` | 3 | 3.6592–3.8850 | 4.489 | 11.8% |
| `random_vit` | 3 | 3.8504–3.8648 | 4.393 | 11.6% |
| **total** | 9 | — | **37.975** | |

Against the plan's carried estimate of 33 laptop-hours that is **+15.1%**. The arm ratio is
**6.1–7.0×** across all cell pairs (6.3–7.0× same-seed on steps/s, 6.26–6.90× same-seed on
wall seconds, 6.48–6.62× on arm mean hours), not the plan's 5.9×, and the miss is on the pixel
arm: measured 0.575 steps/s mean against a predicted 0.67 (86% of prediction), while the
feature arms hit 3.82 against 3.98 (96%). The 6–7× gap is an **input-pipeline** cost —
26,382,304 encoder parameters over raw 112×112×3 pixels every step versus 12,320 parameters
over features cached once at collection time — and is not evidence about representation
quality.

Evaluation is negligible: `seconds` is the whole cell while `steps_per_second` is training
only, and the difference is 85.9–110.2 s per cell, **14.2 min total = 0.6%** of the study.
Training is 99.4% of the bill. That figure is also the best anchor for sizing M4: 86–110 s per
cell covers the full evaluation block — 229 windows × 45 imagined steps, five probe gathers,
reward, criterion 4 and the gain with a 1,000-resample bootstrap — so imagination on a frozen
feature checkpoint runs at the order of 10² imagined transitions per second on this machine,
not the 0.55–3.9 *training* steps/s that produced the 37.975 h. An M4 actor trained in
imagination on frozen checkpoints does not cost what this study cost; its bill will be
dominated by actor/critic updates and by whether the world model stays frozen.

**Provenance, and the one question it opens is closed here.** The study was **not** the single
unattended GPU run Task 7 describes. Measured throughput is at or below the plan's own
local-laptop baseline; `study.log` covers 7 of 9 cells (both earlier `frozen_ssl` cells predate
it), spans 2026-09-08 to 2026-09-10 across **five** driver invocations, and shows `cnn`/s0
started at 11:54:29 producing no step output and relaunched 8.6 h later. The obvious follow-up
— did the same code produce all nine cells? — has an answer: the last commit touching `src/` or
`scripts/` is **c1d1d4f, 2026-09-06 23:06**, the earliest record mtime is
`result_frozen_ssl_seed0.json` at 2026-09-07 16:13, and the latest is `result_cnn_seed2.json`
at 2026-09-10 10:17. One code state produced all nine cells. That had to be reconstructed from
file mtimes, because **the record schema stores no git SHA, no device, and no lr/batch/
optimiser** — a mid-study code change would be undetectable from the data alone. No provider,
instance type, hourly rate, or environment-parity test count was recorded, so the plan's parity
gate has no evidence behind it. And **37.975 h is the sum of nine per-job `seconds` values, not
a calendar span** — the driver is resumable and the calendar span is not recoverable from the
records.

### Logged, not active: the gate can pass without criterion 5

`evaluate_gate`'s `passed` is the conjunction of **eight** computed criteria;
`invariant_tests_green` appears only under `not_evaluated`, and `main` maps `passed` straight
onto the exit status. No script under `scripts/` runs pytest. Demonstrated end to end: a
synthetic all-pass copy of the nine records (`gap_final = 0.5`, `steps_degenerate = 0`,
`latent_beats_embedding = True`) makes `report_study.py` print `GATE: PASSED` and **exit 0**
while still printing `[ n/a] invariant_tests_green`.

**It cannot have affected this verdict**, and doubly so: three of the eight computed criteria
fail, so `passed` is False and the status is 10 whatever criterion 5 says; and the suite is
independently green. The exposure is narrow and **machine-readable-only** — a human reading
the report sees the criterion named as NOT EVALUATED with a wrapped explanation of what to
run; a wrapper reading `$?` alone cannot distinguish "criterion 5 checked and green" from
"criterion 5 never checked". Both activation conditions (all eight computed criteria passing
*and* a red suite) would have to hold together. Logged as a defect to close before any
unattended gate run, not as a fault in this result.

For completeness, the five exit statuses were each reproduced: 0 (gate passed), 7 (no
records), 8 (a file's name and its arm/seed disagree), 9 (a matching file that will not load
back), 10 (report printed, verdict NOT PASSED). 2 (argparse) and 1 (traceback) are excluded
deliberately, and 3–6 are reserved by `run_study.py`.

### What the study establishes cleanly

Worth stating, because a section that only lists failures reads as weaker than the evidence.

- **DINOv2 features support a better linear position/angle readout than randomly-initialised
  ViT features.** This is the study's one confirmed pre-registered prediction, and it is
  confirmed at the maximum strength this design can express: complete 3-vs-3 separation on
  **two independent measures**, both in the spec's pre-registered direction, exact one-sided
  permutation p = 1/20 = 0.050 — the design's floor.
  - Encoder floor at h=45 (lower is better): `frozen_ssl` 124.49 / 117.72 / 122.62 against
    `random_vit` 157.12 / 138.18 / 155.34. max(`frozen_ssl`) < min(`random_vit`).
  - Criterion 4's raw-encoder `embedding_r2` (higher is better): `frozen_ssl`
    0.3321 / 0.3451 / 0.3353 against `random_vit` 0.3195 / 0.3241 / 0.3256.
    min(`frozen_ssl`) > max(`random_vit`).

  **The evidentiary standard, stated once and applied to both directions.** The floor
  difference, the criterion-4 readout difference and the `gap_closed` ordering are *all*
  complete 3-vs-3 separations at the design's exact floor. What distinguishes them is not
  strength, it is pre-registration: the two readout results run in the direction spec §3.1
  registers, so a one-sided 1/20 is legitimate there; the `gap_closed` ordering runs in a
  direction the spec registers no branch for, so only the two-sided 2/20 = 0.100 applies to
  it. And what the readout result does **not** extend to: readout quality is not rollout
  quality, and the arm with the better readout is the one that rolls out worse.
- The split is **byte-identical across all nine cells** (verified by episode-name hash), so
  cross-cell comparison is legitimate and no result is a split artefact.
- **Arm parity holds.** `encoders.py` is still the only module whose behaviour varies by arm;
  re-running the plan's own grep, every other `cfg.arm` / `.kind` reference is a log prefix, a
  checkpoint filename or a record label (in `world_model.py` and `autoencoder.py`).
  `frozen_ssl` vs `random_vit` is a genuine one-variable contrast.
- The record schema preserves NaN losslessly (JSON `null` plus a `nonfinite` map, restored
  only by `load_record`), and the aggregation **refuses to average a cell that measured
  nothing** — a NaN `gap_final` is the contract for a non-positive band, not a parse failure
  and not a zero. Reading these files with plain `json.loads` silently turns the study's most
  important failure mode into `None`.
- The report prints its own ridge-disagreement warning, its own finite/degenerate/floor
  columns, and names the criterion nobody checked rather than letting absence read as a pass.
- The training code wrote the pixel-arm finding **at the time**, three times, in
  `study.log` — and the gate has no criterion that reads it.
- The irony worth naming: `random_vit`, the arm designed to be the null, is the only one whose
  numbers need no qualification — 0 degenerate steps on both metrics, 45/45 finite gaps.

### What the evidence does NOT support

- **"`cnn` closed 133% of the persistence-to-floor gap"** / any `cnn` row read from the mean
  column. One cell of three, a 0.397-unit band inside a 299-unit error, 16/45 degenerate
  steps, measured through a probe that reads a constant.
- **"Pixel-space world models are worse than feature-space ones"** (or the reverse). The
  pixel arm trained no dynamics prior; the harness's own warning refuses the comparison; and
  `cnn` differs from the feature arms on three axes at once.
- **"`loss_last20` ranks the arms"**, in either direction. Within an arm it is a legitimate
  convergence trace; across arms it licenses nothing.
- **"DINOv2 pretraining hurts"** / **"pretraining buys nothing"**. The ordering is real and
  maximally seed-consistent for this design, and it is confounded by a KL rate separated
  3-vs-3 in the same direction, in a direction the spec registers no branch for, and with the
  imagination sampling seed tied to the training seed.
- **"The gain shows `frozen_ssl`'s `h` carries history `random_vit`'s does not."** The matched
  gain's arm ordering reverses between the two defensible decades: +0.027 in `frozen_ssl`'s
  favour at 1e3 (t(2) = 1.12, resting on one seed), −0.011 against it at 1e5 with
  `random_vit` positive 3/3.
- **Any p < 0.100 two-sided, any t-test, any between-arm "significant".**
- **"The deterministic state `h` is inert."** Unresolved, not refuted — criterion 4's two
  sides are unmatched by construction.
- **"The world model learned no dynamics."** Positive at h=1 in 5 of 6 feature cells, though
  by 0.26–4.7 map units on 3–10-unit bands with no interval; crossover between h=1 and h=17.
- **"There is a usable short imagination horizon."** No horizon has all six feature cells
  above persistence; at h=5 half are already worse than a do-nothing baseline; one control
  cell is behind from the first imagined step.
- **"The arms do not separate, so representation choice does not matter."** Failing to reject
  at a design whose two-sided p-floor is 0.100 is not evidence of equivalence.
- **"Reward prediction is accurate"** *or* **"reward prediction failed."** Two events.
- **"20,000 steps is a converged budget."** No learning curve is persisted in any record. On
  the one statistic that holds in all seven logged cells — **mean loss over the last 5,000
  steps against mean loss over the first 15,000**, parsed from `study.log`'s 200 log points per
  cell — the loss is **higher at the end in 7 of 7**: `cnn`/s0 +0.20%, `cnn`/s1 +0.24%,
  `cnn`/s2 +1.79%, `frozen_ssl`/s2 +39.40%, `random_vit`/s0 +5.41%, `random_vit`/s1 +23.10%,
  `random_vit`/s2 +5.27%. (Final-quarter versus preceding-quarter means give the same
  direction in 6 of 7: −20.97 / −9.43 / −11.49 / −2.60 / −0.52 / +0.07 / −1.74 percent, where
  negative means the loss rose.) This is an unsmoothed training loss logged every 100 steps
  with no validation curve, so it bounds nothing about convergence beyond "not visibly still
  descending". "Train longer" is a live, unanswered objection.
- **"The negative result generalises."** One scenario (`my_way_home`), one frozen 122-episode
  dataset from a random+scripted mix in which 16 of 122 episodes reach the goal, one split
  realisation, one context/horizon (5/45), one hyperparameter set, one budget, one imagination
  sampling realisation per cell.

### Measurement caveats that must travel with these numbers

- **Two denominators run through one report.** `evaluate_rollout` uses all 24 val episodes
  (229 windows); the probe, gain and reward tables use only the first 20 (189 windows /
  9,450 probe rows / 9,949 reward steps). Tables in the same report describe different row
  sets — including the two chance-level estimates in the pixel section.
- **The imagination seed is the training seed.** `run_job` passes `seed=job.seed` into
  `evaluate_rollout`, `fit_probes`, `reward_accuracy`, `filtering_report` and `filtering_gain`
  (`study.py:403-430`), and `evaluate_rollout` seeds a deliberately stochastic imagination.
  Every cell is one realisation, never repeated. No number in this study is separated from
  imagination-sampling noise.
- **Persistence is not the true displacement.** `rollout.py`'s docstring says the persistence
  error *is* the displacement; that holds only in the perfect-probe limit. Measured mean
  displacement at h=45 over the 229 rollout windows is **126.22** map units, against reported
  persistence of **164–187** (`frozen_ssl`, 1.30–1.48× per cell), **194–209** (`random_vit`,
  1.54–1.66×) and **298–299** (`cnn`, 2.36–2.37×). Each arm is scored against a *different*
  baseline for identical physical motion, which is precisely why raw errors do not travel
  across arms and only the ratio does.
- **Part of the band is manufactured by the protocol.** The floor's filtering context grows
  from 5 to 50 frames across the horizon while persistence stays frozen at 5. Measured
  directly: the floor error *falls* from h=1 to h=45 by 27.1 / 32.6 / 9.3 (`frozen_ssl`) and
  23.9 / 28.8 / 34.0 (`random_vit`) map units, i.e. **22.5–63.3% (mean 48.9%) of the final
  band width**. That derivation conflates two effects — the growing filtering context and the
  changing target — and the repository's own context-isolated measurement attributes less to
  the first: `gather_probe_data`'s docstring reports probe error 317.3 / 223.4 / 213.5 / 215.3
  at contexts 1 / 5 / 20 / 50, so filtering depth alone is worth **roughly 8 units**, which is
  11.5–19.4% of these cells' final bands. Take the 8-unit figure as the isolated attribution
  and the 22–63% as the total h=1→h=45 movement. Either way the direction of the argument is
  the same, and it is the one that matters: because every feature-arm gap is negative, a wider
  band makes the arms look **less bad** than they are.
- **Spec §3.4's coherence claim is void in the implementation.** §3.4 states that "the
  encoder-embedding reference and the persistence baseline coincide in the imagination setting
  — probing the last context embedding *is* the persistence prediction", and concludes that
  "the two gate criteria therefore reduce to one measurement, which is a sign the design is
  coherent". They do not coincide. Persistence is the last context step's **predicted**
  embedding held (`rollout.py`: "Predicted, not raw, so it shares the probe's distribution"),
  while criterion 4's embedding side is the **raw encoder** embedding under a probe fit on the
  train split and scored on val (`filtering_report`). Two different fits on two different
  arrays. The criteria are two measurements, not one.
- **The implemented floor is §3.2a's, not §3.2's.** The spec's table says "probe the encoder
  embedding of the real frame"; the implementation runs the posterior on real frames through
  the shared embedding head, so all three references pass the identical pipeline. The
  correction is well argued and was measured (steps with band ≤ 0: 9/45 → 2/45; median band
  +21.15 → +53.23), but the write-up must say which definition produced these numbers.
- **`gap_closed` is not bounded in [0, 1].** The spec says "under normal conditions" it is;
  measured, it ranges **−152.603** (`frozen_ssl`/s0 angle) to **+12.810** (`cnn`/s1 angle)
  across arm-metric cells, and only **9 of 18** stay inside [−2, +2] over every step where the
  gap is defined — 8 if a cell with any non-finite step is counted against it. All nine are
  feature-arm cells: the three `frozen_ssl` position cells and all six `random_vit` cells
  (`random_vit`/s2 angle is the one with a single non-finite step). `cnn`/s1 position produced
  no finite gap at any of its 45 steps.
- **The angle metric carries almost no weight, and none for `cnn`.** True mean |Δangle| at
  h=45 is 88.95°; the six feature cells sit within ~11° of the 90° chance level and the three
  `cnn` cells about 12° above it (101.78 / 101.85 / 102.03). `cnn`'s angle *band* is under
  0.05° wide (0.0457 / 0.0361 / 0.0488) on a ~102° error, with 14–18 degenerate steps and
  20–30 floor-above-persistence steps per seed; its positive angle mean (+0.6914) is a warning
  about the metric, not a finding about the arm. Angle is correctly excluded from criterion 1
  — but it *does* count toward `band_is_usable`, and `frozen_ssl`'s entire failure of that
  criterion is **one angle step in seed 0** (its position band is 0/45 degenerate, 0/45
  floor-above-persistence, 45/45 finite). `random_vit` alone would pass `band_is_usable`; it
  fails at the conjunction over arms, driven by `cnn`'s 18.
- **The plan's carried constant "0/45 degenerate steps" did not survive the study.** Measured:
  up to 18 degenerate steps and up to 45 floor-above-persistence steps in a single cell. Nor
  did spec §3.3's note that "the band is roughly 25% of the error magnitude" — measured
  **22–37%** for the feature arms at h=45 (17–27% on the per-step relative median) and
  0.03–0.26% for `cnn`.

### What this study says about the spec's own open questions

Spec §9 leaves two questions for M3 and this study answers both, one of them in the opposite
direction to the spec's own annotation.

- **"Is 45 imagined steps the right horizon for `my_way_home`?"** §9 says the error-vs-horizon
  curve answers it directly, and it does. Model position error reaches 95% of its h=45 value
  only at h = **26–40** for the six feature cells (40/35/27 for `frozen_ssl`, 26/32/32 for
  `random_vit`); persistence saturates earlier, at h = **12–32**. True mean displacement at
  h=45 is 126.22 map units against persistence errors of 164–209. So 45 is long relative to
  the motion this scenario produces, the model's own error has not saturated by then, and the
  gate metric is read at the horizon where the model is furthest behind the baseline.
- **"Does `gap_closed` behave well when an arm's floor is very close to persistence?"** §9
  marks this **struck through and ANSWERED** by the §3.2a pipeline fix. **Reopen it.** The
  pixel arm is direct evidence that routing all three references through one pipeline is not
  sufficient when the encoder itself is degenerate: `cnn`/s1 has the floor at or above
  persistence on 45 of 45 steps and produced zero usable measurements; `cnn`/s0 on 37 of 45;
  and `cnn`/s2's surviving ratio is +1.3317 on a 0.397-unit band. The fix removed a
  distribution mismatch. It did not make the ratio safe on a collapsed encoder — which is why
  `band_is_usable` had to be added as a separate guard.

### What is open, ranked by what it costs to close

Cheapest first, so the ordering is usable rather than a list.

1. **Horizon-truncated gap** — *free, from the stored curves.* Done above: the positive-cell
   count per horizon step. Nothing further to run.
2. **Common-ridge gain re-score** — *minutes; measured 32–43 s per cell on this machine for a
   full three-split gather, both fits, all five decades and a 1,000-resample bootstrap.*
   Answers open question "does `h` carry history at all" in the only form that is
   interpretable: both sides at one pre-registered decade. Done for the tables above; what
   remains is to pre-register the decade before the next study rather than after.
3. **Sampling-seed repeat** — *no retraining, ~1–2 min per cell.* Re-run `evaluate_rollout` on
   the nine shipped checkpoints at 2–3 extra sampling seeds. Until this exists, the
   treatment/control ordering is not separated from imagination noise, which is a stronger and
   more honest limit than the permutation p-floor argument.
4. **Action-shuffled imagination and k-step re-grounding** — *RUN, 2026-09-11; see the
   Addendum above.* Shuffled and resampled actions are null in 9/9; a held-action contrast
   (FORWARD − NOOP) responds in 3/3 `frozen_ssl` cells at z = 4.2–7.4 with the physical
   sign. The prior is action-conditioned to the horizon-average and blind to order and
   counts. M4 is not blocked by action-blindness; the blocker is compounding error past
   k ≈ 5. Two follow-ups the ladder itself surfaced:
   - **Probe-free embedding-space reading of each rung** — *~2 min per cell.* Per-window L2
     between intervened and real head embeddings against a two-real-imaginations noise
     reference. Gives the ladder an effect size that does not pass through a ridge probe
     whose R² is 0.30–0.38 on the feature arms and −0.036 on the pixel arm.
   - **Pooled cross-cell statistic for the ladder** — *minutes, no re-scoring.* The per-cell
     verdicts disagree across `random_vit` seeds (nominal, nominal, null) while the
     cross-cell sign is 9/9. A meta-analytic z over cells, or a mixed model over the shared
     windows, would replace nine verdicts with one.
5. **Continue-head accuracy / AUC on the same 20 val episodes** — *minutes, one forward pass.*
   The head M4 terminates imagined trajectories with, currently trained and never measured.
6. **Persist `history["loss"]` and `history["parts"]`** — *one line in `study.py:440-443`, zero
   compute.* Converts two of the open questions below from unanswerable to measurable.
7. **Re-run the pixel arm at 3 seeds** — *29.09 h*, and only after the target-design question
   is settled.
8. **Arm-parity-preserving change (one new shared floor, re-run everywhere)** — *37.98 h plus
   calibration.*

The open questions themselves:

1. **The KL-rate confound between treatment and control.** `random_vit` 0.956–0.976 against
   `frozen_ssl` 0.746–0.900, separated 3-vs-3 in the same direction as the outcome. The one
   within-arm check available contradicts the "more prior training rolls out better"
   mechanism inside `frozen_ssl` (r = −0.97) and supports it inside `random_vit` (r = +0.99),
   on three points each. The experiment that separates representation from prior-training was
   not run.
2. **Horizon failure or capacity failure?** Nothing here distinguishes them: one horizon was
   run. The crossover between h=1 and h=17 is the shape of error accumulation, but whether a
   larger RSSM, a longer budget, or a shorter target horizon fixes it is untested. Item 4
   above is the cheapest discriminator.
3. **Convergence.** No learning curve is persisted in the records, though
   `train_world_model` returns one. The surviving log shows the loss higher over the final
   5,000 steps than over the first 15,000 in 7 of 7 logged cells. Open because of a record
   projection, not because of physics — see item 6.
4. **Whether the pixel collapse is an encoder, probe, or budget failure** — and which of the
   collapsed target and the clamped KL moved first. The discarded `history["parts"]` carries
   the per-step embedding loss that would show it. Same cause, same one-line fix.
5. **Whether `h` carries history at all.** Criterion 4 cannot answer it (unmatched sides); the
   gain that could has two defensible decades that order the arms oppositely. The clean
   version — both sides at a common decade, with the decade choice pre-registered — is item 2.
6. **`KL_FREE_BITS = 0.20` is validated for the control arm only.** The constant's docstring
   in `rssm.py` names no arm ("Measured over 2,000 steps on the real dataset"); spec §3.2a does
   — "All figures come from a 2,000-step `random_vit` run". So 0.20 is calibrated on
   `random_vit` and was never re-measured on the pixel arm or on `frozen_ssl`. What the records
   show is that at 0.20 the feature arms clear the floor on 75–98% of steps, consistent with
   the docstring's target, and the pixel arm on 0.08%.
7. **Split sensitivity.** One split realisation across all nine cells — excellent for internal
   validity, zero evidence about robustness.

### Deferred

- **Any re-run of the pixel arm.** 29.09 h at three seeds, and the fix is a target-design
  question (stop-gradient/EMA target, variance or covariance regulariser, or a decoder a
  constant cannot satisfy), not a free-bits value. Deferred until that design exists.
- **Per-arm free-bits tuning.** Structurally unavailable without breaking the one-variable
  invariant; an arm-parity-preserving change means re-running all nine cells (37.98 h plus
  calibration).
- **More seeds.** The correct response to a narrow separation, and the only thing that raises
  the design's p-floor of 0.100 two-sided. Not run here.
- **Closing the gate's criterion-5 hole** (have `report_study.py` invoke the suite, or add a
  tenth criterion fed by an explicit test-status input).
- **A `kl_rate` gate criterion.** Currently reported and warned about but not gated; the
  existing warning threshold already implies the natural rule, e.g. `kl_rate >= 0.5` per cell.
- **A fresh mutation pass over the study code.** The trust argument above rests on Tasks 1–6's
  own mutation tables plus the 851 tests; no mutation table was re-run against the code state
  that produced these nine records.

### Carried constraints for M4

- **`kl_rate` is a precondition for reading ANY rollout number from a cell.** Below the
  free-bits floor the dynamics prior receives no gradient, `imagine()` runs at
  initialisation, and every downstream number measures the initialisation. Check it before
  reading a gap, a band, or a probe.
- **The dynamics prior is action-conditioned to the horizon-average and blind to order and
  counts** (Addendum above). An M4 actor will find that holding an action moves the imagined
  agent with the right sign, and that two plans with the same action histogram imagine the
  same future. Do not attribute an actor's failure to plan to a missing action pathway; the
  pathway exists and is coarse.
- **Do not imagine with the pixel checkpoints as they stand.** `imagine()` there samples `z`
  from an essentially uniform 32×32 categorical (prior entropy within 0.003 nat of an
  untrained RSSM driven by the same trained encoder). Any M4 actor built on them would be
  learning against action-driven noise.
- **The feature checkpoints are the only usable world models from M3, and they lose to
  persistence at h=45.** There is **no horizon an M4 actor can inherit from this study.**
  Measured over the six feature cells, the count of cells with `gap_closed > 0` is 5/6 at
  h=1–2, 4/6 at h=3–4, 3/6 at h=5–9, 2/6 at h=10, 1/6 at h=11–16 and **0/6 from h=17 on**. No
  common positive horizon exists; the best case is h ≤ 2 at 5 of 6; and one control cell
  (`random_vit`/s0) is already behind persistence at the first imagined step. An M4 actor has
  to re-measure the usable horizon per checkpoint rather than adopt a range from here.
- **The re-grounded horizon is ~5 steps, not 45.** Re-observing the real frame every 1–5
  steps holds the six feature cells at their floor; every 15 loses about half the
  recoverable error; the 45-step open loop is worse than persistence (Addendum above). An
  M4 design that imagines in short re-grounded segments has a usable horizon this study
  measured; one that imagines 45 steps open-loop does not.
- **The `continue` head is trained and unmeasured.** M4 terminates and discounts imagined
  trajectories with it, and M3 recorded nothing about it — no field, no column, no criterion.
  Given a 0.03% terminal rate and a reward head that is indistinguishable from a constant
  predictor in 9/9 cells, assume it is degenerate until measured. Imagined returns are bounded
  by an unvalidated termination signal.
- **Report the estimator with the number.** Both the rollout probe and the gain probe select
  a ridge per cell from a five-point decade grid, and that selection moved a gain by up to
  0.0633 — past its own CI — and flipped one cell's sign. Any cross-arm comparison must score
  both sides at a common, **pre-registered** decade, and must state that the decade was fixed
  before the scores were seen.
- **Every interval in this codebase is a within-cell evaluation-window interval.** It covers
  no seed, fit, selection, sampling or split variation. For the one arm whose three cells share
  an estimator, between-seed SD is 3.9× the mean within-cell bootstrap SE — **~15× in
  variance.**
- **Fix the record schema before the next study.** Persist `history["loss"]` and
  `history["parts"]` (one line, `study.py:440-443`); stamp the git SHA and the device into
  every record. The first converts two open questions into measurements at zero compute cost;
  the second is why "did one code state produce all nine cells?" had to be answered from file
  mtimes. Note also that the study ran on MPS and a CPU re-derivation does not reproduce the
  shipped gain block — the device belongs in the record for that reason alone.
- **The reward signal cannot supervise anything at this density.** 2 events in 9,949 held-out
  steps; every arm is the constant predictor to within 0.15% of MSE. An M4 actor needs either
  a denser scenario, a shaped signal, or an explicitly exploration-driven objective.
- **`encoders.py` remains the only arm-varying module**, and `KL_FREE_BITS` has no per-arm
  route by design. Any change that gives one arm its own optimisation setting confounds the
  treatment/control contrast and must be paired with a re-run of all nine cells.
- **The pixel arm is 76.6% of any full-study bill** at 0.55–0.60 steps/s. Size accordingly,
  and run the 2,000-step free-bits calibration (0.93–1.01 h per candidate) *before* spending
  9.3 h on a cell — the pixel arm was already dead at step 500–600, 14–18 minutes in.
- **Size M4 from the imagination cost, not the training bill.** The whole evaluation block —
  229 windows × 45 imagined steps plus five probe gathers, reward and both filtering
  diagnostics — cost 86–110 s per cell, 0.6% of the study. Imagination on a frozen feature
  checkpoint is cheap; 37.975 h is the cost of *training* nine world models and is the wrong
  anchor for an actor trained on frozen ones.
