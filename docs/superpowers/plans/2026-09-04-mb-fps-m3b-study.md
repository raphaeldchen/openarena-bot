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
  --steps 20 --seq-len 8 --device cpu --arms random_vit --seeds 0
.venv/bin/python scripts/report_study.py --out runs/m3_smoke
```

Expected: the suite green, one record written, the report printing a table and a
gate verdict of NOT PASSED (one cell of nine).

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

- [ ] `pytest` fully green, with the counts each task states.
- [ ] Angle carries the same degeneracy guards as position (Task 1).
- [ ] Probes are fit under the rollout's own context protocol, verified on a real checkpoint (Task 2).
- [ ] `filtering_report` is called by the eval script, so spec §4 criterion 4 is produced (Task 3).
- [ ] Every mutation table row is caught, using a harness self-checked all three ways.
- [ ] Nine records present, one per (arm, seed).
- [ ] Reward-prediction accuracy reported per arm, with its baseline and degeneracy flag (spec §4.3).
- [ ] The gate verdict is recorded per criterion, whatever it is.
- [ ] `encoders.py` still the only arm-varying module (`grep -rn "cfg.arm\|\.kind" src/ --include=*.py | grep -v encoders.py`).

---

## Task 2 results

*(fill in during execution)*

| quantity | value |
|---|---|
| embedding probe held-out R² under matched protocol | |
| comparison to the pre-fix mismatched figure (~0.29) | |

## Task 3 results

*(fill in during execution)*

| quantity | value |
|---|---|
| `latent_r2` | |
| `embedding_r2` | |
| `latent_beats_embedding` | |

## Task 7 results

*(fill in during execution)*

**Provider / instance / rate:**

| arm | steps/s | seed 0 | seed 1 | seed 2 | mean | unanimous > 0 |
|---|---|---|---|---|---|---|
| `cnn` | | | | | | |
| `frozen_ssl` | | | | | | |
| `random_vit` | | | | | | |

| gate criterion | verdict |
|---|---|
| all nine cells present | |
| reward-prediction accuracy reported per arm (spec §4.3) | |
| beats persistence (unanimous across seeds) | |
| band is usable (no degenerate steps) | |
| filtering probe beats the embedding probe | |
| **GATE** | |
