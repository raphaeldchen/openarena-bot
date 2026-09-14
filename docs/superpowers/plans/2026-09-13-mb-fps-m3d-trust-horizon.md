# MB-FPS M3d — Trust Horizon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One diagnostic over the nine M3c checkpoints that (1) tests, under pre-registered rules, whether the h=45 `beats_persistence` gate rewards a prior that drifts slowly over one that moves correctly, and (2) reports, per arm and per channel, how many open-loop steps imagination can be trusted for — the horizon M4 designs around. The M3 gate and both study verdicts are untouched.

**Architecture:** Pure numpy functions (`trust.py`: crossing steps on the moved basis, a probe-normalised and a probe-free displacement ratio, direction agreement, a fold-wise scale correction with boundary detection, survival curves) over per-window trajectories that a `keep_trajectories` flag makes the ladder's own canonical pass keep (`diagnostics.py`); pure decision logic over already-pooled inputs (`trust_readings.py`: every rule of the spec's section 3, each ambiguous outcome a named non-"supported" status); one script (`trust_horizon.py`) that loads cells exactly as `diagnose_dynamics.py` does, refuses to print a reading until its curves reproduce the stored diagnostic's bitwise, writes one record per cell, pools through `pooling.py` unchanged, and prints both readings. Then one run and one results section.

**Tech Stack:** Python 3.12, PyTorch 2.13 on MPS, numpy; pytest with the repo's mutation-testing harness; `runs/m3_study_v2` as the only data (gitignored).

**Spec:** `docs/superpowers/specs/2026-09-13-mb-fps-m3d-trust-horizon-design.md` (every section number below refers to it).

## Global Constraints

- The M3 gate is not changed: `report_study.py`, `aggregate.py`, `evaluate_gate` and the recorded verdicts of `runs/m3_study` and `runs/m3_study_v2` are untouched (spec §3, §4).
- The nine diagnostic records `runs/m3_study_v2/diagnostic_<arm>_seed<n>.json` are read, never rewritten; the trust pass must reproduce their `curves.reference_position` and `curves.persistence_position` with **max |Δ| == 0.0** and their `windows.total` / `windows.episode` before any reading is printed, or exit **30** (spec §2.3).
- Every quantity is defined on the ladder's 229 windows from 24 validation episodes of `episode_split(seed=0)`, `window_starts(context=5, horizon=45)`; position is `(pos_x, pos_y)`; `p(h)` is the frame the h-th horizon action produced, `start + context + h` (spec §2.1).
- `MIN_MOVE = 5.0` map units; the moved mask is ground truth and serves both channels. Crossings are STRICT (`>`), on the moved basis, `horizon + 1` for never, NaN for never-moved, and are **never seed-averaged** (spec §2.2).
- `e(h)` is the encoder's embedding of frame `start + context + h` — `handle.embeddings[0, context + h − 1]` — the embedding head's training target; the two imagination-vs-truth series are new (spec §2.1).
- Scale correction: α on `np.linspace(0, 2, 201)`, fold A = even `window_episode` labels, fold B = odd, fit by the median over moved rows, scored on the other fold, α on a grid endpoint is a **boundary**; fewer than two labels → `folds_available = False` (spec §2.2).
- Pooling: `pool_arm` / `paired_contrast` (episode-clustered, z against t(clusters − 1)) for means and contrasts; `pool_ratio` (ratio of medians, 2000 bootstrap draws, seed 0) for ratios; **FAMILY = 8**; **q = 0.75 pre-registered**, 0.5 and 0.9 reported; `H*_min` over the **probe-free** channel; the ladder's `probe_is_measurable` rule for inclusion; a sensitivity line at probe R² < 0.1 that changes no verdict (spec §3.1, §3.3).
- Reading 1's five rules and their status precedence are those of spec §3.2 verbatim; an ambiguous outcome is NOT_TESTABLE / UNRESOLVED_PROBE / UNRESOLVED_ALPHA / undecided, never SUPPORTED.
- Exit statuses: 0, 11, 12, 14 with `diagnose_dynamics.py`'s meanings, **30** `EXIT_SELF_CHECK_FAILED`; distinct from the other tools' 1–23 and argparse's 2, pinned by the distinctness test.
- Tests never depend on `runs/`; the script tests build their own cell at context 2 / horizon 3 on the six-episode fixture (one validation episode: clustered SEs NaN, folds unavailable — both disclosed, both exit 0). Every guard is mutation-tested with a harness self-checked three ways (PYTHONPATH at an export, a known-fatal mutation first, `__pycache__` cleared with `PYTHONDONTWRITEBYTECODE=1`).
- Nothing under `runs/` or `data/` is committed. Commit messages end with a blank line and `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

**Measured facts this plan is built on** (2026-09-13, from the M3c records — do not re-derive): the mean open-loop error first exceeds persistence at step 10·5·2 (`frozen_ssl` s0·s1·s2), 9·2·6 (`pixel_ae`), 1·11·17 (`random_vit`); true 2-D displacement median 4.0 / 21.5 / 56.5 / 115.6 map units at h = 1 / 5 / 15 / 45; probe floor 120–250 units; probe selection R² 0.018 (`pixel_ae`/s1) to 0.383, every other cell ≥ 0.249; persistence-to-floor band at h=45 20–70 units. `cluster_threshold(8, 24)` = 3.01 (the ladder's `cluster_threshold(24, 24)` is 3.47). The full suite at the reference commit b382194 is 1278 passed, 0 warnings.

---

> **Line numbers** in every Files block are as of commit `b382194`. Each task shifts them for the next; every edit is also named by the symbol or test it touches — locate by that, treat the ranges as hints. **Test counts** stated as absolutes are stale by construction; the binding expectation is the delta each step names on top of what the previous task's last step measured — record the measured number in the task report. Tasks are strictly sequential: Task N assumes Tasks 1..N−1 have landed and been committed.

---

## Interface contract

### File map (one responsibility each)

| file | responsibility | task |
|---|---|---|
| `src/mbfps/eval/trust.py` | pure numpy functions: crossing, margin, survival/horizon (Task 1); decomposition, embedding ratio, scale correction (Task 2) | 1, 2 |
| `src/mbfps/eval/diagnostics.py` | `keep_trajectories` flag on `_diagnose`, nine optional `_Pass` fields, public `reference_trajectories(...)` | 3 |
| `src/mbfps/eval/trust_readings.py` | pure decision logic and formatting for Reading 1 and Reading 2 over already-pooled inputs | 5 |
| `scripts/trust_horizon.py` | CLI: load cells as diagnose_dynamics does, checks 11/12/14/30 in order, reference pass, per-cell `trust_<arm>_seed<n>.json` (Task 4); pooling glue, readings, `trust.txt` (Task 6) | 4, 6 |
| `tests/eval/test_trust.py` | Task 1 + Task 2 pure-function tests (Task 2 APPENDS to the file Task 1 creates; it does not recreate the header) | 1, 2 |
| `tests/eval/test_diagnostics.py` | Task 3 appends the keep_trajectories tests | 3 |
| `tests/eval/test_trust_readings.py` | Task 5 | 5 |
| `tests/eval/test_trust_horizon_script.py` | Task 4 creates (loading, checks, record); Task 6 APPENDS (pooling glue, readings, trust.txt, exit distinctness) | 4, 6 |
| `tests/eval/test_diagnose_dynamics_script.py` | Task 6 adds `trust_horizon` to the exit-status distinctness test | 6 |
| `docs/superpowers/plans/2026-09-13-mb-fps-m3d-trust-horizon.md` | Task 7 fills `## Task 7 results` | 7 |

### `src/mbfps/eval/trust.py` (Tasks 1 and 2)

```python
MIN_MOVE: float = 5.0                       # map units; the moved-mask threshold (spec 2.1)
ALPHAS: np.ndarray = np.linspace(0.0, 2.0, 201)

def moved_mask(p_true: np.ndarray, p_true0: np.ndarray, min_move: float = MIN_MOVE) -> np.ndarray
    # p_true (n, H, 2), p_true0 (n, 2) -> (n, H) bool: |p_true[:, h] - p_true0| >= min_move

def crossing_step(model_err: np.ndarray, persist_err: np.ndarray, moved: np.ndarray) -> np.ndarray
    # (n, H), (n, H), (n, H) bool -> (n,) float64. Per row: h0 = first h (1-based) with moved;
    # result = first h >= h0 with model_err > persist_err (STRICT); H + 1 if never; NaN if no moved step.

def persistence_margin(model_err: np.ndarray, persist_err: np.ndarray) -> np.ndarray
    # (n, H) -> (n, H): persist_err - model_err  (positive = beats persistence)

def survival(crossings: np.ndarray, horizon: int) -> np.ndarray
    # (n,) with NaN allowed -> (H + 1,) float: S[h] = fraction of FINITE crossings with crossing > h, h = 0..H.
    # S[0] == 1.0 whenever any finite crossing exists; all-NaN input -> all NaN.

def trust_horizon(surv: np.ndarray, q: float) -> int
    # largest h with surv[h] >= q; 0 if surv[0] < q (cannot happen for q <= 1 when finite draws exist); -1 if surv is all NaN.

@dataclass(frozen=True)
class Decomposition:
    ratio_probe: np.ndarray        # (n, H) |d_hat| / |d_hat_real|; NaN where not moved or |d_hat_real| == 0
    ratio_raw: np.ndarray          # (n, H) |d_hat| / |d|;          NaN where not moved
    cosine: np.ndarray             # (n, H) cos(d_hat, d);           NaN where not moved or |d_hat| == 0
    moved: np.ndarray              # (n, H) bool (the mask passed in)
    zero_displacement: np.ndarray  # (n, H) bool: moved and |d_hat| == 0

def displacement_decomposition(p_hat, p_true, p_hat0, p_true0, p_hat_real, moved) -> Decomposition
    # p_hat (n, H, 2), p_true (n, H, 2), p_hat0 (n, 2), p_true0 (n, 2), p_hat_real (n, H, 2), moved (n, H) bool
    # d_hat = p_hat - p_hat0[:, None]; d = p_true - p_true0[:, None]; d_hat_real = p_hat_real - p_hat0[:, None]

def embedding_ratio(e_hat_disp: np.ndarray, e_true_disp: np.ndarray, moved: np.ndarray) -> np.ndarray
    # (n, H) norms ||e_hat(h) - e_hat(0)||, (n, H) norms ||e(h) - e(0)||, moved -> (n, H): e_hat_disp / e_true_disp; NaN where not moved or e_true_disp == 0

@dataclass(frozen=True)
class ScaleCorrection:
    alpha_a: np.ndarray      # (H,) alpha fit on fold A (even episode labels), NaN if unavailable
    alpha_b: np.ndarray      # (H,) alpha fit on fold B (odd labels)
    score_a: np.ndarray      # (H,) median over fold B rows of |p_hat0 + alpha_a * d_hat - p_true|  (fit A, scored on B)
    score_b: np.ndarray      # (H,) median over fold A rows of the same with alpha_b               (fit B, scored on A)
    held_out: np.ndarray     # (n, H) every row scored with the alpha fit on the OTHER fold; NaN where not moved
    boundary: np.ndarray     # (H,) bool: alpha_a or alpha_b equals alphas[0] or alphas[-1]
    folds_available: bool    # False when fewer than 2 distinct episode labels; then every array is NaN / False

def scale_corrected_error(p_hat, p_true, p_hat0, moved, episode, alphas=ALPHAS) -> ScaleCorrection
    # episode (n,) int labels; fold A = rows with episode % 2 == 0, fold B = episode % 2 == 1.
    # Fit objective per h and fold: argmin over alphas of median over the fold's MOVED rows of |p_hat0 + alpha*d_hat - p_true[:, h]|
    # (2-D Euclidean norm). Ties in the argmin -> the smallest alpha.
```

### `src/mbfps/eval/diagnostics.py` (Task 3)

```python
def _diagnose(..., keep_trajectories: bool = False) -> _Pass   # existing signature plus this keyword-only flag

# nine new Optional fields on _Pass, default None, populated ONLY when keep_trajectories=True, reference arm only:
positions: np.ndarray | None                       # (n_windows, horizon, 2)   p_hat(h)
positions_at_context: np.ndarray | None            # (n_windows, 2)            p_hat(0)   (== rollout.py's pers_pred anchor)
positions_real: np.ndarray | None                  # (n_windows, horizon, 2)   p_hat_real(h): probe of floor_embeddings
true_positions: np.ndarray | None                  # (n_windows, horizon, 2)   p(h) = privileged (pos_x, pos_y) at frame start+context+h
true_at_context: np.ndarray | None                 # (n_windows, 2)            p(0) at frame start+context
embedding_distance_to_truth: np.ndarray | None     # (n_windows, horizon)      ||e_hat(h) - e(h)||,  e(h) = handle.embeddings[0, context + h - 1]
embedding_persistence_distance: np.ndarray | None  # (n_windows, horizon)      ||e_hat(0) - e(h)||,  e_hat(0) = last_context_embedding
embedding_displacement: np.ndarray | None          # (n_windows, horizon)      ||e_hat(h) - e_hat(0)||
true_embedding_displacement: np.ndarray | None     # (n_windows, horizon)      ||e(h) - e(0)||,  e(0) = handle.embeddings[0, context - 1]

@dataclass(frozen=True)
class Trajectories:
    # the nine arrays above, same names and shapes, all present (not Optional), plus:
    window_episode: np.ndarray     # (n_windows,) int -- the existing _Pass.window_episode
    windows_total: int
    reference_position: np.ndarray    # (horizon,) mean over windows of |p_hat(h) - p(h)|   -- for the self-check
    persistence_position: np.ndarray  # (horizon,) mean over windows of |p_hat(0) - p(h)|   -- for the self-check

def reference_trajectories(model, val_paths, embedding_probe_weights: dict, *, context: int, horizon: int,
                           seed: int, device, feature_backbone) -> Trajectories
    # calls _diagnose(model, val_paths, embedding_probe_weights, arms={}, context=context, horizon=horizon, seed=seed,
    #                 device=device, feature_backbone=feature_backbone, noise_reference=True, keep_trajectories=True)
    # and repackages. Decorated @torch.no_grad() like the other public entry points.
```
The means in `Trajectories` are `np.stack(rows).mean(axis=0)` over per-window rows, the same reduction `_diagnose` uses for its curves, so the self-check's max|delta| == 0.0 is achievable.

### `scripts/trust_horizon.py` (Task 4, extended by Task 6)

```python
EXIT_OK = 0
EXIT_NO_CHECKPOINTS = 11        # reused with diagnose_dynamics.py's meaning
EXIT_SPLIT_MISMATCH = 12        # reused
EXIT_RECORD_MISMATCH = 14       # reused: evaluate_rollout no longer reproduces the study record's curves.rssm_position
EXIT_SELF_CHECK_FAILED = 30     # new: the trust pass's curves / windows do not equal the diagnostic's

def _parser() -> argparse.ArgumentParser
    # --out (default "runs/m3_study_v2"), --device (default "mps"), --data (default "data/my_way_home"),
    # --context / --horizon (default None -> read from each cell's diagnostic_<arm>_seed<n>.json; a value that
    #   differs from the diagnostic's recorded context/horizon is refused up front under EXIT_RECORD_MISMATCH), --arms (choices=ARMS, default all),
    #   --seeds (default 0 1 2)
def load_cell(out_dir: Path, arm: str, seed: int) -> Cell          # checkpoint + study record + diagnostic record, or raises the typed error mapped to 11
def self_check(traj: Trajectories, diagnostic: dict) -> SelfCheck  # max|delta| on the two curves, windows.total, windows.episode; .ok
def trust_record(arm, seed, traj: Trajectories, diagnostic: dict, *, context, horizon, device) -> dict
    # per h: crossing steps (probe, free) per window; margin; ratio_probe; ratio_raw; ratio_free; cosine; held_out; alpha_a; alpha_b; boundary;
    # excluded counts (not-moved, zero-displacement, never-moved); self_check deltas; probe R^2 (from the diagnostic); git_sha of HEAD; device; torch version
def write_trust_record(out_dir: Path, record: dict) -> Path       # runs/<out>/trust_<arm>_seed<n>.json
def main(argv: list[str] | None = None) -> int
```
Record top-level keys (exact): `arm, seed, context, horizon, split_seed, device, torch_version, git_sha, episodes {val: [...]}, windows {total, episode}, probe {selection_r2, measurable}, self_check {reference_position_max_delta, persistence_position_max_delta, windows_total_match, windows_episode_match, ok}, crossing {probe: [n], free: [n]}, margin [n][H], ratio_probe [n][H], ratio_raw [n][H], ratio_free [n][H], cosine [n][H], scale {alpha_a [H], alpha_b [H], score_a [H], score_b [H], held_out [n][H], boundary [H], folds_available}, counts {not_moved [H], zero_displacement [H], never_moved}, displacement {probe_hat [n][H], probe_real [n][H], free_hat [n][H], free_true [n][H]}, nonfinite {}` (the four displacement norms the ratios were built from: |d_hat|, |d_hat_real|, ||e_hat(h) - e_hat(0)||, ||e(h) - e(0)||).

### `src/mbfps/eval/trust_readings.py` (Task 5) — pure over pooled inputs

```python
ARMS_ORDER = ("pixel_ae", "frozen_ssl", "random_vit")
TREATMENT = "frozen_ssl"; CONTROL = "random_vit"
FAMILY = 8                                           # spec 3.1: the eight clustered contrasts of Reading 1
Q_PREREGISTERED = 0.75; Q_REPORTED = (0.5, 0.75, 0.9)

@dataclass(frozen=True)
class Contrast:          # what paired_contrast yields, reduced to what the rules read
    estimate: float; se: float; z: float; n_windows: int
@dataclass(frozen=True)
class Ratio:             # what pool_ratio yields, reduced
    estimate: float; low: float; high: float

@dataclass(frozen=True)
class ReadingOneInputs:
    delta_contrast: dict[tuple[str, str], Contrast]   # keys (a, b) for the three unordered pairs, estimate = a - b
    ratio_probe: dict[str, Ratio]; ratio_free: dict[str, Ratio]   # at h = 45, per arm
    cosine_contrast: Contrast                          # TREATMENT - CONTROL, cos(45)
    corrected_contrast_a: Contrast; corrected_contrast_b: Contrast   # TREATMENT - CONTROL held-out c(45), folds A and B
    alpha_boundary: dict[str, bool]                    # per arm: boundary at h = 45 on either fold
    crossing_contrast_probe: Contrast; crossing_contrast_free: Contrast   # TREATMENT - CONTROL paired per draw
    per_seed: dict[int, "ReadingOneInputs"] | None     # the same inputs computed within each seed; None at the leaf

class Status(str, Enum): SUPPORTED, NOT_SUPPORTED, NOT_TESTABLE, UNRESOLVED_PROBE, UNRESOLVED_ALPHA

@dataclass(frozen=True)
class ConditionResult: name: str; holds: bool | None; detail: str    # holds None = undecided / undecidable / unreadable
@dataclass(frozen=True)
class ReadingOne: status: Status; conditions: tuple[ConditionResult, ...]; best_delta_arm: str | None; least_moving_arm: str | None; per_seed_agreement: dict[str, int]; reason: str

def best_delta_arm(delta_contrast, z_fam) -> str | None     # the arm whose contrast against EACH other arm has z > z_fam (sign oriented a - b); None if no arm
def least_moving_arm(ratio_probe, ratio_free) -> str | None # smallest ratio_probe estimate; None if the argmin under ratio_free differs
def condition_i(inputs, z_fam) -> ConditionResult
def condition_ii(inputs, z_fam) -> ConditionResult          # cosine_contrast.z > z_fam
def condition_iii(inputs, z_fam) -> ConditionResult         # both folds z < -z_fam -> True; either fold z > z_fam -> False; else None; boundary on TREATMENT or CONTROL -> None with detail "unreadable"
def probe_control(inputs, z_fam) -> bool                    # True (= fires) iff both crossing contrasts |z| > z_fam with opposite signs
def reading_one(inputs, z_fam, h: int = 45) -> ReadingOne      # h names the step in every detail line; Task 6 passes the run's horizon
    # order: probe_control fires -> UNRESOLVED_PROBE; condition_iii unreadable -> UNRESOLVED_ALPHA; condition_i undecidable (no best-delta arm) -> NOT_TESTABLE;
    # least_moving_arm None (channels disagree) -> UNRESOLVED_PROBE; all three hold pooled AND each holds in >= 2 of the 3 per_seed entries -> SUPPORTED; else NOT_SUPPORTED.
    # per_seed None at the top level -> ValueError (Task 6 always passes the per-seed leaves).

@dataclass(frozen=True)
class ReadingTwo:
    survival: dict[tuple[str, str], np.ndarray]     # (arm, channel) -> S(h), channel in ("probe", "free")
    horizons: dict[tuple[str, str, float], int]     # (arm, channel, q) -> H*_q
    h_min: int                                      # min over arms of horizons[(arm, "free", 0.75)]
def reading_two(survival: dict[tuple[str, str], np.ndarray], qs=Q_REPORTED) -> ReadingTwo
def format_reading_one(r: ReadingOne, inputs: ReadingOneInputs, z_fam: float, h: int = 45) -> str   # h is printed in the header and every detail line
def format_reading_two(r: ReadingTwo) -> str
```
`z_fam` is `pooling.cluster_threshold(FAMILY, clusters)` computed by the caller (Task 6) from the actual cluster count.

### Pooling glue (Task 6, in `scripts/trust_horizon.py`)

```python
def cell_series(arm, seed, name, values: np.ndarray, changed: np.ndarray, record: dict) -> pooling.CellSeries
    # values (n,), changed (n,) bool; rung=name, channel="position"; embedding/noise None; identity fields from the record
def pooled_inputs(records: dict[tuple[str, int], dict], *, h: int = 45) -> tuple[ReadingOneInputs, int]   # (inputs, clusters)
def survival_by_arm(records) -> dict[tuple[str, str], np.ndarray]
def write_readings(out_dir: Path, text: str) -> Path   # runs/<out>/trust.txt
```
Pooled means/contrasts go through `pooling.pool_arm` / `pooling.paired_contrast` on the per-window seed-mean series (crossing contrasts on per-draw series with seeds stacked); ratios through `pooling.pool_ratio` (bootstrap=2000, seed=0) on the numerator/denominator series of moved windows, seeds stacked.

### Repo facts every writer needs

- Python: `.venv/bin/python`; pytest: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest <path> -q -p no:cacheprovider`. Full suite 1278 passed, 0 warnings, ~6 min.
- Scripts are loaded by path in tests (`importlib.util.spec_from_file_location`), see `tests/eval/test_diagnose_dynamics_script.py` and `tests/data/test_cache_features_script.py`.
- The tiny cell used by script tests: `run_job(job, small_buffer, tmp_path, steps=5, seq_len=4, context=2, horizon=3, device="cpu")` (tests/eval/test_aggregate.py:811, test_run_study.py:104); `small_buffer` (tests/eval/conftest.py) has six 40-step episodes -> ONE validation episode -> `cluster_standard_error` NaN below two clusters and `scale_corrected_error.folds_available == False`. `window_starts(40, 5, 45)` is EMPTY, so the fixture must run at context 2 / horizon 3.
- `diagnose_dynamics.main` writes `diagnostic_<arm>_seed<n>.json` for a cell; the script tests produce the diagnostic that way before calling `trust_horizon.main`.
- `pooling.CellSeries` fields: arm, seed, rung, channel, delta (n,), changed (n,) bool, episode (n,) int, embedding (n,)|None, noise (n,)|None, windows_total, val (tuple[str,...]), horizon, context, device, torch_version. `pool_arm(cells)`, `paired_contrast(treatment_cells, control_cells)`, `pool_ratio(cells, *, bootstrap=2000, seed=0)` (reads `.embedding` as numerator and `.noise` as denominator), `cluster_threshold(family, clusters)`.
- `diagnose_dynamics.py` exit statuses 11..17; `probe_is_measurable(cell, widest_se)` at scripts/diagnose_dynamics.py:932 uses band = persistence - floor at the final step > 0.
- Existing exit statuses across tools: 1, 3, 4, 5, 6, 23 (run_study), 7-10 (report_study), 10 (spike), 11-17 (diagnose), 18-22 (pool). 30 is unused. Distinctness test: `test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own` in tests/eval/test_diagnose_dynamics_script.py.
- Every guard is mutation-tested; harnesses self-check three ways: PYTHONPATH at an export of the tree, a known-fatal mutation first, `__pycache__` cleared with PYTHONDONTWRITEBYTECODE=1. Every plan task ends with a mutation table (mutation -> the test that catches it) and a commit step.
- Commit messages: imperative subject in the repo's style ("feat: ...", "test: ...", "docs: ..."), a body when the why is not obvious, ending with a blank line and exactly `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Nothing under `runs/` or `data/` is committed. Tests never depend on `runs/m3_study_v2` (they build their own cell).

---

### Task 1: trust.py, part 1: moved_mask, crossing_step, persistence_margin, survival, trust_horizon

The M3 gate reads two *mean* curves, and a mean curve cannot say on how many windows the model was still ahead of "you did not move" at step h. This task lands the per-window half of `src/mbfps/eval/trust.py`: the ground-truth `moved` mask that gates every other quantity, the per-row crossing step `h×` (first strict loss to persistence at or after the first moved step; `H + 1` for never; NaN for never moved), the per-window persistence margin `Δ`, and the survival curve `S(h)` with its integer summary `H*_q`. Pure numpy, no torch, no files, no randomness; every function is pinned by a synthetic prior whose answer is typed by hand in the test. `ALPHAS`, the two dataclasses, `displacement_decomposition`, `embedding_ratio` and `scale_corrected_error` are **Task 2's**, which appends them to the same module and the same test file; this task must not define them.

**Files:**
- Create: `src/mbfps/eval/trust.py` -- module docstring, `MIN_MOVE`, `moved_mask`, `crossing_step`, `persistence_margin`, `survival`, `trust_horizon`, in that order (Task 2 appends `ALPHAS` and its functions after `trust_horizon`)
- Create: `tests/eval/test_trust.py` -- module docstring, a `# --- moved_mask` block, a `# --- crossing_step and persistence_margin` block, a `# --- survival and trust_horizon` block (Task 2 appends after the last test; it does not repeat the header)

> `src/mbfps/eval/__init__.py` is empty at b382194 and stays empty: the package has no export list to register a new module in. `tests/eval/__init__.py` exists, so the new test module is collected by `testpaths = ["tests"]` with nothing added.

**Interfaces:**
- Consumes: nothing from an earlier task (this is the first). `numpy` only.
- Produces, for Tasks 2, 4, 5 and 6 (signatures verbatim from the contract):
  - `MIN_MOVE: float = 5.0` -- map units, the moved-mask threshold (spec 2.1).
  - `moved_mask(p_true: np.ndarray, p_true0: np.ndarray, min_move: float = MIN_MOVE) -> np.ndarray` -- `p_true (n, H, 2)`, `p_true0 (n, 2)` -> `(n, H)` bool: `|p_true[:, h] - p_true0| >= min_move`, 2-D Euclidean.
  - `crossing_step(model_err: np.ndarray, persist_err: np.ndarray, moved: np.ndarray) -> np.ndarray` -- `(n, H)`, `(n, H)`, `(n, H)` bool -> `(n,)` float64. Per row: `h0` = first h (1-based) with `moved`; result = first `h >= h0` with `model_err > persist_err` (STRICT); `H + 1` if never; NaN if no moved step.
  - `persistence_margin(model_err: np.ndarray, persist_err: np.ndarray) -> np.ndarray` -- `(n, H)` -> `(n, H)`: `persist_err - model_err` (positive = beats persistence).
  - `survival(crossings: np.ndarray, horizon: int) -> np.ndarray` -- `(n,)` with NaN allowed -> `(H + 1,)` float: `S[h]` = fraction of FINITE crossings with `crossing > h`, h = 0..H. `S[0] == 1.0` whenever any finite crossing exists; all-NaN input -> all NaN.
  - `trust_horizon(surv: np.ndarray, q: float) -> int` -- largest h with `surv[h] >= q`; 0 if no h qualifies; -1 if `surv` is all NaN.
  - Task 4's `trust_record` calls all five; Task 6's `survival_by_arm` calls `survival` on each arm's stacked per-draw crossings and Task 5's `reading_two` calls `trust_horizon` on the result.

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_trust.py` with exactly this content. `HORIZON = 6` is deliberately test-local (the shipped horizon is 45) so every survival curve can be checked by eye; `NEVER = 7.0` is what `crossing_step` returns for "never crossed" at that horizon.

```python
"""The trust horizon's pure functions, pinned by synthetic priors with known answers.

`mbfps.eval.trust` has no model, no files and no randomness: every input here
is a small array typed by hand and every expectation is typed beside it,
never read back from the function under test. The priors are the ones the
M3d design names, and each is a mutation the others cannot catch:

  * THE PERSISTENCE CLONE (`model_err == persist_err` everywhere) never
    crosses -- `h x = horizon + 1` on every moved row -- and its margin is 0.
  * THE PERFECT PREDICTOR (`model_err == 0`) never crosses either; its
    crossing is IDENTICAL to the clone's and only the margin (persistence's
    own error, positive) tells them apart. A test that read the crossing
    alone could not see a model that sits still.
  * A PRIOR THAT CROSSES AT A KNOWN STEP on some rows and never on others,
    with one row that loses BEFORE it moves (ignored: the search starts at
    the first moved step) and one that returns near its start after moving
    (still crosses: the mask gates the start of the search, not each step).
  * ROWS THAT NEVER MOVE are NaN, and NaN rows leave the survival curve's
    denominator -- counted, not scored as failures.
  * TIES NEVER CROSS. `>` not `>=`: a clone is not worse than persistence.
  * AN EVEN-COUNT CROSSING MULTISET, whose median is a half-step, gives an
    integer `H*_q` because `H*_q` reads the survival curve, not a quantile.

Task 2 appends the decomposition, embedding-ratio and scale-correction tests
below this file's last test; the header is not repeated.
"""

import numpy as np
import pytest

from mbfps.eval.trust import (
    MIN_MOVE,
    crossing_step,
    moved_mask,
    persistence_margin,
    survival,
    trust_horizon,
)

# Deliberately TEST-LOCAL: the shipped horizon is 45, but every answer below
# is typed by hand at a horizon small enough to check by eye.
HORIZON = 6
NEVER = float(HORIZON + 1)
T, F = True, False


# --- moved_mask ---------------------------------------------------------------


def test_min_move_is_five_map_units():
    assert MIN_MOVE == 5.0


def test_moved_mask_is_the_euclidean_displacement_at_or_above_the_threshold():
    anchor = np.array([[10.0, 20.0]])
    positions = np.array([[
        [13.0, 24.0],   # (3, 4): norm 5.0 -> moved, on the threshold (>=)
        [12.5, 22.5],   # (2.5, 2.5): norm 3.54 -> not moved (L1 would say 5)
        [10.0, 24.0],   # (0, 4): norm 4 -> not moved
        [10.0, 15.0],   # (0, -5): norm 5 -> moved; sign is irrelevant
        [10.0, 20.0],   # did not move at all
    ]])
    mask = moved_mask(positions, anchor)
    assert mask.dtype == bool
    assert mask.shape == (1, 5)
    assert mask.tolist() == [[T, F, F, T, F]]


def test_moved_mask_honours_min_move():
    anchor = np.array([[10.0, 20.0]])
    positions = np.array([[[13.0, 24.0], [12.5, 22.5], [10.0, 24.0], [10.0, 15.0], [10.0, 20.0]]])
    assert moved_mask(positions, anchor, min_move=4.0).tolist() == [[T, F, T, T, F]]
    assert moved_mask(positions, anchor, min_move=5.5).tolist() == [[F, F, F, F, F]]


def test_moved_mask_measures_each_row_against_its_own_anchor():
    anchors = np.array([[0.0, 0.0], [100.0, 100.0]])
    positions = np.array([
        [[6.0, 0.0], [0.0, 0.0]],        # row 0: moved 6, then back home
        [[6.0, 0.0], [100.0, 106.0]],    # row 1: (6, 0) is 137 units from ITS anchor
    ])
    # A mask that read every row against anchors[0] would say [[T, F], [T, T]].
    assert moved_mask(positions, anchors).tolist() == [[T, F], [T, T]]
    positions[1, 0] = [100.0, 100.0]
    assert moved_mask(positions, anchors).tolist() == [[T, F], [F, T]]


def test_moved_mask_refuses_the_wrong_shapes():
    with pytest.raises(ValueError, match=r"p_true must be \(n, H, 2\)"):
        moved_mask(np.zeros((2, 6)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match=r"p_true must be \(n, H, 2\)"):
        moved_mask(np.zeros((2, 6, 3)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match=r"p_true0 must be \(n, 2\) = \(2, 2\)"):
        moved_mask(np.zeros((2, 6, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match=r"p_true0 must be \(n, 2\)"):
        moved_mask(np.zeros((2, 6, 2)), np.zeros((2,)))


# --- crossing_step and persistence_margin ------------------------------------


def test_a_persistence_clone_never_crosses_and_its_margin_is_zero():
    persist = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 3)
    model = persist.copy()
    moved = np.ones((3, HORIZON), dtype=bool)
    crossing = crossing_step(model, persist, moved)
    assert crossing.dtype == np.float64
    assert crossing.tolist() == [NEVER, NEVER, NEVER]
    assert persistence_margin(model, persist).tolist() == [[0.0] * HORIZON] * 3


def test_a_perfect_predictor_never_crosses_and_its_margin_is_persistences_error():
    persist = np.array([[3.0, 8.0, 12.0, 20.0, 25.0, 31.0]])
    model = np.zeros((1, HORIZON))
    moved = np.ones((1, HORIZON), dtype=bool)
    assert crossing_step(model, persist, moved).tolist() == [NEVER]
    assert persistence_margin(model, persist).tolist() == [[3.0, 8.0, 12.0, 20.0, 25.0, 31.0]]


def test_the_clone_and_the_perfect_predictor_share_a_crossing_and_differ_in_margin():
    # The crossing cannot tell "sits still" from "predicts exactly"; the margin
    # can, and Task 2's ratio does too. Pinned so a reader of `h x` alone is
    # never mistaken for a reader of trust.
    persist = np.array([[3.0, 8.0, 12.0, 20.0, 25.0, 31.0]])
    moved = np.ones((1, HORIZON), dtype=bool)
    clone, perfect = persist.copy(), np.zeros((1, HORIZON))
    assert crossing_step(clone, persist, moved).tolist() == crossing_step(perfect, persist, moved).tolist() == [NEVER]
    assert (persistence_margin(clone, persist) == 0.0).all()
    assert (persistence_margin(perfect, persist) > 0.0).all()


def test_crossing_is_the_first_strict_exceedance_at_or_after_the_first_moved_step():
    persist = np.full((4, HORIZON), 10.0)
    model = np.array([
        [5.0, 5.0, 11.0, 11.0, 11.0, 11.0],   # moved from step 1; loses from step 3 -> 3
        [12.0, 12.0, 5.0, 5.0, 12.0, 5.0],    # loses at 1, 2 BEFORE moving; moves at 3; loses at 5 -> 5
        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],       # never loses -> never (7)
        [11.0, 9.0, 9.0, 9.0, 9.0, 9.0],      # loses at the first moved step itself -> 1
    ])
    moved = np.array([
        [T, T, T, T, T, T],
        [F, F, T, T, T, T],
        [F, T, T, T, T, T],
        [T, T, T, T, T, T],
    ])
    assert crossing_step(model, persist, moved).tolist() == [3.0, 5.0, NEVER, 1.0]


def test_a_window_that_returns_near_its_start_after_moving_still_crosses():
    # The mask gates the START of the search (h0), not each step after it:
    # a loss at a step where the agent has come back within MIN_MOVE of its
    # start still counts once the window has moved at all.
    persist = np.full((1, HORIZON), 10.0)
    model = np.array([[5.0, 5.0, 11.0, 5.0, 5.0, 5.0]])
    moved = np.array([[F, T, F, T, T, T]])   # moved at 2, back home at 3, gone again
    assert crossing_step(model, persist, moved).tolist() == [3.0]


def test_ties_never_cross():
    persist = np.full((3, HORIZON), 10.0)
    model = np.array([
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],   # equal everywhere -> never
        [10.0, 10.0, 10.5, 10.0, 10.0, 10.0],   # strictly worse at 3 only -> 3
        [9.9, 9.9, 9.9, 9.9, 9.9, 9.9],         # always slightly better -> never
    ])
    moved = np.ones((3, HORIZON), dtype=bool)
    assert crossing_step(model, persist, moved).tolist() == [NEVER, 3.0, NEVER]


def test_a_row_that_never_moves_is_nan_whatever_its_errors_say():
    persist = np.full((3, HORIZON), 10.0)
    model = np.full((3, HORIZON), 11.0)   # loses everywhere on every row
    moved = np.array([
        [F, F, F, F, F, F],   # never moved -> NaN, not 1
        [T, T, T, T, T, T],   # -> 1
        [F, F, F, F, F, T],   # moves only at the last step -> 6
    ])
    crossing = crossing_step(model, persist, moved)
    assert np.isnan(crossing[0])
    assert crossing[1:].tolist() == [1.0, 6.0]


def test_crossing_step_refuses_mismatched_shapes_and_nonfinite_errors():
    ok = np.ones((2, HORIZON))
    moved = np.ones((2, HORIZON), dtype=bool)
    with pytest.raises(ValueError, match=r"model_err must be \(n, H\)"):
        crossing_step(np.ones(HORIZON), ok, moved)
    with pytest.raises(ValueError, match="share one \\(n, H\\) shape"):
        crossing_step(ok, np.ones((2, HORIZON - 1)), moved)
    with pytest.raises(ValueError, match="share one \\(n, H\\) shape"):
        crossing_step(ok, ok, np.ones((3, HORIZON), dtype=bool))
    with pytest.raises(ValueError, match="must be finite"):
        crossing_step(np.where(np.eye(2, HORIZON) > 0, np.nan, 1.0), ok, moved)
    with pytest.raises(ValueError, match="must be finite"):
        crossing_step(ok, np.full((2, HORIZON), np.inf), moved)


def test_persistence_margin_is_persistence_minus_model():
    model = np.array([[1.0, 4.0], [10.0, 0.0]])
    persist = np.array([[3.0, 2.0], [10.0, 5.0]])
    assert persistence_margin(model, persist).tolist() == [[2.0, -2.0], [0.0, 5.0]]
    with pytest.raises(ValueError, match="must share a shape"):
        persistence_margin(model, persist[:1])


# --- survival and trust_horizon ----------------------------------------------


def test_the_persistence_clones_survival_is_one_everywhere_and_its_horizon_is_the_horizon():
    crossings = np.full(3, NEVER)   # what crossing_step gives the clone
    curve = survival(crossings, HORIZON)
    assert curve.shape == (HORIZON + 1,)
    assert curve.tolist() == [1.0] * (HORIZON + 1)
    for q in (0.5, 0.75, 0.9, 1.0):
        assert trust_horizon(curve, q) == HORIZON


def test_survival_is_the_fraction_of_finite_crossings_strictly_beyond_h():
    crossings = np.array([2.0, 4.0, 4.0, NEVER])
    # h:            0    1    2     3     4     5     6
    expected = [1.0, 1.0, 0.75, 0.75, 0.25, 0.25, 0.25]
    assert survival(crossings, HORIZON).tolist() == expected


def test_survival_excludes_never_moved_rows_from_the_denominator():
    crossings = np.array([2.0, np.nan, np.nan, 4.0])
    # Over the two finite rows {2, 4}; a denominator of four would halve it.
    assert survival(crossings, 4).tolist() == [1.0, 1.0, 0.5, 0.5, 0.0]


@pytest.mark.parametrize(
    "crossings",
    [[1.0], [1.0, np.nan], [NEVER], [1.0, 3.0, NEVER, np.nan, 2.0]],
    ids=["all-cross-at-1", "one-nan", "never", "mixed"],
)
def test_survival_starts_at_one_whenever_any_crossing_is_finite(crossings):
    assert survival(np.array(crossings), HORIZON)[0] == 1.0


def test_survival_is_all_nan_without_a_finite_crossing_and_the_horizon_is_minus_one():
    curve = survival(np.array([np.nan, np.nan]), HORIZON)
    assert curve.shape == (HORIZON + 1,)
    assert np.isnan(curve).all()
    for q in (0.5, 0.75, 0.9):
        assert trust_horizon(curve, q) == -1


def test_survival_refuses_crossings_outside_one_to_horizon_plus_one():
    with pytest.raises(ValueError, match="crossings must lie in 1..7"):
        survival(np.array([0.0, 3.0]), HORIZON)          # a 0-based step
    with pytest.raises(ValueError, match="crossings must lie in 1..7"):
        survival(np.array([3.0, NEVER + 1.0]), HORIZON)  # a longer horizon's crossing
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        survival(np.array([1.0]), 0)
    with pytest.raises(ValueError, match=r"crossings must be \(n,\)"):
        survival(np.array([[1.0, 2.0]]), HORIZON)


def test_an_even_count_crossing_multiset_gives_an_integer_horizon_without_rounding():
    crossings = np.array([2.0, 2.0, NEVER, NEVER])
    assert np.median(crossings) == 4.5   # the half-step a quantile reading would print
    # h:            0    1    2    3    4    5    6
    expected = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]
    curve = survival(crossings, HORIZON)
    assert curve.tolist() == expected
    horizons = {q: trust_horizon(curve, q) for q in (0.5, 0.75, 0.9)}
    assert horizons == {0.5: 6, 0.75: 1, 0.9: 1}
    assert all(type(h) is int for h in horizons.values())


def test_trust_horizon_is_the_largest_step_at_or_above_q():
    curve = np.array([1.0, 1.0, 0.75, 0.75, 0.5, 0.25, 0.0])
    assert trust_horizon(curve, 1.0) == 1
    assert trust_horizon(curve, 0.9) == 1
    assert trust_horizon(curve, 0.75) == 3    # >=, not >: S(3) == 0.75 qualifies
    assert trust_horizon(curve, 0.5) == 4
    assert trust_horizon(curve, 0.25) == 5
    assert trust_horizon(curve, 0.1) == 5     # S(6) == 0 < 0.1
    assert trust_horizon(curve, 0.0) == 6


def test_trust_horizon_is_zero_when_no_step_reaches_q():
    assert trust_horizon(np.array([0.5, 0.25]), 0.75) == 0
    assert trust_horizon(np.array([1.0, 1.0]), 1.5) == 0
    assert type(trust_horizon(np.array([0.5, 0.25]), 0.75)) is int


def test_trust_horizon_refuses_a_curve_that_is_not_one_dimensional():
    with pytest.raises(ValueError, match=r"surv must be \(H \+ 1,\)"):
        trust_horizon(np.ones((2, HORIZON + 1)), 0.75)


def test_crossings_survive_into_the_horizon_end_to_end():
    # The four-row prior from the crossing test plus one never-moved row:
    # crossings {3, 5, never, 1} and NaN. Hand-built S over the four finite:
    # h:            0    1     2     3    4    5     6
    expected = [1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25]
    persist = np.full((5, HORIZON), 10.0)
    model = np.array([
        [5.0, 5.0, 11.0, 11.0, 11.0, 11.0],
        [12.0, 12.0, 5.0, 5.0, 12.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        [11.0, 9.0, 9.0, 9.0, 9.0, 9.0],
        [11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
    ])
    moved = np.array([
        [T, T, T, T, T, T],
        [F, F, T, T, T, T],
        [F, T, T, T, T, T],
        [T, T, T, T, T, T],
        [F, F, F, F, F, F],
    ])
    curve = survival(crossing_step(model, persist, moved), HORIZON)
    assert curve.tolist() == expected
    assert trust_horizon(curve, 0.75) == 2
    assert trust_horizon(curve, 0.5) == 4
    assert trust_horizon(curve, 0.9) == 0   # only S(0) reaches 0.9: no step is trusted
```

Why these particular numbers, so the engineer does not "simplify" them away: the `(3, 4)` / `(2.5, 2.5)` pair in the threshold test is the only pair that separates the Euclidean norm from both the L1 (`(2.5, 2.5)` sums to 5) and the per-axis-max (`(3, 4)` maxes at 4) readings in one row; row 1 of the crossing prior loses at steps 1 and 2 *before* it moves so that a search that ignores `h0` returns 1 instead of 5; the returns-near-start row has its only loss at a step where `moved` is False so that a search that requires `moved` at the crossing step returns 7 instead of 3; and `{2, 2, 7, 7}` has median 4.5, the half-step an `np.median` reading of the multiset would print where `H*_0.5` prints 6.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_trust.py -q -p no:cacheprovider`

Expected: the whole module fails at collection, since the import at the top of the file has nothing to import:

```
tests/eval/test_trust.py:31: in <module>
    from mbfps.eval.trust import (
E   ModuleNotFoundError: No module named 'mbfps.eval.trust'
=========================== short test summary info ============================
ERROR tests/eval/test_trust.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.07s
```

No individual test runs red here -- there is no partial state to read. Do not create an empty `trust.py` to "see the ImportErrors"; Step 3 is the whole module.

- [ ] **Step 3: Implement the five functions**

Create `src/mbfps/eval/trust.py` with exactly this content. Nothing in it imports torch or reads a file.

```python
"""The trust horizon: per-window crossing steps, margins and survival curves.

`scripts/diagnose_dynamics.py` decides the M3 gate from two MEAN curves --
`k45_position` and `persistence_position` -- and a mean curve cannot say on how
many windows the model was still ahead of "you did not move" at step h. These
functions work on the PER-WINDOW error rows the M3d reference pass keeps, and
every one of them is a pure numpy function with no torch, no files and no
randomness, pinned by synthetic priors with hand-built answers:

  * `moved_mask` -- the ground-truth gate on every other quantity. A window
    that has not left its starting point by step h has a persistence error of
    zero there, and any model error at all "loses" to it; the mask is the 2-D
    Euclidean displacement `|p(h) - p(0)| >= MIN_MOVE`, computed once from the
    true positions so the probe channel and the probe-free channel score the
    same rows.
  * `crossing_step` -- the first step, at or after the first moved step, at
    which the model's error STRICTLY exceeds persistence's. Ties never cross:
    a persistence clone is not worse than persistence, so its crossing is
    `horizon + 1` on every row, the same as a perfect predictor's -- the two
    are told apart by the margin (and, in the decomposition, by the ratio),
    not by the crossing.
  * `persistence_margin` -- `persist_err - model_err`, map units, positive
    when the model beats persistence at that step. It is `gap_closed`'s
    numerator PER WINDOW, and it is the decision statistic instead of
    `gap_closed` because a ratio of mean curves has no per-window estimator to
    cluster by episode.
  * `survival` and `trust_horizon` -- `S(h)`, the fraction of finite crossings
    strictly beyond h, and `H*_q`, the largest h with `S(h) >= q`. `H*_q` is an
    integer by construction: it reads the curve, never a quantile of the
    multiset, so an even-count multiset gives no half-step.
"""

import numpy as np

MIN_MOVE: float = 5.0


def moved_mask(
    p_true: np.ndarray, p_true0: np.ndarray, min_move: float = MIN_MOVE
) -> np.ndarray:
    """`(n, H, 2)` true positions and `(n, 2)` anchors -> `(n, H)` bool.

    True where the 2-D Euclidean displacement `|p_true[:, h] - p_true0|` is at
    least `min_move` -- `>=`, so a window that moves exactly the threshold is
    scored. Euclidean, not per-axis: `(3, 4)` has moved 5 units, `(2.5, 2.5)`
    has moved 3.54, and only the norm orders them the way the map does.
    """
    p_true = np.asarray(p_true, dtype=np.float64)
    p_true0 = np.asarray(p_true0, dtype=np.float64)
    if p_true.ndim != 3 or p_true.shape[-1] != 2:
        raise ValueError(f"p_true must be (n, H, 2), got {p_true.shape}")
    if p_true0.shape != (p_true.shape[0], 2):
        raise ValueError(
            f"p_true0 must be (n, 2) = ({p_true.shape[0]}, 2), got {p_true0.shape}"
        )
    displacement = np.linalg.norm(p_true - p_true0[:, None, :], axis=-1)
    return displacement >= min_move


def crossing_step(
    model_err: np.ndarray, persist_err: np.ndarray, moved: np.ndarray
) -> np.ndarray:
    """Per row, the first 1-based step at or after the first moved step where
    `model_err > persist_err` STRICTLY; `H + 1` if never; NaN if the row never
    moves.

    All three arguments are `(n, H)`; the result is `(n,)` float64 (NaN needs
    a float row). Steps BEFORE the first moved step are ignored even when the
    model loses there -- at those steps persistence's error is the probe's
    floor on a frame that has not changed, and nothing about the model is
    being measured.
    """
    model_err = np.asarray(model_err, dtype=np.float64)
    persist_err = np.asarray(persist_err, dtype=np.float64)
    moved = np.asarray(moved, dtype=bool)
    if model_err.ndim != 2:
        raise ValueError(f"model_err must be (n, H), got {model_err.shape}")
    if persist_err.shape != model_err.shape or moved.shape != model_err.shape:
        raise ValueError(
            "model_err, persist_err and moved must share one (n, H) shape, got "
            f"{model_err.shape}, {persist_err.shape}, {moved.shape}"
        )
    if not (np.isfinite(model_err).all() and np.isfinite(persist_err).all()):
        raise ValueError("crossing_step: model_err and persist_err must be finite")
    n, horizon = model_err.shape
    ever_moved = moved.any(axis=1)
    first_moved = moved.argmax(axis=1)  # 0-based index of the first True per row
    at_or_after = np.arange(horizon)[None, :] >= first_moved[:, None]
    crosses = (model_err > persist_err) & at_or_after
    crossing = np.where(
        crosses.any(axis=1), crosses.argmax(axis=1) + 1, horizon + 1
    ).astype(np.float64)
    crossing[~ever_moved] = np.nan
    return crossing


def persistence_margin(model_err: np.ndarray, persist_err: np.ndarray) -> np.ndarray:
    """`persist_err - model_err`, elementwise on `(n, H)`: positive where the
    model beats persistence at that step, zero for a persistence clone."""
    model_err = np.asarray(model_err, dtype=np.float64)
    persist_err = np.asarray(persist_err, dtype=np.float64)
    if model_err.shape != persist_err.shape:
        raise ValueError(
            f"model_err and persist_err must share a shape, got {model_err.shape} "
            f"and {persist_err.shape}"
        )
    return persist_err - model_err


def survival(crossings: np.ndarray, horizon: int) -> np.ndarray:
    """`S(h)` for h = 0..horizon: the fraction of FINITE crossings strictly
    greater than h. `(horizon + 1,)` float; all NaN when no crossing is finite.

    NaN crossings are windows that never moved; they are excluded from the
    denominator, not counted as failures. Every finite crossing must lie in
    `1..horizon + 1` (the value `crossing_step` produces): a 0 would be a
    0-based step and a `horizon + 2` a crossing from a longer horizon, and
    either would move `S` without any test noticing.
    """
    crossings = np.asarray(crossings, dtype=np.float64)
    if crossings.ndim != 1:
        raise ValueError(f"crossings must be (n,), got {crossings.shape}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    finite = crossings[np.isfinite(crossings)]
    if finite.size == 0:
        return np.full(horizon + 1, np.nan)
    if (finite < 1).any() or (finite > horizon + 1).any():
        raise ValueError(
            f"crossings must lie in 1..{horizon + 1}, got min {finite.min()} "
            f"max {finite.max()}"
        )
    steps = np.arange(horizon + 1)
    return (finite[None, :] > steps[:, None]).mean(axis=1)


def trust_horizon(surv: np.ndarray, q: float) -> int:
    """`H*_q`: the largest h with `surv[h] >= q`; 0 when no h qualifies; -1
    when `surv` is all NaN (no finite crossing to read)."""
    surv = np.asarray(surv, dtype=np.float64)
    if surv.ndim != 1:
        raise ValueError(f"surv must be (H + 1,), got {surv.shape}")
    if np.isnan(surv).all():
        return -1
    qualifying = np.flatnonzero(surv >= q)
    if qualifying.size == 0:
        return 0
    return int(qualifying[-1])
```

Two things the code does that the contract states and a "cleaner" rewrite would lose: `crosses` is masked by `at_or_after` (every step from the first moved one onward), **not** by `moved` itself -- the spec's `h×` is "the first h ≥ h₀", and a window that comes back within 5 units of its start after moving still crosses at a loss there; and `survival`'s denominator is `finite.size`, never `crossings.size` -- a never-moved window is counted and excluded, not scored as a loss.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_trust.py -q -p no:cacheprovider`

Expected: `28 passed` (25 test functions; `test_survival_starts_at_one_whenever_any_crossing_is_finite` is parametrised four ways). Measured `28 passed in 0.04s` while writing this task.

- [ ] **Step 5: Mutation-test**

Set up a self-checked harness. The three self-checks all matter here: the package is installed editable, so a harness that copies the tree and forgets `PYTHONPATH` tests the *repo's* `trust.py` and reports every mutation caught; a same-length mutation (`>=` for `>`) can leave a stale `.pyc`; and an unproven harness proves nothing. `scratchpad/` is gitignored; save this there, run it, and read the first three lines before the table:

```python
# scratchpad/mutate_task1.py -- gitignored, never committed
import os, shutil, subprocess, sys
from pathlib import Path

REPO = Path("/Users/raphaelchen/Desktop/csgo-bot")
S = Path(__file__).resolve().parent / "tree"            # a copy of src/ tests/ scripts/ pyproject.toml
SRC = S / "src" / "mbfps" / "eval" / "trust.py"
PY = str(REPO / ".venv" / "bin" / "python")
ENV = dict(os.environ, PYTHONPATH=str(S / "src"), PYTHONDONTWRITEBYTECODE="1")

def clear_pyc():
    for d in S.rglob("__pycache__"):
        shutil.rmtree(d)

def run():
    clear_pyc()
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "tests/eval/test_trust.py"], cwd=S, env=ENV,
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
 "FATAL: persistence_margin raises": ("    return persist_err - model_err", '    raise RuntimeError("harness self-check")'),
 "M1 MIN_MOVE = 4.0": ("MIN_MOVE: float = 5.0", "MIN_MOVE: float = 4.0"),
 "M2 moved_mask > not >=": ("    return displacement >= min_move", "    return displacement > min_move"),
 "M3 moved_mask per-axis max (Linf)": ("    displacement = np.linalg.norm(p_true - p_true0[:, None, :], axis=-1)", "    displacement = np.abs(p_true - p_true0[:, None, :]).max(axis=-1)"),
 "M4 moved_mask L1": ("    displacement = np.linalg.norm(p_true - p_true0[:, None, :], axis=-1)", "    displacement = np.abs(p_true - p_true0[:, None, :]).sum(axis=-1)"),
 "M5 moved_mask row-0 anchor for every row": ("p_true0[:, None, :], axis=-1)", "p_true0[:1, None, :], axis=-1)"),
 "M6 moved_mask drop p_true guard": ("    if p_true.ndim != 3 or p_true.shape[-1] != 2:", "    if False:"),
 "M7 moved_mask drop p_true0 guard": ("    if p_true0.shape != (p_true.shape[0], 2):", "    if False:"),
 "M8 crossing >= (ties cross)": ("    crosses = (model_err > persist_err) & at_or_after", "    crosses = (model_err >= persist_err) & at_or_after"),
 "M9 crossing ignores h0": ("    crosses = (model_err > persist_err) & at_or_after", "    crosses = (model_err > persist_err)"),
 "M10 crossing requires moved at the step": ("    crosses = (model_err > persist_err) & at_or_after", "    crosses = (model_err > persist_err) & moved"),
 "M11 crossing 0-based": ("crosses.argmax(axis=1) + 1, horizon + 1", "crosses.argmax(axis=1), horizon + 1"),
 "M12 crossing never = horizon": ("crosses.argmax(axis=1) + 1, horizon + 1", "crosses.argmax(axis=1) + 1, horizon"),
 "M13 crossing never-moved not NaN": ("    crossing[~ever_moved] = np.nan\n", ""),
 "M14 crossing drop ndim guard": ("    if model_err.ndim != 2:", "    if False:"),
 "M15 crossing drop shape guard": ("    if persist_err.shape != model_err.shape or moved.shape != model_err.shape:", "    if False:"),
 "M16 crossing drop finite guard": ("    if not (np.isfinite(model_err).all() and np.isfinite(persist_err).all()):", "    if False:"),
 "M17 margin sign flipped": ("    return persist_err - model_err", "    return model_err - persist_err"),
 "M18 margin drop shape guard": ("    if model_err.shape != persist_err.shape:", "    if False:"),
 "M19 survival NaN in denominator": ("    return (finite[None, :] > steps[:, None]).mean(axis=1)", "    return (finite[None, :] > steps[:, None]).sum(axis=1) / crossings.size"),
 "M20 survival >= h": ("    return (finite[None, :] > steps[:, None]).mean(axis=1)", "    return (finite[None, :] >= steps[:, None]).mean(axis=1)"),
 "M21 survival drop range guard": ("    if (finite < 1).any() or (finite > horizon + 1).any():", "    if False:"),
 "M22 survival drop horizon guard": ("    if horizon < 1:", "    if False:"),
 "M23 survival drop ndim guard": ("    if crossings.ndim != 1:", "    if False:"),
 "M24 survival all-NaN -> zeros": ("        return np.full(horizon + 1, np.nan)", "        return np.zeros(horizon + 1)"),
 "M25 survival drops h = 0": ("    steps = np.arange(horizon + 1)", "    steps = np.arange(1, horizon + 1)"),
 "M26 trust_horizon > not >=": ("    qualifying = np.flatnonzero(surv >= q)", "    qualifying = np.flatnonzero(surv > q)"),
 "M27 trust_horizon first not largest": ("    return int(qualifying[-1])", "    return int(qualifying[0])"),
 "M28 trust_horizon all-NaN -> 0": ("        return -1\n    qualifying", "        return 0\n    qualifying"),
 "M29 trust_horizon none-qualify -> -1": ("    if qualifying.size == 0:\n        return 0", "    if qualifying.size == 0:\n        return -1"),
 "M30 trust_horizon drop ndim guard": ("    if surv.ndim != 1:", "    if False:"),
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
        print(f"{'CAUGHT' if rc else 'SURVIVED':8} | {name} | {last} | {failed}")
        if name.startswith("FATAL") and rc == 0:
            sys.exit("harness cannot detect a fatal mutation")
finally:
    SRC.write_text(original)
    clear_pyc()   # self-check 3: never leave bytecode from a mutated source behind
```

Build the copy with `mkdir -p scratchpad/tree && cp -R src tests scripts pyproject.toml scratchpad/tree/` (after Steps 1 and 3, so the copy carries the new files) and run `.venv/bin/python scratchpad/mutate_task1.py` from the repo root. The first line must name `scratchpad/tree/src/mbfps/__init__.py`, the second must read `baseline: 28 passed`, and the third must be `CAUGHT | FATAL ...` with 4 failures -- a green FATAL row means the harness is not testing the copy; stop and fix it. Every row below must read `CAUGHT`:

| mutation | must be caught by |
|---|---|
| FATAL (harness self-check, run first): `persistence_margin` raises | 4 tests: `test_a_persistence_clone_never_crosses_and_its_margin_is_zero`, `test_a_perfect_predictor_never_crosses_and_its_margin_is_persistences_error`, `test_the_clone_and_the_perfect_predictor_share_a_crossing_and_differ_in_margin`, `test_persistence_margin_is_persistence_minus_model` |
| M1: `MIN_MOVE = 4.0` | `test_min_move_is_five_map_units`, `test_moved_mask_is_the_euclidean_displacement_at_or_above_the_threshold` (`(0, 4)` becomes moved) |
| M2: `moved_mask` uses `>` not `>=` | `test_moved_mask_is_the_euclidean_displacement_at_or_above_the_threshold` (the `(3, 4)` row sits exactly on 5), `test_moved_mask_honours_min_move` |
| M3: `moved_mask` takes the per-axis max instead of the norm | `test_moved_mask_is_the_euclidean_displacement_at_or_above_the_threshold` (the ONLY catcher: `(3, 4)` reads 4 and is no longer moved) |
| M4: `moved_mask` takes the L1 sum instead of the norm | `test_moved_mask_is_the_euclidean_displacement_at_or_above_the_threshold` (`(2.5, 2.5)` reads 5 and is moved), `test_moved_mask_honours_min_move` |
| M5: `moved_mask` reads row 0's anchor for every row (`p_true0[:1]`) | `test_moved_mask_measures_each_row_against_its_own_anchor` (the ONLY catcher: every other fixture has one row) |
| M6: `moved_mask` drops the `p_true` shape guard | `test_moved_mask_refuses_the_wrong_shapes` |
| M7: `moved_mask` drops the `p_true0` shape guard | `test_moved_mask_refuses_the_wrong_shapes` |
| M8: `crossing_step` uses `>=` (ties cross) | `test_ties_never_cross`, `test_a_persistence_clone_never_crosses_and_its_margin_is_zero`, `test_the_clone_and_the_perfect_predictor_share_a_crossing_and_differ_in_margin` |
| M9: `crossing_step` ignores `h0` (drops `& at_or_after`) | `test_crossing_is_the_first_strict_exceedance_at_or_after_the_first_moved_step` (row 1 reads 1, not 5), `test_a_row_that_never_moves_is_nan_whatever_its_errors_say` (row 2 reads 1, not 6), `test_crossings_survive_into_the_horizon_end_to_end` |
| M10: `crossing_step` requires `moved` at the crossing step (`& moved` for `& at_or_after`) | `test_a_window_that_returns_near_its_start_after_moving_still_crosses` (the ONLY catcher: 7 instead of 3) |
| M11: `crossing_step` returns 0-based steps | `test_crossing_is_the_first_strict_exceedance_at_or_after_the_first_moved_step`, `test_ties_never_cross`, `test_a_row_that_never_moves_is_nan_whatever_its_errors_say`, `test_a_window_that_returns_near_its_start_after_moving_still_crosses`, `test_crossings_survive_into_the_horizon_end_to_end` |
| M12: never = `horizon` instead of `horizon + 1` | the clone, perfect-predictor and clone-vs-perfect tests, `test_ties_never_cross`, `test_crossing_is_the_first_strict_exceedance_at_or_after_the_first_moved_step`, `test_crossings_survive_into_the_horizon_end_to_end` |
| M13: never-moved rows are not set to NaN | `test_a_row_that_never_moves_is_nan_whatever_its_errors_say`, `test_crossings_survive_into_the_horizon_end_to_end` (the NaN row enters the denominator) |
| M14: `crossing_step` drops the ndim guard | `test_crossing_step_refuses_mismatched_shapes_and_nonfinite_errors` |
| M15: `crossing_step` drops the shared-shape guard | `test_crossing_step_refuses_mismatched_shapes_and_nonfinite_errors` |
| M16: `crossing_step` drops the finite guard (a NaN error silently never crosses) | `test_crossing_step_refuses_mismatched_shapes_and_nonfinite_errors` |
| M17: `persistence_margin` sign flipped (`model - persist`) | `test_persistence_margin_is_persistence_minus_model`, `test_a_perfect_predictor_never_crosses_and_its_margin_is_persistences_error`, `test_the_clone_and_the_perfect_predictor_share_a_crossing_and_differ_in_margin` |
| M18: `persistence_margin` drops the shape guard | `test_persistence_margin_is_persistence_minus_model` |
| M19: `survival` divides by `crossings.size` (NaN rows in the denominator) | `test_survival_excludes_never_moved_rows_from_the_denominator`, `test_survival_starts_at_one_whenever_any_crossing_is_finite[one-nan]` and `[mixed]`, `test_crossings_survive_into_the_horizon_end_to_end` |
| M20: `survival` counts `crossing >= h` | `test_survival_is_the_fraction_of_finite_crossings_strictly_beyond_h`, `test_survival_excludes_never_moved_rows_from_the_denominator`, `test_an_even_count_crossing_multiset_gives_an_integer_horizon_without_rounding`, `test_crossings_survive_into_the_horizon_end_to_end` |
| M21: `survival` drops the `1..horizon + 1` range guard | `test_survival_refuses_crossings_outside_one_to_horizon_plus_one` |
| M22: `survival` drops the `horizon >= 1` guard | `test_survival_refuses_crossings_outside_one_to_horizon_plus_one` |
| M23: `survival` drops the ndim guard | `test_survival_refuses_crossings_outside_one_to_horizon_plus_one` |
| M24: `survival` returns zeros instead of NaN when nothing is finite | `test_survival_is_all_nan_without_a_finite_crossing_and_the_horizon_is_minus_one` (and `trust_horizon` then reads 0, not -1) |
| M25: `survival` starts at h = 1 (shape `(H,)`) | 8 tests, every one that reads `S(0)` or the curve's length: `test_the_persistence_clones_survival_is_one_everywhere_and_its_horizon_is_the_horizon`, `test_survival_is_the_fraction_of_finite_crossings_strictly_beyond_h`, `test_survival_excludes_never_moved_rows_from_the_denominator`, `test_survival_starts_at_one_whenever_any_crossing_is_finite[all-cross-at-1]`, `[one-nan]`, `[mixed]`, `test_an_even_count_crossing_multiset_gives_an_integer_horizon_without_rounding`, `test_crossings_survive_into_the_horizon_end_to_end` |
| M26: `trust_horizon` uses `>` not `>=` | `test_trust_horizon_is_the_largest_step_at_or_above_q` (q = 0.75 reads 1, not 3), `test_an_even_count_crossing_multiset_gives_an_integer_horizon_without_rounding`, the clone-survival test, `test_crossings_survive_into_the_horizon_end_to_end` |
| M27: `trust_horizon` returns the first qualifying h, not the largest | the same four tests as M26 |
| M28: `trust_horizon` returns 0 for an all-NaN curve | `test_survival_is_all_nan_without_a_finite_crossing_and_the_horizon_is_minus_one` |
| M29: `trust_horizon` returns -1 when no step qualifies | `test_trust_horizon_is_zero_when_no_step_reaches_q` (the ONLY catcher) |
| M30: `trust_horizon` drops the ndim guard | `test_trust_horizon_refuses_a_curve_that_is_not_one_dimensional` |

All thirty-one rows were run against this exact test file and this exact implementation on 2026-09-13; each row's catcher is what actually fired (M3, M5, M10 and M29 are each caught by exactly one test, which is why those tests exist). The `[never]` case of `starts_at_one` survives M25 by itself -- `survival([7.0], 6)` under M25 is `[1.0] * 6` and index 0 still reads 1 -- which is why the clone-survival test asserts the curve's *shape*. Any mutation that survives on your run is a missing test. Add it before committing.

- [ ] **Step 6: Run the full suite**

Run from the repo root: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`

Expected: **28 more than the baseline at b382194** -- the contract's baseline is 1278 passed, 0 warnings, so 1306 -- and no other file moves; record the measured number, since the absolute count is stale by construction and every later task states its delta against what this step measured. Nothing outside `tests/eval/test_trust.py` imports `mbfps.eval.trust` yet, so a failure elsewhere is unrelated to this task and must not be "fixed" here. ~6 min on this Mac; the slow real-data tests read `data/my_way_home` relative to the cwd, so run it from the repo root.

- [ ] **Step 7: Commit**

`git status` must show exactly the two new files (the `scratchpad/` harness and its `tree/` copy are gitignored; `study.log` is pre-existing and untracked, leave it).

```bash
git add src/mbfps/eval/trust.py tests/eval/test_trust.py
git commit -m "feat: trust.py -- per-window crossing step, persistence margin, survival curve and trust horizon

The M3 gate reads two mean curves and cannot say on how many windows the
model was still ahead of persistence at step h. These are the pure numpy
per-window functions M3d reads instead: the ground-truth moved mask
(|p(h) - p(0)| >= 5 map units, Euclidean) that gates every quantity; the
crossing step h x, the first STRICT loss to persistence at or after the
first moved step (H + 1 for never, NaN for never moved -- ties never
cross, so a persistence clone and a perfect predictor share H + 1 and are
told apart by the margin); the persistence margin, gap_closed's numerator
per window; and the survival curve S(h) with H*_q, the largest h with
S(h) >= q -- an integer by construction, read from the curve and never
from a quantile of the crossing multiset. Pinned by hand-built priors;
Task 2 appends the displacement decomposition and the scale correction.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: trust.py, part 2: displacement_decomposition, embedding_ratio, scale_corrected_error

Task 1 landed the crossing step, the margin and the survival curve; this task lands the three quantities that separate "the prior drifts slowly" from "the prior moves wrong" (spec 2.2, items 2 and 4): the magnitude ratios (probe-normalised and raw), the direction cosine, the probe-free embedding ratio, and the cross-fitted scale correction with its boundary flag. All pure numpy over `(n, H)` arrays, no torch, no files. Every function is pinned by a synthetic prior whose answer is written by hand in the test -- persistence clone, perfect predictor, exact 2x overshoot, pure-noise `d_hat`, a probe-attenuated prior -- and each prior is the mutation the others cannot catch: the clone and the perfect predictor share `h× = H + 1` (Task 1) and are told apart here by the ratio; the overshoot is told from the perfect predictor by `alpha`, not by the ratio's sign; the attenuated prior is why `ratio_probe` exists at all.

**Depends on Task 1** having created `src/mbfps/eval/trust.py` (with `MIN_MOVE`, `moved_mask`, `crossing_step`, `persistence_margin`, `survival`, `trust_horizon`) and `tests/eval/test_trust.py` with its header and imports. Nothing here restates them.

**Files:**
- Modify: `src/mbfps/eval/trust.py` -- APPEND after Task 1's last function (`trust_horizon`): `ALPHAS`, `Decomposition`, `displacement_decomposition`, `embedding_ratio`, `ScaleCorrection`, `_argmin_smallest_alpha`, `scale_corrected_error`. The header gains `from dataclasses import dataclass` if Task 1 did not already import it (Task 1 defines no dataclass, so it may not have).
- Test: `tests/eval/test_trust.py` -- two import statements added to the header immediately after Task 1's `from mbfps.eval.trust import (...)` block; everything else APPENDED at the end of the file. Task 1's tests are untouched.

**Interfaces:**
- Consumes (Task 1, `src/mbfps/eval/trust.py`): the module itself and its header imports (`import numpy as np`); `MIN_MOVE: float = 5.0` is referenced only in a docstring (`moved` means `|d| >= MIN_MOVE > 0`, which is why `ratio_raw` needs no zero guard). No Task 1 function is called: every mask in this task's tests is built by hand, so the tests do not depend on `moved_mask`.
- Produces (verbatim from the contract, consumed by Task 4's `trust_record` and Task 6's ratio pooling):

```python
ALPHAS: np.ndarray = np.linspace(0.0, 2.0, 201)

@dataclass(frozen=True)
class Decomposition:
    ratio_probe: np.ndarray        # (n, H) |d_hat| / |d_hat_real|; NaN where not moved or |d_hat_real| == 0
    ratio_raw: np.ndarray          # (n, H) |d_hat| / |d|;          NaN where not moved
    cosine: np.ndarray             # (n, H) cos(d_hat, d);           NaN where not moved or |d_hat| == 0
    moved: np.ndarray              # (n, H) bool (the mask passed in)
    zero_displacement: np.ndarray  # (n, H) bool: moved and |d_hat| == 0

def displacement_decomposition(p_hat, p_true, p_hat0, p_true0, p_hat_real, moved) -> Decomposition
    # p_hat (n, H, 2), p_true (n, H, 2), p_hat0 (n, 2), p_true0 (n, 2), p_hat_real (n, H, 2), moved (n, H) bool
    # d_hat = p_hat - p_hat0[:, None]; d = p_true - p_true0[:, None]; d_hat_real = p_hat_real - p_hat0[:, None]

def embedding_ratio(e_hat_disp: np.ndarray, e_true_disp: np.ndarray, moved: np.ndarray) -> np.ndarray
    # (n, H) norms ||e_hat(h) - e_hat(0)||, (n, H) norms ||e(h) - e(0)||, moved -> (n, H): e_hat_disp / e_true_disp; NaN where not moved or e_true_disp == 0

@dataclass(frozen=True)
class ScaleCorrection:
    alpha_a: np.ndarray      # (H,) alpha fit on fold A (even episode labels), NaN if unavailable
    alpha_b: np.ndarray      # (H,) alpha fit on fold B (odd labels)
    score_a: np.ndarray      # (H,) median over fold B rows of |p_hat0 + alpha_a * d_hat - p_true|  (fit A, scored on B)
    score_b: np.ndarray      # (H,) median over fold A rows of the same with alpha_b               (fit B, scored on A)
    held_out: np.ndarray     # (n, H) every row scored with the alpha fit on the OTHER fold; NaN where not moved
    boundary: np.ndarray     # (H,) bool: alpha_a or alpha_b equals alphas[0] or alphas[-1]
    folds_available: bool    # False when fewer than 2 distinct episode labels; then every array is NaN / False

def scale_corrected_error(p_hat, p_true, p_hat0, moved, episode, alphas=ALPHAS) -> ScaleCorrection
    # episode (n,) int labels; fold A = rows with episode % 2 == 0, fold B = episode % 2 == 1.
    # Fit objective per h and fold: argmin over alphas of median over the fold's MOVED rows of |p_hat0 + alpha*d_hat - p_true[:, h]|
    # (2-D Euclidean norm). Ties in the argmin -> the smallest alpha.
```

  Two behaviours the contract leaves open and this task fixes (both tested, both in the mutation table): a fold with **no moved rows at a step** cannot be fit there -- `alpha` for that fold, its `score`, and the other fold's `held_out` entries are NaN at that step, `boundary[h]` is False, and `folds_available` is unchanged; `folds_available` is a Python `bool`, so `is True` / `is False` hold.

> RESOLVED AT RECONCILIATION: the spec (section 5) writes `embedding_ratio(e_hat, e_true, e_hat0, e_true0, moved)`; the contract pins `embedding_ratio(e_hat_disp, e_true_disp, moved)` over the two norms the Task 3 hook already reduces (`Trajectories.embedding_displacement`, `true_embedding_displacement`), and the contract governs: Task 4 calls `embedding_ratio(traj.embedding_displacement, traj.true_embedding_displacement, moved)`. The spec's five-argument form is superseded.

> RESOLVED AT RECONCILIATION: the contract is silent on a fold with no moved rows at one step (real at small h: ~58 % of windows are not moved at h=1), and on two distinct labels of the same parity (one fold empty at every step; `folds_available` stays True by the contract's literal rule and that fold's `alpha` is NaN everywhere). The per-step NaN rule above covers both, as written: Task 6's `probe_cells` masks `held_out` with `np.isfinite(values)`, so an all-NaN column at a step drops out of the pool.

- [ ] **Step 1: Write the failing tests**

(a) In the header of `tests/eval/test_trust.py`, immediately after Task 1's `from mbfps.eval.trust import (...)` block, add these two statements (a second import from the same module is legal; Task 1's block stays as it is):

```python
import warnings

from mbfps.eval.trust import (  # Task 2: decomposition, embedding ratio, scale correction
    ALPHAS,
    Decomposition,
    ScaleCorrection,
    displacement_decomposition,
    embedding_ratio,
    scale_corrected_error,
)
```

(b) Append at the end of the file:

```python
# ============================================================================
# Task 2: displacement_decomposition, embedding_ratio, scale_corrected_error
# ============================================================================
#
# One ground truth serves every prior below, so the priors differ ONLY in what
# the model does with it. Row i starts at (10 i, 0) and walks +x by 5 (i + 1)
# map units per step, so |d(i, h)| = 5 (i + 1) (h + 1) -- every cell is moved
# (>= MIN_MOVE = 5), every displacement is an exact integer along one axis, and
# the fold split (episode i, even -> A, odd -> B) puts rows 0, 2 in A and rows
# 1, 3 in B. The probe reads every position with a constant bias of 3 units in
# +y, orthogonal to every walk: the prior's anchor p_hat(0) is the truth plus
# that bias, and so is a perfect prediction. The bias makes the perfect
# predictor's raw error a non-zero 3.0 at every cell, so "held-out error equals
# the raw error" is a real number and not 0 == 0.

N, H = 4, 3
BIAS = np.array([0.0, 3.0])


def _truth():
    p_true0 = np.stack([10.0 * np.arange(N), np.zeros(N)], axis=1)  # (N, 2)
    steps = 5.0 * np.arange(1, N + 1)[:, None] * np.arange(1, H + 1)[None, :]  # (N, H)
    p_true = p_true0[:, None, :] + np.stack([steps, np.zeros_like(steps)], axis=-1)
    return p_true0, p_true


def _displacement():
    """|d(i, h)| = 5 (i + 1) (h + 1), by hand."""
    return 5.0 * np.arange(1, N + 1)[:, None] * np.arange(1, H + 1)[None, :]


def _prior(scale: float):
    """A prior that predicts the anchor plus `scale` times the true displacement.

    scale 0 = persistence clone, 1 = perfect predictor, 2 = exact 2x overshoot.
    p_hat_real is the same probe reading the real future frames: the truth plus
    the bias, i.e. the perfect predictor.
    """
    p_true0, p_true = _truth()
    p_hat0 = p_true0 + BIAS
    d = p_true - p_true0[:, None]
    p_hat = p_hat0[:, None] + scale * d
    p_hat_real = p_true + BIAS
    return p_hat, p_true, p_hat0, p_true0, p_hat_real


ALL_MOVED = np.ones((N, H), dtype=bool)
EPISODE = np.arange(N)  # rows 0, 2 -> fold A; rows 1, 3 -> fold B


def test_alphas_is_the_pinned_grid():
    assert ALPHAS.shape == (201,)
    assert ALPHAS[0] == 0.0 and ALPHAS[-1] == 2.0
    assert ALPHAS[1] == pytest.approx(0.01)
    # the two values the priors below land on are exact grid points
    assert ALPHAS[100] == 1.0 and ALPHAS[50] == 0.5


# --- displacement_decomposition ---------------------------------------------


def test_persistence_clone_reads_ratio_zero_cosine_nan_and_flags_zero_displacement():
    dec = displacement_decomposition(*_prior(0.0), ALL_MOVED)
    assert isinstance(dec, Decomposition)
    np.testing.assert_array_equal(dec.ratio_probe, np.zeros((N, H)))
    np.testing.assert_array_equal(dec.ratio_raw, np.zeros((N, H)))
    assert np.isnan(dec.cosine).all()
    np.testing.assert_array_equal(dec.zero_displacement, ALL_MOVED)
    assert dec.moved is ALL_MOVED or np.array_equal(dec.moved, ALL_MOVED)


def test_perfect_predictor_reads_ratio_one_cosine_one():
    dec = displacement_decomposition(*_prior(1.0), ALL_MOVED)
    np.testing.assert_array_equal(dec.ratio_probe, np.ones((N, H)))
    np.testing.assert_array_equal(dec.ratio_raw, np.ones((N, H)))
    np.testing.assert_array_equal(dec.cosine, np.ones((N, H)))
    assert not dec.zero_displacement.any()


def test_exact_overshoot_reads_ratio_two_cosine_one():
    dec = displacement_decomposition(*_prior(2.0), ALL_MOVED)
    np.testing.assert_array_equal(dec.ratio_probe, np.full((N, H), 2.0))
    np.testing.assert_array_equal(dec.ratio_raw, np.full((N, H), 2.0))
    np.testing.assert_array_equal(dec.cosine, np.ones((N, H)))


def test_probe_attenuation_cancels_in_ratio_probe_but_not_in_ratio_raw():
    # A flatter probe reads BOTH the imagined and the real displacement at 0.3
    # of their true length; the probe-normalised ratio is the imagined
    # displacement over the same probe's reading of the real one, so the
    # attenuation cancels and the prior is read as moving the right distance.
    p_hat, p_true, p_hat0, p_true0, _ = _prior(0.3)
    p_hat_real = p_hat0[:, None] + 0.3 * (p_true - p_true0[:, None])
    dec = displacement_decomposition(p_hat, p_true, p_hat0, p_true0, p_hat_real, ALL_MOVED)
    np.testing.assert_allclose(dec.ratio_probe, np.ones((N, H)))
    np.testing.assert_allclose(dec.ratio_raw, np.full((N, H), 0.3))
    np.testing.assert_allclose(dec.cosine, np.ones((N, H)))


def test_decomposition_respects_the_moved_mask_and_a_dead_probe_reading():
    # Row 0 is perfect but marked not-moved at h = 1; row 1 is a persistence
    # clone (d_hat = 0) marked not-moved everywhere; row 2's real-frame probe
    # reading never leaves the anchor (|d_hat_real| = 0); row 3 is perfect.
    p_hat, p_true, p_hat0, p_true0, p_hat_real = (a.copy() for a in _prior(1.0))
    p_hat[1] = p_hat0[1]
    p_hat_real[2] = p_hat0[2]
    moved = np.ones((N, H), dtype=bool)
    moved[0, 1] = False
    moved[1, :] = False
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no divide-by-zero RuntimeWarning may escape
        dec = displacement_decomposition(p_hat, p_true, p_hat0, p_true0, p_hat_real, moved)
    # not moved -> every channel NaN, whatever the prior did there
    assert np.isnan(dec.ratio_probe[0, 1]) and np.isnan(dec.ratio_raw[0, 1]) and np.isnan(dec.cosine[0, 1])
    assert np.isnan(dec.ratio_probe[1]).all() and np.isnan(dec.ratio_raw[1]).all() and np.isnan(dec.cosine[1]).all()
    # a zero d_hat on a not-moved row is NOT a zero-displacement exclusion
    assert not dec.zero_displacement[1].any() and not dec.zero_displacement[0, 1]
    assert not dec.zero_displacement.any()
    # the dead probe reading: ratio_probe NaN, ratio_raw and cosine still read
    assert np.isnan(dec.ratio_probe[2]).all()
    np.testing.assert_array_equal(dec.ratio_raw[2], np.ones(H))
    np.testing.assert_array_equal(dec.cosine[2], np.ones(H))
    # everything else untouched
    np.testing.assert_array_equal(dec.ratio_probe[3], np.ones(H))
    np.testing.assert_array_equal(dec.ratio_probe[0, [0, 2]], np.ones(2))
    np.testing.assert_array_equal(dec.moved, moved)
    assert dec.ratio_probe.shape == dec.ratio_raw.shape == dec.cosine.shape == (N, H)


# --- embedding_ratio -----------------------------------------------------------


def test_embedding_ratio_is_the_imagined_over_the_true_norm():
    e_hat_disp = np.array([[2.0, 4.0], [0.0, 1.0]])
    e_true_disp = np.array([[1.0, 4.0], [3.0, 2.0]])
    out = embedding_ratio(e_hat_disp, e_true_disp, np.ones((2, 2), dtype=bool))
    np.testing.assert_array_equal(out, np.array([[2.0, 1.0], [0.0, 0.5]]))


def test_embedding_ratio_is_nan_where_not_moved_or_where_the_truth_did_not_move():
    e_hat_disp = np.array([[2.0, 4.0], [7.0, 1.0]])
    e_true_disp = np.array([[1.0, 4.0], [0.0, 2.0]])
    moved = np.array([[True, False], [True, True]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = embedding_ratio(e_hat_disp, e_true_disp, moved)
    assert out[0, 0] == 2.0
    assert np.isnan(out[0, 1])  # not moved
    assert np.isnan(out[1, 0])  # moved, but the true embedding did not move: 7 / 0 is not a ratio
    assert out[1, 1] == 0.5
    assert out.shape == (2, 2)


# --- scale_corrected_error ----------------------------------------------------


def test_perfect_predictor_fits_alpha_one_on_both_folds_and_held_out_equals_its_raw_error():
    p_hat, p_true, p_hat0, _, _ = _prior(1.0)
    sc = scale_corrected_error(p_hat, p_true, p_hat0, ALL_MOVED, EPISODE)
    assert isinstance(sc, ScaleCorrection)
    assert sc.folds_available is True
    np.testing.assert_array_equal(sc.alpha_a, np.ones(H))
    np.testing.assert_array_equal(sc.alpha_b, np.ones(H))
    raw_error = np.linalg.norm(p_hat - p_true, axis=-1)  # 3.0 everywhere: the probe's bias
    np.testing.assert_array_equal(raw_error, np.full((N, H), 3.0))
    np.testing.assert_array_equal(sc.held_out, raw_error)
    np.testing.assert_array_equal(sc.score_a, np.full(H, 3.0))
    np.testing.assert_array_equal(sc.score_b, np.full(H, 3.0))
    assert not sc.boundary.any()
    assert sc.alpha_a.shape == sc.alpha_b.shape == sc.score_a.shape == sc.score_b.shape == sc.boundary.shape == (H,)
    assert sc.held_out.shape == (N, H)


def test_exact_overshoot_fits_alpha_half_and_its_held_out_error_is_the_perfect_predictors():
    p_hat, p_true, p_hat0, _, _ = _prior(2.0)
    sc = scale_corrected_error(p_hat, p_true, p_hat0, ALL_MOVED, EPISODE)
    np.testing.assert_array_equal(sc.alpha_a, np.full(H, 0.5))
    np.testing.assert_array_equal(sc.alpha_b, np.full(H, 0.5))
    # halved, the overshoot IS the perfect predictor: 3.0 of probe bias, nothing else
    np.testing.assert_array_equal(sc.held_out, np.full((N, H), 3.0))
    # ... while its raw error is far larger, so the correction did something
    raw_error = np.linalg.norm(p_hat - p_true, axis=-1)
    np.testing.assert_array_equal(raw_error, np.sqrt(9.0 + _displacement() ** 2))
    assert not sc.boundary.any()


def test_pure_noise_displacement_is_a_boundary_on_both_folds():
    # d_hat orthogonal to the truth on every row, sign alternating, from an
    # anchor exactly on the truth (no probe bias here: a biased anchor lets the
    # rows whose noise points against the bias cancel it, and that is a
    # correction, not noise). Then |alpha d_hat - d| = |d| sqrt(1 + alpha^2) on
    # every row: no positive alpha reduces the median error, the argmin lands
    # on alpha = 0 -- the grid edge -- and the record must say so. The
    # corrected number is NOT read: a boundary is not a correction (spec 2.2).
    _, p_true, _, p_true0, _ = _prior(1.0)
    p_hat0 = p_true0
    d = p_true - p_true0[:, None]
    orthogonal = np.stack([-d[..., 1], d[..., 0]], axis=-1)  # rotate d by 90 degrees
    signs = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)[:, None, None]
    p_hat = p_hat0[:, None] + signs * orthogonal
    sc = scale_corrected_error(p_hat, p_true, p_hat0, ALL_MOVED, EPISODE)
    np.testing.assert_array_equal(sc.alpha_a, np.zeros(H))
    np.testing.assert_array_equal(sc.alpha_b, np.zeros(H))
    np.testing.assert_array_equal(sc.boundary, np.ones(H, dtype=bool))
    assert sc.folds_available is True


@pytest.mark.parametrize("grid", [ALPHAS, ALPHAS[::-1]], ids=["ascending", "descending"])
def test_persistence_clone_ties_every_alpha_and_the_smallest_alpha_wins(grid):
    # d_hat = 0, so alpha changes nothing and every grid point ties. The tie
    # rule is the SMALLEST alpha, not the first index -- pinned by running the
    # grid backwards too. At alpha = 0 the "corrected" error is persistence
    # itself, and the flag says boundary.
    p_hat, p_true, p_hat0, _, _ = _prior(0.0)
    sc = scale_corrected_error(p_hat, p_true, p_hat0, ALL_MOVED, EPISODE, alphas=grid)
    np.testing.assert_array_equal(sc.alpha_a, np.zeros(H))
    np.testing.assert_array_equal(sc.alpha_b, np.zeros(H))
    np.testing.assert_array_equal(sc.boundary, np.ones(H, dtype=bool))
    persistence_error = np.sqrt(9.0 + _displacement() ** 2)  # |p_hat0 - p(h)| by hand
    np.testing.assert_array_equal(sc.held_out, persistence_error)


def test_fewer_than_two_episode_labels_disables_both_folds():
    p_hat, p_true, p_hat0, _, _ = _prior(1.0)
    sc = scale_corrected_error(p_hat, p_true, p_hat0, ALL_MOVED, np.zeros(N, dtype=int))
    assert sc.folds_available is False
    for arr in (sc.alpha_a, sc.alpha_b, sc.score_a, sc.score_b):
        assert arr.shape == (H,) and np.isnan(arr).all()
    assert sc.held_out.shape == (N, H) and np.isnan(sc.held_out).all()
    assert sc.boundary.shape == (H,) and sc.boundary.dtype == bool and not sc.boundary.any()


def test_folds_are_even_and_odd_episode_labels_and_each_scores_the_other():
    # Fold A (rows 0, 2) is perfect; fold B (rows 1, 3) overshoots 2x. Then
    # alpha_a = 1 and alpha_b = 0.5, B's rows are scored with alpha_a = 1
    # (the overshoot left as is: sqrt(9 + |d|^2)) and A's rows with
    # alpha_b = 0.5 (the perfect prediction halved: sqrt(9 + |d|^2 / 4)).
    perfect = _prior(1.0)
    over = _prior(2.0)
    p_hat = np.where((np.arange(N) % 2 == 0)[:, None, None], perfect[0], over[0])
    _, p_true, p_hat0, _, _ = perfect
    sc = scale_corrected_error(p_hat, p_true, p_hat0, ALL_MOVED, EPISODE)
    np.testing.assert_array_equal(sc.alpha_a, np.ones(H))
    np.testing.assert_array_equal(sc.alpha_b, np.full(H, 0.5))
    disp = _displacement()
    expected = np.empty((N, H))
    expected[[1, 3]] = np.sqrt(9.0 + disp[[1, 3]] ** 2)  # fold B scored with alpha_a = 1
    expected[[0, 2]] = np.sqrt(9.0 + (0.5 * disp[[0, 2]]) ** 2)  # fold A scored with alpha_b = 0.5
    np.testing.assert_allclose(sc.held_out, expected)
    # the scoring-fold medians: score_a is over B's rows, score_b over A's
    np.testing.assert_allclose(sc.score_a, np.median(expected[[1, 3]], axis=0))
    np.testing.assert_allclose(sc.score_b, np.median(expected[[0, 2]], axis=0))
    assert not sc.boundary.any()


def test_the_fit_is_a_median_so_a_minority_outlier_row_does_not_move_alpha():
    # Fold A gets a third row (episode 4): two perfect rows and one 2x
    # overshoot. The median error at alpha = 1 is 3.0 (two of three rows) and
    # larger at every other alpha; a mean would be pulled toward 0.5.
    perfect = _prior(1.0)
    over = _prior(2.0)
    p_hat = np.concatenate([perfect[0], over[0][:1]])  # row 4 = row 0's walk, overshot
    p_true = np.concatenate([perfect[1], perfect[1][:1]])
    p_hat0 = np.concatenate([perfect[2], perfect[2][:1]])
    episode = np.array([0, 1, 2, 3, 4])
    sc = scale_corrected_error(p_hat, p_true, p_hat0, np.ones((5, H), dtype=bool), episode)
    np.testing.assert_array_equal(sc.alpha_a, np.ones(H))
    np.testing.assert_array_equal(sc.alpha_b, np.ones(H))


def test_scale_correction_fits_and_scores_only_moved_rows():
    # Row 2 (fold A) is a 2x overshoot marked not-moved everywhere. Fit on the
    # moved rows only, fold A is row 0 alone -> alpha_a = 1; had row 2 been
    # counted the two-row median would leave 1. Row 2 is never scored.
    perfect = _prior(1.0)
    over = _prior(2.0)
    p_hat = perfect[0].copy()
    p_hat[2] = over[0][2]
    _, p_true, p_hat0, _, _ = perfect
    moved = np.ones((N, H), dtype=bool)
    moved[2] = False
    sc = scale_corrected_error(p_hat, p_true, p_hat0, moved, EPISODE)
    np.testing.assert_array_equal(sc.alpha_a, np.ones(H))
    np.testing.assert_array_equal(sc.alpha_b, np.ones(H))
    assert np.isnan(sc.held_out[2]).all()
    np.testing.assert_array_equal(sc.held_out[[0, 1, 3]], np.full((3, H), 3.0))
    np.testing.assert_array_equal(sc.score_a, np.full(H, 3.0))
    np.testing.assert_array_equal(sc.score_b, np.full(H, 3.0))


def test_a_fold_with_no_moved_rows_at_a_step_is_nan_at_that_step_only():
    # At h = 0 neither fold-B row moved: fold B cannot be fit there (alpha_b
    # NaN), fold A's rows have no held-out alpha there (NaN), and B's rows are
    # not scored there either; h = 1, 2 are untouched.
    p_hat, p_true, p_hat0, _, _ = _prior(1.0)
    moved = np.ones((N, H), dtype=bool)
    moved[[1, 3], 0] = False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sc = scale_corrected_error(p_hat, p_true, p_hat0, moved, EPISODE)
    assert sc.folds_available is True
    assert sc.alpha_a[0] == 1.0 and np.isnan(sc.alpha_b[0])
    assert np.isnan(sc.score_a[0]) and np.isnan(sc.score_b[0])
    assert np.isnan(sc.held_out[:, 0]).all()
    assert not sc.boundary[0]
    np.testing.assert_array_equal(sc.alpha_a[1:], np.ones(2))
    np.testing.assert_array_equal(sc.alpha_b[1:], np.ones(2))
    np.testing.assert_array_equal(sc.held_out[:, 1:], np.full((N, 2), 3.0))


@pytest.mark.parametrize("edge_fold", ["a", "b"])
def test_boundary_flags_the_top_of_the_grid_on_either_fold(edge_fold):
    # One fold predicts a quarter of the displacement: its best alpha is 4,
    # off the top of the grid, so the fit stops at 2.0 -- a boundary. The
    # other fold is perfect (alpha 1). Either fold on an edge flags the step.
    perfect = _prior(1.0)
    under = _prior(0.25)
    edge_rows = (np.arange(N) % 2 == 0) if edge_fold == "a" else (np.arange(N) % 2 == 1)
    p_hat = np.where(edge_rows[:, None, None], under[0], perfect[0])
    _, p_true, p_hat0, _, _ = perfect
    sc = scale_corrected_error(p_hat, p_true, p_hat0, ALL_MOVED, EPISODE)
    edge_alpha, other_alpha = (sc.alpha_a, sc.alpha_b) if edge_fold == "a" else (sc.alpha_b, sc.alpha_a)
    np.testing.assert_array_equal(edge_alpha, np.full(H, 2.0))
    np.testing.assert_array_equal(other_alpha, np.ones(H))
    np.testing.assert_array_equal(sc.boundary, np.ones(H, dtype=bool))
```

Twenty tests: 18 functions, two of them parametrised twice. Why each exists, against the spec's list: the clone (`ratio_probe` 0, cosine NaN, `zero_displacement` True where moved) and the perfect predictor (ratio 1, cosine 1, alpha 1 on both folds, held-out == raw error) are the pair `h×` cannot separate; the overshoot (ratio 2, cosine 1, alpha 0.5, held-out == the perfect predictor's 3.0) is the "moves the right way, too far" reading; pure noise pins the boundary flag; the attenuated prior (ratio_probe 1, ratio_raw 0.3) is the reason the probe-normalised ratio exists; the mask test pins NaN-not-zero on every channel plus the dead-probe NaN; the tie test runs the grid backwards so "first index" and "smallest alpha" differ; the fold test pins which fold's alpha scores which rows and which rows the score medians run over; the median test is the only one where mean and median disagree; the no-moved-rows-at-a-step test pins the open behaviour in the reconciliation note.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_trust.py -q -p no:cacheprovider`

Expected: the whole file fails at collection (so Task 1's tests do not run either in this step -- the import block is at module level):

```
tests/eval/test_trust.py:<line of the Task 2 import block>: in <module>
    from mbfps.eval.trust import (  # Task 2: decomposition, embedding ratio, scale correction
E   ImportError: cannot import name 'ALPHAS' from 'mbfps.eval.trust' (/Users/raphaelchen/Desktop/csgo-bot/src/mbfps/eval/trust.py)
=========================== short test summary info ============================
ERROR tests/eval/test_trust.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.07s
```

- [ ] **Step 3: Implement**

(a) The header of `src/mbfps/eval/trust.py` must import `dataclass`. Check: `grep -n '^from dataclasses import dataclass' src/mbfps/eval/trust.py`. If it prints nothing, add this line to the header's imports, directly above `import numpy as np`:

```python
from dataclasses import dataclass
```

(b) Append at the end of `src/mbfps/eval/trust.py`, after `trust_horizon`:

```python


# ---------------------------------------------------------------------------
# Displacement decomposition, embedding ratio, scale-corrected error (spec 2.2,
# items 2 and 4). Every quantity is per (row, step) on `moved` cells only; a
# cell that is not moved is NaN in every array, never 0, so a pooled mean
# cannot be diluted by windows where the agent stood still.
# ---------------------------------------------------------------------------

ALPHAS: np.ndarray = np.linspace(0.0, 2.0, 201)
"""The scale grid: 0 (the prior sits still) to 2 (twice the distance), step 0.01.

1.0 (index 100) and 0.5 (index 50) are exact grid points, so a perfect
predictor and an exact 2x overshoot fit exactly; the endpoints are the
`boundary` flag's edges.
"""


@dataclass(frozen=True)
class Decomposition:
    """`displacement_decomposition`'s per-(row, step) channels, all `(n, H)`.

    `ratio_probe` is `|d_hat| / |d_hat_real|` -- the probe's reading of the
    imagined displacement over the SAME probe's reading of the real one, so a
    flatter probe cancels; it is the ratio the readings decide on. `ratio_raw`
    (`|d_hat| / |d|`) is stored beside it as a secondary column: a probe that
    reads every displacement at 0.3 of its length reads 0.3 here and 1.0 in
    `ratio_probe`. `cosine` is the direction agreement of `d_hat` with `d`.
    `zero_displacement` marks moved cells where the prior did not move at all
    (`|d_hat| == 0`): their cosine is NaN and they are counted, not pooled.
    """

    ratio_probe: np.ndarray
    ratio_raw: np.ndarray
    cosine: np.ndarray
    moved: np.ndarray
    zero_displacement: np.ndarray


def displacement_decomposition(p_hat, p_true, p_hat0, p_true0, p_hat_real, moved) -> Decomposition:
    """Magnitude ratios and direction agreement of the imagined displacement.

    `p_hat`, `p_true`, `p_hat_real` are `(n, H, 2)`; `p_hat0`, `p_true0` are
    `(n, 2)`; `moved` is `(n, H)` bool. `d_hat = p_hat - p_hat0`,
    `d = p_true - p_true0`, `d_hat_real = p_hat_real - p_hat0` per row.
    NaN where not moved; `ratio_probe` also NaN where `|d_hat_real| == 0` (a
    probe that reads no real displacement gives no ratio, not an infinite
    one); `cosine` NaN where `|d_hat| == 0`. `ratio_raw` needs no zero guard:
    `moved` means `|d| >= MIN_MOVE > 0`.
    """
    p_hat = np.asarray(p_hat, dtype=np.float64)
    p_true = np.asarray(p_true, dtype=np.float64)
    p_hat0 = np.asarray(p_hat0, dtype=np.float64)
    p_true0 = np.asarray(p_true0, dtype=np.float64)
    p_hat_real = np.asarray(p_hat_real, dtype=np.float64)
    moved = np.asarray(moved, dtype=bool)
    d_hat = p_hat - p_hat0[:, None]
    d = p_true - p_true0[:, None]
    d_hat_real = p_hat_real - p_hat0[:, None]
    norm_hat = np.linalg.norm(d_hat, axis=-1)
    norm_true = np.linalg.norm(d, axis=-1)
    norm_real = np.linalg.norm(d_hat_real, axis=-1)
    # np.where evaluates both branches, so the divisions run on the masked-out
    # cells too; their warnings are silenced here, their values discarded.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_probe = np.where(moved & (norm_real > 0.0), norm_hat / norm_real, np.nan)
        ratio_raw = np.where(moved, norm_hat / norm_true, np.nan)
        # 0 / 0 where |d_hat| = 0 is NaN by IEEE; `zero_displacement` counts those
        cosine = np.where(moved, (d_hat * d).sum(axis=-1) / (norm_hat * norm_true), np.nan)
    return Decomposition(
        ratio_probe=ratio_probe,
        ratio_raw=ratio_raw,
        cosine=cosine,
        moved=moved,
        zero_displacement=moved & (norm_hat == 0.0),
    )


def embedding_ratio(e_hat_disp, e_true_disp, moved) -> np.ndarray:
    """`R_free`: `||e_hat(h) - e_hat(0)|| / ||e(h) - e(0)||`, from the two norms.

    Both inputs are `(n, H)` norms the diagnostics hook already reduced --
    `diagnostics.Trajectories.embedding_displacement` and
    `true_embedding_displacement`, which is why this takes the norms and not
    the four embedding arrays. NaN where not moved or where the true
    embedding did not move (`e_true_disp == 0`).
    """
    e_hat_disp = np.asarray(e_hat_disp, dtype=np.float64)
    e_true_disp = np.asarray(e_true_disp, dtype=np.float64)
    moved = np.asarray(moved, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(moved & (e_true_disp > 0.0), e_hat_disp / e_true_disp, np.nan)


@dataclass(frozen=True)
class ScaleCorrection:
    """`scale_corrected_error`'s per-step fit and per-row held-out score.

    Fold A is the rows with an even episode label, fold B the odd ones.
    `alpha_a[h]` is fit on A's moved rows at step h and scores B's
    (`score_a[h]` = the median over B's moved rows); `alpha_b` the reverse.
    `held_out[i, h]` is row i scored with the alpha fit on the OTHER fold, NaN
    where not moved. `boundary[h]` is True when either fold's alpha sits on a
    grid endpoint: at 0 the "corrected" error is persistence itself, at 2 the
    grid ran out, and neither is a correction. With fewer than two distinct
    episode labels there is nothing to hold out: every array is NaN (`boundary`
    all False) and `folds_available` is False.
    """

    alpha_a: np.ndarray
    alpha_b: np.ndarray
    score_a: np.ndarray
    score_b: np.ndarray
    held_out: np.ndarray
    boundary: np.ndarray
    folds_available: bool


def _argmin_smallest_alpha(medians: np.ndarray, alphas: np.ndarray) -> int:
    """Index of the smallest median; among ties, the index of the SMALLEST alpha.

    `np.argmin` alone returns the first index, which is the smallest alpha
    only on an ascending grid. A persistence clone ties every alpha and must
    read alpha = 0 whichever way the grid is ordered.
    """
    ties = np.flatnonzero(medians == medians.min())
    return int(ties[np.argmin(alphas[ties])])


def scale_corrected_error(p_hat, p_true, p_hat0, moved, episode, alphas=ALPHAS) -> ScaleCorrection:
    """Cross-fitted scale correction of the imagined displacement (spec 2.2, item 4).

    Per step h and fold, `alpha = argmin over alphas of median over the fold's
    MOVED rows of |p_hat0 + alpha * d_hat - p_true[:, h]|` (2-D Euclidean
    norm), and every moved row of the other fold is scored with it. The median
    is the objective on purpose: one wild row cannot drag the scale. A fold
    with no moved rows at a step cannot be fit there -- its alpha, its score
    and the other fold's held-out errors are NaN at that step, and the step is
    not a boundary. `p_hat`, `p_true` `(n, H, 2)`; `p_hat0` `(n, 2)`; `moved`
    `(n, H)` bool; `episode` `(n,)` int labels.
    """
    p_hat = np.asarray(p_hat, dtype=np.float64)
    p_true = np.asarray(p_true, dtype=np.float64)
    p_hat0 = np.asarray(p_hat0, dtype=np.float64)
    moved = np.asarray(moved, dtype=bool)
    episode = np.asarray(episode)
    alphas = np.asarray(alphas, dtype=np.float64)
    n, horizon, _ = p_hat.shape
    nan_h = np.full(horizon, np.nan)
    if len(np.unique(episode)) < 2:
        return ScaleCorrection(
            alpha_a=nan_h.copy(), alpha_b=nan_h.copy(), score_a=nan_h.copy(), score_b=nan_h.copy(),
            held_out=np.full((n, horizon), np.nan), boundary=np.zeros(horizon, dtype=bool),
            folds_available=False,
        )
    d_hat = p_hat - p_hat0[:, None]
    # candidate error for every (row, step, alpha): |p_hat0 + alpha * d_hat - p_true|, shape (n, H, A)
    corrected = p_hat0[:, None, None, :] + alphas[None, None, :, None] * d_hat[:, :, None, :]
    candidates = np.linalg.norm(corrected - p_true[:, :, None, :], axis=-1)
    fold_a = episode % 2 == 0
    folds = (fold_a, ~fold_a)
    alpha = [nan_h.copy(), nan_h.copy()]
    score = [nan_h.copy(), nan_h.copy()]
    held_out = np.full((n, horizon), np.nan)
    for h in range(horizon):
        for k, fit_rows in enumerate(folds):
            fit = fit_rows & moved[:, h]
            score_rows = folds[1 - k] & moved[:, h]
            if not fit.any():
                continue
            index = _argmin_smallest_alpha(np.median(candidates[fit, h, :], axis=0), alphas)
            alpha[k][h] = alphas[index]
            held_out[score_rows, h] = candidates[score_rows, h, index]
            if score_rows.any():
                score[k][h] = np.median(candidates[score_rows, h, index])
    edges = (alphas[0], alphas[-1])
    boundary = np.isin(alpha[0], edges) | np.isin(alpha[1], edges)
    return ScaleCorrection(
        alpha_a=alpha[0], alpha_b=alpha[1], score_a=score[0], score_b=score[1],
        held_out=held_out, boundary=boundary, folds_available=True,
    )
```

On the shipped instrument (`n = 229`, `H = 45`, 201 alphas) `corrected` is `(229, 45, 201, 2)` float64, 33 MB, and `candidates` 17 MB -- one vectorised pass per cell, no per-alpha Python loop. `np.isin` against a NaN alpha is False, which is what makes an unfit step "not a boundary" without a separate branch.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_trust.py -q -p no:cacheprovider`

Expected: PASS -- Task 1's count plus **20** (record the measured number), 0 warnings. Then once more with warnings promoted, because the full suite is read at 0 warnings and `np.where` runs the masked divisions:

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_trust.py -q -p no:cacheprovider -W error`

Expected: the same count, PASS. A `RuntimeWarning: invalid value encountered in divide` here means an `errstate` block was dropped.

- [ ] **Step 5: Mutation-test**

Set up a self-checked harness. All three checks matter here (M3b plan, Global Constraints): the package is installed editable, so a harness that copies the tree and forgets `PYTHONPATH` tests the *repo's* `trust.py` and reports every mutation caught; a same-length mutation can leave a stale `.pyc`; and an unproven harness proves nothing. Nothing is committed yet, so export by copy, not `git archive`. Save this under `scratchpad/` (gitignored, never committed), run it, and read the first three lines before the table:

```python
# scratchpad/mutate_m3d_task2.py -- do not commit
import os, shutil, subprocess, sys
from pathlib import Path

REPO = Path("/Users/raphaelchen/Desktop/csgo-bot")
S = Path(__file__).resolve().parent / "tree"            # a copy of src/ tests/ scripts/ pyproject.toml
SRC = S / "src" / "mbfps" / "eval" / "trust.py"
PY = str(REPO / ".venv" / "bin" / "python")
ENV = dict(os.environ, PYTHONPATH=str(S / "src"), PYTHONDONTWRITEBYTECODE="1")
TEST = "tests/eval/test_trust.py"


def clear_pyc():
    for d in S.rglob("__pycache__"):
        shutil.rmtree(d)


def run():
    clear_pyc()  # self-check 3: no stale bytecode can survive a same-length mutation
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", TEST],
                       cwd=S, env=ENV, capture_output=True, text=True)
    failed = sorted({l.split("::")[-1].split(" ")[0] for l in r.stdout.splitlines()
                     if l.startswith(("FAILED", "ERROR"))})
    return r.returncode, failed, r.stdout.splitlines()[-1]


# self-check 1: the copy shadows the editable install
where = subprocess.run([PY, "-c", "import mbfps.eval.trust as t; print(t.__file__)"], cwd=S, env=ENV,
                       capture_output=True, text=True).stdout.strip()
assert where == str(SRC), where
print("shadowing OK:", where)

original = SRC.read_text()
MUTATIONS = {
 # self-check 2: a known-fatal mutation FIRST; the harness must see it
 "FATAL: ALPHAS grid of 200 points (1.0 and 0.5 fall off the grid)":
   ("ALPHAS: np.ndarray = np.linspace(0.0, 2.0, 201)", "ALPHAS: np.ndarray = np.linspace(0.0, 2.0, 200)"),
 "ratio_probe divides by the true displacement, not the probe's real reading":
   ("ratio_probe = np.where(moved & (norm_real > 0.0), norm_hat / norm_real, np.nan)",
    "ratio_probe = np.where(moved & (norm_real > 0.0), norm_hat / norm_true, np.nan)"),
 "ratio_probe drops the |d_hat_real| > 0 guard (inf where the probe reads no real displacement)":
   ("ratio_probe = np.where(moved & (norm_real > 0.0), norm_hat / norm_real, np.nan)",
    "ratio_probe = np.where(moved, norm_hat / norm_real, np.nan)"),
 "ratio_probe ignores the moved mask":
   ("ratio_probe = np.where(moved & (norm_real > 0.0), norm_hat / norm_real, np.nan)",
    "ratio_probe = np.where(norm_real > 0.0, norm_hat / norm_real, np.nan)"),
 "ratio_raw ignores the moved mask":
   ("ratio_raw = np.where(moved, norm_hat / norm_true, np.nan)",
    "ratio_raw = norm_hat / norm_true"),
 "cosine ignores the moved mask":
   ("cosine = np.where(moved,", "cosine = np.where(True,"),
 "cosine normalised by |d|^2 instead of |d_hat| |d|":
   ("(d_hat * d).sum(axis=-1) / (norm_hat * norm_true)", "(d_hat * d).sum(axis=-1) / (norm_true * norm_true)"),
 "zero_displacement flags every zero d_hat, moved or not":
   ("zero_displacement=moved & (norm_hat == 0.0)", "zero_displacement=(norm_hat == 0.0)"),
 "embedding_ratio drops the e_true_disp > 0 guard":
   ("return np.where(moved & (e_true_disp > 0.0), e_hat_disp / e_true_disp, np.nan)",
    "return np.where(moved, e_hat_disp / e_true_disp, np.nan)"),
 "embedding_ratio ignores the moved mask":
   ("return np.where(moved & (e_true_disp > 0.0), e_hat_disp / e_true_disp, np.nan)",
    "return np.where(e_true_disp > 0.0, e_hat_disp / e_true_disp, np.nan)"),
 "embedding_ratio inverted (true over imagined)":
   ("e_hat_disp / e_true_disp, np.nan)", "e_true_disp / e_hat_disp, np.nan)"),
 "folds never disabled (fewer than 2 labels)":
   ("if len(np.unique(episode)) < 2:", "if len(np.unique(episode)) < 1:"),
 "fold A = odd labels":
   ("fold_a = episode % 2 == 0", "fold_a = episode % 2 == 1"),
 "argmin by first index, not smallest alpha":
   ("return int(ties[np.argmin(alphas[ties])])", "return int(ties[0])"),
 "fit objective is the mean, not the median":
   ("index = _argmin_smallest_alpha(np.median(candidates[fit, h, :], axis=0), alphas)",
    "index = _argmin_smallest_alpha(np.mean(candidates[fit, h, :], axis=0), alphas)"),
 "fit includes not-moved rows":
   ("fit = fit_rows & moved[:, h]", "fit = fit_rows"),
 "held-out scores the fitting fold with its own alpha":
   ("score_rows = folds[1 - k] & moved[:, h]", "score_rows = fit_rows & moved[:, h]"),
 "held-out written for not-moved rows too":
   ("score_rows = folds[1 - k] & moved[:, h]", "score_rows = folds[1 - k]"),
 "empty-fit guard dropped":
   ("            if not fit.any():\n                continue\n", ""),
 "scoring-fold median taken over the fitting fold":
   ("score[k][h] = np.median(candidates[score_rows, h, index])", "score[k][h] = np.median(candidates[fit, h, index])"),
 "correction scales the position, not the displacement":
   ("corrected = p_hat0[:, None, None, :] + alphas[None, None, :, None] * d_hat[:, :, None, :]",
    "corrected = alphas[None, None, :, None] * p_hat[:, :, None, :]"),
 "boundary reads fold A only":
   ("boundary = np.isin(alpha[0], edges) | np.isin(alpha[1], edges)", "boundary = np.isin(alpha[0], edges)"),
 "boundary reads fold B only":
   ("boundary = np.isin(alpha[0], edges) | np.isin(alpha[1], edges)", "boundary = np.isin(alpha[1], edges)"),
 "boundary is the lower edge only":
   ("edges = (alphas[0], alphas[-1])", "edges = (alphas[0],)"),
 "boundary is the upper edge only":
   ("edges = (alphas[0], alphas[-1])", "edges = (alphas[-1],)"),
 "folds_available reported False on the success path":
   ("held_out=held_out, boundary=boundary, folds_available=True,", "held_out=held_out, boundary=boundary, folds_available=False,"),
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
        print(f"{'CAUGHT' if rc else 'SURVIVED':8} | {name} | {len(failed)} | {failed[:4]}")
        if name.startswith("FATAL") and rc == 0:
            sys.exit("harness cannot detect a fatal mutation")
finally:
    SRC.write_text(original)
    clear_pyc()   # self-check 3: never leave bytecode from a mutated source behind
```

Build the copy with `rm -rf scratchpad/tree && mkdir -p scratchpad/tree && cp -R src tests scripts pyproject.toml scratchpad/tree/` and run `.venv/bin/python scratchpad/mutate_m3d_task2.py`. The first two lines must read `shadowing OK: .../scratchpad/tree/src/mbfps/eval/trust.py` and `baseline: <Task 1's count + 20> passed`; every row must read `CAUGHT`. The `assert original.count(old) == 1` line is the harness's own guard that each mutation target is unique in the file -- the Task 1 half of `trust.py` shares none of these strings, and a count of 0 or 2 stops the run rather than silently mutating nothing.

| mutation | must be caught by |
|---|---|
| `ALPHAS = np.linspace(0.0, 2.0, 200)` (harness self-check, fatal: 1.0 and 0.5 leave the grid) | 9 tests, from `test_alphas_is_the_pinned_grid` and every `alpha == 1.0` / `0.5` assertion |
| `ratio_probe` divides by `\|d\|` instead of `\|d_hat_real\|` | `test_probe_attenuation_cancels_in_ratio_probe_but_not_in_ratio_raw` (the one test where the two denominators differ) |
| `ratio_probe` drops the `norm_real > 0` guard (5 / 0 = inf, not NaN) | `test_decomposition_respects_the_moved_mask_and_a_dead_probe_reading` |
| `ratio_probe` ignores `moved` | `test_decomposition_respects_the_moved_mask_and_a_dead_probe_reading` |
| `ratio_raw` ignores `moved` | `test_decomposition_respects_the_moved_mask_and_a_dead_probe_reading` |
| `cosine` ignores `moved` | `test_decomposition_respects_the_moved_mask_and_a_dead_probe_reading` |
| `cosine` normalised by `\|d\|^2` instead of `\|d_hat\| \|d\|` | `test_exact_overshoot_reads_ratio_two_cosine_one` (reads 2), `test_persistence_clone_...` (reads 0, not NaN), `test_probe_attenuation_...` (reads 0.3) |
| `zero_displacement` without `moved &` | `test_decomposition_respects_the_moved_mask_and_a_dead_probe_reading` (row 1's zero `d_hat` is not moved) |
| `embedding_ratio` drops the `e_true_disp > 0` guard (7 / 0 = inf) | `test_embedding_ratio_is_nan_where_not_moved_or_where_the_truth_did_not_move` |
| `embedding_ratio` ignores `moved` | `test_embedding_ratio_is_nan_where_not_moved_or_where_the_truth_did_not_move` |
| `embedding_ratio` inverted (true over imagined) | `test_embedding_ratio_is_the_imagined_over_the_true_norm`, `..._is_nan_where_...` |
| `< 2` distinct labels becomes `< 1` (folds never disabled) | `test_fewer_than_two_episode_labels_disables_both_folds` |
| fold A = odd labels | `test_folds_are_even_and_odd_episode_labels_and_each_scores_the_other` (`alpha_a` reads 0.5), `test_boundary_flags_the_top_of_the_grid_on_either_fold[a]`, `[b]`, `test_a_fold_with_no_moved_rows_at_a_step_is_nan_at_that_step_only` |
| tie broken by first index (`ties[0]`) instead of smallest alpha | `test_persistence_clone_ties_every_alpha_and_the_smallest_alpha_wins[descending]` (reads 2.0) -- the ascending case passes under this mutation, which is why the grid runs backwards |
| fit objective `np.mean` instead of `np.median` | `test_the_fit_is_a_median_so_a_minority_outlier_row_does_not_move_alpha` (the only fixture where they disagree) |
| fit includes not-moved rows (`fit = fit_rows`) | `test_scale_correction_fits_and_scores_only_moved_rows` (`alpha_a` leaves 1), `test_a_fold_with_no_moved_rows_at_a_step_...` (`alpha_b[0]` is a number) |
| held-out scored with the row's OWN fold's alpha | `test_folds_are_even_and_odd_episode_labels_and_each_scores_the_other` (fold A rows read 3.0, not `sqrt(9 + \|d\|^2 / 4)`), `test_a_fold_with_no_moved_rows_at_a_step_...` |
| held-out written for not-moved rows | `test_scale_correction_fits_and_scores_only_moved_rows` (row 2 is scored), `test_a_fold_with_no_moved_rows_at_a_step_...` |
| empty-fit `continue` dropped (`np.median` of no rows) | `test_a_fold_with_no_moved_rows_at_a_step_is_nan_at_that_step_only` (`ValueError` from `argmin` of an empty tie set) |
| scoring-fold median taken over the fitting fold | `test_folds_are_even_and_odd_episode_labels_and_each_scores_the_other` (`score_a` reads 3.0) |
| correction scales the position (`alpha * p_hat`) instead of the displacement | 11 tests, from `test_perfect_predictor_fits_alpha_one_...` and `test_exact_overshoot_fits_alpha_half_...` |
| `boundary` reads fold A only | `test_boundary_flags_the_top_of_the_grid_on_either_fold[b]` |
| `boundary` reads fold B only | `test_boundary_flags_the_top_of_the_grid_on_either_fold[a]` |
| `boundary` checks the lower edge only | `test_boundary_flags_the_top_of_the_grid_on_either_fold[a]`, `[b]`, `test_persistence_clone_ties_...[descending]` (0.0 is `alphas[-1]` there) |
| `boundary` checks the upper edge only | `test_pure_noise_displacement_is_a_boundary_on_both_folds`, `test_persistence_clone_ties_...[ascending]` |
| `folds_available=False` on the success path | `test_perfect_predictor_fits_alpha_one_...`, `test_pure_noise_...`, `test_a_fold_with_no_moved_rows_...` (`is True`) |

All twenty-six rows were run against this exact test file and this exact implementation on 2026-09-13 and every one was caught; the catchers are what actually fired. Any mutation that survives on your run is a missing test. Add it before committing. Then `rm -rf scratchpad/tree` and `git status --short` must show only `src/mbfps/eval/trust.py` and `tests/eval/test_trust.py` modified (plus the untracked `study.log` already there).

- [ ] **Step 6: Run the full suite**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`

Expected: PASS, 0 warnings, **20 more than the count Task 1's full-suite step left** (record the measured number); nothing outside `tests/eval/test_trust.py` changes count. About 6 minutes.

- [ ] **Step 7: Commit**

```bash
git add src/mbfps/eval/trust.py tests/eval/test_trust.py
git commit -m "feat: displacement decomposition, embedding ratio and cross-fitted scale correction in trust.py

The three per-window quantities that separate slow drift from wrong
direction (M3d spec 2.2, items 2 and 4): the probe-normalised and raw
magnitude ratios, the direction cosine, the probe-free embedding ratio,
and the two-fold scale correction with its grid-boundary flag. NaN, never
0, off the moved mask; ties in the alpha argmin go to the smallest alpha;
a fold with no moved rows at a step is NaN at that step. Pinned by the
synthetic priors the spec names -- clone, perfect, 2x overshoot, pure
noise, probe-attenuated -- each with its answer written by hand.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: diagnostics.py: keep_trajectories on _diagnose, nine _Pass fields, reference_trajectories

The ladder stores mean error curves, and a mean curve cannot tell an imagination that drifts slowly from one that moves the wrong way (spec §1). The trust horizon needs the per-window rows those curves are means of -- `p̂(h)`, `p̂(0)`, `p̂_real(h)`, `p(h)`, `p(0)` -- and four embedding-space series that read the imagination against the TRUTH: `D̂ = ‖ê(h) − e(h)‖`, `D₀ = ‖ê(0) − e(h)‖`, `‖ê(h) − ê(0)‖`, `‖e(h) − e(0)‖`, where `e(h)` is the encoder's own embedding of the frame the h-th action produced, `handle.embeddings[0, context + h − 1]`, and `e(0) = handle.embeddings[0, context − 1]` (spec §2.1). These four are **new**: every embedding channel the ladder reads today is imagination-vs-imagination (an intervened draw against the canonical one, or the canonical one against a second draw); nothing in the module measures imagination-vs-truth. This task adds one flag, `keep_trajectories`, to `_diagnose`, nine optional fields on `_Pass` that the flag fills for the canonical pass only, and the public `reference_trajectories(...)`, which runs `_diagnose` with no intervention arm and the noise reference drawn -- the ladder's stream sequence -- and repackages the rows as a frozen `Trajectories` together with the pass's own two curves, so the trust script (Task 4) can demand `max |Δ| == 0.0` against the stored diagnostic before it reads anything.

Two properties carry the whole task and both are invisible to shapes. With the flag False the pass must be **byte-identical** to today's -- the ladder and the sweep never pass the flag, and their bitwise pins against the nine shipped records are the strongest correctness evidence in the repo -- so the flag draws nothing from the sampling stream and does its extra work in numpy on rows already in hand. With the flag True every row must come from the **same pipeline and the same frame** as the curve it explains: `positions_at_context` is row 0 of the persistence prediction the curve was scored on (not a fresh one-row `apply_probe`, which a different matmul shape is not promised to reproduce to the last bit), `true_positions` is the first two columns of the very `truth` slice, and `e(h)` is the encoder row for frame `start + context + h`, one frame after the last context frame -- the same off-by-one `evaluate_rollout`'s alignment test guards. The oracle rigs of `test_rollout.py` give every one of the nine arrays a closed form, and the real sampling rig pins the bitwise identities the self-check depends on.

**Depends on nothing from Tasks 1-2.** `trust.py` is consumed by Task 4, not here; this task touches `diagnostics.py` and its test file only, and can be executed with or without Tasks 1-2 landed.

**Files:**
- Modify: `src/mbfps/eval/diagnostics.py` -- the module docstring (one paragraph appended before its closing `"""` at line 165); `_Pass` (lines 392-424: nine `Optional` fields with `= None` defaults appended after `window_episode`'s docstring at line 424, and a new module tuple `_TRAJECTORY_FIELDS` after the class); `_diagnose` -- the signature (the flag after `noise_reference` at line 439), the docstring (a paragraph before "Deliberately NOT separately decorated" at line 466), the collectors (one line after `window_episode: list[int] = []` at line 493), the per-window block (inserted after the six reference rows, i.e. after line 596 and before `arm_embeddings = {}` at line 598), and the return (one `**` line after `window_episode=...` at line 634); a new section with `Trajectories` and `reference_trajectories` inserted after `_diagnose`'s closing `)` at line 635 and before the `# Diagnostic 1` banner at line 638
- Test: `tests/eval/test_diagnostics.py` -- the `dataclasses` import (line 28); the `mbfps.eval.diagnostics` import block (lines 38-46) gains `Trajectories` and `reference_trajectories`; a new `from mbfps.eval.probe import apply_probe` after it; the `tests.eval.test_rollout` import block (lines 49-61) gains `DX` and `DY`; a new "reference pass's trajectories" section appended at the END of the file, after `test_the_ladder_on_a_shipped_checkpoint_leaves_the_shuffled_rung_bitwise_the_record` (line 3051)

> No other file constructs `_Pass` or calls `_diagnose` (verified: `grep -rn "_Pass\|_diagnose" scripts tests src` finds only docstring mentions outside `diagnostics.py`), so the nine defaulted fields and the keyword-only flag change no caller. `scripts/diagnose_dynamics.py` reaches `_diagnose` only through `action_intervention_ladder` and `regrounding_sweep`, neither of which passes the flag.

**Interfaces:**
- Consumes (existing, unchanged): `mbfps.eval.diagnostics._diagnose(model, val_paths, embedding_probe_weights, *, arms, context, horizon, seed, device, feature_backbone, arm_pairs=None, noise_reference=True) -> _Pass`; `_Pass.reference: RolloutResult`, `_Pass.window_episode: np.ndarray`, `_Pass.windows_total: int`; `_embedding_distance(a, b) -> np.ndarray` (float64 L2 per row); `mbfps.eval.probe.apply_probe(probe, latents)`, `probe_targets(privileged, keys)`; `mbfps.eval.rollout.evaluate_rollout` (the independent pin). From `tests/eval/test_rollout.py`: `OracleModel`, `DriftingModel`, `synthetic_episode`, `oracle_probe`, `real_model_and_probe`, `CONTEXT = 3`, `HORIZON = 5`, `DX = 10.0`, `DY = -3.0`, `STEP`, `T_SYNTHETIC = 20`.
- Produces (verbatim from the contract, for Task 4's `scripts/trust_horizon.py`):

```python
def _diagnose(..., keep_trajectories: bool = False) -> _Pass   # existing signature plus this keyword-only flag

# nine new Optional fields on _Pass, default None, populated ONLY when keep_trajectories=True, reference arm only:
positions: np.ndarray | None                       # (n_windows, horizon, 2)   p_hat(h)
positions_at_context: np.ndarray | None            # (n_windows, 2)            p_hat(0)   (== rollout.py's pers_pred anchor)
positions_real: np.ndarray | None                  # (n_windows, horizon, 2)   p_hat_real(h): probe of floor_embeddings
true_positions: np.ndarray | None                  # (n_windows, horizon, 2)   p(h) = privileged (pos_x, pos_y) at frame start+context+h
true_at_context: np.ndarray | None                 # (n_windows, 2)            p(0) at frame start+context
embedding_distance_to_truth: np.ndarray | None     # (n_windows, horizon)      ||e_hat(h) - e(h)||,  e(h) = handle.embeddings[0, context + h - 1]
embedding_persistence_distance: np.ndarray | None  # (n_windows, horizon)      ||e_hat(0) - e(h)||,  e_hat(0) = last_context_embedding
embedding_displacement: np.ndarray | None          # (n_windows, horizon)      ||e_hat(h) - e_hat(0)||
true_embedding_displacement: np.ndarray | None     # (n_windows, horizon)      ||e(h) - e(0)||,  e(0) = handle.embeddings[0, context - 1]

@dataclass(frozen=True)
class Trajectories:
    # the nine arrays above, same names and shapes, all present (not Optional), plus:
    window_episode: np.ndarray     # (n_windows,) int -- the existing _Pass.window_episode
    windows_total: int
    reference_position: np.ndarray    # (horizon,) mean over windows of |p_hat(h) - p(h)|   -- for the self-check
    persistence_position: np.ndarray  # (horizon,) mean over windows of |p_hat(0) - p(h)|   -- for the self-check

def reference_trajectories(model, val_paths, embedding_probe_weights: dict, *, context: int, horizon: int,
                           seed: int, device, feature_backbone) -> Trajectories
    # calls _diagnose(model, val_paths, embedding_probe_weights, arms={}, context=context, horizon=horizon, seed=seed,
    #                 device=device, feature_backbone=feature_backbone, noise_reference=True, keep_trajectories=True)
    # and repackages. Decorated @torch.no_grad() like the other public entry points.
```

  Every array is `float64` (the probe's output dtype, the truth's, and `_embedding_distance`'s). `reference_position` and `persistence_position` are the pass's own `reference.rssm_position` and `reference.persistence_position` -- `np.stack(rows).mean(axis=0)` over the per-window rows -- and this task's tests pin that `np.linalg.norm(positions - true_positions, axis=-1).mean(axis=0)` and `np.linalg.norm(positions_at_context[:, None, :] - true_positions, axis=-1).mean(axis=0)` reproduce them **bitwise** on the real sampling rig, so Task 4's `self_check` can compute exactly those two expressions and demand `max |Δ| == 0.0`. One private helper is also added and is NOT part of the contract: `_TRAJECTORY_FIELDS: tuple[str, ...]`, the nine field names in `_Pass` order, the one list `_diagnose` collects by and `reference_trajectories` repackages by.

- [ ] **Step 1: Write the failing tests**

Three edits to the import block of `tests/eval/test_diagnostics.py` first, then the new section appended at the very end of the file.

(a) Line 28 -- `FrozenInstanceError`, for the frozen-dataclass assertion:

```python
from dataclasses import FrozenInstanceError, replace
```

(b) Lines 38-47 -- the diagnostics import gains the two new public names, and `apply_probe` is imported for the hand recomputation (the block stays alphabetical; `apply_probe` sits on its own line between the diagnostics and rollout imports):

```python
from mbfps.eval.diagnostics import (
    LADDER,
    LADDER_PERTURBS,
    REGROUNDING_KS,
    RegroundingSweep,
    Trajectories,
    action_intervention_ladder,
    action_shuffled_rollout,
    reference_trajectories,
    regrounding_sweep,
)
from mbfps.eval.probe import apply_probe
from mbfps.eval.rollout import RolloutResult, evaluate_rollout
```

(c) Lines 49-61 -- the rollout-rig import gains `DX` and `DY`, the synthetic agent's per-frame displacement, which the closed forms below are written in:

```python
from tests.eval.test_rollout import (
    CONTEXT,
    DX,
    DY,
    HORIZON,
    STEP,
    T_SYNTHETIC,
    DriftingModel,
    OracleModel,
    _OracleRSSM,
    _StatefulRSSM,
    oracle_probe,
    real_model_and_probe,
    synthetic_episode,
)
```

(d) Append this section at the END of the file, after `test_the_ladder_on_a_shipped_checkpoint_leaves_the_shuffled_rung_bitwise_the_record` (after line 3051). It uses `write`, `DEVICES`, `_two_episode_rig`, `_imagined_actions`, `_per_rung`, `_tee`, `varied_action_episode` and `rollout` from earlier in the file, unchanged:

```python
# ---------------------------------------------------------------------------
# The reference pass's trajectories (M3d): `keep_trajectories` on `_diagnose`
# and `reference_trajectories`. The ladder stores mean error curves; the trust
# horizon needs the per-window rows those curves are means of, plus four
# embedding-space series that read the imagination against the TRUTH --
# something no ladder channel does. Two things can go wrong without breaking
# a shape: the flag can move a curve the ladder pins bitwise, and a row can
# be taken one frame, one pipeline, or one anchor away from where the spec
# puts it. The oracle rigs give every row a closed form; the real sampling
# rig pins the bitwise identities the trust pass's self-check depends on.
# ---------------------------------------------------------------------------

TRAJECTORY_FIELDS = (
    "positions", "positions_at_context", "positions_real",
    "true_positions", "true_at_context",
    "embedding_distance_to_truth", "embedding_persistence_distance",
    "embedding_displacement", "true_embedding_displacement",
)
"""Literal, deliberately NOT `diagnostics_module._TRAJECTORY_FIELDS`: a field
dropped from the module's tuple must be a missing attribute here, not a
shorter loop."""


def trajectories(model, paths, probe, *, device=None, seed=0):
    return reference_trajectories(
        model, paths, probe, context=CONTEXT, horizon=HORIZON, seed=seed,
        device=device or torch.device("cpu"), feature_backbone=None,
    )


def reference_pass(model, paths, probe, *, keep, device=None, seed=0):
    """`_diagnose` as `reference_trajectories` calls it, with the flag chosen.

    Under `no_grad` because `_diagnose` is deliberately undecorated (its
    docstring says why) and a real model's encoder output carries grad."""
    with torch.no_grad():
        return diagnostics_module._diagnose(
            model, paths, probe, arms={}, context=CONTEXT, horizon=HORIZON,
            seed=seed, device=device or torch.device("cpu"), feature_backbone=None,
            noise_reference=True, keep_trajectories=keep,
        )


def _three_length_oracle(tmp_path):
    """Four windows from two contributing episodes, at the lengths the window
    rule is sensitive to: `need` (no window), `need + 1` (one), `3 * need`
    (three). `(episodes, paths, probe, expected (episode, start) per window)`."""
    need = CONTEXT + HORIZON
    episodes = [synthetic_episode(length=n) for n in (need, need + 1, 3 * need)]
    paths = [write(tmp_path, episode, index) for index, episode in enumerate(episodes)]
    cut = [
        (episode, start)
        for episode in episodes
        for start in window_starts(episode.length, CONTEXT, HORIZON)
    ]
    assert len(cut) == 4, [len(window_starts(e.length, CONTEXT, HORIZON)) for e in episodes]
    return episodes, paths, oracle_probe(episodes[-1]), cut


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_without_the_flag_the_pass_carries_no_trajectories_and_is_bitwise_what_it_was(
    tmp_path, device
):
    """The ladder's pin, from the other side: the flag is the ONLY difference
    between two passes on the real sampling rig, and every field the ladder
    reads must be bitwise the same across them -- the six curves, the six
    per-window matrices, the noise reference and its two self-checks, the
    counts and the labels -- and the generator must end at the same state,
    so the flag drew nothing from the stream. Without the flag all nine
    trajectory fields are None; with it none of them is."""
    model, paths, probe = _two_episode_rig(tmp_path, device)

    plain = reference_pass(model, paths, probe, keep=False, device=device)
    left_by_plain = diagnostics_module._rng_snapshot(device)
    kept = reference_pass(model, paths, probe, keep=True, device=device)
    left_by_kept = diagnostics_module._rng_snapshot(device)

    for name in TRAJECTORY_FIELDS:
        assert getattr(plain, name) is None, name
        assert getattr(kept, name) is not None, name
    for name in ("horizon", "rssm_position", "persistence_position", "floor_position",
                 "rssm_angle", "persistence_angle", "floor_angle"):
        np.testing.assert_array_equal(
            getattr(kept.reference, name), getattr(plain.reference, name), err_msg=name
        )
    assert kept.reference_windows.keys() == plain.reference_windows.keys()
    for name, rows in plain.reference_windows.items():
        np.testing.assert_array_equal(kept.reference_windows[name], rows, err_msg=name)
    np.testing.assert_array_equal(kept.noise_embedding, plain.noise_embedding)
    np.testing.assert_array_equal(kept.noise_bitwise_real, plain.noise_bitwise_real)
    np.testing.assert_array_equal(kept.noise_stream_restored, plain.noise_stream_restored)
    assert kept.windows_total == plain.windows_total == 4
    np.testing.assert_array_equal(kept.window_episode, plain.window_episode)
    assert kept.arms == plain.arms == {} and kept.arm_pairs == plain.arm_pairs == {}
    assert set(left_by_kept) == set(left_by_plain)
    for key in left_by_plain:
        assert torch.equal(left_by_kept[key], left_by_plain[key]), key


def test_the_flag_is_keyword_only_and_off_by_default():
    """Off by default so that the ladder and the sweep -- which never pass it
    -- keep running the pass they pin bitwise against the records."""
    parameter = inspect.signature(diagnostics_module._diagnose).parameters["keep_trajectories"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_the_flagged_pass_and_the_trajectories_carry_the_nine_arrays_at_their_shapes(
    tmp_path, device
):
    """Four windows, five horizon steps, two map coordinates: the per-step
    arrays are `(4, 5, 2)` or `(4, 5)`, the two context anchors `(4, 2)`, all
    float64 like the curves they reduce to. `Trajectories` carries the same
    nine under the same names, with the pass's counts and labels, and is
    frozen -- a consumer cannot quietly overwrite a row it then reads."""
    model, paths, probe = _two_episode_rig(tmp_path, device)
    expected = {
        "positions": (4, HORIZON, 2), "positions_at_context": (4, 2),
        "positions_real": (4, HORIZON, 2), "true_positions": (4, HORIZON, 2),
        "true_at_context": (4, 2), "embedding_distance_to_truth": (4, HORIZON),
        "embedding_persistence_distance": (4, HORIZON),
        "embedding_displacement": (4, HORIZON),
        "true_embedding_displacement": (4, HORIZON),
    }
    assert set(expected) == set(TRAJECTORY_FIELDS)

    kept = reference_pass(model, paths, probe, keep=True, device=device)
    result = trajectories(model, paths, probe, device=device)

    assert isinstance(result, Trajectories)
    for name, shape in expected.items():
        assert getattr(kept, name).shape == shape, name
        assert getattr(kept, name).dtype == np.float64, name
        assert getattr(result, name).shape == shape, name
        assert getattr(result, name).dtype == np.float64, name
    assert result.windows_total == 4
    np.testing.assert_array_equal(result.window_episode, np.array([0, 0, 1, 1]))
    assert result.reference_position.shape == result.persistence_position.shape == (HORIZON,)
    with pytest.raises(FrozenInstanceError):
        result.windows_total = 0


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_the_trajectories_curves_are_the_passes_own_and_the_rollouts_bitwise(
    tmp_path, device
):
    """The two curves the trust pass checks itself against. Bitwise the
    flagged pass's `reference.rssm_position` / `reference.persistence_position`
    AND bitwise `evaluate_rollout`'s on the same rig and seed -- the second
    is the independent one: it is the function the shipped records were
    written by, and a pass whose stream had drifted (an arm left in, the
    noise reference skipped) would still equal itself."""
    model, paths, probe = _two_episode_rig(tmp_path, device)
    rollout_result = rollout(model, paths, probe, device=device)
    kept = reference_pass(model, paths, probe, keep=True, device=device)
    result = trajectories(model, paths, probe, device=device)

    np.testing.assert_array_equal(result.reference_position, kept.reference.rssm_position)
    np.testing.assert_array_equal(result.persistence_position, kept.reference.persistence_position)
    np.testing.assert_array_equal(result.reference_position, rollout_result.rssm_position)
    np.testing.assert_array_equal(result.persistence_position, rollout_result.persistence_position)
    assert result.reference_position.min() > 0.0


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_the_per_window_rows_reduce_to_the_curves_bitwise(tmp_path, device):
    """THE identity the self-check in `scripts/trust_horizon.py` demands with
    max |delta| == 0.0: the Euclidean distance of `positions` to
    `true_positions`, meaned over windows, IS `reference_position`; the
    distance of the held `positions_at_context` to `true_positions`, meaned,
    IS `persistence_position`. Bitwise, on the real sampling rig, so the
    anchor must be the very rows the persistence curve was scored on -- a
    second `apply_probe` over a different shape is not promised the same
    last bit -- and the reduction must be `np.stack(rows).mean(axis=0)`."""
    model, paths, probe = _two_episode_rig(tmp_path, device)
    result = trajectories(model, paths, probe, device=device)

    model_error = np.linalg.norm(result.positions - result.true_positions, axis=-1)
    persistence_error = np.linalg.norm(
        result.positions_at_context[:, None, :] - result.true_positions, axis=-1
    )
    assert model_error.shape == persistence_error.shape == (4, HORIZON)
    np.testing.assert_array_equal(model_error.mean(axis=0), result.reference_position)
    np.testing.assert_array_equal(persistence_error.mean(axis=0), result.persistence_position)


def test_true_positions_are_the_privileged_positions_at_the_frames_the_actions_produced(
    tmp_path
):
    """The truth slice, element for element, per window: `true_positions[w, h-1]`
    is `privileged[start + context + h, (pos_x, pos_y)]` -- the frame the h-th
    horizon action PRODUCED, one after the last context frame -- and
    `true_at_context[w]` is `privileged[start + context]`. Indexed by hand
    from `window_starts` over three episode lengths, and checked bitwise
    against the closed form `(DX * t, DY * t)` as well, so a slice off by one
    frame, or an anchor taken at the first horizon frame, reads as a whole
    step of displacement rather than as noise."""
    _, paths, probe, cut = _three_length_oracle(tmp_path)
    result = trajectories(OracleModel(), paths, probe)

    assert result.windows_total == 4
    np.testing.assert_array_equal(result.window_episode, np.array([0, 1, 1, 1]))
    for w, (episode, start) in enumerate(cut):
        for h in range(1, HORIZON + 1):
            frame = start + CONTEXT + h
            np.testing.assert_array_equal(
                result.true_positions[w, h - 1],
                episode.privileged[frame, 1:3].astype(np.float64),
                err_msg=f"window {w} step {h}",
            )
            np.testing.assert_array_equal(
                result.true_positions[w, h - 1], np.array([DX * frame, DY * frame])
            )
        anchor = start + CONTEXT
        np.testing.assert_array_equal(
            result.true_at_context[w], episode.privileged[anchor, 1:3].astype(np.float64)
        )
        np.testing.assert_array_equal(
            result.true_at_context[w], np.array([DX * anchor, DY * anchor])
        )


def test_the_probe_channel_reads_the_imagination_the_floor_and_the_last_context_frame(
    tmp_path
):
    """Three probe readings, three different sources, told apart on a rig
    whose dynamics DRIFT: `DriftingModel` advances two frames per action
    while its filter is exact. So p_hat(h) is the position at frame
    `start + context + 2h`, p_hat_real(h) -- the floor, a posterior over the
    real frames -- is the position at `start + context + h`, and p_hat(0) is
    the position at `start + context`. Each is the closed form `(DX * t,
    DY * t)` through a probe exact to 1e-6, so the imagination taken from
    the floor, the floor taken from the imagination, or the anchor taken from
    step 1 is off by whole steps of displacement."""
    _, paths, probe, cut = _three_length_oracle(tmp_path)
    result = trajectories(DriftingModel(), paths, probe)

    steps = np.arange(1, HORIZON + 1)
    for w, (_, start) in enumerate(cut):
        imagined_frames = start + CONTEXT + 2 * steps
        real_frames = start + CONTEXT + steps
        np.testing.assert_allclose(
            result.positions[w], np.stack([DX * imagined_frames, DY * imagined_frames], axis=1),
            atol=1e-6, err_msg=f"window {w}: positions",
        )
        np.testing.assert_allclose(
            result.positions_real[w], np.stack([DX * real_frames, DY * real_frames], axis=1),
            atol=1e-6, err_msg=f"window {w}: positions_real",
        )
        np.testing.assert_allclose(
            result.positions_at_context[w],
            np.array([DX * (start + CONTEXT), DY * (start + CONTEXT)]),
            atol=1e-6, err_msg=f"window {w}: positions_at_context",
        )
    # The drift is a whole frame per step, so the imagination and the floor
    # are STEP apart at h = 1 -- the readings cannot be the same source.
    assert np.abs(result.positions - result.positions_real).max() > STEP / 2


@pytest.mark.parametrize(
    "model, drift",
    [(OracleModel(), 1), (DriftingModel(), 2)],
    ids=["exact", "drifting"],
)
def test_the_embedding_channel_has_the_closed_form_of_the_frame_tags(tmp_path, model, drift):
    """The four series against the truth in embedding space, on rigs where
    every embedding is a frame tag. The encoder tags frame t with `[t]`, the
    head returns the tag, and `imagine` advances `drift` tags per step, so
    with e(h) the tag of frame `start + context + h`:

        e_hat(h) - e(h)     = (drift - 1) * h      -> distance_to_truth
        e_hat(0) - e(h)     = -h                   -> persistence_distance
        e_hat(h) - e_hat(0) = drift * h            -> displacement
        e(h) - e(0)         = h                    -> true_displacement

    Integers in float32 are exact, so these are bitwise. The exact rig gives
    a distance to truth of 0 at every step -- which is what separates e(h)
    from e(h + 1): one frame late reads 1 there, not 0 -- and the drifting
    rig separates persistence from distance-to-truth (equal on the exact
    rig) and the imagined displacement from the true one."""
    _, paths, probe, cut = _three_length_oracle(tmp_path)
    result = trajectories(model, paths, probe)

    h = np.arange(1, HORIZON + 1, dtype=np.float64)
    for w in range(len(cut)):
        np.testing.assert_array_equal(
            result.embedding_distance_to_truth[w], (drift - 1) * h, err_msg=f"window {w}"
        )
        np.testing.assert_array_equal(
            result.embedding_persistence_distance[w], h, err_msg=f"window {w}"
        )
        np.testing.assert_array_equal(
            result.embedding_displacement[w], drift * h, err_msg=f"window {w}"
        )
        np.testing.assert_array_equal(
            result.true_embedding_displacement[w], h, err_msg=f"window {w}"
        )


def test_every_row_is_recomputed_by_hand_from_the_encoder_the_head_and_the_probe(
    tmp_path, monkeypatch
):
    """The single-pipeline rule, on the real sampling rig, where nothing has
    a closed form: every one of the nine arrays is rebuilt in the test from
    what the model actually emitted, and must match bitwise.

    Spied per window: the encoder's output (the window's `need` rows, from
    which e(0) is row `context - 1` and e(1..H) the rows after it), the
    context filter's and the floor's `observe` outputs (call order: filter,
    floor -- stride 2), and the canonical `imagine` output (call order:
    canonical, noise reference -- stride 2). The head is then applied BY THE
    TEST to those latents, the probe by the test to those rows, and the
    norms taken in float64 -- so a series read off the raw latent instead of
    the head, the noise draw instead of the canonical one, the imagination's
    rows for the floor's, or a norm taken in float32, all differ."""
    paths = [write(tmp_path, synthetic_episode())]
    model, probe = real_model_and_probe()
    encoded: list[np.ndarray] = []
    observed: list[torch.Tensor] = []
    imagined: list[torch.Tensor] = []
    real_encoder = model.encoder.forward
    real_observe = model.rssm.observe
    real_imagine = model.rssm.imagine
    monkeypatch.setattr(
        model.encoder, "forward",
        lambda obs: _tee(encoded, real_encoder(obs)),
    )

    def spy_observe(embeddings, actions, state=None):
        out = real_observe(embeddings, actions, state=state)
        observed.append(out["latent"].clone())
        return out

    def spy_imagine(actions, state):
        out = real_imagine(actions, state)
        imagined.append(out["latent"].clone())
        return out

    monkeypatch.setattr(model.rssm, "observe", spy_observe)
    monkeypatch.setattr(model.rssm, "imagine", spy_imagine)
    result = trajectories(model, paths, probe)

    assert result.windows_total == 2
    assert len(encoded) == 2 and len(observed) == 4 and len(imagined) == 4
    with torch.no_grad():
        for w in range(2):
            embeddings = encoded[w].detach().cpu().numpy()          # (need, E), float32
            assert embeddings.shape == (CONTEXT + HORIZON, 2048)
            e0, e = embeddings[CONTEXT - 1], embeddings[CONTEXT:]   # e(0), e(1..H)
            filter_latent, floor_latent = observed[2 * w], observed[2 * w + 1]
            canonical_latent = imagined[2 * w]
            e_hat = model.heads(canonical_latent)["embedding"][0].cpu().numpy()
            e_hat0 = model.heads(filter_latent[:, -1:])["embedding"][0, 0].cpu().numpy()
            e_real = model.heads(floor_latent)["embedding"][0].cpu().numpy()
            assert e_hat.shape == e_real.shape == (HORIZON, 2048) and e_hat0.shape == (2048,)

            np.testing.assert_array_equal(result.positions[w], apply_probe(probe, e_hat)[:, :2])
            np.testing.assert_array_equal(
                result.positions_real[w], apply_probe(probe, e_real)[:, :2]
            )
            np.testing.assert_array_equal(
                result.positions_at_context[w],
                apply_probe(probe, np.repeat(e_hat0[None, :], HORIZON, axis=0))[0, :2],
            )
            f64 = lambda x: np.asarray(x, dtype=np.float64)  # noqa: E731
            np.testing.assert_array_equal(
                result.embedding_distance_to_truth[w],
                np.linalg.norm(f64(e_hat) - f64(e), axis=-1),
            )
            np.testing.assert_array_equal(
                result.embedding_persistence_distance[w],
                np.linalg.norm(f64(e_hat0)[None, :] - f64(e), axis=-1),
            )
            np.testing.assert_array_equal(
                result.embedding_displacement[w],
                np.linalg.norm(f64(e_hat) - f64(e_hat0)[None, :], axis=-1),
            )
            np.testing.assert_array_equal(
                result.true_embedding_displacement[w],
                np.linalg.norm(f64(e) - f64(e0)[None, :], axis=-1),
            )
    # None of the four embedding series is degenerate on a real model: the
    # imagination is neither the truth nor the anchor, and the frames move.
    for name in TRAJECTORY_FIELDS[5:]:
        assert getattr(result, name).min() > 0.0, name


def test_reference_trajectories_runs_the_canonical_pass_alone_with_the_noise_reference(
    tmp_path, monkeypatch
):
    """The call `reference_trajectories` makes, pinned by its arguments and by
    what reaches `imagine`: no intervention arm, the noise reference drawn,
    the flag on, and every protocol argument forwarded verbatim. Behaviourally,
    `imagine` is called exactly twice per window -- the canonical pass and
    the noise draw -- and the second call carries the first's real actions;
    an arm left in would add a call, a skipped noise reference would remove
    one, and either moves the stream the curves are pinned on."""
    paths = [write(tmp_path, varied_action_episode())]
    model, probe = real_model_and_probe()
    calls: list[dict] = []
    real_diagnose = diagnostics_module._diagnose

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_diagnose(*args, **kwargs)

    monkeypatch.setattr(diagnostics_module, "_diagnose", spy)
    seen = _imagined_actions(monkeypatch, model)
    result = reference_trajectories(
        model, paths, probe, context=CONTEXT, horizon=HORIZON, seed=7,
        device=torch.device("cpu"), feature_backbone=None,
    )

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["arms"] == {}
    assert kwargs["noise_reference"] is True
    assert kwargs["keep_trajectories"] is True
    assert kwargs["context"] == CONTEXT and kwargs["horizon"] == HORIZON
    assert kwargs["seed"] == 7 and kwargs["feature_backbone"] is None
    assert kwargs["device"] == torch.device("cpu")
    assert result.windows_total == 2
    _, canonical = _per_rung(seen, arms=(), windows=2)
    assert len(canonical) == 2


def test_reference_trajectories_runs_in_eval_mode_and_takes_no_gradient(
    tmp_path, monkeypatch
):
    """Sampled AT the moment the RSSM is called, as the ladder's own test
    does it. On the real model this is not decoration: the encoder's output
    carries grad, and `.numpy()` on it inside the flag's block raises unless
    the entry point is under `no_grad`."""
    paths = [write(tmp_path, synthetic_episode())]
    model, probe = real_model_and_probe()
    model.train(True)
    states = []
    real = model.rssm.imagine
    monkeypatch.setattr(
        model.rssm, "imagine",
        lambda actions, state: states.append((model.training, torch.is_grad_enabled()))
        or real(actions, state),
    )
    result = trajectories(model, paths, probe)
    assert result.windows_total == 2 and states
    assert not any(training for training, _ in states)
    assert not any(grad for _, grad in states)


def test_reference_trajectories_raises_when_no_window_is_long_enough(tmp_path):
    """The same refusal as the ladder's, through the same `_no_window_error`:
    an empty stack of trajectories would otherwise surface as a numpy error
    about zero-length stacking, far from the split/horizon problem it is."""
    episode = synthetic_episode(length=CONTEXT + HORIZON)  # exactly `need`: excluded
    path = write(tmp_path, episode)
    with pytest.raises(ValueError, match="no diagnostic window"):
        trajectories(OracleModel(), [path], oracle_probe(episode))
```

Why these rigs and not others, in one place: the three-length oracle (`need`, `need + 1`, `3 * need`) is the window-set fixture the ladder's own tests use, so a `true_positions` slice off by one frame and a re-inlined window rule both have a failing case; `DriftingModel` is the one rig on which the imagination, the floor and the anchor are three DIFFERENT closed forms (on `OracleModel` all three coincide, so a `positions_real` taken from the imagination would pass); the exact rig is the one on which `embedding_distance_to_truth` is exactly zero, which is what separates `e(h)` from `e(h − 1)`; and the real sampling rig (`real_model_and_probe`, a `cnn` `WorldModel` at `embed_dim` 2048) is the only one on which `ê(0) ≠ e(0)`, so it alone catches a displacement series anchored on the wrong side -- and the only one on which "bitwise the pass's curve" is a statement about a stochastic model rather than about integers.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_diagnostics.py -q -p no:cacheprovider`
Expected: the whole file fails to COLLECT, because the import block names two symbols the module does not have yet:

```
tests/eval/test_diagnostics.py:38: in <module>
    from mbfps.eval.diagnostics import (
E   ImportError: cannot import name 'Trajectories' from 'mbfps.eval.diagnostics' (.../src/mbfps/eval/diagnostics.py)
=========================== short test summary info ============================
ERROR tests/eval/test_diagnostics.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.52s
```

This takes the ladder's 108 existing tests down with it; that is expected at this step and only at this step. (Measured 2026-09-13 on the unpatched module with the tests above appended: exactly this error.)

- [ ] **Step 3: Implement the flag, the fields, `Trajectories` and `reference_trajectories`**

Seven edits to `src/mbfps/eval/diagnostics.py`, top to bottom.

(a) The module docstring: append this paragraph before its closing `"""` (line 165), after the "WHY THE ROLLOUT BODY IS DUPLICATED HERE" paragraph:

```python
THE TRUST HORIZON (M3d) READS THE SAME CANONICAL PASS, PER WINDOW. The ladder
stores mean error curves, and a mean curve cannot tell an imagination that
drifts slowly from one that moves the wrong way. `keep_trajectories` on
`_diagnose` keeps, for the canonical pass only, the per-window rows those
curves are means of, plus four embedding-space series read against the
ENCODER'S embedding of the real future frames -- the one reading in this
module that is imagination-vs-truth rather than imagination-vs-imagination --
and `reference_trajectories` is the entry point that runs the pass with no
intervention arm and hands them over. The flag draws nothing from the stream
and, when False, leaves the pass byte-identical to what the ladder pins.
"""
```

(b) `_Pass` (line 392-424): append the nine fields after `window_episode`'s docstring (the line ending `229 independent draws from 24 clusters."""`), and the field-name tuple after the class. `_Pass` is a plain `@dataclass` with no defaults so far, so defaulted fields must come last -- they do:

```python
    come from 24 episodes at 3 to 10 windows each, so this is what separates
    229 independent draws from 24 clusters."""

    # The trust horizon's per-window trajectories (M3d). None unless the
    # traversal ran with `keep_trajectories=True`; then every one is set, for
    # the CANONICAL pass only -- no intervention arm carries them.
    positions: np.ndarray | None = None
    """`(n_windows, horizon, 2)`: p_hat(h), the probe of the imagined
    embedding at step h -- rows `model_pred[:, :2]`, bitwise."""
    positions_at_context: np.ndarray | None = None
    """`(n_windows, 2)`: p_hat(0), the probe of the last posterior latent's
    embedding, taken from the persistence prediction's own first row so
    that `|p_hat(0) - p(h)|` recomputed from it is bitwise the persistence
    curve's row."""
    positions_real: np.ndarray | None = None
    """`(n_windows, horizon, 2)`: p_hat_real(h), the probe of the floor's
    embedding at step h -- the same probe reading the posterior over the
    real future frames."""
    true_positions: np.ndarray | None = None
    """`(n_windows, horizon, 2)`: p(h), the privileged `(pos_x, pos_y)` at
    frame `start + context + h` -- the first two columns of the truth slice
    the curves are scored against."""
    true_at_context: np.ndarray | None = None
    """`(n_windows, 2)`: p(0), the privileged position at frame
    `start + context`, the last context frame."""
    embedding_distance_to_truth: np.ndarray | None = None
    """`(n_windows, horizon)`: `||e_hat(h) - e(h)||`, the imagination's
    embedding against the ENCODER'S embedding of the frame the action
    produced. See `_diagnose`: this is imagination-vs-truth, which no
    ladder channel measures."""
    embedding_persistence_distance: np.ndarray | None = None
    """`(n_windows, horizon)`: `||e_hat(0) - e(h)||`, the embedding-space
    persistence error -- the last context embedding held for the horizon,
    against the encoder's embedding of each future frame."""
    embedding_displacement: np.ndarray | None = None
    """`(n_windows, horizon)`: `||e_hat(h) - e_hat(0)||`, how far the
    imagination moved in embedding space."""
    true_embedding_displacement: np.ndarray | None = None
    """`(n_windows, horizon)`: `||e(h) - e(0)||`, how far the encoder's
    embedding of the real frames moved."""


_TRAJECTORY_FIELDS: tuple[str, ...] = (
    "positions", "positions_at_context", "positions_real",
    "true_positions", "true_at_context",
    "embedding_distance_to_truth", "embedding_persistence_distance",
    "embedding_displacement", "true_embedding_displacement",
)
"""The nine `_Pass` fields `keep_trajectories` fills, in `_Pass` order; the
one list `_diagnose` collects by and `reference_trajectories` repackages
by, so a field added to one cannot be forgotten by the other."""
```

(c) `_diagnose`'s signature (line 439): the flag goes after `noise_reference`, keyword-only like everything after the `*`:

```python
    arm_pairs: dict | None = None,
    noise_reference: bool = True,
    keep_trajectories: bool = False,
) -> _Pass:
```

(d) `_diagnose`'s docstring: insert this paragraph after the `arm_pairs` paragraph (the one ending `whose real arm cancels.`) and before "Deliberately NOT separately decorated" (line 466):

```python
    `keep_trajectories` (M3d) additionally carries, on the CANONICAL pass
    only, the per-window rows the curves are means of -- the probe's
    imagined, persistence and floor positions, the true positions -- and
    FOUR EMBEDDING-SPACE SERIES THAT ARE NEW: every embedding channel the
    ladder reads is imagination-vs-imagination (an intervened draw against
    the canonical one, or the canonical one against a second draw), so
    nothing above measures the imagination against the TRUTH. These do. The
    truth in embedding space is the encoder's own embedding of the frame
    the action produced, `e(h) = handle.embeddings[0, context + h - 1]` --
    the embedding head's training target (`world_model.py` fits
    `predictions["embedding"]` to `embeddings.detach()`) -- with
    `e(0) = handle.embeddings[0, context - 1]`, the last context frame.
    Against it: `e_hat(h)`, the head's output on the imagined latent (the
    rows `real_embedding` already holds), and `e_hat(0)`, the head's output
    on the last posterior latent (`last_context_embedding`, the persistence
    anchor). With the flag False the pass is byte-identical to today's:
    the extra work is numpy on rows already in hand, draws nothing from the
    stream, and calls `probe_targets` once more only inside the flag.

```

(e) The collectors (after line 493, `window_episode: list[int] = []`):

```python
    noise_restored: list[bool] = []
    window_episode: list[int] = []
    kept: dict[str, list[np.ndarray]] | None = (
        {name: [] for name in _TRAJECTORY_FIELDS} if keep_trajectories else None
    )
```

(f) The per-window block, inserted after the six `reference_windows[...]` appends (after line 596) and before `arm_embeddings = {}` (line 598). It sits AFTER the noise reference has drawn and restored the stream and reads only arrays already computed for the curves -- `model_pred`, `floor_pred`, `pers_pred`, `truth`, `real_embedding`, `last_context_embedding` -- plus one slice of `embeddings` and one `probe_targets` call for `p(0)`:

```python
            reference_windows["persistence_angle"].append(
                angle_error_degrees(pers_pred, truth)
            )

            if kept is not None:
                # `embeddings[0, k]` is the encoder's row for frame
                # `start + 1 + k`, so row `context + h - 1` is frame
                # `start + context + h` -- the frame `imagine`'s step h
                # predicts and `truth[h - 1]` scores -- and row `context - 1`
                # is the last context frame, `start + context`. One slice
                # from that row on: `true_embedding[0]` is e(0),
                # `true_embedding[1:]` is e(1..horizon).
                true_embedding = embeddings[0, context - 1 :].cpu().numpy()
                held = np.repeat(last_context_embedding[None, :], horizon, axis=0)
                held_truth = np.repeat(true_embedding[:1], horizon, axis=0)
                kept["positions"].append(model_pred[:, :2])
                # Row 0 of the persistence prediction, not a fresh
                # `apply_probe` on one row: the curve's persistence error is
                # scored on these rows, and a separate matmul over a
                # different shape is not guaranteed the same last bit.
                kept["positions_at_context"].append(pers_pred[0, :2])
                kept["positions_real"].append(floor_pred[:, :2])
                kept["true_positions"].append(truth[:, :2])
                kept["true_at_context"].append(
                    probe_targets(
                        episode.privileged[start + context : start + context + 1],
                        episode.privileged_keys,
                    )[0, :2]
                )
                kept["embedding_distance_to_truth"].append(
                    _embedding_distance(real_embedding, true_embedding[1:])
                )
                kept["embedding_persistence_distance"].append(
                    _embedding_distance(held, true_embedding[1:])
                )
                kept["embedding_displacement"].append(
                    _embedding_distance(real_embedding, held)
                )
                kept["true_embedding_displacement"].append(
                    _embedding_distance(true_embedding[1:], held_truth)
                )

            arm_embeddings = {}
```

`_embedding_distance` casts both arguments to float64 before the norm, so every embedding series is device-independent given the embeddings, exactly like the ladder's own embedding channel; `embeddings` is the encoder's float32 output on `device`, hence the `.cpu()`.

(g) The return (line 634) gains one line, and the new section follows `_diagnose`'s closing `)` (line 635), before the `# Diagnostic 1` banner:

```python
        windows_total=stacked["rssm_position"].shape[0],
        window_episode=np.array(window_episode, dtype=int),
        **({} if kept is None else {name: np.stack(rows) for name, rows in kept.items()}),
    )


# ---------------------------------------------------------------------------
# The reference pass's trajectories, for the trust horizon (M3d).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trajectories:
    """The canonical pass, per window and per step, instead of as mean curves.

    What `scripts/trust_horizon.py` reads. The nine arrays are `_Pass`'s
    `keep_trajectories` fields under the same names and shapes, none of them
    optional here; `window_episode` and `windows_total` are the pass's own,
    so a consumer clusters by the same labels the ladder clusters by; and
    the two curves are the pass's `reference.rssm_position` and
    `reference.persistence_position` -- `np.stack(rows).mean(axis=0)` over
    the per-window rows, the same reduction the diagnostic records were
    written with -- so a consumer that recomputes them from the rows and
    compares against the stored diagnostic can demand max |delta| == 0.0.
    """

    positions: np.ndarray
    positions_at_context: np.ndarray
    positions_real: np.ndarray
    true_positions: np.ndarray
    true_at_context: np.ndarray
    embedding_distance_to_truth: np.ndarray
    embedding_persistence_distance: np.ndarray
    embedding_displacement: np.ndarray
    true_embedding_displacement: np.ndarray
    window_episode: np.ndarray
    windows_total: int
    reference_position: np.ndarray
    persistence_position: np.ndarray


@torch.no_grad()
def reference_trajectories(
    model,
    val_paths,
    embedding_probe_weights: dict,
    *,
    context: int,
    horizon: int,
    seed: int,
    device,
    feature_backbone,
) -> Trajectories:
    """The canonical pass alone, with its per-window trajectories kept.

    `_diagnose` with NO intervention arm and the noise reference drawn, so
    the sampling stream is walked exactly as the ladder walks it: the
    context filter, the per-window snapshot, the canonical `imagine` from
    it, the floor, then the noise draw and its restore. The curves this
    returns are therefore bitwise the ladder's reference curves -- and,
    through the ladder's own pin, bitwise `evaluate_rollout`'s -- which is
    what lets the trust pass check itself against the stored diagnostic
    before reading anything off the rows.

    No defaults for `context`, `horizon`, `seed`, `device` or
    `feature_backbone`: the caller reads every one of them from the cell's
    diagnostic record, and a default here would let a mismatch pass in
    silence.
    """
    result = _diagnose(
        model, val_paths, embedding_probe_weights,
        arms={}, context=context, horizon=horizon, seed=seed, device=device,
        feature_backbone=feature_backbone, noise_reference=True,
        keep_trajectories=True,
    )
    return Trajectories(
        **{name: getattr(result, name) for name in _TRAJECTORY_FIELDS},
        window_episode=result.window_episode,
        windows_total=result.windows_total,
        reference_position=result.reference.rssm_position,
        persistence_position=result.reference.persistence_position,
    )


# ---------------------------------------------------------------------------
# Diagnostic 1: the action-intervention ladder.
# ---------------------------------------------------------------------------
```

Three things about this implementation that are choices rather than transcription, and why:

- `positions_at_context` is `pers_pred[0, :2]`, the first row of the persistence prediction the curve is scored on, rather than `apply_probe(weights, last_context_embedding[None, :])[0, :2]`. Measured on this box the two are bitwise equal (50 random probes at E = 2048, every row of the repeated product identical to the single-row product), but BLAS is not obliged to keep it that way across shapes, and the self-check in Task 4 demands `max |Δ| == 0.0` on the persistence curve. Taking the row the curve used makes the identity true by construction.
- `true_at_context` goes through `probe_targets` on a one-row slice rather than indexing `episode.privileged` by column: it is the same column lookup and float64 cast the truth slice gets, so `p(0)` and `p(h)` cannot disagree about which columns are `(pos_x, pos_y)`. The call sits INSIDE the flag, so the ladder's `_truth_slices` spy (which counts `probe_targets` calls through `diagnostics_module`) sees exactly the calls it saw before.
- `e(h)` is taken as ONE slice `embeddings[0, context - 1 :]` -- `(horizon + 1, E)`, row 0 = `e(0)`, rows `1:` = `e(1..horizon)` -- rather than two index expressions, so the anchor and the series cannot be off from each other by a frame. `embeddings` has exactly `need = context + horizon` rows, so the slice is exactly `horizon + 1` long and a wrong offset changes a shape rather than a value.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_diagnostics.py -q -p no:cacheprovider -k "trajector or keyword_only or true_positions_are or probe_channel_reads or embedding_channel_has or recomputed_by_hand or per_window_rows_reduce"`
Expected: `17 passed, 108 deselected` -- the twelve new test functions, four of them parametrised over `DEVICES` (2 each on a box with MPS) and one over the two oracle rigs. Measured ~4 s. On a box without MPS the four device-parametrised tests collect once each: `13 passed`.

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_diagnostics.py -q -p no:cacheprovider`
Expected: **17 more than before this task** -- the ladder's file measured `108 passed` at b382194 on this box (all 108, including the three `shipped` self-checks, which need `runs/m3_study`, `data/my_way_home` and MPS) and measures **125 passed** with this task applied; record the measured number. On a box without the shipped artefacts the three `shipped` tests skip and the line reads `122 passed, 3 skipped`. Every one of the 108 existing tests is untouched and stays green: the ladder's bitwise pins against `evaluate_rollout` and the shipped records are what "the flag is off by default" is measured by. Measured 254 s in the repo (the three shipped tests are ~4 min of it), 11 s without them.

- [ ] **Step 5: Mutation-test**

Set up a self-checked harness. All three checks matter here and each has silently lied in this project: the package is installed editable, so a copy of the tree that is not first on `PYTHONPATH` tests the REPO's `diagnostics.py` and reports every mutation caught; a same-length mutation can leave a stale `.pyc`; and an unproven harness proves nothing. Nothing is committed yet, so export by copy. Save this under `scratchpad/` (gitignored, never committed), run it, and read the first two printed lines before the table:

```python
# scratchpad/mutate_task3.py -- do not commit
import os, shutil, subprocess, sys
from pathlib import Path

REPO = Path("/Users/raphaelchen/Desktop/csgo-bot")
S = Path(__file__).resolve().parent / "tree"            # a copy of src/ tests/ scripts/ pyproject.toml
SRC = S / "src" / "mbfps" / "eval" / "diagnostics.py"
PY = str(REPO / ".venv" / "bin" / "python")
ENV = dict(os.environ, PYTHONPATH=str(S / "src"), PYTHONDONTWRITEBYTECODE="1")
NEW_TESTS = (
    "trajector or keyword_only or true_positions_are or probe_channel_reads or "
    "embedding_channel_has or recomputed_by_hand or per_window_rows_reduce"
)

def clear_pyc():
    for d in S.rglob("__pycache__"):
        shutil.rmtree(d)

def run():
    clear_pyc()
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "tests/eval/test_diagnostics.py", "-k", NEW_TESTS],
                       cwd=S, env=ENV, capture_output=True, text=True)
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
 "FATAL: _TRAJECTORY_FIELDS emptied": (
    '_TRAJECTORY_FIELDS: tuple[str, ...] = (\n    "positions", "positions_at_context", "positions_real",\n    "true_positions", "true_at_context",\n    "embedding_distance_to_truth", "embedding_persistence_distance",\n    "embedding_displacement", "true_embedding_displacement",\n)',
    '_TRAJECTORY_FIELDS: tuple[str, ...] = ()'),
 "flag on by default": ("    keep_trajectories: bool = False,\n", "    keep_trajectories: bool = True,\n"),
 "fields never populated": ("        **({} if kept is None else {name: np.stack(rows) for name, rows in kept.items()}),\n", ""),
 "e(h) one frame early": ("_embedding_distance(real_embedding, true_embedding[1:])", "_embedding_distance(real_embedding, true_embedding[:-1])"),
 "e(0) is the first horizon frame": ("held_truth = np.repeat(true_embedding[:1], horizon, axis=0)", "held_truth = np.repeat(true_embedding[1:2], horizon, axis=0)"),
 "persistence distance uses e_hat(h)": ("_embedding_distance(held, true_embedding[1:])", "_embedding_distance(real_embedding, true_embedding[1:])"),
 "persistence distance against e(0)": ("_embedding_distance(held, true_embedding[1:])", "_embedding_distance(held, held_truth)"),
 "displacement against e(0) not e_hat(0)": ("_embedding_distance(real_embedding, held)\n", "_embedding_distance(real_embedding, held_truth)\n"),
 "true displacement against e_hat(0)": ("_embedding_distance(true_embedding[1:], held_truth)", "_embedding_distance(true_embedding[1:], held)"),
 "distance to truth in float32": ("_embedding_distance(real_embedding, true_embedding[1:])", "np.linalg.norm(real_embedding - true_embedding[1:], axis=-1)"),
 "positions from the floor": ('kept["positions"].append(model_pred[:, :2])', 'kept["positions"].append(floor_pred[:, :2])'),
 "positions_real from the imagination": ('kept["positions_real"].append(floor_pred[:, :2])', 'kept["positions_real"].append(model_pred[:, :2])'),
 "positions_at_context from horizon step 1": ('kept["positions_at_context"].append(pers_pred[0, :2])', 'kept["positions_at_context"].append(model_pred[0, :2])'),
 "true_positions one frame early": ('kept["true_positions"].append(truth[:, :2])', 'kept["true_positions"].append(probe_targets(episode.privileged[start + context : start + need], episode.privileged_keys)[:, :2])'),
 "true_at_context at horizon step 1": ("episode.privileged[start + context : start + context + 1],", "episode.privileged[start + context + 1 : start + context + 2],"),
 "the flag draws from the stream": ("                true_embedding = embeddings[0, context - 1 :].cpu().numpy()\n", "                model.rssm.imagine(handle.horizon_actions, handle.state)\n                true_embedding = embeddings[0, context - 1 :].cpu().numpy()\n"),
 "reference_trajectories skips the noise reference": ("        feature_backbone=feature_backbone, noise_reference=True,\n        keep_trajectories=True,", "        feature_backbone=feature_backbone, noise_reference=False,\n        keep_trajectories=True,"),
 "reference_trajectories carries an arm": ("        arms={}, context=context, horizon=horizon, seed=seed, device=device,\n        feature_backbone=feature_backbone, noise_reference=True,", "        arms={\"extra\": lambda handle: model.rssm.imagine(handle.horizon_actions, handle.state)[\"latent\"]}, context=context, horizon=horizon, seed=seed, device=device,\n        feature_backbone=feature_backbone, noise_reference=True,"),
 "reference_trajectories not under no_grad": ("@torch.no_grad()\ndef reference_trajectories(", "def reference_trajectories("),
 "reference_position is the persistence curve": ("        reference_position=result.reference.rssm_position,", "        reference_position=result.reference.persistence_position,"),
 "persistence_position is the floor curve": ("        persistence_position=result.reference.persistence_position,", "        persistence_position=result.reference.floor_position,"),
 "Trajectories not frozen": ("@dataclass(frozen=True)\nclass Trajectories:", "@dataclass\nclass Trajectories:"),
 "window_episode zeroed": ("        window_episode=result.window_episode,\n        windows_total=result.windows_total,\n        reference_position", "        window_episode=np.zeros_like(result.window_episode),\n        windows_total=result.windows_total,\n        reference_position"),
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
        print(f"{'CAUGHT' if rc else 'SURVIVED':8} | {name} | {len(failed)} | {failed[:4]}")
        if name.startswith("FATAL") and rc == 0:
            sys.exit("harness cannot detect a fatal mutation")
finally:
    SRC.write_text(original)
    clear_pyc()   # self-check 3: never leave bytecode from a mutated source behind
```

Build the copy with `rm -rf scratchpad/tree && mkdir -p scratchpad/tree && cp -R src tests scripts pyproject.toml scratchpad/tree/` and run `.venv/bin/python scratchpad/mutate_task3.py` (a `scratchpad/tree` left by Task 2's harness would otherwise be copied over, not replaced). The first line must name a file under `scratchpad/tree`, the second must read `baseline: 17 passed, 108 deselected`, and every row must read `CAUGHT`:

| mutation | must be caught by |
|---|---|
| `_TRAJECTORY_FIELDS = ()` (harness self-check, fatal) | 15 tests, everything that touches a trajectory field -- `Trajectories(**{})` is missing nine arguments |
| `keep_trajectories` defaults to `True` | `test_the_flag_is_keyword_only_and_off_by_default` |
| the nine fields are never populated (the `**` line in the return dropped) | 11 tests, from `test_the_flagged_pass_and_the_trajectories_carry_the_nine_arrays_at_their_shapes` (`is not None`) through every closed-form test (`TypeError` on `None`) |
| `e(h)` indexed one frame early (`true_embedding[:-1]` for `[1:]`; one frame LATE runs off the `need`-row array and is a shape error) | `..._has_the_closed_form_of_the_frame_tags[exact]` (distance to truth 1, not 0), `[drifting]`, `test_every_row_is_recomputed_by_hand_...` |
| `e(0)` taken at the first horizon frame (`true_embedding[1:2]` for `[:1]`) | both closed-form tests (`true_embedding_displacement` reads `h − 1`), `..._recomputed_by_hand_...` |
| persistence distance anchored on `ê(h)` instead of `ê(0)` (`real_embedding` for `held`) | `..._closed_form_...[exact]` (persistence 0 where it must be `h`), `..._recomputed_by_hand_...` |
| persistence distance against `e(0)` instead of `e(h)` (`held_truth` for `true_embedding[1:]`) | both closed-form tests (reads 0), `..._recomputed_by_hand_...` |
| imagined displacement anchored on `e(0)` instead of `ê(0)` | `..._recomputed_by_hand_...` ONLY -- on every oracle rig the filter is exact so `ê(0) == e(0)`; this row is why the real-model test exists |
| true displacement anchored on `ê(0)` instead of `e(0)` | `..._recomputed_by_hand_...` only, same reason |
| distance to truth taken in float32 (`np.linalg.norm` on the float32 rows) | `..._nine_arrays_at_their_shapes[cpu]`/`[mps]` (dtype), `..._recomputed_by_hand_...` (values) |
| `positions` taken from the floor (`floor_pred` for `model_pred`) | `..._probe_channel_reads_...` (drifting: two frames per step vs one), `test_the_per_window_rows_reduce_to_the_curves_bitwise[cpu]`/`[mps]`, `..._recomputed_by_hand_...` |
| `positions_real` taken from the imagination (`model_pred` for `floor_pred`) | `..._probe_channel_reads_...`, `..._recomputed_by_hand_...` |
| `positions_at_context` taken from horizon step 1 (`model_pred[0]` for `pers_pred[0]`) | `..._probe_channel_reads_...`, `..._per_window_rows_reduce_...[cpu]`/`[mps]` (the persistence identity breaks), `..._recomputed_by_hand_...` |
| `true_positions` sliced one frame early (`start + context : start + need`) | `test_true_positions_are_the_privileged_positions_at_the_frames_the_actions_produced`, `..._per_window_rows_reduce_...[cpu]`/`[mps]` |
| `true_at_context` taken at horizon step 1 | `test_true_positions_are_...` (the anchor check) |
| the flag draws from the stream (an extra `imagine` inside the block) | `test_without_the_flag_..._is_bitwise_what_it_was[cpu]`/`[mps]` (curves and generator state), `..._curves_are_the_passes_own_and_the_rollouts_bitwise[cpu]`/`[mps]`, `..._runs_the_canonical_pass_alone_...` (3 `imagine` calls per window), `..._recomputed_by_hand_...` |
| `reference_trajectories` passes `noise_reference=False` | `..._runs_the_canonical_pass_alone_with_the_noise_reference` (kwargs, and `_per_rung` sees 1 call per window), `..._recomputed_by_hand_...` (call-count assertion) |
| `reference_trajectories` carries an intervention arm | `..._runs_the_canonical_pass_alone_...` (`arms != {}`, 3 calls per window), `..._recomputed_by_hand_...` |
| `@torch.no_grad()` dropped from `reference_trajectories` | 9 tests: `..._runs_in_eval_mode_and_takes_no_gradient`, and every real-model test (`.numpy()` on a tensor that requires grad raises `RuntimeError`) |
| `Trajectories.reference_position` is the persistence curve | `..._curves_are_the_passes_own_and_the_rollouts_bitwise[cpu]`/`[mps]`, `..._per_window_rows_reduce_...[cpu]`/`[mps]` |
| `Trajectories.persistence_position` is the floor curve | the same four |
| `Trajectories` not frozen | `..._nine_arrays_at_their_shapes[cpu]`/`[mps]` (`FrozenInstanceError` not raised) |
| `Trajectories.window_episode` zeroed | `..._nine_arrays_at_their_shapes[cpu]`/`[mps]` (`[0, 0, 1, 1]`), `test_true_positions_are_...` (`[0, 1, 1, 1]`) |

All twenty-three rows were run and caught on 2026-09-13 with that harness (baseline `17 passed, 108 deselected in 3.96s`). Any mutation that survives is a missing test; add it before committing.

One deliberate NON-row, stated so nobody adds it and reads its survival as a gap: replacing `pers_pred[0, :2]` with `apply_probe(embedding_probe_weights, last_context_embedding[None, :])[0, :2]` SURVIVES on this box (measured: bitwise equal for 50 random probes at E = 2048). That line is not a guard -- it is the construction that makes Task 4's persistence self-check an identity rather than a BLAS coincidence -- and its test is `test_the_per_window_rows_reduce_to_the_curves_bitwise`, which would fail on a BLAS whose one-row and repeated-row products differ in the last bit.

- [ ] **Step 6: Run the full suite**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: `0 failed`, **17 more passed than Task 2's final full-suite step measured** (the contract's baseline is 1278 at b382194; Tasks 1 and 2 add their own, so record the number Task 2 left and add 17). Run from the repo root: the real-data tests read `data/my_way_home` and `runs/m3_study` relative to the cwd (measured: `tests/eval/test_rollout.py::test_rollout_produces_the_full_band_on_real_data` fails with `need at least one episode on each side` from any other directory -- that is the cwd, not this task). Measured ~6 min. No file outside `tests/eval/test_diagnostics.py` changes count: this task adds no test elsewhere and `_Pass`'s new fields are defaulted, so nothing that builds one moves.

- [ ] **Step 7: Commit**

```bash
git add src/mbfps/eval/diagnostics.py tests/eval/test_diagnostics.py
git commit -m "feat: the reference pass keeps its per-window trajectories, and four embedding series against the truth

keep_trajectories on _diagnose fills nine optional _Pass fields for the
canonical pass only: the probe's imagined, floor and persistence-anchor
positions, the true positions, and -- new, since every ladder channel is
imagination-vs-imagination -- the imagination's and the anchor's distance
to the encoder's embedding of the real future frame, and both
displacements. reference_trajectories runs the pass with no arm and the
noise reference drawn, so its curves are bitwise the ladder's, and hands
the rows over as a frozen Trajectories for the trust horizon's self-check.
With the flag off the pass is byte-identical; the ladder's pins say so.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: scripts/trust_horizon.py, part 1: loading as diagnose_dynamics does, the four checks in order, the reference pass, the per-cell record

The script that turns nine frozen checkpoints into nine `trust_<arm>_seed<n>.json` records. It loads each cell exactly as `diagnose_dynamics.py` does (checkpoint validated for arm AND seed, the study record, the split by name, the probe refit on the train split at the cell's seed), judges the four checks of spec section 2.3 in their fixed order -- 11, 12, 14, then the trust pass's own 30 -- runs ONE reference pass with the trajectories kept (`reference_trajectories`, Task 3), fills the record with the pure functions of Tasks 1 and 2, and writes it atomically through `write_record`. Nothing is pooled and no reading is printed here; Task 6 appends the pooling glue and the two readings after the per-cell loop. The script tests do not stub the model: they train the same tiny cell `test_aggregate.py` trains (`run_job(..., steps=5, seq_len=4, context=2, horizon=3, device="cpu")` on `small_buffer`), have `diagnose_dynamics.main` write the diagnostic for it, and then drive `main` -- so the self-check's `max |delta| == 0.0` is asserted against a diagnostic the ladder really wrote, on the same machine.

**Depends on Tasks 1-3** having landed `mbfps.eval.trust` (`moved_mask`, `crossing_step`, `persistence_margin`, `displacement_decomposition`, `embedding_ratio`, `scale_corrected_error`) and `mbfps.eval.diagnostics.reference_trajectories` / `Trajectories`. Step 2 goes red with or without them (the script file does not exist); Step 4 cannot go green without them.

**Files:**
- Create: `scripts/trust_horizon.py`
- Create: `tests/eval/test_trust_horizon_script.py`
- Read, not modified: `scripts/diagnose_dynamics.py` -- the exit-status block (`EXIT_OK` .. `EXIT_UNSUPPORTED_DEVICE`, lines 177-190), `checkpoint_path` (210), `load_checkpoint_model` (220), `probe_is_measurable` (932), `diagnose_cell` (1275-1410: the loading path this script reproduces call for call), `diagnostic_record_path` (1413), `build_record` (1491: the diagnostic's shape -- `context`, `horizon`, `episodes.val`, `probe.embedding_selection_r2`, `windows.total`, `windows.episode`, `curves.reference_position`, `curves.persistence_position`, `curves.floor_position`), `parse_args` (1612), `main` (1667-1808: the order the checks are judged in and the wording of each refusal)
- Read, not modified: `tests/eval/conftest.py` `small_buffer` (six 40-step episodes, `pos_x = 3t`, `pos_y = -2t`); `tests/eval/test_aggregate.py` `REAL_JOB` / `REAL_JOB_KW` (line 811); `tests/eval/test_diagnose_dynamics_script.py` lines 39-44 (the by-path script import every script test uses)

> The loading path is IMPORTED from `scripts/diagnose_dynamics.py` by path, not duplicated. Four names come from there -- `checkpoint_path`, `load_checkpoint_model` (which validates the checkpoint's arm and seed separately and builds the model), `diagnostic_record_path`, and `probe_is_measurable` -- because each is a rule the ladder already applied to these nine cells, and a second copy here could drift from the ladder's while both kept passing, leaving the two tools reading one cell through two loaders. Everything else on the path already lives in `src/` and is imported from there: `episode_split` / `VAL_FRACTION` / `SPLIT_SEED`, `fit_probes`, `evaluate_rollout`, `load_record` / `write_record` / `git_sha`, `get_config`, `encoder_backbone`, `get_device`. The by-path import is the same `importlib.util.spec_from_file_location` the tests use, resolved against `Path(__file__)`, so it works whether the script is run as `python scripts/trust_horizon.py` or loaded by a test.

**Interfaces:**
- Consumes (Task 1, `mbfps.eval.trust`): `moved_mask(p_true, p_true0, min_move=MIN_MOVE) -> np.ndarray` (`(n, H)` bool), `crossing_step(model_err, persist_err, moved) -> np.ndarray` (`(n,)` float64, `H + 1` never, NaN never-moved), `persistence_margin(model_err, persist_err) -> np.ndarray` (`persist_err - model_err`).
- Consumes (Task 2, `mbfps.eval.trust`): `displacement_decomposition(p_hat, p_true, p_hat0, p_true0, p_hat_real, moved) -> Decomposition` (fields `ratio_probe, ratio_raw, cosine, moved, zero_displacement`), `embedding_ratio(e_hat_disp, e_true_disp, moved) -> np.ndarray`, `scale_corrected_error(p_hat, p_true, p_hat0, moved, episode, alphas=ALPHAS) -> ScaleCorrection` (fields `alpha_a, alpha_b, score_a, score_b, held_out, boundary, folds_available`; ties in the argmin -> the smallest alpha; `folds_available False` and every array NaN / False below two distinct episode labels).
- Consumes (Task 3, `mbfps.eval.diagnostics`): `reference_trajectories(model, val_paths, embedding_probe_weights, *, context, horizon, seed, device, feature_backbone) -> Trajectories`; `Trajectories` with fields `positions (n, H, 2)`, `positions_at_context (n, 2)`, `positions_real (n, H, 2)`, `true_positions (n, H, 2)`, `true_at_context (n, 2)`, `embedding_distance_to_truth (n, H)`, `embedding_persistence_distance (n, H)`, `embedding_displacement (n, H)`, `true_embedding_displacement (n, H)`, `window_episode (n,) int`, `windows_total int`, `reference_position (H,)`, `persistence_position (H,)` -- the two means being `np.stack(rows).mean(axis=0)`, the reduction `_diagnose` uses for its own curves.
- Consumes (existing): `scripts/diagnose_dynamics.py` `checkpoint_path(out_dir, arm, seed) -> Path`, `load_checkpoint_model(out_dir, arm, seed, cfg, device)`, `diagnostic_record_path(out_dir, arm, seed) -> Path`, `probe_is_measurable(cell: dict, widest_se: float) -> bool`; `mbfps.eval.study` `SPLIT_SEED`, `StudyJob`, `git_sha`, `job_record_path`, `load_record`, `write_record`; `mbfps.eval.probe.fit_probes`; `mbfps.eval.rollout.evaluate_rollout`; `mbfps.data.split.episode_split` / `VAL_FRACTION`; `mbfps.utils.config.ARMS` / `get_config`; `mbfps.eval.aggregate.SEEDS`; `mbfps.utils.device.get_device`.
- Produces (for Task 6, all in `scripts/trust_horizon.py`, signatures verbatim from the contract):
  - `EXIT_OK = 0`, `EXIT_NO_CHECKPOINTS = 11`, `EXIT_SPLIT_MISMATCH = 12`, `EXIT_RECORD_MISMATCH = 14`, `EXIT_SELF_CHECK_FAILED = 30`
  - `_parser() -> argparse.ArgumentParser` -- `--out` (Path, default `runs/m3_study_v2`), `--device` (default `mps`), `--data` (Path, default `data/my_way_home`), `--context` / `--horizon` (int, default None -> read from each cell's diagnostic), `--arms` (nargs +, choices `ARMS`, default all), `--seeds` (nargs +, int, default `[0, 1, 2]`)
  - `load_cell(out_dir: Path, arm: str, seed: int) -> Cell` -- raises `CellMissing` (a `FileNotFoundError` subclass, the typed error `main` maps to 11). `Cell` is a frozen dataclass `arm, seed, checkpoint: Path, record: dict, diagnostic: dict`; the model is built from `checkpoint` by `load_checkpoint_model` once the device is known.
  - `self_check(traj: Trajectories, diagnostic: dict) -> SelfCheck` -- frozen dataclass `reference_position_max_delta, reference_position_step, persistence_position_max_delta, persistence_position_step, windows_total_match, windows_episode_match`, property `ok`, methods `failures() -> list[str]` and `record() -> dict` (the five contract keys `reference_position_max_delta, persistence_position_max_delta, windows_total_match, windows_episode_match, ok`). The two curves it judges are recomputed from the ROWS the record is built from (`np.linalg.norm(positions - true_positions, axis=-1).mean(axis=0)` and the same with `positions_at_context[:, None, :]`, spec 2.3), not read off `Trajectories.reference_position` / `persistence_position` -- those are the pass's own reduction of the same rows, which Task 3 pins bitwise equal to these on a clean run.
  - `trust_record(arm, seed, traj: Trajectories, diagnostic: dict, *, context, horizon, device) -> dict` -- the LIVE record (numpy arrays, real NaNs, no `nonfinite` key); `write_record` sanitises it and adds `nonfinite {}` on disk. Beside the contract's keys it carries `displacement {probe_hat [n][H] = |d_hat|, probe_real [n][H] = |d_hat_real|, free_hat [n][H] = ||e_hat(h) - e_hat(0)||, free_true [n][H] = ||e(h) - e(0)||}` -- the numerator and denominator series Task 6's `pool_ratio` (a ratio of MEDIANS, spec 3.1) needs and cannot recover from the per-window ratios.
  - `trust_record_path(out_dir: Path, arm: str, seed: int) -> Path` -- `runs/<out>/trust_<arm>_seed<n>.json`; `write_trust_record(out_dir: Path, record: dict) -> Path`.
  - `_run_cell(args, cell: Cell, device, train, val) -> tuple[int, dict | None]` -- `(EXIT_OK, record)` when the cell's record was written, `(status, None)` on any refusal.
  - `main(argv: list[str] | None = None) -> int` -- loads every requested cell first (11 before any refit), then per cell in `args.arms x args.seeds` order: 12, the protocol check (14), the refit, the reproduction (14), the reference pass, 30, the record. Collects the live records in `records: dict[tuple[str, int], dict]` keyed `(arm, seed)` and the resolved horizon in `horizon` (identical across cells: the protocol check refuses a cell at another horizon). Returns `EXIT_OK` after the loop; Task 6 replaces that `return` with the pooling and the readings, reading `records`, `horizon` and `args.out`.
  - `tests/eval/test_trust_horizon_script.py` helpers Task 6 appends after: `_load_script(name)`, `script`, `diagnose`, `JOB`, `JOB_KW`, `CONTEXT`, `HORIZON`, `N_WINDOWS`, `RECORD` / `DIAGNOSTIC` / `TRUST` (file names), the `cell` fixture (`.out`, `.data`), `_argv(cell, *extra)`, `_doctor(path, edit)`, `FAULTS`, `_fabricated()`.

> RESOLVED AT RECONCILIATION: the contract lists 0 / 11 / 12 / 14 / 30 for this script and nothing else, and that is kept. Two of `diagnose_dynamics.py`'s statuses have no counterpart here and are NOT invented: a mislabelled checkpoint (`MislabelledCheckpoint`, 16 there) propagates out of `main` as an uncaught traceback, and an unsupported device (17 there) surfaces as the library's own `NotImplementedError` from `_diagnose`'s RNG snapshot. The script docstring says so; no code maps them.

> RESOLVED AT RECONCILIATION: "a `--context` that differs from the record's is refused the way `diagnose_dynamics.py` refuses it" -- `diagnose_dynamics.py` has no explicit check (its `main` at 1667-1808 judges only the split, the protocol, the stream and the record); a wrong `--context` there surfaces as `EXIT_RECORD_MISMATCH` (14) after the refit, because a rollout at another context cannot reproduce `curves.rssm_position`. This task refuses it under the same status (14), with the same `RECORD MISMATCH` prefix, UP FRONT against the diagnostic's recorded `context` / `horizon` -- before the ~20 s refit -- and the test pins that the refit never runs. The contract's phrase is to be read as "refused up front under `EXIT_RECORD_MISMATCH` against the diagnostic's recorded context/horizon".

> RESOLVED AT RECONCILIATION: `trust_record_path` is not in the contract and stays public anyway: Task 6 pools the in-memory records and does not read the files back, but Task 7's acceptance check globs the same `trust_<arm>_seed<n>.json` name, and one function owning it is cheaper than two spellings. No change.

> The reused statuses 11 / 12 / 14 collide with `diagnose_dynamics.py`'s BY DESIGN; Task 6's rewrite of `test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own` exempts exactly `{11, 12, 14}` against `diagnose_dynamics` alone.

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_trust_horizon_script.py`:

```python
"""scripts/trust_horizon.py, part 1: load as diagnose_dynamics does, the four
checks in order, one reference pass, one record per cell.

The script is loaded by path, the way `test_diagnose_dynamics_script.py`
loads its script. Unlike that file, nothing here stubs the model: the cell
under test is the same tiny cell `test_aggregate.py` trains --
`run_job(..., steps=5, seq_len=4, context=2, horizon=3, device="cpu")` on the
shared `small_buffer` -- and `diagnose_dynamics.main` writes the diagnostic
for it, so the self-check's `max |delta| == 0.0` is asserted against a
diagnostic the ladder really wrote on the same machine, and every refusal is
produced by doctoring ONE file the way a real drift would.

THE FIXTURE'S FACTS, read off `conftest.small_buffer`, not off the script:
six 40-step episodes; `episode_split(val_fraction=0.2, seed=0)` holds out
ONE, so every window carries episode label 0, the folds of the scale
correction are unavailable and every clustered SE is NaN; `window_starts(40,
2, 3)` = range(0, 36, 5) cuts EIGHT windows from it; `pos_x = 3t`,
`pos_y = -2t`, so the true displacement over h steps is h * sqrt(13) = 3.61,
7.21, 10.82 map units -- below the 5-unit moved threshold at h=1, above it
from h=2 on, in every window.

The pure pieces -- `self_check`, `trust_record`, `write_trust_record` -- are
also driven from a fabricated `Trajectories` with answers worked by hand, so
a swapped argument (persistence for model error, the probe's channel for the
embedding's) is caught by a value and not by a shape.
"""

import dataclasses
import importlib.util
import json
import math
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import mbfps.eval.study as study
from mbfps.eval.diagnostics import Trajectories
from mbfps.eval.study import StudyJob, job_record_path, load_record, run_job
from mbfps.utils.config import ARMS

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script("trust_horizon")
diagnose = _load_script("diagnose_dynamics")

JOB = StudyJob("random_vit", 1)
JOB_KW = dict(steps=5, seq_len=4, context=2, horizon=3, device="cpu")
"""`test_aggregate.py`'s REAL_JOB / REAL_JOB_KW: `random_vit` because its
backbone needs no downloaded weights; context 2 / horizon 3 because
`window_starts(40, 5, 45)` is empty on 40-step episodes."""
CONTEXT, HORIZON = JOB_KW["context"], JOB_KW["horizon"]
N_WINDOWS = 8
"""`window_starts(40, 2, 3)` = range(0, 36, 5) -> starts 0, 5, ..., 35."""
RECORD = "result_random_vit_seed1.json"
DIAGNOSTIC = "diagnostic_random_vit_seed1.json"
TRUST = "trust_random_vit_seed1.json"

EXPECTED_KEYS = {
    "arm", "seed", "context", "horizon", "split_seed", "device", "torch_version",
    "git_sha", "episodes", "windows", "probe", "self_check", "crossing", "margin",
    "ratio_probe", "ratio_raw", "ratio_free", "cosine", "displacement", "scale", "counts",
    "nonfinite",
}
"""The record's top-level keys, exactly. `nonfinite` is `write_record`'s map
and is on disk only; the live dict `trust_record` returns must NOT carry it
(`to_json_record` refuses a record that already does). `displacement` holds
the four norms the pooling's ratio of medians is built from."""


@pytest.fixture
def cell(tmp_path, small_buffer, capsys):
    """One real cell and the diagnostic the ladder writes for it.

    ~3 s: five training steps, the probe refit, and `diagnose_dynamics.main`
    with `--ks 1 3` (the sweep insists the horizon is among its ks). The
    ladder's own output is drained so a test reading stdout sees only
    `trust_horizon`'s.
    """
    out = tmp_path / "out"
    run_job(JOB, small_buffer, out, **JOB_KW)
    status = diagnose.main([
        "--out", str(out), "--data", str(small_buffer.root), "--device", "cpu",
        "--arms", JOB.arm, "--seeds", str(JOB.seed),
        "--context", str(CONTEXT), "--horizon", str(HORIZON), "--ks", "1", str(HORIZON),
    ])
    assert status == diagnose.EXIT_OK, "the fixture's diagnostic was not written cleanly"
    assert (out / DIAGNOSTIC).exists()
    capsys.readouterr()
    return types.SimpleNamespace(out=out, data=small_buffer.root)


def _argv(cell, *extra: str) -> list[str]:
    return [
        "--out", str(cell.out), "--data", str(cell.data), "--device", "cpu",
        "--arms", JOB.arm, "--seeds", str(JOB.seed), *extra,
    ]


def _doctor(path: Path, edit) -> None:
    """Edit one JSON file in place through plain `json`, so the record's
    `nonfinite` map is carried through untouched and nothing re-sanitises."""
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


def _never_refit(*args, **kwargs):
    raise AssertionError("fit_probes ran; this refusal must come before the refit")


# One fault per file a real drift would touch. Each is applied to a fresh
# fixture, and the ordering test applies two at once.
FAULTS = {
    "diagnostic": lambda cell: (cell.out / DIAGNOSTIC).unlink(),
    "split": lambda cell: _doctor(
        cell.out / RECORD,
        lambda r: r["episodes"].__setitem__("val", ["ep_000099_len00040.npz"]),
    ),
    "record": lambda cell: _doctor(
        cell.out / RECORD,
        lambda r: r["curves"]["rssm_position"].__setitem__(
            0, r["curves"]["rssm_position"][0] + 6.5
        ),
    ),
    "self_check": lambda cell: _doctor(
        cell.out / DIAGNOSTIC,
        lambda d: d["curves"]["reference_position"].__setitem__(
            0, d["curves"]["reference_position"][0] + 1e-3
        ),
    ),
}


# ---------------------------------------------------------------------------
# The parser and the statuses.
# ---------------------------------------------------------------------------


def test_the_parser_defaults_to_the_study_directory_and_reads_the_protocol_from_the_diagnostic():
    args = script._parser().parse_args([])
    assert args.out == Path("runs/m3_study_v2") and isinstance(args.out, Path)
    assert args.data == Path("data/my_way_home") and isinstance(args.data, Path)
    assert args.device == "mps"
    assert args.context is None and args.horizon is None, (
        "context and horizon default to whatever each cell's diagnostic was written at")
    assert args.arms == list(ARMS) == ["pixel_ae", "frozen_ssl", "random_vit"]
    assert args.seeds == [0, 1, 2]
    explicit = script._parser().parse_args(["--context", "2", "--horizon", "3", "--seeds", "1"])
    assert (explicit.context, explicit.horizon, explicit.seeds) == (2, 3, [1])
    with pytest.raises(SystemExit) as refused:
        script._parser().parse_args(["--arms", "cnn"])
    assert refused.value.code == 2


def test_the_reused_statuses_are_diagnose_dynamics_own_and_thirty_is_new():
    """11 / 12 / 14 carry `diagnose_dynamics.py`'s meanings, so a wrapper
    learns the same thing from either tool; 30 is this script's alone."""
    assert script.EXIT_OK == 0
    assert script.EXIT_NO_CHECKPOINTS == diagnose.EXIT_NO_CHECKPOINTS == 11
    assert script.EXIT_SPLIT_MISMATCH == diagnose.EXIT_SPLIT_MISMATCH == 12
    assert script.EXIT_RECORD_MISMATCH == diagnose.EXIT_RECORD_MISMATCH == 14
    assert script.EXIT_SELF_CHECK_FAILED == 30
    mine = {k: v for k, v in vars(script).items() if k.startswith("EXIT_")}
    assert set(mine) == {
        "EXIT_OK", "EXIT_NO_CHECKPOINTS", "EXIT_SPLIT_MISMATCH",
        "EXIT_RECORD_MISMATCH", "EXIT_SELF_CHECK_FAILED",
    }
    assert len(set(mine.values())) == len(mine)
    theirs = {v for k, v in vars(diagnose).items() if k.startswith("EXIT_")}
    assert 30 not in theirs and 1 not in mine.values() and 2 not in mine.values()


# ---------------------------------------------------------------------------
# self_check, on fabricated trajectories.
# ---------------------------------------------------------------------------


def _fabricated() -> tuple[Trajectories, dict]:
    """Two windows, two steps, two episodes. Window 0 is a PERFECT predictor
    (imagined position == truth, floor == truth); window 1 is a PERSISTENCE
    CLONE (imagined position held at p_hat(0)). Every number below is chosen
    so the two rows have different signatures on every channel.

    Truth: window 0 moves (6, 0) then (12, 0) from (0, 0); window 1 moves
    (0, 6) then (0, 12) from (10, 10). Both are >= 5 map units at every h,
    so `moved` is all True.

      model error      row 0 [0, 0]    row 1 [6, 12]
      persistence err  row 0 [6, 12]   row 1 [6, 12]
      -> crossing (probe): row 0 never (3 = H + 1); row 1 ties never cross (3)
      -> margin: row 0 [6, 12]; row 1 [0, 0]
      -> ratio_probe / ratio_raw: row 0 [1, 1]; row 1 [0, 0]
      -> cosine: row 0 [1, 1]; row 1 NaN (|d_hat| = 0), counted as zero_displacement
      -> displacement: probe_hat row 0 [6, 12], row 1 [0, 0]; probe_real [6, 12] on
         both rows (the floor is the truth); free_hat / free_true are the two
         embedding norms below, as they are

    Embedding channel, chosen to DIFFER from the probe channel's answers:
      D_hat  row 0 [0, 0]   row 1 [5, 5]      (distance of e_hat(h) to e(h))
      D_0    row 0 [3, 4]   row 1 [3, 4]      (distance of e_hat(0) to e(h))
      -> crossing (free): row 0 never (3); row 1 5 > 3 at h=1 -> 1
      ||e_hat(h) - e_hat(0)||  row 0 [2, 4]   row 1 [1, 2]
      ||e(h) - e(0)||          row 0 [2, 4]   row 1 [2, 4]
      -> ratio_free: row 0 [1, 1]; row 1 [0.5, 0.5]

    Scale correction (episode labels 0 and 1 -> fold A = row 0, fold B = row 1):
      alpha_a fit on row 0: |alpha * d - d| = |alpha - 1| * |d| -> 1.0 at both h
      alpha_b fit on row 1: d_hat = 0, the error is |p_hat0 - p| whatever alpha
        -> a tie over the whole grid -> the smallest alpha, 0.0 -> BOUNDARY
      score_a (fit A, scored on row 1 with alpha 1): [6, 12]
      score_b (fit B, scored on row 0 with alpha 0): [6, 12]
      held_out: row 0 with alpha_b = 0 -> [6, 12]; row 1 with alpha_a = 1 -> [6, 12]
      boundary [True, True]; folds_available True

    The diagnostic's curves are the means of the rows above -- reference
    [3, 6], persistence [6, 12] -- and its floor [1, 1] gives a positive
    persistence-to-floor band (12 - 1 = 11) at the final step: measurable.
    """
    true_at_context = np.array([[0.0, 0.0], [10.0, 10.0]])
    true_positions = np.array([[[6.0, 0.0], [12.0, 0.0]], [[10.0, 16.0], [10.0, 22.0]]])
    positions_at_context = true_at_context.copy()
    positions = np.array([[[6.0, 0.0], [12.0, 0.0]], [[10.0, 10.0], [10.0, 10.0]]])
    traj = Trajectories(
        positions=positions,
        positions_at_context=positions_at_context,
        positions_real=true_positions.copy(),
        true_positions=true_positions,
        true_at_context=true_at_context,
        embedding_distance_to_truth=np.array([[0.0, 0.0], [5.0, 5.0]]),
        embedding_persistence_distance=np.array([[3.0, 4.0], [3.0, 4.0]]),
        embedding_displacement=np.array([[2.0, 4.0], [1.0, 2.0]]),
        true_embedding_displacement=np.array([[2.0, 4.0], [2.0, 4.0]]),
        window_episode=np.array([0, 1]),
        windows_total=2,
        reference_position=np.array([3.0, 6.0]),
        persistence_position=np.array([6.0, 12.0]),
    )
    diagnostic = {
        "context": 2,
        "horizon": 2,
        "episodes": {"val": ["ep_a.npz", "ep_b.npz"]},
        "probe": {"embedding_selection_r2": 0.31},
        "windows": {"total": 2, "episode": [0, 1]},
        "curves": {
            "reference_position": [3.0, 6.0],
            "persistence_position": [6.0, 12.0],
            "floor_position": [1.0, 1.0],
        },
    }
    return traj, diagnostic


def _with_curve(diagnostic: dict, name: str, values) -> dict:
    doctored = json.loads(json.dumps(diagnostic))
    doctored["curves"][name] = list(values)
    return doctored


def _with_windows(diagnostic: dict, total=None, episode="keep") -> dict:
    doctored = json.loads(json.dumps(diagnostic))
    if total is not None:
        doctored["windows"]["total"] = total
    if episode != "keep":
        doctored["windows"]["episode"] = episode
    return doctored


def test_self_check_reads_zero_when_the_trajectories_reproduce_the_diagnostic():
    traj, diagnostic = _fabricated()
    check = script.self_check(traj, diagnostic)
    assert check.ok is True
    assert check.failures() == []
    assert check.reference_position_max_delta == 0.0
    assert check.persistence_position_max_delta == 0.0
    assert check.windows_total_match is True and check.windows_episode_match is True
    assert check.record() == {
        "reference_position_max_delta": 0.0,
        "persistence_position_max_delta": 0.0,
        "windows_total_match": True,
        "windows_episode_match": True,
        "ok": True,
    }


def test_self_check_names_the_curve_and_the_step_of_the_largest_delta():
    """The rule is `max |delta| == 0.0`, the ladder's `record_reproduction`
    rule: a delta of 2^-40 (exactly representable beside 6.0, so the
    difference is exactly 2^-40) fails it. Each curve is judged on its own:
    the persistence curve doctored alone fails too, and the step named is
    the 1-based horizon step of the largest delta."""
    traj, diagnostic = _fabricated()
    tiny = 2.0 ** -40
    reference_only = script.self_check(
        traj, _with_curve(diagnostic, "reference_position", [3.0, 6.0 + tiny])
    )
    assert reference_only.ok is False
    assert reference_only.reference_position_max_delta == tiny
    assert reference_only.reference_position_step == 2
    assert reference_only.persistence_position_max_delta == 0.0
    assert reference_only.record()["ok"] is False
    (message,) = reference_only.failures()
    assert "reference_position" in message and "step 2" in message

    persistence_only = script.self_check(
        traj, _with_curve(diagnostic, "persistence_position", [6.5, 12.0])
    )
    assert persistence_only.ok is False
    assert persistence_only.reference_position_max_delta == 0.0
    assert persistence_only.persistence_position_max_delta == 0.5
    assert persistence_only.persistence_position_step == 1
    (message,) = persistence_only.failures()
    assert "persistence_position" in message and "step 1" in message


def test_self_check_judges_the_rows_the_record_is_built_from_not_the_passs_own_curve():
    """Spec 2.3: the mean over windows of |p_hat(h) - p(h)| and of
    |p_hat(0) - p(h)| -- computed HERE from the rows the record is built
    from -- must equal the diagnostic's curves. `Trajectories.reference_position`
    is the pass's own reduction of the same rows and equals this bitwise on a
    clean run (Task 3 pins it), but it is not what is judged: a regression
    that kept the curve right and the rows wrong would otherwise write a
    record. Window 0's imagined position at step 2 is moved by one map unit
    while the pass's curve stays [3, 6]: the row mean there is (1 + 12) / 2
    = 6.5, so the check reads 0.5 at step 2 and fails. Then window 1's anchor
    is moved by (0, 2) while the pass's persistence curve stays [6, 12]: its
    persistence errors become [4, 10], the means [5, 11], a delta of 1.0 at
    both steps, the first of which is named."""
    traj, diagnostic = _fabricated()
    positions = traj.positions.copy()
    positions[0, 1, 0] += 1.0
    rows_moved = script.self_check(dataclasses.replace(traj, positions=positions), diagnostic)
    assert rows_moved.ok is False
    assert rows_moved.reference_position_max_delta == 0.5
    assert rows_moved.reference_position_step == 2
    assert rows_moved.persistence_position_max_delta == 0.0

    at_context = traj.positions_at_context.copy()
    at_context[1, 1] += 2.0
    anchor_moved = script.self_check(
        dataclasses.replace(traj, positions_at_context=at_context), diagnostic
    )
    assert anchor_moved.ok is False
    assert anchor_moved.reference_position_max_delta == 0.0
    assert anchor_moved.persistence_position_max_delta == 1.0
    assert anchor_moved.persistence_position_step == 1


def test_self_check_refuses_a_curve_of_the_wrong_length_rather_than_raising():
    """A diagnostic written at another horizon has a curve of another length;
    `np.abs(a - b)` on mismatched shapes raises, and a traceback is exit 1 --
    the status reserved for a defect, not for a record that does not match."""
    traj, diagnostic = _fabricated()
    check = script.self_check(
        traj, _with_curve(diagnostic, "reference_position", [3.0, 6.0, 9.0])
    )
    assert check.ok is False
    assert check.reference_position_max_delta == math.inf
    assert check.reference_position_step == 0
    (message,) = check.failures()
    assert "reference_position" in message and "length" in message


def test_self_check_refuses_a_window_count_or_episode_index_that_differs():
    traj, diagnostic = _fabricated()
    wrong_total = script.self_check(traj, _with_windows(diagnostic, total=3))
    assert wrong_total.windows_total_match is False
    assert wrong_total.windows_episode_match is True
    assert wrong_total.ok is False
    assert any("windows.total" in m for m in wrong_total.failures())

    wrong_index = script.self_check(traj, _with_windows(diagnostic, episode=[0, 0]))
    assert wrong_index.windows_total_match is True
    assert wrong_index.windows_episode_match is False
    assert wrong_index.ok is False
    assert any("windows.episode" in m for m in wrong_index.failures())

    # The diagnostic writes `null` when the ladder carried no clustering; a
    # trust record cannot be clustered on nothing, so that is a mismatch too.
    no_index = script.self_check(traj, _with_windows(diagnostic, episode=None))
    assert no_index.windows_episode_match is False and no_index.ok is False


# ---------------------------------------------------------------------------
# trust_record and write_trust_record, on the fabricated trajectories.
# ---------------------------------------------------------------------------


def test_trust_record_wires_every_channel_the_way_the_spec_names_it():
    traj, diagnostic = _fabricated()
    record = script.trust_record(
        "frozen_ssl", 2, traj, diagnostic, context=2, horizon=2, device=torch.device("cpu")
    )
    assert set(record) == EXPECTED_KEYS - {"nonfinite"}, (
        "the live record must not carry `nonfinite`; write_record adds it")

    for key, value in {
        "arm": "frozen_ssl", "seed": 2, "context": 2, "horizon": 2, "split_seed": 0,
        "device": "cpu", "torch_version": torch.__version__, "git_sha": study.git_sha(),
        "episodes": {"val": ["ep_a.npz", "ep_b.npz"]},
        "windows": {"total": 2, "episode": [0, 1]},
        "probe": {"selection_r2": 0.31, "measurable": True},
        "self_check": {
            "reference_position_max_delta": 0.0, "persistence_position_max_delta": 0.0,
            "windows_total_match": True, "windows_episode_match": True, "ok": True,
        },
    }.items():
        assert record[key] == value, key

    equal = np.testing.assert_array_equal  # NaN == NaN by position
    equal(record["crossing"]["probe"], [3.0, 3.0])
    equal(record["crossing"]["free"], [3.0, 1.0])
    equal(record["margin"], [[6.0, 12.0], [0.0, 0.0]])
    equal(record["ratio_probe"], [[1.0, 1.0], [0.0, 0.0]])
    equal(record["ratio_raw"], [[1.0, 1.0], [0.0, 0.0]])
    equal(record["ratio_free"], [[1.0, 1.0], [0.5, 0.5]])
    equal(record["cosine"], [[1.0, 1.0], [np.nan, np.nan]])

    # The four norms the pooling's ratio of medians is built from: the
    # probe's imagined and REAL displacements (the floor is the truth here,
    # so row 1's real displacement is its true one), and the two embedding
    # norms as the pass reduced them.
    displacement = record["displacement"]
    assert set(displacement) == {"probe_hat", "probe_real", "free_hat", "free_true"}
    equal(displacement["probe_hat"], [[6.0, 12.0], [0.0, 0.0]])
    equal(displacement["probe_real"], [[6.0, 12.0], [6.0, 12.0]])
    equal(displacement["free_hat"], [[2.0, 4.0], [1.0, 2.0]])
    equal(displacement["free_true"], [[2.0, 4.0], [2.0, 4.0]])

    scale = record["scale"]
    assert set(scale) == {
        "alpha_a", "alpha_b", "score_a", "score_b", "held_out", "boundary", "folds_available",
    }
    equal(scale["alpha_a"], [1.0, 1.0])
    equal(scale["alpha_b"], [0.0, 0.0])
    equal(scale["score_a"], [6.0, 12.0])
    equal(scale["score_b"], [6.0, 12.0])
    equal(scale["held_out"], [[6.0, 12.0], [6.0, 12.0]])
    equal(scale["boundary"], [True, True])
    assert scale["folds_available"] is True

    counts = record["counts"]
    assert set(counts) == {"not_moved", "zero_displacement", "never_moved"}
    equal(counts["not_moved"], [0, 0])
    equal(counts["zero_displacement"], [1, 1])
    assert counts["never_moved"] == 0


def test_measurability_is_the_persistence_to_floor_band_at_the_final_step():
    """The ladder's `probe_is_measurable` rule, asked with no null band: the
    band is `persistence - floor` at the LAST step and must be positive. A
    floor that equals persistence there is a zero band -- not measurable --
    and the band is read off the persistence curve, never the reference."""
    traj, diagnostic = _fabricated()

    def measurable(floor):
        return script.trust_record(
            "frozen_ssl", 2, traj, _with_curve(diagnostic, "floor_position", floor),
            context=2, horizon=2, device="cpu",
        )["probe"]["measurable"]

    # persistence [6, 12]: band 12 - 8 = 4 > 0. (Off the reference curve,
    # [3, 6], it would read 6 - 8 < 0.)
    assert measurable([1.0, 8.0]) is True
    assert measurable([1.0, 12.0]) is False   # band exactly 0
    assert measurable([1.0, 13.0]) is False   # floor above persistence


def test_write_trust_record_names_both_the_arm_and_the_seed_and_round_trips_nan(tmp_path):
    """Written through `write_record`: atomic, strict JSON, `null` for a NaN
    with the token map that restores it -- never a bare `NaN` token."""
    record = {"arm": "random_vit", "seed": 1, "margin": [[float("nan"), 2.0]]}
    path = script.write_trust_record(tmp_path, record)
    assert path == tmp_path / "trust_random_vit_seed1.json"
    assert script.trust_record_path(tmp_path, "pixel_ae", 0) == tmp_path / "trust_pixel_ae_seed0.json"
    raw = json.loads(path.read_text())
    assert raw["margin"][0][0] is None and raw["nonfinite"] == {"margin.0.0": "nan"}
    back = load_record(path)
    assert math.isnan(back["margin"][0][0]) and back["margin"][0][1] == 2.0


# ---------------------------------------------------------------------------
# main, on the real tiny cell.
# ---------------------------------------------------------------------------


def test_a_clean_cell_exits_ok_and_writes_a_record_with_every_key_and_shape(cell):
    """Explicit `--context 2 --horizon 3`, EQUAL to the diagnostic's: a value
    that matches is not a mismatch. (The next test passes neither flag and
    exercises the defaults.)"""
    assert script.main(_argv(cell, "--context", str(CONTEXT), "--horizon", str(HORIZON))) == script.EXIT_OK
    path = cell.out / TRUST
    assert path.exists()
    record = load_record(path)
    assert set(record) == EXPECTED_KEYS

    assert (record["arm"], record["seed"]) == (JOB.arm, JOB.seed)
    assert (record["context"], record["horizon"]) == (CONTEXT, HORIZON)
    assert record["split_seed"] == 0
    assert record["device"] == "cpu"
    assert record["torch_version"] == torch.__version__
    assert record["git_sha"] == study.git_sha()
    study_record = load_record(job_record_path(cell.out, JOB))
    assert record["episodes"]["val"] == study_record["episodes"]["val"]
    assert len(record["episodes"]["val"]) == 1, "six episodes at val_fraction 0.2 hold out one"
    assert record["windows"] == {"total": N_WINDOWS, "episode": [0] * N_WINDOWS}

    assert len(record["crossing"]["probe"]) == N_WINDOWS
    assert len(record["crossing"]["free"]) == N_WINDOWS
    for key in ("margin", "ratio_probe", "ratio_raw", "ratio_free", "cosine"):
        assert np.asarray(record[key], dtype=float).shape == (N_WINDOWS, HORIZON), key
    assert set(record["displacement"]) == {"probe_hat", "probe_real", "free_hat", "free_true"}
    for key in ("probe_hat", "probe_real", "free_hat", "free_true"):
        assert np.asarray(record["displacement"][key], dtype=float).shape == (N_WINDOWS, HORIZON), key
    scale = record["scale"]
    assert set(scale) == {
        "alpha_a", "alpha_b", "score_a", "score_b", "held_out", "boundary", "folds_available",
    }
    for key in ("alpha_a", "alpha_b", "score_a", "score_b", "boundary"):
        assert len(scale[key]) == HORIZON, key
    assert np.asarray(scale["held_out"], dtype=float).shape == (N_WINDOWS, HORIZON)
    assert set(record["counts"]) == {"not_moved", "zero_displacement", "never_moved"}
    assert len(record["counts"]["not_moved"]) == HORIZON
    assert len(record["counts"]["zero_displacement"]) == HORIZON

    diagnostic = load_record(cell.out / DIAGNOSTIC)
    assert set(record["probe"]) == {"selection_r2", "measurable"}
    assert record["probe"]["selection_r2"] == diagnostic["probe"]["embedding_selection_r2"]
    band = diagnostic["curves"]["persistence_position"][-1] - diagnostic["curves"]["floor_position"][-1]
    assert record["probe"]["measurable"] is (band > 0.0)


def test_the_self_check_reads_zero_the_single_episode_is_disclosed_and_h1_never_moves(cell, capsys):
    """No `--context` / `--horizon`: both come from the diagnostic."""
    assert script.main(_argv(cell)) == script.EXIT_OK
    out = capsys.readouterr().out
    record = load_record(cell.out / TRUST)

    assert record["self_check"] == {
        "reference_position_max_delta": 0.0,
        "persistence_position_max_delta": 0.0,
        "windows_total_match": True,
        "windows_episode_match": True,
        "ok": True,
    }, "same windows, same rollout, same refit probe -- or this is not what the ladder measured"

    # ONE validation episode: no second fold, so the scale correction is NaN
    # and says so, and the run discloses it rather than printing a number.
    assert record["scale"]["folds_available"] is False
    for key in ("alpha_a", "alpha_b", "score_a", "score_b"):
        assert all(math.isnan(v) for v in record["scale"][key]), key
    assert all(math.isnan(v) for row in record["scale"]["held_out"] for v in row)
    assert record["scale"]["boundary"] == [False] * HORIZON
    assert f"{N_WINDOWS} windows from 1 validation episode;" in out
    assert "folds UNAVAILABLE" in out
    assert "random_vit seed 1" in out

    # pos_x = 3t, pos_y = -2t: h * sqrt(13) = 3.61 at h=1 (< 5, not moved in
    # any window), 7.21 and 10.82 after (moved in every window).
    assert record["counts"]["not_moved"] == [N_WINDOWS, 0, 0]
    assert record["counts"]["never_moved"] == 0
    for key in ("ratio_probe", "ratio_raw", "ratio_free", "cosine"):
        assert all(math.isnan(row[0]) for row in record[key]), f"{key} is defined at h=1"
    # Every window first moves at h=2, so a crossing is 2, 3, or never (4);
    # never NaN. The margin is unmasked and finite everywhere.
    assert all(c in (2.0, 3.0, 4.0) for c in record["crossing"]["probe"])
    assert all(c in (2.0, 3.0, 4.0) for c in record["crossing"]["free"])
    assert all(math.isfinite(v) for row in record["margin"] for v in row)


def test_a_missing_diagnostic_is_exit_11_naming_the_file(cell, capsys):
    FAULTS["diagnostic"](cell)
    assert script.main(_argv(cell)) == script.EXIT_NO_CHECKPOINTS
    out = capsys.readouterr().out
    assert DIAGNOSTIC in out and "random_vit seed 1" in out
    assert not (cell.out / TRUST).exists()


def test_a_doctored_reference_curve_is_exit_30_and_no_record_is_written(cell, capsys):
    """One value of `curves.reference_position` moved by 1e-3: the trust pass
    no longer reproduces the ladder's curve, the cell, the curve and the step
    are named, and NOTHING is written -- a record beside a failed self-check
    would be pooled by the next part as if it measured what the ladder did."""
    FAULTS["self_check"](cell)
    assert script.main(_argv(cell)) == script.EXIT_SELF_CHECK_FAILED
    out = capsys.readouterr().out
    assert "SELF-CHECK FAILED for random_vit seed 1" in out
    assert "reference_position" in out and "step 1" in out
    assert not (cell.out / TRUST).exists()


def test_a_context_or_horizon_that_disagrees_with_the_diagnostic_is_refused_before_the_refit(
    cell, capsys, monkeypatch
):
    """Refused as `diagnose_dynamics.py` refuses it -- a RECORD MISMATCH, 14:
    a rollout at another context cannot reproduce the record's curve -- but
    up front, against the protocol the diagnostic says it was written at,
    not twenty seconds later after a probe refit at the wrong depth."""
    monkeypatch.setattr(script, "fit_probes", _never_refit)
    assert script.main(_argv(cell, "--context", "3")) == script.EXIT_RECORD_MISMATCH
    out = capsys.readouterr().out
    assert "RECORD MISMATCH for random_vit seed 1" in out
    assert "--context 3" in out and f"({CONTEXT})" in out
    assert script.main(_argv(cell, "--horizon", "4")) == script.EXIT_RECORD_MISMATCH
    out = capsys.readouterr().out
    assert "--horizon 4" in out and f"({HORIZON})" in out
    assert not (cell.out / TRUST).exists()


def test_a_split_that_is_not_the_records_is_exit_12_before_the_refit(cell, capsys, monkeypatch):
    monkeypatch.setattr(script, "fit_probes", _never_refit)
    FAULTS["split"](cell)
    assert script.main(_argv(cell)) == script.EXIT_SPLIT_MISMATCH
    out = capsys.readouterr().out
    assert "SPLIT MISMATCH for random_vit seed 1" in out
    assert "ep_000099_len00040.npz" in out, "the record's names are printed beside the split's"
    assert not (cell.out / TRUST).exists()


def test_a_study_record_the_rollout_no_longer_reproduces_is_exit_14(cell, capsys):
    """`curves.rssm_position` moved by 6.5 map units -- the size of the cpu
    miss `diagnose_dynamics.py` documents -- and the environment is named.
    The spec's 'wrong device' case cannot be staged on a cpu-trained cell;
    the doctored record produces the same status through the same
    `_max_delta` branch, and Task 7's pre-flight item 6 covers the device on
    the real run."""
    FAULTS["record"](cell)
    assert script.main(_argv(cell)) == script.EXIT_RECORD_MISMATCH
    out = capsys.readouterr().out
    assert "RECORD MISMATCH for random_vit seed 1" in out
    assert "device=cpu" in out
    assert not (cell.out / TRUST).exists()


@pytest.mark.parametrize(
    "earlier, later, status",
    [
        ("split", "self_check", "EXIT_SPLIT_MISMATCH"),
        ("record", "self_check", "EXIT_RECORD_MISMATCH"),
        ("diagnostic", "split", "EXIT_NO_CHECKPOINTS"),
    ],
)
def test_the_checks_are_judged_in_the_order_11_12_14_30(cell, earlier, later, status):
    """Two faults at once, and the EARLIER check's status is the one reported:
    the order separates a missing cell from a wrong split from an environment
    difference from a trust-pass defect, and a later status reported for an
    earlier fault sends the reader to the wrong place."""
    FAULTS[earlier](cell)
    FAULTS[later](cell)
    assert script.main(_argv(cell)) == getattr(script, status)
    assert not (cell.out / TRUST).exists()


def test_arms_and_seeds_select_the_cells_and_an_absent_cell_is_exit_11(cell, capsys):
    """Every REQUESTED cell must be present -- the readings need all of them --
    so the default nine on a directory holding one cell is 11, naming the
    first absent cell in `ARMS x seeds` order, before any refit."""
    base = ["--out", str(cell.out), "--data", str(cell.data), "--device", "cpu"]
    assert script.main(base) == script.EXIT_NO_CHECKPOINTS
    assert "pixel_ae seed 0" in capsys.readouterr().out
    assert script.main(base + ["--arms", "random_vit"]) == script.EXIT_NO_CHECKPOINTS
    assert "random_vit seed 0" in capsys.readouterr().out
    assert not (cell.out / TRUST).exists()
    assert script.main(base + ["--arms", "random_vit", "--seeds", "1"]) == script.EXIT_OK
    assert (cell.out / TRUST).exists()
    with pytest.raises(SystemExit) as refused:
        script.main(base + ["--arms", "cnn"])
    assert refused.value.code == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_trust_horizon_script.py -q -p no:cacheprovider 2>&1 | tail -5`

Expected: one collection error, because `_load_script("trust_horizon")` executes a file that does not exist yet:

```
E   FileNotFoundError: [Errno 2] No such file or directory: '/Users/raphaelchen/Desktop/csgo-bot/scripts/trust_horizon.py'
1 error in 1.5s
```

(If Task 3 has not landed, the import of `Trajectories` fails first with `ImportError: cannot import name 'Trajectories' from 'mbfps.eval.diagnostics'` -- red either way, but Step 4 needs Task 3.)

- [ ] **Step 3: Write `scripts/trust_horizon.py`**

```python
"""The trust horizon over the M3c cells, part 1: load, check, one reference pass, one record per cell.

M3c's ladder records store per-step MEAN error curves. Whether `random_vit`
loses least at h=45 because its prior drifts slowly -- stays near persistence,
which the metric rewards -- or because `frozen_ssl` moves the wrong way cannot
be read off a mean: magnitude and direction are per-window quantities. This
script re-runs the ladder's canonical reference pass on each frozen checkpoint
with the per-window trajectories KEPT (`reference_trajectories`), and writes
one `trust_<arm>_seed<n>.json` per cell holding, per window and per horizon
step, the crossing step in both channels, the persistence margin, the three
displacement ratios, the direction cosine, and the held-out scale-corrected
error with its fold alphas -- the quantities of the M3d design, section 2.2.
The pooling over the nine records and the two readings are the second part
of this script and come after the per-cell loop in `main`.

LOADING IS `diagnose_dynamics.py`'S, NOT A SECOND COPY. The checkpoint path
and its arm/seed validation (`load_checkpoint_model`), the diagnostic's path
and the probe-measurability rule are imported from that script by path -- the
way every script test loads a script -- because each is a rule the ladder
already applied to these nine cells, and a duplicate here could drift from
the ladder's while both kept passing, leaving the two tools reading one cell
through two loaders. Everything else on the loading path (`episode_split`,
`fit_probes`, `evaluate_rollout`, the record I/O) lives in `src/` and is
imported from there. The probe is REFIT, exactly as the ladder refits it --
`fit_probes` on the train split at the cell's seed and protocol -- because no
probe weights exist on disk; the refit is part of what the self-check pins.

THE CHECKS RUN IN A FIXED ORDER, PER CELL, BEFORE ITS RECORD IS WRITTEN,
and each has its own status:

  EXIT_NO_CHECKPOINTS (11)    -- a REQUESTED cell lacks its checkpoint, its
    study record or its diagnostic. Every requested cell is loaded before any
    probe refit, so a run asking for nine and finding eight stops in a
    second, not twenty minutes in. (Stricter than `diagnose_dynamics.py`,
    which runs whatever cells it finds: the readings this script exists for
    need every arm.)
  EXIT_SPLIT_MISMATCH (12)    -- the split by name is not the record's.
  EXIT_RECORD_MISMATCH (14)   -- `evaluate_rollout` no longer reproduces the
    study record's `curves.rssm_position` (an ENVIRONMENT difference: on cpu
    the mps-trained cells miss by 6-12 map units). A `--context` or
    `--horizon` that disagrees with the protocol the diagnostic records is
    refused under this status too, UP FRONT: a rollout at another protocol
    cannot reproduce the record, which is how `diagnose_dynamics.py` surfaces
    the same mismatch -- after its refit, twenty seconds later.
  EXIT_SELF_CHECK_FAILED (30) -- the trust pass's own mean curves are not
    bitwise the diagnostic's `curves.reference_position` and
    `curves.persistence_position` (`max |delta| == 0.0`, the ladder's own
    `record_reproduction` rule), or its windows are not the diagnostic's.
    Same windows, same rollout, same refit probe, or this is not measuring
    what the ladder measured -- and the cell's record is NOT written, because
    the second part pools every record it finds.

11, 12 and 14 carry `diagnose_dynamics.py`'s meanings ON PURPOSE -- a wrapper
reading the status learns the same thing from either tool -- and 30 is new,
distinct from every other tool's (1-23) and from argparse's own 2. A
mislabelled checkpoint (that script's 16) and an unsupported device (its 17)
have no status here and surface as the library's own exceptions.
"""

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.aggregate import SEEDS
from mbfps.eval.diagnostics import Trajectories, reference_trajectories
from mbfps.eval.probe import fit_probes
from mbfps.eval.rollout import evaluate_rollout
from mbfps.eval.study import (
    SPLIT_SEED,
    StudyJob,
    git_sha,
    job_record_path,
    load_record,
    write_record,
)
from mbfps.eval.trust import (
    crossing_step,
    displacement_decomposition,
    embedding_ratio,
    moved_mask,
    persistence_margin,
    scale_corrected_error,
)
from mbfps.models.encoders import encoder_backbone
from mbfps.utils.config import ARMS, get_config
from mbfps.utils.device import get_device


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script.

    Resolved against THIS file, so it works run as `python scripts/...` and
    loaded by a test alike; a bare `import diagnose_dynamics` would need the
    scripts directory on `sys.path`, which only the first of those provides.
    """
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_trust_horizon", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_diagnose_dynamics = _sibling("diagnose_dynamics")
checkpoint_path = _diagnose_dynamics.checkpoint_path
load_checkpoint_model = _diagnose_dynamics.load_checkpoint_model
diagnostic_record_path = _diagnose_dynamics.diagnostic_record_path
probe_is_measurable = _diagnose_dynamics.probe_is_measurable

EXIT_OK = 0
EXIT_NO_CHECKPOINTS = 11
EXIT_SPLIT_MISMATCH = 12
EXIT_RECORD_MISMATCH = 14
EXIT_SELF_CHECK_FAILED = 30
"""11, 12 and 14 are `diagnose_dynamics.py`'s, with its meanings. 30 is new
and is in no other tool's range: run_study 1/3-6/23, report_study 7-10,
spike 10, diagnose 11-17, pool 18-22, argparse 2, a traceback 1."""


class CellMissing(FileNotFoundError):
    """A requested cell lacks its checkpoint, its study record or its
    diagnostic -- the ONE error `main` maps to `EXIT_NO_CHECKPOINTS`.

    A type of its own so `main` catches exactly this: `load_record` raises the
    base `FileNotFoundError` for a file that vanished between the existence
    check and the read, and that is a race worth a traceback, not a status.
    """


@dataclass(frozen=True)
class Cell:
    """One cell's three files, read. The model is built from `checkpoint` by
    `load_checkpoint_model` once the device is known."""

    arm: str
    seed: int
    checkpoint: Path
    record: dict
    diagnostic: dict


def load_cell(out_dir: Path, arm: str, seed: int) -> Cell:
    """The checkpoint, the study record and the diagnostic of one cell, or
    `CellMissing` naming the first of the three that is not there."""
    out_dir = Path(out_dir)
    checkpoint = checkpoint_path(out_dir, arm, seed)
    record_path = job_record_path(out_dir, StudyJob(arm=arm, seed=seed))
    diagnostic_path = diagnostic_record_path(out_dir, arm, seed)
    for kind, path in (
        ("checkpoint", checkpoint),
        ("study record", record_path),
        ("diagnostic", diagnostic_path),
    ):
        if not path.exists():
            raise CellMissing(f"{arm} seed {seed}: no {kind} at {path}")
    return Cell(
        arm=arm,
        seed=seed,
        checkpoint=checkpoint,
        record=load_record(record_path),
        diagnostic=load_record(diagnostic_path),
    )


def _max_delta(ours, theirs) -> tuple[float, int]:
    """`max |ours - theirs|` and the 1-based step it is at; `(inf, 0)` when
    the two are not even the same shape -- a curve written at another horizon
    is a mismatch to report, not a broadcasting traceback."""
    ours = np.asarray(ours, dtype=float)
    theirs = np.asarray(theirs, dtype=float)
    if ours.shape != theirs.shape:
        return float("inf"), 0
    delta = np.abs(ours - theirs)
    return float(delta.max()), int(delta.argmax()) + 1


@dataclass(frozen=True)
class SelfCheck:
    """Does the trust pass reproduce the diagnostic it is about to be read
    beside. Two exact curve equalities (the ladder's `record_reproduction`
    rule, `max |delta| == 0.0`) and two window identities."""

    reference_position_max_delta: float
    reference_position_step: int
    persistence_position_max_delta: float
    persistence_position_step: int
    windows_total_match: bool
    windows_episode_match: bool

    def failures(self) -> list[str]:
        out = []
        if self.reference_position_max_delta != 0.0:
            out.append(_curve_failure(
                "reference_position", self.reference_position_max_delta, self.reference_position_step
            ))
        if self.persistence_position_max_delta != 0.0:
            out.append(_curve_failure(
                "persistence_position", self.persistence_position_max_delta, self.persistence_position_step
            ))
        if not self.windows_total_match:
            out.append("windows.total is not the diagnostic's")
        if not self.windows_episode_match:
            out.append("windows.episode is not the diagnostic's")
        return out

    @property
    def ok(self) -> bool:
        return not self.failures()

    def record(self) -> dict:
        return {
            "reference_position_max_delta": self.reference_position_max_delta,
            "persistence_position_max_delta": self.persistence_position_max_delta,
            "windows_total_match": self.windows_total_match,
            "windows_episode_match": self.windows_episode_match,
            "ok": self.ok,
        }


def _curve_failure(name: str, delta: float, step: int) -> str:
    if step == 0:
        return f"{name} has a different length from the diagnostic's"
    return f"{name} differs from the diagnostic's by max |delta| {delta:.3e} at step {step}"


def self_check(traj: Trajectories, diagnostic: dict) -> SelfCheck:
    """The two curves are recomputed HERE from the rows the record is built
    from -- the mean over windows of |p_hat(h) - p(h)| and of |p_hat(0) -
    p(h)|, spec 2.3 -- and compared with the diagnostic's. They are NOT read
    off `traj.reference_position` / `traj.persistence_position`: those are
    the pass's own reduction of the same rows (Task 3 pins them bitwise
    equal to these on a clean run), and judging them would let a regression
    that kept the curve right and the rows wrong write a record."""
    curves = diagnostic["curves"]
    reference_rows = _distance(traj.positions, traj.true_positions)
    persistence_rows = _distance(traj.positions_at_context[:, None, :], traj.true_positions)
    reference, reference_step = _max_delta(
        reference_rows.mean(axis=0), curves["reference_position"]
    )
    persistence, persistence_step = _max_delta(
        persistence_rows.mean(axis=0), curves["persistence_position"]
    )
    windows = diagnostic["windows"]
    # `null` when the ladder carried no clustering: a trust record cannot be
    # clustered on nothing, so that is a mismatch, never `range(n)`.
    episode = windows["episode"]
    return SelfCheck(
        reference_position_max_delta=reference,
        reference_position_step=reference_step,
        persistence_position_max_delta=persistence,
        persistence_position_step=persistence_step,
        windows_total_match=int(traj.windows_total) == int(windows["total"]),
        windows_episode_match=(
            episode is not None
            and [int(e) for e in traj.window_episode] == [int(e) for e in episode]
        ),
    )


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2-D Euclidean distance over the last axis -- `position_error`'s norm."""
    return np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), axis=-1)


def trust_record(arm, seed, traj: Trajectories, diagnostic: dict, *, context, horizon, device) -> dict:
    """The LIVE record for one cell: numpy arrays and real NaNs. `write_record`
    sanitises it (`null` plus the `nonfinite` map) on the way to disk, so
    this dict must not carry a `nonfinite` key of its own.

    Every quantity goes through the pure functions of `mbfps.eval.trust`;
    this function only names which of the pass's arrays is which. Both
    channels' errors are the same shape and both crossings are the same
    call, so the wiring -- the probe's positions to the probe crossing, the
    embedding distances to the free crossing, `positions_real` as the third
    argument of the decomposition and never `positions` -- is what the
    fabricated-trajectory tests pin by value.
    """
    moved = moved_mask(traj.true_positions, traj.true_at_context)
    model_err = _distance(traj.positions, traj.true_positions)
    persist_err = _distance(traj.positions_at_context[:, None, :], traj.true_positions)
    decomposition = displacement_decomposition(
        traj.positions, traj.true_positions, traj.positions_at_context,
        traj.true_at_context, traj.positions_real, moved,
    )
    scale = scale_corrected_error(
        traj.positions, traj.true_positions, traj.positions_at_context, moved, traj.window_episode,
    )
    check = self_check(traj, diagnostic)
    curves = diagnostic["curves"]
    return {
        "arm": arm,
        "seed": seed,
        "context": int(context),
        "horizon": int(horizon),
        "split_seed": SPLIT_SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "git_sha": git_sha(),
        "episodes": {"val": list(diagnostic["episodes"]["val"])},
        "windows": {
            "total": int(traj.windows_total),
            "episode": [int(e) for e in traj.window_episode],
        },
        "probe": {
            "selection_r2": float(diagnostic["probe"]["embedding_selection_r2"]),
            # The ladder's rule, asked with no null band: the persistence-to-
            # floor band at the FINAL step must be positive, and it is read off
            # the persistence curve, never the reference.
            "measurable": bool(probe_is_measurable(
                {
                    "persistence": curves["persistence_position"][-1],
                    "floor": curves["floor_position"][-1],
                },
                widest_se=0.0,
            )),
        },
        "self_check": check.record(),
        "crossing": {
            "probe": crossing_step(model_err, persist_err, moved),
            "free": crossing_step(
                traj.embedding_distance_to_truth, traj.embedding_persistence_distance, moved
            ),
        },
        "margin": persistence_margin(model_err, persist_err),
        "ratio_probe": decomposition.ratio_probe,
        "ratio_raw": decomposition.ratio_raw,
        "ratio_free": embedding_ratio(
            traj.embedding_displacement, traj.true_embedding_displacement, moved
        ),
        "cosine": decomposition.cosine,
        # The numerator and denominator series of the two ratios, kept
        # beside the per-window ratios: the pooling's estimand is a ratio of
        # MEDIANS (spec 3.1), which the ratios alone cannot recover.
        "displacement": {
            "probe_hat": _distance(traj.positions, traj.positions_at_context[:, None, :]),
            "probe_real": _distance(traj.positions_real, traj.positions_at_context[:, None, :]),
            "free_hat": np.asarray(traj.embedding_displacement, dtype=float),
            "free_true": np.asarray(traj.true_embedding_displacement, dtype=float),
        },
        "scale": {
            "alpha_a": scale.alpha_a,
            "alpha_b": scale.alpha_b,
            "score_a": scale.score_a,
            "score_b": scale.score_b,
            "held_out": scale.held_out,
            "boundary": scale.boundary,
            "folds_available": bool(scale.folds_available),
        },
        "counts": {
            "not_moved": (~moved).sum(axis=0),
            "zero_displacement": decomposition.zero_displacement.sum(axis=0),
            # Windows that never move within the horizon: their crossing is
            # NaN and they are counted here, not pooled.
            "never_moved": int((~moved.any(axis=1)).sum()),
        },
    }


def trust_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    """One file per cell, named by BOTH the arm and the seed -- see
    `study.job_record_path` for what a colliding name costs."""
    return Path(out_dir) / f"trust_{arm}_seed{seed}.json"


def write_trust_record(out_dir: Path, record: dict) -> Path:
    """Atomic and strict-JSON, through `write_record`: a NaN ratio is `null`
    plus its token, never a bare `NaN` the pooling reader would choke on."""
    path = trust_record_path(out_dir, record["arm"], record["seed"])
    write_record(path, record)
    return path


def protocol_mismatch(args, diagnostic: dict) -> str | None:
    """The flag that disagrees with the protocol the diagnostic was written
    at, or None. A flag left at None takes the diagnostic's value."""
    for flag in ("context", "horizon"):
        asked = getattr(args, flag)
        recorded = int(diagnostic[flag])
        if asked is not None and asked != recorded:
            return f"--{flag} {asked} is not the {flag} the diagnostic was written at ({recorded})"
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/m3_study_v2"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    # None -> each cell's diagnostic says what it was written at. A value
    # that disagrees with the diagnostic is refused, see `protocol_mismatch`.
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    return parser


def _run_cell(args, cell: Cell, device, train, val) -> tuple[int, dict | None]:
    """One cell: 12, the protocol (14), the refit, the reproduction (14), the
    reference pass, 30, the record. Returns `(EXIT_OK, record)` once the
    record is written, `(status, None)` on any refusal: `main` keeps the live
    records for the pooling that follows the loop."""
    arm, seed = cell.arm, cell.seed
    names = [p.name for p in val]
    if cell.record["episodes"]["val"] != names:
        print(
            f"\nSPLIT MISMATCH for {arm} seed {seed}: the held-out episodes are not "
            "the ones the record was scored on.\n"
            f"  split:  {names}\n  record: {cell.record['episodes']['val']}"
        )
        return EXIT_SPLIT_MISMATCH, None

    mismatch = protocol_mismatch(args, cell.diagnostic)
    if mismatch is not None:
        print(
            f"\nRECORD MISMATCH for {arm} seed {seed}: {mismatch}. A rollout at "
            "another protocol cannot reproduce the record's curve, so nothing is refit."
        )
        return EXIT_RECORD_MISMATCH, None
    context = int(cell.diagnostic["context"]) if args.context is None else args.context
    horizon = int(cell.diagnostic["horizon"]) if args.horizon is None else args.horizon

    cfg = get_config(arm, seed=seed, device=args.device)
    model = load_checkpoint_model(args.out, arm, seed, cfg, device)
    backbone = encoder_backbone(cfg.encoder)
    # The probe is REFIT at the rollout's own context/horizon and at the
    # cell's seed, exactly as the ladder refit it: it is applied to latents
    # filtered from a zero state for exactly `context` real frames.
    _, embedding_probe = fit_probes(
        model, train, backbone, device, context=context, horizon=horizon, seed=seed,
    )
    common = dict(
        context=context, horizon=horizon, seed=seed, device=device, feature_backbone=backbone,
    )
    reference = evaluate_rollout(model, val, embedding_probe, **common)
    reproduction, step = _max_delta(
        reference.rssm_position, cell.record["curves"]["rssm_position"]
    )
    if reproduction != 0.0:
        print(
            f"\nRECORD MISMATCH for {arm} seed {seed}: evaluate_rollout no longer "
            f"reproduces the study record's curves.rssm_position (max abs "
            f"{reproduction:.3e} at step {step}). Measured, the records reproduce "
            f"bitwise on mps and miss on cpu by an arm-dependent 6-12 map units. "
            f"This run used device={device} torch={torch.__version__}."
        )
        return EXIT_RECORD_MISMATCH, None

    traj = reference_trajectories(model, val, embedding_probe, **common)
    check = self_check(traj, cell.diagnostic)
    if not check.ok:
        print(
            f"\nSELF-CHECK FAILED for {arm} seed {seed}: " + "; ".join(check.failures())
            + ". Same windows, same rollout, same refit probe -- or this is not "
            "measuring what the ladder measured. No record written."
        )
        return EXIT_SELF_CHECK_FAILED, None

    record = trust_record(
        arm, seed, traj, cell.diagnostic, context=context, horizon=horizon, device=device,
    )
    path = write_trust_record(args.out, record)
    episodes = len(set(int(e) for e in traj.window_episode))
    folds = (
        "available" if record["scale"]["folds_available"] else
        "UNAVAILABLE (one episode cannot make two folds: alpha, the held-out "
        "error and every episode-clustered SE are NaN)"
    )
    print(
        f"{arm} seed {seed}: {traj.windows_total} windows from {episodes} validation "
        f"episode{'' if episodes == 1 else 's'}; folds {folds}; self-check max|delta| "
        f"reference {check.reference_position_max_delta:.1e} persistence "
        f"{check.persistence_position_max_delta:.1e}; wrote {path}"
    )
    return EXIT_OK, record


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = get_device(prefer=args.device)
    # EVERY requested cell, before any refit: the readings need all of them,
    # and a missing ninth cell found after eight refits is twenty minutes late.
    try:
        cells = [load_cell(args.out, arm, seed) for arm in args.arms for seed in args.seeds]
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS

    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    # The study's own split: the shared VAL_FRACTION, and SPLIT_SEED -- fixed
    # at 0 and deliberately NOT the cell's seed.
    train, val = episode_split(
        buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED
    )
    # The live records, keyed (arm, seed), and the run's horizon -- the same
    # on every cell, since the protocol check refuses a cell at another one.
    # The pooling after the loop reads both.
    records: dict[tuple[str, int], dict] = {}
    horizon = 0
    for cell in cells:
        status, record = _run_cell(args, cell, device, train, val)
        if status != EXIT_OK:
            return status
        records[(cell.arm, cell.seed)] = record
        horizon = int(record["horizon"])
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_trust_horizon_script.py -q -p no:cacheprovider 2>&1 | tail -3`

Expected: `21 passed` (19 test functions, the ordering test parametrised three ways), roughly 40 s -- eleven of them build the tiny cell, ~3 s each. Record the measured time beside the count.

If `test_the_self_check_reads_zero_...` fails on `reference_position_max_delta`, Task 3's `positions` / `true_positions` rows are not the rows `_diagnose` averages into `curves.reference_position` (the check recomputes the mean from the rows, spec 2.3) -- fix it THERE, not by loosening the `!= 0.0` here. If `test_trust_record_wires_every_channel_...` fails on `alpha_b` or `boundary`, Task 2's tie rule is not "the smallest alpha" -- same answer.

- [ ] **Step 5: Mutation-test**

The script is loaded by path from the working tree, so the harness mutates `scripts/trust_horizon.py` in place and restores it from a copy. The three self-checks are built in: a known-fatal mutation runs FIRST (a harness green on it is not running the mutated file); the mutated file is confirmed to be the file under test (`grep -cF` on the very path the test's `_load_script` loads -- `mbfps` itself is imported from the editable install at `src/`, which IS the code under test, and no export is involved); and `__pycache__` is cleared before every run under `PYTHONDONTWRITEBYTECODE=1` with `-p no:cacheprovider`, so a same-byte-length mutation cannot leave a stale `.pyc` behind. Each run is the whole file (~40 s); the table takes ~15 min.

```bash
cp scripts/trust_horizon.py /tmp/trust_horizon.py.orig
run() {
  find . -name __pycache__ -exec rm -rf {} + 2>/dev/null
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_trust_horizon_script.py \
    -q -p no:cacheprovider 2>&1 | grep -E "^(FAILED|ERROR|[0-9]+ (passed|failed))" | sed 's/ - .*//'
}
mut() {  # mut "<label>" "<exact text to replace>" "<replacement>"
  echo "=== $1"
  cp /tmp/trust_horizon.py.orig scripts/trust_horizon.py
  .venv/bin/python - "$2" "$3" <<'EOF'
import sys
from pathlib import Path
p = Path("scripts/trust_horizon.py"); s = p.read_text()
assert s.count(sys.argv[1]) == 1, f"mutation target not unique: {sys.argv[1]!r}"
p.write_text(s.replace(sys.argv[1], sys.argv[2]))
EOF
  # Fixed-string, never a regex: the replacements carry `[`, `(` and `.`.
  echo "mutated lines in the file under test: $(grep -cF -- "$3" scripts/trust_horizon.py)   # must be >= 1"
  run
}

mut "M0 known-fatal: no main"                 'def main(argv: list[str] | None = None) -> int:' 'def main_(argv: list[str] | None = None) -> int:'
mut "M1 reference delta judged at 1e-6"       'if self.reference_position_max_delta != 0.0:' 'if self.reference_position_max_delta > 1e-6:'
mut "M2 persistence curve never judged"       'if self.persistence_position_max_delta != 0.0:' 'if False and self.persistence_position_max_delta != 0.0:'
mut "M3 windows.total never judged"           'if not self.windows_total_match:' 'if False and not self.windows_total_match:'
mut "M4 windows.episode never judged"         'if not self.windows_episode_match:' 'if False and not self.windows_episode_match:'
mut "M5 shape guard removed"                  'if ours.shape != theirs.shape:' 'if False and ours.shape != theirs.shape:'
mut "M6 self-check never fails the cell"      '    if not check.ok:' '    if False and not check.ok:'
mut "M7 split mismatch not a refusal"         '        return EXIT_SPLIT_MISMATCH, None' '        return EXIT_OK, None'
mut "M8 protocol mismatch never refused"      '    if mismatch is not None:' '    if False and mismatch is not None:'
mut "M9 --horizon not checked"                'for flag in ("context", "horizon"):' 'for flag in ("context",):'
mut "M10 --arms ignored"                      'for arm in args.arms for seed in args.seeds' 'for arm in ARMS for seed in args.seeds'
mut "M11 --seeds ignored"                     'for arm in args.arms for seed in args.seeds' 'for arm in args.arms for seed in SEEDS'
mut "M12 probe crossing fed swapped errors"   '"probe": crossing_step(model_err, persist_err, moved),' '"probe": crossing_step(persist_err, model_err, moved),'
mut "M13 margin sign flipped"                 '"margin": persistence_margin(model_err, persist_err),' '"margin": persistence_margin(persist_err, model_err),'
mut "M14 free crossing fed the probe errors"  'traj.embedding_distance_to_truth, traj.embedding_persistence_distance, moved' 'model_err, persist_err, moved'
mut "M15 ratio_free is ratio_probe"           '"ratio_free": embedding_ratio(
            traj.embedding_displacement, traj.true_embedding_displacement, moved
        ),' '"ratio_free": decomposition.ratio_probe,'
mut "M16 decomposition given positions as real" 'traj.true_at_context, traj.positions_real, moved,' 'traj.true_at_context, traj.positions, moved,'
mut "M17 never_moved counts any unmoved step" '"never_moved": int((~moved.any(axis=1)).sum()),' '"never_moved": int((~moved).any(axis=1).sum()),'
mut "M18 band read off the reference curve"   '"persistence": curves["persistence_position"][-1],' '"persistence": curves["reference_position"][-1],'
mut "M19 git_sha not asked"                   '"git_sha": git_sha(),' '"git_sha": "unknown",'
mut "M20 record named by the arm alone"       'f"trust_{arm}_seed{seed}.json"' 'f"trust_{arm}.json"'
mut "M21 diagnostic existence not checked"    '        ("diagnostic", diagnostic_path),
' ''
mut "M22 30 collides with 14"                 'EXIT_SELF_CHECK_FAILED = 30' 'EXIT_SELF_CHECK_FAILED = 14'
mut "M23 split judged after the refit"        '    names = [p.name for p in val]
    if cell.record["episodes"]["val"] != names:' '    names = [p.name for p in val]
    if cell.record["episodes"]["val"] != names and fit_probes(None, None, None, None):'
mut "M24 self_check reads the pass's reference curve, not the rows" 'reference_rows.mean(axis=0), curves["reference_position"]' 'traj.reference_position, curves["reference_position"]'
mut "M25 self_check reads the pass's persistence curve, not the rows" 'persistence_rows.mean(axis=0), curves["persistence_position"]' 'traj.persistence_position, curves["persistence_position"]'
mut "M26 probe_real displacement taken off the imagination" '"probe_real": _distance(traj.positions_real, traj.positions_at_context[:, None, :]),' '"probe_real": _distance(traj.positions, traj.positions_at_context[:, None, :]),'

cp /tmp/trust_horizon.py.orig scripts/trust_horizon.py
cmp /tmp/trust_horizon.py.orig scripts/trust_horizon.py && echo "restored byte-for-byte"
rm /tmp/trust_horizon.py.orig
# NOT `git diff --stat ... must print nothing`: this task's file is untracked
# until Step 7. The `cmp` against the copy taken before the first mutation
# is the check that means something here.
echo "=== restored"; run                    # must print: 21 passed
```

(M21 deletes a line: its `grep -cF` prints the count of the EMPTY replacement, which is every line -- confirm that mutation by `grep -cF '("diagnostic", diagnostic_path)' scripts/trust_horizon.py` printing 0 instead.)

| mutation | must be caught by |
|---|---|
| M0 known-fatal: `main` renamed | all eleven `cell`-fixture items (the eight `main` tests, the ordering test three ways) fail on `AttributeError`. A green run here means the harness is not running the mutated file; stop and fix the harness. |
| M1 the reference curve is judged at `> 1e-6` instead of `!= 0.0` | `test_self_check_names_the_curve_and_the_step_of_the_largest_delta` (a delta of 2^-40 must fail the check) |
| M2 the persistence curve is never judged | `test_self_check_names_the_curve_...` (the persistence-only case: `ok` reads True) |
| M3 `windows.total` never judged | `test_self_check_refuses_a_window_count_or_episode_index_that_differs` |
| M4 `windows.episode` never judged | `test_self_check_refuses_a_window_count_...` (both the `[0, 0]` and the `None` case) |
| M5 shape guard removed from `_max_delta` | `test_self_check_refuses_a_curve_of_the_wrong_length_rather_than_raising` (`ValueError: operands could not be broadcast`) |
| M6 the self-check never fails the cell (the record is written regardless) | `test_a_doctored_reference_curve_is_exit_30_and_no_record_is_written` (reads 0 and finds a record). The ordering test is unaffected -- its earlier status still wins -- so this one test is the catch |
| M7 a split mismatch returns `EXIT_OK` | `test_a_split_that_is_not_the_records_is_exit_12_before_the_refit`, `test_the_checks_are_judged_in_the_order_11_12_14_30[split-self_check-EXIT_SPLIT_MISMATCH]` |
| M8 a `--context` / `--horizon` mismatch is never refused | `test_a_context_or_horizon_that_disagrees_...` (`_never_refit` raises `AssertionError` from inside `main`) |
| M9 only `--context` is checked | `test_a_context_or_horizon_that_disagrees_...` (the `--horizon 4` call reaches the refit) |
| M10 `--arms` ignored (every arm requested) | `test_arms_and_seeds_select_the_cells_and_an_absent_cell_is_exit_11` (`--arms random_vit --seeds 1` reads 11 naming `pixel_ae seed 1`), and every `cell` test (`_argv` passes `--arms`) |
| M11 `--seeds` ignored | `test_arms_and_seeds_select_...` (`--arms random_vit --seeds 1` reads 11 naming `random_vit seed 0`), and every `cell` test |
| M12 the probe crossing is fed persistence-for-model | `test_trust_record_wires_every_channel_...` (`crossing.probe` reads `[1, 3]`, not `[3, 3]`: the perfect predictor "crosses" at h=1) |
| M13 the margin's sign is flipped | `test_trust_record_wires_every_channel_...` (`margin` row 0 reads `[-6, -12]`) |
| M14 the free crossing is fed the probe's errors | `test_trust_record_wires_every_channel_...` (`crossing.free` reads `[3, 3]`, not `[3, 1]`) |
| M15 `ratio_free` is a copy of `ratio_probe` | `test_trust_record_wires_every_channel_...` (row 1 reads `[0, 0]`, not `[0.5, 0.5]`) |
| M16 the decomposition is given `positions` where `positions_real` belongs | `test_trust_record_wires_every_channel_...` (`ratio_probe` row 1 becomes NaN -- `|d_hat_real| = 0` -- instead of 0) |
| M17 `never_moved` counts a window with ANY unmoved step | `test_the_self_check_reads_zero_the_single_episode_is_disclosed_and_h1_never_moves` (reads 8, not 0: every window is unmoved at h=1) |
| M18 the measurability band is read off the reference curve | `test_measurability_is_the_persistence_to_floor_band_at_the_final_step` (floor `[1, 8]`: 12 - 8 > 0 is measurable, 6 - 8 is not) |
| M19 `git_sha` hardcoded | `test_trust_record_wires_every_channel_...`, `test_a_clean_cell_exits_ok_...` (`!= study.git_sha()`; this checkout has a HEAD) |
| M20 the record file is named by the arm alone | `test_write_trust_record_names_both_the_arm_and_the_seed_...`, and every `cell` test that opens `TRUST` |
| M21 `load_cell` does not check the diagnostic exists | `test_a_missing_diagnostic_is_exit_11_naming_the_file` and `test_the_checks_are_judged_in_the_order...[diagnostic-split-...]` (`load_record` raises the base `FileNotFoundError`, which `main` does not catch -- a traceback, not 11) |
| M22 `EXIT_SELF_CHECK_FAILED = 14` | `test_the_reused_statuses_are_diagnose_dynamics_own_and_thirty_is_new` (30 pinned; `test_a_doctored_reference_curve_...` would NOT catch it, since it compares against the constant -- which is why the constant is pinned by value) |
| M23 the split is judged only after a refit | `test_a_split_that_is_not_the_records_is_exit_12_before_the_refit` (`_never_refit` raises `AssertionError` out of `main`), `test_the_checks_are_judged_in_the_order...[split-self_check-...]` (the real `fit_probes(None, ...)` raises). The clean runs are NOT a catch: `and` short-circuits on a matching split |
| M24 `self_check` judges `traj.reference_position` instead of the rows | `test_self_check_judges_the_rows_the_record_is_built_from_not_the_passs_own_curve` (the moved row leaves the pass's curve at [3, 6]: reads 0.0, not 0.5). The fixture run is NOT a catch: there the two are bitwise equal |
| M25 `self_check` judges `traj.persistence_position` instead of the rows | `test_self_check_judges_the_rows_...` (the moved anchor: reads 0.0, not 1.0) |
| M26 `displacement.probe_real` is the imagined displacement | `test_trust_record_wires_every_channel_...` (`probe_real` row 1 reads [0, 0], not [6, 12]) |

Any mutation that survives on your run is a missing test. Add it before committing, and record the kill counts the harness printed beside the table.

- [ ] **Step 6: Run the full suite**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -3`

Expected: green, 0 warnings, 21 more than the previous task left (record the measured number). No other file moves: nothing here modifies `src/` or an existing test, and `test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own` does not yet load `trust_horizon` (Task 6 adds it).

- [ ] **Step 7: Commit**

```bash
git add scripts/trust_horizon.py tests/eval/test_trust_horizon_script.py
git commit -m "feat: trust_horizon.py part 1 -- load as diagnose_dynamics does, the four checks in order, one trust record per cell

The loading path is diagnose_dynamics.py's, imported by path (checkpoint_path,
load_checkpoint_model, diagnostic_record_path, probe_is_measurable) rather than
copied, so the two tools cannot read one cell through two loaders. Per cell:
11 (every requested cell present, before any refit), 12 (the split by name),
14 (evaluate_rollout reproduces curves.rssm_position; a --context/--horizon
that disagrees with the diagnostic is refused up front under the same status),
then one reference pass with the trajectories kept and 30 if its mean curves
or windows are not bitwise the diagnostic's -- with no record written. The
record holds the per-window crossing steps in both channels, the margin, the
three ratios and the four displacement norms they are built from, the cosine,
the held-out scale-corrected error with its fold alphas and boundary flags,
the excluded counts and the self-check deltas. The self-check recomputes the
two curves from the rows the record is built from, not from the pass's own
curve, so a regression in the rows alone cannot write a record.

The script tests train the tiny cell test_aggregate.py trains, have
diagnose_dynamics.main write its diagnostic, and drive main on it: the
self-check reads exactly 0.0 on both curves, the single validation episode
is disclosed with folds_available false, and each refusal is produced by
doctoring one file. The pure pieces are pinned by a fabricated two-window
trajectory whose answers -- a perfect predictor beside a persistence clone --
are worked by hand on every channel.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: trust_readings.py: the pre-registered rules of Reading 1 and Reading 2 over pooled inputs

Spec section 3 states both readings before the run, and section 5 says every rule of section 3 is tested "on fabricated pooled tables, each rule mutated one at a time". That is only possible if the rules live apart from the pooling: a module that takes the z of a clustered contrast, the estimate of a ratio of medians, a boundary flag and a survival curve -- numbers that are already pooled -- and returns the verdict with the rule that decided it. This task is that module and nothing else: no pooling, no records, no files, no torch. Task 6 computes the pooled numbers with `pooling.pool_arm` / `paired_contrast` / `pool_ratio`, reduces them to `Contrast` / `Ratio`, fills `ReadingOneInputs`, and prints what `format_reading_one` / `format_reading_two` return. Every sentence of section 3.2 / 3.3 that is a rule is quoted in the docstring of the function that implements it, and every comparison against the family bar is strict: a z equal to `z_fam` does not clear it, and a NaN or infinite z never does (the ladder's `_responds` policy in `scripts/pool_dynamics.py:146` -- an infinite z is a zero standard error, a degenerate ruler, not an infinitely precise one).

Two things the contract does not spell out and this task decides (both accepted at reconciliation): where "(i) unresolved through the probe" -- the least-moving arm differing between `R_probe` and `R_free` -- sits in the status precedence (after NOT_TESTABLE and before the three-condition verdict, as `UNRESOLVED_PROBE`; the contract's precedence line is to be read as `... condition_i undecidable -> NOT_TESTABLE; least_moving_arm None -> UNRESOLVED_PROBE; all three hold ...`, which spec 3.2's "or (i) is *unresolved through the probe*" supports), and that `reading_one` refuses pooled inputs with `per_seed=None` rather than reporting "0 of 0 seeds" (Task 6 always passes per-seed leaves). One more, so the printout is right on any horizon: `condition_i` / `condition_ii` / `condition_iii`, `reading_one` and `format_reading_one` take an optional `h: int = 45` that names the step in every detail string and in the Reading 1 header; the contract's signatures are unchanged for a caller that omits it, and Task 6 passes the run's horizon.

**Depends on Task 1** having landed `trust_horizon(surv, q) -> int` in `src/mbfps/eval/trust.py`: `reading_two` reads `H*_q` off each survival curve with it. Nothing else from Tasks 1-4 is imported.

**Files:**
- Create: `src/mbfps/eval/trust_readings.py` -- the constants, the six dataclasses and the `Status` enum, `best_delta_arm`, `least_moving_arm`, `condition_i` / `condition_ii` / `condition_iii`, `probe_control`, `reading_one`, `reading_two`, `format_reading_one`, `format_reading_two`, and five private helpers (`_fmt`, `_clears`, `_pair_z`, `_argmin_arm`, `_alpha_unreadable`)
- Test: `tests/eval/test_trust_readings.py` (new)

> `src/mbfps/eval/__init__.py` is empty apart from its docstring and nothing registers submodules; no edit there. `mbfps.utils.config.ARMS` (`src/mbfps/utils/config.py:16`) is `("pixel_ae", "frozen_ssl", "random_vit")`; the module restates it as `ARMS_ORDER` per the contract and the first test pins the two equal.

**Interfaces:**
- Consumes: `mbfps.eval.trust.trust_horizon(surv: np.ndarray, q: float) -> int` (Task 1) -- "largest h with surv[h] >= q; 0 if surv[0] < q; -1 if surv is all NaN". `mbfps.utils.config.ARMS` (existing, test only).
- Produces, verbatim from the contract, all in `mbfps.eval.trust_readings`:
  - `ARMS_ORDER = ("pixel_ae", "frozen_ssl", "random_vit")`; `TREATMENT = "frozen_ssl"`; `CONTROL = "random_vit"`; `FAMILY = 8`; `Q_PREREGISTERED = 0.75`; `Q_REPORTED = (0.5, 0.75, 0.9)`
  - `@dataclass(frozen=True) class Contrast: estimate: float; se: float; z: float; n_windows: int`
  - `@dataclass(frozen=True) class Ratio: estimate: float; low: float; high: float`
  - `@dataclass(frozen=True) class ReadingOneInputs: delta_contrast: dict[tuple[str, str], Contrast]; ratio_probe: dict[str, Ratio]; ratio_free: dict[str, Ratio]; cosine_contrast: Contrast; corrected_contrast_a: Contrast; corrected_contrast_b: Contrast; alpha_boundary: dict[str, bool]; crossing_contrast_probe: Contrast; crossing_contrast_free: Contrast; per_seed: dict[int, "ReadingOneInputs"] | None`
  - `class Status(str, Enum): SUPPORTED, NOT_SUPPORTED, NOT_TESTABLE, UNRESOLVED_PROBE, UNRESOLVED_ALPHA` (values `"supported"`, `"not supported"`, `"not testable"`, `"unresolved through the probe"`, `"unresolved (alpha on the grid boundary)"`)
  - `@dataclass(frozen=True) class ConditionResult: name: str; holds: bool | None; detail: str`
  - `@dataclass(frozen=True) class ReadingOne: status: Status; conditions: tuple[ConditionResult, ...]; best_delta_arm: str | None; least_moving_arm: str | None; per_seed_agreement: dict[str, int]; reason: str` -- `per_seed_agreement` is keyed `"(i)"`, `"(ii)"`, `"(iii)"`; `conditions` is those three in that order, always all three
  - `best_delta_arm(delta_contrast, z_fam) -> str | None`; `least_moving_arm(ratio_probe, ratio_free) -> str | None`; `condition_i(inputs, z_fam, h=45) -> ConditionResult`; `condition_ii(inputs, z_fam, h=45) -> ConditionResult`; `condition_iii(inputs, z_fam, h=45) -> ConditionResult`; `probe_control(inputs, z_fam) -> bool`; `reading_one(inputs, z_fam, h=45) -> ReadingOne` -- `h` only names the step in the detail strings (`Δ(h)`, `cos(h)`, `c(h)`, `R_probe(h)`) and decides nothing
  - `@dataclass(frozen=True) class ReadingTwo: survival: dict[tuple[str, str], np.ndarray]; horizons: dict[tuple[str, str, float], int]; h_min: int`; `reading_two(survival, qs=Q_REPORTED) -> ReadingTwo` -- `survival` keyed `(arm, channel)` with channel in `("probe", "free")`, one `(H + 1,)` curve each; `horizons` keyed `(arm, channel, q)` for every arm in `ARMS_ORDER`, both channels, every q in `qs`
  - `format_reading_one(r: ReadingOne, inputs: ReadingOneInputs, z_fam: float, h: int = 45) -> str` (first line `--- Reading 1: does the h={h} gate reward slow drift?  (z_fam ..., family 8, treatment frozen_ssl, control random_vit)`); `format_reading_two(r: ReadingTwo) -> str` (first line `--- Reading 2: the horizon M4 designs around  (S(h) = ...; no verdict)`)
  - Raised: `KeyError` from `best_delta_arm` / `condition_i` / `reading_one` when a pair is missing from `delta_contrast` (either orientation is accepted; the estimate of key `(a, b)` is `a - b`); `ValueError` from `best_delta_arm` when a table keyed in both orientations lets two arms win; `ValueError` from `reading_one` when `inputs.per_seed is None`; `KeyError` from `reading_two` naming the missing `(arm, channel)`; `ValueError` from `reading_two` when `Q_PREREGISTERED` is not in `qs`.
  - `z_fam` is `pooling.cluster_threshold(FAMILY, clusters)` computed by Task 6 from the actual cluster count; this module never computes it and reads whatever bar it is given.

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_trust_readings.py` in full:

```python
"""The pre-registered rules of Reading 1 and Reading 2, on fabricated pooled inputs.

Every test here builds its inputs by hand and states the answer by hand;
nothing reads an expectation from the function under test. The baseline
fixture is the slow-drift story told exactly as spec 3.2 would find it
supported -- `random_vit` best on Δ(45) against each other arm and least
moving under both ratio channels, the treatment's cosine contrast clearing
the bar, its held-out corrected error smaller on both folds, no α on a grid
boundary, the h× control silent -- and every test after it mutates ONE rule's
input and states what the rule must then say. The bar `z_fam` is a hand-chosen
3.0: the rules read whatever bar they are given (`cluster_threshold(8, 24)` is
3.01 on the shipped split, and that is Task 6's to compute).
"""

import math

import numpy as np
import pytest

from mbfps.eval.trust_readings import (
    ARMS_ORDER,
    CONTROL,
    FAMILY,
    Q_PREREGISTERED,
    Q_REPORTED,
    TREATMENT,
    ConditionResult,
    Contrast,
    Ratio,
    ReadingOne,
    ReadingOneInputs,
    ReadingTwo,
    Status,
    best_delta_arm,
    condition_i,
    condition_ii,
    condition_iii,
    format_reading_one,
    format_reading_two,
    least_moving_arm,
    probe_control,
    reading_one,
    reading_two,
)
from mbfps.utils.config import ARMS

Z_FAM = 3.0
NAN = float("nan")
HORIZON = 45


def _c(z: float, n: int = 229) -> Contrast:
    """A contrast whose z is `z`; se fixed at 0.1 so the estimate is 0.1 z."""
    return Contrast(estimate=0.1 * z, se=0.1, z=z, n_windows=n)


def _r(estimate: float) -> Ratio:
    return Ratio(estimate=estimate, low=estimate - 0.1, high=estimate + 0.1)


def _leaf(**overrides) -> ReadingOneInputs:
    """One seed's (or the pooled) inputs: the slow-drift story, supported.

    Keys are oriented `(a, b)` with `a` earlier in ARMS_ORDER, so `random_vit`
    -- the best-Δ arm -- is always the SUBTRACTED arm: its two contrasts read
    -4 and -5 here and must be flipped to +4 and +5 to be read as
    `random_vit - other`."""
    fields = dict(
        delta_contrast={
            ("pixel_ae", "frozen_ssl"): _c(-1.0),
            ("pixel_ae", "random_vit"): _c(-4.0),
            ("frozen_ssl", "random_vit"): _c(-5.0),
        },
        ratio_probe={"pixel_ae": _r(0.9), "frozen_ssl": _r(1.3), "random_vit": _r(0.4)},
        ratio_free={"pixel_ae": _r(0.8), "frozen_ssl": _r(1.2), "random_vit": _r(0.3)},
        cosine_contrast=_c(4.0),
        corrected_contrast_a=_c(-4.0),
        corrected_contrast_b=_c(-3.5),
        alpha_boundary={"pixel_ae": False, "frozen_ssl": False, "random_vit": False},
        crossing_contrast_probe=_c(1.0),
        crossing_contrast_free=_c(0.5),
        per_seed=None,
    )
    fields.update(overrides)
    return ReadingOneInputs(**fields)


def _inputs(pooled: dict | None = None, seeds: dict | None = None) -> ReadingOneInputs:
    """Pooled inputs from `pooled` overrides over three per-seed leaves;
    `seeds` maps a seed to the overrides for that seed's leaf alone."""
    seeds = seeds or {}
    return _leaf(**(pooled or {}), per_seed={s: _leaf(**seeds.get(s, {})) for s in (0, 1, 2)})


def _step(k: int) -> np.ndarray:
    """A survival curve that is 1 through h = k and 0 after: H*_q = k for every q <= 1."""
    return np.array([1.0] * (k + 1) + [0.0] * (HORIZON - k))


def _curves(**per_key: np.ndarray) -> dict:
    """All six (arm, channel) curves, defaulting to 'never crosses' (S = 1
    everywhere, H*_q = 45); `per_key` overrides by `arm__channel`."""
    curves = {(arm, channel): np.ones(HORIZON + 1) for arm in ARMS_ORDER for channel in ("probe", "free")}
    for name, curve in per_key.items():
        arm, channel = name.rsplit("__", 1)
        curves[(arm, channel)] = curve
    return curves


# --- Constants ---------------------------------------------------------------


def test_the_constants_are_the_specs():
    """ARMS_ORDER is the study's arm order restated (the module imports
    nothing but numpy and trust); the family is the eight contrasts of spec
    3.1; q = 0.75 is pre-registered and printed among 0.5 and 0.9."""
    assert ARMS_ORDER == ("pixel_ae", "frozen_ssl", "random_vit") == ARMS
    assert (TREATMENT, CONTROL) == ("frozen_ssl", "random_vit")
    assert FAMILY == 8
    assert Q_PREREGISTERED == 0.75
    assert Q_REPORTED == (0.5, 0.75, 0.9)
    assert [s.name for s in Status] == [
        "SUPPORTED", "NOT_SUPPORTED", "NOT_TESTABLE", "UNRESOLVED_PROBE", "UNRESOLVED_ALPHA"
    ]


# --- (i): the best-Δ arm ------------------------------------------------------


def test_best_delta_arm_is_the_arm_whose_delta_contrast_clears_z_fam_against_each_other_arm():
    """random_vit - pixel_ae is +4 and random_vit - frozen_ssl is +5, both
    above 3.0; no other arm clears even one. Both of random_vit's contrasts
    are keyed the other way round, so a lookup that forgets to flip the sign
    reads -4 and -5 and finds no arm."""
    assert best_delta_arm(_leaf().delta_contrast, Z_FAM) == "random_vit"


def test_best_delta_arm_clearing_one_contrast_but_not_the_other_is_none_and_reading_one_is_not_testable():
    """random_vit still clears pixel_ae (+4) but its contrast with frozen_ssl
    drops to +2: "if no arm clears both, (i) is undecidable at this precision
    and Reading 1 is not testable, never supported." Everything else in the
    fixture holds, so any status but NOT_TESTABLE means a rule was skipped."""
    table = dict(_leaf().delta_contrast)
    table[("frozen_ssl", "random_vit")] = _c(-2.0)
    assert best_delta_arm(table, Z_FAM) is None
    result = reading_one(_inputs(pooled={"delta_contrast": table}), Z_FAM)
    assert result.status is Status.NOT_TESTABLE
    assert result.best_delta_arm is None
    assert result.conditions[0].holds is None
    assert "undecidable at this precision" in result.conditions[0].detail
    assert result.reason.startswith("(i) undecidable at this precision")
    # (ii) and (iii) were still evaluated and reported.
    assert result.conditions[1].holds is True and result.conditions[2].holds is True


def test_best_delta_arm_reads_a_pair_keyed_the_other_way_round_with_its_sign_flipped():
    """The same table keyed `(random_vit, other)`: +4 and +5 are read
    directly and must NOT be flipped. Together with the test above this pins
    the flip to the reversed orientation only."""
    reversed_keys = {
        ("frozen_ssl", "pixel_ae"): _c(+1.0),
        ("random_vit", "pixel_ae"): _c(+4.0),
        ("random_vit", "frozen_ssl"): _c(+5.0),
    }
    assert best_delta_arm(reversed_keys, Z_FAM) == "random_vit"


def test_best_delta_arm_refuses_a_missing_pair_and_an_inconsistent_table():
    """A pair absent from the table is a KeyError naming it, not "does not
    clear"; a table keyed in both orientations with values that let two arms
    win is a ValueError, not the first arm in ARMS_ORDER."""
    table = dict(_leaf().delta_contrast)
    del table[("pixel_ae", "random_vit")]
    with pytest.raises(KeyError) as excinfo:
        best_delta_arm(table, Z_FAM)
    assert "no Δ(45) contrast for the pair" in str(excinfo.value)
    assert "'pixel_ae'" in str(excinfo.value) and "'random_vit'" in str(excinfo.value)
    inconsistent = {
        ("pixel_ae", "frozen_ssl"): _c(+4.0), ("frozen_ssl", "pixel_ae"): _c(+4.0),
        ("pixel_ae", "random_vit"): _c(+4.0), ("random_vit", "pixel_ae"): _c(+4.0),
        ("frozen_ssl", "random_vit"): _c(+4.0), ("random_vit", "frozen_ssl"): _c(+4.0),
    }
    with pytest.raises(ValueError, match="inconsistent"):
        best_delta_arm(inconsistent, Z_FAM)


def test_a_z_exactly_at_z_fam_does_not_clear_on_any_rule():
    """Every comparison is strict: z == z_fam is not `z > z_fam`. Pinned on
    (i), (ii) and (iii) separately, because each has its own comparison."""
    table = dict(_leaf().delta_contrast)
    table[("frozen_ssl", "random_vit")] = _c(-3.0)
    assert best_delta_arm(table, Z_FAM) is None
    assert condition_ii(_leaf(cosine_contrast=_c(3.0)), Z_FAM).holds is False
    assert condition_iii(_leaf(corrected_contrast_b=_c(-3.0)), Z_FAM).holds is None


# --- (i): the least-moving arm ------------------------------------------------


def test_least_moving_arm_is_the_smallest_probe_ratio_when_the_free_channel_agrees():
    """random_vit reads 0.4 under R_probe and 0.3 under R_free, the smallest
    in each; the answer is the arm, and condition (i) holds because it is
    also the best-Δ arm."""
    leaf = _leaf()
    assert least_moving_arm(leaf.ratio_probe, leaf.ratio_free) == "random_vit"
    result = condition_i(leaf, Z_FAM)
    assert (result.name, result.holds) == ("(i)", True)
    assert "best-Δ arm random_vit" in result.detail and "least-moving arm random_vit" in result.detail


def test_least_moving_arm_differing_between_channels_leaves_condition_i_unresolved_through_the_probe():
    """R_probe still says random_vit (0.4) but R_free now says pixel_ae
    (0.1): "it must be the same arm under R_free(45), or (i) is unresolved
    through the probe." The condition is None, not False, and Reading 1 is
    UNRESOLVED_PROBE even though (ii) and (iii) hold."""
    free = dict(_leaf().ratio_free)
    free["pixel_ae"] = _r(0.1)
    assert least_moving_arm(_leaf().ratio_probe, free) is None
    result = condition_i(_leaf(ratio_free=free), Z_FAM)
    assert result.holds is None
    assert "unresolved through the probe" in result.detail
    assert "random_vit under R_probe(45) but pixel_ae under R_free(45)" in result.detail
    verdict = reading_one(_inputs(pooled={"ratio_free": free}), Z_FAM)
    assert verdict.status is Status.UNRESOLVED_PROBE
    assert verdict.best_delta_arm == "random_vit" and verdict.least_moving_arm is None
    assert verdict.reason.startswith("(i) unresolved through the probe")


def test_least_moving_arm_is_undefined_on_a_nan_or_a_tied_ratio():
    """A NaN estimate cannot be compared, and an exact tie has no single
    smallest arm: both are None. The degenerate values are placed where a
    plain `min` would still name an arm that AGREES with the other channel
    -- frozen_ssl's probe ratio NaN leaves min at random_vit, which is the
    free channel's answer; a pixel_ae/random_vit tie in BOTH channels leaves
    min at pixel_ae in both -- so a missing guard returns an arm, not None."""
    leaf = _leaf()
    nan_probe = dict(leaf.ratio_probe)
    nan_probe["frozen_ssl"] = _r(NAN)
    assert least_moving_arm(nan_probe, leaf.ratio_free) is None
    tied_probe = dict(leaf.ratio_probe)
    tied_free = dict(leaf.ratio_free)
    tied_probe["pixel_ae"], tied_free["pixel_ae"] = _r(0.4), _r(0.3)
    assert least_moving_arm(tied_probe, tied_free) is None
    assert condition_i(_leaf(ratio_probe=tied_probe, ratio_free=tied_free), Z_FAM).holds is None


def test_condition_i_holds_iff_the_least_moving_arm_is_the_best_delta_arm():
    """frozen_ssl moves least under both channels (0.1, 0.1) while random_vit
    is still the best-Δ arm: (i) is decided -- False -- and the detail names
    both arms."""
    probe = dict(_leaf().ratio_probe)
    free = dict(_leaf().ratio_free)
    probe["frozen_ssl"], free["frozen_ssl"] = _r(0.1), _r(0.1)
    result = condition_i(_leaf(ratio_probe=probe, ratio_free=free), Z_FAM)
    assert result.holds is False
    assert "best-Δ arm random_vit" in result.detail
    assert "least-moving arm frozen_ssl" in result.detail
    assert result.detail.endswith("different arms")


# --- (ii) ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "z, holds",
    [(3.0001, True), (2.9999, False), (4.0, True), (-4.0, False), (NAN, False), (math.inf, False)],
)
def test_condition_ii_reads_the_cosine_z_strictly_above_z_fam(z, holds):
    """`z > z_fam`, signed: a large NEGATIVE cosine contrast (the treatment
    points the wrong way) does not hold, and neither does a NaN (fewer than
    two clusters) or an infinite z (a zero standard error)."""
    result = condition_ii(_leaf(cosine_contrast=_c(z)), Z_FAM)
    assert result.name == "(ii)" and result.holds is holds
    assert "cos(45) frozen_ssl-random_vit" in result.detail


# --- (iii) --------------------------------------------------------------------


@pytest.mark.parametrize(
    "z_a, z_b, holds",
    [
        (-4.0, -3.5, True),      # both folds below -z_fam
        (-4.0, -1.0, None),      # fold A only: not holds, not fails
        (-1.0, -4.0, None),      # fold B only
        (-1.0, -1.0, None),      # neither fold resolved
        (-4.0, +4.0, False),     # a fold above +z_fam fails whatever the other says
        (+4.0, -4.0, False),
        (+3.5, +1.0, False),
        (NAN, -4.0, None),       # a NaN fold is undecided, not a failure
    ],
)
def test_condition_iii_needs_both_folds_below_minus_z_fam(z_a, z_b, holds):
    """"holds iff z < -z_fam ...; fails iff z > z_fam; undecided otherwise.
    (iii) holds only if it holds on both folds." """
    result = condition_iii(
        _leaf(corrected_contrast_a=_c(z_a), corrected_contrast_b=_c(z_b)), Z_FAM
    )
    assert result.name == "(iii)" and result.holds is holds
    assert "fold A" in result.detail and "fold B" in result.detail
    if holds is None:
        assert result.detail.startswith("undecided")


def test_condition_iii_is_unreadable_on_a_boundary_alpha_of_the_treatment_or_the_control_but_not_of_pixel_ae():
    """"unreadable -- Reading 1 unresolved -- if either arm's α is on the
    grid boundary on either fold": either arm of the CONTRAST. The control's
    boundary and the treatment's each make (iii) None with "unreadable" and
    the verdict UNRESOLVED_ALPHA, even with both folds far below -z_fam;
    pixel_ae's boundary decides nothing and the reading stays SUPPORTED."""
    for arm in (CONTROL, TREATMENT):
        flags = {"pixel_ae": False, "frozen_ssl": False, "random_vit": False, arm: True}
        result = condition_iii(_leaf(alpha_boundary=flags), Z_FAM)
        assert result.holds is None and result.detail.startswith("unreadable"), arm
        assert arm in result.detail
        verdict = reading_one(_inputs(pooled={"alpha_boundary": flags}), Z_FAM)
        assert verdict.status is Status.UNRESOLVED_ALPHA, arm
        assert verdict.reason.startswith("(iii) unreadable")
    pixel_only = {"pixel_ae": True, "frozen_ssl": False, "random_vit": False}
    assert condition_iii(_leaf(alpha_boundary=pixel_only), Z_FAM).holds is True
    assert reading_one(_inputs(pooled={"alpha_boundary": pixel_only}), Z_FAM).status is Status.SUPPORTED


# --- The probe control --------------------------------------------------------


@pytest.mark.parametrize(
    "z_probe, z_free, fires",
    [
        (+4.0, -4.0, True),
        (-4.0, +4.0, True),
        (+4.0, +4.0, False),    # same sign: the channels agree
        (-4.0, -4.0, False),
        (+4.0, -1.0, False),    # the free channel is inside the bar
        (+1.0, -4.0, False),    # the probe channel is inside the bar
        (0.0, -4.0, False),     # a tie contradicts nothing
        (NAN, -4.0, False),
        (+3.0, -4.0, False),    # exactly at the bar does not clear
    ],
)
def test_probe_control_fires_only_when_both_channels_clear_with_opposite_signs(z_probe, z_free, fires):
    """"if both channels clear z_fam with opposite signs, Reading 1 is
    unresolved through the probe ... A tie, or an interval covering 0 in
    either channel, contradicts nothing." """
    leaf = _leaf(crossing_contrast_probe=_c(z_probe), crossing_contrast_free=_c(z_free))
    assert probe_control(leaf, Z_FAM) is fires


def test_probe_control_firing_is_unresolved_through_the_probe_whatever_the_conditions_said():
    """Every condition holds pooled and in all three seeds; the control alone
    fires, and the verdict is UNRESOLVED_PROBE with the two z's in the
    reason -- and the conditions are still reported as holding."""
    fire = {"crossing_contrast_probe": _c(+4.0), "crossing_contrast_free": _c(-4.0)}
    result = reading_one(_inputs(pooled=fire), Z_FAM)
    assert result.status is Status.UNRESOLVED_PROBE
    assert result.reason.startswith("probe control")
    assert "probe z +4.00" in result.reason and "free z -4.00" in result.reason
    assert [c.holds for c in result.conditions] == [True, True, True]
    assert result.per_seed_agreement == {"(i)": 3, "(ii)": 3, "(iii)": 3}


# --- Per-seed agreement and the verdict ----------------------------------------


def test_per_seed_agreement_three_two_and_one_of_three():
    """Pooled everything holds throughout. 3/3 seeds: SUPPORTED. Seed 2's
    cosine contrast drops to +1: (ii) holds in 2 of 3 -- still SUPPORTED,
    "at least two of the three seeds". Seeds 1 and 2 both drop: 1 of 3,
    NOT_SUPPORTED, and the reason says which condition and the count."""
    three = reading_one(_inputs(), Z_FAM)
    assert three.status is Status.SUPPORTED
    assert three.per_seed_agreement == {"(i)": 3, "(ii)": 3, "(iii)": 3}
    assert three.best_delta_arm == "random_vit" and three.least_moving_arm == "random_vit"
    assert "(ii) 3/3" in three.reason

    two = reading_one(_inputs(seeds={2: {"cosine_contrast": _c(1.0)}}), Z_FAM)
    assert two.status is Status.SUPPORTED
    assert two.per_seed_agreement == {"(i)": 3, "(ii)": 2, "(iii)": 3}

    one = reading_one(
        _inputs(seeds={1: {"cosine_contrast": _c(1.0)}, 2: {"cosine_contrast": _c(1.0)}}), Z_FAM
    )
    assert one.status is Status.NOT_SUPPORTED
    assert one.per_seed_agreement == {"(i)": 3, "(ii)": 1, "(iii)": 3}
    assert [c.holds for c in one.conditions] == [True, True, True]
    assert "(ii) holds in 1 of 3 seeds" in one.reason and "2 of 3 required" in one.reason


def test_an_undecided_seed_does_not_count_as_agreeing():
    """In seeds 1 and 2 the best-Δ arm is undecidable (random_vit - frozen_ssl
    reads +2): (i) is None there, which is not "holds". Agreement on (i) is
    1 of 3 and the reading is NOT_SUPPORTED; counting None as not-False
    would call it 3 of 3."""
    table = dict(_leaf().delta_contrast)
    table[("frozen_ssl", "random_vit")] = _c(-2.0)
    result = reading_one(
        _inputs(seeds={1: {"delta_contrast": table}, 2: {"delta_contrast": table}}), Z_FAM
    )
    assert result.status is Status.NOT_SUPPORTED
    assert result.per_seed_agreement == {"(i)": 1, "(ii)": 3, "(iii)": 3}
    assert "(i) holds in 1 of 3 seeds" in result.reason


def test_a_pooled_condition_failing_is_not_supported_even_when_every_seed_agrees_and_ii_alone_is_named_informative():
    """The pooled cosine contrast reads +1 while all three seeds hold: the
    rule is "all hold pooled AND ...", so NOT_SUPPORTED. And "(ii) failing
    alone is the informative failure: the prior does not drift slowly, it
    moves wrong" -- the reason says so when (ii) is the only failure, and
    does not when (i) fails beside it."""
    only_ii = reading_one(_inputs(pooled={"cosine_contrast": _c(1.0)}), Z_FAM)
    assert only_ii.status is Status.NOT_SUPPORTED
    assert only_ii.per_seed_agreement == {"(i)": 3, "(ii)": 3, "(iii)": 3}
    assert only_ii.reason.startswith("pooled: (ii) not holding")
    assert "it moves wrong" in only_ii.reason

    probe = dict(_leaf().ratio_probe)
    free = dict(_leaf().ratio_free)
    probe["frozen_ssl"], free["frozen_ssl"] = _r(0.1), _r(0.1)
    i_and_ii = reading_one(
        _inputs(pooled={"cosine_contrast": _c(1.0), "ratio_probe": probe, "ratio_free": free}), Z_FAM
    )
    assert i_and_ii.status is Status.NOT_SUPPORTED
    assert i_and_ii.reason.startswith("pooled: (i), (ii) not holding")
    assert "it moves wrong" not in i_and_ii.reason


def test_the_status_precedence_is_probe_then_alpha_then_testable_then_least_moving_then_the_conditions():
    """Four inputs, each carrying every failure of the rungs below it."""
    undecidable = dict(_leaf().delta_contrast)
    undecidable[("frozen_ssl", "random_vit")] = _c(-2.0)
    disagreeing_free = dict(_leaf().ratio_free)
    disagreeing_free["pixel_ae"] = _r(0.1)
    control_boundary = {"pixel_ae": False, "frozen_ssl": False, "random_vit": True}
    everything = dict(
        crossing_contrast_probe=_c(+4.0), crossing_contrast_free=_c(-4.0),
        alpha_boundary=control_boundary, delta_contrast=undecidable,
        ratio_free=disagreeing_free, cosine_contrast=_c(1.0),
    )
    assert reading_one(_inputs(pooled=everything), Z_FAM).status is Status.UNRESOLVED_PROBE

    del everything["crossing_contrast_probe"], everything["crossing_contrast_free"]
    assert reading_one(_inputs(pooled=everything), Z_FAM).status is Status.UNRESOLVED_ALPHA

    del everything["alpha_boundary"]
    assert reading_one(_inputs(pooled=everything), Z_FAM).status is Status.NOT_TESTABLE

    del everything["delta_contrast"]
    assert reading_one(_inputs(pooled=everything), Z_FAM).status is Status.UNRESOLVED_PROBE

    del everything["ratio_free"]
    assert reading_one(_inputs(pooled=everything), Z_FAM).status is Status.NOT_SUPPORTED


def test_reading_one_refuses_inputs_without_per_seed_entries():
    """The two-of-three rule cannot be evaluated on a leaf; a leaf passed as
    the pooled inputs is a ValueError, not a SUPPORTED with 0 of 0 seeds."""
    with pytest.raises(ValueError, match="per_seed"):
        reading_one(_leaf(), Z_FAM)


# --- Reading 2 -----------------------------------------------------------------


def test_reading_two_on_a_bimodal_survival_curve():
    """Half the draws cross at h× = 2, half never (h× = 46): S(0) = S(1) = 1,
    S(2..45) = 0.5. H*_0.5 is 45 (S(45) = 0.5 >= 0.5), H*_0.75 is 1 and
    H*_0.9 is 1 -- the median reading says "trust it to the end", the
    pre-registered quantile says "one step". The other five curves never
    cross, so H*_min is this arm's 1."""
    bimodal = np.array([1.0, 1.0] + [0.5] * 44)
    assert bimodal.shape == (HORIZON + 1,)
    result = reading_two(_curves(frozen_ssl__free=bimodal))
    assert isinstance(result, ReadingTwo)
    assert result.horizons[("frozen_ssl", "free", 0.5)] == 45
    assert result.horizons[("frozen_ssl", "free", 0.75)] == 1
    assert result.horizons[("frozen_ssl", "free", 0.9)] == 1
    assert result.horizons[("frozen_ssl", "probe", 0.75)] == 45
    assert result.h_min == 1
    assert set(result.horizons) == {(a, ch, q) for a in ARMS_ORDER for ch in ("probe", "free") for q in Q_REPORTED}
    np.testing.assert_array_equal(result.survival[("frozen_ssl", "free")], bimodal)


def test_reading_two_on_an_even_count_curve_is_an_integer_with_no_rounding():
    """Four draws, h× = {3, 3, 7, 7}: S = 1 for h < 3, 0.5 for 3 <= h < 7,
    0 from h = 7. H*_0.5 is 6 -- the largest h with S >= 0.5 -- an int, not
    np.median's interpolated 5.0; H*_0.75 and H*_0.9 are 2."""
    even = np.array([1.0] * 3 + [0.5] * 4 + [0.0] * 39)
    assert even.shape == (HORIZON + 1,)
    result = reading_two(_curves(pixel_ae__probe=even))
    assert result.horizons[("pixel_ae", "probe", 0.5)] == 6
    assert type(result.horizons[("pixel_ae", "probe", 0.5)]) is int
    assert result.horizons[("pixel_ae", "probe", 0.5)] != np.median([3, 3, 7, 7])
    assert result.horizons[("pixel_ae", "probe", 0.75)] == 2
    assert result.horizons[("pixel_ae", "probe", 0.9)] == 2
    assert result.h_min == 45


def test_h_min_is_the_minimum_of_the_probe_free_h_star_at_the_preregistered_q():
    """Probe-free H*_0.75: pixel_ae 10, frozen_ssl 1 (the bimodal curve,
    whose H*_0.5 is 45), random_vit 12 -> H*_min = 1. The probe channel
    reads 0, 20, 2 and must not enter: over both channels the answer would
    be 0, at q = 0.5 it would be 10, with max in place of min 12."""
    bimodal = np.array([1.0, 1.0] + [0.5] * 44)
    result = reading_two(_curves(
        pixel_ae__probe=_step(0), frozen_ssl__probe=_step(20), random_vit__probe=_step(2),
        pixel_ae__free=_step(10), frozen_ssl__free=bimodal, random_vit__free=_step(12),
    ))
    assert [result.horizons[(a, "free", 0.75)] for a in ARMS_ORDER] == [10, 1, 12]
    assert [result.horizons[(a, "probe", 0.75)] for a in ARMS_ORDER] == [0, 20, 2]
    assert result.horizons[("frozen_ssl", "free", 0.5)] == 45
    assert result.h_min == 1
    assert type(result.h_min) is int


def test_reading_two_refuses_a_missing_arm_or_channel_and_a_q_list_without_the_preregistered_q():
    """A curve missing for random_vit/free is a KeyError naming it -- a
    minimum over two arms would print as one over three -- and a `qs`
    without 0.75 is a ValueError: H*_min is defined at the pre-registered q."""
    curves = _curves()
    del curves[("random_vit", "free")]
    with pytest.raises(KeyError, match=r"survival curve.*\('random_vit', 'free'\)"):
        reading_two(curves)
    with pytest.raises(ValueError, match="pre-registered q 0.75"):
        reading_two(_curves(), qs=(0.5, 0.9))


# --- Formatting ------------------------------------------------------------------


def test_format_reading_one_prints_every_condition_detail_every_seed_and_the_deciding_rule():
    """Seed 2 fails (ii) with z = +1 and the pooled control fires, so the
    printout must carry: the pooled details of all three conditions, the
    probe-control line with both z's, a block per seed holding that seed's
    own details (seed 2's (ii) reads z +1.00, the others +4.00), the
    agreement counts, and the verdict line naming the status and the rule
    that decided it."""
    inputs = _inputs(
        pooled={"crossing_contrast_probe": _c(+4.0), "crossing_contrast_free": _c(-4.0)},
        seeds={2: {"cosine_contrast": _c(1.0)}},
    )
    result = reading_one(inputs, Z_FAM, h=45)
    assert result.status is Status.UNRESOLVED_PROBE
    text = format_reading_one(result, inputs, Z_FAM, h=45)
    lines = text.splitlines()
    assert lines[0].startswith("--- Reading 1: does the h=45 gate reward slow drift?")
    assert "z_fam 3.00" in lines[0] and "family 8" in lines[0]
    for condition in result.conditions:
        assert condition.detail in text, condition.name
        assert f"{condition.name:<6}holds" in text, condition.name
    assert "probe control: fires" in text and "probe z +4.00" in text and "free z -4.00" in text
    for seed in (0, 1, 2):
        assert f"seed {seed}:" in text
    seed_2 = text[text.index("seed 2:"):]
    assert f"{'(ii)':<6}fails" in seed_2 and "z +1.00 not > z_fam 3.00" in seed_2
    assert text.count("z +4.00 > z_fam 3.00") == 3   # pooled, seed 0, seed 1
    assert "per-seed agreement: (i) 3/3, (ii) 2/3, (iii) 3/3  (2 of 3 required)" in text
    assert "best-Δ arm: random_vit; least-moving arm: random_vit" in text
    assert lines[-1] == f"verdict: Reading 1 {Status.UNRESOLVED_PROBE.value} -- decided by: {result.reason}"
    # `h` names the step everywhere it is printed and decides nothing: the
    # same inputs read at h = 3 give the same status with every "45" now "3".
    at_three = reading_one(inputs, Z_FAM, h=3)
    assert at_three.status is result.status
    text_three = format_reading_one(at_three, inputs, Z_FAM, h=3)
    assert text_three.splitlines()[0].startswith("--- Reading 1: does the h=3 gate reward slow drift?")
    assert "cos(3) frozen_ssl-random_vit" in text_three and "held-out c(3)" in text_three
    assert "(45)" not in text_three and "h=45" not in text_three


def test_format_reading_two_prints_every_survival_value_every_horizon_and_h_min():
    """One line per arm and channel carrying H*_0.5, H*_0.75, H*_0.9 and all
    46 values of S(h); then H*_min with the probe-based H*_0.75 beside it."""
    bimodal = np.array([1.0, 1.0] + [0.5] * 44)
    result = reading_two(_curves(frozen_ssl__free=bimodal, random_vit__probe=_step(7)))
    text = format_reading_two(result)
    lines = text.splitlines()
    assert lines[0].startswith("--- Reading 2: the horizon M4 designs around")
    assert "no verdict" in lines[0]
    assert lines[1].split()[:5] == ["arm", "channel", "H*_0.5", "H*_0.75", "H*_0.9"]
    body = lines[2:8]
    assert [line.split()[:2] for line in body] == [
        [arm, channel] for arm in ARMS_ORDER for channel in ("probe", "free")
    ]
    frozen_free = next(line for line in body if line.startswith(f"{'frozen_ssl':<12}{'free':<9}"))
    tokens = frozen_free.split()
    assert tokens[2:5] == ["45", "1", "1"]
    assert tokens[5:] == ["1.00", "1.00"] + ["0.50"] * 44
    random_probe = next(line for line in body if line.startswith(f"{'random_vit':<12}{'probe':<9}"))
    assert random_probe.split()[2:5] == ["7", "7", "7"]
    assert lines[-1].startswith("H*_min = 1  (min over arms of the probe-free H*_0.75")
    assert "pixel_ae 45, frozen_ssl 45, random_vit 7" in lines[-1]
```

Why these and not fewer. The baseline `_leaf()` keys `random_vit`'s two Δ contrasts the other way round (`(pixel_ae, random_vit)` = -4, `(frozen_ssl, random_vit)` = -5), so the very first `best_delta_arm` test catches a lookup that forgets to flip the sign, and the reversed-key test catches one that flips both orientations; only the two together pin `_pair_z`. The NaN/tie test puts its degenerate value where a plain `min` would still name an arm that agrees with the other channel -- otherwise `least_moving_arm` returns None for the wrong reason and a missing guard survives (it did, on the first draft of that test). The precedence test carries every failure at once and peels them off top-down, so a swapped pair of rules is the only thing it can fail on. `condition_ii` has a `-4.0` row because `|z| > z_fam` passes every positive case. The pixel_ae-boundary case in the (iii) test is the only one that distinguishes "either arm of the contrast" from "any arm". Both Reading 2 curves are built as literal arrays and their `H*_q` stated by hand -- 45 / 1 / 1 for the bimodal curve, 6 / 2 / 2 for the even-count one, against `np.median`'s 5.0 -- so `trust_horizon` is checked through the composition, not trusted.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_trust_readings.py -q -p no:cacheprovider`
Expected: **1 error** at collection --

```
tests/eval/test_trust_readings.py:20: in <module>
    from mbfps.eval.trust_readings import (
E   ModuleNotFoundError: No module named 'mbfps.eval.trust_readings'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.10s
```

- [ ] **Step 3: Implement**

Create `src/mbfps/eval/trust_readings.py` in full:

```python
"""Reading 1 and Reading 2 of the trust-horizon diagnostic, over pooled inputs.

The M3d spec (section 3) states both readings BEFORE the run. This module is
those rules and nothing else: pure functions over numbers that are already
pooled -- the z of a clustered contrast, the estimate of a ratio of medians,
a boundary flag, a survival curve -- so every rule can be tested on
fabricated inputs with each branch mutated one at a time, and the driver
(`scripts/trust_horizon.py`) only has to put the pooled numbers into
`ReadingOneInputs` and print what comes back.

Reading 1 asks whether the h=45 gate rewards slow drift. It is *supported*
only if three conditions hold pooled AND each holds within at least two of
the three seeds; any other outcome is *not supported*, *not testable* or
*unresolved*, and the verdict line names the rule that decided it. Nothing
here decides on a magnitude the caller did not pass: the bar `z_fam` is
`pooling.cluster_threshold(FAMILY, clusters)` computed by the caller from the
actual cluster count, and every comparison against it is STRICT -- a z equal
to the bar does not clear it, and a NaN or infinite z never clears it (the
ladder's `_responds` policy: an infinite z is a zero standard error, which is
a degenerate ruler, not an infinitely precise one).

Reading 2 attaches no verdict. It turns each arm's survival curve into
`H*_q` -- the largest h with `S(h) >= q`, an integer by construction because
it is read off the curve and never interpolated -- and takes `H*_min` over
the PROBE-FREE channel only, so a dead probe cannot set the horizon M4
designs around.

Every sentence of section 3.2 / 3.3 that is a rule is quoted above the
function that implements it.
"""

from dataclasses import dataclass
from enum import Enum
from itertools import combinations

import numpy as np

from mbfps.eval.trust import trust_horizon

ARMS_ORDER: tuple[str, ...] = ("pixel_ae", "frozen_ssl", "random_vit")
"""`mbfps.utils.config.ARMS`, restated so this module imports nothing but
numpy and `trust`; the test pins the two equal."""
TREATMENT = "frozen_ssl"
CONTROL = "random_vit"
FAMILY = 8
"""Spec 3.1: "Reading 1 makes eight clustered contrasts (three pairwise
Δ(45), one cos(45), one held-out c(45) per fold, and the two-channel h×
control); `cluster_threshold(family=8, clusters=24)` is the bar `z_fam`
every one of them is read against." """
Q_PREREGISTERED = 0.75
Q_REPORTED: tuple[float, ...] = (0.5, 0.75, 0.9)
"""Spec 3.3: "q = 0.75 pre-registered, with q = 0.5 and 0.9 printed beside
it (`H*_0.5` is the old median reading)." """

_CHANNELS: tuple[str, ...] = ("probe", "free")
_CONDITION_NAMES: tuple[str, ...] = ("(i)", "(ii)", "(iii)")
_SEEDS_REQUIRED = 2
"""Spec 3.2: "each holds within at least two of the three seeds." """


# ---------------------------------------------------------------------------
# The inputs, reduced to what the rules read.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Contrast:
    """What `pooling.paired_contrast` yields, reduced to what the rules read."""

    estimate: float
    se: float
    z: float
    n_windows: int


@dataclass(frozen=True)
class Ratio:
    """What `pooling.pool_ratio` yields, reduced."""

    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class ReadingOneInputs:
    """Every pooled number Reading 1 reads, at h = 45.

    `delta_contrast` is keyed by the three unordered pairs, in whichever
    orientation the caller built them; the estimate of key `(a, b)` is
    `a - b`, and `_pair_z` flips the sign when a pair is looked up the other
    way round. `per_seed` holds the same inputs computed within each seed
    alone (the leaf entries carry `per_seed=None`).
    """

    delta_contrast: dict[tuple[str, str], Contrast]
    ratio_probe: dict[str, Ratio]
    ratio_free: dict[str, Ratio]
    cosine_contrast: Contrast
    corrected_contrast_a: Contrast
    corrected_contrast_b: Contrast
    alpha_boundary: dict[str, bool]
    crossing_contrast_probe: Contrast
    crossing_contrast_free: Contrast
    per_seed: "dict[int, ReadingOneInputs] | None"


class Status(str, Enum):
    """The outcomes spec 3.2 names: "*Supported* only if (i), (ii) and (iii)
    all hold pooled and each holds within at least two of the three seeds;
    any other outcome is *not supported*, *not testable* or *unresolved*,
    written as such." The two unresolved outcomes are kept apart because
    they name different failures: one of the probe, one of the scale fit."""

    SUPPORTED = "supported"
    NOT_SUPPORTED = "not supported"
    NOT_TESTABLE = "not testable"
    UNRESOLVED_PROBE = "unresolved through the probe"
    UNRESOLVED_ALPHA = "unresolved (alpha on the grid boundary)"


@dataclass(frozen=True)
class ConditionResult:
    """`holds` is True / False / None, where None is undecided, undecidable
    or unreadable -- never folded into False, because the spec's statuses
    tell them apart."""

    name: str
    holds: bool | None
    detail: str


@dataclass(frozen=True)
class ReadingOne:
    status: Status
    conditions: tuple[ConditionResult, ...]
    best_delta_arm: str | None
    least_moving_arm: str | None
    per_seed_agreement: dict[str, int]
    reason: str


# ---------------------------------------------------------------------------
# Small shared pieces.
# ---------------------------------------------------------------------------


def _fmt(value: float, spec: str) -> str:
    """NaN and inf print as themselves, never as a number that looks measured."""
    return format(value, spec) if np.isfinite(value) else str(value)


def _clears(z: float, z_fam: float) -> bool:
    """STRICTLY above the bar. NaN never clears (nothing to read); an
    infinite z never clears either -- `pooling._z` returns ±inf for a zero
    standard error, which is a degenerate ruler, and the ladder's
    `_responds` refuses it the same way."""
    return bool(np.isfinite(z) and z > z_fam)


def _pair_z(delta_contrast: dict[tuple[str, str], Contrast], a: str, b: str) -> float:
    """The z of the Δ(45) contrast oriented as `a - b`, whichever way the
    caller keyed the pair. A missing pair is refused by name rather than
    read as "does not clear"."""
    if (a, b) in delta_contrast:
        return delta_contrast[(a, b)].z
    if (b, a) in delta_contrast:
        return -delta_contrast[(b, a)].z
    raise KeyError(
        f"no Δ(45) contrast for the pair {(a, b)!r}; the table has {sorted(delta_contrast)}"
    )


def _argmin_arm(ratios: dict[str, Ratio]) -> str | None:
    """The arm with the smallest ratio estimate. None when the smallest is
    not defined: a NaN estimate cannot be compared, and an exact tie has no
    single least-moving arm."""
    estimates = {arm: float(ratios[arm].estimate) for arm in ARMS_ORDER}
    if any(not np.isfinite(v) for v in estimates.values()):
        return None
    smallest = min(estimates.values())
    arms = [arm for arm, v in estimates.items() if v == smallest]
    return arms[0] if len(arms) == 1 else None


def _alpha_unreadable(inputs: ReadingOneInputs) -> list[str]:
    """Spec 3.2 (iii): "unreadable ... if either arm's α is on the grid
    boundary on either fold" -- EITHER ARM means the two arms in the
    contrast, the treatment and the control; `pixel_ae`'s boundary flag is
    carried for the record and decides nothing here."""
    return [arm for arm in (TREATMENT, CONTROL) if inputs.alpha_boundary[arm]]


# ---------------------------------------------------------------------------
# Reading 1 -- does the h=45 gate reward slow drift?
# ---------------------------------------------------------------------------


def best_delta_arm(delta_contrast: dict[tuple[str, str], Contrast], z_fam: float) -> str | None:
    """Spec 3.2 (i): "The *best-Δ arm* is the arm whose pooled `Δ(45)`
    contrast against *each* of the other two has `z > z_fam`; if no arm
    clears both, (i) is *undecidable at this precision*."

    Two arms cannot both clear both: `a - b` and `b - a` have opposite
    signs. Two winners can only come from a table keyed in BOTH
    orientations with inconsistent values, and that is refused rather than
    resolved by ARMS_ORDER.
    """
    winners = [
        arm
        for arm in ARMS_ORDER
        if all(_clears(_pair_z(delta_contrast, arm, other), z_fam) for other in ARMS_ORDER if other != arm)
    ]
    if len(winners) > 1:
        raise ValueError(
            f"{winners} each clear the Δ(45) contrast against every other arm; "
            "the contrast table carries both orientations of a pair with inconsistent values"
        )
    return winners[0] if winners else None


def least_moving_arm(ratio_probe: dict[str, Ratio], ratio_free: dict[str, Ratio]) -> str | None:
    """Spec 3.2 (i): "The *least-moving arm* is the arm with the smallest
    pooled `R_probe(45)` -- and it must be the same arm under `R_free(45)`,
    or (i) is *unresolved through the probe*." """
    probe_arm = _argmin_arm(ratio_probe)
    free_arm = _argmin_arm(ratio_free)
    if probe_arm is None or probe_arm != free_arm:
        return None
    return probe_arm


def condition_i(inputs: ReadingOneInputs, z_fam: float, h: int = 45) -> ConditionResult:
    """Spec 3.2 (i): "The arm that moves least is the arm the gate likes
    best. ... (i) holds iff the least-moving arm is the best-Δ arm." `h`
    only names the step in the detail."""
    pairs = ", ".join(
        f"Δ {a}-{b} z {_fmt(_pair_z(inputs.delta_contrast, a, b), '+.2f')}"
        for a, b in combinations(ARMS_ORDER, 2)
    )
    best = best_delta_arm(inputs.delta_contrast, z_fam)
    if best is None:
        return ConditionResult(
            "(i)",
            None,
            f"undecidable at this precision: no arm's Δ({h}) contrast clears "
            f"z_fam {_fmt(z_fam, '.2f')} against each of the other two ({pairs})",
        )
    ratios = ", ".join(
        f"{arm} R_probe {_fmt(inputs.ratio_probe[arm].estimate, '.3f')} "
        f"R_free {_fmt(inputs.ratio_free[arm].estimate, '.3f')}"
        for arm in ARMS_ORDER
    )
    least = least_moving_arm(inputs.ratio_probe, inputs.ratio_free)
    if least is None:
        probe_arm = _argmin_arm(inputs.ratio_probe) or "undefined"
        free_arm = _argmin_arm(inputs.ratio_free) or "undefined"
        return ConditionResult(
            "(i)",
            None,
            f"unresolved through the probe: the least-moving arm is {probe_arm} under "
            f"R_probe({h}) but {free_arm} under R_free({h}) ({ratios})",
        )
    holds = least == best
    return ConditionResult(
        "(i)",
        holds,
        f"best-Δ arm {best} ({pairs}); least-moving arm {least} under both channels "
        f"({ratios}): {'the same arm' if holds else 'different arms'}",
    )


def condition_ii(inputs: ReadingOneInputs, z_fam: float, h: int = 45) -> ConditionResult:
    """Spec 3.2 (ii): "The treatment moves the right way.
    `paired_contrast(frozen_ssl − random_vit)` on per-window seed-mean
    `cos(45)` has `z > z_fam`." """
    c = inputs.cosine_contrast
    holds = _clears(c.z, z_fam)
    return ConditionResult(
        "(ii)",
        holds,
        f"cos({h}) {TREATMENT}-{CONTROL}: estimate {_fmt(c.estimate, '+.3f')}, "
        f"se {_fmt(c.se, '.3f')}, z {_fmt(c.z, '+.2f')} {'>' if holds else 'not >'} "
        f"z_fam {_fmt(z_fam, '.2f')} (windows {c.n_windows})",
    )


def condition_iii(inputs: ReadingOneInputs, z_fam: float, h: int = 45) -> ConditionResult:
    """Spec 3.2 (iii): "Take the magnitude out and the gate ranking goes
    away. On each fold, `paired_contrast(frozen_ssl − random_vit)` on the
    held-out `c_w(45)`: holds iff `z < −z_fam` (the treatment's corrected
    error is smaller); fails iff `z > z_fam`; *undecided* otherwise. (iii)
    holds only if it holds on both folds, and is *unreadable* -- Reading 1
    *unresolved* -- if either arm's α is on the grid boundary on either
    fold." """
    a, b = inputs.corrected_contrast_a, inputs.corrected_contrast_b
    folds = (
        f"held-out c({h}) {TREATMENT}-{CONTROL}: fold A z {_fmt(a.z, '+.2f')}, "
        f"fold B z {_fmt(b.z, '+.2f')}, bar ±{_fmt(z_fam, '.2f')}"
    )
    on_boundary = _alpha_unreadable(inputs)
    if on_boundary:
        return ConditionResult(
            "(iii)",
            None,
            f"unreadable: alpha on the grid boundary at h={h} for {', '.join(on_boundary)} ({folds})",
        )
    holds_a, holds_b = _clears(-a.z, z_fam), _clears(-b.z, z_fam)
    if holds_a and holds_b:
        return ConditionResult("(iii)", True, f"holds on both folds ({folds})")
    if _clears(a.z, z_fam) or _clears(b.z, z_fam):
        return ConditionResult(
            "(iii)", False, f"fails: the treatment's corrected error is larger on a fold ({folds})"
        )
    which = "fold A" if holds_a else "fold B" if holds_b else "neither fold"
    return ConditionResult(
        "(iii)", None, f"undecided: {which} clears -z_fam, both are required ({folds})"
    )


def probe_control(inputs: ReadingOneInputs, z_fam: float) -> bool:
    """Spec 3.2, probe control: "The paired per-draw difference of `h×`
    (frozen_ssl − random_vit) in each channel: if both channels clear
    `z_fam` with *opposite* signs, Reading 1 is *unresolved through the
    probe* whatever (i)–(iii) said. A tie, or an interval covering 0 in
    either channel, contradicts nothing." True means it fires."""
    zp, zf = inputs.crossing_contrast_probe.z, inputs.crossing_contrast_free.z
    both_clear = _clears(abs(zp), z_fam) and _clears(abs(zf), z_fam)
    return bool(both_clear and (zp > 0) != (zf > 0))


def _conditions(inputs: ReadingOneInputs, z_fam: float, h: int = 45) -> tuple[ConditionResult, ...]:
    return (condition_i(inputs, z_fam, h), condition_ii(inputs, z_fam, h), condition_iii(inputs, z_fam, h))


def reading_one(inputs: ReadingOneInputs, z_fam: float, h: int = 45) -> ReadingOne:
    """Spec 3.2: "*Supported* only if (i), (ii) and (iii) all hold pooled
    **and** each holds within at least two of the three seeds; any other
    outcome is *not supported*, *not testable* or *unresolved*, written as
    such." And: "(ii) failing alone is the informative failure: the prior
    does not drift slowly, it moves wrong."

    The order of precedence: the probe control fires -> UNRESOLVED_PROBE;
    (iii) unreadable -> UNRESOLVED_ALPHA; (i) undecidable (no best-Δ arm)
    -> NOT_TESTABLE; (i) unresolved through the probe (the least-moving arm
    differs between channels) -> UNRESOLVED_PROBE; then the three conditions
    pooled and per seed -> SUPPORTED or NOT_SUPPORTED. Every condition is
    computed and returned whatever decided the status, so the printout can
    show what (i)-(iii) said. `h` names the step in the details and decides
    nothing; the caller (Task 6) passes the run's horizon.
    """
    if inputs.per_seed is None:
        raise ValueError(
            "reading_one needs the per-seed inputs (ReadingOneInputs.per_seed) to test the "
            "two-of-three rule; per_seed=None is only for a per-seed leaf"
        )
    conditions = _conditions(inputs, z_fam, h)
    best = best_delta_arm(inputs.delta_contrast, z_fam)
    least = least_moving_arm(inputs.ratio_probe, inputs.ratio_free)
    per_seed = {seed: _conditions(sub, z_fam, h) for seed, sub in sorted(inputs.per_seed.items())}
    n_seeds = len(per_seed)
    agreement = {
        name: sum(1 for conds in per_seed.values() if conds[k].holds is True)
        for k, name in enumerate(_CONDITION_NAMES)
    }
    i, ii, iii = conditions
    zp, zf = inputs.crossing_contrast_probe.z, inputs.crossing_contrast_free.z

    if probe_control(inputs, z_fam):
        status = Status.UNRESOLVED_PROBE
        reason = (
            f"probe control: the h× contrasts clear z_fam {_fmt(z_fam, '.2f')} with opposite "
            f"signs (probe z {_fmt(zp, '+.2f')}, free z {_fmt(zf, '+.2f')}), whatever (i)-(iii) said"
        )
    elif _alpha_unreadable(inputs):
        status = Status.UNRESOLVED_ALPHA
        reason = f"(iii) unreadable: {iii.detail}"
    elif best is None:
        status = Status.NOT_TESTABLE
        reason = f"(i) undecidable at this precision: {i.detail}"
    elif least is None:
        status = Status.UNRESOLVED_PROBE
        reason = f"(i) unresolved through the probe: {i.detail}"
    else:
        pooled_failed = [c.name for c in conditions if c.holds is not True]
        seeds_failed = [name for name, count in agreement.items() if count < _SEEDS_REQUIRED]
        if not pooled_failed and not seeds_failed:
            status = Status.SUPPORTED
            counts = ", ".join(f"{name} {agreement[name]}/{n_seeds}" for name in _CONDITION_NAMES)
            reason = (
                f"(i), (ii) and (iii) hold pooled and each in at least {_SEEDS_REQUIRED} of "
                f"{n_seeds} seeds ({counts})"
            )
        else:
            status = Status.NOT_SUPPORTED
            parts = []
            if pooled_failed:
                parts.append(f"pooled: {', '.join(pooled_failed)} not holding")
            if seeds_failed:
                parts.append(
                    "per seed: "
                    + ", ".join(
                        f"{name} holds in {agreement[name]} of {n_seeds} seeds" for name in seeds_failed
                    )
                    + f" ({_SEEDS_REQUIRED} of {n_seeds} required)"
                )
            reason = "; ".join(parts)
            if set(pooled_failed) | set(seeds_failed) == {"(ii)"}:
                reason += (
                    " -- (ii) failing alone is the informative failure: the prior does not "
                    "drift slowly, it moves wrong"
                )
    return ReadingOne(
        status=status,
        conditions=conditions,
        best_delta_arm=best,
        least_moving_arm=least,
        per_seed_agreement=agreement,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Reading 2 -- the horizon M4 designs around.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadingTwo:
    survival: dict[tuple[str, str], np.ndarray]
    horizons: dict[tuple[str, str, float], int]
    h_min: int


def reading_two(
    survival: dict[tuple[str, str], np.ndarray], qs: tuple[float, ...] = Q_REPORTED
) -> ReadingTwo:
    """Spec 3.3: "Per arm and per channel, the **survival curve** `S(h)` =
    the fraction of (window, seed) draws with `h× > h`, h = 0..45, over
    draws that moved. `H*_q` = the largest h with `S(h) ≥ q`; **q = 0.75
    pre-registered**, with q = 0.5 and 0.9 printed beside it. ... One
    derived number: **`H*_min` = min over the three arms of the probe-free
    `H*_0.75`** -- probe-independent, so a dead probe cannot set it. ... No
    verdict is attached."

    `H*_q` is `trust.trust_horizon`, read off the curve: an integer with no
    interpolation, so an even-count multiset of crossings never yields a
    half-step. A curve missing for an arm or channel is refused by name: a
    minimum over two arms would print as the minimum over three.
    """
    keys = [(arm, channel) for arm in ARMS_ORDER for channel in _CHANNELS]
    missing = [key for key in keys if key not in survival]
    if missing:
        raise KeyError(f"reading_two needs a survival curve per arm and channel; missing {missing}")
    if Q_PREREGISTERED not in qs:
        raise ValueError(
            f"the pre-registered q {Q_PREREGISTERED} must be among the reported qs, got {tuple(qs)}"
        )
    curves = {key: np.asarray(survival[key], dtype=float) for key in keys}
    horizons = {
        (arm, channel, float(q)): trust_horizon(curves[(arm, channel)], q)
        for arm, channel in keys
        for q in qs
    }
    h_min = min(horizons[(arm, "free", Q_PREREGISTERED)] for arm in ARMS_ORDER)
    return ReadingTwo(survival=curves, horizons=horizons, h_min=int(h_min))


# ---------------------------------------------------------------------------
# Printing, in the ladder's style: the numbers, then the verdict with the
# rule that decided it.
# ---------------------------------------------------------------------------


def _word(holds: bool | None) -> str:
    return "holds" if holds is True else "fails" if holds is False else "undecided"


def _condition_lines(conditions: tuple[ConditionResult, ...], indent: str) -> list[str]:
    return [f"{indent}{c.name:<6}{_word(c.holds):<10} {c.detail}" for c in conditions]


def format_reading_one(r: ReadingOne, inputs: ReadingOneInputs, z_fam: float, h: int = 45) -> str:
    """The pooled conditions with their details, the probe control, the
    same three conditions within each seed, the agreement counts, and the
    verdict with the rule it was decided by. Every detail is printed even
    when an earlier rule decided the status: the reader must be able to see
    what (i)-(iii) said under an unresolved verdict."""
    zp, zf = inputs.crossing_contrast_probe.z, inputs.crossing_contrast_free.z
    fires = probe_control(inputs, z_fam)
    lines = [
        f"--- Reading 1: does the h={h} gate reward slow drift?  "
        f"(z_fam {_fmt(z_fam, '.2f')}, family {FAMILY}, treatment {TREATMENT}, control {CONTROL})",
        "pooled:",
        *_condition_lines(r.conditions, "  "),
        f"  probe control: {'fires' if fires else 'does not fire'} "
        f"(h× {TREATMENT}-{CONTROL} per draw: probe z {_fmt(zp, '+.2f')}, free z {_fmt(zf, '+.2f')}; "
        f"fires iff both |z| > z_fam with opposite signs)",
    ]
    for seed, sub in sorted((inputs.per_seed or {}).items()):
        lines.append(f"seed {seed}:")
        lines.extend(_condition_lines(_conditions(sub, z_fam, h), "  "))
    n_seeds = len(inputs.per_seed or {})
    counts = ", ".join(f"{name} {r.per_seed_agreement[name]}/{n_seeds}" for name in _CONDITION_NAMES)
    lines.append(f"per-seed agreement: {counts}  ({_SEEDS_REQUIRED} of {n_seeds} required)")
    lines.append(
        f"best-Δ arm: {r.best_delta_arm or 'none'}; least-moving arm: {r.least_moving_arm or 'none'}"
    )
    lines.append(f"verdict: Reading 1 {r.status.value} -- decided by: {r.reason}")
    return "\n".join(lines)


def format_reading_two(r: ReadingTwo) -> str:
    """Per arm and channel: `H*_q` for every reported q, then `S(h)` at every
    h -- the table M4 cites is the curve itself. Then `H*_min` with the
    probe-based `H*_0.75` beside it so M4 can see whether the channels
    agree. No verdict."""
    qs = sorted({q for (_, _, q) in r.horizons})
    header = f"{'arm':<12}{'channel':<9}" + "".join(f"{f'H*_{q:g}':>9}" for q in qs) + "   S(h), h = 0..H"
    lines = [
        "--- Reading 2: the horizon M4 designs around  "
        f"(S(h) = fraction of moved draws with h× > h; H*_q = largest h with S(h) >= q; "
        f"q = {Q_PREREGISTERED:g} pre-registered; no verdict)",
        header,
    ]
    for arm in ARMS_ORDER:
        for channel in _CHANNELS:
            curve = " ".join(_fmt(s, ".2f") for s in r.survival[(arm, channel)])
            horizons = "".join(f"{r.horizons[(arm, channel, q)]:>9d}" for q in qs)
            lines.append(f"{arm:<12}{channel:<9}{horizons}   {curve}")
    probe_side = ", ".join(
        f"{arm} {r.horizons[(arm, 'probe', Q_PREREGISTERED)]}" for arm in ARMS_ORDER
    )
    lines.append(
        f"H*_min = {r.h_min}  (min over arms of the probe-free H*_{Q_PREREGISTERED:g}; "
        f"probe-based H*_{Q_PREREGISTERED:g} beside it: {probe_side})"
    )
    return "\n".join(lines)
```

Three choices worth a sentence each. `_pair_z` accepts either key orientation because Task 6 builds the three contrasts with `paired_contrast(treatment_cells, control_cells)` in whatever order it iterates the pairs, and a rule that silently read a missing orientation as "does not clear" would turn a keying mistake into NOT_TESTABLE. `condition_i` returns `holds=None` for both "undecidable" and "unresolved through the probe", so `reading_one` does not parse the detail string to tell them apart: it calls `best_delta_arm` and `least_moving_arm` itself, exactly as the condition does, and the two `None`s reach different statuses through those calls. `reading_two` restricts the stored curves to the six `(arm, channel)` keys and casts them to float so the printed table is the table the horizons were read from.

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_trust_readings.py -q -p no:cacheprovider`
Expected: **46 passed** -- 23 test functions plus the three parametrised ones (6 + 8 + 9 cases), 0 warnings, well under a second (no torch is imported).

Then, because `condition_i`'s detail string is the only place the Δ pairs are formatted and a KeyError there would surface in every reading:

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c "import mbfps.eval.trust_readings as m; print(sorted(n for n in dir(m) if not n.startswith('_')))"`
Expected: the list contains `ARMS_ORDER, CONTROL, ConditionResult, Contrast, FAMILY, Q_PREREGISTERED, Q_REPORTED, Ratio, ReadingOne, ReadingOneInputs, ReadingTwo, Status, TREATMENT, best_delta_arm, condition_i, condition_ii, condition_iii, format_reading_one, format_reading_two, least_moving_arm, probe_control, reading_one, reading_two` (plus `Enum`, `combinations`, `dataclass`, `np`, `trust_horizon` from the imports) and nothing named `pooling`, `torch` or `Path`: the module is pure over its inputs.

- [ ] **Step 5: Mutation-test**

Set up a self-checked harness. All three checks below have silently lied in this project (M3b plan, Global Constraints). The work is not yet committed, so export by copy rather than `git archive`:

```bash
# 1. The package is installed EDITABLE, so a naive export is shadowed by
#    src/. Run with PYTHONPATH at the mutated copy and confirm it wins.
S=/tmp/m3d-task5 && rm -rf $S && mkdir -p $S && cp -R src tests pyproject.toml $S/
cd $S && PYTHONPATH=$S/src PYTHONDONTWRITEBYTECODE=1 \
  /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -c "import mbfps; print(mbfps.__file__)"
# must print /tmp/m3d-task5/src/mbfps/__init__.py, NOT .../csgo-bot/src/...

# 2. An unproven harness proves nothing: run the known-fatal mutation FIRST
#    (M0 below: best_delta_arm returns None unconditionally) and confirm
#    TWELVE tests fail before trusting any other row.

# 3. A same-byte-length mutation can leave a stale .pyc that survives a
#    correct restore: before EVERY run,
find $S/src -name __pycache__ -exec rm -rf {} + ; export PYTHONDONTWRITEBYTECODE=1
```

Apply each mutation to `$S/src/mbfps/eval/trust_readings.py`, run `PYTHONPATH=$S/src PYTHONDONTWRITEBYTECODE=1 /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest tests/eval/test_trust_readings.py -q -p no:cacheprovider` from `$S`, restore from the repo copy (`cp /Users/raphaelchen/Desktop/csgo-bot/src/mbfps/eval/trust_readings.py $S/src/mbfps/eval/`), and confirm the restored file passes 46 again at the end. Each row names the exact source line to change.

| mutation | must be caught by |
|---|---|
| M0 (known-fatal, run first): `best_delta_arm` ends `return None` instead of `return winners[0] if winners else None` | 12 tests: every test that reads a SUPPORTED or an arm-named verdict -- `test_best_delta_arm_is_the_arm_whose_delta_contrast_clears_z_fam_against_each_other_arm`, `..._reads_a_pair_keyed_the_other_way_round...`, `test_least_moving_arm_is_the_smallest_probe_ratio...`, `..._differing_between_channels...`, `test_condition_i_holds_iff...`, `test_condition_iii_is_unreadable_on_a_boundary_alpha...`, `test_probe_control_firing_is_unresolved...`, `test_per_seed_agreement_three_two_and_one_of_three`, `test_an_undecided_seed_does_not_count_as_agreeing`, `test_a_pooled_condition_failing_is_not_supported...`, `test_the_status_precedence_is_...`, `test_format_reading_one_prints_...` |
| M1: `_clears` uses `z >= z_fam` | `test_a_z_exactly_at_z_fam_does_not_clear_on_any_rule`, `test_probe_control_fires_only_when_both_channels_clear_with_opposite_signs[3.0--4.0-False]` |
| M2: `_pair_z` returns `delta_contrast[(b, a)].z` without the minus | 11 tests, first among them `test_best_delta_arm_is_the_arm_whose_delta_contrast_clears_z_fam_against_each_other_arm` (the baseline finds no arm) |
| M3: `_pair_z` negates the direct `(a, b)` lookup too | `test_best_delta_arm_reads_a_pair_keyed_the_other_way_round_with_its_sign_flipped`, `test_best_delta_arm_refuses_a_missing_pair_and_an_inconsistent_table` |
| M4: `best_delta_arm` uses `any(...)` over the other arms instead of `all(...)` | `test_best_delta_arm_clearing_one_contrast_but_not_the_other_is_none_and_reading_one_is_not_testable`, `test_a_z_exactly_at_z_fam_does_not_clear_on_any_rule`, `test_an_undecided_seed_does_not_count_as_agreeing`, `test_the_status_precedence_is_...` |
| M5: `if len(winners) > 1:` becomes `> 99` (two winners resolved by ARMS_ORDER) | `test_best_delta_arm_refuses_a_missing_pair_and_an_inconsistent_table` (the ONLY test that catches it) |
| M6: `least_moving_arm` ignores `ratio_free` -- `if probe_arm is None:` alone | `test_least_moving_arm_differing_between_channels_leaves_condition_i_unresolved_through_the_probe`, `test_the_status_precedence_is_...` |
| M7: `_argmin_arm` becomes `return min(estimates, key=estimates.get)` (no NaN / tie guard) | `test_least_moving_arm_is_undefined_on_a_nan_or_a_tied_ratio` (the ONLY test that catches it; see its docstring for why the degenerate values sit where they do) |
| M8: `condition_ii` reads `_clears(abs(c.z), z_fam)` | `test_condition_ii_reads_the_cosine_z_strictly_above_z_fam[-4.0-False]` (the ONLY case that catches it) |
| M9: `condition_iii`: `if holds_a or holds_b:` | `test_a_z_exactly_at_z_fam_does_not_clear_on_any_rule` and the `[-4.0--1.0-None]`, `[-1.0--4.0-None]`, `[-4.0-4.0-False]`, `[4.0--4.0-False]`, `[nan--4.0-None]` cases of `test_condition_iii_needs_both_folds_below_minus_z_fam` |
| M10: `condition_iii`: fails only when BOTH folds clear `+z_fam` (`and` for `or`) | the `[-4.0-4.0-False]`, `[4.0--4.0-False]`, `[3.5-1.0-False]` cases of `test_condition_iii_needs_both_folds_below_minus_z_fam` |
| M11: `_alpha_unreadable` iterates `ARMS_ORDER` (pixel_ae's boundary fires) | `test_condition_iii_is_unreadable_on_a_boundary_alpha_of_the_treatment_or_the_control_but_not_of_pixel_ae` (the ONLY test that catches it) |
| M12: `_alpha_unreadable` iterates `(TREATMENT,)` only | `test_condition_iii_is_unreadable_on_a_boundary_alpha_...`, `test_the_status_precedence_is_...` |
| M13: `probe_control` returns `bool(both_clear)` (fires on same signs) | `test_probe_control_fires_only_when_both_channels_clear_with_opposite_signs[4.0-4.0-False]` and `[-4.0--4.0-False]` |
| M14: `probe_control`: `both_clear = ... or ...` | the `[4.0--1.0-False]`, `[1.0--4.0-False]`, `[3.0--4.0-False]` cases of `test_probe_control_fires_only_when_both_channels_clear_with_opposite_signs` |
| M15: `reading_one` tests `_alpha_unreadable` before `probe_control` | `test_the_status_precedence_is_probe_then_alpha_then_testable_then_least_moving_then_the_conditions` (the ONLY test that catches it) |
| M16: `reading_one` tests `best is None` before `_alpha_unreadable` | `test_the_status_precedence_is_...` (the ONLY test that catches it) |
| M17: the `elif least is None:` branch removed (channel disagreement falls through to NOT_SUPPORTED) | `test_least_moving_arm_differing_between_channels_...`, `test_the_status_precedence_is_...` |
| M18: `_SEEDS_REQUIRED = 3` | `test_per_seed_agreement_three_two_and_one_of_three`, `test_format_reading_one_prints_...` |
| M19: `_SEEDS_REQUIRED = 1` | `test_per_seed_agreement_three_two_and_one_of_three`, `test_an_undecided_seed_does_not_count_as_agreeing`, `test_format_reading_one_prints_...` |
| M20: agreement counts `conds[k].holds is not False` (None counted as holding) | `test_an_undecided_seed_does_not_count_as_agreeing` (the ONLY test that catches it) |
| M21: SUPPORTED reads `if not seeds_failed:` alone (pooled ignored) | `test_a_pooled_condition_failing_is_not_supported_even_when_every_seed_agrees_and_ii_alone_is_named_informative`, `test_the_status_precedence_is_...` |
| M22: the "(ii) failing alone" sentence never appended (`if False:`) | `test_a_pooled_condition_failing_is_not_supported_...` (the ONLY test that catches it) |
| M23: the `per_seed is None` guard replaced by treating it as `{}` | `test_reading_one_refuses_inputs_without_per_seed_entries` (the ONLY test that catches it) |
| M24: `h_min` minimised over both channels | `test_h_min_is_the_minimum_of_the_probe_free_h_star_at_the_preregistered_q`, `test_reading_two_on_an_even_count_curve_is_an_integer_with_no_rounding` |
| M25: `h_min` read at q = 0.5 instead of `Q_PREREGISTERED` | `test_reading_two_on_a_bimodal_survival_curve`, `test_h_min_is_the_minimum_...`, `test_format_reading_two_prints_...` |
| M26: `trust_horizon(...)` replaced by `int(round(float(np.interp(q, curve[::-1], np.arange(len(curve))[::-1]))))` (an interpolated horizon) | `test_reading_two_on_a_bimodal_survival_curve`, `test_reading_two_on_an_even_count_curve_is_an_integer_with_no_rounding`, `test_h_min_is_the_minimum_...`, `test_format_reading_two_prints_...` |
| M27: the missing-curve guard removed (`if False:`) | `test_reading_two_refuses_a_missing_arm_or_channel_and_a_q_list_without_the_preregistered_q` (the raw dict KeyError does not say "survival curve") |
| M28: the `Q_PREREGISTERED not in qs` guard removed | `test_reading_two_refuses_a_missing_arm_or_channel_...` (a KeyError from the horizons lookup is not the ValueError asserted) |
| M29: `format_reading_one` drops the per-seed blocks | `test_format_reading_one_prints_every_condition_detail_every_seed_and_the_deciding_rule` (the ONLY test that catches it) |
| M30: `format_reading_one`'s verdict line drops ` -- decided by: {r.reason}` | `test_format_reading_one_prints_...` (the ONLY test that catches it) |
| M31: `format_reading_two` drops `   {curve}` from the per-row line | `test_format_reading_two_prints_every_survival_value_every_horizon_and_h_min` (the ONLY test that catches it) |
| M32: the `H*_min` line's "beside it" list reads `'free'` instead of `'probe'` | `test_format_reading_two_prints_...` (the ONLY test that catches it) |
| M33: `h` ignored -- the header hard-codes `h=45` (or `_conditions` drops the `h` it is passed) | `test_format_reading_one_prints_...` (the h = 3 block: the header reads `h=45`, or `cos(45)` survives in the details) |

Rows M0-M32 were run against the pre-reconciliation test file and implementation with the harness above (M33 was added at reconciliation with the `h` parameter and has not been run: run it, and record what fired); each row's catcher is what actually fired, and the rows marked ONLY are caught by exactly one test. M7 survived the first draft of its test (a NaN in one channel only makes the two channels disagree, so `least_moving_arm` returned None for the wrong reason); the test now places the NaN and the tie where a plain `min` agrees across channels. Any mutation that survives is a missing test. Add it before committing.

- [ ] **Step 6: Run the full suite**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: **46 more passed than the previous task left** (record the measured number; the contract's 1278 is the count at b382194, before Tasks 1-4 added theirs), 0 failed, 0 warnings, the same skips as before. Nothing outside `tests/eval/test_trust_readings.py` imports the new module yet -- Task 6 is its first caller -- so no other test can move.

- [ ] **Step 7: Commit**

```bash
git add src/mbfps/eval/trust_readings.py tests/eval/test_trust_readings.py
git commit -m "feat: the pre-registered rules of Reading 1 and Reading 2, pure over pooled inputs

Spec section 3 states the readings before the run and section 5 asks for
each rule to be tested on fabricated pooled tables, mutated one at a
time. trust_readings.py is those rules apart from the pooling: it reads
the z of a clustered contrast, the estimate of a ratio of medians, a
boundary flag and a survival curve, and returns the status with the
rule that decided it. Every comparison against z_fam is strict and a
NaN or infinite z never clears; (iii) is unreadable on a boundary alpha
of the treatment or the control only; the probe control fires only on
both h-cross channels clearing with opposite signs; SUPPORTED needs all
three conditions pooled and each in at least two of three seeds; H*_q
is read off the survival curve with trust_horizon, never interpolated,
and H*_min is over the probe-free channel alone.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

---

### Task 6: scripts/trust_horizon.py, part 2: pooling glue, the two readings, trust.txt, exit-status distinctness

Task 4 left `scripts/trust_horizon.py` writing one `trust_<arm>_seed<n>.json` per cell and returning `EXIT_OK`; Task 5 left `src/mbfps/eval/trust_readings.py` deciding both readings over *already-pooled* inputs. This task is the glue between them: it turns the per-cell records into `pooling.CellSeries`, pools them exactly as spec 3.1 says (`pool_arm` / `paired_contrast` on per-window seed-mean series; the `h×` contrasts on per-draw series with the seeds stacked; `pool_ratio` on the moved windows' numerator/denominator series; `z_fam = cluster_threshold(FAMILY, clusters)`; the same inputs again within each seed), hands them to `reading_one` / `reading_two`, and prints and writes `trust.txt`. Nothing statistical is defined here -- every estimator is `pooling.py`'s and every rule is Task 5's -- so the tests pin the *wiring*: the right series under the right mask into the right estimator, with answers computed by hand on six fabricated windows. On the shared fixture (one validation episode) every clustered SE is NaN; `main` still exits 0, prints both readings with every contrast `n/a` and Reading 1 `NOT_TESTABLE`, and says why. Finally `trust_horizon` joins the exit-status distinctness test, which has to learn that this script deliberately *reuses* three of `diagnose_dynamics.py`'s codes.

**Files:**
- Modify: `scripts/trust_horizon.py` -- append the constants `BOOTSTRAP`, `BOOTSTRAP_SEED`, `R2_SENSITIVITY`, `STACKED_SEED`, the four contract functions `cell_series`, `pooled_inputs`, `survival_by_arm`, `write_readings`, the per-arm block `ArmSummary` / `arm_summaries`, the private helpers `_stacked`, `_stacked_crossings`, `_contrast`, `_ratio`, `_column`, `_moved`, `_moved_series`, `_pooled_mean`, `_measurable`, `_inputs`, `_informational_crossing`, `_r2_filtered`, `_num`, `_self_check_table`, `_pooling_notes`, `_arm_table`, `_sensitivity_table`, `_readings_text`, and extend `main` (replace its final `return EXIT_OK`) so it pools, prints and writes `trust.txt`. The imports at the top of the file gain `dataclasses`, `itertools`, and the Task 5 names listed under Interfaces.
- Modify: `tests/eval/test_trust_horizon_script.py` -- APPEND a "Pooling glue, readings, trust.txt" section after Task 4's last test (do not touch Task 4's header, loader or tests).
- Modify: `tests/eval/test_diagnose_dynamics_script.py` -- `test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own` (line 463 as of b382194) gains the `trust_horizon` block.

**Interfaces:**
- Consumes (Task 4, `scripts/trust_horizon.py`): `EXIT_OK = 0`, `main(argv: list[str] | None = None) -> int`, and the per-cell record dicts `main` builds -- top-level keys exactly `arm, seed, context, horizon, split_seed, device, torch_version, git_sha, episodes {val: [...]}, windows {total, episode}, probe {selection_r2, measurable}, self_check {...}, crossing {probe: [n], free: [n]}, margin [n][H], ratio_probe [n][H], ratio_raw [n][H], ratio_free [n][H], cosine [n][H], displacement {probe_hat [n][H], probe_real [n][H], free_hat [n][H], free_true [n][H]}, scale {alpha_a [H], alpha_b [H], score_a [H], score_b [H], held_out [n][H], boundary [H], folds_available}, counts {not_moved [H], zero_displacement [H], never_moved}, nonfinite {}`. The moved mask at a step is read off `ratio_raw`, which the contract's `Decomposition` pins as NaN exactly where not moved. `main` (Task 4) holds the live records in `records: dict[tuple[str, int], dict]` keyed `(arm, seed)`, the parsed flags in `args` and the run's horizon in `horizon` when its final `return EXIT_OK` is reached.
- Consumes (Task 1, `src/mbfps/eval/trust.py`): `survival(crossings: np.ndarray, horizon: int) -> np.ndarray`.
- Consumes (Task 5, `src/mbfps/eval/trust_readings.py`): `ARMS_ORDER`, `TREATMENT`, `CONTROL`, `FAMILY`, `Contrast(estimate, se, z, n_windows)`, `Ratio(estimate, low, high)`, `ReadingOneInputs(delta_contrast, ratio_probe, ratio_free, cosine_contrast, corrected_contrast_a, corrected_contrast_b, alpha_boundary, crossing_contrast_probe, crossing_contrast_free, per_seed)`, `reading_one(inputs, z_fam, h=45) -> ReadingOne` (with `.status: Status`, `.reason: str`), `reading_two(survival, qs=Q_REPORTED) -> ReadingTwo`, `format_reading_one(r, inputs, z_fam, h=45) -> str` (its first line is the Reading 1 header, `--- Reading 1: does the h={h} gate reward slow drift?  (z_fam ..., family 8, treatment frozen_ssl, control random_vit)`), `format_reading_two(r) -> str` (first line `--- Reading 2: the horizon M4 designs around  (S(h) = ...; no verdict)`). This task passes the run's horizon as `h` to both and prints no header of its own for a computed reading.
- Consumes (existing, `src/mbfps/eval/pooling.py`): `CellSeries`, `require_compatible(cells)`, `pool_arm(cells) -> PooledMean` (fields read: `mean`, `se`, `z`, `windows`, `seeds`), `pool_ratio(cells, *, bootstrap=2000, seed=0)` (reads `.embedding` as numerator, `.noise` as denominator, over `.changed` rows), `paired_contrast(treatment_cells, control_cells)`, `cluster_threshold(family, clusters)`.
- Produces (contract, `scripts/trust_horizon.py`):
  - `cell_series(arm, seed, name, values: np.ndarray, changed: np.ndarray, record: dict) -> pooling.CellSeries` -- `values (n,)`, `changed (n,) bool`; `rung=name`, `channel="position"`; `embedding` / `noise` None; identity fields from the record; `ValueError` if the three `(n,)` shapes disagree.
  - `pooled_inputs(records: dict[tuple[str, int], dict], *, h: int = 45) -> tuple[ReadingOneInputs, int]` -- `(inputs, clusters)`; `KeyError` if any `ARMS_ORDER` arm lacks a record at any seed present; `ValueError` if a record's horizon is below `h`.
  - `survival_by_arm(records) -> dict[tuple[str, str], np.ndarray]` -- `(arm, channel) -> S(h)`, seeds stacked as draws, `channel in ("probe", "free")`.
  - `write_readings(out_dir: Path, text: str) -> Path` -- writes `runs/<out>/trust.txt`, returns the path.
  - `ArmSummary` (frozen dataclass `delta: PooledMean | None, cosine: PooledMean | None, ratio_probe: Ratio, ratio_free: Ratio, ratio_raw: float, moved: int, zero_displacement: int`) and `arm_summaries(records, inputs: ReadingOneInputs, *, h) -> dict[str, ArmSummary]` -- the per-arm block of spec 5 ("per-arm block, then contrasts, then the verdict lines"): `pool_arm` on each arm's own measurable cells for `Δ(h)` and `cos(h)`, the two ratios as `pooled_inputs` pooled them, `R_raw(h)` as the median of the moved draws' own `|d̂| / |d|` (for information), and the moved / zero-`d̂` counts summed over the arm's cells.
  - `_informational_crossing(records, arm, channel) -> Contrast` -- `arm - CONTROL` on `h×` per draw, the probe control's estimator; printed for `pixel_ae` (spec 3.2: "`pixel_ae`'s pair is printed for information") and decided on by nothing.

> RESOLVED AT RECONCILIATION: spec 3.1 pools `R_probe` and `R_free` as a **ratio of medians** -- `median(numerators) / median(denominators)` through `pool_ratio`, which needs the numerator and denominator series separately; the per-window ratios cannot recover them. Task 4's `trust_record` writes the `displacement` block -- `displacement {probe_hat [n][H] = |d̂|, probe_real [n][H] = |d̂_real|, free_hat [n][H] = ‖ê(h) − ê(0)‖, free_true [n][H] = ‖e(h) − e(0)‖}` -- beside the contract's keys, and this task reads it. The fabricated records in this task's tests carry it.

> RESOLVED AT RECONCILIATION: a pooling refusal (`pooling.IncompatibleCells`, `DegenerateNoise`, `StaleRecord` raised while pooling the trust records) PROPAGATES -- an uncaught traceback, exit 1 -- and no status is invented. Spec 5 fixes 0 / 11 / 12 / 14 / 30; on the shipped records none of these can fire (every cell scores the same 229 windows on one device, and a moved window's `|d̂_real|` and `‖e(h) − e(0)‖` medians are not 0), so one firing is a Task 4/6 defect, which is what Task 7's status table already says exit 1 means.

> `main` is extended below by replacing Task 4's final `return EXIT_OK`; at that point Task 4's `main` holds the live records in `records` (keyed `(arm, seed)`), the parsed flags in `args` and the run's horizon in `horizon`, by Task 4's Interfaces. Task 5's `format_reading_one` renders a NaN `z` as `nan` through `_fmt`; this task prints its own `n/a` line and reason before it, so the fixture test does not depend on the formatter's rendering.

- [ ] **Step 1: Write the failing tests for the glue**

Append to `tests/eval/test_trust_horizon_script.py`, after Task 4's last test. The block uses Task 4's header as it stands -- `script` and `diagnose` (the two modules loaded by path), `N_WINDOWS` (Task 4's eight fixture windows, NOT rebound here: this block's six fabricated windows are `N_FAB`), the `cell` fixture, `_argv`, and the imports `math`, `types`, `Path`, `np`, `pytest`, `StudyJob`, `run_job` -- and adds only what is new to it: `itertools`, `pooling`, and the Task 5 names.

```python
# ---------------------------------------------------------------------------
# Pooling glue, the two readings, trust.txt  (Task 6)
# ---------------------------------------------------------------------------
# Nothing statistical lives in the glue: every estimator is `pooling.py`'s and
# every rule is `trust_readings.py`'s. What can go wrong here is WIRING -- the
# wrong series under the wrong mask into the right estimator -- so the tests
# below hand `pooled_inputs` six fabricated windows from four episodes, two
# seeds, three arms, at h = 3, and assert every field of `ReadingOneInputs`
# against a number worked by hand from the same six windows (the clustered
# standard error is `sqrt(G / (G - 1) * sum_g (sum of residuals in g)^2) / n`,
# `pooling.cluster_standard_error`'s own definition, written out per test).

# Task 4's header stays as it is: `script`, `diagnose`, `N_WINDOWS` (its eight
# fixture windows), the `cell` fixture, `_argv`, and the imports `math`,
# `types`, `Path`, `np`, `pytest`, `StudyJob`, `run_job` are all reused, and
# only the genuinely new imports are added.
import itertools

import mbfps.eval.pooling as pooling
from mbfps.eval.trust_readings import (
    ARMS_ORDER,
    CONTROL,
    FAMILY,
    TREATMENT,
    Contrast,
    Ratio,
    ReadingOneInputs,
)

NAN = float("nan")
N_FAB = 6
"""Six fabricated windows -- NOT Task 4's `N_WINDOWS`, which is the fixture's
eight and is read by Task 4's tests at import time."""
H_FAB = 3
HI = H_FAB - 1
EPISODE = [0, 0, 1, 1, 2, 3]
"""Four episodes over six windows: fold A (even labels) is episodes 0 and 2,
windows {0, 1, 4}; fold B (odd) is episodes 1 and 3, windows {2, 3, 5}. Four,
not two, so that each fold still has two clusters after window 0 is dropped."""
MOVED = [False, True, True, True, True, True]
"""Window 0 did not move at h = 3 in any cell; every probe-based series is NaN
there and `margin` -- finite everywhere by construction -- must be masked."""


def _tile(column):
    """One per-window column at h = 3, repeated to every step: the glue reads
    column `h - 1` only, and a value that leaked from another column would be
    the same value, so the tiling hides nothing the tests care about."""
    return np.tile(np.asarray(column, dtype=float)[:, None], (1, H_FAB)).tolist()


def _fab_record(arm, seed, *, margin, cosine, held, boundary, crossing_probe, crossing_free,
                probe_hat, probe_real, free_hat, free_true, measurable=True, r2=0.3):
    """A trust record in the contract's shape, from per-window values at h = 3."""
    moved = np.asarray(MOVED)
    hat, real = np.asarray(probe_hat, float), np.asarray(probe_real, float)
    e_hat, e_true = np.asarray(free_hat, float), np.asarray(free_true, float)
    return {
        "arm": arm, "seed": seed, "context": 2, "horizon": H_FAB, "split_seed": 0,
        "device": "cpu", "torch_version": "2.x", "git_sha": "0" * 7,
        "episodes": {"val": ["ep_a.npz", "ep_b.npz", "ep_c.npz", "ep_d.npz"]},
        "windows": {"total": N_FAB, "episode": list(EPISODE)},
        "probe": {"selection_r2": r2, "measurable": measurable},
        "self_check": {
            "reference_position_max_delta": 0.0, "persistence_position_max_delta": 0.0,
            "windows_total_match": True, "windows_episode_match": True, "ok": True,
        },
        "crossing": {"probe": list(crossing_probe), "free": list(crossing_free)},
        "margin": _tile(margin),
        "ratio_probe": _tile(np.where(moved, hat / real, NAN)),
        "ratio_raw": _tile(np.where(moved, 1.0, NAN)),
        "ratio_free": _tile(np.where(moved, e_hat / e_true, NAN)),
        "cosine": _tile(cosine),
        "scale": {
            "alpha_a": [1.0] * H_FAB, "alpha_b": [1.0] * H_FAB,
            "score_a": [0.0] * H_FAB, "score_b": [0.0] * H_FAB,
            "held_out": _tile(held), "boundary": list(boundary), "folds_available": True,
        },
        "displacement": {
            "probe_hat": _tile(hat), "probe_real": _tile(real),
            "free_hat": _tile(e_hat), "free_true": _tile(e_true),
        },
        "counts": {"not_moved": [1] * H_FAB, "zero_displacement": [0] * H_FAB, "never_moved": 0},
        "nonfinite": {},
    }


def _fab_records():
    """Three arms x two seeds. The numbers are chosen so every pooled field has
    a closed-form answer; each test names the ones it reads."""
    ones = [1.0] * N_FAB
    return {
        ("pixel_ae", 0): _fab_record(
            "pixel_ae", 0, margin=[3] * 6, cosine=[0.5] * 6, held=[10] * 6,
            boundary=[False] * 3, crossing_probe=[3] * 6, crossing_free=[4] * 6,
            probe_hat=[6] * 6, probe_real=[3] * 6, free_hat=[1] * 6, free_true=[2] * 6,
        ),
        ("pixel_ae", 1): _fab_record(
            "pixel_ae", 1, margin=[3] * 6, cosine=[0.5] * 6, held=[10] * 6,
            boundary=[False] * 3, crossing_probe=[3] * 6, crossing_free=[4] * 6,
            probe_hat=[2] * 6, probe_real=[1] * 6, free_hat=[2] * 6, free_true=[4] * 6,
        ),
        ("frozen_ssl", 0): _fab_record(
            "frozen_ssl", 0, margin=[50, 4, 2, 2, 4, 3], cosine=[1.0] * 6, held=[9] * 6,
            boundary=[False] * 3, crossing_probe=[4, 4, 2, NAN, 3, 4], crossing_free=[4] * 6,
            probe_hat=[100, 1, 1, 3, 3, 3], probe_real=[100, 2, 2, 2, 2, 2],
            free_hat=[3] * 6, free_true=[2] * 6,
        ),
        ("frozen_ssl", 1): _fab_record(
            "frozen_ssl", 1, margin=[52, 6, 4, 4, 6, 5],
            cosine=[0.5, 0.5, 0.5, 0.5, 0.5, NAN], held=[11] * 6,
            boundary=[False, False, True], crossing_probe=[4, 1, 2, 2, 3, 4],
            crossing_free=[4] * 6, probe_hat=[100, 2, 2, 2, 6, 6],
            probe_real=[100, 4, 4, 4, 4, 4], free_hat=[6] * 6, free_true=[4] * 6,
        ),
        ("random_vit", 0): _fab_record(
            "random_vit", 0, margin=ones, cosine=[0.25, 0.25, 0.75, 0.75, 0.25, 0.25],
            held=[0, 12, 14, 14, 16, 16], boundary=[False] * 3, crossing_probe=[1] * 6,
            crossing_free=[4] * 6, probe_hat=[0] * 6, probe_real=[5] * 6,
            free_hat=[1] * 6, free_true=[4] * 6,
        ),
        ("random_vit", 1): _fab_record(
            "random_vit", 1, margin=ones, cosine=[0.25, 0.25, 0.75, 0.75, 0.25, 0.25],
            held=[0, 12, 14, 14, 16, 16], boundary=[False] * 3, crossing_probe=[1] * 6,
            crossing_free=[4] * 6, probe_hat=[0] * 6, probe_real=[5] * 6,
            free_hat=[1] * 6, free_true=[4] * 6,
        ),
    }


# --- cell_series -------------------------------------------------------------


def test_cell_series_carries_the_moved_mask_as_changed_and_the_records_identity():
    """`changed` is what `pool_arm` and `paired_contrast` drop windows by, so
    the mask handed in must come out as handed in -- not `window_steps_changed
    > 0`, which is the ladder's mask and not the trust pass's. The identity
    fields are what `require_compatible` refuses a pool over, so each is
    asserted to be the RECORD's, not a default."""
    record = _fab_records()[("frozen_ssl", 1)]
    values = np.array([50.0, 4.0, 2.0, 2.0, 4.0, 3.0])
    changed = np.array(MOVED)
    series = script.cell_series("frozen_ssl", 1, "margin", values, changed, record)
    assert isinstance(series, pooling.CellSeries)
    assert series.arm == "frozen_ssl" and series.seed == 1
    assert series.rung == "margin" and series.channel == "position"
    np.testing.assert_array_equal(series.delta, values)
    np.testing.assert_array_equal(series.changed, changed)
    assert series.changed.dtype == bool
    np.testing.assert_array_equal(series.episode, np.array(EPISODE))
    assert series.embedding is None and series.noise is None
    assert series.windows_total == N_FAB
    assert series.val == ("ep_a.npz", "ep_b.npz", "ep_c.npz", "ep_d.npz")
    assert series.horizon == H_FAB and series.context == 2
    assert series.device == "cpu" and series.torch_version == "2.x"


def test_cell_series_refuses_a_series_that_is_not_one_value_per_window():
    """A `(n, H)` array handed in where `(n,)` was meant would broadcast
    silently inside `np.mean` and cluster by the wrong axis; a `changed` of the
    wrong length would drop the wrong windows. Both are refused by name."""
    record = _fab_records()[("pixel_ae", 0)]
    with pytest.raises(ValueError, match="must all be"):
        script.cell_series("pixel_ae", 0, "margin", np.zeros((6, 3)), np.ones(6, bool), record)
    with pytest.raises(ValueError, match="must all be"):
        script.cell_series("pixel_ae", 0, "margin", np.zeros(6), np.ones(5, bool), record)


# --- pooled_inputs -----------------------------------------------------------


def test_pooled_inputs_pairs_the_three_arms_as_a_minus_b_and_reads_a_finite_family_threshold():
    """The three unordered pairs in `ARMS_ORDER` order, each estimate `a - b`
    on the per-window SEED-MEAN margin over the MOVED windows, clustered by
    episode. frozen_ssl - random_vit: seed means [51, 5, 3, 3, 5, 4] minus
    [1, 1, 1, 1, 1, 1], window 0 dropped (not moved), leaves [4, 2, 2, 4, 3]
    on episodes [0, 1, 1, 2, 3]: mean 3, residuals [1, -1, -1, 1, 0], cluster
    sums [1, -2, 1, 0], sum of squares 6, times G / (G - 1) = 4 / 3 is 8, so
    se = sqrt(8) / 5 and z = 15 / (2 sqrt 2). pixel_ae - frozen_ssl: [3 - 5,
    3 - 3, 3 - 3, 3 - 5, 3 - 4] = [-2, 0, 0, -2, -1], mean -1, the same
    residual pattern, z = -5 / (2 sqrt 2). pixel_ae - random_vit is a constant
    2, se exactly 0, and `pooling._z` reads +inf there. `clusters` is the
    number of distinct episode labels the windows carry -- four -- not the
    kept-window count, and `cluster_threshold(FAMILY, 4)` is finite."""
    inputs, clusters = script.pooled_inputs(_fab_records(), h=H_FAB)
    assert isinstance(inputs, ReadingOneInputs)
    assert list(inputs.delta_contrast) == list(itertools.combinations(ARMS_ORDER, 2)) == [
        ("pixel_ae", "frozen_ssl"), ("pixel_ae", "random_vit"), ("frozen_ssl", "random_vit"),
    ]
    se = 2 * math.sqrt(2) / 5
    fr = inputs.delta_contrast[("frozen_ssl", "random_vit")]
    assert isinstance(fr, Contrast)
    assert fr.estimate == pytest.approx(3.0) and fr.se == pytest.approx(se)
    assert fr.z == pytest.approx(15 / (2 * math.sqrt(2))) and fr.n_windows == 5
    pf = inputs.delta_contrast[("pixel_ae", "frozen_ssl")]
    assert pf.estimate == pytest.approx(-1.0) and pf.se == pytest.approx(se)
    assert pf.z == pytest.approx(-5 / (2 * math.sqrt(2))) and pf.n_windows == 5
    pr = inputs.delta_contrast[("pixel_ae", "random_vit")]
    assert pr.estimate == pytest.approx(2.0) and pr.se == 0.0 and pr.z == math.inf
    assert clusters == 4
    z_fam = pooling.cluster_threshold(FAMILY, clusters)
    assert math.isfinite(z_fam) and z_fam == pooling.cluster_threshold(8, 4)


def test_pooled_inputs_cosine_and_corrected_contrasts_exclude_nan_windows_and_split_the_folds():
    """cos(3), frozen_ssl - random_vit on the seed-mean cosine: frozen_ssl
    seed 1 has a zero imagined displacement at window 5 (cosine NaN), so that
    window leaves the pairing along with unmoved window 0 -- kept [1, 2, 3, 4]
    on episodes [0, 1, 1, 2] with differences [0.75 - 0.25, 0.75 - 0.75,
    0.75 - 0.75, 0.75 - 0.25] = [0.5, 0, 0, 0.5]: mean 0.25, residuals
    [+-0.25], cluster sums [0.25, -0.5, 0.25], sum of squares 0.375, times
    3 / 2 is 0.5625, se = 0.75 / 4, z = 4 / 3. The held-out contrasts are on
    each FOLD's rows alone: fold A (even labels) keeps windows {1, 4} with
    seed-mean held-out [10, 10] - [12, 16] = [-2, -6] on episodes [0, 2],
    mean -4, cluster sums [2, -2], se = sqrt(2 * 8) / 2 = 2, z = -2; fold B
    keeps {2, 3, 5}: [10, 10, 10] - [14, 14, 16] = [-4, -4, -6] on [1, 1, 3],
    mean -14/3, residuals [2/3, 2/3, -4/3], cluster sums [4/3, -4/3], sum of
    squares 32/9, times 2 is 64/9, se = (8/3) / 3 = 8/9, z = -5.25."""
    inputs, _ = script.pooled_inputs(_fab_records(), h=H_FAB)
    cos = inputs.cosine_contrast
    assert cos.estimate == pytest.approx(0.25) and cos.se == pytest.approx(0.1875)
    assert cos.z == pytest.approx(4 / 3) and cos.n_windows == 4
    a = inputs.corrected_contrast_a
    assert a.estimate == pytest.approx(-4.0) and a.se == pytest.approx(2.0)
    assert a.z == pytest.approx(-2.0) and a.n_windows == 2
    b = inputs.corrected_contrast_b
    assert b.estimate == pytest.approx(-14 / 3) and b.se == pytest.approx(8 / 9)
    assert b.z == pytest.approx(-5.25) and b.n_windows == 3


def test_pooled_inputs_crossing_contrasts_are_per_draw_with_the_seeds_stacked():
    """Spec 2.2: nothing about a crossing step is ever seed-averaged. The
    probe-channel contrast pairs frozen_ssl's twelve (window, seed) draws
    [4, 4, 2, NaN, 3, 4 | 4, 1, 2, 2, 3, 4] with random_vit's twelve 1s; the
    NaN draw (never moved) leaves alone, so eleven differences [3, 3, 1, 2, 3
    | 3, 0, 1, 1, 2, 3] on episodes [0, 0, 1, 2, 3 | 0, 0, 1, 1, 2, 3]: sum 22,
    mean 2, residuals [1, 1, -1, 0, 1 | 1, -2, -1, -1, 0, 1], cluster sums
    [1, -3, 0, 2], sum of squares 14, times 4 / 3 is 56 / 3, se = sqrt(56 / 3)
    / 11. Seed-averaging first would pair five windows, not eleven draws. The
    free channel is 4 against 4 everywhere: estimate 0, se 0, and `_z` reads
    0 / 0 as 0 over all twelve draws."""
    inputs, _ = script.pooled_inputs(_fab_records(), h=H_FAB)
    probe = inputs.crossing_contrast_probe
    assert probe.estimate == pytest.approx(2.0)
    assert probe.se == pytest.approx(math.sqrt(56 / 3) / 11)
    assert probe.z == pytest.approx(22 * math.sqrt(3 / 56)) and probe.n_windows == 11
    free = inputs.crossing_contrast_free
    assert free.estimate == 0.0 and free.se == 0.0 and free.z == 0.0 and free.n_windows == 12


def test_pooled_inputs_ratios_are_ratios_of_medians_with_each_cell_in_units_of_its_own_denominator():
    """`pool_ratio`'s estimand, on the moved windows with the seeds stacked.
    frozen_ssl's probe ratio: seed 0 has |d_hat| [1, 1, 3, 3, 3] over
    |d_hat_real| [2, 2, 2, 2, 2] and seed 1 has [2, 2, 2, 6, 6] over [4, 4, 4,
    4, 4]; each cell in units of its own denominator median gives numerators
    [.5, .5, 1.5, 1.5, 1.5 | .5, .5, .5, 1.5, 1.5] with median 1.0 over
    denominators of median 1.0: ratio 1.0. A raw stacked median would read
    2.5 / 3. pixel_ae is 2x its denominator in both seeds (2.0, interval
    exactly [2, 2]); random_vit's numerator is 0 everywhere (0.0, [0, 0]).
    The probe-free ratio, which no probe touches: pixel_ae 0.5, frozen_ssl
    1.5, random_vit 0.25, each constant per window so each interval is the
    point."""
    inputs, _ = script.pooled_inputs(_fab_records(), h=H_FAB)
    fr = inputs.ratio_probe["frozen_ssl"]
    assert isinstance(fr, Ratio)
    assert fr.estimate == pytest.approx(1.0) and fr.low <= 1.0 <= fr.high
    assert inputs.ratio_probe["pixel_ae"] == Ratio(estimate=2.0, low=2.0, high=2.0)
    assert inputs.ratio_probe["random_vit"] == Ratio(estimate=0.0, low=0.0, high=0.0)
    assert inputs.ratio_free["pixel_ae"] == Ratio(estimate=0.5, low=0.5, high=0.5)
    assert inputs.ratio_free["frozen_ssl"] == Ratio(estimate=1.5, low=1.5, high=1.5)
    assert inputs.ratio_free["random_vit"] == Ratio(estimate=0.25, low=0.25, high=0.25)
    assert list(inputs.ratio_probe) == list(inputs.ratio_free) == list(ARMS_ORDER)


def test_pooled_inputs_carries_alpha_boundaries_at_h_and_the_same_inputs_within_each_seed():
    """A boundary at h in ANY seed of an arm flags the arm (frozen_ssl seed 1
    is on the boundary at step 3 and nowhere else). The per-seed inputs are
    computed from that seed's cells alone and carry no seeds of their own:
    seed 0's frozen_ssl - random_vit margin is [4, 2, 2, 4, 3] - 1, mean 2,
    not the pooled 3; seed 0's frozen_ssl boundary is False."""
    inputs, _ = script.pooled_inputs(_fab_records(), h=H_FAB)
    assert inputs.alpha_boundary == {"pixel_ae": False, "frozen_ssl": True, "random_vit": False}
    assert sorted(inputs.per_seed) == [0, 1]
    seed0, seed1 = inputs.per_seed[0], inputs.per_seed[1]
    assert isinstance(seed0, ReadingOneInputs) and seed0.per_seed is None and seed1.per_seed is None
    assert seed0.delta_contrast[("frozen_ssl", "random_vit")].estimate == pytest.approx(2.0)
    assert seed0.delta_contrast[("frozen_ssl", "random_vit")].n_windows == 5
    assert seed0.alpha_boundary["frozen_ssl"] is False
    assert seed1.alpha_boundary["frozen_ssl"] is True
    assert seed0.crossing_contrast_probe.n_windows == 5, "one seed: five finite draws"


def test_an_unmeasurable_cell_leaves_the_probe_based_pooling_and_only_that():
    """Spec 3.1: a cell enters the probe-based pooling iff its persistence-to-
    floor band is positive. random_vit seed 1 is marked unmeasurable and given
    a margin of 3 everywhere: excluded, random_vit's seed mean stays 1 and the
    contrast reads 3; included, it would be 2 and read 2. The per-draw probe
    crossing contrast pairs the seeds measurable in BOTH arms (seed 0 only:
    five finite draws), while the probe-free crossing contrast and ratio keep
    all twelve draws. With every cell unmeasurable there is nothing to pool:
    every probe-based field is NaN with zero windows, no boundary is flagged,
    and the probe-free fields are untouched."""
    records = _fab_records()
    records[("random_vit", 1)] = _fab_record(
        "random_vit", 1, margin=[3] * 6, cosine=[0.25] * 6, held=[0, 12, 14, 14, 16, 16],
        boundary=[False] * 3, crossing_probe=[1] * 6, crossing_free=[4] * 6,
        probe_hat=[0] * 6, probe_real=[5] * 6, free_hat=[1] * 6, free_true=[4] * 6,
        measurable=False,
    )
    inputs, _ = script.pooled_inputs(records, h=H_FAB)
    assert inputs.delta_contrast[("frozen_ssl", "random_vit")].estimate == pytest.approx(3.0)
    assert inputs.crossing_contrast_probe.n_windows == 5
    assert inputs.crossing_contrast_free.n_windows == 12
    assert inputs.ratio_free["random_vit"].estimate == pytest.approx(0.25)
    # The per-arm block pools the same measurable cells: random_vit's
    # pooled delta is seed 0's 1, not (1 + 3) / 2, and names the one seed.
    block = script.arm_summaries(records, inputs, h=H_FAB)
    assert block["random_vit"].delta.mean == pytest.approx(1.0)
    assert block["random_vit"].delta.seeds == (0,)

    for record in records.values():
        record["probe"]["measurable"] = False
    none, _ = script.pooled_inputs(records, h=H_FAB)
    nothing = script.arm_summaries(records, none, h=H_FAB)
    assert all(nothing[arm].delta is None and nothing[arm].cosine is None for arm in ARMS_ORDER)
    assert nothing["frozen_ssl"].ratio_free.estimate == pytest.approx(1.5)
    for pair in itertools.combinations(ARMS_ORDER, 2):
        assert math.isnan(none.delta_contrast[pair].z) and none.delta_contrast[pair].n_windows == 0
    assert math.isnan(none.cosine_contrast.z) and math.isnan(none.crossing_contrast_probe.z)
    assert math.isnan(none.corrected_contrast_a.z) and math.isnan(none.corrected_contrast_b.z)
    assert all(math.isnan(none.ratio_probe[arm].estimate) for arm in ARMS_ORDER)
    assert none.alpha_boundary == {arm: False for arm in ARMS_ORDER}
    assert none.crossing_contrast_free.n_windows == 12
    assert none.ratio_free["frozen_ssl"].estimate == pytest.approx(1.5)


def test_pooled_inputs_refuses_a_missing_cell_and_a_step_past_the_horizon():
    """Reading 1 is three arms at every seed; a pool over five cells printed
    under six cells' names is the table this repo has been burned by. And
    column h - 1 of a record with a shorter horizon is an IndexError at best
    and the wrong step at worst."""
    records = _fab_records()
    del records[("pixel_ae", 1)]
    with pytest.raises(KeyError, match="pixel_ae seed 1"):
        script.pooled_inputs(records, h=H_FAB)
    with pytest.raises(ValueError, match="no step h=4"):
        script.pooled_inputs(_fab_records(), h=4)


def test_the_r2_filter_marks_low_r2_cells_unmeasurable_without_touching_the_originals():
    """The sensitivity line of spec 3.1 recomputes every probe-based
    statistic with cells of selection R^2 < 0.1 excluded, and changes no
    verdict -- so the filter must produce NEW records and leave the ones the
    verdict was read from exactly as they were."""
    records = _fab_records()
    records[("frozen_ssl", 1)]["probe"]["selection_r2"] = 0.05
    filtered = script._r2_filtered(records, script.R2_SENSITIVITY)
    assert script.R2_SENSITIVITY == 0.1
    assert sorted(filtered) == sorted(records)
    assert filtered[("frozen_ssl", 1)]["probe"]["measurable"] is False
    assert filtered[("frozen_ssl", 0)]["probe"]["measurable"] is True
    assert records[("frozen_ssl", 1)]["probe"]["measurable"] is True
    assert filtered[("frozen_ssl", 1)]["margin"] is records[("frozen_ssl", 1)]["margin"], (
        "the series are shared, not copied: only the probe block is rewritten"
    )


# --- the per-arm block, pixel_ae's pair, the self-check table -----------------


def test_the_per_arm_block_pools_each_arms_own_margin_and_cosine_and_reads_the_counts():
    """Spec 5 prints "per-arm block, then contrasts, then the verdict lines":
    the block is `pool_arm` on ONE arm's measurable cells, the seed-mean
    per window over the moved-and-finite windows, clustered by episode.
    pixel_ae's margin is 3 in both seeds on every moved window: mean 3, se
    exactly 0, 5 windows. frozen_ssl's seed-mean margin over windows 1-5 is
    [5, 3, 3, 5, 4]: mean 4, residuals [1, -1, -1, 1, 0], cluster sums
    [1, -2, 1, 0], se = sqrt(8) / 5 -- the same ruler as its contrast with
    random_vit, whose margin is 1 everywhere. random_vit's cosine over the
    same windows is [0.25, 0.75, 0.75, 0.25, 0.25]: mean 0.45, residuals
    [-0.2, 0.3, 0.3, -0.2, -0.2], cluster sums [-0.2, 0.6, -0.2, -0.2], sum
    of squares 0.48, times 4 / 3 is 0.64, se = 0.8 / 5 = 0.16. frozen_ssl's
    cosine loses seed 1's NaN window 5 to the finiteness mask: four windows
    at (1.0 + 0.5) / 2. The ratios are `pooled_inputs`' own objects; R_raw is
    the median of the moved draws' per-window |d_hat| / |d| (1.0 everywhere
    in the fabricated records, NaN on the unmoved window and so excluded);
    the counts are summed over the arm's two cells."""
    inputs, _ = script.pooled_inputs(_fab_records(), h=H_FAB)
    block = script.arm_summaries(_fab_records(), inputs, h=H_FAB)
    assert list(block) == list(ARMS_ORDER)
    assert all(isinstance(v, script.ArmSummary) for v in block.values())
    pa = block["pixel_ae"]
    assert isinstance(pa.delta, pooling.PooledMean)
    assert pa.delta.mean == pytest.approx(3.0) and pa.delta.se == 0.0 and pa.delta.windows == 5
    assert pa.cosine.mean == pytest.approx(0.5) and pa.cosine.se == 0.0
    fs = block["frozen_ssl"]
    assert fs.delta.mean == pytest.approx(4.0) and fs.delta.se == pytest.approx(2 * math.sqrt(2) / 5)
    assert fs.delta.windows == 5 and fs.delta.seeds == (0, 1)
    assert fs.cosine.mean == pytest.approx(0.75) and fs.cosine.windows == 4
    rv = block["random_vit"]
    assert rv.delta.mean == pytest.approx(1.0) and rv.delta.se == 0.0
    assert rv.cosine.mean == pytest.approx(0.45) and rv.cosine.se == pytest.approx(0.16)
    assert rv.cosine.windows == 5
    for arm in ARMS_ORDER:
        assert block[arm].ratio_probe == inputs.ratio_probe[arm]
        assert block[arm].ratio_free == inputs.ratio_free[arm]
        assert block[arm].ratio_raw == 1.0
        assert (block[arm].moved, block[arm].zero_displacement) == (10, 0)
    table = script._arm_table(block, H_FAB).splitlines()
    assert table[0].split()[0] == "arm" and len(table) == 4
    assert [line.split()[0] for line in table[1:]] == list(ARMS_ORDER)
    assert "n/a" not in "\n".join(table)


def test_pixel_aes_crossing_pair_is_computed_for_information_with_the_seeds_stacked():
    """Spec 3.2: "`pixel_ae`'s pair is printed for information." The same
    per-draw estimator as the probe control's contrast, `arm - random_vit`:
    pixel_ae's probe crossings are 3 in every draw against random_vit's 1 --
    twelve differences of exactly 2, se 0, and `_z` reads +inf -- and the
    free channel is 4 against 4, 0 over twelve draws. frozen_ssl's pair
    through the same function is the contrast `pooled_inputs` decides on, so
    the two must agree. Nothing reads it: `ReadingOneInputs` has no slot."""
    probe = script._informational_crossing(_fab_records(), "pixel_ae", "probe")
    assert isinstance(probe, Contrast)
    assert probe.estimate == pytest.approx(2.0) and probe.se == 0.0
    assert probe.z == math.inf and probe.n_windows == 12
    free = script._informational_crossing(_fab_records(), "pixel_ae", "free")
    assert free.estimate == 0.0 and free.n_windows == 12
    inputs, _ = script.pooled_inputs(_fab_records(), h=H_FAB)
    assert script._informational_crossing(_fab_records(), TREATMENT, "probe") == inputs.crossing_contrast_probe


def test_the_self_check_table_has_one_row_per_cell_in_arm_order_with_the_deltas_and_the_probe():
    """One row per record, in ARMS_ORDER then seed -- not alphabetical, which
    would put frozen_ssl first -- carrying both deltas, the window count,
    the episode-label match, the probe's R^2 and measurability, the
    never-moved count and `ok`. What Task 7 fills its self-check table from."""
    records = _fab_records()
    records[("random_vit", 1)]["probe"]["measurable"] = False
    records[("random_vit", 1)]["self_check"]["reference_position_max_delta"] = 2.0 ** -40
    lines = script._self_check_table(records).splitlines()
    assert lines[0].split()[0] == "cell" and len(lines) == 7
    assert [line.split()[0] for line in lines[1:]] == [
        f"{arm}/s{seed}" for arm in ARMS_ORDER for seed in (0, 1)
    ]
    assert lines[1].split()[1:3] == ["0.0e+00", "0.0e+00"]
    last = lines[-1].split()
    assert last[0] == "random_vit/s1" and last[1] == "9.1e-13" and "False" in last
    assert "True" in lines[1].split() and "0.300" in lines[1]


# --- survival_by_arm, write_readings ------------------------------------------


def test_survival_by_arm_stacks_the_seeds_per_channel():
    """`survival` over the (window, seed) draws that moved. frozen_ssl's probe
    crossings over both seeds are [4, 4, 2, NaN, 3, 4, 4, 1, 2, 2, 3, 4] --
    eleven finite draws, one at 1, three at 2, two at 3, five at 4 (= horizon
    + 1, never crossed): S(0) = 1, S(1) = 10/11, S(2) = 7/11, S(3) = 5/11.
    random_vit's probe crossings are all 1: S = [1, 0, 0, 0]. Every arm's
    free channel is 4 everywhere: S = [1, 1, 1, 1]."""
    curves = script.survival_by_arm(_fab_records())
    assert sorted(curves) == sorted(itertools.product(ARMS_ORDER, ("probe", "free")))
    np.testing.assert_allclose(curves[("frozen_ssl", "probe")], [1.0, 10 / 11, 7 / 11, 5 / 11])
    np.testing.assert_allclose(curves[("random_vit", "probe")], [1.0, 0.0, 0.0, 0.0])
    for arm in ARMS_ORDER:
        np.testing.assert_allclose(curves[(arm, "free")], [1.0, 1.0, 1.0, 1.0])


def test_write_readings_writes_trust_txt_under_out_and_returns_its_path(tmp_path):
    path = script.write_readings(tmp_path, "--- Reading 1 ---\nline\n")
    assert path == tmp_path / "trust.txt"
    assert path.read_text() == "--- Reading 1 ---\nline\n"


# --- main on the fixture: the NaN path ------------------------------------------


@pytest.fixture
def trust_run(tmp_path, small_buffer, capsys):
    """Three arms at seed 0 on the shared fixture (~3 s per cell: `run_job` for
    five steps, then `diagnose_dynamics.main` to write the diagnostic the
    self-check reads), then `trust_horizon.main` over all three. The fixture's
    split holds ONE validation episode, so every clustered standard error and
    `cluster_threshold(FAMILY, 1)` are NaN."""
    for arm in ARMS_ORDER:
        run_job(StudyJob(arm, 0), small_buffer, tmp_path,
                steps=5, seq_len=4, context=2, horizon=3, device="cpu")
        assert diagnose.main([
            "--out", str(tmp_path), "--data", str(small_buffer.root), "--arms", arm,
            "--seeds", "0", "--context", "2", "--horizon", "3", "--ks", "1", "3",
            "--device", "cpu",
        ]) == diagnose.EXIT_OK
    capsys.readouterr()
    status = script.main([
        "--out", str(tmp_path), "--data", str(small_buffer.root), "--arms", *ARMS_ORDER,
        "--seeds", "0", "--device", "cpu",
    ])
    return types.SimpleNamespace(status=status, out=capsys.readouterr().out, out_dir=tmp_path)


def test_the_fixture_run_writes_trust_txt_with_both_readings(trust_run):
    """Exit 0, and the text on disk is the text on screen, with this task's
    own section header for each reading -- so a reader of `trust.txt` alone
    can find both."""
    assert trust_run.status == script.EXIT_OK
    text = (trust_run.out_dir / "trust.txt").read_text()
    # The section order of `_readings_text`: the self-check table, the
    # pooling notes, the per-arm block, Reading 1 (Task 5's header, from
    # `format_reading_one` at THIS run's horizon), pixel_ae's h_x pair for
    # information, the sensitivity block, Reading 2 (Task 5's header).
    assert "--- self-check per cell" in text
    assert "--- per arm at h = 3" in text
    assert "--- Reading 1: does the h=3 gate reward slow drift?" in text
    assert "for information: h_x probe pixel_ae - random_vit" in text
    assert "--- Reading 2: the horizon M4 designs around" in text
    assert text.count("--- Reading 1:") == 1 and text.count("--- Reading 2:") == 1
    assert text.rstrip("\n") in trust_run.out
    order = [text.index(s) for s in (
        "--- self-check per cell", "clusters: 1 validation", "--- per arm at h = 3",
        "--- Reading 1:", "for information: h_x probe", "--- sensitivity", "--- Reading 2:",
    )]
    assert order == sorted(order)
    for arm in ARMS_ORDER:
        assert f"{arm}/s0" in text[:text.index("clusters:")], "the self-check table names every cell"


def test_the_fixture_run_says_why_every_contrast_is_n_a(trust_run):
    """One validation episode: the episode-clustered standard error needs at
    least two clusters, so every pooled z is NaN, every contrast is n/a, and
    Reading 1 cannot find a best-delta arm -- NOT_TESTABLE, by `reading_one`'s
    own order. The reason is printed by THIS script, before the formatted
    reading, so it does not depend on how the formatter renders a NaN."""
    text = (trust_run.out_dir / "trust.txt").read_text()
    assert "clusters: 1 validation episode(s)" in text
    assert "at least two" in text and "n/a" in text
    assert "reading 1: NOT_TESTABLE" in text
    assert "sensitivity" in text and "changes no verdict" in text


def test_a_single_arm_run_still_exits_ok_and_says_both_readings_are_not_computed(cell, capsys):
    """Task 4's fixture is ONE arm at one seed. Reading 1 pools all three
    arms and Reading 2 needs every arm's survival curve (`reading_two`
    refuses a missing `(arm, channel)` by name), so neither is computed --
    and the run says so under BOTH headers, still exits 0, still writes
    trust.txt and the self-check table, rather than ending in
    `reading_two`'s KeyError after nine clean cells."""
    assert script.main(_argv(cell)) == script.EXIT_OK
    text = (cell.out / "trust.txt").read_text()
    assert text.count("not computed:") == 2
    assert "Reading 1 pools all of" in text
    assert "Reading 2 needs every arm's survival curve" in text
    assert "['random_vit']" in text
    assert "--- self-check per cell" in text and "random_vit/s1" in text
    assert "--- Reading 1:" in text and "--- Reading 2:" in text
    assert text.rstrip("\n") in capsys.readouterr().out
```

Why these and not fewer. The Δ, cosine, held-out and crossing tests each pin a *different mask*: `margin` is finite on the unmoved window and only the moved mask drops it; `cosine` has a NaN only from the zero-displacement window, so a `changed = moved` without the finiteness AND lets a NaN into the mean; the held-out contrasts are the only ones restricted to a fold, and the two folds have different hand answers, so swapping A and B is caught; the crossing test's eleven draws (not five windows) is what tells stacking from seed-averaging, and its NaN draw exists in one seed only so per-draw exclusion is distinguishable from per-window. The ratio test's two frozen_ssl cells have denominators 2 and 4, so a raw stacked median (2.5 / 3) is not the normalised one (1.0). The unmeasurable test changes the excluded cell's margin so that inclusion and exclusion read different numbers, and asserts the probe-free fields did not move. The per-seed test asserts a per-seed number that differs from the pooled one, so per-seed inputs computed from all records are caught. The fixture tests assert only what this task prints itself.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_trust_horizon_script.py -q -p no:cacheprovider -k "cell_series or pooled_inputs or unmeasurable or r2_filter or per_arm_block or pixel_aes_crossing or self_check_table or survival_by_arm or write_readings or fixture_run or single_arm"`
Expected: **18 failed** -- fifteen with `AttributeError: module 'trust_horizon_script' has no attribute 'cell_series'` (`pooled_inputs`, `arm_summaries`, `_informational_crossing`, `_self_check_table`, `_r2_filtered` / `R2_SENSITIVITY`, `survival_by_arm`, `write_readings` respectively), and the three `main` tests (the two `fixture_run` ones and the single-arm one) with `FileNotFoundError: ... trust.txt` after `main` returned 0 without writing it. Task 4's own tests in the file are unaffected (`-k` selects the new ones; run the whole file once to confirm nothing else moved: the count of passes is Task 4's, and its `N_WINDOWS` still reads 8).

- [ ] **Step 3: Implement the glue functions**

Three edits to `scripts/trust_horizon.py`.

(a) The imports at the top of the file gain (keep Task 4's, add these; `numpy` and `Path` are already there):

```python
import dataclasses
import itertools

import mbfps.eval.pooling as pooling
from mbfps.eval.trust import survival
from mbfps.eval.trust_readings import (
    ARMS_ORDER,
    CONTROL,
    FAMILY,
    TREATMENT,
    Contrast,
    Ratio,
    ReadingOneInputs,
    format_reading_one,
    format_reading_two,
    reading_one,
    reading_two,
)
```

(b) Append after `write_trust_record` (Task 4's last per-cell function, before `main`):

```python
# ---------------------------------------------------------------------------
# Pooling glue: per-cell records -> the inputs the two readings are decided on.
# ---------------------------------------------------------------------------
# Every estimator is `pooling.py`'s -- the same clustered ruler the ladder was
# read against -- and every rule is `trust_readings.py`'s. What is decided here
# is only WHICH series goes in under WHICH mask: means and contrasts on the
# per-window seed-mean series over the moved windows; the crossing contrasts
# on per-(window, seed) draws with the seeds stacked, because spec 2.2 says
# nothing about a crossing step is ever seed-averaged; ratios as ratios of
# medians over the moved windows' numerator and denominator series through
# `pool_ratio`, which puts each cell in units of its own denominator median.

BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
"""`pool_ratio`'s interval: 2000 episode resamples, seed 0 -- spec 3.1."""
R2_SENSITIVITY = 0.1
"""Cells whose probe selection R^2 is below this are dropped from the
SENSITIVITY line only (spec 3.1) -- chosen knowing pixel_ae/s1 reads 0.018 and
every other shipped cell >= 0.249. It changes no verdict."""
STACKED_SEED = -1
"""The `seed` a CellSeries carries when it holds every seed's draws stacked;
`CellSeries.seed` is identity for `require_compatible`, and -1 is no seed."""


def cell_series(arm, seed, name, values: np.ndarray, changed: np.ndarray, record: dict) -> pooling.CellSeries:
    """One per-window series of one cell, in the shape the pool reads.

    `changed` is the trust pass's own mask -- the moved windows, less the NaN
    ones -- never the ladder's `window_steps_changed`. The identity fields
    (windows.episode, episodes.val, horizon, context, device, torch_version)
    are the record's, so `require_compatible` refuses a pool over cells that
    did not score the same windows on the same device, as it does for the
    ladder. Refuses a `values` or `changed` that is not one entry per window:
    an `(n, H)` array here would broadcast inside `np.mean` and cluster by
    the wrong axis.
    """
    values = np.asarray(values, dtype=float)
    changed = np.asarray(changed, dtype=bool)
    episode = np.asarray(record["windows"]["episode"], dtype=int)
    if not (values.shape == changed.shape == episode.shape):
        raise ValueError(
            f"{arm} seed {seed} {name!r}: values {values.shape}, changed {changed.shape} and "
            f"windows.episode {episode.shape} must all be (n_windows,)"
        )
    return pooling.CellSeries(
        arm=arm, seed=int(seed), rung=name, channel="position",
        delta=values, changed=changed, episode=episode, embedding=None, noise=None,
        windows_total=int(record["windows"]["total"]), val=tuple(record["episodes"]["val"]),
        horizon=int(record["horizon"]), context=int(record["context"]),
        device=str(record["device"]), torch_version=str(record["torch_version"]),
    )


def _stacked(cells) -> pooling.CellSeries:
    """One series holding every cell's draws end to end, the episode labels
    tiled with them, so `paired_contrast` on one stacked series per arm is
    the paired PER-DRAW difference clustered by episode. Both arms stack the
    same seeds in the same order, so their tiled labels are identical and
    `_require_same_windows` accepts the pair."""
    return dataclasses.replace(
        cells[0], seed=STACKED_SEED,
        delta=np.concatenate([c.delta for c in cells]),
        changed=np.concatenate([c.changed for c in cells]),
        episode=np.concatenate([c.episode for c in cells]),
        windows_total=sum(c.windows_total for c in cells),
    )


def _stacked_crossings(records: dict, arm: str, channel: str, seeds) -> pooling.CellSeries:
    """One series per arm holding the per-(window, seed) crossing draws of
    `seeds` end to end -- spec 2.2: nothing about a crossing step is ever
    seed-averaged -- a never-moved draw (NaN) leaving on its own."""
    cells = []
    for seed in seeds:
        record = records[(arm, seed)]
        crossing = np.asarray(record["crossing"][channel], dtype=float)
        cells.append(
            cell_series(arm, seed, f"crossing_{channel}", crossing, np.isfinite(crossing), record)
        )
    return _stacked(cells)


def _moved_series(arm: str, seed: int, name: str, record: dict, hi: int) -> pooling.CellSeries:
    """Column `hi` of the record's `name` series under moved-and-finite: the
    mask every per-window mean and contrast is pooled over."""
    values = _column(record, name, hi)
    return cell_series(arm, seed, name, values, _moved(record, hi) & np.isfinite(values), record)


def _pooled_mean(cells) -> pooling.PooledMean | None:
    """`pool_arm` on the cells, or None when there is nothing to pool -- no
    cell (every seed of the arm unmeasurable) or no window surviving every
    cell's mask -- decided BEFORE `pool_arm`, so an empty series never
    reaches `np.mean`."""
    if not cells:
        return None
    pooling.require_compatible(cells)
    if not np.logical_and.reduce([c.changed for c in cells]).any():
        return None
    return pooling.pool_arm(cells)


_NO_CONTRAST = Contrast(estimate=float("nan"), se=float("nan"), z=float("nan"), n_windows=0)
_NO_RATIO = Ratio(estimate=float("nan"), low=float("nan"), high=float("nan"))


def _contrast(treatment, control) -> Contrast:
    """`paired_contrast` reduced to what the rules read. NaN with zero windows
    when either side has no cell (every cell of an arm unmeasurable) or no
    window survives every cell's mask -- decided BEFORE `paired_contrast`, so
    an empty series never reaches `np.mean`. The compatibility refusals are
    still `pooling`'s, raised first."""
    if not treatment or not control:
        return _NO_CONTRAST
    pooling.require_compatible(treatment)
    pooling.require_compatible(control)
    if not np.logical_and.reduce([c.changed for c in treatment + control]).any():
        return _NO_CONTRAST
    result = pooling.paired_contrast(treatment, control)
    return Contrast(estimate=result.mean, se=result.se, z=result.z, n_windows=result.windows)


def _ratio(cells) -> Ratio:
    """`pool_ratio` reduced to (ratio, low, high); NaN when no cell has a
    moved window to pool."""
    if not cells or not any(c.changed.any() for c in cells):
        return _NO_RATIO
    result = pooling.pool_ratio(cells, bootstrap=BOOTSTRAP, seed=BOOTSTRAP_SEED)
    return Ratio(estimate=result.ratio, low=result.ci_low, high=result.ci_high)


def _column(block: dict, key: str, hi: int) -> np.ndarray:
    """Column `hi` (0-based step) of an `[n][H]` series."""
    return np.asarray(block[key], dtype=float)[:, hi]


def _moved(record: dict, hi: int) -> np.ndarray:
    """The moved mask at a step. `ratio_raw` is NaN exactly where the window
    did not move (the contract's `Decomposition`) and finite everywhere else
    -- |d| >= MIN_MOVE there -- so its finiteness IS the mask."""
    return np.isfinite(_column(record, "ratio_raw", hi))


def _measurable(record: dict) -> bool:
    """Spec 3.1: a cell enters the probe-based pooling iff its persistence-
    to-floor band at the final step is positive -- decided by Task 4 with the
    ladder's `probe_is_measurable` rule and carried on the record."""
    return bool(record["probe"]["measurable"])


def pooled_inputs(records: dict, *, h: int = 45) -> tuple[ReadingOneInputs, int]:
    """Every input Reading 1 is decided on, pooled over the cells handed in,
    at step `h`, and the cluster count `z_fam` is read against.

    Probe-based series (margin, cosine, held-out error, R_probe, the probe
    crossing) pool the MEASURABLE cells; probe-free ones (R_free, the free
    crossing) pool every cell. The same inputs are computed again within
    each seed -- one cell per arm, no seed averaging, that seed's windows
    clustered by episode -- under `per_seed`; those carry `per_seed=None`.
    Refuses a missing (arm, seed) by name and a record with no step `h`.
    """
    return _inputs(records, h=h, per_seed=True)


def _inputs(records: dict, *, h: int, per_seed: bool) -> tuple[ReadingOneInputs, int]:
    seeds = sorted({seed for _, seed in records})
    for arm in ARMS_ORDER:
        for seed in seeds:
            if (arm, seed) not in records:
                raise KeyError(
                    f"no record for {arm} seed {seed}: Reading 1 pools every arm at every seed "
                    f"handed in, and the cells here are {sorted(records)}"
                )
    for (arm, seed), record in records.items():
        if int(record["horizon"]) < h:
            raise ValueError(f"{arm} seed {seed}: horizon {record['horizon']} has no step h={h}")
    hi = h - 1
    first = next(iter(records.values()))
    # The clusters are the validation EPISODES that contribute windows -- 24
    # on the shipped split -- read off the window index, not off the kept
    # windows of any one contrast: the family threshold is one number.
    clusters = int(np.unique(np.asarray(first["windows"]["episode"], dtype=int)).size)

    def series(arm, seed, name, values, changed):
        return cell_series(arm, seed, name, values, changed, records[(arm, seed)])

    def probe_cells(name, values_of, fold=None):
        """Per arm, one CellSeries per MEASURABLE seed: the series at `hi`
        under moved-and-finite, restricted to one fold's rows if asked."""
        out = {}
        for arm in ARMS_ORDER:
            cells = []
            for seed in seeds:
                record = records[(arm, seed)]
                if not _measurable(record):
                    continue
                values = values_of(record)
                changed = _moved(record, hi) & np.isfinite(values)
                if fold is not None:
                    changed &= np.asarray(record["windows"]["episode"], dtype=int) % 2 == fold
                cells.append(series(arm, seed, name, values, changed))
            out[arm] = cells
        return out

    margin = probe_cells("margin", lambda r: _column(r, "margin", hi))
    cosine = probe_cells("cosine", lambda r: _column(r, "cosine", hi))
    held = lambda r: _column(r["scale"], "held_out", hi)  # noqa: E731
    # Fold A = even episode labels, fold B = odd (`scale_corrected_error`'s
    # own assignment); each fold's contrast is over that fold's rows alone,
    # which is where the held-out alpha is the OTHER fold's.
    corrected_a = probe_cells("held_out_a", held, fold=0)
    corrected_b = probe_cells("held_out_b", held, fold=1)

    def ratio_cells(name, numerator_key, denominator_key, probe_based):
        """Per arm, the moved windows' numerator and denominator series in
        `pool_ratio`'s slots (`embedding`, `noise`); `delta` carries the
        record's own per-window ratio for the reader, and is not pooled."""
        out = {}
        for arm in ARMS_ORDER:
            cells = []
            for seed in seeds:
                record = records[(arm, seed)]
                if probe_based and not _measurable(record):
                    continue
                numerator = _column(record["displacement"], numerator_key, hi)
                denominator = _column(record["displacement"], denominator_key, hi)
                changed = _moved(record, hi) & np.isfinite(numerator) & np.isfinite(denominator)
                cell = series(arm, seed, name, _column(record, name, hi), changed)
                cells.append(dataclasses.replace(cell, embedding=numerator, noise=denominator))
            out[arm] = cells
        return out

    ratio_probe = ratio_cells("ratio_probe", "probe_hat", "probe_real", probe_based=True)
    ratio_free = ratio_cells("ratio_free", "free_hat", "free_true", probe_based=False)

    def crossing_contrast(channel, probe_based):
        """TREATMENT - CONTROL paired per (window, seed) draw: the seeds
        both arms carry (both measurable, for the probe channel) stacked in
        one series per arm, a never-moved draw (NaN) leaving on its own."""
        common = [
            seed for seed in seeds
            if not probe_based
            or (_measurable(records[(TREATMENT, seed)]) and _measurable(records[(CONTROL, seed)]))
        ]
        if not common:
            return _NO_CONTRAST
        return _contrast(
            [_stacked_crossings(records, TREATMENT, channel, common)],
            [_stacked_crossings(records, CONTROL, channel, common)],
        )

    # A boundary alpha in ANY measurable seed of an arm flags the arm: the
    # corrected contrast pooled those seeds, and a correction that is
    # persistence itself in one of them is not a correction (spec 2.2).
    alpha_boundary = {
        arm: any(
            bool(records[(arm, seed)]["scale"]["boundary"][hi])
            for seed in seeds if _measurable(records[(arm, seed)])
        )
        for arm in ARMS_ORDER
    }
    inputs = ReadingOneInputs(
        delta_contrast={
            (a, b): _contrast(margin[a], margin[b]) for a, b in itertools.combinations(ARMS_ORDER, 2)
        },
        ratio_probe={arm: _ratio(ratio_probe[arm]) for arm in ARMS_ORDER},
        ratio_free={arm: _ratio(ratio_free[arm]) for arm in ARMS_ORDER},
        cosine_contrast=_contrast(cosine[TREATMENT], cosine[CONTROL]),
        corrected_contrast_a=_contrast(corrected_a[TREATMENT], corrected_a[CONTROL]),
        corrected_contrast_b=_contrast(corrected_b[TREATMENT], corrected_b[CONTROL]),
        alpha_boundary=alpha_boundary,
        crossing_contrast_probe=crossing_contrast("probe", probe_based=True),
        crossing_contrast_free=crossing_contrast("free", probe_based=False),
        per_seed=(
            {
                seed: _inputs(
                    {key: r for key, r in records.items() if key[1] == seed}, h=h, per_seed=False
                )[0]
                for seed in seeds
            }
            if per_seed else None
        ),
    )
    return inputs, clusters


def _r2_filtered(records: dict, min_r2: float) -> dict:
    """The records again with every cell of probe selection R^2 below
    `min_r2` marked unmeasurable -- NEW record dicts sharing every series
    with the originals, only the `probe` block rewritten, so the records the
    verdict was read from are untouched."""
    return {
        key: {
            **record,
            "probe": {
                **record["probe"],
                "measurable": bool(record["probe"]["measurable"])
                and float(record["probe"]["selection_r2"]) >= min_r2,
            },
        }
        for key, record in records.items()
    }


@dataclasses.dataclass(frozen=True)
class ArmSummary:
    """One arm's line of the per-arm block at step h (spec 5: "per-arm block,
    then contrasts, then the verdict lines"). `delta` and `cosine` are
    `pool_arm` over the arm's MEASURABLE cells -- seed-mean per window over
    the moved-and-finite windows, mean +- episode-clustered se -- or None
    when there is nothing to pool; the two ratios are the ones
    `pooled_inputs` decided on; `ratio_raw` is the median of the moved
    draws' own |d_hat| / |d| over every cell (spec 2.2: "stored beside it as
    a secondary column, not decided on"); the counts are summed over the
    arm's cells. Nothing here is read by a rule."""

    delta: pooling.PooledMean | None
    cosine: pooling.PooledMean | None
    ratio_probe: Ratio
    ratio_free: Ratio
    ratio_raw: float
    moved: int
    zero_displacement: int


def arm_summaries(records: dict, inputs: ReadingOneInputs, *, h: int) -> dict:
    """`arm -> ArmSummary` in ARMS_ORDER, at step `h`, from the same records
    and masks `pooled_inputs` pooled the contrasts from."""
    hi = h - 1
    out = {}
    for arm in ARMS_ORDER:
        cells = [(seed, r) for (a, seed), r in sorted(records.items()) if a == arm]
        measurable = [(seed, r) for seed, r in cells if _measurable(r)]
        raw = np.concatenate([_column(r, "ratio_raw", hi) for _, r in cells])
        raw = raw[np.isfinite(raw)]
        out[arm] = ArmSummary(
            delta=_pooled_mean([_moved_series(arm, seed, "margin", r, hi) for seed, r in measurable]),
            cosine=_pooled_mean([_moved_series(arm, seed, "cosine", r, hi) for seed, r in measurable]),
            ratio_probe=inputs.ratio_probe[arm],
            ratio_free=inputs.ratio_free[arm],
            ratio_raw=float(np.median(raw)) if raw.size else float("nan"),
            moved=sum(
                int(r["windows"]["total"]) - int(r["counts"]["not_moved"][hi]) for _, r in cells
            ),
            zero_displacement=sum(int(r["counts"]["zero_displacement"][hi]) for _, r in cells),
        )
    return out


def _informational_crossing(records: dict, arm: str, channel: str) -> Contrast:
    """`arm - CONTROL` on h_x per (window, seed) draw, the probe control's
    own estimator over the seeds both arms carry (both measurable, for the
    probe channel). Printed for pixel_ae -- spec 3.2: "`pixel_ae`'s pair is
    printed for information" -- and decided on by nothing: `ReadingOneInputs`
    has no slot for it."""
    seeds = sorted({seed for _, seed in records})
    if channel == "probe":
        seeds = [
            seed for seed in seeds
            if _measurable(records[(arm, seed)]) and _measurable(records[(CONTROL, seed)])
        ]
    if not seeds:
        return _NO_CONTRAST
    return _contrast(
        [_stacked_crossings(records, arm, channel, seeds)],
        [_stacked_crossings(records, CONTROL, channel, seeds)],
    )


def survival_by_arm(records: dict) -> dict:
    """`(arm, channel) -> S(h)` for Reading 2: `trust.survival` over every
    (window, seed) draw of the arm, seeds stacked, never-moved draws (NaN)
    excluded by `survival` itself. The horizon is the records'."""
    arms = sorted({arm for arm, _ in records}, key=ARMS_ORDER.index)
    seeds = sorted({seed for _, seed in records})
    curves = {}
    for arm in arms:
        cells = [records[(arm, seed)] for seed in seeds if (arm, seed) in records]
        horizon = int(cells[0]["horizon"])
        for channel in ("probe", "free"):
            crossings = np.concatenate(
                [np.asarray(cell["crossing"][channel], dtype=float) for cell in cells]
            )
            curves[(arm, channel)] = survival(crossings, horizon)
    return curves


def write_readings(out_dir: Path, text: str) -> Path:
    """`runs/<out>/trust.txt`: the readings exactly as printed."""
    path = Path(out_dir) / "trust.txt"
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# The printed readings, in the ladder's style: the self-check table, the
# pooling notes, the per-arm block, Reading 1 (summary line, then Task 5's
# block under its own header) with pixel_ae's h_x pair for information, the
# sensitivity block, Reading 2 (Task 5's block under its own header).
# `_readings_text` is what `main` prints and what trust.txt holds.
# ---------------------------------------------------------------------------


def _num(value: float, spec: str = ".3f") -> str:
    """NaN prints as `n/a`, never as something that looks measured -- and NaN
    is what every clustered ruler is below two clusters. An infinite z (a
    ruler of exactly 0 under a nonzero mean, `pooling._z`'s policy) prints
    as itself, as the ladder prints it."""
    return "n/a" if np.isnan(value) else format(value, spec)


def _self_check_table(records: dict) -> str:
    """One row per cell, in ARMS_ORDER then seed: the evidence that the
    readings below pool what the ladder measured. Both self-check deltas
    (exactly 0.0 on every cell that reached this point -- a non-zero delta
    is exit 30 before any reading), the window count and whether the
    episode labels matched, the probe's selection R^2 and measurability,
    the never-moved count, and `ok`. Task 7 fills its self-check table
    from this."""
    rows = [
        f"{'cell':<16}{'ref_max|delta|':>15}{'pers_max|delta|':>16}{'windows':>9}"
        f"{'episodes_match':>16}{'probe_r2':>10}{'measurable':>12}{'never_moved':>13}{'ok':>7}"
    ]
    for (arm, seed), r in sorted(records.items(), key=lambda kv: (ARMS_ORDER.index(kv[0][0]), kv[0][1])):
        sc = r["self_check"]
        rows.append(
            f"{f'{arm}/s{seed}':<16}{float(sc['reference_position_max_delta']):>15.1e}"
            f"{float(sc['persistence_position_max_delta']):>16.1e}{int(r['windows']['total']):>9}"
            f"{str(bool(sc['windows_episode_match'])):>16}{float(r['probe']['selection_r2']):>10.3f}"
            f"{str(_measurable(r)):>12}{int(r['counts']['never_moved']):>13}{str(bool(sc['ok'])):>7}"
        )
    return "\n".join(rows)


def _arm_table(summaries: dict, h: int) -> str:
    """The per-arm block, one row per arm in ARMS_ORDER."""
    rows = [
        f"{'arm':<12}{f'delta({h})':>11}{'se':>9}{'n':>5}{f'R_probe({h})':>13}{'[95% CI]':>18}"
        f"{f'R_free({h})':>12}{'[95% CI]':>18}{f'R_raw({h})':>11}{f'cos({h})':>9}{'se':>8}{'n':>5}"
        f"{'moved':>7}{'zero_dhat':>11}"
    ]
    for arm, s in summaries.items():
        delta = (
            f"{_num(s.delta.mean, '+.3f'):>11}{_num(s.delta.se):>9}{s.delta.windows:>5}"
            if s.delta is not None else f"{'n/a':>11}{'n/a':>9}{0:>5}"
        )
        cosine = (
            f"{_num(s.cosine.mean, '+.3f'):>9}{_num(s.cosine.se):>8}{s.cosine.windows:>5}"
            if s.cosine is not None else f"{'n/a':>9}{'n/a':>8}{0:>5}"
        )
        rows.append(
            f"{arm:<12}{delta}"
            f"{_num(s.ratio_probe.estimate):>13}{f'[{_num(s.ratio_probe.low)}, {_num(s.ratio_probe.high)}]':>18}"
            f"{_num(s.ratio_free.estimate):>12}{f'[{_num(s.ratio_free.low)}, {_num(s.ratio_free.high)}]':>18}"
            f"{_num(s.ratio_raw):>11}{cosine}{s.moved:>7}{s.zero_displacement:>11}"
        )
    return "\n".join(rows)


def _pooling_notes(records: dict, clusters: int, z_fam: float, h: int) -> list[str]:
    """What the pooled numbers stand on, before any of them is printed: the
    cluster count and the family threshold, and -- when the clustered ruler
    cannot exist -- WHY every contrast below reads n/a."""
    lines = [
        f"clusters: {clusters} validation episode(s) contribute windows; "
        f"z_fam = cluster_threshold({FAMILY}, {clusters}) = {_num(z_fam, '.2f')} "
        f"(Bonferroni over the {FAMILY} clustered contrasts of Reading 1, read against t({clusters - 1}))"
    ]
    if clusters < 2:
        lines.append(
            "the episode-clustered standard error needs at least two clusters, so on this "
            "split every pooled se and z is NaN, every contrast reads n/a, and Reading 1 has "
            "no best-delta arm: NOT_TESTABLE by construction, not by evidence"
        )
    excluded = [f"{arm}/s{seed}" for (arm, seed), r in sorted(records.items()) if not _measurable(r)]
    lines.append(
        f"probe-based pooling: {len(records) - len(excluded)} of {len(records)} cells measurable "
        f"(persistence-to-floor band at h={h} > 0); excluded: {', '.join(excluded) or 'none'}"
    )
    not_moved = ", ".join(
        f"{arm}/s{seed}={int(r['counts']['not_moved'][h - 1])}" for (arm, seed), r in sorted(records.items())
    )
    lines.append(f"windows not moved at h={h} (excluded from every pooled series): {not_moved}")
    return lines


def _sensitivity_table(inputs: ReadingOneInputs, h: int) -> str:
    """Every probe-based statistic of Reading 1, one row each, as recomputed
    with the low-R^2 cells excluded. Estimates and rulers only: this block
    changes no verdict and prints none."""
    rows = [f"{'statistic (h_x per draw)':<40}{'estimate':>12}{'se':>10}{'z':>10}{'windows':>10}"]
    contrast = lambda label, c: rows.append(  # noqa: E731
        f"{label:<40}{_num(c.estimate, '+.3f'):>12}{_num(c.se):>10}{_num(c.z, '+.2f'):>10}"
        f"{c.n_windows:>10}"
    )
    for (a, b), c in inputs.delta_contrast.items():
        contrast(f"delta({h}) {a} - {b}", c)
    contrast(f"cos({h}) {TREATMENT} - {CONTROL}", inputs.cosine_contrast)
    contrast(f"c({h}) fold A {TREATMENT} - {CONTROL}", inputs.corrected_contrast_a)
    contrast(f"c({h}) fold B {TREATMENT} - {CONTROL}", inputs.corrected_contrast_b)
    contrast(f"h_x probe {TREATMENT} - {CONTROL}", inputs.crossing_contrast_probe)
    for arm, ratio in inputs.ratio_probe.items():
        rows.append(
            f"{f'R_probe({h}) {arm}':<40}{_num(ratio.estimate):>12}"
            f"{f'[{_num(ratio.low)}, {_num(ratio.high)}]':>30}"
        )
    return "\n".join(rows)


def _readings_text(records: dict, *, h: int) -> str:
    """The whole readings block, in this order: the run header; the
    self-check table (always); then, when every arm is present, the pooling
    notes, the per-arm block, the `reading 1:` summary line followed by
    Task 5's Reading 1 block under its own header, pixel_ae's h_x pair for
    information, the sensitivity block, and Task 5's Reading 2 block under
    its own header. Both readings need all three arms -- Reading 1 pools
    them and `reading_two` refuses a missing (arm, channel) by name -- so a
    run over fewer prints `not computed` under BOTH headers and still exits
    0 with the self-check table on disk."""
    arms = sorted({arm for arm, _ in records}, key=ARMS_ORDER.index)
    seeds = sorted({seed for _, seed in records})
    lines = [
        f"--- trust readings at h = {h}: {len(records)} cells, arms {arms}, seeds {seeds}; "
        f"means and contrasts seed-averaged per window and clustered by episode, crossings "
        f"per (window, seed) draw, ratios as ratios of medians over the moved windows "
        f"(bootstrap={BOOTSTRAP} bootstrap_seed={BOOTSTRAP_SEED}) ---",
        "\n--- self-check per cell (spec 2.3): the trust pass's window-mean curves against "
        "the diagnostic's, max |delta| exactly 0.0 (a non-zero delta is exit 30 before any "
        "reading), and the windows ---",
        _self_check_table(records),
    ]
    complete = all(arm in arms for arm in ARMS_ORDER)
    if complete:
        inputs, clusters = pooled_inputs(records, h=h)
        z_fam = pooling.cluster_threshold(FAMILY, clusters)
        lines.append("")
        lines += _pooling_notes(records, clusters, z_fam, h)
        lines.append(
            f"\n--- per arm at h = {h}: delta and cos seed-averaged per window over the "
            f"measurable cells' moved windows, mean +- episode-clustered se; ratios of medians "
            f"with the 95% episode-bootstrap interval; R_raw the median of the moved draws' "
            f"|d_hat| / |d|, for information ---"
        )
        lines.append(_arm_table(arm_summaries(records, inputs, h=h), h))
        reading = reading_one(inputs, z_fam, h=h)
        lines.append("")
        lines.append(f"reading 1: {reading.status.name} -- {reading.reason}")
        lines.append(format_reading_one(reading, inputs, z_fam, h=h))
        for channel in ("probe", "free"):
            c = _informational_crossing(records, "pixel_ae", channel)
            lines.append(
                f"for information: h_x {channel} pixel_ae - {CONTROL} per draw: estimate "
                f"{_num(c.estimate, '+.3f')}, se {_num(c.se)}, z {_num(c.z, '+.2f')} "
                f"(windows {c.n_windows}); decides nothing"
            )
        filtered = _r2_filtered(records, R2_SENSITIVITY)
        dropped = [
            f"{arm}/s{seed}" for (arm, seed), r in sorted(records.items())
            if _measurable(r) and not _measurable(filtered[(arm, seed)])
        ]
        sensitivity, _ = pooled_inputs(filtered, h=h)
        lines.append(
            f"\n--- sensitivity: the probe-based statistics with probe selection R^2 < "
            f"{R2_SENSITIVITY} excluded ({len(dropped)} of {len(records)} cells: "
            f"{', '.join(dropped) or 'none'}); this changes no verdict ---"
        )
        lines.append(_sensitivity_table(sensitivity, h))
        lines.append("")
        lines.append(format_reading_two(reading_two(survival_by_arm(records))))
    else:
        lines.append(
            f"\n--- Reading 1: does the h={h} gate reward slow drift? ---\n"
            f"not computed: Reading 1 pools all of {list(ARMS_ORDER)} and this run has {arms}"
        )
        lines.append(
            "\n--- Reading 2: the horizon M4 designs around ---\n"
            f"not computed: Reading 2 needs every arm's survival curve and this run has {arms}"
        )
    return "\n".join(lines) + "\n"
```

The reading is at the records' final step: `main` passes the run's horizon as `h`, which is 45 on the shipped records and 3 on the fixture; `h` goes through to `reading_one` / `format_reading_one`, so the fixture's Reading 1 header and every condition detail read `h=3` (the fixture tests assert that string). The two reading headers in a complete run are Task 5's -- `--- Reading 1: does the h={h} gate reward slow drift?  (z_fam ..., family 8, treatment frozen_ssl, control random_vit)` and `--- Reading 2: the horizon M4 designs around  (S(h) = ...; no verdict)` -- and this script prints none of its own for them; only the `not computed` branch carries its own two headers.

- [ ] **Step 4: Run the glue tests**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_trust_horizon_script.py -q -p no:cacheprovider -k "cell_series or pooled_inputs or unmeasurable or r2_filter or per_arm_block or pixel_aes_crossing or self_check_table or survival_by_arm or write_readings or fixture_run or single_arm"`
Expected: **15 passed, 3 failed** -- the two `fixture_run` tests and the single-arm test still fail with `FileNotFoundError: ... trust.txt`, because `main` does not call any of this yet.

- [ ] **Step 5: Extend `main`**

In `main`, replace the final `return EXIT_OK` (the one reached after every cell's record has been written and every check has held) with:

```python
    # Pooling and the two readings -- only after every cell's checks held
    # and every record is on disk, so trust.txt never describes cells a
    # later line disowns. `records` is keyed (arm, seed); `horizon` is the
    # run's resolved horizon, so the reading is at the final step (45 on
    # the shipped records).
    text = _readings_text(records, h=horizon)
    print(text, end="")
    write_readings(args.out, text)
    return EXIT_OK
```

`records` (keyed `(arm, seed)`, filled from `_run_cell`'s `(EXIT_OK, record)` returns), `args` and `horizon` are Task 4's locals, named in its Interfaces. Nothing else in `main` changes: the exit statuses 11 / 12 / 14 / 30 are all decided before this block is reached, and a run over fewer than three arms (Task 4's own fixture) reaches it too and prints `not computed` under both headers.

- [ ] **Step 6: Run the file**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_trust_horizon_script.py -q -p no:cacheprovider`
Expected: **PASS** -- 18 more than the file held after Task 4 (record the measured number). The two `trust_run` fixture tests take ~10 s each (three `run_job` + three `diagnose_dynamics.main` calls, then `trust_horizon.main`); the single-arm test is one `cell` build, ~3 s.

- [ ] **Step 7: Add `trust_horizon` to the exit-status distinctness test**

`trust_horizon.py` REUSES `11`, `12` and `14` under `diagnose_dynamics.py`'s names and meanings (the contract; spec 5), so adding it to the existing `for other in (...)` loop would fail on exactly those three. The test has to say which codes are shared on purpose and that everything else is the script's own. Replace the body of `test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own` in `tests/eval/test_diagnose_dynamics_script.py` (line 463) with:

```python
def test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own():
    """The study scripts run one after another in the same shell and a
    wrapper reads the status, so a collision reports one script's failure under
    another's meaning. 1 is an uncaught traceback and 2 is argparse's usage
    error, so neither may be reused either.

    `trust_horizon.py` is the one script that shares codes with this one ON
    PURPOSE: 11, 12 and 14 mean the same thing there (a checkpoint or record
    missing, the split by name, `evaluate_rollout` no longer reproducing the
    record) and are exported under the same names. The literal below is what
    is shared; anything else the two have in common is a collision, and
    everything else `trust_horizon` exits with must be nobody else's."""
    import importlib.util as util

    def statuses(name):
        spec = util.spec_from_file_location(
            f"{name}_status", Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
        )
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {
            key: value for key, value in vars(module).items() if key.startswith("EXIT_")
        }

    mine = statuses("diagnose_dynamics")
    assert len(set(mine.values())) == len(mine), mine
    assert 1 not in mine.values() and 2 not in mine.values()
    for other in ("run_study", "report_study", "pool_dynamics"):
        theirs = statuses(other)
        clash = (set(mine.values()) & set(theirs.values())) - {0}
        assert not clash, f"diagnose_dynamics collides with {other} on {clash}"

    trust = statuses("trust_horizon")
    reused = {"EXIT_NO_CHECKPOINTS": 11, "EXIT_SPLIT_MISMATCH": 12, "EXIT_RECORD_MISMATCH": 14}
    shared = {name: value for name, value in trust.items() if value in set(mine.values()) - {0}}
    assert shared == reused, (
        f"trust_horizon shares {shared} with diagnose_dynamics; only {reused} is shared on purpose"
    )
    assert {name: mine[name] for name in reused} == reused
    assert len(set(trust.values())) == len(trust), trust
    assert 1 not in trust.values() and 2 not in trust.values()
    assert trust["EXIT_SELF_CHECK_FAILED"] == 30
    own = set(trust.values()) - set(reused.values()) - {0}
    for other in ("run_study", "report_study", "pool_dynamics"):
        clash = own & set(statuses(other).values())
        assert not clash, f"trust_horizon collides with {other} on {clash}"
```

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/eval/test_diagnose_dynamics_script.py -q -p no:cacheprovider -k every_exit_status`
Expected: **1 passed**. (Also run `tests/eval/test_pool_dynamics_script.py -k exit_statuses`: its literal set of the other scripts' codes is untouched by a fourth script that reuses 11/12/14 and adds 30, so it still passes -- 1 passed.)

- [ ] **Step 8: Mutation-test**

Set up a self-checked harness. All three checks below have silently lied in this project (M3b plan, Global Constraints):

```bash
# 1. The package is installed EDITABLE, so a naive export is shadowed by
#    src/. Run with PYTHONPATH at the mutated copy and confirm it wins.
S=/tmp/m3d-task6 && rm -rf $S && mkdir -p $S && cp -R src tests scripts pyproject.toml $S/
cd $S && PYTHONPATH=$S/src PYTHONDONTWRITEBYTECODE=1 \
  /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -c "import mbfps; print(mbfps.__file__)"
# must print /tmp/m3d-task6/src/mbfps/__init__.py, NOT .../csgo-bot/src/...
# The tests load scripts/trust_horizon.py by a path RELATIVE TO THE TEST FILE,
# so running pytest from $S loads $S/scripts/trust_horizon.py: confirm with
PYTHONPATH=$S/src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -c \
  "import importlib.util as u, pathlib; s=u.spec_from_file_location('t', pathlib.Path('tests/eval/test_trust_horizon_script.py').resolve().parents[2]/'scripts'/'trust_horizon.py'); print(s.origin)"
# must print /tmp/m3d-task6/scripts/trust_horizon.py

# 2. An unproven harness proves nothing: run the known-fatal mutation FIRST
#    (M0 below, the moved mask replaced by all-True) and confirm SIX tests
#    fail before trusting any other row.

# 3. A same-byte-length mutation can leave a stale .pyc that survives a
#    correct restore: before EVERY run,
find $S -name __pycache__ -exec rm -rf {} + ; export PYTHONDONTWRITEBYTECODE=1
```

Apply each mutation to `$S/scripts/trust_horizon.py`, run from `$S`:

```bash
PYTHONPATH=$S/src /Users/raphaelchen/Desktop/csgo-bot/.venv/bin/python -m pytest \
  tests/eval/test_trust_horizon_script.py tests/eval/test_diagnose_dynamics_script.py -q -p no:cacheprovider \
  -k "cell_series or pooled_inputs or unmeasurable or r2_filter or per_arm_block or pixel_aes_crossing or self_check_table or survival_by_arm or write_readings or fixture_run or single_arm or every_exit_status"
```

restore from the repo copy, and confirm the restored file passes the same selection at the end (18 + 1 = 19 of this task's tests, plus whatever else in the two files the `-k` happens to match -- record the number on the first clean run and require it again at the end).

| mutation | must be caught by |
|---|---|
| M0 (known-fatal, run first): `_moved` returns `np.ones(len(record["ratio_raw"]), dtype=bool)` -- the unmoved window 0 enters every pooled series | `test_pooled_inputs_pairs_the_three_arms_as_a_minus_b_and_reads_a_finite_family_threshold` (window 0's 50 enters the margin mean), `test_pooled_inputs_cosine_and_corrected_contrasts_exclude_nan_windows_and_split_the_folds`, `test_pooled_inputs_ratios_are_ratios_of_medians_with_each_cell_in_units_of_its_own_denominator`, `test_pooled_inputs_carries_alpha_boundaries_at_h_and_the_same_inputs_within_each_seed`, `test_an_unmeasurable_cell_leaves_the_probe_based_pooling_and_only_that`, `test_the_per_arm_block_pools_each_arms_own_margin_and_cosine_and_reads_the_counts` (frozen_ssl's delta reads 51 in) -- six |
| M1: `cell_series` sets `channel="angle"` | `test_cell_series_carries_the_moved_mask_as_changed_and_the_records_identity` (the ONLY test that reads the channel) |
| M2: delete the shape guard in `cell_series` (`if False:`) | `test_cell_series_refuses_a_series_that_is_not_one_value_per_window` |
| M3: the delta pairs computed as `b - a` -- `_contrast(margin[b], margin[a])` | `test_pooled_inputs_pairs_the_three_arms_as_a_minus_b_and_reads_a_finite_family_threshold` (every sign flips), `test_pooled_inputs_carries_alpha_boundaries_at_h_and_the_same_inputs_within_each_seed`, `test_an_unmeasurable_cell_leaves_the_probe_based_pooling_and_only_that` |
| M4: `clusters = int(len(first["windows"]["episode"]))` (windows counted as clusters) | `test_pooled_inputs_pairs_the_three_arms_as_a_minus_b_and_reads_a_finite_family_threshold` (`clusters == 4`, not 6 -- the ONLY test that reads it) |
| M5: `changed = _moved(record, hi)` in `probe_cells`, without `& np.isfinite(values)` | `test_pooled_inputs_cosine_and_corrected_contrasts_exclude_nan_windows_and_split_the_folds` (frozen_ssl seed 1's NaN cosine at window 5 reaches the mean -- the ONLY test with a NaN on a moved window) |
| M6: folds swapped -- `corrected_a` built with `fold=1`, `corrected_b` with `fold=0` | `test_pooled_inputs_cosine_and_corrected_contrasts_exclude_nan_windows_and_split_the_folds` (fold A reads -14/3 instead of -4) |
| M7: the crossing contrast seed-averaged -- `stacked` returns the per-seed cells and `_contrast` receives them as lists | `test_pooled_inputs_crossing_contrasts_are_per_draw_with_the_seeds_stacked` (five windows, not eleven draws), `test_an_unmeasurable_cell_leaves_the_probe_based_pooling_and_only_that` |
| M8: numerator and denominator swapped -- `dataclasses.replace(cell, embedding=denominator, noise=numerator)` | `test_pooled_inputs_ratios_are_ratios_of_medians_with_each_cell_in_units_of_its_own_denominator` and every other `pooled_inputs` test: random_vit's all-zero numerator lands in the ruler slot and `pool_ratio` raises `DegenerateNoise` |
| M9: `alpha_boundary` reads `boundary[0]` instead of `boundary[hi]` | `test_pooled_inputs_carries_alpha_boundaries_at_h_and_the_same_inputs_within_each_seed` (frozen_ssl is on the boundary at step 3 only) |
| M10: `per_seed` computed from ALL records (drop `if key[1] == seed`) | `test_pooled_inputs_carries_alpha_boundaries_at_h_and_the_same_inputs_within_each_seed` (seed 0's contrast reads the pooled 3, not 2) |
| M11: `_measurable` returns `True` | `test_an_unmeasurable_cell_leaves_the_probe_based_pooling_and_only_that` (the ONLY test with an unmeasurable cell) |
| M12: `_r2_filtered` rewrites `record["probe"]["measurable"]` in place and returns `records` | `test_the_r2_filter_marks_low_r2_cells_unmeasurable_without_touching_the_originals` |
| M13: `survival_by_arm` stacks only the first seed (`cells = cells[:1]`) | `test_survival_by_arm_stacks_the_seeds_per_channel` (frozen_ssl's probe curve is over five draws, not eleven) |
| M14: `write_readings` writes `readings.txt` | `test_write_readings_writes_trust_txt_under_out_and_returns_its_path`, `test_the_fixture_run_writes_trust_txt_with_both_readings` |
| M15: `_pooling_notes` drops the `clusters < 2` explanation (`if False:`) | `test_the_fixture_run_says_why_every_contrast_is_n_a` (`"at least two"` appears in no other line) |
| M16: `EXIT_SELF_CHECK_FAILED = 14` | `test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own` in `test_diagnose_dynamics_script.py` (`shared` gains a fourth name) |
| M17: `main` writes trust.txt but does not print it (delete the `print`) | `test_the_fixture_run_writes_trust_txt_with_both_readings`, `test_a_single_arm_run_still_exits_ok_...` (`text in out`) |
| M18: the Reading 2 guard dropped -- `format_reading_two(reading_two(survival_by_arm(records)))` appended unconditionally (the `else` branch's Reading 2 line replaced by it) | `test_a_single_arm_run_still_exits_ok_and_says_both_readings_are_not_computed` (`reading_two` raises `KeyError` naming `('pixel_ae', 'probe')` out of `main`; the three-arm fixture is NOT a catch) |
| M19: `arm_summaries` pools every arm's cells into each row (drop `if a == arm`) | `test_the_per_arm_block_pools_each_arms_own_margin_...` (pixel_ae's delta is no longer 3 with se 0), `test_an_unmeasurable_cell_...` (`seeds` is six-long) |
| M20: `arm_summaries` pools the unmeasurable cells too (`measurable = cells`) | `test_an_unmeasurable_cell_leaves_the_probe_based_pooling_and_only_that` (random_vit's delta reads 2, not 1; `nothing[arm].delta` is not None) |
| M21: `arm_summaries.moved` counts `not_moved` (drop `int(r["windows"]["total"]) -`) | `test_the_per_arm_block_pools_each_arms_own_margin_...` (`moved` reads 2, not 10) |
| M22: `_informational_crossing` contrasts `TREATMENT` whatever `arm` is asked | `test_pixel_aes_crossing_pair_is_computed_for_information_with_the_seeds_stacked` (pixel_ae's probe z is finite, not +inf) |
| M23: `_self_check_table` sorts the rows by `sorted(records.items())` (alphabetical: frozen_ssl first) | `test_the_self_check_table_has_one_row_per_cell_in_arm_order_...` (the ONLY test that reads the row order) |
| M24: `reading_one` / `format_reading_one` called without `h=h` | `test_the_fixture_run_writes_trust_txt_with_both_readings` (the Reading 1 header reads `h=45` on a horizon-3 run: `"does the h=3 gate"` is absent) |

M0-M14 and M16 were run against the pre-reconciliation test block and glue in an export of the tree with Tasks 1, 4 and 5 present only as the contract's names (a stub `survival`, the `trust_readings` dataclasses, the `EXIT_*` constants); each row's catcher is what actually fired, and M1, M4, M5, M11 are each caught by exactly one test. M15, M17, M18 and M24 need a `main` run: their catchers are the tests that assert the string or the exception each mutation produces, and nothing else prints it. M19-M23 were added at reconciliation with the per-arm block, pixel_ae's pair and the self-check table, and have not been run: run them, and record what fired. Any mutation that survives is a missing test. Add it before committing.

- [ ] **Step 9: Full suite**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: **PASS, 0 warnings** -- 18 more than the previous task left (record the measured number): 15 pure glue tests plus the 3 `main`-driven tests (two on the three-arm `trust_run` fixture, one on Task 4's single-arm `cell`) in `test_trust_horizon_script.py`; the distinctness test in `test_diagnose_dynamics_script.py` is modified, not added. `pool_dynamics`'s own distinctness test (`test_pool_exit_statuses_are_disjoint_from_the_other_three_scripts_and_none_is_argparses_own`) still passes: its literal set is the other three scripts', and `trust_horizon` is not among them. If the fixture-run tests fail with `KeyError: 'displacement'`, Task 4's `trust_record` was landed without the `displacement` block its Interfaces name; add it THERE, not to the glue.

- [ ] **Step 10: Commit**

```bash
git add scripts/trust_horizon.py tests/eval/test_trust_horizon_script.py tests/eval/test_diagnose_dynamics_script.py
git commit -m "feat: trust_horizon pools the per-cell records, prints both readings and writes trust.txt

The per-cell trust records become pooling.CellSeries -- margin, cosine
and held-out error on the per-window seed-mean series over the moved
windows, the crossing contrasts on per-(window, seed) draws with the
seeds stacked, the ratios as ratios of medians through pool_ratio --
and go to reading_one / reading_two under z_fam =
cluster_threshold(8, clusters), pooled and again within each seed.
trust.txt opens with the per-cell self-check table and the per-arm
block (pool_arm per arm, the ratios, the counts) before the contrasts,
in the ladder's order; pixel_ae's h-cross pair is printed for
information. A run over fewer than three arms prints `not computed`
under both readings and still exits 0.
Unmeasurable cells leave the probe-based pooling only; a second
sensitivity block drops cells of probe R^2 < 0.1 and changes no
verdict. On a split with one validation episode every clustered ruler
is NaN: the script still exits 0, prints every contrast as n/a with
the reason, and calls Reading 1 NOT_TESTABLE. trust_horizon joins the
exit-status distinctness test, which now says which three codes it
shares with diagnose_dynamics on purpose.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Run on runs/m3_study_v2, record the two readings

Everything before this task changed code. This task spends ~10-15 laptop-minutes producing the nine `trust_<arm>_seed<n>.json` records and `trust.txt` the spec's Section 5 asks for, reads them in the order the spec fixes -- the self-check table, then Reading 2 (the survival curves and `H*_q`), then Reading 1 (the three conditions, the probe control, the status) -- and writes one section of this plan from them. No module under `src/` or `scripts/` changes here. What this task adds to the repository is the `## Task 7 results` section, filled from artefacts that live under `runs/` and are gitignored, so the numbers in the plan are the only copy that survives a clone. **A `NOT_TESTABLE`, `NOT_SUPPORTED`, `UNRESOLVED_PROBE` or `UNRESOLVED_ALPHA` status is a result to record, not a failure to hide**: the spec pre-registers all five statuses and says (Section 3.2) that "(ii) failing alone is the informative failure". The only outcomes that are not results are the non-zero exit statuses, and each of those names what to fix.

The nine checkpoints were produced under `ca3e140` (the M3c run, one device `mps`); this pass reads them under the M3d HEAD. Two shas, and both are recorded: the study records carry the first, the trust records carry the second, and the acceptance check asserts each against its literal.

**Files:**
- Create: nothing under version control. `runs/m3_study_v2/trust_<arm>_seed<n>.json` x9, `trust.txt`, `trust.log`, `trust.exit`, `trust.started`, `trust.head`, `trust_provenance.txt`, and `runs/m3d_check_trust.py` (the acceptance check below) are all under `runs/`, which `.gitignore` excludes; the doctored copies of Step 8 go under `scratchpad/m3d_doctored/` -- `scratchpad/` is the name `.gitignore` excludes (verified; `scratch/` is not in it) -- and are deleted at the end of that step.
- Modify: `docs/superpowers/plans/2026-09-13-mb-fps-m3d-trust-horizon.md` (this plan -- the file this task section lives in): append `## Task 7 results`.
- Test: the acceptance check `runs/m3d_check_trust.py`, written in Step 1, shown to fail in Steps 2 and 8, and passed in Step 8. It is a check on the run's artefacts, not on code, so it is not a pytest file; the pytest delta of this task is 0.

> Read before starting: `scripts/trust_horizon.py` as Tasks 4 and 6 landed it -- `_parser` (the defaults `--out runs/m3_study_v2 --device mps --data data/my_way_home`), the `EXIT_*` block (`0 / 11 / 12 / 14 / 30`), `main`, and `_readings_text` (Task 6) with `format_reading_one` / `format_reading_two` (Task 5), because Step 6 reads `trust.txt` by the literal headers those print; `scripts/diagnose_dynamics.py` lines 100-140 (why 14 is an environment status, not a code defect) and `main` at line 1667 (the tables-then-verdict shape `trust_horizon.py` copies); `src/mbfps/eval/study.py:246-315` (`write_record` / `load_record`: every non-finite float is written as `null` and restored from the `nonfinite` map, which is why the check reads records through `load_record` and never `json.loads`).

**Interfaces:**
- Consumes: `scripts/trust_horizon.py` CLI (Tasks 4 and 6) -- `--out`, `--device`, `--data`, `--context`, `--horizon`, `--arms` (`choices=ARMS`), `--seeds`; `main(argv: list[str] | None = None) -> int` with `EXIT_OK = 0`, `EXIT_NO_CHECKPOINTS = 11`, `EXIT_SPLIT_MISMATCH = 12`, `EXIT_RECORD_MISMATCH = 14`, `EXIT_SELF_CHECK_FAILED = 30`; `write_trust_record(out_dir: Path, record: dict) -> Path` writing `runs/<out>/trust_<arm>_seed<n>.json` with the top-level keys `arm, seed, context, horizon, split_seed, device, torch_version, git_sha, episodes {val: [...]}, windows {total, episode}, probe {selection_r2, measurable}, self_check {reference_position_max_delta, persistence_position_max_delta, windows_total_match, windows_episode_match, ok}, crossing {probe: [n], free: [n]}, margin [n][H], ratio_probe [n][H], ratio_raw [n][H], ratio_free [n][H], cosine [n][H], displacement {probe_hat [n][H], probe_real [n][H], free_hat [n][H], free_true [n][H]}, scale {alpha_a [H], alpha_b [H], score_a [H], score_b [H], held_out [n][H], boundary [H], folds_available}, counts {not_moved [H], zero_displacement [H], never_moved}, nonfinite {}`; `write_readings(out_dir: Path, text: str) -> Path` writing `runs/<out>/trust.txt`; `mbfps.eval.trust_readings.Status` with the members `SUPPORTED, NOT_SUPPORTED, NOT_TESTABLE, UNRESOLVED_PROBE, UNRESOLVED_ALPHA`, `FAMILY = 8`, `Q_PREREGISTERED = 0.75`, `Q_REPORTED = (0.5, 0.75, 0.9)`, `TREATMENT = "frozen_ssl"`, `CONTROL = "random_vit"`; `mbfps.eval.pooling.cluster_threshold(family: int, clusters: int) -> float` (= 3.009 at `(8, 24)`); `mbfps.eval.study.load_record(path: Path) -> dict`; `mbfps.utils.config.ARMS`; `mbfps.eval.aggregate.SEEDS`; the nine `runs/m3_study_v2/diagnostic_<arm>_seed<n>.json` the M3c ladder wrote (`windows {total, episode}`, `probe {embedding_selection_r2}`, `curves {reference_position, persistence_position, floor_position, ...}`, `episodes {val}`, `device`, `torch_version`, `context`, `horizon`, `split_seed`) and the nine `result_<arm>_seed<n>.json` (`git_sha`).
- Produces: `runs/m3_study_v2/trust_<arm>_seed<n>.json` x9, `trust.txt`, `trust.log`, `trust.exit`, `trust_provenance.txt` (none committed); the `## Task 7 results` section of this plan (committed). Nothing later in this plan consumes them; M4's design reads `trust.txt`'s `S(h)` table and the `H*_min` line by name.

- [ ] **Step 1: Write the acceptance check -- the failing test**

This is the test the run has to pass, written before the run so that "the run worked" is a check the run can fail rather than an impression from the log. `trust_horizon.py` already refuses the ways a run can be *wrong* (11 / 12 / 14 / 30) and prints its own self-check table; what this check adds is everything no line of `trust.txt` asserts: that all nine records came from **one git SHA which is HEAD**, on **one device which is `mps`**, at the **literal geometry** (context 5, horizon 45, split seed 0, 229 windows over 24 episodes), against **the diagnostics of this study** (field-by-field equality with `diagnostic_<arm>_seed<n>.json`, not with the trust pass's own copy of them), with the **self-check deltas exactly `0.0` independently of the `ok` flag**, the **boundary flag recomputed from the alphas**, and the **crossing arrays in range with `never_moved` equal to their NaN count**. It then recomputes the survival curves and `H*_q` from the crossing arrays by hand -- a fraction loop, not `mbfps.eval.trust.survival` -- so that Reading 2 in `trust.txt` has an independent second reading to be checked against in Step 9, and prints the provenance rows the results template is filled from.

```python
# runs/m3d_check_trust.py
"""Acceptance check for the M3d trust-horizon run. Exits 1 on the first failure.

Not a pytest file and not under version control (`runs/` is gitignored): it
checks ARTEFACTS, not code, and is only meaningful against the one directory
the run wrote. Usage:

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python runs/m3d_check_trust.py [OUT]

OUT defaults to runs/m3_study_v2. Step 8 points it at a doctored COPY of the
records first -- it must fail there, on the doctored field and not on the
provenance the copy shares with the real directory, BEFORE it is trusted on v2.
"""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from mbfps.eval.aggregate import SEEDS
from mbfps.eval.study import load_record
from mbfps.utils.config import ARMS

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/m3_study_v2")

# HAND-WRITTEN LITERALS, deliberately not derived from `ARMS` or `SEEDS`.
# A check parametrised over the collection under test shrinks with it: drop
# an arm from ARMS and the expected set drops the same cells, and a six-cell
# run passes as complete. The literals are compared for EQUALITY against the
# package's tuples so that a drift between the two is itself a failure.
EXPECTED_ARMS = ("pixel_ae", "frozen_ssl", "random_vit")
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_CELLS = {(arm, seed) for arm in EXPECTED_ARMS for seed in EXPECTED_SEEDS}
# Spec section 4, last bullet: one environment, one split, one geometry.
CONTEXT, HORIZON, SPLIT_SEED = 5, 45, 0
N_WINDOWS, N_EPISODES = 229, 24
# The code state the nine checkpoints were trained under (M3c Task 8 results).
# The trust pass runs under a LATER sha; both are asserted, each to its own.
CHECKPOINT_SHA = "ca3e140772d6bc741d4d04312763afe3dd754166"
ALPHA_GRID_ENDS = (0.0, 2.0)
QS = (0.5, 0.75, 0.9)
CHANNELS = ("probe", "free")

assert tuple(ARMS) == EXPECTED_ARMS, f"ARMS is {ARMS}, not {EXPECTED_ARMS}"
assert tuple(SEEDS) == EXPECTED_SEEDS, f"SEEDS is {SEEDS}, not {EXPECTED_SEEDS}"

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


def cell_of(path: Path, prefix: str) -> tuple[str, int]:
    """('frozen_ssl', 0) from 'trust_frozen_ssl_seed0.json'."""
    stem = path.name[len(prefix):-len(".json")]
    arm, _, seed = stem.rpartition("_seed")
    return arm, int(seed)


trust_paths = sorted(OUT.glob("trust_*_seed*.json"))
trust = {cell_of(p, "trust_"): load_record(p) for p in trust_paths}
missing = sorted(EXPECTED_CELLS - set(trust))
assert not missing, (
    f"expected 9 trust records in {OUT}, found {len(trust)}; missing {missing}")
strangers = sorted(set(trust) - EXPECTED_CELLS)
assert not strangers, f"trust records in {OUT} that are not study cells: {strangers}"
assert len(trust_paths) == 9, f"{len(trust_paths)} trust files for 9 cells: a duplicate"
for (arm, seed), r in trust.items():
    assert (r["arm"], r["seed"]) == (arm, seed), (
        f"trust_{arm}_seed{seed}.json says arm={r['arm']!r} seed={r['seed']!r}")

diag_missing = sorted(
    c for c in EXPECTED_CELLS if not (OUT / f"diagnostic_{c[0]}_seed{c[1]}.json").exists())
assert not diag_missing, f"no diagnostic to check against for {diag_missing}"
diag = {c: load_record(OUT / f"diagnostic_{c[0]}_seed{c[1]}.json") for c in EXPECTED_CELLS}
study = {c: load_record(OUT / f"result_{c[0]}_seed{c[1]}.json") for c in EXPECTED_CELLS}

shas = {r["git_sha"] for r in trust.values()}
assert shas == {head}, (
    f"git_sha across the nine trust records is {sorted(shas)}; HEAD is {head}. "
    "Either two code states produced these records, or HEAD moved after the run.")
checkpoint_shas = {r["git_sha"] for r in study.values()}
assert checkpoint_shas == {CHECKPOINT_SHA}, (
    f"the study records carry git_sha {sorted(checkpoint_shas)}, not the M3c "
    f"code state {CHECKPOINT_SHA}; these are not the checkpoints the spec reads")
devices = {r["device"] for r in trust.values()}
# `get_device` falls back to cpu SILENTLY when MPS is unavailable, and on cpu
# the shipped curves do not reproduce (diagnose_dynamics.py: 6-12 map units).
# The script would have exited 14 -- unless the fallback happened on a cell
# whose miss rounded to zero, which is why the device is asserted here too.
assert devices == {"mps"}, f"device across the nine trust records is {sorted(devices)}"
torches = {r["torch_version"] for r in trust.values()}
assert len(torches) == 1, f"torch_version varies across the run: {sorted(torches)}"

rows = []
crossings: dict[tuple[str, str], list[float]] = {(a, ch): [] for a in EXPECTED_ARMS for ch in CHANNELS}
for (arm, seed), r in sorted(trust.items()):
    d = diag[(arm, seed)]
    tag = f"{arm}/s{seed}"
    assert (r["context"], r["horizon"], r["split_seed"]) == (CONTEXT, HORIZON, SPLIT_SEED), (
        f"{tag}: context/horizon/split_seed = "
        f"{(r['context'], r['horizon'], r['split_seed'])}, not {(CONTEXT, HORIZON, SPLIT_SEED)}")
    assert (d["context"], d["horizon"], d["split_seed"]) == (CONTEXT, HORIZON, SPLIT_SEED), (
        f"{tag}: the DIAGNOSTIC is at {(d['context'], d['horizon'], d['split_seed'])}")
    assert (r["device"], r["torch_version"]) == (d["device"], d["torch_version"]), (
        f"{tag}: trust pass on {r['device']} torch {r['torch_version']}, diagnostic on "
        f"{d['device']} torch {d['torch_version']} -- not the same environment")
    # The same split, by NAME, as the diagnostic and as the study record.
    assert r["episodes"]["val"] == d["episodes"]["val"] == study[(arm, seed)]["episodes"]["val"], (
        f"{tag}: episodes.val differs between the trust record, the diagnostic and the study record")
    assert len(r["episodes"]["val"]) == N_EPISODES, (
        f"{tag}: {len(r['episodes']['val'])} validation episodes, not {N_EPISODES}")
    # Windows: the count and the per-window episode labels, against the
    # diagnostic's own -- NOT against the trust record's `windows_*_match`
    # flags, which are the script's opinion of the same comparison.
    assert r["windows"]["total"] == d["windows"]["total"] == N_WINDOWS, (
        f"{tag}: windows.total trust={r['windows']['total']} diagnostic="
        f"{d['windows']['total']}, expected {N_WINDOWS}")
    assert list(r["windows"]["episode"]) == list(d["windows"]["episode"]), (
        f"{tag}: windows.episode differs from the diagnostic's")
    assert len(set(r["windows"]["episode"])) == N_EPISODES, (
        f"{tag}: {len(set(r['windows']['episode']))} distinct episode labels, not {N_EPISODES}")
    # The probe: the diagnostic's R^2, copied exactly, and the measurability
    # band recomputed from the diagnostic's own curves (spec 3.1: positive
    # persistence-to-floor band at h=45; all nine shipped cells pass).
    r2 = r["probe"]["selection_r2"]
    assert r2 == d["probe"]["embedding_selection_r2"], (
        f"{tag}: probe.selection_r2={r2} is not the diagnostic's "
        f"{d['probe']['embedding_selection_r2']}")
    band = float(d["curves"]["persistence_position"][-1]) - float(d["curves"]["floor_position"][-1])
    assert band > 0.0, f"{tag}: persistence-to-floor band at h={HORIZON} is {band:.3f} <= 0"
    assert r["probe"]["measurable"] is True, (
        f"{tag}: probe.measurable={r['probe']['measurable']!r} with band {band:.3f} > 0")
    # The self-check, read from the deltas and NOT from `ok`: a record whose
    # `ok` was set by hand still fails here. Exactly 0.0, the spec's rule.
    sc = r["self_check"]
    assert sc["reference_position_max_delta"] == 0.0, (
        f"{tag}: reference_position_max_delta={sc['reference_position_max_delta']!r} "
        f"!= 0.0 -- not the ladder's rollout")
    assert sc["persistence_position_max_delta"] == 0.0, (
        f"{tag}: persistence_position_max_delta={sc['persistence_position_max_delta']!r} != 0.0")
    assert sc["windows_total_match"] is True and sc["windows_episode_match"] is True, (
        f"{tag}: self_check window flags {sc}")
    assert sc["ok"] is True, f"{tag}: self_check.ok is {sc['ok']!r} with every delta 0.0"
    assert "nonfinite" in r, f"{tag}: no `nonfinite` map; was this written by write_trust_record?"
    # Shapes and ranges. Crossings are 1..46 or NaN (never moved); the two
    # channels share the moved mask, so they share the NaN set, and
    # counts.never_moved is that set's size.
    n_nan = {}
    for ch in CHANNELS:
        c = np.asarray(r["crossing"][ch], dtype=float)
        assert c.shape == (N_WINDOWS,), f"{tag}: crossing.{ch} has shape {c.shape}"
        finite = c[np.isfinite(c)]
        assert np.all(finite == np.round(finite)), f"{tag}: crossing.{ch} has a non-integer step"
        assert finite.size and finite.min() >= 1 and finite.max() <= HORIZON + 1, (
            f"{tag}: crossing.{ch} outside 1..{HORIZON + 1}: "
            f"min {finite.min() if finite.size else 'n/a'} max {finite.max() if finite.size else 'n/a'}")
        n_nan[ch] = int(np.isnan(c).sum())
        crossings[(arm, ch)].extend(finite.tolist())
    assert n_nan["probe"] == n_nan["free"] == r["counts"]["never_moved"], (
        f"{tag}: never_moved={r['counts']['never_moved']} but crossing NaNs are {n_nan}")
    for name in ("margin", "ratio_probe", "ratio_raw", "ratio_free", "cosine"):
        a = np.asarray(r[name], dtype=float)
        assert a.shape == (N_WINDOWS, HORIZON), f"{tag}: {name} has shape {a.shape}"
    assert set(r["displacement"]) == {"probe_hat", "probe_real", "free_hat", "free_true"}, (
        f"{tag}: displacement keys {sorted(r['displacement'])}")
    for name in ("probe_hat", "probe_real", "free_hat", "free_true"):
        a = np.asarray(r["displacement"][name], dtype=float)
        assert a.shape == (N_WINDOWS, HORIZON), f"{tag}: displacement.{name} has shape {a.shape}"
    held = np.asarray(r["scale"]["held_out"], dtype=float)
    assert held.shape == (N_WINDOWS, HORIZON), f"{tag}: scale.held_out has shape {held.shape}"
    for name in ("alpha_a", "alpha_b", "score_a", "score_b", "boundary"):
        assert len(r["scale"][name]) == HORIZON, (
            f"{tag}: scale.{name} has {len(r['scale'][name])} entries, not {HORIZON}")
    for name in ("not_moved", "zero_displacement"):
        assert len(r["counts"][name]) == HORIZON, (
            f"{tag}: counts.{name} has {len(r['counts'][name])} entries, not {HORIZON}")
    # 24 episodes -> both folds exist; every alpha is on the grid; and the
    # boundary flag is RECOMPUTED from the alphas, not read back.
    assert r["scale"]["folds_available"] is True, f"{tag}: folds_available is not True with 24 episodes"
    alpha_a = np.asarray(r["scale"]["alpha_a"], dtype=float)
    alpha_b = np.asarray(r["scale"]["alpha_b"], dtype=float)
    assert np.all((alpha_a >= 0.0) & (alpha_a <= 2.0)) and np.all((alpha_b >= 0.0) & (alpha_b <= 2.0)), (
        f"{tag}: an alpha is off the [0, 2] grid")
    recomputed = [
        (a in ALPHA_GRID_ENDS) or (b in ALPHA_GRID_ENDS) for a, b in zip(alpha_a, alpha_b)]
    assert [bool(x) for x in r["scale"]["boundary"]] == recomputed, (
        f"{tag}: scale.boundary does not equal (alpha_a or alpha_b on a grid end) at steps "
        f"{[h + 1 for h, (x, y) in enumerate(zip(r['scale']['boundary'], recomputed)) if bool(x) != y]}")
    rows.append((
        arm, seed, r["git_sha"][:9], r["device"], r2, band,
        r["counts"]["never_moved"], r["counts"]["not_moved"][0], r["counts"]["not_moved"][4],
        r["counts"]["not_moved"][HORIZON - 1],
        float(np.nanmedian(np.asarray(r["crossing"]["probe"], dtype=float))),
        float(np.nanmedian(np.asarray(r["crossing"]["free"], dtype=float))),
        alpha_a[HORIZON - 1], alpha_b[HORIZON - 1], bool(r["scale"]["boundary"][HORIZON - 1]),
    ))

print(f"{'cell':<16}{'git_sha':>10}{'device':>7}{'probe_r2':>10}{'band@45':>9}"
      f"{'never':>6}{'nm@1':>6}{'nm@5':>6}{'nm@45':>7}{'hx_probe':>9}{'hx_free':>8}"
      f"{'aA@45':>7}{'aB@45':>7}{'bndry':>7}")
for arm, seed, sha, dev, r2, band, never, nm1, nm5, nm45, hxp, hxf, aa, ab, bd in rows:
    print(f"{f'{arm}/s{seed}':<16}{sha:>10}{dev:>7}{r2:>+10.4f}{band:>9.2f}{never:>6}"
          f"{nm1:>6}{nm5:>6}{nm45:>7}{hxp:>9.1f}{hxf:>8.1f}{aa:>7.2f}{ab:>7.2f}{str(bd):>7}")

# Reading 2, recomputed by hand from the crossing arrays: S(h) = fraction of
# finite (window, seed) draws with crossing > h; H*_q = the largest h with
# S(h) >= q. Printed, not asserted -- Step 9 compares it with trust.txt.
print(f"\n{'arm':<12}{'channel':<8}{'draws':>6}{'S(1)':>7}{'S(5)':>7}{'S(15)':>7}{'S(45)':>7}"
      f"{'H*0.5':>7}{'H*0.75':>8}{'H*0.9':>7}")
h_free_075: dict[str, int] = {}
for arm in EXPECTED_ARMS:
    for ch in CHANNELS:
        c = np.asarray(crossings[(arm, ch)], dtype=float)
        assert c.size == 3 * N_WINDOWS - sum(
            trust[(arm, s)]["counts"]["never_moved"] for s in EXPECTED_SEEDS), (
            f"{arm}/{ch}: {c.size} finite draws")
        s = [float(np.mean(c > h)) for h in range(HORIZON + 1)]
        assert s[0] == 1.0, f"{arm}/{ch}: S(0)={s[0]} -- a crossing below 1"
        hq = {q: max(h for h in range(HORIZON + 1) if s[h] >= q) for q in QS}
        if ch == "free":
            h_free_075[arm] = hq[0.75]
        print(f"{arm:<12}{ch:<8}{c.size:>6}{s[1]:>7.3f}{s[5]:>7.3f}{s[15]:>7.3f}{s[45]:>7.3f}"
              f"{hq[0.5]:>7}{hq[0.75]:>8}{hq[0.9]:>7}")
print(f"H*_min (min over arms of probe-free H*_0.75) = {min(h_free_075.values())}  "
      f"{h_free_075}")
print(f"checkpoints {CHECKPOINT_SHA[:9]}; trust pass HEAD {head}; torch {sorted(torches)[0]}")
print("OK: nine trust records, one git_sha == HEAD, one device == mps, self-check exactly 0.0 on 9/9")
```

- [ ] **Step 2: Run it against the study directory before the run -- it must fail**

Run:
```bash
cd /Users/raphaelchen/Desktop/csgo-bot
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python runs/m3d_check_trust.py
```
Expected: FAIL --
```
AssertionError: expected 9 trust records in runs/m3_study_v2, found 0; missing [('frozen_ssl', 0), ('frozen_ssl', 1), ('frozen_ssl', 2), ('pixel_ae', 0), ('pixel_ae', 1), ('pixel_ae', 2), ('random_vit', 0), ('random_vit', 1), ('random_vit', 2)]
```
(The glob over a directory with nine `diagnostic_*.json` and nine `result_*.json` but no `trust_*.json` is empty; that is the correct "nothing has run" reading.) If it instead fails on the `dirty` assertion, commit Task 6's work first -- the run must be made from a committed tree or `git_sha` names a state nobody can check out. If it fails on `ARMS is ...`, a preceding task changed the arm tuple and this plan's "nine" is no longer nine: stop.

- [ ] **Step 3: Pre-flight -- everything the script will not check for you**

Each of these is a way the run spends its minutes on the wrong measurement, in the order to look at them. All must hold before Step 4.

```bash
cd /Users/raphaelchen/Desktop/csgo-bot

# 1. The suite is green on the tree that will run. Expected: "N passed", exit 0,
#    with N the number Task 6's full-suite step recorded -- never below the
#    1278 the contract states for the tree before Task 1.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider

# 2. The tree that will run is committed under src/ and scripts/. Expected: no
#    output. (The untracked `study.log` at the repo root is M3b's leftover and
#    is outside these two paths; leave it.)
git status --porcelain -- src scripts

# 3. No stale bytecode: the nine records are produced by the committed source.
find src scripts -name __pycache__ -type d -exec rm -rf {} +

# 4. The nine diagnostics, checkpoints and study records are present.
#    Expected: 9, 9, 9.
ls runs/m3_study_v2/diagnostic_*_seed*.json | wc -l
ls runs/m3_study_v2/world_model_*_seed*.pt | wc -l
ls runs/m3_study_v2/result_*_seed*.json | wc -l

# 5. The diagnostics are the ladder's own, from the M3c run that exited 0 on
#    mps: every self-check column exactly 0 and every device mps. Expected:
#    "0" (diagnose.exit), then nine lines "mps 2.13.0 0.0 229".
cat runs/m3_study_v2/diagnose.exit; echo
.venv/bin/python -c "
import glob, json
for p in sorted(glob.glob('runs/m3_study_v2/diagnostic_*_seed*.json')):
    r = json.load(open(p))
    print(r['device'], r['torch_version'], r['self_checks']['record_reproduction'], r['windows']['total'])"

# 6. MPS is what the reference pass will run on. `get_device` falls back to
#    cpu silently, and on cpu the shipped curves miss by 6-12 map units and
#    the run ends 14 after the first cell's probe refit. Expected: True
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"

# 7. The data the split is cut from: 122 episodes, three feature caches.
#    Expected: 122 / 122 / 122.
ls data/my_way_home/*.features.npy | wc -l
ls data/my_way_home/*.features_random_vit.npy | wc -l
ls data/my_way_home/*.features_pixel_ae.npy | wc -l

# 8. The script's defaults are the ones this task passes explicitly anyway.
#    Expected: the usage text naming --out, --device, --data, --context,
#    --horizon, --arms, --seeds; exit 0.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/trust_horizon.py --help

# 9. Nothing from an earlier attempt is in the way. Expected: no output --
#    no trust_*.json, no trust.txt, no trust.exit. (If a previous attempt left
#    them, delete ONLY those: `rm -f runs/m3_study_v2/trust*`. Never anything
#    else in the directory; the checkpoints are 13.5 h of compute.)
ls runs/m3_study_v2/trust* 2>/dev/null

# 10. Record HEAD before launch, so the sha the records will carry is on disk
#     beside them and the run cannot be misattributed afterwards.
git rev-parse HEAD | tee runs/m3_study_v2/trust.head
```

There is no smoke step: the whole run is 10-15 minutes, each cell's self-check is decided before its record is written, and a failure at cell 4 costs five minutes, not 4.5 hours. The script tests of Tasks 4 and 6 already drove `main` end to end on the fixture cell.

**Operating rule for the run: no HEAD movement of any kind (commit, checkout, switch, reset, rebase, stash, pull) on this checkout until `trust.exit` exists.** `git_sha` is sampled at record-build time, per cell; a HEAD move mid-run splits the nine records across two shas and the check's `shas == {head}` assertion fails, with the cells from the older sha to be re-run.

- [ ] **Step 4: Run the trust pass**

One invocation, all arms, all seeds. **Not through any tool with a ten-minute ceiling**: the run is launched detached with `nohup`, returns to the shell at once, and is polled. `caffeinate -dimsu`, not `-i`: the M3c plan's Global Constraints record that `-i` was insufficient on this host. `-u` is Python's unbuffered stdout, so `tee` sees each line as it is printed. `set -o pipefail` inside the `bash -c` makes the `$?` written to `trust.exit` the script's status rather than `tee`'s.

```bash
cd /Users/raphaelchen/Desktop/csgo-bot
date -u +%FT%TZ | tee runs/m3_study_v2/trust.started
nohup bash -c '
  set -o pipefail
  PYTHONDONTWRITEBYTECODE=1 caffeinate -dimsu .venv/bin/python -u scripts/trust_horizon.py \
    --out runs/m3_study_v2 --device mps --data data/my_way_home 2>&1 | tee runs/m3_study_v2/trust.log
  echo $? > runs/m3_study_v2/trust.exit
' > /dev/null 2>&1 &
echo "launched pid $!"
```

Expected duration **~10-15 min** on this Mac: per cell, the checkpoint load, the ~20 s probe refit (`fit_probes` on the 98 training episodes at the cell's seed), the `evaluate_rollout` reproduction that decides 14, and one reference pass over the 229 windows with `keep_trajectories=True`; the M3c ladder ran nine passes per cell in ~31 min for nine cells, so one pass plus one rollout is ~1-1.5 min per cell. Then the pooling (two `pool_ratio` bootstraps of 2000 draws per arm and channel, under a minute) and the readings.

Poll with short commands, each returning at once -- never a `sleep` loop inside a tool call:
```bash
cd /Users/raphaelchen/Desktop/csgo-bot
ls runs/m3_study_v2/trust_*_seed*.json 2>/dev/null | wc -l    # 0 -> 9 as the cells complete
tail -5 runs/m3_study_v2/trust.log
cat runs/m3_study_v2/trust.exit 2>/dev/null                    # absent until the process ends
```

**What the log looks like.** Each cell prints ONE progress line as its record is written (Task 4's `_run_cell`), in the shape

```
pixel_ae seed 0: 229 windows from 24 validation episodes; folds available; self-check max|delta| reference 0.0e+00 persistence 0.0e+00; wrote runs/m3_study_v2/trust_pixel_ae_seed0.json
```

so `trust.log` grows by one line per cell, ~1-1.5 min apart, in `ARMS x seeds` order (`pixel_ae` s0 s1 s2, `frozen_ssl` ..., `random_vit` ...), and the `trust_*_seed*.json` count climbs from 0 to 9 beside it. The tables and the readings are printed after the ninth line, all at once. Records are written per cell (`write_trust_record` after each cell's checks pass), so a run that dies on cell 7 leaves six complete records, six progress lines, and no `trust.txt`.

When `trust.exit` exists:
```bash
cat runs/m3_study_v2/trust.exit; date -u +%FT%TZ
```
Expected: `0`. The last line of `trust.log` is Reading 2's `H*_min = ...` line (Reading 2 is printed last; Reading 1's `reading 1: <STATUS> -- <reason>` line and its `verdict:` line are above it), and `trust.txt` is byte-identical to the report part of the log -- everything from the `--- trust readings at h = 45: ...` header on (`write_readings` writes the same text `main` printed; the nine progress lines above it are in the log only).

**The non-zero statuses, and what each means for this task.** None is a result; each names what to fix, and every one is decided **before any reading is printed** (spec Section 2.3: no reading is printed unless every cell's checks hold).

| exit | name | meaning here | action |
|---|---|---|---|
| `11` | `EXIT_NO_CHECKPOINTS` | a planned cell lacks its checkpoint, study record or diagnostic | Step 3 item 4 was not satisfied; nothing was overwritten -- restore the file and re-run |
| `12` | `EXIT_SPLIT_MISMATCH` | `episode_split(seed=0)` on `--data` does not name the record's `episodes.val` | `--data` points at a different directory than the study's; fix the path |
| `14` | `EXIT_RECORD_MISMATCH` | `evaluate_rollout` did not reproduce the study record's `curves.rssm_position` bitwise | the environment: cpu (silent MPS fallback, item 6) or a different torch; nothing in this task's code -- fix the device and re-run |
| `30` | `EXIT_SELF_CHECK_FAILED` | the trust pass's mean curves or windows are not the diagnostic's; the message names the cell, the curve and the step | the reference pass is not the ladder's (wrong `noise_reference`, arms not `{}`, a probe refit at another context or seed) or the diagnostic is not this study's -- a Task 3/4 defect or a foreign diagnostic; find which before re-running |
| `2` | argparse | a bad flag | fix the command line |
| `1` | uncaught traceback | not a status this script defines | read the traceback at the end of `trust.log`; a Task 4/6 defect |

A `14` or `30` after a partial set of `trust_*.json` leaves those records on disk; they are the records of cells whose checks passed, but the run is re-launched from scratch after the fix (`rm -f runs/m3_study_v2/trust*` first, then Step 3 item 10 and Step 4 again), because the pooling and `trust.txt` need all nine in one process.

- [ ] **Step 5: Run the acceptance check on the real records -- provenance first**

```bash
cd /Users/raphaelchen/Desktop/csgo-bot
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python runs/m3d_check_trust.py | tee runs/m3_study_v2/trust_provenance.txt
```
Expected: the nine provenance rows, then the recomputed Reading 2 block, then
```
checkpoints ca3e14077; trust pass HEAD <sha>; torch 2.13.0
OK: nine trust records, one git_sha == HEAD, one device == mps, self-check exactly 0.0 on 9/9
```
with `<sha>` equal to `runs/m3_study_v2/trust.head`. This is the *first* pass of the check on the real directory; Step 8 proves the check can fail on a doctored copy before the results section is written from it. A failure here after an exit-0 run is a finding: the script and the check disagree about the same records, and whichever is wrong is a defect to fix before anything is recorded.

- [ ] **Step 6: Read `trust.txt` in the spec's order**

```bash
cd /Users/raphaelchen/Desktop/csgo-bot
cat runs/m3_study_v2/trust.txt
```

Read it top to bottom; the sections come in the order `_readings_text` (Task 6) prints them, and the results template in Step 10 is filled section by section. The literal headers, verbatim from Tasks 5 and 6, are:

```
--- trust readings at h = 45: 9 cells, arms ['pixel_ae', 'frozen_ssl', 'random_vit'], seeds [0, 1, 2]; means and contrasts seed-averaged per window and clustered by episode, crossings per (window, seed) draw, ratios as ratios of medians over the moved windows (bootstrap=2000 bootstrap_seed=0) ---

--- self-check per cell (spec 2.3): the trust pass's window-mean curves against the diagnostic's, max |delta| exactly 0.0 (a non-zero delta is exit 30 before any reading), and the windows ---
cell             ref_max|delta| pers_max|delta|  windows  episodes_match  probe_r2  measurable  never_moved     ok
...nine rows...

clusters: 24 validation episode(s) contribute windows; z_fam = cluster_threshold(8, 24) = 3.01 (Bonferroni over the 8 clustered contrasts of Reading 1, read against t(23))
probe-based pooling: 9 of 9 cells measurable (persistence-to-floor band at h=45 > 0); excluded: none
windows not moved at h=45 (excluded from every pooled series): frozen_ssl/s0=..., ...

--- per arm at h = 45: delta and cos seed-averaged per window over the measurable cells' moved windows, mean +- episode-clustered se; ratios of medians with the 95% episode-bootstrap interval; R_raw the median of the moved draws' |d_hat| / |d|, for information ---
arm           delta(45)       se    n  R_probe(45)          [95% CI]  R_free(45)          [95% CI]  R_raw(45)  cos(45)      se    n  moved  zero_dhat
...three rows...

reading 1: <STATUS> -- <reason>
--- Reading 1: does the h=45 gate reward slow drift?  (z_fam 3.01, family 8, treatment frozen_ssl, control random_vit)
pooled:
  (i)   ...
  (ii)  ...
  (iii) ...
  probe control: ...
seed 0:
  ...
seed 1:
  ...
seed 2:
  ...
per-seed agreement: (i) ./3, (ii) ./3, (iii) ./3  (2 of 3 required)
best-Δ arm: ...; least-moving arm: ...
verdict: Reading 1 <status value> -- decided by: <reason>
for information: h_x probe pixel_ae - random_vit per draw: estimate ..., se ..., z ... (windows ...); decides nothing
for information: h_x free pixel_ae - random_vit per draw: estimate ..., se ..., z ... (windows ...); decides nothing

--- sensitivity: the probe-based statistics with probe selection R^2 < 0.1 excluded (1 of 9 cells: pixel_ae/s1); this changes no verdict ---
statistic (h_x per draw)                    estimate        se         z   windows
...

--- Reading 2: the horizon M4 designs around  (S(h) = fraction of moved draws with h× > h; H*_q = largest h with S(h) >= q; q = 0.75 pre-registered; no verdict)
arm         channel     H*_0.5  H*_0.75   H*_0.9   S(h), h = 0..H
...six rows...
H*_min = ...  (min over arms of the probe-free H*_0.75; probe-based H*_0.75 beside it: pixel_ae ..., frozen_ssl ..., random_vit ...)
```

**(1) The self-check table -- nine rows, both deltas exactly `0.0e+00`.** One row per cell in `ARMS x seeds` order: `ref_max|delta|`, `pers_max|delta|`, `windows` (229), `episodes_match` (True), `probe_r2` (the diagnostic's; `pixel_ae`/s1 reads `0.018`, every other cell >= `0.249`), `measurable` (True on all nine: the persistence-to-floor band at h=45 is 19.67-69.36 map units, positive everywhere), `never_moved`, and `ok`. Every delta reads exactly `0.0e+00`, not `1.0e-06`: the rule is the ladder's `record_reproduction` rule, bitwise on mps. If the run exited 0 this table cannot show anything else -- a non-zero delta is exit 30 before the table is printed -- so what this table is *for* is the results section: it is the evidence that the nine records measure what the ladder measured (same windows, same rollout, same refit probe).

**(2) The pooling notes.** `clusters: 24 validation episode(s) contribute windows; z_fam = cluster_threshold(8, 24) = 3.01`, read against t(23). `3.47` here means the family was counted as the ladder's 24, not this reading's 8; `2.76` means 229 clusters, i.e. windows rather than episodes; `nan` means fewer than two clusters -- all three are wrong and each is a mutation row in Step 8. Then `probe-based pooling: 9 of 9 cells measurable ...; excluded: none` (the sensitivity block below is the ONLY place a cell is excluded), and the `windows not moved at h=45` count per cell (near 1 %).

**(3) The per-arm block and Reading 1 -- the contrasts, the per-condition lines, the status.** In the ladder's style:
- the per-arm block at h=45, one row per arm: pooled `delta(45)` (mean, cluster SE, kept windows), `R_probe(45)` and `R_free(45)` (ratio of medians with the 95 % episode-bootstrap interval), `R_raw(45)` beside them for information, `cos(45)` (mean, cluster SE, kept windows), and the moved / zero-`d̂` counts summed over the arm's three cells;
- the `reading 1: <STATUS> -- <reason>` summary line, then Task 5's block under its own header: the three condition lines pooled, each stating the rule it was decided by -- **(i)** the best-Δ arm (the arm whose `Δ(45)` contrast against *each* other arm has z > 3.01, or undecidable) and the least-moving arm (smallest `R_probe(45)`, and the same arm under `R_free(45)`, or unresolved through the probe) -- holds iff they are the same arm; **(ii)** the `cos(45)` contrast `frozen_ssl − random_vit`, z > 3.01; **(iii)** the held-out `c(45)` contrast on fold A and on fold B: both z < −3.01 holds, either z > 3.01 fails, otherwise undecided, and *unreadable* if `frozen_ssl`'s or `random_vit`'s α is on a grid end on either fold at h=45; the **probe control** line with both `h×` z's: fires iff both clear 3.01 with opposite signs;
- the same three conditions within each seed alone (the clustered SE over that seed's windows), the per-seed agreement count per condition, the best-Δ / least-moving line, and the `verdict:` line naming the status and the rule it was decided by;
- `pixel_ae`'s `h×` pair against `random_vit`, probe and free, for information (it decides nothing);
- the sensitivity block: every probe-based statistic recomputed with cells of selection R² < 0.1 excluded -- on these records that is `pixel_ae`/s1 alone, so its header reads `(1 of 9 cells: pixel_ae/s1)` -- **and it changes no verdict**;
- the **status** -- one of `SUPPORTED`, `NOT_SUPPORTED`, `NOT_TESTABLE`, `UNRESOLVED_PROBE`, `UNRESOLVED_ALPHA` -- and its reason, decided in `reading_one`'s order: probe control fires → `UNRESOLVED_PROBE`; (iii) unreadable → `UNRESOLVED_ALPHA`; (i) undecidable (no best-Δ arm) → `NOT_TESTABLE`; (i) unresolved through the probe (the least-moving arm differs between `R_probe` and `R_free`) → `UNRESOLVED_PROBE`; all three hold pooled and each in at least two of three seeds → `SUPPORTED`; else `NOT_SUPPORTED`.

**(4) Reading 2 -- the survival table, `H*_q`, and `H*_min`.** Per arm and per channel (`probe`, `free`): `H*_0.5`, `H*_0.75` (pre-registered), `H*_0.9` -- integers -- then `S(h)` at every h = 0..45 over the draws that moved (687 per arm less the `never_moved` counts, which the self-check table prints), `S(0) = 1.00` on every line. Then the one derived number: `H*_min = min over arms of the probe-free H*_0.75`, with the probe-based `H*_0.75` beside it. Check three things as you read: `S(h)` is non-increasing in h on every line (a crossing is a first crossing); every `H*_q` is the largest h at which `S(h) >= q` on its own line; and the block agrees with the hand recomputation in `trust_provenance.txt` -- same `S(1)`, `S(5)`, `S(15)`, `S(45)`, same `H*_q`, same `H*_min` (Step 9 makes this a mutation row). No verdict is attached to Reading 2; `H*_min` is a number M4 designs around, not a claim.

**Every one of the five statuses is a pre-registered result.** `NOT_TESTABLE` says the three arms' `Δ(45)` cannot be ordered at this precision -- with `gap_closed` means of −0.46 / −0.72 / −0.81 on a floor of 120-250 map units it is the likely outcome, and the spec wrote the rule knowing that. `NOT_SUPPORTED` with (ii) failing alone is the reading the spec calls informative: the prior does not drift slowly, it moves wrong. `UNRESOLVED_*` says the instrument could not separate the two readings, and names which part of it. None of these is re-run, re-thresholded or re-read; the section in Step 10 records the status, the rule that produced it and the numbers it was produced from.

- [ ] **Step 7: Read the three probe-free twins against their probe-based readings**

Spec Section 4: probe-based numbers inherit the probe's R² and are comparable between arms but not in absolute units, so every probe-based reading has a probe-free twin. Before writing the results, put the three pairs side by side from `trust.txt` and say, in the results section, whether they agree:

| probe-based | probe-free twin | agree means |
|---|---|---|
| `R_probe(45)` argmin over arms | `R_free(45)` argmin | the least-moving arm is the same arm (else (i) is unresolved through the probe, and the status line already says so) |
| `H*_0.75` per arm, probe channel | `H*_0.75` per arm, free channel | the arms are ordered the same way, and `H*_min` (which is free by construction) would be the same number read through the probe |
| the `h×` contrast, probe channel | the `h×` contrast, free channel | same sign, or either interval covering 0 (the probe control fires only on opposite signs that both clear 3.01) |

The `pixel_ae`/s1 cell (R² 0.018) is where the channels are most likely to part; the sensitivity line says whether the probe-based numbers move without it.

- [ ] **Step 8: Mutation-test -- what a wrong run looks like, and which check catches it**

There is no module under test in this task, so the harness's three self-checks apply to the *run* rather than to a mutated function, and each has a real meaning here:

1. **Run at the export, not the working tree** ↔ the nine records are produced by the *committed* tree: Step 3 item 2's `git status --porcelain -- src scripts` is empty, `__pycache__` is cleared, `PYTHONDONTWRITEBYTECODE=1` is on every command. `git_sha` is only evidence of a code state under those three conditions, and the check's `dirty` assertion enforces the first.
2. **Prove the harness on a known-fatal mutation first** ↔ the doctored copies below: the check must fail on each *before* Step 10 is written from its output on the real directory. (Step 2 showed it fails on an empty directory; that proves only the presence assertion.)
3. **No stale bytecode** ↔ Step 3 item 3 and `PYTHONDONTWRITEBYTECODE=1` throughout.

The doctored copies. `scratchpad/` is the directory `.gitignore` excludes (`scratch/` is not in it), so the copy goes under `scratchpad/m3d_doctored/` and is deleted at the end of this step anyway; `git status` at Step 12 must not show it. The trust and diagnostic records are copied (350 KB and a few MB each); the nine 5 MB study records are symlinked -- the check reads only their `git_sha`.

```bash
cd /Users/raphaelchen/Desktop/csgo-bot
rm -rf scratchpad/m3d_doctored && mkdir -p scratchpad/m3d_doctored
cp runs/m3_study_v2/trust_*_seed*.json runs/m3_study_v2/diagnostic_*_seed*.json scratchpad/m3d_doctored/
for f in runs/m3_study_v2/result_*_seed*.json; do ln -s "$PWD/$f" scratchpad/m3d_doctored/; done
# The unmodified copy passes: the harness is not failing on the copy mechanism.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python runs/m3d_check_trust.py scratchpad/m3d_doctored | tail -1
```
Expected: `OK: nine trust records, one git_sha == HEAD, one device == mps, self-check exactly 0.0 on 9/9`.

Then each doctoring, applied to the copy and reverted by re-copying the one file, with the assertion it must trip. The doctor writes with `json.loads`/`json.dumps`, which keeps `null` and the `nonfinite` map intact, so `load_record` in the check still restores every NaN:

```bash
cd /Users/raphaelchen/Desktop/csgo-bot
doctor() {  # $1 = file, $2 = python statement over `r`
  .venv/bin/python -c "
import json, sys
from pathlib import Path
p = Path('scratchpad/m3d_doctored') / sys.argv[1]
r = json.loads(p.read_text())
exec(sys.argv[2])
p.write_text(json.dumps(r))" "$1" "$2"
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python runs/m3d_check_trust.py scratchpad/m3d_doctored 2>&1 | grep -m1 "AssertionError" || echo "SURVIVED: $2"
  cp "runs/m3_study_v2/$1" scratchpad/m3d_doctored/
}
# FATAL first: a self-check delta set by hand while `ok` stays True.
doctor trust_frozen_ssl_seed0.json "r['self_check']['reference_position_max_delta'] = 1e-06"
doctor trust_frozen_ssl_seed0.json "r['self_check']['persistence_position_max_delta'] = 1e-06"
doctor trust_frozen_ssl_seed0.json "r['self_check']['ok'] = False"
doctor trust_pixel_ae_seed1.json   "r['windows']['episode'][0] = 23"
doctor trust_pixel_ae_seed1.json   "r['windows']['total'] = 228"
doctor trust_random_vit_seed2.json "r['probe']['selection_r2'] = 0.3"
doctor trust_random_vit_seed2.json "r['probe']['measurable'] = False"
doctor trust_random_vit_seed2.json "r['device'] = 'cpu'"
doctor trust_random_vit_seed2.json "r['git_sha'] = 'ca3e140772d6bc741d4d04312763afe3dd754166'"
doctor trust_frozen_ssl_seed1.json "r['horizon'] = 30"
doctor trust_frozen_ssl_seed1.json "r['episodes']['val'] = r['episodes']['val'][::-1]"
# A never-moved window is `null` in the file and is restored to NaN from the
# `nonfinite` map by load_record OVER whatever the doctor wrote there, so the
# crossing doctorings pick the first window that is finite in the file.
doctor trust_frozen_ssl_seed1.json "i = next(k for k, v in enumerate(r['crossing']['probe']) if v is not None); r['crossing']['probe'][i] = 47"
doctor trust_frozen_ssl_seed1.json "i = next(k for k, v in enumerate(r['crossing']['free']) if v is not None); r['crossing']['free'][i] = 0"
doctor trust_frozen_ssl_seed1.json "r['counts']['never_moved'] += 1"
doctor trust_frozen_ssl_seed1.json "r['scale']['boundary'][44] = not r['scale']['boundary'][44]"
doctor trust_frozen_ssl_seed1.json "r['scale']['alpha_a'][44] = 2.5"
doctor trust_frozen_ssl_seed1.json "r['scale']['folds_available'] = False"
# One row ADDED, not removed: a dotted path in the `nonfinite` map that
# indexes past a shortened list would raise IndexError inside load_record,
# not the AssertionError this table is checking for.
doctor trust_frozen_ssl_seed1.json "r['margin'].append(r['margin'][0])"
doctor trust_frozen_ssl_seed1.json "r['displacement']['probe_real'].append(r['displacement']['probe_real'][0])"
doctor trust_frozen_ssl_seed1.json "del r['nonfinite']"
doctor diagnostic_pixel_ae_seed0.json "r['curves']['floor_position'][-1] = r['curves']['persistence_position'][-1] + 1.0"
rm scratchpad/m3d_doctored/trust_random_vit_seed1.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python runs/m3d_check_trust.py scratchpad/m3d_doctored 2>&1 | grep -m1 "AssertionError"
cp runs/m3_study_v2/trust_random_vit_seed1.json scratchpad/m3d_doctored/
rm -rf scratchpad/m3d_doctored
```

Every line must print an `AssertionError`, none a `SURVIVED`. What each must trip:

| doctoring | must be caught by |
|---|---|
| `reference_position_max_delta = 1e-06`, `ok` left `True` (harness self-check, fatal) | `frozen_ssl/s0: reference_position_max_delta=1e-06 != 0.0 -- not the ladder's rollout` |
| `persistence_position_max_delta = 1e-06` | `...persistence_position_max_delta=1e-06 != 0.0` |
| `ok = False` with both deltas `0.0` | `...self_check.ok is False with every delta 0.0` |
| `windows.episode[0] = 23` | `pixel_ae/s1: windows.episode differs from the diagnostic's` |
| `windows.total = 228` | `...windows.total trust=228 diagnostic=229, expected 229` |
| `probe.selection_r2 = 0.3` | `random_vit/s2: probe.selection_r2=0.3 is not the diagnostic's 0.24859528825463967` |
| `probe.measurable = False` | `...probe.measurable=False with band 53.756 > 0` |
| `device = 'cpu'` | `device across the nine trust records is ['cpu', 'mps']` |
| `git_sha` = the checkpoint sha | `git_sha across the nine trust records is [...]; HEAD is ...` |
| `horizon = 30` | `frozen_ssl/s1: context/horizon/split_seed = (5, 30, 0), not (5, 45, 0)` |
| `episodes.val` reversed | `...episodes.val differs between the trust record, the diagnostic and the study record` |
| the first finite `crossing.probe` entry set to 47 | `...crossing.probe outside 1..46: min ... max 47.0` |
| the first finite `crossing.free` entry set to 0 | `...crossing.free outside 1..46: min 0.0 ...` |
| `counts.never_moved + 1` | `...never_moved=... but crossing NaNs are {...}` |
| `scale.boundary[44]` flipped | `...scale.boundary does not equal (alpha_a or alpha_b on a grid end) at steps [45]` |
| `scale.alpha_a[44] = 2.5` | `...an alpha is off the [0, 2] grid` |
| `folds_available = False` | `...folds_available is not True with 24 episodes` |
| `margin` one row longer | `...margin has shape (230, 45)` |
| `displacement.probe_real` one row longer | `...displacement.probe_real has shape (230, 45)` |
| `nonfinite` key deleted | `...no \`nonfinite\` map; was this written by write_trust_record?` |
| the diagnostic's floor raised above persistence at h=45 | `pixel_ae/s0: persistence-to-floor band at h=45 is -1.000 <= 0` |
| `trust_random_vit_seed1.json` removed | `expected 9 trust records in scratchpad/m3d_doctored, found 8; missing [('random_vit', 1)]` |

Then the table this step is really about. Every row is a way the run could be *wrong while producing nine records and a complete `trust.txt`*, or wrong in a way `trust_horizon.py` itself refuses; each names the check that refuses it. A row with no check would be a missing check, and there is none.

| corruption of the run | caught by |
|---|---|
| run on cpu (`--device cpu`, or MPS unavailable and `get_device` fell back silently) | exit **14** `EXIT_RECORD_MISMATCH` on the first cell, before any reading (`evaluate_rollout` misses the record's `curves.rssm_position` by 6-12 map units on cpu); the check's `device across the nine trust records is ['cpu']` if a cell's miss ever rounded to zero |
| a diagnostic from another run copied in (M3b's `runs/m3_study/diagnostic_frozen_ssl_seed0.json`, or one regenerated on cpu) | exit **30** `EXIT_SELF_CHECK_FAILED`, naming the cell, `reference_position` and the first step where max\|Δ\| ≠ 0 (the reference pass reproduces *this* checkpoint's imagination on mps; a foreign curve does not match it), or `windows.total` if its windows differ; the check's `windows.episode differs from the diagnostic's` / `episodes.val differs` |
| a diagnostic, checkpoint or study record missing for a planned cell | exit **11** `EXIT_NO_CHECKPOINTS` (`load_cell`'s typed error); Step 3 item 4 |
| `--data` pointing at a different episode directory | exit **12** `EXIT_SPLIT_MISMATCH`: `episode_split(seed=0)` names episodes the record's `episodes.val` does not |
| `--horizon 30` or `--context 3` against records at 5 / 45 | refused as `diagnose_dynamics.py` refuses it -- the record's `curves.rssm_position` at 45 steps is not reproduced by a rollout at another geometry, exit **14** -- and the check's `context/horizon/split_seed = ..., not (5, 45, 0)` |
| a trust record edited after the run (a hand-set `ok`, a crossing changed, a boundary flag flipped) | the check: every delta asserted `== 0.0` independently of `ok`; crossings in `1..46` or NaN with `never_moved` equal to the NaN count; `boundary` recomputed from `alpha_a`/`alpha_b`. And it cannot reach `trust.txt`: the readings are pooled in-process from the arrays before the file is written, so the only route from a record to a reading is a re-run, which rewrites the record |
| a pooled input that is non-finite (a NaN cluster SE, an all-NaN series) | the reading prints `n/a` for that statistic and the condition reads `None` (undecided / undecidable), never `holds`; a `SUPPORTED` status cannot be built on an `n/a` |
| a seed missing (`--seeds 0 1`, or a cell's checkpoint deleted so the planned set is eight) | the check: `expected 9 trust records ..., found 6; missing [...]`; in `trust.txt` the run header reads `6 cells, ..., seeds [0, 1]` and the per-seed agreement line reads `(2 of 2 required)` -- two seeds agreeing satisfies "at least two of the three seeds", so the *check's count* is the guard, not the reading |
| the tree edited after the run, or before it without committing | the `dirty` assertion; and `shas == {head}` once HEAD moves |
| a commit made mid-run (two code states across the nine records) | `git_sha across the nine trust records is [a, b]; HEAD is b` |
| stale bytecode from a mutated source | `PYTHONDONTWRITEBYTECODE=1` on every command and Step 3 item 3 |
| the probe not refit as `diagnose_dynamics.py` refits it (another context, another seed, the train split by the cell seed) | exit **14**: `evaluate_rollout` with that probe misses `rssm_position` (~25 map units for a context mismatch, per `diagnose_dynamics.py`'s own comment); if the rollout somehow reproduced, exit **30** on `reference_position` |
| the reference pass not the ladder's (`noise_reference=False`, `arms` not `{}`, RNG snapshot not matched) | exit **30**: the stream sequence differs from the ladder's and `reference_position` max\|Δ\| ≠ 0 |
| the trust pass's own mean over windows taken another way (`np.mean` over a list of lists, a `float32` accumulation) | exit **30**: bitwise equality with the diagnostic's `np.stack(rows).mean(axis=0)` is the rule, and 1e-06 is not 0.0 |
| the pooled table clustered by window rather than by episode | the pooling notes read `clusters: 229 validation episode(s) ...; z_fam = cluster_threshold(8, 229) = 2.76` rather than `clusters: 24 ...; z_fam = cluster_threshold(8, 24) = 3.01` |
| the family counted as the ladder's 24 rather than Reading 1's 8 | `z_fam` reads `3.47` rather than `3.01` |
| a cell with R² < 0.1 dropped from the main pooling rather than reported on the sensitivity line | the pooling notes still read `probe-based pooling: 9 of 9 cells measurable ...; excluded: none` and the per-arm block's `pixel_ae` row still pools three seeds (its `moved` count is summed over three cells, ~680, not two); only the sensitivity header reads `(1 of 9 cells: pixel_ae/s1)`. The sensitivity block is the only place a cell is excluded, and it changes no verdict |
| `--out runs/m3_study` (the M3b directory) | exit **11**: no `pixel_ae` checkpoints and no `diagnostic_*.json` for its cells; nothing there is overwritten |
| a `trust.txt` left over from an earlier attempt beside a run that died | `trust.exit` non-zero or absent; fewer than nine `trust_*.json`; `trust.txt` older than `trust.started` |
| `H*_q` computed with rounding or interpolation, or `S(h)` over all draws including the never-moved | disagreement with the hand recomputation in `trust_provenance.txt` (Step 9): same `S(1)`, `S(5)`, `S(15)`, `S(45)`, `H*_0.5`, `H*_0.75`, `H*_0.9`, `H*_min` on every arm and channel, or the results are not written |

All rows of the doctoring table were run and caught before Step 10 was written -- record the date in the results section. Any corruption you can think of that no row catches is a missing assertion in `runs/m3d_check_trust.py`. Add it, re-run the doctoring table, then Step 5 again, before Step 10.

- [ ] **Step 9: The two readings of Reading 2 must agree**

`trust.txt`'s Reading 2 block (Task 6, through `mbfps.eval.trust.survival` and `trust_horizon`) and `trust_provenance.txt`'s recomputation (Step 1, a fraction loop over the same crossing arrays) are two implementations of the spec's Section 3.3 definition. Put them side by side:

```bash
cd /Users/raphaelchen/Desktop/csgo-bot
grep -A 8 "^arm         channel" runs/m3_study_v2/trust_provenance.txt
grep -n "H\*" runs/m3_study_v2/trust.txt
```
Expected: for each of the six (arm, channel) lines, the same `S(1)`, `S(5)`, `S(15)`, `S(45)` to three decimals and the same integer `H*_0.5`, `H*_0.75`, `H*_0.9`; and the same `H*_min`. A disagreement is a defect in one of the two -- the check's loop is eleven lines and is read first -- and nothing is recorded until it is found.

- [ ] **Step 10: Record the results honestly**

Append the section below to this plan as `## Task 7 results`, at the end of the plan, and fill every `…` from the named artefact -- `runs/m3_study_v2/trust.txt`, `trust_provenance.txt`, `trust.started`, `trust.exit`, `trust.log`, and the nine `trust_<arm>_seed<n>.json` (through `load_record`, never `json.loads`, or every NaN reads as `None`). Numbers are copied, not rounded further than the tool printed them. **Whatever the status is, it is the result**: `NOT_TESTABLE` and `UNRESOLVED_*` say what the instrument could not separate and are written as such; `NOT_SUPPORTED` with (ii) failing alone is the informative failure the spec names. The spec's Section 4 lists what this design does not claim, and the results must not claim it either.

````markdown

## Task 7 results

**Provenance.** The trust pass read the nine M3c checkpoints of `runs/m3_study_v2` (all
trained under `git_sha` `ca3e140772d6bc741d4d04312763afe3dd754166`, device `mps`, torch
`2.13.0`, per the study records) under one code state, `git_sha` = `…` (= `git rev-parse
HEAD` = `trust.head`, tree clean under `src/` and `scripts/`), on one device, `mps`, torch
`…`, in all nine `trust_<arm>_seed<n>.json`. Launched `…` (`trust.started`) under
`caffeinate -dimsu`, `PYTHONDONTWRITEBYTECODE=1`, `__pycache__` cleared; finished `…`;
wall `…` min; `trust.exit` = `…` (0 = both readings printed). Geometry: context 5, horizon
45, `split_seed` 0, 229 windows over 24 validation episodes, identical to the diagnostics
field by field (`runs/m3d_check_trust.py`: `OK: nine trust records, one git_sha == HEAD,
one device == mps, self-check exactly 0.0 on 9/9`; its 22 doctorings were run and caught on
`…`, the hand-recomputed Reading 2 agrees with `trust.txt` on every line). Family-wise
threshold `z_fam` = `…` (= `cluster_threshold(8, 24)`, expected 3.01), read against t(23).

**Self-check** (spec section 2.3), from `trust.txt`'s first table and the records'
`self_check`: the trust pass's window-mean curves against the diagnostic's, and its windows.

| cell | `reference_position` max\|Δ\| | `persistence_position` max\|Δ\| | `windows.total` | episode labels match | probe R² | band@45 | measurable | `never_moved` | `not_moved` @1 / @5 / @45 | ok |
|---|---|---|---|---|---|---|---|---|---|---|
| `pixel_ae`/s0 | … | … | … | … | … | … | … | … | … / … / … | … |
| `pixel_ae`/s1 | … | … | … | … | … | … | … | … | … / … / … | … |
| `pixel_ae`/s2 | … | … | … | … | … | … | … | … | … / … / … | … |
| `frozen_ssl`/s0 | … | … | … | … | … | … | … | … | … / … / … | … |
| `frozen_ssl`/s1 | … | … | … | … | … | … | … | … | … / … / … | … |
| `frozen_ssl`/s2 | … | … | … | … | … | … | … | … | … / … / … | … |
| `random_vit`/s0 | … | … | … | … | … | … | … | … | … / … / … | … |
| `random_vit`/s1 | … | … | … | … | … | … | … | … | … / … / … | … |
| `random_vit`/s2 | … | … | … | … | … | … | … | … | … / … / … | … |

Both deltas exactly `0.0` on `…`/9 cells; `measurable` True on `…`/9. Read: the nine records
measure `…` (the same windows, the same rollout, the same refit probe as the ladder --
or not, and which cell).

**Reading 2 -- the horizon M4 designs around** (spec section 3.3; no verdict attached).
`S(h)` = the fraction of (window, seed) draws with `h× > h`, over draws that moved
(`…` / `…` / `…` draws per arm of 687, the rest never moved within the horizon); the full
table at every h is in `trust.txt`, these are its columns at h = 1, 2, 3, 5, 10, 15, 30, 45.
`H*_q` = the largest h with `S(h) ≥ q`; q = 0.75 pre-registered.

| arm | channel | draws | S(1) | S(2) | S(3) | S(5) | S(10) | S(15) | S(30) | S(45) | H*_0.5 | **H*_0.75** | H*_0.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pixel_ae` | probe | … | … | … | … | … | … | … | … | … | … | **…** | … |
| `pixel_ae` | free | … | … | … | … | … | … | … | … | … | … | **…** | … |
| `frozen_ssl` | probe | … | … | … | … | … | … | … | … | … | … | **…** | … |
| `frozen_ssl` | free | … | … | … | … | … | … | … | … | … | … | **…** | … |
| `random_vit` | probe | … | … | … | … | … | … | … | … | … | … | **…** | … |
| `random_vit` | free | … | … | … | … | … | … | … | … | … | … | **…** | … |

**`H*_min` = `…`** (min over arms of the probe-free `H*_0.75`; the probe-based `H*_0.75`
beside it reads `…` / `…` / `…` for `pixel_ae` / `frozen_ssl` / `random_vit`). The two
channels order the arms `…` (the same way / differently: …). `H*_0.5` per arm through the
probe, `…` / `…` / `…`, is the old median reading; the mean-curve crossing of spec section 1
(median step 6, range 1-17) was a statement about curves, this is one about draws, and they
`…`. Read for M4: `…` (the number of open-loop steps over which every arm's imagination
has not yet strictly lost to persistence in at least three quarters of the moved draws, probe-free -- read it beside the unmoved fraction u(h) and the conditional survival S_c(h): a draw that has not moved by h survives vacuously).

**Reading 1 -- does the h=45 gate reward slow drift?** (spec section 3.2.) Pooled over
seeds per window, episode-clustered; the pooling notes read `clusters: 24 validation
episode(s) contribute windows; z_fam = cluster_threshold(8, 24) = …` (expected 3.01).

Per arm at h=45 (`Δ(45)` mean ± cluster SE; ratios as median/median with the 95 %
episode-bootstrap interval; `cos(45)` mean over moved windows):

| arm | `Δ(45)` ± SE | `R_probe(45)` [CI] | `R_free(45)` [CI] | `R_raw(45)` | `cos(45)` ± SE | moved / zero-`d̂` |
|---|---|---|---|---|---|---|
| `pixel_ae` | … | … | … | … | … | … / … |
| `frozen_ssl` | … | … | … | … | … | … / … |
| `random_vit` | … | … | … | … | … | … / … |

The eight clustered contrasts of the family, each read against `z_fam` = `…`:

| contrast | statistic | estimate | cluster SE | z | clears `z_fam`? |
|---|---|---|---|---|---|
| `frozen_ssl` − `random_vit` | `Δ(45)` | … | … | … | … |
| `frozen_ssl` − `pixel_ae` | `Δ(45)` | … | … | … | … |
| `pixel_ae` − `random_vit` | `Δ(45)` | … | … | … | … |
| `frozen_ssl` − `random_vit` | `cos(45)` | … | … | … | … |
| `frozen_ssl` − `random_vit` | held-out `c(45)`, fold A | … | … | … | … (holds iff z < −z_fam) |
| `frozen_ssl` − `random_vit` | held-out `c(45)`, fold B | … | … | … | … (holds iff z < −z_fam) |
| `frozen_ssl` − `random_vit` | `h×`, probe channel | … | … | … | … |
| `frozen_ssl` − `random_vit` | `h×`, free channel | … | … | … | … |

`pixel_ae` − `random_vit` on `h×`, for information: probe `…` (z `…`), free `…` (z `…`).
Scale correction at h=45 per arm, `α_A` / `α_B` (boundary = on a grid end, 0 or 2):
`pixel_ae` `…` / `…` (`…`), `frozen_ssl` `…` / `…` (`…`), `random_vit` `…` / `…` (`…`).

The conditions, each with the rule it was decided by:

| condition | rule | pooled | seeds holding (of 3) | holds |
|---|---|---|---|---|
| (i) least-moving arm = best-Δ arm | best-Δ: z > z_fam against *each* other arm; least-moving: argmin `R_probe`, same under `R_free` | best-Δ arm `…`; least-moving `…` (free: `…`) | … | … |
| (ii) treatment moves the right way | `cos(45)` contrast z > z_fam | z = `…` | … | … |
| (iii) magnitude out, ranking gone | both folds z < −z_fam; either > z_fam fails; boundary → unreadable | A z = `…`, B z = `…`; boundary `…` | … | … |
| probe control | both `h×` channels \|z\| > z_fam with opposite signs → fires | probe z `…`, free z `…` | — | fires: `…` |

Per-seed agreement (`per_seed_agreement`): (i) `…`/3, (ii) `…`/3, (iii) `…`/3. Per seed,
the same contrasts alone: seed 0 `…`; seed 1 `…`; seed 2 `…`.

**Sensitivity** (spec section 3.1, changes no verdict): with cells of selection R² < 0.1
excluded -- `pixel_ae`/s1 (R² 0.018) and no other -- the probe-based statistics read:
`Δ(45)` contrasts `…` / `…` / `…`, `R_probe(45)` `…` / `…` / `…`, `cos(45)` contrast z `…`,
`c(45)` fold A / B z `…` / `…`, `h×` probe z `…`. Which of them moved: `…`.

**Status: `…`.** Reason, as `trust.txt` printed it: `…`. Decided by the rule `…` (probe
control fires → UNRESOLVED_PROBE; (iii) unreadable → UNRESOLVED_ALPHA; (i) undecidable →
NOT_TESTABLE; (i) unresolved through the probe → UNRESOLVED_PROBE; all three hold pooled and
in ≥ 2 of 3 seeds → SUPPORTED; else NOT_SUPPORTED).

**The probe-free twins** (Step 7): least-moving arm through the probe `…`, probe-free `…`
(`…` agree); `H*_0.75` orders the arms `…` through the probe and `…` probe-free; the `h×`
contrast reads `…` through the probe and `…` probe-free (`…`).

**What this run establishes, and what it does not.** `…` -- stated against spec section 4's
four non-claims: it does not change the M3 gate or either study's verdict (`beats_persistence`
at h=45 stays the recorded 45-step stress test, and every one of the nine cells still fails
it); it does not say which arm is best for control (that is M4's live-deployment criterion,
and `H*_min` = `…` is one of its inputs, not a ranking); its probe-based numbers inherit a
probe R² of 0.018-0.383 and a floor larger than the displacement they read, so they are
comparable between arms and not trustworthy in absolute units, which is why `H*_min` is
probe-free and every probe-based reading above has its twin beside it; and everything here
is `my_way_home`'s 24 validation episodes at context 5 / horizon 45, one split, the M3c
checkpoints -- a different environment, split or horizon is a different measurement.
````

- [ ] **Step 11: Full suite -- the delta this task adds is 0**

No test file changes in this task. Run the suite once more on the tree the results were produced from, so the number recorded beside the results is the number for that tree:

```bash
cd /Users/raphaelchen/Desktop/csgo-bot
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
```
Expected: `N passed`, 0 warnings, exit 0, with N **0 more than Task 6's full-suite step left** -- record the measured number in the results section's provenance paragraph (`pytest: … passed`). A different N means the tree moved between Step 3 and now, and the `git_sha` in the records no longer names the tree the suite was run on: find what moved before committing.

- [ ] **Step 12: Commit**

Only the plan. `runs/` is gitignored and stays so -- the nine trust records, `trust.txt`, the logs and the acceptance check are not committed, which is why every number the section quotes is in the section. `scratchpad/m3d_doctored` was removed at the end of Step 8 (and is gitignored anyway) and must not appear.

```bash
cd /Users/raphaelchen/Desktop/csgo-bot
git status --porcelain            # expected: only the plan file modified (and M3b's untracked study.log at the root, untouched)
git add docs/superpowers/plans/2026-09-13-mb-fps-m3d-trust-horizon.md
git commit -m "docs: M3d trust horizon on runs/m3_study_v2 -- the self-check, the survival horizons and H*_min, and Reading 1's status

The two readings of the M3d spec, recorded from the nine trust records
and trust.txt of one run on mps under one code state. The status of
Reading 1 is recorded whatever it is; the spec pre-registers all five.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Exit criteria for this plan

- [ ] `pytest` fully green with zero warnings; the delta over 1278 is the sum of the seven tasks' stated deltas (record the measured number).
- [ ] `_diagnose(..., keep_trajectories=False)` is byte-identical to today's: every ladder test green, `diagnostic_*.json` unchanged (never rewritten).
- [ ] `trust_horizon.py` on `runs/m3_study_v2` exits 0 with the self-check table reading `0.0` / `0.0` / windows match on all nine rows.
- [ ] Nine `trust_<arm>_seed<n>.json` carrying every record key of the contract; `trust.txt` carrying the self-check table, the pooling notes, Reading 1 with each condition's statistic, z, `z_fam` and holds, the sensitivity line, and Reading 2's survival table with `H*_q` per arm and channel and `H*_min`.
- [ ] Reading 1's status is one of the five and its reason names the rule that decided it; Reading 2's `H*_min` is over the probe-free channel.
- [ ] `## Task 7 results` filled from `trust.txt` and the nine records, numbers copied not rounded, the closing paragraph stated against spec §4's non-claims. A NOT_TESTABLE or UNRESOLVED status is a result.
- [ ] Every mutation table row is caught, harness self-checked three ways.
- [ ] `runs/m3_study` and `runs/m3_study_v2`'s existing files are untouched (`ls -lt` shows nothing newer than the run's own outputs except `trust_*`, `trust.txt`, `trust.log`).

## Task 7 results

**Provenance.** The trust pass read the nine M3c checkpoints of `runs/m3_study_v2` (all
trained under `git_sha` `ca3e140772d6bc741d4d04312763afe3dd754166`, device `mps`, torch
`2.13.0`, per the study records) under one code state, `git_sha` =
`e5feeb9351de4e2090da29c17b110c3c56c6699c` (= `git rev-parse HEAD` = `trust.head`, tree clean
under `src/` and `scripts/`), on one device, `mps`, torch `2.13.0`, in all nine
`trust_<arm>_seed<n>.json`. This is the RE-RUN after the whole-branch review's fix wave
(`.superpowers/sdd/m3d-final-fix-list.md`), launched `2026-09-14T05:23:18Z` (`trust.started`)
under `caffeinate -dimsu`, `PYTHONDONTWRITEBYTECODE=1`, `__pycache__` cleared; finished
`2026-09-14T05:31:40Z` (mtime of `trust.exit`); wall `8 min 22 s`; `trust.exit` = `0` (0 = both
readings printed). The first run, at `19ed6a225ab8aaab2389df16cc92d10716f226d8` (launched
`2026-09-14T03:43:31Z`, wall 9 min 51 s, exit 0), produced a `trust.txt` whose every number and
status this re-run reproduced exactly (`diff` of the two: the re-run adds two pooling-note lines
and the conditional-survival block below, and drops the doubled `unreadable:` from two lines;
nothing else differs). `trust.txt` is byte-identical to `trust.log` from the `--- trust readings`
header on (the nine per-cell progress lines above it are in the log only). Geometry: context 5,
horizon 45, `split_seed` 0, 229 windows over 24 validation episodes, identical to the
diagnostics field by field (`runs/m3d_check_trust.py` on the re-run's records: `OK: nine trust
records, one git_sha == HEAD, one device == mps, self-check exactly 0.0 on 9/9` with HEAD
`e5feeb9`; its 22 doctorings were run and caught on the first run's records on
`2026-09-14T03:54:56Z`, every one on the named assertion and none surviving, and the
hand-recomputed Reading 2 in `trust_provenance.txt` -- the first run's, whose per-cell columns
are identical to the re-run's -- agrees with `trust.txt` on every line: the same `H*_0.5` /
`H*_0.75` / `H*_0.9` = 3 / 1 / 0 on all six (arm, channel) lines, the same `H*_min` = 1, and
`trust.txt`'s two-decimal `S(h)` the round of the check's three-decimal value at h = 1, 5, 15, 45
on every line). Family-wise threshold `z_fam` = `3.01` (= `cluster_threshold(8, 24)`, expected
3.01), read against t(23). Full suite on the first run's tree, before its launch and again after
its results were read: `pytest: 1428 passed`, 0 warnings, both times (Task 6 left 1428; this
task's delta is 0); on the re-run's tree, before its commit: `pytest: 1437 passed`, 0 warnings
(the fix wave's nine new tests).

**Self-check** (spec section 2.3), from `trust.txt`'s first table and the records'
`self_check`: the trust pass's window-mean curves against the diagnostic's, and its windows.
(`band@45` = the diagnostic's `persistence_position[-1] - floor_position[-1]`, from
`trust_provenance.txt`; `not_moved` from the records' `counts`.)

| cell | `reference_position` max\|Δ\| | `persistence_position` max\|Δ\| | `windows.total` | episode labels match | probe R² | band@45 | measurable | `never_moved` | `not_moved` @1 / @5 / @45 | ok |
|---|---|---|---|---|---|---|---|---|---|---|
| `pixel_ae`/s0 | 0.0e+00 | 0.0e+00 | 229 | True | 0.366 | 65.68 | True | 0 | 133 / 36 / 3 | True |
| `pixel_ae`/s1 | 0.0e+00 | 0.0e+00 | 229 | True | 0.018 | 19.67 | True | 0 | 133 / 36 / 3 | True |
| `pixel_ae`/s2 | 0.0e+00 | 0.0e+00 | 229 | True | 0.272 | 64.10 | True | 0 | 133 / 36 / 3 | True |
| `frozen_ssl`/s0 | 0.0e+00 | 0.0e+00 | 229 | True | 0.360 | 48.82 | True | 0 | 133 / 36 / 3 | True |
| `frozen_ssl`/s1 | 0.0e+00 | 0.0e+00 | 229 | True | 0.334 | 69.36 | True | 0 | 133 / 36 / 3 | True |
| `frozen_ssl`/s2 | 0.0e+00 | 0.0e+00 | 229 | True | 0.383 | 41.14 | True | 0 | 133 / 36 / 3 | True |
| `random_vit`/s0 | 0.0e+00 | 0.0e+00 | 229 | True | 0.297 | 44.62 | True | 0 | 133 / 36 / 3 | True |
| `random_vit`/s1 | 0.0e+00 | 0.0e+00 | 229 | True | 0.330 | 55.85 | True | 0 | 133 / 36 / 3 | True |
| `random_vit`/s2 | 0.0e+00 | 0.0e+00 | 229 | True | 0.249 | 53.76 | True | 0 | 133 / 36 / 3 | True |

Both deltas exactly `0.0` on `9`/9 cells; `measurable` True on `9`/9. Read: the nine records
measure what the ladder measured -- the same 229 windows with the same episode labels, the same
rollout (`evaluate_rollout` reproduced every study record's `curves.rssm_position` bitwise, or
the run would have exited 14 before the reference pass), and the same refit probe (the
diagnostic's `reference_position` and `persistence_position` reproduced bitwise on every cell).
The moved mask is the ground truth's and is the same on every cell of a seed's split: 133 of
229 windows have not moved 5 map units by h=1, 36 by h=5, 3 by h=45, and every window has
moved at some step within the horizon (`never_moved` = 0 on all nine), so every arm's survival
curve is over the full 3 x 229 = 687 draws.

**Reading 2 -- the horizon M4 designs around** (spec section 3.3; no verdict attached).
`S(h)` = the fraction of (window, seed) draws with `h× > h`, over draws that moved: 687 draws
per arm on every line (3 seeds x 229 windows; `never_moved` = 0 on every cell, so no draw is
excluded). The full table at every h is in `trust.txt`, these are its columns at h = 1, 2, 3, 5,
10, 15, 30, 45. `H*_q` = the largest h with `S(h) ≥ q`; q = 0.75 pre-registered. What `S(h)`
counts, by spec 2.2 / 3.3: `h×` is searched from the window's FIRST moved step h0 onward, so a
draw whose window has not yet moved 5 map units at h has `h× ≥ h0 > h` and survives h without
anything having been measured there -- `S(h)` is the fraction of moved draws not yet lost to
persistence by h, where a window not yet moved by h cannot yet have lost. The second table
below says how much of each `S(h)` that is.

| arm | channel | draws | S(1) | S(2) | S(3) | S(5) | S(10) | S(15) | S(30) | S(45) | H*_0.5 | **H*_0.75** | H*_0.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pixel_ae` | probe | 687 | 0.80 | 0.64 | 0.52 | 0.41 | 0.27 | 0.23 | 0.12 | 0.09 | 3 | **1** | 0 |
| `pixel_ae` | free | 687 | 0.82 | 0.64 | 0.54 | 0.40 | 0.25 | 0.17 | 0.07 | 0.04 | 3 | **1** | 0 |
| `frozen_ssl` | probe | 687 | 0.81 | 0.65 | 0.52 | 0.42 | 0.29 | 0.22 | 0.12 | 0.08 | 3 | **1** | 0 |
| `frozen_ssl` | free | 687 | 0.80 | 0.64 | 0.53 | 0.39 | 0.22 | 0.14 | 0.04 | 0.02 | 3 | **1** | 0 |
| `random_vit` | probe | 687 | 0.81 | 0.65 | 0.56 | 0.45 | 0.33 | 0.27 | 0.17 | 0.12 | 3 | **1** | 0 |
| `random_vit` | free | 687 | 0.81 | 0.64 | 0.53 | 0.39 | 0.24 | 0.18 | 0.08 | 0.05 | 3 | **1** | 0 |

**`H*_min` = `1`** (min over arms of the probe-free `H*_0.75`; the probe-based `H*_0.75`
beside it reads `1` / `1` / `1` for `pixel_ae` / `frozen_ssl` / `random_vit`). The two
channels order the arms the same way -- which is to say not at all: every `H*_q` is identical
on all six lines, 3 / 1 / 0, so neither channel orders the arms at any reported q; the curves
part only below q = 0.5, where `random_vit`'s probe channel stays highest, 0.12 against 0.09
and 0.08 at h = 45, and `frozen_ssl`'s free channel lowest, 0.02 against 0.04 and 0.05.
`H*_0.5` per arm through the probe, `3` / `3` / `3`, is the old median reading; the mean-curve
crossing of spec section 1 (median step 6, range 1-17) was a statement about curves, this is
one about draws, and they disagree: `H*_0.5` = 3 says more than half the draws are still ahead
of persistence at h = 3 and fewer than half at h = 4 -- `S(3)` 0.52-0.56, `S(4)` 0.44-0.49 on
every line -- so the pooled median draw crosses at step 4 on every arm and channel, and per cell
at 3.0-5.0, `hx_probe` / `hx_free` in `trust_provenance.txt`; against the mean-curve crossings
the M3c records reported per cell, 10·5·2 / 9·2·6 / 1·11·17, the median draw through the probe
sits below the mean curve's crossing on six of the nine cells and above it on three,
`frozen_ssl`/s2, `pixel_ae`/s1 and `random_vit`/s0 -- the two statistics do not track each other
cell by cell.

**Beside `S(h)`: the unmoved fraction and the conditional survival** (`trust.txt`'s second
Reading 2 table, added after the review; NOT pre-registered -- `S(h)`, `H*_q` and `H*_min` above
are the spec's and stand as printed). `u(h)` = the fraction of the same 687 draws whose window
has not yet moved at h (h0 > h), every one of which survives h vacuously; `S_c(h)` = (`S(h)` −
`u(h)`) / (1 − `u(h)`) = the survival among the draws whose window HAD moved by h, the ones on
which something was measured (n/a where `u(h)` = 1, so at h = 0); `H*c_q` = the largest h with
`S_c(h) ≥ q`. The moved mask is the truth's and identical on every cell, so `u(h)` is the same on
all six lines: 133 / 81 / 49 / 29 / 12 / 8 / 0 / 0 of the 229 windows have h0 > h at h = 1 / 2 /
3 / 5 / 10 / 15 / 30 / 45. (These are not the per-step `not_moved` counts of the self-check
table, 133 / 81 / 52 / 36 / 21 / 18 / 9 / 3: a window that moved and came back within 5 units at
h is unmoved AT h but was measured from its h0 on, and its crossing search started there.)

| arm | channel | draws | u(1) | u(2) | u(3) | u(5) | u(10) | u(15) | u(30) | u(45) |
|---|---|---|---|---|---|---|---|---|---|---|
| every line | probe and free | 687 | 0.58 | 0.35 | 0.21 | 0.13 | 0.05 | 0.03 | 0.00 | 0.00 |

| arm | channel | draws | S_c(1) | S_c(2) | S_c(3) | S_c(5) | S_c(10) | S_c(15) | S_c(30) | S_c(45) | H*c_0.5 | H*c_0.75 | H*c_0.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pixel_ae` | probe | 687 | 0.52 | 0.45 | 0.39 | 0.33 | 0.23 | 0.20 | 0.12 | 0.09 | 1 | 0 | 0 |
| `pixel_ae` | free | 687 | 0.56 | 0.44 | 0.41 | 0.32 | 0.20 | 0.14 | 0.07 | 0.04 | 1 | 0 | 0 |
| `frozen_ssl` | probe | 687 | 0.55 | 0.45 | 0.39 | 0.34 | 0.25 | 0.19 | 0.12 | 0.08 | 1 | 0 | 0 |
| `frozen_ssl` | free | 687 | 0.53 | 0.44 | 0.41 | 0.30 | 0.18 | 0.11 | 0.04 | 0.02 | 1 | 0 | 0 |
| `random_vit` | probe | 687 | 0.55 | 0.46 | 0.44 | 0.37 | 0.29 | 0.25 | 0.17 | 0.12 | 1 | 0 | 0 |
| `random_vit` | free | 687 | 0.54 | 0.45 | 0.40 | 0.30 | 0.19 | 0.15 | 0.08 | 0.05 | 1 | 0 | 0 |

So of `S(1)` = 0.80-0.82, 0.58 is windows that had not moved 5 units by h = 1; among the draws
whose window had, 0.52-0.56 were still ahead of persistence at h = 1 -- about half, not three
quarters -- and 0.44-0.46 at h = 2, 0.30-0.37 at h = 5. From h = 30 on `u(h)` = 0 and `S_c(h)` =
`S(h)`. The conditional `H*c_0.75` is 0 on every line (no h has `S_c(h) ≥ 0.75`), `H*c_0.5` = 1,
against the pre-registered `H*_0.75` = 1 and `H*_0.5` = 3. Read for M4: `H*_min` = `1` is, by
the spec's definition, the largest h at which at least three quarters of the moved draws have
not yet lost to persistence on every arm probe-free (`S(1)` 0.80-0.82, `S(2)` 0.64-0.65 on every
line) -- where a draw whose window has not yet moved by h cannot yet have lost, and 58 % of the
windows are in that state at h = 1; conditional on the window having moved, the per-step
reliability at h = 1 is about one half on every arm and channel, and by h = 45 between 2 % and
12 % of the draws are still ahead of persistence.

**Reading 1 -- does the h=45 gate reward slow drift?** (spec section 3.2.) Pooled over
seeds per window, episode-clustered; the pooling notes read `clusters: 24 validation
episode(s) contribute windows; z_fam = cluster_threshold(8, 24) = 3.01` (expected 3.01),
`probe-based pooling: 9 of 9 cells measurable (persistence-to-floor band at h=45 > 0);
excluded: none`, and `windows not moved at h=45 (excluded from every pooled series)`: 3 on
every cell.

Per arm at h=45 (`Δ(45)` mean ± cluster SE; ratios as median/median with the 95 %
episode-bootstrap interval; `cos(45)` mean over moved windows), the per-arm block of
`trust.txt` verbatim. On the ratios: `pool_ratio` normalises each cell by its own denominator
median before stacking, so `R_probe` / `R_free` are `median(num_c / med(den_c)) /
median(den_c / med(den_c))` over the stacked draws, not the spec-literal median of the raw
stacked numerators over that of the raw stacked denominators (spec 3.1, sentence added after
the run); the spec-literal stacked values from the same records are `R_probe` 0.993 / 0.873 /
0.827 and `R_free` 0.189 / 0.257 / 0.210 (`pixel_ae` / `frozen_ssl` / `random_vit`) against
the tool's 1.019 / 0.861 / 0.797 and 0.169 / 0.266 / 0.209 below, and the argmin is the same
arm under either estimand on both channels (`random_vit` through the probe, `pixel_ae`
probe-free), so `least-moving arm: none` stands either way.

| arm | `Δ(45)` ± SE | `R_probe(45)` [CI] | `R_free(45)` [CI] | `R_raw(45)` | `cos(45)` ± SE | moved / zero-`d̂` |
|---|---|---|---|---|---|---|
| `pixel_ae` | -44.313 ± 7.064 (n 226) | 1.019 [0.886, 1.164] | 0.169 [0.158, 0.190] | 0.959 | +0.006 ± 0.026 (n 226) | 678 / 0 |
| `frozen_ssl` | -38.294 ± 5.975 (n 226) | 0.861 [0.759, 0.985] | 0.266 [0.234, 0.297] | 0.914 | -0.032 ± 0.027 (n 226) | 678 / 0 |
| `random_vit` | -24.624 ± 5.445 (n 226) | 0.797 [0.713, 0.928] | 0.209 [0.183, 0.248] | 0.828 | -0.001 ± 0.025 (n 226) | 678 / 0 |

The eight clustered contrasts of the family, each read against `z_fam` = `3.01`. The z of every
row is printed in `trust.txt` (the pooled condition lines, the probe-control line, and the
sensitivity block); the estimate and SE of the two `Δ(45)` contrasts involving `pixel_ae` and
of the free-channel `h×` contrast are not printed there and were read from the nine records
through the script's own `pooled_inputs` (the same `paired_contrast` call that produced the
printed z; every printed z was reproduced). The `Δ(45)` rows are in the orientation the tool
prints, `pixel_ae − frozen_ssl` rather than the template's `frozen_ssl − pixel_ae`, so the sign
is the tool's and not a re-derivation. "Fold A" in the `c(45)` rows names the ROWS: fold A = the
even-label windows, scored with the alpha fit on fold B (α_B), fold B = the odd-label windows
scored with α_A -- while the `α_A` / `α_B` listed below name the FIT (α_A is fit on fold A's
rows and scores fold B's), as `trust.txt`'s `folds:` note now says. The two `h×` rows are the
paired per-draw contrast over 687 (window, seed) draws; with `never_moved` = 0 every window
carries all three seeds, so that contrast is numerically identical to the 229-window
seed-averaged one (same mean, same clustered SE, checked on the records: -1.968 / 1.154 and
-1.217 / 0.532 either way) -- the 687 is not a sharper ruler than the 229.

| contrast | statistic | estimate | cluster SE | z | clears `z_fam`? |
|---|---|---|---|---|---|
| `frozen_ssl` − `random_vit` | `Δ(45)` | -13.670 | 7.231 | -1.89 | no |
| `pixel_ae` − `frozen_ssl` | `Δ(45)` | -6.019 | 8.944 | -0.67 | no |
| `pixel_ae` − `random_vit` | `Δ(45)` | -19.689 | 5.546 | -3.55 | yes (\|z\| > 3.01; `random_vit`'s Δ is the larger, but `random_vit` − `frozen_ssl` does not clear, so no arm clears against *each* other arm) |
| `frozen_ssl` − `random_vit` | `cos(45)` | -0.031 | 0.037 | -0.83 | no |
| `frozen_ssl` − `random_vit` | held-out `c(45)`, fold A | -25.946 | 13.934 | -1.86 | no (holds iff z < −z_fam) |
| `frozen_ssl` − `random_vit` | held-out `c(45)`, fold B | -17.592 | 13.652 | -1.29 | no (holds iff z < −z_fam) |
| `frozen_ssl` − `random_vit` | `h×`, probe channel | -1.968 | 1.154 | -1.71 | no |
| `frozen_ssl` − `random_vit` | `h×`, free channel | -1.217 | 0.532 | -2.29 | no |

`pixel_ae` − `random_vit` on `h×`, for information: probe `-1.985, se 1.025` (z `-1.94`),
free `-0.157, se 0.659` (z `-0.24`); decides nothing.
Scale correction at h=45 per cell, `α_A` / `α_B` from the records' `scale` (boundary = on a
grid end, 0 or 2; the arm is flagged when any of its measurable seeds is on a boundary on
either fold, which is how `pooled_inputs` builds `alpha_boundary` -- the pooled reading of spec
3.2 (iii)'s "either arm's α is on the grid boundary" is "in any measurable seed of that arm",
a sentence added to the spec after the run so a re-run cannot be read either way; it is the rule
that decided this status, since exactly one seed of each of the treatment and the control hit
α = 0):
`pixel_ae` s0 `0.11` / `0.19` (`no`), s1 `0.00` / `0.00` (`boundary`), s2 `0.00` / `0.77`
(`boundary`) -- arm flagged; `frozen_ssl` s0 `0.11` / `0.25` (`no`), s1 `0.00` / `0.32`
(`boundary`), s2 `0.96` / `0.51` (`no`) -- arm flagged; `random_vit` s0 `0.49` / `0.00`
(`boundary`), s1 `0.55` / `0.20` (`no`), s2 `0.44` / `0.45` (`no`) -- arm flagged. Every
boundary hit is the grid's lower end, `α = 0`: on that fold and seed, the scale of the imagined
displacement that minimises the median error at h=45 is zero -- the corrected prediction is
persistence itself, which is why the spec calls the correction unreadable there.

The conditions, each with the rule it was decided by:

| condition | rule | pooled | seeds holding (of 3) | holds |
|---|---|---|---|---|
| (i) least-moving arm = best-Δ arm | best-Δ: z > z_fam against *each* other arm; least-moving: argmin `R_probe`, same under `R_free` | best-Δ arm `none` (Δ z: `pixel_ae`-`frozen_ssl` -0.67, `pixel_ae`-`random_vit` -3.55, `frozen_ssl`-`random_vit` -1.89); least-moving `none` (probe argmin `random_vit` 0.797; free argmin `pixel_ae` 0.169 -- the channels disagree) | 0 | undecided (undecidable at this precision) |
| (ii) treatment moves the right way | `cos(45)` contrast z > z_fam | z = `-0.83` (estimate -0.031, se 0.037, windows 226) | 0 | fails |
| (iii) magnitude out, ranking gone | both folds z < −z_fam; either > z_fam fails; boundary → unreadable | A z = `-1.86`, B z = `-1.29`; boundary `frozen_ssl, random_vit` (and `pixel_ae`) | 0 | undecided (unreadable) |
| probe control | both `h×` channels \|z\| > z_fam with opposite signs → fires | probe z `-1.71`, free z `-2.29` | — | fires: `no` (does not fire) |

Per-seed agreement (`per_seed_agreement`): (i) `0`/3, (ii) `0`/3, (iii) `0`/3. Per seed,
the same contrasts alone: seed 0 `(i) undecided (Δ z -1.81 / -4.60 / -1.65 in the order
above), (ii) fails (cos z -0.35, estimate -0.023), (iii) unreadable (random_vit on a boundary;
fold A z -1.77, fold B z -0.69)`; seed 1 `(i) undecided (Δ z +3.58 / +1.09 / -2.00), (ii)
fails (cos z -1.51, estimate -0.079), (iii) unreadable (frozen_ssl on a boundary; fold A z
+0.08, fold B z -0.81)`; seed 2 `(i) undecided (Δ z -1.27 / -1.20 / +0.32), (ii) fails (cos z
+0.19, estimate +0.010), (iii) undecided -- readable, neither fold clears −z_fam (fold A z
-2.73, fold B z -1.70)`. Note on seed 1: `pixel_ae` − `frozen_ssl` clears at z +3.58 but
`pixel_ae` − `random_vit` (z +1.09) does not, so seed 1 has no best-Δ arm either; and its sign
is opposite to the pooled contrast's (-0.67) -- the three seeds do not agree on the order of
`pixel_ae` and `frozen_ssl` at h=45.

**Sensitivity** (spec section 3.1, changes no verdict): with cells of selection R² < 0.1
excluded -- `pixel_ae`/s1 (R² 0.018) and no other -- the probe-based statistics read:
`Δ(45)` contrasts `pixel_ae − frozen_ssl -22.413, se 11.704, z -1.92` / `pixel_ae − random_vit
-36.083, se 8.334, z -4.33` / `frozen_ssl − random_vit -13.670, se 7.231, z -1.89`,
`R_probe(45)` `pixel_ae 1.149 [1.008, 1.373]` / `frozen_ssl 0.861 [0.759, 0.985]` /
`random_vit 0.797 [0.713, 0.928]`, `cos(45)` contrast z `-0.83`, `c(45)` fold A / B z `-1.86` /
`-1.29`, `h×` probe z `-1.71`. Which of them moved: `only the pixel_ae statistics -- its two
Δ(45) contrasts (from -6.019 / -19.689 to -22.413 / -36.083; the pixel_ae − random_vit z from
-3.55 to -4.33) and its R_probe(45) (from 1.019 to 1.149, the interval now excluding 1); every
frozen_ssl − random_vit statistic is identical to the main pooling, as it must be since neither
of those arms lost a cell. No condition's outcome changes: (i) still has no best-Δ arm (the
frozen_ssl − random_vit contrast still does not clear), (ii) still fails, (iii) is still
unreadable, the probe control still does not fire.`

**Status: `UNRESOLVED_ALPHA`.** Reason, as the re-run's `trust.txt` printed it: `(iii)
unreadable: alpha on the grid boundary at h=45 for frozen_ssl, random_vit (held-out c(45)
frozen_ssl-random_vit: fold A z -1.86, fold B z -1.29, bar ±3.01)` (the `verdict:` line reads
`Reading 1 unresolved (alpha on the grid boundary) -- decided by:` and the same text; the first
run's text read `(iii) unreadable: unreadable: ...`, `reading_one` prefixing a detail that
already began with the word -- a formatting slip in `trust_readings.py`, not two findings,
fixed in the wave and gone from the re-run). Decided by the rule `(iii) unreadable →
UNRESOLVED_ALPHA` -- the second rule in the precedence, reached because the first (the probe
control) did not fire (probe control fires → UNRESOLVED_PROBE; (iii) unreadable →
UNRESOLVED_ALPHA; (i) undecidable → NOT_TESTABLE; (i) unresolved through the probe →
UNRESOLVED_PROBE; all three hold pooled and in ≥ 2 of 3 seeds → SUPPORTED; else
NOT_SUPPORTED). For the record, what the later rules would have said had (iii) been readable:
(i) is undecidable (no best-Δ arm) and, separately, unresolved through the probe (the
least-moving arm differs between channels), so the status would have been `NOT_TESTABLE`; and
(ii) fails pooled and in every seed. Two of spec 3.2's statuses therefore apply to this run --
`NOT_TESTABLE` via (i) and `UNRESOLVED_ALPHA` via (iii) -- and the spec does not rank them; the
plan header does (its interface contract for `reading_one` fixes the order quoted above, probe
control first, then (iii) unreadable, then (i) undecidable, implemented as such in
`trust_readings.reading_one`), and that ranking is what chose `UNRESOLVED_ALPHA` over
`NOT_TESTABLE`. Neither is `SUPPORTED`, and the choice is the plan's, not the spec's.

**The probe-free twins** (Step 7): least-moving arm through the probe `random_vit`
(`R_probe(45)` 0.797 against 0.861 and 1.019), probe-free `pixel_ae` (`R_free(45)` 0.169
against 0.266 and 0.209) (`do not` agree -- which is why `least-moving arm: none`; the probe-free
ratios say `pixel_ae`'s imagined displacement is the smallest fraction of the true one at h=45
and `frozen_ssl`'s the largest, the probe-based ratios say `random_vit`'s imagined position
displacement is the smallest fraction of the probe's own and `pixel_ae`'s the largest; the
sensitivity line moves `pixel_ae`'s `R_probe` further up, to 1.149, so the disagreement is not
the R² 0.018 cell's alone); `H*_0.75` orders the arms `not at all (1 / 1 / 1)` through the
probe and `not at all (1 / 1 / 1)` probe-free; the `h×` contrast reads `-1.968 (z -1.71)`
through the probe and `-1.217 (z -2.29)` probe-free (`agree: the same sign, neither clearing
z_fam` -- `frozen_ssl`'s draws cross persistence one to two steps earlier than `random_vit`'s
on both channels, short of the family bar).

**What this run establishes, and what it does not.** `Three things. (1) The instrument is the
ladder's: nine records at max|Δ| exactly 0.0 on both curves, the same windows, the same
rollout, the same refit probe, one code state, one device. (2) Reading 2, the number M4 designs
around: H*_min = 1 -- h = 1 is the last step at which at least three quarters of the moved
draws have not yet lost to persistence on every arm (S(1) 0.80-0.82, S(2) 0.64-0.65, S(5)
0.39-0.45, S(45) 0.02-0.12), where a draw whose window has not yet moved by h cannot yet have
lost and 58 % of the windows are in that state at h = 1; conditional on the window having moved,
about half the draws are still ahead of persistence at h = 1 (S_c(1) 0.52-0.56), 0.44-0.46 at
h = 2 and 0.30-0.37 at h = 5, on every arm and channel; the pooled median draw crosses at step 4
on every arm and channel, and the three arms are indistinguishable at every reported q. (3) Reading 1 is unresolved: the
question whether the h=45 gate rewards slow drift could not be put to these records, because
the cross-fitted scale correction hit α = 0 on at least one fold in one seed of each of the
treatment and the control (and of pixel_ae), so condition (iii) is unreadable by the spec's own
rule. What the other rules said, for information only: no arm's Δ(45) is ordered against both
others at z_fam (the only clearing contrast is pixel_ae − random_vit, z -3.55, in the direction
of random_vit losing least), the least-moving arm differs between the probe-based and the
probe-free ratio, the treatment's cos(45) is not better than the control's (z -0.83; every
arm's cos(45) is within 1.2 SE of 0: +0.006 ± 0.026, -0.032 ± 0.027, -0.001 ± 0.025), the probe control does not fire,
and no condition holds in any seed. An α of 0 at h=45 is itself consistent with a cos(45) of 0:
at 45 steps the imagined displacement's direction is uncorrelated with the true one on every
arm, and no rescaling of it improves on persistence. Whether random_vit's smallest loss at h=45
comes from drifting slowly is therefore not answered here -- the reading is UNRESOLVED_ALPHA,
not NOT_SUPPORTED and not SUPPORTED.` -- stated against spec section 4's four non-claims: it
does not change the M3 gate or either study's verdict (`beats_persistence` at h=45 stays the
recorded 45-step stress test, and every one of the nine cells still fails it -- `Δ(45)` is
-24.6 to -44.3 map units per arm here, negative on all three); it does not say which arm is
best for control (that is M4's live-deployment criterion, and `H*_min` = `1` is one of its
inputs, not a ranking -- the three arms tie at every reported q); its probe-based numbers
inherit a probe R² of 0.018-0.383 and a floor larger than the displacement they read, so they
are comparable between arms and not trustworthy in absolute units, which is why `H*_min` is
probe-free and every probe-based reading above has its twin beside it (and one pair of twins,
the least-moving arm, disagrees); and everything here is `my_way_home`'s 24 validation episodes
at context 5 / horizon 45, one split, the M3c checkpoints -- a different environment, split or
horizon is a different measurement.
