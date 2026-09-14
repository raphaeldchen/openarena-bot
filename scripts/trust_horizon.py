"""The trust horizon over the M3c cells: load, check, one reference pass, one record per cell, then pool and print both readings.

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
After the per-cell loop, `main` pools the LIVE records of that run (the dict
it just built, never a stale file on disk) through `pooling.py`, decides the
two readings through `trust_readings.py`, prints them and writes `trust.txt`
-- the pooling glue and the printed tables below `write_trust_record`.

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
    what the ladder measured -- and the cell's record is NOT written, and the
    run stops: the pooling after the loop reads only the records this run
    built, and it needs every requested cell.

11, 12 and 14 carry `diagnose_dynamics.py`'s meanings ON PURPOSE -- a wrapper
reading the status learns the same thing from either tool -- and 30 is new,
distinct from every other tool's (1-23) and from argparse's own 2. A
mislabelled checkpoint (that script's 16) and an unsupported device (its 17)
have no status here and surface as the library's own exceptions.
"""

import argparse
import dataclasses
import importlib.util
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import mbfps.eval.pooling as pooling
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
    survival,
    trust_horizon,
)
from mbfps.eval.trust_readings import (
    ARMS_ORDER,
    CONTROL,
    FAMILY,
    Q_REPORTED,
    TREATMENT,
    Contrast,
    Ratio,
    ReadingOneInputs,
    format_reading_one,
    format_reading_two,
    reading_one,
    reading_two,
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
    # Fold A = the even-label windows, scored with the alpha fit on fold B
    # (alpha_B); fold B = the odd-label windows, scored with the alpha fit on
    # fold A (alpha_A) -- `scale_corrected_error`'s own assignment, where
    # `held_out` scores every row with the OTHER fold's alpha. So "fold A"
    # names the ROWS of a contrast here, while `alpha_a` / `score_a` on the
    # record name the FIT (alpha_A is fit on A's rows and scores B's).
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
        # The treatment pair is the probe control's own estimator, one code
        # path with the informational pixel_ae pair.
        crossing_contrast_probe=_crossing_contrast(records, TREATMENT, "probe"),
        crossing_contrast_free=_crossing_contrast(records, TREATMENT, "free"),
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


def _crossing_contrast(records: dict, arm: str, channel: str) -> Contrast:
    """`arm - CONTROL` on h_x per (window, seed) draw, paired per draw with
    the seeds stacked (spec 2.2: never seed-averaged), over the seeds both
    arms carry -- both measurable, for the probe channel (spec 3.1). ONE
    estimator for two uses: with `arm = TREATMENT` it is the probe control's
    own contrast (`ReadingOneInputs.crossing_contrast_probe` / `_free`, which
    `probe_control` decides on); with `arm = "pixel_ae"` it is printed for
    information -- spec 3.2: "`pixel_ae`'s pair is printed for information"
    -- and decided on by nothing."""
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


def _reading_two_cells(records: dict, arm: str, channel: str) -> list[dict]:
    """The cells one arm's Reading 2 curve is built from, in seed order: the
    probe channel takes the MEASURABLE cells only -- spec 3.1's rule for
    every probe-based pooling, applied here as `probe_cells` and
    `_crossing_contrast` apply it -- and the free channel every cell."""
    seeds = sorted({seed for _, seed in records})
    cells = [records[(arm, seed)] for seed in seeds if (arm, seed) in records]
    if channel == "probe":
        cells = [cell for cell in cells if _measurable(cell)]
    return cells


def _crossing_draws(records: dict, arm: str, channel: str) -> np.ndarray:
    """One arm's per-(window, seed) crossing draws in one channel, seeds
    stacked in seed order, over `_reading_two_cells`; empty when the arm has
    no cell in the channel (every seed unmeasurable, for the probe)."""
    cells = _reading_two_cells(records, arm, channel)
    if not cells:
        return np.zeros(0)
    return np.concatenate([np.asarray(cell["crossing"][channel], dtype=float) for cell in cells])


def survival_by_arm(records: dict) -> dict:
    """`(arm, channel) -> S(h)` for Reading 2: `trust.survival` over the
    (window, seed) draws of `_crossing_draws` -- every cell in the free
    channel, the measurable cells in the probe channel -- never-moved draws
    (NaN) excluded by `survival` itself. The horizon is the records'."""
    arms = sorted({arm for arm, _ in records}, key=ARMS_ORDER.index)
    horizon = int(next(iter(records.values()))["horizon"])
    return {
        (arm, channel): survival(_crossing_draws(records, arm, channel), horizon)
        for arm in arms
        for channel in ("probe", "free")
    }


# --- Beside S(h): the unmoved fraction and the conditional survival ------------
# Spec 3.3's S(h) counts every moved draw with h_x > h, and `crossing_step`
# starts its search at h0, the window's FIRST moved step: a draw whose window
# has not yet moved at h has h_x >= h0 > h and survives h with nothing having
# been measured there. On the shipped split 133 of 229 windows are unmoved at
# h = 1, so S(1) is at least 0.58 before any model is read. The two series
# below say how much of S(h) that is. S(h), H*_q and H*_min stay exactly as
# pre-registered; these are printed beside them and decide nothing.


def _first_moved(record: dict) -> np.ndarray:
    """h0 per window, 1-based -- the first step at which the window has moved,
    `crossing_step`'s own search start -- NaN where it never moves. Read off
    the per-step moved mask (`_moved`, column by column), never off
    `counts.not_moved`, which counts windows per step and cannot say which."""
    moved = np.stack([_moved(record, hi) for hi in range(int(record["horizon"]))], axis=1)
    return np.where(moved.any(axis=1), moved.argmax(axis=1) + 1.0, np.nan)


def _first_moved_draws(records: dict, arm: str, channel: str) -> np.ndarray:
    """h0 per (window, seed) draw, stacked exactly as `_crossing_draws` stacks
    the crossings, so the two align draw by draw."""
    cells = _reading_two_cells(records, arm, channel)
    if not cells:
        return np.zeros(0)
    return np.concatenate([_first_moved(cell) for cell in cells])


def unmoved_fraction(crossings: np.ndarray, first_moved: np.ndarray, horizon: int) -> np.ndarray:
    """`u(h)` for h = 0..horizon: among the draws `S(h)` counts (the finite
    crossings), the fraction whose window has not yet moved at h -- `h0 > h`
    -- and so survives h vacuously. `(horizon + 1,)` float, all NaN when no
    crossing is finite; `u(0) == 1.0` whenever any is (no window has moved at
    h = 0). Refuses a finite crossing whose window never moved: `crossing_step`
    cannot produce one, so the record's `crossing` and its moved mask disagree."""
    crossings = np.asarray(crossings, dtype=np.float64)
    first_moved = np.asarray(first_moved, dtype=np.float64)
    if crossings.shape != first_moved.shape or crossings.ndim != 1:
        raise ValueError(
            f"crossings and first_moved must share one (n,) shape, got "
            f"{crossings.shape} and {first_moved.shape}"
        )
    finite = np.isfinite(crossings)
    h0 = first_moved[finite]
    if h0.size == 0:
        return np.full(horizon + 1, np.nan)
    if not np.isfinite(h0).all():
        raise ValueError(
            f"{int((~np.isfinite(h0)).sum())} draw(s) have a finite crossing but a window that "
            "never moved: the record's crossing and its moved mask disagree"
        )
    steps = np.arange(horizon + 1)
    return (h0[None, :] > steps[:, None]).mean(axis=1)


def conditional_survival(surv: np.ndarray, unmoved: np.ndarray) -> np.ndarray:
    """`S_c(h) = (S(h) - u(h)) / (1 - u(h))` where `u(h) < 1`, NaN otherwise:
    the survival among the draws whose window HAS moved by h -- the ones on
    which something was measured -- since every draw counted by `u(h)` is in
    `S(h)` too. `S_c(0)` is always NaN."""
    surv = np.asarray(surv, dtype=np.float64)
    unmoved = np.asarray(unmoved, dtype=np.float64)
    if surv.shape != unmoved.shape:
        raise ValueError(f"S(h) {surv.shape} and u(h) {unmoved.shape} must share a shape")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(unmoved < 1.0, (surv - unmoved) / (1.0 - unmoved), np.nan)


@dataclasses.dataclass(frozen=True)
class Conditional:
    """One (arm, channel)'s series beside its `S(h)`: the finite-crossing draw
    count (S's denominator), `u(h)` and `S_c(h)`, h = 0..H."""

    draws: int
    unmoved: np.ndarray
    survival: np.ndarray


def conditional_by_arm(records: dict, curves: dict) -> dict:
    """`(arm, channel) -> Conditional` for every key of `curves` (the output
    of `survival_by_arm` on the same records), from the same draws in the
    same order."""
    horizon = int(next(iter(records.values()))["horizon"])
    out = {}
    for (arm, channel), surv in curves.items():
        crossings = _crossing_draws(records, arm, channel)
        unmoved = unmoved_fraction(crossings, _first_moved_draws(records, arm, channel), horizon)
        out[(arm, channel)] = Conditional(
            draws=int(np.isfinite(crossings).sum()),
            unmoved=unmoved,
            survival=conditional_survival(surv, unmoved),
        )
    return out


def write_readings(out_dir: Path, text: str) -> Path:
    """`runs/<out>/trust.txt`: the readings exactly as printed."""
    path = Path(out_dir) / "trust.txt"
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# The printed readings, in the ladder's style: the self-check table, the
# pooling notes, the per-arm block, Reading 1 (summary line, then Task 5's
# block under its own header) with pixel_ae's h_x pair for information, the
# sensitivity block, Reading 2 (Task 5's block under its own header) with
# the unmoved-fraction / conditional-survival block beside it.
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
    lines.append(
        "Reading 2's probe channel (S(h), H*_q through the probe) pools the same measurable cells; "
        f"excluded from it: {', '.join(excluded) or 'none'}; the free channel pools every cell"
    )
    lines.append(
        "folds: fold A = the even-label windows, scored with the alpha fit on fold B (alpha_B); "
        "fold B = the odd-label windows, scored with the alpha fit on fold A (alpha_A)"
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


def _conditional_table(conditional: dict, qs=Q_REPORTED) -> str:
    """The block printed under Reading 2's table: per arm and channel (in
    Reading 2's order), the draw count, then `u(h)` on one row and `S_c(h)`
    on the next, each at every h like `S(h)` above them, with the
    conditional `H*c_q` -- the largest h with `S_c(h) >= q`, `trust_horizon`
    on `S_c` -- beside `S_c(h)` under labels of its own. NaN prints `n/a`
    (`S_c(0)`, and every h where no counted window has moved)."""
    qs = sorted(qs)
    rows = [
        "--- Reading 2, beside S(h) (not pre-registered; S(h), H*_q and H*_min above are the spec's): "
        "u(h) = the fraction of the same moved draws whose window has not yet moved at h (h0 > h: "
        "h_x >= h0, so they survive h with nothing measured); S_c(h) = (S(h) - u(h)) / (1 - u(h)) = "
        "the survival among the draws whose window has moved by h, n/a where u(h) = 1; "
        "H*c_q = largest h with S_c(h) >= q ---",
        f"{'arm':<12}{'channel':<9}{'draws':>6}  {'series':<7}"
        + "".join(f"{f'H*c_{q:g}':>9}" for q in qs) + "   value, h = 0..H",
    ]
    horizons = "".join(f"{'':>9}" for _ in qs)
    for arm in ARMS_ORDER:
        for channel in ("probe", "free"):
            if (arm, channel) not in conditional:
                continue
            c = conditional[(arm, channel)]
            unmoved = " ".join(_num(v, ".2f") for v in c.unmoved)
            cond = " ".join(_num(v, ".2f") for v in c.survival)
            cond_horizons = "".join(f"{trust_horizon(c.survival, q):>9d}" for q in qs)
            rows.append(f"{arm:<12}{channel:<9}{c.draws:>6}  {'u(h)':<7}{horizons}   {unmoved}")
            rows.append(f"{arm:<12}{channel:<9}{'':>6}  {'S_c(h)':<7}{cond_horizons}   {cond}")
    return "\n".join(rows)


def _readings_text(records: dict, *, h: int) -> str:
    """The whole readings block, in this order: the run header; the
    self-check table (always); then, when every arm is present, the pooling
    notes, the per-arm block, the `reading 1:` summary line followed by
    Task 5's Reading 1 block under its own header, pixel_ae's h_x pair for
    information, the sensitivity block, Task 5's Reading 2 block under its
    own header, and the unmoved-fraction / conditional-survival block
    beside it. Both readings need all three arms -- Reading 1 pools them
    and `reading_two` refuses a missing (arm, channel) by name -- so a run
    over fewer prints `not computed` under BOTH headers and still exits 0
    with the self-check table on disk."""
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
            c = _crossing_contrast(records, "pixel_ae", channel)
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
        curves = survival_by_arm(records)
        lines.append(format_reading_two(reading_two(curves)))
        lines.append("")
        lines.append(_conditional_table(conditional_by_arm(records, curves)))
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
    # Pooling and the two readings -- only after every cell's checks held
    # and every record is on disk, so trust.txt never describes cells a
    # later line disowns. `records` is keyed (arm, seed); `horizon` is the
    # run's resolved horizon, so the reading is at the final step (45 on
    # the shipped records).
    text = _readings_text(records, h=horizon)
    print(text, end="")
    write_readings(args.out, text)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
