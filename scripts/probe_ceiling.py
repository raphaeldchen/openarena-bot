"""M3o: what can the readout reach, independently of the model? -- measure and read.

M3n returned NO_MAJORITY because a one-step signal of 0.576-11.824 map units sat
under readout errors of 131.88-259.06. `mbfps.eval.ceiling` is the pure half: the
estimators, the ridge grid, the scoring layer, the reading and its table. This
script is the impure half: the rows, the checkpoints, the records.

THE ROWS ARE FRAMES, AND ARE MODEL-FREE. The 24 validation episodes are divided
three ways BY EPISODE (`episode_split` twice, at `SPLIT_SEED`: 8 / 8 / 8 on the
shipped split) into fit / select / score. A row is a frame `t` of one episode with
`SEQUENCE_FRAMES - 1` frames of real history behind it, so the first
`SEQUENCE_FRAMES - 1` frames of every episode are not rows. One stride, recorded as
`frame_stride`, is chosen so that no split exceeds the budget (`ceiling.FRAME_BUDGET`
or `--frames`); the rows each split actually holds are recorded beside it as
`rows_per_split`, because the budget does NOT bind at the shipped split (n is about
4,000 per split, against a cap of 12,000) and a change that lets it bind turns a
17-minute run into a 7.7-hour one without any other symptom.

THREE FEATURE SETS, ONE ROW SET. `pixels` is the raw frame, 112 x 112 x 3 = 37,632
values, from `episode.obs`. `embedding` is the model's ENCODER output for that frame
(`model.encoder` on `rollout.source_for`'s array, the same call
`probe.gather_probe_data` makes). `sequence` is the encoder outputs of frames
`t - SEQUENCE_FRAMES + 1 .. t` concatenated, oldest first. They describe the same
rows in the same order, which is what lets `ceiling.difference_interval` subtract
the readouts of different feature sets: it resamples ONE set of score targets and
ONE set of episode labels.

`SEQUENCE_FRAMES` is the M3o design's open choice (the spec writes `t-k..t` and
names no `k`). It is the rollout's own context length, the history the shipped probe
is ever handed. The sequence readouts are per MODEL (spec section 5: A and B are one
number for the study, C, D, E and F nine), so the window is of encoder embeddings;
a window of raw frames would be 188,160 values a row.

THE RAW-PIXEL READOUTS USE NO MODEL. They are scored once per run (`prepare_shared`)
and the SAME `Readout` objects go into every cell's record, so the nine records carry
bitwise identical `pixels_linear` and `pixels_rbf` -- which the read phase asserts, as
a free control that no model leaked into a model-free row.

PER CELL, THE RECORD CARRIES (every constant it was taken at is stored, so none is a
literal that survives the tests):

  applied      the shipped rollout probe's r2 and map-unit errors on the
               distribution `evaluate_rollout` applies it to: its own output at
               each imagined step (`diagnostics.reference_trajectories`'
               `positions`) against the truth (`true_positions`). Spec section 3.1.
               `mean_error` is carried beside the brief's `median_error` because
               M3n's 131.88-259.06 is a MEAN.
  readouts     eight `Readout`s, each as r2, mean_error, median_error, capacity and
               jitter. The eighth, `embedding_linear_shipped`, is `fit_probe` over
               `probe.RIDGES`: the reconciliation anchor.
  shipped_grid_r2   that eighth readout's r2. Exit 49 and `harness_r2_matches_study`
               read THIS and never `embedding_linear`, whose finer grid is expected
               up to ~0.02 above the shipped one.
  null_floor   `ceiling.null_floor(n, d)` at THIS cell's shape: n the fit split's
               rows, d the embedding width. UNCLAMPED (the clamp is
               `probe_gain_over_floor`'s, taken once, after the mean).
  gains        `D - C - max(null_floor, 0)` through `probe_gain_over_floor`,
               `B - D` and `F - B` through `difference_interval`, each as
               (point, ci_low, ci_high), drawn at `cell_bootstrap_seed(...)`.
  controls     four numbers with known answers, MEASURED and never asserted: the
               harness against the study, the widest RBF rung against linear, and
               the zero-information baseline recomputed on the shipped definition.

`capacity` IS `inf` ON EVERY CELL WHERE NOTHING NONLINEAR HELPS, and `inf` is not
JSON. The kernel ladder's linear rung has `bandwidth = inf`; `json.dumps` writes the
bare token `Infinity`, which no strict reader takes, and `study.write_record` would
null it and file it under `nonfinite`. It is written as the STRING `"inf"`, before
`write_record` sees it, so `nonfinite` stays `{}` and the file is strict JSON. The
read phase refuses a record with a bare `Infinity` or `NaN` anywhere in its text, and
a capacity that is neither a finite number nor exactly `"inf"`.

THE STUDY NUMBER THE HARNESS IS RECONCILED AGAINST IS NOT `probe.embedding_selection_r2`.
The spec's 0.320-0.345 is `filtering.criterion_4.embedding_r2`, the probe on REAL
ENCODER embeddings (0.3195-0.3451 across the nine M3c records). The field the spec
and the brief name, `probe.embedding_selection_r2`, belongs to the ROLLOUT probe, fit
on the model's PREDICTED embeddings, and spans 0.018-0.383 across the same nine cells
(pixel_ae seed 1 reads 0.018). Gating `C_shipped` against that would exit 49 on a
correct run. Both are recorded (`study_embedding_r2`, gated; `study_embedding_selection_r2`,
not), and `HARNESS_R2_TOLERANCE` says how far apart they may sit.

NOTHING IN THE MEASURE PHASE DECIDES ANYTHING. A control that missed its known answer
is recorded and named in the cell's printed line, and the run carries on: the records
are the artefact. The READ phase refuses it, by name or by number (49).

THE READ PHASE (`--phase read`) pools the records, from the records alone and with no
GPU. It refuses what it cannot read, in three different ways kept apart on purpose:

  * BY NAME, with no number (`SystemExit` carrying the message, status 1): a plan
    narrower than `ARMS_REQUIRED` arms, records that disagree on any
    `_PROTOCOL_FIELDS` entry, a record filed under another cell's name, a bare
    `Infinity`/`NaN` in a record, a capacity that is not a number or `"inf"`, raw-pixel
    readouts that differ between cells, a harness flag that contradicts the numbers
    beside it, and `--phase all` over fewer than `SEEDS_MINIMUM` seeds (refused before
    any cell is measured).
  * BY NUMBER: 49 when any cell's `C_shipped` fails to reproduce the study (a
    statement about the HARNESS, raised before any gain is read), 50 when no verdict
    held a strict majority, 11 for a missing cell. 50 is a FINDING, so its table is
    printed and written. The three model verdicts exit 0.
  * BY TRACEBACK: `reading_ceiling`'s `ValueError` for an arm short of
    `SEEDS_MINIMUM` seeds.

`ceiling.txt` is written only for a reading, and is the same bytes stdout carries. A
read that takes no reading also removes the one an earlier read of the same `--out`
left.

MAP UNITS ARE COMPARED HERE, and nowhere in `ceiling.py`. The reading is in r2; the
section this script appends to it sets each readout's `mean_error` against the
zero-information MEAN and its `median_error` against `ZERO_INFORMATION_MEDIAN`, each
ratio naming its statistic, and sets M3n's 131.88-259.06 -- a mean -- only against
`mean_error`.

EXIT NUMBERS. 49 and 50 are returned by the read phase and in no other tool's range
(47/48 M3n, 45/46 M3m, 43/44 M3l, 41/42 M3k, 39/40 M3j, 38 M3i, run_study 1/3-6/23,
report_study 7-10, diagnose 11-17, pool 18-22, trust 30, split_gap 31, ladder 32-33,
stages 34, sharper_latent 35-37, argparse 2, a traceback 1). 0 / 11 / 12 / 14 are
`trust_horizon.py`'s, on purpose. This script RETURNS its statuses, as
`motion_headroom.py` does, and an exception from a cell propagates.

LOADING IS `trust_horizon.py`'s: `load_cell` and `prepare_cell` are imported by path
through `_sibling`, so a cell is refused for the same reasons in the same words as
every other diagnostic, and the record it reproduces is shown to reproduce before a
pass of this script's own is paid for.
"""

import argparse
import importlib.util
import json
import math
import sys
import types
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import load_episode
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval import ceiling
from mbfps.eval.aggregate import SEEDS
from mbfps.eval.diagnostics import reference_trajectories
from mbfps.eval.probe import _mean_r2, position_error, probe_targets
from mbfps.eval.rollout import source_for
from mbfps.eval.study import SPLIT_SEED, git_sha, load_record, write_record
from mbfps.utils.config import ARMS
from mbfps.utils.device import get_device


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_probe_ceiling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_trust = _sibling("trust_horizon")
CellMissing = _trust.CellMissing
load_cell = _trust.load_cell
prepare_cell = _trust.prepare_cell

EXIT_OK = _trust.EXIT_OK
EXIT_NO_CHECKPOINTS = _trust.EXIT_NO_CHECKPOINTS
EXIT_SPLIT_MISMATCH = _trust.EXIT_SPLIT_MISMATCH
EXIT_RECORD_MISMATCH = _trust.EXIT_RECORD_MISMATCH

EXIT_HARNESS_DISAGREES: int = 49
"""RETURNED BY THE READ PHASE when a cell's `C_shipped` (`shipped_grid_r2`) fails to
reproduce the study's own embedding probe.

M3o is then measuring something other than what the study measured, and no verdict
about the model is available. This is a statement about the HARNESS, not about the
representations, and it is raised before any gain is read."""

EXIT_NO_MAJORITY: int = 50
"""RETURNED BY THE READ PHASE when no verdict held a strict majority."""

READ_EXITS: dict[str, int] = {ceiling.NO_MAJORITY: EXIT_NO_MAJORITY}
"""The three model verdicts exit 0: each is an answer. 49 is returned before the tally,
from the controls, so it is not keyed here."""

PHASES: tuple[str, ...] = ("all", "measure", "read")
"""The phases this script runs. `all` is `measure` and then `read`.

THE DEFAULT IS THE LAST, `read`: it spends no GPU time, and without records it
refuses by name (11). A default of `measure` would make a bare invocation pay for
nine cells."""

SEQUENCE_FRAMES: int = 5
"""Frames in the sequence window, `t - SEQUENCE_FRAMES + 1 .. t`: the rollout's own
context length. STATED, not derived; stored in every record as `sequence_frames`."""

HARNESS_R2_TOLERANCE: float = 0.10
"""How far `shipped_grid_r2` may sit from the study's `embedding_r2` before the harness is
said to measure something else.

NOT A DERIVED QUANTITY, and the two numbers are not the same measurement: the study's is
fit on 20 TRAINING episodes (about 10,000 rows) and selected on the rows it is scored on;
M3o's is fit, selected and scored on three disjoint thirds of the 24 VALIDATION episodes,
8 episodes and about 4,000 rows each. The nine study values span 0.3195-0.3451.

MEASURED, BY A ONE-OFF CHECK OUTSIDE THE STUDY (the nine cells' encoder rows through this
module's own `build_splits` and `encode_rows`, the two linear readouts only, no record
written): under the 8 / 8 / 8 split `C_shipped` reads -0.026 to +0.166, a gap of 0.17-0.35 to
the study in every one of the nine cells, so this tolerance refuses all nine; fit on the
study's own 20 training episodes instead, the same rows and targets read 0.300-0.342 against
0.320-0.345 (worst gap 0.035). The plumbing reproduces the study; the 8-episode fit does not,
and no tolerance that catches a broken harness also admits it."""

M3N_FLOOR_ERROR_MEAN: tuple[float, float] = (131.88, 259.06)
"""The shipped rollout probe's FLOOR error at h=1 across M3n's nine cells,
`curves.floor_position[0]` of `runs/m3n_motion`: 131.8767 to 259.0584. A MEAN over
validation windows (`rollout.evaluate_rollout` averages the per-window error), so it is
compared with `mean_error` ONLY."""

READOUTS: dict[str, tuple[str, str]] = {
    "pixels_linear": ("pixels", "linear"),
    "pixels_rbf": ("pixels", "rbf"),
    "embedding_linear": ("embedding", "linear"),
    "embedding_rbf": ("embedding", "rbf"),
    "embedding_knn": ("embedding", "knn"),
    "sequence_linear": ("sequence", "linear"),
    "sequence_rbf": ("sequence", "rbf"),
    "embedding_linear_shipped": ("embedding", "linear_shipped"),
}
"""Record name -> (feature set, `ceiling._FITTERS` entry). Eight: the seven the
brief names, and `embedding_linear_shipped` (C_shipped). The names `pixels_rbf`,
`embedding_rbf` and `sequence_rbf` are all the ESTIMATOR `"rbf"`, which is why
`B - D` and `F - B` have no name guard (see `measure_cell`)."""

PIXEL_READOUTS: tuple[str, ...] = tuple(
    name for name, (kind, _) in READOUTS.items() if kind == "pixels"
)
"""The model-free ones: scored once, shared by every cell, identical in every record."""

_ENCODE_CHUNK = 64
"""Frames per encoder call. Memory, not statistics: nothing downstream depends on it."""


def ceiling_record_path(out: Path, arm: str, seed: int) -> Path:
    """One file per CELL. The seed is in the name, not only in the payload.

    M3m's fixtures all used seed 0, which hid the collision this prevents: in the
    real run, three seeds of an arm would have been written to one filename,
    destroying 6 of 9 records and surfacing only at the read phase after the
    compute had been spent."""
    return Path(out) / f"ceiling_{arm}_seed{seed}.json"


def cell_bootstrap_seed(base_seed: int, arm: str, seed: int) -> int:
    """The seed one cell's bootstrap intervals are drawn at: a pure function of the
    run's base seed and the cell's identity.

    The verdict is a strict majority across nine cells and a majority count reads the
    cells as independent. Drawn at ONE shared seed, every cell resamples the same
    episode draws and the count reads one coin flip nine times. NEVER `hash()`: Python
    salts `hash(str)` per process (`PYTHONHASHSEED`), so no record would reproduce.
    `zlib.crc32` over the UTF-8 bytes of `"base:arm:seed"` is the same unsigned 32-bit
    integer on every process and platform."""
    return zlib.crc32(f"{int(base_seed)}:{arm}:{int(seed)}".encode("utf-8"))


# ---------------------------------------------------------------------------
# measure: the model-free rows, then one cell, then every cell.
# ---------------------------------------------------------------------------


def _share(total: int, take: int) -> float:
    """A `val_fraction` for which `episode_split` holds out exactly `take` of `total`.

    `episode_split` truncates `int(total * val_fraction)`, so `1/3` of 24 is a float
    that lands on 8.0 only by a rounding tie. Aiming at the middle of the integer
    cannot miss."""
    return (take + 0.5) / total


def three_way(val_paths, *, seed: int) -> dict[str, list]:
    """The validation episodes as `{"fit", "select", "score"}`, by episode.

    `episode_split` twice at the study's `SPLIT_SEED`: a third held out as the score
    split, the rest halved into fit and select (24 -> 8 / 8 / 8). A pure function of
    the episode NAMES, so it is the same for every cell and every run. Episodes, never
    frames: adjacent Doom frames are near-duplicates."""
    paths = list(val_paths)
    n_score = len(paths) // 3
    n_select = (len(paths) - n_score) // 2
    rest, score = episode_split(paths, val_fraction=_share(len(paths), n_score), seed=seed)
    fit, select = episode_split(rest, val_fraction=_share(len(rest), n_select), seed=seed)
    return {"fit": fit, "select": select, "score": score}


def _all_targets(path: Path) -> np.ndarray:
    """`(T + 1, 4)` probe targets of EVERY frame of one episode. Reads only the
    privileged arrays, so the 98 training episodes cost no pixels."""
    with np.load(path) as data:
        return probe_targets(data["privileged"], tuple(data["privileged_keys"].tolist()))


@dataclass(frozen=True)
class SplitRows:
    """One split's rows, model-free: which frames of which episodes, their targets,
    and the episode each belongs to."""

    paths: tuple
    frames: tuple
    """Per episode, the frame indices taken as rows (an int array)."""
    targets: np.ndarray
    groups: np.ndarray
    """Per row, the index of its episode within THIS split."""

    @property
    def n(self) -> int:
        return int(self.targets.shape[0])


def _split_rows(paths, targets: dict, stride: int) -> SplitRows:
    first = SEQUENCE_FRAMES - 1
    frames, rows, groups = [], [], []
    for index, path in enumerate(paths):
        taken = np.arange(first, len(targets[path]), stride)
        frames.append(taken)
        rows.append(targets[path][taken])
        groups.append(np.full(taken.shape[0], index, dtype=np.int64))
    stacked = np.concatenate(rows)
    if stacked.shape[0] == 0:
        raise SystemExit(
            f"a split of {len(paths)} episode(s) holds no frame with "
            f"SEQUENCE_FRAMES={SEQUENCE_FRAMES} frames of history"
        )
    return SplitRows(
        paths=tuple(paths), frames=tuple(frames), targets=stacked,
        groups=np.concatenate(groups),
    )


def build_splits(val_paths, *, budget: int) -> tuple[int, dict[str, SplitRows]]:
    """`(stride, {"fit", "select", "score"} -> SplitRows)`.

    The stride is the smallest that brings the LARGEST split within `budget` rows, so
    every split is subsampled by the one stride and `frame_stride` is one number.
    At the shipped split the largest split holds ~4,000 rows and the stride is 1."""
    if budget < 1:
        raise SystemExit(f"a frame budget of {budget} holds no frame")
    by_split = three_way(val_paths, seed=SPLIT_SEED)
    targets = {path: _all_targets(path) for paths in by_split.values() for path in paths}
    first = SEQUENCE_FRAMES - 1
    largest = max(
        sum(max(0, len(targets[p]) - first) for p in paths) for paths in by_split.values()
    )
    stride = max(1, -(-largest // budget))
    return stride, {name: _split_rows(paths, targets, stride) for name, paths in by_split.items()}


def load_pixels(split: SplitRows) -> np.ndarray:
    """`(n, 37632)` uint8: the raw frames of the rows. Left as uint8 on purpose: every
    fitter casts to float64 itself, exactly, so this holds 0.15 GB per split rather
    than 1.12 GiB."""
    parts = []
    for path, frames in zip(split.paths, split.frames):
        obs = load_episode(path).obs
        parts.append(obs[frames].reshape(len(frames), -1))
    return np.concatenate(parts)


@torch.no_grad()
def encode_rows(model, split: SplitRows, backbone, device) -> tuple[np.ndarray, np.ndarray]:
    """`(embedding, sequence)` of a split's rows under THIS model.

    `embedding` is `(n, E)`, the encoder output at the row's frame; `sequence` is
    `(n, SEQUENCE_FRAMES * E)`, the encoder outputs at `t - SEQUENCE_FRAMES + 1 .. t`,
    oldest first. Every frame of every episode is encoded, so a row's history is real
    frames whether or not the stride took them as rows."""
    singles, windows = [], []
    for path, frames in zip(split.paths, split.frames):
        episode = load_episode(path)
        source = source_for(model, path, episode, backbone)
        embedded = np.concatenate([
            model.encoder(torch.as_tensor(source[a:a + _ENCODE_CHUNK]).to(device))
            .float().cpu().numpy()
            for a in range(0, len(source), _ENCODE_CHUNK)
        ])
        singles.append(embedded[frames])
        windows.append(np.concatenate(
            [embedded[frames - back] for back in range(SEQUENCE_FRAMES - 1, -1, -1)], axis=1,
        ))
    return np.concatenate(singles), np.concatenate(windows)


@dataclass(frozen=True)
class Shared:
    """What one run computes ONCE because it uses no model."""

    budget: int
    stride: int
    splits: dict
    pixel_readouts: dict
    zero_information: dict
    """`{"mean", "median"}` on the SPEC's definition: the TRAINING mean position,
    against every validation frame. Not this run's fit split against its score split."""


def _require_finite_readouts(label: str, readouts: dict) -> None:
    for name, r in readouts.items():
        bad = [
            f"{key}={value!r}"
            for key, value in (("r2", r.r2), ("mean_error", r.mean_error),
                               ("median_error", r.median_error))
            if not math.isfinite(value)
        ]
        if bad:
            raise SystemExit(f"{label}: readout {name} is not finite ({', '.join(bad)})")


def prepare_shared(train_paths, val_paths, *, budget: int) -> Shared:
    """The rows, the two raw-pixel readouts, and the zero-information baseline."""
    stride, splits = build_splits(val_paths, budget=budget)
    triples = {
        name: (load_pixels(split), split.targets) for name, split in splits.items()
    }
    readouts = {
        name: ceiling.score_readout(
            READOUTS[name][1], triples["fit"], triples["select"], triples["score"]
        )
        for name in PIXEL_READOUTS
    }
    del triples
    _require_finite_readouts("the raw-pixel readouts", readouts)
    zero = ceiling.zero_information_error(
        np.concatenate([_all_targets(p) for p in train_paths]),
        np.concatenate([_all_targets(p) for p in val_paths]),
    )
    return Shared(
        budget=budget, stride=stride, splits=splits, pixel_readouts=readouts,
        zero_information=zero,
    )


def applied_quality(trajectories) -> dict:
    """The rollout probe's r2 and map-unit errors on its OWN imagined output.

    Spec section 3.1, and the reason it is measured first: the 2x2 is fit on REAL
    encoder embeddings at frame t, while the error that made M3n unreadable comes from
    this distribution. The rows are `reference_trajectories`' `positions` (the probe's
    output at each imagined step, `(n_windows, horizon, 2)`) against `true_positions`.
    `mean_error` is carried beside the median because M3n's 131.88-259.06 is a mean."""
    predicted = trajectories.positions.reshape(-1, 2)
    truth = trajectories.true_positions.reshape(-1, 2)
    errors = position_error(predicted, truth)
    return {
        "r2": float(_mean_r2(predicted, truth)),
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
    }


def widest_rbf_r2(fit, select, score) -> float:
    """The score r2 of the RBF at the WIDEST bandwidth alone, jitter selected on `select`.

    `ceiling.score_readout` scores the whole ladder (linear rung included), so the
    widest rung is not separately reachable through it; this repeats the ladder's own
    steps for that one rung, over the same `CEILING_RIDGES`. It is the control whose
    claim is the OLD asymptotic one (an RBF approaches linear as its bandwidth grows),
    which `ceiling.fit_kernel_probe` records as false in practice at high linear r2:
    the ladder nests linear by a rung of its own, so this number is expected to read
    BELOW the linear r2 there, and is recorded as a measurement, never gated."""
    x = np.asarray(fit[0], dtype=np.float64)
    y = np.asarray(fit[1], dtype=np.float64)
    mean, scale = x.mean(0), x.std(0) + 1e-8
    xs = (x - mean) / scale
    selected = (np.asarray(select[0], dtype=np.float64) - mean) / scale
    scored = (np.asarray(score[0], dtype=np.float64) - mean) / scale
    gamma = 1.0 / (2.0 * (max(ceiling.BANDWIDTHS) * ceiling._median_scale(xs)) ** 2)
    gram = ceiling._rbf(xs, xs, gamma)
    on_select = ceiling._rbf(selected, xs, gamma)
    on_score = ceiling._rbf(scored, xs, gamma)
    values, vectors = np.linalg.eigh(gram)
    centre = y.mean(0)
    coords = vectors.T @ (y - centre)
    best = None
    for jitter in ceiling.CEILING_RIDGES:
        alpha = vectors @ (coords / (values + jitter)[:, None])
        got = _mean_r2(on_select @ alpha + centre, select[1])
        if best is None or got > best[0]:
            best = (got, alpha)
    return float(_mean_r2(on_score @ best[1] + centre, score[1]))


def _capacity(value: float):
    """`float(value)`, or the string `"inf"` for a positive infinity: the linear rung
    of the kernel ladder. NaN and -inf are not capacities and are refused."""
    value = float(value)
    if math.isinf(value) and value > 0.0:
        return "inf"
    if not math.isfinite(value):
        raise ValueError(f"a readout's capacity is {value}; only +inf (a linear rung) is a capacity")
    return value


def _readout_record(readout) -> dict:
    jitter = readout.jitter
    if jitter is not None and not math.isfinite(jitter):
        raise ValueError(f"a readout's jitter is {jitter}; it must be finite or None")
    return {
        "r2": float(readout.r2),
        "mean_error": float(readout.mean_error),
        "median_error": float(readout.median_error),
        "capacity": _capacity(readout.capacity),
        "jitter": None if jitter is None else float(jitter),
    }


def _study_numbers(arm: str, seed: int, study_record: dict) -> dict:
    """The four things read off the cell's own study record, STRICTLY: a record
    without them is not a study record."""
    try:
        return {
            "step": int(study_record["steps"]),
            "record_git_sha": study_record["git_sha"],
            "embedding_r2": float(study_record["filtering"]["criterion_4"]["embedding_r2"]),
            "embedding_selection_r2": float(study_record["probe"]["embedding_selection_r2"]),
        }
    except (KeyError, TypeError) as error:
        raise SystemExit(
            f"{arm} seed {seed}: the study record lacks `steps`, `git_sha`, "
            "`filtering.criterion_4.embedding_r2` or `probe.embedding_selection_r2`; "
            "it is not a study record"
        ) from error


def measure_cell(
    model, val_paths, probe, *, arm: str, seed: int, context: int, horizon: int, device,
    feature_backbone, study_record: dict, bootstrap_seed: int, shared: Shared,
) -> dict:
    """One cell: the applied pass, eight readouts, three gains, four controls.

    THE PROBE GAIN GOES THROUGH `ceiling.probe_gain_over_floor` AND NOTHING ELSE. It is
    the only thing that forms `D - C - max(null_floor, 0)`; a `difference_interval` here
    would read every cell against ZERO, and nothing downstream could tell.

    `B - D` AND `F - B` CANNOT BE NAME-GUARDED: B, D and F are all the estimator
    `"rbf"`, so exchanging the two arguments returns a gain of the wrong sign without a
    word. They are pinned by value in the tests, on a fixture where the signs are known.

    EXIT 49 AND THE RECONCILIATION READ `shipped_grid_r2`, the `embedding_linear_shipped`
    readout, never `embedding_linear`. The finer grid of the latter is expected up to
    ~0.02 above the study's.

    `bootstrap_seed` IS THE RUN'S BASE; every interval of this cell is drawn at
    `cell_bootstrap_seed(bootstrap_seed, arm, seed)`. Both are recorded.

    `null_floor` IS MEASURED AT THIS CELL'S SHAPE (the fit split's rows, the embedding
    width) and recorded as the unclamped mean it is.

    The record has no `nonfinite` key: `study.write_record` owns it and raises for a
    record that carries one.
    """
    study = _study_numbers(arm, seed, study_record)
    union = sorted(p.name for split in shared.splits.values() for p in split.paths)
    if union != sorted(Path(p).name for p in val_paths):
        raise SystemExit(
            f"{arm} seed {seed}: the rollout's validation episodes are not the episodes "
            "the shared rows were drawn from"
        )
    applied = applied_quality(reference_trajectories(
        model, val_paths, probe, context=context, horizon=horizon, seed=seed,
        device=device, feature_backbone=feature_backbone,
    ))
    encoded = {
        name: encode_rows(model, split, feature_backbone, device)
        for name, split in shared.splits.items()
    }
    features = {
        name: {"embedding": single, "sequence": window}
        for name, (single, window) in encoded.items()
    }
    targets = {name: split.targets for name, split in shared.splits.items()}

    readouts = {}
    for name, (kind, estimator) in READOUTS.items():
        if kind == "pixels":
            readouts[name] = shared.pixel_readouts[name]
        else:
            readouts[name] = ceiling.score_readout(
                estimator,
                *((features[s][kind], targets[s]) for s in ("fit", "select", "score")),
            )
    _require_finite_readouts(f"{arm} seed {seed}", readouts)

    n_fit, width = shared.splits["fit"].n, int(features["fit"]["embedding"].shape[1])
    floor = ceiling.null_floor(n_fit, width)
    cell_seed = cell_bootstrap_seed(bootstrap_seed, arm, seed)
    groups, truth = shared.splits["score"].groups, targets["score"]
    c, d = readouts["embedding_linear"], readouts["embedding_rbf"]
    b, f = readouts["pixels_rbf"], readouts["sequence_rbf"]
    gains = {
        "probe": ceiling.probe_gain_over_floor(d, c, floor, truth, groups, seed=cell_seed),
        "pixel": ceiling.difference_interval(b, d, truth, groups, seed=cell_seed),
        "sequence": ceiling.difference_interval(f, b, truth, groups, seed=cell_seed),
    }

    shipped = readouts["embedding_linear_shipped"].r2
    triples = [(features[s]["embedding"], targets[s]) for s in ("fit", "select", "score")]
    return {
        "arm": arm,
        "seed": seed,
        "step": study["step"],
        "git_sha": git_sha(),
        "record_git_sha": study["record_git_sha"],
        "torch_version": torch.__version__,
        "device": str(device),
        "frames": shared.budget,
        "frame_stride": shared.stride,
        "rows_per_split": {name: split.n for name, split in shared.splits.items()},
        "sequence_frames": SEQUENCE_FRAMES,
        "confidence": ceiling.CONFIDENCE,  # a level the estimator takes, so it moves with it
        "resamples": ceiling.RESAMPLES,
        "null_draws": ceiling.NULL_DRAWS,
        "bootstrap_seed": bootstrap_seed,
        "cell_bootstrap_seed": cell_seed,
        "split_seed": SPLIT_SEED,
        "zero_information_median": ceiling.ZERO_INFORMATION_MEDIAN,
        "harness_r2_tolerance": HARNESS_R2_TOLERANCE,
        "episodes": {
            name: [p.name for p in split.paths] for name, split in shared.splits.items()
        },
        "groups": groups.tolist(),
        "applied": applied,
        "ceiling_ridges": [float(r) for r in ceiling.CEILING_RIDGES],
        "readouts": {name: _readout_record(r) for name, r in readouts.items()},
        "shipped_grid_r2": float(shipped),
        "null_floor": float(floor),
        "null_floor_shape": {"n": n_fit, "d": width},
        "gains": {
            name: {key: float(interval[key]) for key in ("point", "ci_low", "ci_high")}
            for name, interval in gains.items()
        },
        "controls": {
            "harness_r2_matches_study": bool(
                abs(shipped - study["embedding_r2"]) <= HARNESS_R2_TOLERANCE
            ),
            "study_embedding_r2": study["embedding_r2"],
            "study_embedding_selection_r2": study["embedding_selection_r2"],
            "nesting_rbf_at_widest_vs_linear": float(
                widest_rbf_r2(*triples) - readouts["embedding_linear"].r2
            ),
            "zero_information_recomputed": dict(shared.zero_information),
        },
    }


def _triple(interval: dict) -> str:
    return f"{interval['point']:+.4f} [{interval['ci_low']:+.4f},{interval['ci_high']:+.4f}]"


def _cell_line(record: dict, path: Path) -> str:
    """One line per measured cell: the three gains, the floor, whether the harness
    agrees with the study, and the rows. The controls are RECORDED here, not judged --
    but a missed one is NAMED, because the measure carries on and this is the only place
    an operator sees it before the read refuses it."""
    controls, gains = record["controls"], record["gains"]
    agrees = (
        "harness agrees" if controls["harness_r2_matches_study"]
        else "HARNESS DISAGREES with the study"
    )
    rows = record["rows_per_split"]
    return (
        f"{record['arm']} seed {record['seed']}: D-C-floor {_triple(gains['probe'])}; "
        f"B-D {_triple(gains['pixel'])}; F-B {_triple(gains['sequence'])}; "
        f"null floor {record['null_floor']:+.4f}; C_shipped {record['shipped_grid_r2']:.3f} "
        f"vs study {controls['study_embedding_r2']:.3f} ({agrees}); rows "
        f"{rows['fit']}/{rows['select']}/{rows['score']} at stride {record['frame_stride']}; "
        f"wrote {path}"
    )


def _cell_args(args, source: Path) -> types.SimpleNamespace:
    """`prepare_cell` reads `out`, `device`, `context` and `horizon` off its args, and
    its `out` is the STUDY directory the checkpoint is loaded from. This script's own
    `--out` holds the ceiling records and nothing else, so handing `args` through would
    look for the nine checkpoints in the output directory and refuse EVERY cell. Same
    shim, same reason, as `scripts/motion_headroom.py`."""
    return types.SimpleNamespace(
        out=Path(source), device=args.device, context=args.context, horizon=args.horizon,
    )


def measure_phase(args) -> int:
    """Every requested cell: `prepare_cell`'s refusals, then `measure_cell`, then one
    record per cell through `write_record`.

    EVERY CELL IS LOADED BEFORE ANY IS MEASURED, so a missing ninth cell is found before
    eight have been paid for (11). THE FIRST NON-`EXIT_OK` STATUS IS RETURNED and nothing
    is written for the cell that earned it. AN EXCEPTION FROM A CELL IS NOT CAUGHT: it
    reaches the caller as a traceback, the cells before it keep their records.

    The model-free rows and raw-pixel readouts are computed ONCE, after the first cell
    has passed `prepare_cell` (a refused first cell costs no pixel readouts), and the same
    `Shared` goes to every cell. CELLS ARE LOADED FROM `args.source`, NOT `args.out`."""
    arms, seeds = _plan(args)
    device = get_device(prefer=args.device)
    try:
        loaded = [load_cell(args.source, arm, seed) for arm in arms for seed in seeds]
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    train, val = episode_split(
        buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    budget = ceiling.FRAME_BUDGET if args.frames is None else args.frames
    shared = None
    for cell in loaded:
        status, prepared = prepare_cell(
            _cell_args(args, args.source), cell, device, train, val
        )
        if prepared is None:
            return status
        if shared is None:
            shared = prepare_shared(train, val, budget=budget)
        record = measure_cell(
            prepared.model, val, prepared.embedding_probe, arm=cell.arm, seed=cell.seed,
            context=prepared.context, horizon=prepared.horizon, device=device,
            feature_backbone=prepared.common["feature_backbone"], study_record=cell.record,
            bootstrap_seed=args.bootstrap_seed, shared=shared,
        )
        path = ceiling_record_path(args.out, cell.arm, cell.seed)
        write_record(path, record)
        print(_cell_line(record, path))
    return EXIT_OK


# ---------------------------------------------------------------------------
# read: what pooling these records requires, and the reading.
# ---------------------------------------------------------------------------


_PROTOCOL_FIELDS = (
    ("git_sha", lambda r: str(r["git_sha"])),
    ("torch_version", lambda r: str(r["torch_version"])),
    ("device", lambda r: str(r["device"])),
    ("step", lambda r: int(r["step"])),
    ("frames", lambda r: int(r["frames"])),
    ("frame_stride", lambda r: int(r["frame_stride"])),
    ("rows_per_split.fit", lambda r: int(r["rows_per_split"]["fit"])),
    ("rows_per_split.select", lambda r: int(r["rows_per_split"]["select"])),
    ("rows_per_split.score", lambda r: int(r["rows_per_split"]["score"])),
    ("sequence_frames", lambda r: int(r["sequence_frames"])),
    ("confidence", lambda r: float(r["confidence"])),
    ("resamples", lambda r: int(r["resamples"])),
    ("null_draws", lambda r: int(r["null_draws"])),
    ("split_seed", lambda r: int(r["split_seed"])),
    ("bootstrap_seed", lambda r: int(r["bootstrap_seed"])),
    ("zero_information_median", lambda r: float(r["zero_information_median"])),
    ("harness_r2_tolerance", lambda r: float(r["harness_r2_tolerance"])),
    ("zero_information.mean",
     lambda r: float(r["controls"]["zero_information_recomputed"]["mean"])),
    ("zero_information.median",
     lambda r: float(r["controls"]["zero_information_recomputed"]["median"])),
    ("ceiling_ridges", lambda r: [float(v) for v in r["ceiling_ridges"]]),
    ("episodes.fit", lambda r: list(r["episodes"]["fit"])),
    ("episodes.select", lambda r: list(r["episodes"]["select"])),
    ("episodes.score", lambda r: list(r["episodes"]["score"])),
    ("groups", lambda r: [int(g) for g in r["groups"]]),
)
"""Every field `require_one_protocol` compares, as `(name, pick)`, in one module-level
table that the function ITERATES and a test reads -- so a field cannot be added to the
comparison without the test noticing it has no disagreement case.

What each says: `git_sha` and `torch_version` which code and build; `step` which
checkpoints; `frames`, `frame_stride`, `rows_per_split.*`, `sequence_frames`,
`episodes.*` and `groups` that the rows were the same; `confidence`, `resamples`,
`null_draws`, `ceiling_ridges` what every interval, floor and readout was taken at;
`bootstrap_seed` the run's BASE (each cell's own `cell_bootstrap_seed` differs by
construction and is not compared); `zero_information.*` the one model-free baseline
every map-unit ratio is read against, which must therefore be one number.

NOT COMPARED, on purpose: `record_git_sha`, `applied`, `null_floor` and everything
else a property of the CELL. The lists are separate fields (`episodes.fit`, ...) because
`_versus` can only say where two long LISTS differ.

A record WITHOUT a field is refused by name (`_pick`), not defaulted."""


def _pick(cell, field: str, pick, record: dict):
    """`pick(record)`, or the named refusal for a record that lacks `field`."""
    try:
        return pick(record)
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(
            f"{cell[0]} seed {cell[1]} lacks {field}: it is not a ceiling record this "
            "script wrote"
        ) from error


_LINE = 60


def _brief(value) -> str:
    text = repr(value)
    return text if len(text) <= _LINE else text[:_LINE - 3] + "..."


def _versus(mine, reference) -> str:
    """`mine vs reference`, in a form a person can act on: a value that fits on a line
    as it is; two long lists by LENGTH when it differs and by the FIRST INDEX where they
    differ when it does not. No fallback: no `_PROTOCOL_FIELDS` pick is a long value that
    is not a list."""
    shown = f"{_brief(mine)} vs {_brief(reference)}"
    if max(len(repr(mine)), len(repr(reference))) <= _LINE:
        return shown
    if len(mine) != len(reference):
        return f"{len(mine)} vs {len(reference)} rows"
    at = next(i for i, (a, b) in enumerate(zip(mine, reference)) if a != b)
    return (
        f"{len(mine)} rows, first differing at index {at}: "
        f"{_brief(mine[at])} vs {_brief(reference[at])}"
    )


def _require_agreement(records: dict, fields) -> list:
    """The sorted `(cell, record)` pairs, after refusing any two that disagree on one of
    `fields` -- named by cell and field. EVERY record is compared with the first."""
    items = sorted(records.items())
    if not items:
        raise SystemExit("no ceiling record to read")
    (first_cell, first), rest = items[0], items[1:]
    picks = dict(_PROTOCOL_FIELDS)
    for field in fields:
        reference = _pick(first_cell, field, picks[field], first)
        for cell, record in rest:
            mine = _pick(cell, field, picks[field], record)
            if mine != reference:
                raise SystemExit(
                    f"{cell[0]} seed {cell[1]} and {first_cell[0]} seed "
                    f"{first_cell[1]} disagree on {field}: {_versus(mine, reference)}; "
                    "they are not one measurement and cannot be read as one"
                )
    return items


def require_one_protocol(records: dict) -> None:
    """Every record reports the same protocol on the same rows, or the two that disagree
    are named with the field."""
    _require_agreement(records, [name for name, _ in _PROTOCOL_FIELDS])


def _is_finite_number(value) -> bool:
    """A real, finite number. Not a bool, not NaN, not a string, not None."""
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _valid_capacity(value) -> bool:
    """A finite number, or exactly the string `"inf"` (a linear rung's bandwidth)."""
    return _is_finite_number(value) or (isinstance(value, str) and value == "inf")


_READOUT_FIELDS = ("r2", "mean_error", "median_error", "capacity", "jitter")


def require_sound_records(records: dict) -> None:
    """Refuse a record whose measurement is not to be trusted, by name, before any
    statistic is read from it.

    * The sanitiser's `nonfinite` map is not empty: this script writes `{}` (it turns
      the one legitimate infinity, a linear rung's bandwidth, into `"inf"`).
    * `applied` or `null_floor` (the two figures the map-unit section prints and the
      floor the gain was net of) is missing or not a finite number.
    * A readout lacks one of its five fields, carries a non-finite r2 or error, a
      capacity that is neither a finite number nor `"inf"`, or a jitter that is neither
      null nor finite.
    * The raw-pixel readouts differ between cells. They use no model and one `Readout`
      is written into all nine records, so a difference means a model leaked into a
      model-free measurement."""
    items = sorted(records.items())
    for cell, record in items:
        found = _pick(cell, "nonfinite", lambda r: dict(r["nonfinite"]), record)
        if found:
            raise SystemExit(
                f"{cell[0]} seed {cell[1]} carries non-finite value(s) at "
                f"{', '.join(sorted(found))}, which this script never writes; "
                "this is not a record it wrote"
            )
        applied = _pick(cell, "applied", lambda r: dict(r["applied"]), record)
        for key in ("r2", "mean_error", "median_error"):
            if not _is_finite_number(applied.get(key)):
                raise SystemExit(
                    f"{cell[0]} seed {cell[1]}: applied.{key} is {applied.get(key)!r}, "
                    "not a finite number"
                )
        floor = _pick(cell, "null_floor", lambda r: r["null_floor"], record)
        shape = _pick(cell, "null_floor_shape",
                      lambda r: (r["null_floor_shape"]["n"], r["null_floor_shape"]["d"]), record)
        if not _is_finite_number(floor) or not all(_is_finite_number(v) for v in shape):
            raise SystemExit(
                f"{cell[0]} seed {cell[1]}: null_floor {floor!r} at shape {shape!r} is not "
                "a finite number at a finite shape"
            )
        for name in READOUTS:
            entry = _pick(cell, f"readouts.{name}", lambda r: dict(r["readouts"][name]), record)
            missing = [key for key in _READOUT_FIELDS if key not in entry]
            if missing:
                raise SystemExit(
                    f"{cell[0]} seed {cell[1]}: readouts.{name} lacks {', '.join(missing)}"
                )
            for key in ("r2", "mean_error", "median_error"):
                value = entry[key]
                if not _is_finite_number(value):
                    raise SystemExit(
                        f"{cell[0]} seed {cell[1]}: readouts.{name}.{key} is {value!r}, "
                        "not a finite number"
                    )
            if not _valid_capacity(entry["capacity"]):
                raise SystemExit(
                    f"{cell[0]} seed {cell[1]}: readouts.{name}.capacity is "
                    f"{entry['capacity']!r}; a capacity is a finite number or the string "
                    '"inf", never a bare non-finite token'
                )
            jitter = entry["jitter"]
            if jitter is not None and not _is_finite_number(jitter):
                raise SystemExit(
                    f"{cell[0]} seed {cell[1]}: readouts.{name}.jitter is {jitter!r}; "
                    "it is null or a finite number"
                )
    (first_cell, first), rest = items[0], items[1:]
    for name in PIXEL_READOUTS:
        for cell, record in rest:
            if record["readouts"][name] != first["readouts"][name]:
                raise SystemExit(
                    f"{cell[0]} seed {cell[1]} and {first_cell[0]} seed {first_cell[1]} "
                    f"disagree on readouts.{name}, which uses no model and must be "
                    "identical in every cell: a model leaked into a model-free measurement"
                )


def harness_failures(records: dict) -> list[tuple]:
    """The cells whose `harness_r2_matches_study` is not exactly `True`, as `(cell, record)`.

    A record whose flag contradicts the numbers beside it (`shipped_grid_r2`,
    `study_embedding_r2`, `harness_r2_tolerance`) is refused by name rather than
    believed either way. The flag is read as `is True`: anything else, `null`
    included, is a disagreement."""
    failed = []
    for cell, record in sorted(records.items()):
        controls = _pick(cell, "controls", lambda r: dict(r["controls"]), record)
        flag = _pick(cell, "controls.harness_r2_matches_study",
                     lambda r: r["controls"]["harness_r2_matches_study"], record)
        expected = _pick(cell, "the harness numbers", lambda r: bool(
            abs(float(r["shipped_grid_r2"]) - float(r["controls"]["study_embedding_r2"]))
            <= float(r["harness_r2_tolerance"])
        ), record)
        if isinstance(flag, bool) and flag != expected:
            raise SystemExit(
                f"{cell[0]} seed {cell[1]}: harness_r2_matches_study is {flag} but "
                f"shipped_grid_r2 {record['shipped_grid_r2']!r} against study_embedding_r2 "
                f"{controls['study_embedding_r2']!r} at tolerance "
                f"{record['harness_r2_tolerance']!r} says {expected}; this is not a "
                "record this script wrote"
            )
        if flag is not True:
            failed.append((cell, record))
    return failed


def require_readable_plan(arms) -> None:
    """Refuse a plan with fewer than `ARMS_REQUIRED` DISTINCT arms: asked here it costs
    nothing, asked after `--phase all` it is nine cells of compute. The caller passes the
    DISTINCT arms (`_plan`)."""
    if len(arms) < ceiling.ARMS_REQUIRED:
        raise SystemExit(
            f"a plan of {len(arms)} arm(s) cannot be read: a verdict needs "
            f"ARMS_REQUIRED={ceiling.ARMS_REQUIRED} arms, since one arm's finding is not a "
            "finding about the model"
        )


def require_seeds_for_a_verdict(seeds) -> None:
    """Refuse a plan with fewer than `SEEDS_MINIMUM` DISTINCT seeds. CALLED BY `main` FOR
    `--phase all` ONLY, never by `read_phase`: a `read` over a short plan must REACH
    `reading_ceiling`, whose `ValueError` names the arm and its seeds."""
    if len(seeds) < ceiling.SEEDS_MINIMUM:
        raise SystemExit(
            f"a plan of {len(seeds)} seed(s) cannot reach a verdict: every arm would "
            f"carry fewer than SEEDS_MINIMUM={ceiling.SEEDS_MINIMUM} seeds, and "
            "`reading_ceiling` refuses such an arm once the cells are already "
            f"measured. Pass at least {ceiling.SEEDS_MINIMUM} distinct seeds, or run "
            "--phase measure for a smoke"
        )


def _bare_constant(path: Path):
    def refuse(token: str):
        raise SystemExit(
            f"{path.name} contains the bare token {token}, which is not JSON; a "
            'non-finite capacity is written as the string "inf" and nothing else is '
            "ever non-finite. This is not a record this script wrote"
        )

    return refuse


def load_ceiling(out_dir: Path, arms, seeds) -> dict:
    """Every planned cell's record, keyed by `(arm, seed)` -- or `CellMissing` naming the
    first that is not on disk.

    Each file is parsed ONCE STRICTLY first: `json.loads` accepts a bare `Infinity` or
    `NaN` by default, which is how a non-finite capacity would round-trip unseen. A record
    whose own `arm`/`seed` disagree with the file it was read under is refused too."""
    records: dict[tuple[str, int], dict] = {}
    for arm in arms:
        for seed in seeds:
            path = ceiling_record_path(out_dir, arm, int(seed))
            if not path.exists():
                raise CellMissing(
                    f"{arm} seed {int(seed)}: no ceiling record at {path}; "
                    "run --phase measure first"
                )
            json.loads(path.read_text(), parse_constant=_bare_constant(path))
            record = load_record(path)
            if record.get("arm") != arm or int(record.get("seed", -1)) != int(seed):
                raise SystemExit(
                    f"{path.name} was read for {arm} seed {int(seed)} but its "
                    f"record says arm={record.get('arm')!r} seed={record.get('seed')!r}"
                )
            records[(arm, int(seed))] = record
    return records


def _gain(cell, record: dict, name: str) -> dict:
    try:
        entry = record["gains"][name]
        return {key: float(entry[key]) for key in ("point", "ci_low", "ci_high")}
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(
            f"{cell[0]} seed {cell[1]} lacks gains.{name}: it is not a ceiling record "
            "this script wrote"
        ) from error


def ceiling_inputs(records: dict) -> ceiling.CeilingInputs:
    """The pooled records as `CeilingInputs`.

    EVERY CELL IS BUILT WITH KEYWORDS: the three gains are identically typed dicts, and
    `CeilingCell` is keyword-only for exactly that reason. REFUSES a record whose
    episode labels carry fewer than two distinct episodes, rather than reading an
    interval that was resampled from one unit."""
    cells = {}
    for (arm, seed), record in sorted(records.items()):
        labels = _pick((arm, seed), "groups", lambda r: [int(g) for g in r["groups"]], record)
        if len(set(labels)) < 2:
            raise SystemExit(
                f"{arm} seed {seed} carries {len(set(labels))} distinct episode label(s); "
                "an episode-clustered interval needs at least two"
            )
        cells[(arm, seed)] = ceiling.CeilingCell(
            arm=arm, seed=seed,
            probe_gain=_gain((arm, seed), record, "probe"),
            pixel_gain=_gain((arm, seed), record, "pixel"),
            sequence_gain=_gain((arm, seed), record, "sequence"),
        )
    return ceiling.CeilingInputs(cells=cells)


def _pct(ratio: float) -> str:
    return f"{100.0 * ratio:5.1f}%"


def _unit_line(name: str, entry: dict, zero_mean: float, zero_median: float) -> str:
    capacity = entry["capacity"]
    capacity = capacity if isinstance(capacity, str) else f"{capacity:.4g}"
    jitter = "-" if entry["jitter"] is None else f"{entry['jitter']:.4g}"
    return (
        f"    {name:<26}{entry['r2']:+8.4f}{entry['mean_error']:>12.2f}"
        f"{_pct(entry['mean_error'] / zero_mean):>9}{entry['median_error']:>15.2f}"
        f"{_pct(entry['median_error'] / zero_median):>9}  {capacity:>8}{jitter:>10}"
    )


_UNIT_HEADER = (
    f"    {'readout':<26}{'r2':>8}{'mean_error':>12}{'/ mean':>9}{'median_error':>15}"
    f"{'/ median':>9}  {'capacity':>8}{'jitter':>10}"
)


def format_map_units(records: dict) -> str:
    """The map-unit section appended to the reading: every readout's `mean_error` against
    the zero-information MEAN and its `median_error` against the MEDIAN, each ratio under a
    column that names its statistic, with M3n's 131.88-259.06 -- a mean -- stated as
    comparable to the mean column only.

    The baselines come from the records: the mean is `controls.zero_information_recomputed`
    and the median is the stored `zero_information_median` (the protocol check has shown
    both identical across cells). The raw-pixel readouts are printed once, since they are
    one number for the study."""
    items = sorted(records.items())
    first = items[0][1]
    zero_mean = float(first["controls"]["zero_information_recomputed"]["mean"])
    zero_median = float(first["zero_information_median"])
    rows = first["rows_per_split"]
    low, high = M3N_FLOOR_ERROR_MEAN
    lines = [
        "",
        "map units",
        f"  Rows per split: fit {rows['fit']}, select {rows['select']}, score {rows['score']}; "
        f"frame stride {first['frame_stride']} under a budget of {first['frames']}.",
        f"  Zero-information baselines, one set of frames: MEAN {zero_mean:.2f} (recomputed), "
        f"MEDIAN {zero_median:.2f} (pinned). Each error is read against its OWN statistic:",
        "  mean_error against the mean, median_error against the median; a mean over a "
        "median is not a ratio.",
        f"  M3n's floor error, {low:.2f}-{high:.2f} at h=1, is a MEAN over windows: it is "
        "comparable to the mean_error column ONLY.",
        "",
        "  raw-pixel readouts (no model; identical in every cell)",
        _UNIT_HEADER,
    ]
    lines += [
        _unit_line(name, first["readouts"][name], zero_mean, zero_median)
        for name in PIXEL_READOUTS
    ]
    for (arm, seed), record in items:
        applied = record["applied"]
        lines += [
            "",
            f"  {arm}_seed{seed}",
            f"    applied (the rollout probe on its imagined positions): r2 {applied['r2']:+.4f}, "
            f"mean_error {applied['mean_error']:.2f} ({_pct(applied['mean_error'] / zero_mean)}"
            f" of the zero-information mean), median_error {applied['median_error']:.2f} "
            f"({_pct(applied['median_error'] / zero_median)} of the zero-information median)",
            _UNIT_HEADER,
        ]
        lines += [
            _unit_line(name, record["readouts"][name], zero_mean, zero_median)
            for name in READOUTS if name not in PIXEL_READOUTS
        ]
        lines.append(
            f"    null floor {record['null_floor']:+.4f} at n={record['null_floor_shape']['n']}, "
            f"d={record['null_floor_shape']['d']}"
        )
    lines.append("")
    return "\n".join(lines)


def _plan(args) -> tuple[list[str], list[int]]:
    """The DISTINCT arms and seeds, in the order given: measuring from the lists as given
    would measure a cell twice and write its record twice."""
    return (
        list(dict.fromkeys(args.arms)),
        list(dict.fromkeys(int(s) for s in args.seeds)),
    )


def _harness_message(failed) -> str:
    lines = [
        "HARNESS DISAGREES: C_shipped (shipped_grid_r2) does not reproduce the study's "
        "own encoder-embedding probe in the cell(s) below, so M3o is measuring something "
        "other than what the study measured and no verdict about the model is available."
    ]
    for (arm, seed), record in failed:
        lines.append(
            f"  {arm} seed {seed}: C_shipped {record['shipped_grid_r2']:.4f} against the "
            f"study's {record['controls']['study_embedding_r2']:.4f} "
            f"(tolerance {record['harness_r2_tolerance']})"
        )
    return "\n".join(lines)


def read_phase(args) -> int:
    """Refuse a plan that cannot be read, pool every requested cell (11 names the first
    missing), refuse records that are not one measurement or not sound, return 49 if any
    cell's harness disagrees with the study, and otherwise take the reading and return
    `READ_EXITS.get(reading.verdict, EXIT_OK)`: 0 for a model verdict, 50 for no majority.

    THE READING IS PRINTED AND WRITTEN AS ONE STRING, for every verdict, 50 included: it
    is a finding, and the table is what a reader needs to see it. The text carries the
    verdict AND THE RULE, then the map-unit section. It is printed with `end=""` and
    written as it is, never `rstrip()`ped.

    A REFUSAL WRITES NOTHING, AND LEAVES NOTHING. The `ceiling.txt` an EARLIER read of
    the same `--out` left is removed FIRST, so every way out that does not reach the write
    leaves the directory without a reading this invocation did not produce.

    NOTHING CATCHES `reading_ceiling` HERE: its `ValueError` for an arm short of
    `SEEDS_MINIMUM` seeds is a malformed plan, not something found in the data."""
    reading_path = Path(args.out) / "ceiling.txt"
    reading_path.unlink(missing_ok=True)
    arms, seeds = _plan(args)
    require_readable_plan(arms)
    try:
        records = load_ceiling(args.out, arms, seeds)
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    require_one_protocol(records)
    require_sound_records(records)
    failed = harness_failures(records)
    if failed:
        print(_harness_message(failed))
        return EXIT_HARNESS_DISAGREES
    inputs = ceiling_inputs(records)
    reading = ceiling.reading_ceiling(inputs)
    text = ceiling.format_reading_ceiling(reading, inputs) + format_map_units(records)
    reading_path.write_text(text)
    print(text, end="")
    return READ_EXITS.get(reading.verdict, EXIT_OK)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class _Csv(argparse.Action):
    """A list flag that reads commas and spaces alike: `--seeds 0,1,2` and `--seeds 0 1 2`
    are one plan, and so is `--seeds 0,1 2`. The run's own command line passes comma-joined
    lists; `prediction_burden.py` takes space-separated ones. Repeats are dropped, in the
    order given."""

    cast = staticmethod(str)

    def check(self, parser, option, items) -> None:
        """Hook for a subclass that restricts the values."""

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            items = [
                self.cast(part) for token in values for part in token.split(",") if part
            ]
        except ValueError:
            parser.error(f"{option_string}: cannot read {' '.join(values)!r}")
        if not items:
            parser.error(f"{option_string}: no value given")
        items = list(dict.fromkeys(items))
        self.check(parser, option_string, items)
        setattr(namespace, self.dest, items)


class _Arms(_Csv):
    def check(self, parser, option, items) -> None:
        unknown = [arm for arm in items if arm not in ARMS]
        if unknown:
            parser.error(f"{option}: {unknown!r} are not among the arms {list(ARMS)!r}")


class _Ints(_Csv):
    cast = staticmethod(int)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "M3o: what can the readout reach, independently of the model? "
            "-- measure the cells, read the verdict"
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("runs/m3o_ceiling"))
    parser.add_argument("--source", type=Path, default=Path("runs/m3_study_v2"),
                        help="the M3c study directory: the nine 20,000-step cells")
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--device", default="mps")
    # None -> the cell's diagnostic says what it was written at; a value that
    # disagrees is refused (`protocol_mismatch`, 14).
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--arms", nargs="+", action=_Arms, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", action=_Ints, default=list(SEEDS))
    parser.add_argument("--frames", type=int, default=None,
                        help="the frame budget per split (default: ceiling.FRAME_BUDGET); "
                             "the stride that brings the largest split within it is recorded")
    parser.add_argument("--bootstrap-seed", type=int, default=0,
                        help="the run's base seed; each cell draws its intervals at "
                             "cell_bootstrap_seed(base, arm, seed), and both are "
                             "recorded in the cell")
    parser.add_argument("--phase", choices=PHASES, default=PHASES[-1])
    return parser


def main(argv: list[str] | None = None) -> int:
    """`measure`, `read`, or `all` -- the measure and then the read, which does not run
    when the measure stopped with a status: a cell that was refused wrote no record.

    `--phase all` REFUSES A PLAN `read` COULD NOT TAKE A READING FROM before any cell is
    measured: fewer than `ARMS_REQUIRED` arms, or fewer than `SEEDS_MINIMUM` seeds.
    `--phase measure` is allowed either (the milestone's smoke is one cell); `--phase
    read` is allowed a short SEED plan, because that is where `reading_ceiling`'s
    ValueError names the short arm."""
    args = _parser().parse_args(argv)
    if args.phase == "all":
        arms, seeds = _plan(args)
        require_readable_plan(arms)
        require_seeds_for_a_verdict(seeds)
    if args.phase in ("measure", "all"):
        status = measure_phase(args)
        if status != EXIT_OK:
            return status
    if args.phase in ("read", "all"):
        return read_phase(args)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
