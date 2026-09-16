"""The split gap over the M3c cells: load, check, three strata per cell, one record per cell; then pool, read the gap, draw the learning curves.

Every rollout number the M3 study recorded is on the 24 held-out episodes. The
20,000-step budget is ~430 passes over the 98 training episodes, and whether
the h=45 failure is memorisation has never been measured. This script
evaluates each frozen checkpoint on THREE strata of episodes under the ONE
refit probe its record was scored with:

  val          the record's 24 held-out episodes -- the self-check anchor
  train_held   train[PROBE_EPISODE_LIMIT:] -- 78 episodes the model trained on
               and the probe never saw -- THE DECISION STRATUM
  train_probe  train[:PROBE_EPISODE_LIMIT] -- the probe's own 20; confounded
               by construction and labelled so wherever it is printed

and writes one `split_gap_<arm>_seed<n>.json` per cell holding, per stratum,
the band and its gap_closed, the crossing step in both channels, the
persistence margin in both channels, the survival curve and H*_q, and the
counts -- `mbfps.eval.split_gap.stratum_summary`. After the per-cell loop,
`main` pools the LIVE records, decides Reading G (spec 3.3), prints the
sensitivity lines and the learning-curve table, draws `learning_curves.png`
and writes `split_gap.txt`.

LOADING IS `trust_horizon.py`'S, NOT A SECOND COPY. `Cell`, `load_cell`,
`self_check`, `protocol_mismatch` and the checkpoint/diagnostic paths are
imported from that script by path -- the way every script test loads a
script, and the way trust_horizon itself imports diagnose_dynamics -- so the
three tools read one cell through one loader.

THE CHECKS RUN IN A FIXED ORDER and each has its own status:

  EXIT_NO_CHECKPOINTS (11)          a REQUESTED cell lacks its checkpoint,
                                    study record or diagnostic. Every cell is
                                    loaded before anything else.
  EXIT_STRATA_NOT_A_PARTITION (31)  NEW. The three strata are not a partition
                                    of the buffer, `train_probe` is not what
                                    `probe_episodes` hands `fit_probes`, or
                                    `train_held` is empty. Judged ONCE, up
                                    front, before any refit: a code or
                                    fixture defect, never a data one.
  EXIT_SPLIT_MISMATCH (12)          the split by name is not the record's.
  EXIT_RECORD_MISMATCH (14)         `evaluate_rollout` on val no longer
                                    reproduces the record's curves.rssm_position
                                    (an ENVIRONMENT difference: cpu misses by
                                    6-12 map units), or --context/--horizon
                                    disagree with the diagnostic's protocol.
  EXIT_SELF_CHECK_FAILED (30)       the val stratum's mean curves are not
                                    bitwise the diagnostic's, or its windows
                                    are not the diagnostic's. Same windows,
                                    same rollout, same refit probe -- or the
                                    train strata are not read against the
                                    ruler M3d validated.

11 / 12 / 14 / 30 carry `trust_horizon.py`'s meanings on purpose; 31 is in no
other tool's range (run_study 1/3-6/23, report_study 7-10, spike 10, diagnose
11-17, pool 18-22, trust 30, argparse 2, a traceback 1).
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

import mbfps.eval.pooling as pooling
from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.aggregate import SEEDS
from mbfps.eval.diagnostics import Trajectories, reference_trajectories
from mbfps.eval.probe import probe_episodes
from mbfps.eval.split_gap import (
    CHANNELS,
    DECISION,
    DECISION_H,
    FAMILY,
    REPORTED_H,
    STRATA,
    TERMS,
    ArmInputs,
    GapInputs,
    StrataNotAPartition,
    StratumContrast,
    format_reading_gap,
    learning_curve_summary,
    q_key,
    reading_gap,
    strata_partition,
    stratum_summary,
)
from mbfps.eval.study import SPLIT_SEED, git_sha, write_record
from mbfps.eval.trust import survival, trust_horizon
from mbfps.eval.trust_readings import ARMS_ORDER, Q_REPORTED
from mbfps.models.rssm import KL_FREE_BITS
from mbfps.utils.config import ARMS
from mbfps.utils.device import get_device


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_split_gap", path)
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

EXIT_OK = _trust.EXIT_OK
EXIT_NO_CHECKPOINTS = _trust.EXIT_NO_CHECKPOINTS
EXIT_SPLIT_MISMATCH = _trust.EXIT_SPLIT_MISMATCH
EXIT_RECORD_MISMATCH = _trust.EXIT_RECORD_MISMATCH
EXIT_SELF_CHECK_FAILED = _trust.EXIT_SELF_CHECK_FAILED
EXIT_STRATA_NOT_A_PARTITION = 31
"""0 / 11 / 12 / 14 / 30 are `trust_horizon.py`'s, imported so they cannot
drift; 31 is new and in no other tool's range."""


# ---------------------------------------------------------------------------
# One cell: the record.
# ---------------------------------------------------------------------------


def split_gap_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    """One file per cell, named by BOTH the arm and the seed."""
    return Path(out_dir) / f"split_gap_{arm}_seed{seed}.json"


def split_gap_record(
    cell: Cell, strata_traj: dict, strata_paths: dict, check, *, context, horizon, device
) -> dict:
    """The LIVE record for one cell: numpy arrays and real NaNs; `write_record`
    sanitises it on the way to disk, so no top-level `nonfinite` key here."""
    curves = cell.diagnostic["curves"]
    return {
        "arm": cell.arm,
        "seed": cell.seed,
        "context": int(context),
        "horizon": int(horizon),
        "split_seed": SPLIT_SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "git_sha": git_sha(),
        "checkpoint_git_sha": str(cell.record.get("git_sha", "unknown")),
        "probe": {
            "selection_r2": float(cell.diagnostic["probe"]["embedding_selection_r2"]),
            "measurable": bool(probe_is_measurable(
                {
                    "persistence": curves["persistence_position"][-1],
                    "floor": curves["floor_position"][-1],
                },
                widest_se=0.0,
            )),
        },
        "episodes": {name: [p.name for p in strata_paths[name]] for name in STRATA},
        "self_check": check.record(),
        "strata": {name: stratum_summary(strata_traj[name], horizon) for name in STRATA},
    }


def write_split_gap_record(out_dir: Path, record: dict) -> Path:
    path = split_gap_record_path(out_dir, record["arm"], record["seed"])
    write_record(path, record)
    return path


# ---------------------------------------------------------------------------
# The driver.
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/m3_study_v2"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    # None -> each cell's diagnostic says what it was written at; a value that
    # disagrees is refused (`protocol_mismatch`, 14).
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--figure", type=Path, default=None,
                        help="learning-curve figure; default <out>/learning_curves.png")
    parser.add_argument("--window", type=int, default=100,
                        help="moving-mean window (steps) for the learning curves")
    return parser


def _run_cell(args, cell: Cell, device, train, val, strata: dict) -> tuple[int, dict | None]:
    """One cell: trust_horizon's checks and refit (12, 14, 14) through
    `prepare_cell`, then the val pass and 30, the two train passes, the record."""
    arm, seed = cell.arm, cell.seed
    status, prepared = prepare_cell(args, cell, device, train, val)
    if status != EXIT_OK:
        return status, None
    model, embedding_probe, common = prepared.model, prepared.embedding_probe, prepared.common

    # val FIRST: it is the anchor, and a failed self-check costs no train pass.
    traj = {"val": reference_trajectories(model, val, embedding_probe, **common)}
    check = self_check(traj["val"], cell.diagnostic)
    if not check.ok:
        print(
            f"\nSELF-CHECK FAILED for {arm} seed {seed}: " + "; ".join(check.failures())
            + ". Same windows, same rollout, same refit probe -- or the train strata "
            "would be read against a ruler M3d did not validate. No record written."
        )
        return EXIT_SELF_CHECK_FAILED, None
    for name in ("train_held", "train_probe"):
        traj[name] = reference_trajectories(model, strata[name], embedding_probe, **common)

    record = split_gap_record(
        cell, traj, strata, check,
        context=prepared.context, horizon=prepared.horizon, device=device,
    )
    path = write_split_gap_record(args.out, record)
    counts = "; ".join(
        f"{name} {traj[name].windows_total} windows / {len(strata[name])} episodes"
        for name in STRATA
    )
    print(
        f"{arm} seed {seed}: {counts}; self-check max|delta| reference "
        f"{check.reference_position_max_delta:.1e} persistence "
        f"{check.persistence_position_max_delta:.1e}; wrote {path}"
    )
    return EXIT_OK, record


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = get_device(prefer=args.device)
    try:
        cells = [load_cell(args.out, arm, seed) for arm in args.arms for seed in args.seeds]
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS

    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    train, val = episode_split(
        buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED
    )
    # The strata are the SAME for every cell (one split, one probe rule), so
    # they are built and judged once, before the first refit.
    used, _ = probe_episodes(train)
    try:
        strata = {"val": val, **strata_partition(buffer.episode_paths(), train, val, used)}
    except StrataNotAPartition as error:
        print(f"STRATA NOT A PARTITION: {error}")
        return EXIT_STRATA_NOT_A_PARTITION

    records: dict[tuple[str, int], dict] = {}
    horizon = 0
    for cell in cells:
        status, record = _run_cell(args, cell, device, train, val, strata)
        if status != EXIT_OK:
            return status
        records[(cell.arm, cell.seed)] = record
        horizon = int(record["horizon"])
    # Task 6 appends the pooling, Reading G, the learning curves and split_gap.txt here.
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
