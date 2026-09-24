"""The sharper latent over the M3c cells: sweep the rollout's sampling temperature, and -- only if the rollout is noise-limited -- retrain the nine cells at the temperature the sweep names.

M3g localised the M3 failure to the prior: from the true state, one prior step
predicts the next posterior's mode less often than "the mode did not change"
does. Its own disclosure, and the measurements in spec 2026-09-23 M3h section
1, say that failure is a near-tie inside a much larger problem -- the
posterior carries 0.71-0.87 nats per group over 32 groups, so every imagined
step injects ~23-28 nats of fresh entropy against ~0.4-0.6 nats of dynamics
information, and from h = 5 onward two draws of the SAME model from the SAME
state separate by more than that model's entire imagined displacement. So:

  sweep     re-run the canonical pass on each shipped cell at every rollout
            temperature of TAU_GRID, sharpening the prior inside `imagine`
            and nothing else. tau = 1.0 runs first and must reproduce the
            cell's diagnostic bitwise, or nothing below it is read.
  train     read the sweep's records, decide Reading N, refuse (35) unless it
            says NOISE_LIMITED, check the tau = 1.0 identity (37), then
            retrain nine cells at tau*.
  evaluate  score each retrained cell through the study's own evaluation half
            and the trust pass, rung-style -- a retrained cell has no prior
            diagnostic to self-check against, exactly as M3f's rungs had none.
  read      pool both phases, decide Reading M, print the tables, write
            sweep.txt / sharper.txt and the figures.

LOADING IS `trust_horizon.py`'S: `Cell`, `load_cell`, `self_check` and
`prepare_cell` are imported by path, so a cell is refused here for the reasons
and in the words the other tools refuse it. `prepare_cell` reads the
temperature off the args it is handed (M3h), so the model it returns samples
the way the checkpoint it loaded was trained to.

THE CHECKS, BY PHASE, each with its own status:

  sweep:    EXIT_NO_CHECKPOINTS (11)     a requested cell lacks its checkpoint,
                                          record or diagnostic; judged for every
                                          cell before any pass runs.
            EXIT_SPLIT_MISMATCH (12)      the split by name is not the record's.
            EXIT_RECORD_MISMATCH (14)     --context/--horizon disagree with the
                                          protocol, or evaluate_rollout no longer
                                          reproduces the record's curve.
            EXIT_SELF_CHECK_FAILED (30)   the tau = 1.0 pass is not bitwise the
                                          cell's diagnostic -- judged BEFORE any
                                          other temperature of that cell runs.
  train:    EXIT_NO_CHECKPOINTS (11)     a sweep record is missing.
            EXIT_NOT_NOISE_LIMITED (35)   NEW. Reading N does not say the rollout
                                          is noise-limited: the retrain is gated
                                          in code, not by a person reading a table.
            EXIT_IDENTITY_CHECK_FAILED (37) NEW. the tau = 1.0 retrain's losses are
                                          not the M3c record's prefix exactly.
  evaluate: EXIT_NO_CHECKPOINTS (11)     a retrained checkpoint is missing.
            EXIT_TEMPERATURE_MISMATCH (36) NEW. a checkpoint's recorded temperature
                                          is not the one it is being evaluated at.
  read:     EXIT_NO_CHECKPOINTS (11)     a requested record is missing.
            EXIT_SELF_CHECK_FAILED (30)   a sweep record's self-check is not ok.

11 / 12 / 14 / 30 carry `trust_horizon.py`'s meanings on purpose; 35, 36 and
37 are in no other tool's range (run_study 1/3-6/23, report_study 7-10, spike
10, diagnose 11-17, pool 18-22, trust 30, split_gap 31, ladder 32-33, stages
34, argparse 2, a traceback 1).

TWO CORRECTIONS TO THE M3H TASK-6 BRIEF:

  * `tau_key` does not format as `f"{tau:.3f}"` (e.g. "1.000"): that string
    contains a '.', and `study.write_record` refuses a record key that does
    -- the non-finite map addresses fields by dotted path, so a dotted key
    would alias a real nested field on `load_record` (`split_gap.q_key`'s
    docstring states the identical rule, for the same reason). The key is
    `q_key`'s own idiom instead, `0.7 -> "tau70"`, and `tau_value` is its
    inverse.

The brief also assumed `Trajectories` carried the pass's noise reference. It
did not, and reading it from a second `_diagnose` call would have doubled
every temperature's rollout compute; `diagnostics.Trajectories` now carries
`noise_embedding` (M3h), which is where this script reads it.
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
from mbfps.eval.sharper import (
    DECISION_H,
    REFERENCE_TAU,
    TAU_GRID,
)
from mbfps.eval.split_gap import decision_horizon
from mbfps.eval.study import SPLIT_SEED, git_sha, load_record, write_record
from mbfps.utils.config import ARMS
from mbfps.utils.device import get_device


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_sharper_latent", path)
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
EXIT_NOT_NOISE_LIMITED = 35
EXIT_TEMPERATURE_MISMATCH = 36
EXIT_IDENTITY_CHECK_FAILED = 37
"""0 / 11 / 12 / 14 / 30 are `trust_horizon.py`'s, imported so they cannot
drift; 35, 36 and 37 are new and in no other tool's range."""

_split = _sibling("split_gap")
stratum_summary = _split.stratum_summary

PHASES: tuple[str, ...] = ("sweep",)


# ---------------------------------------------------------------------------
# Paths and keys.
# ---------------------------------------------------------------------------


def sweep_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    return Path(out_dir) / f"sweep_{arm}_seed{seed}.json"


def tau_key(tau: float) -> str:
    """`0.7 -> "tau70"`, `split_gap.q_key`'s idiom: a record key may not
    contain '.', since `write_record` addresses non-finite fields by dotted
    path. Hundredths are the grid's own resolution, so 0.3 and 0.30 are one
    entry and a float's repr never becomes part of the record's shape."""
    return f"tau{int(round(float(tau) * 100))}"


def tau_value(key: str) -> float:
    """`"tau70" -> 0.7`: `tau_key`'s inverse, so a reader of a record can get
    back the temperature an entry was scored at without parsing the entry."""
    return int(key.removeprefix("tau")) / 100.0


# ---------------------------------------------------------------------------
# sweep: one cell at every temperature.
# ---------------------------------------------------------------------------


def _cell_args(args, source: Path, tau: float = REFERENCE_TAU) -> types.SimpleNamespace:
    """`prepare_cell` reads `out`, `device`, `context`, `horizon` and (M3h)
    `sample_temperature` off its args. The sweep asks for 1.0 -- it sharpens
    the ROLLOUT of a model trained at 1.0, not the model."""
    return types.SimpleNamespace(
        out=Path(source), device=args.device, context=args.context, horizon=args.horizon,
        sample_temperature=float(tau),
    )


def tau_entry(traj, record: dict, horizon: int) -> dict:
    """One temperature of one cell: the gate's own metric off the pass's own
    band (not the study record's -- every temperature has its own rollout),
    the probe's measurability, and the trust pass reduced by
    `stratum_summary`, the same reduction M3e and M3f read. The probe's
    selection R^2 is the STUDY RECORD's, which the reproduction discipline
    keeps equal to the probe `prepare_cell` refit, and is shared by every
    temperature because the probe does not depend on the rollout; its measurability is judged on THIS temperature's
    band, since the persistence-to-floor band is what makes the probe
    channel readable and the floor does not move but the model's curve does.
    `noise` is filled by the caller from the pass's noise reference."""
    summary = stratum_summary(traj, horizon)
    position = summary["band"]["position"]
    r2 = record["probe"]["embedding_selection_r2"]
    curves = summary["curves"]
    return {
        "gate": {
            "gap_final": position["gap_final"],
            "degenerate": position["steps_degenerate"],
        },
        "probe": {
            "selection_r2": float(r2) if r2 is not None else float("nan"),
            "measurable": bool(probe_is_measurable(
                {
                    "persistence": float(curves["persistence_position"][-1]),
                    "floor": float(curves["floor_position"][-1]),
                },
                widest_se=0.0,
            )),
        },
        "noise": {"curve": None, "median": None},
        "summary": summary,
    }


def sweep_cell(args, cell: Cell, device, train, val, taus) -> tuple[int, dict | None]:
    """One cell: 12/14 from `prepare_cell`, the reference temperature and its
    self-check (30) BEFORE any other temperature, then the rest of the grid.

    The reference is scored first because the caller's order says so, not
    because the pre-registered grid happens to list it first: a cell whose
    pass at the shipped temperature does not reproduce its diagnostic must
    cost one pass, not five, and on the real nine cells five passes is an
    hour. A grid without the reference temperature is refused before any of
    it runs, for the same reason -- every contrast is against it.
    """
    taus = [float(t) for t in taus]
    if REFERENCE_TAU not in taus:
        raise ValueError(
            f"the grid {taus} does not contain the reference temperature {REFERENCE_TAU}; "
            "every contrast is against it and the self-check is taken on it"
        )
    keys = [tau_key(t) for t in taus]
    if len(set(keys)) != len(keys):
        raise ValueError(
            f"the grid {taus} has two temperatures that spell the same record key "
            f"({keys}); one entry would overwrite the other and `taus` would list a "
            "temperature no entry was scored at"
        )
    status, prepared = prepare_cell(_cell_args(args, args.reference), cell, device, train, val)
    if status != EXIT_OK:
        return status, None
    h, _ = decision_horizon(prepared.horizon)
    entries: dict[str, dict] = {}
    check = None
    for tau in [REFERENCE_TAU] + [t for t in taus if t != REFERENCE_TAU]:
        traj = reference_trajectories(
            prepared.model, val, prepared.embedding_probe,
            rollout_temperature=float(tau), **prepared.common,
        )
        if float(tau) == REFERENCE_TAU:
            check = self_check(traj, cell.diagnostic)
            if not check.ok:
                print(
                    f"\nSELF-CHECK FAILED for {cell.arm} seed {cell.seed} at tau="
                    f"{REFERENCE_TAU}: " + "; ".join(check.failures())
                    + ". The pass at the shipped temperature is not the one the gate scored, "
                    "so no sharper temperature of this cell is evaluated. No record written."
                )
                return EXIT_SELF_CHECK_FAILED, None
        entry = tau_entry(traj, cell.record, prepared.horizon)
        noise = np.asarray(traj.noise_embedding, dtype=float)
        entry["tau"] = float(tau)
        entry["noise"] = {
            "curve": noise.mean(axis=0),
            "median": float(np.median(noise)),
        }
        entries[tau_key(tau)] = entry
    record = {
        "arm": cell.arm,
        "seed": int(cell.seed),
        "source": str(args.reference),
        "step": int(cell.record["steps"]),
        "record_git_sha": cell.record.get("git_sha", "unknown"),
        "context": int(prepared.context),
        "horizon": int(prepared.horizon),
        "decision_h": int(h),
        "split_seed": SPLIT_SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "git_sha": git_sha(),
        "taus": [float(t) for t in taus],
        "episodes": {"val": list(cell.record["episodes"]["val"])},
        "windows": {
            "total": int(entries[tau_key(REFERENCE_TAU)]["summary"]["windows"]["total"]),
            "episode": [
                int(e) for e in entries[tau_key(REFERENCE_TAU)]["summary"]["windows"]["episode"]
            ],
            "clusters": int(
                entries[tau_key(REFERENCE_TAU)]["summary"]["windows"]["clusters"]
            ),
        },
        "self_check": check.record(),
        "entries": entries,
    }
    path = sweep_record_path(args.sweep_out, cell.arm, cell.seed)
    write_record(path, record)
    gaps = " / ".join(
        f"tau {entries[tau_key(t)]['tau']:.1f}: gap {entries[tau_key(t)]['gate']['gap_final']:+.3f}"
        for t in taus
    )
    print(f"{cell.arm} seed {cell.seed}: {gaps}; wrote {path}")
    return EXIT_OK, record


def sweep_phase(args, cells, device, train, val, taus) -> int:
    try:
        loaded = [load_cell(args.reference, arm, seed) for arm, seed in cells]
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    args.sweep_out.mkdir(parents=True, exist_ok=True)
    for cell in loaded:
        status, _ = sweep_cell(args, cell, device, train, val, taus)
        if status != EXIT_OK:
            return status
    return EXIT_OK


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/m3h_sharper"))
    parser.add_argument("--sweep-out", type=Path, default=Path("runs/m3h_sweep"))
    parser.add_argument("--reference", type=Path, default=Path("runs/m3_study_v2"),
                        help="the M3c study directory: the nine 20,000-step cells")
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--device", default="mps")
    # None -> the cell's diagnostic says what it was written at; a value that
    # disagrees is refused (`protocol_mismatch`, 14).
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--phase", choices=PHASES, default=PHASES[-1])
    parser.add_argument("--figure", type=Path, default=None,
                        help="the curves figure; default <sweep-out>/sweep_curves.png")
    return parser


def _check_usage(parser: argparse.ArgumentParser, args) -> None:
    for name, other in (("--reference", args.reference), ("--sweep-out", args.sweep_out)):
        if args.out.resolve() == Path(other).resolve():
            parser.error(
                f"--out and {name} are the same directory; the retrain would write its "
                "checkpoints over the ones it is being compared against"
            )


def main(argv: list[str] | None = None, *, taus=TAU_GRID) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _check_usage(parser, args)
    device = get_device(prefer=args.device)
    cells = [(arm, int(seed)) for arm in args.arms for seed in args.seeds]
    if args.phase in ("sweep", "all"):
        buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
        train, val = episode_split(buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
        status = sweep_phase(args, cells, device, train, val, taus)
        if status != EXIT_OK:
            return status
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
