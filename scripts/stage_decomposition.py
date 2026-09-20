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
trustworthy = _trust.trustworthy
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

PHASES: tuple[str, ...] = ("evaluate", "read", "all")
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
    # `rung_cell` is `checkpoint_ladder.py`'s own, and that script loads
    # `trust_horizon.py` by path itself -- a second, independent execution of
    # the same file, so its `CellMissing` is a distinct class object from
    # this module's own `_trust.CellMissing` even though both are raised for
    # the same reason. A control's missing checkpoint must be caught too.
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
            if record["kind"] != kind or record["arm"] != arm or int(record["seed"]) != int(seed):
                raise ValueError(
                    f"{path}: the record inside is {record['kind']} {record['arm']} seed "
                    f"{record['seed']}, not the {kind} {arm} seed {seed} its name says"
                )
            records[(str(record["label"]), int(seed))] = record
    return EXIT_OK, records


def read_phase(args, cells, controls) -> int:
    """Load every record (11), refuse one whose self-check is not ok or
    whose identity is unset (30), refuse records at different protocols,
    read the control FIRST: its encode must read failed (34), then the
    figure, the text, `stages.txt`."""
    status, records = load_records(args, cells, controls)
    if status != EXIT_OK:
        return status
    for (label, seed), r in records.items():
        if r["kind"] == "cell" and not trustworthy(r["self_check"]):
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
    encode = control.stages["encode"]
    if control.status is not Status.ENCODE_FAILS or encode.wording != "failed":
        print(control_text, end="")
        print(
            f"\nCONTROL MISREAD: the known-blind control reads {control.status.name.replace('_', ' ')} with "
            f"encode {encode.wording!r} -- decided by: {control.rule}. The control's encode must read failed "
            "-- a clear the wrong way or a non-positive estimate, not merely not passed -- because a "
            "posterior equal to its prior injects no information at all; a not-shown encode is absence of "
            "evidence and validates nothing. No arm's reading is printed and stages.txt is not written."
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
    if args.phase in ("read", "all"):
        return read_phase(args, cells, controls)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
