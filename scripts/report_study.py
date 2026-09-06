"""Print the nine-cell table, the M3 gate verdict, and the filtering gain.

THIS IS THE LAST THING BETWEEN A 33-HOUR STUDY AND THE NUMBER A HUMAN READS,
so it is arranged around what a reader can be misled by rather than around what
is convenient to print.

  * EVERY PER-SEED COLUMN IS INDEXED BY ITS SEED, never by position in a list.
    An arm with seed 1 missing has two gaps in `gap_final_by_seed`, and
    printing that list under headers "seed 0 / seed 1 / seed 2" puts seed 2's
    number in seed 1's column -- a full-looking table describing a study nobody
    ran. A cell with no record prints MISSING in its own column.
  * AN INCOMPLETE STUDY IS OBVIOUS. The record count, the MISSING cells and
    the failing `all_nine_cells_present` criterion all say so, and the gate
    cannot pass without all nine.
  * A FAILING CRITERION IS PRINTED AS LOUDLY AS A PASSING ONE, and the
    criteria that CANNOT be computed from the records are printed too, by name,
    as NOT EVALUATED. Criterion 4 fails on the real trained model; the correct
    output for that study is a legible NOT PASSED, not a softer arrangement of
    the same numbers.
  * THE FILTERING GAIN IS PRINTED WITH ITS INTERVAL AND ITS RIDGE. The gain's
    SIGN moves with the selected ridge decade -- measured -0.0706, -0.0208 and
    +0.0325 for one model under three defensible selection designs -- so nine
    cells averaged across different decades is a mean over different
    estimators. The grouping is printed whether or not the cells agree, and a
    disagreement is called out in words.

THE DENOMINATOR OF EVERY MEAN IS PRINTED BESIDE IT. `gap_closed` is NaN by
contract whenever the persistence-to-floor band is non-positive, so a mean over
three seeds can be a mean over one; the `finite` column says which. See
`mbfps.eval.aggregate`'s module docstring for the whole degenerate-record
policy -- this script only renders it.
"""

import argparse
import math
import textwrap
from pathlib import Path

from mbfps.eval.aggregate import (
    ARMS,
    CRITERIA_NOT_EVALUATED_HERE,
    GATE_CRITERIA,
    GATE_METRIC,
    MAX_DEGENERATE_STEPS,
    RECORD_GLOB,
    SEEDS,
    MislabelledRecord,
    UnreadableRecord,
    evaluate_gate,
    load_records,
    mean_curve,
    record_cell,
)
from mbfps.eval.summary import METRICS

EXIT_OK = 0
EXIT_NO_RECORDS = 7
EXIT_MISLABELLED = 8
EXIT_UNREADABLE = 9
EXIT_GATE_NOT_PASSED = 10
"""The five statuses this script can end on. All distinct, NONE OF THEM IS 2,
and none of them collides with `scripts/run_study.py`'s.

2 is argparse's own usage status and 1 is what an uncaught traceback exits
with; a wrapper that cannot tell "the command line was wrong" from "the gate
did not pass" is a wrapper that reports a milestone on the strength of a typo.

3 to 6 are skipped because the driver already uses them --
`EXIT_CONFIG_MISMATCH` 3, `EXIT_LOCKED` 4, `EXIT_NO_DATA` 5,
`EXIT_OUT_UNUSABLE` 6. The two scripts run one after another in the same shell,
often into the same wrapper, and a status is worth more when it means one thing
across both. This numbering started at 5 and the cross-check in
`test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own` is what
found "no episodes on the box" and "no records in --out" sharing a status.

`EXIT_GATE_NOT_PASSED` IS NOT AN ERROR. The report is printed in full first and
the verdict is legible on its own line; the status exists so an unattended
wrapper can tell a milestone that did not pass from a report that could not be
produced. A study that fails its gate is a result, and this status is how that
result reaches a script.
"""


def _fmt(value, spec: str = ".4f") -> str:
    """Format a number that may legitimately be non-finite, None, or missing.

    A deliberate twin of `scripts/run_study.py::_fmt`, for the same reason: NaN
    is a NORMAL value in these records -- `gap_final` is NaN by contract
    whenever the band is non-positive and `reward.r2` is NaN when the target
    has no variance -- while `None` is what a truncated or half-converted
    record holds where a number belongs, and `format(None, "+.4f")` raises
    `unsupported format string passed to NoneType.__format__`. Raising here
    would lose the whole report to one bad field after the study is already
    paid for.
    """
    if value is None:
        return "n/a"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _count(value) -> str:
    """Render an integer count that may be NaN because no record carried it."""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if math.isnan(number) else f"{int(number)}"


def _flagtext(value) -> str:
    """Render a boolean the record may not carry.

    "n/a" for a missing flag rather than Python's `None`: a column reading
    `None` invites the eye to read it as `False`, and "the record does not say"
    is not "the record says no" -- only the criteria collapse those two, and
    they collapse them into a FAIL.
    """
    if isinstance(value, bool):
        return str(value)
    return "n/a" if value is None else str(value)


def _cells(records: list[dict]) -> dict:
    """`{(arm, seed): record}` -- the lookup every per-seed column indexes by."""
    return {
        cell: record
        for cell, record in ((record_cell(r), r) for r in records)
        if cell is not None
    }


def _get(node, *keys):
    """`node[k1][k2]...`, or None if any step is missing or not a mapping.

    The twin of `scripts/run_study.py::_get`, and it earns its place the same
    way: this is the PRINTING path, and it must never be the thing that kills
    the report after the study is paid for. BOTH HALVES CARRY WEIGHT. `key not
    in node` covers a record one field short; `not isinstance(node, dict)`
    covers a record where a whole block has been replaced by a scalar or a
    `null`, against which `key not in node` raises `TypeError` rather than
    answering.

    Used for the per-arm summary too -- `_get(summary, arm, ...)` returns None
    for an arm that produced no record at all, which is what the MISSING and
    n/a markers in every table are rendered from.
    """
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def metric_table(records, summary, arms=ARMS, seeds=SEEDS) -> str:
    """The 3x3 table, one row per arm and metric, indexed by seed.

    `gap_closed` is the dimensionless fraction of the persistence-to-floor band
    the model closed, so it needs no units; the columns that do are named in
    the header. Position is the gate metric (spec section 4 criterion 1) and
    angle is reported beside it, labelled, never folded into it.
    """
    cells = _cells(records)
    header = (
        f"{'arm':<12}{'metric':<10}"
        + "".join(f"{f'seed {s}':>12}" for s in seeds)
        + f"{'mean':>12}{'finite':>8}{'unanimous':>11}"
        f"{'degenerate':>12}{'floor>=pers':>13}"
    )
    lines = [header]
    for arm in arms:
        for metric in METRICS:
            row = f"{arm:<12}{metric:<10}"
            for seed in seeds:
                record = cells.get((arm, seed))
                value = (
                    "MISSING"
                    if record is None
                    else _fmt(_get(record, metric, "gap_final"), "+.4f")
                )
                row += f"{value:>12}"
            block = ("metrics", metric)
            row += f"{_fmt(_get(summary, arm, *block, 'gap_final_mean'), '+.4f'):>12}"
            # The denominator is how many cells the arm SHOULD have, not how
            # many it has: with seed 1 missing, "2/2" reads as complete while
            # "2/3" reads as what it is. Which of the two a cell failed at --
            # absent, or present and undefined -- is what the MISSING marker in
            # its own column above says.
            finite = _get(summary, arm, *block, "n_seeds_with_finite_gap")
            # An arm with no record at all really did produce zero finite gaps
            # out of the three it owed, and "0/3" is what that says; "n/a/3"
            # was unreadable and the three MISSING markers to its left already
            # say which of the two ways each cell failed.
            row += f"{f'{0 if finite is None else _count(finite)}/{len(seeds)}':>8}"
            row += f"{_flagtext(_get(summary, arm, *block, 'all_seeds_positive')):>11}"
            row += f"{_count(_get(summary, arm, *block, 'max_degenerate_steps')):>12}"
            floor_above = _get(
                summary, arm, *block, "max_steps_floor_above_persistence")
            row += f"{_count(floor_above):>13}"
            lines.append(row)
    return "\n".join(lines)


def training_table(records, arms=ARMS, seeds=SEEDS) -> str:
    """Throughput and the KL rate, per cell, so a dead arm is visible."""
    cells = _cells(records)
    lines = [f"{'cell':<16}{'steps/s':>10}{'kl_rate':>10}{'kl_dyn_max':>12}"
             f"{'loss_last20':>13}{'wall_h':>9}"]
    for arm in arms:
        for seed in seeds:
            record = cells.get((arm, seed))
            label = f"{arm}/s{seed}"
            if record is None:
                lines.append(f"{label:<16}{'MISSING':>10}")
                continue
            seconds = _get(record, "seconds")
            try:
                hours = float(seconds) / 3600.0
            except (TypeError, ValueError):
                # Whatever is there is handed to `_fmt`, which renders "n/a"
                # for None and the value itself for anything else it cannot
                # make a float of. A wall-clock column is not worth a crash.
                hours = seconds
            lines.append(
                f"{label:<16}{_fmt(_get(record, 'steps_per_second'), '.2f'):>10}"
                f"{_fmt(_get(record, 'kl_rate_above_free_bits'), '.3f'):>10}"
                f"{_fmt(_get(record, 'kl_dyn_max'), '.3f'):>12}"
                f"{_fmt(_get(record, 'loss_last20'), '.4f'):>13}"
                f"{_fmt(hours, '.2f'):>9}"
            )
    return "\n".join(lines)


def criterion_4_table(records, arms=ARMS, seeds=SEEDS) -> str:
    """Gate criterion 4 per cell: the posterior latent against the raw
    embedding of the same frame.

    Its own table, beside `gain_table` and never merged into it. The two arms
    of this comparison are NOT matched -- the latent is `h` plus a 32x32
    categorical `z`, at most 160 bits, against 2048 continuous floats -- so a
    model can carry real history in `h` and still lose here on the bottleneck
    alone. That is why the gain is reported too, and why reading the two
    tables' R^2 columns as if they were the same quantity would be wrong.
    """
    cells = _cells(records)
    lines = [f"{'cell':<16}{'latent_r2':>13}{'embed_r2':>13}"
             f"{'latent_beats_embedding':>25}"]
    for arm in arms:
        for seed in seeds:
            record = cells.get((arm, seed))
            label = f"{arm}/s{seed}"
            if record is None:
                lines.append(f"{label:<16}{'MISSING':>13}")
                continue
            criterion = _get(record, "filtering", "criterion_4") or {}
            lines.append(
                f"{label:<16}{_fmt(_get(criterion, 'latent_r2'), '+.4f'):>13}"
                f"{_fmt(_get(criterion, 'embedding_r2'), '+.4f'):>13}"
                f"{_flagtext(_get(criterion, 'latent_beats_embedding')):>25}"
            )
    return "\n".join(lines)


def gain_table(records, arms=ARMS, seeds=SEEDS) -> str:
    """The bottleneck-free companion: R2([e_t + h_t]) - R2([e_t]), per cell.

    BOTH LEVELS ARE PRINTED, not only their difference: the gain is ~0.02 while
    the levels are ~0.31, and a reader cannot tell a small difference between
    two good probes from a small difference between two useless ones without
    seeing them. The interval and the two ridges are in the row for the same
    reason -- the gain's SIGN moves with the selected decade.
    """
    cells = _cells(records)
    lines = [
        f"{'cell':<16}{'joint_r2':>12}{'embed_r2':>12}{'gain':>12}"
        f"{'95% CI':>27}{'CI excl 0':>11}"
        f"{'joint_ridge':>13}{'embed_ridge':>13}{'selected':>10}{'windows':>9}"
    ]
    for arm in arms:
        for seed in seeds:
            record = cells.get((arm, seed))
            label = f"{arm}/s{seed}"
            if record is None:
                lines.append(f"{label:<16}{'MISSING':>12}")
                continue
            gain = _get(record, "filtering", "gain") or {}
            low, high = _get(gain, "ci_low"), _get(gain, "ci_high")
            interval = f"[{_fmt(low, '+.4f')}, {_fmt(high, '+.4f')}]"
            lines.append(
                f"{label:<16}{_fmt(_get(gain, 'joint_r2'), '+.4f'):>12}"
                f"{_fmt(_get(gain, 'embedding_r2'), '+.4f'):>12}"
                f"{_fmt(_get(gain, 'gain'), '+.4f'):>12}"
                f"{interval:>27}{_excludes_zero(low, high):>11}"
                f"{_fmt(_get(gain, 'joint_ridge'), '.1e'):>13}"
                f"{_fmt(_get(gain, 'embedding_ridge'), '.1e'):>13}"
                f"{_flagtext(_get(gain, 'ridge_selected')):>10}"
                f"{_count(_get(gain, 'n_scored_windows')):>9}"
            )
    return "\n".join(lines)


def _excludes_zero(low, high) -> str:
    """Whether a confidence interval is entirely on one side of zero.

    The reader's actual question about a gain of -0.02: is the interval clear
    of zero, and on which side. "n/a" when either end is missing or undefined,
    because an interval with one end is not an interval -- rendering that as
    `False` would say "the interval covers zero", which nobody measured.
    """
    try:
        low, high = float(low), float(high)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(low) or math.isnan(high):
        return "n/a"
    return str(bool(low > 0.0 or high < 0.0))


def ridge_block(groups: list[dict]) -> str:
    """The cells grouped by the ridge decade their gain was computed at.

    Printed even when all nine agree, because "they all selected 1e3" is itself
    the disclosure a reader needs to interpret the mean; and when they do NOT
    agree the mean is called out in words as a mean over different estimators,
    since there is no correction for it, only the disclosure.
    """
    lines = [
        f"{'selected':<10}{'joint_ridge':>13}{'embed_ridge':>13}{'cells':>7}"
        f"{'gain_mean':>11}  members"
    ]
    for group in groups:
        members = " ".join(f"{arm}/s{seed}" for arm, seed in group["cells"])
        lines.append(
            f"{_flagtext(group['ridge_selected']):<10}"
            f"{_fmt(group['joint_ridge'], '.1e'):>13}"
            f"{_fmt(group['embedding_ridge'], '.1e'):>13}"
            f"{group['n_cells']:>7}"
            f"{_fmt(group['gain_mean'], '+.4f'):>11}  {members}"
        )
    if len(groups) > 1:
        lines.append(
            "WARNING: the nine cells did not all select the same ridge decade. "
            "The scored R^2 moves ~0.10 per decade while the gain is ~0.02, so "
            "a mean across these groups is a mean over different estimators "
            "and its SIGN is not interpretable. Read the groups, not the mean."
        )
    if any(group["ridge_selected"] is not True for group in groups):
        lines.append(
            "WARNING: at least one cell computed its gain with NO ridge "
            "selection (fit_probe's default penalty). That estimate is "
            "unbiased but weaker, and it is not the same estimator as a "
            "selected one."
        )
    return "\n".join(lines)


def reward_table(records, arms=ARMS, seeds=SEEDS) -> str:
    """Spec criterion 3: reward-prediction accuracy REPORTED, per cell.

    `degenerate_target` is printed and is deliberately NOT a gate criterion:
    `my_way_home`'s reward is near-constant by construction -- 19,417 of 19,424
    steps share the living penalty -- so gating on it would fail the milestone
    for a property of the scenario that was known before the study started. The
    baseline sits beside the MSE for the same reason: an MSE over a constant
    target looks precise and means nothing without it.
    """
    cells = _cells(records)
    lines = [f"{'cell':<16}{'mse':>14}{'baseline_mse':>16}{'r2':>13}"
             f"{'events':>9}{'steps':>9}{'degenerate_target':>19}"]
    for arm in arms:
        for seed in seeds:
            record = cells.get((arm, seed))
            label = f"{arm}/s{seed}"
            if record is None:
                lines.append(f"{label:<16}{'MISSING':>14}")
                continue
            reward = _get(record, "reward") or {}
            lines.append(
                f"{label:<16}{_fmt(_get(reward, 'mse'), '.5f'):>14}"
                f"{_fmt(_get(reward, 'baseline_mse'), '.5f'):>16}"
                f"{_fmt(_get(reward, 'r2'), '+.4f'):>13}"
                f"{_count(_get(reward, 'n_reward_events')):>9}"
                f"{_count(_get(reward, 'n_steps')):>9}"
                f"{_flagtext(_get(reward, 'is_degenerate')):>19}"
            )
    return "\n".join(lines)


def gate_block(verdict: dict) -> str:
    """Every criterion with its own verdict, then the conjunction.

    The criteria that cannot be computed from the records are printed by name
    as NOT EVALUATED. Printing only the computable ones is how a five-criterion
    gate gets reported as a four-criterion pass.
    """
    lines = []
    for name in GATE_CRITERIA:
        ok = verdict["criteria"][name]
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    for name, why in verdict.get("not_evaluated", {}).items():
        lines.append(f"  [ n/a] {name}")
        # Wrapped: this is the one multi-sentence explanation in the report and
        # a 400-character line in a terminal is a line nobody reads.
        lines.append(textwrap.fill(
            why, width=86, initial_indent="         ",
            subsequent_indent="         "))
    if verdict["missing_cells"]:
        listing = ", ".join(f"{arm}/s{seed}" for arm, seed in verdict["missing_cells"])
        lines.append(
            f"  MISSING CELLS ({len(verdict['missing_cells'])} of "
            f"{verdict['n_expected']}): {listing}"
        )
    if verdict["duplicate_cells"]:
        listing = ", ".join(
            f"{arm}/s{seed}" for arm, seed in verdict["duplicate_cells"]
        )
        lines.append(f"  DUPLICATE CELLS: {listing}")
    if verdict["unexpected_cells"]:
        lines.append(
            "  RECORDS THAT ARE NOT STUDY CELLS: "
            + ", ".join(verdict["unexpected_cells"])
        )
    lines.append("")
    lines.append(f"GATE: {'PASSED' if verdict['passed'] else 'NOT PASSED'}")
    return "\n".join(lines)


def report(records: list[dict], verdict: dict, out_dir, arms=ARMS, seeds=SEEDS) -> str:
    """The whole report, in the order a reader needs it.

    The per-arm summary comes out of the VERDICT rather than being recomputed
    here, so the table and the criteria above it cannot be two views of two
    different reductions of the same records.
    """
    summary = verdict["per_arm"]
    expected = len(arms) * len(seeds)
    lines = [
        "=== M3 study report ===",
        f"  --out       {out_dir}",
        f"  records     {len(records)} of {expected} expected cells "
        f"({len(arms)} arms x {len(seeds)} seeds)",
        f"  gate metric {GATE_METRIC}; degenerate bands allowed per arm: "
        f"{MAX_DEGENERATE_STEPS}",
        "  means skip cells whose gap_closed is undefined (NaN by contract "
        "when the",
        "  persistence-to-floor band is non-positive); the finite column is "
        "the denominator",
        "  each mean was taken over. A degenerate band is never dropped from a "
        "mean -- it",
        "  fails band_is_usable instead.",
        "",
        "--- gap_closed at the final horizon step, by arm and seed "
        "(fraction of the persistence-to-floor band) ---",
        metric_table(records, summary, arms, seeds),
        "",
        "--- training ---",
        training_table(records, arms, seeds),
        "",
        "--- filtering, gate criterion 4: does the posterior latent beat the "
        "raw encoder embedding of the same frame? ---",
        criterion_4_table(records, arms, seeds),
        "",
        "--- filtering, the bottleneck-free companion: what appending h to "
        "that same embedding buys ---",
        gain_table(records, arms, seeds),
        "",
        "--- the ridge decade each gain was computed at ---",
        ridge_block(verdict["ridge_groups"]),
        "",
        "--- reward prediction (spec criterion 3: reported per arm, and "
        "deliberately not gated -- the target is near-constant by "
        "construction) ---",
        reward_table(records, arms, seeds),
        "",
        "--- M3 exit gate (spec section 4) ---",
        gate_block(verdict),
    ]
    return "\n".join(lines)


def write_figure(records: list[dict], figure, arms=ARMS) -> str:
    """Draw the error-vs-horizon curves, model against persistence and floor.

    Spec criterion 2 asks for the curve of each arm with both baselines on the
    SAME axes; drawing the model alone would make an arm that never beat
    persistence look like a success story. Returns a line for the report.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = Path(figure)
    def arm_of(record):
        cell = record_cell(record)
        return cell[0] if cell is not None else None

    drawn = [arm for arm in arms if any(arm_of(r) == arm for r in records)]
    if not drawn:
        return "figure NOT written: no arm has a record"
    fig, axes = plt.subplots(
        1, len(drawn), figsize=(5 * len(drawn), 4), squeeze=False
    )
    try:
        for ax, arm in zip(axes[0], drawn):
            rows = [r for r in records if arm_of(r) == arm]
            for name, style in (
                (f"persistence_{GATE_METRIC}", "--"),
                (f"rssm_{GATE_METRIC}", "-"),
                (f"floor_{GATE_METRIC}", ":"),
            ):
                curve = mean_curve(rows, name)
                ax.plot(
                    range(1, len(curve) + 1), curve, style,
                    label=name.replace(f"_{GATE_METRIC}", ""),
                )
            ax.set_title(f"{arm} (mean of {len(rows)} seed(s))")
            ax.set_xlabel("imagined step")
            ax.set_ylabel(f"{GATE_METRIC} error (Doom map units)")
            ax.legend()
        fig.tight_layout()
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=110)
    except (ValueError, OSError) as error:
        # A ragged curve set or an unwritable path must cost the FIGURE, not
        # the gate verdict that has already been printed above it.
        return f"figure NOT written: {error}"
    finally:
        plt.close(fig)
    return f"figure={figure}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # --out and --figure stay STRINGS, exactly as `scripts/run_study.py` keeps
    # them: everything downstream coerces, and leaving them as argparse hands
    # them over means the report is exercised on the type the command line
    # really produces rather than on a `Path` the tests invented.
    parser.add_argument("--out", default="runs/m3_study")
    parser.add_argument("--figure", default=None)
    return parser


def main(argv=None) -> int:
    """Print the report. Returns the process exit status."""
    args = _parser().parse_args(argv)
    try:
        records = load_records(args.out)
    except MislabelledRecord as error:
        print(error, flush=True)
        return EXIT_MISLABELLED
    except UnreadableRecord as error:
        print(error, flush=True)
        return EXIT_UNREADABLE
    if not records:
        # "the path is wrong" and "the study has not run" want opposite
        # responses from the operator, and both used to print the same line.
        where = Path(args.out)
        why = ("does not exist" if not where.exists()
               else "is not a directory" if not where.is_dir()
               else f"holds no {RECORD_GLOB}")
        print(
            f"nothing to report: --out {args.out} {why}. Point --out at the "
            "study's own output directory, or run scripts/run_study.py first.",
            flush=True,
        )
        return EXIT_NO_RECORDS

    verdict = evaluate_gate(records)
    print(report(records, verdict, args.out), flush=True)
    if args.figure:
        print(write_figure(records, args.figure), flush=True)
    return EXIT_OK if verdict["passed"] else EXIT_GATE_NOT_PASSED


if __name__ == "__main__":
    raise SystemExit(main())
