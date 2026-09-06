"""Aggregate the nine study records and judge M3's exit gate.

This is the last computation between a 33-hour study and the number a human
reads, so everything here is arranged so that a wrong number is louder than a
missing one.

THREE RULES THIS MODULE IS BUILT AROUND.

1. RECORDS ARE READ WITH `study.load_record`, NEVER WITH `json.loads`.
   `write_record` stores a non-finite float as JSON `null` plus an entry in the
   record's top-level `nonfinite` map naming its dotted path and which token it
   was; only `load_record` inverts that. Read with a bare `json.loads`, every
   NaN comes back as `None`, `gaps > 0` raises `TypeError`, and
   `np.isfinite(...)` on the resulting object-dtype array raises too. And NaN
   is the NORMAL path, not an edge case: `gap_final` and `gap_mean` are NaN by
   `metric_summary`'s contract whenever the persistence-to-floor band is
   non-positive, and `reward.r2` is NaN when the target has no variance. The
   study already paid for this once, one layer up, when the driver crashed
   formatting a sanitised `gap_final`.

2. THE FILE'S NAME AND THE RECORD'S CONTENTS ARE BOTH CHECKED, AGAINST EACH
   OTHER. Reporting one arm's numbers under another arm's name is the worst
   outcome this study has and it has no symptom: the table is full, every
   column is a plausible number, and nothing anywhere says which model produced
   it. The driver quarantines a mislabelled record it writes itself, but a
   record left in `--out` by an older run predates that guard, so
   `load_records` compares each file's name with the arm and seed inside it and
   REFUSES the whole directory on any disagreement. See `MislabelledRecord`.

3. A FAILING CRITERION IS A RESULT. Every criterion is reported separately with
   its own verdict and `passed` is their conjunction; nothing here is arranged
   so that a criterion can be quietly dropped, averaged away, or softened.
   Criterion 4 is measured failing on the real trained model -- latent R^2
   +0.1193 against the encoder embedding's +0.3287, with the bottleneck-free
   companion agreeing at gain -0.0208, 95% CI [-0.0395, -0.0040] -- and the
   correct output for that study is a legible NOT PASSED.

THE DEGENERATE-RECORD POLICY, STATED ONCE AND APPLIED EVERYWHERE.

A record can be present and still measure nothing: `gap_final` NaN because the
band was non-positive, `steps_degenerate > 0` because the band was positive but
numerically junk, `reward.is_degenerate` because the target is near-constant.
Averaging those silently yields NaN; excluding them silently changes the
denominator. This module does neither silently:

  * NOTHING IS DROPPED FROM A DENOMINATOR WITHOUT THE DENOMINATOR BEING
    PRINTED. `gap_final_mean` is `np.nanmean`, i.e. undefined cells are skipped
    -- and `n_seeds` and `n_seeds_with_finite_gap` sit beside it in the same
    row, so "0.31 over three seeds" can never be confused with "0.31 over the
    one seed that produced a number". When no seed is finite the mean is NaN
    rather than an exception.
  * A CELL THAT MEASURED NOTHING IS NOT A CELL THAT PASSED. `all_seeds_positive`
    is unanimity over finite, strictly positive gaps: NaN is not positive, and
    neither is +inf, which is what a band that underflowed produces.
  * DEGENERACY DOES NOT MOVE A MEAN, IT FAILS A CRITERION. A degenerate band
    fails `band_is_usable` for its arm; that is the criterion's whole purpose,
    and quietly dropping the cell from the mean instead would hide it.
"""

import math
from pathlib import Path

import numpy as np

from mbfps.eval.study import StudyJob, job_record_path, load_record
from mbfps.eval.summary import METRICS
from mbfps.utils.config import ARMS

SEEDS: tuple[int, ...] = (0, 1, 2)
"""The study's seeds. Must match `scripts/run_study.py`'s own tuple.

Pinned against the driver's by `test_the_aggregation_expects_the_cells_the_
driver_runs`: if the driver ran four seeds and the gate expected three, the
gate would report NOT PASSED on a complete study, or -- worse the other way --
call a 3x2 study complete.
"""

GATE_METRIC = "position"
"""The metric spec section 4 criterion 1 is judged on.

Position error in Doom map units is the headline number the whole study is
scored in. `angle` is summarised beside it and printed, but it does not gate:
the spec asks for `gap_closed > 0`, and the band, the floor and the persistence
reference are all defined against the position curve in section 3. Angle is
reported so the reader can see it, never averaged into the position column --
`per_arm` keeps the two in separate blocks for exactly that reason.
"""

MAX_DEGENERATE_STEPS = 0
"""How many horizon steps may have a numerically degenerate band.

Zero, and for BOTH metrics. At 20,000 steps the measured value was 0/45; a
nonzero count means `gap_closed` divided by a band that is positive but junk --
measured values of 125.2, 9.06 and -0.68 at adjacent steps on such a band --
and the raw curves should be read instead of the ratio.

Angle counts as well as position. Task 1 of this plan exists because the
degeneracy guards were computed for position only while the angle ratio was
printed bare over a curve that ranged +8.68 to -23.04; a gate that re-checked
position alone would reintroduce that at the last step of the study.
"""

CURVE_NAMES: tuple[str, ...] = tuple(
    f"{reference}_{metric}"
    for reference in ("rssm", "persistence", "floor")
    for metric in METRICS
)
"""The six per-step curves `run_job` records, derived from the one `METRICS`.

Spec section 4 criterion 2 asks for the error-vs-horizon curve of each arm with
persistence and floor on the same axes, so all three references for both
metrics have to be on disk or the criterion cannot be met.
"""

RECORD_GLOB = "result_*_seed*.json"
"""What a record file is called; see `study.job_record_path`.

The `.json` suffix is what keeps the driver's quarantined `*.mislabelled` files
and the `study.lock` claim file out of the aggregation.
"""

GATE_CRITERIA: tuple[str, ...] = (
    "all_nine_cells_present",
    "no_duplicate_cells",
    "every_record_is_a_study_cell",
    "beats_persistence",
    "band_is_usable",
    "filtering_beats_embedding",
    "reward_reported",
    "curves_produced",
)
"""Every criterion `evaluate_gate` reports, in the order it reports them.

Spelled out as a tuple so that a criterion silently dropped from the dict is a
named test failure rather than a conjunction over one fewer term. `passed` is
the conjunction of exactly these.
"""

CRITERIA_NOT_EVALUATED_HERE: dict[str, str] = {
    "invariant_tests_green": (
        "spec section 4 criterion 5: overfit-one-batch, gradient-flow, "
        "privileged-state isolation, determinism, golden rollout and split "
        "stability. Those are the pytest suite, not a field in any record; "
        "run it and read its exit status. Reported here as NOT EVALUATED so "
        "that a criterion nobody checked cannot be mistaken for one that "
        "passed by being absent from the list."
    ),
}
"""Gate criteria that exist in the spec and CANNOT be computed from records.

Printing the criteria that can be computed and staying silent about the rest is
how a five-criterion gate is reported as a four-criterion pass.
"""


class RecordsUnusable(Exception):
    """`--out` holds a file the aggregation must not silently skip."""


class MislabelledRecord(RecordsUnusable):
    """A record file's name and its contents name different cells.

    REFUSED RATHER THAN DROPPED, and refused for the whole directory rather
    than for the one file. Dropping it silently turns a 3x3 study into a 3x2
    one with no missing-cell message, because the file that would have been the
    ninth cell is still sitting there; and continuing with the other eight is
    the same bet the driver already refuses to make in `stale_records`. The
    operator has to look at the file, because only they can tell whether the
    name or the contents is the lie.
    """


class UnreadableRecord(RecordsUnusable):
    """A file matching the record glob could not be read back as a record.

    Separate from `MislabelledRecord` because the two want opposite responses:
    a mislabelled record is a real result under the wrong name and must be
    moved by hand, while an unreadable one is a cell that has to be re-run. The
    driver's `EXIT_NO_DATA` renumbering exists for this species of collapse one
    step earlier in the pipeline.
    """


def _is_number(value) -> bool:
    """Whether `value` is a number a mean may be taken over.

    `bool` is deliberately excluded even though it is an `int` in Python: a
    `True` where `gap_final` belongs would otherwise be averaged in as 1.0, and
    a criterion would read a flag as a measurement.
    """
    return isinstance(
        value, (int, float, np.integer, np.floating)
    ) and not isinstance(value, (bool, np.bool_))


def _num(record, *keys) -> float:
    """The number at `record[k1][k2]...`, or NaN if it is not there.

    NaN rather than an exception, because this runs after the GPU hours: a
    record one field short must cost that field, not the eight cells that would
    otherwise be reported. NaN is already this module's value for "this cell
    measured nothing", and every consumer of it -- `np.nanmean`, the positivity
    unanimity, the `_fmt` in the report script -- already handles it.

    A `None` (what a bare `json.loads` leaves where a non-finite float was)
    lands here as NaN too, so the aggregation degrades to "undefined" rather
    than raising `TypeError` deep inside a comparison. That is a backstop, not
    the policy: `load_records` restores the real floats.
    """
    node = record
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return float("nan")
        node = node[key]
    return float(node) if _is_number(node) else float("nan")


def _flag(record, *keys) -> bool | None:
    """The bool at `record[k1][k2]...`, or None if the record does not say.

    None rather than False, because "the record says the probe lost" and "the
    record does not carry the field at all" are different findings -- the first
    is a measurement, the second is a broken record. Both fail the criterion;
    only the caller may collapse them, and the report prints them differently.
    """
    node = record
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return bool(node) if isinstance(node, (bool, np.bool_)) else None


def record_cell(record) -> tuple[str, int] | None:
    """`(arm, seed)` if `record` names a cell, else None.

    THE TYPES ARE PART OF THE NAME. A record whose seed is the string `"1"`
    would satisfy every `f"...{seed}"` comparison and still miss every
    `(arm, seed) == (arm, 1)` set membership, so it would pass the filename
    check and then be counted as a missing cell -- a study reported incomplete
    with all nine files present, or a duplicate cell reported as two. `bool` is
    excluded for the same reason `_is_number` excludes it: `True` is an `int`,
    and `seed=True` is not seed 1.
    """
    arm = record.get("arm") if isinstance(record, dict) else None
    seed = record.get("seed") if isinstance(record, dict) else None
    if not isinstance(arm, str):
        return None
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        return None
    return (arm, int(seed))


def record_names_its_file(record, path) -> bool:
    """Whether `record`'s own arm and seed put it at `path`.

    The expected name comes from `study.job_record_path`, the one home of the
    convention, rather than from a regular expression here. A private parser
    would be a second home that could drift from the writer's, and the drift
    would show up as the aggregation quietly rejecting every real record or, in
    the other direction, accepting a mislabelled one.
    """
    path = Path(path)
    cell = record_cell(record)
    if cell is None:
        return False
    return path.name == job_record_path(path.parent, StudyJob(*cell)).name


def _describe_cell(record) -> str:
    """How a record names itself, with the types visible.

    `repr` on purpose: `seed=1` and `seed='1'` are a mislabelling this message
    has to be able to show, and `str` renders them identically.
    """
    arm = record.get("arm") if isinstance(record, dict) else None
    seed = record.get("seed") if isinstance(record, dict) else None
    return f"arm={arm!r} seed={seed!r}"


def mislabelled_report(bad: list[tuple[Path, dict]], out_dir) -> str:
    """What the operator sees instead of a table with a stranger's numbers in it.

    Names the file, what the record inside it claims to be, and where a record
    with those contents would have been written -- all three, because the fix
    depends on which of the two is wrong, and only a human can tell. The
    directory is named on the remedy line itself and not only in the banner:
    the remedy tells the operator to move files, and a remedy that does not say
    where costs the study a directory.
    """
    lines = [
        f"MISLABELLED RECORD: {len(bad)} file(s) in the --out directory "
        f"{out_dir} name a different cell than their filename does.",
        "Reporting one arm's numbers under another arm's name is the worst "
        "outcome this study has and it has no symptom, so the aggregation "
        "refuses the whole directory rather than guessing which half is "
        "wrong.",
    ]
    for path, record in bad:
        cell = record_cell(record)
        belongs = (
            job_record_path(Path(path).parent, StudyJob(*cell)).name
            if cell is not None
            else "no valid record path (arm must be a str, seed an int)"
        )
        lines.append(
            f"  {Path(path).name}  holds {_describe_cell(record)}  "
            f"-> belongs in {belongs}"
        )
    lines.append(
        "Remedy: in the --out directory "
        f"{out_dir}, move each file above to the name its contents ask for "
        "(if the contents are right), or out of the directory entirely (if "
        "the file is a leftover), then re-run the driver for whatever cell is "
        "then missing. Do not rename a file to make the table look complete: "
        "the arm inside it is the arm that was trained."
    )
    return "\n".join(lines)


def load_records(out_dir) -> list[dict]:
    """Every study record in `out_dir`, in filename order, NaNs restored.

    `out_dir` may be a `str`; argparse hands one over.

    Read through `study.load_record`, which is the only inverse of
    `write_record`'s non-finite encoding -- see this module's docstring for why
    `json.loads` is not an option here.

    Raises `MislabelledRecord` if any file's name disagrees with the arm and
    seed inside it, and `UnreadableRecord` if a file matching the glob cannot
    be parsed. Neither is skipped: a file the aggregation cannot account for is
    a file that might be the cell the table is missing.
    """
    out_dir = Path(out_dir)
    records: list[dict] = []
    bad: list[tuple[Path, dict]] = []
    for path in sorted(out_dir.glob(RECORD_GLOB)):
        try:
            record = load_record(path)
        except (
            OSError, ValueError, TypeError, AttributeError, KeyError, IndexError
        ) as error:
            # `UnicodeDecodeError` is a `ValueError`, not an `OSError`, and a
            # truncated write parses as neither -- both arrive here.
            #
            # THE FOUR AFTER `ValueError` ARE THE NON-FINITE RESTORE STEP.
            # `load_record` walks each dotted path in the record's `nonfinite`
            # map and writes the float back, and that walk is where a file
            # which is valid JSON but is not a well-formed record comes apart:
            # a JSON list or scalar has no `.get` (`AttributeError`), a `null`
            # block indexed on the way down raises `TypeError`, a dotted path
            # naming a field that is gone raises `KeyError`, and one naming a
            # list index past the end raises `IndexError`. That is the NORMAL
            # path, not an exotic one -- NaN is what `gap_final` holds whenever
            # the band is non-positive, so the restore step runs on real
            # records -- and letting any of the four escape costs the operator
            # the named refusal below and exits 1, the one status this
            # pipeline's numbering says must mean "uncaught traceback".
            raise UnreadableRecord(
                f"{path} matches {RECORD_GLOB!r} but could not be read back as "
                f"a record ({type(error).__name__}: {error}). It is not "
                "skipped: a file the aggregation cannot read may be the cell "
                "the table is missing. Re-run that cell, or move the file out "
                "of the --out directory."
            ) from error
        if not isinstance(record, dict) or not record_names_its_file(record, path):
            bad.append((path, record if isinstance(record, dict) else {}))
            continue
        records.append(record)
    if bad:
        raise MislabelledRecord(mislabelled_report(bad, out_dir))
    return records


def _metric_summary(rows: list[dict], metric: str) -> dict:
    """One metric's cross-seed block for one arm. See the module docstring."""
    gaps = np.array([_num(r, metric, "gap_final") for r in rows], dtype=float)
    finite = np.isfinite(gaps)
    return {
        "gap_final_by_seed": [float(g) for g in gaps],
        "gap_mean_by_seed": [_num(r, metric, "gap_mean") for r in rows],
        "band_median_by_seed": [_num(r, metric, "band_median") for r in rows],
        "n_seeds_with_finite_gap": int(finite.sum()),
        "gap_final_mean": (
            float(np.nanmean(gaps)) if finite.any() else float("nan")
        ),
        # Unanimity over FINITE and strictly positive gaps, and both halves
        # carry weight. `(gaps > 0).all()` is False for NaN on its own, but
        # +inf -- what a band that underflowed to zero produces, and a value
        # the record format has a token for -- is greater than zero and is not
        # a seed that beat persistence by any amount anyone measured.
        "all_seeds_positive": bool(finite.all() and (gaps > 0).all()),
        "max_degenerate_steps": _max_count(rows, metric, "steps_degenerate"),
        "max_steps_floor_above_persistence": _max_count(
            rows, metric, "steps_floor_above_persistence"
        ),
    }


def _max_count(rows: list[dict], *keys) -> float:
    """The largest of a count across an arm's seeds, NaN if none is a number.

    NaN is deliberately NOT zero: a record missing `steps_degenerate` has not
    demonstrated that its band was usable, and `nan <= 0` is False, so the
    criterion fails rather than passing on an absent field.
    """
    values = [_num(r, *keys) for r in rows]
    finite = [v for v in values if not math.isnan(v)]
    if len(finite) < len(values):
        return float("nan")
    return max(finite) if finite else float("nan")


def _nanmin(values: list[float]) -> float:
    """The smallest finite value, or NaN -- without numpy's all-NaN warning.

    NaN rather than an exception or `inf`: an arm whose records carry no KL
    rate has not reported one, and `inf` would render in the table as a rate
    nobody measured.
    """
    finite = [v for v in values if not math.isnan(v)]
    return min(finite) if finite else float("nan")


def _curve_of(record, name):
    """One named curve out of a record, or None if the record does not carry it.

    The `not isinstance(curves, dict)` half is not decoration: a record whose
    whole `curves` block came back as a `null` or a scalar is the same
    corruption species `_get` and `_num` guard against, and `{}.get` on it
    raises rather than answering.
    """
    curves = record.get("curves") if isinstance(record, dict) else None
    return curves.get(name) if isinstance(curves, dict) else None


def curve_gaps(record) -> list[str]:
    """Which of the six curves this record does not carry usably, by name.

    All six curves (`CURVE_NAMES`), all the same length, that length equal to
    the horizon the record says it was evaluated at. A record with three of the
    six cannot have persistence and floor drawn on the same axes as the model,
    which is what spec criterion 2 asks for; and a set of curves shorter than
    the horizon is a truncated rollout being plotted as a complete one.

    NAMES rather than a bool, because `curves_produced` is the one gate
    criterion a reader cannot locate from the report: every other failing
    criterion has a per-cell column or a named cell list, and this one used to
    print `[FAIL] curves_produced` and nothing else, leaving the operator to
    open nine JSON files. `_curves_ok` is this list being empty, so the
    criterion and the line that explains it cannot drift apart.
    """
    curves = record.get("curves") if isinstance(record, dict) else None
    if not isinstance(curves, dict):
        return list(CURVE_NAMES)
    horizon = record.get("horizon") if isinstance(record, dict) else None
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        return list(CURVE_NAMES)
    gaps = []
    for name in CURVE_NAMES:
        curve = curves.get(name)
        if not isinstance(curve, list) or not curve or len(curve) != horizon:
            gaps.append(name)
    return gaps


def _curves_ok(record) -> bool:
    """Whether this record carries a complete error-vs-horizon curve set."""
    return not curve_gaps(record)


def per_arm(records: list[dict]) -> dict:
    """Per-arm summary across seeds, NaN-aware throughout.

    Rows are ordered by seed, not by the order the records arrived in, so
    `gap_final_by_seed[0]` is seed 0's number whatever order the directory
    listed. `seeds` is reported beside it: a list of three numbers whose seeds
    nobody printed is a list nobody can check.
    """
    summary: dict[str, dict] = {}
    for arm in sorted({r["arm"] for r in records if isinstance(r.get("arm"), str)}):
        rows = sorted(
            (r for r in records if r.get("arm") == arm),
            key=lambda r: (record_cell(r) is None, record_cell(r) or ("", 0)),
        )
        metrics = {metric: _metric_summary(rows, metric) for metric in METRICS}
        gate = metrics[GATE_METRIC]
        degenerate = [metrics[m]["max_degenerate_steps"] for m in METRICS]
        gains = np.array([_num(r, "filtering", "gain", "gain") for r in rows])
        ci_low = [_num(r, "filtering", "gain", "ci_low") for r in rows]
        ci_high = [_num(r, "filtering", "gain", "ci_high") for r in rows]
        ridges = [
            {
                "ridge_selected": _flag(r, "filtering", "gain", "ridge_selected"),
                "joint_ridge": _num(r, "filtering", "gain", "joint_ridge"),
                "embedding_ridge": _num(r, "filtering", "gain", "embedding_ridge"),
            }
            for r in rows
        ]
        finite_gains = np.isfinite(gains)
        summary[arm] = {
            "n_seeds": len(rows),
            "seeds": [r.get("seed") for r in rows],
            # The gate metric's block, flat, because that is the column the
            # study is scored in. It is the same data as `metrics["position"]`
            # and `test_the_flat_keys_are_the_gate_metrics_own` pins it there,
            # so the flat view cannot come to hold angle's numbers.
            "gap_final_by_seed": gate["gap_final_by_seed"],
            "gap_final_mean": gate["gap_final_mean"],
            "n_seeds_with_finite_gap": gate["n_seeds_with_finite_gap"],
            "all_seeds_positive": gate["all_seeds_positive"],
            "metrics": metrics,
            # Degeneracy is judged over BOTH metrics; see MAX_DEGENERATE_STEPS.
            "max_degenerate_steps": (
                float("nan")
                if any(math.isnan(d) for d in degenerate)
                else max(degenerate)
            ),
            "any_degenerate": not all(
                d <= MAX_DEGENERATE_STEPS for d in degenerate
            ),
            # Criterion 4 lives at filtering.criterion_4, NOT at the top of the
            # filtering block: `run_job` writes {"criterion_4": ..., "gain":
            # ...} so that the bottleneck-free companion is recorded beside the
            # criterion rather than instead of it.
            "latent_r2_by_seed": [
                _num(r, "filtering", "criterion_4", "latent_r2") for r in rows
            ],
            "embedding_r2_by_seed": [
                _num(r, "filtering", "criterion_4", "embedding_r2") for r in rows
            ],
            "latent_beats_embedding_by_seed": [
                _flag(r, "filtering", "criterion_4", "latent_beats_embedding")
                for r in rows
            ],
            # A record that does not carry the flag has not passed it.
            "filtering_all_pass": bool(rows) and all(
                _flag(r, "filtering", "criterion_4", "latent_beats_embedding") is True
                for r in rows
            ),
            "gain_by_seed": [float(g) for g in gains],
            "gain_ci_low_by_seed": ci_low,
            "gain_ci_high_by_seed": ci_high,
            # None, NOT False, when either end is undefined: an interval with
            # one end is not an interval, and `bool(low > 0 or high < 0)` is a
            # DISJUNCTION -- one finite end on the right side of zero satisfies
            # it on its own, so a gain whose lower bound was never computed
            # would be reported as significant. `report_study._excludes_zero`
            # renders exactly this rule as "n/a" and
            # `test_the_two_readings_of_the_interval_agree` pins the two
            # together; they used to disagree, with the API being the wrong one.
            "gain_ci_excludes_zero_by_seed": [
                None
                if math.isnan(low) or math.isnan(high)
                else bool(low > 0.0 or high < 0.0)
                for low, high in zip(ci_low, ci_high)
            ],
            "gain_mean": (
                float(np.nanmean(gains)) if finite_gains.any() else float("nan")
            ),
            "n_seeds_with_finite_gain": int(finite_gains.sum()),
            "gain_ridges": ridges,
            # R1: the gain's SIGN moves with the selected ridge decade, so a
            # mean over seeds that selected different decades is a mean over
            # different estimators. Flagged rather than corrected -- there is
            # no correction, only the disclosure.
            "gain_ridges_agree": all(r == ridges[0] for r in ridges),
            "reward_mse_by_seed": [_num(r, "reward", "mse") for r in rows],
            "reward_baseline_mse_by_seed": [
                _num(r, "reward", "baseline_mse") for r in rows
            ],
            "reward_r2_by_seed": [_num(r, "reward", "r2") for r in rows],
            # Spec criterion 3 asks for reward accuracy REPORTED, not for it to
            # be good: `my_way_home`'s reward is near-constant by construction
            # (19,417 of 19,424 steps share the living penalty), so gating on a
            # non-degenerate target would fail the milestone for a property of
            # the scenario that was known before the study started. The flag is
            # therefore surfaced beside the number and is not a criterion.
            # `is not False`, so a record that does not carry the flag counts
            # as degenerate here. The flag exists to stop an MSE over a
            # near-constant target being read as precision, and a cell that
            # never said whether its target was constant has not earned the
            # benefit of the doubt.
            "any_reward_degenerate": any(
                _flag(r, "reward", "is_degenerate") is not False for r in rows
            ),
            "reward_reported": bool(rows) and all(
                not math.isnan(_num(r, "reward", "mse"))
                and not math.isnan(_num(r, "reward", "baseline_mse"))
                and _num(r, "reward", "n_steps") > 0
                for r in rows
            ),
            "curves_ok": bool(rows) and all(_curves_ok(r) for r in rows),
            "steps_per_second_by_seed": [_num(r, "steps_per_second") for r in rows],
            "kl_rate_min": _nanmin(
                [_num(r, "kl_rate_above_free_bits") for r in rows]
            ),
        }
    return summary


def ridge_groups(records: list[dict]) -> list[dict]:
    """The nine cells grouped by the ridge each one's gain was computed at.

    R1, and the reason it is a requirement: `RIDGES` is a decade grid, the
    scored R^2 moves ~0.10 per decade, and the gain being measured is ~0.02 --
    so the gain's SIGN moves with the selected decade. Measured under three
    defensible selection designs the same model gave -0.0706, -0.0208 and
    +0.0325. A mean over nine cells that did not all select the same decade is
    a mean over different estimators and no reader can tell from the number
    alone, so the grouping is printed whether or not the cells agree.
    """
    groups: dict[tuple, dict] = {}
    for record in records:
        key = (
            _flag(record, "filtering", "gain", "ridge_selected"),
            _num(record, "filtering", "gain", "joint_ridge"),
            _num(record, "filtering", "gain", "embedding_ridge"),
        )
        # NaN != NaN, so a missing ridge would open a new group per record and
        # the table would grow one row per broken cell. Keyed on the rendered
        # form for that reason.
        bucket = groups.setdefault(
            (str(key[0]), repr(key[1]), repr(key[2])),
            {
                "ridge_selected": key[0],
                "joint_ridge": key[1],
                "embedding_ridge": key[2],
                "cells": [],
                "gains": [],
            },
        )
        cell = record_cell(record)
        bucket["cells"].append(
            cell if cell is not None else (record.get("arm"), record.get("seed"))
        )
        bucket["gains"].append(_num(record, "filtering", "gain", "gain"))
    out = []
    for bucket in groups.values():
        gains = np.array(bucket["gains"], dtype=float)
        finite = np.isfinite(gains)
        out.append(
            {
                **bucket,
                "n_cells": len(bucket["cells"]),
                # THE DENOMINATOR THE MEAN WAS ACTUALLY TAKEN OVER, beside the
                # count of cells in the group. `gain_mean` is `np.nanmean`, so
                # a group of nine cells three of which never computed a gain
                # published `cells 9` next to a mean over six -- the one row in
                # this module that broke the module docstring's blanket promise
                # that nothing is dropped from a denominator without the
                # denominator being printed. `_metric_summary` publishes
                # `n_seeds_with_finite_gap` and `per_arm` publishes
                # `n_seeds_with_finite_gain` for the same reason.
                "n_finite_gains": int(finite.sum()),
                "gain_mean": (
                    float(np.nanmean(gains)) if finite.any() else float("nan")
                ),
            }
        )
    return sorted(
        out,
        key=lambda g: (
            g["ridge_selected"] is not True,
            _sort_key(g["joint_ridge"]),
            _sort_key(g["embedding_ridge"]),
        ),
    )


def _sort_key(value: float) -> float:
    """NaN sorts last rather than making the ordering depend on input order."""
    return math.inf if math.isnan(value) else value


def mean_curve(records: list[dict], name: str) -> np.ndarray:
    """The elementwise mean of one curve across `records`, NaN-aware.

    Raises `ValueError` on a ragged set rather than letting numpy build an
    object array and average the arms' horizons together: two cells evaluated
    at different horizons are not two samples of one curve, and a figure that
    silently plots the shorter one is a figure of a study nobody ran.
    """
    if name not in CURVE_NAMES:
        raise ValueError(f"{name!r} is not one of {CURVE_NAMES}")
    # `record.get("curves", {})` answers with the block that IS there, so a
    # record whose whole `curves` key came back as a `null` or a scalar --
    # the corruption species `_get` and `_curves_ok` both guard against --
    # would reach `.get(name)` and raise `AttributeError`. `write_figure`
    # catches `ValueError` and `OSError`, so that escapes main() AFTER the
    # verdict has been printed and costs the report its exit status.
    curves = [_curve_of(record, name) for record in records]
    missing = [
        record_cell(r) for r, c in zip(records, curves) if not isinstance(c, list)
    ]
    if missing:
        raise ValueError(f"no {name!r} curve for {missing}")
    lengths = {len(c) for c in curves}
    if len(lengths) != 1:
        raise ValueError(
            f"{name!r} curves have different lengths {sorted(lengths)}; cells "
            "evaluated at different horizons are not samples of one curve"
        )
    return np.nanmean(np.array(curves, dtype=float), axis=0)


def _arm_flag(summary: dict, arm: str, key: str) -> bool:
    """An arm's boolean, False when the arm produced no record at all."""
    return bool(summary.get(arm, {}).get(key, False))


def evaluate_gate(records: list[dict], arms=ARMS, seeds=SEEDS) -> dict:
    """Spec section 4. Every criterion is reported; `passed` is the conjunction.

    The conjunction is over exactly `GATE_CRITERIA`, and a criterion that
    cannot be computed from the records is reported by name under
    `not_evaluated` rather than left out -- see `CRITERIA_NOT_EVALUATED_HERE`.

    A criterion is judged over `arms`, not over whichever arms happen to have
    records: an arm with no record at all fails every one of them, which is
    what stops a study whose expensive arm never ran from passing on the two
    that did.
    """
    arms_summary = per_arm(records)
    cells = [record_cell(r) for r in records]
    expected = {(a, s) for a in arms for s in seeds}
    present = [c for c in cells if c is not None]
    duplicates = sorted({c for c in present if present.count(c) > 1})
    unexpected = sorted(
        {
            _describe_cell(r)
            for r, c in zip(records, cells)
            if c is None or c not in expected
        }
    )
    # Which cell, and which of its six curves. `curves_produced` was the one
    # criterion the report could not attribute: `beats_persistence` and
    # `band_is_usable` have per-seed columns, `filtering_beats_embedding` and
    # `reward_reported` have their own tables, and the three cell-accounting
    # criteria name their cells -- while a short `floor_angle` in one cell
    # printed `[FAIL] curves_produced` and left the operator nine JSON files to
    # open. Sorted by cell so the line is stable across directory order.
    incomplete_curves = sorted(
        (
            (c if c is not None else _describe_cell(r), curve_gaps(r))
            for r, c in zip(records, cells)
            if curve_gaps(r)
        ),
        key=lambda item: str(item[0]),
    )

    criteria = {
        "all_nine_cells_present": not (expected - set(present)),
        "no_duplicate_cells": not duplicates,
        "every_record_is_a_study_cell": not unexpected,
        "beats_persistence": all(
            _arm_flag(arms_summary, arm, "all_seeds_positive") for arm in arms
        ),
        # `nan <= 0` is False, so an arm with a record missing the count fails
        # here instead of passing on an absent field.
        "band_is_usable": all(
            arms_summary.get(arm, {}).get("max_degenerate_steps", 1)
            <= MAX_DEGENERATE_STEPS
            for arm in arms
        ),
        "filtering_beats_embedding": all(
            _arm_flag(arms_summary, arm, "filtering_all_pass") for arm in arms
        ),
        "reward_reported": all(
            _arm_flag(arms_summary, arm, "reward_reported") for arm in arms
        ),
        "curves_produced": all(
            _arm_flag(arms_summary, arm, "curves_ok") for arm in arms
        ),
    }
    if set(criteria) != set(GATE_CRITERIA):
        raise AssertionError(
            f"the gate computed {sorted(criteria)} but reports "
            f"{sorted(GATE_CRITERIA)}; a criterion that is computed and not "
            "reported, or reported and not computed, is a gate nobody can read"
        )
    return {
        "criteria": {name: bool(criteria[name]) for name in GATE_CRITERIA},
        "passed": all(criteria[name] for name in GATE_CRITERIA),
        "not_evaluated": dict(CRITERIA_NOT_EVALUATED_HERE),
        "per_arm": arms_summary,
        "ridge_groups": ridge_groups(records),
        "missing_cells": sorted(expected - set(present)),
        "duplicate_cells": duplicates,
        "unexpected_cells": unexpected,
        "incomplete_curves": incomplete_curves,
        "n_records": len(records),
        "n_expected": len(expected),
    }


__all__ = [
    "ARMS",
    "CRITERIA_NOT_EVALUATED_HERE",
    "CURVE_NAMES",
    "GATE_CRITERIA",
    "GATE_METRIC",
    "MAX_DEGENERATE_STEPS",
    "MislabelledRecord",
    "RECORD_GLOB",
    "RecordsUnusable",
    "SEEDS",
    "UnreadableRecord",
    "curve_gaps",
    "evaluate_gate",
    "load_records",
    "mean_curve",
    "mislabelled_report",
    "per_arm",
    "record_cell",
    "record_names_its_file",
    "ridge_groups",
]
