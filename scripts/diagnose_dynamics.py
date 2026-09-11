"""Both M3b diagnostics over the shipped nine cells. NO RETRAINING.

The M3 study reads NOT PASSED and every arm loses to persistence past h=16, so
the evidence points at what the arms SHARE rather than at the representation
contrast the study was built to test. This script runs the two diagnostics that
turn that inference into a mechanism -- action-shuffled imagination and the
k-step re-grounding sweep -- against the frozen checkpoints in `--out`.

WHAT MAKES THE OUTPUT READABLE, and why each piece is here:

  * EVERY PER-SEED COLUMN IS INDEXED BY ITS SEED, never by position in a list.
    A cell with no checkpoint prints MISSING in its own column; printing a list
    of the cells that DID run under headers "seed 0 / seed 1 / seed 2" puts seed
    2's number in seed 1's column, and the table looks complete.
  * THE SELF-CHECKS ARE PRINTED BEFORE THE FINDINGS. The three NUMERIC columns
    are exact equalities against `evaluate_rollout` and the shipped records, so
    a non-zero anywhere in that block means the numbers below it are not what
    they claim to be. The fourth column is an alarm rather than an equality:
    whether the smallest k's curve has collapsed onto the floor, which would
    mean the re-grounding consumed the frame it is scored on.
  * A CELL WITH NO USABLE COMPARISON GETS NO VERDICT AT ALL. Two ways to have
    none: the permutation was a no-op on every window, and fewer than two
    windows changed -- a spread cannot be estimated from one sample, and the
    resulting NaN must never fall through into an affirmative claim about M4.
  * THE VERDICT IS READ AGAINST A BAND, never against zero, AND THE BAND IS
    GIVEN A SCALE. The per-window spread of the delta is the ruler, and a delta
    inside it is "not measurably different", not "the same" -- but "not
    measurably different" is the same sentence for evidence that differs by
    three orders of magnitude, so the band is also quoted as a fraction of the
    persistence-to-floor range the model would have to close to be worth
    anything, which turns the verdict into an equivalence bound.
  * THE WINDOWS ARE NOT INDEPENDENT DRAWS. The shipped 229 come from 24
    validation episodes, so the ruler is the episode-clustered standard error
    and the naive one is reported beside it under its own name.

THE FAILURE STATUSES ARE DELIBERATELY DISTINCT, because they call for different
actions:

  EXIT_PROTOCOL_DIVERGED -- the sweep's k == horizon curve is not bitwise the
    open-loop rollout's. The re-grounding path has diverged from
    `evaluate_rollout` and everything else it reports is suspect. A CODE defect.
  EXIT_STREAM_DIVERGED -- the shuffle's real arm is not bitwise the open-loop
    rollout's. The two arms did not share a sampling stream, so the measured
    delta is noise. Also a code defect, and a different one.
  EXIT_RECORD_MISMATCH -- `evaluate_rollout` itself no longer reproduces the
    shipped record's curve. Measured, the records reproduce BITWISE on mps under
    torch 2.13.0 and miss on cpu, with an identical probe and split, by an
    ARM-DEPENDENT amount: ~6.5 map units (2.96% relative) on cnn/seed0 and ~12.4
    on frozen_ssl/seed0. So this status most likely means the ENVIRONMENT
    differs from the study's, not that the diagnostic is wrong -- and collapsing
    it into EXIT_PROTOCOL_DIVERGED would make a machine difference read as a
    code defect. Neither the checkpoints (which carry only arm and seed) nor the
    records carry the device, so it can be reported but never checked; the
    diagnostic records this script writes DO carry it.

    IT IS NOT ONLY THE REPRODUCTION THAT MOVES. On cpu the headline delta itself
    moved from +0.995 +-2.93 to +1.872 +-3.11 on frozen_ssl/seed0, and k=1 from
    126.68 to 130.16 -- the quantity being interpreted is not stable across
    devices at the precision it is being interpreted at. What DOES hold across
    devices is the protocol itself: open_loop_k and stream_drift both read
    exactly 0 on cpu, so only the record reproduction is device-locked. A
    non-zero here therefore qualifies every number below it without
    invalidating the protocol, which is why `verdict_block` carries the notice
    INSIDE the verdict rather than leaving it to the block printed underneath,
    where a reader who stops at the verdict never sees it.

  EXIT_UNSUPPORTED_DEVICE -- the device has no verified RNG snapshot, so the
    arms cannot be held on one sampling stream. Refused UP FRONT, before the
    checkpoint load and the ~20 s probe refit, and as a named status rather than
    the uncaught `NotImplementedError` that would surface as exit 1.
  EXIT_MISLABELLED_CHECKPOINT -- the checkpoint on disk is not the cell it was
    loaded for. Raised as its own EXCEPTION TYPE and caught as that type alone:
    ValueError reaches `diagnose_cell` from at least eight other places under
    `fit_probes`, `evaluate_rollout` and the two diagnostics, and catching the
    base class reported every one of them -- a horizon too large for the data,
    an empty split, a zero-variance probe target -- as a labelling problem.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.aggregate import SEEDS
from mbfps.eval.diagnostics import (
    REGROUNDING_KS,
    action_shuffled_rollout,
    regrounding_sweep,
    supports_matched_stream,
)
from mbfps.eval.probe import fit_probes
from mbfps.eval.rollout import evaluate_rollout
from mbfps.eval.study import SPLIT_SEED, StudyJob, job_record_path, load_record, write_record
from mbfps.models.encoders import encoder_backbone
from mbfps.training.world_model import WorldModel
from mbfps.utils.config import ARMS, get_config
from mbfps.utils.device import get_device

EXIT_OK = 0
EXIT_NO_CHECKPOINTS = 11
EXIT_SPLIT_MISMATCH = 12
EXIT_PROTOCOL_DIVERGED = 13
EXIT_RECORD_MISMATCH = 14
EXIT_STREAM_DIVERGED = 15
EXIT_MISLABELLED_CHECKPOINT = 16
EXIT_UNSUPPORTED_DEVICE = 17
"""Disjoint from `run_study.py`'s (0, 1, 3, 4, 5, 6) and `report_study.py`'s
(0, 7, 8, 9, 10), and none of them 1 (an uncaught traceback) or 2 (argparse's
own usage error). The three scripts run one after another in the same shell and
a wrapper reads the status; a collision would report one script's failure under
another's meaning."""

MISSING = "MISSING"


class MislabelledCheckpoint(ValueError):
    """The checkpoint on disk is not the cell it was loaded for.

    A TYPE rather than a bare `ValueError`, because `main` has to catch exactly
    this and nothing else. `diagnose_cell` runs `fit_probes`,
    `evaluate_rollout`, `regrounding_sweep` and `action_shuffled_rollout`, and
    ValueError is raised from at least eight places under them -- the no-window
    guards in `evaluate_rollout` and `_diagnose`, and probe.py's
    context/horizon/empty-path/zero-variance checks. Catching the base class
    reported every one of them as "MISLABELLED CHECKPOINT", which sends the
    reader to inspect checkpoint labels for what is a flag or a data problem.
    Everything else is left to surface as the traceback exit 1 is reserved for.
    """


def checkpoint_path(out_dir: Path, arm: str, seed: int) -> Path:
    """BOTH the arm and the seed are in the name.

    Either one missing collides two of the nine cells onto one file, and the
    diagnostic would then be reported under a name whose model it never saw --
    the failure mode `aggregate.py` names as this study's worst outcome.
    """
    return Path(out_dir) / f"world_model_{arm}_seed{seed}.pt"


def load_checkpoint_model(out_dir: Path, arm: str, seed: int, cfg, device):
    """The checkpoint for this cell, validated against the cell it is loaded for.

    The arm and the seed are checked SEPARATELY of one another, exactly as
    `study.run_job` does: the architecture is shared across seeds, so a
    seed-wrong checkpoint loads cleanly and nothing else would raise, and an
    arm-wrong one would be evaluated and reported under this arm's name.
    """
    checkpoint = torch.load(
        checkpoint_path(out_dir, arm, seed), map_location=device, weights_only=True
    )
    if checkpoint.get("arm") != arm or checkpoint.get("seed") != seed:
        raise MislabelledCheckpoint(
            f"checkpoint in {out_dir} is arm={checkpoint.get('arm')!r} "
            f"seed={checkpoint.get('seed')!r}, not this cell's arm={arm!r} "
            f"seed={seed!r}"
        )
    model = WorldModel(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def delta_band(result) -> np.ndarray:
    """Two standard errors of the per-window delta, over the CHANGED windows.

    The only ruler available for the shuffle. `+inf` below two changed windows,
    because a spread cannot be estimated from one sample -- and an infinite band
    makes every delta read as "not measurably different", which is the honest
    reading of a single window rather than a licence to call it a finding.
    """
    rows = result.window_position_delta[result.changed]
    if rows.shape[0] < 2:
        return np.full(result.window_position_delta.shape[1], np.inf)
    return 2.0 * rows.std(axis=0, ddof=1) / np.sqrt(rows.shape[0])


def _cluster_standard_error(values: np.ndarray, labels: np.ndarray) -> float:
    """Standard error of `values.mean()` when the observations come in clusters.

    The shipped 229 windows are cut from 24 validation episodes at 3 to 10
    windows each, and windows from one episode share its map, its route and its
    difficulty -- so `std / sqrt(229)` claims a precision the data does not
    support. Measured on the shipped cells the cluster-robust standard error is
    up to 1.32x the naive one, and the interval it produces is exactly what the
    verdict's equivalence statement rests on.

    The estimator is the usual sandwich for a sample mean: sum the residuals
    WITHIN each cluster, square those sums, and carry the small-sample
    correction `G / (G - 1)`. Summing within the cluster before squaring is
    what makes correlated residuals add rather than cancel; squaring first
    would give the naive variance back under a longer name.

    NaN below two clusters -- a between-cluster spread cannot be estimated from
    one -- so a caller can tell "not clustered" from "clustered, and small".
    """
    groups = np.unique(labels)
    if groups.size < 2:
        return float("nan")
    residual = values - values.mean()
    total = float(sum(residual[labels == group].sum() ** 2 for group in groups))
    correction = groups.size / (groups.size - 1)
    return float(np.sqrt(correction * total) / values.size)


def delta_summary(result) -> dict:
    """The shuffle's effect as ONE comparison rather than forty-five.

    A per-step band invites a multiple-comparison error that is severe at this
    horizon: two standard errors is roughly a 95% interval PER STEP, so under
    the null about 2 of 45 horizon steps fall outside it by chance and "one
    step exceeded its band" is not evidence of anything. The aggregate is a
    paired window-level statistic instead -- each changed window contributes
    its own horizon-mean delta, and the mean of those is read against its own
    standard error over windows. One comparison, measured against the
    window-to-window variation that the spread actually is.

    The per-step exceedance count is returned BESIDE it, with the number chance
    alone would produce, because the aggregate can hide an effect that is real
    at some steps and cancels across the horizon -- and because a permutation
    preserves the action multiset, so dynamics that depend only on that
    multiset are invisible at the final step by construction.
    """
    rows = result.window_position_delta[result.changed]
    band = delta_band(result)
    steps = int(band.size)
    if rows.shape[0] < 2:
        return {
            "mean": float("nan"), "se": float("inf"), "se_independent": float("inf"),
            "episodes": 0, "delta_final": float("nan"),
            "steps_outside": 0, "steps": steps, "expected_outside": 0.05 * steps,
        }
    per_window = rows.mean(axis=1)
    independent = float(per_window.std(ddof=1) / np.sqrt(per_window.size))
    # The episode labels are what separate 229 draws from 24 clusters. A
    # hand-built result carries none, and then the naive figure is all there is
    # -- which is reported under its own name either way, so the record says
    # which ruler the verdict used rather than leaving it to be inferred.
    clustered = float("nan")
    episodes = 0
    if result.window_episode is not None:
        labels = np.asarray(result.window_episode)[result.changed]
        episodes = int(np.unique(labels).size)
        clustered = _cluster_standard_error(per_window, labels)
    return {
        "mean": float(per_window.mean()),
        "se": independent if not np.isfinite(clustered) else clustered,
        "se_independent": independent,
        "episodes": episodes,
        "delta_final": float(result.position_delta()[-1]),
        "steps_outside": int((np.abs(result.position_delta()) > band).sum()),
        "steps": steps,
        "expected_outside": 0.05 * steps,
    }


def selfcheck_table(cells: dict, arms, seeds, smallest_k: int) -> str:
    """The three exact equalities, per cell. Anything but 0 invalidates the rest.

    THE LAST COLUMN CARRIES THE k IT WAS COMPUTED AT. The floor alarm is
    evaluated at the smallest k in `--ks`, which is 1 only by default; under
    `--ks 5 45` it is a statement about k=5, and self-check 2's whole point is
    the prior/posterior separation at ONE step -- at k=5 it is a different and
    weaker claim. A hardcoded "k1" heading would report that claim under the
    other one's name.
    """
    lines = [
        f"{'arm':<12}{'seed':>6}{'open_loop_k':>14}{'record_repro':>14}"
        f"{'stream_drift':>14}{f'k{smallest_k}_is_floor':>13}"
    ]
    for arm in arms:
        for seed in seeds:
            cell = cells.get((arm, seed))
            row = f"{arm:<12}{seed:>6}"
            if cell is None:
                row += f"{MISSING:>14}{MISSING:>14}{MISSING:>14}{MISSING:>13}"
            else:
                row += (
                    f"{cell['open_loop']:>14.3e}{cell['record']:>14.3e}"
                    f"{cell['stream']:>14.3e}{str(cell['smallest_k_is_floor']):>13}"
                )
            lines.append(row)
    return "\n".join(lines)


def shuffle_table(cells: dict, arms, seeds) -> str:
    """Per-arm rows, per-SEED columns, each cell keyed by `(arm, seed)`.

    THREE rows per arm, and the first two are different statistics rather than
    the same one twice. `horizon-mean` is the decision statistic -- the mean
    over the horizon of each changed window's delta, read against its own
    spread -- and `final step` is the endpoint the study's headline is quoted
    at. Measured on frozen_ssl/seed0 they differ by a factor of four (0.995
    against 4.065), because the delta curve trends monotonically from -0.589 at
    step 1 to +4.065 at step 45 and the horizon mean is exactly the summary
    that cancels a late-horizon effect against the early steps. Printing one
    under the other's name is a fourfold misreport with nothing to show for it.

    The third row is the changed/total window count that says whether either
    number means anything at all.
    """
    lines = [f"{'arm':<12}{'row':<16}" + "".join(f"{f'seed {s}':>18}" for s in seeds)]
    for arm in arms:
        delta = f"{arm:<12}{'horizon-mean':<16}"
        final = f"{'':<12}{'final step':<16}"
        counts = f"{'':<12}{'changed/total':<16}"
        for seed in seeds:
            cell = cells.get((arm, seed))
            if cell is None:
                delta += f"{MISSING:>18}"
                final += f"{MISSING:>18}"
                counts += f"{MISSING:>18}"
                continue
            delta += f"{cell['mean']:>+11.3f} +-{2 * cell['se']:>4.2f}"
            final += f"{cell['delta_final']:>+18.3f}"
            counts += f"{cell['windows_changed']:>10}/{cell['windows_total']:<7}"
        lines.append(delta)
        lines.append(final)
        lines.append(counts)
    return "\n".join(lines)


def sweep_table(cells: dict, arms, seeds, ks) -> str:
    """Final-horizon-step position error per re-grounding period.

    The floor and persistence are the SAME measurement for every k -- they are
    computed once per window and shared -- so the k columns are read against one
    bracket rather than against a re-drawn one.

    `spread` IS NOT THE BAR FOR A DIFFERENCE BETWEEN TWO k COLUMNS, and it was
    labelled `se` and documented as exactly that. It is the open-loop curve's
    own between-window spread. Every k is a mean over the SAME windows from the
    same per-window RNG snapshot, and per-window error is dominated by window
    difficulty, which they all share -- so the comparison between two k columns
    is paired and its ruler is the spread of the per-window DIFFERENCE.
    Measured on the shipped cells the unpaired figure overstates the bar for an
    adjacent-k difference by 1.7x to 3.9x, which is enough to hide the k=1
    versus k=3 separation entirely. `sweep_ruler_table` prints the paired bars
    for the comparisons this table invites; this column describes the open-loop
    curve on its own.
    """
    lines = [
        f"{'arm':<12}{'seed':>5}"
        + "".join(f"{f'k={k}':>11}" for k in ks)
        + f"{'floor':>11}{'persist':>11}{'spread':>9}"
    ]
    for arm in arms:
        for seed in seeds:
            cell = cells.get((arm, seed))
            row = f"{arm:<12}{seed:>5}"
            if cell is None:
                row += "".join(f"{MISSING:>11}" for _ in ks) + (
                    f"{MISSING:>11}{MISSING:>11}{MISSING:>9}"
                )
            else:
                row += "".join(f"{cell['k'][k]:>11.3f}" for k in ks)
                row += (
                    f"{cell['floor']:>11.3f}{cell['persistence']:>11.3f}"
                    f"{cell['spread_final']:>9.3f}"
                )
            lines.append(row)
    return "\n".join(lines)


def pair_names(ks) -> list[str]:
    """The adjacent-k comparisons the sweep table invites, in its own order."""
    return [f"{a}v{b}" for a, b in zip(ks, ks[1:])]


def sweep_ruler_table(cells: dict, arms, seeds, ks) -> str:
    """What each adjacent-k difference has to clear: the PAIRED two-SE bar.

    One column per adjacent pair in `--ks` order, because those are the
    comparisons a reader makes going across `sweep_table` -- "error becomes
    unrecoverable between k=5 and k=15" is a claim about one of these columns
    and nothing else. The bar is `2 * std_windows(curve_a - curve_b) / sqrt(n)`,
    which is the correct ruler precisely because the two curves are means over
    the same windows.

    The last pair of columns is self-check 2's quantitative half: the smallest
    k's margin over the floor, and that margin's own paired bar. A margin that
    is positive but inside its bar is noise around the floor rather than the
    strict prior/posterior separation the sweep predicts -- which is what the
    shipped checkpoints actually show, and it is only readable if the margin
    and its ruler are printed together.
    """
    smallest = min(ks)
    lines = [
        f"{'arm':<12}{'seed':>5}"
        + "".join(f"{f'{name} +-2se':>13}" for name in pair_names(ks))
        + f"{f'k{smallest}-floor':>13}{'+-2se':>9}"
    ]
    for arm in arms:
        for seed in seeds:
            cell = cells.get((arm, seed))
            row = f"{arm:<12}{seed:>5}"
            if cell is None:
                row += "".join(f"{MISSING:>13}" for _ in pair_names(ks))
                row += f"{MISSING:>13}{MISSING:>9}"
            else:
                row += "".join(
                    f"{cell['paired'][name]:>13.3f}" for name in pair_names(ks)
                )
                row += (
                    f"{cell['floor_margin']:>+13.3f}{cell['floor_margin_se']:>9.3f}"
                )
            lines.append(row)
    return "\n".join(lines)


def verdict_block(arm: str, seed: int, cell: dict) -> str:
    """What the shuffle licenses saying about M4, for one cell.

    FOUR outcomes, and the first two are not verdicts:

      * NO CHANGED WINDOWS -- the permutation was a no-op on every window, so
        the delta is undefined. Reported as UNINTERPRETABLE with the count, and
        NEITHER verdict is printed. An all-NaN delta must never fall through to
        "no difference".
      * NO ESTIMABLE SPREAD -- one changed window. `delta_summary` returns
        `mean=nan, se=inf` there, and the count guard above does not fire
        because 1 is not 0. `abs(nan) <= 2*inf` is FALSE, so this used to fall
        into the positive branch and print "The imagination does respond to the
        action: ... by +nan map units against +-2 SE (inf) ... M4 is NOT
        blocked on this evidence" -- an affirmative claim manufactured out of a
        NaN, and the exact opposite of what the infinite band exists to say.
        The gate is therefore on the STATISTIC BEING DEFINED, not on the count.
      * The aggregate delta is inside two standard errors -- the dynamics prior
        is not using the action, an actor trained inside this world model
        cannot learn, and that blocks M4.
      * The aggregate delta is outside it -- say so with the magnitude, and say
        M4 is not blocked ON THIS EVIDENCE. One permutation seed is one draw;
        the size of the effect against its spread is the finding, not its sign.

    ACCEPTING THE NULL REQUIRES A SCALE, or the same sentence covers evidence
    that differs by three orders of magnitude -- measured across the nine
    shipped cells the band runs from 0.014 to 3.45 map units. So the band is
    also reported as a fraction of the persistence-to-floor range at the final
    horizon step, which is the range the model would have to close to be worth
    anything, and the verdict states the equivalence bound that follows. Where
    that range is NON-POSITIVE -- the floor above persistence, which is every
    `cnn` cell -- there is no scale to quote and saying so is the honest
    report; a percentage of a negative range is worse than none.

    THE DECISION IS THE AGGREGATE, never the per-step exceedance count. Two
    standard errors is ~95% PER STEP, so under the null roughly 2 of 45 steps
    fall outside by chance and a rule that fired on one of them would call
    almost every cell responsive. The count is PRINTED beside the verdict, with
    what chance alone gives, because the aggregate can also hide an effect that
    cancels across the horizon -- neither number is trustworthy without the
    other.
    """
    head = f"--- {arm} seed {seed} ---"
    if cell["windows_changed"] == 0:
        return (
            f"{head}\nUNINTERPRETABLE: 0/{cell['windows_total']} windows had their "
            "actions changed by the permutation, so the shuffled arm IS the real "
            "arm and no comparison was made. No verdict."
        )
    mean, se = float(cell["mean"]), float(cell["se"])
    if not (np.isfinite(mean) and np.isfinite(se)):
        return (
            f"{head}\nUNINTERPRETABLE: {cell['windows_changed']}/"
            f"{cell['windows_total']} windows changed, and a spread cannot be "
            "estimated from one sample. The aggregate delta and its standard "
            "error are both undefined, so no verdict is available in either "
            "direction."
        )
    peak = float(np.nanmax(np.abs(np.asarray(cell["delta"], dtype=float))))
    counted = (
        f"({cell['windows_changed']}/{cell['windows_total']} windows changed in "
        f"{cell['episodes']} episodes; per-step |delta| outside its own band at "
        f"{cell['steps_outside']}/{cell['steps']} horizon steps, against "
        f"{cell['expected_outside']:.1f} expected by chance)"
    )
    verdict = (
        f"{head}\nTHE DYNAMICS PRIOR IS NOT USING THE ACTION: permuting the "
        f"horizon actions moved the mean position error by {mean:+.3f} map "
        f"units, inside +-2 SE ({2 * se:.3f}); the largest per-step move was "
        f"{peak:.3f}. {counted}{_equivalence(cell)} An actor trained inside this "
        "world model cannot learn anything, and M4 is BLOCKED."
        if abs(mean) <= 2.0 * se
        else
        f"{head}\nThe imagination does respond to the action: permuting the "
        f"horizon actions moved the mean position error by {mean:+.3f} map units "
        f"against +-2 SE ({2 * se:.3f}); the largest per-step move was "
        f"{peak:.3f}. {counted}{_equivalence(cell)} M4 is NOT blocked on this "
        "evidence."
    )
    return verdict + _reproduction_note(cell)


def _equivalence(cell: dict) -> str:
    """The band, against the range the model would have to close to matter.

    Without it "the prior is not using the action" is the same sentence for a
    cell whose band is 0.014 map units and one whose band is 3.45. With it the
    claim becomes a bound a reader can argue with: no effect larger than X% of
    the actionable range would have been detected at this many windows.
    """
    band = float(cell["persistence"]) - float(cell["floor"])
    if band <= 0.0:
        return (
            " The persistence-to-floor range at the final horizon step is "
            f"non-positive ({band:+.3f} map units: the floor sits above "
            "persistence), so there is no actionable range to scale this "
            "against and no equivalence bound follows."
        )
    return (
        f" Against the persistence-to-floor range of {band:.3f} map units at the "
        f"final horizon step, that band is {200.0 * float(cell['se']) / band:.1f}"
        "% of the range, so what this licenses is the equivalence statement: an "
        "action effect larger than that fraction of the actionable range would "
        "have been detected here, and none was."
    )


def _reproduction_note(cell: dict) -> str:
    """Carried INSIDE the verdict when the shipped record did not reproduce.

    The RECORD MISMATCH block is printed below every verdict, so a reader who
    stops at the verdict never sees it -- and measured, a CPU run of
    frozen_ssl/seed0 misses the record by 12.4 map units AND moves the headline
    delta from +0.995 to +1.872. The number the verdict interprets is not
    stable across devices at the precision it is being interpreted at, so the
    verdict says so where the verdict is read.
    """
    if float(cell["record"]) == 0.0:
        return ""
    return (
        f"\n  NOTE: this run does not reproduce the shipped record for this cell "
        f"(max abs {float(cell['record']):.3e} map units). Measured, the delta "
        "itself moves with the device at this precision, so read the verdict "
        "above as conditional on the environment recorded beside it."
    )


def diagnose_cell(args, arm: str, seed: int, device) -> dict:
    """One cell, end to end, on the frozen checkpoint. Returns its numbers."""
    cfg = get_config(arm, seed=seed, device=args.device)
    model = load_checkpoint_model(args.out, arm, seed, cfg, device)
    backbone = encoder_backbone(cfg.encoder)

    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    # The study's own split: the shared VAL_FRACTION, and SPLIT_SEED -- which is
    # fixed at 0 and deliberately NOT the cell's seed. Splitting at the cell
    # seed would score the model on episodes it trained on.
    train, val = episode_split(
        buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED
    )
    record = load_record(job_record_path(args.out, StudyJob(arm=arm, seed=seed)))
    # The record carries the held-out episodes BY NAME precisely so that "this
    # is the split the shipped numbers were scored on" is checkable after the
    # fact rather than assumed.
    names = [p.name for p in val]
    split_ok = names == record["episodes"]["val"]

    # The probe is fit at the ROLLOUT's own context/horizon and at the cell's
    # seed: it is applied to latents filtered from a zero state for exactly
    # `context` real frames, and fitting it at a different depth is a
    # distribution mismatch worth ~25 map units of position error.
    _, embedding_probe = fit_probes(
        model, train, backbone, device,
        context=args.context, horizon=args.horizon, seed=seed,
    )
    common = dict(
        context=args.context, horizon=args.horizon, seed=seed,
        device=device, feature_backbone=backbone,
    )
    reference = evaluate_rollout(model, val, embedding_probe, **common)
    sweep = regrounding_sweep(model, val, embedding_probe, ks=args.ks, **common)
    shuffle = action_shuffled_rollout(
        model, val, embedding_probe, permutation_seed=args.permutation_seed, **common
    )
    return {
        "arm": arm,
        "seed": seed,
        "split_ok": split_ok,
        "split_names": names,
        "record_names": record["episodes"]["val"],
        "open_loop": sweep.open_loop_divergence(reference),
        "record": float(
            np.abs(
                np.asarray(reference.rssm_position)
                - np.asarray(record["curves"]["rssm_position"])
            ).max()
        ),
        "stream": float(
            np.abs(shuffle.real.rssm_position - reference.rssm_position).max()
        ),
        # THE SMALLEST k, and the table and the record both say which one that
        # is. Computed at `max(args.ks)` the curve IS the open loop, which can
        # never be the floor, so the alarm would read False forever and
        # self-check 2 would be silently switched off in the table and in all
        # nine records. Deliberately WITHOUT its own EXIT_* status, unlike the
        # other three self-checks: those are exact equalities that must hold,
        # while this one is an alarm on a condition that has never fired on the
        # shipped checkpoints, and a status for it would be a gate nothing has
        # ever exercised end to end.
        "smallest_k": min(args.ks),
        "smallest_k_is_floor": sweep.is_bitwise_the_floor(min(args.ks), "position"),
        "windows_total": shuffle.windows_total,
        "windows_changed": shuffle.windows_changed,
        "delta": shuffle.position_delta(),
        "band": delta_band(shuffle),
        **delta_summary(shuffle),
        "k": {k: float(sweep.curve(k)[-1]) for k in args.ks},
        "floor": float(reference.floor_position[-1]),
        "persistence": float(reference.persistence_position[-1]),
        # NOT "se". `delta_summary` above contributes the shuffle's own
        # standard error under that name and the verdict decides on it; the
        # sweep's spread of the open-loop error is a different quantity that
        # the table prints, and a collision between the two is invisible --
        # both are plausible positive floats, the table still fills and the
        # verdict still reads as a sentence.
        "spread_final": float(sweep.curve_standard_error(args.horizon)[-1]),
        # The bars for the comparisons the sweep table actually invites. These
        # are PAIRED -- the k columns are means over the same windows -- and
        # the unpaired spread above overstates them by 1.7x to 3.9x on the
        # shipped cells, which is enough to hide the k=1 versus k=3 separation.
        "paired": {
            f"{a}v{b}": float(2 * sweep.paired_standard_error(a, b)[-1])
            for a, b in zip(args.ks, args.ks[1:])
        },
        "floor_margin": float(
            sweep.curve(min(args.ks))[-1] - reference.floor_position[-1]
        ),
        "floor_margin_se": float(
            2 * sweep.floor_margin_standard_error(min(args.ks))[-1]
        ),
        "curves": {
            "reference_position": reference.rssm_position,
            "shuffled_position": shuffle.shuffled_position,
            "floor_position": reference.floor_position,
            "persistence_position": reference.persistence_position,
            **{f"k{k}_position": sweep.curve(k) for k in args.ks},
        },
    }


def diagnostic_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    """One file per cell, named by BOTH the arm and the seed -- see
    `study.job_record_path` for what a colliding name costs."""
    return Path(out_dir) / f"diagnostic_{arm}_seed{seed}.json"


def build_record(cell: dict, args, device) -> dict:
    """The cell's numbers, plus the environment they are only reproducible in.

    The DEVICE and the torch version are recorded because the record
    reproduction is locked to them -- bitwise on mps under torch 2.13.0, and off
    on cpu by an arm-dependent ~6.5 (cnn/seed0) to ~12.4 (frozen_ssl/seed0) map
    units -- and nothing else in the study carries that. Without it a future
    reader cannot tell a sound diagnostic run on another machine from a broken
    one, and the shuffle delta itself moves with the device by more than its own
    standard error.
    """
    return {
        "arm": cell["arm"],
        "seed": cell["seed"],
        "context": args.context,
        "horizon": args.horizon,
        "ks": list(args.ks),
        "permutation_seed": args.permutation_seed,
        "split_seed": SPLIT_SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "episodes": {"val": cell["split_names"]},
        "self_checks": {
            "open_loop_divergence": cell["open_loop"],
            "record_reproduction": cell["record"],
            "stream_drift": cell["stream"],
            "smallest_k": cell["smallest_k"],
            "smallest_k_is_bitwise_the_floor": cell["smallest_k_is_floor"],
        },
        "windows": {
            "total": cell["windows_total"],
            "changed": cell["windows_changed"],
            # The clusters, not just the count. 229 windows cut from 24
            # episodes are not 229 independent draws, and a later reader
            # recomputing an interval from `total` alone would understate it.
            "episodes_changed": cell["episodes"],
        },
        "shuffle": {
            # The decision statistic and its spread, then the per-step curves it
            # was computed from -- the aggregate alone hides a cancelling
            # effect, the curves alone invite the 45-comparison error.
            "position_delta_mean": cell["mean"],
            "position_delta_final_step": cell["delta_final"],
            # The ruler the verdict used, and the naive one beside it under its
            # own name -- the two differ by up to 1.32x on the shipped cells and
            # a record that reported only one leaves the reader unable to tell
            # which interval the equivalence claim rests on.
            "position_delta_se": cell["se"],
            "position_delta_se_independent_windows": cell["se_independent"],
            "steps_outside_band": cell["steps_outside"],
            "steps_expected_outside_by_chance": cell["expected_outside"],
            "position_delta": [float(v) for v in cell["delta"]],
            "position_band": [float(v) for v in cell["band"]],
        },
        "curves": {
            name: [float(v) for v in curve]
            for name, curve in cell["curves"].items()
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/m3_study"))
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=45)
    parser.add_argument("--ks", nargs="+", type=int, default=list(REGROUNDING_KS))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--permutation-seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.ks = tuple(args.ks)
    if args.horizon not in args.ks:
        # argparse's OWN status (2), not one of the EXIT_* codes: this is a bad
        # flag combination, not a finding about the checkpoints. Letting
        # `regrounding_sweep` raise instead would leave main as an uncaught
        # traceback -- exit 1, which the EXIT_* block reserves.
        parser.error(
            f"--ks {list(args.ks)} omits --horizon {args.horizon}; the "
            "k == horizon pass IS the self-check that the sweep reproduces "
            "evaluate_rollout bitwise, and a sweep without it reports curves "
            "nothing has checked"
        )
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    device = get_device(prefer=args.device)
    # BEFORE the checkpoint load and the ~20 s probe refit, and asked of the
    # same code that does the refusing so the two cannot drift apart. Without
    # this the `NotImplementedError` from `_rng_snapshot` propagates out of
    # `main` as exit 1 -- the status the EXIT_* block reserves for an uncaught
    # traceback -- twenty seconds into a run that was never going to work.
    if not supports_matched_stream(device):
        print(
            f"UNSUPPORTED DEVICE: {device} has no verified RNG snapshot, so the "
            "shuffled and canonical arms cannot be held on one sampling stream "
            "and any delta measured here would be sampling noise. Re-run with "
            "--device mps or --device cpu."
        )
        return EXIT_UNSUPPORTED_DEVICE

    cells: dict[tuple[str, int], dict] = {}
    for arm in args.arms:
        for seed in args.seeds:
            if not checkpoint_path(args.out, arm, seed).exists():
                continue
            if not job_record_path(args.out, StudyJob(arm=arm, seed=seed)).exists():
                continue
            try:
                cell = diagnose_cell(args, arm, seed, device)
            except MislabelledCheckpoint as error:
                print(f"MISLABELLED CHECKPOINT: {error}")
                return EXIT_MISLABELLED_CHECKPOINT
            cells[(arm, seed)] = cell
            # Written as soon as the cell is computed, not after the whole
            # sweep: nine cells cost roughly half an hour, and a run that dies
            # on the last one would otherwise have nothing to show for the
            # eight that succeeded. `write_record` is atomic, so a reader sees
            # a complete record or none.
            write_record(
                diagnostic_record_path(args.out, arm, seed),
                build_record(cell, args, device),
            )

    if not cells:
        print(
            f"no cell in {args.out} has both a checkpoint and a record; nothing "
            "to diagnose"
        )
        return EXIT_NO_CHECKPOINTS

    print("\n--- self-checks (all three numeric columns must read exactly 0) ---")
    print(selfcheck_table(cells, args.arms, args.seeds, min(args.ks)))
    print("\n--- action-shuffled imagination (position, horizon-mean delta) ---")
    print(shuffle_table(cells, args.arms, args.seeds))
    print("\n--- k-step re-grounding sweep (position, final horizon step) ---")
    print(sweep_table(cells, args.arms, args.seeds, args.ks))
    print("\n--- what a difference between two k columns has to clear (paired) ---")
    print(sweep_ruler_table(cells, args.arms, args.seeds, args.ks))
    print("\n--- verdicts ---")
    for cell in cells.values():
        print(verdict_block(cell["arm"], cell["seed"], cell))

    # The self-checks are judged AFTER the whole report is on screen, in the
    # order that separates a code defect from an environment difference.
    for cell in cells.values():
        if not cell["split_ok"]:
            print(
                f"\nSPLIT MISMATCH for {cell['arm']} seed {cell['seed']}: the "
                "held-out episodes are not the ones the record was scored on.\n"
                f"  split:  {cell['split_names']}\n  record: {cell['record_names']}"
            )
            return EXIT_SPLIT_MISMATCH
    for cell in cells.values():
        if cell["open_loop"] != 0.0:
            print(
                f"\nPROTOCOL DIVERGED for {cell['arm']} seed {cell['seed']}: the "
                f"sweep at k={args.horizon} differs from evaluate_rollout by "
                f"{cell['open_loop']:.3e}. The re-grounding path is not the "
                "shipped protocol and every number above is suspect."
            )
            return EXIT_PROTOCOL_DIVERGED
    for cell in cells.values():
        if cell["stream"] != 0.0:
            print(
                f"\nSTREAM DIVERGED for {cell['arm']} seed {cell['seed']}: the "
                f"shuffle's real arm differs from evaluate_rollout by "
                f"{cell['stream']:.3e}, so the two arms did not share a sampling "
                "stream and the delta above is noise."
            )
            return EXIT_STREAM_DIVERGED
    for cell in cells.values():
        if cell["record"] != 0.0:
            print(
                f"\nRECORD MISMATCH for {cell['arm']} seed {cell['seed']}: "
                f"evaluate_rollout no longer reproduces the shipped curve "
                f"(max abs {cell['record']:.3e}). The protocol checks above "
                f"PASSED, so this points at the environment -- measured, the "
                f"records reproduce bitwise on mps and miss on cpu by an "
                f"arm-dependent ~6.5 (cnn/seed0) to ~12.4 (frozen_ssl/seed0) map "
                f"units. This run used device={device} torch={torch.__version__}."
            )
            return EXIT_RECORD_MISMATCH
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
