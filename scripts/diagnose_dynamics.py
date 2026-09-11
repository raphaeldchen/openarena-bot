"""Both M3b diagnostics over the shipped nine cells. NO RETRAINING.

The M3 study reads NOT PASSED and every arm loses to persistence past h=16, so
the evidence points at what the arms SHARE rather than at the representation
contrast the study was built to test. This script runs the two diagnostics that
turn that inference into a mechanism -- the action-intervention ladder and the
k-step re-grounding sweep -- against the frozen checkpoints in `--out`.

THE LADDER IS PRINTED AS A LADDER. Its rungs -- shuffled, resampled, constant --
are in strictly increasing order of perturbation, and the finding is the shape
of the response across them: a null at every rung bounds the effect of THOSE
perturbations; a response that first appears at a HIGHER rung says the prior is
action-conditioned but insensitive to what the rungs below it perturb, which is
a different and better finding and is reported as that. So the table prints the
rungs in `LADDER`'s order under each arm whatever order `--rungs` was typed in,
each rung carries its OWN changed/total, steps-moved and multiset-distance
counts on the same footing as its delta, and the verdict is decided over the
ladder rather than over any one rung by name. The top rung is a CONTRAST between
two held actions -- MOVE_FORWARD minus NOOP unless `--contrast` says otherwise
-- and every held action's own delta is printed beneath it, because measured on
these checkpoints a single held action reads null or +31 map units depending
only on which action it holds.

THE VERDICT IS CORRECTED OVER THE FAMILY AND READS THE SIGN. A run makes rungs
x cells x channels comparisons; a per-cell verdict at +-2 SE turns the null's
own expected false positives into findings, so every rung is decided against
the Bonferroni threshold for the whole family, the nominal band is printed
beside it, and a cross-cell block prints how many cells sit outside each ruler
and the sign of every cell's delta. A response is what unblocks M4 only with
the sign conditioning predicts -- the intervened sequence imagining WORSE than
the one it replaced -- because a decrease in error under an intervened sequence
is not conditioning an actor can use.

THE NULL HEADLINE IS CONDITIONAL. "The prior does not measurably use the
action" is printed only when the position probe can register an effect at all
-- on every `cnn` cell it cannot: selection R^2 -0.036, floor = persistence =
open loop, so the cell reads UNMEASURABLE THROUGH THIS PROBE -- and only when
the per-action contrast was among the rungs, because a null on order and
counts alone is exactly what an action-conditioned prior that reads the
action's identity produces.

EVERY RUNG IS ALSO READ WHERE NO PROBE INTERVENES. Beside the position and
angle deltas each rung prints its embedding-space ratio: the median per-window
L2 between the intervened and real imaginations' embeddings, over the median
L2 between the real imagination and a second draw of it from a different
stream point -- the TWO-DRAW distance, about sqrt(2) times one draw's own
spread, which is the landmark "1" means. Exactly 0 is an action-blind prior;
below 1 is a response smaller than the two-draw distance; above 1 is larger.
It is the one reading the pixel arm's degenerate probe cannot attenuate, so
the UNMEASURABLE verdict quotes it. THE TOP RUNG IS PRINTED ON THE SAME AXIS
AS THE OTHER TWO: the contrast's own numerator is the distance between two
held imaginations -- a between-counterfactual distance, structurally larger
than either's distance to real -- so the ladder row prints the two contrast
actions' held-vs-real ratios and the contrast gets its own row. The record
persists the per-window series behind every statistic -- each rung's
horizon-mean delta per window, its embedding numerator, both ratio summaries
and the per-step ratio, the noise reference and the window -> episode index
-- so `scripts/pool_dynamics.py` can pool the nine cells by episode.

TWO MORE SELF-CHECKS GUARD THE RULER. The noise draw is one more `imagine`
per window, and it must leave the stream exactly where `evaluate_rollout`
leaves it or every later window's rungs move: `noise_restored` must read True
and `noise_same` -- the windows whose reference was the canonical latent
bitwise -- must read 0. Either failing is `EXIT_STREAM_DIVERGED`, the same
defect class as `stream_drift`: the arms no longer share a stream.

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
  * A RUNG WITH NO USABLE COMPARISON GETS NO READING AT ALL, and a cell whose
    every rung is like that gets no verdict. Two ways to have none: the
    intervention was a no-op on every window, and fewer than two windows
    changed -- a spread cannot be estimated from one sample, and the resulting
    NaN must never fall through into an affirmative claim about M4. An
    uninterpretable rung is named as such and EXCLUDED from the ladder's
    decision, never counted as a null.
  * THE VERDICT IS READ AGAINST A BAND, never against zero, AND THE BAND IS
    GIVEN A SCALE. The per-window spread of the delta is the ruler, and a delta
    inside it is "not measurably different", not "the same" -- but "not
    measurably different" is the same sentence for evidence that differs by
    three orders of magnitude, so the band is also quoted as a fraction of the
    persistence-to-floor range the model would have to close to be worth
    anything, which turns the verdict into an equivalence bound on the
    perturbations that were run.
  * BOTH CHANNELS ARE READ. 38% of the actions turn rather than move, the
    angle probe is the one channel any shipped model beats persistence on, and
    a prior whose action pathway drives heading passes every position-only
    rung undetected -- so every rung is decided on position AND angle.
  * THE WINDOWS ARE NOT INDEPENDENT DRAWS. The shipped 229 come from 24
    validation episodes, so the ruler is the episode-clustered standard error
    and the naive one is reported beside it under its own name.

THE FAILURE STATUSES ARE DELIBERATELY DISTINCT, because they call for different
actions:

  EXIT_PROTOCOL_DIVERGED -- the sweep's k == horizon curve is not bitwise the
    open-loop rollout's. The re-grounding path has diverged from
    `evaluate_rollout` and everything else it reports is suspect. A CODE defect.
  EXIT_STREAM_DIVERGED -- the ladder's real arm is not bitwise the open-loop
    rollout's. The rungs did not share a sampling stream with it, so every
    measured delta is noise. Also a code defect, and a different one.
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
from math import comb
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.aggregate import SEEDS
from mbfps.eval.diagnostics import (
    CONTRAST,
    LADDER,
    LADDER_PERTURBS,
    REGROUNDING_KS,
    action_intervention_ladder,
    action_name,
    ladder_order,
    regrounding_sweep,
    supports_matched_stream,
)
from mbfps.eval.pooling import cluster_standard_error
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
"""Disjoint from `run_study.py`'s (0, 1, 3, 4, 5, 6), `report_study.py`'s
(0, 7, 8, 9, 10) and `pool_dynamics.py`'s (0, 18-22), and none of them 1 (an
uncaught traceback) or 2 (argparse's own usage error). The scripts run one
after another in the same shell and a wrapper reads the status; a collision
would report one script's failure under another's meaning."""

MISSING = "MISSING"


class MislabelledCheckpoint(ValueError):
    """The checkpoint on disk is not the cell it was loaded for.

    A TYPE rather than a bare `ValueError`, because `main` has to catch exactly
    this and nothing else. `diagnose_cell` runs `fit_probes`,
    `evaluate_rollout`, `regrounding_sweep` and `action_intervention_ladder`,
    and ValueError is raised from at least nine places under them -- the
    no-window guards in `evaluate_rollout`, `_diagnose` and the ladder's
    marginal pre-pass, and probe.py's
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


CHANNELS: tuple[str, ...] = ("position", "angle")
"""The two probed channels every rung is read on.

Position alone is not enough: 38% of the actions are TURN_LEFT/TURN_RIGHT,
whose first-order effect is on heading, and the angle probe is the one
channel on which any shipped model beats persistence at all. A prior whose
action pathway drives heading and not displacement passes every position-only
rung undetected -- measured, holding TURN_RIGHT on frozen_ssl/1 moves the
angle delta by +6.3 degrees while the position rungs read null.
"""

NOMINAL_Z = 2.0
"""The per-comparison band: two standard errors, the ruler the shuffled arm
was always read against. Kept as the printed band; the DECISION is taken
against `family_threshold` below."""


def family_threshold(family: int) -> float:
    """The z a comparison must clear to count as a response, Bonferroni-
    corrected over the whole family: two-sided 5% divided by the number of
    comparisons the run makes -- rungs times cells times channels.

    Without it the verdict turns the null's own expected false-positive count
    into a finding. Measured on the nine shipped cells with three rungs, one
    cell read "RESPONDS at constant" at p = 0.007 -- one rejection in a family
    of 27, where 1.35 are expected under the null at 5% and Holm rejects
    nothing -- and the per-cell verdict printed "action-conditioned ... M4 NOT
    blocked" on the strength of it. At family 54 the threshold is z = 3.31; a
    family of one gives back the nominal 1.96.
    """
    if family < 1:
        raise ValueError(f"a family of {family} comparisons has no threshold")
    return float(NormalDist().inv_cdf(1.0 - 0.025 / family))


def _window_deltas(result, metric: str) -> np.ndarray:
    if metric not in CHANNELS:
        raise KeyError(f"no {metric!r} channel; the channels are {CHANNELS}")
    return result.window_position_delta if metric == "position" else result.window_angle_delta


def delta_band(result, metric: str = "position") -> np.ndarray:
    """Two standard errors of the per-window delta, over the CHANGED windows.

    The only per-step ruler available for a rung. `+inf` below two changed windows,
    because a spread cannot be estimated from one sample -- and an infinite band
    makes every delta read as "not measurably different", which is the honest
    reading of a single window rather than a licence to call it a finding.
    """
    rows = _window_deltas(result, metric)[result.changed]
    if rows.shape[0] < 2:
        return np.full(rows.shape[1], np.inf)
    return NOMINAL_Z * rows.std(axis=0, ddof=1) / np.sqrt(rows.shape[0])


def delta_summary(result, metric: str = "position") -> dict:
    """One rung's effect on one channel as ONE comparison rather than forty-five.

    Written against the rung's per-window deltas, `changed`, `position_delta`/
    `angle_delta` and `window_episode`, which every rung of the ladder carries
    on the same footing -- so the shuffled rung's statistic and the constant
    rung's are the same computation, not two that could drift, and the angle
    channel's is the position channel's with the other array.

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
    multiset are invisible at the final step of the shuffled rung by
    construction, which is what the rungs above it exist to catch.
    """
    rows = _window_deltas(result, metric)[result.changed]
    band = delta_band(result, metric)
    steps = int(band.size)
    curve = result.position_delta() if metric == "position" else result.angle_delta()
    # Over ALL windows, in traversal order and index-aligned with the
    # episode index -- an unchanged window contributes its exact zero here
    # and is EXCLUDED from `mean` below; the persisted mask is what lets a
    # reader reproduce the one from the other.
    window_mean = _window_deltas(result, metric).mean(axis=1)
    if rows.shape[0] < 2:
        return {
            "mean": float("nan"), "se": float("inf"), "se_independent": float("inf"),
            "delta_final": float("nan"), "steps_outside": 0,
            "expected_outside": 0.05 * steps,
            "delta": curve, "band": band, "window_mean": window_mean,
        }
    per_window = rows.mean(axis=1)
    independent = float(per_window.std(ddof=1) / np.sqrt(per_window.size))
    # The episode labels are what separate 229 draws from 24 clusters. A
    # hand-built result carries none, and then the naive figure is all there is
    # -- which is reported under its own name either way, so the record says
    # which ruler the verdict used rather than leaving it to be inferred.
    clustered = float("nan")
    if result.window_episode is not None:
        labels = np.asarray(result.window_episode)[result.changed]
        clustered = cluster_standard_error(per_window, labels)
    return {
        "mean": float(per_window.mean()),
        "se": independent if not np.isfinite(clustered) else clustered,
        "se_independent": independent,
        "delta_final": float(curve[-1]),
        "steps_outside": int((np.abs(curve) > band).sum()),
        "expected_outside": 0.05 * steps,
        "delta": curve,
        "band": band,
        "window_mean": window_mean,
    }


def embedding_block(result) -> dict | None:
    """One rung's probe-free reading, or None for a rung never measured there.

    The per-window numerator and its per-step curve, the two medians and
    their ratio -- see `PairedDelta.embedding_ratio` for what 0, below 1 and
    above 1 mean -- then the OTHER summary of the same series, the median of
    per-window ratios, and the per-step ratio of the two window-mean curves.
    Both extras exist because the horizon-mean ratio of medians hides two
    things measured on the shipped cells: the two summaries straddle 1.0 on
    frozen_ssl (0.97 against 1.06), and the per-step ratio rises from 0.67 at
    step 1 to 1.14 at step 45 while the horizon mean reads "about 1". The
    curve ratio is NaN, never inf, at a step where the ruler's curve is 0.
    None, never a zero block, for a hand-built result: nothing downstream
    may print an unmeasured rung as a null.
    """
    if result.window_embedding_distance is None or result.window_noise_distance is None:
        return None
    if result.noise_distance_curve is None:
        raise ValueError(
            f"rung {result.name!r} carries a noise ruler per window but no ruler curve, "
            "so its per-step ratio cannot be read; the library sets both together"
        )
    curve = np.asarray(result.embedding_distance_curve, dtype=float)
    noise_curve = np.asarray(result.noise_distance_curve, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        curve_ratio = np.where(noise_curve == 0.0, np.nan, curve / noise_curve)
    return {
        "window_distance": np.asarray(result.window_embedding_distance, dtype=float),
        "curve": curve,
        "median": result.embedding_median(),
        "noise_median": result.noise_median(),
        "ratio_of_medians": result.embedding_ratio(),
        "median_of_ratios": result.median_of_ratios(),
        "curve_ratio": curve_ratio,
    }


def rung_block(result) -> dict:
    """One rung's numbers: its OWN counts, then one `delta_summary` per channel.

    The same assembly for every rung and for every held action of the
    constant rung, so the shuffled rung's numbers are what they were when it
    ran alone and the constant rung's cannot be a different statistic under
    the same heading. A contrast additionally carries the pair it is the
    difference of and each held action's own block against the real sequence
    -- the contrast's count says the two held sequences differ, and only the
    held blocks say how far each sits from the window it replaced.
    """
    episodes = 0
    if result.window_episode is not None:
        episodes = int(np.unique(np.asarray(result.window_episode)[result.changed]).size)
    block = {
        "windows_total": result.windows_total,
        "windows_changed": result.windows_changed,
        "episodes": episodes,
        "steps": int(result.horizon),
        "steps_changed_mean": result.mean_steps_changed,
        "steps_changed_min": result.min_steps_changed,
        "multiset_distance_mean": result.mean_multiset_distance,
        # The per-window mask, persisted so a reader can reproduce the
        # changed-window mean from the all-window series.
        "window_steps_changed": np.asarray(result.window_steps_changed, dtype=int),
        **{metric: delta_summary(result, metric) for metric in CHANNELS},
        "embedding": embedding_block(result),
    }
    held = getattr(result, "held", None)
    if held is not None:
        block["contrast"] = tuple(int(a) for a in result.contrast)
        block["held"] = {int(action): rung_block(arm) for action, arm in held.items()}
    return block


def selfcheck_table(cells: dict, arms, seeds, smallest_k: int) -> str:
    """The three exact equalities, per cell, and the probe they are read through.

    THE LAST COLUMN BUT ONE CARRIES THE k IT WAS COMPUTED AT. The floor alarm
    is evaluated at the smallest k in `--ks`, which is 1 only by default; under
    `--ks 5 45` it is a statement about k=5, and self-check 2's whole point is
    the prior/posterior separation at ONE step -- at k=5 it is a different and
    weaker claim. A hardcoded "k1" heading would report that claim under the
    other one's name.

    `probe_r2` is the embedding probe's selection R^2 from the study record --
    the one number that says whether the position channel can register
    anything. On every `cnn` cell it is -0.036: the probe is a constant
    predictor, floor equals persistence equals open loop, and no action effect
    of any size could reach the ladder through it. Printing it beside the
    self-checks is what stops a null through a degenerate probe from being
    read as a null about the dynamics.

    THE LAST TWO COLUMNS GUARD THE EMBEDDING-SPACE RULER. `noise_restored`
    is whether the stream came back bitwise -- both generators -- after every
    window's noise draw; `noise_same` is how many windows' noise reference
    was the canonical imagination bitwise. The first must read True and the
    second 0, or the rungs no longer share a stream and every ratio above is
    read against a ruler that measured nothing.
    """
    lines = [
        f"{'arm':<12}{'seed':>6}{'open_loop_k':>14}{'record_repro':>14}"
        f"{'stream_drift':>14}{f'k{smallest_k}_is_floor':>13}{'probe_r2':>10}"
        f"{'noise_restored':>16}{'noise_same':>12}"
    ]
    for arm in arms:
        for seed in seeds:
            cell = cells.get((arm, seed))
            row = f"{arm:<12}{seed:>6}"
            if cell is None:
                row += (
                    f"{MISSING:>14}{MISSING:>14}{MISSING:>14}{MISSING:>13}{MISSING:>10}"
                    f"{MISSING:>16}{MISSING:>12}"
                )
            else:
                row += (
                    f"{cell['open_loop']:>14.3e}{cell['record']:>14.3e}"
                    f"{cell['stream']:>14.3e}{str(cell['smallest_k_is_floor']):>13}"
                    f"{cell['probe_selection_r2']:>10.3f}"
                    f"{str(cell['noise_restored']):>16}{cell['noise_collapsed']:>12}"
                )
            lines.append(row)
    return "\n".join(lines)


LADDER_ROWS: tuple[str, ...] = (
    "horizon-mean", "angle horizon-mean", "final step", "changed/total",
    "steps moved/horizon", "multiset dist/horizon", "embed ratio num/noise",
    "embed contrast/noise", "embed step 1|H ratio",
)
"""`ladder_table`'s rows per rung, in print order."""


def _ratio_text(embedding: dict | None) -> str:
    """The embedding-ratio cell: MISSING for a rung never measured there, the
    ratio to three places, or -- undefined -- WHY: `nan (noise 0)` for a ruler
    that measured no spread, never a 0.000 that would read as action-blind;
    `nan (no window)` when the ruler was positive and no window changed."""
    if embedding is None:
        return MISSING
    ratio = float(embedding["ratio_of_medians"])
    if np.isfinite(ratio):
        return f"{ratio:.3f}"
    return "nan (noise 0)" if float(embedding["noise_median"]) == 0.0 else "nan (no window)"


def _step_ratio_text(embedding: dict | None) -> str:
    """The per-step ratio at the first and last horizon step, `a|b`, or
    MISSING; a NaN step prints as `nan`, never as a number."""
    if embedding is None:
        return MISSING
    curve = np.asarray(embedding["curve_ratio"], dtype=float)
    return "|".join(f"{v:.3f}" if np.isfinite(v) else "nan" for v in (curve[0], curve[-1]))


def ladder_table(cells: dict, arms, seeds, rungs) -> str:
    """Per-arm, per-RUNG rows in `LADDER`'s order; per-SEED columns keyed by
    `(arm, seed)`.

    SEVEN rows per rung. The first three are three different statistics rather
    than one printed three times. `horizon-mean` is the position decision
    statistic -- the mean over the horizon of each changed window's delta,
    read against its own spread -- and `angle horizon-mean` is the same on
    the heading channel, in degrees. `final step` is the position endpoint the
    study's headline is quoted at. Measured on frozen_ssl/seed0 the position
    mean and endpoint differ by a factor of four (0.995 against 4.065),
    because the delta curve trends monotonically from -0.589 at step 1 to
    +4.065 at step 45 and the horizon mean is exactly the summary that cancels
    a late-horizon effect against the early steps. Printing one under the
    other's name is a fourfold misreport with nothing to show for it.

    The seventh row is the reading no probe intervenes in: the rung's median
    embedding-space distance to the REAL imagination over the median distance
    between two draws of that imagination -- 0 for an action-blind prior,
    below 1 for a response smaller than the two-draw distance, above 1 for
    one larger. It is the row that can be read on the pixel arm, whose probe
    reads a constant. FOR THE CONSTANT RUNG IT PRINTS THE TWO CONTRAST
    ACTIONS' OWN HELD-VS-REAL RATIOS, `first|second` in contrast order, so the
    row is the same estimand at every rung; the contrast's own numerator is
    the distance between two COUNTERFACTUALS, structurally larger than
    either's distance to real, and printing it in this row put a different
    estimand on top of the ladder. It gets the eighth row, `embed
    contrast/noise`, which the sequence rungs leave blank. The ninth is the
    rung's own per-step ratio at step 1 and step H -- for the constant rung,
    the contrast's, directly under its row -- because a horizon mean near 1
    hides a per-step ratio that crosses 1 late (0.67 -> 1.14 on frozen_ssl).

    The three rows before it are what say whether any of those numbers means
    anything: how many windows the rung changed at all, how many of the
    horizon's steps it moved per window on average AND at least, and how far
    the multiset moved. None is redundant with another. A held action on a
    window already dominated by it "changes" that window by one step, and the
    minimum is the only row that shows it; the resampled rung moves 35 of 45
    steps as a sequence while moving the counts by 15, and the multiset row
    is the only one that shows THAT -- it reads 0 / 15 / 45 up the ladder on
    the shipped split, which is the monotonicity the table is meant to show.

    `rungs` is the ladder to print, in order; a cell that did not run one of
    them prints MISSING in that rung's rows rather than another rung's numbers,
    so a partial `--rungs` re-run beside full cells cannot look complete.
    """
    lines = [
        f"{'arm':<12}{'rung':<11}{'row':<22}" + "".join(f"{f'seed {s}':>18}" for s in seeds)
    ]
    for arm in arms:
        # The arm label is printed once per arm, on its first rung; the lookup
        # key below is `arm` itself and must not be the blanked label.
        for index, rung in enumerate(rungs):
            label = arm if index == 0 else ""
            rows = {name: f"{label if name == LADDER_ROWS[0] else '':<12}"
                          f"{rung if name == LADDER_ROWS[0] else '':<11}{name:<22}"
                    for name in LADDER_ROWS}
            for seed in seeds:
                cell = cells.get((arm, seed))
                block = None if cell is None else cell["rungs"].get(rung)
                if block is None:
                    for name in LADDER_ROWS:
                        rows[name] += f"{MISSING:>18}"
                    continue
                position, angle = block["position"], block["angle"]
                steps = block["steps"]
                rows["horizon-mean"] += (
                    f"{position['mean']:>+11.3f} +-{NOMINAL_Z * position['se']:>4.2f}"
                )
                rows["angle horizon-mean"] += (
                    f"{angle['mean']:>+11.3f} +-{NOMINAL_Z * angle['se']:>4.2f}"
                )
                rows["final step"] += f"{position['delta_final']:>+18.3f}"
                rows["changed/total"] += (
                    f"{block['windows_changed']:>10}/{block['windows_total']:<7}"
                )
                moved = (
                    f"{block['steps_changed_mean']:.1f}(min {block['steps_changed_min']})/{steps}"
                )
                rows["steps moved/horizon"] += f"{moved:>18}"
                rows["multiset dist/horizon"] += (
                    f"{block['multiset_distance_mean']:>10.1f}/{steps:<7}"
                )
                if "contrast" in block:
                    held = block.get("held", {})
                    pair = "|".join(
                        _ratio_text(held.get(action, {}).get("embedding"))
                        for action in block["contrast"]
                    )
                    rows["embed ratio num/noise"] += f"{pair:>18}"
                    rows["embed contrast/noise"] += f"{_ratio_text(block.get('embedding')):>18}"
                else:
                    rows["embed ratio num/noise"] += f"{_ratio_text(block.get('embedding')):>18}"
                    rows["embed contrast/noise"] += f"{'':>18}"
                rows["embed step 1|H ratio"] += f"{_step_ratio_text(block.get('embedding')):>18}"
            lines.extend(rows[name] for name in LADDER_ROWS)
    return "\n".join(lines)


def held_table(cells: dict, arms, seeds, rungs) -> str:
    """Every held action of the constant rung against the REAL sequence.

    The contrast is the decision, and it is a difference of two of these
    rows; the rows themselves are what let a reader see which held actions
    move the imagined position and in which direction, and how far each held
    sequence sits from the windows it replaced. Four rows per action: the
    position horizon-mean delta with its band, steps moved per window with
    the minimum, then the held action's OWN embedding-space ratio against
    the real imagination -- the same axis as the ladder's sequence rungs,
    which is what makes the top rung readable beside them -- and that ratio
    at step 1 and step H. Printed only when the constant rung ran.
    """
    if "constant" not in rungs:
        return ""
    lines = [
        f"{'arm':<12}{'held':<13}{'row':<22}" + "".join(f"{f'seed {s}':>18}" for s in seeds)
    ]
    for arm in arms:
        actions = sorted(
            {
                action
                for seed in seeds
                for action in (cells.get((arm, seed), {}).get("rungs", {})
                               .get("constant", {}).get("held", {}))
            }
        )
        for index, action in enumerate(actions):
            label = arm if index == 0 else ""
            delta = f"{label:<12}{action_name(action):<13}{'horizon-mean':<22}"
            moved = f"{'':<12}{'':<13}{'steps moved/horizon':<22}"
            ratio = f"{'':<12}{'':<13}{'embed ratio num/noise':<22}"
            steps = f"{'':<12}{'':<13}{'embed step 1|H ratio':<22}"
            for seed in seeds:
                cell = cells.get((arm, seed))
                block = (
                    None if cell is None
                    else cell["rungs"].get("constant", {}).get("held", {}).get(action)
                )
                if block is None:
                    delta += f"{MISSING:>18}"
                    moved += f"{MISSING:>18}"
                    ratio += f"{MISSING:>18}"
                    steps += f"{MISSING:>18}"
                    continue
                position = block["position"]
                delta += f"{position['mean']:>+11.3f} +-{NOMINAL_Z * position['se']:>4.2f}"
                text = (
                    f"{block['steps_changed_mean']:.1f}(min {block['steps_changed_min']})"
                    f"/{block['steps']}"
                )
                moved += f"{text:>18}"
                ratio += f"{_ratio_text(block.get('embedding')):>18}"
                steps += f"{_step_ratio_text(block.get('embedding')):>18}"
            lines.extend((delta, moved, ratio, steps))
    return "\n".join(lines)


def noise_table(cells: dict, arms, seeds) -> str:
    """The ruler's own shape: its window-mean curve at step 1 and step H, and
    the growth between them, per cell.

    The horizon-mean ratio cannot show this, and measured on the shipped
    cells it is the difference between the arms: cnn's ruler is flat from
    step 2 (0.12 -> 0.13) while frozen_ssl's grows ~6x (2.1 -> 12.5). A flat
    ruler means the imagination does not diverge with the horizon in
    embedding space -- a one-step response on a memoryless latent -- and a
    reader shown only "ratio 0.3 at every rung" would call the two arms the
    same shape. MISSING for a cell that did not run.
    """
    lines = [f"{'arm':<12}{'row':<22}" + "".join(f"{f'seed {s}':>18}" for s in seeds)]
    for arm in arms:
        ruler = f"{arm:<12}{'noise ruler step 1->H':<22}"
        growth = f"{'':<12}{'ruler growth H/1':<22}"
        for seed in seeds:
            cell = cells.get((arm, seed))
            if cell is None or cell.get("noise_reference") is None:
                ruler += f"{MISSING:>18}"
                growth += f"{MISSING:>18}"
                continue
            curve = np.asarray(cell["noise_reference"]["curve"], dtype=float)
            first, last = float(curve[0]), float(curve[-1])
            ruler += f"{f'{first:.3f}->{last:.3f}':>18}"
            factor = f"x{last / first:.2f}" if first > 0.0 else "nan (step 1 = 0)"
            growth += f"{factor:>18}"
        lines.extend((ruler, growth))
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


STATUS_ORDER: tuple[str, ...] = ("uninterpretable", "null", "nominal", "responds")
"""A rung's status is the STRONGEST of its channels', in this order."""


def _channel_status(channel: dict, threshold: float) -> str:
    """One channel's reading against BOTH rulers.

      * "null"     -- inside the nominal +-2 SE band.
      * "nominal"  -- outside it, but inside the family-wise threshold: the
                      excursion a family of this size produces by chance, and
                      NOT counted as a response. Named rather than folded into
                      "null", because a null headline over a nominal excursion
                      would overclaim in the other direction.
      * "responds" -- outside the family-wise threshold.

    `abs(mean) <= NOMINAL_Z * se` rather than a z-ratio, so an exact zero over
    an exact zero spread -- an action-blind prior on the matched stream --
    reads null and not NaN.
    """
    mean, se = float(channel["mean"]), float(channel["se"])
    if not (np.isfinite(mean) and np.isfinite(se)):
        return "uninterpretable"
    if abs(mean) <= NOMINAL_Z * se:
        return "null"
    if abs(mean) <= threshold * se:
        return "nominal"
    return "responds"


def _z(channel: dict) -> float:
    mean, se = float(channel["mean"]), float(channel["se"])
    if se == 0.0:
        return 0.0 if mean == 0.0 else float("inf")
    return abs(mean) / se


def _rung_reading(name: str, block: dict, threshold: float, family: int) -> tuple[str, str]:
    """One rung's line of the verdict, and its status.

    FOUR statuses, and the first is not a reading:

      * "uninterpretable" -- the intervention was a no-op on every window, so
        the delta is undefined; or exactly one window changed, so
        `delta_summary` returned `mean=nan, se=inf` and the count guard does
        not fire because 1 is not 0. `abs(nan) <= 2*inf` is FALSE, so without
        this branch the second case fell into the positive one and printed "by
        +nan map units against +-2 SE (inf) ... M4 is NOT blocked" -- an
        affirmative claim manufactured out of a NaN. The gate is on the
        STATISTIC BEING DEFINED, not on the count.
      * "null", "nominal", "responds" -- see `_channel_status`; the rung takes
        the strongest of its two channels', so a response on the angle channel
        alone is a response.

    THE DECISION IS THE AGGREGATE, never the per-step exceedance count. Two
    standard errors is ~95% PER STEP, so under the null roughly 2 of 45 steps
    fall outside by chance and a rule that fired on one of them would call
    almost every cell responsive. The count is PRINTED on the line, with what
    chance alone gives, because the aggregate can also hide an effect that
    cancels across the horizon -- neither number is trustworthy without the
    other. The steps-moved count, its minimum and the multiset distance are
    printed beside the changed-window count for the reason `ladder_table`
    gives: a rung that moved one step of forty-five "changed" the window and
    intervened on almost nothing.

    THE SIGN IS PRINTED WITH EVERY EXCURSION, because the verdict reads it. A
    positive delta is the intervened sequence imagining WORSE than the real
    one -- for the contrast, held MOVE_FORWARD running further than held
    NOOP -- which is what conditioning on the action looks like. A negative
    one is an intervened sequence doing better, which is not conditioning an
    actor can use.
    """
    counted = f"{block['windows_changed']}/{block['windows_total']} windows changed"
    if block["windows_changed"] == 0:
        return "uninterpretable", (
            f"  {name:<10} UNINTERPRETABLE: {counted}, so this rung compared a "
            "sequence with itself and no comparison was made."
        )
    statuses = {metric: _channel_status(block[metric], threshold) for metric in CHANNELS}
    if all(status == "uninterpretable" for status in statuses.values()):
        return "uninterpretable", (
            f"  {name:<10} UNINTERPRETABLE: {counted}, and a spread cannot be "
            "estimated from one sample; the aggregate delta and its standard "
            "error are both undefined."
        )
    status = max(statuses.values(), key=STATUS_ORDER.index)
    excursions = [metric for metric in CHANNELS if statuses[metric] in ("nominal", "responds")]
    if status == "null":
        verdict = "inside the +-2 SE band on both channels"
    elif status == "nominal":
        verdict = (
            f"outside the +-2 SE band on {_join(excursions)} but inside the family-wise "
            f"threshold (z={threshold:.2f} for {family} comparisons), which is the "
            "excursion chance produces in a family this size"
        )
    else:
        verdict = (
            f"OUTSIDE the family-wise threshold (z={threshold:.2f} for {family} "
            f"comparisons) on {_join(m for m in excursions if statuses[m] == 'responds')}"
        )
    what = f"perturbing {LADDER_PERTURBS[name]}"
    if "contrast" in block:
        first, second = block["contrast"]
        what += f" -- held {action_name(first)} minus held {action_name(second)}"
    position, angle = block["position"], block["angle"]
    peak = float(np.nanmax(np.abs(np.asarray(position["delta"], dtype=float))))
    signs = "; ".join(
        f"sign on {metric} {'+' if block[metric]['mean'] > 0 else '-'}"
        f" ({'worse' if block[metric]['mean'] > 0 else 'BETTER'} under the intervention)"
        for metric in excursions
    )
    return status, (
        f"  {name:<10} {verdict}, {what}: position {position['mean']:+.3f} map units "
        f"against +-2 SE ({NOMINAL_Z * position['se']:.3f}), z={_z(position):.2f}; angle "
        f"{angle['mean']:+.3f} deg against +-2 SE ({NOMINAL_Z * angle['se']:.3f}), "
        f"z={_z(angle):.2f}{'; ' + signs if signs else ''}; largest per-step position "
        f"move {peak:.3f}; {counted} in {block['episodes']} episodes, "
        f"{block['steps_changed_mean']:.1f}/{block['steps']} steps moved per window "
        f"(min {block['steps_changed_min']}), multiset distance "
        f"{block['multiset_distance_mean']:.1f}/{block['steps']}; per-step |position "
        f"delta| outside its own band at {position['steps_outside']}/{block['steps']} "
        f"steps, against {position['expected_outside']:.1f} expected by chance."
    )


def _join(phrases) -> str:
    phrases = list(phrases)
    if len(phrases) < 2:
        return "".join(phrases)
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def _usable(block: dict, threshold: float) -> bool:
    """Is this rung's response one an actor could use: positive on at least
    one channel that clears the threshold. A response that is only a DECREASE
    in error under an intervened sequence is not conditioning an actor can
    use, whatever its z."""
    return any(
        _channel_status(block[metric], threshold) == "responds" and block[metric]["mean"] > 0
        for metric in CHANNELS
    )


def probe_is_measurable(cell: dict, widest_se: float) -> bool:
    """Can the position probe register an action effect at all.

    Two ways for the answer to be no, and every `cnn` cell is both: the
    persistence-to-floor range at the final horizon step is non-positive --
    the floor sits above persistence, so the probe is a constant predictor and
    no effect of any size can reach the ladder through it -- or the widest null
    band is wider than the whole range, so the equivalence bound would be over
    100% and "no effect larger than the entire actionable range" bounds
    nothing. A null through such a probe is a statement about the probe.
    """
    band = float(cell["persistence"]) - float(cell["floor"])
    return band > 0.0 and NOMINAL_Z * widest_se <= band


def verdict_block(arm: str, seed: int, cell: dict, *, family: int) -> str:
    """What the ladder licenses saying about M4, for one cell.

    One line per rung, in the ladder's order, each with its own numbers and
    its own status -- then ONE sentence decided over the ladder as a whole.
    `family` is the number of comparisons the whole run makes (rungs x cells x
    channels), and every rung is decided against `family_threshold(family)`
    rather than against its own +-2 SE band, because a per-cell, uncorrected
    verdict turns the null's expected false-positive count into a finding.
    The decision reads the rungs' statuses by POSITION, never by name, so a
    fourth rung would take its place in the same logic:

      * NO INTERPRETABLE RUNG -- every rung was a no-op or left a single
        changed window. No comparison was made anywhere; no verdict.
      * A RESPONSE at some rung -- say so with the rungs that responded and
        the rungs that did not, and their signs. Rungs BELOW the lowest
        response that did not respond are the finding: the dynamics are
        action-conditioned but insensitive to what those rungs perturb, which
        is the better result the ladder was built to reach and is reported as
        that rather than folded into the null story. Rungs ABOVE a response
        that did not respond are flagged NON-MONOTONE: the rungs are in
        strictly increasing order of perturbation, so a response that
        vanishes under a larger perturbation is not a ladder reading, and one
        seed is one draw. A response is what UNBLOCKS M4 on this evidence
        only when it has the sign conditioning predicts; a wrong-signed one
        is named as that and unblocks nothing.
      * NOMINAL EXCURSIONS ONLY -- some rung is outside its own band and
        inside the family-wise threshold, none clears it. Inconclusive: not a
        response, and not the clean null the equivalence statement needs.
      * EVERY INTERPRETABLE RUNG NULL -- the prior did not measurably use the
        action under any perturbation run. This licenses blocking M4 only
        when (a) the position probe can register an effect at all, (b) the
        per-action contrast was among the rungs, so the null is not about
        order and counts alone, and (c) the family-wise threshold was the
        ruler -- which it always is here. Without (a) the verdict reads
        UNMEASURABLE THROUGH THIS PROBE; without (b) it says what was and was
        not tested.

    Uninterpretable rungs are named on their own line and EXCLUDED from the
    decision -- never counted as nulls, which is what a no-op arm would read
    as if it fell through.

    ACCEPTING THE NULL REQUIRES A SCALE, or the same sentence covers evidence
    that differs by three orders of magnitude -- measured across the nine
    shipped cells the band runs from 0.014 to 3.45 map units. So the band is
    also reported as a fraction of the persistence-to-floor range at the final
    horizon step, which is the range the model would have to close to be worth
    anything, and the verdict states the equivalence bound that follows. Over
    a ladder the bound is quoted at the WIDEST band among the null rungs: an
    effect smaller than the widest band could have hidden at that rung, so
    quoting the narrowest would claim one rung's precision for the whole
    ladder. The bound is a bound on the effect of the SPECIFIC perturbations
    run, and the sentence says so; it is not a bound on "an action effect".
    """
    threshold = family_threshold(family)
    head = f"--- {arm} seed {seed} ---"
    order = tuple(cell["rung_order"])
    readings = {
        name: _rung_reading(name, cell["rungs"][name], threshold, family) for name in order
    }
    lines = [head, *(text for _, text in readings.values())]
    by_status = {
        status: [name for name in order if readings[name][0] == status]
        for status in STATUS_ORDER
    }
    null, nominal, responds = by_status["null"], by_status["nominal"], by_status["responds"]

    if not (null or nominal or responds):
        lines.append(
            "UNINTERPRETABLE: no rung of the ladder changed enough windows to "
            "estimate a spread, so no comparison was made at any rung. No verdict."
        )
        return "\n".join(lines)

    if responds:
        lowest = order.index(responds[0])
        quiet = null + nominal
        below = [name for name in order if name in quiet and order.index(name) < lowest]
        above = [name for name in order if name in quiet and order.index(name) > lowest]
        sentence = (
            f"THE IMAGINATION RESPONDS TO THE ACTION at rung(s) {', '.join(responds)}"
        )
        if quiet:
            sentence += f", and not at {', '.join(name for name in order if name in quiet)}"
        if nominal:
            sentence += (
                f" (nominal excursion only at {', '.join(nominal)}, inside the "
                f"family-wise threshold)"
            )
        sentence += "."
        if below:
            sentence += (
                " The dynamics are action-conditioned but insensitive to "
                f"{_join(LADDER_PERTURBS[name] for name in below)}, which is what the "
                "rung(s) below the response perturb."
            )
        if above:
            sentence += (
                f" NON-MONOTONE: {', '.join(above)} perturb strictly more than "
                f"{responds[0]} yet resolved nothing, so this is not a ladder reading "
                "-- one intervention seed is one draw, and neither finding follows "
                "from it."
            )
        usable = [name for name in responds if _usable(cell["rungs"][name], threshold)]
        wrong = [name for name in responds if name not in usable]
        if wrong:
            sentence += (
                f" The response at {', '.join(wrong)} is WRONG-SIGNED: the intervened "
                "sequence imagined a smaller error than the one it replaced, which is "
                "not conditioning an actor can use -- read it as confusion under an "
                "input the model was not trained on, not as a usable action pathway."
            )
        if usable:
            sentence += (
                f" The response at {', '.join(usable)} has the sign conditioning "
                "predicts. M4 is NOT blocked on this evidence."
            )
        else:
            sentence += " M4 is neither blocked nor unblocked by a wrong-signed response."
        lines.append(sentence)
        return "\n".join(lines) + _reproduction_note(cell)

    if nominal:
        lines.append(
            f"INCONCLUSIVE: {', '.join(nominal)} sit outside their own +-2 SE band "
            f"but inside the family-wise threshold (z={threshold:.2f} for {family} "
            f"comparisons; {0.05 * family:.2f} such excursions are expected by chance "
            "across the family), and no rung clears it. That is not a response, and "
            "it is not the clean null an equivalence statement needs, so neither "
            "finding follows and M4 is neither blocked nor unblocked on this evidence."
        )
        return "\n".join(lines) + _reproduction_note(cell)

    widest = max(float(cell["rungs"][name]["position"]["se"]) for name in null)
    if not probe_is_measurable(cell, widest):
        lines.append(
            "UNMEASURABLE THROUGH THIS PROBE: every rung is inside its band, but the "
            f"position probe cannot register an action effect here (selection R^2 "
            f"{float(cell['probe_selection_r2']):+.3f}; persistence-to-floor range "
            f"{float(cell['persistence']) - float(cell['floor']):+.3f} map units at the "
            f"final step against a widest null band of {NOMINAL_Z * widest:.3f}). No "
            "action effect of any size could have reached the ladder through it, so "
            "this says nothing about whether the prior uses the action and licenses "
            "no claim about M4." + _embedding_note(cell)
        )
        return "\n".join(lines) + _reproduction_note(cell)
    if "constant" not in null:
        lines.append(
            "NO RESPONSE TO THE PERTURBATIONS RUN: at every rung tested "
            f"({', '.join(null)}) the delta stayed inside its band on both channels."
            f"{_equivalence(cell, widest, null)} The per-action contrast was NOT among "
            "the rungs, so this is a null about order and counts only -- an "
            "action-conditioned prior that reads the identity of the action would "
            "pass it -- and it does not license blocking M4."
        )
        return "\n".join(lines) + _reproduction_note(cell)
    lines.append(
        "THE DYNAMICS PRIOR DOES NOT MEASURABLY USE THE ACTION under any perturbation "
        f"run: at every rung ({', '.join(null)}) the delta stayed inside its band on "
        f"both channels, including the held-action contrast.{_equivalence(cell, widest, null)} "
        "An actor trained inside this world model would find every action sequence "
        "tested imagines the same future, and M4 is BLOCKED on this evidence."
    )
    return "\n".join(lines) + _reproduction_note(cell)


def _embedding_note(cell: dict) -> str:
    """The one reading a degenerate probe cannot attenuate, quoted where the
    verdict says the probe registered nothing. Appended only when the
    constant rung ran and was measured there; the decision logic is
    untouched -- the pooled script is the decision surface for this number.

    FOUR THINGS, in this order, because each corrects a misreading of the
    one before. The two contrast actions' OWN held-vs-real ratios -- the same
    axis as the rungs below. The between-held contrast, named as a distance
    between two counterfactuals and NOT on that axis. The contrast's per-step
    ratio at step 1 and step H, because its horizon mean sits near 1 on
    frozen_ssl while the per-step ratio runs 0.67 -> 1.14. And the ruler's
    own growth over the horizon, because a ratio of 0.3 at every rung reads
    the same on a ruler that is flat (cnn) and one that grows 6x
    (frozen_ssl), and those are not the same shape of response.
    """
    block = cell["rungs"].get("constant")
    if block is None or block.get("embedding") is None:
        return ""
    first, second = block["contrast"]
    held = block.get("held", {})
    own = "; ".join(
        f"held {action_name(action)} vs real {_ratio_text(held.get(action, {}).get('embedding'))}"
        for action in (first, second)
    )
    embedding = block["embedding"]
    at_first, at_last = _step_ratio_text(embedding).split("|")
    steps = int(np.asarray(embedding["curve_ratio"]).size)
    note = (
        f" In embedding space, where no probe intervenes, each held action's distance "
        f"to the real imagination over the distance between two draws of it reads: "
        f"{own} (0 would be action-blind; 1 is the two-draw distance, about sqrt(2) "
        f"times one draw's own spread). The contrast's numerator is the distance "
        f"between the two held imaginations -- held {action_name(first)} against held "
        f"{action_name(second)} -- and reads {_ratio_text(embedding)}; it is a "
        f"between-counterfactual distance, structurally larger than either held "
        f"action's distance to real, and not on the same axis as the rungs. Its "
        f"per-step ratio is step 1 {at_first}, step {steps} {at_last}"
    )
    noise = cell.get("noise_reference")
    if noise is not None:
        curve = np.asarray(noise["curve"], dtype=float)
        note += (
            f"; the ruler grows {curve[0]:.3f} -> {curve[-1]:.3f} over the horizon "
            f"(a flat ruler is an imagination that does not diverge with the horizon)"
        )
    return note + "."


def _equivalence(cell: dict, se: float, rungs) -> str:
    """The band, against the range the model would have to close to matter.

    Without it "the prior does not use the action" is the same sentence for a
    cell whose band is 0.014 map units and one whose band is 3.45. With it the
    claim becomes a bound a reader can argue with -- and it is a bound on the
    effect of the perturbations that were RUN, named, not on an action effect
    in general: measured, a per-action contrast on the same checkpoints found
    an effect of 110% of this range where the shuffled and resampled rungs
    together bounded it at 16%.
    """
    band = float(cell["persistence"]) - float(cell["floor"])
    return (
        f" Against the persistence-to-floor range of {band:.3f} map units at the "
        f"final horizon step, the widest null band is {200.0 * se / band:.1f}"
        "% of the range, so what this licenses is the equivalence statement: an "
        f"effect of {_join(LADDER_PERTURBS[name] for name in rungs)} larger than that "
        "fraction of the actionable range would have been detected, and none was. It "
        "bounds those perturbations and no other."
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


def sign_test_p(positives: int, signed: int) -> float:
    """Exact two-sided sign test: the probability, under a symmetric null,
    of a sign split at least as lopsided as `positives` of `signed`.

    Nine cells all positive -- measured, the constant rung's contrast -- is
    2 * 0.5^9 = 0.0039, a pattern no per-cell z can see. It is a PATTERN
    statistic and not an independent replication: the nine cells share the
    same 229 windows, so the header that prints it says so. 1.0 when no cell
    carries a sign.
    """
    if signed == 0:
        return 1.0
    extreme = max(positives, signed - positives)
    tail = sum(comb(signed, k) for k in range(extreme, signed + 1)) * 0.5**signed
    return float(min(1.0, 2.0 * tail))


def cross_cell_table(cells: dict, arms, seeds, rungs, *, family: int) -> str:
    """The pattern the per-cell verdicts cannot see: per rung and channel,
    across every cell, how many sit outside their own band, how many clear the
    family-wise threshold, and the SIGN of every cell's delta in cell order.

    Two things only this table shows. A single "responds" cell is one
    rejection in a family; the count of nominal excursions against the number
    chance produces (5% of the cells) is what says whether the family is
    behaving like a null. And a rung whose delta carries the same sign in
    every cell -- measured, the contrast is positive in 9 of 9 -- is a
    cross-cell pattern with no per-cell z at all, so the exact sign-test
    probability is printed beside the signs, under the caveat that the cells
    share their windows.
    """
    threshold = family_threshold(family)
    lines = [
        f"family: {family} comparisons ({len(rungs)} rungs x {len(cells)} cells x "
        f"{len(CHANNELS)} channels); family-wise threshold z={threshold:.2f}; "
        f"{0.05 * len(cells):.2f} excursions per rung and channel expected by chance; "
        "sign p is the exact two-sided sign test over the signed cells, and the cells "
        "share windows",
        f"{'rung':<11}{'channel':<10}{'cells':>6}{'outside 2SE':>13}{'clear z_fw':>12}"
        f"{'signs (cell order)':>22}{'sign p':>8}",
    ]
    for rung in rungs:
        for metric in CHANNELS:
            blocks = [
                cells[(arm, seed)]["rungs"][rung]
                for arm in arms for seed in seeds
                if (arm, seed) in cells and rung in cells[(arm, seed)]["rungs"]
            ]
            statuses = [
                "uninterpretable" if block["windows_changed"] == 0
                else _channel_status(block[metric], threshold)
                for block in blocks
            ]
            # `0` for an EXACT zero: the sign an action-blind prior produces
            # bitwise on the matched stream, and neither + nor -.
            signs = "".join(
                "?" if status == "uninterpretable"
                else "0" if block[metric]["mean"] == 0.0
                else "+" if block[metric]["mean"] > 0 else "-"
                for block, status in zip(blocks, statuses)
            )
            outside = sum(status in ("nominal", "responds") for status in statuses)
            clear = sum(status == "responds" for status in statuses)
            p = sign_test_p(signs.count("+"), signs.count("+") + signs.count("-"))
            lines.append(
                f"{rung:<11}{metric:<10}{len(blocks):>6}{outside:>13}{clear:>12}{signs:>22}"
                f"{p:>8.3f}"
            )
    return "\n".join(lines)


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
    ladder = action_intervention_ladder(
        model, val, embedding_probe, arms=args.rungs,
        intervention_seed=args.intervention_seed, contrast=args.contrast, **common,
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
            np.abs(ladder.real.rssm_position - reference.rssm_position).max()
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
        # The probe the whole ladder is read through, from the study record:
        # read STRICTLY, because a record without it is not a study record and
        # a default here would print a plausible number for a probe nobody
        # fit. On every cnn cell this is -0.036, and the verdict reads it.
        "probe_selection_r2": float(record["probe"]["embedding_selection_r2"]),
        "windows_total": ladder.windows_total,
        # The window -> episode index, ONCE, so a reader can cluster every
        # per-window series below by it.
        "window_episode": ladder.window_episode,
        # The embedding-space ruler and its two self-checks. Read STRICTLY:
        # a ladder without a noise reference is not one this script can
        # gate, and a default here would print a ratio nobody measured.
        "noise_reference": {
            "window_distance": ladder.noise_reference.window_embedding_distance,
            "curve": ladder.noise_reference.embedding_distance_curve,
            "median": float(np.median(ladder.noise_reference.window_embedding_distance)),
        },
        "noise_restored": bool(ladder.noise_reference.stream_restored),
        "noise_collapsed": int(ladder.noise_reference.windows_collapsed),
        "rung_order": ladder.order,
        # One block per rung, in the ladder's order, each carrying its OWN
        # counts beside its own per-channel delta -- the same assembly for
        # every rung and every held action.
        "rungs": {name: rung_block(arm) for name, arm in ladder.arms.items()},
        "held_actions": ladder.held_actions,
        "contrast": ladder.contrast,
        "action_marginal": (
            None if ladder.action_marginal_values is None else {
                "values": [int(v) for v in ladder.action_marginal_values],
                "counts": [int(c) for c in ladder.action_marginal_counts],
            }
        ),
        "k": {k: float(sweep.curve(k)[-1]) for k in args.ks},
        "floor": float(reference.floor_position[-1]),
        "persistence": float(reference.persistence_position[-1]),
        # NOT "se". `delta_summary` contributes each rung's own standard error
        # under that name inside its block and the verdict decides on it; the
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
            # Each rung says which curves it has: one for a sequence rung,
            # one per held action for the contrast, which has no error curve
            # of its own.
            **{
                f"{name}{suffix}_position": curve
                for name, arm in ladder.arms.items()
                for suffix, curve in arm.curves().items()
            },
            "floor_position": reference.floor_position,
            "persistence_position": reference.persistence_position,
            **{f"k{k}_position": sweep.curve(k) for k in args.ks},
        },
    }


def diagnostic_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    """One file per cell, named by BOTH the arm and the seed -- see
    `study.job_record_path` for what a colliding name costs."""
    return Path(out_dir) / f"diagnostic_{arm}_seed{seed}.json"


def _channel_record(channel: dict) -> dict:
    """One channel of one rung, as the record writes it.

    The decision statistic and its spread, then the per-step curves it was
    computed from -- the aggregate alone hides a cancelling effect, the curves
    alone invite the 45-comparison error. The ruler the verdict used and the
    naive one beside it under its own name: the two differ by up to 1.32x on
    the shipped cells and a record that reported only one leaves the reader
    unable to tell which interval the equivalence claim rests on.
    """
    return {
        "delta_mean": channel["mean"],
        "delta_final_step": channel["delta_final"],
        "delta_se": channel["se"],
        "delta_se_independent_windows": channel["se_independent"],
        "steps_outside_band": channel["steps_outside"],
        "steps_expected_outside_by_chance": channel["expected_outside"],
        "delta": [float(v) for v in channel["delta"]],
        "band": [float(v) for v in channel["band"]],
        # Per window over ALL windows, traversal order, index-aligned with
        # `windows.episode`; `delta_mean` above is the mean over the CHANGED
        # ones, and the rung's `window_steps_changed` says which those are.
        "window_delta_mean": [float(v) for v in channel["window_mean"]],
    }


def _embedding_record(block: dict | None) -> dict | None:
    if block is None:
        return None
    return {
        "window_distance": [float(v) for v in block["window_distance"]],
        "curve": [float(v) for v in block["curve"]],
        "median": float(block["median"]),
        "noise_median": float(block["noise_median"]),
        # The decision-bearing summary, and the other one beside it -- see
        # `PairedDelta.embedding_ratio` for why the ratio of medians decides.
        "ratio_of_medians": float(block["ratio_of_medians"]),
        "median_of_ratios": float(block["median_of_ratios"]),
        "curve_ratio": [float(v) for v in block["curve_ratio"]],
    }


def _rung_record(block: dict) -> dict:
    """One rung -- or one held action -- as the record writes it.

    This rung's OWN counts first: how many windows it changed, the clusters
    those came from -- 229 windows cut from 24 episodes are not 229
    independent draws, and a later reader recomputing an interval from
    `total` alone would understate it -- how far it moved each window on
    average and AT LEAST, and how far the multiset moved. Then one block per
    channel. A contrast adds the pair it is the difference of and every held
    action's own block against the real sequence.
    """
    record = {
        "windows_changed": block["windows_changed"],
        "episodes_changed": block["episodes"],
        "steps_changed_mean": block["steps_changed_mean"],
        "steps_changed_min": block["steps_changed_min"],
        "multiset_distance_mean": block["multiset_distance_mean"],
        "window_steps_changed": [int(v) for v in block["window_steps_changed"]],
        **{metric: _channel_record(block[metric]) for metric in CHANNELS},
        # None for a rung never measured in embedding space -- never a zero.
        "embedding": _embedding_record(block.get("embedding")),
    }
    if "contrast" in block:
        record["contrast"] = [int(a) for a in block["contrast"]]
        record["held"] = {
            str(action): _rung_record(held) for action, held in block["held"].items()
        }
    return record


def build_record(cell: dict, args, device, *, family: int) -> dict:
    """The cell's numbers, plus the environment they are only reproducible in.

    The DEVICE and the torch version are recorded because the record
    reproduction is locked to them -- bitwise on mps under torch 2.13.0, and off
    on cpu by an arm-dependent ~6.5 (cnn/seed0) to ~12.4 (frozen_ssl/seed0) map
    units -- and nothing else in the study carries that. Without it a future
    reader cannot tell a sound diagnostic run on another machine from a broken
    one, and the shuffled delta itself moves with the device by more than its
    own standard error.

    THE LADDER'S CHOICES ARE RECORDED BESIDE ITS NUMBERS. The seed every rung
    was derived from, the actions the constant rung held and the pair it
    contrasted, the marginal the resampled rung drew from, and the family the
    verdict was corrected over are none of them recoverable from the deltas.
    The shuffled rung's position block is bitwise what the standalone shuffle
    wrote under `shuffle` before the ladder existed, under the keys `delta_*`
    in place of `position_delta_*`; it is written ONCE, under the rung's own
    name, so a reader diffing the nine shipped cells has exactly one copy.

    EVERY PER-WINDOW SERIES IS CHECKED AGAINST `windows.total` BEFORE THE
    RECORD EXISTS -- the episode index, each rung's and held action's delta
    series and changed mask, the noise reference. A series persisted over
    the changed windows only, or a stale index, would let a pooling reader
    cluster row `w` on episode `w'` with every count looking right; refused
    here by name rather than discovered nine cells later.
    """
    total = int(cell["windows_total"])
    episode = cell["window_episode"]
    if episode is not None and len(episode) != total:
        raise ValueError(
            f"the window -> episode index has {len(episode)} entries for {total} "
            "windows; a stale index would cluster every per-window series wrongly"
        )
    for name, block in cell["rungs"].items():
        _require_window_series(block, name, total)
    noise = cell["noise_reference"]
    if noise is not None and len(noise["window_distance"]) != total:
        raise ValueError(
            f"the noise reference has {len(noise['window_distance'])} entries for "
            f"{total} windows"
        )
    return {
        "arm": cell["arm"],
        "seed": cell["seed"],
        "context": args.context,
        "horizon": args.horizon,
        "ks": list(args.ks),
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
            "noise_reference_stream_restored": cell["noise_restored"],
            "noise_reference_windows_collapsed": cell["noise_collapsed"],
        },
        # The embedding-space ruler, ONCE: every rung's ratio is read against
        # this one series, so a copy under each rung would be nine copies.
        "noise_reference": None if noise is None else {
            "window_distance": [float(v) for v in noise["window_distance"]],
            "curve": [float(v) for v in noise["curve"]],
            "median": float(noise["median"]),
            "windows_collapsed": cell["noise_collapsed"],
            "stream_restored": cell["noise_restored"],
        },
        "probe": {"embedding_selection_r2": cell["probe_selection_r2"]},
        "windows": {
            "total": cell["windows_total"],
            # Per window, in traversal order; null when the ladder carried no
            # clustering, never `range(n)` -- a pooling reader must refuse
            # to cluster on nothing rather than treat every window as its
            # own episode.
            "episode": None if episode is None else [int(e) for e in episode],
        },
        "ladder": {
            "rungs": list(cell["rung_order"]),
            "intervention_seed": args.intervention_seed,
            "held_actions": (
                None if cell["held_actions"] is None else [int(a) for a in cell["held_actions"]]
            ),
            "contrast": (
                None if cell["contrast"] is None else [int(a) for a in cell["contrast"]]
            ),
            "action_marginal": cell["action_marginal"],
            "family": family,
            "family_threshold_z": family_threshold(family),
        },
        "interventions": {
            name: _rung_record(cell["rungs"][name]) for name in cell["rung_order"]
        },
        "curves": {
            name: [float(v) for v in curve]
            for name, curve in cell["curves"].items()
        },
    }


def _require_window_series(block: dict, name: str, total: int) -> None:
    """Every per-window series in one rung's block -- and its held actions'
    -- has exactly `total` entries, or the record is refused naming the rung."""
    lengths = {
        "window_steps_changed": len(block["window_steps_changed"]),
        **{f"{metric}.window_delta_mean": len(block[metric]["window_mean"]) for metric in CHANNELS},
    }
    if block.get("embedding") is not None:
        lengths["embedding.window_distance"] = len(block["embedding"]["window_distance"])
    wrong = {field: n for field, n in lengths.items() if n != total}
    if wrong:
        raise ValueError(
            f"rung {name!r}: per-window series {wrong} do not span the {total} windows; "
            "a series persisted over the changed windows only cannot be clustered"
        )
    for action, held in block.get("held", {}).items():
        _require_window_series(held, f"{name} held {action}", total)


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
    # `--rungs`, not `--arms`: the study's arms are the three representations
    # and that flag is taken. Defaults to the WHOLE ladder -- defaulting to the
    # shuffled rung alone would leave the two rungs that close the multiset
    # blind spot off unless a flag was typed.
    parser.add_argument("--rungs", nargs="+", default=list(LADDER), choices=list(LADDER))
    # `--permutation-seed` is the name the nine shipped records were produced
    # under, and it seeds the shuffled rung identically -- see
    # `action_intervention_ladder` -- so it stays accepted as an alias.
    parser.add_argument(
        "--intervention-seed", "--permutation-seed", dest="intervention_seed",
        type=int, default=0,
    )
    # The two held actions the constant rung is decided on, as indices into
    # the action set: MOVE_FORWARD minus NOOP by default. See `CONTRAST`.
    parser.add_argument("--contrast", nargs=2, type=int, default=list(CONTRAST))
    args = parser.parse_args(argv)
    args.ks = tuple(args.ks)
    args.contrast = tuple(args.contrast)
    # Put back into `LADDER`'s order by the one authority on it, so the table
    # prints the ladder as a ladder whatever order the flag was typed in.
    args.rungs = ladder_order(args.rungs)
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
    if args.contrast[0] == args.contrast[1]:
        # The same reasoning: a contrast of an action with itself is a bad
        # flag, refused here rather than as the library's ValueError twenty
        # seconds into the probe refit. Support membership is NOT checked
        # here -- it is a property of the data, only known once the split is
        # read -- so that refusal stays the library's.
        parser.error(
            f"--contrast {list(args.contrast)} names the same held action twice; a "
            "contrast of an action with itself compares a sequence to itself"
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
            "intervened and canonical arms cannot be held on one sampling stream "
            "and any delta measured here would be sampling noise. Re-run with "
            "--device mps or --device cpu."
        )
        return EXIT_UNSUPPORTED_DEVICE

    # The family the verdict is corrected over is fixed BEFORE any cell runs:
    # every cell with a checkpoint and a record is a comparison whether or
    # not it turns out interpretable, and a family counted from the cells
    # that happened to finish would shrink the threshold on a partial run.
    planned = [
        (arm, seed)
        for arm in args.arms for seed in args.seeds
        if checkpoint_path(args.out, arm, seed).exists()
        and job_record_path(args.out, StudyJob(arm=arm, seed=seed)).exists()
    ]
    family = len(args.rungs) * len(planned) * len(CHANNELS)

    cells: dict[tuple[str, int], dict] = {}
    for arm, seed in planned:
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
            build_record(cell, args, device, family=family),
        )

    if not cells:
        print(
            f"no cell in {args.out} has both a checkpoint and a record; nothing "
            "to diagnose"
        )
        return EXIT_NO_CHECKPOINTS

    print(
        "\n--- self-checks (the three numeric columns must read exactly 0; "
        "noise_restored True and noise_same 0) ---"
    )
    print(selfcheck_table(cells, args.arms, args.seeds, min(args.ks)))
    print("\n--- action-intervention ladder (position and angle, horizon-mean delta) ---")
    print(ladder_table(cells, args.arms, args.seeds, args.rungs))
    held = held_table(cells, args.arms, args.seeds, args.rungs)
    if held:
        print("\n--- held actions of the constant rung, each against the real sequence ---")
        print(held)
    print("\n--- the embedding-space ruler's own shape (window-mean two-draw distance per step) ---")
    print(noise_table(cells, args.arms, args.seeds))
    print("\n--- k-step re-grounding sweep (position, final horizon step) ---")
    print(sweep_table(cells, args.arms, args.seeds, args.ks))
    print("\n--- what a difference between two k columns has to clear (paired) ---")
    print(sweep_ruler_table(cells, args.arms, args.seeds, args.ks))
    print("\n--- verdicts ---")
    for cell in cells.values():
        print(verdict_block(cell["arm"], cell["seed"], cell, family=family))
    print("\n--- across cells, per rung and channel ---")
    print(cross_cell_table(cells, args.arms, args.seeds, args.rungs, family=family))

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
                f"ladder's real arm differs from evaluate_rollout by "
                f"{cell['stream']:.3e}, so the rungs did not share a sampling "
                "stream with it and every delta above is noise."
            )
            return EXIT_STREAM_DIVERGED
    # The same defect class, two more ways to have it: the noise reference's
    # draw did not leave the stream where it found it (every later window's
    # rungs moved with it), or the reference was the canonical imagination
    # bitwise in some window (it drew the canonical's uniforms, and the ratio
    # there is x / 0).
    for cell in cells.values():
        if not cell["noise_restored"]:
            print(
                f"\nSTREAM DIVERGED for {cell['arm']} seed {cell['seed']}: the noise "
                "reference's draw did not restore the sampling stream on every "
                "window (both generators), so every window after the first started "
                "somewhere evaluate_rollout never started it and every delta above "
                "is noise."
            )
            return EXIT_STREAM_DIVERGED
        if cell["noise_collapsed"] != 0:
            print(
                f"\nSTREAM DIVERGED for {cell['arm']} seed {cell['seed']}: the noise "
                f"reference was the canonical imagination bitwise on "
                f"{cell['noise_collapsed']} of {cell['windows_total']} windows, so it "
                "drew the canonical pass's own uniforms and the embedding ratios "
                "above are read against a ruler that measured nothing."
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
