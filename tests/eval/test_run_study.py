"""The driver over the nine cells: which ones are pending, in what order.

This script decides whether a ~1.5-hour cell gets paid for a second time, and
it is the only thing standing between a ~13.5-hour unattended run and a morning
with nothing to show. Two failure modes are worth more than everything else
here and every test below serves one of them:

  * a job that IS done is re-run       -- at worst a lost day;
  * a job that is NOT done is skipped  -- a record that is truncated, stale, or
    describes another cell is read by the aggregation as this cell's result,
    and the study reports a number nobody measured.

The second is the one with no symptom, so "complete" is defined by contents
rather than by the file existing.
"""

import importlib.util
import inspect
import json
import os
import socket
import subprocess
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from mbfps.eval import study
from mbfps.eval.study import (
    NONFINITE_KEY,
    SPLIT_SEED,
    StudyJob,
    job_record_path,
    to_json_record,
    write_record,
)
from mbfps.utils.config import ARMS, Config

_SPEC = importlib.util.spec_from_file_location(
    "run_study", Path(__file__).resolve().parents[2] / "scripts" / "run_study.py")
run_study = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_study)

SEEDS = (0, 1, 2)


# ---------------------------------------------------------------------------
# THE FIXTURE'S PARAMETERS, PAIRWISE DISTINCT BY CONSTRUCTION.
#
# Task 4 shipped roughly fifty assertions that could not fail, and they were
# not fifty mistakes -- they were one, made in the fixture. `steps == horizon`,
# `job.seed == SPLIT_SEED`, `job.seed == context`: wherever two quantities in
# the fixture were the same number, every assertion telling one from the other
# read the same value on both sides, and exchanging them in the source was a
# numerical no-op.
#
# The values below are therefore chosen once, together, so that no two that a
# plausible mutation could exchange are equal, and
# `test_the_fixture_parameters_are_pairwise_distinct_so_no_assertion_is_vacuous`
# asserts that property itself.
# ---------------------------------------------------------------------------

# What the command line asks for...
CLI_STEPS = 11
CLI_SEQ_LEN = 6
CLI_SEED = 17             # deliberately OUTSIDE SEEDS: this is the seed a
                          # MISLABELLED record names, and a wrong seed that
                          # happened to be one of the study's own could not be
                          # told from a real cell's record
CLI_DEVICE = "device-from-the-command-line"
"""Not a real device name, and that is the point.

The study runs on a rented CUDA box that nobody is watching, and this Mac has
no CUDA, so an assertion like `device == "cpu"` would pass here for the wrong
reason and a `device="mps"` hardcoded in the driver would only bite on the box.
A string no implementation would ever hardcode cannot be matched by accident.
"""

# ...what the parser falls back to when it asks for nothing...
DEFAULT_STEPS = 20_000
DEFAULT_SEQ_LEN = 64
DEFAULT_DEVICE = "cuda"
"""The parser's default device is the RENTED BOX, not this laptop.

`run_job`'s own default is "mps". If the driver stopped forwarding `device`,
every cell of a CUDA run would silently fall back through `get_device` to the
CPU and the study would take weeks. Nothing on this machine would notice.
"""

# ...what a synthetic record on disk claims it was run with...
RECORD_STEPS = 13
RECORD_SEQ_LEN = 9
RECORD_CONTEXT = 8
RECORD_HORIZON = 41

# ...and what the real `run_job` calls in this file are given.
RUN_STEPS = 7
RUN_SEQ_LEN = 4
RUN_CONTEXT = 2
RUN_HORIZON = 3
RUN_KW = dict(steps=RUN_STEPS, seq_len=RUN_SEQ_LEN, context=RUN_CONTEXT,
              horizon=RUN_HORIZON, device="cpu")

# ...what the EVALUATION PROTOCOL is, read off its one home.
#
# `main` has no --context/--horizon flags on purpose, so the configuration the
# driver demands of a finished record is the command line's steps/seq_len plus
# `run_job`'s own defaults. Reading them here from `run_job` rather than
# restating them is what makes a literal smuggled into `run_study` show up:
# the day `run_job`'s defaults move, a hardcoded copy in the driver goes red.
_RUN_JOB_PARAMETERS = inspect.signature(study.run_job).parameters
PROTOCOL_CONTEXT = _RUN_JOB_PARAMETERS["context"].default
PROTOCOL_HORIZON = _RUN_JOB_PARAMETERS["horizon"].default

# ...and the wrong value each config field takes in the mismatch tests. One per
# key, so each is exercised ALONE: with all four wrong at once, a CONFIG_KEYS
# short of any one of them would still refuse and the deletion would survive.
SMOKE_STEPS = 30
"""Stands for the plan's own Task 7 Step 1 smoke run.

Not literally 3, only because `RUN_HORIZON` is 3 and every number in this
module has to be distinct from every other one -- see the invariant below.
"""
CONFIG_MISMATCHES = {
    "steps": SMOKE_STEPS,
    "seq_len": 71,
    "context": 23,
    "horizon": 97,
}

# ...the wall-clock seconds the driver's own timer reports for a cell.
#
# The spy returns in microseconds, so `time.perf_counter()` really does read
# ~0.0 either side of a call and `wall={_fmt(elapsed, '.1f')}s` renders "0.0" --
# which is also what the mutation `time.perf_counter() - started -> 0.0`
# renders. The clock is therefore driven, not measured; see `_FakeClock`.
SUMMARY_WALL_SECONDS = 42.5

# ...and the value of every field `job_summary` renders, all different from one
# another, so that a mutation exchanging any two of them changes the text.
#
# `job_summary` formats roughly twenty-five fields and exactly four of them
# were pinned. Every number an operator uses at 8am to decide whether a cell is
# broken could be swapped with its neighbour, replaced by a literal, or dropped
# outright with a green suite. The whole block is asserted character for
# character below, and that assertion is only worth anything if no two of these
# are equal -- `band_median` showing `band_min` is invisible if they agree.
SUMMARY_VALUES = {
    "steps": 5101, "seq_len": 5102, "context": 5103, "horizon": 5104,
    "split_seed": 5105,
    "seconds": 5106.0, "steps_per_second": 5107.0,
    "kl_rate_above_free_bits": 5108.0, "kl_dyn_max": 5109.0,
    "loss_last20": 5110.0,
    "position.gap_final": 5111.0, "position.gap_mean": 5112.0,
    "position.gap_finite": 5113, "position.n_steps": 5114,
    "position.band_median": 5115.0, "position.band_min": 5116.0,
    "position.steps_degenerate": 5117,
    "angle.gap_final": 5118.0, "angle.gap_mean": 5119.0,
    "angle.gap_finite": 5120, "angle.n_steps": 5121,
    "angle.band_median": 5122.0, "angle.band_min": 5123.0,
    "angle.steps_degenerate": 5124,
    "filtering.latent_r2": 5125.0, "filtering.embedding_r2": 5126.0,
    "filtering.gain": 5127.0, "filtering.ci_low": 5128.0,
    "filtering.ci_high": 5129.0,
    "reward.mse": 5130.0, "reward.baseline_mse": 5131.0, "reward.r2": 5132.0,
    "encoder_params": 5133,
    "wall_seconds": SUMMARY_WALL_SECONDS,
}
"""One distinct value per rendered field. Booleans are deliberately absent.

`False == 0` and `True == 1` in Python, so a bool in a distinctness table
either collides with a real number or forces one out of the table.

THE TWO BOOLEANS THEREFORE NEED A GUARD OF THEIR OWN, AND ONE RENDERING IS NOT
IT. This docstring used to claim they were "pinned by the character-for-
character assertion, and chosen to differ from the values `_record_body` uses
so that a hardcoded literal in their place still fails". That was wrong twice
over, and the claim is what stopped anyone adding the real guard.

Differing from `_record_body` is beside the point: a mutation replacing
`_get(record, "filtering", "criterion_4", "latent_beats_embedding")` with a
literal writes `True` or `False`, not whatever the test fixture happens to
hold elsewhere. And a bool has only TWO values, so any single rendering
excludes only ONE of the two literals -- the fixture said `True` here, the
mutation said `True` there, and `EXPECTED_SUMMARY` could not tell them apart.
Both booleans were free to be constants in all nine cells: gate criterion 4
reported as passing whatever the record said, and a degenerate reward target
reported as healthy.

So the two are pinned twice, and each pinning kills the literal the other
cannot: `_summary_record` now holds the OPPOSITE of the literal a mutation
would use (`False` for the filtering flag, `True` for the reward flag), which
`EXPECTED_SUMMARY` checks character for character, and
`test_job_summary_reads_its_two_booleans_out_of_the_record_both_ways` renders
the block again with both flipped.
"""

#: The two booleans `job_summary` prints, as `_summary_record` writes them.
#
# Each is the NEGATION of the literal its own mutation substitutes -- the
# filtering line's mutation is `-> True`, the reward line's is `-> False` --
# so `EXPECTED_SUMMARY` fails on both. Kept as named constants rather than
# spelled inline so `EXPECTED_SUMMARY` and the fixture cannot drift apart into
# an agreement neither of them meant.
SUMMARY_BOOLEANS = {
    "filtering.latent_beats_embedding": False,
    "reward.is_degenerate": True,
}

#: The two STRINGS `job_summary` prints on its provenance line.
#
# Out of the distinctness table for the same reason the booleans are: they are
# not numbers. The sha's first eight characters differ from its remaining
# thirty-two on purpose, so the block can tell `git_sha[:8]` from the whole
# sha, from `[:7]` and from `[:9]`; and both differ from `_record_body`'s own
# "0123456789abcdef..." / "device-from-the-record" and from `CLI_DEVICE`, so a
# line that read either field from anywhere but this record would show it.
SUMMARY_GIT_SHA = "f00d5134" + "e" * 32
SUMMARY_DEVICE = "device-from-the-summary-record"

#: Every number above that a mutation could exchange for another. All distinct.
PAIRWISE_DISTINCT_PARAMETERS = {
    "cli.steps": CLI_STEPS,
    "cli.seq_len": CLI_SEQ_LEN,
    "cli.seed": CLI_SEED,
    "default.steps": DEFAULT_STEPS,
    "default.seq_len": DEFAULT_SEQ_LEN,
    "record.steps": RECORD_STEPS,
    "record.seq_len": RECORD_SEQ_LEN,
    "record.context": RECORD_CONTEXT,
    "record.horizon": RECORD_HORIZON,
    "run.steps": RUN_STEPS,
    "run.seq_len": RUN_SEQ_LEN,
    "run.context": RUN_CONTEXT,
    "run.horizon": RUN_HORIZON,
    "protocol.context": PROTOCOL_CONTEXT,
    "protocol.horizon": PROTOCOL_HORIZON,
    # The study's own split seed. `_record_body` writes it into every record's
    # `split_seed` field, and `job_summary` prints that field on the same line
    # as the recorded steps/seq_len. It is 0, and so is the seed of a third of
    # the study's cells: a summary test using a job with seed 0 cannot tell
    # `split_seed={record}` from `split_seed={job.seed}`, which is the L1
    # species inside the one guard job_summary had.
    "study.split_seed": SPLIT_SEED,
    **{f"mismatch.{key}": value for key, value in CONFIG_MISMATCHES.items()},
    **{f"summary.{key}": value for key, value in SUMMARY_VALUES.items()},
}

# The three scenario cells: one per arm AND one per seed, so that no test can
# pass by hardcoding either field, and so that all three arms are exercised.
# `frozen_ssl` is the arm Task 4's tests never ran and the only one whose
# backbone is not its own name; leaving it out of a fixture is how three seeds
# of the study died hours in.
DONE_JOB = StudyJob("pixel_ae", 1)
CORRUPT_JOB = StudyJob("frozen_ssl", 2)
INCOMPLETE_JOB = StudyJob("random_vit", 0)

# `len(ARMS) == len(SEEDS) == 3`, so a 3x3 fixture cannot tell the arms
# argument from the seeds argument: swapping them still yields nine jobs. This
# pair is uneven on purpose and is what makes that swap visible.
UNEVEN_ARMS = ("frozen_ssl", "random_vit")
UNEVEN_SEEDS = (0, 1, 2, CLI_SEED)

# ---------------------------------------------------------------------------
# THE COMPLETENESS SET, WRITTEN OUT INDEPENDENTLY.
#
# `REQUIRED_RECORD_KEYS` used to be guarded by two tests that both shrank with
# it: one parametrised OVER the set, so deleting a key deleted its own test
# case, and one asserting the set was a SUBSET of a real record's keys, which a
# smaller set satisfies just as well. Every one of "position", "curves",
# "filtering", "reward" and "probe" could be removed with a green suite, and a
# pixel_ae record with no position block would then have been marked done: the
# ~1.5-hour cell never re-runs and the gate is evaluated on a record carrying no
# position metric.
#
# So this literal is typed out by hand and compared for EQUALITY. Equality is
# what fails in both directions -- a key deleted AND a key added -- and being
# an independent copy is what stops it moving when the thing it guards moves.
# ---------------------------------------------------------------------------

EXPECTED_RECORD_KEYS = frozenset({
    "arm", "seed", "steps", "seq_len", "context", "horizon", "split_seed",
    "seconds", "steps_per_second", "kl_rate_above_free_bits", "episodes",
    "probe", "position", "angle", "filtering", "reward", "curves",
    "git_sha", "device", "encoder_params", "history",
    "nonfinite",
})
"""Spelled out, `NONFINITE_KEY` included, so a RENAME is caught as well."""

M3C_RECORD_KEYS = frozenset({"git_sha", "device", "encoder_params", "history"})
"""The four provenance keys the M3c re-run added, as their own literal.

REQUIRED, NOT OPTIONAL. `runs/m3_study_v2` is a fresh directory and the M3b
cells are not reused (spec 2.4), so no record the driver will ever resume
off legitimately lacks them -- and requiring them is what makes "one code
state produced all nine cells" a field every finished cell must carry rather
than one some of them happen to. A second literal beside
`EXPECTED_RECORD_KEYS` rather than a slice of it, so that the four can be
named in a parametrisation without deriving the cases from the collection
under test; `test_the_m3c_keys_are_in_both_literals` ties the two together.
"""

OPTIONAL_RECORD_KEYS = frozenset({"kl_dyn_max", "loss_last20"})
"""The top-level keys `run_job` writes that a record is allowed to lack.

`run_job` writes twenty-four top-level keys; twenty-two of them are required.
These two are training diagnostics printed in the log, and a record without
them is still a finished cell. Naming them here is what lets the drift guard
below be an EQUALITY against a real record instead of a subset test.
"""

# ---------------------------------------------------------------------------
# THE EXIT STATUSES, ALSO WRITTEN OUT INDEPENDENTLY.
#
# Nobody is watching the ~13.5-hour run, so the process status is what a wrapper,
# an `&&` or a CI step actually reads. `EXIT_NO_DATA` used to be 2 -- which is
# argparse's own usage status, and not ours to reuse -- so `--data` misspelt
# and a box that mounted no episodes were indistinguishable to everything
# downstream, and those two want opposite responses.
#
# Hand-written literals compared for EQUALITY, like `EXPECTED_RECORD_KEYS`
# above and for the same reason: a guard reading the driver's own constants
# moves when they move, and renumbering one onto another would take its own
# test case with it.
# ---------------------------------------------------------------------------

ARGPARSE_USAGE_STATUS = 2
"""What `parser.error` exits with. Not ours to choose, and not ours to take."""

EXPECTED_EXIT_STATUS = {
    "EXIT_OK": 0,
    "EXIT_JOB_FAILED": 1,
    "EXIT_CONFIG_MISMATCH": 3,
    "EXIT_LOCKED": 4,
    "EXIT_NO_DATA": 5,
    "EXIT_OUT_UNUSABLE": 6,
    "EXIT_FOREIGN_RECORDS": 23,
}
"""Every status `main` can return, spelled out. 2 is deliberately absent.

`EXIT_OUT_UNUSABLE` was added for the same reason `EXIT_NO_DATA` stopped
being 2: `--out` naming an existing file, a read-only mount or a full disk
came out of `main` as an uncaught traceback, and an uncaught traceback exits 1
-- which IS `EXIT_JOB_FAILED`. The wrapper reading the overnight run's status
was told "some cells failed, re-run to retry exactly those" when nothing had
run.

`EXIT_FOREIGN_RECORDS` is the newest, and 23 rather than 7 because the four M3
tools share one numbering (7-22 are the report's, the diagnostics' and the
pooling script's; `tests/eval/test_pool_dynamics_script.py` holds the
cross-script literal). It is the status for a `--out` that another code state
wrote -- M3b's `runs/m3_study`, whose nine records became "pending" the day
the four provenance keys were required, so that a driver pointed there would
have re-run over the only copy of six cells.
"""


def _record_body(job: StudyJob, **overrides) -> dict:
    """A complete record for `job`, shaped like `run_job`'s but cheap.

    `NONFINITE_KEY` is absent: `write_record` adds it, and `to_json_record`
    raises if it is already there.
    """
    metric = {
        "final_model": 30.0, "final_persistence": 40.0, "final_floor": 10.0,
        "band_min": 5.0, "band_median": 12.5, "band_max": 30.0,
        "relative_min": 0.1, "relative_median": 0.3, "relative_max": 0.7,
        "steps_floor_above_persistence": 0, "steps_degenerate": 0,
        "gap_finite": 45, "gap_mean": 0.25, "gap_min": -0.1, "gap_max": 0.6,
        "gap_final": 0.33, "n_steps": 45,
    }
    body = {
        "arm": job.arm,
        "seed": job.seed,
        "steps": RECORD_STEPS,
        "seq_len": RECORD_SEQ_LEN,
        "context": RECORD_CONTEXT,
        "horizon": RECORD_HORIZON,
        "split_seed": SPLIT_SEED,
        "seconds": 5044.0,
        "steps_per_second": 3.98,
        "kl_rate_above_free_bits": 0.412,
        "kl_dyn_max": 1.25,
        "loss_last20": 0.75,
        # The M3c provenance fields. Synthetic values no implementation would
        # produce, in the spirit of CLI_DEVICE: nothing here is compared
        # against a real `git rev-parse` or a real parameter count (that is
        # `tests/eval/test_study.py`'s job); the driver only needs them PRESENT.
        "git_sha": "0123456789abcdef0123456789abcdef01234567",
        "device": "device-from-the-record",
        "encoder_params": 4242,
        "history": {
            "loss": [0.9, 0.8],
            "parts": [
                {"embedding": 0.5, "reward": 0.1, "continue": 0.2,
                 "kl_dyn": 0.05, "kl_rep": 0.05},
                {"embedding": 0.4, "reward": 0.1, "continue": 0.2,
                 "kl_dyn": 0.05, "kl_rep": 0.05},
            ],
        },
        "episodes": {"train": ["ep_a.npz", "ep_b.npz"], "val": ["ep_c.npz"]},
        "probe": {
            "latent_ridge": 1.0, "latent_ridge_selected": True,
            "latent_selection_r2": 0.2, "embedding_ridge": 10.0,
            "embedding_ridge_selected": True, "embedding_selection_r2": 0.3,
        },
        "position": dict(metric),
        "angle": dict(metric),
        "filtering": {
            "criterion_4": {
                "latent_r2": 0.31, "embedding_r2": 0.33,
                "latent_beats_embedding": False,
            },
            "gain": {
                "gain": -0.0208, "ci_low": -0.05, "ci_high": 0.01,
                "confidence": 0.95, "joint_r2": 0.31, "embedding_r2": 0.33,
                "n_scored_windows": 120, "ridge_selected": True,
                "joint_ridge": 1.0, "embedding_ridge": 10.0,
            },
        },
        "reward": {
            "mse": 0.001, "baseline_mse": 0.002, "r2": 0.5, "n_steps": 900,
            "n_reward_events": 6, "is_degenerate": True,
        },
        "curves": {name: [1.0, 2.0] for name in (
            "rssm_position", "persistence_position", "floor_position",
            "rssm_angle", "persistence_angle", "floor_angle")},
    }
    body.update(overrides)
    return body


def _write_complete(out_dir, job: StudyJob, **overrides) -> Path:
    """Put a complete, on-disk record for `job` under `out_dir`."""
    path = job_record_path(out_dir, job)
    write_record(path, _record_body(job, **overrides))
    return path


#: The flags whose configuration `_write_matching` writes into a record.
MATCHING_ARGV = ["--steps", str(CLI_STEPS), "--seq-len", str(CLI_SEQ_LEN),
                 "--allow-short"]
"""...and `--allow-short`, because `CLI_STEPS`/`CLI_SEQ_LEN` really are short.

The fixture's numbers are deliberately tiny and deliberately unlike anything
the study asks for, which is what makes "the log echoes the flags we passed"
detectable -- and it is also exactly what `MIN_STEPS`/`MIN_SEQ_LEN` refuse. A
deliberate short run says so on the command line; that is the whole
distinction the floors draw, so the fixture says so too rather than the floors
being lowered until the fixture slips under them.
"""


def _write_matching(out_dir, job: StudyJob, **overrides) -> Path:
    """A complete record at the configuration `MATCHING_ARGV` will request.

    `_record_body`'s own RECORD_* numbers are deliberately unlike anything the
    command line asks for -- that is what makes "the log echoes the flags we
    passed" detectable -- so a record written with them is, correctly, a record
    from another configuration. A test that wants `main` to SKIP a finished
    cell has to write one the driver will accept as current.
    """
    fields = {"steps": CLI_STEPS, "seq_len": CLI_SEQ_LEN,
              "context": PROTOCOL_CONTEXT, "horizon": PROTOCOL_HORIZON}
    return _write_complete(out_dir, job, **{**fields, **overrides})


def _sanitised(job: StudyJob, **overrides) -> dict:
    """Exactly what the file would hold, as a dict we can then damage."""
    return to_json_record(_record_body(job, **overrides))


#: The cell `SUMMARY_RECORD` was asked for...
SUMMARY_JOB = StudyJob("frozen_ssl", 1)
#: ...and the cell it actually names. DELIBERATELY A DIFFERENT ONE.
#
# The `cell` line is built as `arm=... seed=... (requested .../s...)`, i.e.
# built to show the two disagreeing, and the mutation that reads both halves
# off the job is invisible whenever they agree. `SUMMARY_JOB.seed` is also
# neither `SPLIT_SEED` nor `SUMMARY_RECORD_JOB.seed`, so `split_seed` on the
# next line can be told from both.
SUMMARY_RECORD_JOB = StudyJob("pixel_ae", 2)


def _summary_record() -> dict:
    """A record with a DISTINCT value in every field `job_summary` renders.

    `_record_body`'s own values repeat -- `position` and `angle` are literally
    the same dict -- so a mutation printing one metric block's number under the
    other's label reads identically. Here every leaf is its own number, drawn
    from `SUMMARY_VALUES`, which the pairwise-distinctness invariant guards.
    """
    value = SUMMARY_VALUES

    def metric(prefix: str) -> dict:
        block = dict(_record_body(SUMMARY_RECORD_JOB)["position"])
        block.update({
            "gap_final": value[f"{prefix}.gap_final"],
            "gap_mean": value[f"{prefix}.gap_mean"],
            "gap_finite": value[f"{prefix}.gap_finite"],
            "n_steps": value[f"{prefix}.n_steps"],
            "band_median": value[f"{prefix}.band_median"],
            "band_min": value[f"{prefix}.band_min"],
            "steps_degenerate": value[f"{prefix}.steps_degenerate"],
        })
        return block

    return _record_body(
        SUMMARY_RECORD_JOB,
        steps=value["steps"], seq_len=value["seq_len"],
        context=value["context"], horizon=value["horizon"],
        split_seed=value["split_seed"],
        seconds=value["seconds"],
        steps_per_second=value["steps_per_second"],
        kl_rate_above_free_bits=value["kl_rate_above_free_bits"],
        kl_dyn_max=value["kl_dyn_max"], loss_last20=value["loss_last20"],
        git_sha=SUMMARY_GIT_SHA, device=SUMMARY_DEVICE,
        encoder_params=value["encoder_params"],
        # Two train episodes and one val: the counts differ, so the mutation
        # that reports each under the other's label changes the text.
        episodes={"train": ["ep_a.npz", "ep_b.npz"], "val": ["ep_c.npz"]},
        position=metric("position"), angle=metric("angle"),
        filtering={
            "criterion_4": {
                "latent_r2": value["filtering.latent_r2"],
                "embedding_r2": value["filtering.embedding_r2"],
                # The OPPOSITE of the literal the mutation writes -- see
                # SUMMARY_BOOLEANS. The other direction is covered by
                # `test_job_summary_reads_its_two_booleans_out_of_the_record...`
                "latent_beats_embedding":
                    SUMMARY_BOOLEANS["filtering.latent_beats_embedding"],
            },
            "gain": {
                "gain": value["filtering.gain"],
                "ci_low": value["filtering.ci_low"],
                "ci_high": value["filtering.ci_high"],
                "confidence": 0.95, "joint_r2": 0.31, "embedding_r2": 0.33,
                "n_scored_windows": 120, "ridge_selected": True,
                "joint_ridge": 1.0, "embedding_ridge": 10.0,
            },
        },
        reward={
            "mse": value["reward.mse"],
            "baseline_mse": value["reward.baseline_mse"],
            "r2": value["reward.r2"],
            "n_steps": 900, "n_reward_events": 6,
            # Likewise the opposite of its own mutation's literal.
            "is_degenerate": SUMMARY_BOOLEANS["reward.is_degenerate"],
        },
    )


#: `job_summary(SUMMARY_JOB, _summary_record(), SUMMARY_WALL_SECONDS)`, by hand.
#
# WRITTEN OUT RATHER THAN COMPUTED, for the reason `EXPECTED_RECORD_KEYS` is:
# a guard that formats the record the way the function does moves whenever the
# function moves, and every field swap the audit found survives it. Comparing
# the whole block against a literal is what fails on a field read from the
# wrong place, a field replaced by a constant, a field dropped, a label
# detached from its numbers, and `"\n".join` becoming `" ".join`.
EXPECTED_SUMMARY = "\n".join([
    "  cell      arm=pixel_ae seed=2 (requested frozen_ssl/s1)",
    "  recorded  steps=5101 seq_len=5102 context=5103 horizon=5104 "
    "split_seed=5105",
    "  code      git_sha=f00d5134 device=device-from-the-summary-record "
    "encoder_params=5133",
    "  timing    wall=42.5s recorded=5106.0s steps_per_second=5107.00",
    "  training  kl_rate_above_free_bits=5108.000 kl_dyn_max=5109.000 "
    "loss_last20=5110.0000",
    "  episodes  train=2 val=1",
    "  position  gap_final=+5111.0000 gap_mean=+5112.0000 finite=5113/5114 "
    "band_median=5115.000 degenerate=5117",
    "  angle     gap_final=+5118.0000 gap_mean=+5119.0000 finite=5120/5121 "
    "band_median=5122.000 degenerate=5124",
    "  filtering criterion_4_latent_beats_embedding=False gain=+5127.0000 "
    "CI[+5128.0000, +5129.0000]",
    "  reward    mse=5130.00000 baseline=5131.00000 r2=+5132.0000 "
    "degenerate_target=True",
])


class _FakeClock:
    """A `time` stand-in whose `perf_counter` advances by a known amount.

    The spy returns in microseconds, so the driver's real elapsed time is
    ~0.0 and `wall=0.0s` is what both the correct code and the mutation
    `time.perf_counter() - started -> 0.0` print. Driving the clock is what
    makes the difference visible: every cell's block must read `wall=42.5s`.
    """

    def __init__(self, step: float = SUMMARY_WALL_SECONDS):
        self.step = step
        self.now = 0.0

    def perf_counter(self) -> float:
        self.now += self.step
        return self.now


@pytest.fixture
def episode_dir(tmp_path) -> Path:
    """A data directory `ReplayBuffer` reports at least one episode in.

    The contents never matter here: every test that uses it also replaces
    `run_job`, so nothing reads an episode. Only the driver's "is there data at
    all" check looks at this.
    """
    root = tmp_path / "data"
    root.mkdir()
    (root / "ep_000000_len00040.npz").write_bytes(b"")
    return root


def _recorded_config(kwargs: dict) -> dict:
    """The config fields `run_job` would have written, given these arguments.

    THE SPY HAS TO BE HONEST ABOUT THIS. `run_job` records what it was actually
    trained with, and the driver's staleness check compares that against what
    the command line asked for. A stand-in that always wrote the fixture's own
    RECORD_* numbers would make every cell it ran look stale on the next
    resume, and the check's "a matching re-run still skips" half would be
    untestable. Binding against the real signature is what keeps the two in
    step, defaults included.
    """
    bound = inspect.signature(study.run_job).bind_partial(**kwargs)
    bound.apply_defaults()
    return {key: bound.arguments[key] for key in run_study.CONFIG_KEYS}


def _spy(behaviour=None):
    """A stand-in for `run_job` that records its calls.

    `behaviour(job)` may raise (a failing cell) or return a dict of record
    overrides. Like the real thing it writes the record and returns the LIVE
    dict, so the resume path and the printing path are both exercised, and it
    records the configuration it was called with.

    EVERY KEY RECORDED HERE IS READ BY AN ASSERTION SOMEWHERE, and that is a
    rule rather than an accident. `"buffer"` was captured on every call from
    the day this spy was written and no assertion ever looked at it, so
    `run_job(job, None, ...)` -- which on the rented box makes all nine cells
    raise inside `run_job`, be swallowed by the per-cell handler, and end the
    session 0/9 with nine identical tracebacks -- survived the entire suite. A
    value a spy records but nothing checks is not a guard; it is the shape of
    one. The readers are:

      "job"      -- the order and identity tests;
      "buffer"   -- `test_main_hands_every_cell_the_one_buffer_it_built`;
      "out_dir"  -- `test_main_hands_run_job_the_out_dir_string_argparse_produced`;
      "kwargs"   -- the flag-forwarding and defaults tests;
      "record"   -- `test_main_prints_the_per_cell_summary_block`.
    """
    calls: list[dict] = []

    def fake(job, buffer, out_dir, **kwargs):
        calls.append({"job": job, "buffer": buffer, "out_dir": out_dir,
                      "kwargs": kwargs})
        overrides = behaviour(job) if behaviour is not None else None
        body = _record_body(
            job, **{**_recorded_config(kwargs), **(overrides or {})})
        write_record(job_record_path(out_dir, job), body)
        calls[-1]["record"] = body
        return body

    return fake, calls


# ---------------------------------------------------------------------------
# the fixture itself: the guard on the SPECIES, not the instance
# ---------------------------------------------------------------------------

def test_the_fixture_parameters_are_pairwise_distinct_so_no_assertion_is_vacuous():
    """Every "field X carries quantity Y" assertion in this module can only
    fail if Y differs from the other quantities X might have been filled from.
    That is a property of the fixture, and Task 4 lost six review rounds to it.
    If two of these are ever made equal again, THIS fails by name instead of
    quietly hollowing out the assertions downstream.
    """
    values = list(PAIRWISE_DISTINCT_PARAMETERS.values())
    collisions = {
        value: sorted(name for name, other in PAIRWISE_DISTINCT_PARAMETERS.items()
                      if other == value)
        for value in set(values) if values.count(value) > 1
    }
    assert not collisions, (
        "these fixture parameters collide, so every assertion that tells one "
        "from the other is now a tautology and a mutation exchanging them "
        f"survives the suite: {collisions}")
    assert len(set(values)) == len(PAIRWISE_DISTINCT_PARAMETERS) == 54, (
        "an entry was dropped from the table rather than made distinct; the "
        "count is spelled out so that deleting a colliding parameter cannot "
        "be mistaken for fixing it")
    assert len(SUMMARY_VALUES) == 34, (
        "every field `job_summary` renders needs a distinct value here, or "
        "the character-for-character assertion on the block cannot tell that "
        "field from the one beside it")

    # The two booleans are out of the table above on purpose (`False == 0`),
    # so their distinctness is asserted here instead. They must disagree with
    # EACH OTHER, or the character-for-character block cannot tell a mutation
    # that renders one under the other's label from the correct code; and each
    # must be a real `bool`, since `0`/`1` would render as digits and quietly
    # change what `EXPECTED_SUMMARY` is checking.
    assert set(SUMMARY_BOOLEANS) == {"filtering.latent_beats_embedding",
                                     "reward.is_degenerate"}
    assert all(value is True or value is False
               for value in SUMMARY_BOOLEANS.values())
    assert (SUMMARY_BOOLEANS["filtering.latent_beats_embedding"]
            is not SUMMARY_BOOLEANS["reward.is_degenerate"]), (
        "the two booleans `job_summary` prints agree in the fixture, so a "
        "mutation printing either under the other's label is invisible")

    # The two provenance strings: the sha's head must differ from its tail
    # (or `[:8]` cannot be told from the whole sha), and neither string may
    # be one that any other fixture or the command line already uses.
    assert len(SUMMARY_GIT_SHA) == 40
    assert SUMMARY_GIT_SHA[:8] not in SUMMARY_GIT_SHA[8:]
    assert SUMMARY_GIT_SHA != _record_body(SUMMARY_RECORD_JOB)["git_sha"]
    assert SUMMARY_DEVICE not in {
        CLI_DEVICE, DEFAULT_DEVICE, "mps", "cpu",
        _record_body(SUMMARY_RECORD_JOB)["device"]}

    # The parser's defaults are two of those thirteen, and they are only
    # distinct-by-construction if they really are the defaults.
    defaults = vars(run_study._parser().parse_args([]))
    assert defaults["steps"] == DEFAULT_STEPS
    assert defaults["seq_len"] == DEFAULT_SEQ_LEN
    assert defaults["device"] == DEFAULT_DEVICE

    # The device is categorical and has the same failure mode as the numbers:
    # the command-line value must differ from the default, from `run_job`'s own
    # default, and from anything this laptop would fall back to.
    assert CLI_DEVICE not in {DEFAULT_DEVICE, "mps", "cpu"}

    # The three scenario cells cover every arm and every seed exactly once, so
    # no test below can pass by hardcoding either field -- and `frozen_ssl`,
    # the arm whose backbone name differs from its own, is among them.
    scenarios = (DONE_JOB, CORRUPT_JOB, INCOMPLETE_JOB)
    assert {j.arm for j in scenarios} == set(ARMS)
    assert {j.seed for j in scenarios} == set(SEEDS)

    # The uneven fixture exists because the natural one is square.
    assert len(ARMS) == len(SEEDS), (
        "the study is no longer 3x3; UNEVEN_ARMS/UNEVEN_SEEDS may need "
        "rechoosing, but the arms/seeds swap must still be detectable")
    assert len(UNEVEN_ARMS) != len(UNEVEN_SEEDS)
    assert len(UNEVEN_ARMS) != len(ARMS) and len(UNEVEN_SEEDS) != len(SEEDS)
    assert CLI_SEED not in SEEDS, (
        "the seed a mislabelled record names must not be one of the study's "
        "own, or a mislabel cannot be told from a real cell's record")

    # The configuration the driver will demand of a finished record is the
    # command line's steps/seq_len plus `run_job`'s own context/horizon. Each
    # of the four has to differ from what `_record_body` writes, or the
    # mismatch tests below cannot tell a stale record from a current one.
    for key, wrong in CONFIG_MISMATCHES.items():
        assert wrong != run_study.requested_config(CLI_STEPS, CLI_SEQ_LEN)[key]
    assert (RECORD_STEPS, RECORD_SEQ_LEN, RECORD_CONTEXT, RECORD_HORIZON) != (
        CLI_STEPS, CLI_SEQ_LEN, PROTOCOL_CONTEXT, PROTOCOL_HORIZON)


def test_the_driver_uses_the_studys_own_arms_and_seeds():
    """A private copy of either tuple would let the driver and the config
    disagree about what the study is."""
    assert run_study.ARMS is ARMS
    assert run_study.SEEDS == SEEDS
    assert run_study._SLOW_ARMS == (), (
        "M3c retired the end-to-end pixel arm; every study arm now trains at "
        "feature-arm speed (~1.5 h per cell), so no arm is deferred. An "
        "arm listed here would be scheduled last for a cost it does not have")
    assert "cnn" not in ARMS
    assert NONFINITE_KEY in run_study.REQUIRED_RECORD_KEYS, (
        "only `write_record` adds this key, so requiring it is what stops a "
        "hand-made JSON blob at a record path from being mistaken for a "
        "finished ~1.5-hour cell")


def test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own():
    """THE WHOLE POINT OF THESE CODES IS AN UNATTENDED RUN.

    `EXIT_NO_DATA` was 2, and so is argparse's usage status: `run_study.py
    --dat runs/...` exited 2 and so did a run against a directory with no
    episodes in it, so a wrapper could not tell "the command line was wrong and
    nothing ran" from "the box has no data". Those want opposite responses --
    fix the flag, versus go and find the episodes -- and the only person who
    could tell them apart by reading the log is the one who is asleep.

    Pinned against a hand-written literal rather than against the driver's own
    constants, and by NAME SET as well as by value: an `EXIT_*` added later
    without a line here is one that is free to land on 2 again, or on top of
    another of the five.
    """
    actual = {name: getattr(run_study, name) for name in EXPECTED_EXIT_STATUS}
    assert actual == EXPECTED_EXIT_STATUS, (
        "an exit status was renumbered; every wrapper, `&&` and CI step that "
        "reads this run's status reads these numbers")
    assert len(set(actual.values())) == len(EXPECTED_EXIT_STATUS) == 7, (
        "two statuses collide, so the run cannot say which thing went wrong")
    assert ARGPARSE_USAGE_STATUS not in actual.values(), (
        "this status belongs to argparse's own usage errors; a study status "
        "sharing it makes a typo and a real failure the same event downstream")

    declared = {name for name in vars(run_study) if name.startswith("EXIT_")}
    assert declared == set(EXPECTED_EXIT_STATUS), (
        "the driver's exit statuses and the ones pinned here have drifted "
        "apart; an unpinned status is free to collide with argparse's 2 or "
        f"with one of the others: {declared ^ set(EXPECTED_EXIT_STATUS)}")


def test_the_parser_defaults_name_the_studys_own_directories():
    """`--out` and `--data` were the two defaults nothing pinned.

    `--steps`, `--seq-len` and `--device` are pinned by
    `test_main_forwards_the_documented_defaults`; these two were not, and they
    are exactly the pair the documented no-flag invocation depends on. A silent
    drift in `--out` writes the nine records where the aggregation does not
    read: the study looks unstarted, and the next resume pays for all 13.5 hours
    again. A drift in `--data` points the driver at a directory with no
    episodes, where every cell fails identically.
    """
    defaults = vars(run_study._parser().parse_args([]))
    assert defaults["out"] == "runs/m3_study_v2", (
        "the nine records are the study's only artifact and this is where the "
        "aggregation and the plan's own commands look for them. It is NOT "
        "runs/m3_study: that directory holds M3b's nine records, which carry "
        "none of the four provenance keys, so a bare invocation defaulting "
        "there finds every M3b cell pending and overwrites six unrecoverable "
        "checkpoints and records")
    assert defaults["out"] != "runs/m3_study"
    assert defaults["data"] == "data/my_way_home"

    # `--data` has a second home, and the two drifting apart would train the
    # study on episodes nothing else in the codebase is configured to score.
    data_root = next(f for f in fields(Config) if f.name == "data_root")
    assert defaults["data"] == data_root.default, (
        "the driver's default episode directory and `Config.data_root` "
        "disagree; one of them is now pointing somewhere nothing else reads")

    # Both stay `str` end to end -- see `pending_jobs`' own tolerance test.
    assert isinstance(defaults["out"], str)
    assert isinstance(defaults["data"], str)


def test_required_record_keys_is_exactly_this_hand_written_set():
    """THE GUARD THAT DOES NOT SHRINK WITH WHAT IT GUARDS.

    Both of the old guards moved with `REQUIRED_RECORD_KEYS`: one was
    parametrised over it, so deleting a key deleted the case that would have
    caught the deletion, and the other asked only for a SUBSET of a real
    record's keys, which a smaller set still is. `EXPECTED_RECORD_KEYS` is an
    independent literal and this is an EQUALITY, so a key removed and a key
    added both fail here by name.
    """
    assert run_study.REQUIRED_RECORD_KEYS == EXPECTED_RECORD_KEYS, (
        "the driver's completeness set no longer matches the one written out "
        "in this file; a key dropped from it marks a record short of that "
        "field as a finished cell, and a key added to it makes every cell "
        "pending forever")
    # Spelled out above rather than imported, so that RENAMING the constant in
    # `study.py` fails too; this keeps the two linked deliberately.
    assert NONFINITE_KEY in EXPECTED_RECORD_KEYS
    assert not (EXPECTED_RECORD_KEYS & OPTIONAL_RECORD_KEYS), (
        "a key cannot be both required and optional")


# ---------------------------------------------------------------------------
# pending_jobs: discovery
# ---------------------------------------------------------------------------

def test_all_nine_jobs_are_pending_on_a_fresh_directory(tmp_path):
    jobs = run_study.pending_jobs(tmp_path, ARMS, SEEDS)
    assert len(jobs) == 9
    assert {(j.arm, j.seed) for j in jobs} == {(a, s) for a in ARMS for s in SEEDS}


@pytest.mark.parametrize("job", [DONE_JOB, CORRUPT_JOB, INCOMPLETE_JOB],
                         ids=lambda j: f"{j.arm}_s{j.seed}")
def test_a_job_with_a_complete_record_is_skipped(tmp_path, job):
    """Resumability. Every cell is ~1.5 h; re-running one is a lost
    afternoon. Parametrised over all three arms because a skip that only fires for
    one of them would leave the study re-running the other six cells."""
    _write_complete(tmp_path, job)
    jobs = run_study.pending_jobs(tmp_path, ARMS, SEEDS)
    assert job not in jobs
    assert len(jobs) == 8


def test_pending_jobs_reads_the_arms_and_seeds_it_is_given(tmp_path):
    """The study is 3x3, so a square fixture cannot tell the two arguments
    apart -- swapping them still yields nine jobs, with arms in the seed field.
    These are 2 and 4."""
    jobs = run_study.pending_jobs(tmp_path, UNEVEN_ARMS, UNEVEN_SEEDS)
    assert len(jobs) == len(UNEVEN_ARMS) * len(UNEVEN_SEEDS) == 8
    assert {j.arm for j in jobs} == set(UNEVEN_ARMS)
    assert {j.seed for j in jobs} == set(UNEVEN_SEEDS)
    assert all(isinstance(j.seed, int) for j in jobs)
    assert "pixel_ae" not in {j.arm for j in jobs}, (
        "the excluded arm was run anyway; the defaults are being used instead "
        "of the caller's arguments")


def test_a_repeated_arm_or_seed_does_not_buy_the_same_cell_twice(tmp_path):
    """`--arms pixel_ae pixel_ae` is a typo, not a request for 9 GPU-hours."""
    assert run_study.pending_jobs(tmp_path, ("pixel_ae", "pixel_ae"), (0, 0)) == [
        StudyJob("pixel_ae", 0)]


def test_pending_jobs_accepts_the_string_argparse_hands_it(tmp_path):
    """`--out` arrives as a `str`. This is the driver's very first action, so
    a `Path`-only implementation raises before any GPU time is spent."""
    out_dir = str(tmp_path / "study")
    Path(out_dir).mkdir()
    _write_complete(out_dir, DONE_JOB)
    jobs = run_study.pending_jobs(out_dir, ARMS, SEEDS)
    assert len(jobs) == 8 and DONE_JOB not in jobs


# ---------------------------------------------------------------------------
# pending_jobs: what counts as "already done"
# ---------------------------------------------------------------------------

def test_a_corrupt_record_is_treated_as_pending(tmp_path):
    """A truncated JSON file from a killed process must not mark a job done."""
    job_record_path(tmp_path, CORRUPT_JOB).write_text("{not json")
    assert CORRUPT_JOB in run_study.pending_jobs(tmp_path, ARMS, SEEDS)


def test_an_empty_record_file_is_treated_as_pending(tmp_path):
    """Zero bytes is what a process killed between `open` and `write` left
    behind before the write became atomic."""
    job_record_path(tmp_path, CORRUPT_JOB).write_text("")
    assert CORRUPT_JOB in run_study.pending_jobs(tmp_path, ARMS, SEEDS)


@pytest.mark.parametrize("text", ["null", "[]", "3", '"done"'])
def test_a_record_that_is_not_an_object_is_treated_as_pending(tmp_path, text):
    job_record_path(tmp_path, CORRUPT_JOB).write_text(text)
    assert CORRUPT_JOB in run_study.pending_jobs(tmp_path, ARMS, SEEDS)


def _assert_missing_key_makes_the_cell_pending(tmp_path, missing: str) -> None:
    """Delete `missing` from an otherwise complete record and check the driver
    calls the cell pending; then write the record intact and check it does not.

    The body shared by the two parametrised tests below, so the two cannot
    drift: each keeps its own literal list of keys, and this is the one place
    that says what "missing makes it pending" means -- including the control,
    without which an implementation that calls everything incomplete would
    pass every parametrisation of both.
    """
    path = job_record_path(tmp_path, INCOMPLETE_JOB)
    damaged = _sanitised(INCOMPLETE_JOB)
    del damaged[missing]
    path.write_text(json.dumps(damaged))
    assert not run_study.record_is_complete(path, INCOMPLETE_JOB)
    assert INCOMPLETE_JOB in run_study.pending_jobs(tmp_path, ARMS, SEEDS)

    # The control. Without it, an implementation that calls everything
    # incomplete would pass all twenty-two parametrisations of the first
    # caller and all four of the second.
    path.write_text(json.dumps(_sanitised(INCOMPLETE_JOB)))
    assert run_study.record_is_complete(path, INCOMPLETE_JOB)
    assert INCOMPLETE_JOB not in run_study.pending_jobs(tmp_path, ARMS, SEEDS)


@pytest.mark.parametrize("missing", sorted(EXPECTED_RECORD_KEYS))
def test_a_record_missing_any_field_is_treated_as_pending(tmp_path, missing):
    """THE DANGEROUS CASE: valid JSON that is not a complete record.

    A file that fails to parse costs one re-run. A file that parses and is
    short of `position` is read by the aggregation as this cell's result, and
    the study reports a cell nobody measured. Every required key is removed in
    turn; the control in `_assert_missing_key_makes_the_cell_pending` writes
    the same record intact.

    PARAMETRISED OVER `EXPECTED_RECORD_KEYS`, NOT OVER THE DRIVER'S OWN SET.
    Deriving the cases from the collection under test made this shrink with it:
    delete "position" from `REQUIRED_RECORD_KEYS` and the position case simply
    stopped being generated, so the suite stayed green while a pixel_ae record with
    no position block became a finished cell.
    """
    _assert_missing_key_makes_the_cell_pending(tmp_path, missing)


def test_the_m3c_keys_are_in_both_literals():
    """The four are a slice of `EXPECTED_RECORD_KEYS` AND of the driver's set.

    `M3C_RECORD_KEYS` is what the parametrisation below reads. If it drifted
    from `EXPECTED_RECORD_KEYS` -- a key renamed in one literal and not the
    other -- the parametrised guard would be exercising a key the driver was
    never asked to require, and pass because deleting an absent key from a
    record changes nothing. Pinned against the driver's own set too, so the
    four cannot quietly become optional.
    """
    assert M3C_RECORD_KEYS == {"git_sha", "device", "encoder_params", "history"}
    assert M3C_RECORD_KEYS <= EXPECTED_RECORD_KEYS
    assert M3C_RECORD_KEYS <= run_study.REQUIRED_RECORD_KEYS
    assert not (M3C_RECORD_KEYS & OPTIONAL_RECORD_KEYS)


@pytest.mark.parametrize("missing", sorted(M3C_RECORD_KEYS))
def test_a_record_missing_a_provenance_field_is_treated_as_pending(
    tmp_path, missing
):
    """Each of the four M3c keys, ALONE. A record from the M3b code state --
    every field the gate reads, none of the provenance -- is exactly what a
    stale checkout on the box would write, and it must be re-run, not
    reported: the whole point of `git_sha` is that a record without it
    cannot be shown to come from the same code as the other eight.

    Parametrised over the LITERAL `M3C_RECORD_KEYS`, never over the driver's
    set: derive the cases from `REQUIRED_RECORD_KEYS` and dropping `history`
    from it deletes the case that would have caught the drop.
    """
    assert missing in _record_body(INCOMPLETE_JOB), (
        f"the fixture record never carried {missing!r}, so deleting it is a "
        "no-op and this case cannot fail")
    _assert_missing_key_makes_the_cell_pending(tmp_path, missing)


def test_a_record_holding_exactly_the_required_keys_is_complete(tmp_path):
    """THE OTHER DIRECTION, WHICH NO FIXTURE PRODUCED: nothing spare.

    `complete_record` asks `REQUIRED_RECORD_KEYS <= set(record)`. Turning that
    subset test into a PROPER subset (`<`) survived the whole suite, because
    `_record_body` always writes all twenty-four top-level keys and a
    proper-subset relation holds for every fixture in this file. The one
    record shape that tells `<` from `<=` is the one with exactly the
    twenty-two required keys and neither optional diagnostic -- a finished
    cell that legitimately lacks
    `kl_dyn_max` and `loss_last20`, which `OPTIONAL_RECORD_KEYS` above declares
    a record is allowed to lack.

    Under `<` such a cell is permanently pending: every resume re-runs it, a
    ~1.5-hour `pixel_ae` cell is paid for again on every attempt, and the
    study never converges. That is the disaster `REQUIRED_RECORD_KEYS`' own docstring calls
    one with no symptom, and the driver and its own tests would flatly disagree
    about what a finished cell is.
    """
    path = job_record_path(tmp_path, INCOMPLETE_JOB)
    exact = _sanitised(INCOMPLETE_JOB)
    for key in OPTIONAL_RECORD_KEYS:
        del exact[key]
    assert set(exact) == set(EXPECTED_RECORD_KEYS), (
        "this record must hold the required keys and NOTHING ELSE, or it is "
        "still a proper superset and cannot tell `<` from `<=`")
    path.write_text(json.dumps(exact))

    assert run_study.record_is_complete(path, INCOMPLETE_JOB)
    assert INCOMPLETE_JOB not in run_study.pending_jobs(tmp_path, ARMS, SEEDS)


def test_a_record_carrying_a_bare_nan_token_is_treated_as_pending(tmp_path):
    """`write_record` cannot emit one; a file that has one is not ours.

    Python's `json` accepts the bare `NaN` token, so this file would load here
    and then be rejected by the strict parser the aggregation uses -- after the
    study is over. It is cheaper to re-run the cell.
    """
    path = job_record_path(tmp_path, CORRUPT_JOB)
    damaged = _sanitised(CORRUPT_JOB)
    damaged["position"]["gap_final"] = float("nan")
    path.write_text(json.dumps(damaged))          # allow_nan defaults to True
    assert "NaN" in path.read_text()
    assert not run_study.record_is_complete(path, CORRUPT_JOB)

    # The control: the same field as the `null` the real writer emits.
    damaged["position"]["gap_final"] = None
    path.write_text(json.dumps(damaged))
    assert run_study.record_is_complete(path, CORRUPT_JOB)


def test_a_record_naming_a_different_arm_is_treated_as_pending(tmp_path):
    """Exercised ALONE, with the seed correct: a check written as one `or`
    over both fields would otherwise pass with either half deleted."""
    other_arm = next(a for a in ARMS if a != INCOMPLETE_JOB.arm)
    path = _write_complete(tmp_path, INCOMPLETE_JOB, arm=other_arm)
    assert json.loads(path.read_text())["seed"] == INCOMPLETE_JOB.seed
    assert not run_study.record_is_complete(path, INCOMPLETE_JOB)
    assert INCOMPLETE_JOB in run_study.pending_jobs(tmp_path, ARMS, SEEDS)


def test_a_record_naming_a_different_seed_is_treated_as_pending(tmp_path):
    """The other half, alone: the arm is right and only the seed is wrong."""
    path = _write_complete(tmp_path, INCOMPLETE_JOB, seed=CLI_SEED)
    assert json.loads(path.read_text())["arm"] == INCOMPLETE_JOB.arm
    assert not run_study.record_is_complete(path, INCOMPLETE_JOB)
    assert INCOMPLETE_JOB in run_study.pending_jobs(tmp_path, ARMS, SEEDS)


def test_a_directory_at_the_record_path_is_treated_as_pending(tmp_path):
    """`read_text` on a directory raises `IsADirectoryError`, and the resume
    check must answer "pending", not take the whole run down with it."""
    job_record_path(tmp_path, CORRUPT_JOB).mkdir(parents=True)
    assert CORRUPT_JOB in run_study.pending_jobs(tmp_path, ARMS, SEEDS)


def test_a_byte_damaged_record_is_treated_as_pending(tmp_path):
    """`read_text` raises `UnicodeDecodeError` on this, and that is a
    `ValueError`, NOT an `OSError`.

    The directory case above is the `OSError` half. This is the other one, and
    it is the likelier of the two: a truncated `scp`, a half-flushed page, a
    file recovered off a failing disk. Caught only as `OSError`, one damaged
    file among the nine takes the whole resume down at hour zero -- before any
    cell has started, and with the other eight records sitting there intact.
    """
    path = job_record_path(tmp_path, CORRUPT_JOB)
    path.write_bytes(b'{"arm": "pixel_ae", "seed": \xff\xfe\x00 not utf-8}')
    with pytest.raises(UnicodeDecodeError):
        path.read_text()

    assert not run_study.record_is_complete(path, CORRUPT_JOB)
    assert CORRUPT_JOB in run_study.pending_jobs(tmp_path, ARMS, SEEDS)


@pytest.mark.parametrize("arm", ARMS)
def test_a_real_record_from_run_job_is_recognised_as_complete(
        tmp_path, small_buffer, arm):
    """THE DRIFT GUARD, and the reason it is worth running the real thing.

    `REQUIRED_RECORD_KEYS` is a second copy of what `run_job` writes, and drift
    is catastrophic in both directions with no symptom either way: a key listed
    that `run_job` never writes makes every cell permanently pending, so the
    study restarts from zero on every resume and never finishes; a key dropped
    from the set lets a record short of that field count as done.

    Run for all three arms because `frozen_ssl` is the only one whose feature
    cache is not named after it, and an arm that no test ever runs is the arm
    that dies three seeds deep into the study.
    """
    job = StudyJob(arm, INCOMPLETE_JOB.seed)
    record = study.run_job(job, small_buffer, str(tmp_path), **RUN_KW)

    path = job_record_path(tmp_path, job)
    assert path.is_file()
    written = set(json.loads(path.read_text()))
    # EQUALITY, not a subset. A subset test is satisfied by a set that has lost
    # keys, so it could not see the direction that matters: `run_job` writing
    # a field the driver has stopped requiring.
    assert run_study.REQUIRED_RECORD_KEYS == written - OPTIONAL_RECORD_KEYS, (
        "the driver's completeness set and `run_job`'s actual output have "
        "drifted apart: a key demanded but not written makes every cell "
        "pending forever, and a key written but not demanded lets a record "
        "short of it count as a finished ~1.5-hour cell")
    assert OPTIONAL_RECORD_KEYS <= written, (
        "these are subtracted above to make that an equality; if `run_job` "
        "has stopped writing them the subtraction quietly stops testing "
        "anything")
    assert run_study.record_is_complete(path, job)
    assert job not in run_study.pending_jobs(tmp_path, ARMS, SEEDS)
    # The live record is what `main` prints from, and it is not the file.
    assert run_study.record_names_the_job(record, job)


# ---------------------------------------------------------------------------
# foreign records: another code state's --out
# ---------------------------------------------------------------------------
#
# The day the four provenance keys became required, every M3b record in
# `runs/m3_study` turned "pending": each parses, names its cell, carries every
# key the gate reads and none of the four. A driver pointed there saw nine
# pending cells and would have RE-RUN OVER the only copy of six ~1.5-hour
# cells. The `--out` default fix stops the bare invocation; `foreign_records`
# is the guard that survives the next rename.

PRE_M3C_RECORD_KEYS = EXPECTED_RECORD_KEYS - M3C_RECORD_KEYS
"""What an M3b record holds, as this file's own literal difference."""

M3B_STUDY_DIR = Path(__file__).resolve().parents[2] / "runs" / "m3_study"
"""The real M3b directory, READ ONLY and never required: the test that copies
two of its records into `tmp_path` skips when it is absent (a fresh clone has
no `runs/`), and every other test here builds its foreign files by hand."""


def _write_m3b_shaped(out_dir, job: StudyJob) -> Path:
    """A record exactly as the M3b code state wrote one: complete for the
    gate, without any of the four provenance keys. Through `write_record`, so
    `nonfinite` is present as it is in the real files."""
    body = _record_body(job)
    for key in M3C_RECORD_KEYS:
        del body[key]
    assert set(body) | {NONFINITE_KEY} == PRE_M3C_RECORD_KEYS | OPTIONAL_RECORD_KEYS
    path = job_record_path(out_dir, job)
    write_record(path, body)
    return path


def test_the_pre_m3c_keys_are_the_drivers_own_difference():
    """Pinned against the driver's split of its set, so that a key moved from
    one side to the other -- `history` demoted to optional, say -- shows up
    here rather than as a directory that is neither pending nor foreign."""
    assert run_study.PROVENANCE_KEYS == M3C_RECORD_KEYS
    assert run_study.PRE_M3C_RECORD_KEYS == PRE_M3C_RECORD_KEYS
    assert run_study.PRE_M3C_RECORD_KEYS | run_study.PROVENANCE_KEYS == \
        run_study.REQUIRED_RECORD_KEYS
    assert not (run_study.PRE_M3C_RECORD_KEYS & run_study.PROVENANCE_KEYS)


def test_an_m3b_shaped_record_is_foreign_and_a_truncated_one_of_ours_is_not(
        tmp_path):
    """The second shape, and its two neighbours that must NOT match it.

    ALL FOUR keys absent is the M3b code state. Only SOME of the four absent
    is a truncated or hand-edited record of our own -- `complete_record`
    already calls that pending and a re-run is the right answer, so it must
    not be refused over. And a complete record is a finished cell.
    """
    path = _write_m3b_shaped(tmp_path, DONE_JOB)
    assert not run_study.record_is_complete(path, DONE_JOB), (
        "fixture: the M3b shape must still read as pending to the resume "
        "check; foreign is a refusal on TOP of that, not instead of it")
    found = run_study.foreign_records(tmp_path)
    assert len(found) == 1, found
    assert path.name in found[0]
    assert f"{DONE_JOB.arm}/s{DONE_JOB.seed}" in found[0]
    for key in sorted(M3C_RECORD_KEYS):
        assert key in found[0], "the refusal names the four missing keys"

    # Some, not all, of the four missing: ours, pending, not foreign. Each
    # of the four is put back ALONE, so a guard written as "any of the four
    # absent" fails on every one of these rather than on none.
    for kept in sorted(M3C_RECORD_KEYS):
        body = _record_body(DONE_JOB)
        for key in M3C_RECORD_KEYS - {kept}:
            del body[key]
        write_record(path, body)
        assert not run_study.record_is_complete(path, DONE_JOB)
        assert run_study.foreign_records(tmp_path) == [], (
            f"a record carrying only {kept!r} of the four is a damaged record "
            "of OURS, and refusing over it would turn one truncated file "
            "into a study that cannot be resumed")

    # The control: a complete record is a finished cell, not a foreigner.
    _write_complete(tmp_path, DONE_JOB)
    assert run_study.record_is_complete(path, DONE_JOB)
    assert run_study.foreign_records(tmp_path) == []


def test_an_m3b_shaped_record_named_for_another_cell_is_not_foreign(tmp_path):
    """The shape is refused only when the record NAMES the cell its filename
    does: a mislabelled file is `record_names_the_job`'s problem, and one
    missing a gate key is merely pending. Each half of the condition alone."""
    path = _write_m3b_shaped(tmp_path, DONE_JOB)
    body = json.loads(path.read_text())
    body["seed"] = CLI_SEED
    path.write_text(json.dumps(body))
    assert run_study.foreign_records(tmp_path) == []

    body = json.loads(_write_m3b_shaped(tmp_path, DONE_JOB).read_text())
    del body["position"]
    path.write_text(json.dumps(body))
    assert run_study.foreign_records(tmp_path) == [], (
        "short of a gate key it is not a complete M3b record either; it is "
        "pending, and pending is re-run, not refused")


def test_a_cell_named_for_a_retired_arm_is_foreign_by_name_alone(tmp_path):
    """The first shape: `cnn` was the study's arm under M3b and is not in
    `ARMS`. The record need not even parse -- the name is enough, and so is a
    checkpoint with no record beside it."""
    (tmp_path / "result_cnn_seed0.json").write_text("not even json")
    (tmp_path / "world_model_cnn_seed1.pt").write_bytes(b"")
    # Neither of these is a cell file: the quarantined suffix is exactly what
    # keeps a mislabelled record out of the aggregation's glob, and a
    # checkpoint for a study arm is what every finished cell leaves behind.
    (tmp_path / "result_cnn_seed2.json.mislabelled").write_text("{}")
    (tmp_path / f"world_model_{DONE_JOB.arm}_seed{DONE_JOB.seed}.pt").write_bytes(b"")
    (tmp_path / "study.log").write_text("cnn seed 0\n")

    found = run_study.foreign_records(tmp_path)
    assert len(found) == 2, found
    assert "result_cnn_seed0.json" in found[0] and "record" in found[0]
    assert "world_model_cnn_seed1.pt" in found[1] and "checkpoint" in found[1]
    assert all("'cnn'" in line for line in found)
    assert "mislabelled" not in "".join(found)


def test_foreign_records_is_empty_for_an_absent_or_empty_out(tmp_path):
    """The common case, in both forms: `--out` not created yet, and created
    with nothing in it. Neither may raise -- this runs before the mkdir in
    `acquire_lock` and before anything else has looked at `--out`."""
    assert run_study.foreign_records(tmp_path / "not_yet") == []
    assert run_study.foreign_records(tmp_path) == []
    assert run_study.foreign_records(str(tmp_path)) == [], (
        "argparse hands over a str, as everywhere else in the driver")


def test_main_refuses_a_foreign_out_with_its_own_status_and_runs_nothing(
        tmp_path, monkeypatch, capsys):
    """THE REFUSAL, end to end: a named non-zero status, no cell trained, no
    claim file left behind, `--data` never touched, and a remedy that says
    "fresh --out" and never "delete"."""
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "m3b_directory"
    _write_m3b_shaped(out, CORRUPT_JOB)
    (out / "result_cnn_seed0.json").write_text("{}")
    gone = tmp_path / "data_not_mounted_yet"

    status = run_study.main(["--data", str(gone), "--out", str(out),
                             *MATCHING_ARGV])

    assert status == run_study.EXIT_FOREIGN_RECORDS == 23
    assert calls == [], "a cell was trained over another study's records"
    printed = capsys.readouterr().out
    assert printed.startswith("FOREIGN RECORDS:"), (
        "the refusal is the first thing in the log: a 'pending' line above "
        "it would describe cells that are not this study's")
    assert "recorded no provenance" in printed
    assert "choose a fresh --out" in printed
    assert "result_cnn_seed0.json" in printed
    assert job_record_path(out, CORRUPT_JOB).name in printed
    assert "delete" not in printed.replace("Do NOT delete", ""), (
        "the remedy must never be a deletion; these files may be the only "
        "copy of the cells they describe")
    assert not (out / run_study.LOCK_NAME).exists(), (
        "the refusal came after the lock was taken")
    assert not gone.exists(), (
        "the refusal came after `ReplayBuffer` was asked for the episodes")


def test_main_refuses_before_the_lock_so_a_stale_claim_does_not_hide_it(
        tmp_path, monkeypatch, capsys):
    """Order pinned: a foreign directory that also holds a claim file is
    reported as FOREIGN, not LOCKED. The lock's remedy is "delete the claim
    and start again", which for this directory is the disaster itself."""
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "m3b_directory"
    _write_m3b_shaped(out, CORRUPT_JOB)
    (out / run_study.LOCK_NAME).write_text('{"pid": 1}')

    status = run_study.main(["--data", str(tmp_path / "none"),
                             "--out", str(out), *MATCHING_ARGV])

    assert status == run_study.EXIT_FOREIGN_RECORDS
    assert calls == []
    assert "FOREIGN RECORDS" in capsys.readouterr().out
    assert (out / run_study.LOCK_NAME).read_text() == '{"pid": 1}', (
        "the stale claim must be left exactly as found")


@pytest.mark.skipif(not M3B_STUDY_DIR.is_dir(),
                    reason="the M3b directory runs/m3_study is not on this "
                           "machine (runs/ is gitignored)")
def test_main_refuses_a_copy_of_the_real_m3b_records(
        tmp_path, monkeypatch, capsys):
    """Against the real thing: two of M3b's own records, COPIED into
    `tmp_path` (the M3b directory is read and never written), one of each
    shape -- `cnn`, an arm not in `ARMS`, and `frozen_ssl`, a complete
    pre-M3c record with no provenance. Both fixture facts are asserted before
    the driver sees them, so this cannot pass by the first shape alone."""
    out = tmp_path / "copy_of_m3b"
    out.mkdir()
    names = ("result_cnn_seed0.json", "result_frozen_ssl_seed0.json")
    originals = {}
    for name in names:
        source = M3B_STUDY_DIR / name
        if not source.is_file():
            pytest.skip(f"{source} is not on this machine")
        originals[name] = source.read_bytes()
        (out / name).write_bytes(originals[name])
    assert "cnn" not in ARMS
    frozen = json.loads(originals["result_frozen_ssl_seed0.json"])
    assert frozen["arm"] == "frozen_ssl" and frozen["seed"] == 0
    assert PRE_M3C_RECORD_KEYS <= set(frozen)
    assert not (M3C_RECORD_KEYS & set(frozen)), (
        "the M3b record carries a provenance key; the second shape is not "
        "what this test thinks it is")

    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    status = run_study.main(["--data", str(tmp_path / "none"),
                             "--out", str(out), *MATCHING_ARGV])

    assert status == run_study.EXIT_FOREIGN_RECORDS
    assert calls == []
    printed = capsys.readouterr().out
    assert all(name in printed for name in names), printed
    assert not (out / run_study.LOCK_NAME).exists()
    for name in names:
        assert (out / name).read_bytes() == originals[name]
        assert (M3B_STUDY_DIR / name).read_bytes() == originals[name], (
            "the M3b directory was written to")


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------

def test_no_arm_is_deferred_and_the_order_is_arm_then_seed(tmp_path):
    """With `_SLOW_ARMS` empty the whole schedule is `(arm, seed)`, arms in
    string order. Under M3b the 8.3-hour pixel arm ran last so an interruption
    still left the treatment/control contrast complete; every M3c arm costs
    the same ~1.5 h, so there is nothing to defer and the order is pinned
    here IN FULL rather than as "the slow one is last" -- the arm names are
    spelled out so a shrunken ARMS cannot pass this by iterating less."""
    jobs = run_study.pending_jobs(tmp_path, ARMS, SEEDS)
    assert sorted(ARMS) == ["frozen_ssl", "pixel_ae", "random_vit"]
    assert jobs == [
        StudyJob("frozen_ssl", 0), StudyJob("frozen_ssl", 1),
        StudyJob("frozen_ssl", 2),
        StudyJob("pixel_ae", 0), StudyJob("pixel_ae", 1),
        StudyJob("pixel_ae", 2),
        StudyJob("random_vit", 0), StudyJob("random_vit", 1),
        StudyJob("random_vit", 2),
    ]
    assert len(jobs) == 9


def test_the_order_is_fully_determined_and_seeds_ascend(tmp_path):
    """The seeds are handed over SHUFFLED on purpose.

    `sorted` is stable, so with the seed missing from the sort key the arms
    would still be grouped correctly and only the seeds inside each arm would
    follow the caller's arbitrary order. An interrupted study should always
    hold seeds 0..k of an arm, never an arbitrary subset of them.
    """
    jobs = run_study.pending_jobs(tmp_path, ("random_vit", "pixel_ae"), (2, 0, 1))
    assert jobs == [
        StudyJob("pixel_ae", 0), StudyJob("pixel_ae", 1), StudyJob("pixel_ae", 2),
        StudyJob("random_vit", 0), StudyJob("random_vit", 1),
        StudyJob("random_vit", 2),
    ]


def test_the_order_holds_after_a_partial_resume(tmp_path):
    """Resume and ordering interact: what is left must still be in
    `(arm, seed)` order, with the completed cells simply absent."""
    _write_complete(tmp_path, StudyJob("frozen_ssl", 0))
    _write_complete(tmp_path, StudyJob("pixel_ae", 0))
    jobs = run_study.pending_jobs(tmp_path, ARMS, SEEDS)
    assert jobs == [
        StudyJob("frozen_ssl", 1), StudyJob("frozen_ssl", 2),
        StudyJob("pixel_ae", 1), StudyJob("pixel_ae", 2),
        StudyJob("random_vit", 0), StudyJob("random_vit", 1),
        StudyJob("random_vit", 2),
    ]


# ---------------------------------------------------------------------------
# the write path: a record is complete or it is absent
# ---------------------------------------------------------------------------

def test_a_failed_write_leaves_the_previous_record_intact_and_no_litter(
        tmp_path, monkeypatch):
    """The resume decision is only as good as the file it reads.

    `write_text` truncates the destination and then streams into it, so a
    process killed in between leaves a prefix AT THE REAL PATH -- and the
    prefix that happens to parse is the one that marks an unfinished cell done.
    The write therefore goes to a temporary beside the destination and is swapped
    in with `os.replace`. Here the swap is made to fail, which stands for the
    process dying at the worst possible moment: the old record must survive
    byte for byte, and no temporary may be left for the aggregation's glob.
    """
    path = _write_complete(tmp_path, DONE_JOB)
    before = path.read_bytes()

    def boom(src, dst):
        raise RuntimeError("killed between the write and the rename")

    monkeypatch.setattr(study, "os",
                        SimpleNamespace(getpid=os.getpid, replace=boom))
    with pytest.raises(RuntimeError):
        write_record(path, _record_body(DONE_JOB, steps=RECORD_STEPS + 1))

    assert path.read_bytes() == before, (
        "the destination was written in place, so a killed process can leave a "
        "partial record where the driver looks for a complete one")
    assert [p.name for p in tmp_path.iterdir()] == [path.name], (
        f"a temporary survived the failed write: {sorted(p.name for p in tmp_path.iterdir())}")
    assert run_study.record_is_complete(path, DONE_JOB)


def test_a_successful_write_leaves_only_the_record(tmp_path):
    """The control for the test above: the temporary must not survive success
    either, or every cell leaves a file behind."""
    path = _write_complete(tmp_path, DONE_JOB)
    assert [p.name for p in tmp_path.iterdir()] == [path.name]
    assert json.loads(path.read_text())["steps"] == RECORD_STEPS


# ---------------------------------------------------------------------------
# what the morning's study.log says
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,spec,expected", [
    (float("nan"), "+.4f", "+nan"),
    (float("inf"), "+.4f", "+inf"),
    (float("-inf"), "+.4f", "-inf"),
    (None, "+.4f", "n/a"),
    (0.5, "+.4f", "+0.5000"),
    (3.98, ".2f", "3.98"),
    # The `except` clause, each half alone. Nothing used to enter it at all.
    # NOT the string "n/a": that is what the `None` branch above returns, and
    # a mutation replacing this branch's `str(value)` with a literal "n/a"
    # would then survive.
    ("not-a-number", ".4f", "not-a-number"),   # float(...) raises ValueError
    ("", ".4f", ""),                # float("")     raises ValueError
    (["gap"], ".4f", "['gap']"),    # float(["gap"]) raises TypeError
    ({"gap": 1}, ".4f", "{'gap': 1}"),
])
def test_fmt_handles_every_value_a_record_field_can_hold(value, spec, expected):
    """`gap_final` is NaN by contract on a degenerate cell and `None` in a
    sanitised one. `format(None, "+.4f")` raises, and raising in the per-job
    print would end the run after cell one with eight cells never started.

    The last four cases are the `try`, which no test used to enter: a
    hand-edited or half-converted record can hold a string or a list where a
    number belongs, and `float()` raises `ValueError` on the one and
    `TypeError` on the other. Both halves of the `except` tuple are therefore
    exercised on their own, and neither can be dropped.
    """
    assert run_study._fmt(value, spec) == expected


@pytest.mark.parametrize("record,keys,expected", [
    # The control. Without it an implementation that always answered None
    # would pass every case below.
    ({"position": {"gap_final": 0.5}}, ("position", "gap_final"), 0.5),
    # `key not in node`, alone: the node IS a dict and the key is absent.
    ({"position": {"gap_final": 0.5}}, ("angle", "gap_final"), None),
    ({"position": {"band_median": 1.0}}, ("position", "gap_final"), None),
    # `not isinstance(node, dict)`, alone: the key would be found if the node
    # were a mapping, and `"gap_final" in 3.0` raises TypeError instead.
    ({"position": 3.0}, ("position", "gap_final"), None),
    ({"position": None}, ("position", "gap_final"), None),
    ({"position": ["gap_final"]}, ("position", "gap_final"), None),
    ("the whole record is a string", ("position",), None),
], ids=["control", "block_missing", "field_missing", "block_is_a_float",
        "block_is_null", "block_is_a_list", "record_is_not_a_mapping"])
def test_get_answers_none_rather_than_raising_on_a_damaged_record(
        record, keys, expected):
    """Both halves of `_get`'s guard, each exercised on its own.

    `_get` exists so that the per-cell print can never kill the loop, and the
    two halves fail differently: a missing key raises `KeyError` without the
    second, and a block replaced by a scalar raises `TypeError` without the
    first. A record whose `position` is a bare number is exactly what a
    half-converted or hand-edited file looks like.
    """
    assert run_study._get(record, *keys) == expected


def test_job_summary_survives_a_record_whose_blocks_are_not_mappings():
    """The same hole, reached the way the driver would reach it: after the
    record is safely on disk, between cell three and cell four."""
    job = StudyJob("random_vit", 0)
    record = {"arm": job.arm, "seed": job.seed, "position": 3.0,
              "angle": None, "filtering": "n/a", "reward": [1.0],
              "episodes": 7, "steps_per_second": "fast"}
    text = run_study.job_summary(job, record, 1.0)
    assert "random_vit" in text and "n/a" in text


def test_job_summary_never_raises_on_a_record_full_of_holes():
    """The print must not be the thing that kills a ~13.5-hour run.

    Every numeric field is simultaneously NaN, None or missing outright -- the
    shape a degenerate cell and a partially-sanitised record can both take.
    """
    job = StudyJob("frozen_ssl", 2)
    record = {
        "arm": job.arm, "seed": job.seed,
        "steps_per_second": float("nan"),
        "kl_rate_above_free_bits": None,
        "position": {"gap_final": float("nan"), "band_median": None},
        # "angle", "filtering", "reward", "episodes" and the rest missing
    }
    text = run_study.job_summary(job, record, float("nan"))
    assert "frozen_ssl" in text and "n/a" in text and "nan" in text


def test_job_summary_renders_every_field_from_the_place_it_claims_to():
    """THE WHOLE BLOCK, CHARACTER FOR CHARACTER. Four of ~25 fields were pinned.

    `job_summary` is the morning's only view of what the nine cells measured,
    and its own docstring promises the configuration line is "READ BACK OUT OF
    THE RECORD, not echoed from this script's own flags". The steps/seq_len
    half of that promise was guarded. Nothing else was: the arm and the seed on
    that very line could be echoed from the job, `split_seed` could be made a
    literal, `wall` and `recorded` could be exchanged, `band_median` could show
    `band_min`, `finite` and `n_steps` could be swapped, `mse` and
    `baseline_mse` could be swapped, `r2` could read `mse`, the angle block
    could vanish, the reward block could vanish, and the lines could be joined
    by spaces -- all with a green suite.

    An equality against a hand-written block fails on every one of those.
    `SUMMARY_JOB` and `SUMMARY_RECORD_JOB` are different cells on purpose, so
    the `arm=... seed=... (requested .../s...)` line shows the two disagreeing
    rather than agreeing by construction.
    """
    text = run_study.job_summary(
        SUMMARY_JOB, _summary_record(), SUMMARY_WALL_SECONDS)

    assert text == EXPECTED_SUMMARY, (
        "the per-cell log block changed; every number below is what an "
        "operator reads at 8am to decide whether a cell is broken")

    # The two halves of the cell line really are distinguishable, or the
    # equality above would hold for a block that echoed the job on both sides.
    assert SUMMARY_JOB.arm != SUMMARY_RECORD_JOB.arm
    assert SUMMARY_JOB.seed != SUMMARY_RECORD_JOB.seed
    assert SUMMARY_RECORD_JOB.seed != SPLIT_SEED != SUMMARY_JOB.seed, (
        "split_seed is printed on the same line as the recorded seed; if "
        "either seed equalled it, a mutation reading one for the other would "
        "be a no-op")


@pytest.mark.parametrize("criterion_4", [False, True], ids=["fails", "passes"])
def test_job_summary_reads_its_two_booleans_out_of_the_record_both_ways(
        criterion_4):
    """A BOOL HAS TWO VALUES, SO ONE RENDERING EXCLUDES ONLY ONE LITERAL.

    `EXPECTED_SUMMARY` above is an equality over the whole block and still
    could not tell `_get(record, "filtering", "criterion_4",
    "latent_beats_embedding")` from the literal `True`, nor `_get(record,
    "reward", "is_degenerate")` from the literal `False`, because the fixture
    said exactly what those literals say -- the L1 species, inside the guard
    that was supposed to be the thorough one. Both are now written the other
    way round there, which kills those two mutations; this renders the block
    with both flipped again, which kills the two mutations that go the other
    way. Neither test alone is enough and neither is redundant.

    WHAT IS AT STAKE. These are the two verdicts an operator reads per cell:
    whether the gate's criterion 4 passed, and whether the reward target was
    degenerate. A constant in either place reports the same verdict for all
    nine cells whatever the records say -- a study whose headline finding is
    printed by the logger rather than measured.

    The two are flipped in OPPOSITE directions in each case, so a mutation
    that renders one of them under the other's label is caught as well.
    """
    record = _summary_record()
    record["filtering"]["criterion_4"]["latent_beats_embedding"] = criterion_4
    record["reward"]["is_degenerate"] = not criterion_4

    text = run_study.job_summary(
        SUMMARY_JOB, record, SUMMARY_WALL_SECONDS)

    assert f"criterion_4_latent_beats_embedding={criterion_4} " in text, (
        "the gate's criterion 4 is not being read from the record; the "
        "per-cell log would report the same verdict for all nine cells")
    assert f"degenerate_target={not criterion_4}" in text, (
        "the reward target's degeneracy is not being read from the record; a "
        "target with no variance would be logged as healthy in every cell")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def test_main_runs_every_pending_job_once_in_cheap_first_order(
        tmp_path, episode_dir, monkeypatch, capsys):
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"

    assert run_study.main(["--data", str(episode_dir), "--out", str(out)]) == 0

    assert [c["job"] for c in calls] == run_study.pending_jobs(
        tmp_path / "empty", ARMS, SEEDS)
    assert len(calls) == 9
    printed = capsys.readouterr().out
    for job in (DONE_JOB, CORRUPT_JOB, INCOMPLETE_JOB):
        assert f"{job.arm} seed {job.seed}" in printed
    assert "9/9 job(s) completed" in printed


def test_main_hands_every_cell_the_one_buffer_it_built(
        tmp_path, episode_dir, monkeypatch):
    """NOTHING REQUIRED THE DRIVER TO HAND `run_job` THE BUFFER IT BUILT.

    The spy has recorded `"buffer"` on every call since it was written and no
    assertion ever read the key -- captured and never checked. So
    `run_job(job, None, ...)` passed the whole suite, and on the rented box all
    nine cells raise inside `run_job`, are swallowed by the per-cell
    `except Exception`, and the run ends 0/9 with nine identical tracebacks
    after burning the session. This is the sibling of the defect where nothing
    required the evaluated model to be the TRAINED one.

    THE ASSERTION IS IDENTITY, NOT EQUALITY. `BUFFER_CAPACITY`'s docstring
    rests the whole study on it -- "the nine cells must be trained and scored
    on the identical episode set or the study compares arms against different
    data" -- and two buffers built over the same directory are equal in every
    respect an equality could reach while being exactly the thing the docstring
    forbids. Counting the constructions is the other half: a throwaway buffer
    built per cell would still be `is` itself.
    """
    built: list[dict] = []
    real = run_study.ReplayBuffer

    def recording(root, **kwargs):
        buffer = real(root, **kwargs)
        built.append({"root": root, "kwargs": kwargs, "buffer": buffer})
        return buffer

    monkeypatch.setattr(run_study, "ReplayBuffer", recording)
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)

    assert run_study.main(["--data", str(episode_dir),
                           "--out", str(tmp_path / "study")]) == 0

    assert len(built) == 1, (
        "the driver built more than one buffer; a second one rebuilt inside "
        "the loop is how the nine cells stop sharing an episode set")
    assert built[0]["root"] == str(episode_dir), (
        "the buffer must be built over --data. Built over --out it has no "
        "episodes, which the EXIT_NO_DATA guard catches only by accident")
    assert built[0]["kwargs"] == {
        "capacity_transitions": run_study.BUFFER_CAPACITY}
    assert len(calls) == 9
    for call in calls:
        assert call["buffer"] is built[0]["buffer"], (
            f"{call['job']} was handed {call['buffer']!r} rather than the "
            "buffer the driver built")


def test_main_prints_the_per_cell_summary_block(
        tmp_path, episode_dir, monkeypatch, capsys):
    """THE BLOCK WAS CHECKED FOR NOT RAISING AND FOR NOTHING ELSE.

    The whole `print(job_summary(...))` statement could be deleted from `_run`
    with a green suite: `job_summary` has unit tests but nothing connected it
    to the driver's output, and the integration assertions only looked at the
    `===== [i/n] =====` banner, the pending header and the completion line.
    With the nine records, `study.log` is the entire product of a ~13.5-hour
    unattended run, and that deletion empties it of every metric while leaving
    every status line intact.

    The arguments are pinned too, each of the three separately: the block was
    equally free to be fed a blank record (every column reads `n/a`), a job
    naming a seed that does not exist, or a constant wall time.
    """
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    # The spy returns in microseconds, so the real elapsed time rounds to the
    # same "0.0" the constant-wall mutation prints. Drive the clock instead.
    monkeypatch.setattr(run_study, "time", _FakeClock())
    out = tmp_path / "study"

    assert run_study.main(["--data", str(episode_dir), "--out", str(out)]) == 0

    printed = capsys.readouterr().out
    assert len(calls) == 9
    for call in calls:
        block = run_study.job_summary(
            call["job"], call["record"], SUMMARY_WALL_SECONDS)
        assert block in printed, (
            f"{call['job']}'s summary block is not in the log; the morning "
            f"has the status lines and no metrics:\n{block}")
    assert printed.count("  reward    mse=") == 9, (
        "one block per cell, or a single block is standing in for nine")

    # The three arguments really are distinguishable here, or the assertion
    # above would hold for a driver that passed the wrong ones.
    assert len({c["job"] for c in calls}) == 9
    assert "wall=0.0s" not in printed, (
        "the driven clock must make a constant wall time visible")
    assert "arm=None" not in printed, "a blank record would render like this"


def test_main_forwards_the_flags_it_was_given(
        tmp_path, episode_dir, monkeypatch):
    """A driver that drops `--device` on the rented CUDA box falls back through
    `get_device` to the CPU and the study takes weeks -- silently, and not on
    this laptop, which has no CUDA either way."""
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)

    assert run_study.main([
        "--data", str(episode_dir), "--out", str(tmp_path / "study"),
        *MATCHING_ARGV,
        "--device", CLI_DEVICE, "--arms", "frozen_ssl",
        "--seeds", "0", "2",
    ]) == 0

    # Two of the three seeds, so ignoring `--seeds` (three calls) and taking
    # only its first value (one call) are both visible.
    assert [c["job"] for c in calls] == [
        StudyJob("frozen_ssl", 0), StudyJob("frozen_ssl", 2)]
    call = calls[0]
    assert call["kwargs"]["steps"] == CLI_STEPS
    assert call["kwargs"]["seq_len"] == CLI_SEQ_LEN
    assert call["kwargs"]["device"] == CLI_DEVICE


def test_main_forwards_the_documented_defaults(
        tmp_path, episode_dir, monkeypatch):
    """The default device is the rented box, not `run_job`'s "mps"."""
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)

    run_study.main(["--data", str(episode_dir), "--out", str(tmp_path / "s"),
                    "--arms", "pixel_ae", "--seeds", "0"])

    assert calls[0]["kwargs"] == {
        "steps": DEFAULT_STEPS, "seq_len": DEFAULT_SEQ_LEN,
        "device": DEFAULT_DEVICE,
    }, (
        "`context` and `horizon` are deliberately absent: `run_job` forwards "
        "one value to the probe fit, the rollout and both filtering "
        "diagnostics, and a second home for them here would let the nine "
        "records carry this script's stale defaults")


def test_main_hands_run_job_the_out_dir_string_argparse_produced(
        tmp_path, episode_dir, monkeypatch):
    """argparse hands over a `str`, and every consumer accepts one. Keeping it
    a `str` end to end is what makes that tolerance real rather than assumed."""
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = str(tmp_path / "study")

    run_study.main(["--data", str(episode_dir), "--out", out,
                    "--arms", "random_vit", "--seeds", "1"])

    assert calls[0]["out_dir"] == out
    assert isinstance(calls[0]["out_dir"], str)
    assert job_record_path(out, StudyJob("random_vit", 1)).is_file()


def test_main_skips_the_cells_that_already_have_a_record(
        tmp_path, episode_dir, monkeypatch, capsys):
    """End to end: the second run of the same command must cost nothing."""
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = str(tmp_path / "study")
    argv = ["--data", str(episode_dir), "--out", out]

    assert run_study.main(argv) == 0
    assert len(calls) == 9
    calls.clear()

    assert run_study.main(argv) == 0
    assert calls == [], "a completed study was run again"
    assert "0 job(s) pending" in capsys.readouterr().out


@pytest.mark.parametrize("error", [
    RuntimeError("cuda out of memory on this cell"),
    FileNotFoundError("no cached features for frozen_ssl on this box"),
    ValueError("checkpoint is arm='pixel_ae', not this job's"),
    KeyError("privileged"),
], ids=lambda e: type(e).__name__)
def test_a_failing_cell_does_not_abort_the_other_eight(
        tmp_path, episode_dir, monkeypatch, capsys, error):
    """THE POLICY, tested. Nobody is watching a ~13.5-hour unattended run, and the
    realistic failures are per-cell; aborting throws away every remaining cell
    to punish one. The run continues, the exit status is non-zero, and the
    traceback is in the log.

    PARAMETRISED OVER THE EXCEPTION TYPE because `except Exception` is the
    width the policy needs and only `RuntimeError` used to be exercised, so
    narrowing it to the types we happen to have seen survived. The docstring's
    own headline example -- a missing feature cache for one arm -- raises
    `FileNotFoundError`, an `OSError`: under `except (RuntimeError,
    ValueError)` that arm's three cells would kill the run this policy exists
    to protect.
    """
    victim = StudyJob("random_vit", 1)

    def behaviour(job):
        if job == victim:
            raise error
        return None

    fake, calls = _spy(behaviour)
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"

    status = run_study.main(["--data", str(episode_dir), "--out", str(out)])

    assert status == run_study.EXIT_JOB_FAILED == 1
    assert len(calls) == 9, "the run stopped at the failing cell"
    assert [c["job"] for c in calls][-1] == StudyJob("random_vit", 2), (
        "the cell after the failing one was never reached")
    printed = capsys.readouterr().out
    assert type(error).__name__ in printed, "no traceback logged"
    assert "8/9 job(s) completed" in printed

    # EXACTLY the cells that failed, and not one more. The morning's first
    # question is which cells to retry, and a line naming all nine -- which is
    # what iterating `jobs` here instead of `failed` prints -- answers it
    # wrongly and sends the operator back for another 13.5 hours.
    listed = printed.split("FAILED:")[-1].split(" -- ")[0].strip()
    assert listed == f"{victim.arm}/s{victim.seed}"


def test_the_progress_counter_numbers_the_cells_from_one(
        tmp_path, episode_dir, monkeypatch, capsys):
    """`[3/9]` in the log is how the morning tells how far an interrupted run
    got. `enumerate(jobs)` without a start counts 0..8, so the last cell reads
    `[8/9]` and a finished run looks like it stopped one short."""
    fake, _ = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)

    run_study.main(["--data", str(episode_dir),
                    "--out", str(tmp_path / "study")])

    printed = capsys.readouterr().out
    assert "[1/9]" in printed and "[9/9]" in printed
    assert "[0/9]" not in printed


def test_a_failing_cell_leaves_no_record_so_a_rerun_retries_exactly_it(
        tmp_path, episode_dir, monkeypatch):
    """Continuing past a failure is only safe because nothing is written for
    it. A failure record at the cell's path would make the miss permanent."""
    victim = StudyJob("frozen_ssl", 2)

    def behaviour(job):
        if job == victim:
            raise RuntimeError("boom")
        return None

    fake, _ = _spy(behaviour)
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"

    run_study.main(["--data", str(episode_dir), "--out", str(out)])

    assert not job_record_path(out, victim).exists()
    assert run_study.pending_jobs(out, ARMS, SEEDS) == [victim]


def test_a_keyboard_interrupt_stops_the_whole_run(
        tmp_path, episode_dir, monkeypatch):
    """A human pressing Ctrl-C means the run, not this cell. `except Exception`
    rather than `except BaseException` is what keeps that true."""
    def behaviour(job):
        raise KeyboardInterrupt

    fake, calls = _spy(behaviour)
    monkeypatch.setattr(run_study, "run_job", fake)

    with pytest.raises(KeyboardInterrupt):
        run_study.main(["--data", str(episode_dir),
                        "--out", str(tmp_path / "study")])
    assert len(calls) == 1


@pytest.mark.parametrize("overrides, claimed", [
    ({"arm": "pixel_ae"}, "arm='pixel_ae' seed=1"),          # wrong arm, RIGHT seed
    ({"seed": CLI_SEED}, f"arm='frozen_ssl' seed={CLI_SEED}"),  # wrong seed
], ids=["wrong_arm", "wrong_seed"])
def test_main_quarantines_a_record_that_names_another_cell(
        tmp_path, episode_dir, monkeypatch, capsys, overrides, claimed):
    """The worst outcome the study has, and it used to be LEFT ON DISK.

    The driver printed MISLABELLED, counted the cell failed, exited 1 -- and
    left `result_frozen_ssl_seed1.json` sitting at the cell's own path with
    `arm='pixel_ae'` inside it, exactly where the aggregation's glob finds it. The
    driver's own resume is safe, but an operator who reads the log in the
    morning, fixes the cause and runs the AGGREGATION without re-running the
    driver gets two `pixel_ae` cells and no `frozen_ssl`/s1: one arm's numbers
    reported under another's name, which is the thing this branch's own comment
    calls the worst outcome the study has.

    Each half of `record_names_the_job` is exercised alone, and the message is
    pinned to the RECORD's claim rather than the job's: echoing the job on both
    sides prints "the record says arm='frozen_ssl' ..., not 'frozen_ssl'/1",
    a self-contradictory sentence that destroys the one fact needed to work out
    whose numbers these are.
    """
    liar = StudyJob("frozen_ssl", 1)

    def behaviour(job):
        return overrides if job == liar else None

    fake, _ = _spy(behaviour)
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"

    status = run_study.main(["--data", str(episode_dir), "--out", str(out)])

    assert status == run_study.EXIT_JOB_FAILED
    printed = capsys.readouterr().out
    assert "MISLABELLED" in printed
    assert claimed in printed, (
        "the message must say what the RECORD claims, not repeat the cell we "
        "asked for on both sides of the sentence")
    assert f"{liar.arm}/s{liar.seed}" in printed.split("FAILED:")[-1]
    assert "8/9 job(s) completed" in printed

    # The file is off the aggregation's path...
    record_path = job_record_path(out, liar)
    assert not record_path.exists(), (
        "the mislabelled record is still at the cell's own path, where the "
        "aggregation reads it as this cell's result")
    # ...and still on disk, under a name that is not a record's.
    quarantined = record_path.with_name(
        record_path.name + run_study.MISLABELLED_SUFFIX)
    assert quarantined.is_file(), "the record was destroyed rather than moved"
    assert json.loads(quarantined.read_text())["arm"] == overrides.get(
        "arm", liar.arm), "the quarantined file is not the record we saw"
    assert str(quarantined) in printed, (
        "the log must say where the file went, or the operator cannot find it")

    # The other eight cells still ran: one bad cell does not abort the run.
    assert len(list(out.glob("result_*.json"))) == 8


def test_the_quarantine_suffix_cannot_be_read_as_a_record(tmp_path):
    """The same argument `LOCK_NAME` makes, for the same directory.

    The aggregation globs `--out`. A quarantined record whose name still ends
    in `.json` is quarantined in name only, and a name that still starts with
    `result_` is picked up by the narrower glob too.
    """
    assert not run_study.MISLABELLED_SUFFIX.endswith(".json")

    record = job_record_path(tmp_path, DONE_JOB)
    record.write_text('{"arm": "pixel_ae"}')
    assert [p.name for p in tmp_path.glob("result_*.json")] == [record.name], (
        "the control: the glob really does find a record at this path, so an "
        "empty result below means the move worked and not that nothing was "
        "ever there")

    moved = run_study.quarantine_record(record)
    assert not moved.name.endswith(".json")
    assert list(tmp_path.glob("result_*.json")) == []
    assert list(tmp_path.glob("*.json")) == [], (
        "the aggregation globs the whole directory; a quarantined record that "
        "still ends in .json is quarantined in name only")


def test_quarantining_twice_does_not_overwrite_the_first_file(tmp_path):
    """Two mislabelled records for one cell across two runs are two facts.

    Overwriting would leave the operator with the second and no sign of the
    first, and each is the output of a real ~1.5-hour training run.
    """
    path = job_record_path(tmp_path, DONE_JOB)
    path.write_text('{"arm": "first"}')
    first = run_study.quarantine_record(path)
    path.write_text('{"arm": "second"}')
    second = run_study.quarantine_record(path)

    assert first != second
    assert json.loads(first.read_text())["arm"] == "first"
    assert json.loads(second.read_text())["arm"] == "second"
    assert not path.exists()


def test_quarantining_a_file_that_is_not_there_does_not_kill_the_run(tmp_path):
    """This runs inside the per-cell loop, after the record is on disk, and
    must not become the thing that ends an unattended run. `None` is the
    signal that the message has to tell the operator to move it by hand."""
    assert run_study.quarantine_record(tmp_path / "no_such_record.json") is None


def test_main_stops_before_the_first_cell_when_there_are_no_episodes(
        tmp_path, monkeypatch, capsys):
    """With no data every one of the nine cells fails identically inside
    `episode_split`, hours apart. Fail on the first second instead."""
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    empty = tmp_path / "no_data"
    empty.mkdir()

    status = run_study.main(["--data", str(empty), "--out", str(tmp_path / "s")])

    assert status == run_study.EXIT_NO_DATA == 5
    assert status != ARGPARSE_USAGE_STATUS, (
        "an empty --data used to exit 2, the same status argparse gives a "
        "misspelt flag, so a wrapper could not tell the box having no episodes "
        "from the command line being wrong")
    assert calls == []
    assert "no episodes" in capsys.readouterr().out


def test_main_does_not_need_the_data_when_nothing_is_pending(
        tmp_path, monkeypatch):
    """A finished study must not depend on the episodes still being on the box,
    and `ReplayBuffer` creates the directory merely by being asked."""
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"
    for arm in ARMS:
        for seed in SEEDS:
            _write_matching(out, StudyJob(arm, seed))
    gone = tmp_path / "data_left_on_the_other_box"

    assert run_study.main(["--data", str(gone), "--out", str(out),
                           *MATCHING_ARGV]) == 0
    assert calls == []
    assert not gone.exists(), "the driver created the data directory"


def test_main_refuses_an_arm_the_study_does_not_have(tmp_path, episode_dir):
    """`--arms cnn2` must not silently run zero cells overnight."""
    with pytest.raises(SystemExit) as excinfo:
        run_study.main(["--data", str(episode_dir), "--out", str(tmp_path),
                        "--arms", "cnn2"])
    assert excinfo.value.code == ARGPARSE_USAGE_STATUS


def test_main_refuses_the_retired_pixel_arm_by_name(tmp_path, episode_dir,
                                                    monkeypatch):
    """`cnn` is a KIND `get_config` still builds, so the only thing that
    keeps the study from training a cell of the retired arm is `--arms`
    taking `choices=ARMS` rather than `choices=KINDS`. A driver that accepted
    it would write `result_cnn_seed0.json` into the M3c directory and the
    aggregation would read a fourth arm. `run_job` is stubbed so that if the
    refusal is missing this fails on the exit status, not by training."""
    from mbfps.utils.config import KINDS, get_config

    assert "cnn" in KINDS and get_config("cnn").arm == "cnn", (
        "the precondition: the name IS buildable, so only argparse refuses it")
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    with pytest.raises(SystemExit) as excinfo:
        run_study.main(["--data", str(episode_dir), "--out", str(tmp_path),
                        "--arms", "cnn", "--seeds", "0"])
    assert excinfo.value.code == ARGPARSE_USAGE_STATUS
    assert calls == [], "a cell of the retired arm was trained"
    assert tuple(run_study._parser().parse_args([]).arms) == ARMS


@pytest.mark.parametrize("argv", [
    ["--seeds", "10"],
    ["--seeds", "0", "10"],
    ["--arms"],
    ["--seeds"],
], ids=["unknown_seed", "one_unknown_seed", "empty_arms", "empty_seeds"])
def test_main_refuses_a_selection_that_would_run_the_wrong_cells(
        tmp_path, episode_dir, argv):
    """A shell typo must not buy a night of a rented box.

    `--seeds` was unrestricted while `--arms` was not, so `--seeds 10` trained
    a tenth cell the aggregation -- which demands exactly nine -- would never
    look at, and exited 0. And `nargs="*"` accepts the flag with no values at
    all: `--arms` alone selected nothing, printed "0 job(s) pending" and exited
    0, which is indistinguishable from a finished study to any wrapper.
    """
    with pytest.raises(SystemExit) as excinfo:
        run_study.main(["--data", str(episode_dir), "--out", str(tmp_path),
                        *argv])
    assert excinfo.value.code == ARGPARSE_USAGE_STATUS
    assert excinfo.value.code not in set(EXPECTED_EXIT_STATUS.values()), (
        "argparse's usage status must stay clear of the study's own, or a "
        "shell typo and a real failure are the same event to any wrapper")


@pytest.mark.parametrize("argv, expected, floor", [
    (["--steps", "200"], "--steps 200", run_study.MIN_STEPS),
    (["--steps", str(run_study.MIN_STEPS - 1)],
     f"--steps {run_study.MIN_STEPS - 1}", run_study.MIN_STEPS),
    (["--seq-len", "6"], "--seq-len 6", run_study.MIN_SEQ_LEN),
    (["--seq-len", str(run_study.MIN_SEQ_LEN - 1)],
     f"--seq-len {run_study.MIN_SEQ_LEN - 1}", run_study.MIN_SEQ_LEN),
], ids=["steps_typo", "steps_just_under", "seq_len_typo", "seq_len_just_under"])
def test_a_mistyped_training_length_is_refused_at_the_command_line(
        tmp_path, episode_dir, argv, expected, floor, capsys):
    """THE SMOKE-RUN TRAP IN A NEW COAT, and the config guard cannot help.

    `--arms` and `--seeds` are choice-restricted with the stated rationale that
    a shell typo must not buy a night of a rented box; the two flags that
    decide HOW MUCH training happens were unrestricted. `--steps 200` for
    `--steps 20000` writes nine complete records at a smoke configuration and
    exits 0 -- and `record_config_mismatch` cannot catch it, because the
    records agree with the flags they were made with. The corrective re-run at
    20000 is then REFUSED with `EXIT_CONFIG_MISMATCH` until all nine records
    are deleted by hand. The typo guard stopped exactly where a typo is most
    expensive.

    EACH FLAG ALONE, AND EACH JUST UNDER ITS OWN FLOOR: a check that compared
    `steps` against `MIN_SEQ_LEN`, or that tested only one of the two flags,
    would pass for half of these.

    THE ASSERTIONS ARE MADE AGAINST THE REFUSAL ALONE, NOT AGAINST `stderr`.
    `parser.error` writes argparse's own usage banner to stderr before the
    message, and that banner lists every flag the parser has -- `--allow-short`
    included. `assert "--allow-short" in capsys.readouterr().err` was therefore
    satisfied by argparse whether or not the refusal mentioned the waiver at
    all: the one sentence telling the operator how to authorise a deliberate
    short run could be deleted with a green suite, leaving a floor with no
    documented way over it in the message that announces it. Both the string
    and the output were right; the assertion still could not fail. So the
    banner is split off and the guards below read only what this module wrote.
    """
    with pytest.raises(SystemExit) as excinfo:
        run_study.main(["--data", str(episode_dir), "--out", str(tmp_path),
                        *argv])
    assert excinfo.value.code == ARGPARSE_USAGE_STATUS
    assert excinfo.value.code not in set(EXPECTED_EXIT_STATUS.values())
    banner, separator, refusal = capsys.readouterr().err.partition("error: ")
    assert separator and refusal.strip(), (
        "argparse's error line was not found, so the split below would leave "
        "an empty string that every `in` assertion is vacuously true about")
    assert "--allow-short" in banner, (
        "THIS SPLIT IS WHY THIS TEST CAN FAIL. argparse's usage banner names "
        "every flag the parser has, so an assertion against the whole of "
        "stderr is answered by the banner rather than by the refusal")

    assert refusal.startswith(expected), (
        "the refusal must OPEN with the flag and the value that was typed. "
        "A bare `in` is answered by the typo example the message goes on to "
        "give, which spells out `--steps 200` and `--seq-len 6` literally, so "
        "two of these four cases could not fail on the interpolation at all")
    assert f"floor of {floor}" in refusal, (
        "the refusal must cite the floor of the flag it is refusing; citing "
        "the other one tells an operator who typed --seq-len 6 that the "
        "minimum is 5000")
    assert "Pass --allow-short if the short run is deliberate" in refusal, (
        "a deliberate short run has to be told how to say so, or the floor is "
        "just a wall")


@pytest.mark.parametrize("argv", [
    ["--steps", "0"], ["--steps", "-5"],
    ["--seq-len", "0"], ["--seq-len", "-3"],
], ids=["steps_zero", "steps_negative", "seq_len_zero", "seq_len_negative"])
def test_a_non_positive_training_length_is_refused_even_with_allow_short(
        tmp_path, episode_dir, argv):
    """`--allow-short` waives the FLOOR, not arithmetic.

    `--steps 0` trains nothing and `--seq-len 0` has no meaning, so there is no
    deliberate run they could be. All four were accepted, wrote nine complete
    records and exited 0.
    """
    with pytest.raises(SystemExit) as excinfo:
        run_study.main(["--data", str(episode_dir), "--out", str(tmp_path),
                        "--allow-short", *argv])
    assert excinfo.value.code == ARGPARSE_USAGE_STATUS


def test_a_deliberate_short_run_is_allowed_when_it_says_so(
        tmp_path, episode_dir, monkeypatch):
    """THE CONTROL. Without it a driver that refused every configuration --
    the study's own 20000/64 included -- would pass all eight cases above.

    The plan's Step 1 smoke run is a real thing, and the floor exists to tell
    a deliberate one from a typo, not to forbid short runs.
    """
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)

    assert run_study.main([
        "--data", str(episode_dir), "--out", str(tmp_path / "smoke"),
        "--steps", str(SMOKE_STEPS), "--seq-len", "1", "--allow-short",
        "--arms", "pixel_ae", "--seeds", "0"]) == 0
    assert calls[0]["kwargs"]["steps"] == SMOKE_STEPS
    assert calls[0]["kwargs"]["seq_len"] == 1


@pytest.mark.parametrize("steps, seq_len", [
    (run_study.MIN_STEPS, run_study.MIN_SEQ_LEN + 1),
    (run_study.MIN_STEPS + 1, run_study.MIN_SEQ_LEN),
], ids=["steps_exactly_at_its_floor", "seq_len_exactly_at_its_floor"])
def test_a_run_exactly_at_a_floor_is_accepted_without_allow_short(
        steps, seq_len):
    """THE ACCEPT SIDE OF THE BOUNDARY, WHICH NOTHING PINNED.

    The refusals above are parametrised at `MIN_STEPS - 1` and
    `MIN_SEQ_LEN - 1`, and both of those are refused by `value < floor` and by
    `value <= floor` alike -- so the off-by-one survived the whole suite. Under
    it, `--steps 5000` and `--seq-len 32` are refused as typos: the operator on
    the rented box is told the minimum is 5000 while typing exactly 5000, and
    the only way out of the message is `--allow-short`, which is the flag that
    says the run is NOT what the floor is for. A floor that refuses its own
    value is a floor one higher that nobody wrote down.

    Each flag sits on its own floor with the other clear of its own, so a
    single comparison covering only one of them cannot pass for both.
    """
    assert run_study.short_config_complaint(
        steps, seq_len, allow_short=False) is None, (
        "a run exactly at the floor was refused; the comparison has become "
        "`<=` and the real floor is one above the documented one")


def test_the_allow_short_help_names_both_floors():
    """`--help` IS WHAT AN OPERATOR READS BEFORE SPENDING A NIGHT'S RENT.

    Nothing pinned this text, so collapsing it to "permit a short run" survived
    the suite -- and this is the only place `--help` names either floor. The
    numbers appear nowhere else in the help: `--steps` and `--seq-len` are
    declared with no `help=` at all, so their defaults are not shown either.
    Someone deciding what to type has the two floors here or nowhere.

    EACH FLOOR IS CHECKED BESIDE ITS OWN FLAG, so a text that swapped them --
    "--steps below 32 or --seq-len below 5000" -- fails too; that reading sends
    a 20000-step run through `--allow-short` and a 6-token one without it.
    """
    help_text = run_study._parser().format_help()
    # The LAST occurrence: the first is in argparse's own usage banner, which
    # lists every flag and no help text at all. What follows the last one is
    # this option's help and nothing else, since `--allow-short` is added last.
    entry = " ".join(help_text.rpartition("--allow-short")[2].split())
    assert entry, "the --allow-short entry was not found in --help"

    assert f"--steps below {run_study.MIN_STEPS}" in entry, (
        "--help must name the --steps floor; it is named nowhere else there")
    assert f"--seq-len below {run_study.MIN_SEQ_LEN}" in entry, (
        "--help must name the --seq-len floor; likewise")
    assert run_study.MIN_STEPS != run_study.MIN_SEQ_LEN, (
        "the two floors are equal, so the pairing assertions above can no "
        "longer tell a swapped help text from a correct one")


def test_the_studys_own_configuration_is_above_both_floors():
    """A floor raised past the study itself would refuse the real run.

    The defaults are what the plan's Step 3 command uses with no flags at all,
    so a `MIN_STEPS` above 20000 turns the ~13.5-hour run into a usage error.
    """
    defaults = vars(run_study._parser().parse_args([]))
    assert defaults["steps"] >= run_study.MIN_STEPS > 0
    assert defaults["seq_len"] >= run_study.MIN_SEQ_LEN > 0
    assert not defaults["allow_short"], (
        "the waiver must be off by default, or the floors are decoration")
    # And the floors really do bite on a dropped digit, which is the typo they
    # exist for: 20000 -> 2000 and 64 -> 6.
    assert defaults["steps"] // 10 < run_study.MIN_STEPS
    assert defaults["seq_len"] // 10 < run_study.MIN_SEQ_LEN


def test_the_script_exits_with_mains_status_when_run_as_a_script(tmp_path):
    """AS A SUBPROCESS, because nothing else here executes the `__main__` block.

    This module is loaded by `importlib`, so `if __name__ == "__main__"` never
    runs in the suite and `raise SystemExit(main())` could be replaced by a
    bare `main()` with everything green. The script would then always exit 0:
    a failed overnight run is invisible to every wrapper, every `&&` and every
    CI step that checks a status.
    """
    empty = tmp_path / "no_data"
    empty.mkdir()
    src = str(Path(study.__file__).resolve().parents[2])
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join(
               p for p in (src, os.environ.get("PYTHONPATH", "")) if p)}

    completed = subprocess.run(
        [sys.executable, run_study.__file__,
         "--data", str(empty), "--out", str(tmp_path / "s")],
        capture_output=True, text=True, env=env, timeout=600)

    assert completed.returncode == run_study.EXIT_NO_DATA == 5, (
        f"stdout={completed.stdout!r}\nstderr={completed.stderr[-3000:]!r}")
    assert completed.returncode != ARGPARSE_USAGE_STATUS, (
        "a status argparse also uses could be produced by the interpreter "
        "rejecting the command line, which would tell us nothing about "
        "whether `main`'s return value reached the shell at all")
    assert "no episodes" in completed.stdout


# ---------------------------------------------------------------------------
# "done" is not "done at some other configuration"
# ---------------------------------------------------------------------------

def test_the_driver_demands_run_jobs_own_context_and_horizon():
    """One home for the evaluation protocol.

    `main` has no --context/--horizon flags, so the driver has to know what
    `run_job` will default to in order to say whether a finished record was
    produced at the configuration being asked for. It reads them off
    `run_job`'s signature rather than restating them: a literal here would go
    stale the day those defaults move, and the staleness check would then
    reject the whole study over a number nobody changed.
    """
    assert run_study.PROTOCOL_CONTEXT == PROTOCOL_CONTEXT
    assert run_study.PROTOCOL_HORIZON == PROTOCOL_HORIZON
    assert tuple(run_study.CONFIG_KEYS) == (
        "steps", "seq_len", "context", "horizon")
    assert run_study.requested_config(CLI_STEPS, CLI_SEQ_LEN) == {
        "steps": CLI_STEPS, "seq_len": CLI_SEQ_LEN,
        "context": PROTOCOL_CONTEXT, "horizon": PROTOCOL_HORIZON}


def test_record_config_mismatch_names_the_recorded_and_requested_value():
    """All four keys wrong at once, so the mapping's shape is pinned; the
    per-key tests below are what exercise each one alone."""
    config = run_study.requested_config(CLI_STEPS, CLI_SEQ_LEN)
    assert run_study.record_config_mismatch(_record_body(DONE_JOB), config) == {
        "steps": (RECORD_STEPS, CLI_STEPS),
        "seq_len": (RECORD_SEQ_LEN, CLI_SEQ_LEN),
        "context": (RECORD_CONTEXT, PROTOCOL_CONTEXT),
        "horizon": (RECORD_HORIZON, PROTOCOL_HORIZON),
    }
    matching = _record_body(DONE_JOB, steps=CLI_STEPS, seq_len=CLI_SEQ_LEN,
                            context=PROTOCOL_CONTEXT, horizon=PROTOCOL_HORIZON)
    assert run_study.record_config_mismatch(matching, config) == {}, (
        "the control: a check that called everything a mismatch would pass "
        "every assertion above and refuse to run the study at all")


@pytest.mark.parametrize("key", sorted(CONFIG_MISMATCHES))
def test_a_record_from_another_configuration_refuses_rather_than_skipping(
        tmp_path, episode_dir, monkeypatch, capsys, key):
    """Each config field ALONE: the other three agree with what was requested.

    With all four wrong together, a `CONFIG_KEYS` short of any one of them
    would still refuse and the deletion would survive. This is the same
    compound-condition hole that let a mislabelled-record check pass with
    either half of its `and` removed.
    """
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"
    _write_matching(out, DONE_JOB, **{key: CONFIG_MISMATCHES[key]})

    status = run_study.main(["--data", str(episode_dir), "--out", str(out),
                             *MATCHING_ARGV])

    assert status == run_study.EXIT_CONFIG_MISMATCH == 3
    assert calls == [], "a cell was trained despite the refusal"
    printed = capsys.readouterr().out
    assert "CONFIGURATION MISMATCH: 1 finished record(s)" in printed, (
        "one stale record, and the banner says so; the nine-record test below "
        "is the other direction, and between them the count cannot be a "
        "literal")
    assert f"{DONE_JOB.arm}/s{DONE_JOB.seed}" in printed
    assert f"{key}: recorded {CONFIG_MISMATCHES[key]!r} != requested" in printed


def test_nine_smoke_records_refuse_instead_of_reporting_a_finished_study(
        tmp_path, monkeypatch, capsys):
    """THE TRAP THIS EXISTS FOR, and the plan's own Task 7 Step 1 walks into it.

    `record_is_complete` validated the arm and the seed and nothing about the
    configuration, so nine records from a short smoke run made the real
    ~13.5-hour study a no-op: every cell read done, the driver exited 0 having trained
    nothing, and the gate would have been computed from smoke-run models.

    It REFUSES rather than treating them as pending. Re-running would fix the
    smoke trap and create a worse one in the other direction: after the real
    study finishes, a quick 100-step check into the same --out would leave the
    driver deciding all nine records are stale and OVERWRITING 13.5 hours of the
    study's only artifact, unrecoverably. Refusing costs at most one night of a
    rented box, says so on the first screen with a non-zero status, and is
    undone by one flag or one `rm`.
    """
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"
    for arm in ARMS:
        for seed in SEEDS:
            _write_matching(out, StudyJob(arm, seed), steps=SMOKE_STEPS)
    gone = tmp_path / "data_not_mounted_yet"

    status = run_study.main(["--data", str(gone), "--out", str(out),
                             *MATCHING_ARGV])

    assert status == run_study.EXIT_CONFIG_MISMATCH
    assert calls == []
    printed = capsys.readouterr().out
    assert printed.splitlines()[0].endswith("0 job(s) pending: none"), (
        "the refusal must come AFTER the header the log's first line is read "
        "from, and instead of the 'nothing to do' that used to follow it")
    assert "nothing to do" not in printed, (
        "the check has to run before the early return, or nine smoke records "
        "still exit 0")
    assert printed.count(f"recorded {SMOKE_STEPS!r} != requested") == 9
    assert (f"CONFIGURATION MISMATCH: {len(ARMS) * len(SEEDS)} "
            "finished record(s)") in printed, (
        "the banner's count was the one part of the refusal nothing pinned, so "
        "it could say 1 while listing nine; it is the number an operator reads "
        "before deciding whether one file or the whole study is stale")
    assert "Remedy" in printed
    assert not gone.exists(), (
        "the refusal came after `ReplayBuffer` was asked for the episodes; it "
        "has to fail in the first second, not the fourteenth hour")
    assert not (out / run_study.LOCK_NAME).exists()


def test_a_matching_rerun_still_skips_the_cells_that_are_finished(
        tmp_path, episode_dir, monkeypatch, capsys):
    """THE OTHER HALF, and the driver's whole point.

    A guard that called every record stale would refuse every resume, which
    costs the study the 13.5 hours the resume exists to save.
    """
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"
    for job in (DONE_JOB, CORRUPT_JOB):
        _write_matching(out, job)

    status = run_study.main(["--data", str(episode_dir), "--out", str(out),
                             *MATCHING_ARGV])

    assert status == 0
    assert {c["job"] for c in calls}.isdisjoint({DONE_JOB, CORRUPT_JOB})
    assert len(calls) == 7
    assert "7/7 job(s) completed" in capsys.readouterr().out


def test_an_incomplete_record_is_pending_rather_than_a_refusal(
        tmp_path, episode_dir, monkeypatch):
    """A corrupt file has no configuration worth comparing, and it is going to
    be re-run anyway. Calling it stale would turn one damaged file into a
    refusal to do anything at all."""
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"
    out.mkdir()
    damaged = _sanitised(INCOMPLETE_JOB, steps=SMOKE_STEPS)
    del damaged["position"]
    job_record_path(out, INCOMPLETE_JOB).write_text(json.dumps(damaged))

    assert run_study.main(["--data", str(episode_dir), "--out", str(out),
                           *MATCHING_ARGV]) == 0
    assert len(calls) == 9


def test_only_the_selected_arms_are_checked_for_staleness(
        tmp_path, episode_dir, monkeypatch, capsys):
    """A stale record for an arm this invocation is not running must not block
    it -- otherwise one old file makes every future `--arms` run impossible.

    Half of a compound contract. `--seeds` is left at all three here, so this
    test says nothing about the seed argument; that is the test below.
    """
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"
    _write_matching(out, DONE_JOB, steps=SMOKE_STEPS)   # pixel_ae/s1
    assert DONE_JOB.arm == "pixel_ae"

    status = run_study.main(["--data", str(episode_dir), "--out", str(out),
                             "--arms", "frozen_ssl", *MATCHING_ARGV])

    assert status == run_study.EXIT_OK
    assert len(calls) == 3
    assert "CONFIGURATION MISMATCH" not in capsys.readouterr().out

    # THE CONTROL: the same directory with the record's own arm selected. The
    # record really is stale, so a run that does look at it refuses. Without
    # this, every assertion above would pass just as well against a staleness
    # check that never found anything at all.
    calls.clear()
    assert run_study.main(["--data", str(episode_dir), "--out", str(out),
                           "--arms", DONE_JOB.arm, *MATCHING_ARGV]
                          ) == run_study.EXIT_CONFIG_MISMATCH
    assert calls == []


def test_only_the_selected_seeds_are_checked_for_staleness(
        tmp_path, episode_dir, monkeypatch, capsys):
    """THE OTHER HALF OF THE SAME CONTRACT, EXERCISED ALONE.

    The arms test above varies `--arms` and leaves `--seeds` at all three, so
    it could not see `stale_records` ignoring the seeds it was passed:
    replacing its inner loop with a loop over the module-level `SEEDS` survived
    the whole suite. `pending_jobs` pins both halves and its equivalent
    mutation dies; this is the missing half of that pair.

    What it costs is the targeted re-run. One cell died on the box, its
    neighbour holds an old record from a different configuration, and `--seeds
    1` -- the one-cell repair the resume exists to make possible -- refuses to
    do anything at all because of a cell it was never asked to touch.

    THE STALE RECORD'S ARM IS ONE THIS INVOCATION IS RUNNING, so nothing but
    the seed selection can be what excludes it.
    """
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"
    stale = StudyJob("frozen_ssl", 2)
    _write_matching(out, stale, steps=SMOKE_STEPS)
    argv = ["--data", str(episode_dir), "--out", str(out),
            "--arms", stale.arm, *MATCHING_ARGV]
    selected = ("0", "1")
    assert str(stale.seed) not in selected, (
        "the stale record's seed must be the one left out, or this test is "
        "asking whether a selected cell blocks the run")

    status = run_study.main([*argv, "--seeds", *selected])

    assert status == run_study.EXIT_OK
    assert [c["job"] for c in calls] == [StudyJob(stale.arm, 0),
                                         StudyJob(stale.arm, 1)]
    assert "CONFIGURATION MISMATCH" not in capsys.readouterr().out

    # THE CONTROL, same directory, only the seed selection widened to include
    # the stale cell: it really is stale, and a run that selects it refuses.
    calls.clear()
    assert run_study.main([*argv, "--seeds", *selected, str(stale.seed)]
                          ) == run_study.EXIT_CONFIG_MISMATCH
    assert calls == [], "a cell was trained despite the refusal"
    assert f"{stale.arm}/s{stale.seed}" in capsys.readouterr().out


def test_a_repeated_arm_or_seed_is_reported_stale_only_once(tmp_path):
    """`--arms frozen_ssl frozen_ssl` must not list one cell twice.

    Cosmetic beside the seeds hole above, and it is the same asymmetry:
    `pending_jobs`' dedup is tested and its deletion dies, `stale_records`'
    was not and its deletion survived. A refusal that names nine cells as
    eighteen is a refusal an operator has to count twice at 8am.
    """
    job = StudyJob("frozen_ssl", 0)
    _write_matching(tmp_path, job, steps=SMOKE_STEPS)
    config = run_study.requested_config(CLI_STEPS, CLI_SEQ_LEN)

    stale = run_study.stale_records(
        tmp_path, (job.arm, job.arm), (job.seed, job.seed), config)

    assert [found for found, _ in stale] == [job]
    assert stale[0][1] == {"steps": (SMOKE_STEPS, CLI_STEPS)}, (
        "only `steps` was made wrong, so a mismatch naming any other field "
        "means the fixture, not the driver, decided this")
    report = run_study.stale_report(stale, tmp_path)
    assert report.count(f"{job.arm}/s{job.seed}") == 1
    assert "CONFIGURATION MISMATCH: 1 finished record(s)" in report


def test_the_refusal_lists_a_cells_mismatched_fields_in_a_stable_order(
        tmp_path):
    """Nine cells times four fields, read by a human at 8am.

    The order must not be whichever order `CONFIG_KEYS` happens to declare, or
    the same four disagreements read differently between two runs of the same
    command.
    """
    assert sorted(run_study.CONFIG_KEYS) != list(run_study.CONFIG_KEYS), (
        "`CONFIG_KEYS` is now declared in alphabetical order, so this test can "
        "no longer tell a sorted report from the insertion-ordered one and "
        "dropping the sort would survive it")

    job = DONE_JOB
    _write_matching(tmp_path, job, **CONFIG_MISMATCHES)   # all four disagree
    config = run_study.requested_config(CLI_STEPS, CLI_SEQ_LEN)

    stale = run_study.stale_records(tmp_path, (job.arm,), (job.seed,), config)
    report = run_study.stale_report(stale, tmp_path)

    detail, = [line for line in report.splitlines()
               if f"{job.arm}/s{job.seed}" in line]
    assert detail == "  {}/s{}  {}".format(job.arm, job.seed, "  ".join(
        f"{key}: recorded {CONFIG_MISMATCHES[key]!r} != requested "
        f"{config[key]!r}"
        for key in sorted(CONFIG_MISMATCHES)))


def test_the_refusal_says_delete_from_out_and_names_which_directory(
        tmp_path, episode_dir, monkeypatch, capsys):
    """THE LOG USED TO TELL THE OPERATOR TO DELETE THE DATASET.

    `print(stale_report(stale, args.out))` with `args.data` in its place
    survived the whole suite, and the banner then names the EPISODE directory
    as the place holding the stale records while the Remedy line says to delete
    those records. An operator following that at 8am deletes
    `data/my_way_home`, which is not reproducible from anything in the
    repository and takes the study with it. The directory could also be dropped
    from the banner entirely, which leaves the operator told to delete files
    and not told where they are -- the same instruction, aimed at nothing.

    So `--out` is named, and named ON THE REMEDY LINE ITSELF rather than only
    in the banner four lines up, and `--data` is named as the thing not to
    touch. The two directories in this test are siblings, so neither path is a
    substring of the other and "names --out" cannot pass by naming --data.
    """
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"
    _write_matching(out, DONE_JOB, steps=SMOKE_STEPS)
    assert str(out) not in str(episode_dir) and str(episode_dir) not in str(out)

    status = run_study.main(["--data", str(episode_dir), "--out", str(out),
                             *MATCHING_ARGV])

    assert status == run_study.EXIT_CONFIG_MISMATCH
    assert calls == []
    printed = capsys.readouterr().out
    assert str(episode_dir) not in printed, (
        "the refusal named the episode directory as the place holding the "
        "stale records, and its own remedy says to delete them")
    remedy, = [line for line in printed.splitlines()
               if line.startswith("Remedy:")]
    assert str(out) in remedy, (
        "the line that tells the operator to delete files must say which "
        "directory to delete them from")
    assert "--out" in remedy and "NOT from --data" in remedy
    banner = printed.splitlines()[1]
    assert str(out) in banner and "--out" in banner


# ---------------------------------------------------------------------------
# one driver per --out
# ---------------------------------------------------------------------------

def test_a_second_driver_on_the_same_out_refuses_to_start(
        tmp_path, episode_dir, monkeypatch, capsys):
    """Two drivers pointed at one --out both saw all nine cells pending and
    both ran all nine: 27 GPU-hours instead of 13.5, racing on the same record
    and checkpoint paths, with no warning in either log.

    THE PATH IS DEMANDED ON BOTH LINES SEPARATELY. The refusal interpolates the
    claim file twice -- once to say what is held, once to say what to delete --
    so `assert str(held) in printed` is answered by whichever of the two
    survives, and either could be dropped alone with a green suite. They do
    different jobs: the first identifies the directory the operator has been
    refused, the second is the remedy they act on at 2am, and a remedy reading
    "delete it and start again" with no `it` is not one.
    """
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"
    held = run_study.acquire_lock(out)
    assert held is not None and held.is_file()

    status = run_study.main(["--data", str(episode_dir), "--out", str(out)])

    assert status == run_study.EXIT_LOCKED == 4
    assert calls == [], "the second driver trained a cell anyway"
    printed = capsys.readouterr().out
    announcement, = [line for line in printed.splitlines()
                     if line.startswith("another driver already holds")]
    remedy, = [line for line in printed.splitlines()
               if "delete" in line and "start again" in line]
    assert str(held) in announcement, (
        "the refusal must name the claim it is refusing on; without it the "
        "operator is told someone holds something, somewhere")
    assert str(held) in remedy, (
        "the remedy line must name the file to delete rather than leaning on "
        "the line above it")
    assert str(os.getpid()) in printed, (
        "the message must name who holds it, or an operator cannot tell a "
        "live run from a crashed one and will not dare delete the file")
    assert held.read_bytes(), "the claim must survive the refusal"


def test_the_claim_names_the_pid_the_host_and_the_start_time(tmp_path):
    """All three fields, because the docstring rests on all three.

    `acquire_lock` leaves a crashed run's claim behind deliberately, and the
    justification is that "the file names the pid, the host and the start time,
    so an operator can tell a live run from a dead one". Only the pid was
    pinned: the host could be dropped or replaced by a literal and the start
    time could become the string "unknown", with a green suite. A bare pid is
    ambiguous across the laptop and the rented box, and a claim with no start
    time is exactly the claim an operator will not dare delete -- so they leave
    it, and the resume they came to run refuses.
    """
    before = datetime.now().replace(microsecond=0)
    path = run_study.acquire_lock(tmp_path / "study")
    after = datetime.now()

    claim = json.loads(path.read_text())
    assert claim["pid"] == os.getpid()
    assert claim["host"] == socket.gethostname(), (
        "the host tells the laptop's run from the rented box's")
    # Parsed rather than matched, so the literal "unknown" fails here, and
    # bounded, so a frozen timestamp does too.
    assert before <= datetime.fromisoformat(claim["started"]) <= after
    assert set(claim) == {"pid", "host", "started"}, (
        "a field was added or dropped; every one of them is what the operator "
        "decides on before deleting a file that may belong to a live run")


@pytest.mark.parametrize("content", [b"", b"   \n\t\n  "],
                         ids=["zero_byte", "whitespace_only"])
def test_a_claim_that_is_empty_still_says_something_to_decide_on(
        tmp_path, episode_dir, monkeypatch, capsys, content):
    """A ZERO-BYTE CLAIM IS THE LIKELIEST DAMAGED ONE, and reading it SUCCEEDS.

    It is what a process killed between `os.open` and `json.dump` leaves --
    the code concedes the lock "is written by a process that can be killed
    mid-write" -- and `read_text` returns `""` rather than raising, so the
    `<unreadable>` guard cannot see it. The refusal printed "another driver
    already holds <path>: " and stopped at the colon, and the operator at 2am
    then has to decide whether to delete a claim on no evidence at all: delete
    a live run's and the study runs twice, leave a dead one's and the resume
    they came to do refuses.

    AND WHITESPACE IS THE SAME CLAIM WITH A NEWLINE IN IT. The read is
    `read_text().strip()`, and the `.strip()` was doing all the work here with
    nothing to hold it: dropped, the zero-byte case still passed, and a claim
    holding a lone newline -- an interrupted `echo`, an editor that saves a
    trailing newline into a file it emptied, a partial write that landed on a
    separator -- went straight back to printing the bare colon this branch was
    written to remove. The two cases share every line below because they are
    one defect; only the bytes on disk differ.
    """
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"
    out.mkdir()
    held = out / run_study.LOCK_NAME
    held.write_bytes(content)
    assert not held.read_text().strip(), (
        "this case exists because the read SUCCEEDS and yields nothing an "
        "operator could act on -- not because it raises")

    status = run_study.main(["--data", str(episode_dir), "--out", str(out)])

    assert status == run_study.EXIT_LOCKED
    assert calls == []
    printed = capsys.readouterr().out
    assert str(held) in printed
    assert "<empty" in printed, (
        "the refusal must name the empty claim as empty rather than trailing "
        "off after the colon")
    assert f"{held}: \n" not in printed
    assert held.exists(), "the claim must survive the refusal"


def test_out_two_levels_deep_is_created_rather_than_crashing(
        tmp_path, episode_dir, monkeypatch):
    """`parents=True` IS LOAD-BEARING, and every test writes a one-level --out.

    `runs/` is gitignored with zero tracked files, so on a freshly cloned repo
    on the rented GPU box the directory does not exist, and `acquire_lock`'s
    mkdir is the FIRST thing in the whole driver that creates `--out` --
    `pending_jobs` and `stale_records` only build paths. With `parents=False`
    the plan's own Step 3 command dies at second zero with a bare
    `FileNotFoundError` raised from inside the lock helper, a traceback that
    never mentions `--out` or a missing directory. Every other test here uses
    `tmp_path / "study"`, whose parent always exists.
    """
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "runs" / "m3_study"
    assert not out.parent.exists(), "the missing parent is the whole point"

    assert run_study.main(["--data", str(episode_dir), "--out", str(out),
                           "--arms", "pixel_ae", "--seeds", "0"]) == 0

    assert len(calls) == 1
    assert job_record_path(out, StudyJob("pixel_ae", 0)).is_file()


def test_an_out_that_cannot_be_created_is_not_reported_as_a_failed_cell(
        tmp_path, episode_dir, monkeypatch, capsys):
    """AN UNCAUGHT TRACEBACK EXITS 1, AND 1 IS `EXIT_JOB_FAILED`.

    `--out` naming an existing file made the mkdir raise `FileExistsError`
    straight out of `main`, and the wrapper reading the overnight run's status
    was told "some cells failed, re-run to retry exactly those" when nothing
    had run and the command line was wrong. That is the ambiguity the
    `EXIT_NO_DATA` 2->5 renumbering existed to remove, reintroduced through the
    uncaught-exception path.

    THE PATH IS ASSERTED AGAINST OUR OWN HALF OF THE LINE, NOT THE WHOLE LINE.
    The message is `--out {out_dir} cannot be used as a directory: {error}`,
    and the `OSError` interpolated on the right already reads `[Errno 21] Is a
    directory: '<path>'` -- so `assert str(out) in printed` was answered by the
    strerror, and dropping our own interpolation left `--out cannot be used as
    a directory:` followed by an errno, with a green suite. That message names
    no flag value at all: an operator reading it at 2am cannot tell which of
    `--out` and `--data` the driver is complaining about, which is the mistake
    `stale_report` was already repaired for one function over. So the line is
    split at the colon this module wrote and the path is demanded on the left.
    """
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "not_a_directory"
    out.write_text("this is a file")

    status = run_study.main(["--data", str(episode_dir), "--out", str(out)])

    # Not `EXIT_JOB_FAILED`, which is what an uncaught traceback exits with;
    # the two are pinned apart by `test_every_exit_status_is_distinct_...`,
    # so naming the right one here is enough.
    assert status == run_study.EXIT_OUT_UNUSABLE
    assert calls == []
    printed = capsys.readouterr().out
    ours, separator, strerror = printed.partition(
        " cannot be used as a directory: ")
    assert separator, (
        "the refusal no longer has the shape this test splits on; without the "
        "split every assertion below is made against an empty string")
    assert str(out) in ours, (
        "the message must name --out itself. THIS IS WHY THE SPLIT IS HERE: "
        "the OSError's own text carries the path too, so asserting against "
        "the whole line is satisfied whether or not this message says which "
        "flag it means")
    assert str(out) in strerror, (
        "the OSError really does supply the path on its own -- if it ever "
        "stops, the split above is no longer what makes this test able to "
        "fail and the assertion before it is passing for a new reason")
    assert "no claim file to delete" in printed, (
        "the held-lock remedy is actively wrong here: there is no lock")


def test_a_lock_that_cannot_be_created_is_not_reported_as_one_already_held(
        tmp_path, episode_dir, monkeypatch, capsys):
    """ENOSPC, EROFS, EACCES AND EMFILE ARE NOT "SOMEONE ELSE HOLDS THIS".

    `except FileExistsError` widened to `except OSError` survives the suite,
    and then every failure of `os.open` at all is reported as another driver's
    claim: the run exits `EXIT_LOCKED` and the printed remedy tells the
    operator to delete a lock file that does not exist, at 2am, on a box they
    are paying for, while the actual fault is the filesystem. The two want
    opposite responses -- go and find the other run, versus fix the disk --
    which is the same argument the module makes for why `EXIT_NO_DATA` must not
    be argparse's 2.

    The `FileExistsError` half is exercised alone by
    `test_a_second_driver_on_the_same_out_refuses_to_start`; this is the other
    half alone.
    """
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    real_open = os.open

    def refusing(path, flags, *args, **kwargs):
        if str(path).endswith(run_study.LOCK_NAME):
            raise PermissionError(13, "Permission denied")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(run_study.os, "open", refusing)
    out = tmp_path / "study"

    status = run_study.main(["--data", str(episode_dir), "--out", str(out)])

    # Not `EXIT_LOCKED`: a full disk and a second driver want opposite
    # responses and must not be the same event to the wrapper. The two codes
    # are pinned apart by `test_every_exit_status_is_distinct_...`.
    assert status == run_study.EXIT_OUT_UNUSABLE
    assert calls == []
    printed = capsys.readouterr().out
    assert "Permission denied" in printed, "the real fault must be reported"
    assert str(out / run_study.LOCK_NAME) in printed, (
        "the message must name the claim path it could not create. The errno "
        "raised here carries no filename -- unlike the mkdir failure, whose "
        "strerror was answering that test's path assertion for it -- so "
        "nothing else in this output can supply it")
    assert "already holds" not in printed, (
        "this is not a claim, and telling the operator to delete one sends "
        "them looking for a file that does not exist")


def test_the_release_survives_a_claim_that_is_already_gone(
        tmp_path, episode_dir, monkeypatch):
    """`missing_ok=True` EXISTS SO THE RELEASE CANNOT REPLACE THE RUN'S STATUS.

    The `finally` is there so a `return` cannot leak the claim; with
    `missing_ok=False` the release itself becomes a way to fail, and a
    `FileNotFoundError` out of a `finally` clause discards whatever `_run`
    returned. An operator who deletes the claim while the run is live -- having
    decided from a stale-looking pid that it was dead -- turns a clean 9/9 into
    a traceback and an exit status of 1.
    """
    out = tmp_path / "study"

    def behaviour(job):
        (out / run_study.LOCK_NAME).unlink(missing_ok=True)
        return None

    fake, calls = _spy(behaviour)
    monkeypatch.setattr(run_study, "run_job", fake)

    assert run_study.main(["--data", str(episode_dir), "--out", str(out),
                           "--arms", "pixel_ae", "--seeds", "0"]) == 0
    assert len(calls) == 1
    assert not (out / run_study.LOCK_NAME).exists()


def test_a_byte_damaged_lock_file_still_produces_the_refusal(
        tmp_path, episode_dir, monkeypatch, capsys):
    """`read_text` raises `UnicodeDecodeError` here, and that is a
    `ValueError`, NOT an `OSError`.

    THE SAME DISTINCTION `complete_record` TURNS ON, MISSED AGAIN ONE FUNCTION
    OVER -- in code written by the commit that had just fixed it there. The
    lock is written by a process that can be killed between `open` and `write`
    and read by a second driver that is about to be told to go away, so a
    half-written or byte-damaged claim is the ordinary case, not an exotic one.
    Caught only as `OSError`, it came out of `main` as a decode traceback that
    never mentions the lock at all: the operator at 2am is then debugging a
    crashed second driver instead of reading one line saying who holds the
    directory.
    """
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"
    out.mkdir()
    held = out / run_study.LOCK_NAME
    held.write_bytes(b'{"pid": 4242, "host": "\xff\xfe not utf-8"}')
    with pytest.raises(UnicodeDecodeError):
        held.read_text()

    status = run_study.main(["--data", str(episode_dir), "--out", str(out)])

    assert status == run_study.EXIT_LOCKED
    assert calls == [], "the second driver trained a cell anyway"
    printed = capsys.readouterr().out
    assert str(held) in printed, "the message must name the file to delete"
    assert "<unreadable>" in printed
    assert held.read_bytes(), "the claim must survive the refusal"


def test_a_lock_that_cannot_be_read_at_all_still_produces_the_refusal(
        tmp_path, episode_dir, monkeypatch, capsys):
    """The `OSError` half of that same guard, alone.

    A directory at the claim path raises `IsADirectoryError`, which a lone
    `except ValueError` would not catch. The two halves fail on different
    inputs and neither can be dropped -- the same pairing `complete_record`
    already has, in the one place it was missing.
    """
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"
    held = out / run_study.LOCK_NAME
    held.mkdir(parents=True)
    with pytest.raises(OSError):
        held.read_text()

    status = run_study.main(["--data", str(episode_dir), "--out", str(out)])

    assert status == run_study.EXIT_LOCKED
    assert calls == []
    printed = capsys.readouterr().out
    assert str(held) in printed
    assert "<unreadable>" in printed
    assert held.is_dir(), "the claim must survive the refusal"


def test_the_lock_is_released_when_the_run_finishes(
        tmp_path, episode_dir, monkeypatch):
    """A claim left behind after a clean run would make every later resume
    refuse -- which is worse than the double-run it prevents."""
    fake, _ = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"
    argv = ["--data", str(episode_dir), "--out", str(out)]

    assert run_study.main(argv) == 0
    assert not (out / run_study.LOCK_NAME).exists()
    assert run_study.main(argv) == 0, "the second run was locked out"
    assert not run_study.LOCK_NAME.endswith(".json"), (
        "the claim lives beside the nine records and the aggregation globs "
        "that directory; it must not look like one of them")


def test_the_lock_is_released_when_a_human_interrupts(
        tmp_path, episode_dir, monkeypatch):
    """Ctrl-C on the box must not leave a claim the operator has to discover
    and delete before resuming."""
    def behaviour(job):
        raise KeyboardInterrupt

    fake, _ = _spy(behaviour)
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"

    with pytest.raises(KeyboardInterrupt):
        run_study.main(["--data", str(episode_dir), "--out", str(out)])
    assert not (out / run_study.LOCK_NAME).exists()


def test_the_lock_is_released_when_every_cell_fails(
        tmp_path, episode_dir, monkeypatch):
    """The failure path returns rather than falling off the end, and a `return`
    that skips the release is exactly the kind of leak a `finally` prevents."""
    def behaviour(job):
        raise RuntimeError("every cell fails")

    fake, _ = _spy(behaviour)
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"

    status = run_study.main(["--data", str(episode_dir), "--out", str(out)])

    assert status == run_study.EXIT_JOB_FAILED
    assert not (out / run_study.LOCK_NAME).exists()


def test_no_claim_is_left_behind_when_there_is_nothing_to_do(
        tmp_path, monkeypatch):
    """Nothing pending means nothing to protect, and the finished study's
    directory must not acquire a file on every `--out` inspection."""
    fake, _ = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"
    for arm in ARMS:
        for seed in SEEDS:
            _write_matching(out, StudyJob(arm, seed))

    assert run_study.main(["--data", str(tmp_path / "gone"), "--out", str(out),
                           *MATCHING_ARGV]) == 0
    assert not (out / run_study.LOCK_NAME).exists()


def test_main_lists_what_it_is_about_to_do_before_it_starts(
        tmp_path, episode_dir, monkeypatch, capsys):
    """13.5 hours later the log's first line is how you tell what was attempted
    from what was skipped."""
    fake, _ = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"
    _write_matching(out, DONE_JOB)

    run_study.main(["--data", str(episode_dir), "--out", str(out),
                    *MATCHING_ARGV])

    header = capsys.readouterr().out.splitlines()[0]
    assert "8 job(s) pending" in header
    assert f"{DONE_JOB.arm}/s{DONE_JOB.seed}" not in header
    assert f"{CORRUPT_JOB.arm}/s{CORRUPT_JOB.seed}" in header
