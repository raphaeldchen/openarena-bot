"""The driver over the nine cells: which ones are pending, in what order.

This script decides whether an 8.3-hour cell gets paid for a second time, and
it is the only thing standing between a 33-hour unattended run and a morning
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
import subprocess
import sys
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
from mbfps.utils.config import ARMS

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
    **{f"mismatch.{key}": value for key, value in CONFIG_MISMATCHES.items()},
}

# The three scenario cells: one per arm AND one per seed, so that no test can
# pass by hardcoding either field, and so that all three arms are exercised.
# `frozen_ssl` is the arm Task 4's tests never ran and the only one whose
# backbone is not its own name; leaving it out of a fixture is how three seeds
# of the study died hours in.
DONE_JOB = StudyJob("cnn", 1)
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
# cnn record with no position block would then have been marked done: the
# 8.3-hour cell never re-runs and the gate is evaluated on a record carrying no
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
    "nonfinite",
})
"""Spelled out, `NONFINITE_KEY` included, so a RENAME is caught as well."""

OPTIONAL_RECORD_KEYS = frozenset({"kl_dyn_max", "loss_last20"})
"""The top-level keys `run_job` writes that a record is allowed to lack.

`run_job` writes twenty top-level keys; eighteen of them are required. These
two are training diagnostics printed in the log, and a record without them is
still a finished cell. Naming them here is what lets the drift guard below be
an EQUALITY against a real record instead of a subset test.
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
MATCHING_ARGV = ["--steps", str(CLI_STEPS), "--seq-len", str(CLI_SEQ_LEN)]


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
    """
    calls: list[dict] = []

    def fake(job, buffer, out_dir, **kwargs):
        calls.append({"job": job, "buffer": buffer, "out_dir": out_dir,
                      "kwargs": kwargs})
        overrides = behaviour(job) if behaviour is not None else None
        body = _record_body(
            job, **{**_recorded_config(kwargs), **(overrides or {})})
        write_record(job_record_path(out_dir, job), body)
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
    assert len(set(values)) == 19, "all nineteen must still be listed"

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
    assert set(run_study._SLOW_ARMS) < set(ARMS), (
        "the expensive arms must be a proper subset, or nothing runs first")
    assert NONFINITE_KEY in run_study.REQUIRED_RECORD_KEYS, (
        "only `write_record` adds this key, so requiring it is what stops a "
        "hand-made JSON blob at a record path from being mistaken for a "
        "finished 8.3-hour cell")


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
    """Resumability. The pixel arm is 8.3 h per seed; re-running it is a lost
    day. Parametrised over all three arms because a skip that only fires for
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
    assert "cnn" not in {j.arm for j in jobs}, (
        "the excluded arm was run anyway; the defaults are being used instead "
        "of the caller's arguments")


def test_a_repeated_arm_or_seed_does_not_buy_the_same_cell_twice(tmp_path):
    """`--arms cnn cnn` is a typo, not a request for 16.6 GPU-hours."""
    assert run_study.pending_jobs(tmp_path, ("cnn", "cnn"), (0, 0)) == [
        StudyJob("cnn", 0)]


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


@pytest.mark.parametrize("missing", sorted(EXPECTED_RECORD_KEYS))
def test_a_record_missing_any_field_is_treated_as_pending(tmp_path, missing):
    """THE DANGEROUS CASE: valid JSON that is not a complete record.

    A file that fails to parse costs one re-run. A file that parses and is
    short of `position` is read by the aggregation as this cell's result, and
    the study reports a cell nobody measured. Every required key is removed in
    turn; the control below writes the same record intact.

    PARAMETRISED OVER `EXPECTED_RECORD_KEYS`, NOT OVER THE DRIVER'S OWN SET.
    Deriving the cases from the collection under test made this shrink with it:
    delete "position" from `REQUIRED_RECORD_KEYS` and the position case simply
    stopped being generated, so the suite stayed green while a cnn record with
    no position block became a finished cell.
    """
    path = job_record_path(tmp_path, INCOMPLETE_JOB)
    damaged = _sanitised(INCOMPLETE_JOB)
    del damaged[missing]
    path.write_text(json.dumps(damaged))
    assert not run_study.record_is_complete(path, INCOMPLETE_JOB)
    assert INCOMPLETE_JOB in run_study.pending_jobs(tmp_path, ARMS, SEEDS)

    # The control. Without it, an implementation that calls everything
    # incomplete would pass all eighteen parametrisations above.
    path.write_text(json.dumps(_sanitised(INCOMPLETE_JOB)))
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
    path.write_bytes(b'{"arm": "cnn", "seed": \xff\xfe\x00 not utf-8}')
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
        "short of it count as a finished 8.3-hour cell")
    assert OPTIONAL_RECORD_KEYS <= written, (
        "these are subtracted above to make that an equality; if `run_job` "
        "has stopped writing them the subtraction quietly stops testing "
        "anything")
    assert run_study.record_is_complete(path, job)
    assert job not in run_study.pending_jobs(tmp_path, ARMS, SEEDS)
    # The live record is what `main` prints from, and it is not the file.
    assert run_study.record_names_the_job(record, job)


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------

def test_cheap_arms_run_first(tmp_path):
    """Feature arms are 5.9x faster; running them first means an interruption
    still leaves the treatment/control contrast complete."""
    jobs = run_study.pending_jobs(tmp_path, ARMS, SEEDS)
    cnn_positions = [i for i, j in enumerate(jobs) if j.arm == "cnn"]
    other_positions = [i for i, j in enumerate(jobs) if j.arm != "cnn"]
    assert min(cnn_positions) > max(other_positions)


def test_the_order_is_fully_determined_and_seeds_ascend(tmp_path):
    """The seeds are handed over SHUFFLED on purpose.

    `sorted` is stable, so with the seed missing from the sort key the arms
    would still be grouped correctly and only the seeds inside each arm would
    follow the caller's arbitrary order. An interrupted study should always
    hold seeds 0..k of an arm, never an arbitrary subset of them.
    """
    jobs = run_study.pending_jobs(tmp_path, ("cnn", "random_vit"), (2, 0, 1))
    assert jobs == [
        StudyJob("random_vit", 0), StudyJob("random_vit", 1),
        StudyJob("random_vit", 2),
        StudyJob("cnn", 0), StudyJob("cnn", 1), StudyJob("cnn", 2),
    ]


def test_the_order_holds_after_a_partial_resume(tmp_path):
    """Resume and ordering interact: what is left must still be cheap-first."""
    _write_complete(tmp_path, StudyJob("frozen_ssl", 0))
    _write_complete(tmp_path, StudyJob("cnn", 0))
    jobs = run_study.pending_jobs(tmp_path, ARMS, SEEDS)
    assert jobs == [
        StudyJob("frozen_ssl", 1), StudyJob("frozen_ssl", 2),
        StudyJob("random_vit", 0), StudyJob("random_vit", 1),
        StudyJob("random_vit", 2),
        StudyJob("cnn", 1), StudyJob("cnn", 2),
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
    """The print must not be the thing that kills a 33-hour run.

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


def test_job_summary_reports_the_configuration_the_record_carries():
    """A log that echoes the flags we asked for cannot show that something
    else was used. Task 4 shipped exactly that defect: the record reported
    `run_job`'s parameters rather than the config actually trained with, so a
    steps/seq_len swap trained every cell wrong and reported it right.
    """
    job = StudyJob("cnn", 0)
    record = _record_body(job)
    text = run_study.job_summary(job, record, 61.0)
    assert f"steps={RECORD_STEPS}" in text
    assert f"seq_len={RECORD_SEQ_LEN}" in text
    assert f"context={RECORD_CONTEXT}" in text
    assert f"horizon={RECORD_HORIZON}" in text
    assert str(CLI_STEPS) not in text and str(DEFAULT_STEPS) not in text


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


def test_main_forwards_the_flags_it_was_given(
        tmp_path, episode_dir, monkeypatch):
    """A driver that drops `--device` on the rented CUDA box falls back through
    `get_device` to the CPU and the study takes weeks -- silently, and not on
    this laptop, which has no CUDA either way."""
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)

    assert run_study.main([
        "--data", str(episode_dir), "--out", str(tmp_path / "study"),
        "--steps", str(CLI_STEPS), "--seq-len", str(CLI_SEQ_LEN),
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
                    "--arms", "cnn", "--seeds", "0"])

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
    ValueError("checkpoint is arm='cnn', not this job's"),
    KeyError("privileged"),
], ids=lambda e: type(e).__name__)
def test_a_failing_cell_does_not_abort_the_other_eight(
        tmp_path, episode_dir, monkeypatch, capsys, error):
    """THE POLICY, tested. Nobody is watching a 33-hour unattended run, and the
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
    assert [c["job"] for c in calls][-1].arm == "cnn", (
        "the expensive arm was never reached after the failure")
    printed = capsys.readouterr().out
    assert type(error).__name__ in printed, "no traceback logged"
    assert "8/9 job(s) completed" in printed

    # EXACTLY the cells that failed, and not one more. The morning's first
    # question is which cells to retry, and a line naming all nine -- which is
    # what iterating `jobs` here instead of `failed` prints -- answers it
    # wrongly and sends the operator back for another 33 hours.
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


@pytest.mark.parametrize("overrides", [
    {"arm": "cnn"},          # wrong arm, RIGHT seed
    {"seed": CLI_SEED},      # right arm, WRONG seed
], ids=["wrong_arm", "wrong_seed"])
def test_main_reports_a_record_that_names_another_cell(
        tmp_path, episode_dir, monkeypatch, capsys, overrides):
    """The worst outcome the study has, and the reason each half is exercised
    alone: a record at this cell's path describing another cell would be
    aggregated under the wrong arm's name."""
    liar = StudyJob("frozen_ssl", 1)

    def behaviour(job):
        return overrides if job == liar else None

    fake, _ = _spy(behaviour)
    monkeypatch.setattr(run_study, "run_job", fake)

    status = run_study.main(["--data", str(episode_dir),
                             "--out", str(tmp_path / "study")])

    assert status == run_study.EXIT_JOB_FAILED
    printed = capsys.readouterr().out
    assert "MISLABELLED" in printed
    assert f"{liar.arm}/s{liar.seed}" in printed.split("FAILED:")[-1]
    assert "8/9 job(s) completed" in printed


def test_main_stops_before_the_first_cell_when_there_are_no_episodes(
        tmp_path, monkeypatch, capsys):
    """With no data every one of the nine cells fails identically inside
    `episode_split`, hours apart. Fail on the first second instead."""
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    empty = tmp_path / "no_data"
    empty.mkdir()

    status = run_study.main(["--data", str(empty), "--out", str(tmp_path / "s")])

    assert status == run_study.EXIT_NO_DATA == 2
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
    assert excinfo.value.code == 2


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
    assert excinfo.value.code == 2


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

    assert completed.returncode == run_study.EXIT_NO_DATA == 2, (
        f"stdout={completed.stdout!r}\nstderr={completed.stderr[-3000:]!r}")
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
    assert "CONFIGURATION MISMATCH" in printed
    assert f"{DONE_JOB.arm}/s{DONE_JOB.seed}" in printed
    assert f"{key}: recorded {CONFIG_MISMATCHES[key]!r} != requested" in printed


def test_nine_smoke_records_refuse_instead_of_reporting_a_finished_study(
        tmp_path, monkeypatch, capsys):
    """THE TRAP THIS EXISTS FOR, and the plan's own Task 7 Step 1 walks into it.

    `record_is_complete` validated the arm and the seed and nothing about the
    configuration, so nine records from a short smoke run made the real 33-hour
    study a no-op: every cell read done, the driver exited 0 having trained
    nothing, and the gate would have been computed from smoke-run models.

    It REFUSES rather than treating them as pending. Re-running would fix the
    smoke trap and create a worse one in the other direction: after the real
    study finishes, a quick 100-step check into the same --out would leave the
    driver deciding all nine records are stale and OVERWRITING 33 hours of the
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
    assert "Remedy" in printed
    assert not gone.exists(), (
        "the refusal came after `ReplayBuffer` was asked for the episodes; it "
        "has to fail in the first second, not the thirty-third hour")
    assert not (out / run_study.LOCK_NAME).exists()


def test_a_matching_rerun_still_skips_the_cells_that_are_finished(
        tmp_path, episode_dir, monkeypatch, capsys):
    """THE OTHER HALF, and the driver's whole point.

    A guard that called every record stale would refuse every resume, which
    costs the study the 33 hours the resume exists to save.
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


def test_only_the_selected_cells_are_checked_for_staleness(
        tmp_path, episode_dir, monkeypatch):
    """A stale record for an arm this invocation is not running must not block
    it -- otherwise one old file makes every future `--arms` run impossible."""
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"
    _write_matching(out, DONE_JOB, steps=SMOKE_STEPS)   # cnn/s1
    assert DONE_JOB.arm == "cnn"

    status = run_study.main(["--data", str(episode_dir), "--out", str(out),
                             "--arms", "frozen_ssl", *MATCHING_ARGV])

    assert status == 0
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# one driver per --out
# ---------------------------------------------------------------------------

def test_a_second_driver_on_the_same_out_refuses_to_start(
        tmp_path, episode_dir, monkeypatch, capsys):
    """Two drivers pointed at one --out both saw all nine cells pending and
    both ran all nine: 66 GPU-hours instead of 33, racing on the same record
    and checkpoint paths, with no warning in either log."""
    fake, calls = _spy()
    monkeypatch.setattr(run_study, "run_job", fake)
    out = tmp_path / "study"
    held = run_study.acquire_lock(out)
    assert held is not None and held.is_file()

    status = run_study.main(["--data", str(episode_dir), "--out", str(out)])

    assert status == run_study.EXIT_LOCKED == 4
    assert calls == [], "the second driver trained a cell anyway"
    printed = capsys.readouterr().out
    assert str(held) in printed
    assert str(os.getpid()) in printed, (
        "the message must name who holds it, or an operator cannot tell a "
        "live run from a crashed one and will not dare delete the file")
    assert held.read_bytes(), "the claim must survive the refusal"


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
    """33 hours later the log's first line is how you tell what was attempted
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
