"""Run the 3 arms x 3 seeds study, resumably.

Measured costs at seq_len=64 on the development Mac: feature arms 3.98 steps/s
(1.4 h per 20k-step job), pixel arm 0.67 steps/s (8.3 h). The whole study is
about 33 laptop-hours, of which the pixel arm is 25. Feature arms are therefore
scheduled first: if the run is interrupted, the treatment/control contrast
(`frozen_ssl` against `random_vit`) is already complete, and what is missing is
the baseline rather than the comparison the study exists to make.

Resumable off the per-job JSON records, so an interrupted run costs at most one
job rather than the whole study.

WHAT "ALREADY DONE" MEANS HERE, AND WHY IT IS NOT "THE FILE EXISTS". The record
is the study's only artifact and this script is the only thing that decides
whether an 8.3-hour cell gets paid for twice. Three states have to be told
apart:

  * no file                -> pending, obviously;
  * a file that does not parse, or parses but is not a complete record
                           -> pending. A process killed mid-write used to be
                              able to leave a prefix at the real path; the
                              write is atomic now (see `study.write_record`),
                              but records also arrive from older code, from
                              half-finished manual edits and from a `cp` that
                              was interrupted, and a truncation that happens to
                              parse is the dangerous one -- it would mark the
                              cell done and the aggregation would then read a
                              record with no `position` block in it;
  * a complete record naming this very arm and seed
                           -> done, skip it.

The last clause is not paranoia about our own filenames. `job_record_path` puts
both fields in the name precisely so two cells cannot collide, and checking the
contents as well is what turns a mis-copied or mis-renamed file into a re-run
instead of one cell being reported twice under two names.

AND "DONE" IS ALSO NOT "DONE AT SOME OTHER CONFIGURATION". A complete record
says what it was trained with, and a record from a three-step smoke run is a
complete record: without the check below, nine of them turn the real 33-hour
study into a no-op that exits 0, and the gate is then computed from three-step
models. The driver REFUSES and stops -- see `stale_records`.

FAILURE POLICY: ONE BAD CELL DOES NOT ABORT THE RUN. A job that raises is
reported with its full traceback, counted, and the driver moves on to the next
cell; the process exits non-zero at the end if any cell failed. The alternative
-- letting the exception out of the loop -- optimises for noticing quickly, and
nobody is watching: this runs unattended overnight against a rented box. The
realistic failures are per-cell (a missing feature cache for one arm, a CUDA
OOM on one seed, a corrupt episode), so aborting throws away every remaining
cell to punish one, and the eight survivors are exactly what makes the failure
diagnosable in the morning. Nothing is lost by continuing either: NO RECORD IS
WRITTEN FOR A FAILED CELL, so re-running the script picks up precisely the
cells that failed. `KeyboardInterrupt` and `SystemExit` are deliberately not
caught -- when a human does interrupt, they mean the whole run.

`except Exception` IS THE WIDTH THE POLICY NEEDS. The headline example above --
a missing feature cache for one arm -- raises `FileNotFoundError`, which is an
`OSError` and neither a `RuntimeError` nor a `ValueError`. Narrowing this to
the exception types we happen to have seen would let that arm's three cells
take down the run the policy exists to protect.
"""

import argparse
import inspect
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from mbfps.data.buffer import ReplayBuffer
from mbfps.eval.study import NONFINITE_KEY, StudyJob, job_record_path, run_job
from mbfps.utils.config import ARMS

SEEDS: tuple[int, ...] = (0, 1, 2)
"""The three seeds. Three cells per arm is what makes a spread reportable."""

_SLOW_ARMS: tuple[str, ...] = ("cnn",)
"""Arms that predict in pixel space, at 0.67 steps/s against the others' 3.98.

Membership of this tuple is the whole scheduling policy: everything not in it
runs first. It is a tuple rather than a test on the arm name so that a fourth
arm can be added on the correct side of the split by editing one line.
"""

BUFFER_CAPACITY = 10**9
"""Effectively unbounded: this driver reads episodes and must never evict one.

`ReplayBuffer` deletes the oldest episodes past its capacity, and the nine
cells must be trained and scored on the identical episode set or the study
compares arms against different data.
"""

REQUIRED_RECORD_KEYS = frozenset({
    "arm", "seed", "steps", "seq_len", "context", "horizon", "split_seed",
    "seconds", "steps_per_second", "kl_rate_above_free_bits", "episodes",
    "probe", "position", "angle", "filtering", "reward", "curves",
    NONFINITE_KEY,
})
"""Every top-level key `run_job` writes. A record missing one is not done.

This is the completeness test that "the file parses" is not. `NONFINITE_KEY` is
in the set on purpose: only `write_record` adds it, so a hand-made or
half-converted JSON blob at a record path is treated as pending rather than
mistaken for a finished 8.3-hour cell.

DRIFT IN EITHER DIRECTION IS A DISASTER WITH NO SYMPTOM: a key listed here that
`run_job` never writes makes EVERY cell permanently pending, so the study
re-runs from zero on every resume; a key DROPPED from here lets a record short
of that field count as a finished cell.

The guards are deliberately not derived from this set. `EXPECTED_RECORD_KEYS`
in `tests/eval/test_run_study.py` is an independent hand-written literal
compared for EQUALITY, and the drift test compares this set for equality
against a real `run_job` record. A guard parametrised over this collection
would have shrunk with it: deleting a key deleted its own test case, and every
one of `position`, `curves`, `filtering`, `reward` and `probe` could be removed
with a green suite.
"""

CONFIG_KEYS: tuple[str, ...] = ("steps", "seq_len", "context", "horizon")
"""The fields that say what a record was trained with, not just which cell.

`record_is_complete` answers "is this cell finished"; these answer "finished at
the configuration we are asking for". See `stale_records`.
"""

_RUN_JOB_PARAMETERS = inspect.signature(run_job).parameters
PROTOCOL_CONTEXT: int = _RUN_JOB_PARAMETERS["context"].default
PROTOCOL_HORIZON: int = _RUN_JOB_PARAMETERS["horizon"].default
"""`context` and `horizon` READ OFF `run_job`, never re-declared here.

`main` deliberately has no `--context`/`--horizon` flags (see its docstring):
`run_job` forwards one value to the probe fit, the rollout and both filtering
diagnostics, and a second home for them in this file would let the nine records
carry this script's stale copies. The staleness check needs to know what will
be requested, so it asks the one home rather than starting a second one.
"""

LOCK_NAME = "study.lock"
"""Name of the claim file inside `--out`; see `acquire_lock`."""

EXIT_OK = 0
EXIT_JOB_FAILED = 1
EXIT_CONFIG_MISMATCH = 3
EXIT_LOCKED = 4
EXIT_NO_DATA = 5
"""The five statuses an unattended run can end on. All distinct, and NONE OF
THEM IS 2.

2 is argparse's own usage status: `--data` misspelt, `--arms cnn2`, `--seeds`
with no values -- `parser.error` exits 2 and there is no way to stop it. This
file used to number `EXIT_NO_DATA` 2 as well, so a wrapper reading the status
of an overnight run could not tell "I typed the flag wrong and nothing ran" from
"the box mounted the wrong volume and there are no episodes". Those want
opposite responses -- fix the command line, versus go and find the data -- and
distinguishing them is the entire reason these codes exist, since nobody is
watching the log they would otherwise have to read. 2 is therefore left to
argparse and the study's own statuses are numbered around it.
"""


def _reject_nonstandard(constant: str):
    """`json.loads` calls this for the bare `NaN`/`Infinity` tokens.

    Python's parser accepts them; strict parsers do not, and `write_record`
    guarantees the study never emits them. A record that contains one came from
    somewhere else, so it is not one of ours and the cell is re-run.
    """
    raise ValueError(f"record contains the non-JSON token {constant!r}")


def _get(record, *keys):
    """`record[k1][k2]...`, or None if any step is missing or not a mapping.

    Used by the printing path, which must never be the thing that kills a
    33-hour run: a record that is one field short should print `n/a` in that
    column, not raise a KeyError between cell three and cell four.

    BOTH HALVES OF THE GUARD CARRY WEIGHT AND ARE TESTED ALONE. `key not in
    node` covers the missing field; `not isinstance(node, dict)` covers a
    record where a whole block has been replaced by a scalar or a `null`,
    against which `key not in node` raises `TypeError` rather than answering.
    """
    node = record
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def record_names_the_job(record, job: StudyJob) -> bool:
    """Whether `record` is this cell's record rather than another cell's."""
    return _get(record, "arm") == job.arm and _get(record, "seed") == job.seed


def complete_record(path, job: StudyJob) -> dict | None:
    """The parsed record at `path` if it is a finished one for `job`, else None.

    None for: no file, a directory, an unreadable file, a file that is not
    valid UTF-8, an empty file, JSON that does not parse, JSON that is not an
    object, an object missing any of `REQUIRED_RECORD_KEYS`, one carrying a
    non-standard NaN token, and one that names a different arm or seed. Every
    one of those means the cell has to be run; only a complete record for this
    exact cell is worth skipping.

    The record itself is returned rather than a bool because the caller that
    decides "already done" and the caller that decides "done at the requested
    configuration" must read the same bytes and agree.
    """
    try:
        text = Path(path).read_text()
    except (OSError, ValueError):
        # `UnicodeDecodeError` is a `ValueError`, NOT an `OSError`. One
        # byte-damaged file among the nine must cost that one cell, not take
        # the whole resume down at hour zero.
        return None
    try:
        record = json.loads(text, parse_constant=_reject_nonstandard)
    except ValueError:  # JSONDecodeError, and _reject_nonstandard's ValueError
        return None
    if not isinstance(record, dict):
        return None
    if not REQUIRED_RECORD_KEYS <= set(record):
        return None
    if not record_names_the_job(record, job):
        return None
    return record


def record_is_complete(path, job: StudyJob) -> bool:
    """Whether `path` holds a finished record for `job`."""
    return complete_record(path, job) is not None


def requested_config(steps: int, seq_len: int) -> dict:
    """What this invocation is asking every cell to be trained at."""
    return {"steps": steps, "seq_len": seq_len,
            "context": PROTOCOL_CONTEXT, "horizon": PROTOCOL_HORIZON}


def record_config_mismatch(record, config: dict) -> dict:
    """`{key: (recorded, requested)}` for every `CONFIG_KEYS` that disagrees."""
    return {key: (_get(record, key), config[key])
            for key in CONFIG_KEYS
            if _get(record, key) != config[key]}


def stale_records(out_dir, arms, seeds, config: dict) -> list:
    """`[(job, mismatch)]` for finished cells recorded at another configuration.

    Only COMPLETE records are considered. An incomplete one is already pending
    and will simply be re-run; calling it stale would turn one corrupt file
    into a refusal to do anything at all.
    """
    stale: list[tuple[StudyJob, dict]] = []
    seen: set[StudyJob] = set()
    for arm in arms:
        for seed in seeds:
            job = StudyJob(arm, seed)
            if job in seen:
                continue
            seen.add(job)
            record = complete_record(job_record_path(out_dir, job), job)
            if record is None:
                continue
            mismatch = record_config_mismatch(record, config)
            if mismatch:
                stale.append((job, mismatch))
    return stale


def _cost_key(job: StudyJob) -> tuple:
    """Sort key: cheap arms first, then by arm, then by seed.

    The seed is in the key so that the order does not depend on the order the
    caller happened to list the seeds in -- an interrupted study should always
    have completed seeds 0..k of an arm, never an arbitrary subset.
    """
    return (job.arm in _SLOW_ARMS, job.arm, job.seed)


def pending_jobs(out_dir, arms=ARMS, seeds=SEEDS) -> list[StudyJob]:
    """The jobs without a complete record, cheapest arms first.

    `out_dir` may be a `str`; argparse hands one over and `job_record_path`
    accepts it.
    """
    jobs: list[StudyJob] = []
    for arm in arms:
        for seed in seeds:
            job = StudyJob(arm, seed)
            if job in jobs:
                continue  # `--arms cnn cnn` must not buy the same 8.3 h twice
            if record_is_complete(job_record_path(out_dir, job), job):
                continue
            jobs.append(job)
    return sorted(jobs, key=_cost_key)


def acquire_lock(out_dir):
    """Claim `--out` for this process, or return None if someone already has.

    Two drivers pointed at one `--out` both see all nine cells pending and both
    run all nine: 66 GPU-hours instead of 33, racing on the same record and
    checkpoint paths, with no warning in either log. `O_CREAT | O_EXCL` is the
    cheapest thing that makes the second one say so.

    A crashed run leaves the file behind. That is deliberate -- the file names
    the pid, the host and the start time, so an operator can tell a live run
    from a dead one, and the cost of being wrong is one `rm` against the cost
    of silently paying for the study twice.
    """
    path = Path(out_dir) / LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    with os.fdopen(handle, "w") as stream:
        json.dump({"pid": os.getpid(), "host": socket.gethostname(),
                   "started": datetime.now().isoformat(timespec="seconds")},
                  stream)
    return path


def _fmt(value, spec: str = ".4f") -> str:
    """Format a number that may legitimately be non-finite, None, or missing.

    NaN IS A NORMAL VALUE IN THESE RECORDS -- `gap_final` is NaN by contract
    whenever the persistence-to-floor band is non-positive, and `reward.r2` is
    NaN when the target has no variance. Floats format fine when they are
    non-finite (`+nan`), so the hazard is not the NaN itself but the `None` a
    sanitised or truncated record can hold there: `format(None, "+.4f")` raises
    `unsupported format string passed to NoneType.__format__`, and raising in
    the per-job print would kill the loop after the record was safely written
    and lose every cell that had not started yet.

    THE `try` IS NOT DECORATION AND EACH HALF IS TESTED ALONE. `float("n/a")`
    raises `ValueError` and `float(["a"])` raises `TypeError`; a record hand-
    edited or half-converted can hold either where a number belongs, and this
    function exists precisely so that the per-cell print cannot be the thing
    that ends an unattended run.
    """
    if value is None:
        return "n/a"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def job_summary(job: StudyJob, record: dict, wall_seconds: float) -> str:
    """The block printed after a cell finishes, for the morning's `study.log`.

    THE CONFIGURATION LINE IS READ BACK OUT OF THE RECORD, not echoed from this
    script's own flags. A log that prints what we asked for cannot show that
    something else was used; printing what was recorded is what would make a
    steps/seq_len swap visible on the first cell rather than in the write-up.
    """
    lines = [
        f"  cell      arm={_get(record, 'arm')} seed={_get(record, 'seed')} "
        f"(requested {job.arm}/s{job.seed})",
        f"  recorded  steps={_get(record, 'steps')} "
        f"seq_len={_get(record, 'seq_len')} context={_get(record, 'context')} "
        f"horizon={_get(record, 'horizon')} "
        f"split_seed={_get(record, 'split_seed')}",
        f"  timing    wall={_fmt(wall_seconds, '.1f')}s "
        f"recorded={_fmt(_get(record, 'seconds'), '.1f')}s "
        f"steps_per_second={_fmt(_get(record, 'steps_per_second'), '.2f')}",
        f"  training  kl_rate_above_free_bits="
        f"{_fmt(_get(record, 'kl_rate_above_free_bits'), '.3f')} "
        f"kl_dyn_max={_fmt(_get(record, 'kl_dyn_max'), '.3f')} "
        f"loss_last20={_fmt(_get(record, 'loss_last20'), '.4f')}",
        f"  episodes  train={len(_get(record, 'episodes', 'train') or [])} "
        f"val={len(_get(record, 'episodes', 'val') or [])}",
    ]
    for metric in ("position", "angle"):
        lines.append(
            f"  {metric:<9} gap_final="
            f"{_fmt(_get(record, metric, 'gap_final'), '+.4f')} "
            f"gap_mean={_fmt(_get(record, metric, 'gap_mean'), '+.4f')} "
            f"finite={_get(record, metric, 'gap_finite')}"
            f"/{_get(record, metric, 'n_steps')} "
            f"band_median={_fmt(_get(record, metric, 'band_median'), '.3f')} "
            f"degenerate={_get(record, metric, 'steps_degenerate')}"
        )
    lines.append(
        f"  filtering criterion_4_latent_beats_embedding="
        f"{_get(record, 'filtering', 'criterion_4', 'latent_beats_embedding')} "
        f"gain={_fmt(_get(record, 'filtering', 'gain', 'gain'), '+.4f')} "
        f"CI[{_fmt(_get(record, 'filtering', 'gain', 'ci_low'), '+.4f')}, "
        f"{_fmt(_get(record, 'filtering', 'gain', 'ci_high'), '+.4f')}]"
    )
    lines.append(
        f"  reward    mse={_fmt(_get(record, 'reward', 'mse'), '.5f')} "
        f"baseline={_fmt(_get(record, 'reward', 'baseline_mse'), '.5f')} "
        f"r2={_fmt(_get(record, 'reward', 'r2'), '+.4f')} "
        f"degenerate_target={_get(record, 'reward', 'is_degenerate')}"
    )
    return "\n".join(lines)


def stale_report(stale: list, out_dir) -> str:
    """What the operator sees instead of a 33-hour run that does nothing."""
    lines = [
        f"CONFIGURATION MISMATCH: {len(stale)} finished record(s) in {out_dir} "
        "were produced at a different configuration.",
        "Skipping them would report those cells at settings nobody asked for; "
        "re-running them would overwrite the study's only artifact. "
        "Refusing instead.",
    ]
    for job, mismatch in stale:
        detail = "  ".join(
            f"{key}: recorded {recorded!r} != requested {wanted!r}"
            for key, (recorded, wanted) in sorted(mismatch.items()))
        lines.append(f"  {job.arm}/s{job.seed}  {detail}")
    lines.append(
        "Remedy: point --out somewhere else, or delete those records, or ask "
        "for the configuration they were run at.")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # --data and --out stay STRINGS. Everything downstream takes a str
    # (`job_record_path`, `run_job`, `write_record`, `load_record` all coerce),
    # and leaving them as argparse hands them over means the study exercises
    # the same path the tests do rather than a `Path`-only one.
    parser.add_argument("--data", default="data/my_way_home")
    parser.add_argument("--out", default="runs/m3_study")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arms", nargs="*", choices=ARMS, default=list(ARMS))
    # `--seeds` is restricted exactly as `--arms` is. A shell typo -- `--seeds
    # 10` -- otherwise buys a night of a rented box training a tenth cell the
    # aggregation will never look at, and exits 0.
    parser.add_argument("--seeds", nargs="*", type=int, choices=SEEDS,
                        default=list(SEEDS))
    return parser


def main(argv=None) -> int:
    """Run every pending cell. Returns the process exit status.

    `context` and `horizon` are deliberately NOT flags here. `run_job` forwards
    one value to the probe fit, the rollout and both filtering diagnostics
    because they must not default apart, and a flag in this script would be a
    second home for the study's evaluation protocol -- the day `run_job`'s
    defaults move, the nine records would still carry this file's stale ones.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    # `nargs="*"` accepts the flag with no values at all, which selects nothing
    # and would print "0 job(s) pending", exit 0, and look like a finished
    # study to any wrapper.
    if not args.arms:
        parser.error("--arms was given no values, so no cell would run")
    if not args.seeds:
        parser.error("--seeds was given no values, so no cell would run")

    jobs = pending_jobs(args.out, tuple(args.arms), tuple(args.seeds))
    listing = ", ".join(f"{j.arm}/s{j.seed}" for j in jobs) or "none"
    print(f"{datetime.now().isoformat(timespec='seconds')} "
          f"{len(jobs)} job(s) pending: {listing}", flush=True)

    # Before the "nothing to do" return, or nine smoke records still exit 0;
    # and before `ReplayBuffer`, so this fails in the first second rather than
    # the thirty-third hour.
    config = requested_config(args.steps, args.seq_len)
    stale = stale_records(args.out, tuple(args.arms), tuple(args.seeds), config)
    if stale:
        print(stale_report(stale, args.out), flush=True)
        return EXIT_CONFIG_MISMATCH

    if not jobs:
        # Return before touching --data: a finished study must not depend on
        # the episodes still being on the box, and `ReplayBuffer` would create
        # the directory as a side effect of being asked.
        print(f"nothing to do; all records already in {args.out}")
        return EXIT_OK

    lock = acquire_lock(args.out)
    if lock is None:
        held = Path(args.out) / LOCK_NAME
        try:
            holder = held.read_text()
        except (OSError, ValueError):
            # `UnicodeDecodeError` is a `ValueError`, NOT an `OSError` -- the
            # same distinction `complete_record` above turns on, and this read
            # got it wrong in the commit that fixed it there. The lock is
            # written by a process that can be killed mid-write and read by a
            # second driver that is about to be told to go away: a byte-damaged
            # claim file must produce the refusal message with an unreadable
            # holder, not a traceback out of `main` that says nothing about the
            # lock at all and leaves the operator with a crashed second driver
            # to diagnose instead of a one-line "someone else holds this".
            holder = "<unreadable>"
        print(f"another driver already holds {held}: {holder}\n"
              "Two drivers on one --out run all nine cells twice -- 66 "
              "GPU-hours instead of 33, racing on the same paths. If no run "
              f"is alive, delete {held} and start again.", flush=True)
        return EXIT_LOCKED
    try:
        return _run(args, jobs)
    finally:
        lock.unlink(missing_ok=True)


def _run(args, jobs: list[StudyJob]) -> int:
    """The loop itself, with `--out` already claimed by `acquire_lock`."""
    buffer = ReplayBuffer(args.data, capacity_transitions=BUFFER_CAPACITY)
    if not buffer.episode_paths():
        # Fail on the first second rather than the thirty-third hour: with no
        # episodes every one of the nine cells fails identically inside
        # `episode_split`, and the log would be nine copies of one mistake.
        print(f"no episodes in {args.data}; nothing can be trained", flush=True)
        return EXIT_NO_DATA

    failed: list[StudyJob] = []
    for index, job in enumerate(jobs, 1):
        print(f"\n===== [{index}/{len(jobs)}] {job.arm} seed {job.seed} "
              f"{datetime.now().isoformat(timespec='seconds')} =====", flush=True)
        started = time.perf_counter()
        try:
            record = run_job(job, buffer, args.out, steps=args.steps,
                             seq_len=args.seq_len, device=args.device)
        except Exception:
            # Not `BaseException`: KeyboardInterrupt and SystemExit mean the
            # human wants the whole run stopped, not this cell skipped. And
            # not a narrower tuple: a missing feature cache -- the policy's own
            # headline example -- is a `FileNotFoundError`.
            failed.append(job)
            print(f"  FAILED after {time.perf_counter() - started:.1f}s; "
                  "no record written, so a re-run picks this cell up again",
                  flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            continue
        print(job_summary(job, record, time.perf_counter() - started), flush=True)
        if not record_names_the_job(record, job):
            # The record at this cell's path describes another cell. Left
            # alone this is the worst outcome the study has: the aggregation
            # would report one arm's numbers under another's name, and the
            # resume check would keep the cell pending forever.
            failed.append(job)
            print(f"  MISLABELLED: the record says arm={_get(record, 'arm')!r} "
                  f"seed={_get(record, 'seed')!r}, not {job.arm!r}/{job.seed!r}",
                  flush=True)

    done = len(jobs) - len(failed)
    print(f"\n{done}/{len(jobs)} job(s) completed; records in {args.out}")
    if failed:
        # `failed`, not `jobs`: the morning's first question is which cells to
        # retry, and a list naming all nine answers it wrongly.
        print("FAILED: " + ", ".join(f"{j.arm}/s{j.seed}" for j in failed)
              + " -- re-run this script to retry exactly those cells")
        return EXIT_JOB_FAILED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
