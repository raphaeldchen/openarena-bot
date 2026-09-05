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
"""

import argparse
import json
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

The set is pinned against `run_job`'s actual output by
`test_a_real_record_from_run_job_is_recognised_as_complete`. Drift in either
direction is a disaster with no symptom: a key listed here that `run_job` never
writes makes EVERY cell permanently pending, so the study re-runs from zero on
every resume.
"""

EXIT_OK = 0
EXIT_JOB_FAILED = 1
EXIT_NO_DATA = 2


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


def record_is_complete(path, job: StudyJob) -> bool:
    """Whether `path` holds a finished record for `job`.

    False for: no file, a directory, an unreadable file, an empty file, JSON
    that does not parse, JSON that is not an object, an object missing any of
    `REQUIRED_RECORD_KEYS`, one carrying a non-standard NaN token, and one that
    names a different arm or seed. Every one of those means the cell has to be
    run; only a complete record for this exact cell is worth skipping.
    """
    try:
        text = Path(path).read_text()
    except OSError:
        return False
    try:
        record = json.loads(text, parse_constant=_reject_nonstandard)
    except ValueError:  # JSONDecodeError, and _reject_nonstandard's ValueError
        return False
    if not isinstance(record, dict):
        return False
    if not REQUIRED_RECORD_KEYS <= set(record):
        return False
    return record_names_the_job(record, job)


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
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    return parser


def main(argv=None) -> int:
    """Run every pending cell. Returns the process exit status.

    `context` and `horizon` are deliberately NOT flags here. `run_job` forwards
    one value to the probe fit, the rollout and both filtering diagnostics
    because they must not default apart, and a flag in this script would be a
    second home for the study's evaluation protocol -- the day `run_job`'s
    defaults move, the nine records would still carry this file's stale ones.
    """
    args = _parser().parse_args(argv)

    jobs = pending_jobs(args.out, tuple(args.arms), tuple(args.seeds))
    listing = ", ".join(f"{j.arm}/s{j.seed}" for j in jobs) or "none"
    print(f"{datetime.now().isoformat(timespec='seconds')} "
          f"{len(jobs)} job(s) pending: {listing}", flush=True)
    if not jobs:
        # Return before touching --data: a finished study must not depend on
        # the episodes still being on the box, and `ReplayBuffer` would create
        # the directory as a side effect of being asked.
        print(f"nothing to do; all records already in {args.out}")
        return EXIT_OK

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
            # human wants the whole run stopped, not this cell skipped.
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
        print("FAILED: " + ", ".join(f"{j.arm}/s{j.seed}" for j in failed)
              + " -- re-run this script to retry exactly those cells")
        return EXIT_JOB_FAILED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
