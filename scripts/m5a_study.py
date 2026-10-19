"""M5a study: run the ten runs; read the pre-registered result (spec §7).

Exit codes (read):
  0   all ten runs present and valid; the reading is printed (any verdict)
  90  MISSING_RUNS   a run's record is absent or incomplete
  91  INVALID_RUNS   a run fails §7.3; the reading is not printed
Missing takes precedence over invalid."""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from mbfps.eval.efficiency import (ARMS, SEEDS, auc, gated_returns, pairwise_wins,
                                   run_problems, steps_to_threshold, verdict)
from mbfps.eval.study import load_record
from mbfps.training.online import OnlineConfig, checkpoint_steps, verify_record

ROOT = Path(__file__).resolve().parent
EXIT_OK, EXIT_MISSING_RUNS, EXIT_INVALID_RUNS = 0, 90, 91


def _record_path(runs: Path, arm: str, seed: int) -> Path:
    return runs / f"{arm}_seed{seed}" / "record.json"


def _done(path: Path) -> bool:
    """A verified record whose resume state is gone is finished: train_online
    refuses to start over it, so it must never be relaunched. A verified
    record whose state.pt remains is relaunched, to resume and delete."""
    if not path.exists():
        return False
    return not verify_record(load_record(path), OnlineConfig()) \
        and not (path.parent / "state.pt").exists()


def run(args) -> int:
    for seed in args.seeds:
        for arm in args.arms:
            path = _record_path(args.runs, arm, seed)
            if _done(path):
                print(f"skip (done): {arm} seed {seed}")
                continue
            cmd = [sys.executable, str(ROOT / "train_online.py"), "--arm", arm,
                   "--seed", str(seed), "--out", str(path.parent)]
            print("run:", " ".join(cmd), flush=True)
            code = subprocess.call(cmd)
            if code != 0:
                print(f"FAILED ({code}): {arm} seed {seed}")
                return code
    return EXIT_OK


def read(args) -> int:
    cfg = OnlineConfig()
    valid, missing, invalid = {}, [], []
    for arm in ARMS:
        for seed in SEEDS:
            path = _record_path(args.runs, arm, seed)
            if not path.exists():
                missing.append(f"{arm}_seed{seed}")
                continue
            rec = load_record(path)
            if rec.get("complete") is not True:
                missing.append(f"{arm}_seed{seed} (incomplete)")
                continue
            problems = run_problems(rec, cfg, arm, seed)
            if problems:
                invalid.append(f"{arm}_seed{seed}: " + "; ".join(problems))
            else:
                valid[(arm, seed)] = rec

    lines = ["M5a -- online Dreamer vs DQN on basic, 100,000 steps x 5 seeds"]
    for (arm, seed), rec in sorted(valid.items()):
        curve = " ".join(f"{np.mean(gated_returns(rec, s)):+7.1f}" for s in checkpoint_steps(cfg))
        lines.append(f"  {arm:<7s} s{seed}  AUC {auc(rec, cfg):+8.2f}  "
                     f"steps-to-+50 {steps_to_threshold(rec, cfg) or '>100k'}  "
                     f"rounds {rec.get('update_rounds')}  seconds {rec.get('seconds', 0):.0f}")
        lines.append(f"      curve(0..100k): {curve}")
    if missing:
        lines.append("MISSING_RUNS: " + "; ".join(missing))
        code = EXIT_MISSING_RUNS
    elif invalid:
        lines.append("INVALID_RUNS: " + "; ".join(invalid))
        code = EXIT_INVALID_RUNS
    else:
        d = [auc(valid[("dreamer", s)], cfg) for s in SEEDS]
        q = [auc(valid[("dqn", s)], cfg) for s in SEEDS]
        lines += [f"Dreamer AUCs: {', '.join(f'{x:+.2f}' for x in d)}",
                  f"DQN AUCs:     {', '.join(f'{x:+.2f}' for x in q)}",
                  f"Dreamer wins {pairwise_wins(d, q):g} of 25 (bar 21; ties count 1/2)",
                  f"median AUC difference (Dreamer - DQN): {np.median(d) - np.median(q):+.2f}",
                  f"VERDICT: {verdict(d, q)}"]
        code = EXIT_OK
    args.runs.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (args.runs / "study.txt").write_text(text)
    print(text, end="")
    return code


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("run", "read"), required=True)
    p.add_argument("--runs", type=Path, default=Path("runs/m5a"))
    p.add_argument("--arms", type=lambda s: s.split(","), default=list(ARMS))
    p.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=list(SEEDS))
    args = p.parse_args(argv)
    return run(args) if args.phase == "run" else read(args)


if __name__ == "__main__":
    sys.exit(main())
