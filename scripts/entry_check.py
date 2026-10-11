"""Stage A's entry check (M4 spec §4.5-§4.7).

    measure --role cell     one world model over its data's held-out split
    measure --role control  §4.6: the M3 my_way_home model; must FAIL Reading R
    read                    the verdict, refusing without the control

Exit codes (read):
  0   ENTRY_PASS        both readings' intervals lie above zero
  60  HEAD_OR_DATA      a reading fails and its posterior reference fails too
  61  IMAGINATION       a reading fails and its posterior reference passes
  62  CONTROL_BROKEN    the control's return reading passed: nothing is read
  11  MISSING           a record the read needs is absent
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import load_episode
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.entry import (
    HORIZON, RESAMPLES, constant_baselines, continue_reading, control_broken,
    entry_verdict, episode_arrays, return_reading,
)
from mbfps.eval.rollout import source_for
from mbfps.eval.study import git_sha, load_record, write_record
from mbfps.models.encoders import encoder_backbone
from mbfps.training.world_model import load_world_model
from mbfps.utils.device import get_device
from mbfps.utils.seeding import fork_seed

EXIT_OK = 0
EXIT_MISSING = 11
EXIT_HEAD_OR_DATA = 60
EXIT_IMAGINATION = 61
EXIT_CONTROL_BROKEN = 62
READ_EXITS = {"ENTRY_PASS": EXIT_OK, "HEAD_OR_DATA": EXIT_HEAD_OR_DATA,
              "IMAGINATION": EXIT_IMAGINATION}
SPLIT_SEED = 0
BOOTSTRAP_SEED = 0


def record_path(out: Path, role: str, arm: str, seed: int) -> Path:
    return out / ("entry_control.json" if role == "control" else f"entry_{arm}_seed{seed}.json")


def _readings(stack: dict, labels, c_r, c_c, resamples) -> tuple[dict, dict]:
    imagined = {
        "return": return_reading(stack["imagined_reward"], stack["real_reward"], stack["valid"],
                                 labels, c_r, resamples, BOOTSTRAP_SEED),
        "continue": continue_reading(stack["imagined_continue"], stack["real_continue"],
                                     stack["valid"], labels, c_c, resamples, BOOTSTRAP_SEED),
    }
    posterior = {
        "return": return_reading(stack["posterior_reward"], stack["real_reward"], stack["valid"],
                                 labels, c_r, resamples, BOOTSTRAP_SEED),
        "continue": continue_reading(stack["posterior_continue"], stack["real_continue"],
                                     stack["valid"], labels, c_c, resamples, BOOTSTRAP_SEED),
    }
    return imagined, posterior


def measure(args) -> int:
    start = time.perf_counter()
    device = get_device(prefer=args.device)
    model = load_world_model(args.checkpoint, device)
    arm, seed = model.cfg.arm, model.cfg.train.seed
    backbone = encoder_backbone(model.cfg.encoder) or "dinov2"
    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    train_paths, val_paths = episode_split(buffer.episode_paths(),
                                           val_fraction=VAL_FRACTION, seed=SPLIT_SEED)

    train_eps = []
    for path in train_paths:
        with np.load(path) as d:
            train_eps.append((d["rewards"], d["terminated"], d["truncated"]))
    c_r, c_c = constant_baselines(train_eps, HORIZON)

    fields = ("imagined_reward", "imagined_continue", "posterior_reward",
              "posterior_continue", "real_reward", "real_continue", "valid")
    rows = {k: [] for k in fields}
    labels, kills, terminations = [], 0, 0
    for label, path in enumerate(val_paths):
        ep = load_episode(path)
        if ep.length < 2:
            continue
        a = episode_arrays(
            model, source_for(model, path, ep, backbone), ep.actions, ep.rewards,
            ep.terminated, ep.truncated, device,
            seed=fork_seed(seed, f"entry:{path.name}"),
        )
        for k in fields:
            rows[k].append(getattr(a, k))
        labels.append(np.full(ep.length - 1, label))
        kills += int(np.sum(ep.rewards > 50.0))
        terminations += int(ep.terminated.any())
    stack = {k: np.concatenate(v) for k, v in rows.items()}
    labels = np.concatenate(labels)
    imagined, posterior = _readings(stack, labels, c_r, c_c, args.resamples)

    position = None
    if not args.no_position:
        from mbfps.eval.probe import fit_probes
        from mbfps.eval.rollout import evaluate_rollout

        _, embedding_probe = fit_probes(model, train_paths, backbone, device,
                                        context=5, horizon=HORIZON, seed=seed)
        rollout = evaluate_rollout(model, val_paths, embedding_probe, context=5,
                                   horizon=HORIZON, seed=seed, device=device,
                                   feature_backbone=backbone)
        position = float(rollout.position_gap_closed()[-1])

    record = {
        "arm": arm, "seed": seed, "role": args.role,
        "checkpoint": str(args.checkpoint), "data": str(args.data),
        "reward_transform": model.cfg.train.reward_transform,
        "n_actions": model.cfg.train.n_actions,
        "horizon": HORIZON, "resamples": args.resamples,
        "c_R": c_r, "c_C": c_c, "imagined": imagined, "posterior": posterior,
        "counts": {"episodes": int(np.unique(labels).size), "starts": int(labels.size),
                   "slots": int(stack["valid"].sum()), "kills": kills,
                   "terminations": terminations},
        "position_gap_closed_h15": position,
        "git_sha": git_sha(), "seconds": time.perf_counter() - start,
    }
    if args.role == "cell":
        record["verdict"] = entry_verdict(imagined, posterior)
    write_record(record_path(args.out, args.role, arm, seed), record)
    print(f"wrote {record_path(args.out, args.role, arm, seed)}")
    return EXIT_OK


def _line(name: str, iv: dict) -> str:
    return (f"  {name:<20s} {iv['point']:+.4f} [{iv['ci_low']:+.4f}, {iv['ci_high']:+.4f}]"
            f"  rows={iv['n_rows']} episodes={iv['n_episodes']}")


def read(args) -> int:
    cell_path = record_path(args.out, "cell", args.arm, args.seed)
    control_path = record_path(args.out, "control", args.arm, args.seed)
    missing = [p for p in (cell_path, control_path) if not p.is_file()]
    if missing:
        print(f"MISSING: {[str(p) for p in missing]}")
        return EXIT_MISSING
    cell, control = load_record(cell_path), load_record(control_path)
    broken = control_broken(control["imagined"]["return"])
    lines = [
        f"M4 Stage A entry check -- {args.arm} seed {args.seed}",
        f"horizon {cell['horizon']}, resamples {cell['resamples']}, "
        f"c_R {cell['c_R']:+.3f}, c_C {cell['c_C']:.4f}",
        f"counts {cell['counts']}",
        "imagined (GATED):",
        _line("return R^2", cell["imagined"]["return"]),
        _line("continue Brier skill", cell["imagined"]["continue"]),
        "posterior reference (reported):",
        _line("return R^2", cell["posterior"]["return"]),
        _line("continue Brier skill", cell["posterior"]["continue"]),
        f"position gap_closed at h=15 (reported): {cell['position_gap_closed_h15']}",
        "control (M3 my_way_home model; Reading R must FAIL):",
        _line("return R^2", control["imagined"]["return"]),
        _line("continue Brier skill", control["imagined"]["continue"]),
    ]
    verdict = "CONTROL_BROKEN" if broken else cell["verdict"]
    lines.append(f"VERDICT: {verdict}")
    text = "\n".join(lines) + "\n"
    (args.out / "entry.txt").write_text(text)
    print(text, end="")
    return EXIT_CONTROL_BROKEN if broken else READ_EXITS[cell["verdict"]]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("measure", "read"), required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--out", type=Path, default=Path("runs/m4a_entry"))
    p.add_argument("--role", choices=("cell", "control"), default="cell")
    p.add_argument("--resamples", type=int, default=RESAMPLES)
    p.add_argument("--no-position", action="store_true")
    p.add_argument("--device", default="mps")
    p.add_argument("--arm")
    p.add_argument("--seed", type=int)
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.phase == "measure":
        if args.checkpoint is None or args.data is None:
            raise SystemExit("--phase measure needs --checkpoint and --data")
        return measure(args)
    if args.arm is None or args.seed is None:
        raise SystemExit("--phase read needs --arm and --seed")
    return read(args)


if __name__ == "__main__":
    sys.exit(main())
