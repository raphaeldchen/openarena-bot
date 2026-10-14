"""Stage C: the 3 x 3 study (M4 spec §6).

run    one arm at a time, so its feature cache can be cleared before the next
       arm's is written (one ~3 GB ViT cache fits; two may not). Every step is
       skipped when its output already exists, so an interrupted run resumes.
read   every cell's entry and live readings, and the §6.2 cross-arm reading:
       arm X ranks above arm Y only if every seed of X beats every seed of Y.

Exit codes (read): 0 all nine cells present and valid; 80 MISSING_CELLS; 81 INVALID_CELLS.
A cell is invalid when its live record lacks the final rung, when its episode seeds are
not `live["seeds"]` in order (random, untrained, and the final rung's sample and argmax),
when its training record's `isolation_ok` is not True (spec §5.5), or when its untrained
control passes the paired gate (spec §5.6: the evaluation is broken). Invalid cells are
listed on an INVALID_CELLS line and the §6.2 cross-arm reading is withheld. Missing cells
(80) take precedence: with both, the exit code is 80 and only MISSING_CELLS is printed.
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ARMS = ("frozen_ssl", "pixel_ae", "random_vit")
SEEDS = (0, 1, 2)
FINAL_STEP = 10_000
PIXEL_AE_BASIC = Path("runs/m2_basic/autoencoder_cnn.pt")
"""The autoencoder Stage C trains on `basic` and writes the pixel_ae cache with.
Live evaluation must encode with the same one (Task 12's --pixel-ae-checkpoint)."""
EXIT_OK, EXIT_MISSING_CELLS, EXIT_INVALID_CELLS = 0, 80, 81


def separation(scores: dict) -> dict:
    out = {}
    for a, b in itertools.combinations(sorted(scores), 2):
        sa, sb = np.asarray(scores[a], float), np.asarray(scores[b], float)
        if np.all(sa[:, None] > sb[None, :]):
            out[(a, b)] = "ABOVE"
        elif np.all(sb[:, None] > sa[None, :]):
            out[(a, b)] = "BELOW"
        else:
            out[(a, b)] = "NOT_SEPARATED"
    return out


def _paths(runs: Path, arm: str, seed: int) -> dict:
    agent = runs / "m4c_agent" / f"{arm}_seed{seed}"
    return {
        "world": runs / "m4c_world" / f"world_model_{arm}_seed{seed}.pt",
        "entry": runs / "m4c_entry" / f"entry_{arm}_seed{seed}.json",
        "agent": agent,
        "final": agent / f"actor_critic_{arm}_seed{seed}_step{FINAL_STEP}.pt",
        "train": agent / f"train_{arm}_seed{seed}.json",
        "live": agent / f"live_{arm}_seed{seed}.json",
    }


def _commands(runs: Path, data: Path, arm: str, seed: int) -> list[tuple[Path, list[str]]]:
    py, p = sys.executable, _paths(runs, arm, seed)
    live_extra = ["--pixel-ae-checkpoint", str(PIXEL_AE_BASIC)] if arm == "pixel_ae" else []
    return [
        (p["world"], [py, str(ROOT / "train_world_model.py"), "--arm", arm, "--seed", str(seed),
                      "--data", str(data), "--out", str(p["world"].parent), "--n-actions", "4",
                      "--pad-short", "--reward-transform", "symlog",
                      "--record", str(p["world"].parent / f"train_{arm}_seed{seed}.json")]),
        (p["entry"], [py, str(ROOT / "entry_check.py"), "--phase", "measure", "--role", "cell",
                      "--checkpoint", str(p["world"]), "--data", str(data),
                      "--out", str(p["entry"].parent)]),
        (p["final"], [py, str(ROOT / "train_actor_critic.py"), "--world-model", str(p["world"]),
                      "--data", str(data), "--out", str(p["agent"]), "--seed", str(seed)]),
        (p["live"], [py, str(ROOT / "eval_live.py"), "--phase", "measure",
                     "--world-model", str(p["world"]), "--agent-dir", str(p["agent"]),
                     "--out", str(p["agent"]), *live_extra]),
    ]


def run(args) -> int:
    for seed in args.seeds:
        for output, cmd in _commands(args.runs, args.data, args.arm, seed):
            if output.exists():
                print(f"skip (exists): {output}")
                continue
            print("run:", " ".join(cmd), flush=True)
            if args.phase == "plan":
                continue
            code = subprocess.call(cmd)
            if code != 0:
                print(f"FAILED ({code}): {' '.join(cmd)}")
                return code
    return EXIT_OK


def read(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mbfps.eval.live import paired_gate, seed_mismatches
    from mbfps.eval.study import load_record

    missing, invalid, scores, lines = [], [], {arm: [] for arm in ARMS}, ["M4 Stage C -- 3 arms x 3 seeds"]
    for arm in ARMS:
        for seed in SEEDS:
            p = _paths(args.runs, arm, seed)
            absent = [k for k in ("entry", "live", "train") if not p[k].is_file()]
            if absent:
                missing.append(f"{arm}/seed{seed}: {absent}")
                continue
            entry, live, train = load_record(p["entry"]), load_record(p["live"]), load_record(p["train"])
            cell = f"{arm}/seed{seed}"
            if str(FINAL_STEP) not in live["rungs"]:
                invalid.append(f"{cell}: no rung {FINAL_STEP} in {sorted(live['rungs'], key=int)}")
                lines.append(f"  {arm:<11s} s{seed}  INVALID: no rung {FINAL_STEP}")
                continue
            final = live["rungs"][str(FINAL_STEP)]["sample"]
            rand = [e["return"] for e in live["random"]]
            gate = paired_gate([e["return"] for e in final], rand)
            control = paired_gate([e["return"] for e in live["untrained"]], rand)
            score = float(np.mean([e["return"] for e in final]))
            iv = entry["imagined"]
            reasons = []
            if (bad := seed_mismatches(live, steps=[str(FINAL_STEP)])):
                reasons.append(f"episode seeds differ from the record's seeds in {bad}")
            if train.get("isolation_ok") is not True:
                reasons.append(f"isolation_ok is {train.get('isolation_ok')}")
            if control["passed"]:
                reasons.append("untrained control passes the gate")
            if reasons:
                invalid.append(f"{cell}: " + ", ".join(reasons))
            else:
                scores[arm].append(score)
            lines.append(
                f"  {arm:<11s} s{seed}  score {score:+8.2f}  gate {gate['point']:+7.2f} "
                f"[{gate['ci_low']:+7.2f},{gate['ci_high']:+7.2f}] {'PASS' if gate['passed'] else 'fail'}"
                f"  control {'BROKEN' if control['passed'] else 'ok'}"
                f"  isolation {train.get('isolation_ok')}"
                f"  entry {entry['verdict']} (R {iv['return']['ci_low']:+.3f}, C {iv['continue']['ci_low']:+.3f})"
                + (f"  INVALID: {'; '.join(reasons)}" if reasons else ""))
    if missing:
        lines.append("MISSING_CELLS: " + "; ".join(missing))
    elif invalid:
        lines.append("INVALID_CELLS: " + "; ".join(invalid))
    else:
        lines.append("cross-arm reading (9 of 9 pairwise wins to rank):")
        for (a, b), verdict in separation(scores).items():
            lines.append(f"  {a} vs {b}: {verdict}")
    out = args.runs / "m4c_study"
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (out / "study.txt").write_text(text)
    print(text, end="")
    return EXIT_MISSING_CELLS if missing else EXIT_INVALID_CELLS if invalid else EXIT_OK


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("plan", "run", "read"), required=True)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--data", type=Path, default=Path("data/basic"))
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=list(SEEDS))
    args = p.parse_args(argv)
    if args.phase in ("plan", "run"):
        if args.arm is None:
            raise SystemExit("--phase plan/run needs --arm (one arm per invocation)")
        return run(args)
    return read(args)


if __name__ == "__main__":
    sys.exit(main())
