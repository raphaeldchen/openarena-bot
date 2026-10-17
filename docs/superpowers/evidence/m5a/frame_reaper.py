"""Ops watcher (user-approved 2026-10-15): delete a finished M5a run's replay
episode frames once its record verifies, its feature caches are flagged deleted
and state.pt is gone. Records are never touched."""
import sys, time
from pathlib import Path
from mbfps.training.online import OnlineConfig, verify_record
from mbfps.eval.efficiency import ARMS, SEEDS
from mbfps.eval.study import load_record

ROOT, CFG, LOG = Path("runs/m5a"), OnlineConfig(), Path(sys.argv[1])
done = set()

def log(msg):
    with LOG.open("a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M ") + msg + "\n")

while len(done) < len(ARMS) * len(SEEDS):
    for arm in ARMS:
        for seed in SEEDS:
            d = ROOT / f"{arm}_seed{seed}"
            if (arm, seed) in done or not (d / "record.json").exists() or (d / "state.pt").exists():
                continue
            rec = load_record(d / "record.json")
            problems = verify_record(rec, CFG)
            if rec.get("arm") != arm or rec.get("seed") != seed:
                problems.append("record identity differs from directory")
            if rec.get("replay_features_deleted") is not True:
                problems.append("features not flagged deleted")
            if list((d / "replay").glob("*.features*")) if (d / "replay").exists() else []:
                problems.append("feature files still present")
            if problems:
                continue          # not finished yet (or not deletable): look again next pass
            files = sorted((d / "replay").glob("*.npz")) if (d / "replay").exists() else []
            n = sum(f.stat().st_size for f in files)
            for f in files:
                f.unlink()
            leftovers = list((d / "replay").iterdir()) if (d / "replay").exists() else []
            if (d / "replay").exists() and not leftovers:
                (d / "replay").rmdir()
            log(f"{arm} seed {seed}: verified; deleted {len(files)} episode frame files "
                f"({n / 2**30:.2f} GiB); leftovers {[p.name for p in leftovers]}")
            done.add((arm, seed))
    time.sleep(120)
log("all ten runs reaped; exiting")
