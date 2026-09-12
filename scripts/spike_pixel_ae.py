"""The M3c spike: 2,000 steps of `pixel_ae`, and the three checks that license the study.

Spec section 3. The nine-cell study (13.5 h) runs only if one 2,000-step
`pixel_ae` run at seed 0 shows all three of:

  1. `embedding_loss` at step 0 in [0.1, 1.0]      -- the target is informative.
     The retired end-to-end pixel arm read 0.0008: a randomly initialised CNN
     through the bottleneck emits a near-constant embedding, predicting it is
     trivial, nothing forces the posterior to encode anything, and the run is
     at a fixed point from step 0. The feature arms read 0.3165 (frozen_ssl)
     and 0.4104 (random_vit) -- that is the band.
  2. `kl_rate_above_free_bits` > 0.5               -- the dynamics prior trains.
     `train_world_model` already warns below 0.5 on `_kl_rate`'s own
     criterion; here it is a verdict. M3b's pixel arm cleared the floor on 15
     to 18 of 20,000 steps, so `prior_net` was never trained and `imagine()`
     ran on its initialisation.
  3. held-out linear-probe R^2 on the CACHED pixel_ae features > 0
                                                   -- the features carry position.
     Measured on the ViT caches at spec 3.2a: DINOv2 0.422, random_vit 0.369.
     The M3b pixel arm's probe collapsed to a constant at -0.036.

WHY THIS IS A SCRIPT AND NOT A SHELL NOTE. M3b calibrated `KL_FREE_BITS` on a
2,000-step `random_vit` run and then judged the pixel arm's failure by eye
from checkpoint diffs, because nothing had kept the per-step history. The
three criteria are functions here, the verdict is an exit status, and the
record on disk carries `history["loss"]` and `history["parts"]` IN FULL
(M3b write-up open item 6) beside the git SHA and device that produced it.

WHICH PROBE. Check 3 asks about the cache itself, before any trainable layer.
`fit_probes` and `gather_probe_data` take a model and probe what the MODEL
produces (its encoder output, its latent, its predicted embedding), so they
answer a different question; this script uses the array-level `fit_probe`,
`probe_r2` and `probe_targets` from the same module directly. The (64, 32)
rows are flattened to the encoder's own 2048-vector: spec 2.1 says the
partition is a reshape that invents nothing, so a patch-mean -- 3.2a's
reduction for the ViT caches, where the 64 rows are 64 real patches -- would
average unrelated dimensions here. The split mirrors `fit_probes(limit=20,
select_episodes=4)`: weights on 16 training episodes, ridge selected on the
next 4, scored on the first 20 validation episodes of the study's own split.

THE SIDE-BY-SIDE. Check 1's band is only meaningful next to the arms it was
taken from, so the script also prints the step-0 embedding loss for
`frozen_ssl` and `random_vit`, from one forward pass each with no training.
`step0_embedding_loss` is `train_world_model` run for exactly one step with
nothing saved: the same seed, split, loader draw and posterior sample by
construction, and tested equal to `history["parts"][0]["embedding"]` on every
arm -- a copy of the setup block would have to be kept in step by hand.

EXIT STATUS: 0 iff all three checks pass, 10 otherwise. 10 is
`report_study.py`'s `EXIT_GATE_NOT_PASSED` and means the same thing here -- a
verdict that was computed and printed in full, not an error. Anything else
(no data, a missing cache) is an uncaught traceback and exit 1; this spike is
attended, unlike the study driver, and a traceback is the right report.
"""

import argparse
import contextlib
import io
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import load_episode
from mbfps.data.features import BACKBONE_GEOMETRY
from mbfps.data.loader import feature_suffix
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.probe import fit_probe, probe_r2, probe_targets
from mbfps.eval.study import SPLIT_SEED, write_record
from mbfps.models.encoders import encoder_backbone
from mbfps.models.rssm import KL_FREE_BITS
from mbfps.training.world_model import train_world_model
from mbfps.utils.config import ARMS, get_config
from mbfps.utils.device import get_device

SPIKE_ARM = "pixel_ae"
SPIKE_SEED = 0
SPIKE_STEPS = 2_000
"""One arm, one seed, 2,000 steps -- spec section 3. Constants, not flags: a
spike on any other arm passes its own checks and licenses nothing about this
one, and the seed is the study's seed 0 so the run is one of its nine cells
cut short, not a tenth configuration."""

EMBEDDING_LOSS_BAND: tuple[float, float] = (0.1, 1.0)
"""Closed band for check 1. The feature arms measured 0.3165 and 0.4104 at
step 0; the retired pixel arm 0.0008 (M3c design section 1)."""

MIN_KL_RATE = 0.5
"""Strict floor for check 2 -- the same 0.5 `train_world_model` warns below."""

MIN_PROBE_R2 = 0.0
"""Strict floor for check 3. Exactly zero is the constant predictor."""

PROBE_LIMIT = 20
PROBE_SELECT_EPISODES = 4
"""`fit_probes`' own split: the first `PROBE_LIMIT` training episodes, of
which the last `PROBE_SELECT_EPISODES` select the ridge; scored on the first
`PROBE_LIMIT` validation episodes."""

CHECK_KEYS: tuple[str, ...] = (
    "embedding_loss_step0", "kl_rate_above_free_bits", "probe_r2_cached_features",
)
"""The three checks, in spec order. `all_passed` and `report` read this."""

RUN_STUDY = "RUN_STUDY"
STOP_RECALIBRATE_FREE_BITS = "STOP_RECALIBRATE_FREE_BITS"
STOP_BACKBONE_UNINFORMATIVE = "STOP_BACKBONE_UNINFORMATIVE"
DECISION_TEXT = {
    RUN_STUDY: "all three checks pass; run the nine cells (Task 8)",
    STOP_RECALIBRATE_FREE_BITS: (
        "checks 1 and 3 pass but the prior did not train; STOP and re-plan "
        "KL_FREE_BITS as a three-arm calibration with the KL trajectory "
        "recorded per step (spec section 3)"),
    STOP_BACKBONE_UNINFORMATIVE: (
        "check 1 or 3 failed: the frozen backbone is not informative, and no "
        "free-bits value creates a signal it does not have (spec section 1); "
        "STOP, do not run the study"),
}

EXIT_OK = 0
EXIT_SPIKE_FAILED = 10


# --------------------------------------------------------------------------
# the three checks
# --------------------------------------------------------------------------

def check_embedding_loss(parts: list[dict]) -> dict:
    """Check 1 on `history["parts"]`: the FIRST step's embedding loss.

    The first, not the last or the mean: section 1's failure is a target that
    is born collapsed, and from 0.0008 the loss can only fall further.
    """
    if not parts:
        raise ValueError("history has no steps; the step-0 embedding loss does not exist")
    value = float(parts[0]["embedding"])
    low, high = EMBEDDING_LOSS_BAND
    # `low <= value <= high` is False for NaN, which is the right answer.
    return {"value": value, "low": low, "high": high, "passed": bool(low <= value <= high)}


def check_kl_rate(history: dict) -> dict:
    """Check 2 on the RATE, never `kl_dyn_max`: cnn/s1 peaked at 0.800 nat
    while clearing the floor on 16 of 20,000 steps."""
    value = float(history["kl_rate_above_free_bits"])
    return {"value": value, "threshold": MIN_KL_RATE, "passed": bool(value > MIN_KL_RATE)}


def check_probe_r2(r2: float) -> dict:
    """Check 3: held-out R^2 on the cached features, strictly positive."""
    value = float(r2)
    return {"value": value, "threshold": MIN_PROBE_R2, "passed": bool(value > MIN_PROBE_R2)}


def evaluate_checks(history: dict, feature_probe_r2: float) -> dict:
    """All three, keyed by `CHECK_KEYS`, each judged from its own input only."""
    return {
        "embedding_loss_step0": check_embedding_loss(history["parts"]),
        "kl_rate_above_free_bits": check_kl_rate(history),
        "probe_r2_cached_features": check_probe_r2(feature_probe_r2),
    }


def all_passed(checks: dict) -> bool:
    """A conjunction over `CHECK_KEYS`; a missing check is an error, not a pass."""
    missing = [key for key in CHECK_KEYS if key not in checks]
    if missing:
        raise ValueError(f"checks are missing {missing}; a check that was never run cannot pass")
    return all(bool(checks[key]["passed"]) for key in CHECK_KEYS)


def decision(checks: dict) -> str:
    """Spec section 3's decision rule.

    The backbone verdict DOMINATES: if check 1 or 3 fails, the target is not
    informative and section 1 shows that no `KL_FREE_BITS` value can create a
    learning signal the arm does not have -- so the recalibration route is not
    offered even when check 2 failed too. Only a lone check-2 failure sends
    the floor back for a genuine three-arm calibration.
    """
    if not (checks["embedding_loss_step0"]["passed"]
            and checks["probe_r2_cached_features"]["passed"]):
        return STOP_BACKBONE_UNINFORMATIVE
    if not checks["kl_rate_above_free_bits"]["passed"]:
        return STOP_RECALIBRATE_FREE_BITS
    return RUN_STUDY


def exit_status(checks: dict) -> int:
    return EXIT_OK if all_passed(checks) else EXIT_SPIKE_FAILED


# --------------------------------------------------------------------------
# the side-by-side
# --------------------------------------------------------------------------

def step0_embedding_loss(cfg, buffer: ReplayBuffer) -> float:
    """`train_world_model`'s step-0 embedding loss for `cfg`, from `train_world_model`.

    One step of the real training loop with nothing saved. The seed, the split,
    the loader draw and the posterior sample are the ones the check judges
    BECAUSE THIS IS THAT CODE PATH -- there is no copy of the setup block to
    keep in step with `world_model.py`. The price is one backward pass and one
    Adam step per arm, which is seconds. The one-step run's own "kl_dyn
    exceeded the floor on only x% of steps" warning is meaningless at a single
    step and is swallowed; the 2,000-step run's warning is not.
    """
    one_step = replace(cfg, train=replace(cfg.train, steps=1))
    with contextlib.redirect_stdout(io.StringIO()):
        history = train_world_model(one_step, buffer, out_dir=None, log_every=0)
    return float(history["parts"][0]["embedding"])


# --------------------------------------------------------------------------
# check 3's probe on the cache
# --------------------------------------------------------------------------

def load_cached_features(path: Path, backbone: str) -> np.ndarray:
    """`(T+1, n_patches * patch_dim)` float64 rows of `backbone`'s cache for one episode.

    Validated against `BACKBONE_GEOMETRY` by name, like the loader: a cache
    at another geometry is refused here rather than flattened into a probe of
    the wrong width that scores something.
    """
    n_patches, patch_dim = BACKBONE_GEOMETRY[backbone]
    feature_path = path.with_suffix(feature_suffix(backbone))
    features = np.load(feature_path)
    if tuple(features.shape[1:]) != (n_patches, patch_dim):
        raise ValueError(
            f"{feature_path.name} holds rows of shape {tuple(features.shape[1:])}, "
            f"but backbone {backbone!r} caches {(n_patches, patch_dim)}"
        )
    return features.reshape(features.shape[0], n_patches * patch_dim).astype(np.float64)


def _rows(paths, backbone: str) -> tuple[np.ndarray, np.ndarray]:
    """Row-aligned `(features, targets)` over `paths`, one row per FRAME."""
    xs, ys = [], []
    for path in paths:
        episode = load_episode(path)
        x = load_cached_features(path, backbone)
        y = probe_targets(episode.privileged, episode.privileged_keys)
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"{path.name}: the {backbone} cache has {x.shape[0]} rows against "
                f"{y.shape[0]} frames; a silent truncation would fit on shifted frames"
            )
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def cached_feature_probe(
    train_paths, val_paths, backbone: str,
    limit: int = PROBE_LIMIT, select_episodes: int = PROBE_SELECT_EPISODES,
) -> dict:
    """Held-out linear-probe R^2 on `backbone`'s cache -- check 3.

    Split at EPISODE granularity, like `fit_probes`: the first `limit`
    training episodes, of which the last `select_episodes` select the ridge
    (consecutive frames are near-duplicates, so a row-wise split would put the
    same scene on both sides and every ridge would look equal); scored on the
    first `limit` VALIDATION episodes, which neither the weights nor the
    selection saw. The sizes actually used are returned, because on a small
    buffer they are not 16 / 4 / 20 and the record must say what they were.
    """
    used = list(train_paths)[:limit]
    scored = list(val_paths)[:limit]
    spare = len(used) - select_episodes
    if spare < 1:
        raise ValueError(
            f"{len(used)} training episodes leave none to fit on after holding "
            f"{select_episodes} out for ridge selection"
        )
    if not scored:
        raise ValueError("no validation episodes to score the probe on")

    x_fit, y_fit = _rows(used[:spare], backbone)
    x_select, y_select = _rows(used[spare:], backbone)
    x_val, y_val = _rows(scored, backbone)
    probe = fit_probe(x_fit, y_fit, x_select, y_select)
    return {
        "backbone": backbone,
        "r2": float(probe_r2(probe, x_val, y_val)),
        "ridge": float(probe["ridge"]),
        "selection_r2": float(probe["r2"]),
        "n_fit_episodes": spare,
        "n_select_episodes": len(used) - spare,
        "n_scored_episodes": len(scored),
        "n_scored_rows": int(y_val.shape[0]),
        "n_columns": int(x_fit.shape[1]),
    }


# --------------------------------------------------------------------------
# record, report, main
# --------------------------------------------------------------------------

def git_sha() -> str:
    """`git rev-parse HEAD`, or "unknown". Never raises: a missing `git` on the
    box must not be the thing that loses a 10-minute run's record."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out or "unknown"


def spike_record_path(out_dir) -> Path:
    return Path(out_dir) / f"spike_{SPIKE_ARM}_seed{SPIKE_SEED}.json"


def _verdict(check: dict) -> str:
    return "PASS" if check["passed"] else "FAIL"


def report(record: dict) -> str:
    """The block to paste into the plan's results section, verbatim."""
    checks = record["checks"]
    c1, c2, c3 = (checks[key] for key in CHECK_KEYS)
    side = record["embedding_loss_step0"]
    probe = record["probe"]
    lines = [
        f"=== M3c spike: {record['arm']} seed {record['seed']}, {record['steps']} steps "
        f"on {record['device']} at {record['git_sha']} ===",
        "embedding_loss_step0  "
        + "  ".join(f"{arm}={side[arm]:.4f}" for arm in ARMS)
        + "   (M3c design section 1 measured cnn=0.0008 frozen_ssl=0.3165 random_vit=0.4104)",
        f"training              steps_per_second={record['steps_per_second']:.2f} "
        f"kl_dyn_max={record['kl_dyn_max']:.4f}",
        f"check 1  embedding_loss_step0       {c1['value']:.4f}  "
        f"in [{c1['low']}, {c1['high']}]  {_verdict(c1)}",
        f"check 2  kl_rate_above_free_bits    {c2['value']:.4f}  "
        f"> {c2['threshold']}  {_verdict(c2)}   (floor KL_FREE_BITS={KL_FREE_BITS})",
        f"check 3  probe_r2_cached_features   {c3['value']:+.4f}  "
        f"> {c3['threshold']}  {_verdict(c3)}   "
        f"(ridge {probe['ridge']:g}, fit {probe['n_fit_episodes']} / select "
        f"{probe['n_select_episodes']} / scored {probe['n_scored_episodes']} val "
        f"episodes, {probe['n_scored_rows']} rows x {probe['n_columns']} columns)",
        f"decision: {record['decision']} -- {DECISION_TEXT[record['decision']]}",
    ]
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # No --arm and no --seed, on purpose; see SPIKE_ARM.
    parser.add_argument("--data", default="data/my_way_home")
    parser.add_argument("--out", default="runs/m3c_spike")
    parser.add_argument("--steps", type=int, default=SPIKE_STEPS)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="mps")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    out_dir = Path(args.out)
    overrides = dict(steps=args.steps, seq_len=args.seq_len,
                     batch_size=args.batch_size, device=args.device)

    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    train_paths, val_paths = episode_split(
        buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED
    )
    cfg = get_config(SPIKE_ARM, seed=SPIKE_SEED, **overrides)
    backbone = encoder_backbone(cfg.encoder)

    history = train_world_model(cfg, buffer, out_dir=out_dir)
    probe = cached_feature_probe(train_paths, val_paths, backbone)
    # Every arm, the spike's included: the pixel_ae value here is tested equal
    # to history["parts"][0]["embedding"] on CPU, and printing both on MPS is
    # what would show a device-level discrepancy if one ever appeared.
    side_by_side = {
        arm: step0_embedding_loss(get_config(arm, seed=SPIKE_SEED, **overrides), buffer)
        for arm in ARMS
    }
    checks = evaluate_checks(history, probe["r2"])

    record = {
        "arm": SPIKE_ARM,
        "seed": SPIKE_SEED,
        "steps": args.steps,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "split_seed": SPLIT_SEED,
        "git_sha": git_sha(),
        "device": str(get_device(prefer=args.device)),
        "seconds": float(history["seconds"]),
        "steps_per_second": float(history["steps"] / history["seconds"]),
        "kl_rate_above_free_bits": float(history["kl_rate_above_free_bits"]),
        "kl_dyn_max": float(history["kl_dyn_max"]),
        # IN FULL. Open item 6 of the M3b write-up: section 1 of the M3c design
        # had to be reconstructed because no run kept this.
        "history": {
            "loss": [float(v) for v in history["loss"]],
            "parts": [{k: float(v) for k, v in p.items()} for p in history["parts"]],
        },
        "embedding_loss_step0": side_by_side,
        "probe": probe,
        "checks": checks,
        "decision": decision(checks),
        "episodes": {
            "train": [p.name for p in train_paths],
            "val": [p.name for p in val_paths],
        },
    }
    # `write_record`, not `json.dump`: a `kl_rep` of NaN in one of 2,000 parts
    # is a bare `NaN` token no strict parser reads, and the write is atomic.
    write_record(spike_record_path(out_dir), record)
    print(report(record), flush=True)
    print(f"record: {spike_record_path(out_dir)}", flush=True)
    return exit_status(checks)


if __name__ == "__main__":
    sys.exit(main())
