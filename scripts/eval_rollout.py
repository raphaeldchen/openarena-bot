"""Fit the probe on train episodes, evaluate rollouts on held-out ones."""

import argparse
import json
from pathlib import Path

import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import episode_split
from mbfps.eval.probe import filtering_report, fit_probes
from mbfps.eval.rollout import evaluate_rollout
from mbfps.eval.summary import DEGENERATE, METRICS, metric_summary
from mbfps.models.encoders import encoder_backbone
from mbfps.training.world_model import WorldModel
from mbfps.utils.config import ARMS, get_config
from mbfps.utils.device import get_device

# Thin alias so the script reads naturally; the maths lives in the library
# because Task 4's study job needs the identical computation.
band_report = metric_summary


def print_report(result, reports: dict) -> None:
    """Print both metrics with identical guards."""
    units = {"position": "Doom map units", "angle": "degrees"}
    for metric in METRICS:
        r = reports[metric]
        print(f"\n--- {metric} ({units[metric]}) ---")
        print(f"{metric}_final          rssm={r['final_model']:9.2f} "
              f"persistence={r['final_persistence']:9.2f} floor={r['final_floor']:9.2f}")
        print(f"{metric}_band_width     min={r['band_min']:9.3f} "
              f"median={r['band_median']:9.3f} max={r['band_max']:9.3f}")
        print(f"{metric}_band_relative  min={r['relative_min']:.5f} "
              f"median={r['relative_median']:.5f} max={r['relative_max']:.5f}")
        print(f"{metric}_steps_with_floor_above_persistence="
              f"{r['steps_floor_above_persistence']}/{r['n_steps']}")
        print(f"{metric}_steps_with_degenerate_positive_band="
              f"{r['steps_degenerate']}/{r['n_steps']}  (band < {DEGENERATE} x persistence)")
        print(f"{metric}_gap_closed     finite={r['gap_finite']}/{r['n_steps']} "
              f"mean={r['gap_mean']:+.4f} min={r['gap_min']:+.4f} max={r['gap_max']:+.4f} "
              f"final={r['gap_final']:+.4f}")
        if r["gap_finite"] == 0:
            print(f"WARNING: {metric} gap_closed is NaN at every horizon step -- the band "
                  "is non-positive throughout, so this metric says nothing here. Report "
                  "the raw curves instead.")
        elif r["steps_degenerate"]:
            print(f"WARNING: {metric} band is numerically degenerate at "
                  f"{r['steps_degenerate']}/{r['n_steps']} steps; the ratio is unstable "
                  "there and the raw curves should be read directly.")


def print_filtering(filtering: dict) -> None:
    """Print spec section 4's fourth gate criterion.

    A `False` verdict is a RESULT, not a failure of the harness: it says the
    posterior latent explains the privileged state no better than the raw
    encoding of the same frame does, and since the posterior has already seen
    that frame, the only thing it could have added is history. So `False`
    means the deterministic state `h` is carrying none -- which is exactly the
    actionable bug this diagnostic exists to surface, and is reported rather
    than tuned away.
    """
    print("\n--- filtering probe (spec section 4, criterion 4) ---")
    print(f"latent_r2={filtering['latent_r2']:+.4f} "
          f"embedding_r2={filtering['embedding_r2']:+.4f}")
    print(f"latent_beats_embedding={filtering['latent_beats_embedding']}")
    if not filtering["latent_beats_embedding"]:
        print("WARNING: the deterministic state h adds nothing over the ENCODER "
              "embedding of the same frame -- it is carrying no history.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = get_device(prefer=args.device)
    cfg = get_config(args.arm, seed=args.seed, device=args.device)
    model = WorldModel(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("arm") != args.arm:
        raise SystemExit(
            f"checkpoint is for arm {checkpoint.get('arm')!r}, not {args.arm!r}"
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    # Split seed fixed at 0, NOT args.seed: every arm and every seed must be
    # held out on the same episodes, matching train_world_model exactly.
    train, val = episode_split(buffer.episode_paths(), val_fraction=0.2, seed=0)
    backbone = encoder_backbone(cfg.encoder)

    # The probe MUST be fit at the same context/horizon the rollout evaluates
    # at: it is applied to latents filtered from a zero state for exactly
    # `context` real frames, and fitting it at a different filtering depth is
    # a distribution mismatch worth ~25 map units of position error. Passing
    # the parsed flags rather than the defaults is what keeps the two in step
    # when a caller overrides them.
    latent_probe, embedding_probe = fit_probes(
        model, train, backbone, device,
        context=args.context, horizon=args.horizon, seed=args.seed,
    )
    result = evaluate_rollout(
        model,
        val,
        embedding_probe,
        context=args.context,
        horizon=args.horizon,
        seed=args.seed,
        device=device,
        feature_backbone=backbone,
    )

    print(f"arm={args.arm} seed={args.seed} horizon={args.horizon}")
    print(f"episodes  train={len(train)} val={len(val)}")
    # The probe is the measuring instrument. A band that looks degenerate
    # because the probe is noise is a different finding from one that is
    # degenerate because the model matches its floor, and only these numbers
    # tell the two apart.
    for name, probe in (("latent", latent_probe), ("embedding", embedding_probe)):
        r2 = probe.get("r2")
        print(
            f"probe_{name:<9} ridge={probe['ridge']:<9g} "
            f"selection_r2={'n/a' if r2 is None else f'{r2:.4f}'}"
        )
    reports = {m: metric_summary(result, m) for m in METRICS}
    print_report(result, reports)

    # Gate criterion 4. Fit on TRAIN windows and scored on VAL windows, at the
    # rollout's own context/horizon and seed -- the same reasons those are
    # forwarded to `fit_probes` apply here, and a probe fit and scored on one
    # pool would make the criterion vacuous.
    print_filtering(
        filtering_report(
            model, train, val, backbone, device,
            context=args.context, horizon=args.horizon, seed=args.seed,
        )
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({k: v.tolist() for k, v in vars(result).items()}))
        print(f"curves={args.out}")


if __name__ == "__main__":
    main()
