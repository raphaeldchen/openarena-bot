"""Fit the probe on train episodes, evaluate rollouts on held-out ones."""

import argparse
import json
from pathlib import Path

import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import episode_split
from mbfps.eval.probe import filtering_gain, filtering_report, fit_probes
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

    A `False` verdict is a RESULT, not a failure of the harness: the posterior
    latent explains the privileged state no better than the raw encoding of the
    same frame does. It is reported rather than tuned away.

    THE WARNING STATES ONLY WHAT THE COMPARISON SUPPORTS. It used to say the
    deterministic state `h` "is carrying no history", and that is more than the
    criterion measures. Criterion 4 puts `h` plus a 32x32 categorical `z` -- at
    most 160 bits -- against 2048 continuous encoder floats, so it can fail on
    the bottleneck alone. Measured on the 20k checkpoint, `h` on its own scores
    +0.1508 and its RETRODICTION of earlier frames RISES with lag (+0.1508 /
    +0.1587 / +0.1878 / +0.2124 / +0.2272 at lag 0/1/5/10/20) while the current
    frame's own falls (+0.3287 -> +0.2870): `h` holds a smeared memory. What it
    cannot do is add anything over the current frame, which is what this
    criterion tests. `filtering_gain` asks the sharper question with the
    bottleneck removed.
    """
    print("\n--- filtering probe (spec section 4, criterion 4) ---")
    print(f"latent_r2={filtering['latent_r2']:+.4f} "
          f"embedding_r2={filtering['embedding_r2']:+.4f}")
    print(f"latent_beats_embedding={filtering['latent_beats_embedding']}")
    if not filtering["latent_beats_embedding"]:
        print("WARNING: the posterior latent does not improve on the raw ENCODER "
              "embedding of the same frame, so criterion 4 fails. That is what "
              "this criterion tests; it does NOT establish that h is empty -- the "
              "latent's categorical z is a far narrower channel than the 2048-d "
              "embedding, so the comparison can fail on the bottleneck alone. See "
              "the filtering_gain block below, which puts the raw embedding in "
              "both arms.")


def print_filtering_gain(gain: dict) -> None:
    """Print the bottleneck-free companion to criterion 4.

    `gain = R2([e + h] -> s) - R2([e] -> s)`: the raw encoder embedding is in
    BOTH arms, so what is left is whatever the deterministic state adds over the
    current frame. Unlike criterion 4's two R^2 -- each a max over five ridges
    taken on the rows it is scored on -- these levels are selected on a third
    split, so the number is quotable as well as comparable.

    The interval is printed because the sign alone is not the finding: a gain of
    -0.02 whose interval straddles zero says "no measurable contribution", while
    one that excludes zero says the deterministic state is actively diluting the
    frame's own encoding, and those are different results about the model.
    """
    print("\n--- filtering gain (encoder embedding in BOTH arms) ---")
    print(f"gain={gain['gain']:+.4f} "
          f"{gain['confidence']:.0%} CI [{gain['ci_low']:+.4f}, {gain['ci_high']:+.4f}]")
    print(f"joint_r2={gain['joint_r2']:+.4f} embedding_r2={gain['embedding_r2']:+.4f} "
          f"windows={gain['n_scored_windows']} ridge_selected={gain['ridge_selected']}")
    if gain["ci_high"] < 0.0:
        print("WARNING: the deterministic state h does not add to the current "
              "frame's own encoding -- the gain is negative and its interval "
              "excludes zero, so h's contribution here is not merely unmeasured.")


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

    # The fair companion, reported beside criterion 4 and never instead of it.
    # Same paths, same context/horizon/seed, so its scored windows are the very
    # rows criterion 4 was scored on and the two can be read against each other.
    print_filtering_gain(
        filtering_gain(
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
