"""Fit the probe on train episodes, evaluate rollouts on held-out ones."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import episode_split
from mbfps.eval.probe import fit_probes
from mbfps.eval.rollout import evaluate_rollout
from mbfps.models.encoders import encoder_backbone
from mbfps.training.world_model import WorldModel
from mbfps.utils.config import ARMS, get_config
from mbfps.utils.device import get_device

DEGENERATE_FRACTION = 1e-3
"""Below this share of the persistence error, a POSITIVE band is still junk.

`gap_closed` already returns NaN for a non-positive band, but a band that is
positive and merely tiny divides just fine and returns nonsense: measured on an
untrained model with a noise probe, a band ~1e-4 map units wide produced 125.2,
9.06 and -0.68 at adjacent horizon steps. Real data puts the band around 0.6%
of the error magnitude, so 0.1% is comfortably below the healthy regime while
still catching the degenerate one.
"""


def band_report(result) -> dict:
    """Everything needed to judge whether `gap_closed` means anything here.

    A bare `gap_closed` is not reportable on its own -- the ratio's denominator
    is the persistence-to-floor band, and when that band collapses the ratio
    stops carrying information without ever raising. So the width travels with
    the ratio, always.
    """
    band = result.persistence_position - result.floor_position
    ratio = result.position_gap_closed()
    relative = np.divide(
        band,
        result.persistence_position,
        out=np.full_like(band, np.nan),
        where=result.persistence_position > 0,
    )
    degenerate = (band > 0) & (relative < DEGENERATE_FRACTION)
    return {
        "band": band,
        "relative": relative,
        "ratio": ratio,
        "n_nonpositive": int((band <= 0).sum()),
        "n_degenerate": int(degenerate.sum()),
        "n_finite": int(np.isfinite(ratio).sum()),
        "n_steps": int(len(band)),
    }


def print_report(result, report: dict) -> None:
    ratio, band = report["ratio"], report["band"]
    print(
        f"position_error_final   rssm={result.rssm_position[-1]:9.2f} "
        f"persistence={result.persistence_position[-1]:9.2f} "
        f"floor={result.floor_position[-1]:9.2f}  (Doom map units)"
    )
    print(
        f"angle_error_final_deg  rssm={result.rssm_angle[-1]:6.2f} "
        f"persistence={result.persistence_angle[-1]:6.2f} "
        f"floor={result.floor_angle[-1]:6.2f}"
    )
    print(f"position_gap_closed_final={ratio[-1]:.4f}")
    print(f"angle_gap_closed_final={result.angle_gap_closed()[-1]:.4f}")

    # The WHOLE curve, not just its last point. The band is narrow and can
    # invert at some horizons and not others, so a single final value hides
    # both the shape and the NaNs.
    print(
        f"band_width  min={band.min():.3f} median={np.median(band):.3f} "
        f"max={band.max():.3f}"
    )
    print(
        f"band_as_fraction_of_persistence  min={np.nanmin(report['relative']):.5f} "
        f"median={np.nanmedian(report['relative']):.5f} "
        f"max={np.nanmax(report['relative']):.5f}"
    )
    print(
        f"steps_with_floor_above_persistence={report['n_nonpositive']}/"
        f"{report['n_steps']}"
    )
    print(
        f"steps_with_degenerate_positive_band={report['n_degenerate']}/"
        f"{report['n_steps']}  (band < {DEGENERATE_FRACTION:g} x persistence)"
    )
    print(
        f"gap_closed  finite={report['n_finite']}/{report['n_steps']} "
        f"mean={np.nanmean(ratio):+.4f} min={np.nanmin(ratio):+.4f} "
        f"max={np.nanmax(ratio):+.4f}"
    )

    if not np.isfinite(ratio).any():
        print(
            "WARNING: gap_closed is NaN at every horizon step -- the band is "
            "non-positive throughout, so this metric says nothing here. Report "
            "raw errors instead (spec 9, open question 1)."
        )
    elif report["n_degenerate"]:
        print(
            f"WARNING: {report['n_degenerate']} horizon steps have a positive but "
            "numerically degenerate band. gap_closed divides by it without "
            "complaint and the resulting ratio is not trustworthy at those steps."
        )


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

    latent_probe, embedding_probe = fit_probes(
        model, train, backbone, device, seed=args.seed
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
    print_report(result, band_report(result))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({k: v.tolist() for k, v in vars(result).items()}))
        print(f"curves={args.out}")


if __name__ == "__main__":
    main()
