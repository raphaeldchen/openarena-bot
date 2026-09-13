"""Cache frozen-backbone features for every episode in a buffer.

Run once per backbone. Because a ViT cache is ~2.93 GB and this machine has
had as little as 2.3 GB free, the workflow is one cache at a time:

    cache_features.py --backbone dinov2      # train arm 2
    cache_features.py --backbone dinov2 --clear
    cache_features.py --backbone random_vit  # train arm 3

The `pixel_ae` cache is the exception. Its backbone is the M2 pixel
autoencoder's encoder, read from `--checkpoint` (default
`PIXEL_AE_CHECKPOINT`), and its rows are (64, 32) rather than (64, 384), so the
whole cache is ~0.24 GB and coexists with either ViT cache:

    cache_features.py --backbone pixel_ae --checkpoint runs/m2_fixed/autoencoder_cnn.pt

Features must be generated once on one device: CPU and MPS outputs differ in
~3% of float16 elements, so mixing them would silently break the input
equality the study depends on.
"""

import argparse
import time
from pathlib import Path

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.features import (
    BACKBONE_GEOMETRY,
    BACKBONES,
    PIXEL_AE_CHECKPOINT,
    FeatureExtractor,
    cache_episode_features,
    require_free_bytes,
)
from mbfps.data.loader import feature_suffix


def cache_bytes(frames: int, backbone: str) -> int:
    """Bytes a float16 cache of `frames` rows of `backbone`'s geometry occupies.

    Read from the registry, not from a module constant: the estimate used to
    be `frames * 64 * 384 * 2` for every backbone, which for `pixel_ae` would
    demand 2.93 GB of free disk to write 0.24 GB. On the 2.3-GB-free machine
    this script was written for, that refusal would block the one cache that
    fits.

    Raises:
        KeyError: if `backbone` has no registered geometry.
    """
    n_patches, patch_dim = BACKBONE_GEOMETRY[backbone]
    return frames * n_patches * patch_dim * 2  # float16


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--backbone", choices=BACKBONES, default="dinov2")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    # Forwarded to FeatureExtractor for EVERY backbone; build_backbone is the
    # one place that knows pixel_ae is the backbone that reads it. A
    # conditional here would be a second copy of that knowledge.
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PIXEL_AE_CHECKPOINT,
        help="M2 autoencoder checkpoint whose encoder is the pixel_ae backbone "
        "(ignored by the ViT backbones)",
    )
    parser.add_argument(
        "--clear", action="store_true", help="delete existing caches and exit"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)

    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    paths = buffer.episode_paths()
    if not paths:
        raise SystemExit(f"no episodes found in {args.data}")

    if args.clear:
        removed = 0
        for path in paths:
            feature_path = path.with_suffix(feature_suffix(args.backbone))
            if feature_path.is_file():
                feature_path.unlink()
                removed += 1
        print(f"removed={removed} {args.backbone} feature files from {args.data}")
        return

    frames = sum(ep.length + 1 for ep in buffer.load_all())
    needed = cache_bytes(frames, args.backbone)
    print(f"episodes={len(paths)} frames={frames} needs={needed / 1e9:.2f} GB")
    require_free_bytes(args.data, needed)

    extractor = FeatureExtractor(
        backbone=args.backbone,
        device=args.device,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )
    start = time.perf_counter()
    for i, path in enumerate(paths, start=1):
        cache_episode_features(path, extractor)
        if i % 20 == 0 or i == len(paths):
            print(f"[{i}/{len(paths)}] elapsed={time.perf_counter() - start:.0f}s")

    elapsed = time.perf_counter() - start
    print(f"backbone={args.backbone} device={args.device} seed={args.seed}")
    print(f"features_cached={len(paths)} elapsed_s={elapsed:.0f}")
    print(f"frames_per_second={frames / elapsed:.0f}")


if __name__ == "__main__":
    main()
