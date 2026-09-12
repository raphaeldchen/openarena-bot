"""Cache frozen-backbone features for every episode in a buffer.

Run once per backbone. Because one cache is ~2.93 GB and this machine has had
as little as 2.3 GB free, the workflow is one cache at a time:

    cache_features.py --backbone dinov2      # train arm 2
    cache_features.py --backbone dinov2 --clear
    cache_features.py --backbone random_vit  # train arm 3

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
    FeatureExtractor,
    cache_episode_features,
    require_free_bytes,
)
from mbfps.data.loader import feature_suffix


def cache_bytes(frames: int, backbone: str) -> int:
    """Bytes the float16 cache of `frames` frames will occupy for `backbone`.

    Reads the backbone's own `(n_patches, patch_dim)`: a backbone with rows
    narrower than the ViTs' 384 must not be sized as if it were 384 wide, or
    a disk with room for its cache refuses to start the run.

    Raises:
        KeyError: if `backbone` has no registered geometry.
    """
    n_patches, patch_dim = BACKBONE_GEOMETRY[backbone]
    return frames * n_patches * patch_dim * 2  # float16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--backbone", choices=BACKBONES, default="dinov2")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--clear", action="store_true", help="delete existing caches and exit"
    )
    args = parser.parse_args()

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
        backbone=args.backbone, device=args.device, seed=args.seed
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
