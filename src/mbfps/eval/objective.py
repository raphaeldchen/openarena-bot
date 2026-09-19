"""The training objective on held-out episodes -- milestone M3f.

`train_world_model` records the loss on every TRAINING batch and nothing on
validation, so every "minimum" M3e read off a learning curve is a training
minimum. This measures the same objective -- `WorldModel.forward`'s loss and
its five parts -- on sequences drawn from the validation episodes, so a
training minimum can be set beside a validation one. Descriptive: nothing is
decided on it (spec 2.3).
"""

import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.loader import SequenceLoader
from mbfps.models.encoders import encoder_backbone
from mbfps.utils.config import Config
from mbfps.utils.device import to_device


def val_objective(
    model, buffer: ReplayBuffer, paths, cfg: Config, device, *, batches: int = 50, seed: int = 0
) -> dict[str, float]:
    """Mean `loss` and mean of each part over `batches` draws of the training
    loader's own shape (`cfg.train.batch_size` x `cfg.train.seq_len`) from
    `paths`, with the model in eval mode, no gradient, and the categorical
    sampler pinned by `torch.manual_seed(seed)` as `evaluate_rollout` pins it.
    The loader is built once, so `seed` fixes the whole sequence of draws.
    Returns `{"loss", "embedding", "reward", "continue", "kl_dyn", "kl_rep"}`.
    The model is returned to the mode it was found in."""
    batches = int(batches)
    if batches < 1:
        raise ValueError(f"batches must be >= 1, got {batches}")
    loader = SequenceLoader(
        buffer,
        batch_size=cfg.train.batch_size,
        seq_len=cfg.train.seq_len,
        seed=seed,
        load_obs=model.input_kind == "obs",
        load_features=model.input_kind == "features",
        feature_backbone=encoder_backbone(cfg.encoder) or "dinov2",
        paths=list(paths),
    )
    was_training = model.training
    model.eval()
    torch.manual_seed(seed)
    totals: dict[str, float] = {}
    try:
        with torch.no_grad():
            for _ in range(batches):
                loss, parts = model(to_device(loader.sample(), device))
                totals["loss"] = totals.get("loss", 0.0) + float(loss)
                for name, value in parts.items():
                    totals[name] = totals.get(name, 0.0) + float(value)
    finally:
        model.train(was_training)
    return {name: value / batches for name, value in totals.items()}
