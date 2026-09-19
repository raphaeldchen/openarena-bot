"""`mbfps.eval.objective.val_objective`: the training objective on the
validation episodes -- the one number M3e could not read.

The model is a freshly initialised `random_vit` world model on the six-episode
fixture at `batch_size=2, seq_len=4`; nothing here trains. What is pinned is
the arithmetic (the mean of the model's own parts over seeded draws from the
paths it is given), the seeding, and that the call leaves the model as it
found it.
"""

import pytest
import torch

import mbfps.eval.objective as objective
from mbfps.data.loader import SequenceLoader
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.objective import val_objective
from mbfps.eval.study import SPLIT_SEED
from mbfps.models.encoders import encoder_backbone
from mbfps.training.world_model import WorldModel
from mbfps.utils.config import get_config
from mbfps.utils.device import to_device
from mbfps.utils.seeding import seed_everything

PARTS = ("embedding", "reward", "continue", "kl_dyn", "kl_rep")
DEVICE = torch.device("cpu")


@pytest.fixture
def fresh(small_buffer):
    cfg = get_config("random_vit", steps=1, batch_size=2, seq_len=4, device="cpu")
    seed_everything(0)
    model = WorldModel(cfg).to(DEVICE)
    _, val = episode_split(small_buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
    return model, cfg, val


def test_one_batch_is_the_models_own_loss_and_parts_on_that_batch(small_buffer, fresh):
    """`batches=1`, seed 7: the same draw the loader makes at seed 7 and the
    same categorical sample `torch.manual_seed(7)` pins, in eval mode."""
    model, cfg, val = fresh
    out = val_objective(model, small_buffer, val, cfg, DEVICE, batches=1, seed=7)

    loader = SequenceLoader(
        small_buffer, batch_size=cfg.train.batch_size, seq_len=cfg.train.seq_len, seed=7,
        load_obs=model.input_kind == "obs", load_features=model.input_kind == "features",
        feature_backbone=encoder_backbone(cfg.encoder) or "dinov2", paths=list(val),
    )
    model.eval()
    torch.manual_seed(7)
    with torch.no_grad():
        loss, parts = model(to_device(loader.sample(), DEVICE))
    assert set(out) == {"loss", *PARTS}
    assert out["loss"] == pytest.approx(float(loss))
    for name in PARTS:
        assert out[name] == pytest.approx(parts[name])


def test_the_mean_over_batches_is_the_mean_of_single_batch_calls_in_the_same_draw_order(small_buffer, fresh):
    """Three batches at seed 3 average the same three draws a three-batch call
    makes -- which a loader re-seeded per call could not reproduce, so this
    also pins that the loader is built ONCE per call."""
    model, cfg, val = fresh
    three = val_objective(model, small_buffer, val, cfg, DEVICE, batches=3, seed=3)
    loader = SequenceLoader(
        small_buffer, batch_size=cfg.train.batch_size, seq_len=cfg.train.seq_len, seed=3,
        load_obs=model.input_kind == "obs", load_features=model.input_kind == "features",
        feature_backbone=encoder_backbone(cfg.encoder) or "dinov2", paths=list(val),
    )
    model.eval()
    torch.manual_seed(3)
    losses = []
    with torch.no_grad():
        for _ in range(3):
            loss, _ = model(to_device(loader.sample(), DEVICE))
            losses.append(float(loss))
    assert three["loss"] == pytest.approx(sum(losses) / 3)


def test_the_seed_pins_the_draws_and_the_sampler(small_buffer, fresh):
    model, cfg, val = fresh
    a = val_objective(model, small_buffer, val, cfg, DEVICE, batches=2, seed=1)
    b = val_objective(model, small_buffer, val, cfg, DEVICE, batches=2, seed=1)
    c = val_objective(model, small_buffer, val, cfg, DEVICE, batches=2, seed=2)
    assert a == b
    assert a != c


def test_it_reads_only_the_paths_it_is_given(small_buffer, fresh, monkeypatch):
    model, cfg, val = fresh
    seen = {}
    real = objective.SequenceLoader

    def spy(buffer, **kw):
        seen["paths"] = kw["paths"]
        seen["batch_size"], seen["seq_len"] = kw["batch_size"], kw["seq_len"]
        return real(buffer, **kw)

    monkeypatch.setattr(objective, "SequenceLoader", spy)
    val_objective(model, small_buffer, val, cfg, DEVICE, batches=1, seed=0)
    assert seen["paths"] == list(val)
    assert (seen["batch_size"], seen["seq_len"]) == (cfg.train.batch_size, cfg.train.seq_len)


def test_it_leaves_the_model_in_the_mode_it_found_it_and_builds_no_graph(small_buffer, fresh):
    model, cfg, val = fresh
    model.train()
    val_objective(model, small_buffer, val, cfg, DEVICE, batches=1, seed=0)
    assert model.training
    model.eval()
    val_objective(model, small_buffer, val, cfg, DEVICE, batches=1, seed=0)
    assert not model.training
    assert all(p.grad is None for p in model.parameters())


def test_two_different_models_read_differently(small_buffer, fresh):
    model, cfg, val = fresh
    seed_everything(1)
    other = WorldModel(cfg).to(DEVICE)
    a = val_objective(model, small_buffer, val, cfg, DEVICE, batches=1, seed=0)
    b = val_objective(other, small_buffer, val, cfg, DEVICE, batches=1, seed=0)
    assert a["loss"] != b["loss"]


def test_zero_batches_is_refused(small_buffer, fresh):
    model, cfg, val = fresh
    with pytest.raises(ValueError, match="batches"):
        val_objective(model, small_buffer, val, cfg, DEVICE, batches=0, seed=0)
