"""Reward accuracy: the study's third criterion, measured on a near-constant target.

Split out of `test_study.py` by concern. `reward_accuracy` and
`_summarise_reward` are the one part of a study job whose target is degenerate
by construction -- 19,417 of 19,424 `my_way_home` steps share the living
penalty -- so every test here is about a number that looks precise and means
nothing unless the baseline, the event count and the degeneracy flag are
reported beside it, and about the pairing, the device and the seed the score
is computed at. The shared job fixture and its pairwise-distinct parameters
are `test_study.py`'s and are imported from it.
"""

import math

import numpy as np
import pytest

from mbfps.eval import study
from mbfps.eval.study import _summarise_reward
from tests.eval.test_study import (  # noqa: F401 -- `record` is a fixture
    JOB_ARM,
    JOB_SEED,
    record,
)


# --------------------------------------------------------------------------
# reward accuracy
# --------------------------------------------------------------------------

def test_reward_accuracy_reports_its_own_degeneracy(small_buffer):
    """Spec 4.3 asks for reward accuracy, but this scenario's reward is
    near-constant: 19,417 of 19,424 real steps share one value and six carry the
    goal. A bare MSE would look precise and mean nothing, so the report must
    carry the baseline and the event count and flag the degeneracy itself."""
    import torch

    from mbfps.data.split import episode_split
    from mbfps.eval.study import reward_accuracy
    from mbfps.models.encoders import encoder_backbone
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    cfg = get_config("random_vit", device="cpu", seed=0)
    model = WorldModel(cfg)
    model.eval()
    _, val = episode_split(small_buffer.episode_paths(), val_fraction=0.2, seed=0)
    out = reward_accuracy(model, val, encoder_backbone(cfg.encoder),
                          torch.device("cpu"), limit=2)

    assert set(out) == {"mse", "baseline_mse", "r2", "n_steps",
                        "n_reward_events", "is_degenerate"}
    assert out["n_steps"] > 0
    assert out["baseline_mse"] >= 0.0
    assert isinstance(out["is_degenerate"], bool)


def test_reward_accuracy_flags_a_constant_reward_as_degenerate():
    """A constant target makes R^2 undefined; the flag is what stops a caller
    reporting a confident-looking number over it."""
    constant = _summarise_reward(np.zeros(50), np.zeros(50))
    assert constant["is_degenerate"] is True
    assert constant["n_reward_events"] == 0
    assert math.isnan(constant["r2"]), "a constant target has no R^2 to report"

    varied = _summarise_reward(np.arange(50, dtype=float), np.arange(50, dtype=float))
    assert varied["is_degenerate"] is False
    assert varied["r2"] > 0.99


def test_reward_summary_counts_the_events_and_the_baseline():
    """The baseline is the target's own variance: an MSE below it is the only
    thing that means the head learned anything."""
    true = np.zeros(1000)
    true[:2] = 10.0
    out = _summarise_reward(np.zeros(1000), true)
    assert out["n_reward_events"] == 2
    assert out["n_steps"] == 1000
    assert out["is_degenerate"] is True          # 998/1000 share the modal value
    assert out["baseline_mse"] == pytest.approx(np.var(true))
    assert out["mse"] == pytest.approx(np.mean(true ** 2))
    assert out["r2"] == pytest.approx(
        1.0 - np.sum(true ** 2) / np.sum((true - true.mean()) ** 2))


def test_reward_summary_refuses_arrays_it_cannot_compare():
    """Both messages are load-bearing. Without the guards numpy still raises --
    on an empty array it is `zero-size array to reduction operation maximum`,
    from three frames deep in a modal-share calculation -- and that is the
    message someone reads after a job has already burned its GPU hours. The
    `match` is what keeps these guards from being deletable no-ops."""
    with pytest.raises(ValueError, match="no reward steps to summarise"):
        _summarise_reward(np.zeros(0), np.zeros(0))
    with pytest.raises(ValueError, match="are not aligned"):
        _summarise_reward(np.zeros(10), np.zeros(11))


# --------------------------------------------------------------------------
# reward-accuracy semantics
# --------------------------------------------------------------------------

def test_degeneracy_is_a_property_of_the_target_not_the_prediction(record):
    """Computing the modal share over the PREDICTIONS survived: every existing
    case passes `predicted == true`, or a constant prediction against a
    degenerate target, so the two agree by accident. The dangerous direction is
    a model that collapses to a constant -- it would report is_degenerate on
    all nine cells and a reader would blame `my_way_home`'s reward rather than
    the arm's reward head, which is the confusion the flag exists to prevent."""
    on_a_constant_target = _summarise_reward(
        np.arange(50, dtype=float), np.zeros(50))
    assert on_a_constant_target["is_degenerate"] is True

    on_a_varied_target = _summarise_reward(
        np.zeros(50), np.arange(50, dtype=float))
    assert on_a_varied_target["is_degenerate"] is False

    assert isinstance(record["reward"]["is_degenerate"], bool)


def test_the_degeneracy_threshold_is_bracketed_on_both_sides():
    """0.99 was pinned only from above: the three existing cases sit at modal
    shares 1.0, 0.998 and 0.02, so ANY threshold in (0.02, 0.998] classified
    them identically and `0.99 -> 0.50` passed. Lowered, every arm is flagged
    degenerate and criterion 3 is suppressed across all nine cells."""
    assert study.DEGENERATE_REWARD_FRACTION == 0.99

    under = np.zeros(1000)
    under[:11] = np.arange(1, 12, dtype=float)          # modal share 0.989
    assert _summarise_reward(np.zeros(1000), under)["is_degenerate"] is False

    over = np.zeros(1000)
    over[:9] = np.arange(1, 10, dtype=float)            # modal share 0.991
    assert _summarise_reward(np.zeros(1000), over)["is_degenerate"] is True


def test_the_reward_rounding_resolves_a_fine_grained_target():
    """`np.round(true, 6)` feeds both the modal share and the event count, and
    `-> np.round(true, 0)` survived because both existing targets are on an
    integer scale. `my_way_home`'s living penalty is -0.0004: at 0 decimals
    every step collapses onto the modal value and the goal events vanish."""
    fine = np.arange(51, dtype=float) * 1e-4
    out = _summarise_reward(fine.copy(), fine)
    assert out["is_degenerate"] is False, (
        "a 1e-4 grid must not be rounded into a single constant value")
    assert out["n_reward_events"] == 50


def test_reward_accuracy_pairs_each_action_with_the_frame_it_led_to(
    small_buffer, monkeypatch
):
    """The third hand-rolled copy of `WorldModel.embed`'s action-time
    convention, and the only one with no test. `[:, 1:] -> [:, :-1]` survived:
    both slices have length T so the shape guard cannot see it, and a value
    pin cannot either -- measured, the two differ in the 6th significant digit
    of the MSE on this fixture and were bit-identical for M3b's pixel arm. So
    compare the embeddings actually handed to `observe` against `model.embed`,
    which is where the convention is defined."""
    import torch

    from mbfps.data.episode import load_episode
    from mbfps.eval.rollout import source_for
    from mbfps.eval.study import reward_accuracy
    from mbfps.models.encoders import encoder_backbone
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    cfg = get_config("random_vit", device="cpu", seed=0)
    model = WorldModel(cfg)
    model.eval()
    backbone = encoder_backbone(cfg.encoder)
    path = small_buffer.episode_paths()[0]

    captured: dict = {}
    real_observe = model.rssm.observe

    def spy(embeddings, actions):
        captured["embeddings"] = embeddings.detach().clone()
        captured["actions"] = actions.detach().clone()
        return real_observe(embeddings, actions)

    monkeypatch.setattr(model.rssm, "observe", spy)
    reward_accuracy(model, [path], backbone, torch.device("cpu"), limit=1)

    episode = load_episode(path)
    source = torch.as_tensor(source_for(model, path, episode, backbone))
    with torch.no_grad():
        window = {"obs": source.unsqueeze(0), "features": source.unsqueeze(0)}
        causal = model.embed(window)                       # embeddings[:, 1:]
        acausal = model.encoder(source).unsqueeze(0)[:, :-1]

    assert not torch.equal(causal, acausal), (
        "this episode's consecutive frames encode identically, so it cannot "
        "tell the causal slice from the acausal one and the assertion below "
        "would pass under either")
    assert torch.equal(captured["embeddings"], causal), (
        "reward_accuracy paired each action with the frame it was taken FROM, "
        "not the frame it led to -- an acausal posterior, and the reward "
        "accuracy of all nine cells measured against the wrong frame")
    assert captured["actions"].shape[1] == causal.shape[1]


def test_reward_accuracy_scores_the_reward_head_against_the_recorded_reward(
    small_buffer, monkeypatch
):
    """TWO Critical mutations at one call, both green across the suite.

      * `_summarise_reward(predicted, true)` -> `(true, predicted)`. Measured
        on three real `my_way_home` episodes: `is_degenerate` True -> False,
        `r2` NaN -> -47.11, `n_reward_events` 0 -> 1574; only `mse` survives,
        because it is symmetric. `is_degenerate` is the field that exists to
        stop a reader quoting a four-digit MSE over a near-constant target,
        and it INVERTS in all nine records.
        `test_degeneracy_is_a_property_of_the_target_not_the_prediction` pins
        `_summarise_reward` itself; nothing pinned the ORDER at the one call
        site that feeds it.

      * `model.heads(...)["reward"]` -> `["continue_logit"]`. The two heads
        have the same (B, T) shape, so no guard fires. Measured on real
        episodes the MSE goes 0.00258653 -> 0.00186895 -- a 28% BETTER-looking
        number for a head that never predicts reward.

    Both are killed by naming, for each argument, the tensor it must be.
    """
    import torch

    from mbfps.data.episode import load_episode
    from mbfps.eval.study import reward_accuracy
    from mbfps.models.encoders import encoder_backbone
    from mbfps.models.heads import WorldModelHeads
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    cfg = get_config(JOB_ARM, device="cpu", seed=JOB_SEED)
    model = WorldModel(cfg)
    model.eval()
    paths = small_buffer.episode_paths()[:2]

    head_outputs: list[dict] = []
    real_forward = WorldModelHeads.forward

    def forward_spy(self, latent):
        out = real_forward(self, latent)
        head_outputs.append(out)
        return out

    summarised: dict = {}
    real_summarise = study._summarise_reward

    def summarise_spy(*args, **kwargs):
        summarised["args"] = args
        return real_summarise(*args, **kwargs)

    monkeypatch.setattr(WorldModelHeads, "forward", forward_spy)
    monkeypatch.setattr(study, "_summarise_reward", summarise_spy)

    reward_accuracy(model, paths, encoder_backbone(cfg.encoder),
                    torch.device("cpu"), limit=2)

    assert len(head_outputs) == len(paths), (
        f"expected one head call per episode, saw {len(head_outputs)}")
    assert "args" in summarised, "_summarise_reward was never called"
    assert len(summarised["args"]) == 2, (
        "the summary is called positionally; this test reads the order off "
        "those two positions")
    got_predicted, got_true = summarised["args"]

    reward_head = np.concatenate(
        [out["reward"][0].float().cpu().numpy() for out in head_outputs])
    continue_head = np.concatenate(
        [out["continue_logit"][0].float().cpu().numpy() for out in head_outputs])
    recorded = np.concatenate([load_episode(p).rewards for p in paths])

    # Check-it-can-fail, three ways: the reward head and the continue head
    # really disagree, the prediction and the target really disagree, and the
    # two arguments have the SAME SHAPE, so exchanging them raises nothing.
    assert reward_head.shape == continue_head.shape
    assert not np.allclose(reward_head, continue_head), (
        "the reward head and the continue head agree on this fixture, so "
        "reading the wrong one is invisible to the assertion below")
    assert reward_head.shape == recorded.shape
    assert not np.allclose(reward_head, recorded), (
        "the prediction equals the target here, so exchanging the two "
        "arguments of _summarise_reward is a no-op and cannot be seen")

    assert np.array_equal(got_predicted, reward_head), (
        "reward accuracy did not summarise the REWARD head's output; the "
        "continue head has the same shape and yields a plausible, better "
        "looking MSE for a head that never predicts reward")
    assert not np.array_equal(got_predicted, continue_head)
    assert np.array_equal(got_true, recorded), (
        "the episode's recorded reward is not the second argument -- with the "
        "two exchanged, `is_degenerate` becomes a property of the model's "
        "predictions instead of the target and inverts in all nine records")


def test_reward_accuracys_own_inputs_are_placed_on_the_device_it_was_given(
    small_buffer, monkeypatch
):
    """The two per-episode input tensors, and the reason the study box dies.

    `torch.as_tensor(source).to(device)` -> `torch.as_tensor(source)`, and the
    same one line down for the actions, are byte-identical to pristine on CPU
    -- and every test in this suite runs `device="cpu"`, so 529/529 stayed
    green. Demonstrated on MPS, standing in for the rented CUDA box:
    `RuntimeError: Tensor for argument input is on cpu but expected on mps`.
    On the GPU box every job dies inside `reward_accuracy` AFTER paying its
    training hours -- ~1.5 h per cell -- and the ~13.5-hour unattended run
    produces nothing at all.

    The device test for `run_job` pins `.to()` and `map_location=` by OBJECT
    IDENTITY, which is what lets a device guard bite on a CUDA-less machine.
    The same trick reaches these two statements through a spy on
    `torch.Tensor.to`: the device object this test hands `reward_accuracy` is
    not equal-but-fresh, it is THE object, and nothing the mutation can
    construct satisfies `is`.
    """
    import torch

    from mbfps.eval.study import reward_accuracy
    from mbfps.models.encoders import encoder_backbone
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    device = torch.device("cpu")
    # Check-it-can-fail: an equal-but-freshly-built device -- exactly what a
    # hardcoding mutation constructs -- is NOT this object, on this machine,
    # where both of them are plain CPU.
    fresh = torch.device(device.type)
    assert fresh == device and fresh is not device, (
        "object identity can no longer tell a hardcoded device from the one "
        "the caller passed, so this test would pass on the GPU-box regression")

    cfg = get_config(JOB_ARM, device="cpu", seed=JOB_SEED)
    model = WorldModel(cfg)
    model.eval()

    placed: list = []
    real_to = torch.Tensor.to

    def to_spy(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if isinstance(target, torch.device):
            placed.append((self.dtype, target))
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to_spy)
    reward_accuracy(model, small_buffer.episode_paths()[:1],
                    encoder_backbone(cfg.encoder), device, limit=1)
    monkeypatch.undo()

    assert len(placed) == 2, (
        "reward_accuracy places exactly two tensors per episode -- the encoder "
        "input and the action sequence. Seeing "
        f"{len(placed)} means one of them was left where numpy put it, which "
        f"is a RuntimeError on any non-CPU device: {placed}")
    dtypes = [dtype for dtype, _ in placed]
    assert torch.int64 in dtypes, (
        "the action tensor never reached a device: `.unsqueeze(0).to(device)` "
        "-> `.unsqueeze(0)`")
    assert any(dtype != torch.int64 for dtype in dtypes), (
        "the encoder input never reached a device: "
        "`torch.as_tensor(source).to(device)` -> `torch.as_tensor(source)`")
    for dtype, target in placed:
        assert target is device, (
            f"the {dtype} input was moved to a device reward_accuracy built "
            "for itself rather than the one it was handed; both are CPU here, "
            f"so only object identity can see it (got {target})")


def test_reward_accuracy_is_reproducible_from_its_own_seed(small_buffer):
    """`observe` SAMPLES from the posterior, so the RNG really moves the number.

    Two mutations, both green:

      * `torch.manual_seed(seed)` -> `torch.manual_seed(0)`. All three seeds of
        an arm then draw the same posterior sample stream, criterion 3's
        seed-to-seed spread is understated, and the three seeds stop being
        independent replicates for that field -- which is the only thing the
        3-seed design buys.
      * the line deleted outright. Reward accuracy then depends on whatever
        RNG state training and the probe fit happen to leave behind, so
        re-running a cell need not reproduce its own record -- while the
        driver resumes off these records precisely because a cell is supposed
        to be reproducible.

    The first assertion kills the hardcode; the second kills the deletion, and
    is only meaningful because the first has shown the stream matters.
    """
    import torch

    from mbfps.eval.study import reward_accuracy
    from mbfps.models.encoders import encoder_backbone
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    cfg = get_config(JOB_ARM, device="cpu", seed=JOB_SEED)
    model = WorldModel(cfg)
    model.eval()
    backbone = encoder_backbone(cfg.encoder)
    paths = small_buffer.episode_paths()[:2]

    def score(seed):
        return reward_accuracy(model, paths, backbone, torch.device("cpu"),
                               limit=2, seed=seed)["mse"]

    at_one, at_two = score(1), score(2)
    assert at_one != at_two, (
        "two different seeds produce the same reward MSE, so the posterior is "
        "not being sampled here and neither assertion in this test can fail "
        "-- `torch.manual_seed(seed) -> torch.manual_seed(0)` would be a no-op")

    # Check-it-can-fail for the DELETION: the two ambient states below really
    # are different streams, so a reward_accuracy that did not reseed would
    # return two different numbers for one seed.
    torch.manual_seed(999)
    ambient_a = torch.rand(1).item()
    torch.manual_seed(12345)
    ambient_b = torch.rand(1).item()
    assert ambient_a != ambient_b, "the two ambient RNG states coincide"

    torch.manual_seed(999)
    from_ambient_a = score(1)
    torch.manual_seed(12345)
    from_ambient_b = score(1)
    assert from_ambient_a == from_ambient_b == at_one, (
        "the reward MSE moved with the AMBIENT RNG state, so this cell is not "
        "reproducible from its own seed: re-running it would not reproduce "
        "its own record, and the driver resumes off exactly that promise")


def test_the_reward_event_count_rounds_the_array_and_the_median_onto_one_grid():
    """`rounded` and the median it is compared against must share a grid.

    Two mutations, both green, both giving the same measured outcome on the
    real 59,511-step `my_way_home` reward stream: `n_reward_events` 21 ->
    59,511, i.e. EVERY STEP counted as a reward event.

      * `np.round(np.median(true), 6)` -> `np.median(true)`;
      * `np.round(true, 6)` -> `np.round(true, 12)`.

    The mechanism is the same in both: `my_way_home`'s living penalty is a
    float32 -0.0004, which widens to -0.00039999998989515007 in float64, so
    the rounded array and the unrounded (or finer-rounded) median stop being
    equal and every step differs from the modal value. The count that makes
    criterion 3 readable -- "six steps carry the goal" -- reports the exact
    opposite finding, at a magnitude nobody would flag as impossible.

    `test_the_reward_rounding_resolves_a_fine_grained_target` pins the decimals
    DOWNWARD only (6 -> 0), on a synthetic float64 grid where both roundings
    agree, so it cannot see either of these.
    """
    living_penalty = np.float32(-0.0004)
    true = np.full(1000, living_penalty, dtype=np.float32)
    true[:3] = 1.0                                  # three goal events

    # Check-it-can-fail: the float32 penalty is NOT representable on the
    # 6-decimal grid, which is the entire mechanism. On a target where the two
    # agree, both mutations are numerical no-ops.
    widened = np.asarray(true, dtype=np.float64)
    assert float(np.median(widened)) != float(np.round(np.median(widened), 6)), (
        "this target's modal value survives the float64 widening unchanged, "
        "so dropping the median's rounding is invisible here")
    assert float(np.round(widened, 12)[10]) != float(np.round(np.median(widened), 6)), (
        "a 12-decimal grid agrees with the 6-decimal median on this target, "
        "so widening the array's rounding is invisible here")

    out = _summarise_reward(np.zeros(1000, dtype=np.float64), true)
    assert out["n_reward_events"] == 3, (
        "the event count is not counting reward EVENTS: with the array and "
        "the median rounded onto different grids every step differs from the "
        "modal value and the count becomes the step count")
    assert out["n_reward_events"] != out["n_steps"]
    assert out["is_degenerate"] is True
