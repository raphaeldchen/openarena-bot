import numpy as np
import pytest
import torch

from mbfps.eval.entry import (
    HORIZON, constant_baselines, continue_reading, control_broken, entry_verdict,
    real_arrays, return_reading,
)


def _episode(rewards, terminated):
    r = np.asarray(rewards, dtype=np.float64)
    t = np.asarray(terminated, dtype=bool)
    return r, t, np.zeros_like(t)


def test_real_arrays_slot_alignment():
    r, t, tr = _episode([10, 11, 12, 13, 14], [0, 0, 0, 0, 1])
    real_r, real_c, valid = real_arrays(r, t, tr, horizon=3)
    assert real_r.shape == (4, 3)
    assert real_r[0].tolist() == [11, 12, 13]
    assert real_r[2].tolist()[:2] == [13, 14]
    assert valid[2].tolist() == [True, True, False]
    assert valid[3].tolist() == [True, False, False]
    assert real_c[3, 0] == 0.0          # the terminating transition
    assert real_c[0].tolist() == [1.0, 1.0, 1.0]


def test_kill_then_terminal_on_separate_steps():
    """basic: +97 arrives with terminated=False, then -1 with terminated=True
    on the next step (19 of 109 measured kills). Each lands on its own slot."""
    r, t, tr = _episode([-4, -4, 97, -1], [0, 0, 0, 1])
    real_r, real_c, valid = real_arrays(r, t, tr, horizon=15)
    assert real_r[0, :3].tolist() == [-4, 97, -1]
    assert real_c[0, :3].tolist() == [1.0, 1.0, 0.0]
    assert valid[0].sum() == 3


def test_truncation_is_not_termination():
    r, t, _ = _episode([-4, -4, -4], [0, 0, 0])
    tr = np.array([False, False, True])
    _, real_c, _ = real_arrays(r, t, tr, horizon=3)
    assert real_c[0].tolist()[:2] == [1.0, 1.0]


def test_constant_baselines_are_means_over_valid_slots():
    eps = [_episode([1, 2, 3], [0, 0, 1])]
    c_r, c_c = constant_baselines(eps, horizon=15)
    # starts 0 and 1: returns 2+3=5 and 3 -> mean 4
    assert c_r == pytest.approx(4.0)
    # valid slots: (0,0)=1,(0,1)=0,(1,0)=0 -> mean 1/3
    assert c_c == pytest.approx(1 / 3)


def _two_episode_rows(n=40, seed=0):
    rng = np.random.default_rng(seed)
    real = rng.normal(0, 10, size=(n, HORIZON))
    valid = np.ones((n, HORIZON), dtype=bool)
    labels = np.repeat([0, 1, 2, 3], n // 4)
    return real, valid, labels


def test_return_reading_true_rewards_score_one():
    real, valid, labels = _two_episode_rows()
    out = return_reading(real, real, valid, labels, c_R=0.0, resamples=200, seed=0)
    assert out["point"] == pytest.approx(1.0)
    assert out["ci_low"] > 0


def test_return_reading_constant_scores_at_most_zero():
    real, valid, labels = _two_episode_rows()
    c_r = float(real.sum(1).mean())
    pred = np.zeros_like(real)
    pred[:, 0] = c_r
    out = return_reading(pred, real, valid, labels, c_R=c_r, resamples=200, seed=0)
    assert out["point"] == pytest.approx(0.0, abs=1e-12)
    assert not out["ci_low"] > 0


def test_continue_reading_perfect_and_constant():
    rng = np.random.default_rng(1)
    c = (rng.random((40, HORIZON)) > 0.1).astype(float)
    valid = np.ones_like(c, dtype=bool)
    labels = np.repeat([0, 1, 2, 3], 10)
    rate = float(c.mean())
    perfect = continue_reading(c, c, valid, labels, c_C=rate, resamples=200, seed=0)
    constant = continue_reading(np.full_like(c, rate), c, valid, labels, c_C=rate,
                                resamples=200, seed=0)
    assert perfect["point"] == pytest.approx(1.0)
    assert constant["point"] == pytest.approx(0.0, abs=1e-12)
    assert perfect["ci_low"] > 0 and not constant["ci_low"] > 0


def test_invalid_slots_are_ignored():
    real, valid, labels = _two_episode_rows()
    valid[:, 10:] = False
    garbage = real.copy()
    garbage[:, 10:] = 1e9
    out = return_reading(garbage, real, valid, labels, c_R=0.0, resamples=50, seed=0)
    assert out["point"] == pytest.approx(1.0)


def _iv(low):
    return {"point": low, "ci_low": low, "ci_high": low + 1}


def test_verdicts():
    good, bad = _iv(0.1), _iv(-0.1)
    assert entry_verdict({"return": good, "continue": good},
                         {"return": good, "continue": good}) == "ENTRY_PASS"
    assert entry_verdict({"return": bad, "continue": good},
                         {"return": bad, "continue": good}) == "HEAD_OR_DATA"
    assert entry_verdict({"return": bad, "continue": good},
                         {"return": good, "continue": good}) == "IMAGINATION"
    nan = {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    assert entry_verdict({"return": nan, "continue": good},
                         {"return": good, "continue": good}) == "IMAGINATION"


def test_control_broken_compares_with_greater_than():
    assert control_broken(_iv(0.01))
    assert not control_broken(_iv(0.0))
    assert not control_broken({"ci_low": float("nan")})


def test_episode_arrays_on_a_tiny_model():
    from mbfps.eval.entry import episode_arrays
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    cfg = get_config("cnn", device="cpu", n_actions=4, reward_transform="symlog")
    model = WorldModel(cfg).eval()
    t = 8
    obs = np.zeros((t + 1, 112, 112, 3), dtype=np.uint8)
    r, term, trunc = _episode([-4] * 6 + [97, -1], [0] * 7 + [1])
    actions = np.arange(t) % 4
    a = episode_arrays(model, obs, actions, r, term, trunc, torch.device("cpu"), seed=0)
    assert a.imagined_reward.shape == (t - 1, HORIZON)
    # transition 6 is the kill (+97, continues); transition 7 terminates (-1).
    # Start i reads transition i + 1 + j at slot j.
    assert a.real_reward[0, 5] == 97 and a.real_continue[0, 5] == 1.0
    assert a.real_reward[0, 6] == -1 and a.real_continue[0, 6] == 0.0
    assert a.real_continue[1, 5] == 0.0
    assert a.posterior_reward.shape == a.imagined_reward.shape
    assert a.valid.sum() == sum(min(HORIZON, t - 1 - i) for i in range(t - 1))
    b = episode_arrays(model, obs, actions, r, term, trunc, torch.device("cpu"), seed=0)
    assert np.array_equal(a.imagined_reward, b.imagined_reward)  # seeded


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")
def test_episode_arrays_run_on_mps():
    """MPS has no float64: the arrays must be moved to the CPU before the cast."""
    from mbfps.eval.entry import episode_arrays
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    device = torch.device("mps")
    cfg = get_config("cnn", device="mps", n_actions=4, reward_transform="symlog")
    model = WorldModel(cfg).to(device).eval()
    t = 8
    obs = np.zeros((t + 1, 112, 112, 3), dtype=np.uint8)
    r, term, trunc = _episode([-4] * 6 + [97, -1], [0] * 7 + [1])
    a = episode_arrays(model, obs, np.arange(t) % 4, r, term, trunc, device, seed=0)
    for name in ("imagined_reward", "imagined_continue", "posterior_reward",
                 "posterior_continue", "real_reward", "real_continue"):
        arr = getattr(a, name)
        assert isinstance(arr, np.ndarray) and arr.dtype == np.float64, name
        assert arr.shape == (t - 1, HORIZON), name
