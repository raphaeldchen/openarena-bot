import numpy as np
import pytest
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.policies import RandomPolicy
from mbfps.training.arms import DQNArm, DreamerArm
from mbfps.training.dqn import DQNConfig
from mbfps.training.online_rollout import OnlineRollout
from tests.training.fake_env import FakeEnv, fake_encode

CPU = torch.device("cpu")


def _fill(tmp_path, arm, steps=60):
    replay = ReplayBuffer(tmp_path / "replay", capacity_transitions=10**9)
    r = OnlineRollout(FakeEnv(), fake_encode, replay, run_seed=0, on_episode=arm.add_episode)
    p = RandomPolicy(4, seed=0)
    for _ in range(steps):
        r.step(p)
    r.close_episode_truncated()
    return replay


def _dreamer(tmp_path):
    replay = ReplayBuffer(tmp_path / "replay", capacity_transitions=10**9)
    return DreamerArm(seed=0, device=CPU, encode=fake_encode, replay=replay,
                      batch_size=2, seq_len=8)


def test_reward_bounds_track_observed_rewards(tmp_path):
    arm = _dreamer(tmp_path)
    assert arm.reward_bounds is None
    _fill(tmp_path, arm, steps=40)
    lo, hi = arm.reward_bounds
    assert lo <= -4.0 and hi >= -4.0


def test_dreamer_round_trains_world_model_and_keeps_isolation(tmp_path):
    arm = _dreamer(tmp_path)
    _fill(tmp_path, arm)
    before = [p.detach().clone() for p in arm.world_model.parameters()]
    out = arm.update_round()
    assert arm.isolation_ok is True
    assert any(not torch.equal(b, p) for b, p in zip(before, arm.world_model.parameters()))
    for key in ("wm_loss", "actor_loss", "critic_loss", "n_dropped", "clipped_share"):
        assert key in out


def test_dreamer_state_round_trips(tmp_path):
    a = _dreamer(tmp_path / "a")
    _fill(tmp_path / "a", a)
    a.update_round()
    b = _dreamer(tmp_path / "b")
    _fill(tmp_path / "b", b)
    b.load_state_dict(a.state_dict())
    torch.manual_seed(5)
    out_a = a.update_round()
    torch.manual_seed(5)
    out_b = b.update_round()
    assert out_a == out_b


def test_dqn_round_and_state_round_trip(tmp_path):
    cfg = DQNConfig(batch_size=8)
    a = DQNArm(seed=0, device=CPU, encode=fake_encode, cfg=cfg)
    _fill(tmp_path / "a", a)
    a.update_round()
    b = DQNArm(seed=0, device=CPU, encode=fake_encode, cfg=cfg)
    _fill(tmp_path / "b", b)
    b.load_state_dict(a.state_dict())
    assert a.update_round() == b.update_round()


def test_eval_policies_put_the_gated_mode_first(tmp_path):
    d = _dreamer(tmp_path / "d")
    q = DQNArm(seed=0, device=CPU, encode=fake_encode)
    assert list(d.eval_policies())[0] == d.gated_mode == "sample"
    assert list(q.eval_policies())[0] == q.gated_mode == "greedy"
    assert set(d.eval_policies()) == {"sample", "argmax"}
    assert set(q.eval_policies()) == {"greedy", "eps0_05"}


def test_dqn_behaviour_policy_follows_the_schedule(tmp_path):
    q = DQNArm(seed=0, device=CPU, encode=fake_encode)
    step = {"n": 0}
    p = q.behaviour_policy(lambda: step["n"])
    assert p.epsilon_fn() == 1.0
    step["n"] = 50_000
    assert p.epsilon_fn() == pytest.approx(0.05)


def test_both_arms_start_from_the_same_seeded_encoder(tmp_path):
    """Spec §5.1: same seeded initialisation of the shared feature stack."""
    d = DreamerArm(seed=3, device=CPU, encode=fake_encode,
                   replay=ReplayBuffer(tmp_path / "r3", capacity_transitions=10**9),
                   batch_size=2, seq_len=8)
    q = DQNArm(seed=3, device=CPU, encode=fake_encode)
    sd_d, sd_q = d.world_model.encoder.state_dict(), q.q.encoder.state_dict()
    assert list(sd_d) == list(sd_q)
    for key in sd_d:
        assert torch.equal(sd_d[key], sd_q[key]), key
    x = torch.randn(2, 64, 384)
    d.world_model.encoder.eval(); q.q.encoder.eval()
    with torch.no_grad():
        assert torch.equal(d.world_model.encoder(x), q.q.encoder(x))


def test_state_dict_is_a_snapshot_not_a_view(tmp_path):
    arm = _dreamer(tmp_path)
    _fill(tmp_path, arm)
    arm.update_round()
    sd = arm.state_dict()
    before = [p.clone() for p in sd["world_model"].values()]
    arm.update_round()
    assert all(torch.equal(b, p) for b, p in zip(before, sd["world_model"].values()))


@pytest.mark.parametrize("encoder, backbone", [("frozen_ssl", "dinov2"),
                                               ("pixel_ae", "pixel_ae"),
                                               ("random_vit", "random_vit")])
def test_arms_read_the_encoder_kinds_backbone(tmp_path, encoder, backbone):
    replay = ReplayBuffer(tmp_path / "replay", capacity_transitions=10**9)
    d = DreamerArm(seed=0, device=CPU, encode=fake_encode, replay=replay, batch_size=2,
                   seq_len=8, encoder=encoder, n_actions=3)
    q = DQNArm(seed=0, device=CPU, encode=fake_encode, encoder=encoder, n_actions=3)
    assert d.backbone == q.backbone == backbone
    assert d.n_actions == q.n_actions == 3
    assert d.world_model.cfg.train.n_actions == 3
    assert [m for m in d.learner.actor.modules() if isinstance(m, torch.nn.Linear)][-1].out_features == 3
    assert q.q.n_actions == 3


def test_arms_refuse_a_pixel_encoder(tmp_path):
    replay = ReplayBuffer(tmp_path / "replay", capacity_transitions=10**9)
    with pytest.raises(ValueError, match="cached features"):
        DreamerArm(seed=0, device=CPU, encode=fake_encode, replay=replay, encoder="cnn")
    with pytest.raises(ValueError, match="cached features"):
        DQNArm(seed=0, device=CPU, encode=fake_encode, encoder="cnn")
