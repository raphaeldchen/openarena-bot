import inspect
from pathlib import Path

import numpy as np
import pytest

from mbfps.eval.rollout import gap_closed


def test_gap_closed_is_one_when_the_model_reaches_the_floor():
    np.testing.assert_allclose(gap_closed(np.array([10.0]), np.array([2.0]), np.array([2.0])), [1.0])


def test_gap_closed_is_zero_when_the_model_only_matches_persistence():
    np.testing.assert_allclose(gap_closed(np.array([10.0]), np.array([10.0]), np.array([2.0])), [0.0])


def test_gap_closed_is_negative_when_worse_than_persistence():
    """Reported, never clipped: worse-than-persistence is a real finding and
    hiding it behind a floor of zero would make a broken model look adequate."""
    assert gap_closed(np.array([10.0]), np.array([12.0]), np.array([2.0]))[0] < 0


def test_gap_closed_is_nan_when_the_band_collapses():
    """If persistence and the floor coincide there is no headroom to close, and
    the ratio is 0/0. NaN is correct; a silent 0 or 1 would be a lie."""
    result = gap_closed(np.array([5.0]), np.array([5.0]), np.array([5.0]))
    assert np.isnan(result[0])


def test_gap_closed_is_nan_on_a_zero_band_even_when_the_model_differs():
    """A zero-width band with a non-zero numerator divides by zero and yields
    +-inf, not NaN, so the collapse test above passes for the wrong reason:
    there the numerator is also zero and 0/0 happens to be NaN already.

    Found by mutation testing -- weakening the guard to `band < 0` survived
    every other test in this file. An infinite gap_closed would propagate
    through the cross-arm mean and destroy the M3 headline number."""
    result = gap_closed(np.array([5.0]), np.array([3.0]), np.array([5.0]))
    assert np.isnan(result[0]), "a zero band must be NaN, never +-inf"


def test_gap_closed_is_nan_when_the_floor_exceeds_persistence():
    """A negative band inverts the ratio's sign, so a model WORSE than
    persistence would score positive -- and "gap_closed > 0" is the M3 gate.
    Measured on real data the floor does exceed persistence, so this is the
    common case, not a corner."""
    result = gap_closed(np.array([10.0]), np.array([12.0]), np.array([14.0]))
    assert np.isnan(result[0]), "a negative band must not produce a signed score"


def test_gap_closed_is_elementwise_over_the_horizon():
    persistence = np.array([10.0, 20.0])
    model = np.array([6.0, 20.0])
    floor = np.array([2.0, 0.0])
    np.testing.assert_allclose(gap_closed(persistence, model, floor), [0.5, 0.0])


# ---------------------------------------------------------------------------
# An ORACLE rig, used to pin the truth alignment.
#
# The whole point of `evaluate_rollout` is the number it reports, and an
# off-by-one in the ground-truth slice shifts every one of those numbers
# without changing a single shape. Nothing derived from a random untrained
# model can catch that, because every curve is noise. So: a synthetic episode
# whose privileged position is an exact linear function of the frame index,
# plus a model that is EXACTLY right -- its encoder tags each frame with its
# own index, and its dynamics advance that tag by exactly one per action.
#
# A perfect predictor must score zero. Any other alignment gives it a constant,
# irreducible error of |offset| steps of true displacement.
# ---------------------------------------------------------------------------

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import mbfps.eval.rollout as rollout_module  # noqa: E402
from mbfps.data.episode import Episode, save_episode  # noqa: E402
from mbfps.envs.protocol import OBS_SHAPE  # noqa: E402
from mbfps.eval.probe import fit_probe, probe_targets  # noqa: E402
from mbfps.eval.rollout import RolloutResult, evaluate_rollout, source_for  # noqa: E402

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")
DX, DY = 10.0, -3.0
"""Per-frame displacement of the synthetic agent, in Doom map units."""
STEP = float(np.hypot(DX, DY))
"""Euclidean length of one frame of true displacement: sqrt(10^2 + 3^2)."""
T_SYNTHETIC = 20
CONTEXT, HORIZON = 3, 5


def synthetic_episode(length: int = T_SYNTHETIC) -> Episode:
    """`pos_x = 10*t`, `pos_y = -3*t`, and `obs[t]` carries `t` verbatim."""
    obs = np.zeros((length + 1, *OBS_SHAPE), dtype=np.uint8)
    obs[:, 0, 0, 0] = np.arange(length + 1, dtype=np.uint8)
    t = np.arange(length + 1, dtype=np.float32)
    privileged = np.zeros((length + 1, len(KEYS)), dtype=np.float32)
    privileged[:, 0] = 100.0  # health, constant
    privileged[:, 1] = DX * t
    privileged[:, 2] = DY * t
    privileged[:, 4] = 0.0  # angle, constant
    return Episode(
        obs=obs,
        actions=np.arange(length, dtype=np.int32) % 6,
        rewards=np.zeros(length, dtype=np.float32),
        terminated=np.zeros(length, dtype=bool),
        truncated=np.zeros(length, dtype=bool),
        privileged=privileged,
        privileged_keys=KEYS,
        policy_name="synthetic",
        seed=0,
        scenario="alignment",
    )


class _TagEncoder(nn.Module):
    """`emb[n] = [index of the frame obs[n]]` -- an exact, invertible tag."""

    def forward(self, obs):
        return obs[:, 0, 0, 0].to(torch.float32).unsqueeze(-1)


class _OracleRSSM(nn.Module):
    """A perfect filter and a perfect dynamics model over the frame tag."""

    def observe(self, embeddings, actions, state=None):
        tag = embeddings[..., :1]  # the posterior sees the frame, so it knows
        return {"h": tag, "z": tag, "latent": torch.cat([tag, tag], dim=-1)}

    def imagine(self, actions, state):
        h0 = state[0]  # (B, 1): tag of the last observed frame
        steps = torch.arange(1, actions.shape[1] + 1, dtype=h0.dtype).view(1, -1, 1)
        tag = h0.unsqueeze(1) + steps  # every action advances the frame by one
        return {"h": tag, "z": tag, "latent": torch.cat([tag, tag], dim=-1)}


class _OracleHeads(nn.Module):
    def forward(self, latent):
        return {"embedding": latent[..., :1]}


class OracleModel(nn.Module):
    """Duck-typed like `WorldModel`, but exactly right about the world."""

    input_kind = "obs"

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _TagEncoder()
        self.rssm = _OracleRSSM()
        self.heads = _OracleHeads()
        self.dummy = nn.Parameter(torch.zeros(1))


class _DriftingRSSM(_OracleRSSM):
    """An exact filter with BROKEN dynamics: two frames of travel per action."""

    def imagine(self, actions, state):
        h0 = state[0]
        steps = torch.arange(1, actions.shape[1] + 1, dtype=h0.dtype).view(1, -1, 1)
        tag = h0.unsqueeze(1) + 2 * steps
        return {"h": tag, "z": tag, "latent": torch.cat([tag, tag], dim=-1)}


class DriftingModel(OracleModel):
    def __init__(self) -> None:
        super().__init__()
        self.rssm = _DriftingRSSM()


class _StatefulRSSM(nn.Module):
    """Unlike `_OracleRSSM` (which ignores `state` entirely), this filter's
    `observe` output genuinely depends on the incoming `state`: it folds
    `state[0]` into every output step, the way a real posterior carries its
    belief forward instead of starting cold. That is what makes dropping
    `state=state` from the floor's `observe` call an OBSERVABLE difference
    here, unlike against `_OracleRSSM` or against an untrained real model
    (where the effect measures ~1e-4, indistinguishable from noise)."""

    def observe(self, embeddings, actions, state=None):
        base = embeddings[..., :1]
        carry = torch.zeros_like(base[:, :1, :]) if state is None else state[0].unsqueeze(1)
        h = base + carry
        return {"h": h, "z": h, "latent": torch.cat([h, h], dim=-1)}

    def imagine(self, actions, state):
        h0 = state[0]
        steps = torch.arange(1, actions.shape[1] + 1, dtype=h0.dtype).view(1, -1, 1)
        tag = h0.unsqueeze(1) + steps
        return {"h": tag, "z": tag, "latent": torch.cat([tag, tag], dim=-1)}


class StatefulModel(OracleModel):
    def __init__(self) -> None:
        super().__init__()
        self.rssm = _StatefulRSSM()


def oracle_probe(episode: Episode) -> dict:
    """The exact linear map from a frame tag to that frame's privileged state."""
    tags = np.arange(episode.privileged.shape[0], dtype=np.float64)[:, None]
    return fit_probe(tags, probe_targets(episode.privileged, KEYS), ridge=1e-8)


@pytest.fixture
def oracle(tmp_path):
    """`(model, [path], probe)` for an exactly-predictable synthetic episode."""
    episode = synthetic_episode()
    path = tmp_path / f"ep_000000_len{T_SYNTHETIC:05d}.npz"
    save_episode(episode, path)
    return OracleModel(), [path], oracle_probe(episode), episode


def run_oracle(oracle, **kwargs) -> RolloutResult:
    model, paths, probe, _ = oracle
    options = {
        "context": CONTEXT,
        "horizon": HORIZON,
        "device": torch.device("cpu"),
        "feature_backbone": None,
    }
    options.update(kwargs)
    return evaluate_rollout(model, paths, probe, **options)


def test_a_perfect_predictor_scores_zero_only_on_the_frame_the_action_produced(oracle):
    """THE alignment test. `imagined[k]` is driven by `actions[context + k]`,
    and an action produces the NEXT frame, so the truth for step k is
    `privileged[start + context + 1 + k]` -- not `privileged[start + context + k]`.

    Measured on this rig, the wrong slice hands a model that is exactly right a
    flat error of one step of true displacement (10.4403 map units) at every
    horizon step, and a slice off by d costs exactly |d| steps. Nothing about
    the shapes changes, so only a value assertion can see it.
    """
    result = run_oracle(oracle)
    np.testing.assert_allclose(result.rssm_position, np.zeros(HORIZON), atol=1e-6)
    np.testing.assert_allclose(result.rssm_angle, np.zeros(HORIZON), atol=1e-6)


def test_the_encoder_floor_is_exact_when_it_sees_the_real_future_frame(oracle):
    """The floor's posterior is conditioned on the REAL frame at every step, so
    with an exact encoder and an exact head it has nothing left to get wrong.
    A non-zero floor here means the truth it is scored against is misaligned --
    it caught the same off-by-one as the test above, independently of the
    dynamics path."""
    result = run_oracle(oracle)
    np.testing.assert_allclose(result.floor_position, np.zeros(HORIZON), atol=1e-6)


def test_the_floor_is_conditioned_on_the_real_frames_not_on_the_imagination(tmp_path):
    """The floor is the lower bracket: what the encoder still permits once the
    dynamics are removed from the question. If it were run on the IMAGINED
    latent it would just be the model again, the band would collapse to zero
    width by construction, and `gap_closed` would be NaN everywhere.

    Checked with an exact encoder but DELIBERATELY BROKEN dynamics -- two frames
    of travel per action. The floor still sees every real frame, so it must stay
    exactly zero while the model drifts by one extra step per horizon step."""
    episode = synthetic_episode()
    path = tmp_path / f"ep_000000_len{T_SYNTHETIC:05d}.npz"
    save_episode(episode, path)
    result = evaluate_rollout(
        DriftingModel(), [path], oracle_probe(episode),
        context=CONTEXT, horizon=HORIZON, device=torch.device("cpu"),
    )
    np.testing.assert_allclose(result.floor_position, np.zeros(HORIZON), atol=1e-6)
    np.testing.assert_allclose(
        result.rssm_position, STEP * np.arange(1, HORIZON + 1), rtol=1e-6
    )


def test_the_floor_resumes_the_context_state_instead_of_starting_cold(tmp_path):
    """The floor must be the SAME filter that produced the context's state,
    continuing from where it left off -- not a fresh posterior started from a
    zero belief. Dropping `state=state` from
    `model.rssm.observe(embeddings[:, context:], actions[:, context:], state=state)`
    would do exactly that, and it is invisible against `_OracleRSSM` (which
    ignores `state`) or against an untrained real model (measured delta
    ~1.3e-4). `_StatefulRSSM` makes `observe` genuinely depend on `state`.

    With a single window starting at 0, the context's own last posterior
    state works out to exactly CONTEXT (map-index tag units), since the
    context call itself always starts cold (`state=None`) and its last
    output is simply the tag of the last context frame. A floor that resumes
    that state is offset from the truth by CONTEXT map-index units at every
    horizon step -- a constant position error of CONTEXT * STEP. A floor
    started cold (state dropped) would carry no such offset and score exactly
    zero, indistinguishable from
    `test_the_encoder_floor_is_exact_when_it_sees_the_real_future_frame`."""
    length = CONTEXT + HORIZON + 1  # exactly one window, starting at 0
    episode = synthetic_episode(length=length)
    path = tmp_path / f"ep_000000_len{length:05d}.npz"
    save_episode(episode, path)
    result = evaluate_rollout(
        StatefulModel(), [path], oracle_probe(episode),
        context=CONTEXT, horizon=HORIZON, device=torch.device("cpu"),
    )
    expected = CONTEXT * STEP
    np.testing.assert_allclose(
        result.floor_position, np.full(HORIZON, expected), rtol=1e-4, atol=1e-6
    )


def test_persistence_error_is_exactly_the_true_displacement(oracle):
    """The module docstring's claim, checked: persistence holds the last context
    state, so its error IS how far the agent really travelled. One step of
    displacement at horizon step 1, k+1 steps at horizon step k+1.

    A zero at the first horizon step -- which the off-by-one slice produces --
    would mean "the agent never moved" scored as perfect one step into the
    future, which is impossible unless the truth is the held state itself."""
    result = run_oracle(oracle)
    expected = STEP * np.arange(1, HORIZON + 1)
    np.testing.assert_allclose(result.persistence_position, expected, rtol=1e-6)
    assert result.persistence_position[0] > 0.0


def test_truth_is_sliced_one_frame_after_the_last_context_frame(oracle, monkeypatch):
    """The index arithmetic itself, pinned directly rather than inferred from an
    error curve."""
    _, _, _, episode = oracle
    seen = []
    real = rollout_module.probe_targets
    monkeypatch.setattr(
        rollout_module,
        "probe_targets",
        lambda privileged, keys: seen.append(privileged) or real(privileged, keys),
    )
    run_oracle(oracle)
    need = CONTEXT + HORIZON
    # First window starts at 0; the loop then strides by `need`.
    np.testing.assert_array_equal(
        seen[0], episode.privileged[CONTEXT + 1 : need + 1]
    )
    np.testing.assert_array_equal(
        seen[1], episode.privileged[need + CONTEXT + 1 : 2 * need + 1]
    )


def test_every_window_contributes_and_windows_do_not_overlap(oracle, monkeypatch):
    """Stride is the full window, not 1: overlapping windows would reuse frames
    and make the averaged curve look tighter than the data supports."""
    seen = []
    real = rollout_module.probe_targets
    monkeypatch.setattr(
        rollout_module,
        "probe_targets",
        lambda privileged, keys: seen.append(privileged) or real(privileged, keys),
    )
    run_oracle(oracle)
    # length 20, need 8 -> range(0, 12, 8) = [0, 8]
    assert len(seen) == 2


def test_the_final_window_is_included_when_length_is_an_exact_multiple_of_need(
    tmp_path, monkeypatch
):
    """`start = episode.length - need` is a LEGAL window -- it reads
    `privileged[...start+need]` and `obs[...start+need]`, both in range for a
    T+1-row array -- but `range(0, episode.length - need, need)` excludes it
    whenever `episode.length % need == 0`. Checked with a length that is
    exactly `2 * need`, where the broken bound gives only the start=0 window
    and the correct bound (`- need + 1`) also gives start=need."""
    need = CONTEXT + HORIZON
    length = 2 * need
    episode = synthetic_episode(length=length)
    path = tmp_path / f"ep_000000_len{length:05d}.npz"
    save_episode(episode, path)
    seen = []
    real = rollout_module.probe_targets
    monkeypatch.setattr(
        rollout_module,
        "probe_targets",
        lambda privileged, keys: seen.append(privileged) or real(privileged, keys),
    )
    evaluate_rollout(
        OracleModel(), [path], oracle_probe(episode),
        context=CONTEXT, horizon=HORIZON, device=torch.device("cpu"),
    )
    assert len(seen) == 2, "the window starting at episode.length - need was dropped"


def test_all_three_references_are_probed_with_the_very_same_probe(oracle, monkeypatch):
    """The band must measure information content, not probe fit. Fitting one
    probe on real embeddings and another on predicted ones was measured
    destroying the signal (band below 2 SE at 18 of 45 horizon steps against 2
    of 45 once the pipeline is shared)."""
    probes = []
    real = rollout_module.apply_probe
    monkeypatch.setattr(
        rollout_module,
        "apply_probe",
        lambda probe, latents: probes.append(id(probe)) or real(probe, latents),
    )
    run_oracle(oracle)
    assert len(probes) == 6  # three references x two windows
    assert len(set(probes)) == 1


def test_persistence_holds_one_embedding_for_the_whole_horizon(oracle, monkeypatch):
    """"Persistence" means the last context prediction, unchanged. If it varied
    over the horizon it would be some other baseline entirely."""
    calls = []
    real = rollout_module.apply_probe
    monkeypatch.setattr(
        rollout_module,
        "apply_probe",
        lambda probe, latents: calls.append(np.asarray(latents)) or real(probe, latents),
    )
    run_oracle(oracle)
    persistence_input = calls[2]  # model, floor, persistence -- in that order
    assert persistence_input.shape[0] == HORIZON
    assert np.unique(persistence_input, axis=0).shape[0] == 1


def test_imagine_is_driven_by_the_post_context_actions_alone(oracle, monkeypatch):
    """The imagined arm must never touch a future embedding, and must consume
    exactly the horizon's actions."""
    model, _, _, episode = oracle
    seen = []
    real = model.rssm.imagine
    monkeypatch.setattr(
        model.rssm,
        "imagine",
        lambda actions, state: seen.append(actions.clone()) or real(actions, state),
    )
    run_oracle(oracle)
    assert seen[0].shape == (1, HORIZON)
    np.testing.assert_array_equal(
        seen[0][0].numpy(), episode.actions[CONTEXT : CONTEXT + HORIZON]
    )


def test_the_reported_curve_is_the_mean_over_every_window(tmp_path):
    """The mean, not the first window and not the median. Four single-window
    episodes -- three the probe predicts exactly, one whose pos_x is displaced
    by a constant 100 map units -- give per-window errors of 0, 0, 0, 100. The
    mean of those is 25; the median is 0.

    Deliberately NOT 0, 0, 100, 100 (mean == median == 50 there): that fixture
    let `np.median` masquerade as `np.mean` and survive mutation testing.
    Each episode is sized to contribute exactly one window, so the four
    values above are exactly the four numbers averaged, not four windows
    diluted by more of the same."""
    length = CONTEXT + HORIZON + 1  # exactly one window per episode
    exact = synthetic_episode(length=length)
    displaced = synthetic_episode(length=length)
    displaced.privileged = displaced.privileged.copy()
    displaced.privileged[:, 1] += 100.0
    episodes = [exact, exact, exact, displaced]
    paths = []
    for index, episode in enumerate(episodes):
        path = tmp_path / f"ep_{index:06d}_len{length:05d}.npz"
        save_episode(episode, path)
        paths.append(path)
    result = evaluate_rollout(
        OracleModel(), paths, oracle_probe(exact),
        context=CONTEXT, horizon=HORIZON, device=torch.device("cpu"),
    )
    np.testing.assert_allclose(result.rssm_position, np.full(HORIZON, 25.0), atol=1e-5)
    np.testing.assert_allclose(result.floor_position, np.full(HORIZON, 25.0), atol=1e-5)


def test_all_three_references_are_emitted_by_the_embedding_head(tmp_path, monkeypatch):
    """The shared-pipeline constraint, on a REAL model where the head's output
    and the raw encoder embedding actually differ.

    Every reference -- imagined, floor and persistence -- must be an embedding
    the model itself PREDICTED, so the one probe sees one distribution. Probing
    the raw encoder embedding for the floor (or the raw last context embedding
    for persistence) is the distribution mismatch that was measured pushing the
    band below 2 SE at 18 of 45 horizon steps, against 2 of 45 once shared."""
    episode = synthetic_episode()
    path = tmp_path / f"ep_000000_len{T_SYNTHETIC:05d}.npz"
    save_episode(episode, path)
    model, probe = real_model_and_probe()

    head_rows: set[bytes] = set()
    real_heads = model.heads.forward

    def spy_heads(latent):
        out = real_heads(latent)
        for row in out["embedding"].reshape(-1, out["embedding"].shape[-1]):
            head_rows.add(row.cpu().numpy().tobytes())
        return out

    probed: list[np.ndarray] = []
    real_apply = rollout_module.apply_probe
    monkeypatch.setattr(model.heads, "forward", spy_heads)
    monkeypatch.setattr(
        rollout_module,
        "apply_probe",
        lambda p, latents: probed.append(np.asarray(latents)) or real_apply(p, latents),
    )
    evaluate_rollout(
        model, [path], probe, context=CONTEXT, horizon=HORIZON,
        device=torch.device("cpu"),
    )
    assert len(probed) == 6  # three references x two windows
    for reference in probed:
        for row in reference:
            assert row.astype(np.float32).tobytes() in head_rows


def test_result_arrays_all_span_the_horizon(oracle):
    result = run_oracle(oracle)
    for name in (
        "horizon", "rssm_position", "persistence_position", "floor_position",
        "rssm_angle", "persistence_angle", "floor_angle",
    ):
        value = getattr(result, name)
        assert isinstance(value, np.ndarray), name
        assert value.shape == (HORIZON,), name


def test_horizon_axis_is_one_based(oracle):
    """Step 1 is one action into the future, not zero. A zero-based axis would
    mislabel every point on the published error-versus-horizon curve."""
    result = run_oracle(oracle)
    np.testing.assert_array_equal(result.horizon, np.arange(1, HORIZON + 1))


def test_evaluate_rollout_defaults_are_the_spec_values():
    """context=5, horizon=45, seed=0 are the numbers M3 is reported at. A
    silently changed default would move the headline number."""
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(evaluate_rollout).parameters.items()
    }
    assert defaults["context"] == 5
    assert defaults["horizon"] == 45
    assert defaults["seed"] == 0
    assert defaults["device"] is None
    assert defaults["feature_backbone"] is None


def test_position_gap_closed_reads_persistence_model_and_floor_in_that_order():
    """Swapping any two of the three arguments still typechecks and still
    returns a plausible-looking number."""
    result = RolloutResult(
        horizon=np.array([1]),
        rssm_position=np.array([6.0]),
        persistence_position=np.array([10.0]),
        floor_position=np.array([2.0]),
        rssm_angle=np.array([20.0]),
        persistence_angle=np.array([50.0]),
        floor_angle=np.array([10.0]),
    )
    # Deliberately different ratios for position and angle: with both at 0.5,
    # `angle_gap_closed` reading the POSITION curves survived mutation testing.
    np.testing.assert_allclose(result.position_gap_closed(), [0.5])
    np.testing.assert_allclose(result.angle_gap_closed(), [0.75])


def test_episodes_shorter_than_the_window_are_skipped_not_fatal(tmp_path):
    short = synthetic_episode(length=4)
    save_episode(short, tmp_path / "ep_000000_len00004.npz")
    long = synthetic_episode()
    save_episode(long, tmp_path / f"ep_000001_len{T_SYNTHETIC:05d}.npz")
    result = evaluate_rollout(
        OracleModel(),
        [tmp_path / "ep_000000_len00004.npz", tmp_path / f"ep_000001_len{T_SYNTHETIC:05d}.npz"],
        oracle_probe(long),
        context=CONTEXT, horizon=HORIZON, device=torch.device("cpu"),
    )
    np.testing.assert_allclose(result.rssm_position, np.zeros(HORIZON), atol=1e-6)


def test_evaluate_rollout_raises_when_no_window_is_long_enough(tmp_path):
    """Silently returning an empty or NaN curve would be reported as a result."""
    short = synthetic_episode(length=4)
    path = tmp_path / "ep_000000_len00004.npz"
    save_episode(short, path)
    with pytest.raises(ValueError, match="no validation window reached 9 frames"):
        evaluate_rollout(
            OracleModel(), [path], oracle_probe(short),
            context=CONTEXT, horizon=HORIZON, device=torch.device("cpu"),
        )


def test_the_model_is_switched_to_eval_mode(oracle):
    model, _, _, _ = oracle
    model.train(True)
    run_oracle(oracle)
    assert model.training is False


def test_no_gradient_is_taken_anywhere_in_the_rollout(oracle, monkeypatch):
    """`privileged_state` is evaluation-only and must never reach a training
    tensor. The decorator is the guard; this checks it is actually in force."""
    model, _, _, _ = oracle
    grad_states = []
    real = model.encoder.forward
    monkeypatch.setattr(
        model.encoder,
        "forward",
        lambda obs: grad_states.append(torch.is_grad_enabled()) or real(obs),
    )
    result = run_oracle(oracle)
    assert grad_states and not any(grad_states)
    assert model.dummy.grad is None
    assert isinstance(result.rssm_position, np.ndarray)


def test_source_for_returns_pixels_for_the_pixel_arm(tmp_path):
    episode = synthetic_episode()
    model = OracleModel()
    assert source_for(model, tmp_path / "unused.npz", episode, None) is episode.obs


def test_source_for_reads_the_arms_own_namespaced_feature_cache(tmp_path):
    """Arms 2 and 3 both cache "features"; a shared filename would silently feed
    the control arm the treatment arm's inputs."""
    episode = synthetic_episode()
    model = OracleModel()
    model.input_kind = "features"
    path = tmp_path / "ep_000000_len00020.npz"
    expected = np.arange(6, dtype=np.float32).reshape(3, 2)
    np.save(path.with_suffix(".features_random_vit.npy"), expected)
    np.testing.assert_array_equal(
        source_for(model, path, episode, "random_vit"), expected
    )


# ---------------------------------------------------------------------------
# The real RSSM: sampling behaviour and the end-to-end band.
# ---------------------------------------------------------------------------


def real_model_and_probe(arm: str = "cnn"):
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    cfg = get_config(arm, device="cpu", seed=0)
    model = WorldModel(cfg)
    embeds = np.random.default_rng(0).normal(size=(64, cfg.encoder.embed_dim))
    targets = np.random.default_rng(1).normal(size=(64, 4))
    return model, fit_probe(embeds, targets)


def test_rollouts_are_reproducible_for_a_seed_and_differ_between_seeds(tmp_path):
    """Sampling stays STOCHASTIC. Reproducibility comes from the `seed`
    argument, not from taking the categorical mode -- measured, the mode
    collapses the imagined trajectory to 3 distinct latents out of 45 and
    roughly quadruples position error, and it degrades WITH training, so no
    untrained smoke test would reveal it.

    Two claims, and both matter: same seed twice must match exactly (or the
    reported number is not reproducible), and two seeds must differ (or the
    sampling has been replaced by a deterministic mode).

    The second claim is checked for BIT inequality, not for a tolerance. An
    argmax path makes the two runs byte-identical, which is exactly what
    `array_equal` catches; on an untrained model the two sampled trajectories
    land close together in probe space, so a tolerance-based check would pass
    for the wrong reason."""
    episode = synthetic_episode()
    path = tmp_path / f"ep_000000_len{T_SYNTHETIC:05d}.npz"
    save_episode(episode, path)
    model, probe = real_model_and_probe()

    def run(seed):
        return evaluate_rollout(
            model, [path], probe, context=CONTEXT, horizon=HORIZON,
            seed=seed, device=torch.device("cpu"),
        ).rssm_position

    np.testing.assert_array_equal(run(0), run(0))
    assert not np.array_equal(run(0), run(1))


@pytest.mark.slow
def test_rollout_produces_the_full_band_on_real_data():
    """The gate artifact: three curves over the horizon, no NaNs, floor below
    persistence. An untrained model need not beat persistence -- this checks the
    harness produces a readable band, not that the model is good."""
    from mbfps.data.buffer import ReplayBuffer
    from mbfps.data.split import episode_split
    from mbfps.models.encoders import encoder_backbone
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    cfg = get_config("random_vit", device="cpu", seed=0)
    model = WorldModel(cfg)
    buffer = ReplayBuffer(Path("data/my_way_home"), capacity_transitions=10**9)
    _, val = episode_split(buffer.episode_paths(), val_fraction=0.2, seed=0)

    embeds = np.random.default_rng(0).normal(size=(64, 2048))
    targets = np.random.default_rng(1).normal(size=(64, 4))
    weights = fit_probe(embeds, targets)

    result = evaluate_rollout(
        model, val[:2], weights, context=5, horizon=10,
        device=torch.device("cpu"), feature_backbone=encoder_backbone(cfg.encoder),
    )
    assert result.rssm_position.shape == (10,)
    assert np.isfinite(result.rssm_position).all()
    assert np.isfinite(result.persistence_position).all()
    assert np.isfinite(result.floor_position).all()

    # The band is REPORTED, not asserted. With the earlier posterior-latent
    # floor it was measured inverting at 8 of 10 horizon steps; the floor is now
    # the encoder-embedding probe per spec 3.2, which should behave better, but
    # "should" is not evidence and this plan does not assert unverified
    # relationships. Task 12 Step 5 records the real numbers.
    band = result.persistence_position - result.floor_position
    inverted = int((band <= 0).sum())
    print(
        f"\nband width: min={band.min():.3f} median={np.median(band):.3f} "
        f"max={band.max():.3f}; floor above persistence at {inverted}/{len(band)} steps"
    )
    if inverted:
        print(
            "NOTE: gap_closed is NaN at those steps by design -- a negative band "
            "would otherwise invert the sign of the M3 gate criterion."
        )
