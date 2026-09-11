"""The two M3b diagnostics: action-shuffled imagination, and the k-step sweep.

Both run on the SHIPPED checkpoints under `evaluate_rollout`'s exact protocol,
so the tests here are arranged around the three ways their numbers can be
meaningless while every shape stays right:

  * THE WINDOW SET. If the diagnostics score different windows from
    `evaluate_rollout`, their curves are not comparable to the records at all.
  * THE SAMPLING STREAM. `RSSM._sample` is stochastic at evaluation by design
    (see its docstring: taking the mode collapses 45 imagined latents to 3 and
    quadruples position error), so two arms that do not share a matched stream
    differ by sampling noise and the measured effect is uninterpretable.
    Measured on this box, a CPU-ONLY RNG snapshot (`torch.get_rng_state`)
    silently fails to restore the MPS generator, and the shipped records were
    produced on MPS -- so every stream test is parametrised over both devices
    and the MPS half is the only thing that can catch it.
  * THE RE-GROUNDING INDEX. Re-grounding through the frame being scored turns
    k=1 into the floor itself and the sweep says nothing. That is pinned twice:
    exactly, on a deterministic rig, and bitwise, on a real stochastic one.

RIG CHOICE IS LOAD-BEARING AND NEVER INCIDENTAL. The oracle rigs in
`test_rollout.py` draw no random numbers, so every stream mutation is invisible
against them; `_OracleRSSM` ignores `state`, so the warm-start mutation is
invisible against it too. Each test below says which rig it needs and why.
"""

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import mbfps.eval.diagnostics as diagnostics_module
import mbfps.eval.rollout as rollout_module
from mbfps.data.episode import save_episode
from mbfps.eval.diagnostics import (
    REGROUNDING_KS,
    RegroundingSweep,
    action_shuffled_rollout,
    regrounding_sweep,
)
from mbfps.eval.rollout import RolloutResult, evaluate_rollout
from mbfps.eval.windows import window_starts
from tests.eval.test_rollout import (
    CONTEXT,
    HORIZON,
    STEP,
    T_SYNTHETIC,
    DriftingModel,
    OracleModel,
    _OracleRSSM,
    _StatefulRSSM,
    oracle_probe,
    real_model_and_probe,
    synthetic_episode,
)

DEVICES = [torch.device("cpu")] + (
    [torch.device("mps")] if torch.backends.mps.is_available() else []
)
"""Both devices, because the CPU-only-RNG-snapshot defect is invisible on CPU."""

ONE_WINDOW = CONTEXT + HORIZON + 1
"""An episode length that yields exactly one window (`need + 1`)."""


# ---------------------------------------------------------------------------
# Fixtures and rigs.
# ---------------------------------------------------------------------------


def write(tmp_path, episode, index=0):
    path = tmp_path / f"ep_{index:06d}_len{episode.length:05d}.npz"
    save_episode(episode, path)
    return path


def uniform_action_episode(length=T_SYNTHETIC, action=3):
    """Every action identical, so ANY permutation of them is a genuine no-op."""
    episode = synthetic_episode(length=length)
    episode.actions = np.full(length, action, dtype=np.int32)
    return episode


def varied_action_episode(length=T_SYNTHETIC):
    """Each window's horizon actions are DISTINCT, and its multiset is its own.

    Distinct within a window (window i gets the values `i .. i + HORIZON - 1`,
    shuffled) so the permutation INDEX is recoverable from the sequence
    `imagine` was handed -- which is what makes "one permutation reused for
    every window" detectable at all. Different multisets across windows so
    borrowing another window's actions dies on the sorted comparison rather
    than passing on a coincidence.
    """
    episode = synthetic_episode(length=length)
    actions = np.arange(length, dtype=np.int32) % 6
    rng = np.random.default_rng(5)
    for index, start in enumerate(window_starts(length, CONTEXT, HORIZON)):
        horizon_slice = slice(start + CONTEXT, start + CONTEXT + HORIZON)
        actions[horizon_slice] = rng.permutation(np.arange(HORIZON) + index)
    episode.actions = actions
    return episode


class _ActionSumRSSM(_OracleRSSM):
    """An exact filter whose dynamics depend on the action VALUE.

    `imagine` advances the frame tag by the action's own value each step, so
    the imagined tag after j steps is the CUMULATIVE SUM of the actions -- a
    closed-form curve that a permutation of those actions moves at every
    interior step, and leaves EXACTLY invariant at the last one, since a
    permutation preserves the sum.
    """

    def imagine(self, actions, state):
        h0 = state[0]
        cumulative = torch.cumsum(actions.to(h0.dtype), dim=1).unsqueeze(-1)
        tag = h0.unsqueeze(1) + cumulative
        return {"h": tag, "z": tag, "latent": torch.cat([tag, tag], dim=-1)}


class ActionSumModel(OracleModel):
    def __init__(self) -> None:
        super().__init__()
        self.rssm = _ActionSumRSSM()


class _PoisonedObserveRSSM(_OracleRSSM):
    """Exact `h`/`z`, but every LATENT `observe` emits is tagged +1000.

    The state handed forward stays correct, so the imagination is unaffected;
    what changes is that any scored row taken from an `observe` output instead
    of from `imagine` is off by a thousand map-index units and cannot be
    mistaken for noise.
    """

    POISON = 1000.0

    def observe(self, embeddings, actions, state=None):
        tag = embeddings[..., :1]
        poisoned = tag + self.POISON
        return {"h": tag, "z": tag, "latent": torch.cat([poisoned, poisoned], dim=-1)}


class PoisonedObserveModel(OracleModel):
    def __init__(self) -> None:
        super().__init__()
        self.rssm = _PoisonedObserveRSSM()


class _DriftingStatefulRSSM(_StatefulRSSM):
    """`_StatefulRSSM`'s state-folding `observe` with BROKEN dynamics.

    `observe` genuinely depends on the incoming `state` (unlike `_OracleRSSM`,
    which ignores it), which is what makes dropping `state=state` from the
    re-grounding call an observable difference; `imagine` advances two frames
    per action, so a re-grounded segment and an open-loop one are also
    distinguishable.
    """

    def imagine(self, actions, state):
        h0 = state[0]
        steps = torch.arange(1, actions.shape[1] + 1, dtype=h0.dtype).view(1, -1, 1)
        tag = h0.unsqueeze(1) + 2 * steps
        return {"h": tag, "z": tag, "latent": torch.cat([tag, tag], dim=-1)}


class DriftingStatefulModel(OracleModel):
    def __init__(self) -> None:
        super().__init__()
        self.rssm = _DriftingStatefulRSSM()


def action_blind_model():
    """A REAL `RSSM` that provably ignores actions, and samples exactly as much.

    Zeroing the action one-hot inside `_step` leaves every tensor shape and
    every categorical draw identical to the real model's -- the number of
    uniforms `Categorical.sample` consumes depends on shape alone, measured on
    both CPU and MPS -- while making the dynamics independent of the action.
    This is the model for which a matched-stream shuffle must be BIT-identical:
    a random model would merely be close, and closeness is what the CPU-only
    snapshot defect also produces.
    """
    model, probe = real_model_and_probe()

    class _ActionBlindRSSM(type(model.rssm)):
        def _step(self, h, z, action_onehot):
            return super()._step(h, z, torch.zeros_like(action_onehot))

    model.rssm.__class__ = _ActionBlindRSSM
    return model, probe


def sweep(model, paths, probe, ks, *, horizon=HORIZON, device=None, **kwargs):
    return regrounding_sweep(
        model, paths, probe, ks=ks, context=CONTEXT, horizon=horizon,
        device=device or torch.device("cpu"), **kwargs,
    )


def shuffle(model, paths, probe, *, device=None, **kwargs):
    return action_shuffled_rollout(
        model, paths, probe, context=CONTEXT, horizon=HORIZON,
        device=device or torch.device("cpu"), **kwargs,
    )


def rollout(model, paths, probe, *, device=None):
    return evaluate_rollout(
        model, paths, probe, context=CONTEXT, horizon=HORIZON,
        device=device or torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
# The window set: comparability with the shipped records.
# ---------------------------------------------------------------------------


def _truth_slices(monkeypatch, module, run):
    seen: list[np.ndarray] = []
    real = module.probe_targets
    monkeypatch.setattr(
        module,
        "probe_targets",
        lambda privileged, keys: seen.append(np.asarray(privileged)) or real(privileged, keys),
    )
    run()
    return seen


@pytest.mark.parametrize("diagnostic", ["shuffle", "sweep"])
def test_the_diagnostics_score_the_rollouts_exact_window_set(
    tmp_path, monkeypatch, diagnostic
):
    """Compared as TRUTH SLICES, element for element -- not as counts.

    THE WINDOW-SET HALF OF THIS TEST CAN NO LONGER FAIL ON DIVERGENCE, the same
    caveat `test_gather_probe_data_windows_match_the_rollouts_length_guard_exactly`
    carries after the same extraction. `_diagnose` and `evaluate_rollout` both
    take their starts from `mbfps.eval.windows.window_starts`, so the two
    halves the fixture lengths were chosen to separate -- `need` for the length
    guard, `3 * need` for the stride's `+ 1` -- are now guaranteed by shared
    code and cannot diverge between these two callers. That separation lives in
    `tests/eval/test_windows.py`, on the iterator itself.

    What this test still carries is the rest of the protocol, which is NOT
    shared: the TRUTH SLICE, compared element for element rather than by count,
    so an off-by-one in `start + context + 1` -- which leaves the window count
    untouched and breaks no shape -- fails here; and a re-inline of the range
    inside `_diagnose`, which is caught because the fixture spans the exact
    lengths at which a re-inlined copy would differ. The three episode lengths
    are kept for that second reason.
    """
    need = CONTEXT + HORIZON
    episodes = [synthetic_episode(length=n) for n in (need, need + 1, 3 * need)]
    paths = [write(tmp_path, episode, index) for index, episode in enumerate(episodes)]
    probe = oracle_probe(episodes[-1])

    expected = _truth_slices(
        monkeypatch, rollout_module,
        lambda: rollout(OracleModel(), paths, probe),
    )
    run = (
        (lambda: shuffle(OracleModel(), paths, probe))
        if diagnostic == "shuffle"
        else (lambda: sweep(OracleModel(), paths, probe, ks=(1, HORIZON)))
    )
    got = _truth_slices(monkeypatch, diagnostics_module, run)

    assert len(got) == len(expected) > 0
    for mine, theirs in zip(got, expected):
        np.testing.assert_array_equal(mine, theirs)


def test_the_sweep_reports_the_window_count_it_actually_scored(tmp_path):
    """`windows_total` is what makes every other number in the sweep readable,
    so it is asserted against an independently computed count rather than
    merely being present."""
    lengths = (CONTEXT + HORIZON, CONTEXT + HORIZON + 1, 3 * (CONTEXT + HORIZON))
    episodes = [synthetic_episode(length=n) for n in lengths]
    paths = [write(tmp_path, episode, index) for index, episode in enumerate(episodes)]
    expected = sum(len(window_starts(n, CONTEXT, HORIZON)) for n in lengths)
    assert expected == 4, "fixture must span more than one window per episode"
    result = sweep(OracleModel(), paths, oracle_probe(episodes[-1]), ks=(1, HORIZON))
    assert result.windows_total == expected


# ---------------------------------------------------------------------------
# Self-check 1: k == horizon reproduces the open-loop rollout.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_the_sweep_at_k_equals_the_horizon_is_bitwise_the_open_loop_rollout(
    tmp_path, device
):
    """SELF-CHECK 1, in its fast form: equality, never a tolerance.

    Requires the REAL stochastic RSSM. Against an oracle rig no random number
    is drawn at all, so every mutation that displaces the sampling stream --
    a missing per-window restore, a torch-drawn permutation, a CPU-only RNG
    snapshot on MPS -- is invisible and this test proves nothing.

    TWO EPISODES, NOT ONE, and that is the point of the fixture rather than a
    detail. The design's central claim is that the canonical pass runs LAST so
    the global stream leaves each window exactly where `evaluate_rollout` leaves
    it -- which is a statement ACROSS windows, and the seeding happens once
    before the episode loop. On a single-episode fixture, moving
    `torch.manual_seed(seed)` inside the episode loop (a per-episode reseed --
    exactly the stream perturbation the whole design exists to prevent) changes
    nothing and the mutation survives. With two episodes the second episode's
    windows draw from a different point in the stream in the two runs, and the
    equality below fails.
    """
    paths = [
        write(tmp_path, synthetic_episode(), index) for index in range(2)
    ]
    model, probe = real_model_and_probe()
    model = model.to(device)

    result = sweep(model, paths, probe, ks=(1, HORIZON), device=device)
    reference = rollout(model, paths, probe, device=device)

    assert result.windows_total > 2, (
        "the fixture collapsed to a single episode's windows and the "
        "cross-episode half of this test is not being exercised"
    )
    np.testing.assert_array_equal(result.curve(HORIZON), reference.rssm_position)
    np.testing.assert_array_equal(result.curve(HORIZON, "angle"), reference.rssm_angle)
    assert result.open_loop_divergence(reference) == 0.0


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_the_sweeps_floor_and_persistence_are_bitwise_the_rollouts(tmp_path, device):
    """"Approaches the floor" is only a claim if it is the SAME floor. The
    sweep's lower and upper brackets must be the record's own, bit for bit, or
    a k-curve read against them is read against a different measurement."""
    path = write(tmp_path, synthetic_episode())
    model, probe = real_model_and_probe()
    model = model.to(device)

    result = sweep(model, [path], probe, ks=(1, HORIZON), device=device)
    reference = rollout(model, [path], probe, device=device)

    np.testing.assert_array_equal(
        result.reference.floor_position, reference.floor_position
    )
    np.testing.assert_array_equal(
        result.reference.persistence_position, reference.persistence_position
    )
    np.testing.assert_array_equal(result.reference.floor_angle, reference.floor_angle)


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_the_shuffles_real_arm_is_bitwise_the_open_loop_rollout(tmp_path, device):
    """The shuffle's own baseline must BE the shipped protocol, or the delta is
    measured against something the records never reported.

    TWO EPISODES, for the reason spelled out in
    `test_the_sweep_at_k_equals_the_horizon_is_bitwise_the_open_loop_rollout`:
    the "canonical pass last, so the stream is left where `evaluate_rollout`
    leaves it" claim is a cross-WINDOW one, and the traversal is seeded once
    before the episode loop. A single episode cannot separate that seeding from
    a per-episode reseed.
    """
    paths = [
        write(tmp_path, synthetic_episode(), index) for index in range(2)
    ]
    model, probe = real_model_and_probe()
    model = model.to(device)

    result = shuffle(model, paths, probe, device=device)
    reference = rollout(model, paths, probe, device=device)

    assert result.windows_total > 2, (
        "the fixture collapsed to a single episode's windows and the "
        "cross-episode half of this test is not being exercised"
    )
    np.testing.assert_array_equal(result.real.rssm_position, reference.rssm_position)
    np.testing.assert_array_equal(result.real.rssm_angle, reference.rssm_angle)
    np.testing.assert_array_equal(result.real.floor_position, reference.floor_position)
    np.testing.assert_array_equal(
        result.real.persistence_position, reference.persistence_position
    )


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_every_k_is_grounded_identically_at_the_first_horizon_step(tmp_path, device):
    """Horizon step 1 is one prior step from the context state for EVERY k --
    no re-grounding has happened yet -- so the only thing that can make the ks
    differ there is the sampling stream. Exactly one distinct value, on a real
    stochastic model over two windows.

    This is the assertion with a single interpretation: a spread at step 1 is
    stream drift, never re-grounding.
    """
    path = write(tmp_path, synthetic_episode())
    model, probe = real_model_and_probe()
    model = model.to(device)
    ks = (1, 2, 3, HORIZON)

    result = sweep(model, [path], probe, ks=ks, device=device)
    first = np.array([result.curve(k)[0] for k in ks])
    assert len(np.unique(first)) == 1, first


# ---------------------------------------------------------------------------
# Self-check 2: k=1 approaches the floor from above without reaching it.
# ---------------------------------------------------------------------------


def test_the_regrounded_curve_is_the_within_segment_position_under_a_drifting_oracle(
    tmp_path,
):
    """The whole sweep as a closed form, on a rig where the prior and the
    posterior genuinely differ.

    `DriftingModel` filters exactly but imagines two frames of travel per
    action, so `k` steps after a re-grounding the error is exactly `k` steps of
    true displacement: `STEP * ((j mod k) + 1)`. Against `OracleModel` every k
    scores zero and the sweep is vacuous, which is why the drifting rig is
    mandatory here.

    `k = 2` is the RAGGED case: 5 = 2 + 2 + 1, and every default k divides the
    shipped horizon of 45, so the short final segment never runs in production
    and would otherwise ship untested.
    """
    path = write(tmp_path, synthetic_episode())
    episode = synthetic_episode()
    ks = (1, 2, 3, HORIZON)
    result = sweep(DriftingModel(), [path], oracle_probe(episode), ks=ks)
    for k in ks:
        expected = STEP * ((np.arange(HORIZON) % k) + 1)
        np.testing.assert_allclose(result.curve(k), expected, rtol=1e-5, err_msg=f"k={k}")


def test_k_equal_one_sits_exactly_one_prior_step_above_the_floor(tmp_path):
    """SELF-CHECK 2, made deterministic so it rests on arithmetic, not a
    threshold.

    The floor is a ZERO-step posterior at every future step -- it sees the very
    frame it is scored on. k=1 is a ONE-step PRIOR from a posterior-grounded
    state. So k=1 must sit STRICTLY above the floor, and on this rig it sits
    exactly one step of true displacement above it. Re-grounding with the
    posterior OF the scored step instead collapses k=1 onto the floor and the
    sweep measures nothing.
    """
    path = write(tmp_path, synthetic_episode())
    episode = synthetic_episode()
    result = sweep(DriftingModel(), [path], oracle_probe(episode), ks=(1, HORIZON))
    np.testing.assert_allclose(result.curve(1), np.full(HORIZON, STEP), rtol=1e-5)
    np.testing.assert_allclose(
        result.reference.floor_position, np.zeros(HORIZON), atol=1e-6
    )
    assert (result.curve(1) > result.reference.floor_position).all()


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_k_equal_one_is_never_bitwise_the_floor_curve(tmp_path, device):
    """The same off-by-one, in the form that survives on real checkpoints.

    On the shipped models the prior/posterior separation may be smaller than
    the sampling noise, so "k=1 is above the floor" can be satisfied or
    defeated by chance. Re-grounding through the scored frame does not merely
    shrink the gap -- it makes the k=1 curve the floor TENSOR, so bitwise
    identity is a noise-free alarm.
    """
    path = write(tmp_path, synthetic_episode())
    model, probe = real_model_and_probe()
    model = model.to(device)
    result = sweep(model, [path], probe, ks=(1, HORIZON), device=device)
    assert result.is_bitwise_the_floor(1, "position") is False
    assert result.is_bitwise_the_floor(1, "angle") is False

    # And the alarm can fire: handed a k=1 curve that HAS collapsed onto the
    # floor, it says so. Without this half, "is False" would be satisfied by a
    # predicate that is False for everything.
    collapsed = replace(
        result, position=dict(result.position) | {1: result.reference.floor_position}
    )
    assert collapsed.is_bitwise_the_floor(1, "position") is True


def test_the_re_grounding_replays_the_segment_and_stops_one_frame_short_of_the_scored_one(
    tmp_path, monkeypatch
):
    """The index arithmetic itself, read out of the tensors rather than
    inferred from an error curve -- two compensating index errors can leave a
    curve looking right.

    `_TagEncoder` makes every embedding row its own frame index, so the last
    row a re-grounding `observe` consumed is checked directly against the frame
    the NEXT segment's first scored step predicts, minus one.
    """
    path = write(tmp_path, synthetic_episode(length=ONE_WINDOW))
    episode = synthetic_episode(length=ONE_WINDOW)
    model = DriftingModel()
    k = 2
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    real = model.rssm.observe
    monkeypatch.setattr(
        model.rssm,
        "observe",
        lambda embeddings, actions, state=None: (
            calls.append((embeddings.clone(), actions.clone()))
            or real(embeddings, actions, state=state)
        ),
    )
    sweep(model, [path], oracle_probe(episode), ks=(k, HORIZON))

    # calls[0] is the context filter; the floor's is last. Between them sit the
    # re-groundings, in segment order, for k=2 then for k=HORIZON (which has
    # none). The single window starts at 0.
    regroundings = calls[1:-1]
    assert len(regroundings) == len(range(k, HORIZON, k)), calls
    for index, (embeddings, grounding_actions) in enumerate(regroundings):
        segment_start = (index + 1) * k       # first horizon step of the next segment
        first_scored_frame = CONTEXT + 1 + segment_start
        # THE WHOLE previous segment is replayed, not just its last frame.
        # `RSSM.observe` advances `h` with every action before forming each
        # posterior, so a single-frame grounding hands the next segment an `h`
        # that saw one action instead of k -- and it breaks no shape.
        seen = embeddings[0, :, 0].cpu().numpy()
        expected = np.arange(first_scored_frame - k, first_scored_frame)
        np.testing.assert_array_equal(seen, expected, err_msg=f"segment {index}")
        np.testing.assert_array_equal(
            grounding_actions[0].cpu().numpy(),
            episode.actions[CONTEXT + index * k : CONTEXT + (index + 1) * k],
        )


def test_the_re_grounding_observe_is_warm_started_from_the_previous_segment(tmp_path):
    """`RSSM.observe` carries a belief forward; re-grounding from a ZERO state
    would hand each segment a filter that has forgotten everything before it.

    Invisible against `_OracleRSSM` (which ignores `state`) and, measured, worth
    ~1e-4 on an untrained real model -- so `_DriftingStatefulRSSM`, whose
    `observe` folds the incoming state into every output, is what makes it
    observable. One window starting at 0, so the context state's tag is exactly
    CONTEXT and the k=1 curve is the hand-traced [1, 4, 8, 13, 19] tag units.
    A cold-started re-grounding gives the flat [1, 1, 1, 1, 1] instead --
    which is exactly the correct curve for the state-free `DriftingModel`, so
    the two rigs must not be confused.
    """
    path = write(tmp_path, synthetic_episode(length=ONE_WINDOW))
    episode = synthetic_episode(length=ONE_WINDOW)
    result = sweep(DriftingStatefulModel(), [path], oracle_probe(episode), ks=(1, HORIZON))
    np.testing.assert_allclose(
        result.curve(1), STEP * np.array([1.0, 4.0, 8.0, 13.0, 19.0]), rtol=1e-4
    )


def test_every_scored_latent_comes_from_imagine_and_none_from_the_re_grounding(
    tmp_path,
):
    """Provenance, asserted on the scored tensor.

    `_PoisonedObserveRSSM` keeps `h`/`z` exact -- so the imagination is
    unaffected -- while tagging every latent `observe` emits by +1000. A model
    curve that stays at zero therefore proves no segment boundary was scored
    from the re-grounding posterior; borrowing one would show up as a thousand
    map-index units at exactly that step.
    """
    path = write(tmp_path, synthetic_episode())
    episode = synthetic_episode()
    result = sweep(PoisonedObserveModel(), [path], oracle_probe(episode), ks=(2, 3, HORIZON))
    for k in (2, 3, HORIZON):
        np.testing.assert_allclose(
            result.curve(k), np.zeros(HORIZON), atol=1e-5, err_msg=f"k={k}"
        )


@pytest.mark.parametrize(
    "ks,expected",
    [((HORIZON,), 2), ((2, HORIZON), 4), ((1, HORIZON), 6)],
)
def test_the_sweep_re_grounds_between_segments_and_never_after_the_last_one(
    tmp_path, monkeypatch, ks, expected
):
    """`observe` calls per window: the context filter, the floor, and one
    re-grounding per segment BOUNDARY -- `ceil(H/k) - 1`, not `ceil(H/k)`.

    Three values of k, because a single one leaves the boundary case
    unexercised: at k == HORIZON there is no boundary at all and the count must
    be exactly two.
    """
    path = write(tmp_path, synthetic_episode(length=ONE_WINDOW))
    episode = synthetic_episode(length=ONE_WINDOW)
    model = OracleModel()
    calls = []
    real = model.rssm.observe
    monkeypatch.setattr(
        model.rssm,
        "observe",
        lambda embeddings, actions, state=None: calls.append(1)
        or real(embeddings, actions, state=state),
    )
    sweep(model, [path], oracle_probe(episode), ks=ks)
    assert len(calls) == expected


def test_the_sweep_scores_every_horizon_step_exactly_once_for_every_k(
    tmp_path, monkeypatch
):
    """Segment structure, asserted rather than merely produced: `ceil(H/k)`
    segments, none longer than k, summing to exactly H imagined steps."""
    path = write(tmp_path, synthetic_episode(length=ONE_WINDOW))
    episode = synthetic_episode(length=ONE_WINDOW)
    model = OracleModel()
    lengths = []
    real = model.rssm.imagine
    monkeypatch.setattr(
        model.rssm,
        "imagine",
        lambda actions, state: lengths.append(int(actions.shape[1]))
        or real(actions, state),
    )
    k = 2
    sweep(model, [path], oracle_probe(episode), ks=(k, HORIZON))
    # Two open-loop calls of the full horizon -- the k == HORIZON arm and the
    # canonical pass -- plus this k's own segments. Compared as an unordered
    # structure so a legitimate reordering of the arms does not fail it.
    assert lengths.count(HORIZON) == 2, lengths
    segments = [n for n in lengths if n != HORIZON]
    assert len(segments) == -(-HORIZON // k), lengths
    assert sum(segments) == HORIZON, lengths
    assert max(segments) <= k, lengths


@pytest.mark.parametrize("ks", [(0, HORIZON), (-1, HORIZON), (HORIZON + 1, HORIZON)])
def test_the_sweep_rejects_a_k_outside_one_to_the_horizon(tmp_path, ks):
    """k=0 makes the segment loop non-terminating deep inside the traversal,
    after the encoder work; k above the horizon silently means "open loop"
    under a name that claims otherwise."""
    path = write(tmp_path, synthetic_episode())
    episode = synthetic_episode()
    with pytest.raises(ValueError, match="regrounding period"):
        sweep(OracleModel(), [path], oracle_probe(episode), ks=ks)


def test_the_sweep_refuses_a_ks_that_omits_the_horizon_itself(tmp_path):
    """The k == horizon pass IS self-check 1. A sweep without it reports curves
    nothing has checked against the shipped protocol, so it is refused rather
    than run self-check-free."""
    path = write(tmp_path, synthetic_episode())
    episode = synthetic_episode()
    with pytest.raises(ValueError, match="self-check"):
        sweep(OracleModel(), [path], oracle_probe(episode), ks=(1, 2))


# ---------------------------------------------------------------------------
# Diagnostic 1: the action-shuffled imagination.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_a_model_whose_dynamics_ignore_actions_is_bit_identical_under_the_permutation(
    tmp_path, device
):
    """THE reference behaviour the diagnostic exists to detect, on a real
    sampling RSSM whose `_step` provably discards the action.

    Bit-identical, not close: closeness is also what a broken matched stream
    produces. `windows_changed == windows_total > 0` is asserted in the SAME
    test so the equality cannot be explained by "nothing was permuted", and it
    also rules out the truth slice being permuted alongside the actions -- that
    would move this curve.
    """
    path = write(tmp_path, varied_action_episode())
    model, probe = action_blind_model()
    model = model.to(device)
    result = shuffle(model, [path], probe, device=device)

    np.testing.assert_array_equal(result.shuffled_position, result.real.rssm_position)
    np.testing.assert_array_equal(result.shuffled_angle, result.real.rssm_angle)
    assert result.windows_changed == result.windows_total > 0


def test_a_model_whose_dynamics_use_the_action_moves_under_the_permutation(
    tmp_path, monkeypatch
):
    """The positive half, asserted as exact closed-form values rather than as
    "the curves differ" -- which sampling noise alone also satisfies.

    `ActionSumModel` advances the frame tag by the action's own value, so the
    imagined tag after j steps is the cumulative sum of the actions it was
    handed. The expectation is computed from the action tensor the spy captured
    ON THE WAY INTO `imagine`, so a diagnostic that permutes something else, or
    nothing, cannot satisfy it.
    """
    path = write(tmp_path, varied_action_episode())
    episode = varied_action_episode()
    model = ActionSumModel()
    seen: list[np.ndarray] = []
    real = model.rssm.imagine
    monkeypatch.setattr(
        model.rssm,
        "imagine",
        lambda actions, state: seen.append(actions[0].cpu().numpy().copy())
        or real(actions, state),
    )
    result = shuffle(model, [path], oracle_probe(episode))

    # Two arms per window, in call order: the permuted one, then the canonical.
    permuted, canonical = seen[0::2], seen[1::2]
    steps = np.arange(1, HORIZON + 1)
    expected = lambda arms: np.mean(  # noqa: E731
        [STEP * np.abs(np.cumsum(a) - steps) for a in arms], axis=0
    )
    # atol only because the fixture happens to make one step exactly zero; the
    # other four are tens of map units and carry the assertion.
    np.testing.assert_allclose(
        result.shuffled_position, expected(permuted), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        result.real.rssm_position, expected(canonical), rtol=1e-5, atol=1e-6
    )
    assert not np.array_equal(result.shuffled_position, result.real.rssm_position)


def test_the_permuted_curve_is_not_pinned_by_its_endpoint_alone(tmp_path):
    """A permutation preserves the action MULTISET, so any dynamics that depend
    only on the sum -- a plausible degenerate solution -- are indistinguishable
    from action-blind at the FINAL horizon step, which is the study's headline
    number. `ActionSumModel` is exactly such a model: its last step is
    invariant under the permutation while its interior steps are not.

    So the delta CURVE must be read, never its endpoint. This test exists to
    make that trap visible, and it fails if the diagnostic is ever "fixed" by
    resampling actions instead of permuting them.
    """
    path = write(tmp_path, varied_action_episode())
    episode = varied_action_episode()
    result = shuffle(ActionSumModel(), [path], oracle_probe(episode))
    assert result.shuffled_position[-1] == pytest.approx(
        result.real.rssm_position[-1], rel=1e-6
    )
    assert result.shuffled_position[1] != pytest.approx(
        result.real.rssm_position[1], rel=1e-6
    )


def test_the_permuted_actions_reach_imagine_and_preserve_the_multiset(
    tmp_path, monkeypatch
):
    """The structural half: what `imagine` was handed, per window.

    Each window's permuted sequence must be a rearrangement of THAT window's
    own horizon actions -- the fixture gives adjacent windows disjoint action
    alphabets, so borrowing another window's actions or reusing one window's
    permutation index dies on the multiset comparison rather than passing on a
    coincidence.
    """
    episode = varied_action_episode()
    path = write(tmp_path, episode)
    model = ActionSumModel()
    seen: list[np.ndarray] = []
    real = model.rssm.imagine
    monkeypatch.setattr(
        model.rssm,
        "imagine",
        lambda actions, state: seen.append(actions[0].cpu().numpy().copy())
        or real(actions, state),
    )
    shuffle(model, [path], oracle_probe(episode))

    starts = window_starts(episode.length, CONTEXT, HORIZON)
    assert len(starts) >= 2, "fixture must span more than one window"
    permuted = seen[0::2]
    orders = []
    for index, start in enumerate(starts):
        window = episode.actions[start + CONTEXT : start + CONTEXT + HORIZON]
        np.testing.assert_array_equal(np.sort(permuted[index]), np.sort(window))
        assert not np.array_equal(permuted[index], window), index
        # The window's actions are distinct, so the permutation INDEX itself is
        # recoverable -- without that, one index reused across every window
        # still produces different sequences and goes unnoticed.
        lookup = {int(value): position for position, value in enumerate(window)}
        orders.append(tuple(lookup[int(value)] for value in permuted[index]))
    assert len(set(orders)) > 1, "every window was permuted by the same index"


def test_only_the_imagination_sees_the_permuted_actions(tmp_path, monkeypatch):
    """The intervention's SCOPE. The context filter and the floor are the fixed
    brackets of the comparison: if either moved, the measured delta would not
    isolate the dynamics prior, and the floor -- which consumes the horizon
    actions too -- would stop being the record's own."""
    episode = varied_action_episode()
    path = write(tmp_path, episode)
    model = ActionSumModel()
    seen: list[np.ndarray] = []
    real = model.rssm.observe
    monkeypatch.setattr(
        model.rssm,
        "observe",
        lambda embeddings, actions, state=None: seen.append(actions[0].cpu().numpy().copy())
        or real(embeddings, actions, state=state),
    )
    shuffle(model, [path], oracle_probe(episode))

    context_calls, floor_calls = seen[0::2], seen[1::2]
    for index, start in enumerate(window_starts(episode.length, CONTEXT, HORIZON)):
        np.testing.assert_array_equal(
            context_calls[index], episode.actions[start : start + CONTEXT]
        )
        np.testing.assert_array_equal(
            floor_calls[index],
            episode.actions[start + CONTEXT : start + CONTEXT + HORIZON],
        )


def test_a_window_whose_horizon_actions_are_already_uniform_is_reported_unchanged(
    tmp_path,
):
    """Measured on the shipped split, 229 of 229 windows really change, so this
    whole variant is unreachable from real data and can only be exercised here
    -- which is exactly why it is a first-class output rather than a debug aid.

    A no-difference result computed over windows where the permutation was a
    no-op says nothing about the dynamics, so the delta is NaN there --
    undefined, following `gap_closed`'s policy -- and never a zero that reads
    as "the model ignored the actions".
    """
    episode = uniform_action_episode()
    path = write(tmp_path, episode)
    result = shuffle(ActionSumModel(), [path], oracle_probe(episode))

    assert result.windows_total > 0
    assert result.windows_changed == 0
    assert result.is_interpretable() is False
    assert np.isnan(result.position_delta()).all()
    assert np.isnan(result.angle_delta()).all()
    # The underlying curves are still real numbers: it is the DELTA that is
    # undefined, not the measurement.
    np.testing.assert_array_equal(result.shuffled_position, result.real.rssm_position)


def test_windows_changed_counts_action_sequences_not_index_permutations(
    tmp_path, monkeypatch
):
    """"The actions were not uniform" is not the same claim as "the
    intervention intervened". An identity permutation on a varied window
    changes nothing, and must be counted as unchanged.

    Forced by a stub generator that returns the identity order, on the VARIED
    fixture -- so a `changed` computed from the permutation index, or from
    whether the window's actions were uniform, miscounts it.
    """
    episode = varied_action_episode()
    path = write(tmp_path, episode)

    class _IdentityRng:
        def permutation(self, n):
            return np.arange(n)

    monkeypatch.setattr(
        diagnostics_module, "default_rng", lambda seed: _IdentityRng()
    )
    result = shuffle(ActionSumModel(), [path], oracle_probe(episode))
    assert result.windows_total > 0
    assert result.windows_changed == 0


def test_windows_changed_equals_windows_total_when_every_window_really_moves(tmp_path):
    """The other side of the counter, against an independently computed number
    -- a hardcoded `windows_changed = windows_total`, or a count of episodes
    rather than windows, satisfies only one of the two tests."""
    episode = varied_action_episode()
    path = write(tmp_path, episode)
    expected = len(window_starts(episode.length, CONTEXT, HORIZON))
    assert expected >= 2
    result = shuffle(ActionSumModel(), [path], oracle_probe(episode))
    assert result.windows_total == expected
    assert result.windows_changed == expected
    assert result.is_interpretable() is True


def test_the_delta_averages_only_the_windows_the_permutation_actually_changed(
    tmp_path,
):
    """A window the permutation left alone contributes an exact zero, which
    DILUTES the measured effect toward "no difference" -- the very conclusion
    the diagnostic is being used to draw. Two episodes, one varied and one
    uniform: the delta must be the varied window's own, not half of it.
    """
    varied = varied_action_episode(length=ONE_WINDOW)
    uniform = uniform_action_episode(length=ONE_WINDOW)
    paths = [write(tmp_path, varied, 0), write(tmp_path, uniform, 1)]
    probe = oracle_probe(varied)
    model = ActionSumModel()

    both = shuffle(model, paths, probe)
    alone = shuffle(model, [paths[0]], probe)

    assert both.windows_total == 2 and both.windows_changed == 1
    np.testing.assert_allclose(both.position_delta(), alone.position_delta(), rtol=1e-6)
    assert np.abs(alone.position_delta()).max() > 0.0


def test_the_permutation_is_reproducible_under_its_seed_and_moves_with_it(tmp_path):
    """Both halves. Without the seed the diagnostic is not reproducible; with a
    seed that no longer reaches the permutation it is not an intervention."""
    episode = varied_action_episode()
    path = write(tmp_path, episode)
    model, probe = ActionSumModel(), oracle_probe(episode)

    first = shuffle(model, [path], probe, permutation_seed=0).shuffled_position
    again = shuffle(model, [path], probe, permutation_seed=0).shuffled_position
    other = shuffle(model, [path], probe, permutation_seed=1).shuffled_position

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_the_shuffle_records_the_permutation_seed_it_used(tmp_path):
    """The seed is part of the result, not part of the run's shell history: a
    single permutation is one draw from a null distribution, and reading the
    number later requires knowing which draw it was."""
    episode = varied_action_episode()
    path = write(tmp_path, episode)
    result = shuffle(ActionSumModel(), [path], oracle_probe(episode), permutation_seed=7)
    assert result.permutation_seed == 7


# ---------------------------------------------------------------------------
# Shared guards.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("diagnostic", ["shuffle", "sweep"])
def test_the_diagnostics_read_the_arms_own_namespaced_feature_cache(
    tmp_path, diagnostic
):
    """Two of the three arms feed the RSSM cached features rather than pixels,
    and the caches are namespaced per backbone -- a diagnostic that reached for
    `episode.obs`, or for an unnamespaced `.features.npy`, would report one
    arm's numbers under another's.

    The cache carries the frame tag OFFSET BY +100, so reading the wrong source
    is a closed-form 100 map-index units of error rather than merely wrong
    provenance.
    """
    episode = synthetic_episode()
    path = write(tmp_path, episode)
    tags = np.arange(episode.length + 1, dtype=np.float32) + 100.0
    np.save(
        path.with_suffix(".features_random_vit.npy"),
        tags.reshape(-1, 1, 1, 1),
    )
    model = OracleModel()
    model.input_kind = "features"
    probe = oracle_probe(episode)

    run = (
        (lambda: shuffle(model, [path], probe, feature_backbone="random_vit").real)
        if diagnostic == "shuffle"
        else (
            lambda: sweep(
                model, [path], probe, ks=(HORIZON,), feature_backbone="random_vit"
            ).reference
        )
    )
    reference = run()
    np.testing.assert_allclose(
        reference.rssm_position, np.full(HORIZON, 100.0 * STEP), rtol=1e-4
    )


@pytest.mark.parametrize("diagnostic", ["shuffle", "sweep"])
def test_every_arm_is_probed_by_the_same_probe_through_the_embedding_head(
    tmp_path, monkeypatch, diagnostic
):
    """The single-pipeline rule extended to the intervention arms.

    The band only measures information content if every reference AND every arm
    goes through `encode -> RSSM -> embedding head` and is scored by ONE probe.
    Two claims, both asserted: exactly one probe object is ever used, and every
    probed row is a row the embedding head emitted -- so an arm scored on the
    raw latent, or through a probe refit inside the diagnostic, is caught.

    A REAL model, because on the oracle rig the head is the identity on the
    latent's first column and the two are indistinguishable.
    """
    path = write(tmp_path, synthetic_episode())
    model, probe = real_model_and_probe()

    head_rows: set[bytes] = set()
    real_heads = model.heads.forward

    def spy_heads(latent):
        out = real_heads(latent)
        for row in out["embedding"].reshape(-1, out["embedding"].shape[-1]):
            head_rows.add(row.cpu().numpy().tobytes())
        return out

    probed: list[tuple[int, np.ndarray]] = []
    real_apply = diagnostics_module.apply_probe
    monkeypatch.setattr(model.heads, "forward", spy_heads)
    monkeypatch.setattr(
        diagnostics_module,
        "apply_probe",
        lambda p, latents: probed.append((id(p), np.asarray(latents)))
        or real_apply(p, latents),
    )
    ks = (1, HORIZON)
    if diagnostic == "shuffle":
        shuffle(model, [path], probe)
        expected = (3 + 1) * 2       # three references plus one arm, two windows
    else:
        sweep(model, [path], probe, ks=ks)
        expected = (3 + len(ks)) * 2

    assert len(probed) == expected, len(probed)
    assert len({identity for identity, _ in probed}) == 1
    for _, rows in probed:
        for row in rows:
            assert row.astype(np.float32).tobytes() in head_rows


@pytest.mark.parametrize("diagnostic", ["shuffle", "sweep"])
def test_the_diagnostics_raise_when_no_window_is_long_enough(tmp_path, diagnostic):
    """A silently empty or zero-filled curve would be read as a measurement."""
    episode = synthetic_episode(length=CONTEXT + HORIZON)  # exactly `need`: excluded
    path = write(tmp_path, episode)
    probe = oracle_probe(episode)
    with pytest.raises(ValueError, match="no diagnostic window"):
        if diagnostic == "shuffle":
            shuffle(OracleModel(), [path], probe)
        else:
            sweep(OracleModel(), [path], probe, ks=(HORIZON,))


@pytest.mark.parametrize("diagnostic", ["shuffle", "sweep"])
def test_the_diagnostics_run_in_eval_mode_and_take_no_gradient(
    tmp_path, monkeypatch, diagnostic
):
    """Sampled AT the moment the RSSM is called, not after the call returns --
    asserting `model.training is False` afterwards cannot fail, because the
    function's own `model.eval()` has already run by then."""
    path = write(tmp_path, synthetic_episode())
    episode = synthetic_episode()
    model = OracleModel()
    model.train(True)
    states = []
    real = model.rssm.imagine
    monkeypatch.setattr(
        model.rssm,
        "imagine",
        lambda actions, state: states.append(
            (model.training, torch.is_grad_enabled())
        )
        or real(actions, state),
    )
    probe = oracle_probe(episode)
    if diagnostic == "shuffle":
        shuffle(model, [path], probe)
    else:
        sweep(model, [path], probe, ks=(1, HORIZON))
    assert states
    assert not any(training for training, _ in states)
    assert not any(grad for _, grad in states)


def test_the_diagnostics_defaults_are_the_spec_values():
    """context=5, horizon=45, seed=0 are the numbers the nine records were
    produced at; the ks are the sweep's own spec. Every other test here
    parametrises over TEST-LOCAL ks, so shrinking this tuple cannot silently
    shrink the test matrix -- this is the one test that reads it."""
    assert REGROUNDING_KS == (1, 3, 5, 15, 45)
    for function in (action_shuffled_rollout, regrounding_sweep):
        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(function).parameters.items()
        }
        assert defaults["context"] == 5, function.__name__
        assert defaults["horizon"] == 45, function.__name__
        assert defaults["seed"] == 0, function.__name__
        assert defaults["device"] is None, function.__name__
        assert defaults["feature_backbone"] is None, function.__name__
    assert inspect.signature(regrounding_sweep).parameters["ks"].default is REGROUNDING_KS
    assert (
        inspect.signature(action_shuffled_rollout).parameters["permutation_seed"].default
        == 0
    )


# ---------------------------------------------------------------------------
# The RNG snapshot: the one guard whose failure has no symptom at all.
# ---------------------------------------------------------------------------


def test_the_rng_snapshot_refuses_a_device_it_has_no_verified_restore_for():
    """The refusal branch, exercised directly, because nothing else reaches it.

    A CPU-only snapshot on an accelerator restores the CPU generator and leaves
    the accelerator's where it had got to, so the arms stop sharing a stream --
    measured on MPS, that inflates the apparent action effect by ~100x while
    shapes, curves and window counts all still look right. On a CUDA box this
    guard is the only thing between that artefact and a published number, and
    it is reached on `device.type` alone: no such device need be present.

    The CPU half is asserted in the same test because "raises for everything"
    would satisfy the first half on its own.
    """
    with pytest.raises(NotImplementedError, match="no verified RNG snapshot"):
        diagnostics_module._rng_snapshot(torch.device("cuda"))

    state = diagnostics_module._rng_snapshot(torch.device("cpu"))
    assert set(state) == {"cpu"}, state


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="the MPS half needs MPS"
)
def test_the_rng_snapshot_captures_the_accelerators_generator_as_well_as_the_cpus():
    """The half the whole guard exists for: on MPS, BOTH generators are taken.

    Capturing only `cpu` here is exactly the silent defect `_rng_snapshot`
    refuses to fall back to, and it passes every CPU test.
    """
    state = diagnostics_module._rng_snapshot(torch.device("mps"))
    assert set(state) == {"cpu", "mps"}, state


# ---------------------------------------------------------------------------
# The rulers: what a difference between two reported numbers has to clear.
# ---------------------------------------------------------------------------


def _hand_built_sweep(window_curves: dict, floor_rows=None, horizon=None, angle=None):
    """A `RegroundingSweep` from per-window rows typed out here.

    Every ruler below is a few lines of arithmetic over `window_position`, and
    driving them from a real traversal would pin them against the very code
    that computes them. `window_curves` maps k -> (n_windows, horizon) rows;
    the k-curves are their means, so the aggregate and the per-window rows
    cannot disagree.

    `floor_rows` is the floor PER WINDOW, and its rows are deliberately never
    built by tiling its own mean. A tiled floor makes
    `window_floor_position - mean` identically zero, so the paired floor margin
    and an unpaired-against-the-mean one agree and the mutation that swaps them
    survives -- the L1 fixture coincidence this repo names.
    """
    window_curves = {k: np.asarray(v, float) for k, v in window_curves.items()}
    rows_n, steps = next(iter(window_curves.values())).shape
    horizon = max(window_curves) if horizon is None else horizon
    if floor_rows is None:
        # Varies across windows, so a floor read as its own mean is a different
        # number rather than the same one.
        floor_rows = np.arange(rows_n * steps, dtype=float).reshape(rows_n, steps)
    floor_rows = np.asarray(floor_rows, float)
    assert floor_rows.shape == (rows_n, steps)
    position = {k: rows.mean(axis=0) for k, rows in window_curves.items()}
    return RegroundingSweep(
        ks=tuple(window_curves),
        horizon=horizon,
        reference=RolloutResult(
            horizon=np.arange(1, steps + 1),
            rssm_position=position[horizon],
            floor_position=floor_rows.mean(axis=0),
            persistence_position=np.full(steps, 99.0),
            rssm_angle=(position[horizon] if angle is None else np.asarray(angle, float)),
            floor_angle=np.zeros(steps),
            persistence_angle=np.full(steps, 44.0),
        ),
        position=position,
        angle={k: rows.mean(axis=0) for k, rows in window_curves.items()},
        window_position=window_curves,
        window_floor_position=floor_rows,
        windows_total=rows_n,
    )


def test_a_curve_is_refused_for_a_metric_the_sweep_does_not_carry():
    """`curve` is what every reported number is built from, and it used to fall
    back to the ANGLE curve for any string that was not exactly "position" --
    a typo silently returned degrees under a map-units heading. Its siblings
    already raise, so this also makes the class consistent about it."""
    swept = _hand_built_sweep({1: [[1.0, 2.0]], 2: [[3.0, 4.0]]})
    with pytest.raises(KeyError, match="postion"):
        swept.curve(1, "postion")
    np.testing.assert_array_equal(swept.curve(1, "angle"), swept.angle[1])


def test_the_open_loop_divergence_fires_on_an_angle_only_divergence(tmp_path):
    """SELF-CHECK 1 covers BOTH metrics, and only the position half is
    reachable from a real traversal.

    `test_the_sweep_at_k_equals_the_horizon_is_bitwise_the_open_loop_rollout`
    asserts the angle curve separately and then asserts the divergence is 0.0,
    so that second assertion is satisfied by the first no matter what the
    method actually covers -- restricting its generator to `("position",)`
    survives. The script's EXIT_PROTOCOL_DIVERGED gate reads ONLY this method,
    so an angle-only divergence would exit OK.
    """
    swept = _hand_built_sweep({1: [[1.0, 2.0]], 2: [[3.0, 4.0]]}, angle=[3.0, 99.0])
    reference = swept.reference
    # Position agrees exactly; angle does not.
    np.testing.assert_array_equal(swept.curve(2), reference.rssm_position)
    assert swept.open_loop_divergence(reference) == pytest.approx(95.0)


def test_a_curves_own_spread_is_its_window_standard_error_at_its_own_k():
    """`curve_standard_error` computes the spread of THIS k's rows.

    Reading `window_position[self.horizon]` instead of `[k]` gives every k the
    open-loop curve's spread, which is a plausible positive number of the same
    magnitude -- so the k's rows are deliberately made to differ from the
    horizon k's, and the value is asserted exactly rather than for finiteness.
    Dropping the `/ sqrt(n)` is a factor of ~15 on the shipped 229 windows.
    """
    rows = {
        1: np.array([[1.0, 2.0], [3.0, 10.0], [5.0, 12.0]]),
        2: np.array([[0.0, 0.0], [0.0, 0.0], [30.0, 60.0]]),
    }
    swept = _hand_built_sweep(rows)
    for k in (1, 2):
        expected = rows[k].std(axis=0, ddof=1) / np.sqrt(3)
        np.testing.assert_allclose(swept.curve_standard_error(k), expected)
    assert not np.allclose(
        swept.curve_standard_error(1), swept.curve_standard_error(2)
    ), "the two ks must have different spreads or the k lookup is not separated"


def test_a_curves_own_spread_is_zero_rather_than_undefined_on_one_window():
    """A single window has no spread to estimate. Zeros, not NaN and not inf:
    the sweep table prints this column and a NaN there formats as a hole while
    an inf makes every k column read as unresolvable."""
    swept = _hand_built_sweep({1: [[1.0, 2.0]], 2: [[3.0, 4.0]]})
    np.testing.assert_array_equal(swept.curve_standard_error(1), np.zeros(2))


def test_only_the_position_curves_are_retained_per_window():
    """The per-window angle rows are not kept, so asking for their spread is a
    KeyError rather than a silently wrong number off the position rows."""
    swept = _hand_built_sweep({1: [[1.0, 2.0]], 2: [[3.0, 4.0]]})
    with pytest.raises(KeyError, match="angle"):
        swept.curve_standard_error(1, "angle")


def test_a_difference_between_two_ks_is_ruled_by_their_PAIRED_spread():
    """The k columns are means over the SAME windows, so the ruler for a
    difference between them is the spread of the per-window DIFFERENCE.

    Per-window error is dominated by window difficulty, which is common to
    every k, so the unpaired between-window spread can be arbitrarily larger
    than the paired one and declares real separations unresolvable. This is the
    same multiple-comparison-class error the shuffle already avoids by going
    paired through `window_position_delta`.

    The fixture makes the point quantitative: the two ks differ by exactly 2.0
    in every window, so the paired spread is EXACTLY zero while each curve's
    own spread is large.
    """
    rows = {
        1: np.array([[10.0, 20.0], [50.0, 90.0], [200.0, 400.0]]),
        2: np.array([[12.0, 22.0], [52.0, 92.0], [202.0, 402.0]]),
    }
    swept = _hand_built_sweep(rows)
    np.testing.assert_allclose(swept.paired_standard_error(1, 2), np.zeros(2), atol=1e-12)
    assert (swept.curve_standard_error(1) > 30.0).all(), (
        "the fixture must make the unpaired spread large, or the two rulers are "
        "not separated"
    )

    # And it is a real standard error, not a hardcoded zero.
    uneven = _hand_built_sweep({
        1: np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        2: np.array([[1.0, 4.0], [2.0, 8.0], [6.0, 12.0]]),
    })
    difference = np.array([[1.0, 4.0], [2.0, 8.0], [6.0, 12.0]])
    np.testing.assert_allclose(
        uneven.paired_standard_error(1, 2),
        difference.std(axis=0, ddof=1) / np.sqrt(3),
    )


def test_the_margin_between_a_k_and_the_floor_is_paired_against_the_same_windows():
    """Self-check 2's quantitative half needs the floor PER WINDOW.

    "k=1 sits above the floor" is a paired comparison -- the same windows, the
    same RNG snapshot -- so the sweep retains the floor's own per-window rows
    rather than only its mean. Without them there is no paired ruler for the
    k-vs-floor margin at all, and the unpaired curve spread is the wrong one by
    the same argument as between two ks.
    """
    floor_rows = np.array([[1.0, 2.0], [7.0, 20.0]])
    swept = _hand_built_sweep(
        {1: np.array([[3.0, 5.0], [4.0, 6.0]]), 2: np.array([[9.0, 9.0], [9.0, 9.0]])},
        floor_rows=floor_rows,
    )
    margin = np.array([[3.0, 5.0], [4.0, 6.0]]) - floor_rows
    np.testing.assert_allclose(
        swept.floor_margin_standard_error(1),
        margin.std(axis=0, ddof=1) / np.sqrt(2),
    )
    # The floor genuinely varies window to window, so subtracting its MEAN --
    # the only other array of the right shape in reach -- is a different
    # number rather than the same one.
    against_mean = np.array([[3.0, 5.0], [4.0, 6.0]]) - floor_rows.mean(axis=0)
    assert not np.allclose(
        margin.std(axis=0, ddof=1), against_mean.std(axis=0, ddof=1)
    ), "the fixture's floor does not vary across windows; the two are not separated"


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_the_sweeps_per_window_floor_rows_are_the_rollouts_own_floor(tmp_path, device):
    """The retained floor rows must average to the floor the record reports, or
    the paired margin is measured against a different quantity from the one the
    table prints."""
    paths = [write(tmp_path, synthetic_episode(), index) for index in range(2)]
    model, probe = real_model_and_probe()
    model = model.to(device)
    result = sweep(model, paths, probe, ks=(1, HORIZON), device=device)
    assert result.window_floor_position.shape == (result.windows_total, HORIZON)
    np.testing.assert_array_equal(
        result.window_floor_position.mean(axis=0), result.reference.floor_position
    )
    # And they are the REAL rows, not the mean tiled to the right shape --
    # which satisfies the equality above exactly and destroys every paired
    # ruler built on them, since the pairing term would then be constant.
    assert result.window_floor_position.std(axis=0).max() > 0.0, (
        "every window's floor row is identical; these are a tiled mean, not "
        "per-window measurements"
    )


# ---------------------------------------------------------------------------
# The shuffle's angle delta, and the episode each window came from.
# ---------------------------------------------------------------------------


def test_the_angle_delta_is_built_from_the_angle_curves_and_not_the_position_ones(
    tmp_path, monkeypatch
):
    """`window_angle_delta` is public and exported, and nothing read what was
    in it -- building it from the POSITION arrays survived the suite. A reader
    inspecting a `ShuffleResult` would get map units labelled as degrees.

    Asserted as a closed form on `ActionSumModel`, the same rig and the same
    captured action tensors that pin the position half, so it cannot be
    satisfied by a diagnostic that permuted something else.
    """
    path = write(tmp_path, varied_action_episode())
    episode = varied_action_episode()
    model = ActionSumModel()
    seen: list[np.ndarray] = []
    real = model.rssm.imagine
    monkeypatch.setattr(
        model.rssm,
        "imagine",
        lambda actions, state: seen.append(actions[0].cpu().numpy().copy())
        or real(actions, state),
    )
    result = shuffle(model, [path], oracle_probe(episode))

    assert result.windows_changed == result.windows_total > 0
    np.testing.assert_allclose(
        result.angle_delta(),
        result.shuffled_angle - result.real.rssm_angle,
        rtol=1e-6, atol=1e-9,
    )
    assert not np.allclose(
        result.angle_delta(), result.position_delta(), atol=1e-9
    ), "the angle delta is numerically the position delta; the fixture separates them"


def test_the_shuffle_records_which_episode_each_window_was_cut_from(tmp_path):
    """The 229 shipped windows come from 24 episodes, not from 229 independent
    draws, and a standard error that divides by sqrt(229) claims a precision
    the data does not support.

    The label is per WINDOW and in traversal order, so a consumer can cluster
    by it. Two episodes of different lengths, so a label that counted windows
    rather than episodes -- or reset per episode -- is caught by the counts.
    """
    episodes = [synthetic_episode(length=T_SYNTHETIC), synthetic_episode(length=ONE_WINDOW)]
    paths = [write(tmp_path, episode, index) for index, episode in enumerate(episodes)]
    expected = [len(window_starts(e.length, CONTEXT, HORIZON)) for e in episodes]
    assert expected == [2, 1], expected
    result = shuffle(OracleModel(), paths, oracle_probe(episodes[0]))
    np.testing.assert_array_equal(result.window_episode, np.array([0, 0, 1]))


def test_an_episode_that_yields_no_window_does_not_consume_an_episode_label(tmp_path):
    """A too-short episode is skipped entirely, so it must not shift the labels
    of the episodes after it -- a label taken from `enumerate(val_paths)`
    rather than from the episodes that actually contributed windows would leave
    a gap, and a consumer counting distinct labels would report the wrong
    number of clusters."""
    lengths = [T_SYNTHETIC, CONTEXT + HORIZON, T_SYNTHETIC]
    episodes = [synthetic_episode(length=n) for n in lengths]
    paths = [write(tmp_path, episode, index) for index, episode in enumerate(episodes)]
    result = shuffle(OracleModel(), paths, oracle_probe(episodes[0]))
    assert result.windows_total == 4
    assert sorted(set(result.window_episode.tolist())) == [0, 1], (
        "the skipped middle episode still consumed a label"
    )
    np.testing.assert_array_equal(result.window_episode, np.array([0, 0, 1, 1]))


# ---------------------------------------------------------------------------
# The two self-checks on the SHIPPED checkpoints. The strongest correctness
# evidence available: bitwise equality, on a stochastic model, against numbers
# computed months before this module existed.
# ---------------------------------------------------------------------------

SHIPPED_RUNS = Path("runs/m3_study")
SHIPPED_DATA = Path("data/my_way_home")
SHIPPED_ARM, SHIPPED_SEED = "cnn", 0
SHIPPED_WINDOWS = 229
"""22 val episodes of 525 transitions give 10 windows each, one of 345 gives 6,
one of 154 gives 3. Measured, and derived independently in the test below."""

shipped = pytest.mark.skipif(
    not (SHIPPED_RUNS.exists() and SHIPPED_DATA.exists()
         and torch.backends.mps.is_available()),
    reason=(
        "needs the shipped study artifacts and MPS. Measured, the records "
        "reproduce BITWISE on mps under torch 2.13.0 and miss on cpu, with an "
        "identical probe and split, by an ARM-DEPENDENT amount -- ~6.5 map units "
        "(2.96% relative) on cnn/seed0 and ~12.4 on frozen_ssl/seed0 -- so this "
        "is a statement about this box rather than a portable one, and the "
        "figure is a range rather than one number"
    ),
)


@pytest.fixture(scope="module")
def shipped_cell():
    """`(model, val, probe, record, device, backbone, reference)` for one cell.

    Module-scoped because the probe refit and the open-loop rollout cost ~45 s
    together and both slow tests below need them. The record does NOT store the
    probe weights, so reproduction requires the refit -- and the selected ridge
    is asserted against the record's first, as the cheap precondition that fails
    loudly on environment drift before any curve is compared. (It is a proxy,
    not a proof: two different weight vectors can select the same ridge.)
    """
    from mbfps.data.buffer import ReplayBuffer
    from mbfps.data.split import VAL_FRACTION, episode_split
    from mbfps.eval.probe import fit_probes
    from mbfps.eval.study import SPLIT_SEED, load_record
    from mbfps.models.encoders import encoder_backbone
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    device = torch.device("mps")
    cfg = get_config(SHIPPED_ARM, seed=SHIPPED_SEED, device="mps")
    model = WorldModel(cfg).to(device)
    checkpoint = torch.load(
        SHIPPED_RUNS / f"world_model_{SHIPPED_ARM}_seed{SHIPPED_SEED}.pt",
        map_location=device, weights_only=True,
    )
    assert checkpoint["arm"] == SHIPPED_ARM and checkpoint["seed"] == SHIPPED_SEED
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    buffer = ReplayBuffer(SHIPPED_DATA, capacity_transitions=10**9)
    train, val = episode_split(
        buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED
    )
    record = load_record(
        SHIPPED_RUNS / f"result_{SHIPPED_ARM}_seed{SHIPPED_SEED}.json"
    )
    assert [p.name for p in val] == record["episodes"]["val"], (
        "the split is not the one the record was scored on, so nothing below is "
        "comparable to it"
    )
    backbone = encoder_backbone(cfg.encoder)
    _, probe = fit_probes(
        model, train, backbone, device, context=5, horizon=45, seed=SHIPPED_SEED
    )
    assert probe["ridge"] == record["probe"]["embedding_ridge"]
    reference = evaluate_rollout(
        model, val, probe, context=5, horizon=45, seed=SHIPPED_SEED,
        device=device, feature_backbone=backbone,
    )
    np.testing.assert_array_equal(
        reference.rssm_position, np.asarray(record["curves"]["rssm_position"])
    )
    return model, val, probe, record, device, backbone, reference


@pytest.mark.slow
@shipped
def test_the_sweep_reproduces_the_shipped_record_at_k_equals_the_horizon(shipped_cell):
    """SELF-CHECK 1 on the artifact. Measured: max abs difference 0.0, for the
    model curve AND the floor, on cnn/seed 0.

    SELF-CHECK 2's real-data half is here too, but in the only form the shipped
    checkpoints support. Measured on this cell: k=1's horizon mean is 295.572
    against the floor's 295.528, so it does sit above -- but it dips BELOW the
    floor at some individual steps, by at most 0.297 map units against a
    per-window standard error of ~9.1. That is noise, not an off-by-one, and it
    is unsurprising here: the embedding probe's own selection R^2 on this cell
    is -0.0357, so the "floor" is not an informative lower bound on this
    checkpoint at all -- the study already reports the persistence-to-floor band
    inverting. A strict per-step inequality would therefore be an assertion
    about noise, so what is asserted is what noise CANNOT satisfy: k=1 is not
    the floor tensor, and re-grounding demonstrably changes the curve.

    ks is (1, 45) rather than the full REGROUNDING_KS to keep the default
    suite's runtime honest; the intermediate periods are what the CLI reports.
    """
    model, val, probe, record, device, backbone, reference = shipped_cell
    result = regrounding_sweep(
        model, val, probe, ks=(1, 45), context=5, horizon=45, seed=SHIPPED_SEED,
        device=device, feature_backbone=backbone,
    )
    np.testing.assert_array_equal(
        result.curve(45), np.asarray(record["curves"]["rssm_position"])
    )
    np.testing.assert_array_equal(
        result.reference.floor_position,
        np.asarray(record["curves"]["floor_position"]),
    )
    assert result.open_loop_divergence(reference) == 0.0
    assert result.windows_total == SHIPPED_WINDOWS

    assert result.is_bitwise_the_floor(1, "position") is False
    assert result.is_bitwise_the_floor(1, "angle") is False
    assert not np.array_equal(result.curve(1), result.curve(45)), (
        "re-grounding every step produced the open-loop curve; k is being ignored"
    )
    # Step 1 is one prior step from the context state for BOTH k, so the only
    # thing that could separate them there is the sampling stream.
    assert result.curve(1)[0] == result.curve(45)[0]


@pytest.mark.slow
@shipped
def test_the_shuffle_on_a_shipped_checkpoint_reports_both_window_counts(shipped_cell):
    """The shuffle's baseline must BE the shipped protocol, and its window
    counts are what make the delta readable.

    Measured on cnn/seed 0: 229 of 229 windows really change, so the
    `windows_changed` guard is inert against the real checkpoints and can only
    be exercised synthetically -- which is exactly why it is a reported output
    rather than a debug aid. The count is checked against the window rule
    applied to the record's own val episodes, not against a number typed here.
    """
    from mbfps.data.episode import load_episode

    model, val, probe, record, device, backbone, reference = shipped_cell
    result = action_shuffled_rollout(
        model, val, probe, context=5, horizon=45, seed=SHIPPED_SEED,
        device=device, feature_backbone=backbone, permutation_seed=0,
    )
    np.testing.assert_array_equal(
        result.real.rssm_position, np.asarray(record["curves"]["rssm_position"])
    )
    np.testing.assert_array_equal(result.real.floor_position, reference.floor_position)
    np.testing.assert_array_equal(
        result.real.persistence_position, reference.persistence_position
    )
    expected = sum(
        len(window_starts(load_episode(path).length, 5, 45)) for path in val
    )
    assert expected == SHIPPED_WINDOWS
    assert result.windows_total == expected
    assert result.windows_changed == expected
    assert result.is_interpretable() is True
