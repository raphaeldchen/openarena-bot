"""The two M3b diagnostics: the action-intervention ladder, and the k-step sweep.

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
    LADDER,
    LADDER_PERTURBS,
    REGROUNDING_KS,
    RegroundingSweep,
    action_intervention_ladder,
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
    # canonical pass -- plus this k's own segments, and NOTHING else: the
    # sweep reads no embedding-space ratio, so it draws no noise reference,
    # and a third full-horizon call here is a 45-step `imagine` per window
    # that nobody reads. Compared as an unordered structure so a legitimate
    # reordering of the arms does not fail it.
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

    # Three `imagine` calls per window, in call order: the permuted arm, the
    # canonical pass, then the NOISE REFERENCE -- a second imagination over the
    # real actions, asserted here to be handed the canonical call's tensor
    # bitwise, not merely skipped over.
    assert len(seen) == 3 * 2, len(seen)
    permuted, canonical, noise = seen[0::3], seen[1::3], seen[2::3]
    for index, (a, b) in enumerate(zip(canonical, noise)):
        np.testing.assert_array_equal(a, b, err_msg=f"window {index}")
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
    # Permuted arm, canonical pass, noise reference: three calls per window.
    assert len(seen) == 3 * len(starts), len(seen)
    permuted = seen[0::3]
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

    # The same exclusion through the LADDER's own result type, which is what
    # the CLI reads for every rung's `position_delta` curve, final step and
    # per-step exceedance count. A diluting mean there would drag those toward
    # the null while the aggregate `delta_summary` computes separately would
    # not, and the record would carry two inconsistent statistics.
    ladder_both = ladder(model, paths, probe, arms=("shuffled",)).arms["shuffled"]
    assert ladder_both.windows_total == 2 and ladder_both.windows_changed == 1
    np.testing.assert_allclose(
        ladder_both.position_delta(), alone.position_delta(), rtol=1e-6
    )
    np.testing.assert_allclose(ladder_both.angle_delta(), alone.angle_delta(), rtol=1e-6)


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
# Diagnostic 1b: the intervention LADDER (shuffled -> resampled -> constant).
#
# A permutation preserves the action MULTISET, so the shuffled rung tests ORDER
# sensitivity and nothing else: dynamics that read the COUNTS of each action
# rather than their sequence are action-conditioned and still bit-identical
# under it. The two rungs added here close that gap, and the tests below are
# arranged around the three ways a rung can look like a null result without
# being one -- it was a no-op, it never reached `imagine`, or it moved the
# sampling stream the arms are matched on.
# ---------------------------------------------------------------------------


def ladder(model, paths, probe, *, device=None, arms=LADDER, **kwargs):
    return action_intervention_ladder(
        model, paths, probe, arms=arms, context=CONTEXT, horizon=HORIZON,
        device=device or torch.device("cpu"), **kwargs,
    )


def action_skewed_episode(length=T_SYNTHETIC):
    """Horizon actions dominated by ONE value, with a unique rarest one.

    Three separable actions, so a test can tell which distribution a rung
    read and how far each held action travels:

      * the SCORED WINDOWS' modal action is 1 (7 of the 10 horizon steps);
      * the SCORED WINDOWS' rarest action is 4 (1 of 10);
      * the WHOLE EPISODE's rarest action is 0 (1 of 20), and it never appears
        in a scored horizon at all -- the rest of the episode is filled with 4,
        which is the whole episode's MOST common action.

    That last pair is what makes "the marginal is taken over the scored
    windows" a claim with a failing case rather than a restatement: a marginal
    counted over `episode.actions` has 0 in its support, one counted over the
    horizon slices does not, and the two rank rarity in opposite directions.
    The scored support is {1, 2, 4}, so the default MOVE_FORWARD-minus-NOOP
    contrast is outside it and every constant-rung test here holds
    `SKEWED_CONTRAST` instead.

    The domination is the point of the fixture. Holding the window's own most
    frequent action leaves 4 of window 0's 5 steps untouched -- a near-no-op
    wearing the name of the ladder's MAXIMAL perturbation, which is the L1
    fixture coincidence in a new costume -- and the held-action accounting is
    what has to make that visible.
    """
    episode = synthetic_episode(length=length)
    windows = window_starts(length, CONTEXT, HORIZON)
    assert len(windows) == 2, "fixture assumes exactly two windows"
    actions = np.full(length, 4, dtype=np.int32)
    actions[0] = 0
    for start, values in zip(windows, ([1, 1, 1, 1, 2], [1, 1, 1, 2, 4])):
        actions[start + CONTEXT : start + CONTEXT + HORIZON] = values
    episode.actions = actions
    return episode


def _imagined_actions(monkeypatch, model):
    """Every action tensor `imagine` was handed, in call order."""
    seen: list[np.ndarray] = []
    real = model.rssm.imagine
    monkeypatch.setattr(
        model.rssm,
        "imagine",
        lambda actions, state: seen.append(actions[0].cpu().numpy().copy())
        or real(actions, state),
    )
    return seen


def _per_rung(seen, arms, windows):
    """Split `imagine`'s call log into one list per intervention arm.

    `arms` is every arm in CALL order -- a rung name, or `("constant", a)` for
    the held action `a`, since the constant rung holds every action in the
    support and each is its own call. Each window makes one call per arm, then
    the canonical pass, then the NOISE REFERENCE -- a second imagination over
    the real actions -- so the stride is `len(arms) + 2`, the canonical calls
    sit at `len(arms)` and the noise calls at `len(arms) + 1`. The noise call
    is asserted HERE to carry the canonical call's actions bitwise, so no
    caller can quietly slice it away: a noise reference handed an intervened
    sequence would be measuring the intervention twice. An arm whose actions
    never reached `imagine` shows up as the wrong tensor in its own slot rather
    than as a missing call.
    """
    arms = list(arms)
    stride = len(arms) + 2
    assert len(seen) == stride * windows, (len(seen), stride, windows)
    canonical = seen[len(arms) :: stride]
    noise = seen[len(arms) + 1 :: stride]
    for index, (a, b) in enumerate(zip(canonical, noise)):
        np.testing.assert_array_equal(
            a, b, err_msg=f"window {index}: the noise reference was not handed the real actions"
        )
    return {name: seen[index::stride] for index, name in enumerate(arms)}, canonical


def _ladder_arms(result):
    """The arms of a `LadderResult` in the order `imagine` was called with them."""
    arms = []
    for name in result.order:
        if name == "constant":
            arms.extend(("constant", action) for action in result.held_actions)
        else:
            arms.append(name)
    return arms


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_every_rung_is_bit_identical_on_a_model_whose_dynamics_ignore_actions(
    tmp_path, device
):
    """The two-sided model test, extended to the whole ladder.

    On a real sampling RSSM whose `_step` provably discards the action, EVERY
    rung must be bitwise the real arm -- not merely the permutation. A rung
    that is only close is a rung whose sampling stream moved, which is the
    artefact the per-window snapshot exists to prevent and which no shape or
    window count would reveal.

    `windows_changed == windows_total > 0` is asserted per rung IN THE SAME
    TEST, so the equality can never be explained by "the intervention did
    nothing".
    """
    path = write(tmp_path, varied_action_episode())
    model, probe = action_blind_model()
    model = model.to(device)
    result = ladder(model, [path], probe, device=device)

    assert result.order == LADDER
    for name in ("shuffled", "resampled"):
        arm = result.arms[name]
        np.testing.assert_array_equal(arm.position, result.real.rssm_position, name)
        np.testing.assert_array_equal(arm.angle, result.real.rssm_angle, name)
        assert arm.windows_changed == arm.windows_total > 0, name
    # The constant rung: EVERY held action is bitwise the real arm, so the
    # contrast between any two of them is exactly zero in every window --
    # which is what makes the contrast's null exact rather than approximate.
    rung = result.arms["constant"]
    assert len(rung.held) >= 2
    for action, held in rung.held.items():
        np.testing.assert_array_equal(held.position, result.real.rssm_position, action)
        np.testing.assert_array_equal(held.angle, result.real.rssm_angle, action)
        assert held.windows_changed == held.windows_total > 0, action
    np.testing.assert_array_equal(rung.window_position_delta, 0.0)
    np.testing.assert_array_equal(rung.window_angle_delta, 0.0)
    assert rung.windows_changed == rung.windows_total > 0

    # THE EMBEDDING-SPACE HALF, in the same body. Intervened and real share the
    # per-window snapshot, so on an action-blind model the numerator is bitwise
    # 0.0 in every window and the ratio exactly 0.0 -- not approx, not NaN. And
    # the zero is only evidence beside a NOISE REFERENCE that is NOT zero: the
    # second imagination is drawn from a different stream point, so it differs
    # from the canonical one in every window even here. A numerator that never
    # ran, or a noise reference replayed from the canonical snapshot, both give
    # 0 / 0 and are excluded by the `> 0` half.
    noise = result.noise_reference
    assert noise.windows_collapsed == 0
    assert noise.stream_restored is True
    assert (noise.window_embedding_distance > 0.0).all()
    assert (noise.embedding_distance_curve > 0.0).all()
    readings = [result.arms["shuffled"], result.arms["resampled"], rung, *rung.held.values()]
    for arm in readings:
        np.testing.assert_array_equal(arm.window_embedding_distance, 0.0, err_msg=arm.name)
        np.testing.assert_array_equal(arm.embedding_distance_curve, 0.0, err_msg=arm.name)
        assert arm.noise_median() > 0.0, arm.name
        assert arm.embedding_median() == 0.0, arm.name
        assert arm.embedding_ratio() == 0.0, arm.name


def test_every_rung_moves_under_a_model_whose_dynamics_use_the_action(
    tmp_path, monkeypatch
):
    """The positive half, as exact closed-form values per rung.

    `ActionSumModel` advances the frame tag by the action's own value, so each
    arm's curve is a closed form of the action tensor THAT ARM handed
    `imagine` -- captured by the spy. An arm that computed a sequence and
    imagined a different one, or that shared a buffer with its neighbour, lands
    on the wrong curve rather than merely "differing".
    """
    episode = action_skewed_episode()
    path = write(tmp_path, episode)
    model = ActionSumModel()
    seen = _imagined_actions(monkeypatch, model)
    result = ladder(model, [path], oracle_probe(episode), contrast=SKEWED_CONTRAST)

    arms, canonical = _per_rung(seen, _ladder_arms(result), windows=2)
    steps = np.arange(1, HORIZON + 1)
    expected = lambda arms: np.mean(  # noqa: E731
        [STEP * np.abs(np.cumsum(a) - steps) for a in arms], axis=0
    )
    np.testing.assert_allclose(
        result.real.rssm_position, expected(canonical), rtol=1e-5, atol=1e-6
    )
    curves = []
    for name in ("shuffled", "resampled"):
        np.testing.assert_allclose(
            result.arms[name].position, expected(arms[name]), rtol=1e-5, atol=1e-6,
            err_msg=name,
        )
        assert not np.array_equal(
            result.arms[name].position, result.real.rssm_position
        ), name
        curves.append(tuple(result.arms[name].position))
    for action, held in result.arms["constant"].held.items():
        np.testing.assert_allclose(
            held.position, expected(arms[("constant", action)]), rtol=1e-5, atol=1e-6,
            err_msg=str(action),
        )
        curves.append(tuple(held.position))
    assert np.abs(result.arms["constant"].position_delta()).max() > 0.0
    # And every arm is a DIFFERENT intervention, not one computed under
    # several names.
    assert len(set(curves)) == len(curves), curves


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_adding_rungs_leaves_the_shuffled_rung_bitwise_where_it_was(tmp_path, device):
    """THE regression guard, at unit scale: the new rungs must not move the old.

    Every rung replays from ONE per-window RNG snapshot, so the arms are
    order-independent BY CONSTRUCTION -- but only as long as no rung draws from
    a stream another rung depends on. The permutation comes from a numpy
    generator seeded at `intervention_seed`; a resampled rung that drew from
    that SAME generator would leave every window after the first permuted
    differently, and nothing about the shapes, the window counts or the
    self-checks would say so.

    A REAL sampling model, because on an oracle rig no stream exists to move
    and on an action-blind one every rung is identical anyway.
    """
    path = write(tmp_path, varied_action_episode())
    model, probe = real_model_and_probe()
    model = model.to(device)

    alone = ladder(model, [path], probe, arms=("shuffled",), device=device)
    whole = ladder(model, [path], probe, device=device)

    np.testing.assert_array_equal(
        whole.arms["shuffled"].position, alone.arms["shuffled"].position
    )
    np.testing.assert_array_equal(
        whole.arms["shuffled"].angle, alone.arms["shuffled"].angle
    )
    np.testing.assert_array_equal(
        whole.arms["shuffled"].window_position_delta,
        alone.arms["shuffled"].window_position_delta,
    )
    # And the shared brackets too: the reference, the floor and persistence are
    # the record's own numbers, and a moved stream shifts them as well.
    for curve in ("rssm_position", "floor_position", "persistence_position"):
        np.testing.assert_array_equal(
            getattr(whole.real, curve), getattr(alone.real, curve), err_msg=curve
        )
    # The embedding-space reading and the noise reference are per-window
    # functions of (seed, window) alone: the noise reference is drawn from the
    # stream point the canonical pass leaves, never from wherever the last arm
    # happened to stop, so it cannot move with the rungs selected.
    np.testing.assert_array_equal(
        whole.arms["shuffled"].window_embedding_distance,
        alone.arms["shuffled"].window_embedding_distance,
    )
    np.testing.assert_array_equal(
        whole.noise_reference.window_embedding_distance,
        alone.noise_reference.window_embedding_distance,
    )
    assert alone.noise_reference.window_embedding_distance.min() > 0.0
    assert whole.arms["shuffled"].windows_changed == whole.windows_total > 0


def test_the_shuffled_rung_permutes_with_the_literal_stream_that_produced_the_shipped_records(
    tmp_path, monkeypatch
):
    """The shipped permutations are `np.random.default_rng(seed).permutation(H)`,
    drawn ONCE PER WINDOW in traversal order from ONE generator, and this is
    the portable pin on that contract.

    The standalone-versus-ladder equality cannot carry it: `action_shuffled_
    rollout` delegates to the ladder, so the two agree whatever stream either
    uses, and the only other guard runs on a shipped checkpoint under MPS. A
    ladder that re-derived the generator -- `default_rng([seed, 7])`, a
    SeedSequence child, a per-window reseed -- would re-permute all nine
    records while every shape, count and self-check stayed right.

    The permutation INDEX is recovered from what `imagine` was handed, on the
    fixture whose windows carry distinct actions for exactly this purpose, and
    compared draw for draw. `default_rng([0, 0])` equals `default_rng(0)`, so
    the seed is 3 rather than the default: at seed 0 a trailing-zero
    re-derivation would be invisible.
    """
    episode = varied_action_episode()
    path = write(tmp_path, episode)
    model = ActionSumModel()
    seen = _imagined_actions(monkeypatch, model)
    ladder(model, [path], oracle_probe(episode), arms=("shuffled",), intervention_seed=3)

    starts = window_starts(episode.length, CONTEXT, HORIZON)
    assert len(starts) >= 2, "fixture must span more than one window"
    rungs, _ = _per_rung(seen, ("shuffled",), windows=len(starts))
    stream = np.random.default_rng(3)
    for index, start in enumerate(starts):
        window = episode.actions[start + CONTEXT : start + CONTEXT + HORIZON]
        lookup = {int(value): position for position, value in enumerate(window)}
        recovered = [lookup[int(value)] for value in rungs["shuffled"][index]]
        np.testing.assert_array_equal(
            recovered, stream.permutation(HORIZON), err_msg=f"window {index}"
        )


def test_the_resampled_rung_draws_i_i_d_from_the_scored_windows_action_marginal(
    tmp_path, monkeypatch
):
    """The rung's whole claim: a DIFFERENT multiset from the same marginal.

    Three things, and each has its own way of being wrong. The marginal is
    counted over the horizon actions of the SCORED WINDOWS -- the fixture's
    whole-episode counts disagree with it in both directions. Every drawn
    action lies in that support, so each individual action stays plausible.
    And at least one window's drawn MULTISET differs from the real one, or the
    rung has collapsed into the permutation it exists to go beyond.
    """
    episode = action_skewed_episode()
    path = write(tmp_path, episode)
    model = ActionSumModel()
    seen = _imagined_actions(monkeypatch, model)
    result = ladder(model, [path], oracle_probe(episode), arms=("resampled",))

    horizon_actions = np.concatenate([
        episode.actions[start + CONTEXT : start + CONTEXT + HORIZON]
        for start in window_starts(episode.length, CONTEXT, HORIZON)
    ])
    values, counts = np.unique(horizon_actions, return_counts=True)
    np.testing.assert_array_equal(result.action_marginal_values, values)
    np.testing.assert_array_equal(result.action_marginal_counts, counts)
    # Counted over the whole episode instead, the support would include 0 and
    # be dominated by 4 -- a different distribution, and the test says so.
    assert 0 in set(episode.actions.tolist()) and 0 not in set(values.tolist())

    rungs, _ = _per_rung(seen, ("resampled",), windows=2)
    drawn = rungs["resampled"]
    for index, sequence in enumerate(drawn):
        assert set(sequence.tolist()) <= set(values.tolist()), (index, sequence)
    real_multisets = [
        sorted(episode.actions[start + CONTEXT : start + CONTEXT + HORIZON].tolist())
        for start in window_starts(episode.length, CONTEXT, HORIZON)
    ]
    assert any(
        sorted(sequence.tolist()) != real
        for sequence, real in zip(drawn, real_multisets)
    ), "every resampled window was a permutation of the real one"
    # And the draws are i.i.d. ACROSS windows, not one sequence dealt to every
    # window: a per-window reseed, or a generator that drew once and handed
    # the same array out again, would leave the shipped 229 windows reading
    # 229/229 changed and a plausible null on a single draw.
    assert len({tuple(sequence.tolist()) for sequence in drawn}) > 1, (
        "every window was handed the identical resampled sequence"
    )


def test_the_resampled_rung_draws_with_the_marginals_own_probabilities(
    tmp_path, monkeypatch
):
    """"Same marginal, different multiset" is the rung's whole claim over the
    permutation, and it is a claim about the PROBABILITIES handed to the draw.

    Support membership and a differing multiset are both satisfied by a draw
    that is uniform over the support -- which on the 7:2:1 fixture would hand
    the rarest action seven times its real frequency and the `action_marginal`
    written to the record would be a distribution nothing drew from. So the
    generator is stubbed to capture `p` and it is pinned to `counts / sum`.
    """
    episode = action_skewed_episode()
    path = write(tmp_path, episode)
    captured: list[tuple[np.ndarray, np.ndarray]] = []

    class _CapturingRng:
        def choice(self, values, size, p):
            captured.append((np.asarray(values).copy(), np.asarray(p).copy()))
            return np.asarray(values)[np.argmax(p)].repeat(size).astype(np.int64)

    monkeypatch.setattr(diagnostics_module, "default_rng", lambda seed: _CapturingRng())
    result = ladder(
        ActionSumModel(), [path], oracle_probe(episode), arms=("resampled",),
        contrast=(4, 1),
    )

    assert len(captured) == result.windows_total == 2
    for values, p in captured:
        np.testing.assert_array_equal(values, [1, 2, 4])
        np.testing.assert_allclose(p, [0.7, 0.2, 0.1])
    assert not np.allclose(captured[0][1], np.full(3, 1.0 / 3.0)), (
        "the fixture's marginal is uniform; it cannot separate the two draws"
    )


# The fixture's scored support is {1, 2, 4}, so the default MOVE_FORWARD-minus-
# NOOP contrast is outside it. Every constant-rung test on that fixture holds
# 4 against 1: 4 is the rarest scored action and 1 the modal one, so the two
# held sequences differ from the real windows by very different amounts.
SKEWED_CONTRAST = (4, 1)


def test_the_constant_rung_holds_every_action_in_the_support_and_decides_on_the_contrast(
    tmp_path, monkeypatch
):
    """The top rung is a CONTRAST between two held actions, not one held action
    against the real sequence.

    Measured on the shipped checkpoints, holding STRAFE_RIGHT -- the rarest
    scored action -- for the whole horizon reads as a null while holding
    MOVE_FORWARD through the identical machinery moves the error by +31 map
    units (frozen_ssl/0): a single held action asks "does the prior respond"
    with whichever action it happens to hold. Every action in the scored
    support is held, from the same per-window snapshot, and the decision is
    the paired difference between two NAMED held actions, so the sequence-level
    off-distribution confound -- 45 identical steps, where the data's longest
    run is 13 -- is the same on both sides and cancels.

    On `ActionSumModel` each held arm's curve is a closed form of the tensor
    THAT arm handed `imagine`, and the contrast's per-window delta is the
    difference of the two named arms' per-window errors -- asserted by value,
    so a contrast built from the wrong pair, or from one arm twice, lands on
    the wrong numbers rather than merely "differing".
    """
    episode = action_skewed_episode()
    path = write(tmp_path, episode)
    model = ActionSumModel()
    seen = _imagined_actions(monkeypatch, model)
    result = ladder(
        model, [path], oracle_probe(episode), arms=("constant",), contrast=SKEWED_CONTRAST
    )

    rung = result.arms["constant"]
    assert result.held_actions == (1, 2, 4)
    assert result.contrast == rung.contrast == SKEWED_CONTRAST
    assert set(rung.held) == {1, 2, 4}
    # Every held action reached `imagine` as a full-horizon constant, once per
    # window, in support order, before the canonical pass.
    arms, canonical = _per_rung(seen, [("constant", a) for a in (1, 2, 4)], windows=2)
    for action in (1, 2, 4):
        for index, sequence in enumerate(arms[("constant", action)]):
            assert sequence.tolist() == [action] * HORIZON, (action, index, sequence)
    steps = np.arange(1, HORIZON + 1)
    expected = lambda arms: np.mean(  # noqa: E731
        [STEP * np.abs(np.cumsum(a) - steps) for a in arms], axis=0
    )
    for action in (1, 2, 4):
        np.testing.assert_allclose(
            rung.held[action].position, expected(arms[("constant", action)]),
            rtol=1e-5, atol=1e-6, err_msg=str(action),
        )
    np.testing.assert_allclose(result.real.rssm_position, expected(canonical), rtol=1e-5, atol=1e-6)
    # The contrast is held-4 minus held-1, per window, on both channels.
    np.testing.assert_allclose(
        rung.window_position_delta,
        rung.held[4].window_position_delta - rung.held[1].window_position_delta,
    )
    np.testing.assert_allclose(
        rung.window_angle_delta,
        rung.held[4].window_angle_delta - rung.held[1].window_angle_delta,
    )
    assert np.abs(rung.window_position_delta).max() > 0.0
    assert not np.allclose(
        rung.window_position_delta, rung.held[4].window_position_delta
    ), "the contrast is one held arm against the real sequence, not against the other arm"


def test_the_constant_rung_reports_how_far_each_held_action_is_from_the_real_windows(
    tmp_path,
):
    """A held action is a no-op exactly where the window already held it, and
    on the shipped data nothing would ever reveal that it had been. So every
    held arm reports its per-window STEP distance from the real sequence, the
    minimum over windows beside the mean, and its multiset distance -- and the
    numbers here are the fixture's own, computed independently.

    Window 0 is `[1,1,1,1,2]` and window 1 is `[1,1,1,2,4]`: holding 4 changes
    5 and 4 steps, holding the modal 1 changes 1 and 2, holding 2 changes 4 and
    4. The CONTRAST's own count is the number of steps at which the two held
    sequences differ from EACH OTHER -- every step, since 4 is not 1 -- because
    that, not the distance from the real window, is what would make the
    comparison a no-op.
    """
    episode = action_skewed_episode()
    path = write(tmp_path, episode)
    rung = ladder(
        ActionSumModel(), [path], oracle_probe(episode), arms=("constant",),
        contrast=SKEWED_CONTRAST,
    ).arms["constant"]

    np.testing.assert_array_equal(rung.held[4].window_steps_changed, [5, 4])
    np.testing.assert_array_equal(rung.held[1].window_steps_changed, [1, 2])
    np.testing.assert_array_equal(rung.held[2].window_steps_changed, [4, 4])
    assert rung.held[4].mean_steps_changed == pytest.approx(4.5)
    # `min_steps_changed` is NOT pinned here: on this fixture every held arm's
    # windows differ by one step, so `int(mean)` equals the minimum and the
    # assertion would be satisfied by the wrong statistic. It is pinned in
    # `test_each_rung_carries_its_own_counts_and_its_own_delta`, on `[0, 4]`.
    # Multiset distance: how many steps must change to turn the real multiset
    # into the held one. Holding 1 on [1,1,1,1,2] is one step; on [1,1,1,2,4]
    # two.
    np.testing.assert_array_equal(rung.held[1].window_multiset_distance, [1, 2])
    np.testing.assert_array_equal(rung.held[4].window_multiset_distance, [5, 4])
    np.testing.assert_array_equal(rung.window_steps_changed, [HORIZON, HORIZON])
    assert rung.windows_changed == rung.windows_total == 2
    assert rung.is_interpretable() is True


@pytest.mark.parametrize(
    "contrast,match",
    [((4, 4), "distinct"), ((3, 1), "3"), ((4, 0), "0")],
)
def test_a_contrast_that_cannot_be_held_is_refused_rather_than_run(tmp_path, contrast, match):
    """Two held actions that are the same action compare a sequence with
    itself; a held action outside the SCORED support is one the model never
    saw in a horizon, so a null under it says nothing about the dynamics. Both
    are refused BEFORE the traversal, by name, rather than run to a
    meaningless number. The fixture's whole-episode actions include 0, so the
    third case also pins that the support is the scored windows' and not the
    episode's."""
    episode = action_skewed_episode()
    path = write(tmp_path, episode)
    assert 0 in set(episode.actions.tolist())
    with pytest.raises(ValueError, match=match):
        ladder(ActionSumModel(), [path], oracle_probe(episode), contrast=contrast)


def test_the_default_contrast_is_move_forward_minus_noop_and_names_both():
    """The contrast is pre-registered, not chosen from the data: the
    displacement-carrying action against the stationary one, in the
    my_way_home button order the shipped episodes were collected under
    (`build_action_set`'s no-op, then TURN_LEFT, TURN_RIGHT, MOVE_FORWARD,
    MOVE_LEFT, MOVE_RIGHT). Its sign is therefore a physical prediction -- an
    action-conditioned prior must run the imagined position further under
    MOVE_FORWARD than under NOOP -- which is what makes a response at this
    rung readable as conditioning rather than as confusion."""
    from mbfps.eval.diagnostics import ACTION_NAMES, CONTRAST, action_name

    assert CONTRAST == (3, 0)
    assert ACTION_NAMES[CONTRAST[0]] == "MOVE_FORWARD"
    assert ACTION_NAMES[CONTRAST[1]] == "NOOP"
    assert action_name(5) == "MOVE_RIGHT"
    assert action_name(6) == "action 6", "an index outside the set is named as an index"
    assert inspect.signature(action_intervention_ladder).parameters["contrast"].default is CONTRAST


def test_a_rung_that_is_a_no_op_on_every_window_is_uninterpretable_not_null(tmp_path):
    """The accounting that makes a no-op impossible to mistake for a null.

    Every horizon action is the same value, so the permutation rearranges
    nothing and the marginal is a point mass, so every resample reproduces the
    real sequence. Both rungs are genuine no-ops, and each must report it in
    its own counts and hand back an UNDEFINED delta rather than the exact zero
    that reads as "the model ignored the action". The constant rung cannot run
    here at all -- a point-mass support has no two actions to contrast -- and
    is refused by name in the same test, so the ladder on a degenerate split
    fails loudly rather than reporting a contrast of one action with itself.
    """
    episode = uniform_action_episode()
    path = write(tmp_path, episode)
    result = ladder(
        ActionSumModel(), [path], oracle_probe(episode), arms=("shuffled", "resampled")
    )

    for name in ("shuffled", "resampled"):
        arm = result.arms[name]
        assert arm.windows_total > 0, name
        assert arm.windows_changed == 0, name
        assert arm.is_interpretable() is False, name
        assert np.isnan(arm.position_delta()).all(), name
        assert np.isnan(arm.angle_delta()).all(), name
        np.testing.assert_array_equal(arm.window_steps_changed, [0] * arm.windows_total)
        np.testing.assert_array_equal(arm.window_multiset_distance, [0] * arm.windows_total)
        # The CURVES are still real numbers: it is the delta that is undefined.
        np.testing.assert_array_equal(arm.position, result.real.rssm_position, name)
    with pytest.raises(ValueError, match="support"):
        ladder(ActionSumModel(), [path], oracle_probe(episode), contrast=(3, 0))


def test_a_resample_that_reproduces_the_real_multiset_is_still_counted_by_sequence(
    tmp_path, monkeypatch
):
    """"The counts were randomised" is not the same claim as "the intervention
    intervened", and the two come apart exactly when a draw happens to be a
    rearrangement of the real window.

    Forced with a stub generator that returns a ROTATION of the window's own
    actions: the multiset is identical, the sequence is not, and a `changed`
    computed from the multiset -- the obvious thing to reach for on a rung
    whose point is the multiset -- reports 0 where the truth is 1.
    """
    episode = varied_action_episode(length=ONE_WINDOW)
    path = write(tmp_path, episode)
    window = episode.actions[CONTEXT : CONTEXT + HORIZON]
    rotated = np.roll(window, 1)
    assert not np.array_equal(rotated, window)

    class _RotatingRng:
        def choice(self, values, size, p):
            return rotated.astype(np.int64)

    monkeypatch.setattr(
        diagnostics_module, "default_rng", lambda seed: _RotatingRng()
    )
    arm = ladder(
        ActionSumModel(), [path], oracle_probe(episode), arms=("resampled",)
    ).arms["resampled"]

    assert arm.windows_total == 1
    assert arm.windows_changed == 1
    assert int(arm.window_steps_changed.sum()) == int((rotated != window).sum())
    # And the multiset distance -- the number the resampled rung is FOR -- is
    # exactly zero here, which is what separates "the sequence moved" from
    # "the counts moved" in the record.
    np.testing.assert_array_equal(arm.window_multiset_distance, [0])


def test_the_multiset_distance_is_the_steps_needed_to_turn_one_multiset_into_the_other(
    tmp_path, monkeypatch
):
    """The resampled rung's steps-moved count is SEQUENCE distance, and on the
    shipped split it reads 35 of 45 while the counts moved by only 15 -- a
    reader of the table cannot tell that the multiset moved by less than half
    of what the row suggests. So every arm also carries the multiset distance,
    half the L1 between the two bincounts, pinned here on a hand-built draw:
    the window's actions are a permutation of `[0..4]` and the draw is
    `[0, 0, 0, 2, 2]`, so the bincounts differ by |1-3|+|1-0|+|1-2|+|1-0|+|1-0|
    = 6 and the distance is 3, while the sequence distance is 4 or 5.
    """
    episode = varied_action_episode(length=ONE_WINDOW)
    path = write(tmp_path, episode)
    window = episode.actions[CONTEXT : CONTEXT + HORIZON]
    assert sorted(window.tolist()) == [0, 1, 2, 3, 4]
    draw = np.array([0, 0, 0, 2, 2], dtype=np.int64)

    class _FixedRng:
        def choice(self, values, size, p):
            return draw

    monkeypatch.setattr(diagnostics_module, "default_rng", lambda seed: _FixedRng())
    arm = ladder(
        ActionSumModel(), [path], oracle_probe(episode), arms=("resampled",)
    ).arms["resampled"]
    np.testing.assert_array_equal(arm.window_multiset_distance, [3])
    assert arm.mean_multiset_distance == pytest.approx(3.0)
    assert int(arm.window_steps_changed[0]) == int((draw != window).sum()) >= 4
    # The shuffled rung's is zero by construction, on a window it did change.
    monkeypatch.undo()
    shuffled = ladder(
        ActionSumModel(), [path], oracle_probe(episode), arms=("shuffled",)
    ).arms["shuffled"]
    assert shuffled.windows_changed == 1
    np.testing.assert_array_equal(shuffled.window_multiset_distance, [0])


def test_each_rung_carries_its_own_counts_and_its_own_delta(tmp_path):
    """Two episodes on which the rungs DISAGREE, so a single shared `changed`
    mask -- or one delta reported under three names -- is caught.

    The uniform window (all MOVE_FORWARD) is a genuine no-op for the
    permutation, which can only rearrange; for the constant rung it is where
    the held-MOVE_FORWARD arm is itself a no-op while the held-NOOP arm moves
    every step -- and the contrast between them is a genuine intervention. A
    `changed` mask computed once for the ladder cannot be right for all of
    them.
    """
    uniform = uniform_action_episode(length=ONE_WINDOW, action=3)
    varied = varied_action_episode(length=ONE_WINDOW)
    paths = [write(tmp_path, uniform, 0), write(tmp_path, varied, 1)]
    result = ladder(ActionSumModel(), paths, oracle_probe(varied))

    assert result.contrast == (3, 0)
    counts = {name: result.arms[name].windows_changed for name in LADDER}
    assert counts["shuffled"] == 1, counts
    assert counts["constant"] == 2, counts
    # The permutation left the uniform window alone; so did holding its own
    # action, and each says so in its own count, while holding NOOP there and
    # the contrast between the two did not.
    np.testing.assert_array_equal(result.arms["shuffled"].window_steps_changed[:1], [0])
    rung = result.arms["constant"]
    assert rung.held[3].window_steps_changed[0] == 0
    # The MINIMUM, on windows where it is not the integer part of the mean:
    # holding MOVE_FORWARD moved 0 steps on the uniform window and 4 on the
    # varied one, so `int(mean)` is 2 and the minimum is 0.
    np.testing.assert_array_equal(rung.held[3].window_steps_changed, [0, HORIZON - 1])
    assert rung.held[3].min_steps_changed == 0
    assert rung.held[3].mean_steps_changed == pytest.approx((HORIZON - 1) / 2)
    assert rung.held[0].window_steps_changed[0] == HORIZON
    assert rung.window_steps_changed[0] == HORIZON
    deltas = [tuple(result.arms[name].position_delta()) for name in LADDER]
    assert len(set(deltas)) == len(LADDER), deltas


def test_only_the_imagination_sees_the_intervened_actions(tmp_path, monkeypatch):
    """The intervention's SCOPE, for the whole ladder.

    The context filter and the floor are the fixed brackets of every comparison
    here, and the floor consumes the horizon actions too -- a rung that wrote
    through `_Window.horizon_actions` instead of building its own tensor would
    move both, and the floor would stop being the record's own.

    The canonical `imagine` call is checked in the same test: it must still be
    handed the REAL horizon actions after three rungs have run, which is what
    rules out one rung's buffer leaking into another's or into the reference.
    """
    episode = action_skewed_episode()
    path = write(tmp_path, episode)
    model = ActionSumModel()
    imagined = _imagined_actions(monkeypatch, model)
    observed: list[np.ndarray] = []
    real = model.rssm.observe
    monkeypatch.setattr(
        model.rssm,
        "observe",
        lambda embeddings, actions, state=None: observed.append(
            actions[0].cpu().numpy().copy()
        )
        or real(embeddings, actions, state=state),
    )
    result = ladder(model, [path], oracle_probe(episode), contrast=SKEWED_CONTRAST)

    _, canonical = _per_rung(imagined, _ladder_arms(result), windows=2)
    context_calls, floor_calls = observed[0::2], observed[1::2]
    for index, start in enumerate(window_starts(episode.length, CONTEXT, HORIZON)):
        horizon = episode.actions[start + CONTEXT : start + CONTEXT + HORIZON]
        np.testing.assert_array_equal(
            context_calls[index], episode.actions[start : start + CONTEXT]
        )
        np.testing.assert_array_equal(floor_calls[index], horizon)
        np.testing.assert_array_equal(canonical[index], horizon)


def test_the_ladder_is_reported_in_increasing_perturbation_order(tmp_path):
    """The ladder's order is the reader's only way to see monotonicity, so it
    is the LADDER's order and not the caller's argument order -- a report whose
    columns came out in whatever sequence the flag was typed in invites reading
    a non-monotone ladder as a monotone one."""
    episode = varied_action_episode()
    path = write(tmp_path, episode)
    model, probe = ActionSumModel(), oracle_probe(episode)

    assert LADDER == ("shuffled", "resampled", "constant")
    assert ladder(model, [path], probe, arms=("constant", "shuffled")).order == (
        "shuffled", "constant",
    )
    assert ladder(model, [path], probe, arms=tuple(reversed(LADDER))).order == LADDER


def test_every_rung_names_what_it_perturbs_and_nothing_else_does():
    """The verdict says "action-conditioned but insensitive to <what the null
    rungs perturb>", and that phrase is the ONLY place the rungs are described
    to a reader. A rung without a phrase would leave a hole in the sentence
    that decides M4; a phrase without a rung is a description of nothing. So
    the two are pinned to be the same set, and the phrases pinned distinct --
    two rungs described identically cannot be told apart in the verdict."""
    assert set(LADDER_PERTURBS) == set(LADDER)
    phrases = [LADDER_PERTURBS[name] for name in LADDER]
    assert len(set(phrases)) == len(LADDER), phrases
    assert all(phrase and phrase == phrase.strip() for phrase in phrases)


@pytest.mark.parametrize("arms", [(), ("shuffled", "nonesuch")])
def test_the_ladder_refuses_a_rung_it_does_not_have_and_an_empty_ladder(tmp_path, arms):
    """An unknown rung silently dropped leaves a report that is missing a
    column nobody asked after; an empty ladder produces a full-looking run with
    no intervention in it at all."""
    episode = varied_action_episode()
    path = write(tmp_path, episode)
    with pytest.raises(ValueError, match="rung"):
        ladder(ActionSumModel(), [path], oracle_probe(episode), arms=arms)


def test_the_resampled_rung_is_reproducible_under_the_seed_and_moves_with_it(tmp_path):
    """Both halves. Without the seed the rung is not reproducible; with a seed
    that no longer reaches the draw it is not an intervention. Asserted on the
    resampled rung specifically, because it is the only NEW rung that draws."""
    episode = action_skewed_episode()
    path = write(tmp_path, episode)
    model, probe = ActionSumModel(), oracle_probe(episode)
    curve = lambda seed: ladder(  # noqa: E731
        model, [path], probe, arms=("resampled",), intervention_seed=seed
    ).arms["resampled"].position

    np.testing.assert_array_equal(curve(0), curve(0))
    assert not np.array_equal(curve(0), curve(1))


def test_the_ladder_records_the_seed_the_held_actions_and_the_contrast_it_used(tmp_path):
    """All three are choices the reader cannot recover from the numbers, and
    the contrast in particular decides what the top rung's sign means -- see
    `action_skewed_episode`. When the constant rung does not run, the two that
    belong to it are None rather than a default that reads as a choice made."""
    episode = action_skewed_episode()
    path = write(tmp_path, episode)
    result = ladder(
        ActionSumModel(), [path], oracle_probe(episode), intervention_seed=7,
        contrast=SKEWED_CONTRAST,
    )
    assert result.intervention_seed == 7
    assert result.held_actions == (1, 2, 4)
    assert result.contrast == SKEWED_CONTRAST

    without = ladder(
        ActionSumModel(), [path], oracle_probe(episode), arms=("shuffled",)
    )
    assert without.held_actions is None and without.contrast is None


# ---------------------------------------------------------------------------
# The embedding-space reading and its noise reference.
#
# The position delta passes through a ridge probe whose selection R^2 is
# 0.30-0.38 on the feature arms and -0.036 on the pixel arm, so the effect size
# it reports is attenuated by probe quality and the pixel arm cannot be read
# at all. The reading below is probe-free: per window, the L2 between the
# intervened imagination's EMBEDDING and the real one's, per horizon step, and
# a NOISE REFERENCE -- a second real-action imagination from a different stream
# point -- as the ruler. The tests are arranged around the one way the ruler
# can be wrong without any shape saying so: the second imagination moves the
# global stream, so the NEXT window's snapshot moves and the canonical pass
# drifts. Every stream test runs on both devices for the reason the module
# docstring gives.
# ---------------------------------------------------------------------------


def _two_episode_rig(tmp_path, device):
    """Two synthetic episodes on the real sampling RSSM: four windows, so the
    stream tests below are cross-WINDOW and cross-EPISODE statements."""
    paths = [write(tmp_path, synthetic_episode(), index) for index in range(2)]
    model, probe = real_model_and_probe()
    return model.to(device), paths, probe


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_the_noise_reference_leaves_the_stream_exactly_where_evaluate_rollout_does(
    tmp_path, device
):
    """THE stream guard, on the generator state itself and not only on curves.

    The noise reference is one more `imagine` per window. If it runs before the
    canonical pass, or after it without putting the stream back, or restores
    the CPU generator only on MPS, window `w + 1` starts from a different state
    than `evaluate_rollout` started it from -- and every curve after window 1
    moves. Asserted two ways in one test: the device-aware generator state at
    the end of the traversal is bitwise the state `evaluate_rollout` leaves
    (both keys), AND the real arm's curves are bitwise the rollout's. A curve
    equality alone could pass on a one-window fixture; the state equality
    cannot, and the fixture spans four windows besides.

    The noise reference is asserted to have RUN in the same test -- positive
    in every window, stream reported restored -- so the equality cannot be
    explained by the reference not existing.
    """
    model, paths, probe = _two_episode_rig(tmp_path, device)

    reference = rollout(model, paths, probe, device=device)
    left_by_rollout = diagnostics_module._rng_snapshot(device)
    result = ladder(model, paths, probe, device=device)
    left_by_ladder = diagnostics_module._rng_snapshot(device)

    assert result.windows_total == 4
    assert set(left_by_ladder) == set(left_by_rollout)
    for key in left_by_rollout:
        assert torch.equal(left_by_ladder[key], left_by_rollout[key]), (
            f"the {key} generator ends somewhere else than evaluate_rollout leaves it"
        )
    np.testing.assert_array_equal(result.real.rssm_position, reference.rssm_position)
    np.testing.assert_array_equal(result.real.rssm_angle, reference.rssm_angle)
    np.testing.assert_array_equal(result.real.floor_position, reference.floor_position)
    noise = result.noise_reference
    assert noise.stream_restored is True
    assert noise.windows_collapsed == 0
    assert noise.window_embedding_distance.shape == (4,)
    assert noise.window_embedding_distance.min() > 0.0


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_a_noise_reference_that_forgets_to_restore_the_stream_moves_the_canonical_pass(
    tmp_path, device, monkeypatch
):
    """The proof that the guard above CAN fail -- the mutation, run inside the
    test on both devices.

    `_noise_reference` is wrapped so that after the real helper has drawn and
    restored, one more `imagine` is consumed and NOT restored. The self-check
    is computed by `_diagnose` from a snapshot taken BEFORE the helper was
    called, so it must read False here -- a self-check computed inside the
    helper, or hardcoded True, passes the test above and fails this one. And
    the real arm must drift from `evaluate_rollout` from window 2 on, which is
    what a real sampling rig shows and a deterministic one never would.
    """
    model, paths, probe = _two_episode_rig(tmp_path, device)
    reference = rollout(model, paths, probe, device=device)
    real_helper = diagnostics_module._noise_reference

    def forgetful(model, handle, tail):
        latent = real_helper(model, handle, tail)
        model.rssm.imagine(handle.horizon_actions, handle.state)
        return latent

    monkeypatch.setattr(diagnostics_module, "_noise_reference", forgetful)
    result = ladder(model, paths, probe, device=device)

    assert result.noise_reference.stream_restored is False
    assert not np.array_equal(result.real.rssm_position, reference.rssm_position), (
        "an unrestored draw did not move the canonical pass; the real arm is not "
        "being replayed from the per-window snapshot"
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="the MPS half needs MPS"
)
def test_the_stream_restored_self_check_reads_false_under_a_cpu_only_restore_on_mps(
    tmp_path, monkeypatch
):
    """The MPS half of the self-check: `torch.set_rng_state` alone restores the
    CPU generator and leaves the MPS one where the noise draw left it.

    Measured this session: an `imagine` on MPS moves ONLY the MPS state, so a
    self-check that compared the `cpu` key alone would read True under a
    CPU-only restore -- both keys must be compared. Unpatched in the same test,
    the check reads True and the real arm is bitwise the rollout's.
    """
    device = torch.device("mps")
    model, paths, probe = _two_episode_rig(tmp_path, device)
    reference = rollout(model, paths, probe, device=device)

    with monkeypatch.context() as patch:
        patch.setattr(
            diagnostics_module, "_rng_restore",
            lambda state: torch.set_rng_state(state["cpu"]),
        )
        broken = ladder(model, paths, probe, device=device)
    assert broken.noise_reference.stream_restored is False
    assert not np.array_equal(broken.real.rssm_position, reference.rssm_position)

    intact = ladder(model, paths, probe, device=device)
    assert intact.noise_reference.stream_restored is True
    np.testing.assert_array_equal(intact.real.rssm_position, reference.rssm_position)


def test_every_rungs_embedding_distance_is_the_closed_form_of_the_actions_it_imagined(
    tmp_path, monkeypatch
):
    """The positive half of the two-sided test, as exact per-window values.

    On `ActionSumModel` the embedding head is the latent's first column, the
    frame TAG, and the imagined tag after k steps is `h0 + cumsum(actions)[k]`
    -- so the per-step L2 between an intervened imagination and the real one
    is `|cumsum(a_int) - cumsum(a_real)|[k]`, and the window's numerator is
    its mean over the horizon. Computed from the action tensors the spy caught
    ON THE WAY INTO `imagine`, per window, in order; asserted on the SERIES and
    never on its mean, so a reversed or sorted series is caught. The latent is
    `cat[tag, tag]`, so an L2 taken in latent space rather than through the
    head is exactly sqrt(2) too large -- the rig separates the two pipelines
    by a known factor rather than by coincidence.

    The constant rung's numerator is the L2 between the two HELD imaginations
    directly -- `|cumsum(first) - cumsum(second)|`, the real arm cancelling as
    it does in the position contrast -- and each held action keeps its own
    distance to the real imagination beside it.

    This rig draws no random numbers, so the noise reference is bitwise the
    canonical pass in every window: `windows_collapsed == windows_total` and
    every ratio is NaN, which is the documented "no spread to read against"
    state and not a defect. The noise call is asserted to have received the
    real actions by `_per_rung`.
    """
    episode = action_skewed_episode()
    path = write(tmp_path, episode)
    model = ActionSumModel()
    seen = _imagined_actions(monkeypatch, model)
    result = ladder(model, [path], oracle_probe(episode), contrast=SKEWED_CONTRAST)

    arms, canonical = _per_rung(seen, _ladder_arms(result), windows=2)
    closed_form = lambda a, b: np.array(  # noqa: E731
        [np.mean(np.abs(np.cumsum(x) - np.cumsum(y))) for x, y in zip(a, b)]
    )
    curve = lambda a, b: np.mean(  # noqa: E731
        [np.abs(np.cumsum(x) - np.cumsum(y)) for x, y in zip(a, b)], axis=0
    )
    noise = result.noise_reference
    assert noise.windows_collapsed == noise.window_embedding_distance.size == 2
    np.testing.assert_array_equal(noise.window_embedding_distance, 0.0)
    varied = []
    for name in ("shuffled", "resampled"):
        arm = result.arms[name]
        np.testing.assert_allclose(
            arm.window_embedding_distance, closed_form(arms[name], canonical),
            rtol=1e-12, err_msg=name,
        )
        np.testing.assert_allclose(
            arm.embedding_distance_curve, curve(arms[name], canonical), rtol=1e-12,
        )
        assert arm.window_embedding_distance.min() > 0.0, name
        assert np.isnan(arm.embedding_ratio()), name
        varied.append(arm.window_embedding_distance[0] != arm.window_embedding_distance[1])
    rung = result.arms["constant"]
    for action, held in rung.held.items():
        np.testing.assert_allclose(
            held.window_embedding_distance,
            closed_form(arms[("constant", action)], canonical), rtol=1e-12,
            err_msg=str(action),
        )
        assert np.isnan(held.embedding_ratio()), action
        varied.append(held.window_embedding_distance[0] != held.window_embedding_distance[1])
    first, second = SKEWED_CONTRAST
    np.testing.assert_allclose(
        rung.window_embedding_distance,
        closed_form(arms[("constant", first)], arms[("constant", second)]), rtol=1e-12,
    )
    np.testing.assert_allclose(
        rung.embedding_distance_curve,
        curve(arms[("constant", first)], arms[("constant", second)]), rtol=1e-12,
    )
    assert rung.window_embedding_distance.min() > 0.0
    assert np.isnan(rung.embedding_ratio())
    assert any(varied), "every per-window series is flat, so its order is untestable"


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_the_embedding_distance_is_positive_and_below_sampling_noise_on_a_real_model(
    tmp_path, device
):
    """The positive half on the SAMPLING rig, where the ratio is finite.

    A real `RSSM`'s `_step` sees the action one-hot, so every rung's numerator
    is positive in every window -- and on this untrained model it is far
    smaller than what a second draw of the same imagination produces: measured,
    the ratio is ~0.05 on both devices. That is the `0 < ratio < 1` branch of
    the documented interpretation, which no deterministic rig can reach (its
    noise is exactly zero) and which is asserted here so the branch is
    exercised rather than merely described. Asserted per rung AND per held
    action, so a numerator computed real-against-real on one of them is caught
    by name.
    """
    path = write(tmp_path, varied_action_episode())
    model, probe = real_model_and_probe()
    model = model.to(device)
    result = ladder(model, [path], probe, device=device)

    rung = result.arms["constant"]
    readings = [result.arms["shuffled"], result.arms["resampled"], rung, *rung.held.values()]
    for arm in readings:
        assert arm.windows_changed == arm.windows_total > 0, arm.name
        assert (arm.window_embedding_distance > 0.0).all(), arm.name
        assert (arm.embedding_distance_curve > 0.0).all(), arm.name
        ratio = arm.embedding_ratio()
        assert np.isfinite(ratio), arm.name
        assert 0.0 < ratio < 1.0, (arm.name, ratio)


def test_the_noise_rulers_per_window_series_and_curve_are_the_means_of_its_per_step_distances(
    tmp_path, monkeypatch
):
    """The denominator of every ratio, pinned to the per-step matrix it is
    reduced from -- on the SAMPLING rig, where that matrix varies over both
    axes. Every other assertion on the noise series is a shape, a `> 0`, or
    an equality between two ladders, and the deterministic rig's noise is
    exactly 0 whatever the reduction; so `max(axis=1)`, `[:, 0]` and
    `max(axis=0)` all passed until this test existed.

    `_embedding_distance` is spied; the noise call is the LAST per window
    (`_diagnose` draws the reference after every arm and the canonical pass),
    so the captured rows are sliced at the per-window stride. The matrix is
    asserted to vary along both axes first, or the reductions could coincide
    on it; then the per-window ruler is its horizon mean and the curve its
    window mean, bitwise. And every rung carries the same ruler and the same
    curve, so a rung's ratio is computable from the rung alone.
    """
    model, paths, probe = _two_episode_rig(tmp_path, torch.device("cpu"))
    captured: list[np.ndarray] = []
    real_distance = diagnostics_module._embedding_distance
    monkeypatch.setattr(
        diagnostics_module, "_embedding_distance",
        lambda a, b: _tee(captured, real_distance(a, b)),
    )
    result = ladder(model, paths, probe)

    per_window = len(captured) // result.windows_total
    assert len(captured) == per_window * result.windows_total == 10 * 4, len(captured)
    per_step = np.stack(captured[per_window - 1 :: per_window])
    assert per_step.shape == (4, HORIZON)
    assert (per_step.std(axis=1) > 0.0).all(), "flat over the horizon: the reduction is untestable"
    assert (per_step.std(axis=0) > 0.0).all(), "flat over windows: the reduction is untestable"

    noise = result.noise_reference
    np.testing.assert_array_equal(noise.window_embedding_distance, per_step.mean(axis=1))
    np.testing.assert_array_equal(noise.embedding_distance_curve, per_step.mean(axis=0))
    rung = result.arms["constant"]
    for arm in (result.arms["shuffled"], result.arms["resampled"], rung, *rung.held.values()):
        np.testing.assert_array_equal(arm.window_noise_distance, per_step.mean(axis=1), err_msg=arm.name)
        np.testing.assert_array_equal(arm.noise_distance_curve, per_step.mean(axis=0), err_msg=arm.name)


def _tee(sink: list, value):
    sink.append(value)
    return value


def test_the_standalone_shuffle_carries_the_shuffled_rungs_embedding_reading(
    tmp_path, monkeypatch
):
    """`action_shuffled_rollout` is the ladder's shuffled rung under the field
    names the shipped records were written from, and it carries the rung's
    probe-free reading too -- asserted through THAT entry point, on the closed
    form, because nothing in `src/` or `scripts/` consumes it there and a
    dropped kwarg would leave the standalone view with `None` in three fields
    and every test green.

    On `ActionSumModel` the numerator is `|cumsum(permuted) - cumsum(real)|`
    per step, horizon-meaned per window; the rig draws nothing, so the noise
    series and its curve are exactly 0 in every window and step.
    """
    episode = action_skewed_episode()
    path = write(tmp_path, episode)
    model = ActionSumModel()
    seen = _imagined_actions(monkeypatch, model)
    result = shuffle(model, [path], oracle_probe(episode))

    assert len(seen) == 3 * 2
    permuted, canonical = seen[0::3], seen[1::3]
    distances = [np.abs(np.cumsum(a) - np.cumsum(b)) for a, b in zip(permuted, canonical)]
    np.testing.assert_allclose(
        result.window_embedding_distance, [d.mean() for d in distances], rtol=1e-12
    )
    np.testing.assert_allclose(result.embedding_distance_curve, np.mean(distances, axis=0), rtol=1e-12)
    assert result.window_embedding_distance.min() > 0.0
    np.testing.assert_array_equal(result.window_noise_distance, np.zeros(2))
    np.testing.assert_array_equal(result.noise_distance_curve, np.zeros(HORIZON))
    assert np.isnan(result.embedding_ratio())


def test_the_embedding_distance_is_computed_on_rows_the_embedding_head_emitted(
    tmp_path, monkeypatch
):
    """The single-pipeline rule, for the new reading.

    Every row handed to `_embedding_distance` -- both arguments, every call --
    must be a row the embedding head emitted: the intervened imagination, the
    real one, or the noise reference, all through `model.heads`. An L2 taken
    on the raw latent is caught by rows the head never produced. The call
    count is typed first: two sequence rungs, six held actions and the noise
    reference against the real imagination, plus the contrast's held pair,
    per window over two windows. And `apply_probe` is called exactly as often
    as before -- the noise reference is NOT probed; it is an embedding-space
    ruler only.
    """
    path = write(tmp_path, synthetic_episode())
    model, probe = real_model_and_probe()

    head_rows: set[bytes] = set()
    real_heads = model.heads.forward

    def spy_heads(latent):
        out = real_heads(latent)
        for row in out["embedding"].reshape(-1, out["embedding"].shape[-1]):
            head_rows.add(row.cpu().numpy().astype(np.float32).tobytes())
        return out

    distances: list[tuple[np.ndarray, np.ndarray]] = []
    real_distance = diagnostics_module._embedding_distance
    probed: list[int] = []
    real_apply = diagnostics_module.apply_probe
    monkeypatch.setattr(model.heads, "forward", spy_heads)
    monkeypatch.setattr(
        diagnostics_module, "_embedding_distance",
        lambda a, b: distances.append((np.asarray(a), np.asarray(b))) or real_distance(a, b),
    )
    monkeypatch.setattr(
        diagnostics_module, "apply_probe",
        lambda p, rows: probed.append(1) or real_apply(p, rows),
    )
    result = ladder(model, [path], probe)

    assert result.windows_total == 2
    assert len(result.held_actions) == 6
    assert len(distances) == (2 + 6 + 1 + 1) * 2, len(distances)
    assert len(probed) == (3 + 2 + 6) * 2, len(probed)
    for a, b in distances:
        assert a.shape == b.shape == (HORIZON, 2048), (a.shape, b.shape)
        for row in np.concatenate([a, b]):
            assert row.astype(np.float32).tobytes() in head_rows


def test_the_noise_reference_imagines_the_real_actions_from_the_context_state_at_a_different_stream_point(
    tmp_path, monkeypatch
):
    """What ELSE gives a nonzero noise reference: an intervened sequence, or a
    cold state. Neither is a sampling-noise ruler, so both are excluded here.

    Per window, the canonical `imagine` and the noise `imagine` must receive
    the SAME action tensor and the SAME `(h, z)` state -- bitwise -- and return
    DIFFERENT latents, because the only thing allowed to differ between them
    is the point in the sampling stream. A noise reference replayed from the
    per-window snapshot returns the canonical latent and is caught by the
    inequality; one handed a permuted sequence or `state=None` is caught by
    the equalities.
    """
    path = write(tmp_path, varied_action_episode())
    model, probe = real_model_and_probe()
    calls: list[tuple] = []
    real = model.rssm.imagine

    def spy(actions, state):
        out = real(actions, state)
        calls.append((actions.clone(), tuple(s.clone() for s in state), out["latent"].clone()))
        return out

    monkeypatch.setattr(model.rssm, "imagine", spy)
    result = ladder(model, [path], probe, arms=("shuffled",))

    stride = 1 + 2
    assert len(calls) == stride * result.windows_total
    for window in range(result.windows_total):
        canonical = calls[window * stride + 1]
        noise = calls[window * stride + 2]
        assert torch.equal(canonical[0], noise[0]), window
        assert all(torch.equal(a, b) for a, b in zip(canonical[1], noise[1])), window
        assert not torch.equal(canonical[2], noise[2]), (
            f"window {window}: the noise reference is the canonical imagination bitwise"
        )
    assert result.noise_reference.windows_collapsed == 0


def _hand_built_arm(numerator, changed, noise):
    """An `ArmResult` with only what the embedding reading consumes."""
    numerator = None if numerator is None else np.asarray(numerator, float)
    changed = np.asarray(changed, bool)
    return diagnostics_module.ArmResult(
        name="shuffled",
        position=np.zeros(HORIZON), angle=np.zeros(HORIZON),
        window_position_delta=np.zeros((changed.size, HORIZON)),
        window_angle_delta=np.zeros((changed.size, HORIZON)),
        window_steps_changed=np.where(changed, HORIZON, 0),
        window_multiset_distance=np.zeros(changed.size, int),
        horizon=HORIZON,
        window_embedding_distance=numerator,
        embedding_distance_curve=None if numerator is None else np.zeros(HORIZON),
        window_noise_distance=None if noise is None else np.asarray(noise, float),
    )


def test_the_embedding_ratio_is_a_ratio_of_medians_over_the_changed_windows_and_undefined_without_spread():
    """The statistic, pinned by values chosen so every wrong summary is a
    different number.

    Numerator [1, 2, 9, 100] over changed [T, T, T, F], noise [3, 4, 5, 100]:
    the medians over the CHANGED windows are 2 and 4, ratio 0.5. Over ALL
    windows they would be 5.5 and 4.5; the means over the changed windows are
    4 and 4, ratio 1.0. Both series are taken over the same windows -- the
    ones the intervention changed -- so the ratio is a paired comparison and
    a noisier unchanged window cannot move its denominator.

    Three ways to have no ratio, each on its own: no changed window (NaN, not
    0), a noise median of exactly 0 -- a deterministic rig -- (NaN, never inf
    and never 0), and a rung that was never measured in embedding space
    (ValueError naming it, so a hand-built result cannot print as a null).
    A zero numerator over a POSITIVE noise median is exactly 0.0.
    """
    arm = _hand_built_arm([1.0, 2.0, 9.0, 100.0], [True, True, True, False], [3.0, 4.0, 5.0, 100.0])
    assert arm.embedding_median() == 2.0
    assert arm.noise_median() == 4.0
    assert arm.embedding_ratio() == 0.5

    zero = _hand_built_arm([0.0, 0.0, 0.0, 0.0], [True] * 4, [3.0, 4.0, 5.0, 100.0])
    assert zero.embedding_ratio() == 0.0

    no_spread = _hand_built_arm([1.0, 2.0, 9.0, 100.0], [True] * 4, [0.0] * 4)
    assert np.isnan(no_spread.embedding_ratio()) and not np.isinf(no_spread.embedding_ratio())
    both_zero = _hand_built_arm([0.0] * 4, [True] * 4, [0.0] * 4)
    assert np.isnan(both_zero.embedding_ratio())

    unchanged = _hand_built_arm([1.0, 2.0, 9.0, 100.0], [False] * 4, [3.0, 4.0, 5.0, 100.0])
    assert np.isnan(unchanged.embedding_ratio())
    assert np.isnan(unchanged.embedding_median())

    with pytest.raises(ValueError, match="not measured"):
        _hand_built_arm(None, [True] * 4, [3.0, 4.0, 5.0, 100.0]).embedding_ratio()
    with pytest.raises(ValueError, match="not measured"):
        _hand_built_arm([1.0] * 4, [True] * 4, None).embedding_ratio()


def test_the_median_of_per_window_ratios_is_the_second_summary_and_differs_from_the_first():
    """Two summaries of the same per-window series, both persisted, because on
    the shipped frozen_ssl cells they straddle the 1.0 landmark (0.97 against
    1.06 on the held contrast). The ratio of medians is the decision-bearing
    one -- each window's denominator is a SINGLE draw of a distance, so a
    per-window division uses a very noisy ruler and carries the E[1/X] > 1/E[X]
    bias, while the population median is well estimated from 229 draws -- and
    the median of ratios is kept beside it so the sensitivity is readable.

    Numerator [1, 4, 9, 100] over noise [2, 4, 1, 100], changed [T, T, T, F]:
    the medians are 4 and 2, ratio 2.0; the per-window ratios are 0.5, 1.0
    and 9.0, median 1.0. Chosen so the two DIFFER -- on the first fixture
    above they coincide at 0.5, which is the L1 trap. NaN, never inf, when
    any changed window's noise is exactly 0; NaN when no window changed;
    exactly 0.0 on a zero numerator over positive noise.
    """
    arm = _hand_built_arm([1.0, 4.0, 9.0, 100.0], [True, True, True, False], [2.0, 4.0, 1.0, 100.0])
    assert arm.embedding_ratio() == 2.0
    assert arm.median_of_ratios() == 1.0

    zero = _hand_built_arm([0.0] * 4, [True] * 4, [3.0, 4.0, 5.0, 100.0])
    assert zero.median_of_ratios() == 0.0
    one_zero = _hand_built_arm([1.0, 4.0, 9.0, 100.0], [True] * 4, [2.0, 0.0, 1.0, 100.0])
    assert np.isnan(one_zero.median_of_ratios()) and not np.isinf(one_zero.median_of_ratios())
    # The zero sits in an UNCHANGED window: it is not in the ratio's support.
    unchanged_zero = _hand_built_arm([1.0, 4.0, 9.0, 100.0], [True, True, True, False], [2.0, 4.0, 1.0, 0.0])
    assert unchanged_zero.median_of_ratios() == 1.0
    assert np.isnan(_hand_built_arm([1.0] * 4, [False] * 4, [2.0] * 4).median_of_ratios())
    with pytest.raises(ValueError, match="not measured"):
        _hand_built_arm([1.0] * 4, [True] * 4, None).median_of_ratios()


def test_the_ladder_reads_the_feature_cache_for_the_embedding_distance_as_well(
    tmp_path, monkeypatch
):
    """Both encoder input kinds reach the new path. Under the feature backbone
    the +100 tag cache is what the oracle's context state carries, so every
    REAL-side row handed to `_embedding_distance` is at least 100 -- a ladder
    that read `episode.obs` for the feature arms would hand rows below it.

    What this does NOT show, said so it is not read into it: the numerator
    series itself is invariant to the offset (both sides carry it), so only
    the rows, not the distances, can tell the two sources apart.
    """
    episode = synthetic_episode()
    path = write(tmp_path, episode)
    tags = np.arange(episode.length + 1, dtype=np.float32) + 100.0
    np.save(path.with_suffix(".features_random_vit.npy"), tags.reshape(-1, 1, 1, 1))
    model = OracleModel()
    model.input_kind = "features"
    real_rows: list[np.ndarray] = []
    real_distance = diagnostics_module._embedding_distance
    monkeypatch.setattr(
        diagnostics_module, "_embedding_distance",
        lambda a, b: real_rows.append(np.asarray(b)) or real_distance(a, b),
    )
    result = ladder(model, [path], oracle_probe(episode), feature_backbone="random_vit")
    assert real_rows and all(rows.min() >= 100.0 for rows in real_rows)
    assert result.windows_total == 2


# ---------------------------------------------------------------------------
# Shared guards.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("diagnostic", ["shuffle", "ladder", "sweep"])
def test_the_diagnostics_read_the_arms_own_namespaced_feature_cache(
    tmp_path, diagnostic
):
    """Two of the three arms feed the RSSM cached features rather than pixels,
    and the caches are namespaced per backbone -- a diagnostic that reached for
    `episode.obs`, or for an unnamespaced `.features.npy`, would report one
    arm's numbers under another's.

    The cache carries the frame tag OFFSET BY +100, so reading the wrong source
    is a closed-form 100 map-index units of error rather than merely wrong
    provenance. The whole ladder is run under the feature backbone as well as
    the shuffled rung alone: the two new rungs share `_diagnose` with it and
    their marginal pre-pass reads only `actions`, so nothing SHOULD differ --
    which is a claim covered by construction until a test runs it.
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

    if diagnostic == "shuffle":
        reference = shuffle(model, [path], probe, feature_backbone="random_vit").real
    elif diagnostic == "ladder":
        result = ladder(model, [path], probe, feature_backbone="random_vit")
        reference = result.real
        for name in LADDER:
            assert result.arms[name].windows_changed > 0, name
    else:
        reference = sweep(
            model, [path], probe, ks=(HORIZON,), feature_backbone="random_vit"
        ).reference
    np.testing.assert_allclose(
        reference.rssm_position, np.full(HORIZON, 100.0 * STEP), rtol=1e-4
    )


@pytest.mark.parametrize("diagnostic", ["shuffle", "ladder", "sweep"])
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
    elif diagnostic == "ladder":
        result = ladder(model, [path], probe)
        # Three references, the two sequence rungs, and one held arm per
        # action in the support -- each probed once per window.
        assert len(result.held_actions) == 6, result.held_actions
        expected = (3 + 2 + len(result.held_actions)) * 2
    else:
        sweep(model, [path], probe, ks=ks)
        expected = (3 + len(ks)) * 2

    assert len(probed) == expected, len(probed)
    assert len({identity for identity, _ in probed}) == 1
    for _, rows in probed:
        for row in rows:
            assert row.astype(np.float32).tobytes() in head_rows


@pytest.mark.parametrize("diagnostic", ["shuffle", "ladder", "sweep"])
def test_the_diagnostics_raise_when_no_window_is_long_enough(tmp_path, diagnostic):
    """A silently empty or zero-filled curve would be read as a measurement.

    The ladder reaches this FIRST, in its own pre-pass over the action
    marginal, and must raise the same refusal there: a marginal counted over no
    window at all has no support to draw from and no action to hold, and the
    contrast check on an empty support would surface as a different error
    rather than as the split/horizon problem it is.
    """
    episode = synthetic_episode(length=CONTEXT + HORIZON)  # exactly `need`: excluded
    path = write(tmp_path, episode)
    probe = oracle_probe(episode)
    with pytest.raises(ValueError, match="no diagnostic window"):
        if diagnostic == "shuffle":
            shuffle(OracleModel(), [path], probe)
        elif diagnostic == "ladder":
            ladder(OracleModel(), [path], probe)
        else:
            sweep(OracleModel(), [path], probe, ks=(HORIZON,))


@pytest.mark.parametrize("diagnostic", ["shuffle", "ladder", "sweep"])
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
    elif diagnostic == "ladder":
        ladder(model, [path], probe)
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
    for function in (action_shuffled_rollout, action_intervention_ladder, regrounding_sweep):
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
    ladder_defaults = inspect.signature(action_intervention_ladder).parameters
    assert ladder_defaults["intervention_seed"].default == 0
    # The default ladder is ALL THREE rungs. Defaulting to the shuffled rung
    # alone would leave the two arms that close the multiset blind spot off
    # unless a flag was typed, and the shipped conclusion would be the old one
    # under a new name.
    assert ladder_defaults["arms"].default is LADDER


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


@pytest.mark.slow
@shipped
def test_the_ladder_on_a_shipped_checkpoint_leaves_the_shuffled_rung_bitwise_the_record(
    shipped_cell,
):
    """THE regression guard on the artifact: adding rungs must not move the old.

    `tests/eval/fixtures/shuffled_cnn_seed0_pre_ladder.json` is the shuffled
    arm's curve and per-window delta as `scripts/diagnose_dynamics.py` wrote
    them at commit 3688311 (pre-ladder), when the permutation was the ONLY intervention --
    copied verbatim before the ladder existed, and committed, because the
    record under `runs/` is gitignored and is rewritten by every run of the
    new code, so it cannot be the reference for the code that rewrites it.
    With every rung running, the shuffled rung must be bitwise the fixture: a
    resampled rung that drew from the permutation's generator, any rung that
    drew from torch, or a re-derived permutation stream would move it while
    every shape, every window count and every self-check stayed right. The
    other rungs are asserted to have changed every window in the same test, so
    the equality cannot be explained by the ladder having done nothing.

    THE EMBEDDING-SPACE READING RUNS ON THIS CELL TOO, in the same body, so the
    fixture pin above is a pin WITH the noise reference drawing one more
    imagination per window: the stream must come back after every one of the
    229 draws, none may be the canonical latent bitwise, and every rung's
    ratio must be finite. The ratio VALUES are not pinned -- they are new
    information about the pixel arm, which the position probe (R^2 -0.036)
    could not read -- they are printed. Measured on this box: shuffled
    0.309, resampled 0.401, held FORWARD-vs-NOOP 1.022.
    """
    import json

    model, val, probe, record, device, backbone, reference = shipped_cell
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "shuffled_cnn_seed0_pre_ladder.json").read_text()
    )
    assert (fixture["arm"], fixture["seed"]) == (SHIPPED_ARM, SHIPPED_SEED)
    assert fixture["device"] == str(device) and fixture["torch_version"] == torch.__version__
    result = action_intervention_ladder(
        model, val, probe, context=5, horizon=45, seed=SHIPPED_SEED,
        device=device, feature_backbone=backbone,
        intervention_seed=fixture["permutation_seed"],
    )

    assert result.order == LADDER
    np.testing.assert_array_equal(
        result.real.rssm_position, np.asarray(record["curves"]["rssm_position"])
    )
    shuffled = result.arms["shuffled"]
    np.testing.assert_array_equal(
        shuffled.position, np.asarray(fixture["shuffled_position"])
    )
    np.testing.assert_array_equal(
        shuffled.position_delta(), np.asarray(fixture["position_delta"])
    )
    for name in LADDER:
        arm = result.arms[name]
        assert arm.windows_total == SHIPPED_WINDOWS == fixture["windows_total"], name
        assert arm.windows_changed == SHIPPED_WINDOWS, name
        assert arm.mean_steps_changed > 0.0, name
    assert result.contrast == (3, 0)
    assert result.held_actions == (0, 1, 2, 3, 4, 5)
    for action, held in result.arms["constant"].held.items():
        assert held.windows_changed == SHIPPED_WINDOWS, action

    noise = result.noise_reference
    assert noise.stream_restored is True
    assert noise.windows_collapsed == 0
    assert noise.window_embedding_distance.shape == (SHIPPED_WINDOWS,)
    assert noise.window_embedding_distance.min() > 0.0
    assert shuffled.window_embedding_distance.shape == (SHIPPED_WINDOWS,)
    ratios = {name: result.arms[name].embedding_ratio() for name in LADDER}
    assert all(np.isfinite(ratio) for ratio in ratios.values()), ratios
    assert all(ratio > 0.0 for ratio in ratios.values()), ratios
    # Not pinned -- printed, so a `-s` run carries the reading in its log.
    print(f"cnn/seed0 embedding ratios: {ratios}")
