"""The window rule, extracted -- and both of its existing call sites pinned to it.

`evaluate_rollout` and `gather_probe_data` each carried their own copy of three
coupled facts: `need = context + horizon`, the `< need + 1` length guard, and
the `+ 1` on the stride's stop. The two copies were pinned against each other by
`test_gather_probe_data_windows_match_the_rollouts_length_guard_exactly`, which
probes ONE length -- `episode.length == need`, the only length at which the
guard's two forms disagree. The stride's `+ 1` is invisible there, so a copy
that dropped it would have passed: on the shipped data that silently drops the
final window of every episode whose length is an exact multiple of `need`.

The diagnostics in `mbfps.eval.diagnostics` would have made four copies of the
rule pinned by pairwise tests, so the rule now lives in one place. These tests
guard the extraction from both sides: the boundary table pins the iterator's own
behaviour, and the routing tests require the two production call sites to
actually consume it rather than to merely agree with it today.
"""

import numpy as np
import pytest
import torch

import mbfps.eval.probe as probe_module
import mbfps.eval.rollout as rollout_module
from mbfps.data.episode import save_episode
from mbfps.eval.probe import gather_probe_data
from mbfps.eval.rollout import evaluate_rollout
from mbfps.eval.windows import window_starts
from tests.eval.test_rollout import (
    DX,
    OracleModel,
    oracle_probe,
    synthetic_episode,
)

# Deliberately TEST-LOCAL, never read from the module under test: parametrising
# over the production constants would let a shrunk production value silently
# shrink the test matrix.
CONTEXT, HORIZON = 2, 3
NEED = CONTEXT + HORIZON


def test_window_starts_over_the_boundary_lengths():
    """The whole rule as a table of exact lists, not of counts.

    Each row kills a different weakening. Length 5 (`== need`) separates the
    `< need + 1` guard from a `< need` one; lengths 10 and 15 (exact multiples
    of `need`) separate the stride's `- need + 1` stop from a `- need` one; the
    lists rather than their lengths separate a stride of `need` from a stride
    of 1.
    """
    table = {
        0: [],
        4: [],
        5: [],            # == need: one legal window, deliberately excluded
        6: [0],           # need + 1: the shortest admissible episode
        7: [0],
        9: [0],
        10: [0, 5],       # == 2 * need: the final window is legal and included
        11: [0, 5],
        15: [0, 5, 10],   # == 3 * need
    }
    for length, expected in table.items():
        assert window_starts(length, CONTEXT, HORIZON) == expected, length


def test_window_starts_admits_the_final_window_when_length_is_an_exact_multiple_of_need():
    """`start = length - need` reads `source[start : start + need + 1]`, exactly
    the T+1 rows on hand, so it is legal. `range(0, length - need, need)` drops
    it whenever `length % need == 0` -- on the shipped validation split that is
    22 of 229 windows, gone with no shape or smoke test noticing.

    Named separately from the table above because this is the half the existing
    `gather_probe_data` precedent test cannot see: it probes only
    `length == need`, where the stride never runs at all.
    """
    assert window_starts(3 * NEED, CONTEXT, HORIZON) == [0, NEED, 2 * NEED]


def test_window_starts_is_empty_at_exactly_need_frames_and_non_empty_one_frame_later():
    """One guard, two sides, both asserted. A single-sided assertion -- "length
    `need` yields nothing" -- is also satisfied by an iterator that yields
    nothing for anything, and "length `need + 1` yields one window" is satisfied
    by a guard that admits everything.
    """
    assert window_starts(NEED, CONTEXT, HORIZON) == []
    assert window_starts(NEED + 1, CONTEXT, HORIZON) == [0]


def test_window_starts_never_reads_past_the_episodes_last_frame():
    """A window at `start` reads rows `start .. start + need` of a T+1-row
    array, so `start + need + 1 <= length + 1` must hold for every start at
    every length. A bounds invariant rather than a table, so a stop that is one
    too generous is caught at whichever length happens to expose it."""
    for length in range(0, 4 * NEED + 4):
        for start in window_starts(length, CONTEXT, HORIZON):
            assert start >= 0, (length, start)
            assert start + NEED + 1 <= length + 1, (length, start)


def _tagged_episode(tmp_path, length, index=0):
    episode = synthetic_episode(length=length)
    path = tmp_path / f"ep_{index:06d}_len{length:05d}.npz"
    save_episode(episode, path)
    return episode, path


def _recorded_starts(monkeypatch, module, offset, run):
    """The window starts a function ACTUALLY used, recovered from its truth slices.

    `synthetic_episode` makes `privileged[t, 1] == DX * t`, so the first row of
    a truth slice names the frame it begins at, and subtracting the function's
    own slice offset recovers `start`. This reads what the function did rather
    than re-asking the iterator, which is what keeps the comparison honest.
    """
    seen: list[np.ndarray] = []
    real = module.probe_targets
    monkeypatch.setattr(
        module,
        "probe_targets",
        lambda privileged, keys: seen.append(privileged) or real(privileged, keys),
    )
    try:
        run()
    except ValueError as error:  # no window was long enough; zero starts is the answer
        assert "window" in str(error)
    return [int(round(float(rows[0, 1]) / DX)) - offset for rows in seen]


@pytest.mark.parametrize("length", [NEED, NEED + 1, 2 * NEED, 2 * NEED + 1, 3 * NEED])
def test_evaluate_rollout_takes_its_windows_from_the_shared_iterator(
    tmp_path, monkeypatch, length
):
    """The starts `evaluate_rollout` really scored, against the iterator's list.

    Lengths chosen so that each weakening shows up somewhere: `NEED` separates
    the guard, the exact multiples separate the stride's stop.
    """
    episode, path = _tagged_episode(tmp_path, length)
    starts = _recorded_starts(
        monkeypatch,
        rollout_module,
        CONTEXT + 1,  # evaluate_rollout slices truth at start + context + 1
        lambda: evaluate_rollout(
            OracleModel(), [path], oracle_probe(episode),
            context=CONTEXT, horizon=HORIZON, device=torch.device("cpu"),
        ),
    )
    assert starts == window_starts(length, CONTEXT, HORIZON)


@pytest.mark.parametrize(
    "length", [NEED - 1, NEED, NEED + 1, 2 * NEED, 2 * NEED + 1, 3 * NEED]
)
def test_gather_probe_data_and_evaluate_rollout_agree_at_every_length(
    tmp_path, monkeypatch, length
):
    """The precedent test's claim, generalised off its single boundary length.

    It compared the two functions only at `episode.length == need`. The stride
    bug lives at the exact multiples and the guard bug at `need`, so agreeing at
    one of them is not agreeing on the protocol. Compared as START SEQUENCES,
    not as counts: two functions can produce the same number of windows from
    different frames.

    The two TRUTH SLICES are legitimately different -- the rollout's latents
    begin one step after the context, `gather_probe_data`'s begin at the
    context's first step -- so each side's own offset is subtracted before the
    starts are compared. That difference must never be "unified" away.
    """
    episode, path = _tagged_episode(tmp_path, length)
    rollout = _recorded_starts(
        monkeypatch, rollout_module, CONTEXT + 1,
        lambda: evaluate_rollout(
            OracleModel(), [path], oracle_probe(episode),
            context=CONTEXT, horizon=HORIZON, device=torch.device("cpu"),
        ),
    )
    gather = _recorded_starts(
        monkeypatch, probe_module, 1,  # gather_probe_data slices truth at start + 1
        lambda: gather_probe_data(
            OracleModel(), [path], None, torch.device("cpu"),
            context=CONTEXT, horizon=HORIZON,
        ),
    )
    assert gather == rollout == window_starts(length, CONTEXT, HORIZON)


def test_both_call_sites_route_through_the_shared_iterator(tmp_path, monkeypatch):
    """Not "the two agree", but "the two consume the one function".

    Once both sides call `window_starts`, an agreement test can no longer fail
    on divergence -- it is satisfied by two calls to one function. The value it
    used to carry is transferred here: the iterator is replaced by a stub that
    yields only the FIRST start, and BOTH functions must lose their later
    windows. A call site that re-inlined `range(...)` would keep them.
    """
    length = 3 * NEED
    episode, path = _tagged_episode(tmp_path, length)
    assert len(window_starts(length, CONTEXT, HORIZON)) == 3, "fixture must have room"

    stub = lambda length, context, horizon: window_starts(  # noqa: E731
        length, context, horizon
    )[:1]
    monkeypatch.setattr(rollout_module, "window_starts", stub)
    monkeypatch.setattr(probe_module, "window_starts", stub)

    rollout = _recorded_starts(
        monkeypatch, rollout_module, CONTEXT + 1,
        lambda: evaluate_rollout(
            OracleModel(), [path], oracle_probe(episode),
            context=CONTEXT, horizon=HORIZON, device=torch.device("cpu"),
        ),
    )
    gather = _recorded_starts(
        monkeypatch, probe_module, 1,
        lambda: gather_probe_data(
            OracleModel(), [path], None, torch.device("cpu"),
            context=CONTEXT, horizon=HORIZON,
        ),
    )
    assert rollout == [0], "evaluate_rollout did not take its windows from the iterator"
    assert gather == [0], "gather_probe_data did not take its windows from the iterator"


def test_window_starts_does_not_validate_context_or_horizon():
    """Deliberately no guard here. `gather_probe_data` rejects `context < 1` and
    `horizon < 1` at its own entry (with its own message) and `evaluate_rollout`
    does not; moving that check into the shared iterator would change
    `evaluate_rollout`'s behaviour as a side effect of sharing a range."""
    assert window_starts(10, 0, 3) == [0, 3, 6]
    assert window_starts(10, -1, 3) == [0, 2, 4, 6, 8]
