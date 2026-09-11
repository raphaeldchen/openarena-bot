"""The rollout's window rule, owned in one place.

Four things score windows cut from a validation episode -- `evaluate_rollout`,
`gather_probe_data`, and the two diagnostics in `mbfps.eval.diagnostics` -- and
every number any of them reports is only comparable to the shipped records if
all four cut the SAME windows. The rule is three coupled facts, and each of them
has a weakening that changes what is scored while breaking no shape and failing
no smoke test, so it lives here rather than in four copies.

This is the same call `mbfps.data.split.VAL_FRACTION` records: a rule duplicated
at every call site drifts invisibly, and pinning copies against each other is
quadratic in the number of copies -- the pin that already exists,
`test_gather_probe_data_windows_match_the_rollouts_length_guard_exactly`, probes
exactly one length and is narrower than the protocol it names.
"""


def window_starts(episode_length: int, context: int, horizon: int) -> list[int]:
    """Every legal window start for an episode of `episode_length` transitions.

    A window at `start` spans `need + 1 = context + horizon + 1` frames, read as
    `source[start : start + need + 1]` out of an episode's T+1 rows.

    THE LENGTH GUARD IS `< need + 1`, NOT `< need`, and the difference is not a
    fast path. At `episode_length == need` the stride below is NOT empty:
    `range(0, episode_length - need + 1, need)` = `range(0, 1, need)` yields ONE
    legal window (start=0, reading `source[0 : need + 1]` -- exactly the T+1
    rows on hand). That window is deliberately EXCLUDED; a weaker `< need` check
    admits it instead, and `episode_length == need` is the one length at which
    the two forms disagree.

    THE STRIDE'S STOP CARRIES A `+ 1`. `start = episode_length - need` is a
    legal window -- it reads `privileged[...start+need]` and `obs[...start+need]`,
    both in range for a T+1-row array -- and `range(0, episode_length - need,
    need)` excludes it whenever `episode_length % need == 0`, silently dropping
    the final window of such an episode. The stride itself is the full window,
    so windows never overlap: overlapping windows reuse frames and make an
    averaged curve look tighter than the data supports.

    NO VALIDATION OF `context` OR `horizon` HAPPENS HERE, deliberately.
    `gather_probe_data` rejects `context < 1` and `horizon < 1` at its own entry
    with its own message and `evaluate_rollout` does not; hoisting that check
    into the shared rule would change `evaluate_rollout`'s behaviour as a side
    effect of sharing a range.
    """
    need = context + horizon
    if episode_length < need + 1:
        return []
    return list(range(0, episode_length - need + 1, need))
