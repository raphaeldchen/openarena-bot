"""Discrete action set.

The action set is derived from whatever buttons a scenario makes available, so
it is scenario-agnostic: a no-op plus one one-hot vector per button.

This deliberately forbids simultaneous presses (no move-and-shoot). That is a
real limitation, accepted for M0 under YAGNI -- the spec defers action-space
refinement to an open question, to be revisited once M5 results on
`deadly_corridor` show whether it binds.
"""


def build_action_set(n_buttons: int) -> list[list[int]]:
    """Return `n_buttons + 1` button vectors: a no-op, then one-hots.

    Raises:
        ValueError: if `n_buttons` is not positive.
    """
    if n_buttons < 1:
        raise ValueError(f"need at least one button, got {n_buttons}")
    noop = [0] * n_buttons
    one_hots = [[1 if i == j else 0 for j in range(n_buttons)] for i in range(n_buttons)]
    return [noop, *one_hots]
