"""The study's reproduction rule: when a fresh pass reproduces a STORED artefact.

Through M3g every such comparison demanded `max |delta| == 0.0` and passed,
because every run had happened on one macOS build. The macOS 27.0 upgrade of
2026-09-24 changed the MPS kernels' reduction order. Measured over the nine
M3c cells, same checkpoints, same code (spec 2.4):

- the shipped `rssm_position` reproduces to at most 1.705303e-13 -- 6 ULPs of
  float64, largest relative 7.656e-16 -- and two of nine cells still exactly;
- no reported digit moves: every cell's endpoint is identical to ten decimals;
- the computation is bitwise deterministic WITHIN the new build;
- `ca3e140`, the commit the records were written at, and this branch compute
  bit-identical curves here, so no code changed the computation.

A wrong DEVICE -- the thing this gate exists to refuse -- misses by 6-12 MAP
UNITS, about thirteen orders of magnitude above the platform band. That ratio
is what makes a bound possible without blunting the gate.

`REPRODUCTION_ULPS` is calibrated, not theoretical: 64 is about ten times the
measured worst case, and far tighter than the ~N*ulp a reduction over hundreds
of windows could produce. A delta above it means re-characterise the platform,
not raise the bound.

Two things this rule is NOT. It is not for comparisons inside ONE run --
`diagnose_dynamics`'s `open_loop` and `stream` keep exactly 0.0, because
determinism within a build is measured, so nothing need be granted. And it is
not a rule about TRAINING: a retrain's per-step losses diverge from the
record's by far more than any ULP band, because the straight-through
categorical sampler makes training a discrete system where a last-bit change
flips a sampled class and the trajectory jumps. `checkpoint_ladder`'s anchor
and `sharper_latent`'s identity check therefore keep the exact rule and
refuse across a platform change, which is the right answer rather than a
tolerance to widen.
"""

from __future__ import annotations

import math

# Ten times the worst case measured across the nine M3c cells on macOS 27.0
# (6.000002 ULPs, frozen_ssl/s1), and thirteen orders below a wrong device.
REPRODUCTION_ULPS: int = 64


def reproduction_bound(magnitude, *, ulps: int = REPRODUCTION_ULPS) -> float:
    """The largest `max |delta|` that counts as reproducing a stored value of
    this magnitude.

    `0.0` for a zero or non-finite magnitude, and for `ulps <= 0`: nothing is
    granted a tolerance around a number that has no scale, and `ulps=0` is how
    a caller asks for the exact rule in so many words.
    """
    magnitude = abs(float(magnitude))
    if int(ulps) <= 0 or not math.isfinite(magnitude) or magnitude == 0.0:
        return 0.0
    return int(ulps) * math.ulp(magnitude)


def reproduces(delta, magnitude, *, ulps: int = REPRODUCTION_ULPS) -> bool:
    """Is `delta` within the bound for a stored value of `magnitude`.

    A non-finite delta never reproduces: `trust_horizon._max_delta` returns
    `inf` for a shape mismatch and `ladder.anchor_delta` for a NaN, and both
    must stay refusals rather than becoming unorderable comparisons. The
    comparison below would refuse those two on its own; the guard is what also
    refuses `-inf`, which no caller can produce -- both helpers return a max of
    absolute values -- but which the predicate should not silently accept.
    """
    delta = float(delta)
    if not math.isfinite(delta):
        return False
    return delta <= reproduction_bound(magnitude, ulps=ulps)
