"""M3j's script: the three gathers, the shared bases, and the record schema."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mbfps.eval.retention import K_REPORTED, RUNGS, TARGETS

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "latent_retention.py"


def _load():
    spec = importlib.util.spec_from_file_location("latent_retention_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load()


def _gathered(n_windows: int = 8, steps: int = 20, width: int = 6, h_dim: int = 3,
              seed: int = 0) -> dict:
    """A `gather_probe_data`-shaped dict with REAL structure: the deterministic
    half of the latent remembers the previous frame's position, so the
    deterministic rung must show a gain and a flat rung must not."""
    rng = np.random.default_rng(seed)
    rows = n_windows * steps
    window = np.repeat(np.arange(n_windows), steps)
    step = np.tile(np.arange(steps), n_windows)
    episode = window // 2
    # A per-row noise term on top of the linear trend. WITHOUT it, `p(t) -
    # p(t-k)` is a pure function of k alone (the trend's slope is fixed across
    # window and step), so every backward displacement is a CONSTANT vector --
    # zero variance, and `probe.fit_probe`'s own ridge selection raises
    # "every target column has zero variance" before any rung is scored. The
    # noise is exactly what only `h`'s lagged position (this row's OWN noisy
    # `p(t-1)`, not a resampled draw) can resolve: `enc(t)` alone sees this
    # row's noise but has no way to know an earlier row's independent draw,
    # while `enc(t) (+) h(t)` has both `p(t)` and `p(t-1)` as separate inputs
    # and recovers `p(t) - p(t-1)` by exact subtraction regardless of it.
    pos = np.column_stack([
        window * 100.0 + step * 3.0, window * 50.0 - step * 2.0,
    ]) + rng.normal(scale=5.0, size=(rows, 2))
    radians = (
        np.deg2rad((step * 7.0 + window * 11.0) % 360.0)
        + rng.normal(scale=0.2, size=rows)
    )
    targets = np.column_stack([pos, np.sin(radians), np.cos(radians)])
    enc = np.column_stack([pos, rng.normal(size=(rows, width - 2))])
    lagged = np.roll(pos, 1, axis=0)
    latent = np.column_stack([
        lagged, rng.normal(size=(rows, h_dim - 2)), rng.normal(size=(rows, 4)),
    ])
    return {
        "latent": latent, "embedding": enc.copy(), "encoder_embedding": enc,
        "targets": targets, "window": window, "step": step, "episode": episode,
    }


def test_exit_codes_are_39_and_40_and_do_not_collide_with_m3i():
    """M3i holds 38 (`EXIT_CONTROL_LEAKED`). A reused number makes two different
    refusals indistinguishable to a caller reading `$?`."""
    assert script.EXIT_BASE_UNRESOLVED == 39
    assert script.EXIT_MOTION_UNRESOLVED == 40
    assert script.EXIT_OK == 0


def test_phases_are_measure_read_all():
    assert script.PHASES == ("measure", "read", "all")


def test_episode_budget_is_the_filtering_gain_defaults():
    """`select_episodes` at 20, not 4. `probe.filtering_gain`'s docstring records
    that a 4-episode selection split picks the wrong ridge and FLIPS THE SIGN of
    the gain (+0.0325 against -0.0208 on the same checkpoint). Shrinking this to
    save a gather does not make the number noisier, it makes it wrong."""
    assert script.FIT_EPISODES == 20
    assert script.SELECT_EPISODES == 20


def test_record_path_names_both_the_arm_and_the_seed(tmp_path):
    """M3i's task reports collided because two milestones wrote the same
    filename; a record named by only one of (arm, seed) collides nine ways."""
    path = script.retention_record_path(tmp_path, "pixel_ae", 2)
    assert path.name == "retention_pixel_ae_seed2.json"


def test_k_key_carries_no_dot():
    """`write_record` addresses non-finite fields by dotted path, so a key with
    a '.' in it would be unaddressable."""
    assert script.k_key(15) == "k15"
    assert "." not in script.k_key(4)


def test_cell_ladder_covers_every_target_rung_and_horizon():
    """The rule is a disjunction over k across four rungs and two targets. A
    ladder missing a cell would silently make the disjunction narrower."""
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    ladder = script.cell_ladder(fit, select, score, h_dim=3, ks=(1, 2))
    assert set(ladder["ladder"]) == set(TARGETS)
    for target in TARGETS:
        assert set(ladder["ladder"][target]) == {"k1", "k2"}
        for k in ("k1", "k2"):
            assert set(ladder["ladder"][target][k]) == set(RUNGS)


def test_cell_ladder_hands_every_rung_a_byte_identical_base(monkeypatch):
    """Spec 2.1: the four gains at one `(target, k)` must be differences against
    the SAME base, or they are not comparable to each other -- only each to its
    own fit. `fit_probe` is deterministic, so identical base arrays and an
    identical target give an identical base level, which is therefore the
    property to pin is therefore that the ARRAYS are identical, which four
    independent row selections would silently break while still producing four
    plausible gains.

    Both targets share a row set at a given k (Task 4 pins that), so the base is
    byte-identical across targets too and one distinct array is the correct
    count, not two."""
    calls = []
    real = script.gain_from_blocks

    def counting(fit, select, score, **kwargs):
        calls.append(np.asarray(fit.base).tobytes())
        return real(fit, select, score, **kwargs)

    monkeypatch.setattr(script, "gain_from_blocks", counting)
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    script.cell_ladder(fit, select, score, h_dim=3, ks=(1,))
    assert len(calls) == len(TARGETS) * len(RUNGS), "one call per (target, rung)"
    assert len(set(calls)) == 1, (
        "the rungs were not handed one identical base array; their gains are "
        "against different bases and cannot be compared to each other"
    )


def test_cell_ladder_groups_the_bootstrap_on_episodes_not_windows(monkeypatch):
    """Spec 3.1. 229 non-overlapping windows cut from 24 trajectories are not 229
    independent observations, and every reading from M3e onward clusters on the
    episodes. Passing `window` here would silently narrow every interval."""
    seen = []
    real = script.gain_from_blocks

    def recording(fit, select, score, **kwargs):
        seen.append(np.asarray(kwargs["groups"]).copy())
        return real(fit, select, score, **kwargs)

    monkeypatch.setattr(script, "gain_from_blocks", recording)
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    script.cell_ladder(fit, select, score, h_dim=3, ks=(1,))
    for groups in seen:
        assert np.unique(groups).size == np.unique(score["episode"]).size, (
            "the bootstrap is grouped on windows, not episodes"
        )


def test_cell_ladder_scores_every_rung_on_the_same_rows():
    """All four rungs are gains against one shared base, which is only true if
    they were scored on one row set. The row count per k is recorded so the
    claim is checkable from the artefact."""
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    ladder = script.cell_ladder(fit, select, score, h_dim=3, ks=(1, 4))
    # 8 windows of 20 steps: k=1 drops 8 rows, k=4 drops 32.
    assert ladder["rows"] == {"k1": 152, "k4": 128}
    for target in TARGETS:
        counts = {
            ladder["ladder"][target]["k4"][rung]["n_scored_windows"] for rung in RUNGS
        }
        assert len(counts) == 1, f"{target}: rungs scored on different groups: {counts}"


def test_cell_ladder_finds_the_gain_a_lagged_latent_carries():
    """A known-answer case. The deterministic half of `latent` is the PREVIOUS
    frame's position, so `enc(t) (+) h(t)` determines `p(t) - p(t-1)` exactly
    while `enc(t)` alone cannot. If the deterministic rung shows no gain here,
    the rows, the block and the target have come apart and no null this script
    produces would mean anything."""
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    ladder = script.cell_ladder(fit, select, score, h_dim=3, ks=(1,))
    deterministic = ladder["ladder"]["translation"]["k1"]["deterministic"]
    assert deterministic["gain"] > 0.1, deterministic
    assert deterministic["ci_low"] > 0.0, deterministic


def test_cell_ladder_shows_no_gain_for_a_rung_that_adds_only_noise():
    """The other half of the known-answer pair. `z` here is pure noise, so its
    rung must not clear -- otherwise the statistic is rewarding width and the
    base is not really in both arms."""
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    ladder = script.cell_ladder(fit, select, score, h_dim=3, ks=(1,))
    stochastic = ladder["ladder"]["translation"]["k1"]["stochastic"]
    assert stochastic["ci_low"] <= 0.0, stochastic


def test_base_control_reports_the_level_not_a_gain():
    """`enc(t)` -> absolute position. There is nothing to take an increment
    over: this IS the arm the increments are measured against."""
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    control = script.base_control(fit, select, score)
    assert control["r2"] > 0.5, (
        "enc(t) carries position exactly in this fixture; a low r2 means the "
        "control is reading the wrong array"
    )
    assert "gain" not in control, "the base control is a level, not a gain"


def test_cell_ladder_refuses_a_horizon_no_window_is_long_enough_for():
    """A k with no rows is a protocol refusal, not an empty table. Reporting a
    horizon whose gain was computed on zero rows would put a number in the
    record that describes nothing."""
    with pytest.raises(ValueError, match="no window is long enough"):
        script.cell_ladder(
            _gathered(steps=6), _gathered(steps=6, seed=1), _gathered(steps=6, seed=2),
            h_dim=3, ks=(15,),
        )
