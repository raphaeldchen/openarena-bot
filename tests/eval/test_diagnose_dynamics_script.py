"""The CLI that runs both M3b diagnostics over the shipped nine cells.

The script is the last thing between nine frozen checkpoints and a sentence
about whether M4 is blocked, so these tests are arranged around what a reader
can be misled by: a number reported under the wrong arm's name, a held-out set
that is not the one the records were scored on, a probe fit at the wrong
filtering depth, a full-looking table describing a study nobody ran, and -- the
defect this project has been burned by before -- an evaluation of a model that
was never loaded from its checkpoint.

The table and verdict functions are module-level and pure, so they are driven
from fabricated results rather than from a GPU run, following
`scripts/eval_rollout.py`'s `print_report` split and its importlib-loaded test
module.
"""

import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import mbfps.eval.pooling as pooling
from mbfps.eval.diagnostics import (
    LADDER,
    LADDER_PERTURBS,
    ArmResult,
    ContrastResult,
    LadderResult,
    NoiseReference,
    RegroundingSweep,
)
from mbfps.eval.rollout import RolloutResult
from mbfps.eval.study import StudyJob, job_record_path, load_record, write_record

_SPEC = importlib.util.spec_from_file_location(
    "diagnose_dynamics_script",
    Path(__file__).resolve().parents[2] / "scripts" / "diagnose_dynamics.py",
)
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)

H = 3
KS = (1, H)
VAL_NAMES = ["ep_000001_len00051.npz", "ep_000002_len00051.npz"]


def _rollout(rssm, floor=None, persistence=None) -> RolloutResult:
    rssm = np.asarray(rssm, dtype=float)
    return RolloutResult(
        horizon=np.arange(1, rssm.size + 1),
        rssm_position=rssm,
        floor_position=np.zeros(rssm.size) if floor is None else np.asarray(floor, float),
        persistence_position=(
            np.full(rssm.size, 9.0) if persistence is None else np.asarray(persistence, float)
        ),
        rssm_angle=rssm / 10.0,
        floor_angle=np.zeros(rssm.size),
        persistence_angle=np.full(rssm.size, 4.0),
    )


def _sweep(reference: RolloutResult, k_curves: dict, spreads=None) -> RegroundingSweep:
    """Two windows per k, straddling the curve so their mean IS the curve.

    `spreads` sets each k's own half-width, so the paired ruler between two ks
    is a number the test controls rather than an artefact of the fixture -- with
    one shared half-width every pair of ks differs by a constant in every window
    and every paired standard error is exactly zero.
    """
    spreads = {} if spreads is None else spreads
    rows = lambda k, v: np.stack([  # noqa: E731
        np.asarray(v, float) - spreads.get(k, 1.0),
        np.asarray(v, float) + spreads.get(k, 1.0),
    ])
    floor = np.asarray(reference.floor_position, float)
    return RegroundingSweep(
        ks=tuple(k_curves),
        horizon=reference.rssm_position.size,
        reference=reference,
        position={k: np.asarray(v, dtype=float) for k, v in k_curves.items()},
        angle={k: np.asarray(v, dtype=float) / 10.0 for k, v in k_curves.items()},
        window_position={k: rows(k, v) for k, v in k_curves.items()},
        window_floor_position=np.stack([floor - 0.5, floor + 0.5]),
        windows_total=2,
    )


def _arm(
    name, reference: RolloutResult, curve, changed, deltas=None, episodes=None,
    steps_changed=None, multiset=None, angle_deltas=None, embedding=None, noise=None,
    embedding_curve=None, noise_curve=None,
) -> ArmResult:
    """One fabricated intervened arm against `reference`.

    `steps_changed` defaults to "every step, or none" from `changed`, which is
    what a real permutation of distinct actions produces; a test about the
    steps-moved accounting passes its own so the count and the flag are
    separable. The multiset distance defaults to the step count -- a test
    about the multiset row passes its own. The angle deltas default to the
    position deltas over ten, so the two channels are numerically distinct.
    `embedding` and `noise` are the per-window embedding-space numerator and
    ruler; absent, the arm was "not measured in embedding space", which is
    what every hand-built result carried before the reading existed. The
    per-step curves are the flat series of each window mean unless
    `embedding_curve` / `noise_curve` hand in their own -- a test about the
    per-step ratio passes both, so step 1 and step H differ.
    """
    curve = np.asarray(curve, dtype=float)
    changed = np.asarray(changed, dtype=bool)
    if deltas is None:
        deltas = np.tile(curve - reference.rssm_position, (changed.size, 1))
    deltas = np.asarray(deltas, dtype=float)
    if steps_changed is None:
        steps_changed = np.where(changed, curve.size, 0)
    steps_changed = np.asarray(steps_changed, dtype=int)
    flat = lambda series: None if series is None else np.full(curve.size, float(np.mean(series)))  # noqa: E731
    return ArmResult(
        name=name,
        position=curve,
        angle=curve / 10.0,
        window_position_delta=deltas,
        window_angle_delta=deltas / 10.0 if angle_deltas is None else np.asarray(angle_deltas, float),
        window_steps_changed=steps_changed,
        window_multiset_distance=(
            steps_changed.copy() if multiset is None else np.asarray(multiset, dtype=int)
        ),
        horizon=curve.size,
        window_episode=(None if episodes is None else np.asarray(episodes, int)),
        window_embedding_distance=(None if embedding is None else np.asarray(embedding, float)),
        embedding_distance_curve=(
            flat(embedding) if embedding_curve is None else np.asarray(embedding_curve, float)
        ),
        window_noise_distance=(None if noise is None else np.asarray(noise, float)),
        noise_distance_curve=flat(noise) if noise_curve is None else np.asarray(noise_curve, float),
    )


def _shuffle(
    reference: RolloutResult, shuffled, changed, deltas=None, episodes=None
) -> ArmResult:
    """The shuffled rung alone, for the statistics tests that need one rung."""
    return _arm("shuffled", reference, shuffled, changed, deltas, episodes)


def _contrast(
    reference: RolloutResult, held: dict, contrast=(3, 0), changed=None, steps_changed=None,
    embedding=None,
) -> ContrastResult:
    """The constant rung: `held` maps an action to its `ArmResult`, and the
    contrast's deltas are the difference of the two named ones -- exactly as
    the library builds it. Its own count is every step, since the two held
    sequences differ everywhere, unless a test says otherwise."""
    first, second = contrast
    n = held[first].windows_total
    if changed is None:
        changed = [True] * n
    changed = np.asarray(changed, dtype=bool)
    if steps_changed is None:
        steps_changed = np.where(changed, held[first].horizon, 0)
    steps_changed = np.asarray(steps_changed, dtype=int)
    return ContrastResult(
        name="constant",
        held=held,
        contrast=tuple(contrast),
        window_position_delta=(
            held[first].window_position_delta - held[second].window_position_delta
        ),
        window_angle_delta=held[first].window_angle_delta - held[second].window_angle_delta,
        window_steps_changed=steps_changed,
        window_multiset_distance=steps_changed.copy(),
        horizon=held[first].horizon,
        window_episode=held[first].window_episode,
        window_embedding_distance=(None if embedding is None else np.asarray(embedding, float)),
        embedding_distance_curve=(
            None if embedding is None
            else np.full(held[first].horizon, float(np.mean(embedding)))
        ),
        window_noise_distance=held[first].window_noise_distance,
        noise_distance_curve=held[first].noise_distance_curve,
    )


def _noise(distance, collapsed=0, restored=True) -> NoiseReference:
    distance = np.asarray(distance, float)
    return NoiseReference(
        window_embedding_distance=distance,
        embedding_distance_curve=np.full(H, float(distance.mean())),
        windows_collapsed=collapsed,
        stream_restored=restored,
    )


def _ladder(reference: RolloutResult, arms: dict, noise: NoiseReference | None = None) -> LadderResult:
    """`arms` maps a rung name to its result; the order is `LADDER`'s."""
    order = tuple(name for name in LADDER if name in arms)
    constant = arms.get("constant")
    return LadderResult(
        real=reference,
        arms={name: arms[name] for name in order},
        order=order,
        intervention_seed=0,
        held_actions=None if constant is None else tuple(sorted(constant.held)),
        contrast=None if constant is None else constant.contrast,
        action_marginal_values=np.array([1, 2, 4]),
        action_marginal_counts=np.array([7, 2, 1]),
        windows_total=next(iter(arms.values())).windows_total,
        window_episode=next(iter(arms.values())).window_episode,
        noise_reference=noise,
    )


class _TinyModel(torch.nn.Module):
    """Stands in for `WorldModel`, but carries REAL parameters.

    The stub evaluation below reads those parameters, so "the model that was
    evaluated is the model that was loaded" is a claim about VALUES rather than
    about a call having happened -- a spy on `load_state_dict` is satisfied by a
    call whose effect is thrown away.
    """

    def __init__(self, cfg=None) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(3))

    def signature(self) -> float:
        return float(self.weight.detach().sum())


TRAINED = [11.0, 13.0, 17.0]
TRAINED_SIGNATURE = float(sum(TRAINED))
PROBE_R2 = 0.31
"""The study record's `embedding_selection_r2`, as the stub writes it: a
recognisable non-round number, so the self-check column and the record are
asserted against a value that nothing else in the fixture equals."""


def _write_record(out_dir, arm, seed, curve, val_names=None, probe_r2=PROBE_R2):
    write_record(
        job_record_path(out_dir, StudyJob(arm=arm, seed=seed)),
        {
            "arm": arm,
            "seed": seed,
            "episodes": {"val": list(VAL_NAMES if val_names is None else val_names)},
            "probe": {"embedding_selection_r2": probe_r2},
            "curves": {"rssm_position": [float(v) for v in curve]},
        },
    )


STUB_STEPS = {"shuffled": (3, 1), "resampled": (2, 3), "constant": (3, 3)}
"""Each stub rung's per-window steps moved, DISTINCT per rung, so a record or
table that reported one rung's count under every rung's name is caught by
which value lands where -- the same discipline the +4/+5/+6 curve offsets
apply to the deltas."""

STUB_HELD = (3, 0, 1)
"""The actions the stub's constant rung holds: the default contrast pair plus
one more, so the held table has a row that is neither side of the contrast."""


def _stub(
    monkeypatch,
    tmp_path,
    *,
    cells=(("pixel_ae", 0),),
    checkpoint_labels=None,
    record_curve=None,
    record_names=None,
    sweep_offset=0.0,
    smallest_k_is_floor=False,
    shuffle_offset=0.0,
    changed=(True, True),
    rung_changed=None,
    rung_steps=None,
    shuffle_deltas=None,
    rung_deltas=None,
    episodes=None,
    rung_embedding=None,
    noise_distance=None,
    noise_collapsed=0,
    noise_restored=True,
    state=None,
):
    """Everything `main` touches, replaced; the checkpoint and record are real files.

    `state` collects the calls a test wants to inspect. `shuffle_deltas` sets
    the shuffled rung's per-window deltas alone; `rung_deltas` maps a rung name
    to its own, so a test can make the rungs disagree. `changed` is every
    rung's mask unless `rung_changed` gives a rung its own; `rung_steps` does
    the same for the steps moved, over `STUB_STEPS`. `episodes` is the
    per-window episode index the ladder reports, None meaning unknown.

    THE EMBEDDING-SPACE READING IS DERIVED FROM THE LOADED MODEL by default:
    every rung's and held action's per-window numerator is
    `signature + offset + w` -- the same +4/+5/+6 offsets as the curves, +6 /
    +1.5 / +0.5 for the held actions -- and the noise reference is
    `signature` in every window, so the record's new fields are VALUES that
    only the checkpoint's parameters produce. `rung_embedding` overrides a
    rung's numerator (a rung name, or `("held", action)`), and an explicit
    `None` there removes the reading -- a rung not measured in embedding space.
    `noise_distance` overrides the noise series.
    """
    state = {} if state is None else state
    state.setdefault("fit_probes", [])
    state.setdefault("split", [])
    state.setdefault("diagnostics", [])
    state.setdefault("evaluated", [])

    for arm, seed in cells:
        script.checkpoint_path(tmp_path, arm, seed).write_bytes(b"")
        _write_record(
            tmp_path, arm, seed,
            [TRAINED_SIGNATURE] * H if record_curve is None else record_curve,
            record_names,
        )

    def fake_load(path, map_location=None, weights_only=True):
        labels = checkpoint_labels
        if labels is None:
            name = Path(path).stem  # world_model_<arm>_seed<n>
            arm = name[len("world_model_") : name.rindex("_seed")]
            labels = {"arm": arm, "seed": int(name[name.rindex("_seed") + 5 :])}
        return {**labels, "state_dict": {"weight": torch.tensor(TRAINED)}}

    def fake_evaluate(model, val, probe, **kwargs):
        state["evaluated"].append(model.signature())
        return _rollout([model.signature()] * H)

    def fake_sweep(model, val, probe, ks=KS, **kwargs):
        state["diagnostics"].append(("sweep", kwargs, ks))
        reference = _rollout([model.signature()] * H)
        curves = {
            # Each k gets its OWN number, so a script that reports one k's curve
            # under every k's heading is caught by WHICH value lands where.
            k: reference.rssm_position + (sweep_offset if k == max(ks) else 1.0)
            for k in ks
        }
        if smallest_k_is_floor:
            # The smallest k alone collapses onto the floor; the largest keeps
            # its own curve, so `min(ks)` and `max(ks)` are separated.
            curves[min(ks)] = reference.floor_position
        # A DIFFERENT half-width per k, so the paired ruler between two ks is
        # not the same number as either curve's own spread. With one shared
        # half-width the two differ by a constant in every window and every
        # paired standard error is exactly zero -- which the unpaired one is
        # not, but which no assertion could then separate from a fixture
        # artefact.
        return _sweep(reference, curves, spreads={k: 1.0 / (i + 1) ** 2
                                                  for i, k in enumerate(sorted(ks))})

    def fake_ladder(model, val, probe, arms=LADDER, **kwargs):
        state["diagnostics"].append(("ladder", kwargs, tuple(arms)))
        reference = _rollout([model.signature()] * H)
        deltas = dict(rung_deltas or {})
        if shuffle_deltas is not None:
            deltas["shuffled"] = shuffle_deltas
        masks = {name: changed for name in LADDER} | dict(rung_changed or {})
        # A window the mask says is unchanged moved no steps, whatever the
        # per-rung count says -- the two must not disagree in a fixture.
        steps = {
            name: np.where(masks[name], (dict(STUB_STEPS) | dict(rung_steps or {}))[name], 0)
            for name in LADDER
        }
        # Each rung gets its OWN curve offset -- +4, +5, +6 in `LADDER` order --
        # so a table or record that reports one rung's number under every
        # rung's name is caught by WHICH value lands where, not only by a gap.
        # The rungs are built in REVERSED order so the ladder's order cannot
        # be the dict's insertion order by coincidence.
        real = _rollout(reference.rssm_position + shuffle_offset)
        windows = len(masks["shuffled"])
        signature = model.signature()
        noise = (
            [signature] * windows if noise_distance is None else list(noise_distance)
        )
        embeddings = dict(rung_embedding or {})

        def numerator(key, offset):
            if key in embeddings:
                return embeddings[key]
            return [signature + offset + w for w in range(windows)]

        rungs = {}
        for name in reversed(LADDER):
            if name not in arms:
                continue
            offset = 4.0 + LADDER.index(name)
            if name != "constant":
                rungs[name] = _arm(
                    name, real, reference.rssm_position + offset, masks[name],
                    deltas=deltas.get(name), steps_changed=steps[name], episodes=episodes,
                    embedding=numerator(name, offset), noise=noise,
                )
                continue
            # Held arms at +6 (the constant rung's own offset) for held
            # MOVE_FORWARD, +1.5 for held NOOP and +0.5 for the third, so the
            # contrast is +4.5 -- distinct from every sequence rung's number
            # and from every held arm's -- unless a test hands the contrast
            # its own deltas.
            held = {
                action: _arm(
                    name, real, reference.rssm_position + (offset, 1.5, 0.5)[i],
                    masks[name], steps_changed=steps[name], episodes=episodes,
                    embedding=numerator(("held", action), (offset, 1.5, 0.5)[i]), noise=noise,
                )
                for i, action in enumerate(STUB_HELD)
            }
            rungs[name] = _contrast(
                real, held, steps_changed=steps[name], embedding=numerator(name, offset),
            )
            if deltas.get(name) is not None:
                rungs[name].window_position_delta = np.asarray(deltas[name], float)
                rungs[name].window_angle_delta = np.asarray(deltas[name], float) / 10.0
        return _ladder(
            real, rungs, noise=_noise(noise, collapsed=noise_collapsed, restored=noise_restored),
        )

    def fake_split(paths, **kwargs):
        state["split"].append(kwargs)
        return (["t0", "t1"], [Path(name) for name in VAL_NAMES])

    def fake_fit_probes(model, paths, backbone, device, **kwargs):
        state["fit_probes"].append(kwargs)
        return ({"ridge": 1e3}, {"ridge": 1e3})

    monkeypatch.setattr(script, "get_device", lambda prefer: torch.device("cpu"))
    monkeypatch.setattr(
        script, "get_config", lambda arm, **kw: types.SimpleNamespace(encoder="E")
    )
    monkeypatch.setattr(script, "WorldModel", _TinyModel)
    monkeypatch.setattr(script.torch, "load", fake_load)
    monkeypatch.setattr(
        script, "ReplayBuffer",
        lambda data, **kw: types.SimpleNamespace(episode_paths=lambda: ["e0", "e1"]),
    )
    monkeypatch.setattr(script, "episode_split", fake_split)
    monkeypatch.setattr(script, "encoder_backbone", lambda encoder: "BACKBONE")
    monkeypatch.setattr(script, "fit_probes", fake_fit_probes)
    monkeypatch.setattr(script, "evaluate_rollout", fake_evaluate)
    monkeypatch.setattr(script, "regrounding_sweep", fake_sweep)
    monkeypatch.setattr(script, "action_intervention_ladder", fake_ladder)
    return state


def _argv(tmp_path, **extra):
    argv = [
        "--out", str(tmp_path), "--data", str(tmp_path), "--device", "cpu",
        "--horizon", str(H), "--ks", *[str(k) for k in KS],
    ]
    for flag, value in extra.items():
        argv += [f"--{flag.replace('_', '-')}", *[str(v) for v in np.atleast_1d(value)]]
    return argv


# ---------------------------------------------------------------------------
# Exit statuses.
# ---------------------------------------------------------------------------


def test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own():
    """The three study scripts run one after another in the same shell and a
    wrapper reads the status, so a collision reports one script's failure under
    another's meaning. 1 is an uncaught traceback and 2 is argparse's usage
    error, so neither may be reused either."""
    import importlib.util as util

    def statuses(name):
        spec = util.spec_from_file_location(
            f"{name}_status", Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
        )
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {
            key: value for key, value in vars(module).items() if key.startswith("EXIT_")
        }

    mine = statuses("diagnose_dynamics")
    assert len(set(mine.values())) == len(mine), mine
    assert 1 not in mine.values() and 2 not in mine.values()
    for other in ("run_study", "report_study", "pool_dynamics"):
        theirs = statuses(other)
        clash = (set(mine.values()) & set(theirs.values())) - {0}
        assert not clash, f"diagnose_dynamics collides with {other} on {clash}"


def test_a_protocol_divergence_and_a_record_mismatch_report_different_statuses(
    monkeypatch, tmp_path, capsys
):
    """They call for different actions and must never collapse onto one code.

    A sweep whose k == horizon curve differs from `evaluate_rollout` is a CODE
    defect in the re-grounding path. A record the rollout no longer reproduces,
    with every protocol check passing, points at the ENVIRONMENT -- measured,
    the shipped records reproduce bitwise on mps and miss by ~6.5 map units on
    cpu. Reporting the second as the first would make a machine difference read
    as a broken diagnostic.
    """
    _stub(monkeypatch, tmp_path, sweep_offset=0.5)
    assert script.main(_argv(tmp_path)) == script.EXIT_PROTOCOL_DIVERGED
    assert "PROTOCOL DIVERGED" in capsys.readouterr().out

    _stub(monkeypatch, tmp_path, record_curve=[TRAINED_SIGNATURE + 6.5] * H)
    assert script.main(_argv(tmp_path)) == script.EXIT_RECORD_MISMATCH
    out = capsys.readouterr().out
    assert "RECORD MISMATCH" in out
    assert "device=cpu" in out, "the environment the mismatch points at was not named"


def test_a_shuffle_whose_arms_drifted_apart_reports_its_own_status(
    monkeypatch, tmp_path, capsys
):
    """The shuffle's real arm must BE `evaluate_rollout`'s. If it is not, the
    two arms did not share a sampling stream and the delta printed above is
    noise -- which is neither a protocol divergence nor an environment
    difference."""
    _stub(monkeypatch, tmp_path, shuffle_offset=1e-3)
    assert script.main(_argv(tmp_path)) == script.EXIT_STREAM_DIVERGED
    assert "STREAM DIVERGED" in capsys.readouterr().out


def test_the_verdict_reads_the_shuffles_own_spread_and_not_the_sweeps(
    monkeypatch, tmp_path, capsys
):
    """`diagnose_cell` assembles one flat dict from three sources, and TWO of
    them have a natural claim on the name "se": the shuffle's own standard
    error of the mean delta, which the verdict decides on, and the sweep's
    spread of the open-loop error across windows, which the table prints.

    Nothing about a collision between them is visible -- both are positive
    floats of a plausible size, the table still fills, and the verdict still
    reads as a sentence. Measured on the shipped checkpoints it silently
    inflated the decision band from ~0.17 to ~18 map units and turned every
    cell into "M4 is BLOCKED" regardless of what the shuffle measured.

    So the fixture makes the two numbers far apart and requires the verdict to
    quote the SHUFFLE's: the per-window deltas here have a standard error of
    50, while the sweep's is 0.5. Run at the shuffled rung alone, so the
    ladder's decision IS that rung's. A band of 100 against the stub's
    persistence-to-floor range of 9 is wider than the range, so the honest
    verdict is that nothing could register -- and the band it quotes there is
    the shuffle's, which is the assertion.
    """
    _stub(
        monkeypatch, tmp_path,
        shuffle_deltas=np.array([[0.0] * H, [100.0] * H]),
    )
    assert script.main(_argv(tmp_path) + ["--rungs", "shuffled"]) == script.EXIT_OK
    out = capsys.readouterr().out
    verdict = out[out.index("--- verdicts ---"):]
    assert "+-2 SE (100.000)" in verdict, verdict
    assert "widest null band of 100.000" in verdict, (
        "the verdict decided on a spread that is not the shuffle's"
    )
    assert "M4 is BLOCKED" not in verdict and "RESPONDS" not in verdict


def test_a_clean_run_exits_ok(monkeypatch, tmp_path, capsys):
    _stub(monkeypatch, tmp_path)
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    out = capsys.readouterr().out
    assert "self-checks" in out and "verdicts" in out


def test_a_directory_with_no_diagnosable_cell_exits_rather_than_printing_nothing(
    monkeypatch, tmp_path, capsys
):
    """An empty table would read as "nine cells, no findings"."""
    _stub(monkeypatch, tmp_path, cells=())
    assert script.main(_argv(tmp_path)) == script.EXIT_NO_CHECKPOINTS
    assert "nothing to diagnose" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Wiring: the split, the probe, the checkpoint.
# ---------------------------------------------------------------------------


def test_the_script_checks_the_val_episode_names_against_the_record(
    monkeypatch, tmp_path, capsys
):
    """The record carries the held-out episodes BY NAME for exactly this check.
    Without it, a diagnostic scored on a different split from the record's would
    be compared against that record's curve and the difference blamed on the
    code."""
    _stub(monkeypatch, tmp_path)
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    capsys.readouterr()

    _stub(monkeypatch, tmp_path, record_names=["someone_elses_episode.npz"])
    assert script.main(_argv(tmp_path)) == script.EXIT_SPLIT_MISMATCH
    out = capsys.readouterr().out
    assert "SPLIT MISMATCH" in out
    assert "someone_elses_episode.npz" in out, "the differing names were not printed"


def test_the_script_splits_at_the_shared_val_fraction_and_at_split_seed_zero(
    monkeypatch, tmp_path
):
    """Two couplings, one test.

    Asserting `val_fraction == 0.2` would pass again the moment someone
    re-hardcodes the literal, so the shared constant is MOVED to a value no
    literal in this file equals and the call site is required to follow it. And
    the split seed is `study.SPLIT_SEED`, fixed at 0 and deliberately NOT the
    cell's seed -- splitting at the cell seed would diagnose each model on
    episodes it trained on.
    """
    import mbfps.data.split as split
    import mbfps.eval.study as study

    assert script.VAL_FRACTION is split.VAL_FRACTION
    assert script.SPLIT_SEED is study.SPLIT_SEED

    state = _stub(monkeypatch, tmp_path, cells=(("pixel_ae", 2),))
    monkeypatch.setattr(script, "VAL_FRACTION", 0.375)
    script.main(_argv(tmp_path) + ["--seeds", "2"])

    assert state["split"], "episode_split was never called"
    assert state["split"][0]["val_fraction"] == pytest.approx(0.375)
    assert state["split"][0]["seed"] == 0, "the split followed the cell's seed"


def test_the_probe_and_both_diagnostics_run_at_the_parsed_context_and_horizon(
    monkeypatch, tmp_path
):
    """Three different values -- context 2, horizon 3, seed 7 -- so an exchanged
    pair is visible; a fixture where two of them coincide makes the assertion
    blind to a swap.

    The probe is applied to latents filtered from a zero state for exactly
    `context` frames, and fitting it at a different depth is a distribution
    mismatch worth ~25 map units, so the probe and both diagnostics must be
    given the same three numbers rather than being allowed to default apart.
    """
    state = _stub(monkeypatch, tmp_path, cells=(("pixel_ae", 7),))
    script.main(
        ["--out", str(tmp_path), "--data", str(tmp_path), "--device", "cpu",
         "--context", "2", "--horizon", "3", "--ks", "1", "3", "--seeds", "7"]
    )
    assert state["fit_probes"] == [{"context": 2, "horizon": 3, "seed": 7}]
    assert len(state["diagnostics"]) == 2
    for _, kwargs, _ in state["diagnostics"]:
        assert kwargs["context"] == 2
        assert kwargs["horizon"] == 3
        assert kwargs["seed"] == 7


@pytest.mark.parametrize("flag", ["--intervention-seed", "--permutation-seed"])
def test_the_intervention_seed_reaches_the_ladder_under_either_flag(
    monkeypatch, tmp_path, flag
):
    """One draw per rung is one draw; which draw it was has to be controllable,
    or a null distribution cannot be built from repeated runs. `--permutation-
    seed` is the name the nine shipped records were produced under and it seeds
    the shuffled rung IDENTICALLY, so it stays accepted."""
    state = _stub(monkeypatch, tmp_path)
    script.main(_argv(tmp_path) + [flag, "3"])
    ladders = [kwargs for kind, kwargs, _ in state["diagnostics"] if kind == "ladder"]
    assert ladders and ladders[0]["intervention_seed"] == 3


def test_the_rungs_flag_selects_the_ladder_and_defaults_to_all_three(
    monkeypatch, tmp_path
):
    """The default is the WHOLE ladder. Defaulting to the shuffled rung alone
    would leave the two rungs that close the multiset blind spot off unless a
    flag was typed, and the shipped conclusion would be the old one under a
    new name. The selection is passed through in `LADDER`'s order whatever
    order it was typed in -- the library refuses nothing here, so the script
    has to hand it the order the table will print."""
    state = _stub(monkeypatch, tmp_path)
    script.main(_argv(tmp_path))
    assert [arms for kind, _, arms in state["diagnostics"] if kind == "ladder"] == [
        LADDER
    ]

    state = _stub(monkeypatch, tmp_path)
    script.main(_argv(tmp_path) + ["--rungs", "constant", "shuffled"])
    assert [arms for kind, _, arms in state["diagnostics"] if kind == "ladder"] == [
        ("shuffled", "constant")
    ]


def test_the_default_out_is_the_m3c_study_directory():
    """`runs/m3_study` is M3b's directory, and this script OVERWRITES
    `diagnostic_*.json` in `--out`; the default must be the directory the M3c
    driver writes, and a revert would point nine diagnostics at M3b's
    checkpoints, whose `cnn` arm is not even in `ARMS`."""
    assert script.parse_args([]).out == Path("runs/m3_study_v2")
    assert script.parse_args([]).out != Path("runs/m3_study")


def test_a_rung_the_ladder_does_not_have_is_an_argparse_usage_error(capsys):
    """Refused as a bad FLAG (argparse's status 2) rather than as the
    ValueError the library raises twenty seconds into the probe refit, which
    would surface as the uncaught-traceback status 1."""
    with pytest.raises(SystemExit) as exit_info:
        script.parse_args(["--rungs", "nonesuch"])
    assert exit_info.value.code == 2
    assert "nonesuch" in capsys.readouterr().err


def test_a_ks_that_omits_the_horizon_is_an_argparse_usage_error(capsys):
    """The k == horizon pass IS self-check 1, so a sweep without it is refused
    -- and refused as a bad FLAG COMBINATION (argparse's status 2) rather than
    as one of this script's findings, or as the uncaught traceback that letting
    `regrounding_sweep` raise from inside `main` would produce."""
    with pytest.raises(SystemExit) as exit_info:
        script.parse_args(["--horizon", "45", "--ks", "1", "3"])
    assert exit_info.value.code == 2
    assert "omits --horizon 45" in capsys.readouterr().err


def test_a_checkpoint_wrong_in_the_arm_alone_is_refused(monkeypatch, tmp_path, capsys):
    """Split from the seed case deliberately: a fixture wrong in both exercises
    only whichever half is checked first."""
    _stub(
        monkeypatch, tmp_path, cells=(("frozen_ssl", 0),),
        checkpoint_labels={"arm": "pixel_ae", "seed": 0},
    )
    assert script.main(_argv(tmp_path) + ["--arms", "frozen_ssl"]) == (
        script.EXIT_MISLABELLED_CHECKPOINT
    )
    out = capsys.readouterr().out
    assert "arm='pixel_ae'" in out, "the checkpoint's OWN labels were not reported"


def test_a_checkpoint_wrong_in_the_seed_alone_is_refused(monkeypatch, tmp_path, capsys):
    """The architecture is shared across seeds, so a seed-wrong checkpoint loads
    perfectly cleanly and nothing else in the run would raise."""
    _stub(
        monkeypatch, tmp_path, cells=(("pixel_ae", 1),),
        checkpoint_labels={"arm": "pixel_ae", "seed": 0},
    )
    assert script.main(_argv(tmp_path) + ["--seeds", "1"]) == (
        script.EXIT_MISLABELLED_CHECKPOINT
    )
    assert "seed=0" in capsys.readouterr().out


def test_the_evaluated_model_is_the_one_loaded_from_the_checkpoint(
    monkeypatch, tmp_path
):
    """This project's worst defect class: nothing in 524 tests once required the
    EVALUATED model to be the TRAINED one, and deleting `load_state_dict` passed
    the entire suite.

    Asserted on VALUES, not on a call: the stub evaluation reads the model's own
    parameters, and the record carries the checkpoint's signature. A freshly
    initialised model scores something else and the run comes back
    EXIT_RECORD_MISMATCH. The freshly-initialised signature is asserted to
    differ first, so the test cannot pass because the two coincide.
    """
    assert _TinyModel().signature() != TRAINED_SIGNATURE
    state = _stub(monkeypatch, tmp_path)
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    assert state["evaluated"] == [TRAINED_SIGNATURE]


def test_the_checkpoint_path_carries_both_the_arm_and_the_seed(tmp_path):
    """Either one missing collides two of the nine cells onto one file."""
    assert (
        script.checkpoint_path(tmp_path, "random_vit", 2).name
        == "world_model_random_vit_seed2.pt"
    )
    names = {
        script.checkpoint_path(tmp_path, arm, seed).name
        for arm in ("pixel_ae", "random_vit")
        for seed in (0, 1, 2)
    }
    assert len(names) == 6


# ---------------------------------------------------------------------------
# The report itself.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The report itself.
# ---------------------------------------------------------------------------


FAMILY = 1
"""The family most verdict tests decide under: ONE comparison, whose
Bonferroni threshold is the nominal 1.96, so "outside +-2 SE" and "responds"
coincide and each test is about the ladder logic and not the correction. The
correction has its own tests below, at families where the two rulers differ.
"""


def _channel(delta, band, mean=None, se=0.5, outside=0):
    """One channel's block of a fabricated rung: the aggregate the verdict
    decides on and the per-step curves it reports beside that."""
    delta = np.asarray(delta, float)
    mean = (float(np.nanmean(delta)) if delta.size and not np.isnan(delta).all()
            else float("nan")) if mean is None else mean
    return {
        "delta": delta, "band": np.asarray(band, float),
        "mean": mean,
        "se": se, "se_independent": se,
        "delta_final": float(delta[-1]) if delta.size else float("nan"),
        "steps_outside": outside, "expected_outside": 0.05 * delta.size,
        # Per window over ALL windows; `_rung` sizes it to the cell's total.
        "window_mean": None,
    }


def _rung(
    delta, band, changed=4, total=4, mean=None, se=0.5, outside=0, episodes=4,
    steps_changed_mean=None, steps_changed_min=None, multiset=None, angle=None,
    contrast=None, held=None, embedding=None,
):
    """One rung's block of a fabricated cell: the counts that say whether the
    statistics mean anything, then one channel block each for position and
    angle. The angle channel is a clean null unless the test hands one in, so
    a position-only test is not accidentally decided on the angle.

    The counts default to values that DIFFER from one another and from the
    step count, so a row that printed one under another's name is not
    satisfied by the fixture: 4 changed of 4 total, 2.5 steps moved of `len(
    delta)`, minimum 1, multiset 0.5. `embedding` is the rung's embedding
    block as `rung_block` builds it, None for a rung not measured there.
    """
    delta = np.asarray(delta, float)
    block = {
        "windows_total": total, "windows_changed": changed, "episodes": episodes,
        "steps": int(delta.size),
        "steps_changed_mean": (
            (2.5 if changed else 0.0) if steps_changed_mean is None else steps_changed_mean
        ),
        "steps_changed_min": (
            (1 if changed else 0) if steps_changed_min is None else steps_changed_min
        ),
        "multiset_distance_mean": (
            (0.5 if changed else 0.0) if multiset is None else multiset
        ),
        "position": _channel(delta, band, mean, se, outside),
        "angle": (
            _channel(np.zeros(delta.size), np.full(delta.size, 1.0), mean=0.0, se=1.0)
            if angle is None else angle
        ),
    }
    # The per-window series the record persists: every window carries the
    # mean, and the changed ones come first -- sized to `total`.
    block["window_steps_changed"] = np.array([2] * changed + [0] * (total - changed))
    for name in ("position", "angle"):
        if block[name]["window_mean"] is None:
            block[name]["window_mean"] = np.full(total, block[name]["mean"])
    block["embedding"] = embedding
    if contrast is not None:
        block["contrast"] = tuple(contrast)
        block["held"] = {} if held is None else held
    return block


def _ladder_cell(
    rungs: dict, open_loop=0.0, record=0.0, stream=0.0, smallest_k_is_floor=False,
    floor=1.0, persistence=9.0, total=4, probe_r2=PROBE_R2,
):
    """A fabricated cell over several rungs. `rungs` maps a name to `_rung(...)`,
    and its INSERTION order is deliberately not consulted for `rung_order`: the
    order is `LADDER`'s, which is what the report has to print in. A
    `constant` rung without a contrast is given the default one."""
    rungs = dict(rungs)
    if "constant" in rungs and "contrast" not in rungs["constant"]:
        rungs["constant"] = rungs["constant"] | {"contrast": (3, 0), "held": {}}
    return {
        "arm": "pixel_ae", "seed": 0, "split_names": list(VAL_NAMES), "curves": {},
        "action_marginal": None,
        "open_loop": open_loop, "record": record, "stream": stream,
        "smallest_k_is_floor": smallest_k_is_floor,
        "probe_selection_r2": probe_r2,
        "windows_total": total,
        "rung_order": tuple(name for name in LADDER if name in rungs),
        "rungs": rungs,
        "held_actions": (3, 0) if "constant" in rungs else None,
        "contrast": rungs["constant"]["contrast"] if "constant" in rungs else None,
        "k": {1: 5.0, 3: 7.0}, "floor": floor, "persistence": persistence,
        "spread_final": 0.25,
        "paired": {"1v3": 0.125},
        "floor_margin": 4.0, "floor_margin_se": 0.75, "smallest_k": 1,
        "window_episode": None,
        "noise_reference": None, "noise_restored": True, "noise_collapsed": 0,
    }


def _cell(
    delta, band, changed=4, total=4, mean=None, se=0.5, outside=0,
    open_loop=0.0, record=0.0, stream=0.0, smallest_k_is_floor=False,
    floor=1.0, persistence=9.0, episodes=4, rung="shuffled",
):
    """A fabricated cell with ONE rung. `mean`/`se` are the AGGREGATE the
    verdict decides on; `delta`/`band` are the per-step curves it reports
    beside that.

    The three self-check numbers are PARAMETERS rather than a shared 0.0. Left
    equal they make every column of `selfcheck_table` interchangeable, so
    printing one where another belongs is satisfied by the fixture and not by
    the code -- the L1 coincidence this repo names. They default to 0.0 because
    that is the clean-run rendering most tests want; the column test passes
    three different ones.
    """
    return _ladder_cell(
        {rung: _rung(delta, band, changed, total, mean, se, outside, episodes)},
        open_loop=open_loop, record=record, stream=stream,
        smallest_k_is_floor=smallest_k_is_floor, floor=floor, persistence=persistence,
        total=total,
    )


def _null_ladder(**overrides):
    """All three rungs inside their bands, each with its own numbers -- the
    clean null the BLOCKED headline needs, with a contrast in it."""
    rungs = {
        "shuffled": _rung([0.0] * 4, [0.2] * 4, mean=0.01, se=0.10),
        "resampled": _rung([0.0] * 4, [0.2] * 4, mean=0.02, se=0.20),
        "constant": _rung([0.0] * 4, [0.2] * 4, mean=0.03, se=0.30),
    }
    return _ladder_cell(rungs, **overrides)


def _verdict(cell, family=FAMILY, arm="pixel_ae", seed=0):
    return script.verdict_block(arm, seed, cell, family=family)


def _line(text: str, name: str) -> str:
    return next(line for line in text.splitlines() if line.strip().startswith(name))


ROW_SHAPED = {
    "selfcheck_table": lambda cells: script.selfcheck_table(cells, ["pixel_ae"], [0, 1, 2], 1),
    "sweep_table": lambda cells: script.sweep_table(cells, ["pixel_ae"], [0, 1, 2], (1, 3)),
    "sweep_ruler_table": lambda cells: script.sweep_ruler_table(
        cells, ["pixel_ae"], [0, 1, 2], (1, 3)
    ),
}


@pytest.mark.parametrize("table", sorted(ROW_SHAPED))
def test_the_row_shaped_tables_print_a_row_per_cell_and_MISSING_for_a_gap(table):
    """A cell with no checkpoint gets its own row saying so, rather than being
    omitted -- an omitted row makes an incomplete study look complete."""
    cells = {
        ("pixel_ae", 0): _cell([1.0], [0.5]),
        ("pixel_ae", 2): _cell([2.0], [0.5]) | {"floor": 42.0, "k": {1: 42.0, 3: 42.0}},
    }
    rows = {line.split()[1]: line for line in ROW_SHAPED[table](cells).splitlines()[1:]}
    assert set(rows) == {"0", "1", "2"}, rows
    assert "MISSING" in rows["1"]
    assert "MISSING" not in rows["0"] and "MISSING" not in rows["2"]
    if table == "sweep_table":
        assert "42.000" in rows["2"], rows["2"]
        assert "42.000" not in rows["1"], "seed 2's number landed in seed 1's row"


LADDER_WIDTH, LADDER_PREFIX = 18, 12 + 11 + 22
"""`ladder_table`'s geometry: arm, rung and row labels, then 18 per seed."""
ROWS_PER_RUNG = len(script.LADDER_ROWS)


def _seed_columns(row: str) -> list[str]:
    return [
        row[LADDER_PREFIX + i * LADDER_WIDTH : LADDER_PREFIX + (i + 1) * LADDER_WIDTH]
        for i in range(3)
    ]


def _rung_rows(table: str, index: int) -> dict[str, str]:
    """The block of rows for the `index`-th rung, keyed by row label."""
    body = table.splitlines()[1:]
    rows = body[ROWS_PER_RUNG * index : ROWS_PER_RUNG * (index + 1)]
    return {row[23:45].strip(): row for row in rows}


def test_the_ladder_table_puts_every_seeds_number_in_its_own_column():
    """`report_study.py`'s first rule, on the one table with per-seed COLUMNS.

    With seed 1 absent, rendering the cells that DID run by list position puts
    seed 2's number in seed 1's column and the table looks complete. The two
    present seeds carry different numbers, so a positional renderer is caught by
    WHICH value lands where, not only by the gap.
    """
    cells = {
        # The horizon-MEAN and the FINAL-step delta are deliberately different
        # numbers per cell. Measured on the real checkpoints they differ by a
        # factor of four, and with one fixture value for both, a row that
        # printed the mean under "final step" -- or the reverse -- is satisfied
        # by the fixture rather than by the code.
        ("pixel_ae", 0): _cell([1.0, 5.0], [0.5, 0.5], mean=1.0),
        ("pixel_ae", 2): _cell([2.0, 6.0], [0.5, 0.5], mean=2.0, changed=3, total=5),
    }
    table = script.ladder_table(cells, ["pixel_ae"], [0, 1, 2], ("shuffled",))
    header = table.splitlines()[0]
    rows = _rung_rows(table, 0)
    columns_of = lambda row: [c.strip() for c in _seed_columns(row)]  # noqa: E731
    assert columns_of(header) == ["seed 0", "seed 1", "seed 2"], header
    assert "+1.000" in _seed_columns(rows["horizon-mean"])[0]
    assert columns_of(rows["horizon-mean"])[1] == "MISSING"
    assert "+2.000" in _seed_columns(rows["horizon-mean"])[2], (
        "seed 2's delta is not in seed 2's column"
    )
    assert columns_of(rows["final step"])[0] == "+5.000", rows["final step"]
    assert columns_of(rows["final step"])[1] == "MISSING"
    assert columns_of(rows["final step"])[2] == "+6.000", "seed 2's endpoint is in the wrong column"
    assert "4/4" in columns_of(rows["changed/total"])[0].replace(" ", "")
    assert columns_of(rows["changed/total"])[1] == "MISSING"
    assert "3/5" in columns_of(rows["changed/total"])[2].replace(" ", "")
    # The steps-moved row is the rung's DISTANCE from the real sequence, with
    # the least any window moved beside the mean -- what separates "changed"
    # from "changed by one step out of forty-five".
    assert "2.5(min1)/2" in columns_of(rows["steps moved/horizon"])[0].replace(" ", ""), rows
    assert columns_of(rows["steps moved/horizon"])[1] == "MISSING"


def test_the_ladder_table_prints_the_rungs_in_increasing_perturbation_order():
    """The ladder's monotonicity is only readable if the rungs are printed in
    `LADDER`'s order under one arm, each with its OWN rows -- and the fixture
    inserts them backwards and gives each a different number in every row, so
    a renderer that took the cell's insertion order, or printed one rung's
    numbers under every rung's name, is caught by which value lands where.
    Every count row is asserted by value: the multiset row is the one that
    reads 0 / 15 / 45 up the shipped ladder, and the minimum is the one that
    would show a held action leaving a window one step from where it was.
    """
    rungs = {
        "constant": _rung([3.0, 3.0], [0.5, 0.5], mean=3.0, changed=4, total=4,
                          steps_changed_mean=2.0, steps_changed_min=2, multiset=2.0,
                          angle=_channel([0.3, 0.3], [0.05, 0.05], mean=0.3, se=0.025)),
        "resampled": _rung([2.0, 2.0], [0.5, 0.5], mean=2.0, changed=3, total=4,
                           steps_changed_mean=1.5, steps_changed_min=1, multiset=1.0,
                           angle=_channel([0.2, 0.2], [0.05, 0.05], mean=0.2, se=0.025)),
        "shuffled": _rung([1.0, 1.0], [0.5, 0.5], mean=1.0, changed=2, total=4,
                          steps_changed_mean=1.0, steps_changed_min=0, multiset=0.0,
                          angle=_channel([0.1, 0.1], [0.05, 0.05], mean=0.1, se=0.025)),
    }
    cells = {("pixel_ae", 0): _ladder_cell(rungs)}
    table = script.ladder_table(cells, ["pixel_ae"], [0], LADDER)
    lines = table.splitlines()
    assert len(lines) == 1 + ROWS_PER_RUNG * len(LADDER), lines
    labelled = [line for line in lines[1:] if line[12:23].strip()]
    assert [line[12:23].strip() for line in labelled] == list(LADDER), labelled
    for index, name in enumerate(LADDER):
        rows = _rung_rows(table, index)
        assert list(rows) == list(script.LADDER_ROWS), rows
        assert rows["horizon-mean"][12:23].strip() == name
        assert f"+{index + 1}.000" in rows["horizon-mean"], (name, rows)
        # The angle row is degrees, its own number, not the position row's.
        assert f"+0.{index + 1}00 +-0.05" in rows["angle horizon-mean"], (name, rows)
        assert f"+{index + 1}.000" in rows["final step"]
        assert f"{index + 2}/4" in rows["changed/total"].replace(" ", ""), (name, rows)
        moved = rows["steps moved/horizon"].replace(" ", "")
        assert f"{1.0 + 0.5 * index:.1f}(min{index})/2" in moved, (name, moved)
        assert f"{index:.1f}/2" in rows["multiset dist/horizon"].replace(" ", ""), (name, rows)


def test_the_ladder_table_prints_MISSING_for_a_rung_a_cell_did_not_run():
    """A cell that ran fewer rungs than the table has rows for -- a partial
    re-run under `--rungs` beside full cells -- must say so per rung rather
    than being padded with another rung's numbers or dropped."""
    cells = {
        ("pixel_ae", 0): _ladder_cell({"shuffled": _rung([1.0], [0.5], mean=1.0)}),
    }
    lines = script.ladder_table(cells, ["pixel_ae"], [0], LADDER).splitlines()
    body = lines[1:]
    assert "+1.000" in body[0] and "MISSING" not in body[0]
    for line in body[ROWS_PER_RUNG:]:
        assert "MISSING" in line, line
        assert "1.000" not in line, "the shuffled rung's number was padded into another rung"


def test_the_held_table_prints_every_held_action_against_the_real_sequence():
    """The contrast is a difference of two held rows, and the held rows are
    what let a reader see WHICH actions move the imagined position and in
    which direction -- measured, held MOVE_FORWARD reads +31 where the
    single-held-action rung read -0.7. Each action is a named row with its
    own delta, band and steps moved (with the minimum), in action order, and
    the table is empty when the constant rung did not run.
    """
    held = {
        3: _rung([31.0] * 2, [12.0] * 2, mean=31.0, se=6.0, steps_changed_mean=41.0,
                 steps_changed_min=22),
        0: _rung([-11.6] * 2, [7.9] * 2, mean=-11.6, se=3.95, steps_changed_mean=38.0,
                 steps_changed_min=7),
    }
    cells = {("pixel_ae", 0): _ladder_cell({
        "constant": _rung([42.6] * 2, [14.0] * 2, mean=42.6, se=7.0, contrast=(3, 0), held=held),
    })}
    table = script.held_table(cells, ["pixel_ae"], [0], LADDER)
    lines = table.splitlines()
    assert len(lines) == 1 + 4 * len(held), lines
    noop, forward = lines[1:5], lines[5:9]
    assert noop[0][12:25].strip() == "NOOP" and forward[0][12:25].strip() == "MOVE_FORWARD"
    assert "-11.600 +-7.90" in noop[0] and "+31.000 +-12.00" in forward[0], lines
    assert "38.0(min 7)/2" in noop[1] and "41.0(min 22)/2" in forward[1], lines
    assert "42.6" not in table, "the contrast's own number was printed as a held row"
    assert script.held_table(cells, ["pixel_ae"], [0], ("shuffled", "resampled")) == ""


def test_the_verdict_names_m4_only_when_the_aggregate_delta_is_inside_two_se():
    """Both branches, in one test: either alone leaves the other unexercised.

    The decision is the AGGREGATE -- the mean per-window delta against its own
    standard error -- and never the per-step exceedance count. Two standard
    errors is ~95% per step, so under the null about 2 of 45 horizon steps fall
    outside their band by chance, and a rule that fired on one of them would
    call almost every cell responsive. The count is still printed, next to what
    chance alone gives, so a reader can see an effect the aggregate cancels.
    """
    inside = _verdict(_null_ladder() | {"rungs": _null_ladder()["rungs"] | {
        "shuffled": _rung([0.4, -0.4, 0.0] * 15, [0.2] * 45, mean=0.05, se=0.1, outside=3),
    }})
    assert "M4 is BLOCKED" in inside
    assert "NOT blocked" not in inside
    assert "3/45" in inside and "2.2 expected by chance" in inside, (
        "the per-step count and its chance expectation were not printed beside "
        "the aggregate"
    )

    outside = _verdict(
        _cell([0.4, -0.4, 0.0] * 15, [0.2] * 45, mean=9.0, se=0.1, outside=3)
    )
    assert "M4 is NOT blocked" in outside
    assert "M4 is BLOCKED" not in outside
    assert "+9.000" in outside, "the magnitude of the effect was not reported"
    # The "largest per-step move" is the peak of the DELTA, not of the band --
    # the band is the only other array of the right shape in reach, and
    # reporting its width as the size of the measured effect is the sentence
    # that decides whether M4 is blocked. The fixture separates them: the
    # delta peaks at 0.400, the band is a flat 0.200.
    for text in (inside, outside):
        assert "0.400" in text, text
        assert "was 0.200" not in text, "the band's width was reported as the effect"


def test_a_ladder_that_is_null_at_every_rung_blocks_m4_and_names_every_rung():
    """The clean null: nothing moved at any rung including the held-action
    contrast, on a probe that could have registered it, so M4 is blocked on
    this evidence -- and the verdict has to say WHICH rungs it rests on, each
    with its own numbers, or a null on one rung reads as a null on three. The
    equivalence sentence bounds the perturbations RUN, by name, and says so.
    """
    text = _verdict(_null_ladder())
    assert "M4 is BLOCKED" in text
    assert "NOT blocked" not in text
    assert "NOT USING THE ACTION" not in text, "the retracted headline is back"
    for name, two_se in (("shuffled", "0.200"), ("resampled", "0.400"), ("constant", "0.600")):
        line = _line(text, name)
        assert f"+-2 SE ({two_se})" in line, (name, line)
        assert LADDER_PERTURBS[name] in line, "the rung's perturbation was not named"
    assert "shuffled, resampled, constant" in text, text
    assert "bounds those perturbations and no other" in text, text
    assert "held MOVE_FORWARD minus held NOOP" in _line(text, "constant")


def test_a_null_ladder_without_the_contrast_does_not_license_blocking_m4():
    """A null on order and counts alone is exactly what an action-conditioned
    prior that reads the action's identity produces -- measured, the
    per-action contrast found +42.6 map units on a cell where the shuffled and
    resampled rungs were both null. So the BLOCKED headline requires the
    contrast among the rungs, and without it the verdict says what was and
    was not tested."""
    rungs = {
        "shuffled": _rung([0.0] * 4, [0.2] * 4, mean=0.01, se=0.10),
        "resampled": _rung([0.0] * 4, [0.2] * 4, mean=0.02, se=0.20),
    }
    text = _verdict(_ladder_cell(rungs))
    assert "NO RESPONSE TO THE PERTURBATIONS RUN" in text
    assert "M4 is BLOCKED" not in text and "NOT blocked" not in text
    assert "contrast was NOT among the rungs" in text
    assert "does not license blocking M4" in text
    # The single-rung fixture every older test used is this case too.
    alone = _verdict(_cell([0.0] * 4, [0.2] * 4, mean=0.01, se=0.2))
    assert "M4 is BLOCKED" not in alone and "does not license blocking M4" in alone


def test_a_ladder_where_a_later_rung_resolves_reports_action_conditioned_but_insensitive():
    """THE better finding, reported exactly and not smoothed into the null story.

    The shuffled rung is null and the two above it resolve with the sign
    conditioning predicts: the model is action-conditioned but insensitive to
    what the shuffled rung perturbs -- the ORDER -- and that is what the
    verdict must say. It must not say the prior ignores the action, and it
    must not present the responding rungs as if the whole ladder had
    responded.
    """
    rungs = {
        "shuffled": _rung([0.0] * 4, [0.2] * 4, mean=0.01, se=0.10),
        "resampled": _rung([5.0] * 4, [0.2] * 4, mean=5.0, se=0.20),
        "constant": _rung([9.0] * 4, [0.2] * 4, mean=9.0, se=0.30),
    }
    text = _verdict(_ladder_cell(rungs))
    assert "M4 is NOT blocked" in text
    assert "M4 is BLOCKED" not in text
    assert "DOES NOT MEASURABLY USE" not in text and "NOT USING THE ACTION" not in text
    assert "RESPONDS TO THE ACTION" in text
    assert "resampled, constant" in text and "not at shuffled" in text, text
    assert "action-conditioned but insensitive to " + LADDER_PERTURBS["shuffled"] in text, text
    assert "NON-MONOTONE" not in text
    assert "WRONG-SIGNED" not in text


def test_a_ladder_that_resolves_below_a_null_rung_is_flagged_as_non_monotone():
    """The rungs are in strictly increasing order of perturbation, so a lower
    rung resolving while a higher one is null is not a ladder reading at all.
    It must be flagged rather than folded into either finding -- and M4 is
    still not blocked, because something did respond with the right sign.
    Nothing is below the response, so the "insensitive to" clause must be
    ABSENT: a verdict that is simultaneously NON-MONOTONE and "insensitive
    to the counts" is two findings that contradict each other."""
    rungs = {
        "shuffled": _rung([5.0] * 4, [0.2] * 4, mean=5.0, se=0.10),
        "resampled": _rung([0.0] * 4, [0.2] * 4, mean=0.01, se=0.20),
        "constant": _rung([0.0] * 4, [0.2] * 4, mean=0.02, se=0.30),
    }
    text = _verdict(_ladder_cell(rungs))
    assert "NON-MONOTONE" in text
    assert "resampled, constant" in text, "the null rungs above the response were not named"
    assert "action-conditioned but insensitive" not in text, text
    assert "M4 is NOT blocked" in text and "M4 is BLOCKED" not in text


def test_a_sandwich_ladder_names_only_the_middle_rung_as_non_monotone():
    """Responds at shuffled and constant, null at resampled -- a pattern one
    seed can easily produce. The lowest response is the SHUFFLED rung, so
    nothing sits below it and the middle rung sits above it: NON-MONOTONE
    names `resampled` alone and no "insensitive to" clause is printed. A
    decision that took the HIGHEST response as its reference would file
    `resampled` below it and print "action-conditioned but insensitive to
    the counts" -- the exact smoothing the ladder forbids."""
    rungs = {
        "shuffled": _rung([5.0] * 4, [0.2] * 4, mean=5.0, se=0.10),
        "resampled": _rung([0.0] * 4, [0.2] * 4, mean=0.01, se=0.20),
        "constant": _rung([9.0] * 4, [0.2] * 4, mean=9.0, se=0.30),
    }
    text = _verdict(_ladder_cell(rungs))
    verdict = text.splitlines()[-1]
    assert "NON-MONOTONE: resampled perturb" in verdict, verdict
    assert "action-conditioned but insensitive" not in verdict, verdict
    assert "at rung(s) shuffled, constant" in verdict and "not at resampled" in verdict
    assert "M4 is NOT blocked" in verdict and "M4 is BLOCKED" not in verdict


def test_a_wrong_signed_response_unblocks_nothing_and_is_named_as_such():
    """The sign is read. A response whose intervened sequence imagined a
    SMALLER error than the real one is not conditioning an actor can use --
    measured, the single-held-action rung was negative in 9 of 9 cells and
    the one cell that cleared 2 SE did so at -10.6 -- so it neither blocks
    nor unblocks M4, and the verdict says so where the sign is. A positive
    response at the same magnitude does unblock, in the same test, so the
    assertion is about the sign and not the magnitude."""
    wrong = _verdict(_ladder_cell({
        "shuffled": _rung([0.0] * 4, [0.2] * 4, mean=0.01, se=0.10),
        "constant": _rung([-9.0] * 4, [0.2] * 4, mean=-9.0, se=0.30),
    }))
    assert "WRONG-SIGNED" in wrong and "constant" in wrong
    assert "not conditioning an actor can use" in wrong
    assert "neither blocked nor unblocked" in wrong
    assert "M4 is NOT blocked" not in wrong and "M4 is BLOCKED" not in wrong
    assert "sign on position - (BETTER under the intervention)" in _line(wrong, "constant")

    right = _verdict(_ladder_cell({
        "shuffled": _rung([0.0] * 4, [0.2] * 4, mean=0.01, se=0.10),
        "constant": _rung([9.0] * 4, [0.2] * 4, mean=9.0, se=0.30),
    }))
    assert "M4 is NOT blocked" in right and "WRONG-SIGNED" not in right
    assert "has the sign conditioning predicts" in right
    assert "sign on position + (worse under the intervention)" in _line(right, "constant")


def test_a_response_on_the_angle_channel_alone_is_a_response():
    """38% of the actions turn rather than move, and the angle probe is the
    one channel any shipped model beats persistence on -- a prior whose
    action pathway drives heading passes every position-only rung. So the
    verdict reads both channels: a rung null on position and clear on angle
    responds, the line names the channel, and the record and table carry the
    angle statistic under its own name."""
    rungs = {
        "shuffled": _rung([0.0] * 4, [0.2] * 4, mean=0.01, se=0.10),
        "constant": _rung(
            [0.0] * 4, [0.2] * 4, mean=0.02, se=0.30,
            angle=_channel([6.3] * 4, [1.0] * 4, mean=6.3, se=0.5),
        ),
    }
    text = _verdict(_ladder_cell(rungs))
    assert "RESPONDS TO THE ACTION at rung(s) constant" in text
    line = _line(text, "constant")
    assert "on angle" in line and "angle +6.300 deg" in line, line
    assert "sign on angle +" in line
    assert "M4 is NOT blocked" in text


def test_family_threshold_is_bonferroni_over_the_whole_run():
    """z for two-sided 5% over the family: 1.96 at one comparison, 3.31 at
    the shipped run's 54 (3 rungs x 9 cells x 2 channels), monotone in the
    family, and refused at zero."""
    assert script.family_threshold(1) == pytest.approx(1.95996, abs=1e-4)
    assert script.family_threshold(54) == pytest.approx(3.3121, abs=1e-3)
    assert script.family_threshold(27) < script.family_threshold(54)
    with pytest.raises(ValueError):
        script.family_threshold(0)


def test_an_excursion_inside_the_family_wise_threshold_is_inconclusive_not_a_response():
    """One rejection at 2 SE in a family of 54 is what 1.35 expected false
    positives look like -- measured, that is exactly the frozen_ssl/1
    "RESPONDS at constant" (p = 0.007, Holm rejects nothing) the uncorrected
    verdict turned into "M4 NOT blocked". So a rung with z=2.5 responds at a
    family of one and is NOMINAL at a family of 54: the verdict then reads
    INCONCLUSIVE, blocks nothing, unblocks nothing and makes no equivalence
    claim. The threshold and the family size are printed beside it."""
    rungs = {
        "shuffled": _rung([0.0] * 4, [0.2] * 4, mean=0.01, se=0.10),
        "resampled": _rung([0.0] * 4, [0.2] * 4, mean=0.02, se=0.20),
        "constant": _rung([2.5] * 4, [0.2] * 4, mean=2.5, se=1.0),
    }
    alone = _verdict(_ladder_cell(rungs), family=1)
    assert "RESPONDS TO THE ACTION" in alone and "M4 is NOT blocked" in alone

    corrected = _verdict(_ladder_cell(rungs), family=54)
    assert "INCONCLUSIVE" in corrected, corrected
    assert "M4 is BLOCKED" not in corrected and "M4 is NOT blocked" not in corrected
    assert "RESPONDS TO THE ACTION" not in corrected
    assert "bounds those perturbations" not in corrected, "an equivalence bound was claimed"
    assert "z=3.31 for 54 comparisons" in corrected
    assert "2.70 such excursions are expected" in corrected
    line = _line(corrected, "constant")
    assert "outside the +-2 SE band on position but inside the family-wise" in line
    assert "z=2.50" in line


def test_a_nominal_excursion_below_a_family_wise_response_is_named_but_not_a_response():
    """With a response above it, a nominal excursion is neither a null nor a
    second response: it is named as nominal, it counts as "did not respond"
    for the ladder's shape, and the "insensitive to" clause still covers it.
    """
    rungs = {
        "shuffled": _rung([0.0] * 4, [0.2] * 4, mean=0.01, se=0.10),
        "resampled": _rung([2.5] * 4, [0.2] * 4, mean=2.5, se=1.0),
        "constant": _rung([9.0] * 4, [0.2] * 4, mean=9.0, se=0.30),
    }
    text = _verdict(_ladder_cell(rungs), family=54)
    verdict = text.splitlines()[-1]
    assert "at rung(s) constant" in verdict and "not at shuffled, resampled" in verdict
    assert "nominal excursion only at resampled" in verdict, verdict
    assert "insensitive to " + LADDER_PERTURBS["shuffled"] in verdict
    assert LADDER_PERTURBS["resampled"] in verdict
    assert "M4 is NOT blocked" in verdict


def test_an_uninterpretable_rung_is_named_and_excluded_from_the_ladder_decision():
    """A rung that changed no window -- or one, so its spread is undefined --
    says nothing about the dynamics, so it is neither a null nor a response.
    It is reported as UNINTERPRETABLE with its count, and the decision is
    taken over the rungs that did intervene. Both routes to uninterpretable
    are in one ladder so neither can pass by the other's guard."""
    rungs = {
        "shuffled": _rung([np.nan] * 4, [np.nan] * 4, changed=0, total=7),
        "resampled": _rung([1.0] * 4, [np.inf] * 4, changed=1, total=7,
                           mean=float("nan"), se=float("inf"),
                           angle=_channel([np.nan] * 4, [np.inf] * 4, mean=float("nan"),
                                          se=float("inf"))),
        "constant": _rung([0.0] * 4, [0.2] * 4, mean=0.01, se=0.10, changed=7, total=7),
    }
    text = _verdict(_ladder_cell(rungs))
    shuffled, resampled = _line(text, "shuffled"), _line(text, "resampled")
    assert "UNINTERPRETABLE" in shuffled and "0/7" in shuffled, shuffled
    assert "UNINTERPRETABLE" in resampled and "1/7" in resampled, resampled
    assert "nan" not in text.lower(), text
    assert "M4 is BLOCKED" in text
    # And the rung the decision rests on is the ONLY one the sentence names.
    verdict = text.splitlines()[-1]
    assert "constant" in verdict and "shuffled" not in verdict and "resampled" not in verdict, verdict


def test_a_ladder_with_no_interpretable_rung_gets_no_verdict_at_all():
    rungs = {
        "shuffled": _rung([np.nan] * 3, [np.nan] * 3, changed=0, total=7),
        "constant": _rung([np.nan] * 3, [np.nan] * 3, changed=0, total=7),
    }
    text = _verdict(_ladder_cell(rungs), seed=1)
    assert "UNINTERPRETABLE" in text
    assert "M4" not in text
    assert "No verdict" in text


def test_the_equivalence_bound_is_quoted_at_the_widest_null_band():
    """Accepting the null over a ladder needs ONE bound, and the honest one is
    the widest band among the rungs the null rests on: an effect smaller than
    the widest band could have hidden at that rung. Quoting the narrowest
    would claim the tightest rung's precision for the whole ladder."""
    rungs = {
        "shuffled": _rung([0.0] * 4, [0.2] * 4, mean=0.01, se=0.10),
        "constant": _rung([0.0] * 4, [0.2] * 4, mean=0.02, se=1.00),
    }
    text = _verdict(_ladder_cell(rungs, floor=1.0, persistence=21.0))
    # 2*se = 2.0 against a range of 20.0 -> 10.0%, not the shuffled rung's 1.0%.
    assert "10.0%" in text, text
    assert "1.0%" not in text, "the narrowest rung's band was quoted for the ladder"


def test_the_verdict_line_prints_the_steps_moved_and_the_windows_changed_by_value():
    """The reader-facing half of "a no-op cannot be mistaken for a null": the
    line carries how many windows changed AND how many steps moved per window
    with the minimum, and the multiset distance. The counts are chosen so no
    two of them, nor the step count, nor the integer part of the mean,
    coincide -- 3 of 7 windows, 1.5 of 4 steps, minimum 0, multiset 0.75 --
    so a clause that printed one under another's name cannot pass."""
    text = _verdict(_ladder_cell({
        "shuffled": _rung([0.0] * 4, [0.2] * 4, mean=0.01, se=0.10, changed=3, total=7,
                          steps_changed_mean=1.5, steps_changed_min=0, multiset=0.75),
        "constant": _rung([0.0] * 4, [0.2] * 4, mean=0.01, se=0.10),
    }))
    line = _line(text, "shuffled")
    assert "3/7 windows changed" in line, line
    assert "1.5/4 steps moved per window (min 0)" in line, line
    assert "multiset distance 0.8/4" in line, line


def test_a_single_step_outside_its_band_is_not_by_itself_a_verdict():
    """The 45-comparison error, pinned. A cell whose aggregate sits inside two
    standard errors must still read as BLOCKED even when several individual
    steps exceed their own per-step bands -- which is what chance produces at
    this horizon."""
    cell = _null_ladder()
    cell["rungs"]["shuffled"] = _rung([0.0] * 45, [0.2] * 45, mean=0.01, se=0.2, outside=4)
    text = _verdict(cell, seed=1)
    assert "M4 is BLOCKED" in text
    assert "4/45" in text


def test_the_aggregate_delta_is_a_paired_window_level_statistic():
    """One comparison over WINDOWS, not forty-five over horizon steps.

    Each changed window contributes its own horizon-mean delta and the spread is
    measured across those, which is the variation the effect has to clear.
    Averaging over the horizon axis first and calling the result's spread the
    standard error would give the same MEAN and a different, much smaller
    spread, so the standard error is asserted as well as the mean.
    """
    reference = _rollout([1.0] * H)
    # Deliberately NOT square and not symmetric: with an n x n matrix whose row
    # and column means have the same spread, averaging the wrong axis first
    # gives the same standard error and the mutation survives. Here the two
    # axes differ in length AND in spread, and the MEAN is identical either way
    # -- which is why the standard error is what carries this assertion.
    deltas = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
        [0.0, 10.0, 20.0],
        [5.0, 5.0, 5.0],
    ])
    result = _shuffle(reference, [2.0] * H, [True] * 4, deltas=deltas)

    summary = script.delta_summary(result)
    per_window = deltas.mean(axis=1)
    assert summary["mean"] == pytest.approx(per_window.mean())
    assert summary["se"] == pytest.approx(per_window.std(ddof=1) / np.sqrt(4))
    assert summary["se"] != pytest.approx(
        deltas.mean(axis=0).std(ddof=1) / np.sqrt(H)
    ), "the horizon axis was averaged first; the spread is over WINDOWS"
    assert summary["expected_outside"] == pytest.approx(0.05 * H)


def test_the_angle_summary_is_the_same_statistic_over_the_angle_deltas():
    """The angle channel is `delta_summary` with the other array, and the
    fixture gives the two channels deltas that are not a scalar multiple of
    each other -- so a summary that read the position rows under the angle
    name lands on the wrong mean AND the wrong spread."""
    reference = _rollout([1.0] * H)
    position = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
    angle = np.array([[0.0, 6.0, 0.0], [0.0, 0.0, 12.0]])
    result = _shuffle(reference, [2.0] * H, [True, True], deltas=position)
    result.window_angle_delta = angle

    got = script.delta_summary(result, "angle")
    per_window = angle.mean(axis=1)
    assert got["mean"] == pytest.approx(per_window.mean()) == pytest.approx(3.0)
    assert got["se"] == pytest.approx(per_window.std(ddof=1) / np.sqrt(2))
    assert got["mean"] != pytest.approx(script.delta_summary(result, "position")["mean"])
    np.testing.assert_allclose(got["delta"], angle.mean(axis=0))
    with pytest.raises(KeyError, match="channel"):
        script.delta_summary(result, "heading")


def test_the_aggregate_delta_skips_the_windows_the_permutation_left_alone():
    """An unchanged window contributes an exact zero, which drags the aggregate
    toward "no difference" -- the very conclusion this statistic decides."""
    reference = _rollout([1.0] * H)
    deltas = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])
    result = _shuffle(reference, [2.0] * H, [True, False, True], deltas=deltas)
    assert script.delta_summary(result)["mean"] == pytest.approx(
        deltas[[0, 2]].mean(axis=1).mean()
    )


def test_a_cell_with_no_changed_windows_gets_no_verdict_at_all():
    """An all-NaN delta must not fall through to "no difference". A permutation
    of an already-uniform action sequence is a no-op, and nothing about the
    dynamics follows from it."""
    text = _verdict(_cell([np.nan] * 3, [np.nan] * 3, changed=0, total=7), seed=1)
    assert "UNINTERPRETABLE" in text
    assert "0/7" in text
    assert "M4" not in text


def test_the_band_is_infinite_below_two_changed_windows():
    """A spread cannot be estimated from one sample. An infinite band makes a
    single-window delta read as "not measurably different", which is the honest
    reading; a zero band would make any delta at all look significant."""
    reference = _rollout([1.0] * H)
    one = _shuffle(reference, [2.0] * H, changed=[True, False])
    assert np.isinf(script.delta_band(one)).all()
    many = _shuffle(reference, [2.0] * H, changed=[True, True])
    assert np.isfinite(script.delta_band(many)).all()


def test_each_self_check_lands_in_its_own_column():
    """The three numbers are DIFFERENT here, so a swapped column is caught.

    Left all-zero -- the natural clean-cell fixture -- every column of this
    block is interchangeable and printing `open_loop` where `stream` belongs
    passes. The block is the one the module docstring says invalidates
    everything below it, and its three statuses are deliberately distinct
    because they call for different actions: reporting a stream divergence as a
    protocol divergence sends the reader to the wrong code. The probe's
    selection R^2 is the last column, and it is the study record's own number.
    """
    cells = {("pixel_ae", 0): _cell([1.0], [0.5], open_loop=1e-1, record=2e-2, stream=3e-3)}
    header, row = script.selfcheck_table(cells, ["pixel_ae"], [0], 1).splitlines()
    assert header.split() == [
        "arm", "seed", "open_loop_k", "record_repro", "stream_drift", "k1_is_floor", "probe_r2",
        "noise_restored", "noise_same",
    ], header
    # Sliced by the header's own column offsets, so the assertion is about
    # WHERE each number is printed and not merely that it appears somewhere.
    columns = {}
    for name in ("open_loop_k", "record_repro", "stream_drift"):
        end = header.index(name) + len(name)
        columns[name] = row[end - 14 : end].strip()
    assert columns["open_loop_k"] == "1.000e-01", columns
    assert columns["record_repro"] == "2.000e-02", columns
    assert columns["stream_drift"] == "3.000e-03", columns
    assert row.split()[-3] == f"{PROBE_R2:.3f}", row


def test_the_self_check_column_names_the_k_the_alarm_was_computed_at():
    """The alarm is computed at the SMALLEST k, which is 1 only by default.

    `--ks 5 45` makes it a statement about k=5, and self-check 2's whole point
    is the prior/posterior separation at ONE step -- at k=5 it is a different,
    weaker claim. The header therefore carries the k rather than a hardcoded
    "k1", and it agrees with the record's `smallest_k_is_bitwise_the_floor`.
    """
    cells = {("pixel_ae", 0): _cell([1.0], [0.5])}
    header = script.selfcheck_table(cells, ["pixel_ae"], [0], 5).splitlines()[0]
    assert "k5_is_floor" in header, header
    assert "k1_is_floor" not in header, header


def test_the_verdict_is_refused_when_the_spread_cannot_be_estimated():
    """ONE changed window is not zero changed windows, and it used to fall
    straight through to an affirmative claim.

    `delta_summary` returns `mean=nan, se=inf` below two changed windows.
    `windows_changed == 1` skips the count guard, and `abs(nan) <= 2*inf` is
    False -- so control reached the POSITIVE branch and the script printed "The
    imagination does respond to the action: ... by +nan map units against +-2
    SE (inf) ... M4 is NOT blocked on this evidence." That is the exact
    opposite of what `delta_band`'s docstring promises an infinite band means,
    and it is the one case the +inf exists to protect.

    So the gate is on the STATISTIC being defined, not on the count.
    """
    cell = _cell([1.0, 2.0, 3.0], [np.inf] * 3, changed=1, total=229,
                 mean=float("nan"), se=float("inf"))
    cell["rungs"]["shuffled"]["angle"] = _channel(
        [np.nan] * 3, [np.inf] * 3, mean=float("nan"), se=float("inf")
    )
    text = _verdict(cell, seed=1)
    assert "UNINTERPRETABLE" in text
    assert "1/229" in text
    assert "M4" not in text, text
    assert "nan" not in text.lower(), text


def test_the_verdict_scales_the_effect_against_the_actionable_band():
    """Accepting a null needs a scale, or the same sentence covers evidence
    that differs by orders of magnitude.

    Measured across the nine shipped cells the band the verdict is read against
    spans 0.014 to 3.45 map units, i.e. 1.6% to 8.4% of the persistence-to-floor
    band on the feature arms. "The prior does not use the action" is the same
    words for all of them; "no effect of these perturbations larger than X% of
    the actionable range is detectable" is not.
    """
    text = _verdict(_null_ladder(floor=1.0, persistence=21.0))
    assert "M4 is BLOCKED" in text
    # The widest null band is the constant rung's 2*0.3 = 0.6 against 20.0 -> 3.0%.
    assert "3.0%" in text, text
    assert "20.000" in text, "the band the effect was scaled against was not named"
    assert "an effect of " + LADDER_PERTURBS["shuffled"] in text, (
        "the bound does not name the perturbations it bounds"
    )


@pytest.mark.parametrize(
    "floor,persistence,reason",
    [
        (9.0, 8.91, "the floor sits above persistence, so the range is negative"),
        (1.0, 1.5, "the range is 0.5 and the widest null band is 0.6: over 100%"),
    ],
)
def test_a_null_through_a_probe_that_cannot_register_an_effect_is_unmeasurable(
    floor, persistence, reason
):
    """On every M3b `cnn` cell the floor EXCEEDED persistence at the final horizon
    step: the position probe was a constant predictor (selection R^2 -0.036;
    the M3c `pixel_ae` arm's probe scores +0.305 and is not this case) and no
    action effect of any size could register through it -- the FWD/NOOP
    contrast reads +0.6 there with the right sign and nothing to scale it. A
    null through such a probe used to print "NOT USING THE ACTION ... M4 is
    BLOCKED" beside a note that there was no range to scale it against. It
    must read UNMEASURABLE THROUGH THIS PROBE, quote the probe's R^2, and
    contain neither headline. The second case is the same defect at a
    positive range the widest band exceeds."""
    text = _verdict(_null_ladder(floor=floor, persistence=persistence, probe_r2=-0.036))
    assert "UNMEASURABLE THROUGH THIS PROBE" in text, (reason, text)
    assert "M4 is BLOCKED" not in text and "NOT blocked" not in text, reason
    assert "NOT USING" not in text and "DOES NOT MEASURABLY USE" not in text, reason
    assert "%" not in text, "a percentage was printed against a range the band exceeds"
    assert "-0.036" in text, "the probe's own R^2 was not quoted"
    assert "licenses no claim about M4" in text


def test_a_response_is_still_reported_through_an_unmeasurable_probe():
    """The probe gate is on the NULL headline only. A response that reached
    the ladder despite a degenerate position probe -- on the angle channel,
    say -- is a response, and must not be swallowed by the probe caveat."""
    cell = _null_ladder(floor=9.0, persistence=8.91)
    cell["rungs"]["constant"] = _rung(
        [0.0] * 4, [0.2] * 4, mean=0.02, se=0.30,
        angle=_channel([6.3] * 4, [1.0] * 4, mean=6.3, se=0.5),
    )
    text = _verdict(cell)
    assert "RESPONDS TO THE ACTION at rung(s) constant" in text
    assert "UNMEASURABLE" not in text


def test_the_verdict_carries_a_qualifier_when_the_shipped_record_did_not_reproduce():
    """The RECORD MISMATCH notice is printed BELOW every verdict, so a reader
    who stops at the verdict never sees it.

    Measured: on CPU the record reproduction for frozen_ssl/seed0 is 12.4 map
    units and the headline delta itself moves from +0.995 to +1.872 -- the
    number the verdict interprets is not stable across devices at the precision
    it is being interpreted at. So the verdict says so where it is read.
    """
    text = _verdict(_cell([0.0] * 4, [0.2] * 4, record=12.4))
    assert "1.240e+01" in text, text
    clean = _verdict(_cell([0.0] * 4, [0.2] * 4, record=0.0))
    assert "1.240e+01" not in clean and "does not reproduce" not in clean


def test_the_aggregate_delta_is_clustered_by_the_episode_each_window_came_from():
    """229 windows are not 229 independent draws; they come from 24 episodes.

    Dividing by sqrt(229) claims a precision the data does not support --
    measured, up to 1.32x too narrow on the shipped cells -- and that interval
    is exactly what the equivalence statement in the verdict rests on. So the
    ruler is the cluster-robust standard error when the labels are known, and
    the naive one is kept beside it under its own name.

    The fixture makes the two disagree loudly: within each episode the deltas
    are identical and between episodes they are far apart, which is the case
    where treating windows as independent understates the spread most.
    """
    reference = _rollout([1.0] * H)
    deltas = np.array([[0.0] * H, [0.0] * H, [40.0] * H, [40.0] * H])
    clustered = script.delta_summary(
        _shuffle(reference, [2.0] * H, [True] * 4, deltas=deltas, episodes=[0, 0, 1, 1])
    )
    naive = script.delta_summary(
        _shuffle(reference, [2.0] * H, [True] * 4, deltas=deltas)
    )
    assert clustered["mean"] == pytest.approx(20.0)
    assert clustered["se_independent"] == pytest.approx(naive["se"])
    # Two clusters of two identical values: every within-cluster residual is
    # +-20, so the clustered SE is exactly 20 while the naive one is 11.55.
    assert clustered["se"] == pytest.approx(20.0)
    assert clustered["se"] > clustered["se_independent"], (
        "the clustered ruler collapsed onto the independent one"
    )
    # The cluster count is the rung's, beside its other counts.
    with_labels = script.rung_block(
        _shuffle(reference, [2.0] * H, [True] * 4, deltas=deltas, episodes=[0, 0, 1, 1])
    )
    assert with_labels["episodes"] == 2
    assert script.rung_block(_shuffle(reference, [2.0] * H, [True] * 4, deltas=deltas))["episodes"] == 0, (
        "unknown clustering must not be reported as any"
    )


def test_the_bands_spread_is_the_sample_standard_deviation():
    """`ddof=1`, pinned by value the way `delta_summary`'s already is.

    Its sibling statistic has its `ddof` pinned exactly and this one had only
    an inf-versus-finite check, so the two were inconsistently guarded. With
    two changed windows `ddof=0` and `ddof=1` differ by sqrt(2).
    """
    reference = _rollout([1.0] * H)
    rows = np.array([[1.0, 2.0, 3.0], [5.0, 9.0, 4.0]])
    result = _shuffle(reference, [2.0] * H, [True, True], deltas=rows)
    np.testing.assert_allclose(
        script.delta_band(result),
        2.0 * rows.std(axis=0, ddof=1) / np.sqrt(2),
    )


def test_the_rung_block_of_a_contrast_carries_every_held_action_and_the_pair():
    """`rung_block` is what both the record and the tables read, so the
    contrast's block must carry the pair and one full block per held action
    -- each with its OWN counts and channels, not the contrast's. The held
    arms are given distinct step counts so a block that copied the contrast's
    count into every held action is caught by value."""
    reference = _rollout([1.0] * H)
    held = {
        3: _arm("constant", reference, [4.0] * H, [True, True], steps_changed=[3, 2]),
        0: _arm("constant", reference, [2.0] * H, [True, True], steps_changed=[1, 3]),
    }
    block = script.rung_block(_contrast(reference, held))
    assert block["contrast"] == (3, 0)
    assert set(block["held"]) == {3, 0}
    assert block["position"]["mean"] == pytest.approx(2.0)      # (4 - 1) - (2 - 1)
    assert block["held"][3]["position"]["mean"] == pytest.approx(3.0)
    assert block["held"][0]["position"]["mean"] == pytest.approx(1.0)
    assert block["steps_changed_mean"] == pytest.approx(3.0)     # every step
    assert block["held"][3]["steps_changed_mean"] == pytest.approx(2.5)
    assert block["held"][0]["steps_changed_min"] == 1
    assert "held" not in block["held"][3], "a held block carries no held blocks of its own"


def test_the_cross_cell_table_counts_excursions_and_prints_every_cells_sign():
    """The pattern a per-cell verdict cannot see. Three cells, one rung: the
    position deltas are +0.01 (null), +2.5 at se 1 (outside 2 SE, inside the
    family-wise threshold) and -9 at se 0.3 (clears it), so the row reads 3
    cells, 2 outside, 1 clear, signs `++-` in cell order -- and the header
    carries the family, its threshold and the chance expectation. The
    angle row is asserted separately with its own numbers, so a table that
    read the position channel for both is caught: two of its cells are the
    fixture's EXACT-zero angle null, which prints as `0` -- the sign an
    action-blind prior produces bitwise, and neither + nor -."""
    cells = {
        ("pixel_ae", 0): _ladder_cell({"shuffled": _rung([0.0] * 4, [0.2] * 4, mean=0.01, se=0.10)}),
        ("pixel_ae", 1): _ladder_cell({"shuffled": _rung(
            [2.5] * 4, [0.2] * 4, mean=2.5, se=1.0,
            angle=_channel([-0.5] * 4, [0.2] * 4, mean=-0.5, se=0.1),
        )}),
        ("pixel_ae", 2): _ladder_cell({"shuffled": _rung([-9.0] * 4, [0.2] * 4, mean=-9.0, se=0.3)}),
    }
    text = script.cross_cell_table(cells, ["pixel_ae"], [0, 1, 2], ("shuffled",), family=54)
    header, columns, position, angle = text.splitlines()
    assert "family: 54 comparisons (1 rungs x 3 cells x 2 channels)" in header
    assert "z=3.31" in header and "0.15 excursions" in header, header
    # The last column is the exact two-sided sign test over the cells whose
    # sign is defined: `++-` is 2 of 3, p = 1.0; the angle row has one
    # signed cell, p = 1.0 as well -- the unanimous case is pinned below.
    assert position.split() == ["shuffled", "position", "3", "2", "1", "++-", "1.000"], position
    assert angle.split() == ["shuffled", "angle", "3", "1", "1", "0-0", "1.000"], angle


def test_the_cross_cell_sign_test_is_the_exact_binomial_over_the_signed_cells():
    """Nine cells all positive is a pattern no per-cell z can see -- measured,
    the constant rung's contrast is positive in 9 of 9 -- and its exact
    two-sided sign-test probability under a symmetric null is 2 * 0.5^9 =
    0.0039. Pinned by value against the closed form, with the caveat the
    header carries: the cells share windows, so this is a pattern statistic
    and not an independent replication."""
    assert script.sign_test_p(9, 9) == pytest.approx(2 * 0.5**9)
    assert script.sign_test_p(5, 9) == pytest.approx(1.0)
    assert script.sign_test_p(8, 9) == pytest.approx(2 * (1 + 9) * 0.5**9)
    assert script.sign_test_p(0, 0) == pytest.approx(1.0)
    cells = {
        ("pixel_ae", seed): _ladder_cell({"constant": _rung([1.0] * 4, [0.2] * 4, mean=1.0, se=1.0)})
        for seed in (0, 1, 2)
    }
    text = script.cross_cell_table(cells, ["pixel_ae"], [0, 1, 2], ("constant",), family=6)
    header, columns, position, angle = text.splitlines()
    assert "share windows" in header, header
    assert position.split() == ["constant", "position", "3", "0", "0", "+++", "0.250"], position


def test_the_sweep_ruler_table_prints_the_paired_bar_for_each_adjacent_pair():
    """The k columns are means over the SAME windows, so what a difference
    between two of them has to clear is the PAIRED spread.

    The unpaired between-window spread was printed under a docstring calling it
    "the scale a difference between k columns has to clear"; measured, it
    overstates the bar for an adjacent-k difference by 1.7x to 3.9x and hides
    real separations. The paired bar is printed per adjacent pair, plus the
    smallest k's margin over the floor, which is self-check 2's quantitative
    half.
    """
    cells = {("pixel_ae", 0): _cell([1.0], [0.5])}
    header, row = script.sweep_ruler_table(cells, ["pixel_ae"], [0], (1, 3)).splitlines()
    assert "1v3" in header, header
    assert "k1-floor" in header, header
    assert "0.125" in row, row
    assert "+4.000" in row and "0.750" in row, row


def test_the_sweep_table_labels_its_spread_column_as_one_curves_own():
    """The column is the open-loop curve's between-window spread and nothing
    else; it was documented as the bar for a k-to-k difference, which it is
    not. The heading and the docstring have to agree with the arithmetic."""
    cells = {("pixel_ae", 0): _cell([1.0], [0.5])}
    header = script.sweep_table(cells, ["pixel_ae"], [0], (1, 3)).splitlines()[0]
    assert "spread" in header, header
    assert "se" not in header.split(), header
    assert "paired" in script.sweep_table.__doc__, (
        "the column's docstring still has to point the reader at the ruler that "
        "IS the bar for a between-k difference"
    )


def test_the_sweep_table_reports_each_ks_own_final_step_number(
    monkeypatch, tmp_path, capsys
):
    """Diagnostic 2's headline table, assembled from a real `main` run.

    `sweep_table` is otherwise only ever driven from a fabricated `"k"` dict,
    so rendering is pinned and ASSEMBLY is not: replacing the comprehension
    with `sweep.curve(max(args.ks))[-1]` for every k -- the sweep reporting
    that re-grounding does nothing -- passed the whole suite. The stub gives
    each k a recognisably different curve, so the mutation is caught by WHICH
    number lands under which heading.
    """
    _stub(monkeypatch, tmp_path)
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    out = capsys.readouterr().out
    block = out[out.index("re-grounding sweep") :]
    header, row = block.splitlines()[1], block.splitlines()[2]
    column = lambda name: row[  # noqa: E731
        header.index(name) + len(name) - 11 : header.index(name) + len(name)
    ].strip()
    # k=1 carries the open-loop number + 1.0, k=3 (== the horizon) carries it.
    assert column("k=1") == f"{TRAINED_SIGNATURE + 1.0:.3f}", block
    assert column("k=3") == f"{TRAINED_SIGNATURE:.3f}", block
    assert column("k=1") != column("k=3"), "every k column printed the same curve"


def test_the_floor_alarm_is_computed_at_the_smallest_k_and_reaches_the_record(
    monkeypatch, tmp_path, capsys
):
    """Self-check 2's only carrier in the CLI, in the state it actually fires.

    Two failures were possible at once and neither was covered: `min(args.ks)`
    could be `max(args.ks)` -- at k == horizon the curve IS the open loop,
    which can never be the floor, so the alarm would read False forever and
    the check would be silently off in the table AND in all nine records -- and
    the True path was never rendered anywhere, because every stub produced False.

    The stub collapses the SMALLEST k onto the floor and leaves the largest
    alone, so `min` and `max` are separated by the value that comes back.
    """
    _stub(monkeypatch, tmp_path, smallest_k_is_floor=True)
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    out = capsys.readouterr().out
    row = out[out.index("self-checks") :].splitlines()[2]
    assert row.split()[-2] == "True", row
    written = load_record(script.diagnostic_record_path(tmp_path, "pixel_ae", 0))
    assert written["self_checks"]["smallest_k_is_bitwise_the_floor"] is True
    assert written["self_checks"]["smallest_k"] == 1


def test_the_written_record_carries_its_self_checks_and_a_curve_for_every_k(
    monkeypatch, tmp_path
):
    """The record is the durable artifact -- nine files a later reader trusts
    because the environment they are reproducible in is recorded beside them --
    and two whole blocks of it were unasserted: replacing `self_checks` with
    `{}` passed, and so did `curves`.
    """
    _stub(monkeypatch, tmp_path)
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    written = load_record(script.diagnostic_record_path(tmp_path, "pixel_ae", 0))

    assert written["self_checks"] == {
        "open_loop_divergence": 0.0,
        "record_reproduction": 0.0,
        "stream_drift": 0.0,
        "smallest_k": 1,
        "smallest_k_is_bitwise_the_floor": False,
        "noise_reference_stream_restored": True,
        "noise_reference_windows_collapsed": 0,
    }
    # The probe the ladder was read through, from the study record.
    assert written["probe"] == {"embedding_selection_r2": PROBE_R2}
    expected = {"reference_position", "shuffled_position", "resampled_position",
                "constant_held3_position", "constant_held0_position", "constant_held1_position",
                "floor_position", "persistence_position", "k1_position", "k3_position"}
    assert set(written["curves"]) == expected, written["curves"].keys()
    assert all(len(curve) == H for curve in written["curves"].values())
    # Distinct per k, so an assembly that reported one k's curve under every k's
    # name is caught here as well as in the table.
    assert written["curves"]["k1_position"] != written["curves"]["k3_position"]
    assert written["curves"]["k3_position"] == written["curves"]["reference_position"]
    # And distinct per ARM: the stub offsets each rung's and each held action's
    # curve by its own amount, so one curve written under every name is caught.
    names = ("shuffled", "resampled", "constant_held3", "constant_held0", "constant_held1")
    arm_curves = [tuple(written["curves"][f"{name}_position"]) for name in names]
    assert len(set(arm_curves)) == len(names), arm_curves
    assert written["curves"]["shuffled_position"] == [TRAINED_SIGNATURE + 4.0] * H
    assert written["curves"]["constant_held3_position"] == [TRAINED_SIGNATURE + 6.0] * H
    assert written["curves"]["constant_held0_position"] == [TRAINED_SIGNATURE + 1.5] * H


def test_the_written_record_carries_every_rung_with_its_own_counts_and_the_choices_made(
    monkeypatch, tmp_path
):
    """The record is what a later reader diffs the nine shipped cells against,
    so every rung is in it under its own name with its OWN counts and delta,
    in the ladder's order -- and the CHOICES the numbers cannot be recovered
    from are recorded beside them: the seed every rung was derived from, the
    actions the constant rung held and the pair it contrasted, and the family
    the verdict was corrected over.

    The rungs are given deltas AND changed masks AND step counts that
    disagree, so a record that wrote one rung's block -- or one rung's counts
    -- under three names is caught by the values, not the keys. The shuffled
    rung changes two windows of three; the other two change all three.
    """
    _stub(
        monkeypatch, tmp_path,
        changed=(True, True, True),
        rung_deltas={
            "shuffled": np.array([[1.0] * H] * 3),
            "resampled": np.array([[2.0] * H] * 3),
            "constant": np.array([[3.0] * H] * 3),
        },
        rung_changed={"shuffled": (True, True, False)},
        rung_steps={"shuffled": (2, 1, 0), "resampled": (1, 2, 3), "constant": (3, 3, 3)},
    )
    assert script.main(_argv(tmp_path) + ["--intervention-seed", "5"]) == script.EXIT_OK
    written = load_record(script.diagnostic_record_path(tmp_path, "pixel_ae", 0))

    assert written["ladder"]["rungs"] == list(LADDER)
    assert written["ladder"]["intervention_seed"] == 5
    assert written["ladder"]["held_actions"] == sorted(STUB_HELD)
    assert written["ladder"]["contrast"] == [3, 0]
    assert written["ladder"]["action_marginal"] == {"values": [1, 2, 4], "counts": [7, 2, 1]}
    # One cell, three rungs, two channels.
    assert written["ladder"]["family"] == 6
    assert written["ladder"]["family_threshold_z"] == pytest.approx(script.family_threshold(6))
    assert written["windows"] == {"total": 3, "episode": None}
    assert list(written["interventions"]) == list(LADDER)
    expected_counts = {
        "shuffled": (2, 1.0, 0), "resampled": (3, 2.0, 1), "constant": (3, 3.0, 3),
    }
    for index, name in enumerate(LADDER):
        block = written["interventions"][name]
        assert block["position"]["delta_mean"] == pytest.approx(index + 1.0), name
        assert block["position"]["delta"] == [index + 1.0] * H, name
        assert block["angle"]["delta_mean"] == pytest.approx((index + 1.0) / 10.0), name
        changed, mean, minimum = expected_counts[name]
        assert block["windows_changed"] == changed, name
        assert block["steps_changed_mean"] == pytest.approx(mean), name
        assert block["steps_changed_min"] == minimum, name
        assert block["episodes_changed"] == 0, name
        assert "delta_se" in block["position"] and "band" in block["angle"], name
    # The contrast's held actions, each with its OWN block, under the pair.
    constant = written["interventions"]["constant"]
    assert constant["contrast"] == [3, 0]
    assert set(constant["held"]) == {str(a) for a in STUB_HELD}
    assert constant["held"]["3"]["position"]["delta_mean"] == pytest.approx(6.0)
    assert constant["held"]["0"]["position"]["delta_mean"] == pytest.approx(1.5)
    assert constant["held"]["3"]["steps_changed_mean"] == pytest.approx(3.0)
    # No `shuffle` block under the old name: a record carrying the shuffled
    # rung twice invites a reader to diff the wrong copy.
    assert "shuffle" not in written and "permutation_seed" not in written


def test_the_family_the_verdict_is_corrected_over_is_the_whole_planned_run(
    monkeypatch, tmp_path, capsys
):
    """rungs x cells x channels, counted over the cells that WILL run, and
    printed. Two cells, two rungs, two channels: 8. A family of one -- the
    uncorrected verdict -- would print z=1.96; a family counted per cell would
    print 4."""
    _stub(monkeypatch, tmp_path, cells=(("pixel_ae", 0), ("pixel_ae", 1)))
    assert script.main(_argv(tmp_path) + ["--rungs", "shuffled", "constant"]) == script.EXIT_OK
    out = capsys.readouterr().out
    assert "family: 8 comparisons (2 rungs x 2 cells x 2 channels)" in out, out
    assert f"z={script.family_threshold(8):.2f}" in out
    assert "z=1.96" not in out and "for 4 comparisons" not in out
    for seed in (0, 1):
        written = load_record(script.diagnostic_record_path(tmp_path, "pixel_ae", seed))
        assert written["ladder"]["family"] == 8


def test_the_contrast_flag_reaches_the_ladder_and_a_self_contrast_is_a_usage_error(
    monkeypatch, tmp_path, capsys
):
    """The pair the top rung is decided on is a CHOICE and the CLI carries
    it; the default is the pre-registered `CONTRAST`. A contrast of an action
    with itself is refused as a bad flag (argparse's 2) rather than as the
    library's ValueError twenty seconds into the probe refit."""
    state = _stub(monkeypatch, tmp_path)
    script.main(_argv(tmp_path))
    ladders = [kwargs for kind, kwargs, _ in state["diagnostics"] if kind == "ladder"]
    assert ladders and ladders[0]["contrast"] == script.CONTRAST == (3, 0)

    state = _stub(monkeypatch, tmp_path)
    script.main(_argv(tmp_path) + ["--contrast", "1", "2"])
    ladders = [kwargs for kind, kwargs, _ in state["diagnostics"] if kind == "ladder"]
    assert ladders and ladders[0]["contrast"] == (1, 2)

    with pytest.raises(SystemExit) as exit_info:
        script.parse_args(["--contrast", "4", "4"])
    assert exit_info.value.code == 2
    assert "same held action twice" in capsys.readouterr().err


def test_the_report_prints_the_held_table_and_the_cross_cell_block(
    monkeypatch, tmp_path, capsys
):
    """Both new blocks are assembled from a real `main` run: the held table
    carries the stub's held MOVE_FORWARD (+6) and NOOP (+1.5) rows by name,
    and the cross-cell block follows the verdicts."""
    _stub(monkeypatch, tmp_path)
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    out = capsys.readouterr().out
    held = out[out.index("held actions of the constant rung") :].splitlines()
    forward = next(line for line in held if "MOVE_FORWARD" in line)
    noop = next(line for line in held if "NOOP" in line)
    assert "+6.000" in forward and "+1.500" in noop, (forward, noop)
    assert out.index("--- verdicts ---") < out.index("across cells, per rung and channel")
    block = out[out.index("across cells, per rung and channel") :]
    assert "constant   position" in block and "constant   angle" in block


def test_the_paired_bars_in_the_report_come_from_the_paired_statistic(
    monkeypatch, tmp_path, capsys
):
    """The ruler table's ASSEMBLY, from a real `main` run.

    `sweep_ruler_table` is otherwise driven from a fabricated `"paired"` dict,
    so its rendering is pinned and what fills it is not -- substituting the
    unpaired `curve_standard_error` for `paired_standard_error` survives that.
    That substitution is the exact defect the paired ruler was added to remove,
    and on the shipped cells it inflates the bar by 1.7x to 3.9x, which is
    enough to declare the k=1 versus k=3 separation unresolvable.

    The stub gives each k its own half-width, so the paired bar (1.500) and the
    unpaired one (2.000) are different numbers, and the floor margin's paired
    bar (1.000) differs from the same margin taken against the floor's MEAN
    (2.000).
    """
    _stub(monkeypatch, tmp_path)
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    out = capsys.readouterr().out
    block = out[out.index("has to clear (paired)") :].splitlines()
    header, row = block[1], block[2]
    # Sliced by the header's own offsets: the margin value "+42.000" contains
    # the substring "2.000", so a bare `in` cannot separate the columns.
    column = lambda name, width: row[  # noqa: E731
        header.index(name) + len(name) - width : header.index(name) + len(name)
    ].strip()
    assert column("1v3 +-2se", 13) == "1.500", row
    assert column("k1-floor", 13) == f"+{TRAINED_SIGNATURE + 1.0:.3f}", row
    assert row[-9:].strip() == "1.000", "the floor margin's bar is not the paired one"

    # And the sweep table's own column is that curve's spread, which is a third
    # number again -- so the two tables cannot be reporting the same quantity.
    sweep_row = out[out.index("re-grounding sweep") :].splitlines()[2]
    assert sweep_row.strip().endswith("0.250"), sweep_row


def test_a_horizon_too_large_for_the_data_is_not_reported_as_a_mislabelled_checkpoint(
    monkeypatch, tmp_path
):
    """The catch was far broader than the condition it named.

    `diagnose_cell` calls `fit_probes`, `evaluate_rollout`, `regrounding_sweep`
    and `action_intervention_ladder`, and ValueError is raised from at least
    nine places under them -- the no-window guards in `evaluate_rollout`,
    `_diagnose` and the ladder's marginal pre-pass, and probe.py's
    context/horizon/empty-path/zero-variance checks. Every one of them printed "MISLABELLED CHECKPOINT" and exited
    EXIT_MISLABELLED_CHECKPOINT, sending the reader to look at checkpoint
    labels for what is a flag or a data problem. Reachable today with
    `--horizon 500` against the shipped 525-transition episodes.

    Only the labelling check is caught now; everything else is the uncaught
    traceback that exit status 1 is reserved for.
    """
    _stub(monkeypatch, tmp_path)

    def no_window(*args, **kwargs):
        raise ValueError("no validation window reached 501 frames")

    monkeypatch.setattr(script, "evaluate_rollout", no_window)
    with pytest.raises(ValueError, match="no validation window"):
        script.main(_argv(tmp_path))


def test_an_unsupported_device_is_a_named_refusal_before_any_work_is_done(
    monkeypatch, tmp_path, capsys
):
    """`_rng_snapshot` refuses an accelerator it cannot restore -- correctly --
    but the NotImplementedError propagated out of `main` as exit 1, the status
    the EXIT_* block reserves for an uncaught traceback, and only after the
    checkpoint load and a ~20 s probe refit.

    Checked against the same authority that does the refusing, so the two
    cannot drift apart, and checked FIRST so nothing is loaded for a run that
    cannot produce a matched stream.
    """
    _stub(monkeypatch, tmp_path, state=(state := {}))
    monkeypatch.setattr(script, "get_device", lambda prefer: torch.device("cuda"))
    assert script.main(_argv(tmp_path)) == script.EXIT_UNSUPPORTED_DEVICE
    assert "UNSUPPORTED DEVICE" in capsys.readouterr().out
    assert state["fit_probes"] == [], "the probe was refit for a run that cannot run"


def test_the_ladder_section_names_the_statistic_it_actually_prints(
    monkeypatch, tmp_path, capsys
):
    """The section was headed "final horizon step" while the number under it
    was the horizon MEAN of the per-window delta.

    Measured on frozen_ssl/seed0 the two differ by a factor of four -- 0.995
    against 4.065 -- because the delta curve trends monotonically from -0.589
    at step 1 to +4.065 at step 45, and the horizon mean is precisely the
    summary that cancels a late-horizon effect against the early steps. The
    mean is the right decision statistic and the endpoint is the study's
    headline, so both are printed and the heading names them.
    """
    _stub(monkeypatch, tmp_path)
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    out = capsys.readouterr().out
    heading = next(
        line for line in out.splitlines() if "action-intervention ladder" in line
    )
    assert "horizon-mean" in heading, heading
    assert "final horizon step" not in heading, heading
    block = out[out.index("action-intervention ladder") :]
    assert "final step" in block.splitlines()[4], block


def test_the_diagnostic_record_names_both_arm_and_seed_and_carries_its_device(
    monkeypatch, tmp_path
):
    """Written through `study.write_record`, so a NaN survives the round trip as
    a NaN rather than as `null` or 0.0 -- and the record carries the DEVICE and
    torch version, because the record reproduction is locked to them and nothing
    else in the study records that."""
    _stub(monkeypatch, tmp_path, changed=(False, False))
    assert script.main(_argv(tmp_path)) == script.EXIT_OK

    path = script.diagnostic_record_path(tmp_path, "pixel_ae", 0)
    assert path.name == "diagnostic_pixel_ae_seed0.json"
    written = load_record(path)
    assert written["arm"] == "pixel_ae" and written["seed"] == 0
    assert written["device"] == "cpu"
    assert written["torch_version"] == torch.__version__
    assert written["episodes"]["val"] == VAL_NAMES
    assert written["windows"] == {"total": 2, "episode": None}
    for name in LADDER:
        block = written["interventions"][name]
        assert block["windows_changed"] == 0 and block["episodes_changed"] == 0, name
        # The delta is undefined when nothing changed, and must come back as
        # NaN -- not as None and not as 0.0, which are different findings
        # about a cell.
        for metric in script.CHANNELS:
            assert all(np.isnan(v) for v in block[metric]["delta"]), (name, metric)
    # And it reaches disk as STRICT json: Python writes a NaN as the bare token
    # `NaN`, which is not JSON and which strict parsers reject, while Python's
    # own loader accepts it -- so a round trip through `json` alone cannot see
    # this. `write_record` writes `null` plus the map that inverts it.
    text = path.read_text()
    assert "NaN" not in text, "a bare NaN token reached the file"
    json.loads(text, parse_constant=lambda token: pytest.fail(f"non-JSON {token!r}"))


# ---------------------------------------------------------------------------
# The per-window series, the episode index and the embedding-space reading
# in the record; the noise reference's self-checks in the gate.
# ---------------------------------------------------------------------------


def test_the_record_persists_every_rungs_per_window_deltas_in_traversal_order_and_the_episode_index(
    monkeypatch, tmp_path
):
    """Today each rung persists only the 45-step horizon-mean delta; the
    per-window series lives in memory and is discarded. It is what a pooled
    cross-cell statistic needs, so every rung -- and every held action --
    writes its per-window horizon-mean delta over ALL windows in traversal
    order, and the record writes the window -> episode index ONCE, at the top
    level, so a reader can cluster by it.

    Distinct per window AND per rung (`offset + w`), so a reversed series,
    one rung's series under another's name, or `[delta_mean] * n` are all
    caught by value. Window 2 is UNCHANGED under the shuffled rung and
    carries a nonzero delta on purpose: the all-window series must carry it
    while the decision statistic `delta_mean` must not, and the persisted
    `window_steps_changed` is what reproduces the one from the other. The
    angle series is the position series over ten. Unequal clusters ([0, 0,
    1]), so an index that counted windows is caught.

    THE PER-WINDOW DELTA VARIES OVER THE HORIZON. Each window's row is
    `linspace(0, 2 * (4 + i + 0.1 w), H)`, whose mean is `4 + i + 0.1 w` and
    whose max, min, first and last steps are all different numbers -- with a
    constant row, as this fixture first had, `mean(axis=1)`, `max(axis=1)`
    and `[:, 0]` coincide and the horizon reduction of the very series the
    pooling reader consumes was pinned by nothing.
    """
    windows = 3
    per_rung = {
        name: np.stack([np.linspace(0.0, 2.0 * (4.0 + i + 0.1 * w), H) for w in range(windows)])
        for i, name in enumerate(LADDER)
    }
    for rows in per_rung.values():
        assert (rows.max(axis=1) != rows.mean(axis=1)).all() and (rows[:, 0] != rows.mean(axis=1)).all()
    _stub(
        monkeypatch, tmp_path,
        changed=(True, True, True),
        rung_deltas=per_rung,
        rung_changed={"shuffled": (True, True, False)},
        rung_steps={"shuffled": (2, 1, 0), "resampled": (1, 2, 3), "constant": (3, 3, 3)},
        episodes=[0, 0, 1],
    )
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    written = load_record(script.diagnostic_record_path(tmp_path, "pixel_ae", 0))

    assert written["windows"] == {"total": 3, "episode": [0, 0, 1]}
    for i, name in enumerate(LADDER):
        block = written["interventions"][name]
        expected = [4.0 + i + 0.1 * w for w in range(windows)]
        assert len(block["position"]["window_delta_mean"]) == 3, name
        assert block["position"]["window_delta_mean"] == pytest.approx(expected), name
        # Exactly the horizon MEAN of the per-window rows, bitwise.
        np.testing.assert_array_equal(
            block["position"]["window_delta_mean"], per_rung[name].mean(axis=1), err_msg=name
        )
        np.testing.assert_array_equal(
            block["angle"]["window_delta_mean"], (per_rung[name] / 10.0).mean(axis=1), err_msg=name
        )
        assert "episode" not in block and "window_episode" not in block, name
    shuffled = written["interventions"]["shuffled"]
    assert shuffled["window_steps_changed"] == [2, 1, 0]
    assert written["interventions"]["resampled"]["window_steps_changed"] == [1, 2, 3]
    # The decision statistic is the mean over the CHANGED windows only, and
    # the persisted mask is what reproduces it from the all-window series.
    series = np.asarray(shuffled["position"]["window_delta_mean"])
    mask = np.asarray(shuffled["window_steps_changed"]) > 0
    assert shuffled["position"]["delta_mean"] == series[mask].mean()
    assert shuffled["position"]["delta_mean"] != pytest.approx(series.mean())
    for name in LADDER:
        block = written["interventions"][name]
        series = np.asarray(block["position"]["window_delta_mean"])
        mask = np.asarray(block["window_steps_changed"]) > 0
        assert block["position"]["delta_mean"] == series[mask].mean(), name
    # Each held action carries its OWN series (+6 / +1.5 / +0.5 offsets).
    held = written["interventions"]["constant"]["held"]
    assert held["3"]["position"]["window_delta_mean"] == pytest.approx([6.0] * 3)
    assert held["0"]["position"]["window_delta_mean"] == pytest.approx([1.5] * 3)
    assert held["3"]["window_steps_changed"] == [3, 3, 3]


def test_the_record_writes_a_null_episode_index_when_the_ladder_reports_none(
    monkeypatch, tmp_path
):
    """A hand-built ladder carries no clustering; the record must say so with
    `null` rather than invent `range(n)` -- which a pooling reader would then
    cluster on as if every window were its own episode."""
    _stub(monkeypatch, tmp_path)
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    written = load_record(script.diagnostic_record_path(tmp_path, "pixel_ae", 0))
    assert written["windows"] == {"total": 2, "episode": None}


def test_the_record_refuses_a_per_window_series_or_an_episode_index_of_the_wrong_length(
    tmp_path
):
    """A stale index -- or a series persisted over the changed windows only --
    would let a pooling reader cluster row `w` on episode `w'`. Refused by
    name at write time, before a record that looks complete exists."""
    args = script.parse_args(_argv(tmp_path))
    device = torch.device("cpu")
    clean = _null_ladder(total=4) | {"window_episode": [0, 0, 1, 1]}
    assert script.build_record(clean, args, device, family=1)["windows"]["episode"] == [0, 0, 1, 1]

    with pytest.raises(ValueError, match="episode"):
        script.build_record(clean | {"window_episode": [0, 1]}, args, device, family=1)

    short = _null_ladder(total=4)
    short["rungs"]["resampled"]["position"]["window_mean"] = np.zeros(2)
    with pytest.raises(ValueError, match="resampled"):
        script.build_record(short, args, device, family=1)
    short = _null_ladder(total=4)
    short["rungs"]["constant"]["window_steps_changed"] = np.zeros(3, int)
    with pytest.raises(ValueError, match="constant"):
        script.build_record(short, args, device, family=1)

    # The three variants that were never exercised: a HELD action's series
    # (reached only through the recursion into `held`), a rung's embedding
    # numerator, and the top-level noise reference -- each wrong alone, each
    # named. Deleting the recursion, the embedding entry or the noise check
    # passed every test until these existed.
    block = lambda n: {  # noqa: E731
        "window_distance": np.ones(n), "curve": np.ones(H), "median": 1.0,
        "noise_median": 1.0, "ratio_of_medians": 1.0, "median_of_ratios": 1.0,
        "curve_ratio": np.ones(H),
    }
    short = _null_ladder(total=4)
    short["rungs"]["constant"]["held"] = {3: _rung([0.0] * H, [0.2] * H, total=4, embedding=block(4))}
    short["rungs"]["constant"]["held"][3]["position"]["window_mean"] = np.zeros(3)
    with pytest.raises(ValueError, match="held 3"):
        script.build_record(short, args, device, family=1)
    short = _null_ladder(total=4)
    short["rungs"]["resampled"]["embedding"] = block(5)
    with pytest.raises(ValueError, match="embedding.window_distance"):
        script.build_record(short, args, device, family=1)
    short = _null_ladder(total=4) | {
        "noise_reference": {"window_distance": np.ones(3), "curve": np.ones(H), "median": 1.0},
    }
    with pytest.raises(ValueError, match="noise reference"):
        script.build_record(short, args, device, family=1)
    # And each of those, at the right length, is accepted.
    whole = _null_ladder(total=4) | {
        "noise_reference": {"window_distance": np.ones(4), "curve": np.ones(H), "median": 1.0},
    }
    whole["rungs"]["resampled"]["embedding"] = block(4)
    whole["rungs"]["constant"]["held"] = {3: _rung([0.0] * H, [0.2] * H, total=4, embedding=block(4))}
    assert script.build_record(whole, args, device, family=1)["interventions"]["constant"]["held"]["3"]


def test_the_record_carries_the_embedding_reading_per_rung_and_the_noise_reference_once(
    monkeypatch, tmp_path
):
    """Per rung: the per-window numerator, its curve, the two medians and their
    ratio -- distinct values per rung, so one rung's block written under every
    rung's name is caught (0.5 against 0.75). A rung not measured in embedding
    space writes `null`, never 0. The noise reference is written ONCE at the
    top level with its two self-checks, and never under a rung; the self-check
    block carries both.
    """
    _stub(
        monkeypatch, tmp_path,
        changed=(True, True, True),
        rung_steps={"shuffled": (2, 1, 3), "resampled": (1, 2, 3), "constant": (3, 3, 3)},
        rung_embedding={"shuffled": [1.0, 2.0, 9.0], "resampled": [3.0, 3.0, 3.0], "constant": None},
        noise_distance=[4.0, 4.0, 4.0],
    )
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    written = load_record(script.diagnostic_record_path(tmp_path, "pixel_ae", 0))

    shuffled = written["interventions"]["shuffled"]["embedding"]
    assert shuffled["window_distance"] == [1.0, 2.0, 9.0]
    assert shuffled["curve"] == [4.0] * H
    assert shuffled["median"] == 2.0
    assert shuffled["noise_median"] == 4.0
    assert shuffled["ratio_of_medians"] == 0.5
    assert written["interventions"]["resampled"]["embedding"]["ratio_of_medians"] == 0.75
    assert written["interventions"]["constant"]["embedding"] is None
    # Held actions keep their own reading beside the contrast's.
    held = written["interventions"]["constant"]["held"]["3"]["embedding"]
    assert held["window_distance"] == [TRAINED_SIGNATURE + 6.0 + w for w in range(3)]
    noise = written["noise_reference"]
    assert noise == {
        "window_distance": [4.0, 4.0, 4.0], "curve": [4.0] * H, "median": 4.0,
        "windows_collapsed": 0, "stream_restored": True,
    }
    for name in LADDER:
        assert "noise_reference" not in written["interventions"][name]
        assert "window_noise_distance" not in written["interventions"][name]
    assert written["self_checks"]["noise_reference_stream_restored"] is True
    assert written["self_checks"]["noise_reference_windows_collapsed"] == 0


def test_an_undefined_embedding_ratio_round_trips_as_nan_and_never_as_zero(
    monkeypatch, tmp_path
):
    """A noise median of exactly 0 -- a rig that draws nothing -- makes every
    ratio undefined. It must reach the record as NaN through the `nonfinite`
    map and come back as NaN from `load_record`: 0.0 there would read as
    "action-blind" for a rung whose ruler measured nothing."""
    _stub(monkeypatch, tmp_path, noise_distance=[0.0, 0.0], noise_collapsed=0)
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    written = load_record(script.diagnostic_record_path(tmp_path, "pixel_ae", 0))
    ratio = written["interventions"]["shuffled"]["embedding"]["ratio_of_medians"]
    assert np.isnan(ratio)
    raw = json.loads(script.diagnostic_record_path(tmp_path, "pixel_ae", 0).read_text())
    assert raw["interventions"]["shuffled"]["embedding"]["ratio_of_medians"] is None
    assert "interventions.shuffled.embedding.ratio_of_medians" in raw["nonfinite"]


def test_the_new_record_fields_are_derived_from_the_loaded_checkpoint(
    monkeypatch, tmp_path
):
    """M8, for the new path: the embedding block's VALUES are what the loaded
    parameters produce. The fake ladder derives every numerator from
    `model.signature()`, so a script that evaluated a freshly initialised model
    writes a different number here -- asserted to differ first."""
    assert _TinyModel().signature() != TRAINED_SIGNATURE
    _stub(monkeypatch, tmp_path)
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    written = load_record(script.diagnostic_record_path(tmp_path, "pixel_ae", 0))
    assert written["interventions"]["shuffled"]["embedding"]["window_distance"] == [
        TRAINED_SIGNATURE + 4.0 + w for w in range(2)
    ]
    assert written["noise_reference"]["window_distance"] == [TRAINED_SIGNATURE] * 2


def test_a_noise_reference_that_moved_the_stream_or_collapsed_is_a_stream_divergence(
    monkeypatch, tmp_path, capsys
):
    """Two ways the ruler can be wrong, one status: the stream did not come
    back after the noise draw (every later window's rungs moved with it), or
    the reference was bitwise the canonical imagination in some window (it
    drew the canonical's uniforms and the ratio there is x / 0). Both are the
    defect class `EXIT_STREAM_DIVERGED` names -- the rungs no longer share
    the stream -- each exercised alone, each named in the output, and the
    clean pair exits OK.
    """
    _stub(monkeypatch, tmp_path, noise_restored=False)
    assert script.main(_argv(tmp_path)) == script.EXIT_STREAM_DIVERGED
    out = capsys.readouterr().out
    assert "STREAM DIVERGED" in out and "noise reference" in out and "restore" in out

    _stub(monkeypatch, tmp_path, noise_collapsed=2)
    assert script.main(_argv(tmp_path)) == script.EXIT_STREAM_DIVERGED
    out = capsys.readouterr().out
    assert "STREAM DIVERGED" in out and "noise reference" in out and "2 of 2" in out

    _stub(monkeypatch, tmp_path)
    assert script.main(_argv(tmp_path)) == script.EXIT_OK


def test_the_noise_self_checks_land_in_their_own_columns():
    """Beside the three exact equalities: whether the stream came back after
    every noise draw, and how many windows' reference was the canonical
    latent. Given values that differ from every other column."""
    cells = {("pixel_ae", 0): _cell([1.0], [0.5]) | {"noise_restored": False, "noise_collapsed": 7}}
    header, row = script.selfcheck_table(cells, ["pixel_ae"], [0], 1).splitlines()
    assert header.split()[-2:] == ["noise_restored", "noise_same"]
    assert row.split()[-2:] == ["False", "7"], row


def test_the_ladder_table_prints_the_embedding_ratio_in_its_own_row_and_MISSING_without_one():
    """One more row per rung: the probe-free ratio against sampling noise.
    Typed here, not read from `LADDER_ROWS`. 0.5 and 0.75 land under their
    own rungs; a rung with no embedding block prints MISSING there and
    nowhere else; an undefined ratio prints as `nan (noise 0)`, never as
    0.000."""
    assert list(script.LADDER_ROWS) == [
        "horizon-mean", "angle horizon-mean", "final step", "changed/total",
        "steps moved/horizon", "multiset dist/horizon", "embed ratio num/noise",
        "embed contrast/noise", "embed step 1|H ratio",
    ]
    block = lambda ratio, noise=4.0: {  # noqa: E731
        "window_distance": np.array([1.0, 2.0]), "curve": np.array([1.5] * 2),
        "median": 1.5, "noise_median": noise, "ratio_of_medians": ratio,
        "median_of_ratios": ratio, "curve_ratio": np.array([ratio, ratio]),
    }
    rungs = {
        "shuffled": _rung([1.0, 1.0], [0.5, 0.5], mean=1.0, embedding=block(0.5)),
        "resampled": _rung([2.0, 2.0], [0.5, 0.5], mean=2.0, embedding=block(0.75)),
        "constant": _rung([3.0, 3.0], [0.5, 0.5], mean=3.0, embedding=None),
    }
    table = script.ladder_table({("pixel_ae", 0): _ladder_cell(rungs)}, ["pixel_ae"], [0], LADDER)
    row = lambda index: _seed_columns(_rung_rows(table, index)["embed ratio num/noise"])[0].strip()  # noqa: E731
    assert row(0) == "0.500"
    assert row(1) == "0.750"
    # The constant rung's row is the held PAIR, and a cell with no held
    # readings prints MISSING for each of the two; its contrast row too.
    assert row(2) == "MISSING|MISSING"
    assert _seed_columns(_rung_rows(table, 2)["embed contrast/noise"])[0].strip() == "MISSING"
    assert "MISSING" not in _rung_rows(table, 2)["horizon-mean"]

    rungs["shuffled"] = _rung([1.0, 1.0], [0.5, 0.5], mean=1.0, embedding=block(float("nan"), 0.0))
    table = script.ladder_table({("pixel_ae", 0): _ladder_cell(rungs)}, ["pixel_ae"], [0], LADDER)
    assert row(0) == "nan (noise 0)"
    assert "0.000" not in _rung_rows(table, 0)["embed ratio num/noise"]
    # Undefined over a POSITIVE ruler is the other way to have no ratio: no
    # window changed. Named as that, never as the ruler's fault.
    assert script._ratio_text(block(float("nan"), 4.0)) == "nan (no window)"
    assert script._ratio_text(None) == "MISSING"


def test_the_verdict_quotes_the_embedding_ratio_where_the_probe_cannot_register():
    """On every M3b `cnn` cell the position probe was a constant predictor and the
    verdict reads UNMEASURABLE THROUGH THIS PROBE. The embedding-space reading
    is the one thing that CAN be read there, so that branch quotes the held
    contrast's ratio -- and only when the rung carries one. The decision
    logic itself is unchanged: the sentence is appended, not substituted."""
    unmeasurable = dict(floor=9.0, persistence=1.0)
    plain = _verdict(_null_ladder(**unmeasurable))
    assert "UNMEASURABLE THROUGH THIS PROBE" in plain
    assert "embedding space" not in plain

    cell = _null_ladder(**unmeasurable)
    cell["rungs"]["constant"]["embedding"] = _embedding(1.7)
    quoted = _verdict(cell)
    assert "UNMEASURABLE THROUGH THIS PROBE" in quoted
    assert "In embedding space" in quoted and "1.700" in quoted, quoted
    assert "held MOVE_FORWARD against held NOOP" in quoted


def _embedding(ratio, *, noise=4.0, curve_ratio=None, median_of_ratios=None) -> dict:
    """An embedding block as `rung_block` builds it, with every scalar its
    own number: the ratio of medians, the median of ratios (defaults to a
    DIFFERENT value, so the two cannot be printed for each other) and the
    per-step ratio curve (defaults to a ramp, so step 1 and step H differ)."""
    return {
        "window_distance": np.array([1.0, 2.0]), "curve": np.full(H, 1.5),
        "median": 1.5, "noise_median": noise, "ratio_of_medians": ratio,
        "median_of_ratios": ratio + 0.11 if median_of_ratios is None else median_of_ratios,
        "curve_ratio": (
            np.linspace(ratio / 2.0, 2.0 * ratio, H) if curve_ratio is None
            else np.asarray(curve_ratio, float)
        ),
    }


def test_the_ladder_table_reads_the_constant_rung_on_the_same_axis_as_the_rungs_below_it():
    """The top rung's numerator is the distance between two COUNTERFACTUALS
    (held FORWARD against held NOOP), which by the triangle inequality is
    structurally larger than either's distance to real -- a different
    estimand from the shuffled and resampled rows, and printing all three
    under one label manufactured a monotone ladder. So the `embed ratio
    num/noise` row prints, for the constant rung, the two contrast actions'
    OWN held-vs-real ratios in contrast order (`first|second`); the
    between-held distance goes in its own row, `embed contrast/noise`, which
    the sequence rungs leave blank; and `embed step 1|H ratio` prints the
    rung's own per-step ratio at the first and last step -- the contrast's
    for the constant rung, right under the contrast row. Every number is
    distinct, so a row that read another's is caught by value; the held
    pair is NOT the sorted order (0.74 comes first because held 3 is first
    in the contrast).
    """
    held = {
        3: _rung([31.0] * H, [12.0] * H, mean=31.0, embedding=_embedding(0.74)),
        0: _rung([-11.6] * H, [7.9] * H, mean=-11.6, embedding=_embedding(0.35)),
        1: _rung([2.0] * H, [1.0] * H, mean=2.0, embedding=_embedding(0.55)),
    }
    rungs = {
        "shuffled": _rung([1.0] * H, [0.5] * H, mean=1.0, embedding=_embedding(0.21, curve_ratio=[0.15, 0.2, 0.29])),
        "resampled": _rung([2.0] * H, [0.5] * H, mean=2.0, embedding=_embedding(0.4)),
        "constant": _rung(
            [3.0] * H, [0.5] * H, mean=3.0, contrast=(3, 0), held=held,
            embedding=_embedding(1.02, curve_ratio=[0.67, 0.9, 1.14]),
        ),
    }
    table = script.ladder_table({("pixel_ae", 0): _ladder_cell(rungs)}, ["pixel_ae"], [0], LADDER)
    cell = lambda index, row: _seed_columns(_rung_rows(table, index)[row])[0].strip()  # noqa: E731
    assert cell(0, "embed ratio num/noise") == "0.210"
    assert cell(1, "embed ratio num/noise") == "0.400"
    assert cell(2, "embed ratio num/noise") == "0.740|0.350", "the constant rung's row is not the held pair"
    assert cell(0, "embed contrast/noise") == "" and cell(1, "embed contrast/noise") == ""
    assert cell(2, "embed contrast/noise") == "1.020"
    assert cell(0, "embed step 1|H ratio") == "0.150|0.290"
    assert cell(2, "embed step 1|H ratio") == "0.670|1.140"
    assert "1.02" not in _rung_rows(table, 2)["embed ratio num/noise"]
    # A constant rung whose held blocks were never measured prints MISSING
    # for the pair, and a contrast with no reading prints MISSING too.
    bare = {"constant": _rung([3.0] * H, [0.5] * H, mean=3.0, contrast=(3, 0), held={
        3: _rung([31.0] * H, [12.0] * H, mean=31.0), 0: _rung([-11.6] * H, [7.9] * H, mean=-11.6),
    })}
    table = script.ladder_table({("pixel_ae", 0): _ladder_cell(bare)}, ["pixel_ae"], [0], ("constant",))
    assert cell(0, "embed ratio num/noise") == "MISSING|MISSING"
    assert cell(0, "embed contrast/noise") == "MISSING"
    assert cell(0, "embed step 1|H ratio") == "MISSING"


def test_the_held_table_prints_each_held_actions_embedding_ratio_and_its_step_1_and_H():
    """The same-axis reading of the top rung is each held action's own
    distance to real, so the held table carries it: the horizon-mean ratio
    and the per-step ratio at step 1 and step H, per action, beside the
    delta and steps-moved rows -- four rows per action. MISSING when the
    action was not measured in embedding space."""
    held = {
        3: _rung([31.0] * 2, [12.0] * 2, mean=31.0, se=6.0,
                 embedding=_embedding(0.74, curve_ratio=[0.5, 0.9])),
        0: _rung([-11.6] * 2, [7.9] * 2, mean=-11.6, se=3.95),
    }
    cells = {("pixel_ae", 0): _ladder_cell({
        "constant": _rung([42.6] * 2, [14.0] * 2, mean=42.6, se=7.0, contrast=(3, 0), held=held),
    })}
    lines = script.held_table(cells, ["pixel_ae"], [0], LADDER).splitlines()
    assert len(lines) == 1 + 4 * len(held), lines
    noop, forward = lines[1:5], lines[5:9]
    assert forward[0][12:25].strip() == "MOVE_FORWARD"
    assert [line[25:47].strip() for line in forward] == [
        "horizon-mean", "steps moved/horizon", "embed ratio num/noise", "embed step 1|H ratio",
    ]
    assert forward[2].split()[-1] == "0.740"
    assert forward[3].split()[-1] == "0.500|0.900"
    assert noop[2].split()[-1] == "MISSING" and noop[3].split()[-1] == "MISSING"


def test_the_record_persists_both_ratio_summaries_and_the_per_step_ratio_curve(
    monkeypatch, tmp_path
):
    """Two summaries of one series -- the ratio of medians (decision-bearing)
    and the median of per-window ratios -- and the per-step ratio of the two
    mean curves, so a reader can see where in the horizon the response sits.
    On the stub the numerators are `signature + offset + w` over a noise of
    [1, 6, 2] per window -- THREE windows, so the noise mean (3) and median
    (2) differ and a curve ratio taken over the median instead of the ruler's
    own curve is caught -- so the two summaries are typed by hand and
    differ; the curve ratio is the flat numerator curve over the flat noise
    curve. NaN over a zero ruler in every one of them, never 0."""
    noise = np.array([1.0, 6.0, 2.0])
    _stub(
        monkeypatch, tmp_path, changed=(True, True, True), noise_distance=list(noise),
        rung_steps={"shuffled": (2, 1, 3), "resampled": (1, 2, 3), "constant": (3, 3, 3)},
    )
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    written = load_record(script.diagnostic_record_path(tmp_path, "pixel_ae", 0))
    shuffled = written["interventions"]["shuffled"]["embedding"]
    numerator = np.array([TRAINED_SIGNATURE + 4.0 + w for w in range(3)])
    assert shuffled["ratio_of_medians"] == np.median(numerator) / 2.0
    assert shuffled["median_of_ratios"] == np.median(numerator / noise)
    assert shuffled["ratio_of_medians"] != shuffled["median_of_ratios"]
    assert shuffled["curve_ratio"] == [numerator.mean() / 3.0] * H
    held = written["interventions"]["constant"]["held"]["0"]["embedding"]
    assert held["median_of_ratios"] == np.median((TRAINED_SIGNATURE + 1.5 + np.arange(3)) / noise)

    _stub(monkeypatch, tmp_path, noise_distance=[0.0, 0.0])
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    written = load_record(script.diagnostic_record_path(tmp_path, "pixel_ae", 0))
    shuffled = written["interventions"]["shuffled"]["embedding"]
    assert np.isnan(shuffled["median_of_ratios"])
    assert all(np.isnan(v) for v in shuffled["curve_ratio"])


def test_the_verdicts_embedding_note_separates_the_held_pair_from_the_contrast_and_quotes_the_horizon():
    """Where the probe cannot register, the note quotes the two contrast
    actions' own held-vs-real ratios (the same axis as the rungs below),
    the between-held contrast SEPARATELY and named as a between-counterfactual
    distance, the contrast's per-step ratio at step 1 and step H, and how
    much the noise ruler itself grows over the horizon -- pixel_ae's is flat and
    frozen_ssl's grows 6x, and the horizon-mean hides both."""
    cell = _null_ladder(floor=9.0, persistence=1.0)
    cell["rungs"]["constant"]["embedding"] = _embedding(1.02, curve_ratio=np.linspace(0.67, 1.14, H))
    cell["rungs"]["constant"]["held"] = {
        3: _rung([0.0] * H, [0.2] * H, embedding=_embedding(0.355)),
        0: _rung([0.0] * H, [0.2] * H, embedding=_embedding(0.74)),
    }
    cell["noise_reference"] = {"window_distance": np.ones(4), "curve": np.linspace(2.1, 12.5, H), "median": 1.0}
    note = script._embedding_note(cell)
    assert "held MOVE_FORWARD vs real 0.355" in note and "held NOOP vs real 0.740" in note, note
    assert "between the two held imaginations" in note and "1.020" in note
    assert "not on the same axis" in note
    assert "step 1 0.670" in note and f"step {H} 1.140" in note, note
    assert "ruler grows 2.100 -> 12.500" in note, note
    assert "two draws" in note
    cell["rungs"]["constant"]["held"][3]["embedding"] = None
    assert "held MOVE_FORWARD vs real MISSING" in script._embedding_note(cell)


def test_the_noise_growth_table_prints_each_cells_ruler_at_step_1_and_step_H():
    """Per cell, the noise ruler's window-mean curve at the first and last
    horizon step, so a flat ruler (no compounding dynamics in embedding
    space) and a growing one are told apart at a glance -- the shape the
    horizon-mean ratio cannot show. MISSING for a cell that did not run."""
    cells = {
        ("pixel_ae", 0): _cell([1.0], [0.5]) | {
            "noise_reference": {"window_distance": np.ones(4), "curve": np.array([0.12, 0.13]), "median": 1.0},
        },
        ("pixel_ae", 2): _cell([1.0], [0.5]) | {
            "noise_reference": {"window_distance": np.ones(4), "curve": np.array([2.1, 12.5]), "median": 1.0},
        },
    }
    header, ruler, growth = script.noise_table(cells, ["pixel_ae"], [0, 1, 2]).splitlines()
    assert header.split() == ["arm", "row", "seed", "0", "seed", "1", "seed", "2"]
    assert ruler.split() == ["pixel_ae", "noise", "ruler", "step", "1->H", "0.120->0.130", "MISSING", "2.100->12.500"], ruler
    assert growth.split() == ["ruler", "growth", "H/1", "x1.08", "MISSING", "x5.95"], growth


def test_the_script_reads_the_cluster_standard_error_from_the_pooling_module():
    """ONE implementation. The per-cell ruler and the pooled ruler are the
    same estimator, so the script imports it from `pooling` rather than
    keeping a copy that could drift while both kept passing. The clustered
    fixture still reads 20.0 through the script, and the same rows give the
    same number through the module."""
    assert script.cluster_standard_error is pooling.cluster_standard_error
    assert not hasattr(script, "_cluster_standard_error")
    reference = _rollout([1.0] * H)
    deltas = np.array([[0.0] * H, [0.0] * H, [40.0] * H, [40.0] * H])
    summary = script.delta_summary(
        _shuffle(reference, [2.0] * H, [True] * 4, deltas=deltas, episodes=[0, 0, 1, 1])
    )
    assert summary["se"] == pytest.approx(20.0)
    assert pooling.cluster_standard_error(deltas.mean(axis=1), [0, 0, 1, 1]) == pytest.approx(20.0)
    assert summary["window_mean"] == pytest.approx([0.0, 0.0, 40.0, 40.0])


def test_a_record_this_script_writes_is_readable_by_the_pooling_reader(monkeypatch, tmp_path):
    """The two schemas are pinned to each other here: a record the diagnose
    script writes -- with an episode index -- is exactly what
    `pooling.read_series` consumes, series and identity alike, and one
    written by a ladder with no clustering is refused by it rather than
    read naively."""
    _stub(
        monkeypatch, tmp_path, changed=(True, True, True), episodes=[0, 0, 1],
        rung_steps={"shuffled": (2, 1, 0), "resampled": (1, 2, 3), "constant": (3, 3, 3)},
        rung_changed={"shuffled": (True, True, False)},
    )
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    record = load_record(script.diagnostic_record_path(tmp_path, "pixel_ae", 0))
    for rung in LADDER:
        cell = pooling.read_series(record, rung, "angle")
        assert (cell.arm, cell.seed, cell.rung, cell.channel) == ("pixel_ae", 0, rung, "angle")
        np.testing.assert_array_equal(cell.episode, [0, 0, 1])
        assert cell.delta.shape == cell.changed.shape == cell.embedding.shape == cell.noise.shape == (3,)
        assert (cell.device, cell.torch_version, cell.horizon, cell.context) == ("cpu", torch.__version__, H, 5)
    np.testing.assert_array_equal(
        pooling.read_series(record, "shuffled").changed, [True, True, False]
    )

    _stub(monkeypatch, tmp_path)
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    with pytest.raises(pooling.StaleRecord, match="episode"):
        pooling.read_series(load_record(script.diagnostic_record_path(tmp_path, "pixel_ae", 0)), "shuffled")
