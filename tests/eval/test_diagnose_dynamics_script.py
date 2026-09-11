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

from mbfps.eval.diagnostics import RegroundingSweep, ShuffleResult
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


def _shuffle(
    reference: RolloutResult, shuffled, changed, deltas=None, episodes=None
) -> ShuffleResult:
    shuffled = np.asarray(shuffled, dtype=float)
    changed = np.asarray(changed, dtype=bool)
    if deltas is None:
        deltas = np.tile(shuffled - reference.rssm_position, (changed.size, 1))
    deltas = np.asarray(deltas, dtype=float)
    return ShuffleResult(
        real=reference,
        shuffled_position=shuffled,
        shuffled_angle=shuffled / 10.0,
        changed=changed,
        window_position_delta=deltas,
        window_angle_delta=deltas / 10.0,
        permutation_seed=0,
        window_episode=(None if episodes is None else np.asarray(episodes, int)),
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


def _write_record(out_dir, arm, seed, curve, val_names=None):
    write_record(
        job_record_path(out_dir, StudyJob(arm=arm, seed=seed)),
        {
            "arm": arm,
            "seed": seed,
            "episodes": {"val": list(VAL_NAMES if val_names is None else val_names)},
            "curves": {"rssm_position": [float(v) for v in curve]},
        },
    )


def _stub(
    monkeypatch,
    tmp_path,
    *,
    cells=(("cnn", 0),),
    checkpoint_labels=None,
    record_curve=None,
    record_names=None,
    sweep_offset=0.0,
    smallest_k_is_floor=False,
    shuffle_offset=0.0,
    changed=(True, True),
    shuffle_deltas=None,
    state=None,
):
    """Everything `main` touches, replaced; the checkpoint and record are real files.

    `state` collects the calls a test wants to inspect.
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

    def fake_shuffle(model, val, probe, **kwargs):
        state["diagnostics"].append(("shuffle", kwargs, None))
        reference = _rollout([model.signature()] * H)
        return _shuffle(
            _rollout(reference.rssm_position + shuffle_offset),
            reference.rssm_position + 4.0,
            changed,
            deltas=shuffle_deltas,
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
    monkeypatch.setattr(script, "action_shuffled_rollout", fake_shuffle)
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
    for other in ("run_study", "report_study"):
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
    50, while the sweep's is 0.5.
    """
    _stub(
        monkeypatch, tmp_path,
        shuffle_deltas=np.array([[0.0] * H, [100.0] * H]),
    )
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    out = capsys.readouterr().out
    assert "+-2 SE (100.000)" in out, out[out.index("--- verdicts ---"):]
    assert "M4 is BLOCKED" in out, (
        "the verdict decided on a spread that is not the shuffle's"
    )


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

    state = _stub(monkeypatch, tmp_path, cells=(("cnn", 2),))
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
    state = _stub(monkeypatch, tmp_path, cells=(("cnn", 7),))
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


def test_the_permutation_seed_reaches_the_shuffle(monkeypatch, tmp_path):
    """One permutation is one draw; which draw it was has to be controllable, or
    a null distribution cannot be built from repeated runs."""
    state = _stub(monkeypatch, tmp_path)
    script.main(_argv(tmp_path) + ["--permutation-seed", "3"])
    shuffles = [kwargs for kind, kwargs, _ in state["diagnostics"] if kind == "shuffle"]
    assert shuffles and shuffles[0]["permutation_seed"] == 3


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
        checkpoint_labels={"arm": "cnn", "seed": 0},
    )
    assert script.main(_argv(tmp_path) + ["--arms", "frozen_ssl"]) == (
        script.EXIT_MISLABELLED_CHECKPOINT
    )
    out = capsys.readouterr().out
    assert "arm='cnn'" in out, "the checkpoint's OWN labels were not reported"


def test_a_checkpoint_wrong_in_the_seed_alone_is_refused(monkeypatch, tmp_path, capsys):
    """The architecture is shared across seeds, so a seed-wrong checkpoint loads
    perfectly cleanly and nothing else in the run would raise."""
    _stub(
        monkeypatch, tmp_path, cells=(("cnn", 1),),
        checkpoint_labels={"arm": "cnn", "seed": 0},
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
        for arm in ("cnn", "random_vit")
        for seed in (0, 1, 2)
    }
    assert len(names) == 6


# ---------------------------------------------------------------------------
# The report itself.
# ---------------------------------------------------------------------------


def _cell(
    delta, band, changed=4, total=4, mean=None, se=0.5, outside=0,
    open_loop=0.0, record=0.0, stream=0.0, smallest_k_is_floor=False,
    floor=1.0, persistence=9.0, episodes=4,
):
    """A fabricated cell. `mean`/`se` are the AGGREGATE the verdict decides on;
    `delta`/`band` are the per-step curves it reports beside that.

    The three self-check numbers are PARAMETERS rather than a shared 0.0. Left
    equal they make every column of `selfcheck_table` interchangeable, so
    printing one where another belongs is satisfied by the fixture and not by
    the code -- the L1 coincidence this repo names. They default to 0.0 because
    that is the clean-run rendering most tests want; the column test passes
    three different ones.
    """
    delta = np.asarray(delta, float)
    return {
        "open_loop": open_loop, "record": record, "stream": stream,
        "smallest_k_is_floor": smallest_k_is_floor,
        "windows_total": total, "windows_changed": changed,
        "delta": delta, "band": np.asarray(band, float),
        "mean": (float(np.nanmean(delta)) if delta.size and not np.isnan(delta).all()
                 else float("nan")) if mean is None else mean,
        "se": se, "se_independent": se, "episodes": episodes,
        "steps_outside": outside, "steps": delta.size,
        "expected_outside": 0.05 * delta.size,
        "delta_final": float(delta[-1]) if delta.size else float("nan"),
        "k": {1: 5.0, 3: 7.0}, "floor": floor, "persistence": persistence,
        "spread_final": 0.25,
        "paired": {"1v3": 0.125},
        "floor_margin": 4.0, "floor_margin_se": 0.75, "smallest_k": 1,
    }


ROW_SHAPED = {
    "selfcheck_table": lambda cells: script.selfcheck_table(cells, ["cnn"], [0, 1, 2], 1),
    "sweep_table": lambda cells: script.sweep_table(cells, ["cnn"], [0, 1, 2], (1, 3)),
    "sweep_ruler_table": lambda cells: script.sweep_ruler_table(
        cells, ["cnn"], [0, 1, 2], (1, 3)
    ),
}


@pytest.mark.parametrize("table", sorted(ROW_SHAPED))
def test_the_row_shaped_tables_print_a_row_per_cell_and_MISSING_for_a_gap(table):
    """A cell with no checkpoint gets its own row saying so, rather than being
    omitted -- an omitted row makes an incomplete study look complete."""
    cells = {
        ("cnn", 0): _cell([1.0], [0.5]),
        ("cnn", 2): _cell([2.0], [0.5]) | {"floor": 42.0, "k": {1: 42.0, 3: 42.0}},
    }
    rows = {line.split()[1]: line for line in ROW_SHAPED[table](cells).splitlines()[1:]}
    assert set(rows) == {"0", "1", "2"}, rows
    assert "MISSING" in rows["1"]
    assert "MISSING" not in rows["0"] and "MISSING" not in rows["2"]
    if table == "sweep_table":
        assert "42.000" in rows["2"], rows["2"]
        assert "42.000" not in rows["1"], "seed 2's number landed in seed 1's row"


def test_the_shuffle_table_puts_every_seeds_number_in_its_own_column():
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
        ("cnn", 0): _cell([1.0, 5.0], [0.5, 0.5], mean=1.0),
        ("cnn", 2): _cell([2.0, 6.0], [0.5, 0.5], mean=2.0, changed=3, total=5),
    }
    header, delta_row, final_row, counts_row = script.shuffle_table(
        cells, ["cnn"], [0, 1, 2]
    ).splitlines()
    width, prefix = 18, 12 + 16
    columns = lambda row: [  # noqa: E731
        row[prefix + i * width : prefix + (i + 1) * width] for i in range(3)
    ]
    columns_of = lambda row: [c.strip() for c in columns(row)]  # noqa: E731
    assert columns_of(header) == ["seed 0", "seed 1", "seed 2"], header
    assert "+1.000" in columns(delta_row)[0]
    assert columns(delta_row)[1].strip() == "MISSING"
    assert "+2.000" in columns(delta_row)[2], "seed 2's delta is not in seed 2's column"
    assert columns_of(final_row)[0] == "+5.000", final_row
    assert columns_of(final_row)[1] == "MISSING"
    assert columns_of(final_row)[2] == "+6.000", "seed 2's endpoint is in the wrong column"
    assert "4/4" in columns(counts_row)[0].replace(" ", "")
    assert columns(counts_row)[1].strip() == "MISSING"
    assert "3/5" in columns(counts_row)[2].replace(" ", "")


def test_the_verdict_names_m4_only_when_the_aggregate_delta_is_inside_two_se(
    capsys,
):
    """Both branches, in one test: either alone leaves the other unexercised.

    The decision is the AGGREGATE -- the mean per-window delta against its own
    standard error -- and never the per-step exceedance count. Two standard
    errors is ~95% per step, so under the null about 2 of 45 horizon steps fall
    outside their band by chance, and a rule that fired on one of them would
    call almost every cell responsive. The count is still printed, next to what
    chance alone gives, so a reader can see an effect the aggregate cancels.
    """
    inside = script.verdict_block(
        "cnn", 0, _cell([0.4, -0.4, 0.0] * 15, [0.2] * 45, mean=0.05, se=0.1, outside=3)
    )
    assert "M4 is BLOCKED" in inside
    assert "NOT blocked" not in inside
    assert "3/45" in inside and "2.2 expected by chance" in inside, (
        "the per-step count and its chance expectation were not printed beside "
        "the aggregate"
    )

    outside = script.verdict_block(
        "cnn", 0, _cell([0.4, -0.4, 0.0] * 15, [0.2] * 45, mean=9.0, se=0.1, outside=3)
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


def test_a_single_step_outside_its_band_is_not_by_itself_a_verdict(capsys):
    """The 45-comparison error, pinned. A cell whose aggregate sits inside two
    standard errors must still read as BLOCKED even when several individual
    steps exceed their own per-step bands -- which is what chance produces at
    this horizon."""
    text = script.verdict_block(
        "cnn", 1, _cell([0.0] * 45, [0.2] * 45, mean=0.01, se=0.2, outside=4)
    )
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
    assert summary["steps"] == H
    assert summary["expected_outside"] == pytest.approx(0.05 * H)


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
    text = script.verdict_block(
        "cnn", 1, _cell([np.nan] * 3, [np.nan] * 3, changed=0, total=7)
    )
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
    protocol divergence sends the reader to the wrong code.
    """
    cells = {("cnn", 0): _cell([1.0], [0.5], open_loop=1e-1, record=2e-2, stream=3e-3)}
    header, row = script.selfcheck_table(cells, ["cnn"], [0], 1).splitlines()
    assert header.split() == [
        "arm", "seed", "open_loop_k", "record_repro", "stream_drift", "k1_is_floor"
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


def test_the_self_check_column_names_the_k_the_alarm_was_computed_at():
    """The alarm is computed at the SMALLEST k, which is 1 only by default.

    `--ks 5 45` makes it a statement about k=5, and self-check 2's whole point
    is the prior/posterior separation at ONE step -- at k=5 it is a different,
    weaker claim. The header therefore carries the k rather than a hardcoded
    "k1", and it agrees with the record's `smallest_k_is_bitwise_the_floor`.
    """
    cells = {("cnn", 0): _cell([1.0], [0.5])}
    header = script.selfcheck_table(cells, ["cnn"], [0], 5).splitlines()[0]
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
    text = script.verdict_block(
        "cnn", 1,
        _cell([1.0, 2.0, 3.0], [np.inf] * 3, changed=1, total=229,
              mean=float("nan"), se=float("inf")),
    )
    assert "UNINTERPRETABLE" in text
    assert "1/229" in text
    assert "M4" not in text, text
    assert "nan" not in text.lower(), text


def test_the_verdict_scales_the_effect_against_the_actionable_band():
    """Accepting a null needs a scale, or the same sentence covers evidence
    that differs by orders of magnitude.

    Measured across the nine shipped cells the band the verdict is read against
    spans 0.014 to 3.45 map units, i.e. 1.6% to 8.4% of the persistence-to-floor
    band on the feature arms. "The prior is not using the action" is the same
    words for all of them; "no action effect larger than X% of the actionable
    range is detectable" is not.
    """
    text = script.verdict_block(
        "cnn", 0,
        _cell([0.0] * 4, [0.2] * 4, mean=0.05, se=0.5, floor=1.0, persistence=21.0),
    )
    assert "M4 is BLOCKED" in text
    # 2*se = 1.0 against a band of 20.0 -> 5.0%.
    assert "5.0%" in text, text
    assert "20.000" in text, "the band the effect was scaled against was not named"


def test_a_non_positive_band_is_said_to_have_no_scale_rather_than_scaled_anyway():
    """On every `cnn` cell the floor EXCEEDS persistence at the final horizon
    step, so the "actionable range" is negative and a percentage of it is
    meaningless -- and would print as a negative or absurd percent beside a
    confident verdict. Named as absent instead."""
    text = script.verdict_block(
        "cnn", 0,
        _cell([0.0] * 4, [0.2] * 4, mean=0.05, se=0.5, floor=9.0, persistence=8.91),
    )
    assert "M4 is BLOCKED" in text
    assert "no actionable range" in text, text
    assert "%" not in text, "a percentage was printed against a non-positive band"


def test_the_verdict_carries_a_qualifier_when_the_shipped_record_did_not_reproduce():
    """The RECORD MISMATCH notice is printed BELOW every verdict, so a reader
    who stops at the verdict never sees it.

    Measured: on CPU the record reproduction for frozen_ssl/seed0 is 12.4 map
    units and the headline delta itself moves from +0.995 to +1.872 -- the
    number the verdict interprets is not stable across devices at the precision
    it is being interpreted at. So the verdict says so where it is read.
    """
    text = script.verdict_block("cnn", 0, _cell([0.0] * 4, [0.2] * 4, record=12.4))
    assert "1.240e+01" in text, text
    clean = script.verdict_block("cnn", 0, _cell([0.0] * 4, [0.2] * 4, record=0.0))
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
    assert clustered["episodes"] == 2
    assert naive["episodes"] == 0, "unknown clustering must not be reported as any"
    assert clustered["se_independent"] == pytest.approx(naive["se"])
    # Two clusters of two identical values: every within-cluster residual is
    # +-20, so the clustered SE is exactly 20 while the naive one is 11.55.
    assert clustered["se"] == pytest.approx(20.0)
    assert clustered["se"] > clustered["se_independent"], (
        "the clustered ruler collapsed onto the independent one"
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
    cells = {("cnn", 0): _cell([1.0], [0.5])}
    header, row = script.sweep_ruler_table(cells, ["cnn"], [0], (1, 3)).splitlines()
    assert "1v3" in header, header
    assert "k1-floor" in header, header
    assert "0.125" in row, row
    assert "+4.000" in row and "0.750" in row, row


def test_the_sweep_table_labels_its_spread_column_as_one_curves_own():
    """The column is the open-loop curve's between-window spread and nothing
    else; it was documented as the bar for a k-to-k difference, which it is
    not. The heading and the docstring have to agree with the arithmetic."""
    cells = {("cnn", 0): _cell([1.0], [0.5])}
    header = script.sweep_table(cells, ["cnn"], [0], (1, 3)).splitlines()[0]
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
    which can never be the floor, so the alarm would read False forever and the
    check would be silently off in the table AND in all nine records -- and the
    True path was never rendered anywhere, because every stub produced False.

    The stub collapses the SMALLEST k onto the floor and leaves the largest
    alone, so `min` and `max` are separated by the value that comes back.
    """
    _stub(monkeypatch, tmp_path, smallest_k_is_floor=True)
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    out = capsys.readouterr().out
    row = out[out.index("self-checks") :].splitlines()[2]
    assert row.split()[-1] == "True", row
    written = load_record(script.diagnostic_record_path(tmp_path, "cnn", 0))
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
    written = load_record(script.diagnostic_record_path(tmp_path, "cnn", 0))

    assert written["self_checks"] == {
        "open_loop_divergence": 0.0,
        "record_reproduction": 0.0,
        "stream_drift": 0.0,
        "smallest_k": 1,
        "smallest_k_is_bitwise_the_floor": False,
    }
    expected = {"reference_position", "shuffled_position", "floor_position",
                "persistence_position", "k1_position", "k3_position"}
    assert set(written["curves"]) == expected, written["curves"].keys()
    assert all(len(curve) == H for curve in written["curves"].values())
    # Distinct per k, so an assembly that reported one k's curve under every k's
    # name is caught here as well as in the table.
    assert written["curves"]["k1_position"] != written["curves"]["k3_position"]
    assert written["curves"]["k3_position"] == written["curves"]["reference_position"]


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
    and `action_shuffled_rollout`, and ValueError is raised from at least eight
    places under them -- the no-window guards in both `evaluate_rollout` and
    `_diagnose`, and probe.py's context/horizon/empty-path/zero-variance
    checks. Every one of them printed "MISLABELLED CHECKPOINT" and exited
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


def test_the_shuffle_section_names_the_statistic_it_actually_prints(
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
        line for line in out.splitlines() if "action-shuffled imagination" in line
    )
    assert "horizon-mean" in heading, heading
    assert "final horizon step" not in heading, heading
    block = out[out.index("action-shuffled imagination") :]
    assert "final step" in block.splitlines()[2] or "final" in block, block


def test_the_diagnostic_record_names_both_arm_and_seed_and_carries_its_device(
    monkeypatch, tmp_path
):
    """Written through `study.write_record`, so a NaN survives the round trip as
    a NaN rather than as `null` or 0.0 -- and the record carries the DEVICE and
    torch version, because the record reproduction is locked to them and nothing
    else in the study records that."""
    _stub(monkeypatch, tmp_path, changed=(False, False))
    assert script.main(_argv(tmp_path)) == script.EXIT_OK

    path = script.diagnostic_record_path(tmp_path, "cnn", 0)
    assert path.name == "diagnostic_cnn_seed0.json"
    written = load_record(path)
    assert written["arm"] == "cnn" and written["seed"] == 0
    assert written["device"] == "cpu"
    assert written["torch_version"] == torch.__version__
    assert written["episodes"]["val"] == VAL_NAMES
    assert written["windows"] == {"total": 2, "changed": 0, "episodes_changed": 0}
    # The delta is undefined when nothing changed, and must come back as NaN --
    # not as None and not as 0.0, which are different findings about a cell.
    assert all(np.isnan(v) for v in written["shuffle"]["position_delta"])
    # And it reaches disk as STRICT json: Python writes a NaN as the bare token
    # `NaN`, which is not JSON and which strict parsers reject, while Python's
    # own loader accepts it -- so a round trip through `json` alone cannot see
    # this. `write_record` writes `null` plus the map that inverts it.
    text = path.read_text()
    assert "NaN" not in text, "a bare NaN token reached the file"
    json.loads(text, parse_constant=lambda token: pytest.fail(f"non-JSON {token!r}"))
