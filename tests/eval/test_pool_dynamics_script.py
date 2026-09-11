"""The CLI that pools the nine diagnostic records into one table.

Driven from synthetic records in the post-change schema. The tests are about
what the table can mislead by: a cell missing and the table printed anyway
under nine cells' names, a record that is not the cell it was read for, a
pre-change record read as if it carried per-window series, and two cells
that do not score the same windows pooled as if they did. Every refusal is
its own exit status, disjoint from the other three scripts'.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mbfps.eval.pooling import PATHWAY_READINGS, cluster_standard_error, cluster_threshold
from mbfps.eval.study import write_record
from tests.eval.test_pooling import ARMS, EPISODES, RUNGS, SEEDS, synthetic_record


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_for_pool_tests", Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load("pool_dynamics")
diagnose = _load("diagnose_dynamics")


def _write_nine(out_dir, **overrides):
    for arm in ARMS:
        for seed in SEEDS:
            write_record(
                diagnose.diagnostic_record_path(out_dir, arm, seed),
                synthetic_record(arm, seed, **overrides),
            )


def _argv(tmp_path, **extra):
    argv = ["--out", str(tmp_path), "--bootstrap", "50"]
    for flag, value in extra.items():
        argv += [f"--{flag.replace('_', '-')}", *[str(v) for v in np.atleast_1d(value)]]
    return argv


ARM_ROWS = [
    f"{arm:<12}{rung:<11}{statistic:<16}"
    for arm in ARMS for rung in RUNGS
    for statistic in ("position delta", "angle delta", "embedding ratio")
]
"""The per-arm block's rows, TYPED: three arms x three rungs x three
statistics, in that nesting, each row beginning with these labels."""

CONTRAST_ROWS = [
    f"{rung:<11}{statistic:<16}"
    for rung in RUNGS for statistic in ("position delta", "angle delta", "embedding ratio")
]


def test_the_pool_prints_every_arm_rung_and_statistic_and_the_between_arm_block(tmp_path, capsys):
    """One report, three blocks, every row present and typed here rather
    than enumerated from the module: 27 per-arm rows, 9 contrast rows, then
    the pathway readings. The header names the pairing, the clustering, the
    fixed-seed scope ("for these three seeds, over episodes"), the family --
    EVERY z comparison the table prints: 3 arms x 3 rungs x 2 channels plus
    3 rungs x 2 channels of contrasts, 24 -- and its threshold read against
    t(G - 1) on the fixture's TWO clusters, not the normal; and the counts,
    `n_seeds=3` so a two-seed run cannot read as three. MISSING never
    appears in a complete run. One per-arm row is checked against the
    numbers computed by hand from the fixture: the seed-averaged windows,
    clustered by episode, with the naive standard error over those four
    rows labelled as not the ruler.
    """
    _write_nine(tmp_path)
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    out = capsys.readouterr().out
    assert "MISSING" not in out
    assert "frozen_ssl - random_vit" in out
    assert "paired per window" in out and "clustered by episode" in out
    assert "for these three seeds, over episodes" in out
    assert "n_seeds=3" in out and "n_windows=4" in out and "n_episodes=2" in out
    family = len(ARMS) * len(RUNGS) * 2 + len(RUNGS) * 2
    assert family == 24
    assert f"family: {family} comparisons" in out
    assert f"t({2 - 1})" in out and f"z={cluster_threshold(family, 2):.2f}" in out
    assert f"z={diagnose.family_threshold(family):.2f}" not in out, "the normal quantile was printed"
    assert "not the ruler" in out
    assert "through each arm's own probe" in out and "probe-free" in out
    lines = out.splitlines()
    for row in ARM_ROWS + CONTRAST_ROWS:
        assert any(line.startswith(row) for line in lines), row

    # cnn x constant x position: rows 100*0 + 10*seed + 2 + w, seed-averaged
    # per window -> [12, 13, 14, 15], mean 13.5, clustered over [0, 0, 1, 1].
    series = np.array([np.mean([10.0 * seed + 2 + w for seed in SEEDS]) for w in range(4)])
    se = cluster_standard_error(series, EPISODES)
    naive = series.std(ddof=1) / 2.0
    line = next(l for l in lines if l.startswith(f"{'cnn':<12}{'constant':<11}{'position delta':<16}"))
    assert f"{series.mean():+.3f}" in line and f"{se:.3f}" in line and f"{series.mean() / se:+.2f}" in line
    assert f"{naive:.3f}" in line
    assert "windows=4" in line and "clusters=2" in line and "seeds=3" in line
    assert "rows=12" not in line
    contrast = next(l for l in lines if l.startswith(f"{'constant':<11}{'position delta':<16}"))
    assert "-100.000" in contrast and "windows=4" in contrast


def test_a_missing_cell_is_refused_by_name_and_never_pooled_as_two_of_three(tmp_path, capsys):
    """The repo's burn: a table over eight cells printed under nine cells'
    names. The missing cell is named, no table row prints, and the status is
    the pool's own. An EXPLICIT two-seed plan is complete on its own terms
    and prints `n_seeds=2`, so it cannot be read as three."""
    _write_nine(tmp_path)
    diagnose.diagnostic_record_path(tmp_path, "random_vit", 2).unlink()
    assert script.main(_argv(tmp_path)) == script.EXIT_MISSING_CELL
    out = capsys.readouterr().out
    assert "random_vit seed 2" in out
    assert not any(line.startswith(ARM_ROWS[0]) for line in out.splitlines())

    assert script.main(_argv(tmp_path, seeds=[0, 1])) == script.EXIT_OK
    out = capsys.readouterr().out
    assert "n_seeds=2" in out and "n_seeds=3" not in out
    assert "seeds=2" in out and "rows=8" in out


@pytest.mark.parametrize(
    "break_one, status, named",
    [
        (lambda r: r["windows"].pop("episode"), "EXIT_STALE_RECORD", "re-run"),
        (lambda r: r.__setitem__("device", "cpu"), "EXIT_INCOMPATIBLE_RECORDS", "device"),
        (lambda r: r.__setitem__("seed", 0), "EXIT_MISLABELLED_RECORD", "seed"),
        (
            lambda r: r["noise_reference"].__setitem__("window_distance", [0.0] * 4),
            "EXIT_DEGENERATE_NOISE", "noise",
        ),
    ],
    ids=["stale", "incompatible", "mislabelled", "degenerate"],
)
def test_a_stale_incompatible_mislabelled_or_degenerate_record_is_a_named_status(
    tmp_path, capsys, break_one, status, named
):
    """Four refusals, four statuses, each exercised alone on cnn seed 1, each
    naming the cell -- and none of them a traceback (exit 1), which is what a
    KeyError on a pre-change record would otherwise be."""
    _write_nine(tmp_path)
    record = synthetic_record("cnn", 1)
    break_one(record)
    write_record(diagnose.diagnostic_record_path(tmp_path, "cnn", 1), record)
    assert script.main(_argv(tmp_path)) == getattr(script, status)
    out = capsys.readouterr().out
    assert "cnn seed 1" in out and named in out
    assert not any(line.startswith(ARM_ROWS[0]) for line in out.splitlines())


def test_a_mismatched_window_count_between_cells_is_refused_not_truncated(tmp_path, capsys):
    _write_nine(tmp_path)
    write_record(
        diagnose.diagnostic_record_path(tmp_path, "frozen_ssl", 2),
        synthetic_record("frozen_ssl", 2, episodes=(0, 0, 1, 1, 1)),
    )
    assert script.main(_argv(tmp_path)) == script.EXIT_INCOMPATIBLE_RECORDS
    out = capsys.readouterr().out
    assert "frozen_ssl seed 2" in out and "windows" in out


def test_the_treatment_and_control_flags_reach_the_contrast_and_the_bootstrap_seed_is_recorded(
    tmp_path, capsys
):
    """random_vit minus cnn is +200 on the fixture; the default pair reads
    -100, and so would cnn minus frozen_ssl -- the L1 coincidence a first
    draft of this test fell into, so the pair here is the one whose value
    differs from the default's."""
    _write_nine(tmp_path)
    assert script.main(
        _argv(tmp_path, treatment="random_vit", control="cnn", bootstrap_seed=7)
    ) == script.EXIT_OK
    out = capsys.readouterr().out
    assert "random_vit - cnn" in out and "frozen_ssl - random_vit" not in out
    assert "bootstrap_seed=7" in out and "bootstrap=50" in out
    for rung in RUNGS:
        line = next(l for l in out.splitlines() if l.startswith(f"{rung:<11}{'position delta':<16}"))
        assert "+200.000" in line, line


def test_pool_exit_statuses_are_disjoint_from_the_other_three_scripts_and_none_is_argparses_own():
    """Four scripts run one after another in the same shell and a wrapper
    reads the status. The literal codes of the other three are typed here,
    not read from a collection built from the module under test."""
    mine = {key: value for key, value in vars(script).items() if key.startswith("EXIT_")}
    assert len(set(mine.values())) == len(mine), mine
    assert 1 not in mine.values() and 2 not in mine.values()
    others = {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
    clash = (set(mine.values()) & others) - {0}
    assert not clash, clash
    for name in ("run_study", "report_study", "diagnose_dynamics"):
        theirs = {v for k, v in vars(_load(name)).items() if k.startswith("EXIT_")}
        assert theirs <= others | {0}, (name, theirs - others)
    assert mine["EXIT_OK"] == 0


def test_the_pathway_block_reads_the_three_verdicts_per_rung_and_names_the_probe_caveat(tmp_path, capsys):
    """After the contrast table, one line per rung and channel with the
    treatment verdict, the control verdict, the contrast sign and the
    reading `pathway_reading` gives them. On the fixture every arm's
    position mean is huge against a clustered SE of a few units -- but the
    fixture has TWO clusters, so the t(1) family threshold is in the
    hundreds and nothing "responds": the position line reads `neither`,
    and its channel is labelled as read through each arm's own probe. The
    embedding ratios are typed on the fixture too: nonzero on both arms
    (the bootstrap interval excludes 0), contrast unresolved, so the
    probe-free line reads `shares`. The block cannot print a reading the
    module did not compute: the readings are typed from the module's own
    vocabulary.
    """
    _write_nine(tmp_path)
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    out = capsys.readouterr().out
    block = out[out.index("--- pathway reading") :].splitlines()
    assert "treatment responds, control responds, contrast sign" in block[0] + block[1]
    body = block[2:]
    assert len(body) == len(RUNGS) * 3, body
    position = next(l for l in body if l.startswith(f"{'constant':<11}{'position delta':<16}"))
    embedding = next(l for l in body if l.startswith(f"{'constant':<11}{'embedding ratio':<16}"))
    assert position.split()[-1] == "neither" and "null" in position and "own-probe" in position, position
    assert embedding.split()[-1] == "shares" and "nonzero" in embedding and "probe-free" in embedding
    assert "through each arm's own probe" in out
    for line in body:
        assert line.split()[-1] in PATHWAY_READINGS, line


def test_the_pathway_reading_moves_with_the_pooled_numbers(tmp_path, capsys):
    """The reading is a function of the numbers, not a fixed string: with the
    control's constant-rung embedding numerator scaled up 100x over the
    treatment's, the paired ratio contrast resolves NEGATIVE while both arms
    stay nonzero, and the line reads `lacks`."""
    _write_nine(tmp_path)
    for seed in SEEDS:
        record = synthetic_record("random_vit", seed, embedding={"constant": [100.0 + w for w in range(4)]})
        write_record(diagnose.diagnostic_record_path(tmp_path, "random_vit", seed), record)
    assert script.main(_argv(tmp_path)) == script.EXIT_OK
    out = capsys.readouterr().out
    block = out[out.index("--- pathway reading") :].splitlines()
    embedding = next(l for l in block if l.startswith(f"{'constant':<11}{'embedding ratio':<16}"))
    assert embedding.split()[-1] == "lacks" and " -1 " in embedding, embedding
    shuffled = next(l for l in block if l.startswith(f"{'shuffled':<11}{'embedding ratio':<16}"))
    assert shuffled.split()[-1] == "shares", shuffled
