"""Pooling the nine diagnostic records by episode.

Pure post-processing on synthetic records with KNOWN structure, arranged
around the ways a pooled table can be wrong while every count looks right:

  * THE MEAN IS PAIRING-INVARIANT. `mean(t - c) == mean(t) - mean(c)` under
    any window alignment, and the stacked-row mean equals the mean of per-cell
    scalars when every cell has the same window count -- so a test on means
    alone cannot tell a paired series from a misaligned one, or a pool that
    read the per-window series from one that read the scalars. The per-window
    SERIES and the clustered STANDARD ERROR are the discriminating assertions.
  * THE L1 TRAP. With values that vary inside an episode, the naive and the
    clustered standard error can coincide by accident. Every clustering
    fixture here holds the values IDENTICAL within an episode and far apart
    between, so every wrong clustering -- by window, by seed, by (seed,
    episode), naive, missing correction -- lands on a different number.
  * A DROPPED CELL. This repo has been burned by tables that dropped a missing
    cell and looked complete; every refusal is a typed exception naming the
    cell and the field, raised BEFORE any statistic is computed.
"""

from collections import Counter

import numpy as np
import pytest

import mbfps.eval.pooling as pooling
from mbfps.eval.pooling import (
    CellSeries,
    DegenerateNoise,
    IncompatibleCells,
    MislabelledRecord,
    MissingCell,
    PooledMean,
    StaleRecord,
    UnpairedContrast,
    _z,
    cluster_standard_error,
    cluster_threshold,
    clustered_interval,
    load_cells,
    paired_contrast,
    paired_ratio_contrast,
    pathway_reading,
    pool_arm,
    pool_ladder,
    pool_ratio,
    read_series,
    require_compatible,
    student_t_quantile,
    unpaired_contrast,
)
from mbfps.eval.study import write_record

RUNGS = ("shuffled", "resampled", "constant")
ARMS = ("pixel_ae", "frozen_ssl", "random_vit")
SEEDS = (0, 1, 2)
EPISODES = (0, 0, 1, 1)
VAL = ("ep_000001_len00051.npz", "ep_000002_len00051.npz")


def synthetic_record(
    arm, seed, *, deltas=None, steps=None, embedding=None, noise=None,
    episodes=EPISODES, val=VAL, device="mps", torch_version="2.13.0",
    horizon=45, context=5, windows=None,
):
    """One diagnostic record in the post-change schema, with only what the
    pool reads. `deltas` maps a rung to its per-window position series (the
    angle series is a tenth of it); `steps` maps a rung to its changed mask
    as step counts; `embedding` maps a rung to its per-window numerator;
    `noise` is the noise reference's per-window series. Defaults are
    distinct per (arm, seed, rung, window): `100 * arm + 10 * seed + rung +
    w`, so a row that came from the wrong cell is caught by value.
    """
    n = len(episodes) if windows is None else windows
    base = 100.0 * ARMS.index(arm) + 10.0 * seed
    deltas = dict(deltas or {})
    steps = dict(steps or {})
    embedding = dict(embedding or {})
    noise_series = [2.0 + 0.1 * w for w in range(n)] if noise is None else list(noise)

    def rung_block(rung, index):
        series = list(deltas.get(rung, [base + index + w for w in range(n)]))
        changed = list(steps.get(rung, [45] * n))
        numerator = embedding.get(rung, [1.0 + index + 0.1 * w for w in range(n)])
        block = {
            "windows_changed": int(sum(1 for s in changed if s > 0)),
            "window_steps_changed": [int(s) for s in changed],
            "position": {"delta_mean": 0.0, "window_delta_mean": [float(v) for v in series]},
            "angle": {"delta_mean": 0.0, "window_delta_mean": [float(v) / 10.0 for v in series]},
            "embedding": None if numerator is None else {
                "window_distance": [float(v) for v in numerator],
                "median": float(np.median(numerator)),
                "noise_median": float(np.median(noise_series)),
                # The record's own policy: NaN, never inf, over a zero ruler.
                "ratio_of_medians": (
                    float("nan") if np.median(noise_series) == 0.0
                    else float(np.median(numerator) / np.median(noise_series))
                ),
            },
        }
        if rung == "constant":
            block["contrast"] = [3, 0]
            block["held"] = {}
        return block

    return {
        "arm": arm, "seed": seed, "context": context, "horizon": horizon,
        "device": device, "torch_version": torch_version,
        "episodes": {"val": list(val)},
        "self_checks": {"stream_drift": 0.0},
        "windows": {"total": n, "episode": None if episodes is None else [int(e) for e in episodes]},
        "ladder": {"rungs": list(RUNGS)},
        "interventions": {rung: rung_block(rung, i) for i, rung in enumerate(RUNGS)},
        "noise_reference": {
            "window_distance": [float(v) for v in noise_series],
            "median": float(np.median(noise_series)),
        },
    }


def series(arm, seed, delta, *, changed=None, episodes=EPISODES, embedding=None, noise=None,
           rung="constant", channel="position"):
    """A `CellSeries` straight from arrays, for the statistics tests."""
    return read_series(
        synthetic_record(
            arm, seed, deltas={rung: delta},
            steps=None if changed is None else {rung: [45 if c else 0 for c in changed]},
            embedding=None if embedding is None else {rung: embedding},
            noise=noise, episodes=episodes,
        ),
        rung, channel,
    )


# ---------------------------------------------------------------------------
# Reading a record.
# ---------------------------------------------------------------------------


def test_read_series_carries_the_cells_identity_and_its_per_window_series():
    cell = read_series(synthetic_record("frozen_ssl", 1), "resampled", "angle")
    assert isinstance(cell, CellSeries)
    assert (cell.arm, cell.seed, cell.rung, cell.channel) == ("frozen_ssl", 1, "resampled", "angle")
    np.testing.assert_array_equal(cell.delta, np.array([111.0, 112.0, 113.0, 114.0]) / 10.0)
    np.testing.assert_array_equal(cell.changed, [True] * 4)
    np.testing.assert_array_equal(cell.episode, EPISODES)
    np.testing.assert_allclose(cell.embedding, [2.0, 2.1, 2.2, 2.3])
    np.testing.assert_allclose(cell.noise, [2.0, 2.1, 2.2, 2.3])
    assert (cell.device, cell.torch_version, cell.horizon, cell.context) == ("mps", "2.13.0", 45, 5)
    assert cell.val == VAL


@pytest.mark.parametrize(
    "strip, match",
    [
        (lambda r: r["windows"].pop("episode"), "windows.episode"),
        (lambda r: r["windows"].__setitem__("episode", None), "windows.episode"),
        (lambda r: r["interventions"]["constant"]["position"].pop("window_delta_mean"), "constant"),
        (lambda r: r["interventions"]["constant"].pop("window_steps_changed"), "constant"),
        (lambda r: r["interventions"].pop("constant"), "constant"),
        (lambda r: r.pop("noise_reference"), "noise_reference"),
    ],
    ids=["no-episode-index", "null-episode-index", "no-window-delta", "no-changed-mask",
         "no-rung", "no-noise-reference"],
)
def test_a_record_without_the_per_window_series_is_refused_by_name_and_never_read_naively(
    strip, match
):
    """Every record produced before this change is exactly the first shape:
    it carries `delta_mean` but no `window_delta_mean` and no episode index.
    Falling back to the scalars, or clustering on `range(n)`, would print a
    pooled table over numbers that cannot be pooled. Refused naming the cell,
    the field and the regeneration command."""
    record = synthetic_record("random_vit", 2)
    strip(record)
    with pytest.raises(StaleRecord, match=match) as raised:
        read_series(record, "constant", "position")
    assert "random_vit seed 2" in str(raised.value)
    assert "diagnose_dynamics.py" in str(raised.value)


def test_a_record_whose_series_disagree_in_length_is_refused():
    record = synthetic_record("pixel_ae", 0)
    record["interventions"]["shuffled"]["position"]["window_delta_mean"] = [1.0, 2.0]
    with pytest.raises(StaleRecord, match="shuffled"):
        read_series(record, "shuffled", "position")


@pytest.mark.parametrize(
    "mutate, field",
    [
        (lambda r: r["windows"].__setitem__("total", 5), "windows"),
        (lambda r: r["windows"].__setitem__("episode", [0, 1, 1, 1]), "episode"),
        (lambda r: r["episodes"].__setitem__("val", ["other.npz", "b.npz"]), "episodes.val"),
        (lambda r: r.__setitem__("device", "cpu"), "device"),
        (lambda r: r.__setitem__("torch_version", "2.12.0"), "torch_version"),
        (lambda r: r.__setitem__("horizon", 30), "horizon"),
        (lambda r: r.__setitem__("context", 3), "context"),
    ],
    ids=["window-count", "episode-index", "split", "device", "torch", "horizon", "context"],
)
def test_two_cells_that_do_not_score_the_same_windows_are_refused_naming_both(mutate, field):
    """Pooling is only valid across cells that score the SAME windows under
    the SAME protocol on the SAME device -- the addendum measured the deltas
    moving across devices by more than their standard error. Each condition
    is triggered alone; the refusal names both cells and the field. A
    mismatched window count in particular is refused, never truncated to the
    shorter cell."""
    other = synthetic_record("pixel_ae", 1)
    mutate(other)
    cells = [read_series(synthetic_record("pixel_ae", 0), "constant", "position")]
    if field == "windows":
        other["windows"]["episode"] = [0, 0, 1, 1, 1]
        for rung in RUNGS:
            block = other["interventions"][rung]
            block["window_steps_changed"] = [45] * 5
            block["position"]["window_delta_mean"] = [1.0] * 5
            block["angle"]["window_delta_mean"] = [0.1] * 5
            block["embedding"]["window_distance"] = [1.0] * 5
        other["noise_reference"]["window_distance"] = [2.0] * 5
    cells.append(read_series(other, "constant", "position"))
    with pytest.raises(IncompatibleCells, match=field) as raised:
        require_compatible(cells)
    assert "pixel_ae seed 0" in str(raised.value) and "pixel_ae seed 1" in str(raised.value)
    with pytest.raises(IncompatibleCells):
        pool_arm(cells)


def test_cells_of_different_arms_or_the_same_cell_twice_are_refused_before_pooling():
    """What ELSE `require_compatible` accepted: three cells from three arms
    pooled under the first one's name, the same cell three times pooled as
    twelve rows, and a between-arm contrast of an arm with itself. Each is
    refused by name, and none by the script's argparse alone -- the module's
    contract is that the refusal happens here."""
    mixed = [series(arm, seed, [1.0, 2.0, 3.0, 4.0]) for seed, arm in enumerate(ARMS)]
    with pytest.raises(IncompatibleCells, match="arm") as raised:
        require_compatible(mixed)
    assert "pixel_ae seed 0" in str(raised.value) and "frozen_ssl seed 1" in str(raised.value)
    with pytest.raises(IncompatibleCells, match="arm"):
        pool_arm(mixed)
    twice = [series("pixel_ae", 0, [1.0, 2.0, 3.0, 4.0]) for _ in SEEDS]
    with pytest.raises(IncompatibleCells, match="pixel_ae seed 0 .*more than once") as raised:
        require_compatible(twice)
    with pytest.raises(IncompatibleCells, match="more than once"):
        pool_ratio(twice, bootstrap=10)
    same = [series("pixel_ae", seed, [1.0, 2.0, 3.0, 4.0]) for seed in SEEDS]
    with pytest.raises(IncompatibleCells, match="pixel_ae.*against itself"):
        paired_contrast(same, [series("pixel_ae", seed, [1.0, 2.0, 3.0, 4.0]) for seed in SEEDS])
    with pytest.raises(IncompatibleCells, match="against itself"):
        paired_ratio_contrast(same, same, bootstrap=10)


def test_pool_ladder_refuses_a_treatment_or_control_that_is_not_among_the_arms():
    records = {(a, s): synthetic_record(a, s) for a in ARMS for s in SEEDS}
    with pytest.raises(MissingCell, match="treatment 'nonesuch'"):
        pool_ladder(records, ARMS, SEEDS, RUNGS, "nonesuch", "random_vit", bootstrap=10)
    with pytest.raises(MissingCell, match="control 'pixel_ae'"):
        pool_ladder(records, ("frozen_ssl", "random_vit"), SEEDS, RUNGS, "frozen_ssl", "pixel_ae", bootstrap=10)


def test_load_cells_reads_every_planned_cell_and_refuses_gaps_and_mislabels(tmp_path):
    """M8 for the pool: the record read must be the cell reported. A swapped
    pair of files would otherwise pool one cell under another's name with
    every count right. And a missing cell is named, never skipped -- while an
    explicit two-seed plan is complete on its own terms."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "diagnose_for_pool", Path(__file__).resolve().parents[2] / "scripts" / "diagnose_dynamics.py"
    )
    diagnose = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnose)
    path = lambda arm, seed: diagnose.diagnostic_record_path(tmp_path, arm, seed)  # noqa: E731

    for arm in ARMS:
        for seed in SEEDS:
            write_record(path(arm, seed), synthetic_record(arm, seed))
    records = load_cells(tmp_path, ARMS, SEEDS)
    assert set(records) == {(a, s) for a in ARMS for s in SEEDS}
    assert records[("random_vit", 2)]["arm"] == "random_vit"

    path("random_vit", 2).unlink()
    with pytest.raises(MissingCell, match="random_vit seed 2"):
        load_cells(tmp_path, ARMS, SEEDS)
    # An explicit two-seed plan is allowed: nothing is missing from it.
    assert set(load_cells(tmp_path, ARMS, (0, 1))) == {(a, s) for a in ARMS for s in (0, 1)}

    write_record(path("random_vit", 2), synthetic_record("random_vit", 0))
    with pytest.raises(MislabelledRecord, match="seed") as raised:
        load_cells(tmp_path, ARMS, SEEDS)
    assert "random_vit seed 2" in str(raised.value) and "seed=0" in str(raised.value)
    write_record(path("random_vit", 2), synthetic_record("pixel_ae", 2))
    with pytest.raises(MislabelledRecord, match="arm"):
        load_cells(tmp_path, ARMS, SEEDS)


# ---------------------------------------------------------------------------
# Per arm x rung: the stacked-seed mean and its episode-clustered ruler.
# ---------------------------------------------------------------------------


def test_pool_arm_averages_the_seeds_per_window_and_clusters_by_episode():
    """Three seeds of [0, 0, 40, 40] + seed over episodes [0, 0, 1, 1].

    Four seed-averaged windows [1, 1, 41, 41], two clusters, mean 21. The
    values are identical within an episode, so the clustered standard error
    is EXACTLY 20 (cluster residual sums -40 and +40, correction 2) and every
    wrong clustering is a different number: by window 11.55, by seed 0.0,
    without the correction 14.14 -- and STACKING the twelve rows and
    clustering by episode gives the same 20, which is why the estimator is
    described as seed-averaged rather than as 687 rows: the naive standard
    error over the stacked rows (6.035) triple-counts and is not printed.
    The one printed is over the four seed-averaged rows (11.547), and it is
    still not the ruler. The seed offsets make the mean (21) differ from
    seed 0's (20), so a pool that read one seed is caught too.
    """
    cells = [series("pixel_ae", seed, np.array([0.0, 0.0, 40.0, 40.0]) + seed) for seed in SEEDS]
    pooled = pool_arm(cells)
    assert (pooled.arm, pooled.rung, pooled.channel, pooled.seeds) == ("pixel_ae", "constant", "position", (0, 1, 2))
    assert pooled.windows == 4 and pooled.clusters == 2 and pooled.windows_excluded == 0
    assert pooled.mean == 21.0
    assert pooled.se == pytest.approx(20.0)
    np.testing.assert_array_equal(pooled.series, [1.0, 1.0, 41.0, 41.0])
    stacked = np.concatenate([c.delta for c in cells])
    assert pooled.se == pytest.approx(cluster_standard_error(stacked, np.tile(EPISODES, 3)))
    assert pooled.se_independent == pytest.approx(pooled.series.std(ddof=1) / 2.0)
    assert pooled.se_independent == pytest.approx(11.547, abs=1e-3)
    assert pooled.se_independent != pytest.approx(6.035, abs=1e-3), "the naive se is over the stacked rows"
    assert pooled.se > pooled.se_independent
    assert pooled.z == pytest.approx(21.0 / 20.0)


def test_the_pooled_rows_exclude_unchanged_windows_and_the_z_never_falls_back_to_the_naive_se():
    """A window a rung left unchanged in ANY seed leaves the pool -- its
    exact zero there is not a measurement, and averaging it into the seed
    mean would dilute the window toward null -- given an absurd value here
    so inclusion is loud, both when every seed left it and when ONE did.
    Below two clusters the clustered standard error is NaN and so is z:
    never the naive ruler under the clustered name."""
    cells = [series("pixel_ae", seed, [1.0, 1.0, 1.0, 1e6], changed=[True, True, True, False])
             for seed in SEEDS]
    pooled = pool_arm(cells)
    assert pooled.windows == 3 and pooled.windows_excluded == 1 and pooled.excluded_windows == (3,)
    assert pooled.mean == 1.0
    one_seed = [
        series("pixel_ae", seed, [1.0, 1e6 if seed == 1 else 1.0, 1.0, 1.0],
               changed=[True, seed != 1, True, True])
        for seed in SEEDS
    ]
    pooled = pool_arm(one_seed)
    assert pooled.windows == 3 and pooled.excluded_windows == (1,) and pooled.mean == 1.0

    one_episode = [series("pixel_ae", seed, [1.0, 2.0, 3.0, 4.0], episodes=(0, 0, 0, 0)) for seed in SEEDS]
    pooled = pool_arm(one_episode)
    assert pooled.clusters == 1
    assert np.isnan(pooled.se) and np.isnan(pooled.z)
    assert np.isfinite(pooled.se_independent)


def test_the_pooled_z_is_signed_and_defined_over_a_zero_ruler():
    """`z = mean / se`, SIGNED -- the between-arm contrast reads it -- with the
    script's own policy on a zero ruler: 0 / 0 is 0, x / 0 is +-inf."""
    assert pooling._z(2.0, 4.0) == 0.5 and pooling._z(-2.0, 4.0) == -0.5
    assert pooling._z(0.0, 0.0) == 0.0
    assert pooling._z(3.0, 0.0) == float("inf") and pooling._z(-3.0, 0.0) == float("-inf")
    assert np.isnan(pooling._z(3.0, float("nan")))


# ---------------------------------------------------------------------------
# Between arms: paired per window, clustered by episode.
# ---------------------------------------------------------------------------


def test_the_paired_contrast_is_zero_on_identical_arms_and_exactly_the_injected_difference():
    """Identical arms: series all 0.0 bitwise, mean 0, se 0, z 0 -- not NaN.

    Then the treatment carries `d = [1, 2, 3, 4]` in the SEED MEAN only (seed
    values d + 1, d, d - 1), over episodes [0, 0, 1, 1]: the paired series is
    bitwise d, the mean 2.5, and the clustered standard error exactly 1.0
    (residuals -1.5, -0.5, +0.5, +1.5; cluster sums -2, +2; correction 2).
    Seed 0 alone would give a mean of 3.5; differencing seed by seed before
    averaging would give twelve rows and another ruler -- asserted through
    `windows == 4` and the series; a reversed or sorted treatment gives a
    different series though the same mean, which is why the mean is not the
    assertion that carries this test. Direction: treatment minus control.
    """
    control = [series("random_vit", seed, [10.0, 20.0, 30.0, 40.0]) for seed in SEEDS]
    same = [series("frozen_ssl", seed, [10.0, 20.0, 30.0, 40.0]) for seed in SEEDS]
    contrast = paired_contrast(same, control)
    assert (contrast.treatment, contrast.control) == ("frozen_ssl", "random_vit")
    np.testing.assert_array_equal(contrast.series, 0.0)
    assert contrast.mean == 0.0 and contrast.se == 0.0 and contrast.z == 0.0

    d = np.array([1.0, 2.0, 3.0, 4.0])
    treatment = [
        series("frozen_ssl", seed, np.array([10.0, 20.0, 30.0, 40.0]) + d + (1, 0, -1)[seed])
        for seed in SEEDS
    ]
    contrast = paired_contrast(treatment, control)
    np.testing.assert_array_equal(contrast.series, d)
    assert contrast.windows == 4 and contrast.windows_excluded == 0 and contrast.clusters == 2
    assert contrast.mean == 2.5
    assert contrast.se == pytest.approx(1.0)
    assert contrast.z == pytest.approx(2.5)


def test_the_paired_contrast_drops_a_window_unchanged_in_any_cell_of_either_arm_and_says_which():
    """A window one control seed left unchanged leaves the pairing entirely
    -- its exact zero there is not a measurement -- with the count and the
    index reported. Inert on the shipped split, where every window changes
    under every rung, so only this fixture exercises it."""
    control = [
        series("random_vit", seed, [10.0, 20.0, 1e6 if seed == 1 else 30.0, 40.0],
               changed=[True, True, seed != 1, True])
        for seed in SEEDS
    ]
    treatment = [series("frozen_ssl", seed, [11.0, 22.0, 33.0, 44.0]) for seed in SEEDS]
    contrast = paired_contrast(treatment, control)
    assert contrast.windows == 3 and contrast.windows_excluded == 1
    assert contrast.excluded_windows == (2,)
    np.testing.assert_array_equal(contrast.series, [1.0, 2.0, 4.0])
    assert contrast.mean == pytest.approx(7.0 / 3.0)


# ---------------------------------------------------------------------------
# The embedding ratio: pooled medians with an episode-cluster bootstrap.
# ---------------------------------------------------------------------------


def test_pool_ratio_is_the_ratio_of_pooled_medians_with_a_cluster_bootstrap_interval():
    """Rows where the numerator is exactly twice the (varied) noise: ratio
    2.0, and every episode resample gives 2.0 too, so the interval is (2, 2)
    with a bootstrap standard error of 0 -- but a ROW bootstrap would give
    that as well, which is why the second fixture exists.

    Second fixture: episodes [0, 0, 1, 1] with per-window numerators
    [1, 3, 1, 3] in every seed and noise 1. Both episodes carry the same
    multiset, so EVERY episode resample yields equal counts of 1s and 3s and
    a median of exactly 2: the interval is exactly (2.0, 2.0). A bootstrap
    over rows draws unequal counts and its interval spans 1 to 3. Then a
    varied fixture pins the point estimate by hand -- on rows NORMALISED by
    each cell's own noise median, see the scale test below -- and the
    interval's shape.
    """
    noise = [1.0, 2.0, 4.0, 8.0]
    doubled = [series("pixel_ae", seed, [0.0] * 4, embedding=[2.0 * v for v in noise], noise=noise)
               for seed in SEEDS]
    pooled = pool_ratio(doubled, bootstrap=200, seed=0)
    assert (pooled.arm, pooled.rung, pooled.rows, pooled.clusters) == ("pixel_ae", "constant", 12, 2)
    assert pooled.ratio == 2.0
    assert (pooled.ci_low, pooled.ci_high) == (2.0, 2.0) and pooled.bootstrap_se == 0.0
    assert (pooled.bootstrap, pooled.seed) == (200, 0)

    symmetric = [series("pixel_ae", seed, [0.0] * 4, embedding=[1.0, 3.0, 1.0, 3.0], noise=[1.0] * 4)
                 for seed in SEEDS]
    pooled = pool_ratio(symmetric, bootstrap=500, seed=3)
    assert pooled.ratio == 2.0
    assert (pooled.ci_low, pooled.ci_high) == (2.0, 2.0), "the resample is not by episode"

    varied = [
        series("pixel_ae", seed, [0.0] * 4, embedding=[1.0 + seed, 5.0, 9.0 + seed, 100.0],
               noise=[2.0, 4.0, 4.0 + seed, 8.0], episodes=(0, 0, 1, 2))
        for seed in SEEDS
    ]
    scale = [np.median(c.noise) for c in varied]          # 4, 4.5, 5: they differ
    assert len(set(scale)) == 3
    numerators = np.concatenate([c.embedding / m for c, m in zip(varied, scale)])
    noises = np.concatenate([c.noise / m for c, m in zip(varied, scale)])
    pooled = pool_ratio(varied, bootstrap=2000, seed=0)
    assert pooled.ratio == np.median(numerators) / np.median(noises)
    assert pooled.numerator_median == np.median(numerators)
    assert pooled.noise_median == np.median(noises)
    raw = np.concatenate([c.embedding for c in varied]), np.concatenate([c.noise for c in varied])
    assert pooled.ratio != np.median(raw[0]) / np.median(raw[1]), "the rows were pooled raw"
    assert pooled.ci_low <= pooled.ratio <= pooled.ci_high
    assert pooled.ci_low < pooled.ci_high and pooled.bootstrap_se > 0.0
    again = pool_ratio(varied, bootstrap=2000, seed=0)
    assert (again.ci_low, again.ci_high, again.bootstrap_se) == (
        pooled.ci_low, pooled.ci_high, pooled.bootstrap_se
    )
    # A different seed draws different episodes. Over three episodes the
    # replicate distribution is discrete and its percentiles can coincide
    # across seeds, so the spread of the replicates is what is compared.
    other = pool_ratio(varied, bootstrap=2000, seed=1)
    assert other.bootstrap_se != pooled.bootstrap_se


def _scaled_cells(scales, *, arm="pixel_ae", episodes=EPISODES):
    """Three cells whose per-cell ratios of medians are 0.5, 0.5 and 2.0
    whatever `scales` says: seed s's rows are the same shape times
    `scales[s]`, standing in for an embedding head whose scale differs
    between independently trained seeds (70x between arms on the shipped
    checkpoints)."""
    noise = np.array([1.0, 2.0, 4.0, 8.0])
    numerator = {0: noise / 2.0, 1: noise / 2.0, 2: noise * 2.0}
    return [
        series(arm, seed, [0.0] * 4, embedding=numerator[seed] * scale, noise=noise * scale,
               episodes=episodes)
        for seed, scale in zip(SEEDS, scales)
    ]


def test_the_pooled_ratio_is_invariant_to_each_seeds_embedding_scale():
    """Stacking RAW distances from three independently trained heads and
    taking one median makes the pooled number a function of which seed's
    rows land in the middle: with the per-cell ratios held at (0.5, 0.5,
    2.0), raw pooling reads 1.0 at equal scales and moves with the scales.
    So each cell is normalised by ITS OWN noise median before stacking, and
    the pooled ratio, both medians, the interval and the paired between-arm
    contrast are then BITWISE the same under scales of 1, 128 and 16384 --
    powers of two, so the scaling is exact in floating point and the claim
    can be bitwise rather than approximate. The unnormalised pool is
    asserted to differ on the same rows, so the invariance is not a
    property of the fixture.

    Typed by hand: normalised noise rows are [1, 2, 4, 8] / 3 in every cell,
    numerators [1, 2, 4, 8] / 6 in seeds 0-1 and [1, 2, 4, 8] * 2 / 3 in
    seed 2; sorted, the twelve numerators' middle two are both 2/3 and the
    noise rows' are 2/3 and 4/3, so the pooled medians are 2/3 over 1.0 --
    ratio 2/3, the number raw pooling gives at EQUAL scales and leaves as
    soon as the scales differ.
    """
    flat = pool_ratio(_scaled_cells((1.0, 1.0, 1.0)), bootstrap=300, seed=0)
    scaled = pool_ratio(_scaled_cells((1.0, 128.0, 16384.0)), bootstrap=300, seed=0)
    assert flat.ratio == 2.0 / 3.0 and flat.noise_median == 1.0
    assert (scaled.ratio, scaled.numerator_median, scaled.noise_median) == (
        flat.ratio, flat.numerator_median, flat.noise_median
    )
    assert (scaled.ci_low, scaled.ci_high, scaled.bootstrap_se) == (flat.ci_low, flat.ci_high, flat.bootstrap_se)
    raw = lambda cells: np.median(np.concatenate([c.embedding for c in cells])) / np.median(  # noqa: E731
        np.concatenate([c.noise for c in cells])
    )
    assert raw(_scaled_cells((1.0, 128.0, 16384.0))) != raw(_scaled_cells((1.0, 1.0, 1.0)))

    control = _scaled_cells((1.0, 1.0, 1.0), arm="random_vit", episodes=(0, 0, 1, 2))
    same = paired_ratio_contrast(
        _scaled_cells((1.0, 1.0, 1.0), arm="frozen_ssl", episodes=(0, 0, 1, 2)), control,
        bootstrap=300, seed=0,
    )
    rescaled = paired_ratio_contrast(
        _scaled_cells((16384.0, 1.0, 128.0), arm="frozen_ssl", episodes=(0, 0, 1, 2)), control,
        bootstrap=300, seed=0,
    )
    assert (same.treatment_ratio, same.contrast, same.ci_low, same.ci_high) == (
        rescaled.treatment_ratio, rescaled.contrast, rescaled.ci_low, rescaled.ci_high
    )


def test_a_bootstrap_replicate_whose_noise_median_is_zero_is_refused_rather_than_a_nan_percentile():
    """Episodes [0, 0, 1, 1] with noise [0, 0, 3, 3]: the cell's own median
    is 1.5 and passes the per-cell check, but a replicate that draws episode
    0 twice divides by zero, and a NaN in the replicates reaches the
    percentile as a NaN interval printed beside a finite ratio. Refused, as
    the same `DegenerateNoise`, naming the cell."""
    cells = [series("pixel_ae", seed, [0.0] * 4, embedding=[1.0] * 4, noise=[0.0, 0.0, 3.0, 3.0])
             for seed in SEEDS]
    with pytest.raises(DegenerateNoise, match="replicate") as raised:
        pool_ratio(cells, bootstrap=50, seed=0)
    assert "pixel_ae" in str(raised.value)
    control = [series("random_vit", seed, [0.0] * 4, embedding=[1.0] * 4, noise=[2.0] * 4)
               for seed in SEEDS]
    treatment = [series("frozen_ssl", seed, [0.0] * 4, embedding=[1.0] * 4, noise=[0.0, 0.0, 3.0, 3.0])
                 for seed in SEEDS]
    with pytest.raises(DegenerateNoise, match="replicate"):
        paired_ratio_contrast(treatment, control, bootstrap=50, seed=0)


def test_pool_ratio_excludes_unchanged_windows_and_refuses_a_cell_whose_noise_measured_no_spread():
    cells = [
        series("pixel_ae", seed, [0.0] * 4, changed=[True, True, True, False],
               embedding=[1.0, 1.0, 1.0, 1e6], noise=[2.0, 2.0, 2.0, 2.0])
        for seed in SEEDS
    ]
    pooled = pool_ratio(cells, bootstrap=50, seed=0)
    assert pooled.rows == 9 and pooled.ratio == 0.5

    degenerate = [
        series("pixel_ae", seed, [0.0] * 4, embedding=[1.0] * 4, noise=[0.0] * 4 if seed == 1 else [2.0] * 4)
        for seed in SEEDS
    ]
    with pytest.raises(DegenerateNoise, match="pixel_ae seed 1"):
        pool_ratio(degenerate, bootstrap=50, seed=0)
    unmeasured = [
        read_series(synthetic_record("pixel_ae", seed, embedding={"constant": None}), "constant")
        for seed in SEEDS
    ]
    with pytest.raises(StaleRecord, match="embedding"):
        pool_ratio(unmeasured, bootstrap=50, seed=0)


def test_the_paired_ratio_contrast_is_zero_on_identical_arms_and_resamples_both_arms_in_step():
    """Identical arms with VARIED windows: the contrast is 0.0 and its
    interval exactly (0.0, 0.0), because every replicate applies the same
    episode draw to both arms -- independent draws would give a nonzero
    spread on identical arms. Scaling the treatment's numerator by 3 gives
    exactly `2 x R_control`, which is what excludes a bootstrap that never
    ran."""
    numerator = [1.0, 5.0, 9.0, 100.0]
    noise = [2.0, 4.0, 4.0, 8.0]
    control = [series("random_vit", seed, [0.0] * 4, embedding=numerator, noise=noise,
                      episodes=(0, 0, 1, 2)) for seed in SEEDS]
    same = [series("frozen_ssl", seed, [0.0] * 4, embedding=numerator, noise=noise,
                   episodes=(0, 0, 1, 2)) for seed in SEEDS]
    contrast = paired_ratio_contrast(same, control, bootstrap=500, seed=0)
    assert (contrast.treatment, contrast.control, contrast.rung) == ("frozen_ssl", "random_vit", "constant")
    assert contrast.contrast == 0.0
    assert (contrast.ci_low, contrast.ci_high) == (0.0, 0.0) and contrast.bootstrap_se == 0.0
    assert contrast.windows == 4 and contrast.clusters == 3

    scaled = [series("frozen_ssl", seed, [0.0] * 4, embedding=[3.0 * v for v in numerator],
                     noise=noise, episodes=(0, 0, 1, 2)) for seed in SEEDS]
    contrast = paired_ratio_contrast(scaled, control, bootstrap=500, seed=0)
    assert contrast.control_ratio == np.median(numerator) / np.median(noise)
    assert contrast.treatment_ratio == pytest.approx(3.0 * contrast.control_ratio)
    assert contrast.contrast == pytest.approx(2.0 * contrast.control_ratio)
    assert contrast.ci_low <= contrast.contrast <= contrast.ci_high
    # Three episodes carrying different windows: the replicates vary, so a
    # bootstrap that never ran (a point interval) is caught here.
    assert contrast.ci_low < contrast.ci_high and contrast.bootstrap_se > 0.0


# ---------------------------------------------------------------------------
# The whole ladder.
# ---------------------------------------------------------------------------


def test_pool_ladder_refuses_a_missing_cell_by_name_before_computing_anything():
    records = {(a, s): synthetic_record(a, s) for a in ARMS for s in SEEDS}
    del records[("random_vit", 2)]
    with pytest.raises(MissingCell, match="random_vit seed 2"):
        pool_ladder(records, ARMS, SEEDS, RUNGS, "frozen_ssl", "random_vit", bootstrap=10)


def test_each_pooled_row_is_the_mean_over_exactly_that_arms_three_cells():
    """Every cell's series is unique -- `100 * arm + 10 * seed + rung + w` --
    so each arm's pooled mean is typed from that formula and the three arms
    differ; a pool of all nine under every arm, or of another arm's seeds,
    lands on a different number."""
    records = {(a, s): synthetic_record(a, s) for a in ARMS for s in SEEDS}
    pooled = pool_ladder(records, ARMS, SEEDS, RUNGS, "frozen_ssl", "random_vit", bootstrap=10)
    for index, arm in enumerate(ARMS):
        for rung_index, rung in enumerate(RUNGS):
            entry = pooled["arms"][(arm, rung, "position")]
            assert entry.mean == pytest.approx(100.0 * index + 10.0 + rung_index + 1.5), (arm, rung)
            assert entry.windows == 4 and entry.seeds == SEEDS
            assert pooled["arms"][(arm, rung, "angle")].mean == pytest.approx(entry.mean / 10.0)
            assert pooled["ratios"][(arm, rung)].rows == 12
    # frozen_ssl sits at 100 + ..., random_vit at 200 + ...: treatment minus
    # control is -100 exactly, and the sign is the assertion.
    contrast = pooled["contrasts"][("constant", "position")]
    assert contrast.mean == pytest.approx(-100.0)
    assert set(pooled["contrasts"]) == {(r, c) for r in RUNGS for c in ("position", "angle")}
    assert set(pooled["ratio_contrasts"]) == set(RUNGS)
    assert pooled["ratio_contrasts"]["constant"].contrast == 0.0


# ---------------------------------------------------------------------------
# The ruler's reference distribution, and the reading of the three z's.
# ---------------------------------------------------------------------------


def test_the_cluster_robust_threshold_is_a_student_t_quantile_on_the_cluster_count():
    """A cluster-robust z with 24 clusters is read against t(23), not the
    normal: the effective replication is the cluster count. Pinned against
    the standard table -- t(23) at 0.975 is 2.0687, at 0.995 2.8073, at
    0.9995 3.7676; t(1) at 0.975 is 12.706 -- and against the normal in the
    limit. Then the family threshold: Bonferroni over the family at G - 1
    degrees of freedom, so family 1 at G = 24 is 2.0687 (not 1.96), and it
    grows with the family and shrinks with the clusters. Below two clusters
    there is no distribution to read against: NaN, never the normal.
    """
    assert student_t_quantile(0.975, 23) == pytest.approx(2.068658, abs=1e-5)
    assert student_t_quantile(0.995, 23) == pytest.approx(2.807336, abs=1e-5)
    assert student_t_quantile(0.9995, 23) == pytest.approx(3.767627, abs=1e-5)
    assert student_t_quantile(0.975, 1) == pytest.approx(12.7062, abs=1e-3)
    assert student_t_quantile(0.975, 10**6) == pytest.approx(1.959964, abs=1e-4)
    assert student_t_quantile(0.5, 23) == 0.0
    assert student_t_quantile(0.025, 23) == pytest.approx(-2.068658, abs=1e-5)
    # Near the median at LARGE df the incomplete beta is evaluated at x
    # close to 1, where the continued fraction only converges through the
    # symmetry `I_x(a, b) = 1 - I_{1-x}(b, a)`; measured, dropping the flip
    # moves the CDF there by up to 0.58 while every table value above
    # still passes. The Cauchy closed form (df = 1) covers the other end.
    from math import pi, tan
    from statistics import NormalDist

    assert student_t_quantile(0.504, 1000) == pytest.approx(NormalDist().inv_cdf(0.504), abs=2e-5)
    assert student_t_quantile(0.51, 100) == pytest.approx(NormalDist().inv_cdf(0.51), abs=1e-4)
    for p in (0.6, 0.75, 0.9):
        assert student_t_quantile(p, 1) == pytest.approx(tan(pi * (p - 0.5)), abs=1e-9)

    assert cluster_threshold(1, 24) == pytest.approx(2.068658, abs=1e-5)
    assert cluster_threshold(1, 24) == student_t_quantile(0.975, 23)
    assert cluster_threshold(6, 24) == student_t_quantile(1.0 - 0.025 / 6, 23)
    assert cluster_threshold(6, 24) == pytest.approx(2.89, abs=0.01)
    assert cluster_threshold(54, 24) == pytest.approx(3.80, abs=0.01)
    assert cluster_threshold(24, 24) > cluster_threshold(6, 24) > cluster_threshold(1, 24)
    assert cluster_threshold(6, 240) < cluster_threshold(6, 24)
    assert np.isnan(cluster_threshold(6, 1))
    with pytest.raises(ValueError, match="family"):
        cluster_threshold(0, 24)
    with pytest.raises(ValueError):
        student_t_quantile(1.0, 23)


@pytest.mark.parametrize(
    "treatment, control, contrast, expected",
    [
        (True, True, 0, "shares"),
        (True, True, +1, "exceeds"),
        (True, False, +1, "exceeds"),
        (True, True, -1, "lacks"),
        (False, True, -1, "lacks"),
        (False, False, 0, "neither"),
        (True, False, 0, "inconclusive"),
        (False, True, 0, "inconclusive"),
        (False, False, -1, "inconclusive"),
        (False, False, +1, "inconclusive"),
        (False, True, +1, "inconclusive"),
        (True, False, -1, "inconclusive"),
    ],
)
def test_the_pathway_reading_is_a_function_of_the_three_verdicts_and_not_of_the_contrast_alone(
    treatment, control, contrast, expected
):
    """Share / exceed / lack is a reading of (treatment responds, control
    responds, contrast sign in {-1, 0, +1}), never of the contrast alone: a
    null contrast between two arms that both lack the pathway is "neither",
    not "shares". All twelve cells are typed here. "shares" is both arms
    responding with the contrast unresolved; "exceeds" is a resolved
    positive contrast on a responding treatment; "lacks" is a resolved
    negative contrast beside a responding control; "neither" is nothing
    responding and nothing resolved; every other combination is
    "inconclusive" -- a resolved contrast whose favoured arm did not itself
    respond, or one arm responding with the difference unresolved."""
    assert pathway_reading(treatment, control, contrast) == expected
    with pytest.raises(ValueError, match="sign"):
        pathway_reading(treatment, control, 2)


# ---------------------------------------------------------------------------
# unpaired_contrast -- two pools over DIFFERENT windows (two strata of one arm).
# ---------------------------------------------------------------------------


def _pooled(arm, rung, mean, se, *, clusters=24, windows=229, channel="free") -> PooledMean:
    """A `PooledMean` by hand: only the fields the contrast reads carry
    meaning; the rest are shaped correctly and otherwise inert."""
    return PooledMean(
        arm=arm, rung=rung, channel=channel, seeds=(0, 1, 2),
        windows=windows, windows_excluded=0, excluded_windows=(), clusters=clusters,
        series=np.full(windows, float(mean)), mean=float(mean), se=float(se),
        se_independent=float("nan"), z=_z(float(mean), float(se)),
    )


def test_unpaired_contrast_differences_the_means_adds_the_variances_and_takes_the_smaller_cluster_count():
    """0.62 - 0.22 = 0.40; se sqrt(0.03^2 + 0.04^2) = 0.05 EXACTLY (a 3-4-5
    triangle, so no rounding); z = 8. Clusters 24 against 78: the ruler is
    read against the smaller, conservative for the larger."""
    held = _pooled("frozen_ssl", "train_held", 0.62, 0.03, clusters=78, windows=740)
    val = _pooled("frozen_ssl", "val", 0.22, 0.04, clusters=24, windows=229)
    c = unpaired_contrast(held, val)
    assert isinstance(c, UnpairedContrast)
    assert (c.a, c.b, c.channel) == ("frozen_ssl/train_held", "frozen_ssl/val", "free")
    assert c.mean == pytest.approx(0.40)
    assert c.se == pytest.approx(0.05)
    assert c.z == pytest.approx(8.0)
    assert (c.windows_a, c.windows_b, c.clusters) == (740, 229, 24)
    flipped = unpaired_contrast(val, held)
    assert flipped.mean == pytest.approx(-0.40) and flipped.z == pytest.approx(-8.0)


def test_unpaired_contrast_identical_means_give_zero_and_a_lone_ruler_passes_through():
    c = unpaired_contrast(
        _pooled("pixel_ae", "train_held", 0.5, 0.02), _pooled("pixel_ae", "val", 0.5, 0.0)
    )
    assert c.mean == 0.0 and c.se == pytest.approx(0.02) and c.z == 0.0


def test_unpaired_contrast_nan_ruler_propagates_and_never_falls_back():
    """A clustered SE that could not be estimated (one cluster) is NaN, and so
    is the contrast's -- `_z`'s policy, never the naive standard error."""
    c = unpaired_contrast(
        _pooled("pixel_ae", "train_held", 0.5, float("nan"), clusters=1),
        _pooled("pixel_ae", "val", 0.2, 0.01),
    )
    assert c.mean == pytest.approx(0.3)
    assert np.isnan(c.se) and np.isnan(c.z)
    assert c.clusters == 1


def test_unpaired_contrast_refuses_a_different_reading_and_a_pool_against_itself():
    with pytest.raises(IncompatibleCells, match="not the same reading"):
        unpaired_contrast(
            _pooled("pixel_ae", "train_held", 0.5, 0.1, channel="free"),
            _pooled("pixel_ae", "val", 0.5, 0.1, channel="probe"),
        )
    with pytest.raises(IncompatibleCells, match="against itself"):
        unpaired_contrast(_pooled("pixel_ae", "val", 0.5, 0.1), _pooled("pixel_ae", "val", 0.4, 0.1))


# Constants captured by running the current code on the varied fixture
POOLED_RATIO_AT_SEED_0 = 1.725
POOLED_CI_LOW_AT_SEED_0 = 1.2307692307692308
POOLED_CI_HIGH_AT_SEED_0 = 12.5
POOLED_SE_AT_SEED_0 = 4.156273027301207


def test_the_bootstrap_helpers_are_public_and_draw_whole_episodes():
    """M3m consumes these from another module, so they carry public names.

    THE MUTATION THIS EXISTS FOR: renaming them back to `_episode_bootstrap` /
    `_interval` breaks the import; drawing rows instead of episodes makes the
    yielded index arrays contain partial episodes.
    """
    from mbfps.eval.pooling import episode_bootstrap, percentile_interval

    labels = np.array([0, 0, 0, 1, 1, 2])
    draws = list(episode_bootstrap(labels, bootstrap=50, seed=0))
    assert len(draws) == 50
    # The draws must actually vary; 50 identical draws would satisfy every
    # per-row check below while the bootstrap sampled nothing.
    assert len({tuple(sorted(d.tolist())) for d in draws}) > 1
    members = {0: {0, 1, 2}, 1: {3, 4}, 2: {5}}
    for index in draws:
        # Every drawn episode contributes ALL of its rows, so the multiset of
        # rows is a union of whole episodes -- never a partial one.
        counts = Counter(index.tolist())
        for episode, rows in members.items():
            per_row = {counts.get(row, 0) for row in rows}
            assert len(per_row) == 1, (episode, counts)

    replicates = np.arange(1001, dtype=np.float64)
    low, high, se = percentile_interval(replicates)
    assert low == pytest.approx(25.0)
    assert high == pytest.approx(975.0)
    assert se == pytest.approx(float(replicates.std(ddof=1)))
    assert np.isnan(percentile_interval(np.array([3.0]))[2])


def test_promoting_the_helpers_changed_no_pooled_number():
    """The promotion is a rename. `pool_ratio`'s output is unchanged.

    THE FIXTURE MUST DISCRIMINATE, AND THIS TEST ASSERTS THAT ABOUT ITSELF.
    An earlier version used `_scaled_cells((1.0, 1.0, 1.0))`, which gives 2
    clusters, 12 rows and an interval of EXACTLY zero width -- every bootstrap
    replicate identical -- so a changed percentile moved nothing and a changed
    `ddof` moved only a figure the test did not assert. It could not fail under
    two of the three mutations it exists to catch. `varied` spans three
    episodes and its interval has width.

    WHAT THIS TEST PINS, AND WHAT IT DOES NOT. It catches `ddof=0`, drawing
    without replacement, a changed draw count, lost episode multiplicity and a
    changed seed stream. It does NOT pin the percentiles: three episodes give
    only seven distinct replicate values, so `ci_high` is 12.5 for any upper
    percentile from ~84.75 to 100 and `ci_low` is 1.2308 for any lower
    percentile below 3.4. The percentiles are pinned exactly by the sibling
    `test_the_bootstrap_helpers_are_public_and_draw_whole_episodes`, whose
    `arange(1001)` lands them on exact indices. Nor does `pooled.ratio` move
    under ANY mutation of the two helpers -- `pool_ratio` computes it without
    touching either -- so that assertion is a fixture-drift tripwire, not a
    behaviour pin.
    """
    varied = [
        series(
            "pixel_ae", seed, [0.0] * 4,
            embedding=[1.0 + seed, 5.0, 9.0 + seed, 100.0],
            noise=[2.0, 4.0, 4.0 + seed, 8.0], episodes=(0, 0, 1, 2),
        )
        for seed in SEEDS
    ]
    pooled = pool_ratio(varied, bootstrap=2000, seed=0)

    # The guard this test was missing: a zero-width interval or a zero
    # standard error would make every comparison below hold for the wrong
    # reason, and the test would silently stop testing anything.
    assert pooled.clusters == 3
    assert pooled.ci_low < pooled.ci_high
    assert pooled.bootstrap_se > 0.0

    assert pooled.ratio == pytest.approx(POOLED_RATIO_AT_SEED_0, abs=1e-12)
    assert pooled.ci_low == pytest.approx(POOLED_CI_LOW_AT_SEED_0, abs=1e-12)
    assert pooled.ci_high == pytest.approx(POOLED_CI_HIGH_AT_SEED_0, abs=1e-12)
    assert pooled.bootstrap_se == pytest.approx(POOLED_SE_AT_SEED_0, abs=1e-12)


def test_clustered_interval_resamples_episodes_not_windows():
    """A window-level bootstrap would return a visibly narrower interval.

    Six episodes of twelve windows each, with the episode mean carrying all the
    signal: resampling windows averages the between-episode variation away,
    resampling episodes keeps it. The fixture's own discrimination is asserted
    first, because an interval of zero width would pass any comparison.

    WHY TWELVE WINDOWS AND NOT FOUR. The widening factor is capped by the
    square root of the windows per episode: with the episode mean carrying all
    the signal the clustered standard error is sqrt(m) times the iid one. At
    four windows per episode that is a ceiling of 2, and MEASURED the factor
    sits at 1.89 whether the offsets are 1x, 10x or 100x the within-episode
    noise, so no re-scaling of the offsets can clear the 2x floor below. Twelve
    windows per episode (the shipped cells have 229 over 24, about 9.5) puts the
    ceiling at 3.46 and the measured factor at about 3.2.

    THE MUTATION THIS EXISTS FOR: resampling windows instead of episodes, which
    makes the two intervals below identical and the factor exactly 1.
    """
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(6), 12)
    offsets = np.array([-9.0, -5.0, -1.0, 1.0, 5.0, 9.0])
    rows = (offsets[groups][:, None] + rng.normal(0.0, 0.1, size=(72, 3)))

    assert np.unique(groups).size == 6
    assert rows.shape == (72, 3)

    point, low, high = clustered_interval(
        rows, groups, h=1, resamples=500, seed=0
    )
    assert low < high, "fixture must produce a nonzero-width interval"
    assert low < point < high

    window_groups = np.arange(72)
    _p, w_low, w_high = clustered_interval(
        rows, window_groups, h=1, resamples=500, seed=0
    )
    assert (high - low) > 2.0 * (w_high - w_low), (
        "clustering by episode must widen the interval against clustering by "
        f"window; got {high - low:.4f} against {w_high - w_low:.4f}"
    )


def test_clustered_interval_refuses_a_defaulted_seed_and_a_single_cluster():
    """`seed` is keyword-required with no default, and one episode is refused.

    A defaulted seed is how a previous milestone shipped every cell drawing
    the same resamples; the point estimate is seed-free, so only the bounds
    move and a reader cannot tell nine identical draws from nine independent
    ones. One cluster is not a bootstrap.
    """
    rows = np.arange(12.0).reshape(4, 3)
    with pytest.raises(TypeError):
        clustered_interval(rows, np.arange(4), h=1, resamples=10)
    with pytest.raises(ValueError, match="at least two episodes"):
        clustered_interval(
            rows, np.zeros(4, dtype=int), h=1, resamples=10, seed=0
        )


def test_clustered_interval_refuses_a_nonfinite_value_outside_the_read_column():
    """A NaN anywhere is refused, not only in the column `h` reads.

    A NaN that reaches the mean returns `(nan, nan, nan)` with no exception
    and no indication of which input was bad, and that triple is what a
    verdict is read from.
    """
    rows = np.ones((4, 3))
    rows[2, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        clustered_interval(rows, np.arange(4) // 2, h=1, resamples=10, seed=0)


# ---------------------------------------------------------------------------
# clustered_interval -- moved from `eval.burden.margin_interval` (M3m) with the
# name and the first argument generalised; `resamples` lost its default.
# ---------------------------------------------------------------------------


def test_the_interval_clusters_on_episodes_and_requires_its_seed():
    """The resampling unit is the EPISODE, not the window: 229 windows over 24
    episodes, and consecutive Doom frames are near-duplicates, so a
    window-level bootstrap counts correlated observations as independent ones
    (1.26x to 2.17x too narrow on M3m's `window_margin`, and no claim about any
    other quantity).

    THE MUTATION THIS EXISTS FOR, TWICE OVER: resampling windows instead of
    episodes narrows the interval; and `seed: int = 0` lets every cell share
    one seed, which is how a previous milestone shipped nine identical draws.
    """
    # SIX episodes, not two. MEASURED: with two episodes the resample admits
    # only THREE distinct replicate values, so all 20 seeds I tried return
    # byte-identical bounds and the seed assertion below is dead. Six episodes
    # give 216 distinct replicates and no other seed reproduces seed 0's bounds.
    rng = np.random.default_rng(0)
    window_margin = np.vstack([
        rng.normal(mean, 0.05, size=(20, 3))
        for mean in (6.0, 4.0, 2.0, -2.0, -4.0, -6.0)
    ])
    groups = np.repeat(np.arange(6), 20)

    point, low, high = clustered_interval(
        window_margin, groups, h=1, resamples=400, seed=0
    )
    assert point == pytest.approx(window_margin[:, 0].mean(), abs=1e-12)
    # Clustered on episodes the interval spans the spread of episode means; a
    # window-level bootstrap would collapse it near the mean.
    assert high - low > 5.0

    # The fixture must discriminate, and this test asserts that about itself:
    # a resample with too few clusters yields too few distinct replicates for
    # any percentile to choose between, and every assertion below then holds
    # for the wrong reason.
    replicates = {
        clustered_interval(window_margin, groups, h=1, resamples=400, seed=s)[1:]
        for s in range(8)
    }
    assert len(replicates) > 1, "the bounds do not move with the seed at all"

    # Different seeds move the BOUNDS but never the point estimate.
    other = clustered_interval(window_margin, groups, h=1, resamples=400, seed=1)
    assert other[0] == pytest.approx(point, abs=1e-12)
    assert (other[1], other[2]) != (low, high)
    # The same seed twice is identical.
    again = clustered_interval(window_margin, groups, h=1, resamples=400, seed=0)
    assert (again[1], again[2]) == (low, high)

    # `seed` and `resamples` have no default.
    import inspect
    sig = inspect.signature(clustered_interval)
    assert sig.parameters["seed"].default is inspect.Parameter.empty
    assert sig.parameters["seed"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["resamples"].default is inspect.Parameter.empty
    assert sig.parameters["resamples"].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_interval_refuses_what_it_cannot_cluster():
    """`windows.episode` is null on a record whose ladder carried no
    clustering, and the record's own comment says a reader must REFUSE rather
    than treat every window as its own episode.

    THE MUTATION THIS EXISTS FOR: falling back to `np.arange(n)` for missing
    labels, which silently converts an episode bootstrap into a window one.
    """
    window_margin = np.zeros((6, 3))
    with pytest.raises(ValueError, match="one label per window"):
        clustered_interval(window_margin, np.array([0, 0, 1]), h=1, resamples=10, seed=0)
    with pytest.raises(ValueError, match="at least two episodes"):
        clustered_interval(
            window_margin, np.zeros(6, dtype=int), h=1, resamples=10, seed=0
        )

    # The record's own null. `windows.episode` is None when the ladder carried
    # no clustering, and a fallback to arange(n) here would silently convert
    # this into the window-level bootstrap the docstring rules out.
    with pytest.raises(ValueError, match="one label per window"):
        clustered_interval(window_margin, None, h=1, resamples=10, seed=0)


def test_the_interval_is_the_percentile_interval_of_the_clustered_replicate_means():
    """The reduction, the column, the seed and the resample count, pinned
    against an oracle assembled from the two Task-1 pieces the interval is
    defined in terms of.

    The brief's interval test reads only `h=1`, so a hard-coded column 0 passes
    it, and its fixture never distinguishes a mean from a median. This one reads
    `h=3` of four columns, uses episodes of UNEQUAL size (3, 7, 4, 9, 5 windows)
    so that the window-pooled mean the replicates take differs from the mean of
    episode means, and then asserts about itself that each alternative reading
    lands on different bounds. MEASURED: the closest alternative (400 -> 2000
    resamples) moves a bound by 0.404, a different seed by 0.615, and every other
    alternative by more than 2.6; the point estimate differs from the mean of
    episode means by 2.26. The assertion floor is 0.1, a quarter of the closest.

    THE MUTATIONS THIS EXISTS FOR: `h - 1` -> `h` or a hard-coded column;
    `.mean()` -> `.median()` or `.sum()` in the replicate; `seed` or `resamples`
    not reaching the bootstrap; and a point estimate taken as the mean of episode
    means rather than of the windows.
    """
    from mbfps.eval.pooling import episode_bootstrap, percentile_interval

    rng = np.random.default_rng(3)
    sizes = (3, 7, 4, 9, 5)
    episode_mean = np.array([5.0, -1.0, 2.0, -4.0, 1.0])
    column_scale = np.array([1.0, -0.5, 2.0, 0.25])  # columns are not interchangeable
    window_margin = np.vstack([
        episode_mean[episode] * column_scale + rng.normal(0.0, 0.3, size=(size, 4))
        for episode, size in enumerate(sizes)
    ])
    groups = np.repeat(np.arange(5), sizes)
    h, resamples, seed = 3, 400, 5

    def oracle(column, labels, draws, draw_seed, reduce=np.mean):
        replicates = np.array([
            reduce(column[index]) for index in episode_bootstrap(labels, draws, draw_seed)
        ])
        low, high, _se = percentile_interval(replicates)
        return low, high

    column = window_margin[:, h - 1]
    expected = oracle(column, groups, resamples, seed)
    point, low, high = clustered_interval(
        window_margin, groups, h=h, resamples=resamples, seed=seed
    )
    assert point == pytest.approx(column.mean(), abs=1e-12)
    assert (low, high) == pytest.approx(expected, abs=1e-12)

    # The fixture must discriminate: each alternative lands on other bounds.
    alternatives = {
        "h=1": oracle(window_margin[:, 0], groups, resamples, seed),
        "h=2": oracle(window_margin[:, 1], groups, resamples, seed),
        "h=4": oracle(window_margin[:, 3], groups, resamples, seed),
        "median": oracle(column, groups, resamples, seed, reduce=np.median),
        "sum": oracle(column, groups, resamples, seed, reduce=np.sum),
        "seed 0": oracle(column, groups, resamples, 0),
        "resamples 2000": oracle(column, groups, 2000, seed),
        "windows as episodes": oracle(column, np.arange(column.size), resamples, seed),
    }
    for name, bounds in alternatives.items():
        assert max(abs(bounds[0] - expected[0]), abs(bounds[1] - expected[1])) > 0.1, name

    episode_means = [column[groups == episode].mean() for episode in range(5)]
    assert abs(point - np.mean(episode_means)) > 2.0


def test_the_interval_refuses_a_malformed_margin_a_bad_resample_count_and_a_bad_horizon():
    """Each remaining guard, with the operand that fires it exercised alone.

    A groups array of shape (6, 1) has the right SIZE and the wrong RANK, so only
    the rank operand of the label guard rejects it. A horizon of 0 or -1 is not
    refused by NumPy at all -- `h - 1` is then -1 or -2, which index the LAST and
    the SECOND-TO-LAST column -- so only the lower bound stands between it and a
    silently wrong interval; a horizon past the end is caught by the upper bound
    (a missing one surfaces as an IndexError, which is not the ValueError
    asserted here). The two edges that must WORK are exercised too, so a bound
    moved one step inward also fails.

    THE MUTATIONS THIS EXISTS FOR: dropping the rank check on `window_margin`;
    dropping the rank operand of the label guard; `resamples < 1` -> `< 0`; and
    either bound of `1 <= h <= horizon` dropped or moved inward by one.
    """
    window_margin = np.arange(18.0).reshape(6, 3)
    groups = np.repeat([0, 1], 3)
    run = {"resamples": 10, "seed": 0}

    clustered_interval(window_margin, groups, h=1, **run)  # first column
    clustered_interval(window_margin, groups, h=3, **run)  # last column

    with pytest.raises(ValueError, match="windows, horizon"):
        clustered_interval(window_margin[:, 0], groups, h=1, **run)
    with pytest.raises(ValueError, match="one label per window"):
        clustered_interval(window_margin, groups.reshape(6, 1), h=1, **run)
    for bad in (0, -1):
        with pytest.raises(ValueError, match="resamples must be"):
            clustered_interval(window_margin, groups, h=1, resamples=bad, seed=0)
    for bad in (0, -1, 4):
        with pytest.raises(ValueError, match="horizon step must be"):
            clustered_interval(window_margin, groups, h=bad, **run)


def test_the_interval_refuses_a_nonfinite_margin_wherever_it_sits():
    """A NaN in the column read at `h` used to return `(nan, nan, nan)` with no
    refusal. That triple is what Reading H's verdict is read from, so a silent
    one is a silent wrong verdict. MEASURED on this fixture before the guard
    existed: a NaN at (0, 0) gave `(nan, nan, nan)` at `h=1`, while one at (3, 1)
    or (5, 2) gave a finite interval -- the corrupt cell was silently ignored.

    The positions are the first, a middle and the last cell of the matrix. At
    `h=1` only the first lies in the column that is read; the middle and the last
    lie in columns that are NOT, so they pin that the refusal covers the whole
    array rather than the one column the mean is taken over.

    THE MUTATIONS THIS EXISTS FOR: dropping the finiteness guard, and narrowing
    it to the column read at `h` (or to the first row).
    """
    window_margin = np.arange(18.0).reshape(6, 3)
    groups = np.repeat([0, 1], 3)
    run = {"h": 1, "resamples": 10, "seed": 0}

    # The control: the same call on the unpoisoned array is finite, so each
    # refusal below is attributable to the one cell that was changed.
    assert np.isfinite(clustered_interval(window_margin, groups, **run)).all()

    for bad in (np.nan, np.inf, -np.inf):
        for row, column in ((0, 0), (3, 1), (5, 2)):
            poisoned = window_margin.copy()
            poisoned[row, column] = bad
            with pytest.raises(ValueError, match="finite"):
                clustered_interval(poisoned, groups, **run)


def test_the_interval_accepts_the_plain_lists_a_json_record_hands_over():
    """`windows.episode` and the per-window margins come out of a JSON record as
    Python lists, which have no `.ndim` for the guards to read.

    THE MUTATIONS THIS EXISTS FOR: dropping either `np.asarray` coercion at the
    top of `clustered_interval`. Dropping `dtype=np.float64` from the margin one is
    an equivalent mutant -- an integer column's mean is the same float -- so it
    is not pinned.
    """
    rng = np.random.default_rng(1)
    window_margin = rng.normal(0.0, 1.0, size=(12, 3))
    groups = np.repeat(np.arange(4), 3)

    from_arrays = clustered_interval(window_margin, groups, h=2, resamples=50, seed=7)
    from_lists = clustered_interval(
        window_margin.tolist(), groups.tolist(), h=2, resamples=50, seed=7
    )
    assert from_lists == from_arrays
