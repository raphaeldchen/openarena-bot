"""M3o's script: the rows, the measure phase, the record it writes, and the read phase.

THREE KINDS OF FIXTURE, each used where the others cannot reach.

  * A SYNTHETIC WORLD (`_make_world`). Position is coded in a few pixels of every
    frame, once as a "pixel view" (noise `SIGMA_P`) and once per "embedding view"
    (noise `SIGMA_E`), and the fake model's encoder reads only the embedding views.
    So the raw-pixel readouts see both, the embedding readouts see the encoder's, and
    the sequence readouts see five frames of it: `B - D > 0` and `F - B > 0` BY
    CONSTRUCTION, and the tests below show the realised readouts discriminate before
    they assert anything about a gain. The LINEAR world has an embedding that is
    linear in position (so `D - C` is near zero and the ridge optimum sits off the
    shipped grid, which makes `C` differ from `C_shipped`); the RING world codes
    position on 1.6 turns of a circle, which a linear probe cannot undo and the RBF
    can, so `D - C` is large and positive. A world holds 22 TRAINING episodes (the fit
    split is the first 20 of them, as `probe.probe_episodes` takes them, so two are held
    back and a test can see that) and 24 validation episodes (12 / 12 as shipped).
  * THE REAL MEASURE PATH over those worlds: real `prepare_shared`, real fits, real
    `difference_interval`, real `write_record`. Only the rollout (`reference_trajectories`)
    is replaced where a test needs to know its output.
  * RECORDS ON DISK for the read phase, each a deep copy of the record `measure_cell`
    wrote with only the fields the reading reads overwritten, so the schema is the
    writer's own.

EXPECTATIONS ARE NEVER READ FROM THE CODE UNDER TEST. The embedding features are rebuilt
by the test from the world's own pixels, the targets from the episodes' privileged arrays,
and the shipped-grid r2, the zero-information baseline and the widest-rung control are
recomputed here with `fit_probe` and `numpy.linalg.solve`.
"""

import contextlib
import copy
import importlib.util
import io
import json
import math
import re
import types
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from mbfps.data.episode import Episode, load_episode, save_episode
from mbfps.envs.protocol import OBS_SHAPE
from mbfps.eval import ceiling
from mbfps.eval.probe import (
    RIDGES, _mean_r2, apply_probe, fit_probe, position_error, probe_targets,
)
from mbfps.eval.study import SPLIT_SEED, git_sha, load_record, write_record
from mbfps.eval.windows import window_starts
from tests.eval.test_rollout import KEYS, DriftingModel, OracleModel, synthetic_episode

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "probe_ceiling.py"


def _load():
    spec = importlib.util.spec_from_file_location("probe_ceiling_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load()

CPU = torch.device("cpu")
CONTEXT, HORIZON = 3, 5
SEED = 3
"""The cell's seed. It is neither 0 nor `BOOTSTRAP_SEED`, so a bootstrap drawn at the
cell's seed, at 0, or at the base seed are three different draws."""
BOOTSTRAP_SEED = 7
SMALL = 60
"""A resample count the tests patch in, so recomputing an interval costs milliseconds and
a record can be told to have used the PATCHED count and not 2000."""

NINE_CELLS = tuple(
    (arm, seed) for arm in ("pixel_ae", "frozen_ssl", "random_vit") for seed in (0, 1, 2)
)
"""The shipped plan, spelled out. A test iterating `ARMS` x `SEEDS` would shrink with
either one."""

STUDY = {
    "steps": 20000,
    "git_sha": "ca3e140772d6bc741d4d04312763afe3dd754166",
    "filtering": {"criterion_4": {"embedding_r2": 0.33}},
    "probe": {"embedding_selection_r2": 0.02},
}


def _study(embedding_r2: float = 0.33, selection_r2: float = 0.02) -> dict:
    record = copy.deepcopy(STUDY)
    record["filtering"]["criterion_4"]["embedding_r2"] = embedding_r2
    record["probe"]["embedding_selection_r2"] = selection_r2
    return record


@pytest.fixture(autouse=True)
def _small_bootstrap(monkeypatch):
    monkeypatch.setattr(ceiling, "RESAMPLES", SMALL)


# ---------------------------------------------------------------------------
# The synthetic world.
# ---------------------------------------------------------------------------

SIGMA_P, SIGMA_E, SPEED = 60.0, 160.0, 0.05
LINEAR_VIEWS = 36
"""Embedding views PER MODEL in the linear world. Many noisy views make the embedding
wide enough that the best ridge is interior, between the shipped rungs."""
LINEAR_SIGMA_E = 220.0
"""The linear world's embedding noise. At the default `SIGMA_E` the ladder's RBF rung edged
the linear rung on selection (capacity 2.0, no `inf`) once the fit split held 20 training
episodes rather than 8 validation ones: the pixel coding is clipped to 0-255, which more rows
let an RBF rung pick up. Noisier embeddings put the linear rung back on top, and the
`inf` test needs a world where it wins."""
MODELS_IN_WORLD = 1


def _world_episode(length, index, *, sigma_p, sigma_e, views):
    """`obs[t, 0, 0, 0:2]` is the pixel view of `(s_x, s_y)`; `obs[t, 1 + j, 0, 0:2]` is
    embedding view `j`, each its own independent noise. Everything else is constant, so
    the 37,632 pixel columns carry 2 + 2 * views informative ones."""
    rng = np.random.default_rng(1000 + index)
    t = np.arange(length + 1, dtype=np.float64)
    a, b = SPEED * (1 + 0.1 * (index % 5)), SPEED * 0.8 * (1 + 0.1 * (index % 7))
    s = np.stack(
        [0.5 + 0.4 * np.sin(a * t + 0.7 * index), 0.5 + 0.4 * np.cos(b * t - 0.3 * index)],
        axis=1,
    )

    def code(noise):
        return np.clip(
            np.round(127 + 200 * (s - 0.5) + noise * rng.normal(size=s.shape)), 0, 255
        ).astype(np.uint8)

    obs = np.zeros((length + 1, *OBS_SHAPE), dtype=np.uint8)
    obs[:, 0, 0, 0:2] = code(sigma_p)
    for j in range(views):
        obs[:, 1 + j, 0, 0:2] = code(sigma_e)
    privileged = np.zeros((length + 1, len(KEYS)), dtype=np.float32)
    privileged[:, 1], privileged[:, 2] = 600 * s[:, 0], 600 * s[:, 1]
    return Episode(
        obs=obs, actions=np.arange(length, dtype=np.int32) % 6,
        rewards=np.zeros(length, np.float32), terminated=np.zeros(length, bool),
        truncated=np.zeros(length, bool), privileged=privileged, privileged_keys=KEYS,
        policy_name="synthetic", seed=0, scenario="world",
    )


class _WorldEncoder(nn.Module):
    """Reads embedding views `first .. first + views` only. `kind="ring"` codes each
    value on 1.6 turns of a circle: injective, and not linear in position."""

    def __init__(self, views, kind, first):
        super().__init__()
        self.views, self.kind, self.first = views, kind, first

    def forward(self, obs):
        v = obs[:, 1 + self.first:1 + self.first + self.views, 0, 0:2].to(torch.float32)
        if self.kind == "curved":
            v = v + 0.004 * (v - 127) ** 2
        if self.kind == "ring":
            u = 1.6 * torch.pi * ((v - 127) / 200 + 0.4) / 0.8
            v = torch.stack([torch.cos(u), torch.sin(u)], dim=-1)
        return v.reshape(obs.shape[0], -1)


class WorldModel(OracleModel):
    def __init__(self, views, kind="linear", first=0):
        super().__init__()
        self.encoder = _WorldEncoder(views, kind, first)


def _make_world(directory, *, kind, views, models=1, sigma_p=SIGMA_P, sigma_e=SIGMA_E,
                n_val=24, n_train=22):
    """Twenty-four validation episodes (12 select / 12 score, as shipped) and twenty-two
    training ones (the first 20 are the fit split, as shipped), lengths 44-50, named so
    `ReplayBuffer` can read them. Twelve score-split episodes are enough that two
    bootstrap seeds draw visibly different intervals; with four, the 2.5th percentile of a
    few dozen resamples was the same extreme draw under both."""
    directory.mkdir(parents=True, exist_ok=True)
    total = views * models
    val, train = [], []
    for i in range(n_val + n_train):
        episode = _world_episode(
            44 + 2 * (i % 4), i, sigma_p=sigma_p, sigma_e=sigma_e, views=total,
        )
        path = directory / f"ep_{i:06d}_len{episode.length:05d}.npz"
        save_episode(episode, path)
        (val if i < n_val else train).append(path)
    tags = np.concatenate([
        np.arange(load_episode(p).privileged.shape[0], dtype=np.float64)[:, None] for p in val
    ])
    targets = np.concatenate([probe_targets(load_episode(p).privileged, KEYS) for p in val])
    return types.SimpleNamespace(
        val=val, train=train, kind=kind, views=views,
        probe=fit_probe(tags, targets, ridge=1e-8),
        model=lambda first=0: WorldModel(views, kind, first * views),
    )


def _cell(world, shared, *, first=0, arm="pixel_ae", seed=SEED, study=None, **over):
    kwargs = dict(
        model=world.model(first), val_paths=world.val, probe=world.probe, arm=arm,
        seed=seed, context=CONTEXT, horizon=HORIZON, device=CPU, feature_backbone=None,
        study_record=study or _study(), bootstrap_seed=BOOTSTRAP_SEED, shared=shared,
    )
    kwargs.update(over)
    return script.measure_cell(**kwargs)


@pytest.fixture(scope="module")
def linear_world(tmp_path_factory):
    world = _make_world(
        tmp_path_factory.mktemp("linear_world"), kind="linear", views=LINEAR_VIEWS,
        models=MODELS_IN_WORLD, sigma_e=LINEAR_SIGMA_E,
    )
    world.shared = script.prepare_shared(world.train, world.val, budget=12000)
    return world


@pytest.fixture(scope="module")
def ring_world(tmp_path_factory):
    world = _make_world(
        tmp_path_factory.mktemp("ring_world"), kind="ring", views=1, sigma_e=5.0,
    )
    world.shared = script.prepare_shared(world.train, world.val, budget=12000)
    return world


@pytest.fixture(scope="module")
def linear_record(linear_world):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ceiling, "RESAMPLES", SMALL)
        return _cell(linear_world, linear_world.shared)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The CURVED world: the embedding is a mild nonlinearity of position, so the RBF beats
    the linear readout a little (`D - C` has a nonzero width, which a purely linear world
    does not: there the ladder picks the linear rung and `D` and `C` predict identically)."""
    built = _make_world(
        tmp_path_factory.mktemp("curved_world"), kind="curved", views=LINEAR_VIEWS,
    )
    built.shared = script.prepare_shared(built.train, built.val, budget=12000)
    return built


@pytest.fixture(scope="module")
def record(world):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ceiling, "RESAMPLES", SMALL)
        return _cell(world, world.shared)


def _embedding_rows(world, shared, name, *, first=0):
    """The embedding features of one split, rebuilt from the world's own pixels, and
    the targets from the episodes' privileged arrays -- not through `encode_rows`, and
    not through `SplitRows.targets`."""
    xs, ys = [], []
    split = shared.splits[name]
    for path, frames in zip(split.paths, split.frames):
        episode = load_episode(path)
        v = episode.obs[frames][:, 1 + first * world.views:1 + (first + 1) * world.views, 0, 0:2]
        v = v.astype(np.float64)
        if world.kind == "curved":
            v = v + 0.004 * (v - 127) ** 2
        xs.append(v.reshape(len(frames), -1))
        ys.append(probe_targets(episode.privileged[frames], KEYS))
    return np.concatenate(xs), np.concatenate(ys)


# ---------------------------------------------------------------------------
# Constants, exit numbers, names.
# ---------------------------------------------------------------------------


def test_the_exit_codes_are_this_milestones_own():
    """43/44 are M3l's, 45/46 M3m's, 47/48 M3n's. The numbers are literals, so a swap of
    the two constants' own definitions is seen."""
    assert script.EXIT_HARNESS_DISAGREES == 49
    assert script.EXIT_NO_MAJORITY == 50


def test_read_exits_key_only_the_status_that_is_not_a_model_verdict():
    """The three verdicts exit 0: each is an answer. 49 is returned from the controls,
    before the tally, so it is not keyed here."""
    assert script.READ_EXITS == {"NO_MAJORITY": 50}
    assert set(script.READ_EXITS).isdisjoint(
        {"PROBE_LIMITED", "REPRESENTATION_LIMITED", "OBSERVATION_LIMITED"}
    )


def test_the_eight_readouts_are_the_ones_the_schema_names():
    """Spelled out: a table that lost an entry would shrink with a test that iterated it."""
    assert script.READOUTS == {
        "pixels_linear": ("pixels", "linear"),
        "pixels_rbf": ("pixels", "rbf"),
        "embedding_linear": ("embedding", "linear"),
        "embedding_rbf": ("embedding", "rbf"),
        "embedding_knn": ("embedding", "knn"),
        "sequence_linear": ("sequence", "linear"),
        "sequence_rbf": ("sequence", "rbf"),
        "embedding_linear_shipped": ("embedding", "linear_shipped"),
    }
    assert script.PIXEL_READOUTS == ("pixels_linear", "pixels_rbf")


def test_the_record_path_separates_seeds_and_arms(tmp_path):
    """Three seeds of one arm must not collide on one filename: M3m's fixtures all used
    seed 0, which hid exactly this (6 of 9 records destroyed, found at the read phase
    after the compute was spent). The names are spelled out so the read-phase fixtures,
    which write by literal name, agree with what the script looks for."""
    paths = {cell: script.ceiling_record_path(tmp_path, *cell) for cell in NINE_CELLS}
    assert len(set(paths.values())) == 9
    for (arm, seed), path in paths.items():
        assert path == tmp_path / f"ceiling_{arm}_seed{seed}.json"


@pytest.mark.parametrize("base", [0, BOOTSTRAP_SEED, 12345])
def test_the_derived_seed_differs_for_every_one_of_the_nine_cells(base):
    """The verdict counts nine cells as nine independent readings, and cells that
    resample the same episode draws are one reading nine times."""
    derived = {cell: script.cell_bootstrap_seed(base, *cell) for cell in NINE_CELLS}
    assert len(set(derived.values())) == 9, derived
    assert base not in derived.values()


def test_the_derived_seed_is_crc32_and_not_python_hash():
    """The literals are what `zlib.crc32` of `b"base:arm:seed"` returns (computed from the
    definition, not from the function); a `hash()` swapped back in is salted per process
    and misses them."""
    assert script.cell_bootstrap_seed(0, "pixel_ae", 0) == 235428016
    assert script.cell_bootstrap_seed(0, "pixel_ae", 1) == 2031052838
    assert script.cell_bootstrap_seed(7, "frozen_ssl", 2) == 3554091843
    assert script.cell_bootstrap_seed(0, "random_vit", 2) == 1082770177
    assert script.cell_bootstrap_seed(0, "pixel_ae", 0) == zlib.crc32(b"0:pixel_ae:0")


def test_the_derived_seed_moves_when_the_base_moves_for_every_cell():
    for cell in NINE_CELLS:
        assert script.cell_bootstrap_seed(0, *cell) != script.cell_bootstrap_seed(1, *cell), cell


# ---------------------------------------------------------------------------
# The rows: the split, the stride, the targets, the window.
# ---------------------------------------------------------------------------


def _names(count):
    return [Path(f"/data/ep_{i:06d}_len00100.npz") for i in range(count)]


def test_the_shipped_validation_set_of_24_episodes_is_halved_12_and_12():
    split = script.validation_halves(_names(24), seed=SPLIT_SEED)
    assert set(split) == {"select", "score"}
    assert {name: len(paths) for name, paths in split.items()} == {"select": 12, "score": 12}
    union = [p for paths in split.values() for p in paths]
    assert sorted(union) == _names(24) and len(set(union)) == 24


def test_the_halves_are_a_function_of_the_names_and_not_of_their_order():
    paths = _names(24)
    shuffled = list(reversed(paths))
    assert script.validation_halves(paths, seed=SPLIT_SEED) == script.validation_halves(
        shuffled, seed=SPLIT_SEED,
    )
    assert script.validation_halves(paths, seed=SPLIT_SEED) != script.validation_halves(
        paths, seed=SPLIT_SEED + 1,
    )


def test_the_held_out_half_is_the_score_split_and_the_rest_is_select():
    """`episode_split` returns `(train, val)`: the side it HOLDS OUT is the second. The
    score split is that side, so the roles are not exchanged, and the two are told apart
    by an odd count, where the halves differ in size.

    THE MUTATION THIS EXISTS FOR: `{"select": score, "score": select}`."""
    from mbfps.data.split import episode_split

    rest, held = episode_split(_names(25), val_fraction=0.5, seed=SPLIT_SEED)
    assert (len(rest), len(held)) == (13, 12)
    split = script.validation_halves(_names(25), seed=SPLIT_SEED)
    assert split["select"] == rest and split["score"] == held


@pytest.mark.parametrize("count, expected", [(2, (1, 1)), (9, (5, 4)), (24, (12, 12)), (49, (25, 24))])
def test_the_halves_hold_the_odd_episode_in_select(count, expected):
    split = script.validation_halves(_names(count), seed=SPLIT_SEED)
    assert (len(split["select"]), len(split["score"])) == expected


def test_a_row_is_a_frame_with_real_history_and_its_target_is_that_frames(linear_world):
    shared = linear_world.shared
    assert shared.stride == 1
    for name, split in shared.splits.items():
        assert [int(f[0]) for f in split.frames] == [script.SEQUENCE_FRAMES - 1] * len(split.paths)
        expected_n = 0
        targets = []
        for path, frames in zip(split.paths, split.frames):
            episode = load_episode(path)
            assert frames[-1] == episode.length  # the last frame, `T`, is a row
            assert np.array_equal(frames, np.arange(script.SEQUENCE_FRAMES - 1, episode.length + 1))
            expected_n += episode.length + 1 - (script.SEQUENCE_FRAMES - 1)
            targets.append(probe_targets(episode.privileged[frames], KEYS))
        assert split.n == expected_n
        np.testing.assert_array_equal(split.targets, np.concatenate(targets))
        assert split.groups.shape == (split.n,)
        assert sorted(set(split.groups.tolist())) == list(range(len(split.paths)))


def test_the_fit_split_is_the_studys_first_twenty_training_episodes(linear_world):
    """THE FIT SPLIT IS THE TRAINING EPISODES the probe is fit on, `probe_episodes(train)[0]`:
    the leading 20 of the 22 the world holds, in the caller's order, none of them a
    validation episode. The two held back are what show it is the first 20 and not
    `train` whole, and the order what shows it is the LEADING block.

    THE MUTATIONS THIS EXISTS FOR: `fit` drawn from `val_paths` (the 8-episode revision this
    replaced), `fit` the whole of `train_paths`, and `fit` the last 20."""
    shared, world = linear_world.shared, linear_world
    assert len(world.train) == 22 and len(world.val) == 24
    assert shared.splits["fit"].paths == tuple(world.train[:20])
    assert not set(shared.splits["fit"].paths) & set(world.val)
    assert not set(world.train[20:]) & {p for s in shared.splits.values() for p in s.paths}
    assert [len(s.paths) for s in shared.splits.values()] == [20, 12, 12]


def test_select_and_score_are_the_validation_halves_and_share_no_episode(linear_world):
    """Selection and scoring stay entirely within validation and share no episode, so the
    three-split discipline is unchanged: only the fit set moved. The two halves are the
    ones `validation_halves` draws at the study's split seed, and their union is the whole
    validation set the rollout is run on."""
    shared, world = linear_world.shared, linear_world
    select, score = shared.splits["select"].paths, shared.splits["score"].paths
    assert not set(select) & set(score)
    assert sorted(select + score) == sorted(world.val)
    halves = script.validation_halves(world.val, seed=SPLIT_SEED)
    assert list(select) == halves["select"] and list(score) == halves["score"]


def test_the_stride_is_the_smallest_that_brings_every_split_within_the_budget(linear_world):
    """The rows KEPT, counted by the test from the episode lengths: an episode of `m` rows
    keeps `ceil(m / stride)`, so a split's total is not `m_total / stride`. The stride is the
    smallest one under which no split is over the budget (so one stride fewer WOULD be over),
    and a budget that binds divides EVERY split by the one stride, with `rows_per_split` saying
    what was left. The FIT split is the largest here (20 training episodes against 12).

    At budget 100 `ceil(largest / budget)` is stride 9, under which the fit split keeps 105
    rows: 5 over. THE MUTATION THIS EXISTS FOR: the stride taken as `ceil(largest / budget)`."""
    first = script.SEQUENCE_FRAMES - 1
    lengths = {
        name: [load_episode(p).length + 1 - first for p in split.paths]
        for name, split in linear_world.shared.splits.items()
    }

    def kept(stride):
        return {n: sum(-(-m // stride) for m in ms) for n, ms in lengths.items()}

    largest = max(kept(1).values())
    assert kept(1)["fit"] == largest > max(kept(1)["select"], kept(1)["score"])
    assert -(-largest // 100) == 9 and kept(9)["fit"] == 105  # the quotient's stride overshoots
    train, val = linear_world.train, linear_world.val
    for budget in (largest, largest - 1, 100, 64):
        stride, splits = script.build_splits(train, val, budget=budget)
        assert max(kept(stride).values()) <= budget, budget
        assert stride == 1 or max(kept(stride - 1).values()) > budget, budget
        for name, split in splits.items():
            assert split.n == kept(stride)[name] <= budget, (budget, name)
            assert all(int(f[0]) == first for f in split.frames)
            assert all(set(np.diff(f)) <= {stride} for f in split.frames)
    assert script.build_splits(train, val, budget=largest)[0] == 1
    assert script.build_splits(train, val, budget=largest - 1)[0] == 2
    assert script.build_splits(train, val, budget=100)[0] == 10
    with pytest.raises(SystemExit, match="budget"):
        script.build_splits(train, val, budget=0)


def test_a_budget_smaller_than_the_episodes_of_a_split_is_refused_not_looped_on(linear_world):
    """Every episode keeps at least one row at any stride, so 20 fit episodes cannot be brought
    under a budget of 19: refused by name, where a search for a stride would never end."""
    with pytest.raises(SystemExit, match="cannot hold one row of each episode"):
        script.build_splits(linear_world.train, linear_world.val, budget=19)
    script.build_splits(linear_world.train, linear_world.val, budget=20)


def test_the_pixel_rows_are_the_raw_frames_at_the_rows(linear_world):
    split = linear_world.shared.splits["select"]
    pixels = script.load_pixels(split)
    assert pixels.dtype == np.uint8 and pixels.shape == (split.n, 112 * 112 * 3)
    path, frames = split.paths[1], split.frames[1]
    start = int(sum(len(f) for f in split.frames[:1]))
    expected = load_episode(path).obs[frames].reshape(len(frames), -1)
    np.testing.assert_array_equal(pixels[start:start + len(frames)], expected)


def _tag_split(tmp_path, lengths=(30, 41)):
    paths = []
    for i, n in enumerate(lengths):
        path = tmp_path / f"ep_{i:06d}_len{n:05d}.npz"
        save_episode(synthetic_episode(n), path)
        paths.append(path)
    return paths


@pytest.mark.parametrize("stride", [1, 3])
def test_the_sequence_row_is_the_five_frames_ending_at_the_row_oldest_first(tmp_path, stride):
    """The tag encoder's embedding of frame `t` is `t` itself, so a row's `sequence` can be
    read off exactly: `[t - 4, ..., t]`. At stride 3 the history frames are NOT rows, and
    must still be the real preceding frames, not the preceding ROWS."""
    paths = _tag_split(tmp_path)
    targets = {p: script._all_targets(p) for p in paths}
    split = script._split_rows(paths, targets, stride)
    embedding, sequence = script.encode_rows(OracleModel(), split, None, CPU)
    rows = np.concatenate([f for f in split.frames])
    assert embedding.shape == (len(rows), 1) and sequence.shape == (len(rows), 5)
    np.testing.assert_array_equal(embedding[:, 0], rows)
    expected = rows[:, None] + np.arange(-4, 1)[None, :]
    np.testing.assert_array_equal(sequence, expected)
    if stride == 3:
        assert rows[1] - rows[0] == 3 and sequence[1, 0] == rows[1] - 4


def test_the_sequence_window_is_the_recorded_constant_wide(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "SEQUENCE_FRAMES", 3)
    paths = _tag_split(tmp_path)
    split = script._split_rows(paths, {p: script._all_targets(p) for p in paths}, 1)
    assert int(split.frames[0][0]) == 2
    _, sequence = script.encode_rows(OracleModel(), split, None, CPU)
    assert sequence.shape[1] == 3 and list(sequence[0]) == [0.0, 1.0, 2.0]


# ---------------------------------------------------------------------------
# measure_cell on the curved world: what the record holds, and what it was built from.
# ---------------------------------------------------------------------------


def _r(record, name):
    return record["readouts"][name]


def test_the_world_discriminates_what_the_assertions_below_are_about(record):
    """FIXTURES MUST BE SHOWN TO DISCRIMINATE before anything is asserted about the
    result. What the gain, shipped-grid and capacity tests read off this record:

      * raw pixels beat the encoder (`B > D`) and a window of encoder frames beats raw
        pixels (`F > B`), each by a margin no rounding could produce, and the two margins
        are not the same number;
      * the finer ridge grid finds something the shipped one cannot (`C > C_shipped`);
      * the score split has several episodes and every interval has width (the RBF beats
        linear here, so `D - C` has a spread: it is zero-width on a purely linear world).
    """
    d, b, f = (_r(record, n)["r2"] for n in ("embedding_rbf", "pixels_rbf", "sequence_rbf"))
    c, shipped = _r(record, "embedding_linear")["r2"], record["shipped_grid_r2"]
    assert b - d > 0.03 and f - b > 0.05
    assert abs((b - d) - (f - b)) > 0.05
    assert c - shipped > 0.005
    assert len(set(record["groups"])) >= 3
    for gain in record["gains"].values():
        assert gain["ci_high"] - gain["ci_low"] > 1e-3


def test_the_pixel_gain_is_b_minus_d_and_the_sequence_gain_is_f_minus_b_by_value(record):
    """B, D and F are all the estimator `"rbf"`, so `difference_interval` cannot tell
    `B - D` from `D - B`: exchanging the two arguments returns the wrong-sign gain with
    nothing to refuse it. Each is pinned by VALUE against the readouts the record carries,
    on a world where raw pixels beat the encoder and history beats one frame.

    THE MUTATIONS THIS EXISTS FOR, each run against it: `difference_interval(d, b, ...)` for
    the pixel gain, `difference_interval(b, f, ...)` for the sequence gain, and the two
    gains' arguments exchanged with each other."""
    gains = record["gains"]
    d = _r(record, "embedding_rbf")["r2"]
    b = _r(record, "pixels_rbf")["r2"]
    f = _r(record, "sequence_rbf")["r2"]
    assert gains["pixel"]["point"] == b - d
    assert gains["sequence"]["point"] == f - b
    assert gains["pixel"]["point"] > 0.0 and gains["sequence"]["point"] > 0.0
    assert gains["pixel"]["point"] != gains["sequence"]["point"]


def test_the_probe_gain_is_d_minus_c_minus_the_clamped_floor_by_value(world, monkeypatch):
    """THE PROBE GAIN GOES THROUGH `probe_gain_over_floor`: it is the only thing that forms
    `D - C - max(null_floor, 0)`, and a `difference_interval` here reads every cell against
    ZERO with nothing downstream able to tell. The floor is patched to values the real
    ladder never returns, so a gain formed without it is a different number.

    A positive floor is subtracted from the point and BOTH bounds, exactly. A negative one
    is a bar that cannot be lowered: the gain is the one at floor 0, while the record keeps
    the UNCLAMPED floor it measured (the clamp is taken once, after the mean).

    THE MUTATIONS THIS EXISTS FOR: `difference_interval(d, c, ...)` in place of
    `probe_gain_over_floor`, and `probe_gain_over_floor(..., floor=0.0, ...)`."""
    records = {}
    for floor in (0.0, 0.05, -0.3):
        monkeypatch.setattr(ceiling, "null_floor", lambda n, d, _f=floor, **k: _f)
        records[floor] = _cell(world, world.shared)
    base = records[0.0]["gains"]["probe"]
    d = _r(records[0.0], "embedding_rbf")["r2"]
    c = _r(records[0.0], "embedding_linear")["r2"]
    assert base["point"] == d - c
    for key in ("point", "ci_low", "ci_high"):
        assert records[0.05]["gains"]["probe"][key] == base[key] - 0.05
        assert records[-0.3]["gains"]["probe"][key] == base[key]
    assert records[0.05]["gains"]["probe"]["point"] < base["point"]
    assert [records[f]["null_floor"] for f in (0.0, 0.05, -0.3)] == [0.0, 0.05, -0.3]


def test_the_floor_is_measured_at_the_fit_splits_rows_and_the_embedding_width(
    world, monkeypatch,
):
    """`null_floor(n, d)` at THIS cell's shape: n the rows the kernel ladder is fit on,
    d the embedding feature set's width. The sequence width is five times that, and the
    other splits' sizes differ from the fit split's, so each wrong choice is a different
    number."""
    seen = []
    monkeypatch.setattr(ceiling, "null_floor", lambda n, d, **k: seen.append((n, d)) or 0.0)
    record = _cell(world, world.shared)
    shared = world.shared
    assert seen == [(shared.splits["fit"].n, LINEAR_VIEWS * 2)]
    assert len({s.n for s in shared.splits.values()}) == 3  # fit, select and score all differ
    assert record["null_floor_shape"] == {"n": shared.splits["fit"].n, "d": LINEAR_VIEWS * 2}
    assert seen[0][1] != LINEAR_VIEWS * 2 * script.SEQUENCE_FRAMES  # not the sequence width


def test_two_cells_on_the_same_rows_share_every_point_and_not_their_bounds(world):
    """Each cell draws at `cell_bootstrap_seed(base, arm, seed)`. The point estimates are
    seed-free, so only the bounds move: a seed hard-coded, or the BASE seed passed through
    unchanged, makes two cells' bounds equal.

    A STOCHASTIC fixture on purpose: the intervals are percentiles of resamples, so a
    seed that was never USED would leave two cells byte-identical here, which a
    deterministic fake cannot distinguish from a seed that was.

    THE MUTATIONS THIS EXISTS FOR: `seed=0`, and `seed=bootstrap_seed`."""
    first = _cell(world, world.shared, arm="pixel_ae", seed=1)
    second = _cell(world, world.shared, arm="frozen_ssl", seed=2)
    for name in ("probe", "pixel", "sequence"):
        assert first["gains"][name]["point"] == second["gains"][name]["point"], name
        assert first["gains"][name]["ci_low"] != second["gains"][name]["ci_low"], name
        assert first["gains"][name]["ci_high"] != second["gains"][name]["ci_high"], name
    assert first["cell_bootstrap_seed"] == script.cell_bootstrap_seed(BOOTSTRAP_SEED, "pixel_ae", 1)
    assert second["cell_bootstrap_seed"] == script.cell_bootstrap_seed(
        BOOTSTRAP_SEED, "frozen_ssl", 2,
    )
    assert first["bootstrap_seed"] == second["bootstrap_seed"] == BOOTSTRAP_SEED
    again = _cell(world, world.shared, arm="pixel_ae", seed=1)
    assert again == first


def test_the_base_seed_is_a_lever(world):
    """A different `--bootstrap-seed` is a different run: the bounds move, the points do not."""
    moved = _cell(world, world.shared, bootstrap_seed=BOOTSTRAP_SEED + 1)
    base = _cell(world, world.shared)
    assert moved["gains"]["pixel"]["point"] == base["gains"]["pixel"]["point"]
    assert moved["gains"]["pixel"]["ci_low"] != base["gains"]["pixel"]["ci_low"]


def test_the_shipped_grid_r2_is_the_shipped_probe_on_the_encoder_rows_and_not_c(
    world, record,
):
    """`shipped_grid_r2` is `fit_probe` over `probe.RIDGES`, recomputed here from the
    world's own pixels. It is NOT `embedding_linear`, whose finer grid reads higher (shown
    by `test_the_world_discriminates...`), and exit 49 reads it."""
    shared = world.shared
    fit, select, score = (_embedding_rows(world, shared, n) for n in ("fit", "select", "score"))
    probe = fit_probe(fit[0], fit[1], select[0], select[1])
    expected = _mean_r2(apply_probe(probe, score[0]), score[1])
    # the encoder computes the quadratic in float32, the rebuild in float64: ~1e-8 apart
    assert record["shipped_grid_r2"] == pytest.approx(expected, abs=1e-6)
    assert _r(record, "embedding_linear_shipped")["r2"] == record["shipped_grid_r2"]
    assert _r(record, "embedding_linear_shipped")["capacity"] in RIDGES
    assert _r(record, "embedding_linear")["r2"] != record["shipped_grid_r2"]
    assert (
        _r(record, "embedding_linear_shipped")["capacity"]
        == _r(record, "embedding_linear_shipped")["jitter"]
    )


def test_the_harness_flag_reads_the_shipped_grid_and_never_c(world, record, monkeypatch):
    """The study number is set to the shipped-grid r2 for one record and to C's for the
    other, at a tolerance too small for the two to be confused. If the flag read `C`, the
    first would be False and the second True.

    THE MUTATION THIS EXISTS FOR, run against it: `readouts["embedding_linear"].r2` for
    `readouts["embedding_linear_shipped"].r2` in the control."""
    monkeypatch.setattr(script, "HARNESS_R2_TOLERANCE", 1e-9)
    shipped = record["shipped_grid_r2"]
    c = _r(record, "embedding_linear")["r2"]
    assert abs(shipped - c) > 1e-3
    agree = _cell(world, world.shared, study=_study(embedding_r2=shipped))
    assert agree["controls"]["harness_r2_matches_study"] is True
    disagree = _cell(world, world.shared, study=_study(embedding_r2=c))
    assert disagree["controls"]["harness_r2_matches_study"] is False
    assert disagree["controls"]["study_embedding_r2"] == c
    assert disagree["harness_r2_tolerance"] == 1e-9


def test_the_harness_flag_is_read_against_the_criterion_4_probe_not_the_rollout_probe(
    world, record, monkeypatch,
):
    """The study's 0.320-0.345 is `filtering.criterion_4.embedding_r2`; the field the spec
    names, `probe.embedding_selection_r2`, is the ROLLOUT probe's and spans 0.018-0.383.
    Both are recorded and only the first is gated: a study record whose rollout probe
    reads 0.02 (as pixel_ae seed 1's does) must not make the harness disagree."""
    monkeypatch.setattr(script, "HARNESS_R2_TOLERANCE", 1e-9)
    shipped = record["shipped_grid_r2"]
    record = _cell(
        world, world.shared,
        study=_study(embedding_r2=shipped, selection_r2=0.018),
    )
    assert record["controls"]["harness_r2_matches_study"] is True
    assert record["controls"]["study_embedding_selection_r2"] == 0.018
    assert record["controls"]["study_embedding_r2"] == shipped


def test_the_tolerance_sits_between_what_a_correct_pipeline_measures_and_what_a_starved_fit_does():
    """The justification of `HARNESS_R2_TOLERANCE = 0.10`, as numbers, so the docstring cannot
    drift from them. With the fit split the study's 20 training episodes, the RECORDED RUN's
    worst gap between `C_shipped` and the study's `criterion_4.embedding_r2` is 0.0561
    (`controls.study_embedding_r2 - shipped_grid_r2` over the nine records, frozen_ssl
    seed 2; the pre-run one-off check had read 0.0398 and the run did not reproduce it); fit
    on 8 validation episodes every gap was 0.17 to 0.35 (a superseded revision, not
    recomputable). A tolerance at or below the first fires on a correct run (exit 49); at or
    above the second it admits a broken harness."""
    assert script.HARNESS_R2_TOLERANCE == 0.10
    assert 0.0562 < script.HARNESS_R2_TOLERANCE < 0.17


def test_fine_minus_shipped_is_c_less_c_shipped_on_the_score_split_by_value(world, record):
    """`C - C_shipped`, both read on the SCORE split: the finer ridge grid's r2 less the
    shipped grid's, recomputed here with `fit_probe` over each grid on the world's own rows.
    The curved world is shown to put C above C_shipped (by more than rounding), so the two
    signs of the difference are different numbers, and the difference is not the record's
    `shipped_grid_r2` or `embedding_linear` alone.

    THE MUTATIONS THIS EXISTS FOR: `shipped - C`, and a literal `0.0` where the difference
    stood."""
    shared = world.shared
    fit, select, score = (_embedding_rows(world, shared, n) for n in ("fit", "select", "score"))
    fine = fit_probe(fit[0], fit[1], select[0], select[1], ridges=ceiling.CEILING_RIDGES)
    shipped = fit_probe(fit[0], fit[1], select[0], select[1])
    expected = (
        _mean_r2(apply_probe(fine, score[0]), score[1])
        - _mean_r2(apply_probe(shipped, score[0]), score[1])
    )
    assert expected > 0.005
    assert record["fine_minus_shipped_on_score"] == pytest.approx(expected, abs=1e-6)
    assert record["fine_minus_shipped_on_score"] == (
        _r(record, "embedding_linear")["r2"] - record["shipped_grid_r2"]
    )
    assert record["fine_minus_shipped_on_score"] not in (
        record["shipped_grid_r2"], _r(record, "embedding_linear")["r2"],
    )


def test_a_nan_study_number_reads_as_a_disagreement_not_a_match(world):
    record = _cell(world, world.shared, study=_study(embedding_r2=float("nan")))
    assert record["controls"]["harness_r2_matches_study"] is False


def test_the_zero_information_control_is_the_training_mean_against_every_validation_frame(
    world, record,
):
    """The SPEC's definition (283.05 is `the median error of predicting the training mean
    on the 12,073 validation frames`), recomputed here from the privileged arrays: the mean
    over ALL the training episodes (22 here, not only the 20 the fit split holds), against
    EVERY validation frame. The mean is recorded beside the median.

    It is shown to differ from the two nearby quantities it could be mistaken for: the mean
    of the 20 FIT episodes only, and the fit split's rows against the score split's rows.
    Their medians sit apart by more than a map unit, so a control that took either would not
    pass."""
    train = np.concatenate([
        probe_targets(load_episode(p).privileged, KEYS)[:, :2] for p in world.train
    ])
    val = np.concatenate([probe_targets(load_episode(p).privileged, KEYS) for p in world.val])
    errors = np.linalg.norm(val[:, :2] - train.mean(axis=0), axis=1)
    control = record["controls"]["zero_information_recomputed"]
    assert control["mean"] == pytest.approx(float(errors.mean()), abs=1e-9)
    assert control["median"] == pytest.approx(float(np.median(errors)), abs=1e-9)
    assert control["mean"] != control["median"]
    fit_only = np.concatenate([
        probe_targets(load_episode(p).privileged, KEYS)[:, :2] for p in world.train[:20]
    ])
    assert abs(
        float(np.median(np.linalg.norm(val[:, :2] - fit_only.mean(axis=0), axis=1)))
        - control["median"]
    ) > 0.5
    shared = world.shared
    own = ceiling.zero_information_error(
        shared.splits["fit"].targets, shared.splits["score"].targets,
    )
    assert abs(own["median"] - control["median"]) > 1.0


def test_the_widest_rung_control_is_the_widest_rbf_against_linear_by_value(
    world, record,
):
    """`nesting_rbf_at_widest_vs_linear` is the score r2 of the RBF at the WIDEST bandwidth
    alone, jitter selected on the select split, less the linear readout's. Recomputed here
    with `numpy.linalg.solve` per jitter -- not the eigendecomposition the script and the
    ladder use -- on the world's own embedding rows. The length unit is
    `ceiling._median_scale`'s (the fit split holds more than the 512 rows it caps its pairwise
    sample at, and its sampling rule is pinned in `test_ceiling.py`, not re-derived here); what
    this test owns is the widest rung, the jitter chosen on the SELECT split and the score
    read on the SCORE split."""
    shared = world.shared
    (xf, yf), (xs, ys), (xc, yc) = (
        _embedding_rows(world, shared, n) for n in ("fit", "select", "score")
    )
    assert len(xf) > 512  # the cap is reached, so the unit is the sampled one
    mean, scale = xf.mean(0), xf.std(0) + 1e-8
    zf, zs, zc = (xf - mean) / scale, (xs - mean) / scale, (xc - mean) / scale

    def sq(a, b):
        return ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)

    gamma = 1.0 / (2.0 * (max(ceiling.BANDWIDTHS) * ceiling._median_scale(zf)) ** 2)
    gram, on_select, on_score = (np.exp(-gamma * sq(a, zf)) for a in (zf, zs, zc))
    centre = yf.mean(0)
    best = None
    for jitter in ceiling.CEILING_RIDGES:
        alpha = np.linalg.solve(gram + jitter * np.eye(len(zf)), yf - centre)
        got = _mean_r2(on_select @ alpha + centre, ys)
        if best is None or got > best[0]:
            best = (got, alpha)
    widest = _mean_r2(on_score @ best[1] + centre, yc)
    expected = widest - _r(record, "embedding_linear")["r2"]
    assert record["controls"]["nesting_rbf_at_widest_vs_linear"] == pytest.approx(
        expected, abs=1e-6,
    )
    assert expected != 0.0


def test_a_linear_rung_is_written_as_the_string_inf_and_the_file_is_strict_json(
    linear_record, tmp_path,
):
    """`capacity` is `inf` wherever the ladder's linear rung wins, and `inf` is not JSON.
    The record carries the STRING `"inf"` (not `null`, which `write_record` would produce
    from a float inf while filing it under `nonfinite`), the file parses with a
    `parse_constant` that refuses every bare token, and a finite capacity stays a number.

    The PURELY LINEAR world is used because it is where the linear rung wins: it is shown
    to produce an `inf` and a finite capacity in one record before anything is asserted.

    THE MUTATION THIS EXISTS FOR, run against it: `float(readout.capacity)` for
    `_capacity(readout.capacity)`, which `write_record` then nulls."""
    capacities = {name: _r(linear_record, name)["capacity"] for name in script.READOUTS}
    assert capacities["embedding_rbf"] == "inf"
    finite = {n: v for n, v in capacities.items() if v != "inf"}
    assert finite and all(isinstance(v, float) and math.isfinite(v) for v in finite.values())
    path = tmp_path / "record.json"
    write_record(path, linear_record)
    text = path.read_text()
    assert "Infinity" not in text and "NaN" not in text

    def refuse(token):
        raise AssertionError(f"bare {token} in the file")

    loaded = json.loads(text, parse_constant=refuse)
    assert loaded["nonfinite"] == {}
    assert loaded["readouts"]["embedding_rbf"]["capacity"] == "inf"
    assert loaded["readouts"]["embedding_knn"]["jitter"] is None
    assert isinstance(loaded["readouts"]["embedding_knn"]["capacity"], float)
    assert load_record(path)["readouts"]["embedding_rbf"]["capacity"] == "inf"


def test_a_capacity_is_a_finite_number_or_positive_infinity_and_nothing_else():
    """`_capacity` IS `ceiling.capacity_for_record`, the function `motion_headroom` writes
    its bandwidth with: the two scripts once spelled the same datum `"inf"` and `null`."""
    assert script._capacity is ceiling.capacity_for_record
    assert script._capacity(float("inf")) == "inf"
    assert script._capacity(np.float64(0.25)) == 0.25
    assert script._capacity(16) == 16.0
    for bad in (float("nan"), float("-inf")):
        with pytest.raises(ValueError, match="capacity"):
            script._capacity(bad)
    readout = ceiling.Readout(
        name="rbf", r2=0.1, mean_error=1.0, median_error=1.0, capacity=2.0,
        jitter=float("inf"), predicted=np.zeros((2, 4)),
    )
    with pytest.raises(ValueError, match="jitter"):
        script._readout_record(readout)


def test_the_record_has_the_schema_and_not_the_key_write_record_owns(record):
    """`study.write_record` owns the top-level `nonfinite` and RAISES for a record that
    carries one: a record built with `"nonfinite": 0`, as an earlier schema spelled it,
    would crash the first write of the run, after the compute."""
    assert set(record) == {
        "arm", "seed", "step", "git_sha", "record_git_sha", "torch_version", "device",
        "frames", "frame_stride", "rows_per_split", "sequence_frames", "confidence",
        "resamples", "null_draws", "bootstrap_seed", "cell_bootstrap_seed", "split_seed",
        "zero_information_median", "harness_r2_tolerance", "episodes", "groups", "applied",
        "ceiling_ridges", "readouts", "shipped_grid_r2", "fine_minus_shipped_on_score",
        "null_floor", "null_floor_shape", "gains", "controls",
    }
    assert "nonfinite" not in record
    assert list(record["readouts"]) == [
        "pixels_linear", "pixels_rbf", "embedding_linear", "embedding_rbf",
        "embedding_knn", "sequence_linear", "sequence_rbf", "embedding_linear_shipped",
    ]
    assert set(record["gains"]) == {"probe", "pixel", "sequence"}
    for entry in record["readouts"].values():
        assert set(entry) == {"r2", "mean_error", "median_error", "capacity", "jitter"}
    assert set(record["controls"]) == {
        "harness_r2_matches_study", "study_embedding_r2", "study_embedding_selection_r2",
        "nesting_rbf_at_widest_vs_linear", "zero_information_recomputed",
    }
    assert set(record["applied"]) == {"real_latent", "predicted"}
    for entry in record["applied"].values():
        assert set(entry) == {"r2", "mean_error", "mean_error_h1", "median_error", "compares_to"}


def test_the_record_survives_a_round_trip_through_write_record(record, tmp_path):
    path = tmp_path / "ceiling_pixel_ae_seed3.json"
    write_record(path, record)
    loaded = load_record(path)
    assert loaded.pop("nonfinite") == {}
    assert loaded == record


def test_the_record_names_its_provenance_and_its_rows(world, record):
    shared = world.shared
    assert (record["arm"], record["seed"]) == ("pixel_ae", SEED)
    assert record["step"] == STUDY["steps"]
    assert record["record_git_sha"] == STUDY["git_sha"]
    assert record["git_sha"] == git_sha()
    assert record["torch_version"] == torch.__version__
    assert record["device"] == "cpu"
    assert record["split_seed"] == SPLIT_SEED
    assert record["frames"] == 12000 and record["frame_stride"] == 1
    assert record["rows_per_split"] == {n: s.n for n, s in shared.splits.items()}
    assert record["episodes"] == {
        n: [p.name for p in s.paths] for n, s in shared.splits.items()
    }
    assert record["groups"] == shared.splits["score"].groups.tolist()
    assert len(record["groups"]) == record["rows_per_split"]["score"]
    assert len(record["ceiling_ridges"]) == 33
    assert record["ceiling_ridges"] == list(ceiling.CEILING_RIDGES)


CONSTANTS = [
    ("CONFIDENCE", ceiling, 0.9, "confidence"),
    ("RESAMPLES", ceiling, 7, "resamples"),
    ("NULL_DRAWS", ceiling, 2, "null_draws"),
    ("ZERO_INFORMATION_MEDIAN", ceiling, 1.5, "zero_information_median"),
    ("CEILING_RIDGES", ceiling, tuple(RIDGES), "ceiling_ridges"),
    ("SEQUENCE_FRAMES", script, 3, "sequence_frames"),
    ("HARNESS_R2_TOLERANCE", script, 0.5, "harness_r2_tolerance"),
]


@pytest.mark.parametrize("name, owner, patched, key", CONSTANTS, ids=[c[0] for c in CONSTANTS])
def test_each_recorded_constant_moves_when_the_modules_own_name_moves(
    ring_world, monkeypatch, name, owner, patched, key,
):
    """The record states the constant THE MODULE HOLDS, patched where it is DEFINED, so
    the measurement and the record move together and the record cannot name a value the
    computation did not take. Comparing a record with `ceiling.CONFIDENCE` cannot tell a
    read from a literal -- a literal `0.95` EQUALS it -- so each is patched to a value the
    shipped constant is not, ONE AT A TIME: a combined patch would let a literal in one
    place hide behind the right value in another.

    THE MUTATIONS THIS EXISTS FOR, one per case, each a literal where the constant stood:
    `"confidence": 0.95`, `"resamples": 2000`, `"null_draws": 3`,
    `"zero_information_median": 283.05`, `"ceiling_ridges": [...]`, `"sequence_frames": 5`
    and `"harness_r2_tolerance": 0.1`."""
    assert getattr(owner, name) != patched
    monkeypatch.setattr(owner, name, patched)
    shared = script.prepare_shared(ring_world.train, ring_world.val, budget=12000)
    record = _cell(ring_world, shared)
    expected = list(patched) if name == "CEILING_RIDGES" else patched
    assert record[key] == expected


def test_the_budget_the_rows_were_drawn_under_is_what_the_record_states(ring_world):
    """`frames` is the budget `prepare_shared` was handed, and `rows_per_split` says what
    that left: at 64 the budget BINDS, the stride is above 1 and no split exceeds it.

    THE MUTATION THIS EXISTS FOR, run against it: `"frames": 12000`."""
    shared = script.prepare_shared(ring_world.train, ring_world.val, budget=64)
    record = _cell(ring_world, shared)
    assert record["frames"] == 64
    assert record["frame_stride"] == shared.stride > 1
    assert max(record["rows_per_split"].values()) <= 64
    assert record["rows_per_split"] == {n: s.n for n, s in shared.splits.items()}


# ---------------------------------------------------------------------------
# The applied pass: the rollout probe on its own imagined output.
# ---------------------------------------------------------------------------


def _hand_trajectories():
    """`(windows, steps, 2)` arrays for BOTH distributions, on one truth. Each error
    distribution is skewed (so a mean and a median are different numbers), the predicted
    errors are larger than the real-latent ones and GROW with the horizon (so the first
    step's mean is not the pooled mean), and the two distributions are drawn
    independently: no two of the eight statistics coincide, which the first test checks."""
    rng = np.random.default_rng(5)
    truth = rng.normal(size=(7, 9, 2)) * 100.0 + 500.0

    def errors(scale, growth):
        raw = rng.exponential(scale=scale, size=(7, 9, 1)) * rng.choice([-1.0, 1.0], size=(7, 9, 2))
        return raw * (1.0 + growth * np.arange(9))[None, :, None]

    return types.SimpleNamespace(
        positions=truth + errors(40.0, 0.4), positions_real=truth + errors(15.0, 0.0),
        true_positions=truth,
    )


def _definitions(predicted, truth):
    """The five numbers one distribution carries, written out from the definitions: r2 per
    column over all (window, step) rows, the Euclidean error pooled over every row, its two
    statistics, and the mean over windows at step 0 alone."""
    p, t = predicted.reshape(-1, 2), truth.reshape(-1, 2)
    r2 = np.mean([
        1.0 - ((t[:, c] - p[:, c]) ** 2).sum() / ((t[:, c] - t[:, c].mean()) ** 2).sum()
        for c in range(2)
    ])
    pooled = np.sqrt(((p - t) ** 2).sum(axis=1))
    first = np.sqrt(((predicted[:, 0] - truth[:, 0]) ** 2).sum(axis=1))
    return {
        "r2": r2, "mean_error": pooled.mean(), "mean_error_h1": first.mean(),
        "median_error": np.median(pooled),
    }


def test_applied_is_both_distributions_each_with_r2_and_the_map_unit_statistics():
    """Each distribution recomputed here from the arrays, and the fixture shown to
    discriminate first: the two distributions differ in EVERY statistic, and within each
    the pooled mean, the first-step mean and the median are three different numbers, so an
    entry built from the other distribution's arrays, or from the pooled rows where the
    first step belongs, is a different number."""
    trajectories = _hand_trajectories()
    expected = {
        "real_latent": _definitions(trajectories.positions_real, trajectories.true_positions),
        "predicted": _definitions(trajectories.positions, trajectories.true_positions),
    }
    for key in expected["real_latent"]:
        assert abs(expected["real_latent"][key] - expected["predicted"][key]) > 0.01, key
    for stats in expected.values():
        assert len({stats["mean_error"], stats["mean_error_h1"], stats["median_error"]}) == 3
        assert abs(stats["mean_error_h1"] - stats["mean_error"]) > 1.0
    applied = script.applied_quality(trajectories)
    assert set(applied) == {"real_latent", "predicted"}
    for name, stats in expected.items():
        assert applied[name]["r2"] == pytest.approx(stats["r2"], abs=1e-12), name
        for key in ("mean_error", "mean_error_h1", "median_error"):
            assert applied[name][key] == pytest.approx(stats[key], abs=1e-9), (name, key)
        assert set(applied[name]) == {"r2", "mean_error", "mean_error_h1", "median_error", "compares_to"}


def test_each_distribution_is_read_from_its_own_positions_and_the_two_are_not_exchanged():
    """`real_latent` is `positions_real` (the embedding head on the posterior over the true
    frames) and `predicted` is `positions` (the model's own imagined embeddings): handing
    the two arrays over in the other order exchanges the two entries, number for number.

    THE MUTATION THIS EXISTS FOR: `real_latent` built from `positions` and `predicted` from
    `positions_real`."""
    trajectories = _hand_trajectories()
    swapped = types.SimpleNamespace(
        positions=trajectories.positions_real, positions_real=trajectories.positions,
        true_positions=trajectories.true_positions,
    )
    straight, exchanged = script.applied_quality(trajectories), script.applied_quality(swapped)
    for key in ("r2", "mean_error", "mean_error_h1", "median_error"):
        assert straight["real_latent"][key] == exchanged["predicted"][key]
        assert straight["predicted"][key] == exchanged["real_latent"][key]
        assert straight["real_latent"][key] != straight["predicted"][key]


def test_each_distribution_names_the_comparison_it_supports():
    """M3n's 131.88-259.06 is the FLOOR, so it belongs to the real-latent distribution and
    only as a MEAN, at the first step; the predicted distribution has no prior figure. The
    sentence travels with the numbers: it is in the record entry, not only a docstring."""
    applied = script.applied_quality(_hand_trajectories())
    real, predicted = applied["real_latent"]["compares_to"], applied["predicted"]["compares_to"]
    assert "131.88-259.06" in real and "MEAN" in real and "mean_error_h1 ONLY" in real
    assert "nothing earlier" in predicted and "no prior figure" in predicted
    assert "mean_error_h1 ONLY" not in predicted
    assert real != predicted


def test_the_comparison_a_record_states_is_the_modules_own_text(world, monkeypatch):
    """The sentence in the record is the module's constant READ when the cell is measured, so
    it moves with it: patched to values the shipped text is not, one entry at a time.

    THE MUTATION THIS EXISTS FOR: the sentences written into `_applied_entry` as literals."""
    patched = {"real_latent": "an A text", "predicted": "a B text"}
    assert script.APPLIED_COMPARISONS != patched
    monkeypatch.setattr(script, "APPLIED_COMPARISONS", patched)
    record = _cell(world, world.shared)
    assert {n: e["compares_to"] for n, e in record["applied"].items()} == patched


def test_the_record_carries_the_applied_figures_of_the_rollout_it_was_handed(world, monkeypatch):
    """`measure_cell` runs the rollout at the cell's own seed, context, horizon, device and
    backbone through the cell's own probe, and records `applied_quality` of what comes
    back. The stand-in returns the hand-built arrays and records what it was called with, so
    each argument is bound to a value the test chose (the seed is neither 0 nor the base)."""
    calls = []

    def fake(model, val_paths, probe, **kwargs):
        calls.append((model, list(val_paths), probe, kwargs))
        return _hand_trajectories()

    monkeypatch.setattr(script, "reference_trajectories", fake)
    model = world.model()
    record = _cell(world, world.shared, model=model, feature_backbone="dinov2")
    assert record["applied"] == script.applied_quality(_hand_trajectories())
    ((got_model, got_paths, got_probe, kwargs),) = calls
    assert got_model is model and got_probe is world.probe and got_paths == world.val
    assert kwargs == {
        "context": CONTEXT, "horizon": HORIZON, "seed": SEED, "device": CPU,
        "feature_backbone": "dinov2",
    }


def _wobbling_episode(length: int, index: int):
    episode = synthetic_episode(length)
    t = np.arange(length + 1, dtype=np.float64)
    episode.privileged[:, 1] = (10.0 * t + 4.0 * np.sin(0.9 * t + 0.9 * index)).astype(np.float32)
    episode.privileged[:, 2] = (-3.0 * t + 3.0 * np.cos(0.55 * t - 0.9 * index)).astype(np.float32)
    return episode


def test_applied_over_the_real_rollout_is_the_closed_form_of_a_drifting_model(tmp_path):
    """The real `reference_trajectories`, on `DriftingModel` (a perfect filter whose dynamics
    advance the frame tag by TWO per action) over wobbling episodes, against a closed form
    derived without it. The window starting at `s` ends its context at tag `s + context`.

      * PREDICTED: step h imagines tag `s + context + 2h`; the probe is a linear map from tag
        to position.
      * REAL-LATENT: the filter sees the real future frame, whose tag is `s + context + h`,
        so the probe's error there is its own residual and nothing the dynamics did.

    The truth is the privileged position at frame `s + context + h` for both. The two
    distributions are shown to be far apart (the drift is 2x), so the entries cannot be
    exchanged unnoticed.

    AND `mean_error_h1` IS M3n's STATISTIC: `evaluate_rollout`'s own curves, which M3n's
    131.88-259.06 was read from (`floor_position[0]`), reproduce it. That is rollout.py's
    computation, not this script's."""
    from mbfps.eval.diagnostics import reference_trajectories
    from mbfps.eval.rollout import evaluate_rollout

    episodes = [_wobbling_episode(n, i) for i, n in enumerate((20, 31, 26))]
    paths = []
    for i, episode in enumerate(episodes):
        path = tmp_path / f"ep_{i:06d}_len{episode.length:05d}.npz"
        save_episode(episode, path)
        paths.append(path)
    tags = np.concatenate([np.arange(e.length + 1, dtype=np.float64)[:, None] for e in episodes])
    probe = fit_probe(
        tags, np.concatenate([probe_targets(e.privileged, KEYS) for e in episodes]), ridge=1e-8,
    )
    predicted, real, truth = [], [], []
    for episode in episodes:
        for start in window_starts(episode.length, CONTEXT, HORIZON):
            steps = np.arange(1, HORIZON + 1, dtype=np.float64)
            predicted.append(apply_probe(probe, (start + CONTEXT + 2 * steps)[:, None])[:, :2])
            real.append(apply_probe(probe, (start + CONTEXT + steps)[:, None])[:, :2])
            truth.append(probe_targets(episode.privileged, KEYS)[start + CONTEXT + 1:
                                                                 start + CONTEXT + 1 + HORIZON, :2])
    predicted, real, truth = (np.stack(a) for a in (predicted, real, truth))
    expected = {
        "predicted": _definitions(predicted, truth), "real_latent": _definitions(real, truth),
    }
    assert expected["predicted"]["mean_error"] > 5 * expected["real_latent"]["mean_error"] > 0.0
    applied = script.applied_quality(reference_trajectories(
        DriftingModel(), paths, probe, context=CONTEXT, horizon=HORIZON, seed=SEED,
        device=CPU, feature_backbone=None,
    ))
    for name, stats in expected.items():
        assert applied[name]["mean_error"] == pytest.approx(stats["mean_error"], abs=1e-6), name
        assert applied[name]["mean_error_h1"] == pytest.approx(stats["mean_error_h1"], abs=1e-6)
        assert applied[name]["median_error"] == pytest.approx(stats["median_error"], abs=1e-6), name
        assert applied[name]["r2"] == pytest.approx(stats["r2"], abs=1e-9), name
    curves = evaluate_rollout(
        DriftingModel(), paths, probe, context=CONTEXT, horizon=HORIZON, seed=SEED,
        device=CPU, feature_backbone=None,
    )
    assert applied["real_latent"]["mean_error_h1"] == pytest.approx(
        float(curves.floor_position[0]), abs=1e-6,
    )
    assert applied["predicted"]["mean_error_h1"] == pytest.approx(
        float(curves.rssm_position[0]), abs=1e-6,
    )
    assert applied["real_latent"]["mean_error_h1"] != applied["predicted"]["mean_error_h1"]


# ---------------------------------------------------------------------------
# measure_cell's refusals.
# ---------------------------------------------------------------------------


def test_a_non_finite_readout_is_refused_by_name_before_any_interval(world, monkeypatch):
    real = ceiling.score_readout

    def poisoned(name, fit, select, score):
        readout = real(name, fit, select, score)
        if name == "knn":
            readout = ceiling.Readout(
                name=readout.name, r2=float("nan"), mean_error=readout.mean_error,
                median_error=readout.median_error, capacity=readout.capacity,
                jitter=readout.jitter, predicted=readout.predicted,
            )
        return readout

    monkeypatch.setattr(ceiling, "score_readout", poisoned)
    with pytest.raises(SystemExit, match=r"embedding_knn is not finite \(r2=nan\)"):
        _cell(world, world.shared)


def test_a_rollout_over_other_episodes_than_the_rows_were_drawn_from_is_refused(world):
    with pytest.raises(SystemExit, match="not the episodes the shared select and score rows"):
        _cell(world, world.shared, val_paths=world.val[:-1])


def test_a_fit_split_that_holds_a_validation_episode_is_refused(world, tmp_path):
    """The fit split is the study's TRAINING episodes, scored on none of them. Rows built
    with a validation episode among the 'training' ones would be fit on what they are
    scored on, and the refusal names it. (The shared rows here are built from a training
    list that holds one validation episode.)

    THE MUTATION THIS EXISTS FOR: the overlap check dropped."""
    leaky = script.prepare_shared([world.val[0]] + world.train[:19], world.val, budget=12000)
    assert world.val[0] in leaky.splits["fit"].paths
    with pytest.raises(SystemExit, match=r"fit split holds validation episode\(s\) \['ep_000000"):
        _cell(world, leaky)


@pytest.mark.parametrize("remove", [
    ("steps",), ("git_sha",), ("filtering",), ("probe",),
    ("filtering", "criterion_4"), ("probe", "embedding_selection_r2"),
])
def test_a_study_record_missing_a_number_is_refused_not_defaulted(world, remove):
    study = _study()
    holder = study
    for key in remove[:-1]:
        holder = holder[key]
    del holder[remove[-1]]
    with pytest.raises(SystemExit, match="not a study record"):
        _cell(world, world.shared, study=study)


# ---------------------------------------------------------------------------
# measure_phase: the cells, in order, each from its own model.
# ---------------------------------------------------------------------------

PHASE_CELLS = (("pixel_ae", 1), ("pixel_ae", 2), ("frozen_ssl", 1), ("frozen_ssl", 2))
PHASE_VIEWS = 6
assert all(seed not in (0, SPLIT_SEED, BOOTSTRAP_SEED) for _, seed in PHASE_CELLS)


def _phase_env(tmp_path, monkeypatch, *, missing=(), refuse=None, study=None):
    """A real `ReplayBuffer`, the real study split and halves, the real `prepare_shared` and
    `measure_cell`; only `load_cell` and `prepare_cell` are faked, because they need nine
    checkpoints. The fakes BEHAVE: `prepare_cell` refuses a cell unless `args.out` is the
    directory holding its checkpoint (the study directory, which the record directory is
    not), and hands each cell a DIFFERENT model -- its own four-view block of the world --
    so a record paired with the wrong cell reads differently."""
    data = tmp_path / "data"
    world = _make_world(
        data, kind="linear", views=PHASE_VIEWS, models=len(PHASE_CELLS), n_val=45, n_train=0,
        sigma_e=60.0,
    )
    source = tmp_path / "study"
    source.mkdir()
    for arm, seed in PHASE_CELLS:
        (source / f"world_model_{arm}_seed{seed}.pt").write_bytes(b"")
    models = {cell: world.model(first=i) for i, cell in enumerate(PHASE_CELLS)}
    prepared_for, seen_args = [], []

    def fake_load_cell(directory, arm, seed):
        if (arm, seed) in missing:
            raise script.CellMissing(f"{arm} seed {seed}: no checkpoint at {directory}")
        return types.SimpleNamespace(
            arm=arm, seed=seed, record=study or _study(), diagnostic={},
            checkpoint=Path(directory) / f"world_model_{arm}_seed{seed}.pt",
        )

    def fake_prepare_cell(args, cell, device, train, val):
        prepared_for.append((cell.arm, cell.seed))
        seen_args.append(args)
        if not (Path(args.out) / f"world_model_{cell.arm}_seed{cell.seed}.pt").exists():
            return script.EXIT_NO_CHECKPOINTS, None
        if refuse == (cell.arm, cell.seed):
            return script.EXIT_RECORD_MISMATCH, None
        return script.EXIT_OK, types.SimpleNamespace(
            model=models[(cell.arm, cell.seed)], embedding_probe=world.probe,
            common={"feature_backbone": None}, context=CONTEXT, horizon=HORIZON,
            reference=None,
        )

    monkeypatch.setattr(script, "load_cell", fake_load_cell)
    monkeypatch.setattr(script, "prepare_cell", fake_prepare_cell)
    args = types.SimpleNamespace(
        out=tmp_path / "ceiling_out", source=source, data=data, device="cpu", context=None,
        horizon=None, arms=["pixel_ae", "frozen_ssl"], seeds=[1, 2], frames=None,
        bootstrap_seed=BOOTSTRAP_SEED, models=models, prepared_for=prepared_for,
        seen_args=seen_args,
    )
    return args


def _val_split(args):
    from mbfps.data.buffer import ReplayBuffer
    from mbfps.data.split import VAL_FRACTION, episode_split

    return episode_split(
        ReplayBuffer(args.data, capacity_transitions=10**9).episode_paths(),
        val_fraction=VAL_FRACTION, seed=SPLIT_SEED,
    )


@pytest.fixture(scope="module")
def phase_run(tmp_path_factory):
    """One real run of the phase over four cells, shared by the tests that only READ what
    it wrote. Tests that change a patch or a refusal build their own."""
    tmp_path = tmp_path_factory.mktemp("phase")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ceiling, "RESAMPLES", SMALL)
        args = _phase_env(tmp_path, patch)
        built = []
        real = script.prepare_shared
        patch.setattr(script, "prepare_shared", lambda *a, **k: built.append(real(*a, **k)) or built[-1])
        status = script.measure_phase(args)
    args.status, args.shared_built = status, built
    return args


def test_the_phase_writes_one_labelled_record_per_cell_from_that_cells_own_model(phase_run):
    """Nine records once went out serialised to ONE payload with the labels dropped. Here
    the four cells carry four different models, and each file must hold its own cell's
    numbers. THE SEED IS BOUND THREE TIMES, because the cells' seeds are 1 and 2 and the
    bootstrap seed is 7: in the file NAME, in the record (`seed`), and in the interval seed
    derived from it.

    THE MUTATIONS THIS EXISTS FOR, each of which a one-seed-0 fixture lets through: the
    record path built with `0` for `cell.seed`, `seed=0` handed to `measure_cell`, and
    `bootstrap_seed=0`."""
    args = phase_run
    assert args.status == script.EXIT_OK
    assert args.prepared_for == list(PHASE_CELLS)
    assert sorted(p.name for p in args.out.iterdir()) == [
        "ceiling_frozen_ssl_seed1.json", "ceiling_frozen_ssl_seed2.json",
        "ceiling_pixel_ae_seed1.json", "ceiling_pixel_ae_seed2.json",
    ]
    seen = set()
    for arm, seed in PHASE_CELLS:
        carried = load_record(args.out / f"ceiling_{arm}_seed{seed}.json")
        assert (carried["arm"], carried["seed"]) == (arm, seed)
        assert carried["bootstrap_seed"] == BOOTSTRAP_SEED
        assert carried["cell_bootstrap_seed"] == script.cell_bootstrap_seed(
            BOOTSTRAP_SEED, arm, seed,
        )
        assert carried["step"] == STUDY["steps"] and carried["git_sha"] == git_sha()
        assert carried["nonfinite"] == {}
        seen.add(carried["readouts"]["embedding_linear"]["r2"])
    assert len(seen) == len(PHASE_CELLS)


def test_the_raw_pixel_readouts_are_computed_once_and_identical_in_every_record(phase_run):
    """They use no model: one `Shared` is built, and the same two `Readout`s are written into
    every cell's record -- the free bitwise control the read phase asserts."""
    args = phase_run
    assert len(args.shared_built) == 1
    expected = {
        name: script._readout_record(r) for name, r in args.shared_built[0].pixel_readouts.items()
    }
    for arm, seed in PHASE_CELLS:
        carried = load_record(args.out / f"ceiling_{arm}_seed{seed}.json")
        for name in script.PIXEL_READOUTS:
            assert carried["readouts"][name] == expected[name], (arm, seed, name)
    embeddings = [
        load_record(args.out / f"ceiling_{arm}_seed{seed}.json")["readouts"]["embedding_rbf"]["r2"]
        for arm, seed in PHASE_CELLS
    ]
    assert len(set(embeddings)) == len(PHASE_CELLS)


def test_the_phase_fits_on_the_studys_training_episodes_and_halves_its_validation_set(phase_run):
    """THE STUDY'S OWN SPLIT, rebuilt here through the buffer and `episode_split`: the fit
    split is the first 20 of its training episodes, and select and score are the two halves of
    its validation episodes (5 and 4 of 9 here, the odd episode in `select`).

    THE MUTATIONS THIS EXISTS FOR: the 8-episode fit drawn from validation, and a fit split
    that is not the leading block of the study's own training list."""
    args = phase_run
    train, val = _val_split(args)
    assert len(train) == 36 and len(val) == 9
    record = load_record(args.out / "ceiling_pixel_ae_seed1.json")
    assert record["episodes"]["fit"] == [p.name for p in train[:20]]
    held = sorted(record["episodes"]["select"] + record["episodes"]["score"])
    assert held == sorted(p.name for p in val)
    assert not set(record["episodes"]["fit"]) & set(held)
    assert not set(record["episodes"]["select"]) & set(record["episodes"]["score"])
    assert {k: len(v) for k, v in record["episodes"].items()} == {
        "fit": 20, "select": 5, "score": 4,
    }
    assert record["split_seed"] == SPLIT_SEED
    assert record["frame_stride"] == 1 and record["frames"] == 12000
    first = script.SEQUENCE_FRAMES - 1
    rows = lambda names: sum(  # noqa: E731
        load_episode(next(p for p in train + val if p.name == n)).length + 1 - first for n in names
    )
    assert record["rows_per_split"] == {
        name: rows(names) for name, names in record["episodes"].items()
    }


def test_the_zero_information_baseline_is_the_training_mean_over_the_studys_val_frames(phase_run):
    train, val = _val_split(phase_run)
    mu = np.concatenate([probe_targets(load_episode(p).privileged, KEYS) for p in train])[:, :2].mean(0)
    truth = np.concatenate([probe_targets(load_episode(p).privileged, KEYS) for p in val])[:, :2]
    errors = np.linalg.norm(truth - mu, axis=1)
    got = load_record(phase_run.out / "ceiling_pixel_ae_seed1.json")["controls"][
        "zero_information_recomputed"
    ]
    assert got["mean"] == pytest.approx(float(errors.mean()), abs=1e-9)
    assert got["median"] == pytest.approx(float(np.median(errors)), abs=1e-9)


def test_prepare_cell_is_handed_the_study_directory_and_not_the_record_directory(phase_run):
    """`prepare_cell` loads the checkpoint from `args.out`; the record directory holds none
    of the nine, and handing `args` through would refuse EVERY cell. The fake refuses unless
    it is the study directory, so all four cells reaching a record is the evidence; the
    shim's other fields are the flags'."""
    for seen in phase_run.seen_args:
        assert seen.out == phase_run.source != phase_run.out
        assert (seen.device, seen.context, seen.horizon) == ("cpu", None, None)
    assert len(phase_run.seen_args) == len(PHASE_CELLS)


def test_the_phase_prints_one_line_per_cell_and_names_the_numbers(tmp_path, monkeypatch, capsys):
    args = _phase_env(tmp_path, monkeypatch)
    args.arms, args.seeds = ["pixel_ae"], [1]
    assert script.measure_phase(args) == script.EXIT_OK
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    record = load_record(args.out / "ceiling_pixel_ae_seed1.json")
    assert lines[0].startswith("pixel_ae seed 1: D-C-floor ")
    for needle in (
        f"{record['gains']['pixel']['point']:+.4f}", f"{record['gains']['sequence']['point']:+.4f}",
        f"null floor {record['null_floor']:+.4f}", f"C_shipped {record['shipped_grid_r2']:.3f}",
        "rows " + "/".join(str(record["rows_per_split"][s]) for s in ("fit", "select", "score")),
        "at stride 1", str(args.out / "ceiling_pixel_ae_seed1.json"),
    ):
        assert needle in lines[0], needle
    assert ("harness agrees" in lines[0]) == record["controls"]["harness_r2_matches_study"]
    assert ("HARNESS DISAGREES" in lines[0]) != record["controls"]["harness_r2_matches_study"]


def test_a_cell_whose_harness_disagrees_is_named_in_its_line_and_the_run_carries_on(
    tmp_path, monkeypatch, capsys,
):
    """Nothing in the measure phase decides anything: a missed control is RECORDED and
    NAMED, the status is still `EXIT_OK` and the record is written. The read refuses it."""
    monkeypatch.setattr(script, "HARNESS_R2_TOLERANCE", 1e-9)
    args = _phase_env(tmp_path, monkeypatch, study=_study(embedding_r2=0.999))
    args.arms, args.seeds = ["pixel_ae"], [1]
    assert script.measure_phase(args) == script.EXIT_OK
    assert "HARNESS DISAGREES with the study" in capsys.readouterr().out
    record = load_record(args.out / "ceiling_pixel_ae_seed1.json")
    assert record["controls"]["harness_r2_matches_study"] is False


def test_the_budget_flag_reaches_the_rows_and_its_default_is_the_modules_own(
    tmp_path, monkeypatch,
):
    """`--frames` is the budget per split; left unset it is `ceiling.FRAME_BUDGET`, read when
    the phase RUNS. Patched to 64 where no flag is given, the budget binds (stride above 1,
    no split above 64) and the record says 64; an explicit flag wins over the module's.

    THE MUTATIONS THIS EXISTS FOR: a literal `12000` where the default stood, and `frames`
    never reaching `prepare_shared`."""
    monkeypatch.setattr(ceiling, "FRAME_BUDGET", 64)
    args = _phase_env(tmp_path, monkeypatch)
    args.arms, args.seeds = ["pixel_ae"], [1]
    assert script.measure_phase(args) == script.EXIT_OK
    record = load_record(args.out / "ceiling_pixel_ae_seed1.json")
    assert record["frames"] == 64 and record["frame_stride"] > 1
    assert max(record["rows_per_split"].values()) <= 64
    args.frames = 90
    args.out = tmp_path / "other_out"
    assert script.measure_phase(args) == script.EXIT_OK
    explicit = load_record(args.out / "ceiling_pixel_ae_seed1.json")
    assert explicit["frames"] == 90
    assert max(explicit["rows_per_split"].values()) <= 90
    assert explicit["frame_stride"] < record["frame_stride"]


def test_the_phase_names_a_missing_cell_and_measures_nothing(tmp_path, monkeypatch, capsys):
    args = _phase_env(tmp_path, monkeypatch, missing={("frozen_ssl", 2)})
    assert script.measure_phase(args) == script.EXIT_NO_CHECKPOINTS
    assert "frozen_ssl seed 2" in capsys.readouterr().out
    assert args.prepared_for == []
    assert not args.out.exists() or list(args.out.iterdir()) == []


def test_the_phase_stops_at_the_first_refusal_and_writes_nothing_for_it(tmp_path, monkeypatch):
    """THE MUTATION THIS EXISTS FOR, run against it: `return status` -> `continue` where
    `measure_phase` meets a cell `prepare_cell` refused: the phase would exit 0 after a
    refused cell, the worst available outcome of a run that costs compute."""
    args = _phase_env(tmp_path, monkeypatch, refuse=("frozen_ssl", 1))
    assert script.measure_phase(args) == script.EXIT_RECORD_MISMATCH
    assert sorted(p.name for p in args.out.iterdir()) == [
        "ceiling_pixel_ae_seed1.json", "ceiling_pixel_ae_seed2.json",
    ]


def test_a_refused_first_cell_costs_no_pixel_readouts(tmp_path, monkeypatch):
    """The model-free rows are built after the FIRST cell has passed `prepare_cell`, so a
    refusal there is cheap."""
    args = _phase_env(tmp_path, monkeypatch, refuse=("pixel_ae", 1))
    built = []
    monkeypatch.setattr(script, "prepare_shared", lambda *a, **k: built.append(1))
    assert script.measure_phase(args) == script.EXIT_RECORD_MISMATCH
    assert built == []


def test_the_phase_measures_a_repeated_arm_once(tmp_path, monkeypatch):
    args = _phase_env(tmp_path, monkeypatch)
    args.arms, args.seeds = ["pixel_ae", "pixel_ae"], [1, 1]
    assert script.measure_phase(args) == script.EXIT_OK
    assert args.prepared_for == [("pixel_ae", 1)]


def test_a_cell_that_raises_stops_the_phase_and_is_not_swallowed(tmp_path, monkeypatch):
    """The OTHER way a failed cell can exit 0: an exception caught and dropped. The cell that
    raised wrote nothing, the cells after it are never measured, and the exception reaches the
    caller (a traceback, status 1) rather than a quiet `EXIT_OK`.

    THE MUTATION THIS EXISTS FOR: wrapping the `measure_cell` call in `try: ... except
    Exception: continue`."""
    args = _phase_env(tmp_path, monkeypatch)
    real = script.measure_cell

    def explode_on_the_second_cell(*a, **k):
        if k["arm"] == "pixel_ae" and k["seed"] == 2:
            raise RuntimeError("boom")
        return real(*a, **k)

    monkeypatch.setattr(script, "measure_cell", explode_on_the_second_cell)
    with pytest.raises(RuntimeError, match="boom"):
        script.main([
            "--phase", "measure", "--source", str(args.source), "--out", str(args.out),
            "--data", str(args.data), "--device", "cpu", "--arms", "pixel_ae,frozen_ssl",
            "--seeds", "1,2",
        ])
    assert args.prepared_for == [("pixel_ae", 1), ("pixel_ae", 2)]
    assert [p.name for p in args.out.iterdir()] == ["ceiling_pixel_ae_seed1.json"]


def test_a_failed_cell_is_a_nonzero_exit_through_main(tmp_path, monkeypatch):
    """The brief's test patched `measure_cell` to raise and expected a nonzero RETURN; this
    script, like `motion_headroom.py`, RETURNS the statuses it can name (a refused cell is
    14) and lets an exception propagate (above). Either way the exit is not 0."""
    args = _phase_env(tmp_path, monkeypatch, refuse=("frozen_ssl", 1))
    status = script.main([
        "--phase", "measure", "--source", str(args.source), "--out", str(args.out),
        "--data", str(args.data), "--device", "cpu", "--arms", "pixel_ae,frozen_ssl",
        "--seeds", "1,2",
    ])
    assert status == script.EXIT_RECORD_MISMATCH == 14


# ---------------------------------------------------------------------------
# main: the three phases, the plan gates, and the command line.
# ---------------------------------------------------------------------------


def _main(monkeypatch, argv, *, measure=0, read=0):
    """`main(argv)` with BOTH phases replaced by recorders, returning `(status, phases_run)`.
    What this pins is the WIRING: which phase runs, in which order, and whether a failed
    measure stops the read."""
    ran = []

    def fake_measure(args):
        ran.append("measure")
        return measure

    def fake_read(args):
        ran.append("read")
        return read

    monkeypatch.setattr(script, "measure_phase", fake_measure)
    monkeypatch.setattr(script, "read_phase", fake_read)
    return script.main(argv), ran


def test_main_runs_the_phases_the_flag_names_and_returns_the_last_status(monkeypatch):
    assert _main(monkeypatch, ["--phase", "measure"]) == (0, ["measure"])
    assert _main(monkeypatch, ["--phase", "read"]) == (0, ["read"])
    assert _main(monkeypatch, ["--phase", "all"]) == (0, ["measure", "read"])
    assert _main(monkeypatch, ["--phase", "read"], read=49) == (49, ["read"])
    assert _main(monkeypatch, ["--phase", "all"], read=50) == (50, ["measure", "read"])


def test_a_failed_measure_stops_phase_all_before_it_reads(monkeypatch):
    assert _main(monkeypatch, ["--phase", "all"], measure=14, read=0) == (14, ["measure"])


@pytest.mark.parametrize("status", [11, 12, 14])
def test_a_failed_measure_is_the_exit_status_of_phase_measure_too(monkeypatch, status):
    """`--phase measure` has no read after it, and that is exactly where a failed measure's
    status could be dropped: the guard that stops the read in `--phase all` is also what
    RETURNS the status. The statuses are literals, and `read` answers 0 so a status read from
    the wrong phase is a different number.

    THE MUTATION THIS EXISTS FOR: narrowing `main`'s `if status != EXIT_OK:` to `and
    args.phase == "all"`."""
    assert _main(monkeypatch, ["--phase", "measure"], measure=status, read=0) == (
        status, ["measure"],
    )


def test_the_default_phase_is_read_and_the_parser_names_the_three():
    """A default of `measure` would make a bare invocation pay for nine cells; `read` costs
    none and refuses by name when there are no records."""
    args = script._parser().parse_args([])
    assert args.phase == "read"
    assert script.PHASES == ("all", "measure", "read")
    with pytest.raises(SystemExit) as caught:
        script._parser().parse_args(["--phase", "readd"])
    assert caught.value.code == 2


def test_the_parser_defaults_are_the_full_nine_cell_plan():
    args = script._parser().parse_args([])
    assert args.arms == ["pixel_ae", "frozen_ssl", "random_vit"]
    assert args.seeds == [0, 1, 2]
    assert args.out == Path("runs/m3o_ceiling")
    assert args.source == Path("runs/m3_study_v2")
    assert args.data == Path("data/my_way_home")
    assert args.frames is None
    assert args.bootstrap_seed == 0
    assert args.context is None and args.horizon is None


@pytest.mark.parametrize(
    "argv",
    [
        ["--arms", "frozen_ssl,pixel_ae,random_vit", "--seeds", "0,1,2"],
        ["--arms", "frozen_ssl", "pixel_ae", "random_vit", "--seeds", "0", "1", "2"],
        ["--arms", "frozen_ssl,pixel_ae", "random_vit", "--seeds", "0,1", "2"],
    ],
    ids=["commas", "spaces", "mixed"],
)
def test_a_list_flag_reads_commas_and_spaces_alike(argv):
    """The plan's own commands pass `--arms frozen_ssl,pixel_ae,random_vit --seeds 0,1,2`
    -- one comma-joined token each -- and `prediction_burden`'s spelling is space-separated.
    Read as a single token, `"0,1,2"` is not an integer and `"frozen_ssl,pixel_ae,random_vit"`
    is not an arm, so a parser that took only one spelling would refuse the run's own
    command line after the checkpoints were already loaded."""
    args = script._parser().parse_args(argv)
    assert args.arms == ["frozen_ssl", "pixel_ae", "random_vit"]
    assert args.seeds == [0, 1, 2]
    assert all(isinstance(value, int) for value in args.seeds)


def test_the_runs_own_command_line_parses_with_every_flag_it_uses():
    args = script._parser().parse_args([
        "--phase", "measure", "--out", "runs/m3o_ceiling",
        "--arms", "frozen_ssl,pixel_ae,random_vit", "--seeds", "0,1,2",
        "--frames", "12000", "--bootstrap-seed", "0",
    ])
    assert (args.frames, args.bootstrap_seed, args.phase) == (12000, 0, "measure")
    assert args.out == Path("runs/m3o_ceiling")


def test_a_list_flag_drops_repeats_and_refuses_what_it_cannot_read():
    args = script._parser().parse_args(["--arms", "pixel_ae,pixel_ae", "--seeds", "1,1,2"])
    assert (args.arms, args.seeds) == (["pixel_ae"], [1, 2])
    for argv in (["--arms", "pixel_aee"], ["--seeds", "zero"], ["--seeds", "1,,x"], ["--arms", ","]):
        with pytest.raises(SystemExit) as caught:
            script._parser().parse_args(argv)
        assert caught.value.code == 2, argv


def test_phase_all_refuses_a_plan_it_could_not_read_before_measuring_anything(monkeypatch):
    """`--phase all --arms pixel_ae` would be the whole measure followed by a refusal at the
    read. `--phase measure` is allowed the same plan: the milestone's smoke is one cell.

    THE MUTATION THIS EXISTS FOR: dropping `require_readable_plan` from `main`."""
    ran = []
    monkeypatch.setattr(script, "measure_phase", lambda args: ran.append("measure") or 0)
    with pytest.raises(SystemExit) as caught:
        script.main(["--phase", "all", "--arms", "pixel_ae", "--seeds", "0,1,2"])
    assert f"ARMS_REQUIRED={ceiling.ARMS_REQUIRED}" in str(caught.value)
    assert ran == []
    assert _main(monkeypatch, ["--phase", "measure", "--arms", "pixel_ae"]) == (0, ["measure"])


@pytest.mark.parametrize(
    "seeds, distinct",
    [
        pytest.param(["0", "1"], 2, id="one short of the minimum"),
        pytest.param(["5"], 1, id="one seed"),
        pytest.param(["5"] * ceiling.SEEDS_MINIMUM, 1, id="enough listed, one distinct"),
    ],
)
def test_phase_all_refuses_too_few_seeds_before_measuring_anything(monkeypatch, seeds, distinct):
    """`--phase all --seeds 0 1` would measure all six cells and then raise at the read: every
    arm carries two seeds. The plan is known at the start, so the refusal is made there, and
    `ran == []` says it came BEFORE the measure.

    THE MUTATIONS THIS EXISTS FOR: dropping the check from `main`, and counting the seeds as
    listed instead of as distinct."""
    ran = []
    monkeypatch.setattr(script, "measure_phase", lambda args: ran.append("measure") or 0)
    monkeypatch.setattr(script, "read_phase", lambda args: ran.append("read") or 0)
    with pytest.raises(SystemExit) as caught:
        script.main(["--phase", "all", "--seeds", *seeds])
    message = str(caught.value)
    assert f"SEEDS_MINIMUM={ceiling.SEEDS_MINIMUM}" in message, message
    assert f"{distinct} seed(s)" in message, message
    assert ran == []


def test_a_seed_plan_short_of_the_minimum_is_refused_only_where_it_would_waste_a_measure(
    monkeypatch,
):
    """The seed check belongs to `--phase all` and nothing else: `--phase measure` is the
    smoke and may run one seed, and `--phase read` must REACH `read_phase` with a short plan
    (where the arm-short ValueError comes from). Exactly `SEEDS_MINIMUM` seeds is a plan.

    THE MUTATIONS THIS EXISTS FOR: the check applied to `--phase measure`, to `--phase read`,
    and `<=` for `<`."""
    short = ["--seeds", *(str(i) for i in range(ceiling.SEEDS_MINIMUM - 1))]
    assert _main(monkeypatch, ["--phase", "measure", *short]) == (0, ["measure"])
    assert _main(monkeypatch, ["--phase", "read", *short], read=50) == (50, ["read"])
    enough = ["--seeds", *(str(i) for i in range(ceiling.SEEDS_MINIMUM))]
    assert _main(monkeypatch, ["--phase", "all", *enough]) == (0, ["measure", "read"])


def test_the_smoke_command_is_refused_before_it_touches_anything(tmp_path, monkeypatch):
    """The plan's step 6: `--phase all --arms pixel_ae --seeds 0 --frames 400` must be a
    NONZERO exit refusing on the gate, with no output directory created and no cell loaded.
    A zero exit would mean the gate is not wired."""
    loaded = []
    monkeypatch.setattr(script, "load_cell", lambda *a: loaded.append(a))
    out = tmp_path / "smoke"
    with pytest.raises(SystemExit) as caught:
        script.main([
            "--phase", "all", "--out", str(out), "--arms", "pixel_ae", "--seeds", "0",
            "--frames", "400", "--bootstrap-seed", "0",
        ])
    assert caught.value.code not in (0, None)
    assert "ARMS_REQUIRED" in str(caught.value)
    assert loaded == [] and not out.exists()


# ---------------------------------------------------------------------------
# read: what pooling these records requires, and the reading.
# ---------------------------------------------------------------------------


def _iv(point, half=0.05):
    return {"point": point, "ci_low": point - half, "ci_high": point + half}


GAINS_FOR = {
    # probe resolvably above zero; pixel and sequence stay out of it
    "PROBE_LIMITED": {"probe": _iv(0.20), "pixel": _iv(0.30), "sequence": _iv(0.02)},
    # probe not resolvable, pixel resolvable
    "REPRESENTATION_LIMITED": {"probe": _iv(-0.02), "pixel": _iv(0.30), "sequence": _iv(0.12)},
    # neither resolvable
    "OBSERVATION_LIMITED": {"probe": _iv(0.01), "pixel": _iv(0.02), "sequence": _iv(-0.20)},
}
"""The three gains of a cell that reads as each verdict, each triple distinct from the other
two so that no exchange of two fields goes unseen."""


ZERO_RECOMPUTED = {"mean": 291.36, "median": 275.02}
"""The zero-information pair as the real run recomputes it, over every training episode
(`test_the_real_split_and_the_zero_information_figures_are_what_the_script_documents`). The
records carry it beside `zero_information_median`, the spec's pinned 283.05 from the leading
24 episodes, and the two MEDIANS DIFFER (275.02 vs 283.05) exactly as they do on the real
records: a median ratio read against the pinned figure is 2.9% low, and every test that
reads a percentage can see it."""


def _record_for(base, cell, verdict, *, harness=True):
    r = copy.deepcopy(base)
    r["arm"], r["seed"] = cell
    r["cell_bootstrap_seed"] = script.cell_bootstrap_seed(BOOTSTRAP_SEED, *cell)
    r["gains"] = copy.deepcopy(GAINS_FOR[verdict])
    r["controls"]["harness_r2_matches_study"] = harness
    r["controls"]["study_embedding_r2"] = (
        r["shipped_grid_r2"] if harness else r["shipped_grid_r2"] + 2 * r["harness_r2_tolerance"]
    )
    r["controls"]["zero_information_recomputed"] = dict(ZERO_RECOMPUTED)
    return r


def _records_from(base, verdicts, *, harness=True):
    """`verdicts` is `{cell: verdict}` or a list of nine verdicts in `NINE_CELLS` order."""
    if not isinstance(verdicts, dict):
        verdicts = dict(zip(NINE_CELLS, verdicts))
    return {cell: _record_for(base, cell, v, harness=harness) for cell, v in verdicts.items()}


def _write_records(out, records):
    out.mkdir(parents=True, exist_ok=True)
    for (arm, seed), record in records.items():
        write_record(script.ceiling_record_path(out, arm, seed), record)
    return out


def _args(out, *, arms=("pixel_ae", "frozen_ssl", "random_vit"), seeds=(0, 1, 2)):
    return types.SimpleNamespace(out=Path(out), arms=list(arms), seeds=list(seeds))


def _capture_read(out, **plan):
    """`(status, stdout)` of the read phase over `out`."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        status = script.read_phase(_args(out, **plan))
    return status, buffer.getvalue()


def _write_nine(tmp_path, base, *, harness_matches, verdicts=None):
    """Nine records over three arms and three seeds, with the harness flag set from the
    argument and its numbers consistent with it. Asserts its own discrimination before
    returning: seeds other than 0, at least three distinct episode labels, and gains whose
    intervals have width."""
    records = _records_from(base, verdicts or ["PROBE_LIMITED"] * 9, harness=harness_matches)
    assert {s for _, s in records} == {0, 1, 2}
    assert len(set(base["groups"])) >= 3
    assert all(
        g["ci_high"] > g["ci_low"] for r in records.values() for g in r["gains"].values()
    )
    return _write_records(tmp_path / "nine", records)


SPLIT = ["PROBE_LIMITED"] * 3 + ["REPRESENTATION_LIMITED"] * 3 + ["OBSERVATION_LIMITED"] * 3


def test_the_read_fixtures_reach_the_verdicts_they_are_named_for(record):
    for verdict, gains in GAINS_FOR.items():
        cell = ceiling.CeilingCell(
            arm="pixel_ae", seed=0, probe_gain=gains["probe"], pixel_gain=gains["pixel"],
            sequence_gain=gains["sequence"],
        )
        assert cell.verdict() == verdict
    assert len({tuple(g["probe"].values()) for g in GAINS_FOR.values()}) == 3


def test_the_read_returns_49_when_the_harness_does_not_reproduce_the_study(
    tmp_path, record, capsys,
):
    """Exit 49 says M3o is measuring something other than what the study measured, so no
    verdict about the model is available: it is raised from the controls before any gain is
    tallied, and the text names the cell and both numbers.

    THE MUTATION THIS EXISTS FOR, run against it: skipping the `harness_r2_matches_study`
    check."""
    out = _write_nine(tmp_path, record, harness_matches=False)
    status, text = _capture_read(out)
    assert status == script.EXIT_HARNESS_DISAGREES == 49
    assert "HARNESS DISAGREES" in text and "no verdict about the model is available" in text
    assert "pixel_ae seed 0" in text and "random_vit seed 2" in text
    assert f"{record['shipped_grid_r2']:.4f}" in text
    assert "verdict:" not in text
    assert not (out / "ceiling.txt").exists()


def test_one_cell_whose_harness_disagrees_is_enough_for_49_and_is_the_only_one_named(
    tmp_path, record,
):
    records = _records_from(record, ["PROBE_LIMITED"] * 9)
    records[("random_vit", 2)] = _record_for(record, ("random_vit", 2), "PROBE_LIMITED", harness=False)
    out = _write_records(tmp_path / "one", records)
    status, text = _capture_read(out)
    assert status == 49
    assert "random_vit seed 2" in text
    assert "pixel_ae seed 0" not in text and "frozen_ssl" not in text


@pytest.mark.parametrize("flag", [None, 1, "true"], ids=["null", "one", "string"])
def test_a_harness_flag_that_is_not_exactly_true_is_a_disagreement(tmp_path, record, flag):
    """`null` (what a sanitised NaN would leave), `1` and `"true"` are not a match, though the
    last two are truthy. The check is `is True`. THE MUTATION THIS EXISTS FOR: `if not flag:`."""
    records = _records_from(record, ["PROBE_LIMITED"] * 9)
    for r in records.values():
        r["controls"]["harness_r2_matches_study"] = flag
    status, _ = _capture_read(_write_records(tmp_path / "flag", records))
    assert status == 49


@pytest.mark.parametrize("flag, offset", [(True, 1.0), (False, 0.0)], ids=["true but far", "false but equal"])
def test_a_harness_flag_that_contradicts_the_numbers_beside_it_is_refused_by_name(
    tmp_path, record, flag, offset,
):
    """The flag, `shipped_grid_r2`, `study_embedding_r2` and `harness_r2_tolerance` are one
    measurement. A record whose flag says one thing and whose numbers say the other is not one
    this script wrote, and is refused rather than believed either way."""
    records = _records_from(record, ["PROBE_LIMITED"] * 9)
    bad = records[("frozen_ssl", 1)]
    bad["controls"]["harness_r2_matches_study"] = flag
    bad["controls"]["study_embedding_r2"] = bad["shipped_grid_r2"] + offset
    out = _write_records(tmp_path / "contradiction", records)
    with pytest.raises(SystemExit, match=r"frozen_ssl seed 1: harness_r2_matches_study is"):
        _capture_read(out)


def test_the_read_returns_50_when_no_verdict_holds_a_majority(tmp_path, record):
    """Three of each of three verdicts: none reaches the bar of five, which `strict_majority`
    computes from the nine cells present. 50 is a FINDING, so the table is printed AND written."""
    out = _write_nine(tmp_path, record, harness_matches=True, verdicts=SPLIT)
    status, text = _capture_read(out)
    assert status == script.EXIT_NO_MAJORITY == 50
    assert "verdict: NO_MAJORITY" in text
    assert "no verdict reached 5 of 9 cells" in text
    assert (out / "ceiling.txt").read_text() == text


@pytest.mark.parametrize("verdict", list(GAINS_FOR))
def test_each_model_verdict_exits_zero_and_prints_its_rule(tmp_path, record, verdict):
    """The three verdicts are answers: they exit 0. THE RULE LINE IS PRINTED, not only the
    verdict, because it is the line that says how many cells held it and against which bar."""
    out = _write_nine(tmp_path, record, harness_matches=True, verdicts=[verdict] * 9)
    status, text = _capture_read(out)
    assert status == 0
    assert f"verdict: {verdict}" in text
    assert f"rule:    {verdict} in 9 of 9 cells (strict majority 5, computed from the cells present" in text
    assert (out / "ceiling.txt").read_text() == text


def test_the_bar_is_computed_from_the_cells_present_and_not_stored(tmp_path, record):
    """Six cells (two arms): a strict majority of six is 4, not 5. THE MUTATION THIS EXISTS
    FOR: a stored bar of 5."""
    cells = [c for c in NINE_CELLS if c[0] != "random_vit"]
    verdicts = dict(zip(cells, ["PROBE_LIMITED"] * 4 + ["OBSERVATION_LIMITED"] * 2))
    out = _write_records(tmp_path / "six", _records_from(record, verdicts))
    status, text = _capture_read(out, arms=("pixel_ae", "frozen_ssl"))
    assert status == 0 and "PROBE_LIMITED in 4 of 6 cells (strict majority 4" in text


def test_a_probe_gain_that_lies_below_the_null_floor_does_not_read_probe_limited(
    ring_world, tmp_path, monkeypatch,
):
    """The same nine cells, measured twice on a world where the RBF beats the linear readout
    by a resolvable margin: with the floor at 0 every cell reads PROBE_LIMITED; with the floor
    above that margin the interval of `D - C - floor` lies below zero and none does. The read
    phase sees only the record, so this is the end-to-end statement that the floor was
    SUBTRACTED before the interval was written -- not applied after, and not against zero.

    THE MUTATION THIS EXISTS FOR: `difference_interval(d, c, ...)` for
    `probe_gain_over_floor(d, c, floor, ...)` in `measure_cell`."""
    verdicts = {}
    for floor in (0.0, 0.9):
        monkeypatch.setattr(ceiling, "null_floor", lambda n, d, _f=floor, **k: _f)
        base = _cell(ring_world, ring_world.shared)
        assert base["gains"]["probe"]["point"] > 0.0 or floor > 0.0
        records = {}
        for cell in NINE_CELLS:
            r = copy.deepcopy(base)
            r["arm"], r["seed"] = cell
            r["controls"]["harness_r2_matches_study"] = True
            r["controls"]["study_embedding_r2"] = r["shipped_grid_r2"]
            r["controls"]["zero_information_recomputed"] = dict(ZERO_RECOMPUTED)
            records[cell] = r
        status, text = _capture_read(_write_records(tmp_path / f"floor_{floor}", records))
        assert status in (0, 50)
        verdicts[floor] = (text.split("verdict: ")[1].split("\n")[0], base["gains"]["probe"])
    assert verdicts[0.0][0] == "PROBE_LIMITED" and verdicts[0.0][1]["ci_low"] > 0.0
    assert verdicts[0.9][0] != "PROBE_LIMITED" and verdicts[0.9][1]["ci_high"] < 0.0


def test_the_cell_takes_its_three_gains_by_keyword_and_each_lands_in_its_own_field(
    tmp_path, record,
):
    """`CeilingCell` is keyword-only and its three gain fields are identically typed dicts, so
    each single-field exchange is a verdict flip nothing else can see. The record's three
    gains are distinct triples chosen so the verdict depends on WHICH is which: with `probe`
    unresolvable and `pixel` resolvable the cell is REPRESENTATION_LIMITED, and exchanging
    those two reads PROBE_LIMITED. Each field is also compared by VALUE.

    THE MUTATIONS THIS EXISTS FOR: `probe_gain=` and `pixel_gain=` exchanged, and `pixel_gain=`
    and `sequence_gain=` exchanged, in `ceiling_inputs`."""
    gains = GAINS_FOR["REPRESENTATION_LIMITED"]
    assert len({tuple(g.values()) for g in gains.values()}) == 3
    records = _records_from(record, {("pixel_ae", 0): "REPRESENTATION_LIMITED"})
    cell = script.ceiling_inputs(records).cells[("pixel_ae", 0)]
    assert cell.probe_gain == gains["probe"]
    assert cell.pixel_gain == gains["pixel"]
    assert cell.sequence_gain == gains["sequence"]
    assert cell.verdict() == "REPRESENTATION_LIMITED"
    swapped = ceiling.CeilingCell(
        arm="pixel_ae", seed=0, probe_gain=gains["pixel"], pixel_gain=gains["probe"],
        sequence_gain=gains["sequence"],
    )
    assert swapped.verdict() == "PROBE_LIMITED"


def test_a_cell_built_positionally_is_a_type_error():
    with pytest.raises(TypeError):
        ceiling.CeilingCell("pixel_ae", 0, _iv(0.0), _iv(0.0), _iv(0.0))


@pytest.mark.parametrize("gain", ["probe", "pixel", "sequence"])
def test_a_record_without_one_of_its_gains_is_refused_by_name(record, gain):
    records = _records_from(record, {("pixel_ae", 0): "PROBE_LIMITED"})
    del records[("pixel_ae", 0)]["gains"][gain]
    with pytest.raises(SystemExit, match=f"pixel_ae seed 0 lacks gains.{gain}"):
        script.ceiling_inputs(records)


def test_a_record_with_one_episode_is_refused_rather_than_read(record):
    records = _records_from(record, {("pixel_ae", 0): "PROBE_LIMITED"})
    records[("pixel_ae", 0)]["groups"] = [0] * len(record["groups"])
    with pytest.raises(SystemExit, match="at least two"):
        script.ceiling_inputs(records)


def test_the_reading_text_is_the_formatters_and_the_same_on_two_reads(tmp_path, record):
    """The reading is compared against text built from the SPEC path -- the formatter handed
    the cells the test constructs itself -- and not against a second call of the code under
    test: two reads of the same code strip the same bytes, so a byte-identity test against
    itself cannot catch an `rstrip()`. The text is also what `ceiling.txt` holds, byte for byte,
    and the map-unit section follows the formatter's text without altering it."""
    out = _write_nine(tmp_path, record, harness_matches=True, verdicts=SPLIT)
    expected_cells = {
        cell: ceiling.CeilingCell(
            arm=cell[0], seed=cell[1], probe_gain=GAINS_FOR[v]["probe"],
            pixel_gain=GAINS_FOR[v]["pixel"], sequence_gain=GAINS_FOR[v]["sequence"],
        )
        for cell, v in zip(NINE_CELLS, SPLIT)
    }
    inputs = ceiling.CeilingInputs(cells=expected_cells)
    expected = ceiling.format_reading_ceiling(ceiling.reading_ceiling(inputs), inputs)
    first_status, first = _capture_read(out)
    second_status, second = _capture_read(out)
    assert first == second and first_status == second_status == 50
    assert first.startswith(expected)
    assert expected.endswith("\n") and not expected.endswith("\n\n")
    assert (out / "ceiling.txt").read_bytes() == first.encode()
    assert first[len(expected):].startswith("\nmap units\n")


def test_two_arms_of_the_nine_are_a_readable_plan_and_only_they_are_pooled(tmp_path, record):
    out = _write_nine(tmp_path, record, harness_matches=True, verdicts=["PROBE_LIMITED"] * 9)
    status, text = _capture_read(out, arms=("pixel_ae", "frozen_ssl"))
    assert status == 0
    assert "PROBE_LIMITED in 6 of 6 cells" in text
    assert "random_vit" not in text


# ---------------------------------------------------------------------------
# read: the protocol every record must share, and what makes a record sound.
# ---------------------------------------------------------------------------

PATHS = {
    "git_sha": ("git_sha",),
    "torch_version": ("torch_version",),
    "device": ("device",),
    "step": ("step",),
    "frames": ("frames",),
    "frame_stride": ("frame_stride",),
    "rows_per_split.fit": ("rows_per_split", "fit"),
    "rows_per_split.select": ("rows_per_split", "select"),
    "rows_per_split.score": ("rows_per_split", "score"),
    "sequence_frames": ("sequence_frames",),
    "confidence": ("confidence",),
    "resamples": ("resamples",),
    "null_draws": ("null_draws",),
    "split_seed": ("split_seed",),
    "bootstrap_seed": ("bootstrap_seed",),
    "zero_information_median": ("zero_information_median",),
    "harness_r2_tolerance": ("harness_r2_tolerance",),
    "zero_information.mean": ("controls", "zero_information_recomputed", "mean"),
    "zero_information.median": ("controls", "zero_information_recomputed", "median"),
    "ceiling_ridges": ("ceiling_ridges",),
    "episodes.fit": ("episodes", "fit"),
    "episodes.select": ("episodes", "select"),
    "episodes.score": ("episodes", "score"),
    "groups": ("groups",),
}
"""Every protocol field and where a record keeps it, spelled out here: the table below
asserts the two agree, so a field cannot be added to the comparison without a disagreement
case, nor dropped without the required set shrinking."""


def _holder(record, path):
    for key in path[:-1]:
        record = record[key]
    return record


def _different(value):
    if isinstance(value, str):
        return value + "-other"
    if isinstance(value, list):
        return value[:-1] + [_different(value[-1])]
    return value + 1


def test_the_protocol_table_names_every_field_and_each_has_a_case():
    assert [name for name, _ in script._PROTOCOL_FIELDS] == list(PATHS)


@pytest.mark.parametrize("victim", [("frozen_ssl", 0), ("frozen_ssl", 1), ("random_vit", 2)],
                         ids=["first", "second", "last"])
@pytest.mark.parametrize("field", sorted(PATHS))
def test_one_record_that_disagrees_on_any_protocol_field_is_refused_naming_it(
    tmp_path, record, field, victim,
):
    """Cells pooled into one reading must describe the same rows and the same protocol: two
    protocols pooled as one would be a reading over a union nothing measured. EVERY record is
    held to the first's value, so a record that sorts last is as accountable as one that
    sorts second; the victim is the first, the second and the last, so a comparison that only
    looked at one neighbour is seen."""
    records = _records_from(record, ["PROBE_LIMITED"] * 9)
    path = PATHS[field]
    holder = _holder(records[victim], path)
    holder[path[-1]] = _different(holder[path[-1]])
    out = _write_records(tmp_path / "disagree", records)
    with pytest.raises(SystemExit) as caught:
        _capture_read(out)
    message = str(caught.value)
    assert f"disagree on {field}:" in message, message
    assert f"{victim[0]} seed {victim[1]}" in message
    assert "cannot be read as one" in message
    assert not (out / "ceiling.txt").exists()


@pytest.mark.parametrize("field", sorted(PATHS))
def test_a_record_without_a_protocol_field_is_refused_by_name_not_a_key_error(
    tmp_path, record, field,
):
    records = _records_from(record, ["PROBE_LIMITED"] * 9)
    path = PATHS[field]
    del _holder(records[("random_vit", 1)], path)[path[-1]]
    out = _write_records(tmp_path / "lacking", records)
    with pytest.raises(SystemExit, match=f"random_vit seed 1 lacks {field}:"):
        _capture_read(out)


def test_long_lists_that_differ_are_told_apart_by_where_not_by_a_truncation(tmp_path, record):
    records = _records_from(record, ["PROBE_LIMITED"] * 9)
    records[("pixel_ae", 1)]["groups"][-3] += 1
    with pytest.raises(SystemExit, match=r"first differing at index \d+: "):
        _capture_read(_write_records(tmp_path / "long", records))
    records = _records_from(record, ["PROBE_LIMITED"] * 9)
    records[("pixel_ae", 1)]["groups"] = records[("pixel_ae", 1)]["groups"][:-1]
    with pytest.raises(SystemExit, match=r"\d+ vs \d+ rows"):
        _capture_read(_write_records(tmp_path / "short", records))


def _bare_write(path, record):
    """What a writer that skipped `write_record` would leave: `json.dumps` with its defaults,
    which writes a float infinity as the bare token `Infinity`."""
    path.write_text(json.dumps(record))


def test_a_bare_infinity_anywhere_in_a_record_is_refused_by_name(tmp_path, record):
    """`json.loads` ACCEPTS `Infinity` by default, which is how a non-finite capacity would
    round-trip unseen. The read parses each file strictly first, so the refusal names the file
    and the token. THE MUTATION THIS EXISTS FOR: dropping the `parse_constant` from
    `load_ceiling`."""
    out = _write_nine(tmp_path, record, harness_matches=True)
    path = script.ceiling_record_path(out, "frozen_ssl", 1)
    broken = json.loads(path.read_text())
    broken["readouts"]["embedding_rbf"]["capacity"] = float("inf")
    _bare_write(path, broken)
    assert "Infinity" in path.read_text()
    with pytest.raises(SystemExit, match=r"ceiling_frozen_ssl_seed1.json contains the bare token Infinity"):
        _capture_read(out)
    for token, value in (("NaN", float("nan")), ("-Infinity", float("-inf"))):
        broken["readouts"]["embedding_rbf"]["capacity"] = value
        _bare_write(path, broken)
        with pytest.raises(SystemExit, match=f"bare token {token}"):
            _capture_read(out)


@pytest.mark.parametrize("name", sorted(script.READOUTS))
@pytest.mark.parametrize("bad", ["nan", "Infinity", "INF", None, True, [1.0]],
                         ids=["nan string", "Infinity string", "upper INF", "null", "bool", "list"])
def test_a_capacity_that_is_not_a_number_or_the_string_inf_is_refused_for_every_readout(
    tmp_path, record, name, bad,
):
    """A record that carries a bare token in the file is refused above; this is the other half,
    a value that is legal JSON and is not a capacity. `null` is what `write_record` leaves of a
    non-finite float (filed under `nonfinite`), so it is refused whichever way it arrives."""
    out = _write_nine(tmp_path, record, harness_matches=True)
    path = script.ceiling_record_path(out, "pixel_ae", 2)
    broken = load_record(path)
    broken.pop("nonfinite")
    broken["readouts"][name]["capacity"] = bad
    write_record(path, broken)
    with pytest.raises(SystemExit, match=rf"pixel_ae seed 2: readouts\.{name}\.capacity is"):
        _capture_read(out)


def test_the_string_inf_and_every_finite_capacity_are_accepted(tmp_path, record):
    records = _records_from(record, ["PROBE_LIMITED"] * 9)
    for r in records.values():
        r["readouts"]["embedding_rbf"]["capacity"] = "inf"
        r["readouts"]["pixels_rbf"]["capacity"] = 0.25
    records = {c: dict(r) for c, r in records.items()}
    out = _write_records(tmp_path / "inf", records)
    status, text = _capture_read(out)
    assert status == 0 and "inf" in text


@pytest.mark.parametrize("key", ["r2", "mean_error", "median_error"])
@pytest.mark.parametrize("bad", ["nan", None, "0.5"])
def test_a_readout_figure_that_is_not_a_finite_number_is_refused(tmp_path, record, key, bad):
    out = _write_nine(tmp_path, record, harness_matches=True)
    path = script.ceiling_record_path(out, "pixel_ae", 0)
    broken = load_record(path)
    broken.pop("nonfinite")
    broken["readouts"]["embedding_linear"][key] = bad
    write_record(path, broken)
    with pytest.raises(SystemExit, match=rf"readouts\.embedding_linear\.{key} is"):
        _capture_read(out)


@pytest.mark.parametrize("bad", ["inf", "nan", True])
def test_a_jitter_is_null_or_a_finite_number(tmp_path, record, bad):
    out = _write_nine(tmp_path, record, harness_matches=True)
    path = script.ceiling_record_path(out, "pixel_ae", 0)
    broken = load_record(path)
    broken.pop("nonfinite")
    broken["readouts"]["embedding_rbf"]["jitter"] = bad
    write_record(path, broken)
    with pytest.raises(SystemExit, match=r"embedding_rbf\.jitter is"):
        _capture_read(out)


def test_a_readout_missing_a_field_or_the_whole_readout_is_refused_by_name(tmp_path, record):
    out = _write_nine(tmp_path, record, harness_matches=True)
    path = script.ceiling_record_path(out, "pixel_ae", 0)
    broken = load_record(path)
    broken.pop("nonfinite")
    del broken["readouts"]["embedding_knn"]["jitter"]
    write_record(path, broken)
    with pytest.raises(SystemExit, match=r"readouts\.embedding_knn lacks jitter"):
        _capture_read(out)
    del broken["readouts"]["embedding_knn"]
    write_record(path, broken)
    with pytest.raises(SystemExit, match=r"lacks readouts\.embedding_knn"):
        _capture_read(out)


def test_a_record_carrying_a_non_finite_value_anywhere_is_refused_for_the_one_it_is(
    tmp_path, record,
):
    """This script writes `nonfinite == {}`: it turns the one legitimate infinity into
    `"inf"`. A record whose sanitiser map is not empty has a NaN somewhere the measure should
    have refused, and is refused whichever field it is."""
    out = _write_nine(tmp_path, record, harness_matches=True)
    path = script.ceiling_record_path(out, "pixel_ae", 0)
    broken = load_record(path)
    broken.pop("nonfinite")
    broken["applied"]["r2"] = float("nan")
    write_record(path, broken)
    assert load_record(path)["nonfinite"] == {"applied.r2": "nan"}
    with pytest.raises(SystemExit, match=r"carries non-finite value\(s\) at applied.r2"):
        _capture_read(out)


@pytest.mark.parametrize("name", ["pixels_linear", "pixels_rbf"])
@pytest.mark.parametrize("key", ["r2", "mean_error", "median_error", "capacity", "jitter"])
def test_raw_pixel_readouts_that_differ_between_cells_are_refused_naming_the_cell(
    tmp_path, record, name, key,
):
    """They use no model: one `Readout` is written into all nine records, so a difference in
    any of its five fields means a model leaked into a model-free measurement. Held to the
    first record's value for every cell, so a difference at the LAST is seen too."""
    out = _write_nine(tmp_path, record, harness_matches=True)
    path = script.ceiling_record_path(out, "random_vit", 2)
    broken = load_record(path)
    broken.pop("nonfinite")
    entry = broken["readouts"][name]
    entry[key] = 0.5 if entry[key] is None else (entry[key] + 1.0 if not isinstance(entry[key], str) else 1.0)
    write_record(path, broken)
    with pytest.raises(SystemExit, match=rf"random_vit seed 2 and frozen_ssl seed 0 disagree on readouts\.{name}"):
        _capture_read(out)


def test_the_embedding_readouts_may_differ_between_cells(tmp_path, record):
    """The control is for the model-free row only: every other readout is a property of its
    cell, and nine records that agree on all of them would be the failure."""
    records = _records_from(record, ["PROBE_LIMITED"] * 9)
    for index, r in enumerate(records.values()):
        r["readouts"]["embedding_rbf"]["r2"] += 0.01 * index
        r["readouts"]["embedding_knn"]["capacity"] = 4.0 * (index + 1)
    status, _ = _capture_read(_write_records(tmp_path / "varied", records))
    assert status == 0


def test_a_record_filed_under_another_cells_name_is_refused(tmp_path, record):
    out = _write_nine(tmp_path, record, harness_matches=True)
    other = script.ceiling_record_path(out, "pixel_ae", 0).read_text()
    script.ceiling_record_path(out, "pixel_ae", 1).write_text(other)
    with pytest.raises(SystemExit, match=r"ceiling_pixel_ae_seed1.json was read for pixel_ae seed 1 but its record says arm='pixel_ae' seed=0"):
        _capture_read(out)


def test_read_phase_names_the_first_missing_cell_and_exits_eleven(tmp_path, record):
    out = _write_nine(tmp_path, record, harness_matches=True)
    script.ceiling_record_path(out, "frozen_ssl", 1).unlink()
    status, text = _capture_read(out)
    assert status == script.EXIT_NO_CHECKPOINTS == 11
    assert "NO CELL: frozen_ssl seed 1: no ceiling record at" in text
    assert "verdict:" not in text


def test_read_phase_refuses_a_plan_narrower_than_the_bar_before_loading_anything(tmp_path):
    with pytest.raises(SystemExit, match=f"ARMS_REQUIRED={ceiling.ARMS_REQUIRED}"):
        _capture_read(tmp_path / "nothing_here", arms=("pixel_ae",))
    assert not (tmp_path / "nothing_here").exists()


def test_an_arm_short_of_the_minimum_seeds_raises_and_is_never_caught_into_an_exit(
    tmp_path, record,
):
    """`reading_ceiling`'s ValueError for an arm short of `SEEDS_MINIMUM` seeds is a malformed
    plan, not something found in the data; a `try` around it would turn that into an exit
    number. `--phase read --seeds 0 1` reaches it."""
    out = _write_nine(tmp_path, record, harness_matches=True)
    with pytest.raises(ValueError, match=f"SEEDS_MINIMUM={ceiling.SEEDS_MINIMUM}"):
        _capture_read(out, seeds=(0, 1))
    assert not (out / "ceiling.txt").exists()


def _stale_reading(out):
    (out / "ceiling.txt").write_text("a reading an earlier run left\n")


def _refusal_protocol(out):
    path = script.ceiling_record_path(out, "random_vit", 2)
    broken = load_record(path)
    broken.pop("nonfinite")
    broken["step"] += 1
    write_record(path, broken)


def _refusal_bare_infinity(out):
    path = script.ceiling_record_path(out, "random_vit", 2)
    broken = json.loads(path.read_text())
    broken["readouts"]["embedding_rbf"]["capacity"] = float("inf")
    _bare_write(path, broken)


def _refusal_missing(out):
    script.ceiling_record_path(out, "random_vit", 2).unlink()


def _refusal_pixels(out):
    path = script.ceiling_record_path(out, "random_vit", 2)
    broken = load_record(path)
    broken.pop("nonfinite")
    broken["readouts"]["pixels_rbf"]["r2"] += 1.0
    write_record(path, broken)


REFUSALS = {
    "protocol": (_refusal_protocol, SystemExit),
    "bare infinity": (_refusal_bare_infinity, SystemExit),
    "pixels differ": (_refusal_pixels, SystemExit),
    "missing cell": (_refusal_missing, None),
}


@pytest.mark.parametrize("refusal", sorted(REFUSALS))
def test_a_refusal_removes_the_ceiling_txt_an_earlier_read_left_and_nothing_else(
    tmp_path, record, refusal,
):
    """A reading beside a refusal would let a later reader mistake a stale reading for a
    current one. The file is removed FIRST, before any refusal can be reached, and only
    `ceiling.txt` is touched: the records stay."""
    corrupt, raised = REFUSALS[refusal]
    out = _write_nine(tmp_path, record, harness_matches=True)
    corrupt(out)
    _stale_reading(out)
    before = sorted(p.name for p in out.iterdir() if p.name != "ceiling.txt")
    if raised is None:
        assert _capture_read(out)[0] == 11
    else:
        with pytest.raises(raised):
            _capture_read(out)
    assert not (out / "ceiling.txt").exists()
    assert sorted(p.name for p in out.iterdir()) == before


def test_exit_49_also_removes_a_stale_reading_and_writes_none(tmp_path, record):
    out = _write_nine(tmp_path, record, harness_matches=False)
    _stale_reading(out)
    assert _capture_read(out)[0] == 49
    assert not (out / "ceiling.txt").exists()


# ---------------------------------------------------------------------------
# read: map units, compared here and nowhere else.
# ---------------------------------------------------------------------------


def _unit_reading(tmp_path, record, **entries):
    records = _records_from(record, ["PROBE_LIMITED"] * 9)
    for r in records.values():
        r["readouts"]["embedding_rbf"].update(entries)
    return _capture_read(_write_records(tmp_path / "units", records))[1]


def _readout_line(text, name, *, cell="pixel_ae_seed0"):
    block = text.split(f"  {cell}\n")[1].split("\n\n")[0]
    (line,) = [l for l in block.splitlines() if l.startswith(f"    {name} ")]
    return line


def test_each_error_is_read_against_its_own_statistics_baseline_and_says_which(tmp_path, record):
    """A mean error of 145.68 is HALF of the zero-information MEAN (291.36), and a median error
    of 137.51 is half of the MEDIAN (275.02): both read 50.0%. Against the wrong baseline the
    two read 53.0% (the mean over the median) and 47.2% (the median over the mean), which is
    what an exchange of the denominators prints; the median over the spec's PINNED 283.05
    reads 48.6%, and the mean over the pinned pair's 292.82 reads 49.8%.

    THE MUTATIONS THIS EXISTS FOR: `mean_error / zero_median`, `median_error / zero_mean`, and
    either divisor taken from the pinned constants."""
    text = _unit_reading(tmp_path, record, mean_error=145.68, median_error=137.51)
    line = _readout_line(text, "embedding_rbf")
    assert "145.68" in line and "137.51" in line
    mean_part, median_part = line.split("137.51")[0], line.split("137.51")[1]
    assert " 50.0%" in mean_part and " 50.0%" in median_part
    for wrong in ("53.0%", "47.2%", "48.6%", "49.8%"):
        assert wrong not in line, wrong
    assert "MEAN 291.36, MEDIAN 275.02" in text
    assert "both recomputed over every training episode" in text
    assert "(pinned)" not in text and "283.05" not in text and "292.82" not in text
    assert "mean_error against the mean, median_error against the median" in text
    assert "a mean over a median is not a ratio" in text
    header = [l for l in text.splitlines() if l.startswith("    readout ")][0]
    assert "mean_error" in header and "/ mean" in header
    assert "median_error" in header and "/ median" in header
    assert header.index("/ mean") < header.index("median_error") < header.index("/ median")


def test_the_baselines_are_the_recomputed_pair_and_neither_is_a_literal(tmp_path, record):
    """BOTH ratios divide by `controls.zero_information_recomputed`: the mean by its `mean`, the
    median by its `median`. The pinned `zero_information_median` is provenance and divides
    nothing, so it is set here to a third figure (200) that no divisor may be: a median ratio
    that reads the stored constant prints 18.0%, one that reads the literal 283.05 prints
    12.7%, one that reads the literal 275.02 prints 13.1%, and a mean over the literal 291.36
    prints 17.2%. The two ratios are made different (50% and 40%) so that an exchange shows too,
    and the same figures go through both `applied` distributions.

    THE MUTATIONS THIS EXISTS FOR: either divisor replaced by a literal, the median divisor
    read from `zero_information_median`, the two exchanged."""
    records = _records_from(record, ["PROBE_LIMITED"] * 9)
    for r in records.values():
        r["controls"]["zero_information_recomputed"] = {"mean": 100.0, "median": 90.0}
        r["zero_information_median"] = 200.0
        r["readouts"]["embedding_rbf"].update(mean_error=50.0, median_error=36.0)
        for name in ("real_latent", "predicted"):
            r["applied"][name].update(mean_error=50.0, median_error=36.0)
    text = _capture_read(_write_records(tmp_path / "moved", records))[1]
    line = _readout_line(text, "embedding_rbf")
    assert re.findall(r"\d+\.\d%", line) == ["50.0%", "40.0%"]  # 50 / 100, 36 / 90
    figures = [l for l in text.split("  pixel_ae_seed0\n")[1].split("\n\n")[0].splitlines()
               if l.startswith("      r2 ")]
    assert len(figures) == 2
    for figure in figures:
        assert "mean_error 50.00 ( 50.0% of the zero-information mean)" in figure
        assert "median_error 36.00 ( 40.0% of the zero-information median)" in figure
    assert "MEAN 100.00, MEDIAN 90.00" in text
    assert "200.00" not in text  # the stored constant is provenance: not a divisor, not printed


def test_m3ns_floor_error_is_stated_as_a_mean_and_belongs_to_the_real_latent_distribution_only(
    tmp_path, record,
):
    """131.88-259.06 is a MEAN over windows (`evaluate_rollout` averages the per-window error)
    AND the FLOOR, the probe on the posterior over the true frames: so the sentence that sets
    it beside the table says it is a mean, never a median, and belongs to the real-latent
    distribution alone. Every other line that names the figure is a `compares to:` line, and
    those say the same of their own distribution. No line sets it beside a median."""
    text = _unit_reading(tmp_path, record)
    (header,) = [l for l in text.splitlines() if l.startswith("  M3n's floor error")]
    assert "is a MEAN over windows" in header
    assert "comparable to a mean_error and never to a median_error" in header
    assert "belongs to real-latent ONLY (its mean_error_h1)" in header
    assert "predicted distribution has no prior figure" in header
    mentions = [l for l in text.splitlines() if "131.88-259.06" in l and l != header]
    assert len(mentions) == 18  # two `compares to:` lines in each of nine cells
    assert all(l.startswith("      compares to: ") for l in mentions)
    assert sum("mean_error_h1 ONLY" in l for l in mentions) == 9
    assert sum("nothing earlier" in l for l in mentions) == 9
    assert script.M3N_FLOOR_ERROR_MEAN == (131.88, 259.06)


def test_m3ns_floor_error_range_is_what_its_records_hold():
    """Pinned against the nine M3n records where they are on disk (`runs/` is a symlink to a
    directory outside every worktree, so this is skipped where it is absent): the h=1 floor
    mean spans exactly the constant's two figures."""
    directory = Path(__file__).resolve().parents[2] / "runs" / "m3n_motion"
    files = sorted(directory.glob("headroom_*.json"))
    if len(files) != 9:
        pytest.skip("runs/m3n_motion is not on disk")
    floors = [json.loads(p.read_text())["curves"]["floor_position"][0] for p in files]
    assert (round(min(floors), 2), round(max(floors), 2)) == script.M3N_FLOOR_ERROR_MEAN


REPO = Path(__file__).resolve().parents[2]


def test_the_study_number_the_harness_reads_spans_the_band_the_spec_means_on_the_real_records():
    """`_study_numbers` over the nine real M3c records (skipped where `runs/` is absent):
    `criterion_4.embedding_r2` is the band 0.3195-0.3451, and the rollout probe's
    `embedding_selection_r2` is a DIFFERENT quantity spanning 0.0180-0.3829, a factor of 21.
    Reading either field for the other moves the figure the harness is reconciled against
    from one span to the other, which only real records can show.

    THE MUTATION THIS EXISTS FOR: `_study_numbers` reading `embedding_r2` from the rollout
    probe's field."""
    files = sorted((REPO / "runs" / "m3_study_v2").glob("result_*_seed*.json"))
    if len(files) != 9:
        pytest.skip("runs/m3_study_v2 is not on disk")
    numbers = [script._study_numbers("arm", 0, json.loads(p.read_text())) for p in files]
    gated = [n["embedding_r2"] for n in numbers]
    rollout = [n["embedding_selection_r2"] for n in numbers]
    assert (round(min(gated), 4), round(max(gated), 4)) == (0.3195, 0.3451)
    assert (round(min(rollout), 4), round(max(rollout), 4)) == (0.0180, 0.3829)


def test_the_real_split_and_the_zero_information_figures_are_what_the_script_documents():
    """The numbers `Shared.zero_information`'s and the module's docstrings state, recomputed
    from the real episodes (skipped where `data/my_way_home` is absent): the fit, select and
    score splits hold 9,044 / 5,713 / 6,264 rows at stride 1 under the shipped budget, and the
    zero-information baseline reads, as mean / median,

      all 98 training episodes      291.36 / 275.02   (this script's definition)
      the 20 fit episodes only      295.48 / 284.78
      the leading 24 training       292.82 / 283.05   (the spec's figures, to two decimals)

    so the spec's 283.05 is NOT what the mean over all the training episodes gives, and the
    control cannot be held to 'within 1.0 of 283.05'. A change to the split or to the definition moves these and
    the docstrings with them."""
    from mbfps.data.buffer import ReplayBuffer
    from mbfps.data.split import VAL_FRACTION, episode_split

    data = REPO / "data" / "my_way_home"
    if not data.exists():
        pytest.skip("data/my_way_home is not on disk")
    train, val = episode_split(
        ReplayBuffer(data, capacity_transitions=10**9).episode_paths(),
        val_fraction=VAL_FRACTION, seed=SPLIT_SEED,
    )
    assert (len(train), len(val)) == (98, 24)
    stride, splits = script.build_splits(train, val, budget=ceiling.FRAME_BUDGET)
    assert stride == 1
    assert {name: split.n for name, split in splits.items()} == {
        "fit": 9044, "select": 5713, "score": 6264,
    }
    every = np.concatenate([script._all_targets(p) for p in val])
    assert len(every) == 12073
    for episodes, expected in (
        (train, (291.36, 275.02)),
        (train[:20], (295.48, 284.78)),
        (train[:24], (292.82, 283.05)),
    ):
        got = ceiling.zero_information_error(
            np.concatenate([script._all_targets(p) for p in episodes]), every,
        )
        assert (round(got["mean"], 2), round(got["median"], 2)) == expected, len(episodes)


def test_the_section_prints_the_rows_the_applied_figures_and_the_floor_of_every_cell(
    tmp_path, record,
):
    """Both distributions of every cell, each under its own name and with ITS OWN figures:
    the eighteen entries are all different numbers, so an entry printed from the other
    distribution, or from another cell, is a different line."""
    records = _records_from(record, ["PROBE_LIMITED"] * 9)
    for index, (cell, r) in enumerate(records.items()):
        for k, name in enumerate(("real_latent", "predicted")):
            base = 100.0 * (k + 1) + index
            r["applied"][name].update(
                r2=0.1 * index + 0.01 * k, mean_error=base, mean_error_h1=base + 0.5,
                median_error=base - 7.0,
            )
        r["null_floor"] = 0.001 * index
    text = _capture_read(_write_records(tmp_path / "section", records))[1]
    rows = record["rows_per_split"]
    assert (
        f"Rows per split: fit {rows['fit']} (training episodes), select {rows['select']}, "
        f"score {rows['score']} (validation halves); frame stride {record['frame_stride']} "
        f"under a budget of {record['frames']}."
    ) in text
    for index, (arm, seed) in enumerate(NINE_CELLS):
        block = text.split(f"  {arm}_seed{seed}\n")[1].split("\n\n")[0]
        lines = block.splitlines()
        for k, (name, label) in enumerate((("real_latent", "real-latent"), ("predicted", "predicted"))):
            base = 100.0 * (k + 1) + index
            (at,) = [i for i, l in enumerate(lines) if l.startswith(f"    applied, {label} (")]
            figures, comparison = lines[at + 1], lines[at + 2]
            assert f"r2 {0.1 * index + 0.01 * k:+.4f};" in figures
            assert f"mean_error {base:.2f} (" in figures
            assert f"mean_error_h1 {base + 0.5:.2f};" in figures
            assert f"median_error {base - 7.0:.2f} (" in figures
            assert comparison == f"      compares to: {record['applied'][name]['compares_to']}"
        assert lines.index(next(l for l in lines if l.startswith("    applied, real-latent"))) < \
            lines.index(next(l for l in lines if l.startswith("    applied, predicted")))
        assert f"null floor {0.001 * index:+.4f} at n={record['null_floor_shape']['n']}" in block
    assert text.count("raw-pixel readouts (no model; identical in every cell)") == 1
    assert text.count("    pixels_rbf ") == 1


def test_each_applied_figure_is_read_against_its_own_zero_information_statistic(
    tmp_path, record,
):
    """A mean error of 145.68 is half of the zero-information MEAN (291.36) and a median of
    137.51 half of the MEDIAN (275.02), in BOTH distributions: `mean_error_h1` carries no
    percentage because it is compared with M3n's figure, not with a baseline. Exchanged
    denominators print 53.0% and 47.2%, and the median over the pinned 283.05 prints 48.6%."""
    records = _records_from(record, ["PROBE_LIMITED"] * 9)
    for r in records.values():
        for name in ("real_latent", "predicted"):
            r["applied"][name].update(mean_error=145.68, median_error=137.51)
    text = _capture_read(_write_records(tmp_path / "applied_units", records))[1]
    block = text.split("  pixel_ae_seed0\n")[1].split("\n\n")[0]
    figures = [l for l in block.splitlines() if l.startswith("      r2 ")]
    assert len(figures) == 2
    for line in figures:
        mean_part, median_part = line.split("median_error")
        assert "mean_error 145.68 ( 50.0% of the zero-information mean)" in mean_part
        assert "137.51 ( 50.0% of the zero-information median)" in median_part
        for wrong in ("53.0%", "47.2%", "48.6%", "49.8%"):
            assert wrong not in line, wrong


def test_a_pixel_row_is_printed_once_and_a_model_row_once_per_cell(tmp_path, record):
    text = _unit_reading(tmp_path, record)
    assert text.count("\n    pixels_linear ") == 1 and text.count("\n    pixels_rbf ") == 1
    for name in ("embedding_linear", "embedding_rbf", "embedding_knn", "sequence_linear",
                 "sequence_rbf", "embedding_linear_shipped"):
        assert text.count(f"\n    {name} ") == 9, name


@pytest.mark.parametrize("name", ["real_latent", "predicted"])
@pytest.mark.parametrize("key", ["r2", "mean_error", "mean_error_h1", "median_error"])
@pytest.mark.parametrize("bad", ["nan", None])
def test_an_applied_figure_that_is_not_a_finite_number_is_refused_by_name(
    tmp_path, record, name, key, bad,
):
    """`applied` is printed beside the table and read by nobody else, so a missing or broken one
    would surface as a KeyError or a printed `nan` in the section: refused by name instead, for
    each figure of each of the two distributions."""
    out = _write_nine(tmp_path, record, harness_matches=True)
    path = script.ceiling_record_path(out, "pixel_ae", 0)
    broken = load_record(path)
    broken.pop("nonfinite")
    broken["applied"][name][key] = bad
    write_record(path, broken)
    with pytest.raises(SystemExit, match=rf"pixel_ae seed 0: applied\.{name}\.{key} is"):
        _capture_read(out)


@pytest.mark.parametrize("name", ["real_latent", "predicted"])
def test_a_record_missing_either_applied_distribution_is_refused_by_name(tmp_path, record, name):
    """An `applied` of the earlier, flat schema (one distribution's figures at the top) is
    refused too: it has neither entry."""
    out = _write_nine(tmp_path, record, harness_matches=True)
    path = script.ceiling_record_path(out, "pixel_ae", 0)
    broken = load_record(path)
    broken.pop("nonfinite")
    del broken["applied"][name]
    write_record(path, broken)
    with pytest.raises(SystemExit, match=rf"pixel_ae seed 0 lacks applied\.{name}:"):
        _capture_read(out)
    broken["applied"] = {"r2": 0.3, "mean_error": 150.0, "median_error": 140.0}
    write_record(path, broken)
    with pytest.raises(SystemExit, match=r"pixel_ae seed 0 lacks applied\.real_latent:"):
        _capture_read(out)
    del broken["applied"]
    write_record(path, broken)
    with pytest.raises(SystemExit, match=r"pixel_ae seed 0 lacks applied\.real_latent:"):
        _capture_read(out)


@pytest.mark.parametrize("field", ["null_floor", "null_floor_shape"])
def test_a_record_without_its_floor_or_the_floors_shape_is_refused_by_name(tmp_path, record, field):
    out = _write_nine(tmp_path, record, harness_matches=True)
    path = script.ceiling_record_path(out, "pixel_ae", 0)
    broken = load_record(path)
    broken.pop("nonfinite")
    del broken[field]
    write_record(path, broken)
    with pytest.raises(SystemExit, match=f"pixel_ae seed 0 lacks {field}:"):
        _capture_read(out)
    broken[field] = "nan" if field == "null_floor" else {"n": 10, "d": None}
    write_record(path, broken)
    with pytest.raises(SystemExit, match="null_floor .* is not a finite number"):
        _capture_read(out)


_BAD_BASELINES = [
    "Infinity", "-Infinity", "inf", "nan", "NaN", "291.36", "0", None, True, [291.36], 0, 0.0,
    -1.0, -291.36,
]
_BAD_BASELINE_IDS = [
    "Infinity string", "-Infinity string", "inf string", "nan string", "NaN string",
    "numeric string", "zero string", "null", "bool", "list", "zero int", "zero float",
    "negative", "negative mean",
]


@pytest.mark.parametrize("victims", ["all nine", "the last cell only"])
@pytest.mark.parametrize("statistic", ["mean", "median"])
@pytest.mark.parametrize("bad", _BAD_BASELINES, ids=_BAD_BASELINE_IDS)
def test_a_zero_information_baseline_that_is_not_a_finite_positive_number_is_refused_by_name(
    tmp_path, record, statistic, bad, victims,
):
    """Every ratio in the map-unit table is DIVIDED by one of the two recomputed baselines, so
    each is as printed a figure as the three the sound check already finite-checks. `float()`
    accepts the strings `"Infinity"` and `"nan"`, so a record whose baseline is one clears
    anything that converts first: the read exited 0 and printed `MEAN inf` and `0.0%` for every
    mean ratio in all nine cells. A zero or negative baseline makes the ratios meaningless (a
    zero raises ZeroDivisionError once the verdict is taken).

    Run both ways: the SAME bad figure in all nine records (the protocol check finds nine
    records that agree, which is how the infinity got through) and in one record only (where a
    NaN used to be found as `disagree on zero_information.mean: nan vs nan`, which says nothing
    of what is wrong). Both are refused by the sound check, naming the field and the first cell
    that carries it, and the refusal writes no reading.

    THE MUTATIONS THIS EXISTS FOR: the baselines left out of `require_sound_records`, the
    positivity half dropped, and the sound check run after the protocol check."""
    records = _records_from(record, ["PROBE_LIMITED"] * 9)
    cells = sorted(records)
    hit = cells if victims == "all nine" else [cells[-1]]
    for cell in hit:
        records[cell]["controls"]["zero_information_recomputed"][statistic] = bad
    out = _write_records(tmp_path / "baseline", records)
    with pytest.raises(SystemExit) as caught:
        _capture_read(out)
    message = str(caught.value)
    arm, seed = hit[0]
    assert message.startswith(f"{arm} seed {seed}: zero_information.{statistic} "), message
    assert f"(controls.zero_information_recomputed.{statistic}) is {bad!r}, not a finite " \
        "number above zero" in message, message
    assert "disagree" not in message
    assert not (out / "ceiling.txt").exists()


@pytest.mark.parametrize("statistic", ["mean", "median"])
@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan"), 0.0, -3.5],
                         ids=["inf", "-inf", "nan", "zero", "negative"])
def test_the_sound_check_itself_refuses_a_float_baseline_that_is_not_finite_and_positive(
    record, statistic, bad,
):
    """The float forms, which a file cannot carry to the check (`write_record` files a float
    infinity or NaN under `nonfinite` as `null`, and the read refuses that first, by path):
    handed to `require_sound_records` in memory with an empty `nonfinite`, as a record this
    script wrote would have, the check refuses it itself. The clean records pass first, so the
    refusal is the baseline's and nothing else's."""
    records = _records_from(record, ["PROBE_LIMITED"] * 9)
    for r in records.values():
        r["nonfinite"] = {}
    script.require_sound_records(records)
    records[("random_vit", 2)]["controls"]["zero_information_recomputed"][statistic] = bad
    with pytest.raises(
        SystemExit,
        match=rf"random_vit seed 2: zero_information\.{statistic} "
              rf"\(controls\.zero_information_recomputed\.{statistic}\) is .*, not a finite "
              "number above zero",
    ):
        script.require_sound_records(records)


@pytest.mark.parametrize("statistic", ["mean", "median"])
@pytest.mark.parametrize("bad", [float("inf"), float("nan")], ids=["inf", "nan"])
def test_a_non_finite_float_baseline_in_a_file_is_refused_by_the_sanitisers_map(
    tmp_path, record, statistic, bad,
):
    """What a float infinity or NaN does on the way through a file: `write_record` writes
    `null` and files the path under `nonfinite`, and the read refuses the record for it, by
    that path, before any baseline is divided by."""
    records = _records_from(record, ["PROBE_LIMITED"] * 9)
    records[("pixel_ae", 0)]["controls"]["zero_information_recomputed"][statistic] = bad
    out = _write_records(tmp_path / "float_baseline", records)
    with pytest.raises(
        SystemExit,
        match=rf"pixel_ae seed 0 carries non-finite value\(s\) at "
              rf"controls\.zero_information_recomputed\.{statistic}",
    ):
        _capture_read(out)


def test_a_read_over_no_records_still_says_so(tmp_path):
    """`require_sound_records` now runs before the protocol check, so the empty case it used to
    be spared (`items[0]` of nothing) is named by it. THE MUTATION THIS EXISTS FOR: dropping
    the guard, which is an IndexError."""
    with pytest.raises(SystemExit, match="no ceiling record to read"):
        _capture_read(tmp_path, seeds=())
    with pytest.raises(SystemExit, match="no ceiling record to read"):
        script.require_sound_records({})


def test_the_sound_check_runs_before_the_protocol_check_in_the_read(tmp_path, record):
    """A record that is unsound AND disagrees with the rest on protocol is refused as unsound:
    a NaN baseline is a bad number before it is a disagreement. (The nan cases above reach
    this through the baseline; this pins the order itself with a protocol disagreement the
    protocol check would have named first.)"""
    records = _records_from(record, ["PROBE_LIMITED"] * 9)
    victim = records[("random_vit", 2)]
    victim["step"] = victim["step"] + 1  # a protocol disagreement
    victim["fine_minus_shipped_on_score"] = "nan"  # and an unsound figure
    out = _write_records(tmp_path / "both", records)
    with pytest.raises(SystemExit, match=r"random_vit seed 2: fine_minus_shipped_on_score is"):
        _capture_read(out)


# ---------------------------------------------------------------------------
# The controls that are recorded and never gated, and that the read prints.
# ---------------------------------------------------------------------------


def _controls_reading(tmp_path, record, edit=None, *, name="controls"):
    records = _records_from(record, ["PROBE_LIMITED"] * 9)
    for index, (cell, r) in enumerate(records.items()):
        r["fine_minus_shipped_on_score"] = 0.001 * index - 0.0123
        r["controls"]["nesting_rbf_at_widest_vs_linear"] = -0.01 * index - 0.0071
        r["controls"]["study_embedding_selection_r2"] = 0.018 + 0.03 * index
        if edit is not None:
            edit(index, r)
    return _capture_read(_write_records(tmp_path / name, records))


def test_the_controls_section_prints_each_cells_own_ungated_figures(tmp_path, record):
    """One line per cell, each carrying its OWN `C - C_shipped`, widest-rung-minus-linear, and
    the study's two numbers LABELLED (the criterion-4 one the harness is gated on, the rollout
    probe's beside it): nine different values of each, so a figure printed from another cell or
    one number printed under the other's label is a different line. Printed in cell order."""
    status, text = _controls_reading(tmp_path, record)
    assert status == 0
    section = text.split("controls (recorded; only the first is gated, by exit 49)\n")[1]
    lines = [l for l in section.splitlines() if l.startswith("  ") and "_seed" in l]
    assert len(lines) == 9
    for index, ((arm, seed), line) in enumerate(zip(sorted(NINE_CELLS), lines)):
        cell_index = list(NINE_CELLS).index((arm, seed))
        r = _records_from(record, ["PROBE_LIMITED"] * 9)[(arm, seed)]
        assert line.startswith(f"  {arm}_seed{seed}: C_shipped {r['shipped_grid_r2']:.3f} vs study ")
        assert f"vs study {r['controls']['study_embedding_r2']:.3f} (study rollout-probe " in line
        assert f"(study rollout-probe {0.018 + 0.03 * cell_index:.3f})" in line
        assert f"C - C_shipped {0.001 * cell_index - 0.0123:+.4f}" in line
        assert f"widest RBF - linear {-0.01 * cell_index - 0.0071:+.4f}" in line
    assert "expected negative on a good cell" in section


def test_a_large_negative_fine_minus_shipped_is_recorded_and_printed_and_never_gates(
    tmp_path, record,
):
    """Recorded, not gated: a finer grid that overfits selection (`C - C_shipped` of -0.16 on
    every cell, as the 8-episode fit produced on `random_vit`) is a FINDING, so the read still
    takes its reading and exits 0, and the number is on the page rather than only in the
    records."""
    status, text = _controls_reading(
        tmp_path, record, lambda i, r: r.__setitem__("fine_minus_shipped_on_score", -0.16),
    )
    assert status == 0
    assert text.count("C - C_shipped -0.1600") == 9
    assert "PROBE_LIMITED" in text


def test_the_widest_rung_minus_linear_never_gates_whatever_it_reads(tmp_path, record):
    """The asymptotic nesting claim is EXPECTED negative on a good cell (nesting is structural,
    pinned in `test_ceiling.py`), so no value of it refuses a read: a large negative one and a
    large positive one both read, at exit 0, and are printed."""
    for value in (-0.5, 0.5):
        status, text = _controls_reading(
            tmp_path, record, name=f"nest{value}",
            edit=lambda i, r, v=value: r["controls"].__setitem__(
                "nesting_rbf_at_widest_vs_linear", v,
            ),
        )
        assert status == 0
        assert text.count(f"widest RBF - linear {value:+.4f}") == 9


@pytest.mark.parametrize("field", [
    "fine_minus_shipped_on_score", "nesting_rbf_at_widest_vs_linear",
    "study_embedding_selection_r2",
])
@pytest.mark.parametrize("bad", ["nan", None, "0.1"])
def test_an_ungated_control_that_is_not_a_finite_number_is_refused_by_name(
    tmp_path, record, field, bad,
):
    """They are printed, so they are read: a `nan` or a missing one would otherwise print as
    such, or raise a bare KeyError from the formatter."""
    out = _write_nine(tmp_path, record, harness_matches=True)
    path = script.ceiling_record_path(out, "pixel_ae", 0)
    broken = load_record(path)
    broken.pop("nonfinite")
    in_controls = field != "fine_minus_shipped_on_score"
    target = broken["controls"] if in_controls else broken
    target[field] = bad
    write_record(path, broken)
    label = f"controls.{field}" if in_controls else field
    with pytest.raises(SystemExit, match=rf"pixel_ae seed 0: {label} is"):
        _capture_read(out)
    del target[field]
    write_record(path, broken)
    with pytest.raises(SystemExit, match=rf"pixel_ae seed 0 lacks {label}:"):
        _capture_read(out)


def test_the_tests_the_widest_rung_docstring_cites_exist():
    """`widest_rbf_r2` says nesting is structural and pinned by test, and names the tests. A
    docstring that cites a test that was renamed away claims a pin that is gone."""
    text = (Path(__file__).with_name("test_ceiling.py")).read_text()
    for name in (
        "test_the_kernel_readout_cannot_select_below_the_linear_probe_on_its_selection_split",
        "test_the_linear_rung_is_the_linear_reference_and_not_an_approximation_of_it",
    ):
        assert f"def {name}(" in text, name
        assert name in script.widest_rbf_r2.__doc__
