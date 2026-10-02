"""M3l's script: one gather per cell, the three estimator checks recorded per
cell, and the two companions that make a high `bits_carried` readable.

EVERY FIXTURE HERE CARRIES ITS OWN PAYLOAD. This project has shipped nine
records serialised to ONE payload with the labels dropped, so an implementation
pairing one arm with another's seeds passed every test; and a fixture in which
the encoder's raw embedding and the model's PREDICTED embedding were
byte-identical, so reading the wrong one passed all 13 tests while every number
answered a different question. So: the posterior and the prior are decisively
different arrays (the prior is collapsed and reads ~0 bits), the predicted
`embedding` carries no position and is DELETED by a test rather than compared,
and every per-cell number below is a function of the cell's own `(arm, seed)`.
"""

import importlib.util
import itertools
import json
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from mbfps.eval.capacity import (
    CEILING_BITS, bits_carried, bits_interval, episode_stats, floor_bits,
    live_classes, reading_capacity, redundancy_bits, redundancy_floor,
    redundancy_ratio,
)
from mbfps.eval.probe import _mean_r2, apply_probe, fit_probe
from mbfps.eval.retention import (
    ARMS_REQUIRED, BASE_R2_FLOOR, CONFIDENCE, RESAMPLES, SEEDS_REQUIRED,
)
from mbfps.eval.study import load_record
from mbfps.models.rssm import RSSMConfig
from mbfps.utils.config import ARMS

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "latent_capacity.py"

CATS, CLASSES = RSSMConfig.z_cats, RSSMConfig.z_classes
COLUMNS = CATS * CLASSES


def _load():
    spec = importlib.util.spec_from_file_location("latent_capacity_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load()

FEW_RESAMPLES = 20
"""Bootstrap draws for the tests that run the REAL estimator and probe.

`cell_capacity` defaults to the protocol's `RESAMPLES` and `measure_cell` never
passes this, so a production run cannot depart from the pre-registered figure.
A test may pay for fewer only when nothing it asserts depends on the draws --
never one that pins an interval bound to a side of a cut by a small margin."""


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _gathered(n_windows=12, steps=5, enc=12, seed=0, *, live_cats=CATS, drivers=4):
    """A `gather_probe_data`-shaped dict, at the REAL `(z_cats, z_classes)`
    shape and in the gather's own ROW ORDER.

    The latent shape is the real one because `capacity._require_distributions`
    refuses anything else -- a flattened array cannot tell the categorical
    groups apart -- but the row count and `encoder_embedding` width are small:
    the expensive part of the measure path is the ridge solve, and
    `tests/eval/test_latent_width_script.py` already costs 75 s.

    FOUR PROPERTIES ARE DELIBERATE.

    (1) `post_probs` IS A FUNCTION OF THE SAME DRIVERS AS `encoder_embedding`,
    so `frame_share` is a real number rather than noise around zero.

    (2) `prior_probs` IS A DIFFERENT ARRAY AND IS COLLAPSED -- every row the
    same distribution -- so `bits_carried(prior)` is ~0 while the posterior's
    is several bits. Reading one for the other is then decisively wrong, not
    marginally so.

    (3) `window` AND `step` ARE THE GATHER'S OWN LABELS: windows in ascending
    contiguous runs, `step` restarting at 0 and incrementing by 1 inside each.
    `tests/eval/test_probe.py::test_gather_probe_data_labels_every_row_with_
    its_window_and_step` pins that the real gather builds them that way;
    `redundancy_floor` needs it (see `require_temporal_order`).

    (4) THE ANGLE WRAPS AND CARRIES NOISE, so the heading columns are NOT a
    near-linear function of `(window, step)` the way position is. Without the
    wrap `sin(angle)` is nearly linear in the same drivers the encoder carries,
    the 4-column `r2` rises to meet `position_r2`, and the test that the two are
    different numbers cannot fail. Without the noise the columns are degenerate
    and `_per_column_r2` reports NaN for a reason about the fixture.

    `live_cats < CATS` collapses the remaining categoricals onto one
    distribution, which is how the zero-variance columns `frame_share` must
    exclude are made.

    `embedding` -- the model's PREDICTED embedding -- is a decoy of the same
    scale with no position in it. Nothing in this script may read it, and a
    test DELETES the key rather than comparing numbers against it.
    """
    rng = np.random.default_rng(seed)
    rows = n_windows * steps
    window = np.repeat(np.arange(n_windows), steps)
    step = np.tile(np.arange(steps), n_windows)
    episode = window // 2
    pos = np.column_stack([
        window * 100.0 + step * 3.0 + rng.normal(scale=5.0, size=rows),
        window * 50.0 - step * 2.0 + rng.normal(scale=5.0, size=rows),
    ])
    radians = (
        np.deg2rad((step * 73.0 + window * 131.0) % 360.0)
        + rng.normal(scale=0.2, size=rows)
    )
    targets = np.column_stack([pos, np.sin(radians), np.cos(radians)])

    # One shared driver set, so `encoder_embedding` and `post_probs` are two
    # views of the same latent facts -- as one model's two outputs would be.
    driver = np.column_stack([
        pos / 100.0, rng.normal(size=(rows, drivers - 2)),
    ])
    mixing = np.random.default_rng(1000 + enc).normal(size=(driver.shape[1], enc))
    encoder = driver @ mixing + 1e-3 * rng.normal(size=(rows, enc))

    logits = np.zeros((rows, CATS, CLASSES))
    if live_cats:
        code = np.random.default_rng(2000).normal(
            size=(driver.shape[1], live_cats * CLASSES)
        )
        logits[:, :live_cats] = (driver @ code).reshape(rows, live_cats, CLASSES)
    if live_cats < CATS:
        logits[:, live_cats:] = np.random.default_rng(3000).normal(
            size=(CATS - live_cats, CLASSES)
        )
    post = _softmax(logits)
    prior = np.broadcast_to(
        _softmax(np.random.default_rng(4000).normal(size=(1, CATS, CLASSES))),
        (rows, CATS, CLASSES),
    ).copy()
    return {
        "latent": rng.normal(size=(rows, 8)),
        "encoder_embedding": encoder,
        "embedding": rng.normal(size=(rows, enc)) * float(encoder.std()),
        "targets": targets,
        "window": window, "step": step, "episode": episode,
        "post_probs": post.astype(np.float32),
        "prior_probs": prior.astype(np.float32),
    }


def _paths(prefix: str, count: int) -> list[Path]:
    """Episode paths as `episode_split` yields them: `Path`, not `str`.

    The record names its episodes by `p.name`, and `prepare_cell` compares the
    split by name against the study record, so a fixture of bare strings would
    let a `str(p)` regression through -- and `str(Path("data/x.npz"))` is not
    `"x.npz"`."""
    return [Path("data/my_way_home") / f"{prefix}{i}.npz" for i in range(count)]


def _splits(seeds=(0, 1, 2), **kwargs):
    """The `(fit, select, score)` triple `gather_once` returns."""
    return tuple(_gathered(seed=s, **kwargs) for s in seeds)


def _probs(**kwargs) -> np.ndarray:
    return np.asarray(_gathered(**kwargs)["post_probs"], dtype=np.float64)


# ---------------------------------------------------------------------------
# The exit codes
# ---------------------------------------------------------------------------


def test_exit_codes_are_43_and_44_and_do_not_collide():
    """38 is M3i, 39-40 M3j, 41-42 M3k. A reused number makes two different
    refusals indistinguishable to a caller reading `$?`."""
    assert script.EXIT_ESTIMATOR_BROKEN == 43
    assert script.EXIT_BASE_UNRESOLVED == 44
    assert len({script.EXIT_ESTIMATOR_BROKEN, script.EXIT_BASE_UNRESOLVED}) == 2
    assert {43, 44}.isdisjoint({0, 11, 12, 14, 30, 38, 39, 40, 41, 42})


def test_the_measure_phase_records_a_failed_check_and_carries_on(
    monkeypatch, tmp_path,
):
    """43 and 44 are the READ phase's (Task 5), raised from the pooled records.
    A measure that raised on a failed check would discard the evidence of which
    cell's estimator broke, and the nine records are the artefact -- so the
    record is still written and the phase still returns `EXIT_OK`."""
    _, measured, written = _stub_phase(monkeypatch, checks_ok=False)
    args = types.SimpleNamespace(out=tmp_path / "out", source=tmp_path, phase="measure")
    cells = [("pixel_ae", 0), ("pixel_ae", 1)]
    assert script.measure_phase(args, cells, "cpu", [], []) == script.EXIT_OK
    assert measured == cells, "a failed check must not stop the remaining cells"
    assert len(written) == 2
    assert all(
        record["capacity"]["checks"]["floor_ok"] is False for _, record in written
    ), "the failed check must be ON the record, not swallowed"


# ---------------------------------------------------------------------------
# Blocker 1: `prepare_cell` is handed the STUDY directory
# ---------------------------------------------------------------------------


def _measure_args(source="runs/m3_study_v2", out="runs/m3l_capacity", **extra):
    """The attributes `measure_cell` reads off its args. `source` is the STUDY
    directory: `prepare_cell` loads the checkpoint from the `out` of the args it
    is handed, so passing this script's own `--out` through would hunt for the
    nine M3c checkpoints among the capacity records."""
    return types.SimpleNamespace(
        source=source, out=out, device="cpu", context=None, horizon=None, **extra,
    )


def _cell(arm="pixel_ae", seed=3):
    return types.SimpleNamespace(
        arm=arm, seed=seed, record={"steps": 20000, "git_sha": "ca3e140"},
        diagnostic={"whatever": True},
    )


class _Check:
    def __init__(self, ok):
        self.ok = ok

    def failures(self):
        return ["windows do not match the diagnostic"]

    def record(self):
        return {"ok": self.ok}


def _prepared(z_cats=CATS, z_classes=CLASSES):
    return types.SimpleNamespace(
        model=types.SimpleNamespace(
            rssm=types.SimpleNamespace(
                cfg=types.SimpleNamespace(z_cats=z_cats, z_classes=z_classes)
            )
        ),
        embedding_probe=object(),
        common={"feature_backbone": "the-backbone", "device": "cpu"},
        context=5, horizon=45,
    )


def test_prepare_cell_is_handed_the_study_directory_not_the_record_directory(monkeypatch):
    """`prepare_cell` loads the checkpoint from the `out` of the args IT is
    handed (`trust_horizon.load_checkpoint_model(args.out, ...)`), and that must
    be the STUDY directory. Passing `args` through unchanged makes it hunt for
    the nine checkpoints among our own output records and refuse the first cell
    -- a real run blocker on M3j, fixed with this same shim.

    NOTHING ELSE COVERS THIS LINE: every other test either stubs `prepare_cell`
    or feeds `cell_capacity` directly, so the one line that routes the two
    directories is exercised only here. `context`, `horizon` and `device` are
    read off the same shim by `prepare_cell`'s protocol check, so they are
    pinned with it: a shim that dropped `--context` would silently ignore the
    override."""
    seen = {}

    def recording_prepare_cell(cell_args, cell, device, train, val):
        seen.update(
            out=str(cell_args.out), device=cell_args.device,
            context=cell_args.context, horizon=cell_args.horizon,
        )
        return script.EXIT_SELF_CHECK_FAILED, None

    monkeypatch.setattr(script, "prepare_cell", recording_prepare_cell)
    args = _measure_args(source="runs/m3_study_v2", out="runs/m3l_capacity")
    args.context, args.horizon = 7, 9
    status, record = script.measure_cell(args, _cell(), "cpu", [], [])

    assert status == script.EXIT_SELF_CHECK_FAILED and record is None
    assert seen["out"] == "runs/m3_study_v2", (
        f"prepare_cell was handed out={seen['out']!r}; it must be the STUDY "
        "directory (--source), never this script's record directory (--out)"
    )
    assert seen["out"] != args.out
    assert (seen["device"], seen["context"], seen["horizon"]) == ("cpu", 7, 9)


# ---------------------------------------------------------------------------
# Blocker 2: every validation episode is scored
# ---------------------------------------------------------------------------


def test_every_validation_episode_is_scored(monkeypatch):
    """`filtering_gain`'s `limit` caps the FIT split and the SCORED split at the
    same number. M3j mirrored it and silently scored 20 of 24 validation
    episodes -- 17% of the evaluation data discarded. The test meant to pin it
    could not, because `len(val)` was ALSO 20 in its fixture. So this fixture's
    two numbers MUST differ."""
    val = [f"e{i}.npz" for i in range(24)]
    assert len(val) != script.FIT_EPISODES, "the fixture cannot fail if these agree"
    seen = []

    def recorder(model, paths, backbone, device, *, context, horizon, limit, seed):
        seen.append(limit)
        return _gathered(enc=4)

    monkeypatch.setattr(script, "gather_probe_data", recorder)
    script.gather_once(_prepared(), [f"t{i}.npz" for i in range(20)], val, seed=0)
    assert seen[-1] == len(val) == 24


@pytest.mark.parametrize("n_val", [10, 24])
def test_the_scored_gather_contains_every_validation_episode(monkeypatch, n_val):
    """The same property on what the gather CONTAINS, which is what the reading
    is taken on.

    TWO FIXTURES, BECAUSE ONE CANNOT FAIL. With 10 validation episodes a cap of
    `FIT_EPISODES` (20) never binds inside `gather_probe_data` itself, so a
    `min(FIT_EPISODES, len(val))` regression reads identically to the fix; with
    24 it binds. So the recorder emulates the real function's own cap
    (`paths[:limit]`) and the assertion is on the episodes present."""
    def recorder(model, paths, backbone, device, *, context, horizon, limit, seed):
        kept = list(paths)[:limit]          # `gather_probe_data`'s own cap
        return _gathered(n_windows=2 * len(kept), steps=5, enc=4)

    monkeypatch.setattr(script, "gather_probe_data", recorder)
    train = [f"t{i}.npz" for i in range(50)]
    val = [f"v{i}.npz" for i in range(n_val)]
    _, _, score = script.gather_once(_prepared(), train, val, seed=0)
    assert np.unique(score["episode"]).size == n_val, (
        f"the scored split holds {np.unique(score['episode']).size} of {n_val} "
        "validation episodes; the reading must be taken on all of them"
    )


def test_gather_once_takes_the_filtering_gain_draws(monkeypatch):
    """The three gathers carry the exact `(paths, seed)` pairs
    `probe.filtering_gain` uses: fit on `train[:FIT_EPISODES]` at `seed`, select
    on the NEXT `SELECT_EPISODES` at `seed + 2`, score on `val` at `seed + 1`.
    A transposed seed would score a different draw from criterion 4's while
    producing entirely plausible numbers."""
    calls = []

    def recorder(model, paths, backbone, device, *, context, horizon, limit, seed):
        calls.append({"paths": list(paths), "seed": seed, "limit": limit,
                      "model": model, "backbone": backbone, "device": device,
                      "context": context, "horizon": horizon})
        return _gathered(enc=4)

    monkeypatch.setattr(script, "gather_probe_data", recorder)
    prepared = _prepared()
    train = [f"t{i}" for i in range(50)]
    val = [f"v{i}" for i in range(10)]
    script.gather_once(prepared, train, val, seed=7)
    fit, select, score = calls
    assert fit["paths"] == train[:script.FIT_EPISODES] and fit["seed"] == 7
    assert select["paths"] == train[
        script.FIT_EPISODES:script.FIT_EPISODES + script.SELECT_EPISODES
    ]
    assert select["seed"] == 9
    assert score["paths"] == val and score["seed"] == 8
    # `limit` is `len(paths)` for EVERY gather: the slices above are the cap and
    # the argument never binds, so there is not a second number that could
    # disagree with them (`gather_once` says why).
    assert fit["limit"] == len(fit["paths"]) == script.FIT_EPISODES
    assert select["limit"] == len(select["paths"]) == script.SELECT_EPISODES
    assert score["limit"] == len(val) != script.FIT_EPISODES
    assert all(c["model"] is prepared.model for c in calls)
    assert {c["backbone"] for c in calls} == {"the-backbone"}
    assert {c["device"] for c in calls} == {"cpu"}
    assert {(c["context"], c["horizon"]) for c in calls} == {(5, 45)}


def test_gather_once_has_no_select_split_when_the_pool_ends_at_the_fit_set(monkeypatch):
    calls = []

    def recorder(model, paths, backbone, device, *, context, horizon, limit, seed):
        calls.append(list(paths))
        return _gathered(enc=4)

    monkeypatch.setattr(script, "gather_probe_data", recorder)
    train = [f"t{i}" for i in range(script.FIT_EPISODES)]
    fit, select, score = script.gather_once(_prepared(), train, ["v0", "v1"], seed=0)
    assert select is None and fit is not None and score is not None
    assert calls == [train, ["v0", "v1"]]


def test_gather_once_refuses_an_empty_training_pool():
    with pytest.raises(ValueError, match="no training episodes"):
        script.gather_once(_prepared(), [], ["v0"], seed=0)


# ---------------------------------------------------------------------------
# Blocker 3: `require_one_protocol` compares the MEASURED shape
# ---------------------------------------------------------------------------


def test_require_one_protocol_compares_the_measured_shape_not_only_constants():
    """M3k's review found `projection_seed` and `rung_width` were CONSTANTS that
    cannot vary between two records of one code version, while the measured
    `h_dim` went unchecked. `z_cats`/`z_classes` are the same trap here: they
    set the ceiling every cut is a fraction of."""
    fields = {f for f, _ in script._PROTOCOL_FIELDS}
    assert {"git_sha", "torch_version", "episodes.val", "windows.episode",
            "z_cats", "z_classes"} <= fields


def _protocol_records():
    """Nine records carrying only what `require_one_protocol` reads, each with
    its OWN copies of every list and dict, so a mutation to one victim cell
    cannot alias into the rest."""
    return {
        (arm, seed): {
            "windows": {"episode": list(range(24)), "window": list(range(24))},
            "episodes": {"val": [f"e{i}.npz" for i in range(24)]},
            "context": 5, "horizon": 45, "device": "mps",
            "torch_version": "2.13.0", "git_sha": "abc123",
            "z_cats": CATS, "z_classes": CLASSES,
        }
        for arm in ARMS for seed in (0, 1, 2)
    }


# Every field `require_one_protocol` compares, mapped to a mutation that changes
# it on one victim cell. `z_cats` and `z_classes` are THE MEASURED SHAPE, read
# off the checkpoint's own `rssm.cfg`: two cells whose latents disagree on them
# are measured against different ceilings, and every cut in Reading G is a
# fraction of that ceiling.
_PROTOCOL_FIELD_MUTATIONS = {
    "windows.episode": lambda r: r["windows"].update(episode=list(range(5))),
    # Same episode labels, different window indices within them: `windows.episode`
    # alone sees no disagreement here.
    "windows.window": lambda r: r["windows"].update(window=list(range(1, 25))),
    "episodes.val": lambda r: r["episodes"].update(val=["different.npz"]),
    "context": lambda r: r.update(context=999),
    "horizon": lambda r: r.update(horizon=999),
    "device": lambda r: r.update(device="cpu"),
    "torch_version": lambda r: r.update(torch_version="1.9.0"),
    "git_sha": lambda r: r.update(git_sha="def456"),
    "z_cats": lambda r: r.update(z_cats=CATS + 1),
    "z_classes": lambda r: r.update(z_classes=CLASSES + 1),
}


_REQUIRED_PROTOCOL_FIELDS = {
    "windows.episode", "windows.window", "episodes.val", "context", "horizon",
    "device", "torch_version", "git_sha", "z_cats", "z_classes",
}


def test_the_protocol_table_names_every_field_the_check_compares():
    """The fields the check compares are `script._PROTOCOL_FIELDS`, the table
    `require_one_protocol` ITERATES -- so a field added to the comparison is a
    row there, and this fails until it has a disagreement case above. (M3k's
    version of this compared the mutation table to a literal copy of itself,
    which no edit to the function could ever move.)

    The second assertion is the other direction: a comparison REMOVED from both
    the function and the table would leave them equal, so every field named in
    `_REQUIRED_PROTOCOL_FIELDS` stays named here."""
    compared = [field for field, _ in script._PROTOCOL_FIELDS]
    assert len(compared) == len(set(compared)), f"a field is listed twice: {compared}"
    assert set(compared) == set(_PROTOCOL_FIELD_MUTATIONS), (
        f"compared but with no disagreement case: "
        f"{sorted(set(compared) - set(_PROTOCOL_FIELD_MUTATIONS))}; "
        f"has a case but is not compared: "
        f"{sorted(set(_PROTOCOL_FIELD_MUTATIONS) - set(compared))}"
    )
    assert _REQUIRED_PROTOCOL_FIELDS <= set(compared), (
        f"no longer compared: {sorted(_REQUIRED_PROTOCOL_FIELDS - set(compared))}"
    )


def test_require_one_protocol_compares_exactly_the_fields_it_publishes(monkeypatch):
    """The table test above is only as good as the function's USE of the table.
    A function keeping a private copy of the tuple would leave
    `_PROTOCOL_FIELDS` decorative -- the table test would pass while the real
    comparison went unwatched, which is how a meta-test becomes a tautology. So
    append a field to the PUBLISHED table and require the function to refuse a
    disagreement on it."""
    monkeypatch.setattr(
        script, "_PROTOCOL_FIELDS",
        script._PROTOCOL_FIELDS + (("appended", lambda r: r["appended"]),),
    )
    records = _protocol_records()
    for record in records.values():
        record["appended"] = 0
    script.require_one_protocol(records)
    records[("pixel_ae", 1)]["appended"] = 1
    with pytest.raises(SystemExit, match="disagree on appended"):
        script.require_one_protocol(records)


@pytest.mark.parametrize("field", sorted(_PROTOCOL_FIELD_MUTATIONS))
def test_require_one_protocol_refuses_a_disagreement_on_every_compared_field(field):
    records = _protocol_records()
    _PROTOCOL_FIELD_MUTATIONS[field](records[("pixel_ae", 1)])
    with pytest.raises(SystemExit) as raised:
        script.require_one_protocol(records)
    message = str(raised.value)
    assert f"disagree on {field}" in message
    assert "pixel_ae seed 1" in message


def test_require_one_protocol_accepts_nine_records_that_agree():
    """The other side: it must not refuse agreement."""
    script.require_one_protocol(_protocol_records())


def test_require_one_protocol_refuses_an_empty_pool():
    with pytest.raises(SystemExit, match="no capacity record"):
        script.require_one_protocol({})


# ---------------------------------------------------------------------------
# Blocker 4: the plan is refused BEFORE the probe
# ---------------------------------------------------------------------------


def test_the_plan_is_refused_before_the_probe_not_after():
    """`reading_capacity` raises on too few arms, mixed `seeds_total` and a
    shared `seeds_total` below `SEEDS_REQUIRED` INSIDE the reading, i.e. after
    all the gather work. `latent_width.py`'s `require_readable_plan` refuses a
    too-narrow plan up front for exactly that reason -- without it,
    `--arms frozen_ssl` does the whole measure and then crashes.

    EACH REFUSAL NAMES ITS OWN CONSTANT AND NOT THE OTHER. A single message
    mentioning both would satisfy both `match=` patterns no matter which branch
    fired, which is how this pair of assertions becomes untestable."""
    with pytest.raises(SystemExit, match="ARMS_REQUIRED") as arms_raised:
        script.require_readable_plan(["frozen_ssl"], (0, 1, 2))
    assert "SEEDS_REQUIRED" not in str(arms_raised.value)

    with pytest.raises(SystemExit, match="SEEDS_REQUIRED") as seeds_raised:
        script.require_readable_plan(list(ARMS), (0,))
    assert "ARMS_REQUIRED" not in str(seeds_raised.value)

    script.require_readable_plan(list(ARMS), (0, 1, 2))  # does not raise
    # The bar is the imported constant, never a re-spelled literal.
    script.require_readable_plan(list(ARMS)[:ARMS_REQUIRED], tuple(range(SEEDS_REQUIRED)))


def _stub_phase(monkeypatch, *, loaded=None, measured=None, written=None,
                status=None, checks_ok=True):
    """`load_cell`, `measure_cell`, `write_record` and `git_sha` replaced, so a
    `measure_phase` test costs no gather and no ridge solve. Each list records
    what reached that step, in call order, so a test can assert that NOTHING
    did."""
    loaded = [] if loaded is None else loaded
    measured = [] if measured is None else measured
    written = [] if written is None else written

    def fake_load_cell(directory, arm, seed):
        loaded.append((directory, arm, seed))
        return _cell(arm=arm, seed=seed)

    def fake_measure_cell(args, cell, device, train, val):
        measured.append((cell.arm, cell.seed))
        if status is not None:
            return status, None
        return script.EXIT_OK, _record(cell.arm, cell.seed, checks_ok=checks_ok)

    monkeypatch.setattr(script, "load_cell", fake_load_cell)
    monkeypatch.setattr(script, "measure_cell", fake_measure_cell)
    monkeypatch.setattr(
        script, "write_record", lambda path, record: written.append((path, record)),
    )
    monkeypatch.setattr(script, "git_sha", lambda: "deadbeef")
    return loaded, measured, written


def _record(arm, seed, *, checks_ok=True, base_r2=0.65):
    """A record whose every number is a function of `(arm, seed)`.

    NINE DISTINCT PAYLOADS, never one template with the labels changed: a
    fixture of nine identical payloads lets an implementation pair one arm with
    another's seeds, or collapse every arm onto seed 0, and pass regardless."""
    rank = (list(ARMS).index(arm) if arm in ARMS else 3) * 3 + int(seed)
    return {
        "arm": arm, "seed": int(seed), "step": 20000,
        "capacity": {
            "bits": 40.0 + rank, "ci_low": 39.0 + rank, "ci_high": 41.0 + rank,
            "frame_share": 0.20 + rank / 100.0,
            "frame_low": 0.10 + rank / 100.0, "frame_high": 0.30 + rank / 100.0,
            "live": COLUMNS - rank, "prior_bits": 5.0 + rank,
            "redundancy_bits": 0.10 + rank / 1000.0,
            "redundancy_floor": 0.05 + rank / 1000.0,
            "checks": {"floor_ok": checks_ok, "routes_ok": True, "bracket_ok": True},
        },
        "base_control": {"position_r2": base_r2 + rank / 100.0},
    }


@pytest.mark.parametrize("cells", [
    [("frozen_ssl", s) for s in (0, 1, 2)],                       # one arm
    [(a, 0) for a in ARMS],                                       # one seed
])
def test_measure_phase_all_refuses_a_narrow_plan_before_any_cell_work(
    monkeypatch, tmp_path, cells,
):
    """`--arms frozen_ssl --phase all` would otherwise do the whole measure and
    then crash inside `reading_capacity`. The refusal must come before a single
    cell is LOADED, let alone measured."""
    loaded, measured, written = _stub_phase(monkeypatch)
    args = types.SimpleNamespace(out=tmp_path / "out", source=tmp_path, phase="all")
    with pytest.raises(SystemExit):
        script.measure_phase(args, cells, "cpu", [], [])
    assert loaded == [] and measured == [] and written == []


def test_measure_phase_measure_only_still_takes_a_one_cell_smoke(monkeypatch, tmp_path):
    """The refusal is for a plan a READ will follow. The milestone's smoke is
    `--phase measure --arms pixel_ae --seeds 0`, one cell, and a plan check that
    fired on every measure would make that impossible."""
    _, measured, _ = _stub_phase(monkeypatch)
    args = types.SimpleNamespace(out=tmp_path / "out", source=tmp_path, phase="measure")
    assert script.measure_phase(
        args, [("pixel_ae", 0)], "cpu", [], [],
    ) == script.EXIT_OK
    assert measured == [("pixel_ae", 0)]


def test_measure_phase_all_takes_the_full_grid(monkeypatch, tmp_path):
    """The other side of the refusal: it must not fire on a readable plan."""
    _, measured, _ = _stub_phase(monkeypatch)
    args = types.SimpleNamespace(out=tmp_path / "out", source=tmp_path, phase="all")
    cells = [(a, s) for a in ARMS for s in (0, 1, 2)]
    assert script.measure_phase(args, cells, "cpu", [], []) == script.EXIT_OK
    assert sorted(measured) == sorted(cells)


_ABSENT = object()


@pytest.mark.parametrize("phase", [_ABSENT, None, "", "al", "ALL", "measures", "bogus"])
def test_measure_phase_refuses_a_phase_that_is_not_one_of_the_known_ones(
    monkeypatch, tmp_path, phase,
):
    """The plan check is made only for `--phase all`, so `args.phase` is what
    decides whether a one-arm plan is refused or run and lost to a traceback.
    Read with a default, a MISSING phase looks exactly like `measure` and the
    check is skipped without a word.

    The plan here is the full, readable 3 x 3 grid, so `require_readable_plan`
    cannot be what refuses it: the only thing left to refuse is the phase."""
    loaded, measured, written = _stub_phase(monkeypatch)
    args = types.SimpleNamespace(out=tmp_path / "out", source=tmp_path)
    if phase is not _ABSENT:
        args.phase = phase
    cells = [(a, s) for a in ARMS for s in (0, 1, 2)]
    with pytest.raises(SystemExit, match="unknown phase"):
        script.measure_phase(args, cells, "cpu", [], [])
    assert loaded == [] and measured == [] and written == []


def test_measure_phase_names_a_missing_cell_and_measures_nothing(
    monkeypatch, tmp_path, capsys,
):
    """Every cell is loaded first, so a long measure never starts on a grid that
    cannot finish. Caught with this script's OWN `CellMissing` -- the class
    `_sibling` loaded -- not another importer's copy of it."""
    _, measured, written = _stub_phase(monkeypatch)

    def fake_load_cell(directory, arm, seed):
        if seed == 1:
            raise script.CellMissing("pixel_ae seed 1: no checkpoint")
        return _cell(arm=arm, seed=seed)

    monkeypatch.setattr(script, "load_cell", fake_load_cell)
    args = types.SimpleNamespace(out=tmp_path / "out", source=tmp_path, phase="measure")
    status = script.measure_phase(
        args, [("pixel_ae", 0), ("pixel_ae", 1)], "cpu", [], [],
    )
    assert status == script.EXIT_NO_CHECKPOINTS
    assert measured == [] and written == []
    assert "pixel_ae seed 1" in capsys.readouterr().out


def test_measure_phase_stops_at_the_first_refusal_and_writes_nothing_for_it(
    monkeypatch, tmp_path,
):
    _, measured, written = _stub_phase(
        monkeypatch, status=script.EXIT_SELF_CHECK_FAILED,
    )
    args = types.SimpleNamespace(out=tmp_path / "out", source=tmp_path, phase="measure")
    status = script.measure_phase(
        args, [("pixel_ae", 0), ("pixel_ae", 1)], "cpu", [], [],
    )
    assert status == script.EXIT_SELF_CHECK_FAILED
    assert measured == [("pixel_ae", 0)] and written == []


def test_measure_phase_writes_one_record_per_cell_with_the_runs_git_sha(
    monkeypatch, tmp_path,
):
    """`measure_cell` builds the record; this function stamps the provenance and
    writes it through the shared `write_record`, so the non-finite scan and the
    `git_sha` are the same for every diagnostic in the project."""
    _, _, written = _stub_phase(monkeypatch)
    out_dir = tmp_path / "out"
    args = types.SimpleNamespace(out=out_dir, source=tmp_path, phase="measure")
    cells = [(a, s) for a in ARMS for s in (0, 1, 2)]
    assert script.measure_phase(args, cells, "cpu", [], []) == script.EXIT_OK
    assert {path for path, _ in written} == {
        script.capacity_record_path(out_dir, arm, seed) for arm, seed in cells
    }
    assert all(record["git_sha"] == "deadbeef" for _, record in written)
    # Order-independent: the record written under a path must be that cell's.
    for path, record in written:
        assert script.capacity_record_path(
            out_dir, record["arm"], record["seed"],
        ) == path
    assert out_dir.exists(), "the output directory must still be created"


def test_measure_phase_prints_a_broken_estimator_loudly_and_only_a_broken_one(
    monkeypatch, tmp_path, capsys,
):
    """A failed check is RECORDED and the run continues -- the read phase raises
    43 -- but the line must be unmissable in a long log, and must not cry wolf
    when every check held."""
    args = types.SimpleNamespace(out=tmp_path / "out", source=tmp_path, phase="measure")
    _stub_phase(monkeypatch, checks_ok=False)
    script.measure_phase(args, [("pixel_ae", 0)], "cpu", [], [])
    broken = capsys.readouterr().out
    assert "BROKEN" in broken and "floor" in broken

    _stub_phase(monkeypatch, checks_ok=True)
    script.measure_phase(args, [("pixel_ae", 0)], "cpu", [], [])
    fine = capsys.readouterr().out
    assert "BROKEN" not in fine and "checks ok" in fine


def test_capacity_record_path_is_named_by_both_the_arm_and_the_seed():
    """A colliding name silently overwrites a cell: see `study.job_record_path`
    for what that costs."""
    paths = {
        script.capacity_record_path(Path("out"), arm, seed)
        for arm in ARMS for seed in (0, 1, 2)
    }
    assert len(paths) == 9
    assert script.capacity_record_path(Path("out"), "pixel_ae", 2).name == (
        "capacity_pixel_ae_seed2.json"
    )


# ---------------------------------------------------------------------------
# estimator_checks: the two theorems, and the one that is NOT a theorem
# ---------------------------------------------------------------------------


def test_estimator_checks_pass_on_a_plausible_posterior():
    checks = script.estimator_checks(_probs())
    assert all(checks[flag] for flag in script.CHECK_FLAGS)
    assert set(checks) == {
        *script.CHECK_FLAGS, "bits", "floor", "ceiling", "argmax_marginal",
    }


def test_estimator_checks_report_the_numbers_not_only_the_booleans(monkeypatch):
    """A check that reports only a boolean cannot be audited from the record:
    the reader of `capacity.txt` must be able to see HOW far the floor drifted
    and WHERE the two ceiling routes landed.

    WHAT THIS ESTABLISHES, AND WHAT IT DOES NOT. The four estimators are each
    stubbed to a DIFFERENT sentinel, so every key is pinned to its own source:
    a transposed assignment -- `"bits"` and `"floor"` swapped, or `"ceiling"` and
    `"argmax_marginal"` swapped -- fails here. The earlier form of this test
    compared each key to the very function that fills it on the real fixture:
    one code path, which caught `bits`/`floor` swapped and could not catch
    `ceiling`/`argmax_marginal` swapped, because `ceiling_bits` and
    `argmax_marginal_bits` are two routes to ONE number and agree on real data.
    It does NOT establish that any estimator is RIGHT on real data --
    `tests/eval/test_capacity.py` owns that -- nor that a flag is computed from
    the right number: the flag tests above stub the estimators for that."""
    sentinels = {
        "bits_carried": 12.5, "floor_bits": 0.0625,
        "ceiling_bits": 7.25, "argmax_marginal_bits": 3.75,
    }
    assert len(set(sentinels.values())) == len(sentinels), "the stubs must differ"
    for name, value in sentinels.items():
        monkeypatch.setattr(script, name, lambda probs, _v=value: _v)
    checks = script.estimator_checks(_probs())
    assert (
        checks["bits"], checks["floor"], checks["ceiling"], checks["argmax_marginal"],
    ) == (12.5, 0.0625, 7.25, 3.75)
    assert all(type(checks[k]) is float for k in
               ("bits", "floor", "ceiling", "argmax_marginal"))


def test_estimator_checks_report_the_real_estimators_numbers_on_a_real_posterior():
    """The other half: on the fixture's own posterior the numbers on the record
    are the estimators' own, finite, and inside the derived ceiling's range.
    `test_estimator_checks_report_the_numbers_not_only_the_booleans` pins which
    key holds which estimator; this pins that the figures are not placeholders."""
    probs = _probs()
    checks = script.estimator_checks(probs)
    assert checks["bits"] == pytest.approx(bits_carried(probs))
    assert checks["floor"] == pytest.approx(floor_bits(probs), abs=1e-12)
    assert checks["ceiling"] == pytest.approx(checks["argmax_marginal"], abs=1e-9)
    assert 0.0 < checks["bits"] <= CEILING_BITS


def test_estimator_checks_refuse_a_floor_away_from_zero(monkeypatch):
    """`floor_bits` is zero by algebra -- `H(m) - mean_n H(m)` -- so a reading
    away from zero beyond summation drift is an arithmetic defect. Only the
    floor flag moves: the fixture clears the other two."""
    monkeypatch.setattr(script, "floor_bits", lambda probs: 1e-6)
    checks = script.estimator_checks(_probs())
    assert checks["floor_ok"] is False
    assert checks["routes_ok"] is True and checks["bracket_ok"] is True


def test_estimator_checks_accept_a_floor_inside_the_tolerance(monkeypatch):
    """1e-9, not equality. The floor drifts to ~5e-13 at the ~11,000 rows a real
    cell carries, and an M3h sweep once refused on an 8.527e-14 mismatch that
    was exactly summation order."""
    monkeypatch.setattr(script, "floor_bits", lambda probs: -5e-13)
    assert script.estimator_checks(_probs())["floor_ok"] is True


def test_estimator_checks_refuse_two_ceiling_routes_that_disagree(monkeypatch):
    """`ceiling_bits` and `argmax_marginal_bits` are two independent routes to
    one answer, and the agreement is what pins the routing: a mean over the
    wrong axis, or a substitution leaking a row's own distribution, breaks it."""
    monkeypatch.setattr(script, "argmax_marginal_bits", lambda probs: 1.0)
    checks = script.estimator_checks(_probs())
    assert checks["routes_ok"] is False
    assert checks["floor_ok"] is True and checks["bracket_ok"] is True
    assert checks["argmax_marginal"] == 1.0, "the number must be recorded too"


@pytest.mark.parametrize("bits", [-1.0, CEILING_BITS + 1.0])
def test_estimator_checks_refuse_bits_outside_the_first_theorem(monkeypatch, bits):
    """`-1e-9 <= bits <= CEILING_BITS + 1e-9`. `bits = H(marginal) - E_n H(row)
    <= H(marginal) <= z_cats * log2(z_classes)`, so both ends are theorems."""
    monkeypatch.setattr(script, "bits_carried", lambda probs: bits)
    checks = script.estimator_checks(_probs())
    assert checks["bracket_ok"] is False
    assert checks["floor_ok"] is True and checks["routes_ok"] is True


@pytest.mark.parametrize("ceiling", [-1.0, CEILING_BITS + 1.0])
def test_estimator_checks_refuse_a_ceiling_outside_the_second_theorem(
    monkeypatch, ceiling,
):
    """`0 <= ceiling <= CEILING_BITS + 1e-9`, with the SAME 1e-9 on the lower
    end: `ceiling_bits` is `bits_carried` of the one-hot substitution and drifts
    negative by the same summation order the floor does."""
    monkeypatch.setattr(script, "ceiling_bits", lambda probs: ceiling)
    monkeypatch.setattr(script, "argmax_marginal_bits", lambda probs: ceiling)
    checks = script.estimator_checks(_probs())
    assert checks["bracket_ok"] is False
    assert checks["floor_ok"] is True
    assert checks["routes_ok"] is True, "the routes still agree; only the bracket moved"


@pytest.mark.parametrize("ceiling", [-5e-13, -1e-10, CEILING_BITS + 5e-13])
def test_estimator_checks_accept_a_ceiling_inside_the_tolerance_at_both_ends(
    monkeypatch, ceiling,
):
    """Spec 2.3 writes the second theorem as a BARE `0 <= ceiling_bits`, and
    this file applies the same 1e-9 there that it applies everywhere else. The
    reason is measurable: `ceiling_bits` is `bits_carried` of the one-hot
    substitution, i.e. the same summation, and `test_capacity.py` records that
    summation drifting to ~8e-13 at the ~11,000 rows a real cell carries. A
    strict `0 <=` would therefore refuse a code whose argmax never moves -- the
    collapsed case, and the regime this milestone exists to investigate -- for a
    float artefact, and `UNRESOLVED_ESTIMATOR` outranks every reading.

    This is the test the counterexample fixture CANNOT be: that fixture's
    `ceiling_bits` reads exactly 0.0 here, so it passes under either spelling.
    `-5e-13` is the measured run-scale drift and only the tolerant form accepts
    it."""
    monkeypatch.setattr(script, "ceiling_bits", lambda probs: ceiling)
    monkeypatch.setattr(script, "argmax_marginal_bits", lambda probs: ceiling)
    checks = script.estimator_checks(_probs())
    assert checks["bracket_ok"] is True, (
        f"a ceiling of {ceiling} is inside TOLERANCE={script.TOLERANCE} of the "
        "theorem's bounds and is float summation order, not a defect"
    )


def test_estimator_checks_do_not_refuse_the_counterexample_to_the_third_inequality():
    """`bits_carried <= ceiling_bits` IS NOT A THEOREM and must not be refused
    on. This is the very fixture
    `tests/eval/test_capacity.py::test_ceiling_bits_is_not_an_upper_bound_on_
    bits_carried` carries: 500 rows whose argmax never moves while the tail
    varies read 7.52 bits against a `ceiling_bits` of 0.00.

    Refusing on it would reject valid readings precisely in the diffuse,
    low-information regime this milestone exists to investigate -- the worst
    possible place for a spurious refusal, because it is where the answer
    lives. A strict `0 <= ceiling` on the lower end would also refuse it at run
    scale, which is why that end carries the tolerance too."""
    rng = np.random.default_rng(0)
    tail = rng.dirichlet(np.ones(CLASSES - 1), size=(500, CATS)) * 0.4
    probs = np.concatenate([np.full((500, CATS, 1), 0.6), tail], axis=-1)
    assert (probs.argmax(axis=-1) == 0).all(), "the premise: the argmax never moves"

    checks = script.estimator_checks(probs)
    assert checks["bits"] > 1.0 and checks["ceiling"] == pytest.approx(0.0, abs=1e-9)
    assert checks["bits"] > checks["ceiling"], "the premise: bits exceeds the ceiling"
    assert all(checks[flag] for flag in script.CHECK_FLAGS), (
        f"estimator_checks refused a valid diffuse reading: {checks}"
    )


def test_the_check_flags_are_exactly_the_three_booleans():
    """`CHECK_FLAGS` is the table every reader iterates, so a fourth check
    cannot be added without a reader noticing, and the record cannot carry a
    boolean no reader reads."""
    checks = script.estimator_checks(_probs())
    assert script.CHECK_FLAGS == ("floor_ok", "routes_ok", "bracket_ok")
    assert all(isinstance(checks[flag], bool) for flag in script.CHECK_FLAGS)
    assert {k for k, v in checks.items() if isinstance(v, bool)} == set(script.CHECK_FLAGS)


# ---------------------------------------------------------------------------
# cell_capacity
# ---------------------------------------------------------------------------


CAPACITY_KEYS = {
    "bits", "ci_low", "ci_high", "confidence", "resamples", "n_episodes",
    "frame_share", "frame_low", "frame_high", "frame_confidence", "frame_resamples",
    "frame_ridge", "frame_ridge_selected",
    "live", "prior_bits", "redundancy_bits", "redundancy_floor", "checks", "rows",
}


def test_cell_capacity_keys_are_pinned_exactly():
    """Pinned EXACTLY, in both directions: a key Task 5 reads cannot vanish, and
    a key no reader reads cannot accumulate. `redundancy_bits` and
    `redundancy_floor` sit beside `bits` because `bits_carried` sums the
    PER-CATEGORICAL informations and so upper-bounds the joint -- 32
    categoricals all copying one 5-bit variable read the full 160 while carrying
    5 bits jointly. A reading above a cut establishes nothing about capacity in
    use unless the redundancy sits near its floor, and Task 5 needs the ratio
    beside a `CAPACITY_BOUND` verdict."""
    cell = script.cell_capacity(_splits(), seed=0, resamples=FEW_RESAMPLES)
    assert set(cell) == CAPACITY_KEYS
    assert {"redundancy_bits", "redundancy_floor", "prior_bits", "bits"} <= set(cell)


def test_cell_capacity_carries_both_redundancy_numbers_as_plain_floats():
    """They ship as two plain floats, never as a pre-formed ratio: the ratio is
    `redundancy_ratio` at read time, and it is None for a collapsed code -- a
    value JSON cannot carry and a cut cannot be applied to. And they GATE
    NOTHING: no check, status or refusal here depends on them."""
    cell = script.cell_capacity(_splits(), seed=0, resamples=FEW_RESAMPLES)
    # `type is float`, not `isinstance`: `np.float64` IS a `float` subclass, so
    # an `isinstance` check cannot tell a plain float from a numpy scalar and
    # the casts would be untested. `study._sanitise` coerces numpy on the way
    # out, which means a numpy value reaches the JSON intact and nothing in the
    # round trip notices -- so the type discipline has to be pinned here.
    assert type(cell["redundancy_bits"]) is float
    assert type(cell["redundancy_floor"]) is float
    assert np.isfinite(cell["redundancy_bits"]) and np.isfinite(cell["redundancy_floor"])
    assert "redundancy_ratio" not in cell and "red_ratio" not in cell
    # Nothing in the checks moves when the redundancy does.
    assert all(cell["checks"][flag] for flag in script.CHECK_FLAGS)


def test_every_measured_value_is_a_plain_python_scalar():
    """No numpy leaves on the record.

    `study._sanitise` coerces numpy scalars on the way out, so a `np.float32`
    reaches the JSON looking right and the round trip notices nothing -- while
    an `np.ndarray` would silently become a LIST where a reader expects a
    number. The discipline is therefore pinned at the source."""
    cell = script.cell_capacity(_splits(), seed=0, resamples=FEW_RESAMPLES)
    leaves = [(k, v) for k, v in cell.items() if k != "checks"]
    leaves += [(f"checks.{k}", v) for k, v in cell["checks"].items()]
    offenders = [(k, type(v).__name__) for k, v in leaves
                 if type(v) not in (bool, int, float, str)]
    assert not offenders, f"not plain Python scalars: {offenders}"
    assert type(cell["live"]) is int and type(cell["n_episodes"]) is int


def test_cell_capacity_reads_the_prior_from_prior_probs_not_the_posterior():
    """Spec 3.3 requires `prior_bits` per cell as a companion that decides
    nothing: it is free from the same forward pass and speaks to M3g's finding
    that the prior is the failing stage.

    The fixture's prior is COLLAPSED -- every row the same distribution -- so it
    reads ~0 bits while the posterior reads several. An implementation reading
    `post_probs` for both would report the posterior's figure twice, and an
    implementation reading `prior_probs` for `bits` would report ~0 there. Both
    are caught, in both directions."""
    cell = script.cell_capacity(_splits(), seed=0, resamples=FEW_RESAMPLES)
    assert cell["prior_bits"] == pytest.approx(0.0, abs=1e-9)
    assert cell["bits"] > 1.0
    assert cell["prior_bits"] != pytest.approx(cell["bits"], abs=1e-3)


def test_cell_capacity_measures_both_redundancy_numbers_on_the_posterior():
    """THE RATIO IS ONLY MEANINGFUL IF ITS TWO HALVES DESCRIBE ONE ARRAY.

    THE MUTATION THIS EXISTS FOR: `redundancy_bits(probs)` ->
    `redundancy_bits(np.asarray(score["prior_probs"], ...))`, with
    `redundancy_floor` on the next line still reading `probs`. It left all 452
    targeted tests green, and it would have made the printed ratio a posterior
    FLOOR divided into a PRIOR redundancy -- a quotient of two different codes.
    The write-up leans on this companion to argue the true joint information is
    well below the reported sum, so a meaningless ratio is a meaningless
    argument.

    `test_cell_capacity_reads_the_prior_from_prior_probs_not_the_posterior`
    already catches the reverse swap for `prior_bits`, by the same fixture
    asymmetry: the prior is COLLAPSED, so it has no pair of varying
    categoricals and reads ~0 against the posterior's real dependence. That
    asymmetry was simply never asserted for the redundancy pair. Measured on
    this fixture: posterior `redundancy_bits` 0.3627 and `redundancy_floor`
    0.0934, against -4e-9 for both on the prior -- a collapsed code's
    circular shift is the identity, so its redundancy equals its floor BIT FOR
    BIT and the naive ratio would be exactly 1.0.

    Compared to direct estimator calls on `post_probs`, not merely bracketed:
    the two halves must be the same two numbers the record's ratio is formed
    from."""
    fit, select, score = _splits()
    post = np.asarray(score["post_probs"], dtype=np.float64)
    prior = np.asarray(score["prior_probs"], dtype=np.float64)
    cell = script.cell_capacity((fit, select, score), seed=0, resamples=FEW_RESAMPLES)

    # The premise: the prior is collapsed, so reading it for either half is
    # decisively wrong rather than marginally so.
    assert redundancy_bits(prior) == pytest.approx(0.0, abs=1e-6)
    assert redundancy_floor(prior, seed=0) == pytest.approx(0.0, abs=1e-6)
    assert redundancy_bits(post) > 0.1 and redundancy_floor(post, seed=0) > 0.01

    assert cell["redundancy_bits"] == pytest.approx(redundancy_bits(post), abs=1e-12)
    assert cell["redundancy_floor"] == pytest.approx(
        redundancy_floor(post, seed=0), abs=1e-12,
    )
    # And each is decisively away from what the prior reads, in both directions.
    assert cell["redundancy_bits"] > 0.1, "redundancy_bits reads the collapsed prior"
    assert cell["redundancy_floor"] > 0.01, "redundancy_floor reads the collapsed prior"
    ratio = redundancy_ratio(cell["redundancy_bits"], cell["redundancy_floor"])
    assert ratio == pytest.approx(
        redundancy_bits(post) / redundancy_floor(post, seed=0), abs=1e-9,
    ), "the printed ratio is not one code's redundancy over its own floor"


def test_cell_capacity_never_reads_the_models_predicted_embedding():
    """The raw encoder output and the model's PREDICTED embedding answer
    different questions, and `frame_share` asks what the CURRENT FRAME explains
    -- so it must read `encoder_embedding`. A fixture whose two embeddings were
    byte-identical let an earlier milestone read the wrong one and pass 13
    tests, so the KEY IS DELETED rather than its numbers compared."""
    splits = _splits()
    for data in splits:
        del data["embedding"]
        del data["latent"]
    cell = script.cell_capacity(splits, seed=0, resamples=FEW_RESAMPLES)
    assert np.isfinite(cell["frame_share"])
    control = script.base_control(*splits)
    assert np.isfinite(control["position_r2"])


def test_cell_capacity_puts_its_point_estimates_inside_their_own_intervals():
    """`capacity_arm` REFUSES a point estimate outside its own interval, and it
    refuses it after the whole measure. The two intervals are built from the
    same episode statistics as their point estimates for exactly that reason."""
    cell = script.cell_capacity(_splits(), seed=0, resamples=FEW_RESAMPLES)
    assert cell["ci_low"] <= cell["bits"] <= cell["ci_high"]
    assert cell["frame_low"] <= cell["frame_share"] <= cell["frame_high"]
    assert cell["n_episodes"] == len(np.unique(_splits()[2]["episode"])) == 6


def test_cell_capacity_excludes_zero_variance_columns_and_reports_their_count():
    """`frame_share` is the mean over the 1024 flattened columns with the
    ZERO-VARIANCE ones excluded: R^2 is undefined for them, and this project has
    already refused a fixture for exactly that. Their count ships as `live`.

    Half the categoricals are collapsed onto one distribution here, so 512 of
    the 1024 columns never vary. `live` must be the count of the columns the
    mean was actually taken over -- an implementation that averaged all 1024
    would read NaN, and one that reported 1024 while averaging 512 would print a
    saturation signal that is not about the number beside it."""
    splits = _splits(live_cats=CATS // 2)
    cell = script.cell_capacity(splits, seed=0, resamples=FEW_RESAMPLES)
    expected = (CATS // 2) * CLASSES
    assert cell["live"] == expected < COLUMNS
    assert cell["live"] == live_classes(
        np.asarray(splits[2]["post_probs"], dtype=np.float64)
    )
    assert np.isfinite(cell["frame_share"])
    assert np.isfinite(cell["frame_low"]) and np.isfinite(cell["frame_high"])


def test_cell_capacity_refuses_a_live_count_that_is_not_the_averaged_columns(
    monkeypatch,
):
    """`live` is the count of the columns `frame_share` averaged, and the pure
    module's `live_classes` is the definition of "varies". The two are tied
    together so that a change to either rule cannot leave the printed
    saturation signal describing a different set of columns from the number
    beside it."""
    monkeypatch.setattr(script, "live_classes", lambda probs: 7)
    with pytest.raises(ValueError, match="live"):
        script.cell_capacity(_splits(), seed=0, resamples=FEW_RESAMPLES)


def test_cell_capacity_uses_every_validation_row_for_both_statistics():
    """ONE gather means ONE row set, so `bits` and `frame_share` describe the
    same rows. A second gather would reintroduce the row-alignment hazard M3k
    needed two layered guards for."""
    splits = _splits()
    cell = script.cell_capacity(splits, seed=0, resamples=FEW_RESAMPLES)
    assert cell["rows"] == splits[2]["post_probs"].shape[0]
    assert cell["bits"] == pytest.approx(
        bits_carried(np.asarray(splits[2]["post_probs"], dtype=np.float64))
    )


def test_the_records_two_routes_to_bits_agree():
    """`capacity.bits` comes from the episode SUFFICIENT STATISTICS the
    bootstrap resamples; `capacity.checks.bits` is `bits_carried` run straight
    on the rows. Two routes to one number, both on the record, and nothing
    would otherwise notice if they parted company -- a `_bits_from_stats` that
    normalised by the wrong total would move the interval's point estimate
    while the check's number stayed right.

    `TOLERANCE`, not equality: `EpisodeStats` is exact up to summation ORDER,
    and an M3h sweep once refused on an 8.527e-14 mismatch that was exactly
    that."""
    cell = script.cell_capacity(_splits(), seed=0, resamples=FEW_RESAMPLES)
    assert cell["bits"] == pytest.approx(cell["checks"]["bits"], abs=script.TOLERANCE)
    assert cell["checks"]["bits"] == pytest.approx(
        bits_carried(np.asarray(_splits()[2]["post_probs"], dtype=np.float64)),
    )


def test_frame_share_is_exactly_probes_own_mean_r2_on_the_live_columns():
    """The point estimate is taken from the same per-episode statistics as the
    interval, so it cannot sit outside its own bounds -- and it must still be
    the project's own number. `_r2_from_stats` at `picked = every episode once`
    collapses to `1 - SSE_full / SST_full`, which is exactly
    `probe._mean_r2`, so the two are pinned against each other here rather than
    left to agree by inspection."""
    fit, select, score = _splits()
    columns = script._varying_columns(np.asarray(score["post_probs"], dtype=np.float64))
    fitted = fit_probe(
        np.asarray(fit["encoder_embedding"], dtype=np.float64),
        script._flat_probs(fit)[:, columns],
        np.asarray(select["encoder_embedding"], dtype=np.float64),
        script._flat_probs(select)[:, columns],
    )
    predicted = apply_probe(
        fitted, np.asarray(score["encoder_embedding"], dtype=np.float64),
    )
    expected = _mean_r2(predicted, script._flat_probs(score)[:, columns])

    cell = script.cell_capacity((fit, select, score), seed=0, resamples=FEW_RESAMPLES)
    assert cell["frame_share"] == pytest.approx(expected, abs=1e-12)


def test_the_frame_interval_bounds_are_measurements_and_not_float_debris():
    """A per-draw denominator makes this interval explode, and the explosion is
    silent: `capacity_arm` accepts any finite bound.

    `post_probs` spans 32 orders of magnitude, so a softmax column sitting at
    7e-33 is `max > min` live and has a healthy full-sample variance only
    because it is large in SOME episodes. A bootstrap draw that misses those
    episodes leaves it effectively constant. MEASURED ON THIS FIXTURE, with the
    denominator recomputed per draw as `probe._block_bootstrap_ci` does, over
    this test's own draws (200, `seed=0`): the worst column-draw has SST 4.1e-32
    against an SSE of 1.9e-4 and reads R^2 -4.6e27; 77 of the 200 draws read a
    mean below -1000, the worst -4.7e24, with 790 of the 1024 columns past -1000
    in that draw. The draw-to-full SST ratios of the degenerate columns (4.9e-30
    to 1.0e-3) overlap the healthy ones' (8.2e-6 to 4.5), so no tolerance
    separates them. The figures depend on the draw sequence; the conclusion does
    not.

    So `_r2_stats` holds the denominator at the full sample's variance, and the
    bound this asserts is the one that regression would destroy. The window is
    deliberately loose -- this fixture has 6 clusters and a genuinely wide
    interval -- because the defect it catches is more than 20 orders of
    magnitude away."""
    cell = script.cell_capacity(_splits(), seed=0, resamples=200)
    assert -5.0 < cell["frame_low"] <= cell["frame_share"] <= cell["frame_high"] < 5.0
    assert cell["frame_low"] < cell["frame_high"], "a collapsed interval is not one"


def _frame_inputs():
    """The arguments `frame_probe` takes, built the way `cell_capacity` builds
    them: the scored split's own varying columns and episode labels."""
    fit, select, score = _splits()
    columns = script._varying_columns(np.asarray(score["post_probs"], dtype=np.float64))
    return fit, select, score, columns, np.asarray(score["episode"])


SEED_TEST_RESAMPLES = 50
"""Draws for the seed test -- NOT the protocol's `RESAMPLES`, and on purpose.

Measured on this fixture, `frame_low` at seeds 0 / 1 / 2:

    50 draws      -0.3083  -0.4114  -0.8526     (seeds 1 and 2 are 0.441 apart)
    1000 draws    -0.5460  -0.6109  -0.6018     (seeds 1 and 2 are 0.009 apart)

A percentile bound's sampling noise falls with the draw count, so the seed's
footprint on the interval is largest where draws are few. At the protocol's 1000
the seeds read 0.009 apart, and a test there would have to claim a margin its own
measurement does not support. 50 draws is where the effect is plainly visible,
and the call costs ~0.03 s."""

SEED_TEST_MARGIN = 0.1
"""Chosen from the measurement above: a quarter of the 0.441 that seeds 1 and 2
are measured apart at `SEED_TEST_RESAMPLES`. An implementation that ignores its
seed reads a gap of EXACTLY 0.0, so any margin up to the measured gap tells the
two apart; this one is wide enough that float debris cannot pass for a moved
interval and narrow enough to leave the measurement 4x of room."""


def test_the_frame_intervals_seed_is_used_and_not_merely_passed():
    """THE SEED IS DRAWN FROM, not just handed over.

    A STANDING HAZARD, NAMED: A TEST THAT ASSERTS A VALUE WAS PASSED RATHER THAN
    USED. Asserting a call's arguments proves the HAND-OFF, never the behaviour.
    `test_every_estimator_call_gets_the_cells_own_bootstrap_seed` wraps
    `frame_probe` and reads `kwargs["seed"]`, so a `frame_probe` whose body says
    `default_rng(0)` receives the cell's seed, satisfies the Counter, and passes
    -- which a text-mutated copy of this file did, with all 88 tests green. An
    earlier milestone shipped every cell sharing one bootstrap seed, which
    correlates the interval noise the seeds x arms agreement rule treats as
    independent; the Counter pins that no caller defaults the seed, and this
    pins that the callee does not either.

    `frame_share` IS IDENTICAL ACROSS THE TWO SEEDS, and that identity is what
    makes this a test of the INTERVAL'S seed rather than of the measurement: the
    point estimate has no randomness in it (`_r2_from_stats` at every episode
    once), so a `frame_low` that moves while `frame_share` does not can only have
    moved because the DRAWS did. Were the point estimate seed-dependent, a moving
    bound could be the measurement and the test would be asserting nothing about
    the bootstrap.

    Measured at `SEED_TEST_RESAMPLES` draws on this fixture: frame_share 0.2615
    at every seed; frame_low -0.3083 / -0.4114 / -0.8526 at seeds 0 / 1 / 2.
    Seeds 1 and 2 are 0.441 apart and `SEED_TEST_MARGIN` is a quarter of that.

    WHAT WRONG IMPLEMENTATION EACH ASSERTION CATCHES. The gap catches a constant
    or ignored seed (`default_rng(0)`): the gap is exactly 0.0. The repeat
    catches an UNSEEDED generator (`default_rng()`), which would pass the gap
    and make every record unreproducible."""
    splits = _splits()
    one = script.cell_capacity(splits, seed=1, resamples=SEED_TEST_RESAMPLES)
    two = script.cell_capacity(splits, seed=2, resamples=SEED_TEST_RESAMPLES)
    again = script.cell_capacity(splits, seed=1, resamples=SEED_TEST_RESAMPLES)

    assert one["frame_share"] == two["frame_share"], (
        "the point estimate must not depend on the seed, or a moving frame_low "
        "is not evidence about the interval's draws"
    )
    gap = abs(one["frame_low"] - two["frame_low"])
    assert gap > SEED_TEST_MARGIN, (
        f"seeds 1 and 2 put frame_low {gap:.4f} apart; this fixture measures "
        f"0.441 at {SEED_TEST_RESAMPLES} draws and the margin is "
        f"{SEED_TEST_MARGIN}. The frame interval is not drawing from its seed"
    )
    assert (again["frame_low"], again["frame_high"]) == (
        one["frame_low"], one["frame_high"],
    ), "the same seed twice must give the same interval"


def test_the_record_says_what_level_and_how_many_draws_the_frame_interval_took():
    """`confidence` on the record is `bits_interval`'s and describes the `bits`
    interval ONLY; a reader taking it to describe the frame interval too would
    be assuming the project's standard bootstrap, which the frame interval
    deliberately is not (its denominator is held at the full sample). So the
    frame interval carries its own level and draw count.

    The values are asserted, not merely the keys. Each is compared to a number
    this test did not take from the code under test: `CONFIDENCE` and
    `RESAMPLES` are the protocol's own constants, and `FEW_RESAMPLES` is chosen
    to differ from `RESAMPLES`, so an implementation that hardcodes the draw
    count (or ignores the argument) fails at the first call, and one that halves
    the confidence fails at the second.

    A RECORDED VALUE IS AN ECHO, so the bounds are ALSO compared to a direct
    `frame_probe` call at the protocol's level and the stated draws: a
    `cell_capacity` that recorded `CONFIDENCE` while handing `frame_probe`
    something else would read the right keys beside the wrong interval."""
    splits = _splits()
    cell = script.cell_capacity(splits, seed=0, resamples=FEW_RESAMPLES)
    assert type(cell["frame_confidence"]) is float
    assert cell["frame_confidence"] == CONFIDENCE
    assert type(cell["frame_resamples"]) is int
    assert cell["frame_resamples"] == FEW_RESAMPLES != RESAMPLES

    fit, select, score, columns, groups = _frame_inputs()
    direct = script.frame_probe(
        fit, select, score, columns=columns, groups=groups,
        resamples=FEW_RESAMPLES, confidence=CONFIDENCE, seed=0,
    )
    assert (cell["frame_low"], cell["frame_high"]) == (
        direct["frame_low"], direct["frame_high"],
    ), "cell_capacity's frame interval is not the one the protocol's level gives"

    # The production call: `measure_cell` never passes `resamples`.
    production = script.cell_capacity(splits, seed=0)
    assert production["frame_resamples"] == RESAMPLES
    assert production["frame_confidence"] == CONFIDENCE


def test_the_frame_interval_is_taken_at_the_confidence_it_records():
    """`frame_confidence` is only evidence if the interval was TAKEN at it --
    the same hazard as the seed: a `frame_probe` that echoed its argument while
    cutting the tails at a hardcoded 2.5% would record 0.5 beside a 95%
    interval. Same seed, same draws, so a 50% interval must sit strictly inside
    the 95% one at BOTH ends; an ignored `confidence` makes them identical."""
    fit, select, score, columns, groups = _frame_inputs()

    def at(confidence):
        return script.frame_probe(
            fit, select, score, columns=columns, groups=groups,
            resamples=FEW_RESAMPLES, confidence=confidence, seed=0,
        )

    wide, narrow = at(0.95), at(0.50)
    assert (wide["frame_confidence"], narrow["frame_confidence"]) == (0.95, 0.50)
    assert wide["frame_low"] < narrow["frame_low"] <= narrow["frame_high"] < wide["frame_high"]


def test_the_frame_interval_takes_the_number_of_draws_it_records(monkeypatch):
    """`frame_resamples` is only evidence if that many draws were taken. Counted
    as a DIFFERENCE between two runs, so the one extra evaluation that produces
    the point estimate drops out and the test does not depend on how it is
    computed: 30 draws must cost exactly 20 more bootstrap evaluations than 10.
    A loop of a hardcoded length costs 0 more."""
    fit, select, score, columns, groups = _frame_inputs()
    real, calls = script._r2_from_stats, []
    monkeypatch.setattr(
        script, "_r2_from_stats", lambda *a: calls.append(1) or real(*a),
    )

    def evaluations(resamples):
        calls.clear()
        out = script.frame_probe(
            fit, select, score, columns=columns, groups=groups,
            resamples=resamples, seed=0,
        )
        assert out["frame_resamples"] == resamples
        return len(calls)

    assert evaluations(30) - evaluations(10) == 20


def _bits_inputs():
    """The argument `bits_interval` takes, built the way `cell_capacity` builds
    it: `episode_stats` of the SCORED split's posterior over its own episode
    labels."""
    _, _, score = _splits()
    return episode_stats(
        np.asarray(score["post_probs"], dtype=np.float64),
        np.asarray(score["episode"]),
    )


def test_the_record_says_what_level_and_how_many_draws_the_bits_interval_took():
    """`ci_high` IS THE NUMBER THE VERDICT IS READ FROM -- `SPARE_CAPACITY` is
    `ci_high < SPARE_CUT` per seed -- and a narrower interval makes that status
    EASIER to clear. So the level and the draw count it was taken at are pinned
    here and recorded on the permanent artefact, exactly as c9f6a3b pinned them
    for the frame half.

    THE MUTATION THIS EXISTS FOR: `cell_capacity`'s `bits_interval(...)` call
    with `resamples=11, confidence=0.50`. It left all 484 tests green --
    `confidence` was on the record but nothing compared it, and the draw count
    was not recorded at all.

    The values are asserted, not merely the keys, and each against a number this
    test did not take from the code under test: `CONFIDENCE` and `RESAMPLES` are
    the protocol's own constants and `FEW_RESAMPLES` is chosen to differ from
    `RESAMPLES`, so an implementation that hardcodes the draw count fails at the
    first call and one that halves the confidence at the second.

    A RECORDED VALUE IS AN ECHO, so the bounds are ALSO compared to a direct
    `bits_interval` call at the protocol's level and the stated draws: a
    `cell_capacity` that recorded `CONFIDENCE` while handing `bits_interval`
    something else would read the right keys beside the wrong interval."""
    splits = _splits()
    cell = script.cell_capacity(splits, seed=0, resamples=FEW_RESAMPLES)
    assert type(cell["confidence"]) is float
    assert cell["confidence"] == CONFIDENCE
    assert type(cell["resamples"]) is int
    assert cell["resamples"] == FEW_RESAMPLES != RESAMPLES

    direct = bits_interval(
        _bits_inputs(), resamples=FEW_RESAMPLES, confidence=CONFIDENCE, seed=0,
    )
    assert (cell["ci_low"], cell["ci_high"]) == (
        direct["ci_low"], direct["ci_high"],
    ), "cell_capacity's bits interval is not the one the protocol's level gives"
    assert cell["bits"] == direct["bits"]

    # The production call: `measure_cell` never passes `resamples`.
    production = script.cell_capacity(splits, seed=0)
    assert production["resamples"] == RESAMPLES
    assert production["confidence"] == CONFIDENCE


def test_neither_bootstrap_entry_point_defaults_its_seed():
    """A defaulted seed is how M3j shipped every cell drawing from seed 0, which
    correlates the interval noise the seeds x arms agreement rule treats as
    independent. `redundancy_floor(probs, *, seed)` has required it since it was
    written; `bits_interval` and `frame_probe` defaulted it to 0, so a caller
    that forgot it got a silent collapse onto one bootstrap rather than a
    `TypeError`. Closed BY CONSTRUCTION here, in addition to the behavioural
    seed tests -- the Counter at
    `test_every_estimator_call_gets_the_cells_own_bootstrap_seed` can only pin
    the callers this script happens to have."""
    fit, select, score, columns, groups = _frame_inputs()
    with pytest.raises(TypeError, match="seed"):
        bits_interval(_bits_inputs(), resamples=FEW_RESAMPLES)
    with pytest.raises(TypeError, match="seed"):
        script.frame_probe(
            fit, select, score, columns=columns, groups=groups,
            resamples=FEW_RESAMPLES,
        )


def test_frame_probe_refuses_columns_that_do_not_vary_on_the_scored_rows():
    """`frame_probe` takes `columns` from its caller, so a caller that passed
    every column -- or the FIT split's live mask instead of the SCORED split's
    -- would divide by a zero variance. Under the suite's `-W error` that is a
    bare `RuntimeWarning` turned into an exception from inside a bootstrap
    loop, naming nothing; this makes it a refusal that says which columns and
    where the mask should have come from."""
    fit, select, score = _splits(live_cats=CATS // 2)
    everything = np.ones(COLUMNS, dtype=bool)
    with pytest.raises(ValueError, match="zero full-sample variance"):
        script.frame_probe(
            fit, select, score, columns=everything,
            groups=np.asarray(score["episode"]),
            resamples=FEW_RESAMPLES, seed=0,
        )


def test_frame_probe_refuses_a_code_in_which_nothing_varies():
    """R^2 is undefined for every column of a fully collapsed code, and
    `probe._mean_r2` raises in exactly that case. A NaN on the record instead
    would be refused by `capacity_arm` one phase later with a message about
    non-finite values rather than about a collapsed code."""
    fit, select, score = _splits(live_cats=0)
    assert live_classes(np.asarray(score["post_probs"], dtype=np.float64)) == 0
    with pytest.raises(ValueError, match="zero variance"):
        script.cell_capacity((fit, select, score), seed=0, resamples=FEW_RESAMPLES)


def test_cell_capacity_runs_without_a_select_split():
    """`gather_once` yields `select=None` when the training pool ends at the fit
    set, and `fit_probe` then takes its default penalty. `frame_ridge_selected`
    says which happened."""
    fit, _, score = _splits()
    cell = script.cell_capacity((fit, None, score), seed=0, resamples=FEW_RESAMPLES)
    assert cell["frame_ridge_selected"] is False
    assert np.isfinite(cell["frame_share"])
    selected = script.cell_capacity(_splits(), seed=0, resamples=FEW_RESAMPLES)
    assert selected["frame_ridge_selected"] is True


# ---------------------------------------------------------------------------
# The rows arrive in TEMPORAL ORDER, and nothing else enforces it
# ---------------------------------------------------------------------------


def test_cell_capacity_refuses_scored_rows_that_are_not_in_temporal_order():
    """`redundancy_floor` PERMUTES EACH CATEGORICAL BY AN INDEPENDENT CIRCULAR
    SHIFT, which preserves each series' own autocorrelation -- and that only
    works if consecutive rows are consecutive frames. Nothing in
    `gather_probe_data` or in `capacity.py` enforces it: the gather does produce
    rows in window order, but a caller that sorted, shuffled or subsampled them
    would silently turn the circular-shift floor into the plain permutation
    floor it replaces, against which independent-but-persistent categoricals
    read up to 27x their true ratio.

    So the order is ASSERTED rather than assumed, and the refusal names the
    floor. The shuffle here is a pure permutation: every row, every label and
    every count survives it, so no row count, cluster count or shape check
    anywhere else can see it."""
    fit, select, score = _splits()
    order = np.random.default_rng(0).permutation(score["post_probs"].shape[0])
    shuffled = {
        key: (value[order] if getattr(value, "shape", (0,))[:1] == (order.size,)
              else value)
        for key, value in score.items()
    }
    assert sorted(shuffled["step"].tolist()) == sorted(score["step"].tolist())
    assert not np.array_equal(shuffled["step"], score["step"]), "the fixture must shuffle"

    with pytest.raises(ValueError, match="temporal order"):
        script.cell_capacity((fit, select, shuffled), seed=0, resamples=FEW_RESAMPLES)


def test_require_temporal_order_accepts_the_gathers_own_row_order():
    """The other side: the real gather's labels must not be refused.
    `tests/eval/test_probe.py::test_gather_probe_data_labels_every_row_with_its_
    window_and_step` pins that the gather builds `window` in ascending
    contiguous runs with `step` restarting at 0 inside each, which is what this
    fixture reproduces."""
    script.require_temporal_order(_splits()[2])


@pytest.mark.parametrize("damage", ["reversed", "dropped", "interleaved"])
def test_require_temporal_order_catches_every_way_the_order_can_break(damage):
    """Three shapes of damage, each of which a row count cannot see.

    `reversed` keeps every window contiguous but runs time backwards.
    `dropped` removes one row from the middle of a window, so consecutive rows
    are no longer consecutive FRAMES while `step` is still ascending.
    `interleaved` keeps `step` ascending within each window but splits one
    window into two runs, which is what a stable sort on another key does."""
    data = _splits()[2]
    rows = data["post_probs"].shape[0]
    if damage == "reversed":
        index = np.arange(rows)[::-1]
    elif damage == "dropped":
        index = np.delete(np.arange(rows), 2)
    else:
        index = np.concatenate([np.arange(0, rows, 2), np.arange(1, rows, 2)])
    damaged = {
        key: (value[index] if getattr(value, "shape", (0,))[:1] == (rows,) else value)
        for key, value in data.items()
    }
    with pytest.raises(ValueError, match="temporal order"):
        script.require_temporal_order(damaged)


# ---------------------------------------------------------------------------
# base_control: three numbers recorded, position_r2 the one a tally is read from
# ---------------------------------------------------------------------------


def test_base_control_records_all_three_r2s_and_they_are_not_the_same_number():
    """`BASE_R2_FLOOR` stays 0.10 and is gated on POSITION ALONE (M3k's
    correction), so the record carries `r2` (the 4-column mean M3j gated on,
    kept to bridge to its records), `position_r2` (the gated number) and
    `per_column_r2` (so the split between position and heading is checkable
    without refitting anything).

    The fixture's heading is far harder to read than its position, so the two
    numbers differ decisively: a `position_r2` that was actually the 4-column
    mean would be caught, and no tally can be read from the wrong one."""
    control = script.base_control(*_splits())
    assert control["position_r2"] > 0.5
    assert control["position_r2"] > control["r2"] + 0.1, (
        "the 4-column mean must be decisively below the position-only figure, "
        "or this fixture cannot tell them apart"
    )
    assert len(control["per_column_r2"]) == 4
    assert np.mean(control["per_column_r2"]) == pytest.approx(control["r2"])
    assert np.mean(control["per_column_r2"][:2]) == pytest.approx(
        control["position_r2"],
    )
    assert control["rows"] == _splits()[2]["targets"].shape[0]
    assert control["ridge_selected"] is True


def test_base_control_reports_nan_for_a_zero_variance_target_column():
    """`_per_column_r2` duplicates `probe._mean_r2`'s per-column formula rather
    than importing its private name, so a zero-variance column reports NaN here
    -- a fact about that column -- instead of silently vanishing from the
    average the way it does inside `_mean_r2`."""
    splits = tuple({**data} for data in _splits())
    for data in splits:
        targets = np.array(data["targets"], dtype=np.float64)
        targets[:, 3] = 0.5
        data["targets"] = targets
    control = script.base_control(*splits)
    assert np.isnan(control["per_column_r2"][3])
    assert all(np.isfinite(v) for v in control["per_column_r2"][:3])
    assert np.isfinite(control["r2"]) and np.isfinite(control["position_r2"])


# ---------------------------------------------------------------------------
# measure_cell: the wiring, and the cell's own seed on every estimator call
# ---------------------------------------------------------------------------


def _stub_measure(monkeypatch, *, self_check_ok=True, gathers=None,
                  z_cats=CATS, z_classes=CLASSES):
    """`prepare_cell`, the reference pass and `self_check` replaced; the gathers
    are `_gathered` and everything downstream of them is the REAL code."""
    gathers = [] if gathers is None else gathers
    prepared = _prepared(z_cats=z_cats, z_classes=z_classes)

    def recorder(model, paths, backbone, device, *, context, horizon, limit, seed):
        gathers.append(seed)
        return _gathered(seed=seed)

    monkeypatch.setattr(
        script, "prepare_cell", lambda *a, **k: (script.EXIT_OK, prepared),
    )
    monkeypatch.setattr(script, "reference_trajectories", lambda *a, **k: "a-trajectory")
    monkeypatch.setattr(script, "self_check", lambda traj, diag: _Check(self_check_ok))
    monkeypatch.setattr(script, "gather_probe_data", recorder)
    return gathers


def test_measure_cell_refuses_a_failed_self_check_before_any_gather(monkeypatch):
    """The 30 check is not free the way `prepare_cell`'s 12/14 are: `self_check`
    reduces the PER-WINDOW rows, not the mean curves `prepare_cell` already
    compared. It must refuse BEFORE the gathers -- the expensive part -- and
    write no record."""
    gathers = _stub_measure(monkeypatch, self_check_ok=False)
    status, record = script.measure_cell(
        _measure_args(), _cell(), "cpu",
        _paths("t", 25), _paths("v", 24),
    )
    assert status == script.EXIT_SELF_CHECK_FAILED and record is None
    assert gathers == [], "a cell that failed its self-check still paid for a gather"


RECORD_KEYS = {
    "arm", "seed", "step", "record_git_sha", "device", "context", "horizon",
    "z_cats", "z_classes", "ceiling_bits", "split_seed", "torch_version",
    "capacity", "base_control", "clusters", "rows", "windows", "self_check",
    "episodes",
}


def test_measure_cell_builds_the_whole_record_and_it_survives_a_round_trip(
    monkeypatch, tmp_path,
):
    """End to end with the real `cell_capacity`, `estimator_checks` and
    `base_control`, then through the real `write_record`/`load_record`.

    The record is written only AFTER every gather and fit of the cell, so a
    value `write_record` cannot serialise would be discovered at the end of the
    cell rather than at the start -- which is why the round trip is pinned
    here and not left to the read phase."""
    _stub_measure(monkeypatch)
    train, val = _paths("t", 45), _paths("v", 24)
    status, record = script.measure_cell(
        _measure_args(), _cell(arm="pixel_ae", seed=3), "cpu", train, val,
    )
    assert status == script.EXIT_OK
    assert set(record) == RECORD_KEYS
    assert (record["arm"], record["seed"]) == ("pixel_ae", 3)
    assert record["z_cats"] == CATS and record["z_classes"] == CLASSES
    assert record["ceiling_bits"] == CEILING_BITS
    # Episode NAMES, which is what `prepare_cell` compares the split against --
    # `str(p)` would carry the directory and stop matching the study record.
    assert record["episodes"] == {
        "fit": [p.name for p in train[:script.FIT_EPISODES]],
        "select": [
            p.name for p in
            train[script.FIT_EPISODES:script.FIT_EPISODES + script.SELECT_EPISODES]
        ],
        "val": [p.name for p in val],
    }
    assert record["episodes"]["val"][0] == "v0.npz"
    assert len(record["windows"]["episode"]) == len(record["windows"]["window"])
    assert record["rows"] == len(record["windows"]["episode"])
    assert record["clusters"] == 6
    assert record["self_check"] == {"ok": True}
    assert set(record["capacity"]) == CAPACITY_KEYS

    path = tmp_path / "capacity.json"
    script.write_record(path, dict(record, git_sha="deadbeef"))
    loaded = load_record(path)
    assert loaded["capacity"]["bits"] == pytest.approx(record["capacity"]["bits"])
    assert loaded["capacity"]["redundancy_bits"] == pytest.approx(
        record["capacity"]["redundancy_bits"],
    )
    assert loaded["capacity"]["redundancy_floor"] == pytest.approx(
        record["capacity"]["redundancy_floor"],
    )
    assert loaded["capacity"]["prior_bits"] == pytest.approx(
        record["capacity"]["prior_bits"], abs=1e-9,
    )
    assert loaded["capacity"]["checks"] == record["capacity"]["checks"]
    assert loaded["base_control"]["position_r2"] == pytest.approx(
        record["base_control"]["position_r2"],
    )


def test_measure_cell_reads_the_latent_shape_off_the_checkpoint_not_the_config(
    monkeypatch,
):
    """`z_cats`/`z_classes` are the MEASURED shape, read off
    `prepared.model.rssm.cfg` -- the checkpoint's own -- which is the whole point
    of comparing them across records. Written from `RSSMConfig`'s class
    defaults they would be constants that cannot vary between two records of one
    code version, which is the trap M3k's review found in `rung_width`."""
    _stub_measure(monkeypatch, z_cats=CATS, z_classes=CLASSES + 1)
    _, record = script.measure_cell(
        _measure_args(), _cell(), "cpu",
        _paths("t", 45), _paths("v", 24),
    )
    assert record["z_classes"] == CLASSES + 1 != RSSMConfig.z_classes


ESTIMATOR_CALLS = ("bits_interval", "redundancy_floor", "frame_probe")
"""Every call `measure_cell` must hand the CELL'S OWN seed.

`redundancy_floor`'s `seed` is required rather than defaulted for exactly this
reason, and `bits_interval` and `frame_probe` each draw their own bootstrap."""


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_estimator_call_gets_the_cells_own_bootstrap_seed(monkeypatch, seed):
    """M3j shipped every cell sharing bootstrap seed 0, which correlates the
    interval noise the seeds x arms agreement rule treats as independent -- and
    `redundancy_floor`'s `seed` is REQUIRED for the same reason.

    A `Counter` OVER ALL THE CALLS, not the first: a check on one call passes
    against an implementation that threads the seed into the first estimator and
    defaults the rest. There are three estimator calls per cell, so that
    implementation is caught here -- and the parametrisation runs a cell whose
    seed is NOT 0, so a hardcoded default cannot pass either.

    Order-independent: the counter is keyed by `(function, seed)`, never by
    position in the call sequence."""
    estimator_seeds = Counter()
    gathers = _stub_measure(monkeypatch)
    for name in ESTIMATOR_CALLS:
        real = getattr(script, name)

        def wrapper(*args, _real=real, _name=name, **kwargs):
            estimator_seeds[(_name, kwargs["seed"])] += 1
            return _real(*args, **kwargs)

        monkeypatch.setattr(script, name, wrapper)

    script.measure_cell(
        _measure_args(), _cell(seed=seed), "cpu",
        _paths("t", 45), _paths("v", 24),
    )
    assert {s for _, s in estimator_seeds} == {seed}, (
        f"cell seed {seed} drew its bootstrap from {dict(estimator_seeds)}"
    )
    assert {name for name, _ in estimator_seeds} == set(ESTIMATOR_CALLS)
    assert sum(estimator_seeds.values()) == len(ESTIMATOR_CALLS)
    # The gather's three draws are `filtering_gain`'s, derived from the same
    # cell seed -- so an implementation handing every cell seed 0 fails here
    # as well as above.
    assert set(gathers) == {seed, seed + 1, seed + 2}


def test_measure_cell_never_passes_a_resample_count_of_its_own(monkeypatch):
    """`cell_capacity` defaults to the protocol's `RESAMPLES` and `measure_cell`
    must not override it, so a production run cannot depart from the
    pre-registered figure. The cheap-test lever is the default's own module
    constant, not an argument the measure path threads."""
    seen = {}
    real = script.cell_capacity

    def recorder(gathered, *, seed, **kwargs):
        seen.update(kwargs)
        return real(gathered, seed=seed, resamples=FEW_RESAMPLES)

    _stub_measure(monkeypatch)
    monkeypatch.setattr(script, "cell_capacity", recorder)
    script.measure_cell(
        _measure_args(), _cell(), "cpu",
        _paths("t", 45), _paths("v", 24),
    )
    assert seen == {}, f"measure_cell passed {sorted(seen)} to cell_capacity"


def test_the_cell_line_prints_the_numbers_a_reader_checks_the_verdict_against(
    monkeypatch, tmp_path, capsys,
):
    """One line per measured cell. `bits` beside its interval, `frame`, `live`
    and the checks: the position control and the three companions a reader needs
    to see a self-contradicting cell in a long log."""
    _, _, written = _stub_phase(monkeypatch)
    args = types.SimpleNamespace(out=tmp_path / "out", source=tmp_path, phase="measure")
    script.measure_phase(args, [("pixel_ae", 1)], "cpu", [], [])
    line = capsys.readouterr().out
    assert "pixel_ae seed 1" in line
    record = written[0][1]
    assert f"{record['capacity']['bits']:.4f}" in line
    assert f"{record['capacity']['frame_share']:.4f}" in line
    assert str(record["capacity"]["live"]) in line
    assert f"{record['base_control']['position_r2']:.3f}" in line
    assert str(written[0][0]) in line


# ---------------------------------------------------------------------------
# load_capacity: named, never skipped
# ---------------------------------------------------------------------------


def test_load_capacity_names_the_first_missing_cell_instead_of_pooling_what_is_there(
    tmp_path,
):
    """A pool over the cells that happen to be present, printed under the nine
    cells' names, is exactly the failure `pooling.MissingCell` exists for."""
    script.write_record(
        script.capacity_record_path(tmp_path, "pixel_ae", 0), _record("pixel_ae", 0),
    )
    with pytest.raises(script.CellMissing, match="pixel_ae seed 1"):
        script.load_capacity(tmp_path, ["pixel_ae"], [0, 1])


def test_load_capacity_refuses_a_record_whose_labels_are_not_its_filenames(tmp_path):
    """A swapped pair of files would pool one cell under another's name with
    every count still right, so the record's own `arm`/`seed` are checked
    against the path it was read under."""
    path = script.capacity_record_path(tmp_path, "pixel_ae", 0)
    script.write_record(path, _record("frozen_ssl", 2))
    with pytest.raises(SystemExit, match="frozen_ssl"):
        script.load_capacity(tmp_path, ["pixel_ae"], [0])


def test_load_capacity_returns_nine_distinct_payloads_keyed_by_cell(tmp_path):
    """The nine records must come back keyed by their own `(arm, seed)` and
    carrying NINE DISTINCT PAYLOADS. A fixture whose nine records serialise to
    one payload with the labels dropped let an earlier milestone pair one arm
    with another's seeds and pass every test, so the distinctness is asserted
    on the payloads with the labels REMOVED."""
    cells = [(arm, seed) for arm in ARMS for seed in (0, 1, 2)]
    for arm, seed in cells:
        script.write_record(
            script.capacity_record_path(tmp_path, arm, seed), _record(arm, seed),
        )
    records = script.load_capacity(tmp_path, list(ARMS), [0, 1, 2])
    assert set(records) == set(cells)
    assert all(
        records[cell]["arm"] == cell[0] and records[cell]["seed"] == cell[1]
        for cell in cells
    )
    stripped = {
        json.dumps(
            {k: v for k, v in record.items() if k not in ("arm", "seed")},
            sort_keys=True,
        )
        for record in records.values()
    }
    assert len(stripped) == 9, (
        "the fixture's nine records are not nine distinct payloads once the "
        "labels are dropped, so nothing here could catch an arm paired with "
        "another arm's seeds"
    )


# ---------------------------------------------------------------------------
# read: the fixture the whole section stands on
# ---------------------------------------------------------------------------

READ_CLUSTERS: int = 11
READ_WINDOWS: int = 13
READ_STEPS: int = 3
READ_ROWS: int = READ_WINDOWS * READ_STEPS
"""The three counts the self-check table prints, CHOSEN SO THAT NO ONE OF THEM
IS A SUBSTRING OF ANOTHER OR OF `step`.

They were 3 / 6 / 30 against a `step` of 20000-20008, and the three assertions
on them were substring matches against the WHOLE table: `"6"` was satisfied by
`20006` and `"3"` by `20003`, so neither could fail. The assertions now read
the row's own columns, and these values are a second, independent guard --
`test_the_self_check_counts_cannot_be_read_off_another_column` pins them, so a
later edit cannot quietly reintroduce an ambiguity the column parse is then the
only thing standing against.

TWO DIGITS, NECESSARILY: `step` runs 20000-20008 across the nine cells, so
EVERY digit 0-8 appears in some step and no one-digit count can avoid being a
substring of one. 11 / 13 / 39 avoid `20`, `00`, `0X` and the longer runs."""

_WINDOW_LABELS = [w for w in range(READ_WINDOWS) for _ in range(READ_STEPS)]
_EPISODE_LABELS = [
    w % READ_CLUSTERS for w in range(READ_WINDOWS) for _ in range(READ_STEPS)
]
"""One label per gathered row, IDENTICAL in every record: `windows.episode` and
`windows.window` are compared by `require_one_protocol`, so the per-cell rank
cannot live there. It lives in every MEASURED number instead."""


def _rank(arm, seed) -> int:
    """A distinct 0..8 per cell, threaded through every measured number."""
    return list(ARMS).index(arm) * 3 + int(seed)


def _position_r2(arm, seed, *, base=True) -> float:
    """The GATED figure: `enc(t)` -> position, per cell.

    TWO OF EACH ARM'S THREE SEEDS CLEAR `BASE_R2_FLOOR` and the third does not,
    on purpose. A base-control loop that collapsed onto any ONE seed would read
    a tally of 0 or 3, and both differ from the 2 this shape requires -- which
    is what makes the collapse visible in the TALLY rather than only in a
    cross-arm distinctness check it can walk straight past."""
    rank = _rank(arm, seed)
    if not base:
        return 0.01 + rank / 1000.0
    return 0.02 + rank / 1000.0 if int(seed) == 2 else 0.60 + rank / 100.0


def _four_column_r2(arm, seed) -> float:
    """The record's key literally named `r2`: the 4-column mean over pos_x,
    pos_y, sin(angle) and cos(angle) that M3j gated on and this reading does NOT.

    Separated from `_position_r2` by more than a whole unit of r2, and on the
    OTHER side of `BASE_R2_FLOOR`, so a read that takes the obvious key prints a
    figure below the floor beside a tally that says the floor was cleared."""
    return -1.50 - _rank(arm, seed) / 100.0


def _read_record(arm, seed, *, spare=False, framey=False, base=True,
                 collapsed=False, broken_flags=(), clusters=READ_CLUSTERS,
                 rows=READ_ROWS) -> dict:
    """One capacity record, every measured number a function of `(arm, seed)`.

    NINE DISTINCT PAYLOADS, never one template with the labels changed: M3j
    shipped a fixture whose nine records serialised to ONE payload once the
    labels were dropped, so an implementation pairing `frozen_ssl` with
    `pixel_ae`'s seeds passed every test in its read section."""
    rank = _rank(arm, seed)
    bits = (
        {"bits": 40.0 + rank, "ci_low": 30.0 + rank, "ci_high": 50.0 + rank} if spare
        else {"bits": 150.0 - rank / 10.0, "ci_low": 140.0 - rank / 10.0,
              "ci_high": 158.0 - rank / 10.0}
    )
    frame = (
        {"frame_share": 0.80 - rank / 100.0, "frame_low": 0.70 - rank / 100.0,
         "frame_high": 0.90 - rank / 100.0} if framey
        else {"frame_share": 0.20 + rank / 100.0, "frame_low": 0.10 + rank / 100.0,
              "frame_high": 0.30 + rank / 100.0}
    )
    redundancy = (
        {"redundancy_bits": 0.0, "redundancy_floor": 0.0} if collapsed
        else {"redundancy_bits": 0.10 + rank / 1000.0,
              "redundancy_floor": 0.05 + rank / 1000.0}
    )
    return {
        "arm": arm, "seed": int(seed), "step": 20000 + rank,
        "record_git_sha": "ca3e140", "device": "cpu", "context": 8, "horizon": 15,
        "z_cats": CATS, "z_classes": CLASSES, "ceiling_bits": CEILING_BITS,
        "split_seed": 1234, "torch_version": "2.4.0", "git_sha": "deadbeef",
        "capacity": {
            **bits, **frame, **redundancy,
            "confidence": CONFIDENCE, "resamples": RESAMPLES,
            "n_episodes": clusters,
            "frame_confidence": CONFIDENCE, "frame_resamples": RESAMPLES,
            "live": COLUMNS - rank, "prior_bits": 5.0 + rank,
            "checks": {
                flag: flag not in broken_flags for flag in script.CHECK_FLAGS
            },
            "rows": rows,
        },
        "base_control": {
            "r2": _four_column_r2(arm, seed),
            "position_r2": _position_r2(arm, seed, base=base),
            "per_column_r2": [0.1 + rank / 100.0] * 4,
            "ridge": 1000.0, "ridge_selected": True, "rows": rows,
        },
        "clusters": clusters,
        "rows": rows,
        "windows": {"episode": list(_EPISODE_LABELS), "window": list(_WINDOW_LABELS)},
        "self_check": {"ok": True},
        "episodes": {"fit": [], "select": [], "val": [f"e{i}" for i in range(24)]},
    }


def _pool(knobs=None, *, broken=None) -> dict:
    """The nine cells, keyed by `(arm, seed)`.

    `knobs` maps an ARM to the keyword overrides shared by its three records;
    `broken` maps a CELL to the `CHECK_FLAGS` it reports False."""
    knobs, broken = knobs or {}, broken or {}
    return {
        (arm, seed): _read_record(
            arm, seed, broken_flags=broken.get((arm, seed), ()), **knobs.get(arm, {}),
        )
        for arm in ARMS for seed in (0, 1, 2)
    }


def _write_pool(directory, records) -> None:
    for (arm, seed), record in records.items():
        script.write_record(script.capacity_record_path(directory, arm, seed), record)


def _main_read(*argv):
    """`main([...])` with the MEASURE half made unreachable.

    A mutation of `main` that let `--phase read` fall into the measure branch
    would otherwise load the real checkpoints from `--source`'s default and run a
    real cell on the GPU inside pytest. That must be a loud, instant
    AssertionError, never a measurement -- the guard `scripts/latent_width.py`'s
    suite grew after its absence cost a real 7-minute measure."""
    def forbidden(*args, **kwargs):
        raise AssertionError("--phase read reached the measure phase")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(script, "measure_phase", forbidden)
        patch.setattr(script, "ReplayBuffer", forbidden)
        patch.setattr(script, "get_device", forbidden)
        return script.main(list(argv))


def _read(tmp_path, capsys, records, *extra):
    """Write `records`, run the real read phase, return `(status, stdout)`."""
    _write_pool(tmp_path, records)
    status = _main_read("--phase", "read", "--out", str(tmp_path), *extra)
    return status, capsys.readouterr().out


def _section(text: str, fragment: str) -> str:
    """The one block of `capacity_text` whose heading carries `fragment`.

    `capacity_text` joins its sections with a blank line, so a section is a
    block. Selected by fragment and asserted UNIQUE: a per-cell row and a
    self-check row both begin with an arm and a seed, so a line picked by its
    first two tokens alone would be ambiguous between two tables."""
    blocks = [block for block in text.split("\n\n") if fragment in block]
    assert len(blocks) == 1, f"{fragment!r} matched {len(blocks)} blocks"
    return blocks[0]


def _line_with(text: str, fragment: str) -> str:
    lines = [line for line in text.splitlines() if fragment in line]
    assert len(lines) == 1, f"{fragment!r} matched {len(lines)} lines: {lines}"
    return lines[0]


def _row_for(table: str, *tokens) -> str:
    """The one row of `table` whose leading whitespace-separated tokens are
    `tokens`.

    By TOKEN, never by substring: every prose line in these sections also
    contains every arm name, and `158.0000` contains `8.0000`, so a row picked
    by `fragment in line` is ambiguous in both tables at once."""
    wanted = [str(token) for token in tokens]
    rows = [
        line for line in table.splitlines()
        if line.split()[: len(wanted)] == wanted
    ]
    assert len(rows) == 1, f"{wanted} matched {len(rows)} rows: {rows}"
    return rows[0]


def _mean(records, cells, *path) -> float:
    values = []
    for cell in cells:
        value = records[cell]
        for key in path:
            value = value[key]
        values.append(float(value))
    return float(np.mean(values))


def _cells_of(arm) -> list[tuple[str, int]]:
    return [(arm, seed) for seed in (0, 1, 2)]


def test_the_read_fixture_carries_nine_distinct_payloads():
    """Guards the FIXTURE every read test below stands on. Nine records that
    differed only in their labels would let a cross-arm pairing, a seed collapse
    or a first-record-stands-for-all pass every one of them -- M3j shipped
    exactly that, so the labels are DROPPED before the payloads are compared."""
    records = _pool()
    payloads = {
        json.dumps(
            {k: v for k, v in record.items() if k not in ("arm", "seed")},
            sort_keys=True,
        )
        for record in records.values()
    }
    assert len(payloads) == 9, "the nine records are not nine distinct payloads"
    for name, pick in (
        ("bits", lambda r: r["capacity"]["bits"]),
        ("ci_low", lambda r: r["capacity"]["ci_low"]),
        ("ci_high", lambda r: r["capacity"]["ci_high"]),
        ("frame_share", lambda r: r["capacity"]["frame_share"]),
        ("frame_low", lambda r: r["capacity"]["frame_low"]),
        ("live", lambda r: r["capacity"]["live"]),
        ("prior_bits", lambda r: r["capacity"]["prior_bits"]),
        ("redundancy_bits", lambda r: r["capacity"]["redundancy_bits"]),
        ("redundancy_floor", lambda r: r["capacity"]["redundancy_floor"]),
        ("position_r2", lambda r: r["base_control"]["position_r2"]),
        ("r2", lambda r: r["base_control"]["r2"]),
    ):
        assert len({pick(r) for r in records.values()}) == 9, (
            f"{name} is not distinct in all nine cells, so nothing below could "
            "catch one cell's number printed in another cell's row"
        )


# ---------------------------------------------------------------------------
# READ_EXITS: keyed on exactly the two refusal statuses
# ---------------------------------------------------------------------------


def test_read_exits_is_keyed_on_exactly_the_two_refusal_statuses():
    """`reading_capacity` returns five statuses and only two are refusals. A
    third key would exit non-zero on a FINDING, which makes "the run worked" and
    "the news was good" the same signal; a missing one exits 0 on a reading that
    is not a reading."""
    assert script.READ_EXITS == {
        "UNRESOLVED_ESTIMATOR": script.EXIT_ESTIMATOR_BROKEN,
        "UNRESOLVED_BASE": script.EXIT_BASE_UNRESOLVED,
    }


# ---------------------------------------------------------------------------
# The five statuses, end to end through `main --phase read`
# ---------------------------------------------------------------------------

_SPARE = {arm: {"spare": True} for arm in ARMS}
_FRAMEY = {arm: {"framey": True} for arm in ARMS}
_BASE_FAILS = {arm: {"base": False} for arm in list(ARMS)[:2]}
_BROKEN_CELL = ("random_vit", 1)

STATUS_POOLS = {
    "UNRESOLVED_ESTIMATOR": ({}, {_BROKEN_CELL: ("floor_ok",)}),
    "UNRESOLVED_BASE": (_BASE_FAILS, {}),
    "SPARE_CAPACITY": (_SPARE, {}),
    "FRAME_REENCODING": (_FRAMEY, {}),
    "CAPACITY_BOUND": ({}, {}),
}
"""One `(knobs, broken)` pair per status `reading_capacity` can return.

M3k's formatter was exercised on ONE status of five: its `verdict:` line could be
deleted and its `ok`/`BROKEN` inverted with all 65 tests still passing. So every
one of the five is driven end to end below."""

STATUS_EXITS = {
    "UNRESOLVED_ESTIMATOR": 43,
    "UNRESOLVED_BASE": 44,
    "SPARE_CAPACITY": 0,
    "FRAME_REENCODING": 0,
    "CAPACITY_BOUND": 0,
}


def test_the_five_status_pools_are_the_five_statuses_the_reading_can_return():
    """A pool that drifted onto another status would silently stop exercising
    the branch it is named for, which is how a five-way parametrisation becomes
    one case repeated five times."""
    assert set(STATUS_POOLS) == set(STATUS_EXITS)
    assert set(STATUS_POOLS) >= set(script.READ_EXITS)
    reached = set()
    for status, (knobs, broken) in STATUS_POOLS.items():
        inputs = script.capacity_inputs(_pool(knobs, broken=broken))
        reached.add(reading_capacity(inputs).status)
    assert reached == set(STATUS_POOLS), f"the pools reach only {sorted(reached)}"


@pytest.mark.parametrize("status", sorted(STATUS_POOLS))
def test_every_status_is_driven_end_to_end_and_prints_its_own_verdict(
    tmp_path, capsys, status,
):
    """All five statuses through `main --phase read`: the exit code, the
    `verdict:` line naming THAT status, and the estimator-checks line's
    `ok`/`BROKEN` matching the records cell by cell.

    M3k's `verdict:` line could be DELETED and its `ok`/`BROKEN` INVERTED with
    all 65 of its tests passing, because its formatter was exercised on one
    status of five. Each of those two mutations must fail this test for every
    status."""
    knobs, broken = STATUS_POOLS[status]
    records = _pool(knobs, broken=broken)
    code, out = _read(tmp_path, capsys, records)

    assert code == STATUS_EXITS[status]
    assert code == script.READ_EXITS.get(status, script.EXIT_OK)

    verdict = _line_with(out, "verdict:")
    assert f"verdict: {status.replace('_', ' ')} --" in verdict, verdict
    for other in STATUS_POOLS:
        if other != status:
            assert other.replace("_", " ") not in verdict, (
                f"the {status} verdict line also names {other}"
            )

    checks = _line_with(out, "estimator checks (")
    for arm in ARMS:
        every_flag_held = all(
            records[cell]["capacity"]["checks"][flag]
            for cell in _cells_of(arm) for flag in script.CHECK_FLAGS
        )
        assert f"{arm}={'ok' if every_flag_held else 'BROKEN'}" in checks, checks
        assert f"{arm}={'BROKEN' if every_flag_held else 'ok'}" not in checks, checks

    companion = _section(out, "redundancy companion")
    for arm in ARMS:
        assert f"{arm}: redundancy_bits" in companion, (
            "both redundancy numbers and their ratio print in EVERY status, "
            "because they are the only thing that makes a high bits readable"
        )


def test_the_fall_through_says_it_stands_by_default_rather_than_by_evidence():
    """`CAPACITY_BOUND` is what remains when neither objective-lever status
    clears an interval, so the lever M3j's evidence already favours is the
    EASIEST status to reach. The rule text has to say so."""
    inputs = script.capacity_inputs(_pool())
    reading = reading_capacity(inputs)
    assert reading.status == "CAPACITY_BOUND"
    assert "BY DEFAULT rather than by evidence" in reading.rule


def test_an_arm_can_clear_both_cuts_and_the_table_says_spare_plus_frame(
    tmp_path, capsys,
):
    """The two readings are NOT mutually exclusive: they are tallies on two
    different quantities, so a code that carries little and of that little
    mostly the frame clears both. Precedence picks the label; the column must
    still report the overlap, or a reader comparing it with `arms_frame` would
    read an arm that did not clear the frame cut."""
    knobs = {arm: {"spare": True, "framey": True} for arm in ARMS}
    records = _pool(knobs)
    code, out = _read(tmp_path, capsys, records)
    assert code == script.EXIT_OK
    assert "verdict: SPARE CAPACITY --" in out
    table = _section(out, "--- Reading G:")
    for arm in ARMS:
        row = _row_for(table, arm)
        assert "spare+frame" in row, row


# ---------------------------------------------------------------------------
# base_controls: the `r2` name collision, and the per-seed tally
# ---------------------------------------------------------------------------


def test_the_printed_base_r2_is_the_position_mean_and_never_the_four_column_one(
    tmp_path, capsys,
):
    """THE defect this task exists to avoid. `BaseControl` has a field named
    `r2`; the RECORD has a key named `r2` too, and the record's is the 4-COLUMN
    mean over position and heading that M3j gated on and this reading does NOT.
    `format_reading_capacity` prints `inputs.base[a].r2` immediately beside
    `seeds_clear/seeds_total`, so taking the obvious key prints the ungated
    figure beside the tally the whole reading is gated on -- the "one tally
    beside a verdict read from the other" defect this project has shipped three
    times. `base_control`'s own docstring warns of it and nothing enforces it.
    """
    records = _pool()
    controls = script.base_controls(records)
    assert set(controls) == set(ARMS)
    for arm in ARMS:
        cells = _cells_of(arm)
        position = _mean(records, cells, "base_control", "position_r2")
        four = _mean(records, cells, "base_control", "r2")
        assert abs(position - four) > 1.0, (
            "the fixture must separate the two means by more than a whole unit "
            "of r2, or the swap would not be unmistakable in a +.3f cell"
        )
        assert position > BASE_R2_FLOOR > four, (
            "the two means must also sit on OPPOSITE sides of the floor, so the "
            "swap prints a figure below the gate beside a tally that cleared it"
        )
        assert controls[arm].r2 == pytest.approx(position)
        assert controls[arm].r2 != pytest.approx(four)

    code, out = _read(tmp_path, capsys, records)
    assert code == script.EXIT_OK
    line = _line_with(out, "base control (enc(t)")
    for arm in ARMS:
        cells = _cells_of(arm)
        position = _mean(records, cells, "base_control", "position_r2")
        four = _mean(records, cells, "base_control", "r2")
        clear = controls[arm].seeds_clear
        assert f"{arm} r2={position:+.3f} {clear}/3" in line, line
        assert f"{four:+.3f}" not in out, (
            "the 4-column mean is the ungated figure and prints NOWHERE: it is "
            "kept on the record to bridge to M3j's and nothing more"
        )


def test_the_base_control_tally_is_counted_per_seed_and_cannot_collapse(tmp_path):
    """A seed collapse in the BASE-CONTROL loop specifically -- the worst place
    for one, because the base control is a GATE.

    The dangerous variant is every arm reporting its OWN seed 2 three times: the
    three arms' r2 values still all differ, so a cross-arm DISTINCTNESS check
    walks straight past it. What catches it is the TALLY: two of each arm's three
    seeds clear the floor and the third does not, so a collapse onto any single
    seed reads 0 or 3 and never 2."""
    records = _pool()
    controls = script.base_controls(records)
    for arm in ARMS:
        levels = [
            float(records[cell]["base_control"]["position_r2"])
            for cell in _cells_of(arm)
        ]
        expected = sum(1 for level in levels if level > BASE_R2_FLOOR)
        assert 0 < expected < 3, (
            "the fixture must mix clearing and non-clearing seeds INSIDE each "
            "arm, or a collapse onto one seed would still read the right tally"
        )
        assert controls[arm].seeds_total == 3
        assert controls[arm].seeds_clear == expected
        assert controls[arm].r2 == pytest.approx(float(np.mean(levels)))
    assert len({controls[arm].r2 for arm in ARMS}) == 3, (
        "the arms' means must differ too, which is the weaker check the tally "
        "above exists to back up"
    )


def test_the_base_control_gate_is_strict_and_stays_at_the_imported_floor():
    """`BASE_R2_FLOOR` is 0.10, applied with a strict `>` per seed, and imported
    from `retention` rather than re-spelled. A seed exactly ON the floor does not
    clear."""
    assert script.BASE_R2_FLOOR is BASE_R2_FLOOR
    assert BASE_R2_FLOOR == 0.10
    records = _pool()
    for cell in records:
        records[cell]["base_control"]["position_r2"] = BASE_R2_FLOOR
    controls = script.base_controls(records)
    assert all(controls[arm].seeds_clear == 0 for arm in ARMS)


def test_the_base_control_refuses_a_non_finite_position_r2():
    """A NaN compares False against every threshold, so it would read as "did
    not clear" AND "did not fail" at once -- an error about the measurement."""
    records = _pool()
    victim = sorted(records)[-1]
    records[victim]["base_control"]["position_r2"] = None
    with pytest.raises(SystemExit, match="random_vit seed 2"):
        script.base_controls(records)


# ---------------------------------------------------------------------------
# control_flags: every seed, every flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", list(script.CHECK_FLAGS))
def test_one_broken_flag_in_one_seed_breaks_that_arm_and_only_that_arm(flag):
    """`control_flags` is `all` over every one of an arm's seeds and every one of
    `CHECK_FLAGS` -- iterated, never re-spelled, so a fourth check cannot be
    added without a consumer. `any` would let a broken estimator vote; reading
    one flag of three would let two of them break in silence."""
    records = _pool(broken={_BROKEN_CELL: (flag,)})
    flags = script.control_flags(records)
    assert set(flags) == set(ARMS)
    assert flags[_BROKEN_CELL[0]] is False
    assert all(flags[arm] is True for arm in ARMS if arm != _BROKEN_CELL[0])


def test_control_flags_hold_when_every_seed_passes_every_check():
    assert script.control_flags(_pool()) == dict.fromkeys(ARMS, True)


def test_control_flags_refuse_a_check_that_is_not_a_boolean():
    """`"false"` is truthy, so a damaged record would otherwise read as a check
    that PASSED -- and `UNRESOLVED_ESTIMATOR` outranks every other status."""
    records = _pool()
    victim = sorted(records)[-1]
    records[victim]["capacity"]["checks"]["routes_ok"] = "false"
    with pytest.raises(SystemExit, match="random_vit seed 2"):
        script.control_flags(records)


# ---------------------------------------------------------------------------
# capacity_inputs: named refusals where the reading would raise
# ---------------------------------------------------------------------------


def test_capacity_inputs_summarises_each_arm_over_its_own_seeds(tmp_path):
    """A cross-arm pairing leaves every count right. The arm's `bits` is the mean
    of ITS OWN three cells and nothing else's."""
    records = _pool()
    inputs = script.capacity_inputs(records)
    assert set(inputs.arms) == set(ARMS)
    assert set(inputs.base) == set(ARMS) and set(inputs.controls) == set(ARMS)
    for arm in ARMS:
        cells = _cells_of(arm)
        arm_reading = inputs.arms[arm]
        assert arm_reading.seeds_total == 3
        assert arm_reading.bits == pytest.approx(_mean(records, cells, "capacity", "bits"))
        assert arm_reading.frame_share == pytest.approx(
            _mean(records, cells, "capacity", "frame_share")
        )
        assert arm_reading.bits_low == pytest.approx(
            min(records[c]["capacity"]["ci_low"] for c in cells)
        )
        assert arm_reading.bits_high == pytest.approx(
            max(records[c]["capacity"]["ci_high"] for c in cells)
        )
        assert arm_reading.live == min(records[c]["capacity"]["live"] for c in cells)
        assert arm_reading.redundancy_bits == pytest.approx(
            _mean(records, cells, "capacity", "redundancy_bits")
        )
        assert arm_reading.redundancy_floor == pytest.approx(
            _mean(records, cells, "capacity", "redundancy_floor")
        )
    assert len({inputs.arms[arm].bits for arm in ARMS}) == 3


@pytest.mark.parametrize("path", [
    ("capacity", "bits"),
    ("capacity", "ci_low"),
    ("capacity", "frame_low"),
    ("capacity", "live"),
    ("capacity", "redundancy_bits"),
    ("capacity", "redundancy_floor"),
    ("base_control", "position_r2"),
    ("capacity", "checks"),
    ("clusters",),
    ("rows",),
])
def test_a_missing_field_is_refused_by_cell_and_by_dotted_path(path):
    """Every field the reading needs goes through `_get`, so a record that lacks
    one stops the read with the CELL and the DOTTED PATH rather than surfacing
    as a bare `KeyError` after the reader has paid for the measure.
    `redundancy_bits` and `redundancy_floor` are among them: they are threaded
    into each per-seed dict for `capacity_arm`."""
    records = _pool()
    victim = sorted(records)[-1]
    container = records[victim]
    for key in path[:-1]:
        container = container[key]
    del container[path[-1]]
    with pytest.raises(SystemExit) as raised:
        script.capacity_inputs(records)
    message = str(raised.value)
    assert "random_vit seed 2" in message, message
    assert ".".join(path) in message, message


def test_capacity_inputs_refuses_too_few_arms_by_name():
    """`reading_capacity` raises on fewer than `ARMS_REQUIRED` arms INSIDE the
    reading. Asked here it is a named refusal; asked there it is a traceback
    after every gather and every fit."""
    records = {
        cell: record for cell, record in _pool().items() if cell[0] == list(ARMS)[0]
    }
    with pytest.raises(SystemExit) as raised:
        script.capacity_inputs(records)
    assert str(ARMS_REQUIRED) in str(raised.value)


def test_capacity_inputs_refuses_arms_that_do_not_share_a_seed_count():
    """Arms that disagree on `seeds_total` have no true "N of M seeds" rule, and
    `reading_capacity` hands back a status the formatter then refuses to print --
    the verdict a caller would act on before it crashed at print time."""
    records = _pool()
    del records[(list(ARMS)[0], 2)]
    with pytest.raises(SystemExit) as raised:
        script.capacity_inputs(records)
    assert "seed count" in str(raised.value)


def test_capacity_inputs_refuses_a_seed_count_below_the_bar_naming_the_arm():
    """A shared `seeds_total` below `SEEDS_REQUIRED` makes BOTH objective-lever
    statuses structurally unreachable, so the reading would fall through to
    `CAPACITY_BOUND` -- a DIRECTIONAL verdict read from a run too small to clear
    either bar."""
    records = {
        cell: record for cell, record in _pool().items() if cell[1] == 0
    }
    with pytest.raises(SystemExit) as raised:
        script.capacity_inputs(records)
    message = str(raised.value)
    assert str(SEEDS_REQUIRED) in message, message
    assert any(arm in message for arm in ARMS), message


def test_capacity_inputs_names_the_arm_when_a_seed_is_not_a_measurement():
    """`capacity_arm`'s refusals -- a non-finite value, an inverted interval, a
    point estimate outside its own interval -- are errors about the measurement,
    and they become a refusal naming the arm and its seeds."""
    records = _pool()
    records[("frozen_ssl", 1)]["capacity"]["ci_low"] = 999.0
    with pytest.raises(SystemExit) as raised:
        script.capacity_inputs(records)
    message = str(raised.value)
    assert "frozen_ssl" in message and "[0, 1, 2]" in message, message


@pytest.mark.parametrize("field", ["clusters", "rows"])
def test_a_caption_figure_the_records_disagree_on_is_refused_naming_the_cells(field):
    """`clusters` and `rows` are SINGLE caption figures speaking for nine cells,
    and `require_one_protocol` deliberately does not compare them. The victim
    sorts LAST, so an implementation that took the FIRST record's value would see
    nothing at all and print a caption true of eight cells out of nine."""
    records = _pool()
    victim = sorted(records)[-1]
    assert victim == ("random_vit", 2), "the victim must sort last"
    records[victim][field] = records[victim][field] + 1
    with pytest.raises(SystemExit) as raised:
        script.capacity_inputs(records)
    message = str(raised.value)
    assert field in message, message
    assert "random_vit seed 2" in message, message


def test_the_caption_figures_come_from_the_records_and_reach_the_caption(
    tmp_path, capsys,
):
    records = _pool()
    inputs = script.capacity_inputs(records)
    assert (inputs.clusters, inputs.rows) == (READ_CLUSTERS, READ_ROWS)
    _, out = _read(tmp_path, capsys, records)
    assert f"{READ_ROWS} rows over {READ_CLUSTERS} clusters" in out


# ---------------------------------------------------------------------------
# capacity_text
# ---------------------------------------------------------------------------


def test_the_evidence_is_printed_before_the_conclusion(tmp_path, capsys):
    """The self-check, then the per-cell table, then Reading G: a reader meets
    the instrument and the numbers before the verdict, `motion.txt`'s order."""
    _, out = _read(tmp_path, capsys, _pool())
    order = [
        out.index("self-check per record"),
        out.index("per cell:"),
        out.index("redundancy companion"),
        out.index("--- Reading G:"),
        out.index("verdict:"),
    ]
    assert order == sorted(order), out


def test_every_cell_s_numbers_land_in_its_own_row(tmp_path, capsys):
    """One cell's number printed in another cell's row is this project's
    recurring table defect. Every number here is a function of `(arm, seed)`, so
    a row carrying a neighbour's value is visible."""
    records = _pool()
    _, out = _read(tmp_path, capsys, records)
    table = _section(out, "per cell:")
    header, *rows = [line for line in table.splitlines() if "---" not in line]
    assert len(rows) == 9
    for (arm, seed), record in sorted(records.items()):
        capacity = record["capacity"]
        row = _row_for(table, arm, seed)
        assert f"{capacity['prior_bits']:.4f}" in row
        assert f"{capacity['bits']:.4f}" in row
        assert f"{capacity['ci_low']:.4f}" in row
        assert f"{capacity['ci_high']:.4f}" in row
        assert f"{capacity['frame_share']:.4f}" in row
        assert f"{capacity['frame_low']:.4f}" in row
        assert str(capacity["live"]) in row
        assert f"{record['base_control']['position_r2']:+.4f}" in row
        assert len(row) == len(header), (
            "a value as wide as its column butts against its neighbour and a "
            "wider one shifts the whole row"
        )


def test_the_read_table_spells_a_broken_check_the_way_the_measure_line_does(
    tmp_path, capsys,
):
    """`checks BROKEN(floor)` is the spelling a long log is grepped for. The
    measure line and the read table must not drift apart on it."""
    records = _pool(broken={_BROKEN_CELL: ("floor_ok", "bracket_ok")})
    _, out = _read(tmp_path, capsys, records)
    table = _section(out, "per cell:")
    broken_row = _row_for(table, *_BROKEN_CELL)
    assert "BROKEN(floor,bracket)" in broken_row, broken_row
    assert _line_with(table, "BROKEN(") == broken_row
    measure_line = script._cell_line(records[_BROKEN_CELL], Path("somewhere.json"))
    assert "BROKEN(floor,bracket)" in measure_line, measure_line
    assert sum(1 for line in table.splitlines() if "BROKEN(" in line) == 1
    assert sum(1 for line in table.splitlines() if " ok" in line) == 8


def test_the_redundancy_companion_prints_undefined_for_a_collapsed_code(
    tmp_path, capsys,
):
    """A collapsed code reads its redundancy EXACTLY equal to its floor -- every
    row identical makes the circular shift the identity -- so the naive ratio is
    exactly 1.0, the single value that says "independent, so the capacity is in
    use", for a code with nothing in it. `redundancy_ratio` returns None and the
    table must print `undefined`: not `n/a`, and never a bare 1.0."""
    arm = "pixel_ae"
    records = _pool({arm: {"collapsed": True}})
    code, out = _read(tmp_path, capsys, records)
    assert code == script.EXIT_OK
    line = _line_with(_section(out, "redundancy companion"), f"{arm}: redundancy_bits")
    assert "undefined" in line, line
    assert "1.000" not in line and "n/a" not in line, line
    row = _row_for(_section(out, "--- Reading G:"), arm)
    assert "undefined" in row, row
    for other in ARMS:
        if other == arm:
            continue
        ratio = redundancy_ratio(
            _mean(records, _cells_of(other), "capacity", "redundancy_bits"),
            _mean(records, _cells_of(other), "capacity", "redundancy_floor"),
        )
        other_line = _line_with(
            _section(out, "redundancy companion"), f"{other}: redundancy_bits",
        )
        assert f"{ratio:.3f}" in other_line, other_line
        assert "undefined" not in other_line, other_line


def test_the_redundancy_companion_prints_both_numbers_not_only_the_ratio(
    tmp_path, capsys,
):
    """The ratio alone cannot be audited: a reader has to see how many bits per
    pair the categoricals duplicate and where the circular-shift null landed."""
    records = _pool()
    _, out = _read(tmp_path, capsys, records)
    companion = _section(out, "redundancy companion")
    for arm in ARMS:
        cells = _cells_of(arm)
        bits = _mean(records, cells, "capacity", "redundancy_bits")
        floor = _mean(records, cells, "capacity", "redundancy_floor")
        line = _line_with(companion, f"{arm}: redundancy_bits")
        assert f"{bits:.4f}" in line and f"{floor:.4f}" in line, line


def test_the_self_check_counts_cannot_be_read_off_another_column():
    """GUARDS THE FIXTURE the test below stands on, the way
    `test_the_sentinel_still_carries_what_it_is_for` guards the byte sentinel.

    `windows`, `gathered` and `clusters` are three counts printed side by side
    with `step`. While they were 6 / 30 / 3 against steps of 20000-20008, a
    substring assertion on any of them was satisfied by the `step` column alone
    -- which is how three assertions that looked like measurements could not
    fail. A later edit that set them back to such values would leave the column
    parse as the only thing standing against it; this says so out loud."""
    counts = {
        "windows": str(READ_WINDOWS),
        "gathered": str(READ_ROWS),
        "clusters": str(READ_CLUSTERS),
    }
    assert len(set(counts.values())) == 3, counts
    for (one, first), (two, second) in itertools.permutations(counts.items(), 2):
        assert first not in second, f"{one}={first} is a substring of {two}={second}"
    steps = {str(record["step"]) for record in _pool().values()}
    assert len(steps) == 9, steps
    for name, value in counts.items():
        for step in steps:
            assert value not in step, f"{name}={value} is a substring of step {step}"
    # And the labels really do produce those three counts, so the fixture's
    # constants are not three numbers the table never prints.
    assert len(set(zip(_EPISODE_LABELS, _WINDOW_LABELS, strict=True))) == READ_WINDOWS
    assert len(_EPISODE_LABELS) == READ_ROWS
    assert len(set(_EPISODE_LABELS)) == READ_CLUSTERS


def test_the_self_check_table_reports_what_each_cell_was_scored_on(
    tmp_path, capsys,
):
    """BY COLUMN, never by substring against the whole table.

    The assertion this replaces was `str(READ_WINDOWS) in table` for each of
    the three counts, and it PROVABLY COULD NOT FAIL: the `step` column carries
    20000-20008, so `"6"` was satisfied by `20006` and `"3"` by `20003` no
    matter what the `windows` and `clusters` columns held. Two mutations of
    `_window_count` survived it -- `return len(episode)`, which would have
    printed the shipped table's `windows` as 11450 instead of 229, and
    `return len(set(episode))`, which would have printed 24.

    Indexed through `SELF_CHECK_COLUMNS` rather than by position, so a column
    inserted ahead of `windows` cannot silently re-aim the assertion, and
    `strict=True` pins the row's own width against the header's."""
    records = _pool()
    _, out = _read(tmp_path, capsys, records)
    table = _section(out, "self-check per record")
    assert len([line for line in table.splitlines() if "---" not in line]) == 10
    for (arm, seed), record in sorted(records.items()):
        # Selected by the cell's own `step`, which is unique per cell: an arm
        # name matches three rows and a seed matches three more.
        row = _row_for(table, arm, seed)
        cells = dict(zip(script.SELF_CHECK_COLUMNS, row.split(), strict=True))
        assert cells["step"] == str(record["step"]), row
        assert cells["windows"] == str(READ_WINDOWS), row
        assert cells["gathered"] == str(READ_ROWS), row
        assert cells["clusters"] == str(READ_CLUSTERS), row
        assert cells["ok"] == "yes", row


def test_the_self_check_table_refuses_window_lists_of_different_lengths():
    """Both lists are one entry per gathered row and cannot describe one set of
    rows if they differ, so the record is refused BY NAME rather than through a
    bare `zip(strict=True)` traceback naming neither."""
    records = _pool()
    victim = sorted(records)[-1]
    records[victim]["windows"]["window"] = records[victim]["windows"]["window"][:-1]
    with pytest.raises(SystemExit, match="random_vit seed 2"):
        script.capacity_text(
            records, script.capacity_inputs(records),
            reading_capacity(script.capacity_inputs(records)),
        )


# ---------------------------------------------------------------------------
# capacity.txt is byte for byte what is printed
# ---------------------------------------------------------------------------

BYTE_SENTINEL = "verdict: SENTINEL\t   \n\n\n"
"""A text whose tail is exactly what a `rstrip()` destroys: trailing spaces, a
tab and three trailing newlines.

Both of this project's previous byte-identity tests were satisfied by
`path.write_text(text.rstrip())`, because both compared texts that had no
trailing whitespace to lose. Nothing weaker than this catches it."""


def test_the_sentinel_still_carries_what_it_is_for():
    """The sentinel IS the test. A drifted constant would turn the byte-identity
    assertion below into one `write_text(text.rstrip())` satisfies again."""
    assert BYTE_SENTINEL.endswith("\n\n\n")
    assert "\t" in BYTE_SENTINEL
    assert BYTE_SENTINEL[:-3].endswith(" ")
    assert BYTE_SENTINEL.rstrip() != BYTE_SENTINEL


def test_capacity_txt_is_byte_identical_to_stdout(monkeypatch, tmp_path, capsys):
    """`capacity.txt` is the artefact and stdout is the log, and they must be ONE
    string. Driven with `capacity_text` substituted by a sentinel whose trailing
    whitespace a `rstrip()` would eat."""
    _write_pool(tmp_path, _pool())
    monkeypatch.setattr(script, "capacity_text", lambda *a, **k: BYTE_SENTINEL)
    code = _main_read("--phase", "read", "--out", str(tmp_path))
    printed = capsys.readouterr().out
    written = (tmp_path / "capacity.txt").read_bytes()
    assert code == script.EXIT_OK
    assert written == BYTE_SENTINEL.encode(), (
        "capacity.txt is not byte for byte the text; a `write_text(text.rstrip())` "
        "would pass every weaker comparison"
    )
    assert printed == BYTE_SENTINEL
    assert written.decode() == printed


def test_the_real_text_is_written_and_printed_as_one_string(tmp_path, capsys):
    """The sentinel pins the WRITE; this pins the two against the real text, so
    a formatter change cannot be written one way and printed another."""
    records = _pool()
    _, out = _read(tmp_path, capsys, records)
    written = (tmp_path / "capacity.txt").read_bytes()
    assert written.decode() == out
    inputs = script.capacity_inputs(records)
    assert out == script.capacity_text(records, inputs, reading_capacity(inputs))


def test_a_second_read_phase_reproduces_capacity_txt_byte_for_byte(
    tmp_path, capsys,
):
    """The artefact has to be reproducible from the records alone: a second
    `--phase read` over the same nine records writes the same bytes."""
    records = _pool()
    first_code, first_out = _read(tmp_path, capsys, records)
    first = (tmp_path / "capacity.txt").read_bytes()
    second_code = _main_read("--phase", "read", "--out", str(tmp_path))
    second_out = capsys.readouterr().out
    assert (first_code, second_code) == (script.EXIT_OK, script.EXIT_OK)
    assert (tmp_path / "capacity.txt").read_bytes() == first
    assert second_out == first_out


def test_a_refused_read_writes_no_artefact(tmp_path, capsys):
    """A refused read has no artefact: `capacity.txt` must not be left holding
    the previous run's verdict under a refusal."""
    records = _pool()
    records[sorted(records)[-1]]["clusters"] += 1
    _write_pool(tmp_path, records)
    with pytest.raises(SystemExit):
        _main_read("--phase", "read", "--out", str(tmp_path))
    capsys.readouterr()
    assert not (tmp_path / "capacity.txt").exists()


# ---------------------------------------------------------------------------
# read_phase and main
# ---------------------------------------------------------------------------


def test_read_phase_names_the_first_missing_cell_and_exits_eleven(tmp_path, capsys):
    records = _pool()
    del records[("random_vit", 2)]
    _write_pool(tmp_path, records)
    code = _main_read("--phase", "read", "--out", str(tmp_path))
    out = capsys.readouterr().out
    assert code == script.EXIT_NO_CHECKPOINTS
    assert "NO CELL" in out and "random_vit seed 2" in out
    assert not (tmp_path / "capacity.txt").exists()


def test_read_phase_refuses_records_that_disagree_on_the_protocol(tmp_path, capsys):
    """THE WIRING, not the function: `require_one_protocol` is well tested in
    isolation, and deleting its one call from `read_phase` left all 452 targeted
    tests green. Every sibling wiring on this path already fails under a skip
    mutation -- `capacity_inputs` -> `_require_readable_records`,
    `capacity_inputs` -> `_pooled_shape`, `cell_capacity` ->
    `require_temporal_order`, `read_phase` -> `require_readable_plan`,
    `reading_capacity` -> `_common_seeds_total`. This was the one unpinned link.

    `git_sha` is the field the function's own docstring records as having
    already shipped unchecked once, pooling records from different torch builds
    into one finding with no refusal anywhere. Driven through `main --phase
    read`, so the refusal is pinned at the ENTRY POINT a run uses rather than at
    a function a test chose to call.

    AND NOTHING IS WRITTEN: the refusal must come before `capacity.txt`, or a
    pool that cannot be read leaves an artefact behind that says it was."""
    records = _pool()
    victim = sorted(records)[-1]
    assert victim == ("random_vit", 2)
    records[victim]["git_sha"] = records[victim]["git_sha"] + "-dirty"
    _write_pool(tmp_path, records)

    with pytest.raises(SystemExit) as raised:
        _main_read("--phase", "read", "--out", str(tmp_path))
    message = str(raised.value)
    assert "random_vit seed 2" in message, message
    assert "git_sha" in message, message
    assert not (tmp_path / "capacity.txt").exists(), (
        "a pool that cannot be read must leave no artefact claiming it was"
    )
    assert "verdict:" not in capsys.readouterr().out


def test_read_phase_reads_records_written_before_resamples_was_recorded(
    tmp_path, capsys,
):
    """THE NINE RECORDS ON DISK PREDATE `capacity.resamples`, and the run's
    reproducibility rests on their still being readable: a field added to the
    measure half must not make the shipped artefact unreadable by the half that
    reads it.

    Driven as a BYTE COMPARISON of the two readings rather than as a bare
    "does not raise": the field is not merely tolerated, it changes nothing the
    reading prints. An `ARM_FIELDS` or `_self_check_table` that grew a
    `_get(record, cell, "capacity", "resamples")` would refuse the stripped pool
    by name through `_get` and fail here; one that printed it would pass a
    does-not-raise test and fail this one."""
    records = _pool()
    with_field, out_with = _read(tmp_path / "with", capsys, records)
    for record in records.values():
        del record["capacity"]["resamples"]
    assert all("resamples" not in r["capacity"] for r in records.values())
    without_field, out_without = _read(tmp_path / "without", capsys, records)

    assert (with_field, without_field) == (script.EXIT_OK, script.EXIT_OK)
    assert out_without == out_with, (
        "the reading changed when `capacity.resamples` was removed; the nine "
        "shipped records do not carry it and must read identically"
    )
    assert (tmp_path / "without" / "capacity.txt").read_bytes() == (
        tmp_path / "with" / "capacity.txt"
    ).read_bytes()


def test_read_phase_refuses_a_plan_it_cannot_read_before_loading_anything(tmp_path):
    """`--arms frozen_ssl` cannot be read at all, and the refusal must come
    before the records are even looked for."""
    with pytest.raises(SystemExit) as raised:
        _main_read("--phase", "read", "--out", str(tmp_path), "--arms", "frozen_ssl")
    assert str(ARMS_REQUIRED) in str(raised.value)
    with pytest.raises(SystemExit) as raised:
        _main_read("--phase", "read", "--out", str(tmp_path), "--seeds", "0")
    assert str(SEEDS_REQUIRED) in str(raised.value)


def test_read_phase_counts_distinct_arms_and_seeds_not_the_lists(tmp_path, capsys):
    """`--arms a a a` is ONE arm and `--seeds 0 0` one seed: counting the lists
    would let a plan through that the records dict, keyed by cell, cannot
    honour."""
    with pytest.raises(SystemExit):
        _main_read(
            "--phase", "read", "--out", str(tmp_path),
            "--arms", "pixel_ae", "pixel_ae", "pixel_ae",
        )
    records = _pool()
    _write_pool(tmp_path, records)
    code = _main_read(
        "--phase", "read", "--out", str(tmp_path),
        "--arms", *ARMS, *ARMS, "--seeds", "0", "1", "2", "0",
    )
    out = capsys.readouterr().out
    assert code == script.EXIT_OK
    assert len(_section(out, "per cell:").splitlines()) == 11


def test_the_default_phase_is_read_and_the_parser_names_the_three():
    """A default of `measure` would make a bare invocation pay for the whole
    gather and print no reading."""
    args = script._parser().parse_args([])
    assert args.phase == "read"
    assert script.PHASES == ("all", "measure", "read")
    assert args.out == Path("runs/m3l_capacity")
    assert args.source == Path("runs/m3_study_v2")


def test_main_exits_zero_having_read_a_reading(tmp_path, capsys):
    _write_pool(tmp_path, _pool())
    assert _main_read("--phase", "read", "--out", str(tmp_path)) == script.EXIT_OK
    assert "verdict:" in capsys.readouterr().out


def test_every_phase_read_test_routes_through_the_measure_guard():
    """`_main_read` is what keeps the measure half unreachable, and until this
    test nothing made using it MECHANICAL -- a future test calling
    `script.main(["--phase", "read", ...])` directly would load the real
    checkpoints from `--source`'s default and run a real cell on the GPU inside
    pytest."""
    # Built at runtime so this scan's own source line cannot match itself.
    call = "script" + ".main("
    phase = '"' + "read" + '"'
    offenders = [
        line.strip() for line in Path(__file__).read_text().splitlines()
        if call in line and phase in line
        and not line.lstrip().startswith(("`", "#", '"'))
    ]
    assert not offenders, (
        "call `_main_read(...)` instead of `script.main([...])` so the measure "
        f"half stays unreachable: {offenders}"
    )


def _main_phase(monkeypatch, phase, out, *, measure_status=None):
    """`main` on `phase` with the measure half stubbed out.

    Returns `(status, phases_run)`. The measure half is the expensive one and the
    read half is the one under test, so what this pins is the WIRING: which half
    runs, in which order, and whether a failed measure stops the read."""
    ran = []

    def fake_measure_phase(args, cells, device, train, val):
        ran.append("measure")
        return script.EXIT_OK if measure_status is None else measure_status

    def fake_read_phase(args):
        ran.append("read")
        return script.EXIT_OK

    monkeypatch.setattr(script, "measure_phase", fake_measure_phase)
    monkeypatch.setattr(script, "read_phase", fake_read_phase)
    monkeypatch.setattr(script, "get_device", lambda prefer=None: "cpu")
    monkeypatch.setattr(
        script, "ReplayBuffer",
        lambda data, **kw: types.SimpleNamespace(episode_paths=lambda: []),
    )
    monkeypatch.setattr(script, "episode_split", lambda paths, **kw: ([], []))
    status = script.main(["--phase", phase, "--out", str(out)])
    return status, ran


@pytest.mark.parametrize("phase, expected", [
    ("measure", ["measure"]),
    ("read", ["read"]),
    ("all", ["measure", "read"]),
])
def test_main_runs_exactly_the_halves_its_phase_names(
    monkeypatch, tmp_path, phase, expected,
):
    """`--phase measure` must not read (there may be nothing to read yet) and
    `--phase read` must not measure (that is ~13.5 GPU-hours)."""
    status, ran = _main_phase(monkeypatch, phase, tmp_path)
    assert (status, ran) == (script.EXIT_OK, expected)


def test_a_failed_measure_stops_before_the_read_and_returns_its_own_status(
    monkeypatch, tmp_path,
):
    """A read over records the measure refused to finish writing would pool a
    partial grid -- or, worse, the PREVIOUS run's records -- under this run's
    verdict."""
    status, ran = _main_phase(
        monkeypatch, "all", tmp_path, measure_status=script.EXIT_SELF_CHECK_FAILED,
    )
    assert status == script.EXIT_SELF_CHECK_FAILED
    assert ran == ["measure"], "the read phase ran after a refused measure"


def test_the_artefact_is_written_before_anything_is_printed(
    monkeypatch, tmp_path, capsys,
):
    """`capacity.txt` GATES THE LOG, which is a claim `read_phase`'s docstring
    makes and nothing else could falsify. A read whose artefact could not be
    written -- a full disk, a read-only output directory -- must not first print
    a verdict a reader would then go looking for in a file that is not there."""
    _write_pool(tmp_path, _pool())

    def refuse(path, text):
        raise OSError("no space left on device")

    monkeypatch.setattr(script, "write_text", refuse)
    with pytest.raises(OSError, match="no space left"):
        _main_read("--phase", "read", "--out", str(tmp_path))
    assert capsys.readouterr().out == "", (
        "the verdict reached stdout before the artefact reached disk"
    )


def test_a_record_the_measure_half_really_built_is_readable_by_the_read_half(
    monkeypatch, tmp_path, capsys,
):
    """THE BRIDGE BETWEEN THE TWO FIXTURES, and the last guard before the run.

    `_record` (the measure half's fixture) and `_read_record` (the read half's)
    are independent hand-built dicts. Every other test on either side uses one
    of them, so a schema divergence between what `measure_cell` WRITES and what
    `capacity_inputs` READS would be invisible until the nine-cell run had
    already spent its GPU hours -- and the failure would surface at the read,
    with the measure's artefact already on disk and no way to tell whether the
    records or the reader were wrong.

    So this drives a record built by the REAL `measure_cell` -- real
    `cell_capacity`, `estimator_checks`, `base_control`, `frame_probe`, real
    `write_record` -- through the REAL `capacity_inputs`, `reading_capacity` and
    `capacity_text`. Only `prepare_cell`, the reference pass and
    `gather_probe_data` are stubbed, exactly as the measure tests stub them.

    It asserts the artefact is COMPLETE, not merely that nothing raised: a
    reader that silently dropped a section would otherwise pass."""
    _stub_measure(monkeypatch)
    train, val = _paths("t", 45), _paths("v", 24)
    status, built = script.measure_cell(
        _measure_args(), _cell(arm="pixel_ae", seed=0), "cpu", train, val,
    )
    assert status == script.EXIT_OK, "the premise: the measure half produced a record"

    # Nine cells from that one real record, relabelled -- the read path's own
    # per-cell distinctness is pinned elsewhere; what is under test here is the
    # SCHEMA agreeing across the two halves.
    for arm in ARMS:
        for seed in (0, 1, 2):
            record = dict(built, arm=arm, seed=seed, git_sha="bridge")
            script.write_record(script.capacity_record_path(tmp_path, arm, seed), record)

    exit_code = _main_read("--phase", "read", "--out", str(tmp_path))
    out = capsys.readouterr().out

    assert exit_code in (script.EXIT_OK, script.EXIT_ESTIMATOR_BROKEN,
                         script.EXIT_BASE_UNRESOLVED), f"unexpected exit {exit_code}"
    for section in ("self-check per record", "Reading G", "base control",
                    "estimator checks", "verdict:"):
        assert section in out, f"the artefact is missing its {section!r} section"
    # Scoped to the Reading G section: the arm name also heads one self-check
    # row per cell, so an unscoped search finds seven lines and proves nothing
    # about the reading's own table.
    reading = out.split("Reading G", 1)[1]
    for arm in ARMS:
        rows = [ln for ln in reading.splitlines() if ln.split()[:1] == [arm]]
        assert len(rows) == 1, f"expected one Reading G row for {arm}, got {len(rows)}"
        # Every number the reading needs survived the round trip as a number.
        from mbfps.eval.capacity import READING_COLUMNS

        assert len(rows[0].split()) == len(READING_COLUMNS), rows[0]
    written = (tmp_path / "capacity.txt").read_bytes()
    assert written == out.encode(), "the artefact and stdout diverged"
