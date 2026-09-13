"""Provenance: `git_sha`, `device`, `encoder_params` and the full `history` (M3c).

Split out of `test_study.py` by concern. The M3b write-up had to RECONSTRUCT
three things after the fact: which code state produced the nine cells (from
file mtimes), where each cell ran (from the log), and whether the pixel arm's
KL ever cleared the free-bits floor (from a scalar rate, because the per-step
history was thrown away). Each is now a field, and every guard here pins the
field to a value the test computes INDEPENDENTLY -- the test's own `git
rev-parse`, its own `numel` sum, the history it handed `train_world_model` --
because "is a string" and "is a list" are satisfied by a hardcoded literal.
The shared job fixture and its pairwise-distinct parameters are
`test_study.py`'s and are imported from it.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from mbfps.eval import study
from mbfps.eval.study import (
    NONFINITE_KEY,
    StudyJob,
    job_record_path,
    load_record,
    run_job,
    to_json_record,
    write_record,
)
from tests.eval.test_study import (  # noqa: F401 -- `record` is a fixture
    JOB,
    JOB_KW,
    JOB_SEED,
    PAIRWISE_DISTINCT_PARAMETERS,
    record,
    strict_loads,
)


#: What the record must hold under `history`, and the five loss terms every
#: `parts[i]` carries, spelled out as literals so a term dropped from
#: `WorldModel.forward`'s dict fails here by name.
HISTORY_KEYS = frozenset({"loss", "parts"})
PART_KEYS = frozenset({"embedding", "reward", "continue", "kl_dyn", "kl_rep"})

#: Spec 2.2's one stated asymmetry, per arm: `pixel_ae`'s bottleneck is
#: `Linear(32 -> 32)` = 32*32 + 32; the ViT arms' is `Linear(384 -> 32)` =
#: 384*32 + 32. A LITERAL, not a computation, so that a geometry change that
#: silently alters the bottleneck width shows up as a number, not as two
#: computations agreeing with each other.
EXPECTED_ENCODER_PARAMS = {
    "pixel_ae": 1_056,
    "frozen_ssl": 12_320,
    "random_vit": 12_320,
}


def _git_head(cwd: Path) -> str | None:
    """`git rev-parse HEAD` at `cwd`, run by the TEST, or None if git cannot."""
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd,
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def test_the_record_carries_the_sha_git_reports_for_the_code_that_ran(record):
    """`git_sha` must equal what `git rev-parse HEAD` says HERE, in this
    checkout -- the test runs git itself and compares. "Is a 40-character
    string" is satisfied by a literal pasted into `run_job`, and a literal is
    precisely the thing that would make nine records from two code states
    look like one.

    The sha is asked of the directory `study.py` lives in, not the process's
    cwd; the two are the same repository here, and this test pins the value,
    not the cwd (that is `test_git_sha_is_asked_of_the_code_not_the_cwd`).

    PUBLIC: `git_sha` is imported by `scripts/spike_pixel_ae.py` too, so the
    spike's record carries the same sha under the same semantics; the spike's
    own test pins its value the same way.
    """
    repo_root = Path(__file__).resolve().parents[2]
    expected = _git_head(repo_root)
    if expected is None:
        pytest.skip("git cannot report HEAD for this checkout, so the value "
                    "cannot be pinned here; the failure path is tested below")
    assert len(expected) == 40 and set(expected) <= set("0123456789abcdef"), (
        f"the test's own git call returned {expected!r}, not a sha; the "
        "comparison below would be against garbage")
    assert record["git_sha"] == expected
    assert isinstance(record["git_sha"], str)
    assert record["git_sha"] != study.UNKNOWN_GIT_SHA


def test_git_sha_is_asked_of_the_code_not_the_cwd(monkeypatch):
    """`cwd=` must be `study.py`'s own directory. A study launched from
    `/scratch` must report the code's commit, not `/scratch`'s (or fail
    because `/scratch` is not a repository)."""
    import subprocess

    seen: dict = {}
    real_run = subprocess.run

    def spy(args, **kwargs):
        seen["args"] = list(args)
        seen["cwd"] = kwargs.get("cwd")
        return real_run(args, **kwargs)

    monkeypatch.setattr(study.subprocess, "run", spy)
    study.git_sha()
    assert seen["args"] == ["git", "rev-parse", "HEAD"]
    assert Path(seen["cwd"]).resolve() == Path(study.__file__).resolve().parent, (
        f"git was asked at {seen['cwd']!r}, not where the code lives")


@pytest.mark.parametrize("failure", [
    "no_git_executable", "git_hangs", "not_a_repository", "empty_stdout",
])
def test_git_sha_is_unknown_rather_than_an_exception_when_git_cannot_answer(
    monkeypatch, failure
):
    """A rented box without git must not lose a cell to a provenance field.

    Four ways the subprocess can fail to produce a sha, EACH EXERCISED ALONE
    so that no single guard covers for another: the executable is missing
    (`OSError`), it hangs (`TimeoutExpired`, a `SubprocessError`), it runs but
    the directory is not a repository (non-zero status, and git's usual
    empty stdout), and the degenerate zero-status-empty-stdout -- which a
    guard on the status alone would pass through as `""`.

    `not_a_repository` deliberately puts a non-sha ON stdout with the
    non-zero status, so the returncode guard is what has to catch it; the
    `or UNKNOWN` fallback alone would let "abc" through.
    """
    import subprocess
    from types import SimpleNamespace

    def fake_run(args, **kwargs):
        if failure == "no_git_executable":
            raise FileNotFoundError(2, "No such file or directory", "git")
        if failure == "git_hangs":
            raise subprocess.TimeoutExpired(args, kwargs.get("timeout", 0))
        if failure == "not_a_repository":
            return SimpleNamespace(returncode=128, stdout="abc\n", stderr="fatal")
        return SimpleNamespace(returncode=0, stdout="\n", stderr="")

    monkeypatch.setattr(study.subprocess, "run", fake_run)
    assert study.git_sha() == study.UNKNOWN_GIT_SHA == "unknown"


def test_the_record_says_where_the_cell_ran_not_what_was_asked_for(
    tmp_path, small_buffer, monkeypatch
):
    """`device` is `str(torch_device)` -- the device `get_device` RESOLVED --
    not the `device` string the job was launched with. On a box without CUDA,
    `--device cuda` runs on the CPU, and a record that says "cuda" is the lie
    the field exists to prevent.

    `get_device` is pinned to CPU here so the guard is machine-independent:
    without the pin, `get_device(prefer="cuda")` already IS cpu on this Mac
    and would be cuda on the box, so `"device": device` would pass on the
    box for the wrong reason. With it, the request ("cuda") and the answer
    ("cpu") differ everywhere."""
    import torch

    requested = "cuda"
    resolved = torch.device("cpu")
    seen: dict = {}

    def pinned_get_device(prefer="mps"):
        seen["prefer"] = prefer
        return resolved

    monkeypatch.setattr(study, "get_device", pinned_get_device)
    result = run_job(JOB, small_buffer, tmp_path,
                     **dict(JOB_KW, device=requested))

    assert seen["prefer"] == requested, "run_job did not ask for the job's device"
    assert str(resolved) != requested, (
        "the pinned device spells the same as the request, so the assertion "
        "below cannot tell the resolved device from the requested string")
    assert result["device"] == str(resolved) == "cpu"
    assert isinstance(result["device"], str)
    assert result["device"] != requested


@pytest.mark.parametrize("arm", sorted(EXPECTED_ENCODER_PARAMS))
def test_encoder_params_counts_the_trainable_encoder_parameters_of_each_arm(
    tmp_path, small_buffer, arm
):
    """`encoder_params` is the sum of `numel` over the ENCODER's trainable
    parameters, computed here independently of `run_job` -- and equal to the
    spec's own stated number for the arm.

    Run for every arm because the value is the one place the arms differ and
    a literal (12,320) would pass on two of the three. The whole-model count
    is asserted different so that `model.parameters()` in place of
    `model.encoder.parameters()` is a visible mutation and not a coincidence.
    """
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import ARMS, get_config

    assert set(EXPECTED_ENCODER_PARAMS) == set(ARMS), (
        "the expected counts must name exactly the study's arms; an arm "
        "without a stated count is an arm whose asymmetry is unrecorded")

    job = StudyJob(arm, JOB_SEED)
    result = run_job(job, small_buffer, tmp_path, **JOB_KW)

    model = WorldModel(get_config(arm, device="cpu", seed=JOB_SEED))
    expected = sum(p.numel() for p in model.encoder.parameters()
                   if p.requires_grad)
    whole_model = sum(p.numel() for p in model.parameters())
    assert expected != whole_model, (
        "the encoder is the whole model, so counting everything would pass")
    assert result["encoder_params"] == expected == EXPECTED_ENCODER_PARAMS[arm]
    assert isinstance(result["encoder_params"], int)
    assert not isinstance(result["encoder_params"], bool)


def test_encoder_params_excludes_frozen_parameters():
    """The `requires_grad` filter, alone. On every study arm all encoder
    parameters train, so `run_job` cannot show the filter; a module with one
    frozen tensor can. Without it a future encoder wrapping a frozen backbone
    would report the backbone's millions as "trained"."""
    import torch

    module = torch.nn.Module()
    module.trained = torch.nn.Parameter(torch.zeros(3, 4))          # 12
    module.frozen = torch.nn.Parameter(torch.zeros(100), requires_grad=False)
    module.register_buffer("stat", torch.zeros(1000))               # never
    assert study._trainable_parameters(module) == 12


def test_the_record_carries_the_full_history_at_the_jobs_own_length(record):
    """One `loss` and one five-term `parts` entry PER STEP, for every step
    the job trained -- `steps` entries, not the last twenty and not a mean.
    The M3b write-up's section 1 had to reconstruct the pixel arm's KL
    trajectory because this list was discarded."""
    history = record["history"]
    assert set(history) == HISTORY_KEYS
    assert isinstance(history["loss"], list)
    assert isinstance(history["parts"], list)
    assert len(history["loss"]) == JOB_KW["steps"]
    assert len(history["parts"]) == JOB_KW["steps"]
    # The length check alone cannot see a `[-20:]` slice over `study.py`: at
    # JOB_KW["steps"] == 5 (as at 20) the slice is the whole list. The slice
    # mutation is caught by exact equality against a 23-step history in
    # test_the_history_is_the_one_training_returned_and_not_a_summary_of_it.
    for value in history["loss"]:
        assert isinstance(value, float)
    for part in history["parts"]:
        assert set(part) == PART_KEYS, part
        assert all(isinstance(v, float) for v in part.values())
    # The scalar convergence number is derived from the SAME curve.
    assert record["loss_last20"] == pytest.approx(
        float(np.mean(history["loss"][-20:])))


def test_the_history_is_the_one_training_returned_and_not_a_summary_of_it(
    tmp_path, small_buffer, monkeypatch
):
    """Exact equality against a history the test built: 23 distinct losses
    and 23 distinct `parts` rows, longer than 20 (so a `[-20:]` slice
    shortens it), not equal to `steps` (so a `[:steps]` slice shortens it),
    with a NaN in one row so the non-finite policy is exercised on a value
    FOUR path segments deep -- `history.parts.<k>.kl_dyn` -- through the
    file and back."""
    real_train = study.train_world_model
    n = 23
    assert n not in PAIRWISE_DISTINCT_PARAMETERS.values() and n > 20
    losses = [100.0 + i for i in range(n)]
    nan_row = 7
    parts = [
        {"embedding": 1.0 + i, "reward": 2.0 + i, "continue": 3.0 + i,
         "kl_dyn": float("nan") if i == nan_row else 4.0 + i,
         "kl_rep": 5.0 + i}
        for i in range(n)
    ]

    def doctored(cfg, buffer, out_dir, log_every=100):
        history = real_train(cfg, buffer, out_dir=out_dir, log_every=log_every)
        history["loss"] = list(losses)
        history["parts"] = [dict(p) for p in parts]
        return history

    monkeypatch.setattr(study, "train_world_model", doctored)
    result = run_job(JOB, small_buffer, tmp_path, **JOB_KW)

    assert result["history"]["loss"] == losses
    assert len(result["history"]["parts"]) == n
    for i, (got, want) in enumerate(zip(result["history"]["parts"], parts)):
        assert set(got) == set(want) == PART_KEYS
        for key in PART_KEYS:
            if i == nan_row and key == "kl_dyn":
                assert math.isnan(got[key])
            else:
                assert got[key] == want[key]

    # Through the file and back: the NaN is a `null` plus a four-segment
    # dotted path, and `load_record` restores a float NaN, not 0.0 or None.
    written = to_json_record(result)
    dotted = f"history.parts.{nan_row}.kl_dyn"
    assert written["history"]["parts"][nan_row]["kl_dyn"] is None
    assert written[NONFINITE_KEY][dotted] == "nan"
    restored = load_record(job_record_path(tmp_path, JOB))
    assert math.isnan(restored["history"]["parts"][nan_row]["kl_dyn"])
    assert restored["history"]["parts"][nan_row]["kl_rep"] == 5.0 + nan_row
    assert restored["history"]["loss"] == losses


def test_history_record_keeps_every_step_coerces_each_value_and_lets_nan_through():
    """The policy at unit scale, on the helper both writers call. 23 distinct
    steps (longer than 20, not equal to any fixture's `steps`), numpy scalars
    in the input, a non-string key, and one NaN: the output is the same
    length, every leaf a plain `float`, every key a `str`, and the NaN still
    a NaN -- not dropped, not zeroed, not `None` (that is `write_record`'s
    job, downstream)."""
    n = 23
    history = {
        "loss": [np.float32(100.0 + i) for i in range(n)],
        "parts": [
            {"embedding": np.float64(1.0 + i), "reward": 2.0 + i,
             "continue": np.float32(3.0 + i),
             "kl_dyn": float("nan") if i == 7 else 4.0 + i, "kl_rep": 5.0 + i}
            for i in range(n)
        ],
        "steps": n, "seconds": 1.0,          # ignored: not part of the block
    }
    block = study.history_record(history)
    assert set(block) == HISTORY_KEYS
    assert block["loss"] == [100.0 + i for i in range(n)]
    assert all(type(v) is float for v in block["loss"])
    assert len(block["parts"]) == n
    for i, part in enumerate(block["parts"]):
        assert set(part) == PART_KEYS
        assert all(type(k) is str for k in part)
        assert all(type(v) is float for v in part.values())
        if i == 7:
            assert math.isnan(part["kl_dyn"])
        else:
            assert part["kl_dyn"] == 4.0 + i
        assert part["kl_rep"] == 5.0 + i
    # The input is left alone: the writer, not the helper, owns the copy.
    assert isinstance(history["loss"][0], np.float32)
    assert math.isnan(history["parts"][7]["kl_dyn"])


def test_a_nan_inside_parts_round_trips_through_the_nonfinite_map(tmp_path):
    """The dotted-path scheme at unit scale, on the record's new shape:
    `parts` is a LIST of dicts, so the path carries an integer segment
    (`history.parts.1.kl_dyn`), and `load_record` must index the list rather
    than look the string "1" up in a dict."""
    original = {
        "history": {
            "loss": [0.5, float("inf")],
            "parts": [{"kl_dyn": 0.1}, {"kl_dyn": float("nan"), "kl_rep": 0.2}],
        },
    }
    written = write_record(tmp_path / "r.json", original)
    assert written["history"]["parts"][1]["kl_dyn"] is None
    assert written["history"]["parts"][1]["kl_rep"] == 0.2
    assert written["history"]["loss"][1] is None
    assert written[NONFINITE_KEY] == {
        "history.loss.1": "inf",
        "history.parts.1.kl_dyn": "nan",
    }
    text = (tmp_path / "r.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert strict_loads(text) == written

    back = load_record(tmp_path / "r.json")
    assert math.isnan(back["history"]["parts"][1]["kl_dyn"])
    assert back["history"]["parts"][1]["kl_rep"] == 0.2
    assert back["history"]["parts"][0]["kl_dyn"] == 0.1
    assert back["history"]["loss"][1] == float("inf")
    # The caller's dict still holds the real NaN; only the copy is lossy.
    assert math.isnan(original["history"]["parts"][1]["kl_dyn"])
