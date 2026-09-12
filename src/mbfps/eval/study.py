"""One cell of the 3 arms x 3 seeds study.

A job trains one arm at one seed, evaluates it, and emits a JSON-serialisable
record. The driver in `scripts/run_study.py` is resumable off these records, so
a job that has already produced one is never re-run -- which matters when the
pixel arm costs 8.3 h per seed against the feature arms' 1.4 h.

WHAT THIS MODULE DOES NOT DO: recompute anything. The band statistics and the
degeneracy threshold come from `mbfps.eval.summary`, which
`scripts/eval_rollout.py` also consumes, so a single run and the nine-cell
study can never disagree about what "degenerate" means. Re-implementing
`metric_summary` here would give DEGENERATE two homes and let them drift.

NON-FINITE NUMBERS ARE NORMAL HERE, AND `json.dump` CANNOT WRITE THEM. Both
`gap_final`/`gap_mean` (NaN by `metric_summary`'s contract whenever the band is
non-positive) and `reward["r2"]` (NaN when the target has no variance) are
undefined by design rather than by accident. Python writes those as the bare
token `NaN`, which is not JSON and which strict parsers reject -- and the
aggregation step reads nine of these files back. See `write_record` for the
policy and `load_record` for the inverse.
"""

import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.probe import filtering_gain, filtering_report, fit_probes
from mbfps.eval.rollout import evaluate_rollout
from mbfps.eval.summary import METRICS, metric_summary
from mbfps.models.encoders import encoder_backbone
from mbfps.training.world_model import WorldModel, train_world_model
from mbfps.utils.config import get_config
from mbfps.utils.device import get_device

SPLIT_SEED = 0
"""The episode split seed, fixed and deliberately NOT the job's seed.

`train_world_model` splits at `seed=0`. Evaluating at `job.seed` instead would
hand seed 1 a different held-out set from the one its own training run held
out -- i.e. it would evaluate on episodes it trained on -- and the nine cells
would no longer be scored on the same episodes. The record carries the episode
names so that this is checkable after the fact rather than assumed.
"""

NONFINITE_KEY = "nonfinite"
"""Top-level key of the record's map from field path to non-finite token."""

_TOKENS: dict[str, float] = {
    "nan": float("nan"),
    "inf": float("inf"),
    "-inf": float("-inf"),
}


@dataclass(frozen=True)
class StudyJob:
    """One cell of the study: one arm at one seed."""

    arm: str
    seed: int


def job_record_path(out_dir: Path, job: StudyJob) -> Path:
    """Where this job's record lives.

    BOTH the arm and the seed are in the name. Either one missing collides two
    of the nine cells onto one file, and because the driver resumes off these
    files the second cell would not be re-run -- it would be silently reported
    as the first one's numbers.
    """
    return Path(out_dir) / f"result_{job.arm}_seed{job.seed}.json"


DEGENERATE_REWARD_FRACTION = 0.99
"""If this share of steps carry the modal reward, the target is degenerate.

Measured on `my_way_home`: 19,417 of 19,424 steps share the living penalty and
six carry the goal, so the modal share is 0.9996. A scalar accuracy over that
is dominated by the constant.
"""


def _summarise_reward(predicted: np.ndarray, true: np.ndarray) -> dict:
    """Reward error beside the baseline that makes it readable.

    `r2` is NaN when the target has no variance -- reporting 0.0 there would
    read as "explains nothing" when the truth is "there was nothing to
    explain". The NaN survives to disk; see `write_record`.
    """
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if true.size == 0:
        raise ValueError("no reward steps to summarise")
    if predicted.shape != true.shape:
        raise ValueError(
            f"predicted {predicted.shape} and true {true.shape} rewards are not "
            "aligned; a shape mismatch here would broadcast into a meaningless MSE"
        )
    denominator = float(((true - true.mean()) ** 2).sum())
    mse = float(((predicted - true) ** 2).mean())
    baseline = float(((true - true.mean()) ** 2).mean())
    rounded = np.round(true, 6)
    modal_share = float(
        np.bincount(np.unique(rounded, return_inverse=True)[1]).max() / true.size
    )
    return {
        "mse": mse,
        "baseline_mse": baseline,
        "r2": (
            float("nan")
            if denominator == 0.0
            else 1.0 - (mse * true.size) / denominator
        ),
        "n_steps": int(true.size),
        "n_reward_events": int((rounded != np.round(np.median(true), 6)).sum()),
        "is_degenerate": bool(modal_share >= DEGENERATE_REWARD_FRACTION),
    }


@torch.no_grad()
def reward_accuracy(
    model, val_paths, backbone, device, limit: int = 20, seed: int = 0
) -> dict:
    """Spec section 4, criterion 3: reward-prediction accuracy per arm.

    Filters each held-out episode and compares the reward head's output against
    the recorded reward. Reported with its baseline and event count because
    `my_way_home`'s reward is near-constant -- see DEGENERATE_REWARD_FRACTION.
    A bare MSE over that target looks precise and means nothing.

    Seeds the global RNG: `observe` samples from the posterior, and
    reproducibility here comes from the seed, never from taking the mode.
    """
    from mbfps.data.episode import load_episode
    from mbfps.eval.rollout import source_for

    torch.manual_seed(seed)
    predicted, true = [], []
    for path in list(val_paths)[:limit]:
        episode = load_episode(path)
        source = source_for(model, path, episode, backbone)
        # Drop the FIRST frame: `embeddings[k]` must be the frame `actions[k]`
        # led to, per RSSM.observe's action-time convention.
        embeddings = model.encoder(
            torch.as_tensor(source).to(device)
        ).unsqueeze(0)[:, 1:]
        actions = (
            torch.as_tensor(episode.actions.astype(np.int64)).unsqueeze(0).to(device)
        )
        out = model.rssm.observe(embeddings, actions)
        predicted.append(
            model.heads(out["latent"])["reward"][0].float().cpu().numpy()
        )
        true.append(episode.rewards)
    if not predicted:
        raise ValueError("no validation episode produced reward predictions")
    return _summarise_reward(np.concatenate(predicted), np.concatenate(true))


def _token(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0.0 else "-inf"


def _sanitise(value: Any, path: str, found: dict[str, str]) -> Any:
    """Recursively make `value` strict-JSON-writable, recording what was lost.

    numpy scalars become Python scalars; non-finite floats become `null` and
    their dotted path is recorded in `found`.
    """
    if isinstance(value, dict):
        clean = {}
        for k, v in value.items():
            key = str(k)
            if "." in key:
                where = path or "<root>"
                raise ValueError(
                    f"record key {key!r} under {where} contains a '.'; the non-finite "
                    "map addresses fields by dotted path, so this key would alias a "
                    "real nested field and load_record would write a NaN over it"
                )
            clean[key] = _sanitise(v, f"{path}.{key}" if path else key, found)
        return clean
    if isinstance(value, np.ndarray) and value.ndim == 0:
        # A 0-d array is a scalar, not a sequence: the list branch below raises
        # `iteration over a 0-d array` on it, which would be a write-time crash
        # after the GPU hours rather than a value coerced.
        value = value[()]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [
            _sanitise(v, f"{path}.{i}" if path else str(i), found)
            for i, v in enumerate(value)
        ]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isfinite(number):
            return number
        found[path] = _token(number)
        return None
    return value


def to_json_record(record: dict) -> dict:
    """The strict-JSON form of `record`, plus the map that inverts the loss.

    THE POLICY, and why it is not "replace NaN with 0.0". A NaN `gap_final`
    means the persistence-to-floor band was non-positive, so the fraction of it
    the model closed is undefined; 0.0 means the model closed none of a real
    band. Those are different findings about an arm, and the aggregation over
    nine records averages them -- coercing the first into the second would pull
    the mean toward "no better than persistence" using cells that measured
    nothing at all.

    So: a non-finite float is written as `null`, which every strict parser
    accepts and which no consumer can mistake for a number, and its dotted path
    is recorded under `NONFINITE_KEY` with the token it came from. `null` alone
    would lose NaN-versus-infinity; the map keeps it, and `load_record`
    restores the exact float.
    """
    if NONFINITE_KEY in record:
        raise ValueError(
            f"record already carries a top-level {NONFINITE_KEY!r} key; the map is "
            "written under that name and would silently discard it"
        )
    found: dict[str, str] = {}
    clean = _sanitise(record, "", found)
    clean[NONFINITE_KEY] = found
    return clean


def write_record(path: Path, record: dict) -> dict:
    """Write `record` as strict JSON and return exactly what was written.

    `allow_nan=False` is the belt to `to_json_record`'s braces: if any
    non-finite value ever escapes the sanitiser, this raises instead of writing
    a bare `NaN` token that the aggregation step would only discover nine runs
    and thirty GPU-hours later.

    THE WRITE IS ATOMIC, and that is a requirement of the driver rather than a
    nicety. `scripts/run_study.py` decides whether an 8.3-hour cell has already
    been paid for by reading this file, so the file must only ever exist in two
    states: absent, or a complete record. A plain `write_text` truncates the
    destination first and then streams; a process killed in between -- the
    overnight run's laptop lid, an OOM kill, a Ctrl-C landing inside the write
    -- leaves a prefix at the real path. Most prefixes fail to parse and the
    driver would re-run that cell, which is merely expensive; the dangerous one
    is a prefix that happens to parse and re-runs nothing. So the bytes go to a
    temporary file beside the destination and `os.replace` swaps it in, which
    is atomic within a directory: a reader sees the old record or the new one,
    never half of either. The temporary is removed on any failure so a crashed
    write leaves no litter for the aggregation's glob to find.

    This defends against the PROCESS dying, not the machine. Surviving power
    loss would need an `fsync` of the file and of the directory; the study runs
    on a rented box we do not expect to lose mid-write, and the cost of being
    wrong about that is one re-run cell, caught by the driver's completeness
    check.
    """
    clean = to_json_record(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(clean, indent=2, allow_nan=False)
    # The pid keeps two processes writing the same cell from sharing one
    # temporary; the leading dot and the suffix keep it out of a `*.json` glob.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return clean


def load_record(path: Path) -> dict:
    """Read a record back, restoring the non-finite floats `null` stands for.

    The inverse of `write_record`. Consumers that would rather see `None` can
    use `json.loads` directly; this is for the aggregation, which needs NaN
    back so `np.nanmean` can skip the undefined cells instead of counting the
    `None`s as zeros.
    """
    record = json.loads(Path(path).read_text())
    for dotted, token in record.get(NONFINITE_KEY, {}).items():
        if token not in _TOKENS:
            raise ValueError(f"unknown non-finite token {token!r} at {dotted!r}")
        segments = dotted.split(".")
        container = record
        for segment in segments[:-1]:
            container = (
                container[int(segment)]
                if isinstance(container, list)
                else container[segment]
            )
        last = segments[-1]
        if isinstance(container, list):
            container[int(last)] = _TOKENS[token]
        else:
            container[last] = _TOKENS[token]
    return record


UNKNOWN_GIT_SHA = "unknown"
"""What `_git_sha` reports when git cannot answer. Never an exception."""


def _git_sha() -> str:
    """The commit the code that is running was checked out from, or "unknown".

    Asked of the directory THIS FILE lives in, not of the process's working
    directory. The record exists to say "one code state produced all nine
    cells", and the code state is wherever `mbfps` was imported from -- the
    editable install on the laptop, or a checkout on the rented box. A run
    launched from somewhere else (`cd /scratch && python .../run_study.py`)
    must not report the sha of /scratch.

    NEVER RAISES. A rented box without git installed, a checkout unpacked from
    a tarball with no `.git`, a `git` that hangs on a stale lock: every one of
    those is a provenance gap, not a reason to lose a 1.5-hour cell. The
    provenance field says "unknown" and the cell is still written. Only the
    two failures a subprocess call can actually produce are caught -- `OSError`
    (no such executable, permissions) and `subprocess.SubprocessError`
    (`TimeoutExpired`) -- so a programming error in this function is still a
    traceback and not a quiet "unknown" in all nine records.

    The sha alone does not say the tree was clean. That is deliberate: this is
    a label for grouping records by code state, and a `-dirty` suffix would
    make two cells from one uncommitted tree look like two code states.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_GIT_SHA
    if completed.returncode != 0:
        return UNKNOWN_GIT_SHA
    # `or`: a zero status with nothing on stdout is not a sha either. Written
    # as a second guard rather than folded into the one above so each half
    # can be mutation-tested alone.
    return completed.stdout.strip() or UNKNOWN_GIT_SHA


def _trainable_parameters(module: torch.nn.Module) -> int:
    """How many parameters `module` trains: `requires_grad` ones only.

    Every study arm's encoder is a `BottleneckEncoder` whose parameters are all
    trainable, so on the study's own models the `requires_grad` filter is a
    numerical no-op -- which is exactly why it is a separate function with its
    own test: the day an encoder carries a frozen sub-module, `encoder_params`
    must still report what the arm TRAINS, because that is the asymmetry the
    spec records it for (1,056 for `pixel_ae` against 12,320 for the ViT arms).
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _probe_summary(latent_probe: dict, embedding_probe: dict) -> dict:
    """The measuring instrument's own settings.

    A band that looks degenerate because the probe is noise is a different
    finding from one that is degenerate because the model sits on its floor,
    and across nine runs only these numbers tell the two apart.

    `*_ridge_selected` mirrors the gain block's flag and exists because
    `fit_probe` only returns an `r2` when a selection split was used. Without
    it, "selection was skipped" and "the selection R^2 was undefined" are both
    a bare `null` in the file, and an aggregation that reads nine of these
    through `np.array(..., dtype=float)` turns both into NaN.
    """
    return {
        "latent_ridge": float(latent_probe["ridge"]),
        "latent_ridge_selected": "r2" in latent_probe,
        "latent_selection_r2": latent_probe.get("r2"),
        "embedding_ridge": float(embedding_probe["ridge"]),
        "embedding_ridge_selected": "r2" in embedding_probe,
        "embedding_selection_r2": embedding_probe.get("r2"),
    }


def run_job(
    job: StudyJob,
    buffer: ReplayBuffer,
    out_dir: Path,
    steps: int = 20_000,
    seq_len: int = 64,
    context: int = 5,
    horizon: int = 45,
    device: str = "mps",
) -> dict:
    """Train, evaluate, and write one result record. Returns the LIVE record.

    THE RETURNED DICT IS NOT THE FILE'S CONTENT. It carries the real
    non-finite floats; the file carries `null` plus the map that inverts it.
    The difference is load-bearing: `scripts/run_study.py` prints
    `f"...{record['position']['gap_final']:+.4f}"` after every job, and
    `gap_final` is NaN by contract whenever the band is non-positive. A float
    NaN formats as `+nan` and compares False against 0, so the driver survives
    a degenerate cell -- while the sanitised `None` raises
    `unsupported format string passed to NoneType.__format__`, AFTER the record
    is safely on disk, killing the loop and leaving the other eight cells of an
    unattended overnight run unstarted.

    A caller that wants what the aggregation will read should apply
    `to_json_record` to this, or re-read the file with `load_record`. That is
    also how two jobs are compared for equality: NaN is not equal to itself,
    so the comparison has to happen in the sanitised projection.

    `context` and `horizon` are forwarded to the probe fit, the rollout AND
    both filtering diagnostics. They must not be allowed to default apart: the
    probe is applied to latents filtered from a zero state for exactly
    `context` frames, and fitting it at a different depth is a distribution
    mismatch worth ~25 map units of position error.

    FOUR PROVENANCE FIELDS, added for the M3c re-run and each answering a
    question the M3b write-up had to reconstruct after the fact:

      * `git_sha`        -- "one code state produced all nine cells" is a
                            field to compare, not an inference from mtimes;
      * `device`         -- where the cell actually RAN (`str(torch_device)`),
                            which on a box without CUDA is "cpu" however the
                            command line spelled it. The request string is
                            already in the log; the record holds the answer;
      * `encoder_params` -- the one place the arms are not byte-identical:
                            `pixel_ae`'s bottleneck is `Linear(32 -> 32)`,
                            the ViT arms' `Linear(384 -> 32)`. Recorded so it
                            is visible in every record rather than hidden;
      * `history`        -- the FULL per-step `loss` and `parts`, so the next
                            "the pixel arm's KL never cleared the floor" is a
                            measurement read off nine files instead of a
                            reconstruction. ~3 MB per record at 20,000 steps
                            with `indent=2`; the aggregation reads the scalar
                            fields and never touches it, and `_sanitise` maps
                            a non-finite loss at step k to `history.loss.k`
                            like any other nested field.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = get_config(job.arm, steps=steps, seq_len=seq_len, seed=job.seed, device=device)
    torch_device = get_device(prefer=device)

    started = time.perf_counter()
    history = train_world_model(cfg, buffer, out_dir=out_dir)
    train_paths, val_paths = episode_split(
        buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED
    )
    backbone = encoder_backbone(cfg.encoder)

    model = WorldModel(cfg).to(torch_device)
    checkpoint = torch.load(
        out_dir / f"world_model_{job.arm}_seed{job.seed}.pt",
        map_location=torch_device,
        weights_only=True,
    )
    if checkpoint.get("arm") != job.arm or checkpoint.get("seed") != job.seed:
        raise ValueError(
            f"checkpoint in {out_dir} is arm={checkpoint.get('arm')!r} "
            f"seed={checkpoint.get('seed')!r}, not this job's "
            f"arm={job.arm!r} seed={job.seed!r}"
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    # Counted on the EVALUATED model's encoder, after the load: the same
    # object every number below is measured on.
    encoder_params = _trainable_parameters(model.encoder)

    latent_probe, embedding_probe = fit_probes(
        model, train_paths, backbone, torch_device,
        context=context, horizon=horizon, seed=job.seed,
    )
    reward = reward_accuracy(
        model, val_paths, backbone, torch_device, seed=job.seed
    )
    result = evaluate_rollout(
        model, val_paths, embedding_probe,
        context=context, horizon=horizon, seed=job.seed,
        device=torch_device, feature_backbone=backbone,
    )
    # Gate criterion 4 and its bottleneck-free companion. BOTH are recorded,
    # never one instead of the other: criterion 4 puts the 160-bit latent
    # against 2048 encoder floats and can fail on the bottleneck alone, while
    # the gain puts the raw embedding in both arms. And the gain's SIGN is
    # ridge-grid dependent -- measured -0.0706, -0.0208 and +0.0325 under three
    # defensible selection designs, because RIDGES is a decade grid and the
    # scored R^2 moves ~0.10 per decade while the effect is ~0.02. Recording
    # which ridge each arm selected is what keeps that sensitivity visible
    # across nine runs instead of averaging it into an uninterpretable number.
    criterion_4 = filtering_report(
        model, train_paths, val_paths, backbone, torch_device,
        context=context, horizon=horizon, seed=job.seed,
    )
    gain = filtering_gain(
        model, train_paths, val_paths, backbone, torch_device,
        context=context, horizon=horizon, seed=job.seed,
    )

    record = {
        "arm": job.arm,
        "seed": job.seed,
        "steps": steps,
        "seq_len": seq_len,
        "context": context,
        "horizon": horizon,
        "split_seed": SPLIT_SEED,
        "git_sha": _git_sha(),
        "device": str(torch_device),
        "encoder_params": int(encoder_params),
        "seconds": float(time.perf_counter() - started),
        "steps_per_second": float(history["steps"] / history["seconds"]),
        "kl_rate_above_free_bits": float(history["kl_rate_above_free_bits"]),
        "kl_dyn_max": float(history["kl_dyn_max"]),
        "loss_last20": float(np.mean(history["loss"][-20:])),
        # The whole curve, every step, both the total and its five terms.
        # Coerced element by element so the record holds plain floats whatever
        # `train_world_model` appended (a non-finite one survives the coercion
        # and is written through the `nonfinite` map, never dropped).
        "history": {
            "loss": [float(v) for v in history["loss"]],
            "parts": [
                {str(k): float(v) for k, v in part.items()}
                for part in history["parts"]
            ],
        },
        # The held-out episodes by name, so "all nine cells were scored on the
        # same episodes" is checkable after the fact rather than assumed.
        "episodes": {
            "train": [p.name for p in train_paths],
            "val": [p.name for p in val_paths],
        },
        "probe": _probe_summary(latent_probe, embedding_probe),
        "position": metric_summary(result, "position"),
        "angle": metric_summary(result, "angle"),
        "filtering": {"criterion_4": criterion_4, "gain": gain},
        "reward": reward,
        "curves": {
            name: [float(v) for v in getattr(result, name)]
            for name in (
                "rssm_position", "persistence_position", "floor_position",
                "rssm_angle", "persistence_angle", "floor_angle",
            )
        },
    }
    assert set(METRICS) <= set(record), "every metric summary must be recorded"
    write_record(job_record_path(out_dir, job), record)
    return record
