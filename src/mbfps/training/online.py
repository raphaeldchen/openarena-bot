"""The M5a driver (spec §3): prefill, then act 100 / train 20 until the budget
is spent, evaluating at step 0 and every 10,000 budget steps. After each
evaluation the full state is written; a rerun resumes from it and is
bit-identical to an uninterrupted run (§3.4)."""

import json
import random
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.policies import RandomPolicy
from mbfps.eval.live import EVAL_SEEDS, run_episodes
from mbfps.eval.study import NONFINITE_KEY, git_sha, load_record, write_record
from mbfps.training.arms import DQNArm, DreamerArm
from mbfps.training.online_rollout import OnlineRollout


@dataclass(frozen=True)
class OnlineConfig:
    budget: int = 100_000
    prefill: int = 5_000
    act_steps: int = 100
    rounds_per_act: int = 20
    eval_every: int = 10_000
    eval_seeds: tuple[int, ...] = EVAL_SEEDS

    def __post_init__(self) -> None:
        if (self.budget - self.prefill) % self.act_steps or self.eval_every % self.act_steps \
                or self.prefill % self.act_steps or self.budget % self.eval_every:
            raise ValueError("prefill, eval_every and budget must align with act_steps "
                             "and budget with eval_every, so every checkpoint is reached exactly")
        if self.prefill >= self.eval_every or self.prefill >= self.budget:
            raise ValueError("prefill must be below eval_every and below budget: the step-0 "
                             "evaluation happens right after the prefill, so it must precede "
                             "the first checkpoint (else it is mislabelled step 0 and the "
                             "eval_every checkpoint is never evaluated)")


def checkpoint_steps(cfg: OnlineConfig) -> list[int]:
    return list(range(0, cfg.budget + 1, cfg.eval_every))


def _make_arm(name: str, seed: int, device, encode, replay, kwargs):
    if name == "dreamer":
        return DreamerArm(seed, device, encode, replay, **kwargs)
    if name == "dqn":
        return DQNArm(seed, device, encode, **kwargs)
    raise ValueError(f"unknown arm {name!r}")


DEVICE_TYPES = ("cpu", "mps", "cuda")
OWNER = ".owner.json"


def _rng_state(device) -> dict:
    """Every generator a resume must restore. A device type this does not know
    raises: a silently unsaved generator would break the bit-identical resume (§3.4)."""
    if device.type not in DEVICE_TYPES:
        raise ValueError(f"cannot save the RNG state of a {device.type!r} device")
    out = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch": torch.get_rng_state()}
    if device.type == "mps":
        out["mps"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        out["cuda"] = torch.cuda.get_rng_state_all()
    return out


def _set_rng_state(state: dict, device) -> None:
    if device.type not in DEVICE_TYPES:
        raise ValueError(f"cannot restore the RNG state of a {device.type!r} device")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "mps":
        torch.mps.set_rng_state(state["mps"])
    elif device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])


def _meta(arm_name: str, seed: int, cfg: OnlineConfig, arm_kwargs: dict, device,
          scenario: str | None = None, n_actions: int | None = None) -> dict:
    """What identifies a run. A dataclass kwarg (DQNConfig) is stored by value.

    The commit, the device type, the scenario and its action count are part of
    it: a resume on other code, on another device or in another environment is
    not the run that was started, and is refused."""
    kwargs = {k: asdict(v) if is_dataclass(v) else v for k, v in arm_kwargs.items()}
    return {"arm": arm_name, "seed": seed, "cfg": asdict(cfg), "arm_kwargs": kwargs,
            "git_sha": git_sha(), "device": device.type, "scenario": scenario,
            "n_actions": n_actions}


def _prune_replay(root: Path, keep: set[str]) -> None:
    """Delete every episode file (and its feature caches) whose stem is not in `keep`.

    Episodes written after the last checkpoint by a process that then died are
    orphans: the saved state never counted them, so a resume would otherwise
    re-add them at the next checkpoint and diverge from an uninterrupted run.
    Run this before the ReplayBuffer is built so its index counter restarts
    where the uninterrupted run's was.
    """
    for pattern in ("ep_*.npz", "ep_*.features*.npy"):
        for path in root.glob(pattern):
            if path.name.split(".")[0] not in keep:
                path.unlink(missing_ok=True)


def _owner(meta: dict) -> dict:
    """Which run a replay belongs to: its identity without the commit and the
    device, so a run that died in its prefill restarts after a commit (its
    partial prefill is pruned, as before) while another run's files are never
    touched."""
    return _jsonable({k: v for k, v in meta.items() if k not in ("git_sha", "device")})


def _check_owner(replay_dir: Path, meta: dict, resuming: bool) -> None:
    """Refuse a fresh start over episode files some other run wrote: orphan
    pruning deletes every episode the saved state does not list, so it must
    only ever see this run's own files. A resume is pinned by state.pt."""
    owner_path = replay_dir / OWNER
    if not resuming and any(replay_dir.glob("ep_*")):
        owner = json.loads(owner_path.read_text()) if owner_path.exists() else None
        if owner != _owner(meta):
            raise RuntimeError(
                f"{replay_dir} holds episode files this run does not own (owner: {owner}); "
                "move or delete them, or choose another output directory")


def _write_owner(replay_dir: Path, meta: dict) -> None:
    """Mark `replay_dir` as this run's, before its first episode."""
    owner_path = replay_dir / OWNER
    tmp = owner_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_owner(meta), sort_keys=True))
    tmp.replace(owner_path)


def _evaluate(arm, env, seeds) -> dict:
    return {mode: run_episodes(env, policy, seeds) for mode, policy in arm.eval_policies().items()}


def train_online(arm_name: str, seed: int, out_dir: Path, env_factory, encode, device,
                 cfg: OnlineConfig = OnlineConfig(), arm_kwargs: dict | None = None,
                 stop_after_step: int | None = None) -> dict:
    if device.type not in DEVICE_TYPES:
        # Checked before any work: the first checkpoint would otherwise refuse
        # only after the prefill and the step-0 evaluation.
        raise ValueError(f"unsupported device type {device.type!r}; expected one of "
                         f"{DEVICE_TYPES}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path, record_path = out_dir / "state.pt", out_dir / "record.json"
    if record_path.exists() and not state_path.exists():
        # A finished run whose resumable state was deleted after verification.
        # Starting fresh here would rerun the prefill and append to its replay.
        raise RuntimeError(f"{out_dir} already holds record.json but no state.pt: the run "
                           "is finished and its resumable state was removed; refusing to "
                           "start a fresh run on top of it. To run it again, delete "
                           f"{out_dir} (or choose another output directory).")
    arm_kwargs = arm_kwargs or {}
    if "n_actions" in arm_kwargs:
        raise ValueError("n_actions comes from the environment's action space; do not pass it "
                         "in arm_kwargs")
    # The environments are built before the arm: the action count is the
    # environment's, and the arm's seed_everything resets every global RNG anyway.
    train_env = env_factory()
    try:
        eval_env = env_factory()
    except BaseException:
        train_env.close()
        raise
    n_actions = int(train_env.action_space.n)
    meta = _meta(arm_name, seed, cfg, arm_kwargs, device,
                 scenario=getattr(train_env, "scenario", None), n_actions=n_actions)
    saved = None
    if state_path.exists():
        # weights_only=False: the state holds NumPy RNG states and frame stacks,
        # which the restricted unpickler refuses. The file is written only by
        # `_checkpoint` below, into this run's own directory, and never shipped.
        saved = torch.load(state_path, weights_only=False)
        if saved["meta"] != meta:
            train_env.close(), eval_env.close()
            raise RuntimeError(
                f"{out_dir} holds a run that differs from this call: saved {saved['meta']}, "
                f"called with {meta}. To finish the saved run, check out "
                f"{saved['meta'].get('git_sha')} and repeat its exact call (arm, seed, config, "
                f"arm_kwargs, device, scenario); to start over, delete {out_dir}.")
    replay_dir = out_dir / "replay"
    try:
        replay_dir.mkdir(parents=True, exist_ok=True)
        _check_owner(replay_dir, meta, resuming=saved is not None)
        # state.pt is the source of truth. Anything in replay/ it does not list was
        # written by a process that died after the last checkpoint (or during the
        # prefill, when there is no state yet); drop it before the buffer counts it.
        _prune_replay(replay_dir, {Path(p).name.split(".")[0] for p in saved["replay_paths"]}
                      if saved else set())
        replay = ReplayBuffer(replay_dir, capacity_transitions=10**9)
        arm = _make_arm(arm_name, seed, device, encode, replay,
                        {**arm_kwargs, "n_actions": n_actions})
    except BaseException:
        train_env.close(), eval_env.close()
        raise
    _write_owner(replay_dir, meta)
    rollout = OnlineRollout(train_env, encode, replay, run_seed=seed, backbone=arm.backbone,
                            on_episode=arm.add_episode, policy_name=arm.name)
    policy = arm.behaviour_policy(lambda: rollout.budget_step)
    timing = {"act": 0.0, "train": 0.0, "eval": 0.0}
    seeds = list(cfg.eval_seeds)

    if saved is not None:
        for path in saved["replay_paths"]:
            arm.add_episode(replay_dir / Path(path).name)
        arm.load_state_dict(saved["arm"])
        rollout.restore(saved["rollout"], policy)
        if saved["policy"] is not None:
            policy.set_state(saved["policy"])
        _set_rng_state(saved["rng"], device)
        record, update_rounds, timing = saved["record"], saved["update_rounds"], saved["timing"]
        write_record(record_path, record)   # record.json is a derived copy; resync it
    else:
        prefill_policy = RandomPolicy(n_actions, seed=seed)
        rollout.policy_name = "random"
        t0 = time.perf_counter()
        while rollout.budget_step < cfg.prefill:
            rollout.step(prefill_policy)
        rollout.close_episode_truncated()
        rollout.policy_name = arm.name
        timing["act"] += time.perf_counter() - t0
        record = {"arm": arm.name, "seed": seed, "gated_mode": arm.gated_mode, "seeds": seeds,
                  "config": asdict(cfg), "git_sha": meta["git_sha"],
                  "scenario": meta["scenario"], "encoder": arm.encoder, "n_actions": n_actions,
                  "arm_kwargs": meta["arm_kwargs"], "arm_settings": _jsonable(arm.settings()),
                  "evals": [], "complete": False,
                  "random": run_episodes(eval_env, RandomPolicy(n_actions, seed=0), seeds)}
        update_rounds = 0
        t0 = time.perf_counter()
        record["evals"].append({"step": 0, "modes": _evaluate(arm, eval_env, seeds),
                                "diagnostics": arm.diagnostics(), "update_rounds": 0})
        timing["eval"] += time.perf_counter() - t0
        _checkpoint(arm, rollout, policy, replay, record, update_rounds, timing, device,
                    state_path, record_path, meta)

    while rollout.budget_step < cfg.budget:
        if stop_after_step is not None and rollout.budget_step >= stop_after_step:
            return record
        t0 = time.perf_counter()
        for _ in range(cfg.act_steps):
            rollout.step(policy)
        if rollout.budget_step == cfg.budget:
            rollout.close_episode_truncated()
        timing["act"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(cfg.rounds_per_act):
            arm.update_round()
            update_rounds += 1
        timing["train"] += time.perf_counter() - t0
        if rollout.budget_step % cfg.eval_every == 0:
            t0 = time.perf_counter()
            record["evals"].append({"step": rollout.budget_step,
                                    "modes": _evaluate(arm, eval_env, seeds),
                                    "diagnostics": arm.diagnostics(),
                                    "update_rounds": update_rounds})
            timing["eval"] += time.perf_counter() - t0
            record["complete"] = rollout.budget_step == cfg.budget
            _checkpoint(arm, rollout, policy, replay, record, update_rounds, timing, device,
                        state_path, record_path, meta)
    return record


def _checkpoint(arm, rollout, policy, replay, record, update_rounds, timing, device,
                state_path: Path, record_path: Path, meta: dict) -> None:
    record.update({"budget_steps": rollout.budget_step, "update_rounds": update_rounds,
                   "isolation_ok": arm.isolation_ok, "timing": dict(timing),
                   "seconds": sum(timing.values())})
    tmp = state_path.with_suffix(".tmp")
    torch.save({"arm": arm.state_dict(), "rollout": rollout.state(),
                "policy": policy.get_state() if rollout.in_progress else None,
                "replay_paths": [str(p) for p in replay.episode_paths()],
                "rng": _rng_state(device), "meta": meta, "record": record,
                "update_rounds": update_rounds, "timing": dict(timing)}, tmp)
    tmp.replace(state_path)
    write_record(record_path, record)


_MALFORMED = (KeyError, TypeError, AttributeError, StopIteration)


def _jsonable(x):
    return json.loads(json.dumps(x))


def verify_record(record: dict, cfg: OnlineConfig) -> list[str]:
    """Why a run's record is not complete and well-formed (§3.4, §7.3); [] if it is.
    A record missing a field or holding one of the wrong shape is reported, not raised."""
    try:
        return _record_problems(record, cfg)
    except _MALFORMED as exc:
        return [f"malformed record: {type(exc).__name__}: {exc}"]


def _record_problems(record: dict, cfg: OnlineConfig) -> list[str]:
    problems = []
    if _jsonable(record.get("config")) != _jsonable(asdict(cfg)):
        problems.append(f"config {record.get('config')} != {asdict(cfg)}")
    rounds = (cfg.budget - cfg.prefill) // cfg.act_steps * cfg.rounds_per_act
    if record.get("update_rounds") != rounds:
        problems.append(f"update_rounds {record.get('update_rounds')} != {rounds}")
    seeds = list(cfg.eval_seeds)
    if record.get("complete") is not True:
        problems.append("record is not complete")
    if record.get("budget_steps") != cfg.budget:
        problems.append(f"budget_steps {record.get('budget_steps')} != {cfg.budget}")
    steps = [e["step"] for e in record.get("evals", [])]
    if steps != checkpoint_steps(cfg):
        problems.append(f"eval steps {steps} != {checkpoint_steps(cfg)}")
    if [e["seed"] for e in record.get("random", [])] != seeds:
        problems.append("random baseline seeds differ from the evaluation seeds")
    for e in record.get("evals", []):
        if record.get("gated_mode") not in e["modes"]:
            problems.append(f"step {e['step']}: gated mode missing")
        for mode, episodes in e["modes"].items():
            if [x["seed"] for x in episodes] != seeds:
                problems.append(f"step {e['step']} {mode}: seeds differ from the evaluation seeds")
    return problems


def delete_replay_features(out_dir: Path, cfg: OnlineConfig) -> int:
    """§9: once a finished run's record verifies, delete everything it no longer
    needs -- its resumable state, the replay's feature caches, the episode frames
    and the owner marker -- leaving `replay/` empty and the record as the run's
    only output. Returns the number of files deleted."""
    out_dir = Path(out_dir)
    record_path = out_dir / "record.json"
    record = load_record(record_path)
    problems = verify_record(record, cfg)
    if problems:
        raise ValueError("refusing to delete replay features: " + "; ".join(problems))
    replay_dir = out_dir / "replay"
    state = out_dir / "state.pt"
    owner = replay_dir / OWNER
    victims = (([state] if state.exists() else [])
               + sorted(replay_dir.glob("*.features*.npy"))
               + sorted(replay_dir.glob("ep_*.npz"))
               + ([owner] if owner.exists() else []))
    # Flag the record first: a crash after this leaves a flagged run that can be
    # deleted again (idempotent), never a deleted run whose record says otherwise.
    # load_record restores the NaN/inf floats but leaves the map that recorded
    # them in the dict; write_record rebuilds that map and refuses a stale one.
    record["replay_features_deleted"] = True
    record["replay_frames_deleted"] = True
    record.pop(NONFINITE_KEY, None)
    write_record(record_path, record)
    # state.pt goes first, so a crash between the unlinks leaves no resume state
    # pointing at files that are gone; the frames go after their features.
    for path in victims:
        path.unlink()
    return len(victims)
