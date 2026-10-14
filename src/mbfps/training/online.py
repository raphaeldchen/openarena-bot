"""The M5a driver (spec §3): prefill, then act 100 / train 20 until the budget
is spent, evaluating at step 0 and every 10,000 budget steps. After each
evaluation the full state is written; a rerun resumes from it and is
bit-identical to an uninterrupted run (§3.4)."""

import random
import time
from dataclasses import asdict, dataclass
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


def checkpoint_steps(cfg: OnlineConfig) -> list[int]:
    return list(range(0, cfg.budget + 1, cfg.eval_every))


def _make_arm(name: str, seed: int, device, encode, replay, kwargs):
    if name == "dreamer":
        return DreamerArm(seed, device, encode, replay, **kwargs)
    if name == "dqn":
        return DQNArm(seed, device, encode, **kwargs)
    raise ValueError(f"unknown arm {name!r}")


def _rng_state(device) -> dict:
    out = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch": torch.get_rng_state()}
    if device.type == "mps":
        out["mps"] = torch.mps.get_rng_state()
    return out


def _set_rng_state(state: dict, device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "mps":
        torch.mps.set_rng_state(state["mps"])


def _evaluate(arm, env, seeds) -> dict:
    return {mode: run_episodes(env, policy, seeds) for mode, policy in arm.eval_policies().items()}


def train_online(arm_name: str, seed: int, out_dir: Path, env_factory, encode, device,
                 cfg: OnlineConfig = OnlineConfig(), arm_kwargs: dict | None = None,
                 stop_after_step: int | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path, record_path = out_dir / "state.pt", out_dir / "record.json"
    if record_path.exists() and not state_path.exists():
        # A finished run whose resumable state was deleted after verification.
        # Starting fresh here would rerun the prefill and append to its replay.
        raise RuntimeError(f"{out_dir} already holds record.json but no state.pt: the run "
                           "is finished and its resumable state was removed; refusing to "
                           "start a fresh run on top of its replay")
    replay = ReplayBuffer(out_dir / "replay", capacity_transitions=10**9)
    arm = _make_arm(arm_name, seed, device, encode, replay, arm_kwargs or {})
    train_env, eval_env = env_factory(), env_factory()
    rollout = OnlineRollout(train_env, encode, replay, run_seed=seed,
                            on_episode=arm.add_episode, policy_name=arm.name)
    policy = arm.behaviour_policy(lambda: rollout.budget_step)
    timing = {"act": 0.0, "train": 0.0, "eval": 0.0}
    seeds = list(cfg.eval_seeds)

    if state_path.exists():
        # weights_only=False: the state holds NumPy RNG states and frame stacks,
        # which the restricted unpickler refuses. The file is written only by
        # `_checkpoint` below, into this run's own directory, and never shipped.
        saved = torch.load(state_path, weights_only=False)
        for path in saved["replay_paths"]:
            arm.add_episode(Path(path))
        arm.load_state_dict(saved["arm"])
        rollout.restore(saved["rollout"], policy)
        if saved["policy"] is not None:
            policy.set_state(saved["policy"])
        _set_rng_state(saved["rng"], device)
        record = load_record(record_path)
        record.pop(NONFINITE_KEY, None)    # write_record re-derives it and refuses a stale one
        update_rounds = record["update_rounds"]
        timing = record["timing"]
    else:
        prefill_policy = RandomPolicy(4, seed=seed)
        rollout.policy_name = "random"
        t0 = time.perf_counter()
        while rollout.budget_step < cfg.prefill:
            rollout.step(prefill_policy)
        rollout.close_episode_truncated()
        rollout.policy_name = arm.name
        timing["act"] += time.perf_counter() - t0
        record = {"arm": arm.name, "seed": seed, "gated_mode": arm.gated_mode, "seeds": seeds,
                  "config": asdict(cfg), "evals": [], "complete": False,
                  "random": run_episodes(eval_env, RandomPolicy(4, seed=0), seeds)}
        update_rounds = 0
        t0 = time.perf_counter()
        record["evals"].append({"step": 0, "modes": _evaluate(arm, eval_env, seeds),
                                "diagnostics": arm.diagnostics(), "update_rounds": 0})
        timing["eval"] += time.perf_counter() - t0
        _checkpoint(arm, rollout, policy, replay, record, update_rounds, timing, device,
                    state_path, record_path)

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
                        state_path, record_path)
    return record


def _checkpoint(arm, rollout, policy, replay, record, update_rounds, timing, device,
                state_path: Path, record_path: Path) -> None:
    record.update({"budget_steps": rollout.budget_step, "update_rounds": update_rounds,
                   "isolation_ok": arm.isolation_ok, "timing": dict(timing),
                   "seconds": sum(timing.values()), "git_sha": git_sha()})
    tmp = state_path.with_suffix(".tmp")
    torch.save({"arm": arm.state_dict(), "rollout": rollout.state(),
                "policy": policy.get_state() if rollout.in_progress else None,
                "replay_paths": [str(p) for p in replay.episode_paths()],
                "rng": _rng_state(device)}, tmp)
    tmp.replace(state_path)
    write_record(record_path, record)
