"""Stage B live evaluation (M4 spec §5.4-§5.7).

measure: for one cell, the random baseline, the UNTRAINED actor (the §5.6
control) and every requested rung, each over the same 100 seeds; each rung in
both modes ("sample" is the gated one, "argmax" is reported).

read: the §5.5 gate at --final-step, refusing to report if the control passes
or the training record does not certify isolation.

Exit codes (read):
  0   PASS              the trained agent's paired interval lies above zero
  70  AGENT_NOT_BETTER  it does not
  71  CONTROL_BROKEN    the untrained actor passes the same gate
  72  ISOLATION_BROKEN  the training record's isolation_ok is not true
  11  MISSING           a record the read needs is absent
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mbfps.data.policies import RandomPolicy
from mbfps.envs.registry import make_env
from mbfps.eval.live import EVAL_SEEDS, AgentPolicy, paired_gate, run_episodes
from mbfps.eval.study import git_sha, load_record, write_record
from mbfps.models.encoders import encoder_backbone
from mbfps.models.heads import Actor, Critic
from mbfps.training.ac_trainer import load_actor
from mbfps.training.world_model import load_world_model
from mbfps.utils.device import get_device

EXIT_OK, EXIT_MISSING = 0, 11
EXIT_AGENT_NOT_BETTER, EXIT_CONTROL_BROKEN, EXIT_ISOLATION_BROKEN = 70, 71, 72
SCENARIO = "basic"


def _encoder_fn(model, device, stub: bool, pixel_ae_checkpoint: Path | None = None):
    """The frozen backbone that produced this arm's training cache, run live.

    `pixel_ae_checkpoint` MUST be the autoencoder the arm's cache was written
    with: `FeatureExtractor` otherwise loads `PIXEL_AE_CHECKPOINT`, the
    my_way_home autoencoder, and the agent would act on features from a
    different encoder than it was trained on. Ignored by the ViT backbones."""
    backbone = encoder_backbone(model.cfg.encoder)
    if backbone is None:
        return lambda frames: frames
    from mbfps.data.geometry import BACKBONE_GEOMETRY
    if stub:
        shape = BACKBONE_GEOMETRY[backbone]
        return lambda frames: np.zeros((len(frames), *shape), dtype=np.float16)
    from mbfps.data import features
    return features.FeatureExtractor(backbone=backbone, device=device.type,
                                     checkpoint=pixel_ae_checkpoint).encode


def _critic_slow(path: Path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    critic = Critic(seed=int(payload["seed"])).to(device)
    critic.load_state_dict(payload["critic_slow"])
    return critic.eval()


def live_path(out: Path, arm: str, seed: int) -> Path:
    return out / f"live_{arm}_seed{seed}.json"


def measure(args) -> int:
    start = time.perf_counter()
    device = get_device(prefer=args.device)
    model = load_world_model(args.world_model, device)
    arm, seed, n_actions = model.cfg.arm, model.cfg.train.seed, model.cfg.train.n_actions
    encode = _encoder_fn(model, device, args.encode_stub, args.pixel_ae_checkpoint)
    seeds = EVAL_SEEDS[: args.episodes]
    env = make_env("vizdoom", scenario=SCENARIO, frame_skip=4)
    n_episodes = 0
    try:
        record = {"arm": arm, "seed": seed, "seeds": list(seeds), "rungs": {}}
        record["random"] = run_episodes(env, RandomPolicy(env.action_space.n), seeds)
        untrained = AgentPolicy(model, Actor(n_actions, seed=seed).to(device).eval(),
                                encode, device)
        record["untrained"] = run_episodes(env, untrained, seeds)
        n_episodes += 2 * len(seeds)
        for step in args.steps:
            path = args.agent_dir / f"actor_critic_{arm}_seed{seed}_step{step}.pt"
            actor, critic_slow = load_actor(path, device), _critic_slow(path, device)
            record["rungs"][str(step)] = {
                mode: run_episodes(env, AgentPolicy(model, actor, encode, device, mode=mode,
                                                    critic_slow=critic_slow), seeds)
                for mode in ("sample", "argmax")
            }
            n_episodes += 2 * len(seeds)
            print(f"[live {arm} s{seed}] step {step} done")
    finally:
        env.close()
    seconds = time.perf_counter() - start
    record.update({"world_model": str(args.world_model), "agent_dir": str(args.agent_dir),
                   "git_sha": git_sha(), "seconds": seconds,
                   "seconds_per_episode": seconds / max(n_episodes, 1)})
    write_record(live_path(args.out, arm, seed), record)
    print(f"wrote {live_path(args.out, arm, seed)}")
    return EXIT_OK


def _summary(episodes) -> str:
    r = np.array([e["return"] for e in episodes])
    kills = np.mean([e["killed"] for e in episodes])
    length = np.median([e["length"] for e in episodes])
    return f"return {r.mean():+8.2f} (sd {r.std():6.2f})  kill {kills:.2f}  median_len {length:.0f}"


def read(args) -> int:
    path = live_path(args.out, args.arm, args.seed)
    train_path = args.agent_dir / f"train_{args.arm}_seed{args.seed}.json"
    if not path.is_file() or not train_path.is_file():
        print(f"MISSING: {path if not path.is_file() else train_path}")
        return EXIT_MISSING
    record, train = load_record(path), load_record(train_path)
    key = str(args.final_step)
    if key not in record["rungs"]:
        print(f"MISSING: rung {key} not in {sorted(record['rungs'])}")
        return EXIT_MISSING
    rand = [e["return"] for e in record["random"]]
    control = paired_gate([e["return"] for e in record["untrained"]], rand)
    final = record["rungs"][key]
    gate = paired_gate([e["return"] for e in final["sample"]], rand)
    imagined = [e["imagined_return"] for e in final["sample"]]
    real = [e["return"] for e in final["sample"]]

    lines = [f"M4 Stage B live evaluation -- {args.arm} seed {args.seed}, {len(rand)} seeds",
             f"  random      {_summary(record['random'])}",
             f"  untrained   {_summary(record['untrained'])}",
             f"    control gate: {control['point']:+.2f} [{control['ci_low']:+.2f}, {control['ci_high']:+.2f}]"
             f"  (must NOT pass)"]
    for step in sorted(record["rungs"], key=int):
        lines.append(f"  step {step:>6s} sample {_summary(record['rungs'][step]['sample'])}")
        lines.append(f"  step {step:>6s} argmax {_summary(record['rungs'][step]['argmax'])}")
    lines += [f"GATE at step {key}: {gate['point']:+.2f} [{gate['ci_low']:+.2f}, {gate['ci_high']:+.2f}]",
              f"imagined-vs-real gap at step {key}: {np.mean(imagined) - np.mean(real):+.2f} "
              f"(imagined {np.mean(imagined):+.2f}, real {np.mean(real):+.2f})",
              f"isolation_ok: {train.get('isolation_ok')}"]
    if train.get("isolation_ok") is not True:
        verdict, code = "ISOLATION_BROKEN", EXIT_ISOLATION_BROKEN
    elif control["passed"]:
        verdict, code = "CONTROL_BROKEN", EXIT_CONTROL_BROKEN
    elif gate["passed"]:
        verdict, code = "PASS", EXIT_OK
    else:
        verdict, code = "AGENT_NOT_BETTER", EXIT_AGENT_NOT_BETTER
    lines.append(f"VERDICT: {verdict}")
    text = "\n".join(lines) + "\n"
    path.with_suffix(".txt").write_text(text)
    print(text, end="")
    return code


def _steps(text: str) -> list[int]:
    return [int(s) for s in text.split(",") if s.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("measure", "read"), required=True)
    p.add_argument("--world-model", type=Path)
    p.add_argument("--agent-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/m4b_agent"))
    p.add_argument("--steps", type=_steps, default=[2000, 4000, 6000, 8000, 10000])
    p.add_argument("--episodes", type=int, default=len(EVAL_SEEDS))
    p.add_argument("--device", default="mps")
    p.add_argument("--encode-stub", action="store_true", help="tests only: zero features")
    p.add_argument("--pixel-ae-checkpoint", type=Path, default=None,
                   help="the autoencoder the pixel_ae cache was written with (Stage C: "
                        "runs/m2_basic/autoencoder_cnn.pt)")
    p.add_argument("--arm")
    p.add_argument("--seed", type=int)
    p.add_argument("--final-step", type=int, default=10_000)
    args = p.parse_args(argv)
    if args.phase == "measure":
        if args.world_model is None:
            raise SystemExit("--phase measure needs --world-model")
        return measure(args)
    if args.arm is None or args.seed is None:
        raise SystemExit("--phase read needs --arm and --seed")
    return read(args)


if __name__ == "__main__":
    sys.exit(main())
