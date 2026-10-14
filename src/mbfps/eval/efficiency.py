"""M5a's pre-registered reading (spec §7)."""

import numpy as np

from mbfps.eval.live import paired_gate
from mbfps.training.online import OnlineConfig, checkpoint_steps, verify_record

ARMS: tuple[str, ...] = ("dreamer", "dqn")
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
WIN_BAR: float = 21.0
THRESHOLD: float = 50.0
VERDICTS = ("DREAMER_MORE_EFFICIENT", "DQN_MORE_EFFICIENT", "NOT_SEPARATED")


def _eval_at(record: dict, step: int) -> dict:
    return next(e for e in record["evals"] if e["step"] == step)


def gated_returns(record: dict, step: int) -> list[float]:
    return [float(e["return"]) for e in _eval_at(record, step)["modes"][record["gated_mode"]]]


def auc(record: dict, cfg: OnlineConfig) -> float:
    """§7.1: mean over the non-zero checkpoints of the mean gated return."""
    return float(np.mean([np.mean(gated_returns(record, s)) for s in checkpoint_steps(cfg)[1:]]))


def pairwise_wins(a, b) -> float:
    """Wins of `a` over `b` across all pairs; a tie counts one half."""
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None, :]
    return float((a > b).sum() + 0.5 * (a == b).sum())


def verdict(dreamer_aucs, dqn_aucs) -> str:
    if pairwise_wins(dreamer_aucs, dqn_aucs) >= WIN_BAR:
        return "DREAMER_MORE_EFFICIENT"
    if pairwise_wins(dqn_aucs, dreamer_aucs) >= WIN_BAR:
        return "DQN_MORE_EFFICIENT"
    return "NOT_SEPARATED"


def run_problems(record: dict, cfg: OnlineConfig) -> list[str]:
    """§7.3: everything that makes a run invalid; [] if it is valid."""
    problems = verify_record(record, cfg)
    if problems:
        return problems
    random_returns = [float(e["return"]) for e in record["random"]]
    if paired_gate(gated_returns(record, 0), random_returns)["passed"]:
        problems.append("step-0 control passes the paired gate against random")
    if record["arm"] == "dreamer" and record.get("isolation_ok") is not True:
        problems.append("isolation_ok is not true")
    return problems


def steps_to_threshold(record: dict, cfg: OnlineConfig, threshold: float = THRESHOLD):
    """First non-zero checkpoint whose mean gated return reaches `threshold`."""
    for s in checkpoint_steps(cfg)[1:]:
        if np.mean(gated_returns(record, s)) >= threshold:
            return s
    return None
