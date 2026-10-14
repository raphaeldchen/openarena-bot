import pytest

from mbfps.eval.efficiency import auc, pairwise_wins, run_problems, steps_to_threshold, verdict
from mbfps.training.online import OnlineConfig

CFG = OnlineConfig(budget=30, prefill=10, act_steps=10, eval_every=10, eval_seeds=(1, 2))


def _rec(arm="dreamer", curve=(-160.0, 10.0, 60.0, 70.0), step0=-170.0, random=-165.0,
         isolation=True):
    def eps(r):
        return [{"seed": s, "return": r} for s in (1, 2)]
    mode = "sample" if arm == "dreamer" else "greedy"
    evals = [{"step": 10 * i, "modes": {mode: eps(step0 if i == 0 else curve[i])}}
             for i in range(4)]
    return {"arm": arm, "seed": 0, "gated_mode": mode, "complete": True, "budget_steps": 30,
            "random": eps(random), "evals": evals,
            "isolation_ok": isolation if arm == "dreamer" else None}


def test_auc_is_the_mean_over_non_zero_checkpoints():
    assert auc(_rec(), CFG) == pytest.approx((10 + 60 + 70) / 3)


def test_pairwise_wins_count_ties_half():
    assert pairwise_wins([3, 2], [1, 2]) == 3.5


@pytest.mark.parametrize("d, expected_wins, expected", [
    ([9, 9, 9, 9, 9], 25, "DREAMER_MORE_EFFICIENT"),
    ([9, 9, 9, 9, 1], 21, "DREAMER_MORE_EFFICIENT"),
    ([9, 9, 9, 9, -1], 20, "NOT_SEPARATED"),
    ([1, 1, 1, 1, -1], 4, "DQN_MORE_EFFICIENT"),
    ([1, 1, 1, 1, 1], 5, "NOT_SEPARATED"),
])
def test_verdict_bar(d, expected_wins, expected):
    q = [0, 2, 4, 6, 8]
    assert pairwise_wins(d, q) == expected_wins
    assert verdict(d, q) == expected


def test_step0_control_must_fail():
    assert run_problems(_rec(), CFG) == []
    bad = _rec(step0=95.0)
    assert any("step-0 control" in p for p in run_problems(bad, CFG))


def test_isolation_required_for_dreamer_only():
    assert any("isolation" in p for p in run_problems(_rec(isolation=False), CFG))
    assert run_problems(_rec(arm="dqn"), CFG) == []


def test_steps_to_threshold():
    assert steps_to_threshold(_rec(), CFG) == 20
    assert steps_to_threshold(_rec(curve=(-160, 1, 2, 3)), CFG) is None
