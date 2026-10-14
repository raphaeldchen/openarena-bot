"""Experience collection for the online loop (M5a §3.1, §3.3, §3.4).

Every `env.step` here is one budget step. An episode in progress when an act
phase ends simply continues on the next `step` call. A finished episode is
written once, whole, with its features alongside, and announced through
`on_episode`. The engine cannot be snapshotted mid-episode, so `state()`
records the episode's seed and actions and `restore()` replays them (M0
proved an episode replays exactly from its actions)."""

from pathlib import Path
from typing import Callable

import numpy as np

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.collector import privileged_row
from mbfps.data.episode import Episode
from mbfps.data.loader import feature_suffix

_EPISODE_SEED_BASE = 2_000_000
_EPISODES_PER_RUN = 100_000


def train_episode_seed(run_seed: int, k: int) -> int:
    """Seed of the run's k-th training episode, disjoint from EVAL_SEEDS."""
    assert 0 <= k < _EPISODES_PER_RUN, k
    return _EPISODE_SEED_BASE + _EPISODES_PER_RUN * int(run_seed) + int(k)


def act_with(policy, obs, features) -> int:
    if hasattr(policy, "act_from_features"):
        return int(policy.act_from_features(features))
    return int(policy.act(obs))


class OnlineRollout:
    def __init__(self, env, encode: Callable[[np.ndarray], np.ndarray], replay: ReplayBuffer,
                 run_seed: int, on_episode: Callable[[Path], None],
                 backbone: str = "dinov2", policy_name: str = "") -> None:
        self.env, self.encode, self.replay = env, encode, replay
        self.run_seed, self.on_episode = run_seed, on_episode
        self.backbone, self.policy_name = backbone, policy_name
        self.budget_step = 0
        self.episodes_done = 0
        self._ep: dict | None = None

    @property
    def in_progress(self) -> bool:
        return self._ep is not None

    @property
    def current_length(self) -> int:
        return len(self._ep["actions"]) if self._ep else 0

    def _start(self, policy) -> None:
        seed = train_episode_seed(self.run_seed, self.episodes_done)
        obs, _ = self.env.reset(seed=seed)
        policy.reset(seed=seed)
        state = self.env.privileged_state
        keys = tuple(state) if state else ()
        self._ep = {"seed": seed, "keys": keys, "obs": [obs.copy()],
                    "features": [self.encode(obs[None])[0]],
                    "privileged": [privileged_row(state, keys, previous=None)],
                    "actions": [], "rewards": [], "terminated": [], "truncated": []}

    def step(self, policy) -> None:
        if self._ep is None:
            self._start(policy)
        ep = self._ep
        action = act_with(policy, ep["obs"][-1], ep["features"][-1])
        obs, reward, terminated, truncated, _ = self.env.step(action)
        self.budget_step += 1
        self._record(action, obs, reward, terminated, truncated)
        if terminated or truncated:
            self._finish()

    def _record(self, action, obs, reward, terminated, truncated) -> None:
        ep = self._ep
        ep["actions"].append(int(action)); ep["rewards"].append(float(reward))
        ep["terminated"].append(bool(terminated)); ep["truncated"].append(bool(truncated))
        ep["obs"].append(obs.copy())
        ep["features"].append(self.encode(obs[None])[0])
        ep["privileged"].append(privileged_row(self.env.privileged_state, ep["keys"],
                                               previous=ep["privileged"][-1]))

    def close_episode_truncated(self) -> None:
        """End the in-progress episode as truncated (prefill end, budget end)."""
        if self._ep is not None and self._ep["actions"]:
            self._ep["truncated"][-1] = True
            self._finish()
        self._ep = None

    def _finish(self) -> None:
        ep = self._ep
        episode = Episode(
            obs=np.stack(ep["obs"]), actions=np.asarray(ep["actions"], np.int32),
            rewards=np.asarray(ep["rewards"], np.float32),
            terminated=np.asarray(ep["terminated"], bool),
            truncated=np.asarray(ep["truncated"], bool),
            privileged=np.stack(ep["privileged"]).astype(np.float32),
            privileged_keys=ep["keys"], policy_name=self.policy_name,
            seed=ep["seed"], scenario=self.env.scenario)
        path = self.replay.add(episode)
        np.save(path.with_suffix(feature_suffix(self.backbone)), np.stack(ep["features"]))
        self.episodes_done += 1
        self._ep = None
        self.on_episode(path)

    def state(self) -> dict:
        ep = self._ep
        return {"budget_step": self.budget_step, "episodes_done": self.episodes_done,
                "episode": None if ep is None else {
                    "seed": ep["seed"], "actions": list(ep["actions"])}}

    def restore(self, state: dict, policy) -> None:
        """Replay the in-progress episode's actions into the engine, re-encoding
        its frames. The policy's internal state is restored by the caller
        (its own `set_state`); `policy.reset` is called here only to mirror
        `_start`, and the caller overwrites the state afterwards."""
        self.budget_step = int(state["budget_step"])
        self.episodes_done = int(state["episodes_done"])
        self._ep = None
        saved = state["episode"]
        if saved is None:
            return
        self._start(policy)
        assert self._ep["seed"] == saved["seed"], (self._ep["seed"], saved["seed"])
        for action in saved["actions"]:
            obs, reward, terminated, truncated, _ = self.env.step(int(action))
            self._record(action, obs, reward, terminated, truncated)
