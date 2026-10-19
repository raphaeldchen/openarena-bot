"""M5a's default behaviour, frozen before the M5b-prerequisite cleanup.

The fixtures were captured at 9de983d (M5a as merged). The cleanup generalises
the arms, hardens deletion and the read, and speeds up both samplers. None of
that may change what M5a's defaults compute. A key the cleanup adds to the
record (e.g. `arm_kwargs`) is ignored here; every key the fixture holds must
match exactly."""

import hashlib
import json
from pathlib import Path

import numpy as np

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.loader import SequenceLoader
from mbfps.training.dqn import DQNConfig, TransitionSampler
from mbfps.training.online import train_online
from tests.data.test_loader import make_episode
from tests.training.fake_env import FakeEnv, fake_encode
from tests.training.test_online import CPU, DREAMER_KW, TINY, _strip

FIXTURES = Path(__file__).parent / "fixtures"


def _sha(batch: dict) -> str:
    m = hashlib.sha256()
    for k in sorted(batch):
        v = batch[k]
        if isinstance(v, np.ndarray):
            m.update(k.encode())
            m.update(np.ascontiguousarray(v).tobytes())
    return m.hexdigest()


def test_online_records_match_m5a(tmp_path):
    want = json.loads((FIXTURES / "online_defaults_m5a.json").read_text())
    for arm, kw in (("dreamer", DREAMER_KW), ("dqn", {"cfg": DQNConfig(batch_size=8)})):
        rec = train_online(arm, seed=0, out_dir=tmp_path / arm, env_factory=FakeEnv,
                           encode=fake_encode, device=CPU, cfg=TINY, arm_kwargs=kw)
        got = json.loads(_strip(rec))
        assert {k: got.get(k) for k in want[arm]} == want[arm], arm


def _draws(tmp_path):
    rng = np.random.default_rng(123)
    s = TransitionSampler(seed=7, cfg=DQNConfig(n_step=3, frame_stack=4))
    dqn = []
    for _ in range(30):
        for _ in range(10):
            T = int(rng.integers(1, 40))
            feats = rng.standard_normal((T + 1, 2, 3)).astype(np.float16)
            term, trunc = np.zeros(T, bool), np.zeros(T, bool)
            (term if rng.random() < 0.5 else trunc)[-1] = True
            s.add(feats, rng.integers(0, 4, T), rng.standard_normal(T).astype(np.float32),
                  term, trunc)
        dqn.append(_sha(s.sample(64)))
    buf = ReplayBuffer(tmp_path, capacity_transitions=1_000_000)
    loader = SequenceLoader(buf, batch_size=8, seq_len=16, seed=5, paths=[], pad_short=True)
    seq, fill = [], 0
    for _ in range(20):
        for _ in range(3):      # interleaved with sampling: a cache must see every add
            fill = (fill % 250) + 1
            loader.add(buf.add(make_episode(t=int(rng.integers(1, 40)), fill=fill)))
        b = loader.sample()
        seq.append({"episode_index": b["episode_index"].tolist(),
                    "window_start": b["window_start"].tolist(), "sha": _sha(b)})
    return dqn, seq


def test_sampler_draws_match_m5a(tmp_path):
    want = json.loads((FIXTURES / "sampler_draws_m5a.json").read_text())
    dqn, seq = _draws(tmp_path)
    assert dqn == want["transition_sampler"]
    assert seq == want["sequence_loader"]
