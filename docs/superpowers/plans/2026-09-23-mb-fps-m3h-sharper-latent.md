# MB-FPS M3h — The Sharper Latent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ask whether the M3 rollout is noise-limited — by sampling the prior more sharply on the nine shipped checkpoints — and, only if it is, retrain those nine cells at the temperature the sweep names and read the gate.

**Architecture:** One constant threads a sampling temperature into `RSSM._sample`, the single place both the posterior and the prior draw; at 1.0 the division is skipped entirely so today's path is bitwise unchanged, and at every temperature the generator is drawn from exactly once per step so everything downstream of the rollout stays the one the gate scored. The canonical diagnostic pass gains a rollout-only override, a pure module holds the two readings, and one script runs `sweep → (gate) → train → evaluate → read`, refusing to retrain unless the sweep's own record says the rollout is noise-limited.

**Tech Stack:** Python 3.12, torch 2.13 (MPS), numpy, matplotlib (figures only), pytest. Spec: `docs/superpowers/specs/2026-09-23-mb-fps-m3h-sharper-latent-design.md`.

## Global Constraints

- **Pre-registered, never tuned after a run:** `TAU_GRID = (1.0, 0.7, 0.5, 0.3, 0.0)`, `TAU_CANDIDATES = (0.7, 0.5, 0.3)`, `REFERENCE_TAU = 1.0`, `DECISION_H = 15` (imported from `split_gap`), `SWEEP_FAMILY = 9`, `RETRAIN_FAMILY = 6`, `SEEDS_REQUIRED = 2`, `ARMS_REQUIRED = 2`, `STEPS = 20000`, `IDENTITY_STEPS = 500`. `SAMPLE_TEMPERATURE = 1.0` is the shipped default every M3b–M3g artefact was trained and evaluated at.
- **Bitwise at 1.0.** `_sample` skips the division when the temperature is 1.0, so the default path is byte-identical to today's by construction, not by relying on `x / 1.0`. Every default-path pin in `tests/models/test_rssm.py`, `tests/eval/test_diagnostics.py` and the script tests must stay green untouched.
- **One draw per step at every temperature**, including τ = 0, where the categorical draw is taken and **discarded** before the argmax. A sampler that skipped the draw would leave the generator at a different point and the floor's `observe` — which runs after the rollout — would no longer be the one the gate scored.
- **Reading N** (sweep): per arm and candidate τ, the paired contrast `S_free(15)[τ] − S_free(15)[1.0]` on the same windows; `NOISE_LIMITED` iff some candidate clears `z_fam = cluster_threshold(9, 24) = 3.06` pooled **and** in ≥ 2 of 3 seeds, in **≥ 2 of the 3 arms**; `τ*` = the largest pooled contrast among those clearing, ties to the **larger** τ. `SHARPER_WORSE` when none clears positively in two arms and some clears negatively in two or more. Otherwise `NOT_NOISE_LIMITED`. τ = 0.0 is an endpoint, never a candidate.
- **Reading M** (retrain): per arm, `S_free(15)[M3h] − S_free(15)[M3c]` with its probe twin as the control; `z_fam = cluster_threshold(6, 24) = 2.89`; precedence `UNRESOLVED_PROBE → SHARPER_BETTER → SHARPER_WORSE → NO_DIFFERENCE`, `SEEDS_REQUIRED = 2`, a single-seed leaf reading "this seed alone".
- **The M3 gate is reported, never decided on**, in both phases.
- **Exit codes:** 0 / 11 / 12 / 14 / 30 imported from `trust_horizon`; **35 `EXIT_NOT_NOISE_LIMITED`**, **36 `EXIT_TEMPERATURE_MISMATCH`**, **37 `EXIT_IDENTITY_CHECK_FAILED`** new; argparse 2 for `--out` equal to `--reference` or `--sweep-out`.
- **The retrain is gated in code**, not by a person: `train` reads the sweep record and refuses (35) unless it says `NOISE_LIMITED`, taking τ* from it.
- Nothing under `runs/` is removed; no shipped checkpoint is altered; the M3 gate, `aggregate.py`, `report_study.py` and every M3b–M3g verdict and record stand. Results live in this plan's `## Task 10 results`; earlier plans are closed. Never `git stash`. Do not commit while a run is in progress.
- **Style:** match the neighbouring docstring register (the *why*); tests carry hand-typed expected values, one rule mutated per test; every commit ends with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Run from the worktree root with `.venv/bin/python -m pytest ...`; `runs/` and `data/` are symlinks to the main checkout.

---

## File Structure

| file | responsibility |
|---|---|
| `src/mbfps/models/rssm.py` (modify) | `SAMPLE_TEMPERATURE`; `RSSMConfig.sample_temperature`; `_sample(logits, temperature=None)`; `imagine(..., temperature=None)` |
| `src/mbfps/utils/config.py` (modify) | `TrainConfig.sample_temperature`, so `get_config(arm, sample_temperature=τ)` reaches the model |
| `src/mbfps/training/world_model.py` (modify) | thread the temperature into `RSSMConfig`; `_save_checkpoint`'s payload gains `sample_temperature` |
| `scripts/diagnose_dynamics.py` (modify) | `load_checkpoint_model` applies and checks the payload's temperature (`TemperatureMismatch`) |
| `scripts/trust_horizon.py` (modify) | `prepare_cell` builds its config at the temperature its args name |
| `src/mbfps/eval/diagnostics.py` (modify) | `rollout_temperature` on `_diagnose` / `reference_trajectories` / the noise reference |
| `src/mbfps/eval/sharper.py` (new) | the constants, `tau_star`, Reading N and Reading M with their statuses, sentences and formatters |
| `scripts/sharper_latent.py` (new) | `sweep`, `train`, `evaluate`, `read`; the gate between the phases; the tables, `sweep.txt`, `sharper.txt`, the figures |
| tests | `tests/models/test_rssm.py`, `tests/utils/test_config.py`, `tests/training/test_world_model.py`, `tests/eval/test_diagnose_dynamics_script.py`, `tests/eval/test_trust_horizon_script.py`, `tests/eval/test_diagnostics.py` (modify); `tests/eval/test_sharper.py`, `tests/eval/test_sharper_latent_script.py` (new) |

Tasks 1–5 are independent of any run; 6 is the sweep, 7 the retrain, 8 the reading, 9 the registry and the smoke, 10 the run and its results.

---

### Task 1: The sampler's temperature

**Files:**
- Modify: `src/mbfps/models/rssm.py` (`RSSMConfig`, `_sample`, `imagine`), `src/mbfps/utils/config.py` (`TrainConfig`), `src/mbfps/training/world_model.py` (`WorldModel.__init__`)
- Test: `tests/models/test_rssm.py` (append), `tests/utils/test_config.py` (append)

**Interfaces:**
- Consumes: `RSSMConfig`, `RSSM._sample(logits)`, `RSSM.imagine(actions, state)`, `TrainConfig`, `WorldModel.__init__` as they are.
- Produces: `rssm.SAMPLE_TEMPERATURE = 1.0`; `RSSMConfig.sample_temperature: float = SAMPLE_TEMPERATURE`; `RSSM._sample(logits, temperature: float | None = None)`; `RSSM.imagine(actions, state, temperature: float | None = None)`; `TrainConfig.sample_temperature: float = SAMPLE_TEMPERATURE`; `WorldModel` builds its `RSSMConfig` with `sample_temperature=cfg.train.sample_temperature`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/models/test_rssm.py`:

```python


# ---------------------------------------------------------------------------
# The sampling temperature (M3h). The imagination's own resampling noise
# exceeds its imagined displacement from h = 5 onward, so M3h asks what a
# sharper sampler does. Two things can go wrong without breaking a shape: the
# default path can stop being bitwise what every shipped record was produced
# with, and a temperature can change how much randomness the sampler CONSUMES
# -- which would move the floor that runs after the rollout.
# ---------------------------------------------------------------------------


def _temperature_logits(seed: int = 0, batch: int = 4) -> torch.Tensor:
    """Logits with a deliberate spread, so tempering visibly changes the draw.

    Named apart from the module's existing `_logits` (the KL tests' peak-based
    helper): redefining that one would shadow it for the whole file."""
    return torch.randn(batch, 32 * 32, generator=torch.Generator().manual_seed(seed)) * 2.0


def test_the_temperature_defaults_to_one_and_is_the_shipped_value():
    assert rssm_module.SAMPLE_TEMPERATURE == 1.0
    assert RSSMConfig().sample_temperature == 1.0


def test_at_one_the_sample_is_bitwise_what_the_untempered_sampler_draws(rssm):
    """The default path is skipped, not divided: `x / 1.0` is exact in IEEE 754,
    but the pin is on the code taking no division at all, so a future change to
    the tempering arithmetic cannot move a shipped record."""
    logits = _temperature_logits()
    torch.manual_seed(7)
    explicit = rssm._sample(logits, temperature=1.0)
    torch.manual_seed(7)
    default = rssm._sample(logits)
    assert torch.equal(explicit, default)
    # The equality above cannot catch the regression this test exists for:
    # `x / 1.0` is bit-exact, so replacing the special case with an
    # unconditional `shaped / tau` would keep every number identical. The
    # guarantee is structural, so it is pinned on the source, as Task 2 pins
    # `prepare_cell`'s threading.
    source = inspect.getsource(rssm_module.RSSM._sample)
    division = [line for line in source.splitlines() if "/ tau" in line]
    assert division, "no tempering division found in _sample; has it been rewritten?"
    assert all("tau in (0.0, 1.0)" in line for line in division), (
        "the tempering division must be guarded so that 1.0 (and 0.0) take no division at "
        f"all; found {division}"
    )


def test_a_lower_temperature_draws_the_argmax_more_often_and_zero_always_does(rssm):
    """Sharper means sharper: over many draws of the same logits the sampled
    class agrees with the argmax monotonically more often as the temperature
    falls, and at 0 it agrees always."""
    logits = _temperature_logits(seed=1, batch=64)
    argmax = logits.view(64, 32, 32).argmax(-1)

    def agreement(tau):
        torch.manual_seed(11)
        drawn = rssm._sample(logits, temperature=tau).view(64, 32, 32).argmax(-1)
        return float((drawn == argmax).float().mean())

    one, seven, three, zero = agreement(1.0), agreement(0.7), agreement(0.3), agreement(0.0)
    assert zero == 1.0
    assert one < seven < three < zero, (one, seven, three, zero)


def test_every_temperature_consumes_exactly_one_draw_per_call(rssm):
    """THE stream pin. The floor's `observe` runs AFTER the rollout in the
    diagnostic pass, so a sampler that skipped the categorical draw at tau = 0
    -- the obvious implementation -- would leave the generator at a different
    point and the floor, the persistence anchor and the probe would all move.
    Every temperature draws once and tau = 0 discards what it drew."""
    logits = _temperature_logits(seed=2)
    states = {}
    for tau in (1.0, 0.7, 0.3, 0.0):
        torch.manual_seed(5)
        rssm._sample(logits, temperature=tau)
        states[tau] = torch.random.get_rng_state()
    for tau, state in states.items():
        assert torch.equal(state, states[1.0]), f"tau={tau} left the generator elsewhere"


def test_the_straight_through_gradient_survives_every_temperature(rssm):
    """The backward pass is the softmax of the UNTEMPERED logits at tau = 0 (no
    tempered distribution exists there) and of the tempered ones otherwise; in
    both cases a gradient reaches the logits, which is what lets the encoder
    train at all."""
    for tau in (1.0, 0.5, 0.0):
        logits = _temperature_logits(seed=3).requires_grad_(True)
        rssm._sample(logits, temperature=tau).sum().backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all(), tau
        assert logits.grad.abs().sum() > 0, tau


def test_a_negative_temperature_is_refused(rssm):
    with pytest.raises(ValueError, match="temperature"):
        rssm._sample(_temperature_logits(), temperature=-0.5)


def test_imagine_takes_a_rollout_override_and_observe_does_not(rssm):
    """The sweep sharpens the ROLLOUT on shipped checkpoints: `imagine` takes
    an override, `observe` always uses the model's own, so the context filter
    and the floor are untouched by it."""
    import inspect

    assert inspect.signature(rssm.imagine).parameters["temperature"].default is None
    assert "temperature" not in inspect.signature(rssm.observe).parameters
    b, t = 2, 5
    actions = torch.zeros(b, t, dtype=torch.long)
    state = (torch.zeros(b, 512), torch.zeros(b, 1024))
    torch.manual_seed(13)
    warm = rssm.imagine(actions, state)
    torch.manual_seed(13)
    sharp = rssm.imagine(actions, state, temperature=0.0)
    assert not torch.equal(warm["z"], sharp["z"])
    torch.manual_seed(13)
    again = rssm.imagine(actions, state, temperature=1.0)
    assert torch.equal(warm["z"], again["z"])


def test_a_configured_temperature_drives_both_paths(rssm):
    """`RSSMConfig.sample_temperature` is what a TRAINED model samples at, in
    `observe` as well as `imagine` -- the retrain's whole intervention."""
    cfg = RSSMConfig(sample_temperature=0.0)
    sharp = RSSM(cfg, seed=0)
    logits = _temperature_logits(seed=4)
    assert torch.equal(
        sharp._sample(logits), sharp._sample(logits, temperature=0.0)
    )
    b, t = 2, 4
    embeddings = torch.randn(b, t, cfg.embed_dim, generator=torch.Generator().manual_seed(9))
    actions = torch.zeros(b, t, dtype=torch.long)
    torch.manual_seed(17)
    out = sharp.observe(embeddings, actions)
    modes = out["post_logits"].argmax(-1)
    drawn = out["z"].view(b, t, 32, 32).argmax(-1)
    assert torch.equal(modes, drawn), "a configured tau=0 posterior must draw its own mode"
```

Add to the imports at the top of `tests/models/test_rssm.py` (it already imports `pytest`, `torch`, and `RSSM`/`RSSMConfig` from `mbfps.models.rssm`, and defines a module-level `rssm` fixture returning `RSSM(RSSMConfig())`): `import inspect` and `import mbfps.models.rssm as rssm_module`. The file's existing `_logits(b=2, t=3, peak=0.0)` helper is left exactly as it is — the new helper above is named apart from it, because redefining it would shadow it for every KL test in the file.

Append to `tests/utils/test_config.py`:

```python


def test_the_sample_temperature_is_a_train_config_field_at_the_shipped_default():
    """`get_config(arm, sample_temperature=tau)` is how M3h reaches the model:
    the override path every script already uses, so no script needs a new
    argument to build a sharper model."""
    from mbfps.models.rssm import SAMPLE_TEMPERATURE
    from mbfps.utils.config import TrainConfig, get_config

    assert TrainConfig().sample_temperature == SAMPLE_TEMPERATURE == 1.0
    assert get_config("pixel_ae").train.sample_temperature == 1.0
    assert get_config("pixel_ae", sample_temperature=0.5).train.sample_temperature == 0.5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/models/test_rssm.py tests/utils/test_config.py -q -k "temperature or consumes or straight_through or imagine_takes"`
Expected: FAIL — `AttributeError: module 'mbfps.models.rssm' has no attribute 'SAMPLE_TEMPERATURE'`, and `TypeError: _sample() got an unexpected keyword argument 'temperature'`.

- [ ] **Step 3: Add the constant and the config field**

In `src/mbfps/models/rssm.py`, after the imports and before `RSSMConfig`:

```python
SAMPLE_TEMPERATURE: float = 1.0
"""The temperature the categorical state is drawn at, in `observe` and in
`imagine` alike. 1.0 is what every M3b-M3g artefact was trained and evaluated
at, and `_sample` skips the division entirely at that value, so the shipped
path is bitwise unchanged by this parameter existing.

Below 1.0 the draw concentrates on the mode. M3h's measurement is why the
parameter is here: at 20k the posterior carries 0.71-0.87 nats per group over
32 groups, so every imagined step injects ~23-28 nats of fresh entropy against
the ~0.4-0.6 nats of dynamics information -- and two draws of the same model
from the same state separate, from h = 5 onward, by more than that model's
entire imagined displacement.
"""
```

and add the field to `RSSMConfig`, last, so every positional construction still works:

```python
    hidden: int = 512
    sample_temperature: float = SAMPLE_TEMPERATURE
```

In `src/mbfps/utils/config.py`, add to `TrainConfig` after `device`:

```python
    sample_temperature: float = SAMPLE_TEMPERATURE
```

with `from mbfps.models.rssm import SAMPLE_TEMPERATURE` at the top of the file if the import does not create a cycle; if it does, restate the literal `1.0` **and** add this assertion to the module so the two cannot drift:

```python
# The one place the default lives is rssm.py; this file only mirrors it.
assert TrainConfig().sample_temperature == _rssm_sample_temperature()
```

(Prefer the import. Check by running `.venv/bin/python -c "import mbfps.utils.config"` after the edit.)

- [ ] **Step 4: Temper the sampler**

Replace `RSSM._sample`'s body (its docstring stays, with the paragraph below appended):

```python
    def _sample(self, logits: torch.Tensor, temperature: float | None = None) -> torch.Tensor:
```

Append to its docstring:

```
        `temperature` (M3h) divides the logits before the softmax; None uses
        the model's own `cfg.sample_temperature`. At 1.0 NO division is taken
        -- the shipped path is bitwise unchanged by construction rather than
        by `x / 1.0` happening to be exact. At 0.0 the draw is the argmax, and
        the straight-through gradient is the untempered softmax.

        **Every temperature consumes exactly one categorical draw**, including
        0.0, where the draw is taken and discarded. The diagnostic pass runs
        the floor's `observe` AFTER the rollout, so a sampler that skipped the
        draw would leave the generator at a different point and the floor, the
        persistence anchor and the probe -- the fixed brackets of every
        comparison -- would move with the temperature.
```

and the body:

```python
        tau = self.cfg.sample_temperature if temperature is None else float(temperature)
        if tau < 0.0:
            raise ValueError(f"sampling temperature must be >= 0, got {tau}")
        shaped = logits.view(*logits.shape[:-1], self.cfg.z_cats, self.cfg.z_classes)
        # No division at 1.0 or 0.0: the first is the shipped path, the second
        # has no tempered distribution to build.
        probs = F.softmax(shaped if tau in (0.0, 1.0) else shaped / tau, dim=-1)
        index = torch.distributions.Categorical(probs=probs).sample()
        if tau == 0.0:
            # Drawn and discarded above: the stream must not depend on tau.
            index = shaped.argmax(dim=-1)
        onehot = F.one_hot(index, self.cfg.z_classes).to(probs.dtype)
        return (probs + (onehot - probs).detach()).flatten(-2)
```

- [ ] **Step 5: Give `imagine` its override**

```python
    def imagine(self, actions, state, temperature: float | None = None) -> dict[str, torch.Tensor]:
        """Roll forward on the prior alone -- no embeddings consumed.

        `temperature` (M3h) sharpens THIS rollout only; `observe` always draws
        at the model's own, so a sweep over rollout temperatures leaves the
        context filter and the floor exactly as the gate scored them."""
        h, z = state
        actions_onehot = self._onehot_actions(actions)
        hs, zs, priors = [], [], []
        for i in range(actions.shape[1]):
            h = self._step(h, z, actions_onehot[:, i])
            prior_logits = self.prior_net(h)
            z = self._sample(prior_logits, temperature)
            hs.append(h); zs.append(z); priors.append(prior_logits)
        return self._pack(hs, zs, priors, None)
```

- [ ] **Step 6: Thread it into the world model**

In `src/mbfps/training/world_model.py`, `WorldModel.__init__`:

```python
        self.rssm = RSSM(
            RSSMConfig(
                embed_dim=cfg.encoder.embed_dim,
                sample_temperature=cfg.train.sample_temperature,
            ),
            seed=cfg.train.seed,
        )
```

- [ ] **Step 7: Run the new tests, then the files they belong to**

Run: `.venv/bin/python -m pytest tests/models/test_rssm.py tests/utils/test_config.py -q`
Expected: every test passes — the 26 pre-existing `test_rssm.py` tests unchanged (including the anti-collapse pin `test_.._distinct_latents..`), plus the eight new ones and the config one.

- [ ] **Step 8: Run the consumers of the sampler**

Run: `.venv/bin/python -m pytest tests/eval/test_rollout.py tests/eval/test_diagnostics.py tests/training/test_world_model.py -q`
Expected: PASS with no change — the default path is bitwise what it was, which is exactly what these files pin.

- [ ] **Step 9: Commit**

```bash
git add src/mbfps/models/rssm.py src/mbfps/utils/config.py src/mbfps/training/world_model.py tests/models/test_rssm.py tests/utils/test_config.py
git commit -m "feat: a sampling temperature on the categorical state -- skipped entirely at 1.0 so the shipped path is bitwise unchanged, and drawing once per step at every temperature so the floor after the rollout does not move

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: The temperature travels with the checkpoint

**Files:**
- Modify: `src/mbfps/training/world_model.py` (`_save_checkpoint`), `scripts/diagnose_dynamics.py` (`load_checkpoint_model`, a new error class), `scripts/trust_horizon.py` (`prepare_cell`'s `get_config` call)
- Test: `tests/training/test_world_model.py` (the payload pins), `tests/eval/test_diagnose_dynamics_script.py` (append), `tests/eval/test_trust_horizon_script.py` (append)

**Interfaces:**
- Consumes: Task 1's `TrainConfig.sample_temperature`; `_save_checkpoint(model, cfg, directory, step=None)`; `load_checkpoint_model(out_dir, arm, seed, cfg, device)`; `prepare_cell(args, cell, device, train, val)`.
- Produces: the payload key `"sample_temperature"` (always written); `diagnose_dynamics.TemperatureMismatch(ValueError)`; `prepare_cell` building its config at `getattr(args, "sample_temperature", SAMPLE_TEMPERATURE)`.

- [ ] **Step 1: Write the failing tests**

In `tests/training/test_world_model.py`, change the two payload pins (around lines 748–768) to include the new key, and append the new test:

```python
    assert set(payload) == {"arm", "seed", "state_dict", "sample_temperature"}
```

(the final-save pin), and for the rung pin:

```python
        assert set(payload) == {"arm", "seed", "step", "state_dict", "sample_temperature"}
```

Then append:

```python
def test_the_payload_carries_the_temperature_the_model_was_trained_at(tmp_path, buffer):
    """A checkpoint trained sharper and loaded into a default configuration
    would be evaluated at 1.0 in silence -- the model would sample differently
    from the one whose weights are being read, and nothing would say so. The
    temperature travels in the payload so the loader can refuse."""
    from mbfps.training.world_model import train_world_model
    from mbfps.utils.config import get_config

    out = tmp_path / "sharp"
    cfg = get_config("cnn", steps=2, seq_len=4, seed=3, device="cpu", sample_temperature=0.5)
    train_world_model(cfg, buffer, out_dir=out)
    payload = torch.load(out / "world_model_cnn_seed3.pt", weights_only=True)
    assert payload["sample_temperature"] == 0.5

    warm = tmp_path / "warm"
    train_world_model(
        get_config("cnn", steps=2, seq_len=4, seed=3, device="cpu"), buffer, out_dir=warm
    )
    assert torch.load(warm / "world_model_cnn_seed3.pt", weights_only=True)["sample_temperature"] == 1.0
```

Append to `tests/eval/test_diagnose_dynamics_script.py`:

```python


def test_the_loader_applies_the_payloads_temperature_and_refuses_a_mismatch(tmp_path):
    """The loader builds the model it is asked for and then checks that the
    weights it is about to load were produced by a model that samples the same
    way. A shipped checkpoint (no key) reads as 1.0, so every M3b-M3g artefact
    loads unchanged; a sharper checkpoint asked for at 1.0 is refused by name
    rather than evaluated as something it is not."""
    import torch

    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    cfg = get_config("random_vit", seed=1, device="cpu")
    model = WorldModel(cfg)
    path = tmp_path / "world_model_random_vit_seed1.pt"

    torch.save({"arm": "random_vit", "seed": 1, "state_dict": model.state_dict()}, path)
    loaded = script.load_checkpoint_model(tmp_path, "random_vit", 1, cfg, torch.device("cpu"))
    assert loaded.rssm.cfg.sample_temperature == 1.0

    torch.save(
        {"arm": "random_vit", "seed": 1, "state_dict": model.state_dict(), "sample_temperature": 0.5},
        path,
    )
    with pytest.raises(script.TemperatureMismatch, match="0.5"):
        script.load_checkpoint_model(tmp_path, "random_vit", 1, cfg, torch.device("cpu"))
    sharp = get_config("random_vit", seed=1, device="cpu", sample_temperature=0.5)
    loaded = script.load_checkpoint_model(tmp_path, "random_vit", 1, sharp, torch.device("cpu"))
    assert loaded.rssm.cfg.sample_temperature == 0.5
    assert issubclass(script.TemperatureMismatch, ValueError)
```

Bind the class the script can raise beside the other `_diagnose_dynamics.*` names in `scripts/trust_horizon.py` (each `_sibling` load builds a distinct class object, so a caller that catches its own copy would miss this one):

```python
TemperatureMismatch = _diagnose_dynamics.TemperatureMismatch
```

Append to `tests/eval/test_trust_horizon_script.py`:

```python


def test_prepare_cell_builds_its_config_at_the_temperature_its_args_name(cell):
    """Every existing caller sets no temperature and therefore gets 1.0, which
    is what every cell they read was trained at; M3h's script sets it, so the
    model `prepare_cell` returns samples the way its checkpoint was trained to.

    Pinned through the LOADER rather than on the source: this cell's checkpoint
    was trained at 1.0, so asking for it at 0.5 must be refused -- and it can
    only be refused if the 0.5 travelled from the args through `get_config`
    into the configuration `load_checkpoint_model` checks the payload against.
    A test that read the source for the attribute's name would pass on code
    that named it and then used the wrong one.
    """
    import argparse

    from mbfps.data.buffer import ReplayBuffer
    from mbfps.data.split import VAL_FRACTION, episode_split
    from mbfps.models.rssm import SAMPLE_TEMPERATURE
    from mbfps.utils.device import get_device

    loaded = script.load_cell(cell.out, JOB.arm, JOB.seed)
    train, val = episode_split(
        ReplayBuffer(cell.data, capacity_transitions=10**9).episode_paths(),
        val_fraction=VAL_FRACTION, seed=0,
    )
    device = get_device(prefer="cpu")

    warm = argparse.Namespace(out=cell.out, device="cpu", context=None, horizon=None)
    status, prepared = script.prepare_cell(warm, loaded, device, train, val)
    assert status == script.EXIT_OK
    assert prepared.model.rssm.cfg.sample_temperature == SAMPLE_TEMPERATURE == 1.0

    sharp = argparse.Namespace(
        out=cell.out, device="cpu", context=None, horizon=None, sample_temperature=0.5
    )
    # `script.TemperatureMismatch`, not the test's own `diagnose` copy: each
    # `_sibling` load builds a distinct class object, so the one `prepare_cell`
    # raises is trust_horizon's, and a caller catching any other misses it.
    with pytest.raises(script.TemperatureMismatch, match="0.5"):
        script.prepare_cell(sharp, loaded, device, train, val)

```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/training/test_world_model.py tests/eval/test_diagnose_dynamics_script.py tests/eval/test_trust_horizon_script.py -q -k "temperature or payload"`
Expected: FAIL — the payload has three keys, and `script.TemperatureMismatch` does not exist.

- [ ] **Step 3: Write the temperature into the payload**

In `src/mbfps/training/world_model.py`, `_save_checkpoint`:

```python
    payload: dict[str, Any] = {
        "arm": cfg.arm,
        "seed": cfg.train.seed,
        "sample_temperature": float(cfg.train.sample_temperature),
        "state_dict": model.state_dict(),
    }
```

and extend its docstring: `` The payload also carries the `sample_temperature` the model was trained at (M3h), so a loader can refuse a checkpoint whose sampler is not the one it is about to build. ``

- [ ] **Step 4: Apply and check it in the loader**

In `scripts/diagnose_dynamics.py`, beside `MislabelledCheckpoint`:

```python
class TemperatureMismatch(ValueError):
    """The checkpoint was trained at a sampling temperature other than the one
    this configuration builds. Loading it anyway would evaluate weights with a
    sampler their training never saw -- silently, since nothing about the
    state_dict says how it was drawn from. Absent from the payload means 1.0,
    which is every M3b-M3g artefact."""
```

and in `load_checkpoint_model`, after the arm/seed check:

```python
    trained_at = float(checkpoint.get("sample_temperature", SAMPLE_TEMPERATURE))
    asked_for = float(cfg.train.sample_temperature)
    if trained_at != asked_for:
        raise TemperatureMismatch(
            f"checkpoint in {out_dir} was trained at sample_temperature={trained_at}, "
            f"but this configuration samples at {asked_for}"
        )
```

with `from mbfps.models.rssm import SAMPLE_TEMPERATURE` added to the script's imports.

- [ ] **Step 5: Let `prepare_cell` carry it**

In `scripts/trust_horizon.py`, `prepare_cell`, replace the `cfg = get_config(...)` line:

```python
    # Every existing caller sets no temperature and gets the shipped 1.0, which
    # is what every cell they read was trained at; M3h's script sets it, so the
    # model returned here samples the way its checkpoint was trained to.
    cfg = get_config(
        arm, seed=seed, device=args.device,
        sample_temperature=getattr(args, "sample_temperature", SAMPLE_TEMPERATURE),
    )
```

with `from mbfps.models.rssm import SAMPLE_TEMPERATURE` added to that script's imports.

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/training/test_world_model.py tests/eval/test_diagnose_dynamics_script.py tests/eval/test_trust_horizon_script.py -q`
Expected: all pass.

- [ ] **Step 7: Run every consumer of a shipped checkpoint**

Run: `.venv/bin/python -m pytest tests/eval/test_split_gap_script.py tests/eval/test_checkpoint_ladder_script.py tests/eval/test_stage_decomposition_script.py -q`
Expected: PASS — their fixtures train and read at 1.0 throughout, so nothing changes.

- [ ] **Step 8: Commit**

```bash
git add src/mbfps/training/world_model.py scripts/diagnose_dynamics.py scripts/trust_horizon.py tests/training/test_world_model.py tests/eval/test_diagnose_dynamics_script.py tests/eval/test_trust_horizon_script.py
git commit -m "feat: the sampling temperature travels in the checkpoint payload and the loader refuses a mismatch; prepare_cell builds its config at the temperature its args name

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: A rollout temperature on the canonical pass

**Files:**
- Modify: `src/mbfps/eval/diagnostics.py` (`_noise_reference`, `_diagnose`, `reference_trajectories`)
- Test: `tests/eval/test_diagnostics.py` (append)

**Interfaces:**
- Consumes: Task 1's `imagine(..., temperature=None)`; `_diagnose(..., keep_trajectories=False, keep_latents=False)`; `reference_trajectories(...)`; `_noise_reference(model, handle, tail)`.
- Produces: `_diagnose(..., rollout_temperature: float | None = None)`, `reference_trajectories(..., rollout_temperature: float | None = None)`, `_noise_reference(model, handle, tail, temperature=None)`, and `Trajectories.noise_embedding` — the pass already computes the noise reference on every call and used to drop it, so Task 6 would otherwise have had to run the whole pass a second time to read the one statistic that says whether a rollout's motion is its own dynamics or its own sampling. **None means the model's own temperature and the keyword is not passed to `imagine` at all** — a default of 1.0 would evaluate a retrained cell with a sampler it was never trained for, and would break every existing two-argument stand-in for `imagine` in the suite.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_diagnostics.py`:

```python


# ---------------------------------------------------------------------------
# The rollout temperature (M3h). The sweep sharpens the canonical rollout on
# shipped checkpoints and must leave everything the rollout is judged against
# exactly where it was: the context filter, the floor, the persistence anchor
# and the probe. The noise reference moves WITH the rollout, because "two
# draws of the same model" has to mean two draws of the same sampler.
# ---------------------------------------------------------------------------


def temperature_pass(model, paths, probe, *, tau, device=None, seed=0):
    with torch.no_grad():
        return diagnostics_module._diagnose(
            model, paths, probe, arms={}, context=CONTEXT, horizon=HORIZON,
            seed=seed, device=device or torch.device("cpu"), feature_backbone=None,
            noise_reference=True, keep_trajectories=True, rollout_temperature=tau,
        )


def test_reference_trajectories_carries_the_passs_noise_reference(tmp_path):
    """`_diagnose` draws the noise reference on every pass -- a second
    imagination from the canonical one's own stream point -- and used to drop
    it on the way out, so a consumer that needed it had to run the whole pass
    again. It is the statistic that says whether a rollout's motion is its
    own dynamics or its own sampling (M3h), so it makes the trip."""
    paths = [write(tmp_path, varied_action_episode())]
    model, probe = real_model_and_probe()
    result = reference_trajectories(
        model, paths, probe, context=CONTEXT, horizon=HORIZON, seed=7,
        device=torch.device("cpu"), feature_backbone=None,
    )
    assert result.noise_embedding.shape == (result.windows_total, HORIZON)
    assert np.isfinite(result.noise_embedding).all()
    assert (result.noise_embedding > 0).any(), "a sampling model's two draws differ"
    # Additive: a fabricated Trajectories that reads no noise need not build one.
    assert Trajectories.__dataclass_fields__["noise_embedding"].default is None


def test_the_rollout_temperature_is_keyword_only_and_the_models_own_by_default():
    """None, not 1.0: a model retrained at another temperature must roll out at
    the temperature it was trained at, and a hardcoded 1.0 here would evaluate
    it with a sampler its training never saw."""
    for function in (diagnostics_module._diagnose, reference_trajectories):
        parameter = inspect.signature(function).parameters["rollout_temperature"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, function
        assert parameter.default is None, function


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_without_an_override_the_rollout_is_drawn_at_the_models_own_temperature(tmp_path, device):
    """THE pin on the default's meaning. A model whose own temperature is 0
    rolls out deterministically when nothing is asked for, and asking for 1.0
    explicitly is a different pass -- so the default cannot be a synonym for
    1.0, and the keyword is not passed on the default path at all."""
    model, paths, probe = _two_episode_rig(tmp_path, device)
    model.rssm.cfg = replace(model.rssm.cfg, sample_temperature=0.0)

    own = temperature_pass(model, paths, probe, tau=None, device=device)
    again = temperature_pass(model, paths, probe, tau=None, device=device)
    sharp = temperature_pass(model, paths, probe, tau=0.0, device=device)
    warm = temperature_pass(model, paths, probe, tau=1.0, device=device)

    np.testing.assert_array_equal(own.positions, sharp.positions)
    np.testing.assert_array_equal(own.positions, again.positions)
    assert not np.array_equal(own.positions, warm.positions)


def test_the_default_path_passes_no_temperature_keyword_to_imagine(tmp_path):
    """A stand-in for `imagine` with the signature it has always had must keep
    working, which is what the suite's own doubles rely on; an override is the
    only thing that adds the keyword."""
    model, paths, probe = _two_episode_rig(tmp_path, torch.device("cpu"))
    seen = []
    real = model.rssm.imagine

    def recording(actions, state, *args, **kwargs):
        seen.append(("temperature" in kwargs, kwargs.get("temperature")))
        return real(actions, state, *args, **kwargs)

    model.rssm.imagine = recording
    temperature_pass(model, paths, probe, tau=None)
    assert seen and all(passed is False for passed, _ in seen), seen
    seen.clear()
    temperature_pass(model, paths, probe, tau=0.3)
    assert seen and all(passed and value == 0.3 for passed, value in seen), seen


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_at_one_the_pass_is_bitwise_what_it_is_without_the_argument(tmp_path, device):
    """The anchor the sweep's self-check depends on: asking for 1.0 explicitly
    and not asking at all are the same pass, field for field, and both leave
    the generator in the same state."""
    model, paths, probe = _two_episode_rig(tmp_path, device)
    plain = reference_pass(model, paths, probe, keep=True, device=device)
    left_by_plain = diagnostics_module._rng_snapshot(device)
    tempered = temperature_pass(model, paths, probe, tau=1.0, device=device)
    left_by_tempered = diagnostics_module._rng_snapshot(device)

    for name in TRAJECTORY_FIELDS:
        np.testing.assert_array_equal(getattr(tempered, name), getattr(plain, name), err_msg=name)
    for name in ("rssm_position", "persistence_position", "floor_position",
                 "rssm_angle", "persistence_angle", "floor_angle"):
        np.testing.assert_array_equal(
            getattr(tempered.reference, name), getattr(plain.reference, name), err_msg=name
        )
    np.testing.assert_array_equal(tempered.noise_embedding, plain.noise_embedding)
    for key in left_by_plain:
        assert torch.equal(left_by_tempered[key], left_by_plain[key]), key


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_a_sharper_rollout_moves_the_imagination_and_nothing_it_is_judged_against(tmp_path, device):
    """The whole point, pinned: at tau = 0 the imagined rows differ from tau = 1
    and the floor, the persistence anchor and the true positions are bitwise
    identical -- because every temperature consumes the same draws, so the
    floor's `observe`, which runs after the rollout, starts from the same
    generator state."""
    model, paths, probe = _two_episode_rig(tmp_path, device)
    warm = temperature_pass(model, paths, probe, tau=1.0, device=device)
    sharp = temperature_pass(model, paths, probe, tau=0.0, device=device)

    assert not np.array_equal(sharp.positions, warm.positions)
    assert not np.array_equal(sharp.reference.rssm_position, warm.reference.rssm_position)
    for name in ("positions_real", "positions_at_context", "true_positions", "true_at_context",
                 "true_embedding_displacement"):
        np.testing.assert_array_equal(getattr(sharp, name), getattr(warm, name), err_msg=name)
    for name in ("floor_position", "persistence_position", "floor_angle", "persistence_angle"):
        np.testing.assert_array_equal(
            getattr(sharp.reference, name), getattr(warm.reference, name), err_msg=name
        )


@pytest.mark.parametrize("device", DEVICES, ids=[d.type for d in DEVICES])
def test_the_noise_reference_is_drawn_at_the_rollouts_temperature(tmp_path, device):
    """It measures what sampling alone produces; at tau = 0 sampling produces
    nothing, so the second draw is the canonical one bitwise and the reference
    collapses to zero. At tau = 1 it does not."""
    model, paths, probe = _two_episode_rig(tmp_path, device)
    warm = temperature_pass(model, paths, probe, tau=1.0, device=device)
    sharp = temperature_pass(model, paths, probe, tau=0.0, device=device)
    assert (warm.noise_embedding > 0).any()
    assert sharp.noise_bitwise_real.all(), "at tau = 0 two draws must be the same trajectory"
    np.testing.assert_array_equal(sharp.noise_embedding, np.zeros_like(sharp.noise_embedding))
    assert sharp.noise_stream_restored.all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_diagnostics.py -q -k "rollout_temperature or sharper_rollout or noise_reference_is_drawn or bitwise_what_it_is_without"`
Expected: FAIL — `_diagnose() got an unexpected keyword argument 'rollout_temperature'`.

- [ ] **Step 3: Thread it through the pass**

In `src/mbfps/eval/diagnostics.py`, `_noise_reference` gains a temperature and passes it to its `imagine` call:

```python
def _noise_reference(model, handle: "_Window", tail: dict, temperature: float | None = None) -> torch.Tensor:
```

(inside, the `model.rssm.imagine(handle.horizon_actions, handle.state)` call becomes `model.rssm.imagine(handle.horizon_actions, handle.state, temperature=temperature)`).

`_diagnose`'s signature gains, after `keep_latents`:

```python
    rollout_temperature: float | None = None,
```

with this appended to its docstring:

```
    `rollout_temperature` (M3h) sharpens the CANONICAL rollout and the noise
    reference and nothing else: the context filter, the floor's `observe`, the
    persistence anchor and the probe stay exactly what the ladder and the gate
    scored. The noise reference moves with the rollout because it measures
    what sampling alone produces, which is a statement about the sampler in
    use.

    None -- the default -- means the MODEL'S OWN `cfg.sample_temperature`, and
    the keyword is then not passed to `imagine` at all, so the default path is
    the call it has always been. That default is not a synonym for 1.0: a
    model retrained at another temperature must roll out at the temperature it
    was trained at, and hardcoding 1.0 here would evaluate it with a sampler
    its training never saw -- the mis-evaluation the checkpoint payload's
    temperature exists to catch.
```

and in the loop the two calls become:

```python
            # The keyword is passed only when an override was asked for, so
            # the default path is the call it has always been -- which is what
            # keeps `imagine`'s stand-ins in the suite valid -- and a model
            # trained at its own temperature rolls out at that temperature
            # rather than at a default someone chose for it.
            imagined = (
                model.rssm.imagine(actions[:, context:], state)
                if rollout_temperature is None
                else model.rssm.imagine(
                    actions[:, context:], state, temperature=rollout_temperature
                )
            )
```

```python
                noise_latent = _noise_reference(model, handle, tail, rollout_temperature)
```

- [ ] **Step 4: Thread it through `reference_trajectories`**

```python
    feature_backbone,
    keep_latents: bool = False,
    rollout_temperature: float | None = None,
) -> Trajectories:
```

appending to its docstring: `` `rollout_temperature` (M3h) is passed to the pass unchanged; None, the default, leaves the model sampling at its own temperature and every field bitwise what it is without the argument. `` — and the `_diagnose(...)` call gains `rollout_temperature=rollout_temperature`.

- [ ] **Step 5: Run the new tests, then the whole file**

Run: `.venv/bin/python -m pytest tests/eval/test_diagnostics.py -q`
Expected: every test passes on cpu and mps — the M3d, M3g and the new pins together (147 at the time of writing).

- [ ] **Step 6: Run the consumers**

Run: `.venv/bin/python -m pytest tests/eval/test_trust_horizon_script.py tests/eval/test_split_gap_script.py tests/eval/test_stage_decomposition_script.py -q`
Expected: PASS unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/mbfps/eval/diagnostics.py tests/eval/test_diagnostics.py
git commit -m "feat: a rollout temperature on the canonical pass -- the imagination and its noise reference sharpen together and the floor, the persistence anchor and the probe do not move

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `mbfps/eval/sharper.py` — the constants and Reading N

**Files:**
- Create: `src/mbfps/eval/sharper.py`
- Test: `tests/eval/test_sharper.py`

**Interfaces:**
- Consumes: `mbfps.eval.split_gap.{DECISION_H, StratumContrast, clears, fmt_z}`.
- Produces: `TAU_GRID = (1.0, 0.7, 0.5, 0.3, 0.0)`, `TAU_CANDIDATES = (0.7, 0.5, 0.3)`, `REFERENCE_TAU = 1.0`, `SWEEP_FAMILY = 9`, `RETRAIN_FAMILY = 6`, `SEEDS_REQUIRED = 2`, `ARMS_REQUIRED = 2`, `STEPS = 20000`, `IDENTITY_STEPS = 500`, `REPORTED_H = (5, 15, 45)`, `DECISION_H` (re-exported); `TauInputs(free, per_seed)`, `SweepArm(taus: dict[float, TauInputs])`, `SweepInputs(arms, z_fam, h)`, `SweepStatus`, `ArmTau(arm, tau, clears_up, clears_down, seeds_up, seeds_down, seeds_total)`, `SweepReading(arms_clearing, tau_star, status, rule, cells)`, `reading_noise(inputs) -> SweepReading`, `format_reading_noise(reading, inputs) -> str`. Task 5 appends Reading M to the same module.

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_sharper.py`:

```python
"""mbfps.eval.sharper, the pure half of M3h: Reading N (is the rollout
noise-limited, and at which temperature) and, from Task 5, Reading M (does
training it sharper help).

Every contrast is fabricated and every expected status is typed by hand, so a
rule that changes has to break a named test rather than agree with itself. The
contrast type is `split_gap.StratumContrast`, the same reduction of a pooled
contrast Readings G, T and S are decided on.
"""

import numpy as np
import pytest

from mbfps.eval.split_gap import StratumContrast
from mbfps.eval.sharper import (
    ARMS_REQUIRED,
    DECISION_H,
    IDENTITY_STEPS,
    REFERENCE_TAU,
    REPORTED_H,
    SEEDS_REQUIRED,
    STEPS,
    SWEEP_FAMILY,
    RETRAIN_FAMILY,
    TAU_CANDIDATES,
    TAU_GRID,
    ArmTau,
    SweepArm,
    SweepInputs,
    SweepReading,
    SweepStatus,
    TauInputs,
    format_reading_noise,
    reading_noise,
)

Z_FAM = 3.06


def c(z: float, estimate: float | None = None) -> StratumContrast:
    if estimate is None:
        estimate = float("nan") if not np.isfinite(z) else 0.01 * z
    return StratumContrast(estimate=estimate, se=0.01, z=z, clusters=24)


def tau_inputs(pooled: float, seeds=None) -> TauInputs:
    """One (arm, tau): the pooled contrast and its three per-seed leaves,
    which default to the pooled z so a pooled clear replicates by default."""
    seeds = (pooled, pooled, pooled) if seeds is None else seeds
    return TauInputs(free=c(pooled), per_seed={i: c(z) for i, z in enumerate(seeds)})


def arm(**by_tau) -> SweepArm:
    """`arm(t07=..., t05=..., t03=..., t00=...)` -- keyword spellings of the
    grid, so a test reads as the table does."""
    spelling = {"t07": 0.7, "t05": 0.5, "t03": 0.3, "t00": 0.0}
    taus = {spelling[k]: (v if isinstance(v, TauInputs) else tau_inputs(v)) for k, v in by_tau.items()}
    for tau in TAU_CANDIDATES:
        taus.setdefault(tau, tau_inputs(0.0))
    taus.setdefault(0.0, tau_inputs(0.0))
    return SweepArm(taus=taus)


def read(**arms) -> SweepReading:
    return reading_noise(SweepInputs(arms=arms, z_fam=Z_FAM, h=15))


def test_the_pre_registered_constants_are_the_specs():
    assert TAU_GRID == (1.0, 0.7, 0.5, 0.3, 0.0)
    assert TAU_CANDIDATES == (0.7, 0.5, 0.3)
    from mbfps.models.rssm import SAMPLE_TEMPERATURE

    assert REFERENCE_TAU == SAMPLE_TEMPERATURE == 1.0 and REFERENCE_TAU not in TAU_CANDIDATES
    assert 0.0 not in TAU_CANDIDATES, "argmax is an endpoint, never a candidate"
    assert set(TAU_CANDIDATES) | {REFERENCE_TAU, 0.0} == set(TAU_GRID)
    assert SWEEP_FAMILY == len(TAU_CANDIDATES) * 3 == 9
    assert RETRAIN_FAMILY == 6 and SEEDS_REQUIRED == 2 and ARMS_REQUIRED == 2
    assert DECISION_H == 15 and REPORTED_H == (5, 15, 45)
    assert STEPS == 20000 and IDENTITY_STEPS == 500


def test_two_arms_clearing_at_one_tau_read_noise_limited_at_that_tau():
    reading = read(pixel_ae=arm(t05=5.0), frozen_ssl=arm(t05=4.0), random_vit=arm())
    assert isinstance(reading, SweepReading)
    assert reading.status is SweepStatus.NOISE_LIMITED
    assert reading.tau_star == 0.5
    assert reading.arms_clearing[0.5] == ("pixel_ae", "frozen_ssl")
    assert "tau=0.5" in reading.rule and "2 of 3 arms" in reading.rule
    cell = reading.cells[("pixel_ae", 0.5)]
    assert isinstance(cell, ArmTau) and cell.clears_up and cell.seeds_up == 3


def test_one_arm_alone_is_not_enough_and_the_sentence_names_it():
    reading = read(pixel_ae=arm(t05=9.0), frozen_ssl=arm(), random_vit=arm())
    assert reading.status is SweepStatus.NOT_NOISE_LIMITED
    assert reading.tau_star is None
    assert "pixel_ae" in reading.rule and "1 of 3 arms" in reading.rule


def test_a_pooled_clear_that_replicates_in_one_seed_does_not_count_for_its_arm():
    thin = tau_inputs(5.0, seeds=(5.0, 1.0, 1.0))
    reading = read(pixel_ae=arm(t05=thin), frozen_ssl=arm(t05=5.0), random_vit=arm())
    assert reading.status is SweepStatus.NOT_NOISE_LIMITED
    assert reading.cells[("pixel_ae", 0.5)].seeds_up == 1
    assert not reading.cells[("pixel_ae", 0.5)].counts
    assert reading.cells[("frozen_ssl", 0.5)].counts


def test_the_largest_pooled_contrast_among_the_clearing_taus_wins():
    small, large = tau_inputs(4.0), tau_inputs(9.0)
    assert small.free.z == 4.0 and large.free.z == 9.0
    reading = read(
        pixel_ae=arm(t07=small, t03=large),
        frozen_ssl=arm(t07=small, t03=large),
        random_vit=arm(),
    )
    assert reading.status is SweepStatus.NOISE_LIMITED
    assert reading.tau_star == 0.3, "0.3 pools the larger contrast"


def test_a_tie_between_two_clearing_taus_goes_to_the_larger_temperature():
    """The milder intervention, pre-registered: an equal estimate at 0.7 and
    0.3 retrains at 0.7."""
    tied = {"t07": tau_inputs(5.0), "t03": tau_inputs(5.0)}
    reading = read(pixel_ae=arm(**tied), frozen_ssl=arm(**tied), random_vit=arm())
    assert reading.status is SweepStatus.NOISE_LIMITED and reading.tau_star == 0.7


def test_two_arms_clearing_negatively_read_sharper_worse():
    reading = read(pixel_ae=arm(t03=-6.0), frozen_ssl=arm(t03=-5.0), random_vit=arm())
    assert reading.status is SweepStatus.SHARPER_WORSE
    assert reading.tau_star is None
    assert "tau=0.3" in reading.rule and "-" in reading.rule


def test_a_positive_clear_in_two_arms_outranks_a_negative_clear_elsewhere():
    """Precedence: the sweep exists to find a helpful temperature, and a
    different temperature hurting does not withdraw one that helps."""
    reading = read(
        pixel_ae=arm(t07=5.0, t03=-6.0), frozen_ssl=arm(t07=5.0, t03=-6.0), random_vit=arm()
    )
    assert reading.status is SweepStatus.NOISE_LIMITED and reading.tau_star == 0.7


def test_the_endpoint_never_decides_anything():
    """tau = 0 may be in the inputs and is reported; it cannot make the status
    NOISE_LIMITED and cannot be tau_star."""
    reading = read(pixel_ae=arm(t00=9.0), frozen_ssl=arm(t00=9.0), random_vit=arm(t00=9.0))
    assert reading.status is SweepStatus.NOT_NOISE_LIMITED and reading.tau_star is None
    assert reading.cells[("pixel_ae", 0.0)].clears_up, "still reported"


def test_z_exactly_at_the_bar_does_not_clear_and_a_nan_never_does():
    at_bar = read(pixel_ae=arm(t05=Z_FAM), frozen_ssl=arm(t05=Z_FAM), random_vit=arm())
    assert at_bar.status is SweepStatus.NOT_NOISE_LIMITED
    nan = read(
        pixel_ae=arm(t05=tau_inputs(float("nan"))),
        frozen_ssl=arm(t05=tau_inputs(float("nan"))),
        random_vit=arm(),
    )
    assert nan.status is SweepStatus.NOT_NOISE_LIMITED
    infinite = read(pixel_ae=arm(t05=float("inf")), frozen_ssl=arm(t05=float("inf")), random_vit=arm())
    assert infinite.status is SweepStatus.NOT_NOISE_LIMITED


def test_format_reading_noise_prints_every_cell_and_the_verdict():
    inputs = SweepInputs(
        arms={"pixel_ae": arm(t05=5.0), "frozen_ssl": arm(t05=4.0), "random_vit": arm()},
        z_fam=Z_FAM, h=15,
    )
    text = format_reading_noise(reading_noise(inputs), inputs)
    assert text.startswith("--- Reading N: is the rollout noise-limited at h=15")
    assert "z_fam = 3.06" in text
    for tau in TAU_GRID[1:]:
        assert f"{tau:.1f}" in text
    assert "verdict: NOISE LIMITED" in text and "tau=0.5" in text
    assert text.endswith("\n")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_sharper.py -q`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'mbfps.eval.sharper'`.

- [ ] **Step 3: Write the module**

Create `src/mbfps/eval/sharper.py`:

```python
"""The sharper latent's pure rules -- milestone M3h.

Two readings, neither of which touches torch or a file:

  * spec 3.2 -- Reading N: over the nine shipped cells, does sampling the
    prior more sharply at rollout time keep MORE of the moved draws ahead of
    embedding-space persistence at h = 15? The status gates the retrain and,
    when it passes, names the temperature the retrain runs at.
  * spec 3.3 -- Reading M: does a model TRAINED at that temperature roll out
    better than the one it replaces?

The contrasts arrive as `split_gap.StratumContrast` (estimate, se, z,
clusters) -- the same reduction Readings G, T and S are decided on -- built
by `scripts/sharper_latent.py` from `pooling.paired_contrast`.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from mbfps.eval.split_gap import DECISION_H, StratumContrast, clears, fmt_z

TAU_GRID: tuple[float, ...] = (1.0, 0.7, 0.5, 0.3, 0.0)
"""Every rollout temperature the sweep evaluates, reference first."""
TAU_CANDIDATES: tuple[float, ...] = (0.7, 0.5, 0.3)
"""The temperatures the retrain may run at. 0.0 is an ENDPOINT, reported and
never a candidate: M3a measured the argmax collapsing the imagined trajectory
to 3 distinct latents out of 45 at roughly four times the position error, and
a model cannot be trained toward a sampler that draws nothing."""
REFERENCE_TAU: float = 1.0
"""What every M3b-M3g artefact was trained and evaluated at; the contrast's
control and the pass the self-check is taken on."""
STEPS: int = 20000
"""The retrain's length -- M3c's, so the only difference is the sampler."""
IDENTITY_STEPS: int = 500
"""The tau = 1.0 retrain that must reproduce the M3c loss prefix exactly."""
# DECISION_H is M3e's, imported: one decision horizon (15) across the study.
REPORTED_H: tuple[int, ...] = (5, 15, 45)
SWEEP_FAMILY: int = 9
"""Reading N's family: three arms x three candidate temperatures (spec 3.2)."""
RETRAIN_FAMILY: int = 6
"""Reading M's family: three arms x two channels (spec 3.3)."""
SEEDS_REQUIRED: int = 2
ARMS_REQUIRED: int = 2
"""Spec 3.2: a temperature counts when it clears in at least two ARMS. One arm
alone is a result about that arm, not about the sampler, and retraining nine
cells on it would be spending fifteen hours on a single cell's evidence."""


# ---------------------------------------------------------------------------
# Reading N -- is the rollout noise-limited? (spec 3.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TauInputs:
    """One (arm, temperature): the pooled free-channel contrast against
    tau = 1.0 and the same contrast within each seed alone."""

    free: StratumContrast
    per_seed: dict[int, StratumContrast]


@dataclass(frozen=True)
class SweepArm:
    """One arm's inputs, keyed by temperature. The reference temperature is
    not a key: it is what every contrast is against."""

    taus: dict[float, TauInputs]


@dataclass(frozen=True)
class SweepInputs:
    arms: dict[str, SweepArm]
    z_fam: float
    h: int


class SweepStatus(str, Enum):
    """Spec 3.2's three outcomes."""

    NOISE_LIMITED = "noise limited"
    SHARPER_WORSE = "sharper worse"
    NOT_NOISE_LIMITED = "not noise limited"


@dataclass(frozen=True)
class ArmTau:
    """One cell of the sweep table: whether it cleared, which way, and in how
    many seeds. `counts` is the conjunction the arm tally is taken over --
    cleared pooled AND replicated -- so the table and the rule cannot
    disagree about what a cell contributed."""

    arm: str
    tau: float
    clears_up: bool
    clears_down: bool
    seeds_up: int
    seeds_down: int
    seeds_total: int

    @property
    def counts(self) -> bool:
        return self.clears_up and self.seeds_up >= SEEDS_REQUIRED

    @property
    def counts_against(self) -> bool:
        return self.clears_down and self.seeds_down >= SEEDS_REQUIRED


@dataclass(frozen=True)
class SweepReading:
    """`arms_clearing` maps every temperature to the arms that counted for it,
    in the caller's arm order; `cells` is the whole table, endpoint included."""

    cells: dict[tuple[str, float], ArmTau]
    arms_clearing: dict[float, tuple[str, ...]]
    arms_against: dict[float, tuple[str, ...]]
    tau_star: float | None
    status: SweepStatus
    rule: str
    h: int
    z_fam: float


def _cell(arm: str, tau: float, inputs: TauInputs, z_fam: float) -> ArmTau:
    leaves = list(inputs.per_seed.values())
    return ArmTau(
        arm=arm, tau=tau,
        clears_up=clears(inputs.free.z, z_fam),
        clears_down=clears(-inputs.free.z, z_fam),
        seeds_up=sum(1 for leaf in leaves if clears(leaf.z, z_fam)),
        seeds_down=sum(1 for leaf in leaves if clears(-leaf.z, z_fam)),
        seeds_total=len(leaves),
    )


def reading_noise(inputs: SweepInputs) -> SweepReading:
    """Spec 3.2, in one pass over the table: build every cell, tally the arms
    per candidate temperature, and read the status. `tau_star` is the
    candidate with the largest POOLED estimate among those counting in at
    least `ARMS_REQUIRED` arms, ties to the larger temperature."""
    cells = {
        (arm, tau): _cell(arm, tau, tau_inputs, inputs.z_fam)
        for arm, sweep_arm in inputs.arms.items()
        for tau, tau_inputs in sweep_arm.taus.items()
    }
    arms_clearing: dict[float, tuple[str, ...]] = {}
    arms_against: dict[float, tuple[str, ...]] = {}
    for tau in TAU_GRID:
        if tau == REFERENCE_TAU:
            continue
        arms_clearing[tau] = tuple(
            arm for arm in inputs.arms
            if (arm, tau) in cells and cells[(arm, tau)].counts
        )
        arms_against[tau] = tuple(
            arm for arm in inputs.arms
            if (arm, tau) in cells and cells[(arm, tau)].counts_against
        )
    qualifying = [t for t in TAU_CANDIDATES if len(arms_clearing.get(t, ())) >= ARMS_REQUIRED]
    against = [t for t in TAU_CANDIDATES if len(arms_against.get(t, ())) >= ARMS_REQUIRED]
    bar = f"{inputs.z_fam:.2f}"
    if qualifying:
        # Ties to the LARGER temperature: the milder intervention.
        def pooled(tau: float) -> float:
            estimates = [
                inputs.arms[arm].taus[tau].free.estimate for arm in arms_clearing[tau]
            ]
            return float(np.mean(estimates))

        best = max(pooled(t) for t in qualifying)
        tau_star = max(t for t in qualifying if pooled(t) == best)
        names = ", ".join(arms_clearing[tau_star])
        rule = (
            f"tau={tau_star:.1f} clears +{bar} pooled and in >= {SEEDS_REQUIRED} seeds in "
            f"{len(arms_clearing[tau_star])} of {len(inputs.arms)} arms ({names}); "
            f"pooled estimate {fmt_z(best, '+.4f')}"
        )
        return SweepReading(
            cells=cells, arms_clearing=arms_clearing, arms_against=arms_against,
            tau_star=tau_star, status=SweepStatus.NOISE_LIMITED, rule=rule,
            h=inputs.h, z_fam=inputs.z_fam,
        )
    if against:
        tau = against[0]
        names = ", ".join(arms_against[tau])
        rule = (
            f"no candidate temperature clears +{bar} in {ARMS_REQUIRED} arms, and tau={tau:.1f} "
            f"clears -{bar} in {len(arms_against[tau])} of {len(inputs.arms)} arms ({names})"
        )
        return SweepReading(
            cells=cells, arms_clearing=arms_clearing, arms_against=arms_against,
            tau_star=None, status=SweepStatus.SHARPER_WORSE, rule=rule,
            h=inputs.h, z_fam=inputs.z_fam,
        )
    best_tau = max(TAU_CANDIDATES, key=lambda t: len(arms_clearing.get(t, ())))
    counted = arms_clearing.get(best_tau, ())
    detail = (
        f"the most any candidate reaches is tau={best_tau:.1f} in {len(counted)} of "
        f"{len(inputs.arms)} arms ({', '.join(counted)})" if counted
        else f"no candidate temperature clears +{bar} pooled and in >= {SEEDS_REQUIRED} seeds in any arm"
    )
    rule = f"{ARMS_REQUIRED} arms required; {detail}"
    return SweepReading(
        cells=cells, arms_clearing=arms_clearing, arms_against=arms_against,
        tau_star=None, status=SweepStatus.NOT_NOISE_LIMITED, rule=rule,
        h=inputs.h, z_fam=inputs.z_fam,
    )


def format_reading_noise(reading: SweepReading, inputs: SweepInputs) -> str:
    """The contrast table over the grid and the verdict, in `ladder.txt`'s style."""
    lines = [
        f"--- Reading N: is the rollout noise-limited at h={reading.h} (S_free(h) at tau minus "
        f"at tau={REFERENCE_TAU}, paired on the val windows); z_fam = {reading.z_fam:.2f} ---",
        f"  {'arm':<12}{'tau':>5}{'estimate':>11}{'se':>9}{'z':>8}{'seeds':>7}  clears  counts",
    ]
    for arm, sweep_arm in inputs.arms.items():
        for tau in TAU_GRID:
            if tau == REFERENCE_TAU or tau not in sweep_arm.taus:
                continue
            cell = reading.cells[(arm, tau)]
            contrast = sweep_arm.taus[tau].free
            direction = "up" if cell.clears_up else ("down" if cell.clears_down else "no")
            endpoint = "" if tau in TAU_CANDIDATES else "  (endpoint, never a candidate)"
            lines.append(
                f"  {arm:<12}{tau:>5.1f}{fmt_z(contrast.estimate, '+.4f'):>11}"
                f"{fmt_z(contrast.se, '.4f'):>9}{fmt_z(contrast.z):>8}"
                f"{f'{cell.seeds_up}/{cell.seeds_total}':>7}  {direction:<6}  "
                f"{'yes' if cell.counts else 'no'}{endpoint}"
            )
    lines.append(
        f"  verdict: {reading.status.name.replace('_', ' ')} -- decided by: {reading.rule}"
    )
    if reading.tau_star is not None:
        lines.append(f"  the retrain runs at SAMPLE_TEMPERATURE = {reading.tau_star:.1f}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_sharper.py -q`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/eval/sharper.py tests/eval/test_sharper.py
git commit -m "feat: Reading N -- the sweep's rule: a candidate temperature counts when it clears pooled and in two seeds in two arms, and tau* is the largest pooled contrast among those, ties to the milder temperature

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `sharper.py` — Reading M

**Files:**
- Modify: `src/mbfps/eval/sharper.py` (append)
- Test: `tests/eval/test_sharper.py` (append)

**Interfaces:**
- Consumes: Task 4's constants and `split_gap.{StratumContrast, clears, fmt_z}`.
- Produces: `RetrainStatus` (`UNRESOLVED_PROBE`, `SHARPER_BETTER`, `SHARPER_WORSE`, `NO_DIFFERENCE`), `RetrainArm(free, probe, per_seed)`, `RetrainInputs(arms, z_fam, h, tau)`, `RetrainArmReading(arm, status, rule, free_clears_up, free_clears_down, probe_clears_up, probe_clears_down, seeds_up, seeds_down, seeds_total)`, `RetrainReading(arms, h, z_fam, tau)`, `reading_sharper(inputs)`, `format_reading_sharper(reading, inputs)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_sharper.py`:

```python


# ---------------------------------------------------------------------------
# Reading M (spec 3.3): does a model TRAINED at tau* roll out better than the
# one it replaces? The same four-status shape as M3f's Reading T, with the
# probe channel as the control and the single-seed clause built in.
# ---------------------------------------------------------------------------

from mbfps.eval.sharper import (  # noqa: E402
    RetrainArm,
    RetrainArmReading,
    RetrainInputs,
    RetrainReading,
    RetrainStatus,
    format_reading_sharper,
    reading_sharper,
)

M_Z_FAM = 2.89


def retrain_arm(free=5.0, probe=5.0, seeds=None) -> RetrainArm:
    seeds = (free, free, free) if seeds is None else seeds
    return RetrainArm(
        free=c(free), probe=c(probe),
        per_seed={i: RetrainArm(free=c(z), probe=c(probe), per_seed=None) for i, z in enumerate(seeds)},
    )


def read_m(**arms) -> RetrainReading:
    return reading_sharper(RetrainInputs(arms=arms, z_fam=M_Z_FAM, h=15, tau=0.5))


def test_a_positive_clear_replicated_in_two_seeds_reads_sharper_better():
    reading = read_m(pixel_ae=retrain_arm(free=5.0, seeds=(5.0, 5.0, 1.0)))
    r = reading.arms["pixel_ae"]
    assert isinstance(r, RetrainArmReading) and r.status is RetrainStatus.SHARPER_BETTER
    assert r.seeds_up == 2 and r.seeds_total == 3
    assert "2 of 3 seeds" in r.rule and "+5.00" in r.rule
    assert reading.tau == 0.5 and reading.h == 15


def test_a_negative_clear_replicated_reads_sharper_worse():
    r = read_m(a=retrain_arm(free=-5.0, probe=-5.0)).arms["a"]
    assert r.status is RetrainStatus.SHARPER_WORSE
    assert r.rule.startswith("M_free z -5.00 < -2.89 pooled and in 3 of 3 seeds")


def test_channels_clearing_with_opposite_signs_are_unresolved_through_the_probe():
    r = read_m(a=retrain_arm(free=5.0, probe=-5.0)).arms["a"]
    assert r.status is RetrainStatus.UNRESOLVED_PROBE
    assert "opposite signs" in r.rule


def test_a_pooled_clear_that_does_not_replicate_reads_no_difference_and_says_why():
    r = read_m(a=retrain_arm(free=5.0, seeds=(5.0, 1.0, 1.0))).arms["a"]
    assert r.status is RetrainStatus.NO_DIFFERENCE
    assert "only 1 of 3 seeds" in r.rule and f">= {SEEDS_REQUIRED}" in r.rule


def test_no_clear_at_all_reads_no_difference():
    r = read_m(a=retrain_arm(free=1.0, probe=1.0)).arms["a"]
    assert r.status is RetrainStatus.NO_DIFFERENCE
    assert r.rule == "M_free z +1.00 does not clear +-2.89"


def test_a_single_seed_leaf_read_alone_has_a_vacuous_replication_clause():
    leaf = RetrainArm(free=c(5.0), probe=c(5.0), per_seed=None)
    r = read_m(a=leaf).arms["a"]
    assert r.status is RetrainStatus.SHARPER_BETTER
    assert "this seed alone" in r.rule and r.seeds_total == 1


def test_z_exactly_at_the_bar_does_not_clear_and_a_nan_never_does_on_the_retrain():
    # Named apart from Reading N's test of the same rule: two module-level
    # functions of one name shadow each other and one silently stops running.
    assert read_m(a=retrain_arm(free=M_Z_FAM)).arms["a"].status is RetrainStatus.NO_DIFFERENCE
    assert read_m(a=retrain_arm(free=float("nan"))).arms["a"].status is RetrainStatus.NO_DIFFERENCE
    assert read_m(a=retrain_arm(free=float("inf"))).arms["a"].status is RetrainStatus.NO_DIFFERENCE


def test_arms_are_read_independently_and_in_the_callers_order():
    reading = read_m(random_vit=retrain_arm(free=-5.0, probe=-5.0), pixel_ae=retrain_arm())
    assert list(reading.arms) == ["random_vit", "pixel_ae"]
    assert reading.arms["random_vit"].status is RetrainStatus.SHARPER_WORSE
    assert reading.arms["pixel_ae"].status is RetrainStatus.SHARPER_BETTER


def test_format_reading_sharper_prints_both_channels_and_the_verdicts():
    inputs = RetrainInputs(
        arms={"pixel_ae": retrain_arm(free=5.0), "frozen_ssl": retrain_arm(free=1.0, probe=1.0)},
        z_fam=M_Z_FAM, h=15, tau=0.5,
    )
    text = format_reading_sharper(reading_sharper(inputs), inputs)
    assert text.startswith("--- Reading M: the retrain at tau=0.5 against the M3c cells at h=15")
    assert "z_fam = 2.89" in text
    assert text.count("free") >= 2 and text.count("probe") >= 2
    assert "verdict: pixel_ae    SHARPER BETTER" in text
    assert "verdict: frozen_ssl  NO DIFFERENCE" in text
    assert text.endswith("\n")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_sharper.py -q`
Expected: FAIL at the second import block — `ImportError: cannot import name 'RetrainArm'`.

- [ ] **Step 3: Append Reading M to the module**

```python


# ---------------------------------------------------------------------------
# Reading M -- does training it sharper help? (spec 3.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrainArm:
    """One arm's paired contrasts, `M3h at tau* minus M3c at 1.0`, on both
    channels, and the same within each seed alone (leaves carry
    `per_seed=None`)."""

    free: StratumContrast
    probe: StratumContrast
    per_seed: "dict[int, RetrainArm] | None"


@dataclass(frozen=True)
class RetrainInputs:
    arms: dict[str, RetrainArm]
    z_fam: float
    h: int
    tau: float


class RetrainStatus(str, Enum):
    """Spec 3.3's four outcomes, in the table's order of precedence."""

    UNRESOLVED_PROBE = "unresolved through the probe"
    SHARPER_BETTER = "sharper better"
    SHARPER_WORSE = "sharper worse"
    NO_DIFFERENCE = "no difference"


@dataclass(frozen=True)
class RetrainArmReading:
    arm: str
    status: RetrainStatus
    rule: str
    free_clears_up: bool
    free_clears_down: bool
    probe_clears_up: bool
    probe_clears_down: bool
    seeds_up: int
    seeds_down: int
    seeds_total: int


@dataclass(frozen=True)
class RetrainReading:
    arms: dict[str, RetrainArmReading]
    h: int
    z_fam: float
    tau: float


def _retrain_arm_reading(arm: str, a: RetrainArm, z_fam: float) -> RetrainArmReading:
    """Spec 3.3's rules in the table's precedence, each status carrying the
    sentence that decided it."""
    free_up, free_down = clears(a.free.z, z_fam), clears(-a.free.z, z_fam)
    probe_up, probe_down = clears(a.probe.z, z_fam), clears(-a.probe.z, z_fam)
    if a.per_seed is None:
        # A single-seed leaf, read alone: there is nothing to replicate across.
        seeds_up, seeds_down, seeds_total = int(free_up), int(free_down), 1
        replicated_up = replicated_down = True
        up_words = down_words = "this seed alone"
    else:
        seeds_up = sum(1 for leaf in a.per_seed.values() if clears(leaf.free.z, z_fam))
        seeds_down = sum(1 for leaf in a.per_seed.values() if clears(-leaf.free.z, z_fam))
        seeds_total = len(a.per_seed)
        replicated_up = seeds_up >= SEEDS_REQUIRED
        replicated_down = seeds_down >= SEEDS_REQUIRED
        up_words = f"{seeds_up} of {seeds_total} seeds"
        down_words = f"{seeds_down} of {seeds_total} seeds"
    fz, pz, bar = fmt_z(a.free.z), fmt_z(a.probe.z), f"{z_fam:.2f}"
    if (free_up and probe_down) or (free_down and probe_up):
        status = RetrainStatus.UNRESOLVED_PROBE
        rule = f"M_free z {fz} and M_probe z {pz} both clear +-{bar} with opposite signs"
    elif free_up and replicated_up:
        status = RetrainStatus.SHARPER_BETTER
        rule = f"M_free z {fz} > {bar} pooled and in {up_words}"
    elif free_down and replicated_down:
        status = RetrainStatus.SHARPER_WORSE
        rule = f"M_free z {fz} < -{bar} pooled and in {down_words}"
    elif free_up or free_down:
        words = up_words if free_up else down_words
        sign = "+" if free_up else "-"
        status = RetrainStatus.NO_DIFFERENCE
        rule = (f"M_free z {fz} clears {sign}{bar} pooled but in only {words} "
                f"(>= {SEEDS_REQUIRED} required)")
    else:
        status = RetrainStatus.NO_DIFFERENCE
        rule = f"M_free z {fz} does not clear +-{bar}"
    return RetrainArmReading(
        arm=arm, status=status, rule=rule,
        free_clears_up=free_up, free_clears_down=free_down,
        probe_clears_up=probe_up, probe_clears_down=probe_down,
        seeds_up=seeds_up, seeds_down=seeds_down, seeds_total=seeds_total,
    )


def reading_sharper(inputs: RetrainInputs) -> RetrainReading:
    """One reading per arm, in the caller's order; arms never read each other
    (spec 4: no ranking)."""
    return RetrainReading(
        arms={
            arm: _retrain_arm_reading(arm, a, inputs.z_fam) for arm, a in inputs.arms.items()
        },
        h=inputs.h, z_fam=inputs.z_fam, tau=inputs.tau,
    )


def format_reading_sharper(reading: RetrainReading, inputs: RetrainInputs) -> str:
    """The contrast table and the verdict lines, in `ladder.txt`'s style."""
    lines = [
        f"--- Reading M: the retrain at tau={reading.tau:.1f} against the M3c cells at "
        f"h={reading.h} (paired on the val windows); z_fam = {reading.z_fam:.2f} ---",
        f"  {'arm':<12}{'channel':<9}{'estimate':>10}{'se':>9}{'z':>8}  clears",
    ]
    for arm, a in inputs.arms.items():
        for channel, contrast in (("free", a.free), ("probe", a.probe)):
            verdict = "yes" if clears(abs(contrast.z), reading.z_fam) else "no"
            lines.append(
                f"  {arm:<12}{channel:<9}{fmt_z(contrast.estimate, '+.4f'):>10}"
                f"{fmt_z(contrast.se, '.4f'):>9}{fmt_z(contrast.z):>8}  {verdict}"
            )
    for arm, r in reading.arms.items():
        lines.append(
            f"  verdict: {arm:<12}{r.status.name.replace('_', ' ')} -- decided by: {r.rule}"
        )
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_sharper.py -q`
Expected: 20 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/eval/sharper.py tests/eval/test_sharper.py
git commit -m "feat: Reading M -- the retrain's paired contrast against the M3c cells with the probe channel as the control, four statuses in precedence and the single-seed clause built in

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: `scripts/sharper_latent.py` — the sweep

**Files:**
- Create: `scripts/sharper_latent.py`
- Test: `tests/eval/test_sharper_latent_script.py`

**Interfaces:**
- Consumes: Task 3's `reference_trajectories(..., rollout_temperature=τ)`; Task 4's constants; `trust_horizon.{Cell, CellMissing, load_cell, self_check, prepare_cell, EXIT_*}` and `split_gap.stratum_summary` by path; `study.{SPLIT_SEED, git_sha, load_record, write_record}`.
- Produces: `EXIT_NOT_NOISE_LIMITED = 35`, `EXIT_TEMPERATURE_MISMATCH = 36`, `EXIT_IDENTITY_CHECK_FAILED = 37`; `PHASES = ("sweep",)` (Task 7 extends it); `sweep_record_path(out, arm, seed)`; `tau_key(tau) -> str` / `tau_value(key) -> float` (the record's key idiom, `0.7 -> "tau70"`: `write_record` refuses a key containing '.'); `tau_entry(traj, record, horizon) -> dict`; `sweep_cell(args, cell, device, train, val) -> (status, record | None)`; `sweep_phase(args, cells, device, train, val) -> status`; `_parser()`, `main(argv=None, *, taus=TAU_GRID) -> int`.

The record, one per cell (`sweep_<arm>_seed<n>.json`), top-level keys exactly: `arm, seed, source, step, record_git_sha, context, horizon, decision_h, split_seed, device, torch_version, git_sha, taus (the grid, in order), episodes {val}, windows {total, episode, clusters}, self_check (the τ = 1.0 pass against the cell's diagnostic), entries {<τ as `tau_key` spells it, e.g. `tau70`>: {tau, gate {gap_final, degenerate}, probe {selection_r2, measurable}, noise {curve, median}, displacement {curve, median}, summary}}, nonfinite`.

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_sharper_latent_script.py`:

```python
"""scripts/sharper_latent.py: the rollout-temperature sweep over shipped cells
(Task 6), then the gated retrain and its reading (Tasks 7-8).

The script is loaded by path. The REFERENCE is a real tiny cell on
`wide_buffer` -- thirty 40-step episodes, `run_job(..., steps=5, seq_len=4,
context=2, horizon=3, device="cpu")` -- with the diagnostic
`diagnose_dynamics.main` writes, exactly as the ladder and stage tests build
theirs. The sweep runs a two-value grid so a test is seconds rather than
minutes; `main` takes `taus` as a parameter with `TAU_GRID` as its default, so
a test never patches the pre-registration.

THE FIXTURE'S FACTS: six validation episodes, `window_starts(40, 2, 3)` cuts
eight windows per episode -- 48 windows over 6 clusters at every temperature;
the decision horizon clamps from 15 to the run's 3.
"""

import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pytest

from mbfps.eval.sharper import REFERENCE_TAU, TAU_GRID
from mbfps.eval.study import StudyJob, load_record, run_job

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script("sharper_latent")
diagnose = _load_script("diagnose_dynamics")
trust = _load_script("trust_horizon")

JOB = StudyJob("random_vit", 1)
JOB_KW = dict(steps=5, seq_len=4, context=2, horizon=3, device="cpu")
CONTEXT, HORIZON = JOB_KW["context"], JOB_KW["horizon"]
SWEEP_TAUS = (1.0, 0.0)
WINDOWS, CLUSTERS = 48, 6
RECORD = "result_random_vit_seed1.json"
DIAGNOSTIC = "diagnostic_random_vit_seed1.json"
SWEEP = "sweep_random_vit_seed1.json"

SWEEP_KEYS = {
    "arm", "seed", "source", "step", "record_git_sha", "context", "horizon", "decision_h",
    "split_seed", "device", "torch_version", "git_sha", "taus", "episodes", "windows",
    "self_check", "entries", "nonfinite",
}
ENTRY_KEYS = {"tau", "gate", "probe", "noise", "summary"}


@pytest.fixture
def reference(tmp_path, wide_buffer, capsys):
    """One real cell and the diagnostic the ladder writes for it."""
    out = tmp_path / "reference"
    run_job(JOB, wide_buffer, out, **JOB_KW)
    status = diagnose.main([
        "--out", str(out), "--data", str(wide_buffer.root), "--device", "cpu",
        "--arms", JOB.arm, "--seeds", str(JOB.seed),
        "--context", str(CONTEXT), "--horizon", str(HORIZON), "--ks", "1", str(HORIZON),
    ])
    assert status == diagnose.EXIT_OK, "the fixture's diagnostic was not written cleanly"
    capsys.readouterr()
    return types.SimpleNamespace(
        out=out, data=wide_buffer.root, sweep=tmp_path / "sweep", sharper=tmp_path / "sharper",
    )


def _argv(ref, *extra: str) -> list[str]:
    return [
        "--out", str(ref.sharper), "--sweep-out", str(ref.sweep), "--reference", str(ref.out),
        "--data", str(ref.data), "--device", "cpu", "--arms", JOB.arm, "--seeds", str(JOB.seed),
        *extra,
    ]


def _run(ref, *extra: str) -> int:
    return script.main(_argv(ref, *extra), taus=SWEEP_TAUS)


@pytest.fixture
def swept(reference, capsys):
    assert _run(reference, "--phase", "sweep") == script.EXIT_OK
    capsys.readouterr()
    return reference


def _doctor(path: Path, edit) -> None:
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


def _never_pass(*args, **kwargs):
    raise AssertionError("reference_trajectories ran; this refusal must come before any pass")


# ---------------------------------------------------------------------------
# The parser and the statuses.
# ---------------------------------------------------------------------------


def test_the_parser_defaults_are_the_specs():
    args = script._parser().parse_args([])
    assert args.out == Path("runs/m3h_sharper") and args.sweep_out == Path("runs/m3h_sweep")
    assert args.reference == Path("runs/m3_study_v2") and args.data == Path("data/my_way_home")
    assert args.device == "mps" and args.context is None and args.horizon is None
    assert args.arms == ["pixel_ae", "frozen_ssl", "random_vit"] and args.seeds == [0, 1, 2]
    assert args.phase == script.PHASES[-1] and args.figure is None


def test_out_equal_to_reference_or_to_the_sweep_is_argparses_own_usage_error(tmp_path):
    same = str(tmp_path / "study")
    for flag in ("--reference", "--sweep-out"):
        with pytest.raises(SystemExit) as raised:
            script.main(["--out", same, flag, same], taus=SWEEP_TAUS)
        assert raised.value.code == 2


def test_the_reused_statuses_are_trust_horizons_and_35_36_37_are_new():
    assert script.EXIT_OK == trust.EXIT_OK == 0
    assert script.EXIT_NO_CHECKPOINTS == trust.EXIT_NO_CHECKPOINTS == 11
    assert script.EXIT_SPLIT_MISMATCH == trust.EXIT_SPLIT_MISMATCH == 12
    assert script.EXIT_RECORD_MISMATCH == trust.EXIT_RECORD_MISMATCH == 14
    assert script.EXIT_SELF_CHECK_FAILED == trust.EXIT_SELF_CHECK_FAILED == 30
    assert script.EXIT_NOT_NOISE_LIMITED == 35
    assert script.EXIT_TEMPERATURE_MISMATCH == 36
    assert script.EXIT_IDENTITY_CHECK_FAILED == 37


# ---------------------------------------------------------------------------
# sweep.
# ---------------------------------------------------------------------------


def test_a_missing_cell_is_exit_11_before_any_pass_runs(reference, monkeypatch, capsys):
    monkeypatch.setattr(script, "reference_trajectories", _never_pass)
    assert _run(reference, "--phase", "sweep", "--seeds", "0") == script.EXIT_NO_CHECKPOINTS
    assert "NO CELL" in capsys.readouterr().out
    assert not reference.sweep.exists()


def test_the_sweep_writes_one_record_per_cell_holding_every_temperature(swept):
    record = load_record(swept.sweep / SWEEP)
    assert set(record) == SWEEP_KEYS
    assert record["taus"] == list(SWEEP_TAUS)
    assert record["context"] == CONTEXT and record["horizon"] == HORIZON
    assert record["decision_h"] == HORIZON  # 15 clamped to the run's horizon
    assert record["windows"] == {
        "total": WINDOWS, "episode": record["windows"]["episode"], "clusters": CLUSTERS,
    }
    assert len(record["windows"]["episode"]) == WINDOWS
    assert record["step"] == JOB_KW["steps"] and record["source"] == str(swept.out)
    assert set(record["entries"]) == {script.tau_key(t) for t in SWEEP_TAUS}
    for key, entry in record["entries"].items():
        assert set(entry) == ENTRY_KEYS
        assert entry["tau"] == script.tau_value(key)
        assert set(entry["gate"]) == {"gap_final", "degenerate"}
        assert set(entry["summary"]) >= {"windows", "curves", "band", "crossing", "margin", "survival", "counts"}
        assert set(entry["probe"]) == {"selection_r2", "measurable"}
        assert np.asarray(entry["noise"]["curve"]).shape == (HORIZON,)
        crossing = np.asarray(entry["summary"]["crossing"]["free"], dtype=float)
        assert crossing.shape == (WINDOWS,)


def test_the_reference_temperature_reproduces_the_cells_diagnostic_bitwise(swept):
    """THE anchor: at tau = 1.0 the pass must be the one the gate scored, or
    no temperature below it means anything. Exact, not close."""
    record = load_record(swept.sweep / SWEEP)
    assert record["self_check"]["ok"] is True
    assert record["self_check"]["reference_position_max_delta"] == 0.0
    assert record["self_check"]["persistence_position_max_delta"] == 0.0


def test_a_sharper_temperature_moves_the_rollout_and_not_the_probe_or_the_windows(swept):
    """The sweep's whole premise, on real records: the crossings differ between
    temperatures while the probe, the window identity and the truth-derived
    masks are shared -- they are computed before the rollout or from the truth."""
    record = load_record(swept.sweep / SWEEP)
    warm = record["entries"][script.tau_key(REFERENCE_TAU)]
    sharp = record["entries"][script.tau_key(0.0)]
    assert warm["probe"]["selection_r2"] == sharp["probe"]["selection_r2"]
    assert warm["summary"]["counts"]["not_moved"] == sharp["summary"]["counts"]["not_moved"]
    assert warm["summary"]["crossing"]["free"] != sharp["summary"]["crossing"]["free"]
    assert sharp["noise"]["median"] == 0.0, "two draws at tau = 0 are one trajectory"


def test_a_doctored_diagnostic_is_exit_30_and_writes_no_record(reference, capsys):
    _doctor(reference.out / DIAGNOSTIC, lambda d: d["curves"]["reference_position"].__setitem__(
        0, d["curves"]["reference_position"][0] + 1e-3))
    assert _run(reference, "--phase", "sweep") == script.EXIT_SELF_CHECK_FAILED
    assert "SELF-CHECK FAILED" in capsys.readouterr().out
    assert not (reference.sweep / SWEEP).exists()


def test_the_self_check_is_judged_before_any_other_temperature_runs(reference, monkeypatch, capsys):
    """A cell whose reference pass does not reproduce must cost one pass, not
    five: the temperatures below 1.0 are never evaluated."""
    _doctor(reference.out / DIAGNOSTIC, lambda d: d["curves"]["reference_position"].__setitem__(
        0, d["curves"]["reference_position"][0] + 1e-3))
    seen = []
    real = script.reference_trajectories

    def counting(*args, **kwargs):
        seen.append(kwargs.get("rollout_temperature"))
        return real(*args, **kwargs)

    monkeypatch.setattr(script, "reference_trajectories", counting)
    assert _run(reference, "--phase", "sweep") == script.EXIT_SELF_CHECK_FAILED
    capsys.readouterr()
    assert seen == [REFERENCE_TAU], seen


def test_a_split_that_is_not_the_records_is_exit_12_and_a_protocol_flag_is_exit_14(
    reference, monkeypatch, capsys
):
    monkeypatch.setattr(script, "reference_trajectories", _never_pass)
    assert _run(reference, "--phase", "sweep", "--context", "3") == script.EXIT_RECORD_MISMATCH
    assert "RECORD MISMATCH" in capsys.readouterr().out
    _doctor(reference.out / RECORD, lambda r: r["episodes"].__setitem__("val", ["ep_000099_len00040.npz"]))
    assert _run(reference, "--phase", "sweep") == script.EXIT_SPLIT_MISMATCH
    assert "SPLIT MISMATCH" in capsys.readouterr().out


def test_the_sweep_is_deterministic_on_cpu(swept, capsys):
    before = (swept.sweep / SWEEP).read_text()
    assert _run(swept, "--phase", "sweep") == script.EXIT_OK
    capsys.readouterr()
    assert (swept.sweep / SWEEP).read_text() == before
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_sharper_latent_script.py -q`
Expected: FAIL at collection — `FileNotFoundError` loading `scripts/sharper_latent.py`.

- [ ] **Step 3: Write the script**

Create `scripts/sharper_latent.py`:

```python
"""The sharper latent over the M3c cells: sweep the rollout's sampling temperature, and -- only if the rollout is noise-limited -- retrain the nine cells at the temperature the sweep names.

M3g localised the M3 failure to the prior: from the true state, one prior step
predicts the next posterior's mode less often than "the mode did not change"
does. Its own disclosure, and the measurements in spec 2026-09-23 M3h section
1, say that failure is a near-tie inside a much larger problem -- the
posterior carries 0.71-0.87 nats per group over 32 groups, so every imagined
step injects ~23-28 nats of fresh entropy against ~0.4-0.6 nats of dynamics
information, and from h = 5 onward two draws of the SAME model from the SAME
state separate by more than that model's entire imagined displacement. So:

  sweep     re-run the canonical pass on each shipped cell at every rollout
            temperature of TAU_GRID, sharpening the prior inside `imagine`
            and nothing else. tau = 1.0 runs first and must reproduce the
            cell's diagnostic bitwise, or nothing below it is read.
  train     read the sweep's records, decide Reading N, refuse (35) unless it
            says NOISE_LIMITED, check the tau = 1.0 identity (37), then
            retrain nine cells at tau*.
  evaluate  score each retrained cell through the study's own evaluation half
            and the trust pass, rung-style -- a retrained cell has no prior
            diagnostic to self-check against, exactly as M3f's rungs had none.
  read      pool both phases, decide Reading M, print the tables, write
            sweep.txt / sharper.txt and the figures.

LOADING IS `trust_horizon.py`'S: `Cell`, `load_cell`, `self_check` and
`prepare_cell` are imported by path, so a cell is refused here for the reasons
and in the words the other tools refuse it. `prepare_cell` reads the
temperature off the args it is handed (M3h), so the model it returns samples
the way the checkpoint it loaded was trained to.

THE CHECKS, BY PHASE, each with its own status:

  sweep:    EXIT_NO_CHECKPOINTS (11)     a requested cell lacks its checkpoint,
                                          record or diagnostic; judged for every
                                          cell before any pass runs.
            EXIT_SPLIT_MISMATCH (12)      the split by name is not the record's.
            EXIT_RECORD_MISMATCH (14)     --context/--horizon disagree with the
                                          protocol, or evaluate_rollout no longer
                                          reproduces the record's curve.
            EXIT_SELF_CHECK_FAILED (30)   the tau = 1.0 pass is not bitwise the
                                          cell's diagnostic -- judged BEFORE any
                                          other temperature of that cell runs.
  train:    EXIT_NO_CHECKPOINTS (11)     a sweep record is missing.
            EXIT_NOT_NOISE_LIMITED (35)   NEW. Reading N does not say the rollout
                                          is noise-limited: the retrain is gated
                                          in code, not by a person reading a table.
            EXIT_IDENTITY_CHECK_FAILED (37) NEW. the tau = 1.0 retrain's losses are
                                          not the M3c record's prefix exactly.
  evaluate: EXIT_NO_CHECKPOINTS (11)     a retrained checkpoint is missing.
            EXIT_TEMPERATURE_MISMATCH (36) NEW. a checkpoint's recorded temperature
                                          is not the one it is being evaluated at.
  read:     EXIT_NO_CHECKPOINTS (11)     a requested record is missing.
            EXIT_SELF_CHECK_FAILED (30)   a sweep record's self-check is not ok.

11 / 12 / 14 / 30 carry `trust_horizon.py`'s meanings on purpose; 35, 36 and
37 are in no other tool's range (run_study 1/3-6/23, report_study 7-10, spike
10, diagnose 11-17, pool 18-22, trust 30, split_gap 31, ladder 32-33, stages
34, argparse 2, a traceback 1).
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.aggregate import SEEDS
from mbfps.eval.diagnostics import reference_trajectories
from mbfps.eval.sharper import (
    DECISION_H,
    REFERENCE_TAU,
    TAU_GRID,
)
from mbfps.eval.split_gap import decision_horizon
from mbfps.eval.study import SPLIT_SEED, git_sha, load_record, write_record
from mbfps.utils.config import ARMS
from mbfps.utils.device import get_device


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_sharper_latent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_trust = _sibling("trust_horizon")
Cell = _trust.Cell
CellMissing = _trust.CellMissing
load_cell = _trust.load_cell
self_check = _trust.self_check
prepare_cell = _trust.prepare_cell
probe_is_measurable = _trust.probe_is_measurable

EXIT_OK = _trust.EXIT_OK
EXIT_NO_CHECKPOINTS = _trust.EXIT_NO_CHECKPOINTS
EXIT_SPLIT_MISMATCH = _trust.EXIT_SPLIT_MISMATCH
EXIT_RECORD_MISMATCH = _trust.EXIT_RECORD_MISMATCH
EXIT_SELF_CHECK_FAILED = _trust.EXIT_SELF_CHECK_FAILED
EXIT_NOT_NOISE_LIMITED = 35
EXIT_TEMPERATURE_MISMATCH = 36
EXIT_IDENTITY_CHECK_FAILED = 37
"""0 / 11 / 12 / 14 / 30 are `trust_horizon.py`'s, imported so they cannot
drift; 35, 36 and 37 are new and in no other tool's range."""

_split = _sibling("split_gap")
stratum_summary = _split.stratum_summary

PHASES: tuple[str, ...] = ("sweep",)


# ---------------------------------------------------------------------------
# Paths and keys.
# ---------------------------------------------------------------------------


def sweep_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    return Path(out_dir) / f"sweep_{arm}_seed{seed}.json"


def tau_key(tau: float) -> str:
    """`0.7 -> "tau70"`, `split_gap.q_key`'s idiom: a record key may not
    contain '.', since `write_record` addresses non-finite fields by dotted
    path. Hundredths are the grid's own resolution, so 0.3 and 0.30 are one
    entry and a float's repr never becomes part of the record's shape."""
    return f"tau{int(round(float(tau) * 100))}"


def tau_value(key: str) -> float:
    """`"tau70" -> 0.7`: `tau_key`'s inverse, so a reader of a record can get
    back the temperature an entry was scored at without parsing the entry."""
    return int(key.removeprefix("tau")) / 100.0


# ---------------------------------------------------------------------------
# sweep: one cell at every temperature.
# ---------------------------------------------------------------------------


def _cell_args(args, source: Path, tau: float = REFERENCE_TAU) -> types.SimpleNamespace:
    """`prepare_cell` reads `out`, `device`, `context`, `horizon` and (M3h)
    `sample_temperature` off its args. The sweep asks for 1.0 -- it sharpens
    the ROLLOUT of a model trained at 1.0, not the model."""
    return types.SimpleNamespace(
        out=Path(source), device=args.device, context=args.context, horizon=args.horizon,
        sample_temperature=float(tau),
    )


def tau_entry(traj, record: dict, horizon: int) -> dict:
    """One temperature of one cell: the gate's own metric off the pass's own
    band (not the study record's -- every temperature has its own rollout),
    the probe's measurability, and the trust pass reduced by
    `stratum_summary`, the same reduction M3e and M3f read. The probe's
    selection R^2 is the cell's, refit once by `prepare_cell` and shared by
    every temperature; its measurability is judged on THIS temperature's
    band, since the persistence-to-floor band is what makes the probe
    channel readable and the floor does not move but the model's curve does.
    `noise` is filled by the caller from the pass's noise reference."""
    summary = stratum_summary(traj, horizon)
    position = summary["band"]["position"]
    r2 = record["probe"]["embedding_selection_r2"]
    curves = summary["curves"]
    return {
        "gate": {
            "gap_final": position["gap_final"],
            "degenerate": position["steps_degenerate"],
        },
        "probe": {
            "selection_r2": float(r2) if r2 is not None else float("nan"),
            "measurable": bool(probe_is_measurable(
                {
                    "persistence": float(curves["persistence_position"][-1]),
                    "floor": float(curves["floor_position"][-1]),
                },
                widest_se=0.0,
            )),
        },
        "noise": {"curve": None, "median": None},
        "displacement": {"curve": None, "median": None},
        "summary": summary,
    }


def sweep_cell(args, cell: Cell, device, train, val, taus) -> tuple[int, dict | None]:
    """One cell: 12/14 from `prepare_cell`, the reference temperature and its
    self-check (30) BEFORE any other temperature, then the rest of the grid.

    The reference is scored first because the caller's order says so, not
    because the pre-registered grid happens to list it first: a cell whose
    pass at the shipped temperature does not reproduce its diagnostic must
    cost one pass, not five, and on the real nine cells five passes is an
    hour. A grid without the reference temperature is refused before any of
    it runs, for the same reason -- every contrast is against it.
    """
    taus = [float(t) for t in taus]
    if REFERENCE_TAU not in taus:
        raise ValueError(
            f"the grid {taus} does not contain the reference temperature {REFERENCE_TAU}; "
            "every contrast is against it and the self-check is taken on it"
        )
    keys = [tau_key(t) for t in taus]
    if len(set(keys)) != len(keys):
        raise ValueError(
            f"the grid {taus} has two temperatures that spell the same record key "
            f"({keys}); one entry would overwrite the other and `taus` would list a "
            "temperature no entry was scored at"
        )
    status, prepared = prepare_cell(_cell_args(args, args.reference), cell, device, train, val)
    if status != EXIT_OK:
        return status, None
    h, _ = decision_horizon(prepared.horizon)
    entries: dict[str, dict] = {}
    check = None
    for tau in [REFERENCE_TAU] + [t for t in taus if t != REFERENCE_TAU]:
        traj = reference_trajectories(
            prepared.model, val, prepared.embedding_probe,
            rollout_temperature=float(tau), **prepared.common,
        )
        if float(tau) == REFERENCE_TAU:
            check = self_check(traj, cell.diagnostic)
            if not check.ok:
                print(
                    f"\nSELF-CHECK FAILED for {cell.arm} seed {cell.seed} at tau="
                    f"{REFERENCE_TAU}: " + "; ".join(check.failures())
                    + ". The pass at the shipped temperature is not the one the gate scored, "
                    "so no sharper temperature of this cell is evaluated. No record written."
                )
                return EXIT_SELF_CHECK_FAILED, None
        entry = tau_entry(traj, cell.record, prepared.horizon)
        noise = np.asarray(traj.noise_embedding, dtype=float)
        entry["tau"] = float(tau)
        entry["noise"] = {
            "curve": noise.mean(axis=0),
            "median": float(np.median(noise)),
        }
        entries[tau_key(tau)] = entry
    record = {
        "arm": cell.arm,
        "seed": int(cell.seed),
        "source": str(args.reference),
        "step": int(cell.record["steps"]),
        "record_git_sha": cell.record["git_sha"],
        "context": int(prepared.context),
        "horizon": int(prepared.horizon),
        "decision_h": int(h),
        "split_seed": SPLIT_SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "git_sha": git_sha(),
        "taus": [float(t) for t in taus],
        "episodes": {"val": list(cell.record["episodes"]["val"])},
        "windows": {
            "total": int(entries[tau_key(REFERENCE_TAU)]["summary"]["windows"]["total"]),
            "episode": [
                int(e) for e in entries[tau_key(REFERENCE_TAU)]["summary"]["windows"]["episode"]
            ],
            "clusters": int(
                entries[tau_key(REFERENCE_TAU)]["summary"]["windows"]["clusters"]
            ),
        },
        "self_check": check.record(),
        "entries": entries,
    }
    path = sweep_record_path(args.sweep_out, cell.arm, cell.seed)
    write_record(path, record)
    gaps = " / ".join(
        f"tau {entries[tau_key(t)]['tau']:.1f}: gap {entries[tau_key(t)]['gate']['gap_final']:+.3f}"
        for t in taus
    )
    print(f"{cell.arm} seed {cell.seed}: {gaps}; wrote {path}")
    return EXIT_OK, record


def sweep_phase(args, cells, device, train, val, taus) -> int:
    try:
        loaded = [load_cell(args.reference, arm, seed) for arm, seed in cells]
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    args.sweep_out.mkdir(parents=True, exist_ok=True)
    for cell in loaded:
        status, _ = sweep_cell(args, cell, device, train, val, taus)
        if status != EXIT_OK:
            return status
    return EXIT_OK


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/m3h_sharper"))
    parser.add_argument("--sweep-out", type=Path, default=Path("runs/m3h_sweep"))
    parser.add_argument("--reference", type=Path, default=Path("runs/m3_study_v2"),
                        help="the M3c study directory: the nine 20,000-step cells")
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--device", default="mps")
    # None -> the cell's diagnostic says what it was written at; a value that
    # disagrees is refused (`protocol_mismatch`, 14).
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--phase", choices=PHASES, default=PHASES[-1])
    parser.add_argument("--figure", type=Path, default=None,
                        help="the curves figure; default <sweep-out>/sweep_curves.png")
    return parser


def _check_usage(parser: argparse.ArgumentParser, args) -> None:
    for name, other in (("--reference", args.reference), ("--sweep-out", args.sweep_out)):
        if args.out.resolve() == Path(other).resolve():
            parser.error(
                f"--out and {name} are the same directory; the retrain would write its "
                "checkpoints over the ones it is being compared against"
            )


def main(argv: list[str] | None = None, *, taus=TAU_GRID) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _check_usage(parser, args)
    device = get_device(prefer=args.device)
    cells = [(arm, int(seed)) for arm in args.arms for seed in args.seeds]
    if args.phase in ("sweep", "all"):
        buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
        train, val = episode_split(buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
        status = sweep_phase(args, cells, device, train, val, taus)
        if status != EXIT_OK:
            return status
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_sharper_latent_script.py -q`
Expected: 15 passed. (The fixture trains a tiny cell and runs the diagnostic; expect three or four minutes.)

`stratum_summary` returns `windows / curves / band / moved / first_moved / crossing / margin / survival / trust_horizon / counts`, and `band["position"]` carries `gap_final` and `steps_degenerate` (`mbfps.eval.summary.metric_summary`); it carries no probe block, which is why `tau_entry` takes the cell's record for the R². If any of those names differ from what the installed code returns, use the names it actually returns and say so in your report rather than reshaping the record.

- [ ] **Step 5: Commit**

```bash
git add scripts/sharper_latent.py tests/eval/test_sharper_latent_script.py
git commit -m "feat: sharper_latent.py sweep -- each shipped cell re-scored at every rollout temperature, the reference temperature first and self-checked bitwise before any sharper one runs

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: `scripts/sharper_latent.py` — the gate, the identity check, the retrain, the evaluation

**Files:**
- Modify: `scripts/sharper_latent.py` (imports, `PHASES`, append the train/evaluate half, extend `main`)
- Test: `tests/eval/test_sharper_latent_script.py` (append)

**Interfaces:**
- Consumes: Task 4/5's `sharper.*`; Task 1's `get_config(..., sample_temperature=τ)`; Task 2's payload and `prepare_cell` threading; `checkpoint_ladder.rung_cell` by path; `world_model.train_world_model`; `study.{StudyJob, evaluate_job}`; `ladder.anchor_delta`.
- Produces: `PHASES = ("sweep", "train", "evaluate", "read", "all")`; `sweep_inputs(records, *, arms, seeds, h) -> SweepInputs`; `load_sweep(args, cells) -> (status, records)`; `identity_check(args, buffer, arm, seed, reference) -> (status, float, int | None)`; `train_cell(args, buffer, arm, seed, reference, tau) -> (status, dict | None)`; `train_phase`, `evaluate_cell`, `evaluate_phase`; `retrain_record_path(out, arm, seed)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_sharper_latent_script.py`:

```python


# ---------------------------------------------------------------------------
# The gate, the identity check and the retrain (Task 7). The sweep's own
# record decides whether fifteen hours are spent; these tests drive that
# decision by fabricating the reading rather than by finding a cell that
# happens to clear, so the gate is tested and not the fixture's luck.
# ---------------------------------------------------------------------------

from mbfps.eval.sharper import IDENTITY_STEPS, SweepStatus  # noqa: E402

RETRAIN = "retrain_random_vit_seed1.json"
CHECKPOINT = "world_model_random_vit_seed1.pt"
RETRAIN_TAU = 0.5
RETRAIN_KEYS = {
    "arm", "seed", "tau", "steps", "seq_len", "history", "seconds", "identity",
    "device", "torch_version", "git_sha", "reference_git_sha", "nonfinite",
}


def _reading(status: SweepStatus, tau=None):
    """A `reading_noise` stand-in with a chosen status and tau*."""
    from mbfps.eval.sharper import SweepReading

    def fake(inputs):
        return SweepReading(
            cells={}, arms_clearing={}, arms_against={}, tau_star=tau,
            status=status, rule="fabricated", h=inputs.h, z_fam=inputs.z_fam,
        )

    return fake


def _never_train(*args, **kwargs):
    raise AssertionError("train_world_model ran; this refusal must come before any training")


@pytest.fixture
def trained(swept, monkeypatch, capsys):
    monkeypatch.setattr(script, "reading_noise", _reading(SweepStatus.NOISE_LIMITED, RETRAIN_TAU))
    assert _run(swept, "--phase", "train", "--steps", "4") == script.EXIT_OK
    capsys.readouterr()
    return swept


def test_train_without_a_sweep_record_is_exit_11(reference, capsys):
    assert _run(reference, "--phase", "train") == script.EXIT_NO_CHECKPOINTS
    assert "run --phase sweep first" in capsys.readouterr().out


def test_the_real_reading_is_what_the_gate_consults(swept, monkeypatch, capsys):
    """Every other gate test fabricates the reading to drive the decision.
    This one does not: the fixture is one arm and one seed, so the real
    `reading_noise` cannot reach NOISE_LIMITED (it needs two arms and two
    seeds), and the refusal must come from it rather than from a stand-in.
    Without this, a swapped treatment/control in `_tau_inputs` would invert
    the gate and every fabricated test would still pass."""
    monkeypatch.setattr(script, "train_world_model", _never_train)
    assert _run(swept, "--phase", "train") == script.EXIT_NOT_NOISE_LIMITED
    out = capsys.readouterr().out
    assert "NOT NOISE LIMITED" in out
    assert "fabricated" not in out
    assert "arms required" in out or "of 1 arms" in out or "of 3 arms" in out


def test_the_glue_pairs_a_temperature_against_the_reference_not_the_other_way(swept):
    """The contrast's orientation is the whole reading: `S_free(h)` at tau
    MINUS at the reference. Swapped, `SHARPER_WORSE` would read as
    `NOISE_LIMITED` and spend fifteen hours on it.

    Taken through `_tau_inputs` itself, not through a reconstruction of what
    it does: a first version of this test called `_series`/`_paired` directly
    and chose its own treatment and control, so a swap INSIDE `_tau_inputs`
    left the test and the bug each self-consistent and the suite green.
    """
    import types as _types

    status, records = script.load_sweep(
        _types.SimpleNamespace(sweep_out=swept.sweep), [(JOB.arm, JOB.seed)]
    )
    assert status == script.EXIT_OK
    record = records[(JOB.arm, JOB.seed)]
    h = int(record["decision_h"])
    inputs = script._tau_inputs(records, JOB.arm, [JOB.seed], 0.0, h)

    # The same difference, computed straight off the record without touching
    # the glue: the survival indicator at tau = 0 minus the one at tau = 1,
    # over the windows both temperatures changed.
    def survival(tau):
        return survival_indicator(record["entries"][script.tau_key(tau)]["summary"], "free", h)

    values, changed = survival(0.0)
    reference_values, reference_changed = survival(REFERENCE_TAU)
    keep = changed & reference_changed
    direct = float(np.mean(values[keep] - reference_values[keep]))

    assert inputs.free.estimate == pytest.approx(direct, abs=1e-12), (
        inputs.free.estimate, direct
    )
    assert inputs.per_seed[JOB.seed].estimate == pytest.approx(direct, abs=1e-12)


def test_a_sweep_that_is_not_noise_limited_refuses_to_retrain(swept, monkeypatch, capsys):
    """THE gate: fifteen hours are not spent on a refuted premise, and the
    refusal is the script's, not a person's reading of a table."""
    monkeypatch.setattr(script, "reading_noise", _reading(SweepStatus.NOT_NOISE_LIMITED))
    monkeypatch.setattr(script, "train_world_model", _never_train)
    assert _run(swept, "--phase", "train") == script.EXIT_NOT_NOISE_LIMITED
    out = capsys.readouterr().out
    assert "NOT NOISE LIMITED" in out and "fabricated" in out
    assert not (swept.sharper / CHECKPOINT).exists()


def test_a_sharper_worse_sweep_also_refuses(swept, monkeypatch, capsys):
    monkeypatch.setattr(script, "reading_noise", _reading(SweepStatus.SHARPER_WORSE))
    monkeypatch.setattr(script, "train_world_model", _never_train)
    assert _run(swept, "--phase", "train") == script.EXIT_NOT_NOISE_LIMITED
    assert "SHARPER WORSE" in capsys.readouterr().out


def test_a_broken_identity_check_refuses_before_any_cell_is_retrained(swept, monkeypatch, capsys):
    """The tau = 1.0 retrain must reproduce the M3c loss prefix exactly -- it
    is what makes the shipped cells the control arm without spending fifteen
    hours on one. Doctoring the reference history breaks it."""
    monkeypatch.setattr(script, "reading_noise", _reading(SweepStatus.NOISE_LIMITED, RETRAIN_TAU))
    _doctor(swept.out / RECORD, lambda r: r["history"]["loss"].__setitem__(0, r["history"]["loss"][0] + 1e-3))
    assert _run(swept, "--phase", "train", "--steps", "4") == script.EXIT_IDENTITY_CHECK_FAILED
    out = capsys.readouterr().out
    assert "IDENTITY CHECK FAILED" in out and "step 1" in out
    assert not (swept.sharper / CHECKPOINT).exists()


def test_the_identity_check_is_the_shipped_temperature_and_its_steps_are_pinned(swept, monkeypatch, capsys):
    """It is a tau = 1.0 retrain, not a tau* one: it asks whether the edit is a
    no-op where it must be."""
    monkeypatch.setattr(script, "reading_noise", _reading(SweepStatus.NOISE_LIMITED, RETRAIN_TAU))
    seen = []
    real = script.train_world_model

    def recording(cfg, *args, **kwargs):
        seen.append(cfg.train.sample_temperature)
        return real(cfg, *args, **kwargs)

    monkeypatch.setattr(script, "train_world_model", recording)
    assert _run(swept, "--phase", "train", "--steps", "4") == script.EXIT_OK
    capsys.readouterr()
    assert seen[0] == REFERENCE_TAU, "the identity check runs first, at the shipped temperature"
    assert seen[1:] == [RETRAIN_TAU], "then every requested cell at tau*"
    assert IDENTITY_STEPS == 500


def test_the_retrain_writes_a_checkpoint_carrying_its_temperature_and_a_record(trained):
    import torch

    payload = torch.load(trained.sharper / CHECKPOINT, weights_only=True)
    assert payload["sample_temperature"] == RETRAIN_TAU
    assert (payload["arm"], payload["seed"]) == (JOB.arm, JOB.seed)
    record = load_record(trained.sharper / RETRAIN)
    assert set(record) == RETRAIN_KEYS
    assert record["tau"] == RETRAIN_TAU and record["steps"] == 4
    assert record["identity"]["max_delta"] == 0.0 and record["identity"]["first_step"] is None
    assert record["reference_git_sha"] == load_record(trained.out / RECORD)["git_sha"]
    assert len(record["history"]["loss"]) == 4


def test_evaluate_scores_the_retrained_cell_at_its_own_temperature(trained, capsys):
    assert _run(trained, "--phase", "evaluate", "--steps", "4") == script.EXIT_OK
    capsys.readouterr()
    study = load_record(trained.sharper / RECORD)
    assert study["steps"] == 4
    trust_record = load_record(trained.sharper / "trust_random_vit_seed1.json")
    assert trust_record["windows"]["total"] == WINDOWS
    assert np.asarray(trust_record["crossing"]["free"], dtype=float).shape == (WINDOWS,)


def test_evaluating_a_checkpoint_at_the_wrong_temperature_is_exit_36(trained, monkeypatch, capsys):
    """The payload says what sampler produced these weights; evaluating them
    with another one would report a model that never existed."""
    import torch

    path = trained.sharper / CHECKPOINT
    payload = torch.load(path, weights_only=True)
    payload["sample_temperature"] = 0.25
    torch.save(payload, path)
    assert _run(trained, "--phase", "evaluate", "--steps", "4") == script.EXIT_TEMPERATURE_MISMATCH
    assert "TEMPERATURE MISMATCH" in capsys.readouterr().out


def test_evaluate_without_a_retrain_record_is_exit_11(swept, capsys):
    assert _run(swept, "--phase", "evaluate") == script.EXIT_NO_CHECKPOINTS
    assert "run --phase train first" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_sharper_latent_script.py -q -k "train or identity or retrain or evaluate"`
Expected: FAIL — `--phase train` is `SystemExit: 2` until `PHASES` grows, and `script.reading_noise` does not exist.

- [ ] **Step 3: Extend the imports and `PHASES`**

Replace the `from mbfps.eval.sharper import (...)` block:

```python
from mbfps.eval.sharper import (
    ARMS_REQUIRED,
    DECISION_H,
    IDENTITY_STEPS,
    REFERENCE_TAU,
    RETRAIN_FAMILY,
    SEEDS_REQUIRED,
    STEPS,
    SWEEP_FAMILY,
    TAU_CANDIDATES,
    TAU_GRID,
    RetrainArm,
    RetrainInputs,
    SweepArm,
    SweepInputs,
    SweepStatus,
    TauInputs,
    format_reading_noise,
    format_reading_sharper,
    reading_noise,
    reading_sharper,
)
```

and add beside the existing imports:

```python
import mbfps.eval.pooling as pooling
from mbfps.eval.split_gap import StratumContrast, cell_series, decision_horizon, fmt_z, survival_indicator
from mbfps.eval.study import SPLIT_SEED, StudyJob, evaluate_job, git_sha, job_record_path, load_record, write_record
from mbfps.eval.trust_readings import ARMS_ORDER
from mbfps.eval.ladder import anchor_delta
from mbfps.training.world_model import train_world_model
from mbfps.utils.config import ARMS, get_config
```

(dropping the now-duplicated `from mbfps.eval.study import ...` and `from mbfps.utils.config import ARMS` lines), bind the ladder's rung cell beside the other siblings:

```python
_ladder = _sibling("checkpoint_ladder")
rung_cell = _ladder.rung_cell
```

and change:

```python
PHASES: tuple[str, ...] = ("sweep", "train", "evaluate", "read", "all")
```

- [ ] **Step 4: Append the pooling glue and the train/evaluate half**

Insert before the `# main` banner:

```python
# ---------------------------------------------------------------------------
# The sweep's records -> Reading N's inputs.
# ---------------------------------------------------------------------------
# Every estimator is `pooling.py`'s. What is decided here is only WHICH series
# goes in: the survival indicator over the windows that moved within the
# horizon, at one temperature against the same cell at REFERENCE_TAU. The two
# temperature groups of one cell are labelled as two "arms" so
# `paired_contrast` -- which refuses an arm against itself -- reads them as
# two groups on the same windows, exactly as M3f paired a rung against 20000.


def _entry(record: dict, tau: float) -> dict:
    return record["entries"][tau_key(tau)]


def _series(record: dict, tau: float, h: int, label: str) -> pooling.CellSeries:
    values, changed = survival_indicator(_entry(record, tau)["summary"], "free", h)
    return cell_series(
        _entry(record, tau)["summary"], values, changed,
        arm=label, seed=record["seed"], rung="val", channel="S/free",
        val=record["episodes"]["val"], horizon=record["horizon"], context=record["context"],
        device=record["device"], torch_version=record["torch_version"],
    )


_NO_CONTRAST = StratumContrast(estimate=float("nan"), se=float("nan"), z=float("nan"), clusters=0)


def _paired(treatment, control) -> StratumContrast:
    """`pooling.paired_contrast`, or the NaN contrast when there is nothing to
    pair -- no cell, or no window every cell of both groups changed."""
    treatment, control = list(treatment), list(control)
    if not treatment or not control:
        return _NO_CONTRAST
    if not np.logical_and.reduce([c.changed for c in treatment + control]).any():
        return _NO_CONTRAST
    c = pooling.paired_contrast(treatment, control)
    return StratumContrast(estimate=c.mean, se=c.se, z=c.z, clusters=c.clusters)


def _tau_inputs(records: dict, arm: str, seeds, tau: float, h: int) -> TauInputs:
    def contrast(seed_list):
        kept = [records[(arm, int(s))] for s in seed_list]
        return _paired(
            [_series(r, tau, h, f"{arm}@{tau_key(tau)}") for r in kept],
            [_series(r, REFERENCE_TAU, h, f"{arm}@{tau_key(REFERENCE_TAU)}") for r in kept],
        )

    return TauInputs(
        free=contrast(seeds),
        per_seed={int(s): contrast([s]) for s in seeds},
    )


def _ordered(arms) -> list[str]:
    return sorted(arms, key=lambda a: ARMS_ORDER.index(a) if a in ARMS_ORDER else len(ARMS_ORDER))


def _clusters(records: dict) -> int:
    return int(next(iter(records.values()))["windows"]["clusters"])


def sweep_inputs(records: dict, *, arms, seeds, h: int, taus) -> SweepInputs:
    """Every arm's inputs at `h`, one entry per non-reference temperature, and
    `z_fam` over the val stratum's cluster count (spec 3.1)."""
    candidates = [float(t) for t in taus if float(t) != REFERENCE_TAU]
    return SweepInputs(
        arms={
            arm: SweepArm(taus={t: _tau_inputs(records, arm, seeds, t, h) for t in candidates})
            for arm in _ordered(arms)
        },
        z_fam=pooling.cluster_threshold(SWEEP_FAMILY, _clusters(records)),
        h=h,
    )


def load_sweep(args, cells) -> tuple[int, dict]:
    """Every requested sweep record, keyed by `(arm, seed)`; 11 names the
    first missing, 30 the first whose reference self-check is not exact."""
    records: dict[tuple[str, int], dict] = {}
    for arm, seed in cells:
        path = sweep_record_path(args.sweep_out, arm, seed)
        if not path.exists():
            print(f"NO CELL: {arm} seed {seed}: no sweep record at {path}; run --phase sweep first")
            return EXIT_NO_CHECKPOINTS, {}
        records[(arm, int(seed))] = load_record(path)
    for (arm, seed), record in records.items():
        if not _trust.trustworthy(record["self_check"]):
            print(
                f"\nSELF-CHECK FAILED for {arm} seed {seed}: the sweep record's self-check is "
                f"{record['self_check']!r}; it was not taken against a pass that reproduced."
            )
            return EXIT_SELF_CHECK_FAILED, {}
    return EXIT_OK, records


# ---------------------------------------------------------------------------
# train: the gate, the identity check, the retrains.
# ---------------------------------------------------------------------------


def retrain_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    return Path(out_dir) / f"retrain_{arm}_seed{seed}.json"


def identity_check(args, buffer, cell: Cell) -> tuple[int, dict | None]:
    """A short retrain at REFERENCE_TAU whose per-step losses must equal the
    M3c record's prefix exactly. M3f measured this box's MPS training to be
    bitwise deterministic, which is what makes the shipped cells the control
    arm without a second fifteen-hour run; this is that premise verified
    rather than assumed, and it is also the only check that the temperature
    edit is a no-op where it must be.

    Run for EVERY cell that retrains, not once for the first: the result is
    written into that cell's record, and a 0.0 measured on another cell would
    be a number that means nothing sitting where one that means something
    goes. `prepare_cell`'s own reproduction message notes the delta is
    arm-dependent across devices, so one cell's result does not stand for
    another's."""
    steps = min(IDENTITY_STEPS, int(args.steps))
    if steps < 1:
        raise ValueError(f"--steps {args.steps} leaves no identity check to run")
    cfg = get_config(
        cell.arm, steps=steps, seq_len=int(cell.record["seq_len"]), seed=cell.seed,
        device=args.device, sample_temperature=REFERENCE_TAU,
    )
    history = train_world_model(cfg, buffer, out_dir=None)
    delta, first = anchor_delta(history["loss"], cell.record["history"]["loss"], steps)
    if delta != 0.0:
        print(
            f"\nIDENTITY CHECK FAILED for {cell.arm} seed {cell.seed}: a retrain at tau="
            f"{REFERENCE_TAU} does not reproduce the reference run's per-step loss over "
            f"{steps} steps (max abs {delta:.3e}, first at step {first}). The shipped cells "
            "are the control arm only if a retrain at the shipped temperature IS the shipped "
            "run. No cell is retrained."
        )
        return EXIT_IDENTITY_CHECK_FAILED, None
    print(
        f"identity check: {cell.arm} seed {cell.seed} at tau={REFERENCE_TAU} reproduces "
        f"{steps} steps exactly (max abs 0.0e+00)"
    )
    # Self-describing: the result names the cell it was measured on, so no
    # record can carry a 0.0 that was measured somewhere else.
    return EXIT_OK, {
        "arm": cell.arm, "seed": int(cell.seed), "tau": float(REFERENCE_TAU),
        "steps": int(steps), "max_delta": float(delta), "first_step": first,
    }


def train_cell(args, buffer, cell: Cell, tau: float, identity: dict) -> tuple[int, dict]:
    """One cell retrained at `tau` from the study's own seed and sequence
    length, into `--out` in the study's layout."""
    cfg = get_config(
        cell.arm, steps=int(args.steps), seq_len=int(cell.record["seq_len"]), seed=cell.seed,
        device=args.device, sample_temperature=float(tau),
    )
    history = train_world_model(cfg, buffer, out_dir=args.out)
    record = {
        "arm": cell.arm,
        "seed": int(cell.seed),
        "tau": float(tau),
        "steps": int(args.steps),
        "seq_len": int(cell.record["seq_len"]),
        "history": history_record(history),
        "seconds": float(history["seconds"]),
        "identity": dict(identity),
        "device": str(args.device),
        "torch_version": torch.__version__,
        "git_sha": git_sha(),
        "reference_git_sha": cell.record["git_sha"],
    }
    path = retrain_record_path(args.out, cell.arm, cell.seed)
    write_record(path, record)
    print(
        f"{cell.arm} seed {cell.seed}: retrained {args.steps} steps at tau={tau} in "
        f"{history['seconds']:.0f}s; wrote {path}"
    )
    return EXIT_OK, record


def train_phase(args, cells, device, buffer, taus) -> int:
    """The sweep's records (11, 30), Reading N, the gate (35), the identity
    check (37), then every requested cell at tau*."""
    status, records = load_sweep(args, cells)
    if status != EXIT_OK:
        return status
    h = int(next(iter(records.values()))["decision_h"])
    inputs = sweep_inputs(
        records, arms=list(args.arms), seeds=[int(s) for s in args.seeds], h=h, taus=taus,
    )
    reading = reading_noise(inputs)
    print(format_reading_noise(reading, inputs), end="")
    if reading.status is not SweepStatus.NOISE_LIMITED or reading.tau_star is None:
        print(
            f"\n{reading.status.name.replace('_', ' ')}: {reading.rule}. The retrain is gated on "
            "the sweep, so nothing is trained: a sharper sampler does not help these models at "
            "this grid, and fifteen hours would be spent on a refuted premise."
        )
        return EXIT_NOT_NOISE_LIMITED
    tau = float(reading.tau_star)
    try:
        loaded = [load_cell(args.reference, arm, seed) for arm, seed in cells]
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    args.out.mkdir(parents=True, exist_ok=True)
    for cell in loaded:
        status, identity = identity_check(args, buffer, cell)
        if status != EXIT_OK:
            return status
        status, _ = train_cell(args, buffer, cell, tau, identity)
        if status != EXIT_OK:
            return status
    return EXIT_OK


# ---------------------------------------------------------------------------
# evaluate: the study's own evaluation half, then the trust pass, rung-style.
# ---------------------------------------------------------------------------


def evaluate_cell(args, buffer, arm: str, seed: int, device, train, val) -> tuple[int, dict | None]:
    """One retrained cell: its retrain record (11), `evaluate_job` at the
    cell's own temperature, then `prepare_cell` on a rung-style cell (36, 12,
    14) and the trust pass. A retrained cell has no prior diagnostic to
    self-check against -- exactly as M3f's rungs had none -- so the check
    that stands here is the temperature the payload carries."""
    path = retrain_record_path(args.out, arm, seed)
    if not path.exists():
        print(f"NO CELL: {arm} seed {seed}: no retrain record at {path}; run --phase train first")
        return EXIT_NO_CHECKPOINTS, None
    record = load_record(path)
    tau = float(record["tau"])
    # The retrain record carries no protocol of its own (`RETRAIN_KEYS`): a
    # retrained cell is evaluated at the M3c reference's own context/horizon,
    # read off its study record, unless a flag overrides it -- the same
    # None-means-the-cell's-own-value rule `_cell_args`/`protocol_mismatch`
    # apply everywhere else in this file.
    reference_record = load_record(job_record_path(args.reference, StudyJob(arm=arm, seed=seed)))
    context = int(args.context) if args.context is not None else int(reference_record["context"])
    horizon = int(args.horizon) if args.horizon is not None else int(reference_record["horizon"])
    # The payload's temperature is checked BEFORE `evaluate_job` runs: it
    # writes `result_<arm>_seed<n>.json` unconditionally, so refusing only
    # afterwards would persist -- and overwrite any earlier valid record
    # with -- a record describing a model that never existed.
    trained_at = float(
        torch.load(
            _trust.checkpoint_path(args.out, arm, seed), map_location="cpu", weights_only=True
        ).get("sample_temperature", REFERENCE_TAU)
    )
    if trained_at != tau:
        print(
            f"\nTEMPERATURE MISMATCH for {arm} seed {seed}: the checkpoint was trained at "
            f"sample_temperature={trained_at}, but its retrain record says {tau}. Nothing is "
            "evaluated: a record written from the wrong sampler describes a model that never "
            "existed."
        )
        return EXIT_TEMPERATURE_MISMATCH, None
    try:
        evaluate_job(
            StudyJob(arm=arm, seed=seed), buffer, args.out,
            history=history_from_record(record), steps=int(record["steps"]),
            seq_len=int(record["seq_len"]), context=context,
            horizon=horizon, device=args.device, sample_temperature=tau,
        )
        cell = rung_cell(args.out, arm, seed)
        status, prepared = prepare_cell(_cell_args(args, args.out, tau), cell, device, train, val)
    # `_trust.TemperatureMismatch`, NOT this script's own `_diagnose` copy:
    # `_sibling` executes the file afresh per importer, so the class
    # `prepare_cell` raises is trust_horizon's, and catching any other copy
    # lets it escape as a traceback instead of exit 36.
    except _trust.TemperatureMismatch as error:
        print(f"\nTEMPERATURE MISMATCH for {arm} seed {seed}: {error}")
        return EXIT_TEMPERATURE_MISMATCH, None
    if status != EXIT_OK:
        return status, None
    traj = reference_trajectories(prepared.model, val, prepared.embedding_probe, **prepared.common)
    # `cell.diagnostic` (from `rung_cell`) is only `{context, horizon}` --
    # exactly what `checkpoint_ladder.py` reads there itself, because a rung
    # has no diagnostic either (`rung_entry` there passes its own trust pass
    # `check=None`). `_trust.trust_record` has no such escape hatch: it always
    # calls `self_check`, which needs a diagnostic's `curves.reference_position`,
    # `curves.persistence_position` and `windows` to compare the pass against.
    # A retrained cell has no earlier pass to compare to, so the comparison is
    # built from THIS SAME pass -- exact against itself by construction -- and
    # `curves.floor_position` / `probe.embedding_selection_r2`, which self_check
    # never compares, come from the study record `evaluate_job` just wrote.
    # The meaningful check for a retrained cell is the temperature the payload
    # carries (already refused above, 36), not a bitwise reproduction with
    # nothing earlier to reproduce.
    diagnostic = {
        "curves": {
            "reference_position": _trust._distance(
                traj.positions, traj.true_positions
            ).mean(axis=0).tolist(),
            "persistence_position": _trust._distance(
                traj.positions_at_context[:, None, :], traj.true_positions
            ).mean(axis=0).tolist(),
            "floor_position": list(cell.record["curves"]["floor_position"]),
        },
        "probe": {"embedding_selection_r2": cell.record["probe"]["embedding_selection_r2"]},
        "windows": {"total": int(traj.windows_total), "episode": [int(e) for e in traj.window_episode]},
        "episodes": {"val": list(cell.record["episodes"]["val"])},
    }
    trust_record = _trust.trust_record(
        arm, seed, traj, diagnostic,
        context=prepared.context, horizon=prepared.horizon, device=device,
    )
    # The self-check that `trust_record` just computed compared this pass with
    # itself, so it reads 0.0 by construction. A sweep record's 0.0 means the
    # pass reproduced the shipped diagnostic bitwise; this one would mean
    # nothing, and the two are indistinguishable once written. So the record
    # says there was nothing to reproduce, and the check that DOES bind a
    # retrained cell -- the temperature its payload carries -- has already
    # refused above (36).
    trust_record["self_check"] = None
    written = _trust.write_trust_record(args.out, trust_record)
    print(f"{arm} seed {seed}: evaluated at tau={tau}; wrote {written}")
    return EXIT_OK, trust_record


def evaluate_phase(args, cells, device, buffer, train, val) -> int:
    for arm, seed in cells:
        status, _ = evaluate_cell(args, buffer, arm, seed, device, train, val)
        if status != EXIT_OK:
            return status
    return EXIT_OK
```

with `history_record` imported from `mbfps.eval.study`. **`history_from_record` is this script's own function, not `checkpoint_ladder.history_from_train_record`**: that one reads `kl_dyn_max`, `kl_rate_above_free_bits` and `checkpoint_seconds` off its record's top level, where the ladder's `train_record` keeps them; this script's retrain record carries none of the three (`history_record`'s own policy keeps only `loss` and `parts`), so calling it raises `KeyError('kl_dyn_max')` before a single rollout runs. Recompute the two KL summaries from the stored per-step `parts`, exactly as `train_world_model` computes them from the live history.

`evaluate_job` does not take a `sample_temperature` today. Give it one, defaulting to `SAMPLE_TEMPERATURE`, passed to the `get_config` it already calls, and pin in `tests/eval/test_run_study.py` that the default leaves its record unchanged:

```python
def test_evaluate_job_takes_the_temperature_its_checkpoint_was_trained_at(wide_buffer, tmp_path):
    """A retrained cell's weights are read by a model that must sample the way
    they were trained to; the default is the shipped 1.0, so every existing
    caller is unchanged."""
    import inspect

    from mbfps.eval.study import evaluate_job
    from mbfps.models.rssm import SAMPLE_TEMPERATURE

    parameter = inspect.signature(evaluate_job).parameters["sample_temperature"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == SAMPLE_TEMPERATURE == 1.0
```

- [ ] **Step 5: Extend `main`**

```python
def main(argv: list[str] | None = None, *, taus=TAU_GRID) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _check_usage(parser, args)
    device = get_device(prefer=args.device)
    cells = [(arm, int(seed)) for arm in args.arms for seed in args.seeds]
    if args.phase in ("sweep", "train", "evaluate", "all"):
        buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
        train, val = episode_split(buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
        if args.phase in ("sweep", "all"):
            status = sweep_phase(args, cells, device, train, val, taus)
            if status != EXIT_OK:
                return status
        if args.phase in ("train", "all"):
            status = train_phase(args, cells, device, buffer, taus)
            if status != EXIT_OK:
                return status
        if args.phase in ("evaluate", "all"):
            status = evaluate_phase(args, cells, device, buffer, train, val)
            if status != EXIT_OK:
                return status
    if args.phase in ("read", "all"):
        return read_phase(args, cells)
    return EXIT_OK
```

and add `--steps` to the parser (`parser.add_argument("--steps", type=int, default=STEPS, help="the retrain's length")`). Task 8 supplies `read_phase`; until then, add a placeholder that raises `NotImplementedError("read is Task 8")` **and remove it in Task 8** — no test in this task exercises `read`.

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/eval/test_sharper_latent_script.py tests/eval/test_run_study.py -q`
Expected: the Task 6 tests plus the nine new ones pass, and `test_run_study.py` is green with its new pin.

- [ ] **Step 7: Commit**

```bash
git add scripts/sharper_latent.py src/mbfps/eval/study.py tests/eval/test_sharper_latent_script.py tests/eval/test_run_study.py
git commit -m "feat: sharper_latent.py train and evaluate -- Reading N gates the retrain in code, the tau=1 identity check proves the shipped cells are the control arm, and every retrained cell is scored at the temperature its payload carries

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: `scripts/sharper_latent.py` — the reading

**Files:**
- Modify: `scripts/sharper_latent.py` (append the read half, drop Task 7's placeholder)
- Test: `tests/eval/test_sharper_latent_script.py` (append)

**Interfaces:**
- Consumes: Tasks 4–7; `trust_horizon`'s trust records for the retrained cells and the M3c cells' own `trust_<arm>_seed<n>.json` in `--reference`.
- Produces: `retrain_inputs(sweep_records, new_trust, old_trust, *, arms, seeds, h, tau) -> RetrainInputs`; `_sweep_table`, `_gate_table`, `_noise_table`, `_survival_table`, `_objective_free`; `write_curves`; `sweep_text`, `sharper_text`; `write_text(path, text)`; `read_phase(args, cells) -> int`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_sharper_latent_script.py`:

```python


# ---------------------------------------------------------------------------
# read (Task 8): the sweep's table and verdict always; the retrain's, when a
# retrain exists. stages.txt's discipline -- what is printed is what is
# written, byte for byte.
# ---------------------------------------------------------------------------

SWEEP_SECTIONS = (
    "--- self-check per cell",
    "--- the gate at every temperature",
    "--- the noise reference against the imagined displacement",
    "--- survival by temperature",
    "--- Reading N: is the rollout noise-limited at h=3",
)


@pytest.fixture
def swept_read(swept, capsys):
    assert _run(swept, "--phase", "read") == script.EXIT_OK
    text = capsys.readouterr().out
    return types.SimpleNamespace(ref=swept, text=text, path=swept.sweep / "sweep.txt")


def test_read_without_a_sweep_record_is_exit_11(reference, capsys):
    assert _run(reference, "--phase", "read") == script.EXIT_NO_CHECKPOINTS
    assert "run --phase sweep first" in capsys.readouterr().out


def test_read_refuses_a_sweep_record_whose_self_check_is_not_exact(swept, capsys):
    _doctor(swept.sweep / SWEEP, lambda r: r["self_check"].update({"reference_position_max_delta": 1e-6}))
    assert _run(swept, "--phase", "read") == script.EXIT_SELF_CHECK_FAILED
    assert "SELF-CHECK FAILED" in capsys.readouterr().out
    assert not (swept.sweep / "sweep.txt").exists()


def test_read_prints_the_sweep_sections_and_writes_them_byte_identical(swept_read):
    for section in SWEEP_SECTIONS:
        assert section in swept_read.text, section
    assert swept_read.path.read_text() == swept_read.text
    assert "z_fam = cluster_threshold(9, 6)" in swept_read.text
    assert "verdict:" in swept_read.text
    assert f"{REFERENCE_TAU:.1f}" in swept_read.text
    assert "NOTE: decision horizon clamped to the run's horizon h=3" in swept_read.text


def test_the_sweep_read_alone_writes_no_retrain_text(swept_read):
    assert not (swept_read.ref.sharper / "sharper.txt").exists()
    assert "Reading M" not in swept_read.text


def test_read_after_a_retrain_adds_reading_m_against_the_reference_cells(trained, capsys, monkeypatch):
    """With both phases on disk the read pairs the retrained cells against the
    M3c ones on the same windows and prints Reading M beside Reading N."""
    monkeypatch.setattr(script, "reading_noise", _reading(SweepStatus.NOISE_LIMITED, RETRAIN_TAU))
    assert _run(trained, "--phase", "evaluate", "--steps", "4") == script.EXIT_OK
    capsys.readouterr()
    assert _run(trained, "--phase", "read", "--steps", "4") == script.EXIT_OK
    text = capsys.readouterr().out
    assert "--- Reading M: the retrain at tau=0.5" in text
    assert (trained.sharper / "sharper.txt").read_text() in text
    assert "verdict: random_vit" in text


def test_reading_m_is_the_retrain_minus_the_m3c_cells_not_the_other_way(trained, capsys, monkeypatch):
    """Reading M's orientation is the study's central claim: `retrain - M3c`.
    Swapped, "sharper is worse" reads as "sharper is better" and the
    milestone reports the opposite of what it measured. Pinned through
    `retrain_inputs` itself against a difference computed straight off the
    two records, as Reading N's orientation is pinned through `_tau_inputs`.
    """
    import types as _types

    monkeypatch.setattr(script, "reading_noise", _reading(SweepStatus.NOISE_LIMITED, RETRAIN_TAU))
    assert _run(trained, "--phase", "evaluate", "--steps", "4") == script.EXIT_OK
    capsys.readouterr()

    status, records = script.load_sweep(
        _types.SimpleNamespace(sweep_out=trained.sweep), [(JOB.arm, JOB.seed)]
    )
    assert status == script.EXIT_OK
    h = int(records[(JOB.arm, JOB.seed)]["decision_h"])
    new_trust = script._load_trust(trained.sharper, [(JOB.arm, JOB.seed)])
    old_trust = script._reference_trust(records, [(JOB.arm, JOB.seed)])
    assert new_trust and old_trust

    inputs = script.retrain_inputs(
        new_trust, old_trust, arms=[JOB.arm], seeds=[JOB.seed], h=h, tau=RETRAIN_TAU,
    )

    def survival(source, channel):
        return survival_indicator(source[(JOB.arm, JOB.seed)], channel, h)

    for channel, contrast in (("free", inputs.arms[JOB.arm].free),
                              ("probe", inputs.arms[JOB.arm].probe)):
        values, changed = survival(new_trust, channel)
        control, control_changed = survival(old_trust, channel)
        keep = changed & control_changed
        direct = float(np.mean(values[keep] - control[keep]))
        assert contrast.estimate == pytest.approx(direct, abs=1e-12), (channel, contrast.estimate, direct)


def test_read_is_idempotent(swept_read, capsys):
    assert _run(swept_read.ref, "--phase", "read") == script.EXIT_OK
    assert capsys.readouterr().out == swept_read.text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_sharper_latent_script.py -q -k "read"`
Expected: FAIL — `NotImplementedError: read is Task 8`.

- [ ] **Step 3: Write the read half**

Replace Task 7's placeholder with, before the `# main` banner:

```python
# ---------------------------------------------------------------------------
# read: the tables, the two readings, sweep.txt and sharper.txt.
# ---------------------------------------------------------------------------


def _num(value, spec: str = ".3f") -> str:
    return fmt_z(float(value), spec)


def _cells_in_order(records: dict) -> list[tuple[str, int]]:
    return sorted(records, key=lambda c: (_ordered([c[0]] + list(ARMS_ORDER)).index(c[0]), c[1]))


def _self_check_table(records: dict) -> str:
    lines = [
        "--- self-check per cell: the tau=1.0 pass against the cell's own diagnostic (exact) ---",
        f"  {'arm':<12}{'seed':>5}{'step':>7}{'ref max|d|':>12}{'pers max|d|':>13}{'windows':>9}{'clusters':>10}",
    ]
    for arm, seed in _cells_in_order(records):
        r = records[(arm, seed)]
        check = r["self_check"]
        lines.append(
            f"  {arm:<12}{seed:>5}{int(r['step']):>7}"
            f"{_num(check['reference_position_max_delta'], '.1e'):>12}"
            f"{_num(check['persistence_position_max_delta'], '.1e'):>13}"
            f"{int(r['windows']['total']):>9}{int(r['windows']['clusters']):>10}"
        )
    return "\n".join(lines) + "\n"


def _gate_table(records: dict, taus) -> str:
    """`gap_closed(45)` per seed at every temperature, with the unanimity flag.
    Reported, never decided on (spec 3.2)."""
    # The seed columns are the records' own, not a hardcoded (0, 1, 2): a run
    # over a subset would otherwise print seed 1's gap under an `s0` caption.
    seed_columns = sorted({int(seed) for _, seed in records})
    lines = [
        "--- the gate at every temperature: gap_closed at the horizon on position per seed "
        "(NaN = non-positive band); GATE PASSES = > 0 in every seed. Reported, not decided on. ---",
        f"  {'arm':<12}{'tau':>5}" + "".join(f"{f's{s}':>10}" for s in seed_columns)
        + f"{'passes':>9}{'degen(max)':>12}",
    ]
    arms = sorted({arm for arm, _ in records}, key=lambda a: _ordered([a] + list(ARMS_ORDER)).index(a))
    for arm in arms:
        seeds = sorted(seed for a, seed in records if a == arm)
        for tau in taus:
            gaps = [
                float(_entry(records[(arm, s)], tau)["gate"]["gap_final"])
                if (arm, s) in records else float("nan")
                for s in seed_columns
            ]
            degen = max(int(_entry(records[(arm, s)], tau)["gate"]["degenerate"]) for s in seeds)
            passes = bool(gaps) and all(np.isfinite(g) and g > 0 for g in gaps)
            cells = "".join(f"{_num(g, '+.4f'):>10}" for g in gaps)
            lines.append(
                f"  {arm:<12}{float(tau):>5.1f}{cells}{'YES' if passes else 'no':>9}{degen:>12}"
            )
    return "\n".join(lines) + "\n"


def _noise_table(records: dict, taus, horizon: int) -> str:
    """The statistic that motivated the study: two draws of the same model
    against that model's own imagined displacement, at every temperature."""
    hs = [h for h in (1, 5, 15, 45) if h <= horizon]
    lines = [
        "--- the noise reference against the imagined displacement (medians over windows, "
        "embedding units through the same head): two draws of one model, then how far that "
        "model imagined it moved ---",
        f"  {'arm':<12}{'tau':>5}" + "".join(f"{f'noise({h})':>12}{f'moved({h})':>12}" for h in hs),
    ]
    for arm, seed in _cells_in_order(records):
        r = records[(arm, seed)]
        for tau in taus:
            entry = _entry(r, tau)
            noise = np.asarray(entry["noise"]["curve"], dtype=float)
            moved = np.asarray(entry["displacement"]["curve"], dtype=float)
            cells = "".join(
                f"{_num(noise[h - 1]):>12}{_num(moved[h - 1]):>12}" for h in hs
            )
            lines.append(f"  {arm + ' s' + str(seed):<12}{float(tau):>5.1f}{cells}")
    return "\n".join(lines) + "\n"


def _survival_table(records: dict, taus, horizon: int) -> str:
    hs = [h for h in (1, 2, 3, 5, 10, 15, 30, 45) if h <= horizon]
    lines = [
        "--- survival by temperature: S(h) = fraction of moved draws with h_x > h, free channel, "
        "seeds stacked ---",
        f"  {'arm':<12}{'tau':>5}" + "".join(f"{f'S({h})':>8}" for h in hs),
    ]
    arms = sorted({arm for arm, _ in records}, key=lambda a: _ordered([a] + list(ARMS_ORDER)).index(a))
    for arm in arms:
        seeds = sorted(seed for a, seed in records if a == arm)
        for tau in taus:
            stacked = np.concatenate([
                np.asarray(_entry(records[(arm, s)], tau)["summary"]["survival"]["free"], dtype=float)[None, :]
                for s in seeds
            ])
            lines.append(
                f"  {arm:<12}{float(tau):>5.1f}"
                + "".join(f"{_num(stacked[:, h - 1].mean()):>8}" for h in hs)
            )
    return "\n".join(lines) + "\n"


def write_curves(records: dict, taus, figure: Path) -> str:
    """Two panels against the temperature: S(15) on the free channel and the
    gate's own metric, arms coloured, seeds thin. A missing or broken
    matplotlib, or an unwritable path, costs the FIGURE and nothing else."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, ValueError, OSError) as error:
        return f"figure NOT written: {error}"
    colours = {"pixel_ae": "tab:orange", "frozen_ssl": "tab:blue", "random_vit": "tab:gray"}
    order = [float(t) for t in taus]
    x = np.arange(len(order))
    h = int(next(iter(records.values()))["decision_h"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    try:
        survival, gate = axes.flat
        for (arm, seed), record in records.items():
            colour = colours.get(arm, "black")
            survival.plot(
                x, [float(np.asarray(_entry(record, t)["summary"]["survival"]["free"])[h - 1]) for t in order],
                marker="o", markersize=3, linewidth=0.9, color=colour, label=f"{arm} s{seed}",
            )
            gate.plot(
                x, [float(_entry(record, t)["gate"]["gap_final"]) for t in order],
                marker="o", markersize=3, linewidth=0.9, color=colour,
            )
        gate.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
        survival.set_title(f"S({h}), free channel")
        gate.set_title("gap_closed at the horizon (position)")
        for ax in (survival, gate):
            ax.set_xticks(x)
            ax.set_xticklabels([f"{t:.1f}" for t in order])
            ax.set_xlabel("rollout sampling temperature")
        handles, labels = survival.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), fontsize=8)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=110)
    except (ValueError, OSError) as error:
        return f"figure NOT written: {error}"
    finally:
        plt.close(fig)
    return f"figure={figure}"


def sweep_text(records: dict, *, arms, seeds, taus, figure_line: str) -> str:
    """Everything the sweep half of `read` prints, written byte-identical."""
    horizon = int(next(iter(records.values()))["horizon"])
    h = int(next(iter(records.values()))["decision_h"])
    _, clamped = decision_horizon(horizon)
    inputs = sweep_inputs(records, arms=arms, seeds=seeds, h=h, taus=taus)
    reading = reading_noise(inputs)
    note = (f"  NOTE: decision horizon clamped to the run's horizon h={h} "
            f"(pre-registered DECISION_H = {DECISION_H}).\n" if clamped else "")
    return "".join([
        _self_check_table(records),
        _gate_table(records, taus),
        _noise_table(records, taus, horizon),
        _survival_table(records, taus, horizon),
        note,
        f"  pooling: z_fam = cluster_threshold({SWEEP_FAMILY}, {_clusters(records)}) = "
        f"{inputs.z_fam:.2f}; every contrast is this cell at this temperature against the same "
        f"cell at tau={REFERENCE_TAU} on the same val windows; a temperature counts for an arm "
        f"when it clears pooled and in >= {SEEDS_REQUIRED} of its seeds, and the status needs "
        f"{ARMS_REQUIRED} arms.\n",
        format_reading_noise(reading, inputs),
        f"  {figure_line}\n",
    ])


def retrain_inputs(new_trust: dict, old_trust: dict, *, arms, seeds, h: int, tau: float) -> RetrainInputs:
    """Reading M's inputs: each retrained cell paired against the M3c cell of
    the same arm and seed on the same windows, both channels."""
    def contrast(arm, channel, seed_list):
        def series(records, label):
            out = []
            for s in seed_list:
                record = records[(arm, int(s))]
                values, changed = survival_indicator(record, channel, h)
                out.append(cell_series(
                    record, values, changed, arm=label, seed=int(s), rung="val",
                    channel=f"S/{channel}", val=record["episodes"]["val"],
                    horizon=record["horizon"], context=record["context"],
                    device=record["device"], torch_version=record["torch_version"],
                ))
            return out

        return _paired(series(new_trust, f"{arm}@sharper"), series(old_trust, f"{arm}@m3c"))

    def arm_inputs(arm, seed_list, per_seed):
        return RetrainArm(
            free=contrast(arm, "free", seed_list),
            probe=contrast(arm, "probe", seed_list),
            per_seed=per_seed,
        )

    return RetrainInputs(
        arms={
            arm: arm_inputs(
                arm, seeds,
                {int(s): arm_inputs(arm, [s], None) for s in seeds},
            )
            for arm in _ordered(arms)
        },
        z_fam=pooling.cluster_threshold(RETRAIN_FAMILY, _clusters_trust(new_trust)),
        h=h, tau=float(tau),
    )


def _clusters_trust(records: dict) -> int:
    first = next(iter(records.values()))
    return int(np.unique(np.asarray(first["windows"]["episode"])).size)


def _load_trust(directory: Path, cells) -> dict:
    """The trust records of a study directory, or an empty dict when any is
    missing -- the retrain half of the read is simply not printed then."""
    records = {}
    for arm, seed in cells:
        path = Path(directory) / f"trust_{arm}_seed{seed}.json"
        if not path.exists():
            return {}
        records[(arm, int(seed))] = load_record(path)
    return records


def _reference_trust(records: dict, cells) -> dict:
    """The M3c side of Reading M, taken from the sweep's own REFERENCE_TAU
    entry.

    ALWAYS this, never a `trust_<arm>_seed<n>.json` that happens to sit in
    `--reference`. The shipped M3c directory does carry M3d's trust records,
    so preferring them would have meant the real run reading a file and every
    test reading this -- two paths, and the untested one shipping. Beyond
    that, this entry is the better control: `load_sweep` has already proven
    it bitwise identical to the M3c diagnostic in THIS run (exit 30
    otherwise), it scores the same windows at the same protocol as the
    retrain it is compared against, and it carries `crossing`/`windows` in
    the shape a trust record does, for both channels."""
    out = {}
    for arm, seed in cells:
        record = records[(arm, int(seed))]
        summary = _entry(record, REFERENCE_TAU)["summary"]
        out[(arm, int(seed))] = {
            "crossing": summary["crossing"],
            "windows": summary["windows"],
            "episodes": record["episodes"],
            "horizon": record["horizon"],
            "context": record["context"],
            "device": record["device"],
            "torch_version": record["torch_version"],
        }
    return out


def sharper_text(new_trust: dict, old_trust: dict, *, arms, seeds, h: int, tau: float) -> str:
    inputs = retrain_inputs(new_trust, old_trust, arms=arms, seeds=seeds, h=h, tau=tau)
    reading = reading_sharper(inputs)
    return "".join([
        f"  pooling: z_fam = cluster_threshold({RETRAIN_FAMILY}, {_clusters_trust(new_trust)}) = "
        f"{inputs.z_fam:.2f}; each retrained cell is paired against the M3c cell of the same arm "
        "and seed on the same val windows.\n",
        format_reading_sharper(reading, inputs),
    ])


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def read_phase(args, cells) -> int:
    """The sweep's records (11, 30) and its reading always; the retrain's
    reading too when the retrained cells' own trust records are on disk.

    `old_trust`, Reading M's control arm, is ALWAYS `_reference_trust` -- the
    sweep's own REFERENCE_TAU entry, which `load_sweep` has already proven
    bitwise identical to the M3c diagnostic in this run. It is never a
    `trust_<arm>_seed<n>.json` that happens to sit under `--reference`: the
    shipped M3c directory carries M3d's, so preferring those would mean the
    real run reading files and every test reading the derived path."""
    status, records = load_sweep(args, cells)
    if status != EXIT_OK:
        return status
    taus = [float(t) for t in next(iter(records.values()))["taus"]]
    arms, seeds = list(args.arms), [int(s) for s in args.seeds]
    figure = args.figure if args.figure is not None else args.sweep_out / "sweep_curves.png"
    figure_line = write_curves(records, taus, figure)
    text = sweep_text(records, arms=arms, seeds=seeds, taus=taus, figure_line=figure_line)
    print(text, end="")
    write_text(args.sweep_out / "sweep.txt", text)

    new_trust = _load_trust(args.out, cells)
    old_trust = _reference_trust(records, cells)
    if new_trust and old_trust:
        retrain = load_record(retrain_record_path(args.out, *cells[0]))
        h = int(next(iter(records.values()))["decision_h"])
        retrain_text = sharper_text(
            new_trust, old_trust, arms=arms, seeds=seeds, h=h, tau=float(retrain["tau"]),
        )
        print(retrain_text, end="")
        write_text(args.out / "sharper.txt", retrain_text)
    return EXIT_OK
```

**One thing to check while implementing:** `survival_indicator` reads `summary["crossing"][channel]`, and a trust record stores its crossings under the same key (`crossing: {probe, free}`), so a trust record can be passed where a stratum summary is expected. Confirm that by reading `scripts/trust_horizon.py`'s `trust_record`; if the shapes differ, build the indicator from the trust record's `crossing` directly with the same rule (`(crossing > h)`, `isfinite(crossing)`) and say so in your report.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/eval/test_sharper_latent_script.py -q`
Expected: every test in the file passes (Tasks 6, 7 and 8 together).

- [ ] **Step 5: Commit**

```bash
git add scripts/sharper_latent.py tests/eval/test_sharper_latent_script.py
git commit -m "feat: sharper_latent.py read -- the self-check, gate, noise and survival tables by temperature, Reading N always and Reading M when a retrain exists, sweep.txt and sharper.txt byte-identical to what is printed

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: The exit-status registry, the suite, the smoke

**Files:**
- Modify: `tests/eval/test_diagnose_dynamics_script.py` (the registry test)
- Run artefacts (gitignored): `runs/m3h_smoke/`

- [ ] **Step 1: Extend the registry test**

In `test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own`, after the `stage_decomposition` clause, add:

```python
    sharper = statuses("sharper_latent")
    reused_by_sharper = {**reused, "EXIT_SELF_CHECK_FAILED": 30}
    shared_with_trust = {
        name: value for name, value in sharper.items() if value in set(trust.values()) - {0}
    }
    assert shared_with_trust == reused_by_sharper, (
        f"sharper_latent shares {shared_with_trust} with trust_horizon; only "
        f"{reused_by_sharper} is shared on purpose"
    )
    assert len(set(sharper.values())) == len(sharper), sharper
    assert 1 not in sharper.values() and 2 not in sharper.values()
    assert sharper["EXIT_NOT_NOISE_LIMITED"] == 35
    assert sharper["EXIT_TEMPERATURE_MISMATCH"] == 36
    assert sharper["EXIT_IDENTITY_CHECK_FAILED"] == 37
    own = set(sharper.values()) - set(reused_by_sharper.values()) - {0}
    assert own == {35, 36, 37}
    for other in ("run_study", "report_study", "pool_dynamics", "diagnose_dynamics", "split_gap",
                  "checkpoint_ladder", "stage_decomposition"):
        clash = own & set(statuses(other).values())
        assert not clash, f"sharper_latent collides with {other} on {clash}"
```

and add `"sharper_latent"` to the `for other in (...)` tuples of the `checkpoint_ladder` and `stage_decomposition` clauses.

- [ ] **Step 2: Run the registry test**

Run: `.venv/bin/python -m pytest tests/eval/test_diagnose_dynamics_script.py -k distinct -q`
Expected: 1 passed.

- [ ] **Step 3: Run the whole suite**

Run: `.venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -4`
Expected: every test passes, 0 skipped with `runs/` reachable, 0 warnings. Record the count; it goes into `## Task 10 results` from this output, not predicted here.

- [ ] **Step 4: Commit**

```bash
git add tests/eval/test_diagnose_dynamics_script.py
git commit -m "test: the exit-status registry gains sharper_latent -- 35/36/37 new, 11/12/14/30 trust_horizon's, no clash with any other tool

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 5: The smoke run**

One cell, the full grid, into a directory the real run never reads:

```bash
mkdir -p runs/m3h_smoke && git rev-parse HEAD > runs/m3h_smoke/sweep.head && \
PYTHONUNBUFFERED=1 .venv/bin/python scripts/sharper_latent.py \
  --sweep-out runs/m3h_smoke --out runs/m3h_smoke_sharper \
  --arms pixel_ae --seeds 0 --phase sweep > runs/m3h_smoke/sweep.log 2>&1; \
echo $? | tee runs/m3h_smoke/sweep.exit
.venv/bin/python scripts/sharper_latent.py --sweep-out runs/m3h_smoke --out runs/m3h_smoke_sharper --arms pixel_ae --seeds 0 --phase read
```

Expected: exit **0** in about ten minutes (five passes of ~1 min plus the probe refit). The τ = 1.0 self-check reads exactly 0.0; the sweep table shows five temperatures; the single-seed run cannot make any status but `NOT_NOISE_LIMITED` (one arm, `ARMS_REQUIRED = 2`), and the sentence must say so — read it and check that it names the arm count rather than claiming anything about the sampler.

- [ ] **Step 6: Read the smoke's `sweep.txt` for format**

Check, and fix in the script with a test where wrong: every table's header columns align with its rows (M3f shipped a missing space between two header columns — look for it); the gate table's `GATE PASSES` flags read correctly; the noise table's medians fall with τ and reach 0.0 at τ = 0; `NOTE:` appears only if the horizon clamps (it should not, at horizon 45). Commit any fix, then re-run the whole suite if the script changed.

---

### Task 10: The run and the results

**Files:**
- Run artefacts (gitignored): `runs/m3h_sweep/`, and `runs/m3h_sharper/` only if the gate opens
- Modify: this plan (`## Task 10 results`)

- [ ] **Step 1: Confirm the state the run is stamped with**

```bash
git status --porcelain && git log --oneline -1
```

Expected: empty status and the HEAD the suite was green at. Do not commit anything until the run has finished.

- [ ] **Step 2: Phase 1 — the sweep**

```bash
mkdir -p runs/m3h_sweep && git rev-parse HEAD > runs/m3h_sweep/sweep.head && \
date -u +%Y-%m-%dT%H:%M:%SZ > runs/m3h_sweep/sweep.started && \
nohup sh -c 'PYTHONUNBUFFERED=1 caffeinate -dimsu .venv/bin/python scripts/sharper_latent.py --phase sweep > runs/m3h_sweep/sweep.log 2>&1; echo $? > runs/m3h_sweep/sweep.exit; date -u +%Y-%m-%dT%H:%M:%SZ > runs/m3h_sweep/sweep.finished' > /dev/null 2>&1 &
```

Nine cells × five temperatures at about a minute a pass: **~1 h**. Then read it:

```bash
.venv/bin/python scripts/sharper_latent.py --phase read 2>&1 | tail -40
```

- [ ] **Step 3: Sweep acceptance**

```bash
.venv/bin/python - <<'EOF'
import glob, json
from pathlib import Path
head = Path("runs/m3h_sweep/sweep.head").read_text().strip()
records = [json.load(open(p)) for p in sorted(glob.glob("runs/m3h_sweep/sweep_*.json"))]
assert len(records) == 9, len(records)
assert {r["git_sha"] for r in records} == {head}
assert {r["device"] for r in records} == {"mps"}
assert all(r["self_check"]["ok"] and r["self_check"]["reference_position_max_delta"] == 0.0
           and r["self_check"]["persistence_position_max_delta"] == 0.0 for r in records)
assert {(r["windows"]["total"], r["windows"]["clusters"]) for r in records} == {(229, 24)}
assert {tuple(r["taus"]) for r in records} == {(1.0, 0.7, 0.5, 0.3, 0.0)}
assert {r["decision_h"] for r in records} == {15} and {r["step"] for r in records} == {20000}
print("OK: nine sweep records at one git_sha ==", head[:7], "on mps; tau=1.0 self-check 0.0 on 9/9; 229/24 at every temperature")
EOF
grep -n "verdict:" runs/m3h_sweep/sweep.txt
```

- [ ] **Step 4: Phase 2 — the retrain, only if the gate opens**

**If Reading N is `NOT_NOISE_LIMITED` or `SHARPER_WORSE`: STOP.** That is M3h's complete result. Skip to Step 6, and write the results section around the sweep alone — the entropy hypothesis is refuted at this grid, Phase 2 never runs (the script will refuse it with exit 35), and the next study is the shrink or KL-balance option with this evidence.

If it is `NOISE_LIMITED`:

```bash
mkdir -p runs/m3h_sharper && git rev-parse HEAD > runs/m3h_sharper/sharper.head && \
date -u +%Y-%m-%dT%H:%M:%SZ > runs/m3h_sharper/sharper.started && \
nohup sh -c 'PYTHONUNBUFFERED=1 caffeinate -dimsu .venv/bin/python scripts/sharper_latent.py --phase train > runs/m3h_sharper/train.log 2>&1 && PYTHONUNBUFFERED=1 caffeinate -dimsu .venv/bin/python scripts/sharper_latent.py --phase evaluate > runs/m3h_sharper/evaluate.log 2>&1; echo $? > runs/m3h_sharper/sharper.exit; date -u +%Y-%m-%dT%H:%M:%SZ > runs/m3h_sharper/sharper.finished' > /dev/null 2>&1 &
```

~15 h 30 m. Then `--phase read` again for Reading M.

- [ ] **Step 5: Retrain acceptance**

```bash
.venv/bin/python - <<'EOF'
import glob, json, torch
from pathlib import Path
head = Path("runs/m3h_sharper/sharper.head").read_text().strip()
records = [json.load(open(p)) for p in sorted(glob.glob("runs/m3h_sharper/retrain_*.json"))]
assert len(records) == 9, len(records)
assert {r["git_sha"] for r in records} == {head}
taus = {r["tau"] for r in records}
assert len(taus) == 1, taus
assert all(r["identity"]["max_delta"] == 0.0 and r["identity"]["first_step"] is None for r in records)
assert {r["steps"] for r in records} == {20000}
tau = taus.pop()
paths = sorted(glob.glob("runs/m3h_sharper/world_model_*.pt"))
assert len(paths) == 9, paths
for p in paths:
    payload = torch.load(p, map_location="cpu", weights_only=True)
    assert payload["sample_temperature"] == tau, (p, payload["sample_temperature"])
print("OK: nine retrains at tau =", tau, "one git_sha ==", head[:7], "; identity 0.0 on 9/9; every payload carries the temperature")
EOF
```

- [ ] **Step 6: Write `## Task 10 results` into this plan**

Under a new `## Task 10 results` heading at the end of this plan, numbers copied from `sweep.txt` / `sharper.txt` and the records, not rounded:

1. **Provenance** — the `git_sha`, device, torch version, the `.started` / `.finished` markers and wall time, the exit codes, and that `sweep.txt` (and `sharper.txt`) are byte-identical to what `read` printed.
2. **Suite** — the whole-suite count at the launched HEAD (Task 9 Step 3), 0 skipped, 0 warnings.
3. **Acceptance** — Steps 3 and, if it ran, 5.
4. **The self-check table** — 0.0 on 9/9 at τ = 1.0.
5. **Reading N** — the whole table: per arm and τ the estimate ± se, z, seeds, whether it counts; the status, τ* and the sentence; the endpoint row and what it shows.
6. **The gate at every temperature**, the noise-vs-displacement table, the survival table — copied.
7. **If Phase 2 ran:** the identity check; Reading M per arm on both channels with the per-seed statuses; the gate on the retrained cells; the KL regime of the new cells beside M3c's; the validation objective; the M3g stage decomposition re-run on them as a companion.
8. **Read against §4** — one paragraph per arm's status, what §3.2/§3.3 said would follow, and the non-claims restated as measured: no arm ranked, no checkpoint altered, τ* the best of a three-value grid, the KL regime moved and by how much, and what a `NOT_NOISE_LIMITED` does and does not refute.

- [ ] **Step 7: Commit the results**

```bash
git add docs/superpowers/plans/2026-09-23-mb-fps-m3h-sharper-latent.md
git commit -m "docs: M3h -- <Reading N's status and tau*, and Reading M's if it ran>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Exit criteria

- [ ] Tasks 1–8 committed, each reviewed; Task 9's registry clause committed; the whole suite green with 0 warnings at the launched HEAD.
- [ ] The smoke run exit 0, its `sweep.txt` read for format, any defect fixed before the real run.
- [ ] Nine sweep records at one `git_sha` == HEAD on `mps`; the τ = 1.0 self-check exactly 0.0 on 9/9; 229 windows / 24 clusters at every temperature; `sweep.exit` 0.
- [ ] Reading N read and recorded. If `NOISE_LIMITED`: nine retrains at the τ* it named, the identity check exact, every checkpoint carrying that temperature, Reading M read. If not: Phase 2 not run, and the results say so as the milestone's result.
- [ ] `## Task 10 results` written, numbers copied, each status stated against §3.2/§3.3 and the closing paragraphs against §4.

## Plan self-review (writing-plans)

- **Spec coverage.** §1 why → the plan's Goal and Task 4's docstrings. §2.1 the sweep → Tasks 3, 6. §2.2 the retrain → Tasks 1, 7. §2.3 the payload → Task 2. §3.1 pooling → Tasks 7 (`sweep_inputs`), 8 (`retrain_inputs`). §3.2 Reading N → Task 4, gated in Task 7. §3.3 Reading M → Tasks 5, 8. §3.4 companions → Task 8's tables and Task 10's results list (the KL regime, the objective and the stage re-run are read off the retrained cells' records in the results step). §4 non-claims → Task 10 Step 6. §5 code shape → Tasks 1–8 as named; the registry → Task 9. §6 the run → Tasks 9–10. §7 files → the file-structure table.
- **Placeholders.** None: every code step carries its code. Task 7's `read_phase` placeholder is explicitly removed in Task 8, and the results section lists what is copied rather than what it will say.
- **Type consistency.** `TauInputs(free, per_seed)` / `SweepArm(taus)` / `SweepInputs(arms, z_fam, h)` (Task 4) are what Task 7's `sweep_inputs` builds; `RetrainArm(free, probe, per_seed)` / `RetrainInputs(arms, z_fam, h, tau)` (Task 5) are what Task 8's `retrain_inputs` builds; `StratumContrast(estimate, se, z, clusters)` is `split_gap`'s throughout; `tau_key` is the one spelling of a temperature in a record, used by `_entry` and by the tests; `sample_temperature` is the one spelling of the parameter in `TrainConfig`, `RSSMConfig`, the payload, `get_config`, `prepare_cell`'s args and `evaluate_job`; `SAMPLE_TEMPERATURE` (rssm) and `REFERENCE_TAU` (sharper) are both 1.0 and are asserted equal in Task 4's constants test.
