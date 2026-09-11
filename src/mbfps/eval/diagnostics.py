"""Two diagnostics on the SHIPPED M3b checkpoints -- milestone M3b, no retraining.

The nine-cell study reads NOT PASSED, and no arm beats persistence at any
horizon past h=16. Because EVERY arm loses, the evidence points at what the arms
SHARE -- the RSSM, the free-bits schedule, the 20k budget -- rather than at the
representation contrast the study was built to test. These two diagnostics turn
that inference into a mechanism, and both run on the frozen checkpoints:

  AN INTERVENTION LADDER ON THE HORIZON ACTIONS. Re-run the open-loop
  imagination with the horizon actions perturbed, everything else identical.
  THREE RUNGS, in strictly increasing order of perturbation, because a null on
  one of them is a much weaker statement than a null on all three:

    shuffled  -- the window's own actions, PERMUTED. Same multiset, different
                 order, so this tests ORDER sensitivity and nothing else.
    resampled -- i.i.d. draws from the empirical action marginal of the scored
                 windows. Different multiset, same marginal: each individual
                 action stays plausible while the COUNTS are randomised. This
                 is the rung the permutation cannot reach, because dynamics
                 that read how many of each action the horizon contains --
                 rather than in what order -- are bitwise action-blind under a
                 permutation while being fully action-conditioned.
    constant  -- EVERY action in the scored support held for the whole horizon,
                 one arm per action, and the rung's statistic is the paired
                 CONTRAST between two named held actions: MOVE_FORWARD minus
                 NOOP by default. This asks whether the identity of the action
                 reaches the prior at all, and the sign is a physical
                 prediction -- an action-conditioned prior must run the
                 imagined position further under the displacing action than
                 under the stationary one.

  THE RUNGS ARE NOT EQUALLY ON-DISTRIBUTION, and that asymmetry is what
  decides how a response at each is read. The permutation keeps every step
  and every count. The resample keeps the marginal but not the sequence
  statistics -- in the data P(a_t = a_{t-1}) is 0.31 against 0.22 i.i.d. A held
  action is 45 identical steps where the longest run in the whole dataset is
  13, so "trained on" is true only per step: a NULL there is strong evidence,
  but a difference between ONE held action and the real sequence is weak,
  because confusion under an off-distribution input and conditioning on the
  action look the same. That is why the top rung is a contrast between two
  held actions rather than one held action against the real sequence: both
  sides are equally off-distribution, so what is left between them is the
  action's identity, and every held action's own delta from the real sequence
  is recorded beside it for the reader. Measured on the shipped checkpoints
  the choice is not academic -- holding STRAFE_RIGHT reads as a null on all
  nine cells while holding MOVE_FORWARD through the identical machinery moves
  frozen_ssl/0 by +31 map units, so a single held action asks the question
  with whichever action it happens to hold.

  If every rung is null the conclusion is a bound on THESE perturbations --
  order, counts, and the named contrast -- and not on "an action effect" in
  general. If a LATER rung resolves where an earlier one did not, that is a
  different and better finding -- the model is action-conditioned but
  insensitive to what the earlier rung perturbs -- and it must be reported as
  that, not folded into the first rung's story.

  K-STEP RE-GROUNDING SWEEP. Imagine k steps, re-observe the REAL frame, repeat
  across the horizon. This localises the horizon at which error stops being
  recoverable, by interpolating between the pure open loop (k = horizon) and a
  one-step-ahead prior (k = 1).

THREE THINGS MAKE THESE NUMBERS MEAN ANYTHING, and each is a way to be wrong
without breaking a shape.

1. THE WINDOW SET IS `evaluate_rollout`'s, EXACTLY. Every index convention in
   that function is load-bearing -- the `< need + 1` length guard, the `+ 1` on
   the stride's stop, the `[1:]` on the embedding window, the
   `start + context + 1` truth slice -- and a diagnostic that cuts different
   windows reports curves that are not comparable to `curves["rssm_position"]`
   in the records at all. The window rule is taken from `eval.windows`, which
   `evaluate_rollout` itself consumes; the rest of the protocol is reproduced
   here call for call and pinned by the requirement below.

2. THE ARMS SHARE A SAMPLING STREAM, PER WINDOW. `RSSM._sample` is stochastic at
   evaluation BY DESIGN -- its docstring records that taking the mode collapses
   45 imagined latents to 3 and roughly quadruples position error -- so two arms
   that draw from different points of the stream differ by sampling noise, and
   the measured effect is uninterpretable. Every arm here restarts from ONE
   per-window RNG snapshot taken after the context filter, and the window ends
   with the canonical open-loop pass replayed from that same snapshot. Two
   consequences fall out structurally rather than by assertion:

     * the canonical pass IS `evaluate_rollout`'s -- same call, same inputs,
       same point in the stream -- so the reference curves, the floor and
       persistence are bitwise the record's, and the stream leaves each window
       exactly where `evaluate_rollout` leaves it;
     * every arm's first imagined step draws the same uniforms, so a difference
       at horizon step 1 can only be the intervention.

   The snapshot MUST be device-aware. Measured on this box: `torch.get_rng_state`
   / `torch.set_rng_state` and `torch.random.fork_rng(devices=[])` restore the
   CPU generator only and do NOT restore the MPS one -- and the shipped records
   were produced on MPS. A CPU-only snapshot there breaks record reproduction
   AND inflates the apparent action effect by ~100x, i.e. it manufactures
   exactly the artefact the matched stream exists to prevent, while every CPU
   test still passes. `_rng_snapshot` therefore REFUSES an accelerator it has no
   verified snapshot for rather than quietly falling back.

   What makes a matched stream possible at all is measured, not argued:
   `torch.distributions.Categorical(probs=...).sample()` consumes a number of
   uniforms determined by SHAPE alone, not by the probability values -- checked
   on both CPU and MPS. So permuting actions, or re-grounding between segments,
   changes what is sampled without changing how much is sampled at a given step.

3. THE RE-GROUNDING NEVER SEES THE FRAME IT IS SCORED ON. Segment `s` covers
   horizon steps `[s*k, min((s+1)*k, H))` and is grounded by the posterior over
   the real frames of segment `s-1`, warm-started from that segment's own
   starting state -- so the last real frame the grounding consumed is
   `start + context + s*k`, exactly one before the first frame segment `s`
   predicts. THE FLOOR IS NOT THE k -> 0 LIMIT OF THIS SWEEP: the floor is a
   ZERO-step posterior at every future step, seeing the frame it is scored on,
   while k=1 is a ONE-step PRIOR from a posterior-grounded state. k=1 must
   therefore sit STRICTLY above the floor; a k=1 that equals it means the
   grounding consumed the scored frame and the whole sweep is meaningless.

WHY THE ROLLOUT BODY IS DUPLICATED HERE AND ONLY THE WINDOW RULE IS SHARED.
Rewriting `evaluate_rollout` on top of this core would make the k=horizon
degeneracy true by construction and turn the strongest available correctness
evidence -- a BITWISE equality against nine cells of shipped numbers, on a
stochastic model -- into decoration. It would also put the provenance of those
nine cells at risk for no measurable gain. The window rule is shared because its
failure mode is the opposite: silent drift with no loud test available.
"""

from dataclasses import dataclass

import numpy as np
import torch
from numpy.random import default_rng

from mbfps.data.episode import load_episode
from mbfps.eval.probe import (
    angle_error_degrees,
    apply_probe,
    position_error,
    probe_targets,
)
from mbfps.eval.rollout import RolloutResult, source_for
from mbfps.eval.windows import window_starts

REGROUNDING_KS: tuple[int, ...] = (1, 3, 5, 15, 45)
"""The sweep's re-grounding periods, at the study's horizon of 45.

Every one of them divides 45, so the ragged final segment (`horizon % k != 0`)
never runs under the shipped configuration -- it is reachable only through the
CLI's `--horizon`, and is exercised deliberately in the tests rather than
shipped untested.

45 is in the tuple because the k == horizon pass IS the self-check: it must
reproduce the pure open-loop rollout bitwise, and a sweep without it reports
curves nothing has checked against the shipped protocol.
"""

METRICS: tuple[str, ...] = ("position", "angle")

LADDER: tuple[str, ...] = ("shuffled", "resampled", "constant")
"""The intervention rungs, in STRICTLY INCREASING order of perturbation.

The order is the finding's whole shape. `shuffled` preserves the action
multiset, `resampled` preserves only its marginal, `constant` preserves
nothing, so a reader going left to right is reading how much has to be
destroyed before the prior notices -- and a report that printed the rungs in
whatever order a flag was typed in would let a non-monotone ladder read as a
monotone one. Every consumer that orders rungs takes the order from here.

This tuple is the ONLY definition of which rungs exist; `ladder_order` refuses
anything not in it rather than dropping it silently, because a mistyped rung
that vanished would leave a report missing a column nobody goes looking for.
"""

LADDER_PERTURBS: dict[str, str] = {
    "shuffled": "the order of the horizon actions",
    "resampled": "the counts of each action in the horizon",
    "constant": "the identity of the action held for the whole horizon",
}
"""What each rung destroys, in the words the verdict uses.

A rung that resolves above a null one licenses exactly one sentence -- "the
dynamics are action-conditioned but insensitive to <what the null rung
perturbs>" -- and this is where that clause comes from. It lives beside
`LADDER` rather than in the script because a rung added to one and not the
other would leave the deciding sentence with a hole in it; the test pins the
two to the same set.
"""

ACTION_NAMES: tuple[str, ...] = (
    "NOOP", "TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD", "MOVE_LEFT", "MOVE_RIGHT",
)
"""The discrete action set the shipped episodes were collected under.

`envs.actions.build_action_set` is a no-op followed by one one-hot per button,
in the scenario's `available_buttons` order, and `my_way_home.cfg` lists
TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, MOVE_LEFT, MOVE_RIGHT. The episodes carry
only the integer, so the names live here -- they are what make the contrast
below a physical statement rather than a pair of indices.
"""

CONTRAST: tuple[int, int] = (3, 0)
"""The constant rung's decision statistic: held MOVE_FORWARD minus held NOOP.

PRE-REGISTERED, not chosen from the data. A max-minus-min over the held
actions is biased upward under the null -- the largest of six paired
differences is positive even when every one of them is noise -- while a
named pair is centred on zero under an action-blind prior (exactly zero,
bitwise, on the matched stream) and carries a sign the physics predicts:
MOVE_FORWARD is the one action that displaces the agent, NOOP the one that
does not, so an action-conditioned prior must run the imagined position
further from the real trajectory under the first than under the second. A
negative contrast is therefore not "a response" but the wrong-signed one, and
the verdict says which it saw. MOVE_FORWARD is also the action with the
longest natural runs in the data (13 steps), so of the six held sequences it
is the least off-distribution.
"""


def action_name(action: int) -> str:
    """`ACTION_NAMES[action]`, or the bare index for an action outside the set
    -- printed, never raised, because a report over a scenario with more
    buttons must still print its held actions."""
    return ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else f"action {action}"


def _rng_snapshot(device: torch.device) -> dict:
    """The generator state, for the device the sampling actually happens on.

    NOT `torch.get_rng_state()` alone. Measured directly: that call covers the
    CPU generator only, and restoring it leaves an MPS `Categorical.sample`
    drawing from wherever it had got to -- so on the device the shipped records
    were produced on, a CPU-only snapshot silently unmatches the arms. The CPU
    state is captured as well as the accelerator's, so that an op which quietly
    falls back to CPU is matched too.

    An accelerator with no verified snapshot here raises rather than falling
    back, because the fallback has no symptom: shapes, curves and window counts
    all still look right.
    """
    state = {"cpu": torch.get_rng_state()}
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    elif device.type != "cpu":
        raise NotImplementedError(
            f"no verified RNG snapshot for device type {device.type!r}. "
            "torch.get_rng_state() restores the CPU generator ONLY -- measured, "
            "it does not restore the MPS one -- so falling back to it would turn "
            "this comparison into sampling noise with nothing to show for it"
        )
    return state


def supports_matched_stream(device: torch.device) -> bool:
    """Can the arms be matched on this device at all.

    The same authority that does the refusing, asked without raising, so a
    caller can decline a run BEFORE loading a checkpoint and refitting a probe
    instead of dying twenty seconds in on a `NotImplementedError` -- which
    surfaces as exit status 1, the status reserved for an uncaught traceback.
    """
    try:
        _rng_snapshot(torch.device(device))
    except NotImplementedError:
        return False
    return True


def _no_window_error(need: int) -> ValueError:
    """The one refusal for "the window rule cut nothing", raised from two places.

    `_diagnose` reaches it after the traversal; the ladder's marginal pre-pass
    reaches it BEFORE the traversal, over the same window rule. Two wordings
    would make the same condition look like two different problems, and the
    pre-pass without one would refuse the contrast as "not in the support" --
    a true statement about an empty support that sends the reader to the
    wrong flag -- rather than name the split/horizon problem it actually is.
    """
    return ValueError(
        f"no diagnostic window reached {need + 1} frames; "
        "lower context/horizon or check the split"
    )


def _rng_restore(state: dict) -> None:
    torch.set_rng_state(state["cpu"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


@dataclass
class _Window:
    """Everything an imagination arm may read, and nothing it may write.

    `horizon_actions` is the REAL post-context action sequence; an arm that
    intervenes on it builds its own tensor rather than mutating this one, or the
    context filter and the floor -- the fixed brackets of every comparison here
    -- would move with it.
    """

    start: int
    context: int
    horizon: int
    embeddings: torch.Tensor        # (1, need, E), frame `start + 1 + k` at k
    actions: torch.Tensor           # (1, need), int64
    horizon_actions: torch.Tensor   # (1, horizon), == actions[:, context:]
    state: tuple                    # the context filter's last (h, z)


@dataclass
class _Pass:
    """One traversal's per-window errors: the canonical arm and every extra one."""

    reference: RolloutResult
    reference_windows: dict[str, np.ndarray]
    arms: dict          # arm name -> metric -> (n_windows, horizon)
    windows_total: int
    window_episode: np.ndarray
    """Per window, the index of the CONTRIBUTING episode it was cut from.

    Counted over the episodes that actually yielded a window, not over
    `val_paths`: an episode too short for the window rule is skipped entirely,
    and letting it consume a label would leave a gap that a consumer counting
    distinct labels reads as a missing cluster. The shipped split's 229 windows
    come from 24 episodes at 3 to 10 windows each, so this is what separates
    229 independent draws from 24 clusters."""


def _diagnose(
    model,
    val_paths,
    embedding_probe_weights: dict,
    *,
    arms: dict,
    context: int,
    horizon: int,
    seed: int,
    device,
    feature_backbone,
) -> _Pass:
    """`evaluate_rollout`, plus extra imagination arms on a matched stream.

    Per window, in this order: the context filter runs once and is shared; a
    device-aware RNG snapshot is taken; each arm in `arms` runs from that
    restored snapshot; and finally the CANONICAL pass -- `imagine` over the real
    horizon actions, then the floor's posterior -- is replayed from the same
    snapshot. Running the canonical pass LAST is what leaves the global stream
    exactly where `evaluate_rollout` leaves it, so window `w + 1` starts from the
    same state the record's own run started it from.

    Every arm returns a `(1, horizon, LATENT)` tensor, which is scored through
    the same embedding head and the same probe as every reference -- the
    single-pipeline rule the band depends on.

    Deliberately NOT separately decorated with `@torch.no_grad()`: both public
    entry points are, so a second decorator here is a guard no mutation can
    turn red, which this repo treats as a defect rather than as depth.
    """
    model.eval()
    device = device or next(model.parameters()).device
    need = context + horizon
    # One seed for the whole traversal, before the episode loop, exactly as
    # `evaluate_rollout` does it -- the per-window snapshots below rewind within
    # a window, never across one.
    torch.manual_seed(seed)

    reference_windows = {
        name: []
        for name in (
            "rssm_position", "persistence_position", "floor_position",
            "rssm_angle", "persistence_angle", "floor_angle",
        )
    }
    arm_windows = {name: {metric: [] for metric in METRICS} for name in arms}
    window_episode: list[int] = []

    for path in val_paths:
        episode = load_episode(path)
        # THE window rule, from the one place that owns it. If this diverged
        # from `evaluate_rollout`'s the curves below would not be comparable to
        # the shipped records at all.
        starts = window_starts(episode.length, context, horizon)
        if not starts:
            continue
        # Numbered AFTER the skip, so a too-short episode leaves no gap.
        episode_label = len(set(window_episode))
        source = source_for(model, path, episode, feature_backbone)
        for start in starts:
            window_episode.append(episode_label)
            window = slice(start, start + need + 1)
            # `window` spans need+1 frames; drop the FIRST one so embeddings[k]
            # is the frame actions[k] led to, per RSSM.observe's action-time
            # convention.
            embeddings = model.encoder(
                torch.as_tensor(source[window][1:]).to(device)
            ).unsqueeze(0)
            actions = torch.as_tensor(
                episode.actions[start : start + need].astype(np.int64)
            ).unsqueeze(0).to(device)

            observed = model.rssm.observe(
                embeddings[:, :context], actions[:, :context]
            )
            state = (observed["h"][:, -1], observed["z"][:, -1])
            handle = _Window(
                start=start,
                context=context,
                horizon=horizon,
                embeddings=embeddings,
                actions=actions,
                horizon_actions=actions[:, context:],
                state=state,
            )

            snapshot = _rng_snapshot(device)
            arm_latents = {}
            for name, arm in arms.items():
                _rng_restore(snapshot)
                arm_latents[name] = arm(handle)

            # The canonical pass, replayed from the same snapshot and run LAST.
            # This is `evaluate_rollout`'s own sequence, call for call.
            _rng_restore(snapshot)
            imagined = model.rssm.imagine(actions[:, context:], state)
            real = model.rssm.observe(
                embeddings[:, context:], actions[:, context:], state=state
            )
            floor_embeddings = model.heads(real["latent"])["embedding"][0].cpu().numpy()

            # Truth for imagined step k is the frame `actions[context + k]`
            # PRODUCED: `embeddings[k]` is frame `start + 1 + k`, so the last
            # observed frame is `start + context`, and `imagine` reaches
            # `start + context + 1 + k`. A slice off by d costs an exactly-right
            # model |d| steps of true displacement at every horizon step, and
            # breaks no shape -- see
            # test_a_perfect_predictor_scores_zero_only_on_the_frame_the_action_produced.
            truth = probe_targets(
                episode.privileged[start + context + 1 : start + need + 1],
                episode.privileged_keys,
            )
            last_context_embedding = (
                model.heads(observed["latent"][:, -1:])["embedding"][0, 0].cpu().numpy()
            )

            probe = lambda latent: apply_probe(  # noqa: E731
                embedding_probe_weights,
                model.heads(latent)["embedding"][0].cpu().numpy(),
            )
            model_pred = probe(imagined["latent"])
            floor_pred = apply_probe(embedding_probe_weights, floor_embeddings)
            pers_pred = apply_probe(
                embedding_probe_weights,
                np.repeat(last_context_embedding[None, :], horizon, axis=0),
            )
            reference_windows["rssm_position"].append(position_error(model_pred, truth))
            reference_windows["floor_position"].append(position_error(floor_pred, truth))
            reference_windows["persistence_position"].append(
                position_error(pers_pred, truth)
            )
            reference_windows["rssm_angle"].append(angle_error_degrees(model_pred, truth))
            reference_windows["floor_angle"].append(angle_error_degrees(floor_pred, truth))
            reference_windows["persistence_angle"].append(
                angle_error_degrees(pers_pred, truth)
            )

            for name, latent in arm_latents.items():
                predicted = probe(latent)
                arm_windows[name]["position"].append(position_error(predicted, truth))
                arm_windows[name]["angle"].append(angle_error_degrees(predicted, truth))

    if not reference_windows["rssm_position"]:
        raise _no_window_error(need)

    stacked = {name: np.stack(rows) for name, rows in reference_windows.items()}
    return _Pass(
        reference=RolloutResult(
            horizon=np.arange(1, horizon + 1),
            **{name: rows.mean(axis=0) for name, rows in stacked.items()},
        ),
        reference_windows=stacked,
        arms={
            name: {metric: np.stack(rows) for metric, rows in metrics.items()}
            for name, metrics in arm_windows.items()
        },
        windows_total=stacked["rssm_position"].shape[0],
        window_episode=np.array(window_episode, dtype=int),
    )


# ---------------------------------------------------------------------------
# Diagnostic 1: the action-intervention ladder.
# ---------------------------------------------------------------------------


def ladder_order(arms) -> tuple[str, ...]:
    """The requested rungs, deduplicated and put back into `LADDER`'s order.

    The caller's order is DISCARDED on purpose -- see `LADDER`. An unknown rung
    is refused rather than dropped, and so is an empty ladder: a run with no
    rung in it produces a full-looking report containing no intervention at
    all, which is the one outcome that cannot be distinguished from a null by
    reading the numbers.
    """
    requested = tuple(arms)
    unknown = [name for name in requested if name not in LADDER]
    if unknown:
        raise ValueError(
            f"no such intervention rung(s) {unknown!r}; the ladder is {LADDER!r}"
        )
    if not requested:
        raise ValueError(
            "an empty ladder has no intervention rung in it, so the run would "
            f"report the open loop against itself; ask for one or more of {LADDER!r}"
        )
    return tuple(name for name in LADDER if name in set(requested))


def horizon_action_marginal(
    val_paths, context: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """The empirical action distribution over the SCORED WINDOWS' horizons.

    Returns `(values, counts)`, values ascending. Counted over exactly the
    actions the intervention replaces -- `actions[start + context : start +
    context + horizon]` for every window `window_starts` yields -- and NOT over
    `episode.actions`. The two are genuinely different distributions: the
    context actions and any tail the stride does not reach are in one and not
    the other, so a marginal counted over whole episodes can put support on an
    action that never appears in a scored horizon, and can rank rarity
    differently. The resampled rung's claim is "each individual action stays
    plausible where it was substituted", which is a claim about this
    distribution.

    READ WITHOUT `load_episode`, deliberately. Only the `actions` array is
    needed and `.npz` members decompress lazily, so this pre-pass costs a few
    milliseconds per episode instead of decompressing every frame of every
    validation episode a second time, once per cell. The key is the one
    `save_episode` writes and the array is stored verbatim, so there is no
    conversion to drift away from.
    """
    collected: list[np.ndarray] = []
    for path in val_paths:
        with np.load(path) as data:
            actions = np.asarray(data["actions"])
        for start in window_starts(int(actions.shape[0]), context, horizon):
            collected.append(actions[start + context : start + context + horizon])
    if not collected:
        raise _no_window_error(context + horizon)
    values, counts = np.unique(
        np.concatenate(collected).astype(np.int64), return_counts=True
    )
    return values, counts


def _sequence_distances(intervened: np.ndarray, real: np.ndarray) -> tuple[int, int]:
    """`(steps changed, multiset distance)` between two action sequences.

    Two different distances, because the rungs differ in which one they are
    about. Steps changed is the sequence distance -- positions at which the two
    differ -- and is what says whether an intervention intervened at all.
    The multiset distance is the number of steps that must change to turn one
    multiset into the other, half the L1 between the bincounts, and is the
    number the RESAMPLED rung is for: measured on the shipped split its
    sequence distance reads 35 of 45 while the counts moved by 15, and a
    reader shown only the first cannot tell that the counts moved by less
    than half of what the row suggests. The shuffled rung's is 0 by
    construction, which is the monotonicity the table is meant to show.
    """
    width = int(max(intervened.max(), real.max())) + 1
    counts = np.bincount(intervened, minlength=width) - np.bincount(real, minlength=width)
    return int((intervened != real).sum()), int(np.abs(counts).sum() // 2)


@dataclass(kw_only=True)
class PairedDelta:
    """One paired comparison's per-window deltas, and the counts that say
    whether it compared anything.

    The counts are not bookkeeping. A comparison can be a no-op -- the
    permutation on a window whose actions are already uniform, a resample that
    reproduces the real sequence, a held action on a window that already held
    it -- and a no-difference result computed over no-op windows says nothing
    about the dynamics while looking exactly like the null the study is trying
    to establish. So every comparison carries its own `window_steps_changed`
    and every consumer reports it beside the delta.

    `window_position_delta` is `first - second` over whichever two sequences
    the comparison pairs: an intervened arm against the real one for
    `ArmResult`, two held actions against each other for `ContrastResult`.
    ONE implementation of the delta for every view, including the standalone
    shuffle's: a second copy of the exclusion below was untested for as long as
    it existed, and the CLI read the untested one.
    """

    name: str
    window_position_delta: np.ndarray   # (n_windows, horizon)
    window_angle_delta: np.ndarray
    window_steps_changed: np.ndarray
    """Per window: at how many of the `horizon` steps the two paired sequences
    differ.

    A count rather than a flag, because the rungs differ enormously in how far
    they travel and "changed" alone cannot show it. On a window already
    dominated by a held action, a bare flag reads the same as on a window
    where all 45 steps moved -- which is precisely the trap that makes a
    most-frequent-action constant arm a no-op wearing the ladder's top rung's
    name.
    """

    window_multiset_distance: np.ndarray
    """Per window: the multiset distance between the two paired sequences --
    see `_sequence_distances`."""

    horizon: int
    window_episode: np.ndarray | None = None
    """Which episode each window was cut from -- see `_Pass.window_episode`.

    `None` means "clustering unknown", which is what a hand-built result
    carries; a consumer that would divide by `sqrt(n_windows)` must say which
    of the two it did, because the shipped windows are not independent draws.
    """

    @property
    def changed(self) -> np.ndarray:
        """Per window: did the comparison pair two DIFFERENT sequences.

        Derived from the step count, so "the sequence moved" and "how far it
        moved" cannot disagree. Compared as SEQUENCES and never as multisets --
        a resample that happens to be a rearrangement of the real window has
        randomised nothing, and the multiset is exactly what the rung above the
        permutation is supposed to be changing, so the obvious multiset
        comparison would report that no-op as an intervention.
        """
        return self.window_steps_changed > 0

    @property
    def windows_total(self) -> int:
        return int(self.window_steps_changed.size)

    @property
    def windows_changed(self) -> int:
        return int(self.changed.sum())

    @property
    def mean_steps_changed(self) -> float:
        """Mean per-window distance between the paired sequences, in steps.

        All-NaN-free by construction: this is a property of the intervention,
        not of the model, so it is defined even when the delta is not -- and
        on a rung that turned out to be a no-op it is exactly 0.0, which is
        the number that says so.
        """
        return float(self.window_steps_changed.mean())

    @property
    def min_steps_changed(self) -> int:
        """The LEAST any window moved. The mean alone hides a window that a
        held action left one step from its real sequence -- "changed", with
        distance 1 -- and nothing else in the record would let a reader find
        it."""
        return int(self.window_steps_changed.min())

    @property
    def mean_multiset_distance(self) -> float:
        return float(self.window_multiset_distance.mean())

    def is_interpretable(self) -> bool:
        """False when no window's sequences actually differed.

        A no-difference result computed over windows where the comparison was
        a no-op says nothing about the dynamics. Measured on the shipped split
        all 229 windows change under every rung, so this is inert against the
        real checkpoints -- and that is precisely why it is a reported output
        rather than a debug aid: nothing in a real run would ever reveal its
        absence.
        """
        return self.windows_changed > 0

    def position_delta(self) -> np.ndarray:
        return self._delta(self.window_position_delta)

    def angle_delta(self) -> np.ndarray:
        return self._delta(self.window_angle_delta)

    def _delta(self, per_window: np.ndarray) -> np.ndarray:
        """Mean `first - second` over the windows the comparison CHANGED.

        Unchanged windows contribute an exact zero, which dilutes the measured
        effect toward "no difference" -- the very conclusion this diagnostic is
        used to draw -- so they are excluded from the mean rather than averaged
        in. With none of them (the shipped case) this is exactly the mean.

        All-NaN when nothing changed: undefined, not zero, following
        `gap_closed`'s policy. A zero here would read as "the model ignored the
        actions" when the truth is "the actions were never intervened on".
        """
        if not self.is_interpretable():
            return np.full(per_window.shape[1], np.nan)
        return per_window[self.changed].mean(axis=0)


@dataclass(kw_only=True)
class ArmResult(PairedDelta):
    """One intervened sequence against the real one: its curves and its deltas."""

    position: np.ndarray
    angle: np.ndarray

    def curves(self) -> dict[str, np.ndarray]:
        """The position curves a record writes for this rung, keyed by the
        suffix that goes after the rung's name -- one, unsuffixed, here."""
        return {"": self.position}


@dataclass(kw_only=True)
class ContrastResult(PairedDelta):
    """The constant rung: every action held, decided on one named contrast.

    `window_position_delta` is `held[contrast[0]] - held[contrast[1]]` per
    window -- the error under the first held action minus the error under the
    second, the real arm cancelling -- and `window_steps_changed` counts the
    steps at which the two held TENSORS differ, measured on what was imagined:
    every step when the two actions differ, and zero if a defect ever handed
    the same tensor to both, which is what would make the contrast a no-op and
    is the only way it can be one. Each held action's own arm against the real
    sequence is kept in `held`, with its own distances, because a held action
    is a no-op exactly where a window already held it and the contrast's count
    cannot show that.
    """

    held: dict[int, ArmResult]
    contrast: tuple[int, int]

    def curves(self) -> dict[str, np.ndarray]:
        """One curve per held action, suffixed `_held<a>`: a contrast has no
        error curve of its own -- the difference of two error curves is not an
        error -- and writing one under the rung's bare name would put a signed
        difference beside the other rungs' absolute errors."""
        return {f"_held{action}": arm.position for action, arm in self.held.items()}


@dataclass
class LadderResult:
    """Every rung against ONE shared open-loop reference."""

    real: RolloutResult
    """Bitwise `evaluate_rollout`'s own result -- every rung's baseline IS the
    shipped protocol, including the floor and persistence, which no
    intervention touches."""

    arms: dict[str, PairedDelta]
    order: tuple[str, ...]
    """The rungs actually run, in `LADDER`'s increasing-perturbation order."""

    intervention_seed: int
    held_actions: tuple[int, ...] | None
    """Every action the constant rung held -- the scored support, ascending --
    or None when that rung did not run."""

    contrast: tuple[int, int] | None
    """The two held actions the constant rung's statistic is the difference
    of, or None when that rung did not run. Recorded because it is a CHOICE
    the sign of the top rung depends on."""

    action_marginal_values: np.ndarray | None
    action_marginal_counts: np.ndarray | None
    """The distribution the resampled rung drew from and the constant rung's
    support, recorded so a later reader can tell a genuine null from one
    measured against a marginal that had collapsed onto a point mass."""

    windows_total: int
    window_episode: np.ndarray | None = None


@torch.no_grad()
def action_intervention_ladder(
    model,
    val_paths,
    embedding_probe_weights: dict,
    arms: tuple[str, ...] = LADDER,
    context: int = 5,
    horizon: int = 45,
    seed: int = 0,
    device: torch.device | None = None,
    feature_backbone: str | None = None,
    intervention_seed: int = 0,
    contrast: tuple[int, int] = CONTRAST,
) -> LadderResult:
    """Imagine the horizon once per intervention arm, and once on the real actions.

    ONLY `imagine` SEES AN INTERVENED ACTION. The context filter and the floor
    are the fixed brackets of every comparison and consume the real order; the
    floor in particular consumes the horizon actions, so intervening before it
    would corrupt the diagnostic's own lower bound. Each arm therefore BUILDS
    a tensor and never writes through `_Window.horizon_actions`.

    EVERY RUNG DRAWS FROM ITS OWN NUMPY GENERATOR, NEVER FROM TORCH.
    `torch.randperm` would draw from the very generator the arms are matched
    on, displacing the intervened arm's sampling stream and turning the
    measured effect into noise with no symptom. The same hazard exists BETWEEN
    rungs: a resampled rung drawing from the permutation's generator would
    advance it once per window, so every window after the first would be
    permuted differently and the shuffled rung's shipped numbers would move
    while every shape, window count and self-check stayed right. So the
    shuffled rung keeps `default_rng(intervention_seed)` exactly -- it is what
    produced the nine shipped records -- and each other rung gets an
    independent stream derived from the same seed by a distinct SeedSequence
    entropy. A rung's draws do not depend on which OTHER rungs were selected.

    THE CONSTANT RUNG HOLDS EVERY ACTION IN THE SCORED SUPPORT, and decides on
    `contrast` -- see `CONTRAST` for why a named pair and not one held action
    or the spread over all of them. The support is the scored windows'
    marginal, so every held action is one the model saw in a horizon; a
    contrast action outside it, or a contrast of an action with itself, is
    refused by name BEFORE the traversal rather than run to a number that
    means nothing. The support is a property of the split, not of a window,
    so it is the same set of interventions everywhere.

    THE MARGINAL IS THE VALIDATION SPLIT'S, and that is a deliberate,
    documented choice rather than an oversight. "Plausible to the model" is a
    statement about the training distribution, but the two marginals agree to
    within a point on every action and what the resampled rung substitutes
    INTO is a scored window, so the plausibility that matters is at the point
    of substitution. The one decision the marginal used to make on its own --
    which action to hold -- was split-dependent by a 31-count margin between
    a three-way near-tie, and the contrast makes it moot: every action is
    held.
    """
    order = ladder_order(arms)
    values = counts = None
    held_actions = None
    held = None
    if {"resampled", "constant"} & set(order):
        values, counts = horizon_action_marginal(val_paths, context, horizon)
    if "constant" in order:
        held_actions = tuple(int(v) for v in values)
        held = tuple(int(a) for a in contrast)
        if len(held) != 2 or held[0] == held[1]:
            raise ValueError(
                f"contrast={tuple(contrast)!r} must name two distinct held actions; "
                "a contrast of an action with itself compares a sequence to itself"
            )
        missing = [a for a in held if a not in held_actions]
        if missing:
            raise ValueError(
                f"contrast action(s) {missing!r} are not in the scored windows' "
                f"support {held_actions!r}; a held action the model never saw in a "
                "horizon has no interpretable null, so the contrast is refused "
                "rather than run"
            )

    # Every arm's intervened sequence, per window, in traversal order, and the
    # real horizon sequence once per window. Distances are computed from
    # these AFTER the traversal, off the tensor that was actually imagined, so
    # "what was counted" and "what was predicted from" cannot come apart --
    # an arm that built a sequence and then imagined a different one would be
    # counted on the one it did not use.
    arm_keys: list = [name for name in order if name != "constant"]
    if held_actions is not None:
        arm_keys.extend(("constant", action) for action in held_actions)
    sequences: dict = {key: [] for key in arm_keys}
    real_sequences: list[np.ndarray] = []

    def record(key, intervened: torch.Tensor, handle: _Window) -> torch.Tensor:
        if key == arm_keys[0]:
            real_sequences.append(handle.horizon_actions[0].cpu().numpy().astype(np.int64))
        sequences[key].append(intervened[0].cpu().numpy().astype(np.int64))
        return model.rssm.imagine(intervened, handle.state)["latent"]

    # `default_rng(intervention_seed)` for the shuffled rung, UNCHANGED and not
    # derived: this exact call produced the permutations behind the nine
    # shipped diagnostic records, and re-deriving it through a SeedSequence
    # child would silently re-permute all nine.
    permutation_rng = default_rng(intervention_seed)
    # A distinct entropy, not a spawn of the above -- spawning would consume
    # from it and is a function of how many children are asked for, so the
    # resampled draws would depend on which rungs were selected.
    resample_rng = default_rng([intervention_seed, 1])

    def shuffled(handle: _Window) -> torch.Tensor:
        index = torch.as_tensor(
            np.asarray(permutation_rng.permutation(handle.horizon)),
            dtype=torch.long,
            device=handle.horizon_actions.device,
        )
        return record("shuffled", handle.horizon_actions[:, index], handle)

    def resampled(handle: _Window) -> torch.Tensor:
        draw = resample_rng.choice(values, size=handle.horizon, p=counts / counts.sum())
        drawn = torch.as_tensor(
            np.asarray(draw, dtype=np.int64),
            dtype=handle.horizon_actions.dtype,
            device=handle.horizon_actions.device,
        ).unsqueeze(0)
        return record("resampled", drawn, handle)

    def constant(action: int):
        def arm(handle: _Window) -> torch.Tensor:
            # `full_like`, so the arm owns its tensor: an arm that wrote into
            # `handle.horizon_actions` would move the floor and every later
            # arm with it, and nothing about the shapes would say so.
            return record(
                ("constant", action), torch.full_like(handle.horizon_actions, action), handle
            )

        return arm

    builders = {
        key: (
            constant(key[1]) if isinstance(key, tuple)
            else {"shuffled": shuffled, "resampled": resampled}[key]
        )
        for key in arm_keys
    }
    result = _diagnose(
        model, val_paths, embedding_probe_weights,
        arms=builders, context=context, horizon=horizon, seed=seed, device=device,
        feature_backbone=feature_backbone,
    )
    assert len(real_sequences) == result.windows_total

    def arm_result(name: str, key) -> ArmResult:
        distances = np.array(
            [_sequence_distances(mine, real) for mine, real in zip(sequences[key], real_sequences)],
            dtype=int,
        ).reshape(-1, 2)
        return ArmResult(
            name=name,
            position=result.arms[key]["position"].mean(axis=0),
            angle=result.arms[key]["angle"].mean(axis=0),
            window_position_delta=(
                result.arms[key]["position"] - result.reference_windows["rssm_position"]
            ),
            window_angle_delta=(
                result.arms[key]["angle"] - result.reference_windows["rssm_angle"]
            ),
            window_steps_changed=distances[:, 0],
            window_multiset_distance=distances[:, 1],
            horizon=horizon,
            window_episode=result.window_episode,
        )

    rungs: dict[str, PairedDelta] = {}
    for name in order:
        if name != "constant":
            rungs[name] = arm_result(name, name)
            continue
        held_arms = {action: arm_result(name, ("constant", action)) for action in held_actions}
        first, second = held
        distances = np.array(
            [
                _sequence_distances(a, b)
                for a, b in zip(sequences[("constant", first)], sequences[("constant", second)])
            ],
            dtype=int,
        ).reshape(-1, 2)
        rungs[name] = ContrastResult(
            name=name,
            held=held_arms,
            contrast=held,
            window_position_delta=(
                held_arms[first].window_position_delta - held_arms[second].window_position_delta
            ),
            window_angle_delta=(
                held_arms[first].window_angle_delta - held_arms[second].window_angle_delta
            ),
            window_steps_changed=distances[:, 0],
            window_multiset_distance=distances[:, 1],
            horizon=horizon,
            window_episode=result.window_episode,
        )

    return LadderResult(
        real=result.reference,
        arms=rungs,
        order=order,
        intervention_seed=intervention_seed,
        held_actions=held_actions,
        contrast=held,
        action_marginal_values=values,
        action_marginal_counts=counts,
        windows_total=result.windows_total,
        window_episode=result.window_episode,
    )


@dataclass(kw_only=True)
class ShuffleResult(PairedDelta):
    """The open-loop rollout beside itself with the horizon actions permuted.

    The ladder's shuffled rung under the field names the nine shipped records
    were written from; the statistics are `PairedDelta`'s, not a second copy.
    """

    real: RolloutResult
    """Bitwise `evaluate_rollout`'s own result -- the shuffle's baseline IS the
    shipped protocol, including the floor and persistence, which the permutation
    never touches."""

    shuffled_position: np.ndarray
    shuffled_angle: np.ndarray
    permutation_seed: int


@torch.no_grad()
def action_shuffled_rollout(
    model,
    val_paths,
    embedding_probe_weights: dict,
    context: int = 5,
    horizon: int = 45,
    seed: int = 0,
    device: torch.device | None = None,
    feature_backbone: str | None = None,
    permutation_seed: int = 0,
) -> ShuffleResult:
    """The ladder's SHUFFLED rung alone: the real actions against a permutation.

    PERMUTED, NOT RESAMPLED. A permutation preserves the window's action
    multiset exactly, so a difference between the two arms cannot be attributed
    to a distribution shift -- which resampling from the action space does
    introduce. That is the whole reason the ladder has three rungs rather than
    one: this rung buys an unconfounded comparison at the price of being blind
    to dynamics that read the COUNTS rather than the order, and `resampled`
    buys the converse.

    The permutation preserves the multiset, and that is also its blind spot:
    dynamics that depend only on the SUM or the multiset of the horizon actions
    are indistinguishable from action-blind at the FINAL horizon step, which is
    the study's headline number. Read the delta CURVE, not its endpoint -- and
    read the rungs above this one.

    ONE IMPLEMENTATION, TWO VIEWS. This is `action_intervention_ladder` with a
    single rung, repacked into the shape the nine shipped records were written
    from. A second permutation path beside the ladder's could drift from it
    while both kept passing, and the nine records would then be comparable to
    neither. The contract that the ladder's stream IS the shipped one is pinned
    on the permutation index itself, in the tests, because an equality between
    these two entry points can no longer fail.
    """
    ladder = action_intervention_ladder(
        model, val_paths, embedding_probe_weights,
        arms=("shuffled",), context=context, horizon=horizon, seed=seed,
        device=device, feature_backbone=feature_backbone,
        intervention_seed=permutation_seed,
    )
    arm = ladder.arms["shuffled"]
    return ShuffleResult(
        name="shuffled",
        real=ladder.real,
        shuffled_position=arm.position,
        shuffled_angle=arm.angle,
        window_position_delta=arm.window_position_delta,
        window_angle_delta=arm.window_angle_delta,
        window_steps_changed=arm.window_steps_changed,
        window_multiset_distance=arm.window_multiset_distance,
        horizon=horizon,
        permutation_seed=permutation_seed,
        window_episode=ladder.window_episode,
    )


# ---------------------------------------------------------------------------
# Diagnostic 2: the k-step re-grounding sweep.
# ---------------------------------------------------------------------------


@dataclass
class RegroundingSweep:
    """One error curve per re-grounding period, against one shared bracket."""

    ks: tuple[int, ...]
    horizon: int
    reference: RolloutResult
    """The pure open loop, bitwise `evaluate_rollout`'s -- and with it the ONE
    floor and persistence every k is read against. Every k shares them because
    they are computed once per window, so "k approaches the floor" is a
    statement about the same measurement rather than about a re-drawn one."""

    position: dict[int, np.ndarray]
    angle: dict[int, np.ndarray]
    window_position: dict[int, np.ndarray]
    window_floor_position: np.ndarray
    """The floor's OWN per-window rows, retained so that "k sits above the
    floor" has a paired ruler. Keeping only the floor's mean leaves the
    k-vs-floor margin with no correct standard error available at all."""
    windows_total: int

    def curve(self, k: int, metric: str = "position") -> np.ndarray:
        """This k's mean error curve.

        The metric is CHECKED rather than treated as "position or else". An
        `else` here returns the ANGLE curve for any string that is not exactly
        "position" -- a typo included -- and every number this class reports is
        built from this call, so degrees would be printed under a map-units
        heading with nothing to show for it.
        """
        if metric not in METRICS:
            raise KeyError(f"no {metric!r} curve; the metrics are {METRICS}")
        return (self.position if metric == "position" else self.angle)[k]

    def curve_standard_error(self, k: int, metric: str = "position") -> np.ndarray:
        """The spread of THIS k's own curve across windows. NOT a k-to-k ruler.

        Named for what it is. It is the between-window spread of a single
        curve, and the between-window variation is dominated by window
        difficulty -- which every k shares -- so it is far larger than the
        spread of a DIFFERENCE between two ks and would declare real
        separations unresolvable. Measured on the shipped cells it overstates
        the bar for an adjacent-k difference by 1.7x to 3.9x. Use
        `paired_standard_error` for a comparison between two curves and
        `floor_margin_standard_error` for the distance to the floor; this one
        describes one curve on its own.
        """
        if metric != "position":
            raise KeyError(f"no per-window {metric!r} curves are retained")
        return self._spread(self.window_position[k])

    def paired_standard_error(self, k: int, other: int) -> np.ndarray:
        """The ruler for `curve(k) - curve(other)`.

        Every k is computed on the SAME windows from the SAME per-window RNG
        snapshot, so the two curves are strongly correlated across windows and
        the comparison is paired. The spread that a difference has to clear is
        therefore the spread of the per-window difference, not of either curve.
        """
        return self._spread(self.window_position[k] - self.window_position[other])

    def floor_margin_standard_error(self, k: int) -> np.ndarray:
        """The ruler for `curve(k) - floor`, paired the same way.

        Self-check 2 asks whether k sits above the floor, and the floor is
        measured on the same windows in the same traversal, so its per-window
        rows pair with this k's.
        """
        return self._spread(self.window_position[k] - self.window_floor_position)

    @staticmethod
    def _spread(rows: np.ndarray) -> np.ndarray:
        """Standard error of the column means. Zeros on a single window.

        Zeros rather than NaN or inf: one window has no spread to estimate, and
        both of the alternatives propagate into the printed table -- a NaN as a
        hole, an inf as "nothing is resolvable" -- where a reader cannot tell
        them from a computed result.
        """
        if rows.shape[0] < 2:
            return np.zeros(rows.shape[1])
        return rows.std(axis=0, ddof=1) / np.sqrt(rows.shape[0])

    def open_loop_divergence(self, reference: RolloutResult) -> float:
        """SELF-CHECK 1: max |k=horizon curve - open-loop curve|. MUST be 0.0.

        The k == horizon pass makes exactly the same `imagine` call, on the same
        tensor, from the same point in the sampling stream, so equality is
        BITWISE rather than approximate. Anything else means the re-grounding
        path has diverged from `evaluate_rollout` and every other number this
        sweep reports is suspect.
        """
        return float(
            max(
                np.abs(self.curve(self.horizon, metric) - getattr(reference, f"rssm_{metric}")).max()
                for metric in METRICS
            )
        )

    def is_bitwise_the_floor(self, k: int, metric: str = "position") -> bool:
        """SELF-CHECK 2's alarm: did this k's curve collapse onto the floor.

        Re-grounding with the posterior OF the step being scored does not merely
        shrink the gap to the floor -- it makes the two the same tensor. On the
        shipped checkpoints the honest prior/posterior separation may be smaller
        than the sampling noise, so a threshold test could be satisfied or
        defeated by chance; bitwise identity cannot.
        """
        return bool(
            np.array_equal(self.curve(k, metric), getattr(self.reference, f"floor_{metric}"))
        )


@torch.no_grad()
def regrounding_sweep(
    model,
    val_paths,
    embedding_probe_weights: dict,
    ks: tuple[int, ...] = REGROUNDING_KS,
    context: int = 5,
    horizon: int = 45,
    seed: int = 0,
    device: torch.device | None = None,
    feature_backbone: str | None = None,
) -> RegroundingSweep:
    """Imagine k steps, re-observe the real frames, repeat across the horizon.

    THE SEGMENT AND ITS GROUNDING, stated so the off-by-one is hard to write.
    Segment `s` covers horizon steps `[s*k, min((s+1)*k, H))` and is imagined
    from `state`; before segment `s > 0`, `state` is replaced by the POSTERIOR
    over the previous segment's real frames, warm-started from the state that
    segment started from. The last embedding row that grounding consumes is
    `embeddings[context + s*k - 1]`, i.e. frame `start + context + s*k`, exactly
    one before the first frame segment `s` is scored on. Every horizon step is
    imagined exactly once, and every scored latent is a PRIOR step.

    THE GROUNDING REPLAYS THE WHOLE SEGMENT, not just its last frame.
    `RSSM.observe` advances `h` with every action before forming each posterior
    (see its docstring), so a single-frame `observe` at the boundary would hand
    the next segment an `h` that had seen one action instead of k. It costs
    nothing in shapes and is invisible against a filter that ignores `state`.
    Replaying the segment also means the chain of grounding states is exactly
    the floor's own filtering trajectory, chunked -- which is what makes
    "re-observe the real frame" well defined here.
    """
    for k in ks:
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= horizon:
            raise ValueError(
                f"regrounding period k={k!r} must be a whole number of steps in "
                f"1..{horizon}; k=0 makes the segment loop non-terminating deep "
                "inside the traversal, and a k above the horizon is the open loop "
                "under a name that claims otherwise"
            )
    if horizon not in ks:
        raise ValueError(
            f"ks={tuple(ks)!r} omits the horizon {horizon}, and the k == horizon "
            "pass IS the self-check that this sweep reproduces `evaluate_rollout` "
            "bitwise; a sweep without it reports curves nothing has checked"
        )

    def segmented(k: int):
        def arm(handle: _Window) -> torch.Tensor:
            state = handle.state
            latents = []
            for offset in range(0, handle.horizon, k):
                stop = min(offset + k, handle.horizon)
                segment = model.rssm.imagine(
                    handle.horizon_actions[:, offset:stop], state
                )
                latents.append(segment["latent"])
                # Only BETWEEN segments. Re-grounding after the last one would
                # ground a state nothing goes on to use.
                if stop < handle.horizon:
                    grounded = model.rssm.observe(
                        handle.embeddings[
                            :, handle.context + offset : handle.context + stop
                        ],
                        handle.actions[
                            :, handle.context + offset : handle.context + stop
                        ],
                        state=state,
                    )
                    state = (grounded["h"][:, -1], grounded["z"][:, -1])
            return torch.cat(latents, dim=1)

        return arm

    result = _diagnose(
        model, val_paths, embedding_probe_weights,
        arms={k: segmented(k) for k in ks}, context=context, horizon=horizon,
        seed=seed, device=device, feature_backbone=feature_backbone,
    )
    return RegroundingSweep(
        ks=tuple(ks),
        horizon=horizon,
        reference=result.reference,
        position={k: result.arms[k]["position"].mean(axis=0) for k in ks},
        angle={k: result.arms[k]["angle"].mean(axis=0) for k in ks},
        window_position={k: result.arms[k]["position"] for k in ks},
        window_floor_position=result.reference_windows["floor_position"],
        windows_total=result.windows_total,
    )
