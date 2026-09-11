"""Two diagnostics on the SHIPPED M3b checkpoints -- milestone M3b, no retraining.

The nine-cell study reads NOT PASSED, and no arm beats persistence at any
horizon past h=16. Because EVERY arm loses, the evidence points at what the arms
SHARE -- the RSSM, the free-bits schedule, the 20k budget -- rather than at the
representation contrast the study was built to test. These two diagnostics turn
that inference into a mechanism, and both run on the frozen checkpoints:

  ACTION-SHUFFLED IMAGINATION. Re-run the open-loop imagination with the horizon
  actions PERMUTED within each window, everything else identical. If the error
  is unchanged, the dynamics prior is ignoring the action -- and an actor
  trained inside this world model cannot learn anything, which blocks M4
  outright.

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
        raise ValueError(
            f"no diagnostic window reached {need + 1} frames; "
            "lower context/horizon or check the split"
        )

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
# Diagnostic 1: action-shuffled imagination.
# ---------------------------------------------------------------------------


@dataclass
class ShuffleResult:
    """The open-loop rollout beside itself with the horizon actions permuted."""

    real: RolloutResult
    """Bitwise `evaluate_rollout`'s own result -- the shuffle's baseline IS the
    shipped protocol, including the floor and persistence, which the permutation
    never touches."""

    shuffled_position: np.ndarray
    shuffled_angle: np.ndarray
    changed: np.ndarray
    """Per window: did the permutation actually change the action SEQUENCE."""
    window_position_delta: np.ndarray   # (n_windows, horizon)
    window_angle_delta: np.ndarray
    permutation_seed: int
    window_episode: np.ndarray | None = None
    """Which episode each window was cut from -- see `_Pass.window_episode`.

    `None` means "clustering unknown", which is what a hand-built result
    carries; a consumer that would divide by `sqrt(n_windows)` must say which
    of the two it did, because the shipped windows are not independent draws.
    """

    @property
    def windows_total(self) -> int:
        return int(self.changed.size)

    @property
    def windows_changed(self) -> int:
        return int(self.changed.sum())

    def is_interpretable(self) -> bool:
        """False when no window's actions actually moved.

        A no-difference result computed over windows where the permutation was a
        no-op says nothing about the dynamics. Measured on the shipped split all
        229 windows change, so this is inert against the real checkpoints -- and
        that is precisely why it is a reported output rather than a debug aid:
        nothing in a real run would ever reveal its absence.
        """
        return self.windows_changed > 0

    def position_delta(self) -> np.ndarray:
        return self._delta(self.window_position_delta)

    def angle_delta(self) -> np.ndarray:
        return self._delta(self.window_angle_delta)

    def _delta(self, per_window: np.ndarray) -> np.ndarray:
        """Mean `shuffled - real` over the windows the permutation CHANGED.

        Unchanged windows contribute an exact zero, which dilutes the measured
        effect toward "no difference" -- the very conclusion this diagnostic is
        used to draw -- so they are excluded from the mean rather than averaged
        in. With none of them (the shipped case) this is exactly
        `shuffled - real`.

        All-NaN when nothing changed: undefined, not zero, following
        `gap_closed`'s policy. A zero here would read as "the model ignored the
        actions" when the truth is "the actions were never rearranged".
        """
        if not self.is_interpretable():
            return np.full(per_window.shape[1], np.nan)
        return per_window[self.changed].mean(axis=0)


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
    """Imagine the horizon twice: on the real actions, and on a permutation.

    PERMUTED, NOT RESAMPLED. A permutation preserves the window's action
    multiset exactly, so a difference between the two arms cannot be attributed
    to a distribution shift -- which resampling from the action space would
    introduce, and which would be indistinguishable from the effect being
    measured.

    ONLY `imagine` SEES THE PERMUTED ACTIONS. The context filter and the floor
    are the fixed brackets of the comparison and consume the real order; the
    floor in particular consumes the horizon actions, so permuting them before
    it would corrupt the diagnostic's own lower bound.

    THE PERMUTATION COMES FROM NUMPY, NEVER FROM TORCH. `torch.randperm` draws
    from the very generator the arms are being matched on, so it would displace
    the permuted arm's sampling stream and turn the measured effect into noise
    -- with no symptom.

    The permutation preserves the multiset, and that is also its blind spot:
    dynamics that depend only on the SUM or the multiset of the horizon actions
    are indistinguishable from action-blind at the FINAL horizon step, which is
    the study's headline number. Read the delta CURVE, not its endpoint.
    """
    rng = default_rng(permutation_seed)
    changed: list[bool] = []

    def shuffled(handle: _Window) -> torch.Tensor:
        order = rng.permutation(handle.horizon)
        index = torch.as_tensor(
            np.asarray(order), dtype=torch.long, device=handle.horizon_actions.device
        )
        permuted = handle.horizon_actions[:, index]
        # Compared as ACTION SEQUENCES, not as permutation indices: an identity
        # draw and a window whose actions are already uniform are both genuine
        # no-ops, and neither is evidence that the model ignored the action.
        changed.append(not bool(torch.equal(permuted, handle.horizon_actions)))
        return model.rssm.imagine(permuted, handle.state)["latent"]

    result = _diagnose(
        model, val_paths, embedding_probe_weights,
        arms={"shuffled": shuffled}, context=context, horizon=horizon,
        seed=seed, device=device, feature_backbone=feature_backbone,
    )
    arm = result.arms["shuffled"]
    return ShuffleResult(
        real=result.reference,
        shuffled_position=arm["position"].mean(axis=0),
        shuffled_angle=arm["angle"].mean(axis=0),
        changed=np.array(changed, dtype=bool),
        window_position_delta=arm["position"] - result.reference_windows["rssm_position"],
        window_angle_delta=arm["angle"] - result.reference_windows["rssm_angle"],
        permutation_seed=permutation_seed,
        window_episode=result.window_episode,
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
