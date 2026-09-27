"""M3i: does the posterior latent encode step-to-step MOTION, or only absolute position?

M3g localised the M3 failure to the prior and measured that the frame injects
0.290-0.489 nats beyond what the teacher-forced prior already predicted, over
all 32 groups; it also measured that copying the previous latent predicts the
next one BETTER than the model's own prior does. M3h then ruled out the prior's
sampling temperature. Between them they leave M3h section 8's three remaining
levers -- all prior-side -- pulling on a stage with about four tenths of a nat
of headroom.

So this tool asks a different question. The latent demonstrably carries
absolute position (`latent_selection_r2` 0.18-0.34 across the M3c records), and
a latent that encodes "which corridor am I in" can score well on a position
probe while carrying nothing about DISPLACEMENT. The M3 gate scores the
imagined trajectory against PERSISTENCE -- staying put -- so a latent without
displacement cannot beat it whatever the prior does.

  measure   per shipped cell: fit a ridge probe from the posterior latent AT
            THE ROLLOUT'S OWN t0 to `p(t+k) - p(t)` measured from that same
            frame, on TRAINING episodes; score it per VALIDATION window
            against the persistence baseline (`contrast_series`); and score
            the same probe again with the latent->displacement pairing
            PERMUTED. One record per cell.
  read      pool those records per arm -- treatment and control by the SAME
            route -- decide Reading D at `DECISION_H`, print the self-check,
            descriptive and per-k tables beside it, and write motion.txt.

Evaluation only: no training, no checkpoint written or altered.

LOADING IS `trust_horizon.py`'S: `Cell`, `load_cell`, `self_check` and
`prepare_cell` are imported by path, so a cell is refused here for the reasons
and in the words the other tools refuse it. `_sibling` executes the file afresh
per importer, so each importer holds its OWN class object -- never `except`
another module's copy of an exception (M3h shipped that bug once).

THE CHECKS, BY PHASE, each with its own status:

  measure:  EXIT_NO_CHECKPOINTS (11)     a requested cell lacks its checkpoint,
                                          record or diagnostic; judged for every
                                          cell before any pass runs.
            EXIT_SPLIT_MISMATCH (12)      the split by name is not the record's.
            EXIT_RECORD_MISMATCH (14)     --context/--horizon disagree with the
                                          protocol, or evaluate_rollout no longer
                                          reproduces the record's curve.
            EXIT_SELF_CHECK_FAILED (30)   the pass does not reproduce the cell's
                                          diagnostic within the bound.
  read:     EXIT_NO_CHECKPOINTS (11)      a requested cell has no motion record,
                                          named before any number is pooled.
            EXIT_CONTROL_LEAKED (38)      NEW. the permuted control cleared the
                                          bar, so no reading is taken -- and NO
                                          motion.txt is written, because a
                                          suppressed reading must not leave an
                                          artefact a later reader mistakes for
                                          a result.

0 / 11 / 12 / 14 / 30 carry `trust_horizon.py`'s meanings on purpose; 38 is in
no other tool's range (run_study 1/3-6/23, report_study 7-10, spike 10,
diagnose 11-17, pool 18-22, trust 30, split_gap 31, ladder 32-33, stages 34,
sharper_latent 35-37, argparse 2, a traceback 1).

FIVE CORRECTIONS TO THE M3I TASK-6 BRIEF, each found by running its own tests
or by measuring the real protocol. Three of them would have moved a number the
milestone is decided on:

  * THE BRIEF'S `permute_pairing` DOES NOT ALWAYS DERANGE. Its repair swapped
    each fixed point with its right-hand neighbour and wrapped the last index
    onto 0, which can put row 0 back on itself: measured over n in 2..39 and
    seeds 0..1999 it still left a fixed point in 1370 of 76,000 cases. The
    derangement is drawn by rejection instead -- see `permute_pairing`.
  * THE BRIEF'S OWN MUTATION TEST CANNOT CATCH THAT.
    `default_rng(0).permutation(200)` happens to have no fixed point, so at
    the brief's `(n=200, seed=CONTROL_SEED)` a control with no derangement at
    all still passes. The test file keeps that case and adds two that fail.
  * THE DISPLACEMENT PROBE MUST SELECT ITS RIDGE. `fit_probe` with no
    selection data takes its documented `ridge=1e3` fallback, which is tuned
    for POSITION targets of std ~240 map units and recovers nothing from a
    displacement -- max error 5.87 on a signal of magnitude 3, against 0.017
    for the selected 1e-1. Both of the brief's known-answer tests fail against
    the `fit_displacement_probe` the brief supplies. See that function.
  * THE VALIDATION GATHER IS NOT `gather_probe_data`'S DEFAULT. That default,
    `limit=PROBE_EPISODE_LIMIT` (20), is the rule for how many TRAINING
    episodes the probe is FIT on. The shipped validation split is 24 episodes
    and 229 windows; gathering it at the default scores the first 20 episodes,
    189 windows -- a per-window series that no longer lines up, row for row,
    with the `windows.episode` index every downstream contrast is clustered
    on. The val gather therefore passes `limit=len(val)` explicitly, and
    `require_aligned_windows` below refuses any count or order that is not the
    cell's own rather than truncating, padding or reordering to fit.
  * `K_REPORTED` REACHES 45, which needs 46 rows measured from t0. The
    shipped protocol's horizon is 45, so it leaves exactly 46 and every
    reported k fits with none to spare; a test fixture at a shorter horizon
    cannot carry it, so `main` takes `ks` as a parameter with `K_REPORTED` as
    its default -- the sharper_latent `taus` idiom -- and a grid a cell's
    protocol cannot carry is refused for every cell BEFORE any pass runs,
    rather than being silently scored at a smaller k that would then pool as
    if it were this one.

THE PROBE'S ANCHOR IS THE ROLLOUT'S t0, NOT THE WINDOW'S FIRST GATHERED ROW.
`gather_probe_data`'s row 0 is the posterior after ONE real frame out of a
zero RSSM state; displacement is a two-frame quantity, so a latent there
cannot encode velocity even in principle and its only route to a positive
contrast would be a correlation between absolute position and displacement --
biasing this milestone toward NO_MOTION, its own hypothesis, through the
read-out point rather than through the latent. Both the fit and the scoring
read row `context - 1`, the frame the M3 gate's persistence baseline freezes
at and the frame `require_aligned_windows` already checks the gather against.
`anchor_at_t0` does the slicing, so `displacement` still anchors at ITS own
row 0 and its shipped contract is untouched.

Two smaller ones, recorded where they bit: `self_check` is NOT on
`trust_horizon.Prepared` (the record's comes from `self_check(traj,
cell.diagnostic)` on this script's own pass), and `windows` is NOT on the
study result record at all -- `windows.total` and `windows.episode` live on
the DIAGNOSTIC record, which is where they are copied from.

THE DESCRIPTIVE BLOCK reads the posterior over the HORIZON steps
(`post_logits[:, context:]`) against `prior_teacher_logits`, the same slice
`stage_decomposition.cell_statistics` reads, because the teacher-forced prior
exists only over those steps and `latent_description` refuses two shapes that
do not describe the same windows. `information` is recomputed from the same
pass and written beside it (spec 2.3), so this milestone's headline number and
M3g's come off one pass rather than two.
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

import mbfps.eval.pooling as pooling
from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.aggregate import SEEDS
from mbfps.eval.diagnostics import reference_trajectories
from mbfps.eval.motion import (
    CONTROL_SEED,
    K_REPORTED,
    MOTION_FAMILY,
    SEEDS_REQUIRED,
    MotionArm,
    MotionInputs,
    MotionStatus,
    contrast_series,
    displacement,
    format_reading_displacement,
    latent_description,
    motion_threshold,
    reading_displacement,
)
from mbfps.eval.probe import (
    PROBE_EPISODE_LIMIT,
    apply_probe,
    fit_probe,
    gather_probe_data,
    probe_episodes,
)
from mbfps.eval.split_gap import DECISION_H, fmt_z
from mbfps.eval.stages import information
from mbfps.eval.study import SPLIT_SEED, git_sha, load_record, write_record
from mbfps.utils.config import ARMS
from mbfps.utils.device import get_device


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_latent_motion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_trust = _sibling("trust_horizon")
Cell = _trust.Cell
CellMissing = _trust.CellMissing
load_cell = _trust.load_cell
self_check = _trust.self_check
prepare_cell = _trust.prepare_cell
cell_series = _trust.cell_series

EXIT_OK = _trust.EXIT_OK
EXIT_NO_CHECKPOINTS = _trust.EXIT_NO_CHECKPOINTS
EXIT_SPLIT_MISMATCH = _trust.EXIT_SPLIT_MISMATCH
EXIT_RECORD_MISMATCH = _trust.EXIT_RECORD_MISMATCH
EXIT_SELF_CHECK_FAILED = _trust.EXIT_SELF_CHECK_FAILED
EXIT_CONTROL_LEAKED = 38
"""The permuted control cleared the bar. The pairing it scores cannot carry
signal, so a control that clears means the instrument is reading structure
that is not there -- and no reading is taken."""

PHASES: tuple[str, ...] = ("measure", "read", "all")

SELECT_EPISODES: int = 4
"""How many of `PROBE_EPISODE_LIMIT`'s training episodes are held back to
SELECT the displacement probe's ridge rather than fit it -- `fit_probes`'
`select_episodes` default, pinned to it by a test so the two probes are
selected on the same share of the same split."""


# ---------------------------------------------------------------------------
# Paths, the probe and the control.
# ---------------------------------------------------------------------------


def motion_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    """One file per cell, named by BOTH the arm and the seed -- see
    `study.job_record_path` for what a colliding name costs."""
    return Path(out_dir) / f"motion_{arm}_seed{seed}.json"


def k_key(k: int) -> str:
    """`15 -> "k15"`. A record key may not contain '.', since `write_record`
    addresses non-finite fields by dotted path (`split_gap.q_key`'s rule); the
    reported horizons are integers, so the key is simply the integer."""
    return f"k{int(k)}"


MAX_DERANGEMENT_DRAWS: int = 1000
"""How many redraws `permute_pairing` will take before it gives up. A random
permutation is a derangement with probability -> 1/e, so a thousand
consecutive failures has probability ~1e-201: the bound exists so a bug
cannot hang the run, never because it is expected to be reached."""


def permute_pairing(n: int, seed: int) -> np.ndarray:
    """A DERANGEMENT of `n` rows: every window is paired with some other
    window's displacement, never its own.

    A plain shuffle would leave roughly one window in e paired with itself,
    each contributing real signal to a control whose whole claim is that it
    has none. Fixed seed so a refusal can be reproduced.

    DRAWN BY REJECTION -- redraw until no row keeps its own index -- which is
    uniform over the derangements and takes about e draws.

    NOT the task brief's repair, which swapped each fixed point with its
    right-hand neighbour and wrapped the last one onto index 0. That repair
    does not always derange: measured over n in 2..39 and seeds 0..1999 it
    still left a fixed point in 1370 of 76,000 cases. The smallest is n = 2 on
    the identity, where the two fixed points swap to [1, 0] and then straight
    back to [0, 1]; at n = 3 the identity ends [0, 2, 1]. A control that
    silently keeps a window's own displacement is the single failure this
    function exists to prevent, so it is drawn rather than patched.
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"a derangement needs at least 2 rows, got {n}")
    rng = np.random.default_rng(seed)
    index = np.arange(n)
    for _ in range(MAX_DERANGEMENT_DRAWS):
        order = rng.permutation(n)
        if not (order == index).any():
            return order
    raise RuntimeError(
        f"no derangement of {n} rows in {MAX_DERANGEMENT_DRAWS} draws from seed {seed}; "
        "a random permutation deranges with probability ~1/e, so this is a bug, not luck"
    )


def fit_displacement_probe(data: dict, k: int, select: dict | None = None) -> dict:
    """Ridge from the posterior latent to the displacement, in `fit_probe`'s
    own shape so the two probes are fit and applied the same way.

    `data` carries `"latent"` `(N, LATENT)` and `"displacement"` `(N, 2)`,
    row-aligned. `k` is recorded on the probe so a probe fit at one horizon
    cannot be applied at another in silence.

    `select` IS NOT OPTIONAL FOR A REAL READING, and this is the one place in
    this file where a defect would have decided the milestone. `fit_probe`
    with no selection data falls back to `ridge=1e3` -- its documented
    default, chosen for `(N, 4)` POSITION targets whose std is ~240 map units.
    Displacement over k steps is one to two orders of magnitude smaller, and
    measured on the brief's own known-answer case (latents that encode the
    displacement EXACTLY, 64 rows of 8 features) that fallback reads the
    displacement back with max error 5.87 against a signal of magnitude ~3:
    it recovers nothing. The selected 1e-1 reads the same case back to 0.017.
    Taking the fallback would therefore have underfit every displacement probe
    M3i fits and biased the milestone toward NO_MOTION -- its own hypothesis,
    reached through the probe rather than through the latent.

    So `measure_cell` always passes `select`, gathered from held-out TRAINING
    episodes at episode granularity, exactly as `fit_probes` selects the
    position probe's ridge and for the reason its own comment gives.
    """
    latent, target = np.asarray(data["latent"]), np.asarray(data["displacement"])
    if select is None:
        probe = fit_probe(latent, target)
    else:
        probe = fit_probe(
            latent, target,
            np.asarray(select["latent"]), np.asarray(select["displacement"]),
        )
    return {**probe, "k": int(k)}


def apply_displacement_probe(probe: dict, latents) -> np.ndarray:
    """The probe's predicted displacement, `(n, 2)`."""
    return apply_probe(probe, np.asarray(latents))


# ---------------------------------------------------------------------------
# The rows, and the alignment every downstream number rests on.
# ---------------------------------------------------------------------------


def window_rows(data: dict) -> dict:
    """`gather_probe_data`'s flat rows, reshaped per window.

    Returns `"latent"` `(n_windows, steps, LATENT)` and `"positions"`
    `(n_windows, steps, 2)` -- `targets[..., :2]`, pos_x and pos_y in Doom map
    units, never the angle's sin/cos.

    The gather appends whole windows in episode order and numbers them
    0, 1, 2, ..., so the rows are already grouped and ordered; this only folds
    them. Refuses a ragged gather rather than reshaping it: `reshape(n, -1)`
    on unequal windows either raises somewhere far away or, where the counts
    happen to divide, produces a grid whose rows straddle two windows.
    """
    window = np.asarray(data["window"], dtype=np.int64)
    step = np.asarray(data["step"], dtype=np.int64)
    counts = np.bincount(window)
    if counts.size == 0 or not np.all(counts == counts[0]):
        raise ValueError(
            "every gathered window must have the same number of rows; got "
            f"{sorted(set(counts.tolist()))}"
        )
    n_windows, steps = int(counts.size), int(counts[0])
    order = np.lexsort((step, window))
    if not np.array_equal(order, np.arange(window.size)):
        raise ValueError(
            "the gathered rows are not in (window, step) order; the reshape below "
            "would fold another window's step into this one"
        )
    latent = np.asarray(data["latent"], dtype=np.float64).reshape(n_windows, steps, -1)
    positions = np.asarray(data["targets"], dtype=np.float64).reshape(n_windows, steps, -1)[..., :2]
    return {"latent": latent, "positions": positions, "windows": n_windows, "steps": steps}


def require_aligned_windows(
    positions, at_context, true_positions, *, context: int, total: int, arm: str, seed: int
) -> None:
    """Refuse unless the gathered validation windows ARE the cell's own, in the
    cell's own order.

    Every per-window series this script writes is paired POSITIONALLY with the
    record's `windows.episode` by `trust_horizon.cell_series`, which is how the
    contrasts are clustered by episode. A val gather of a different count, or
    of the same count in a different order, therefore scores every window
    against another window's displacement -- and nothing downstream would
    notice, because the shapes still agree and the numbers still look
    plausible. So this compares the gathered truth against the pass's own,
    frame for frame, and refuses: it never truncates, pads or reorders to fit.

    `positions` is `(n, context + horizon, 2)` from `window_rows`, whose row
    `j` is frame `start + 1 + j`; `at_context` is the pass's
    `true_at_context`, frame `start + context`; `true_positions` is its
    `(n, horizon, 2)`, frames `start + context + 1 ...`. Both sides come from
    `probe_targets` on the same `privileged` rows, so equality is exact.
    """
    positions = np.asarray(positions, dtype=float)
    n = int(positions.shape[0])
    if n != int(total):
        raise ValueError(
            f"{arm} seed {seed}: the validation gather produced {n} val windows, but the "
            f"cell's record says windows.total is {int(total)}. Every per-window series is "
            "paired positionally with windows.episode, so these cannot be reconciled by "
            "truncating or padding -- the two passes did not cut the same windows."
        )
    expected = np.concatenate(
        [np.asarray(at_context, dtype=float)[:, None, :],
         np.asarray(true_positions, dtype=float)], axis=1,
    )
    got = positions[:, int(context) - 1:, :]
    if got.shape != expected.shape or not np.array_equal(got, expected):
        raise ValueError(
            f"{arm} seed {seed}: the gathered validation windows are not the windows the "
            f"record was scored on, or not in its order (gathered {got.shape} against the "
            f"pass's {expected.shape}). Pairing them anyway would score every window "
            "against another window's displacement with every shape still agreeing."
        )


def require_reportable_ks(ks, *, context: int, horizon: int, arm: str, seed: int) -> None:
    """Refuse a grid this cell's protocol cannot carry, BEFORE any pass runs.

    `displacement` needs `k + 1` rows, and it is handed `positions[:, context
    - 1:]` -- the rows from the rollout's own t0 onward, of which there are
    `horizon + 1`, the `context - 1` rows before t0 having been consumed by
    the anchor. So the grid fits exactly when `k <= horizon`. The shipped
    protocol's horizon is 45 and `K_REPORTED` tops out at 45, so every
    reported k fits, with none to spare.

    A window silently scored at a smaller k would be a different horizon
    pooled as if it were this one, so the whole grid is refused rather than
    any part of it truncated.
    """
    rows = int(horizon) + 1
    for k in ks:
        if int(k) + 1 > rows:
            raise ValueError(
                f"{arm} seed {seed}: k={int(k)} needs {int(k) + 1} rows from the rollout's t0, "
                f"but this protocol (context {int(context)}, horizon {int(horizon)}) leaves "
                f"{rows} -- the probe anchors at t0, so the context steps before it are gone"
            )


# ---------------------------------------------------------------------------
# measure: one cell.
# ---------------------------------------------------------------------------


def _cell_args(args, source: Path) -> types.SimpleNamespace:
    """`prepare_cell` reads `out`, `device`, `context` and `horizon` off its
    args. `out` is the STUDY directory the checkpoint is loaded from -- this
    script's own `--out` holds the motion records and nothing else."""
    return types.SimpleNamespace(
        out=Path(source), device=args.device, context=args.context, horizon=args.horizon,
    )


def anchor_at_t0(rows: dict, context: int) -> tuple[np.ndarray, np.ndarray]:
    """The probe's read-out point and the positions measured from it.

    Returns `(latent, positions)`: the posterior latent at window row
    `context - 1`, `(n, LATENT)`, and `positions[:, context - 1:]`,
    `(n, horizon + 1, 2)` whose row 0 IS that same frame.

    WHY `context - 1` AND NOT 0. `gather_probe_data`'s row `j` is frame
    `start + 1 + j`, so row 0 is the posterior after observing ONE real frame
    out of a zero RSSM state. Displacement is a two-frame quantity, so a
    latent with one frame of history cannot encode velocity even in
    principle -- its only route to a positive contrast is a correlation
    between absolute position and displacement, which would bias M3i toward
    NO_MOTION, the hypothesis under test, through the read-out point rather
    than through the latent.

    Row `context - 1` is frame `start + context`: the rollout's own t0, the
    frame the M3 gate's persistence baseline freezes at and the frame
    `require_aligned_windows` already compares the gather against
    (`traj.true_at_context`). The latent there has the full context of real
    frames the rollout itself starts from.

    `displacement`'s contract is untouched -- it still anchors at ITS row 0 --
    because the slice happens here, before the call.
    """
    row = int(context) - 1
    if row < 0:
        raise ValueError(f"context must be at least 1 real frame, got {context}")
    return rows["latent"][:, row, :], rows["positions"][:, row:, :]


def _k_entry(
    train_rows: dict, select_rows: dict | None, val_rows: dict, k: int, context: int
) -> dict:
    """One reported horizon: the probe fit on TRAINING windows with its ridge
    selected on held-out TRAINING windows, scored per VALIDATION window
    against staying put, and scored again with the pairing permuted.

    Fitting on train and scoring on val is the point, not a leak:
    `fit_probes`' docstring warns that passing validation paths TO THE FIT
    would leak, and these rows are gathered to SCORE.

    THE PROBE READS THE LATENT AT THE ROLLOUT'S OWN t0 -- window row
    `context - 1` -- and `displacement` measures `p(t+k) - p(t)` from that
    same frame, because `anchor_at_t0` slices the positions to start there.
    The fit and the scoring take the SAME row: a probe fit at one read-out
    point and applied at another is not the probe anyone reported. One row per
    window at every k, so the series stays alignable with the per-window
    episode index the pooling clusters on.

    The control permutes which window's TRUE displacement each prediction is
    read against. Both marginals are untouched -- the same predictions, the
    same displacements -- and only the pairing is destroyed, which is what
    makes a control that clears a broken instrument rather than a result.
    """
    train_latent, train_positions = anchor_at_t0(train_rows, context)
    val_latent, val_positions = anchor_at_t0(val_rows, context)
    train_true = displacement(train_positions, k)
    val_true = displacement(val_positions, k)
    select = None
    if select_rows is not None:
        select_latent, select_positions = anchor_at_t0(select_rows, context)
        select = {
            "latent": select_latent,
            "displacement": displacement(select_positions, k),
        }
    probe = fit_displacement_probe(
        {"latent": train_latent, "displacement": train_true}, k, select,
    )
    predicted = apply_displacement_probe(probe, val_latent)
    order = permute_pairing(val_true.shape[0], CONTROL_SEED)
    return {
        "k": int(k),
        "contrast": contrast_series(predicted, val_true),
        "control": contrast_series(predicted, val_true[order]),
        "ridge": float(probe["ridge"]),
        "train_windows": int(train_true.shape[0]),
    }


def measure_cell(args, cell: Cell, device, train, val, ks=K_REPORTED) -> tuple[int, dict | None]:
    """One cell: 12/14 from `prepare_cell`, the canonical pass and its
    self-check (30), the two gathers, then one entry per reported horizon.

    `(EXIT_OK, record)` once the record is written, `(status, None)` on any
    refusal, with the refusal printed.
    """
    arm, seed = cell.arm, cell.seed
    status, prepared = prepare_cell(_cell_args(args, args.reference), cell, device, train, val)
    if status != EXIT_OK:
        return status, None
    context, horizon = prepared.context, prepared.horizon
    require_reportable_ks(ks, context=context, horizon=horizon, arm=arm, seed=seed)

    # The canonical pass, with the latent fields kept (M3g's `keep_latents`).
    # It serves three purposes at once and must not be split into three: the
    # self-check below, the descriptive block's logits, and the per-window
    # truth `require_aligned_windows` compares the gather against.
    traj = reference_trajectories(
        prepared.model, val, prepared.embedding_probe, keep_latents=True, **prepared.common,
    )
    check = self_check(traj, cell.diagnostic)
    if not check.ok:
        print(
            f"\nSELF-CHECK FAILED for {arm} seed {seed}: " + "; ".join(check.failures())
            + ". Same windows, same rollout, same refit probe -- or this is not measuring "
            "what the gate measured. No record written."
        )
        return EXIT_SELF_CHECK_FAILED, None

    backbone = prepared.common["feature_backbone"]
    gather = lambda paths, s: gather_probe_data(  # noqa: E731
        prepared.model, paths, backbone, device,
        context=context, horizon=horizon, limit=len(paths), seed=s,
    )
    # TRAIN over `PROBE_EPISODE_LIMIT` episodes -- `gather_probe_data`'s own
    # default and the rule `fit_probes` fits the position probe under, so this
    # probe sees the same episodes that one did -- split at EPISODE
    # granularity into the block the weights are fit on and the block the
    # ridge is selected on, which is `fit_probes`' split for its own reason:
    # windows from one episode on both sides is not held out in any
    # meaningful sense, and every ridge would look equally good.
    used, _ = probe_episodes(train, PROBE_EPISODE_LIMIT)
    spare = len(used) - SELECT_EPISODES
    fit_paths, select_paths = (used[:spare], used[spare:]) if spare >= 1 else (used, [])
    train_rows = window_rows(gather(fit_paths, seed))
    select_rows = window_rows(gather(select_paths, seed + 1)) if select_paths else None
    # VAL over EVERY val episode. `gather_probe_data`'s `limit` default is a
    # FIT rule; the shipped val split is 24 episodes and 229 windows, so
    # taking that default here would score the first 20 episodes -- 189
    # windows -- against a 229-row episode index. See the module docstring.
    val_rows = window_rows(gather(list(val), seed))
    require_aligned_windows(
        val_rows["positions"], traj.true_at_context, traj.true_positions,
        context=context, total=cell.diagnostic["windows"]["total"], arm=arm, seed=seed,
    )

    post = np.asarray(traj.post_logits, dtype=float)
    teacher = np.asarray(traj.prior_teacher_logits, dtype=float)
    record = {
        "arm": arm,
        "seed": int(seed),
        "source": str(args.reference),
        "step": int(cell.record["steps"]),
        "record_git_sha": cell.record.get("git_sha", "unknown"),
        "context": int(context),
        "horizon": int(horizon),
        "ks": [int(k) for k in ks],
        "split_seed": SPLIT_SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "git_sha": git_sha(),
        "episodes": {"val": list(cell.record["episodes"]["val"])},
        # COPIED from the cell's own diagnostic, never recomputed: a pool over
        # cells that scored different windows must be refused by
        # `require_compatible`, and a recomputed index would agree with itself.
        "windows": {
            "total": int(cell.diagnostic["windows"]["total"]),
            "episode": [int(e) for e in cell.diagnostic["windows"]["episode"]],
        },
        "self_check": check.record(),
        "k": {
            k_key(k): _k_entry(train_rows, select_rows, val_rows, k, context) for k in ks
        },
        # The posterior over the HORIZON steps against the teacher-forced
        # prior, which exists only there -- `stage_decomposition` reads the
        # same slice, and `latent_description` refuses two shapes that do not
        # describe the same windows.
        "description": latent_description(post[:, context:], teacher),
        "information": _information_block(post, teacher, context),
    }
    path = motion_record_path(args.out, arm, seed)
    write_record(path, record)
    means = " / ".join(
        f"k {k}: {np.mean(record['k'][k_key(k)]['contrast']):+.2f} "
        f"(ctl {np.mean(record['k'][k_key(k)]['control']):+.2f})"
        for k in ks
    )
    print(f"{arm} seed {seed}: {means}; wrote {path}")
    return EXIT_OK, record


def _information_block(post_logits, prior_teacher_logits, context: int) -> dict:
    """KL(posterior || teacher-forced prior) summed over groups, per window
    (spec 2.3). Recomputed from THIS pass so M3i's headline number and M3g's
    0.290 / 0.467 / 0.489 are read off the same measurement rather than quoted
    across milestones. Both reductions, because embedding-style per-window
    distributions on this project are right-skewed and a mean printed under a
    caption that says median is how M3h shipped a premise it never measured."""
    per_window = information(post_logits, prior_teacher_logits, int(context))
    return {
        "mean": float(np.mean(per_window)),
        "median": float(np.median(per_window)),
    }


def measure_phase(args, cells, device, train, val, ks=K_REPORTED) -> int:
    """Every requested cell, refusing the whole run before any pass when a
    cell is missing (11) or the grid does not fit its protocol."""
    try:
        loaded = [load_cell(args.reference, arm, seed) for arm, seed in cells]
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    for cell in loaded:
        require_reportable_ks(
            ks,
            context=args.context if args.context is not None else cell.diagnostic["context"],
            horizon=args.horizon if args.horizon is not None else cell.diagnostic["horizon"],
            arm=cell.arm, seed=cell.seed,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    for cell in loaded:
        status, _ = measure_cell(args, cell, device, train, val, ks=ks)
        if status != EXIT_OK:
            return status
    return EXIT_OK


# ---------------------------------------------------------------------------
# read: the records pooled into Reading D.
# ---------------------------------------------------------------------------


def load_motion(out_dir: Path, arms, seeds) -> dict:
    """Every planned cell's record, keyed by `(arm, seed)` -- or `CellMissing`
    naming the first that is not on disk.

    Named, never skipped: a pool over the cells that happen to be present,
    printed under the nine cells' names, is exactly the failure
    `pooling.MissingCell` exists for. A record whose own `arm`/`seed` disagree
    with the file it was read under is refused too -- a swapped pair of files
    pools one cell under another's name with every count still right.
    """
    records: dict[tuple[str, int], dict] = {}
    for arm in arms:
        for seed in seeds:
            path = motion_record_path(out_dir, arm, int(seed))
            if not path.exists():
                raise CellMissing(
                    f"{arm} seed {int(seed)}: no motion record at {path}; "
                    "run --phase measure first"
                )
            record = load_record(path)
            if record.get("arm") != arm or int(record.get("seed", -1)) != int(seed):
                raise ValueError(
                    f"{path.name} was read for {arm} seed {int(seed)} but its record says "
                    f"arm={record.get('arm')!r} seed={record.get('seed')!r}"
                )
            records[(arm, int(seed))] = record
    return records


def require_one_protocol(records: dict) -> None:
    """Every record reports the same horizons on the same windows, or the two
    that disagree are named with the field.

    `pooling.require_compatible` runs inside `pool_arm` and already refuses a
    pool over cells of ONE arm that did not score the same windows. It never
    sees two arms together here, because the arms are pooled separately and
    only their z's meet in the reading -- so the CROSS-ARM identity has no
    other backstop, and `clusters` (the one number `z_fam` is read against)
    would silently become whichever arm was pooled last. The `ks` grid is
    checked with it: a record measured at another grid either has no entry at
    the horizon this decides on, or has one at a k the others never reported.
    """
    items = sorted(records.items())
    if not items:
        raise ValueError("no motion record to read")
    (first_cell, first) = items[0]
    for cell, record in items[1:]:
        for field, pick in (
            ("ks", lambda r: list(r["ks"])),
            ("windows.episode", lambda r: list(r["windows"]["episode"])),
            ("episodes.val", lambda r: list(r["episodes"]["val"])),
            ("context", lambda r: int(r["context"])),
            ("horizon", lambda r: int(r["horizon"])),
            ("device", lambda r: str(r["device"])),
            ("torch_version", lambda r: str(r["torch_version"])),
        ):
            mine, theirs = pick(record), pick(first)
            if mine != theirs:
                shown = (
                    f" ({len(mine)} vs {len(theirs)} windows)" if field == "windows.episode"
                    else f": {mine!r} vs {theirs!r}"
                )
                raise ValueError(
                    f"{cell[0]} seed {cell[1]} and {first_cell[0]} seed {first_cell[1]} "
                    f"disagree on {field}{shown}; they are not one measurement and their "
                    "arms cannot be read against one bar"
                )


def _arm_from_pool(pooled: pooling.PooledMean, cells) -> MotionArm:
    """One arm's `MotionArm`: the pooled estimate with its episode-clustered
    standard error, and the seed tallies counted PER SEED.

    `seeds_up` / `seeds_down` are the number of this arm's cells whose OWN
    per-seed z clears `+-z_fam` -- each cell pooled alone, that seed's windows
    clustered by episode, which is `trust_horizon`'s `per_seed` idiom and
    `_pooled_mean`'s shape. Reading them off the POOLED z instead would make
    every arm read 0/3 or 3/3 by construction, and `MotionArm.clears_up`'s
    replication clause -- the requirement that an arm hold in at least
    `SEEDS_REQUIRED` of its seeds -- would stop binding on anything.

    The bar is recomputed here from `pooled.clusters` rather than passed in,
    so an arm's tallies are read against the same threshold its row is: every
    cell scores the same windows (`require_one_protocol`) and every window
    counts, so this is the one cluster count in the file.
    """
    z_fam = motion_threshold(int(pooled.clusters))
    per_seed = [pooling.pool_arm([cell]).z for cell in cells]
    return MotionArm(
        estimate=float(pooled.mean),
        se=float(pooled.se),
        z=float(pooled.z),
        seeds_up=int(sum(z >= z_fam for z in per_seed)),
        seeds_down=int(sum(z <= -z_fam for z in per_seed)),
        seeds_total=len(cells),
    )


def motion_inputs(records: dict, k: int) -> MotionInputs:
    """The per-cell records reduced to what Reading D is decided on at `k`.

    TREATMENT AND CONTROL ARE BUILT THE SAME WAY -- both through `cell_series`
    onto the record's own `windows.episode`, then `pool_arm`, then
    `_arm_from_pool` -- differing only in which of the record's two per-window
    series goes in. That is what makes the control a control, rather than a
    differently-computed number that happens to be called one: anything that
    moves the treatment's number by a route other than the data moves the
    control's identically, and the comparison survives it.
    """
    key = k_key(k)
    arms: dict[str, MotionArm] = {}
    control: dict[str, MotionArm] = {}
    clusters = 0
    for arm in sorted({a for a, _ in records}):
        treat_cells, ctrl_cells = [], []
        for (a, seed), record in sorted(records.items()):
            if a != arm:
                continue
            if key not in record["k"]:
                raise KeyError(
                    f"{arm} seed {seed}: no {key!r} entry; the record was measured at "
                    f"ks={list(record['ks'])}"
                )
            entry = record["k"][key]
            # EVERY VALIDATION WINDOW COUNTS, which is why the mask is all
            # ones rather than an oversight. `changed` exists in the trust and
            # ladder readings to drop windows an intervention left untouched,
            # where the recorded delta is an exact zero that is no
            # measurement. There is no such category here: the contrast is
            # `||true|| - ||predicted - true||`, defined at every window,
            # including one whose true displacement is the zero vector -- and
            # there it reads `-||predicted||`, a real penalty on the probe for
            # claiming motion that did not happen. Masking those out would
            # drop precisely the windows the persistence baseline is hardest
            # to beat on and read the latent against an easier question. The
            # series is also paired POSITIONALLY with `windows.episode`
            # (`cell_series`' contract), so the rows must stay the record's
            # own, in the record's own order, neither filtered nor reordered.
            changed = np.ones(len(entry["contrast"]), dtype=bool)
            treat_cells.append(cell_series(
                arm, seed, "displacement",
                np.asarray(entry["contrast"], dtype=float), changed, record,
            ))
            ctrl_cells.append(cell_series(
                arm, seed, "displacement",
                np.asarray(entry["control"], dtype=float), changed, record,
            ))
        treat, ctrl = pooling.pool_arm(treat_cells), pooling.pool_arm(ctrl_cells)
        clusters = int(treat.clusters)
        arms[arm] = _arm_from_pool(treat, treat_cells)
        control[arm] = _arm_from_pool(ctrl, ctrl_cells)
    return MotionInputs(
        arms=arms,
        control=control,
        z_fam=motion_threshold(clusters),
        k=int(k),
        clusters=clusters,
    )


# ---------------------------------------------------------------------------
# read: the tables.
# ---------------------------------------------------------------------------


def _self_check_table(records: dict) -> str:
    """What this script's own pass reproduced, per cell. A reading is only as
    good as the pass under it, so the operator sees the deltas rather than a
    boolean that was decided somewhere out of sight."""
    lines = [
        "--- self-check per record: this script's pass against the cell's diagnostic "
        "(mbfps.eval.reproduction's bound, spec 2.4), and the windows it was scored on ---",
        f"  {'arm':<12}{'seed':>5}{'step':>8}{'ref max|d|':>12}{'pers max|d|':>13}"
        f"{'windows':>9}{'clusters':>10}  ok",
    ]
    for (arm, seed), record in sorted(records.items()):
        check = record["self_check"]
        lines.append(
            f"  {arm:<12}{int(seed):>5}{int(record['step']):>8}"
            f"{fmt_z(float(check['reference_position_max_delta']), '.1e'):>12}"
            f"{fmt_z(float(check['persistence_position_max_delta']), '.1e'):>13}"
            f"{int(record['windows']['total']):>9}"
            f"{len(set(record['windows']['episode'])):>10}"
            f"  {'yes' if check['ok'] else 'NO'}"
        )
    return "\n".join(lines) + "\n"


def _description_table(records: dict) -> str:
    """Spec 2.3's descriptive block, reported and deciding nothing. `live` is
    the MEDIAN over windows of how many groups change argmax within a window
    -- the spec's reduction, named in the caption because a mean printed under
    a caption that says median is how M3h shipped a premise it never
    measured."""
    lines = [
        "--- the latent, described (spec 2.3; reported, deciding nothing): mean posterior "
        "entropy per group against its own ln(classes) ceiling; `live` = the MEDIAN over "
        "windows of how many of `groups` change argmax within the window; top-1 mass for the "
        "posterior and the teacher-forced prior; KL(post || teacher prior) per window, nats ---",
        f"  {'arm':<12}{'seed':>5}{'entropy':>9}{'ln(C)':>8}{'live':>7}{'groups':>8}"
        f"{'top1 post':>11}{'top1 prior':>12}{'KL mean':>9}{'KL med':>9}",
    ]
    for (arm, seed), record in sorted(records.items()):
        d, info = record["description"], record["information"]
        lines.append(
            f"  {arm:<12}{int(seed):>5}"
            f"{fmt_z(float(d['entropy_mean']), '.3f'):>9}"
            f"{fmt_z(float(d['entropy_max']), '.3f'):>8}"
            f"{fmt_z(float(d['live_groups']), '.1f'):>7}"
            f"{len(d['entropy_by_group']):>8}"
            f"{fmt_z(float(d['top1_posterior']), '.3f'):>11}"
            f"{fmt_z(float(d['top1_prior']), '.3f'):>12}"
            f"{fmt_z(float(info['mean']), '.3f'):>9}"
            f"{fmt_z(float(info['median']), '.3f'):>9}"
        )
    return "\n".join(lines) + "\n"


def _per_k_table(per_k: dict, decision_h: int) -> str:
    """Every reported horizon, with the one that decides marked as such.

    The `decides` column is the whole point of printing the others: M3i is
    read at `DECISION_H` alone, exactly as the M3 gate is, and a table of five
    horizons with no column saying which one was decided on is a table whose
    reader picks the horizon that suits them.
    """
    lines = [
        "--- the displacement contrast per reported horizon (per-window ||true|| - "
        "||predicted - true||, map units, against staying put; seeds averaged per window, "
        f"episode-clustered); `ctl z` is the permuted pairing's; k = {int(decision_h)} DECIDES "
        "and every other k is reported and decides nothing ---",
        f"  {'k':<6}{'arm':<12}{'estimate':>12}{'se':>9}{'z':>8}{'up':>7}{'dn':>7}"
        f"{'ctl z':>9}{'decides':>9}",
    ]
    for k, inputs in sorted(per_k.items()):
        for arm in sorted(inputs.arms):
            a, c = inputs.arms[arm], inputs.control[arm]
            lines.append(
                f"  {f'k={int(k)}':<6}{arm:<12}"
                f"{fmt_z(a.estimate, '+.4f'):>12}{fmt_z(a.se, '.4f'):>9}{fmt_z(a.z):>8}"
                f"{f'{a.seeds_up}/{a.seeds_total}':>7}{f'{a.seeds_down}/{a.seeds_total}':>7}"
                f"{fmt_z(c.z):>9}"
                f"{('yes' if int(k) == int(decision_h) else 'no'):>9}"
            )
    return "\n".join(lines) + "\n"


def _pooling_caption(inputs: MotionInputs) -> str:
    return (
        f"  pooling: z_fam = cluster_threshold({MOTION_FAMILY}, {inputs.clusters}) = "
        f"{inputs.z_fam:.2f}; each arm's contrast is a per-window series pooled over its seeds "
        "(seeds averaged per window, episode-clustered), and an arm clears pooled AND in at "
        f"least {SEEDS_REQUIRED} of its own seeds. EVERY validation window counts -- unlike "
        "the trust readings there is no 'moved' mask, because the contrast is a measurement "
        "at every window, including one whose true displacement is zero. The control is the "
        "SAME predictions read against another window's displacement, pooled by the same "
        "route, and it decides nothing except whether a reading is taken at all.\n"
    )


def motion_text(records: dict, per_k: dict, inputs: MotionInputs, reading: MotionStatus) -> str:
    """Everything `read` prints, in `stages.txt`'s style; written to
    `motion.txt` byte-identical -- see `read_phase` for the one case in which
    it is not written at all."""
    return "".join([
        _self_check_table(records),
        _description_table(records),
        _per_k_table(per_k, inputs.k),
        _pooling_caption(inputs),
        format_reading_displacement(reading, inputs),
    ])


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def read_phase(args) -> int:
    """Pool every requested cell (11 names the first missing), decide Reading
    D at `DECISION_H`, print the tables -- and write `motion.txt` only if the
    control did not leak.

    THE FORMATTER STILL PRINTS THE PER-ARM ROWS ON A LEAK, and that is
    deliberate: an operator debugging a control that cleared needs to see the
    numbers it cleared beside. The suppression is purely the file: exit 38
    leaves no `motion.txt` for a later reader -- or a later milestone's quote
    -- to mistake for a result.
    """
    try:
        records = load_motion(args.out, args.arms, [int(s) for s in args.seeds])
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    require_one_protocol(records)
    ks = [int(k) for k in next(iter(records.values()))["ks"]]
    if DECISION_H not in ks:
        raise ValueError(
            f"these records were measured at ks={ks}, which does not include the "
            f"pre-registered DECISION_H = {DECISION_H}. Reading D is decided there and "
            "nowhere else, so it is refused rather than taken at a neighbouring horizon "
            "that would then be read as this one."
        )
    per_k = {k: motion_inputs(records, k) for k in ks}
    inputs = per_k[DECISION_H]
    reading = reading_displacement(inputs)
    text = motion_text(records, per_k, inputs, reading)
    print(text, end="")
    if reading.status == "UNRESOLVED_CONTROL":
        print(
            f"\nCONTROL LEAKED at k = {inputs.k}: {reading.rule}. No reading is taken and no "
            f"{args.out / 'motion.txt'} is written -- a suppressed reading must not leave an "
            "artefact a later reader mistakes for a result."
        )
        return EXIT_CONTROL_LEAKED
    write_text(args.out / "motion.txt", text)
    return EXIT_OK


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/m3i_motion"))
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
    return parser


def main(argv: list[str] | None = None, *, ks=K_REPORTED) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    device = get_device(prefer=args.device)
    cells = [(arm, int(seed)) for arm in args.arms for seed in args.seeds]
    if args.phase in ("measure", "all"):
        buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
        train, val = episode_split(buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
        status = measure_phase(args, cells, device, train, val, ks=ks)
        if status != EXIT_OK:
            return status
    if args.phase in ("read", "all"):
        status = read_phase(args)
        if status != EXIT_OK:
            return status
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
