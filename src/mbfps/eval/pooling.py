"""Pooling the nine diagnostic records into one statistic per arm and rung.

`scripts/diagnose_dynamics.py` decides each cell on its own, and the nine
verdicts disagree where the per-cell power is marginal: on the held-action
contrast `random_vit` reads nominal, nominal, null across its three seeds while
the sign is positive in 9 of 9. This module replaces nine per-cell readings
with two pooled ones, computed from the PER-WINDOW series the records carry:

  PER ARM x RUNG. The three seeds' per-window horizon-mean deltas are
  AVERAGED PER WINDOW (229 seed-mean rows on the shipped split), each tagged
  with the window's EPISODE, and the mean is read against an episode-clustered
  standard error. Episodes nest windows nest seeds -- the three seeds score
  the SAME 229 windows from the SAME 24 episodes -- so clustering by episode
  absorbs both the shared-window correlation and the seed replication in one
  label set; stacking the 687 rows and clustering by episode gives the
  identical mean and standard error, so "687 rows" adds nothing and is not
  how the estimator is described. The inference is over EPISODES with the
  three seeds as a FIXED factor: it licenses a claim about these three
  checkpoints on new episodes, not about a fourth training seed (that would
  have two degrees of freedom), and the effective replication is the 24
  clusters -- which is why the z is read against t(G - 1), not the normal.

  BETWEEN ARMS. The treatment/control contrast -- `frozen_ssl` minus
  `random_vit` -- is PAIRED per window: average the seeds within each arm and
  window first, then difference, then cluster by episode. The mean of a paired
  difference is pairing-invariant, so it is the per-window series and the
  clustered standard error that carry the pairing; the tests assert those.
  ON THE POSITION AND ANGLE CHANNELS THIS IS THE PATHWAY AS READ THROUGH
  EACH ARM'S OWN PROBE: every cell fits its own ridge probe (selection R^2
  0.36 on frozen_ssl, -0.036 on cnn), so the contrast subtracts two readings
  attenuated by different probes and a between-arm difference there is
  partly a probe difference. The probe-free between-arm estimand is the
  embedding-ratio contrast, and `pathway_reading` formalises share / exceed
  / lack as a reading of (treatment responds, control responds, contrast
  sign) rather than of the contrast alone -- the per-arm rows are what
  separate "both have it" from "neither has it".

  THE EMBEDDING RATIO is a ratio of medians, and a ratio of medians has no
  sandwich standard error -- so its ruler is an episode-cluster BOOTSTRAP:
  draw the episodes with replacement, take every row of each drawn episode,
  recompute. The between-arm ratio contrast applies the SAME episode draw to
  both arms in every replicate, which is what makes it paired. EACH CELL IS
  NORMALISED BY ITS OWN NOISE MEDIAN BEFORE ANYTHING IS STACKED: the
  embedding scale differs 70x between arms and differs between independently
  trained seeds within an arm, and a median over raw rows from three heads
  is a function of which seed's rows land in the middle; each cell's ratio
  is scale-free, and dividing its rows by its own noise median makes the
  pooled one scale-free too.

WHAT IS REFUSED, BY NAME, BEFORE ANY NUMBER IS COMPUTED. This repo has been
burned by tables that dropped a missing cell and still looked complete, so a
missing record, a record whose `arm`/`seed` disagree with its file name, a
record written before the per-window series existed, and two cells that do not
score the same windows -- window count, episode labels, held-out split, horizon,
context, device or torch version -- each raise their own exception type. The
addendum measured the deltas moving across devices by more than their standard
error, so cells from two devices are non-comparable and are refused rather than
warned about. Pure post-processing: no model, no torch, CPU-testable on
synthetic records.
"""

from dataclasses import dataclass
from math import exp, lgamma, log
from pathlib import Path

import numpy as np
from numpy.random import default_rng

from mbfps.eval.diagnostics import METRICS
from mbfps.eval.study import load_record

CHANNELS: tuple[str, ...] = METRICS


def cluster_standard_error(values: np.ndarray, labels: np.ndarray) -> float:
    """Standard error of `values.mean()` when the observations come in clusters.

    The shipped 229 windows are cut from 24 validation episodes at 3 to 10
    windows each, and windows from one episode share its map, its route and its
    difficulty -- so `std / sqrt(229)` claims a precision the data does not
    support. Measured on the shipped cells the cluster-robust standard error is
    up to 1.32x the naive one, and the interval it produces is exactly what the
    verdict's equivalence statement rests on.

    The estimator is the usual sandwich for a sample mean: sum the residuals
    WITHIN each cluster, square those sums, and carry the small-sample
    correction `G / (G - 1)`. Summing within the cluster before squaring is
    what makes correlated residuals add rather than cancel; squaring first
    would give the naive variance back under a longer name.

    NaN below two clusters -- a between-cluster spread cannot be estimated from
    one -- so a caller can tell "not clustered" from "clustered, and small".

    ONE implementation, here, imported by `scripts/diagnose_dynamics.py`: a
    second copy in the script would drift from this one while both kept
    passing, and the per-cell and pooled rulers would then be two statistics
    under one name.
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    groups = np.unique(labels)
    if groups.size < 2:
        return float("nan")
    residual = values - values.mean()
    total = float(sum(residual[labels == group].sum() ** 2 for group in groups))
    correction = groups.size / (groups.size - 1)
    return float(np.sqrt(correction * total) / values.size)


def _regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b) by the Lentz continued fraction (Numerical Recipes `betacf`),
    using the symmetry `I_x(a, b) = 1 - I_{1-x}(b, a)` where the fraction
    converges fastest. Here so the t quantile needs no scipy, which this
    environment does not have; checked against the standard t table in the
    tests."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularised_incomplete_beta(b, a, 1.0 - x)
    front = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x)) / a
    tiny = 1e-300
    c, d = 1.0, 1.0 / max(1.0 - (a + b) * x / (a + 1.0), tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 / max(1.0 + numerator * d, tiny)
        c = max(1.0 + numerator / c, tiny)
        h *= d * c
        numerator = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 / max(1.0 + numerator * d, tiny)
        c = max(1.0 + numerator / c, tiny)
        step = d * c
        h *= step
        if abs(step - 1.0) < 1e-15:
            break
    return front * h


def student_t_quantile(p: float, df: float) -> float:
    """The `p` quantile of Student's t with `df` degrees of freedom, by
    bisection on the CDF `1 - I_{df / (df + t^2)}(df / 2, 1 / 2) / 2`. Exact
    0.0 at p = 0.5; refuses p outside (0, 1) and df < 1."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"a quantile needs 0 < p < 1, not {p!r}")
    if df < 1:
        raise ValueError(f"Student's t needs df >= 1, not {df!r}")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -student_t_quantile(1.0 - p, df)
    cdf = lambda t: 1.0 - 0.5 * _regularised_incomplete_beta(  # noqa: E731
        df / 2.0, 0.5, df / (df + t * t)
    )
    low, high = 0.0, 1.0
    while cdf(high) < p:
        high *= 2.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if cdf(mid) < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def cluster_threshold(family: int, clusters: int) -> float:
    """The z a cluster-robust comparison must clear: Bonferroni over the
    family, read against t(G - 1) where G is the cluster count.

    With 24 clusters (the shipped split) the nominal 1.96 becomes 2.07, a
    family of 6 reads 2.89 rather than 2.64 and a family of 54 reads 3.80
    rather than 3.31 -- because the effective replication of a sandwich
    standard error is the number of clusters, and a normal quantile claims
    the precision of 229 independent windows the data does not have. The
    diagnose script's per-cell `family_threshold` is the normal quantile and
    is left as the shipped records carry it; this is the pooled table's.
    NaN below two clusters: there is no distribution to read against, and a
    NaN threshold cannot be cleared.
    """
    if family < 1:
        raise ValueError(f"a family of {family} comparisons has no threshold")
    if clusters < 2:
        return float("nan")
    return student_t_quantile(1.0 - 0.025 / family, clusters - 1)


PATHWAY_READINGS: tuple[str, ...] = ("shares", "exceeds", "lacks", "neither", "inconclusive")


def pathway_reading(treatment_responds: bool, control_responds: bool, contrast_sign: int) -> str:
    """Does the treatment SHARE the pathway, EXCEED it, or LACK it -- read
    from three verdicts, never from the contrast alone.

    `contrast_sign` is +1 or -1 for a between-arm contrast resolved in that
    direction and 0 for one that is not. A null contrast is "shares" only
    when BOTH arms respond: between two arms that both read null it is
    "neither", and the per-arm rows are what tell those apart. "exceeds" is
    a resolved positive contrast on a treatment that itself responds;
    "lacks" a resolved negative one beside a control that responds. A
    resolved contrast whose favoured arm did not respond, or one arm
    responding with the difference unresolved, is "inconclusive" -- named,
    never folded into a neighbour.
    """
    if contrast_sign not in (-1, 0, 1):
        raise ValueError(f"the contrast sign must be -1, 0 or +1, not {contrast_sign!r}")
    if contrast_sign > 0:
        return "exceeds" if treatment_responds else "inconclusive"
    if contrast_sign < 0:
        return "lacks" if control_responds else "inconclusive"
    if treatment_responds and control_responds:
        return "shares"
    if not treatment_responds and not control_responds:
        return "neither"
    return "inconclusive"


# ---------------------------------------------------------------------------
# Reading the records.
# ---------------------------------------------------------------------------


class PoolingError(ValueError):
    """Base of every refusal here, so a script can catch them as one family
    while still telling the causes apart by type."""


class MissingCell(PoolingError):
    """A planned (arm, seed) has no record on disk. Named, never skipped: a
    pool over eight cells printed under nine cells' names is the failure this
    repo has already been burned by."""


class MislabelledRecord(PoolingError):
    """The record's own `arm`/`seed` disagree with the file it was read from.
    Nothing else would notice: a swapped pair of files pools one cell under
    another's name with every count right."""


class StaleRecord(PoolingError):
    """The record predates the per-window series -- every record written
    before this change is exactly this -- or one of its series is missing or
    ragged. There is nothing to pool; the fix is to re-run the diagnostic."""


class IncompatibleCells(PoolingError):
    """Two cells do not score the same windows under the same protocol on the
    same device, so their rows cannot share a table."""


class DegenerateNoise(PoolingError):
    """A cell's noise reference measured no spread over the rows to be
    pooled, so its ratio is x / 0 -- undefined, never inf."""


def _cell_name(record: dict) -> str:
    return f"{record.get('arm')!r} seed {record.get('seed')!r}".replace("'", "")


@dataclass(frozen=True)
class CellSeries:
    """One (cell, rung, channel): every per-window series the pool reads, and
    the identity that decides what it may be pooled with."""

    arm: str
    seed: int
    rung: str
    channel: str
    delta: np.ndarray       # (n_windows,) horizon-mean delta, ALL windows
    changed: np.ndarray     # (n_windows,) bool, from `window_steps_changed > 0`
    episode: np.ndarray     # (n_windows,) int
    embedding: np.ndarray | None   # (n_windows,) the rung's embedding-space numerator
    noise: np.ndarray | None       # (n_windows,) the cell's noise reference
    windows_total: int
    val: tuple[str, ...]
    horizon: int
    context: int
    device: str
    torch_version: str

    @property
    def name(self) -> str:
        return f"{self.arm} seed {self.seed}"


def read_series(record: dict, rung: str, channel: str = "position") -> CellSeries:
    """One rung's per-window series out of one diagnostic record.

    Refuses, naming the cell and the field, a record with no episode index
    (`windows.episode` absent or null -- the pre-change schema, and a
    hand-built ladder respectively; neither can be clustered), a rung that is
    not in it, a rung without its per-window delta or changed mask, a missing
    noise reference, and any two series of different lengths. It never falls
    back to `delta_mean`: a pooled table over the scalars would print a
    standard error that no per-window data supports.
    """
    if channel not in CHANNELS:
        raise KeyError(f"no {channel!r} channel; the channels are {CHANNELS}")
    cell = _cell_name(record)
    rerun = "re-run scripts/diagnose_dynamics.py to regenerate it"
    episode = record.get("windows", {}).get("episode")
    if episode is None:
        raise StaleRecord(
            f"{cell}: the record carries no window -> episode index (windows.episode "
            f"is absent or null), so its windows cannot be clustered; {rerun}"
        )
    block = record.get("interventions", {}).get(rung)
    if block is None:
        raise StaleRecord(f"{cell}: no {rung!r} rung in the record; {rerun} with that rung")
    if "window_steps_changed" not in block or "window_delta_mean" not in block.get(channel, {}):
        raise StaleRecord(
            f"{cell}: rung {rung!r} carries no per-window series (window_steps_changed / "
            f"{channel}.window_delta_mean) -- the record predates them; {rerun}"
        )
    noise = record.get("noise_reference")
    if noise is None:
        raise StaleRecord(f"{cell}: the record carries no noise_reference block; {rerun}")
    embedding = block.get("embedding")
    series = CellSeries(
        arm=record["arm"], seed=int(record["seed"]), rung=rung, channel=channel,
        delta=np.asarray(block[channel]["window_delta_mean"], dtype=float),
        changed=np.asarray(block["window_steps_changed"], dtype=int) > 0,
        episode=np.asarray(episode, dtype=int),
        embedding=None if embedding is None else np.asarray(embedding["window_distance"], dtype=float),
        noise=np.asarray(noise["window_distance"], dtype=float),
        windows_total=int(record["windows"]["total"]),
        val=tuple(record["episodes"]["val"]),
        horizon=int(record["horizon"]), context=int(record["context"]),
        device=str(record["device"]), torch_version=str(record["torch_version"]),
    )
    lengths = {
        "windows.total": series.windows_total, "windows.episode": series.episode.size,
        f"{rung}.window_steps_changed": series.changed.size,
        f"{rung}.{channel}.window_delta_mean": series.delta.size,
        "noise_reference.window_distance": series.noise.size,
    }
    if series.embedding is not None:
        lengths[f"{rung}.embedding.window_distance"] = series.embedding.size
    if len(set(lengths.values())) != 1:
        raise StaleRecord(f"{cell}: rung {rung!r}: per-window series disagree in length: {lengths}")
    return series


def load_cells(out_dir, arms, seeds) -> dict:
    """Every planned (arm, seed) record, keyed by cell -- or a refusal.

    The file name is `diagnostic_<arm>_seed<n>.json`, the diagnose script's
    own convention; a missing file is `MissingCell`, and a record whose own
    `arm`/`seed` disagree with the name it was read under is
    `MislabelledRecord`. Both are raised BEFORE any statistic exists.
    """
    records = {}
    for arm in arms:
        for seed in seeds:
            path = Path(out_dir) / f"diagnostic_{arm}_seed{seed}.json"
            if not path.exists():
                raise MissingCell(
                    f"no diagnostic record for {arm} seed {seed} at {path}; a pool over "
                    "the remaining cells would print under this cell's name"
                )
            record = load_record(path)
            if record.get("arm") != arm or record.get("seed") != seed:
                raise MislabelledRecord(
                    f"{path.name} was read for {arm} seed {seed} but its record says "
                    f"arm={record.get('arm')!r} seed={record.get('seed')!r}"
                )
            records[(arm, seed)] = record
    return records


def require_compatible(cells) -> None:
    """Every cell scores the same windows under the same protocol on the same
    device, or the two that disagree are named with the field.

    The window count, the episode index and the held-out split are what make
    "row w of cell A" and "row w of cell B" the same window; horizon and
    context are what make the deltas the same quantity; and the addendum
    measured the deltas moving across devices by more than their standard
    error, so device and torch version are identity too. Refused, not warned.
    """
    cells = list(cells)
    if not cells:
        raise PoolingError("no cell to pool")
    seen: set = set()
    for cell in cells:
        if (cell.arm, cell.seed) in seen:
            raise IncompatibleCells(
                f"{cell.name} appears more than once; the same cell pooled twice would "
                "count its windows twice under one arm's name"
            )
        seen.add((cell.arm, cell.seed))
    first = cells[0]
    for other in cells[1:]:
        if first.arm != other.arm:
            raise IncompatibleCells(
                f"{first.name} and {other.name} are different arms; a pool over them "
                f"would print under {first.arm!r} alone"
            )
        _require_same_windows(first, other)


def _require_same_windows(first, other) -> None:
    """The window identity and the reading, compared between two cells of
    the same arm or of the two arms of a contrast."""
    # The episode index carries the window count too (`read_series` pins
    # `windows.total` to its length), so comparing the WHOLE index -- never
    # a prefix -- covers both: a separate count check would be a guard
    # nothing could turn red.
    fields = {
        "windows.episode": (first.episode.tolist(), other.episode.tolist()),
        "episodes.val": (first.val, other.val),
        "horizon": (first.horizon, other.horizon),
        "context": (first.context, other.context),
        "device": (first.device, other.device),
        "torch_version": (first.torch_version, other.torch_version),
    }
    for field, (a, b) in fields.items():
        if a != b:
            shown = (
                f" ({len(a)} vs {len(b)} windows)" if field == "windows.episode"
                else f": {a!r} vs {b!r}"
            )
            raise IncompatibleCells(
                f"{first.name} and {other.name} disagree on {field}{shown}; their "
                "windows are not the same windows and cannot share a pooled table"
            )
    if first.rung != other.rung or first.channel != other.channel:
        raise IncompatibleCells(
            f"{first.name} ({first.rung}, {first.channel}) and {other.name} "
            f"({other.rung}, {other.channel}) are not the same reading"
        )


# ---------------------------------------------------------------------------
# The statistics.
# ---------------------------------------------------------------------------


def _z(mean: float, se: float) -> float:
    """`mean / se`, SIGNED -- the between-arm contrast reads the sign -- with
    the diagnose script's policy on a zero ruler: 0 / 0 is 0, x / 0 is +-inf.
    NaN propagates: a clustered ruler that could not be estimated never
    quietly becomes the naive one."""
    if np.isnan(se):
        return float("nan")
    if se == 0.0:
        return 0.0 if mean == 0.0 else float(np.copysign(np.inf, mean))
    return float(mean / se)


@dataclass(frozen=True)
class PooledMean:
    """One arm x rung x channel: the seed-averaged per-window mean and its
    rulers. `se` is the episode-clustered one and the only ruler;
    `se_independent` is the naive standard error over the seed-averaged
    windows, printed under its own name as NOT the ruler."""

    arm: str
    rung: str
    channel: str
    seeds: tuple[int, ...]
    windows: int
    windows_excluded: int
    excluded_windows: tuple[int, ...]
    clusters: int
    series: np.ndarray      # (windows,) the seed-mean delta per kept window
    mean: float
    se: float
    se_independent: float
    z: float


def _seed_mean_windows(cells):
    """The cells, checked compatible, the windows every one of them changed,
    and the seed-mean delta over those -- a window one seed left unchanged
    carries an exact zero there that is not a measurement, so it leaves the
    pool entirely rather than diluting the seed mean toward null."""
    cells = list(cells)
    require_compatible(cells)
    keep = np.logical_and.reduce([cell.changed for cell in cells])
    series = np.mean([cell.delta for cell in cells], axis=0)[keep]
    return cells, keep, series


def pool_arm(cells) -> PooledMean:
    """Average the seeds within each window, mean over windows, cluster by
    episode.

    Every kept window is one row -- 229 on the shipped split -- labelled with
    its EPISODE. Episodes nest windows nest seeds, so that one label set
    absorbs both the shared-window correlation and the seed replication;
    stacking the seeds instead and clustering by episode gives the identical
    mean and clustered standard error (the cluster residual sums are three
    times the seed-mean ones and the row count is three times larger), which
    is why the estimator is described this way and not as 687 rows. The
    naive standard error over the seed-mean rows is kept beside it under its
    own name and is not the ruler; `z` is `mean / se` with `se` the
    clustered one -- NaN below two clusters, never the naive fallback.
    """
    cells, keep, series = _seed_mean_windows(cells)
    labels = cells[0].episode[keep]
    mean = float(series.mean())
    se = cluster_standard_error(series, labels)
    independent = (
        float(series.std(ddof=1) / np.sqrt(series.size)) if series.size > 1 else float("nan")
    )
    return PooledMean(
        arm=cells[0].arm, rung=cells[0].rung, channel=cells[0].channel,
        seeds=tuple(cell.seed for cell in cells),
        windows=int(keep.sum()), windows_excluded=int((~keep).sum()),
        excluded_windows=tuple(int(w) for w in np.flatnonzero(~keep)),
        clusters=int(np.unique(labels).size),
        series=series, mean=mean, se=se, se_independent=independent, z=_z(mean, se),
    )


@dataclass(frozen=True)
class PairedContrast:
    """Treatment minus control, paired per window, clustered by episode."""

    treatment: str
    control: str
    rung: str
    channel: str
    windows: int
    windows_excluded: int
    excluded_windows: tuple[int, ...]
    clusters: int
    series: np.ndarray      # (windows,) the per-window paired difference
    mean: float
    se: float
    z: float


def _paired_windows(treatment, control):
    """The two arms' cells, checked compatible together, and the mask of the
    windows every cell of both arms changed -- a window one seed left
    unchanged carries an exact zero there that is not a measurement, so it
    leaves the pairing entirely."""
    treatment, control = list(treatment), list(control)
    require_compatible(treatment)
    require_compatible(control)
    if treatment[0].arm == control[0].arm:
        raise IncompatibleCells(
            f"the contrast would read {treatment[0].arm!r} against itself; treatment "
            "and control must be two arms"
        )
    # The two arms are compatible with each other on the window identity
    # too -- checked on one cell of each, since each arm is already
    # internally consistent -- and `require_compatible` refuses the arm
    # mismatch itself, so it is called on the identity fields only.
    _require_same_windows(treatment[0], control[0])
    keep = np.logical_and.reduce([cell.changed for cell in treatment + control])
    return treatment, control, keep


def paired_contrast(treatment, control) -> PairedContrast:
    """`mean over seeds within each arm and window, then difference` -- the
    order matters: differencing seed by seed would give three times the rows
    and a ruler over a series that is not the between-arm one.

    The mean of the paired series is pairing-invariant, so what carries the
    pairing is the SERIES itself (returned) and the clustered standard error
    over it; the tests assert those, not the mean.
    """
    treatment, control, keep = _paired_windows(treatment, control)
    seed_mean = lambda cells: np.mean([cell.delta for cell in cells], axis=0)  # noqa: E731
    series = (seed_mean(treatment) - seed_mean(control))[keep]
    labels = treatment[0].episode[keep]
    mean = float(series.mean())
    se = cluster_standard_error(series, labels)
    return PairedContrast(
        treatment=treatment[0].arm, control=control[0].arm,
        rung=treatment[0].rung, channel=treatment[0].channel,
        windows=int(keep.sum()), windows_excluded=int((~keep).sum()),
        excluded_windows=tuple(int(w) for w in np.flatnonzero(~keep)),
        clusters=int(np.unique(labels).size),
        series=series, mean=mean, se=se, z=_z(mean, se),
    )


def _normalised(cells) -> dict:
    """Every cell's numerator and noise rows over ALL its windows, each cell
    divided by ITS OWN noise median over its changed windows -- so a cell's
    rows are in units of its own two-draw distance and cells from
    differently scaled heads pool on one ruler. Refuses a cell never
    measured in embedding space and one whose noise median is exactly 0 --
    x / 0 is undefined, and pooling it would print inf."""
    scaled = {}
    for cell in cells:
        if cell.embedding is None:
            raise StaleRecord(
                f"{cell.name}: rung {cell.rung!r} was not measured in embedding space "
                "(embedding is null); re-run scripts/diagnose_dynamics.py"
            )
        unit = float(np.median(cell.noise[cell.changed]))
        if unit == 0.0:
            raise DegenerateNoise(
                f"{cell.name}: the noise reference's median over the changed windows is "
                "exactly 0, so this cell's ratio is undefined and cannot be pooled"
            )
        scaled[(cell.arm, cell.seed)] = (cell.embedding / unit, cell.noise / unit)
    return scaled


def _ratio_of_medians(numerator: np.ndarray, noise: np.ndarray, name: str) -> float:
    """`median(numerator) / median(noise)` over one index set, refusing a
    replicate whose noise median is 0: a cell can pass the per-cell check
    (median 1.5 over [0, 0, 3, 3]) and still hand an episode resample a zero
    ruler, and a NaN there would reach the percentile as a NaN interval
    beside a finite ratio."""
    ruler = float(np.median(noise))
    if ruler == 0.0:
        raise DegenerateNoise(
            f"{name}: a bootstrap replicate's noise median is exactly 0 -- some episode's "
            "ruler measured no spread -- so its ratio is undefined and the interval "
            "cannot be estimated"
        )
    return float(np.median(numerator) / ruler)


def _episode_bootstrap(labels: np.ndarray, bootstrap: int, seed: int):
    """Yield `bootstrap` index arrays, each the rows of one draw of the
    episodes WITH replacement -- every row of a drawn episode, as many times
    as it was drawn. The ruler for a ratio of medians, which has no sandwich
    standard error: resample the clusters, recompute."""
    rng = default_rng(seed)
    groups = np.unique(labels)
    members = {group: np.flatnonzero(labels == group) for group in groups}
    for _ in range(bootstrap):
        draw = rng.choice(groups, size=groups.size, replace=True)
        yield np.concatenate([members[group] for group in draw])


def _interval(replicates: np.ndarray) -> tuple[float, float, float]:
    low, high = np.percentile(replicates, [2.5, 97.5])
    se = float(replicates.std(ddof=1)) if replicates.size > 1 else float("nan")
    return float(low), float(high), se


@dataclass(frozen=True)
class PooledRatio:
    """One arm x rung: the ratio of pooled medians and its bootstrap interval."""

    arm: str
    rung: str
    seeds: tuple[int, ...]
    rows: int
    clusters: int
    ratio: float
    numerator_median: float
    noise_median: float
    ci_low: float
    ci_high: float
    bootstrap_se: float
    bootstrap: int
    seed: int


def pool_ratio(cells, *, bootstrap: int = 2000, seed: int = 0) -> PooledRatio:
    """Median of every seed's changed-window numerators over the median of
    the matching noise rows, EACH CELL IN UNITS OF ITS OWN NOISE MEDIAN, with
    an episode-cluster bootstrap interval.

    The normalisation is what makes the pooled number scale-free: each
    cell's ratio is invariant to its head's scale, but a median over raw rows
    from three heads is a function of which seed's rows land in the middle
    -- with per-cell ratios of (0.5, 0.5, 2.0), raw pooling reads a
    different number at every set of scales. `numerator_median` and
    `noise_median` are therefore in those units too (the latter is ~1 by
    construction), and `ratio` is their quotient.

    A ratio of medians has no sandwich standard error, so the ruler is a
    resample of the EPISODES -- every row of a drawn episode -- rather than
    of the rows: rows from one episode are not independent, and a row
    bootstrap would narrow the interval by exactly the correlation the
    episode clustering exists to absorb. Reproducible under `seed`.
    """
    cells = list(cells)
    require_compatible(cells)
    scaled = _normalised(cells)
    numerator = np.concatenate([scaled[(c.arm, c.seed)][0][c.changed] for c in cells])
    noise = np.concatenate([scaled[(c.arm, c.seed)][1][c.changed] for c in cells])
    labels = np.concatenate([cell.episode[cell.changed] for cell in cells])
    name = f"{cells[0].arm} seeds {tuple(c.seed for c in cells)}"
    ratio_of = lambda index: _ratio_of_medians(numerator[index], noise[index], name)  # noqa: E731
    everything = np.arange(numerator.size)
    replicates = np.array([ratio_of(index) for index in _episode_bootstrap(labels, bootstrap, seed)])
    low, high, se = _interval(replicates)
    return PooledRatio(
        arm=cells[0].arm, rung=cells[0].rung, seeds=tuple(cell.seed for cell in cells),
        rows=int(numerator.size), clusters=int(np.unique(labels).size),
        ratio=ratio_of(everything),
        numerator_median=float(np.median(numerator)), noise_median=float(np.median(noise)),
        ci_low=low, ci_high=high, bootstrap_se=se, bootstrap=bootstrap, seed=seed,
    )


@dataclass(frozen=True)
class PairedRatioContrast:
    """Treatment ratio minus control ratio, the two arms resampled in step."""

    treatment: str
    control: str
    rung: str
    windows: int
    windows_excluded: int
    clusters: int
    treatment_ratio: float
    control_ratio: float
    contrast: float
    ci_low: float
    ci_high: float
    bootstrap_se: float
    bootstrap: int
    seed: int


def paired_ratio_contrast(treatment, control, *, bootstrap: int = 2000, seed: int = 0) -> PairedRatioContrast:
    """Per window, the seed-mean numerator and noise of each arm -- each
    cell normalised by its own noise median first -- over the windows every
    cell of both arms changed; each arm's ratio is the median over windows
    of the one over the median of the other; the contrast is treatment
    minus control. This is the PROBE-FREE between-arm estimand: no ridge
    probe stands between either arm and the number.

    PAIRED: every bootstrap replicate draws ONE set of episodes and applies it
    to both arms, so on identical arms every replicate is exactly 0 and the
    interval is (0, 0). Independent draws per arm would put a spread on
    identical arms that is not a spread of the contrast.
    """
    treatment, control, keep = _paired_windows(treatment, control)
    scaled = _normalised(treatment + control)
    # Seed-mean of the NORMALISED rows -- each cell in units of its own noise
    # median first, then averaged -- for the same reason `pool_ratio` gives.
    per_arm = lambda cells, which: np.mean(  # noqa: E731
        [scaled[(c.arm, c.seed)][which] for c in cells], axis=0
    )[keep]
    numerator = {"t": per_arm(treatment, 0), "c": per_arm(control, 0)}
    noise = {"t": per_arm(treatment, 1), "c": per_arm(control, 1)}
    labels = treatment[0].episode[keep]
    names = {"t": treatment[0].arm, "c": control[0].arm}
    ratio_of = lambda arm, index: _ratio_of_medians(  # noqa: E731
        numerator[arm][index], noise[arm][index], names[arm]
    )
    everything = np.arange(int(keep.sum()))
    replicates = np.array([
        ratio_of("t", index) - ratio_of("c", index)
        for index in _episode_bootstrap(labels, bootstrap, seed)
    ])
    low, high, se = _interval(replicates)
    return PairedRatioContrast(
        treatment=treatment[0].arm, control=control[0].arm, rung=treatment[0].rung,
        windows=int(keep.sum()), windows_excluded=int((~keep).sum()),
        clusters=int(np.unique(labels).size),
        treatment_ratio=ratio_of("t", everything), control_ratio=ratio_of("c", everything),
        contrast=ratio_of("t", everything) - ratio_of("c", everything),
        ci_low=low, ci_high=high, bootstrap_se=se, bootstrap=bootstrap, seed=seed,
    )


def pool_ladder(
    records: dict, arms, seeds, rungs, treatment: str, control: str,
    channels=CHANNELS, *, bootstrap: int = 2000, seed: int = 0,
) -> dict:
    """Every pooled statistic over the ladder, from records keyed by
    `(arm, seed)`: `arms[(arm, rung, channel)]`, `ratios[(arm, rung)]`,
    `contrasts[(rung, channel)]` and `ratio_contrasts[rung]`. A missing
    planned cell is refused by name before any of them is computed."""
    for role, arm in (("treatment", treatment), ("control", control)):
        if arm not in arms:
            raise MissingCell(
                f"the {role} {arm!r} is not among the arms {tuple(arms)!r}, so no cell of "
                "it was planned and the contrast would be over a cell that was never read"
            )
    for arm in arms:
        for s in seeds:
            if (arm, s) not in records:
                raise MissingCell(f"no record for {arm} seed {s} among the cells handed in")
    read = lambda arm, rung, channel: [  # noqa: E731
        read_series(records[(arm, s)], rung, channel) for s in seeds
    ]
    pooled: dict = {"arms": {}, "ratios": {}, "contrasts": {}, "ratio_contrasts": {}}
    for rung in rungs:
        for arm in arms:
            for channel in channels:
                pooled["arms"][(arm, rung, channel)] = pool_arm(read(arm, rung, channel))
            pooled["ratios"][(arm, rung)] = pool_ratio(
                read(arm, rung, channels[0]), bootstrap=bootstrap, seed=seed,
            )
        for channel in channels:
            pooled["contrasts"][(rung, channel)] = paired_contrast(
                read(treatment, rung, channel), read(control, rung, channel),
            )
        pooled["ratio_contrasts"][rung] = paired_ratio_contrast(
            read(treatment, rung, channels[0]), read(control, rung, channels[0]),
            bootstrap=bootstrap, seed=seed,
        )
    return pooled
