import ast
import inspect
import re

import numpy as np
import pytest

from mbfps.eval import burden
from mbfps.eval import headroom as module
from mbfps.eval.headroom import (
    ARMS_REQUIRED, CONFIDENCE, DECISION_H, DECISION_K,
    DISPLACEMENT_RECORDED_ONLY, IDENTITY_TOLERANCE, NO_MAJORITY, PLACEMENTS,
    READING_COLUMNS, READING_WIDTHS, REPORTED_H, RESAMPLES, SECONDARY_SIGMAS,
    SEEDS_MINIMUM, HeadroomCell, HeadroomInputs, Interval, deficit,
    format_reading_headroom, headroom, reading_headroom, skill,
    triple_residual,
)


def test_the_three_differences_satisfy_the_identity_exactly():
    """skill + deficit == headroom, with `==`, not approx.

    The hold term cancels algebraically, so the only residual is float
    rounding -- and on this fixture there is none. Every subtraction here is
    between operands within a factor of two (Sterbenz's lemma), so `skill`,
    `deficit` and `headroom` are each exact and `skill + deficit` rounds a sum
    that is already representable. The residual is exactly 0.0, and the test
    says so with `==`.

    `pytest.approx(rel=1e-6)` here would be the wrong tool: real data barely
    separates `np.sum` from `np.max` (8 of 9 shipped k=45 cells give 0.0 under
    both, the ninth 1.421e-14 against 5.684e-14), so a tolerance wide enough to
    pass it accepts the `np.sum` mutant inside `triple_residual`, and M3m
    nearly shipped exactly that tolerance against exactly that mutation. (This
    fixture cannot tell `sum` from `max` either -- every step's residual is 0.0
    -- which is what the next test is for.)
    """
    rng = np.random.default_rng(0)
    floor = 100.0 + rng.uniform(0.0, 150.0, size=45)
    rung = floor + rng.uniform(0.0, 5.0, size=45)
    hold = rung + rng.uniform(-2.0, 8.0, size=45)
    # The fixture must exercise what the assertion is about: a spread of floors
    # so the roundings differ per step, a skill of both signs, and the factor-
    # of-two bound that makes every subtraction exact.
    assert np.ptp(floor) > 50.0, "fixture must span a real range of floors"
    assert ((hold - rung) > 0).any() and ((hold - rung) < 0).any()
    assert ((hold - floor) < 0).any(), "headroom must cross zero, as on a real cell"
    assert (hold <= 2.0 * floor).all(), "Sterbenz: every subtraction exact"

    whole = headroom(hold, floor)
    parts = skill(hold, rung) + deficit(rung, floor)
    assert np.array_equal(whole, parts)
    assert triple_residual(hold, rung, floor) == 0.0


def test_the_identity_holds_to_rounding_where_the_subtractions_are_not_exact():
    """Outside the factor-of-two regime the residual is rounding, not zero.

    `burden.IDENTITY_TOLERANCE`'s docstring records this triple (monotone, in
    100-250, ratio 2.23) as one whose decomposition residual is 2**-45, one ulp
    at that magnitude. Here the same arithmetic runs through `headroom`,
    `skill` and `deficit`. The bound is an ABSOLUTE tolerance: three roundings
    of at most 2**-46 each cannot exceed 3 * 2**-46 = 4.3e-14.
    """
    floor = np.array([106.90575077107698])
    rung = np.array([110.35124079639952])
    hold = np.array([238.65922676171652])
    assert hold[0] > 2.0 * floor[0], "this triple is outside the exact regime"

    residual = triple_residual(hold, rung, floor)
    assert residual > 0.0, "the fixture must leave a rounding residual to bound"
    assert residual <= 3 * 2.0**-46
    assert residual < IDENTITY_TOLERANCE


def _fault_skill(monkeypatch, bump):
    """Make `skill` wrong by a known amount at known steps, and prove it took.

    `triple_residual(hold, rung, floor)` has one `floor`, which feeds both
    `headroom` and `deficit`, so no choice of inputs can unbalance the identity
    except through floating-point rounding: a perturbed `floor` cancels. The
    fault therefore goes in through the one place a real plumbing error would,
    one of the three differences returning a wrong number at some step.
    """
    real_skill = module.skill
    monkeypatch.setattr(
        module, "skill", lambda hold, rung: real_skill(hold, rung) + bump
    )


def test_triple_residual_reports_the_worst_step_not_their_sum(monkeypatch):
    """One bad step must not be averaged or summed away.

    `skill` is faulted by 0.5, -0.25 and 0.125 at three steps, all exactly
    representable, so `headroom - (skill + deficit)` is exactly minus the fault
    at each step and the expected value is the largest fault -- read off the
    construction, not off the function.

    It takes MORE THAN ONE bad step to tell `max` from `sum`: with a single one
    the two agree. Here `max` is 0.5, the sum of magnitudes is 0.875 and the
    mean over ten steps is 0.0875, so `==` separates all three. The signs are
    mixed and the largest residual (-0.5) is the negative one, so a `np.max`
    over the signed differences reads 0.25 and is separated too.
    """
    hold, rung, floor = np.full(10, 110.0), np.full(10, 103.0), np.full(10, 100.0)
    assert triple_residual(hold, rung, floor) == 0.0, "the unfaulted base balances"

    bump = np.zeros(10)
    bump[3], bump[1], bump[7] = 0.5, -0.25, 0.125
    assert np.count_nonzero(bump) >= 3, "one bad step cannot tell max from sum"
    _fault_skill(monkeypatch, bump)
    assert module.skill(hold, rung)[3] == 7.5, "the fault must have been applied"

    assert triple_residual(hold, rung, floor) == 0.5


def test_triple_residual_is_a_magnitude_not_a_signed_difference(monkeypatch):
    """A step that is off in either direction counts, and counts the same.

    The only bad step in the first fixture has `headroom - (skill + deficit)`
    NEGATIVE (-0.5), so a `np.max` over the signed differences returns 0.0 and
    only taking the absolute value first returns 0.5. The mirror fixture has
    the opposite sign and must read the same.
    """
    hold, rung, floor = np.full(10, 110.0), np.full(10, 103.0), np.full(10, 100.0)
    over = np.zeros(10)
    over[3] = 0.5

    _fault_skill(monkeypatch, over)
    assert module.skill(hold, rung)[3] == 7.5, "the fault must have been applied"
    assert triple_residual(hold, rung, floor) == 0.5

    monkeypatch.undo()
    _fault_skill(monkeypatch, -over)
    assert module.skill(hold, rung)[3] == 6.5, "the mirror fault must have been applied"
    assert triple_residual(hold, rung, floor) == 0.5


def test_each_difference_has_the_sign_and_the_operands_the_spec_gives():
    """Sign convention and operand order, stated as values rather than prose.

    The three values are distinct (10, 7, 3), so swapping which curve a
    function reads, or the order of its subtraction, changes the answer. The
    identity test does NOT cover a coordinated flip of all three functions:
    with `headroom = floor - hold`, `skill = rung - hold` and
    `deficit = floor - rung` the identity still balances, and every verdict
    would invert. Only a stated value catches that.
    """
    hold, rung, floor = np.array([110.0]), np.array([103.0]), np.array([100.0])

    assert headroom(hold, floor)[0] == 10.0
    assert skill(hold, rung)[0] == 7.0
    assert deficit(rung, floor)[0] == 3.0


def test_deficit_is_burden_on_a_spread_of_values():
    """`deficit`'s docstring says it is IDENTICAL to M3m's `burden(k, h)`.

    Neither function calls the other, deliberately -- `deficit` is named for its
    role on M3n's axis -- so nothing but this test notices if one is edited and
    the other is not. `np.array_equal` is exact: both compute `rung - floor`.

    The fixture must make a mutation to EITHER function move the comparison. A
    constant input would not: with every deficit equal and positive, an `abs`
    changes nothing and neither does reversing or shifting the steps. Here the
    floors span a real range, and the deficits have both signs and no ties, so
    a swap of operands, an `abs`, a reversal, a dropped term and a constant
    offset each give a different array.
    """
    rng = np.random.default_rng(1)
    floor = 100.0 + rng.uniform(0.0, 150.0, size=45)
    rung = floor + rng.uniform(-3.0, 5.0, size=45)

    gap = rung - floor
    assert np.ptp(floor) > 50.0, "fixture must span a real range of floors"
    assert (gap > 0).any() and (gap < 0).any(), "deficits of both signs"
    assert len(np.unique(gap)) == gap.size, "no two steps tie"

    got, reference = deficit(rung, floor), burden.burden(rung, floor)
    assert np.array_equal(got, reference)

    # What the comparison can see: each of these edits would have broken it.
    assert not np.array_equal(got, burden.burden(floor, rung)), "operand order"
    assert not np.array_equal(got, np.abs(reference)), "sign"
    assert not np.array_equal(got, reference + 1e-3), "offset"
    assert not np.array_equal(got, reference[::-1]), "step order"
    assert not np.array_equal(got, rung), "dropped term"


def test_each_difference_is_signed_not_a_magnitude():
    """A negative value is returned as negative.

    `headroom` crosses zero inside the reported grid on a real cell
    (`pixel_ae_seed1`: 0.575649 at h=1, -0.641206 at h=2), and the two-sided
    reading is `skill > 0` and `deficit > 0`, so a difference that returned its
    magnitude would turn a model that LOSES to copying into one that wins. The
    three values below are distinct, and each is the negative of a case above.
    """
    assert headroom(np.array([99.0]), np.array([100.0]))[0] == -1.0
    assert skill(np.array([100.0]), np.array([103.0]))[0] == -3.0
    assert deficit(np.array([97.0]), np.array([100.0]))[0] == -3.0


@pytest.mark.parametrize("difference", [headroom, skill, deficit])
def test_mismatched_curve_lengths_are_refused_by_the_shared_guard(difference):
    """A length mismatch must raise from `checked_pair`, not broadcast.

    numpy would broadcast a length-1 curve against a length-45 one and return
    45 plausible numbers, in either argument order. A 45-against-44 pair makes
    numpy itself raise `ValueError` ("operands could not be broadcast"), so
    that case proves nothing about the guard; `match="same length"` pins the
    message to `checked_pair`'s, which numpy never produces.
    """
    for left, right in (
        (np.ones(45), np.ones(1)),
        (np.ones(1), np.ones(45)),
        (np.ones(45), np.ones(44)),
    ):
        with pytest.raises(ValueError, match="same length"):
            difference(left, right)


def test_triple_residual_refuses_a_mismatched_curve():
    """The residual is built from the three differences and inherits their guard."""
    with pytest.raises(ValueError, match="same length"):
        triple_residual(np.ones(45), np.ones(45), np.ones(1))


@pytest.mark.parametrize("difference", [headroom, skill, deficit])
def test_a_non_finite_curve_is_refused(difference):
    """A NaN would otherwise surface as a NaN difference with no named cause.

    `checked_pair` is the only place the finiteness check lives; a difference
    that bypassed it would return NaN where it should raise.
    """
    for bad in (np.nan, np.inf, -np.inf):
        poisoned = np.ones(5)
        poisoned[2] = bad
        with pytest.raises(ValueError, match="finite"):
            difference(poisoned, np.ones(5))
        with pytest.raises(ValueError, match="finite"):
            difference(np.ones(5), poisoned)


def _references(source: str, name: str) -> list[ast.AST]:
    """Every node in `source` that refers to `name`, in any context.

    A bare `Name` (load, store or delete), a `module.name` attribute access, an
    `import ... as name` alias, and a string constant equal to `name` (which is
    how `globals()["name"]` and `getattr(module, "name")` would reach it).
    """
    found: list[ast.AST] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and node.id == name:
            found.append(node)
        elif isinstance(node, ast.Attribute) and node.attr == name:
            found.append(node)
        elif isinstance(node, ast.alias) and name in (node.name, node.asname):
            found.append(node)
        elif isinstance(node, ast.Constant) and node.value == name:
            found.append(node)
    return found


def _float_literals(source: str, value: float) -> list[ast.Constant]:
    """Every float literal in `source` equal to `value`, wherever it sits.

    `_references` finds a NAME. Writing the number out in place of the name
    (`hold - floor - 0.0 * 3.9694722203504225`) reaches the same quantity
    without ever mentioning the name, so a count of name references alone is
    blind to it. Only `float` constants count: the digits that appear inside a
    docstring are a `str` constant and are not arithmetic.
    """
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and node.value == value
    ]


def test_the_reference_finder_sees_a_use_inside_a_function_body():
    """Positive control for the check below, so that it cannot pass vacuously.

    The recorded-only test asserts a COUNT of references. A finder that walked
    nothing would report zero uses of anything and pass; these sources are the
    cases it must flag, so the zero it reports on the real module is known to
    mean something.
    """
    defined_only = "X: float = 1.5\n\ndef f(a):\n    return a\n"
    used_in_arithmetic = "X: float = 1.5\n\ndef f(a):\n    return a * X\n"
    used_as_attribute = "X: float = 1.5\n\ndef f(a, m):\n    return a * m.X\n"
    used_by_string = "X: float = 1.5\n\ndef f(a):\n    return a * globals()['X']\n"

    assert len(_references(defined_only, "X")) == 1
    assert len(_references(used_in_arithmetic, "X")) == 2
    assert len(_references(used_as_attribute, "X")) == 2
    assert len(_references(used_by_string, "X")) == 2
    used = [n for n in _references(used_in_arithmetic, "X")
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
    assert len(used) == 1, "a use inside a function body must be a Load"


def test_the_literal_finder_sees_a_bare_float_in_arithmetic():
    """Positive control for the literal count below, so it cannot pass vacuously.

    Each source writes the number out instead of naming a constant. The finder
    must flag the use however it is spelled, and must NOT flag the same digits
    sitting in a docstring, which is how the real module mentions the value.
    """
    defined_only = "X: float = 1.5\n\ndef f(a):\n    return a\n"
    used_in_arithmetic = "X: float = 1.5\n\ndef f(a):\n    return a - 0.0 * 1.5\n"
    used_negated = "X: float = 1.5\n\ndef f(a):\n    return a + -1.5\n"
    used_in_exponent_form = "X: float = 1.5\n\ndef f(a):\n    return a * 15e-1\n"
    only_in_a_docstring = 'X: float = 1.5\n\ndef f(a):\n    """1.5"""\n    return a\n'

    assert len(_float_literals(defined_only, 1.5)) == 1
    assert len(_float_literals(used_in_arithmetic, 1.5)) == 2
    assert len(_float_literals(used_negated, 1.5)) == 2
    assert len(_float_literals(used_in_exponent_form, 1.5)) == 2
    assert len(_float_literals(only_in_a_docstring, 1.5)) == 1


def test_the_displacement_constant_is_recorded_and_unused():
    """3.9694722203504225 appears in no arithmetic in this module.

    It is the ground-truth quantity M3m's design mistook for a probe-space
    scale, kept so a reader of the record can see it plays no part. An unused
    constant cannot be caught by a value, so the guard reads the module's
    syntax tree. Two things must both be true: the ONLY reference to the NAME
    `DISPLACEMENT_RECORDED_ONLY` anywhere in the file is the assignment that
    defines it, and the ONLY float literal equal to 3.9694722203504225 is that
    assignment's right-hand side. The second is not redundant with the first:
    `hold - floor - 0.0 * 3.9694722203504225` uses the quantity without ever
    naming the constant, and passed the name check alone.

    The guard is a count over `ast` nodes, not a slice of the source text. A
    text slice comes back empty if the surrounding docstring is edited, and an
    assertion that nothing appears in an empty slice passes whatever the module
    does.
    """
    name = "DISPLACEMENT_RECORDED_ONLY"
    tree = ast.parse(inspect.getsource(module))

    functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert {"headroom", "skill", "deficit", "triple_residual"} <= set(functions), (
        "the walk must reach the function bodies it is meant to police"
    )

    references = _references(inspect.getsource(module), name)
    assert len(references) == 1, (
        f"expected exactly the defining assignment; found {len(references)}"
    )
    (definition,) = references
    assert isinstance(definition, ast.Name)
    assert isinstance(definition.ctx, ast.Store)
    assert DISPLACEMENT_RECORDED_ONLY == 3.9694722203504225

    literals = _float_literals(inspect.getsource(module), 3.9694722203504225)
    assert len(literals) == 1, (
        f"expected exactly the defining literal; found {len(literals)} "
        f"at lines {[n.lineno for n in literals]}"
    )
    (assignment,) = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == name
    ]
    assert (literals[0].lineno, literals[0].col_offset) == (
        assignment.value.lineno, assignment.value.col_offset
    ), "the one literal must be the one in the defining assignment"


def test_reported_h_and_the_decision_cell_are_the_spec_values():
    """Pinned literally. These select which numbers the milestone publishes."""
    assert REPORTED_H == (1, 2, 3, 5, 8, 10, 15, 20, 30, 45)
    assert (DECISION_K, DECISION_H) == (1, 1)


def test_the_remaining_constants_are_the_spec_values():
    """Pinned literally, so an edit to one is a visible change to the protocol.

    `REPORTED_H` is also compared with M3m's, because the module's own
    docstring says the two tables are read on the same grid.
    """
    assert IDENTITY_TOLERANCE == 1e-9
    assert CONFIDENCE == 0.95
    assert RESAMPLES == 2000
    assert SECONDARY_SIGMAS == 2
    assert SEEDS_MINIMUM == 3
    assert ARMS_REQUIRED == 2
    assert REPORTED_H == burden.REPORTED_H


# ---------------------------------------------------------------------------
# The reading: the gate, the placement, the majority, the formatter.
# ---------------------------------------------------------------------------

_ARMS = ("frozen_ssl", "pixel_ae", "random_vit")

# One (headroom, skill, deficit) triple per placement, each as (point, ci_low,
# ci_high). Each row sets only the fields that decide it. The two "straddling"
# intervals have a POSITIVE POINT and a negative lower bound, so a reading that
# looked at the point instead of the interval would put them on the wrong side.
_TRIPLES = {
    "BETWEEN":    ((5.0, 3.0, 7.0), (2.0, 1.0, 3.0), (3.0, 1.5, 4.5)),
    "AT_PERFECT": ((5.0, 3.0, 7.0), (2.0, 1.0, 3.0), (0.1, -0.9, 1.1)),
    "AT_COPYING": ((5.0, 3.0, 7.0), (0.1, -0.9, 1.1), (3.0, 1.5, 4.5)),
    "AMBIGUOUS":  ((5.0, 3.0, 7.0), (0.1, -0.9, 1.1), (0.1, -0.9, 1.1)),
    # Skill AND deficit both resolvably positive: without the gate this reads
    # BETWEEN, so the gate is the only thing standing between it and a verdict.
    "UNREADABLE": ((0.4, -0.9, 1.7), (2.0, 1.0, 3.0), (3.0, 1.5, 4.5)),
}


def _cell(arm, seed, hd, sk, df, clusters=24):
    """A cell, after checking that each interval is a well-formed interval.

    A fixture whose `ci_low` exceeds its point or its `ci_high` would make every
    assertion below about something other than what it names.
    """
    for name, (point, low, high) in (("headroom", hd), ("skill", sk), ("deficit", df)):
        assert low <= point <= high and low < high, f"malformed {name} fixture"
    assert clusters >= 3, "an interval over fewer than three episodes is not one"
    return HeadroomCell(
        arm=arm, seed=seed,
        headroom=Interval(*hd), skill=Interval(*sk), deficit=Interval(*df),
        clusters=clusters,
    )


def _inputs(cells, k=DECISION_K, h=DECISION_H):
    return HeadroomInputs(
        cells={(c.arm, c.seed): c for c in cells},
        decision_k=k, decision_h=h, ks=(1, 3, 5, 15, 45),
    )


def _layout(seeds_by_arm, names):
    """One cell per (arm, seed), in sorted order, placed as `names` says.

    The fixture is checked against `names` before anything is asserted about a
    reading: a layout that did not produce the placements it claims would make
    the majority assertions about some other tally.
    """
    pairs = [(arm, seed) for arm, seeds in seeds_by_arm for seed in seeds]
    assert len(pairs) == len(names), "one placement name per cell"
    cells = [_cell(arm, seed, *_TRIPLES[name]) for (arm, seed), name in zip(pairs, names)]
    assert [c.placement() for c in cells] == list(names), "fixture must place as named"
    return cells


def _nine(names):
    return _layout([(arm, (0, 1, 2)) for arm in _ARMS], names)


def _eleven(names):
    return _layout(
        [("frozen_ssl", (0, 1, 2)), ("pixel_ae", (0, 1, 2, 3)), ("random_vit", (0, 1, 2, 3))],
        names,
    )


def test_the_placements_are_the_five_the_spec_names():
    """Pinned literally, with the no-majority sentinel that is not one of them."""
    assert PLACEMENTS == (
        "BETWEEN", "AT_PERFECT", "AT_COPYING", "AMBIGUOUS", "UNREADABLE",
    )
    assert NO_MAJORITY == "NO_MAJORITY"
    assert NO_MAJORITY not in PLACEMENTS


@pytest.mark.parametrize("skill_and_deficit", [
    # skill resolvably positive, deficit not: without the gate, AT_PERFECT
    ((0.3, 0.1, 0.5), (0.1, -0.2, 0.4)),
    # both resolvably positive: without the gate, BETWEEN
    ((2.0, 1.0, 3.0), (3.0, 1.5, 4.5)),
])
def test_the_gate_reads_unreadable_before_looking_at_skill_or_deficit(skill_and_deficit):
    """A cell whose headroom straddles zero is UNREADABLE whatever the rest say.

    The gate must run FIRST. Both rows have a resolvably positive skill, so
    without the gate the first reads AT_PERFECT and the second BETWEEN -- a
    verdict about a model, taken on a cell where a perfect predictor is
    indistinguishable from a copying one. The headroom has a positive point
    (0.4) and a negative lower bound, so the gate has to be reading the
    interval and not the point.
    """
    sk, df = skill_and_deficit
    cell = _cell("a", 0, (0.4, -0.9, 1.7), sk, df)
    assert cell.headroom.point > 0.0 > cell.headroom.ci_low
    assert cell.skill.ci_low > 0.0, "the fixture must have a skill the gate overrides"
    assert cell.placement() == "UNREADABLE"


def test_a_headroom_below_zero_is_unreadable_too():
    """The shape `pixel_ae_seed1` has at h=2: a negative point estimate.

    Its `headroom(k=45, h=2)` is -0.641206 on the shipped record, so the gate
    refuses it on the point alone. The skill beside it is resolvably POSITIVE
    here, as a gate-less reading would have needed it to be.
    """
    cell = _cell("pixel_ae", 1, (-0.641206, -1.6, -0.1), (2.0, 1.0, 3.0), (3.0, 1.5, 4.5))
    assert cell.headroom.ci_high < 0.0
    assert cell.placement() == "UNREADABLE"


def test_the_four_placements_are_each_reachable_and_distinct():
    """Each row sets only what it needs to, and the four verdicts differ.

    Built as a table so no two rows share a discriminating field by accident.
    The premise is checked from the numbers, not from the code under test: the
    four rows must cover all four (skill, deficit) sign patterns on the lower
    bound, and the two rows that rest on a straddling interval must carry a
    positive POINT, so a reading of the point would mislabel them.
    """
    rows = {name: _TRIPLES[name] for name in PLACEMENTS if name != "UNREADABLE"}
    patterns = {(sk[1] > 0.0, df[1] > 0.0) for _, sk, df in rows.values()}
    assert patterns == {(True, True), (True, False), (False, True), (False, False)}
    assert all(hd[1] > 0.0 for hd, _, _ in rows.values()), "every row passes the gate"
    assert rows["AT_PERFECT"][2][0] > 0.0 > rows["AT_PERFECT"][2][1]
    assert rows["AT_COPYING"][1][0] > 0.0 > rows["AT_COPYING"][1][1]

    seen = {name: _cell("a", 0, *triple).placement() for name, triple in rows.items()}
    assert seen == {name: name for name in rows}
    assert len(set(seen.values())) == 4


def test_an_interval_touching_zero_does_not_exclude_it():
    """`ci_low == 0.0` is not resolvably positive, and `ci_low > 0.0` is.

    Neither the table above nor the gate test has a lower bound of exactly 0.0,
    so a `>=` in place of the `>` would pass both. Here the bound is exactly
    0.0 on each of the three intervals in turn, and the placement follows.
    """
    assert not Interval(0.5, 0.0, 1.0).resolvably_positive()
    assert Interval(0.5, 1e-300, 1.0).resolvably_positive()
    assert not Interval(-0.5, -1.0, -0.1).resolvably_positive()

    touching = (0.5, 0.0, 1.0)
    hd, sk, df = _TRIPLES["BETWEEN"]
    assert _cell("a", 0, hd, sk, df).placement() == "BETWEEN"
    assert _cell("a", 0, touching, sk, df).placement() == "UNREADABLE"
    assert _cell("a", 0, hd, touching, df).placement() == "AT_COPYING"
    assert _cell("a", 0, hd, sk, touching).placement() == "AT_PERFECT"


def test_a_strict_majority_at_nine_cells_is_five():
    """Five of nine carries; four of nine does not, even with a larger loser.

    The four-cell case spreads the other five cells over two placements, so no
    placement reaches the bar: with the other five all one placement, THAT
    placement would hold five of nine and carry the verdict.
    """
    carried = reading_headroom(_inputs(_nine(["BETWEEN"] * 5 + ["AMBIGUOUS"] * 4)))
    assert carried.verdict == "BETWEEN"
    assert carried.tally["BETWEEN"] == (
        ("frozen_ssl", 0), ("frozen_ssl", 1), ("frozen_ssl", 2),
        ("pixel_ae", 0), ("pixel_ae", 1),
    )
    assert carried.tally["AMBIGUOUS"] == (
        ("pixel_ae", 2), ("random_vit", 0), ("random_vit", 1), ("random_vit", 2),
    )
    assert set(carried.tally) == set(PLACEMENTS), "every placement is a key"
    assert carried.tally["AT_COPYING"] == (), "an empty placement is still a key"
    assert "BETWEEN in 5 of 9 cells" in carried.rule
    assert "strict majority 5" in carried.rule

    names = ["BETWEEN"] * 4 + ["AMBIGUOUS"] * 3 + ["AT_PERFECT"] * 2
    missed = reading_headroom(_inputs(_nine(names)))
    assert missed.verdict == "NO_MAJORITY"
    assert [len(missed.tally[n]) for n in PLACEMENTS] == [4, 2, 0, 3, 0]
    assert "no placement reached 5 of 9 cells" in missed.rule


def test_five_of_eleven_cells_is_not_a_majority():
    """FIVE of eleven does not carry. This is the case a stored `5` fails by value.

    At nine cells the bar is 5, so a stored `5` and `strict_majority(len(...))`
    agree on every nine-cell fixture. At eleven the bar is 6, and five cells of
    one placement is a minority that a stored 5 would call a majority: it would
    return `BETWEEN` here where the computed bar returns `NO_MAJORITY`. The
    other six cells are split three and three, so nothing else carries and the
    verdict is read off this one placement's count.
    """
    assert burden.strict_majority(9) == 5 and burden.strict_majority(11) == 6

    names = ["BETWEEN"] * 5 + ["AMBIGUOUS"] * 3 + ["AT_COPYING"] * 3
    five = reading_headroom(_inputs(_eleven(names)))
    assert [len(five.tally[n]) for n in PLACEMENTS] == [5, 0, 3, 3, 0]
    assert five.verdict == "NO_MAJORITY"
    assert "no placement reached 6 of 11 cells" in five.rule


def test_six_of_eleven_cells_is_a_majority():
    """Six of eleven carries, and the five that do not are not counted in.

    Alone this case does not separate a stored `5` from the computed bar (6 >= 5
    both ways); a stored 5 fails it only INCIDENTALLY, because both placements
    clear 5 and the partition assertion fires. The five-cell case above is the
    one that fails by value. This one is what stops the bar being set too HIGH:
    a stored 7 reads six of eleven as no majority.
    """
    six = reading_headroom(_inputs(_eleven(["BETWEEN"] * 6 + ["AMBIGUOUS"] * 5)))
    assert six.verdict == "BETWEEN"
    assert len(six.tally["BETWEEN"]) == 6
    assert "BETWEEN in 6 of 11 cells" in six.rule
    assert "strict majority 6" in six.rule


@pytest.mark.parametrize("winner", PLACEMENTS)
def test_every_placement_can_carry_the_majority_not_only_the_first(winner):
    """Five of nine carries whichever placement it is, UNREADABLE included.

    A majority of UNREADABLE is a verdict -- M3n's exit 47 -- and not a missing
    value, so it has to be able to win.
    """
    other = "AMBIGUOUS" if winner == "BETWEEN" else "BETWEEN"
    reading = reading_headroom(_inputs(_nine([winner] * 5 + [other] * 4)))
    assert reading.verdict == winner
    assert len(reading.tally[winner]) == 5
    assert reading.rule.startswith(f"{winner} in 5 of 9 cells")


def test_two_winners_is_a_broken_partition_not_a_tie(monkeypatch):
    """Lowering the bar until two placements clear it raises, it does not pick.

    Two disjoint sets cannot each exceed half of the cells, so a strict-majority
    bar never produces two winners and no tie branch exists. The `assert` is
    about a partition that has stopped being one. It is reachable only by
    changing the bar, which is what this does, so the failure is shown to be a
    raised AssertionError and not a silently chosen first winner.
    """
    cells = _nine(["BETWEEN"] * 5 + ["AMBIGUOUS"] * 4)
    assert reading_headroom(_inputs(cells)).verdict == "BETWEEN", "unpatched baseline"

    monkeypatch.setattr(module, "strict_majority", lambda n: 1)
    with pytest.raises(AssertionError, match="partition is broken"):
        reading_headroom(_inputs(cells))


@pytest.mark.parametrize("seeds", [1, 2])
def test_an_arm_short_of_seeds_minimum_is_refused_by_name(seeds):
    """One lucky cell must not establish an arm, and two is still too few.

    `strict_majority(1) == 1`. This is the M3j trap: `--arms random_vit` printed
    a row reading `clears = up` beside a verdict of NO DIFFERENCE. Every cell
    here would read BETWEEN, so without the refusal a verdict comes back; the
    refusal names the short arm and its seeds and not the full ones. One and two
    seeds are both tried because the bound is `SEEDS_MINIMUM` and not a smaller
    number.
    """
    assert SEEDS_MINIMUM == 3
    pairs = [("frozen_ssl", s) for s in (0, 1, 2)]
    pairs += [("pixel_ae", s) for s in range(seeds)]
    pairs += [("random_vit", s) for s in (0, 1, 2)]
    cells = [_cell(a, s, *_TRIPLES["BETWEEN"]) for a, s in pairs]
    with pytest.raises(ValueError, match="SEEDS_MINIMUM") as raised:
        reading_headroom(_inputs(cells))
    message = str(raised.value)
    assert str({"pixel_ae": list(range(seeds))}) in message
    assert "frozen_ssl" not in message and "random_vit" not in message


def test_no_cells_is_refused():
    with pytest.raises(ValueError, match="at least one cell"):
        reading_headroom(_inputs([]))


@pytest.mark.parametrize("seeds", [1, 3, 9])
def test_one_arm_is_refused_by_name_whatever_its_seed_count(seeds):
    """A verdict from one arm is the M3j trap, and three seeds does not escape it.

    At three and nine seeds the arm clears `SEEDS_MINIMUM`, so the seeds refusal
    does not fire and only the arm count can refuse. Every cell reads BETWEEN,
    so without the refusal a verdict comes back -- `BETWEEN in 3 of 3 cells
    (strict majority 2)` at three seeds. That is what the previous milestone
    printed for `--arms random_vit`. Nine seeds is tried too: the refusal is
    about how many arms there are, not about how few cells. At one seed BOTH
    refusals apply, and the arm count is the one reported (it is checked first),
    so a lone short arm is named as the missing arm rather than as a short one.
    """
    assert ARMS_REQUIRED == 2
    cells = [_cell("random_vit", s, *_TRIPLES["BETWEEN"]) for s in range(seeds)]
    with pytest.raises(ValueError, match="ARMS_REQUIRED") as raised:
        reading_headroom(_inputs(cells))
    message = str(raised.value)
    assert "ARMS_REQUIRED=2" in message
    assert str(["random_vit"]) in message, "the refusal names the arm that is present"
    assert "SEEDS_MINIMUM" not in message, "the seeds refusal did not fire"


def test_exactly_arms_required_arms_is_enough():
    """Two arms of three seeds each read; the bound is `<`, not `<=`.

    Every other test with three arms would pass a refusal at `<=`, so this is
    the one that pins the boundary. Six cells, bar 4, all BETWEEN.
    """
    cells = [
        _cell(arm, s, *_TRIPLES["BETWEEN"])
        for arm in ("frozen_ssl", "pixel_ae") for s in (0, 1, 2)
    ]
    reading = reading_headroom(_inputs(cells))
    assert reading.verdict == "BETWEEN"
    assert "BETWEEN in 6 of 6 cells" in reading.rule


def _unbalanced(winner_arm_names, other_names):
    """Arms of 3, 3 and 9 seeds: `frozen_ssl` and `pixel_ae` small, `random_vit` large."""
    return _layout(
        [("frozen_ssl", (0, 1, 2)), ("pixel_ae", (0, 1, 2)), ("random_vit", tuple(range(9)))],
        list(other_names) + list(winner_arm_names),
    )


@pytest.mark.parametrize("winner", PLACEMENTS)
def test_a_majority_carried_by_one_arm_alone_is_no_majority_not_an_error(winner):
    """The unbalanced plan the arm-count check passes, and what the second returns.

    Arms of 3, 3 and 9 seeds: three arms, every one at `SEEDS_MINIMUM` or above,
    so neither refusal fires. The nine-seed arm reads `winner` in all nine
    cells and the six others read AMBIGUOUS (BETWEEN when `winner` is
    AMBIGUOUS), so `winner` holds 9 of 15 cells against a bar of 8 -- a numeric
    majority from one arm.

    The result is a STATUS and not a `ValueError`. The inputs were legal and
    the arms simply did not agree, which is a finding about the data and
    reaches exit 48 as `NO_MAJORITY` does; an exception would read as a
    malformed plan. The rule says the placement held a numeric majority and
    spanned one arm, so it cannot be mistaken for `no placement reached` -- the
    other way of getting here. The tally keeps the nine votes: it is what was
    measured. Every placement is tried, UNREADABLE too, because a verdict of
    `UNREADABLE` from one arm is as much one arm's finding as the others.
    """
    other = "AMBIGUOUS" if winner != "AMBIGUOUS" else "BETWEEN"
    cells = _unbalanced([winner] * 9, [other] * 6)
    assert burden.strict_majority(15) == 8
    reading = reading_headroom(_inputs(cells))
    assert reading.verdict == "NO_MAJORITY"
    assert len(reading.tally[winner]) == 9, "the tally is what was measured"
    assert {arm for arm, _ in reading.tally[winner]} == {"random_vit"}
    assert "no placement reached" not in reading.rule
    assert f"{winner} held a numeric majority (9 of 15 cells" in reading.rule
    assert "spanned only 1 arm (random_vit)" in reading.rule
    assert "ARMS_REQUIRED=2" in reading.rule


def test_the_arm_check_counts_arms_not_cells_per_arm():
    """Eight cells of the nine-seed arm and one of another span two arms: a verdict.

    Pins what `reading_headroom`'s docstring says about check 2, so the claim
    cannot drift from the behaviour. It is the check as specified -- the winning
    cells must span `ARMS_REQUIRED` arms -- and so it reads one cell from a
    second arm as spanning it. That is a limit of the rule rather than an
    endorsement: nine of fifteen cells is a verdict here although the nine-seed
    arm reads BETWEEN in eight of its nine alone. A rule that asked more of the
    second arm would fail this test, deliberately.
    """
    cells = _unbalanced(["BETWEEN"] * 8 + ["AMBIGUOUS"], ["AMBIGUOUS"] * 5 + ["BETWEEN"])
    reading = reading_headroom(_inputs(cells))
    assert len(reading.tally["BETWEEN"]) == 9
    assert {arm for arm, _ in reading.tally["BETWEEN"]} == {"pixel_ae", "random_vit"}
    assert reading.verdict == "BETWEEN"


def test_a_majority_spanning_two_arms_is_a_verdict():
    """Five of nine over two arms carries; the second check is `< ARMS_REQUIRED`.

    The five winning cells are three of `frozen_ssl` and two of `pixel_ae`, so
    they span exactly `ARMS_REQUIRED` arms. A check written as `<=` would refuse
    this, and the one-arm cases above would not notice.
    """
    reading = reading_headroom(_inputs(_nine(["BETWEEN"] * 5 + ["AMBIGUOUS"] * 4)))
    assert {arm for arm, _ in reading.tally["BETWEEN"]} == {"frozen_ssl", "pixel_ae"}
    assert reading.verdict == "BETWEEN"


def test_the_tally_and_the_table_do_not_depend_on_the_order_cells_were_built_in():
    """Votes are listed sorted by (arm, seed) however the dict was filled.

    Built from the reversed list, so the dict's insertion order is the
    OPPOSITE of sorted; the premise is asserted. The expected tuples are
    written out from the layout, not read from either reading.
    """
    cells = _nine(["BETWEEN"] * 5 + ["AMBIGUOUS"] * 4)
    forward, backward = _inputs(cells), _inputs(list(reversed(cells)))
    assert list(backward.cells) != sorted(backward.cells), "insertion order must differ"

    reading = reading_headroom(backward)
    assert reading.tally["BETWEEN"] == (
        ("frozen_ssl", 0), ("frozen_ssl", 1), ("frozen_ssl", 2),
        ("pixel_ae", 0), ("pixel_ae", 1),
    )
    assert reading == reading_headroom(forward)
    assert format_reading_headroom(reading, backward) == format_reading_headroom(
        reading_headroom(forward), forward
    )


def _offsets():
    out, at = [], 0
    for width in READING_WIDTHS:
        out.append(at)
        at += width
    return out


def _column(row, index):
    start = _offsets()[index]
    return row[start:start + READING_WIDTHS[index]]


def _body(text):
    """Cell label -> its row, for every table row in `text`."""
    rows = {}
    for line in text.splitlines():
        label = line[:READING_WIDTHS[0]].strip()
        if label.startswith(_ARMS) and "_seed" in label:
            rows[label] = line
    return rows


def _floats(column):
    return [float(x) for x in re.findall(r"[+-]\d+\.\d{3}", column)]


def test_the_formatter_prints_one_aligned_row_per_cell_and_names_the_decision_cell():
    """Row count, alignment, the (k, h) the table is read at, and what is where.

    `str(DECISION_H) in text` would be true of any table containing "1", so
    the assertions are on the formatted label, and the label is also checked
    at a (k, h) that is not (1, 1) so that a hard-coded one cannot pass.

    Column alignment is checked per column and not only by total width: an
    interval wider than its column would push every later column to the right
    and still satisfy a check on the header alone. One cell carries the widest
    values `{:+8.3f}` prints, and every interval column must still end in a
    space, so the columns are separated and not merely adjacent.
    """
    widest = (
        "random_vit", 2, (500.0, 0.001, 999.999), (-500.0, -999.999, -123.456),
        (-999.999, -999.999, 999.999),
    )
    cells = []
    for i, (arm, seed) in enumerate((a, s) for a in _ARMS for s in (0, 1, 2)):
        if (arm, seed) == widest[:2]:
            cells.append(_cell(*widest[:2], *widest[2:], clusters=3))
            continue
        cells.append(_cell(
            arm, seed,
            (10.0 + i, 5.0 + i, 15.0 + i), (2.0 + i, 1.0 + i, 3.0 + i),
            (3.0 + i, 1.5 + i, 4.5 + i), clusters=24 - i,
        ))
    for k, h in ((DECISION_K, DECISION_H), (5, 3)):
        inputs = _inputs(cells, k=k, h=h)
        reading = reading_headroom(inputs)
        text = format_reading_headroom(reading, inputs)
        assert f"k = {k}, h = {h}" in text
        assert f"k = {k}, h = {h}" in reading.rule
        if (k, h) != (DECISION_K, DECISION_H):
            assert f"k = {DECISION_K}, h = {DECISION_H}" not in text
    assert text.endswith("\n")

    header = next(l for l in text.splitlines() if l.startswith(READING_COLUMNS[0]))
    assert len(header) == sum(READING_WIDTHS)
    for i, name in enumerate(READING_COLUMNS):
        assert header[_offsets()[i]:].startswith(name)

    rows = _body(text)
    assert sorted(rows) == [f"{a}_seed{s}" for a in _ARMS for s in (0, 1, 2)]
    assert len(rows) == 9
    assert all(len(row) == len(header) for row in rows.values())
    for index in (2, 3, 4):
        assert all(_column(row, index)[-1] == " " for row in rows.values()), (
            f"column {READING_COLUMNS[index]} must leave a gap"
        )

    for i, (arm, seed) in enumerate((a, s) for a in _ARMS for s in (0, 1, 2)):
        row = rows[f"{arm}_seed{seed}"]
        cell = _inputs(cells).cells[(arm, seed)]
        assert _column(row, 1).strip() == str(cell.clusters)
        for index, interval in ((2, cell.headroom), (3, cell.skill), (4, cell.deficit)):
            assert _floats(_column(row, index)) == [
                interval.point, interval.ci_low, interval.ci_high
            ], f"{arm}_seed{seed} {READING_COLUMNS[index]}"
        # The widest cell has a resolvably NEGATIVE skill and a straddling
        # deficit, so by hand it is AMBIGUOUS; the other eight are BETWEEN.
        expected = "AMBIGUOUS" if (arm, seed) == widest[:2] else "BETWEEN"
        assert _column(row, 6).strip() == expected, f"{arm}_seed{seed}"
    assert len({_column(r, 1).strip() for r in rows.values()}) == 9, "clusters all differ"


def test_the_share_is_skill_over_headroom_where_the_gate_passed_and_dashes_elsewhere():
    """The share column, row by row, against shares worked out by hand.

    Three cells fail the gate and each would print a number if the column did
    not check it: 0.4 against a skill of 2.0 reads 500.0%, the shape
    `pixel_ae_seed1` has at h=2 (headroom -0.641206 beside a skill of -2.1768)
    reads 339.5%, and a lower bound of exactly 0.0 beside a half-sized skill
    reads 50.0%. The six that pass print 40.0, 75.0, 2.0, -20.0, 40.0 and 2.0:
    distinct enough that swapping numerator and denominator (250.0), dropping
    the 100 (0.4), or dividing by a bound instead of the point (66.7) each
    changes a printed digit. Expected strings are written out; none is
    computed from the formatter.
    """
    table = [
        # arm, seed, headroom, skill, deficit, placement, share
        ("frozen_ssl", 0, (5.0, 3.0, 7.0), (2.0, 1.0, 3.0), (3.0, 1.5, 4.5), "BETWEEN", "40.0%"),
        ("frozen_ssl", 1, (8.0, 6.0, 10.0), (6.0, 4.0, 8.0), (2.0, 1.0, 3.0), "BETWEEN", "75.0%"),
        ("frozen_ssl", 2, (5.0, 3.0, 7.0), (0.1, -0.9, 1.1), (0.1, -0.9, 1.1), "AMBIGUOUS", "2.0%"),
        ("pixel_ae", 0, (5.0, 3.0, 7.0), (-1.0, -2.0, 0.5), (3.0, 1.5, 4.5), "AT_COPYING", "-20.0%"),
        ("pixel_ae", 1, (0.4, -0.9, 1.7), (2.0, 1.0, 3.0), (3.0, 1.5, 4.5), "UNREADABLE", "--"),
        ("pixel_ae", 2, (-0.641206, -1.6, 0.4), (-2.1768, -3.0, -1.0), (1.5, 0.5, 2.5), "UNREADABLE", "--"),
        ("random_vit", 0, (0.5, 0.0, 1.0), (0.25, 0.1, 0.4), (0.25, 0.1, 0.4), "UNREADABLE", "--"),
        ("random_vit", 1, (5.0, 3.0, 7.0), (2.0, 1.0, 3.0), (0.1, -0.9, 1.1), "AT_PERFECT", "40.0%"),
        ("random_vit", 2, (5.0, 3.0, 7.0), (0.1, -0.9, 1.1), (3.0, 1.5, 4.5), "AT_COPYING", "2.0%"),
    ]
    cells = [_cell(a, s, hd, sk, df) for a, s, hd, sk, df, _, _ in table]
    inputs = _inputs(cells)
    reading = reading_headroom(inputs)
    assert reading.verdict == "NO_MAJORITY", "a mixed table, so no placement carries"
    assert sum(1 for t in table if t[6] == "--") == 3 and len(table) == 9

    rows = _body(format_reading_headroom(reading, inputs))
    for arm, seed, _, _, _, placement, share in table:
        row = rows[f"{arm}_seed{seed}"]
        assert _column(row, 6).strip() == placement, f"{arm}_seed{seed}"
        assert _column(row, 5).strip() == share, f"{arm}_seed{seed}"
        assert ("%" in row) == (share != "--"), f"{arm}_seed{seed}"


def test_the_formatter_prints_the_verdict_and_the_rule_it_was_reached_by():
    """A verdict line and a rule line, both reading what the reading says."""
    inputs = _inputs(_nine(["BETWEEN"] * 5 + ["AMBIGUOUS"] * 4))
    reading = reading_headroom(inputs)
    assert reading.verdict == "BETWEEN"
    text = format_reading_headroom(reading, inputs)
    assert "verdict: BETWEEN\n" in text
    assert f"rule:    {reading.rule}\n" in text


def test_the_formatter_conditions_every_directional_sentence_on_the_gate():
    """No unconditional claim that a sign means a verdict.

    M3m shipped "both directions are sound" in three places the results
    refuted -- a docstring, the text written into burden.txt, and spec 3.3. The
    legend here must say what the gate makes the sign mean, so the sentence
    cannot be inherited as unconditional.

    Every legend sentence that says what a resolvably positive difference means
    must itself contain the gate, and the sentence stating the gate must come
    first. The count of such claims is asserted (skill's and deficit's) so that
    deleting the claims cannot pass by leaving nothing to check. The table is
    nine UNREADABLE cells, so the verdict beside the legend is the gate's.
    """
    cells = _nine(["UNREADABLE"] * 9)
    inputs = _inputs(cells)
    reading = reading_headroom(inputs)
    assert reading.verdict == "UNREADABLE"
    text = format_reading_headroom(reading, inputs)
    assert "verdict: UNREADABLE\n" in text
    assert all(_column(r, 5).strip() == "--" for r in _body(text).values())

    flat = " ".join(text.split("\nlegend\n", 1)[1].split())
    sentences = [s.strip() for s in flat.split(". ") if s.strip()]
    directional = [s for s in sentences if "resolvably positive" in s]
    assert sum(s.count("resolvably positive") for s in directional) == 2
    for sentence in directional:
        assert "behind that gate" in sentence.lower()
    lowered = flat.lower()
    assert lowered.index("only if headroom is resolvably above zero") < lowered.index(
        "behind that gate"
    )
    assert "no claim about the model" in lowered
    assert "interval ends do not add" in lowered
