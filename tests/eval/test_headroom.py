import ast
import inspect

import numpy as np
import pytest

from mbfps.eval import burden
from mbfps.eval import headroom as module
from mbfps.eval.headroom import (
    ARMS_REQUIRED, CONFIDENCE, DECISION_H, DECISION_K,
    DISPLACEMENT_RECORDED_ONLY, IDENTITY_TOLERANCE, REPORTED_H, RESAMPLES,
    SECONDARY_SIGMAS, SEEDS_MINIMUM, deficit, headroom, skill,
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

    `pytest.approx(rel=1e-6)` here would be the wrong tool: a `np.sum`-for-
    `np.max` mutation inside `triple_residual` changes the result in the 7th
    significant figure, and M3m nearly shipped exactly that tolerance against
    exactly that mutation. (This fixture cannot tell `sum` from `max` -- every
    step's residual is 0.0 -- which is what the next test is for.)
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


def test_the_displacement_constant_is_recorded_and_unused():
    """3.9694722203504225 appears in no arithmetic in this module.

    It is the ground-truth quantity M3m's design mistook for a probe-space
    scale, kept so a reader of the record can see it plays no part. An unused
    constant cannot be caught by a value, so the guard reads the module's
    syntax tree: the ONLY reference to `DISPLACEMENT_RECORDED_ONLY` anywhere in
    the file is the assignment that defines it.

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
