"""P: frozen dataclass point/vector."""

import copy
import dataclasses
import math
import pickle
import random
from typing import Any

import pytest

from geom2d import GeometryError, P, const
from tests.helpers import XY

RAW_PAIR = (1.0, 2.0)


# ----- construction ------------------------------------------------------


def test_fields_are_floats():
    p = P(1, 2)
    assert type(p.x) is float
    assert type(p.y) is float
    assert (p.x, p.y) == (1.0, 2.0)


def test_positional_and_match_args():
    assert P.__match_args__ == ("x", "y")
    match P(3.0, 4.0):
        case P(x, y):
            assert (x, y) == (3.0, 4.0)
        case _:
            pytest.fail("match on positional fields failed")


def test_frozen():
    p: Any = P(1.0, 2.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.x = 5.0
    assert not hasattr(p, "__dict__")


def test_of_identity_and_conversions():
    p = P(1.0, 2.0)
    assert P.of(p) is p
    assert P.of((1, 2)) == p
    assert P.of([1.0, 2.0]) == p
    assert P.of(XY(1, 2)) == p
    assert type(P.of(XY(1, 2)).x) is float


@pytest.mark.parametrize("bad", [(1,), (1, 2, 3), "ab", ("a", "b"), object(), None, 5])
def test_of_rejects_non_points(bad):
    with pytest.raises(GeometryError):
        P.of(bad)


def test_from_polar():
    p = P.from_polar(2.0, math.pi / 2)
    assert p.almost_equal(P(0.0, 2.0))
    assert P(0.0, -3.0).angle == pytest.approx(-math.pi / 2)


# ----- equality and hashing (RC3) ------------------------------------------


def test_eq_and_hash_agree_on_grid_cells():
    # Equal points must hash equal (grid-cell identity).
    p1 = P(0.5e-8 - 1e-12, 0.0)
    p2 = P(0.5e-8 + 1e-12, 0.0)
    assert (p1 == p2) == (hash(p1) == hash(p2))
    assert P(1.0, 1.0) == P(1.0 + 1e-10, 1.0 - 1e-10)
    assert hash(P(1.0, 1.0)) == hash(P(1.0 + 1e-10, 1.0 - 1e-10))
    assert P(1.0, 1.0) != P(1.0 + 1e-7, 1.0)
    assert len({P(1.0, 1.0), P(1.0 + 1e-10, 1.0)}) == 1


def test_eq_implies_equal_hash_property():
    rng = random.Random(12345)
    for _ in range(20_000):
        x = rng.uniform(-1e3, 1e3)
        y = rng.uniform(-1e3, 1e3)
        a = P(x, y)
        b = P(x + rng.uniform(-1e-8, 1e-8), y + rng.uniform(-1e-8, 1e-8))
        if a == b:
            assert hash(a) == hash(b)
        assert a in {a}
        assert {a: 1}[a] == 1


def test_eq_with_non_points_is_false_not_an_error():
    # Comparing with non-points is False, never an error.
    p = P(1.0, 2.0)
    assert p != "ab"
    assert p != (1.0, 2.0)
    assert p != [1.0, 2.0, 3.0]
    assert p != None  # noqa: E711 - exercising __eq__ directly
    assert P(1.0, 2.0) in [p]
    assert "x" not in [p]


def test_almost_equal_is_coincidence():
    a = P(0.0, 0.0)
    assert a.almost_equal(P(0.5e-8, 0.0))
    assert not a.almost_equal(P(2e-8, 0.0))
    assert a.almost_equal(P(0.05, 0.0), tolerance=0.1)


# ----- unpacking, copying, replace ------------------------------------------


def test_unpack_and_iterate():
    x, y = P(1.0, 2.0)
    assert (x, y) == (1.0, 2.0)
    assert tuple(P(3.0, 4.0)) == (3.0, 4.0)
    assert len(list(P(3.0, 4.0))) == 2


def test_not_a_sequence():
    p: Any = P(1.0, 2.0)
    with pytest.raises(TypeError):
        p[0]
    with pytest.raises(TypeError):
        len(p)


def test_copy_pickle_replace():
    p = P(1.5, -2.5)
    assert copy.copy(p) == p
    assert copy.deepcopy(p) == p
    assert pickle.loads(pickle.dumps(p)) == p
    assert dataclasses.replace(p, y=7) == P(1.5, 7.0)
    assert type(dataclasses.replace(p, y=7).y) is float


def test_repr_and_str_are_deterministic():
    assert repr(P(1.0, -2.5)) == "P(x=1.0, y=-2.5)"
    assert str(P(1.0, -2.5)) == "(1.00000000, -2.50000000)"


# ----- arithmetic ----------------------------------------------------------


def test_vector_arithmetic():
    a = P(1.0, 2.0)
    b = P(3.0, -1.0)
    assert a + b == P(4.0, 1.0)
    assert a - b == P(-2.0, 3.0)
    assert -a == P(-1.0, -2.0)
    assert a * 2 == P(2.0, 4.0)
    assert 2.0 * a == P(2.0, 4.0)
    assert a / 2 == P(0.5, 1.0)
    assert sum([a, b, a], P(0, 0)) == P(5.0, 3.0)


@pytest.mark.parametrize(
    "expr",
    [
        lambda a, b: a * b,
        lambda a, _b: a + 1,
        lambda a, _b: 1 + a,
        lambda a, _b: a + RAW_PAIR,
        lambda a, _b: a * "x",
        lambda a, b: a / b,
        lambda a, _b: "x" * a,
    ],
)
def test_unsupported_operands_raise_type_error(expr):
    # Unsupported operands raise TypeError; tuples never concatenate.
    with pytest.raises(TypeError):
        expr(P(1.0, 2.0), P(3.0, 4.0))


def test_division_by_zero():
    zero = 0.0
    with pytest.raises(ZeroDivisionError):
        P(1.0, 2.0) / zero


# ----- derived values -------------------------------------------------------


def test_length_angle_unit_normal():
    p = P(3.0, 4.0)
    assert p.length == 5.0
    assert p.length2 == 25.0
    assert P(0.0, 1.0).angle == pytest.approx(math.pi / 2)
    assert P(-1.0, 0.0).angle == pytest.approx(math.pi)
    assert p.unit.almost_equal(P(0.6, 0.8))
    assert P(0.0, 0.0).unit == P(0.0, 0.0)
    tiny = P(3e-9, 4e-9).unit  # shorter than EPSILON but still a direction
    assert tiny.almost_equal(P(0.6, 0.8))
    assert P(0.0, 0.0).is_zero
    assert P(0.5e-8, 0.0).is_zero
    assert not P(1e-7, 0.0).is_zero
    assert P(1.0, 0.0).normal() == P(0.0, 1.0)
    assert P(1.0, 0.0).normal(left=False) == P(0.0, -1.0)


def test_dot_and_cross_sign_convention():
    x = P(1.0, 0.0)
    y = P(0.0, 1.0)
    assert x.dot(y) == 0.0
    assert x.dot(P(2.0, 3.0)) == 2.0
    assert x.cross(y) == 1.0  # y is to the left of x: positive
    assert y.cross(x) == -1.0


def test_angle2_and_ccw_angle2():
    o = P(0.0, 0.0)
    assert o.angle2(P(1.0, 0.0), P(0.0, 1.0)) == pytest.approx(math.pi / 2)
    assert o.angle2(P(0.0, 1.0), P(1.0, 0.0)) == pytest.approx(-math.pi / 2)
    assert o.angle2(P(1.0, 0.0), P(1.0, 0.0)) == 0.0
    assert o.angle2(o, P(1.0, 0.0)) == 0.0  # coincident with self
    assert o.ccw_angle2(P(0.0, 1.0), P(1.0, 0.0)) == pytest.approx(3 * math.pi / 2)


def test_distances():
    a = P(0.0, 0.0)
    b = P(3.0, 4.0)
    assert a.distance(b) == 5.0
    assert a.distance2(b) == 25.0
    assert P(5.0, 2.0).distance_to_line(P(0.0, 0.0), P(10.0, 0.0)) == 2.0
    assert P(5.0, 2.0).distance_to_line(P(1.0, 1.0), P(1.0, 1.0)) == pytest.approx(math.hypot(4.0, 1.0))
    assert P(10.0, 0.0).normal_projection(P(5.0, 7.0)) == 0.5
    assert P(0.0, 0.0).normal_projection(P(5.0, 7.0)) == 0.0


def test_winding_sign_and_scale_independence():
    # Positive means counter-clockwise (left), consistent with cross().
    o = P(0.0, 0.0)
    assert o.winding(P(1.0, 0.0), P(1.0, 1.0)) == 1
    assert o.winding(P(1.0, 0.0), P(1.0, -1.0)) == -1
    assert o.winding(P(1.0, 1.0), P(2.0, 2.0)) == 0
    # Collinearity is a perpendicular distance, so scale does not matter.
    s = 1e-4
    assert P(0.0, 0.0).winding(P(s, 0.0), P(s, s)) == 1
    big = 1e6
    assert P(0.0, 0.0).winding(P(big, 0.0), P(big, 1e-9)) == 0  # 1e-9 off a 1e6 line
    assert P(0.0, 0.0).winding(P(big, 0.0), P(big, 1e-6)) == 1


def test_colinear():
    assert P(0.0, 0.0).colinear(P(1.0, 1.0), P(5.0, 5.0))
    assert not P(0.0, 0.0).colinear(P(1.0, 1.0), P(5.0, 5.1))
    assert P(0.0, 0.0).colinear(P(1e6, 0.0), P(5e5, 1e-9))
    assert not P(0.0, 0.0).colinear(P(1e-3, 0.0), P(5e-4, 1e-6))


def test_rotate():
    p = P(1.0, 0.0)
    assert p.rotate(math.pi / 2).almost_equal(P(0.0, 1.0))
    assert p.rotate(math.pi / 2, origin=P(1.0, 1.0)).almost_equal(P(2.0, 1.0))
    assert p.rotate(0.0) is p
    # A tiny angle at a large radius is a real displacement and must not be dropped.
    far = P(1e6, 0.0).rotate(1e-9)
    assert far.almost_equal(P(1e6 * math.cos(1e-9), 1e6 * math.sin(1e-9)))
    assert far.distance(P(1e6, 0.0)) == pytest.approx(1e-3, rel=1e-6)


def test_hash_follows_the_current_epsilon(restore_epsilon):
    # Documented policy: set_epsilon is called once at startup, before geometry is hashed.
    p = P(1, 2)
    d = {p: 1}
    assert p in d
    const.set_epsilon(1e-6)
    assert p not in d


def test_to_svg():
    assert P(1.5, 2.0).to_svg() == "1.5,2"
    assert P(1.5, 2.0).to_svg(scale=2.0, precision=1) == "3,4"
    assert P(1.0 / 3.0, 100.0).to_svg(precision=3) == "0.333,100"
