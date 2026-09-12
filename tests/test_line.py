"""Line: directed straight segment."""

import copy
import dataclasses
import math
import pickle
import random

import pytest

from geom2d import GeometryError, P, angle_eq, const
from geom2d.box import Box
from geom2d.line import Line
from tests.helpers import XY

H = Line(P(0, 0), P(10, 0))  # along +x


# ----- construction and dataclass behaviour ---------------------------------


def test_from_polar_accepts_point_like_start():
    assert Line.from_polar((1, 1), 2.0, math.pi / 2).p2.almost_equal(P(1, 3))
    assert Line.from_polar(XY(1, 1), 1.0, 0.0).p2 == P(2, 1)


def test_equality_is_directional_and_hash_consistent():
    a = Line(P(0, 0), P(1, 1))
    b = Line(P(1, 1), P(0, 0))
    assert a != b
    assert a == Line(P(0, 0), P(1 + 1e-10, 1))
    assert hash(a) == hash(Line(P(0, 0), P(1 + 1e-10, 1)))
    assert a.reversed() == b
    assert a.reversed().reversed() == a
    # Hashes of axis-aligned segments must not collapse onto a few buckets.
    assert len({hash(Line(P(i, 0), P(i, 1))) for i in range(200)}) > 190


def test_equal_lines_hash_equal():
    rng = random.Random(21)
    for _ in range(5_000):
        p1 = P(rng.uniform(-1e3, 1e3), rng.uniform(-1e3, 1e3))
        p2 = P(rng.uniform(-1e3, 1e3), rng.uniform(-1e3, 1e3))
        jitter = P(rng.uniform(-1e-8, 1e-8), rng.uniform(-1e-8, 1e-8))
        a = Line(p1, p2)
        b = Line(p1 + jitter, p2 - jitter)
        if a == b:
            assert hash(a) == hash(b)


def test_copy_pickle_replace_match():
    ln = Line(P(0, 0), P(3, 4))
    assert copy.deepcopy(ln) == ln
    assert pickle.loads(pickle.dumps(ln)) == ln
    assert dataclasses.replace(ln, p2=P(6, 8)).length == 10.0
    match ln:
        case Line(p1, p2):
            assert (p1, p2) == (P(0, 0), P(3, 4))
    assert repr(ln) == "Line(p1=P(x=0.0, y=0.0), p2=P(x=3.0, y=4.0))"


# ----- derived values -------------------------------------------------------


def test_lengths_angles_tangents():
    ln = Line(P(0, 0), P(3, 4))
    assert ln.length == 5.0
    assert ln.vector == P(3, 4)
    assert ln.midpoint == P(1.5, 2)
    assert ln.angle == pytest.approx(math.atan2(4, 3))
    assert ln.start_tangent_angle == ln.end_tangent_angle == ln.angle
    assert ln.start_tangent.almost_equal(P(0.6, 0.8))
    assert ln.end_tangent == ln.tangent_at(0.3)
    assert Line(P(1, 0), P(0, 0)).angle == pytest.approx(math.pi)  # leftward is +pi
    assert angle_eq(Line(P(1, 0), P(0, 0)).angle, Line(P(0, 0), P(-1, -1e-13)).angle)
    assert ln.bounding_box == Box(P(0, 0), P(3, 4))


def test_degenerate_segment_documented_returns():
    d = Line(P(1, 1), P(1, 1 + 0.5e-8))
    assert d.is_degenerate
    assert d.angle == 0.0
    assert d.start_tangent_angle == 0.0
    assert d.start_tangent == P(0, 0)
    assert d.mu(P(5, 5)) == 0.0
    assert d.point_at(0.7) == d.p1
    assert d.shift(3.0) is d
    assert d.extend(3.0) is d
    assert d.offset(3.0) is d
    assert d.distance_to_point(P(4, 5)) == pytest.approx(5.0)
    assert d.which_side(P(2, 2)) == 0
    assert d.point_on_line(P(1, 1))
    assert not d.point_on_line(P(2, 1))
    assert not d.is_parallel(H)
    assert not d.intersects(H)
    assert not d.crosses(H)
    assert d.subdivide(0.5) == (Line(d.p1, d.p1), Line(d.p1, d.p2))


# ----- parametric -----------------------------------------------------------


def test_point_at_mu_subdivide():
    ln = Line(P(0, 0), P(10, 0))
    assert ln.point_at(0.25) == P(2.5, 0)
    assert ln.point_at(1.5) == P(15, 0)  # extrapolates
    assert ln.mu(P(2.5, 7)) == 0.25  # projection, signed
    assert ln.mu(P(-5, 0)) == -0.5
    a, b = ln.subdivide(0.3)
    assert a == Line(P(0, 0), P(3, 0))
    assert b == Line(P(3, 0), P(10, 0))
    with pytest.raises(GeometryError):
        ln.subdivide(1.5)
    with pytest.raises(GeometryError):
        ln.subdivide(-0.1)


def test_projection_and_distance():
    ln = Line(P(0, 0), P(10, 0))
    assert ln.normal_projection_point(P(4, 3)) == P(4, 0)
    assert ln.normal_projection_point(P(14, 3)) == P(14, 0)
    assert ln.normal_projection_point(P(14, 3), segment=True) == P(10, 0)
    assert ln.distance_to_point(P(4, 3)) == 3.0
    assert ln.distance_to_point(P(14, 3)) == 3.0
    assert ln.distance_to_point(P(14, 3), segment=True) == 5.0


# ----- transformations -------------------------------------------------------


def test_extend_shift_offset():
    ln = Line(P(0, 0), P(10, 0))
    assert ln.extend(5) == Line(P(0, 0), P(15, 0))
    assert ln.extend(-4) == Line(P(0, 0), P(6, 0))
    assert ln.extend(4, from_midpoint=True) == Line(P(-2, 0), P(12, 0))
    with pytest.raises(GeometryError):
        ln.extend(-11)
    assert ln.shift(3) == Line(P(3, 0), P(13, 0))
    assert ln.shift(-3) == Line(P(-3, 0), P(7, 0))
    # Positive offset is to the left of travel.
    assert ln.offset(2) == Line(P(0, 2), P(10, 2))
    assert ln.offset(-2) == Line(P(0, -2), P(10, -2))
    assert ln.reversed().offset(2) == Line(P(10, -2), P(0, -2))
    assert ln.offset(0.0) is ln


# ----- sides and collinearity -------------------------------------------------


def test_which_side_and_same_side():
    assert H.which_side(P(5, 1)) == 1
    assert H.which_side(P(5, -1)) == -1
    assert H.which_side(P(5, 0)) == 0
    # A1 finding: same_side compared c1 with itself.
    assert not H.same_side(P(5, 1), P(5, -1))
    assert H.same_side(P(5, 1), P(7, 3))
    assert H.same_side(P(5, 0), P(7, -3))


def test_point_on_line_is_scale_independent():
    # Collinearity is a perpendicular distance, independent of the segment's length.
    big = Line(P(0, 0), P(1e6, 0))
    assert big.point_on_line(P(5e5, 1e-9))
    assert not big.point_on_line(P(5e5, 1e-6))
    small = Line(P(0, 0), P(1e-3, 0))
    assert not small.point_on_line(P(5e-4, 1e-6))
    assert small.point_on_line(P(5e-4, 1e-9))
    assert H.point_on_line(P(15, 0))
    assert not H.point_on_line(P(15, 0), segment=True)
    assert H.point_on_line(P(10 + 0.5e-8, 0), segment=True)
    assert H.point_on_line(P(-0.5e-8, 0), segment=True)


def test_is_parallel():
    assert H.is_parallel(Line(P(0, 5), P(3, 5)))
    assert H.is_parallel(Line(P(3, 5), P(0, 5)))  # opposite direction still parallel
    assert not H.is_parallel(Line(P(0, 5), P(3, 6)))
    assert H.is_parallel(Line(P(20, 0), P(30, 0)), inline=True)
    assert not H.is_parallel(Line(P(0, 5), P(3, 5)), inline=True)
    # Tiny perpendicular segments are not parallel.
    s = 1e-5
    assert not Line(P(0, 0), P(s, 0)).is_parallel(Line(P(s / 2, -s), P(s / 2, s)))


# ----- intersections ----------------------------------------------------------


def test_intersection_basic():
    v = Line(P(5, -5), P(5, 5))
    assert H.intersection(v) == P(5, 0)
    assert H.intersection_mu(v) == 0.5
    assert v.intersection_mu(H) == 0.5
    far = Line(P(50, -5), P(50, 5))
    assert H.intersection(far) == P(50, 0)  # infinite lines
    assert H.intersection(far, segment=True) is None
    assert H.intersects(far)
    assert not H.intersects(far, segment=True)
    assert H.intersects(v, segment=True)


def test_intersection_tiny_perpendicular_segments():
    # Tiny segments intersect like any others.
    s = 1e-5
    h = Line(P(0, 0), P(s, 0))
    v = Line(P(s / 2, -s), P(s / 2, s))
    assert h.intersects(v, segment=True)
    assert h.intersection(v, segment=True).almost_equal(P(s / 2, 0), tolerance=1e-12)


def test_intersection_parallel_and_collinear():
    # Collinear overlapping segments intersect.
    assert H.intersection(Line(P(0, 1), P(10, 1))) is None
    assert not H.intersects(Line(P(0, 1), P(10, 1)))
    assert H.intersects(H)
    assert H.intersects(H, segment=True)
    overlap = Line(P(5, 0), P(15, 0))
    assert H.intersects(overlap, segment=True)
    assert H.intersection(overlap, segment=True) == P(5, 0)
    assert H.intersection_mu(overlap) == 0.0
    disjoint = Line(P(20, 0), P(30, 0))
    assert H.intersects(disjoint)
    assert H.intersection(disjoint) == P(0, 0)  # infinite collinear lines meet everywhere; p1 is reported
    assert not H.intersects(disjoint, segment=True)
    assert H.intersection(disjoint, segment=True) is None


def test_segment_bounds_are_distance_tolerant():
    # The endpoint touch is accepted within EPSILON as a distance, at any scale.
    big = Line(P(0, 0), P(1e6, 0))
    touch = Line(P(1e6 + 0.5e-8, -1), P(1e6 + 0.5e-8, 1))
    assert big.intersects(touch, segment=True)
    miss = Line(P(1e6 + 1e-6, -1), P(1e6 + 1e-6, 1))
    assert not big.intersects(miss, segment=True)


def test_crosses_is_strict_and_symmetric():
    # A crossing is strictly interior to both segments.
    a = Line(P(0, 0), P(10, 0))
    t_junction = Line(P(5, 0), P(5, 5))
    assert not a.crosses(t_junction)
    assert not t_junction.crosses(a)
    x = Line(P(5, -5), P(5, 5))
    assert a.crosses(x)
    assert x.crosses(a)
    assert not a.crosses(Line(P(5, 0), P(15, 0)))  # collinear overlap is not a crossing
    assert not a.crosses(Line(P(10, -5), P(10, 5)))  # endpoint touch
    for seg in (a, t_junction, x):
        assert seg.crosses(seg) is False


# ----- output -------------------------------------------------------------------


def test_to_svg_path():
    ln = Line(P(1.5, 2), P(3, 4.25))
    assert ln.to_svg_path() == "L 1.5,2 L 3,4.25"
    assert ln.to_svg_path(add_move=True) == "M 1.5,2 L 3,4.25"
    assert ln.to_svg_path(add_prefix=False) == "1.5,2 3,4.25"
    assert ln.to_svg_path(scale=2, precision=1) == "L 3,4 L 6,8.5"
    assert str(ln) == "Line((1.50000000, 2.00000000), (3.00000000, 4.25000000))"


def test_epsilon_change_applies_at_call_time(restore_epsilon):
    ln = Line(P(0, 0), P(10, 0))
    assert not ln.point_on_line(P(5, 1e-4))
    const.set_epsilon(1e-3)
    assert ln.point_on_line(P(5, 1e-4))
