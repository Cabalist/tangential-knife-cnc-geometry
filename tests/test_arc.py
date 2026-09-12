"""Arc: directed circular arc."""

import copy
import dataclasses
import itertools
import math
import pickle
import random

import pytest

from geom2d import DegenerateGeometryError, GeometryError, P, angle_eq, const, float_eq, normalize_angle
from geom2d.arc import Arc, calc_center, intersect_circles
from geom2d.box import Box
from geom2d.line import Line
from tests.helpers import XY

PI = math.pi
ORIGIN = P(0, 0)

# The four sweep families used throughout: CCW/CW x minor/major, unit circle at the origin.
CCW_Q = Arc.from_sweep(P(1, 0), P(0, 1), 1.0, PI / 2)  # first quadrant, counter-clockwise
CW_Q = Arc.from_sweep(P(0, 1), P(1, 0), 1.0, -PI / 2)  # first quadrant, clockwise
CCW_M = Arc.from_sweep(P(1, 0), P(0, -1), 1.0, 3 * PI / 2)  # major, counter-clockwise (misses 4th quadrant)
CW_M = Arc.from_sweep(P(0, -1), P(1, 0), 1.0, -3 * PI / 2)  # major, clockwise
FAMILIES = [CCW_Q, CW_Q, CCW_M, CW_M]


def _sector_contains(arc: Arc, p: P) -> bool:
    """Independent sector test: within radius and angular position within the signed sweep."""
    if p.distance(arc.center) > arc.radius + 1e-12:
        return False
    theta = ((p - arc.center).angle - (arc.p1 - arc.center).angle) * arc.direction
    theta %= math.tau
    return theta <= abs(arc.angle) + 1e-12


# ----- construction and the invariant ---------------------------------------


def test_from_sweep_families_are_consistent():
    for arc in FAMILIES:
        assert arc.center.almost_equal(ORIGIN)
        assert arc.point_at(0.0) == arc.p1
        assert arc.point_at(1.0) == arc.p2
        for i in range(11):
            assert float_eq(arc.point_at(i / 10).distance(arc.center), 1.0)
        # midpoint lies on the correct side of the chord for CW vs CCW
        assert _sector_contains(arc, arc.midpoint * 0.999)


def test_direct_construction_with_center():
    arc = Arc(P(1, 0), P(0, 1), 1.0, PI / 2, P(0, 0))
    assert arc == CCW_Q
    assert Arc(P(2, 1), P(1, 2), 1.0, PI / 2, P(1, 1)).center == P(1, 1)


def test_factories_accept_foreign_points():
    assert Arc.from_sweep(XY(1, 0), (0, 1), 1.0, PI / 2) == CCW_Q
    assert Arc.from_two_points_and_tangent(XY(1, 1), (3, 1), XY(0, 0)) is not None


def test_invariant_uses_absolute_epsilon():
    bad = P(0, 1 + 1e-6)
    with pytest.raises(GeometryError):
        Arc(P(1, 0), bad, 1.0, PI / 2, ORIGIN)
    shift = P(1000, 1000)
    with pytest.raises(GeometryError):  # the same error after translation is still an error
        Arc(P(1, 0) + shift, bad + shift, 1.0, PI / 2, shift)
    with pytest.raises(GeometryError):  # a 1e-3 radial discrepancy at radius 1e6 is not within EPSILON
        Arc(P(1e6, 0), P(0, 1e6), 1e6 + 1e-3, PI / 2, ORIGIN)
    assert Arc(P(1e6, 0), P(0, 1e6), 1e6, PI / 2, ORIGIN).radius == 1e6


def test_arcs_near_half_turn_construct():
    for a in (PI - 1e-4, PI - 1e-6, PI - 1e-7, PI, PI + 1e-7, PI + 1e-4, -(PI - 1e-5), -PI):
        arc = Arc.from_sweep(P(1, 0), P.from_polar(1, a), 1.0, a)
        assert arc.center.almost_equal(ORIGIN)
        assert arc.mu(arc.point_at(0.5)) == pytest.approx(0.5, abs=1e-9)
    # Rotated exact semicircles and near-semicircles at ordinary radii, where sqrt(ratio**2 - 1) cancels badly.
    for start in (0.3, 1.1, 2.9, -2.0):
        rotated = Arc.from_sweep(P.from_polar(1, start), P.from_polar(1, start + PI), 1.0, PI)
        assert rotated.center.almost_equal(ORIGIN)
    big = Arc.from_sweep(P(10, 0), P.from_polar(10, PI - 1e-7), 10.0, PI - 1e-7)
    assert big.center.almost_equal(ORIGIN)
    assert Arc.from_sweep(P(1, 0), P(-1, 0), 1.0, PI).center == ORIGIN
    with pytest.raises(GeometryError):  # distinct endpoints cannot be joined by a zero sweep
        Arc.from_sweep(P(1, 0), P(0, 1), 1.0, 0.0)


def test_small_sweep_at_large_radius_is_valid():
    arc = Arc(P(1e6, 0), P(1e6 * math.cos(1e-9), 1e6 * math.sin(1e-9)), 1e6, 1e-9, ORIGIN)
    assert not arc.is_degenerate
    assert arc.length == pytest.approx(1e-3)
    assert arc.mu(arc.point_at(0.5)) == pytest.approx(0.5, abs=1e-6)


def test_tiny_arc_tangents_agree_with_tangent_at():
    r = 5e-9
    semi = Arc(P(r, 0), P(-r, 0), r, PI, ORIGIN)  # length 1.57e-8: not degenerate
    assert not semi.is_degenerate
    assert semi.start_tangent.length == pytest.approx(1.0)
    assert semi.start_tangent.almost_equal(semi.tangent_at(0.0))
    assert semi.end_tangent.almost_equal(semi.tangent_at(1.0))
    assert semi.start_tangent.almost_equal(P(0, 1))


@pytest.mark.parametrize(
    ("p1", "p2", "radius", "angle", "center"),
    [
        (P(1, 0), P(0, 1), 1.0, PI / 2, P(0.1, 0)),  # center off: endpoints not on circle
        (P(1, 0), P(0, 1), 1.5, PI / 2, P(0, 0)),  # radius does not match
        (P(1, 0), P(0, 1), 1.0, -PI / 2, P(0, 0)),  # wrong sign: CW sweep from (1,0) lands at (0,-1)
        (P(1, 0), P(0, 1), 1.0, PI, P(0, 0)),  # wrong magnitude
        (P(1, 0), P(0, 1), 0.0, PI / 2, P(0, 0)),  # zero radius
        (P(1, 0), P(0, 1), -1.0, PI / 2, P(0, 0)),
        (P(1, 0), P(1, 0), 1.0, 3 * math.tau, P(0, 0)),  # more than a full turn
        (P(1, 0), P(0, 1), math.nan, PI / 2, P(0, 0)),
    ],
)
def test_invariant_rejects_inconsistent_arcs(p1, p2, radius, angle, center):
    # Inconsistent geometry must raise in every mode: no debug flag, no assert.
    with pytest.raises(GeometryError):
        Arc(p1, p2, radius, angle, center)


def test_replace_revalidates():
    arc = CCW_Q
    assert dataclasses.replace(arc, radius=1.0) == arc
    with pytest.raises(GeometryError):
        dataclasses.replace(arc, angle=-arc.angle)
    bigger = dataclasses.replace(arc, radius=2.0, p1=P(2, 0), p2=P(0, 2))
    assert bigger.length == pytest.approx(PI)


def test_from_sweep_degenerate_and_impossible():
    with pytest.raises(DegenerateGeometryError):
        Arc.from_sweep(P(1, 1), P(1, 1), 1.0, math.tau)
    with pytest.raises(GeometryError):
        Arc.from_sweep(P(0, 0), P(10, 0), 1.0, PI)  # chord longer than diameter
    semi = Arc.from_sweep(P(0, 0), P(2, 0), 1.0, PI)  # chord equals diameter
    assert semi.center == P(1, 0)
    # (0, 0) sits at angle pi about (1, 0); a CCW sweep passes below the chord.
    assert semi.midpoint.almost_equal(P(1, -1))
    assert Arc.from_sweep(P(0, 0), P(2, 0), 1.0, -PI).midpoint.almost_equal(P(1, 1))


def test_zero_sweep_arc_is_degenerate_not_an_error():
    d = Arc(P(1, 0), P(1, 0), 1.0, 0.0, P(0, 0))
    assert d.is_degenerate
    assert d.length == 0.0
    assert d.start_tangent_angle == 0.0
    assert d.start_tangent == P(0, 0)
    assert d.mu(P(0, 1)) == 0.0
    assert d.point_at(0.5) == d.p1
    assert d.extend(1.0) is d
    assert d.offset(0.5) is d


def test_from_two_points_and_tangent_uses_point_semantics():
    # The tangent argument is a point, not a vector; translating everything must not change the result.
    arc = Arc.from_two_points_and_tangent(P(1, 1), P(3, 1), P(0, 0))
    moved = Arc.from_two_points_and_tangent(P(6, 6), P(8, 6), P(5, 5))
    assert arc is not None
    assert moved is not None
    assert float_eq(arc.radius, moved.radius)
    assert float_eq(arc.angle, moved.angle)
    assert angle_eq(arc.start_tangent_angle, 0.0)  # tangent at p1 points along +x toward (3, 1)
    assert Arc.from_two_points_and_tangent(P(0, 0), P(1, 0), P(0, 0)) is None  # coincident endpoints
    assert Arc.from_two_points_and_tangent(P(0, 0), P(0, 0), P(1, 1)) is None  # coincident tangent point
    assert Arc.from_two_points_and_tangent(P(0, 0), P(1, 0), P(5, 0)) is None  # collinear: a line
    # Nearly coincident endpoints are degenerate even when they fall in different hash cells.
    assert Arc.from_two_points_and_tangent(P(0, 0), P(1, 0), P(6e-9, 0)) is None
    rev = Arc.from_two_points_and_tangent(P(1, 1), P(3, 1), P(0, 0), reverse=True)
    assert rev is not None
    assert rev.p1 == P(0, 0)
    assert rev == arc.reversed()


# ----- derived values -----------------------------------------------------------


def test_direction_flags_length():
    assert CCW_Q.direction == 1
    assert not CCW_Q.is_clockwise
    assert CW_Q.direction == -1
    assert CW_Q.is_clockwise
    assert CCW_Q.length == pytest.approx(PI / 2)
    assert CCW_M.large_arc_flag == 1
    assert CCW_Q.large_arc_flag == 0
    assert CCW_Q.sweep_flag == 1
    assert CW_Q.sweep_flag == 0
    assert not CCW_Q.is_full_circle
    assert Arc(P(1, 0), P(1, 0), 1.0, math.tau, ORIGIN).is_full_circle


def test_start_angle_with_offset_center():
    # start_angle is measured about the center, not from the absolute point (1, 0).
    arc = Arc(P(6, 5), P(5, 6), 1.0, PI / 2, P(5, 5))
    assert arc.start_angle == pytest.approx(0.0)
    assert arc.end_angle == pytest.approx(PI / 2)


def test_tangent_angles_and_vectors():
    assert CCW_Q.start_tangent_angle == pytest.approx(PI / 2)
    assert angle_eq(CCW_Q.end_tangent_angle, PI)
    assert CW_Q.start_tangent_angle == pytest.approx(0.0)
    assert CW_Q.end_tangent_angle == pytest.approx(-PI / 2)
    assert CCW_Q.start_tangent.almost_equal(P(0, 1))
    assert CCW_Q.end_tangent.almost_equal(P(-1, 0))
    assert CCW_Q.tangent_at(0.5).almost_equal(P(-1, 1).unit)
    assert CW_Q.tangent_at(0.0).almost_equal(P(1, 0))
    # Reversal flips every tangent.
    for arc in FAMILIES:
        rev = arc.reversed()
        assert angle_eq(rev.start_tangent_angle, normalize_angle(arc.end_tangent_angle + PI, 0.0))
        assert rev.start_tangent.almost_equal(-arc.end_tangent)


def test_height_for_minor_and_major_arcs():
    # The sagitta must be right past a half turn as well.
    assert CCW_Q.height == pytest.approx(1.0 - math.cos(PI / 4))
    major = Arc.from_sweep(P(1, 0), P(0, 1), 1.0, 3 * PI / 2)
    brute = max(major.point_at(i / 2000).distance_to_line(major.p1, major.p2) for i in range(2001))
    assert major.height == pytest.approx(brute, abs=1e-6)
    assert major.height == pytest.approx(1.0 + math.cos(PI / 4))


def test_bounding_box():
    assert CCW_Q.bounding_box == Box(P(0, 0), P(1, 1))
    assert CCW_M.bounding_box == Box(P(-1, -1), P(1, 1))
    small = Arc.from_sweep(P(1, 0), P(math.cos(0.3), math.sin(0.3)), 1.0, 0.3)
    assert small.bounding_box == Box(P(math.cos(0.3), 0), P(1, math.sin(0.3)))
    crossing = Arc.from_sweep(P.from_polar(1, -0.3), P.from_polar(1, 0.3), 1.0, 0.6)
    assert float_eq(crossing.bounding_box.xmax, 1.0)


# ----- parametric ------------------------------------------------------------------


def test_mu_and_point_at_round_trip_for_all_families():
    # mu must not wrap for sweeps beyond half a turn.
    for arc in FAMILIES:
        for t in (0.0, 0.1, 0.5, 0.7, 0.9, 1.0):
            assert arc.mu(arc.point_at(t)) == pytest.approx(t, abs=1e-9)
        assert arc.mu(arc.p1) == 0.0
        assert arc.mu(arc.p2) == pytest.approx(1.0)


def test_mu_outside_the_sweep_exceeds_one():
    assert CCW_Q.mu(P(-1, 0)) == pytest.approx(2.0)
    assert CCW_Q.mu(P(0, -1)) == pytest.approx(3.0)
    assert CCW_Q.point_at_angle(PI / 4).almost_equal(P.from_polar(1, PI / 4))
    assert CW_Q.point_at_angle(PI / 4).almost_equal(P.from_polar(1, PI / 4))


def test_subdivide_family():
    for arc in FAMILIES:
        a, b = arc.subdivide(0.25)
        assert a.p1 == arc.p1
        assert b.p2 == arc.p2
        assert a.p2 == b.p1
        assert float_eq(a.angle + b.angle, arc.angle)
        assert float_eq(a.angle, arc.angle * 0.25)
        a2, b2 = arc.subdivide_at_point(arc.point_at(5 / 6))
        assert float_eq(a2.angle, arc.angle * 5 / 6)
        assert float_eq(b2.angle, arc.angle / 6)
        a3, _b3 = arc.subdivide_at_angle(abs(arc.angle) / 3)
        assert float_eq(a3.angle, arc.angle / 3)
    with pytest.raises(GeometryError):
        CCW_Q.subdivide(1.5)
    with pytest.raises(GeometryError):
        CCW_Q.subdivide_at_point(P(-1, 0))
    with pytest.raises(GeometryError):
        CCW_Q.subdivide_at_angle(3.0)
    first, second = CCW_Q.subdivide(0.0)
    assert first.is_degenerate
    assert second == CCW_Q


def test_subdivide_equal_and_split_max_sweep():
    # Equal sweeps, each within the limit.
    big = Arc.from_sweep(P(1, 0), P.from_polar(1, math.radians(359)), 1.0, math.radians(359))
    parts = big.split_max_sweep(PI / 2)
    assert len(parts) == 4
    assert all(float_eq(p.angle, big.angle / 4) for p in parts)
    assert float_eq(sum(p.angle for p in parts), big.angle)
    for a, b in itertools.pairwise(parts):
        assert a.p2 == b.p1
    assert parts[0].p1 == big.p1
    assert parts[-1].p2 == big.p2
    semi = Arc.from_sweep(P(1, 0), P(-1, 0), 1.0, PI)
    assert len(semi.split_max_sweep(PI / 2)) == 2
    assert len(CCW_Q.split_max_sweep(PI / 2)) == 1  # exactly 90 degrees needs no split
    assert len(Arc.from_sweep(P(1, 0), P.from_polar(1, math.radians(91)), 1.0, math.radians(91)).split_max_sweep()) == 2
    cw = CW_M.split_max_sweep()
    assert len(cw) == 3
    assert all(p.is_clockwise for p in cw)
    assert CCW_Q.subdivide_equal(1) == [CCW_Q]
    with pytest.raises(GeometryError):
        CCW_Q.subdivide_equal(0)
    with pytest.raises(GeometryError):
        CCW_Q.split_max_sweep(0.0)


# ----- transformations ----------------------------------------------------------------


def test_reversed_is_a_different_arc():
    # The signed sweep is part of an arc's identity: the two semicircles on a chord differ.
    for arc in FAMILIES:
        rev = arc.reversed()
        assert rev != arc
        assert rev.reversed() == arc
        assert rev.midpoint.almost_equal(arc.midpoint)
    upper = Arc.from_sweep(P(0, 0), P(2, 0), 1.0, PI)
    lower = Arc.from_sweep(P(0, 0), P(2, 0), 1.0, -PI)
    assert upper != lower
    assert hash(upper) != hash(lower)


def test_extend():
    # extend lengthens the arc along its circle by the given arc length.
    ext = CCW_Q.extend(PI / 2)
    assert float_eq(ext.angle, PI)
    assert ext.p1 == CCW_Q.p1
    assert ext.p2.almost_equal(P(-1, 0))
    shrunk = CCW_Q.extend(-PI / 4)
    assert float_eq(shrunk.angle, PI / 4)
    mid = CCW_Q.extend(PI / 2, from_midpoint=True)
    assert float_eq(mid.angle, PI)
    assert mid.midpoint.almost_equal(CCW_Q.midpoint)
    cw = CW_Q.extend(PI / 2)
    assert float_eq(cw.angle, -PI)
    assert cw.p2.almost_equal(P(0, -1))
    with pytest.raises(GeometryError):
        CCW_Q.extend(-PI)
    with pytest.raises(GeometryError):
        CCW_Q.extend(math.tau)


def test_offset_is_left_of_travel():
    # offset(+d) is to the left of travel, matching Line.offset.
    inner = CCW_Q.offset(0.1)
    assert float_eq(inner.radius, 0.9)
    assert inner.center == CCW_Q.center
    assert float_eq(inner.angle, CCW_Q.angle)
    assert inner.p1.almost_equal(P(0.9, 0))
    outer = CW_Q.offset(0.1)
    assert float_eq(outer.radius, 1.1)
    assert float_eq(CCW_Q.offset(-0.1).radius, 1.1)
    # Agrees with Line.offset on the tangent: the offset arc's start tangent is parallel.
    assert angle_eq(inner.start_tangent_angle, CCW_Q.start_tangent_angle)
    line = Line(CCW_Q.p1, CCW_Q.p1 + CCW_Q.start_tangent)
    assert line.offset(0.1).p1.almost_equal(inner.p1)
    with pytest.raises(GeometryError):
        CCW_Q.offset(1.0)
    assert CCW_Q.offset(0.0) is CCW_Q


# ----- relations ---------------------------------------------------------------------


def test_point_on_arc_for_semicircles_and_major_arcs():
    # Sweep membership must be exact for semicircles and major arcs, endpoints included.
    semi = Arc.from_sweep(P(1, 0), P(-1, 0), 1.0, PI)
    assert semi.point_on_arc(P(0, 1))
    assert not semi.point_on_arc(P(0, -1))
    assert Arc.from_sweep(P(1, 0), P(-1, 0), 1.0, -PI).point_on_arc(P(0, -1))
    assert not Arc.from_sweep(P(1, 0), P(-1, 0), 1.0, -PI).point_on_arc(P(0, 1))
    assert CCW_M.point_on_arc(CCW_M.p1)
    assert CCW_M.point_on_arc(CCW_M.p2)
    assert CCW_M.point_on_arc(P(-1, 0))
    assert not CCW_M.point_on_arc(P.from_polar(1, -PI / 4))
    assert not CCW_Q.point_on_arc(P(0.5, 0.5))  # inside the circle, not on it
    assert CCW_Q.point_on_arc(P.from_polar(1 + 0.5e-8, PI / 4))


def test_point_inside_sector_matches_analytic_test():
    # The sector lies on the swept side, for both directions.
    assert CCW_Q.point_inside(P(0.5, 0.5))
    assert not CCW_Q.point_inside(P(-0.5, 0.5))
    assert CW_Q.point_inside(P(0.5, 0.5))
    assert not CW_Q.point_inside(P(0.5, -0.5))
    rng = random.Random(7)
    for arc in FAMILIES:
        mismatches = 0
        for _ in range(2000):
            p = P(rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2))
            if arc.point_inside(p) != _sector_contains(arc, p):
                mismatches += 1
        assert mismatches <= 2  # only points within tolerance of a boundary may differ
    assert CCW_Q.point_inside(ORIGIN)


def test_normal_projection_point_below_center():
    # Points below the center project radially, never to the antipode.
    circle = Arc.from_sweep(P(1, 0), P(-1, 0), 1.0, PI)
    assert circle.normal_projection_point(P(0, -2)).almost_equal(P(0, -1))
    assert circle.normal_projection_point(P(3, -4)).almost_equal(P(0.6, -0.8))
    assert circle.normal_projection_point(P(3, -4), segment=True) == P(1, 0)  # nearer endpoint
    assert circle.normal_projection_point(P(0, 2)).almost_equal(P(0, 1))
    assert circle.normal_projection_point(ORIGIN) == circle.p1


def test_distance_to_point():
    assert CCW_Q.distance_to_point(P(2, 0)) == pytest.approx(1.0)
    assert CCW_Q.distance_to_point(P(0.5, 0.5)) == pytest.approx(1 - math.hypot(0.5, 0.5))
    assert CCW_Q.distance_to_point(P(-2, 0)) == pytest.approx(1.0)  # circle, not sweep
    assert CCW_Q.distance_to_point(P(-2, 0), segment=True) == pytest.approx(P(-2, 0).distance(P(0, 1)))
    assert CCW_Q.distance_to_point(ORIGIN, segment=True) == pytest.approx(1.0)


def test_intersect_line():
    circle = Arc.from_sweep(P(1, 0), P(-1, 0), 1.0, PI)
    pts = circle.intersect_line(Line(P(-5, 0), P(5, 0)))
    assert len(pts) == 2
    assert {p.x for p in pts} == {1.0, -1.0}
    tangent = circle.intersect_line(Line(P(-5, 1), P(5, 1)))
    assert len(tangent) == 1
    assert tangent[0].almost_equal(P(0, 1))
    assert circle.intersect_line(Line(P(-5, 2), P(5, 2))) == []
    # on_line must respect the *segment* (the old ellipse copy did not).
    assert circle.intersect_line(Line(P(5, 0), P(6, 0)), on_line=True) == []
    assert circle.intersect_line(Line(P(-5, 1), P(-4, 1)), on_line=True) == []
    on_seg = circle.intersect_line(Line(P(0, 0), P(5, 0)), on_line=True)
    assert len(on_seg) == 1
    assert on_seg[0] == P(1, 0)
    # on_arc keeps the upper point only for the upper semicircle.
    vertical = Line(P(0, -5), P(0, 5))
    up = circle.intersect_line(vertical, on_arc=True)
    assert len(up) == 1
    assert up[0].almost_equal(P(0, 1))
    assert circle.intersect_line(Line(P(1, 1), P(1, 1))) == []  # degenerate line
    # The major arc from (1,0) CCW to (0,1) is centred at (1,1); x=1 is a diameter meeting it at p1 and (1,2).
    major = Arc.from_sweep(P(1, 0), P(0, 1), 1.0, 3 * PI / 2)
    assert major.center == P(1, 1)
    both = major.intersect_line(Line(P(1, -5), P(1, 5)), on_arc=True)
    assert len(both) == 2
    assert {p.almost_equal(P(1, 0)) or p.almost_equal(P(1, 2)) for p in both} == {True}


def test_intersect_circles_and_arcs():
    # Externally tangent circles meet r1 along the center line, not at the midpoint of the centers.
    assert intersect_circles(ORIGIN, 1.0, P(4, 0), 3.0) == (P(1, 0),)
    assert intersect_circles(ORIGIN, 3.0, P(4, 0), 1.0) == (P(3, 0),)
    assert intersect_circles(ORIGIN, 3.0, P(2, 0), 1.0) == (P(3, 0),)  # internally tangent, once
    assert intersect_circles(ORIGIN, 1.0, P(5, 0), 1.0) == ()
    assert intersect_circles(ORIGIN, 3.0, P(1, 0), 1.0) == ()  # nested
    assert intersect_circles(ORIGIN, 1.0, ORIGIN, 1.0) == ()
    two = intersect_circles(ORIGIN, 1.0, P(1, 0), 1.0)
    assert len(two) == 2
    assert {round(p.y, 6) for p in two} == {round(math.sqrt(3) / 2, 6), -round(math.sqrt(3) / 2, 6)}
    left = Arc.from_sweep(P(1, 1), P(1, -1), 1.0, PI)  # left-hand semicircle centred (1, 0)
    right = Arc.from_sweep(P(1, 1), P(1, -1), 1.0, -PI)
    assert left.midpoint.almost_equal(P(0, 0))
    assert right.midpoint.almost_equal(P(2, 0))
    assert len(CCW_Q.intersect_arc(right)) == 2  # the circles meet at x = 0.5 ...
    assert CCW_Q.intersect_arc(right, on_arc=True) == []  # ... which is off the right-hand sweep
    hits = CCW_Q.intersect_arc(left, on_arc=True)
    assert len(hits) == 1
    assert hits[0].almost_equal(P(0.5, math.sqrt(3) / 2))


# ----- dataclass protocol and output ----------------------------------------------------


def test_equal_arcs_hash_equal():
    rng = random.Random(22)
    for _ in range(2_000):
        angle = rng.uniform(-3.0, 3.0)
        if abs(angle) < 0.1:
            continue
        radius = rng.uniform(0.5, 50.0)
        center = P(rng.uniform(-1e3, 1e3), rng.uniform(-1e3, 1e3))
        start = rng.uniform(-PI, PI)
        p1 = center + P.from_polar(radius, start)
        p2 = center + P.from_polar(radius, start + angle)
        a = Arc(p1, p2, radius, angle, center)
        angle_jitter = 0.1 * const.EPSILON / radius  # keeps the swept endpoint within EPSILON of p2
        b = Arc(p1, p2, radius + rng.uniform(-5e-9, 5e-9), angle + rng.uniform(-angle_jitter, angle_jitter), center)
        if a == b:
            assert hash(a) == hash(b)


def test_copy_pickle_match_hash():
    arc = CCW_Q
    assert copy.deepcopy(arc) == arc
    assert pickle.loads(pickle.dumps(arc)) == arc
    assert hash(arc) == hash(Arc.from_sweep(P(1, 0), P(0, 1), 1.0, PI / 2))
    assert Arc.__match_args__ == ("p1", "p2", "radius", "angle", "center")
    match arc:
        case Arc(p1, p2, radius, angle, center):
            assert (p1, p2, radius, center) == (P(1, 0), P(0, 1), 1.0, ORIGIN)
            assert angle == pytest.approx(PI / 2)
    assert arc != "arc"
    assert len({arc, copy.deepcopy(arc), arc.reversed()}) == 2


def test_to_svg_path_and_str():
    assert CCW_Q.to_svg_path() == "A 1,1 0 0 1 0,1"
    assert CCW_Q.to_svg_path(add_move=True) == "M 1,0 A 1,1 0 0 1 0,1"
    assert CW_M.to_svg_path(add_prefix=False) == "1,1 0 1 0 1,0"
    assert CCW_Q.to_svg_path(scale=2, precision=1) == "A 2,2 0 0 1 0,2"
    assert str(CCW_Q).startswith("Arc((1.00000000, 0.00000000), (0.00000000, 1.00000000), r=1.00000000, a=1.57079633")


def test_calc_center_all_families():
    for p1, p2, angle in [
        (P(1, 0), P(0, 1), PI / 2),
        (P(0, 1), P(1, 0), -PI / 2),
        (P(1, 0), P(0, -1), 3 * PI / 2),
        (P(0, -1), P(1, 0), -3 * PI / 2),
    ]:
        assert calc_center(p1, p2, 1.0, angle).almost_equal(ORIGIN)


def test_epsilon_governs_the_invariant(restore_epsilon):
    with pytest.raises(GeometryError):
        Arc(P(1, 0), P(0, 1), 1.0, PI / 2, P(1e-4, 0))
    const.set_epsilon(1e-3)
    assert Arc(P(1, 0), P(0, 1), 1.0, PI / 2, P(1e-4, 0)).radius == 1.0
