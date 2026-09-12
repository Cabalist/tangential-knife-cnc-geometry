"""CubicBezier: evaluation, structure, intersections, biarc approximation."""

import copy
import dataclasses
import itertools
import math
import pickle
import random

import pytest

from geom2d import ApproximationError, GeometryError, P, angle_eq, const, segments_are_g1
from geom2d.arc import Arc
from geom2d.bezier import CubicBezier
from geom2d.box import Box
from geom2d.line import Line
from tests.helpers import XY, g1_everywhere, is_connected_chain

ARCH = CubicBezier(P(0, 0), P(1, 2), P(3, 2), P(4, 0))
S_CURVE = CubicBezier(P(0, 0), P(2, 3), P(4, -3), P(6, 0))  # one inflection at t = 0.5
LOOP = CubicBezier(P(0, 0), P(4, 3), P(-2, 3), P(2, 0))


def _de_casteljau(curve: CubicBezier, t: float) -> P:
    pts = [curve.p1, curve.c1, curve.c2, curve.p2]
    while len(pts) > 1:
        pts = [a * (1 - t) + b * t for a, b in itertools.pairwise(pts)]
    return pts[0]


def _random_curve(rng: random.Random, size: float = 10.0) -> CubicBezier:
    pts = [P(rng.uniform(-size, size), rng.uniform(-size, size)) for _ in range(4)]
    return CubicBezier(*pts)


def _segment_distance(seg: Line | Arc, p: P) -> float:
    return seg.distance_to_point(p, segment=True)


# ----- construction ----------------------------------------------------------


def test_from_quadratic_is_exact():
    p0, q, p1 = P(0, 0), P(5, 10), P(10, 0)
    cubic = CubicBezier.from_quadratic(p0, q, p1)
    assert CubicBezier.from_quadratic(XY(0, 0), XY(5, 10), (10, 0)) == cubic  # point-like input
    assert cubic.p1 == p0
    assert cubic.p2 == p1
    for t in (0.0, 0.2, 0.5, 0.8, 1.0):
        quad = p0 * (1 - t) ** 2 + q * 2 * (1 - t) * t + p1 * t * t
        assert cubic.point_at(t).almost_equal(quad, tolerance=1e-12)


def test_equal_curves_hash_equal():
    rng = random.Random(23)
    for _ in range(3_000):
        curve = _random_curve(rng, 1e3)
        jitter = P(rng.uniform(-1e-8, 1e-8), rng.uniform(-1e-8, 1e-8))
        other = CubicBezier(curve.p1 + jitter, curve.c1, curve.c2 - jitter, curve.p2)
        if curve == other:
            assert hash(curve) == hash(other)


def test_dataclass_protocol():
    assert CubicBezier.__match_args__ == ("p1", "c1", "c2", "p2")
    assert copy.deepcopy(ARCH) == ARCH
    assert pickle.loads(pickle.dumps(ARCH)) == ARCH
    assert hash(ARCH) == hash(CubicBezier(P(0, 0), P(1, 2), P(3, 2), P(4, 0)))
    assert dataclasses.replace(ARCH, p2=P(5, 0)).p2 == P(5, 0)
    match ARCH:
        case CubicBezier(p1, c1, c2, p2):
            assert (p1, c1, c2, p2) == (P(0, 0), P(1, 2), P(3, 2), P(4, 0))
    assert str(ARCH).startswith("CubicBezier((0.00000000, 0.00000000)")


# ----- evaluation ------------------------------------------------------------


def test_point_at_matches_de_casteljau():
    rng = random.Random(1)
    for _ in range(50):
        curve = _random_curve(rng)
        for t in (0.0, 0.13, 0.5, 0.77, 1.0):
            assert curve.point_at(t).almost_equal(_de_casteljau(curve, t), tolerance=1e-9)
    assert ARCH.point_at(0.0) is ARCH.p1
    assert ARCH.point_at(1.0) is ARCH.p2
    assert ARCH.midpoint == P(2, 1.5)


def test_derivatives_match_finite_differences():
    h = 1e-6
    for t in (0.1, 0.5, 0.9):
        d1 = ARCH.derivative1(t)
        fd1 = (ARCH.point_at(t + h) - ARCH.point_at(t - h)) / (2 * h)
        assert d1.almost_equal(fd1, tolerance=1e-4)
        d2 = ARCH.derivative2(t)
        fd2 = (ARCH.derivative1(t + h) - ARCH.derivative1(t - h)) / (2 * h)
        assert d2.almost_equal(fd2, tolerance=1e-3)


def test_tangents_and_fallbacks():
    assert ARCH.start_tangent.almost_equal(P(1, 2).unit)
    assert ARCH.end_tangent.almost_equal(P(1, -2).unit)
    assert ARCH.tangent_at(0.5).almost_equal(P(1, 0))
    assert angle_eq(ARCH.start_tangent_angle, math.atan2(2, 1))
    # Coincident control points fall back to the next point, so the tangent is never the zero vector.
    straightish = CubicBezier(P(0, 0), P(0, 0), P(0, 0), P(1, 1))
    assert straightish.start_tangent.almost_equal(P(1, 1).unit)
    assert straightish.end_tangent.almost_equal(P(1, 1).unit)
    assert angle_eq(straightish.start_tangent_angle, math.pi / 4)
    one_side = CubicBezier(P(0, 0), P(0, 0), P(1, 1), P(2, 0))
    assert one_side.start_tangent.almost_equal(P(1, 1).unit)
    cusp = CubicBezier(P(6, 2), P(8, 0.5), P(6, 0.5), P(8, 2))  # velocity vanishes at t = 0.5
    assert cusp.curvature_at(0.5) == 0.0
    assert cusp.tangent_at(0.5).almost_equal(cusp.chord.vector.unit)
    assert ARCH.curvature_at(0.5) < 0  # turning right (clockwise) at the top of the arch


def test_degenerate_curve():
    d = CubicBezier(P(1, 1), P(1, 1), P(1, 1 + 0.5e-8), P(1, 1))
    assert d.is_degenerate
    assert d.start_tangent_angle == 0.0
    assert d.tangent_at(0.5) == P(0, 0)
    assert d.biarc_approximation() == []
    assert d.length == pytest.approx(0.0, abs=1e-8)


def test_reversed_traces_the_same_shape_backwards():
    # Reversal keeps the shape: each control point swaps with its partner.
    rev = ARCH.reversed()
    assert rev == CubicBezier(P(4, 0), P(3, 2), P(1, 2), P(0, 0))
    assert rev.p1 == ARCH.p2
    assert rev.p2 == ARCH.p1
    for t in (0.1, 0.25, 0.5, 0.9):
        assert rev.point_at(t).almost_equal(ARCH.point_at(1 - t), tolerance=1e-12)
    assert rev.reversed() == ARCH
    assert rev != ARCH


# ----- structure ---------------------------------------------------------------


def test_subdivide_is_continuous_and_exact():
    a, b = ARCH.subdivide(0.3)
    assert a.p1 is ARCH.p1
    assert b.p2 is ARCH.p2
    assert a.p2 is b.p1
    assert a.p2.almost_equal(ARCH.point_at(0.3), tolerance=1e-12)
    for t in (0.2, 0.9):
        assert a.point_at(t).almost_equal(ARCH.point_at(0.3 * t), tolerance=1e-9)
        assert b.point_at(t).almost_equal(ARCH.point_at(0.3 + 0.7 * t), tolerance=1e-9)
    with pytest.raises(GeometryError):
        ARCH.subdivide(1.2)


def test_inflections_are_scale_independent():
    # Inflection parameters are affine invariants; scaling the curve must not change them.
    (t,) = S_CURVE.inflections()
    assert t == pytest.approx(0.5)
    assert ARCH.inflections() == ()
    two = CubicBezier(P(-0.7719, 0.6071), P(-0.1997, 8.4966), P(0.0168, 6.6305), P(-2.9215, 7.6570))
    roots = two.inflections()
    assert len(roots) == 2
    for factor in (1e-3, 1e-4, 1e3):
        scaled = CubicBezier(*(p * factor for p in (two.p1, two.c1, two.c2, two.p2)))
        assert scaled.inflections() == pytest.approx(roots, abs=1e-6)
    for t in roots:  # curvature really changes sign there
        assert math.copysign(1, two.curvature_at(t - 0.02)) != math.copysign(1, two.curvature_at(t + 0.02))
    assert len(LOOP.inflections()) == 2  # loop: two split parameters
    pieces = two.subdivide_inflections()
    assert len(pieces) == 3
    assert pieces[0].p1 is two.p1
    assert pieces[-1].p2 is two.p2
    assert pieces[0].p2 is pieces[1].p1
    assert pieces[1].p2 is pieces[2].p1
    assert pieces[1].p1.almost_equal(two.point_at(roots[0]), tolerance=1e-9)
    assert pieces[2].p1.almost_equal(two.point_at(roots[1]), tolerance=1e-9)


def test_find_extrema_handles_linear_axes_and_no_fabricated_roots():
    # An axis linear in t has one extremum candidate; a negative discriminant has none.
    assert CubicBezier(P(0, 0), P(1, 3), P(2, 3), P(3, 0)).find_extrema() == pytest.approx([0.5])
    curve = CubicBezier(P(0, 0), P(1 / 3, 1), P(1 / 6, 2), P(0.5, 0))
    extrema = curve.find_extrema()
    assert extrema == pytest.approx([1 / math.sqrt(3)], abs=1e-9)  # only the y extremum is real
    for t in extrema:
        d = curve.derivative1(t)
        assert min(abs(d.x), abs(d.y)) < 1e-9
    assert ARCH.find_extrema() == pytest.approx([0.5])


def test_bounding_box_is_tight_and_contains_samples():
    rng = random.Random(2)
    for _ in range(100):
        curve = _random_curve(rng)
        box = curve.bounding_box
        samples = [curve.point_at(i / 400) for i in range(401)]
        for p in samples:
            assert box.contains_point(p)
        tight = Box.from_points(samples)  # the sample grid can miss an extreme by a little, never exceed it
        assert 0.0 <= box.width - tight.width < 1e-3
        assert 0.0 <= box.height - tight.height < 1e-3
    assert ARCH.bounding_box == Box(P(0, 0), P(4, 1.5))


def test_bounding_box_contains_thin_curves():
    # A curve 1000 long and a millionth high still has a height: extrema are found per axis.
    thin = CubicBezier(P(0, 0), P(1000 / 3, 1e-6), P(2000 / 3, 1e-6), P(1000, 0))
    box = thin.bounding_box
    assert box.contains_point(thin.midpoint)
    assert box.height == pytest.approx(7.5e-7)
    tall = CubicBezier(P(0, 0), P(1e-6, 1000 / 3), P(1e-6, 2000 / 3), P(0, 1000))
    assert tall.bounding_box.contains_point(tall.midpoint)
    assert tall.bounding_box.width == pytest.approx(7.5e-7)
    for offset in (P(1e3, -1e3), P(1e6, 1e6)):
        moved = CubicBezier(*(q + offset for q in (thin.p1, thin.c1, thin.c2, thin.p2)))
        assert moved.bounding_box.contains_point(moved.midpoint)


# ----- length --------------------------------------------------------------------


def test_length_matches_fine_polyline_and_is_bounded():
    poly = sum(ARCH.point_at(i / 20000).distance(ARCH.point_at((i + 1) / 20000)) for i in range(20000))
    assert ARCH.length == pytest.approx(poly, rel=1e-4)
    assert ARCH.length_within(1e-7) == pytest.approx(poly, rel=1e-6)
    # Non-positive tolerances are rejected instead of recursing forever.
    with pytest.raises(GeometryError):
        ARCH.length_within(0.0)
    with pytest.raises(GeometryError):
        ARCH.length_within(-1.0)
    big = CubicBezier(*(p * 1e8 for p in (ARCH.p1, ARCH.c1, ARCH.c2, ARCH.p2)))
    assert big.length == pytest.approx(poly * 1e8, rel=1e-4)
    assert Line(P(0, 0), P(3, 4)).length == 5.0


# ----- intersections ------------------------------------------------------------------


def test_intersect_line_handles_every_degree():
    # Degree-elevated quadratics and symmetric arches (vanishing cubic term) are found like any cubic.
    quad = CubicBezier.from_quadratic(P(0, 0), P(5, 10), P(10, 0))
    pts = quad.intersect_line(Line(P(-1, 2), P(11, 2)))
    assert len(pts) == 2
    expected = sorted(((1 - math.sqrt(0.6)) / 2 * 10, (1 + math.sqrt(0.6)) / 2 * 10))
    assert sorted(p.x for p in pts) == pytest.approx(expected, abs=1e-12)
    assert all(p.y == pytest.approx(2.0, abs=1e-12) for p in pts)
    symmetric = CubicBezier(P(0, 0), P(3, 4), P(6, 4), P(9, 0))
    assert len(symmetric.intersect_line(Line(P(-1, 1), P(10, 1)))) == 2
    assert symmetric.intersect_line(Line(P(-1, 5), P(10, 5))) == []
    # Genuine cubic: three crossings of the S-curve's chord, endpoints included.
    chord_hits = S_CURVE.intersect_line(Line(P(-1, 0), P(7, 0)))
    assert sorted(p.x for p in chord_hits) == pytest.approx([0.0, 3.0, 6.0], abs=1e-12)
    assert ARCH.intersect_line(Line(P(1, 1), P(1, 1))) == []  # degenerate line


def test_intersect_line_is_exact_near_quadratic_curves():
    # x(t) = t, y(t) = 1e-7 t^3 + t^2 - t + 0.16: a cubic term too small for a closed-form
    # cubic solver to keep its precision, with crossings of y = 0 near t = 0.2 and t = 0.8.
    y0 = 0.16
    curve = CubicBezier(P(0, y0), P(1 / 3, y0 - 1 / 3), P(2 / 3, y0 - 1 / 3), P(1, y0 + 1e-7))
    axis = Line(P(-1, 0), P(2, 0))
    hits = curve.intersect_line(axis)
    assert len(hits) == 2
    for p, near in zip(hits, (0.2, 0.8), strict=True):
        assert abs(p.y) < 1e-14
        assert p.x == pytest.approx(near, abs=1e-6)  # the cubic term shifts the quadratic's roots by up to 1e-7
        assert abs(1e-7 * p.x**3 + p.x**2 - p.x + y0) < 1e-15  # x(t) = t, so p.x is the root itself
    assert len(curve.intersect_line(axis, on_line=True)) == 2
    # The same crossings when the whole picture sits far from the origin.
    shift = P(1e6, 1e6)
    moved = CubicBezier(*(q + shift for q in (curve.p1, curve.c1, curve.c2, curve.p2)))
    far_axis = Line(axis.p1 + shift, axis.p2 + shift)
    far_hits = moved.intersect_line(far_axis)
    assert len(far_hits) == 2
    assert all(far_axis.distance_to_point(p) < 1e-9 for p in far_hits)


def test_intersect_line_reports_tangencies_once():
    top = Line(P(-1, 1.5), P(5, 1.5))  # touches the arch's apex (2, 1.5)
    hits = ARCH.intersect_line(top)
    assert len(hits) == 1
    assert hits[0].almost_equal(P(2, 1.5))
    assert ARCH.intersect_line(Line(P(-1, 1.5 + 1e-6), P(5, 1.5 + 1e-6))) == []
    assert len(ARCH.intersect_line(Line(P(-1, 1.5 - 1e-6), P(5, 1.5 - 1e-6)))) == 2
    straight = CubicBezier(P(0, 0), P(1, 0), P(2, 0), P(3, 0))
    assert straight.intersect_line(Line(P(-5, 0), P(9, 0))) == [P(0, 0), P(3, 0)]  # lies along the line


def test_intersect_line_on_line_filter():
    # on_line restricts hits to the line segment.
    curve = CubicBezier(P(0, 0), P(3, 4), P(6, -4), P(9, 0))
    assert len(curve.intersect_line(Line(P(100, 1), P(101, 1)))) == 2  # infinite line
    assert curve.intersect_line(Line(P(100, 1), P(101, 1)), on_line=True) == []
    assert len(curve.intersect_line(Line(P(0, 1), P(3, 1)), on_line=True)) == 2  # x ~ 0.82 and 2.7
    assert len(curve.intersect_line(Line(P(0, 1), P(2, 1)), on_line=True)) == 1


def test_intersect_line_matches_brute_force_root_count():
    rng = random.Random(3)
    for _ in range(200):
        curve = _random_curve(rng)
        line = Line(P(rng.uniform(-10, 10), rng.uniform(-10, 10)), P(rng.uniform(-10, 10), rng.uniform(-10, 10)))
        if line.is_degenerate:
            continue
        sides = [line.which_side(curve.point_at(i / 2000)) for i in range(2001)]
        crossings = sum(1 for a, b in itertools.pairwise(sides) if a * b < 0)
        found = curve.intersect_line(line)
        assert abs(len(found) - crossings) <= 1  # tangencies and sample-grid touches
        for p in found:
            assert line.distance_to_point(p) < 1e-9


# ----- biarc approximation ---------------------------------------------------------------


def _check_contract(curve: CubicBezier, segments: list[Line | Arc], tolerance: float) -> None:
    assert segments, "a non-degenerate curve must produce segments"
    assert all(isinstance(s, (Line, Arc)) for s in segments)
    assert all(not s.is_degenerate for s in segments)
    assert segments[0].p1 is curve.p1 or segments[0].p1 == curve.p1
    assert segments[-1].p2 is curve.p2 or segments[-1].p2 == curve.p2
    for a, b in itertools.pairwise(segments):
        assert a.p2.almost_equal(b.p1)
        assert segments_are_g1(a, b), (a, b)  # the public check at its default tolerance
    # Two-sided Hausdorff: curve samples to the nearest segment and segment samples to the curve.
    worst = max(min(_segment_distance(s, curve.point_at(i / 400)) for s in segments) for i in range(401))
    assert worst <= tolerance * 1.05, worst
    assert curve.hausdorff_distance(segments, samples=64) <= tolerance * 1.05


def test_biarc_output_contract_on_random_curves():
    rng = random.Random(4)
    tol = 0.01
    for _ in range(80):
        curve = _random_curve(rng)
        segments = curve.biarc_approximation(tol, max_depth=10)
        _check_contract(curve, segments, tol)


def test_biarc_is_g1_at_default_tolerance_far_from_the_origin():
    # The joint construction runs relative to p1, so translation does not cost joint precision.
    rng = random.Random(9)
    curves = [ARCH, S_CURVE, LOOP, *(_random_curve(rng) for _ in range(5))]
    for offset in (P(1e3, -1e3), P(1e6, 1e6)):
        for curve in curves:
            moved = CubicBezier(*(q + offset for q in (curve.p1, curve.c1, curve.c2, curve.p2)))
            segments = moved.biarc_approximation(0.001)
            _check_contract(moved, segments, 0.001)
            assert segments[0].p1 is moved.p1
            assert segments[-1].p2 is moved.p2


def _feature_size(segment: Line | Arc) -> float:
    return segment.radius if isinstance(segment, Arc) else segment.length


def test_biarc_joint_precision_follows_the_coordinate_envelope():
    # The documented rule: a direction derived from a feature of size s at coordinate magnitude |x|
    # carries about 1e-16 * |x| / s radians of rounding. Joints between features larger than
    # 1e-8 * |x| meet the default EPSILON; smaller features meet the rounding allowance.
    rng = random.Random(5)
    magnitude = 1e6
    offset = P(magnitude, magnitude)
    small_features = 0
    for _ in range(40):
        curve = CubicBezier(*(P(rng.uniform(-10, 10), rng.uniform(-10, 10)) + offset for _ in range(4)))
        segments = curve.biarc_approximation(0.001)
        for a, b in itertools.pairwise(segments):
            size = min(_feature_size(a), _feature_size(b))
            if size >= 1e-8 * magnitude:
                assert segments_are_g1(a, b), (a, b)
            else:
                small_features += 1
                assert segments_are_g1(a, b, angle_tolerance=4e-16 * magnitude / size), (a, b)
    assert small_features > 0  # the allowance branch was exercised


def test_sub_epsilon_pieces_are_absorbed_not_dropped():
    # A curve 4e-7 across approximated at the EPSILON floor: pieces too short to be segments
    # are merged into their neighbours, never dropped, so the chain stays exactly connected.
    micro = CubicBezier(P(0, 0), P(1e-7, 2e-7), P(3e-7, 2e-7), P(4e-7, 0))
    for strict in (False, True):
        segments = micro.biarc_approximation(const.EPSILON, strict=strict)
        assert is_connected_chain(micro, segments)
    # A sweep limit that would need arcs shorter than EPSILON is refused explicitly.
    with pytest.raises(GeometryError):
        micro.biarc_approximation(const.EPSILON, max_arc_angle=0.01)
    rng = random.Random(11)
    for _ in range(60):
        size = 10 ** rng.uniform(-7.5, -5.5)
        tiny = CubicBezier(*(P(rng.uniform(-size, size), rng.uniform(-size, size)) for _ in range(4)))
        if tiny.is_degenerate:
            continue
        assert is_connected_chain(tiny, tiny.biarc_approximation(const.EPSILON, strict=False))
        try:
            strict_result = tiny.biarc_approximation(const.EPSILON)
        except ApproximationError:
            continue
        assert is_connected_chain(tiny, strict_result)


def test_non_degenerate_curves_never_vanish():
    # A hairpin whose ends coincide has no chord to fall back on: it is halved even at max_depth=0.
    hairpin = CubicBezier(P(0, 0), P(1, 0), P(1, 0), P(0, 0))
    assert not hairpin.is_degenerate
    assert is_connected_chain(hairpin, hairpin.biarc_approximation(max_depth=0, strict=False))
    assert is_connected_chain(hairpin, hairpin.biarc_approximation(0.001))
    # A hairpin folded within 0.9 EPSILON has no extent: degenerate, so [] is its documented answer,
    # even though its control polygon is 1.8 EPSILON long.
    folded = CubicBezier(P(0, 0), P(0.9e-8, 0), P(0.9e-8, 0), P(0, 0))
    assert folded.polygon_length > const.EPSILON
    assert folded.is_degenerate
    assert folded.biarc_approximation(const.EPSILON) == []
    # With extent exactly EPSILON it is not degenerate, yet nothing EPSILON long can span it: an explicit error.
    edge = CubicBezier(P(0, 0), P(1e-8, 0), P(1e-8, 0), P(0, 0))
    assert not edge.is_degenerate
    for strict in (False, True):
        with pytest.raises(ApproximationError):
            edge.biarc_approximation(const.EPSILON, strict=strict)
    # Between the two: either a connected chain or an explicit error, never [].
    rng = random.Random(13)
    for _ in range(40):
        size = 10 ** rng.uniform(-8.2, -7.0)
        tiny = CubicBezier(P(0, 0), P(rng.uniform(-size, size), rng.uniform(-size, size)), P(size, 0), P(0, 0))
        if tiny.is_degenerate:
            continue
        for strict in (False, True):
            try:
                chain = tiny.biarc_approximation(const.EPSILON, strict=strict)
            except ApproximationError:
                continue
            assert is_connected_chain(tiny, chain)


def test_cleanup_never_leaves_a_degenerate_tail():
    scaled = CubicBezier(*(P(x, y) * 1e-8 for x, y in ((2.3, -3.1), (-4.5, 3.9), (6.5, 5.7), (-4.2, 3.8))))
    chain = scaled.biarc_approximation(const.EPSILON)
    assert is_connected_chain(scaled, chain)
    rng = random.Random(17)
    for _ in range(80):
        pts = [P(rng.uniform(-8, 8), rng.uniform(-8, 8)) * 1e-8 for _ in range(4)]
        curve = CubicBezier(*pts)
        if curve.is_degenerate:
            continue
        try:
            chain = curve.biarc_approximation(const.EPSILON)
        except ApproximationError:
            continue
        assert is_connected_chain(curve, chain)


def test_collinear_curves_that_double_back_report_their_extent():
    # All control points on the x axis: the curve runs out to x = 2.896, back past its start and on to x = 1.
    curve = CubicBezier(*(P(x, 0) for x in (0, 10, -10, 1)))
    assert not curve.is_straight
    assert curve.intersect_line(Line(P(2, 0), P(2.5, 0)), on_line=True) == [P(2, 0), P(2.5, 0)]
    whole = curve.intersect_line(Line(P(2, 0), P(2.5, 0)))
    xs = [curve.point_at(i / 200_000).x for i in range(200_001)]
    assert whole[0].x == pytest.approx(min(xs), abs=1e-6)
    assert whole[1].x == pytest.approx(max(xs), abs=1e-6)
    assert curve.intersect_line(Line(P(2.8, 0), P(4, 0)), on_line=True)[1].x == pytest.approx(max(xs), abs=1e-6)
    assert curve.intersect_line(Line(P(3, 0), P(4, 0)), on_line=True) == []
    lifted = Line(P(2, 0.5e-8), P(2.5, 0.5e-8))  # still within EPSILON of the curve's line
    assert len(curve.intersect_line(lifted, on_line=True)) == 2


def test_biarc_merge_keeps_the_pair_when_the_merged_arc_is_invalid():
    # Two arcs whose radii and centers agree within EPSILON are merged only if the merged arc
    # satisfies the invariant; otherwise the valid pair stays. This curve has a cusp of turning
    # radius 2.8e-9, so exactly one joint is a corner (the documented exception).
    curve = CubicBezier(P(0, 0), P(-0.00051007, -0.00078653), P(-0.00034432, 0.00033176), P(0.000514, -0.00120766))
    segments = curve.biarc_approximation(1e-7)
    assert is_connected_chain(curve, segments)
    corners = [a for a, b in itertools.pairwise(segments) if not segments_are_g1(a, b)]
    assert len(corners) <= 1
    assert min(1 / abs(curve.curvature_at(i / 1000)) for i in range(1, 1000)) < const.EPSILON


def test_approximation_then_sweep_split_stays_connected_and_g1():
    segments = ARCH.biarc_approximation(0.001, max_arc_angle=0.05)
    assert is_connected_chain(ARCH, segments)
    assert all(abs(s.angle) <= 0.05 + 1e-12 for s in segments if isinstance(s, Arc))
    assert g1_everywhere(segments)
    _check_contract(ARCH, segments, 0.001)


def test_intersect_line_overlap_convention():
    # A straight curve along the line shares a stretch with it: the ends of that stretch are reported.
    straight = CubicBezier(P(0, 0), P(1, 0), P(2, 0), P(3, 0))
    assert straight.intersect_line(Line(P(1, 0), P(2, 0))) == [P(0, 0), P(3, 0)]
    assert straight.intersect_line(Line(P(1, 0), P(2, 0)), on_line=True) == [P(1, 0), P(2, 0)]
    assert straight.intersect_line(Line(P(2, 0), P(5, 0)), on_line=True) == [P(2, 0), P(3, 0)]
    assert straight.intersect_line(Line(P(5, 0), P(3, 0)), on_line=True) == [P(3, 0)]
    assert straight.intersect_line(Line(P(5, 0), P(6, 0)), on_line=True) == []
    # The same at an angle and away from the origin.
    turn = math.radians(37)
    shift = P(100, -50)
    pts = [P.from_polar(x, turn) + shift for x in (0, 1, 2, 3)]
    rotated = CubicBezier(*pts)
    inside = Line(P.from_polar(1, turn) + shift, P.from_polar(2, turn) + shift)
    hits = rotated.intersect_line(inside, on_line=True)
    assert len(hits) == 2
    assert hits[0].almost_equal(inside.p1)
    assert hits[1].almost_equal(inside.p2)


def test_intersections_lie_on_both_shapes_and_survive_operand_reversal():
    rng = random.Random(19)
    checked = 0
    for _ in range(150):
        curve = _random_curve(rng)
        line = Line(P(rng.uniform(-10, 10), rng.uniform(-10, 10)), P(rng.uniform(-10, 10), rng.uniform(-10, 10)))
        if line.is_degenerate:
            continue
        hits = curve.intersect_line(line, on_line=True)
        for p in hits:
            assert line.point_on_line(p, segment=True)
            assert min(curve.point_at(i / 4000).distance(p) for i in range(4001)) < 1e-2
        mirrored = curve.intersect_line(line.reversed(), on_line=True)
        assert len(mirrored) == len(hits)
        assert all(any(p.almost_equal(q, 1e-9) for q in mirrored) for p in hits)
        checked += len(hits)
    assert checked > 50


def test_output_is_valid_after_cleanup_splitting_and_translation():
    rng = random.Random(23)
    curves = [ARCH, S_CURVE, LOOP, *(_random_curve(rng) for _ in range(6))]
    for offset in (P(0, 0), P(-1e3, 2e3), P(1e6, 1e6)):
        for curve in curves:
            moved = CubicBezier(*(q + offset for q in (curve.p1, curve.c1, curve.c2, curve.p2)))
            chain = moved.biarc_approximation(0.01, max_arc_angle=math.pi / 4)
            assert is_connected_chain(moved, chain)
            assert all(abs(s.angle) <= math.pi / 4 + 1e-12 for s in chain if isinstance(s, Arc))
            assert g1_everywhere(chain)
            assert moved.hausdorff_distance(chain) <= 0.01 * 1.05


def test_hausdorff_distance_validates_the_sample_count():
    for bad in (0, -1):
        with pytest.raises(GeometryError):
            ARCH.hausdorff_distance([ARCH.chord], samples=bad)


def test_biarc_hausdorff_is_two_sided():
    # Curves bowing inside an arc count as much as curves bowing outside it.
    curve = CubicBezier(P(4.5604, 4.0235), P(-4.6461, 6.2415), P(-2.3519, -7.3951), P(-8.6814, -6.6127))
    segments = curve.biarc_approximation(0.01, max_depth=12)
    _check_contract(curve, segments, 0.01)
    c_curve = CubicBezier(P(0, 0), P(0, 8), P(1, 0.5), P(6, 0))
    _check_contract(c_curve, c_curve.biarc_approximation(0.001, max_depth=12), 0.001)


def test_biarc_keeps_g1_on_tiny_curves():
    # Small curved pieces stay arcs; only straight pieces become lines, so no kinks appear.
    tiny = CubicBezier(P(0, 0), P(0.002, 0), P(0.002, 0), P(0.002, 0.002))
    segments = tiny.biarc_approximation(0.001)
    assert all(isinstance(s, Arc) for s in segments)
    _check_contract(tiny, segments, 0.001)
    assert angle_eq(segments[0].start_tangent_angle, 0.0)
    assert angle_eq(segments[-1].end_tangent_angle, math.pi / 2)


def test_biarc_straight_curve_is_one_line():
    straight = CubicBezier(P(0, 0), P(1, 1), P(2, 2), P(3, 3))
    assert straight.biarc_approximation() == [Line(P(0, 0), P(3, 3))]
    uneven = CubicBezier(P(0, 0), P(0.5, 0), P(2.9, 0), P(3, 0))  # straight, unevenly spaced controls
    assert uneven.is_straight
    assert uneven.biarc_approximation() == [Line(P(0, 0), P(3, 0))]
    almost = CubicBezier(P(0, 0), P(1, 1e-6), P(2, -1e-6), P(3, 0))
    assert not almost.is_straight
    segments = almost.biarc_approximation(0.001)
    assert all(isinstance(s, Arc) for s in segments)  # not straight within EPSILON: arcs, still G1
    _check_contract(almost, segments, 0.001)


def test_biarc_max_arc_angle_and_parameters():
    segments = ARCH.biarc_approximation(0.001, max_depth=8, max_arc_angle=math.pi / 2)
    assert all(abs(s.angle) <= math.pi / 2 + 1e-9 for s in segments if isinstance(s, Arc))
    _check_contract(ARCH, segments, 0.001)
    semi = CubicBezier(P(-1, 0), P(-1, 4 / 3), P(1, 4 / 3), P(1, 0))  # close to a semicircle
    plain = semi.biarc_approximation(0.01)
    limited = semi.biarc_approximation(0.01, max_arc_angle=math.pi / 4)
    assert sum(abs(s.angle) for s in limited if isinstance(s, Arc)) == pytest.approx(
        sum(abs(s.angle) for s in plain if isinstance(s, Arc))
    )
    assert max(abs(s.angle) for s in limited if isinstance(s, Arc)) <= math.pi / 4 + 1e-9
    with pytest.raises(GeometryError):
        ARCH.biarc_approximation(0.1, max_depth=-1)
    coarse = ARCH.biarc_approximation(0.5, max_depth=0)
    assert 1 <= len(coarse) <= 2  # one biarc (possibly merged) with no subdivision


def test_biarc_splits_inflections_and_loops():
    segments = S_CURVE.biarc_approximation(0.001, max_depth=8)
    _check_contract(S_CURVE, segments, 0.001)
    directions = [s.direction for s in segments if isinstance(s, Arc)]
    assert 1 in directions
    assert -1 in directions
    loop_segments = LOOP.biarc_approximation(0.01, max_depth=10)
    _check_contract(LOOP, loop_segments, 0.01)


def test_hausdorff_distance_is_two_sided():
    circle_ish = CubicBezier(P(1, 0), P(1, 0.5523), P(0.5523, 1), P(0, 1))  # quarter circle approximation
    arc = Arc.from_sweep(P(1, 0), P(0, 1), 1.0, math.pi / 2)
    assert circle_ish.hausdorff_distance([arc]) < 3e-4
    assert circle_ish.hausdorff_distance([Line(P(1, 0), P(0, 1))]) == pytest.approx(1 - math.sqrt(0.5), abs=1e-3)
    # Segments that run far past the curve are measured too, not only the curve's distance to them.
    straight = CubicBezier(P(0, 0), P(1 / 3, 0), P(2 / 3, 0), P(1, 0))
    assert straight.hausdorff_distance([Line(P(0, 0), P(100, 0))]) == pytest.approx(99.0, abs=1e-6)
    assert straight.hausdorff_distance([Line(P(0, 0), P(1, 0))]) == pytest.approx(0.0, abs=1e-9)
    # Exact shared endpoints measure zero at any scale; the parameter search must not invent error.
    long = CubicBezier(P(0, 0), P(1e6 / 3, 0), P(2e6 / 3, 0), P(1e6, 0))
    assert long.hausdorff_distance([Line(P(0, 0), P(1e6, 0))]) < 1e-9
    assert circle_ish.hausdorff_distance([arc.subdivide(0.5)[0]]) > 0.5  # half the arc misses half the curve
    with pytest.raises(GeometryError):
        circle_ish.hausdorff_distance([])


def test_flattening_never_breaks_tangent_continuity():
    # A control point 1e-7 from p1 gives a 2.86 degree start tangent; the chord must not replace the piece.
    curve = CubicBezier(P(0, 0), P(1e-7, 5e-9), P(2, 0), P(3, 0))
    segments = curve.biarc_approximation()
    assert angle_eq(segments[0].start_tangent_angle, curve.start_tangent_angle, 1e-6)
    assert g1_everywhere(segments)
    coarse = S_CURVE.biarc_approximation(0.1, max_depth=2)  # even a coarse approximation keeps G1
    assert g1_everywhere(coarse)
    assert angle_eq(coarse[0].start_tangent_angle, S_CURVE.start_tangent_angle, 1e-6)
    assert angle_eq(coarse[-1].end_tangent_angle, S_CURVE.end_tangent_angle, 1e-6)


def test_small_valid_curves_approximate():
    small = CubicBezier(P(0, 0), P(0.001, 0.0015), P(0.003, 0.0015), P(0.004, 0))
    segments = small.biarc_approximation(1e-6, max_depth=8)
    assert segments
    assert g1_everywhere(segments)
    _check_contract(small, segments, 1e-6)


def test_sub_epsilon_cusp_is_the_only_corner():
    # At 1e-6 scale this curve has a hairpin whose turning radius (~1e-10) is far below EPSILON:
    # a corner at the library's resolution. Every other joint must still be tangent-continuous.
    tiny = CubicBezier(*(P(x, y) * 1e-6 for x, y in ((-71, -91), (-69, 51), (93, 58), (14, 46))))
    tip_t = min((0.78 + 0.05 * i / 4000 for i in range(4001)), key=lambda t: abs(1 / tiny.curvature_at(t)))
    assert 1 / abs(tiny.curvature_at(tip_t)) < const.EPSILON
    tip = tiny.point_at(tip_t)
    for strict in (False, True):
        pieces = tiny.biarc_approximation(1e-8, max_depth=10, strict=strict)
        corners = [
            a.p2
            for a, b in itertools.pairwise(pieces)
            if not angle_eq(a.end_tangent_angle, b.start_tangent_angle, 1e-6)
        ]
        assert len(corners) == 1
        assert corners[0].almost_equal(tip, tolerance=10 * const.EPSILON)
        assert angle_eq(pieces[0].start_tangent_angle, tiny.start_tangent_angle, 1e-6)
        assert angle_eq(pieces[-1].end_tangent_angle, tiny.end_tangent_angle, 1e-6)


def test_tolerance_floor_is_epsilon():
    nearly_straight = CubicBezier(P(0, 0), P(1, 1e-9), P(2, 1e-9), P(3, 0))
    assert nearly_straight.is_straight
    assert nearly_straight.biarc_approximation(const.EPSILON, strict=True) == [Line(P(0, 0), P(3, 0))]
    with pytest.raises(GeometryError):  # finer than EPSILON cannot be honoured, so it is refused
        nearly_straight.biarc_approximation(1e-12, strict=True)
    with pytest.raises(GeometryError):
        ARCH.biarc_approximation(0.0)


def test_strict_is_the_default_and_best_effort_is_opt_in():
    wild = CubicBezier(P(0, 0), P(10, 30), P(-20, 30), P(5, 0))
    with pytest.raises(ApproximationError):
        wild.biarc_approximation(1e-6, max_depth=0)
    best_effort = wild.biarc_approximation(1e-6, max_depth=0, strict=False)
    assert best_effort
    assert g1_everywhere(best_effort)  # even a failed tolerance keeps the tangent contract
    fine = wild.biarc_approximation(0.01, max_depth=12)
    assert wild.hausdorff_distance(fine, samples=200) <= 0.01 * 1.1  # sampled estimate; small slack


# ----- output ----------------------------------------------------------------------------


def test_to_svg_path():
    assert ARCH.to_svg_path() == "C 1,2 3,2 4,0"
    assert ARCH.to_svg_path(add_move=True) == "M 0,0 C 1,2 3,2 4,0"
    assert ARCH.to_svg_path(add_prefix=False, scale=2, precision=1) == "2,4 6,4 8,0"


def test_epsilon_governs_degeneracy(restore_epsilon: None):
    small = CubicBezier(P(0, 0), P(1e-6, 0), P(2e-6, 0), P(3e-6, 0))
    assert not small.is_degenerate
    const.set_epsilon(1e-4)
    assert small.is_degenerate
