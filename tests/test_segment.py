"""Segment protocol and path helpers."""

import dataclasses
import itertools
import math

import pytest

from geom2d import (
    GeometryError,
    P,
    Path,
    Segment,
    angle_eq,
    heading_change,
    nearest_vertex,
    path_bounding_box,
    path_is_closed,
    path_length,
    path_reversed,
    path_start_at,
    path_to_polyline,
    polyline_to_path,
    segments_are_g1,
    split_path,
    split_path_where,
)
from geom2d.arc import Arc
from geom2d.bezier import CubicBezier
from geom2d.box import Box
from geom2d.line import Line
from tests.helpers import XY, g1_everywhere

PI = math.pi


def rounded_rectangle(w: float = 10.0, h: float = 6.0, r: float = 1.0) -> list[Line | Arc]:
    """Four lines and four quarter arcs, counter-clockwise, closed. Two joints point leftward (angle +-pi)."""
    return [
        Line(P(r, 0), P(w - r, 0)),
        Arc.from_sweep(P(w - r, 0), P(w, r), r, PI / 2),
        Line(P(w, r), P(w, h - r)),
        Arc.from_sweep(P(w, h - r), P(w - r, h), r, PI / 2),
        Line(P(w - r, h), P(r, h)),
        Arc.from_sweep(P(r, h), P(0, h - r), r, PI / 2),
        Line(P(0, h - r), P(0, r)),
        Arc.from_sweep(P(0, r), P(r, 0), r, PI / 2),
    ]


class Wrapped:
    """A consumer-style segment: geometry plus extra data, satisfying the protocol structurally."""

    def __init__(self, geom: Line | Arc, tag: str) -> None:
        self.geom = geom
        self.tag = tag

    @property
    def p1(self) -> P:
        return self.geom.p1

    @property
    def p2(self) -> P:
        return self.geom.p2

    @property
    def length(self) -> float:
        return self.geom.length

    @property
    def is_degenerate(self) -> bool:
        return self.geom.is_degenerate

    @property
    def start_tangent_angle(self) -> float:
        return self.geom.start_tangent_angle

    @property
    def end_tangent_angle(self) -> float:
        return self.geom.end_tangent_angle

    @property
    def start_tangent(self) -> P:
        return self.geom.start_tangent

    @property
    def end_tangent(self) -> P:
        return self.geom.end_tangent

    @property
    def bounding_box(self) -> Box:
        return self.geom.bounding_box

    def reversed(self) -> Wrapped:
        return Wrapped(self.geom.reversed(), self.tag)

    def point_at(self, t: float) -> P:
        return self.geom.point_at(t)

    def tangent_at(self, t: float) -> P:
        return self.geom.tangent_at(t)

    def subdivide(self, t: float) -> tuple[Wrapped, Wrapped]:
        a, b = self.geom.subdivide(t)
        return (Wrapped(a, self.tag), Wrapped(b, self.tag))


@dataclasses.dataclass(frozen=True, slots=True)
class Cut:
    """A consumer-style frozen record embedding kernel geometry."""

    geom: Line | Arc
    depth: float


def test_frozen_records_embedding_geometry_are_hashable_and_matchable():
    cut = Cut(Line(P(0, 0), P(1, 0)), 2.0)
    same = Cut(Line(P(0, 0), P(1 + 1e-10, 0)), 2.0)
    assert cut == same
    assert hash(cut) == hash(same)
    assert {cut: "a"}[same] == "a"
    arc_cut = Cut(Arc.from_sweep(P(1, 0), P(0, 1), 1.0, PI / 2), 1.0)
    match arc_cut.geom:
        case Arc(p1, p2, radius, angle, center):
            assert (p1, p2, radius, center) == (P(1, 0), P(0, 1), 1.0, P(0, 0))
            assert angle == pytest.approx(PI / 2)
        case Line(p1, p2):
            pytest.fail("matched the wrong class")
    match cut.geom:
        case Arc():
            pytest.fail("matched the wrong class")
        case Line(p1, p2):
            assert (p1, p2) == (P(0, 0), P(1, 0))
    assert dataclasses.replace(cut, depth=3.0).geom is cut.geom


# ----- protocol conformance ---------------------------------------------------


def test_kernel_classes_satisfy_the_protocol():
    segments: list[Segment] = [
        Line(P(0, 0), P(1, 0)),
        Arc.from_sweep(P(1, 0), P(0, 1), 1.0, PI / 2),
        CubicBezier(P(0, 0), P(1, 2), P(3, 2), P(4, 0)),
    ]
    for seg in segments:
        assert seg.point_at(0.0) == seg.p1
        assert seg.point_at(1.0) == seg.p2
        assert seg.tangent_at(0.0).almost_equal(seg.start_tangent)
        assert seg.tangent_at(1.0).almost_equal(seg.end_tangent)
        assert angle_eq(seg.start_tangent.angle, seg.start_tangent_angle)
        assert seg.length > 0
        assert not seg.is_degenerate
        a, b = seg.subdivide(0.5)
        assert a.p2 == b.p1
        assert seg.bounding_box.contains_point(seg.point_at(0.3))
        rev = seg.reversed()
        assert rev.p1 == seg.p2
        assert rev.p2 == seg.p1
        assert angle_eq(rev.start_tangent_angle, math.atan2(-seg.end_tangent.y, -seg.end_tangent.x))


def test_helpers_are_structural_and_keep_the_concrete_type():
    path = [Wrapped(seg, "cut") for seg in rounded_rectangle()]
    as_path: Path = path  # a list of wrappers is a Path
    assert path_is_closed(as_path)
    assert path_length(path) == pytest.approx(2 * 8 + 2 * 4 + 2 * PI)
    assert path_bounding_box(path) == Box(P(0, 0), P(10, 6))
    assert g1_everywhere(path)
    # The helpers that return segments give back the wrapper type, so its extra data stays reachable.
    assert path_reversed(path)[0].tag == "cut"
    assert path_start_at(path, 3)[0].geom == path[3].geom
    pieces = split_path_where(path, lambda a, b: a.tag != b.tag)
    assert pieces == [path]
    assert split_path(path, [4])[1][0].geom == path[4].geom


# ----- whole-path values -------------------------------------------------------


def test_rounded_rectangle_round_trip_is_closed_and_g1():
    path = rounded_rectangle()
    assert path_is_closed(path)
    assert path_is_closed(iter(path))
    joints = [*itertools.pairwise(path), (path[-1], path[0])]
    assert len(joints) == 8
    for a, b in joints:
        assert segments_are_g1(a, b), (a, b)
    # The top edge runs leftward (angle pi), and the arc into it ends leftward too (+pi or -pi).
    top = path[4]
    assert angle_eq(top.angle, PI)
    assert abs(abs(path[3].end_tangent_angle) - PI) < 1e-9
    approximated: list[Segment] = []
    for seg in path:
        if isinstance(seg, Arc):
            bez = CubicBezier(seg.p1, seg.p1 + seg.start_tangent * 0.5523, seg.p2 - seg.end_tangent * 0.5523, seg.p2)
            approximated.extend(bez.biarc_approximation(1e-4, max_depth=6))
        else:
            approximated.append(seg)
    assert path_is_closed(approximated)
    assert g1_everywhere(approximated)


def test_offset_path_stays_closed_and_g1():
    # Offsetting every segment of a rounded rectangle by the same distance keeps the joints
    # shared and tangent-continuous: lines shift along their normals, arcs change radius.
    for distance in (0.3, -0.5):
        moved = [seg.offset(distance) for seg in rounded_rectangle()]
        assert path_is_closed(moved)
        assert g1_everywhere(moved)
        assert path_length(moved) == pytest.approx(2 * 8 + 2 * 4 + 2 * PI * (1 - distance))


def test_reversed_length_bbox():
    path = rounded_rectangle()
    rev = path_reversed(path)
    assert rev[0].p1 == path[-1].p2
    assert rev[-1].p2 == path[0].p1
    assert path_reversed(rev) == path
    assert path_length(rev) == pytest.approx(path_length(path))
    assert path_length([]) == 0.0
    assert not path_is_closed([])
    assert not path_is_closed([Line(P(0, 0), P(1, 0))])
    with pytest.raises(GeometryError):
        path_bounding_box([])


def test_polyline_conversions():
    assert polyline_to_path([]) == []
    assert polyline_to_path([(1, 1)]) == []  # a single node has no segments
    path = polyline_to_path([XY(0, 0), (1, 0), P(1, 1), (1, 1)])
    assert len(path) == 3
    assert path[2].is_degenerate  # the repeated closing point: nothing raised, caller may drop it
    assert [s for s in path if not s.is_degenerate] == [Line(P(0, 0), P(1, 0)), Line(P(1, 0), P(1, 1))]
    assert path_to_polyline(path[:2]) == [P(0, 0), P(1, 0), P(1, 1)]
    assert path_to_polyline([]) == []
    mixed: list[Segment] = [Line(P(0, 0), P(1, 0)), Arc.from_sweep(P(1, 0), P(2, 1), 1.0, PI / 2)]
    assert path_to_polyline(mixed) == [P(0, 0), P(1, 0), P(2, 1)]  # arc endpoints, not its center


def test_nearest_vertex():
    path = rounded_rectangle()
    assert nearest_vertex(path, P(9.1, 6.2)) == 4  # start of the top edge is (9, 6)
    assert nearest_vertex(path, P(10.2, 5.1)) == 3  # start of the top-right arc is (10, 5)
    assert nearest_vertex(path, P(-1, -1)) == 0
    assert nearest_vertex(path, P(1, -0.1)) == 0  # the closing point is vertex 0, not a ninth vertex
    open_path = polyline_to_path([(0, 0), (1, 0), (2, 0)])
    assert nearest_vertex(open_path, P(2.1, 0.1)) == 2  # an open path's final point is a vertex too
    assert nearest_vertex(open_path, P(0.9, 0)) == 1
    with pytest.raises(GeometryError):
        nearest_vertex([], P(0, 0))


# ----- joints -------------------------------------------------------------------


def test_heading_change_sign_and_wrap():
    east = Line(P(0, 0), P(1, 0))
    north = Line(P(1, 0), P(1, 1))
    south = Line(P(1, 0), P(1, -1))
    assert heading_change(east, north) == pytest.approx(PI / 2)  # left turn is positive
    assert heading_change(east, south) == pytest.approx(-PI / 2)
    assert heading_change(east, Line(P(1, 0), P(2, 0))) == 0.0
    # Across the seam: heading -170 degrees to +170 degrees is a 20 degree left turn, not -340.
    a = Line(P(0, 0), P.from_polar(1, math.radians(-170)))
    b = Line(a.p2, a.p2 + P.from_polar(1, math.radians(170)))
    assert heading_change(a, b) == pytest.approx(math.radians(-20))
    assert abs(heading_change(east, east.reversed())) == pytest.approx(PI)  # a U-turn


def test_segments_are_g1_with_explicit_tolerances():
    a = Line(P(0, 0), P(1, 0))
    b = Line(P(1, 0), P(2, 1e-5))
    assert not segments_are_g1(a, b)
    assert segments_are_g1(a, b, angle_tolerance=1e-4)
    gap = Line(P(1 + 1e-5, 0), P(2, 0))
    assert not segments_are_g1(a, gap)
    assert segments_are_g1(a, gap, point_tolerance=1e-4)
    # Scale 1e-4 and 1e4 without touching the global epsilon.
    for scale in (1e-4, 1e4):
        s1 = Line(P(0, 0), P(scale, 0))
        s2 = Line(P(scale, 0), P(2 * scale, scale * 1e-6))
        assert segments_are_g1(s1, s2, angle_tolerance=1e-5, point_tolerance=scale * 1e-6)
    # Tangent continuity across the +-pi seam.
    left1 = Line(P(2, 0), P(1, 0))
    left2 = Line(P(1, 0), P(0, -1e-13))
    assert segments_are_g1(left1, left2)
    assert not segments_are_g1(left1, left2.reversed())
    # A tighter tolerance than EPSILON is honoured: the heading change is not snapped first.
    slight = Line(P(1, 0), P(1 + math.cos(5e-9), math.sin(5e-9)))
    assert heading_change(a, slight) == pytest.approx(5e-9, rel=1e-6)
    assert segments_are_g1(a, slight)
    assert not segments_are_g1(a, slight, angle_tolerance=1e-12)


# ----- splitting and rotating -----------------------------------------------------


def test_split_path_by_indices():
    path = rounded_rectangle()
    pieces = split_path(path, [2, 5])
    assert [len(p) for p in pieces] == [2, 3, 3]
    assert pieces[1][0] is path[2]
    assert split_path(path, []) == [path]
    assert split_path(path, [3, 3]) == [path[:3], path[3:]]
    assert split_path([], [1]) == []
    for bad in (0, 8, -1):
        with pytest.raises(IndexError):
            split_path(path, [bad])


def test_split_path_where_on_open_and_closed_paths():
    square = polyline_to_path([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    corners = split_path_where(square, lambda a, b: abs(heading_change(a, b)) > PI / 4)
    assert [len(p) for p in corners] == [1, 1, 1, 1]  # every joint including the closing one is a corner
    rounded = rounded_rectangle()
    assert split_path_where(rounded, lambda a, b: not segments_are_g1(a, b)) == [rounded]  # no corners at all
    # A closed path whose closing joint is smooth but has interior corners wraps around.
    two_corners: list[Segment] = [
        Line(P(1, 0), P(10, 0)),
        Line(P(10, 0), P(10, 6)),  # corner at (10, 0)
        Line(P(10, 6), P(1, 6)),  # corner at (10, 6)
        rounded[5],  # arc (1,6) -> (0,5), smooth with the top edge
        Line(P(0, 5), P(0, 1)),
        rounded[7],  # arc (0,1) -> (1,0), smooth into the bottom edge: the closing joint is smooth
    ]
    assert path_is_closed(two_corners)
    pieces = split_path_where(two_corners, lambda a, b: abs(heading_change(a, b)) > PI / 4)
    assert len(pieces) == 2
    assert pieces[0] == [two_corners[1]]  # starts at the first corner
    assert pieces[1][0] is two_corners[2]
    assert pieces[1][-1] is two_corners[0]  # wraps through the smooth closing joint
    assert sum(len(p) for p in pieces) == len(two_corners)
    open_path = polyline_to_path([(0, 0), (1, 0), (2, 1), (3, 1)])
    assert [len(p) for p in split_path_where(open_path, lambda a, b: abs(heading_change(a, b)) > 0.1)] == [1, 1, 1]
    nothing: list[Line] = []
    assert split_path_where(nothing, lambda _a, _b: True) == []
    single = [Line(P(0, 0), P(1, 0))]
    assert split_path_where(single, lambda _a, _b: True) == [single]


def test_path_start_at():
    path = rounded_rectangle()
    rotated = path_start_at(path, 3)
    assert rotated[0] is path[3]
    assert rotated[-1] is path[2]
    assert path_is_closed(rotated)
    assert path_start_at(path, 0) == path
    with pytest.raises(IndexError):
        path_start_at(path, 8)
    with pytest.raises(GeometryError):
        path_start_at(path[:3], 1)
