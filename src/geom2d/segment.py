"""The segment protocol and path helpers.

A *segment* is any object with the structural interface below: ``Line``,
``Arc`` and ``CubicBezier`` satisfy it, and so can a consumer's own wrapper
type. A *path* is a sequence of segments; helpers here never dispatch on
concrete classes.
"""

import itertools
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Protocol, Self

from . import const, util
from .box import Box
from .errors import GeometryError
from .line import Line
from .point import P

if TYPE_CHECKING:
    from .point import PointLike


class Segment(Protocol):
    """Structural interface shared by ``Line``, ``Arc`` and ``CubicBezier``."""

    @property
    def p1(self) -> P: ...

    @property
    def p2(self) -> P: ...

    @property
    def length(self) -> float: ...

    @property
    def is_degenerate(self) -> bool: ...

    @property
    def start_tangent_angle(self) -> float: ...

    @property
    def end_tangent_angle(self) -> float: ...

    @property
    def start_tangent(self) -> P: ...

    @property
    def end_tangent(self) -> P: ...

    @property
    def bounding_box(self) -> Box: ...

    def reversed(self) -> Self: ...

    def point_at(self, t: float) -> P: ...

    def tangent_at(self, t: float) -> P: ...

    def subdivide(self, t: float) -> tuple[Self, Self]: ...


type Path = Sequence[Segment]
"""A path is a sequence of segments, consecutive segments sharing an endpoint."""


# ----- whole-path values -----------------------------------------------------


def path_reversed(path: Path) -> list[Segment]:
    """The same path travelled the other way: every segment reversed, in reverse order."""
    return [segment.reversed() for segment in reversed(path)]


def path_length(path: Iterable[Segment]) -> float:
    """Sum of the segment lengths."""
    return sum(segment.length for segment in path)


def path_bounding_box(path: Iterable[Segment]) -> Box:
    """Union of the segments' bounding boxes.

    Raises:
        GeometryError: If the path is empty.
    """
    return Box.from_path(path)


def path_is_closed(path: Iterable[Segment]) -> bool:
    """True if the last segment ends where the first begins (within ``EPSILON``).

    Accepts any iterable, consumed once; an empty path is not closed.
    """
    first: Segment | None = None
    last: Segment | None = None
    for segment in path:
        if first is None:
            first = segment
        last = segment
    if first is None or last is None:
        return False
    return last.p2.almost_equal(first.p1)


def polyline_to_path(points: Iterable[PointLike]) -> list[Line]:
    """Lines joining consecutive points; fewer than two points give an empty path.

    Consecutive coincident points produce degenerate lines, which the caller
    can drop with ``is_degenerate``.
    """
    converted = [P.of(point) for point in points]
    return [Line(a, b) for a, b in itertools.pairwise(converted)]


def path_to_polyline(path: Path) -> list[P]:
    """The segment endpoints in order: every ``p1`` and the final ``p2``; ``[]`` for an empty path."""
    if not path:
        return []
    return [segment.p1 for segment in path] + [path[-1].p2]


def nearest_vertex(path: Path, p: P) -> int:
    """Index of the segment whose start point is nearest to ``p``.

    Raises:
        GeometryError: If the path is empty.
    """
    if not path:
        raise GeometryError("an empty path has no vertices")
    return min(range(len(path)), key=lambda i: path[i].p1.distance2(p))


# ----- joints -----------------------------------------------------------------


def heading_change(seg1: Segment, seg2: Segment) -> float:
    """Signed turn at the joint from ``seg1`` to ``seg2``, in ``[-pi, pi]``.

    Positive is a counter-clockwise (left) turn; wrap-safe across the ``+-pi``
    seam; ``0.0`` where the tangent directions agree.
    """
    return util.calc_rotation(seg1.end_tangent_angle, seg2.start_tangent_angle)


def segments_are_g1(
    seg1: Segment, seg2: Segment, *, angle_tolerance: float | None = None, point_tolerance: float | None = None
) -> bool:
    """True if ``seg2`` continues ``seg1`` with tangent continuity.

    G0: ``seg1.p2`` coincides with ``seg2.p1`` within ``point_tolerance``
    (a distance, default ``EPSILON``). G1: the heading change at the joint
    is zero within ``angle_tolerance`` (radians, default ``EPSILON``).
    """
    if not seg1.p2.almost_equal(seg2.p1, point_tolerance):
        return False
    return const.is_zero(heading_change(seg1, seg2), angle_tolerance)


# ----- splitting and rotating -------------------------------------------------


def split_path(path: Path, indices: Iterable[int]) -> list[list[Segment]]:
    """Split before each segment index in ``indices``.

    Index ``i`` names the joint between segments ``i - 1`` and ``i``, so valid
    values are ``1`` to ``len(path) - 1``; duplicates are ignored. An empty
    path gives ``[]``; no indices give ``[list(path)]``.

    Raises:
        IndexError: If an index is not a joint of the path.
    """
    segments = list(path)
    if not segments:
        return []
    cuts = sorted(set(indices))
    for i in cuts:
        if not 1 <= i < len(segments):
            raise IndexError(f"joint index {i} is out of range for a path of {len(segments)} segments")
    bounds = [0, *cuts, len(segments)]
    return [segments[a:b] for a, b in itertools.pairwise(bounds)]


def split_path_where(path: Path, predicate: Callable[[Segment, Segment], bool]) -> list[list[Segment]]:
    """Split at every joint where ``predicate(before, after)`` is true.

    For a closed path the closing joint (last segment to first) is tested
    too. If it splits, the path simply opens there; if it does not but
    other joints do, the pieces wrap around so no piece straddles the
    closing joint artificially. A closed path with no splitting joint is
    returned whole.
    """
    segments = list(path)
    if len(segments) < 2:
        return [segments] if segments else []
    interior = [i for i in range(1, len(segments)) if predicate(segments[i - 1], segments[i])]
    if path_is_closed(segments) and not predicate(segments[-1], segments[0]) and interior:
        rotated = path_start_at(segments, interior[0])
        shifted = [i - interior[0] for i in interior[1:]]
        return split_path(rotated, shifted)
    return split_path(segments, interior)


def path_start_at(path: Path, index: int) -> list[Segment]:
    """Rotate a closed path so that segment ``index`` comes first.

    Raises:
        GeometryError: If the path is not closed.
        IndexError: If ``index`` is out of range.
    """
    segments = list(path)
    if not path_is_closed(segments):
        raise GeometryError("only a closed path can be restarted at another vertex")
    if not 0 <= index < len(segments):
        raise IndexError(f"segment index {index} is out of range for a path of {len(segments)} segments")
    return segments[index:] + segments[:index]
