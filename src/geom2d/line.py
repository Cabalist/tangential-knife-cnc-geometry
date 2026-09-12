"""Straight line segment."""

import math
from dataclasses import dataclass

from . import const, util
from .box import Box
from .errors import GeometryError
from .point import P, PointLike


@dataclass(frozen=True, slots=True)
class Line:
    """A directed straight segment from ``p1`` to ``p2``.

    Equality is field-wise on the ``EPSILON`` grid, so a line and its
    ``reversed()`` are different lines. A segment shorter than ``EPSILON``
    is *degenerate*: its direction is undefined, so ``angle`` and the
    tangent angles are ``0.0``, the tangent vectors are the zero vector,
    ``mu`` is ``0.0``, ``point_at`` is ``p1`` and ``shift``, ``extend`` and
    ``offset`` return the segment unchanged.

    ``offset(+d)`` moves the segment to the *left* of its direction of
    travel; this convention is shared with ``Arc``. The constructor takes
    ``P`` endpoints; :meth:`from_polar` accepts any point-like start.
    """

    p1: P
    p2: P

    # ----- construction -------------------------------------------------

    @classmethod
    def from_polar(cls, start: PointLike, length: float, angle: float) -> Line:
        """A segment starting at ``start`` with the given length and direction."""
        p1 = P.of(start)
        return cls(p1, p1 + P.from_polar(length, angle))

    # ----- derived values -----------------------------------------------

    @property
    def vector(self) -> P:
        """``p2 - p1``."""
        return P(self.p2.x - self.p1.x, self.p2.y - self.p1.y)

    @property
    def length(self) -> float:
        """Segment length."""
        return math.hypot(self.p2.x - self.p1.x, self.p2.y - self.p1.y)

    @property
    def is_degenerate(self) -> bool:
        """True if the segment is shorter than ``EPSILON``."""
        return self.length < const.EPSILON

    @property
    def angle(self) -> float:
        """Direction of travel in radians, ``(-pi, pi]``; ``0.0`` if degenerate."""
        if self.is_degenerate:
            return 0.0
        return math.atan2(self.p2.y - self.p1.y, self.p2.x - self.p1.x)

    @property
    def start_tangent_angle(self) -> float:
        """Tangent direction at ``p1`` (the segment's angle)."""
        return self.angle

    @property
    def end_tangent_angle(self) -> float:
        """Tangent direction at ``p2`` (the segment's angle)."""
        return self.angle

    @property
    def start_tangent(self) -> P:
        """Unit tangent at ``p1``; the zero vector if degenerate."""
        return P(0.0, 0.0) if self.is_degenerate else self.vector.unit

    @property
    def end_tangent(self) -> P:
        """Unit tangent at ``p2``; the zero vector if degenerate."""
        return P(0.0, 0.0) if self.is_degenerate else self.vector.unit

    @property
    def midpoint(self) -> P:
        """The point halfway along the segment."""
        return P((self.p1.x + self.p2.x) / 2.0, (self.p1.y + self.p2.y) / 2.0)

    @property
    def bounding_box(self) -> Box:
        """Axis-aligned bounding box of the two endpoints."""
        return Box(self.p1, self.p2)

    # ----- parametric ---------------------------------------------------

    def point_at(self, t: float) -> P:
        """The point at parameter ``t`` (``0`` is ``p1``, ``1`` is ``p2``); extrapolates outside."""
        if self.is_degenerate:
            return self.p1
        return P(self.p1.x + (self.p2.x - self.p1.x) * t, self.p1.y + (self.p2.y - self.p1.y) * t)

    def tangent_at(self, t: float) -> P:  # noqa: ARG002 - a line's tangent is constant
        """Unit tangent at parameter ``t`` (constant along a line); the zero vector if degenerate."""
        return self.start_tangent

    def mu(self, p: P) -> float:
        """Parameter of the perpendicular projection of ``p`` onto the infinite line.

        ``0`` at ``p1``, ``1`` at ``p2``; signed, so points behind ``p1`` give
        negative values. ``0.0`` if degenerate.
        """
        dx = self.p2.x - self.p1.x
        dy = self.p2.y - self.p1.y
        len2 = dx * dx + dy * dy
        if len2 < const.EPSILON2:
            return 0.0
        return ((p.x - self.p1.x) * dx + (p.y - self.p1.y) * dy) / len2

    def normal_projection_point(self, p: P, *, segment: bool = False) -> P:
        """The perpendicular projection of ``p`` onto the line.

        With ``segment=True`` the result is clamped to the segment.
        """
        t = self.mu(p)
        if segment:
            t = min(1.0, max(0.0, t))
        return self.point_at(t)

    def distance_to_point(self, p: P, *, segment: bool = False) -> float:
        """Distance from ``p`` to the infinite line, or to the segment with ``segment=True``."""
        t = self.mu(p)
        if segment:
            t = min(1.0, max(0.0, t))
        qx = self.p1.x + (self.p2.x - self.p1.x) * t
        qy = self.p1.y + (self.p2.y - self.p1.y) * t
        return math.hypot(p.x - qx, p.y - qy)

    def subdivide(self, t: float) -> tuple[Line, Line]:
        """Split the segment at parameter ``t`` into two segments.

        Raises:
            GeometryError: If ``t`` is outside ``[0, 1]``.
        """
        if not 0.0 <= t <= 1.0:
            raise GeometryError(f"subdivide parameter must be in [0, 1], got {t!r}")
        p = self.point_at(t)
        return (Line(self.p1, p), Line(p, self.p2))

    # ----- transformations ----------------------------------------------

    def reversed(self) -> Line:
        """The same segment travelled the other way."""
        return Line(self.p2, self.p1)

    def extend(self, amount: float, *, from_midpoint: bool = False) -> Line:
        """Lengthen (or shorten, for negative ``amount``) the segment.

        By default ``p2`` moves; with ``from_midpoint`` both ends move by
        ``amount / 2``. A degenerate segment is returned unchanged.

        Raises:
            GeometryError: If shortening would take the length below zero.
        """
        length = self.length
        if length < const.EPSILON:
            return self
        if amount < -length:
            raise GeometryError(f"cannot shorten a segment of length {length!r} by {-amount!r}")
        if from_midpoint:
            d = self.vector * (amount / (2.0 * length))
            return Line(self.p1 - d, self.p2 + d)
        return Line(self.p1, self.p2 + self.vector * (amount / length))

    def shift(self, distance: float) -> Line:
        """Translate the segment along its own direction (negative moves it backwards).

        A degenerate segment is returned unchanged.
        """
        length = self.length
        if length < const.EPSILON:
            return self
        d = self.vector * (distance / length)
        return Line(self.p1 + d, self.p2 + d)

    def offset(self, distance: float) -> Line:
        """Translate the segment perpendicular to itself; positive is to the left of travel.

        A degenerate segment is returned unchanged.
        """
        length = self.length
        if length < const.EPSILON or const.is_zero(distance):
            return self
        n = self.vector.normal() * (distance / length)
        return Line(self.p1 + n, self.p2 + n)

    # ----- relations ----------------------------------------------------

    def which_side(self, p: P) -> int:
        """``1`` if ``p`` is left of the direction of travel, ``-1`` if right, ``0`` if on the line.

        "On the line" means within ``EPSILON`` perpendicular distance, so the
        answer does not depend on the segment's length. A degenerate segment
        has no sides and returns ``0`` for every point.
        """
        v = self.vector
        length = v.length
        if length < const.EPSILON:
            return 0
        cross = v.cross(p - self.p1)
        if const.cross_is_zero(cross, length):
            return 0
        return 1 if cross > 0 else -1

    def same_side(self, a: P, b: P) -> bool:
        """True if ``a`` and ``b`` are on the same side of the line (a point on the line counts as either)."""
        return self.which_side(a) * self.which_side(b) >= 0

    def point_on_line(self, p: P, *, segment: bool = False) -> bool:
        """True if ``p`` is within ``EPSILON`` of the infinite line, or of the segment with ``segment=True``."""
        length = self.length
        if length < const.EPSILON:
            return self.p1.almost_equal(p)
        if not const.cross_is_zero(self.vector.cross(p - self.p1), length):
            return False
        if not segment:
            return True
        tol = const.EPSILON / length
        return -tol <= self.mu(p) <= 1.0 + tol

    def is_parallel(self, other: Line, *, inline: bool = False) -> bool:
        """True if the two segments are parallel (within ``EPSILON``); with ``inline`` also collinear.

        Degenerate segments are parallel to nothing.
        """
        len1 = self.length
        len2 = other.length
        if len1 < const.EPSILON or len2 < const.EPSILON:
            return False
        v1 = self.vector
        if not const.is_parallel(v1.cross(other.vector), len1, len2):
            return False
        if not inline:
            return True
        return const.cross_is_zero(v1.cross(other.p1 - self.p1), len1)

    def _intersection_params(self, other: Line, *, segment: bool) -> tuple[float, float] | None:
        """Parameters ``(mu_a, mu_b)`` of the intersection, or None.

        Parallel, non-collinear lines have no intersection. Collinear lines
        intersect everywhere: the reported point is ``p1`` for infinite lines
        and the start of the overlap for segments, or None if the segments
        do not overlap.
        """
        len_a = self.length
        len_b = other.length
        if len_a < const.EPSILON or len_b < const.EPSILON:
            return None
        va = self.vector
        vb = other.vector
        denom = va.cross(vb)
        if const.is_parallel(denom, len_a, len_b):
            if not const.cross_is_zero(va.cross(other.p1 - self.p1), len_a):
                return None
            return self._collinear_overlap(other, segment=segment)
        w = other.p1 - self.p1
        mu_a = w.cross(vb) / denom
        mu_b = w.cross(va) / denom
        if segment:
            tol_a = const.EPSILON / len_a
            tol_b = const.EPSILON / len_b
            if mu_a < -tol_a or mu_a > 1.0 + tol_a or mu_b < -tol_b or mu_b > 1.0 + tol_b:
                return None
        return (mu_a, mu_b)

    def _collinear_overlap(self, other: Line, *, segment: bool) -> tuple[float, float] | None:
        if not segment:
            return (0.0, other.mu(self.p1))
        t1 = self.mu(other.p1)
        t2 = self.mu(other.p2)
        lo, hi = min(t1, t2), max(t1, t2)
        tol = const.EPSILON / self.length
        if hi < -tol or lo > 1.0 + tol:
            return None
        mu_a = max(lo, 0.0)
        return (mu_a, other.mu(self.point_at(mu_a)))

    def intersection_mu(self, other: Line, *, segment: bool = False) -> float | None:
        """Parameter along this line of the intersection with ``other``, or None.

        By default the two infinite lines are intersected; ``segment=True``
        requires the point to lie on both segments, tested within ``EPSILON``
        as a distance. Collinear overlapping lines report the start of the
        overlap.
        """
        params = self._intersection_params(other, segment=segment)
        return None if params is None else params[0]

    def intersection(self, other: Line, *, segment: bool = False) -> P | None:
        """The intersection point with ``other``, or None (see :meth:`intersection_mu`)."""
        mu = self.intersection_mu(other, segment=segment)
        return None if mu is None else self.point_at(mu)

    def intersects(self, other: Line, *, segment: bool = False) -> bool:
        """True if the lines (or, with ``segment=True``, the segments) intersect or overlap."""
        return self._intersection_params(other, segment=segment) is not None

    def crosses(self, other: Line) -> bool:
        """True if the segments cross at a point strictly interior to both.

        Touching at an endpoint or overlapping collinearly is not a crossing.
        The relation is symmetric.
        """
        len_a = self.length
        len_b = other.length
        if len_a < const.EPSILON or len_b < const.EPSILON:
            return False
        va = self.vector
        vb = other.vector
        denom = va.cross(vb)
        if const.is_parallel(denom, len_a, len_b):
            return False
        w = other.p1 - self.p1
        mu_a = w.cross(vb) / denom
        mu_b = w.cross(va) / denom
        tol_a = const.EPSILON / len_a
        tol_b = const.EPSILON / len_b
        return tol_a < mu_a < 1.0 - tol_a and tol_b < mu_b < 1.0 - tol_b

    # ----- output -------------------------------------------------------

    def to_svg_path(
        self, *, scale: float = 1.0, precision: int | None = None, add_prefix: bool = True, add_move: bool = False
    ) -> str:
        """SVG path data for the segment.

        Args:
            scale: Coordinate multiplier.
            precision: Digits after the decimal point (defaults to the ``EPSILON`` precision).
            add_prefix: Emit the ``L`` command letter.
            add_move: Start with ``M p1``; otherwise the start point is emitted as an ``L`` (or bare) pair.
        """
        fmt = util.float_formatter(scale=scale, precision=precision)
        first = "M " if add_move else ("L " if add_prefix else "")
        second = "L " if add_prefix else ""
        return f"{first}{fmt(self.p1.x)},{fmt(self.p1.y)} {second}{fmt(self.p2.x)},{fmt(self.p2.y)}"

    def __str__(self) -> str:
        return f"Line({self.p1}, {self.p2})"
