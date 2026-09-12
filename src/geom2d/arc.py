"""Circular arc.

Orientation convention: ``angle`` is the signed sweep in radians, measured
counter-clockwise positive from ``p1`` to ``p2`` about ``center``. A
clockwise arc has a negative ``angle``. ``offset(+d)`` moves the arc to the
*left* of its direction of travel, the same convention as ``Line``.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import const, util
from .box import Box
from .errors import DegenerateGeometryError, GeometryError
from .point import P

if TYPE_CHECKING:
    from .line import Line


@dataclass(frozen=True, slots=True, eq=False)
class Arc:
    """A directed circular arc from ``p1`` to ``p2``.

    Every constructor path validates the geometry: ``radius`` must be
    positive, ``|angle|`` at most a full turn, both endpoints must lie on
    the circle of ``radius`` about ``center``, and sweeping ``p1`` by
    ``angle`` about ``center`` must land on ``p2``. Inconsistent input
    raises :class:`GeometryError`, so ``center - p1`` can be trusted as the
    arc's I/J offset. :meth:`from_sweep` computes the center for you.

    Equality is grid identity on every field (see ``P``), so an arc and
    its ``reversed()`` differ. A zero-sweep arc is *degenerate*: its
    tangent angles are ``0.0``, ``mu`` is ``0.0``, ``point_at`` is ``p1``
    and ``extend``/``offset`` return it unchanged.
    """

    p1: P
    p2: P
    radius: float
    angle: float
    center: P

    def __post_init__(self) -> None:
        if type(self.p1) is not P:
            object.__setattr__(self, "p1", P.of(self.p1))
        if type(self.p2) is not P:
            object.__setattr__(self, "p2", P.of(self.p2))
        if type(self.center) is not P:
            object.__setattr__(self, "center", P.of(self.center))
        radius = float(self.radius)
        angle = float(self.angle)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "angle", angle)
        if not (math.isfinite(radius) and radius > 0.0):
            raise GeometryError(f"arc radius must be positive and finite, got {self.radius!r}")
        if not math.isfinite(angle) or abs(angle) > const.TAU + const.EPSILON:
            raise GeometryError(f"arc sweep must be within one full turn, got {self.angle!r}")
        d1 = self.p1.distance(self.center)
        d2 = self.p2.distance(self.center)
        if not (const.float_eq(d1, radius) and const.float_eq(d2, radius)):
            raise GeometryError(f"arc endpoints are not on the circle: |p1-c|={d1!r}, |p2-c|={d2!r}, radius={radius!r}")
        expected_p2 = self.center + (self.p1 - self.center).rotate(angle)
        if not (const.float_eq(expected_p2.x, self.p2.x) and const.float_eq(expected_p2.y, self.p2.y)):
            raise GeometryError(
                f"sweeping p1 by angle={angle!r} about the center gives {expected_p2}, not p2={self.p2}"
            )

    # ----- construction -------------------------------------------------

    @classmethod
    def from_sweep(cls, p1: P, p2: P, radius: float, angle: float) -> Arc:
        """Create an arc from its endpoints, radius and signed sweep; the center is computed.

        Raises:
            DegenerateGeometryError: If ``p1`` and ``p2`` coincide (the center
                of a full circle cannot be inferred).
            GeometryError: If the chord is longer than the diameter or the
                inputs are otherwise inconsistent.
        """
        return cls(p1, p2, radius, angle, calc_center(p1, p2, radius, angle))

    @classmethod
    def from_two_points_and_tangent(cls, p1: P, tangent_point: P, p2: P, *, reverse: bool = False) -> Arc | None:
        """Create the arc through ``p1`` and ``p2`` whose tangent at ``p1`` points at ``tangent_point``.

        ``tangent_point`` is a point, not a vector: the tangent direction is
        ``tangent_point - p1``. Returns None when the arc is degenerate
        (coincident points) or would be a straight line (collinear points).
        With ``reverse`` the arc runs from ``p2`` back to ``p1``.
        """
        if p1.almost_equal(p2) or p1.almost_equal(tangent_point):
            return None
        angle = 2.0 * p1.angle2(tangent_point, p2)
        if const.is_zero(angle) or const.angle_eq(abs(angle), const.TAU):
            return None
        chord = p1.distance(p2)
        radius = abs(chord / (2.0 * math.sin(angle / 2.0)))
        if reverse:
            return cls.from_sweep(p2, p1, radius, -angle)
        return cls.from_sweep(p1, p2, radius, angle)

    # ----- private angular helpers --------------------------------------

    @property
    def _start_angle(self) -> float:
        return math.atan2(self.p1.y - self.center.y, self.p1.x - self.center.x)

    def _sweep_angle_to(self, p: P) -> float:
        """Unsigned angle swept from ``p1`` to the ray through ``p``, in ``[0, tau)``.

        Measured along the arc's direction of travel; a point just behind
        ``p1`` (within tolerance) reads as ``0.0`` rather than a full turn.
        """
        theta = (math.atan2(p.y - self.center.y, p.x - self.center.x) - self._start_angle) * self.direction
        theta %= const.TAU
        if theta >= const.TAU - self._angle_tolerance:
            theta = 0.0
        return theta

    @property
    def _angle_tolerance(self) -> float:
        """Angular tolerance equivalent to ``EPSILON`` as a distance along the circle."""
        return const.EPSILON / self.radius

    def _sweep_param(self, p: P) -> float:
        """Parameter of ``p``'s ray: 0 at ``p1``, 1 at ``p2``, above 1 beyond ``p2``."""
        sweep = abs(self.angle)
        if sweep < const.EPSILON:
            return 0.0
        return self._sweep_angle_to(p) / sweep

    def _in_sweep(self, p: P) -> bool:
        theta = self._sweep_angle_to(p)
        return theta <= abs(self.angle) + self._angle_tolerance

    def _angle_at(self, t: float) -> float:
        return self._start_angle + self.angle * t

    # ----- derived values -----------------------------------------------

    @property
    def direction(self) -> int:
        """``1`` for counter-clockwise, ``-1`` for clockwise."""
        return -1 if self.angle < 0 else 1

    @property
    def is_clockwise(self) -> bool:
        """True if the sweep is negative."""
        return self.angle < 0

    @property
    def length(self) -> float:
        """Arc length ``radius * |angle|``."""
        return self.radius * abs(self.angle)

    @property
    def is_degenerate(self) -> bool:
        """True if the arc length is below ``EPSILON``."""
        return self.length < const.EPSILON

    @property
    def is_full_circle(self) -> bool:
        """True if the sweep is a full turn (``p1`` coincides with ``p2``)."""
        return abs(self.angle) >= const.TAU - self._angle_tolerance

    @property
    def start_angle(self) -> float:
        """Direction from the center to ``p1`` in radians, ``(-pi, pi]``."""
        return self._start_angle

    @property
    def end_angle(self) -> float:
        """Direction from the center to ``p2`` in radians, ``(-pi, pi]``."""
        return math.atan2(self.p2.y - self.center.y, self.p2.x - self.center.x)

    @property
    def large_arc_flag(self) -> int:
        """SVG large-arc flag: ``1`` if the sweep exceeds half a turn."""
        return 1 if abs(self.angle) > math.pi else 0

    @property
    def sweep_flag(self) -> int:
        """SVG sweep flag: ``1`` for a positive (counter-clockwise) sweep."""
        return 1 if self.angle > 0 else 0

    @property
    def start_tangent(self) -> P:
        """Unit tangent at ``p1`` in the direction of travel; zero vector if degenerate."""
        if self.is_degenerate:
            return P(0.0, 0.0)
        return (self.p1 - self.center).normal().unit * self.direction

    @property
    def end_tangent(self) -> P:
        """Unit tangent at ``p2`` in the direction of travel; zero vector if degenerate."""
        if self.is_degenerate:
            return P(0.0, 0.0)
        return (self.p2 - self.center).normal().unit * self.direction

    @property
    def start_tangent_angle(self) -> float:
        """Direction of travel at ``p1`` in radians, ``(-pi, pi]``; ``0.0`` if degenerate."""
        if self.is_degenerate:
            return 0.0
        return self.start_tangent.angle

    @property
    def end_tangent_angle(self) -> float:
        """Direction of travel at ``p2`` in radians, ``(-pi, pi]``; ``0.0`` if degenerate."""
        if self.is_degenerate:
            return 0.0
        return self.end_tangent.angle

    @property
    def midpoint(self) -> P:
        """The point halfway along the arc."""
        return self.point_at(0.5)

    @property
    def height(self) -> float:
        """Sagitta: the largest distance between the arc and its chord, valid for any sweep."""
        return self.radius * (1.0 - math.cos(abs(self.angle) / 2.0))

    @property
    def bounding_box(self) -> Box:
        """Tight axis-aligned bounding box of the arc."""
        points = [self.p1, self.p2]
        r = self.radius
        c = self.center
        for extreme in (P(c.x + r, c.y), P(c.x, c.y + r), P(c.x - r, c.y), P(c.x, c.y - r)):
            if self._in_sweep(extreme):
                points.append(extreme)
        return Box.from_points(points)

    # ----- parametric ---------------------------------------------------

    def point_at(self, t: float) -> P:
        """The point at parameter ``t`` (``0`` is ``p1``, ``1`` is ``p2``); extrapolates around the circle."""
        if self.is_degenerate:
            return self.p1
        if t == 0.0:
            return self.p1
        if t == 1.0:
            return self.p2
        return self.center + P.from_polar(self.radius, self._angle_at(t))

    def point_at_angle(self, theta: float) -> P:
        """The point swept ``theta`` radians from ``p1`` along the direction of travel."""
        if self.is_degenerate:
            return self.p1
        return self.center + P.from_polar(self.radius, self._start_angle + self.direction * theta)

    def tangent_at(self, t: float) -> P:
        """Unit tangent in the direction of travel at parameter ``t``; zero vector if degenerate."""
        if self.is_degenerate:
            return P(0.0, 0.0)
        return P.from_polar(1.0, self._angle_at(t) + self.direction * math.pi / 2.0)

    def mu(self, p: P) -> float:
        """Parameter of the point on the arc's ray through ``p``: 0 at ``p1``, 1 at ``p2``.

        Values above 1 mean the ray lies beyond ``p2`` (continuing around the
        circle). Correct for any sweep, including more than half a turn.
        ``0.0`` if degenerate.
        """
        return self._sweep_param(p)

    def subdivide(self, t: float) -> tuple[Arc, Arc]:
        """Split at parameter ``t`` into two arcs sharing the center.

        Raises:
            GeometryError: If ``t`` is outside ``[0, 1]``.
        """
        if not 0.0 <= t <= 1.0:
            raise GeometryError(f"subdivide parameter must be in [0, 1], got {t!r}")
        p = self.point_at(t)
        return (
            Arc(self.p1, p, self.radius, self.angle * t, self.center),
            Arc(p, self.p2, self.radius, self.angle * (1.0 - t), self.center),
        )

    def subdivide_at_angle(self, theta: float) -> tuple[Arc, Arc]:
        """Split ``theta`` radians (unsigned) from ``p1`` along the direction of travel.

        Raises:
            GeometryError: If ``theta`` is outside ``[0, |angle|]``.
        """
        sweep = abs(self.angle)
        if sweep < const.EPSILON:
            raise GeometryError("cannot subdivide a degenerate arc by angle")
        if not 0.0 <= theta <= sweep:
            raise GeometryError(f"split angle must be in [0, {sweep!r}], got {theta!r}")
        return self.subdivide(theta / sweep)

    def subdivide_at_point(self, p: P) -> tuple[Arc, Arc]:
        """Split at the point on the arc nearest to ``p`` along its ray.

        Raises:
            GeometryError: If ``p``'s ray does not meet the arc.
        """
        t = self._sweep_param(p)
        if t > 1.0 + self._angle_tolerance / max(abs(self.angle), const.EPSILON):
            raise GeometryError(f"point {p} is not within the arc's sweep")
        return self.subdivide(min(t, 1.0))

    def subdivide_equal(self, n: int) -> list[Arc]:
        """Split into ``n`` arcs of equal sweep.

        Raises:
            GeometryError: If ``n`` is less than 1.
        """
        if n < 1:
            raise GeometryError(f"number of pieces must be at least 1, got {n!r}")
        if n == 1:
            return [self]
        step = self.angle / n
        points = [self.p1] + [self.point_at(i / n) for i in range(1, n)] + [self.p2]
        return [Arc(points[i], points[i + 1], self.radius, step, self.center) for i in range(n)]

    def split_max_sweep(self, max_angle: float = math.pi / 2.0) -> list[Arc]:
        """Split into equal arcs each with ``|angle| <= max_angle`` (defaults to a quarter turn).

        Raises:
            GeometryError: If ``max_angle`` is not positive.
        """
        if max_angle <= 0.0:
            raise GeometryError(f"max_angle must be positive, got {max_angle!r}")
        n = max(1, math.ceil(abs(self.angle) / max_angle - const.EPSILON))
        return self.subdivide_equal(n)

    # ----- transformations ----------------------------------------------

    def reversed(self) -> Arc:
        """The same arc travelled the other way (sweep negated)."""
        return Arc(self.p2, self.p1, self.radius, -self.angle, self.center)

    def extend(self, amount: float, *, from_midpoint: bool = False) -> Arc:
        """Lengthen (or shorten, for negative ``amount``) the arc along its circle.

        By default ``p2`` moves; with ``from_midpoint`` both ends move by
        ``amount / 2``. A degenerate arc is returned unchanged.

        Raises:
            GeometryError: If the result would have non-positive sweep or
                exceed a full turn.
        """
        if self.is_degenerate:
            return self
        sweep = abs(self.angle) + amount / self.radius
        if sweep <= 0.0:
            raise GeometryError(f"cannot shorten an arc of length {self.length!r} by {-amount!r}")
        if sweep > const.TAU + self._angle_tolerance:
            raise GeometryError("cannot extend an arc beyond a full turn")
        new_angle = self.direction * sweep
        if from_midpoint:
            start = self._start_angle - self.direction * (amount / (2.0 * self.radius))
            p1 = self.center + P.from_polar(self.radius, start)
            p2 = self.center + P.from_polar(self.radius, start + new_angle)
            return Arc(p1, p2, self.radius, new_angle, self.center)
        p2 = self.center + P.from_polar(self.radius, self._start_angle + new_angle)
        return Arc(self.p1, p2, self.radius, new_angle, self.center)

    def offset(self, distance: float) -> Arc:
        """Move the arc perpendicular to itself; positive is to the left of travel.

        A counter-clockwise arc shrinks toward its center for positive
        ``distance``; a clockwise one grows. The sweep and center are kept.
        A degenerate arc is returned unchanged.

        Raises:
            GeometryError: If the offset would collapse the arc onto or past its center.
        """
        if self.is_degenerate or const.is_zero(distance):
            return self
        new_radius = self.radius - self.direction * distance
        if new_radius < const.EPSILON:
            raise GeometryError(f"offset {distance!r} collapses an arc of radius {self.radius!r}")
        scale = new_radius / self.radius
        p1 = self.center + (self.p1 - self.center) * scale
        p2 = self.center + (self.p2 - self.center) * scale
        return Arc(p1, p2, new_radius, self.angle, self.center)

    # ----- relations ----------------------------------------------------

    def point_on_arc(self, p: P) -> bool:
        """True if ``p`` lies on the arc (within ``EPSILON``), including its endpoints."""
        if not const.float_eq(self.center.distance(p), self.radius):
            return False
        return self._in_sweep(p)

    def point_inside(self, p: P) -> bool:
        """True if ``p`` lies inside the sector swept by the arc (center to arc, inclusive)."""
        if self.center.distance(p) > self.radius + const.EPSILON:
            return False
        if p.almost_equal(self.center):
            return True
        return self._in_sweep(p)

    def normal_projection_point(self, p: P, *, segment: bool = False) -> P:
        """The point on the arc's circle radially in line with ``p``.

        With ``segment=True`` a projection outside the sweep is replaced by the
        nearer endpoint. ``p`` at the center projects to ``p1``.
        """
        v = p - self.center
        if v.is_zero:
            return self.p1
        q = self.center + v.unit * self.radius
        if segment and not self._in_sweep(q):
            return self.p1 if p.distance2(self.p1) <= p.distance2(self.p2) else self.p2
        return q

    def distance_to_point(self, p: P, *, segment: bool = False) -> float:
        """Distance from ``p`` to the arc's circle, or to the arc itself with ``segment=True``."""
        radial = abs(self.center.distance(p) - self.radius)
        if not segment or self._in_sweep(p) or p.almost_equal(self.center):
            return radial
        return min(p.distance(self.p1), p.distance(self.p2))

    def intersect_line(self, line: Line, *, on_arc: bool = False, on_line: bool = False) -> list[P]:
        """Intersections of the arc's circle with the line through ``line``.

        ``on_arc`` keeps only points on this arc's sweep; ``on_line`` keeps
        only points on the line *segment*. A degenerate line gives no
        intersections.
        """
        if line.is_degenerate:
            return []
        lp1 = line.p1 - self.center
        lp2 = line.p2 - self.center
        dx = lp2.x - lp1.x
        dy = lp2.y - lp1.y
        dr2 = dx * dx + dy * dy
        det = lp1.cross(lp2)
        r2 = self.radius * self.radius
        dsc = r2 * dr2 - det * det
        candidates: list[P] = []
        if const.is_zero_rel(dsc, r2 * dr2):
            candidates.append(line.normal_projection_point(self.center))
        elif dsc > 0.0:
            sgn = -1.0 if dy < 0 else 1.0
            root = math.sqrt(dsc)
            x1 = (det * dy + sgn * dx * root) / dr2
            x2 = (det * dy - sgn * dx * root) / dr2
            y1 = (-det * dx + abs(dy) * root) / dr2
            y2 = (-det * dx - abs(dy) * root) / dr2
            candidates.append(P(x1, y1) + self.center)
            candidates.append(P(x2, y2) + self.center)
        return [
            p
            for p in candidates
            if (not on_arc or self._in_sweep(p)) and (not on_line or line.point_on_line(p, segment=True))
        ]

    def intersect_arc(self, other: Arc, *, on_arc: bool = False) -> list[P]:
        """Intersections of the two arcs' circles; with ``on_arc`` only points on both sweeps."""
        points = intersect_circles(self.center, self.radius, other.center, other.radius)
        if not on_arc:
            return list(points)
        return [p for p in points if self._in_sweep(p) and other._in_sweep(p)]

    # ----- protocol -----------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Arc):
            return NotImplemented
        return (
            self.p1 == other.p1
            and self.p2 == other.p2
            and self.center == other.center
            and const.cell(self.radius) == const.cell(other.radius)
            and const.cell(self.angle) == const.cell(other.angle)
        )

    def __hash__(self) -> int:
        return hash((self.p1, self.p2, self.center, const.cell(self.radius), const.cell(self.angle)))

    def __str__(self) -> str:
        precision = const.EPSILON_PRECISION
        return (
            f"Arc({self.p1}, {self.p2}, r={self.radius:.{precision}f}, a={self.angle:.{precision}f}, c={self.center})"
        )

    def to_svg_path(
        self, *, scale: float = 1.0, precision: int | None = None, add_prefix: bool = True, add_move: bool = False
    ) -> str:
        """SVG elliptical-arc path data for this circular arc.

        See https://www.w3.org/TR/SVG11/paths.html#PathDataEllipticalArcCommands.
        """
        fmt = util.float_formatter(scale=scale, precision=precision)
        prefix = "A " if add_prefix or add_move else ""
        if add_move:
            prefix = f"M {fmt(self.p1.x)},{fmt(self.p1.y)} {prefix}"
        r = fmt(self.radius)
        return f"{prefix}{r},{r} 0 {self.large_arc_flag} {self.sweep_flag} {fmt(self.p2.x)},{fmt(self.p2.y)}"


def calc_center(p1: P, p2: P, radius: float, angle: float) -> P:
    """Center of the arc through ``p1`` and ``p2`` with the given radius and signed sweep.

    Raises:
        DegenerateGeometryError: If ``p1`` and ``p2`` coincide.
        GeometryError: If the chord is longer than the diameter.
    """
    chord = p1.distance(p2)
    if chord < const.EPSILON:
        raise DegenerateGeometryError(f"cannot infer an arc center from coincident endpoints {p1}")
    diameter = 2.0 * radius
    if chord > diameter and not const.float_eq(chord, diameter):
        raise GeometryError(f"chord {chord!r} is longer than the diameter {diameter!r}")
    mid = P((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0)
    if const.float_eq(chord, diameter):
        return mid
    ratio = diameter / chord
    t = math.sqrt(max(0.0, ratio * ratio - 1.0))
    sign = 1.0 if angle > 0 else -1.0
    if abs(angle) > math.pi:
        sign = -sign
    return P(mid.x + sign * ((p1.y - p2.y) / 2.0) * t, mid.y - sign * ((p1.x - p2.x) / 2.0) * t)


def intersect_circles(c1: P, r1: float, c2: P, r2: float) -> tuple[P, ...]:
    """Intersections of two circles.

    Returns two points if the circles intersect, one if they are tangent
    (externally or internally), and none if they are apart, nested, or
    coincident.

    See http://mathworld.wolfram.com/Circle-CircleIntersection.html.
    """
    d = c1.distance(c2)
    apart = d > r1 + r2 and not const.float_eq(d, r1 + r2)
    nested = d < abs(r1 - r2) and not const.float_eq(d, abs(r1 - r2))
    if const.is_zero(d) or apart or nested:
        return ()
    direction = (c2 - c1) / d
    if const.float_eq(d, r1 + r2):
        return (c1 + direction * r1,)
    if const.float_eq(d, abs(r1 - r2)):
        return (c1 + direction * (r1 if r1 > r2 else -r1),)
    a = (d * d - r2 * r2 + r1 * r1) / (2.0 * d)
    h2 = r1 * r1 - a * a
    if h2 < 0.0:
        return ()
    h = math.sqrt(h2)
    foot = c1 + direction * a
    side = direction.normal() * h
    return (foot + side, foot - side)
