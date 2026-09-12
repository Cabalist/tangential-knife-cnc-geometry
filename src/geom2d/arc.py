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
    from .point import PointLike


@dataclass(frozen=True, slots=True, eq=False)
class Arc:
    """A directed circular arc from ``p1`` to ``p2``.

    Every constructor path validates the geometry: ``radius`` must be
    positive, ``|angle|`` at most a full turn, both endpoints must lie on
    the circle of ``radius`` about ``center`` within ``EPSILON`` (an
    absolute distance), and sweeping ``p1`` by ``angle`` about ``center``
    must land within ``EPSILON`` of ``p2``. Inconsistent input raises
    :class:`GeometryError`, so ``center - p1`` can be trusted as the arc's
    I/J offset. :meth:`from_sweep` computes the center for you. The
    dataclass constructor takes ``P`` fields; the factory classmethods accept
    any :data:`~geom2d.point.PointLike`.

    Every query that asks whether a point is on the circle uses ``EPSILON``
    as an absolute distance, the same test as the constructor. The tangent
    at each end is derived from that end's stored point, so two arcs that
    share an endpoint report tangents there that depend only on their
    centers, not on the accumulated sweep.

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
        if not (const.is_zero(d1 - radius) and const.is_zero(d2 - radius)):
            raise GeometryError(f"arc endpoints are not on the circle: |p1-c|={d1!r}, |p2-c|={d2!r}, radius={radius!r}")
        expected_p2 = self.center + (self.p1 - self.center).rotate(angle)
        if not expected_p2.almost_equal(self.p2):
            raise GeometryError(
                f"sweeping p1 by angle={angle!r} about the center gives {expected_p2}, not p2={self.p2}"
            )

    # ----- construction -------------------------------------------------

    @classmethod
    def from_sweep(cls, p1: PointLike, p2: PointLike, radius: float, angle: float) -> Arc:
        """Create an arc from its endpoints, radius and signed sweep; the center is computed.

        Any non-zero sweep is accepted: a sweep of ``1e-9`` at radius ``1e6``
        is a valid arc of length ``0.001``. Whether the sweep, radius and
        chord fit together is decided by the constructor's invariant.

        Raises:
            DegenerateGeometryError: If ``p1`` and ``p2`` coincide (the center
                of a full circle cannot be inferred).
            GeometryError: If the sweep is zero, the chord is longer than the
                diameter, or the inputs are otherwise inconsistent.
        """
        a = P.of(p1)
        b = P.of(p2)
        return cls(a, b, radius, angle, calc_center(a, b, radius, angle))

    @classmethod
    def from_two_points_and_tangent(
        cls, p1: PointLike, tangent_point: PointLike, p2: PointLike, *, reverse: bool = False
    ) -> Arc | None:
        """Create the arc through ``p1`` and ``p2`` whose tangent at ``p1`` points at ``tangent_point``.

        ``tangent_point`` is a point, not a vector: the tangent direction is
        ``tangent_point - p1``. Returns None when the arc is degenerate
        (coincident points) or would be a straight line (the tangent runs
        along the chord, in either direction). With ``reverse`` the arc runs
        from ``p2`` back to ``p1``.
        """
        p1 = P.of(p1)
        p2 = P.of(p2)
        tangent_point = P.of(tangent_point)
        if p1.almost_equal(p2) or p1.almost_equal(tangent_point):
            return None
        angle = 2.0 * p1.angle2(tangent_point, p2)
        if const.angle_eq(angle, 0.0):
            return None
        chord = p1.distance(p2)
        radius = abs(chord / (2.0 * math.sin(angle / 2.0)))
        if reverse:
            return cls.from_sweep(p2, p1, radius, -angle)
        return cls.from_sweep(p1, p2, radius, angle)

    # ----- private angular helpers --------------------------------------

    def _sweep_angle_to(self, p: P) -> float:
        """Unsigned angle swept from ``p1`` to the ray through ``p``, in ``[0, tau)``.

        Measured along the arc's direction of travel; a point just behind
        ``p1`` (within tolerance) reads as ``0.0`` rather than a full turn.
        """
        theta = (math.atan2(p.y - self.center.y, p.x - self.center.x) - self.start_angle) * self.direction
        theta %= const.TAU
        if theta >= const.TAU - self._angle_tolerance:
            theta = 0.0
        return theta

    @property
    def _angle_tolerance(self) -> float:
        """Angular tolerance equivalent to ``EPSILON`` as a distance along the circle."""
        return const.EPSILON / self.radius

    def _in_sweep(self, p: P) -> bool:
        theta = self._sweep_angle_to(p)
        return theta <= abs(self.angle) + self._angle_tolerance

    def _angle_at(self, t: float) -> float:
        """Direction from the center to the point at ``t``; at either end it comes from the stored endpoint."""
        if t == 0.0:
            return self.start_angle
        if t == 1.0:
            return self.end_angle
        return self.start_angle + self.angle * t

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
        """True if the sweep is a full turn, so ``p1`` and ``p2`` coincide.

        :meth:`to_svg_path` writes such an arc as two half turns, because
        SVG drops an arc command whose endpoints are identical.
        """
        return abs(self.angle) >= const.TAU - self._angle_tolerance

    @property
    def start_angle(self) -> float:
        """Direction from the center to ``p1`` in radians, ``(-pi, pi]``."""
        return math.atan2(self.p1.y - self.center.y, self.p1.x - self.center.x)

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
        return self.tangent_at(0.0)

    @property
    def end_tangent(self) -> P:
        """Unit tangent at ``p2`` in the direction of travel; zero vector if degenerate."""
        return self.tangent_at(1.0)

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
        return self.center + P.from_polar(self.radius, self.start_angle + self.direction * theta)

    def tangent_at(self, t: float) -> P:
        """Unit tangent in the direction of travel at parameter ``t``; zero vector if degenerate.

        At ``t = 0`` and ``t = 1`` the tangent is perpendicular to the
        radius through the stored endpoint, so it matches the neighbouring
        segment's tangent at a shared point as closely as the two centers
        allow.
        """
        if self.is_degenerate:
            return P(0.0, 0.0)
        return P.from_polar(1.0, self._angle_at(t) + self.direction * math.pi / 2.0)

    def mu(self, p: P) -> float:
        """Parameter of the point on the arc's ray through ``p``: 0 at ``p1``, 1 at ``p2``.

        Values above 1 mean the ray lies beyond ``p2`` (continuing around the
        circle). Correct for any sweep, including more than half a turn.
        ``0.0`` if degenerate.
        """
        if self.is_degenerate:
            return 0.0
        return self._sweep_angle_to(p) / abs(self.angle)

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
        if self.is_degenerate:
            raise GeometryError("cannot subdivide a degenerate arc by angle")
        if not 0.0 <= theta <= sweep:
            raise GeometryError(f"split angle must be in [0, {sweep!r}], got {theta!r}")
        return self.subdivide(theta / sweep)

    def subdivide_at_point(self, p: P) -> tuple[Arc, Arc]:
        """Split at the point on the arc nearest to ``p`` along its ray.

        Raises:
            GeometryError: If ``p``'s ray does not meet the arc.
        """
        t = self.mu(p)
        if t > 1.0 + self._angle_tolerance / max(abs(self.angle), const.EPSILON):
            raise GeometryError(f"point {p} is not within the arc's sweep")
        return self.subdivide(min(t, 1.0))

    def subdivide_equal(self, n: int) -> list[Arc]:
        """Split into ``n`` arcs of equal sweep.

        Raises:
            GeometryError: If ``n`` is less than 1, or if the pieces would be
                shorter than ``EPSILON`` (a tiny arc cannot be split).
        """
        if n < 1:
            raise GeometryError(f"number of pieces must be at least 1, got {n!r}")
        if n == 1:
            return [self]
        if self.length / n < const.EPSILON:
            raise GeometryError(
                f"splitting an arc of length {self.length!r} into {n} pieces would make them degenerate"
            )
        step = self.angle / n
        points = [self.p1] + [self.point_at(i / n) for i in range(1, n)] + [self.p2]
        return [Arc(points[i], points[i + 1], self.radius, step, self.center) for i in range(n)]

    def split_max_sweep(self, max_angle: float = math.pi / 2.0) -> list[Arc]:
        """Split into equal arcs each with ``|angle| <= max_angle`` (defaults to a quarter turn).

        Raises:
            GeometryError: If ``max_angle`` is not positive, or if meeting it
                would need pieces shorter than ``EPSILON``.
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
            start = self.start_angle - self.direction * (amount / (2.0 * self.radius))
            p1 = self.center + P.from_polar(self.radius, start)
            p2 = self.center + P.from_polar(self.radius, start + new_angle)
            return Arc(p1, p2, self.radius, new_angle, self.center)
        p2 = self.center + P.from_polar(self.radius, self.start_angle + new_angle)
        return Arc(self.p1, p2, self.radius, new_angle, self.center)

    def offset(self, distance: float) -> Arc:
        """Move the arc perpendicular to itself; positive is to the left of travel.

        A counter-clockwise arc shrinks toward its center for positive
        ``distance``; a clockwise one grows. The sweep and center are kept
        and the new endpoints are constructed on the new circle. A degenerate
        arc is returned unchanged.

        Raises:
            GeometryError: If the offset would collapse the arc onto or past its center.
        """
        if self.is_degenerate or const.is_zero(distance):
            return self
        new_radius = self.radius - self.direction * distance
        if new_radius < const.EPSILON:
            raise GeometryError(f"offset {distance!r} collapses an arc of radius {self.radius!r}")
        # Endpoints are placed on the new circle from the sweep, so the slack the invariant
        # allowed in this arc's endpoints is not scaled up with the radius.
        start = self.start_angle
        p1 = self.center + P.from_polar(new_radius, start)
        p2 = self.center + P.from_polar(new_radius, start + self.angle)
        return Arc(p1, p2, new_radius, self.angle, self.center)

    # ----- relations ----------------------------------------------------

    def point_on_arc(self, p: P) -> bool:
        """True if ``p`` lies on the arc (within ``EPSILON`` of the circle), including its endpoints."""
        if not const.is_zero(self.center.distance(p) - self.radius):
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

        A line whose distance from the center is within ``EPSILON`` of the
        radius is tangent and meets the circle once, at the foot of the
        perpendicular from the center. ``on_arc`` keeps only points on this
        arc's sweep; ``on_line`` keeps only points on the line *segment*. The
        points come in order along this arc's direction of travel from
        ``p1`` (continuing around the circle for points off the sweep). A
        degenerate line gives no intersections.
        """
        if line.is_degenerate:
            return []
        foot = line.normal_projection_point(self.center)
        h = foot.distance(self.center)
        r = self.radius
        if const.is_zero(h - r):
            candidates = [foot]
        elif h > r:
            candidates: list[P] = []
        else:
            step = line.vector * (math.sqrt(r * r - h * h) / line.length)
            candidates = [foot - step, foot + step]
        kept = [
            p
            for p in candidates
            if (not on_arc or self._in_sweep(p)) and (not on_line or line.point_on_line(p, segment=True))
        ]
        return sorted(kept, key=self.mu)

    def intersect_arc(self, other: Arc, *, on_arc: bool = False) -> list[P]:
        """Intersections of the two arcs' circles, in order along this arc.

        With ``on_arc`` only points on both sweeps are kept. Two arcs on the
        same circle (centers and radii within ``EPSILON``) have no isolated
        circle intersections, so without ``on_arc`` they give none; with
        ``on_arc`` the ends of the portion they share are returned: one point
        where they only touch, two where they overlap, none where they are
        apart. An end of the shared portion is an endpoint of one arc that
        lies on the other arc (within ``EPSILON`` of its circle as well as
        inside its sweep); a full circle has no ends, so its arbitrary start
        point is never reported.
        """
        same_circle = const.is_zero(self.center.distance(other.center)) and const.is_zero(self.radius - other.radius)
        if same_circle:
            if not on_arc:
                return []
            ends: list[P] = []
            for arc in (self, other):
                if not arc.is_full_circle:
                    ends.extend((arc.p1, arc.p2))
            shared: list[P] = []
            for p in sorted(ends, key=self.mu):
                # Two candidates within 2 * EPSILON of each other, both on both arcs, are one end.
                if (
                    self.point_on_arc(p)
                    and other.point_on_arc(p)
                    and not any(p.almost_equal(q, 2.0 * const.EPSILON) for q in shared)
                ):
                    shared.append(p)
            return shared
        points = intersect_circles(self.center, self.radius, other.center, other.radius)
        if on_arc:
            points = tuple(p for p in points if self._in_sweep(p) and other._in_sweep(p))
        return sorted(points, key=self.mu)

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

        A full circle is written as two half turns through the point
        opposite ``p1``: SVG treats an arc command whose endpoints coincide
        as if it were omitted. Without ``add_prefix`` the second half is a
        further parameter set of the same command.

        See https://www.w3.org/TR/SVG11/paths.html#PathDataEllipticalArcCommands.
        """
        fmt = util.float_formatter(scale=scale, precision=precision)
        prefix = "A " if add_prefix or add_move else ""
        if add_move:
            prefix = f"M {fmt(self.p1.x)},{fmt(self.p1.y)} {prefix}"
        r = fmt(self.radius)
        end = f"{fmt(self.p2.x)},{fmt(self.p2.y)}"
        if self.is_full_circle:
            opposite = self.point_at(0.5)
            joiner = " A " if add_prefix or add_move else " "
            return f"{prefix}{r},{r} 0 0 {self.sweep_flag} {fmt(opposite.x)},{fmt(opposite.y)}{joiner}{r},{r} 0 0 {self.sweep_flag} {end}"
        return f"{prefix}{r},{r} 0 {self.large_arc_flag} {self.sweep_flag} {end}"


def calc_center(p1: P, p2: P, radius: float, angle: float) -> P:
    """Center of the arc through ``p1`` and ``p2`` with the given radius and signed sweep.

    The center sits ``radius * cos(angle / 2)`` from the chord's midpoint
    along the chord's left normal for a positive sweep (right for a negative
    one; a major arc's cosine is negative, which puts the center on the far
    side). The distance comes from the supplied radius and sweep, not from
    the chord, so the rounding of the endpoints is not amplified for a
    shallow arc, and it is exact at a half turn. Only an exactly zero sweep
    is refused here; whether the sweep, radius and chord fit together is
    decided by the ``Arc`` invariant.

    Raises:
        DegenerateGeometryError: If ``p1`` and ``p2`` coincide.
        GeometryError: If the sweep is zero or not finite, or the chord is
            longer than the diameter.
    """
    chord_vector = p2 - p1
    chord = chord_vector.length
    if chord < const.EPSILON:
        raise DegenerateGeometryError(f"cannot infer an arc center from coincident endpoints {p1}")
    if not math.isfinite(angle) or angle == 0.0:
        raise GeometryError(f"a sweep of {angle!r} cannot join distinct endpoints {p1} and {p2}")
    if chord > 2.0 * radius + const.EPSILON:
        raise GeometryError(f"chord {chord!r} is longer than the diameter {2.0 * radius!r}")
    mid = P((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0)
    height = radius * math.cos(angle / 2.0) * (1.0 if angle > 0.0 else -1.0)
    return mid + chord_vector.normal() * (height / chord)


def intersect_circles(c1: P, r1: float, c2: P, r2: float) -> tuple[P, ...]:
    """Intersections of two circles.

    Returns two points if the circles intersect, one if they are tangent
    within ``EPSILON`` (externally or internally), and none if they are
    apart, nested, or concentric. All tests are absolute distances.

    See http://mathworld.wolfram.com/Circle-CircleIntersection.html.
    """
    d = c1.distance(c2)
    if const.is_zero(d):
        return ()
    direction = (c2 - c1) / d
    if const.is_zero(d - (r1 + r2)):
        return (c1 + direction * r1,)
    if const.is_zero(d - abs(r1 - r2)):
        return (c1 + direction * (r1 if r1 > r2 else -r1),)
    if d > r1 + r2 or d < abs(r1 - r2):
        return ()
    a = (d * d - r2 * r2 + r1 * r1) / (2.0 * d)
    foot = c1 + direction * a
    h2 = r1 * r1 - a * a
    if h2 <= 0.0:
        return (foot,)
    side = direction.normal() * math.sqrt(h2)
    return (foot + side, foot - side)
