"""Cubic Bézier curve and its biarc approximation."""

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import const, util
from .arc import Arc
from .box import Box
from .errors import ApproximationError, GeometryError
from .line import Line
from .point import P, PointLike

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_MAX_LENGTH_DEPTH = 24
"""Recursion cap for :meth:`CubicBezier.length`; 2**24 subdivisions is far past float resolution."""


@dataclass(frozen=True, slots=True)
class CubicBezier:
    """A cubic Bézier curve ``p1 -> c1 -> c2 -> p2``.

    Equality is field-wise on the ``EPSILON`` grid; ``reversed()`` gives a
    different curve object tracing the same shape backwards. The constructor
    takes ``P`` fields; :meth:`from_quadratic` accepts any point-like input. A curve whose
    four control points all lie within ``EPSILON`` of each other is
    *degenerate* (a folded hairpin with a long control polygon but no
    extent counts): its tangent angles are ``0.0`` and
    ``biarc_approximation`` returns no segments.
    """

    p1: P
    c1: P
    c2: P
    p2: P

    # ----- construction -------------------------------------------------

    @classmethod
    def from_quadratic(cls, p1: PointLike, control: PointLike, p2: PointLike) -> CubicBezier:
        """Degree-elevate a quadratic Bézier ``p1 -> control -> p2`` to a cubic (exact).

        Accepts any point-like input, such as a parser's own point type.
        """
        a = P.of(p1)
        q = P.of(control)
        b = P.of(p2)
        return cls(a, a + (q - a) * (2.0 / 3.0), b + (q - b) * (2.0 / 3.0), b)

    # ----- derived values -----------------------------------------------

    @property
    def polygon_length(self) -> float:
        """Length of the control polygon ``p1-c1-c2-p2``, an upper bound on the arc length."""
        return self.p1.distance(self.c1) + self.c1.distance(self.c2) + self.c2.distance(self.p2)

    @property
    def is_degenerate(self) -> bool:
        """True if the control points' bounding box is smaller than ``EPSILON`` across.

        The curve lies inside that box, so no segment of length ``EPSILON``
        or more can represent it.
        """
        xs = (self.p1.x, self.c1.x, self.c2.x, self.p2.x)
        ys = (self.p1.y, self.c1.y, self.c2.y, self.p2.y)
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        return dx * dx + dy * dy < const.EPSILON2

    @property
    def length(self) -> float:
        """Arc length to the default tolerance (see :meth:`length_within`)."""
        return self.length_within()

    @property
    def chord(self) -> Line:
        """The straight segment from ``p1`` to ``p2``."""
        return Line(self.p1, self.p2)

    @property
    def flatness(self) -> float:
        """Convex-hull flatness: the larger distance of the control points from the chord segment."""
        chord = self.chord
        return max(chord.distance_to_point(self.c1, segment=True), chord.distance_to_point(self.c2, segment=True))

    @property
    def midpoint(self) -> P:
        """The point at ``t = 0.5``."""
        return self.point_at(0.5)

    @property
    def start_tangent(self) -> P:
        """Unit tangent at ``p1``; falls back through the control points when they coincide with ``p1``."""
        if self.is_degenerate:
            return P(0.0, 0.0)
        for target in (self.c1, self.c2, self.p2):
            v = target - self.p1
            if not v.is_zero:
                return v.unit
        return P(0.0, 0.0)

    @property
    def end_tangent(self) -> P:
        """Unit tangent at ``p2`` in the direction of travel; falls back through the control points."""
        if self.is_degenerate:
            return P(0.0, 0.0)
        for source in (self.c2, self.c1, self.p1):
            v = self.p2 - source
            if not v.is_zero:
                return v.unit
        return P(0.0, 0.0)

    @property
    def start_tangent_angle(self) -> float:
        """Direction of travel at ``p1`` in radians; ``0.0`` if degenerate."""
        if self.is_degenerate:
            return 0.0
        return self.start_tangent.angle

    @property
    def end_tangent_angle(self) -> float:
        """Direction of travel at ``p2`` in radians; ``0.0`` if degenerate."""
        if self.is_degenerate:
            return 0.0
        return self.end_tangent.angle

    @property
    def bounding_box(self) -> Box:
        """Tight axis-aligned bounding box (endpoints plus the axis extrema)."""
        points = [self.p1, self.p2] + [self.point_at(t) for t in self.find_extrema()]
        return Box.from_points(points)

    @property
    def is_straight(self) -> bool:
        """True if the curve is a straight segment: control points within ``EPSILON`` of the chord and both end tangents along it.

        Such a curve can be replaced by its chord without changing any tangent.
        """
        chord = self.chord
        if chord.is_degenerate or self.flatness > const.EPSILON:
            return False
        direction = chord.angle
        return const.angle_eq(self.start_tangent_angle, direction) and const.angle_eq(self.end_tangent_angle, direction)

    # ----- evaluation ---------------------------------------------------

    def point_at(self, t: float) -> P:
        """The point at parameter ``t`` in ``[0, 1]`` (extrapolates outside)."""
        if t == 0.0:
            return self.p1
        if t == 1.0:
            return self.p2
        mt = 1.0 - t
        a = mt * mt * mt
        b = 3.0 * mt * mt * t
        c = 3.0 * mt * t * t
        d = t * t * t
        return P(
            a * self.p1.x + b * self.c1.x + c * self.c2.x + d * self.p2.x,
            a * self.p1.y + b * self.c1.y + c * self.c2.y + d * self.p2.y,
        )

    def derivative1(self, t: float) -> P:
        """First derivative (velocity) at ``t``."""
        mt = 1.0 - t
        a = 3.0 * mt * mt
        b = 6.0 * mt * t
        c = 3.0 * t * t
        return P(
            a * (self.c1.x - self.p1.x) + b * (self.c2.x - self.c1.x) + c * (self.p2.x - self.c2.x),
            a * (self.c1.y - self.p1.y) + b * (self.c2.y - self.c1.y) + c * (self.p2.y - self.c2.y),
        )

    def derivative2(self, t: float) -> P:
        """Second derivative (acceleration) at ``t``."""
        mt = 1.0 - t
        return P(
            6.0 * (mt * (self.c2.x - 2.0 * self.c1.x + self.p1.x) + t * (self.p2.x - 2.0 * self.c2.x + self.c1.x)),
            6.0 * (mt * (self.c2.y - 2.0 * self.c1.y + self.p1.y) + t * (self.p2.y - 2.0 * self.c2.y + self.c1.y)),
        )

    def tangent_at(self, t: float) -> P:
        """Unit tangent at ``t``.

        At the ends, coincident control points are skipped; where the
        velocity vanishes elsewhere (a cusp) the chord direction is returned.
        The zero vector is returned only for a degenerate curve.
        """
        if self.is_degenerate:
            return P(0.0, 0.0)
        if t <= 0.0:
            return self.start_tangent
        if t >= 1.0:
            return self.end_tangent
        d1 = self.derivative1(t)
        if d1.is_zero:
            return self.chord.vector.unit
        return d1.unit

    def curvature_at(self, t: float) -> float:
        """Signed curvature at ``t`` (positive turns left); ``0.0`` where the velocity vanishes."""
        d1 = self.derivative1(t)
        speed2 = d1.length2
        if speed2 < const.EPSILON2:
            return 0.0
        return d1.cross(self.derivative2(t)) / math.pow(speed2, 1.5)

    # ----- parametric structure -------------------------------------------

    def subdivide(self, t: float) -> tuple[CubicBezier, CubicBezier]:
        """Split at parameter ``t`` (de Casteljau); the pieces share the split point exactly.

        Raises:
            GeometryError: If ``t`` is outside ``[0, 1]``.
        """
        if not 0.0 <= t <= 1.0:
            raise GeometryError(f"subdivide parameter must be in [0, 1], got {t!r}")
        mt = 1.0 - t
        d01 = self.p1 * mt + self.c1 * t
        d12 = self.c1 * mt + self.c2 * t
        d23 = self.c2 * mt + self.p2 * t
        d012 = d01 * mt + d12 * t
        d123 = d12 * mt + d23 * t
        split = d012 * mt + d123 * t
        return (CubicBezier(self.p1, d01, d012, split), CubicBezier(split, d123, d23, self.p2))

    def inflections(self) -> tuple[float, ...]:
        """Parameters in ``(0, 1)`` at which the curve should be split for approximation.

        These are the inflection points (where curvature changes sign) when
        the curve has them. A curve with a loop has none; for it the two
        parameters that isolate the loop are returned instead. The test is
        scale-independent.
        """
        v1 = self.c1 - self.p1
        v2 = self.c2 - self.c1 - v1
        v3 = self.p2 - self.c2 - v1 - v2 * 2.0
        a = v2.cross(v3)
        b = v1.cross(v3)
        c = v1.cross(v2)
        scale = (v1.length + v2.length + v3.length) ** 2
        if scale < const.EPSILON2:
            return ()
        roots: list[float] = []
        if const.is_zero_rel(a, scale):
            if not const.is_zero_rel(b, scale):
                roots.append(-c / b)
        else:
            disc = b * b - 4.0 * a * c
            if const.is_zero_rel(disc, scale * scale):
                roots.append(-b / (2.0 * a))
            else:
                root = math.sqrt(abs(disc))
                roots.extend(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
        margin = const.EPSILON
        valid = sorted(t for t in roots if margin < t < 1.0 - margin)
        return tuple(valid)

    def subdivide_inflections(self) -> tuple[CubicBezier, ...]:
        """Split at :meth:`inflections`; one to three curves."""
        params = self.inflections()
        if not params:
            return (self,)
        pieces: list[CubicBezier] = []
        rest = self
        consumed = 0.0
        for t in params:
            local = (t - consumed) / (1.0 - consumed)
            head, rest = rest.subdivide(local)
            pieces.append(head)
            consumed = t
        pieces.append(rest)
        return tuple(pieces)

    def find_extrema(self) -> list[float]:
        """Parameters in ``(0, 1)`` where ``x`` or ``y`` reaches a local extreme, sorted.

        Each axis is solved on its own with no scale-relative zero test, so a
        curve that is long in one direction and thin in the other still
        reports the thin direction's extremes and its bounding box contains
        the whole curve.
        """
        # Coefficients of the derivative  a t^2 + b t + c  per axis.
        va = (self.p2 - self.p1 + (self.c1 - self.c2) * 3.0) * 3.0
        vb = (self.p1 - self.c1 * 2.0 + self.c2) * 6.0
        vc = (self.c1 - self.p1) * 3.0
        params: list[float] = []
        for a, b, c in ((va.x, vb.x, vc.x), (va.y, vb.y, vc.y)):
            params.extend(_quadratic_roots(a, b, c))
        return sorted(t for t in params if 0.0 < t < 1.0)

    # ----- length ----------------------------------------------------------

    def length_within(self, tolerance: float | None = None) -> float:
        """Arc length by Gravesen's control-polygon bisection.

        Args:
            tolerance: Stop subdividing when the polygon and chord lengths agree
                within this distance. Defaults to ``1e-4`` of the control polygon
                length (at least ``EPSILON``).

        Raises:
            GeometryError: If ``tolerance`` is not positive.
        """
        if tolerance is None:
            tolerance = max(const.EPSILON, 1e-4 * self.polygon_length)
        elif not tolerance > 0.0:
            raise GeometryError(f"length tolerance must be positive, got {tolerance!r}")
        return self._length(tolerance, 0)

    def _length(self, tolerance: float, depth: int) -> float:
        l1 = self.polygon_length
        l0 = self.p1.distance(self.p2)
        if depth < _MAX_LENGTH_DEPTH and l1 - l0 > tolerance:
            a, b = self.subdivide(0.5)
            return a._length(tolerance, depth + 1) + b._length(tolerance, depth + 1)
        return 0.5 * l0 + 0.5 * l1

    # ----- intersections -----------------------------------------------------

    def intersect_line(self, line: Line, *, on_line: bool = False) -> list[P]:
        """Points where the curve meets the infinite line through ``line``, in curve order.

        A point counts when it is within ``EPSILON`` of the line, so a curve
        that starts on the line, or touches it without crossing, reports
        that point once. A curve whose four control points are within
        ``EPSILON`` of the line lies along it (it may double back) and
        overlaps it: the two ends of the shared portion are returned, in
        order along the line, which is the curve's extent along the line
        clipped to the line segment when ``on_line`` is set (one point where
        they only touch, none where they do not overlap). Otherwise
        ``on_line`` keeps only points on the line *segment*. Every degree is
        solved the same way, by bisection between the polynomial's critical
        points, so a degree-elevated quadratic or a symmetric arch is located
        as exactly as a full cubic. A degenerate line gives no intersections.
        """
        if line.is_degenerate:
            return []
        if all(line.point_on_line(q) for q in (self.p1, self.c1, self.c2, self.p2)):
            return self._overlap_with_line(line, on_line=on_line)
        v = line.vector
        # cross(v, curve(t) - line.p1) is a cubic in t; divided by |v| it is the signed distance from the line.
        a3 = self.p2 - self.p1 + (self.c1 - self.c2) * 3.0
        a2 = (self.p1 - self.c1 * 2.0 + self.c2) * 3.0
        a1 = (self.c1 - self.p1) * 3.0
        a0 = self.p1 - line.p1
        roots = _cubic_roots_in_unit_interval(
            v.cross(a3), v.cross(a2), v.cross(a1), v.cross(a0), const.EPSILON * line.length
        )
        points = [self.point_at(t) for t in roots]
        if on_line:
            points = [p for p in points if line.point_on_line(p, segment=True)]
        return points

    def _overlap_with_line(self, line: Line, *, on_line: bool) -> list[P]:
        """Ends of the portion a curve lying along ``line`` shares with it, in order along the line."""
        length = line.length
        u = line.vector / length
        s0, s1, s2, s3 = ((q - line.p1).dot(u) for q in (self.p1, self.c1, self.c2, self.p2))
        # The position along the line is a cubic in t; its range over [0, 1] is the curve's extent.
        positions = [s0, s3]
        for t in _quadratic_roots(3.0 * (s3 - s0 + 3.0 * (s1 - s2)), 6.0 * (s0 - 2.0 * s1 + s2), 3.0 * (s1 - s0)):
            if 0.0 < t < 1.0:
                mt = 1.0 - t
                positions.append(mt * mt * mt * s0 + 3.0 * mt * mt * t * s1 + 3.0 * mt * t * t * s2 + t * t * t * s3)
        lo = min(positions)
        hi = max(positions)
        if on_line:
            lo = max(lo, 0.0)
            hi = min(hi, length)
            if lo > hi + const.EPSILON:
                return []
            hi = max(hi, lo)
        start = line.p1 + u * lo
        end = line.p1 + u * hi
        return [start] if start.almost_equal(end) else [start, end]

    # ----- biarc approximation ------------------------------------------------

    def biarc_approximation(
        self,
        tolerance: float = 0.001,
        *,
        max_depth: int = 8,
        max_arc_angle: float | None = None,
        strict: bool = True,
    ) -> list[Line | Arc]:
        """Approximate the curve with tangent-continuous circular arcs (and lines where it is straight).

        The curve is first split at its inflections, then each piece is
        approximated by a biarc; a piece is subdivided while the estimated
        two-sided Hausdorff distance to its biarc exceeds ``tolerance``, up
        to ``max_depth`` halvings. The construction runs relative to ``p1``,
        so the joint precision does not depend on where the curve sits in
        the plane; the result carries the coordinate rounding described in
        :mod:`geom2d.const`.

        Output contract: a list of ``Line`` and ``Arc`` segments with no
        degenerate segment; the first segment starts exactly at ``p1`` and
        the last ends exactly at ``p2``; consecutive segments share their
        endpoint exactly; tangent directions agree at every joint within
        ``angle_eq`` (a ``Line`` is emitted only where the piece is straight:
        control points within ``EPSILON`` of the chord and both end tangents
        along it), with one geometric exception: where the curve itself turns
        through a region smaller than ``EPSILON`` (a cusp or near-cusp, turning
        radius below ``EPSILON``) the output has a corner at that point, because
        no segment of length ``EPSILON`` or more can carry the turn. The
        distance check is an estimate: 16 curve samples per piece are measured
        to the segments and 7 samples per segment to the curve, so a peak
        between samples can exceed ``tolerance`` by a small amount. It is not
        a proof; :meth:`hausdorff_distance` measures a result more finely.
        With ``strict`` (the default) a piece that still fails the check after
        ``max_depth`` halvings raises :class:`ApproximationError`; with
        ``strict=False`` its best biarc is returned instead. Pieces of the
        curve shorter than ``EPSILON`` cannot be segments: they are absorbed
        by their neighbours, which are re-anchored so the chain stays exactly
        connected (such a feature is a corner at the library's resolution),
        and in strict mode the merged result is checked again against
        ``tolerance``. A piece whose ends coincide (a loop or hairpin) is
        halved even when ``max_depth`` is spent, because no single segment
        can represent it. That handling is a robustness measure: features
        that small lie far below any process floor and are normally dropped
        by the caller first (see the resolution floor in :mod:`geom2d.const`);
        ``tolerance`` itself should come from the process. A degenerate
        curve gives ``[]``; a non-degenerate curve always gives at least one
        segment or raises.

        Args:
            tolerance: Maximum allowed distance between curve and arcs.
            max_depth: Maximum number of halvings per inflection-free piece.
            max_arc_angle: If given, every arc is split into equal pieces so
                that ``|angle| <= max_arc_angle``.
            strict: Raise :class:`ApproximationError` when some piece could not
                meet ``tolerance`` within ``max_depth``; otherwise return the
                best effort silently.

        Raises:
            GeometryError: If ``tolerance`` is below ``EPSILON`` (nothing in the library
                resolves finer than that), ``max_depth`` is negative, or
                ``max_arc_angle`` could only be met with arcs shorter than
                ``EPSILON`` (see :meth:`Arc.split_max_sweep`).
            ApproximationError: With ``strict``, if the tolerance could not be met;
                in either mode, if no segment at least ``EPSILON`` long can
                represent a non-degenerate curve.
        """
        if not tolerance >= const.EPSILON:
            raise GeometryError(f"tolerance must be at least EPSILON ({const.EPSILON!r}), got {tolerance!r}")
        if max_depth < 0:
            raise GeometryError(f"max_depth must be at least 0, got {max_depth!r}")
        if self.is_degenerate:
            return []
        # Relative to p1: the joint construction intersects bisectors, which at large
        # coordinates loses the precision the joint tangents need.
        origin = self.p1
        local = CubicBezier(P(0.0, 0.0), self.c1 - origin, self.c2 - origin, self.p2 - origin)
        segments, exhausted = local._approximate(tolerance, max_depth)
        if strict and exhausted:
            raise ApproximationError(
                f"biarc approximation could not reach tolerance {tolerance!r} within max_depth={max_depth!r}"
            )
        segments, merged = _connected(segments, local.p1, local.p2)
        if not segments:
            raise ApproximationError("no segment of length EPSILON or more can represent this curve")
        if merged and strict and not local._within_tolerance(segments, tolerance, samples=16 * len(segments)):
            raise ApproximationError(
                f"biarc approximation could not keep tolerance {tolerance!r} after absorbing pieces shorter than EPSILON"
            )
        if max_arc_angle is not None:
            segments = _split_arcs(segments, max_arc_angle)
        segments = [_shifted(seg, origin) for seg in segments]
        # Shifting back rounds the far endpoint; the curve's own endpoints are restored exactly.
        segments[0] = dataclasses.replace(segments[0], p1=self.p1)
        segments[-1] = dataclasses.replace(segments[-1], p2=self.p2)
        return segments

    def _approximate(self, tolerance: float, max_depth: int) -> tuple[list[Line | Arc], bool]:
        """The raw biarc chain for a curve at the origin; True if some piece could not meet ``tolerance``."""
        if self.is_straight:
            return [self.chord], False
        segments: list[Line | Arc] = []
        exhausted = False
        for piece in self.subdivide_inflections():
            exhausted = piece._biarcs(tolerance, max_depth, 0, segments) or exhausted
        return segments, exhausted

    def _biarcs(self, tolerance: float, max_depth: int, depth: int, out: list[Line | Arc]) -> bool:
        """Append the biarcs for this inflection-free piece; True if some piece could not meet ``tolerance``."""
        if self.is_degenerate:
            return False
        if self.is_straight:
            out.append(self.chord)
            return False
        biarc = self._biarc()
        if biarc is not None and self._within_tolerance(biarc, tolerance):
            out.extend(biarc)
            return False
        if depth >= max_depth and not self.chord.is_degenerate:
            out.extend(biarc if biarc is not None else [self.chord])
            return True
        # Below max_depth, or a piece whose ends coincide: no single segment can stand in for it.
        head, tail = self.subdivide(0.5)
        exhausted_head = head._biarcs(tolerance, max_depth, depth + 1, out)
        exhausted_tail = tail._biarcs(tolerance, max_depth, depth + 1, out)
        return exhausted_head or exhausted_tail

    def _biarc(self) -> list[Line | Arc] | None:
        """The equal-tangent biarc for an inflection-free piece, or None if it cannot be formed."""
        joint = self._joint_point()
        if joint is None:
            return None
        t_start = self.start_tangent
        t_end = self.end_tangent
        arc1 = Arc.from_two_points_and_tangent(self.p1, self.p1 + t_start, joint)
        arc2 = Arc.from_two_points_and_tangent(self.p2, self.p2 - t_end, joint, reverse=True)
        first: Line | Arc = arc1 if arc1 is not None else Line(self.p1, joint)
        second: Line | Arc = arc2 if arc2 is not None else Line(joint, self.p2)
        if (
            isinstance(first, Arc)
            and isinstance(second, Arc)
            and const.is_zero(first.radius - second.radius)
            and first.center.almost_equal(second.center)
            and first.direction == second.direction
        ):
            # One circle within EPSILON: a single arc replaces the pair when it satisfies the
            # invariant; near-EPSILON radii can differ enough that it does not.
            try:
                return [Arc(first.p1, second.p2, first.radius, first.angle + second.angle, first.center)]
            except GeometryError:
                return [first, second]
        return [first, second]

    def _joint_point(self) -> P | None:
        """The biarc joint: where the curve's midpoint ray meets the joint circle through p1 and p2.

        Every point of that circle gives two arcs that meet with a common
        tangent, so the joint is never approximated by a point off the
        circle; if the construction fails, None is returned and the caller
        subdivides.
        """
        chord = self.chord
        if chord.is_degenerate:
            return None
        t_start = self.start_tangent
        t_end = self.end_tangent
        mid = chord.midpoint
        bisector1 = Line(mid, mid + chord.vector.normal())
        u_seg = Line(self.p1 + t_start, self.p2 + t_end)
        if u_seg.is_degenerate:
            return None
        u_mid = u_seg.midpoint
        bisector2 = Line(u_mid, u_mid + u_seg.vector.normal())
        p_mid = self.point_at(0.5)
        center = bisector1.intersection(bisector2)
        if center is None:
            # Symmetric piece: the joint circle degenerates to the chord's bisector, so the
            # curve midpoint's projection onto it is a joint where both arcs meet tangentially.
            return bisector1.normal_projection_point(p_mid)
        radius = center.distance(self.p1)
        v = p_mid - center
        if v.length == 0.0:
            return None
        return center + v.unit * radius

    def _within_tolerance(self, segments: list[Line | Arc], tolerance: float, *, samples: int = 16) -> bool:
        """Sampled two-sided check between the curve and its approximating segments.

        Curve samples must be within ``tolerance`` of the nearest segment and
        segment samples within ``tolerance`` of the curve. Peaks between
        samples can exceed ``tolerance`` slightly; callers that need a
        measurement use :meth:`hausdorff_distance`.
        """
        points = self._sample_polyline(samples)
        for p in points:
            if min(seg.distance_to_point(p, segment=True) for seg in segments) > tolerance:
                return False
        for seg in segments:
            # Endpoints included: the biarc joint is not on the curve, so its deviation counts.
            for k in range(7):
                if self._distance_to_curve(seg.point_at(k / 6), points) > tolerance:
                    return False
        return True

    def _sample_polyline(self, n: int) -> list[P]:
        return [self.point_at(k / n) for k in range(n + 1)]

    def _distance_to_curve(self, p: P, samples: list[P]) -> float:
        """Distance from ``p`` to the curve.

        Every local minimum of the sampled distance is refined by a ternary
        search on the parameter, so a curve that passes near itself (a loop)
        does not hide the true nearest branch behind a closer sample on the
        other branch.
        """
        n = len(samples) - 1
        dist2 = [s.distance2(p) for s in samples]
        best = math.inf
        for i in range(n + 1):
            if (i > 0 and dist2[i - 1] < dist2[i]) or (i < n and dist2[i + 1] < dist2[i]):
                continue
            best = min(best, dist2[i])  # keep the exact sampled candidate (endpoints are often exact)
            lo = max(0, i - 1) / n
            hi = min(n, i + 1) / n
            for _ in range(48):
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                if self.point_at(m1).distance2(p) <= self.point_at(m2).distance2(p):
                    hi = m2
                else:
                    lo = m1
            best = min(best, self.point_at((lo + hi) / 2.0).distance2(p))
        return math.sqrt(best)

    def hausdorff_distance(self, segments: Sequence[Line | Arc], *, samples: int = 64) -> float:
        """Estimated two-sided Hausdorff distance between this curve and its approximating segments.

        Curve samples are measured to the nearest segment, and segment samples
        to the curve (nearest sample refined by a parameter search); the larger
        of the two maxima is returned. ``samples`` sets the curve sampling
        density; each segment is sampled at a quarter of it (at least 8 points).
        The result is a sampled estimate, not a bound.

        Raises:
            GeometryError: If ``segments`` is empty or ``samples`` is less than 1.
        """
        if not segments:
            raise GeometryError("hausdorff_distance needs at least one segment to measure against")
        if samples < 1:
            raise GeometryError(f"samples must be at least 1, got {samples!r}")
        curve_samples = self._sample_polyline(samples)
        worst = 0.0
        for p in curve_samples:
            worst = max(worst, min(seg.distance_to_point(p, segment=True) for seg in segments))
        per_segment = max(8, samples // 4)
        for seg in segments:
            for k in range(per_segment + 1):
                worst = max(worst, self._distance_to_curve(seg.point_at(k / per_segment), curve_samples))
        return worst

    # ----- transformations and output ------------------------------------------

    def reversed(self) -> CubicBezier:
        """The same curve travelled the other way: ``p2 -> c2 -> c1 -> p1``."""
        return CubicBezier(self.p2, self.c2, self.c1, self.p1)

    def to_svg_path(
        self, *, scale: float = 1.0, precision: int | None = None, add_prefix: bool = True, add_move: bool = False
    ) -> str:
        """SVG path data ``C c1 c2 p2`` (optionally preceded by ``M p1``)."""
        fmt = util.float_formatter(scale=scale, precision=precision)
        prefix = "C " if add_prefix or add_move else ""
        if add_move:
            prefix = f"M {fmt(self.p1.x)},{fmt(self.p1.y)} {prefix}"
        return f"{prefix}{fmt(self.c1.x)},{fmt(self.c1.y)} {fmt(self.c2.x)},{fmt(self.c2.y)} {fmt(self.p2.x)},{fmt(self.p2.y)}"

    def __str__(self) -> str:
        return f"CubicBezier({self.p1}, {self.c1}, {self.c2}, {self.p2})"


def _split_arcs(segments: list[Line | Arc], max_angle: float) -> list[Line | Arc]:
    """The chain with every arc split so that ``|angle| <= max_angle``."""
    split: list[Line | Arc] = []
    for seg in segments:
        if isinstance(seg, Arc):
            split.extend(seg.split_max_sweep(max_angle))
        else:
            split.append(seg)
    return split


def _shifted(segment: Line | Arc, offset: P) -> Line | Arc:
    """The segment translated by ``offset``."""
    if isinstance(segment, Line):
        return Line(segment.p1 + offset, segment.p2 + offset)
    return Arc(segment.p1 + offset, segment.p2 + offset, segment.radius, segment.angle, segment.center + offset)


def _same_point(a: P, b: P) -> bool:
    return a.x == b.x and a.y == b.y


def _anchored(segment: Line | Arc, *, start: P | None = None, end: P | None = None) -> Line | Arc:
    """The segment with its start or end moved to a point within a few ``EPSILON`` of the old one.

    An arc keeps its circle when the invariant allows it, otherwise it is
    refitted through the moved point with its start tangent; a line is
    simply redrawn. The result may be degenerate and is filtered by the
    caller.
    """
    p1 = segment.p1 if start is None else start
    p2 = segment.p2 if end is None else end
    if isinstance(segment, Line):
        return Line(p1, p2)
    try:
        return dataclasses.replace(segment, p1=p1, p2=p2)
    except GeometryError:
        refit = Arc.from_two_points_and_tangent(p1, p1 + segment.start_tangent, p2)
        return refit if refit is not None else Line(p1, p2)


def _connected(segments: list[Line | Arc], start: P, end: P) -> tuple[list[Line | Arc], bool]:
    """Drop segments shorter than ``EPSILON`` and close the gaps they leave.

    The returned chain runs exactly from ``start`` to ``end`` with every
    consecutive pair sharing its endpoint and no degenerate segment; the
    second value tells whether anything had to change. A gap is closed by
    re-anchoring the following segment's start, and the final gap by
    re-anchoring the last segment's end (see :func:`_anchored`); a segment
    that becomes degenerate in the process is dropped and the gap moves on
    to its neighbour. The chain is empty only if nothing at least
    ``EPSILON`` long can span ``start`` to ``end``.
    """
    out: list[Line | Arc] = []
    cursor = start
    merged = False
    for seg in segments:
        candidate = seg if _same_point(seg.p1, cursor) else _anchored(seg, start=cursor)
        if candidate is not seg:
            merged = True
        if candidate.is_degenerate:
            merged = True
            continue
        out.append(candidate)
        cursor = candidate.p2
    while out and not _same_point(out[-1].p2, end):
        merged = True
        fixed = _anchored(out.pop(), end=end)
        if not fixed.is_degenerate:
            out.append(fixed)
    if out:
        return out, merged
    connector = Line(start, end)
    return ([connector] if not connector.is_degenerate else []), True


def _quadratic_roots(a: float, b: float, c: float) -> list[float]:
    """Real roots of ``a t^2 + b t + c``, in the form that stays accurate when ``a`` is small.

    A double root is reported once and a negative discriminant gives no
    roots: where a derivative only touches zero it does not change sign, so
    there is no extreme or crossing to report. No scale-relative test is
    involved; only an exactly zero coefficient lowers the degree.
    """
    if a == 0.0:
        if b == 0.0:
            return []
        return [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return []
    q = -0.5 * (b + math.copysign(math.sqrt(disc), b))
    if q == 0.0:
        return [0.0]
    first = q / a
    second = c / q
    return [first] if first == second else [first, second]


def _cubic_roots_in_unit_interval(z0: float, z1: float, z2: float, z3: float, tolerance: float) -> list[float]:
    """Parameters in ``[0, 1]`` where ``z0 t^3 + z1 t^2 + z2 t + z3`` is zero, each once, in order.

    ``[0, 1]`` is cut at the polynomial's critical points into monotonic
    pieces. An end of the interval whose value is within ``tolerance`` of
    zero is a root, and so is a critical point within ``tolerance`` of zero
    that the polynomial does not cross (a tangency). A piece whose ends have
    opposite signs holds exactly one crossing, found by bisection. Every
    degree is handled alike: there is no closed-form step to lose precision
    when the leading coefficient is small.
    """

    def value(t: float) -> float:
        return ((z0 * t + z1) * t + z2) * t + z3

    knots = [0.0, *sorted(c for c in _quadratic_roots(3.0 * z0, 2.0 * z1, z2) if 0.0 < c < 1.0), 1.0]
    values = [value(k) for k in knots]
    signs = [(v > 0.0) - (v < 0.0) for v in values]
    last = len(knots) - 1
    is_root = [False] * len(knots)
    for i, v in enumerate(values):
        if abs(v) >= tolerance:
            continue
        crossing = (i > 0 and signs[i - 1] * signs[i] < 0) or (i < last and signs[i] * signs[i + 1] < 0)
        is_root[i] = i in (0, last) or not crossing
    roots = [k for k, ok in zip(knots, is_root, strict=True) if ok]
    for i in range(last):
        if is_root[i] or is_root[i + 1] or signs[i] * signs[i + 1] >= 0:
            continue
        roots.append(_bisect(value, knots[i], knots[i + 1], values[i]))
    return sorted(roots)


def _bisect(value: Callable[[float], float], lo: float, hi: float, value_lo: float) -> float:
    """The root of a monotonic function between ``lo`` and ``hi``, whose values have opposite signs."""
    negative_lo = value_lo < 0.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        v = value(mid)
        if v == 0.0:
            return mid
        if (v < 0.0) == negative_lo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
