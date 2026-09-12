"""Cubic Bézier curve and its biarc approximation."""

import math
from dataclasses import dataclass

from . import const, util
from .arc import Arc
from .box import Box
from .errors import ApproximationError, GeometryError
from .line import Line
from .point import P, PointLike

_MAX_LENGTH_DEPTH = 24
"""Recursion cap for :meth:`CubicBezier.length`; 2**24 subdivisions is far past float resolution."""


@dataclass(frozen=True, slots=True)
class CubicBezier:
    """A cubic Bézier curve ``p1 -> c1 -> c2 -> p2``.

    Equality is field-wise on the ``EPSILON`` grid; ``reversed()`` gives a
    different curve object tracing the same shape backwards. The constructor
    takes ``P`` fields; :meth:`from_quadratic` accepts any point-like input. A curve whose
    control polygon is shorter than ``EPSILON`` is *degenerate*: its tangent
    angles are ``0.0`` and ``biarc_approximation`` returns no segments.
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
        """True if the whole control polygon fits within ``EPSILON``."""
        return self.polygon_length < const.EPSILON

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
        """Parameters in ``(0, 1)`` where ``x`` or ``y`` reaches a local extreme, sorted."""
        # Coefficients of the derivative  a t^2 + b t + c  per axis.
        va = (self.p2 - self.p1 + (self.c1 - self.c2) * 3.0) * 3.0
        vb = (self.p1 - self.c1 * 2.0 + self.c2) * 6.0
        vc = (self.c1 - self.p1) * 3.0
        scale = self.polygon_length
        if scale < const.EPSILON:
            return []
        params: list[float] = []
        for a, b, c in ((va.x, vb.x, vc.x), (va.y, vb.y, vc.y)):
            params.extend(_quadratic_roots(a, b, c, scale))
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

    def line_intersection(self, line: Line, *, segment: bool = False) -> list[P]:
        """Points where the curve meets the infinite line through ``line``.

        With ``segment=True`` only points on the line *segment* are kept.
        Works for every degree: curves whose cubic term vanishes along the
        line's normal (degree-elevated quadratics, symmetric arches) fall
        back to the quadratic or linear equation.
        """
        if line.is_degenerate:
            return []
        cfs = (
            (self.c1 - self.c2) * 3.0 + self.p2 - self.p1,
            (self.p1 - self.c1 * 2.0 + self.c2) * 3.0,
            (self.c1 - self.p1) * 3.0,
            self.p1,
        )
        yy = line.p2.y - line.p1.y
        xx = line.p1.x - line.p2.x
        z = [yy * cf.x + xx * cf.y for cf in cfs]
        z[3] += line.p1.x * (line.p1.y - line.p2.y) + line.p1.y * (line.p2.x - line.p1.x)
        scale = max(abs(v) for v in z)
        if scale == 0.0:
            return []
        roots = _polynomial_roots(z[0], z[1], z[2], z[3], scale)
        margin = const.EPSILON
        points = [self.point_at(min(1.0, max(0.0, t))) for t in roots if -margin <= t <= 1.0 + margin]
        if segment:
            points = [p for p in points if line.point_on_line(p, segment=True)]
        return points

    # ----- biarc approximation ------------------------------------------------

    def biarc_approximation(
        self,
        tolerance: float = 0.001,
        *,
        max_depth: int = 4,
        max_arc_angle: float | None = None,
        strict: bool = False,
    ) -> list[Line | Arc]:
        """Approximate the curve with tangent-continuous circular arcs (and lines where it is straight).

        The curve is first split at its inflections, then each piece is
        approximated by a biarc; a piece is subdivided while the two-sided
        Hausdorff distance to its biarc exceeds ``tolerance``, up to
        ``max_depth`` halvings.

        Output contract: a list of ``Line`` and ``Arc`` segments with no
        degenerate segment; the first segment starts exactly at ``p1`` and
        the last ends exactly at ``p2``; consecutive segments share their
        endpoint exactly; tangent directions agree at every joint within
        ``angle_eq`` (a ``Line`` is emitted only where the piece is straight:
        control points within ``EPSILON`` of the chord and both end tangents
        along it), with one geometric exception: where the curve itself turns
        through a region smaller than ``EPSILON`` (a cusp or near-cusp, turning
        radius below ``EPSILON``) the output has a corner at that point, because
        no segment of length ``EPSILON`` or more can carry the turn; the two-sided
        Hausdorff distance between each piece and its segments, estimated by
        sampling, is within ``tolerance`` unless ``max_depth`` was exhausted, in
        which case the best effort is returned, or :class:`ApproximationError`
        is raised when ``strict`` is set. A degenerate curve gives ``[]``.

        Args:
            tolerance: Maximum allowed distance between curve and arcs.
            max_depth: Maximum number of halvings per inflection-free piece.
            max_arc_angle: If given, every arc is split into equal pieces so
                that ``|angle| <= max_arc_angle``.
            strict: Raise :class:`ApproximationError` instead of returning a
                best effort when some piece could not meet ``tolerance``
                within ``max_depth``.

        Raises:
            GeometryError: If ``tolerance`` is below ``EPSILON`` (nothing in the library
                resolves finer than that) or ``max_depth`` is negative.
            ApproximationError: With ``strict``, if the tolerance could not be met.
        """
        if not tolerance >= const.EPSILON:
            raise GeometryError(f"tolerance must be at least EPSILON ({const.EPSILON!r}), got {tolerance!r}")
        if max_depth < 0:
            raise GeometryError(f"max_depth must be at least 0, got {max_depth!r}")
        if self.is_degenerate:
            return []
        if self.is_straight:
            return [self.chord]
        segments: list[Line | Arc] = []
        exhausted: list[bool] = []
        for piece in self.subdivide_inflections():
            piece._biarcs(tolerance, max_depth, 0, segments, exhausted)
        if strict and exhausted:
            raise ApproximationError(
                f"biarc approximation could not reach tolerance {tolerance!r} within max_depth={max_depth!r}"
            )
        if max_arc_angle is not None:
            split: list[Line | Arc] = []
            for seg in segments:
                if isinstance(seg, Arc):
                    split.extend(seg.split_max_sweep(max_arc_angle))
                else:
                    split.append(seg)
            segments = split
        return [seg for seg in segments if not seg.is_degenerate]

    def _biarcs(
        self, tolerance: float, max_depth: int, depth: int, out: list[Line | Arc], exhausted: list[bool]
    ) -> None:
        if self.is_degenerate:
            return
        if self.is_straight:
            out.append(self.chord)
            return
        biarc = self._biarc()
        if biarc is not None and self._within_tolerance(biarc, tolerance):
            out.extend(biarc)
            return
        if depth >= max_depth:
            # Depth exhausted: return the best effort and record that tolerance was not met.
            exhausted.append(True)
            out.extend(biarc if biarc is not None else [self.chord])
            return
        head, tail = self.subdivide(0.5)
        head._biarcs(tolerance, max_depth, depth + 1, out, exhausted)
        tail._biarcs(tolerance, max_depth, depth + 1, out, exhausted)

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
            and const.float_eq(first.radius, second.radius)
            and first.center.almost_equal(second.center)
            and first.direction == second.direction
        ):
            merged = Arc(first.p1, second.p2, first.radius, first.angle + second.angle, first.center)
            return [merged]
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
            # Symmetric piece: the joint circle degenerates to the chord's bisector, which the
            # curve midpoint lies on; both arcs meet there tangentially.
            return p_mid
        radius = center.distance(self.p1)
        v = p_mid - center
        if v.length == 0.0:
            return None
        return center + v.unit * radius

    def _within_tolerance(self, segments: list[Line | Arc], tolerance: float) -> bool:
        """Sampled two-sided check between the curve and its approximating segments.

        Curve samples must be within ``tolerance`` of the nearest segment and
        segment samples within ``tolerance`` of the curve. Peaks between
        samples can exceed ``tolerance`` slightly; callers that need a
        measurement use :meth:`hausdorff_distance`.
        """
        samples = self._sample_polyline(16)
        for p in samples:
            if min(seg.distance_to_point(p, segment=True) for seg in segments) > tolerance:
                return False
        for seg in segments:
            # Endpoints included: the biarc joint is not on the curve, so its deviation counts.
            for k in range(7):
                if self._distance_to_curve(seg.point_at(k / 6), samples) > tolerance:
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

    def hausdorff_distance(self, segments: list[Line | Arc], *, samples: int = 64) -> float:
        """Two-sided Hausdorff distance between this curve and its approximating segments.

        Curve samples are measured to the nearest segment, and segment samples
        to the curve (nearest sample refined by a parameter search); the larger
        of the two maxima is returned. ``samples`` sets the curve sampling
        density; each segment is sampled at a quarter of it (at least 8 points).
        """
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


def _quadratic_roots(a: float, b: float, c: float, scale: float) -> list[float]:
    """Real roots of ``a t^2 + b t + c`` with zero tests relative to ``scale`` (a length)."""
    if const.is_zero_rel(a, scale):
        if const.is_zero_rel(b, scale):
            return []
        return [-c / b]
    disc = b * b - 4.0 * a * c
    if const.is_zero_rel(disc, scale * scale):
        return [-b / (2.0 * a)]
    if disc < 0.0:
        return []
    root = math.sqrt(disc)
    return [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]


def _polynomial_roots(z0: float, z1: float, z2: float, z3: float, scale: float) -> list[float]:
    """Real roots of ``z0 t^3 + z1 t^2 + z2 t + z3``, reducing the degree when leading terms vanish."""
    if const.is_zero_rel(z0, scale):
        return _quadratic_roots(z1, z2, z3, scale)
    a = z1 / z0
    b = z2 / z0
    c = z3 / z0
    q = (3.0 * b - a * a) / 9.0
    r = (9.0 * a * b - 27.0 * c - 2.0 * a * a * a) / 54.0
    d = q * q * q + r * r
    shift = -a / 3.0
    if d >= 0.0:
        sd = math.sqrt(d)
        s = math.copysign(abs(r + sd) ** (1.0 / 3.0), r + sd)
        t = math.copysign(abs(r - sd) ** (1.0 / 3.0), r - sd)
        roots = [shift + s + t]
        if const.is_zero(abs(s - t) * math.sqrt(3.0) / 2.0):
            roots.append(shift - (s + t) / 2.0)
        return roots
    theta = math.acos(max(-1.0, min(1.0, r / math.sqrt(-(q * q * q)))))
    q2 = 2.0 * math.sqrt(-q)
    return [
        q2 * math.cos(theta / 3.0) + shift,
        q2 * math.cos((theta + const.TAU) / 3.0) + shift,
        q2 * math.cos((theta + 2.0 * const.TAU) / 3.0) + shift,
    ]
