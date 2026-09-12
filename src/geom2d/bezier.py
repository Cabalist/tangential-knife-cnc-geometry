"""Cubic bezier curve.

Includes biarc approximation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from . import const, util
from .arc import Arc
from .const import float_eq, is_zero
from .line import Line, TLine
from .point import P, TPoint

if TYPE_CHECKING:
    from typing import Self


# pylint: disable=invalid-name


class CubicBezier(tuple[P, P, P, P]):
    """Two dimensional immutable cubic bezier curve.

    For information about Bezier curves see:
    https://pomax.github.io/bezierinfo

    Args:
        p1: Start point as 2-tuple (x, y).
        c1: First control point as 2-tuple (x, y).
        c2: Second control point as 2-tuple (x, y).
        p2: End point as 2-tuple (x, y).
    """

    __slots__ = ()

    def __new__(cls, p1: TPoint, c1: TPoint, c2: TPoint, p2: TPoint) -> Self:
        """Create a new CubicBezier object."""
        return super().__new__(
            cls,
            (P(p1), P(c1), P(c2), P(p2)),  # type: ignore [arg-type]
        )  # type: ignore [type-var]

    @staticmethod
    def from_quadratic(p1: TPoint, q: TPoint, p2: TPoint) -> CubicBezier:
        """Create a CubicBezier from a quadratic Bazier curve.

        Args:
            p1: Start point as 2-tuple (x, y).
            q: Control point as 2-tuple (x, y).
            p2: End point as 2-tuple (x, y).
        """
        q = P(q)
        p1 = P(p1)
        p2 = P(p2)
        c1 = p1 + (2.0 * (q - p1)) / 3.0
        c2 = p2 + (2.0 * (q - p2)) / 3.0
        return CubicBezier(p1, c1, c2, p2)

    @property
    def p1(self) -> P:
        """The start point of curve."""
        return self[0]

    @property
    def c1(self) -> P:
        """The first control point of curve."""
        return self[1]

    @property
    def c2(self) -> P:
        """The second control point of curve."""
        return self[2]

    @property
    def p2(self) -> P:
        """The end point of curve."""
        return self[3]

    def start_tangent_angle(self) -> float:
        """The tangent direction of this curve at the first point.

        This would normally be the same as the angle of the
        first control point vector, unless the control point is
        coincident with the first point.
        Angle in radians: -PI < angle < PI.
        """
        return self.tangent(0.0).angle()

    def end_tangent_angle(self) -> float:
        """Return the end tangent direction of this curve.

        From the end (second) point.
        Angle in radians: -PI < angle < PI.
        """
        return self.tangent(1.0).angle()

    def point_at(self, t: float) -> P:
        """A point on the curve corresponding to <t>.

        This is the parametric function Bezier(t),
        where 0 < t < 1.0.

        Returns:
            A point as 2-tuple (x, y).
        """
        if is_zero(t):
            return self.p1
        if float_eq(t, 1.0):
            return self.p2
        t2 = t * t
        mt = 1 - t
        mt2 = mt * mt
        return self.p1 * mt2 * mt + self.c1 * 3 * t * mt2 + self.c2 * 3 * t2 * mt + self.p2 * t2 * t

    def midpoint(self) -> P:
        """Return the parameteric midpoint of this curve."""
        return self.point_at(0.5)

    # Canonical reference impl:
    #         return (self.p1 * (1 - t)**3 +
    #                 self.c1 * t * 3 * (1 - t)**2 +
    #                 self.c2 * (t * t) * 3 * (1 - t) +
    #                 self.p2 * t**3)

    #     def midpoint(self):
    #         """The parametric midpoint of the curve.
    #         """
    #         # TODO: This doesn't do anything interesting - remove?
    #         return self.point_at(0.5)

    def tangent(self, t: float) -> P:
        """The tangent unit vector at the point on the curve at `t`.

        `t` is the unit distance from first point where 0 <= t <= 1.
        """
        tangent_vector: P
        if is_zero(t):
            if self.c1 == self.p1:
                tangent_vector = self.c2 - self.p1
            else:
                tangent_vector = self.c1 - self.p1
        elif float_eq(t, 1.0):
            if self.c2 == self.p2:
                tangent_vector = (self.c1 - self.p2).mirror()
            else:
                tangent_vector = (self.c2 - self.p2).mirror()
        else:
            tangent_vector = self.derivative1(t)
        return tangent_vector.unit()

    def normal(self, t: float) -> P:
        """Normal unit vector at `t`."""
        return self.tangent(t).normal()

    def flatness(self) -> float:
        """Return the flatness of this curve.

        The maximum distance between the control points and the line segment
        defined by the start and end points of the curve.
        This is known as convex hull flatness and is robust regarding
        degenerate curves.
        """
        # First check if this curve is actually a straight line...
        if self.p1 == self.c1 and self.p2 == self.c2:
            return 0
        chord = Line(self.p1, self.p2)
        d1 = chord.distance_to_point(self.c1, segment=True)
        d2 = chord.distance_to_point(self.c2, segment=True)
        return max(d1, d2)

    def subdivide(self, t: float) -> tuple[CubicBezier, CubicBezier]:
        """Subdivide this curve at the point on the curve at `t`.

        Split curve into two cubic bezier curves, where 0<t<1.
        Uses De Casteljaus's algorithm.

        Returns:
            A tuple of one or two CubicBezier objects.
        """
        if t < 0 or t > 1:
            raise ValueError(f"t={t}")
        cp0, cp1, p, cp2, cp3 = self.controlpoints_at(t)
        curve1 = CubicBezier(self.p1, cp0, cp1, p)
        curve2 = CubicBezier(p, cp2, cp3, self.p2)
        return (curve1, curve2)

    def subdivide_inflections(self) -> tuple[CubicBezier, ...]:
        """Subdivide this curve at the inflection points, if any.

        Returns:
            A list containing one to three curves depending on whether
            there are no inflections, one inflection, or two inflections.
        """
        t1, t2 = self.roots()
        if t2 < 0:
            if t1 > 0:
                return self.subdivide(t1)  # one inflection at t1
            return (self,)  # no inflections
        if t1 < 0:
            if t2 > 0:
                return self.subdivide(t2)  # one inflection at t2
            return (self,)  # no inflections

        # Two roots/inflection points
        assert t1 < t2

        # Subdivide at first inflection
        curve1, curve2x = self.subdivide(t1)

        # Subdivide at second inflection.
        # Need to recalculate roots for subcurve.
        t1, t2 = curve2x.roots()
        t = max(t1, t2)
        assert t > 0
        curve2, curve3 = curve2x.subdivide(t)

        return curve1, curve2, curve3

    def roots(self) -> tuple[float, float]:
        """Find roots of this curve.

        The roots are inflection points are where the curve changes direction,
        has a cusp, or a loop.
        There may be none, one, or two inflections on the curve.
        A loop will have two inflections.

        These inflection points can be used to subdivide the curve.

        See:
            http://web.archive.org/web/20220129063812/https://www.caffeineowl.com/graphics/2d/vectorial/cubic-inflexion.html

        Args:
            imaginary: If True find `imaginary` inflection points.
                These are useful for subdividing curves with loops.
                Default is False.

        Returns:
            A tuple containing the roots (t1, t2).
            The root values will be 0 < t < 1 or -1 if no root.
            If there is only one root it will always be the
            first value of the tuple.
            The roots will be ordered by ascending value if
            there is more than one.
        """
        # Basically the equation to be solved is where the cross product of
        # the first and second derivatives is zero:
        # P' X P'' = 0
        # Where P' and P'' are the first and second derivatives respectively

        # Temporary vectors to simplify the math
        v1 = self.c1 - self.p1
        v2 = self.c2 - self.c1 - v1
        v3 = self.p2 - self.c2 - v1 - 2 * v2

        # Calculate quadratic coefficients
        # of the form a*t**2 + b*t + c = 0
        a = v2.x * v3.y - v2.y * v3.x
        b = v1.x * v3.y - v1.y * v3.x
        c = v1.x * v2.y - v1.y * v2.x

        def _valid_t(t: float) -> float:
            # Check range of t, returns -1 if t is out of range.
            if const.EPSILON < t < 1 - const.EPSILON:
                return t
            return -1

        if const.is_zero(a):
            if not const.is_zero(b):
                # This would be a stright line so there shouldn't really
                # be an inflection point.
                # TODO: investigate this.
                return _valid_t(-c / b), -1
            return -1, -1

        # the discriminant of the quadratic eq.
        dis = b * b - 4 * a * c
        if const.is_zero(dis):
            if a != 0:
                return _valid_t(-b / (2 * a)), -1
            return -1, -1

        # When a curve has a loop the discriminant will be negative
        # so use the absolute value to use the real part of a
        # normally complex number...
        # I can't remember how this was determined besides
        # experimentally.
        # TODO: prove this
        disroot = math.sqrt(abs(dis))
        t1 = _valid_t((-b - disroot) / (2 * a))
        t2 = _valid_t((-b + disroot) / (2 * a))

        # Return in ascending order
        if t1 > 0:
            return (t1, t2) if t1 <= t2 or t2 < 0 else (t2, t1)
        return t2, t1

    def find_extrema_points(self) -> list[P]:
        """Find the extremities of this curve.

        See:
            https://pomax.github.io/bezierinfo/#extremities

        Returns:
            A list of zero to four points.
        """
        return [self.point_at(t) for t in self.find_extrema()]

    def find_extrema(self) -> list[float]:
        """Find the extremities of this curve.

        See:
            https://pomax.github.io/bezierinfo/#extremities
            https://github.polettix.it/ETOOBUSY/2020/07/09/bezier-extremes/

        Returns:
            A list of zero to four parametric (t) values.
        """
        # Get the quadratic coefficients
        v_a = 3 * (-self.p1 + (3 * self.c1) - (3 * self.c2) + self.p2)
        v_b = 6 * (self.p1 - (2 * self.c1) + self.c2)
        v_c = 3 * (self.c1 - self.p1)

        # Discriminants
        disc_x = (v_b.x / 2) ** 2 - v_a.x * v_c.x
        disc_y = (v_b.y / 2) ** 2 - v_a.y * v_c.y

        extrema: list[float] = []
        if const.is_zero(disc_x):
            extrema.append((-v_b.x / 2) / v_a.x)
        elif v_a.x != 0:
            sqrt_x = math.sqrt(abs(disc_x))
            extrema.extend(
                (
                    ((-v_b.x / 2) + sqrt_x) / v_a.x,
                    ((-v_b.x / 2) - sqrt_x) / v_a.x,
                )
            )
        else:
            extrema.append(-v_c.x / v_b.x)
        if const.is_zero(disc_y):
            extrema.append((-v_b.y / 2) / v_a.y)
        elif v_a.y != 0:
            sqrt_y = math.sqrt(abs(disc_y))
            extrema.extend(
                (
                    ((-v_b.y / 2) + sqrt_y) / v_a.y,
                    ((-v_b.y / 2) - sqrt_y) / v_a.y,
                )
            )
        else:
            extrema.append(-v_c.y / v_b.y)

        return [t for t in extrema if 0 < t < 1]

    def controlpoints_at(self, t: float) -> tuple[P, P, P, P, P]:
        """Get the point on this curve at `t` plus control points.

        Useful for subdividing the curve at `t`.

        Args:
            t: location on curve. A value between 0.0 and 1.0

        Returns:
            A tuple of the form (C0, C1, P, C2, C3) where C1 and C2 are
            the control points tangent to P and C0 and C3 would be the
            new control points of the endpoints where this curve to be
            subdivided at P.
        """
        mt = 1 - t
        # First intermediate points
        d01 = mt * self.p1 + t * self.c1
        d12 = mt * self.c1 + t * self.c2
        d23 = mt * self.c2 + t * self.p2
        # Second intermediate points
        d012 = mt * d01 + t * d12
        d123 = mt * d12 + t * d23
        # Finally, the split point
        d0123 = mt * d012 + t * d123
        return (d01, d012, d0123, d123, d23)

    def derivative1(self, t: float) -> P:
        """Calculate the 1st derivative of this curve at `t`.

        Returns:
            The first derivative at `t` as 2-tuple (dx, dy).
        """
        t2 = t * t
        return (
            3 * (t2 - (2 * t) + 1) * (self.c1 - self.p1)
            + 6 * (t - t2) * (self.c2 - self.c1)
            + 3 * t2 * (self.p2 - self.c2)
        )

    def derivative2(self, t: float) -> P:
        """Calculate the 2nd derivative of this curve at `t`.

        Returns:
            The second derivative at `t` as 2-tuple (dx, dy).
        """
        # TODO: confirm this is correct.
        # See: https://pomax.github.io/bezierinfo/#inflections
        return 6 * ((1 - t) * self.p1 + (3 * t - 2) * self.c1 + (1 - 3 * t) * self.c2 + t * self.p2)

    def derivative3(self) -> P:
        """Calculate the 3rd derivative of this curve.

        Returns:
            The third derivative as 2-tuple (dx, dy).
        """
        return (-6 * self.p1) + (18 * self.c1) - (18 * self.c2) + (6 * self.p2)

    def curvature_at(self, t: float) -> float:
        """Calculate the curvature at `t`.

        See http://www.spaceroots.org/documents/ellipse/node6.html

        Returns:
            A scalar value `K` representing the curvature at `t`.
            Negative if curving to the right or positive
            if curving to the left when `t` increases.
        """
        # TODO: test this
        d1 = self.derivative1(t)
        d2 = self.derivative2(t)
        return ((d1.x * d2.y) - (d1.y * d2.x)) / math.pow((d1.x * d1.x) + (d1.y * d1.y), 3.0 / 2)

    def length(self, tolerance: float | None = None) -> float:
        """Approximate arc length of this curve.

        Calculate the approximate arc length of this curve
        within the specified tolerance.
        The resulting computed arc length will be
        cached so that subsequent calls are not expensive.

        Uses a simple and clever numerical algorithm described/invented by
        Jens Gravesen.

        See:
            - Jens Gravesen.
                Adaptive subdivision and the length and energy of
                Bezier curves.
                Comput. Geom., 8:13-31, 1997.
            - B. Guenter and R. Parent.
                Computing the arc length of parametric curves.
                IEEE Comp. Graph. and Appl., 5:72-78, 1990.

        Args:
            tolerance: The approximation tolerance.
                Default is const.EPSILON.

        Returns:
            The approximate arc length of this curve.
        """
        if tolerance is None:
            tolerance = const.EPSILON
        # Algorithm:
        #
        # If you denote the length of the
        # control polygon by L1 i.e.:
        #         L1 = |P0 P1| +|P1 P2| +|P2 P3|
        # and the length of the cord by L0 i.e.:
        #         L0 = |P0 P3|
        # then
        #         L = 1/2*L0 + 1/2*L1
        # is a good approximation of the length of the curve,
        # and the difference (L1-L0) is a measure of the error.
        # If the error is too large, then you just subdivide
        # the curve at parameter value 1/2, and find the length of each half.
        L1 = self.p1.distance(self.c1) + self.c1.distance(self.c2) + self.c2.distance(self.p2)
        L0 = self.p1.distance(self.p2)
        if tolerance < L1 - L0:
            # Subdivide the curve and recursively compute the sum.
            b1, b2 = self.subdivide(0.5)
            len1 = b1.length(tolerance=tolerance)
            len2 = b2.length(tolerance=tolerance)
            return len1 + len2

        return 0.5 * L0 + 0.5 * L1

    def line_intersection(self, line: TLine) -> list[P]:
        """Find intersection points of a line segment and this curve.

        See:
            https://www.particleincell.com/2013/cubic-line-intersection/

        Returns:
            A list of zero to three points where the line intersects this curve.
        """
        # Coefficients
        cfs = (
            3 * (self.c1 - self.c2) + self.p2 - self.p1,
            3 * (self.p1 - 2 * self.c1 + self.c2),
            3 * (self.c1 - self.p1),
            self.p1,
        )

        x1, y1 = line[0]
        x2, y2 = line[1]
        yy = y2 - y1
        xx = x1 - x2
        z = [yy * cf.x + xx * cf.y for cf in cfs]
        z[3] += x1 * (y1 - y2) + y1 * (x2 - x1)

        a, b, c = (p / z[0] for p in z[1:])
        q = (3 * b - (a**2)) / 9
        r = (9 * a * b - 27 * c - 2 * (a**3)) / 54
        d = q**3 + r**2  # discriminant

        if d >= 0:
            d2 = math.sqrt(d)
            t1 = r + d2
            t2 = r - d2
            s = math.copysign(math.pow(abs(t1), 1 / 3), t1)
            t = math.copysign(math.pow(abs(t2), 1 / 3), t2)
            # Imaginary part of root
            im = abs((math.sqrt(3) * (s - t)) / 2)

            roots: tuple[float, ...]
            r1 = -a / 3 + (s + t)  # real root
            if const.float_eq(im, 0):
                r2 = -a / 3 - (s + t) / 2  # real part of complex root
                roots = (r1, r2)
            else:
                roots = (r1,)
        else:
            th = math.acos(r / math.sqrt(-math.pow(q, 3)))
            q2 = 2 * math.sqrt(-q)
            a /= 3
            r1 = q2 * math.cos(th / 3) - a
            r2 = q2 * math.cos((th + 2 * math.pi) / 3) - a
            r3 = q2 * math.cos((th + 4 * math.pi) / 3) - a
            roots = (r1, r2, r3)

        return [self.point_at(r) for r in roots if 0 <= r <= 1]

    def biarc_approximation(
        self,
        tolerance: float = 0.001,
        max_depth: float = 4,
        line_flatness: float = 0.001,
        _recurs_depth: float = 0,
    ) -> list[Arc | Line]:
        """Approximate this curve using biarcs.

        This will recursively subdivide the curve into a series of
        G1 (tangential continuity) connected arcs or lines until the
        Hausdorff distance between the approximation and this bezier
        curve is within the specified tolerance.

        Args:
            tolerance: Approximation tolerance. A lower value increases
                accuracy at the cost of time and number of generated
                biarc segments.
            max_depth: Maximum recursion depth. This limits how many times
                the Bezier curve can be subdivided.
            line_flatness: Segments flatter than this value will be converted
                to straight line segments instead of arcs with huge radii.
                Generally this should be a small value (say <= 0.01) to avoid
                path distortions.

        Returns:
            A list of Arc and/or Line objects. The list will be empty
            if the curve is degenerate (i.e. if the end points
            are coincident).
        """
        # Check for degenerate cases:
        # Bail if the curve endpoints are coincident.
        if self.p1 == self.p2:
            return []

        # Or if the curve is basically a straight line then return a Line.
        if line_flatness > 0 and self.flatness() < line_flatness:
            return [Line(self.p1, self.p2)]

        if _recurs_depth == 0:
            # Subdivide this curve at any inflection points to make sure
            # the curve has monotone curvature with no discontinuities.
            # Recursively approximate each sub-curve.
            # This is only required once before any recursion starts
            # since sub-curves shouldn't have any inflections (right?).
            curves = self.subdivide_inflections()
            if len(curves) > 1:
                biarcs = []
                for curve in curves:
                    sub_biarcs = curve.biarc_approximation(
                        tolerance=tolerance,
                        max_depth=max_depth,
                        line_flatness=line_flatness,
                        _recurs_depth=_recurs_depth + 1,
                    )
                    biarcs.extend(sub_biarcs)
                return biarcs

        # Calculate the arc that intersects the two endpoints of this curve
        # and the set of possible biarc joints.
        j_arc = self._biarc_joint_arc()
        # Another degenerate case which could happen if the curve is too flat
        # or too tiny.
        if (
            j_arc is None
            or j_arc.radius < max(line_flatness, const.EPSILON)
            or j_arc.length() < max(line_flatness, const.EPSILON)
        ):
            return [Line(self.p1, self.p2)]

        # To make this simple for now:
        # The biarc joint J will be the intersection of the line
        # whose endpoints are the center of the joint arc and the
        # maximum of the bezier curve, and the joint arc.
        # In practice, t=.05 instead of the maximum works just as well...
        # TODO: See [A. Riskus, 2006] for a possibly more accurate method
        p = self.point_at(0.5)
        # debug.draw_point(p, color='#ffff00') # DEBUG
        v = p - j_arc.center
        pjoint = v * (j_arc.radius / v.length()) + j_arc.center
        # debug.draw_point(pjoint, color='#00ff00') # DEBUG

        # Subdivide and recurse if pjoint-arc distance is > tolerance
        if _recurs_depth < max_depth and pjoint.distance(p) > tolerance:
            return self._biarc_recurs_subdiv(
                tolerance=tolerance,
                max_depth=max_depth,
                line_flatness=line_flatness,
                _recurs_depth=_recurs_depth,
            )

        # Create the two arcs that define the biarc.
        c1 = self.c1 if self.c1 != self.p1 else self.c2
        c2 = self.c2 if self.c2 != self.p2 else self.c1
        arc1 = Arc.from_two_points_and_tangent(self.p1, c1, pjoint)
        arc2 = Arc.from_two_points_and_tangent(self.p2, c2, pjoint, reverse=True)
        assert arc1
        assert arc2

        if _recurs_depth < max_depth and (
            not self._check_hausdorff(arc2, 0.5, 1.0, tolerance) or not self._check_hausdorff(arc1, 0, 0.5, tolerance)
        ):
            return self._biarc_recurs_subdiv(
                tolerance=tolerance,
                max_depth=max_depth,
                line_flatness=line_flatness,
                _recurs_depth=_recurs_depth,
            )

        # See if the biarcs can be combined into one arc if
        # they happen to have the same radius.
        if const.float_eq(arc1.radius, arc2.radius):
            assert const.float_eq(arc1.angle, arc2.angle)
            arc = Arc(arc1.p1, arc2.p2, arc1.radius, arc1.angle * 2, arc1.center)
            return [arc]

        # Biarc is within tolerance or recursion limit has been reached.
        return [arc1, arc2]

    def _biarc_recurs_subdiv(
        self,
        tolerance: float,
        max_depth: float,
        line_flatness: float,
        _recurs_depth: float,
    ) -> list[Arc | Line]:
        """Recursively subdivide the curve.

        Approximate each sub-curve with biarcs.
        """
        _recurs_depth += 1
        # Note: subdividing at t=0.5 is as good or better
        # than using J or maximum. I've tried it.
        curve1, curve2 = self.subdivide(0.5)
        biarcs1 = curve1.biarc_approximation(
            tolerance=tolerance,
            max_depth=max_depth,
            line_flatness=line_flatness,
            _recurs_depth=_recurs_depth,
        )
        biarcs2 = curve2.biarc_approximation(
            tolerance=tolerance,
            max_depth=max_depth,
            line_flatness=line_flatness,
            _recurs_depth=_recurs_depth,
        )
        return biarcs1 + biarcs2

    def _biarc_joint_arc(self) -> Arc | None:
        """Calculat joint arc.

        Calculate the arc that intersects the two endpoints of this curve
        and the set of possible biarc joints.

        Returns:
            The biarc joint arc or None if one can't be computed.
        """
        # The center <s> of the circle is the intersection of the bisectors
        # of line segments P1->P2 and (P1+unit(C1))->(P2+unit(C2))
        # TODO: in case of c1/c2 coincident with endpoint - calc tangent
        # to create fake unit vector.
        chord = Line(self.p1, self.p2)
        # debug.draw_line(chord, color='#c0c000')
        u1 = self.tangent(0)  # (self.c1 - self.p1).unit()
        u2 = self.tangent(1).mirror()  # (self.c2 - self.p2).unit()
        bisect1 = chord.bisector()
        u_seg = Line(self.p1 + u1, self.p2 + P(-u2.x, -u2.y))
        bisect2 = u_seg.bisector()
        # debug.draw_line(u_seg, color='#c0c000')
        # debug.draw_line(bisect1, color='#ffff00')
        # debug.draw_line(bisect2, color='#ffff00')
        center = bisect1.intersection(bisect2)
        # debug.draw_point(center, color='#c0c000')
        if center is not None:
            radius = center.distance(self.p1)
            angle = center.angle2(self.p1, self.p2)
            # The angle is reversed if the center is also the chord midpoint.
            # This is not strictly necessary...
            if center == chord.midpoint():
                angle = -angle
            return Arc(self.p1, self.p2, radius, angle, center)
            # debug.draw_circle(
            #    j_arc.center, j_arc.radius, color='#c0c0c0'
            # )

        return None

    def _check_hausdorff(self, arc: Arc, t1: float, t2: float, tolerance: float, ndiv: int = 7) -> bool:
        """Check Hausdorff distance to arc.

        Check this curve against the specified arc to see
        if the Hausdorff distance is within `tolerance`.

        The approximation accuracy depends on the number of steps
        specified by `ndiv`. Default is seven.

        Args:
            arc (:obj:`Arc`): The arc to test
            t1 (float): Start location of curve
            t2 (float): End location of curve
            tolerance (float): The maximum distance
            ndiv (int): Number of steps

        Returns:
            True if the Hausdorff distance to the arc is within
            the specified tolerance.
        """
        # This is a fairly rough approximation but it works pretty well.
        t_step = (t2 - t1) * (1.0 / ndiv)
        t = t1
        while t <= t2:
            p = self.point_at(t)
            # debug.draw_point(p, color='#000000') # DEBUG
            d = arc.center.distance(p) - arc.radius
            if d > tolerance:
                return False
            t += t_step
        return True

    def hausdorff_distance(self, arc: Arc, t1: float = 0, t2: float = 1, ndiv: int = 9) -> float:
        """Calculate Hausdorff distance to arc.

        The approximation accuracy depends on the number of steps
        specified by `ndiv`.

        This curve should have no inflections and the arc should
        have the same convexity (ie "bulge" in the same direction).

        Args:
            arc (:obj:`Arc`): The arc to test
            t1 (float): Start location of curve
            t2 (float): End location of curve
            ndiv (int): Number of steps

        Returns:
            Maximum distance along curve.
        """
        # This is a fairly rough approximation but it works pretty well.
        t_step = (t2 - t1) * (1.0 / ndiv)
        t = t1
        hd: float = 0
        while t <= t2:
            p = self.point_at(t)
            d = arc.distance_to_point(p)
            # d = arc.center.distance(p) - arc.radius
            hd = max(hd, d)
            t += t_step

        return hd

    def path_reversed(self) -> CubicBezier:
        """Return a CubicBezier with control points (direction) reversed."""
        return CubicBezier(self.p2, self.c2, self.p1, self.c1)

    def __str__(self) -> str:
        """Concise string representation."""
        return f"CubicBezier({self.p1}, {self.c1}, {self.c2}, {self.p2})"

    def __repr__(self) -> str:
        """Concise string representation."""
        return f"CubicBezier({self.p1!r}, {self.c1!r}, {self.c2!r}, {self.p2!r})"

    def to_svg_path(self, scale: float = 1, add_prefix: bool = True, add_move: bool = False) -> str:
        """CubicBezier to SVG path string.

        Args:
            scale: Scale factor. Default is 1.
            add_prefix: Prefix with the command prefix if True.
            add_move: Prefix with M command if True.

        Returns:
            A string with the SVG path 'd' attribute value
            that corresponds with this curve.
        """
        ff = util.float_formatter()

        prefix = "C " if add_prefix or add_move else ""
        if add_move:
            p1 = self.p1 * scale
            prefix = f"M {ff(p1.x)},{ff(p1.y)} {prefix}"

        c1 = self.c1 * scale
        c2 = self.c2 * scale
        p2 = self.p2 * scale
        return f"{prefix}{ff(c1.x)},{ff(c1.y)} {ff(c2.x)},{ff(c2.y)} {ff(p2.x)},{ff(p2.y)}"

        # Draw midpoint
        # debug.draw_point(curve.point_at(0.5), color='#00ff00')
