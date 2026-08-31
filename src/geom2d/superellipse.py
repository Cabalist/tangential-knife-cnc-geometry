"""Create an approximated superellipse (Lamé curve) using Bezier curves.

Note: This was partly built using ChatGPT/Codex 5.6-Sol-high
    then significantly cleaned up.
"""

import math
from collections.abc import Sequence

from . import TPoint
from .polygon import rect_midpoints

_HALF_PI = math.pi / 2.0
_EPSILON = 1.0e-9

# Min/max number of curves for approximation
_MIN_CURVES = 4
_MAX_CURVES = 40


def _is_zero(n: float) -> bool:
    return abs(n) < _EPSILON


def _is_gt(a: float, b: float) -> bool:
    return a > (b - _EPSILON)


def _lame_point(theta: float, n: float) -> TPoint:
    """First-quadrant Lamé curve."""
    # ------------------------------------------------------------
    # Standard superellipse parameterization:
    #
    #     x = cos(theta) ^ (2/n)
    #     y = sin(theta) ^ (2/n)
    #
    #     0 <= theta <= pi/2
    #
    # This automatically distributes points reasonably well:
    # for square-ish curves they accumulate toward the corner,
    # while for strongly concave curves they follow the long
    # scooped portions near the axes.
    # ------------------------------------------------------------

    # Clamp to axes
    if _is_zero(theta):
        return (1.0, 0.0)
    if _is_gt(theta, _HALF_PI):
        return (0.0, 1.0)

    exp = 2.0 / n
    return math.cos(theta) ** exp, math.sin(theta) ** exp


def _lame_tangent(theta: float, n: float) -> TPoint:
    """Exact tangent DIRECTION to the first-quadrant Lamé curve."""
    # Differentiating
    #
    #     x = cos(theta)^e
    #     y = sin(theta)^e
    #
    # gives, ignoring the common factor e:
    #
    #     dx = -sin(theta) cos(theta)^(e-1)
    #     dy =  cos(theta) sin(theta)^(e-1)
    #
    # At the axes the limiting tangent depends on whether n is
    # above or below 1.
    #
    if _is_zero(theta):
        if n > 1.0:
            # Convex curve: vertical tangent.
            return (0.0, 1.0)
        # Concave curve: horizontal tangent toward center.
        return (-1.0, 0.0)

    if _is_gt(theta, _HALF_PI):
        if n > 1.0:
            # Convex curve: horizontal tangent.
            return (-1.0, 0.0)
        # Concave curve: vertical tangent toward endpoint.
        # This is the one-sided tangent approaching the
        # top axis from the first quadrant.
        return (0.0, 1.0)

    c = math.cos(theta)
    s = math.sin(theta)

    exp = 2.0 / n - 1.0
    tx = -s * (c**exp)
    ty = c * (s**exp)

    # Convert to unit vector
    length = math.hypot(tx, ty)
    if _is_zero(length):
        raise RuntimeError("Zero-length tangent")
    return (tx / length, ty / length)


def _tangent_intersection(p0: TPoint, t0: TPoint, p1: TPoint, t1: TPoint) -> TPoint:
    """Intersection of two tangent lines."""

    def _cross(a: TPoint, b: TPoint) -> float:
        return a[0] * b[1] - a[1] * b[0]

    denominator = _cross(t0, t1)
    if _is_zero(denominator):
        return p0

    delta = (p1[0] - p0[0], p1[1] - p0[1])
    s = _cross(delta, t1) / denominator
    return (p0[0] + t0[0] * s, p0[1] + t0[1] * s)


def rhombus_superellipse(
    points: Sequence[TPoint], k: float, num_curves: int = 16
) -> list[tuple[TPoint, TPoint, TPoint, TPoint]]:
    """Approximate a superellipse inscribed in a rhombus/parallelogram.

    Uses a fixed number of cubic Beziers to approximate a Lamé curve.

    Args:
        points: Sequence of 4 (x, y) rhombus vertices
            in clockwise or counter-clockwise order.
        k: CSS-superellipse-like parameter::

            k < 0   concave / scooped
            k = 0   diamond
            k = 1   ellipse
            k > 1   increasingly square

        num_curves: Total number of Bezier segments. Must be a multiple of 4.
            The default is 16.

    Returns:
        List of (P0, C1, C2, P1) Cubic Bezier segments.
    """
    if len(points) != 4:
        raise ValueError("Exactly four vertices are required")

    # Make sure num curves is a multiple of 4 within a reasonable range.
    num_curves = (num_curves // 4) * 4
    num_curves = max(min(num_curves, _MAX_CURVES), _MIN_CURVES)

    # This would be a simple rectangle of four straight line Beziers
    # connected at rhombus edge midpoints.
    if _is_zero(k):
        mp1, mp2, mp3, mp4 = rect_midpoints(points)
        return [
            (mp1, mp1, mp2, mp2),
            (mp2, mp2, mp3, mp3),
            (mp3, mp3, mp4, mp4),
            (mp4, mp4, mp1, mp1),
        ]

    n = 2.0**k

    # ------------------------------------------------------------
    # First-quadrant segments.
    #
    # Each segment is initially treated as the quadratic:
    #
    #          P0, Q, P1
    #
    # where Q is the intersection of the endpoint tangents.
    #
    # Degree elevation converts it EXACTLY to cubic:
    #
    #     C1 = 1/3 P0 + 2/3 Q
    #     C2 = 2/3 Q  + 1/3 P1
    #
    # Thus the returned cubic has the geometry of a quadratic and
    # cannot acquire an S-shaped inflection.
    # ------------------------------------------------------------

    base_curves = []

    curves_per_quadrant = num_curves // 4
    step = _HALF_PI / curves_per_quadrant

    for i in range(curves_per_quadrant):
        theta = i * step
        p0 = _lame_point(theta, n)
        p1 = _lame_point(theta + step, n)
        t0 = _lame_tangent(theta, n)
        t1 = _lame_tangent(theta + step, n)
        qx, qy = _tangent_intersection(p0, t0, p1, t1)

        # Exact elevation of quadratic P0-Q-P1 to cubic P0-C1-C2-P1.
        c1 = (
            (p0[0] + 2.0 * qx) / 3.0,
            (p0[1] + 2.0 * qy) / 3.0,
        )
        c2 = (
            (p1[0] + 2.0 * qx) / 3.0,
            (p1[1] + 2.0 * qy) / 3.0,
        )

        base_curves.append((p0, c1, c2, p1))

    # ------------------------------------------------------------
    # Affine map from normalized square to rhombus
    # ------------------------------------------------------------

    center = (
        0.25 * sum(x for x, _y in points),
        0.25 * sum(y for _x, y in points),
    )

    ux = 0.5 * (points[0][0] - points[1][0])
    uy = 0.5 * (points[0][1] - points[1][1])

    vx = 0.5 * (points[0][0] - points[3][0])
    vy = 0.5 * (points[0][1] - points[3][1])

    def transform(point: TPoint, quadrant: int) -> TPoint:
        x, y = point
        # First rotate to quadrant
        if quadrant == 1:
            x, y = -y, x
        elif quadrant == 2:
            x, y = -x, -y
        elif quadrant == 3:
            x, y = y, -x
        # Affine transform to rhombus
        return (
            center[0] + x * ux + y * vx,
            center[1] + x * uy + y * vy,
        )

    # ------------------------------------------------------------
    # Generate all four quadrants and transform into the rhombus.
    # ------------------------------------------------------------

    curves: list[tuple[TPoint, TPoint, TPoint, TPoint]] = []

    for quadrant in range(4):
        for curve in base_curves:
            p0, c1, c2, p1 = curve
            curves.append((
                transform(p0, quadrant),
                transform(c1, quadrant),
                transform(c2, quadrant),
                transform(p1, quadrant),
            ))
            # transformed: tuple[TPoint, TPoint, TPoint, TPoint] = tuple(
            #    transform(point, quadrant) for point in curve
            # )
            # curves.append(transformed)

    return curves
