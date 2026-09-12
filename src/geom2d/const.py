"""Tolerance constants and float comparison helpers.

``EPSILON`` is the library-wide tolerance and it is a *distance*. It can be
changed once with :func:`set_epsilon` so a consumer can match it to its own
units; every derived constant is recomputed by the same function at import
time and on change. Other modules read these values through the module
(``const.EPSILON``) at call time and never bind them at import.

Which comparison to use
-----------------------

Absolute (the quantity is a distance):
    :func:`float_eq` and :func:`is_zero` for coordinates, lengths, radii and
    distances; ``P.almost_equal`` for point coincidence.

Relative (the quantity is not a distance):
    :func:`is_zero_rel` for discriminants and areas, scaled by the natural
    size of the problem (``|v| ** 2``, ``|v| ** 4``, a perimeter);
    :func:`cross_is_zero` for a cross product, which scales with the length
    of the reference vector; :func:`is_parallel` for two vectors.

Angles:
    :func:`angle_eq` for direction and tangent angles, where ``+pi`` and
    ``-pi`` are the same direction; :func:`float_eq` for *signed sweep*
    angles, where they are different arcs.

Identity versus coincidence:
    ``P.__eq__`` and ``hash(P)`` use grid cells (:func:`cell`) and agree with
    each other, which makes points usable in sets and dicts. Use
    ``P.almost_equal`` when you mean "geometrically the same point".
"""

import math
from typing import Final

TAU: Final[float] = math.tau
"""The full turn, ``2 * pi``."""

EPSILON: float = 1e-8
"""Tolerance for float comparisons, as a distance. Change it with :func:`set_epsilon`."""

EPSILON2: float = 1e-16
"""``EPSILON ** 2``, for comparing squared distances without a square root."""

EPSILON_PRECISION: int = 8
"""Significant digits after the decimal point that ``EPSILON`` resolves."""

REPSILON: float = 1e8
"""``10 ** EPSILON_PRECISION``; multiplies a coordinate into grid-cell units."""


def _derive(eps: float) -> None:
    """Set ``EPSILON`` and every constant derived from it."""
    global EPSILON, EPSILON2, EPSILON_PRECISION, REPSILON  # noqa: PLW0603 - the mutable module constants are the API
    EPSILON = eps
    EPSILON2 = eps * eps
    EPSILON_PRECISION = max(0, round(abs(math.log10(eps))))
    REPSILON = 10.0**EPSILON_PRECISION


_derive(EPSILON)


def set_epsilon(value: float) -> float:
    """Set the library tolerance and recompute the derived constants.

    Args:
        value: The new tolerance. Must satisfy ``0 < value < 1``.

    Returns:
        The previous tolerance, so it can be restored.

    Raises:
        ValueError: If ``value`` is not a finite number strictly between 0 and 1.
            Nothing is changed in that case.
    """
    eps = float(value)
    if not (math.isfinite(eps) and 0.0 < eps < 1.0):
        raise ValueError(f"epsilon must be a finite number in (0, 1), got {value!r}")
    previous = EPSILON
    _derive(eps)
    return previous


def float_eq(a: float, b: float, tolerance: float | None = None) -> bool:
    """Return True if two floats are equal within ``tolerance``.

    The tolerance is absolute for magnitudes up to 1.0 and scales with the
    larger magnitude above that, so large coordinates compare sensibly.
    """
    tol = EPSILON if tolerance is None else tolerance
    aa = -a if a < 0 else a
    bb = -b if b < 0 else b
    ab_max = aa if aa > bb else bb
    if ab_max > 1.0:
        tol *= ab_max
    return (a - b if a > b else b - a) < tol


def angle_eq(a: float, b: float, tolerance: float | None = None) -> bool:
    """Return True if two direction angles are equal within ``tolerance``.

    Angles at ``+pi`` and ``-pi`` describe the same direction and compare
    equal here; use :func:`float_eq` for signed sweep angles instead.
    """
    tol = EPSILON if tolerance is None else tolerance
    if float_eq(a, b, tol):
        return True
    aa = -a if a < 0 else a
    bb = -b if b < 0 else b
    return abs(math.pi - aa) < tol and abs(math.pi - bb) < tol


def is_zero(value: float, tolerance: float | None = None) -> bool:
    """Return True if ``value`` is within ``tolerance`` of zero (absolute)."""
    tol = EPSILON if tolerance is None else tolerance
    return -tol < value < tol


def is_zero_rel(value: float, scale: float, tolerance: float | None = None) -> bool:
    """Return True if ``value`` is zero relative to ``scale``.

    Use it for quantities that are not distances: a discriminant scaled by
    ``|v| ** 4``, an area scaled by a perimeter, and so on.
    """
    tol = EPSILON if tolerance is None else tolerance
    return abs(value) < tol * scale


def cross_is_zero(cross: float, length: float, tolerance: float | None = None) -> bool:
    """Return True if a 2D cross product is zero relative to its reference vector.

    ``cross = v1 x v2`` equals ``|v1| * d`` where ``d`` is the perpendicular
    distance of ``v2``'s tip from the line through ``v1``, so comparing
    ``|cross| / |v1|`` with ``EPSILON`` keeps ``EPSILON`` a distance.
    """
    tol = EPSILON if tolerance is None else tolerance
    return abs(cross) < tol * length


def is_parallel(cross: float, length1: float, length2: float, tolerance: float | None = None) -> bool:
    """Return True if two vectors with the given cross product are parallel.

    ``|cross| / (|v1| |v2|)`` is the sine of the angle between the vectors.
    """
    tol = EPSILON if tolerance is None else tolerance
    return abs(cross) < tol * length1 * length2


def float_round(value: float) -> float:
    """Round ``value`` to the precision that ``EPSILON`` resolves."""
    return round(value, EPSILON_PRECISION)


def cell(value: float) -> int:
    """Return the grid cell of a coordinate at the current precision.

    Two coordinates in the same cell are equal for hashing purposes.
    """
    return round(value * REPSILON)
