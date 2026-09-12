"""Tolerance constants and float comparison helpers.

``EPSILON`` is the library-wide tolerance and it is a *distance*. It can be
changed once with :func:`set_epsilon` so a consumer can match it to its own
units; every derived constant is recomputed by the same function at import
time and on change. Other modules read these values through the module
(``const.EPSILON``) at call time and never bind them at import.

Which comparison to use
-----------------------

Absolute (the quantity is a distance):
    :func:`float_eq` and :func:`is_zero` compare coordinates, lengths, radii
    and distances against ``EPSILON`` as an absolute difference, at every
    magnitude; ``P.almost_equal`` is the same test for point coincidence.
    Nothing in the library scales a distance tolerance with the size of the
    values, so constructor validation and later queries agree.

Relative (the quantity is not a distance):
    :func:`is_zero_rel` for discriminants and areas, scaled by the natural
    size of the problem (``|v| ** 2``, ``|v| ** 4``, a perimeter);
    :func:`cross_is_zero` for a cross product, which scales with the length
    of the reference vector; :func:`is_parallel` for two vectors. Each of
    these reduces to a distance compared with ``EPSILON``.

Angles:
    :func:`angle_eq` for direction and tangent angles: directions that
    differ by a whole number of turns are the same, so ``+pi`` and ``-pi``
    agree and so do ``0`` and ``tau``. Use :func:`float_eq` for *signed
    sweep* angles, where a full turn and no turn are different arcs.

Identity versus coincidence:
    ``P.__eq__`` and ``hash(P)`` use grid cells (:func:`cell`) and agree with
    each other, which makes points usable in sets and dicts. Use
    ``P.almost_equal`` when you mean "geometrically the same point".

Lifetime of ``EPSILON``:
    Hashes and grid equality depend on the current ``EPSILON``. Call
    :func:`set_epsilon` once, at startup, before any geometry object is
    created or stored in a set or dict; objects hashed under a different
    ``EPSILON`` are not comparable with ones hashed afterwards.

Coordinate envelope:
    A double keeps about 16 significant digits, so a coordinate of magnitude
    ``|x|`` is rounded to about ``1e-16 * |x|``. Tolerances are absolute
    distances, so ``EPSILON`` must stay well above that rounding: roughly
    ``|x| < 1e7`` at the default ``1e-8``. A direction derived from a
    feature of size ``s`` (a radius, a segment length) carries that rounding
    divided by ``s``, about ``1e-16 * |x| / s`` radians, so tangent
    directions agree within ``EPSILON`` only for features larger than
    ``1e-8 * |x|``: at ``|x| = 1e6`` that is a radius or length of ``0.01``.

Resolution floor:
    ``EPSILON`` is a numerical floor, not a physical one. It exists so that
    hashing, coincidence, the ``Arc`` invariant and the approximation checks
    agree with each other; nothing a cutting process can produce comes
    anywhere near it. Choose ``EPSILON`` several orders of magnitude below
    the process resolution (in millimetres, a knife that resolves ``0.01``
    is a million times coarser than the default ``1e-8``) and pass
    tolerances at the process resolution. Above that physical floor the
    contracts in this library describe the geometry a machine will see.
    Below it the library promises only self-consistency: no crashes, no
    silently dropped geometry, connected output, and an explicit error
    where a request cannot be met at all (a sweep limit that would need
    arcs shorter than ``EPSILON``). Drop or simplify features below your
    own floor before approximating them; do not rely on what happens near
    ``EPSILON`` to mean anything physical.
"""

import math
from typing import Final

TAU: Final[float] = math.tau
"""The full turn, ``2 * pi``."""

MIN_EPSILON: Final[float] = 1e-15
"""Smallest accepted ``EPSILON``: below it a double cannot resolve the tolerance at magnitude 1."""

EPSILON: float = 1e-8
"""Tolerance for float comparisons, as a distance. Change it with :func:`set_epsilon`."""

EPSILON2: float = 1e-16
"""``EPSILON ** 2``, for comparing squared distances without a square root."""

EPSILON_PRECISION: int = 8
"""Significant digits after the decimal point that ``EPSILON`` resolves."""

REPSILON: float = 1e8
"""``10 ** EPSILON_PRECISION``; multiplies a coordinate into grid-cell units."""


def _derive(eps: float) -> None:
    """Set ``EPSILON`` and every constant derived from it, computing all of them before assigning any."""
    global EPSILON, EPSILON2, EPSILON_PRECISION, REPSILON
    precision = max(0, round(abs(math.log10(eps))))
    repsilon = 10.0**precision
    EPSILON, EPSILON2, EPSILON_PRECISION, REPSILON = eps, eps * eps, precision, repsilon


_derive(EPSILON)


def set_epsilon(value: float) -> float:
    """Set the library tolerance and recompute the derived constants.

    Args:
        value: The new tolerance, at least :data:`MIN_EPSILON` and below 1.

    Call this once at startup, before creating geometry: it changes the
    grid used by ``__eq__`` and ``__hash__`` on every geometry class, so
    objects already stored in sets or dicts stop matching (see the module
    docstring).

    Returns:
        The previous tolerance, so it can be restored.

    Raises:
        ValueError: If ``value`` is not a finite number in ``[MIN_EPSILON, 1)``.
            Nothing is changed in that case.
    """
    eps = float(value)
    if not (math.isfinite(eps) and MIN_EPSILON <= eps < 1.0):
        raise ValueError(f"epsilon must be a finite number in [{MIN_EPSILON!r}, 1), got {value!r}")
    previous = EPSILON
    _derive(eps)
    return previous


def float_eq(a: float, b: float, tolerance: float | None = None) -> bool:
    """Return True if ``|a - b|`` is below ``tolerance`` (default ``EPSILON``).

    The comparison is absolute at every magnitude, like every distance
    comparison in the library; see the module docstring for the coordinate
    envelope this implies.
    """
    tol = EPSILON if tolerance is None else tolerance
    return (a - b if a > b else b - a) < tol


def angle_eq(a: float, b: float, tolerance: float | None = None) -> bool:
    """Return True if two direction angles agree within ``tolerance`` (default ``EPSILON``).

    Directions that differ by a whole number of turns are the same: ``+pi``
    and ``-pi`` compare equal, and so do ``0`` and ``tau``. The difference
    is reduced with a signed remainder, so it stays exact near zero and the
    comparison is symmetric at any tolerance. Use :func:`float_eq` for
    signed sweep angles instead.
    """
    tol = EPSILON if tolerance is None else tolerance
    return abs(math.remainder(a - b, TAU)) < tol


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
    ``|cross| / |v1|`` with ``EPSILON`` keeps ``EPSILON`` a distance. An
    exactly zero cross product is parallel at any scale, including a zero
    reference length.
    """
    if cross == 0.0:
        return True
    tol = EPSILON if tolerance is None else tolerance
    return abs(cross) < tol * length


def is_parallel(cross: float, length1: float, length2: float, tolerance: float | None = None) -> bool:
    """Return True if two vectors with the given cross product are parallel.

    ``|cross| / (|v1| |v2|)`` is the sine of the angle between the vectors.
    An exactly zero cross product is parallel at any scale.
    """
    if cross == 0.0:
        return True
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
