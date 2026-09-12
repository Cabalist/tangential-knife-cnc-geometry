"""Immutable 2D point, also used as a vector.

Orientation convention (shared by the whole package): angles are measured
counter-clockwise from the positive x axis; a positive cross product,
``winding`` or area means counter-clockwise (left).
"""

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from . import const, util
from .errors import GeometryError


class HasXY(Protocol):
    """Anything that exposes ``x`` and ``y`` coordinates, such as a parser's point type."""

    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...


type PointLike = P | HasXY | Sequence[float]
"""Inputs that :meth:`P.of` converts to a :class:`P`."""


@dataclass(frozen=True, slots=True, eq=False)
class P:
    """A 2D point or vector with float coordinates.

    ``P`` is a frozen dataclass: hashable, picklable, positional
    (``P(x, y)``, ``match P(x, y)``) and unpackable (``x, y = p``). It is not
    a sequence: use ``.x`` and ``.y``.

    Equality and hashing use the ``EPSILON`` grid (see :func:`geom2d.const.cell`),
    so equal points always hash equal and points can be dictionary keys. That
    is identity for dedup; for geometric coincidence use :meth:`almost_equal`.

    Arithmetic: ``p + q``, ``p - q``, ``-p``, ``p * k``, ``k * p``, ``p / k``.
    Operands of other types return ``NotImplemented`` (a ``TypeError`` to the
    caller). ``sum(points, P(0, 0))`` works.
    """

    x: float
    y: float

    def __post_init__(self) -> None:
        # Normalise ints and other reals so the fields are always float.
        if type(self.x) is not float:
            object.__setattr__(self, "x", float(self.x))
        if type(self.y) is not float:
            object.__setattr__(self, "y", float(self.y))

    # ----- construction -------------------------------------------------

    @classmethod
    def of(cls, obj: PointLike) -> P:
        """Convert a point-like object to a ``P``.

        Accepts a ``P`` (returned as is), any object with ``x`` and ``y``
        attributes, or a sequence of two numbers.

        Raises:
            GeometryError: If ``obj`` cannot be read as two numeric coordinates.
        """
        if isinstance(obj, P):
            return obj
        if isinstance(obj, Sequence):
            try:
                x, y = obj
            except ValueError as exc:
                raise GeometryError(f"a point needs exactly two coordinates, got {obj!r}") from exc
            try:
                return cls(float(x), float(y))
            except (TypeError, ValueError) as exc:
                raise GeometryError(f"point coordinates must be numbers, got {obj!r}") from exc
        try:
            return cls(float(obj.x), float(obj.y))
        except (AttributeError, TypeError, ValueError) as exc:
            raise GeometryError(f"cannot convert {obj!r} to a point") from exc

    @classmethod
    def from_polar(cls, mag: float, angle: float) -> P:
        """Create a point from polar coordinates (radius, angle in radians)."""
        return cls(mag * math.cos(angle), mag * math.sin(angle))

    # ----- protocol -----------------------------------------------------

    def __iter__(self) -> Iterator[float]:
        return iter((self.x, self.y))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, P):
            return NotImplemented
        return const.cell(self.x) == const.cell(other.x) and const.cell(self.y) == const.cell(other.y)

    def __hash__(self) -> int:
        return hash((const.cell(self.x), const.cell(self.y)))

    def __str__(self) -> str:
        precision = const.EPSILON_PRECISION
        return f"({self.x:.{precision}f}, {self.y:.{precision}f})"

    # ----- arithmetic ---------------------------------------------------

    def __add__(self, other: P) -> P:
        if not isinstance(other, P):
            return NotImplemented
        return P(self.x + other.x, self.y + other.y)

    def __sub__(self, other: P) -> P:
        if not isinstance(other, P):
            return NotImplemented
        return P(self.x - other.x, self.y - other.y)

    def __neg__(self) -> P:
        return P(-self.x, -self.y)

    def __mul__(self, scalar: float) -> P:
        if type(scalar) is not float and type(scalar) is not int:
            return NotImplemented
        return P(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> P:
        if type(scalar) is not float and type(scalar) is not int:
            return NotImplemented
        return P(self.x / scalar, self.y / scalar)

    def __abs__(self) -> float:
        return math.hypot(self.x, self.y)

    # ----- derived values -----------------------------------------------

    @property
    def length(self) -> float:
        """Distance from the origin (the vector's magnitude)."""
        return math.hypot(self.x, self.y)

    @property
    def length2(self) -> float:
        """Squared length; cheaper than ``length`` when only comparing."""
        return self.x * self.x + self.y * self.y

    @property
    def angle(self) -> float:
        """Direction of the vector in radians, in ``(-pi, pi]``."""
        return math.atan2(self.y, self.x)

    @property
    def is_zero(self) -> bool:
        """True if the vector is within ``EPSILON`` of the origin."""
        return self.length2 < const.EPSILON2

    @property
    def unit(self) -> P:
        """The vector scaled to unit length; the zero vector stays zero."""
        if self.is_zero:
            return P(0.0, 0.0)
        ln = self.length
        return P(self.x / ln, self.y / ln)

    def to_polar(self) -> tuple[float, float]:
        """Return ``(radius, angle)``."""
        return (self.length, self.angle)

    def normal(self, *, left: bool = True) -> P:
        """Return the perpendicular vector, to the left of travel by default."""
        return P(-self.y, self.x) if left else P(self.y, -self.x)

    def rotate(self, angle: float, origin: P | None = None) -> P:
        """Return a copy rotated counter-clockwise by ``angle`` radians about ``origin``."""
        if const.is_zero(angle):
            return self
        ox, oy = (0.0, 0.0) if origin is None else (origin.x, origin.y)
        dx = self.x - ox
        dy = self.y - oy
        c = math.cos(angle)
        s = math.sin(angle)
        return P(ox + dx * c - dy * s, oy + dx * s + dy * c)

    # ----- relations with other points -----------------------------------

    def almost_equal(self, other: P, tolerance: float | None = None) -> bool:
        """True if the distance to ``other`` is below ``tolerance`` (default ``EPSILON``).

        This is geometric coincidence; ``==`` is grid identity.
        """
        tol = const.EPSILON if tolerance is None else tolerance
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy < tol * tol

    def dot(self, other: P) -> float:
        """Dot product."""
        return self.x * other.x + self.y * other.y

    def cross(self, other: P) -> float:
        """2D cross product (perp-dot product); positive if ``other`` is to the left."""
        return self.x * other.y - self.y * other.x

    def distance(self, other: P) -> float:
        """Euclidean distance to ``other``."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def distance2(self, other: P) -> float:
        """Squared distance to ``other``."""
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy

    def angle2(self, p1: P, p2: P) -> float:
        """Signed angle of the turn ``p1 -> self -> p2``, in ``(-pi, pi]``.

        Positive when ``p2`` is counter-clockwise from ``p1`` as seen from
        ``self``; ``0.0`` if either point coincides with ``self``.
        """
        v1 = p1 - self
        v2 = p2 - self
        if v1.is_zero or v2.is_zero or v1.almost_equal(v2):
            return 0.0
        return math.atan2(v1.cross(v2), v1.dot(v2))

    def ccw_angle2(self, p1: P, p2: P) -> float:
        """Counter-clockwise angle of the turn ``p1 -> self -> p2``, in ``[0, 2*pi)``."""
        return util.normalize_angle(self.angle2(p1, p2))

    def distance_to_line(self, p1: P, p2: P) -> float:
        """Perpendicular distance to the infinite line through ``p1`` and ``p2``.

        A degenerate line (``p1`` coincident with ``p2``) gives the distance to ``p1``.
        """
        v1 = p2 - p1
        seglen = v1.length
        if seglen < const.EPSILON:
            return self.distance(p1)
        return abs(v1.cross(self - p1)) / seglen

    def normal_projection(self, other: P) -> float:
        """Parameter of ``other``'s projection onto this vector (0 at the origin, 1 at the tip).

        The zero vector projects everything to ``0.0``.
        """
        vlen2 = self.length2
        if vlen2 < const.EPSILON2:
            return 0.0
        return self.dot(other) / vlen2

    def winding(self, p2: P, p3: P) -> int:
        """Orientation of the turn ``self -> p2 -> p3``.

        Returns ``1`` for counter-clockwise (left), ``-1`` for clockwise
        (right) and ``0`` when the points are collinear within ``EPSILON``
        (measured as a perpendicular distance, so it does not depend on scale).
        """
        v1 = p2 - self
        v2 = p3 - self
        cross = v1.cross(v2)
        if const.cross_is_zero(cross, max(v1.length, v2.length)):
            return 0
        return 1 if cross > 0 else -1

    def colinear(self, p2: P, p3: P, tolerance: float | None = None) -> bool:
        """True if ``self``, ``p2`` and ``p3`` lie on one line within ``tolerance``."""
        v1 = p2 - self
        v2 = p3 - self
        return const.cross_is_zero(v1.cross(v2), max(v1.length, v2.length), tolerance)

    # ----- output -------------------------------------------------------

    def to_svg(self, *, scale: float = 1.0, precision: int | None = None) -> str:
        """Return ``"x,y"`` formatted for an SVG path."""
        fmt = util.float_formatter(scale=scale, precision=precision)
        return f"{fmt(self.x)},{fmt(self.y)}"
