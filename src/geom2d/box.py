"""Axis-aligned bounding box."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from . import const
from .errors import GeometryError
from .point import P

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .point import PointLike


class HasBoundingBox(Protocol):
    """Anything with a ``bounding_box`` property, such as a path segment."""

    @property
    def bounding_box(self) -> Box: ...


@dataclass(frozen=True, slots=True)
class Box:
    """An axis-aligned rectangle given by its minimum and maximum corners.

    The corners are canonicalised on construction, so ``Box(a, b)`` and
    ``Box(b, a)`` are the same box. A box may be degenerate (zero width or
    height); that is how a single point or an axis-aligned segment bounds.
    """

    p1: P
    """Minimum corner (lowest x and y)."""
    p2: P
    """Maximum corner (highest x and y)."""

    def __post_init__(self) -> None:
        a = P.of(self.p1)
        b = P.of(self.p2)
        object.__setattr__(self, "p1", P(min(a.x, b.x), min(a.y, b.y)))
        object.__setattr__(self, "p2", P(max(a.x, b.x), max(a.y, b.y)))

    # ----- construction -------------------------------------------------

    @classmethod
    def from_points(cls, points: Iterable[PointLike]) -> Box:
        """The bounding box of an iterable of points (consumed once).

        Raises:
            GeometryError: If ``points`` is empty.
        """
        xmin = ymin = float("inf")
        xmax = ymax = float("-inf")
        for obj in points:
            p = P.of(obj)
            xmin = min(xmin, p.x)
            ymin = min(ymin, p.y)
            xmax = max(xmax, p.x)
            ymax = max(ymax, p.y)
        if xmin > xmax:
            raise GeometryError("cannot build a bounding box from no points")
        return cls(P(xmin, ymin), P(xmax, ymax))

    @classmethod
    def from_path(cls, path: Iterable[HasBoundingBox]) -> Box:
        """The union of the bounding boxes of a path's segments (consumed once).

        Raises:
            GeometryError: If ``path`` is empty.
        """
        result: Box | None = None
        for segment in path:
            box = segment.bounding_box
            result = box if result is None else result.union(box)
        if result is None:
            raise GeometryError("cannot build a bounding box from an empty path")
        return result

    # ----- derived values -----------------------------------------------

    @property
    def xmin(self) -> float:
        """Left edge."""
        return self.p1.x

    @property
    def ymin(self) -> float:
        """Bottom edge."""
        return self.p1.y

    @property
    def xmax(self) -> float:
        """Right edge."""
        return self.p2.x

    @property
    def ymax(self) -> float:
        """Top edge."""
        return self.p2.y

    @property
    def width(self) -> float:
        """Extent along x."""
        return self.p2.x - self.p1.x

    @property
    def height(self) -> float:
        """Extent along y."""
        return self.p2.y - self.p1.y

    @property
    def size(self) -> P:
        """``(width, height)`` as a vector."""
        return P(self.width, self.height)

    @property
    def center(self) -> P:
        """The centre point."""
        return P((self.p1.x + self.p2.x) / 2.0, (self.p1.y + self.p2.y) / 2.0)

    @property
    def area(self) -> float:
        """``width * height``."""
        return self.width * self.height

    @property
    def is_degenerate(self) -> bool:
        """True if the box has no width or no height (within ``EPSILON``)."""
        return self.width < const.EPSILON or self.height < const.EPSILON

    # ----- relations ----------------------------------------------------

    def union(self, other: Box) -> Box:
        """The smallest box containing both boxes."""
        return Box(
            P(min(self.p1.x, other.p1.x), min(self.p1.y, other.p1.y)),
            P(max(self.p2.x, other.p2.x), max(self.p2.y, other.p2.y)),
        )

    def intersection(self, other: Box) -> Box | None:
        """The overlap of the two boxes, or None if they do not overlap.

        Boxes that only touch along an edge or at a corner return a
        degenerate box, not None.
        """
        xmin = max(self.p1.x, other.p1.x)
        ymin = max(self.p1.y, other.p1.y)
        xmax = min(self.p2.x, other.p2.x)
        ymax = min(self.p2.y, other.p2.y)
        if xmin > xmax + const.EPSILON or ymin > ymax + const.EPSILON:
            return None
        return Box(P(xmin, ymin), P(max(xmin, xmax), max(ymin, ymax)))

    def contains_point(self, p: P) -> bool:
        """True if ``p`` lies inside the box or on its boundary (within ``EPSILON``)."""
        eps = const.EPSILON
        return self.p1.x - eps <= p.x <= self.p2.x + eps and self.p1.y - eps <= p.y <= self.p2.y + eps

    def contains_box(self, other: Box) -> bool:
        """True if ``other`` lies entirely inside this box (within ``EPSILON``)."""
        return self.contains_point(other.p1) and self.contains_point(other.p2)

    def __str__(self) -> str:
        return f"Box({self.p1}, {self.p2})"
