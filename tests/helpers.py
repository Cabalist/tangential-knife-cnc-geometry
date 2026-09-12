"""Helpers shared by the test modules."""

import itertools

from geom2d import angle_eq


class XY:
    """A foreign point type exposing x/y attributes, like an SVG parser's point."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def g1_everywhere(segments, tolerance: float = 1e-6) -> bool:
    """True if consecutive segments meet with matching tangent directions."""
    return all(angle_eq(a.end_tangent_angle, b.start_tangent_angle, tolerance) for a, b in itertools.pairwise(segments))
