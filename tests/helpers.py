"""Helpers shared by the test modules."""

import itertools
from typing import TYPE_CHECKING

from geom2d import path_is_closed, segments_are_g1

if TYPE_CHECKING:
    from collections.abc import Sequence

    from geom2d import Segment
    from geom2d.arc import Arc
    from geom2d.bezier import CubicBezier
    from geom2d.line import Line


class XY:
    """A foreign point type exposing x/y attributes, like an SVG parser's point."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def is_connected_chain(curve: CubicBezier, segments: Sequence[Line | Arc]) -> bool:
    """The biarc output contract's structural part: non-empty, exact ends, exact joints, nothing degenerate."""
    return (
        bool(segments)
        and segments[0].p1 is curve.p1
        and segments[-1].p2 is curve.p2
        and all(not s.is_degenerate for s in segments)
        and all(a.p2.x == b.p1.x and a.p2.y == b.p1.y for a, b in itertools.pairwise(segments))
    )


def g1_everywhere(segments: Sequence[Segment], tolerance: float | None = None) -> bool:
    """True if every joint is tangent-continuous within ``tolerance`` (default ``EPSILON``).

    The closing joint of a closed path counts too.
    """
    joints = list(itertools.pairwise(segments))
    if path_is_closed(segments):
        joints.append((segments[-1], segments[0]))
    return all(segments_are_g1(a, b, angle_tolerance=tolerance) for a, b in joints)
