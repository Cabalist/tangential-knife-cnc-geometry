"""2D geometry package."""

from .arc import Arc
from .bezier import CubicBezier
from .box import Box
from .const import (
    TAU,
    float_eq,
    float_round,
    is_zero,
    set_epsilon,
)
from .line import Line, TLine
from .point import P, TPoint
from .util import calc_rotation, normalize_angle, segments_are_g1

__all__ = [
    "TAU",
    "Arc",
    "Box",
    "CubicBezier",
    "Line",
    "P",
    "TLine",
    "TPoint",
    "calc_rotation",
    "float_eq",
    "float_round",
    "is_zero",
    "normalize_angle",
    "segments_are_g1",
    "set_epsilon",
]
