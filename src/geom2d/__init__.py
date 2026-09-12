"""2D geometry kernel for toolpath work: points, lines, circular arcs, cubic Béziers."""

from .const import TAU, angle_eq, float_eq, is_zero, set_epsilon
from .errors import DegenerateGeometryError, GeometryError
from .point import HasXY, P, PointLike
from .util import calc_rotation, normalize_angle

__all__ = [
    "TAU",
    "DegenerateGeometryError",
    "GeometryError",
    "HasXY",
    "P",
    "PointLike",
    "angle_eq",
    "calc_rotation",
    "float_eq",
    "is_zero",
    "normalize_angle",
    "set_epsilon",
]
