"""2D geometry kernel for toolpath work: points, lines, circular arcs, cubic Béziers, paths."""

from .arc import Arc
from .bezier import CubicBezier
from .box import Box
from .const import TAU, angle_eq, float_eq, is_zero, set_epsilon
from .errors import ApproximationError, DegenerateGeometryError, GeometryError
from .line import Line
from .point import HasXY, P, PointLike
from .segment import (
    Path,
    Segment,
    heading_change,
    nearest_vertex,
    path_bounding_box,
    path_is_closed,
    path_length,
    path_reversed,
    path_start_at,
    path_to_polyline,
    polyline_to_path,
    segments_are_g1,
    split_path,
    split_path_where,
)
from .util import calc_rotation, normalize_angle

__all__ = [
    "TAU",
    "ApproximationError",
    "Arc",
    "Box",
    "CubicBezier",
    "DegenerateGeometryError",
    "GeometryError",
    "HasXY",
    "Line",
    "P",
    "Path",
    "PointLike",
    "Segment",
    "angle_eq",
    "calc_rotation",
    "float_eq",
    "heading_change",
    "is_zero",
    "nearest_vertex",
    "normalize_angle",
    "path_bounding_box",
    "path_is_closed",
    "path_length",
    "path_reversed",
    "path_start_at",
    "path_to_polyline",
    "polyline_to_path",
    "segments_are_g1",
    "set_epsilon",
    "split_path",
    "split_path_where",
]
