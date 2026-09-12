"""Exceptions raised for invalid geometry."""


class GeometryError(ValueError):
    """Public input describes geometry that cannot be constructed or operated on."""


class DegenerateGeometryError(GeometryError):
    """The input is degenerate: zero length, zero area, or coincident points."""
