"""Exception hierarchy."""

import pytest

from geom2d import DegenerateGeometryError, GeometryError


def test_hierarchy():
    assert issubclass(GeometryError, ValueError)
    assert issubclass(DegenerateGeometryError, GeometryError)


def test_catchable_as_value_error():
    with pytest.raises(ValueError, match="zero"):
        raise DegenerateGeometryError("zero length")
