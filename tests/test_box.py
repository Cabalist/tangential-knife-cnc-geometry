"""Box: axis-aligned bounding box."""

import copy
import pickle

import pytest

from geom2d import GeometryError, P
from geom2d.box import Box


def test_corners_are_canonicalised():
    a = Box(P(3.0, 4.0), P(1.0, 2.0))
    assert a.p1 == P(1.0, 2.0)
    assert a.p2 == P(3.0, 4.0)
    assert a == Box(P(1.0, 2.0), P(3.0, 4.0))


def test_dimensions():
    b = Box(P(1.0, 2.0), P(4.0, 6.0))
    assert (b.xmin, b.ymin, b.xmax, b.ymax) == (1.0, 2.0, 4.0, 6.0)
    assert b.width == 3.0
    assert b.height == 4.0
    assert b.size == P(3.0, 4.0)
    assert b.center == P(2.5, 4.0)
    assert b.area == 12.0
    assert not b.is_degenerate
    assert Box(P(1.0, 1.0), P(1.0, 5.0)).is_degenerate
    assert str(b) == "Box((1.00000000, 2.00000000), (4.00000000, 6.00000000))"


def test_from_points_single_pass_generator():
    # A generator is consumed once.
    b = Box.from_points(P(x, 2 * x) for x in range(3))
    assert b == Box(P(0.0, 0.0), P(2.0, 4.0))
    assert Box.from_points([(5, 5)]) == Box(P(5.0, 5.0), P(5.0, 5.0))  # PointLike input
    with pytest.raises(GeometryError):
        Box.from_points([])
    with pytest.raises(GeometryError):
        Box.from_points(iter(()))


def test_from_path_unions_segment_boxes():
    class Seg:
        def __init__(self, box: Box) -> None:
            self.bounding_box = box

    segs = [Seg(Box(P(0, 0), P(1, 1))), Seg(Box(P(-2, 3), P(0, 4)))]
    assert Box.from_path(segs) == Box(P(-2.0, 0.0), P(1.0, 4.0))
    assert Box.from_path(iter(segs)) == Box(P(-2.0, 0.0), P(1.0, 4.0))
    with pytest.raises(GeometryError):
        Box.from_path([])


def test_union_and_intersection():
    a = Box(P(0, 0), P(4, 4))
    b = Box(P(2, 2), P(6, 6))
    c = Box(P(5, 5), P(7, 7))
    assert a.union(b) == Box(P(0, 0), P(6, 6))
    assert a.intersection(b) == Box(P(2, 2), P(4, 4))
    assert a.intersection(c) is None
    touching = a.intersection(Box(P(4, 0), P(8, 4)))
    assert touching is not None
    assert touching.is_degenerate
    assert touching.width == 0.0


def test_containment():
    a = Box(P(0, 0), P(4, 4))
    assert a.contains_point(P(2, 2))
    assert a.contains_point(P(4, 4))  # boundary counts
    assert a.contains_point(P(4 + 0.5e-8, 2))
    assert not a.contains_point(P(4.1, 2))
    assert a.contains_box(Box(P(1, 1), P(3, 3)))
    assert not a.contains_box(Box(P(1, 1), P(5, 3)))


def test_dataclass_protocol():
    b = Box(P(0, 0), P(1, 1))
    assert copy.deepcopy(b) == b
    assert pickle.loads(pickle.dumps(b)) == b
    assert hash(b) == hash(Box(P(0, 0), P(1, 1)))
    match b:
        case Box(p1, p2):
            assert (p1, p2) == (P(0, 0), P(1, 1))
