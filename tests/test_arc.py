"""Test geom2d.arc module."""

from __future__ import annotations

import math

import geom2d
from geom2d import const
from geom2d.arc import Arc

# Two arcs that maintain sequential G1 continuity
G1_ARCS = [
    Arc(
        (2.9339552, 4.5527481),
        (2.796207118678724, 5.066830803567288),
        1.0281651740545676,
        0.5235989064807819,
        (1.9057900259454326, 4.5527481),
    ),
    Arc(
        (2.796207118678724, 5.066830803567288),
        (2.4198724, 5.443165499999999),
        1.028165930499886,
        0.5235985729456468,
        (1.9057893708446176, 4.5527477217772585),
    ),
]

ARC_3 = Arc(
    (2.73197710, 3.19424020),
    (3.17520630, 2.75101100),
    0.35000000,
    -4.06426634,
    (3.06375961, 3.08279351),
)

ARC_5 = Arc(
    (2.96989410, 4.80045950),
    (3.41312330, 4.35723030),
    0.31341037,
    -math.pi,
    (3.19150870, 4.57884490),
)


def test_arc_g1() -> None:
    """Test G1 (tangential connection)."""
    assert geom2d.float_eq(G1_ARCS[0].end_tangent_angle(), G1_ARCS[1].start_tangent_angle())


def test_arc_subdivide() -> None:
    """Test Arc subdivision."""
    arcs = ARC_3.subdivide(0.5)
    assert len(arcs) == 2
    assert const.angle_eq(arcs[0].angle, arcs[1].angle)
    assert const.angle_eq(arcs[0].angle + arcs[1].angle, ARC_3.angle)
    assert arcs[0].length() == arcs[1].length()


def test_arc_center() -> None:
    """Test Arc center calculation."""
    c = geom2d.arc.calc_center(ARC_3.p1, ARC_3.p2, ARC_3.radius, ARC_3.angle)
    assert c == ARC_3.center
    c = geom2d.arc.calc_center(ARC_5.p1, ARC_5.p2, ARC_5.radius, ARC_5.angle)
    assert c == ARC_5.center
