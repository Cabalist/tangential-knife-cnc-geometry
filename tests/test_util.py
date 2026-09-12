"""Angle helpers and float formatting."""

import math

import pytest

from geom2d import const, util


@pytest.mark.parametrize(
    ("angle", "center", "expected"),
    [
        (0.0, math.pi, 0.0),
        (-0.5, math.pi, math.tau - 0.5),
        (math.tau + 0.25, math.pi, 0.25),
        (math.tau, math.pi, 0.0),
        (3.0, 0.0, 3.0),
        (4.0, 0.0, 4.0 - math.tau),
        (-math.pi, 0.0, -math.pi),
    ],
)
def test_normalize_angle(angle, center, expected):
    assert util.normalize_angle(angle, center) == pytest.approx(expected)


def test_normalize_angle_ranges():
    for k in range(-20, 21):
        a = k * 0.37
        assert 0.0 <= util.normalize_angle(a) < math.tau
        assert -math.pi <= util.normalize_angle(a, 0.0) < math.pi


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (0.0, 0.5, 0.5),
        (0.5, 0.0, -0.5),
        (0.0, 0.0, 0.0),
        (3.0, -3.0, math.tau - 6.0),  # shortest way across the seam
        (-3.0, 3.0, -(math.tau - 6.0)),
        (math.pi - 1e-9, -math.pi, 1e-9),
        (0.0, math.tau, 0.0),  # full turn is no rotation
        (0.1, 0.1 + math.tau, 0.0),
    ],
)
def test_calc_rotation(start, end, expected):
    assert util.calc_rotation(start, end) == pytest.approx(expected, abs=1e-9)


def test_calc_rotation_bounds():
    for i in range(-30, 31):
        for j in range(-30, 31):
            rotation = util.calc_rotation(i * 0.41, j * 0.29)
            assert -math.pi <= rotation <= math.pi


def test_float_formatter_strips_only_after_a_decimal_point():
    # A1.2: rstrip("0") on "100" produced "1".
    fmt0 = util.float_formatter(precision=0)
    assert fmt0(100.0) == "100"
    assert fmt0(10.0) == "10"
    assert fmt0(0.0) == "0"
    fmt3 = util.float_formatter(precision=3)
    assert fmt3(1.5) == "1.5"
    assert fmt3(100.0) == "100"
    assert fmt3(0.1234) == "0.123"
    assert fmt3(-0.0001) == "0"
    assert fmt3(-1.25) == "-1.25"


def test_float_formatter_default_precision_tracks_epsilon(restore_epsilon):
    assert util.float_formatter()(1.123456789123) == "1.12345679"
    const.set_epsilon(1e-3)
    assert util.float_formatter()(1.123456789123) == "1.123"


def test_float_formatter_scale_and_cache():
    fmt = util.float_formatter(scale=2.0, precision=1)
    assert fmt(1.25) == "2.5"
    assert util.float_formatter(scale=2.0, precision=1) is fmt
    assert util.float_formatter(scale=2.0, precision=2) is not fmt
