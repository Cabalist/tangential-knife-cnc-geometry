"""Tolerance constants and float comparisons."""

import math

import pytest

from geom2d import const


def test_derived_constants_follow_epsilon(restore_epsilon):
    previous = const.set_epsilon(1e-6)
    assert previous == 1e-8
    assert const.EPSILON == 1e-6
    assert const.EPSILON2 == pytest.approx(1e-12)
    assert const.EPSILON_PRECISION == 6
    assert const.REPSILON == 1e6
    const.set_epsilon(1e-9)
    assert const.EPSILON_PRECISION == 9
    assert const.float_eq(1.0, 1.0 + 5e-10)
    assert not const.float_eq(1.0, 1.0 + 5e-9)


@pytest.mark.parametrize("bad", [0, -1e-8, 1.0, 2.0, math.inf, math.nan])
def test_set_epsilon_rejects_bad_values_without_mutating(bad):
    # D.4: the old implementation assigned EPSILON before validating.
    before = (const.EPSILON, const.EPSILON2, const.EPSILON_PRECISION, const.REPSILON)
    with pytest.raises(ValueError, match="epsilon"):
        const.set_epsilon(bad)
    assert (const.EPSILON, const.EPSILON2, const.EPSILON_PRECISION, const.REPSILON) == before
    assert const.float_eq(1.0, 1.0)


def test_float_eq_absolute_below_one():
    assert const.float_eq(0.0, 0.5e-8)
    assert not const.float_eq(0.0, 1.5e-8)
    assert const.float_eq(0.3, 0.1 + 0.2)


def test_float_eq_relative_above_one():
    assert const.float_eq(1e6, 1e6 + 1e-3)
    assert not const.float_eq(1e6, 1e6 + 2e-2)
    assert const.float_eq(-1e6, -1e6 - 1e-3)


def test_float_eq_is_symmetric():
    for a, b in [(1e6, 1e6 + 1e-3), (0.0, 1e-9), (-5.0, -5.0 + 1e-9)]:
        assert const.float_eq(a, b) == const.float_eq(b, a)


def test_float_eq_explicit_tolerance():
    assert const.float_eq(1.0, 1.05, tolerance=0.1)
    assert not const.float_eq(1.0, 1.2, tolerance=0.1)


def test_angle_eq_at_the_pi_seam():
    # A1.1: leftward tangents come out as +pi and -pi.
    assert const.angle_eq(math.pi, -math.pi)
    assert const.angle_eq(math.pi - 1e-12, -math.pi + 1e-12)
    assert const.angle_eq(0.5, 0.5 + 1e-9)
    assert not const.angle_eq(0.5, 0.6)
    assert not const.angle_eq(3.0, -3.0)


def test_is_zero():
    assert const.is_zero(0.0)
    assert const.is_zero(0.5e-8)
    assert const.is_zero(-0.5e-8)
    assert not const.is_zero(1e-8)
    assert const.is_zero(0.05, tolerance=0.1)


def test_relative_helpers_keep_epsilon_a_distance():
    # A cross product scales with the reference length; the perpendicular
    # distance it encodes must be compared with EPSILON, not the raw value.
    length = 1e6
    assert const.cross_is_zero(1e6 * 0.5e-8, length)  # 0.5e-8 off a 1e6 line
    assert not const.cross_is_zero(1e6 * 2e-8, length)
    tiny = 1e-5
    assert not const.cross_is_zero(tiny * tiny, tiny)  # perpendicular segments of length 1e-5
    assert const.is_parallel(0.0, 1.0, 1.0)
    assert const.is_parallel(1e-10, 1e-5, 1e-5) is False
    assert const.is_zero_rel(1e-12, 1e-3)
    assert not const.is_zero_rel(1e-12, 1e-5)


def test_float_round_and_cell():
    assert const.float_round(1.123456789123) == 1.12345679
    assert const.cell(1.0) == 100_000_000
    assert const.cell(1.0 + 0.4e-8) == const.cell(1.0)
    assert const.cell(1.0 + 0.6e-8) == const.cell(1.0) + 1
    assert const.TAU == math.tau
