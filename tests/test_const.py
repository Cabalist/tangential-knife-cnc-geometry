"""Tolerance constants and float comparisons."""

import math

import pytest

from geom2d import const


def test_derived_constants_follow_epsilon(restore_epsilon: None):
    previous = const.set_epsilon(1e-6)
    assert previous == 1e-8
    assert const.EPSILON == 1e-6
    assert pytest.approx(1e-12) == const.EPSILON2
    assert const.EPSILON_PRECISION == 6
    assert const.REPSILON == 1e6
    const.set_epsilon(1e-9)
    assert const.EPSILON_PRECISION == 9
    assert const.float_eq(1.0, 1.0 + 5e-10)
    assert not const.float_eq(1.0, 1.0 + 5e-9)


@pytest.mark.parametrize("bad", [0, -1e-8, 1.0, 2.0, math.inf, math.nan, 1e-16, 1e-320])
def test_set_epsilon_rejects_bad_values_without_mutating(bad: float):
    # Validation happens before any constant changes, and the derived constants
    # are all computed before any is assigned.
    before = (const.EPSILON, const.EPSILON2, const.EPSILON_PRECISION, const.REPSILON)
    with pytest.raises(ValueError, match="epsilon"):
        const.set_epsilon(bad)
    assert before == (const.EPSILON, const.EPSILON2, const.EPSILON_PRECISION, const.REPSILON)
    assert const.float_eq(1.0, 1.0)


def test_set_epsilon_accepts_the_smallest_resolvable_value(restore_epsilon: None):
    const.set_epsilon(const.MIN_EPSILON)
    assert const.EPSILON == 1e-15
    assert const.EPSILON_PRECISION == 15
    assert const.cell(1.0) == 10**15


def test_float_eq_absolute_below_one():
    assert const.float_eq(0.0, 0.5e-8)
    assert not const.float_eq(0.0, 1.5e-8)
    assert const.float_eq(0.3, 0.1 + 0.2)


def test_float_eq_is_absolute_at_every_magnitude():
    # EPSILON is a distance; it does not grow with the values compared, so a
    # query at radius 1000 is as strict as the constructor that validated the arc.
    assert const.float_eq(1e6, 1e6 + 0.5e-8)
    assert not const.float_eq(1e6, 1e6 + 1e-3)
    assert not const.float_eq(1000.0, 1000.0 + 1e-6)
    assert const.float_eq(-1e6, -1e6 - 0.5e-8)


def test_float_eq_is_symmetric():
    for a, b in [(1e6, 1e6 + 1e-3), (0.0, 1e-9), (-5.0, -5.0 + 1e-9)]:
        assert const.float_eq(a, b) == const.float_eq(b, a)


def test_float_eq_explicit_tolerance():
    assert const.float_eq(1.0, 1.05, tolerance=0.1)
    assert not const.float_eq(1.0, 1.2, tolerance=0.1)


def test_angle_eq_at_the_pi_seam():
    # Leftward directions come out as +pi or -pi depending on rounding.
    assert const.angle_eq(math.pi, -math.pi)
    assert const.angle_eq(math.pi - 1e-12, -math.pi + 1e-12)
    assert const.angle_eq(0.5, 0.5 + 1e-9)
    assert not const.angle_eq(0.5, 0.6)
    assert not const.angle_eq(3.0, -3.0)


def test_angle_eq_is_exact_and_symmetric_near_zero():
    # A signed remainder keeps a 1e-16 difference instead of rounding it into a full turn.
    assert not const.angle_eq(0.0, 1e-16, 1e-17)
    assert not const.angle_eq(1e-16, 0.0, 1e-17)
    assert const.angle_eq(0.0, 1e-18, 1e-17)
    assert const.angle_eq(1e-18, 0.0, 1e-17)
    assert const.angle_eq(math.tau, -1e-18, 1e-17)


def test_angle_eq_treats_whole_turns_as_the_same_direction():
    assert const.angle_eq(0.0, math.tau)
    assert const.angle_eq(0.1, 0.1 + 3 * math.tau)
    assert const.angle_eq(-0.1, math.tau - 0.1 - 1e-12)
    assert not const.angle_eq(0.0, math.tau - 1e-6)
    assert const.angle_eq(0.0, math.tau - 1e-6, tolerance=1e-5)


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
    # An exactly zero cross product is parallel at any scale, including a zero reference length.
    assert const.cross_is_zero(0.0, 0.0)
    assert const.is_parallel(0.0, 0.0, 0.0)
    assert not const.cross_is_zero(1e-300, 0.0)
    assert const.is_zero_rel(1e-12, 1e-3)
    assert not const.is_zero_rel(1e-12, 1e-5)


def test_float_round_and_cell():
    assert const.float_round(1.123456789123) == 1.12345679
    assert const.cell(1.0) == 100_000_000
    assert const.cell(1.0 + 0.4e-8) == const.cell(1.0)
    assert const.cell(1.0 + 0.6e-8) == const.cell(1.0) + 1
    assert math.tau == const.TAU
