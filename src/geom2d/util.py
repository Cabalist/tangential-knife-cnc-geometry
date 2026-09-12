"""Angle helpers and number formatting."""

import functools
import math
from typing import TYPE_CHECKING

from . import const

if TYPE_CHECKING:
    from collections.abc import Callable


def normalize_angle(angle: float, center: float = math.pi) -> float:
    """Normalize ``angle`` into a full turn centred on ``center``.

    ``normalize_angle(a)`` maps into ``[0, 2*pi)``;
    ``normalize_angle(a, center=0.0)`` maps into ``[-pi, pi)``.
    """
    return angle - (const.TAU * math.floor((angle + math.pi - center) / const.TAU))


def calc_rotation(start_angle: float, end_angle: float) -> float:
    """Return the shortest signed rotation from ``start_angle`` to ``end_angle``.

    The result lies in ``[-pi, pi]``; positive is counter-clockwise.
    Angles that are equal within ``EPSILON`` give ``0.0``.
    """
    if const.float_eq(start_angle, end_angle):
        return 0.0
    rotation = normalize_angle(end_angle, 0.0) - normalize_angle(start_angle, 0.0)
    if rotation < -math.pi:
        rotation += const.TAU
    elif rotation > math.pi:
        rotation -= const.TAU
    return rotation


@functools.lru_cache(maxsize=32)
def _formatter(scale: float, precision: int) -> Callable[[float], str]:
    def fmt(value: float) -> str:
        text = f"{value * scale:.{precision}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return "0" if text in {"", "-0"} else text

    return fmt


def float_formatter(*, scale: float = 1.0, precision: int | None = None) -> Callable[[float], str]:
    """Return a function that formats floats compactly at a fixed precision.

    Trailing zeros and a trailing decimal point are removed, so ``1.50``
    becomes ``"1.5"`` and ``100.0`` stays ``"100"``. Negative zero prints as
    ``"0"``. Formatters are cached per ``(scale, precision)``.

    Args:
        scale: Multiplier applied before formatting (for unit conversion).
        precision: Digits after the decimal point. Defaults to the precision
            that ``EPSILON`` resolves.
    """
    if precision is None:
        precision = const.EPSILON_PRECISION
    return _formatter(float(scale), int(precision))
