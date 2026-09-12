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
    ``normalize_angle(a, center=0.0)`` maps into ``[-pi, pi)``. The result
    is kept inside the half-open interval even when rounding of the modulo
    would land on its upper end.
    """
    low = center - math.pi
    result = low + (angle - low) % const.TAU
    if result >= center + math.pi:
        result -= const.TAU
    return result


def calc_rotation(start_angle: float, end_angle: float) -> float:
    """Return the signed rotation that turns direction ``start_angle`` into ``end_angle``.

    The result lies in ``(-pi, pi]``; positive is counter-clockwise and a
    half turn is reported as ``+pi``. The difference is reduced with a signed
    remainder, so a rotation of ``-1e-16`` survives, and no tolerance is
    applied: the caller decides with :func:`geom2d.const.is_zero` (at any
    tolerance) whether the two directions agree.
    """
    rotation = math.remainder(end_angle - start_angle, const.TAU)
    return math.pi if rotation == -math.pi else rotation


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
