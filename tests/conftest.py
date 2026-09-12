"""Shared pytest fixtures."""

from typing import TYPE_CHECKING

import pytest

from geom2d import const

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="session", autouse=True)
def _default_epsilon() -> None:
    const.set_epsilon(1e-8)


@pytest.fixture
def restore_epsilon() -> Iterator[None]:
    """Let a test change EPSILON and restore it afterwards (tests run in random order)."""
    previous = const.EPSILON
    try:
        yield
    finally:
        const.set_epsilon(previous)
