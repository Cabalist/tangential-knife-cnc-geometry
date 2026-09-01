"""Pytest fixtures."""

from __future__ import annotations

import pytest

import geom2d


@pytest.fixture(scope="session", autouse=True)
def _initialize() -> None:
    geom2d.set_epsilon(1e-8)
