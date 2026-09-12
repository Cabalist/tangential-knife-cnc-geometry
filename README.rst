======
geom2d
======

A small 2D geometry kernel for turning vector artwork into cutting toolpaths.
It provides the primitives a toolpath generator needs and nothing else:

* ``P`` points and vectors, ``Line`` segments, circular ``Arc`` segments and
  ``CubicBezier`` curves, all immutable dataclasses with a validated geometry;
* biarc approximation of Béziers into tangent-continuous arcs and lines
  (``CubicBezier.biarc_approximation``), with a documented output contract;
* arc splitting to a maximum sweep, tangent directions and turn angles at
  joints, and path helpers (``segment`` module) that work on any object
  satisfying the ``Segment`` protocol;
* one tolerance, ``EPSILON``, with clear rules for when a comparison is a
  distance, an angle, or a relative quantity.

SVG parsing, affine transforms and elliptical arcs are deliberately out of
scope: an external parser applies transforms and converts arcs before this
library sees the geometry, and hands it ``P``-like points.

Requires Python 3.14. Pure standard library, no dependencies.

Install and use
---------------

::

    uv add utl-geom2d

::

    from geom2d import P, Line, Arc, CubicBezier, path_is_closed, segments_are_g1

    curve = CubicBezier.from_quadratic(P(0, 0), P(5, 10), P(10, 0))
    segments = curve.biarc_approximation(0.01, max_arc_angle=3.1416 / 2)  # raises if 0.01 cannot be met
    assert all(segments_are_g1(a, b) for a, b in zip(segments, segments[1:]))

Conventions
-----------

* Angles are radians, counter-clockwise from +x. A positive cross product,
  ``winding`` or turn means counter-clockwise (left). ``Arc.angle`` is the
  signed sweep; positive is counter-clockwise. ``offset(+d)`` moves a segment
  to the left of its direction of travel.
* ``EPSILON`` is an absolute distance at every magnitude; ``angle_eq``
  compares directions a whole turn apart as equal. Coordinates should stay
  below about ``1e7``, and tangent directions resolve to ``EPSILON`` only for
  features larger than about ``1e-8`` times the coordinate magnitude.
* ``EPSILON`` is a numerical floor, not a physical one: set it far below the
  process resolution, pass tolerances at the process resolution, and drop
  features below that floor yourself. Near ``EPSILON`` the library promises
  self-consistency (no crashes, nothing silently dropped, connected output),
  never physical meaning.
* ``==`` and ``hash`` on geometry are grid identity at ``EPSILON`` resolution
  (so objects work in sets and dicts); ``P.almost_equal`` tests geometric
  coincidence. ``set_epsilon`` is called once at startup, before any geometry
  is created.
* Invalid geometric input raises ``GeometryError`` (a ``ValueError``); a biarc
  approximation that cannot meet its tolerance raises ``ApproximationError``
  unless ``strict=False``; degenerate input has documented return values
  instead of arithmetic errors; nothing depends on ``assert``.

Development
-----------

::

    uv sync --group dev
    uv run prek run --all-files      # ruff, ruff-format, ty, pyrefly
    uv run pytest
    uv run python -O -m pytest

* Documentation: built from ``docs/`` with Sphinx (``uv sync --group docs``).
* License: LGPL v3. The library descends from Claude Zervas's utl-geom2d
  and keeps its license; the 1.0 rebuild is documented in ``CHANGELOG.rst``.
