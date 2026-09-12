=========
Changelog
=========

1.0.0
=====

A rebuild of the library as the geometry kernel for turning vector artwork
into cutting toolpaths. The public API changed throughout; nothing from 0.x
is kept for compatibility.

Scope
-----

Removed, with their tests: ``voronoi``, ``voronoiclip``, ``planargraph``,
``superellipse``, ``triangle``, ``fillet``, ``polygon`` (including polygon
offsetting and the vendored Clipper port), ``debug`` and ``plotpath``
(debug drawing), ``ellipse`` and ``transform2d``. Affine transforms,
elliptical arcs and SVG endpoint parametrisation belong to the parser that
feeds this library. ``polyline`` and the path parts of ``util`` became
``segment``.

Requires Python 3.14. No runtime dependencies (``typing_extensions`` is
gone).

Data model
----------

* Every class is a frozen, slotted dataclass: ``P(x, y)``, ``Line(p1, p2)``,
  ``Arc(p1, p2, radius, angle, center)``, ``CubicBezier(p1, c1, c2, p2)``,
  ``Box(p1, p2)``. No tuple subclasses: a point is not a sequence, use
  ``.x``/``.y`` (unpacking ``x, y = p`` still works).
* Constructors take ``P`` fields. Factory classmethods take any point-like
  input (``P``, an object with ``x``/``y``, or a pair of numbers): ``P.of``,
  ``Line.from_polar``, ``Arc.from_sweep``, ``Arc.from_two_points_and_tangent``,
  ``CubicBezier.from_quadratic``, ``Box.from_points``.
* ``==`` and ``hash`` are grid identity at ``EPSILON`` resolution on every
  class, so equal objects hash equal. ``P.almost_equal`` is the coincidence
  test. A ``Line`` and its ``reversed()`` differ; so do the two semicircles
  on a chord.
* ``dataclasses.replace``, positional ``match``, ``copy`` and ``pickle`` work
  for every class.
* Zero-argument values are properties: ``length``, ``angle``, ``midpoint``,
  ``start_tangent_angle``, ``end_tangent_angle``, ``start_tangent``,
  ``end_tangent``, ``is_degenerate``, ``bounding_box``. ``reversed()`` replaces
  ``path_reversed()``.

Geometry
--------

* ``Arc`` validates every construction (endpoints on the circle within
  ``EPSILON``, sweep consistent with the endpoints) and raises
  ``GeometryError`` otherwise; ``from_sweep`` computes the center stably,
  including at and near a half turn. All angular queries (``mu``,
  ``point_on_arc``, ``point_inside``, ``subdivide_at_point``, ``height``,
  ``normal_projection_point``, ``bounding_box``) are correct for any sweep,
  including more than half a turn and clockwise arcs. New:
  ``subdivide_equal``, ``split_max_sweep``, ``is_clockwise``, ``direction``,
  tangent vectors, ``tangent_at``. ``extend`` extends. ``offset(+d)`` is
  left of travel, like ``Line.offset``.
* ``Line``: cross-product and parameter tolerances are distances, so
  ``point_on_line``, ``is_parallel``, ``which_side`` and intersections do not
  depend on the segment's length; collinear overlapping segments intersect;
  ``crosses`` is strict on both segments and symmetric; ``same_side`` is
  fixed; ``shift`` and ``extend`` kept. One ``segment`` flag on intersections.
* ``CubicBezier``: ``from_quadratic`` (exact); tangents fall back through
  coincident control points; ``inflections`` and ``find_extrema`` are
  scale-independent and never fabricate roots; ``line_intersection`` handles
  degree-elevated and symmetric curves and honours ``segment=True``;
  ``length`` is bounded and rejects non-positive tolerances.
* ``biarc_approximation(tolerance, *, max_depth, max_arc_angle, strict)``:
  two-sided Hausdorff check, straight pieces become lines only when their
  tangents follow the chord, so tangent continuity holds at every joint;
  the only corners in the output are genuine sub-``EPSILON`` cusps of the
  source curve. ``max_arc_angle`` splits arcs to a maximum sweep;
  ``strict`` raises ``ApproximationError`` when ``max_depth`` is exhausted.
  Tolerances below ``EPSILON`` are refused.
* ``segment``: ``Segment`` protocol, ``Path``, ``path_reversed``,
  ``path_length``, ``path_bounding_box``, ``path_is_closed``,
  ``polyline_to_path``, ``path_to_polyline``, ``nearest_vertex``,
  ``heading_change``, ``segments_are_g1`` with explicit tolerances,
  ``split_path``, ``split_path_where``, ``path_start_at``.
* ``const``: ``set_epsilon`` validates before mutating; new ``cell``,
  ``is_zero_rel``, ``cross_is_zero``, ``is_parallel``; ``angle_eq`` for
  direction angles. Removed ``EPSILON_MINUS``, the hash primes, ``MAX_XY``,
  ``float_eq1``/``float_eq2`` and ``DEBUG``. The ``DEBUG`` environment
  variable no longer does anything.
* ``util``: ``float_formatter`` strips zeros only after a decimal point and is
  cached; ``normalize_angle`` and ``calc_rotation`` unchanged.

Errors
------

``GeometryError(ValueError)`` for invalid public input,
``DegenerateGeometryError`` for coincident or zero-size input,
``ApproximationError`` for a failed strict approximation. Unsupported
operator operands raise ``TypeError``. No public path relies on ``assert``.
