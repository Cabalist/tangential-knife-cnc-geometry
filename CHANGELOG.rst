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
  including at and near a half turn, and accepts any non-zero sweep (a
  shallow arc at a large radius is valid). All angular queries (``mu``,
  ``point_on_arc``, ``point_inside``, ``subdivide_at_point``, ``height``,
  ``normal_projection_point``, ``bounding_box``) are correct for any sweep,
  including more than half a turn and clockwise arcs, and every "on the
  circle" test (``point_on_arc``, ``intersect_line`` tangency,
  ``intersect_circles`` tangency) uses ``EPSILON`` as the absolute distance
  the constructor used. The tangent at each end is derived from that end's
  stored point. ``calc_center`` places the center ``radius * cos(angle/2)``
  from the chord, so shallow arcs at any orientation construct. ``offset``
  builds its endpoints on the new circle from the sweep. Intersections are
  ordered along the receiving arc; two arcs on one circle report the ends
  of their shared portion with ``on_arc``. ``subdivide_equal`` and
  ``split_max_sweep`` refuse to make pieces shorter than ``EPSILON``.
  ``from_sweep(..., tolerance=)`` repairs arcs whose radius and sweep agree
  with their endpoints only within a job's precision (the sweep is adjusted
  to reach ``p2``, a chord slightly longer than the diameter grows the
  radius), so parser-built arcs become exactly consistent; data consistent
  within ``EPSILON`` keeps its sweep verbatim. New: ``subdivide_equal``,
  ``split_max_sweep``, ``is_clockwise``, ``direction``, tangent vectors,
  ``tangent_at``. ``extend`` extends. ``offset(+d)`` is left of travel,
  like ``Line.offset``. ``to_svg_path`` writes a full circle as two half
  turns.
* ``Line``: cross-product and parameter tolerances are distances, so
  ``point_on_line``, ``is_parallel``, ``which_side`` and intersections do not
  depend on the segment's length; collinear overlapping segments intersect;
  segment intersections and ``crosses`` are decided by the endpoints, so
  long, nearly parallel segments that cross are reported (only infinite
  lines treat directions within ``EPSILON`` as parallel); ``crosses`` is
  strict on both segments and symmetric; ``same_side`` is fixed; ``shift``
  and ``extend`` kept. One ``segment`` flag on intersections.
* ``CubicBezier``: ``from_quadratic`` (exact); tangents fall back through
  coincident control points; ``inflections`` is scale-independent;
  ``find_extrema`` solves each axis on its own with a stable quadratic
  formula, so a long thin curve keeps its thin extent and ``bounding_box``
  always contains the curve; ``intersect_line(line, *, on_line=False)``
  (renamed from ``line_intersection``) finds roots by bisection between the
  polynomial's critical points, so every degree is located to machine
  precision, tangencies are reported once, and ``on_line`` restricts to the
  line segment; ``length`` is bounded and rejects non-positive tolerances.
* ``biarc_approximation(tolerance=0.001, *, max_depth=8, max_arc_angle=None,
  strict=True)``: computed relative to the curve's start point, so joints
  stay tangent-continuous within ``EPSILON`` anywhere in the coordinate
  envelope; sampled two-sided distance check (documented as an estimate);
  straight pieces become lines only when their tangents follow the chord,
  so tangent continuity holds at every joint and the only corners in the
  output are genuine sub-``EPSILON`` cusps of the source curve.
  ``max_arc_angle`` splits arcs to a maximum sweep. ``strict`` is the
  default: a piece that cannot meet the tolerance within ``max_depth``
  raises ``ApproximationError``; ``strict=False`` returns the best effort.
  Tolerances below ``EPSILON`` are refused. Pieces shorter than ``EPSILON``
  are absorbed by their neighbours (at either end of the chain) so the
  output stays exactly connected with nothing degenerate; a piece whose
  ends coincide is halved regardless of ``max_depth``; a non-degenerate
  curve never yields ``[]`` (``ApproximationError`` if nothing at least
  ``EPSILON`` long can span it); a ``max_arc_angle`` that would need arcs
  shorter than ``EPSILON`` raises ``GeometryError``. Two arcs of a biarc
  are merged only when the merged arc is valid. ``is_degenerate`` is the
  control points' extent, so a folded hairpin without extent is degenerate.
  ``hausdorff_distance`` takes any sequence of segments and rejects an
  empty one or a sample count below 1.
* ``segment``: ``Segment`` protocol, ``Path``, ``path_reversed``,
  ``path_length``, ``path_bounding_box``, ``path_is_closed(path,
  tolerance=)``, ``polyline_to_path``, ``path_to_polyline``,
  ``nearest_vertex`` (every vertex, including an open path's final point),
  ``heading_change`` (no tolerance of its own), ``segments_are_g1`` with
  explicit tolerances that are honoured however small, ``split_path``,
  ``split_path_where``, ``path_start_at``. The helpers that decide whether
  a path is closed forward a ``tolerance=`` keyword to ``path_is_closed``,
  so a job's precision is passed explicitly instead of changing the global
  floor. The helpers that return segments are generic and keep the
  concrete segment type of their input.
* ``const``: ``float_eq`` is an absolute comparison at every magnitude;
  ``angle_eq`` treats directions a whole turn apart as equal; ``set_epsilon``
  validates (``MIN_EPSILON`` = 1e-15 to below 1) and computes every derived
  constant before assigning any; ``cross_is_zero`` and ``is_parallel`` treat
  an exactly zero cross product as parallel at any scale, so coincident
  points are collinear. New ``cell``, ``is_zero_rel``, ``cross_is_zero``,
  ``is_parallel``, ``MIN_EPSILON``. Removed ``EPSILON_MINUS``, the hash
  primes, ``MAX_XY``, ``float_eq1``/``float_eq2`` and ``DEBUG``. The
  ``DEBUG`` environment variable no longer does anything. ``angle_eq``
  reduces with a signed remainder, so it is exact near zero and symmetric
  at any tolerance. The documentation distinguishes the numerical floor
  (``EPSILON``, where the library promises self-consistency only) from the
  physical floor (the process resolution, where tolerances belong).
* ``P`` refuses infinite and NaN coordinates at construction.
* ``util``: ``float_formatter`` strips zeros only after a decimal point and is
  cached; ``calc_rotation`` returns the exact signed remainder in
  ``(-pi, pi]`` with no tolerance applied; ``normalize_angle`` unchanged.
* ``CubicBezier.intersect_line``: a curve whose control points lie along
  the line (straight or doubling back) reports the ends of its extent along
  the line, clipped to the segment with ``on_line``.
* ``Arc.intersect_arc`` on one circle reports only endpoints that lie on
  both arcs (within ``EPSILON`` of the circle and inside the sweep), and a
  full circle contributes no ends. ``Line`` collinearity requires directions
  within ``EPSILON`` as well as position, so a short segment inside a line's
  tolerance band at a visible angle is a crossing. ``normalize_angle`` stays
  inside its half-open interval at the boundaries.

Errors
------

``GeometryError(ValueError)`` for invalid geometric input (including
non-finite coordinates), ``DegenerateGeometryError`` for coincident or
zero-size input, ``ApproximationError`` for a biarc approximation that could
not meet its tolerance. Ordinary Python exceptions keep their meaning: a bad
segment index is an ``IndexError``, unsupported operator operands a
``TypeError``, ``p / 0`` a ``ZeroDivisionError``, an out-of-range
``set_epsilon`` a ``ValueError``. No public path relies on ``assert``.

Tooling
-------

Ruff enforces ``FBT`` (keyword-only booleans), ``N``, ``A``, ``ERA``,
``S101`` (no ``assert`` in library code) and ``C90`` in addition to the base
rule set; ty and pyrefly run with no suppressions beyond
``unnecessary-type-conversion`` for ``P.of``. The checks are defined once,
as local hooks in ``prek.toml`` that run the project's own ruff, ty and
pyrefly; ``prek install`` runs them on every commit and CI runs the same
file (plus pytest) on pushes to ``main``, pull requests and release tags
with ``uv sync --locked``. The publish workflow runs CI as a gate before
building a release.
