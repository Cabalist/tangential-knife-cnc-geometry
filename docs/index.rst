======
geom2d
======

A 2D geometry kernel for turning vector artwork into cutting toolpaths:
points, line segments, circular arcs and cubic Bézier curves, the biarc
approximation that turns curves into tangent-continuous arcs, and the path
helpers a toolpath generator needs. Pure standard library, Python 3.14.

Out of scope by design: SVG parsing, affine transforms and elliptical arcs.
An external parser applies transforms and converts elliptical arcs before
geometry reaches this library.

.. toctree::
    :maxdepth: 2

    api


Conventions
-----------

Orientation
    Angles are in radians, measured counter-clockwise from the positive x
    axis. A positive cross product, ``P.winding`` or ``heading_change``
    means a counter-clockwise (left) turn. ``Arc.angle`` is the signed sweep
    from ``p1`` to ``p2`` about the center, counter-clockwise positive; a
    clockwise arc has a negative angle. ``offset(+d)`` moves any segment to
    the left of its direction of travel.

Tolerance
    ``EPSILON`` is a distance, applied as an absolute difference at every
    magnitude. Coordinates, lengths and radii compare with ``float_eq`` or
    ``is_zero``, and the query that asks whether a point is on a circle uses
    the same test as the constructor that validated the arc. Direction
    angles compare with ``angle_eq``, which treats directions a whole turn
    apart (``+pi`` and ``-pi``, ``0`` and ``tau``) as one direction; cross
    products and discriminants compare relative to the natural scale of the
    problem so results do not depend on the size of the geometry. Explicit
    tolerances (``segments_are_g1(..., angle_tolerance=1e-12)``) are honoured
    however small, because ``heading_change`` applies none of its own. Call
    ``set_epsilon`` once at startup, before creating geometry: it changes the
    grid behind ``==`` and ``hash``.

Coordinate envelope
    A double rounds a coordinate of magnitude ``|x|`` to about ``1e-16 *
    |x|``, so ``EPSILON`` is meaningful for ``|x|`` below about ``1e7`` at
    the default. A direction derived from a feature of size ``s`` carries
    that rounding divided by ``s``, so tangent directions agree within
    ``EPSILON`` only for features larger than about ``1e-8 * |x|`` (a radius
    of ``0.01`` at ``|x| = 1e6``). The biarc construction runs relative to
    the curve's start point, so its joints do not pay for the curve's
    position; only the stored result does.

Resolution floor
    ``EPSILON`` is a numerical floor, not a physical one. It keeps hashing,
    coincidence, the arc invariant and the approximation checks consistent
    with each other; no cutting process produces anything near it. Set
    ``EPSILON`` several orders of magnitude below the process resolution
    (a knife that resolves ``0.01`` mm is a million times coarser than the
    default ``1e-8``) and pass tolerances at the process resolution. Above
    that physical floor, the contracts on this page describe the geometry
    a machine will see. Below it, the library promises self-consistency
    only: no crashes, no silently dropped geometry, connected output, and
    an explicit error where a request cannot be met at all. The edge cases
    handled near ``EPSILON`` (pieces too short to be segments, arcs of
    radius ``1e-8``) are there so that micro-features never break a path,
    not because they can be cut; drop or simplify features below your own
    floor before approximating them.

Intersections
    Methods named ``intersect_<other>`` return every meeting point as a list,
    in order along the receiver, with keyword flags naming the operand to
    restrict to its segment: ``Arc.intersect_line(line, on_arc=, on_line=)``,
    ``Arc.intersect_arc(other, on_arc=)``, ``CubicBezier.intersect_line(line,
    on_line=)`` (a curve is always bounded). Two lines meet at most once, so
    ``Line.intersection`` returns a single point or None and ``segment=True``
    restricts both lines; ``intersection_mu``, ``intersects`` and ``crosses``
    belong to that family. A point counts as an intersection when it is
    within ``EPSILON`` of both shapes, so tangencies are reported once.
    Where two shapes share a stretch the ends of the shared portion are
    reported: for collinear segments the start of the overlap along the
    receiver; for a curve whose control points lie along a line (it may
    double back), its extent along the line, in order along the line and
    clipped to the segment with ``on_line``; for two arcs on one circle,
    the endpoints of either arc that lie on the other arc, within
    ``EPSILON`` of its circle and inside its sweep, a full circle
    contributing no ends of its own. Two lines are collinear when their
    directions agree within ``EPSILON`` and one lies within ``EPSILON`` of
    the other; a short segment sitting in a line's tolerance band at a
    visible angle is a crossing, not an overlap. Segment crossings are
    decided by the endpoints: two long segments whose directions differ by
    less than ``EPSILON`` still cross when each one's endpoints lie on
    opposite sides of the other. Only infinite lines treat directions
    within ``EPSILON`` as parallel.

Identity versus coincidence
    ``==`` and ``hash`` on every geometry class are identity on the
    ``EPSILON`` grid, so equal objects always hash equal and can be dictionary
    keys. ``P.almost_equal`` is the test for geometric coincidence, and it is
    what ``path_is_closed`` and ``segments_are_g1`` use.

Errors
    Invalid geometric input raises ``GeometryError``, a ``ValueError``;
    coincident or zero-size input raises its subclass
    ``DegenerateGeometryError``; a biarc approximation that cannot meet its
    tolerance raises ``ApproximationError`` unless ``strict=False`` asks for
    the best effort. A non-finite coordinate is refused when the point is
    built. Ordinary Python exceptions keep their ordinary meaning: a bad
    segment index is an ``IndexError``, an unsupported operator operand a
    ``TypeError``, ``p / 0`` a ``ZeroDivisionError``, and ``set_epsilon``
    with a value outside its range a ``ValueError``. Degenerate segments have
    documented return values (zero tangents, the start point, the segment
    unchanged) instead of arithmetic errors, and nothing in the library
    depends on ``assert``, so ``python -O`` changes nothing.

Approximation
    ``biarc_approximation`` checks its result by sampling (16 curve samples
    per piece against the arcs, 7 samples per arc against the curve), so the
    tolerance is an estimate that a peak between samples can exceed by a
    small amount; ``hausdorff_distance`` gives a finer estimate of a result,
    and neither is a proof. The tangent contract at the joints holds within
    ``EPSILON`` whether or not the tolerance was met, subject to the two
    limits above: the coordinate envelope's feature-size rule, and features
    of the curve smaller than ``EPSILON`` (cusps, and pieces too short to be
    segments), which become corners. A non-degenerate curve (control points
    not all within ``EPSILON`` of each other) never yields an empty result:
    a piece whose ends coincide is halved regardless of ``max_depth``, and
    if nothing at least ``EPSILON`` long can span the curve,
    ``ApproximationError`` is raised in either mode. A sweep limit that
    could only be met with arcs shorter than ``EPSILON`` raises
    ``GeometryError`` rather than dropping geometry.

Data model
    Every class is a frozen, slotted dataclass with positional fields:
    ``P(x, y)``, ``Line(p1, p2)``, ``Arc(p1, p2, radius, angle, center)``,
    ``CubicBezier(p1, c1, c2, p2)``, ``Box(p1, p2)``. Constructors take ``P``;
    factory classmethods (``P.of``, ``Line.from_polar``, ``Arc.from_sweep``,
    ``CubicBezier.from_quadratic``, ``Box.from_points``) accept any point-like
    input. ``dataclasses.replace``, ``match``, ``copy`` and ``pickle`` all
    work; an ``Arc`` re-validates on ``replace``.


Origins
-------

The library descends from Claude Zervas's utl-geom2d, whose computational
geometry drew on the work of Joseph O'Rourke, Paul Bourke, Eric Haines,
W. Randolph Franklin, Pomax's Bézier primer and Adrian Colomitchi's note on
cubic inflection points. Version 1.0 rebuilt it as the kernel described
above; ``CHANGELOG.rst`` records what changed.
