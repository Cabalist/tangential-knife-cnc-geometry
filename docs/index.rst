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
    ``EPSILON`` is a distance. Coordinates, lengths and radii compare with
    ``float_eq`` or ``is_zero``; direction angles compare with ``angle_eq``,
    which treats ``+pi`` and ``-pi`` as one direction; cross products and
    discriminants compare relative to the natural scale of the problem so
    results do not depend on the size of the geometry. Call ``set_epsilon``
    once at startup, before creating geometry: it changes the grid behind
    ``==`` and ``hash``.

Identity versus coincidence
    ``==`` and ``hash`` on every geometry class are identity on the
    ``EPSILON`` grid, so equal objects always hash equal and can be dictionary
    keys. ``P.almost_equal`` is the test for geometric coincidence, and it is
    what ``path_is_closed`` and ``segments_are_g1`` use.

Errors
    Invalid public input raises ``GeometryError``, a ``ValueError``;
    coincident or zero-size input raises its subclass
    ``DegenerateGeometryError``; an approximation that cannot meet its
    tolerance raises ``ApproximationError`` when asked to be strict. Degenerate
    segments have documented return values (zero tangents, the start point,
    the segment unchanged) instead of arithmetic errors, and nothing in the
    library depends on ``assert``, so ``python -O`` changes nothing.

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
