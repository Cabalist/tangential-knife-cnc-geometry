Testing
=======

Install the development environment and run the checks::

    uv sync --group dev
    uv run prek run --all-files      # ruff, ruff-format, ty, pyrefly
    uv run pytest                    # xdist, random order
    uv run python -O -m pytest       # assertions stripped

Tests use only the standard library. One test module per kernel module;
each test is named after the behaviour it pins down, and edge cases found
in review live next to the ordinary tests of the same module. Tests at the
``EPSILON`` scale (curves a few ``1e-7`` across, arcs of radius ``1e-8``)
check that the numerics stay self-consistent and never lose geometry
silently; they say nothing about physical meaning, which starts at the
process resolution (see the resolution floor in ``geom2d.const``).
