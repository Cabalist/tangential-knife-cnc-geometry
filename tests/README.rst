Testing
=======

Install the development environment and run the checks::

    uv sync --group dev
    uv run prek run --all-files      # ruff, ruff-format, ty, pyrefly
    uv run pytest                    # xdist, random order
    uv run python -O -m pytest       # assertions stripped

Tests use only the standard library. One test module per kernel module;
regression tests are named after the finding they cover (see
``docs/rebuild-plan.md``).
