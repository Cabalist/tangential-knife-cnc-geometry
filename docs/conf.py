"""Sphinx configuration for the geom2d documentation."""

project = "tangential-knife-cnc-geometry"
copyright = "2026, Ryan Jarvis"  # noqa: A001
author = "Ryan Jarvis"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_type_aliases = {
    "PointLike": "geom2d.point.PointLike",
    "Path": "geom2d.segment.Path",
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_theme = "furo"
html_title = "tangential-knife-cnc-geometry"
