"""Sphinx project that builds the molisanax API reference and emits it as a
MyST-MD AST (``*.myst.json``) consumable by mystmd's TOC ``pattern:`` entry.

Build with::

    sphinx-build -b myst docs/_sphinx docs/_sphinx/_build/myst

The resulting JSON files are ingested by ``myst build --html`` so the API
section renders inside the same site (and the same theme) as the rest of
the documentation.
"""

from __future__ import annotations

project = "molisanax"
author = "Vadim Bertrand"

extensions = [
    "sphinx_ext_mystmd",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

# Google-style docstrings; keep numpy off so we don't double-parse.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# sphinx-ext-mystmd (as of 2026-05) does not implement a visitor for
# docutils' ``abbreviation`` nodes, which Sphinx's Python domain emits to
# render the ``/`` (PEP 570) and ``*`` (PEP 3102) parameter separators in
# function signatures. Patch a passthrough so the AST build doesn't crash.
def setup(app):  # noqa: D401 - sphinx hook
    from sphinx_ext_mystmd.transform import MySTNodeVisitor

    if not hasattr(MySTNodeVisitor, "visit_abbreviation"):
        def visit_abbreviation(self, node):
            return self.enter_myst_node(
                {"type": "span", "class": ["abbreviation"], "children": []}, node
            )
        MySTNodeVisitor.visit_abbreviation = visit_abbreviation
