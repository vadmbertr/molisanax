"""Minimal Sphinx project whose only job is to invoke sphinx-autodoc2 and write
MyST-flavored API stubs into ``docs/_api/``. The stubs are then consumed by
the top-level mystmd build (``myst build --html``); this Sphinx project never
produces user-facing HTML.

Run with: ``sphinx-build -W -b dummy docs/_sphinx docs/_sphinx/_build``.
"""

from __future__ import annotations

project = "molisanax"
author = "Vadim Bertrand"

extensions = ["autodoc2"]

autodoc2_packages = [
    {
        "path": "../../src/molisanax",
        "module": "molisanax",
    },
]
autodoc2_render_plugin = "myst"
autodoc2_output_dir = "../_api"
autodoc2_docstring_parser_regexes = [
    (r".*", "myst"),
]
autodoc2_hidden_objects = {"private", "dunder"}
autodoc2_skip_module_regexes = [r".*\._[^.]+$"]
autodoc2_index_template = None

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
suppress_warnings = ["autodoc2.*"]
