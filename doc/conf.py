# Copyright 2019 Xanadu Quantum Technologies Inc.
# SPDX-License-Identifier: Apache-2.0
"""PennyLane-IonQ documentation build configuration."""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("_ext"))

# -- General ---------------------------------------------------------------

needs_sphinx = "8.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "edit_on_github",
]

intersphinx_mapping = {
    "pennylane": ("https://docs.pennylane.ai/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
}

autosummary_generate = True
autosummary_imported_members = False
automodapi_toctreedirnm = "code/api"
automodsumm_inherited_members = True
autodoc_member_order = "bysource"

source_suffix = ".rst"
master_doc = "index"

project = "PennyLane-IonQ"
copyright = "2019-2026, Xanadu Quantum Technologies Inc."
author = "Xanadu Inc."

add_module_names = False

import pennylane_ionq

release = pennylane_ionq.__version__
version = re.match(r"^(\d+\.\d+)", release).expand(r"\1")

language = "en"
today_fmt = "%Y-%m-%d"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
show_authors = True
pygments_style = "sphinx"
todo_include_todos = True

# -- HTML ------------------------------------------------------------------

html_favicon = "_static/favicon.ico"
html_static_path = ["_static"]

from pennylane_sphinx_theme import templates_dir

templates_path = [templates_dir()]
html_theme = "pennylane"
html_theme_options = {
    "navbar_name": "PennyLane-IonQ",
    "toc_overview": True,
    "navbar_active_link": 3,
    "google_analytics_tracking_id": "G-C480Z9JL0D",
}

edit_on_github_project = "PennyLaneAI/pennylane-ionq"
edit_on_github_branch = "master/doc"

inheritance_node_attrs = dict(color="lightskyblue1", style="filled")
