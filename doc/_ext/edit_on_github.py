# Sphinx extension: ReadTheDocs-style "Edit on GitHub" sidebar links.
# Loosely based on https://github.com/astropy/astropy/pull/347
# License: BSD (3 clause)
"""Sphinx extension that adds "Edit on GitHub" links to the rendered pages."""

from __future__ import annotations

import os
import warnings


def get_github_url(app, view, path):
    return "https://github.com/{project}/{view}/{branch}/{path}".format(
        project=app.config.edit_on_github_project,
        view=view,
        branch=app.config.edit_on_github_branch,
        path=path,
    )


def html_page_context(app, pagename, templatename, context, doctree):
    if templatename != "page.html":
        return
    if not app.config.edit_on_github_project:
        warnings.warn("edit_on_github_project not specified", stacklevel=2)
        return
    if not doctree:
        return
    path = os.path.relpath(doctree.get("source"), app.builder.srcdir)
    context["show_on_github_url"] = get_github_url(app, "blob", path)
    context["edit_on_github_url"] = get_github_url(app, "edit", path)


def setup(app):
    app.add_config_value("edit_on_github_project", "", True)
    app.add_config_value("edit_on_github_branch", "master", True)
    app.connect("html-page-context", html_page_context)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
