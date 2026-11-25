"""Sphinx configuration for the Time Series Machine Learning Archive website."""

import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from python_utilities.build_website_content import build_website_content  # noqa: E402

# Generate dataset pages, benchmark tables, home stats, and _static/data/catalogue.json
_ARCHIVE_SUMMARY = build_website_content(REPOSITORY_ROOT, DOCS_DIR)

# -- Project information -----------------------------------------------------

project = "TSML Archive"
copyright = "TSML Archive Maintainers (BSD-3-Clause)"
author = "TSML Archive Maintainers"
html_title = "TSML Archive"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

master_doc = "index"
language = "en"

exclude_patterns = [
    "_build",
    "_generated",
    "descriptions",
    ".ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
]

myst_enable_extensions = [
    "colon_fence",
    "attrs_block",
    "attrs_inline",
    "substitution",
]
myst_heading_anchors = 3
myst_substitutions = {
    key: f"{value:,}" if isinstance(value, int) else str(value)
    for key, value in _ARCHIVE_SUMMARY.items()
}

copybutton_exclude = ".linenos, .gp, .go"

# -- Options for HTML output (PyData Sphinx Theme) ---------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = ["js/catalogue.js"]
html_favicon = "_static/images/tsml-favicon.svg"
html_show_sourcelink = False

# Disable left primary sidebar globally — navigation lives cleanly in top header bar
html_sidebars = {
    "**": [],
}

html_theme_options = {
    "logo": {
        "image_light": "_static/images/tsml-logo-light.svg",
        "image_dark": "_static/images/tsml-logo-dark.svg",
        "alt_text": "TSML Archive",
    },
    "header_links_before_dropdown": 5,
    "navbar_align": "left",
    "show_prev_next": False,
    "footer_start": ["copyright"],
    "footer_end": [],
    "secondary_sidebar_items": ["page-toc"],
    "icon_links": [
        {
            "name": "TSML on Zenodo",
            "url": "https://zenodo.org/communities/tsml/records",
            "icon": "fa-solid fa-database",
        },
        {
            "name": "GitHub Repository",
            "url": "https://github.com/time-series-machine-learning/tsml-repo",
            "icon": "fa-brands fa-github",
        },
    ],
}
