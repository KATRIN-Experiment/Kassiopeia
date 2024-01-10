# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os

import sys

os.path.insert(0, os.path.abspath('Gon-na/Kassiopeia'))

# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Kassiopeia'
copyright = '2016-2023, The Kassiopeia developers.'
author = 'The Kassiopeia developers'

# The full version, including alpha/beta/rc tags
release = '4.0.0'


# -- General configuration ---------------------------------------------------

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.githubpages',
    'sphinx.ext.extlinks',
    'sphinx.ext.mathjax',
    'sphinx.ext.graphviz',
    'sphinx_rtd_theme',
    'sphinx_design'
]

extlinks = {
    'gh-issue':     ('https://github.com/KATRIN-Experiment/Kassiopeia/issues/%s',
                     'Issue %s'),
    'gh-pull':      ('https://github.com/KATRIN-Experiment/Kassiopeia/pull/%s',
                     'Pull request %s'),
    'gh-code':      ('https://github.com/KATRIN-Experiment/Kassiopeia/blob/master/%s',
                     'GitHub: %s'),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

highlight_language = ''


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# See https://sphinxawesome.xyz/how-to/load/
# html_theme = 'sphinxawesome_theme'

# Theme options. See https://sphinxawesome.xyz/how-to/options/
# html_theme_options = {
#     "nav_include_hidden": True,
#     "show_nav": True,
#     "show_breadcrumbs": True,
#     "breadcrumbs_separator": " • ",
#     "show_prev_next": True,
#     "show_scrolltop": True,
#     "extra_header_links": {
#         "GitHub": "https://github.com/KATRIN-Experiment/Kassiopeia",
#         "DockerHub": "https://hub.docker.com/r/katrinexperiment/kassiopeia",
#     },
#
#     "html_awesome_headerlinks": True,
#     "html_awesome_code_headers": False,
# }

# html_awesome_docsearch = False
# docsearch_config = {}

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': False,
    'display_version': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#343131',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static', '_images']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom.css',
]

html_logo = '_images/KassiopeiaLogo_1_cropped_bb.png'
html_permalinks_icon = "<span>#</span>"

# "Edit on GitHub" link
if "GITHUB_REPOSITORY_OWNER" in os.environ:
    html_context = {
        "display_github": True,
        "github_user": os.environ["GITHUB_REPOSITORY_OWNER"],
        "github_repo": os.environ["GITHUB_REPOSITORY"].split("/")[1],
        "github_version": os.environ["GITHUB_REF_NAME"],
        "conf_py_path": "/Documentation/gh-pages/source/",
    }
