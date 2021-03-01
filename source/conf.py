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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Kassiopeia'
copyright = '2021, The Kassiopeia developers.'
author = 'The Kassiopeia developers'

# The full version, including alpha/beta/rc tags
release = '3.7'


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
]

extlinks = {'gh-issue': ('https://github.com/KATRIN-Experiment/Kassiopeia/issues/%s',
                         'issue '),
            'gh-file':  ('https://github.com/KATRIN-Experiment/Kassiopeia/tree/master/%s',
                         'file'),
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
#
html_theme = 'sphinxawesome_theme'
html_theme_options = {
    "nav_include_hidden": True,
    "show_nav": True,
    "show_breadcrumbs": True,
    "breadcrumbs_separator": " • ",
    "show_prev_next": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static', '_images']

html_logo = '_images/KassiopeiaLogo_1_cropped_bb.png'
