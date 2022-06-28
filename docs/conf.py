"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import subprocess
import sys

# -- Path setup --------------------------------------------------------------


sys.path.insert(0, os.path.abspath('../src/'))


# -- Project information -----------------------------------------------------

project = 'rushd'
copyright = '2022, Christopher Johnstone, Kasey Love, Conrad Oakes'
author = 'Christopher Johnstone, Kasey Love, Conrad Oakes'

# The full version, including alpha/beta/rc tags
release = '0.2.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Documentation options ---
napoleon_use_param = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'collapse_navigation': False,
    'prev_next_buttons_location': None,
    'style_nav_header_background': '#2c6854',
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Setup lower-left menu. Modified from:
# https://tech.michaelaltfield.net/2020/07/23/sphinx-rtd-github-pages-2/

try:
    html_context
except NameError:
    html_context = {}
html_context['display_lower_left'] = True

if 'REPO_NAME' in os.environ:
    REPO_NAME = os.environ['REPO_NAME']
else:
    REPO_NAME = ''

current_language = 'en'

# tell the theme which language to we're currently building
html_context['current_language'] = current_language

# SET CURRENT_VERSION
if 'current_version' in os.environ:
    # get the current_version env var set by buildDocs.sh
    current_version = os.environ['current_version']
else:
    # the user is probably doing `make html`
    # set this build's current version by looking at the branch
    current_version = 'main'

# tell the theme which version we're currently on ('current_version' affects
# the lower-left rtd menu and 'version' affects the logo-area version)
html_context['current_version'] = current_version
html_context['version'] = current_version

# Set language links
html_context['languages'] = [('en', '/' + REPO_NAME + '/en/' + current_version + '/')]

# Set links to other branches
html_context['versions'] = []

# versions = [branch.name for branch in repo.branches]


versions = []
version_run = subprocess.run(
    ['git', 'for-each-ref', '--format=%(refname:lstrip=-1)', 'refs/remotes/origin', 'refs/tags'],
    capture_output=True,
)
versions_to_skip = {'gh-pages'}
if version_run.returncode == 0:
    for branch in version_run.stdout.decode('utf-8').split('\n'):
        if branch not in versions_to_skip:
            versions.append(branch)
else:
    versions.append('latest')

for version in versions:
    html_context['versions'].append(
        (version, '/' + REPO_NAME + '/' + current_language + '/' + version + '/')
    )
