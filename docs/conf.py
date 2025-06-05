# Configuration file for the Sphinx documentation builder.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))


# -- Project information -----------------------------------------------------

# NOTE: If you installed your project in editable mode, this might be stale.
#       If this is the case, reinstall it to refresh the metadata
info = metadata("xrdantic")
project_name = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}."
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL", []))
repository_url = urls["Source"]

# The full version, including alpha/beta/rc tags
release = info["Version"]

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
nitpicky = True  # Enable strict reference checking with proper ignore list
needs_sphinx = "4.0"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "srivarra",
    "github_repo": project_name,
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx_tabs.tabs",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
    "scanpydoc",
    "sphinxcontrib.autodoc_pydantic",
]


# Better autodoc_pydantic configuration for cleaner docs
autodoc_pydantic_model_show_json = True
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_config = False
autodoc_pydantic_model_show_validator_summary = True
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_field_signature_prefix = "field"
autodoc_pydantic_field_show_constraints = True
autodoc_pydantic_settings_show_json = False

autosummary_generate = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
myst_heading_anchors = 6  # create anchors for h1-h6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "numpydantic": ("https://numpydantic.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_title = project_name

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
}

pygments_style = "default"

nitpick_ignore = [
    # If building the documentation fails because of a missing link that is outside your control,
    # you can add an exception to this list.
    #     ("py:class", "igraph.Graph"),
    # Pydantic ValidationError - external dependency
    # ("py:exc", "ValidationError"),
    # # Class attributes and fields that are auto-generated or computed
    # ("py:obj", "xrdantic.models.Coordinate.check_data_field"),
    # ("py:obj", "xrdantic.models.Coordinate.all_fields"),
    # ("py:obj", "xrdantic.models.Coordinate.validate_and_sanitize_data"),
    # ("py:obj", "xrdantic.models.DataArray.check_data_field"),
    # ("py:obj", "xrdantic.models.DataArray.all_fields"),
    # ("py:obj", "xrdantic.models.DataArray.validate_and_sanitize_data"),
    # ("py:obj", "xrdantic.models.DataTree.check_tree_fields"),
    # ("py:obj", "xrdantic.models.DataTree.all_fields"),
    # ("py:obj", "xrdantic.models.Dataset.check_data_fields"),
    # ("py:obj", "xrdantic.models.Dataset.all_fields"),
    # # XrdanticSettings fields - auto-generated by Pydantic
    # ("py:obj", "xrdantic.config.XrdanticSettings.allow_inf_values"),
    # ("py:obj", "xrdantic.config.XrdanticSettings.allow_nan_values"),
    # ("py:obj", "xrdantic.config.XrdanticSettings.auto_convert_lists"),
    # ("py:obj", "xrdantic.config.XrdanticSettings.debug_mode"),
    # ("py:obj", "xrdantic.config.XrdanticSettings.detailed_error_messages"),
    # ("py:obj", "xrdantic.config.XrdanticSettings.enable_memory_optimization"),
    # ("py:obj", "xrdantic.config.XrdanticSettings.log_validation_errors"),
    # ("py:obj", "xrdantic.config.XrdanticSettings.max_cache_size"),
    # ("py:obj", "xrdantic.config.XrdanticSettings.strict_validation"),
    # ("py:obj", "xrdantic.config.XrdanticSettings.use_validation_cache"),
    # ("py:obj", "xrdantic.config.XrdanticSettings.validate_coordinates"),
    # ("py:obj", "xrdantic.config.XrdanticSettings.validate_dimensions"),
]

# Suppress image warnings for now - images will be handled properly in final build
suppress_warnings = [
    # "image.not_readable",
    # "ref.doc",  # Unknown document references
    # "ref.ref",  # Unknown reference targets
    # "ref.exc",  # Unknown exception references
    # "ref.obj",  # Unknown object references
    # "ref.class",  # Unknown class references
    # "ref.func",  # Unknown function references
    # "ref.meth",  # Unknown method references
    # "ref.attr",  # Unknown attribute references
    # "docutils",  # General docutils warnings including markup issues
    # "autosummary",  # Autosummary warnings
]
