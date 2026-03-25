import os
import sys
from datetime import datetime

# Add project root so autodoc can import cutile_basic
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "cutile-basic"
copyright = f"{datetime.now().year}, NVIDIA Corporation"
author = "NVIDIA Corporation"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "nvidia_sphinx_theme"

html_title = "cutile-basic"

html_theme_options = {
    "icon_links": [],
    "external_links": [
        {"name": "PDF", "url": "_static/cutile-basic.pdf"},
    ],
    "navigation_depth": 4,
    "show_toc_level": 2,
    "navbar_start": ["navbar-logo"],
    "navbar_end": [
        "theme-switcher",
        "navbar-icon-links",
    ],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "collapse_navigation": False,
}

html_static_path = ["_static"]

# -- Options for LaTeX / PDF output ------------------------------------------

latex_logo = os.path.join(os.path.abspath(".."), "graphic.jpg")

latex_engine = "xelatex"
latex_use_xindy = False

latex_documents = [
    ("index", "cutile-basic.tex", "cutile-basic Documentation",
     author, "manual"),
]

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "11pt",
    "extraclassoptions": "openany",
}

latex_domain_indices = False

# -- Extension configuration -------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autodoc_type_hints = "description"

autodoc_mock_imports = [
    "cuda",
    "cuda.tile",
    "cuda.tile._bytecode",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True

pygments_style = "default"
pygments_dark_style = "monokai"


def setup(app):
    if os.path.exists("_static/custom.css"):
        app.add_css_file("custom.css")
