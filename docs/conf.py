project = 'Universal ML Framework'
copyright = '2024, Fathan Akram'
author = 'Fathan Akram'
release = '1.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode'
]

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

autodoc_mock_imports = ['pandas', 'numpy', 'sklearn', 'joblib']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'style_nav_header_background': '#1f77b4',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 3,
    'includehidden': True,
    'titles_only': False
}

html_title = 'Universal ML Framework'
html_short_title = 'Universal ML'
html_logo = None
html_favicon = None

source_suffix = {
    '.rst': None,
    '.md': None,
}