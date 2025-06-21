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

source_suffix = {
    '.rst': None,
    '.md': None,
}