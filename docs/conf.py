project = 'Universal ML Framework'
copyright = '2024, Fathan Akram'
author = 'Fathan Akram'
release = '1.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}