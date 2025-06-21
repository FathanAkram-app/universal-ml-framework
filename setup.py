# SETUP SCRIPT
# Installation and setup for Universal ML Framework

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="universal-ml-framework",
    version="1.2.0",
    author="Fathan Akram",
    author_email="fathan.a.dev@gmail.com",
    description="A complete machine learning pipeline framework for any dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FathanAkram-App/universal-ml-framework",
    packages=find_packages(include=['universal_ml_framework', 'universal_ml_framework.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "jupyter>=1.0",
        ],
        "viz": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "uml-pipeline=universal_ml_framework.core.pipeline:main",
        ],
    },
    keywords="machine learning, pipeline, automation, classification, regression, data science",
    project_urls={
        "Bug Reports": "https://github.com/FathanAkram-App/universal-ml-framework/issues",
        "Source": "https://github.com/FathanAkram-App/universal-ml-framework",
        "Documentation": "https://universal-ml-framework.readthedocs.io/",
    },
)