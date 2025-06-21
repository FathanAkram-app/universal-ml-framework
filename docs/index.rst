Universal ML Framework Documentation
====================================

Welcome to Universal ML Framework - a complete, automated machine learning pipeline framework that works with any dataset.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples

Overview
--------

Universal ML Framework is designed to automate the entire machine learning workflow:

* **Auto Feature Detection** - Automatically identifies numeric, categorical, and binary features
* **Model Comparison** - Tests multiple algorithms with cross-validation
* **Hyperparameter Tuning** - Optimizes the best performing model
* **Production Ready** - Saves trained models and generates predictions

Supported Problems
------------------

* **Classification** - Binary and multi-class classification
* **Regression** - Continuous target prediction

Quick Example
-------------

.. code-block:: python

   from universal_ml_framework import UniversalMLPipeline

   # Create and run pipeline
   pipeline = UniversalMLPipeline(problem_type='classification')
   pipeline.run_pipeline(
       train_path='data.csv',
       target_column='target',
       test_path='test.csv'
   )

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`