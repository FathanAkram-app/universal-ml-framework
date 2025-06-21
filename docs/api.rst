API Reference
=============

Core Classes
------------

UniversalMLPipeline
~~~~~~~~~~~~~~~~~~~

.. autoclass:: universal_ml_framework.UniversalMLPipeline
   :members:
   :undoc-members:
   :show-inheritance:

Helper Functions
----------------

Quick Setup Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: universal_ml_framework.quick_classification_pipeline

.. autofunction:: universal_ml_framework.quick_regression_pipeline

.. autofunction:: universal_ml_framework.run_pipeline_with_config

.. autofunction:: universal_ml_framework.list_available_configs

Data Generation
---------------

DataGenerator
~~~~~~~~~~~~~

.. autoclass:: universal_ml_framework.DataGenerator
   :members:
   :undoc-members:
   :show-inheritance:

Method Details
--------------

Main Pipeline Methods
~~~~~~~~~~~~~~~~~~~~~

.. py:method:: UniversalMLPipeline.run_pipeline(train_path, target_column, test_path=None, problem_type='classification', exclude_columns=None, custom_features=None)

   Main method to execute the complete ML pipeline.

   :param str train_path: Path to training CSV file
   :param str target_column: Name of target column
   :param str test_path: Path to test CSV file (optional)
   :param str problem_type: 'classification' or 'regression'
   :param list exclude_columns: Columns to exclude from features (optional)
   :param list custom_features: Custom feature list (optional)

.. py:method:: UniversalMLPipeline.load_data(train_path, test_path=None, target_column=None)

   Load training and test data from CSV files.

.. py:method:: UniversalMLPipeline.auto_detect_features(df, exclude_columns=None)

   Automatically detect feature types (numeric, categorical, binary).

.. py:method:: UniversalMLPipeline.cross_validate_models()

   Compare multiple models using cross-validation.

.. py:method:: UniversalMLPipeline.hyperparameter_tuning()

   Optimize hyperparameters for the best model.

.. py:method:: UniversalMLPipeline.make_predictions(save_predictions=True)

   Generate predictions on test data.

.. py:method:: UniversalMLPipeline.save_model(filename='best_model.pkl')

   Save trained model and metadata to files.