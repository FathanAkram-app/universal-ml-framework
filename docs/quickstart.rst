Quick Start Guide
=================

.. highlight:: python

Basic Usage
-----------

Classification Problem
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from universal_ml_framework import UniversalMLPipeline

   # Create pipeline
   pipeline = UniversalMLPipeline(problem_type='classification')

   # Run complete pipeline
   pipeline.run_pipeline(
       train_path='train.csv',
       target_column='target',
       test_path='test.csv'
   )

Regression Problem
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from universal_ml_framework import UniversalMLPipeline

   # Create pipeline
   pipeline = UniversalMLPipeline(problem_type='regression')

   # Run complete pipeline
   pipeline.run_pipeline(
       train_path='train.csv',
       target_column='price',
       test_path='test.csv'
   )

Quick Setup Functions
---------------------

One-liner Classification
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from universal_ml_framework import quick_classification_pipeline

   result = quick_classification_pipeline('data.csv', 'target_column')

One-liner Regression
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from universal_ml_framework import quick_regression_pipeline

   result = quick_regression_pipeline('data.csv', 'price_column')

Generate Sample Data
--------------------

.. code-block:: python

   from universal_ml_framework import DataGenerator

   # Generate synthetic datasets
   DataGenerator.generate_customer_churn()
   DataGenerator.generate_house_prices()
   DataGenerator.generate_sales_forecasting()

   # Generate all datasets at once
   DataGenerator.generate_all_datasets()

Customization Options
---------------------

Exclude Columns
~~~~~~~~~~~~~~~

.. code-block:: python

   pipeline.run_pipeline(
       train_path='data.csv',
       target_column='target',
       exclude_columns=['id', 'timestamp', 'name']
   )

Custom Feature Types
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   pipeline.feature_types = {
       'numeric': ['age', 'income', 'score'],
       'categorical': ['city', 'category', 'type'],
       'binary': ['has_feature', 'is_active']
   }

What Happens Automatically
--------------------------

.. note::
   The framework handles the entire ML pipeline automatically:

1. **Data Loading** - Reads CSV files
2. **Feature Detection** - Identifies feature types
3. **Preprocessing** - Handles missing values, encoding, scaling
4. **Model Training** - Tests multiple algorithms
5. **Cross Validation** - Evaluates model performance
6. **Hyperparameter Tuning** - Optimizes best model
7. **Prediction** - Generates test predictions
8. **Model Saving** - Persists trained model

.. tip::
   Check the generated files after running:
   
   * ``predictions.csv`` - Test predictions
   * ``best_model.pkl`` - Trained model
   * ``model_info.json`` - Model metadata