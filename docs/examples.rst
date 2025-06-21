Examples
========

.. highlight:: python

Customer Churn Classification
-----------------------------

.. code-block:: python

   from universal_ml_framework import UniversalMLPipeline, DataGenerator

   # Generate sample data
   DataGenerator.generate_customer_churn()

   # Run classification pipeline
   pipeline = UniversalMLPipeline(problem_type='classification')
   pipeline.run_pipeline(
       train_path='data/customer_train.csv',
       target_column='Churn',
       test_path='data/customer_test.csv'
   )

   print(f"Best model: {pipeline.best_model_name}")
   print(f"Best score: {pipeline.best_score:.4f}")

House Price Prediction
-----------------------

.. code-block:: python

   from universal_ml_framework import UniversalMLPipeline, DataGenerator

   # Generate sample data
   DataGenerator.generate_house_prices()

   # Run regression pipeline
   pipeline = UniversalMLPipeline(problem_type='regression')
   pipeline.run_pipeline(
       train_path='data/house_train.csv',
       target_column='SalePrice',
       test_path='data/house_test.csv'
   )

   print(f"Best model: {pipeline.best_model_name}")
   print(f"Best MSE: {pipeline.best_score:.2f}")

Sales Forecasting
-----------------

.. code-block:: python

   from universal_ml_framework import UniversalMLPipeline, DataGenerator

   # Generate sample data
   DataGenerator.generate_sales_forecasting()

   # Run regression pipeline
   pipeline = UniversalMLPipeline(problem_type='regression')
   pipeline.run_pipeline(
       train_path='data/sales_train.csv',
       target_column='Sales',
       test_path='data/sales_test.csv'
   )

Using Predefined Configurations
-------------------------------

.. code-block:: python

   from universal_ml_framework import run_pipeline_with_config, list_available_configs

   # List available configurations
   list_available_configs()

   # Run with predefined config
   pipeline = run_pipeline_with_config('customer_churn')

Custom Feature Engineering
--------------------------

.. code-block:: python

   from universal_ml_framework import UniversalMLPipeline

   pipeline = UniversalMLPipeline(problem_type='classification')

   # Manually specify feature types
   pipeline.feature_types = {
       'numeric': ['age', 'income', 'tenure'],
       'categorical': ['city', 'job_type', 'education'],
       'binary': ['has_phone', 'has_internet', 'is_senior']
   }

   pipeline.run_pipeline(
       train_path='data.csv',
       target_column='target',
       custom_features=pipeline.feature_types['numeric'] + 
                      pipeline.feature_types['categorical'] + 
                      pipeline.feature_types['binary']
   )

Batch Processing Multiple Datasets
----------------------------------

.. code-block:: python

   from universal_ml_framework import UniversalMLPipeline, DataGenerator

   # Generate all sample datasets
   DataGenerator.generate_all_datasets()

   datasets = [
       ('data/customer_train.csv', 'Churn', 'classification'),
       ('data/house_train.csv', 'SalePrice', 'regression'),
       ('data/sales_train.csv', 'Sales', 'regression')
   ]

   results = {}
   for train_path, target, problem_type in datasets:
       pipeline = UniversalMLPipeline(problem_type=problem_type)
       pipeline.run_pipeline(train_path, target)
       results[train_path] = {
           'best_model': pipeline.best_model_name,
           'best_score': getattr(pipeline, 'best_score', 'N/A')
       }

   for dataset, result in results.items():
       print(f"{dataset}: {result['best_model']} - {result['best_score']}")

Working with Your Own Data
--------------------------

.. code-block:: python

   from universal_ml_framework import UniversalMLPipeline

   # For your own CSV file
   pipeline = UniversalMLPipeline(problem_type='classification')
   pipeline.run_pipeline(
       train_path='your_data.csv',
       target_column='your_target_column',
       test_path='your_test_data.csv',
       exclude_columns=['id', 'timestamp', 'irrelevant_column']
   )

   # Check results
   print(f"Best model: {pipeline.best_model_name}")
   print(f"Cross-validation score: {pipeline.cv_results[pipeline.best_model_name]['mean']:.4f}")
   print(f"Feature types detected: {pipeline.feature_types}")

Loading Saved Models
--------------------

.. code-block:: python

   import joblib
   import json

   # Load saved model
   model = joblib.load('best_model.pkl')

   # Load model metadata
   with open('model_info.json', 'r') as f:
       model_info = json.load(f)

   print(f"Model type: {model_info['best_model']}")
   print(f"Problem type: {model_info['problem_type']}")
   print(f"CV Score: {model_info['cv_score']:.4f}")

   # Make predictions on new data
   # predictions = model.predict(new_data)

.. tip::
   All examples generate output files that you can examine:
   
   * ``predictions.csv`` - Model predictions
   * ``best_model.pkl`` - Trained model
   * ``model_info.json`` - Model metadata and performance