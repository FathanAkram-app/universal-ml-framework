Advanced Usage
==============

Performance Optimization
-------------------------

Fast Mode for Large Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When working with large datasets (>100K rows), enable fast mode for significant speed improvements:

.. code-block:: python

   # Enable fast mode
   pipeline = UniversalMLPipeline(fast_mode=True)
   
   # Or enable during execution
   pipeline.run_pipeline('large_data.csv', 'target', fast_mode=True)

**Fast Mode Optimizations:**

- Reduced model set (2-3 fastest algorithms)
- Fewer cross-validation folds (3 instead of 5)
- Optimized hyperparameters
- 70% speed improvement on average

Multi-Core Processing
~~~~~~~~~~~~~~~~~~~~~

Leverage all available CPU cores for parallel processing:

.. code-block:: python

   # Use all available cores (default)
   pipeline = UniversalMLPipeline(n_jobs=-1)
   
   # Use specific number of cores
   pipeline = UniversalMLPipeline(n_jobs=4)
   
   # Single-core processing
   pipeline = UniversalMLPipeline(n_jobs=1)

**Parallel Components:**

- Model training during cross-validation
- Hyperparameter search (Grid/Random/Bayesian)
- Individual algorithm parallelization

Hyperparameter Tuning Strategies
---------------------------------

Grid Search
~~~~~~~~~~~

Exhaustive search through all parameter combinations:

.. code-block:: python

   pipeline = UniversalMLPipeline(tuning_method='grid')

**Pros:** Thorough exploration, guaranteed optimal within grid
**Cons:** Computationally expensive, scales exponentially

Random Search
~~~~~~~~~~~~~

Random sampling from parameter distributions (default):

.. code-block:: python

   pipeline = UniversalMLPipeline(tuning_method='random')

**Pros:** Efficient, good performance/time ratio
**Cons:** May miss optimal combinations

Bayesian Optimization
~~~~~~~~~~~~~~~~~~~~~

Smart parameter exploration using Gaussian processes:

.. code-block:: python

   # Requires: pip install scikit-optimize
   pipeline = UniversalMLPipeline(tuning_method='bayesian')

**Pros:** Intelligent search, fewer iterations needed
**Cons:** Additional dependency, complex implementation

Custom Feature Engineering
---------------------------

Custom Preprocessing Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply custom transformations before the pipeline:

.. code-block:: python

   def custom_feature_engineering(df):
       # Create interaction features
       df['feature_interaction'] = df['feature1'] * df['feature2']
       
       # Log transformation for skewed features
       df['log_feature'] = np.log1p(df['skewed_feature'])
       
       # Binning continuous variables
       df['age_group'] = pd.cut(df['age'], bins=[0, 25, 50, 75, 100], 
                               labels=['young', 'adult', 'middle', 'senior'])
       
       return df

   pipeline.run_pipeline(
       'data.csv', 
       'target',
       feature_engineering_func=custom_feature_engineering
   )

Manual Feature Selection
~~~~~~~~~~~~~~~~~~~~~~~~

Override automatic feature detection:

.. code-block:: python

   # Define custom feature types
   pipeline.feature_types = {
       'numeric': ['age', 'income', 'score'],
       'categorical': ['city', 'job_type', 'education'],
       'binary': ['has_phone', 'is_married', 'owns_car']
   }
   
   # Or specify exact features to use
   custom_features = ['age', 'income', 'city', 'education']
   pipeline.run_pipeline('data.csv', 'target', custom_features=custom_features)

Advanced Configuration
----------------------

Verbose Mode
~~~~~~~~~~~~

Get detailed progress information:

.. code-block:: python

   pipeline = UniversalMLPipeline(verbose=True)

**Verbose Output Includes:**

- Model-by-model training progress
- Fold-by-fold cross-validation scores
- Detailed hyperparameter tuning results
- Step-by-step pipeline execution

Custom ID Columns
~~~~~~~~~~~~~~~~~

Handle datasets with custom identifier columns:

.. code-block:: python

   # Use PassengerId from Titanic dataset
   pipeline.run_pipeline(
       'titanic_train.csv', 
       'Survived',
       'titanic_test.csv',
       id_column='PassengerId'
   )

Column Exclusion
~~~~~~~~~~~~~~~~

Exclude irrelevant columns from training:

.. code-block:: python

   pipeline.run_pipeline(
       'data.csv',
       'target',
       exclude_columns=['id', 'timestamp', 'name', 'description']
   )

Model Customization
-------------------

Adding Custom Models
~~~~~~~~~~~~~~~~~~~~

Extend the framework with your own algorithms:

.. code-block:: python

   from sklearn.ensemble import ExtraTreesClassifier
   
   # Add custom model after initialization
   pipeline = UniversalMLPipeline()
   pipeline.models['ExtraTrees'] = ExtraTreesClassifier(random_state=42)
   
   # Run pipeline with extended model set
   pipeline.run_pipeline('data.csv', 'target')

Custom Parameter Grids
~~~~~~~~~~~~~~~~~~~~~~

Define custom hyperparameter grids:

.. code-block:: python

   # Override default parameter grids
   custom_grids = {
       'RandomForest': {
           'model__n_estimators': [50, 100, 200, 500],
           'model__max_depth': [5, 10, 20, None],
           'model__min_samples_split': [2, 5, 10, 20]
       }
   }
   
   # Apply custom grids (requires manual implementation)
   pipeline._get_param_grids = lambda: custom_grids

Production Deployment
---------------------

Model Loading and Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load saved models for production use:

.. code-block:: python

   import joblib
   import json
   
   # Load trained model
   model = joblib.load('best_model.pkl')
   
   # Load model metadata
   with open('model_info.json', 'r') as f:
       model_info = json.load(f)
   
   # Make predictions on new data
   predictions = model.predict(new_data)

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple datasets efficiently:

.. code-block:: python

   datasets = [
       ('customer_data.csv', 'churn', 'classification'),
       ('sales_data.csv', 'revenue', 'regression'),
       ('marketing_data.csv', 'conversion', 'classification')
   ]
   
   results = {}
   for data_path, target, problem_type in datasets:
       pipeline = UniversalMLPipeline(
           problem_type=problem_type,
           fast_mode=True  # Speed up batch processing
       )
       pipeline.run_pipeline(data_path, target)
       results[data_path] = {
           'best_model': pipeline.best_model_name,
           'cv_score': pipeline.best_score
       }

Model Monitoring
~~~~~~~~~~~~~~~~

Track model performance over time:

.. code-block:: python

   # Save model metadata with timestamp
   import datetime
   
   model_info = {
       'timestamp': datetime.datetime.now().isoformat(),
       'problem_type': pipeline.problem_type,
       'best_model': pipeline.best_model_name,
       'cv_score': pipeline.best_score,
       'feature_count': len(pipeline.feature_types['numeric'] + 
                           pipeline.feature_types['categorical'] + 
                           pipeline.feature_types['binary']),
       'training_samples': len(pipeline.X)
   }

Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Memory Errors with Large Datasets:**

.. code-block:: python

   # Enable fast mode and reduce cores
   pipeline = UniversalMLPipeline(fast_mode=True, n_jobs=2)

**Slow Training:**

.. code-block:: python

   # Use random search with fewer iterations
   pipeline = UniversalMLPipeline(
       tuning_method='random',
       fast_mode=True
   )

**Poor Model Performance:**

.. code-block:: python

   # Try different tuning method
   pipeline = UniversalMLPipeline(tuning_method='bayesian')
   
   # Or add custom feature engineering
   pipeline.run_pipeline(
       'data.csv', 
       'target',
       feature_engineering_func=your_custom_function
   )

**Missing Dependencies:**

.. code-block:: bash

   # Install optional dependencies
   pip install scikit-optimize  # For Bayesian optimization

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

Track pipeline execution time:

.. code-block:: python

   import time
   
   start_time = time.time()
   pipeline.run_pipeline('data.csv', 'target')
   execution_time = time.time() - start_time
   
   print(f"Pipeline completed in {execution_time:.2f} seconds")