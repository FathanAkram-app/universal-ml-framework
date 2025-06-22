Troubleshooting Guide
====================

Common Issues and Solutions
---------------------------

Installation Issues
~~~~~~~~~~~~~~~~~~~

**ImportError: No module named 'universal_ml_framework'**

.. code-block:: bash

   # Ensure package is installed
   pip install universal-ml-framework
   
   # Or install from source
   pip install -e .

**Missing Dependencies**

.. code-block:: bash

   # Install all required dependencies
   pip install pandas scikit-learn numpy joblib
   
   # For Bayesian optimization (optional)
   pip install scikit-optimize

Data Loading Problems
~~~~~~~~~~~~~~~~~~~~~

**FileNotFoundError: No such file or directory**

.. code-block:: python

   import os
   
   # Check if file exists
   if os.path.exists('your_data.csv'):
       pipeline.run_pipeline('your_data.csv', 'target')
   else:
       print("File not found. Check the file path.")

**UnicodeDecodeError when loading CSV**

.. code-block:: python

   # Try different encodings
   import pandas as pd
   
   try:
       df = pd.read_csv('data.csv', encoding='utf-8')
   except UnicodeDecodeError:
       df = pd.read_csv('data.csv', encoding='latin-1')

**Empty DataFrame after loading**

.. code-block:: python

   # Check data after loading
   pipeline.load_data('train.csv', target_column='target')
   print(f"Training data shape: {pipeline.train_df.shape}")
   
   if pipeline.train_df.empty:
       print("Warning: Empty dataset loaded")

Memory and Performance Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**MemoryError: Unable to allocate memory**

.. code-block:: python

   # Solution 1: Enable fast mode
   pipeline = UniversalMLPipeline(fast_mode=True)
   
   # Solution 2: Reduce parallel jobs
   pipeline = UniversalMLPipeline(n_jobs=2)
   
   # Solution 3: Sample your data
   import pandas as pd
   df = pd.read_csv('large_data.csv').sample(n=10000)
   df.to_csv('sampled_data.csv', index=False)

**Training takes too long**

.. code-block:: python

   # Enable fast mode for quick results
   pipeline = UniversalMLPipeline(
       fast_mode=True,           # Fewer models and CV folds
       tuning_method='random',   # Faster than grid search
       n_jobs=-1                 # Use all CPU cores
   )

**High CPU usage**

.. code-block:: python

   # Limit CPU cores usage
   pipeline = UniversalMLPipeline(n_jobs=2)  # Use only 2 cores

Feature Detection Issues
~~~~~~~~~~~~~~~~~~~~~~~~

**No features detected**

.. code-block:: python

   # Check your data
   print(pipeline.train_df.dtypes)
   print(pipeline.train_df.describe())
   
   # Manually specify features if needed
   custom_features = ['feature1', 'feature2', 'feature3']
   pipeline.run_pipeline('data.csv', 'target', custom_features=custom_features)

**Wrong feature types detected**

.. code-block:: python

   # Override automatic detection
   pipeline.feature_types = {
       'numeric': ['age', 'income', 'score'],
       'categorical': ['city', 'category'],
       'binary': ['is_member', 'has_discount']
   }

**Too many categorical features after one-hot encoding**

.. code-block:: python

   # Exclude high-cardinality categorical columns
   pipeline.run_pipeline(
       'data.csv', 
       'target',
       exclude_columns=['high_cardinality_column']
   )

Model Training Problems
~~~~~~~~~~~~~~~~~~~~~~~

**ValueError: Input contains NaN**

.. code-block:: python

   # Check for missing values
   print(pipeline.train_df.isnull().sum())
   
   # The pipeline should handle this automatically
   # If it persists, check your custom feature engineering function

**All models perform poorly**

.. code-block:: python

   # Check data quality
   print(f"Target distribution: {pipeline.train_df[target_column].value_counts()}")
   
   # Try different problem type
   pipeline = UniversalMLPipeline(problem_type='regression')  # or 'classification'
   
   # Add custom feature engineering
   def improve_features(df):
       # Add your feature engineering here
       return df
   
   pipeline.run_pipeline('data.csv', 'target', feature_engineering_func=improve_features)

**Hyperparameter tuning fails**

.. code-block:: python

   # Try different tuning method
   pipeline = UniversalMLPipeline(tuning_method='random')  # Instead of 'grid'
   
   # Or disable tuning temporarily
   pipeline.hyperparameter_tuning = lambda: None

Prediction Issues
~~~~~~~~~~~~~~~~~

**No test data available warning**

.. code-block:: python

   # Provide test data path
   pipeline.run_pipeline('train.csv', 'target', test_path='test.csv')
   
   # Or make predictions separately
   predictions = pipeline.make_predictions(save_predictions=True)

**ID column mismatch in predictions**

.. code-block:: python

   # Specify correct ID column
   pipeline.run_pipeline(
       'train.csv', 
       'target', 
       'test.csv',
       id_column='PassengerId'  # Use your actual ID column name
   )

**Predictions file not generated**

.. code-block:: python

   # Check if test data is provided
   if pipeline.test_df is not None:
       predictions = pipeline.make_predictions(save_predictions=True)
   else:
       print("No test data provided for predictions")

Error Messages and Solutions
----------------------------

Scikit-learn Related Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**ValueError: Unknown label type**

.. code-block:: python

   # Check target variable type
   print(f"Target type: {type(pipeline.y.iloc[0])}")
   print(f"Unique values: {pipeline.y.unique()}")
   
   # Ensure proper encoding for classification
   if pipeline.problem_type == 'classification':
       from sklearn.preprocessing import LabelEncoder
       le = LabelEncoder()
       pipeline.y = le.fit_transform(pipeline.y)

**ValueError: Input contains infinity or a value too large**

.. code-block:: python

   # Check for infinite values
   import numpy as np
   print(f"Infinite values: {np.isinf(pipeline.train_df.select_dtypes(include=[np.number])).sum().sum()}")
   
   # Replace infinite values
   pipeline.train_df.replace([np.inf, -np.inf], np.nan, inplace=True)

**ConvergenceWarning: lbfgs failed to converge**

.. code-block:: python

   # Increase max_iter for LogisticRegression
   pipeline.models['LogisticRegression'].max_iter = 2000
   
   # Or use different solver
   pipeline.models['LogisticRegression'].solver = 'saga'

Pandas Related Errors
~~~~~~~~~~~~~~~~~~~~~

**KeyError: Column not found**

.. code-block:: python

   # Check column names
   print("Available columns:", pipeline.train_df.columns.tolist())
   
   # Check for extra spaces or different case
   target_column = target_column.strip()

**DtypeWarning: Columns have mixed types**

.. code-block:: python

   # Specify dtypes when loading
   import pandas as pd
   df = pd.read_csv('data.csv', dtype={'mixed_column': str})

Performance Optimization Tips
-----------------------------

Speed Up Training
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Fastest configuration
   pipeline = UniversalMLPipeline(
       fast_mode=True,           # Reduced model set
       tuning_method='random',   # Faster than grid search
       n_jobs=-1,               # Use all cores
       verbose=False            # Reduce output overhead
   )

Reduce Memory Usage
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Memory-efficient configuration
   pipeline = UniversalMLPipeline(
       fast_mode=True,    # Fewer models in memory
       n_jobs=1          # Reduce parallel overhead
   )
   
   # Process data in chunks if very large
   chunk_size = 10000
   for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
       # Process each chunk separately

Debug Mode
~~~~~~~~~~

.. code-block:: python

   # Enable verbose mode for debugging
   pipeline = UniversalMLPipeline(verbose=True)
   
   # Check intermediate results
   pipeline.load_data('train.csv', target_column='target')
   print("Feature types:", pipeline.feature_types)
   
   pipeline.create_preprocessor()
   pipeline.prepare_data()
   print("X shape:", pipeline.X.shape)
   print("y shape:", pipeline.y.shape)

Getting Help
------------

Check Documentation
~~~~~~~~~~~~~~~~~~~

- **API Reference**: Detailed method documentation
- **Examples**: Working code examples
- **Architecture**: Understanding the framework design

Common Debugging Steps
~~~~~~~~~~~~~~~~~~~~~~

1. **Check Data Quality**

   .. code-block:: python
   
      print(df.info())
      print(df.describe())
      print(df.isnull().sum())

2. **Verify Configuration**

   .. code-block:: python
   
      print(f"Problem type: {pipeline.problem_type}")
      print(f"Feature types: {pipeline.feature_types}")
      print(f"Models: {list(pipeline.models.keys())}")

3. **Test with Sample Data**

   .. code-block:: python
   
      # Test with small sample first
      sample_df = df.sample(n=1000)
      sample_df.to_csv('sample.csv', index=False)

4. **Enable Verbose Output**

   .. code-block:: python
   
      pipeline = UniversalMLPipeline(verbose=True)

Report Issues
~~~~~~~~~~~~~

When reporting issues, please include:

- Python version and operating system
- Package versions (pandas, scikit-learn, etc.)
- Complete error traceback
- Minimal code example to reproduce the issue
- Dataset characteristics (size, types, etc.)

**Example Issue Report:**

.. code-block:: text

   Environment:
   - Python 3.8.10
   - universal-ml-framework 1.0.1
   - pandas 1.3.0
   - scikit-learn 1.0.2
   
   Issue:
   MemoryError when processing 100K row dataset
   
   Code:
   pipeline = UniversalMLPipeline()
   pipeline.run_pipeline('large_data.csv', 'target')
   
   Error:
   [Full traceback here]