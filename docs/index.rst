🌟 Universal ML Framework
=========================

.. image:: https://img.shields.io/badge/python-3.7+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.7+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

**A complete, automated machine learning pipeline framework that works with any dataset.**

Build, compare, and deploy ML models with minimal code.

🚀 Key Features
---------------

* 🤖 **Automated Pipeline** - Complete ML workflow from data to deployment
* 🔍 **Auto Feature Detection** - Automatically identifies numeric, categorical, and binary features  
* 📊 **Model Comparison** - Tests multiple algorithms with cross-validation
* ⚙️ **Hyperparameter Tuning** - Automatic parameter optimization
* 🎯 **Multi-Problem Support** - Classification and regression tasks
* 📦 **Production Ready** - Model persistence and metadata tracking

📦 Quick Install
----------------

.. code-block:: bash

   pip install universal-ml-framework

🎯 Quick Start
--------------

.. code-block:: python

   from universal_ml_framework import UniversalMLPipeline

   # Create and run pipeline
   pipeline = UniversalMLPipeline(problem_type='classification')
   pipeline.run_pipeline(
       train_path='data.csv',
       target_column='target',
       test_path='test.csv'
   )

   print(f"Best model: {pipeline.best_model_name}")
   print(f"Best score: {pipeline.best_score:.4f}")

📚 Documentation
----------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   titanic_case_study
   architecture
   advanced
   api
   examples
   troubleshooting

🔧 Supported Algorithms
-----------------------

**Classification:**

* Random Forest Classifier
* Logistic Regression  
* Support Vector Machine

**Regression:**

* Random Forest Regressor
* Linear Regression
* Support Vector Regression

📈 What It Does
---------------

1. **Data Loading** - Reads CSV files automatically
2. **Feature Detection** - Identifies feature types (numeric/categorical/binary)
3. **Preprocessing** - Handles missing values, encoding, scaling
4. **Model Training** - Trains multiple algorithms with cross-validation
5. **Hyperparameter Tuning** - Optimizes best performing model
6. **Prediction** - Generates predictions on test data
7. **Model Saving** - Persists trained model and metadata

🎯 Use Cases
------------

* **Business Analytics** - Customer churn, sales forecasting
* **Finance** - Credit risk, fraud detection
* **Healthcare** - Medical diagnosis, treatment prediction
* **Marketing** - Campaign response, customer segmentation
* **Real Estate** - Price prediction, market analysis
* **HR** - Employee performance, retention prediction