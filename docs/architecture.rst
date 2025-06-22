Architecture & Design
====================

Core Components
---------------

UniversalMLPipeline Class
~~~~~~~~~~~~~~~~~~~~~~~~~

The main pipeline class that orchestrates the entire machine learning workflow.

.. code-block:: python

   class UniversalMLPipeline:
       def __init__(self, problem_type, random_state, verbose, fast_mode, tuning_method, n_jobs)
       def load_data(self, train_path, test_path, target_column)
       def auto_detect_features(self, df, exclude_columns)
       def create_preprocessor(self)
       def prepare_data(self, custom_features)
       def define_models(self)
       def cross_validate_models(self)
       def hyperparameter_tuning(self)
       def make_predictions(self, save_predictions, id_column)
       def save_model(self, filename)
       def run_pipeline(self, **kwargs)

Key Attributes
~~~~~~~~~~~~~~

:problem_type: 'classification' or 'regression'
:fast_mode: Speed optimization for large datasets
:tuning_method: 'grid', 'random', or 'bayesian'
:n_jobs: Number of cores for parallel processing
:feature_types: Dictionary containing detected feature types
:cv_results: Cross-validation results for all models
:best_pipeline: Best performing pipeline after tuning

Pipeline Workflow
-----------------

The framework follows an 8-stage automated workflow:

1. **Data Loading**
   
   - Load CSV files using pandas
   - Create backup of original data
   - Display dataset information
   - Handle missing test data

2. **Feature Detection**
   
   - Automatically categorize features into:
     
     - **Numeric**: Continuous numerical features
     - **Categorical**: Text or discrete categorical features  
     - **Binary**: Boolean or 0/1 features
   
   - Skip columns with >80% missing values

3. **Preprocessing**
   
   - **Numeric**: Median imputation → Standard scaling
   - **Categorical**: Constant imputation → One-hot encoding
   - **Binary**: Zero imputation
   - Uses ColumnTransformer for efficient processing

4. **Model Definition**
   
   **Fast Mode** (for large datasets):
   
   - Classification: RandomForest, LogisticRegression, NaiveBayes
   - Regression: RandomForest, LinearRegression
   
   **Full Mode**:
   
   - Classification: 7 algorithms
   - Regression: 6 algorithms

5. **Cross Validation**
   
   - **Classification**: StratifiedKFold (preserves class distribution)
   - **Regression**: KFold
   - **Folds**: 3 (fast mode) or 5 (normal mode)
   - **Parallel**: Multi-core cross-validation

6. **Hyperparameter Tuning**
   
   - **Grid Search**: Exhaustive parameter search
   - **Random Search**: Random parameter sampling (default)
   - **Bayesian Search**: Smart parameter exploration
   - Comprehensive parameter grids for each algorithm

7. **Prediction Generation**
   
   - Generate predictions on test set
   - Support for custom ID columns
   - Automatic CSV export
   - Prediction statistics

8. **Model Persistence**
   
   - Save trained pipeline (joblib)
   - Export model metadata (JSON)
   - Include performance metrics and feature information

Design Principles
-----------------

Universal Design
~~~~~~~~~~~~~~~~

- Works with any tabular dataset
- No manual feature engineering required
- Automatic problem type detection
- Robust error handling

Automation First
~~~~~~~~~~~~~~~~

- Minimal user configuration
- Smart defaults for all parameters
- Automatic feature type detection
- End-to-end pipeline execution

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

- Multi-core parallel processing
- Fast mode for large datasets
- Memory-efficient transformations
- Scalable architecture

Production Ready
~~~~~~~~~~~~~~~~

- Model persistence and versioning
- Comprehensive metadata tracking
- Reproducible results
- Error handling and validation

Extensibility
~~~~~~~~~~~~~

- Plugin architecture for custom models
- Custom preprocessing functions
- Configurable validation strategies
- Modular component design

Performance Characteristics
---------------------------

Speed Benchmarks
~~~~~~~~~~~~~~~~

**Typical Performance** (10K rows, 20 features):

- **Fast Mode**: 2-5 minutes
- **Normal Mode**: 5-15 minutes  
- **Bayesian Tuning**: 10-30 minutes

Memory Usage
~~~~~~~~~~~~

- **Small Dataset** (<1K rows): ~50MB
- **Medium Dataset** (10K rows): ~200MB
- **Large Dataset** (100K rows): ~1-2GB

Scalability Features
~~~~~~~~~~~~~~~~~~~~

- **Fast Mode**: 70% speed improvement
- **Parallel Processing**: Linear scaling with CPU cores
- **Memory Management**: Efficient data handling
- **Reduced Model Set**: Optimized algorithm selection

Error Handling Strategy
-----------------------

Data Quality Validation
~~~~~~~~~~~~~~~~~~~~~~~

- Missing value threshold checking
- Feature type validation
- Empty dataset handling
- Corrupted file detection

Graceful Degradation
~~~~~~~~~~~~~~~~~~~~

- Fallback mechanisms for missing components
- Default parameter substitution
- Alternative algorithm selection
- Robust preprocessing pipelines

Validation Framework
~~~~~~~~~~~~~~~~~~~~

- Cross-validation integrity checks
- Score normalization and validation
- Pipeline consistency verification
- Output format validation