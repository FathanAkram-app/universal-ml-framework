# API Reference

## Core Classes

### UniversalMLPipeline

```{eval-rst}
.. autoclass:: universal_ml_framework.UniversalMLPipeline
   :members:
   :undoc-members:
   :show-inheritance:
```

## Helper Functions

### Quick Setup Functions

```{eval-rst}
.. autofunction:: universal_ml_framework.quick_classification_pipeline
.. autofunction:: universal_ml_framework.quick_regression_pipeline
.. autofunction:: universal_ml_framework.run_pipeline_with_config
.. autofunction:: universal_ml_framework.list_available_configs
```

## Data Generation

### DataGenerator

```{eval-rst}
.. autoclass:: universal_ml_framework.DataGenerator
   :members:
   :undoc-members:
   :show-inheritance:
```

## Configuration Classes

### PipelineConfigs

```{eval-rst}
.. autoclass:: universal_ml_framework.configs.dataset_configs.PipelineConfigs
   :members:
   :undoc-members:
   :show-inheritance:
```

## Method Details

### UniversalMLPipeline Methods

#### run_pipeline()
Main method to execute the complete ML pipeline.

**Parameters:**
- `train_path` (str): Path to training CSV file
- `target_column` (str): Name of target column
- `test_path` (str, optional): Path to test CSV file
- `problem_type` (str): 'classification' or 'regression'
- `exclude_columns` (list, optional): Columns to exclude from features
- `custom_features` (list, optional): Custom feature list

#### load_data()
Load training and test data from CSV files.

#### auto_detect_features()
Automatically detect feature types (numeric, categorical, binary).

#### create_preprocessor()
Create preprocessing pipeline for data transformation.

#### cross_validate_models()
Compare multiple models using cross-validation.

#### hyperparameter_tuning()
Optimize hyperparameters for the best model.

#### make_predictions()
Generate predictions on test data.

#### save_model()
Save trained model and metadata to files.