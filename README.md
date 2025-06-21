# 🌟 Universal ML Framework

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

A complete, automated machine learning pipeline framework that works with any dataset. Build, compare, and deploy ML models with minimal code.

## 🚀 Key Features

- **🤖 Automated Pipeline** - Complete ML workflow from data to deployment
- **🔍 Auto Feature Detection** - Automatically detects numeric, categorical, and binary features
- **📊 Model Comparison** - Compares multiple algorithms with cross-validation
- **⚙️ Hyperparameter Tuning** - Automatic parameter optimization
- **🎯 Multi-Problem Support** - Classification and regression tasks
- **📦 Production Ready** - Model persistence and metadata tracking

## 📦 Installation

```bash
# Install from PyPI
pip install universal-ml-framework

# Or install from source
git clone https://github.com/FathanAkram-App/universal-ml-framework.git
cd universal-ml-framework
pip install -e .
```

## 🎯 Quick Start

### Basic Usage

```python
from universal_ml_framework import UniversalMLPipeline

# Classification
pipeline = UniversalMLPipeline(problem_type='classification')
pipeline.run_pipeline(
    train_path='data.csv',
    target_column='target',
    test_path='test.csv'
)

# Regression
pipeline = UniversalMLPipeline(problem_type='regression')
pipeline.run_pipeline(
    train_path='data.csv',
    target_column='price'
)
```

### Quick Setup Functions

```python
from universal_ml_framework import quick_classification_pipeline

# One-liner for classification
result = quick_classification_pipeline('data.csv', 'target_column')
```

### Generate Sample Data

```python
from universal_ml_framework import DataGenerator

# Generate synthetic datasets for testing
DataGenerator.generate_customer_churn()
DataGenerator.generate_house_prices()
DataGenerator.generate_sales_forecasting()
```

## 🔧 Supported Algorithms

### Classification
- Random Forest Classifier
- Logistic Regression  
- Support Vector Machine

### Regression
- Random Forest Regressor
- Linear Regression
- Support Vector Regression

## 📊 What It Does

1. **Data Loading** - Reads CSV files automatically
2. **Feature Detection** - Identifies feature types (numeric/categorical/binary)
3. **Preprocessing** - Handles missing values, encoding, scaling
4. **Model Training** - Trains multiple algorithms with cross-validation
5. **Hyperparameter Tuning** - Optimizes best performing model
6. **Prediction** - Generates predictions on test data
7. **Model Saving** - Persists trained model and metadata

## 📈 Output Files

- `predictions.csv` - Test set predictions
- `best_model.pkl` - Trained model (joblib format)
- `model_info.json` - Model metadata and performance

## 🎛️ Customization

### Custom Feature Types
```python
pipeline.feature_types = {
    'numeric': ['age', 'income'],
    'categorical': ['city', 'category'],
    'binary': ['has_feature']
}
```

### Exclude Columns
```python
pipeline.run_pipeline(
    train_path='data.csv',
    target_column='target',
    exclude_columns=['id', 'timestamp']
)
```

## 🧪 Demo

```bash
# Run complete demo with synthetic data
python -c "from universal_ml_framework import DataGenerator; DataGenerator.generate_all_datasets()"
```

## 📁 Project Structure

```
universal_ml_framework/
├── core/
│   └── pipeline.py          # Main pipeline class
├── utils/
│   ├── helpers.py           # Helper functions
│   └── data_generator.py    # Synthetic data generation
├── configs/
│   └── dataset_configs.py   # Predefined configurations
└── examples/
    ├── basic_usage.py       # Basic examples
    └── advanced_usage.py    # Advanced examples
```

## 🎯 Use Cases

- **Business Analytics** - Customer churn, sales forecasting
- **Finance** - Credit risk, fraud detection
- **Healthcare** - Medical diagnosis, treatment prediction
- **Marketing** - Campaign response, customer segmentation
- **Real Estate** - Price prediction, market analysis
- **HR** - Employee performance, retention prediction

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/)
- Powered by [pandas](https://pandas.pydata.org/) and [numpy](https://numpy.org/)