# 🌟 Universal ML Framework

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/universal-ml-framework.svg)](https://badge.fury.io/py/universal-ml-framework)

**A complete, automated machine learning pipeline framework that works with any dataset.**

Build, compare, and deploy ML models with minimal code - from data loading to production-ready predictions.

## 🚀 Key Features

- 🤖 **Automated Pipeline** - Complete ML workflow from data to deployment
- 🔍 **Auto Feature Detection** - Automatically identifies numeric, categorical, and binary features  
- 📊 **Model Comparison** - Tests multiple algorithms with cross-validation
- ⚙️ **Hyperparameter Tuning** - Automatic parameter optimization (Grid/Random/Bayesian)
- 🎯 **Multi-Problem Support** - Classification and regression tasks
- ⚡ **Performance Optimization** - Fast mode and multi-core processing
- 📦 **Production Ready** - Model persistence and metadata tracking

## 📦 Quick Install

```bash
pip install universal-ml-framework
```

## 🎯 Quick Start

### Basic Usage

```python
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
```

### Titanic Example

```python
from universal_ml_framework import UniversalMLPipeline

# Predict Titanic survival with optimal settings
pipeline = UniversalMLPipeline(
    problem_type='classification', 
    tuning_method='bayesian',
    verbose=True
)

pipeline.run_pipeline(
    train_path='titanic_train.csv',
    test_path='titanic_test.csv',
    target_column='Survived',
    id_column='PassengerId'
)
```

**Output:**
```
🏆 Best model: RandomForest
✅ Best Score: 0.8456
📁 Files: predictions.csv, best_model.pkl, model_info.json
```

## 🔧 Supported Algorithms

### Classification (7 models)
- Random Forest Classifier
- Gradient Boosting Classifier  
- Logistic Regression
- Support Vector Machine
- Naive Bayes
- K-Nearest Neighbors
- Decision Tree Classifier

### Regression (6 models)
- Random Forest Regressor
- Gradient Boosting Regressor
- Linear Regression
- Support Vector Regression
- K-Nearest Neighbors
- Decision Tree Regressor

## ⚡ Performance Features

### Fast Mode
```python
# 70% faster for large datasets
pipeline = UniversalMLPipeline(fast_mode=True)
```

### Multi-Core Processing
```python
# Use all CPU cores
pipeline = UniversalMLPipeline(n_jobs=-1)
```

### Advanced Tuning
```python
# Bayesian optimization (requires: pip install scikit-optimize)
pipeline = UniversalMLPipeline(tuning_method='bayesian')
```

## 📊 What It Does Automatically

1. **Data Loading** - Reads CSV files automatically
2. **Feature Detection** - Identifies feature types (numeric/categorical/binary)
3. **Preprocessing** - Handles missing values, encoding, scaling
4. **Model Training** - Trains multiple algorithms with cross-validation
5. **Hyperparameter Tuning** - Optimizes best performing model
6. **Prediction** - Generates predictions on test data
7. **Model Saving** - Persists trained model and metadata

## 🎛️ Configuration Options

### Initialization
```python
UniversalMLPipeline(
    problem_type='classification',  # 'classification' or 'regression'
    random_state=42,               # Reproducibility
    verbose=False,                 # Detailed output
    fast_mode=False,              # Speed optimization
    tuning_method='random',       # 'grid', 'random', 'bayesian'
    n_jobs=-1                     # Parallel processing (-1 = all cores)
)
```

### Runtime Options
```python
pipeline.run_pipeline(
    train_path='data.csv',
    target_column='target',
    test_path='test.csv',          # Optional
    exclude_columns=['id'],        # Columns to exclude
    custom_features=None,          # Manual feature selection
    feature_engineering_func=None, # Custom preprocessing
    id_column='PassengerId'        # Custom ID column
)
```

## 📈 Performance Comparison

| Metric | Universal ML Framework | Manual Implementation |
|--------|----------------------|---------------------|
| **Lines of Code** | 4 | 100+ |
| **Development Time** | 2 minutes | 2-4 hours |
| **Models Tested** | 7 | 1-2 |
| **Hyperparameter Tuning** | Automatic | Manual |
| **Production Ready** | ✅ | ❌ |

## 🎯 Use Cases

- **Business Analytics** - Customer churn, sales forecasting
- **Finance** - Credit risk, fraud detection  
- **Healthcare** - Medical diagnosis, treatment prediction
- **Marketing** - Campaign response, customer segmentation
- **Real Estate** - Price prediction, market analysis
- **HR** - Employee performance, retention prediction

## 📁 Output Files

The framework automatically generates:

- **`predictions.csv`** - Test set predictions with proper ID mapping
- **`best_model.pkl`** - Trained model (joblib format) 
- **`model_info.json`** - Model metadata and performance metrics

## 🔍 Advanced Usage

### Custom Feature Engineering
```python
def custom_features(df):
    df['feature_interaction'] = df['feature1'] * df['feature2']
    df['log_feature'] = np.log1p(df['skewed_feature'])
    return df

pipeline.run_pipeline('data.csv', 'target', feature_engineering_func=custom_features)
```

### Batch Processing
```python
datasets = [
    ('customer_data.csv', 'churn', 'classification'),
    ('sales_data.csv', 'revenue', 'regression')
]

for data_path, target, problem_type in datasets:
    pipeline = UniversalMLPipeline(problem_type=problem_type, fast_mode=True)
    pipeline.run_pipeline(data_path, target)
```

## 📚 Documentation

- **📖 Full Documentation**: https://universal-ml-framework.readthedocs.io
- **🎯 Titanic Case Study**: Complete walkthrough with real dataset
- **🏗️ Architecture Guide**: Understanding the framework design  
- **⚡ Advanced Usage**: Performance optimization and customization
- **🔧 Troubleshooting**: Common issues and solutions

## 🚀 Quick Examples

### Generate Sample Data
```python
from universal_ml_framework import DataGenerator

# Generate synthetic datasets for testing
DataGenerator.generate_customer_churn()
DataGenerator.generate_house_prices() 
DataGenerator.generate_all_datasets()
```

### Load Saved Models
```python
import joblib
import json

# Load trained model
model = joblib.load('best_model.pkl')

# Load model metadata  
with open('model_info.json', 'r') as f:
    model_info = json.load(f)

# Make predictions on new data
predictions = model.predict(new_data)
```

## 🔧 Requirements

- Python 3.7+
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- numpy >= 1.21.0
- joblib >= 1.0.0

**Optional:**
- scikit-optimize (for Bayesian optimization)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⭐ Support

If you find this project helpful, please give it a star on GitHub!

---

**Made with ❤️ for the ML community**