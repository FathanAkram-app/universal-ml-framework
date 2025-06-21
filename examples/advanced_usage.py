# ADVANCED USAGE EXAMPLES
# Demonstrasi penggunaan advanced Universal ML Framework

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import UniversalMLPipeline
from utils.data_generator import DataGenerator
import pandas as pd
import numpy as np

def example_custom_feature_selection():
    """Example: Custom feature selection"""
    print("🎯 ADVANCED EXAMPLE 1: CUSTOM FEATURE SELECTION")
    print("=" * 60)
    
    # Generate data
    DataGenerator.generate_house_prices()
    
    # Load data
    train_df = pd.read_csv('house_train.csv')
    
    # Custom feature selection
    custom_features = ['LotArea', 'YearBuilt', 'BedroomAbvGr', 'BathroomAbvGr', 'Neighborhood']
    
    pipeline = UniversalMLPipeline(problem_type='regression')
    pipeline.load_data('house_train.csv', 'house_test.csv', 'SalePrice')
    
    # Manual feature type definition
    pipeline.feature_types = {
        'numeric': ['LotArea', 'YearBuilt', 'BedroomAbvGr', 'BathroomAbvGr'],
        'categorical': ['Neighborhood'],
        'binary': []
    }
    
    pipeline.create_preprocessor()
    pipeline.prepare_data(custom_features)
    pipeline.define_models()
    pipeline.cross_validate_models()
    pipeline.hyperparameter_tuning()
    pipeline.make_predictions()
    pipeline.save_model('custom_house_model.pkl')
    
    print("✅ Custom feature selection completed")
    return pipeline

def example_model_comparison():
    """Example: Detailed model comparison"""
    print("\n🔍 ADVANCED EXAMPLE 2: DETAILED MODEL COMPARISON")
    print("=" * 60)
    
    # Generate data
    DataGenerator.generate_customer_churn()
    
    pipeline = UniversalMLPipeline(problem_type='classification')
    pipeline.load_data('customer_train.csv', 'customer_test.csv', 'Churn')
    
    # Auto-detect features
    pipeline.auto_detect_features(pipeline.train_df, ['Churn'])
    pipeline.create_preprocessor()
    pipeline.prepare_data()
    pipeline.define_models()
    
    # Cross validate models
    pipeline.cross_validate_models()
    
    # Print detailed comparison
    print("\n📊 DETAILED MODEL COMPARISON:")
    print("-" * 40)
    for model_name, results in pipeline.cv_results.items():
        scores = results['scores']
        print(f"\n{model_name}:")
        print(f"  Mean Accuracy: {results['mean']:.4f}")
        print(f"  Std Deviation: {results['std']:.4f}")
        print(f"  Min Score: {scores.min():.4f}")
        print(f"  Max Score: {scores.max():.4f}")
        print(f"  Individual Scores: {[f'{s:.3f}' for s in scores]}")
    
    pipeline.hyperparameter_tuning()
    pipeline.make_predictions()
    
    print("✅ Detailed model comparison completed")
    return pipeline

def example_feature_engineering():
    """Example: Advanced feature engineering"""
    print("\n🔧 ADVANCED EXAMPLE 3: FEATURE ENGINEERING")
    print("=" * 60)
    
    # Generate base data
    np.random.seed(42)
    n_samples = 1000
    
    # Create more complex dataset
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'education_years': np.random.randint(8, 20, n_samples),
        'work_experience': np.random.randint(0, 40, n_samples),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n_samples),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'has_children': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }
    
    # Create target with complex relationships
    target_prob = (
        (data['income'] > 60000) * 0.3 +
        (data['education_years'] > 16) * 0.2 +
        (data['age'] > 40) * 0.2 +
        (data['work_experience'] > 10) * 0.15 +
        data['has_children'] * 0.1 +
        np.random.random(n_samples) * 0.2
    )
    
    data['target'] = (target_prob > 0.5).astype(int)
    
    df = pd.DataFrame(data)
    
    # Advanced feature engineering
    df['income_per_education'] = df['income'] / (df['education_years'] + 1)
    df['experience_ratio'] = df['work_experience'] / (df['age'] - 18 + 1)
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])
    df['income_quartile'] = pd.qcut(df['income'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    # Save data
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:].drop('target', axis=1)
    
    train_df.to_csv('advanced_train.csv', index=False)
    test_df.to_csv('advanced_test.csv', index=False)
    
    # Run pipeline
    pipeline = UniversalMLPipeline(problem_type='classification')
    pipeline.run_pipeline(
        train_path='advanced_train.csv',
        test_path='advanced_test.csv',
        target_column='target'
    )
    
    print("✅ Advanced feature engineering completed")
    return pipeline

def example_pipeline_customization():
    """Example: Pipeline customization"""
    print("\n⚙️ ADVANCED EXAMPLE 4: PIPELINE CUSTOMIZATION")
    print("=" * 60)
    
    # Generate data
    DataGenerator.generate_sales_forecasting()
    
    # Create custom pipeline
    pipeline = UniversalMLPipeline(problem_type='regression', random_state=123)
    
    # Load data
    pipeline.load_data('sales_train.csv', 'sales_test.csv', 'Sales')
    
    # Custom feature detection with different thresholds
    print("🔍 Custom feature detection...")
    
    numeric_features = []
    categorical_features = []
    binary_features = []
    
    for col in pipeline.train_df.columns:
        if col == 'Sales':
            continue
            
        # Custom logic for feature detection
        if pipeline.train_df[col].dtype in ['int64', 'float64']:
            if pipeline.train_df[col].nunique() == 2:
                binary_features.append(col)
            elif pipeline.train_df[col].nunique() > 5:  # Custom threshold
                numeric_features.append(col)
            else:
                categorical_features.append(col)
        else:
            categorical_features.append(col)
    
    pipeline.feature_types = {
        'numeric': numeric_features,
        'categorical': categorical_features,
        'binary': binary_features
    }
    
    print(f"✅ Custom feature types: {pipeline.feature_types}")
    
    # Continue with pipeline
    pipeline.create_preprocessor()
    pipeline.prepare_data()
    
    # Custom model definition
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    
    pipeline.models = {
        'RandomForest': pipeline.models['RandomForest'],
        'GradientBoosting': GradientBoostingRegressor(random_state=pipeline.random_state),
        'DecisionTree': DecisionTreeRegressor(random_state=pipeline.random_state)
    }
    
    print(f"✅ Custom models: {list(pipeline.models.keys())}")
    
    pipeline.cross_validate_models()
    pipeline.hyperparameter_tuning()
    pipeline.make_predictions()
    pipeline.save_model('custom_sales_model.pkl')
    
    print("✅ Pipeline customization completed")
    return pipeline

if __name__ == "__main__":
    print("🚀 UNIVERSAL ML FRAMEWORK - ADVANCED USAGE EXAMPLES")
    print("=" * 80)
    
    try:
        # Advanced examples
        pipeline1 = example_custom_feature_selection()
        pipeline2 = example_model_comparison()
        pipeline3 = example_feature_engineering()
        pipeline4 = example_pipeline_customization()
        
        print("\n🎉 ALL ADVANCED EXAMPLES COMPLETED!")
        print("=" * 80)
        print("✅ Custom Feature Selection")
        print("✅ Detailed Model Comparison")
        print("✅ Advanced Feature Engineering")
        print("✅ Pipeline Customization")
        
        print("\n💡 KEY LEARNINGS:")
        print("-" * 40)
        print("• Framework is highly customizable")
        print("• Feature engineering can be done manually")
        print("• Model comparison provides detailed insights")
        print("• Pipeline can be extended with custom models")
        print("• All components are modular and flexible")
        
    except Exception as e:
        print(f"❌ Error in advanced examples: {e}")
        import traceback
        traceback.print_exc()