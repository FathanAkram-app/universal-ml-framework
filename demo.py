# DEMO UNIVERSAL ML FRAMEWORK
# Demonstrasi lengkap framework untuk berbagai dataset

import sys
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
import joblib
import json
import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Import our framework components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.pipeline import UniversalMLPipeline
from utils.data_generator import DataGenerator

def demo_1_customer_churn():
    """Demo 1: Customer Churn Classification"""
    print("🎯 DEMO 1: CUSTOMER CHURN CLASSIFICATION")
    print("=" * 60)
    
    # Generate synthetic customer data
    DataGenerator.generate_customer_churn()
    
    # Run pipeline
    pipeline = UniversalMLPipeline(problem_type='classification')
    pipeline.run_pipeline(
        train_path='data/customer_train.csv',
        test_path='data/customer_test.csv',
        target_column='Churn'
    )
    
    return pipeline

def demo_2_house_prices():
    """Demo 2: House Prices Regression"""
    print("\n🏠 DEMO 2: HOUSE PRICES REGRESSION")
    print("=" * 60)
    
    # Generate synthetic house data
    DataGenerator.generate_house_prices()
    
    # Run pipeline
    pipeline = UniversalMLPipeline(problem_type='regression')
    pipeline.run_pipeline(
        train_path='data/house_train.csv',
        test_path='data/house_test.csv',
        target_column='SalePrice'
    )
    
    return pipeline

def demo_3_sales_forecasting():
    """Demo 3: Sales Forecasting Regression"""
    print("\n📈 DEMO 3: SALES FORECASTING REGRESSION")
    print("=" * 60)
    
    # Generate synthetic sales data
    DataGenerator.generate_sales_forecasting()
    
    # Run pipeline
    pipeline = UniversalMLPipeline(problem_type='regression')
    pipeline.run_pipeline(
        train_path='data/sales_train.csv',
        test_path='data/sales_test.csv',
        target_column='Sales'
    )
    
    return pipeline

def demo_4_custom_dataset():
    """Demo 4: Custom Dataset - Employee Performance"""
    print("\n👨‍💼 DEMO 4: EMPLOYEE PERFORMANCE CLASSIFICATION")
    print("=" * 60)
    
    # Generate custom employee performance dataset
    np.random.seed(789)
    n_samples = 800
    
    data = {
        'Age': np.random.randint(22, 65, n_samples),
        'YearsExperience': np.random.randint(0, 30, n_samples),
        'Salary': np.random.normal(60000, 20000, n_samples),
        'Department': np.random.choice(['IT', 'Sales', 'HR', 'Finance', 'Marketing'], n_samples),
        'Education': np.random.choice(['Bachelor', 'Master', 'PhD'], n_samples, p=[0.6, 0.3, 0.1]),
        'WorkHours': np.random.normal(40, 8, n_samples),
        'ProjectsCompleted': np.random.poisson(5, n_samples),
        'TrainingHours': np.random.normal(20, 10, n_samples),
        'HasCertification': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'RemoteWork': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }
    
    # Create performance score based on realistic factors
    performance_score = (
        (data['YearsExperience'] > 5) * 0.25 +
        (data['Salary'] > 70000) * 0.2 +
        (data['ProjectsCompleted'] > 5) * 0.2 +
        (data['TrainingHours'] > 25) * 0.15 +
        data['HasCertification'] * 0.1 +
        (np.array(data['Education']) == 'Master') * 0.05 +
        (np.array(data['Education']) == 'PhD') * 0.1 +
        np.random.random(n_samples) * 0.2
    )
    
    data['HighPerformer'] = (performance_score > 0.6).astype(int)
    
    # Create DataFrame and split
    df = pd.DataFrame(data)
    train_size = int(0.8 * len(df))
    
    train_df = df[:train_size]
    test_df = df[train_size:].drop('HighPerformer', axis=1)
    
    # Save to CSV
    train_df.to_csv('employee_train.csv', index=False)
    test_df.to_csv('employee_test.csv', index=False)
    
    print("✅ Employee performance dataset generated")
    
    # Run pipeline
    pipeline = UniversalMLPipeline(problem_type='classification')
    pipeline.run_pipeline(
        train_path='employee_train.csv',
        test_path='employee_test.csv',
        target_column='HighPerformer'
    )
    
    return pipeline

def demo_5_titanic_if_available():
    """Demo 5: Titanic Dataset (if available)"""
    print("\n🚢 DEMO 5: TITANIC DATASET (IF AVAILABLE)")
    print("=" * 60)
    
    try:
        # Check if Titanic data exists
        if os.path.exists('../titanic/train.csv'):
            pipeline = UniversalMLPipeline(problem_type='classification')
            pipeline.run_pipeline(
                train_path='../titanic/train.csv',
                test_path='../titanic/test.csv',
                target_column='Survived',
                exclude_columns=['PassengerId', 'Name', 'Ticket', 'Cabin']
            )
            return pipeline
        else:
            print("⚠️ Titanic dataset not found, skipping this demo")
            return None
    except Exception as e:
        print(f"⚠️ Error with Titanic dataset: {e}")
        return None

def show_framework_info():
    """Show framework information"""
    print("🌟 UNIVERSAL ML FRAMEWORK INFORMATION")
    print("=" * 70)
    
    info = {
        'version': '1.0.0',
        'supported_problems': ['classification', 'regression'],
        'algorithms': {
            'classification': ['RandomForest', 'LogisticRegression', 'SVM'],
            'regression': ['RandomForest', 'LinearRegression', 'SVM']
        },
        'features': [
            'Auto feature detection',
            'Missing value handling', 
            'Categorical encoding',
            'Feature scaling',
            'Cross validation',
            'Hyperparameter tuning',
            'Model comparison',
            'Automatic predictions',
            'Model persistence'
        ]
    }
    
    print(f"Version: {info['version']}")
    print(f"Supported Problems: {', '.join(info['supported_problems'])}")
    
    print("\n🤖 Algorithms:")
    for problem, algos in info['algorithms'].items():
        print(f"  {problem.capitalize()}: {', '.join(algos)}")
    
    print("\n✨ Features:")
    for feature in info['features']:
        print(f"  ✅ {feature}")
    
    print("\n📊 What the framework does:")
    print("  1. Automatically detects feature types")
    print("  2. Handles missing values and preprocessing")
    print("  3. Compares multiple ML algorithms")
    print("  4. Performs cross-validation")
    print("  5. Tunes hyperparameters automatically")
    print("  6. Generates predictions on test set")
    print("  7. Saves trained model for production")

if __name__ == "__main__":
    print("🚀 UNIVERSAL ML FRAMEWORK - COMPLETE DEMO")
    print("=" * 80)
    
    # Show framework info
    show_framework_info()
    
    print("\n" + "=" * 80)
    print("RUNNING DEMOS...")
    print("=" * 80)
    
    results = {}
    
    try:
        # Demo 1: Customer Churn
        results['customer_churn'] = demo_1_customer_churn()
        
        # Demo 2: House Prices
        results['house_prices'] = demo_2_house_prices()
        
        # Demo 3: Sales Forecasting
        results['sales_forecasting'] = demo_3_sales_forecasting()
        
        # Demo 4: Custom Dataset
        results['employee_performance'] = demo_4_custom_dataset()
        
        # Demo 5: Titanic (if available)
        titanic_result = demo_5_titanic_if_available()
        if titanic_result:
            results['titanic'] = titanic_result
        
        print("\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print("📊 SUMMARY OF RESULTS:")
        print("-" * 40)
        for demo_name, pipeline in results.items():
            if pipeline:
                best_model = pipeline.best_model_name
                best_score = getattr(pipeline, 'best_score', pipeline.cv_results[best_model]['mean'])
                problem_type = pipeline.problem_type
                
                print(f"{demo_name:20}: {best_model:15} ({problem_type}) - Score: {best_score:.4f}")
        
        print("\n💡 FRAMEWORK BENEFITS DEMONSTRATED:")
        print("-" * 40)
        print("✅ Works with any dataset automatically")
        print("✅ Handles both classification and regression")
        print("✅ Compares multiple algorithms")
        print("✅ Provides reliable performance estimates")
        print("✅ Generates production-ready models")
        print("✅ Requires minimal code to get started")
        
        print("\n📁 FILES GENERATED:")
        print("-" * 40)
        print("• predictions.csv - Test set predictions")
        print("• best_model.pkl - Trained model")
        print("• model_info.json - Model metadata")
        print("• *_train.csv, *_test.csv - Generated datasets")
        
        print("\n🎯 NEXT STEPS:")
        print("-" * 40)
        print("1. Try with your own datasets")
        print("2. Customize the pipeline for specific needs")
        print("3. Use the saved models for production")
        print("4. Explore advanced features in examples/")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED - Thank you for trying Universal ML Framework!")
    print("=" * 80)