# BASIC USAGE EXAMPLES
# Demonstrasi penggunaan dasar Universal ML Framework

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import UniversalMLPipeline
from utils.helpers import quick_classification_pipeline, quick_regression_pipeline, run_pipeline_with_config, list_available_configs
from utils.data_generator import DataGenerator

def example_1_quick_setup():
    """Example 1: Quick setup untuk classification"""
    print("📋 EXAMPLE 1: QUICK CLASSIFICATION SETUP")
    print("=" * 50)
    
    # Generate sample data
    DataGenerator.generate_customer_churn()
    
    # Quick classification pipeline
    pipeline = quick_classification_pipeline(
        train_path='../data/customer_train.csv',
        target_column='Churn',
        test_path='../data/customer_test.csv'
    )
    
    return pipeline

def example_2_custom_pipeline():
    """Example 2: Custom pipeline setup"""
    print("\n📋 EXAMPLE 2: CUSTOM PIPELINE SETUP")
    print("=" * 50)
    
    # Generate sample data
    DataGenerator.generate_house_prices()
    
    # Custom pipeline
    pipeline = UniversalMLPipeline(problem_type='regression')
    pipeline.run_pipeline(
        train_path='../data/house_train.csv',
        target_column='SalePrice',
        test_path='../data/house_test.csv',
        exclude_columns=['Id']  # if exists
    )
    
    return pipeline

def example_3_config_based():
    """Example 3: Using predefined configurations"""
    print("\n📋 EXAMPLE 3: CONFIG-BASED SETUP")
    print("=" * 50)
    
    # List available configs
    list_available_configs()
    
    # Generate data for customer churn
    DataGenerator.generate_customer_churn()
    
    # Run with config
    pipeline = run_pipeline_with_config('customer_churn')
    
    return pipeline

def example_4_multiple_datasets():
    """Example 4: Multiple datasets comparison"""
    print("\n📋 EXAMPLE 4: MULTIPLE DATASETS")
    print("=" * 50)
    
    # Generate all datasets
    DataGenerator.generate_all_datasets()
    
    results = {}
    
    # House prices (regression)
    print("\n🏠 HOUSE PRICES REGRESSION:")
    house_pipeline = quick_regression_pipeline(
        train_path='../data/house_train.csv',
        target_column='SalePrice',
        test_path='../data/house_test.csv'
    )
    results['house_prices'] = house_pipeline
    
    # Customer churn (classification)
    print("\n👥 CUSTOMER CHURN CLASSIFICATION:")
    churn_pipeline = quick_classification_pipeline(
        train_path='../data/customer_train.csv',
        target_column='Churn',
        test_path='../data/customer_test.csv'
    )
    results['customer_churn'] = churn_pipeline
    
    # Sales forecasting (regression)
    print("\n📈 SALES FORECASTING REGRESSION:")
    sales_pipeline = quick_regression_pipeline(
        train_path='../data/sales_train.csv',
        target_column='Sales',
        test_path='../data/sales_test.csv'
    )
    results['sales_forecasting'] = sales_pipeline
    
    return results

if __name__ == "__main__":
    print("🌟 UNIVERSAL ML FRAMEWORK - BASIC USAGE EXAMPLES")
    print("=" * 70)
    
    # Run examples
    try:
        # Example 1: Quick setup
        pipeline1 = example_1_quick_setup()
        
        # Example 2: Custom pipeline
        pipeline2 = example_2_custom_pipeline()
        
        # Example 3: Config-based
        pipeline3 = example_3_config_based()
        
        # Example 4: Multiple datasets
        results = example_4_multiple_datasets()
        
        print("\n🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("✅ Quick Classification Setup")
        print("✅ Custom Pipeline Setup")
        print("✅ Config-based Setup")
        print("✅ Multiple Datasets Comparison")
        
    except Exception as e:
        print(f"❌ Error in examples: {e}")
        print("Make sure all dependencies are installed and data files are available.")