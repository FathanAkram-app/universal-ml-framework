# DATA GENERATOR
# Generate synthetic datasets for testing and demonstration

import pandas as pd
import numpy as np

class DataGenerator:
    """Generate synthetic datasets for various ML problems"""
    
    @staticmethod
    def generate_house_prices(n_samples=1000, save_to_csv=True):
        """Generate synthetic house prices dataset"""
        np.random.seed(42)
        
        data = {
            'LotArea': np.random.normal(10000, 3000, n_samples),
            'YearBuilt': np.random.randint(1950, 2020, n_samples),
            'BedroomAbvGr': np.random.randint(1, 6, n_samples),
            'BathroomAbvGr': np.random.randint(1, 4, n_samples),
            'GarageArea': np.random.normal(500, 200, n_samples),
            'Neighborhood': np.random.choice(['Downtown', 'Suburb', 'Rural'], n_samples),
            'HouseStyle': np.random.choice(['1Story', '2Story', 'Split'], n_samples),
            'HasPool': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'HasGarage': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        }
        
        # Create realistic price
        price = (
            data['LotArea'] * 0.01 +
            (2020 - data['YearBuilt']) * -100 +
            data['BedroomAbvGr'] * 15000 +
            data['BathroomAbvGr'] * 10000 +
            data['GarageArea'] * 50 +
            data['HasPool'] * 25000 +
            data['HasGarage'] * 15000 +
            np.random.normal(0, 20000, n_samples)
        )
        
        data['SalePrice'] = np.maximum(price, 50000)
        
        df = pd.DataFrame(data)
        
        if save_to_csv:
            train_size = int(0.8 * len(df))
            train_df = df[:train_size].copy()
            test_df = df[train_size:].drop('SalePrice', axis=1)
            
            train_df.to_csv('data/house_train.csv', index=False)
            test_df.to_csv('data/house_test.csv', index=False)
            
            print("✅ House prices dataset generated")
            print(f"   Train: {train_df.shape}")
            print(f"   Test: {test_df.shape}")
        
        return df
    
    @staticmethod
    def generate_customer_churn(n_samples=800, save_to_csv=True):
        """Generate synthetic customer churn dataset"""
        np.random.seed(123)
        
        data = {
            'Age': np.random.randint(18, 80, n_samples),
            'MonthlyCharges': np.random.normal(65, 20, n_samples),
            'TotalCharges': np.random.normal(2000, 1500, n_samples),
            'Tenure': np.random.randint(1, 72, n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'HasPhoneService': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'HasMultipleLines': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        }
        
        # Create churn based on realistic factors
        churn_prob = (
            (data['MonthlyCharges'] > 80) * 0.3 +
            (data['Tenure'] < 12) * 0.4 +
            (np.array(data['Contract']) == 'Month-to-month') * 0.3 +
            data['SeniorCitizen'] * 0.2 +
            np.random.random(n_samples) * 0.3
        )
        
        data['Churn'] = (churn_prob > 0.5).astype(int)
        
        df = pd.DataFrame(data)
        
        if save_to_csv:
            train_size = int(0.8 * len(df))
            train_df = df[:train_size].copy()
            test_df = df[train_size:].drop('Churn', axis=1)
            
            train_df.to_csv('data/customer_train.csv', index=False)
            test_df.to_csv('data/customer_test.csv', index=False)
            
            print("✅ Customer churn dataset generated")
            print(f"   Train: {train_df.shape}")
            print(f"   Test: {test_df.shape}")
        
        return df
    
    @staticmethod
    def generate_sales_forecasting(n_samples=600, save_to_csv=True):
        """Generate synthetic sales forecasting dataset"""
        np.random.seed(456)
        
        data = {
            'Month': np.random.randint(1, 13, n_samples),
            'DayOfWeek': np.random.randint(1, 8, n_samples),
            'Temperature': np.random.normal(20, 10, n_samples),
            'Humidity': np.random.normal(60, 15, n_samples),
            'WindSpeed': np.random.normal(10, 5, n_samples),
            'Holiday': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'Promotion': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'StoreType': np.random.choice(['Mall', 'Street', 'Online'], n_samples),
            'CompetitorDistance': np.random.normal(2000, 1000, n_samples)
        }
        
        # Create sales based on realistic factors
        sales = (
            1000 +
            (data['Month'] == 12) * 500 +
            (np.array(data['DayOfWeek']).isin([6, 7])) * 300 +
            data['Temperature'] * 10 +
            data['Holiday'] * 800 +
            data['Promotion'] * 400 +
            -data['CompetitorDistance'] * 0.1 +
            np.random.normal(0, 200, n_samples)
        )
        
        data['Sales'] = np.maximum(sales, 100)
        
        df = pd.DataFrame(data)
        
        if save_to_csv:
            train_size = int(0.8 * len(df))
            train_df = df[:train_size].copy()
            test_df = df[train_size:].drop('Sales', axis=1)
            
            train_df.to_csv('data/sales_train.csv', index=False)
            test_df.to_csv('data/sales_test.csv', index=False)
            
            print("✅ Sales forecasting dataset generated")
            print(f"   Train: {train_df.shape}")
            print(f"   Test: {test_df.shape}")
        
        return df
    
    @staticmethod
    def generate_all_datasets():
        """Generate all synthetic datasets"""
        print("🔄 Generating all synthetic datasets...")
        print("-" * 40)
        
        DataGenerator.generate_house_prices()
        DataGenerator.generate_customer_churn()
        DataGenerator.generate_sales_forecasting()
        
        print("\n✅ All datasets generated successfully!")
        print("Ready to use with the pipeline framework.")