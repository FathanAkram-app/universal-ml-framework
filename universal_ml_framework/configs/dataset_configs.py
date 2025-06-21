# DATASET CONFIGURATIONS
# Pre-configured templates for common datasets

class PipelineConfigs:
    """Template konfigurasi untuk berbagai jenis dataset"""
    
    @staticmethod
    def titanic_config():
        return {
            'problem_type': 'classification',
            'train_path': 'data/titanic_train.csv',
            'test_path': 'data/titanic_test.csv',
            'target_column': 'Survived',
            'exclude_columns': ['PassengerId', 'Name', 'Ticket', 'Cabin'],
            'description': 'Prediksi survival penumpang Titanic'
        }
    
    @staticmethod
    def house_prices_config():
        return {
            'problem_type': 'regression',
            'train_path': 'data/house_train.csv',
            'test_path': 'data/house_test.csv',
            'target_column': 'SalePrice',
            'exclude_columns': ['Id'],
            'description': 'Prediksi harga rumah'
        }
    
    @staticmethod
    def customer_churn_config():
        return {
            'problem_type': 'classification',
            'train_path': 'data/customer_train.csv',
            'test_path': 'data/customer_test.csv',
            'target_column': 'Churn',
            'exclude_columns': ['CustomerID'],
            'description': 'Prediksi customer churn'
        }
    
    @staticmethod
    def sales_forecasting_config():
        return {
            'problem_type': 'regression',
            'train_path': 'data/sales_train.csv',
            'test_path': 'data/sales_test.csv',
            'target_column': 'Sales',
            'exclude_columns': ['Date', 'StoreID'],
            'description': 'Prediksi penjualan'
        }
    
    @staticmethod
    def fraud_detection_config():
        return {
            'problem_type': 'classification',
            'train_path': 'data/fraud_train.csv',
            'test_path': 'data/fraud_test.csv',
            'target_column': 'IsFraud',
            'exclude_columns': ['TransactionID', 'UserID'],
            'description': 'Deteksi transaksi fraud'
        }
    
    @staticmethod
    def get_all_configs():
        """Get all available configurations"""
        return {
            'titanic': PipelineConfigs.titanic_config(),
            'house_prices': PipelineConfigs.house_prices_config(),
            'customer_churn': PipelineConfigs.customer_churn_config(),
            'sales_forecasting': PipelineConfigs.sales_forecasting_config(),
            'fraud_detection': PipelineConfigs.fraud_detection_config()
        }