# QUICK START GUIDE - UNIVERSAL ML FRAMEWORK
# Panduan cepat menggunakan framework untuk berbagai dataset

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ..core.pipeline import UniversalMLPipeline
from ..utils.data_generator import DataGenerator
from ..utils.helpers import quick_classification_pipeline, quick_regression_pipeline

def show_framework_info():
    """Tampilkan informasi framework"""
    print("🌟 UNIVERSAL ML FRAMEWORK - QUICK START")
    print("=" * 60)
    print("Framework untuk machine learning yang bekerja dengan dataset apapun!")
    print("\n✨ Fitur Utama:")
    print("  ✅ Auto feature detection")
    print("  ✅ Model comparison (RF, LR, SVM)")
    print("  ✅ Cross validation")
    print("  ✅ Hyperparameter tuning")
    print("  ✅ Production ready output")
    
    print("\n🎯 Supported Problems:")
    print("  📊 Classification (binary/multiclass)")
    print("  📈 Regression (continuous prediction)")

def example_1_basic_usage():
    """Contoh 1: Penggunaan dasar"""
    print("\n" + "="*60)
    print("CONTOH 1: PENGGUNAAN DASAR")
    print("="*60)
    
    print("📝 Kode yang dibutuhkan:")
    print("""
from core.pipeline import UniversalMLPipeline

# Classification
pipeline = UniversalMLPipeline(problem_type='classification')
pipeline.run_pipeline('data.csv', 'target_column', 'test.csv')

# Regression  
pipeline = UniversalMLPipeline(problem_type='regression')
pipeline.run_pipeline('data.csv', 'price_column', 'test.csv')
    """)

def example_2_with_synthetic_data():
    """Contoh 2: Dengan synthetic data"""
    print("\n" + "="*60)
    print("CONTOH 2: DENGAN SYNTHETIC DATA")
    print("="*60)
    
    # Generate customer churn data
    print("🔄 Generating customer churn dataset...")
    DataGenerator.generate_customer_churn()
    
    # Run pipeline
    print("🚀 Running classification pipeline...")
    pipeline = quick_classification_pipeline(
        train_path='customer_train.csv',
        target_column='Churn',
        test_path='data/customer_test.csv'
    )
    
    print(f"✅ Best model: {pipeline.best_model_name}")
    print(f"✅ CV Score: {getattr(pipeline, 'best_score', pipeline.cv_results[pipeline.best_model_name]['mean']):.4f}")

def example_3_titanic():
    """Contoh 3: Dataset Titanic"""
    print("\n" + "="*60)
    print("CONTOH 3: DATASET TITANIC")
    print("="*60)
    
    if os.path.exists('data/train.csv'):
        print("🚢 Running Titanic survival prediction...")
        
        pipeline = UniversalMLPipeline(problem_type='classification')
        pipeline.run_pipeline(
            train_path='data/train.csv',
            test_path='data/test.csv',
            target_column='Survived',
            exclude_columns=['PassengerId', 'Name', 'Ticket', 'Cabin']
        )
        
        print(f"✅ Best model: {pipeline.best_model_name}")
        print(f"✅ CV Score: {getattr(pipeline, 'best_score', pipeline.cv_results[pipeline.best_model_name]['mean']):.4f}")
    else:
        print("⚠️ Titanic dataset not found in data/ folder")

def show_output_files():
    """Tampilkan file output yang dihasilkan"""
    print("\n" + "="*60)
    print("OUTPUT FILES YANG DIHASILKAN")
    print("="*60)
    
    print("📁 Framework otomatis menghasilkan:")
    print("  📄 predictions.csv - Prediksi pada test set")
    print("  🤖 best_model.pkl - Model terlatih (joblib format)")
    print("  📋 model_info.json - Metadata dan konfigurasi model")
    
    print("\n💡 Cara menggunakan model yang tersimpan:")
    print("""
import joblib
import pandas as pd

# Load model
model = joblib.load('best_model.pkl')

# Load new data
new_data = pd.read_csv('new_data.csv')

# Make predictions
predictions = model.predict(new_data)
    """)

def show_customization_options():
    """Tampilkan opsi kustomisasi"""
    print("\n" + "="*60)
    print("OPSI KUSTOMISASI")
    print("="*60)
    
    print("🔧 Custom feature types:")
    print("""
pipeline.feature_types = {
    'numeric': ['age', 'income', 'score'],
    'categorical': ['city', 'category'],
    'binary': ['has_feature']
}
    """)
    
    print("🤖 Custom models:")
    print("""
from sklearn.ensemble import GradientBoostingClassifier
pipeline.models['GradientBoosting'] = GradientBoostingClassifier()
    """)

def main():
    """Main function"""
    show_framework_info()
    example_1_basic_usage()
    
    try:
        example_2_with_synthetic_data()
    except Exception as e:
        print(f"⚠️ Error in synthetic data example: {e}")
    
    try:
        example_3_titanic()
    except Exception as e:
        print(f"⚠️ Error in Titanic example: {e}")
    
    show_output_files()
    show_customization_options()
    
    print("\n" + "="*60)
    print("🎉 QUICK START COMPLETED!")
    print("="*60)
    print("Framework siap digunakan untuk dataset Anda!")
    print("\n📚 Next Steps:")
    print("  1. Siapkan dataset Anda (CSV format)")
    print("  2. Tentukan problem type (classification/regression)")
    print("  3. Jalankan pipeline dengan 3 baris kode")
    print("  4. Gunakan model yang tersimpan untuk production")
    
    print("\n📞 Bantuan:")
    print("  - Lihat examples/ untuk contoh lebih lanjut")
    print("  - Baca README.md untuk dokumentasi lengkap")
    print("  - Cek demo.py untuk demonstrasi komprehensif")

if __name__ == "__main__":
    main()