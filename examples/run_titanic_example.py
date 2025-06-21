# CONTOH PENGGUNAAN FRAMEWORK DENGAN DATASET TITANIC
# Demonstrasi framework dengan dataset yang sudah familiar

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.pipeline import UniversalMLPipeline

def run_titanic_example():
    """Jalankan contoh dengan dataset Titanic"""
    print("🚢 TITANIC SURVIVAL PREDICTION - UNIVERSAL ML FRAMEWORK")
    print("=" * 70)
    
    # Inisialisasi pipeline
    pipeline = UniversalMLPipeline(problem_type='classification')
    
    # Jalankan pipeline lengkap
    pipeline.run_pipeline(
        train_path='data/titanic_train.csv',
        test_path='data/titanic_test.csv',
        target_column='Survived',
        exclude_columns=['PassengerId', 'Name', 'Ticket', 'Cabin']
    )
    
    print("\n🎯 HASIL PREDIKSI TITANIC:")
    print("-" * 40)
    print(f"✅ Model terbaik: {pipeline.best_model_name}")
    print(f"✅ Akurasi CV: {getattr(pipeline, 'best_score', pipeline.cv_results[pipeline.best_model_name]['mean']):.4f}")
    print(f"✅ File output:")
    print(f"   - predictions.csv (prediksi survival)")
    print(f"   - best_model.pkl (model terlatih)")
    print(f"   - model_info.json (metadata model)")
    
    return pipeline

if __name__ == "__main__":
    try:
        pipeline = run_titanic_example()
        print("\n🎉 TITANIC EXAMPLE BERHASIL!")
        print("Framework siap digunakan untuk dataset lainnya.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Pastikan file train.csv dan test.csv ada di folder data/")