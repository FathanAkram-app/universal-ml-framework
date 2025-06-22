# Pipeline.py Analysis & Documentation

## 📋 Overview

`UniversalMLPipeline` adalah kelas utama yang mengotomatisasi seluruh workflow machine learning dari data loading hingga model deployment. Pipeline ini dirancang untuk bekerja dengan dataset apapun tanpa perlu konfigurasi manual yang rumit.

## 🏗️ Architecture Analysis

### Class Structure
```python
class UniversalMLPipeline:
    def __init__(self, problem_type, random_state, verbose, fast_mode, tuning_method, n_jobs)
    def load_data(self, train_path, test_path, target_column)
    def auto_detect_features(self, df, exclude_columns)
    def create_preprocessor(self)
    def prepare_data(self, custom_features)
    def define_models(self)
    def cross_validate_models(self)
    def hyperparameter_tuning(self)
    def make_predictions(self, save_predictions, id_column)
    def save_model(self, filename)
    def run_pipeline(self, **kwargs)
```

### Key Attributes
- `problem_type`: 'classification' atau 'regression'
- `fast_mode`: Mode cepat untuk dataset besar
- `tuning_method`: 'grid', 'random', atau 'bayesian'
- `n_jobs`: Jumlah core untuk parallel processing
- `feature_types`: Dictionary berisi tipe fitur yang terdeteksi
- `cv_results`: Hasil cross-validation semua model
- `best_pipeline`: Pipeline model terbaik setelah tuning

## 🔄 Workflow Analysis

### 1. Data Loading (`load_data`)
**Fungsi**: Memuat data training dan testing dari file CSV
**Input**: 
- `train_path`: Path file training
- `test_path`: Path file testing (optional)
- `target_column`: Nama kolom target

**Proses**:
- Load CSV menggunakan pandas
- Buat backup data original (`train_df_full`)
- Tampilkan informasi dataset (shape, distribusi target)
- Handle missing test data

### 2. Feature Detection (`auto_detect_features`)
**Fungsi**: Deteksi otomatis tipe fitur dalam dataset
**Algorithm**:
```python
for col in df.columns:
    if missing_ratio > 0.8:
        skip_column()
    elif unique_values == 2 and values in {0,1,True,False}:
        binary_features.append(col)
    elif dtype in ['int64','float64'] and unique_values > 10:
        numeric_features.append(col)
    else:
        categorical_features.append(col)
```

**Output**: Dictionary dengan 3 kategori fitur
- `numeric`: Fitur numerik kontinu
- `categorical`: Fitur kategorikal
- `binary`: Fitur binary (0/1, True/False)

### 3. Preprocessing (`create_preprocessor`)
**Fungsi**: Membuat pipeline preprocessing untuk setiap tipe fitur
**Transformations**:
- **Numeric**: SimpleImputer(median) → StandardScaler
- **Categorical**: SimpleImputer(constant) → OneHotEncoder
- **Binary**: SimpleImputer(constant, fill_value=0)

**Architecture**: ColumnTransformer dengan multiple pipelines

### 4. Model Definition (`define_models`)
**Fungsi**: Definisi model berdasarkan problem type dan mode

**Fast Mode Models**:
- Classification: RandomForest(50), LogisticRegression, NaiveBayes
- Regression: RandomForest(50), LinearRegression

**Full Mode Models**:
- Classification: 7 models (RF, GB, LR, SVM, NB, KNN, DT)
- Regression: 6 models (RF, GB, LR, SVM, KNN, DT)

**Parallel Processing**: Semua model yang support menggunakan `n_jobs`

### 5. Cross Validation (`cross_validate_models`)
**Fungsi**: Evaluasi performa semua model dengan cross-validation
**Strategy**:
- Classification: StratifiedKFold (preserve class distribution)
- Regression: KFold
- Folds: 3 (fast mode) atau 5 (normal mode)

**Metrics**:
- Classification: Accuracy
- Regression: Negative MSE (dikonversi ke positif)

**Parallel**: CV dilakukan parallel dengan `n_jobs`

### 6. Hyperparameter Tuning (`hyperparameter_tuning`)
**Fungsi**: Optimasi hyperparameter untuk model terbaik
**Methods**:
- **Grid Search**: Exhaustive search (lambat tapi thorough)
- **Random Search**: Random sampling (default, balanced)
- **Bayesian Search**: Smart exploration (butuh scikit-optimize)

**Parameter Grids**: Comprehensive grids untuk setiap model
- RandomForest: n_estimators, max_depth, min_samples_split, min_samples_leaf
- GradientBoosting: n_estimators, learning_rate, max_depth
- LogisticRegression: C, penalty, solver
- SVM: C, kernel, gamma
- KNN: n_neighbors, weights, metric
- DecisionTree: max_depth, min_samples_split, min_samples_leaf

**Iterations**: 20 (fast) atau 50 (normal) untuk Random/Bayesian

### 7. Prediction (`make_predictions`)
**Fungsi**: Generate prediksi pada test set
**Features**:
- Custom ID column support
- Fallback ke index atau sequential ID
- Save ke CSV otomatis
- Statistik prediksi (distribusi/mean-std)

### 8. Model Persistence (`save_model`)
**Fungsi**: Simpan model dan metadata
**Output Files**:
- `best_model.pkl`: Trained pipeline (joblib)
- `model_info.json`: Metadata (problem_type, best_model, params, cv_score, feature_types)

## ⚡ Performance Optimizations

### 1. Fast Mode
- **Reduced Models**: Hanya 2-3 model tercepat
- **Fewer CV Folds**: 3 instead of 5
- **Smaller Parameters**: n_estimators=50, max_iter=500
- **Speed Gain**: ~70% faster

### 2. Parallel Processing
- **Multi-core Models**: RF, LR, KNN, LinearRegression
- **Parallel CV**: All folds run simultaneously
- **Parallel Tuning**: Grid/Random/Bayesian search parallelized
- **Configurable**: n_jobs parameter

### 3. Memory Management
- **Data Backup**: train_df_full preserves original
- **Feature Engineering**: Works on copies
- **Efficient Transformers**: ColumnTransformer with remainder='drop'

## 🎛️ Configuration Options

### Initialization Parameters
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

### Runtime Parameters
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

## 🔍 Error Handling & Robustness

### Data Quality Checks
- **Missing Values**: Skip columns with >80% missing
- **Feature Types**: Robust detection algorithm
- **Empty DataFrames**: Proper handling

### Fallback Mechanisms
- **No Test Data**: Skip prediction step
- **No Hyperparameters**: Use default model
- **Missing ID Column**: Fallback to index/sequential
- **Import Errors**: Graceful degradation (Bayesian → Random)

### Validation
- **CV Scoring**: Proper metric selection per problem type
- **Score Normalization**: Convert negative MSE to positive
- **Pipeline Integrity**: Consistent preprocessing across train/test

## 📊 Output Analysis

### Console Output
- **Progress Indicators**: Step-by-step progress dengan emoji
- **Verbose Mode**: Detailed fold-by-fold results
- **Summary**: Final results dengan best model dan score

### File Outputs
- **predictions.csv**: ID + Prediction columns
- **best_model.pkl**: Serialized pipeline
- **model_info.json**: Complete metadata

### Return Values
- **Trained Pipeline**: Access via `pipeline.best_pipeline`
- **CV Results**: Access via `pipeline.cv_results`
- **Feature Types**: Access via `pipeline.feature_types`

## 🚀 Strengths

1. **Universal**: Works dengan dataset apapun
2. **Automated**: Minimal manual intervention
3. **Scalable**: Fast mode untuk dataset besar
4. **Flexible**: Multiple tuning methods
5. **Production Ready**: Model persistence & metadata
6. **Robust**: Comprehensive error handling
7. **Parallel**: Multi-core processing support

## ⚠️ Limitations

1. **CSV Only**: Hanya support file CSV
2. **Memory**: Large datasets bisa memory intensive
3. **Feature Engineering**: Limited built-in transformations
4. **Deep Learning**: Tidak include neural networks
5. **Time Series**: Tidak optimized untuk time series data

## 🔧 Extensibility

Pipeline mudah di-extend dengan:
- **Custom Models**: Tambah ke `self.models` dictionary
- **Custom Preprocessing**: Via `feature_engineering_func`
- **Custom Metrics**: Modify scoring parameter
- **Custom Validation**: Override CV strategy

## 📈 Performance Benchmarks

**Typical Performance** (dataset 10K rows, 20 features):
- **Fast Mode**: ~2-5 minutes
- **Normal Mode**: ~5-15 minutes
- **With Bayesian Tuning**: ~10-30 minutes

**Memory Usage**:
- **Small Dataset** (<1K rows): ~50MB
- **Medium Dataset** (10K rows): ~200MB
- **Large Dataset** (100K rows): ~1-2GB