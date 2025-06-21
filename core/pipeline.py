# UNIVERSAL ML PIPELINE - CORE MODULE
# Main pipeline class for universal machine learning

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
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

class UniversalMLPipeline:
    """Universal ML Pipeline untuk Classification dan Regression"""
    
    def __init__(self, problem_type='classification', random_state=42):
        self.problem_type = problem_type
        self.random_state = random_state
        self.preprocessor = None
        self.models = {}
        self.best_pipeline = None
        self.cv_results = {}
        self.feature_types = {}
        
    def load_data(self, train_path, test_path=None, target_column=None):
        """Load data dari file CSV"""
        print(f"📂 Loading data...")
        
        self.train_df = pd.read_csv(train_path)
        if test_path:
            self.test_df = pd.read_csv(test_path)
        else:
            self.test_df = None
            
        self.target_column = target_column
        
        print(f"✅ Training data: {self.train_df.shape}")
        if self.test_df is not None:
            print(f"✅ Test data: {self.test_df.shape}")
        
        if target_column:
            if self.problem_type == 'classification':
                print(f"✅ Target distribution: {self.train_df[target_column].value_counts().to_dict()}")
            else:
                print(f"✅ Target stats: mean={self.train_df[target_column].mean():.2f}, std={self.train_df[target_column].std():.2f}")
    
    def auto_detect_features(self, df, exclude_columns=None):
        """Automatically detect feature types"""
        print("🔍 Auto-detecting feature types...")
        
        if exclude_columns is None:
            exclude_columns = []
            
        numeric_features = []
        categorical_features = []
        binary_features = []
        
        for col in df.columns:
            if col in exclude_columns:
                continue
                
            # Skip if too many missing values
            if df[col].isnull().sum() / len(df) > 0.8:
                print(f"⚠️ Skipping {col} (too many missing values)")
                continue
                
            # Binary features (0/1 or True/False)
            if df[col].nunique() == 2 and set(df[col].dropna().unique()).issubset({0, 1, True, False}):
                binary_features.append(col)
            # Numeric features
            elif df[col].dtype in ['int64', 'float64'] and df[col].nunique() > 10:
                numeric_features.append(col)
            # Categorical features
            elif df[col].dtype == 'object' or df[col].nunique() <= 10:
                categorical_features.append(col)
        
        self.feature_types = {
            'numeric': numeric_features,
            'categorical': categorical_features,
            'binary': binary_features
        }
        
        print(f"✅ Numeric features ({len(numeric_features)}): {numeric_features}")
        print(f"✅ Categorical features ({len(categorical_features)}): {categorical_features}")
        print(f"✅ Binary features ({len(binary_features)}): {binary_features}")
        
        return self.feature_types
    
    def create_preprocessor(self):
        """Create preprocessing pipeline"""
        print("⚙️ Creating preprocessor...")
        
        transformers = []
        
        if self.feature_types['numeric']:
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_transformer, self.feature_types['numeric']))
        
        if self.feature_types['categorical']:
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, self.feature_types['categorical']))
        
        if self.feature_types['binary']:
            binary_transformer = SimpleImputer(strategy='constant', fill_value=0)
            transformers.append(('bin', binary_transformer, self.feature_types['binary']))
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        print("✅ Preprocessor created")
    
    def prepare_data(self, custom_features=None):
        """Prepare data for training"""
        print("🔄 Preparing data...")
        
        if custom_features:
            feature_columns = custom_features
        else:
            feature_columns = (self.feature_types['numeric'] + 
                             self.feature_types['categorical'] + 
                             self.feature_types['binary'])
        
        self.X = self.train_df[feature_columns]
        self.y = self.train_df[self.target_column]
        
        if self.test_df is not None:
            self.X_test = self.test_df[feature_columns]
        
        print(f"✅ Features: {self.X.shape[1]} columns, {self.X.shape[0]} rows")
    
    def define_models(self):
        """Define models based on problem type"""
        print("🤖 Defining models...")
        
        if self.problem_type == 'classification':
            self.models = {
                'RandomForest': RandomForestClassifier(random_state=self.random_state),
                'LogisticRegression': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'SVM': SVC(random_state=self.random_state, probability=True)
            }
        else:
            self.models = {
                'RandomForest': RandomForestRegressor(random_state=self.random_state),
                'LinearRegression': LinearRegression(),
                'SVM': SVR()
            }
        
        print(f"✅ Models: {list(self.models.keys())}")
    
    def cross_validate_models(self):
        """Cross validate all models"""
        print("📊 Cross validating models...")
        
        if self.problem_type == 'classification':
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scoring = 'neg_mean_squared_error'
        
        for model_name, model in self.models.items():
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', model)
            ])
            
            cv_scores = cross_val_score(pipeline, self.X, self.y, cv=cv, scoring=scoring)
            
            if self.problem_type == 'regression':
                cv_scores = -cv_scores
            
            self.cv_results[model_name] = {
                'pipeline': pipeline,
                'scores': cv_scores,
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
            
            metric_name = 'Accuracy' if self.problem_type == 'classification' else 'MSE'
            print(f"{model_name:18}: {cv_scores.mean():.4f} (±{cv_scores.std():.4f}) {metric_name}")
        
        if self.problem_type == 'classification':
            self.best_model_name = max(self.cv_results.keys(), key=lambda x: self.cv_results[x]['mean'])
        else:
            self.best_model_name = min(self.cv_results.keys(), key=lambda x: self.cv_results[x]['mean'])
        
        print(f"\n🏆 Best model: {self.best_model_name}")
    
    def hyperparameter_tuning(self):
        """Hyperparameter tuning for best model"""
        print(f"🎯 Hyperparameter tuning for {self.best_model_name}...")
        
        param_grids = self._get_param_grids()
        best_pipeline = self.cv_results[self.best_model_name]['pipeline']
        param_grid = param_grids.get(self.best_model_name, {})
        
        if param_grid:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state) if self.problem_type == 'classification' else KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scoring = 'accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error'
            
            grid_search = GridSearchCV(
                best_pipeline,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1
            )
            
            grid_search.fit(self.X, self.y)
            
            self.best_pipeline = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
            
            print(f"✅ Best parameters: {self.best_params}")
            print(f"✅ Best CV score: {self.best_score:.4f}")
        else:
            self.best_pipeline = best_pipeline
            self.best_pipeline.fit(self.X, self.y)
            print("✅ No hyperparameters to tune")
    
    def _get_param_grids(self):
        """Get parameter grids for hyperparameter tuning"""
        if self.problem_type == 'classification':
            return {
                'RandomForest': {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [5, 10, None],
                    'model__min_samples_split': [2, 5]
                },
                'LogisticRegression': {
                    'model__C': [0.1, 1, 10],
                    'model__penalty': ['l1', 'l2'],
                    'model__solver': ['liblinear']
                },
                'SVM': {
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['rbf', 'linear']
                }
            }
        else:
            return {
                'RandomForest': {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [5, 10, None]
                },
                'LinearRegression': {},
                'SVM': {
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['rbf', 'linear']
                }
            }
    
    def make_predictions(self, save_predictions=True):
        """Make predictions on test set"""
        if self.test_df is None:
            print("⚠️ No test data available")
            return None
            
        print("🔮 Making predictions...")
        
        predictions = self.best_pipeline.predict(self.X_test)
        
        submission = pd.DataFrame({
            'ID': range(len(predictions)),
            'Prediction': predictions
        })
        
        if save_predictions:
            submission.to_csv('predictions.csv', index=False)
            print(f"✅ Predictions saved to predictions.csv")
        
        if self.problem_type == 'classification':
            print(f"✅ Prediction distribution: {pd.Series(predictions).value_counts().to_dict()}")
        else:
            print(f"✅ Prediction stats: mean={predictions.mean():.2f}, std={predictions.std():.2f}")
        
        return predictions
    
    def save_model(self, filename='best_model.pkl'):
        """Save trained model"""
        print("💾 Saving model...")
        
        joblib.dump(self.best_pipeline, filename)
        
        model_info = {
            'problem_type': self.problem_type,
            'best_model': self.best_model_name,
            'best_params': getattr(self, 'best_params', {}),
            'cv_score': getattr(self, 'best_score', self.cv_results[self.best_model_name]['mean']),
            'feature_types': self.feature_types
        }
        
        with open('model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Model saved as {filename}")
        print("✅ Model info saved as model_info.json")
    
    def run_pipeline(self, train_path, target_column, test_path=None, 
                    problem_type='classification', exclude_columns=None, custom_features=None):
        """Run complete pipeline"""
        print("🚀 STARTING UNIVERSAL ML PIPELINE")
        print("=" * 60)
        
        self.problem_type = problem_type
        
        self.load_data(train_path, test_path, target_column)
        
        exclude_cols = [target_column] + (exclude_columns or [])
        self.auto_detect_features(self.train_df, exclude_cols)
        
        self.create_preprocessor()
        self.prepare_data(custom_features)
        self.define_models()
        self.cross_validate_models()
        self.hyperparameter_tuning()
        
        if self.test_df is not None:
            self.make_predictions()
        
        self.save_model()
        
        print("\n🎉 PIPELINE COMPLETED!")
        print("=" * 60)
        print(f"✅ Problem Type: {self.problem_type}")
        print(f"✅ Best Model: {self.best_model_name}")
        print(f"✅ Best Score: {getattr(self, 'best_score', self.cv_results[self.best_model_name]['mean']):.4f}")
        print("=" * 60)