# UNIVERSAL ML PIPELINE - CORE MODULE
# Main pipeline class for universal machine learning

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer # Keep this import
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
try:
    from skopt import BayesSearchCV
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

class UniversalMLPipeline:
    """Universal ML Pipeline untuk Classification dan Regression"""
    
    def __init__(self, problem_type='classification', random_state=42, verbose=False, fast_mode=False, tuning_method='random', n_jobs=-1):
        self.problem_type = problem_type
        self.random_state = random_state
        self.verbose = verbose
        self.fast_mode = fast_mode
        self.tuning_method = tuning_method  # 'grid', 'random', 'bayesian'
        self.n_jobs = n_jobs  # -1 for all cores, 1 for single core
        self.preprocessor = None
        self.models = {}
        self.best_pipeline = None
        self.cv_results = {}
        self.feature_types = {}
        
    def load_data(self, train_path, test_path=None, target_column=None):
        """Load data dari file CSV"""
        print(f"📂 Loading data...")
        
        self.train_df = pd.read_csv(train_path)
        self.train_df_full = self.train_df.copy()
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
        
        if self.fast_mode:
            # Fast models for large datasets
            if self.problem_type == 'classification':
                self.models = {
                    'RandomForest': RandomForestClassifier(random_state=self.random_state, n_estimators=50, n_jobs=self.n_jobs),
                    'LogisticRegression': LogisticRegression(random_state=self.random_state, max_iter=500, n_jobs=self.n_jobs),
                    'NaiveBayes': GaussianNB()
                }
            else:
                self.models = {
                    'RandomForest': RandomForestRegressor(random_state=self.random_state, n_estimators=50, n_jobs=self.n_jobs),
                    'LinearRegression': LinearRegression(n_jobs=self.n_jobs)
                }
        else:
            # Full model set
            if self.problem_type == 'classification':
                self.models = {
                    'RandomForest': RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs),
                    'GradientBoosting': GradientBoostingClassifier(random_state=self.random_state),
                    'LogisticRegression': LogisticRegression(random_state=self.random_state, max_iter=1000, n_jobs=self.n_jobs),
                    'SVM': SVC(random_state=self.random_state, probability=True),
                    'NaiveBayes': GaussianNB(),
                    'KNN': KNeighborsClassifier(n_jobs=self.n_jobs),
                    'DecisionTree': DecisionTreeClassifier(random_state=self.random_state)
                }
            else:
                self.models = {
                    'RandomForest': RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs),
                    'GradientBoosting': GradientBoostingRegressor(random_state=self.random_state),
                    'LinearRegression': LinearRegression(n_jobs=self.n_jobs),
                    'SVM': SVR(),
                    'KNN': KNeighborsRegressor(n_jobs=self.n_jobs),
                    'DecisionTree': DecisionTreeRegressor(random_state=self.random_state)
                }
        
        print(f"✅ Models: {list(self.models.keys())}")
    
    def cross_validate_models(self):
        """Cross validate all models"""
        print("📊 Cross validating models...")
        
        if self.problem_type == 'classification':
            cv_splits = 3 if self.fast_mode else 5
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
            scoring = 'accuracy'
        else:
            cv_splits = 3 if self.fast_mode else 5
            cv = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
            scoring = 'neg_mean_squared_error'
        
        for i, (model_name, model) in enumerate(self.models.items(), 1):
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', model)
            ])
            
            if self.verbose:
                print(f"\n[{i}/{len(self.models)}] 🔄 Training {model_name}...")
            
            cv_scores = cross_val_score(pipeline, self.X, self.y, cv=cv, scoring=scoring, n_jobs=self.n_jobs)
            
            if self.problem_type == 'regression':
                cv_scores = -cv_scores
            
            self.cv_results[model_name] = {
                'pipeline': pipeline,
                'scores': cv_scores,
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
            
            metric_name = 'Accuracy' if self.problem_type == 'classification' else 'MSE'
            
            if self.verbose:
                for j, score in enumerate(cv_scores, 1):
                    print(f"  Fold {j}/{cv_splits}: {score:.4f}")
                print(f"  ✅ {model_name} completed - Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            else:
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
        best_pipeline = self.cv_results[self.best_model_name]['pipeline'] # Get the pipeline from CV results
        param_grid = param_grids.get(self.best_model_name, {})
        
        if param_grid:
            cv_splits = 3 if self.fast_mode else 5
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state) if self.problem_type == 'classification' else KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
            scoring = 'accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error'
            
            # Choose tuning method
            if self.tuning_method == 'bayesian' and BAYESIAN_AVAILABLE:
                grid_search = BayesSearchCV(
                    best_pipeline,
                    param_grid,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=self.n_jobs,
                    n_iter=20 if self.fast_mode else 50,
                    random_state=self.random_state
                )
            elif self.tuning_method == 'grid':
                grid_search = GridSearchCV(
                    best_pipeline,
                    param_grid,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=self.n_jobs,
                    verbose=1 if self.verbose else 0
                )
            else:  # random (default)
                grid_search = RandomizedSearchCV(
                    best_pipeline,
                    param_grid,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=self.n_jobs,
                    n_iter=20 if self.fast_mode else 50,
                    random_state=self.random_state,
                    verbose=1 if self.verbose else 0
                )
            
            grid_search.fit(self.X, self.y)
            
            self.best_pipeline = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.best_score = abs(grid_search.best_score_) if self.problem_type == 'regression' else grid_search.best_score_
            
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
                    'model__n_estimators': [50, 100, 200, 300],
                    'model__max_depth': [3, 5, 10, 15, None],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                },
                'GradientBoosting': {
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.01, 0.1, 0.2, 0.3],
                    'model__max_depth': [3, 5, 7]
                },
                'LogisticRegression': {
                    'model__C': [0.01, 0.1, 1, 10, 100],
                    'model__penalty': ['l1', 'l2'],
                    'model__solver': ['liblinear', 'saga']
                },
                'SVM': {
                    'model__C': [0.1, 1, 10, 100],
                    'model__kernel': ['rbf', 'linear'],
                    'model__gamma': ['scale', 'auto', 0.001, 0.01]
                },
                'KNN': {
                    'model__n_neighbors': [3, 5, 7, 9, 11],
                    'model__weights': ['uniform', 'distance'],
                    'model__metric': ['euclidean', 'manhattan']
                },
                'DecisionTree': {
                    'model__max_depth': [3, 5, 10, 15, None],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                }
            }
        else:
            return {
                'RandomForest': {
                    'model__n_estimators': [50, 100, 200, 300],
                    'model__max_depth': [3, 5, 10, 15, None],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                },
                'GradientBoosting': {
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.01, 0.1, 0.2, 0.3],
                    'model__max_depth': [3, 5, 7]
                },
                'SVM': {
                    'model__C': [0.1, 1, 10, 100],
                    'model__kernel': ['rbf', 'linear'],
                    'model__gamma': ['scale', 'auto', 0.001, 0.01]
                },
                'KNN': {
                    'model__n_neighbors': [3, 5, 7, 9, 11],
                    'model__weights': ['uniform', 'distance'],
                    'model__metric': ['euclidean', 'manhattan']
                },
                'DecisionTree': {
                    'model__max_depth': [3, 5, 10, 15, None],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                }
            }
    
    def make_predictions(self, save_predictions=True):
        """Make predictions on test set"""
        if self.test_df is None:
            print("⚠️ No test data available")
            return None
            
        print("🔮 Making predictions...")
        
        predictions = self.best_pipeline.predict(self.X_test)
        
        # Use original test data index or create sequential IDs
        if hasattr(self.test_df, 'index'):
            test_ids = self.test_df.index.tolist()
        else:
            test_ids = range(len(predictions))
            
        submission = pd.DataFrame({
            'ID': test_ids,
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
            'cv_score': getattr(self, 'best_score', abs(self.cv_results[self.best_model_name]['mean']) if self.problem_type == 'regression' else self.cv_results[self.best_model_name]['mean']),
            'feature_types': self.feature_types
        }
        
        with open('model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Model saved as {filename}")
        print("✅ Model info saved as model_info.json")
    
    def run_pipeline(self, train_path, target_column, test_path=None, 
                    problem_type='classification', exclude_columns=None, 
                    custom_features=None, feature_engineering_func=None, verbose=None, fast_mode=None, tuning_method=None, n_jobs=None):
        """Run complete pipeline"""
        print("🚀 STARTING UNIVERSAL ML PIPELINE")
        print("=" * 60)
        
        self.problem_type = problem_type
        if verbose is not None:
            self.verbose = verbose
        if fast_mode is not None:
            self.fast_mode = fast_mode
        if tuning_method is not None:
            self.tuning_method = tuning_method
        if n_jobs is not None:
            self.n_jobs = n_jobs
        
        self.load_data(train_path, test_path, target_column)
        
        if feature_engineering_func:
            print("🛠️ Applying feature engineering...")
            self.train_df = feature_engineering_func(self.train_df_full.copy())
            if self.test_df is not None:
                self.test_df = feature_engineering_func(self.test_df)
            print("✅ Feature engineering complete.")
        else:
            self.train_df = self.train_df_full.copy()
        
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
        best_score_display = getattr(self, 'best_score', abs(self.cv_results[self.best_model_name]['mean']) if self.problem_type == 'regression' else self.cv_results[self.best_model_name]['mean'])
        print(f"✅ Best Score: {best_score_display:.4f}")
        print("=" * 60)