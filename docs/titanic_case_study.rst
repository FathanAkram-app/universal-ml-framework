Titanic Case Study
==================

This case study demonstrates the Universal ML Framework using the famous Titanic dataset, showing how to predict passenger survival with minimal code.

Dataset Overview
----------------

The Titanic dataset contains information about passengers aboard the RMS Titanic, including:

**Features:**
- **PassengerId**: Unique identifier for each passenger
- **Pclass**: Passenger class (1st, 2nd, 3rd)
- **Name**: Passenger name
- **Sex**: Gender (male/female)
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)

**Target:**
- **Survived**: Survival status (0=No, 1=Yes)

Implementation
--------------

Complete Titanic Prediction Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from universal_ml_framework import UniversalMLPipeline

   # Create pipeline with optimal settings for Titanic dataset
   pipeline = UniversalMLPipeline(
       problem_type='classification', 
       random_state=42, 
       verbose=True, 
       fast_mode=False, 
       tuning_method='bayesian', 
       n_jobs=-1
   )

   # Run complete pipeline
   pipeline.run_pipeline(
       train_path='titanic_train.csv',
       test_path='titanic_test.csv',
       target_column='Survived',
       problem_type='classification',
       id_column='PassengerId'
   )

What Happens Automatically
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Data Loading**
   
   - Loads training data (891 passengers)
   - Loads test data (418 passengers)
   - Identifies target column (Survived)
   - Uses PassengerId as identifier

2. **Feature Detection**
   
   The framework automatically categorizes features:
   
   - **Numeric**: Age, SibSp, Parch, Fare
   - **Categorical**: Pclass, Name, Sex, Ticket, Cabin, Embarked
   - **Binary**: None (Survived is the target)

3. **Preprocessing**
   
   - **Age**: Median imputation → Standard scaling
   - **Fare**: Median imputation → Standard scaling
   - **Sex, Embarked**: Constant imputation → One-hot encoding
   - **Name, Ticket, Cabin**: Handled as categorical features

4. **Model Training**
   
   Tests 7 classification algorithms:
   
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - Logistic Regression
   - Support Vector Machine
   - Naive Bayes
   - K-Nearest Neighbors
   - Decision Tree

5. **Cross Validation**
   
   - Uses StratifiedKFold (5 folds)
   - Preserves class distribution (survived vs not survived)
   - Parallel processing across all CPU cores

6. **Hyperparameter Tuning**
   
   - Uses Bayesian optimization for intelligent parameter search
   - Optimizes the best performing model
   - Comprehensive parameter grids for each algorithm

7. **Prediction Generation**
   
   - Generates survival predictions for test set
   - Uses PassengerId as identifier
   - Exports to predictions.csv

Expected Results
----------------

Typical Performance Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cross-Validation Results:**

.. code-block:: text

   📊 Cross validating models...
   
   [1/7] 🔄 Training RandomForest...
     Fold 1/5: 0.8324
     Fold 2/5: 0.8202
     Fold 3/5: 0.8315
     Fold 4/5: 0.8427
     Fold 5/5: 0.8258
     ✅ RandomForest completed - Mean: 0.8305 (±0.0081)
   
   [2/7] 🔄 Training GradientBoosting...
     Fold 1/5: 0.8268
     Fold 2/5: 0.8146
     Fold 3/5: 0.8315
     Fold 4/5: 0.8371
     Fold 5/5: 0.8202
     ✅ GradientBoosting completed - Mean: 0.8260 (±0.0078)
   
   🏆 Best model: RandomForest

**Final Results:**

.. code-block:: text

   🎉 PIPELINE COMPLETED!
   ============================================================
   ✅ Problem Type: classification
   ✅ Best Model: RandomForest
   ✅ Best Score: 0.8456
   ============================================================

Feature Importance Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework automatically identifies the most important features for survival prediction:

1. **Sex** - Gender is the strongest predictor
2. **Fare** - Ticket price indicates passenger class/wealth
3. **Age** - Age affects survival probability
4. **Pclass** - Passenger class (1st, 2nd, 3rd)
5. **SibSp/Parch** - Family size relationships

Output Files
------------

Generated Files
~~~~~~~~~~~~~~~

After running the pipeline, you'll find:

**predictions.csv**

.. code-block:: csv

   PassengerId,Prediction
   892,0
   893,1
   894,0
   895,0
   896,1
   ...

**model_info.json**

.. code-block:: json

   {
     "problem_type": "classification",
     "best_model": "RandomForest",
     "best_params": {
       "model__n_estimators": 200,
       "model__max_depth": 10,
       "model__min_samples_split": 2
     },
     "cv_score": 0.8456,
     "feature_types": {
       "numeric": ["Age", "SibSp", "Parch", "Fare"],
       "categorical": ["Pclass", "Name", "Sex", "Ticket", "Cabin", "Embarked"],
       "binary": []
     }
   }

**best_model.pkl**

Serialized trained model ready for production use.

Advanced Usage
--------------

Custom Feature Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~

For better results, you can add custom feature engineering:

.. code-block:: python

   def titanic_feature_engineering(df):
       # Extract title from name
       df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
       
       # Create family size feature
       df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
       
       # Create age groups
       df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                              labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
       
       # Create fare groups
       df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
       
       return df

   # Run with custom feature engineering
   pipeline.run_pipeline(
       train_path='titanic_train.csv',
       test_path='titanic_test.csv',
       target_column='Survived',
       id_column='PassengerId',
       feature_engineering_func=titanic_feature_engineering
   )

Exclude Irrelevant Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Exclude features that don't help prediction
   pipeline.run_pipeline(
       train_path='titanic_train.csv',
       test_path='titanic_test.csv',
       target_column='Survived',
       id_column='PassengerId',
       exclude_columns=['Name', 'Ticket', 'Cabin']  # High cardinality features
   )

Performance Comparison
----------------------

Framework vs Manual Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Universal ML Framework:**

.. code-block:: python

   # 10 lines of code
   from universal_ml_framework import UniversalMLPipeline
   
   pipeline = UniversalMLPipeline(problem_type='classification', tuning_method='bayesian')
   pipeline.run_pipeline('titanic_train.csv', 'Survived', 'titanic_test.csv', id_column='PassengerId')

**Manual Implementation:**

.. code-block:: python

   # 100+ lines of code
   import pandas as pd
   from sklearn.model_selection import cross_val_score, GridSearchCV
   from sklearn.preprocessing import StandardScaler, OneHotEncoder
   from sklearn.compose import ColumnTransformer
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.pipeline import Pipeline
   # ... many more imports and 100+ lines of preprocessing, training, tuning code

**Results Comparison:**

+------------------+-------------------+------------------+
| Metric           | Framework         | Manual           |
+==================+===================+==================+
| **Lines of Code**| 4                 | 100+             |
+------------------+-------------------+------------------+
| **Development**  | 2 minutes         | 2-4 hours        |
| **Time**         |                   |                  |
+------------------+-------------------+------------------+
| **Accuracy**     | 84.56%            | 82-85%           |
+------------------+-------------------+------------------+
| **Models Tested**| 7                 | 1-2              |
+------------------+-------------------+------------------+
| **Tuning**       | Bayesian          | Manual/Grid      |
+------------------+-------------------+------------------+

Key Insights
------------

Why This Works Well
~~~~~~~~~~~~~~~~~~~

1. **Automatic Feature Detection**: Correctly identifies numeric vs categorical features
2. **Proper Preprocessing**: Handles missing values and scaling appropriately
3. **Model Comparison**: Tests multiple algorithms to find the best performer
4. **Smart Tuning**: Bayesian optimization finds optimal hyperparameters efficiently
5. **Production Ready**: Generates all necessary files for deployment

Lessons Learned
~~~~~~~~~~~~~~~

1. **Gender is Key**: Sex is the most important feature for Titanic survival
2. **Class Matters**: Passenger class strongly correlates with survival
3. **Age Factor**: Children and elderly have different survival patterns
4. **Family Size**: Both very small and very large families had lower survival rates
5. **Fare Proxy**: Ticket fare serves as a proxy for socioeconomic status

This case study demonstrates how the Universal ML Framework can achieve competitive results with minimal effort, making machine learning accessible to users of all skill levels.