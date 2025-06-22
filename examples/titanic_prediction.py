from universal_ml_framework import UniversalMLPipeline

pipeline = UniversalMLPipeline(
    problem_type='classification', 
    random_state=42, 
    verbose=True, 
    fast_mode=False, 
    tuning_method='bayesian', 
    n_jobs=-1
)
pipeline.run_pipeline(
    train_path='train.csv',
    test_path='test.csv',
    target_column='Survived',
    problem_type='classification',
    id_column='PassengerId'
)

pipeline.prepare_data()