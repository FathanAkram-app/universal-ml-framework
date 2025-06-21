# HELPER FUNCTIONS
# Quick setup functions and utilities

from core.pipeline import UniversalMLPipeline
from configs.dataset_configs import PipelineConfigs

def quick_classification_pipeline(train_path, target_column, test_path=None, exclude_columns=None):
    """Quick setup untuk classification problem"""
    print("🎯 QUICK CLASSIFICATION PIPELINE")
    print("-" * 40)
    
    pipeline = UniversalMLPipeline(problem_type='classification')
    pipeline.run_pipeline(
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        exclude_columns=exclude_columns or []
    )
    return pipeline

def quick_regression_pipeline(train_path, target_column, test_path=None, exclude_columns=None):
    """Quick setup untuk regression problem"""
    print("📈 QUICK REGRESSION PIPELINE")
    print("-" * 40)
    
    pipeline = UniversalMLPipeline(problem_type='regression')
    pipeline.run_pipeline(
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        exclude_columns=exclude_columns or []
    )
    return pipeline

def run_pipeline_with_config(config_name):
    """Run pipeline dengan konfigurasi yang dipilih"""
    
    configs = PipelineConfigs.get_all_configs()
    
    if config_name not in configs:
        print(f"❌ Config '{config_name}' not found!")
        print(f"Available configs: {list(configs.keys())}")
        return None
    
    config = configs[config_name]
    
    print(f"🚀 Running pipeline: {config['description']}")
    print("=" * 60)
    
    try:
        pipeline = UniversalMLPipeline(problem_type=config['problem_type'])
        pipeline.run_pipeline(
            train_path=config['train_path'],
            test_path=config.get('test_path'),
            target_column=config['target_column'],
            exclude_columns=config.get('exclude_columns', [])
        )
        
        print(f"✅ Pipeline completed for {config_name}")
        return pipeline
        
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return None

def list_available_configs():
    """List all available configurations"""
    configs = PipelineConfigs.get_all_configs()
    
    print("📋 AVAILABLE CONFIGURATIONS:")
    print("-" * 40)
    
    for i, (name, config) in enumerate(configs.items(), 1):
        print(f"{i:2d}. {name:15} - {config['description']}")
        print(f"    Type: {config['problem_type']}")
        print(f"    Target: {config['target_column']}")
        print()

def get_pipeline_info():
    """Get information about the pipeline framework"""
    info = {
        'version': '1.0.0',
        'supported_problems': ['classification', 'regression'],
        'algorithms': {
            'classification': ['RandomForest', 'LogisticRegression', 'SVM'],
            'regression': ['RandomForest', 'LinearRegression', 'SVM']
        },
        'features': [
            'Auto feature detection',
            'Missing value handling',
            'Categorical encoding',
            'Feature scaling',
            'Cross validation',
            'Hyperparameter tuning',
            'Model comparison',
            'Automatic predictions'
        ]
    }
    
    print("🌟 UNIVERSAL ML FRAMEWORK INFO")
    print("=" * 50)
    print(f"Version: {info['version']}")
    print(f"Supported Problems: {', '.join(info['supported_problems'])}")
    print("\nAlgorithms:")
    for problem, algos in info['algorithms'].items():
        print(f"  {problem.capitalize()}: {', '.join(algos)}")
    print("\nFeatures:")
    for feature in info['features']:
        print(f"  ✅ {feature}")
    print("=" * 50)
    
    return info