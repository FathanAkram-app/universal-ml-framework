# Universal ML Framework
# A complete machine learning pipeline framework for any dataset

__version__ = "1.2.0"
__author__ = "Fathan Akram"

from .core.pipeline import UniversalMLPipeline
from .configs.dataset_configs import PipelineConfigs
from .utils.data_generator import DataGenerator
from .utils.helpers import quick_classification_pipeline, quick_regression_pipeline

__all__ = [
    'UniversalMLPipeline',
    'PipelineConfigs', 
    'DataGenerator',
    'quick_classification_pipeline',
    'quick_regression_pipeline'
]