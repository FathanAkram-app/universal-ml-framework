"""Universal ML Framework - A complete machine learning pipeline framework."""

__version__ = "1.0.1"
__author__ = "Fathan Akram"

from .core.pipeline import UniversalMLPipeline
from .utils.helpers import (
    quick_classification_pipeline,
    quick_regression_pipeline,
    run_pipeline_with_config,
    list_available_configs
)
from .utils.data_generator import DataGenerator

__all__ = [
    "UniversalMLPipeline",
    "quick_classification_pipeline", 
    "quick_regression_pipeline",
    "run_pipeline_with_config",
    "list_available_configs",
    "DataGenerator"
]