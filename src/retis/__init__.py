"""
RETIS: Recursive Elimination of Trivial Intermediate Splits
Implementation based on Karalic 1992
"""

from .algorithm import RETISRegressor, RETISClassifier
from .evaluation import evaluate_regression, evaluate_classification, compare_models

__version__ = "0.1.0"
__all__ = ["RETISRegressor", "RETISClassifier", "evaluate_regression", "evaluate_classification", "compare_models"]
