"""
DeepInd 代理模型模块
"""

from .elm import ExtremeLearningMachine
from .gaussian_process import GaussianProcess
from .neural_network import NeuralNetwork
from .xgboost_model import XGBoostModel

__all__ = [
    "ExtremeLearningMachine",
    "GaussianProcess",
    "NeuralNetwork",
    "XGBoostModel",
]
