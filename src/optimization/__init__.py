"""
DeepInd 优化算法模块
实现多种优化算法，支持自动选择和约束处理
"""

from .base import BaseOptimizer, OptimizationResult
from .algorithms.lbfgs import LBFGSOptimizer
from .algorithms.cg import CGOptimizer
from .algorithms.bayesian import BayesianOptimizer
from .algorithms.genetic import GeneticOptimizer
from .algorithms.auto_optimizer import AutoOptimizer

__all__ = [
    "BaseOptimizer",
    "OptimizationResult",
    "LBFGSOptimizer",
    "CGOptimizer",
    "BayesianOptimizer",
    "GeneticOptimizer",
    "AutoOptimizer",
]