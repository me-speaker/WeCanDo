"""
DeepInd 自动优化器选择模块
根据代理模型特性自动选择合适的优化算法
"""

import numpy as np
from typing import List, Callable, Optional, Tuple, Union

from ..base import BaseOptimizer, OptimizationResult
from .lbfgs import LBFGSOptimizer
from .cg import CGOptimizer
from .bayesian import BayesianOptimizer
from .genetic import GeneticOptimizer


class AutoOptimizer:
    """
    自动优化器选择器

    根据以下规则自动选择优化算法:
    1. 如果代理模型可微且优化目标可微 -> L-BFGS
    2. 如果代理模型可微但需要更稳健的搜索 -> CG (共轭梯度)
    3. 如果代理模型不可微或目标函数黑盒 -> 贝叶斯优化
    4. 如果需要全局优化或多模态优化 -> 遗传算法
    5. 多目标优化 -> NSGA-II 或加权遗传算法

    Args:
        optimizer_type: 手动指定优化器类型
        auto_select: 是否自动选择
        surrogate_model: 代理模型（用于检测可微性）
    """

    OPTIMIZER_MAP = {
        "LBFGS": LBFGSOptimizer,
        "CG": CGOptimizer,
        "BAYESIAN": BayesianOptimizer,
        "GENETIC": GeneticOptimizer,
    }

    def __init__(
        self,
        optimizer_type: Optional[str] = None,
        auto_select: bool = True,
        surrogate_model=None,
        **kwargs,
    ):
        """
        初始化自动优化器

        Args:
            optimizer_type: 手动指定 ('LBFGS', 'CG', 'BAYESIAN', 'GENETIC')
            auto_select: 是否启用自动选择
            surrogate_model: 代理模型（用于检测 is_differentiable 属性）
            **kwargs: 传递给具体优化器的参数
        """
        self.optimizer_type = optimizer_type
        self.auto_select = auto_select
        self.surrogate_model = surrogate_model
        self.kwargs = kwargs
        self._optimizer = None
        self._optimizer_name = None

    def _is_differentiable(self) -> bool:
        """检测代理模型是否可微"""
        if self.surrogate_model is not None:
            if hasattr(self.surrogate_model, "is_differentiable"):
                return self.surrogate_model.is_differentiable
            if hasattr(self.surrogate_model, "predict"):
                # 尝试数值检测
                try:
                    test_x = np.random.randn(5)
                    _ = self.surrogate_model.predict(test_x.reshape(1, -1))
                    return True
                except Exception:
                    return False
        return True  # 默认假定可微

    def _select_optimizer(
        self,
        objectives: List[Callable],
        n_objectives: int,
        is_global: bool = False,
    ) -> BaseOptimizer:
        """
        根据条件选择优化器

        Args:
            objectives: 目标函数列表
            n_objectives: 目标函数数量
            is_global: 是否需要全局搜索

        Returns:
            选定的优化器实例
        """
        if self.optimizer_type:
            optimizer_cls = self.OPTIMIZER_MAP.get(self.optimizer_type.upper())
            if optimizer_cls is None:
                raise ValueError(f"未知的优化器类型: {self.optimizer_type}")
            self._optimizer_name = self.optimizer_type.upper()
            return optimizer_cls(**self.kwargs)

        # 自动选择逻辑
        if self.auto_select:
            if is_global or n_objectives > 1:
                # 多目标或全局搜索使用遗传算法
                self._optimizer_name = "GENETIC"
                return GeneticOptimizer(**self.kwargs)
            elif self._is_differentiable():
                # 可微使用 L-BFGS
                self._optimizer_name = "LBFGS"
                return LBFGSOptimizer(**self.kwargs)
            else:
                # 不可微使用贝叶斯优化
                self._optimizer_name = "BAYESIAN"
                return BayesianOptimizer(**self.kwargs)

        # 默认使用 L-BFGS
        self._optimizer_name = "LBFGS"
        return LBFGSOptimizer(**self.kwargs)

    def minimize(
        self,
        fun: Callable,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        constraints: Optional[List[dict]] = None,
        n_objectives: int = 1,
        is_global: bool = False,
    ) -> OptimizationResult:
        """
        执行单目标优化

        Args:
            fun: 目标函数
            x0: 初始点
            bounds: 变量边界
            constraints: 约束条件
            n_objectives: 目标函数数量
            is_global: 是否进行全局搜索

        Returns:
            OptimizationResult
        """
        objectives = [fun]
        optimizer = self._select_optimizer(objectives, n_objectives, is_global)

        print(f"自动优化器: 使用 {self._optimizer_name} 优化器")

        return optimizer.minimize(fun, x0, bounds, constraints)

    def minimize_multi(
        self,
        objectives: List[Callable],
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        constraints: Optional[List[dict]] = None,
        is_global: bool = True,
    ) -> OptimizationResult:
        """
        执行多目标优化

        Args:
            objectives: 目标函数列表
            x0: 初始点
            bounds: 变量边界
            constraints: 约束条件
            is_global: 是否进行全局搜索

        Returns:
            OptimizationResult
        """
        optimizer = self._select_optimizer(objectives, len(objectives), is_global)

        print(f"自动优化器: 使用 {self._optimizer_name} 进行多目标优化")

        if hasattr(optimizer, "minimize_multi"):
            return optimizer.minimize_multi(objectives, x0, bounds, constraints)
        else:
            # 如果优化器没有多目标方法，使用加权求和
            return optimizer.minimize(
                lambda x: sum(obj(x) for obj in objectives) / len(objectives),
                x0,
                bounds,
                constraints,
            )

    def list_available_optimizers(self) -> List[str]:
        """列出所有可用的优化器"""
        return list(self.OPTIMIZER_MAP.keys())


def create_optimizer(
    optimizer_type: str,
    surrogate_model=None,
    **kwargs,
) -> BaseOptimizer:
    """
    工厂函数：创建优化器

    Args:
        optimizer_type: 优化器类型
        surrogate_model: 代理模型
        **kwargs: 优化器参数

    Returns:
        优化器实例
    """
    if optimizer_type.upper() in AutoOptimizer.OPTIMIZER_MAP:
        return AutoOptimizer.OPTIMIZER_MAP[optimizer_type.upper()](**kwargs)
    elif optimizer_type.upper() == "AUTO":
        return AutoOptimizer(surrogate_model=surrogate_model, **kwargs)
    else:
        raise ValueError(f"未知的优化器类型: {optimizer_type}")
