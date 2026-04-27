"""
DeepInd 优化器基类
定义优化器的标准接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Callable, Optional, Tuple, Union
import numpy as np


@dataclass
class OptimizationResult:
    """
    优化结果数据类

    Attributes:
        x: 最优解
        fun: 最优目标函数值
        success: 是否成功收敛
        message: 结果信息
        n_iter: 迭代次数
        all_solutions: 所有探索的解（用于分析）
    """
    x: np.ndarray
    fun: Union[float, np.ndarray]
    success: bool
    message: str
    n_iter: int
    all_solutions: Optional[List[np.ndarray]] = None


class BaseOptimizer(ABC):
    """
    优化器抽象基类

    定义所有优化器必须实现的接口
    """

    def __init__(
        self,
        n_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = True,
    ):
        """
        初始化优化器

        Args:
            n_iter: 最大迭代次数
            tol: 收敛容忍度
            verbose: 是否输出日志
        """
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self._n_iterations = 0

    @abstractmethod
    def minimize(
        self,
        fun: Callable,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        constraints: Optional[List[dict]] = None,
    ) -> OptimizationResult:
        """
        执行单目标优化

        Args:
            fun: 目标函数
            x0: 初始点
            bounds: 变量边界 [(low, high), ...]
            constraints: 约束条件列表

        Returns:
            OptimizationResult: 优化结果
        """
        pass

    @abstractmethod
    def minimize_multi(
        self,
        objectives: List[Callable],
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        constraints: Optional[List[dict]] = None,
    ) -> OptimizationResult:
        """
        执行多目标优化

        Args:
            objectives: 目标函数列表
            x0: 初始点
            bounds: 变量边界
            constraints: 约束条件列表

        Returns:
            OptimizationResult: 优化结果
        """
        pass

    def _check_differentiability(self, fun: Callable, x0: np.ndarray) -> bool:
        """
        检查目标函数是否可微（数值方法）

        Args:
            fun: 目标函数
            x0: 测试点

        Returns:
            bool: 是否可微
        """
        eps = 1e-8
        try:
            f0 = fun(x0)
            for i in range(len(x0)):
                x_plus = x0.copy()
                x_plus[i] += eps
                f_plus = fun(x_plus)
                # 检查偏导数是否存在且有限
                if not np.isfinite(f_plus):
                    return False
            return True
        except Exception:
            return False

    def _project_to_bounds(self, x: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """将解投影到边界内"""
        x_proj = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
        return x_proj
