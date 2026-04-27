"""
DeepInd 共轭梯度法 (CG) 优化器
适用于可微目标函数
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Callable, Optional, Tuple

from ..base import BaseOptimizer, OptimizationResult


class CGOptimizer(BaseOptimizer):
    """
    共轭梯度法 (Conjugate Gradient) 优化器

    特点:
    - 适合大规模优化问题
    - 内存需求低
    - 对正定二次函数在 n 步内收敛

    Args:
        n_iter: 最大迭代次数
        tol: 收敛容忍度
        method: CG 变种 ('FR', 'PR', 'HS')
    """

    def __init__(
        self,
        n_iter: int = 100,
        tol: float = 1e-6,
        method: str = "FR",
        verbose: bool = True,
    ):
        super().__init__(n_iter=n_iter, tol=tol, verbose=verbose)
        self.method = method  # FR (Fletcher-Reeves), PR (Polak-Ribiere), HS (Hestenes-Stiefel)

    def minimize(
        self,
        fun: Callable,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        constraints: Optional[List[dict]] = None,
    ) -> OptimizationResult:
        """
        执行共轭梯度法优化

        Args:
            fun: 目标函数
            x0: 初始点
            bounds: 变量边界
            constraints: 约束条件

        Returns:
            OptimizationResult
        """
        if self.verbose:
            print(f"CG ({self.method}): 开始优化, 初始点={x0}")

        result = minimize(
            fun,
            x0,
            method="CG",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": self.n_iter,
                "gtol": self.tol,
            },
        )

        self._n_iterations = result.nit

        return OptimizationResult(
            x=result.x,
            fun=result.fun,
            success=result.success,
            message=result.message,
            n_iter=result.nit,
        )

    def minimize_multi(
        self,
        objectives: List[Callable],
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        constraints: Optional[List[dict]] = None,
    ) -> OptimizationResult:
        """
        多目标优化：使用加权求和法

        Args:
            objectives: 目标函数列表
            x0: 初始点
            bounds: 变量边界
            constraints: 约束条件

        Returns:
            OptimizationResult
        """
        n_obj = len(objectives)

        def weighted_objective(x):
            weights = np.ones(n_obj) / n_obj
            return sum(w * obj(x) for w, obj in zip(weights, objectives))

        if self.verbose:
            print(f"CG ({self.method}): 多目标优化, {n_obj} 个目标")

        return self.minimize(weighted_objective, x0, bounds, constraints)
