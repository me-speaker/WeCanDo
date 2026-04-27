"""
DeepInd L-BFGS 优化器
适用于可微目标函数的拟牛顿法
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Callable, Optional, Tuple

from ..base import BaseOptimizer, OptimizationResult


class LBFGSOptimizer(BaseOptimizer):
    """
    L-BFGS (Limited-memory BFGS) 优化器

    特点:
    - 适合大规模优化问题
    - 利用梯度信息构造Hessian近似
    - 收敛速度快，内存需求低

    Args:
        n_iter: 最大迭代次数
        tol: 收敛容忍度
        m: L-BFGS 历史步数
    """

    def __init__(
        self,
        n_iter: int = 100,
        tol: float = 1e-6,
        m: int = 10,
        verbose: bool = True,
    ):
        super().__init__(n_iter=n_iter, tol=tol, verbose=verbose)
        self.m = m

    def minimize(
        self,
        fun: Callable,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        constraints: Optional[List[dict]] = None,
    ) -> OptimizationResult:
        """
        执行 L-BFGS 优化

        Args:
            fun: 目标函数 (接受 ndarray 返回 scalar)
            x0: 初始点
            bounds: 变量边界
            constraints: scipy 格式约束

        Returns:
            OptimizationResult
        """
        if self.verbose:
            print(f"L-BFGS: 开始优化, 初始点={x0}")

        # 使用 scipy.optimize.minimize 实现 L-BFGS
        result = minimize(
            fun,
            x0,
            method="L-BFGS-B",
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
            print(f"L-BFGS: 多目标优化, {n_obj} 个目标")

        return self.minimize(weighted_objective, x0, bounds, constraints)
