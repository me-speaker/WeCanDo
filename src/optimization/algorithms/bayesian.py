"""
DeepInd 贝叶斯优化器
适用于不可微目标函数和黑盒优化
"""

import numpy as np
from typing import List, Callable, Optional, Tuple
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from ..base import BaseOptimizer, OptimizationResult


class BayesianOptimizer(BaseOptimizer):
    """
    贝叶斯优化器 (Bayesian Optimization)

    特点:
    - 适用于不可微目标函数
    - 利用代理模型指导搜索
    - 适合黑盒优化和评估代价高的场景

    Args:
        n_iter: 最大迭代次数
        n_initial: 初始采样点数量
        acquisition: 采集函数 ('EI', 'PI', 'UCB')
        verbose: 是否输出日志
    """

    def __init__(
        self,
        n_iter: int = 50,
        n_initial: int = 10,
        acquisition: str = "EI",
        verbose: bool = True,
    ):
        super().__init__(n_iter=n_iter, tol=0.0, verbose=verbose)
        self.n_initial = n_initial
        self.acquisition = acquisition
        self._gp = None
        self._X_obs = []
        self._y_obs = []

    def minimize(
        self,
        fun: Callable,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        constraints: Optional[List[dict]] = None,
    ) -> OptimizationResult:
        """
        执行贝叶斯优化

        Args:
            fun: 目标函数
            x0: 初始点（可选）
            bounds: 变量边界
            constraints: 约束条件（暂不支持）

        Returns:
            OptimizationResult
        """
        if constraints:
            print("警告: 贝叶斯优化暂不支持约束条件")

        n_vars = len(bounds)
        self._X_obs = []
        self._y_obs = []

        # 定义搜索空间
        bounds_array = np.array(bounds)

        if self.verbose:
            print(f"贝叶斯优化: 开始优化, {n_vars} 变量, {self.n_iter} 迭代")

        # 初始采样
        if x0 is not None and len(x0) == n_vars:
            self._X_obs.append(x0.copy())
            self._y_obs.append(fun(x0))
            print(f"  添加初始点: x0={x0}, f(x0)={self._y_obs[0]:.4f}")

        X_init = self._initial_sampling(n_vars, bounds_array)
        for x in X_init:
            y = fun(x)
            self._X_obs.append(x)
            self._y_obs.append(y)

        # 初始化高斯过程
        self._fit_gp()

        # 主循环
        best_x = None
        best_y = float('inf')
        all_solutions = []

        for iteration in range(self.n_iter):
            # 更新最优解
            y_arr = np.array(self._y_obs)
            best_idx = np.argmin(y_arr)
            if y_arr[best_idx] < best_y:
                best_y = y_arr[best_idx]
                best_x = self._X_obs[best_idx].copy()

            all_solutions.append(best_x.copy())

            if self.verbose:
                print(f"  迭代 {iteration+1}/{self.n_iter}: best_f={best_y:.4f}")

            # 计算采集函数并优化
            x_next = self._optimize_acquisition(fun, bounds_array)

            # 评估新点
            y_next = fun(x_next)
            self._X_obs.append(x_next)
            self._y_obs.append(y_next)

            # 更新 GP 模型
            self._fit_gp()

        self._n_iterations = self.n_iter

        return OptimizationResult(
            x=best_x,
            fun=best_y,
            success=True,
            message="优化完成",
            n_iter=self.n_iter,
            all_solutions=all_solutions,
        )

    def minimize_multi(
        self,
        objectives: List[Callable],
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        constraints: Optional[List[dict]] = None,
    ) -> OptimizationResult:
        """
        多目标贝叶斯优化：使用 ParEGO 方法

        Args:
            objectives: 目标函数列表
            x0: 初始点
            bounds: 变量边界
            constraints: 约束条件

        Returns:
            OptimizationResult
        """
        n_obj = len(objectives)

        def scalarized_objective(x, rho=0.05):
            """使用 TS ( Tchebycheff ) 标量化"""
            f_values = np.array([obj(x) for obj in objectives])
            max_f = np.max(f_values)
            weighted_sum = np.sum(f_values)
            return max_f + rho * weighted_sum

        if self.verbose:
            print(f"贝叶斯优化: 多目标优化, {n_obj} 个目标")

        return self.minimize(
            lambda x: scalarized_objective(x),
            x0,
            bounds,
            constraints,
        )

    def _initial_sampling(self, n_vars: int, bounds: np.ndarray) -> np.ndarray:
        """初始拉丁超立方采样"""
        n_samples = max(self.n_initial - len(self._X_obs), 0)
        if n_samples <= 0:
            return np.array([])

        samples = np.zeros((n_samples, n_vars))
        for i in range(n_vars):
            samples[:, i] = np.random.uniform(bounds[i, 0], bounds[i, 1], n_samples)
        return samples

    def _fit_gp(self):
        """训练高斯过程模型"""
        X = np.array(self._X_obs)
        y = np.array(self._y_obs).reshape(-1, 1)

        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(
            noise_level=1e-5, noise_level_bounds=(1e-10, 1e1)
        )

        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            alpha=1e-6,
        )
        self._gp.fit(X, y)

    def _optimize_acquisition(
        self,
        target_fun: Callable,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """
        优化采集函数找到下一个评估点

        Args:
            target_fun: 真实目标函数（用于局部搜索）
            bounds: 边界

        Returns:
            下一个评估点
        """
        n_vars = len(bounds)

        def negative_acquisition(x):
            return -self._acquisition_function(x)

        # 多次随机重启
        best_x = None
        best_acq = float('inf')

        for _ in range(20):
            x0 = np.array([np.random.uniform(bounds[i, 0], bounds[i, 1]) for i in range(n_vars)])
            try:
                result = minimize(negative_acquisition, x0, bounds=bounds, method="L-BFGS-B")
                if result.fun < best_acq:
                    best_acq = result.fun
                    best_x = result.x
            except Exception:
                continue

        # 如果所有尝试都失败，使用随机点
        if best_x is None:
            best_x = x0

        # 确保在边界内
        best_x = np.clip(best_x, bounds[:, 0], bounds[:, 1])
        return best_x

    def _acquisition_function(self, x: np.ndarray) -> float:
        """
        期望改进 (Expected Improvement) 采集函数

        Args:
            x: 候选点

        Returns:
            采集函数值
        """
        x = x.reshape(1, -1)
        mu, sigma = self._gp.predict(x, return_std=True)

        # sklearn GP可能返回None或标量sigma
        if sigma is None:
            sigma = 1e-6
        sigma = np.atleast_1d(sigma).reshape(-1, 1)

        # 当前最优
        y_min = np.min(self._y_obs)

        # 期望改进
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (y_min - mu) / sigma
            ei = (y_min - mu) * self._norm_cdf(z) + sigma * self._norm_pdf(z)
            ei[sigma < 1e-10] = 0.0

        return float(ei[0, 0])

    @staticmethod
    def _norm_cdf(z):
        """标准正态分布 CDF"""
        return 0.5 * (1 + np.erf(z / np.sqrt(2)))

    @staticmethod
    def _norm_pdf(z):
        """标准正态分布 PDF"""
        return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
