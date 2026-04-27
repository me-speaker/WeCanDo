"""
DeepInd 高斯过程代理模型
Gaussian Process surrogate model with uncertainty estimation
"""

import torch
import numpy as np
from typing import Optional, Tuple

from ...core.base import BaseSurrogate
from ...core.registry import register_surrogate
from ...utils.logger import get_logger


@register_surrogate("GaussianProcess")
class GaussianProcess(BaseSurrogate):
    """
    高斯过程代理模型

    使用 RBF (Radial Basis Function) 核函数，支持：
    - 点预测
    - 不确定性估计
    - 增量更新

    Args:
        input_dim: 输入维度
        noise_var: 观测噪声方差，默认 1e-5
        length_scale: RBF 核长度尺度，默认 1.0
        output_scale: 输出缩放因子，默认 1.0
    """

    def __init__(
        self,
        input_dim: int,
        noise_var: float = 1e-5,
        length_scale: float = 1.0,
        output_scale: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.noise_var = noise_var
        self.length_scale = length_scale
        self.output_scale = output_scale

        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._K: Optional[np.ndarray] = None
        self._K_inv: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._is_fitted = False

        self.logger = get_logger("DeepInd::GaussianProcess")

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        RBF 核函数

        K(x1, x2) = output_scale * exp(-||x1 - x2||^2 / (2 * length_scale^2))
        """
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        dist_sq = X1_sq - 2 * X1 @ X2.T + X2_sq.T
        return self.output_scale * np.exp(-dist_sq / (2 * self.length_scale ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        """
        训练高斯过程模型

        Args:
            X: 输入特征，shape (n_samples, n_features)
            y: 目标值，shape (n_samples,) 或 (n_samples, n_objectives)

        Returns:
            self
        """
        self.logger.info("GaussianProcess: 开始训练")

        X = np.atleast_2d(np.asarray(X))
        y = np.asarray(y)

        n_samples = X.shape[0]

        # 计算核矩阵
        K = self._rbf_kernel(X, X)
        K_noise = K + self.noise_var * np.eye(n_samples)

        # 计算逆矩阵（使用 Cholesky 分解保证数值稳定）
        try:
            L = np.linalg.cholesky(K_noise)
            self._K_inv = np.linalg.inv(L)
            self._K_inv = self._K_inv @ self._K_inv.T
        except np.linalg.LinAlgError:
            self.logger.warning("Cholesky 分解失败，使用伪逆")
            self._K_inv = np.linalg.pinv(K_noise)

        # 计算 alpha = K^-1 @ y
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._alpha = self._K_inv @ y

        self._X_train = X.copy()
        self._y_train = y.copy()
        self._is_fitted = True

        self.logger.info(f"GaussianProcess: 训练完成，样本数: {n_samples}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测（仅返回均值）

        Args:
            X: 输入特征，shape (n_query, n_features)

        Returns:
            预测均值，shape (n_query,) 或 (n_query, n_objectives)
        """
        if not self._is_fitted:
            raise RuntimeError("GaussianProcess 未训练，请先调用 fit()")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 计算 k* 核向量
        k_star = self._rbf_kernel(X, self._X_train)

        # 预测均值: mu = k* @ K^-1 @ y = k* @ alpha
        mu = k_star @ self._alpha

        if mu.shape[1] == 1:
            mu = mu.flatten()

        return mu

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        带不确定性预测

        Args:
            X: 输入特征，shape (n_query, n_features)

        Returns:
            (预测均值, 预测方差)，均为 shape (n_query,) 或 (n_query, n_objectives)
        """
        if not self._is_fitted:
            raise RuntimeError("GaussianProcess 未训练，请先调用 fit()")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 计算 k* 核向量
        k_star = self._rbf_kernel(X, self._X_train)

        # 预测均值
        mu = k_star @ self._alpha
        if mu.shape[1] == 1:
            mu = mu.flatten()

        # 预测方差: var = k(x*, x*) - k* @ K^-1 @ k*^T
        k_star_K_inv = k_star @ self._K_inv
        var_diag = self._rbf_kernel(X, X).diagonal().reshape(-1, 1)
        var = var_diag - np.sum(k_star_K_inv * k_star, axis=1, keepdims=True)
        var = np.maximum(var, 1e-10)  # 确保方差非负

        if var.shape[1] == 1:
            var = var.flatten()

        return mu, var

    def update(self, X_new: np.ndarray, y_new: np.ndarray) -> "GaussianProcess":
        """
        增量更新

        Args:
            X_new: 新增数据特征
            y_new: 新增目标值

        Returns:
            self
        """
        self.logger.info("GaussianProcess: 执行增量更新")

        X_new = np.asarray(X_new)
        y_new = np.asarray(y_new)

        if self._X_train is not None:
            X_combined = np.vstack([self._X_train, X_new])
            y_combined = np.vstack([self._y_train, y_new.reshape(-1, 1)])
        else:
            X_combined = X_new
            y_combined = y_new.reshape(-1, 1)

        return self.fit(X_combined, y_combined)

    @property
    def is_differentiable(self) -> bool:
        """高斯过程在训练数据点处可微"""
        return False

    @property
    def n_output(self) -> int:
        """输出维度"""
        if self._y_train is not None:
            return self._y_train.shape[1] if self._y_train.ndim > 1 else 1
        return 0