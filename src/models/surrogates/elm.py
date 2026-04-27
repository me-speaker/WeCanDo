"""
DeepInd ELM (Extreme Learning Machine) 代理模型
极限学习机：输入权重随机初始化，输出权重通过最小二乘法求解
"""

import torch
import torch.nn as nn
import numpy as np

from ...core.base import BaseSurrogate
from ...core.registry import register_surrogate
from ...utils.logger import get_logger


@register_surrogate("ELM")
class ExtremeLearningMachine(BaseSurrogate):
    """
    极限学习机 (Extreme Learning Machine)

    特点:
    - 输入层权重随机初始化，无需训练
    - 隐藏层激活函数通常使用 ReLU 或 Sigmoid
    - 输出层权重通过最小二乘法解析求解
    - 训练速度极快

    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
    """

    def __init__(self, input_dim: int, hidden_dim: int = 20):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 输入权重 (随机初始化)
        self.input_weight = nn.Parameter(
            torch.randn(hidden_dim, input_dim) * 0.5
        )
        # 偏置
        self.bias = nn.Parameter(torch.randn(hidden_dim))
        # 输出权重 (待求解)
        self.output_weight = None

        self.logger = get_logger("DeepInd::ELM")
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtremeLearningMachine":
        """
        训练 ELM

        通过最小二乘法求解输出权重

        Args:
            X: 输入特征，shape (n_samples, n_features)
            y: 目标值，shape (n_samples, n_objectives) 或 (n_samples,)

        Returns:
            self
        """
        self.logger.info("ELM: 开始训练")

        # 保存历史数据（用于增量更新），确保y为2D以便vstack
        self._X_history = X.copy() if isinstance(X, np.ndarray) else X
        y_2d = np.atleast_2d(y)
        if y_2d.shape[0] == 1:
            y_2d = y_2d.T
        self._y_history = y_2d

        # 转换为 tensor
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # 计算隐藏层激活值
        # H = relu(X @ W_in.T + b)
        H = torch.relu(X_tensor @ self.input_weight.T + self.bias)
        self.logger.debug(f"  - 隐藏层激活 shape: {H.shape}")

        # 最小二乘法求解输出权重: W_out = (H^T H)^(-1) H^T y
        # 使用 lstsq 求解最小二乘问题
        self.output_weight = torch.linalg.lstsq(H, y_tensor).solution.T

        self._is_fitted = True
        self.logger.info("ELM: 训练完成")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 输入特征，shape (n_samples, n_features) 或 (n_features,)

        Returns:
            预测值，shape (n_samples, n_objectives) 或 (n_objectives,)
        """
        if not self._is_fitted:
            raise RuntimeError("ELM 未训练，请先调用 fit()")

        X_tensor = torch.FloatTensor(X)

        # 保持输入维度一致性
        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(0)

        with torch.no_grad():
            H = torch.relu(X_tensor @ self.input_weight.T + self.bias)
            y_pred = H @ self.output_weight.T

        return y_pred.numpy()

    def update(self, X_new: np.ndarray, y_new: np.ndarray) -> "ExtremeLearningMachine":
        """
        增量更新（简版：保留原有权重，用新数据微调）

        Args:
            X_new: 新增数据特征
            y_new: 新增目标值

        Returns:
            self
        """
        self.logger.info("ELM: 执行增量更新")

        # 确保 y_new 是 2D 以便 vstack
        y_new_2d = np.atleast_2d(y_new)
        if y_new_2d.shape[0] == 1:
            y_new_2d = y_new_2d.T

        # 合并新旧数据
        X_combined = np.vstack([self._X_history, X_new])
        y_combined = np.vstack([self._y_history, y_new_2d])

        # 重新训练（fit会重新保存为2D）
        return self.fit(X_combined, y_combined)

    @property
    def is_differentiable(self) -> bool:
        """ELM 可微"""
        return True

    def gradient(self, X: np.ndarray) -> np.ndarray:
        """
        计算预测对输入的梯度

        对于 ELM:
        - H = relu(X @ W_in.T + b)
        - y = H @ W_out.T
        - dy/dX = dH/dX @ W_out.T
        - dH/dX = diag(relu'(X @ W_in.T + b)) @ W_in

        Args:
            X: 输入特征，shape (n_samples, n_features) 或 (n_features,)

        Returns:
            梯度，shape (n_samples, n_features) 或 (n_features,)
        """
        if not self._is_fitted:
            raise RuntimeError("ELM 未训练，请先调用 fit()")

        X_tensor = torch.FloatTensor(X)

        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(0)

        X_tensor.requires_grad_(True)

        # 计算隐藏层激活
        H = torch.relu(X_tensor @ self.input_weight.T + self.bias)

        # 计算预测
        y_pred = H @ self.output_weight.T

        # 对每个输出分别计算梯度
        grads = []
        n_outputs = y_pred.shape[1] if y_pred.dim() > 1 else 1
        for i in range(n_outputs):
            if n_outputs == 1:
                grad = torch.autograd.grad(
                    outputs=y_pred,
                    inputs=X_tensor,
                    grad_outputs=torch.ones_like(y_pred),
                    retain_graph=True,
                )[0]
            else:
                grad = torch.autograd.grad(
                    outputs=y_pred[:, i],
                    inputs=X_tensor,
                    grad_outputs=torch.ones_like(y_pred[:, i]),
                    retain_graph=True,
                )[0]
            grads.append(grad)

        if len(grads) == 1:
            return grads[0].numpy()
        return torch.stack(grads, dim=0).numpy()

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple:
        """
        带不确定性预测

        使用训练残差估计不确定性

        Args:
            X: 输入特征，shape (n_samples, n_features)

        Returns:
            (预测值, 不确定性)，shape (n_samples, n_objectives)
        """
        if not self._is_fitted:
            raise RuntimeError("ELM 未训练，请先调用 fit()")

        y_pred = self.predict(X)

        # 使用训练时的残差计算不确定性估计
        if hasattr(self, "_X_history") and hasattr(self, "_y_history"):
            y_train_pred = self.predict(self._X_history)
            residuals = self._y_history - y_train_pred
            # 计算每个输出的残差标准差
            if residuals.ndim == 1:
                residual_std = np.std(residuals)
            else:
                residual_std = np.std(residuals, axis=0)
            # 添加与输入相关的噪声估计
            uncertainty = np.ones_like(y_pred) * residual_std
            return y_pred, uncertainty

        # 如果没有历史数据，返回均匀不确定性
        uncertainty = np.ones_like(y_pred) * 0.1
        return y_pred, uncertainty

    @property
    def n_output(self) -> int:
        """输出维度"""
        if self.output_weight is not None:
            return self.output_weight.shape[0]
        return 0
