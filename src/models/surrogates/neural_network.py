"""
DeepInd 神经网络代理模型
Multi-layer Perceptron surrogate model with PyTorch
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

from ...core.base import BaseSurrogate
from ...core.registry import register_surrogate
from ...utils.logger import get_logger


class MLPRegressor(nn.Module):
    """
    多层感知机回归网络
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 32],
        output_dim: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 选择激活函数
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "sigmoid":
            act_fn = nn.Sigmoid
        else:
            act_fn = nn.ReLU

        # 构建网络
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@register_surrogate("NeuralNetwork")
class NeuralNetwork(BaseSurrogate):
    """
    神经网络代理模型

    使用 PyTorch 实现的多层感知机，支持：
    - 点预测
    - 梯度计算
    - 增量更新（通过继续训练）

    Args:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表，默认 [64, 32]
        output_dim: 输出维度，默认 1
        learning_rate: 学习率，默认 0.001
        epochs: 训练轮数，默认 500
        batch_size: 批次大小，默认 32
        weight_decay: L2 正则化系数，默认 1e-5
        activation: 激活函数类型，默认 "relu"
        dropout: Dropout 比例，默认 0.0
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        output_dim: int = 1,
        learning_rate: float = 0.001,
        epochs: int = 500,
        batch_size: int = 32,
        weight_decay: float = 1e-5,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.activation = activation
        self.dropout = dropout

        # PyTorch 模型
        self._model: Optional[MLPRegressor] = None
        self._optimizer: Optional[torch.optim.Adam] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None
        self._y_mean: Optional[np.ndarray] = None
        self._y_std: Optional[np.ndarray] = None
        self._X_history: Optional[np.ndarray] = None
        self._y_history: Optional[np.ndarray] = None

        self._is_fitted = False
        self.logger = get_logger("DeepInd::NeuralNetwork")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetwork":
        """
        训练神经网络

        Args:
            X: 输入特征，shape (n_samples, n_features)
            y: 目标值，shape (n_samples,) 或 (n_samples, n_objectives)

        Returns:
            self
        """
        self.logger.info("NeuralNetwork: 开始训练")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # 保存历史数据（用于不确定性估计）
        self._X_history = X.copy()

        # 数据标准化
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0)
        self._X_std[self._X_std == 0] = 1.0  # 防止除零
        X_norm = (X - self._X_mean) / self._X_std

        # 保存原始y用于不确定性估计
        self._y_history_orig = y.copy() if y.ndim > 1 else y.copy()

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._y_history = y.copy()
        self._y_mean = y.mean(axis=0)
        self._y_std = y.std(axis=0)
        self._y_std[self._y_std == 0] = 1.0
        y_norm = (y - self._y_mean) / self._y_std

        # 确保输出维度正确
        output_dim = y.shape[1] if y.ndim > 1 else 1

        # 初始化模型
        self._model = MLPRegressor(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            activation=self.activation,
            dropout=self.dropout,
        ).to(self._device)

        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()

        # 转换为 tensor
        X_tensor = torch.FloatTensor(X_norm).to(self._device)
        y_tensor = torch.FloatTensor(y_norm).to(self._device)

        # 训练循环
        self._model.train()
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                self._optimizer.zero_grad()
                y_pred = self._model(batch_X)
                loss = loss_fn(y_pred, batch_y)
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 100 == 0:
                avg_loss = epoch_loss / len(loader)
                self.logger.debug(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

        self._is_fitted = True
        self.logger.info(f"NeuralNetwork: 训练完成，输出维度: {output_dim}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 输入特征，shape (n_samples, n_features)

        Returns:
            预测值，shape (n_samples,) 或 (n_samples, n_objectives)
        """
        if not self._is_fitted:
            raise RuntimeError("NeuralNetwork 未训练，请先调用 fit()")

        self._model.eval()
        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_norm = (X - self._X_mean) / self._X_std
        X_tensor = torch.FloatTensor(X_norm).to(self._device)

        with torch.no_grad():
            y_pred = self._model(X_tensor).cpu().numpy()

        # 反标准化
        y_pred = y_pred * self._y_std + self._y_mean

        if y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()

        return y_pred

    def gradient(self, X: np.ndarray) -> np.ndarray:
        """
        计算预测对输入的梯度

        Args:
            X: 输入特征，shape (n_samples, n_features)

        Returns:
            梯度，shape (n_samples, n_features)
        """
        if not self._is_fitted:
            raise RuntimeError("NeuralNetwork 未训练，请先调用 fit()")

        self._model.eval()
        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_norm = (X - self._X_mean) / self._X_std
        X_tensor = torch.FloatTensor(X_norm).to(self._device)
        X_tensor.requires_grad_(True)

        y_pred = self._model(X_tensor)

        # 对每个输出分别计算梯度
        grads = []
        for i in range(y_pred.shape[1]):
            grad = torch.autograd.grad(
                outputs=y_pred[:, i],
                inputs=X_tensor,
                grad_outputs=torch.ones_like(y_pred[:, i]),
                retain_graph=True,
            )[0]
            # 链式法则：反标准化（先转CPU再与numpy运算）
            grad = grad.cpu() * (self._y_std / self._X_std)
            grads.append(grad.numpy())

        if len(grads) == 1:
            return grads[0]
        return np.stack(grads, axis=0)

    def update(self, X_new: np.ndarray, y_new: np.ndarray) -> "NeuralNetwork":
        """
        增量更新（通过继续训练）

        Args:
            X_new: 新增数据特征
            y_new: 新增目标值

        Returns:
            self
        """
        self.logger.info("NeuralNetwork: 执行增量更新")

        X_new = np.asarray(X_new, dtype=np.float32)
        y_new = np.asarray(y_new, dtype=np.float32)

        if y_new.ndim == 1:
            y_new = y_new.reshape(-1, 1)

        # 标准化新数据
        X_new_norm = (X_new - self._X_mean) / self._X_std
        y_new_norm = (y_new - self._y_mean) / self._y_std

        X_tensor = torch.FloatTensor(X_new_norm).to(self._device)
        y_tensor = torch.FloatTensor(y_new_norm).to(self._device)

        self._model.train()
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        loss_fn = nn.MSELoss()
        for batch_X, batch_y in loader:
            self._optimizer.zero_grad()
            y_pred = self._model(batch_X)
            loss = loss_fn(y_pred, batch_y)
            loss.backward()
            self._optimizer.step()

        self.logger.info("NeuralNetwork: 增量更新完成")
        return self

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
            raise RuntimeError("NeuralNetwork 未训练，请先调用 fit()")

        y_pred = self.predict(X)

        # 使用训练时的残差计算不确定性估计
        if self._X_history is not None and self._y_history is not None:
            y_train_pred = self.predict(self._X_history)
            # 使用原始形状的y_history进行残差计算
            y_history = self._y_history_orig
            if y_history.ndim == 1:
                y_history = y_history.reshape(-1, 1)
            residuals = y_history - y_train_pred.reshape(-1, 1)
            # 计算每个输出的残差标准差
            residual_std = np.std(residuals, axis=0)
            uncertainty = np.ones_like(y_pred) * residual_std
            return y_pred, uncertainty

        uncertainty = np.ones_like(y_pred) * 0.1
        return y_pred, uncertainty

    @property
    def is_differentiable(self) -> bool:
        """神经网络可微"""
        return True

    @property
    def n_output(self) -> int:
        """输出维度"""
        if self._model is not None:
            return self._model.output_dim
        return 0