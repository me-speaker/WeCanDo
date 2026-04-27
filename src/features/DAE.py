"""
DeepInd 降噪自编码器 (Denoising Autoencoder)
用于特征提取和降维
"""

import torch
import torch.nn as nn
import numpy as np

from ..core.base import BaseFeatureEngine
from ..core.registry import register_feature
from ..utils.logger import get_logger


@register_feature("DAE")
class DenoisingAutoencoder(nn.Module, BaseFeatureEngine):
    """
    降噪自编码器

    结构:
    - 编码器: Linear(input_dim, hidden_dim) -> ReLU
    - 解码器: Linear(hidden_dim, input_dim) -> Sigmoid

    用于:
    - 无监督特征学习
    - 数据去噪
    - 降维

    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度（编码维度）
        noise_factor: 噪声系数（用于去噪训练）
        epochs: 预训练轮数
        lr: 学习率
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        noise_factor: float = 0.1,
        epochs: int = 100,
        lr: float = 0.001,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noise_factor = noise_factor
        self.epochs = epochs
        self.lr = lr

        # 编码器
        self.encoder_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        # 解码器
        self.decoder_layer = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self.logger = get_logger("DeepInd::DAE")

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        """编码"""
        return self.encoder_layer(x)

    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        """解码"""
        return self.decoder_layer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: 编码 -> 解码"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        无监督预训练，学习特征表示

        Args:
            X: 输入数据，shape (n_samples, input_dim)
            y: 忽略（为兼容接口）

        Returns:
            编码后的特征，shape (n_samples, hidden_dim)
        """
        self.logger.info(f"DAE: 开始预训练, epochs={self.epochs}, lr={self.lr}")
        self.logger.debug(f"  - 输入维度: {self.input_dim}")
        self.logger.debug(f"  - 隐藏层维度: {self.hidden_dim}")

        X_tensor = torch.FloatTensor(X)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            # 添加噪声
            noise = torch.randn_like(X_tensor) * self.noise_factor
            X_noisy = X_tensor + noise

            # 重构
            reconstructed = self.forward(X_noisy)

            # 计算损失
            loss = criterion(reconstructed, X_tensor)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 日志输出
            if (epoch + 1) % 20 == 0:
                self.logger.debug(f"  Epoch {epoch+1}/{self.epochs}: loss={loss.item():.6f}")

        self.logger.info(f"DAE: 预训练完成, 最终loss={loss.item():.6f}")

        # 返回编码特征
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        转换新数据（仅编码，不重建）

        Args:
            X: 输入数据

        Returns:
            编码后的特征
        """
        self.eval()
        X_tensor = torch.FloatTensor(X)

        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(0)

        with torch.no_grad():
            features = self.encoder(X_tensor)

        return features.numpy()

    @property
    def feature_dim(self) -> int:
        """输出特征维度"""
        return self.hidden_dim
