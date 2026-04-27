"""
DeepInd 特征工程模块 - 特征降维
提供PCA、LDA等降维功能
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .base import BaseFeatureTransformer, register_feature_transformer


@register_feature_transformer("PCA")
class PCATransformer(BaseFeatureTransformer):
    """
    主成分分析降维器
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - n_components: 主成分数量，或 "mle" 或 (0,1) 比例
                - whiten: 是否白化
        """
        super().__init__(config)
        self.n_components = self.config.get("n_components", "mle")
        self.whiten = self.config.get("whiten", False)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "PCATransformer":
        """拟合"""
        self.pca_ = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
        )
        self.pca_.fit(X.values)

        # 记录方差解释比
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        # 记录主成分载荷
        loadings = pd.DataFrame(
            self.pca_.components_.T,
            columns=[f"PC{i+1}" for i in range(self.pca_.n_components_)],
            index=X.columns,
        )
        self.loadings_ = loadings

        self.metadata = {
            "type": "PCA",
            "n_components": self.pca_.n_components_,
            "explained_variance_ratio": self.explained_variance_ratio_.tolist(),
            "cumulative_variance_ratio": self.cumulative_variance_ratio_.tolist(),
            "loadings": self.loadings_.to_dict(),
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """降维"""
        transformed = self.pca_.transform(X.values)
        n_components = transformed.shape[1]
        result = pd.DataFrame(
            transformed,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=X.index,
        )
        return result

    def get_n_components_for_variance(self, threshold: float = 0.95) -> int:
        """
        获取达到指定方差解释比所需的主成分数

        Args:
            threshold: 方差解释比阈值

        Returns:
            所需主成分数
        """
        n_components = np.argmax(self.cumulative_variance_ratio_ >= threshold) + 1
        return n_components


@register_feature_transformer("KernelPCA")
class KernelPCATransformer(BaseFeatureTransformer):
    """
    核主成分分析降维器
    用于非线性降维
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - n_components: 主成分数量
                - kernel: 核函数类型 "linear", "rbf", "poly", "sigmoid"
                - gamma: RBF核参数
                - degree: 多项式核参数
        """
        super().__init__(config)
        self.n_components = self.config.get("n_components", 2)
        self.kernel = self.config.get("kernel", "rbf")
        self.gamma = self.config.get("gamma", None)
        self.degree = self.config.get("degree", 3)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "KernelPCATransformer":
        """拟合"""
        self.kpca_ = KernelPCA(
            n_components=self.n_components,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
        )
        self.kpca_.fit(X.values)

        self.metadata = {
            "type": "KernelPCA",
            "n_components": self.n_components,
            "kernel": self.kernel,
            "gamma": self.gamma,
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """降维"""
        transformed = self.kpca_.transform(X.values)
        result = pd.DataFrame(
            transformed,
            columns=[f"KPC{i+1}" for i in range(self.n_components)],
            index=X.index,
        )
        return result


@register_feature_transformer("IncrementalPCA")
class IncrementalPCATransformer(BaseFeatureTransformer):
    """
    增量主成分分析降维器
    适合大规模数据
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - n_components: 主成分数量
                - batch_size: 批次大小
        """
        super().__init__(config)
        self.n_components = self.config.get("n_components", 2)
        self.batch_size = self.config.get("batch_size", 100)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "IncrementalPCATransformer":
        """拟合"""
        self.ipca_ = IncrementalPCA(
            n_components=self.n_components,
            batch_size=self.batch_size,
        )
        self.ipca_.fit(X.values)

        self.explained_variance_ratio_ = self.ipca_.explained_variance_ratio_
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        self.metadata = {
            "type": "IncrementalPCA",
            "n_components": self.n_components,
            "batch_size": self.batch_size,
            "explained_variance_ratio": self.explained_variance_ratio_.tolist(),
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """降维"""
        transformed = self.ipca_.transform(X.values)
        result = pd.DataFrame(
            transformed,
            columns=[f"PC{i+1}" for i in range(self.n_components)],
            index=X.index,
        )
        return result


@register_feature_transformer("TruncatedSVD")
class TruncatedSVDTransformer(BaseFeatureTransformer):
    """
    截断SVD降维器
    适合稀疏矩阵
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - n_components: 组件数量
        """
        super().__init__(config)
        self.n_components = self.config.get("n_components", 2)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "TruncatedSVDTransformer":
        """拟合"""
        self.svd_ = TruncatedSVD(n_components=self.n_components)
        self.svd_.fit(X.values)

        self.explained_variance_ratio_ = self.svd_.explained_variance_ratio_
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        self.metadata = {
            "type": "TruncatedSVD",
            "n_components": self.n_components,
            "explained_variance_ratio": self.explained_variance_ratio_.tolist(),
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """降维"""
        transformed = self.svd_.transform(X.values)
        result = pd.DataFrame(
            transformed,
            columns=[f"SVD{i+1}" for i in range(self.n_components)],
            index=X.index,
        )
        return result


@register_feature_transformer("LDA")
class LDATransformer(BaseFeatureTransformer):
    """
    线性判别分析降维器
    监督降维方法
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - n_components: 判别分量数量（最大为类别数-1）
        """
        super().__init__(config)
        self.n_components = self.config.get("n_components", None)

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series]] = None) -> "LDATransformer":
        """拟合"""
        if y is None:
            raise ValueError("LDA 需要目标变量 y")

        y_values = y.values.ravel() if isinstance(y, (pd.DataFrame, pd.Series)) else y

        # 自动确定最大组件数
        n_classes = len(np.unique(y_values))
        max_components = min(self.n_components or n_classes - 1, n_classes - 1)

        self.lda_ = LinearDiscriminantAnalysis(n_components=max_components)
        self.lda_.fit(X.values, y_values)

        # 记录判别函数方差比
        self.explained_variance_ratio_ = self.lda_.explained_variance_ratio_

        self.metadata = {
            "type": "LDA",
            "n_components": self.lda_.n_components_,
            "explained_variance_ratio": self.explained_variance_ratio_.tolist() if self.explained_variance_ratio_ is not None else None,
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """降维"""
        transformed = self.lda_.transform(X.values)
        result = pd.DataFrame(
            transformed,
            columns=[f"LDA{i+1}" for i in range(self.lda_.n_components_)],
            index=X.index,
        )
        return result


@register_feature_transformer("Autoencoder")
class AutoencoderReducer(BaseFeatureTransformer):
    """
    自编码器降维器
    使用深度自编码器进行非线性降维
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - encoding_dim: 编码维度
                - hidden_layers: 编码器隐藏层结构
                - epochs: 训练轮数
                - batch_size: 批次大小
        """
        super().__init__(config)
        self.encoding_dim = self.config.get("encoding_dim", 2)
        self.hidden_layers = self.config.get("hidden_layers", [64, 32])
        self.epochs = self.config.get("epochs", 100)
        self.batch_size = self.config.get("batch_size", 32)
        self.lr = self.config.get("lr", 0.001)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "AutoencoderReducer":
        """拟合"""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            self.logger.error("AutoencoderReducer 需要 PyTorch，请先安装: pip install torch")
            self._is_fitted = False
            return self

        # 数据标准化
        self.mean_ = X.values.mean(axis=0)
        self.std_ = X.values.std(axis=0)
        self.std_[self.std_ == 0] = 1.0

        X_normalized = (X.values - self.mean_) / self.std_

        # 定义自编码器
        input_dim = X.shape[1]

        class Autoencoder(nn.Module):
            def __init__(self, input_dim, hidden_layers, encoding_dim):
                super().__init__()
                layers = []
                prev_dim = input_dim
                for h_dim in hidden_layers:
                    layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
                    prev_dim = h_dim
                self.encoder = nn.Sequential(*layers)
                self.bottleneck = nn.Linear(prev_dim, encoding_dim)
                layers = []
                prev_dim = encoding_dim
                for h_dim in reversed(hidden_layers):
                    layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
                    prev_dim = h_dim
                layers.append(nn.Linear(prev_dim, input_dim))
                self.decoder = nn.Sequential(*layers)

            def encode(self, x):
                return self.bottleneck(self.encoder(x))

            def forward(self, x):
                return self.decoder(self.encode(x))

        self.model_ = Autoencoder(input_dim, self.hidden_layers, self.encoding_dim)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # 训练
        dataset = TensorDataset(torch.FloatTensor(X_normalized))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                recon = self.model_(batch[0])
                loss = criterion(recon, batch[0])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 20 == 0:
                self.logger.debug(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")

        self.model_.eval()
        self._is_fitted = True

        self.metadata = {
            "type": "Autoencoder",
            "encoding_dim": self.encoding_dim,
            "hidden_layers": self.hidden_layers,
            "epochs": self.epochs,
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """降维"""
        if not self._is_fitted:
            raise RuntimeError("AutoencoderReducer 尚未拟合")

        X_normalized = (X.values - self.mean_) / self.std_
        self.model_.eval()
        with torch.no_grad():
            encoded = self.model_.encode(torch.FloatTensor(X_normalized)).numpy()

        result = pd.DataFrame(
            encoded,
            columns=[f"AED{i+1}" for i in range(self.encoding_dim)],
            index=X.index,
        )
        return result