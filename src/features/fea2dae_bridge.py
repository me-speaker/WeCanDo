"""
DeepInd 特征工程模块 - DAE桥接器
连接 DataFrame 风格的 FeaturePipeline 与 numpy 风格的 DAE 模型
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

from .base import BaseFeatureTransformer, register_feature_transformer


@register_feature_transformer("DAEBridge")
class DAEFeatureBridge(BaseFeatureTransformer):
    """
    DAE 特征桥接器
    将 DataFrame 格式的特征转换为 DAE 所需的 numpy 格式，
    并在 DAE 编码后转换回 DataFrame 格式

    用法:
        bridge = DAEFeatureBridge(config={
            "dae_model": dae_instance,
            "encoding_dim": 16
        })
        X_encoded = bridge.fit_transform(X_df)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - dae_model: DAE 模型实例
                - encoding_dim: 编码维度
                - preprocess: 是否在编码前进行标准化预处理（默认 True）
        """
        super().__init__(config)
        self.dae_model = self.config.get("dae_model", None)
        self.encoding_dim = self.config.get("encoding_dim", 16)
        self.preprocess = self.config.get("preprocess", True)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "DAEFeatureBridge":
        """
        拟合桥接器（实际上调用 DAE 的 fit_transform）

        Args:
            X: 输入特征 DataFrame
            y: 忽略

        Returns:
            self
        """
        if self.dae_model is None:
            raise ValueError("DAE 模型未设置，请在 config 中提供 dae_model")

        # 转换 DataFrame 为 numpy
        X_numpy = X.values

        # 预处理：标准化
        if self.preprocess:
            self.mean_ = X_numpy.mean(axis=0)
            self.std_ = X_numpy.std(axis=0)
            self.std_[self.std_ == 0] = 1.0
            X_numpy = (X_numpy - self.mean_) / self.std_

        # DAE 拟合并编码
        self.dae_model.fit_transform(X_numpy)

        self.metadata = {
            "type": "DAEBridge",
            "encoding_dim": self.encoding_dim,
            "preprocess": self.preprocess,
            "input_features": list(X.columns),
            "n_input_features": X.shape[1],
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        使用 DAE 编码特征

        Args:
            X: 输入特征 DataFrame

        Returns:
            编码后的特征 DataFrame
        """
        if not self._is_fitted:
            raise RuntimeError("DAEFeatureBridge 必须先拟合")

        X_numpy = X.values

        # 预处理
        if self.preprocess:
            X_numpy = (X_numpy - self.mean_) / self.std_

        # DAE 编码
        X_encoded = self.dae_model.transform(X_numpy)

        # 转换回 DataFrame
        result = pd.DataFrame(
            X_encoded,
            columns=[f"DAE_feat_{i+1}" for i in range(X_encoded.shape[1])],
            index=X.index,
        )

        self.metadata["n_output_features"] = X_encoded.shape[1]
        return result


@register_feature_transformer("DAEPipeline")
class DAEPipelineTransformer(BaseFeatureTransformer):
    """
    DAE 流水线特征转换器
    在流水线中集成 DAE 编码器

    与 DAEFeatureBridge 的区别：
    - DAEFeatureBridge: 独立使用，在流水线末尾将特征编码
    - DAEPipelineTransformer: 可嵌入 FeaturePipeline 中使用
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - encoding_dim: 编码维度
                - noise_factor: DAE 噪声因子
                - epochs: 训练轮数
                - lr: 学习率
        """
        super().__init__(config)
        self.encoding_dim = self.config.get("encoding_dim", 16)
        self.noise_factor = self.config.get("noise_factor", 0.1)
        self.epochs = self.config.get("epochs", 100)
        self.lr = self.config.get("lr", 0.001)
        self._dae_model = None

    def _create_dae_model(self, input_dim: int):
        """创建 DAE 模型"""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch is required for DAEPipelineTransformer")

        class SimpleDAE(nn.Module):
            def __init__(self, input_dim, hidden_dim, noise_factor):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, input_dim),
                    nn.Sigmoid()
                )
                self.noise_factor = noise_factor

            def encode(self, x):
                return self.encoder(x)

            def decode(self, x):
                return self.decoder(x)

            def forward(self, x):
                return self.decoder(self.encode(x))

        return SimpleDAE(input_dim, self.encoding_dim, self.noise_factor)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "DAEPipelineTransformer":
        """拟合 DAE"""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        input_dim = X.shape[1]
        self._dae_model = self._create_dae_model(input_dim)

        # 数据标准化
        self.mean_ = X.values.mean(axis=0)
        self.std_ = X.values.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        X_normalized = (X.values - self.mean_) / self.std_

        # 训练 DAE
        optimizer = torch.optim.Adam(self._dae_model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(torch.FloatTensor(X_normalized))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        self._dae_model.train()
        for epoch in range(self.epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                recon = self._dae_model(batch[0])
                loss = criterion(recon, batch[0])
                loss.backward()
                optimizer.step()

        self._is_fitted = True

        self.metadata = {
            "type": "DAEPipeline",
            "encoding_dim": self.encoding_dim,
            "noise_factor": self.noise_factor,
            "epochs": self.epochs,
            "input_dim": input_dim,
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """编码特征"""
        if not self._is_fitted or self._dae_model is None:
            raise RuntimeError("DAEPipelineTransformer 必须先拟合")

        X_normalized = (X.values - self.mean_) / self.std_
        self._dae_model.eval()

        with torch.no_grad():
            encoded = self._dae_model.encode(torch.FloatTensor(X_normalized)).numpy()

        result = pd.DataFrame(
            encoded,
            columns=[f"DAE_enc_{i+1}" for i in range(encoded.shape[1])],
            index=X.index,
        )

        self.metadata["output_dim"] = encoded.shape[1]
        return result
