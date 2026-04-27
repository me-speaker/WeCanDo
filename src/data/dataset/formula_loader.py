"""
DeepInd 数据加载器 - 配方数据
用于加载 CSV 格式的化工配方数据
"""

import os
import numpy as np

from ...core.base import BaseDataLoader
from ...core.registry import register_dataloader
from ...utils.logger import get_logger


@register_dataloader("FormulaDataLoader")
class FormulaDataLoader(BaseDataLoader):
    """
    树脂配方数据加载器

    数据格式:
    - CSV 文件
    - 第一行为列名（可选）
    - 前 n_vars 列为决策变量
    - 后 n_objectives 列为目标变量

    Args:
        path: 数据文件路径
        delimiter: 分隔符，默认 ","
        n_objectives: 目标变量数量，默认 2
        skip_header: 是否跳过表头，默认 True
    """

    def __init__(
        self,
        path: str,
        delimiter: str = ",",
        n_objectives: int = 2,
        skip_header: bool = True,
    ):
        super().__init__()
        self.path = path
        self.delimiter = delimiter
        self.n_objectives = n_objectives
        self.skip_header = skip_header

        self.logger = get_logger("DeepInd::DataLoader")

        # 预处理参数（fit时计算）
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def load(self) -> tuple:
        """
        加载数据

        Returns:
            X: 决策变量，shape (n_samples, n_vars)
            y: 目标变量，shape (n_samples, n_objectives)
        """
        self.logger.info(f"DataLoader: 加载数据 from {self.path}")

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"数据文件不存在: {self.path}")

        # 加载 CSV
        if self.skip_header:
            data = np.genfromtxt(self.path, delimiter=self.delimiter, skip_header=1)
        else:
            data = np.genfromtxt(self.path, delimiter=self.delimiter)

        # 分割决策变量和目标变量
        n_vars = data.shape[1] - self.n_objectives
        X = data[:, :n_vars]
        y = data[:, n_vars:]

        self.logger.info(f"DataLoader: 数据加载完成, 共 {X.shape[0]} 条记录")
        self.logger.debug(f"  - 决策变量维度: {X.shape[1]}")
        self.logger.debug(f"  - 目标变量维度: {y.shape[1]}")

        return X, y

    def preprocess(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        标准化预处理

        使用 Z-score 标准化:
        X_std = (X - mean) / std
        y_std = (y - mean) / std

        Args:
            X: 决策变量
            y: 目标变量

        Returns:
            标准化后的 X, y
        """
        self.logger.info("DataLoader: 执行标准化预处理")

        # 决策变量标准化
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        # 避免除零
        self.X_std[self.X_std == 0] = 1.0

        X_normalized = (X - self.X_mean) / self.X_std

        # 目标变量标准化
        self.y_mean = y.mean(axis=0)
        self.y_std = y.std(axis=0)
        self.y_std[self.y_std == 0] = 1.0

        y_normalized = (y - self.y_mean) / self.y_std

        self.logger.debug(f"  - X 均值: {self.X_mean.tolist()}")
        self.logger.debug(f"  - y 均值: {self.y_mean.tolist()}")

        return X_normalized, y_normalized

    def inverse_transform_X(self, X_normalized: np.ndarray) -> np.ndarray:
        """
        反标准化（将标准化后的数据还原）

        Args:
            X_normalized: 标准化后的决策变量

        Returns:
            原始尺度的决策变量
        """
        if self.X_mean is None or self.X_std is None:
            raise RuntimeError("请先调用 preprocess()")
        return X_normalized * self.X_std + self.X_mean

    def inverse_transform_y(self, y_normalized: np.ndarray) -> np.ndarray:
        """
        反标准化（将标准化后的数据还原）

        Args:
            y_normalized: 标准化后的目标变量

        Returns:
            原始尺度的目标变量
        """
        if self.y_mean is None or self.y_std is None:
            raise RuntimeError("请先调用 preprocess()")
        return y_normalized * self.y_std + self.y_mean
