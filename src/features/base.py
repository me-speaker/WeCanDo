"""
DeepInd 特征工程模块 - 基础类
提供特征工程的抽象基类和注册机制
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from ..utils.logger import get_logger


class BaseFeatureTransformer(ABC):
    """
    特征转换器抽象基类
    所有特征转换器需继承此类并实现 transform 方法
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数字典
        """
        self.config = config or {}
        self.logger = get_logger(f"DeepInd::Features::{self.__class__.__name__}")
        self._is_fitted = False
        self.metadata = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "BaseFeatureTransformer":
        """
        拟合转换器

        Args:
            X: 输入特征 DataFrame
            y: 可选的标签数据

        Returns:
            self
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        执行特征转换

        Args:
            X: 输入特征 DataFrame

        Returns:
            转换后的 DataFrame
        """
        pass

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        拟合并转换

        Args:
            X: 输入特征 DataFrame
            y: 可选的标签数据

        Returns:
            转换后的 DataFrame
        """
        return self.fit(X, y).transform(X)

    @property
    def is_fitted(self) -> bool:
        """检查是否已拟合"""
        return self._is_fitted

    def get_metadata(self) -> Dict[str, Any]:
        """
        获取特征元数据

        Returns:
            包含特征信息的字典
        """
        return self.metadata


class FeaturePipeline:
    """
    特征工程流水线
    将多个特征转换器串联起来
    """

    def __init__(self, steps: List[BaseFeatureTransformer]):
        """
        Args:
            steps: 特征转换器列表
        """
        self.steps = steps
        self.logger = get_logger("DeepInd::Features::Pipeline")

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "FeaturePipeline":
        """
        拟流水线上所有转换器

        Args:
            X: 输入特征 DataFrame
            y: 可选的标签数据

        Returns:
            self
        """
        X_transformed = X
        for i, step in enumerate(self.steps):
            self.logger.info(f"Pipeline step {i+1}/{len(self.steps)}: {step.__class__.__name__}")
            step.fit(X_transformed, y)
            X_transformed = step.transform(X_transformed)
            self.logger.debug(f"  - 输出特征数: {X_transformed.shape[1]}")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        应用流水线上所有转换器

        Args:
            X: 输入特征 DataFrame

        Returns:
            转换后的 DataFrame
        """
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")

        X_transformed = X
        for step in self.steps:
            X_transformed = step.transform(X_transformed)

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        拟合并转换

        Args:
            X: 输入特征 DataFrame
            y: 可选的标签数据

        Returns:
            转换后的 DataFrame
        """
        return self.fit(X, y).transform(X)

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """
        获取所有步骤的元数据

        Returns:
            元数据列表
        """
        return [step.get_metadata() for step in self.steps]


def register_feature_transformer(name: str):
    """
    特征转换器注册装饰器

    Args:
        name: 转换器名称

    Returns:
        装饰器函数
    """
    def decorator(cls):
        if not hasattr(FeatureRegistry, "_registry"):
            FeatureRegistry._registry = {}
        FeatureRegistry._registry[name] = cls
        return cls
    return decorator


class FeatureRegistry:
    """
    特征转换器注册表
    """
    _registry: Dict[str, type] = {}

    @classmethod
    def get(cls, name: str) -> type:
        """获取注册的转换器类"""
        if name not in cls._registry:
            raise KeyError(f"Unknown feature transformer: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> List[str]:
        """列出所有可用的转换器"""
        return list(cls._registry.keys())