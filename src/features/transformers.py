"""
DeepInd 特征工程模块 - 特征编码与转换
提供标准化、归一化、编码等特征转换功能
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
)

from .base import BaseFeatureTransformer, register_feature_transformer


@register_feature_transformer("StandardScaler")
class StandardScalerTransformer(BaseFeatureTransformer):
    """
    标准化转换器 (Z-score)
    将特征转换为均值为0，标准差为1
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - columns: 要转换的列，默认全部
        """
        super().__init__(config)
        self.columns = self.config.get("columns", None)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "StandardScalerTransformer":
        """拟合"""
        self.scaler_ = StandardScaler()
        cols = self.columns if self.columns else X.columns.tolist()
        self.scaler_.fit(X[cols].values)

        self.metadata = {
            "type": "StandardScaler",
            "columns": cols,
            "mean": self.scaler_.mean_.tolist(),
            "scale": self.scaler_.scale_.tolist(),
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换"""
        result = X.copy()
        cols = self.columns if self.columns else X.columns.tolist()
        result[cols] = self.scaler_.transform(X[cols].values)
        return result


@register_feature_transformer("MinMaxScaler")
class MinMaxScalerTransformer(BaseFeatureTransformer):
    """
    最小最大缩放转换器
    将特征缩放到指定范围
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - columns: 要转换的列
                - feature_range: 目标范围，默认 (0, 1)
        """
        super().__init__(config)
        self.columns = self.config.get("columns", None)
        self.feature_range = self.config.get("feature_range", (0, 1))

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "MinMaxScalerTransformer":
        """拟合"""
        self.scaler_ = MinMaxScaler(feature_range=self.feature_range)
        cols = self.columns if self.columns else X.columns.tolist()
        self.scaler_.fit(X[cols].values)

        self.metadata = {
            "type": "MinMaxScaler",
            "columns": cols,
            "feature_range": self.feature_range,
            "min": self.scaler_.min_.tolist(),
            "scale": self.scaler_.scale_.tolist(),
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换"""
        result = X.copy()
        cols = self.columns if self.columns else X.columns.tolist()
        result[cols] = self.scaler_.transform(X[cols].values)
        return result


@register_feature_transformer("RobustScaler")
class RobustScalerTransformer(BaseFeatureTransformer):
    """
    鲁棒缩放转换器
    使用中位数和四分位距，对异常值更鲁棒
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - columns: 要转换的列
        """
        super().__init__(config)
        self.columns = self.config.get("columns", None)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "RobustScalerTransformer":
        """拟合"""
        self.scaler_ = RobustScaler()
        cols = self.columns if self.columns else X.columns.tolist()
        self.scaler_.fit(X[cols].values)

        self.metadata = {
            "type": "RobustScaler",
            "columns": cols,
            "center": self.scaler_.center_.tolist(),
            "scale": self.scaler_.scale_.tolist(),
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换"""
        result = X.copy()
        cols = self.columns if self.columns else X.columns.tolist()
        result[cols] = self.scaler_.transform(X[cols].values)
        return result


@register_feature_transformer("QuantileTransformer")
class QuantileTransformer(BaseFeatureTransformer):
    """
    分位数转换器
    将特征映射到均匀或正态分布
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - columns: 要转换的列
                - output_distribution: 输出分布 "uniform" 或 "normal"
                - n_quantiles: 分位数数量
        """
        super().__init__(config)
        self.columns = self.config.get("columns", None)
        self.output_distribution = self.config.get("output_distribution", "uniform")
        self.n_quantiles = self.config.get("n_quantiles", 1000)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "QuantileTransformer":
        """拟合"""
        self.scaler_ = QuantileTransformer(
            output_distribution=self.output_distribution,
            n_quantiles=self.n_quantiles,
        )
        cols = self.columns if self.columns else X.columns.tolist()
        self.scaler_.fit(X[cols].values)

        self.metadata = {
            "type": "QuantileTransformer",
            "columns": cols,
            "output_distribution": self.output_distribution,
            "n_quantiles": self.n_quantiles,
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换"""
        result = X.copy()
        cols = self.columns if self.columns else X.columns.tolist()
        result[cols] = self.scaler_.transform(X[cols].values)
        return result


@register_feature_transformer("PowerTransformer")
class PowerTransformer(BaseFeatureTransformer):
    """
    幂转换器
    使用Yeo-Johnson或Box-Cox变换使数据更接近正态分布
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - columns: 要转换的列
                - method: 变换方法 "yeo-johnson" 或 "box-cox"
        """
        super().__init__(config)
        self.columns = self.config.get("columns", None)
        self.method = self.config.get("method", "yeo-johnson")

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "PowerTransformer":
        """拟合"""
        self.transformer_ = PowerTransformer(method=self.method)
        cols = self.columns if self.columns else X.columns.tolist()
        self.transformer_.fit(X[cols].values)

        self.metadata = {
            "type": "PowerTransformer",
            "columns": cols,
            "method": self.method,
            "lambdas": self.transformer_.lambdas_.tolist(),
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换"""
        result = X.copy()
        cols = self.columns if self.columns else X.columns.tolist()
        result[cols] = self.transformer_.transform(X[cols].values)
        return result


@register_feature_transformer("OneHotEncoder")
class OneHotEncoderTransformer(BaseFeatureTransformer):
    """
    独热编码器
    将分类变量转换为二进制向量
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - columns: 要编码的分类列
                - drop: 是否删除原始列 "first", "if_binary", 或 None
        """
        super().__init__(config)
        self.columns = self.config.get("columns", None)
        self.drop = self.config.get("drop", None)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "OneHotEncoderTransformer":
        """拟合"""
        cols = self.columns if self.columns else X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.encoder_ = OneHotEncoder(drop=self.drop, sparse_output=False, handle_unknown="ignore")
        self.encoder_.fit(X[cols].values)

        self.metadata = {
            "type": "OneHotEncoder",
            "columns": cols,
            "categories": self.encoder_.categories_.tolist(),
            "drop": self.drop,
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """编码"""
        cols = self.columns if self.columns else X.select_dtypes(include=["object", "category"]).columns.tolist()

        # 编码分类列
        encoded = self.encoder_.transform(X[cols].values)
        encoded_cols = self.encoder_.get_feature_names_out(cols)

        result = X.drop(columns=cols).copy()
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=X.index)
        result = pd.concat([result, encoded_df], axis=1)

        return result


@register_feature_transformer("OrdinalEncoder")
class OrdinalEncoderTransformer(BaseFeatureTransformer):
    """
    序数编码器
    将分类变量转换为有序整数
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - columns: 要编码的分类列
                - categories: 类别顺序 {"列名": [类别列表]}
        """
        super().__init__(config)
        self.columns = self.config.get("columns", None)
        self.custom_categories = self.config.get("categories", None)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "OrdinalEncoderTransformer":
        """拟合"""
        cols = self.columns if self.columns else X.select_dtypes(include=["object", "category"]).columns.tolist()

        if self.custom_categories:
            self.encoder_ = OrdinalEncoder(categories=[self.custom_categories[c] for c in cols])
        else:
            self.encoder_ = OrdinalEncoder()

        self.encoder_.fit(X[cols].values)

        self.metadata = {
            "type": "OrdinalEncoder",
            "columns": cols,
            "categories": self.encoder_.categories_.tolist() if not self.custom_categories else self.custom_categories,
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """编码"""
        result = X.copy()
        cols = self.columns if self.columns else X.select_dtypes(include=["object", "category"]).columns.tolist()
        result[cols] = self.encoder_.transform(X[cols].values)
        return result


@register_feature_transformer("PolynomialFeatures")
class PolynomialFeaturesTransformer(BaseFeatureTransformer):
    """
    多项式特征生成器
    生成交互项和高阶项
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - degree: 多项式阶数，默认 2
                - interaction_only: 是否仅生成交互项
                - include_bias: 是否包含常数项
        """
        super().__init__(config)
        self.degree = self.config.get("degree", 2)
        self.interaction_only = self.config.get("interaction_only", False)
        self.include_bias = self.config.get("include_bias", False)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "PolynomialFeaturesTransformer":
        """拟合"""
        self.poly_ = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )
        self.poly_.fit(X.values)

        self.metadata = {
            "type": "PolynomialFeatures",
            "degree": self.degree,
            "interaction_only": self.interaction_only,
            "n_features": self.poly_.n_output_features_,
            "input_features": X.columns.tolist(),
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """生成多项式特征"""
        transformed = self.poly_.transform(X.values)
        feature_names = self.poly_.get_feature_names_out(X.columns)

        result = pd.DataFrame(transformed, columns=feature_names, index=X.index)
        return result


@register_feature_transformer("LogTransformer")
class LogTransformer(BaseFeatureTransformer):
    """
    对数转换器
    对特征应用对数变换（处理偏斜分布）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - columns: 要转换的列
                - base: 对数底 "e", "2", "10"
        """
        super().__init__(config)
        self.columns = self.config.get("columns", None)
        self.base = self.config.get("base", "e")

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "LogTransformer":
        """拟合（无需拟合）"""
        cols = self.columns if self.columns else X.columns.tolist()

        # 检查是否有非正值
        for col in cols:
            if (X[col] <= 0).any():
                self.logger.warning(f"列 {col} 包含非正值，对数变换可能失败")

        self.metadata = {
            "type": "LogTransformer",
            "columns": cols,
            "base": self.base,
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换"""
        result = X.copy()
        cols = self.columns if self.columns else X.columns.tolist()

        for col in cols:
            if self.base == "e":
                result[f"log_{col}"] = np.log(X[col].values + 1e-10)
            elif self.base == "2":
                result[f"log2_{col}"] = np.log2(X[col].values + 1e-10)
            elif self.base == "10":
                result[f"log10_{col}"] = np.log10(X[col].values + 1e-10)

        return result


@register_feature_transformer("Interaction")
class InteractionTransformer(BaseFeatureTransformer):
    """
    交互特征生成器
    生成指定特征对的乘积比
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - pairs: 交互特征对列表 [(col1, col2), ...]
                - operations: 运算列表 "multiply", "divide", "add", "subtract"
        """
        super().__init__(config)
        self.pairs = self.config.get("pairs", [])
        self.operations = self.config.get("operations", ["multiply"])

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "InteractionTransformer":
        """拟合（无需拟合）"""
        # 验证特征对是否存在
        for col1, col2 in self.pairs:
            if col1 not in X.columns:
                self.logger.warning(f"列 {col1} 不存在")
            if col2 not in X.columns:
                self.logger.warning(f"列 {col2} 不存在")

        self.metadata = {
            "type": "Interaction",
            "pairs": self.pairs,
            "operations": self.operations,
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """生成交互特征"""
        result = X.copy()

        for col1, col2 in self.pairs:
            if col1 not in X.columns or col2 not in X.columns:
                continue

            x1 = X[col1].values
            x2 = X[col2].values

            for op in self.operations:
                if op == "multiply":
                    result[f"{col1}_x_{col2}"] = x1 * x2
                elif op == "divide":
                    result[f"{col1}_div_{col2}"] = x1 / (x2 + 1e-10)
                elif op == "add":
                    result[f"{col1}_plus_{col2}"] = x1 + x2
                elif op == "subtract":
                    result[f"{col1}_minus_{col2}"] = x1 - x2

        return result