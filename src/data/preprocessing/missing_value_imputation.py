"""
DeepInd 数据预处理模块 - 缺失值填充
提供多种缺失值处理方法
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from .base import BaseDataProcessor, register_data_processor


@register_data_processor("MissingValueImputer")
class MissingValueImputer(BaseDataProcessor):
    """
    缺失值填充器
    支持多种填充方法：均值、中位数、众数、KNN、插值等
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - method: 填充方法 "mean", "median", "mode", "knn", "interpolate", "forward_fill", "backward_fill"
                - columns: 要填充的列，默认全部
                - knn_neighbors: KNN 的邻居数（默认 5）
        """
        super().__init__(config)
        self.method = self.config.get("method", "mean")
        self.columns = self.config.get("columns", None)
        self.knn_neighbors = self.config.get("knn_neighbors", 5)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "MissingValueImputer":
        """
        拟合填充器

        Args:
            X: 输入数据 DataFrame
            y: 忽略

        Returns:
            self
        """
        cols = self.columns if self.columns else X.columns.tolist()
        X_subset = X[cols]

        if self.method == "mean":
            self._fit_mean(X_subset)
        elif self.method == "median":
            self._fit_median(X_subset)
        elif self.method == "mode":
            self._fit_mode(X_subset)
        elif self.method == "knn":
            self._fit_knn(X_subset)
        elif self.method == "interpolate":
            self._fit_interpolate(X_subset)
        elif self.method == "forward_fill":
            self._fit_forward_fill(X_subset)
        elif self.method == "backward_fill":
            self._fit_backward_fill(X_subset)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._is_fitted = True
        return self

    def _fit_mean(self, X: pd.DataFrame):
        """均值填充拟合"""
        self.fill_values_ = X.mean().to_dict()
        self.metadata = {
            "type": "MissingValueImputer",
            "method": "mean",
            "columns": X.columns.tolist(),
            "fill_values": self.fill_values_,
            "n_missing_before": int(X.isna().sum().sum()),
        }

    def _fit_median(self, X: pd.DataFrame):
        """中位数填充拟合"""
        self.fill_values_ = X.median().to_dict()
        self.metadata = {
            "type": "MissingValueImputer",
            "method": "median",
            "columns": X.columns.tolist(),
            "fill_values": self.fill_values_,
            "n_missing_before": int(X.isna().sum().sum()),
        }

    def _fit_mode(self, X: pd.DataFrame):
        """众数填充拟合"""
        fill_values = {}
        for col in X.columns:
            mode_vals = X[col].mode()
            if len(mode_vals) > 0:
                fill_values[col] = mode_vals[0]
            else:
                fill_values[col] = 0
        self.fill_values_ = fill_values
        self.metadata = {
            "type": "MissingValueImputer",
            "method": "mode",
            "columns": X.columns.tolist(),
            "fill_values": self.fill_values_,
            "n_missing_before": int(X.isna().sum().sum()),
        }

    def _fit_knn(self, X: pd.DataFrame):
        """KNN 填充拟合"""
        try:
            from sklearn.impute import KNNImputer
        except ImportError:
            raise ImportError("sklearn is required for KNN imputation")

        self.knn_imputer_ = KNNImputer(n_neighbors=self.knn_neighbors)
        self.knn_imputer_.fit(X.values)

        # 计算填充值用于记录
        X_filled = X.fillna(X.mean())
        self.fill_values_ = X_filled.mean().to_dict()

        self.metadata = {
            "type": "MissingValueImputer",
            "method": "knn",
            "columns": X.columns.tolist(),
            "knn_neighbors": self.knn_neighbors,
            "n_missing_before": int(X.isna().sum().sum()),
        }

    def _fit_interpolate(self, X: pd.DataFrame):
        """插值填充拟合"""
        self.metadata = {
            "type": "MissingValueImputer",
            "method": "interpolate",
            "columns": X.columns.tolist(),
            "n_missing_before": int(X.isna().sum().sum()),
        }

    def _fit_forward_fill(self, X: pd.DataFrame):
        """前向填充拟合"""
        self.metadata = {
            "type": "MissingValueImputer",
            "method": "forward_fill",
            "columns": X.columns.tolist(),
            "n_missing_before": int(X.isna().sum().sum()),
        }

    def _fit_backward_fill(self, X: pd.DataFrame):
        """后向填充拟合"""
        self.metadata = {
            "type": "MissingValueImputer",
            "method": "backward_fill",
            "columns": X.columns.tolist(),
            "n_missing_before": int(X.isna().sum().sum()),
        }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        填充缺失值

        Args:
            X: 输入数据 DataFrame

        Returns:
            填充后的 DataFrame
        """
        cols = self.columns if self.columns else X.columns.tolist()
        result = X.copy()

        if self.method in ("mean", "median", "mode"):
            result[cols] = X[cols].fillna(pd.Series(self.fill_values_))
        elif self.method == "knn":
            X_subset = X[cols].values
            X_imputed = self.knn_imputer_.transform(X_subset)
            result[cols] = X_imputed
        elif self.method == "interpolate":
            result[cols] = X[cols].interpolate(method="linear", limit_direction="both")
        elif self.method == "forward_fill":
            result[cols] = X[cols].fillna(method="ffill")
        elif self.method == "backward_fill":
            result[cols] = X[cols].fillna(method="bfill")

        self.metadata["n_missing_after"] = int(result.isna().sum().sum())
        return result
