"""
DeepInd 数据预处理模块 - 异常值检测
提供多种异常值检测方法
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats

from .base import BaseDataProcessor, register_data_processor


@register_data_processor("OutlierDetector")
class OutlierDetector(BaseDataProcessor):
    """
    异常值检测器
    支持多种检测方法：Z-score、IQR、四分位、Mahalanobis距离等
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - method: 检测方法 "zscore", "iqr", "isolation_forest", "dbscan"
                - threshold: 阈值（zscore: 3, iqr: 1.5）
                - columns: 要检测的列，默认全部
        """
        super().__init__(config)
        self.method = self.config.get("method", "zscore")
        self.threshold = self.config.get("threshold", None)
        self.columns = self.config.get("columns", None)
        self._outlier_mask = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "OutlierDetector":
        """
        拟合检测器

        Args:
            X: 输入数据 DataFrame
            y: 忽略

        Returns:
            self
        """
        cols = self.columns if self.columns else X.columns.tolist()
        X_subset = X[cols]

        if self.method == "zscore":
            self._fit_zscore(X_subset)
        elif self.method == "iqr":
            self._fit_iqr(X_subset)
        elif self.method == "isolation_forest":
            self._fit_isolation_forest(X_subset)
        elif self.method == "dbscan":
            self._fit_dbscan(X_subset)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._is_fitted = True
        return self

    def _fit_zscore(self, X: pd.DataFrame):
        """Z-score 方法拟合"""
        threshold = self.threshold if self.threshold else 3.0
        self.z_thresholds_ = threshold
        self.z_means_ = X.mean().values
        self.z_stds_ = X.std().values

        z_scores = np.abs((X.values - self.z_means_) / (self.z_stds_ + 1e-10))
        self._outlier_mask = z_scores > threshold

        self.metadata = {
            "type": "OutlierDetector",
            "method": "zscore",
            "threshold": threshold,
            "columns": X.columns.tolist(),
            "n_outliers": int(self._outlier_mask.sum()),
            "outlier_ratio": float(self._outlier_mask.sum() / len(X)),
        }

    def _fit_iqr(self, X: pd.DataFrame):
        """IQR 方法拟合"""
        multiplier = self.threshold if self.threshold else 1.5
        self.iqr_multiplier_ = multiplier

        q1 = X.quantile(0.25).values
        q3 = X.quantile(0.75).values
        self.iqr_q1_ = q1
        self.iqr_q3_ = q3
        iqr = q3 - q1
        self.iqr_lower_ = q1 - multiplier * iqr
        self.iqr_upper_ = q3 + multiplier * iqr

        X_arr = X.values
        self._outlier_mask = (X_arr < self.iqr_lower_) | (X_arr > self.iqr_upper_)

        self.metadata = {
            "type": "OutlierDetector",
            "method": "iqr",
            "threshold": multiplier,
            "columns": X.columns.tolist(),
            "n_outliers": int(self._outlier_mask.sum()),
            "outlier_ratio": float(self._outlier_mask.sum() / len(X)),
        }

    def _fit_isolation_forest(self, X: pd.DataFrame):
        """Isolation Forest 方法拟合"""
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError("sklearn is required for IsolationForest method")

        contamination = self.threshold if self.threshold else 0.1
        self.isolation_forest_ = IsolationForest(contamination=contamination, random_state=42)
        self.isolation_forest_.fit(X.values)

        predictions = self.isolation_forest_.predict(X.values)
        self._outlier_mask = predictions == -1

        self.metadata = {
            "type": "OutlierDetector",
            "method": "isolation_forest",
            "threshold": contamination,
            "columns": X.columns.tolist(),
            "n_outliers": int(self._outlier_mask.sum()),
            "outlier_ratio": float(self._outlier_mask.sum() / len(X)),
        }

    def _fit_dbscan(self, X: pd.DataFrame):
        """DBSCAN 方法拟合"""
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            raise ImportError("sklearn is required for DBSCAN method")

        eps = self.threshold if self.threshold else 0.5
        self.dbscan_ = DBSCAN(eps=eps, min_samples=5)
        clusters = self.dbscan_.fit_predict(X.values)
        self._outlier_mask = clusters == -1

        self.metadata = {
            "type": "OutlierDetector",
            "method": "dbscan",
            "threshold": eps,
            "columns": X.columns.tolist(),
            "n_outliers": int(self._outlier_mask.sum()),
            "outlier_ratio": float(self._outlier_mask.sum() / len(X)),
        }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        检测异常值（不删除，标记）

        Returns:
            添加了 outlier 列的 DataFrame
        """
        result = X.copy()
        if self._outlier_mask is not None:
            result["is_outlier"] = self._outlier_mask.any(axis=1).astype(int)
        return result

    def get_outlier_mask(self) -> np.ndarray:
        """获取异常值掩码"""
        return self._outlier_mask

    def remove_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """移除异常值"""
        if self._outlier_mask is None:
            raise RuntimeError("Detector must be fitted before removing outliers")
        return X[~self._outlier_mask.any(axis=1)]
