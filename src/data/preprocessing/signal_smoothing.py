"""
DeepInd 数据预处理模块 - 信号平滑
提供多种信号平滑方法
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, filtfilt, savgol_filter

from .base import BaseDataProcessor, register_data_processor


@register_data_processor("SignalSmoother")
class SignalSmoother(BaseDataProcessor):
    """
    信号平滑器
    支持多种平滑方法：高斯滤波、移动平均、指数平滑、样条平滑等
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - method: 平滑方法 "gaussian", "moving_average", "exponential", "spline", "butterworth"
                - window_size: 窗口大小（移动平均/指数平滑）
                - sigma: 高斯滤波的标准差
                - alpha: 指数平滑的权重因子
                - spline_order: 样条插值的阶数
                - butterworth_order: Butterworth 滤波器的阶数
                - cutoff_freq: Butterworth 截止频率
                - columns: 要平滑的列，默认全部
        """
        super().__init__(config)
        self.method = self.config.get("method", "gaussian")
        self.window_size = self.config.get("window_size", 5)
        self.sigma = self.config.get("sigma", 1.0)
        self.alpha = self.config.get("alpha", 0.3)
        self.spline_order = self.config.get("spline_order", 3)
        self.butterworth_order = self.config.get("butterworth_order", 2)
        self.cutoff_freq = self.config.get("cutoff_freq", 0.1)
        self.columns = self.config.get("columns", None)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "SignalSmoother":
        """
        拟合平滑器

        Args:
            X: 输入数据 DataFrame
            y: 忽略

        Returns:
            self
        """
        cols = self.columns if self.columns else X.columns.tolist()

        self.metadata = {
            "type": "SignalSmoother",
            "method": self.method,
            "columns": cols,
            "original_shape": X.shape,
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        平滑处理

        Args:
            X: 输入数据 DataFrame

        Returns:
            平滑后的 DataFrame
        """
        cols = self.columns if self.columns else X.columns.tolist()
        result = X.copy()

        for col in cols:
            if col not in X.columns:
                self.logger.warning(f"Column {col} not found in data")
                continue

            if self.method == "gaussian":
                result[col] = self._gaussian_smooth(X[col].values)
            elif self.method == "moving_average":
                result[col] = self._moving_average(X[col].values)
            elif self.method == "exponential":
                result[col] = self._exponential_smooth(X[col].values)
            elif self.method == "spline":
                result[col] = self._spline_smooth(X[col].values)
            elif self.method == "butterworth":
                result[col] = self._butterworth_smooth(X[col].values)
            else:
                raise ValueError(f"Unknown method: {self.method}")

        self.metadata["processed_shape"] = result.shape
        return result

    def _gaussian_smooth(self, signal: np.ndarray) -> np.ndarray:
        """高斯平滑"""
        return gaussian_filter1d(signal, sigma=self.sigma)

    def _moving_average(self, signal: np.ndarray) -> np.ndarray:
        """移动平均平滑"""
        return uniform_filter1d(signal, size=self.window_size, mode="nearest")

    def _exponential_smooth(self, signal: np.ndarray) -> np.ndarray:
        """指数平滑"""
        smoothed = np.zeros_like(signal)
        smoothed[0] = signal[0]
        for i in range(1, len(signal)):
            smoothed[i] = self.alpha * signal[i] + (1 - self.alpha) * smoothed[i - 1]
        return smoothed

    def _spline_smooth(self, signal: np.ndarray) -> np.ndarray:
        """样条平滑"""
        try:
            from scipy.interpolate import UnivariateSpline
        except ImportError:
            raise ImportError("scipy.interpolate is required for spline smoothing")

        x = np.arange(len(signal))
        # 使用平滑样条
        spline = UnivariateSpline(x, signal, k=self.spline_order, s=len(signal))
        return spline(x)

    def _butterworth_smooth(self, signal: np.ndarray) -> np.ndarray:
        """Butterworth 低通滤波平滑"""
        nyquist = 0.5  # 归一化频率
        normal_cutoff = self.cutoff_freq / nyquist
        b, a = butter(self.butterworth_order, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a, signal)
