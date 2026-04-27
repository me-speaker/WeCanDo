"""
DeepInd 数据预处理模块 - 噪声抑制
提供多种噪声降低方法
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, medfilt, wiener

from .base import BaseDataProcessor, register_data_processor


@register_data_processor("NoiseReducer")
class NoiseReducer(BaseDataProcessor):
    """
    噪声抑制器
    支持多种降噪方法：低通滤波、卡尔曼滤波、小波变换等
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - method: 降噪方法 "lowpass", "median", "wiener", "savgol", "wavelet"
                - cutoff_freq: 截止频率（低通滤波）
                - window_size: 窗口大小（移动平均/中值滤波）
                - polyorder: Savitzky-Golay 多项式阶数
                - wavelet: 小波类型（默认 'db4'）
                - level: 小波分解层数
        """
        super().__init__(config)
        self.method = self.config.get("method", "median")
        self.cutoff_freq = self.config.get("cutoff_freq", 0.1)
        self.window_size = self.config.get("window_size", 5)
        self.polyorder = self.config.get("polyorder", 3)
        self.wavelet = self.config.get("wavelet", "db4")
        self.level = self.config.get("level", 1)
        self.columns = self.config.get("columns", None)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "NoiseReducer":
        """
        拟合降噪器

        Args:
            X: 输入数据 DataFrame
            y: 忽略

        Returns:
            self
        """
        cols = self.columns if self.columns else X.columns.tolist()

        self.metadata = {
            "type": "NoiseReducer",
            "method": self.method,
            "columns": cols,
            "original_shape": X.shape,
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        降噪处理

        Args:
            X: 输入数据 DataFrame

        Returns:
            降噪后的 DataFrame
        """
        cols = self.columns if self.columns else X.columns.tolist()
        result = X.copy()

        for col in cols:
            if col not in X.columns:
                self.logger.warning(f"Column {col} not found in data")
                continue

            if self.method == "lowpass":
                result[col] = self._lowpass_filter(X[col].values)
            elif self.method == "median":
                result[col] = self._median_filter(X[col].values)
            elif self.method == "wiener":
                result[col] = self._wiener_filter(X[col].values)
            elif self.method == "savgol":
                result[col] = self._savgol_filter(X[col].values)
            elif self.method == "wavelet":
                result[col] = self._wavelet_denoise(X[col].values)
            else:
                raise ValueError(f"Unknown method: {self.method}")

        self.metadata["processed_shape"] = result.shape
        return result

    def _lowpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """简单低通滤波（移动平均）"""
        if len(signal) < self.window_size:
            return signal
        return np.convolve(signal, np.ones(self.window_size) / self.window_size, mode="same")

    def _median_filter(self, signal: np.ndarray) -> np.ndarray:
        """中值滤波"""
        window = self.window_size if self.window_size % 2 == 1 else self.window_size + 1
        return medfilt(signal, kernel_size=window)

    def _wiener_filter(self, signal: np.ndarray) -> np.ndarray:
        """维纳滤波"""
        return wiener(signal, mysize=self.window_size)

    def _savgol_filter(self, signal: np.ndarray) -> np.ndarray:
        """Savitzky-Golay 滤波"""
        window = self.window_size if self.window_size % 2 == 1 else self.window_size + 1
        if len(signal) < window:
            return signal
        if self.polyorder >= window:
            polyorder = window - 1
        else:
            polyorder = self.polyorder
        return savgol_filter(signal, window, polyorder)

    def _wavelet_denoise(self, signal: np.ndarray) -> np.ndarray:
        """小波降噪"""
        try:
            import pywt
        except ImportError:
            raise ImportError("pywt is required for wavelet denoising")

        # 小波分解
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)

        # 软阈值处理
        threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(signal)))
        coeffs[-1] = pywt.threshold(coeffs[-1], threshold, mode="soft")

        # 重构
        return pywt.waverec(coeffs, self.wavelet)[:len(signal)]
