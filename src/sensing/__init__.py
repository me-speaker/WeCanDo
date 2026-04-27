"""
DeepInd 软测量模块
提供软测量预测功能，支持离线批量预测和实时流式预测
"""

from .predictor import SoftSensingPredictor, PredictionResult
from .stream_predictor import StreamPredictor
from .timeseries_predictor import TimeSeriesPredictor

__all__ = [
    "SoftSensingPredictor",
    "PredictionResult",
    "StreamPredictor",
    "TimeSeriesPredictor",
]
