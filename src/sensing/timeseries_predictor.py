"""
DeepInd 时序信号预测器
用于时序数据的软测量预测，支持序列到序列预测
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .predictor import SoftSensingPredictor, PredictionResult


@dataclass
class TimeSeriesConfig:
    """时序预测配置"""
    sequence_length: int = 10
    forecast_horizon: int = 1
    stride: int = 1
    use_historical_features: bool = True


class TimeSeriesPredictor(SoftSensingPredictor):
    """
    时序信号预测器

    特点：
    - 支持多步提前预测
    - 支持外生变量
    - 支持缺失值处理
    - 预留序列模型接口（LSTM/GRU）
    """

    def __init__(
        self,
        model: Any,
        config: Optional[TimeSeriesConfig] = None,
        input_scaler: Optional[Any] = None,
        output_scaler: Optional[Any] = None,
        uncertainty_enabled: bool = False,
    ):
        """
        初始化时序预测器

        Args:
            model: 训练好的代理模型
            config: 时序预测配置
            input_scaler: 输入标准化器
            output_scaler: 输出标准化器
            uncertainty_enabled: 是否启用不确定性估计
        """
        super().__init__(
            model=model,
            input_scaler=input_scaler,
            output_scaler=output_scaler,
            uncertainty_enabled=uncertainty_enabled,
        )
        self.config = config or TimeSeriesConfig()
        self._sequence_buffer = None
        self._last_input = None

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        return_uncertainty: bool = False,
    ) -> PredictionResult:
        """
        执行时序预测

        Args:
            X: 输入数据（当前时刻或历史序列）
            return_uncertainty: 是否返回不确定性

        Returns:
            PredictionResult: 预测结果
        """
        X_array = self.validate_input(X)

        # 序列预测
        predictions, metadata = self._predict_sequence(X_array)

        result = PredictionResult(
            predictions=predictions,
            uncertainties=None,
            metadata=metadata,
        )

        if return_uncertainty and self.uncertainty_enabled:
            result.uncertainties = self._estimate_sequence_uncertainty(X_array)

        return result

    def _predict_sequence(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        执行序列预测

        Args:
            X: 输入数据

        Returns:
            (predictions, metadata): 预测结果和元数据
        """
        n_samples = X.shape[0] if len(X.shape) > 1 else 1
        horizon = self.config.forecast_horizon

        # 迭代多步预测
        all_predictions = []
        current_input = X.copy() if n_samples == 1 else X[-1:]

        for step in range(horizon):
            pred = self._single_step_predict(current_input)
            all_predictions.append(pred)

            # 更新输入用于下一步预测（如果是多步预测）
            if step < horizon - 1 and self.config.use_historical_features:
                current_input = self._update_input_sequence(
                    current_input, pred, X if n_samples > 1 else None
                )

        predictions = np.concatenate(all_predictions, axis=0) if len(all_predictions) > 1 else all_predictions[0]

        metadata = {
            "predictor": "TimeSeriesPredictor",
            "sequence_length": self.config.sequence_length,
            "forecast_horizon": horizon,
            "n_samples": n_samples,
        }

        return predictions, metadata

    def _single_step_predict(self, X: np.ndarray) -> np.ndarray:
        """
        单步预测

        Args:
            X: 当前输入

        Returns:
            np.ndarray: 预测值
        """
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        return X

    def _update_input_sequence(
        self,
        current_input: np.ndarray,
        prediction: np.ndarray,
        historical_input: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        更新输入序列用于多步预测

        Args:
            current_input: 当前输入
            prediction: 当前预测
            historical_input: 历史输入

        Returns:
            np.ndarray: 更新后的输入
        """
        if len(current_input.shape) == 1:
            current_input = current_input.reshape(1, -1)

        # 简单滚动更新：去掉最老的，加入最新的
        updated = np.roll(current_input, -1, axis=0)
        updated[-1] = prediction.flatten()

        return updated

    def _estimate_sequence_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        估计序列预测的不确定性

        Args:
            X: 输入数据

        Returns:
            np.ndarray: 不确定性估计
        """
        # 预留扩展：使用蒙特卡洛 dropout 或集成方法
        horizon = self.config.forecast_horizon
        n_features = X.shape[-1] if len(X.shape) > 1 else 1
        return np.zeros((horizon, n_features))

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
    ) -> "TimeSeriesPredictor":
        """
        训练时序模型

        Args:
            X: 历史输入序列
            y: 对应目标序列

        Returns:
            self: 返回自身以支持链式调用
        """
        # 预处理序列数据
        X_seq, y_seq = self._prepare_sequences(X, y)
        super().fit(X_seq, y_seq)
        return self

    def _prepare_sequences(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备序列训练数据

        Args:
            X: 输入
            y: 目标

        Returns:
            (X_seq, y_seq): 序列化的训练数据
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        # 滑动窗口切分序列
        sequences = []
        for i in range(0, len(X) - self.config.sequence_length, self.config.stride):
            seq_x = X[i : i + self.config.sequence_length]
            seq_y = y[i + self.config.sequence_length]
            sequences.append((seq_x, seq_y))

        if not sequences:
            return X, y

        X_seq = np.array([s[0] for s in sequences])
        y_seq = np.array([s[1] for s in sequences])

        return X_seq, y_seq

    def predict_future(
        self,
        X_history: Union[np.ndarray, pd.DataFrame],
        horizon: int,
    ) -> PredictionResult:
        """
        基于历史数据预测未来

        Args:
            X_history: 历史输入序列
            horizon: 预测步数

        Returns:
            PredictionResult: 预测结果
        """
        old_horizon = self.config.forecast_horizon
        self.config.forecast_horizon = horizon

        X_array = self.validate_input(X_history)
        result = self.predict(X_array)

        self.config.forecast_horizon = old_horizon
        return result

    def set_sequence_length(self, length: int):
        """设置序列长度"""
        self.config.sequence_length = length

    def set_forecast_horizon(self, horizon: int):
        """设置预测步数"""
        self.config.forecast_horizon = horizon
