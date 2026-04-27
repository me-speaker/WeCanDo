"""
DeepInd 软测量预测器基类
定义软测量预测的通用接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PredictionResult:
    """预测结果数据类"""
    predictions: Union[np.ndarray, pd.DataFrame]
    uncertainties: Optional[Union[np.ndarray, pd.DataFrame]] = None
    confidence_interval: Optional[tuple] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """验证预测结果"""
        if isinstance(self.predictions, np.ndarray):
            assert len(self.predictions.shape) >= 1, "predictions must be at least 1D"


class SoftSensingPredictor(ABC):
    """
    软测量预测器基类

    提供软测量预测的通用接口，支持：
    - 单点预测
    - 批量预测
    - 不确定性估计
    - 预测结果缓存
    """

    def __init__(
        self,
        model: Any,
        input_scaler: Optional[Any] = None,
        output_scaler: Optional[Any] = None,
        uncertainty_enabled: bool = False,
    ):
        """
        初始化软测量预测器

        Args:
            model: 训练好的代理模型
            input_scaler: 输入标准化器（可选）
            output_scaler: 输出标准化器（可选）
            uncertainty_enabled: 是否启用不确定性估计
        """
        self.model = model
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.uncertainty_enabled = uncertainty_enabled
        self._is_fitted = False

    @abstractmethod
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        return_uncertainty: bool = False,
    ) -> PredictionResult:
        """
        执行预测

        Args:
            X: 输入数据
            return_uncertainty: 是否返回不确定性

        Returns:
            PredictionResult: 预测结果
        """
        pass

    def predict_batch(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        batch_size: int = 32,
    ) -> PredictionResult:
        """
        批量预测（自动分批处理大数据集）

        Args:
            X: 输入数据
            batch_size: 批处理大小

        Returns:
            PredictionResult: 预测结果
        """
        if isinstance(X, pd.DataFrame):
            n_samples = len(X)
            predictions_list = []
            uncertainties_list = []

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_X = X.iloc[start_idx:end_idx]
                result = self.predict(batch_X, return_uncertainty=self.uncertainty_enabled)
                predictions_list.append(
                    result.predictions.values
                    if isinstance(result.predictions, pd.DataFrame)
                    else result.predictions
                )
                if result.uncertainties is not None:
                    uncertainties_list.append(
                        result.uncertainties.values
                        if isinstance(result.uncertainties, pd.DataFrame)
                        else result.uncertainties
                    )

            predictions = np.concatenate(predictions_list, axis=0)
            uncertainties = (
                np.concatenate(uncertainties_list, axis=0)
                if uncertainties_list
                else None
            )

            return PredictionResult(
                predictions=predictions,
                uncertainties=uncertainties,
            )
        else:
            return self.predict(X, return_uncertainty=self.uncertainty_enabled)

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
    ) -> "SoftSensingPredictor":
        """
        训练软测量模型

        Args:
            X: 输入数据
            y: 目标数据

        Returns:
            self: 返回自身以支持链式调用
        """
        self._is_fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        """检查模型是否已训练"""
        return self._is_fitted

    def validate_input(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        验证和预处理输入数据

        Args:
            X: 输入数据

        Returns:
            np.ndarray: 预处理后的数据
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)

        # 应用输入标准化器
        if self.input_scaler is not None:
            X_array = self.input_scaler.transform(X_array)

        return X_array

    def inverse_transform_output(
        self, y: np.ndarray
    ) -> np.ndarray:
        """
        反标准化输出数据

        Args:
            y: 标准化后的输出

        Returns:
            np.ndarray: 原始尺度的输出
        """
        if self.output_scaler is not None:
            return self.output_scaler.inverse_transform(y)
        return y

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            Dict[str, Any]: 模型信息字典
        """
        return {
            "model_type": self.__class__.__name__,
            "is_fitted": self._is_fitted,
            "uncertainty_enabled": self.uncertainty_enabled,
            "has_input_scaler": self.input_scaler is not None,
            "has_output_scaler": self.output_scaler is not None,
        }
