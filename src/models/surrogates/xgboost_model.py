"""
DeepInd XGBoost 代理模型
XGBoost surrogate model wrapper
"""

import numpy as np
from typing import Optional, Tuple

from ...core.base import BaseSurrogate
from ...core.registry import register_surrogate
from ...utils.logger import get_logger

# XGBoost 可能未安装，尝试导入
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None


@register_surrogate("XGBoost")
class XGBoostModel(BaseSurrogate):
    """
    XGBoost 代理模型

    封装 XGBoost 回归器，支持：
    - 点预测
    - 增量更新
    - 多目标回归

    Args:
        input_dim: 输入维度
        n_estimators: 树的数量，默认 100
        max_depth: 最大深度，默认 6
        learning_rate: 学习率，默认 0.1
        subsample: 子采样比例，默认 1.0
        colsample_bytree: 列采样比例，默认 1.0
        reg_alpha: L1 正则化，默认 0
        reg_lambda: L2 正则化，默认 1
        min_child_weight: 最小子节点权重，默认 1
    """

    def __init__(
        self,
        input_dim: int,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_weight: int = 1,
    ):
        super().__init__()
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost 未安装，请运行: pip install xgboost"
            )

        self.input_dim = input_dim
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight

        self._models: Optional[list] = None
        self._n_outputs: int = 1
        self._is_fitted = False

        self._X_history: Optional[np.ndarray] = None
        self._y_history: Optional[np.ndarray] = None

        self.logger = get_logger("DeepInd::XGBoost")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        """
        训练 XGBoost 模型

        Args:
            X: 输入特征，shape (n_samples, n_features)
            y: 目标值，shape (n_samples,) 或 (n_samples, n_objectives)

        Returns:
            self
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost 未安装")

        self.logger.info("XGBoost: 开始训练")

        X = np.asarray(X)
        y = np.asarray(y)

        # 保存历史数据（用于增量更新）
        self._X_history = X.copy()
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._y_history = y.copy()

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._n_outputs = y.shape[1] if y.ndim > 1 else 1

        # 为每个输出训练一个模型
        self._models = []
        for i in range(self._n_outputs):
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                min_child_weight=self.min_child_weight,
                objective="reg:squarederror",
                verbosity=0,
            )
            model.fit(X, y[:, i])
            self._models.append(model)

        self._is_fitted = True
        self.logger.info(f"XGBoost: 训练完成，输出维度: {self._n_outputs}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 输入特征，shape (n_samples, n_features)

        Returns:
            预测值，shape (n_samples,) 或 (n_samples, n_objectives)
        """
        if not self._is_fitted:
            raise RuntimeError("XGBoost 未训练，请先调用 fit()")

        X = np.asarray(X)

        if self._n_outputs == 1:
            return self._models[0].predict(X)
        else:
            preds = [model.predict(X) for model in self._models]
            return np.column_stack(preds)

    def update(self, X_new: np.ndarray, y_new: np.ndarray) -> "XGBoostModel":
        """
        增量更新

        XGBoost 原生不支持增量训练，
        这里通过合并历史数据重新训练实现

        Args:
            X_new: 新增数据特征
            y_new: 新增目标值

        Returns:
            self
        """
        self.logger.info("XGBoost: 执行增量更新")

        X_new = np.asarray(X_new)
        y_new = np.asarray(y_new)

        if y_new.ndim == 1:
            y_new = y_new.reshape(-1, 1)

        # 合并新旧数据
        if self._X_history is not None and self._y_history is not None:
            X_combined = np.vstack([self._X_history, X_new])
            y_combined = np.vstack([self._y_history, y_new])
        else:
            X_combined = X_new
            y_combined = y_new

        # 重新训练
        return self.fit(X_combined, y_combined)

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple:
        """
        带不确定性预测

        使用预测的残差分位数估计不确定性

        Args:
            X: 输入特征，shape (n_samples, n_features)

        Returns:
            (预测值, 不确定性)，shape (n_samples, n_objectives)
        """
        if not self._is_fitted:
            raise RuntimeError("XGBoost 未训练，请先调用 fit()")

        y_pred = self.predict(X)

        # 使用训练残差估计不确定性
        if self._X_history is not None and self._y_history is not None:
            y_train_pred = self.predict(self._X_history)
            residuals = self._y_history - y_train_pred
            if residuals.ndim == 1:
                residual_std = np.std(residuals)
            else:
                residual_std = np.std(residuals, axis=0)
            uncertainty = np.ones_like(y_pred) * residual_std
            return y_pred, uncertainty

        uncertainty = np.ones_like(y_pred) * 0.1
        return y_pred, uncertainty

    @property
    def is_differentiable(self) -> bool:
        """XGBoost 不可微（树模型）"""
        return False

    @property
    def n_output(self) -> int:
        """输出维度"""
        return self._n_outputs

    @property
    def feature_importances(self) -> np.ndarray:
        """
        获取特征重要性

        Returns:
            特征重要性分数，shape (n_features,)
        """
        if not self._is_fitted:
            raise RuntimeError("XGBoost 未训练，请先调用 fit()")

        if self._n_outputs == 1:
            return self._models[0].feature_importances_
        else:
            # 平均多个输出的重要性
            importances = [m.feature_importances_ for m in self._models]
            return np.mean(importances, axis=0)