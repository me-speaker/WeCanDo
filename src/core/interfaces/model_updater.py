"""
模型更新接口
实现增量更新机制，支持闭环迭代优化
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import logging

from .base import BaseModelUpdater

logger = logging.getLogger(__name__)


class ModelUpdater(BaseModelUpdater):
    """
    模型增量更新器
    封装模型更新逻辑，支持闭环迭代
    """

    def __init__(
        self,
        model: Any,
        buffer_size: int = 100,
        batch_update: bool = True,
        update_strategy: str = "append",  # "append", "replace", "window"
    ):
        """
        初始化模型更新器

        Args:
            model: 底层模型实例，需实现 fit(X, y) 和 predict(X) 方法
            buffer_size: 数据缓冲区大小（用于window策略）
            batch_update: 是否批量更新（True则积累到buffer_size再更新）
            update_strategy: 更新策略
                - "append": 追加新数据到历史数据后完整重训练
                - "replace": 完全替换旧数据进行训练
                - "window": 使用滑动窗口，只保留最新buffer_size条数据
        """
        self.model = model
        self.buffer_size = buffer_size
        self.batch_update = batch_update
        self.update_strategy = update_strategy

        self._X_buffer: list = []
        self._y_buffer: list = []
        self._X_history: list = []
        self._y_history: list = []
        self._n_updates: int = 0
        self._is_fitted: bool = False

    def update(self, X_new: np.ndarray, y_new: np.ndarray) -> bool:
        """
        增量更新模型

        Args:
            X_new: 新增决策变量数据, shape (n_new, n_decision_vars)
            y_new: 新增目标变量数据, shape (n_new, n_objectives)

        Returns:
            更新是否成功

        Raises:
            ValueError: 数据形状不匹配
        """
        # Validate input shapes
        if X_new.shape[0] != y_new.shape[0]:
            raise ValueError(
                f"X_new和y_new的样本数不一致: {X_new.shape[0]} vs {y_new.shape[0]}"
            )

        if len(X_new.shape) == 1:
            X_new = X_new.reshape(-1, 1)
        if len(y_new.shape) == 1:
            y_new = y_new.reshape(-1, 1)

        n_new = X_new.shape[0]
        logger.info(f"接收到 {n_new} 条新数据进行模型更新")

        # Add to buffer
        self._X_buffer.extend(X_new.tolist() if hasattr(X_new, 'tolist') else X_new)
        self._y_buffer.extend(y_new.tolist() if hasattr(y_new, 'tolist') else y_new)

        # Check if we should perform update
        should_update = not self.batch_update or len(self._X_buffer) >= self.buffer_size

        if should_update and self._X_buffer:
            self._perform_update()
            return True

        return False

    def _perform_update(self) -> None:
        """执行实际的模型更新"""
        # Convert buffers to arrays
        X_new_batch = np.array(self._X_buffer)
        y_new_batch = np.array(self._y_buffer)

        # Clear buffers
        self._X_buffer.clear()
        self._y_buffer.clear()

        # Apply update strategy
        if self.update_strategy == "replace":
            self._X_history = X_new_batch.tolist()
            self._y_history = y_new_batch.tolist()
        elif self.update_strategy == "window":
            self._X_history.extend(X_new_batch.tolist())
            self._y_history.extend(y_new_batch.tolist())
            # Keep only latest buffer_size samples
            if len(self._X_history) > self.buffer_size:
                self._X_history = self._X_history[-self.buffer_size:]
                self._y_history = self._y_history[-self.buffer_size:]
        else:  # "append"
            self._X_history.extend(X_new_batch.tolist())
            self._y_history.extend(y_new_batch.tolist())

        # Train model
        X_train = np.array(self._X_history)
        y_train = np.array(self._y_history)

        logger.info(f"使用 {len(X_train)} 条数据训练模型")
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        self._n_updates += 1

    def force_update(self) -> bool:
        """
        强制执行更新（即使缓冲区未满）

        Returns:
            更新是否成功
        """
        if self._X_buffer:
            self._perform_update()
            return True
        return False

    def get_model(self) -> Any:
        """
        获取当前模型实例

        Returns:
            模型实例
        """
        return self.model

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取当前训练数据

        Returns:
            X: 训练数据
            y: 标签数据
        """
        if not self._X_history:
            return np.array([]), np.array([])
        return np.array(self._X_history), np.array(self._y_history)

    @property
    def n_updates(self) -> int:
        """已更新次数"""
        return self._n_updates

    @property
    def n_samples(self) -> int:
        """当前训练样本数"""
        return len(self._X_history)

    @property
    def buffer_fill_ratio(self) -> float:
        """缓冲区填充比例"""
        return len(self._X_buffer) / self.buffer_size if self.buffer_size > 0 else 1.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用当前模型预测

        Args:
            X: 输入数据

        Returns:
            预测值

        Raises:
            RuntimeError: 模型未训练
        """
        if not self._is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 update() 或确保有初始训练数据")
        return self.model.predict(X)

    def reset(self) -> None:
        """重置更新器状态"""
        self._X_buffer.clear()
        self._y_buffer.clear()
        self._X_history.clear()
        self._y_history.clear()
        self._n_updates = 0
        self._is_fitted = False
        logger.info("模型更新器已重置")
