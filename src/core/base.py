"""
DeepInd 核心抽象基类
定义特征工程、代理模型、优化器的标准接口
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Callable, Any
import numpy as np


class BaseFeatureEngine(ABC):
    """
    特征工程抽象基类
    定义数据预处理和特征提取的接口
    """

    @abstractmethod
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        拟合并转换数据

        Args:
            X: 输入数据，shape (n_samples, n_features)
            y: 可选的标签数据，用于监督学习场景

        Returns:
            转换后的特征，shape (n_samples, feature_dim)
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        转换新数据（仅transform，不fit）

        Args:
            X: 输入数据

        Returns:
            转换后的特征
        """
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """输出特征维度"""
        pass


class BaseSurrogate(ABC):
    """
    代理模型抽象基类
    定义预测模型的训练和推理接口
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseSurrogate":
        """
        训练代理模型

        Args:
            X: 输入特征，shape (n_samples, n_features)
            y: 目标值，shape (n_samples, n_objectives) 或 (n_samples,)

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 输入特征

        Returns:
            预测值
        """
        pass

    def update(self, X_new: np.ndarray, y_new: np.ndarray) -> "BaseSurrogate":
        """
        增量更新（可选实现）

        Args:
            X_new: 新增输入数据
            y_new: 新增目标值

        Returns:
            self
        """
        raise NotImplementedError(f"{self.__class__.__name__} 不支持增量更新")

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple:
        """
        带不确定性预测（可选实现）

        Args:
            X: 输入特征

        Returns:
            (预测值, 不确定性)
        """
        raise NotImplementedError(f"{self.__class__.__name__} 不支持不确定性预测")

    @property
    def is_differentiable(self) -> bool:
        """模型是否可微"""
        return False


class BaseOptimizer(ABC):
    """
    优化器抽象基类
    定义优化问题的求解接口
    """

    @abstractmethod
    def optimize(
        self,
        objectives: List[Callable],
        bounds: List[tuple],
        constraints: Optional[List[dict]] = None
    ) -> np.ndarray:
        """
        执行优化

        Args:
            objectives: 目标函数列表，每个函数接受 x 返回标量
            bounds: 决策变量边界，[(low, high), ...]
            constraints: 约束条件列表（可选）

        Returns:
            优化结果，shape (n_solutions, n_decision_vars + n_objectives)
        """
        pass

    @property
    def n_objectives(self) -> int:
        """目标函数数量"""
        return len(self.objectives) if hasattr(self, 'objectives') else 1


class BasePostProcessor(ABC):
    """
    后处理器抽象基类
    定义对优化结果的后处理接口
    """

    @abstractmethod
    def __call__(self, results: dict) -> dict:
        """
        处理结果

        Args:
            results: 包含优化结果的字典

        Returns:
            处理后的结果字典
        """
        pass


class BaseDataLoader(ABC):
    """
    数据加载器抽象基类
    定义数据加载和预处理的接口
    """

    @abstractmethod
    def load(self) -> tuple:
        """
        加载数据

        Returns:
            X: 决策变量
            y: 目标变量
        """
        pass

    def preprocess(self, X, y):
        """
        预处理数据（可选实现）

        Args:
            X: 决策变量
            y: 目标变量

        Returns:
            预处理后的 X, y
        """
        return X, y


class BaseConstraint(ABC):
    """
    约束处理器抽象基类
    定义约束处理接口
    """

    @abstractmethod
    def wrap(self, objectives: List[Callable]) -> List[Callable]:
        """
        将约束包装到目标函数中

        Args:
            objectives: 原始目标函数列表

        Returns:
            包装后的目标函数列表
        """
        pass

    @abstractmethod
    def evaluate_constraint_violation(self, x: np.ndarray) -> float:
        """
        计算约束违反量

        Args:
            x: 决策变量

        Returns:
            约束违反量
        """
        pass
