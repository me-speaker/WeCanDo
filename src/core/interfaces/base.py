"""
DeepInd 接口抽象基类
定义配置解析、数据导入、模型更新的标准接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np


@dataclass
class DecisionVariable:
    """决策变量定义"""
    name: str
    bounds: Tuple[float, float]
    description: str = ""
    unit: str = ""
    var_type: str = "continuous"  # continuous, discrete, integer

    def __post_init__(self):
        assert len(self.bounds) == 2, "bounds must be (low, high)"
        assert self.bounds[0] <= self.bounds[1], "lower bound must be <= upper bound"


@dataclass
class ObjectiveVariable:
    """目标变量定义"""
    name: str
    direction: str  # "minimize" or "maximize"
    description: str = ""
    unit: str = ""
    target_idx: int = 0  # Index in the surrogate model output

    def __post_init__(self):
        assert self.direction in ("minimize", "maximize"), "direction must be 'minimize' or 'maximize'"


@dataclass
class ConstraintSpec:
    """约束条件定义"""
    ctype: str  # "eq" (equality) or "ineq" (inequality)
    expression: str  # Python expression string, e.g., "x[0] + x[1] - 1.0"
    description: str = ""
    penalty_weight: float = 1.0

    def __post_init__(self):
        assert self.ctype in ("eq", "ineq"), "ctype must be 'eq' or 'ineq'"


@dataclass
class ConfigSchema:
    """配置模式定义"""
    task_name: str
    task_type: str = "single_objective"
    decision_variables: List[DecisionVariable] = field(default_factory=list)
    objectives: List[ObjectiveVariable] = field(default_factory=list)
    constraints: List[ConstraintSpec] = field(default_factory=list)
    data_config: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)
    optimizer_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_decision_vars(self) -> int:
        return len(self.decision_variables)

    @property
    def n_objectives(self) -> int:
        return len(self.objectives)

    def get_bounds(self) -> List[Tuple[float, float]]:
        return [dv.bounds for dv in self.decision_variables]

    def get_names(self) -> List[str]:
        return [dv.name for dv in self.decision_variables]


class BaseConfigParser(ABC):
    """
    配置解析器抽象基类
    定义从配置文件加载和解析参数的接口
    """

    @abstractmethod
    def load(self, path: str) -> Dict[str, Any]:
        """
        加载配置文件

        Args:
            path: 配置文件路径

        Returns:
            原始配置字典
        """
        pass

    @abstractmethod
    def parse(self, config: Dict[str, Any]) -> ConfigSchema:
        """
        解析配置字典为ConfigSchema

        Args:
            config: 原始配置字典

        Returns:
            结构化的ConfigSchema对象
        """
        pass

    @abstractmethod
    def validate(self, schema: ConfigSchema) -> bool:
        """
        验证配置的有效性

        Args:
            schema: 配置模式对象

        Returns:
            是否有效
        """
        pass

    def load_and_parse(self, path: str) -> ConfigSchema:
        """
        加载并解析配置文件

        Args:
            path: 配置文件路径

        Returns:
            ConfigSchema对象
        """
        raw_config = self.load(path)
        schema = self.parse(raw_config)
        self.validate(schema)
        return schema


class BaseDataImporter(ABC):
    """
    数据导入器抽象基类
    定义从不同数据源加载数据的接口
    """

    @abstractmethod
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载数据

        Returns:
            X: 决策变量数据, shape (n_samples, n_decision_vars)
            y: 目标变量数据, shape (n_samples, n_objectives)
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        获取数据元信息

        Returns:
            包含数据信息的字典
        """
        pass

    def preprocess(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据预处理（可选实现）

        Args:
            X: 决策变量数据
            y: 目标变量数据

        Returns:
            预处理后的 X, y
        """
        return X, y


class BaseModelUpdater(ABC):
    """
    模型更新器抽象基类
    定义增量更新模型接口，支持闭环迭代
    """

    @abstractmethod
    def update(self, X_new: np.ndarray, y_new: np.ndarray) -> bool:
        """
        增量更新模型

        Args:
            X_new: 新增决策变量数据, shape (n_new, n_decision_vars)
            y_new: 新增目标变量数据, shape (n_new, n_objectives)

        Returns:
            更新是否成功
        """
        pass

    @abstractmethod
    def get_model(self) -> Any:
        """
        获取当前模型实例

        Returns:
            模型实例
        """
        pass

    @property
    @abstractmethod
    def n_updates(self) -> int:
        """已更新次数"""
        pass

    @property
    def is_ready(self) -> bool:
        """模型是否已准备好用于预测"""
        return self.n_updates > 0
