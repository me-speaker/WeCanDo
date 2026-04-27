"""
DeepInd 核心框架
"""

from .base import BaseFeatureEngine, BaseSurrogate, BaseOptimizer, BasePostProcessor
from .registry import (
    FEATURE_ENGINE_REGISTRY,
    SURROGATE_REGISTRY,
    OPTIMIZER_REGISTRY,
    POSTPROCESSOR_REGISTRY,
    CONSTRAINT_HANDLER_REGISTRY,
    DATALOADER_REGISTRY,
    register_feature,
    register_surrogate,
    register_optimizer,
    register_postprocessor,
    register_constraint,
    register_dataloader,
)
from .factory import (
    create_feature_engine,
    create_surrogate,
    create_optimizer,
    create_postprocessor,
    create_constraint_handler,
    create_dataloader,
)
from .runner import TaskRunner
from .config_manager import ConfigManager, create_config_manager, get_default_config_manager

__all__ = [
    # 基类
    "BaseFeatureEngine",
    "BaseSurrogate",
    "BaseOptimizer",
    "BasePostProcessor",
    # 注册器
    "FEATURE_ENGINE_REGISTRY",
    "SURROGATE_REGISTRY",
    "OPTIMIZER_REGISTRY",
    "POSTPROCESSOR_REGISTRY",
    "CONSTRAINT_HANDLER_REGISTRY",
    "DATALOADER_REGISTRY",
    "register_feature",
    "register_surrogate",
    "register_optimizer",
    "register_postprocessor",
    "register_constraint",
    "register_dataloader",
    # 工厂函数
    "create_feature_engine",
    "create_surrogate",
    "create_optimizer",
    "create_postprocessor",
    "create_constraint_handler",
    "create_dataloader",
    # 运行器
    "TaskRunner",
    # 配置管理器
    "ConfigManager",
    "create_config_manager",
    "get_default_config_manager",
]
