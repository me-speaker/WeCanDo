"""
DeepInd 模块注册机制
通过装饰器方式注册特征工程、代理模型、优化器、后处理器
"""

from typing import Dict, Type, Any


class Registry:
    """
    模块注册器
    提供注册和获取功能
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, name: str = None) -> callable:
        """
        注册装饰器

        Args:
            name: 注册名称，默认使用类名

        Usage:
            @FEATURE_ENGINE_REGISTRY.register("DAE")
            class DenoisingAutoencoder:
                ...
        """

        def wrapper(cls):
            cls_name = name or cls.__name__
            if cls_name in self._registry:
                raise ValueError(
                    f"{self.name} '{cls_name}' 已注册，请使用不同的名称"
                )
            self._registry[cls_name] = cls
            return cls

        return wrapper

    def get(self, name: str) -> Type:
        """
        获取注册的类

        Args:
            name: 注册名称

        Returns:
            注册的类

        Raises:
            KeyError: 未找到对应的注册
        """
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise KeyError(
                f"{self.name} '{name}' 未注册。\n"
                f"可用选项: {available}"
            )
        return self._registry[name]

    def list_available(self) -> list:
        """列出所有可用模块"""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry


# 全局注册器实例
FEATURE_ENGINE_REGISTRY = Registry("feature_engine")
SURROGATE_REGISTRY = Registry("surrogate")
OPTIMIZER_REGISTRY = Registry("optimizer")
POSTPROCESSOR_REGISTRY = Registry("postprocessor")
CONSTRAINT_HANDLER_REGISTRY = Registry("constraint_handler")
DATALOADER_REGISTRY = Registry("dataloader")


# 装饰器别名
register_feature = FEATURE_ENGINE_REGISTRY.register
register_surrogate = SURROGATE_REGISTRY.register
register_optimizer = OPTIMIZER_REGISTRY.register
register_postprocessor = POSTPROCESSOR_REGISTRY.register
register_constraint = CONSTRAINT_HANDLER_REGISTRY.register
register_dataloader = DATALOADER_REGISTRY.register
