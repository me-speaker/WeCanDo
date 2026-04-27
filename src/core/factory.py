"""
DeepInd 工厂函数
根据配置动态创建模块实例
"""

from typing import Dict, Any

from .registry import (
    FEATURE_ENGINE_REGISTRY,
    SURROGATE_REGISTRY,
    OPTIMIZER_REGISTRY,
    POSTPROCESSOR_REGISTRY,
    CONSTRAINT_HANDLER_REGISTRY,
    DATALOADER_REGISTRY,
)


def create_feature_engine(cfg: Dict[str, Any]):
    """
    创建特征工程模块

    Args:
        cfg: 配置字典，格式 {"type": "DAE", "params": {...}}

    Returns:
        特征工程实例
    """
    cfg_type = cfg.get("type")
    if not cfg_type:
        raise ValueError("配置中缺少 'type' 字段")

    params = cfg.get("params", {})
    return FEATURE_ENGINE_REGISTRY.get(cfg_type)(**params)


def create_surrogate(cfg: Dict[str, Any]):
    """
    创建代理模型

    Args:
        cfg: 配置字典，格式 {"type": "ELM", "params": {...}}

    Returns:
        代理模型实例
    """
    cfg_type = cfg.get("type")
    if not cfg_type:
        raise ValueError("配置中缺少 'type' 字段")

    params = cfg.get("params", {})
    return SURROGATE_REGISTRY.get(cfg_type)(**params)


def create_optimizer(cfg: Dict[str, Any]):
    """
    创建优化器

    Args:
        cfg: 配置字典，格式 {"type": "NSGA2", "params": {...}}

    Returns:
        优化器实例
    """
    cfg_type = cfg.get("type")
    if not cfg_type:
        raise ValueError("配置中缺少 'type' 字段")

    params = cfg.get("params", {})
    return OPTIMIZER_REGISTRY.get(cfg_type)(**params)


def create_postprocessor(cfg: Dict[str, Any]):
    """
    创建后处理器

    Args:
        cfg: 配置字典，格式 {"type": "saver", "params": {...}}

    Returns:
        后处理器实例
    """
    cfg_type = cfg.get("type")
    if not cfg_type:
        raise ValueError("配置中缺少 'type' 字段")

    params = cfg.get("params", {})
    return POSTPROCESSOR_REGISTRY.get(cfg_type)(**params)


def create_constraint_handler(cfg: Dict[str, Any]):
    """
    创建约束处理器

    Args:
        cfg: 配置字典，格式 {"handler": "penalty", "params": {...}}

    Returns:
        约束处理器实例
    """
    handler_type = cfg.get("handler")
    if not handler_type:
        raise ValueError("配置中缺少 'handler' 字段")

    params = cfg.get("params", {})
    return CONSTRAINT_HANDLER_REGISTRY.get(handler_type)(**params)


def create_dataloader(cfg: Dict[str, Any]):
    """
    创建数据加载器

    Args:
        cfg: 配置字典，格式 {"type": "FormulaDataLoader", "params": {...}}

    Returns:
        数据加载器实例
    """
    cfg_type = cfg.get("type")
    if not cfg_type:
        raise ValueError("配置中缺少 'type' 字段")

    params = cfg.get("params", {})
    return DATALOADER_REGISTRY.get(cfg_type)(**params)
