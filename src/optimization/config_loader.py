"""
DeepInd 优化器配置加载器
从 YAML 文件加载优化器配置
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_optimizer_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载优化器配置文件

    Args:
        config_path: 配置文件路径，默认使用 configs/optimizer_config.yaml

    Returns:
        配置字典
    """
    if config_path is None:
        config_dir = Path(__file__).parent / "configs"
        config_path = config_dir / "optimizer_config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_optimizer_params(optimizer_type: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    获取指定优化器的参数

    Args:
        optimizer_type: 优化器类型 (LBFGS, CG, BAYESIAN, GENETIC, AUTO)
        config: 配置字典

    Returns:
        优化器参数字典
    """
    if config is None:
        config = load_optimizer_config()

    optimizer_type = optimizer_type.upper()
    if optimizer_type in config:
        return config[optimizer_type]

    raise ValueError(f"未找到优化器配置: {optimizer_type}")


def create_optimizer_from_config(
    optimizer_type: str,
    config: Optional[Dict[str, Any]] = None,
    surrogate_model=None,
) -> Any:
    """
    从配置创建优化器实例

    Args:
        optimizer_type: 优化器类型
        config: 配置字典
        surrogate_model: 代理模型

    Returns:
        优化器实例
    """
    from .auto_optimizer import AutoOptimizer, create_optimizer

    params = get_optimizer_params(optimizer_type, config)
    return create_optimizer(
        optimizer_type=optimizer_type,
        surrogate_model=surrogate_model,
        **params,
    )
