"""
DeepInd 配置管理器
统一管理模型配置和优化器配置
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """
    配置管理器

    统一管理模型配置和优化器配置，提供：
    - 默认配置
    - 配置验证
    - 配置加载/保存
    - 配置合并
    """

    # 默认模型配置
    DEFAULT_MODEL_CONFIG = {
        "ELM": {
            "hidden_dim": 64,
            "activation": "relu",
        },
        "GaussianProcess": {
            "noise_var": 1e-5,
            "length_scale": 1.0,
            "output_scale": 1.0,
        },
        "NeuralNetwork": {
            "hidden_dims": [128, 64, 32],
            "learning_rate": 0.001,
            "epochs": 1000,
            "batch_size": 32,
            "weight_decay": 1e-5,
            "activation": "relu",
            "dropout": 0.0,
        },
        "XGBoost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "min_child_weight": 1,
        },
    }

    # 默认优化器配置
    DEFAULT_OPTIMIZER_CONFIG = {
        "AutoOptimizer": {},
        "LBFGS": {
            "n_iter": 100,
            "tol": 1e-6,
            "m": 10,
            "verbose": True,
        },
        "CG": {
            "n_iter": 100,
            "tol": 1e-6,
            "method": "FR",
            "verbose": True,
        },
        "Bayesian": {
            "n_iter": 50,
            "n_initial": 10,
            "acquisition": "EI",
            "verbose": True,
        },
        "Genetic": {
            "n_iter": 100,
            "pop_size": 50,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "elite_ratio": 0.1,
            "verbose": True,
        },
    }

    # 配置参数范围验证
    PARAM_RANGES = {
        # 模型参数范围
        "ELM": {
            "hidden_dim": (1, 10000),
            "activation": ["relu", "sigmoid", "tanh"],
        },
        "GaussianProcess": {
            "noise_var": (1e-10, 1.0),
            "length_scale": (0.01, 100.0),
            "output_scale": (0.01, 100.0),
        },
        "NeuralNetwork": {
            "hidden_dims": None,  # 特殊处理
            "learning_rate": (1e-6, 1.0),
            "epochs": (1, 100000),
            "batch_size": (1, 1024),
            "weight_decay": (0.0, 1.0),
            "activation": ["relu", "sigmoid", "tanh"],
            "dropout": (0.0, 0.9),
        },
        "XGBoost": {
            "n_estimators": (1, 10000),
            "max_depth": (1, 50),
            "learning_rate": (0.001, 1.0),
            "subsample": (0.1, 1.0),
            "colsample_bytree": (0.1, 1.0),
            "reg_alpha": (0.0, 100.0),
            "reg_lambda": (0.0, 100.0),
            "min_child_weight": (0, 100),
        },
        # 优化器参数范围
        "LBFGS": {
            "n_iter": (1, 10000),
            "tol": (1e-12, 1e-2),
            "m": (1, 100),
        },
        "CG": {
            "n_iter": (1, 10000),
            "tol": (1e-12, 1e-2),
            "method": ["FR", "PR", "HS"],
        },
        "Bayesian": {
            "n_iter": (1, 1000),
            "n_initial": (1, 100),
            "acquisition": ["EI", "PI", "UCB"],
        },
        "Genetic": {
            "n_iter": (1, 10000),
            "pop_size": (10, 1000),
            "mutation_rate": (0.001, 0.5),
            "crossover_rate": (0.1, 1.0),
            "elite_ratio": (0.0, 0.5),
        },
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径（可选）
        """
        self.config_path = config_path
        self.config = self._create_default_config()

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def _create_default_config(self) -> Dict[str, Any]:
        """创建默认配置"""
        return {
            "model": {
                "type": "NeuralNetwork",
                "params": self.DEFAULT_MODEL_CONFIG["NeuralNetwork"].copy(),
            },
            "optimizer": {
                "type": "AutoOptimizer",
                "params": {},
            },
            "data": {
                "input_dim": 10,
                "output_dim": 1,
            },
            "optimization": {
                "bounds": None,
                "constraints": [],
            },
        }

    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.config["model"]

    def get_optimizer_config(self) -> Dict[str, Any]:
        """获取优化器配置"""
        return self.config["optimizer"]

    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """获取指定模型的默认参数"""
        return self.DEFAULT_MODEL_CONFIG.get(model_type, {}).copy()

    def get_optimizer_params(self, optimizer_type: str) -> Dict[str, Any]:
        """获取指定优化器的默认参数"""
        return self.DEFAULT_OPTIMIZER_CONFIG.get(optimizer_type, {}).copy()

    def set_model(self, model_type: str, params: Optional[Dict[str, Any]] = None):
        """
        设置模型配置

        Args:
            model_type: 模型类型
            params: 模型参数（可选）
        """
        self.config["model"]["type"] = model_type
        if params is not None:
            self.config["model"]["params"] = params
        else:
            self.config["model"]["params"] = self.DEFAULT_MODEL_CONFIG.get(
                model_type, {}
            ).copy()

    def set_optimizer(
        self, optimizer_type: str, params: Optional[Dict[str, Any]] = None
    ):
        """
        设置优化器配置

        Args:
            optimizer_type: 优化器类型
            params: 优化器参数（可选）
        """
        self.config["optimizer"]["type"] = optimizer_type
        if params is not None:
            self.config["optimizer"]["params"] = params
        else:
            self.config["optimizer"]["params"] = self.DEFAULT_OPTIMIZER_CONFIG.get(
                optimizer_type, {}
            ).copy()

    def validate_params(
        self, param_type: str, params: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        验证参数是否在有效范围内

        Args:
            param_type: 参数类型（模型或优化器名称）
            params: 参数字典

        Returns:
            (是否有效, 错误信息)
        """
        ranges = self.PARAM_RANGES.get(param_type, {})
        if not ranges:
            return True, None

        for key, value in params.items():
            if key not in ranges:
                continue

            range_val = ranges[key]
            if range_val is None:
                continue

            if isinstance(range_val, list):
                if value not in range_val:
                    return False, f"{key} 必须是 {range_val} 之一，得到 {value}"
            elif isinstance(range_val, tuple):
                min_val, max_val = range_val
                if not (min_val <= value <= max_val):
                    return False, f"{key} 必须在 [{min_val}, {max_val}] 范围内，得到 {value}"

        return True, None

    def merge_config(self, partial_config: Dict[str, Any]):
        """
        合并部分配置

        Args:
            partial_config: 部分配置字典
        """
        if "model" in partial_config:
            if "type" in partial_config["model"]:
                self.config["model"]["type"] = partial_config["model"]["type"]
            if "params" in partial_config["model"]:
                self.config["model"]["params"].update(partial_config["model"]["params"])

        if "optimizer" in partial_config:
            if "type" in partial_config["optimizer"]:
                self.config["optimizer"]["type"] = partial_config["optimizer"]["type"]
            if "params" in partial_config["optimizer"]:
                self.config["optimizer"]["params"].update(
                    partial_config["optimizer"]["params"]
                )

        if "data" in partial_config:
            self.config["data"].update(partial_config["data"])

        if "optimization" in partial_config:
            self.config["optimization"].update(partial_config["optimization"])

    def load_config(self, config_path: str) -> bool:
        """
        从文件加载配置

        Args:
            config_path: 配置文件路径

        Returns:
            是否加载成功
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if loaded:
                self.merge_config(loaded)
            return True
        except Exception as e:
            print(f"加载配置失败: {e}")
            return False

    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        保存配置到文件

        Args:
            config_path: 配置文件路径（可选，使用初始化时的路径）

        Returns:
            是否保存成功
        """
        path = config_path or self.config_path
        if not path:
            print("未指定配置文件路径")
            return False

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(
                    self.config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """返回配置字典"""
        return self.config.copy()

    def __repr__(self) -> str:
        return f"ConfigManager(model={self.config['model']['type']}, optimizer={self.config['optimizer']['type']})"


def create_config_manager(
    model_type: str = "NeuralNetwork",
    optimizer_type: str = "AutoOptimizer",
    model_params: Optional[Dict[str, Any]] = None,
    optimizer_params: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
) -> ConfigManager:
    """
    工厂函数：创建配置管理器

    Args:
        model_type: 模型类型
        optimizer_type: 优化器类型
        model_params: 模型参数
        optimizer_params: 优化器参数
        config_path: 配置文件路径

    Returns:
        ConfigManager实例
    """
    manager = ConfigManager(config_path)
    manager.set_model(model_type, model_params)
    manager.set_optimizer(optimizer_type, optimizer_params)
    return manager


# 全局默认配置管理器实例
_default_manager: Optional[ConfigManager] = None


def get_default_config_manager() -> ConfigManager:
    """获取全局默认配置管理器"""
    global _default_manager
    if _default_manager is None:
        _default_manager = ConfigManager()
    return _default_manager