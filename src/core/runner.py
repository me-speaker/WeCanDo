"""
DeepInd 任务运行器
读取配置，实例化模块，执行训练-优化-后处理流程
"""

import os
import json
from typing import Dict, Any, List, Callable, Optional

import numpy as np

# Import optimization module to trigger registry
from ..optimization import constraints  # noqa: F401

from .factory import (
    create_dataloader,
    create_feature_engine,
    create_surrogate,
    create_optimizer,
    create_postprocessor,
    create_constraint_handler,
)
from .base import BaseSurrogate
from ..utils.logger import get_logger, log_section


class TaskRunner:
    """
    任务运行器

    负责:
    1. 根据配置实例化所有模块
    2. 执行数据加载、训练、优化、后处理流程
    3. 处理输出格式（developer/user）
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化任务运行器

        Args:
            config: 任务配置字典
        """
        self.config = config
        self.task_name = config.get("task_name", "DeepInd_Task")
        self.output_mode = config.get("output_mode", "user")

        # Logger
        self.logger = get_logger(f"DeepInd::Runner")

        # 模块实例
        self.dataloader = None
        self.feature_engine = None
        self.surrogate = None
        self.optimizer = None
        self.constraint_handler = None
        self.postprocessors: List = []

        # 训练指标
        self.train_metrics: Dict[str, float] = {}

        # 初始化
        self._build_modules()

    def _build_modules(self):
        """根据配置实例化所有模块"""
        self.logger.info("TaskRunner: 开始初始化模块...")

        # 1. 数据加载器
        data_cfg = self.config.get("data", {})
        if data_cfg:
            self.dataloader = create_dataloader(data_cfg)
            self.logger.debug(f"  - DataLoader: {data_cfg.get('type')}")

        # 2. 特征工程
        feature_cfg = self.config.get("models", {}).get("feature_engine", {})
        if feature_cfg:
            self.feature_engine = create_feature_engine(feature_cfg)
            self.logger.debug(f"  - FeatureEngine: {feature_cfg.get('type')}")

        # 3. 代理模型
        surrogate_cfg = self.config.get("models", {}).get("surrogate", {})
        if surrogate_cfg:
            self.surrogate = create_surrogate(surrogate_cfg)
            self.logger.debug(f"  - Surrogate: {surrogate_cfg.get('type')}")

        # 4. 优化器
        optimizer_cfg = self.config.get("models", {}).get("optimizer", {})
        if optimizer_cfg:
            self.optimizer = create_optimizer(optimizer_cfg)
            self.logger.debug(f"  - Optimizer: {optimizer_cfg.get('type')}")

        # 5. 约束处理器
        constraint_cfg = self.config.get("constraints", {})
        if constraint_cfg:
            self.constraint_handler = create_constraint_handler(constraint_cfg)
            self.logger.debug(f"  - ConstraintHandler: {constraint_cfg.get('handler')}")

        # 6. 后处理器链
        postprocess_cfg = self.config.get("postprocess", {})
        if postprocess_cfg:
            chain = postprocess_cfg.get("chain", [])
            for i, proc_cfg in enumerate(chain):
                proc = create_postprocessor(proc_cfg)
                self.postprocessors.append(proc)
                self.logger.debug(f"  - PostProcessor[{i+1}]: {proc_cfg.get('type')}")

        self.logger.info("TaskRunner: 模块初始化完成")

    def train(self) -> Dict[str, float]:
        """
        执行训练流程

        Returns:
            训练指标字典
        """
        self.logger.info("TaskRunner: 开始训练流程...")
        log_section(self.logger, "训练阶段", '-')

        # 1. 加载数据
        self.logger.info("DataLoader: 加载数据")
        X_raw, y_raw = self.dataloader.load()
        self.logger.debug(f"  - 原始数据 shape: X={X_raw.shape}, y={y_raw.shape}")
        n_samples = X_raw.shape[0]

        # 2. 预处理
        self.logger.info("DataLoader: 执行预处理")
        X, y = self.dataloader.preprocess(X_raw, y_raw)
        self.logger.debug(f"  - 预处理后 shape: X={X.shape}, y={y.shape}")

        # 3. 特征工程
        self.logger.info("FeatureEngine: 特征提取")
        X_features = self.feature_engine.fit_transform(X)
        self.logger.debug(f"  - 特征维度: {self.feature_engine.feature_dim}")

        # 4. 代理模型训练
        self.logger.info("Surrogate: 训练模型")
        self.surrogate.fit(X_features, y)

        # 5. 评估
        y_pred = self.surrogate.predict(X_features)
        rmse = np.sqrt(np.mean((y_pred - y) ** 2))
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        self.train_metrics = {
            "RMSE": float(rmse),
            "R2": float(r2),
            "n_samples": n_samples,
        }
        self.logger.info(f"Surrogate: 训练完成, RMSE={rmse:.4f}, R²={r2:.4f}")

        log_section(self.logger, "训练阶段完成", '-')

        return self.train_metrics

    def optimize(self) -> Dict[str, Any]:
        """
        执行优化流程

        Returns:
            优化结果字典
        """
        self.logger.info("TaskRunner: 开始优化流程...")
        log_section(self.logger, "优化阶段", '-')

        # 1. 构建决策变量边界
        decision_vars = self.config.get("decision_variables", [])
        bounds = [tuple(dv["bounds"]) for dv in decision_vars]
        n_vars = len(bounds)
        self.logger.debug(f"  - 决策变量数: {n_vars}")

        # 2. 构建目标函数
        objectives = self._build_objectives()
        self.logger.debug(f"  - 目标函数数: {len(objectives)}")

        # 3. 构建约束
        constraints = self.config.get("constraints", {}).get("list", [])

        # 4. 执行优化
        self.logger.info(f"Optimizer: 开始优化")
        pareto_solutions = self.optimizer.optimize(objectives, bounds, constraints)
        self.logger.info(f"Optimizer: 优化完成, 找到 {len(pareto_solutions)} 个 Pareto 解")

        # 5. 格式化输出
        results = self._format_output(pareto_solutions)

        log_section(self.logger, "优化阶段完成", '-')

        return results

    def _build_objectives(self) -> List[Callable]:
        """
        构建目标函数列表

        Returns:
            目标函数列表
        """
        objectives = []
        obj_configs = self.config.get("objectives", [])

        for obj_cfg in obj_configs:
            obj_type = obj_cfg.get("type")
            target_idx = obj_cfg.get("target_idx", 0)

            if obj_type == "surrogate_predict":
                # 使用代理模型预测
                def make_obj(idx):
                    def obj(x):
                        x_tensor = np.array(x).reshape(1, -1)
                        # 特征工程
                        x_feat = self.feature_engine.transform(x_tensor)
                        # 预测
                        pred = self.surrogate.predict(x_feat)
                        return pred[0, idx]
                    return obj

                objectives.append(make_obj(target_idx))
            else:
                raise ValueError(f"未知的目标函数类型: {obj_type}")

        return objectives

    def _format_output(self, solutions: np.ndarray) -> Dict[str, Any]:
        """
        格式化输出（根据 output_mode）

        Args:
            solutions: Pareto 解集

        Returns:
            格式化后的结果
        """
        if self.output_mode == "developer":
            return {
                "mode": "developer",
                "solutions": solutions,
                "n_solutions": len(solutions),
                "raw_data": True,
            }
        else:
            # user mode - 生成摘要
            objectives = self.config.get("objectives", [])
            n_obj = len(objectives)

            # 提取各目标值
            obj_values = solutions[:, -n_obj:] if solutions.size > 0 else np.array([])

            recommendations = {}
            if obj_values.size > 0:
                for i, obj_cfg in enumerate(objectives):
                    name = obj_cfg["name"]
                    direction = obj_cfg.get("direction", "maximize")

                    if direction == "maximize":
                        best_idx = np.argmax(obj_values[:, i])
                    else:
                        best_idx = np.argmin(obj_values[:, i])

                    recommendations[f"best_{name}"] = {
                        "x": solutions[best_idx, :-n_obj].tolist(),
                        "objectives": obj_values[best_idx].tolist(),
                        "description": f"{name} 最优方案",
                    }

                # 综合最优（折中解）- 标准化后计算欧氏距离
                obj_normalized = (obj_values - obj_values.min(axis=0)) / (
                    obj_values.max(axis=0) - obj_values.min(axis=0) + 1e-10
                )
                distance_to_ideal = np.sqrt(np.sum(obj_normalized ** 2, axis=1))
                compromise_idx = np.argmin(distance_to_ideal)

                recommendations["best_compromise"] = {
                    "x": solutions[compromise_idx, :-n_obj].tolist(),
                    "objectives": obj_values[compromise_idx].tolist(),
                    "description": "综合最优（各目标权衡）",
                }

            return {
                "mode": "user",
                "recommendations": recommendations,
                "n_solutions": len(solutions),
            }

    def postprocess(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行后处理链

        Args:
            results: 优化结果

        Returns:
            处理后的结果
        """
        if not self.postprocessors:
            self.logger.info("PostProcessor: 无后处理步骤")
            return results

        self.logger.info(f"PostProcessor: 开始执行后处理链 (共 {len(self.postprocessors)} 步)")

        for i, proc in enumerate(self.postprocessors):
            proc_name = proc.__class__.__name__
            self.logger.info(f"  [{i+1}/{len(self.postprocessors)}] {proc_name}: 执行中...")
            results = proc(results)
            self.logger.info(f"  [{i+1}/{len(self.postprocessors)}] {proc_name}: 完成")

        self.logger.info("PostProcessor: 后处理链执行完成")

        return results

    def run(self) -> Dict[str, Any]:
        """
        执行完整流程: 训练 -> 优化 -> 后处理

        Returns:
            最终结果字典
        """
        log_section(self.logger, f"任务 [{self.task_name}] 开始", '=')

        try:
            # 1. 训练
            train_metrics = self.train()
            results = {"metrics": train_metrics}

            # 2. 优化
            optimize_results = self.optimize()
            results.update(optimize_results)

            # 3. 后处理
            results = self.postprocess(results)

            # 4. 最终输出
            log_section(
                self.logger,
                f"任务 [{self.task_name}] 执行完成",
                '='
            )
            self.logger.info(f"  - 训练 RMSE: {train_metrics.get('RMSE', 0):.4f}")
            self.logger.info(f"  - Pareto 解数量: {results.get('n_solutions', 0)}")

            return results

        except Exception as e:
            self.logger.error(f"任务执行出错: {str(e)}")
            raise
