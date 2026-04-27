"""
DeepInd 后处理器 - 结果储存
保存模型参数、配置、优化结果到文件
"""

import os
import json
from typing import Dict, Any

import numpy as np
import torch

from ...core.base import BasePostProcessor
from ...core.registry import register_postprocessor
from ...utils.logger import get_logger


@register_postprocessor("saver")
class Saver(BasePostProcessor):
    """
    结果储存器

    保存内容:
    - 模型参数 (.pth)
    - 训练指标 (.json)
    - Pareto 解集 (.csv)
    - 任务配置 (.json)

    Args:
        output_dir: 输出目录
        save_model: 是否保存模型
        save_config: 是否保存配置
    """

    def __init__(
        self,
        output_dir: str = "output",
        save_model: bool = True,
        save_config: bool = True,
    ):
        self.output_dir = output_dir
        self.save_model = save_model
        self.save_config = save_config

        self.logger = get_logger("DeepInd::Saver")

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        保存结果

        Args:
            results: 结果字典

        Returns:
            原始结果（不做修改）
        """
        self.logger.info(f"Saver: 开始保存结果到 {self.output_dir}")

        # 创建目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 保存 Pareto 解集
        if "solutions" in results:
            solutions = results["solutions"]
            if isinstance(solutions, np.ndarray) and solutions.size > 0:
                save_path = os.path.join(self.output_dir, "pareto_front.csv")
                np.savetxt(save_path, solutions, delimiter=",", fmt="%.6f")
                self.logger.debug(f"  - Pareto 解集已保存: {save_path}")

        # 保存推荐方案
        if "recommendations" in results:
            rec_path = os.path.join(self.output_dir, "recommendations.json")
            with open(rec_path, "w", encoding="utf-8") as f:
                json.dump(results["recommendations"], f, indent=2, ensure_ascii=False)
            self.logger.debug(f"  - 推荐方案已保存: {rec_path}")

        # 保存训练指标
        if "metrics" in results:
            metrics_path = os.path.join(self.output_dir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(results["metrics"], f, indent=2)
            self.logger.debug(f"  - 训练指标已保存: {metrics_path}")

        # 保存分析结果
        if "analysis" in results:
            analysis_path = os.path.join(self.output_dir, "analysis.json")
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(results["analysis"], f, indent=2)
            self.logger.debug(f"  - 分析结果已保存: {analysis_path}")

        # 保存任务配置
        if self.save_config and "config" in results:
            config_path = os.path.join(self.output_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(results["config"], f, indent=2, ensure_ascii=False)
            self.logger.debug(f"  - 任务配置已保存: {config_path}")

        self.logger.info(f"Saver: 结果保存完成")

        return results
