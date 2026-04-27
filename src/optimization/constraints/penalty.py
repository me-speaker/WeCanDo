"""
DeepInd 约束处理 - 罚函数法
将约束条件以惩罚项的形式加入目标函数
"""

import numpy as np
from typing import List, Callable, Dict, Any

from ...core.base import BaseConstraint
from ...core.registry import register_constraint
from ...utils.logger import get_logger


@register_constraint("Penalty")
class PenaltyMethod(BaseConstraint):
    """
    罚函数法约束处理器

    通过惩罚因子将约束违反量加入到目标函数值中

    Args:
        constraints: 约束条件列表
        penalty_factor: 初始惩罚系数
        growth_factor: 惩罚系数增长因子
    """

    def __init__(
        self,
        constraints: List[Dict[str, Any]] = None,
        penalty_factor: float = 1.0,
        growth_factor: float = 10.0,
    ):
        self.constraints = constraints or []
        self.penalty_factor = penalty_factor
        self.growth_factor = growth_factor

        self.logger = get_logger("DeepInd::Constraint::Penalty")

        if self.constraints:
            self.logger.info(f"PenaltyMethod: 初始化, 约束数量={len(self.constraints)}")
        else:
            self.logger.info("PenaltyMethod: 初始化, 无约束")

    def wrap(self, objectives: List[Callable]) -> List[Callable]:
        """
        将约束包装到目标函数中

        Args:
            objectives: 原始目标函数列表

        Returns:
            包装后的目标函数列表
        """
        if not self.constraints:
            return objectives

        def create_wrapped_objective(obj_idx: int, original_obj: Callable):
            def wrapped_objective(x: np.ndarray) -> float:
                original_value = original_obj(x)
                penalty = self._calc_penalty(x)
                return original_value + self.penalty_factor * penalty

            return wrapped_objective

        wrapped_objectives = []
        for i, obj in enumerate(objectives):
            wrapped_objectives.append(create_wrapped_objective(i, obj))

        return wrapped_objectives

    def _calc_penalty(self, x: np.ndarray) -> float:
        """计算惩罚项"""
        total_penalty = 0.0
        for constraint in self.constraints:
            violation = self._constraint_violation(x, constraint)
            total_penalty += violation ** 2
        return total_penalty

    def _constraint_violation(self, x: np.ndarray, constraint: Dict[str, Any]) -> float:
        """计算单个约束的违反量"""
        constraint_type = constraint.get("type", "le")
        expression = constraint.get("expression", "0")

        try:
            # Pass x directly so x[0], x[1] etc work as array indexing
            constraint_value = eval(expression, {"__builtins__": {}}, {'x': x})
        except Exception:
            constraint_value = 0.0

        if constraint_type == "eq":
            return abs(constraint_value)
        else:
            return max(0, constraint_value)

    def evaluate_constraint_violation(self, x: np.ndarray) -> float:
        """计算解的总体约束违反量"""
        return self._calc_penalty(x)

    def update_penalty_factor(self):
        """更新惩罚系数"""
        old_factor = self.penalty_factor
        self.penalty_factor *= self.growth_factor
        self.logger.debug(f"PenaltyMethod: 惩罚系数更新 {old_factor:.2f} -> {self.penalty_factor:.2f}")
