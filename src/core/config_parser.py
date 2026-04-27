"""
YAML配置解析器
从YAML文件加载并解析决策变量、目标变量、约束等配置
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Union

from .interfaces.base import (
    BaseConfigParser,
    ConfigSchema,
    DecisionVariable,
    ObjectiveVariable,
    ConstraintSpec,
)


class YAMLConfigParser(BaseConfigParser):
    """
    YAML配置文件解析器
    支持加载和解析标准化的YAML配置文件
    """

    def __init__(self):
        self._current_schema: ConfigSchema = None

    def load(self, path: str) -> Dict[str, Any]:
        """
        从YAML文件加载配置

        Args:
            path: YAML配置文件路径

        Returns:
            原始配置字典

        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML解析错误
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")

        with open(path_obj, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"配置文件为空: {path}")

        return config

    def parse(self, config: Dict[str, Any]) -> ConfigSchema:
        """
        解析配置字典为ConfigSchema

        Args:
            config: 原始配置字典

        Returns:
            ConfigSchema对象

        Raises:
            ValueError: 配置格式错误
        """
        # Parse decision variables
        decision_vars = self._parse_decision_variables(config.get("decision_variables", []))
        objectives = self._parse_objectives(config.get("objectives", []))
        constraints = self._parse_constraints(config.get("constraints", {}).get("list", []))

        schema = ConfigSchema(
            task_name=config.get("task_name", "default_task"),
            task_type=config.get("task_type", "single_objective"),
            decision_variables=decision_vars,
            objectives=objectives,
            constraints=constraints,
            data_config=config.get("data", {}),
            model_config=config.get("models", {}),
            optimizer_config=config.get("optimizer", {}),
        )

        self._current_schema = schema
        return schema

    def _parse_decision_variables(self, dvars: List[Dict[str, Any]]) -> List[DecisionVariable]:
        """解析决策变量列表"""
        result = []
        for dv in dvars:
            if isinstance(dv, dict):
                result.append(DecisionVariable(
                    name=dv.get("name", ""),
                    bounds=tuple(dv.get("bounds", [0.0, 1.0])),
                    description=dv.get("description", ""),
                    unit=dv.get("unit", ""),
                    var_type=dv.get("type", "continuous"),
                ))
            elif isinstance(dv, (list, tuple)):
                # Support compact format: [name, low, high]
                result.append(DecisionVariable(
                    name=dv[0],
                    bounds=(float(dv[1]), float(dv[2])),
                    description=dv[3] if len(dv) > 3 else "",
                ))
        return result

    def _parse_objectives(self, objectives: List[Dict[str, Any]]) -> List[ObjectiveVariable]:
        """解析目标变量列表"""
        result = []
        for obj in objectives:
            if isinstance(obj, dict):
                result.append(ObjectiveVariable(
                    name=obj.get("name", ""),
                    direction=obj.get("direction", "minimize"),
                    description=obj.get("description", ""),
                    unit=obj.get("unit", ""),
                    target_idx=obj.get("target_idx", 0),
                ))
        return result

    def _parse_constraints(self, constraints: List[Dict[str, Any]]) -> List[ConstraintSpec]:
        """解析约束条件列表"""
        result = []
        for c in constraints:
            if isinstance(c, dict):
                result.append(ConstraintSpec(
                    ctype=c.get("type", "ineq"),
                    expression=c.get("expression", ""),
                    description=c.get("description", ""),
                    penalty_weight=c.get("penalty_weight", 1.0),
                ))
            elif isinstance(c, str):
                # Support expression-only format: "x[0] + x[1] - 1.0"
                # Infer type from expression
                ctype = "eq" if "-1.0" in c or "-1" in c else "ineq"
                result.append(ConstraintSpec(
                    ctype=ctype,
                    expression=c,
                ))
        return result

    def validate(self, schema: ConfigSchema) -> bool:
        """
        验证配置的有效性

        Args:
            schema: 配置模式对象

        Returns:
            是否有效

        Raises:
            ValueError: 配置无效
        """
        if schema.n_decision_vars == 0:
            raise ValueError("至少需要定义一个决策变量")

        if schema.n_objectives == 0:
            raise ValueError("至少需要定义一个目标变量")

        # Validate bounds
        for dv in schema.decision_variables:
            if dv.bounds[0] == dv.bounds[1]:
                raise ValueError(f"决策变量 {dv.name} 的上下界相同")

        # Validate objective directions
        for obj in schema.objectives:
            if obj.direction not in ("minimize", "maximize"):
                raise ValueError(f"目标变量 {obj.name} 的优化方向无效")

        # Validate constraint expressions are not empty
        for c in schema.constraints:
            if not c.expression:
                raise ValueError("约束表达式不能为空")

        return True

    def to_yaml(self, schema: ConfigSchema, path: str = None) -> Union[str, None]:
        """
        将ConfigSchema导出为YAML格式

        Args:
            schema: 配置模式对象
            path: 输出路径，如果为None则返回字符串

        Returns:
            YAML字符串（如果path为None），否则返回None
        """
        config = self._schema_to_dict(schema)
        yaml_str = yaml.dump(config, allow_unicode=True, default_flow_style=False)

        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(yaml_str)
            return None
        return yaml_str

    def _schema_to_dict(self, schema: ConfigSchema) -> Dict[str, Any]:
        """将ConfigSchema转换为字典"""
        return {
            "task_name": schema.task_name,
            "task_type": schema.task_type,
            "decision_variables": [
                {
                    "name": dv.name,
                    "bounds": list(dv.bounds),
                    "description": dv.description,
                    "unit": dv.unit,
                    "type": dv.var_type,
                }
                for dv in schema.decision_variables
            ],
            "objectives": [
                {
                    "name": obj.name,
                    "direction": obj.direction,
                    "description": obj.description,
                    "unit": obj.unit,
                    "target_idx": obj.target_idx,
                }
                for obj in schema.objectives
            ],
            "constraints": {
                "list": [
                    {
                        "type": c.ctype,
                        "expression": c.expression,
                        "description": c.description,
                        "penalty_weight": c.penalty_weight,
                    }
                    for c in schema.constraints
                ]
            },
            "data": schema.data_config,
            "models": schema.model_config,
            "optimizer": schema.optimizer_config,
        }
