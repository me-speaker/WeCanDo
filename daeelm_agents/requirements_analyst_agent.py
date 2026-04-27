"""DAE-ELM Requirements Analyst Agent.

Parses DAE-ELM config files and outputs structured task definitions.
"""

import sys
from pathlib import Path
_parent = Path(__file__).resolve().parents[3]
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from agents.base import BaseAgent
from agents.requirements_analyst import RequirementsAnalystAgent as BaseRequirementsAnalyst
from typing import Dict, Any, List
from .base_agent import DAEELMBaseAgent, DAEELMTaskType


class DAEELMRequirementsAnalystAgent(DAEELMBaseAgent, BaseRequirementsAnalyst):
    """Requirements Analyst Agent for DAE-ELM project.

    Responsibilities:
    - Parse DAE-ELM YAML configuration files
    - Extract decision variables, objectives, and constraints
    - Output structured task definitions
    """

    def __init__(self):
        BaseAgent.__init__(self, name="requirements_analyst", agent_type="analyzer")
        self.current_requirements: Dict = {}

    def parse_config(self, config_path: str) -> Dict:
        """Parse DAE-ELM YAML configuration file."""
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return self._extract_task_definition(config)

    def _extract_task_definition(self, config: Dict) -> Dict:
        """Extract structured task definition from config."""
        return {
            "task_type": config.get("task_type", "optimization"),
            "decision_variables": config.get("decision_variables", []),
            "objectives": config.get("objectives", []),
            "constraints": config.get("constraints", []),
            "module_mapping": {
                "feature_engineering": ["DAE"],
                "surrogate_models": config.get("surrogate_models", ["ELM"]),
                "optimization": config.get("optimizers", ["NSGA2"]),
            },
            "data_requirements": config.get("data_requirements", {}),
        }
