"""DAE-ELM Base Agent.

Base class for all DAE-ELM specialized agents.
Inherits from the core BaseAgent and adds DAE-ELM specific context.
"""

import sys
from pathlib import Path
_parent = Path(__file__).resolve().parents[3]
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from agents.base import BaseAgent, AgentMessage, AgentStatus

from enum import Enum
from typing import Any, Dict, List, Optional


class DAEELMTaskType(Enum):
    """DAE-ELM specific task types."""
    ML_PIPELINE = "ml_pipeline"
    OPTIMIZATION = "optimization"
    SOFT_SENSING = "soft_sensing"
    FEATURE_ENGINEERING = "feature_engineering"
    SURROGATE_MODELING = "surrogate_modeling"
    CLOSED_LOOP = "closed_loop"


class DAEELMMessageType(Enum):
    """DAE-ELM specific message types."""
    TASK_ASSIGNMENT = "task_assignment"
    REQUIREMENTS_ANALYZED = "requirements_analyzed"
    ARCHITECTURE_READY = "architecture_ready"
    IMPLEMENTATION_RESULT = "implementation_result"
    TEST_RESULTS = "test_results"
    DELIVERY_COMPLETE = "delivery_complete"
    DAE_CONFIG = "dae_config"
    SURROGATE_TRAIN = "surrogate_train"
    OPTIMIZE_REQUEST = "optimize_request"
    CLOSED_LOOP_ITERATION = "closed_loop_iteration"
    EVALUATE_FORMULA = "evaluate_formula"
    PARETO_FRONT = "pareto_front"


class DAEELMBaseAgent(BaseAgent):
    """Base class for DAE-ELM specialized agents."""

    def __init__(self, name: str, agent_type: str):
        super().__init__(name=name, agent_type=agent_type)
        self.dae_elm_context: Dict[str, Any] = {
            "three_module_architecture": {
                "feature_engineering": {
                    "components": ["DAE", "transformers", "selectors", "domain"],
                    "src_module": "src.features",
                },
                "surrogate_models": {
                    "components": ["ELM", "GP", "NN", "XGBoost"],
                    "src_module": "src.models.surrogates",
                },
                "optimization": {
                    "components": ["L-BFGS", "CG", "Bayesian", "Genetic", "NSGA2"],
                    "src_module": "src.optimization",
                },
            },
            "src_path": Path(__file__).resolve().parents[1] / "src",
        }

    def get_src_module(self, module_name: str):
        """Get a src module by name."""
        import importlib
        src_path = self.dae_elm_context["src_path"]
        sys.path.insert(0, str(src_path.parent))
        return importlib.import_module(module_name)
