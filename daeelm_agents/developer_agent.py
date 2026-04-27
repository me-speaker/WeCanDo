"""DAE-ELM Developer Agent.

Implements the 3-module pipeline: DAE → Surrogate → Optimizer.
"""

import sys
from pathlib import Path
_parent = Path(__file__).resolve().parents[3]
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from agents.base import BaseAgent
from agents.developer import DeveloperAgent as BaseDeveloperAgent
from typing import Dict, Any, List
from .base_agent import DAEELMBaseAgent


class DAEELMDeveloperAgent(DAEELMBaseAgent, BaseDeveloperAgent):
    """Developer Agent for DAE-ELM project.

    Responsibilities:
    - Implement the DAE → Surrogate → Optimizer pipeline
    - Use src/ modules for core business logic
    - Wire components together
    """

    def __init__(self):
        BaseAgent.__init__(self, name="developer", agent_type="developer")

    def create_pipeline(self, architecture: Dict) -> "PipelineRunner":
        """Create a pipeline runner from architecture design."""
        from src.core.runner import TaskRunner
        return TaskRunner(architecture)

    def implement_dae(self, config: Dict):
        """Implement DAE feature engineering."""
        from src.features.dae import DenoisingAutoencoder
        return DenoisingAutoencoder(**config.get("params", {}))

    def implement_surrogate(self, config: Dict):
        """Implement surrogate model."""
        module_path = config["module"]
        class_name = config["class"]
        # Import from src/
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)(**config.get("params", {}))

    def implement_optimizer(self, config: Dict):
        """Implement optimizer."""
        module_path = config["module"]
        class_name = config["class"]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)(**config.get("params", {}))
