"""DAE-ELM Architect Agent.

Designs system architecture for the 3-module decoupled structure.
"""

import sys
from pathlib import Path
_parent = Path(__file__).resolve().parents[3]
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from agents.base import BaseAgent
from agents.architect import ArchitectAgent as BaseArchitectAgent
from typing import Dict, Any, List
from .base_agent import DAEELMBaseAgent


class DAEELMArchitectAgent(DAEELMBaseAgent, BaseArchitectAgent):
    """Architect Agent for DAE-ELM project.

    Responsibilities:
    - Design system architecture for DAE-ELM 3-module structure
    - Map requirements to module implementations in src/
    - Define interfaces between modules
    """

    def __init__(self):
        BaseAgent.__init__(self, name="architect", agent_type="architect")

    def design_architecture(self, task_definition: Dict) -> Dict:
        """Design architecture for given task."""
        return {
            "modules": {
                "feature_engineering": {
                    "class": "DenoisingAutoencoder",
                    "module": "src.features.dae",
                    "params": {"hidden_dim": 16, "latent_dim": 4}
                },
                "surrogate_models": {
                    "class": "ExtremeLearningMachine", 
                    "module": "src.models.surrogates.elm",
                    "params": {"hidden_dim": 20}
                },
                "optimization": {
                    "class": "NSGA2Optimizer",
                    "module": "src.optimization.algorithms.nsga2",
                    "params": {"pop_size": 50, "n_generations": 100}
                }
            },
            "data_flow": ["data_loading", "feature_engineering", "surrogate_models", "optimization"]
        }
