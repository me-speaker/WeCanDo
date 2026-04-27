"""DAE-ELM CEO Agent.

CEO Agent specialized for DAE-ELM chemical formulation optimization.
Coordinates the 3-module architecture: DAE → Surrogate → Optimizer.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
_parent = Path(__file__).resolve().parents[3]
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from agents.ceo import CEOAgent as BaseCEOAgent
from agents.base import BaseAgent
from .base_agent import DAEELMBaseAgent, DAEELMTaskType, DAEELMMessageType


class DAEELMCEOAgent(DAEELMBaseAgent, BaseCEOAgent):
    """CEO Agent for DAE-ELM project.

    Responsibilities:
    - Evaluate task complexity for chemical formulation optimization
    - Activate appropriate agents based on complexity
    - Coordinate the 3-module architecture (DAE → Surrogate → Optimizer)
    """

    def __init__(self):
        # Initialize BaseAgent directly to avoid CEOAgent's config_path signature conflict
        BaseAgent.__init__(self, name="daeelm_ceo", agent_type="orchestrator")
        self.active_agents: List[str] = []

    def get_activation_list(self, complexity: str) -> List[str]:
        """Get activation list based on DAE-ELM complexity."""
        activation_map = {
            "simple": ["daeelm_ceo", "requirements_analyst", "developer"],
            "medium": ["daeelm_ceo", "requirements_analyst", "developer", "tester"],
            "complex": ["daeelm_ceo", "requirements_analyst", "architect", "developer", "tester", "delivery"]
        }
        return activation_map.get(complexity, activation_map["simple"])

    def evaluate_dae_elm_complexity(self, requirements: str) -> Dict[str, Any]:
        """Evaluate task complexity for DAE-ELM tasks."""
        requirements_lower = requirements.lower()
        
        # Check for multi-objective indicators
        multi_objective_keywords = ["multi-objective", "pareto", "nsga2", "multiple objectives"]
        needs_nsga2 = any(kw in requirements_lower for kw in multi_objective_keywords)
        
        # Check for closed-loop indicators
        closed_loop_keywords = ["closed-loop", "iterative", "feedback", "real-time"]
        needs_closed_loop = any(kw in requirements_lower for kw in closed_loop_keywords)
        
        # Determine complexity
        if needs_nsga2 and needs_closed_loop:
            level = "complex"
        elif needs_nsga2 or needs_closed_loop:
            level = "medium"
        else:
            level = "simple"
        
        return {
            "level": level,
            "n_objectives": 2 if needs_nsga2 else 1,
            "needs_nsga2": needs_nsga2,
            "needs_closed_loop": needs_closed_loop,
        }
