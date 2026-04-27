"""Skill Registry.

Central registry for DAE-ELM agent skills.
"""

from typing import Dict, List, Callable, Any


class SkillRegistry:
    """Registry for agent skills."""

    _skills: Dict[str, Dict[str, Any]] = {
        "data_loading": {
            "name": "data_loading",
            "description": "Load chemical formulation data from CSV",
            "priority": 10,
            "module": "skill_hub.core.data_loader",
        },
        "visualization": {
            "name": "visualization",
            "description": "Visualize Pareto fronts and training curves",
            "priority": 8,
            "module": "skill_hub.oss.visualization",
        },
        "pipeline_execution": {
            "name": "pipeline_execution",
            "description": "Execute DAE → Surrogate → Optimizer pipeline",
            "priority": 9,
            "module": "skill_hub.core.pipeline",
        },
        "model_evaluation": {
            "name": "model_evaluation",
            "description": "Evaluate surrogate model performance",
            "priority": 8,
            "module": "skill_hub.core.evaluation",
        },
    }

    @classmethod
    def get_skill(cls, name: str) -> Dict[str, Any]:
        """Get skill by name."""
        return cls._skills.get(name)

    @classmethod
    def list_skills(cls) -> List[str]:
        """List all available skills."""
        return list(cls._skills.keys())

    @classmethod
    def register_skill(cls, name: str, skill_def: Dict[str, Any]):
        """Register a new skill."""
        cls._skills[name] = skill_def
