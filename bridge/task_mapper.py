"""Task Mapper.

Maps task types to src/ modules and agent skills.
"""

from typing import Dict, List


TASK_TYPE_MAPPING = {
    "ml_pipeline": {
        "modules": ["data_loading", "feature_engineering", "model_building"],
        "agent_skill": "ml_pipeline_execution",
    },
    "optimization": {
        "modules": ["data_loading", "model_building", "optimization"],
        "agent_skill": "optimization_execution",
    },
    "soft_sensing": {
        "modules": ["data_loading", "feature_engineering", "surrogate_models", "sensing"],
        "agent_skill": "soft_sensing_execution",
    },
    "dae_training": {
        "modules": ["feature_engineering"],
        "agent_skill": "dae_training",
        "class": "DenoisingAutoencoder",
        "module": "src.features.DAE",
    },
    "surrogate_training": {
        "modules": ["surrogate_models"],
        "agent_skill": "surrogate_training",
        "class": "BaseSurrogate",
        "module": "src.core.factory",
    },
    "optimize": {
        "modules": ["optimization"],
        "agent_skill": "optimization_execution",
        "class": "BaseOptimizer",
        "module": "src.optimization.base",
    },
}


class TaskMapper:
    """Maps tasks to modules and skills."""

    def __init__(self):
        self._mapping = TASK_TYPE_MAPPING

    def map_task_to_modules(self, task_type: str) -> List[str]:
        """Map task type to required src/ modules."""
        return self._mapping.get(task_type, {}).get("modules", [])

    def map_task_to_skill(self, task_type: str) -> str:
        """Map task type to agent skill."""
        return self._mapping.get(task_type, {}).get("agent_skill", "")

    def map_task_to_class(self, task_type: str) -> str:
        """Map task type to class name."""
        return self._mapping.get(task_type, {}).get("class", "")

    def map_task_to_module(self, task_type: str) -> str:
        """Map task type to module path."""
        return self._mapping.get(task_type, {}).get("module", "")

    def map_pipeline_stage(self, stage: str) -> Dict[str, str]:
        """Map pipeline stage to src/ module and class."""
        mapping = {
            "feature_engineering": {"module": "src.features.DAE", "class": "DenoisingAutoencoder"},
            "surrogate_models": {"module": "src.models.surrogates.elm", "class": "ExtremeLearningMachine"},
            "optimization": {"module": "src.optimization.algorithms.nsga2", "class": "NSGA2"},
        }
        return mapping.get(stage, {})