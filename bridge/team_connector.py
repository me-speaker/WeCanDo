"""Team Connector.

Bridges DAE-ELM agents to the TaskRunner in src/core/runner.py.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.runner import TaskRunner
from src.core.registry import SURROGATE_REGISTRY, OPTIMIZER_REGISTRY
from src.features.DAE import DenoisingAutoencoder


class TeamConnector:
    """Connects agent decisions to TaskRunner execution."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.runner: Optional[TaskRunner] = None
        self.active_pipeline = None
        self.agent_status: Dict[str, str] = {}

    def create_pipeline(self, config: Dict) -> TaskRunner:
        """Create TaskRunner from configuration."""
        self.runner = TaskRunner(config)
        return self.runner

    def map_agent_to_module(self, agent_name: str, task: Dict) -> str:
        """Map agent task to src/ module."""
        mapping = {
            "requirements_analyst": "src.core.config_parser",
            "architect": "src.core.factory",
            "developer": "src.core.runner",
            "tester": "src.analysis.benchmark",
            "delivery": "deployment",
            "ceo": "src.core.config_manager",
        }
        return mapping.get(agent_name, "src.core")

    def sync_status(self) -> Dict[str, str]:
        """Sync status between agents and pipeline."""
        if self.runner:
            return {
                "pipeline_status": "ready",
                "runner": str(self.runner),
                "agent_status": self.agent_status,
            }
        return {"pipeline_status": "not_initialized", "agent_status": self.agent_status}

    def update_agent_status(self, agent_name: str, status: str) -> None:
        """Update status for a specific agent."""
        self.agent_status[agent_name] = status

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline execution status."""
        if not self.runner:
            return {"status": "not_initialized"}

        return {
            "status": "ready",
            "has_dataloader": self.runner.dataloader is not None,
            "has_feature_engine": self.runner.feature_engine is not None,
            "has_surrogate": self.runner.surrogate is not None,
            "has_optimizer": self.runner.optimizer is not None,
        }