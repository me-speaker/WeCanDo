"""CEO Agent - orchestrates all tasks and manages agent lifecycle."""
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentMessage, AgentStatus


class CEOAgent(BaseAgent):
    """Chief Executive Officer agent that orchestrates the system."""

    def __init__(self, config_path: str = "/home/speaker/origin_ws/ai_company/.company/ceo_config.yaml"):
        super().__init__(name="ceo", agent_type="orchestrator")
        self.config_path = config_path
        self.config = self._load_config()
        self.active_agents: List[str] = []

    def _load_config(self) -> Dict[str, Any]:
        """Load CEO configuration from YAML file."""
        path = Path(self.config_path)
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process a message and determine agent activation strategy."""
        self.status = AgentStatus.ACTIVE

        if message.msg_type == "task_submitted":
            return self._handle_task_submitted(message)
        elif message.msg_type == "agent_completed":
            return self._handle_agent_completed(message)
        elif message.msg_type == "task_completed":
            return self._handle_task_completed(message)

        self.status = AgentStatus.IDLE
        return None

    def _handle_task_submitted(self, message: AgentMessage) -> AgentMessage:
        """Handle a new task submission and select appropriate agents."""
        task_content = message.content
        complexity = self._determine_complexity(task_content)

        strategy = self._select_activation_strategy(complexity)
        self.active_agents = strategy.get("agents", [])

        response_content = {
            "strategy": strategy,
            "selected_agents": self.active_agents,
            "complexity": complexity
        }

        return self.send_message(
            recipient=message.sender,
            msg_type="strategy_selected",
            content=response_content
        )

    def _determine_complexity(self, task_content: Any) -> str:
        """Determine task complexity based on content."""
        if isinstance(task_content, dict):
            if "complexity" in task_content:
                return task_content["complexity"]
            if "data_size" in task_content or "requirements" in task_content:
                return "medium"
        return "simple"

    def _select_activation_strategy(self, complexity: str) -> Dict[str, Any]:
        """Select activation strategy based on complexity."""
        strategies = self.config.get("ceo_config", {}).get("activation_strategy", {})
        if complexity == "complex":
            return strategies.get("complex_task", {})
        elif complexity == "medium":
            return strategies.get("medium_task", {})
        else:
            return strategies.get("simple_task", {})

    def _handle_agent_completed(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle agent completion notification."""
        if message.sender in self.active_agents:
            self.active_agents.remove(message.sender)

        if not self.active_agents:
            self.status = AgentStatus.COMPLETED

        return None

    def _handle_task_completed(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle task completion."""
        self.status = AgentStatus.COMPLETED
        return self.send_message(
            recipient="delivery",
            msg_type="deploy",
            content=message.content
        )