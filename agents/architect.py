"""Architect Agent - designs system architecture for complex tasks."""
from typing import Any, Dict, List, Optional
from agents.base import BaseAgent, AgentMessage, AgentStatus


class ArchitectAgent(BaseAgent):
    """Agent responsible for designing system architecture."""

    def __init__(self):
        super().__init__(name="architect", agent_type="designer")
        self.architecture_template: Dict[str, Any] = {}

    def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process architecture design request."""
        self.status = AgentStatus.ACTIVE

        if message.msg_type == "DESIGN_ARCHITECTURE":
            result = self._design_architecture(message.content)
            self.status = AgentStatus.COMPLETED
            return self.send_message(
                recipient=message.sender,
                msg_type="ARCHITECTURE_READY",
                content=result
            )

        self.status = AgentStatus.IDLE
        return None

    def _design_architecture(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Design system architecture based on requirements."""
        task_type = requirements.get("task_type", "simple")
        modules = self._determine_modules(task_type)
        tech_stack = self._select_tech_stack(task_type)

        architecture = {
            "modules": modules,
            "technology_stack": tech_stack,
            "task_type": task_type
        }

        self.architecture_template = architecture
        return architecture

    def _determine_modules(self, task_type: str) -> List[Dict[str, str]]:
        """Determine required modules based on task type."""
        base_modules = [
            {
                "name": "data_processing",
                "responsibility": "Data loading and preprocessing",
                "dependencies": []
            }
        ]

        if task_type in ["medium", "complex"]:
            base_modules.append({
                "name": "optimization_engine",
                "responsibility": "Core optimization algorithms",
                "dependencies": ["data_processing"]
            })

        if task_type == "complex":
            base_modules.append({
                "name": "gui_interface",
                "responsibility": "User interface",
                "dependencies": ["optimization_engine"]
            })

        return base_modules

    def _select_tech_stack(self, task_type: str) -> Dict[str, Any]:
        """Select technology stack based on task type."""
        base_stack = {
            "language": "python",
            "version": "3.10+"
        }

        if task_type in ["medium", "complex"]:
            base_stack["frameworks"] = ["pymoo", "pandas", "numpy"]

        if task_type == "complex":
            base_stack["frameworks"].extend(["streamlit", "plotly"])

        return base_stack
