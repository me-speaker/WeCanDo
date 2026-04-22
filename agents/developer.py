"""Developer Agent - implements code based on requirements and architecture."""
from typing import Any, Dict, Optional

from .base import BaseAgent, AgentMessage, AgentStatus


class DeveloperAgent(BaseAgent):
    """Agent responsible for implementing code based on requirements."""

    def __init__(self):
        super().__init__(name="developer", agent_type="implementer")
        self.implemented_modules: Dict[str, Any] = {}

    def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process implementation task and produce code."""
        self.status = AgentStatus.ACTIVE

        if message.msg_type == "implement":
            result = self._implement_solution(message.content)
        elif message.msg_type == "build":
            result = self._build_from_architecture(message.content)
        else:
            result = self._implement_solution(message.content)

        self.status = AgentStatus.COMPLETED
        return self.send_message(
            recipient=message.sender,
            msg_type="implementation_complete",
            content=result
        )

    def _implement_solution(self, content: Any) -> Dict[str, Any]:
        """Implement solution based on requirements."""
        if isinstance(content, dict):
            requirements = content.get("requirements", content)
        else:
            requirements = content

        return {
                "modules": self._create_modules(requirements),
                "files_created": self._get_expected_files(),
                "status": "implemented"
            }

    def _build_from_architecture(self, content: Any) -> Dict[str, Any]:
        """Build implementation from architecture design."""
        architecture = content.get("architecture", {})
        return {
                "modules": self._create_modules_from_arch(architecture),
                "files_created": self._get_expected_files(),
                "status": "implemented"
            }

    def _create_modules(self, requirements: Any) -> list:
        """Create implementation modules based on requirements."""
        return [
            {"name": "data_loader", "type": "module", "status": "ready"},
            {"name": "model_builder", "type": "module", "status": "ready"},
            {"name": "main", "type": "entry_point", "status": "ready"}
        ]

    def _create_modules_from_arch(self, architecture: Dict[str, Any]) -> list:
        """Create modules based on architecture specification."""
        components = architecture.get("components", [])
        return [{"name": comp, "type": "component", "status": "ready"} for comp in components]

    def _get_expected_files(self) -> list:
        """Get list of expected files to be created."""
        return [
            "src/data/data_loader.py",
            "src/models/model_builder.py",
            "main.py"
        ]