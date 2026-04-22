"""Requirements Analyst Agent - parses requirements and determines task characteristics."""
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentMessage, AgentStatus


class RequirementsAnalystAgent(BaseAgent):
    """Agent responsible for analyzing and parsing requirements."""

    def __init__(self):
        super().__init__(name="requirements_analyst", agent_type="analyzer")
        self.current_requirements: Dict[str, Any] = {}

    def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process requirements and output task characteristics."""
        self.status = AgentStatus.ACTIVE
        self.current_requirements = self._parse_requirements(message.content)

        output = self._create_task_output()

        self.status = AgentStatus.COMPLETED
        return self.send_message(
            recipient=message.sender,
            msg_type="requirements_analyzed",
            content=output
        )

    def _parse_requirements(self, content: Any) -> Dict[str, Any]:
        """Parse requirements from message content."""
        if isinstance(content, dict):
            return {
                "raw_requirements": content.get("requirements", content),
                "source": content.get("source", "user_input")
            }
        return {"raw_requirements": content, "source": "user_input"}

    def _create_task_output(self) -> Dict[str, Any]:
        """Create structured task output with parsed characteristics."""
        complexity_score = self._calculate_complexity()
        task_type = self._determine_task_type()

        return {
            "task_type": task_type,
            "complexity": complexity_score,
            "data": self._extract_data_requirements(),
            "goals": self._extract_goals(),
            "delivery": self._determine_delivery_requirements()
        }

    def _calculate_complexity(self) -> str:
        """Calculate task complexity."""
        req = self.current_requirements.get("raw_requirements", "")
        if isinstance(req, str):
            if len(req) > 500 or any(kw in req.lower() for kw in ["architecture", "distributed", "scalable"]):
                return "complex"
            elif len(req) > 200 or any(kw in req.lower() for kw in ["api", "database", "integration"]):
                return "medium"
        return "simple"

    def _determine_task_type(self) -> str:
        """Determine the type of task."""
        req = str(self.current_requirements.get("raw_requirements", "")).lower()
        if "data" in req or "training" in req or "model" in req:
            return "ml_pipeline"
        elif "api" in req or "endpoint" in req:
            return "api_development"
        elif "interface" in req or "ui" in req or "dashboard" in req:
            return "gui_development"
        return "general"

    def _extract_data_requirements(self) -> List[str]:
        """Extract data requirements from parsed content."""
        return ["dataset", "preprocessing", "feature_engineering"]

    def _extract_goals(self) -> List[str]:
        """Extract goals from requirements."""
        return ["implement_solution", "validate_results", "ensure_quality"]

    def _determine_delivery_requirements(self) -> Dict[str, Any]:
        """Determine delivery requirements."""
        return {
            "format": "code",
            "documentation": True,
            "testing": True
        }