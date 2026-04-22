"""Architect Agent - LLM Native 版本，设计复杂任务的系统架构"""
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentMessage, AgentStatus, ToolDefinition


class ArchitectAgent(BaseAgent):
    """Agent 负责设计系统架构 - 仅复杂任务激活"""

    def __init__(self):
        super().__init__(name="architect", agent_type="designer")
        self.architecture_template: Dict[str, Any] = {}

    def _get_system_prompt(self) -> str:
        return """You are the Architect Agent of AutoEvolve Company.

Your role: Design system architecture for complex tasks.

Capabilities:
- Analyze requirements and determine system structure
- Design modules and their dependencies
- Select appropriate technology stack
- Define interfaces between modules
- Ensure scalability and maintainability

Output format for architecture design:
{
  "modules": [
    {
      "name": "module_name",
      "responsibility": "what this module does",
      "dependencies": ["list of dependent modules"],
      "interface": "API or interface description"
    }
  ],
  "technology_stack": {
    "language": "python",
    "frameworks": ["list of frameworks"],
    "libraries": ["list of libraries"]
  },
  "data_flow": "description of how data flows through the system",
  "reasoning": "why this architecture was chosen"
}"""

    def _get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="design_module",
                description="Design a specific module with its interface",
                input_schema={
                    "type": "object",
                    "properties": {
                        "module_name": {"type": "string"},
                        "responsibility": {"type": "string"},
                        "dependencies": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["module_name", "responsibility"]
                }
            ),
            ToolDefinition(
                name="select_stack",
                description="Select technology stack for the project",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task_type": {"type": "string"},
                        "requirements": {"type": "object"}
                    },
                    "required": ["task_type"]
                }
            )
        ]

    def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理架构设计请求"""
        self.status = AgentStatus.ACTIVE

        if message.msg_type == "design_architecture":
            result = self._design_architecture(message.content)
            self.status = AgentStatus.COMPLETED
            return self.send_message(
                recipient=message.sender,
                msg_type="architecture_ready",
                content=result
            )

        self.status = AgentStatus.IDLE
        return None

    def _design_architecture(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """设计系统架构 - 使用 LLM"""
        task_type = requirements.get("task_type", "general")
        goals = requirements.get("goals", [])

        # 使用 LLM 生成架构
        prompt = f"""Design a system architecture for the following requirements:

Task Type: {task_type}
Goals: {goals}

Consider:
1. Module decomposition - break down into cohesive modules
2. Dependencies between modules
3. Technology stack selection
4. Data flow between components

Return a complete architecture specification."""

        response = self.think(prompt)

        # 解析 LLM 响应
        try:
            import json
            architecture = json.loads(response)
        except json.JSONDecodeError:
            # Fallback to structured design
            architecture = self._generate_structured_architecture(task_type, goals)

        self.architecture_template = architecture
        return architecture

    def _generate_structured_architecture(self, task_type: str, goals: List[str]) -> Dict[str, Any]:
        """生成结构化架构（当 LLM 返回不是 JSON 时使用）"""
        modules = []
        tech_stack = {"language": "python", "frameworks": [], "libraries": []}

        # 根据任务类型添加模块
        if task_type == "ml_pipeline":
            modules = [
                {"name": "data_loader", "responsibility": "Load and preprocess data", "dependencies": [], "interface": "DataFrame"},
                {"name": "feature_engineering", "responsibility": "Feature extraction and transformation", "dependencies": ["data_loader"], "interface": "DataFrame"},
                {"name": "model_training", "responsibility": "Train ML models", "dependencies": ["feature_engineering"], "interface": "Model"},
                {"name": "inference", "responsibility": "Run predictions", "dependencies": ["model_training"], "interface": "Predictions"}
            ]
            tech_stack["frameworks"] = ["scikit-learn", "pytorch", "pandas"]
        elif task_type == "web_api":
            modules = [
                {"name": "endpoints", "responsibility": "API endpoint handlers", "dependencies": [], "interface": "HTTP"},
                {"name": "models", "responsibility": "Data models and validation", "dependencies": [], "interface": "Pydantic"},
                {"name": "services", "responsibility": "Business logic", "dependencies": ["models"], "interface": "Service"},
                {"name": "database", "responsibility": "Data persistence", "dependencies": ["models"], "interface": "ORM"}
            ]
            tech_stack["frameworks"] = ["fastapi", "sqlalchemy", "pydantic"]
        else:
            modules = [
                {"name": "main", "responsibility": "Entry point", "dependencies": [], "interface": "CLI"},
                {"name": "core", "responsibility": "Core business logic", "dependencies": ["main"], "interface": "API"}
            ]
            tech_stack["frameworks"] = ["click"]

        return {
            "modules": modules,
            "technology_stack": tech_stack,
            "data_flow": "Data flows through modules according to dependencies",
            "reasoning": f"Architecture designed for {task_type} task type"
        }