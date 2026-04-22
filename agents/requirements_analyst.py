"""Requirements Analyst Agent - LLM Native 版本"""
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentMessage, AgentStatus, ToolDefinition, ComplexityResult


class RequirementsAnalystAgent(BaseAgent):
    """Agent 负责分析和解析需求"""

    def __init__(self):
        super().__init__(name="requirements_analyst", agent_type="analyzer")
        self.current_requirements: Dict[str, Any] = {}
        self.complexity_result: Optional[ComplexityResult] = None

    def _get_system_prompt(self) -> str:
        return """You are the Requirements Analyst Agent of AutoEvolve Company.

Your role: Analyze and parse user requirements into structured task definitions.

Capabilities:
- Parse natural language requirements
- Identify task type (ml_pipeline, web_api, gui_app, data_pipeline, general)
- Determine complexity with detailed dimensions
- Extract data requirements, goals, and delivery requirements
- Identify external dependencies

Output format for requirements analysis:
{
  "task_type": "ml_pipeline|web_api|gui_app|data_pipeline|general",
  "complexity": "simple|medium|complex",
  "data": {
    "source": "description of data sources",
    "features": ["list of features"],
    "preprocessing": "any preprocessing needs"
  },
  "goals": ["list of goals"],
  "delivery": {
    "format": "code|api|gui|pipeline",
    "documentation": true|false,
    "testing": true|false
  },
  "reasoning": "brief explanation of the analysis"
}"""

    def _get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="parse_requirements",
                description="Parse raw requirements into structured format",
                input_schema={
                    "type": "object",
                    "properties": {
                        "raw_requirements": {"type": "string"}
                    },
                    "required": ["raw_requirements"]
                }
            ),
            ToolDefinition(
                name="assess_complexity",
                description="Assess task complexity with detailed dimensions",
                input_schema={
                    "type": "object",
                    "properties": {
                        "requirements": {"type": "string"}
                    },
                    "required": ["requirements"]
                }
            ),
            ToolDefinition(
                name="identify_dependencies",
                description="Identify external dependencies from requirements",
                input_schema={
                    "type": "object",
                    "properties": {
                        "requirements": {"type": "string"}
                    },
                    "required": ["requirements"]
                }
            )
        ]

    def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理需求分析任务"""
        self.status = AgentStatus.ACTIVE

        # 解析需求
        self.current_requirements = self._parse_requirements(message.content)

        # LLM 评估复杂度
        self.complexity_result = self._assess_complexity_llm(message.content)

        # 创建输出
        output = self._create_task_output()

        self.status = AgentStatus.COMPLETED
        return self.send_message(
            recipient=message.sender,
            msg_type="requirements_analyzed",
            content=output
        )

    def _parse_requirements(self, content: Any) -> Dict[str, Any]:
        """解析需求"""
        if isinstance(content, dict):
            return {
                "raw_requirements": content.get("requirements", content),
                "source": content.get("source", "user_input")
            }
        return {"raw_requirements": content, "source": "user_input"}

    def _assess_complexity_llm(self, requirements: Any) -> ComplexityResult:
        """使用 LLM 评估复杂度（调用基类方法）"""
        req_str = requirements if isinstance(requirements, str) else str(requirements)
        return self._evaluate_complexity_llm(req_str)

    def _create_task_output(self) -> Dict[str, Any]:
        """创建结构化的任务输出"""
        complexity = self.complexity_result.level if self.complexity_result else "simple"

        return {
            "task_type": self._determine_task_type(),
            "complexity": complexity,
            "complexity_reasoning": self.complexity_result.reasoning if self.complexity_result else "",
            "complexity_dimensions": self.complexity_result.dimensions if self.complexity_result else {},
            "data": self._extract_data_requirements(),
            "goals": self._extract_goals(),
            "delivery": self._determine_delivery_requirements()
        }

    def _determine_task_type(self) -> str:
        """使用 LLM 确定任务类型"""
        req = str(self.current_requirements.get("raw_requirements", "")).lower()

        prompt = f"""分析以下需求，确定任务类型：

需求: {req}

任务类型选项:
- ml_pipeline: 机器学习 pipeline（数据处理、模型训练、推理）
- web_api: Web API 开发（REST、GraphQL）
- gui_app: GUI 应用（桌面、Web 可视化界面）
- data_pipeline: 数据 pipeline（ETL、数据流处理）
- general: 一般任务

直接返回任务类型名称。"""

        response = self.think(prompt).strip().lower()

        valid_types = ["ml_pipeline", "web_api", "gui_app", "data_pipeline", "general"]
        if response not in valid_types:
            return "general"
        return response

    def _extract_data_requirements(self) -> Dict[str, Any]:
        """提取数据需求"""
        return {
            "source": "待确定",
            "features": [],
            "preprocessing": "待确定"
        }

    def _extract_goals(self) -> List[str]:
        """提取目标"""
        return ["implement_solution", "validate_results", "ensure_quality"]

    def _determine_delivery_requirements(self) -> Dict[str, Any]:
        """确定交付需求"""
        complexity = self.complexity_result.level if self.complexity_result else "simple"

        return {
            "format": "code",
            "documentation": complexity != "simple",
            "testing": complexity != "simple"
        }