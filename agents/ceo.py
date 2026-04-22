"""CEO Agent - LLM Native 版本"""
import yaml
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentMessage, AgentStatus, ToolDefinition, ComplexityResult


class CEOAgent(BaseAgent):
    """CEO Agent - 战略决策者，驱动整个系统"""

    def __init__(self, config_path: str = "/home/speaker/origin_ws/ai_company/.company/ceo_config.yaml"):
        super().__init__(name="ceo", agent_type="orchestrator")
        self.config_path = config_path
        self.config = self._load_config()
        self.active_agents: List[str] = []

    def _load_config(self) -> Dict[str, Any]:
        """加载 CEO 配置"""
        path = Path(self.config_path)
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def _get_system_prompt(self) -> str:
        return """You are the CEO Agent of AutoEvolve Company.

Your responsibilities:
1. Analyze incoming user requests
2. Determine task complexity (simple/medium/complex) using LLM evaluation
3. Decide which agents to activate based on complexity
4. Delegate tasks to appropriate agents
5. Monitor progress and coordinate workflow
6. Ensure quality and timely delivery

You have access to these agents:
- requirements_analyst: Analyzes and parses requirements
- architect: Designs system architecture (complex tasks only)
- developer: Implements code based on requirements
- tester: Tests and validates implementations
- delivery: Packages and deploys completed projects

Activation Strategy:
- Simple task (max 1000 tokens): [ceo, requirements_analyst, developer]
- Medium task (max 2000 tokens): [ceo, requirements_analyst, developer, tester, delivery]
- Complex task (max 3000 tokens): [ceo, requirements_analyst, architect, developer, tester, delivery]

Output format for decisions:
{
  "action": "delegate|analyze|coordinate|complete",
  "target_agent": "agent_name",
  "task": {...},
  "reasoning": "why this decision was made"
}"""

    def _get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="delegate_task",
                description="Delegate a task to a specific agent",
                input_schema={
                    "type": "object",
                    "properties": {
                        "agent": {"type": "string", "enum": ["requirements_analyst", "architect", "developer", "tester", "delivery"]},
                        "task": {"type": "object"},
                        "priority": {"type": "string", "enum": ["high", "medium", "low"]}
                    },
                    "required": ["agent", "task"]
                }
            ),
            ToolDefinition(
                name="analyze_complexity",
                description="Analyze task complexity based on requirements using LLM",
                input_schema={
                    "type": "object",
                    "properties": {
                        "requirements": {"type": "string"},
                        "estimated_size": {"type": "string"}
                    },
                    "required": ["requirements"]
                }
            ),
            ToolDefinition(
                name="check_team_status",
                description="Check the status of all active agents",
                input_schema={"type": "object", "properties": {}}
            ),
            ToolDefinition(
                name="finalize_project",
                description="Finalize and deliver the completed project",
                input_schema={
                    "type": "object",
                    "properties": {
                        "project_path": {"type": "string"},
                        "summary": {"type": "string"}
                    },
                    "required": ["project_path"]
                }
            )
        ]

    def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理消息 - 基于复杂度评估的决策"""
        self.status = AgentStatus.ACTIVE

        # 1. 评估复杂度
        complexity_result = self.evaluate_complexity(message.content)

        # 2. 根据复杂度获取激活的 Agent 列表
        activation_list = self.get_activation_list(complexity_result.level)
        self.active_agents = activation_list

        # 3. 分发给 requirements_analyst 开始工作流
        return self.send_message(
            recipient="requirements_analyst",
            msg_type="task_assignment",
            content={
                "original_request": message.content,
                "complexity": complexity_result.level,
                "activated_agents": activation_list,
                "complexity_details": {
                    "token_estimate": complexity_result.estimated_tokens,
                    "reasoning": complexity_result.reasoning
                }
            }
        )

    def evaluate_complexity(self, requirements: str) -> ComplexityResult:
        """使用 LLM 评估复杂度"""
        return self._evaluate_complexity_llm(requirements)

    def _execute_delegate(self, action: Dict) -> AgentMessage:
        """执行委托任务"""
        agent = action["target_agent"]
        task = action["task"]

        if agent not in self.active_agents:
            self.active_agents.append(agent)

        return self.send_message(
            recipient=agent,
            msg_type="task_assignment",
            content=task
        )

    def _execute_analyze(self, message: AgentMessage) -> AgentMessage:
        """执行分析任务"""
        return self.send_message(
            recipient="requirements_analyst",
            msg_type="analyze_requirements",
            content=message.content
        )

    def _execute_coordinate(self) -> Optional[AgentMessage]:
        """协调团队工作"""
        return None

    def _execute_complete(self, message: AgentMessage) -> AgentMessage:
        """完成任务"""
        self.status = AgentStatus.COMPLETED
        return self.send_message(
            recipient="delivery",
            msg_type="finalize",
            content=message.content
        )

    def get_activation_list(self, complexity: str) -> List[str]:
        """根据复杂度返回激活的 agent 列表"""
        activation_map = {
            "simple": ["ceo", "requirements_analyst", "developer"],
            "medium": ["ceo", "requirements_analyst", "developer", "tester", "delivery"],
            "complex": ["ceo", "requirements_analyst", "architect", "developer", "tester", "delivery"]
        }
        return activation_map.get(complexity, activation_map["simple"])