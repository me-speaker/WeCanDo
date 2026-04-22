"""CEO Agent 到 Claude Code Team 的桥接器"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class TeamConnector:
    """将 CEO 的决策转换为 Claude Code Team 执行的桥梁"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.active_agents: Dict[str, Dict[str, Any]] = {}

    def execute_agent_task(self, agent_name: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """在 Claude Code Team 中执行指定 Agent 的任务"""
        prompt = self._build_agent_prompt(agent_name, task)

        # TODO: 实现与 Claude Code 的实际连接
        # 可能的实现方式:
        # 1. 使用 Claude API 直接调用
        # 2. 使用 Claude Code 的 CLI 接口
        # 3. 使用 MCP 协议连接

        # 目前返回任务信息，实际执行由 Agent.process() 处理
        return {
            "agent": agent_name,
            "task": task,
            "prompt": prompt,
            "status": "ready"
        }

    def _get_agent_system_prompt(self, agent_name: str) -> str:
        """获取各 Agent 的 system prompt"""
        prompts = {
            "requirements_analyst": """You are the Requirements Analyst Agent.
分析用户需求，输出结构化的任务定义。""",

            "architect": """You are the Architect Agent.
设计系统架构和模块划分。""",

            "developer": """You are the Developer Agent.
根据需求和架构实现代码。""",

            "tester": """You are the Tester Agent.
编写和执行测试。""",

            "delivery": """You are the Delivery Agent.
打包和部署项目。"""
        }
        return prompts.get(agent_name, "")

    def _build_agent_prompt(self, agent_name: str, task: Dict[str, Any]) -> str:
        """构建 Agent 的任务 prompt"""
        return f"""
Task: {task.get('description', 'No description')}
Target: {task.get('target_path', 'src/')}
Requirements: {json.dumps(task.get('requirements', {}), indent=2)}

Please execute this task and report results.
"""

    def _call_claude_code(self, system_prompt: str, user_prompt: str, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """实际调用 Claude Code 执行"""
        import anthropic

        client = anthropic.Anthropic()

        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        return {
            "status": "success",
            "response": response.content[0].text,
            "task_context": task_context
        }

    def spawn_team(self, agent_list: List[str]) -> None:
        """在 Claude Code Team 模式下启动多个 Agent"""
        for agent in agent_list:
            self.active_agents[agent] = {
                "status": "idle",
                "task": None,
                "result": None
            }

    def delegate_task(self, agent: str, task: Dict[str, Any]) -> None:
        """向指定 Agent 分发任务"""
        if agent in self.active_agents:
            self.active_agents[agent]["task"] = task
            result = self.execute_agent_task(agent, task)
            self.active_agents[agent]["result"] = result
            self.active_agents[agent]["status"] = "executing"

    def sync_team_status(self) -> Dict[str, str]:
        """同步团队状态"""
        return {
            agent: info["status"]
            for agent, info in self.active_agents.items()
        }

    def get_agent_result(self, agent: str) -> Optional[Dict[str, Any]]:
        """获取指定 Agent 的执行结果"""
        if agent in self.active_agents:
            return self.active_agents[agent].get("result")
        return None

    def mark_agent_complete(self, agent: str) -> None:
        """标记 Agent 任务完成"""
        if agent in self.active_agents:
            self.active_agents[agent]["status"] = "completed"

    def get_active_agents(self) -> List[str]:
        """获取当前活跃的 Agent 列表"""
        return [
            agent for agent, info in self.active_agents.items()
            if info["status"] in ["idle", "executing"]
        ]