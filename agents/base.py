"""Base Agent - LLM Native 版本"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import anthropic


class AgentStatus(Enum):
    """Agent 状态枚举"""
    IDLE = "idle"
    ACTIVE = "active"
    THINKING = "thinking"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Agent 间通信的消息结构"""
    sender: str
    recipient: str
    msg_type: str
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    input_schema: dict


@dataclass
class ComplexityResult:
    """复杂度评估结果"""
    level: str  # "simple" | "medium" | "complex"
    reasoning: str  # 简短理由
    dimensions: Dict[str, Any] = field(default_factory=dict)
    module_count: int = 0
    external_deps: List[str] = field(default_factory=list)
    test_requirements: str = ""
    estimated_lines: int = 0
    risk_factors: List[str] = field(default_factory=list)
    estimated_tokens: int = 0  # 预估 token 数


class BaseAgent(ABC):
    """LLM-Native 基础 Agent 类"""

    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.agent_type = agent_type
        self.status = AgentStatus.IDLE
        self.context: Dict[str, Any] = {}
        self._message_queue: List[AgentMessage] = []

        # LLM 配置
        self.client = anthropic.Anthropic()
        self.model = "claude-opus-4-6"
        self.max_tokens = 4096

        # 消息历史 (用于 LLM 上下文)
        self.message_history: List[Dict[str, str]] = []

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """子类必须定义自己的 system prompt"""
        pass

    @abstractmethod
    def _get_tools(self) -> List[ToolDefinition]:
        """子类必须定义可用的工具"""
        pass

    def think(self, user_message: str) -> str:
        """核心 LLM 调用方法"""
        self.status = AgentStatus.THINKING

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._get_system_prompt(),
            messages=[{"role": "user", "content": user_message}],
            tools=[self._tool_to_openai(t) for t in self._get_tools()]
        )

        self.status = AgentStatus.ACTIVE
        return response.content[0].text

    def think_with_history(self, user_message: str) -> str:
        """带历史记录的 LLM 调用"""
        messages = self.message_history + [{"role": "user", "content": user_message}]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._get_system_prompt(),
            messages=messages,
            tools=[self._tool_to_openai(t) for t in self._get_tools()]
        )

        # 更新历史
        self.message_history.append({"role": "user", "content": user_message})
        self.message_history.append({"role": "assistant", "content": response.content[0].text})

        return response.content[0].text

    def _tool_to_openai(self, tool: ToolDefinition) -> dict:
        """转换工具定义为 OpenAI 格式"""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema
            }
        }

    @abstractmethod
    def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理消息的抽象方法"""
        pass

    def send_message(self, recipient: str, msg_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> AgentMessage:
        """创建并返回消息"""
        return AgentMessage(
            sender=self.name,
            recipient=recipient,
            msg_type=msg_type,
            content=content,
            metadata=metadata or {}
        )

    def receive_message(self, message: AgentMessage) -> None:
        """接收消息到队列"""
        if self.can_handle(message):
            self._message_queue.append(message)

    def can_handle(self, message: AgentMessage) -> bool:
        """检查是否处理此消息"""
        return message.recipient == self.name

    def get_next_message(self) -> Optional[AgentMessage]:
        """获取下一条消息"""
        if self._message_queue:
            return self._message_queue.pop(0)
        return None

    def clear_queue(self) -> None:
        """清空队列"""
        self._message_queue.clear()

    def _build_complexity_prompt(self, requirements: str) -> str:
        """构建复杂度评估的 prompt"""
        return f"""分析以下需求，判断复杂度等级。

需求内容：
{requirements}

复杂度等级定义：
- simple: 单模块、无测试、无文档需求
- medium: 2-3个模块、需要测试和文档
- complex: 多模块、系统级、需要架构设计

判断维度：
1. 功能模块数量（1个=simple，2-3个=medium，4+=complex）
2. 外部依赖（无=simple，少量=medium，多=complex）
3. 测试需求（无测试=simple，有测试=medium，全面测试=complex）
4. 文档需求（无=simple，基础=medium，完整=complex）
5. 代码量预估（<200=simple，200-1000=medium，>1000=complex）

输出格式：
直接返回: simple / medium / complex
后跟一行简要理由（不超过20字）

示例输出：
medium
涉及API和数据库的简单CRUD应用"""

    def _evaluate_complexity_llm(self, requirements: str) -> ComplexityResult:
        """使用 LLM 评估复杂度"""
        prompt = self._build_complexity_prompt(requirements)
        response = self.think(prompt)

        # 解析 LLM 返回
        lines = response.strip().split('\n')
        level = lines[0].strip().lower() if lines else "simple"
        reasoning = lines[1].strip() if len(lines) > 1 else ""

        # 验证级别
        if level not in ["simple", "medium", "complex"]:
            level = "simple"

        # 根据复杂度级别设置预估 token 数
        token_estimates = {
            "simple": 1000,
            "medium": 2000,
            "complex": 3000
        }
        estimated_tokens = token_estimates.get(level, 1000)

        return ComplexityResult(
            level=level,
            reasoning=reasoning,
            dimensions=self._extract_dimensions(response),
            estimated_tokens=estimated_tokens
        )

    def _extract_dimensions(self, response: str) -> Dict[str, Any]:
        """从 LLM 响应中提取维度信息"""
        result = {}

        # 尝试提取模块数量
        if "module" in response.lower():
            result["module_count"] = self._extract_number(response, r"module[\s_-]*(count|count|number|num)[:\s]*(\d+)")
        if "module_count" not in result:
            result["module_count"] = 0

        # 尝试提取外部依赖
        dep_keywords = ["dependency", "deps", "external", "requires", "libraries"]
        for keyword in dep_keywords:
            if keyword in response.lower():
                result["has_external_deps"] = True
                result["dep_keywords"] = keyword
                break
        if "has_external_deps" not in result:
            result["has_external_deps"] = False

        # 尝试提取测试需求
        test_keywords = ["test", "coverage", "validation"]
        for keyword in test_keywords:
            if keyword in response.lower():
                result["test_required"] = True
                break
        if "test_required" not in result:
            result["test_required"] = False

        # 尝试提取预估行数
        lines_match = self._extract_number(response, r"(?:lines?|code)[:\s]*(\d+)")
        if lines_match:
            result["estimated_lines"] = lines_match
        else:
            result["estimated_lines"] = 0

        # 尝试提取风险因素
        risk_keywords = ["risk", "challenge", "complex", "difficult", "hard"]
        risks = []
        for keyword in risk_keywords:
            if keyword in response.lower():
                risks.append(keyword)
        result["risk_factors"] = risks if risks else []

        # 保留原始响应用于调试
        result["raw_response"] = response

        return result

    def _extract_number(self, text: str, pattern: str) -> Optional[int]:
        """从文本中提取数字"""
        import re
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.group(2):
            try:
                return int(match.group(2))
            except (ValueError, IndexError):
                pass
        return None