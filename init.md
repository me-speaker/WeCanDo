# AutoEvolve Company - AI Agent Team 快速项目生成系统

版本: 2.0
日期: 2026-04-22
状态: 草案

---

## 1. 核心定位

**目标**：基于 Claude Code Team 模式，配合最小原型文件夹，实现任意项目系统的快速生成。

**核心思路**：

```
用户请求 → CEO Agent（LLM驱动）→ Claude Code Team 执行
                ↓
        读取 .company/ 配置
        使用 agents/ 中的角色定义
        按照 skill_hub/ 分配任务
        在目标目录生成完整项目
```

---

## 2. 与 Claude Code Team 的集成架构

### 2.1 架构对比

| 维度 | 传统 Claude Code | 我们构建的系统 |
|------|-----------------|---------------|
| Agent 数量 | 1 (Claude) | 多 (6个专业 Agent) |
| 任务分配 | 人工判断 | CEO 自动决策 |
| 代码生成 | 单一 Claude | 多 Agent 协作 |
| 项目结构 | 手动创建 | 自动生成 |

### 2.2 CEO 到 Claude Code Team 的桥接

```
当前架构 (Python 类):
User → CEO.process() → if/else → return

桥接后的架构 (LLM 驱动):
User → CEO.think() → LLM API → 决策 → Claude Code Team 执行
```

### 2.3 桥接组件设计

```
/home/speaker/origin_ws/ai_company/
├── bridge/
│   ├── team_connector.py    # 连接 CEO 和 Claude Code Team
│   ├── task_executor.py     # 执行生成的任务
│   └── project_scaffold.py  # 项目脚手架生成器
```

**team_connector.py**:

```python
"""CEO Agent 到 Claude Code Team 的桥接器"""
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

class TeamConnector:
    """将 CEO 的决策转换为 Claude Code Team 执行的桥梁"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.active_agents: Dict[str, Any] = {}

    def execute_agent_task(self, agent_name: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """在 Claude Code Team 中执行指定 Agent 的任务"""
        prompt = self._build_agent_prompt(agent_name, task)

        # 调用 Claude Code 执行
        result = self._call_claude_code(
            system_prompt=self._get_agent_system_prompt(agent_name),
            user_prompt=prompt,
            task_context=task
        )

        return result

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

    def _call_claude_code(self, system_prompt: str, user_prompt: str, task_context: Dict) -> Dict[str, Any]:
        """实际调用 Claude Code 执行"""
        # TODO: 实现与 Claude Code 的实际连接
        # 可能的实现方式:
        # 1. 使用 Claude API 直接调用
        # 2. 使用 Claude Code 的 CLI 接口
        # 3. 使用 MCP 协议连接
        pass

    def spawn_team(self, agent_list: list) -> None:
        """在 Claude Code Team 模式下启动多个 Agent"""
        for agent in agent_list:
            self.active_agents[agent] = {
                "status": "active",
                "task": None
            }

    def delegate_task(self, agent: str, task: Dict[str, Any]) -> None:
        """向指定 Agent 分发任务"""
        if agent in self.active_agents:
            self.active_agents[agent]["task"] = task
            result = self.execute_agent_task(agent, task)
            self.active_agents[agent]["result"] = result

    def sync_team_status(self) -> Dict[str, Any]:
        """同步团队状态"""
        return {
            agent: info["status"]
            for agent, info in self.active_agents.items()
        }
```

---

## 3. LLM-Native Agent 改造方案

### 3.1 改造目标

所有 Agent 必须具备：
- LLM 推理能力（调用 Claude API）
- 自主决策能力
- 上下文记忆

### 3.2 BaseAgent LLM 改造

**agents/base.py 改造后**:

```python
"""Base Agent - LLM Native 版本"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import anthropic

class AgentStatus(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    THINKING = "thinking"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentMessage:
    sender: str
    recipient: str
    msg_type: str
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: dict

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
```

### 3.3 CEO Agent LLM 改造

**agents/ceo.py 改造后**:

```python
"""CEO Agent - LLM Native 版本"""
import yaml
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentMessage, AgentStatus, ToolDefinition

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
2. Determine task complexity (simple/medium/complex)
3. Decide which agents to activate
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
                description="Analyze task complexity based on requirements",
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
        """处理消息 - LLM 驱动决策"""
        self.status = AgentStatus.ACTIVE

        # 构建 prompt 给 LLM
        prompt = f"""User request: {message.content}

Current team status: {self.active_agents}

What should you do? Respond with a JSON action."""

        # 调用 LLM 做决策
        decision = self.think(prompt)
        action = json.loads(decision)

        # 执行决策
        if action["action"] == "delegate":
            return self._execute_delegate(action)
        elif action["action"] == "analyze":
            return self._execute_analyze(message)
        elif action["action"] == "coordinate":
            return self._execute_coordinate()
        elif action["action"] == "complete":
            return self._execute_complete(message)

        self.status = AgentStatus.IDLE
        return None

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
```

### 3.4 Developer Agent 实现真正代码生成

**agents/developer.py 改造后**:

```python
"""Developer Agent - LLM Native 版本，实现真正的代码生成"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentMessage, AgentStatus, ToolDefinition

class DeveloperAgent(BaseAgent):
    """Developer Agent - 负责实现代码"""

    def __init__(self, project_root: str = "/home/speaker/origin_ws/ai_company"):
        super().__init__(name="developer", agent_type="implementer")
        self.project_root = Path(project_root)
        self.generated_files: List[str] = []

    def _get_system_prompt(self) -> str:
        return """You are the Developer Agent of AutoEvolve Company.

Your role: Implement code based on requirements and architecture.

Capabilities:
- Write Python code (data processing, models, APIs, GUIs)
- Create project structure
- Implement modules and integrations
- Add proper documentation and comments

Output format:
For code generation tasks, output:
{
  "action": "write_file",
  "file_path": "relative/path/to/file.py",
  "content": "# full file content...",
  "reasoning": "why this code solves the problem"
}"""

    def _get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="write_file",
                description="Write complete code to a file",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                        "overwrite": {"type": "boolean"}
                    },
                    "required": ["file_path", "content"]
                }
            ),
            ToolDefinition(
                name="create_directory",
                description="Create a new directory",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"]
                }
            ),
            ToolDefinition(
                name="read_file",
                description="Read existing file content",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"}
                    },
                    "required": ["file_path"]
                }
            )
        ]

    def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理代码生成任务"""
        self.status = AgentStatus.ACTIVE

        task = message.content
        task_type = task.get("type", "general")

        if task_type == "implement":
            result = self._implement_solution(task)
        elif task_type == "build_module":
            result = self._build_module(task)
        elif task_type == "fix_issues":
            result = self._fix_issues(task)
        else:
            result = self._implement_solution(task)

        self.status = AgentStatus.COMPLETED
        return self.send_message(
            recipient=message.sender,
            msg_type="implementation_result",
            content=result
        )

    def _implement_solution(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """基于需求实现完整解决方案"""
        requirements = task.get("requirements", {})
        target_dir = task.get("target_dir", "src")

        # 调用 LLM 生成代码
        prompt = f"""Generate a complete solution based on:

Requirements: {json.dumps(requirements, indent=2)}

Target directory: {target_dir}

Generate:
1. Main module code
2. Necessary supporting files
3. Configuration

Return JSON with file_path and content for each file."""

        response = self.think(prompt)
        result = json.loads(response)

        # 执行文件写入
        if result.get("action") == "write_file":
            return self._write_files(result)

        return {"status": "completed", "files_generated": []}

    def _build_module(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """构建特定模块"""
        module_type = task.get("module_type", "data")
        specs = task.get("specs", {})

        prompt = f"""Create a {module_type} module with specs:

{specs}

Generate complete, production-ready Python code.
Return JSON with file_path and content."""

        response = self.think(prompt)
        result = json.loads(response)

        if result.get("action") == "write_file":
            return self._write_files(result)

        return {"status": "completed"}

    def _fix_issues(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """修复问题"""
        issues = task.get("issues", [])
        target_file = task.get("target_file", "")

        prompt = f"""Fix the following issues in {target_file}:

{json.dumps(issues, indent=2)}

Return JSON with action 'write_file', file_path (same as input), and fixed content."""

        response = self.think(prompt)
        result = json.loads(response)

        if result.get("action") == "write_file":
            return self._write_files(result)

        return {"status": "completed"}

    def _write_files(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """写入文件到磁盘"""
        file_path = result.get("file_path", "")
        content = result.get("content", "")

        full_path = self.project_root / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, 'w') as f:
            f.write(content)

        self.generated_files.append(file_path)

        return {
            "status": "written",
            "file_path": file_path,
            "files_generated": self.generated_files
        }
```

---

## 4. 快速项目生成工作流

### 4.1 完整工作流程

```
用户: "我要一个图片分类服务"

        ↓

CEO (LLM): 分析需求
  → 调用 LLM evaluate_complexity() 评估复杂度
  → 复杂度: medium (LLM 判断结果)
  → 激活: [ceo, requirements_analyst, developer, tester, delivery]
  → Token 预算: 2000

        ↓

Requirements Analyst (LLM):
  → task_type: ml_pipeline
  → data: image_dataset, augmentation
  → goals: train_model, serve_api
  → complexity_score: 7 (详细评估)
  → 调用 _assess_complexity_llm() 提供多维度分析

        ↓

Architect (LLM - 仅复杂任务):
  → modules: [data_loader, model, api, gui]
  → tech_stack: [pytorch, fastapi, streamlit]

        ↓

Developer (LLM):
  → 生成 src/data/image_loader.py
  → 生成 src/models/classifier.py
  → 生成 src/api/endpoints.py
  → 生成 main.py

        ↓

Tester (LLM):
  → 生成 tests/unit/test_image_loader.py
  → 生成 tests/unit/test_classifier.py
  → 生成 tests/integration/test_api.py

        ↓

Delivery (LLM):
  → 生成 deployment/Dockerfile
  → 生成 deployment/setup.py
  → 生成 docs/README.md

        ↓

最终项目结构:
autoevolve_project/
├── src/
│   ├── data/
│   ├── models/
│   ├── api/
│   └── main.py
├── tests/
├── deployment/
└── docs/
```

### 4.2 项目生成配置文件

**project_generator.yaml**:

```yaml
generation_config:
  project_name_template: "autoevolve_{task_type}_{timestamp}"

  agents:
    ceo:
      model: claude-opus-4-6
      max_tokens: 4096
      temperature: 0.7

    requirements_analyst:
      model: claude-opus-4-6
      max_tokens: 2048
      temperature: 0.5

    developer:
      model: claude-opus-4-6
      max_tokens: 8192
      temperature: 0.3

    tester:
      model: claude-opus-4-6
      max_tokens: 4096
      temperature: 0.4

    delivery:
      model: claude-opus-4-6
      max_tokens: 2048
      temperature: 0.5

  output:
    base_dir: "/home/speaker/origin_ws/generated_projects"
    overwrite: false
    create_git_repo: true

  quality:
    require_tests: true
    require_docs: true
    min_test_coverage: 70
    lint_check: true
```

### 4.3 任务到 Agent 的映射

```python
# bridge/task_mapper.py

TASK_TYPE_MAPPING = {
    "ml_pipeline": {
        "modules": ["data_loading", "model_building", "training", "inference"],
        "test_focus": ["data_validation", "model_evaluation"],
        "delivery_includes": ["model_export", "inference_api"]
    },
    "web_api": {
        "modules": ["endpoints", "models", "middleware", "auth"],
        "test_focus": ["endpoint_tests", "integration_tests"],
        "delivery_includes": ["docker", "api_docs"]
    },
    "gui_app": {
        "modules": ["ui", "state_management", "widgets"],
        "test_focus": ["ui_tests", "user_flow_tests"],
        "delivery_includes": ["executable", "installer"]
    },
    "data_pipeline": {
        "modules": ["extract", "transform", "load", "monitoring"],
        "test_focus": ["data_quality", "pipeline_tests"],
        "delivery_includes": ["scheduler", "monitoring_dashboard"]
    }
}

COMPLEXITY_ACTIVATION = {
    "simple": ["ceo", "requirements_analyst", "developer"],
    "medium": ["ceo", "requirements_analyst", "developer", "tester", "delivery"],
    "complex": ["ceo", "requirements_analyst", "architect", "developer", "tester", "delivery"]
}
```

### 4.4 LLM 驱动的复杂度评估

#### 4.4.1 为什么需要 LLM 评估

传统规则匹配的问题：

| 问题 | 示例 | 错误判断 |
|------|------|---------|
| 长度不是好指标 | "你好" * 100 | 纯垃圾但判定 complex |
| 关键词匹配太粗糙 | "我要一个 api" | 可能简单但判定 medium |
| 无法理解语义 | "画分布式架构图" | 只是要图片但判定 complex |

LLM 评估的优势：
- 理解语义而非表面特征
- 结合上下文综合判断
- 能识别讽刺、隐喻等复杂表达

#### 4.4.2 复杂度定义标准

```python
COMPLEXITY_DEFINITIONS = {
    "simple": """
定义：单模块、无测试、无文档需求
判断标准：
- 功能单一，逻辑清晰
- 无需外部集成（数据库、API、第三方服务）
- 无复杂算法或优化需求
- 代码量预估 < 200 行
示例：
- "打印 Hello World"
- "实现一个计算器"
- "写一个文件读取函数"
    """,

    "medium": """
定义：2-3个模块、需要测试和文档
判断标准：
- 多个功能模块需要协作
- 需要与外部系统交互（数据库、API）
- 有测试和文档要求
- 代码量预估 200-1000 行
示例：
- "创建一个 REST API"
- "实现用户认证系统"
- "构建数据清洗 pipeline"
    """,

    "complex": """
定义：多模块、系统级、需要架构设计
判断标准：
- 需要系统级架构设计
- 多个子系统需要协调
- 高性能、高可用要求
- 复杂的业务逻辑和边界情况
- 代码量预估 > 1000 行
示例：
- "构建分布式微服务系统"
- "实现多目标优化算法"
- "创建实时数据处理平台"
    """
}
```

#### 4.4.3 LLM 复杂度评估 Prompt

```python
def build_complexity_prompt(requirements: str) -> str:
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
```

#### 4.4.4 评估结果结构

```python
@dataclass
class ComplexityResult:
    level: str  # "simple" | "medium" | "complex"
    reasoning: str  # 简短理由
    dimensions: Dict[str, Any]  # 各维度评分

    # 详细维度
    module_count: int  # 预估模块数
    external_deps: List[str]  # 外部依赖列表
    test_requirements: str  # 测试需求描述
    estimated_lines: int  # 预估代码行数
    risk_factors: List[str]  # 风险因素

# CEO 根据 complexity_result 决定激活策略
def decide_activation(result: ComplexityResult) -> List[str]:
    if result.level == "simple":
        return ["ceo", "requirements_analyst", "developer"]
    elif result.level == "medium":
        return ["ceo", "requirements_analyst", "developer", "tester", "delivery"]
    else:
        return ["ceo", "requirements_analyst", "architect", "developer", "tester", "delivery"]
```

#### 4.4.5 实现位置

| 组件 | 方法 | 说明 |
|------|------|------|
| CEO Agent | `evaluate_complexity()` | 接收需求，调用 LLM，返回复杂度 |
| Requirements Analyst | `_assess_complexity_llm()` | 在需求分析时调用，提供详细评估 |

CEO 的 `analyze_complexity` 工具使用此 prompt，Requirements Analyst 在 `process()` 方法中调用。

---

## 5. 目录结构更新

```
/home/speaker/origin_ws/ai_company/
├── .company/                       # 公司体核心配置
│   ├── ceo_config.yaml
│   ├── agent_registry.yaml
│   ├── memory/
│   └── evolution_log/
│
├── agents/                          # Agent 实现 (LLM-Native)
│   ├── __init__.py
│   ├── base.py                      # BaseAgent (含 LLM 调用)
│   ├── ceo.py                       # CEO Agent (决策 + 编排)
│   ├── requirements_analyst.py       # 需求分析
│   ├── architect.py                 # 架构设计
│   ├── developer.py                 # 代码生成
│   ├── tester.py                    # 测试生成
│   └── delivery.py                  # 交付打包
│
├── bridge/                          # CEO ↔ Claude Code Team 桥接
│   ├── __init__.py
│   ├── team_connector.py           # 连接管理
│   ├── task_executor.py             # 任务执行
│   └── project_scaffold.py          # 脚手架生成
│
├── skill_hub/                       # Skill Hub
│   ├── registry.yaml
│   ├── core/
│   ├── oss/
│   └── evolved/
│
├── src/                             # 原型代码 (会被目标项目覆盖)
│   ├── data/
│   ├── models/
│   ├── optimization/
│   └── gui/
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── deployment/
├── docs/
├── experiments/
│
├── init.md                          # 本文档 - 核心架构定义
├── project_generator.yaml           # 项目生成配置
└── bridge/task_mapper.py            # 任务类型映射
```

---

## 6. 待实现清单

### Phase 1: 基础架构 ✅
- [x] 目录结构创建 (Task #9)
- [x] CEO 配置 (Task #1)
- [x] Agent 注册 (Task #2)
- [x] BaseAgent 基类 (Task #7)

### Phase 2: Agent 实现
- [x] BaseAgent Python 类
- [x] CEO Agent Python 类
- [x] Requirements Analyst Python 类
- [x] Developer Python 类
- [ ] **LLM-Native BaseAgent 改造** (本文档定义)
- [ ] **LLM-Native CEO Agent 改造** (本文档定义)
- [ ] **LLM-Native Developer 改造** (本文档定义)

### Phase 3: 桥接组件
- [ ] **team_connector.py** - CEO 到 Claude Code Team 桥接
- [ ] **task_executor.py** - 任务执行器
- [ ] **project_scaffold.py** - 项目脚手架生成

### Phase 4: 集成测试
- [ ] 端到端项目生成测试
- [ ] 多类型任务测试 (ML pipeline, Web API, GUI)

---

## 7. 关键设计决策

### 7.1 为什么用 Python 类而不是其他方案?

| 方案 | 优点 | 缺点 |
|------|------|------|
| Python 类 | 类型安全, IDE支持, 清晰继承 | 需要额外机制调用 LLM |
| Actor 框架 (Ray/Akka) | 内置并发, 消息驱动 | 学习曲线, 依赖重 |
| 状态机 (LangGraph) | 可视化流程, 内置工具 | 绑定特定框架 |

**选择**: Python 类 + 手动 LLM 调用 = 平衡灵活性和控制力

### 7.2 LLM 调用策略

- **按需调用**: 不是每个方法都调 LLM，核心决策点才调
- **缓存结果**: 相同任务不重复调用
- **渐进式**: 先用规则判断，复杂情况才用 LLM

### 7.3 Claude Code Team 集成方式

两种可选方案:

**方案 A - 直接 API 调用**:
```python
# 使用 Anthropic API 直接调用
client = anthropic.Anthropic()
response = client.messages.create(model="claude-opus-4-6", ...)
```

**方案 B - CLI 桥接**:
```python
# 调用 Claude Code CLI
subprocess.run(["claude", "--team", agent_name, "--task", task])
```

**推荐**: 方案 A (直接 API) - 更可控, 无需启动子进程

---

## 8. 使用示例

### 8.1 启动项目生成

```python
from agents.ceo import CEOAgent
from bridge.team_connector import TeamConnector
from bridge.task_mapper import TASK_TYPE_MAPPING

# 初始化
ceo = CEOAgent()
connector = TeamConnector(project_root="/home/speaker/origin_ws/generated")

# 接收用户请求
user_request = "创建一个图片分类的 ML pipeline，需要 REST API 和可视化界面"

# CEO 处理
message = AgentMessage(
    sender="user",
    recipient="ceo",
    msg_type="user_request",
    content=user_request
)

response = ceo.process(message)

# 触发团队执行
if response:
    connector.spawn_team(["requirements_analyst", "developer", "tester", "delivery"])
    connector.delegate_task("requirements_analyst", response.content)
```

### 8.2 监控进度

```python
# 检查团队状态
status = connector.sync_team_status()
print(f"Active agents: {status}")

# 收集结果
for agent in ["requirements_analyst", "developer", "tester", "delivery"]:
    result = connector.get_agent_result(agent)
    if result:
        print(f"{agent}: {result['status']}")
```

---

## 9. 附录

### 9.1 Agent System Prompt 模板

每个 Agent 的 system prompt 应包含:
1. **角色定义**: 你是谁，你的职责
2. **能力边界**: 你能做什么，不能做什么
3. **协作方式**: 如何和其他 Agent 配合
4. **输出格式**: 你的标准输出格式

### 9.2 消息类型定义

```python
MessageTypes = {
    "user_request": "用户原始请求",
    "task_assignment": "CEO 分配的任务",
    "requirements_analyzed": "需求分析结果",
    "architecture_ready": "架构设计完成",
    "implementation_result": "代码实现结果",
    "test_results": "测试执行结果",
    "finalize": "交付最终项目"
}
```

### 9.3 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0 | 2026-04-22 | 初始架构文档 |
| 2.0 | 2026-04-22 | 添加 LLM-Native 改造方案和桥接组件设计 |

---

文档结束