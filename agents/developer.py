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

When generating code, always:
1. Follow PEP 8 style guidelines
2. Add docstrings to all functions and classes
3. Include type hints
4. Handle errors gracefully
5. Add necessary imports

Output format for code generation tasks:
{
  "action": "write_file",
  "file_path": "relative/path/to/file.py",
  "content": "# complete file content...",
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

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            # 如果 LLM 返回的不是 JSON，尝试解析
            result = {"action": "write_file", "file_path": f"{target_dir}/main.py", "content": response}

        # 执行文件写入
        if result.get("action") == "write_file":
            return self._write_files(result)

        return {"status": "completed", "files_generated": self.generated_files}

    def _build_module(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """构建特定模块"""
        module_type = task.get("module_type", "data")
        specs = task.get("specs", {})

        prompt = f"""Create a {module_type} module with specs:

{json.dumps(specs, indent=2)}

Generate complete, production-ready Python code.
Return JSON with file_path and content."""

        response = self.think(prompt)

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {"action": "write_file", "file_path": f"src/{module_type}.py", "content": response}

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

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {"action": "write_file", "file_path": target_file, "content": response}

        if result.get("action") == "write_file":
            return self._write_files(result)

        return {"status": "completed"}

    def _write_files(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """写入文件到磁盘"""
        file_path = result.get("file_path", "")
        content = result.get("content", "")

        if not file_path or not content:
            return {"status": "error", "message": "Missing file_path or content"}

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