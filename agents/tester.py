"""Tester Agent - LLM Native 版本，生成和执行测试"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentMessage, AgentStatus, ToolDefinition


class TesterAgent(BaseAgent):
    """Agent 负责测试和验证实现"""

    def __init__(self, project_root: str = "/home/speaker/origin_ws/ai_company"):
        super().__init__(name="tester", agent_type="validator")
        self.project_root = Path(project_root)
        self.generated_tests: List[str] = []

    def _get_system_prompt(self) -> str:
        return """You are the Tester Agent of AutoEvolve Company.

Your role: Test and validate implementations.

Capabilities:
- Generate unit tests for modules
- Generate integration tests for workflows
- Execute tests and report results
- Identify code quality issues
- Provide suggestions for improvements

Testing principles:
1. Test behavior, not implementation
2. Cover happy path and edge cases
3. Keep tests independent and idempotent
4. Use descriptive test names

Output format for test generation:
{
  "action": "write_test",
  "file_path": "tests/unit/test_module.py",
  "content": "# complete test file content...",
  "test_cases": ["list of test case names"]
}"""

    def _get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="write_test",
                description="Write test file with unit or integration tests",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                        "test_type": {"type": "string", "enum": ["unit", "integration"]}
                    },
                    "required": ["file_path", "content"]
                }
            ),
            ToolDefinition(
                name="run_tests",
                description="Execute tests and return results",
                input_schema={
                    "type": "object",
                    "properties": {
                        "test_path": {"type": "string"}
                    },
                    "required": ["test_path"]
                }
            ),
            ToolDefinition(
                name="generate_test_report",
                description="Generate a test report from results",
                input_schema={
                    "type": "object",
                    "properties": {
                        "results": {"type": "object"}
                    },
                    "required": ["results"]
                }
            )
        ]

    def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理测试任务"""
        self.status = AgentStatus.ACTIVE

        task = message.content
        task_type = task.get("type", "generate_tests")

        if task_type == "generate_tests":
            result = self._generate_tests(task)
        elif task_type == "run_tests":
            result = self._run_tests(task)
        else:
            result = self._generate_tests(task)

        self.status = AgentStatus.COMPLETED
        return self.send_message(
            recipient=message.sender,
            msg_type="test_results",
            content=result
        )

    def _generate_tests(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """生成测试代码"""
        module_name = task.get("module_name", "unknown")
        specs = task.get("specs", {})
        test_type = task.get("test_type", "unit")

        prompt = f"""Generate {test_type} tests for the following module:

Module: {module_name}
Specifications: {json.dumps(specs, indent=2)}

Requirements:
1. Use pytest framework
2. Include docstrings
3. Test both happy path and edge cases
4. Use descriptive test names
5. Follow pytest naming convention (test_*.py)

Return JSON with action 'write_test', file_path, content, and test_cases list."""

        response = self.think(prompt)

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {
                "action": "write_test",
                "file_path": f"tests/{test_type}/test_{module_name}.py",
                "content": response,
                "test_cases": []
            }

        if result.get("action") == "write_test":
            return self._write_test_file(result)

        return {"status": "completed", "tests_generated": self.generated_tests}

    def _run_tests(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """运行测试"""
        test_path = task.get("test_path", "tests/")

        # 实际运行 pytest
        import subprocess
        try:
            result = subprocess.run(
                ["pytest", str(self.project_root / test_path), "-v", "--json-report", "--json-report-file=tests/report.json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            return {
                "status": "completed",
                "passed": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "message": "Tests took too long"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _write_test_file(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """写入测试文件到磁盘"""
        file_path = result.get("file_path", "")
        content = result.get("content", "")

        if not file_path or not content:
            return {"status": "error", "message": "Missing file_path or content"}

        full_path = self.project_root / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, 'w') as f:
            f.write(content)

        self.generated_tests.append(file_path)

        return {
            "status": "written",
            "file_path": file_path,
            "test_cases": result.get("test_cases", []),
            "tests_generated": self.generated_tests
        }