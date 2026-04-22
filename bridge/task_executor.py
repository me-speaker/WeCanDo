"""Task Executor - 执行生成的任务"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base import AgentMessage


class TaskExecutor:
    """执行 Agent 生成的任务"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.execution_history: List[Dict[str, Any]] = []

    def execute_file_write(self, file_path: str, content: str, overwrite: bool = False) -> Dict[str, Any]:
        """执行文件写入任务"""
        full_path = self.project_root / file_path

        # 检查文件是否存在
        if full_path.exists() and not overwrite:
            return {
                "status": "skipped",
                "file_path": file_path,
                "reason": "File already exists and overwrite=False"
            }

        # 创建目录
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入文件
        try:
            with open(full_path, 'w') as f:
                f.write(content)

            self.execution_history.append({
                "action": "write_file",
                "file_path": file_path,
                "status": "success"
            })

            return {
                "status": "success",
                "file_path": file_path,
                "full_path": str(full_path)
            }
        except Exception as e:
            return {
                "status": "error",
                "file_path": file_path,
                "error": str(e)
            }

    def execute_directory_create(self, path: str) -> Dict[str, Any]:
        """执行目录创建任务"""
        full_path = self.project_root / path

        try:
            full_path.mkdir(parents=True, exist_ok=True)

            self.execution_history.append({
                "action": "create_directory",
                "path": path,
                "status": "success"
            })

            return {
                "status": "success",
                "path": path,
                "full_path": str(full_path)
            }
        except Exception as e:
            return {
                "status": "error",
                "path": path,
                "error": str(e)
            }

    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行通用任务"""
        action = task.get("action", "")
        params = task.get("params", {})

        if action == "write_file":
            return self.execute_file_write(
                file_path=params.get("file_path", ""),
                content=params.get("content", ""),
                overwrite=params.get("overwrite", False)
            )
        elif action == "create_directory":
            return self.execute_directory_create(params.get("path", ""))
        else:
            return {
                "status": "unknown_action",
                "action": action
            }

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self.execution_history

    def clear_history(self) -> None:
        """清空执行历史"""
        self.execution_history.clear()

    def execute_agent_process(self, agent: Any, message: AgentMessage) -> Dict[str, Any]:
        """执行 Agent 的 process 方法"""
        try:
            response = agent.process(message)

            result = {
                "status": "success",
                "agent": agent.name,
                "response": response
            }

            if response:
                self.execution_history.append({
                    "action": "agent_process",
                    "agent": agent.name,
                    "msg_type": message.msg_type,
                    "status": "success"
                })

            return result
        except Exception as e:
            return {
                "status": "error",
                "agent": agent.name,
                "error": str(e)
            }