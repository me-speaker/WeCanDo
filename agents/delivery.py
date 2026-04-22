"""Delivery Agent - LLM Native 版本，打包项目和创建文档"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentMessage, AgentStatus, ToolDefinition


class DeliveryAgent(BaseAgent):
    """Agent 负责打包和交付完成的项目"""

    def __init__(self, project_root: str = "/home/speaker/origin_ws/ai_company"):
        super().__init__(name="delivery", agent_type="deployer")
        self.project_root = Path(project_root)
        self.delivery_output: Dict[str, Any] = {}

    def _get_system_prompt(self) -> str:
        return """You are the Delivery Agent of AutoEvolve Company.

Your role: Package and deliver completed projects.

Capabilities:
- Create deployment configuration (Dockerfile, docker-compose)
- Generate setup.py and requirements.txt
- Create README and documentation
- Package project for distribution
- Verify all deliverables are complete

Output format for delivery:
{
  "action": "create_file",
  "file_path": "deployment/file.ext",
  "content": "# file content",
  "description": "what this file is for"
}"""

    def _get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="create_file",
                description="Create a file with specified content",
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
                name="generate_dockerfile",
                description="Generate Dockerfile for the project",
                input_schema={
                    "type": "object",
                    "properties": {
                        "python_version": {"type": "string"},
                        "dependencies": {"type": "array", "items": {"type": "string"}}
                    }
                }
            ),
            ToolDefinition(
                name="generate_readme",
                description="Generate README.md for the project",
                input_schema={
                    "type": "object",
                    "properties": {
                        "project_name": {"type": "string"},
                        "description": {"type": "string"},
                        "features": {"type": "array", "items": {"type": "string"}}
                    }
                }
            )
        ]

    def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理交付请求"""
        self.status = AgentStatus.ACTIVE

        if message.msg_type == "finalize":
            result = self._deliver_project(message.content)
            self.status = AgentStatus.COMPLETED
            return self.send_message(
                recipient=message.sender,
                msg_type="delivery_complete",
                content=result
            )

        self.status = AgentStatus.IDLE
        return None

    def _deliver_project(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """交付项目 - 使用 LLM 生成所有交付物"""
        project_name = project_data.get("project_name", "autoevolve_project")
        task_type = project_data.get("task_type", "general")

        # 使用 LLM 生成完整的交付清单
        prompt = f"""Generate delivery files for the following project:

Project Name: {project_name}
Task Type: {task_type}
Features: {project_data.get('features', [])}

Generate:
1. Dockerfile
2. docker-compose.yml
3. setup.py or pyproject.toml
4. requirements.txt
5. README.md
6. Any other necessary deployment files

Return JSON with action 'create_file' for each file, or a single JSON object with all files."""

        response = self.think(prompt)

        # 尝试解析 LLM 响应
        try:
            files = json.loads(response)
            if isinstance(files, dict) and "action" not in files:
                # LLM 返回了一个包含多个文件的字典
                results = []
                for file_path, content in files.items():
                    if isinstance(content, str):
                        results.append(self._create_file(file_path, content))
                return {"status": "delivered", "files": results}
            elif isinstance(files, list):
                results = []
                for item in files:
                    if item.get("action") == "create_file":
                        results.append(self._create_file(item["file_path"], item["content"]))
                return {"status": "delivered", "files": results}
            else:
                return self._create_file_from_result(files)
        except json.JSONDecodeError:
            # 如果解析失败，使用结构化生成
            return self._generate_structured_delivery(project_name, task_type)

    def _generate_structured_delivery(self, project_name: str, task_type: str) -> Dict[str, Any]:
        """生成结构化交付物"""
        files = []

        # Dockerfile
        dockerfile_content = f'''FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src"]
'''
        files.append(self._create_file("deployment/Dockerfile", dockerfile_content))

        # docker-compose.yml
        compose_content = f'''version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=production
'''
        files.append(self._create_file("deployment/docker-compose.yml", compose_content))

        # requirements.txt
        req_content = '''pandas>=1.5.0
numpy>=1.23.0
pytest>=7.0.0
'''
        files.append(self._create_file("deployment/requirements.txt", req_content))

        # setup.py
        setup_content = f'''from setuptools import setup, find_packages

setup(
    name="{project_name}",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
    ],
    python_requires=">=3.10",
)
'''
        files.append(self._create_file("deployment/setup.py", setup_content))

        # README.md
        readme_content = f'''# {project_name}

Generated by AutoEvolve Company AI Agent Team.

## Overview

This is a {task_type} project.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m src
```

## Docker

```bash
docker-compose up
```
'''
        files.append(self._create_file("README.md", readme_content))

        return {"status": "delivered", "files": files}

    def _create_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """创建文件到磁盘"""
        if not file_path or not content:
            return {"status": "error", "message": "Missing file_path or content"}

        full_path = self.project_root / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, 'w') as f:
            f.write(content)

        return {
            "status": "created",
            "file_path": file_path
        }

    def _create_file_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """从 LLM 结果创建文件"""
        file_path = result.get("file_path", "")
        content = result.get("content", "")

        if file_path and content:
            return self._create_file(file_path, content)

        return {"status": "error", "message": "Invalid result format"}