"""Delivery Agent - packages project and creates documentation."""
from typing import Any, Dict, List, Optional
from agents.base import BaseAgent, AgentMessage, AgentStatus


class DeliveryAgent(BaseAgent):
    """Agent responsible for packaging and delivering completed projects."""

    def __init__(self):
        super().__init__(name="delivery", agent_type="deployer")
        self.delivery_output: Dict[str, Any] = {}

    def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process delivery request."""
        self.status = AgentStatus.ACTIVE

        if message.msg_type == "DELIVER_PROJECT":
            result = self._deliver_project(message.content)
            self.status = AgentStatus.COMPLETED
            return self.send_message(
                recipient=message.sender,
                msg_type="DELIVERY_COMPLETE",
                content=result
            )

        self.status = AgentStatus.IDLE
        return None

    def _deliver_project(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deliver the completed project with all outputs."""
        gui_required = project_data.get("gui_required", False)
        documentation_required = project_data.get("documentation_required", True)

        outputs = {
            "deployment_files": self._create_deployment_files(project_data),
            "documentation": [],
            "gui": None
        }

        if gui_required:
            outputs["gui"] = self._create_gui_components(project_data)

        if documentation_required:
            outputs["documentation"] = self._create_documentation(project_data)

        self.delivery_output = outputs
        return outputs

    def _create_deployment_files(self, project_data: Dict[str, Any]) -> Dict[str, str]:
        """Create deployment configuration files."""
        return {
            "setup.py": self._generate_setup_py(project_data),
            "requirements.txt": self._generate_requirements_txt(project_data),
            "Dockerfile": self._generate_dockerfile(project_data)
        }

    def _generate_setup_py(self, project_data: Dict[str, Any]) -> str:
        """Generate setup.py file."""
        project_name = project_data.get("project_name", "autoevolve_project")
        return f'''from setuptools import setup, find_packages

setup(
    name="{project_name}",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "pymoo>=0.6.0",
    ],
    python_requires=">=3.10",
)
'''

    def _generate_requirements_txt(self, project_data: Dict[str, Any]) -> str:
        """Generate requirements.txt file."""
        return '''pandas>=1.5.0
numpy>=1.23.0
pymoo>=0.6.0
plotly>=5.0.0
streamlit>=1.0.0
pytest>=7.0.0
'''

    def _generate_dockerfile(self, project_data: Dict[str, Any]) -> str:
        """Generate Dockerfile."""
        return '''FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "src"]
'''

    def _create_gui_components(self, project_data: Dict[str, Any]) -> Dict[str, str]:
        """Create GUI components if required."""
        return {
            "app.py": self._generate_streamlit_app(project_data)
        }

    def _generate_streamlit_app(self, project_data: Dict[str, Any]) -> str:
        """Generate Streamlit app file."""
        return '''import streamlit as st
import pandas as pd

st.title("AutoEvolve Project")

st.markdown("Welcome to the generated application.")

# Add your custom components here
'''

    def _create_documentation(self, project_data: Dict[str, Any]) -> List[str]:
        """Create documentation files."""
        return ["README.md", "INSTALL.md", "USAGE.md"]
