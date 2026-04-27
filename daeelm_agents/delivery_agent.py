"""DAE-ELM Delivery Agent.

Packages and deploys the DAE-ELM project.
"""

import sys
from pathlib import Path
_parent = Path(__file__).resolve().parents[3]
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from agents.base import BaseAgent
from agents.delivery import DeliveryAgent as BaseDeliveryAgent
from typing import Dict, Any, List
from .base_agent import DAEELMBaseAgent


class DAEELMDeliveryAgent(DAEELMBaseAgent, BaseDeliveryAgent):
    """Delivery Agent for DAE-ELM project.

    Responsibilities:
    - Package the DAE-ELM project
    - Create Docker configuration
    - Generate deployment files
    """

    def __init__(self):
        BaseAgent.__init__(self, name="delivery", agent_type="delivery")

    def create_package(self, output_dir: str):
        """Create distribution package."""
        import shutil
        project_root = Path(__file__).resolve().parents[1]
        shutil.copytree(project_root / "src", Path(output_dir) / "src")
        shutil.copytree(project_root / "daeelm_agents", Path(output_dir) / "daeelm_agents")
        shutil.copytree(project_root / "bridge", Path(output_dir) / "bridge")
        return {"package_path": output_dir}

    def create_docker(self, output_dir: str):
        """Create Docker configuration."""
        dockerfile = f"""
FROM python:3.9
WORKDIR /app
COPY src/ ./src/
COPY daeelm_agents/ ./daeelm_agents/
COPY bridge/ ./bridge/
RUN pip install torch numpy pandas scikit-learn
CMD ["python", "-m", "src.core.runner"]
"""
        with open(Path(output_dir) / "Dockerfile", "w") as f:
            f.write(dockerfile)
        return {"dockerfile": f"{output_dir}/Dockerfile"}
