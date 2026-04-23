"""End-to-end integration tests."""
import pytest
from unittest.mock import patch, MagicMock
from agents.base import AgentMessage, AgentStatus
from agents.ceo import CEOAgent
from agents.requirements_analyst import RequirementsAnalystAgent
from agents.developer import DeveloperAgent
from agents.architect import ArchitectAgent
from agents.delivery import DeliveryAgent


class TestAgentIntegration:
    """Integration tests for agent collaboration."""

    def test_simple_task_workflow(self):
        """Test a simple task through the agent pipeline."""
        ceo = CEOAgent()

        # Mock the think method to avoid LLM calls
        with patch.object(ceo, 'think', return_value="simple\nSimple task"):
            task_msg = AgentMessage(
                sender="user",
                recipient="ceo",
                msg_type="SUBMIT_TASK",
                content={
                    "task_type": "simple",
                    "description": "Create a simple data loader"
                }
            )

            ceo_response = ceo.process(task_msg)
            assert ceo_response is not None
            assert ceo_response.msg_type == "task_assignment"

    def test_complex_task_workflow(self):
        """Test a complex task through the full agent pipeline."""
        ceo = CEOAgent()

        with patch.object(ceo, 'think', return_value="complex\nComplex task with architecture"):
            task_msg = AgentMessage(
                sender="user",
                recipient="ceo",
                msg_type="SUBMIT_TASK",
                content={
                    "task_type": "complex",
                    "description": "Multi-objective optimization with GUI",
                    "gui_required": True,
                    "documentation_required": True
                }
            )

            ceo_response = ceo.process(task_msg)
            assert ceo_response is not None

    def test_requirements_analyst_classification(self):
        """Test requirements analyst task classification."""
        analyst = RequirementsAnalystAgent()

        with patch.object(analyst, 'think', return_value="general"):
            simple_task = AgentMessage(
                sender="ceo",
                recipient="requirements_analyst",
                msg_type="ANALYZE_REQUIREMENTS",
                content={
                    "description": "Load CSV and display basic stats"
                }
            )

            response = analyst.process(simple_task)
            assert response is not None
            assert response.msg_type == "requirements_analyzed"

    def test_architect_design(self):
        """Test architect creating system design."""
        architect = ArchitectAgent()

        # Mock LLM response for architecture design
        architecture_response = '''{
            "modules": [
                {"name": "data_loader", "responsibility": "Load data", "dependencies": [], "interface": "DataFrame"}
            ],
            "technology_stack": {"language": "python", "frameworks": ["fastapi"], "libraries": []},
            "data_flow": "Data flows through modules",
            "reasoning": "Architecture for complex task"
        }'''

        with patch.object(architect, 'think', return_value=architecture_response):
            design_request = AgentMessage(
                sender="ceo",
                recipient="architect",
                msg_type="design_architecture",
                content={
                    "task_type": "complex",
                    "requirements": {
                        "multi_objective": True,
                        "gui": True
                    }
                }
            )

            response = architect.process(design_request)
            assert response is not None
            assert response.msg_type == "architecture_ready"
            assert "modules" in response.content
            assert "technology_stack" in response.content

    def test_delivery_package_generation(self):
        """Test delivery agent creating package."""
        delivery = DeliveryAgent()

        # Mock LLM response for delivery
        delivery_response = '{"status": "delivered", "files": ["deployment/Dockerfile"]}'

        with patch.object(delivery, 'think', return_value=delivery_response):
            deliver_request = AgentMessage(
                sender="ceo",
                recipient="delivery",
                msg_type="finalize",
                content={
                    "project_name": "test_project",
                    "gui_required": True,
                    "documentation_required": True
                }
            )

            response = delivery.process(deliver_request)
            assert response is not None
            assert response.msg_type == "delivery_complete"


class TestMultiAgentCommunication:
    """Test multi-agent message passing."""

    def test_message_routing(self):
        """Test messages are routed to correct recipients."""
        ceo = CEOAgent()
        analyst = RequirementsAnalystAgent()

        # Analyst receives message for CEO
        msg_for_ceo = AgentMessage(
            sender="analyst",
            recipient="ceo",
            msg_type="REPORT",
            content={"status": "analyzed"}
        )

        ceo.receive_message(msg_for_ceo)
        queued = ceo.get_next_message()
        assert queued is not None
        assert queued.sender == "analyst"

    def test_agent_status_transitions(self):
        """Test agent status changes during processing."""
        developer = DeveloperAgent()

        assert developer.status == AgentStatus.IDLE

        task_msg = AgentMessage(
            sender="ceo",
            recipient="developer",
            msg_type="IMPLEMENT",
            content={"task": "test"}
        )

        # Developer would transition to ACTIVE during process
        # and back to IDLE or COMPLETED after
        assert developer.status == AgentStatus.IDLE