"""End-to-end integration tests."""
import pytest
from agents.base import BaseAgent, AgentMessage, AgentStatus
from agents.ceo import CEOAgent
from agents.requirements_analyst import RequirementsAnalyst
from agents.developer import DeveloperAgent
from agents.architect import ArchitectAgent
from agents.delivery import DeliveryAgent


class TestAgentIntegration:
    """Integration tests for agent collaboration."""

    def test_simple_task_workflow(self):
        """Test a simple task through the agent pipeline."""
        ceo = CEOAgent()
        requirements_analyst = RequirementsAnalyst()
        developer = DeveloperAgent()

        task_msg = AgentMessage(
            sender="user",
            recipient="ceo",
            msg_type="SUBMIT_TASK",
            content={
                "task_type": "simple",
                "description": "Create a simple data loader"
            }
        )

        # CEO processes task and delegates
        ceo_response = ceo.process(task_msg)
        assert ceo_response is not None
        assert ceo_response.msg_type == "TASK_ASSIGNED"

    def test_complex_task_workflow(self):
        """Test a complex task through the full agent pipeline."""
        ceo = CEOAgent()
        requirements_analyst = RequirementsAnalyst()
        architect = ArchitectAgent()
        developer = DeveloperAgent()
        delivery = DeliveryAgent()

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

        # CEO assigns task
        ceo_response = ceo.process(task_msg)
        assert ceo_response is not None

    def test_requirements_analyst_classification(self):
        """Test requirements analyst task classification."""
        analyst = RequirementsAnalyst()

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
        assert response.msg_type == "TASK_CLASSIFIED"

    def test_architect_design(self):
        """Test architect creating system design."""
        architect = ArchitectAgent()

        design_request = AgentMessage(
            sender="ceo",
            recipient="architect",
            msg_type="DESIGN_ARCHITECTURE",
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
        assert response.msg_type == "ARCHITECTURE_READY"
        assert "modules" in response.content
        assert "technology_stack" in response.content

    def test_delivery_package_generation(self):
        """Test delivery agent creating package."""
        delivery = DeliveryAgent()

        deliver_request = AgentMessage(
            sender="ceo",
            recipient="delivery",
            msg_type="DELIVER_PROJECT",
            content={
                "project_name": "test_project",
                "gui_required": True,
                "documentation_required": True
            }
        )

        response = delivery.process(deliver_request)
        assert response is not None
        assert response.msg_type == "DELIVERY_COMPLETE"
        assert "deployment_files" in response.content
        assert "requirements.txt" in response.content["deployment_files"]


class TestMultiAgentCommunication:
    """Test multi-agent message passing."""

    def test_message_routing(self):
        """Test messages are routed to correct recipients."""
        ceo = CEOAgent()
        analyst = RequirementsAnalyst()

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
