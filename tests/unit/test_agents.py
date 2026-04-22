"""Unit tests for agent classes."""
import pytest
from datetime import datetime
from agents.base import BaseAgent, AgentMessage, AgentStatus


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def process(self, message: AgentMessage):
        return None


class TestAgentMessage:
    """Tests for AgentMessage dataclass."""

    def test_agent_message_creation(self):
        """Test creating an AgentMessage."""
        msg = AgentMessage(
            sender="test_sender",
            recipient="test_recipient",
            msg_type="TEST",
            content={"key": "value"}
        )
        assert msg.sender == "test_sender"
        assert msg.recipient == "test_recipient"
        assert msg.msg_type == "TEST"
        assert msg.content == {"key": "value"}
        assert isinstance(msg.timestamp, datetime)

    def test_agent_message_with_metadata(self):
        """Test AgentMessage with metadata."""
        metadata = {"priority": "high", "task_id": "123"}
        msg = AgentMessage(
            sender="sender",
            recipient="recipient",
            msg_type="TYPE",
            content="data",
            metadata=metadata
        )
        assert msg.metadata == metadata


class TestBaseAgent:
    """Tests for BaseAgent class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = MockAgent(name="test_agent", agent_type="tester")
        assert agent.name == "test_agent"
        assert agent.agent_type == "tester"
        assert agent.status == AgentStatus.IDLE
        assert agent.context == {}

    def test_send_message(self):
        """Test sending a message."""
        agent = MockAgent(name="sender", agent_type="tester")
        msg = agent.send_message(
            recipient="receiver",
            msg_type="TEST_MSG",
            content={"data": "test"}
        )
        assert msg.sender == "sender"
        assert msg.recipient == "receiver"
        assert msg.msg_type == "TEST_MSG"
        assert msg.content == {"data": "test"}

    def test_receive_message(self):
        """Test receiving a message."""
        agent = MockAgent(name="receiver", agent_type="tester")
        msg = AgentMessage(
            sender="sender",
            recipient="receiver",
            msg_type="TEST",
            content="test"
        )
        agent.receive_message(msg)
        assert len(agent._message_queue) == 1

    def test_receive_message_wrong_recipient(self):
        """Test that messages are not queued for wrong recipient."""
        agent = MockAgent(name="agent1", agent_type="tester")
        msg = AgentMessage(
            sender="sender",
            recipient="agent2",
            msg_type="TEST",
            content="test"
        )
        agent.receive_message(msg)
        assert len(agent._message_queue) == 0

    def test_get_next_message(self):
        """Test getting next message from queue."""
        agent = MockAgent(name="test", agent_type="tester")
        msg1 = agent.send_message("recipient", "TYPE1", "content1")
        msg2 = agent.send_message("recipient", "TYPE2", "content2")

        agent.receive_message(msg1)
        agent.receive_message(msg2)

        first = agent.get_next_message()
        assert first.msg_type == "TYPE1"

        second = agent.get_next_message()
        assert second.msg_type == "TYPE2"

        assert agent.get_next_message() is None

    def test_clear_queue(self):
        """Test clearing message queue."""
        agent = MockAgent(name="test", agent_type="tester")
        agent.receive_message(agent.send_message("r", "T", "c"))
        agent.receive_message(agent.send_message("r", "T", "c"))
        assert len(agent._message_queue) == 2

        agent.clear_queue()
        assert len(agent._message_queue) == 0

    def test_can_handle(self):
        """Test message handling check."""
        agent = MockAgent(name="my_agent", agent_type="tester")
        msg = AgentMessage(
            sender="sender",
            recipient="my_agent",
            msg_type="TEST",
            content="test"
        )
        assert agent.can_handle(msg) is True

        msg_wrong = AgentMessage(
            sender="sender",
            recipient="other_agent",
            msg_type="TEST",
            content="test"
        )
        assert agent.can_handle(msg_wrong) is False


class TestAgentStatus:
    """Tests for AgentStatus enum."""

    def test_agent_status_values(self):
        """Test AgentStatus enum values."""
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.ACTIVE.value == "active"
        assert AgentStatus.WAITING.value == "waiting"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.ERROR.value == "error"
