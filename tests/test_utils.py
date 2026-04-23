"""Test utilities and mocks for agent testing."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from agents.base import BaseAgent, AgentMessage, AgentStatus, ToolDefinition


class MockLLMResponse:
    """Mock LLM response for testing."""
    def __init__(self, text: str):
        self.content = [type('obj', (object,), {'text': text})()]


class MockAgent(BaseAgent):
    """Mock agent for testing that doesn't make LLM calls."""

    def _get_system_prompt(self) -> str:
        return "Mock agent for testing"

    def _get_tools(self):
        return []

    def process(self, message: AgentMessage):
        return None


def mock_think_response(response_text: str):
    """Decorator to mock the think() method to return a specific response."""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Create a mock response
            mock_response = MockLLMResponse(response_text)
            with patch.object(self, 'think', return_value=response_text):
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


def create_mock_agent_with_think(name: str, think_response: str) -> BaseAgent:
    """Create a mock agent that returns a specific response from think()."""
    agent = MockAgent(name=name, agent_type="tester")
    agent.think = Mock(return_value=think_response)
    return agent


def create_mock_message(sender: str, recipient: str, msg_type: str, content: any) -> AgentMessage:
    """Create a mock AgentMessage."""
    return AgentMessage(
        sender=sender,
        recipient=recipient,
        msg_type=msg_type,
        content=content
    )