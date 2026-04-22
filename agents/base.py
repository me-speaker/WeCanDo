"""Base agent class for AutoEvolve system."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentStatus(Enum):
    """Agent status enumeration."""
    IDLE = "idle"
    ACTIVE = "active"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    sender: str
    recipient: str
    msg_type: str
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.agent_type = agent_type
        self.status = AgentStatus.IDLE
        self.context: Dict[str, Any] = {}
        self._message_queue: List[AgentMessage] = []

    @abstractmethod
    def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process a message and optionally return a response."""
        pass

    def can_handle(self, message: AgentMessage) -> bool:
        """Check if this agent can handle the given message."""
        return message.recipient == self.name

    def send_message(self, recipient: str, msg_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> AgentMessage:
        """Create and return a message to be sent."""
        return AgentMessage(
            sender=self.name,
            recipient=recipient,
            msg_type=msg_type,
            content=content,
            metadata=metadata or {}
        )

    def receive_message(self, message: AgentMessage) -> None:
        """Add a message to the agent's queue."""
        if self.can_handle(message):
            self._message_queue.append(message)

    def get_next_message(self) -> Optional[AgentMessage]:
        """Get the next message from the queue."""
        if self._message_queue:
            return self._message_queue.pop(0)
        return None

    def clear_queue(self) -> None:
        """Clear all messages from the queue."""
        self._message_queue.clear()