# API Modules

## Agents Module (`agents/`)

### BaseAgent (`agents/base.py`)

Base class for all agents.

```python
class BaseAgent(ABC):
    def __init__(self, name: str, agent_type: str)
    def process(self, message: AgentMessage) -> Optional[AgentMessage]
    def send_message(self, recipient: str, msg_type: str, content: Any) -> AgentMessage
    def receive_message(self, message: AgentMessage) -> None
```

### CEO Agent (`agents/ceo.py`)

Orchestrates all agent activities.

### Requirements Analyst (`agents/requirements_analyst.py`)

Analyzes user requirements and determines task complexity.

### Developer Agent (`agents/developer.py`)

Implements code based on requirements.

## Skill Hub Module (`skill_hub/`)

### Registry (`skill_hub/registry.yaml`)

Central registry of all available skills.

## Data Classes

### AgentMessage (`agents/base.py`)

```python
@dataclass
class AgentMessage:
    sender: str
    recipient: str
    msg_type: str
    content: Any
    timestamp: datetime
    metadata: Dict[str, Any]
```

### AgentStatus (`agents/base.py`)

```python
class AgentStatus(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"
```
