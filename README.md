# AutoEvolve Company

A self-evolving AI Agent team that transforms user requirements into complete software projects. Powered by Claude LLM with multi-agent orchestration.

## Overview

AutoEvolve Company is an autonomous software development system that leverages specialized AI agents to analyze requirements, design architecture, implement code, test, and deliver complete projects — all orchestrated by a CEO agent.

## Architecture

```
User Request → CEO Agent → Requirements Analyst → [Architect] → Developer → Tester → Delivery
```

### Agents

| Agent | Purpose |
|-------|---------|
| **CEO** | Strategic orchestrator - analyzes complexity, activates agents, coordinates workflow |
| **Requirements Analyst** | Parses requirements, determines task type and skills needed |
| **Architect** | Designs system architecture (complex tasks only) |
| **Developer** | Implements code based on requirements |
| **Tester** | Generates and executes tests, validates solutions |
| **Delivery** | Packages project, creates docs, generates deployment files |

### Activation by Complexity

| Complexity | Activated Agents | Token Budget |
|-----------|------------------|--------------|
| Simple | CEO, Requirements Analyst, Developer | 1000 |
| Medium | CEO, Requirements Analyst, Developer, Tester, Delivery | 2000 |
| Complex | CEO, Requirements Analyst, Architect, Developer, Tester, Delivery | 3000 |

## Project Structure

```
ai_company/
├── agents/          # AI agent implementations
│   ├── base.py      # BaseAgent - LLM-native foundation
│   ├── ceo.py       # CEO orchestrator
│   ├── requirements_analyst.py
│   ├── architect.py
│   ├── developer.py
│   ├── tester.py
│   └── delivery.py
├── bridge/          # System integration
│   ├── team_connector.py   # CEO to Claude Code bridge
│   ├── task_executor.py     # Task execution
│   ├── task_mapper.py      # Task type mapping
│   └── project_scaffold.py # Project scaffolding
├── skill_hub/       # Reusable skills by category
│   ├── core/        # Core skills (data_loading, model_building, testing)
│   ├── oss/         # Open source skills (fastapi, docker, visualization)
│   └── evolved/     # Learned skills from feedback
├── src/             # Prototype code (can be overwritten)
├── docs/            # Documentation
├── .company/        # System configuration
│   ├── ceo_config.yaml
│   ├── agent_registry.yaml
│   └── memory/      # Persistent memory
└── tests/           # System tests
```

## Key Files

- `agents/ceo.py` - Main entry point, orchestrates all agents
- `bridge/task_mapper.py` - Maps task types to required modules/tests
- `.company/ceo_config.yaml` - CEO activation strategy and evolution settings
- `project_generator.yaml` - Project generation config

## Configuration

### CEO Activation Strategy

```yaml
# .company/ceo_config.yaml
activation_strategy:
  simple:
    - ceo
    - requirements_analyst
    - developer
  medium:
    - ceo
    - requirements_analyst
    - developer
    - tester
    - delivery
  complex:
    - ceo
    - requirements_analyst
    - architect
    - developer
    - tester
    - delivery
```

### Skill Hub

Skills are organized by category with priority levels:
- **core/** - Priority 8-12 (data_loading, model_building, testing)
- **oss/** - Priority 15-20 (fastapi_development, docker_deployment)
- **evolved/** - Self-learned patterns

## Self-Evolution

The system includes a self-evolution mechanism:
1. Collects feedback after each task completion
2. Learns patterns from successful implementations
3. Adjusts skill priorities based on performance
4. Logs all changes for audit trail

## Documentation

- `docs/architecture.md` - System architecture details
- `docs/user_guide/quick_start.md` - Quick start guide
- `docs/api/modules.md` - API reference
- `init.md` - Detailed Chinese architecture specification

## Requirements

- Python 3.x
- Anthropic API access (for Claude LLM)

## Usage

The CEO Agent receives tasks via `AgentMessage` and orchestrates the appropriate agent pipeline based on task complexity.

```python
from agents.ceo import CEOAgent

ceo = CEOAgent()
result = ceo.process(user_request)
```