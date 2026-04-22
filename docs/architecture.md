# AutoEvolve Company Architecture

Version: 1.0
Date: 2026-04-22

## Overview

AutoEvolve Company is a self-evolving AI Agent team that transforms user requirements into complete software projects.

## Architecture Layers

### Strategic Layer
- **CEO Agent**: Orchestrates all tasks, manages agent lifecycle, makes strategic decisions

### Analysis Layer
- **Requirements Analyst**: Parses requirements, determines task complexity, identifies required skills
- **Architect Agent**: Designs system architecture for complex tasks, defines modules and interfaces

### Execution Layer
- **Developer Agent**: Implements code based on requirements and architecture
- **Tester Agent**: Creates and runs tests, validates solutions
- **Delivery Agent**: Generates GUI, packages project, creates documentation

## Agent Activation Strategy

| Complexity | Activated Agents | Token Budget |
|------------|------------------|--------------|
| Simple     | CEO, Requirements Analyst, Developer | 1000 |
| Medium     | CEO, Requirements Analyst, Developer, Tester, Delivery | 2000 |
| Complex    | CEO, Requirements Analyst, Architect, Developer, Tester, Delivery | 3000 |

## Skill Hub Structure

```
skill_hub/
├── core/       # Core skills (priority 8-12)
├── oss/        # Open source skills (priority 15-20)
└── evolved/     # Learned skills from feedback
```

## Directory Structure

```
autoevolve_company/
├── agents/              # Agent implementations
├── skill_hub/           # Skill registry and definitions
├── src/                 # Generated source code
├── tests/               # Test suites
├── docs/                # Documentation
├── deployment/          # Deployment files
└── .company/            # Company configuration
    ├── ceo_config.yaml
    ├── agent_registry.yaml
    └── memory/          # Persistent memory
```

## Communication Protocol

Agents communicate via typed messages:
- `[sender] → [recipient]: [message_type] | [content_summary]`

Example:
```
requirements_analyst → ceo: TASK_CLASSIFIED | type=complex, skills=[data,opt,gui]
```

## Self-Evolution Mechanism

1. **Feedback Collection**: Collects metrics after each task
2. **Pattern Learning**: Identifies successful patterns
3. **Skill Adjustment**: Updates skill priorities based on performance
4. **Evolution Logging**: Records all changes for audit

## Skill Loading

- **Lazy Loading**: Skills are loaded on-demand
- **Caching**: Frequently used skills are cached
- **Priority-based Selection**: Higher priority skills are preferred
