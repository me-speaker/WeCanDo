# DAE-ELM Project - Agent System Configuration

This project follows the AutoEvolve Company Agent Mode architecture.

## Project Context

DAE-ELM (Denoising Autoencoder Extreme Learning Machine) is a surrogate model optimization system for chemical formula optimization.

## Agent Team

When running in company mode, this project uses:
- **CEO Agent**: Coordinates all agents
- **Requirements Analyst**: Parses user requirements into formal specifications
- **Architect**: Designs system architecture
- **Developer**: Implements code in `src/`
- **Tester**: Validates functionality
- **Delivery**: Packages and deploys

## Key Paths

- `src/`: Core business code (DAE, ELM, optimization)
- `daeelm_agents/`: DAE-ELM specialized agents
- `bridge/`: Connects agents to `src/` modules
- `skill_hub/`: Reusable agent skills
- `deployment/`: Docker and deployment configs

## Commands

- Run tests: `pytest tests/`
- Build Docker: `docker-compose -f deployment/docker-compose.yaml build`
