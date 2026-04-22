# Quick Start Guide

## Prerequisites

- Python 3.10+
- Required packages: See `deployment/requirements.txt`

## Installation

```bash
pip install -r deployment/requirements.txt
```

## Basic Usage

1. **Define Your Task**

Create a task specification in YAML format:

```yaml
task_type: multi_objective_optimization
complexity: complex
requirements:
  data:
    source: data/reaction_data.csv
    features: [temperature, pressure, catalyst_type]
  goals:
    maximize: yield
    minimize: energy_consumption
  delivery:
    gui_required: true
    documentation_required: true
```

2. **Submit to AutoEvolve**

```python
from agents.ceo import CEOAgent

ceo = CEOAgent()
result = ceo.process_task(task_spec)
```

3. **Review Results**

Generated files will be placed in:
- `src/` - Source code
- `tests/` - Test suites
- `docs/` - Documentation
- `deployment/` - Deployment files

## Project Structure After Generation

```
generated_project/
├── src/
│   ├── data/
│   ├── models/
│   └── gui/
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
└── deployment/
```

## Configuration

Edit `.company/ceo_config.yaml` to customize:
- Agent activation thresholds
- Token budgets
- Evolution settings
