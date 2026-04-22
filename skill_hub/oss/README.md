# Open Source Skills (OSS)

This directory contains integrations with well-established open source libraries.

## Structure

```
oss/
├── optimization/          # Optimization algorithms
│   └── nsga2_pymoo/     # NSGA-II multi-objective optimization
├── visualization/        # Visualization tools
│   └── pareto_plot/      # Pareto front plotting
└── custom_oss_skills/    # User-defined OSS integrations
```

## Available Skills

### Optimization

#### NSGA-II (nsga2_pymoo)

Multi-objective optimization using the pymoo library's NSGA-II implementation.

**Usage:**

```yaml
skill: nsga2_pymoo
inputs:
  n_variables: 3
  objectives:
    - func: minimize_cost
    - func: maximize_efficiency
  bounds:
    - [0, 100]
    - [0, 50]
    - [10, 90]
  population_size: 100
  n_generations: 50
```

**Dependencies:**
- pymoo>=0.6.0

### Visualization

#### Pareto Plot (pareto_plot)

Interactive Pareto front visualization using Plotly.

**Usage:**

```yaml
skill: pareto_plot
inputs:
  solutions:
    - [0.5, 0.8]
    - [0.7, 0.6]
    - [0.3, 0.9]
  objectives:
    - "Cost"
    - "Performance"
  title: "Optimization Results"
```

**Dependencies:**
- plotly>=5.0.0

## Integration Guide

### Adding New OSS Skills

1. Create a new directory under `oss/<category>/<skill_name>/`
2. Add a `skill.yaml` file with the skill definition
3. Register the skill in `skill_hub/registry.yaml`
4. Document usage in this README

### Skill YAML Format

```yaml
name: skill_name
type: oss
priority: 10-20  # Higher priority than core skills
tags: [tag1, tag2]
description: |
  Detailed description of what the skill does.
dependencies:
  - package>=version
input_schema:
  input_name:
    type: type
    description: description
    required: true/false
    default: value
output_schema:
  output_name:
    type: type
    description: description
```

### Priority Guidelines

- OSS skills: 15-20 (high priority for proven implementations)
- Core skills: 8-12 (standard priority)
- Evolved skills: 10 (initial), adjusted based on performance

### Testing OSS Skills

Before registering an OSS skill:

1. Verify all dependencies are available
2. Test input/output schema compatibility
3. Validate performance on standard benchmarks
4. Document known limitations
