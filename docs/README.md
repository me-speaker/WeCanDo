# DAE-ELM 项目

基于公司模式重构的 DAE-ELM 项目。

## 项目结构

```
DAE-ELM/
├── src/                    # 核心业务代码（来自原始 engine/）
│   ├── features/          # DAE 特征工程
│   ├── models/            # 代理模型 (ELM, GP, NN, XGBoost)
│   ├── optimization/      # 优化器 (L-BFGS, NSGA2, etc.)
│   ├── sensing/          # 软感知
│   ├── data/             # 数据加载/预处理
│   └── core/             # 核心 (runner, registry, factory)
├── daeelm_agents/         # DAE-ELM 专用 Agents
├── bridge/               # 桥接层 - 连接 agents 到 src/
├── skill_hub/           # Agent 技能（可复用能力）
│   ├── core/            # 核心技能 (data_loading, pipeline, evaluation)
│   └── oss/             # 开源技能 (visualization)
├── deployment/           # 部署配置
└── tests/               # 测试
```

## 架构说明

### 三模块架构

1. **Feature Engineering (DAE)** - `src/features/`
2. **Surrogate Models** - `src/models/surrogates/`
3. **Optimization** - `src/optimization/`

### Agent 技能

`skill_hub/` 存放 Agent 可调用的技能，不是核心业务代码：

- `data_loading` - 数据加载技能
- `pipeline_execution` - 流程执行技能
- `model_evaluation` - 模型评估技能
- `visualization` - 可视化技能

## 使用方式

```python
from daeelm_agents import DAEELMCEOAgent
from bridge import TeamConnector

# 创建 CEO Agent
ceo = DAEELMCEOAgent()

# 连接器
connector = TeamConnector(project_root=".")
```
