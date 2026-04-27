# DAE-ELM 架构设计文档

## 1. 模块映射表 (Original → Company Mode)

| 原始位置 | 新位置 | 说明 |
|---------|--------|------|
| `src/features/DAE.py` | `src/features/DAE.py` | DAE 去噪自编码器，保持原路径 |
| `src/features/transformers.py` | `src/features/transformers.py` | 数据预处理变换器 |
| `src/features/feature_selector.py` | `src/features/feature_selector.py` | 特征选择器 |
| `src/features/domain_features.py` | `src/features/domain_features.py` | 领域特征 (化学配方) |
| `src/features/dimensionality_reduction.py` | `src/features/dimensionality_reduction.py` | 降维方法 |
| `src/features/fea2dae_bridge.py` | `src/features/fea2dae_bridge.py` | 特征转 DAE 桥接 |
| `src/models/surrogates/elm.py` | `src/models/surrogates/elm.py` | ELM 极限学习机 |
| `src/models/surrogates/gaussian_process.py` | `src/models/surrogates/gaussian_process.py` | 高斯过程 |
| `src/models/surrogates/neural_network.py` | `src/models/surrogates/neural_network.py` | 神经网络 |
| `src/models/surrogates/xgboost_model.py` | `src/models/surrogates/xgboost_model.py` | XGBoost |
| `src/optimization/algorithms/lbfgs.py` | `src/optimization/algorithms/lbfgs.py` | L-BFGS 优化器 |
| `src/optimization/algorithms/cg.py` | `src/optimization/algorithms/cg.py` | 共轭梯度法 |
| `src/optimization/algorithms/bayesian.py` | `src/optimization/algorithms/bayesian.py` | 贝叶斯优化 |
| `src/optimization/algorithms/genetic.py` | `src/optimization/algorithms/genetic.py` | 遗传算法 |
| `src/optimization/algorithms/nsga2.py` | `src/optimization/algorithms/nsga2.py` | NSGA2 多目标 |
| `src/sensing/predictor.py` | `src/sensing/predictor.py` | 软测量预测器 |
| `src/sensing/stream_predictor.py` | `src/sensing/stream_predictor.py` | 流式预测器 |
| `src/data/` | `src/data/` | 数据加载与预处理 |
| `src/core/runner.py` | `src/core/runner.py` | 核心运行器 |
| `src/core/registry.py` | `src/core/registry.py` | 注册中心 |
| `src/core/factory.py` | `src/core/factory.py` | 工厂模式 |
| `src/core/config_parser.py` | `src/core/config_manager.py` | 配置管理 (重命名) |
| `src/core/base.py` | `src/core/base.py` | 基础接口定义 |
| `N/A` | `daeelm_agents/` | **新增** DAE-ELM 专用 Agents |
| `N/A` | `bridge/` | **新增** Agent 与 src/ 的桥接层 |
| `N/A` | `skill_hub/` | **新增** 可复用技能库 |
| `N/A` | `deployment/` | **新增** Docker 部署配置 |
| `N/A` | `tests/` | **新增** 测试目录 |

---

## 2. 目标目录结构

```
generated/DAE_ELM/
├── src/                          # 原型代码 → 核心业务模块
│   ├── features/                 # 特征工程
│   │   ├── DAE.py               # 降噪自编码器
│   │   ├── base.py              # 特征基类
│   │   ├── transformers.py       # 预处理变换器
│   │   ├── feature_selector.py  # 特征选择器
│   │   ├── domain_features.py    # 化学配方领域特征
│   │   ├── dimensionality_reduction.py
│   │   └── fea2dae_bridge.py    # 特征转 DAE 桥接
│   ├── models/                   # 代理模型
│   │   ├── surrogates/
│   │   │   ├── elm.py           # 极限学习机
│   │   │   ├── gaussian_process.py
│   │   │   ├── neural_network.py
│   │   │   ├── xgboost_model.py
│   │   │   ├── get_fit_para.py
│   │   │   └── funcs_fit.py
│   │   └── postprocessors/
│   │       └── saver.py         # 模型保存/加载
│   ├── optimization/             # 优化算法
│   │   ├── base.py              # 优化器基类
│   │   ├── algorithms/
│   │   │   ├── lbfgs.py
│   │   │   ├── cg.py
│   │   │   ├── bayesian.py
│   │   │   ├── genetic.py
│   │   │   ├── nsga2.py
│   │   │   ├── auto_optimizer.py
│   │   │   ├── funcs_CG.py
│   │   │   └── get_opt_by_CG-FR_foil.py
│   │   ├── constraints/
│   │   │   └── penalty.py       # 惩罚函数
│   │   └── configs/
│   │       └── optimizer_config.yaml
│   ├── sensing/                  # 软感知
│   │   ├── predictor.py          # 预测器基类
│   │   ├── stream_predictor.py   # 流式预测
│   │   └── timeseries_predictor.py
│   ├── data/                     # 数据模块
│   │   ├── dataset/
│   │   │   └── formula_loader.py # 配方加载器
│   │   ├── preprocessing/
│   │   │   ├── base.py
│   │   │   ├── noise_reduction.py
│   │   │   ├── missing_value_imputation.py
│   │   │   ├── outlier_detection.py
│   │   │   └── signal_smoothing.py
│   │   ├── process_generator.py
│   │   ├── recipe_generator.py
│   │   ├── data_factory.py
│   │   ├── performance_generator.py
│   │   └── config.yaml
│   ├── core/                     # 核心模块
│   │   ├── base.py              # 基础接口
│   │   ├── runner.py            # 运行器
│   │   ├── registry.py          # 注册中心
│   │   ├── factory.py           # 工厂
│   │   ├── config_manager.py    # 配置管理
│   │   ├── config_parser.py     # 配置解析
│   │   ├── data_loader.py       # 数据加载
│   │   └── model_updater.py     # 模型更新
│   ├── analysis/                 # 分析工具
│   │   ├── benchmark.py
│   │   ├── memory_tracker.py
│   │   ├── bottleneck_detector.py
│   │   └── profiler.py
│   └── utils/
│       └── logger.py
│
├── daeelm_agents/                # DAE-ELM 专用 Agents
│   ├── __init__.py
│   ├── requirements_analyst.py   # 需求分析 Agent
│   ├── architect.py              # 架构设计 Agent
│   ├── developer.py             # 开发 Agent
│   ├── tester.py                # 测试 Agent
│   └── delivery.py              # 交付 Agent
│
├── bridge/                       # 桥接层
│   ├── __init__.py
│   ├── task_mapper.py           # 任务映射 (Agent Task → src/ 模块)
│   ├── task_executor.py         # 任务执行器
│   ├── team_connector.py        # 团队连接器
│   └── project_scaffold.py     # 项目脚手架生成
│
├── skill_hub/                    # 可复用技能库
│   ├── core/                    # 核心技能
│   │   ├── data_loading/       # 数据加载技能
│   │   │   └── skill.yaml
│   │   ├── model_building/      # 模型构建技能
│   │   │   └── skill.yaml
│   │   └── testing/             # 测试技能
│   │       └── skill.yaml
│   ├── oss/                     # 开源技能
│   │   ├── optimization/        # 优化算法技能
│   │   │   └── nsga2_pymoo/
│   │   │       └── skill.yaml
│   │   └── visualization/       # 可视化技能
│   │       └── pareto_plot/
│   │           └── skill.yaml
│   └── registry.yaml           # 技能注册表
│
├── deployment/                   # 部署配置
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
│
├── tests/                        # 测试
│   ├── unit/                    # 单元测试
│   │   ├── features/
│   │   ├── models/
│   │   ├── optimization/
│   │   └── sensing/
│   ├── integration/            # 集成测试
│   └── benchmarks/              # 性能测试
│
├── docs/                        # 文档
│   ├── REQUIREMENTS.md         # 需求分析文档
│   ├── ARCHITECTURE.md         # 本文档
│   └── API.md                  # API 文档
│
└── CLAUDE.md                    # Agent 系统配置
```

---

## 3. Agent 规格说明 (daeelm_agents/)

### 3.1 基础架构

所有 DAE-ELM Agents 继承自 `/deepind/deepind_v1/agents/base.py` 中的 `BaseAgent`，扩展公司模式。

```
BaseAgent (from /deepind/deepind_v1/agents/base.py)
    │
    └── DaeElmBaseAgent (daeelm_agents/base.py)
            ├── DaeElmRequirementsAnalyst
            ├── DaeElmArchitect
            ├── DaeElmDeveloper
            ├── DaeElmTester
            └── DaeElmDelivery
```

### 3.2 各 Agent 职责

#### DaeElmRequirementsAnalyst
- **职责**: 解析用户对化工配方优化的需求
- **输入**: 用户描述 (如 "优化丙烯酸树脂配方")
- **输出**: `REQUIREMENTS.md` 格式的需求文档
- **工具**: `DataLoader` 分析现有数据格式
- **特定能力**: 理解化工配方变量、目标函数、约束条件

#### DaeElmArchitect
- **职责**: 设计 DAE-ELM 系统的模块架构
- **输入**: `REQUIREMENTS.md`
- **输出**: `ARCHITECTURE.md`
- **工具**: 分析 `src/` 现有代码，设计模块映射
- **特定能力**: 理解特征工程 → 代理模型 → 优化的完整流程

#### DaeElmDeveloper
- **职责**: 实现和重构 `src/` 中的业务代码
- **输入**: `ARCHITECTURE.md`
- **输出**: 符合架构的代码实现
- **工具**: 调用 `bridge/` 连接 `src/` 模块
- **特定能力**:
  - 实现 `DenoisingAutoencoder`
  - 实现 `ExtremeLearningMachine`
  - 实现各类 `Optimizer`

#### DaeElmTester
- **职责**: 验证 `src/` 模块的功能正确性
- **输入**: `ARCHITECTURE.md` + `src/` 实现
- **输出**: 测试报告
- **工具**: `skill_hub/core/testing/`
- **特定能力**: 测试化工配方优化 pipeline

#### DaeElmDelivery
- **职责**: 打包和部署 DAE-ELM 系统
- **输入**: 通过测试的 `src/` 代码
- **输出**: Docker 镜像、pip 包
- **工具**: `deployment/` 配置
- **特定能力**: 构建 `Dockerfile`，生成 `docker-compose.yml`

---

## 4. Bridge 层设计

### 4.1 模块关系图

```
┌─────────────────────────────────────────────────────────┐
│                    Company Mode                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │   CEO    │→ │ Analyst  │→ │Architect │→ │ Dev    │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
│                                                    │     │
└────────────────────────────────────────────────────┼─────┘
                                                     │
                         ┌───────────────────────────┴────┐
                         │         bridge/                  │
                         │  ┌────────────┐ ┌───────────┐  │
                         │  │task_mapper │→│task_exec   │  │
                         │  └────────────┘ └───────────┘  │
                         │  ┌────────────────────────────┐ │
                         │  │   project_scaffold.py      │ │
                         │  └────────────────────────────┘ │
                         └────────────────────┬────────────┘
                                              │
                         ┌────────────────────▼───────────┐
                         │           src/                   │
                         │  features/ models/ optimization/ │
                         │  sensing/ data/ core/            │
                         └─────────────────────────────────┘
```

### 4.2 核心组件

#### TaskMapper (bridge/task_mapper.py)
将 Agent 任务映射到 `src/` 模块的具体函数/类。

```python
class DaeElmTaskMapper:
    """DAE-ELM 任务映射器"""

    TASK_REGISTRY = {
        "feature_engine.dae_train": {
            "module": "src.features.DAE",
            "class": "DenoisingAutoencoder",
            "method": "fit"
        },
        "surrogate.elm_train": {
            "module": "src.models.surrogates.elm",
            "class": "ExtremeLearningMachine",
            "method": "fit"
        },
        "optimizer.minimize": {
            "module": "src.optimization.algorithms",
            "resolver": "resolve_optimizer",  # 根据配置动态解析
        },
        "sensing.predict": {
            "module": "src.sensing.predictor",
            "class": "SoftSensingPredictor",
            "method": "predict"
        }
    }
```

#### TaskExecutor (沿用 /deepind/deepind_v1/bridge/task_executor.py)
执行 `TaskMapper` 解析后的任务，调用 `src/` 中的实际代码。

#### ProjectScaffold (bridge/project_scaffold.py)
根据架构设计生成项目目录结构和初始文件。

```python
class DaeElmScaffoldGenerator:
    """生成 DAE-ELM 项目脚手架"""

    def generate(self, architecture_spec: dict) -> List[FileTask]:
        """根据架构规格生成文件创建任务"""
        tasks = []
        tasks.extend(self._generate_src_structure())
        tasks.extend(self._generate_daeelm_agents())
        tasks.extend(self._generate_bridge())
        tasks.extend(self._generate_skill_hub())
        tasks.extend(self._generate_deployment())
        return tasks
```

#### TeamConnector (沿用 /deepind/deepind_v1/bridge/team_connector.py)
连接各 Agent 之间的通信。

---

## 5. Skill Hub 设计

### 5.1 技能分类

```
skill_hub/
├── core/                    # 核心技能 (系统必须)
│   ├── data_loading/        # 数据加载
│   │   └── skill.yaml
│   ├── model_building/      # 模型构建
│   │   └── skill.yaml
│   └── testing/             # 测试
│       └── skill.yaml
│
├── oss/                     # 开源技能 (可选)
│   ├── optimization/        # 优化算法
│   │   └── nsga2_pymoo/   # 基于 pymoo 的 NSGA2
│   │       └── skill.yaml
│   └── visualization/      # 可视化
│       └── pareto_plot/
│           └── skill.yaml
│
└── registry.yaml           # 技能注册表
```

### 5.2 Skill YAML 格式

```yaml
# skill_hub/core/data_loading/skill.yaml
name: daeelm_data_loading
category: data
version: "1.0"
description: "加载化工配方数据 (CSV/Database)"
capabilities:
  - csv_loading
  - database_connectors
  - formula_parsing
dependencies:
  - core/data_loading  # 继承基础技能

implementation:
  loader_class: src.data.dataset.formula_loader.FormulaLoader
  methods:
    - load_csv
    - load_database
    - parse_formula

# skill_hub/oss/optimization/nsga2_pymoo/skill.yaml
name: nsga2_pymoo
category: optimization
version: "1.0"
description: "基于 pymoo 的 NSGA2 多目标优化"
capabilities:
  - multi_objective_optimization
  - pareto_front
  - constraint_handling
dependencies:
  - core/model_building
external_libs:
  - pymoo
implementation:
  optimizer_class: src.optimization.algorithms.NSGA2Optimizer
```

---

## 6. 数据流图

```
用户需求 (配方优化请求)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  daeelm_agents/                                         │
│                                                         │
│  DaeElmRequirementsAnalyst                             │
│  (解析需求 → DecisionVariable, ObjectiveVariable)       │
│                          │                              │
│                          ▼                              │
│  DaeElmArchitect                                        │
│  (设计 Pipeline: 特征→模型→优化)                         │
│                          │                              │
│                          ▼                              │
│  DaeElmDeveloper                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  bridge/task_mapper.py                           │   │
│  │  将任务映射到 src/ 模块                           │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  src/ (核心业务模块)                                      │
│                                                         │
│  1. data/formula_loader.py                             │
│     └→ DataLoader → 配方数据 (X: 决策变量, y: 目标)       │
│                        │                                │
│                        ▼                                │
│  2. features/DAE.py                                     │
│     └→ DenoisingAutoencoder → 降噪特征                  │
│                        │                                │
│                        ▼                                │
│  3. features/transformers.py                           │
│     └→ 标准化、归一化、领域特征                           │
│                        │                                │
│                        ▼                                │
│  4. models/surrogates/elm.py                          │
│     └→ ExtremeLearningMachine → 预测模型                │
│                        │                                │
│                        ▼                                │
│  5. optimization/algorithms/                           │
│     └→ L-BFGS / Bayesian / Genetic / NSGA2            │
│                        │                                │
│                        ▼                                │
│  6. sensing/predictor.py                               │
│     └→ SoftSensingPredictor → 输出优化结果              │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  daeelm_agents/                                         │
│                                                         │
│  DaeElmTester (验证 Pipeline)                           │
│                          │                              │
│                          ▼                              │
│  DaeElmDelivery (打包部署)                              │
│     └→ Docker 镜像 / pip 包                            │
└─────────────────────────────────────────────────────────┘
```

---

## 7. 实现优先级

### Phase 1: 基础结构 (优先级 1)
1. 创建目录结构 (`src/`, `daeelm_agents/`, `bridge/`, `skill_hub/`, `deployment/`, `tests/`)
2. 配置 `skill_hub/registry.yaml`
3. 实现 `daeelm_agents/base.py` (继承 BaseAgent)
4. 实现 `bridge/task_mapper.py` 任务映射

### Phase 2: 核心模块迁移 (优先级 2)
1. 迁移 `src/features/` 特征工程模块
2. 迁移 `src/models/surrogates/` 代理模型模块
3. 迁移 `src/optimization/` 优化算法模块
4. 迁移 `src/sensing/` 软感知模块
5. 迁移 `src/data/` 数据加载模块
6. 完善 `src/core/` 核心模块 (runner, registry, factory)

### Phase 3: Agent 实现 (优先级 3)
1. 实现 `DaeElmRequirementsAnalyst`
2. 实现 `DaeElmArchitect`
3. 实现 `DaeElmDeveloper`
4. 实现 `DaeElmTester`
5. 实现 `DaeElmDelivery`

### Phase 4: 部署与测试 (优先级 4)
1. 创建 `deployment/Dockerfile`
2. 创建 `deployment/docker-compose.yml`
3. 编写 `tests/` 单元测试
4. 编写 `tests/` 集成测试
5. 添加 `skill_hub/oss/` 开源技能

---

## 8. 关键设计决策

### 8.1 为什么保留 `src/` 结构不变?
原 `src/` 已经按照 `features/` → `models/` → `optimization/` → `sensing/` 的 pipeline 组织，映射到公司模式时保持 `src/` 不变，只在其外层添加 `daeelm_agents/`、`bridge/` 等公司模式组件。

### 8.2 Bridge 层的作用
Bridge 层是公司模式与原型的连接器:
- **TaskMapper**: 将 Agent 的"做什么"映射到 `src/` 的"怎么做"
- **TaskExecutor**: 执行映射后的任务
- **ProjectScaffold**: 根据架构生成项目结构

### 8.3 Skill Hub 与 `src/` 的区别
- **`src/`**: 核心业务代码，包含 DAE、ELM、优化算法等业务逻辑
- **`skill_hub/`**: 可复用的技能定义，如"如何加载数据"、"如何测试"，是 Agent 的能力单元，不直接包含业务逻辑

### 8.4 命名空间
所有 DAE-ELM 专用代码使用 `daeelm_` 前缀:
- `daeelm_agents/`
- `daeelm_bridge/` (如需要独立)
- skill hub 中技能名以 `daeelm_` 开头
