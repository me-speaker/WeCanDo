此Markdown文档直接用于Claude Code构建初始化项目。这份文档包含了详细的项目结构、配置文件和实现指南。

AutoEvolve Company - AI Agent团队架构文档

版本: 1.0  
日期: 2026-04-22  
状态: 最终版

📋 目录

架构概述
核心设计理念
Agent团队架构
Skill Hub设计
项目结构规范
实现方法
自我迭代机制
最佳实践
附录

架构概述

1.1 核心定位

AutoEvolve Company 是一个可自我迭代的智能AI Agent团队，旨在将用户需求转化为完整的软件项目，同时具备持续学习和优化的能力。

1.2 角色定义
角色   定位   职责
用户   客户   提出需求，验收成果

项目   智能公司体   自主运作，持续进化

1.3 核心特性

动态Agent激活：按需激活Agent，优化Token消耗
开源Skill优先：优先使用成熟的开源实现
自我迭代能力：内置反馈循环，持续优化
模块化设计：易于扩展和维护

核心设计理念

2.1 客户-公司体模型

用户（客户） → 提出需求
    ↓
AutoEvolve Company（公司体） → 自主执行
    ↓
完整软件项目 ← 交付产物

2.2 三层架构

战略层（CEO Agent）
    ↓
分析层（需求分析师 + 架构设计师）
    ↓
执行层（开发工程师 + 测试工程师 + 交付工程师）

2.3 动态激活原则

简单任务：激活3个Agent
中等任务：激活5个Agent
复杂任务：激活6个Agent

Agent团队架构

3.1 核心Agent列表
Agent   类型   职责   激活条件
CEO Agent   战略层   战略决策、资源调度、自我进化   始终激活

需求分析师   分析层   需求解析、任务分类、技能识别   始终激活

架构设计师   分析层   系统架构、模块划分、技术选型   复杂任务

开发工程师   执行层   核心代码实现、模块集成   所有任务

测试工程师   执行层   单元测试、集成测试、质量保障   中/复杂任务

交付工程师   执行层   打包部署、文档生成、GUI设计   中/复杂任务

3.2 Agent职责详解

3.2.1 CEO Agent（战略层）

核心职责：
接收用户需求
决定激活哪些Agent
协调Agent间协作
收集反馈并优化策略
管理Skill Hub

决策流程：

接收需求 → 评估复杂度 → 决定激活策略 → 监控执行 → 收集反馈 → 优化策略

3.2.2 需求分析师（分析层）

核心职责：
解析用户需求文本
识别任务类型（简单/中等/复杂）
提取关键要素（数据、目标、约束）
识别所需技能

输出物：
yaml
requirements:
  task_type: multi_objective_optimization
  complexity: complex
  data:
    source: data/reaction_data.csv
    features: [temperature, pressure, catalyst_type]
  goals:
    maximize: yield
    minimize: energy_consumption
  delivery:
    gui_required: true
    documentation_required: true

3.2.3 架构设计师（分析层）

核心职责：
设计系统架构
划分功能模块
选择技术栈
定义模块接口

激活条件：仅当任务复杂度为"复杂"时激活

输出物：
yaml
architecture:
  modules:
    name: data_processing
      responsibility: "数据加载与预处理"
      dependencies: []
    name: optimization_engine
      responsibility: "多目标优化算法"
      dependencies: [data_processing]
    name: gui_interface
      responsibility: "可视化界面"
      dependencies: [optimization_engine]
  technology_stack:
    language: python
    frameworks: [pymoo, streamlit]

3.2.4 开发工程师（执行层）

核心职责：
根据架构设计实现代码
调用Skill Hub中的技能
进行模块集成
编写代码注释

输出物：
src/ 目录下的所有源代码
代码注释和文档字符串

3.2.5 测试工程师（执行层）

核心职责：
生成单元测试
生成集成测试
执行测试并报告结果
识别代码质量问题

激活条件：任务复杂度为"中等"或"复杂"时激活

输出物：
tests/unit/ 单元测试
tests/integration/ 集成测试
test_report.md 测试报告

3.2.6 交付工程师（执行层）

核心职责：
生成GUI界面（如需要）
打包项目（setup.py, Dockerfile）
生成技术文档
生成用户手册

激活条件：任务复杂度为"中等"或"复杂"时激活

输出物：
deployment/ 部署文件
docs/ 文档
src/gui/ GUI代码（如需要）

3.3 Agent激活策略

3.3.1 任务复杂度判断标准
复杂度   判断标准   激活Agent数量
简单   单一模块、无需测试、无需文档   3个

中等   多个模块、需要测试、需要文档   5个

复杂   系统级、多目标优化、GUI需求   6个

3.3.2 激活流程

需求分析师解析需求，判断复杂度
CEO Agent根据复杂度决定激活策略
激活对应Agent
Agent执行任务
任务完成后，闲置Agent休眠

Skill Hub设计

4.1 三层Skill体系

Skill Hub
├── 核心Skill（core/）
│   └── 自研的基础技能
├── 开源Skill（oss/）
│   └── 接入的优秀开源库
└── 进化Skill（evolved/）
    └── 基于历史模式生成的新技能

4.2 Skill注册机制

4.2.1 注册表格式

yaml
skill_hub/registry.yaml
skill_registry:
  # 核心Skill
  data_loading:
    type: core
    priority: 10
    tags: [data, loading]
    description: "加载和解析数据文件"
  
  # 开源Skill（高优先级）
  nsga2_optimization:
    type: oss
    priority: 15
    tags: [optimization, multi-objective]
    dependencies:
      pymoo>=0.6.0
    description: "基于NSGA-II的多目标优化"
  
  # 可视化
  pareto_plot:
    type: oss
    priority: 12
    tags: [visualization, plotting]
    dependencies:
      plotly>=5.0.0
    description: "帕累托前沿可视化"

4.2.2 优先级规则

开源Skill优先级最高（15-20）
核心Skill中等优先级（8-12）
进化Skill动态调整（初始10，根据表现调整）

4.3 Skill接入规范

4.3.1 标准化接口

所有Skill必须遵循以下接口：

yaml
input:
  name: data
    type: DataFrame
    description: "输入数据"
  name: parameters
    type: dict
    description: "算法参数"

output:
  name: result
    type: dict
    description: "执行结果"
  name: metadata
    type: dict
    description: "元数据（性能指标等）"

4.3.2 依赖管理

yaml
dependencies:
  package_name>=version
  package_name==version

4.4 Skill加载策略

4.4.1 懒加载机制

初始加载：仅加载核心Skill摘要（~100 token）
按需加载：任务需要时再加载完整Skill描述
缓存机制：常用Skill缓存在内存中

4.4.2 技能选择流程

需求分析师识别所需技能标签
Skill Hub根据标签和优先级筛选候选技能
选择优先级最高的技能
加载技能并执行

项目结构规范

5.1 完整项目结构

autoevolve_company/
│
├── .company/                          # 公司体核心配置
│   ├── ceo_config.yaml                # CEO配置
│   ├── skill_registry.yaml            # Skill注册表
│   ├── memory/                        # 公司记忆库
│   │   ├── successful_patterns.json   # 成功模式
│   │   ├── skill_performance.json     # Skill表现
│   │   └── task_templates.json        # 任务模板
│   └── evolution_log/                 # 进化日志
│       └── YYYY-MM-DD_HH-MM-SS.json   # 每次进化的记录
│
├── skill_hub/                         # Skill中心
│   ├── registry.yaml                  # Skill注册总表
│   ├── core/                          # 核心Skill
│   │   ├── data_processing.yaml
│   │   ├── model_building.yaml
│   │   └── testing.yaml
│   ├── oss/                           # 开源Skill
│   │   ├── README.md                  # 接入指南
│   │   ├── optimization/
│   │   │   ├── nsga2_pymoo.yaml
│   │   │   └── bayesian_optuna.yaml
│   │   └── custom_oss_skills/         # 用户自定义
│   └── evolved/                       # 进化Skill
│       └── (自动生成)
│
├── src/                               # 源代码
│   ├── data/                          # 数据处理
│   │   ├── init.py
│   │   ├── loader.py
│   │   └── preprocessing.py
│   ├── models/                        # 模型
│   │   ├── init.py
│   │   └── architecture.py
│   ├── optimization/                  # 优化
│   │   ├── init.py
│   │   └── engine.py
│   ├── gui/                           # GUI（如需要）
│   │   ├── init.py
│   │   └── app.py
│   └── init.py
│
├── tests/                             # 测试
│   ├── unit/                          # 单元测试
│   │   ├── test_data.py
│   │   ├── test_models.py
│   │   └── test_optimization.py
│   ├── integration/                   # 集成测试
│   │   └── test_e2e.py
│   ├── init.py
│   └── test_report.md                 # 测试报告
│
├── deployment/                        # 部署
│   ├── Dockerfile
│   ├── setup.py
│   ├── requirements.txt
│   └── config/
│       └── default.yaml
│
├── docs/                              # 文档
│   ├── api/                           # API文档
│   │   ├── modules.md
│   │   └── reference.md
│   ├── user_guide/                    # 用户手册
│   │   ├── quick_start.md
│   │   ├── examples.md
│   │   └── faq.md
│   ├── architecture.md                # 架构说明
│   └── index.md
│
├── experiments/                       # 实验记录
│   ├── runs/
│   └── results/
│
├── README.md                          # 项目说明
├── .gitignore
└── LICENSE

5.2 配置文件详解

5.2.1 CEO配置（.company/ceo_config.yaml）

yaml
ceo_config:
  # 激活策略
  activation_strategy:
    simple_task:
      agents: [ceo, requirements_analyst, developer]
      max_token_budget: 1000
    medium_task:
      agents: [ceo, requirements_analyst, developer, tester, deliverer]
      max_token_budget: 2000
    complex_task:
      agents: [ceo, requirements_analyst, architect, developer, tester, deliverer]
      max_token_budget: 3000
  
  # 进化策略
  evolution:
    feedback_collection: true
    skill_weight_adjustment: true
    pattern_learning: true
    evolution_frequency: per_task
  
  # Skill Hub配置
  skill_hub:
    lazy_loading: true
    cache_size: 10
    priority_threshold: 8

5.2.2 Skill注册表（skill_hub/registry.yaml）

yaml
skill_registry:
  version: "1.0"
  last_updated: "2026-04-22"
  
  skills:
    # 核心Skill
    data_loading:
      type: core
      priority: 10
      tags: [data, loading, preprocessing]
      input_schema:
        file_path: string
        format: string (csv, excel, json)
      output_schema:
        data: DataFrame
        metadata: dict
    
    # 开源Skill
    nsga2_optimization:
      type: oss
      priority: 15
      tags: [optimization, multi-objective, evolutionary]
      dependencies:
        pymoo>=0.6.0
      input_schema:
        objectives: list
        constraints: list
        bounds: dict
      output_schema:
        solutions: list
        pareto_front: list
    
    # 可视化
    pareto_plot:
      type: oss
      priority: 12
      tags: [visualization, plotting, pareto]
      dependencies:
        plotly>=5.0.0
      input_schema:
        solutions: list
        objectives: list
      output_schema:
        figure: Figure
        html_path: string

实现方法

6.1 实现步骤

阶段1：基础框架（第1-2周）

目标：实现最小可行版本

任务清单：
[ ] 实现CEO Agent基础框架
[ ] 实现需求分析师
[ ] 实现开发工程师
[ ] 实现Skill Hub基础结构
[ ] 实现动态激活机制

验收标准：
能够处理简单任务（单模块开发）
Token消耗控制在1000以内
生成的代码可运行

阶段2：质量保障（第3-4周）

目标：添加测试和文档能力

任务清单：
[ ] 实现测试工程师
[ ] 实现交付工程师
[ ] 实现测试生成Skill
[ ] 实现文档生成Skill
[ ] 集成开源测试框架

验收标准：
能够生成单元测试
能够生成基础文档
测试覆盖率达到70%

阶段3：架构优化（第5-6周）

目标：支持复杂任务

任务清单：
[ ] 实现架构设计师
[ ] 实现系统架构设计Skill
[ ] 优化Agent协作流程
[ ] 添加复杂任务示例

验收标准：
能够处理多模块项目
能够生成合理的系统架构
支持模块间依赖管理

阶段4：自我迭代（第7-8周）

目标：实现自我进化能力

任务清单：
[ ] 实现反馈收集机制
[ ] 实现Skill权重调整
[ ] 实现模式学习
[ ] 实现进化日志

验收标准：
能够收集多维度反馈
能够基于反馈优化策略
进化过程可追溯

6.2 关键实现要点

6.2.1 Agent通信协议

轻量级通信格式：

[发件人] → [收件人]: [消息类型] | [内容摘要]

示例：
需求分析师 → CEO: TASK_CLASSIFICATION | type=complex, skills=[data,opt,gui]
CEO → 架构设计师: ACTIVATE | task_id=12345
架构设计师 → 开发工程师: ARCHITECTURE_READY | modules=[data,opt,gui]

6.2.2 上下文管理

上下文压缩策略：
移除冗余：删除重复描述
提取关键：保留核心信息
生成摘要：用摘要代替详细描述
引用机制：用引用代替完整内容

示例：
yaml
压缩前
开发工程师需要实现数据加载模块，该模块需要：
读取CSV文件
处理缺失值
标准化数据
返回DataFrame

压缩后
开发工程师: 实现 data_loading (CSV→DataFrame, 处理缺失, 标准化)

6.2.3 Skill执行流程

接收任务请求
解析所需技能标签
从Skill Hub查询候选技能
选择优先级最高的技能
加载技能描述
执行技能
返回结果
记录执行日志

6.3 技术选型建议

6.3.1 基础设施
组件   推荐方案   说明
编程语言   Python 3.10+   生态丰富，AI支持好

配置管理   YAML   人类可读，易于维护

日志系统   JSON + timestamp   结构化，便于分析

缓存机制   LRU Cache   简单高效

6.3.2 开源Skill接入
领域   推荐库   用途
优化   pymoo, optuna   多目标优化、超参优化

数据处理   pandas, scikit-learn   数据加载、预处理

可视化   plotly, matplotlib   结果可视化

测试   pytest   单元测试框架

GUI   streamlit, gradio   快速GUI开发

自我迭代机制

7.1 反馈收集

7.1.1 反馈维度
维度   指标   收集方式
代码质量   可读性(1-5)、规范性(通过/失败)、性能(执行时间)   静态分析 + 运行测试

Agent表现   任务完成度(0-100%)、错误率(%)、响应时间   执行日志

Skill表现   执行成功率(%)、输出质量(1-5)、资源消耗   Skill执行日志

客户满意度   需求满足度(1-5)、交付质量(1-5)   用户反馈（可选）

7.1.2 反馈格式

json
{
  "task_id": "20260422_120446",
  "timestamp": "2026-04-22T12:04:46+08:00",
  "feedback": {
    "code_quality": {
      "readability": 4,
      "standards_compliance": true,
      "performance": "0.23s"
    },
    "agent_performance": {
      "requirements_analyst": {
        "completion_rate": 100,
        "error_rate": 0,
        "response_time": "1.2s"
      },
      "developer": {
        "completion_rate": 95,
        "error_rate": 5,
        "response_time": "3.5s"
      }
    },
    "skill_performance": {
      "data_loading": {
        "success_rate": 100,
        "quality_score": 4,
        "token_consumption": 150
      },
      "nsga2_optimization": {
        "success_rate": 90,
        "quality_score": 5,
        "token_consumption": 300
      }
    }
  }
}

7.2 进化策略

7.2.1 短期优化（单次任务）

触发条件：任务完成后立即执行

优化内容：
调整当前任务的Skill选择
优化Agent协作顺序
调整Token分配策略

示例：

发现：nsga2_optimization执行成功率为90%
行动：将优先级从15提升到16

7.2.2 中期优化（多次任务）

触发条件：每完成10个任务或每周执行一次

优化内容：
更新Skill优先级
调整Agent激活策略
优化任务模板

示例：

发现：过去10个任务中，80%需要GUI
行动：将GUI生成设为中等任务的默认需求

7.2.3 长期优化（持续学习）

触发条件：每月执行一次或发现新模式时

优化内容：
生成新的复合Skill
优化整体架构
更新成功模式库

示例：

发现：多目标优化 + 可视化是常见组合
行动：生成新Skill "multi_objective_optimization_with_visualization"

7.3 进化日志

7.3.1 日志格式

json
{
  "evolution_id": "evol_20260422_120446",
  "timestamp": "2026-04-22T12:04:46+08:00",
  "trigger": "task_completion",
  "changes": [
    {
      "type": "skill_priority_update",
      "skill_name": "nsga2_optimization",
      "old_priority": 15,
      "new_priority": 16,
      "reason": "success_rate_increased_to_90%"
    },
    {
      "type": "pattern_learning",
      "pattern": "multi_objective_optimization_with_visualization",
      "confidence": 0.85,
      "action": "created_new_skill"
    }
  ],
  "metrics_before": {
    "avg_token_consumption": 1800,
    "success_rate": 0.85
  },
  "metrics_after": {
    "avg_token_consumption": 1750,
    "success_rate": 0.88
  }
}

7.3.2 版本控制

每次进化生成一个日志文件
每周生成一个汇总报告
每月生成一个趋势分析

最佳实践

8.1 Token优化

8.1.1 上下文压缩技巧

使用缩写：
   "开发工程师" → "开发"
   "需求分析师" → "需求"

移除修饰词：
   "负责核心代码实现的开发工程师" → "开发: 代码实现"

使用符号：
   "如果...那么..." → "if...then..."
   "和" → "+"

8.1.2 Agent激活优化

预判任务复杂度：
   根据历史数据预测
   用户可手动指定复杂度

渐进式激活：
   先激活基础Agent
   根据执行情况动态添加

8.2 Skill管理

8.2.1 开源Skill接入流程

评估：评估开源库的成熟度和适用性
封装：按照标准接口封装
测试：验证Skill的正确性
注册：添加到Skill注册表
文档：编写使用说明

8.2.2 Skill优先级调整

定期评估：每周评估Skill表现
动态调整：根据反馈调整优先级
版本管理：保留历史版本

8.3 质量保障

8.3.1 代码质量

遵循规范：PEP 8、Google Python Style Guide
充分注释：函数、类、模块级注释
类型提示：使用Type Hints

8.3.2 测试覆盖

单元测试：覆盖核心逻辑
集成测试：验证模块协作
边界测试：测试极端情况

8.4 文档规范

8.4.1 技术文档

架构说明：系统架构图、模块说明
API文档：函数签名、参数说明、返回值
部署指南：环境要求、安装步骤、配置说明

8.4.2 用户手册

快速开始：5分钟上手指南
使用示例：常见场景示例
FAQ：常见问题解答

附录

9.1 术语表
术语   定义
Agent   具有特定职责的AI实体

Skill   可复用的功能模块

公司体   整个AI Agent团队的统称

动态激活   根据任务需求激活Agent的机制

自我迭代   基于反馈持续优化的能力

9.2 任务复杂度判断矩阵
特征   简单   中等   复杂
模块数量   1   2-3   4+

是否需要测试   否   是   是

是否需要文档   否   是   是

是否需要GUI   否   可选   是

算法复杂度   低   中   高

数据规模   小   中   大

9.3 Skill接入检查清单

[ ] 是否遵循标准接口
[ ] 是否声明了所有依赖
[ ] 是否有完整的输入/输出说明
[ ] 是否有使用示例
[ ] 是否经过测试验证
[ ] 是否有性能基准
[ ] 是否有错误处理
[ ] 是否有版本信息

9.4 常见问题

Q1: 如何添加新的Agent？

A: 
在.company/ceo_config.yaml中定义新Agent
实现Agent的职责和接口
更新激活策略
测试集成

Q2: 如何接入新的开源Skill？

A:
评估开源库的适用性
按照标准接口封装
在skill_hub/registry.yaml中注册
测试验证
更新文档

Q3: Token消耗过高怎么办？

A:
检查是否激活了不必要的Agent
优化上下文压缩策略
调整Skill加载策略
使用缓存机制

Q4: 如何查看进化历史？

A:
查看.company/evolution_log/目录
查看每周/每月汇总报告
分析进化趋势

📝 版本历史
版本   日期   变更内容   作者
1.0   2026-04-22   初始版本   AI Architect

文档结束

这份Markdown文档已经完整整理好了，可以直接保存为 ARCHITECTURE.md 文件，供Claude Code使用来构建初始化项目。文档包含了：

完整的架构说明：6个Agent的详细职责和激活策略
项目结构规范：完整的目录结构和配置文件
实现方法：分阶段的实现步骤和关键技术要点
自我迭代机制：反馈收集、进化策略和日志格式
最佳实践：Token优化、Skill管理、质量保障等

你可以直接将这份文档提供给Claude Code，它应该能够理解并构建出符合架构要求的初始化项目。