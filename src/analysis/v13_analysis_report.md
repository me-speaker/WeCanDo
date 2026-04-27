# DeepInd v1.3 分析报告

## 1. 项目结构分析

### 1.1 整体架构

DeepInd 采用三模块解耦架构：
- **GUI层** (`gui/`)：PyQt5 图形界面
- **引擎层** (`engine/`)：核心ML引擎
- **核心接口层** (`core/`)：抽象基类和注册机制

```
deepind/ (CLI包，符号链接指向外部)
├── cli/           # 命令行入口
├── io/            # 数据导入导出
├── gui -> ../gui/ # GUI符号链接
├── core -> ../core/
└── engine -> ../engine/

engine/
├── core/          # factory.py, registry.py, runner.py, base.py
├── features/      # 特征工程
├── models/        # 代理模型
├── optimization/   # 优化算法
└── analysis/       # 性能分析
```

### 1.2 核心模块职责

| 模块 | 文件 | 职责 |
|------|------|------|
| 注册机制 | `engine/core/registry.py` | 6个全局注册器管理各类型模块 |
| 工厂函数 | `engine/core/factory.py` | 根据配置字典动态创建模块实例 |
| 任务运行器 | `engine/core/runner.py` | 串联数据加载→训练→优化→后处理流程 |
| 优化器 | `engine/optimization/auto_optimizer.py` | 根据模型可微性自动选择优化算法 |

### 1.3 数据流

```
配置文件 (YAML)
    ↓
TaskRunner._build_modules()  # 实例化所有模块
    ↓
dataloader.load() → X_raw, y_raw
    ↓
dataloader.preprocess() → X, y
    ↓
feature_engine.fit_transform() → X_features
    ↓
surrogate.fit() → 训练模型
    ↓
optimizer.optimize() → Pareto解
    ↓
postprocessors[] → 最终结果
```

---

## 2. 参数传递链路分析

### 2.1 GUI配置参数结构

`gui/config_widget.py` 中的 `ConfigWidget` 类维护的配置结构：

```python
self.config = {
    'model_type': 'NeuralNetwork',
    'model_params': {},
    'optimizer': 'AutoOptimizer',
    'optimizer_params': {}
}
```

**GUI参数收集流程** (`on_apply` 方法):
1. 从 UI 控件读取用户输入
2. 根据控件类型（spin/double/line/combo）提取值
3. 组装为 `config` 字典
4. 返回给调用者

### 2.2 GUI→引擎的参数传递（断点分析）

**问题1：配置格式不匹配**

GUI返回的配置格式：
```python
{
    'model_type': 'NeuralNetwork',      # 字符串
    'model_params': {'hidden_dims': [128, 64, 32], ...},  # 字典
    'optimizer': 'AutoOptimizer',       # 字符串
    'optimizer_params': {...}
}
```

引擎期望的配置格式（`factory.py`）:
```python
{"type": "NeuralNetwork", "params": {...}}  # 嵌套字典
```

**问题2：main_window.py 中 on_run_optimization 是空壳**

```python
def on_run_optimization(self):
    # ...
    config = self.config_widget.get_config()
    # TODO: Actually run the optimization with the selected config
    # This would involve:
    # 1. Load data
    # 2. Create and train the selected model with given params
    # 3. Run optimization with selected optimizer
    # 4. Display results
```

**问题3：GUI与TaskRunner的配置不兼容**

GUI配置 → 需要转换 → TaskRunner期望的配置格式

GUI的`model_type`对应`factory.create_surrogate`的`cfg["type"]`
GUI的`model_params`对应`factory.create_surrogate`的`cfg["params"]`

### 2.3 TaskRunner期望的配置格式

```python
self.config = {
    "task_name": "DeepInd_Task",
    "output_mode": "user",
    "data": {"type": "...", "params": {...}},
    "models": {
        "feature_engine": {"type": "...", "params": {...}},
        "surrogate": {"type": "...", "params": {...}},
        "optimizer": {"type": "...", "params": {...}},
    },
    "constraints": {"handler": "...", "params": {...}},
    "postprocess": {"chain": [...]},
    "decision_variables": [...],
    "objectives": [...],
}
```

### 2.4 参数传递链路图

```
GUI (config_widget.py)
    │
    │ get_config() 返回 {model_type, model_params, optimizer, optimizer_params}
    │
    ▼
main_window.py::on_run_optimization()
    │ [TODO: 未实现]
    │
    ▼
(断点) 缺少 GUI配置 → TaskRunner配置 的转换层
    │
    ▼
TaskRunner(engine/core/runner.py)
    │ _build_modules() 期望 models.surrogate {type, params}
    │
    ▼
factory.py::create_surrogate(cfg)
    │ 期望 cfg = {"type": "...", "params": {...}}
    │
    ▼
SURROGATE_REGISTRY.get(type)
```

---

## 3. 问题与瓶颈

### 3.1 GUI配置功能缺失

| 问题 | 严重程度 | 说明 |
|------|---------|------|
| 配置参数收集后未转换 | **高** | GUI返回的格式与引擎期望格式不一致 |
| on_run_optimization是空壳 | **高** | 配置无法传递给引擎执行 |
| 无法选择特征工程模块 | **中** | GUI只有模型和优化器配置，无特征工程选项 |
| 无法配置决策变量和目标 | **高** | 这些是优化的核心参数，GUI完全缺失 |
| 无数据加载配置 | **高** | 无法在GUI中指定数据文件路径 |
| 无后处理器配置 | **低** | 缺失但影响较小 |

### 3.2 参数传递链路问题

1. **格式转换缺失**：`ConfigWidget.get_config()` 输出的格式无法直接传递给 `TaskRunner`
2. **中间层缺失**：没有 `ConfigManager` 或 `TaskConfigAdapter` 来桥接GUI和引擎
3. **配置验证缺失**：GUI输入的参数没有验证机制

### 3.3 配置分离问题

- **GUI配置**：内存字典，由 `ConfigWidget` 管理
- **YAML配置**：文件配置，由 `config/` 目录下的文件管理
- **运行时配置**：`TaskRunner` 期望的内部格式

三者之间缺乏统一的配置抽象层。

---

## 4. 改进建议

### 4.1 短期改进（高优先级）

1. **实现GUI→引擎配置转换层**
   - 创建 `ConfigAdapter` 类
   - 将 GUI 配置格式转换为 `TaskRunner` 格式
   - 在 `main_window.py::on_run_optimization()` 中调用

2. **完善 on_run_optimization 实现**
   - 加载数据文件
   - 实例化 `TaskRunner`
   - 执行 `run()` 流程
   - 展示结果

3. **添加数据文件选择功能**
   - 在 GUI 中增加数据路径配置
   - 支持 CSV/JSON 格式

### 4.2 中期改进（中优先级）

4. **增加决策变量和目标配置UI**
   - 让用户指定哪些列是决策变量
   - 让用户指定哪些列是优化目标

5. **增加特征工程配置UI**
   - 支持选择特征工程类型
   - 配置特征工程参数

6. **添加配置验证**
   - 参数范围验证
   - 必需参数检查

### 4.3 长期改进（低优先级）

7. **统一配置管理系统**
   - 创建 `ConfigManager` 类
   - 支持 YAML/JSON/内存字典三种配置源
   - 实现配置合并和覆盖逻辑

8. **配置文件模板系统**
   - 预设不同场景的配置模板
   - 用户可基于模板修改

---

## 5. 文档问题

### 5.1 文档结构混乱

| 文件 | 问题 |
|------|------|
| `docs/user_manual.md` | 存在[截图片段]，说明文档未完成；引用了不存在的标签页（Data Import） |
| `docs/library_api.md` | 导入路径错误，如 `from engine.features import ...` 应该是 `from engine.features.transformers import ...` |
| `docs/developer_guide.md` | 存在占位符内容 |
| `config/config.md` | 仅包含标题说明，无实际配置说明 |
| `docs/packaging.md` | 已删除但 git status 显示为 deleted 状态 |

### 5.2 具体文档错误

1. **user_manual.md 第67-80行**：YAML配置示例格式不完整
   ```yaml
   model:
     type: elm  # type小写，但实际应使用大写或与注册名匹配
   optimization:
     objectives:
       - hardnness  # 拼写错误
   ```

2. **library_api.md**：导入路径与实际项目结构不符
   - `from engine.features import ...` 缺少子模块路径
   - `from engine.optimization import AutoOptimizer` 实际应为 `from engine.optimization.auto_optimizer import AutoOptimizer`

3. **文档版本信息不一致**
   - user_manual.md: "最后更新：2026-04-14"
   - library_api.md: "最后更新：2026-04-14"
   - developer_guide.md: "最后更新：2026-04-13"

### 5.3 文档维护建议

1. **清理待删除文件**：将 `docs/packing.md` 从 git 索引中移除（如果确实不需要）
2. **统一文档格式**：使用相同的文档模板
3. **增加文档验证**：确保示例代码可执行
4. **补充缺失文档**：
   - 配置文件完整格式说明
   - API接口完整文档
   - 错误码说明

---

## 6. 总结

### 核心瓶颈

1. **GUI→引擎参数传递链断裂**：GUI收集的配置无法直接用于 `TaskRunner`
2. **on_run_optimization 是空壳**：核心执行逻辑未实现
3. **配置格式不统一**：GUI、YAML、运行时配置三者各异

### 关键改进路径

1. 创建配置适配层（`ConfigAdapter`）
2. 实现 `on_run_optimization` 完整逻辑
3. 统一配置管理架构
4. 清理和修复文档

### 影响评估

- 当前状态下 GUI **无法** 执行任何优化任务
- 用户只能通过 CLI 或直接调用 Python API 使用系统
- 文档中的示例代码大部分 **无法直接运行**
