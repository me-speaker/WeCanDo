# DAE-ELM 项目需求分析

## 1. 项目概述

- **项目名称**: DAE-ELM (Denoising Autoencoder - Extreme Learning Machine)
- **核心功能**: 模块化工业智能优化平台，支持化工配方设计闭环迭代优化
- **应用场景**: 化工配方优化（如丙烯酸树脂配方）、工艺参数优化、软测量预测

## 2. 功能需求

### 2.1 特征工程模块

#### 2.1.1 DAE 去噪自编码器
- **类**: `DenoisingAutoencoder`
- **功能**:
  - 无监督预训练，学习特征表示
  - 数据去噪
  - 降维 (input_dim -> hidden_dim)
- **结构**:
  - 编码器: Linear(input_dim, hidden_dim) -> ReLU
  - 解码器: Linear(hidden_dim, input_dim) -> Sigmoid
- **参数**:
  - `input_dim`: 输入维度
  - `hidden_dim`: 隐藏层维度（编码维度）
  - `noise_factor`: 噪声系数（用于去噪训练）
  - `epochs`: 预训练轮数
  - `lr`: 学习率

#### 2.1.2 特征选择器
- `VarianceThresholdSelector`: 方差阈值选择
- `SelectKBestTransformer`: Select-K-Best 特征选择
- `RandomForestImportanceSelector`: 随机森林重要性选择
- `LassoSelectionTransformer`: Lasso 特征选择
- `CorrelationFilterTransformer`: 相关性过滤

#### 2.1.3 领域特征
化学配方领域专用特征:
- `MolarRatioTransformer`: 摩尔比变换
- `PairwiseMolarRatioTransformer`: 成对摩尔比
- `FunctionalGroupCountTransformer`: 官能团计数
- `CrosslinkingDensityTransformer`: 交联密度
- `GlassTransitionTemperatureTransformer`: 玻璃化转变温度
- `SolubilityParameterTransformer`: 溶解度参数
- `PolydispersityIndexTransformer`: 多分散指数
- `ViscosityEstimateTransformer`: 粘度估计

#### 2.1.4 数据预处理
- `StandardScalerTransformer`: 标准化
- `MinMaxScalerTransformer`: 归一化
- `RobustScalerTransformer`: 鲁棒标准化
- `QuantileTransformer`: 分位数变换
- `PowerTransformer`: 幂变换
- `OneHotEncoderTransformer`: 独热编码
- `PolynomialFeaturesTransformer`: 多项式特征

### 2.2 代理模型模块

#### 2.2.1 ELM 极限学习机
- **类**: `ExtremeLearningMachine`
- **特点**:
  - 输入层权重随机初始化，无需训练
  - 输出层权重通过最小二乘法解析求解
  - 训练速度极快
  - 支持增量更新
  - 支持不确定性估计

#### 2.2.2 Gaussian Process
- **类**: `GaussianProcess`
- **功能**: 高斯过程回归，支持不确定性估计

#### 2.2.3 Neural Network
- **类**: `NeuralNetwork`
- **功能**: 神经网络回归模型

#### 2.2.4 XGBoost
- **类**: `XGBoostModel`
- **功能**: XGBoost 回归模型

### 2.3 优化模块

#### 2.3.1 L-BFGS
- **类**: `LBFGSOptimizer`
- **特点**: 拟牛顿法，适合可微目标函数
- **适用场景**: 大规模优化问题

#### 2.3.2 CG (共轭梯度法)
- **类**: `CGOptimizer`
- **特点**: 共轭梯度法，适用于可微函数

#### 2.3.3 Bayesian Optimization
- **类**: `BayesianOptimizer`
- **特点**: 贝叶斯优化，适用于不可微或昂贵评估的函数
- **支持**: 不确定性引导探索

#### 2.3.4 Genetic Algorithm
- **类**: `GeneticOptimizer`
- **特点**: 遗传算法，全局搜索能力强
- **适用场景**: 非凸、多峰、不可微问题

#### 2.3.5 NSGA2
- **类**: `NSGA2Optimizer`
- **特点**: 多目标遗传算法（Non-dominated Sorting GA II）
- **适用场景**: 多目标优化问题

### 2.4 软感知模块

#### 2.4.1 预测器
- **类**: `SoftSensingPredictor`
- **接口**:
  - `predict()`: 单点预测
  - `predict_batch()`: 批量预测
  - `fit()`: 训练模型
  - `validate_input()`: 输入验证
  - `inverse_transform_output()`: 反标准化输出
- **支持**: 不确定性估计

#### 2.4.2 流预测器
- **类**: `StreamPredictor`
- **特点**:
  - 实时流式数据处理
  - 环形缓冲区管理
  - 滑动窗口特征构建
  - 异步批量预测
  - 在线更新接口

## 3. 数据流

```
原始数据 (CSV/Database)
       │
       ▼
┌─────────────────┐
│   DataLoader    │ ─── 加载决策变量 X 和目标变量 y
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FeatureEngine   │ ─── 特征变换、选择、降维、领域特征构造
│  (DAE + 各种   │
│   Transformer)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Surrogate     │ ─── ELM / GP / NN / XGBoost 训练预测
│    Model       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Optimizer      │ ─── L-BFGS / CG / Bayesian / Genetic / NSGA2
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 软测量预测器     │ ─── StreamPredictor 实时预测
│ (Soft Sensing)  │
└─────────────────┘
```

## 4. 接口定义

### 4.1 核心接口

```python
# BaseFeatureEngine
class BaseFeatureEngine:
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray
    def transform(self, X: np.ndarray) -> np.ndarray

# BaseSurrogate
class BaseSurrogate:
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseSurrogate'
    def predict(self, X: np.ndarray) -> np.ndarray

# BaseOptimizer
class BaseOptimizer:
    def minimize(self, fun, x0, bounds, constraints=None) -> OptimizationResult
    def minimize_multi(self, objectives, x0, bounds, constraints=None) -> OptimizationResult
```

### 4.2 数据结构

```python
# 优化结果
@dataclass
class OptimizationResult:
    x: np.ndarray          # 最优解
    fun: float            # 最优目标函数值
    success: bool         # 是否成功收敛
    message: str          # 结果信息
    n_iter: int           # 迭代次数
    all_solutions: List[np.ndarray]  # 所有探索的解

# 预测结果
@dataclass
class PredictionResult:
    predictions: np.ndarray
    uncertainties: np.ndarray = None
    confidence_interval: tuple = None
    metadata: dict = None

# 流式预测配置
@dataclass
class StreamConfig:
    buffer_size: int = 1000
    window_size: int = 10
    update_interval: int = 100
    batch_mode: bool = True
```

### 4.3 变量定义

```python
# 决策变量
DecisionVariable:
    name: str
    bounds: Tuple[float, float]
    var_type: 'continuous' | 'discrete' | 'integer'
    description: str = None
    unit: str = None

# 目标变量
ObjectiveVariable:
    name: str
    direction: 'minimize' | 'maximize'
    target_idx: int

# 约束条件
ConstraintSpec:
    ctype: 'eq' | 'ineq'
    expression: str  # 如 "x[0] + x[1] - 1.0"
    penalty_weight: float = 1.0
```

## 5. 非功能需求

### 5.1 性能
- 支持大规模数据集处理
- ELM 训练速度极快（解析求解）
- 流式预测支持实时性要求

### 5.2 可扩展性
- 插件化架构（Registry 模式）
- 易于添加新的代理模型
- 易于添加新的优化算法
- 支持自定义领域特征

### 5.3 可维护性
- 模块间接口标准化
- 清晰的代码架构
- 完整的类型注解
- 统一的日志系统

### 5.4 部署
- 支持 Docker 环境
- 支持 pip 安装
- 支持 PyInstaller 打包

## 6. 模块依赖关系

```
deepind/              # 主包
├── cli/              # 命令行接口
├── demos/            # 演示脚本
├── io/               # 数据导入导出
├── core/             # 核心接口（符号链接）
└── engine/           # 核心引擎（符号链接）

engine/
├── features/         # 特征工程
│   ├── DAE.py       # 降噪自编码器
│   ├── transformers.py
│   ├── feature_selector.py
│   ├── dimensionality_reduction.py
│   ├── domain_features.py
│   └── fea2dae_bridge.py
├── models/          # 代理模型
│   ├── surrogates/
│   │   ├── elm.py
│   │   ├── gaussian_process.py
│   │   ├── neural_network.py
│   │   └── xgboost_model.py
│   └── postprocessors/
├── optimization/   # 优化算法
│   ├── base.py
│   ├── lbfgs.py
│   ├── cg.py
│   ├── bayesian.py
│   ├── genetic.py
│   └── nsga2.py
└── sensing/        # 软感知
    ├── predictor.py
    └── stream_predictor.py
```
