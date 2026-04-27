"""
DeepInd 性能瓶颈检测器

自动识别代码中的性能瓶颈并提供优化建议
"""

import ast
import time
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Bottleneck:
    """性能瓶颈"""
    location: str
    component: str
    severity: str  # "high", "medium", "low"
    bottleneck_type: str
    description: str
    estimated_overhead_percent: float
    recommendations: List[str]


class BottleneckDetector:
    """
    性能瓶颈检测器

    检测类型:
    - 算法复杂度问题 (O(n^3) 矩阵运算等)
    - 内存分配问题 (频繁分配/释放)
    - GPU/CPU 数据传输
    - 同步/阻塞操作
    - 串行处理可并行化部分
    """

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.bottlenecks: List[Bottleneck] = []

    def detect_all(self) -> List[Bottleneck]:
        """
        执行全面瓶颈检测

        Returns:
            检测到的瓶颈列表
        """
        self.bottlenecks = []
        self._detect_gp_bottlenecks()
        self._detect_nn_bottlenecks()
        self._detect_elm_bottlenecks()
        self._detect_optimizer_bottlenecks()
        return self.bottlenecks

    def _detect_gp_bottlenecks(self):
        """检测高斯过程模型的瓶颈"""
        gp_path = self.project_root / "engine/models/surrogates/gaussian_process.py"
        if not gp_path.exists():
            return

        content = gp_path.read_text()

        # 1. Cholesky 分解 - O(n^3) 复杂度
        if "cholesky" in content.lower():
            self.bottlenecks.append(Bottleneck(
                location="GaussianProcess.fit",
                component="GaussianProcess",
                severity="high",
                bottleneck_type="algorithmic_complexity",
                description="Cholesky分解复杂度为O(n^3)，样本数增加时急剧变慢",
                estimated_overhead_percent=40.0,
                recommendations=[
                    "使用稀疏近似方法(Sparse GP)如 inducing points",
                    "考虑使用SVD分解替代Cholesky",
                    "对大数据集使用随机傅里叶特征近似",
                ],
            ))

        # 2. 核矩阵存储 - O(n^2) 内存
        if "_K_inv" in content and "K_noise" in content:
            self.bottlenecks.append(Bottleneck(
                location="GaussianProcess.fit",
                component="GaussianProcess",
                severity="medium",
                bottleneck_type="memory",
                description="存储完整的核矩阵逆需要O(n^2)内存",
                estimated_overhead_percent=20.0,
                recommendations=[
                    "使用稀疏核近似减少内存占用",
                    "实施分批预测策略",
                ],
            ))

        # 3. 核函数计算效率
        if "np.sum(X1 ** 2, axis=1, keepdims=True)" in content:
            self.bottlenecks.append(Bottleneck(
                location="GaussianProcess._rbf_kernel",
                component="GaussianProcess",
                severity="low",
                bottleneck_type="computation",
                description="核函数计算可进一步向量化优化",
                estimated_overhead_percent=10.0,
                recommendations=[
                    "使用torch替代numpy利用GPU加速",
                ],
            ))

    def _detect_nn_bottlenecks(self):
        """检测神经网络的瓶颈"""
        nn_path = self.project_root / "engine/models/surrogates/neural_network.py"
        if not nn_path.exists():
            return

        content = nn_path.read_text()

        # 1. GPU 可用但未使用
        if "cuda.is_available()" in content and "_device" in content:
            # 检查是否真的把数据移到GPU
            if ".to(self._device)" not in content or content.count(".to(self._device)") < 3:
                self.bottlenecks.append(Bottleneck(
                    location="NeuralNetwork",
                    component="NeuralNetwork",
                    severity="high",
                    bottleneck_type="hardware_utilization",
                    description="GPU可用但未充分利用，数据传输到GPU的代码路径可能有问题",
                    estimated_overhead_percent=50.0,
                    recommendations=[
                        "确保所有tensor都正确迁移到GPU",
                        "检查.to(self._device)调用位置",
                        "使用torch.cuda.synchronize()确保GPU操作完成",
                    ],
                ))

        # 2. 训练循环无 early stopping
        if "for epoch in range(self.epochs)" in content:
            if "early_stopping" not in content.lower():
                self.bottlenecks.append(Bottleneck(
                    location="NeuralNetwork.fit",
                    component="NeuralNetwork",
                    severity="medium",
                    bottleneck_type="training_efficiency",
                    description="训练使用固定epoch数，可能导致过拟合或训练不足",
                    estimated_overhead_percent=15.0,
                    recommendations=[
                        "添加early stopping机制",
                        "监控验证集性能",
                        "使用学习率调度器",
                    ],
                ))

        # 3. 数据加载无缓存
        if "TensorDataset" in content and "cache" not in content.lower():
            self.bottlenecks.append(Bottleneck(
                location="NeuralNetwork.fit",
                component="NeuralNetwork",
                severity="low",
                bottleneck_type="io",
                description="每个epoch都重新创建DataLoader，无数据缓存",
                estimated_overhead_percent=5.0,
                recommendations=[
                    "考虑数据预处理的缓存策略",
                ],
            ))

    def _detect_elm_bottlenecks(self):
        """检测ELM的瓶颈"""
        elm_path = self.project_root / "engine/models/surrogates/elm.py"
        if not elm_path.exists():
            return

        content = elm_path.read_text()

        # 1. LSTSQ 求解器选择
        if "torch.linalg.lstsq" in content:
            self.bottlenecks.append(Bottleneck(
                location="ExtremeLearningMachine.fit",
                component="ExtremeLearningMachine",
                severity="low",
                bottleneck_type="solver",
                description="使用通用lstsq求解器，对于ELM场景可能非最优",
                estimated_overhead_percent=5.0,
                recommendations=[
                    "对于超定系统可使用 QR 分解更快",
                    "考虑使用 torch.lstsq 的 driver 参数优化",
                ],
            ))

        # 2. 无增量学习优化
        if "def update" in content and "np.vstack" in content:
            self.bottlenecks.append(Bottleneck(
                location="ExtremeLearningMachine.update",
                component="ExtremeLearningMachine",
                severity="medium",
                bottleneck_type="incremental_learning",
                description="增量更新使用重新训练，未利用历史权重信息",
                estimated_overhead_percent=20.0,
                recommendations=[
                    "实现真正的增量更新算法",
                    "使用递归最小二乘法(RLS)在线更新",
                ],
            ))

    def _detect_optimizer_bottlenecks(self):
        """检测优化器的瓶颈"""
        opt_base = self.project_root / "engine/optimization"

        # 1. 贝叶斯优化中的 GP 重训练
        bayesian_path = opt_base / "bayesian.py"
        if bayesian_path.exists():
            content = bayesian_path.read_text()

            if "self._fit_gp()" in content and "for iteration in range" in content:
                self.bottlenecks.append(Bottleneck(
                    location="BayesianOptimizer.minimize",
                    component="BayesianOptimizer",
                    severity="high",
                    bottleneck_type="algorithmic_complexity",
                    description="每次迭代都重新训练GP模型，复杂度为O(n_iter * n_obs^3)",
                    estimated_overhead_percent=60.0,
                    recommendations=[
                        "使用增量GP更新而非每次重训练",
                        "限制GP训练数据的最大数量",
                        "考虑使用更快的代理模型如随机森林",
                    ],
                ))

        # 2. LBFGS 无梯度检查
        lbfgs_path = opt_base / "lbfgs.py"
        if lbfgs_path.exists():
            content = lbfgs_path.read_text()

            if "method=" in content and "L-BFGS" in content:
                if "_check_differentiability" not in content:
                    self.bottlenecks.append(Bottleneck(
                        location="LBFGSOptimizer.minimize",
                        component="LBFGSOptimizer",
                        severity="low",
                        bottleneck_type="robustness",
                        description="未检查目标函数可微性，可能导致优化失败",
                        estimated_overhead_percent=5.0,
                        recommendations=[
                            "添加梯度可微性检查",
                            "提供数值梯度作为备选",
                        ],
                    ))

    def generate_report(self) -> str:
        """
        生成瓶颈分析报告

        Returns:
            Markdown格式的报告
        """
        if not self.bottlenecks:
            return "未检测到明显瓶颈"

        # 按严重程度排序
        severity_order = {"high": 0, "medium": 1, "low": 2}
        sorted_bottlenecks = sorted(
            self.bottlenecks,
            key=lambda b: severity_order.get(b.severity, 3)
        )

        lines = [
            "# 性能瓶颈分析报告",
            "",
            f"检测到 {len(self.bottlenecks)} 个瓶颈",
            "",
            "## 按严重程度排序",
            "",
        ]

        current_severity = None
        for b in sorted_bottlenecks:
            if b.severity != current_severity:
                current_severity = b.severity
                severity_emoji = {"high": "[CRITICAL]", "medium": "[WARNING]", "low": "[INFO]"}[current_severity]
                lines.append(f"### {severity_emoji} {current_severity.upper()} 严重度")
                lines.append("")

            lines.append(f"**位置**: {b.location}")
            lines.append(f"**类型**: {b.bottleneck_type}")
            lines.append(f"**描述**: {b.description}")
            lines.append(f"**预估开销**: {b.estimated_overhead_percent}%")
            lines.append("")
            lines.append("**优化建议**:")
            for rec in b.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # 汇总表
        lines.append("## 瓶颈汇总表")
        lines.append("")
        lines.append("| 位置 | 严重度 | 类型 | 开销估计 |")
        lines.append("|------|--------|------|----------|")
        for b in sorted_bottlenecks:
            lines.append(f"| {b.location} | {b.severity} | {b.bottleneck_type} | {b.estimated_overhead_percent}% |")

        return "\n".join(lines)

    def get_optimization_priority(self) -> List[Dict[str, Any]]:
        """
        获取优化优先级列表

        Returns:
            按优先级排序的优化任务列表
        """
        if not self.bottlenecks:
            return []

        sorted_bottlenecks = sorted(
            self.bottlenecks,
            key=lambda b: b.estimated_overhead_percent,
            reverse=True
        )

        return [
            {
                "location": b.location,
                "component": b.component,
                "priority_score": b.estimated_overhead_percent,
                "severity": b.severity,
                "top_recommendation": b.recommendations[0] if b.recommendations else "无",
            }
            for b in sorted_bottlenecks
        ]
