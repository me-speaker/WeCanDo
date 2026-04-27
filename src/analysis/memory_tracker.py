"""
DeepInd 内存追踪模块

追踪代理模型和优化器的内存使用情况
"""

import gc
import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import torch


@dataclass
class MemorySnapshot:
    """内存快照"""
    label: str
    timestamp: float
    allocated_mb: float
    peak_mb: Optional[float] = None
    metadata: Dict[str, Any] = None


class MemoryTracker:
    """
    内存使用追踪器

    功能:
    - 追踪各组件的内存分配
    - 检测内存泄漏
    - 提供内存使用报告
    """

    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self._baseline_mb: Optional[float] = None
        self._peak_mb = 0.0

    def _get_memory_mb(self) -> float:
        """获取当前内存使用（MB）"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback: 使用 torch 内存
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
            return 0.0

    def start_tracking(self, label: str = "baseline"):
        """开始追踪，设置基线"""
        self._baseline_mb = self._get_memory_mb()
        self.snapshots.clear()
        gc.collect()
        self._peak_mb = self._baseline_mb
        self.take_snapshot(label)

    def take_snapshot(self, label: str, metadata: Dict[str, Any] = None) -> MemorySnapshot:
        """
        拍摄内存快照

        Args:
            label: 快照标签
            metadata: 额外元数据

        Returns:
            MemorySnapshot 对象
        """
        import time
        current_mb = self._get_memory_mb()
        self._peak_mb = max(self._peak_mb, current_mb)

        snapshot = MemorySnapshot(
            label=label,
            timestamp=time.time(),
            allocated_mb=current_mb,
            peak_mb=self._peak_mb,
            metadata=metadata or {},
        )
        self.snapshots.append(snapshot)
        return snapshot

    def get_delta(self, label: str = None) -> float:
        """
        获取与基线相比的内存增量

        Args:
            label: 可选的中间快照标签

        Returns:
            内存增量（MB）
        """
        if self._baseline_mb is None:
            return 0.0

        if label:
            for s in self.snapshots:
                if s.label == label:
                    return s.allocated_mb - self._baseline_mb

        return self.snapshots[-1].allocated_mb - self._baseline_mb if self.snapshots else 0.0

    def estimate_model_memory(self, model: Any) -> Dict[str, float]:
        """
        估算模型内存占用

        Args:
            model: 模型实例

        Returns:
            内存使用详情字典
        """
        memory_info = {"model_params_mb": 0.0, "buffers_mb": 0.0}

        # PyTorch 模型
        if hasattr(model, "_model") and isinstance(model._model, torch.nn.Module):
            total_params = sum(p.numel() for p in model._model.parameters())
            memory_info["model_params_mb"] = total_params * 4 / 1024 / 1024  # float32

            buffers = sum(b.numel() for b in model._model.buffers())
            memory_info["buffers_mb"] = buffers * 4 / 1024 / 1024

        # ELM 特殊处理
        if hasattr(model, "input_weight"):
            input_params = model.input_weight.numel()
            bias_params = model.bias.numel() if hasattr(model, "bias") else 0
            output_weight_params = model.output_weight.numel() if model.output_weight is not None else 0
            total = (input_params + bias_params + output_weight_params) * 4 / 1024 / 1024
            memory_info["model_params_mb"] = total

        # GP 核矩阵
        if hasattr(model, "_K") and model._K is not None:
            memory_info["kernel_matrix_mb"] = model._K.nbytes / 1024 / 1024

        # 训练数据
        if hasattr(model, "_X_train") and model._X_train is not None:
            memory_info["training_data_mb"] = model._X_train.nbytes / 1024 / 1024

        return memory_info

    def generate_report(self) -> str:
        """
        生成内存使用报告

        Returns:
            Markdown格式的报告
        """
        if not self.snapshots:
            return "无内存快照数据"

        lines = [
            "# 内存使用分析报告",
            "",
            f"快照数量: {len(self.snapshots)}",
            f"峰值内存: {self._peak_mb:.2f} MB",
            "",
            "## 内存变化",
            "",
            "| 标签 | 内存(MB) | 相对基线(MB) | 元数据 |",
            "|------|----------|-------------|--------|",
        ]

        for s in self.snapshots:
            delta = s.allocated_mb - self._baseline_mb if self._baseline_mb else 0
            metadata_str = ", ".join(f"{k}={v}" for k, v in s.metadata.items()) if s.metadata else "-"
            lines.append(f"| {s.label} | {s.allocated_mb:.2f} | {delta:+.2f} | {metadata_str} |")

        return "\n".join(lines)


@contextmanager
def track_memory(label: str = "operation", tracker: MemoryTracker = None):
    """
    上下文管理器：追踪代码块的内存使用

    用法:
        with track_memory("model.fit"):
            model.fit(X, y)
    """
    if tracker is None:
        tracker = MemoryTracker()

    tracker.take_snapshot(f"{label}_start")
    try:
        yield tracker
    finally:
        tracker.take_snapshot(f"{label}_end")

    import time
