"""
DeepInd 性能分析模块

提供性能基准测试、瓶颈识别和优化建议功能
"""

from .benchmark import BenchmarkRunner, BenchmarkResult, generate_test_data
from .profiler import SurrogateProfiler, OptimizerProfiler, ProfileResult
from .bottleneck_detector import BottleneckDetector, Bottleneck
from .memory_tracker import MemoryTracker, MemorySnapshot, track_memory

__all__ = [
    # Benchmark
    "BenchmarkRunner",
    "BenchmarkResult",
    "generate_test_data",
    # Profiler
    "SurrogateProfiler",
    "OptimizerProfiler",
    "ProfileResult",
    # Bottleneck
    "BottleneckDetector",
    "Bottleneck",
    # Memory
    "MemoryTracker",
    "MemorySnapshot",
    "track_memory",
]
