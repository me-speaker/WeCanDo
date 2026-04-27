"""
DeepInd 性能基准测试模块

提供代理模型和优化器的基准测试功能
"""

import time
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from functools import wraps


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    operation: str
    time_seconds: float
    memory_mb: Optional[float] = None
    n_samples: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "operation": self.operation,
            "time_seconds": self.time_seconds,
            "memory_mb": self.memory_mb,
            "n_samples": self.n_samples,
            "metadata": self.metadata,
        }


def measure_time(func: Callable) -> Callable:
    """装饰器：测量函数执行时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
    return wrapper


class BenchmarkRunner:
    """
    基准测试运行器

    用法:
        runner = BenchmarkRunner()
        runner.benchmark_surrogate("ELM", elm_model, X_train, y_train, X_test)
        runner.benchmark_optimizer("LBFGS", optimizer, objective, x0, bounds)
    """

    def __init__(self, n_warmup: int = 2, n_runs: int = 5):
        """
        Args:
            n_warmup: 热身运行次数
            n_runs: 正式测试运行次数
        """
        self.n_warmup = n_warmup
        self.n_runs = n_runs
        self.results: List[BenchmarkResult] = []

    def _get_memory_usage(self) -> Optional[float]:
        """获取当前内存使用（MB）"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return None

    def benchmark_surrogate(
        self,
        name: str,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        基准测试代理模型

        Args:
            name: 模型名称
            model: 代理模型实例
            X_train: 训练数据
            y_train: 训练标签
            X_test: 测试数据
            y_test: 测试标签（可选）

        Returns:
            包含各操作基准测试结果的字典
        """
        results = {}

        # 1. 测试 fit() 时间
        _, fit_time = self._benchmark_operation(
            model.fit, X_train, y_train, name=f"{name}.fit"
        )
        results["fit"] = BenchmarkResult(
            name=name,
            operation="fit",
            time_seconds=fit_time,
            n_samples=X_train.shape[0],
            metadata={"input_dim": X_train.shape[1]},
        )

        # 2. 测试 predict() 时间
        _, predict_time = self._benchmark_operation(
            model.predict, X_test, name=f"{name}.predict"
        )
        results["predict"] = BenchmarkResult(
            name=name,
            operation="predict",
            time_seconds=predict_time,
            n_samples=X_test.shape[0],
            metadata={"input_dim": X_test.shape[1]},
        )

        # 3. 测试预测精度（如果有标签）
        if y_test is not None:
            y_pred = model.predict(X_test)
            mse = np.mean((y_pred - y_test) ** 2)
            results["mse"] = BenchmarkResult(
                name=name,
                operation="mse",
                time_seconds=0.0,
                n_samples=X_test.shape[0],
                metadata={"mse": float(mse)},
            )

        # 4. 测试增量更新（如果支持）
        if hasattr(model, "update"):
            X_new = X_train[:min(10, len(X_train))]
            y_new = y_train[:min(10, len(y_train))]
            _, update_time = self._benchmark_operation(
                model.update, X_new, y_new, name=f"{name}.update"
            )
            results["update"] = BenchmarkResult(
                name=name,
                operation="update",
                time_seconds=update_time,
                n_samples=X_new.shape[0],
            )

        self.results.extend(results.values())
        return results

    def benchmark_optimizer(
        self,
        name: str,
        optimizer: Any,
        objective: Callable,
        x0: np.ndarray,
        bounds: List,
        constraints: Optional[List] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        基准测试优化器

        Args:
            name: 优化器名称
            optimizer: 优化器实例
            objective: 目标函数
            x0: 初始点
            bounds: 变量边界
            constraints: 约束条件

        Returns:
            包含优化结果的字典
        """
        results = {}

        # 测试 minimize() 时间
        opt_result, opt_time = self._benchmark_operation(
            optimizer.minimize, objective, x0.copy(), bounds, constraints,
            name=f"{name}.minimize"
        )

        results["optimize"] = BenchmarkResult(
            name=name,
            operation="minimize",
            time_seconds=opt_time,
            metadata={
                "success": opt_result.success,
                "n_iter": opt_result.n_iter,
                "final_fun": float(opt_result.fun) if hasattr(opt_result, "fun") else None,
            },
        )

        self.results.extend(results.values())
        return results

    def _benchmark_operation(
        self,
        func: Callable,
        *args,
        name: str = "",
        **kwargs,
    ) -> tuple:
        """执行基准测试操作"""
        # 热身
        for _ in range(self.n_warmup):
            func(*args, **kwargs)

        # 正式测试
        times = []
        for _ in range(self.n_runs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        return result, avg_time

    def compare_surrogates(
        self,
        models: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, BenchmarkResult]]:
        """
        比较多个代理模型的性能

        Args:
            models: 模型名称到实例的字典
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据

        Returns:
            各模型的基准测试结果
        """
        comparison = {}
        for name, model in models.items():
            try:
                results = self.benchmark_surrogate(
                    name, model, X_train, y_train, X_test, y_test
                )
                comparison[name] = results
            except Exception as e:
                comparison[name] = {"error": str(e)}

        return comparison

    def get_summary(self) -> Dict[str, Any]:
        """获取基准测试汇总"""
        if not self.results:
            return {"error": "No benchmark results available"}

        summary = {
            "total_tests": len(self.results),
            "by_operation": {},
        }

        for result in self.results:
            op = result.operation
            if op not in summary["by_operation"]:
                summary["by_operation"][op] = []
            summary["by_operation"][op].append(result.to_dict())

        return summary


def generate_test_data(
    n_samples: int = 1000,
    input_dim: int = 10,
    noise: float = 0.1,
    function: str = "sphere",
) -> tuple:
    """
    生成基准测试用的合成数据

    Args:
        n_samples: 样本数量
        input_dim: 输入维度
        noise: 噪声标准差
        function: 测试函数类型 ("sphere", "rosenbrock", "rastrigin")

    Returns:
        (X, y) 元组
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, input_dim)

    if function == "sphere":
        y = np.sum(X**2, axis=1)
    elif function == "rosenbrock":
        y = np.sum(100 * (X[:, 1:] - X[:, :-1]**2)**2 + (1 - X[:, :-1])**2, axis=1)
    elif function == "rastrigin":
        y = 10 * input_dim + np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=1)
    else:
        y = np.sum(X**2, axis=1)

    y += np.random.randn(n_samples) * noise

    return X, y
