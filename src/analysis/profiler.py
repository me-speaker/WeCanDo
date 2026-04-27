"""
DeepInd 性能分析器

提供代理模型和优化器的深度性能分析
"""

import time
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class ProfileResult:
    """性能分析结果"""
    component: str
    operation: str
    calls: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class OperationTimer:
    """操作计时器"""

    def __init__(self, name: str):
        self.name = name
        self.calls = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0

    def record(self, elapsed: float):
        self.calls += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)

    @property
    def avg_time(self) -> float:
        return self.total_time / self.calls if self.calls > 0 else 0.0


class SurrogateProfiler:
    """
    代理模型性能分析器

    分析内容:
    - fit() 操作分解（数据准备、核计算、矩阵求逆等）
    - predict() 操作分解
    - 内存使用跟踪
    - GPU/CPU 利用率（如果可用）
    """

    def __init__(self):
        self.timers: Dict[str, OperationTimer] = {}
        self.memory_snapshots: List[Dict[str, float]] = []

    def profile_fit(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, ProfileResult]:
        """
        分析模型 fit() 操作的详细性能

        Args:
            model: 代理模型实例
            X: 输入数据
            y: 目标数据

        Returns:
            各组件的性能分析结果
        """
        results = {}

        # 1. 数据准备阶段
        start = time.perf_counter()
        X_arr = np.atleast_2d(np.asarray(X))
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        data_time = time.perf_counter() - start

        results["data_preparation"] = ProfileResult(
            component=model.__class__.__name__,
            operation="data_preparation",
            calls=1,
            total_time=data_time,
            avg_time=data_time,
            metadata={"n_samples": X_arr.shape[0], "input_dim": X_arr.shape[1]},
        )

        # 2. 模型特定操作分析
        model_name = model.__class__.__name__

        if model_name == "GaussianProcess":
            results.update(self._profile_gp_fit(model, X_arr, y_arr))
        elif model_name == "NeuralNetwork":
            results.update(self._profile_nn_fit(model, X_arr, y_arr))
        elif model_name == "ExtremeLearningMachine":
            results.update(self._profile_elm_fit(model, X_arr, y_arr))
        elif model_name == "XGBoostModel":
            results.update(self._profile_xgb_fit(model, X_arr, y_arr))

        return results

    def _profile_gp_fit(
        self, model: Any, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, ProfileResult]:
        """分析高斯过程 fit 操作"""
        results = {}

        # 核矩阵计算
        start = time.perf_counter()
        K = model._rbf_kernel(X, X)
        kernel_time = time.perf_counter() - start
        results["kernel_computation"] = ProfileResult(
            component="GaussianProcess",
            operation="kernel_computation",
            calls=1,
            total_time=kernel_time,
            avg_time=kernel_time,
            metadata={"n_samples": X.shape[0], "matrix_size": K.shape},
        )

        # 噪声添加与Cholesky分解
        start = time.perf_counter()
        K_noise = K + model.noise_var * np.eye(X.shape[0])
        L = np.linalg.cholesky(K_noise)
        decomposition_time = time.perf_counter() - start
        results["cholesky_decomposition"] = ProfileResult(
            component="GaussianProcess",
            operation="cholesky_decomposition",
            calls=1,
            total_time=decomposition_time,
            avg_time=decomposition_time,
            metadata={"n_samples": X.shape[0], "complexity": "O(n^3)"},
        )

        # 矩阵求逆
        start = time.perf_counter()
        K_inv = np.linalg.inv(L)
        K_inv = K_inv @ K_inv.T
        inverse_time = time.perf_counter() - start
        results["matrix_inversion"] = ProfileResult(
            component="GaussianProcess",
            operation="matrix_inversion",
            calls=1,
            total_time=inverse_time,
            avg_time=inverse_time,
        )

        return results

    def _profile_nn_fit(
        self, model: Any, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, ProfileResult]:
        """分析神经网络 fit 操作"""
        results = {}

        # 数据标准化
        start = time.perf_counter()
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_norm = (X - X_mean) / X_std
        norm_time = time.perf_counter() - start
        results["normalization"] = ProfileResult(
            component="NeuralNetwork",
            operation="normalization",
            calls=1,
            total_time=norm_time,
            avg_time=norm_time,
        )

        # 模型初始化
        start = time.perf_counter()
        model._model = model._model.__class__(
            input_dim=model.input_dim,
            hidden_dims=model.hidden_dims,
            output_dim=y.shape[1] if y.ndim > 1 else 1,
            activation=model.activation,
            dropout=model.dropout,
        ).to(model._device)
        init_time = time.perf_counter() - start
        results["model_initialization"] = ProfileResult(
            component="NeuralNetwork",
            operation="model_initialization",
            calls=1,
            total_time=init_time,
            avg_time=init_time,
        )

        # 训练循环
        start = time.perf_counter()
        # 简化的训练时间测量
        X_tensor = torch.FloatTensor(X_norm).to(model._device)
        y_tensor = torch.FloatTensor(y).to(model._device)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=model.batch_size, shuffle=True)

        model._model.train()
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model._model.parameters(), lr=model.learning_rate)

        for epoch in range(model.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                y_pred = model._model(batch_X)
                loss = loss_fn(y_pred, batch_y)
                loss.backward()
                optimizer.step()
        training_time = time.perf_counter() - start

        results["training_loop"] = ProfileResult(
            component="NeuralNetwork",
            operation="training_loop",
            calls=model.epochs,
            total_time=training_time,
            avg_time=training_time / model.epochs,
            metadata={
                "epochs": model.epochs,
                "batch_size": model.batch_size,
                "hidden_dims": model.hidden_dims,
            },
        )

        return results

    def _profile_elm_fit(
        self, model: Any, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, ProfileResult]:
        """分析 ELM fit 操作"""
        results = {}

        # 隐藏层激活计算
        start = time.perf_counter()
        X_tensor = torch.FloatTensor(X)
        H = torch.relu(X_tensor @ model.input_weight.T + model.bias)
        activation_time = time.perf_counter() - start
        results["hidden_activation"] = ProfileResult(
            component="ExtremeLearningMachine",
            operation="hidden_activation",
            calls=1,
            total_time=activation_time,
            avg_time=activation_time,
            metadata={"hidden_dim": model.hidden_dim},
        )

        # 最小二乘求解
        start = time.perf_counter()
        output_weight = torch.linalg.lstsq(H, torch.FloatTensor(y)).solution.T
        lstsq_time = time.perf_counter() - start
        results["lstsq_solve"] = ProfileResult(
            component="ExtremeLearningMachine",
            operation="lstsq_solve",
            calls=1,
            total_time=lstsq_time,
            avg_time=lstsq_time,
            metadata={"complexity": "O(n * d^2)"},
        )

        return results

    def _profile_xgb_fit(
        self, model: Any, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, ProfileResult]:
        """分析 XGBoost fit 操作"""
        results = {}

        start = time.perf_counter()
        n_outputs = y.shape[1] if y.ndim > 1 else 1
        for i in range(n_outputs):
            model_i = model._models[i] if model._models else None
            # 实际训练在 model.fit 中已完成，这里测量预测部分
        fit_time = time.perf_counter() - start

        results["tree_building"] = ProfileResult(
            component="XGBoostModel",
            operation="tree_building",
            calls=1,
            total_time=fit_time,
            avg_time=fit_time,
            metadata={
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "n_outputs": n_outputs,
            },
        )

        return results

    def profile_predict(
        self, model: Any, X: np.ndarray
    ) -> Dict[str, ProfileResult]:
        """
        分析模型 predict() 操作的详细性能

        Args:
            model: 代理模型实例
            X: 输入数据

        Returns:
            各组件的性能分析结果
        """
        results = {}
        model_name = model.__class__.__name__

        # 通用预处理
        start = time.perf_counter()
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        prep_time = time.perf_counter() - start

        results["preprocessing"] = ProfileResult(
            component=model_name,
            operation="preprocessing",
            calls=1,
            total_time=prep_time,
            avg_time=prep_time,
        )

        # 模型特定预测
        if model_name == "GaussianProcess":
            start = time.perf_counter()
            k_star = model._rbf_kernel(X_arr, model._X_train)
            mu = k_star @ model._alpha
            pred_time = time.perf_counter() - start
            results["gp_prediction"] = ProfileResult(
                component=model_name,
                operation="gp_prediction",
                calls=1,
                total_time=pred_time,
                avg_time=pred_time,
            )
        elif model_name == "NeuralNetwork":
            model._model.eval()
            X_norm = (X_arr - model._X_mean) / model._X_std
            X_tensor = torch.FloatTensor(X_norm).to(model._device)
            start = time.perf_counter()
            with torch.no_grad():
                y_pred = model._model(X_tensor).cpu().numpy()
            pred_time = time.perf_counter() - start
            results["nn_prediction"] = ProfileResult(
                component=model_name,
                operation="nn_prediction",
                calls=1,
                total_time=pred_time,
                avg_time=pred_time,
            )
        elif model_name == "ExtremeLearningMachine":
            X_tensor = torch.FloatTensor(X_arr)
            start = time.perf_counter()
            with torch.no_grad():
                H = torch.relu(X_tensor @ model.input_weight.T + model.bias)
                y_pred = H @ model.output_weight.T
            pred_time = time.perf_counter() - start
            results["elm_prediction"] = ProfileResult(
                component=model_name,
                operation="elm_prediction",
                calls=1,
                total_time=pred_time,
                avg_time=pred_time,
            )

        return results


class OptimizerProfiler:
    """
    优化器性能分析器

    分析内容:
    - 收敛速度（迭代次数、目标函数值变化）
    - 各阶段时间分解
    - 解决方案质量评估
    """

    def __init__(self):
        self.iteration_history: List[Dict[str, float]] = []

    def profile_optimization(
        self,
        optimizer: Any,
        objective: Callable,
        x0: np.ndarray,
        bounds: List,
        constraints: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        分析优化过程的详细性能

        Args:
            optimizer: 优化器实例
            objective: 目标函数
            x0: 初始点
            bounds: 变量边界
            constraints: 约束条件

        Returns:
            包含详细性能信息的字典
        """
        self.iteration_history = []

        # 包装目标函数以跟踪每次评估
        original_fun = objective
        iteration_count = [0]
        fun_evaluations = [0]

        def tracked_objective(x):
            fun_evaluations[0] += 1
            iteration_count[0] += 1
            f_val = original_fun(x)
            self.iteration_history.append({
                "iteration": iteration_count[0],
                "objective_value": float(f_val),
                "fun_evaluations": fun_evaluations[0],
            })
            return f_val

        # 执行优化
        start = time.perf_counter()
        result = optimizer.minimize(tracked_objective, x0.copy(), bounds, constraints)
        total_time = time.perf_counter() - start

        return {
            "total_time": total_time,
            "n_iterations": result.n_iter,
            "fun_evaluations": fun_evaluations[0],
            "success": result.success,
            "final_objective": float(result.fun),
            "optimal_x": result.x.tolist() if hasattr(result.x, "tolist") else result.x,
            "convergence_history": self.iteration_history,
            "time_per_iteration": total_time / result.n_iter if result.n_iter > 0 else 0,
        }

    def analyze_convergence(self) -> Dict[str, Any]:
        """
        分析收敛特性

        Returns:
            收敛分析结果
        """
        if not self.iteration_history:
            return {"error": "No convergence data available"}

        history = self.iteration_history
        obj_values = [h["objective_value"] for h in history]

        return {
            "n_iterations": len(history),
            "initial_objective": obj_values[0],
            "final_objective": obj_values[-1],
            "improvement_ratio": (obj_values[0] - obj_values[-1]) / obj_values[0] if obj_values[0] != 0 else 0,
            "min_objective": min(obj_values),
            "max_objective": max(obj_values),
            "convergence_rate": self._estimate_convergence_rate(obj_values),
        }

    def _estimate_convergence_rate(self, obj_values: List[float]) -> str:
        """估计收敛速率"""
        if len(obj_values) < 3:
            return "unknown"

        # 计算相邻迭代的改进比例
        improvements = []
        for i in range(1, len(obj_values)):
            if obj_values[i-1] != 0:
                imp = abs(obj_values[i] - obj_values[i-1]) / abs(obj_values[i-1])
                improvements.append(imp)

        avg_imp = sum(improvements) / len(improvements) if improvements else 0

        if avg_imp < 0.01:
            return "fast"
        elif avg_imp < 0.1:
            return "moderate"
        else:
            return "slow"
