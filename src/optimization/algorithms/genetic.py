"""
DeepInd 遗传算法优化器
适用于不可微目标函数和全局优化
"""

import numpy as np
from typing import List, Callable, Optional, Tuple

from ..base import BaseOptimizer, OptimizationResult


class GeneticOptimizer(BaseOptimizer):
    """
    遗传算法 (Genetic Algorithm) 优化器

    特点:
    - 适用于不可微目标函数
    - 全局搜索能力强
    - 适合多模态优化问题

    Args:
        n_iter: 最大迭代次数
        pop_size: 种群大小
        mutation_rate: 变异率
        crossover_rate: 交叉率
        elite_ratio: 精英保留比例
        verbose: 是否输出日志
    """

    def __init__(
        self,
        n_iter: int = 100,
        pop_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_ratio: float = 0.1,
        verbose: bool = True,
    ):
        super().__init__(n_iter=n_iter, tol=0.0, verbose=verbose)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio

    def minimize(
        self,
        fun: Callable,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        constraints: Optional[List[dict]] = None,
    ) -> OptimizationResult:
        """
        执行遗传算法优化

        Args:
            fun: 目标函数
            x0: 初始点（用于初始化种群）
            bounds: 变量边界
            constraints: 约束条件（通过惩罚法处理）

        Returns:
            OptimizationResult
        """
        n_vars = len(bounds)
        bounds_array = np.array(bounds)

        if self.verbose:
            print(f"遗传算法: 开始优化, 种群大小={self.pop_size}, 迭代={self.n_iter}")

        # 初始化种群
        population = self._init_population(x0, n_vars, bounds_array)
        fitness = self._evaluate_population(population, fun, constraints)

        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_y = fitness[best_idx]
        all_solutions = [best_x.copy()]

        # 主循环
        for gen in range(self.n_iter):
            # 选择
            mating_pool = self._selection(population, fitness)

            # 交叉
            offspring = self._crossover(mating_pool, bounds_array)

            # 变异
            offspring = self._mutation(offspring, bounds_array)

            # 评估
            offspring_fitness = self._evaluate_population(offspring, fun, constraints)

            # 合并种群
            population = offspring
            fitness = offspring_fitness

            # 更新最优
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_y:
                best_y = fitness[best_idx]
                best_x = population[best_idx].copy()

            all_solutions.append(best_x.copy())

            if (gen + 1) % 10 == 0 and self.verbose:
                print(f"  迭代 {gen+1}/{self.n_iter}: best_f={best_y:.4f}")

        self._n_iterations = self.n_iter

        return OptimizationResult(
            x=best_x,
            fun=best_y,
            success=True,
            message="优化完成",
            n_iter=self.n_iter,
            all_solutions=all_solutions,
        )

    def minimize_multi(
        self,
        objectives: List[Callable],
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        constraints: Optional[List[dict]] = None,
    ) -> OptimizationResult:
        """
        多目标遗传算法：使用 NSGA-II 风格方法

        Args:
            objectives: 目标函数列表
            x0: 初始点
            bounds: 变量边界
            constraints: 约束条件

        Returns:
            OptimizationResult
        """
        n_vars = len(bounds)
        bounds_array = np.array(bounds)
        n_obj = len(objectives)

        if self.verbose:
            print(f"遗传算法: 多目标优化, {n_obj} 个目标")

        # 初始化种群
        population = self._init_population(x0, n_vars, bounds_array)
        fitness = self._evaluate_population_multi(population, objectives, constraints)

        # 非支配排序
        fronts = self._non_dominated_sort(fitness)
        crowding = self._calc_crowding_distance(fronts, fitness, n_obj)

        best_x = population[fronts[0][0]].copy()
        best_y = fitness[fronts[0][0]]
        all_solutions = [best_x.copy()]

        # 主循环
        for gen in range(self.n_iter):
            # 选择 (基于等级和拥挤度)
            mating_pool = self._multi_objective_selection(population, fronts, crowding)

            # 交叉
            offspring = self._crossover(mating_pool, bounds_array)

            # 变异
            offspring = self._mutation(offspring, bounds_array)

            # 评估
            offspring_fitness = self._evaluate_population_multi(offspring, objectives, constraints)

            # 合并
            combined_pop = np.vstack([population, offspring])
            combined_fit = np.vstack([fitness, offspring_fitness])

            # 快速非支配排序
            fronts = self._non_dominated_sort(combined_fit)
            crowding = self._calc_crowding_distance(fronts, combined_fit, n_obj)

            # 选择下一代
            population, fitness = self._select_next_generation(
                combined_pop, combined_fit, fronts, crowding
            )

            # 更新最优（第一个前沿的第一个，需要映射到新种群的索引）
            # fronts中的索引是针对combined_pop的，需要找到第一个前沿在selection后的population中的对应个体
            if fronts and len(fronts[0]) > 0:
                first_front_idx = fronts[0][0]
                # 在selection后的population中找第一个前沿的个体
                for i, idx in enumerate(fronts[0]):
                    if idx < len(population):
                        best_x = population[i].copy()
                        best_y = fitness[i]
                        break
                else:
                    # 如果没找到，使用当前最优
                    best_idx = np.argmin(fitness)
                    best_x = population[best_idx].copy()
                    best_y = fitness[best_idx]
            else:
                best_idx = np.argmin(fitness)
                best_x = population[best_idx].copy()
                best_y = fitness[best_idx]
            all_solutions.append(best_x.copy())

            if (gen + 1) % 10 == 0 and self.verbose:
                pareto_size = len(fronts[0]) if fronts else 0
                print(f"  迭代 {gen+1}/{self.n_iter}: Pareto前沿大小={pareto_size}")

        self._n_iterations = self.n_iter

        return OptimizationResult(
            x=best_x,
            fun=best_y,
            success=True,
            message="多目标优化完成",
            n_iter=self.n_iter,
            all_solutions=all_solutions,
        )

    def _init_population(
        self,
        x0: np.ndarray,
        n_vars: int,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """初始化种群"""
        pop = np.zeros((self.pop_size, n_vars))
        # 用 x0 初始化部分种群
        n_from_x0 = max(int(self.pop_size * 0.1), 1)
        for i in range(n_from_x0):
            pop[i] = x0 + np.random.randn(n_vars) * 0.1 * (bounds[:, 1] - bounds[:, 0])
        # 随机初始化其余
        for i in range(n_from_x0, self.pop_size):
            for j in range(n_vars):
                pop[i, j] = np.random.uniform(bounds[j, 0], bounds[j, 1])
        return np.clip(pop, bounds[:, 0], bounds[:, 1])

    def _evaluate_population(
        self,
        population: np.ndarray,
        fun: Callable,
        constraints: Optional[List[dict]],
    ) -> np.ndarray:
        """评估种群适应度"""
        fitness = np.zeros(len(population))
        for i, x in enumerate(population):
            try:
                fitness[i] = fun(x)
                if constraints:
                    fitness[i] += self._penalty(x, constraints)
            except Exception:
                fitness[i] = 1e10
        return fitness

    def _evaluate_population_multi(
        self,
        population: np.ndarray,
        objectives: List[Callable],
        constraints: Optional[List[dict]],
    ) -> np.ndarray:
        """评估多目标种群"""
        n_obj = len(objectives)
        fitness = np.zeros((len(population), n_obj))
        for i, x in enumerate(population):
            for j, obj in enumerate(objectives):
                try:
                    fitness[i, j] = obj(x)
                    if constraints:
                        fitness[i, j] += self._penalty(x, constraints)
                except Exception:
                    fitness[i, j] = 1e10
        return fitness

    def _penalty(self, x: np.ndarray, constraints: List[dict]) -> float:
        """计算惩罚值"""
        penalty = 0.0
        for constraint in constraints:
            violation = self._constraint_violation(x, constraint)
            penalty += violation ** 2
        return penalty

    def _constraint_violation(self, x: np.ndarray, constraint: dict) -> float:
        """计算约束违反量"""
        expr = constraint.get("expression", "0")
        try:
            local_vars = {f"x[{i}]": val for i, val in enumerate(x)}
            value = eval(expr, {"__builtins__": {}}, local_vars)
        except Exception:
            return 0.0

        ctype = constraint.get("type", "le")
        if ctype == "eq":
            return abs(value)
        else:
            return max(0, value)

    def _selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """锦标赛选择"""
        mating_pool = []
        pool_size = len(population)

        for _ in range(pool_size):
            idx1, idx2 = np.random.choice(pool_size, 2, replace=False)
            winner = idx1 if fitness[idx1] < fitness[idx2] else idx2
            mating_pool.append(population[winner].copy())

        return np.array(mating_pool)

    def _crossover(self, mating_pool: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """模拟二进制交叉 (SBX)"""
        n_offspring = len(mating_pool)
        n_vars = mating_pool.shape[1]
        offspring = np.zeros_like(mating_pool)
        eta = 15  # 分布指数

        for i in range(0, n_offspring, 2):
            if i + 1 >= n_offspring:
                offspring[i] = mating_pool[i]
                continue

            parent1, parent2 = mating_pool[i], mating_pool[i + 1]

            if np.random.rand() < self.crossover_rate:
                for j in range(n_vars):
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

                    low, high = bounds[j]
                    offspring[i, j] = 0.5 * ((1 + beta) * parent1[j] + (1 - beta) * parent2[j])
                    offspring[i + 1, j] = 0.5 * ((1 - beta) * parent1[j] + (1 + beta) * parent2[j])
            else:
                offspring[i] = parent1.copy()
                offspring[i + 1] = parent2.copy()

            # 边界约束
            offspring[i] = np.clip(offspring[i], bounds[:, 0], bounds[:, 1])
            offspring[i + 1] = np.clip(offspring[i + 1], bounds[:, 0], bounds[:, 1])

        return offspring

    def _mutation(self, population: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """多项式变异"""
        n_vars = population.shape[1]
        eta = 20
        pop_range = bounds[:, 1] - bounds[:, 0]

        for i in range(len(population)):
            for j in range(n_vars):
                if np.random.rand() < self.mutation_rate:
                    x = population[i, j]
                    delta_l = (x - bounds[j, 0]) / pop_range[j] if pop_range[j] > 0 else 0
                    delta_r = (bounds[j, 1] - x) / pop_range[j] if pop_range[j] > 0 else 0

                    u = np.random.rand()
                    if u < 0.5:
                        delta_q = (2 * u + (1 - 2 * u) * (1 - delta_l) ** (eta + 1)) ** (1 / (eta + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta_r) ** (eta + 1)) ** (1 / (eta + 1))

                    population[i, j] = x + delta_q * pop_range[j]
                    population[i, j] = np.clip(population[i, j], bounds[j, 0], bounds[j, 1])

        return population

    def _non_dominated_sort(self, fitness: np.ndarray) -> List[List[int]]:
        """快速非支配排序"""
        n = len(fitness)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        fronts = [[] for _ in range(n)]

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(fitness[p] <= fitness[q]) and np.any(fitness[p] < fitness[q]):
                    dominated_set[p].append(q)
                elif np.all(fitness[q] <= fitness[p]) and np.any(fitness[q] < fitness[p]):
                    domination_count[p] += 1

            if domination_count[p] == 0:
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_set[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts[i] = next_front

        return [f for f in fronts if f]

    def _calc_crowding_distance(
        self,
        fronts: List[List[int]],
        fitness: np.ndarray,
        n_obj: int,
    ) -> np.ndarray:
        """计算拥挤度距离"""
        n = len(fitness)
        distances = np.zeros(n)

        for front in fronts:
            if len(front) <= 2:
                distances[front] = np.inf
                continue

            for obj in range(n_obj):
                obj_values = fitness[front, obj]
                sorted_idx = np.argsort(obj_values)

                distances[front[sorted_idx[0]]] = np.inf
                distances[front[sorted_idx[-1]]] = np.inf

                obj_range = obj_values.max() - obj_values.min()
                if obj_range == 0:
                    continue

                for i in range(1, len(front) - 1):
                    idx = front[sorted_idx[i]]
                    distances[idx] += (obj_values[sorted_idx[i + 1]] - obj_values[sorted_idx[i - 1]]) / obj_range

        return distances

    def _multi_objective_selection(
        self,
        population: np.ndarray,
        fronts: List[List[int]],
        crowding: np.ndarray,
    ) -> np.ndarray:
        """基于等级和拥挤度的选择"""
        mating_pool = []
        pool_size = len(population)

        while len(mating_pool) < pool_size:
            idx1, idx2 = np.random.choice(len(population), 2, replace=False)

            rank1 = self._get_rank(idx1, fronts)
            rank2 = self._get_rank(idx2, fronts)

            if rank1 < rank2:
                selected = idx1
            elif rank1 > rank2:
                selected = idx2
            else:
                selected = idx1 if crowding[idx1] > crowding[idx2] else idx2

            mating_pool.append(population[selected].copy())

        return np.array(mating_pool)

    def _get_rank(self, idx: int, fronts: List[List[int]]) -> int:
        """获取个体等级"""
        for rank, front in enumerate(fronts):
            if idx in front:
                return rank
        return len(fronts)

    def _select_next_generation(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        fronts: List[List[int]],
        crowding: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """选择下一代"""
        n_to_select = self.pop_size
        new_pop = []
        new_fitness = []

        for front in fronts:
            if len(new_pop) + len(front) <= n_to_select:
                for idx in front:
                    new_pop.append(population[idx])
                    new_fitness.append(fitness[idx])
            else:
                remaining = n_to_select - len(new_pop)
                sorted_front = sorted(front, key=lambda x: crowding[x], reverse=True)
                for idx in sorted_front[:remaining]:
                    new_pop.append(population[idx])
                    new_fitness.append(fitness[idx])
                break

        return np.array(new_pop), np.array(new_fitness)
