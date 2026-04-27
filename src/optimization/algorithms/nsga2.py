"""
DeepInd NSGA-II (Non-dominated Sorting Genetic Algorithm II)
多目标遗传算法实现
"""

import numpy as np
from typing import List, Callable, Optional

from ...core.base import BaseOptimizer
from ...core.registry import register_optimizer
from ...utils.logger import get_logger


@register_optimizer("NSGA2")
class NSGA2Optimizer(BaseOptimizer):
    """
    NSGA-II 多目标遗传算法

    主要步骤:
    1. 快速非支配排序
    2. 拥挤度距离计算
    3. 选择、交叉、变异

    Args:
        n_generations: 迭代代数
        pop_size: 种群大小
        mutation_rate: 变异率
        crossover_rate: 交叉率
    """

    def __init__(
        self,
        n_generations: int = 100,
        pop_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.9,
    ):
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.logger = get_logger("DeepInd::NSGA2")
        self.objectives = None

    def optimize(
        self,
        objectives: List[Callable],
        bounds: List[tuple],
        constraints: Optional[List[dict]] = None
    ) -> np.ndarray:
        """
        执行 NSGA-II 优化

        Args:
            objectives: 目标函数列表
            bounds: 决策变量边界，[(low, high), ...]
            constraints: 约束条件列表（暂未实现）

        Returns:
            Pareto 前沿解集，shape (n_solutions, n_vars + n_objectives)
        """
        self.objectives = objectives
        n_vars = len(bounds)
        n_objectives = len(objectives)

        self.logger.info(
            f"NSGA2: 开始优化, 种群大小={self.pop_size}, 代数={self.n_generations}"
        )
        self.logger.debug(f"  - 决策变量数: {n_vars}")
        self.logger.debug(f"  - 目标函数数: {n_objectives}")

        # 初始化种群
        population = self._init_population(n_vars, bounds)
        self.logger.debug(f"  - 种群初始化完成")

        # 主循环
        for gen in range(self.n_generations):
            # 评估适应度
            fitness = self._evaluate(population, objectives)

            # 快速非支配排序
            fronts = self._fast_non_dominated_sort(fitness)

            # 计算拥挤度距离
            crowding_distances = self._calc_crowding_distance(fronts, fitness, n_objectives)

            # 选择
            mating_pool = self._select(population, fronts, crowding_distances)

            # 交叉
            offspring = self._crossover(mating_pool, bounds)

            # 变异
            offspring = self._mutate(offspring, bounds)

            # 合并父代和子代
            population = offspring

            # 日志输出
            if (gen + 1) % 10 == 0 or gen == 0:
                best_fitness = self._get_best_fitness(fitness)
                self.logger.info(
                    f"NSGA2: Generation {gen+1}/{self.n_generations} 完成, "
                    f"最优目标值: {best_fitness}"
                )

        # 最终评估
        final_fitness = self._evaluate(population, objectives)
        fronts = self._fast_non_dominated_sort(final_fitness)

        # 提取 Pareto 前沿
        pareto_front = self._extract_pareto_front(population, final_fitness, fronts)

        self.logger.info(f"NSGA2: 优化完成, 找到 {len(pareto_front)} 个 Pareto 解")

        return pareto_front

    def _init_population(self, n_vars: int, bounds: List[tuple]) -> np.ndarray:
        """初始化种群"""
        population = np.zeros((self.pop_size, n_vars))
        for i, (low, high) in enumerate(bounds):
            population[:, i] = np.random.uniform(low, high, self.pop_size)
        return population

    def _evaluate(self, population: np.ndarray, objectives: List[Callable]) -> np.ndarray:
        """评估种群适应度"""
        n_objectives = len(objectives)
        fitness = np.zeros((len(population), n_objectives))

        for i, ind in enumerate(population):
            for j, obj in enumerate(objectives):
                try:
                    fitness[i, j] = obj(ind)
                except Exception:
                    fitness[i, j] = 1e10  # 惩罚值

        return fitness

    def _fast_non_dominated_sort(self, fitness: np.ndarray) -> List[List[int]]:
        """
        快速非支配排序

        Args:
            fitness: 适应度，shape (pop_size, n_objectives)

        Returns:
            fronts: 非支配解前沿列表
        """
        n = len(fitness)
        domination_count = np.zeros(n, dtype=int)  # 支配该个体的数量
        dominated_set = [[] for _ in range(n)]  # 该个体支配的集合
        fronts = [[] for _ in range(n)]  # 各前沿

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                # 检查 p 是否支配 q
                if self._dominates(fitness[p], fitness[q]):
                    dominated_set[p].append(q)
                # 检查 q 是否支配 p
                elif self._dominates(fitness[q], fitness[p]):
                    domination_count[p] += 1

            if domination_count[p] == 0:
                fronts[0].append(p)

        # 构建前沿
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

        # 移除空的前沿
        fronts = [f for f in fronts if f]

        return fronts

    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """
        判断 a 是否支配 b（多目标情况下）

        Returns:
            True if a dominates b (a 在所有目标上都不差于 b，且至少在一个目标上严格优于 b)
        """
        # 最小化问题
        return np.all(a <= b) and np.any(a < b)

    def _calc_crowding_distance(
        self,
        fronts: List[List[int]],
        fitness: np.ndarray,
        n_objectives: int
    ) -> np.ndarray:
        """
        计算拥挤度距离

        Args:
            fronts: 非支配解前沿
            fitness: 适应度
            n_objectives: 目标数量

        Returns:
            拥挤度距离数组
        """
        n = len(fitness)
        distances = np.zeros(n)

        for front in fronts:
            if len(front) <= 2:
                distances[front] = np.inf
                continue

            for obj in range(n_objectives):
                obj_values = fitness[front, obj]
                sorted_idx = np.argsort(obj_values)

                # 边界个体距离设为无穷
                distances[front[sorted_idx[0]]] = np.inf
                distances[front[sorted_idx[-1]]] = np.inf

                # 计算其他个体的距离
                obj_range = obj_values.max() - obj_values.min()
                if obj_range == 0:
                    continue

                for i in range(1, len(front) - 1):
                    idx = front[sorted_idx[i]]
                    distances[idx] += (
                        obj_values[sorted_idx[i + 1]] - obj_values[sorted_idx[i - 1]]
                    ) / obj_range

        return distances

    def _select(
        self,
        population: np.ndarray,
        fronts: List[List[int]],
        crowding_distances: np.ndarray
    ) -> np.ndarray:
        """
        基于竞标赛选择

        Args:
            population: 种群
            fronts: 非支配解前沿
            crowding_distances: 拥挤度距离

        Returns:
            选择的个体
        """
        mating_pool = []
        pool_size = len(population)

        while len(mating_pool) < pool_size:
            # 随机选择两个个体
            idx1, idx2 = np.random.choice(len(population), 2, replace=False)

            # 比较：优先选择前沿排名更靠前的，否则选择拥挤度距离更大的
            rank1 = self._get_rank(idx1, fronts)
            rank2 = self._get_rank(idx2, fronts)

            if rank1 < rank2:
                selected = idx1
            elif rank1 > rank2:
                selected = idx2
            else:
                selected = idx1 if crowding_distances[idx1] > crowding_distances[idx2] else idx2

            mating_pool.append(population[selected].copy())

        return np.array(mating_pool)

    def _get_rank(self, idx: int, fronts: List[List[int]]) -> int:
        """获取个体所属的前沿编号"""
        for rank, front in enumerate(fronts):
            if idx in front:
                return rank
        return len(fronts)  # 被支配的个体

    def _crossover(self, mating_pool: np.ndarray, bounds: List[tuple]) -> np.ndarray:
        """
        Simulated Binary Crossover (SBX)

        Args:
            mating_pool: 交配池
            bounds: 边界

        Returns:
            交叉后的子代
        """
        n_offspring = len(mating_pool)
        n_vars = mating_pool.shape[1]
        offspring = np.zeros_like(mating_pool)

        for i in range(0, n_offspring, 2):
            if i + 1 >= n_offspring:
                offspring[i] = mating_pool[i]
                continue

            parent1 = mating_pool[i]
            parent2 = mating_pool[i + 1]

            if np.random.rand() < self.crossover_rate:
                # SBX 交叉
                for j in range(n_vars):
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / 20)  # eta = 20
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / 20)

                    low, high = bounds[j]
                    offspring[i, j] = 0.5 * (
                        (1 + beta) * parent1[j] + (1 - beta) * parent2[j]
                    )
                    offspring[i + 1, j] = 0.5 * (
                        (1 - beta) * parent1[j] + (1 + beta) * parent2[j]
                    )

                    # 边界约束
                    offspring[i, j] = np.clip(offspring[i, j], low, high)
                    offspring[i + 1, j] = np.clip(offspring[i + 1, j], low, high)
            else:
                offspring[i] = parent1.copy()
                offspring[i + 1] = parent2.copy()

        return offspring

    def _mutate(self, population: np.ndarray, bounds: List[tuple]) -> np.ndarray:
        """
        多项式变异

        Args:
            population: 种群
            bounds: 边界

        Returns:
            变异后的种群
        """
        n_vars = population.shape[1]
        eta = 20  # 变异指数

        for i in range(len(population)):
            for j in range(n_vars):
                if np.random.rand() < self.mutation_rate:
                    low, high = bounds[j]
                    x = population[i, j]
                    delta_l = (x - low) / (high - low) if high > low else 0
                    delta_r = (high - x) / (high - low) if high > low else 0

                    u = np.random.rand()

                    if u < 0.5:
                        delta_q = (2 * u + (1 - 2 * u) * (1 - delta_l) ** (eta + 1)) ** (
                            1 / (eta + 1)
                        ) - 1
                    else:
                        delta_q = 1 - (
                            2 * (1 - u) + 2 * (u - 0.5) * (1 - delta_r) ** (eta + 1)
                        ) ** (1 / (eta + 1))

                    population[i, j] = x + delta_q * (high - low)
                    population[i, j] = np.clip(population[i, j], low, high)

        return population

    def _extract_pareto_front(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        fronts: List[List[int]]
    ) -> np.ndarray:
        """
        提取 Pareto 前沿

        Returns:
            包含决策变量和目标值的数组
        """
        if not fronts:
            return np.hstack([population, fitness])

        pareto_population = population[fronts[0]]
        pareto_fitness = fitness[fronts[0]]

        return np.hstack([pareto_population, pareto_fitness])

    def _get_best_fitness(self, fitness: np.ndarray) -> np.ndarray:
        """获取各目标的最优值"""
        return fitness.min(axis=0)
