"""
DeepInd 特征工程模块 - 领域特征构造
用于构造化工配方相关的领域特征，如单体摩尔比等
"""

from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np

from .base import BaseFeatureTransformer, register_feature_transformer


@register_feature_transformer("MolarRatio")
class MolarRatioTransformer(BaseFeatureTransformer):
    """
    单体摩尔比特征构造器

    根据单体分子量和质量分数计算摩尔比
    摩尔比 = (质量分数 / 分子量) 之比
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - monomer_weights: 单体分子量字典 {"单体名": 分子量}
                - composition_col: 成分列名
                - output_prefix: 输出列前缀
        """
        super().__init__(config)
        self.monomer_weights = self.config.get("monomer_weights", {})
        self.composition_col = self.config.get("composition_col", "composition")
        self.output_prefix = self.config.get("output_prefix", "molar_ratio")

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "MolarRatioTransformer":
        """拟合（此转换器无需拟合）"""
        self._is_fitted = True

        # 验证分子量字典是否完整
        for col in X.columns:
            if col not in self.monomer_weights:
                self.logger.warning(f"列 {col} 未在 monomer_weights 中定义分子量")

        self.metadata = {
            "type": "MolarRatio",
            "monomer_weights": self.monomer_weights,
            "input_columns": list(X.columns),
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        计算摩尔比

        Args:
            X: 输入 DataFrame，列为单体质量分数

        Returns:
            包含摩尔比特征的 DataFrame
        """
        result = X.copy()

        # 获取有分子量信息的列
        valid_cols = [col for col in X.columns if col in self.monomer_weights]

        if len(valid_cols) < 2:
            self.logger.warning("摩尔比计算需要至少2个有分子量信息的单体")
            return result

        # 提取质量和分子量
        masses = X[valid_cols].values
        weights = np.array([self.monomer_weights[c] for c in valid_cols])

        # 计算摩尔数
        moles = masses / weights

        # 计算总摩尔数
        total_moles = moles.sum(axis=1, keepdims=True)
        total_moles[total_moles == 0] = 1.0  # 避免除零

        # 计算摩尔分数
        molar_fractions = moles / total_moles

        # 添加摩尔比列
        for i, col in enumerate(valid_cols):
            result[f"{self.output_prefix}_{col}"] = molar_fractions[:, i]

        # 添加总摩尔数特征
        result[f"{self.output_prefix}_total_moles"] = total_moles.flatten()

        self.metadata["output_columns"] = list(result.columns)
        return result


@register_feature_transformer("MolarRatioPairwise")
class PairwiseMolarRatioTransformer(BaseFeatureTransformer):
    """
    成对摩尔比特征构造器
    计算所有单体之间的摩尔比
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - monomer_weights: 单体分子量字典
        """
        super().__init__(config)
        self.monomer_weights = self.config.get("monomer_weights", {})

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "PairwiseMolarRatioTransformer":
        """拟合"""
        self._is_fitted = True
        self.metadata = {
            "type": "PairwiseMolarRatio",
            "monomer_weights": self.monomer_weights,
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """计算成对摩尔比"""
        result = X.copy()

        valid_cols = [col for col in X.columns if col in self.monomer_weights]

        if len(valid_cols) < 2:
            return result

        masses = X[valid_cols].values
        weights = np.array([self.monomer_weights[c] for c in valid_cols])
        moles = masses / weights

        # 计算成对摩尔比 (i/j)
        for i, col_i in enumerate(valid_cols):
            for j, col_j in enumerate(valid_cols):
                if i != j:
                    ratio = moles[:, i] / (moles[:, j] + 1e-10)
                    result[f"ratio_{col_i}_over_{col_j}"] = ratio

        self.metadata["output_columns"] = list(result.columns)
        return result


@register_feature_transformer("FunctionalGroupCount")
class FunctionalGroupCountTransformer(BaseFeatureTransformer):
    """
    官能团数量特征构造器
    根据单体中官能团的数量和单体含量计算总官能团数
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - functional_groups: 官能团字典 {"官能团名": {"单体名": 个数, ...}, ...}
                - output_name: 输出列名
        """
        super().__init__(config)
        self.functional_groups = self.config.get("functional_groups", {})
        self.output_name = self.config.get("output_name", "total_functional_groups")

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "FunctionalGroupCountTransformer":
        """拟合"""
        self._is_fitted = True
        self.metadata = {
            "type": "FunctionalGroupCount",
            "functional_groups": self.functional_groups,
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """计算官能团数量"""
        result = X.copy()

        # 计算每种官能团的总数量
        for group_name, monomer_counts in self.functional_groups.items():
            total_count = np.zeros(len(X))
            for monomer, count in monomer_counts.items():
                if monomer in X.columns:
                    total_count += X[monomer].values * count
            result[f"fg_{group_name}"] = total_count

        self.metadata["output_columns"] = list(result.columns)
        return result


@register_feature_transformer("CrosslinkingDensity")
class CrosslinkingDensityTransformer(BaseFeatureTransformer):
    """
    交联密度特征构造器
    估算聚合物的交联密度
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - functionality: 单体官能度字典 {"单体名": 官能度}
                - crosslinker_threshold: 交联点阈值
        """
        super().__init__(config)
        self.functionality = self.config.get("functionality", {})
        self.crosslinker_threshold = self.config.get("crosslinker_threshold", 2)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "CrosslinkingDensityTransformer":
        """拟合"""
        self._is_fitted = True
        self.metadata = {
            "type": "CrosslinkingDensity",
            "functionality": self.functionality,
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """计算交联密度特征"""
        result = X.copy()

        valid_cols = [col for col in X.columns if col in self.functionality]

        if len(valid_cols) == 0:
            return result

        # 计算平均官能度
        masses = X[valid_cols].values
        functionalities = np.array([self.functionality[c] for c in valid_cols])

        # 加权平均官能度
        total_mass = masses.sum(axis=1, keepdims=True) + 1e-10
        weighted_func = (masses * functionalities).sum(axis=1) / total_mass.flatten()

        result["avg_functionality"] = weighted_func

        # 计算交联点数量（假设交联单体官能度 > threshold）
        crosslinker_mask = functionalities > self.crosslinker_threshold
        if crosslinker_mask.any():
            crosslinker_masses = masses[:, crosslinker_mask]
            crosslinker_funcs = functionalities[crosslinker_mask]
            crosslinker_contribution = (crosslinker_masses * (crosslinker_funcs - 2)).sum(axis=1)
            result["crosslinking_sites"] = crosslinker_contribution

        self.metadata["output_columns"] = list(result.columns)
        return result


@register_feature_transformer("CustomFormula")
class CustomFormulaTransformer(BaseFeatureTransformer):
    """
    自定义公式特征构造器
    根据用户定义的公式计算新特征
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - formulas: 公式列表 [{"name": "新特征名", "expr": "表达式"}, ...]
                - variables: 变量列名映射
        """
        super().__init__(config)
        self.formulas = self.config.get("formulas", [])
        self.variables = self.config.get("variables", {})

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "CustomFormulaTransformer":
        """拟合"""
        self._is_fitted = True

        # 验证公式中的变量是否都存在
        all_vars = set()
        for formula in self.formulas:
            # 简单解析变量名（实际应用中应使用更安全的方法）
            import re
            vars_found = re.findall(r'[A-Za-z_]\w*', formula.get("expr", ""))
            all_vars.update(vars_found)

        self.metadata = {
            "type": "CustomFormula",
            "formulas": self.formulas,
            "variables": self.variables,
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """根据公式计算特征"""
        result = X.copy()

        for formula in self.formulas:
            name = formula["name"]
            expr = formula["expr"]

            # 创建评估上下文
            context = {col: X[col].values for col in X.columns}
            context.update(self.variables)

            try:
                # 安全评估（仅允许基本数学运算）
                result[name] = eval(expr, {"__builtins__": {}}, context)
            except Exception as e:
                self.logger.error(f"公式 '{name}' 计算失败: {e}")

        self.metadata["output_columns"] = list(result.columns)
        return result


@register_feature_transformer("GlassTransitionTemperature")
class GlassTransitionTemperatureTransformer(BaseFeatureTransformer):
    """
    玻璃化转变温度特征构造器
    使用 Fox 方程估算共聚物的玻璃化转变温度

    1/Tg = Σ(wi / Tgi)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - tg_values: 单体玻璃化转变温度字典 {"单体名": Tg值(K)}
                - output_name: 输出列名
        """
        super().__init__(config)
        self.tg_values = self.config.get("tg_values", {})
        self.output_name = self.config.get("output_name", "Tg_est")

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "GlassTransitionTemperatureTransformer":
        """拟合"""
        self._is_fitted = True
        self.metadata = {
            "type": "GlassTransitionTemperature",
            "tg_values": self.tg_values,
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """计算玻璃化转变温度"""
        result = X.copy()

        valid_cols = [col for col in X.columns if col in self.tg_values]

        if len(valid_cols) < 2:
            self.logger.warning("需要至少2个单体的 Tg 值来计算共聚物 Tg")
            return result

        # 获取质量和 Tg 值
        masses = X[valid_cols].values
        tg_array = np.array([self.tg_values[c] for c in valid_cols])

        # 归一化质量分数
        total_mass = masses.sum(axis=1, keepdims=True) + 1e-10
        weight_fractions = masses / total_mass

        # Fox 方程: 1/Tg = Σ(wi / Tgi)
        inverse_tg = (weight_fractions / tg_array).sum(axis=1)
        tg_est = 1.0 / (inverse_tg + 1e-10)

        result[self.output_name] = tg_est
        self.metadata["output_columns"] = list(result.columns)
        return result


@register_feature_transformer("SolubilityParameter")
class SolubilityParameterTransformer(BaseFeatureTransformer):
    """
    溶解度参数特征构造器
    使用 Hansen 溶解度参数估算溶剂相容性

    δ = Σ(δi * wi)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - delta_values: 溶解度参数字典 {"单体名": δ值}
                - output_name: 输出列名
        """
        super().__init__(config)
        self.delta_values = self.config.get("delta_values", {})
        self.output_name = self.config.get("output_name", "solubility_param")

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "SolubilityParameterTransformer":
        """拟合"""
        self._is_fitted = True
        self.metadata = {
            "type": "SolubilityParameter",
            "delta_values": self.delta_values,
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """计算溶解度参数"""
        result = X.copy()

        valid_cols = [col for col in X.columns if col in self.delta_values]

        if len(valid_cols) == 0:
            return result

        # 归一化
        masses = X[valid_cols].values
        total_mass = masses.sum(axis=1, keepdims=True) + 1e-10
        weight_fractions = masses / total_mass

        # 加权求和
        delta_array = np.array([self.delta_values[c] for c in valid_cols])
        solubility_param = (weight_fractions * delta_array).sum(axis=1)

        result[self.output_name] = solubility_param
        self.metadata["output_columns"] = list(result.columns)
        return result


@register_feature_transformer("PolydispersityIndex")
class PolydispersityIndexTransformer(BaseFeatureTransformer):
    """
    多分散指数特征构造器
    估算聚合物的分子量分布宽度

    PDI = Mw / Mn (通常需要 GPC 测量，此处基于单体类型估算)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - mw_values: 单体分子量字典
                - mw_distribution_factor: 分子量分布因子（默认 2.0）
        """
        super().__init__(config)
        self.mw_values = self.config.get("mw_values", {})
        self.mw_distribution_factor = self.config.get("mw_distribution_factor", 2.0)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "PolydispersityIndexTransformer":
        """拟合"""
        self._is_fitted = True
        self.metadata = {
            "type": "PolydispersityIndex",
            "mw_values": self.mw_values,
            "mw_distribution_factor": self.mw_distribution_factor,
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """估算多分散指数"""
        result = X.copy()

        valid_cols = [col for col in X.columns if col in self.mw_values]

        if len(valid_cols) == 0:
            return result

        # 计算数均分子量 (Mn)
        masses = X[valid_cols].values
        mw_array = np.array([self.mw_values[c] for c in valid_cols])
        total_mass = masses.sum(axis=1, keepdims=True) + 1e-10
        mole_fractions = masses / mw_array
        mn = 1.0 / (mole_fractions / mw_array).sum(axis=1)

        # 估算重均分子量 (Mw) - 使用分布因子
        mw_est = mn * self.mw_distribution_factor

        # PDI = Mw / Mn
        pdi = mw_est / (mn + 1e-10)

        result["Mn_est"] = mn
        result["Mw_est"] = mw_est
        result["PDI_est"] = pdi

        self.metadata["output_columns"] = list(result.columns)
        return result


@register_feature_transformer("ViscosityEstimate")
class ViscosityEstimateTransformer(BaseFeatureTransformer):
    """
    黏度估算特征构造器
    使用自由体积理论估算聚合物黏度

    ln(η) = A + B / (T - T0)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - viscosity_params: 黏度参数字典 {"单体名": {"A": float, "B": float, "T0": float}}
                - temperature_col: 温度列名（如果有）
        """
        super().__init__(config)
        self.viscosity_params = self.config.get("viscosity_params", {})
        self.temperature_col = self.config.get("temperature_col", None)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "ViscosityEstimateTransformer":
        """拟合"""
        self._is_fitted = True
        self.metadata = {
            "type": "ViscosityEstimate",
            "viscosity_params": self.viscosity_params,
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """估算黏度"""
        result = X.copy()

        if not self.viscosity_params:
            return result

        # 如果没有温度列，使用默认值 298K
        if self.temperature_col and self.temperature_col in X.columns:
            temperature = X[self.temperature_col].values
        else:
            temperature = np.full(len(X), 298.15)

        valid_cols = [col for col in X.columns if col in self.viscosity_params]

        if len(valid_cols) == 0:
            return result

        masses = X[valid_cols].values
        total_mass = masses.sum(axis=1, keepdims=True) + 1e-10
        weight_fractions = masses / total_mass

        # 加权平均黏度参数
        A_avg = sum(weight_fractions[:, i].mean() * self.viscosity_params[c].get("A", 0)
                    for i, c in enumerate(valid_cols))
        B_avg = sum(weight_fractions[:, i].mean() * self.viscosity_params[c].get("B", 0)
                    for i, c in enumerate(valid_cols))
        T0_avg = sum(weight_fractions[:, i].mean() * self.viscosity_params[c].get("T0", 0)
                     for i, c in enumerate(valid_cols))

        # 使用 VTF 方程估算黏度
        viscosity = np.exp(A_avg + B_avg / (temperature - T0_avg + 1e-10))

        result["viscosity_est"] = viscosity

        self.metadata["output_columns"] = list(result.columns)
        return result