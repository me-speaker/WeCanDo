"""
DeepInd 特征工程模块 - 特征选择
提供基于重要性、相关性等方法的特征选择功能
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_regression,
    f_classif,
    mutual_info_regression,
    mutual_info_classif,
    RFE,
    SelectFromModel,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV

from .base import BaseFeatureTransformer, register_feature_transformer


@register_feature_transformer("VarianceThreshold")
class VarianceThresholdSelector(BaseFeatureTransformer):
    """
    方差阈值特征选择器
    移除方差低于阈值的特征
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - threshold: 方差阈值，默认 0.0
        """
        super().__init__(config)
        self.threshold = self.config.get("threshold", 0.0)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "VarianceThresholdSelector":
        """拟合"""
        self.selector_ = VarianceThreshold(threshold=self.threshold)
        self.selector_.fit(X.values)

        # 记录选中的特征
        self.selected_mask_ = self.selector_.get_support()
        self.selected_features_ = list(X.columns[self.selected_mask_])

        self.metadata = {
            "type": "VarianceThreshold",
            "threshold": self.threshold,
            "selected_features": self.selected_features_,
            "n_selected": len(self.selected_features_),
            "n_dropped": len(self.selected_features_) - sum(self.selected_mask_),
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """选择特征"""
        return X[self.selected_features_]


@register_feature_transformer("KBest")
class SelectKBestTransformer(BaseFeatureTransformer):
    """
    K最佳特征选择器
    选择与目标变量相关性最高的K个特征
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - k: 选择特征数量，默认 "all" 或具体数值
                - score_func: 评分函数
        """
        super().__init__(config)
        self.k = self.config.get("k", "all")
        score_func_name = self.config.get("score_func", "f_regression")
        self.score_func_map = {
            "f_regression": f_regression,
            "f_classif": f_classif,
            "mutual_info_regression": mutual_info_regression,
            "mutual_info_classif": mutual_info_classif,
        }
        self.score_func = self.score_func_map.get(score_func_name, f_regression)

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series]] = None) -> "SelectKBestTransformer":
        """拟合"""
        if y is None:
            raise ValueError("SelectKBest 需要目标变量 y")

        # 转换y为1D
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y_values = y.values.ravel()
        else:
            y_values = y

        # 确定k值
        n_features = X.shape[1]
        if self.k == "all" or self.k > n_features:
            k = n_features
        else:
            k = self.k

        self.selector_ = SelectKBest(score_func=self.score_func, k=k)
        self.selector_.fit(X.values, y_values)

        # 记录选中的特征
        self.selected_mask_ = self.selector_.get_support()
        self.selected_features_ = list(X.columns[self.selected_mask_])

        # 获取分数
        scores = self.selector_.scores_
        self.feature_scores_ = dict(zip(X.columns, scores))

        self.metadata = {
            "type": "SelectKBest",
            "k": k,
            "score_func": self.score_func.__name__,
            "selected_features": self.selected_features_,
            "feature_scores": self.feature_scores_,
            "n_selected": len(self.selected_features_),
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """选择特征"""
        return X[self.selected_features_]


@register_feature_transformer("RandomForestImportance")
class RandomForestImportanceSelector(BaseFeatureTransformer):
    """
    随机森林特征重要性选择器
    基于随机森林模型的特征重要性进行选择
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - n_estimators: 树的数量，默认 100
                - threshold: 重要性阈值，或 "mean" 或 "median"
                - max_features: 最大特征数
        """
        super().__init__(config)
        self.n_estimators = self.config.get("n_estimators", 100)
        self.threshold = self.config.get("threshold", "mean")
        self.max_features = self.config.get("max_features", None)

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series]] = None) -> "RandomForestImportanceSelector":
        """拟合"""
        if y is None:
            raise ValueError("RandomForestImportanceSelector 需要目标变量 y")

        y_values = y.values.ravel() if isinstance(y, (pd.DataFrame, pd.Series)) else y

        # 判断是分类还是回归
        n_classes = len(np.unique(y_values))
        if n_classes > 10 and n_classes > len(y_values) / 10:
            # 回归
            self.model_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                random_state=42,
            )
        else:
            # 分类
            self.model_ = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                random_state=42,
            )

        self.model_.fit(X.values, y_values)

        # 记录重要性
        importances = self.model_.feature_importances_
        self.feature_importances_ = dict(zip(X.columns, importances))

        # 确定阈值
        if self.threshold == "mean":
            threshold_value = np.mean(importances)
        elif self.threshold == "median":
            threshold_value = np.median(importances)
        else:
            threshold_value = self.threshold

        # 选择重要特征
        self.selected_mask_ = importances >= threshold_value
        self.selected_features_ = list(X.columns[self.selected_mask_])

        self.metadata = {
            "type": "RandomForestImportance",
            "n_estimators": self.n_estimators,
            "threshold": self.threshold,
            "threshold_value": float(threshold_value),
            "selected_features": self.selected_features_,
            "feature_importances": self.feature_importances_,
            "n_selected": len(self.selected_features_),
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """选择特征"""
        return X[self.selected_features_]


@register_feature_transformer("LassoSelection")
class LassoSelectionTransformer(BaseFeatureTransformer):
    """
    Lasso特征选择器
    使用Lasso回归的稀疏性进行特征选择
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - alpha: 正则化参数，或 "cv" 自动选择
                - threshold: 系数阈值
        """
        super().__init__(config)
        self.alpha = self.config.get("alpha", "cv")
        self.threshold = self.config.get("threshold", 1e-5)

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series]] = None) -> "LassoSelectionTransformer":
        """拟合"""
        if y is None:
            raise ValueError("LassoSelection 需要目标变量 y")

        y_values = y.values.ravel() if isinstance(y, (pd.DataFrame, pd.Series)) else y

        if self.alpha == "cv":
            self.model_ = LassoCV(cv=5, random_state=42)
        else:
            from sklearn.linear_model import Lasso
            self.model_ = Lasso(alpha=self.alpha, random_state=42)

        self.model_.fit(X.values, y_values)

        # 记录系数
        coefs = self.model_.coef_
        self.feature_coefs_ = dict(zip(X.columns, coefs))

        # 选择非零系数特征
        self.selected_mask_ = np.abs(coefs) > self.threshold
        self.selected_features_ = list(X.columns[self.selected_mask_])

        self.metadata = {
            "type": "LassoSelection",
            "alpha": float(self.alpha) if self.alpha != "cv" else "cv",
            "threshold": self.threshold,
            "selected_features": self.selected_features_,
            "feature_coefs": self.feature_coefs_,
            "n_selected": len(self.selected_features_),
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """选择特征"""
        return X[self.selected_features_]


@register_feature_transformer("CorrelationFilter")
class CorrelationFilterTransformer(BaseFeatureTransformer):
    """
    相关性过滤特征选择器
    移除与目标变量相关性低或高度相关的特征
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - target_threshold: 与目标变量相关性阈值
                - inter_threshold: 特征间相关性阈值
        """
        super().__init__(config)
        self.target_threshold = self.config.get("target_threshold", 0.1)
        self.inter_threshold = self.config.get("inter_threshold", 0.95)

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series]] = None) -> "CorrelationFilterTransformer":
        """拟合"""
        if y is None:
            raise ValueError("CorrelationFilter 需要目标变量 y")

        y_values = y.values.ravel() if isinstance(y, (pd.DataFrame, pd.Series)) else y

        # 计算与目标的相关性
        correlations_with_target = {}
        for col in X.columns:
            corr = np.corrcoef(X[col].values, y_values)[0, 1]
            if not np.isnan(corr):
                correlations_with_target[col] = corr

        # 选择与目标相关性足够的特征
        selected_by_target = [
            col for col, corr in correlations_with_target.items()
            if abs(corr) >= self.target_threshold
        ]

        # 如果没有与目标相关的特征，保留所有
        if len(selected_by_target) == 0:
            selected_by_target = list(X.columns)

        # 计算特征间相关性，移除高度相关的特征
        X_selected = X[selected_by_target]
        corr_matrix = X_selected.corr().abs()

        # 移除高度相关的特征对
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            if any(upper[col] > self.inter_threshold):
                to_drop.add(col)

        self.selected_features_ = [col for col in selected_by_target if col not in to_drop]

        self.metadata = {
            "type": "CorrelationFilter",
            "target_threshold": self.target_threshold,
            "inter_threshold": self.inter_threshold,
            "correlations_with_target": correlations_with_target,
            "selected_features": self.selected_features_,
            "n_selected": len(self.selected_features_),
        }
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """选择特征"""
        return X[self.selected_features_]