"""
DeepInd 特征工程模块
提供特征转换、选择、降维和领域特征构造功能
"""

from .base import BaseFeatureTransformer, FeaturePipeline, FeatureRegistry, register_feature_transformer
from .transformers import (
    StandardScalerTransformer,
    MinMaxScalerTransformer,
    RobustScalerTransformer,
    QuantileTransformer,
    PowerTransformer,
    OneHotEncoderTransformer,
    OrdinalEncoderTransformer,
    PolynomialFeaturesTransformer,
    LogTransformer,
    InteractionTransformer,
)
from .feature_selector import (
    VarianceThresholdSelector,
    SelectKBestTransformer,
    RandomForestImportanceSelector,
    LassoSelectionTransformer,
    CorrelationFilterTransformer,
)
from .dimensionality_reduction import (
    PCATransformer,
    KernelPCATransformer,
    IncrementalPCATransformer,
    TruncatedSVDTransformer,
    LDATransformer,
    AutoencoderReducer,
)
from .domain_features import (
    MolarRatioTransformer,
    PairwiseMolarRatioTransformer,
    FunctionalGroupCountTransformer,
    CrosslinkingDensityTransformer,
    CustomFormulaTransformer,
    GlassTransitionTemperatureTransformer,
    SolubilityParameterTransformer,
    PolydispersityIndexTransformer,
    ViscosityEstimateTransformer,
)
from .fea2dae_bridge import DAEFeatureBridge, DAEPipelineTransformer
from .DAE import DenoisingAutoencoder

__all__ = [
    # Base
    "BaseFeatureTransformer",
    "FeaturePipeline",
    "FeatureRegistry",
    "register_feature_transformer",
    # Transformers
    "StandardScalerTransformer",
    "MinMaxScalerTransformer",
    "RobustScalerTransformer",
    "QuantileTransformer",
    "PowerTransformer",
    "OneHotEncoderTransformer",
    "OrdinalEncoderTransformer",
    "PolynomialFeaturesTransformer",
    "LogTransformer",
    "InteractionTransformer",
    # Feature Selectors
    "VarianceThresholdSelector",
    "SelectKBestTransformer",
    "RandomForestImportanceSelector",
    "LassoSelectionTransformer",
    "CorrelationFilterTransformer",
    # Dimensionality Reduction
    "PCATransformer",
    "KernelPCATransformer",
    "IncrementalPCATransformer",
    "TruncatedSVDTransformer",
    "LDATransformer",
    "AutoencoderReducer",
    # Domain Features
    "MolarRatioTransformer",
    "PairwiseMolarRatioTransformer",
    "FunctionalGroupCountTransformer",
    "CrosslinkingDensityTransformer",
    "CustomFormulaTransformer",
    "GlassTransitionTemperatureTransformer",
    "SolubilityParameterTransformer",
    "PolydispersityIndexTransformer",
    "ViscosityEstimateTransformer",
    # Bridge
    "DAEFeatureBridge",
    "DAEPipelineTransformer",
    # DAE
    "DenoisingAutoencoder",
]
