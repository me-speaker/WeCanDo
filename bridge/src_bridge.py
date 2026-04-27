"""Src Bridge.

Direct bridge to src/ modules for agent access.
"""

import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.DAE import DenoisingAutoencoder
from src.core.registry import SURROGATE_REGISTRY, OPTIMIZER_REGISTRY
from src.core.factory import create_surrogate, create_optimizer
from src.core.runner import TaskRunner


class SrcBridge:
    """Direct access to src/ modules."""

    def __init__(self, src_root: Path = None):
        if src_root is None:
            src_root = Path(__file__).resolve().parents[1] / "src"
        self.src_root = src_root

    def get_dae(self) -> DenoisingAutoencoder:
        """Get DAE class."""
        return DenoisingAutoencoder

    def get_surrogate(self, name: str):
        """Get surrogate model by name."""
        if name in SURROGATE_REGISTRY:
            return SURROGATE_REGISTRY[name]
        return None

    def get_optimizer(self, name: str):
        """Get optimizer by name."""
        if name in OPTIMIZER_REGISTRY:
            return OPTIMIZER_REGISTRY[name]
        return None

    def get_runner(self) -> TaskRunner:
        """Get TaskRunner class."""
        return TaskRunner

    def create_dae(self, config: Dict) -> DenoisingAutoencoder:
        """Create DAE instance from config."""
        return DenoisingAutoencoder(**config)

    def create_surrogate(self, config: Dict):
        """Create surrogate model from config."""
        return create_surrogate(config)

    def create_optimizer(self, config: Dict):
        """Create optimizer from config."""
        return create_optimizer(config)