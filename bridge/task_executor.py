"""Task Executor.

Executes tasks by calling src/ modules.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.DAE import DenoisingAutoencoder
from src.core.factory import create_surrogate, create_optimizer
from src.core.runner import TaskRunner


class TaskExecutor:
    """Executes tasks using src/ business logic."""

    def __init__(self):
        self.results: Dict[str, Any] = {}

    def execute_dae_training(self, X, config: Dict) -> DenoisingAutoencoder:
        """Execute DAE training."""
        input_dim = config.get("input_dim", X.shape[1] if hasattr(X, 'shape') else 100)
        hidden_dim = config.get("hidden_dim", 16)
        noise_factor = config.get("noise_factor", 0.1)
        epochs = config.get("epochs", 100)
        lr = config.get("lr", 0.001)

        dae = DenoisingAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            noise_factor=noise_factor,
            epochs=epochs,
            lr=lr
        )

        if hasattr(X, 'shape'):
            X_features = dae.fit_transform(X)
            return dae
        return dae

    def execute_surrogate_training(self, X, y, surrogate_type: str, config: Dict) -> Any:
        """Execute surrogate model training."""
        config["type"] = surrogate_type
        model = create_surrogate(config)
        model.fit(X, y)
        return model

    def execute_optimization(self, objectives: list, bounds: list, optimizer_type: str, config: Dict) -> Any:
        """Execute optimization."""
        config["type"] = optimizer_type
        optimizer = create_optimizer(config)
        return optimizer.optimize(objectives, bounds)

    def execute_pipeline(self, pipeline_config: Dict) -> Dict[str, Any]:
        """Execute full pipeline."""
        runner = TaskRunner(pipeline_config)
        return runner.run()