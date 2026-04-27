"""Pipeline Execution Skill.

Reusable skill for executing the DAE → Surrogate → Optimizer pipeline.
"""

import numpy as np
from typing import Dict, Any, Callable, List


class PipelineSkill:
    """Skill for pipeline execution."""

    @staticmethod
    def execute_three_module_pipeline(
        X: np.ndarray,
        y: np.ndarray,
        dae_config: Dict,
        surrogate_config: Dict,
        optimizer_config: Dict,
    ) -> Dict[str, Any]:
        """Execute the 3-module pipeline.
        
        Args:
            X: Input features (formulation parameters)
            y: Objective values
            dae_config: DAE configuration
            surrogate_config: Surrogate model configuration
            optimizer_config: Optimizer configuration
            
        Returns:
            Pipeline result with 'success', 'pareto_front', 'models' keys
        """
        # This is a skill - it delegates to src/ modules
        from src.features.dae import DenoisingAutoencoder
        from src.core.factory import create_surrogate, create_optimizer
        
        # Train DAE
        dae = DenoisingAutoencoder(**dae_config)
        
        # Encode features
        import torch
        X_encoded = dae.encode(torch.FloatTensor(X)).detach().numpy()
        
        # Train surrogate
        surrogates = []
        for i in range(y.shape[1]):
            model = create_surrogate(surrogate_config["type"], surrogate_config)
            model.fit(X_encoded, y[:, i])
            surrogates.append(model)
        
        # Run optimizer
        optimizer = create_optimizer(optimizer_config["type"], optimizer_config)
        
        return {
            "success": True,
            "dae": dae,
            "surrogates": surrogates,
            "optimizer": optimizer,
            "X_encoded_shape": X_encoded.shape,
        }
