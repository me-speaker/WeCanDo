"""DAE-ELM Tester Agent.

Tests the DAE-ELM pipeline components.
"""

import sys
from pathlib import Path
_parent = Path(__file__).resolve().parents[3]
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from agents.base import BaseAgent
from agents.tester import TesterAgent as BaseTesterAgent
from typing import Dict, Any, List
from .base_agent import DAEELMBaseAgent


class DAEELMTesterAgent(DAEELMBaseAgent, BaseTesterAgent):
    """Tester Agent for DAE-ELM project.

    Responsibilities:
    - Test DAE encoding/decoding
    - Test surrogate model accuracy
    - Test optimizer convergence
    - Run integration tests
    """

    def __init__(self):
        BaseAgent.__init__(self, name="tester", agent_type="tester")

    def test_dae(self, dae, X):
        """Test DAE encoding/decoding."""
        import torch
        X_tensor = torch.FloatTensor(X)
        encoded = dae.encode(X_tensor)
        decoded = dae.decode(encoded)
        reconstruction_error = torch.nn.MSELoss()(decoded, X_tensor).item()
        return {"reconstruction_error": reconstruction_error, "passed": reconstruction_error < 0.1}

    def test_surrogate(self, model, X, y):
        """Test surrogate model accuracy."""
        y_pred = model.predict(X)
        mse = ((y - y_pred) ** 2).mean()
        return {"mse": mse, "passed": mse < 0.1}

    def test_optimizer(self, optimizer, objective_func, bounds):
        """Test optimizer convergence."""
        result = optimizer.optimize(objective_func, bounds)
        return {"has_pareto_front": len(result) > 0, "n_solutions": len(result)}
