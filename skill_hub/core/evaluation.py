"""Model Evaluation Skill.

Reusable skill for evaluating model performance.
"""

import numpy as np
from typing import Dict, Any


class EvaluationSkill:
    """Skill for model evaluation."""

    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate regression model.
        
        Returns:
            Dictionary with 'mse', 'rmse', 'mae', 'r2' keys
        """
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

    @staticmethod
    def evaluate_pareto_front(pareto_front: np.ndarray) -> Dict[str, Any]:
        """Evaluate Pareto front quality.
        
        Returns:
            Dictionary with 'n_solutions', 'spread', 'hypervolume' keys
        """
        return {
            "n_solutions": len(pareto_front),
            "spread": float(np.std(pareto_front, axis=0).mean()) if len(pareto_front) > 1 else 0.0,
            "hypervolume": float(np.abs(pareto_front).prod()),
        }
