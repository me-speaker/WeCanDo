"""Visualization Skill.

Reusable skill for visualizing results.
"""

import numpy as np
from typing import Optional, List


class VisualizationSkill:
    """Skill for visualization."""

    @staticmethod
    def plot_pareto_front(
        pareto_front: np.ndarray,
        objectives: Optional[List[str]] = None,
        title: str = "Pareto Front",
    ) -> str:
        """Generate Pareto front plot.
        
        Returns:
            Path to generated plot or plot code
        """
        # This is a skill - it could use matplotlib/plotly
        # For now, return plot code
        return f"""
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='blue')
plt.xlabel('{objectives[0] if objectives else "Objective 1"}')
plt.ylabel('{objectives[1] if objectives else "Objective 2"}')
plt.title('{title}')
plt.savefig('pareto_front.png')
"""
    
    @staticmethod
    def plot_training_curves(train_losses: List[float], val_losses: Optional[List[float]] = None) -> str:
        """Generate training curves plot.
        
        Returns:
            Path to generated plot or plot code
        """
        return f"""
import matplotlib.pyplot as plt

plt.figure()
plt.plot({train_losses}, label='Train')
{"plt.plot(" + str(val_losses) + ", label='Val')" if val_losses else ""}
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_curves.png')
"""
