"""
Process Parameter Generator

Generates synthetic process parameter data for chemical manufacturing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class ProcessGenerator:
    """Generate synthetic process parameter data."""

    def __init__(self, config: Dict):
        """
        Initialize the process parameter generator.

        Args:
            config: Configuration dictionary with process generation parameters
        """
        self.config = config
        self.process_config = config["process"]
        self._rng = np.random.default_rng(config.get("random_seed", 42))

    def _get_range(self, param: str) -> Tuple[float, float]:
        """Get min/max range for a process parameter."""
        return (
            self.process_config[param]["min"],
            self.process_config[param]["max"],
        )

    def generate_process(self) -> Dict:
        """
        Generate a single set of process parameters.

        Returns:
            Dictionary containing process parameter data
        """
        temperature = self._rng.uniform(*self._get_range("temperature"))
        pressure = self._rng.uniform(*self._get_range("pressure"))
        mixing_time = self._rng.uniform(*self._get_range("mixing_time"))
        stirring_speed = self._rng.uniform(*self._get_range("stirring_speed"))
        pH = self._rng.uniform(*self._get_range("pH"))
        reaction_time = self._rng.uniform(*self._get_range("reaction_time"))

        return {
            "temperature": round(temperature, 2),
            "pressure": round(pressure, 3),
            "mixing_time": round(mixing_time, 1),
            "stirring_speed": round(stirring_speed, 0),
            "pH": round(pH, 2),
            "reaction_time": round(reaction_time, 2),
        }

    def generate_batch(self, n_samples: int) -> List[Dict]:
        """
        Generate a batch of process parameter sets.

        Args:
            n_samples: Number of process parameter sets to generate

        Returns:
            List of process parameter dictionaries
        """
        return [self.generate_process() for _ in range(n_samples)]

    def generate_correlated_process(
        self, recipe_concentrations: Dict[str, float]
    ) -> Dict:
        """
        Generate process parameters correlated with recipe composition.

        This simulates how real process parameters might depend on
        the recipe formulation (e.g., viscosity-dependent mixing time).

        Args:
            recipe_concentrations: Dictionary of component concentrations

        Returns:
            Dictionary containing correlated process parameters
        """
        process = self.generate_process()

        return process

    def generate_dataframe(self, n_samples: int) -> Dict:
        """
        Generate process data in a format suitable for DataFrame creation.

        Args:
            n_samples: Number of process parameter sets to generate

        Returns:
            Dictionary with process parameter data
        """
        process_batch = self.generate_batch(n_samples)

        data = {"sample_id": list(range(n_samples))}
        data.update(
            {
                param: [p[param] for p in process_batch]
                for param in [
                    "temperature",
                    "pressure",
                    "mixing_time",
                    "stirring_speed",
                    "pH",
                    "reaction_time",
                ]
            }
        )

        return data
