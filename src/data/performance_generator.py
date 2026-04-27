"""
Performance Data Generator

Generates synthetic performance data (Tg, viscosity, etc.) based on
simplified physical relationships.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class PerformanceGenerator:
    """Generate synthetic performance data based on physical relationships."""

    def __init__(self, config: Dict):
        """
        Initialize the performance data generator.

        Args:
            config: Configuration dictionary with performance generation parameters
        """
        self.config = config
        self.perf_config = config["performance"]
        self.physics = config["physics"]
        self._rng = np.random.default_rng(config.get("random_seed", 42))

    def _get_base_variance(self, param: str) -> Tuple[float, float]:
        """Get base and variance for a performance parameter."""
        return (
            self.perf_config[param]["base"],
            self.perf_config[param]["variance"],
        )

    def _is_log_scale(self, param: str) -> bool:
        """Check if parameter uses log scale."""
        return self.perf_config[param].get("log_scale", False)

    def generate_Tg(
        self,
        composition_factor: float = 0.0,
        crosslink_density: float = 0.0,
        temperature: float = 25.0,
    ) -> float:
        """
        Generate glass transition temperature.

        Args:
            composition_factor: Compositional influence factor
            crosslink_density: Crosslinking density (0-1)
            temperature: Process temperature

        Returns:
            Glass transition temperature in Celsius
        """
        base, variance = self._get_base_variance("Tg")

        Tg = base + composition_factor * self.physics["Tg_composition_factor"]
        Tg += crosslink_density * self.physics["crosslink_Tg_factor"]
        Tg += self._rng.normal(0, variance)

        return round(Tg, 2)

    def generate_viscosity(
        self,
        temperature: float = 25.0,
        composition_factor: float = 0.0,
    ) -> float:
        """
        Generate viscosity data.

        Uses Arrhenius-like temperature dependence.

        Args:
            temperature: Process temperature
            composition_factor: Compositional influence factor

        Returns:
            Viscosity in Pa.s
        """
        base, variance = self._get_base_variance("viscosity")

        temp_factor = np.exp(
            -self.physics["viscosity_temp_factor"] * (temperature - 25)
        )
        viscosity = base * temp_factor
        viscosity += composition_factor * 0.5
        viscosity += self._rng.normal(0, variance)
        viscosity = max(0.01, viscosity)

        return round(viscosity, 4)

    def generate_storage_modulus(
        self,
        Tg: float = 80.0,
        crosslink_density: float = 0.0,
    ) -> float:
        """
        Generate storage modulus G'.

        Args:
            Tg: Glass transition temperature
            crosslink_density: Crosslinking density (0-1)

        Returns:
            Storage modulus in MPa
        """
        base, variance = self._get_base_variance("storage_modulus")

        G_prime = base
        if Tg > 50:
            G_prime *= 1 + 0.1 * (Tg - 50)
        G_prime += crosslink_density * 50
        G_prime += self._rng.normal(0, variance)

        return round(max(1.0, G_prime), 2)

    def generate_loss_modulus(
        self,
        G_prime: float = 100.0,
        temperature: float = 25.0,
    ) -> float:
        """
        Generate loss modulus G''.

        Args:
            G_prime: Storage modulus value
            temperature: Process temperature

        Returns:
            Loss modulus in MPa
        """
        base, variance = self._get_base_variance("loss_modulus")

        G_double_prime = base * (G_prime / 100.0)
        G_double_prime += 0.05 * temperature
        G_double_prime += self._rng.normal(0, variance)

        return round(max(0.1, G_double_prime), 2)

    def generate_solvent_resistance(
        self,
        crosslink_density: float = 0.0,
        Tg: float = 80.0,
    ) -> float:
        """
        Generate solvent resistance data.

        Args:
            crosslink_density: Crosslinking density (0-1)
            Tg: Glass transition temperature

        Returns:
            Solvent resistance as % weight retention
        """
        base, variance = self._get_base_variance("solvent_resistance")

        resistance = base
        resistance += crosslink_density * 3.0
        if Tg > 100:
            resistance += 1.0
        resistance += self._rng.normal(0, variance)

        return round(min(100.0, max(50.0, resistance)), 2)

    def generate_hardness(
        self,
        crosslink_density: float = 0.0,
        filler_content: float = 0.0,
    ) -> float:
        """
        Generate hardness (Shore D).

        Args:
            crosslink_density: Crosslinking density (0-1)
            filler_content: Filler content fraction (0-1)

        Returns:
            Hardness in Shore D
        """
        base, variance = self._get_base_variance("hardness")

        hardness = base
        hardness += crosslink_density * 15.0
        hardness += filler_content * 10.0
        hardness += self._rng.normal(0, variance)

        return round(min(100.0, max(20.0, hardness)), 1)

    def generate_adhesion(
        self,
        resin_content: float = 0.0,
        solvent_content: float = 0.0,
    ) -> float:
        """
        Generate adhesion strength.

        Args:
            resin_content: Resin content fraction
            solvent_content: Solvent content fraction

        Returns:
            Adhesion strength in MPa
        """
        base, variance = self._get_base_variance("adhesion")

        adhesion = base
        adhesion += resin_content * 3.0
        adhesion -= solvent_content * 1.0
        adhesion += self._rng.normal(0, variance)

        return round(max(0.1, adhesion), 2)

    def generate_performance(
        self,
        process_params: Optional[Dict] = None,
        recipe_composition: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate a complete set of performance data.

        Args:
            process_params: Process parameters (temperature, etc.)
            recipe_composition: Recipe composition data

        Returns:
            Dictionary containing all performance metrics
        """
        process_params = process_params or {}
        recipe_composition = recipe_composition or {}

        temperature = process_params.get("temperature", 25.0)
        composition_factor = recipe_composition.get("composition_factor", 0.0)
        crosslink_density = recipe_composition.get("crosslink_density", 0.3)
        filler_content = recipe_composition.get("filler_content", 0.1)
        resin_content = recipe_composition.get("resin_content", 0.3)
        solvent_content = recipe_composition.get("solvent_content", 0.2)

        Tg = self.generate_Tg(composition_factor, crosslink_density, temperature)
        viscosity = self.generate_viscosity(temperature, composition_factor)
        G_prime = self.generate_storage_modulus(Tg, crosslink_density)
        G_double_prime = self.generate_loss_modulus(G_prime, temperature)
        solvent_resistance = self.generate_solvent_resistance(
            crosslink_density, Tg
        )
        hardness = self.generate_hardness(crosslink_density, filler_content)
        adhesion = self.generate_adhesion(resin_content, solvent_content)

        return {
            "Tg": Tg,
            "viscosity": viscosity,
            "storage_modulus": G_prime,
            "loss_modulus": G_double_prime,
            "solvent_resistance": solvent_resistance,
            "hardness": hardness,
            "adhesion": adhesion,
        }

    def generate_batch(self, n_samples: int) -> List[Dict]:
        """
        Generate a batch of performance data.

        Args:
            n_samples: Number of performance datasets to generate

        Returns:
            List of performance data dictionaries
        """
        return [self.generate_performance() for _ in range(n_samples)]

    def generate_correlated_batch(
        self,
        process_batch: List[Dict],
        composition_batch: List[Dict],
    ) -> List[Dict]:
        """
        Generate performance data correlated with process and composition.

        Args:
            process_batch: List of process parameter dictionaries
            composition_batch: List of recipe composition dictionaries

        Returns:
            List of correlated performance data dictionaries
        """
        if len(process_batch) != len(composition_batch):
            raise ValueError(
                "process_batch and composition_batch must have same length"
            )

        return [
            self.generate_performance(process, composition)
            for process, composition in zip(process_batch, composition_batch)
        ]

    def generate_dataframe(self, n_samples: int) -> Dict:
        """
        Generate performance data in a format suitable for DataFrame creation.

        Args:
            n_samples: Number of performance datasets to generate

        Returns:
            Dictionary with performance data
        """
        perf_batch = self.generate_batch(n_samples)

        data = {"sample_id": list(range(n_samples))}
        for key in [
            "Tg",
            "viscosity",
            "storage_modulus",
            "loss_modulus",
            "solvent_resistance",
            "hardness",
            "adhesion",
        ]:
            data[key] = [p[key] for p in perf_batch]

        return data
