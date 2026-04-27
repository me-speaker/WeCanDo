"""
Recipe Data Generator

Generates synthetic chemical/formula recipe data with configurable parameters.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class RecipeGenerator:
    """Generate synthetic recipe data for chemical formulations."""

    def __init__(self, config: Dict):
        """
        Initialize the recipe generator.

        Args:
            config: Configuration dictionary with recipe generation parameters
        """
        self.config = config
        self.n_components_range = config["recipe"]["n_components"]
        self.concentration_range = (
            config["recipe"]["concentration"]["min"],
            config["recipe"]["concentration"]["max"],
        )
        self.categories = config["recipe"]["categories"]
        self.components_per_category = config["recipe"]["components_per_category"]

        self._component_pool = self._build_component_pool()
        self._rng = np.random.default_rng(config.get("random_seed", 42))

    def _build_component_pool(self) -> Dict[str, List[str]]:
        """Build a pool of synthetic component names per category."""
        pool = {}
        for category in self.categories:
            pool[category] = [
                f"{category}_{i:03d}"
                for i in range(self.components_per_category)
            ]
        return pool

    def _generate_component_names(
        self, n_components: int
    ) -> List[Tuple[str, str]]:
        """Generate random component names with categories."""
        all_components = []
        for category, components in self._component_pool.items():
            for comp in components:
                all_components.append((comp, category))

        selected = self._rng.choice(
            len(all_components), size=n_components, replace=False
        )
        return [all_components[i] for i in selected]

    def _normalize_concentrations(
        self, concentrations: np.ndarray
    ) -> np.ndarray:
        """Normalize concentrations to sum to 1.0."""
        total = concentrations.sum()
        if total > 0:
            return concentrations / total
        return concentrations

    def generate_recipe(self) -> Dict:
        """
        Generate a single synthetic recipe.

        Returns:
            Dictionary containing recipe data with components and concentrations
        """
        n_components = self._rng.integers(
            self.n_components_range["min"],
            self.n_components_range["max"] + 1,
        )

        component_info = self._generate_component_names(n_components)

        concentrations = self._rng.uniform(
            self.concentration_range[0],
            self.concentration_range[1],
            size=n_components,
        )
        concentrations = self._normalize_concentrations(concentrations)

        recipe = {
            "components": [],
            "concentrations": concentrations.tolist(),
            "n_components": n_components,
        }

        for comp_name, category in component_info:
            recipe["components"].append(
                {"name": comp_name, "category": category}
            )

        return recipe

    def generate_batch(self, n_samples: int) -> List[Dict]:
        """
        Generate a batch of synthetic recipes.

        Args:
            n_samples: Number of recipes to generate

        Returns:
            List of recipe dictionaries
        """
        return [self.generate_recipe() for _ in range(n_samples)]

    def generate_dataframe(
        self, n_samples: int, component_columns: bool = True
    ) -> Dict:
        """
        Generate recipe data in a format suitable for DataFrame creation.

        Args:
            n_samples: Number of recipes to generate
            component_columns: If True, expand components into column format

        Returns:
            Dictionary with recipe data
        """
        recipes = self.generate_batch(n_samples)

        if component_columns:
            max_components = self.n_components_range["max"]
            data = {
                "recipe_id": list(range(n_samples)),
                "n_components": [r["n_components"] for r in recipes],
            }

            for i in range(max_components):
                data[f"component_{i}_name"] = []
                data[f"component_{i}_category"] = []
                data[f"component_{i}_concentration"] = []

            for recipe in recipes:
                for i in range(max_components):
                    if i < len(recipe["components"]):
                        data[f"component_{i}_name"].append(
                            recipe["components"][i]["name"]
                        )
                        data[f"component_{i}_category"].append(
                            recipe["components"][i]["category"]
                        )
                        data[f"component_{i}_concentration"].append(
                            recipe["concentrations"][i]
                        )
                    else:
                        data[f"component_{i}_name"].append(None)
                        data[f"component_{i}_category"].append(None)
                        data[f"component_{i}_concentration"].append(0.0)

            return data
        else:
            return {"recipes": recipes}
