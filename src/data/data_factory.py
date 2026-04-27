"""
Data Factory

Main orchestrator for synthetic data generation, combining recipe,
process, and performance data generators.
"""

import os
import yaml
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path

from .recipe_generator import RecipeGenerator
from .process_generator import ProcessGenerator
from .performance_generator import PerformanceGenerator


class DataFactory:
    """
    Unified factory for generating synthetic chemical/formula data.

    Combines recipe, process, and performance data generation into a
    single interface with support for correlated data generation.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the DataFactory with configuration.

        Args:
            config_path: Path to YAML configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        else:
            config_path = Path(config_path)

        self.config = self._load_config(config_path)
        self.recipe_gen = RecipeGenerator(self.config)
        self.process_gen = ProcessGenerator(self.config)
        self.performance_gen = PerformanceGenerator(self.config)

        self.n_samples = self.config.get("n_samples", 1000)
        self.output_config = self.config.get("output", {})

    def _load_config(self, config_path: Path) -> Dict:
        """Load YAML configuration file."""
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def generate_recipe_data(
        self, n_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate recipe data.

        Args:
            n_samples: Number of samples to generate

        Returns:
            DataFrame containing recipe data
        """
        n = n_samples or self.n_samples
        data = self.recipe_gen.generate_dataframe(n)
        return pd.DataFrame(data)

    def generate_process_data(
        self, n_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate process parameter data.

        Args:
            n_samples: Number of samples to generate

        Returns:
            DataFrame containing process parameter data
        """
        n = n_samples or self.n_samples
        data = self.process_gen.generate_dataframe(n)
        return pd.DataFrame(data)

    def generate_performance_data(
        self, n_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate performance data.

        Args:
            n_samples: Number of samples to generate

        Returns:
            DataFrame containing performance data
        """
        n = n_samples or self.n_samples
        data = self.performance_gen.generate_dataframe(n)
        return pd.DataFrame(data)

    def generate_correlated_data(
        self, n_samples: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate correlated recipe, process, and performance data.

        The performance data is generated based on correlated process
        parameters and recipe composition.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Dictionary containing DataFrames for 'recipe', 'process', and 'performance'
        """
        n = n_samples or self.n_samples

        recipe_df = self.generate_recipe_data(n)
        process_df = self.generate_process_data(n)

        compositions = []
        for _, row in recipe_df.iterrows():
            compositions.append(
                {
                    "composition_factor": row.get(
                        "n_components", 5
                    )
                    / 10.0,
                    "crosslink_density": np.random.uniform(0.2, 0.5),
                    "filler_content": np.random.uniform(0.05, 0.2),
                    "resin_content": np.random.uniform(0.2, 0.4),
                    "solvent_content": np.random.uniform(0.1, 0.3),
                }
            )

        process_list = process_df.to_dict("records")
        performance_list = self.performance_gen.generate_correlated_batch(
            process_list, compositions
        )
        performance_df = pd.DataFrame(performance_list)
        performance_df["sample_id"] = list(range(n))

        return {
            "recipe": recipe_df,
            "process": process_df,
            "performance": performance_df,
        }

    def generate_complete_dataset(
        self, n_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate a complete dataset with all data combined.

        Args:
            n_samples: Number of samples to generate

        Returns:
            DataFrame containing all data merged
        """
        n = n_samples or self.n_samples

        data_dict = self.generate_correlated_data(n)

        recipe_df = data_dict["recipe"]
        process_df = data_dict["process"]
        performance_df = data_dict["performance"]

        combined_df = pd.concat(
            [
                recipe_df,
                process_df.drop(columns=["sample_id"], errors="ignore"),
                performance_df.drop(columns=["sample_id"], errors="ignore"),
            ],
            axis=1,
        )

        return combined_df

    def save_data(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        name: Optional[str] = None,
        format: str = "csv",
    ) -> Dict[str, str]:
        """
        Save generated data to files.

        Args:
            data: DataFrame or dictionary of DataFrames to save
            name: Base name for saved files
            format: Output format ('csv', 'json', 'parquet')

        Returns:
            Dictionary mapping dataset names to saved file paths
        """
        save_dir = Path(self.output_config.get("save_dir", "data/generated"))
        save_dir.mkdir(parents=True, exist_ok=True)

        prefix = name or self.output_config.get("file_prefix", "synthetic")
        saved_paths = {}

        if isinstance(data, pd.DataFrame):
            file_path = save_dir / f"{prefix}_data.{format}"
            self._save_single(data, file_path, format)
            saved_paths["data"] = str(file_path)
        else:
            for key, df in data.items():
                file_path = save_dir / f"{prefix}_{key}.{format}"
                self._save_single(df, file_path, format)
                saved_paths[key] = str(file_path)

        return saved_paths

    def _save_single(
        self, df: pd.DataFrame, path: Path, format: str
    ) -> None:
        """Save a single DataFrame to file."""
        if format == "csv":
            df.to_csv(path, index=False)
        elif format == "json":
            df.to_json(path, orient="records", indent=2)
        elif format == "parquet":
            df.to_parquet(path, index=False)

    def generate_and_save(
        self,
        n_samples: Optional[int] = None,
        dataset_type: str = "complete",
    ) -> Dict[str, str]:
        """
        Generate and save synthetic data.

        Args:
            n_samples: Number of samples to generate
            dataset_type: Type of dataset ('complete', 'correlated', 'recipe',
                         'process', 'performance')

        Returns:
            Dictionary mapping dataset names to saved file paths
        """
        if dataset_type == "complete":
            data = self.generate_complete_dataset(n_samples)
            return self.save_data(data, name="complete")
        elif dataset_type == "correlated":
            data = self.generate_correlated_data(n_samples)
            return self.save_data(data, name="correlated")
        elif dataset_type == "recipe":
            data = self.generate_recipe_data(n_samples)
            return self.save_data(data, name="recipe")
        elif dataset_type == "process":
            data = self.generate_process_data(n_samples)
            return self.save_data(data, name="process")
        elif dataset_type == "performance":
            data = self.generate_performance_data(n_samples)
            return self.save_data(data, name="performance")
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

    @staticmethod
    def load_data(path: str, format: str = "csv") -> pd.DataFrame:
        """
        Load previously generated data.

        Args:
            path: Path to data file
            format: File format ('csv', 'json', 'parquet')

        Returns:
            Loaded DataFrame
        """
        path = Path(path)

        if format == "csv" or path.suffix == ".csv":
            return pd.read_csv(path)
        elif format == "json" or path.suffix == ".json":
            return pd.read_json(path)
        elif format == "parquet" or path.suffix == ".parquet":
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
