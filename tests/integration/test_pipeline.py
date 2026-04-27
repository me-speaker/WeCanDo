"""Integration tests for DAE-ELM pipeline."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TestPipeline:
    """Test full pipeline integration."""

    def test_pipeline_imports(self):
        """Test all pipeline components can be imported."""
        from daeelm_agents import DAEELMDeveloperAgent
        from bridge import TeamConnector, TaskExecutor
        from skill_hub.core.data_loader import DataLoaderSkill
        assert True

    def test_data_loader_skill(self):
        """Test data loader skill."""
        from skill_hub.core.data_loader import DataLoaderSkill
        X = np.random.rand(50, 8)
        y = np.random.rand(50, 2)
        result = DataLoaderSkill.validate_data(X, y)
        assert result["valid"] == True
