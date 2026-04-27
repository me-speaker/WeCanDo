"""Unit tests for DAE-ELM agents."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TestDAEELMAgents:
    """Test DAE-ELM agent implementations."""

    def test_base_agent_import(self):
        """Test base agent can be imported."""
        from daeelm_agents.base_agent import DAEELMBaseAgent
        assert DAEELMBaseAgent is not None

    def test_ceo_agent_import(self):
        """Test CEO agent can be imported."""
        from daeelm_agents.ceo_agent import DAEELMCEOAgent
        assert DAEELMCEOAgent is not None

    def test_complexity_evaluation(self):
        """Test CEO can evaluate complexity."""
        from daeelm_agents.ceo_agent import DAEELMCEOAgent
        ceo = DAEELMCEOAgent()
        result = ceo.evaluate_dae_elm_complexity("multi-objective optimization with NSGA2")
        assert result["needs_nsga2"] == True
        assert result["level"] in ["simple", "medium", "complex"]
