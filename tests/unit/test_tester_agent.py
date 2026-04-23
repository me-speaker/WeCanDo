"""Unit tests for TesterAgent."""
import pytest
from unittest.mock import patch, MagicMock
from agents.base import AgentMessage, AgentStatus
from agents.tester import TesterAgent


class TestTesterAgent:
    """Tests for TesterAgent class."""

    def test_tester_agent_initialization(self):
        """Test TesterAgent initialization."""
        tester = TesterAgent()
        assert tester.name == "tester"
        assert tester.agent_type == "validator"
        assert tester.status == AgentStatus.IDLE
        assert tester.generated_tests == []

    def test_tester_system_prompt(self):
        """Test TesterAgent has correct system prompt."""
        tester = TesterAgent()
        prompt = tester._get_system_prompt()
        assert "Tester Agent" in prompt
        assert "Test and validate implementations" in prompt

    def test_tester_tools_defined(self):
        """Test TesterAgent has required tools."""
        tester = TesterAgent()
        tools = tester._get_tools()
        tool_names = [t.name for t in tools]
        assert "write_test" in tool_names
        assert "run_tests" in tool_names
        assert "generate_test_report" in tool_names

    def test_generate_tests_returns_result(self):
        """Test _generate_tests method returns properly structured result."""
        tester = TesterAgent()

        mock_response = '''{
            "action": "write_test",
            "file_path": "tests/unit/test_example.py",
            "content": "import pytest\\n\\ndef test_example():\\n    assert True",
            "test_cases": ["test_example"]
        }'''

        with patch.object(tester, 'think', return_value=mock_response):
            task = {
                "module_name": "example",
                "specs": {"functionality": "basic"},
                "test_type": "unit"
            }
            result = tester._generate_tests(task)
            assert "status" in result or "tests_generated" in result

    def test_write_test_file_creates_directory(self, tmp_path):
        """Test _write_test_file creates parent directories."""
        tester = TesterAgent()
        tester.project_root = tmp_path

        result = tester._write_test_file({
            "file_path": "tests/unit/test_sample.py",
            "content": "# Test file",
            "test_cases": ["test_sample"]
        })

        assert result["status"] == "written"
        assert "test_sample.py" in result["file_path"]

    def test_process_with_generate_tests_type(self):
        """Test process method with generate_tests task type."""
        tester = TesterAgent()

        mock_response = '''{
            "action": "write_test",
            "file_path": "tests/unit/test_sample.py",
            "content": "import pytest",
            "test_cases": []
        }'''

        with patch.object(tester, 'think', return_value=mock_response):
            message = AgentMessage(
                sender="developer",
                recipient="tester",
                msg_type="TEST_TASK",
                content={"type": "generate_tests", "module_name": "sample"}
            )

            response = tester.process(message)
            assert response is not None
            assert response.msg_type == "test_results"
            assert response.sender == "tester"
            assert response.recipient == "developer"

    def test_process_with_run_tests_type(self):
        """Test process method with run_tests task type."""
        tester = TesterAgent()

        message = AgentMessage(
            sender="developer",
            recipient="tester",
            msg_type="TEST_TASK",
            content={"type": "run_tests", "test_path": "tests/"}
        )

        # Mock subprocess to avoid actual test execution
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="All tests passed",
                stderr=""
            )
            response = tester.process(message)
            assert response is not None

    def test_run_tests_timeout(self):
        """Test run_tests handles timeout."""
        tester = TesterAgent()

        with patch('subprocess.run', side_effect=Exception("Timeout")):
            result = tester._run_tests({"test_path": "tests/"})
            assert result["status"] == "error"

    def test_generate_tests_handles_invalid_json(self):
        """Test _generate_tests handles non-JSON LLM response."""
        tester = TesterAgent()

        with patch.object(tester, 'think', return_value="This is not JSON"):
            task = {"module_name": "test", "test_type": "unit"}
            result = tester._generate_tests(task)
            # Should handle gracefully with fallback
            assert result is not None

    def test_write_test_file_missing_content(self):
        """Test _write_test_file handles missing file_path or content."""
        tester = TesterAgent()

        result = tester._write_test_file({"file_path": "", "content": ""})
        assert result["status"] == "error"
        assert "Missing" in result["message"]


class TestTesterAgentIntegration:
    """Integration tests for TesterAgent with other agents."""

    def test_tester_receives_task_from_developer(self):
        """Test TesterAgent can receive and process task from developer."""
        tester = TesterAgent()

        mock_response = '{"action": "write_test", "file_path": "tests/test_x.py", "content": "# test", "test_cases": []}'

        with patch.object(tester, 'think', return_value=mock_response):
            task_msg = AgentMessage(
                sender="developer",
                recipient="tester",
                msg_type="TEST_TASK",
                content={
                    "type": "generate_tests",
                    "module_name": "data_processor",
                    "specs": {"input": "DataFrame", "output": "DataFrame"}
                }
            )

            response = tester.process(task_msg)
            assert response is not None
            assert response.recipient == "developer"
            assert response.msg_type == "test_results"