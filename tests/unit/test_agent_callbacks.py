"""Unit tests for agent callback handler functionality."""

from unittest.mock import MagicMock, patch
from uuid import uuid4
import sys

# Mock the dependencies before importing llm
sys.modules['langchain_ollama'] = MagicMock()
sys.modules['langchain_openai'] = MagicMock()
sys.modules['langchain_google_genai'] = MagicMock()
sys.modules['deepagents'] = MagicMock()
sys.modules['deepagents.backends'] = MagicMock()

from src.ai_book_composer.llm import AgentProgressCallbackHandler


class TestAgentProgressCallbackHandler:
    """Test the AgentProgressCallbackHandler class."""

    def test_callback_handler_initialization(self):
        """Test callback handler initialization."""
        callback = MagicMock()
        handler = AgentProgressCallbackHandler(progress_callback=callback)
        assert handler.progress_callback == callback

    def test_on_llm_start_calls_callback(self):
        """Test that on_llm_start triggers the callback."""
        callback = MagicMock()
        handler = AgentProgressCallbackHandler(progress_callback=callback)
        
        serialized = {"name": "test_llm"}
        prompts = ["test prompt"]
        
        handler.on_llm_start(serialized, prompts, run_id=uuid4())
        
        callback.assert_called_once_with("llm_start", "Agent is thinking...")

    def test_on_tool_start_calls_callback(self):
        """Test that on_tool_start triggers the callback."""
        callback = MagicMock()
        handler = AgentProgressCallbackHandler(progress_callback=callback)
        
        serialized = {"name": "get_relevant_documents"}
        input_str = "test input"
        
        handler.on_tool_start(serialized, input_str, run_id=uuid4())
        
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "tool_start"
        assert "get_relevant_documents" in args[1]
        assert "test input" in args[1]

    def test_on_tool_end_calls_callback(self):
        """Test that on_tool_end triggers the callback."""
        callback = MagicMock()
        handler = AgentProgressCallbackHandler(progress_callback=callback)
        
        output = "test output"
        
        handler.on_tool_end(output, run_id=uuid4())
        
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "tool_end"
        assert "test output" in args[1]

    def test_on_llm_end_calls_callback(self):
        """Test that on_llm_end triggers the callback."""
        callback = MagicMock()
        handler = AgentProgressCallbackHandler(progress_callback=callback)
        
        response = {"output": "test response"}
        
        handler.on_llm_end(response, run_id=uuid4())
        
        callback.assert_called_once_with("llm_end", "Agent completed thinking. Response: {'output': 'test response'}")

    def test_callback_handler_without_callback(self):
        """Test that handler works without a callback function."""
        handler = AgentProgressCallbackHandler(progress_callback=None)
        
        # Should not raise any exceptions
        handler.on_llm_start({"name": "test"}, ["prompt"], run_id=uuid4())
        handler.on_tool_start({"name": "tool"}, "input", run_id=uuid4())
        handler.on_tool_end("output", run_id=uuid4())
        handler.on_chain_start({"name": "agent"}, {}, run_id=uuid4())
        handler.on_llm_end({}, run_id=uuid4())

    def test_tool_start_truncates_long_input(self):
        """Test that on_tool_start truncates long input strings."""
        callback = MagicMock()
        handler = AgentProgressCallbackHandler(progress_callback=callback)
        
        # Create a long input string
        long_input = "x" * 200
        serialized = {"name": "test_tool"}
        
        handler.on_tool_start(serialized, long_input, run_id=uuid4())
        
        callback.assert_called_once()
        args = callback.call_args[0]
        assert "..." in args[1]  # Should be truncated
        assert len(args[1]) < len(long_input) + 50  # Should be much shorter

    def test_tool_end_truncates_long_output(self):
        """Test that on_tool_end truncates long output strings."""
        callback = MagicMock()
        handler = AgentProgressCallbackHandler(progress_callback=callback)
        
        # Create a long output string
        long_output = "y" * 300
        
        handler.on_tool_end(long_output, run_id=uuid4())
        
        callback.assert_called_once()
        args = callback.call_args[0]
        assert "..." in args[1]  # Should be truncated
        assert len(args[1]) < len(long_output) + 50  # Should be much shorter
