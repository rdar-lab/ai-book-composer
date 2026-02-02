"""Integration test for long-term memory and message history management."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ai_book_composer.agents.agent_base import AgentBase
from ai_book_composer.agents.state import create_initial_state
from ai_book_composer.config import Settings
from ai_book_composer.long_term_memory import LongTermMemory


class TestMessageHistoryManagement:
    """Test that message history stays compact when using long-term memory."""

    def test_get_file_content_returns_compact_response(self):
        """Test that get_file_content tool returns compact responses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            settings = Mock(spec=Settings)
            ltm = LongTermMemory(tmpdir)
            agent = AgentBase(settings, long_term_memory=ltm)
            
            # Create state with gathered content
            state = create_initial_state(
                input_directory="/test",
                output_directory=tmpdir
            )
            
            # Add large file content to state
            large_content = "x" * 50000  # 50KB of content
            state["gathered_content"] = {
                "/test/large_file.txt": {
                    "name": "large_file.txt",
                    "path": "/test/large_file.txt",
                    "type": "text",
                    "content": large_content,
                    "summary": "Large file summary"
                }
            }
            
            # Store in long-term memory
            ltm.store_content("/test/large_file.txt", state["gathered_content"]["/test/large_file.txt"])
            
            agent.state = state
            
            # Get the tool
            tool = agent.get_file_content_tool()
            
            # Call tool to get content
            with patch('ai_book_composer.agents.agent_base.progress_display.progress'):
                result = tool.invoke({"file_name": "large_file.txt", "start_char": 0, "length": 5000})
            
            # Verify response is compact (not full content)
            assert len(result["chunk"]) == 5000  # Only requested chunk
            assert result["total_length"] == 50000
            assert result["has_more"] is True
            assert "chunk" in result
            assert "file_name" in result
            
            # Verify the response doesn't include the full content
            response_str = str(result)
            # Response should be much smaller than original content
            assert len(response_str) < 10000  # Response metadata + 5KB chunk

    def test_long_term_memory_prevents_state_duplication(self):
        """Test that long-term memory provides consistent access without duplicating data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ltm = LongTermMemory(tmpdir)
            
            # Store content
            content_info = {
                "name": "test.txt",
                "path": "/test/test.txt",
                "type": "text",
                "content": "Test content that should be stored once",
                "summary": "Test summary"
            }
            
            ltm.store_content("/test/test.txt", content_info)
            
            # Retrieve multiple times - should get same content without duplication
            content1 = ltm.retrieve_content("/test/test.txt")
            content2 = ltm.retrieve_content("/test/test.txt")
            
            assert content1 == content2
            assert content1 == "Test content that should be stored once"

    def test_tool_response_with_pagination(self):
        """Test that tool responses support pagination for large files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Mock(spec=Settings)
            ltm = LongTermMemory(tmpdir)
            agent = AgentBase(settings, long_term_memory=ltm)
            
            # Create state with large file
            large_content = "A" * 20000  # 20KB
            state = create_initial_state(
                input_directory="/test",
                output_directory=tmpdir
            )
            state["gathered_content"] = {
                "/test/file.txt": {
                    "name": "file.txt",
                    "path": "/test/file.txt",
                    "type": "text",
                    "content": large_content,
                    "summary": "Large file"
                }
            }
            
            ltm.store_content("/test/file.txt", state["gathered_content"]["/test/file.txt"])
            agent.state = state
            
            tool = agent.get_file_content_tool()
            
            # Get first chunk
            with patch('ai_book_composer.agents.agent_base.progress_display.progress'):
                result1 = tool.invoke({"file_name": "file.txt", "start_char": 0, "length": 5000})
            
            assert len(result1["chunk"]) == 5000
            assert result1["start_char"] == 0
            assert result1["end_char"] == 5000
            assert result1["has_more"] is True
            assert result1["chunk"] == "A" * 5000
            
            # Get second chunk
            with patch('ai_book_composer.agents.agent_base.progress_display.progress'):
                result2 = tool.invoke({"file_name": "file.txt", "start_char": 5000, "length": 5000})
            
            assert len(result2["chunk"]) == 5000
            assert result2["start_char"] == 5000
            assert result2["end_char"] == 10000
            assert result2["has_more"] is True
            assert result2["chunk"] == "A" * 5000

    def test_tool_handles_missing_long_term_memory(self):
        """Test that tool falls back to state if long-term memory not available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Mock(spec=Settings)
            # Create agent WITHOUT long-term memory
            agent = AgentBase(settings, long_term_memory=None)
            
            # Create state with content
            state = create_initial_state(
                input_directory="/test",
                output_directory=tmpdir
            )
            state["gathered_content"] = {
                "/test/file.txt": {
                    "name": "file.txt",
                    "path": "/test/file.txt",
                    "type": "text",
                    "content": "Fallback content from state",
                    "summary": "File summary"
                }
            }
            
            agent.state = state
            tool = agent.get_file_content_tool()
            
            # Should still work with state content
            with patch('ai_book_composer.agents.agent_base.progress_display.progress'):
                result = tool.invoke({"file_name": "file.txt", "start_char": 0, "length": 100})
            
            assert result["chunk"] == "Fallback content from state"
            assert not result["has_more"]
