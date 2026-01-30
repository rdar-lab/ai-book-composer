"""Unit tests for MCP result unwrapping."""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch

from ai_book_composer.agents.executor import ExecutorAgent
from ai_book_composer.config import Settings


class TestMCPResultUnwrapping:
    """Test MCP result unwrapping functionality."""

    def test_unwrap_list_of_wrapped_items(self):
        """Test unwrapping a list of wrapped MCP results."""
        # Create executor instance
        with patch('ai_book_composer.agents.executor.mcp_client.get_tools') as mock_get_tools:
            mock_get_tools.return_value = []
            executor = ExecutorAgent(Settings(), "/tmp", "/tmp")

        # Wrapped list result (as returned by langchain-mcp-adapters)
        wrapped_result = [
            {
                'id': 'lc_b3beae62-6e61-4c38-b7d8-b31379eac376',
                'text': json.dumps({
                    "path": "/tmp/test/article2.txt",
                    "name": "article2.txt",
                    "extension": ".txt",
                    "size": 119
                }),
                'type': 'text'
            },
            {
                'id': 'lc_edfba5bf-76f5-4360-8d12-219ff1448ba4',
                'text': json.dumps({
                    "path": "/tmp/test/article1.txt",
                    "name": "article1.txt",
                    "extension": ".txt",
                    "size": 117
                }),
                'type': 'text'
            }
        ]

        # Unwrap
        result = executor._unwrap_mcp_result(wrapped_result)

        # Verify
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "article2.txt"
        assert result[0]["extension"] == ".txt"
        assert result[1]["name"] == "article1.txt"
        assert result[1]["extension"] == ".txt"

    def test_unwrap_single_wrapped_dict(self):
        """Test unwrapping a single wrapped MCP result."""
        with patch('ai_book_composer.agents.executor.mcp_client.get_tools') as mock_get_tools:
            mock_get_tools.return_value = []
            executor = ExecutorAgent(Settings(), "/tmp", "/tmp")

        # Single wrapped dict result
        wrapped_result = {
            'id': 'lc_123',
            'text': json.dumps({"success": True, "file_path": "/tmp/chapter1.txt"}),
            'type': 'text'
        }

        # Unwrap
        result = executor._unwrap_mcp_result(wrapped_result)

        # Verify
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["file_path"] == "/tmp/chapter1.txt"

    def test_unwrap_already_unwrapped_list(self):
        """Test that already unwrapped results pass through unchanged."""
        with patch('ai_book_composer.agents.executor.mcp_client.get_tools') as mock_get_tools:
            mock_get_tools.return_value = []
            executor = ExecutorAgent(Settings(), "/tmp", "/tmp")

        # Already unwrapped list
        unwrapped_result = [
            {"name": "file1.txt", "size": 100},
            {"name": "file2.txt", "size": 200}
        ]

        # Process
        result = executor._unwrap_mcp_result(unwrapped_result)

        # Should be unchanged
        assert result == unwrapped_result

    def test_unwrap_already_unwrapped_dict(self):
        """Test that already unwrapped dict passes through unchanged."""
        with patch('ai_book_composer.agents.executor.mcp_client.get_tools') as mock_get_tools:
            mock_get_tools.return_value = []
            executor = ExecutorAgent(Settings(), "/tmp", "/tmp")

        # Already unwrapped dict
        unwrapped_result = {"success": True, "count": 5}

        # Process
        result = executor._unwrap_mcp_result(unwrapped_result)

        # Should be unchanged
        assert result == unwrapped_result

    def test_unwrap_plain_string_in_text_field(self):
        """Test unwrapping when text field contains plain string (not JSON)."""
        with patch('ai_book_composer.agents.executor.mcp_client.get_tools') as mock_get_tools:
            mock_get_tools.return_value = []
            executor = ExecutorAgent(Settings(), "/tmp", "/tmp")

        # Wrapped with plain text
        wrapped_result = {
            'id': 'lc_456',
            'text': 'This is a plain text response',
            'type': 'text'
        }

        # Unwrap
        result = executor._unwrap_mcp_result(wrapped_result)

        # Should return the plain text
        assert result == 'This is a plain text response'

    def test_invoke_tool_unwraps_result(self):
        """Test that _invoke_tool properly unwraps MCP results."""
        # Create mock tool that returns wrapped result
        mock_tool = Mock()
        mock_tool.name = "list_files"
        
        # Simulate wrapped result from langchain-mcp-adapters
        wrapped_result = [
            {
                'id': 'lc_1',
                'text': json.dumps({"name": "test.txt", "size": 100}),
                'type': 'text'
            }
        ]
        mock_tool.ainvoke = AsyncMock(return_value=wrapped_result)

        # Create executor with mock tool
        with patch('ai_book_composer.agents.executor.mcp_client.get_tools') as mock_get_tools:
            mock_get_tools.return_value = [mock_tool]
            executor = ExecutorAgent(Settings(), "/tmp", "/tmp")

        # Invoke tool
        result = executor._invoke_tool("list_files")

        # Verify result is unwrapped
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "test.txt"
        assert result[0]["size"] == 100
        # Verify it's not wrapped anymore
        assert "id" not in result[0]
        assert "text" not in result[0]
        assert "type" not in result[0]

    def test_invoke_tool_handles_empty_list(self):
        """Test that _invoke_tool handles empty list results."""
        mock_tool = Mock()
        mock_tool.name = "list_files"
        mock_tool.ainvoke = AsyncMock(return_value=[])

        with patch('ai_book_composer.agents.executor.mcp_client.get_tools') as mock_get_tools:
            mock_get_tools.return_value = [mock_tool]
            executor = ExecutorAgent(Settings(), "/tmp", "/tmp")

        # Invoke tool
        result = executor._invoke_tool("list_files")

        # Should return empty list unchanged
        assert result == []

    def test_unwrap_mixed_content_types(self):
        """Test unwrapping when list has mixed content types."""
        with patch('ai_book_composer.agents.executor.mcp_client.get_tools') as mock_get_tools:
            mock_get_tools.return_value = []
            executor = ExecutorAgent(Settings(), "/tmp", "/tmp")

        # List with one wrapped item and one non-standard item
        wrapped_result = [
            {
                'id': 'lc_1',
                'text': json.dumps({"name": "file1.txt"}),
                'type': 'text'
            },
            {
                'id': 'lc_2',
                'text': json.dumps({"name": "file2.txt"}),
                'type': 'text'
            }
        ]

        result = executor._unwrap_mcp_result(wrapped_result)

        assert len(result) == 2
        assert all(isinstance(item, dict) for item in result)
        assert result[0]["name"] == "file1.txt"
        assert result[1]["name"] == "file2.txt"
