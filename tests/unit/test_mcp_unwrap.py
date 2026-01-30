"""Unit tests for MCP result unwrapping."""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch

from ai_book_composer import mcp_client


class TestMCPResultUnwrapping:
    """Test MCP result unwrapping functionality in mcp_client."""

    def test_unwrap_list_of_wrapped_items(self):
        """Test unwrapping a list of wrapped MCP results."""
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
        result = mcp_client._unwrap_mcp_result(wrapped_result)

        # Verify
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "article2.txt"
        assert result[0]["extension"] == ".txt"
        assert result[1]["name"] == "article1.txt"
        assert result[1]["extension"] == ".txt"

    def test_unwrap_single_wrapped_dict(self):
        """Test unwrapping a single wrapped MCP result."""
        # Single wrapped dict result
        wrapped_result = {
            'id': 'lc_123',
            'text': json.dumps({"success": True, "file_path": "/tmp/chapter1.txt"}),
            'type': 'text'
        }

        # Unwrap
        result = mcp_client._unwrap_mcp_result(wrapped_result)

        # Verify
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["file_path"] == "/tmp/chapter1.txt"

    def test_unwrap_already_unwrapped_list(self):
        """Test that already unwrapped results pass through unchanged."""
        # Already unwrapped list
        unwrapped_result = [
            {"name": "file1.txt", "size": 100},
            {"name": "file2.txt", "size": 200}
        ]

        # Process
        result = mcp_client._unwrap_mcp_result(unwrapped_result)

        # Should be unchanged
        assert result == unwrapped_result

    def test_unwrap_already_unwrapped_dict(self):
        """Test that already unwrapped dict passes through unchanged."""
        # Already unwrapped dict
        unwrapped_result = {"success": True, "count": 5}

        # Process
        result = mcp_client._unwrap_mcp_result(unwrapped_result)

        # Should be unchanged
        assert result == unwrapped_result

    def test_unwrap_plain_string_in_text_field(self):
        """Test unwrapping when text field contains plain string (not JSON)."""
        # Wrapped with plain text
        wrapped_result = {
            'id': 'lc_456',
            'text': 'This is a plain text response',
            'type': 'text'
        }

        # Unwrap
        result = mcp_client._unwrap_mcp_result(wrapped_result)

        # Should return the plain text
        assert result == 'This is a plain text response'

    @pytest.mark.asyncio
    async def test_wrapped_tool_unwraps_result(self):
        """Test that _wrap_tool_with_unwrap properly wraps tools."""
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
        
        async def mock_ainvoke(*args, **kwargs):
            return wrapped_result
            
        mock_tool.ainvoke = mock_ainvoke

        # Wrap the tool
        wrapped_tool = mcp_client._wrap_tool_with_unwrap(mock_tool)

        # Invoke tool
        result = await wrapped_tool.ainvoke({})

        # Verify result is unwrapped
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "test.txt"
        assert result[0]["size"] == 100
        # Verify it's not wrapped anymore
        assert "id" not in result[0]
        assert "text" not in result[0]
        assert "type" not in result[0]

    def test_unwrap_empty_list(self):
        """Test that empty list passes through unchanged."""
        result = mcp_client._unwrap_mcp_result([])
        assert result == []

    def test_unwrap_partial_wrapped_list(self):
        """Test list where not all items are wrapped (should return as-is)."""
        # List with mixed content - not all items wrapped
        mixed_result = [
            {'id': 'lc_1', 'text': '{"name": "file1.txt"}', 'type': 'text'},
            {'name': 'file2.txt', 'size': 200}  # Not wrapped
        ]
        
        # Should return as-is since not all items are wrapped
        result = mcp_client._unwrap_mcp_result(mixed_result)
        assert result == mixed_result

    def test_unwrap_dict_with_non_text_type(self):
        """Test that dicts with type != 'text' are returned as-is."""
        wrapped_result = {
            'id': 'lc_123',
            'text': 'Some content',
            'type': 'image'  # Not 'text'
        }
        
        # Should return as-is
        result = mcp_client._unwrap_mcp_result(wrapped_result)
        assert result == wrapped_result

