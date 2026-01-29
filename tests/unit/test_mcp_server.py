"""Unit tests for MCP Server."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import shutil

from ai_book_composer.mcp_server import (
    mcp,
    initialize_tools,
    list_files,
    read_text_file,
    write_chapter,
    write_chapter_list,
    generate_book
)


class TestMCPServerInitialization:
    """Test MCP server initialization."""
    
    def test_mcp_server_exists(self):
        """Test that MCP server instance exists."""
        assert mcp is not None
        assert mcp.name == "ai-book-composer"
        assert mcp.instructions is not None
        assert len(mcp.instructions) > 0
    
    def test_initialize_tools(self, tmp_path):
        """Test tools initialization with valid directories."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Should not raise an exception
        # Skip transcription to avoid downloading Whisper model
        initialize_tools(str(input_dir), str(output_dir), skip_transcription=True)


class TestMCPTools:
    """Test MCP tool functions."""
    
    @pytest.fixture
    def setup_dirs(self, tmp_path):
        """Setup test directories and files."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Create test files
        (input_dir / "test1.txt").write_text("Content 1")
        (input_dir / "test2.md").write_text("# Content 2")
        
        # Initialize tools (skip transcription to avoid downloading Whisper model)
        initialize_tools(str(input_dir), str(output_dir), skip_transcription=True)
        
        return input_dir, output_dir
    
    @pytest.mark.asyncio
    async def test_list_files_tool(self, setup_dirs):
        """Test list_files MCP tool."""
        input_dir, _ = setup_dirs
        
        result = await list_files()
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all("name" in f and "extension" in f for f in result)
    
    @pytest.mark.asyncio
    async def test_read_text_file_tool(self, setup_dirs):
        """Test read_text_file MCP tool."""
        input_dir, _ = setup_dirs
        test_file = str(input_dir / "test1.txt")
        
        result = await read_text_file(test_file)
        
        assert isinstance(result, dict)
        assert "content" in result
        assert result["content"] == "Content 1"
    
    @pytest.mark.asyncio
    async def test_write_chapter_tool(self, setup_dirs):
        """Test write_chapter MCP tool."""
        _, output_dir = setup_dirs
        
        result = await write_chapter(
            chapter_number=1,
            title="Test Chapter",
            content="Test content"
        )
        
        assert isinstance(result, dict)
        assert result["success"] is True
        assert "file_path" in result
        assert Path(result["file_path"]).exists()
    
    @pytest.mark.asyncio
    async def test_write_chapter_list_tool(self, setup_dirs):
        """Test write_chapter_list MCP tool."""
        _, output_dir = setup_dirs
        
        chapters = [
            {"number": 1, "title": "Chapter 1", "description": "First chapter"},
            {"number": 2, "title": "Chapter 2", "description": "Second chapter"}
        ]
        
        result = await write_chapter_list(chapters)
        
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["chapter_count"] == 2
        assert Path(result["file_path"]).exists()
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="BookGeneratorTool has dependency issues - not related to MCP server")
    async def test_generate_book_tool(self, setup_dirs):
        """Test generate_book MCP tool."""
        _, output_dir = setup_dirs
        
        # Create a chapter file first
        await write_chapter(1, "Test Chapter", "Test content")
        
        # Provide the required parameters
        result = await generate_book(
            book_title="Test Book",
            book_author="Test Author",
            language="en-US",
            chapters=[{"title": "Test Chapter", "content": "Test content"}],
            references=[]
        )
        
        assert isinstance(result, dict)
        assert result.get("success") is True or "output_path" in result


class TestMCPToolErrors:
    """Test MCP tool error handling."""
    
    @pytest.mark.asyncio
    async def test_tools_not_initialized_error(self):
        """Test that tools raise error when not initialized."""
        # Reset global tool instances by importing fresh
        import importlib
        import ai_book_composer.mcp_server as mcp_module
        
        # Set tool instances to None
        mcp_module._file_lister = None
        
        with pytest.raises(RuntimeError, match="Tools not initialized"):
            await list_files()
    
    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tmp_path):
        """Test reading a file that doesn't exist."""
        initialize_tools(str(tmp_path), str(tmp_path), skip_transcription=True)
        
        result = await read_text_file("/nonexistent/file.txt")
        
        assert isinstance(result, dict)
        assert "error" in result


class TestMCPServerIntegration:
    """Integration tests for MCP server."""
    
    @pytest.mark.asyncio
    async def test_list_available_tools(self):
        """Test listing all available tools from MCP server."""
        tools = await mcp.list_tools()
        
        assert isinstance(tools, list)
        assert len(tools) == 7  # We have 7 tools
        
        tool_names = [t.name for t in tools]
        expected_tools = [
            "list_files",
            "read_text_file",
            "transcribe_audio",
            "transcribe_video",
            "write_chapter",
            "write_chapter_list",
            "generate_book"
        ]
        
        for expected in expected_tools:
            assert expected in tool_names, f"Tool {expected} not found"
    
    @pytest.mark.asyncio
    async def test_call_tool_via_mcp(self, tmp_path):
        """Test calling a tool through the MCP interface."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Create test file
        (input_dir / "test.txt").write_text("Test content")
        
        # Initialize tools (skip transcription)
        initialize_tools(str(input_dir), str(output_dir), skip_transcription=True)
        
        # Call tool via MCP - it returns (content, structured_content)
        result = await mcp.call_tool("list_files", {})
        
        # MCP returns structured content as second element of tuple
        if isinstance(result, tuple):
            content, structured = result
            # Get the actual result from structured content
            actual_result = structured.get('result', [])
        else:
            actual_result = result
        
        assert isinstance(actual_result, list)
        assert len(actual_result) == 1
        assert actual_result[0]["name"] == "test.txt"


class TestMCPServerConfiguration:
    """Test MCP server configuration."""
    
    def test_server_name(self):
        """Test server has correct name."""
        assert mcp.name == "ai-book-composer"
    
    def test_server_instructions(self):
        """Test server has instructions."""
        assert mcp.instructions is not None
        assert "AI Book Composer" in mcp.instructions
        assert "list_files" in mcp.instructions
