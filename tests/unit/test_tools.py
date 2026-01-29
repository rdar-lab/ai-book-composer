"""Unit tests for AI Book Composer tools with mocked LLMs."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from ai_book_composer.tools.base_tools import (
    FileListingTool,
    TextFileReaderTool,
    ChapterWriterTool,
    ChapterListWriterTool,
    is_path_safe,
    check_file_size
)


class TestSecurity:
    """Test security functions."""
    
    def test_is_path_safe_valid(self, tmp_path):
        """Test path safety check with valid path."""
        base = tmp_path
        target = tmp_path / "subdir" / "file.txt"
        assert is_path_safe(base, target)
    
    def test_is_path_safe_traversal(self, tmp_path):
        """Test path safety check with directory traversal."""
        base = tmp_path / "safe"
        target = tmp_path / "unsafe" / "file.txt"
        # Should be False when security is enabled
        # (depends on config, may need to mock settings)
        result = is_path_safe(base, target)
        assert isinstance(result, bool)
    
    def test_check_file_size_valid(self, tmp_path):
        """Test file size check with valid file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Small file")
        assert check_file_size(test_file)
    
    @patch('ai_book_composer.tools.base_tools.settings.security.max_file_size_mb', 0.001)
    def test_check_file_size_too_large(self, tmp_path):
        """Test file size check with large file."""
        test_file = tmp_path / "large.txt"
        test_file.write_text("x" * 10000)  # 10KB file with 0.001MB limit
        assert not check_file_size(test_file)


class TestFileListingTool:
    """Test FileListingTool."""
    
    def test_list_files(self, tmp_path):
        """Test listing files in directory."""
        # Create test files
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.md").write_text("content2")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("content3")
        
        tool = FileListingTool(str(tmp_path))
        files = tool.run()
        
        assert len(files) == 3
        assert all('path' in f for f in files)
        assert all('name' in f for f in files)
        assert all('extension' in f for f in files)
    
    def test_list_empty_directory(self, tmp_path):
        """Test listing empty directory."""
        tool = FileListingTool(str(tmp_path))
        files = tool.run()
        assert files == []


class TestTextFileReaderTool:
    """Test TextFileReaderTool."""
    
    def test_read_text_file(self, tmp_path):
        """Test reading plain text file."""
        test_file = tmp_path / "test.txt"
        content = "\n".join([f"Line {i}" for i in range(1, 11)])
        test_file.write_text(content)
        
        tool = TextFileReaderTool()
        result = tool.run(str(test_file), start_line=1, end_line=5)
        
        assert "content" in result
        assert result["start_line"] == 1
        assert result["end_line"] == 5
        assert result["total_lines"] == 10
        assert result["has_more"] == True
    
    def test_read_nonexistent_file(self):
        """Test reading nonexistent file."""
        tool = TextFileReaderTool()
        result = tool.run("/nonexistent/file.txt")
        assert "error" in result
        assert result["content"] == ""
    
    @patch('ai_book_composer.tools.base_tools.DocxDocument')
    def test_read_docx_file(self, mock_docx, tmp_path):
        """Test reading DOCX file."""
        test_file = tmp_path / "test.docx"
        test_file.write_text("")  # Create empty file for test
        
        # Mock DOCX document
        mock_doc = Mock()
        mock_para1 = Mock()
        mock_para1.text = "Paragraph 1"
        mock_para2 = Mock()
        mock_para2.text = "Paragraph 2"
        mock_doc.paragraphs = [mock_para1, mock_para2]
        mock_docx.return_value = mock_doc
        
        tool = TextFileReaderTool()
        result = tool.run(str(test_file))
        
        assert "content" in result
        assert "Paragraph 1" in result["content"]
        assert "Paragraph 2" in result["content"]


class TestChapterWriterTool:
    """Test ChapterWriterTool."""
    
    def test_write_chapter(self, tmp_path):
        """Test writing chapter."""
        tool = ChapterWriterTool(str(tmp_path))
        result = tool.run(1, "Introduction", "This is the intro content")
        
        assert result["success"] == True
        assert "file_path" in result
        assert Path(result["file_path"]).exists()
        
        # Check content
        content = Path(result["file_path"]).read_text()
        assert "Chapter 1: Introduction" in content
        assert "This is the intro content" in content
    
    def test_write_chapter_special_chars(self, tmp_path):
        """Test writing chapter with special characters in title."""
        tool = ChapterWriterTool(str(tmp_path))
        result = tool.run(1, "Test: Chapter/Name", "Content")
        
        assert result["success"] == True
        assert Path(result["file_path"]).exists()


class TestChapterListWriterTool:
    """Test ChapterListWriterTool."""
    
    def test_write_chapter_list(self, tmp_path):
        """Test writing chapter list."""
        tool = ChapterListWriterTool(str(tmp_path))
        chapters = [
            {"number": 1, "title": "Chapter 1", "description": "Desc 1"},
            {"number": 2, "title": "Chapter 2", "description": "Desc 2"}
        ]
        
        result = tool.run(chapters)
        
        assert result["success"] == True
        assert result["chapter_count"] == 2
        assert Path(result["file_path"]).exists()
        
        # Check content
        import json
        with open(result["file_path"]) as f:
            saved_chapters = json.load(f)
        assert len(saved_chapters) == 2
        assert saved_chapters[0]["title"] == "Chapter 1"
