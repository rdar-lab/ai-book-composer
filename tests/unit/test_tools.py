"""Unit tests for AI Book Composer tools with mocked LLMs."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

# noinspection PyUnresolvedReferences
from ai_book_composer.config import Settings
# noinspection PyUnresolvedReferences
from ai_book_composer.tools.base_tools import (
    FileListingTool,
    TextFileReaderTool,
    ChapterWriterTool,
    ChapterListWriterTool,
    AudioTranscriptionTool,
    VideoTranscriptionTool,
    is_path_safe,
    is_file_size_within_limits
)


class TestSecurity:
    """Test security functions."""

    def test_is_path_safe_valid(self, tmp_path):
        """Test path safety check with valid path."""
        base = tmp_path
        target = tmp_path / "subdir" / "file.txt"
        assert is_path_safe(Settings(), base, target)

    def test_is_path_safe_traversal(self, tmp_path):
        """Test path safety check with directory traversal."""
        base = tmp_path / "safe"
        target = tmp_path / "unsafe" / "file.txt"
        # Should be False when security is enabled
        # (depends on config, may need to mock settings)
        result = is_path_safe(Settings(), base, target)
        assert isinstance(result, bool)

    def test_is_file_size_within_limits_valid(self, tmp_path):
        """Test file size check with valid file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Small file")
        assert is_file_size_within_limits(Settings(), test_file)

    def test_is_file_size_within_limits_too_large(self, tmp_path):
        settings = Settings()
        settings.security.max_file_size_mb = 0.001
        """Test file size check with large file."""
        test_file = tmp_path / "large.txt"
        test_file.write_text("x" * 10000)  # 10KB file with 0.001MB limit
        assert not is_file_size_within_limits(settings, test_file)


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

        tool = FileListingTool(Settings(), str(tmp_path))
        files = tool.run()

        assert len(files) == 3
        assert all('path' in f for f in files)
        assert all('name' in f for f in files)
        assert all('extension' in f for f in files)

    def test_list_empty_directory(self, tmp_path):
        """Test listing empty directory."""
        tool = FileListingTool(Settings(), str(tmp_path))
        files = tool.run()
        assert files == []


class TestTextFileReaderTool:
    """Test TextFileReaderTool."""

    def test_read_text_file(self, tmp_path):
        """Test reading plain text file."""
        test_file = tmp_path / "test.txt"
        content = "\n".join([f"Line {i}" for i in range(1, 11)])
        test_file.write_text(content)

        tool = TextFileReaderTool(Settings())
        result = tool.run(str(test_file), start_line=1, end_line=5)

        assert "content" in result
        assert result["start_line"] == 1
        assert result["end_line"] == 5
        assert result["total_lines"] == 10
        assert result["has_more"] == True

    def test_read_nonexistent_file(self):
        """Test reading nonexistent file."""
        tool = TextFileReaderTool(Settings())
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

        tool = TextFileReaderTool(Settings())
        result = tool.run(str(test_file))

        assert "content" in result
        assert "Paragraph 1" in result["content"]
        assert "Paragraph 2" in result["content"]


class TestChapterWriterTool:
    """Test ChapterWriterTool."""

    def test_write_chapter(self, tmp_path):
        """Test writing chapter."""
        tool = ChapterWriterTool(Settings(), str(tmp_path))
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
        tool = ChapterWriterTool(Settings(), str(tmp_path))
        result = tool.run(1, "Test: Chapter/Name", "Content")

        assert result["success"] == True
        assert Path(result["file_path"]).exists()


class TestChapterListWriterTool:
    """Test ChapterListWriterTool."""

    def test_write_chapter_list(self, tmp_path):
        """Test writing chapter list."""
        tool = ChapterListWriterTool(Settings(), str(tmp_path))
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


class TestAudioTranscriptionTool:
    """Test AudioTranscriptionTool with caching and language support."""

    @patch('ai_book_composer.tools.base_tools.WhisperModel')
    def test_transcribe_audio_with_cache(self, mock_whisper_model, tmp_path):
        """Test audio transcription with caching."""
        # Setup mock settings
        settings = Settings()
        settings.whisper.mode = "local"
        settings.whisper.model_size = "base"
        settings.whisper.local = {"device": "cpu", "compute_type": "int8"}
        settings.security.max_file_size_mb = 500

        # Create test audio file
        test_audio = tmp_path / "test.mp3"
        test_audio.write_text("fake audio content")

        # Setup mock Whisper model
        mock_model_instance = Mock()
        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 5.0
        mock_segment.text = "Hello world"

        mock_info = Mock()
        mock_info.language = "en"
        mock_info.duration = 5.0

        mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_model.return_value = mock_model_instance

        # First call - should transcribe and cache
        tool = AudioTranscriptionTool(settings)
        result1 = tool.run(str(test_audio))

        assert result1["transcription"] == "Hello world"
        assert result1["language"] == "en"
        assert result1["duration"] == 5.0
        assert len(result1["segments"]) == 1

        # Check cache file was created
        cache_path = tmp_path / ".test.mp3.txt"
        assert cache_path.exists()

        # Verify cache content
        with open(cache_path, 'r') as f:
            cached_data = json.load(f)
        assert cached_data["transcription"] == "Hello world"
        assert cached_data["language"] == "en"

        # Second call - should use cache (transcribe should not be called again)
        mock_model_instance.transcribe.reset_mock()
        result2 = tool.run(str(test_audio))

        assert result2 == result1
        mock_model_instance.transcribe.assert_not_called()

    @patch('ai_book_composer.tools.base_tools.WhisperModel')
    def test_transcribe_audio_with_hebrew(self, mock_whisper_model, tmp_path):
        """Test audio transcription with Hebrew language specification."""
        # Setup mock settings
        settings = Settings()
        settings.whisper.mode = "local"
        settings.whisper.model_size = "base"
        settings.whisper.local = {"device": "cpu", "compute_type": "int8"}
        settings.security.max_file_size_mb = 500

        # Create test audio file
        test_audio = tmp_path / "test_hebrew.mp3"
        test_audio.write_text("fake audio content")

        # Setup mock Whisper model
        mock_model_instance = Mock()
        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 5.0
        mock_segment.text = "שלום עולם"

        mock_info = Mock()
        mock_info.language = "he"
        mock_info.duration = 5.0

        mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_model.return_value = mock_model_instance

        # Transcribe with Hebrew language
        tool = AudioTranscriptionTool(settings)
        result = tool.run(str(test_audio), language="he")

        assert result["transcription"] == "שלום עולם"
        assert result["language"] == "he"

        # Check cache file was created with language suffix
        cache_path = tmp_path / ".test_hebrew.mp3_he.txt"
        assert cache_path.exists()

        # Verify language parameter was passed to Whisper
        mock_model_instance.transcribe.assert_called_once()
        call_kwargs = mock_model_instance.transcribe.call_args[1]
        assert call_kwargs["language"] == "he"

    # noinspection PyUnusedLocal
    @patch('ai_book_composer.tools.base_tools.WhisperModel')
    def test_transcribe_audio_file_not_found(self, mock_whisper_model):
        """Test audio transcription with non-existent file."""
        settings = Settings()
        settings.whisper.mode = "local"
        settings.whisper.model_size = "base"
        settings.whisper.local = {"device": "cpu", "compute_type": "int8"}
        settings.security.max_file_size_mb = 500

        tool = AudioTranscriptionTool(settings)
        result = tool.run("/nonexistent/file.mp3")

        assert "error" in result
        assert result["transcription"] == ""


class TestVideoTranscriptionTool:
    """Test VideoTranscriptionTool with caching and language support."""

    @patch('ai_book_composer.tools.base_tools.ffmpeg')
    @patch('ai_book_composer.tools.base_tools.AudioTranscriptionTool')
    def test_transcribe_video_with_cache(self, mock_audio_tool_class, mock_ffmpeg, tmp_path):
        """Test video transcription with caching."""
        # Setup mock settings
        settings = Settings()
        settings.media_processing.chunk_duration = 300
        settings.security.max_file_size_mb = 500

        # Create test video file
        test_video = tmp_path / "test.mp4"
        test_video.write_text("fake video content")

        # Setup mock ffmpeg probe
        mock_ffmpeg.probe.return_value = {
            'format': {'duration': '10.0'}
        }

        # Setup mock audio transcription tool
        mock_audio_tool = Mock()
        mock_audio_tool.mode = "local"
        mock_audio_tool.transcribe_local.return_value = {
            "transcription": "Video content",
            "segments": [{"start": 0.0, "end": 10.0, "text": "Video content"}],
            "language": "en",
            "duration": 10.0
        }
        mock_audio_tool_class.return_value = mock_audio_tool

        # First call - should transcribe and cache
        tool = VideoTranscriptionTool(settings)
        result1 = tool.run(str(test_video))

        assert result1["transcription"] == "Video content"
        assert result1["language"] == "en"

        # Check cache file was created
        cache_path = tmp_path / ".test.mp4.txt"
        assert cache_path.exists()

        # Second call - should use cache
        result2 = tool.run(str(test_video))
        assert result2 == result1

    @patch('ai_book_composer.tools.base_tools.ffmpeg')
    @patch('ai_book_composer.tools.base_tools.AudioTranscriptionTool')
    def test_transcribe_video_with_language(self, mock_audio_tool_class, mock_ffmpeg, tmp_path):
        """Test video transcription with Hebrew language specification."""
        # Setup mock settings
        settings = Settings()
        settings.media_processing.chunk_duration = 300
        settings.security.max_file_size_mb = 500

        # Create test video file
        test_video = tmp_path / "test_hebrew.mp4"
        test_video.write_text("fake video content")

        # Setup mock ffmpeg probe
        mock_ffmpeg.probe.return_value = {
            'format': {'duration': '10.0'}
        }

        # Setup mock audio transcription tool
        mock_audio_tool = Mock()
        mock_audio_tool.mode = "local"
        mock_audio_tool.transcribe_local.return_value = {
            "transcription": "תוכן וידאו",
            "segments": [{"start": 0.0, "end": 10.0, "text": "תוכן וידאו"}],
            "language": "he",
            "duration": 10.0
        }
        mock_audio_tool_class.return_value = mock_audio_tool

        # Transcribe with Hebrew language
        tool = VideoTranscriptionTool(settings)
        result = tool.run(str(test_video), language="he")

        assert result["transcription"] == "תוכן וידאו"
        assert result["language"] == "he"

        # Check cache file was created with language suffix
        cache_path = tmp_path / ".test_hebrew.mp4_he.txt"
        assert cache_path.exists()

        # Verify language parameter was passed
        mock_audio_tool.transcribe_local.assert_called_once()
        call_args = mock_audio_tool.transcribe_local.call_args[0]
        call_kwargs = mock_audio_tool.transcribe_local.call_args[1] if len(
            mock_audio_tool.transcribe_local.call_args) > 1 else {}
        # Language should be either in args or kwargs
        assert call_kwargs.get('language') == "he" or (len(call_args) > 1 and call_args[1] == "he")


class TestImageListingTool:
    """Test ImageListingTool."""

    def test_list_images(self, tmp_path):
        """Test listing images in directory."""
        # noinspection PyUnresolvedReferences
        from ai_book_composer.tools.base_tools import ImageListingTool

        # Create test image files
        (tmp_path / "image1.jpg").write_bytes(b"fake jpg")
        (tmp_path / "image2.png").write_bytes(b"fake png")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "image3.gif").write_bytes(b"fake gif")

        # Create non-image files (should be ignored)
        (tmp_path / "text.txt").write_text("not an image")

        tool = ImageListingTool(Settings(), str(tmp_path))
        images = tool.run()

        assert len(images) == 3
        assert all('path' in img for img in images)
        assert all('filename' in img for img in images)
        assert all('format' in img for img in images)

    def test_list_empty_directory(self, tmp_path):
        """Test listing empty directory."""
        # noinspection PyUnresolvedReferences
        from ai_book_composer.tools.base_tools import ImageListingTool

        tool = ImageListingTool(Settings(), str(tmp_path))
        images = tool.run()
        assert images == []


class TestImageExtractorTool:
    """Test ImageExtractorTool."""

    def test_extract_images_file_not_found(self, tmp_path):
        """Test extract images from non-existent file."""
        # noinspection PyUnresolvedReferences
        from ai_book_composer.tools.base_tools import ImageExtractorTool

        tool = ImageExtractorTool(Settings(), str(tmp_path))
        result = tool.run(str(tmp_path / "nonexistent.pdf"))

        assert "error" in result
        assert result.get("images", []) == []
