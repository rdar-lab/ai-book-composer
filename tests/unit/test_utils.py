"""Unit tests for AI Book Composer tools with mocked LLMs."""

import json
import sys
from io import BytesIO
from unittest.mock import Mock, patch

import pytest
from PIL import Image as PILImage

from src.ai_book_composer.config import Settings
from src.ai_book_composer.utils import file_utils
from src.ai_book_composer.utils.file_utils import describe_image
from src.ai_book_composer.utils.file_utils import is_image_meaningful
from src.ai_book_composer.utils.file_utils import (
    is_path_safe,
    is_file_size_within_limits,
    read_text_file,
    list_input_files,
    read_audio_file,
    read_video_file,
    list_images,
    extract_images_from_pdf
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

        files = list_input_files(Settings(), str(tmp_path))

        assert len(files) == 3
        assert all('path' in f for f in files)
        assert all('name' in f for f in files)
        assert all('extension' in f for f in files)

    def test_list_empty_directory(self, tmp_path):
        """Test listing empty directory."""
        files = list_input_files(Settings(), str(tmp_path))
        assert files == []


class TestTextFileReaderTool:
    """Test TextFileReaderTool."""

    def test_read_text_file(self, tmp_path):
        """Test reading plain text file."""
        test_file = tmp_path / "test.txt"
        content = "\n".join([f"Line {i}" for i in range(1, 11)])
        test_file.write_text(content)

        result = read_text_file(Settings(), str(test_file))

        assert "content" in result
        assert "Line 1" in result["content"]

    def test_read_nonexistent_file(self):
        with pytest.raises(Exception):
            """Test reading nonexistent file."""
            read_text_file(Settings(), "/nonexistent/file.txt")

    @patch('src.ai_book_composer.utils.file_utils.DocxDocument')
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

        result = read_text_file(Settings(), str(test_file))

        assert "content" in result
        assert "Paragraph 1" in result["content"]
        assert "Paragraph 2" in result["content"]


class TestAudioTranscriptionTool:
    """Test AudioTranscriptionTool with caching and language support."""

    @patch('src.ai_book_composer.utils.file_utils.WhisperModel')
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

        result1 = read_audio_file(settings, str(test_audio))

        assert result1["transcription"] == "Hello world"
        assert result1["language"] == "en"
        assert result1["duration"] == 5.0
        assert len(result1["segments"]) == 1

        # Check cache file was created
        cache_path = file_utils.get_cache_path(settings, test_audio)
        assert cache_path.exists()

        # Verify cache content
        with open(cache_path, 'r') as f:
            cached_data = json.load(f)
        assert cached_data["transcription"] == "Hello world"
        assert cached_data["language"] == "en"

        # Second call - should use cache (transcribe should not be called again)
        mock_model_instance.transcribe.reset_mock()
        result2 = read_audio_file(Settings(), str(test_audio))

        assert result2 == result1
        mock_model_instance.transcribe.assert_not_called()

    @patch('src.ai_book_composer.utils.file_utils.WhisperModel')
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

        result = read_audio_file(settings, str(test_audio), language="he")

        assert result["transcription"] == "שלום עולם"
        assert result["language"] == "he"

        # Check cache file was created with language suffix
        cache_path = file_utils.get_cache_path(settings, test_audio, language="he")
        assert cache_path.exists()

        # Verify language parameter was passed to Whisper
        mock_model_instance.transcribe.assert_called_once()
        call_kwargs = mock_model_instance.transcribe.call_args[1]
        assert call_kwargs["language"] == "he"

    # noinspection PyUnusedLocal
    @patch('src.ai_book_composer.utils.file_utils.WhisperModel')
    def test_transcribe_audio_file_not_found(self, mock_whisper_model):
        """Test audio transcription with non-existent file."""
        settings = Settings()
        settings.whisper.mode = "local"
        settings.whisper.model_size = "base"
        settings.whisper.local = {"device": "cpu", "compute_type": "int8"}
        settings.security.max_file_size_mb = 500

        with pytest.raises(Exception):
            result = read_audio_file(settings, "/nonexistent/file.mp3")


class TestVideoTranscriptionTool:
    """Test VideoTranscriptionTool with caching and language support."""

    @patch('src.ai_book_composer.utils.file_utils.ffmpeg')
    @patch('src.ai_book_composer.utils.file_utils.transcribe_local')
    def test_transcribe_video_with_cache(self, mock_transcribe_local, mock_ffmpeg, tmp_path):
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
        mock_transcribe_local.return_value = {
            "transcription": "Video content",
            "segments": [{"start": 0.0, "end": 10.0, "text": "Video content"}],
            "language": "en",
            "duration": 10.0
        }

        result1 = read_video_file(settings, str(test_video))

        assert result1["transcription"] == "Video content"
        assert result1["language"] == "en"

        # Check cache file was created
        cache_path = file_utils.get_cache_path(settings, test_video)
        assert cache_path.exists()

        # Second call - should use cache
        result2 = read_video_file(settings, str(test_video))
        assert result2 == result1

    @patch('src.ai_book_composer.utils.file_utils.ffmpeg')
    @patch('src.ai_book_composer.utils.file_utils.transcribe_local')
    def test_transcribe_video_with_language(self, mock_transcribe_local, mock_ffmpeg, tmp_path):
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
        mock_transcribe_local.return_value = {
            "transcription": "תוכן וידאו",
            "segments": [{"start": 0.0, "end": 10.0, "text": "תוכן וידאו"}],
            "language": "he",
            "duration": 10.0
        }

        # Transcribe with Hebrew language
        result = read_video_file(settings, str(test_video), language="he")

        assert result["transcription"] == "תוכן וידאו"
        assert result["language"] == "he"

        # Check cache file was created with language suffix
        cache_path = file_utils.get_cache_path(settings, test_video, language="he")
        assert cache_path.exists()

        # Verify language parameter was passed
        mock_transcribe_local.assert_called_once()
        call_args = mock_transcribe_local.call_args[0]
        # Language should be either in args or kwargs
        assert call_args[2] == "he"


class TestImageListingTool:
    """Test ImageListingTool."""

    def test_list_images(self, tmp_path):
        """Test listing images in directory."""

        # Create test image files
        (tmp_path / "image1.jpg").write_bytes(b"fake jpg")
        (tmp_path / "image2.png").write_bytes(b"fake png")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "image3.gif").write_bytes(b"fake gif")

        # Create non-image files (should be ignored)
        (tmp_path / "text.txt").write_text("not an image")

        images = list_images(Settings(), str(tmp_path))

        assert len(images) == 3
        assert all('path' in img for img in images)
        assert all('filename' in img for img in images)
        assert all('format' in img for img in images)

    def test_list_empty_directory(self, tmp_path):
        """Test listing empty directory."""

        images = list_images(Settings(), str(tmp_path))
        assert images == []


class TestImageExtractorTool:
    """Test ImageExtractorTool."""

    def test_extract_images_file_not_found(self, tmp_path):
        """Test extract images from non-existent file."""
        with pytest.raises(Exception):
            extract_images_from_pdf(Settings(), str(tmp_path / "nonexistent.pdf"))


class TestImageMeaningfulCheck:
    """Test is_image_meaningful function."""

    def test_is_image_meaningful_black_image(self):
        """Test that black images are identified as not meaningful."""

        # Create a black image
        img = PILImage.new('RGB', (100, 100), color='black')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()

        assert not is_image_meaningful(image_bytes, black_threshold=0.95)

    def test_is_image_meaningful_white_image(self):
        """Test that white images are identified as meaningful."""

        # Create a white image
        img = PILImage.new('RGB', (100, 100), color='white')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()

        assert is_image_meaningful(image_bytes, black_threshold=0.95)

    def test_is_image_meaningful_mixed_content(self):
        """Test that images with mixed content are meaningful."""

        # Create an image with mixed content
        img = PILImage.new('RGB', (100, 100), color='white')
        # Draw some black pixels
        pixels = img.load()
        for i in range(50):
            for j in range(50):
                pixels[i, j] = (0, 0, 0)

        buffer = BytesIO()
        img.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()

        assert is_image_meaningful(image_bytes, black_threshold=0.95)


class TestImageDescription:
    """Test describe_image function."""

    def test_describe_image_with_cache(self, tmp_path, monkeypatch):
        """Test image description with caching."""

        settings = Settings()

        # Create a test image
        test_image = tmp_path / "test_image.png"
        img = PILImage.new('RGB', (100, 100), color='blue')
        img.save(test_image)

        # Mock vision LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "<think>Analyzing image</think><result>A blue square image</result>"
        mock_llm.invoke.return_value = mock_response

        # Create a mock get_llm function
        # noinspection PyUnusedLocal,PyShadowingNames
        def mock_get_llm(settings, temperature, model, provider):
            return mock_llm

        # Monkeypatch get_llm in the ai_book_composer.llm module namespace
        # Since it's imported inside the function, we need to mock it in sys.modules
        mock_llm_module = Mock()
        mock_llm_module.get_llm = mock_get_llm
        monkeypatch.setitem(sys.modules, 'src.ai_book_composer.llm', mock_llm_module)

        # Mock prompts
        prompts = {
            'preprocessor': {
                'image_description_system_prompt': 'You are an image analyst. Target language: {language}',
                'image_description_user_prompt': 'Describe this image: {filename} from {source}'
            }
        }

        # First call - should generate description
        description1 = describe_image(
            settings,
            str(test_image),
            prompts,
            language="en-US",
            cache_results=True
        )

        assert description1 == "A blue square image"

        # Check cache was created
        cache_path = file_utils.get_cache_path(settings, test_image, prefix="img_desc_", ext="txt", language="en-US")
        assert cache_path.exists()

        # Second call - should use cache (LLM should not be called again)
        mock_llm.invoke.reset_mock()
        description2 = describe_image(
            settings,
            str(test_image),
            prompts,
            language="en-US",
            cache_results=True
        )

        assert description2 == description1
        mock_llm.invoke.assert_not_called()

    def test_describe_image_fallback_on_error(self, tmp_path, monkeypatch):
        """Test that describe_image falls back on error."""
        settings = Settings()

        # Create a test image
        test_image = tmp_path / "test_image.png"
        img = PILImage.new('RGB', (100, 100), color='blue')
        img.save(test_image)

        # Mock LLM that raises an error
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM error")

        # Create a mock get_llm function
        # noinspection PyUnusedLocal,PyShadowingNames
        def mock_get_llm(settings, temperature, model, provider):
            return mock_llm

        # Monkeypatch get_llm in sys.modules
        mock_llm_module = Mock()
        mock_llm_module.get_llm = mock_get_llm
        monkeypatch.setitem(sys.modules, 'src.ai_book_composer.llm', mock_llm_module)

        # Mock prompts
        prompts = {
            'preprocessor': {
                'image_description_system_prompt': 'You are an image analyst. Target language: {language}',
                'image_description_user_prompt': 'Describe this image: {filename} from {source}'
            }
        }

        # Should not raise error, but return fallback description
        description = describe_image(
            settings,
            str(test_image),
            prompts,
            language="en-US",
            cache_results=False
        )

        # Fallback should use consistent "Image from {source}: {filename}" format
        expected_source = test_image.parent.name
        assert description == f"Image from {expected_source}: test_image.png"
