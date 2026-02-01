"""Unit tests for PreprocessAgent."""

from unittest.mock import Mock, patch

from src.ai_book_composer.agents.preprocess_agent import PreprocessAgent
from src.ai_book_composer.agents.state import create_initial_state
from src.ai_book_composer.config import Settings


class TestPreprocessAgentInitialization:
    """Test PreprocessAgent initialization."""

    def test_init_creates_agent(self, tmp_path):
        """Test that PreprocessAgent can be initialized."""
        settings = Settings()
        input_dir = str(tmp_path / "input")
        output_dir = str(tmp_path / "output")
        
        agent = PreprocessAgent(settings, input_dir, output_dir)
        
        assert agent.settings == settings
        assert agent.input_directory == input_dir
        assert agent.output_directory == output_dir


class TestPreprocessAgentListFiles:
    """Test file listing functionality."""

    @patch('src.ai_book_composer.agents.preprocess_agent.list_input_files')
    def test_list_files_success(self, mock_list_files, tmp_path):
        """Test listing files successfully."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        agent.state = {}  # Initialize state
        
        mock_files = [
            {"path": "/path/file1.txt", "name": "file1.txt", "extension": ".txt"},
            {"path": "/path/file2.md", "name": "file2.md", "extension": ".md"}
        ]
        mock_list_files.return_value = mock_files
        
        result = agent.list_files()
        
        assert result == mock_files
        assert agent.state['files'] == mock_files
        mock_list_files.assert_called_once_with(settings, str(tmp_path))

    @patch('src.ai_book_composer.agents.preprocess_agent.list_input_files')
    def test_list_files_empty_directory(self, mock_list_files, tmp_path):
        """Test listing files in empty directory."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        agent.state = {}  # Initialize state
        
        mock_list_files.return_value = []
        
        result = agent.list_files()
        
        assert result == []
        assert agent.state['files'] == []


class TestPreprocessAgentProcessSingleFile:
    """Test processing individual files."""

    @patch('src.ai_book_composer.agents.preprocess_agent.read_text_file')
    def test_process_text_file(self, mock_read_text, tmp_path):
        """Test processing text file."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        agent.state = {"language": "en-US"}
        
        mock_read_text.return_value = {"content": "File content here"}
        
        file_info = {
            "path": "/path/to/file.txt",
            "name": "file.txt",
            "extension": ".txt"
        }
        
        result = agent._process_single_file(file_info)
        
        assert result["status"] == "success"
        assert result["type"] == "text"
        assert result["content"] == "File content here"
        assert result["file_name"] == "file.txt"

    @patch('src.ai_book_composer.agents.preprocess_agent.read_text_file')
    def test_process_markdown_file(self, mock_read_text, tmp_path):
        """Test processing markdown file."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        agent.state = {"language": "en-US"}
        
        mock_read_text.return_value = {"content": "# Markdown content"}
        
        file_info = {
            "path": "/path/to/file.md",
            "name": "file.md",
            "extension": ".md"
        }
        
        result = agent._process_single_file(file_info)
        
        assert result["status"] == "success"
        assert result["type"] == "text"
        assert "Markdown" in result["content"]

    @patch('src.ai_book_composer.agents.preprocess_agent.read_text_file')
    def test_process_docx_file(self, mock_read_text, tmp_path):
        """Test processing DOCX file."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        agent.state = {"language": "en-US"}
        
        mock_read_text.return_value = {"content": "Word document content"}
        
        file_info = {
            "path": "/path/to/document.docx",
            "name": "document.docx",
            "extension": ".docx"
        }
        
        result = agent._process_single_file(file_info)
        
        assert result["status"] == "success"
        assert result["type"] == "text"

    @patch('src.ai_book_composer.agents.preprocess_agent.read_audio_file')
    def test_process_audio_file(self, mock_read_audio, tmp_path):
        """Test processing audio file."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        agent.state = {"language": "en-US"}
        
        mock_read_audio.return_value = {"transcription": "Audio transcription"}
        
        file_info = {
            "path": "/path/to/audio.mp3",
            "name": "audio.mp3",
            "extension": ".mp3"
        }
        
        result = agent._process_single_file(file_info)
        
        assert result["status"] == "success"
        assert result["type"] == "audio_transcription"
        assert result["content"] == "Audio transcription"

    @patch('src.ai_book_composer.agents.preprocess_agent.read_video_file')
    def test_process_video_file(self, mock_read_video, tmp_path):
        """Test processing video file."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        agent.state = {"language": "en-US"}
        
        mock_read_video.return_value = {"transcription": "Video transcription"}
        
        file_info = {
            "path": "/path/to/video.mp4",
            "name": "video.mp4",
            "extension": ".mp4"
        }
        
        result = agent._process_single_file(file_info)
        
        assert result["status"] == "success"
        assert result["type"] == "video_transcription"
        assert result["content"] == "Video transcription"

    def test_process_unsupported_file(self, tmp_path):
        """Test processing unsupported file type."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        agent.state = {"language": "en-US"}
        
        file_info = {
            "path": "/path/to/file.xyz",
            "name": "file.xyz",
            "extension": ".xyz"
        }
        
        result = agent._process_single_file(file_info)
        
        assert result["status"] == "skipped"
        assert result["type"] == "unsupported"
        assert ".xyz" in result["content"]

    @patch('src.ai_book_composer.agents.preprocess_agent.read_text_file')
    def test_process_file_with_error(self, mock_read_text, tmp_path):
        """Test processing file that raises error."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        agent.state = {"language": "en-US"}
        
        mock_read_text.side_effect = Exception("Read error")
        
        file_info = {
            "path": "/path/to/file.txt",
            "name": "file.txt",
            "extension": ".txt"
        }
        
        result = agent._process_single_file(file_info)
        
        assert result["status"] == "error"
        assert result["type"] == "error"
        assert "Read error" in result["content"]


class TestPreprocessAgentExtractImagesFromPDF:
    """Test PDF image extraction."""

    @patch('src.ai_book_composer.agents.preprocess_agent.extract_images_from_pdf')
    def test_extract_images_success(self, mock_extract, tmp_path):
        """Test successful image extraction from PDF."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        
        mock_extract.return_value = {
            "success": True,
            "images": [
                {"path": "/img1.jpg", "filename": "img1.jpg"},
                {"path": "/img2.png", "filename": "img2.png"}
            ]
        }
        
        pdf_file = {
            "path": "/path/to/document.pdf",
            "name": "document.pdf"
        }
        
        result = agent._extract_images_from_single_pdf(pdf_file)
        
        assert result["status"] == "success"
        assert len(result["images"]) == 2
        assert result["pdf_name"] == "document.pdf"

    @patch('src.ai_book_composer.agents.preprocess_agent.extract_images_from_pdf')
    def test_extract_images_no_images(self, mock_extract, tmp_path):
        """Test PDF with no images."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        
        mock_extract.return_value = {
            "success": False,
            "images": []
        }
        
        pdf_file = {
            "path": "/path/to/document.pdf",
            "name": "document.pdf"
        }
        
        result = agent._extract_images_from_single_pdf(pdf_file)
        
        assert result["status"] == "no_images"
        assert result["images"] == []

    @patch('src.ai_book_composer.agents.preprocess_agent.extract_images_from_pdf')
    def test_extract_images_with_error(self, mock_extract, tmp_path):
        """Test image extraction error handling."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        
        mock_extract.side_effect = Exception("Extraction failed")
        
        pdf_file = {
            "path": "/path/to/document.pdf",
            "name": "document.pdf"
        }
        
        result = agent._extract_images_from_single_pdf(pdf_file)
        
        assert result["status"] == "error"
        assert "Extraction failed" in result["error"]
        assert result["images"] == []


class TestPreprocessAgentSummarization:
    """Test file summarization functionality."""

    @patch('src.ai_book_composer.agents.preprocess_agent.write_cache')
    @patch('src.ai_book_composer.agents.preprocess_agent.file_utils.read_cache')
    def test_summarize_short_content(self, mock_read_cache, mock_write_cache, tmp_path):
        """Test that short content is not summarized."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        agent.state = {"language": "en-US"}
        
        mock_read_cache.return_value = None
        
        gathered_file = {
            "name": "short.txt",
            "path": "/path/short.txt",
            "content": "Short content less than 2000 chars"
        }
        
        result = agent._summerize_gathered_file(gathered_file)
        
        # Short content should be used as-is, not summarized
        assert result["summary"] == "Short content less than 2000 chars"

    @patch('src.ai_book_composer.agents.preprocess_agent.write_cache')
    @patch('src.ai_book_composer.agents.preprocess_agent.file_utils.read_cache')
    @patch.object(PreprocessAgent, '_invoke_llm')
    def test_summarize_long_content(self, mock_invoke_llm, mock_read_cache, mock_write_cache, tmp_path):
        """Test summarization of long content."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        agent.state = {"language": "en-US"}
        
        mock_read_cache.return_value = None
        mock_invoke_llm.return_value = "Summarized content"
        
        long_content = "x" * 3000  # Content over 2000 chars
        gathered_file = {
            "name": "long.txt",
            "path": "/path/long.txt",
            "content": long_content
        }
        
        result = agent._summerize_gathered_file(gathered_file)
        
        assert result["summary"] == "Summarized content"
        mock_invoke_llm.assert_called_once()

    @patch('src.ai_book_composer.agents.preprocess_agent.write_cache')
    @patch('src.ai_book_composer.agents.preprocess_agent.file_utils.read_cache')
    def test_summarize_uses_cache(self, mock_read_cache, mock_write_cache, tmp_path):
        """Test that cached summary is used when available."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        agent.state = {"language": "en-US"}
        
        mock_read_cache.return_value = "Cached summary"
        
        gathered_file = {
            "name": "file.txt",
            "path": "/path/file.txt",
            "content": "x" * 3000
        }
        
        result = agent._summerize_gathered_file(gathered_file)
        
        # Should use cached summary
        assert result["summary"] == "Cached summary"

    @patch('src.ai_book_composer.agents.preprocess_agent.write_cache')
    @patch('src.ai_book_composer.agents.preprocess_agent.file_utils.read_cache')
    @patch.object(PreprocessAgent, '_invoke_llm')
    def test_summarize_handles_llm_error(self, mock_invoke_llm, mock_read_cache, mock_write_cache, tmp_path):
        """Test error handling in summarization."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        agent.state = {"language": "en-US"}
        
        mock_read_cache.return_value = None
        mock_invoke_llm.side_effect = Exception("LLM error")
        
        long_content = "x" * 3000
        gathered_file = {
            "name": "file.txt",
            "path": "/path/file.txt",
            "content": long_content
        }
        
        result = agent._summerize_gathered_file(gathered_file)
        
        # Should fall back to first 2000 chars on error
        assert len(result["summary"]) == 2000


class TestPreprocessAgentGatherImages:
    """Test image gathering functionality."""

    @patch('src.ai_book_composer.agents.preprocess_agent.list_images')
    def test_gather_images_from_directory(self, mock_list_images, tmp_path):
        """Test gathering existing images from directory."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        
        mock_images = [
            {"path": "/img1.jpg", "filename": "img1.jpg"},
            {"path": "/img2.png", "filename": "img2.png"}
        ]
        mock_list_images.return_value = mock_images
        
        result = agent._gather_images([])
        
        assert len(result) == 2
        mock_list_images.assert_called_once()

    @patch('src.ai_book_composer.agents.preprocess_agent.list_images')
    @patch('src.ai_book_composer.agents.preprocess_agent.execute_parallel')
    def test_gather_images_from_pdfs(self, mock_execute_parallel, mock_list_images, tmp_path):
        """Test extracting images from PDFs."""
        settings = Settings()
        settings.image_processing.extract_from_pdf = True
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        
        mock_list_images.return_value = []
        mock_execute_parallel.return_value = [
            {
                "pdf_name": "doc.pdf",
                "status": "success",
                "images": [{"path": "/extracted.jpg", "filename": "extracted.jpg"}]
            }
        ]
        
        files = [
            {"path": "/doc.pdf", "name": "doc.pdf", "extension": ".pdf"}
        ]
        
        result = agent._gather_images(files)
        
        assert len(result) == 1
        assert result[0]["filename"] == "extracted.jpg"

    @patch('src.ai_book_composer.agents.preprocess_agent.list_images')
    def test_gather_images_handles_error(self, mock_list_images, tmp_path):
        """Test error handling in image gathering."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        
        mock_list_images.side_effect = Exception("List error")
        
        result = agent._gather_images([])
        
        # Should return empty list on error, not crash
        assert result == []


class TestPreprocessAgentGatherContent:
    """Test content gathering functionality."""

    @patch('src.ai_book_composer.agents.preprocess_agent.execute_parallel')
    def test_gather_all_content(self, mock_execute_parallel, tmp_path):
        """Test gathering content from all files."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        
        mock_execute_parallel.return_value = [
            {
                "file_path": "/file1.txt",
                "file_name": "file1.txt",
                "status": "success",
                "content": "Content 1",
                "type": "text"
            },
            {
                "file_path": "/file2.txt",
                "file_name": "file2.txt",
                "status": "success",
                "content": "Content 2",
                "type": "text"
            }
        ]
        
        files = [
            {"path": "/file1.txt", "name": "file1.txt"},
            {"path": "/file2.txt", "name": "file2.txt"}
        ]
        
        result = agent._gather_all_content(files)
        
        assert len(result) == 2
        assert "/file1.txt" in result
        assert result["/file1.txt"]["content"] == "Content 1"

    @patch('src.ai_book_composer.agents.preprocess_agent.execute_parallel')
    def test_gather_content_with_errors(self, mock_execute_parallel, tmp_path):
        """Test gathering content when some files have errors."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        
        mock_execute_parallel.return_value = [
            {
                "file_path": "/good.txt",
                "file_name": "good.txt",
                "status": "success",
                "content": "Good content",
                "type": "text"
            },
            {
                "file_path": "/bad.txt",
                "file_name": "bad.txt",
                "status": "error",
                "content": "Error: Failed to read",
                "type": "error"
            }
        ]
        
        files = [
            {"path": "/good.txt", "name": "good.txt"},
            {"path": "/bad.txt", "name": "bad.txt"}
        ]
        
        result = agent._gather_all_content(files)
        
        assert len(result) == 2
        assert result["/good.txt"]["content"] == "Good content"
        assert "Error" in result["/bad.txt"]["content"]


class TestPreprocessAgentPreprocess:
    """Test main preprocess method."""

    @patch.object(PreprocessAgent, 'gather_content')
    def test_preprocess_success(self, mock_gather_content, tmp_path):
        """Test successful preprocessing."""
        settings = Settings()
        agent = PreprocessAgent(settings, str(tmp_path), str(tmp_path))
        
        mock_gather_content.return_value = {
            "gathered_content": {"file.txt": {"content": "data"}},
            "images": [{"path": "/img.jpg"}]
        }
        
        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        
        result = agent.preprocess(state)
        
        assert result["status"] == "preprocessed"
        assert "gathered_content" in result
        assert "images" in result
        mock_gather_content.assert_called_once()
