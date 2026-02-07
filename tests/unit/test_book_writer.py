"""Unit tests for BookWriter utility."""

from unittest.mock import patch, MagicMock

from src.ai_book_composer.config import Settings
from src.ai_book_composer.utils.book_writer import BookWriter


class TestBookWriterInitialization:
    """Test BookWriter initialization."""

    def test_init_creates_output_directory(self, tmp_path):
        """Test that initialization creates output directory."""
        output_dir = tmp_path / "output"
        settings = Settings()

        writer = BookWriter(settings, str(output_dir))

        assert output_dir.exists()
        assert writer.output_directory == output_dir

    def test_init_with_existing_directory(self, tmp_path):
        """Test initialization with existing directory."""
        output_dir = tmp_path / "existing"
        output_dir.mkdir()
        settings = Settings()

        writer = BookWriter(settings, str(output_dir))

        assert output_dir.exists()
        assert writer.output_directory == output_dir


class TestBookWriterRun:
    """Test BookWriter run method."""

    @patch('src.ai_book_composer.utils.book_writer.Document')
    @patch('src.ai_book_composer.utils.book_writer.Renderer')
    def test_run_basic_book_generation(self, _, mock_document, tmp_path):
        """Test basic book generation without images."""
        settings = Settings()
        writer = BookWriter(settings, str(tmp_path))

        # Setup mocks
        mock_doc_instance = MagicMock()
        mock_document.return_value = mock_doc_instance
        mock_doc_instance.Sections = []
        mock_doc_instance.StyleSheet = MagicMock()

        # Configure StyleSheet
        mock_ss = mock_doc_instance.StyleSheet
        mock_ss.Fonts = MagicMock()
        mock_ss.Fonts.Arial = "Arial"
        mock_ss.Fonts.TimesNewRoman = "Times New Roman"
        mock_ss.ParagraphStyles = MagicMock()
        mock_ss.ParagraphStyles.Heading1 = MagicMock()
        mock_ss.ParagraphStyles.Normal = MagicMock()

        chapters = [
            {
                "title": "Introduction",
                "content": "This is the introduction.\n\nIt has multiple paragraphs."
            },
            {
                "title": "Chapter One",
                "content": "Content of chapter one."
            }
        ]

        references = ["Reference 1", "Reference 2"]

        result = writer.run(
            title="Test Book",
            author="Test Author",
            chapters=chapters,
            references=references,
            output_filename="test.rtf"
        )

        assert result["success"] is True
        assert "test.rtf" in result["file_path"]
        assert result["chapter_count"] == 2
        assert result["reference_count"] == 2

    @patch('src.ai_book_composer.utils.book_writer.Document')
    @patch('src.ai_book_composer.utils.book_writer.Renderer')
    def test_run_with_images(self, _, mock_document, tmp_path):
        """Test book generation with images."""
        settings = Settings()
        writer = BookWriter(settings, str(tmp_path))

        # Create test image
        test_image = tmp_path / "test_image.jpg"
        test_image.write_bytes(b"fake image data")

        # Setup mocks
        mock_doc_instance = MagicMock()
        mock_document.return_value = mock_doc_instance
        mock_doc_instance.Sections = []
        mock_doc_instance.StyleSheet = MagicMock()

        # Configure StyleSheet
        mock_ss = mock_doc_instance.StyleSheet
        mock_ss.Fonts = MagicMock()
        mock_ss.Fonts.Arial = "Arial"
        mock_ss.Fonts.TimesNewRoman = "Times New Roman"
        mock_ss.ParagraphStyles = MagicMock()
        mock_ss.ParagraphStyles.Heading1 = MagicMock()
        mock_ss.ParagraphStyles.Normal = MagicMock()

        chapters = [
            {
                "title": "Chapter with Images",
                "content": "Chapter content.",
                "images": [
                    {
                        "image_path": str(test_image),
                        "position": "start",
                        "reasoning": "Opening image"
                    }
                ]
            }
        ]

        with patch('src.ai_book_composer.utils.book_writer.Image'):
            result = writer.run(
                title="Test Book",
                author="Test Author",
                chapters=chapters,
                references=[],
                output_filename="book_with_images.rtf"
            )

        assert result["success"] is True
        assert result["image_count"] == 1

    @patch('src.ai_book_composer.utils.book_writer.Document')
    @patch('src.ai_book_composer.utils.book_writer.Renderer')
    def test_run_with_images_at_different_positions(self, _, mock_document, tmp_path):
        """Test image placement at start, middle, and end."""
        settings = Settings()
        writer = BookWriter(settings, str(tmp_path))

        # Create test images
        test_images = []
        for i in range(3):
            img = tmp_path / f"image_{i}.jpg"
            img.write_bytes(b"fake image data")
            test_images.append(img)

        # Setup mocks
        mock_doc_instance = MagicMock()
        mock_document.return_value = mock_doc_instance
        mock_doc_instance.Sections = []
        mock_doc_instance.StyleSheet = MagicMock()

        mock_ss = mock_doc_instance.StyleSheet
        mock_ss.Fonts = MagicMock()
        mock_ss.Fonts.Arial = "Arial"
        mock_ss.Fonts.TimesNewRoman = "Times New Roman"
        mock_ss.ParagraphStyles = MagicMock()
        mock_ss.ParagraphStyles.Heading1 = MagicMock()
        mock_ss.ParagraphStyles.Normal = MagicMock()

        chapters = [
            {
                "title": "Chapter",
                "content": "Para 1\n\nPara 2\n\nPara 3\n\nPara 4",
                "images": [
                    {"image_path": str(test_images[0]), "position": "start", "reasoning": "Start image"},
                    {"image_path": str(test_images[1]), "position": "middle", "reasoning": "Middle image"},
                    {"image_path": str(test_images[2]), "position": "end", "reasoning": "End image"}
                ]
            }
        ]

        with patch('src.ai_book_composer.utils.book_writer.Image'):
            result = writer.run(
                title="Test Book",
                author="Test Author",
                chapters=chapters,
                references=[]
            )

        assert result["success"] is True
        assert result["image_count"] == 3

    @patch('src.ai_book_composer.utils.book_writer.Document')
    @patch('src.ai_book_composer.utils.book_writer.Renderer')
    def test_run_empty_chapters(self, _, mock_document, tmp_path):
        """Test book generation with empty chapters list."""
        settings = Settings()
        writer = BookWriter(settings, str(tmp_path))

        mock_doc_instance = MagicMock()
        mock_document.return_value = mock_doc_instance
        mock_doc_instance.Sections = []
        mock_doc_instance.StyleSheet = MagicMock()

        mock_ss = mock_doc_instance.StyleSheet
        mock_ss.Fonts = MagicMock()
        mock_ss.Fonts.Arial = "Arial"
        mock_ss.Fonts.TimesNewRoman = "Times New Roman"
        mock_ss.ParagraphStyles = MagicMock()
        mock_ss.ParagraphStyles.Heading1 = MagicMock()
        mock_ss.ParagraphStyles.Normal = MagicMock()

        result = writer.run(
            title="Empty Book",
            author="Test Author",
            chapters=[],
            references=[]
        )

        assert result["success"] is True
        assert result["chapter_count"] == 0

    @patch('src.ai_book_composer.utils.book_writer.Document')
    def test_run_handles_error(self, mock_document, tmp_path):
        """Test error handling during book generation."""
        settings = Settings()
        writer = BookWriter(settings, str(tmp_path))

        # Force an error
        mock_document.side_effect = Exception("Test error")

        result = writer.run(
            title="Test Book",
            author="Test Author",
            chapters=[{"title": "Chapter", "content": "Content"}],
            references=[]
        )

        assert result["success"] is False
        assert "error" in result
        assert "Test error" in result["error"]


class TestBookWriterImageHandling:
    """Test image handling in BookWriter."""

    def test_add_image_to_section_success(self, tmp_path):
        """Test adding image to section."""
        settings = Settings()
        writer = BookWriter(settings, str(tmp_path))

        # Create test image
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image")

        mock_section = MagicMock()
        img_info = {
            "image_path": str(test_image),
            "reasoning": "Test image"
        }

        with patch('src.ai_book_composer.utils.book_writer.Image') as mock_image, \
                patch('src.ai_book_composer.utils.book_writer.Paragraph') as _:
            writer._add_image_to_section(mock_section, img_info)

            # Verify Image was created
            mock_image.assert_called_once_with(str(test_image))

    def test_add_image_missing_file(self, tmp_path):
        """Test adding image when file doesn't exist."""
        settings = Settings()
        writer = BookWriter(settings, str(tmp_path))

        mock_section = MagicMock()

        img_info = {
            "image_path": "/nonexistent/image.jpg",
            "reasoning": "Missing image"
        }

        # Should not raise an error, just log warning
        writer._add_image_to_section(mock_section, img_info)

        # Section append should not be called
        assert mock_section.append.call_count == 0

    def test_add_image_without_reasoning(self, tmp_path):
        """Test adding image without reasoning/caption."""
        settings = Settings()
        writer = BookWriter(settings, str(tmp_path))

        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image")

        mock_section = MagicMock()

        img_info = {
            "image_path": str(test_image),
            "reasoning": ""  # Empty reasoning
        }

        with patch('src.ai_book_composer.utils.book_writer.Image') as mock_image, \
                patch('src.ai_book_composer.utils.book_writer.Paragraph'):
            writer._add_image_to_section(mock_section, img_info)

            mock_image.assert_called_once()

    def test_add_image_handles_image_error(self, tmp_path):
        """Test handling error when Image constructor fails."""
        settings = Settings()
        writer = BookWriter(settings, str(tmp_path))

        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image")

        mock_section = MagicMock()

        img_info = {
            "image_path": str(test_image),
            "reasoning": "Test"
        }

        with patch('src.ai_book_composer.utils.book_writer.Image', side_effect=Exception("Image error")), \
                patch('src.ai_book_composer.utils.book_writer.Paragraph'):
            # Should not raise, just log warning
            writer._add_image_to_section(mock_section, img_info)


class TestBookWriterIntegration:
    """Integration tests for BookWriter."""

    @patch('src.ai_book_composer.utils.book_writer.Document')
    @patch('src.ai_book_composer.utils.book_writer.Renderer')
    def test_full_book_with_all_features(self, _, mock_document, tmp_path):
        """Test complete book generation with all features."""
        settings = Settings()
        writer = BookWriter(settings, str(tmp_path))

        # Setup mocks
        mock_doc_instance = MagicMock()
        mock_document.return_value = mock_doc_instance
        mock_doc_instance.Sections = []
        mock_doc_instance.StyleSheet = MagicMock()

        mock_ss = mock_doc_instance.StyleSheet
        mock_ss.Fonts = MagicMock()
        mock_ss.Fonts.Arial = "Arial"
        mock_ss.Fonts.TimesNewRoman = "Times New Roman"
        mock_ss.ParagraphStyles = MagicMock()
        mock_ss.ParagraphStyles.Heading1 = MagicMock()
        mock_ss.ParagraphStyles.Normal = MagicMock()

        # Create test images
        images = []
        for i in range(2):
            img = tmp_path / f"img{i}.jpg"
            img.write_bytes(b"data")
            images.append(img)

        chapters = [
            {
                "title": "First Chapter",
                "content": "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
                "images": [
                    {"image_path": str(images[0]), "position": "start", "reasoning": "Opening image"}
                ]
            },
            {
                "title": "Second Chapter",
                "content": "Content here.",
                "images": []
            },
            {
                "title": "Final Chapter",
                "content": "Final content.",
                "images": [
                    {"image_path": str(images[1]), "position": "end", "reasoning": "Closing image"}
                ]
            }
        ]

        references = [
            "Smith, J. (2020). Example Reference.",
            "Doe, J. (2021). Another Reference."
        ]

        with patch('src.ai_book_composer.utils.book_writer.Image'):
            result = writer.run(
                title="Complete Test Book",
                author="John Doe",
                chapters=chapters,
                references=references,
                output_filename="complete_book.rtf"
            )

        assert result["success"] is True
        assert result["chapter_count"] == 3
        assert result["reference_count"] == 2
        assert result["image_count"] == 2
        assert "complete_book.rtf" in result["file_path"]
