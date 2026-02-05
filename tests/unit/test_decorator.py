"""Unit tests for DecoratorAgent."""

import json
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest
from tenacity import RetryError

from src.ai_book_composer.agents.decorator import DecoratorAgent
from src.ai_book_composer.agents.state import create_initial_state
from src.ai_book_composer.config import Settings

def _create_decorator_no_cache(settings: Optional[Settings] = None) -> DecoratorAgent:
    if settings is None:
        settings = Settings()
    settings.book.use_cached_decorations = False
    agent = DecoratorAgent(settings)
    return agent


class TestDecoratorAgentFormatImages:
    """Test image formatting for prompts."""

    def test_format_images_for_prompt_empty(self):
        """Test formatting empty image list."""
        result = _create_decorator_no_cache()._format_images_for_prompt([])

        assert result == "No images available."

    def test_format_images_for_prompt_single_image(self):
        """Test formatting single image."""
        images = [
            {
                "filename": "photo.jpg",
                "source_file": "documents/article.pdf",
                "format": "jpeg",
                "description": "A photo of a sunset"
            }
        ]

        result = _create_decorator_no_cache()._format_images_for_prompt(images)

        assert "photo.jpg" in result
        assert "A photo of a sunset" in result

    def test_path_extraction(self):
        """Test formatting single image."""
        images = [
            {
                "path": "/folder/subfolder/photo.jpg",
                "description": "A photo of a sunset"
            }
        ]

        result = _create_decorator_no_cache()._format_images_for_prompt(images)

        assert "folder" not in result
        assert "subfolder" not in result
        assert "photo.jpg" in result
        assert "A photo of a sunset" in result

    def test_relative_path_extraction(self):
        """Test formatting single image."""
        settings = Settings()
        cache_folder = settings.general.cache_dir

        images = [
            {
                "path": f"{Path(cache_folder) / 'subfolder' / 'photo.jpg'}",
                "description": "A photo of a sunset"
            }
        ]

        result = _create_decorator_no_cache()._format_images_for_prompt(images)

        assert cache_folder not in result
        assert "subfolder" in result
        assert "photo.jpg" in result
        assert "A photo of a sunset" in result

    def test_format_images_for_prompt_multiple_images(self):
        """Test formatting multiple images."""
        images = [
            {"filename": "image1.png", "source_file": "input", "format": "png", "description": "First image"},
            {"filename": "image2.jpg", "source_file": "doc.pdf", "format": "jpeg", "description": "Second image"},
            {"filename": "image3.gif", "source_file": "input", "format": "gif", "description": "Third image"}
        ]

        result = _create_decorator_no_cache()._format_images_for_prompt(images)

        assert "image1.png" in result
        assert "image2.jpg" in result
        assert "image3.gif" in result
        assert "First image" in result
        assert "Second image" in result
        assert "Third image" in result

    def test_format_images_handles_missing_fields(self):
        """Test formatting when fields are missing."""
        images = [
            {"description": "Not-there"},  # no filename
            {"filename": "test.jpg", "description": "A test image"}  # Has description
        ]

        result = _create_decorator_no_cache()._format_images_for_prompt(images)

        assert "Not-there" not in result
        assert "test.jpg" in result
        assert "A test image" in result


class TestDecoratorAgentDecorate:
    """Test DecoratorAgent decorate method."""

    @patch.object(DecoratorAgent, '_invoke_agent')
    def test_decorate_no_images(self, mock_invoke, tmp_path):
        """Test decoration when no images are available."""
        agent = _create_decorator_no_cache()

        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        state["chapters"] = [
            {"title": "Chapter 1", "content": "Content"}
        ]
        state["images"] = []

        result = agent.decorate(state)

        # Should return state unchanged when no images
        assert result == state
        mock_invoke.assert_not_called()

    def test_decorate_no_chapters_raises_error(self, tmp_path):
        """Test decoration raises error when no chapters."""
        agent = _create_decorator_no_cache()

        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        state["chapters"] = []
        state["images"] = [{"filename": "test.jpg", "path": "/path/to/image"}]

        with pytest.raises(Exception, match="No chapters found"):
            agent.decorate(state)

    @patch.object(DecoratorAgent, '_invoke_agent')
    def test_decorate_with_images(self, mock_invoke, tmp_path):
        """Test successful decoration with images."""
        agent = _create_decorator_no_cache()

        # Mock LLM response
        mock_response = json.dumps({
            "image_placements": [
                {
                    "image_path": "/path/to/image1.jpg",
                    "position": "start",
                    "reasoning": "Good opening image"
                }
            ]
        })
        mock_invoke.return_value = mock_response

        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        state["chapters"] = [
            {"title": "Chapter 1", "content": "Content of chapter one"}
        ]
        state["images"] = [
            {"filename": "image1.jpg", "path": "/path/to/image1.jpg", "format": "jpeg"}
        ]

        result = agent.decorate(state)

        assert result["status"] == "decorated"
        assert len(result["chapters"]) == 1
        assert "images" in result["chapters"][0]
        assert len(result["chapters"][0]["images"]) == 1
        mock_invoke.assert_called_once()

    @patch.object(DecoratorAgent, '_invoke_agent')
    def test_decorate_multiple_chapters(self, mock_invoke, tmp_path):
        """Test decoration with multiple chapters."""
        agent = _create_decorator_no_cache()

        # Mock LLM responses for each chapter
        mock_responses = [
            json.dumps([{"image_path": "/path/img1.jpg", "position": "start"}]),
            json.dumps([{"image_path": "/path/img2.jpg", "position": "middle"}])
        ]
        mock_invoke.side_effect = mock_responses

        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        state["chapters"] = [
            {"title": "Chapter 1", "content": "First chapter"},
            {"title": "Chapter 2", "content": "Second chapter"}
        ]
        state["images"] = [
            {"filename": "img1.jpg", "path": "/path/img1.jpg", "format": "jpeg"},
            {"filename": "img2.jpg", "path": "/path/img2.jpg", "format": "jpeg"}
        ]

        result = agent.decorate(state)

        assert len(result["chapters"]) == 2
        assert mock_invoke.call_count == 2


class TestDecoratorAgentGetImagePlacements:
    """Test image placement retrieval from LLM."""

    @patch.object(DecoratorAgent, '_invoke_agent')
    def test_get_image_placements_json_response(self, mock_invoke, tmp_path):
        """Test parsing JSON response from LLM."""
        agent = _create_decorator_no_cache()
        agent.state = {}

        mock_response = json.dumps([
            {
                "image_path": "/path/to/image.jpg",
                "position": "start",
                "reasoning": "Great opener"
            }
        ])
        mock_invoke.return_value = mock_response

        all_images = [{"path": "/path/to/image.jpg", "filename": "image.jpg"}]

        result = agent._get_image_placements(
            chapter_number=1,
            chapter_title="Test",
            chapter_content_preview="Content",
            available_images="1. image.jpg",
            all_images=all_images,
            language="en-US"
        )

        assert len(result) == 1
        assert result[0]["image_path"] == "/path/to/image.jpg"
        assert result[0]["position"] == "start"

    @patch.object(DecoratorAgent, '_invoke_agent')
    def test_get_image_placements_json_with_code_block(self, mock_invoke, tmp_path):
        """Test parsing JSON inside code block."""
        agent = _create_decorator_no_cache()
        agent.state = {}

        mock_response = """Here's the placement:
```json
[
    {
        "image_path": "/path/to/image.jpg",
        "position": "middle",
        "reasoning": "Relevant image"
    }
]
    
```
"""
        mock_invoke.return_value = mock_response

        all_images = [{"path": "/path/to/image.jpg", "filename": "image.jpg"}]

        result = agent._get_image_placements(
            chapter_number=1,
            chapter_title="Test",
            chapter_content_preview="Content",
            available_images="1. image.jpg",
            all_images=all_images,
            language="en-US"
        )

        assert len(result) == 1
        assert result[0]["position"] == "middle"

    @patch.object(DecoratorAgent, '_invoke_agent')
    def test_get_image_placements_validates_paths(self, mock_invoke, tmp_path):
        """Test that invalid image paths are filtered out."""
        agent = _create_decorator_no_cache()
        agent.state = {}

        mock_response = json.dumps([
            {"image_path": "/valid/image.jpg", "position": "start"},
            {"image_path": "/invalid/image.jpg", "position": "end"}  # Not in available images
        ])
        mock_invoke.return_value = mock_response

        all_images = [{"path": "/valid/image.jpg", "filename": "image.jpg"}]

        result = agent._get_image_placements(
            chapter_number=1,
            chapter_title="Test",
            chapter_content_preview="Content",
            available_images="1. image.jpg",
            all_images=all_images,
            language="en-US"
        )

        # Only valid image should be in result
        assert len(result) == 1
        assert result[0]["image_path"] == "/valid/image.jpg"

    @patch.object(DecoratorAgent, '_invoke_agent')
    def test_get_image_placements_respects_max_images(self, mock_invoke, tmp_path):
        """Test that max images per chapter limit is enforced."""
        settings = Settings()
        settings.image_processing.max_images_per_chapter = 2
        agent = _create_decorator_no_cache(settings)
        agent.state = {}

        mock_response = json.dumps([
            {"image_path": "/img1.jpg", "position": "start"},
            {"image_path": "/img2.jpg", "position": "middle"},
            {"image_path": "/img3.jpg", "position": "end"}
        ])
        mock_invoke.return_value = mock_response

        all_images = [
            {"path": "/img1.jpg", "filename": "img1.jpg"},
            {"path": "/img2.jpg", "filename": "img2.jpg"},
            {"path": "/img3.jpg", "filename": "img3.jpg"}
        ]

        result = agent._get_image_placements(
            chapter_number=1,
            chapter_title="Test",
            chapter_content_preview="Content",
            available_images="images",
            all_images=all_images,
            language="en-US"
        )

        # Should be limited to max_images_per_chapter
        assert len(result) == 2

    @patch.object(DecoratorAgent, '_invoke_agent')
    def test_get_image_placements_invalid_json_raises(self, mock_invoke, tmp_path):
        """Test that invalid JSON raises exception."""
        agent = _create_decorator_no_cache()
        agent.state = {}

        mock_response = "This is not valid JSON at all"
        mock_invoke.return_value = mock_response

        all_images = [{"path": "/img.jpg", "filename": "img.jpg"}]

        with pytest.raises(RetryError):
            agent._get_image_placements(
                chapter_number=1,
                chapter_title="Test",
                chapter_content_preview="Content",
                available_images="1. img.jpg",
                all_images=all_images,
                language="en-US"
            )

    @patch.object(DecoratorAgent, '_invoke_agent')
    def test_get_image_placements_with_style_instructions(self, mock_invoke, tmp_path):
        """Test that style instructions are included in prompt."""
        agent = _create_decorator_no_cache()
        agent.state = {}

        mock_response = json.dumps({"image_placements": []})
        mock_invoke.return_value = mock_response

        all_images = []

        agent._get_image_placements(
            chapter_number=1,
            chapter_title="Test",
            chapter_content_preview="Content",
            available_images="",
            all_images=all_images,
            language="en-US",
            style_instructions="Academic style with formal images"
        )

        # Verify style instructions were passed to invoke
        call_args = mock_invoke.call_args[0]
        system_prompt = call_args[0]
        assert "Academic style with formal images" in system_prompt

    @patch.object(DecoratorAgent, '_invoke_agent')
    def test_get_image_placements_error_propagates(self, mock_invoke, tmp_path):
        """Test that errors from LLM are propagated."""
        agent = _create_decorator_no_cache()
        agent.state = {}

        mock_invoke.side_effect = Exception("LLM error")

        with pytest.raises(RetryError):
            agent._get_image_placements(
                chapter_number=1,
                chapter_title="Test",
                chapter_content_preview="Content",
                available_images="",
                all_images=[],
                language="en-US"
            )


class TestDecoratorAgentIntegration:
    """Integration tests for DecoratorAgent."""

    @patch.object(DecoratorAgent, '_invoke_agent')
    def test_full_decoration_workflow(self, mock_invoke, tmp_path):
        """Test complete decoration workflow."""
        agent = _create_decorator_no_cache()

        # Mock LLM responses
        mock_responses = [
            json.dumps([
                {"image_path": "/img1.jpg", "position": "start", "reasoning": "Introduction"}
            ]),
            json.dumps([
                {"image_path": "/img2.jpg", "position": "middle", "reasoning": "Illustration"},
                {"image_path": "/img3.jpg", "position": "end", "reasoning": "Summary"}
            ])
        ]
        mock_invoke.side_effect = mock_responses

        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        state["chapters"] = [
            {"title": "Introduction", "content": "Intro content"},
            {"title": "Main Chapter", "content": "Main content goes here"}
        ]
        state["images"] = [
            {"path": "/img1.jpg", "filename": "img1.jpg", "format": "jpeg"},
            {"path": "/img2.jpg", "filename": "img2.jpg", "format": "png"},
            {"path": "/img3.jpg", "filename": "img3.jpg", "format": "jpeg"}
        ]
        state["language"] = "en-US"
        state["style_instructions"] = "Professional book"

        result = agent.decorate(state)

        assert result["status"] == "decorated"
        assert len(result["chapters"]) == 2
        assert len(result["chapters"][0]["images"]) == 1
        assert len(result["chapters"][1]["images"]) == 2
        assert result["chapters"][0]["title"] == "Introduction"
        assert result["chapters"][1]["title"] == "Main Chapter"