"""Decorator agent - decides where to place images in chapters."""

import json
import logging
from typing import Dict, Any, List

from .agent_base import AgentBase
from .state import AgentState
from ..config import Settings
from ..progress_display import progress

logger = logging.getLogger(__name__)


class DecoratorAgent(AgentBase):
    """The Decorator - decides on image placements in chapters."""

    def __init__(self, settings: Settings):
        super().__init__(settings=settings, llm_temperature=0.3)

    def decorate(self, state: AgentState) -> Dict[str, Any]:
        """Add image placements to chapters.
        
        Returns:
            Updated state with image-decorated chapters
        """
        self.state = state

        progress.show_action("Decorator: Analyzing image placements...")

        chapters = state.get("chapters", [])
        images = state.get("images", [])
        language = state.get("language", "en-US")
        style_instructions = state.get("style_instructions", "")

        if not images:
            progress.show_observation("Decorator: No images available, skipping decoration")
            # noinspection PyTypeChecker
            return state

        if not chapters:
            raise Exception("Decorator: No chapters found in state to decorate.")

        # Decorate each chapter
        decorated_chapters = []
        for i, chapter in enumerate(chapters):
            chapter_number = i + 1
            chapter_title = chapter.get("title", f"Chapter {chapter_number}")
            chapter_content = chapter.get("content", "")

            # Get content preview (first 1000 characters for analysis)
            content_preview = chapter_content[:1000] + ("..." if len(chapter_content) > 1000 else "")

            # Format available images for the prompt
            images_summary = self._format_images_for_prompt(images)

            # Get image placement suggestions from LLM
            placements = self._get_image_placements(
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                chapter_content_preview=content_preview,
                available_images=images_summary,
                all_images=images,
                language=language,
                style_instructions=style_instructions
            )

            # Create decorated chapter with image placements
            decorated_chapter = {
                "title": chapter_title,
                "content": chapter_content,
                "images": placements
            }
            decorated_chapters.append(decorated_chapter)

            progress.show_observation(f"Decorator: Added {len(placements)} images to chapter {chapter_number}")

        self.state["chapters"] = decorated_chapters

        return {
            "chapters": decorated_chapters,
            "status": "decorated"
        }

    @staticmethod
    def _format_images_for_prompt(images: List[Dict[str, Any]]) -> str:
        """Format image list for the prompt.
        
        Args:
            images: List of image dictionaries
            
        Returns:
            Formatted string describing available images
        """
        if not images:
            return "No images available."

        image_descriptions = []
        for i, img in enumerate(images, 1):
            filename = img.get("filename", "unknown")
            source = img.get("source_file", "input directory")
            format_type = img.get("format", "unknown")
            description = img.get("description", "No description available")

            desc = f"{i}. {filename} (format: {format_type}, source: {source})\n   Description: {description}"
            image_descriptions.append(desc)

        return "\n".join(image_descriptions)

    def _get_image_placements(
            self,
            chapter_number: int,
            chapter_title: str,
            chapter_content_preview: str,
            available_images: str,
            all_images: List[Dict[str, Any]],
            language: str,
            style_instructions: str = ""
    ) -> List[Dict[str, Any]]:
        """Get image placement suggestions from LLM.
        
        Args:
            chapter_number: Chapter number
            chapter_title: Chapter title
            chapter_content_preview: Preview of chapter content
            available_images: Formatted list of available images
            all_images: List of all available image dictionaries
            language: Target language
            style_instructions: Style instructions for the book
            
        Returns:
            List of image placement dictionaries
        """
        try:
            decorator_prompts = self.prompts.get("decorator", {})

            # Format style instructions section
            style_instructions_section = ""
            if style_instructions:
                style_instructions_section = f"Style Instructions: {style_instructions}\nConsider this style when selecting and placing images."

            system_prompt = decorator_prompts.get("system_prompt", "").format(
                language=language,
                max_images_per_chapter=self.settings.image_processing.max_images_per_chapter,
                style_instructions_section=style_instructions_section
            )
            user_prompt = decorator_prompts.get("user_prompt", "").format(
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                chapter_content_preview=chapter_content_preview,
                available_images=available_images
            )

            response_text = self._invoke_agent(system_prompt, user_prompt, self.state)

            # Try to parse JSON response
            try:
                # Look for JSON in the response
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    json_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    json_text = response_text[json_start:json_end].strip()
                else:
                    json_text = response_text.strip()

                placements = json.loads(json_text)

                if isinstance(placements, dict):
                    placements = placements.get('image_placements', [])

                # Validate that image paths in placements exist in available images
                available_image_paths = {img.get("path") for img in all_images}
                validated_placements = []
                for placement in placements:
                    image_path = placement.get("image_path", "")
                    if image_path in available_image_paths:
                        validated_placements.append(placement)
                    else:
                        # Log warning but don't fail - LLM might have made an error
                        progress.show_observation(f"Warning: Image path not found in available images: {image_path}")

                # Limit to max images per chapter
                max_images = self.settings.image_processing.max_images_per_chapter
                if len(validated_placements) > max_images:
                    validated_placements = validated_placements[:max_images]

                return validated_placements
            except json.JSONDecodeError as e:
                logger.exception("Failed to parse decorator response as JSON")
                progress.show_observation(f"Warning: Could not parse decorator response as JSON: {e}")
                raise

        except Exception as e:
            logger.exception("Error getting image placements")
            progress.show_observation(f"Error getting image placements: {e}")
            raise
