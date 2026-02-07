"""Executor agent - Phase 2 of Deep-Agent architecture."""
import logging
from typing import Dict, Any

from .agent_base import AgentBase
from .state import AgentState
from ..config import Settings
from ..progress_display import progress
from ..utils.book_writer import BookWriter

logger = logging.getLogger(__name__)


class WriterAgent(AgentBase):
    """The Writer - writes the final book."""

    def __init__(self, settings: Settings, output_directory: str):
        super().__init__(
            settings,
            output_directory=output_directory
        )

    def write(self, state: AgentState) -> Dict[str, Any]:
        """Write the final book.

        Returns:
            Updated state
        """

        self.state = state

        with progress.agent_context(
                "Writer",
                "Writing the final book content to a file"
        ):
            self._generate_book_inner()

            return {
                "final_output_path": self.state.get("final_output_path", ""),
                "status": "book_generate"
            }

    def _generate_book_inner(self) -> dict[str, int | str | Any]:
        title = self.state.get("book_title", "Composed Book")
        author = self.state.get("book_author", "AI Book Composer")
        chapters = self.state.get("chapters", [])
        references = self.state.get("references", [])

        progress.show_action(f"Generating final book: '{title}' by {author}")
        progress.show_observation(
            f"Compiling {len(chapters)} chapters and {len(references)} references into DOCX format")

        result = BookWriter(self.settings, self.output_directory).run(
            title=title,
            author=author,
            chapters=chapters,
            references=references,
            output_filename="final_book.docx"
        )

        output_path = result.get("file_path")
        progress.show_observation(f"✓ Book generated successfully: {output_path}")

        self.state["final_output_path"] = output_path

        return {
            "final_output_path": output_path,
        }
