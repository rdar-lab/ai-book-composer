"""Critic agent - Phase 3 of Deep-Agent architecture."""

from typing import Dict, Any

from .agent_base import AgentBase
from .state import AgentState
from ..config import Settings
from ..progress_display import progress

# Constants
CHAPTER_PREVIEW_LENGTH = 200


class CriticAgent(AgentBase):
    """The Critic - validates quality and provides feedback."""

    def __init__(self, settings: Settings, quality_threshold: float = 0.7):
        super().__init__(settings=settings, llm_temperature=0.2)
        self.quality_threshold = quality_threshold

    def critique(self, state: AgentState) -> Dict[str, Any]:
        """Critique the generated book and provide feedback.

        Returns:
            Updated state with critique feedback
        """
        self.state = state

        with progress.agent_context(
                "Critic",
                "Evaluating book quality and providing constructive feedback"
        ):
            chapters = state.get("chapters", [])
            references = state.get("references", [])
            book_title = state.get("book_title", "")
            language = state.get("language", "en-US")
            style_instructions = state.get("style_instructions", "")

            if not chapters:
                progress.show_observation("No chapters found to critique")
                return {
                    "critic_feedback": "No chapters generated yet",
                    "quality_score": 0.0,
                    "status": "needs_revision"
                }

            progress.show_thought(
                f"Analyzing {len(chapters)} chapter(s) for quality, coherence, and completeness"
            )

            # Build critique prompt
            chapter_summaries = self._summarize_chapters(chapters)

            progress.show_action("Requesting AI critique of the generated book")

            # Load prompts from YAML and format with placeholders
            system_prompt_template = self.prompts['critic']['system_prompt']
            user_prompt_template = self.prompts['critic']['user_prompt']

            # Format style instructions section
            style_instructions_section = ""
            if style_instructions:
                style_instructions_section = f"Style Instructions: {style_instructions}\nEvaluate whether the book adheres to this requested style."

            system_prompt = system_prompt_template.format(
                language=language,
                style_instructions_section=style_instructions_section
            )

            user_prompt = user_prompt_template.format(
                book_title=book_title,
                chapter_count=len(chapters),
                reference_count=len(references),
                chapter_summaries=chapter_summaries
            )

            response = self._invoke_agent(system_prompt, user_prompt, state)

            progress.show_observation("Received critique feedback, analyzing results")

            # Parse critique (simplified)
            quality_score, feedback, decision = self._parse_critique(response)

            # Show critique summary
            progress.show_critique_summary(quality_score, feedback)

            # Determine next status
            if decision == "approve" or quality_score >= self.quality_threshold:
                status = "approved"
                progress.show_observation(
                    f"✓ Book approved! Quality score ({quality_score:.2%}) meets threshold ({self.quality_threshold:.2%})"
                )
            else:
                status = "needs_revision"
                progress.show_observation(
                    f"⚠ Book needs revision. Quality score ({quality_score:.2%}) below threshold ({self.quality_threshold:.2%})"
                )

        return {
            "critic_feedback": feedback,
            "quality_score": quality_score,
            "status": status,
            "iterations": 1
        }

    @staticmethod
    def _summarize_chapters(chapters: list) -> str:
        """Create summary of chapters for critique.
        
        Args:
            chapters: List of chapter dictionaries
            
        Returns:
            Formatted summary
        """
        summaries = []
        for chapter in chapters:
            num = chapter.get("number", 0)
            title = chapter.get("title", "Untitled")
            content = chapter.get("content", "")
            word_count = len(content.split())
            preview = content[:CHAPTER_PREVIEW_LENGTH] if content else "No content"

            summaries.append(
                f"Chapter {num}: {title} ({word_count} words)\n"
                f"Preview: {preview}..."
            )

        return "\n\n".join(summaries)

    @staticmethod
    def _extract_score(response_lines):
        for line in response_lines:
            if "score" in line or "quality" in line:
                # Try to find a number
                words = line.split()
                for word in words:
                    try:
                        # Remove all special chars except dot
                        word = ''.join(c for c in word if c.isdigit() or c == '.')

                        score = float(word.strip(':.,%'))
                        if 0 <= score <= 1:
                            return score
                        elif 0 <= score <= 10:
                            return score / 10
                        elif 0 <= score <= 100:
                            return score / 100
                    except ValueError:
                        continue
        return 0.0

    @staticmethod
    def _is_approved(response_lines):
        found_reject = False
        found_approve = False

        for line in response_lines:
            if "revise" in line or "needs work" in line or "improve" in line:
                found_reject = True
            elif "approve" in line or "good" in line or "excellent" in line:
                found_approve = True

        return found_approve and not found_reject

    @staticmethod
    def _parse_critique(response_content: str) -> tuple:
        """Parse critique response.
        
        Args:
            response_content: LLM response
            
        Returns:
            Tuple of (quality_score, feedback, decision)
        """
        feedback = response_content

        # Try to extract score from text
        lines = response_content.lower().split('\n')
        quality_score = CriticAgent._extract_score(lines)
        decision = "approve" if CriticAgent._is_approved(lines) else "revise"

        return quality_score, feedback, decision
