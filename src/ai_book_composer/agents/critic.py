"""Critic agent - Phase 3 of Deep-Agent architecture."""

from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage

from .state import AgentState
from ..config import load_prompts, Settings
from ..llm import get_llm
from ..progress_display import progress

# Constants
CHAPTER_PREVIEW_LENGTH = 200


class CriticAgent:
    """The Critic - validates quality and provides feedback."""

    def __init__(self, settings: Settings, quality_threshold: float = 0.7):
        self.llm = get_llm(settings=settings, temperature=0.2)
        self.quality_threshold = quality_threshold
        self.prompts = load_prompts()

    def critique(self, state: AgentState) -> Dict[str, Any]:
        """Critique the generated book and provide feedback.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with critique feedback
        """
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

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = self.llm.invoke(messages)

            progress.show_observation("Received critique feedback, analyzing results")

            # Parse critique (simplified)
            quality_score, feedback, decision = self._parse_critique(response.content)

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
    def _parse_critique(response_content: str) -> tuple:
        """Parse critique response.
        
        Args:
            response_content: LLM response
            
        Returns:
            Tuple of (quality_score, feedback, decision)
        """
        # Simplified parsing
        # In production, use structured output

        quality_score = 0.8  # Default good score
        feedback = response_content
        decision = "approve"

        # Try to extract score from text
        lines = response_content.lower().split('\n')
        for line in lines:
            if "score" in line or "quality" in line:
                # Try to find a number
                words = line.split()
                for word in words:
                    try:
                        score = float(word.strip(':.,%'))
                        if 0 <= score <= 1:
                            quality_score = score
                        elif 0 <= score <= 10:
                            quality_score = score / 10
                        elif 0 <= score <= 100:
                            quality_score = score / 100
                    except ValueError:
                        continue

            if "revise" in line or "needs work" in line or "improve" in line:
                decision = "revise"
            elif "approve" in line or "good" in line or "excellent" in line:
                decision = "approve"

        return quality_score, feedback, decision
