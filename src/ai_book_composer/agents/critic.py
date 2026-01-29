"""Critic agent - Phase 3 of Deep-Agent architecture."""

from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import get_llm
from .state import AgentState

# Constants
CHAPTER_PREVIEW_LENGTH = 200


class CriticAgent:
    """The Critic - validates quality and provides feedback."""
    
    def __init__(self, quality_threshold: float = 0.7):
        self.llm = get_llm(temperature=0.2)
        self.quality_threshold = quality_threshold
    
    def critique(self, state: AgentState) -> Dict[str, Any]:
        """Critique the generated book and provide feedback.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with critique feedback
        """
        chapters = state.get("chapters", [])
        references = state.get("references", [])
        book_title = state.get("book_title", "")
        language = state.get("language", "en-US")
        
        if not chapters:
            return {
                "critic_feedback": "No chapters generated yet",
                "quality_score": 0.0,
                "status": "needs_revision"
            }
        
        # Build critique prompt
        chapter_summaries = self._summarize_chapters(chapters)
        
        system_prompt = """You are a harsh but fair book critic. Your job is to evaluate the quality of a generated book and provide constructive feedback.

Evaluate the book on these criteria:
1. Structure: Is the book well-organized with clear chapters?
2. Content Quality: Is the content comprehensive and informative?
3. Coherence: Do chapters flow logically?
4. Completeness: Are all necessary components present (title page, TOC, references)?
5. Language: Is the language appropriate and well-written?

Provide:
- quality_score: A score from 0.0 to 1.0
- feedback: Specific areas for improvement
- decision: "approve" if quality_score >= threshold, "revise" if needs work"""
        
        user_prompt = f"""Evaluate this book:

Title: {book_title}
Target Language: {language}
Number of Chapters: {len(chapters)}
Number of References: {len(references)}

Chapter summaries:
{chapter_summaries}

Provide your critique."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Parse critique (simplified)
        quality_score, feedback, decision = self._parse_critique(response.content)
        
        # Determine next status
        if decision == "approve" or quality_score >= self.quality_threshold:
            status = "approved"
        else:
            status = "needs_revision"
        
        return {
            "critic_feedback": feedback,
            "quality_score": quality_score,
            "status": status,
            "iterations": 1
        }
    
    def _summarize_chapters(self, chapters: list) -> str:
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
    
    def _parse_critique(self, response_content: str) -> tuple:
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
