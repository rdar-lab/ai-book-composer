"""State management for the AI Book Composer agent."""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
import operator


class AgentState(TypedDict):
    """State for the Deep-Agent workflow."""
    
    # Input
    input_directory: str
    output_directory: str
    language: str
    
    # Files information
    files: List[Dict[str, Any]]
    
    # Planning
    plan: List[Dict[str, Any]]
    current_task_index: int
    
    # Execution
    gathered_content: Dict[str, Any]
    chapter_list: List[Dict[str, Any]]
    chapters: List[Dict[str, str]]
    references: List[str]
    
    # Iteration and feedback
    iterations: Annotated[int, operator.add]
    critic_feedback: Optional[str]
    quality_score: Optional[float]
    
    # Output
    book_title: str
    book_author: str
    final_output_path: Optional[str]
    
    # Status
    status: str
    error: Optional[str]


def create_initial_state(
    input_directory: str,
    output_directory: str,
    language: str = "en-US",
    book_title: str = "Composed Book",
    book_author: str = "AI Book Composer"
) -> AgentState:
    """Create initial agent state.
    
    Args:
        input_directory: Directory with source files
        output_directory: Directory for output
        language: Target language for the book
        book_title: Title of the book
        book_author: Author name
        
    Returns:
        Initial agent state
    """
    return AgentState(
        input_directory=input_directory,
        output_directory=output_directory,
        language=language,
        files=[],
        plan=[],
        current_task_index=0,
        gathered_content={},
        chapter_list=[],
        chapters=[],
        references=[],
        iterations=0,
        critic_feedback=None,
        quality_score=None,
        book_title=book_title,
        book_author=book_author,
        final_output_path=None,
        status="initialized",
        error=None
    )
