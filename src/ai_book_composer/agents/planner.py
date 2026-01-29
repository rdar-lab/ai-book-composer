"""Planner agent - Phase 1 of Deep-Agent architecture."""

from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import get_llm
from .state import AgentState


class PlannerAgent:
    """The Planner (Product Manager) - generates structured plans."""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.3)
    
    def plan(self, state: AgentState) -> Dict[str, Any]:
        """Generate a structured plan for creating the book.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with plan
        """
        files = state.get("files", [])
        language = state.get("language", "en-US")
        
        # Build file summary
        file_summary = self._summarize_files(files)
        
        # Create planning prompt
        system_prompt = """You are an expert book planner. Your job is to analyze source files and create a detailed plan for composing a comprehensive book.

The plan should include:
1. Analysis of available content
2. Proposed book structure (chapters)
3. Content mapping (which files/content go into which chapters)
4. References to collect

Output your plan as a JSON structure with these fields:
- analysis: Brief analysis of the content
- chapters: List of chapters with {number, title, description, source_files}
- references: List of reference sources to cite
"""
        
        user_prompt = f"""Create a detailed plan for a book based on these source files:

{file_summary}

Target language: {language}

Generate a comprehensive book plan."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Parse the plan (simplified - in production, use structured output)
        plan = self._parse_plan(response.content, files)
        
        return {
            "plan": plan,
            "status": "planned"
        }
    
    def _summarize_files(self, files: List[Dict[str, Any]]) -> str:
        """Create a summary of available files.
        
        Args:
            files: List of file information
            
        Returns:
            Formatted file summary
        """
        summary = []
        for i, file_info in enumerate(files, 1):
            name = file_info.get("name", "unknown")
            extension = file_info.get("extension", "")
            size = file_info.get("size", 0)
            summary.append(f"{i}. {name} ({extension}, {size} bytes)")
        return "\n".join(summary)
    
    def _parse_plan(self, response_content: str, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse the LLM response into a structured plan.
        
        Args:
            response_content: LLM response
            files: Available files
            
        Returns:
            Structured plan
        """
        # Simplified parsing - create default plan structure
        # In production, use structured output or JSON parsing
        
        plan = []
        
        # Task 1: Gather all content
        plan.append({
            "task": "gather_content",
            "description": "Read and transcribe all source files",
            "status": "pending",
            "files": [f.get("path") for f in files]
        })
        
        # Task 2: Plan chapters
        plan.append({
            "task": "plan_chapters",
            "description": "Determine book structure and chapters",
            "status": "pending"
        })
        
        # Task 3: Generate chapters
        plan.append({
            "task": "generate_chapters",
            "description": "Write each chapter based on gathered content",
            "status": "pending"
        })
        
        # Task 4: Compile references
        plan.append({
            "task": "compile_references",
            "description": "Compile list of references",
            "status": "pending"
        })
        
        # Task 5: Generate final book
        plan.append({
            "task": "generate_book",
            "description": "Generate final book with all components",
            "status": "pending"
        })
        
        return plan
