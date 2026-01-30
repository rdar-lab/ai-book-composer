"""Planner agent - Phase 1 of Deep-Agent architecture."""

from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, SystemMessage

from .state import AgentState
from ..config import load_prompts, Settings
from ..llm import get_llm
from ..progress_display import progress


class PlannerAgent:
    """The Planner (Product Manager) - generates structured plans."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = get_llm(settings, temperature=0.3)
        self.prompts = load_prompts()

    def plan(self, state: AgentState) -> Dict[str, Any]:
        """Generate a structured plan for creating the book.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with plan
        """
        with progress.agent_context(
                "Planner",
                "Analyzing source files and creating a structured plan for book generation"
        ):
            files = state.get("files", [])
            language = state.get("language", "en-US")
            style_instructions = state.get("style_instructions", "")

            progress.show_thought(
                f"Analyzing {len(files)} source file(s) to determine optimal book structure"
            )

            # Build file summary
            file_summary = self._summarize_files(files)

            progress.show_action("Generating comprehensive book plan with AI")

            # Load prompts from YAML and format with placeholders
            system_prompt_template = self.prompts['planner']['system_prompt']
            user_prompt_template = self.prompts['planner']['user_prompt']

            # Format style instructions section
            style_instructions_section = ""
            if style_instructions:
                style_instructions_section = f"Style Instructions: {style_instructions}\nPlease plan the book structure to match this style."

            system_prompt = system_prompt_template.format(
                language=language,
                style_instructions_section=style_instructions_section
            )

            user_prompt = user_prompt_template.format(
                file_summary=file_summary
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = self.llm.invoke(messages)

            progress.show_observation("Received plan from AI, parsing into structured tasks")

            # Parse the plan (simplified - in production, use structured output)
            plan = self._parse_plan(response.content, files)

            progress.show_plan(plan)
            progress.show_observation(f"Plan created with {len(plan)} major tasks")

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
