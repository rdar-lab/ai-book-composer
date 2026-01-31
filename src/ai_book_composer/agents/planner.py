"""Planner agent - Phase 1 of Deep-Agent architecture."""

from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_base import AgentBase
from .state import AgentState
from ..config import Settings
from ..progress_display import progress


class PlannerAgent(AgentBase):
    """The Planner (Product Manager) - generates structured plans."""

    def __init__(self, settings: Settings):
        super().__init__(settings, llm_temperature=0.3)

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

            if not self.settings.llm.static_plan:
                progress.show_thought(
                    f"Analyzing {len(files)} source file(s) to determine optimal book structure"
                )

                # Build file summary
                file_summary = self._summarize_files(files)

                progress.show_action("Generating comprehensive book plan with AI")

                # Load prompts from YAML and format with placeholders
                system_prompt_template = self.prompts['planner']['system_prompt']
                user_prompt_template = self.prompts['planner']['user_prompt']

                language = state.get("language", "en-US")
                style_instructions = state.get("style_instructions", "")

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
            else:
                plan = self._get_static_plan(files)

            progress.show_plan(plan)
            progress.show_observation(f"Plan created with {len(plan)} major tasks")

        return {
            "plan": plan,
            "status": "planned"
        }

    @staticmethod
    def _summarize_files(files: List[Dict[str, Any]]) -> str:
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

    @staticmethod
    def _get_static_plan(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        plan = [
            # Task 1: Gather all content
            {
                "task": "gather_content",
                "description": "Read and transcribe all source files",
                "status": "pending",
                "files": [f.get("path") for f in files]
            },
            # Task 2: Plan chapters
            {
                "task": "plan_chapters",
                "description": "Determine book structure and chapters",
                "status": "pending"
            },
            # Task 3: Generate chapters
            {
                "task": "generate_chapters",
                "description": "Write each chapter based on gathered content",
                "status": "pending"
            },
            # Task 4: Compile references
            {
                "task": "compile_references",
                "description": "Compile list of references",
                "status": "pending"
            },
            # Task 5: Generate final book
            {
                "task": "generate_book",
                "description": "Generate final book with all components",
                "status": "pending"
            }]

        return plan


    # noinspection PyUnusedLocal
    @staticmethod
    def _parse_plan(response_content: str, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse the LLM response into a structured plan (list of tasks).

        Args:
            response_content: LLM response (should be JSON array or JSON array in markdown/code block)
            files: Available files (list of dicts)
        Returns:
            Structured plan as a list of task dicts
        Raises:
            ValueError if parsing fails or structure is invalid
        """
        import json
        import re

        # Remove markdown code block formatting if present
        content = response_content.strip()
        code_block_match = re.match(r"^```(?:json)?\\n([\s\S]+?)\\n```$", content)
        if code_block_match:
            content = code_block_match.group(1).strip()
        # Try to parse JSON
        try:
            plan = json.loads(content)
        except Exception as e:
            raise ValueError(f"Failed to parse plan as JSON: {e}\nRaw content: {content[:200]}")

        # Validate and normalize structure
        if not isinstance(plan, list):
            raise ValueError("Plan must be a JSON array (list of tasks)")
        for i, task in enumerate(plan):
            if not isinstance(task, dict):
                raise ValueError(f"Task {i} is not a dict")
            for field in ("task", "description", "status"):
                if field not in task:
                    raise ValueError(f"Task {i} missing field: {field}")
            # files is optional, but if present, must be a list
            if "files" in task and not isinstance(task["files"], list):
                task["files"] = []
        return plan
