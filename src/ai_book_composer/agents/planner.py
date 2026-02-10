"""Planner agent - Phase 1 of Deep-Agent architecture."""

from typing import Dict, Any, List

from .agent_base import AgentBase
from .state import AgentState
from ..config import Settings
from ..llm import extract_json_from_llm_response
from ..progress_display import progress
from ..utils import file_utils


class PlannerAgent(AgentBase):
    """The Planner (Product Manager) - generates structured plans."""

    def __init__(self, settings: Settings):
        super().__init__(settings, llm_temperature=settings.llm.temperature.get('planning', 0.3))

    def plan(self, state: AgentState) -> Dict[str, Any]:
        """Generate a structured plan for creating the book.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with plan
        """

        self.state = state

        with progress.agent_context(
                "Planner",
                "Analyzing source files and creating a structured plan for book generation"
        ):
            files = state.get("files", [])

            if not self.settings.llm.static_plan:
                plan_cache_file = file_utils.get_cache_path(self.settings, "planner_plan.json")

                plan = file_utils.read_cache(plan_cache_file) if self.settings.book.use_cached_plan else None

                if not plan:
                    progress.show_thought(
                        f"Analyzing {len(files)} source file(s) to determine optimal book structure"
                    )

                    # Build file summary
                    file_summary = self._get_files_summary()

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

                    response = self._invoke_agent(system_prompt, user_prompt, self.state)

                    progress.show_observation("Received plan from AI, parsing into structured tasks")

                    # Parse the plan (simplified - in production, use structured output)
                    plan = self._parse_plan(response)

                    file_utils.write_cache(plan_cache_file, plan)
            else:
                plan = self._get_static_plan()

            progress.show_plan(plan)

            self.state["plan"] = plan

            progress.show_observation(f"Plan created with {len(plan)} major tasks")

        return {
            "plan": plan,
            "status": "planned"
        }

    @staticmethod
    def _get_static_plan() -> List[Dict[str, Any]]:
        plan = [
            # Task 1: Plan chapters
            {
                "task": "plan_chapters",
                "description": "Determine book structure and chapters",
                "status": "pending"
            },
            # Task 2: Generate chapters
            {
                "task": "generate_chapters",
                "description": "Write each chapter based on gathered content",
                "status": "pending"
            },
            # Task 3: Compile references
            {
                "task": "compile_references",
                "description": "Compile list of references",
                "status": "pending"
            }
        ]

        return plan

    @staticmethod
    def _parse_plan(response_content) -> List[Dict[str, Any]]:
        """Parse the LLM response into a structured plan (list of tasks).

        Args:
            response_content: LLM response (should be JSON array or JSON array in markdown/code block)
        Returns:
            Structured plan as a list of task dicts
        Raises:
            ValueError if parsing fails or structure is invalid
        """
        plan = extract_json_from_llm_response(response_content)

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
