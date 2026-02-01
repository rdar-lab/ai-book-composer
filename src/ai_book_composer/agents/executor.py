"""Executor agent - Phase 2 of Deep-Agent architecture."""
import logging
import re
from typing import Dict, Any, List

from langchain_core.messages import AIMessage
from langchain_core.tools import tool, BaseTool

from .agent_base import AgentBase
from .state import AgentState
from .. import progress_display
from ..config import Settings
from ..parallel_utils import execute_parallel
from ..progress_display import progress
from ..utils import file_utils
from ..utils.book_writer import BookWriter

# Constants
MIN_CHAPTER_COUNT = 3
# noinspection RegExpRedundantEscape
RE_TEXT_CHAPTER = re.compile(r'^(?:Chapter\s+)?(\d+)[\.:]\s*(.*)', re.IGNORECASE)

logger = logging.getLogger(__name__)


class ExecutorAgent(AgentBase):
    """The Executor (Worker) - performs tasks using available tools."""

    def __init__(self, settings: Settings, output_directory: str):
        super().__init__(
            settings,
            llm_temperature=0.7,
            output_directory=output_directory
        )

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """Execute the next task in the plan.

        Returns:
            Updated state
        """

        self.state = state

        with progress.agent_context(
                "Executor",
                "Executing tasks using specialized tools to generate book content"
        ):
            plan = state.get("plan", [])
            current_task_index = state.get("current_task_index", 0)

            if current_task_index >= len(plan):
                progress.show_observation("All tasks completed")
                return {"status": "execution_complete"}

            current_task = plan[current_task_index]
            task_type = current_task.get("task")
            task_description = current_task.get("description", "")

            progress.show_step(
                current_task_index + 1,
                len(plan),
                f"{task_type}: {task_description}"
            )
            progress.show_task(task_type, "started")

            # Execute based on task type
            if task_type == "plan_chapters":
                self._plan_chapters_inner()
            elif task_type == "generate_chapters":
                self._generate_chapters_inner()
            elif task_type == "compile_references":
                self._compile_references_inner()
            elif task_type == "generate_book":
                self._generate_book_inner()
            else:
                self._custom_agent_task(current_task)

            progress.show_task(task_type, "completed")

            return {
                "chapter_list": self.state.get("chapter_list", []),
                "chapters": self.state.get("chapters", []),
                "references": self.state.get("references", []),
                "final_output_path": self.state.get("final_output_path", ""),
                "current_task_index": current_task_index + 1,
                "status": "book_generate" if self.state.get("final_output_path") else "executing"
            }

    def _custom_agent_task(self, current_task: dict[str, Any]) -> str | list[str | dict] | AIMessage:
        system_prompt_template = self.prompts['executor'].get('llm_agent_system_prompt')
        user_prompt_template = self.prompts['executor'].get('llm_agent_user_prompt')
        system_prompt = system_prompt_template.format()
        user_prompt = user_prompt_template.format(state=self.state, current_task=current_task)
        response = self._invoke_agent(system_prompt, user_prompt, self.state,
                                      custom_tools=self.get_custom_agent_tools())
        return response

    def plan_chapters_tool(self):
        @tool
        def plan_chapters() -> Dict[str, Any]:
            """Plan the book chapters based on gathered content."""
            progress_display.progress.show_action("Planning book chapters based on gathered content")
            return self._plan_chapters_inner()

        return plan_chapters

    def _plan_chapters_inner(self) -> dict[str, list[dict[str, Any]] | int | str | Any]:
        cached_chapters_list_file = file_utils.get_cache_path(self.settings, "chapter_list.json")
        chapter_list = file_utils.read_cache(cached_chapters_list_file) if self.settings.book.use_cached_chapters_list else None
        if not chapter_list:
            language = self.state.get("language", "en-US")
            style_instructions = self.state.get("style_instructions", "")

            progress.show_thought("Analyzing gathered content to determine optimal chapter structure")
            progress.show_action("Using AI agent with tools to plan book chapters dynamically")

            # Load prompts from YAML and format with placeholders
            system_prompt_template = self.prompts['executor']['chapter_planning_system_prompt']
            user_prompt_template = self.prompts['executor']['chapter_planning_user_prompt']

            # Format style instructions section
            style_instructions_section = ""
            if style_instructions:
                style_instructions_section = f"Style Instructions: {style_instructions}\nPlease ensure the chapter structure and organization reflects this style."

            system_prompt = system_prompt_template.format(
                language=language,
                style_instructions_section=style_instructions_section
            )
            user_prompt = user_prompt_template.format(
                file_summary=self._get_files_summary()
            )

            # Use agent with tools to allow dynamic content reading
            result = self._invoke_agent(system_prompt, user_prompt, self.state)

            # Parse chapter list (simplified)
            chapter_list = self._parse_chapter_list(result)

            progress.show_observation(f"Planned {len(chapter_list)} chapter(s) for the book")
            for chapter in chapter_list:
                progress.show_observation(f"  • Chapter {chapter.get('number')}: {chapter.get('title')}")
            file_utils.write_cache(cached_chapters_list_file, chapter_list)

        # Save chapter list
        self.state["chapter_list"] = chapter_list

        return {
            "chapter_list": chapter_list,
        }

    # Create a helper function for parallel execution
    def generate_chapter_wrapper(self, chapter_info: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper function to generate a single chapter for parallel execution."""

        language = self.state.get("language", "en-US")
        style_instructions = self.state.get("style_instructions", "")

        chapter_num = chapter_info.get("number", 0)
        chapter_title = chapter_info.get("title", "Untitled")
        chapter_desc = chapter_info.get("description", "")

        progress.show_action(f"Generating Chapter {chapter_num}: {chapter_title}")

        try:
            # Generate chapter content
            content = self._generate_chapter_content(
                chapter_num,
                chapter_title,
                chapter_desc,
                language,
                style_instructions
            )

            word_count = len(content.split())
            progress.show_observation(f"✓ Chapter {chapter_num} complete ({word_count} words)")

            return {
                "number": chapter_num,
                "title": chapter_title,
                "content": content,
                "status": "success"
            }
        except Exception as e:
            logger.exception(f"Error generating Chapter {chapter_num}: {str(e)}")
            progress.show_observation(f"✗ Error generating Chapter {chapter_num}: {str(e)}")
            return {
                "number": chapter_num,
                "title": chapter_title,
                "content": "",
                "status": "error",
                "error": str(e)
            }

    def generate_chapters_tool(self):
        @tool
        def generate_chapters() -> Dict[str, Any]:
            """Generate all chapters of the book.
            """
            progress_display.progress.show_action("Generating all book chapters")
            return self._generate_chapters_inner()

        return generate_chapters

    def _generate_chapters_inner(self) -> Dict[str, Any]:
        chapter_list = self.state.get("chapter_list", [])

        if not chapter_list:
            raise Exception("⚠ No chapters to generate")

        progress.show_thought(f"Generating {len(chapter_list)} chapters in parallel")
        progress.show_observation(f"Parallel execution enabled - generating chapters concurrently")

        # Generate all chapters in parallel
        chapter_results = execute_parallel(self.settings, self.generate_chapter_wrapper, chapter_list)

        # Sort results by chapter number to maintain order
        chapter_results.sort(key=lambda x: x.get("number", 0))

        # Filter out errors and create final chapter list
        chapters = []
        for result in chapter_results:
            if result.get("status") == "success":
                chapters.append({
                    "number": result.get("number"),
                    "title": result.get("title"),
                    "content": result.get("content")
                })
            else:
                raise Exception(
                    f"Chapter {result.get('number')} failed: {result.get('error', 'Unknown error')}")

        progress.show_observation(f"✓ Generated {len(chapters)}/{len(chapter_list)} chapters successfully")

        self.state["chapters"] = chapters

        return {
            "chapters": chapters,
        }

    def _generate_chapter_content(
            self,
            chapter_num: int,
            title: str,
            description: str,
            language: str,
            style_instructions: str = ""
    ) -> str:
        """Generate content for a single chapter using agent with tools."""

        cached_chapter_file = file_utils.get_cache_path(
            self.settings,
            f"chapter_{chapter_num}_content.txt"
        )
        content = file_utils.read_cache(cached_chapter_file) if self.settings.book.use_cached_chapters_content else None

        if not content:
            # Load prompts from YAML and format with placeholders
            system_prompt_template = self.prompts['executor']['chapter_generation_system_prompt']
            user_prompt_template = self.prompts['executor']['chapter_generation_user_prompt']

            # Format style instructions section
            style_instructions_section = ""
            if style_instructions:
                style_instructions_section = f"Style Instructions: {style_instructions}\nPlease write this chapter in accordance with these style guidelines."

            system_prompt = system_prompt_template.format(
                language=language,
                chapter_number=chapter_num,
                title=title,
                description=description,
                style_instructions_section=style_instructions_section
            )

            user_prompt = user_prompt_template.format(
                chapter_number=chapter_num,
                title=title,
                description=description,
                file_summary=self._get_files_summary()
            )

            # Use agent with tools to allow dynamic content reading
            result = self._invoke_agent(system_prompt, user_prompt, self.state)
            content = self._parse_chapter_content_response(result)
            file_utils.write_cache(cached_chapter_file, content)

        return content

    def _parse_chapter_content_response(self, llm_response: str) -> str:
        # noinspection PyBroadException
        try:
            parsed_response = self._extract_json_from_llm_response(llm_response)
            if isinstance(str, parsed_response):
                return parsed_response
            else:
                raise Exception("Was able to detect JSON but it wasn't a string")
        except Exception:
            # Fallback to raw text if JSON parsing fails
            return llm_response

    def compile_references_tool(self):
        @tool
        def compile_references() -> Dict[str, Any]:
            """Compile list of references."""
            progress_display.progress.show_action("Compiling list of source file references")
            return self._compile_references_inner()

        return compile_references

    def _compile_references_inner(self) -> dict[str, list[Any] | int | str | Any]:
        files = self.state.get("files", [])

        progress.show_action("Compiling list of source file references")

        references = []
        for file_info in files:
            file_path = file_info.get("path", "")
            file_name = file_info.get("name", "")
            references.append(f"{file_name} - Source file: {file_path}")

        progress.show_observation(f"Compiled {len(references)} reference(s)")

        self.state["references"] = references

        return {
            "references": references,
        }

    def generate_book_tool(self):
        @tool
        def generate_book() -> Dict[str, Any]:
            """Generate the final book."""
            progress_display.progress.show_action("Generating the final book document")
            return self._generate_book_inner()

        return generate_book

    def _generate_book_inner(self) -> dict[str, int | str | Any]:
        title = self.state.get("book_title", "Composed Book")
        author = self.state.get("book_author", "AI Book Composer")
        chapters = self.state.get("chapters", [])
        references = self.state.get("references", [])

        progress.show_action(f"Generating final book: '{title}' by {author}")
        progress.show_observation(
            f"Compiling {len(chapters)} chapters and {len(references)} references into RTF format")

        result = BookWriter(self.settings, self.output_directory).run(
            title=title,
            author=author,
            chapters=chapters,
            references=references,
            output_filename="final_book.rtf"
        )

        output_path = result.get("file_path")
        progress.show_observation(f"✓ Book generated successfully: {output_path}")

        self.state["final_output_path"] = output_path

        return {
            "final_output_path": output_path,
        }

    def _parse_chapter_list(self, response_content: str) -> List[Dict[str, Any]]:
        """Parse chapter list from LLM response."""
        chapters = []

        try:
            data = self._extract_json_from_llm_response(response_content)
        except Exception as e:
            logger.warning(f"Failed to parse chapter list as JSON: {str(e)}")
            data = None

        if data:
            if isinstance(data, list):
                for i, item in enumerate(data, 1):
                    if isinstance(item, dict):
                        chapters.append({
                            "number": item.get("number", i),
                            "title": item.get("title", f"Chapter {i}"),
                            "description": item.get("description", "")
                        })
            else:
                raise Exception("Chapter list JSON is not a list")
        # Fallback to line-by-line regex if JSON failed
        else:
            last_chapter = None
            for line in response_content.split('\n'):
                line = line.strip()
                match = RE_TEXT_CHAPTER.match(line)
                if match:
                    if last_chapter:
                        chapters.append(last_chapter)

                    num, title = match.groups()
                    last_chapter = {
                        "number": int(num),
                        "title": title.strip() or f"Chapter {num}",
                        "description": ""
                    }
                else:
                    if last_chapter:
                        if last_chapter["description"]:
                            last_chapter["description"] += "\n" + line
                        else:
                            last_chapter["description"] = line
            if last_chapter:
                chapters.append(last_chapter)

        if len(chapters) < MIN_CHAPTER_COUNT:
            raise ValueError(f"Insufficient chapters found: {len(chapters)}")

        return chapters

    def get_custom_agent_tools(self) -> list[BaseTool]:
        """Extend base tools with built-in executor methods as LangChain tools."""
        # Get MCP tools (as in base)
        base_tools = super()._generate_tools()
        # Built-in tool wrappers
        custom_tools: list[BaseTool] = [
            self.plan_chapters_tool(),
            self.generate_chapters_tool(),
            self.compile_references_tool(),
            self.generate_book_tool(),
            self.get_file_content_tool()
        ]
        return list(base_tools) + custom_tools
