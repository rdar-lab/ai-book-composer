"""Executor agent - Phase 2 of Deep-Agent architecture."""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

from .agent_base import AgentBase
from .state import AgentState
from ..config import Settings
from ..parallel_utils import execute_parallel, is_parallel_enabled
from ..progress_display import progress

# Constants
MIN_SUBSTANTIAL_CONTENT_LENGTH = 100
MAX_CONTENT_PREVIEW_LENGTH = 500
# Maximum content length per file when planning chapters (to manage LLM token limits)
MAX_CONTENT_FOR_CHAPTER_PLANNING = 10000
MIN_CHAPTER_COUNT = 3
MAX_CHAPTER_COUNT = 10

logger = logging.getLogger(__name__)


class ExecutorAgent(AgentBase):
    """The Executor (Worker) - performs tasks using available tools."""

    def __init__(self, settings: Settings, input_directory: str, output_directory: str):
        super().__init__(
            settings,
            llm_temperature=0.7,
            cache_llm=not settings.parallel.parallel_execution,
            bind_tools=True,
            input_directory=input_directory,
            output_directory=output_directory
        )
        self.state: Optional[AgentState] = None

    def list_files(self) -> List[Dict[str, Any]]:
        """List all files in the input directory using the appropriate tool.

        Returns:
            List of file information dictionaries
        """
        return self._invoke_tool("list_files")

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """Execute the next task in the plan.
        
        Args:
            state: Current agent state
            
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
            if task_type == "gather_content":
                result = self._gather_content_inner()
            elif task_type == "plan_chapters":
                result = self._plan_chapters_inner()
            elif task_type == "generate_chapters":
                result = self._generate_chapters_inner()
            elif task_type == "compile_references":
                result = self._compile_references_inner()
            elif task_type == "generate_book":
                result = self._generate_book_inner()
            else:
                result = self._custom_agent_task(current_task)

                return {
                    "llm_agent_result": result,
                    "current_task_index": current_task_index + 1,
                    "status": "executing"
                }

            progress.show_task(task_type, "completed")

            return result

    def _custom_agent_task(self, current_task: dict[str, Any]) -> str | list[str | dict] | AIMessage:
        system_prompt_template = self.prompts['executor'].get('llm_agent_system_prompt')
        user_prompt_template = self.prompts['executor'].get('llm_agent_user_prompt')
        system_prompt = system_prompt_template.format()
        user_prompt = user_prompt_template.format(state=self.state, current_task=current_task)
        response = self._invoke_agent(system_prompt, user_prompt, self.state)
        result = response.content if hasattr(response, 'content') else response
        return result

    def _process_single_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single file and return its content.
        
        Args:
            file_info: Dictionary with file information (path, name, extension)
            
        Returns:
            Dictionary with file_path, type, content, and status
        """
        file_path = file_info.get("path", "")
        file_name = file_info.get("name", "")
        extension = file_info.get("extension", "").lower()

        try:
            if extension in [".txt", ".md", ".rst", ".docx", ".rtf", ".pdf"]:
                # Read text file (PDF text content extracted here, images extracted separately)
                result = self._invoke_tool("read_text_file", file_path=file_path)
                return {
                    "file_path": file_path,
                    "file_name": file_name,
                    "type": "text",
                    "content": result.get("content", ""),
                    "status": "success"
                }
            elif extension in [".mp3", ".wav", ".m4a", ".flac"]:
                # Transcribe audio
                result = self._invoke_tool("transcribe_audio", file_path=file_path)
                return {
                    "file_path": file_path,
                    "file_name": file_name,
                    "type": "audio_transcription",
                    "content": result.get("transcription", ""),
                    "status": "success"
                }
            elif extension in [".mp4", ".avi", ".mov", ".mkv"]:
                # Transcribe video
                result = self._invoke_tool("transcribe_video", file_path=file_path)
                return {
                    "file_path": file_path,
                    "file_name": file_name,
                    "type": "video_transcription",
                    "content": result.get("transcription", ""),
                    "status": "success"
                }
            else:
                # Unsupported file type
                return {
                    "file_path": file_path,
                    "file_name": file_name,
                    "type": "unsupported",
                    "content": f"File type {extension} is not supported",
                    "status": "skipped"
                }
        except Exception as e:
            logger.exception(f"Error processing file {file_path}: {str(e)}")
            return {
                "file_path": file_path,
                "file_name": file_name,
                "type": "error",
                "content": f"Error processing file: {str(e)}",
                "status": "error"
            }

    def _extract_images_from_single_pdf(self, pdf_file: Dict[str, Any]) -> Dict[str, Any]:
        """Extract images from a single PDF file.
        
        Args:
            pdf_file: Dictionary with PDF file information
            
        Returns:
            Dictionary with images and status
        """
        pdf_path = pdf_file.get("path", "")
        pdf_name = pdf_file.get("name", "")

        try:
            result = self._invoke_tool("extract_images_from_pdf", file_path=pdf_path)
            if result.get("success"):
                return {
                    "pdf_name": pdf_name,
                    "images": result.get("images", []),
                    "status": "success"
                }
            else:
                return {
                    "pdf_name": pdf_name,
                    "images": [],
                    "status": "no_images"
                }
        except Exception as e:
            logger.exception(f"Error extracting images from PDF {pdf_path}: {str(e)}")

            return {
                "pdf_name": pdf_name,
                "images": [],
                "status": "error",
                "error": str(e)
            }

    @tool
    def gather_content(self) -> Dict[str, Any]:
        """Gather content from all source files."""

        return self._gather_content_inner()

    def _gather_content_inner(self) -> dict[str, dict[Any, Any] | list[Any] | int | str | Any]:
        files = self.state.get("files", [])
        gathered_content = {}

        progress.show_thought(f"Need to process {len(files)} source file(s) and extract their content")

        if is_parallel_enabled(self.settings):
            progress.show_observation(f"Parallel execution enabled - processing files in parallel")

        # Process files (potentially in parallel)
        file_results = execute_parallel(self.settings, self._process_single_file, files)

        # Process results and show progress
        for i, result in enumerate(file_results, 1):
            file_path = result.get("file_path", "")
            file_name = result.get("file_name", "")
            status = result.get("status", "")
            content = result.get("content", "")
            file_type = result.get("type", "")

            progress.show_action(f"Processing file {i}/{len(files)}: {file_name}")

            if status == "success":
                gathered_content[file_path] = {
                    "type": file_type,
                    "content": content
                }
                content_length = len(content)
                progress.show_observation(f"✓ Processed {file_name} ({content_length} characters)")
            elif status == "skipped":
                gathered_content[file_path] = {
                    "type": file_type,
                    "content": content
                }
                progress.show_observation(f"⚠ Skipping unsupported file type: {file_name}")
            else:  # error
                gathered_content[file_path] = {
                    "type": file_type,
                    "content": content
                }
                progress.show_observation(f"✗ Error processing {file_name}: {content}")

        # Gather images from input directory and extract from PDFs
        progress.show_thought("Gathering images from input directory")

        all_images = []

        try:
            # List existing images in input directory
            progress.show_action("Listing existing images in input directory")
            existing_images = self._invoke_tool("list_images")
            all_images.extend(existing_images)
            progress.show_observation(f"✓ Found {len(existing_images)} existing image(s)")
        except Exception as e:
            logger.exception(f"Error listing images: {str(e)}")
            progress.show_observation(f"⚠ Error listing images: {str(e)}")

        # Extract images from PDF files (potentially in parallel)
        pdf_files = [f for f in files if f.get("extension", "").lower() == ".pdf"]
        if pdf_files:
            progress.show_action(f"Extracting images from {len(pdf_files)} PDF file(s)")

            if is_parallel_enabled(self.settings):
                progress.show_observation(f"Parallel execution enabled - extracting images in parallel")

            # Extract images from PDFs (potentially in parallel)
            pdf_results = execute_parallel(self.settings, self._extract_images_from_single_pdf, pdf_files)

            # Process results
            for result in pdf_results:
                pdf_name = result.get("pdf_name", "")
                status = result.get("status", "")
                images = result.get("images", [])

                if status == "success":
                    all_images.extend(images)
                    progress.show_observation(f"✓ Extracted {len(images)} image(s) from {pdf_name}")
                elif status == "error":
                    error_msg = result.get("error", "Unknown error")
                    progress.show_observation(f"⚠ Error extracting images from {pdf_name}: {error_msg}")
                else:
                    progress.show_observation(f"⚠ No images found in {pdf_name}")

        progress.show_observation(f"Image gathering complete: {len(all_images)} total image(s) available")

        self.state["gathered_content"] = gathered_content
        self.state["images"] = all_images

        return {
            "gathered_content": gathered_content,
            "images": all_images,
            "current_task_index": self.state.get("current_task_index", 0) + 1,
            "status": "executing"
        }

    @tool
    def plan_chapters(self) -> Dict[str, Any]:
        """Plan the book chapters based on gathered content."""

        return self._plan_chapters_inner()

    def _plan_chapters_inner(self) -> dict[str, list[dict[str, Any]] | int | str | Any]:
        gathered_content = self.state.get("gathered_content", {})
        language = self.state.get("language", "en-US")
        style_instructions = self.state.get("style_instructions", "")

        progress.show_thought("Analyzing gathered content to determine optimal chapter structure")

        # Create content summary
        content_summary = self._summarize_content(gathered_content)

        progress.show_action("Using AI to plan book chapters")

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
        user_prompt = user_prompt_template.format(content_summary=content_summary)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)

        # Parse chapter list (simplified)
        chapter_list = self._parse_chapter_list(response.content)

        progress.show_observation(f"Planned {len(chapter_list)} chapter(s) for the book")
        for chapter in chapter_list:
            progress.show_observation(f"  • Chapter {chapter.get('number')}: {chapter.get('title')}")

        # Save chapter list
        progress.show_action("Saving chapter plan to disk")
        self._invoke_tool("write_chapter_list", chapters=chapter_list)

        self.state["chapter_list"] = chapter_list

        return {
            "chapter_list": chapter_list,
            "current_task_index": self.state.get("current_task_index", 0) + 1,
            "status": "executing"
        }

    # Create a helper function for parallel execution
    def generate_chapter_wrapper(self, chapter_info: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper function to generate a single chapter for parallel execution."""

        gathered_content = self.state.get("gathered_content", {})
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
                gathered_content,
                language,
                style_instructions
            )

            # Save chapter
            self._invoke_tool("write_chapter", chapter_number=chapter_num, title=chapter_title, content=content)

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

    @tool
    def generate_chapters(self) -> Dict[str, Any]:
        """Generate all chapters of the book.
        """
        return self._generate_chapters_inner()

    def _generate_chapters_inner(self) -> Dict[str, Any]:
        chapter_list = self.state.get("chapter_list", [])

        if not chapter_list:
            progress.show_observation("⚠ No chapters to generate")
            return {
                "current_task_index": self.state.get("current_task_index", 0) + 1,
                "status": "executing"
            }

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

        return {
            "chapters": chapters,
            "current_task_index": self.state.get("current_task_index", 0) + 1,
            "status": "executing"
        }

    def _generate_chapter_content(
            self,
            chapter_num: int,
            title: str,
            description: str,
            gathered_content: Dict[str, Any],
            language: str,
            style_instructions: str = ""
    ) -> str:
        """Generate content for a single chapter."""
        # Prepare content summary
        all_content = []
        for file_path, content_info in gathered_content.items():
            content = content_info.get("content", "")
            if content and len(content) > MIN_SUBSTANTIAL_CONTENT_LENGTH:  # Only include substantial content
                all_content.append(f"From {Path(file_path).name}:\n{content[:MAX_CONTENT_PREVIEW_LENGTH]}...")

        content_text = "\n\n".join(all_content)

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
            content_text=content_text
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)
        return response.content

    @tool
    def compile_references(self) -> Dict[str, Any]:
        """Compile list of references."""
        return self._compile_references_inner()

    def _compile_references_inner(self) -> dict[str, list[Any] | int | str | Any]:
        files = self.state.get("files", [])

        progress.show_action("Compiling list of source file references")

        references = []
        for file_info in files:
            file_path = file_info.get("path", "")
            file_name = file_info.get("name", "")
            references.append(f"{file_name} - Source file: {file_path}")

        progress.show_observation(f"Compiled {len(references)} reference(s)")

        return {
            "references": references,
            "current_task_index": self.state.get("current_task_index", 0) + 1,
            "status": "executing"
        }

    @tool
    def generate_book(self) -> Dict[str, Any]:
        """Generate the final book."""
        return self._generate_book_inner()

    def _generate_book_inner(self) -> dict[str, int | str | Any]:
        title = self.state.get("book_title", "Composed Book")
        author = self.state.get("book_author", "AI Book Composer")
        chapters = self.state.get("chapters", [])
        references = self.state.get("references", [])

        progress.show_action(f"Generating final book: '{title}' by {author}")
        progress.show_observation(
            f"Compiling {len(chapters)} chapters and {len(references)} references into RTF format")

        result = self._invoke_tool(
            "generate_book",
            book_title=title,
            book_author=author,
            chapters=chapters,
            references=references
        )

        output_path = result.get("file_path")
        progress.show_observation(f"✓ Book generated successfully: {output_path}")

        return {
            "final_output_path": output_path,
            "current_task_index": self.state.get("current_task_index", 0) + 1,
            "status": "book_generated"
        }

    @staticmethod
    def _summarize_content(gathered_content: Dict[str, Any]) -> str:
        """Summarize gathered content with full content for chapter planning.
        
        Includes full content from each file (up to MAX_CONTENT_FOR_CHAPTER_PLANNING chars)
        so the LLM can understand all available material when planning chapters.
        For very large files, content is truncated to manage token limits.
        """
        summary = []
        for file_path, content_info in gathered_content.items():
            content = content_info.get("content", "")
            content_type = content_info.get("type", "unknown")

            # Include full content so LLM can understand all files when planning chapters
            # Truncate extremely large files to manage token limits
            if content:
                if len(content) > MAX_CONTENT_FOR_CHAPTER_PLANNING:
                    file_content = content[
                                       :MAX_CONTENT_FOR_CHAPTER_PLANNING] + f"\n\n[Content truncated - {len(content) - MAX_CONTENT_FOR_CHAPTER_PLANNING} more characters]"
                else:
                    file_content = content
            else:
                file_content = "No content"

            summary.append(f"File: {Path(file_path).name} ({content_type})\nContent:\n{file_content}\n")
        return "\n".join(summary)

    @staticmethod
    def _parse_chapter_list(response_content: str) -> List[Dict[str, Any]]:
        """Parse chapter list from LLM response."""
        # Simplified parsing - create default structure
        # In production, use structured output

        chapters = []
        lines = response_content.split('\n')

        chapter_num = 1
        for line in lines:
            line = line.strip()
            if line.startswith("Chapter") or line.startswith(f"{chapter_num}."):
                # Extract title
                parts = line.split(":", 1)
                if len(parts) == 2:
                    title = parts[1].strip()
                else:
                    title = f"Chapter {chapter_num}"

                chapters.append({
                    "number": chapter_num,
                    "title": title,
                    "description": f"Content for {title}",
                    "key_points": []
                })
                chapter_num += 1

        # Ensure we have at least minimum chapters
        while len(chapters) < MIN_CHAPTER_COUNT:
            chapters.append({
                "number": len(chapters) + 1,
                "title": f"Chapter {len(chapters) + 1}",
                "description": "Additional chapter content",
                "key_points": []
            })

        return chapters[:MAX_CHAPTER_COUNT]

    def _generate_tools(self):
        """Extend base tools with built-in executor methods as LangChain tools."""
        # Get MCP tools (as in base)
        mcp_tools = super()._generate_tools()
        # Built-in tool wrappers
        builtin_tools = [self.gather_content, self.plan_chapters, self.generate_chapters, self.compile_references,
                         self.generate_book]
        return list(mcp_tools) + builtin_tools
