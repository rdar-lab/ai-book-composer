"""Executor agent - Phase 2 of Deep-Agent architecture."""

from typing import Dict, Any, List
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
import os
import asyncio

from ..llm import get_llm
from ..config import load_prompts
from ..progress_display import progress
from .state import AgentState

# Constants
MIN_SUBSTANTIAL_CONTENT_LENGTH = 100
MAX_CONTENT_PREVIEW_LENGTH = 500
# Maximum content length per file when planning chapters (to manage LLM token limits)
MAX_CONTENT_FOR_CHAPTER_PLANNING = 10000
MIN_CHAPTER_COUNT = 3
MAX_CHAPTER_COUNT = 10


class ExecutorAgent:
    """The Executor (Worker) - performs tasks using available tools."""
    
    def __init__(self, input_directory: str, output_directory: str):
        self.input_directory = input_directory
        self.output_directory = output_directory
        
        # Get the path to the source directory for running as module
        src_path = Path(__file__).parent.parent.parent  # Go up to project root
        
        # Configure MCP server connection via stdio
        # The server will be launched as a subprocess and communicate via stdio
        mcp_connections = {
            "ai_book_composer_tools": StdioConnection(
                command="python",
                args=["-m", "ai_book_composer.mcp_server", "--stdio"],
                env={
                    "INPUT_DIRECTORY": str(Path(input_directory).resolve()),
                    "OUTPUT_DIRECTORY": str(Path(output_directory).resolve()),
                    "PYTHONPATH": str(src_path.resolve()),
                    "SKIP_TRANSCRIPTION": "1",  # Skip transcription for faster startup
                    **os.environ  # Include existing environment variables
                },
                transport="stdio"
            )
        }
        
        # Initialize MCP client to connect to the tool server
        self.mcp_client = MultiServerMCPClient(connections=mcp_connections)
        
        # Get tools from MCP server (this will start the server subprocess)
        # This is an async operation, so we need to run it in an event loop
        self.langchain_tools = self._get_tools_sync()
        
        # Create a tool name to tool object mapping for direct invocation
        self.tools_map = {tool.name: tool for tool in self.langchain_tools}
        
        # Initialize LLM with tools bound
        self.llm = get_llm(temperature=0.7).bind_tools(self.langchain_tools)
        self.prompts = load_prompts()
    
    def _get_tools_sync(self):
        """Get tools from MCP client synchronously."""
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we need to use a different approach
                # This happens in async contexts like Jupyter notebooks
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            # No event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.mcp_client.get_tools())
    
    def _invoke_tool(self, tool_name: str, **kwargs) -> Any:
        """Invoke a tool by name with arguments.
        
        Args:
            tool_name: Name of the tool to invoke
            **kwargs: Tool arguments
            
        Returns:
            Tool result
        """
        if tool_name not in self.tools_map:
            raise ValueError(f"Tool {tool_name} not found")
        
        tool = self.tools_map[tool_name]
        return tool.invoke(kwargs)
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        """Execute the next task in the plan.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
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
                result = self._gather_content(state)
            elif task_type == "plan_chapters":
                result = self._plan_chapters(state)
            elif task_type == "generate_single_chapter":
                result = self._generate_single_chapter(state, current_task)
            elif task_type == "compile_references":
                result = self._compile_references(state)
            elif task_type == "generate_book":
                result = self._generate_book(state)
            else:
                result = {
                    "current_task_index": current_task_index + 1,
                    "status": "executing"
                }
            
            progress.show_task(task_type, "completed")
            
            return result
    
    def _gather_content(self, state: AgentState) -> Dict[str, Any]:
        """Gather content from all source files."""
        files = state.get("files", [])
        gathered_content = {}
        
        progress.show_thought(f"Need to process {len(files)} source file(s) and extract their content")
        
        for i, file_info in enumerate(files, 1):
            file_path = file_info.get("path", "")
            file_name = file_info.get("name", "")
            extension = file_info.get("extension", "").lower()
            
            progress.show_action(f"Processing file {i}/{len(files)}: {file_name}")
            
            try:
                if extension in [".txt", ".md", ".rst"]:
                    # Read text file
                    progress.show_observation(f"Reading text file: {file_name}")
                    result = self._invoke_tool("read_text_file", file_path=file_path)
                    gathered_content[file_path] = {
                        "type": "text",
                        "content": result.get("content", "")
                    }
                elif extension in [".mp3", ".wav", ".m4a", ".flac"]:
                    # Transcribe audio
                    progress.show_observation(f"Transcribing audio file: {file_name}")
                    result = self._invoke_tool("transcribe_audio", file_path=file_path)
                    gathered_content[file_path] = {
                        "type": "audio_transcription",
                        "content": result.get("transcription", "")
                    }
                elif extension in [".mp4", ".avi", ".mov", ".mkv"]:
                    # Transcribe video
                    progress.show_observation(f"Transcribing video file: {file_name}")
                    result = self._invoke_tool("transcribe_video", file_path=file_path)
                    gathered_content[file_path] = {
                        "type": "video_transcription",
                        "content": result.get("transcription", "")
                    }
                else:
                    # Unsupported file type
                    progress.show_observation(f"⚠ Skipping unsupported file type: {file_name} ({extension})")
                    gathered_content[file_path] = {
                        "type": "unsupported",
                        "content": f"File type {extension} is not supported"
                    }
                    continue
                
                content_length = len(gathered_content.get(file_path, {}).get("content", ""))
                progress.show_observation(f"✓ Processed {file_name} ({content_length} characters)")
                
            except Exception as e:
                progress.show_observation(f"✗ Error processing {file_name}: {str(e)}")
                gathered_content[file_path] = {
                    "type": "error",
                    "content": f"Error processing file: {str(e)}"
                }
        
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
            progress.show_observation(f"⚠ Error listing images: {str(e)}")
        
        # Extract images from PDF files
        pdf_files = [f for f in files if f.get("extension", "").lower() == ".pdf"]
        if pdf_files:
            progress.show_action(f"Extracting images from {len(pdf_files)} PDF file(s)")
            for pdf_file in pdf_files:
                pdf_path = pdf_file.get("path", "")
                try:
                    result = self._invoke_tool("extract_images_from_pdf", file_path=pdf_path)
                    if result.get("success"):
                        extracted = result.get("images", [])
                        all_images.extend(extracted)
                        progress.show_observation(f"✓ Extracted {len(extracted)} image(s) from {pdf_file.get('name')}")
                except Exception as e:
                    progress.show_observation(f"⚠ Error extracting images from {pdf_file.get('name')}: {str(e)}")
        
        progress.show_observation(f"Image gathering complete: {len(all_images)} total image(s) available")
        
        return {
            "gathered_content": gathered_content,
            "images": all_images,
            "current_task_index": state.get("current_task_index", 0) + 1,
            "status": "executing"
        }
    
    def _plan_chapters(self, state: AgentState) -> Dict[str, Any]:
        """Plan the book chapters based on gathered content."""
        gathered_content = state.get("gathered_content", {})
        language = state.get("language", "en-US")
        
        progress.show_thought("Analyzing gathered content to determine optimal chapter structure")
        
        # Create content summary
        content_summary = self._summarize_content(gathered_content)
        
        progress.show_action("Using AI to plan book chapters")
        
        # Load prompts from YAML and format with placeholders
        system_prompt_template = self.prompts['executor']['chapter_planning_system_prompt']
        user_prompt_template = self.prompts['executor']['chapter_planning_user_prompt']
        
        system_prompt = system_prompt_template.format(language=language)
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
        
        # Dynamically add individual chapter generation tasks to the plan
        plan = state.get("plan", [])
        current_task_index = state.get("current_task_index", 0)
        
        progress.show_action("Creating individual tasks for each chapter generation")
        
        # Find the "generate_chapters" task and replace it with individual chapter tasks
        new_plan = []
        for i, task in enumerate(plan):
            if i < current_task_index + 1:
                # Keep tasks that are already done or current
                new_plan.append(task)
            elif task.get("task") == "generate_chapters":
                # Replace with individual chapter tasks
                for chapter_info in chapter_list:
                    chapter_num = chapter_info.get("number")
                    chapter_title = chapter_info.get("title")
                    new_plan.append({
                        "task": "generate_single_chapter",
                        "description": f"Generate Chapter {chapter_num}: {chapter_title}",
                        "status": "pending",
                        "chapter_number": chapter_num,
                        "chapter_title": chapter_title,
                        "chapter_description": chapter_info.get("description", "")
                    })
            else:
                # Keep other tasks (compile_references, generate_book)
                new_plan.append(task)
        
        progress.show_observation(f"Updated plan with {len(chapter_list)} individual chapter generation tasks")
        
        return {
            "chapter_list": chapter_list,
            "plan": new_plan,
            "current_task_index": current_task_index + 1,
            "status": "executing"
        }
    
    def _generate_single_chapter(self, state: AgentState, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single chapter."""
        chapter_num = task.get("chapter_number", 0)
        chapter_title = task.get("chapter_title", "Untitled")
        chapter_desc = task.get("chapter_description", "")
        gathered_content = state.get("gathered_content", {})
        language = state.get("language", "en-US")
        
        progress.show_thought(f"Generating Chapter {chapter_num}: {chapter_title}")
        progress.show_chapter_info(chapter_num, chapter_title, "generating")
        progress.show_action(f"Creating content for Chapter {chapter_num}")
        
        # Generate chapter content
        content = self._generate_chapter_content(
            chapter_num,
            chapter_title,
            chapter_desc,
            gathered_content,
            language
        )
        
        # Save chapter
        self._invoke_tool("write_chapter", chapter_number=chapter_num, title=chapter_title, content=content)
        
        word_count = len(content.split())
        progress.show_observation(f"✓ Chapter {chapter_num} complete ({word_count} words)")
        
        # Add chapter to existing list (create new list to maintain immutability)
        chapters = list(state.get("chapters", []))
        chapters.append({
            "number": chapter_num,
            "title": chapter_title,
            "content": content
        })
        
        return {
            "chapters": chapters,
            "current_task_index": state.get("current_task_index", 0) + 1,
            "status": "executing"
        }
    
    def _generate_chapter_content(
        self,
        chapter_num: int,
        title: str,
        description: str,
        gathered_content: Dict[str, Any],
        language: str
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
        
        system_prompt = system_prompt_template.format(
            language=language,
            chapter_number=chapter_num,
            title=title,
            description=description
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
    
    def _compile_references(self, state: AgentState) -> Dict[str, Any]:
        """Compile list of references."""
        files = state.get("files", [])
        
        progress.show_action("Compiling list of source file references")
        
        references = []
        for file_info in files:
            file_path = file_info.get("path", "")
            file_name = file_info.get("name", "")
            references.append(f"{file_name} - Source file: {file_path}")
        
        progress.show_observation(f"Compiled {len(references)} reference(s)")
        
        return {
            "references": references,
            "current_task_index": state.get("current_task_index", 0) + 1,
            "status": "executing"
        }
    
    def _generate_book(self, state: AgentState) -> Dict[str, Any]:
        """Generate the final book."""
        title = state.get("book_title", "Composed Book")
        author = state.get("book_author", "AI Book Composer")
        chapters = state.get("chapters", [])
        references = state.get("references", [])
        
        progress.show_action(f"Generating final book: '{title}' by {author}")
        progress.show_observation(f"Compiling {len(chapters)} chapters and {len(references)} references into RTF format")
        
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
            "current_task_index": state.get("current_task_index", 0) + 1,
            "status": "book_generated"
        }
    
    def _summarize_content(self, gathered_content: Dict[str, Any]) -> str:
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
                    file_content = content[:MAX_CONTENT_FOR_CHAPTER_PLANNING] + f"\n\n[Content truncated - {len(content) - MAX_CONTENT_FOR_CHAPTER_PLANNING} more characters]"
                else:
                    file_content = content
            else:
                file_content = "No content"
            
            summary.append(f"File: {Path(file_path).name} ({content_type})\nContent:\n{file_content}\n")
        return "\n".join(summary)
    
    def _parse_chapter_list(self, response_content: str) -> List[Dict[str, Any]]:
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
