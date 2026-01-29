"""Executor agent - Phase 2 of Deep-Agent architecture."""

from typing import Dict, Any, List
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import get_llm
from ..config import load_prompts
from ..progress_display import progress
from ..tools import (
    FileListingTool,
    TextFileReaderTool,
    AudioTranscriptionTool,
    VideoTranscriptionTool,
    ChapterWriterTool,
    ChapterListWriterTool,
    BookGeneratorTool
)
from .state import AgentState

# Constants
MIN_SUBSTANTIAL_CONTENT_LENGTH = 100
MAX_CONTENT_PREVIEW_LENGTH = 500
CONTENT_PREVIEW_LENGTH = 200
MIN_CHAPTER_COUNT = 3
MAX_CHAPTER_COUNT = 10


class ExecutorAgent:
    """The Executor (Worker) - performs tasks using available tools."""
    
    def __init__(self, input_directory: str, output_directory: str):
        self.input_directory = input_directory
        self.output_directory = output_directory
        
        # Initialize tools
        self.file_lister = FileListingTool(input_directory)
        self.text_reader = TextFileReaderTool()
        self.audio_transcriber = AudioTranscriptionTool()
        self.video_transcriber = VideoTranscriptionTool()
        self.chapter_writer = ChapterWriterTool(output_directory)
        self.chapter_list_writer = ChapterListWriterTool(output_directory)
        self.book_generator = BookGeneratorTool(output_directory)
        
        self.llm = get_llm(temperature=0.7)
        self.prompts = load_prompts()
    
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
            elif task_type == "generate_chapters":
                result = self._generate_chapters(state)
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
                    result = self.text_reader.run(file_path)
                    gathered_content[file_path] = {
                        "type": "text",
                        "content": result.get("content", "")
                    }
                elif extension in [".mp3", ".wav", ".m4a", ".flac"]:
                    # Transcribe audio
                    progress.show_observation(f"Transcribing audio file: {file_name}")
                    result = self.audio_transcriber.run(file_path)
                    gathered_content[file_path] = {
                        "type": "audio_transcription",
                        "content": result.get("transcription", "")
                    }
                elif extension in [".mp4", ".avi", ".mov", ".mkv"]:
                    # Transcribe video
                    progress.show_observation(f"Transcribing video file: {file_name}")
                    result = self.video_transcriber.run(file_path)
                    gathered_content[file_path] = {
                        "type": "video_transcription",
                        "content": result.get("transcription", "")
                    }
                
                content_length = len(gathered_content.get(file_path, {}).get("content", ""))
                progress.show_observation(f"✓ Processed {file_name} ({content_length} characters)")
                
            except Exception as e:
                progress.show_observation(f"✗ Error processing {file_name}: {str(e)}")
                gathered_content[file_path] = {
                    "type": "error",
                    "content": f"Error processing file: {str(e)}"
                }
        
        progress.show_observation(f"Content gathering complete: {len(gathered_content)} file(s) processed")
        
        return {
            "gathered_content": gathered_content,
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
        self.chapter_list_writer.run(chapter_list)
        
        return {
            "chapter_list": chapter_list,
            "current_task_index": state.get("current_task_index", 0) + 1,
            "status": "executing"
        }
    
    def _generate_chapters(self, state: AgentState) -> Dict[str, Any]:
        """Generate all chapters."""
        chapter_list = state.get("chapter_list", [])
        gathered_content = state.get("gathered_content", {})
        language = state.get("language", "en-US")
        
        progress.show_thought(f"Ready to generate {len(chapter_list)} chapter(s) using gathered content")
        
        chapters = []
        
        for i, chapter_info in enumerate(chapter_list, 1):
            chapter_num = chapter_info.get("number", 0)
            chapter_title = chapter_info.get("title", "Untitled")
            chapter_desc = chapter_info.get("description", "")
            
            progress.show_chapter_info(chapter_num, chapter_title, "generating")
            progress.show_action(f"Generating content for Chapter {chapter_num}: {chapter_title}")
            
            # Generate chapter content
            content = self._generate_chapter_content(
                chapter_num,
                chapter_title,
                chapter_desc,
                gathered_content,
                language
            )
            
            # Save chapter
            self.chapter_writer.run(chapter_num, chapter_title, content)
            
            word_count = len(content.split())
            progress.show_observation(f"✓ Chapter {chapter_num} complete ({word_count} words)")
            
            chapters.append({
                "number": chapter_num,
                "title": chapter_title,
                "content": content
            })
        
        progress.show_observation(f"All {len(chapters)} chapter(s) generated successfully")
        
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
        
        result = self.book_generator.run(
            title=title,
            author=author,
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
        """Summarize gathered content."""
        summary = []
        for file_path, content_info in gathered_content.items():
            content = content_info.get("content", "")
            content_type = content_info.get("type", "unknown")
            preview = content[:CONTENT_PREVIEW_LENGTH] if content else "No content"
            summary.append(f"- {Path(file_path).name} ({content_type}): {preview}...")
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
