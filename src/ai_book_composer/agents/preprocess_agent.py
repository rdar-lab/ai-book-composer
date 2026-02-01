"""Preprocess agent - Initial preprocess phase of Deep-Agent architecture."""
import logging
from pathlib import Path
from typing import Dict, Any, List

from .agent_base import AgentBase
from .state import AgentState
from .. import progress_display
from ..config import Settings
from ..parallel_utils import execute_parallel
from ..progress_display import progress
from ..utils import file_utils
from ..utils.file_utils import list_input_files, read_text_file, read_audio_file, read_video_file, \
    extract_images_from_pdf, list_images, write_cache

logger = logging.getLogger(__name__)

_MIN_LENGTH_FOR_SUMMARIZATION = 2000  # Minimum length of content to consider summarization
_MAX_LENGTH_FOR_SUMMARIZATION = 16384  # Maximum length of content to summarize


class PreprocessAgent(AgentBase):
    """The Executor (Worker) - performs tasks using available tools."""

    def __init__(self, settings: Settings, input_directory: str, output_directory: str):
        super().__init__(
            settings,
            input_directory=input_directory,
            output_directory=output_directory
        )

    def list_files(self) -> List[Dict[str, Any]]:
        """List all files in the input directory using the appropriate tool.

        Returns:
            List of file information dictionaries
        """
        files = list_input_files(self.settings, self.input_directory)

        progress_display.progress.show_files(files)

        self.state['files'] = files
        return files

    def preprocess(self, state: AgentState) -> Dict[str, Any]:
        """Execute the preprocessing agent.

        Returns:
            Updated state
        """

        self.state = state

        with progress.agent_context(
                "PreProcessor Agent",
                "Preprocess Execution Task - gather content from source files"
        ):
            self.gather_content()

            return {
                "status": "preprocessed",
                "files": self.state.get("files", []),
                "gathered_content": self.state.get("gathered_content", {}),
                "images": self.state.get("images", []),
            }

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
        language = self.state.get("language", "en-US")

        try:
            if extension in [".txt", ".md", ".rst", ".docx", ".rtf", ".pdf"]:
                result = read_text_file(self.settings, file_path)
                return {
                    "file_path": file_path,
                    "file_name": file_name,
                    "type": "text",
                    "content": result.get("content", ""),
                    "status": "success"
                }
            elif extension in [".mp3", ".wav", ".m4a", ".flac"]:
                result = read_audio_file(self.settings, file_path, language)
                return {
                    "file_path": file_path,
                    "file_name": file_name,
                    "type": "audio_transcription",
                    "content": result.get("transcription", ""),
                    "status": "success"
                }
            elif extension in [".mp4", ".avi", ".mov", ".mkv"]:
                result = read_video_file(self.settings, file_path, language)
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
            result = extract_images_from_pdf(self.settings, pdf_path)
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

    def gather_content(self) -> Dict[str, Any]:
        progress.show_thought(f"Need to find the files in the input directory and extract their content")

        files = self.list_files()

        gathered_content = self._gather_all_content(files)
        self._summarize_all_files(gathered_content)

        all_images = self._gather_images(files)

        self.state["gathered_content"] = gathered_content
        self.state["images"] = all_images

        return {
            "gathered_content": gathered_content,
            "images": all_images,
            "current_task_index": self.state.get("current_task_index", 0) + 1,
            "status": "executing"
        }

    def _summarize_all_files(self, gathered_content):
        progress.show_thought("I need to summarize gathered file contents for easier processing")
        execute_parallel(self.settings, self._summerize_gathered_file, list(gathered_content.values()))

    def _summerize_gathered_file(self, gathered_file):
        language = self.state.get('language', 'en-US')

        file_name = gathered_file.get('name', '')
        file_path = gathered_file.get('path', '')
        file_content = gathered_file.get('content', '')

        cache_name = file_utils.get_cache_path(self.settings, Path(file_path), prefix="summary_", language=language)
        file_summary = file_utils.read_cache(cache_name)
        if not file_summary:

            if len(file_content) < _MIN_LENGTH_FOR_SUMMARIZATION:
                file_summary = file_content
            else:
                try:
                    system_prompt_template = self.prompts['preprocessor'].get('summarization_system_prompt')
                    user_prompt_template = self.prompts['preprocessor'].get('summarization_user_prompt')

                    system_prompt = system_prompt_template.format(language=language)
                    user_prompt = user_prompt_template.format(
                        file_name=file_name,
                        file_content=file_content[:_MAX_LENGTH_FOR_SUMMARIZATION]
                    )

                    logger.info(f"Summarizing gathered file content for easier processing. file_name={file_name}")
                    progress.show_action(
                        f"Summarizing gathered file content for easier processing. file_name={file_name}")
                    file_summary = self._invoke_llm(system_prompt, user_prompt)

                    logger.info(f"File content summarized successfully. Summary={file_summary}")
                    progress.show_thought("File content summarized successfully.")
                except Exception as exp:
                    logger.exception(
                        f"Error summarizing file content for file_name={file_name}. Using original content. exp={exp}")
                    file_summary = file_content[:_MIN_LENGTH_FOR_SUMMARIZATION]
            write_cache(cache_name, file_summary)

        gathered_file['summary'] = file_summary[:_MIN_LENGTH_FOR_SUMMARIZATION]
        return gathered_file

    def _gather_images(self, files: list[dict[str, Any]]) -> list[Any]:
        # Gather images from input directory and extract from PDFs
        progress.show_thought("Gathering images from input directory")

        all_images = []

        try:
            # List existing images in input directory
            progress.show_action("Listing existing images in input directory")
            existing_images = list_images(self.settings, self.input_directory)
            all_images.extend(existing_images)
            progress.show_observation(f"✓ Found {len(existing_images)} existing image(s)")
        except Exception as e:
            logger.exception(f"Error listing images: {str(e)}")
            progress.show_observation(f"⚠ Error listing images: {str(e)}")

        # Extract images from PDF files (potentially in parallel)
        if self.settings.image_processing.extract_from_pdf:
            pdf_files = [f for f in files if f.get("extension", "").lower() == ".pdf"]
            if pdf_files:
                progress.show_action(f"Extracting images from {len(pdf_files)} PDF file(s)")

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
        return all_images

    def _gather_all_content(self, files: list[dict[str, Any]]) -> dict[Any, Any]:
        gathered_content = {}

        progress.show_thought(f"Need to process {len(files)} source file(s) and extract their content")

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
                    "name": file_name,
                    "path": file_path,
                    "content": content
                }
                content_length = len(content)
                progress.show_observation(f"✓ Processed {file_name} ({content_length} characters)")
            elif status == "skipped":
                gathered_content[file_path] = {
                    "type": file_type,
                    "name": file_name,
                    "path": file_path,
                    "content": content
                }
                progress.show_observation(f"⚠ Skipping unsupported file type: {file_name}")
            else:  # error
                gathered_content[file_path] = {
                    "type": file_type,
                    "name": file_name,
                    "path": file_path,
                    "content": content
                }
                progress.show_observation(f"✗ Error processing {file_name}: {content}")
        return gathered_content
