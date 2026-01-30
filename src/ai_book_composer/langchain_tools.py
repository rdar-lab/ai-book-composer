"""LangChain tools integration with MCP-style tool definitions.

This module provides a bridge between MCP-style tool definitions and LangChain tools,
allowing tools to be easily added while remaining compatible with LangChain's execution model.
"""

from typing import Dict, Any, List, Optional
from langchain_core.tools import StructuredTool

from .tools import (
    FileListingTool,
    TextFileReaderTool,
    AudioTranscriptionTool,
    VideoTranscriptionTool,
    ChapterWriterTool,
    ChapterListWriterTool,
    BookGeneratorTool
)
from .logging_config import logger


class ToolRegistry:
    """Registry for managing tools that can be used with LangChain LLMs."""
    
    def __init__(self, input_directory: str, output_directory: str, skip_transcription: bool = False):
        """Initialize tool registry with tool instances.
        
        Args:
            input_directory: Directory containing source files
            output_directory: Directory for output files
            skip_transcription: Skip initialization of transcription tools (useful for testing)
        """
        self.input_directory = input_directory
        self.output_directory = output_directory
        
        # Initialize tool instances
        self.file_lister = FileListingTool(input_directory)
        self.text_reader = TextFileReaderTool()
        self.chapter_writer = ChapterWriterTool(output_directory)
        self.chapter_list_writer = ChapterListWriterTool(output_directory)
        self.book_generator = BookGeneratorTool(output_directory)
        
        # Initialize transcription tools only if not skipped
        if not skip_transcription:
            self.audio_transcriber = AudioTranscriptionTool()
            self.video_transcriber = VideoTranscriptionTool()
        else:
            self.audio_transcriber = None
            self.video_transcriber = None
        
        logger.info(f"Tool registry initialized: input={input_directory}, output={output_directory}")
    
    def list_files(self) -> List[Dict[str, Any]]:
        """List all files in the input directory.
        
        Returns:
            List of file information dictionaries with path, name, extension, and size.
        """
        logger.debug("Tool: list_files called")
        result = self.file_lister.run()
        logger.debug(f"Tool: list_files returned {len(result)} files")
        return result
    
    def read_text_file(
        self,
        file_path: str,
        start_line: int = 1,
        end_line: Optional[int] = None
    ) -> Dict[str, Any]:
        """Read text file content with optional line range.
        
        Args:
            file_path: Path to the text file
            start_line: Starting line number (1-indexed), defaults to 1
            end_line: Ending line number (inclusive), defaults to max_lines from config
        
        Returns:
            Dictionary with content, line information, and metadata
        """
        logger.debug(f"Tool: read_text_file called for {file_path}")
        result = self.text_reader.run(file_path, start_line, end_line)
        logger.debug(f"Tool: read_text_file returned {len(result.get('content', ''))} characters")
        return result
    
    def transcribe_audio(self, file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio file to text.
        
        Args:
            file_path: Path to the audio file
            language: Optional language code (e.g., 'en', 'he' for Hebrew). If None, auto-detects.
        
        Returns:
            Dictionary with transcription, segments, language, and duration
        """
        if self.audio_transcriber is None:
            raise RuntimeError("Audio transcription not available (skip_transcription=True)")
        
        logger.info(f"Tool: transcribe_audio called for {file_path}, language: {language or 'auto-detect'}")
        result = self.audio_transcriber.run(file_path, language)
        logger.info(f"Tool: transcribe_audio completed for {file_path}")
        return result
    
    def transcribe_video(self, file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe video file to text.
        
        Args:
            file_path: Path to the video file
            language: Optional language code (e.g., 'en', 'he' for Hebrew). If None, auto-detects.
        
        Returns:
            Dictionary with transcription, segments, language, and duration
        """
        if self.video_transcriber is None:
            raise RuntimeError("Video transcription not available (skip_transcription=True)")
        
        logger.info(f"Tool: transcribe_video called for {file_path}, language: {language or 'auto-detect'}")
        result = self.video_transcriber.run(file_path, language)
        logger.info(f"Tool: transcribe_video completed for {file_path}")
        return result
    
    def write_chapter(
        self,
        chapter_number: int,
        title: str,
        content: str
    ) -> Dict[str, Any]:
        """Write a chapter to a file in the output directory.
        
        Args:
            chapter_number: Chapter number
            title: Chapter title
            content: Chapter content
        
        Returns:
            Dictionary with success status, file path, and metadata
        """
        logger.info(f"Tool: write_chapter called for Chapter {chapter_number}: {title}")
        result = self.chapter_writer.run(chapter_number, title, content)
        logger.info(f"Tool: write_chapter completed for Chapter {chapter_number}")
        return result
    
    def write_chapter_list(self, chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Write the list of planned chapters to a file.
        
        Args:
            chapters: List of chapter dictionaries with structure information
        
        Returns:
            Dictionary with success status, file path, and chapter count
        """
        logger.info(f"Tool: write_chapter_list called with {len(chapters)} chapters")
        result = self.chapter_list_writer.run(chapters)
        logger.info(f"Tool: write_chapter_list completed")
        return result
    
    def generate_book(
        self,
        book_title: str,
        book_author: str,
        chapters: Optional[List[Dict[str, str]]] = None,
        references: Optional[List[str]] = None,
        output_filename: str = "book.rtf"
    ) -> Dict[str, Any]:
        """Generate the final book in RTF format.
        
        Args:
            book_title: Title of the book
            book_author: Author name
            chapters: List of chapter dictionaries with 'title' and 'content' (optional)
            references: List of reference strings (optional)
            output_filename: Output filename (default: "book.rtf")
        
        Returns:
            Dictionary with success status and output file path
        """
        logger.info(f"Tool: generate_book called for '{book_title}' by {book_author}")
        
        # If chapters not provided, read from chapter files in output directory
        if chapters is None:
            from pathlib import Path
            chapters = []
            chapter_files = sorted(Path(self.output_directory).glob("chapter_*.txt"))
            for chapter_file in chapter_files:
                with open(chapter_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract title from first line
                    lines = content.split('\n')
                    title_line = lines[0] if lines else "Untitled"
                    chapter_content = '\n'.join(lines[2:]) if len(lines) > 2 else content
                    chapters.append({
                        "title": title_line.replace('#', '').strip(),
                        "content": chapter_content
                    })
        
        # If references not provided, use empty list
        if references is None:
            references = []
        
        result = self.book_generator.run(book_title, book_author, chapters, references, output_filename)
        logger.info(f"Tool: generate_book completed")
        return result
    
    def get_langchain_tools(self) -> List[StructuredTool]:
        """Get LangChain-compatible tools for use with LLM.
        
        Returns:
            List of StructuredTool instances that can be bound to LangChain LLMs
        """
        tools = []
        
        # File operations tools
        tools.append(StructuredTool.from_function(
            func=self.list_files,
            name="list_files",
            description="List all files in the input directory. Returns a list of dictionaries with file information including path, name, extension, and size."
        ))
        
        tools.append(StructuredTool.from_function(
            func=self.read_text_file,
            name="read_text_file",
            description="Read content from text files (txt, md, rst, docx, rtf, pdf) with optional line range. Supports pagination for large files."
        ))
        
        # Transcription tools (only if available)
        if self.audio_transcriber is not None:
            tools.append(StructuredTool.from_function(
                func=self.transcribe_audio,
                name="transcribe_audio",
                description="Transcribe audio files (mp3, wav, m4a, flac, ogg) to text using Whisper. Supports multiple languages including Hebrew ('he'). Returns transcription with segments, language, and duration. Results are cached."
            ))
        
        if self.video_transcriber is not None:
            tools.append(StructuredTool.from_function(
                func=self.transcribe_video,
                name="transcribe_video",
                description="Transcribe video files (mp4, avi, mov, mkv) to text by extracting audio and using Whisper. Supports chunking for large files and multiple languages including Hebrew ('he'). Results are cached."
            ))
        
        # Book generation tools
        tools.append(StructuredTool.from_function(
            func=self.write_chapter,
            name="write_chapter",
            description="Write a chapter to a file in the output directory. Creates a text file with the chapter number, title, and content."
        ))
        
        tools.append(StructuredTool.from_function(
            func=self.write_chapter_list,
            name="write_chapter_list",
            description="Write the list of planned chapters to a JSON file. Useful for saving chapter structure before generating individual chapters."
        ))
        
        tools.append(StructuredTool.from_function(
            func=self.generate_book,
            name="generate_book",
            description="Generate the final book in RTF format. Can read chapters from files or accept them as parameters. Creates a complete book with title page, table of contents, chapters, and references."
        ))
        
        logger.info(f"Created {len(tools)} LangChain tools")
        return tools
