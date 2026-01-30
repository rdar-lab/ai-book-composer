"""MCP (Model Context Protocol) Server for AI Book Composer Tools.

This module exposes all book composition tools through the MCP protocol,
allowing LLMs to easily discover and use them.
"""
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from mcp.server import FastMCP

from .config import Settings
from .tools.base_tools import (
    FileListingTool,
    TextFileReaderTool,
    AudioTranscriptionTool,
    VideoTranscriptionTool,
    ChapterWriterTool,
    ChapterListWriterTool,
    ImageExtractorTool,
    ImageListingTool,
)
from .tools.book_generator import (
    BookGeneratorTool
)

logger = logging.getLogger(__name__)

# Initialize MCP server with configuration
mcp = FastMCP(
    name="AI Book Composer MCP Server",
    instructions="""AI Book Composer MCP Server

This server provides tools for composing books from various source files including text, audio, and video.

Available tools:
- list_files: List all files in the input directory
- list_images: List all image files in the input directory
- read_text_file: Read content from text files (txt, md, rst, docx, rtf, pdf)
- extract_images_from_pdf: Extract images from PDF files
- transcribe_audio: Transcribe audio files (mp3, wav, m4a, flac, ogg)
- transcribe_video: Transcribe video files (mp4, avi, mov, mkv)
- write_chapter: Write a chapter to a file
- write_chapter_list: Write the list of planned chapters
- generate_book: Generate the final book in RTF format with embedded images
""",
)

# Global tool instances (will be initialized when directories are set)
_settings: Optional[Settings] = None
_input_directory: Optional[str] = None
_output_directory: Optional[str] = None
_file_lister: Optional[FileListingTool] = None
_text_reader: Optional[TextFileReaderTool] = None
_audio_transcriber: Optional[AudioTranscriptionTool] = None
_video_transcriber: Optional[VideoTranscriptionTool] = None
_chapter_writer: Optional[ChapterWriterTool] = None
_chapter_list_writer: Optional[ChapterListWriterTool] = None
_image_extractor: Optional[ImageExtractorTool] = None
_image_lister: Optional[ImageListingTool] = None
_book_generator: Optional[BookGeneratorTool] = None


def initialize_tools(settings: Settings, input_directory: str, output_directory: str) -> None:
    """Initialize tool instances with directories.
    
    Args:
        settings: The project settings
        input_directory: Directory containing source files
        output_directory: Directory for output files
    """
    global _settings, _input_directory, _output_directory
    global _file_lister, _text_reader, _audio_transcriber, _video_transcriber
    global _chapter_writer, _chapter_list_writer, _image_extractor, _image_lister, _book_generator

    _settings = settings

    _input_directory = input_directory
    _output_directory = output_directory

    _file_lister = FileListingTool(_settings, input_directory)
    _text_reader = TextFileReaderTool(_settings)
    _image_lister = ImageListingTool(_settings, input_directory)
    _image_extractor = ImageExtractorTool(_settings, output_directory)

    _audio_transcriber = AudioTranscriptionTool(_settings)
    _video_transcriber = VideoTranscriptionTool(_settings)

    _chapter_writer = ChapterWriterTool(_settings, output_directory)
    _chapter_list_writer = ChapterListWriterTool(_settings, output_directory)
    _book_generator = BookGeneratorTool(_settings, output_directory)

    logger.info(f"MCP tools initialized: input={input_directory}, output={output_directory}")


@mcp.tool(description="List all files in the input directory")
async def list_files() -> List[Dict[str, Any]]:
    """List all files in the input directory.

    Returns:
        List of file information dictionaries with path, name, extension, and size.
    """
    if _file_lister is None:
        raise RuntimeError("Tools not initialized. Call initialize_tools() first.")

    logger.debug("MCP: list_files called")
    result = _file_lister.run()
    logger.debug(f"MCP: list_files returned {len(result)} files")
    return result


@mcp.tool(description="List all image files in the input directory")
async def list_images() -> List[Dict[str, Any]]:
    """List all image files in the input directory.

    Returns:
        List of image file information dictionaries with path, filename, format, and size.
    """
    if _image_lister is None:
        raise RuntimeError("Tools not initialized. Call initialize_tools() first.")

    logger.debug("MCP: list_images called")
    result = _image_lister.run()
    logger.debug(f"MCP: list_images returned {len(result)} images")
    return result


@mcp.tool(description="Extract images from PDF files and save them to output directory")
async def extract_images_from_pdf(file_path: str) -> Dict[str, Any]:
    """Extract images from a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        Dictionary with extracted image paths and metadata
    """
    if _image_extractor is None:
        raise RuntimeError("Tools not initialized. Call initialize_tools() first.")

    logger.info(f"MCP: extract_images_from_pdf called for {file_path}")
    result = _image_extractor.run(file_path)
    logger.info(f"MCP: extract_images_from_pdf extracted {result.get('count', 0)} images")
    return result


@mcp.tool(description="Read content from text files (txt, md, rst, docx, rtf, pdf)")
async def read_text_file(
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
    if _text_reader is None:
        raise RuntimeError("Tools not initialized. Call initialize_tools() first.")

    logger.debug(f"MCP: read_text_file called for {file_path}")
    result = _text_reader.run(file_path, start_line, end_line)
    logger.debug(f"MCP: read_text_file returned {len(result.get('content', ''))} characters")
    return result


@mcp.tool(description="Transcribe audio files (mp3, wav, m4a, flac, ogg) with optional language specification")
async def transcribe_audio(file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe audio file to text.

    Args:
        file_path: Path to the audio file
        language: Optional language code (e.g., 'en', 'he' for Hebrew). If None, auto-detects.

    Returns:
        Dictionary with transcription, segments, language, and duration
    """
    if _audio_transcriber is None:
        raise RuntimeError("Tools not initialized. Call initialize_tools() first.")

    logger.info(f"MCP: transcribe_audio called for {file_path}, language: {language or 'auto-detect'}")
    result = _audio_transcriber.run(file_path, language)
    logger.info(f"MCP: transcribe_audio completed for {file_path}")
    return result


@mcp.tool(description="Transcribe video files (mp4, avi, mov, mkv) with optional language specification")
async def transcribe_video(file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe video file to text.

    Args:
        file_path: Path to the video file
        language: Optional language code (e.g., 'en', 'he' for Hebrew). If None, auto-detects.

    Returns:
        Dictionary with transcription, segments, language, and duration
    """
    if _video_transcriber is None:
        raise RuntimeError("Tools not initialized. Call initialize_tools() first.")

    logger.info(f"MCP: transcribe_video called for {file_path}, language: {language or 'auto-detect'}")
    result = _video_transcriber.run(file_path, language)
    logger.info(f"MCP: transcribe_video completed for {file_path}")
    return result


@mcp.tool(description="Write a chapter to a file")
async def write_chapter(
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
    if _chapter_writer is None:
        raise RuntimeError("Tools not initialized. Call initialize_tools() first.")

    logger.info(f"MCP: write_chapter called for Chapter {chapter_number}: {title}")
    result = _chapter_writer.run(chapter_number, title, content)
    logger.info(f"MCP: write_chapter completed for Chapter {chapter_number}")
    return result


@mcp.tool(description="Write the list of planned chapters")
async def write_chapter_list(chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Write the list of planned chapters to a file.

    Args:
        chapters: List of chapter dictionaries with structure information

    Returns:
        Dictionary with success status, file path, and chapter count
    """
    if _chapter_list_writer is None:
        raise RuntimeError("Tools not initialized. Call initialize_tools() first.")

    logger.info(f"MCP: write_chapter_list called with {len(chapters)} chapters")
    result = _chapter_list_writer.run(chapters)
    logger.info(f"MCP: write_chapter_list completed")
    return result


@mcp.tool(description="Generate the final book in RTF format")
async def generate_book(
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
        chapters: List of chapter dictionaries with 'title' and 'content' (optional, will read from output dir if not provided)
        references: List of reference strings (optional, will be empty if not provided)
        output_filename: Output filename (default: "book.rtf")
    
    Returns:
        Dictionary with success status and output file path
    """
    if _book_generator is None:
        raise RuntimeError("Tools not initialized. Call initialize_tools() first.")

    logger.info(f"MCP: generate_book called for '{book_title}' by {book_author}")

    # If chapters not provided, read from chapter files in output directory
    if chapters is None:
        chapters = []
        chapter_files = sorted(_book_generator.output_directory.glob("chapter_*.txt"))
        for chapter_file in chapter_files:
            with open(chapter_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract title from first line (e.g., "# Chapter 1: Title")
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

    result = _book_generator.run(book_title, book_author, chapters, references, output_filename)
    logger.info(f"MCP: generate_book completed")
    return result


def get_mcp_server() -> FastMCP:
    """Get the MCP server instance.
    
    Returns:
        The FastMCP server instance
    """
    return mcp


if __name__ == "__main__":
    """Run the MCP server in standalone mode (stdio or HTTP)."""
    import sys

    # Check if running in stdio mode (when launched by MultiServerMCPClient)
    # In stdio mode, sys.stdin and sys.stdout are used for communication
    if len(sys.argv) > 1 and sys.argv[1] == "--stdio":
        # Stdio mode for MCP adapter integration

        # Get Settings
        settings_path = os.getenv("SETTINGS_PATH")
        settings = Settings(settings_path)

        # Get directories from environment variables
        input_dir = os.getenv("INPUT_DIRECTORY", ".")
        output_dir = os.getenv("OUTPUT_DIRECTORY", "./output")

        # Validate and create directories
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
            sys.exit(1)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize tools
        initialize_tools(settings, str(input_path.resolve()), str(output_path.resolve()))

        mcp.run()
    else:
        raise Exception(
            "HTTP mode is no longer supported in this script. Please use the MultiServerMCPClient to connect via stdio.")
