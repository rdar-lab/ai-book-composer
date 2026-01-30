#!/usr/bin/env python3
"""Standalone MCP server for AI Book Composer.

This script starts an MCP server that exposes all book composition tools,
allowing LLMs to interact with the tools through the Model Context Protocol.

Usage:
    python run_mcp_server.py [input_directory] [output_directory]
    
    Or set environment variables:
    INPUT_DIRECTORY=/path/to/input OUTPUT_DIRECTORY=/path/to/output python run_mcp_server.py
    
Example:
    python run_mcp_server.py ./input ./output
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from ai_book_composer.mcp_server import mcp, initialize_tools
from ai_book_composer.config import settings


def main():
    """Run the MCP server."""
    import asyncio
    
    # Get directories from environment variables or arguments
    input_dir = os.getenv("INPUT_DIRECTORY", ".")
    output_dir = os.getenv("OUTPUT_DIRECTORY", "./output")
    
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Validate directories
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize tools
    initialize_tools(str(input_path.resolve()), str(output_path.resolve()))
    
    # Display startup information
    print("=" * 60)
    print("AI Book Composer MCP Server")
    print("=" * 60)
    print(f"Input directory:  {input_path.resolve()}")
    print(f"Output directory: {output_path.resolve()}")
    print(f"Server endpoint:  http://{settings.mcp_server.host}:{settings.mcp_server.port}")
    print("=" * 60)
    print("\nAvailable tools:")
    print("  - list_files: List all files in the input directory")
    print("  - read_text_file: Read content from text files")
    print("  - transcribe_audio: Transcribe audio files")
    print("  - transcribe_video: Transcribe video files")
    print("  - write_chapter: Write a chapter to a file")
    print("  - write_chapter_list: Write the list of planned chapters")
    print("  - generate_book: Generate the final book in RTF format")
    print("=" * 60)
    print("\nServer is running. Press Ctrl+C to stop.\n")
    
    # Run the server
    try:
        asyncio.run(mcp.run())
    except KeyboardInterrupt:
        print("\n\nShutting down MCP server...")
        sys.exit(0)


if __name__ == "__main__":
    main()
