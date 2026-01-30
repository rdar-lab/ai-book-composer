#!/usr/bin/env python3
"""Example of using the AI Book Composer MCP Server.

This example demonstrates how to:
1. Start the MCP server
2. Connect to it from a client
3. Call tools through the MCP protocol
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from ai_book_composer.mcp_server import (
    mcp,
    initialize_tools,
    list_files,
    read_text_file,
    write_chapter,
    write_chapter_list,
    generate_book
)


async def example_usage():
    """Example of using MCP tools programmatically."""
    
    # Setup test directories
    test_input = Path("/tmp/mcp_example_input")
    test_output = Path("/tmp/mcp_example_output")
    test_input.mkdir(parents=True, exist_ok=True)
    test_output.mkdir(parents=True, exist_ok=True)
    
    # Create some test files
    (test_input / "chapter1.txt").write_text("This is the content of chapter 1.")
    (test_input / "chapter2.md").write_text("# Chapter 2\n\nThis is chapter 2 content.")
    
    print("=" * 60)
    print("AI Book Composer MCP Server - Example Usage")
    print("=" * 60)
    
    # Initialize tools
    initialize_tools(str(test_input), str(test_output))
    print(f"\n✓ Tools initialized")
    print(f"  Input:  {test_input}")
    print(f"  Output: {test_output}")
    
    # List available tools
    tools = await mcp.list_tools()
    print(f"\n✓ Available tools: {len(tools)}")
    for tool in tools:
        print(f"  - {tool.name}")
    
    # Example 1: List files
    print("\n" + "=" * 60)
    print("Example 1: Listing files")
    print("=" * 60)
    files = await list_files()
    print(f"Found {len(files)} files:")
    for file_info in files:
        print(f"  - {file_info['name']} ({file_info['extension']})")
    
    # Example 2: Read a text file
    print("\n" + "=" * 60)
    print("Example 2: Reading a text file")
    print("=" * 60)
    result = await read_text_file(str(test_input / "chapter1.txt"))
    print(f"Content preview: {result['content'][:100]}...")
    print(f"Total lines: {result['total_lines']}")
    
    # Example 3: Write a chapter
    print("\n" + "=" * 60)
    print("Example 3: Writing a chapter")
    print("=" * 60)
    chapter_result = await write_chapter(
        chapter_number=1,
        title="Introduction",
        content="This is the introduction chapter content."
    )
    print(f"✓ Chapter written to: {chapter_result['file_path']}")
    
    # Example 4: Write chapter list
    print("\n" + "=" * 60)
    print("Example 4: Writing chapter list")
    print("=" * 60)
    chapters = [
        {"number": 1, "title": "Introduction", "description": "Introduction to the book"},
        {"number": 2, "title": "Main Content", "description": "Main chapter content"},
        {"number": 3, "title": "Conclusion", "description": "Conclusion and summary"}
    ]
    list_result = await write_chapter_list(chapters)
    print(f"✓ Chapter list written to: {list_result['file_path']}")
    print(f"  Total chapters: {list_result['chapter_count']}")
    
    # Example 5: Generate book
    print("\n" + "=" * 60)
    print("Example 5: Generating final book")
    print("=" * 60)
    book_result = await generate_book(
        book_title="MCP Example Book",
        book_author="AI Book Composer",
        chapters=[{"title": "Introduction", "content": "This is the introduction chapter content."}],
        references=[]
    )
    print(f"✓ Book generated: {book_result.get('output_path', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
    print(f"\nOutput files are in: {test_output}")


if __name__ == "__main__":
    print("Running MCP Server Examples...\n")
    asyncio.run(example_usage())
