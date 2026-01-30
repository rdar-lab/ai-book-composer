# MCP Server Documentation

## Overview

The AI Book Composer now exposes its tools through the **Model Context Protocol (MCP)**, a standardized protocol for tool exposure to LLMs. This makes it easy to:

- Discover available tools dynamically
- Add new tools without modifying the core LLM integration
- Use tools from any MCP-compatible client
- Maintain consistent tool interfaces

## Quick Start

### Running the MCP Server

```bash
# Run with default directories
python run_mcp_server.py

# Specify input and output directories
python run_mcp_server.py /path/to/input /path/to/output

# Using environment variables
INPUT_DIRECTORY=/path/to/input OUTPUT_DIRECTORY=/path/to/output python run_mcp_server.py
```

## Available Tools

The MCP server exposes 7 tools:

1. **list_files** - Lists all files in the input directory
2. **read_text_file** - Reads content from text files (txt, md, rst, docx, rtf, pdf)
3. **transcribe_audio** - Transcribes audio files (mp3, wav, m4a, flac, ogg)
4. **transcribe_video** - Transcribes video files (mp4, avi, mov, mkv)
5. **write_chapter** - Writes a chapter to a file
6. **write_chapter_list** - Writes the list of planned chapters
7. **generate_book** - Generates the final book in RTF format

## Benefits

1. **Easy Tool Discovery** - LLMs can automatically discover available tools
2. **Standardized Interface** - All tools follow the same interface pattern
3. **Type Safety** - Tools have well-defined input/output schemas
4. **Extensibility** - Add new tools without modifying the core system
5. **Cross-Platform Compatibility** - Works with any MCP-compatible client

## References

- [Model Context Protocol](https://modelcontextprotocol.io)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
