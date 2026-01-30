import asyncio
import json
import os
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection


def init_mcp_client(settings, input_directory: str, output_directory: str):
    # Get the path to the source directory for running as module
    src_path = Path(__file__).parent.parent  # Go up to SRC root

    # Get a location for a temporary file for the configuration path
    temp_location = str(Path.cwd() / ".temp_mcp_config.yml")
    settings.save_config(temp_location)

    # Configure MCP server connection via stdio
    # The server will be launched as a subprocess and communicate via stdio
    mcp_connections = {
        "ai_book_composer_tools": StdioConnection(
            cwd=str(src_path.resolve()),
            command="python",
            args=["-m", "ai_book_composer.mcp_server", "--stdio"],
            env={
                "SETTINGS_PATH": temp_location,
                "INPUT_DIRECTORY": str(Path(input_directory).resolve()),
                "OUTPUT_DIRECTORY": str(Path(output_directory).resolve()),
                "PYTHONPATH": str(src_path.resolve()),
                **os.environ  # Include existing environment variables
            },
            transport="stdio"
        )
    }

    # Initialize MCP client to connect to the tool server
    return MultiServerMCPClient(connections=mcp_connections)


def get_tools_sync(mcp_client):
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

    return loop.run_until_complete(mcp_client.get_tools())


def unwrap_mcp_result(result: Any) -> Any:
    """Unwrap MCP tool result from LangChain format.
    
    LangChain MCP adapters wrap tool results in formats like:
    - List: [{'id': '...', 'text': '...', 'type': 'text'}, ...]
    - Single dict: {'id': '...', 'text': '...', 'type': 'text'}
    
    This function unwraps them back to the original format.
    
    Args:
        result: Tool result from langchain-mcp-adapters
        
    Returns:
        Unwrapped result in original format
    """
    # Handle single wrapped item (dict)
    if isinstance(result, dict) and 'text' in result and 'type' in result and 'id' in result:
        # Only unwrap if type is 'text' (standard MCP wrapper format)
        if result.get('type') == 'text':
            try:
                return json.loads(result['text'])
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, return the text as-is
                return result['text']
        # For other types, return as-is (this might be a regular dict, not a wrapper)
        # since we can't be certain it's a wrapped result without type='text'
        return result

    # Handle list of wrapped items
    if isinstance(result, list) and len(result) > 0:
        # Check if all items look like wrapped results
        all_wrapped = all(
            isinstance(item, dict) and
            'text' in item and
            'type' in item and
            'id' in item and
            item.get('type') == 'text'
            for item in result
        )

        if all_wrapped:
            # All items are wrapped - unwrap each one
            unwrapped = []
            for item in result:
                try:
                    unwrapped.append(json.loads(item['text']))
                except (json.JSONDecodeError, TypeError):
                    # If parsing fails, keep the text as-is
                    unwrapped.append(item['text'])
            return unwrapped

    # If it's not wrapped, return as-is
    return result


def get_tools(settings, input_directory: str, output_directory: str):
    """Get MCP tools with automatic result unwrapping.
    
    Returns tools from the MCP server with their ainvoke methods wrapped
    to automatically unwrap LangChain's MCP result format.
    
    Args:
        settings: Application settings
        input_directory: Input directory path
        output_directory: Output directory path

    Returns:
        List of tools with unwrapping applied
    """
    mcp_client = init_mcp_client(settings, input_directory, output_directory)
    tools = get_tools_sync(mcp_client)

    return tools


def invoke_tool(tool, **kwargs) -> Any:
    """Invoke a tool by name with arguments.

    Args:
        tool: The tool to invoke
        **kwargs: Tool arguments

    Returns:
        Tool result
    """

    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, we need to use a different approach
            import nest_asyncio
            nest_asyncio.apply()
            result = loop.run_until_complete(tool.ainvoke(kwargs))
        else:
            raise RuntimeError("Event loop not running")
    except RuntimeError:
        result = asyncio.run(tool.ainvoke(kwargs))

    return unwrap_mcp_result(result)
