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

    The expectation from the MCP server is to return a JSON to overcome this issue
    we expect a text element with JSON payload inside the result.
    We parse it like that, otherwise we just return it as it is
    
    This function unwraps them back to the original format.
    
    Args:
        result: Tool result from langchain-mcp-adapters
        
    Returns:
        Unwrapped result in original format
    """

    if isinstance(result, list) and len(result) > 0 and 'text' in result[0] and result[0].get('type') == 'text':
        json_payload = result[0]['text']
        try:
            return json.loads(json_payload)
        except json.JSONDecodeError:
            return json_payload  # Return as-is if not valid JSON
    elif isinstance(result, dict) and 'text' in result  and result.get('type') == 'text':
        json_payload = result['text']
        try:
            return json.loads(json_payload)
        except json.JSONDecodeError:
            return json_payload  # Return as-is if not valid JSON
    else:
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
