import asyncio
import json
import os
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection


def init_mcp_client(settings, input_directory: str, output_directory: str):
    # Get the path to the source directory for running as module
    src_path = Path(__file__).parent.parent # Go up to SRC root

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


def get_tools(settings, input_directory: str, output_directory: str):
    mcp_client = init_mcp_client(settings, input_directory, output_directory)
    tools = get_tools_sync(mcp_client)
    return tools
