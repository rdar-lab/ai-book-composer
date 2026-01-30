#!/usr/bin/env python3
"""Wrapper script to run the MCP server as a subprocess.

This script is used by the MultiServerMCPClient to start the MCP server
in stdio mode.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# Now import and run the MCP server
from ai_book_composer import mcp_server

if __name__ == "__main__":
    # Run the MCP server's main function
    sys.argv.append("--stdio")
    exec(open(mcp_server.__file__).read())
