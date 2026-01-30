"""AI Book Composer - Using Deep-Agent pattern to generate books."""

__version__ = "0.1.0"

# Lazy imports to avoid requiring all dependencies at import time
def __getattr__(name):
    """Lazy import of main components."""
    if name == "BookComposerWorkflow":
        from .workflow import BookComposerWorkflow
        return BookComposerWorkflow
    elif name == "settings":
        from .config import settings
        return settings
    elif name == "AgentState":
        from .agents import AgentState
        return AgentState
    elif name == "create_initial_state":
        from .agents import create_initial_state
        return create_initial_state
    elif name == "ToolRegistry":
        from .langchain_tools import ToolRegistry
        return ToolRegistry
    elif name == "mcp_server":
        from .mcp_server import mcp, initialize_tools
        return type('MCPServer', (), {'mcp': mcp, 'initialize_tools': initialize_tools})
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BookComposerWorkflow",
    "settings",
    "AgentState",
    "create_initial_state",
    "ToolRegistry",
    "mcp_server"
]
