"""AI Book Composer - Using Deep-Agent pattern to generate books."""

from .workflow import BookComposerWorkflow
from .config import settings
from .agents import AgentState, create_initial_state

__version__ = "0.1.0"

__all__ = [
    "BookComposerWorkflow",
    "settings",
    "AgentState",
    "create_initial_state"
]
