"""LLM provider abstraction."""
import copy
import json
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from huggingface_hub import hf_hub_download
from langchain_community.chat_models import ChatOllama, ChatLlamaCpp
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from .config import Settings

logger = logging.getLogger(__name__)

_mapping_cache: Optional[Dict[str, Any]] = None


def load_model_mappings() -> Dict[str, Any]:
    """Load model mappings from JSON file.

    Returns:
        Dictionary of llm models
    """
    global _mapping_cache
    if _mapping_cache is None:
        if Path("models.json").exists():
            mapping_path = "models.json"
        else:
            mapping_path = Path(__file__).parent.parent.parent / "models.json"

        mapping_path = Path(mapping_path)
        if not mapping_path.exists():
            raise FileNotFoundError(f"LLM models mapping file not found: {mapping_path}")

        with open(mapping_path, 'r') as f:
            _mapping_cache = json.load(f)

    return _mapping_cache


def get_llm(
        settings: Settings,
        temperature: float = 0.7,
        model: Optional[str] = None,
        provider: Optional[str] = None,

) -> BaseChatModel:
    """Get LLM instance based on configuration.
    
    Args:
        settings: The project settings
        temperature: Temperature for generation (0.0 to 1.0)
        model: Optional model override
        provider: Optional provider override
    Returns:
        Configured LLM instance
    """
    provider = provider or settings.llm.provider
    model_name = model or settings.llm.model

    logger.info(f"Initializing LLM: provider={provider}, model={model_name}, temperature={temperature}")

    try:
        if provider == "openai":
            provider_config = settings.get_provider_config("openai")
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=provider_config.get("api_key", "")
            )

        elif provider == "gemini":
            provider_config = settings.get_provider_config("gemini")
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=provider_config.get("api_key", "")
            )

        elif provider == "azure":
            provider_config = settings.get_provider_config("azure")
            return AzureChatOpenAI(
                azure_deployment=provider_config.get("deployment", ""),
                temperature=temperature,
                api_key=provider_config.get("api_key", ""),
                azure_endpoint=provider_config.get("endpoint", "")
            )

        elif provider == "ollama":
            provider_config = settings.get_provider_config("ollama")
            # Use model from provider config if not specified
            base_url = provider_config.get("base_url", "http://localhost:11434")

            logger.info(f"Using Ollama model: {model_name} at {base_url}")
            return ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=base_url
            )

        elif provider == "ollama_embedded":
            provider_config = settings.get_provider_config("ollama_embedded")
            internal_settings = provider_config.get("internal", {})
            run_on_gpu = provider_config.get("run_on_gpu", False)

            # Convert run_on_gpu boolean to n_gpu_layers
            # If GPU enabled, use a high number to offload all layers
            # Otherwise use 0 for CPU-only
            n_gpu_layers = -1 if run_on_gpu else 0

            # Map model names to HuggingFace repo IDs and filenames
            model_mappings = load_model_mappings()

            # Get model info
            if model_name not in model_mappings:
                logger.error(f"Unknown model: {model_name}")
                raise ValueError(
                    f"Unknown embedded model: {model_name}\n"
                    f"Supported models: {', '.join(model_mappings.keys())}\n"
                    f"To add a custom model, use 'model_path' instead of 'model_name' in config."
                )

            model_info = model_mappings[model_name]
            repo_id = model_info["repo_id"]
            filename = model_info["filename"]

            # Download model from HuggingFace Hub
            logger.info(f"Downloading model {model_name} from HuggingFace...")
            logger.info(f"  Repository: {repo_id}")
            logger.info(f"  File: {filename}")

            try:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=Path.home() / ".cache" / "ai-book-composer" / "models"
                )
                logger.info(f"Model downloaded to: {model_path}")
            except Exception as e:
                logger.exception(f"Failed to download model: {e}")
                raise RuntimeError(
                    f"Failed to download model {model_name} from HuggingFace.\n"
                    f"Error: {e}\n"
                    f"Please check your internet connection and try again."
                )

            logger.info(f"Initializing embedded Ollama model: {model_name}")

            return ChatLlamaCpp(
                model_path=model_path,
                temperature=temperature,
                n_gpu_layers=n_gpu_layers,
                **internal_settings,
            )

        else:
            logger.error(f"Unsupported LLM provider: {provider}")
            raise ValueError(f"Unsupported LLM provider: {provider}")

    except Exception as e:
        logger.exception(f"Failed to initialize LLM: {e}")
        raise


class ToolFixer(Runnable[LanguageModelInput, AIMessage]):
    """
    A Runnable wrapper that fixes models outputting <tool_call> XML
    instead of native tool_calls.
    """

    def __init__(self, model):
        self.model = model

    def bind_tools(self, tools, **kwargs):
        """
        Intercepts tool binding.
        Calls the inner model's bind_tools, then wraps the result
        so the bound model ALSO has the XML fix applied.
        """
        bound_model = self.model.bind_tools(tools, **kwargs)
        return ToolFixer(bound_model)

    def invoke(self, input_messages, config: RunnableConfig = None, **kwargs):
        """
        Intercepts the execution.
        """
        # 1. Execute the real model
        logger.info(f"Invoking LLM with ToolFixer.... input={input_messages}, config={config}, kwargs={kwargs}")
        history = input_messages if isinstance(input_messages, list) else []
        msg = self.model.invoke(self._prune_history(input_messages), config=config, **kwargs)
        logger.info(f"LLM response before fix: {msg}")

        # 2. Check and Fix Output
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            # Only run regex if we see the tag (fast check)
            if "<tool_call>" in str(msg.content):
                msg = self._patch_tool_call(msg, history)

        return msg

    @staticmethod
    def _prune_history(messages):
        """
        Creates a copy of messages and aggressively compresses ToolMessages to prevent context overflow.
        
        Strategy:
        - Keep System Prompt intact
        - Keep last 4 message turns full (recent context)
        - Aggressively compress all older ToolMessages (file content responses)
        - Trim large user prompts in middle messages
        
        This prevents message history from exploding when agents repeatedly access file content.
        """
        if not isinstance(messages, list):
            return messages

        # Deep copy so we don't affect the actual agent state, only what the LLM sees
        messages_copy = copy.deepcopy(messages)

        # Keep System Prompt (idx 0) and the last 4 turns (reduced from 6 for more aggressive pruning)
        # This is enough for the LLM to maintain context while preventing overflow
        keep_last_n = 4

        if len(messages_copy) > keep_last_n:
            # Iterate through the "middle" messages (skipping first system msg and last N)
            for msg in messages_copy[1:-keep_last_n]:
                if isinstance(msg, ToolMessage):
                    # Compress tool responses to minimal metadata
                    # File content is stored in long-term memory and can be retrieved if needed
                    tool_name = msg.name if hasattr(msg, 'name') else 'unknown'
                    original_length = len(str(msg.content))
                    msg.content = (
                        f"[Compressed: Tool '{tool_name}' response ({original_length} chars) "
                        f"removed to prevent context overflow. Content stored in long-term memory.]"
                    )
                
                # Also aggressively trim old user prompts if they are large
                elif isinstance(msg, HumanMessage) and len(str(msg.content)) > 800:
                    msg.content = str(msg.content)[:400] + "... [Truncated to save context]"
        
        # Additionally compress recent ToolMessages if they're exceptionally large (>3000 chars)
        # This prevents even recent large file reads from consuming too much context
        for msg in messages_copy[-keep_last_n:]:
            if isinstance(msg, ToolMessage) and len(str(msg.content)) > 3000:
                tool_name = msg.name if hasattr(msg, 'name') else 'unknown'
                original_length = len(str(msg.content))
                # Keep a summary but compress the bulk
                summary = str(msg.content)[:500]
                msg.content = (
                    f"{summary}... [Remaining {original_length - 500} chars compressed. "
                    f"Full content in long-term memory.]"
                )

        return messages_copy

    @staticmethod
    def _extract_json_safely(text):
        """
        Extracts the first valid JSON object starting after <tool_call>.
        Handles nested braces properly, even if </tool_call> is missing.
        """
        # 1. Find start of <tool_call>
        start_tag = "<tool_call>"
        start_index = text.find(start_tag)
        if start_index == -1:
            return None

        # Move past the tag
        content_start = start_index + len(start_tag)

        # 2. Find the first '{'
        json_start = text.find("{", content_start)
        if json_start == -1:
            return None

        # 3. Stack-based extraction (Counts braces)
        brace_count = 0
        json_str = ""
        found_end = False

        for i, char in enumerate(text[json_start:], start=json_start):
            json_str += char
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1

            # If braces balance back to zero, we found the full object
            if brace_count == 0:
                found_end = True
                break

        if found_end:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _is_duplicate_call(name, args, history):
        """
        Scans history to see if this exact tool+args combo was already executed.
        """
        # We look at previous AIMessages that had tool_calls
        for message in reversed(history):
            if isinstance(message, AIMessage) and message.tool_calls:
                for past_call in message.tool_calls:
                    if past_call['name'] == name:
                        # Compare args (handled as dicts, so order doesn't matter)
                        if past_call['args'] == args:
                            return True
        return False

    @staticmethod
    def _patch_tool_call(msg, history):
        try:
            if "<tool_call>" in str(msg.content):
                data = ToolFixer._extract_json_safely(msg.content)
                if data:
                    tool_name = data.get("name")
                    tool_args = data.get("arguments", {})
                    unique_id = f"call_{str(uuid.uuid4())}"

                    if ToolFixer._is_duplicate_call(tool_name, tool_args, history):
                        logger.info(f"🔧 Skipping duplicate tool call: {data}")
                        # we SWAP the call to the system_notification tool.
                        msg.tool_calls = [{
                            "name": "system_notification",
                            "args": {
                                "message": (
                                    f"You just attempted to call '{tool_name}' with args {tool_args}, "
                                    "but this exact call was already executed in the history. "
                                    "Check previous outputs and advance to the next step."
                                )
                            },
                            "id": f"call_{unique_id}",
                            "type": "tool_call"
                        }]
                        # We keep content empty so the agent treats this purely as a tool step
                        msg.content = ""
                    else:
                        # Inject the native tool_call object
                        msg.tool_calls = [{
                            "name": tool_name,
                            "args": tool_args,
                            "id": unique_id,
                            "type": "tool_call"
                        }]
                        msg.content = f'<tool_call>{json.dumps(data)}</tool_call>'
                        logger.info(f"🔧 Patched tool call: {data.get('name')}")
        except Exception as e:
            logger.warning(f"Parsing failed: {e}")
        return msg


@tool
def system_notification(message: str) -> str:
    """
    Used by the system to provide warnings or hints to the Assistant.
    The Assistant should read this message and adjust its strategy.
    """
    return f"SYSTEM ALERT: {message}"
