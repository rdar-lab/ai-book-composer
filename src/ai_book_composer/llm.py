"""LLM provider abstraction."""
import copy
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, cast

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from huggingface_hub import hf_hub_download
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.tools import tool, BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from pydantic import BaseModel, Field

from .config import Settings

logger = logging.getLogger(__name__)

# Message history pruning thresholds
_KEEP_LAST_N_TURNS = 4  # Number of recent message turns to keep uncompressed
_LARGE_TOOL_MESSAGE_THRESHOLD = 3000  # Characters - compress even recent messages above this
_LARGE_USER_MESSAGE_THRESHOLD = 800  # Characters - trim old user messages above this
_USER_MESSAGE_TRIM_LENGTH = 400  # Characters to keep when trimming user messages

_RE_THINK_BLOCK = re.compile(r'<think>.*?</think>', re.DOTALL)
_RE_THINK_TAGS = re.compile(r'^<think>|</think>$', re.DOTALL)
_RE_JSON_BLOCK = re.compile(r'```(?:json)?\s*([\s\S]*?)\s*```')
# noinspection RegExpRedundantEscape
_RE_JSON_EXTRACT = re.compile(r'([\[\{][\s\S]*[\]\}])')
_RE_RESULT_BLOCK = re.compile(r'<result>.*?</result>', re.DOTALL)
_RE_RESULT_TAGS = re.compile(r'^<result>|</result>$', re.DOTALL)

_mapping_cache: Optional[Dict[str, Any]] = None


class ThinkAndRespondFormat(BaseModel):
    think: str = Field(description="The thoughts")
    result: str = Field(description="The final result")


class ToolFixer(Runnable[LanguageModelInput, AIMessage]):
    """
    A Runnable wrapper that fixes models outputting <tool_call> XML
    instead of native tool_calls.
    """

    def __init__(self, model: BaseChatModel, bounded_model: Optional[Runnable] = None):
        self.model = model
        self.bounded_model = bounded_model

    # noinspection PyProtectedMember
    @property
    def _llm_type(self):
        return self.model._llm_type

    @property
    def profile(self):
        return self.model.profile

    def bind_tools(self, tools, **kwargs):
        """
        Intercepts tool binding.
        Calls the inner model's bind_tools, then wraps the result
        so the bound model ALSO has the XML fix applied.
        """
        bound_model = patched_bind_tools(self.model, tools, **kwargs)
        return ToolFixer(self.model, bound_model)

    def invoke(self, input_messages, config: RunnableConfig = None, **kwargs):
        """
        Intercepts the execution.
        """
        # 1. Execute the real model
        logger.info(f"Invoking LLM with ToolFixer.... input={input_messages}, config={config}, kwargs={kwargs}")
        history = input_messages if isinstance(input_messages, list) else []

        if self.bounded_model:
            model = self.bounded_model
        else:
            model = self.model

        msg = model.invoke(self._prune_history(input_messages), config=config, **kwargs)
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

        # Keep System Prompt (idx 0) and the last N turns
        # This is enough for the LLM to maintain context while preventing overflow
        keep_last_n = _KEEP_LAST_N_TURNS

        if len(messages_copy) > keep_last_n:
            # Iterate through the "middle" messages (skipping first system msg and last N)
            for msg in messages_copy[1:-keep_last_n]:
                if isinstance(msg, ToolMessage):
                    # Keep a brief preview of the tool response content instead of removing completely
                    # This helps the LLM understand what was retrieved previously
                    tool_name = msg.name if hasattr(msg, 'name') else 'unknown'
                    original_content = str(msg.content)
                    original_length = len(original_content)

                    # Keep first 200 chars as preview + metadata (only if content is longer)
                    if original_length > 200:
                        preview = original_content[:200]
                        msg.content = (
                            f"{preview}... [Content truncated. Original length: {original_length} chars. "
                            f"Tool: {tool_name}]"
                        )
                    # If content is 200 chars or less, keep as-is

                # Don't compress HumanMessages - they contain important user prompts
                # that are crucial for generation

        # For recent ToolMessages, keep more content but still compress if very large
        # This prevents even recent large file reads from consuming too much context
        for msg in messages_copy[-keep_last_n:]:
            if isinstance(msg, ToolMessage) and len(str(msg.content)) > _LARGE_TOOL_MESSAGE_THRESHOLD:
                tool_name = msg.name if hasattr(msg, 'name') else 'unknown'
                original_content = str(msg.content)
                original_length = len(original_content)
                # Keep first 1000 chars for recent messages (more than old ones)
                preview = original_content[:1000]
                msg.content = (
                    f"{preview}... [Content truncated to save context. "
                    f"Original length: {original_length} chars. Tool: {tool_name}]"
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


def patched_bind_tools(model, tools, tool_choice=None, **kwargs):
    # This bypasses the strict check that causes the ValueError
    # while still passing the tools to the underlying llama-cpp-python logic
    if type(model).__name__ == 'ChatLlamaCpp':
        from langchain_core.utils.function_calling import convert_to_openai_tool
        bind_tools = [convert_to_openai_tool(tool_obj) for tool_obj in tools]
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        return model.bind(tools=bind_tools, **kwargs)
    else:
        # For other models, just call the normal bind_tools
        return model.bind_tools(tools=tools, **kwargs)


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


def _init_ollama_cpp(**args):
    from langchain_community.chat_models import ChatLlamaCpp
    return ChatLlamaCpp(
        **args,
    )


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

            from langchain_community.chat_models import ChatLlamaCpp
            return _init_ollama_cpp(
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


def _extract_llm_response(result: Any) -> Any:
    # If it is STR return it
    if isinstance(result, str):
        return result

    # If it is base message - get the content out of it and parse again
    if isinstance(result, BaseMessage):
        return _extract_llm_response(result.content)

    if isinstance(result, dict):
        # If it contains a structured_response, return it
        if 'structured_response' in result:
            return result['structured_response']

        # If it is a list of messages - grab the last one and parse again
        if 'messages' in result:
            return _extract_llm_response(result["messages"][-1])

        if 'output' in result:
            return result['output']

        if 'content' in result:
            return result['output']

    return result


def _extract_thought_and_action(llm_response: Any) -> tuple[str, str]:
    """
    Extracts the <think> block content and the remaining action text.
    Returns a tuple: (thought, action)
    """

    if isinstance(llm_response, ThinkAndRespondFormat):
        final_thought = llm_response.think
        action = llm_response.result
    else:
        result_text = str(llm_response)

        think_match = _RE_THINK_BLOCK.search(result_text)
        thought = think_match.group(0) if think_match else ""
        # Remove <think> tags if present
        if thought:
            # Extract only the inner content of <think>...</think>
            final_thought = _RE_THINK_TAGS.sub('', thought).strip()
        else:
            final_thought = ""
        # Remove the <think> block from the result to get the action
        action = _RE_THINK_BLOCK.sub('', result_text).strip()

        # In case there is a <result> block, extract its content as the action
        result_action_match = _RE_RESULT_BLOCK.search(action)
        result_action = result_action_match.group(0) if result_action_match else ""
        if result_action:
            action = _RE_RESULT_TAGS.sub('', result_action).strip()

    if '<think>' in action:
        raise Exception("Agent returned another <think> block in action, which is not allowed.")

    return final_thought, action


def extract_json_from_llm_response(clean_text: str) -> Optional[Any]:
    """
    Generic utility to extract JSON from LLM responses.
    Handles mark-down code blocks, and leading/trailing text.
    """
    markdown_match = _RE_JSON_BLOCK.search(clean_text)
    if markdown_match:
        target_text = markdown_match.group(1)
    else:
        json_match = _RE_JSON_EXTRACT.search(clean_text)
        target_text = json_match.group(1) if json_match else clean_text

    if not target_text:
        raise Exception("Could not find JSON object in the LLM response.")

    try:
        return json.loads(target_text.strip())
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse extracted JSON: {e}")
        raise


def invoke_agent(settings: Settings, model, system_prompt: str, user_prompt: str, tools=None,
                 response_format: Optional[type[BaseModel]] = ThinkAndRespondFormat):
    try:
        if settings.llm.use_tool_fixer:
            inner_model = cast(BaseChatModel, cast(Runnable, ToolFixer(model)))
        else:
            inner_model = model

        if not tools:
            tools = generate_default_tools()

        tool_names = [tool_obj.name for tool_obj in tools] if tools else []

        if settings.llm.use_deep_agent:
            agent = create_deep_agent(
                model=inner_model,
                tools=tools,
                debug=settings.llm.agent_debug_mode,
                backend=lambda tool_runtime: StateBackend(tool_runtime),
                response_format=response_format
            )
        else:
            agent = create_agent(
                model=inner_model,
                tools=tools,
                debug=settings.llm.agent_debug_mode
            )

        logger.info(
            f"Invoking agent with: \n***system prompt***\n{system_prompt}\n***user prompt***\n{user_prompt}\ntools: {tool_names}")

        # noinspection PyTypeChecker
        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
            }
        )

        logger.info(f"Agent response: {result}")

        response_content = _extract_llm_response(result)
        thought, action = _extract_thought_and_action(response_content)

        logger.info(f"\n***Agent Thought***\n{thought}\n*** Action ***\n{action}")

        return thought, action
    except Exception as exp:
        logger.exception(f"Agent invocation failed: {exp}")
        raise


# noinspection PyUnusedLocal
def invoke_llm(settings: Settings, model, system_prompt: str, user_prompt: str):
    try:
        logger.info(
            f"Invoking llm with: \n***system prompt***\n{system_prompt}\n***user prompt***\n{user_prompt}\n")

        # noinspection PyTypeChecker
        result = model.invoke(f'{system_prompt}\n{user_prompt}')

        logger.info(f"LLM Response: {result}")

        response_content = _extract_llm_response(result)
        thought, action = _extract_thought_and_action(response_content)

        logger.info(f"\n***LLM Thought***\n{thought}\n*** Action ***\n{action}")

        return thought, action
    except Exception as exp:
        logger.exception(f"LLM invocation failed: {exp}")
        raise


@tool
def system_notification(message: str) -> str:
    """
    Used by the system to provide warnings or hints to the Assistant.
    The Assistant should read this message and adjust its strategy.
    """
    return f"SYSTEM ALERT: {message}"


def generate_default_tools() -> list[BaseTool]:
    return [system_notification]
