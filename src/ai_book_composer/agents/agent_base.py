import json
import logging
import re
from pathlib import Path
from typing import Any, Optional
from typing import Dict

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool, BaseTool

from .state import AgentState
from .. import progress_display
from ..config import load_prompts, Settings
from ..llm import get_llm, ToolFixer, system_notification

logger = logging.getLogger(__name__)

_RE_THINK_BLOCK = re.compile(r'<think>.*?</think>', re.DOTALL)
_RE_THINK_TAGS = re.compile(r'^<think>|</think>$', re.DOTALL)
_RE_JSON_BLOCK = re.compile(r'```(?:json)?\s*([\s\S]*?)\s*```')
# noinspection RegExpRedundantEscape
_RE_JSON_EXTRACT = re.compile(r'([\[\{][\s\S]*[\]\}])')
_RE_RESULT_BLOCK = re.compile(r'<result>.*?</result>', re.DOTALL)
_RE_RESULT_TAGS = re.compile(r'^<result>|</result>$', re.DOTALL)


class AgentBase:
    def __init__(self,
                 settings: Settings,
                 llm_temperature=0.0,
                 input_directory: Optional[str] = None,
                 output_directory: Optional[str] = None):
        self.settings = settings
        self.llm_temperature = llm_temperature
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.prompts = load_prompts()
        self.state = None

    def _get_llm(self):
        llm_instance = get_llm(self.settings, temperature=self.llm_temperature)
        return llm_instance

    def _generate_tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = [
            system_notification,
            self.get_file_content_tool()
        ]

        return tools

    def _invoke_llm(self, system_prompt: str, user_prompt: str):
        llm = self._get_llm()
        logger.info(
            f"Invoking llm with: \n***system prompt***\n{system_prompt}\n***user prompt***\n{user_prompt}\n")
        progress_display.progress.show_action("Running LLM...")

        # noinspection PyTypeChecker
        result = llm.invoke(f'f{system_prompt}\n{user_prompt}')

        logger.info(f"LLM Response: {result}")

        response_content = self._extract_llm_response(result)

        response_content = str(response_content)
        thought, action = self._extract_thought_and_action(response_content)

        logger.info(f"\n***LLM Thought***\n{thought}\n*** Action ***\n{action}")

        progress_display.progress.show_agent_response(thought, action)
        return action

    @staticmethod
    def _extract_llm_response(result: Any) -> Any:
        if isinstance(result, BaseMessage):
            return result.content

        if "messages" in result:
            response_content = result["messages"][-1].content
        elif 'output' in result:
            response_content = result['output']
        elif 'content' in result:
            response_content = result['content']
        else:
            response_content = result
        return response_content

    # noinspection PyUnusedLocal
    def _invoke_agent(self, system_prompt: str, user_prompt: str, state: AgentState, custom_tools: list = None) -> Any:
        if not custom_tools:
            tools = self._generate_tools()
        else:
            tools = custom_tools

        llm = ToolFixer(self._get_llm())

        # noinspection PyTypeChecker
        agent = create_agent(
            model=llm,
            tools=tools,
            debug=True
        )

        tool_names = [tool_obj.name for tool_obj in tools]

        logger.info(
            f"Invoking agent with: \n***system prompt***\n{system_prompt}\n***user prompt***\n{user_prompt}\ntools: {tool_names}")
        progress_display.progress.show_action("Running agent...")

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

        response_content = self._extract_llm_response(result)

        response_content = str(response_content)
        thought, action = self._extract_thought_and_action(response_content)

        logger.info(f"\n***Agent Thought***\n{thought}\n*** Action ***\n{action}")

        progress_display.progress.show_agent_response(thought, action)

        return action

    def get_file_content_tool(self):
        @tool
        def get_file_content(file_name: str, start_char: int = 0, length: int = 5000) -> Dict[str, Any]:
            """Get content from a file in the gathered content state.

            Args:
                file_name: Name of the file to read (e.g., 'document.txt')
                start_char: Starting character position (default: 0)
                length: Number of characters to read (default: 5000, max: 10000)

            Returns:
                Dictionary with file content, metadata, and pagination info
            """
            progress_display.progress.show_action(
                f"Fetching file content. file_name={file_name}, start_char={start_char}, length={length}")
            logger.info(f"Fetching file content. file_name={file_name}, start_char={start_char}, length={length}")
            gathered_content = self.state.get("gathered_content", {})

            # Find the file by name in gathered_content
            file_path = None
            for path in gathered_content.keys():
                if Path(path).name == file_name:
                    file_path = path
                    break

            if not file_path:
                return {
                    "error": f"File '{file_name}' not found in gathered content",
                    "available_files": [Path(p).name for p in gathered_content.keys()]
                }

            content_info = gathered_content[file_path]
            content = content_info.get("content", "")
            content_type = content_info.get("type", "unknown")

            # Limit length to avoid excessive token usage
            length = min(length, 10000)

            # Extract the requested chunk
            end_char = start_char + length
            chunk = content[start_char:end_char]

            progress_display.progress.show_observation(
                f"Fetched content chunk from '{file_name}': start_char={start_char}, end_char={end_char}")

            response = {
                "file_name": file_name,
                "file_type": content_type,
                "chunk": chunk,
                "start_char": start_char,
                "end_char": min(end_char, len(content)),
                "total_length": len(content),
                "has_more": end_char < len(content)
            }
            logger.info(
                f"Fetched content chunk from '{file_name}': start_char={start_char}, end_char={end_char}. Full response: {response}")

            return response

        return get_file_content

    @staticmethod
    def _extract_thought_and_action(result_text: str) -> tuple[str, str]:
        """
        Extracts the <think> block content and the remaining action text.
        Returns a tuple: (thought, action)
        """
        think_match = _RE_THINK_BLOCK.search(result_text)
        thought = think_match.group(0) if think_match else ""
        # Remove <think> tags if present
        if thought:
            # Extract only the inner content of <think>...</think>
            inner_thought = _RE_THINK_TAGS.sub('', thought).strip()
        else:
            inner_thought = ""
        # Remove the <think> block from the result to get the action
        action = _RE_THINK_BLOCK.sub('', result_text).strip()

        # In case there is a <result> block, extract its content as the action
        result_action_match = _RE_RESULT_BLOCK.search(action)
        result_action = result_action_match.group(0) if result_action_match else ""
        if result_action:
            action = _RE_RESULT_TAGS.sub('', result_action).strip()

        return inner_thought, action

    @staticmethod
    def _extract_json_from_llm_response(clean_text: str) -> Optional[Any]:
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

    def _get_files_summary(self):
        gathered_content = self.state.get("gathered_content", {})
        file_list = []
        for file_path, content_info in gathered_content.items():
            content = content_info.get("content", "")
            content_type = content_info.get("type", "unknown")
            summary = content_info.get("summary", "")
            file_list.append({
                "name": Path(file_path).name,
                "type": content_type,
                "size": len(content),
                "summary": summary
            })

        file_summary = "\n".join([
            f"- {f['name']} ({f['type']}, {f['size']} chars) [Summary: {f['summary']}]"
            for f in file_list
        ])
        return file_summary
