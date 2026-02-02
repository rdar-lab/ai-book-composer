import json
import logging
import re
from pathlib import Path
from typing import Any, Optional
from typing import Dict

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool, BaseTool
from tenacity import retry, stop_after_attempt, wait_fixed

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

# Constants for state summary
MAX_CRITIC_FEEDBACK_LENGTH = 200  # Maximum length for critic feedback in state summary


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

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(60))
    def _invoke_llm(self, system_prompt: str, user_prompt: str):
        llm = self._get_llm()
        
        # Add agent state summary to system prompt
        state_summary = self._get_agent_state_summary()
        if state_summary:
            system_prompt = f"{system_prompt}\n\n## Current Agent State\n{state_summary}"
        
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

        if '<think>' in action:
            raise Exception("Agent returned another <think> block in action, which is not allowed.")

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
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(60))
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
        
        # Add agent state summary to system prompt
        state_summary = self._get_agent_state_summary()
        if state_summary:
            system_prompt = f"{system_prompt}\n\n## Current Agent State\n{state_summary}"

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

        if '<think>' in action:
            raise Exception("Agent returned another <think> block in action, which is not allowed.")

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
                Dictionary with file content chunk, metadata, and pagination info.
                Note: Response is kept compact to minimize message history size.
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

            # Limit length to avoid excessive token usage in tool response
            length = min(length, 10000)

            # Extract the requested chunk
            end_char = start_char + length
            chunk = content[start_char:end_char]

            progress_display.progress.show_observation(
                f"Fetched content chunk from '{file_name}': start_char={start_char}, end_char={end_char}")

            # Return response with file metadata
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
                f"Fetched content chunk from '{file_name}': start_char={start_char}, "
                f"end_char={end_char}, chunk_size={len(chunk)}, total={len(content)}")

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

    def _get_agent_state_summary(self) -> str:
        """Generate a minimal summary of the agent state to include in prompts.
        
        Returns:
            Formatted state summary string
        """
        if not self.state:
            return ""
        
        summary_parts = []
        
        # Add plan steps with their status
        plan = self.state.get("plan", [])
        if plan:
            summary_parts.append("Plan Steps:")
            for i, task in enumerate(plan, 1):
                task_name = task.get("task", "Unknown")
                task_desc = task.get("description", "")
                status = task.get("status", "pending")
                current_task_index = self.state.get("current_task_index", 0)
                
                # Mark current task
                marker = " <- CURRENT" if i - 1 == current_task_index else ""
                summary_parts.append(f"  {i}. [{status.upper()}] {task_name}: {task_desc}{marker}")
        
        # Add critic feedback if present
        critic_feedback = self.state.get("critic_feedback")
        if critic_feedback:
            # Truncate long feedback to keep context minimal
            if len(critic_feedback) > MAX_CRITIC_FEEDBACK_LENGTH:
                feedback_text = f"{critic_feedback[:MAX_CRITIC_FEEDBACK_LENGTH]}..."
            else:
                feedback_text = critic_feedback
            summary_parts.append(f"\nCritic Feedback: {feedback_text}")
        
        # Add iteration count
        iterations = self.state.get("iterations", 0)
        if iterations > 0:
            summary_parts.append(f"\nIteration: {iterations}")
        
        # Add quality score if available (score is in 0.0-1.0 range)
        quality_score = self.state.get("quality_score")
        if quality_score is not None:
            summary_parts.append(f"Quality Score: {quality_score:.2%}")
        
        if not summary_parts:
            return ""
        
        return "\n".join(summary_parts)
