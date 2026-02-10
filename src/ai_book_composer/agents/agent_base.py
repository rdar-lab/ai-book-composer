import json
import logging
import random
from pathlib import Path
from typing import Any, Optional
from typing import Dict

from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed, after_log

from .state import AgentState
from .. import progress_display
from ..config import load_prompts, Settings
from ..llm import get_llm, ThinkAndRespondFormat, invoke_agent, invoke_llm, generate_default_tools

logger = logging.getLogger(__name__)

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
            self.get_relevant_documents_tool()
        ]

        return generate_default_tools() + tools

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(60), after=after_log(logger, logging.INFO))  # type: ignore
    def _invoke_llm(self, system_prompt: str, user_prompt: str, include_agent_state: bool = True):
        progress_display.progress.show_action("Running LLM...")

        # Add agent state summary to system prompt if enabled
        if include_agent_state:
            state_summary = self._get_agent_state_summary()
            if state_summary:
                system_prompt = f"{system_prompt}\n\n## Current Agent State\n{state_summary}\n\n"

        thought, action = invoke_llm(self.settings, self._get_llm(), system_prompt, user_prompt)

        progress_display.progress.show_agent_response(thought, action)
        return action

    # noinspection PyUnusedLocal
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(60), after=after_log(logger, logging.INFO))  # type: ignore
    def _invoke_agent(self, system_prompt: str, user_prompt: str, state: AgentState = None, custom_tools: list = None,
                      include_agent_state: bool = True,
                      response_format: Optional[type[BaseModel]] = ThinkAndRespondFormat) -> Any:
        progress_display.progress.show_action("Running agent...")

        if not custom_tools:
            tools = self._generate_tools()
        else:
            tools = custom_tools

        if not state:
            state = self.state

        # Add agent state summary to system prompt if enabled
        if include_agent_state:
            state_summary = self._get_agent_state_summary()
            if state_summary:
                system_prompt = f"{system_prompt}\n\n## Current Agent State\n{state_summary}"

        thought, action = invoke_agent(self.settings, self._get_llm(), system_prompt, user_prompt, tools,
                                       response_format=response_format)

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

    def get_relevant_documents_tool(self):
        @tool
        def get_relevant_documents(query: str, num_results: int = 5) -> Dict[str, Any]:
            """Retrieve relevant document chunks based on a semantic search query.

            Args:
                query: Natural language query describing the information needed
                num_results: Number of relevant document chunks to retrieve (default: 5, max: 10)

            Returns:
                Dictionary with relevant document chunks and their metadata.
                Each result includes content, source file, and relevance information.
            """
            progress_display.progress.show_action(
                f"Searching for relevant documents. query='{query[:100]}...', num_results={num_results}")
            logger.info(f"Searching for relevant documents. query='{query}', num_results={num_results}")

            rag_manager = self.state.get("rag_manager")
            if not rag_manager:
                return {
                    "error": "RAG system not initialized. Cannot retrieve documents.",
                    "results": []
                }

            # Limit num_results to reasonable range
            num_results = min(max(1, num_results), 10)

            # Retrieve relevant documents
            documents = rag_manager.retrieve_relevant_documents(
                query=query,
                k=num_results
            )

            if not documents:
                progress_display.progress.show_observation(
                    f"No relevant documents found for query: '{query[:50]}...'")
                return {
                    "query": query,
                    "results": [],
                    "message": "No relevant documents found"
                }

            # Format results for LLM consumption
            results = []
            for i, doc in enumerate(documents, 1):
                metadata = doc.get("metadata", {})
                results.append({
                    "rank": i,
                    "content": doc.get("content", ""),
                    "file_name": metadata.get("file_name", "unknown"),
                    "file_type": metadata.get("file_type", "unknown"),
                    "similarity_score": doc.get("similarity_score", 0.0),
                    "chunk_info": f"{metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}"
                })

            progress_display.progress.show_observation(
                f"Retrieved {len(results)} relevant document(s) for query")

            logger.info(
                f"Retrieved {len(results)} relevant documents for query: '{query}'. "
                f"Top file: {results[0]['file_name'] if results else 'none'}")

            return {
                "query": query,
                "num_results": len(results),
                "results": results
            }

        return get_relevant_documents

    def _get_files_summary(self, sample_size: int = 100) -> str:
        gathered_content = self.state.get("gathered_content", {})
        file_list = []

        if len(gathered_content) > sample_size:
            # Sample randomly the gathered content to avoid overwhelming the prompt with too much information, while still providing a representative overview of the files available.
            sampled_paths = random.sample(list(gathered_content.keys()), sample_size)
            sampled_content = {path: gathered_content[path] for path in sampled_paths}
            sampled_text = f'(Sampled {sample_size} out of {len(gathered_content)} files)\n'
        else:
            sampled_content = gathered_content
            sampled_text = ''

        for file_path, content_info in sampled_content.items():
            summary = content_info.get("summary", "")
            file_list.append({
                "name": Path(file_path).name,
                "summary": summary
            })

        file_summary = f'{sampled_text}{json.dumps(file_list)}'

        # Add key terms if available
        key_terms = self.state.get("key_terms", [])
        if key_terms:
            terms_str = ", ".join(key_terms[:30])  # Show first 30 terms
            file_summary += f"\n\nKey terms across all documents: {terms_str}"

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

        # Add execution history (captures actual execution which may deviate from plan)
        execution_history = self.state.get("execution_history", [])
        if execution_history:
            summary_parts.append("\nExecution History:")
            # Show last 3 executions to keep context minimal
            recent_history = execution_history[-3:]
            for exec_record in recent_history:
                node = exec_record.get("node", exec_record.get("task_type", "Unknown"))
                exec_status = exec_record.get("status", "unknown")

                # If task details are available, show the task info
                if exec_record.get("task_type"):
                    task_type = exec_record.get("task_type")
                    summary_parts.append(f"  - {node}: {task_type} - {exec_status}")
                else:
                    summary_parts.append(f"  - {node}: {exec_status}")

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
