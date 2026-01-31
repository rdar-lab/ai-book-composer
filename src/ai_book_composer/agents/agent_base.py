from typing import Any, Optional

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from .state import AgentState
from .. import mcp_client
from ..config import load_prompts, Settings
from ..llm import get_llm


class AgentBase:
    def __init__(self, settings: Settings,
                 llm_temperature=0.0,
                 cache_llm: bool = True,
                 bind_tools: bool = False,
                 input_directory: Optional[str] = None,
                 output_directory: Optional[str] = None):
        self.settings = settings
        self.llm_temperature = llm_temperature
        self.is_cache_llm = cache_llm
        self.bind_tools = bind_tools
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.tools_map: Optional[dict[str, Any]] = None
        self._llm_instance_cache = None
        self.prompts = load_prompts()

    @property
    def llm(self):
        if self._llm_instance_cache is not None:
            return self._llm_instance_cache

        llm_instance = get_llm(self.settings, temperature=self.llm_temperature)

        if self.bind_tools:
            self._init_tools_map()
            llm_instance = llm_instance.bind_tools(list(self.tools_map.values()))

        if self.is_cache_llm:
            self._llm_instance_cache = llm_instance

        return llm_instance

    def _init_tools_map(self):
        if self.tools_map is None:
            tools = self._generate_tools()
            self.tools_map = {tool.name: tool for tool in tools}

    def _generate_tools(self) -> Any:
        if not self.bind_tools:
            return []

        tools = mcp_client.get_tools(self.settings, self.input_directory, self.output_directory)
        return tools

    def _invoke_agent(self, system_prompt: str, user_prompt: str, state: AgentState) -> Any:
        if not self.bind_tools:
            raise Exception("Agent tools are not bound. Cannot invoke agent.")

        self._init_tools_map()

        # 1. Attach state to self so the @tools can grab it
        self.current_run_state = state

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),  # Essential for agent internal tracking
        ])

        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=list(self.tools_map.values()),
            prompt=prompt
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=list(self.tools_map.values()),
            verbose=True
        )

        result = agent_executor.invoke({"input": user_prompt})
        return result

    def _invoke_tool(self, tool_name: str, **kwargs) -> Any:
        """Invoke a tool by name with arguments.

        Args:
            tool_name: Name of the tool to invoke
            **kwargs: Tool arguments

        Returns:
            Tool result
        """
        if not self.bind_tools:
            raise Exception("Agent tools are not bound. Cannot invoke tool.")

        # Just in case, we ensure tools map is initialized
        self._init_tools_map()

        if tool_name not in self.tools_map:
            raise ValueError(f"Tool {tool_name} not found")

        tool = self.tools_map[tool_name]

        return mcp_client.invoke_tool(tool, **kwargs)
