"""LangGraph workflow for Deep-Agent architecture."""

from typing import Dict, Any
from langgraph.graph import StateGraph, END

from .agents import (
    AgentState,
    create_initial_state,
    PlannerAgent,
    ExecutorAgent,
    CriticAgent
)
from .tools import FileListingTool


class BookComposerWorkflow:
    """Main workflow for AI Book Composer using LangGraph."""
    
    def __init__(
        self,
        input_directory: str,
        output_directory: str,
        language: str = "en-US",
        book_title: str = "Composed Book",
        book_author: str = "AI Book Composer",
        max_iterations: int = 3
    ):
        """Initialize the workflow.
        
        Args:
            input_directory: Directory with source files
            output_directory: Directory for output
            language: Target language
            book_title: Title of the book
            book_author: Author name
            max_iterations: Maximum revision iterations
        """
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.language = language
        self.book_title = book_title
        self.book_author = book_author
        self.max_iterations = max_iterations
        
        # Initialize agents
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent(input_directory, output_directory)
        self.critic = CriticAgent()
        self.file_lister = FileListingTool(input_directory)
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow.
        
        Returns:
            Compiled state graph
        """
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("list_files", self._list_files_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("critique", self._critique_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Set entry point
        workflow.set_entry_point("list_files")
        
        # Add edges
        workflow.add_edge("list_files", "plan")
        workflow.add_edge("plan", "execute")
        
        # Conditional edge from execute
        workflow.add_conditional_edges(
            "execute",
            self._should_continue_execution,
            {
                "continue": "execute",
                "critique": "critique"
            }
        )
        
        # Conditional edge from critique
        workflow.add_conditional_edges(
            "critique",
            self._should_revise,
            {
                "revise": "execute",
                "finalize": "finalize"
            }
        )
        
        # End edge
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _list_files_node(self, state: AgentState) -> Dict[str, Any]:
        """Node to list all files in input directory."""
        files = self.file_lister.run()
        return {"files": files, "status": "files_listed"}
    
    def _plan_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for planning phase."""
        return self.planner.plan(state)
    
    def _execute_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for execution phase."""
        return self.executor.execute(state)
    
    def _critique_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for critique phase."""
        return self.critic.critique(state)
    
    def _finalize_node(self, state: AgentState) -> Dict[str, Any]:
        """Node to finalize the workflow."""
        return {"status": "completed"}
    
    def _should_continue_execution(self, state: AgentState) -> str:
        """Determine if execution should continue or move to critique.
        
        Args:
            state: Current state
            
        Returns:
            Next node name
        """
        status = state.get("status", "")
        current_task_index = state.get("current_task_index", 0)
        plan = state.get("plan", [])
        
        if status == "book_generated" or current_task_index >= len(plan):
            return "critique"
        else:
            return "continue"
    
    def _should_revise(self, state: AgentState) -> str:
        """Determine if book should be revised or finalized.
        
        Args:
            state: Current state
            
        Returns:
            Next node name
        """
        status = state.get("status", "")
        iterations = state.get("iterations", 0)
        
        if status == "approved" or iterations >= self.max_iterations:
            return "finalize"
        else:
            # Reset task index for revision
            return "revise"
    
    def run(self) -> Dict[str, Any]:
        """Run the workflow.
        
        Returns:
            Final state
        """
        # Create initial state
        initial_state = create_initial_state(
            input_directory=self.input_directory,
            output_directory=self.output_directory,
            language=self.language,
            book_title=self.book_title,
            book_author=self.book_author
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return final_state
