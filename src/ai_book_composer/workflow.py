"""LangGraph workflow for Deep-Agent architecture."""
import logging
from typing import Dict, Any

from langgraph.graph import StateGraph, END
from tenacity import Retrying, stop_after_attempt, wait_fixed

from .agents.critic import CriticAgent
from .agents.decorator import DecoratorAgent
from .agents.executor import ExecutorAgent
from .agents.planner import PlannerAgent
from .agents.preprocess_agent import PreprocessAgent
from .agents.state import AgentState, create_initial_state
from .agents.writer import WriterAgent
from .config import Settings
from .progress_display import progress, show_workflow_start, show_node_transition

logger = logging.getLogger(__name__)


class BookComposerWorkflow:
    """Main workflow for AI Book Composer using LangGraph."""

    def __init__(
            self,
            settings: Settings,
            input_directory: str,
            output_directory: str,
            language: str = "en-US",
            book_title: str = "Composed Book",
            book_author: str = "AI Book Composer",
            max_iterations: int = 3,
            style_instructions: str = ""
    ):
        """Initialize the workflow.
        
        Args:
            input_directory: Directory with source files
            output_directory: Directory for output
            language: Target language
            book_title: Title of the book
            book_author: Author name
            max_iterations: Maximum revision iterations
            style_instructions: Instructions to guide AI on book style
        """
        self.settings = settings
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.language = language
        self.book_title = book_title
        self.book_author = book_author
        self.max_iterations = max_iterations
        self.style_instructions = style_instructions

        # Initialize agents
        self.preprocessor = PreprocessAgent(settings, input_directory, output_directory)
        self.planner = PlannerAgent(settings)
        self.executor = ExecutorAgent(settings)
        self.decorator = DecoratorAgent(settings)
        self.critic = CriticAgent(settings)
        self.writer = WriterAgent(settings, output_directory)

        # Build graph
        self.graph = self._build_graph()

    @staticmethod
    def _record_execution(state: AgentState, node_name: str, status: str = "completed", step_index: int = None) -> \
            Dict[str, Any]:
        """Record node execution in history.
        
        Args:
            state: Current agent state
            node_name: Name of the node that was executed
            status: Execution status (default: "completed")
            step_index: Optional index of the plan step that was executed (for execute nodes)
            
        Returns:
            Dictionary with updated execution_history
        """
        execution_history = state.get("execution_history", [])
        execution_record: dict[str, Any] = {
            "node": node_name,
            "status": status
        }

        # If step_index is provided, get task details from the plan
        if step_index is not None:
            plan = state.get("plan", [])
            if 0 <= step_index < len(plan):
                task = plan[step_index]
                execution_record["task_index"] = step_index
                execution_record["task_type"] = task.get("task")
                execution_record["task_description"] = task.get("description", "")

        execution_history.append(execution_record)
        return {"execution_history": execution_history}

    # noinspection PyTypeChecker
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow.
        
        Returns:
            Compiled state graph
        """
        # Create graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("preprocess", self._preprocess_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("decorate", self._decorate_node)
        workflow.add_node("critique", self._critique_node)
        workflow.add_node("write", self._writer_node)
        workflow.add_node("finalize", self._finalize_node)

        # Set entry point
        workflow.set_entry_point("preprocess")

        # Add edges
        workflow.add_edge("preprocess", "plan")
        workflow.add_edge("plan", "execute")

        # Conditional edge from execute
        workflow.add_conditional_edges(
            "execute",
            self._should_continue_execution,
            {
                "continue": "execute",
                "decorate": "decorate"
            }
        )

        # Edge from decorate to critique
        workflow.add_edge("decorate", "critique")

        # Conditional edge from critique
        workflow.add_conditional_edges(
            "critique",
            self._should_revise,
            {
                "revise": "execute",
                "write": "write"
            }
        )

        workflow.add_edge("write", "finalize")

        # End edge
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _preprocess_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for planning phase."""
        show_node_transition(None, "preprocess", "Start Execution")
        for attempt in Retrying(stop=stop_after_attempt(3), wait=wait_fixed(60)):
            with attempt:
                logger.info("Starting preprocessing phase.")
                try:
                    result = self.preprocessor.preprocess(state)
                    # Merge execution history tracking
                    result.update(self._record_execution(state, "preprocess", "completed"))
                    return result
                except Exception as e:
                    logger.exception(f"Preprocessing attempt failed: {e}")
                    raise
        raise RuntimeError("Preprocessing failed after multiple attempts.")

    def _plan_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for planning phase."""
        show_node_transition("preprocess", "plan", "Files discovered and processed")
        for attempt in Retrying(stop=stop_after_attempt(3), wait=wait_fixed(60)):
            with attempt:
                logger.info("Starting planning phase.")
                try:
                    result = self.planner.plan(state)
                    # Merge execution history tracking
                    result.update(self._record_execution(state, "plan", "completed"))
                    return result
                except Exception as e:
                    logger.exception(f"Planning attempt failed: {e}")
                    raise
        raise RuntimeError("Planning failed after multiple attempts.")

    def _execute_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for execution phase."""
        prev_node = "plan" if state.get("current_task_index", 0) == 0 else "execute"
        show_node_transition(prev_node, "execute", "Executing next task")

        # Get the current task index before execution
        current_task_index = state.get("current_task_index", 0)

        for attempt in Retrying(stop=stop_after_attempt(3), wait=wait_fixed(60)):
            with attempt:
                logger.info("Starting execution phase.")
                try:
                    result = self.executor.execute(state)

                    # Update task as done
                    plan = state.get("plan", [])
                    if plan:
                        if current_task_index < len(plan):
                            task = plan[current_task_index]
                            logger.info(
                                f"Completed task {current_task_index}: {task.get('task')} - {task.get('description', '')}")
                            task["status"] = "completed"

                    # Record execution with the step index from the plan
                    result.update(self._record_execution(state, "execute", "completed", step_index=current_task_index))
                    return result
                except Exception as e:
                    logger.exception(f"Execution attempt failed: {e}")
                    raise
        raise RuntimeError("Execution failed after multiple attempts.")

    def _decorate_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for decorator phase."""
        show_node_transition("execute", "decorate", "Adding images to chapters")
        for attempt in Retrying(stop=stop_after_attempt(3), wait=wait_fixed(60)):
            with attempt:
                logger.info("Starting decoration phase.")
                try:
                    result = self.decorator.decorate(state)
                    # Merge execution history tracking
                    result.update(self._record_execution(state, "decorate", "completed"))
                    return result
                except Exception as e:
                    logger.exception(f"Decoration attempt failed: {e}")
                    raise
        logger.warning("Decoration failed after multiple attempts. Skipping decoration.")
        result = {"status": "decoration_failed"}
        # Still record the attempt even if it failed
        result.update(self._record_execution(state, "decorate", "failed"))
        return result

    def _critique_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for critique phase."""
        show_node_transition("decorate", "critique", "Image decoration complete, evaluating quality")
        for attempt in Retrying(stop=stop_after_attempt(3), wait=wait_fixed(60)):
            with attempt:
                logger.info("Starting critique phase.")
                try:
                    result = self.critic.critique(state)
                    # Merge execution history tracking
                    result.update(self._record_execution(state, "critique", "completed"))
                    return result
                except Exception as e:
                    logger.exception(f"Critique attempt failed: {e}")
                    raise
        raise RuntimeError("Critique failed after multiple attempts.")

    def _writer_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for critique phase."""
        show_node_transition("critique", "writer", "Quality approved")
        for attempt in Retrying(stop=stop_after_attempt(3), wait=wait_fixed(60)):
            with attempt:
                logger.info("Starting writer phase.")
                try:
                    result = self.writer.write(state)
                    # Merge execution history tracking
                    result.update(self._record_execution(state, "writer", "completed"))
                    return result
                except Exception as e:
                    logger.exception(f"Writer attempt failed: {e}")
                    raise
        raise RuntimeError("Writer failed after multiple attempts.")

    # noinspection PyMethodMayBeStatic
    def _finalize_node(self, state: AgentState) -> Dict[str, Any]:
        """Node to finalize the workflow."""
        show_node_transition("writer", "finalize", "Book generated")

        progress.show_phase(
            "Finalization",
            "Completing workflow and preparing final output"
        )

        # Show completion summary
        stats = {
            "Chapters": len(state.get("chapters", [])),
            "References": len(state.get("references", [])),
            "Iterations": state.get("iterations", 0),
            "Quality Score": f"{state.get('quality_score', 0):.2%}" if state.get('quality_score') is not None else "N/A"
        }

        progress.show_completion(state.get("final_output_path"), stats)

        return {"status": "completed"}

    # noinspection PyMethodMayBeStatic
    def _should_continue_execution(self, state: AgentState) -> str:
        """Determine if execution should continue or move to decorator.
        
        Args:
            state: Current state
            
        Returns:
            Next node name
        """
        current_task_index = state.get("current_task_index", 0)
        plan = state.get("plan", [])

        if current_task_index >= len(plan):
            return "decorate"
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
            return "write"
        else:
            # Reset the state for revision
            state['current_task_index'] = 0

            # Change settings to ignore cached content for revision
            self.settings.book.use_cached_chapters_list = False
            self.settings.book.use_cached_chapters_content = False

            # Update all plan items to pending

            # Update task as done
            plan = state.get("plan", [])
            if plan:
                for task in plan:
                    task["status"] = "pending"

            # Reset task index for revision
            return "revise"

    def run(self) -> Dict[str, Any]:
        """Run the workflow.
        
        Returns:
            Final state
        """
        # Show workflow start
        show_workflow_start(
            self.input_directory,
            self.output_directory,
            {
                "book_title": self.book_title,
                "book_author": self.book_author,
                "language": self.language,
                "style_instructions": self.style_instructions
            }
        )

        # Create initial state
        initial_state = create_initial_state(
            input_directory=self.input_directory,
            output_directory=self.output_directory,
            language=self.language,
            book_title=self.book_title,
            book_author=self.book_author,
            style_instructions=self.style_instructions
        )

        # Run the graph
        # noinspection PyUnresolvedReferences
        final_state = self.graph.invoke(initial_state)

        return final_state
