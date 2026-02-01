"""Unit tests for BookComposerWorkflow."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.ai_book_composer.workflow import BookComposerWorkflow
from src.ai_book_composer.agents.state import create_initial_state
from src.ai_book_composer.config import Settings


class TestWorkflowInitialization:
    """Test workflow initialization."""

    def test_init_creates_workflow(self, tmp_path):
        """Test that workflow can be initialized."""
        settings = Settings()
        input_dir = str(tmp_path / "input")
        output_dir = str(tmp_path / "output")
        
        workflow = BookComposerWorkflow(
            settings=settings,
            input_directory=input_dir,
            output_directory=output_dir
        )
        
        assert workflow.settings == settings
        assert workflow.input_directory == input_dir
        assert workflow.output_directory == output_dir
        assert workflow.language == "en-US"
        assert workflow.book_title == "Composed Book"
        assert workflow.book_author == "AI Book Composer"
        assert workflow.max_iterations == 3

    def test_init_with_custom_parameters(self, tmp_path):
        """Test initialization with custom parameters."""
        settings = Settings()
        
        workflow = BookComposerWorkflow(
            settings=settings,
            input_directory=str(tmp_path),
            output_directory=str(tmp_path),
            language="es-ES",
            book_title="Mi Libro",
            book_author="Autor",
            max_iterations=5,
            style_instructions="Academic style"
        )
        
        assert workflow.language == "es-ES"
        assert workflow.book_title == "Mi Libro"
        assert workflow.book_author == "Autor"
        assert workflow.max_iterations == 5
        assert workflow.style_instructions == "Academic style"

    def test_init_creates_agents(self, tmp_path):
        """Test that all agents are created during initialization."""
        settings = Settings()
        
        workflow = BookComposerWorkflow(
            settings=settings,
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        
        assert workflow.preprocessor is not None
        assert workflow.planner is not None
        assert workflow.executor is not None
        assert workflow.decorator is not None
        assert workflow.critic is not None

    def test_init_builds_graph(self, tmp_path):
        """Test that graph is built during initialization."""
        settings = Settings()
        
        workflow = BookComposerWorkflow(
            settings=settings,
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        
        assert workflow.graph is not None


class TestWorkflowNodeMethods:
    """Test individual workflow node methods."""

    @patch.object(BookComposerWorkflow, '__init__', lambda x, **kwargs: None)
    def test_preprocess_node_success(self, tmp_path):
        """Test preprocess node execution."""
        workflow = BookComposerWorkflow()
        workflow.preprocessor = Mock()
        workflow.preprocessor.preprocess = Mock(return_value={
            "status": "preprocessed",
            "files": [],
            "gathered_content": {},
            "images": []
        })
        
        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        
        result = workflow._preprocess_node(state)
        
        assert result["status"] == "preprocessed"
        workflow.preprocessor.preprocess.assert_called_once()

    @patch.object(BookComposerWorkflow, '__init__', lambda x, **kwargs: None)
    def test_plan_node_success(self, tmp_path):
        """Test plan node execution."""
        workflow = BookComposerWorkflow()
        workflow.planner = Mock()
        workflow.planner.plan = Mock(return_value={
            "status": "planned",
            "plan": [{"task": "Generate chapter 1"}]
        })
        
        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        
        result = workflow._plan_node(state)
        
        assert result["status"] == "planned"
        workflow.planner.plan.assert_called_once()

    @patch.object(BookComposerWorkflow, '__init__', lambda x, **kwargs: None)
    def test_execute_node_success(self, tmp_path):
        """Test execute node execution."""
        workflow = BookComposerWorkflow()
        workflow.executor = Mock()
        workflow.executor.execute = Mock(return_value={
            "status": "executing",
            "current_task_index": 1
        })
        
        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        
        result = workflow._execute_node(state)
        
        assert result["status"] == "executing"
        workflow.executor.execute.assert_called_once()

    @patch.object(BookComposerWorkflow, '__init__', lambda x, **kwargs: None)
    def test_decorate_node_success(self, tmp_path):
        """Test decorate node execution."""
        workflow = BookComposerWorkflow()
        workflow.decorator = Mock()
        workflow.decorator.decorate = Mock(return_value={
            "status": "decorated",
            "chapters": []
        })
        
        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        
        result = workflow._decorate_node(state)
        
        assert result["status"] == "decorated"
        workflow.decorator.decorate.assert_called_once()

    @patch.object(BookComposerWorkflow, '__init__', lambda x, **kwargs: None)
    def test_critique_node_success(self, tmp_path):
        """Test critique node execution."""
        workflow = BookComposerWorkflow()
        workflow.critic = Mock()
        workflow.critic.critique = Mock(return_value={
            "status": "approved",
            "quality_score": 0.95
        })
        
        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        
        result = workflow._critique_node(state)
        
        assert result["status"] == "approved"
        workflow.critic.critique.assert_called_once()

    @patch.object(BookComposerWorkflow, '__init__', lambda x, **kwargs: None)
    def test_critique_node_success(self, tmp_path):
        """Test critique node execution."""
        workflow = BookComposerWorkflow()
        workflow.critic = Mock()
        workflow.critic.critique = Mock(return_value={
            "status": "approved",
            "quality_score": 0.95
        })
        
        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        
        result = workflow._critique_node(state)
        
        assert result["status"] == "approved"
        workflow.critic.critique.assert_called_once()

    @patch.object(BookComposerWorkflow, '__init__', lambda x, **kwargs: None)
    def test_finalize_node(self, tmp_path):
        """Test finalize node execution."""
        workflow = BookComposerWorkflow()
        
        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        state["chapters"] = [{"title": "Ch1"}, {"title": "Ch2"}]
        state["references"] = ["Ref1"]
        state["iterations"] = 2
        state["quality_score"] = 0.92
        
        result = workflow._finalize_node(state)
        
        assert result["status"] == "completed"


class TestWorkflowConditionalLogic:
    """Test workflow conditional edge logic."""

    @patch.object(BookComposerWorkflow, '__init__', lambda x, **kwargs: None)
    def test_should_continue_execution_when_tasks_remain(self, tmp_path):
        """Test execution continues when tasks remain."""
        workflow = BookComposerWorkflow()
        
        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        state["status"] = "executing"
        state["current_task_index"] = 2
        state["plan"] = [
            {"task": "Task 1"},
            {"task": "Task 2"},
            {"task": "Task 3"}
        ]
        
        result = workflow._should_continue_execution(state)
        
        assert result == "continue"

    @patch.object(BookComposerWorkflow, '__init__', lambda x, **kwargs: None)
    def test_should_continue_execution_when_completed(self, tmp_path):
        """Test execution moves to decorate when completed."""
        workflow = BookComposerWorkflow()
        
        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        state["status"] = "book_generated"
        
        result = workflow._should_continue_execution(state)
        
        assert result == "decorate"

    @patch.object(BookComposerWorkflow, '__init__', lambda x, **kwargs: None)
    def test_should_continue_execution_when_plan_finished(self, tmp_path):
        """Test execution moves to decorate when plan is finished."""
        workflow = BookComposerWorkflow()
        
        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        state["current_task_index"] = 3
        state["plan"] = [
            {"task": "Task 1"},
            {"task": "Task 2"},
            {"task": "Task 3"}
        ]
        
        result = workflow._should_continue_execution(state)
        
        assert result == "decorate"

    @patch.object(BookComposerWorkflow, '__init__', lambda x, **kwargs: None)
    def test_should_revise_when_approved(self, tmp_path):
        """Test workflow finalizes when approved."""
        workflow = BookComposerWorkflow()
        workflow.max_iterations = 3
        workflow.settings = Settings()
        
        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        state["status"] = "approved"
        state["iterations"] = 1
        
        result = workflow._should_revise(state)
        
        assert result == "finalize"

    @patch.object(BookComposerWorkflow, '__init__', lambda x, **kwargs: None)
    def test_should_revise_when_max_iterations_reached(self, tmp_path):
        """Test workflow finalizes when max iterations reached."""
        workflow = BookComposerWorkflow()
        workflow.max_iterations = 3
        workflow.settings = Settings()
        
        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        state["status"] = "needs_revision"
        state["iterations"] = 3
        
        result = workflow._should_revise(state)
        
        assert result == "finalize"

    @patch.object(BookComposerWorkflow, '__init__', lambda x, **kwargs: None)
    def test_should_revise_when_needs_improvement(self, tmp_path):
        """Test workflow revises when improvement needed."""
        workflow = BookComposerWorkflow()
        workflow.max_iterations = 3
        workflow.settings = Settings()
        
        state = create_initial_state(
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        state["status"] = "needs_revision"
        state["iterations"] = 1
        state["current_task_index"] = 5
        
        result = workflow._should_revise(state)
        
        assert result == "revise"
        # Should reset task index for revision
        assert state["current_task_index"] == 0
        # Should disable caching for revision
        assert workflow.settings.book.use_cached_chapters_list is False
        assert workflow.settings.book.use_cached_chapters_content is False


    @patch.object(BookComposerWorkflow, '_build_graph')
    def test_run_creates_initial_state(self, mock_build_graph, tmp_path):
        """Test that run creates proper initial state."""
        settings = Settings()
        mock_graph = Mock()
        mock_graph.invoke = Mock(return_value={
            "status": "completed",
            "chapters": [],
            "references": []
        })
        mock_build_graph.return_value = mock_graph
        
        workflow = BookComposerWorkflow(
            settings=settings,
            input_directory=str(tmp_path / "input"),
            output_directory=str(tmp_path / "output"),
            language="fr-FR",
            book_title="Mon Livre",
            book_author="Auteur",
            style_instructions="Style formel"
        )
        
        workflow.run()
        
        # Verify graph was invoked
        mock_graph.invoke.assert_called_once()
        
        # Check initial state passed to graph
        call_args = mock_graph.invoke.call_args[0][0]
        assert call_args["language"] == "fr-FR"
        assert call_args["book_title"] == "Mon Livre"
        assert call_args["book_author"] == "Auteur"
        assert call_args["style_instructions"] == "Style formel"

    @patch.object(BookComposerWorkflow, '_build_graph')
    def test_run_returns_final_state(self, mock_build_graph, tmp_path):
        """Test that run returns final state from graph."""
        settings = Settings()
        expected_final_state = {
            "status": "completed",
            "chapters": [{"title": "Ch1"}],
            "references": ["Ref1"],
            "iterations": 2,
            "quality_score": 0.95
        }
        
        mock_graph = Mock()
        mock_graph.invoke = Mock(return_value=expected_final_state)
        mock_build_graph.return_value = mock_graph
        
        workflow = BookComposerWorkflow(
            settings=settings,
            input_directory=str(tmp_path),
            output_directory=str(tmp_path)
        )
        
        result = workflow.run()
        
        assert result == expected_final_state
