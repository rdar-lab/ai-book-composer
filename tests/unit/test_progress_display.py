"""Unit tests for progress display module."""

from unittest.mock import patch, MagicMock

from ai_book_composer.progress_display import (
    ProgressDisplay,
    show_workflow_start,
    show_node_transition
)


class TestProgressDisplay:
    """Test progress display functionality."""

    def test_initialization(self):
        """Test ProgressDisplay initialization."""
        display = ProgressDisplay()
        assert display._current_phase is None
        assert display._live is None

    @patch('ai_book_composer.progress_display.Console')
    def test_show_phase(self, mock_console_class):
        """Test showing a phase."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        display = ProgressDisplay()
        display.show_phase("Planning", "Creating a plan", "bold cyan")

        # Verify console.print was called
        assert mock_console.print.called
        assert display._current_phase == "Planning"

    @patch('ai_book_composer.progress_display.Console')
    def test_show_thought(self, mock_console_class):
        """Test showing a thought."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        display = ProgressDisplay()
        display.show_thought("Analyzing data")

        assert mock_console.print.called

    @patch('ai_book_composer.progress_display.Console')
    def test_show_action(self, mock_console_class):
        """Test showing an action."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        display = ProgressDisplay()
        display.show_action("Processing file")

        assert mock_console.print.called

    @patch('ai_book_composer.progress_display.Console')
    def test_show_observation(self, mock_console_class):
        """Test showing an observation."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        display = ProgressDisplay()
        display.show_observation("Found 5 files")

        assert mock_console.print.called

    @patch('ai_book_composer.progress_display.Console')
    def test_show_step(self, mock_console_class):
        """Test showing a step."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        display = ProgressDisplay()
        display.show_step(2, 5, "Processing data")

        assert mock_console.print.called

    @patch('ai_book_composer.progress_display.Console')
    def test_show_task_started(self, mock_console_class):
        """Test showing a task with started status."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        display = ProgressDisplay()
        display.show_task("gather_content", "started")

        assert mock_console.print.called

    @patch('ai_book_composer.progress_display.Console')
    def test_show_task_completed(self, mock_console_class):
        """Test showing a task with completed status."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        display = ProgressDisplay()
        display.show_task("gather_content", "completed")

        assert mock_console.print.called

    @patch('ai_book_composer.progress_display.Console')
    def test_show_plan(self, mock_console_class):
        """Test showing a plan."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        display = ProgressDisplay()
        plan = [
            {"task": "task1", "description": "First task", "status": "pending"},
            {"task": "task2", "description": "Second task", "status": "pending"}
        ]
        display.show_plan(plan)

        assert mock_console.print.called

    @patch('ai_book_composer.progress_display.Console')
    def test_show_files(self, mock_console_class):
        """Test showing files."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        display = ProgressDisplay()
        files = [
            {"name": "file1.txt", "extension": ".txt", "size": 1024},
            {"name": "file2.md", "extension": ".md", "size": 2048}
        ]
        display.show_files(files)

        assert mock_console.print.called

    @patch('ai_book_composer.progress_display.Console')
    def test_show_files_with_many_files(self, mock_console_class):
        """Test showing files when there are more than 10."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        display = ProgressDisplay()
        files = [
            {"name": f"file{i}.txt", "extension": ".txt", "size": 1024}
            for i in range(15)
        ]
        display.show_files(files)

        assert mock_console.print.called

    @patch('ai_book_composer.progress_display.Console')
    def test_show_chapter_info(self, mock_console_class):
        """Test showing chapter info."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        display = ProgressDisplay()
        display.show_chapter_info(1, "Introduction", "generating")

        assert mock_console.print.called

    @patch('ai_book_composer.progress_display.Console')
    def test_show_critique_summary(self, mock_console_class):
        """Test showing critique summary."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        display = ProgressDisplay()
        display.show_critique_summary(0.85, "Great work!")

        assert mock_console.print.called

    @patch('ai_book_composer.progress_display.Console')
    def test_show_critique_summary_without_score(self, mock_console_class):
        """Test showing critique summary without score."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        display = ProgressDisplay()
        display.show_critique_summary(None, "Needs improvement")

        assert mock_console.print.called

    @patch('ai_book_composer.progress_display.Console')
    def test_show_completion(self, mock_console_class):
        """Test showing completion."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        display = ProgressDisplay()
        stats = {"Chapters": 5, "References": 10}
        display.show_completion("/path/to/book.rtf", stats)

        assert mock_console.print.called

    @patch('ai_book_composer.progress_display.Console')
    def test_agent_context(self, mock_console_class):
        """Test agent context manager."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        display = ProgressDisplay()

        with display.agent_context("Planner", "Planning the book"):
            # Inside context
            pass

        # Verify context manager printed start and end messages
        assert mock_console.print.call_count >= 2


@patch('ai_book_composer.progress_display.console')
def test_show_workflow_start(mock_console):
    """Test showing workflow start."""
    config = {
        "book_title": "Test Book",
        "book_author": "Test Author",
        "language": "en-US"
    }

    show_workflow_start("/input", "/output", config)

    assert mock_console.print.called


@patch('ai_book_composer.progress_display.console')
def test_show_node_transition_with_from_node(mock_console):
    """Test showing node transition with from_node."""
    show_node_transition("plan", "execute", "Ready to execute")

    assert mock_console.print.called


@patch('ai_book_composer.progress_display.console')
def test_show_node_transition_without_from_node(mock_console):
    """Test showing node transition without from_node."""
    show_node_transition(None, "plan", "")

    assert mock_console.print.called
