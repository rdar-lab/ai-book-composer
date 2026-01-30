"""Unit tests for agents with mocked LLMs."""

import tempfile
from unittest.mock import Mock, patch

from ai_book_composer.agents.critic import CriticAgent
from ai_book_composer.agents.executor import ExecutorAgent
from ai_book_composer.agents.planner import PlannerAgent
from ai_book_composer.agents.state import AgentState, create_initial_state
from ai_book_composer.config import Settings

class TestAgentState:
    """Test agent state management."""

    def test_create_initial_state(self):
        """Test creating initial state."""
        state = create_initial_state(
            input_directory="/tmp/input",
            output_directory="/tmp/output",
            language="en-US",
            book_title="Test Book",
            book_author="Test Author"
        )

        assert state["input_directory"] == "/tmp/input"
        assert state["output_directory"] == "/tmp/output"
        assert state["language"] == "en-US"
        assert state["book_title"] == "Test Book"
        assert state["book_author"] == "Test Author"
        assert state["status"] == "initialized"
        assert state["iterations"] == 0
        assert isinstance(state["files"], list)
        assert isinstance(state["chapters"], list)

    def test_state_defaults(self):
        """Test state with default values."""
        state = create_initial_state(
            input_directory="/tmp/input",
            output_directory="/tmp/output"
        )

        assert state["language"] == "en-US"
        assert state["book_title"] == "Composed Book"
        assert state["book_author"] == "AI Book Composer"


class TestPlannerAgent:
    """Test planner agent with mocked LLM."""

    @patch('ai_book_composer.agents.planner.get_llm')
    def test_plan_generation(self, mock_get_llm):
        """Test plan generation."""
        # Mock LLM response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Generated plan content"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        planner = PlannerAgent(Settings())

        state = create_initial_state(
            input_directory="/tmp/input",
            output_directory="/tmp/output"
        )
        state["files"] = [
            {"name": "file1.txt", "path": "/tmp/input/file1.txt", "extension": ".txt"},
            {"name": "file2.txt", "path": "/tmp/input/file2.txt", "extension": ".txt"}
        ]

        result = planner.plan(state)

        assert "plan" in result
        assert "status" in result
        assert result["status"] == "planned"
        assert isinstance(result["plan"], list)
        assert mock_llm.invoke.called


class TestCriticAgent:
    """Test critic agent with mocked LLM."""

    @patch('ai_book_composer.agents.critic.get_llm')
    def test_critique_good_quality(self, mock_get_llm):
        """Test critique with good quality score."""
        # Mock LLM response indicating good quality
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Quality score: 0.9\nDecision: approve\nThe book is excellent."
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        critic = CriticAgent(Settings(), quality_threshold=0.7)

        state = create_initial_state(
            input_directory="/tmp/input",
            output_directory="/tmp/output"
        )
        state["chapters"] = [
            {"number": 1, "title": "Chapter 1", "content": "Content 1"},
            {"number": 2, "title": "Chapter 2", "content": "Content 2"}
        ]
        state["references"] = ["Ref 1", "Ref 2"]

        result = critic.critique(state)

        assert "critic_feedback" in result
        assert "quality_score" in result
        assert "status" in result
        # Status should be approved due to high score
        assert result["status"] in ["approved", "needs_revision"]
        assert mock_llm.invoke.called

    @patch('ai_book_composer.agents.critic.get_llm')
    def test_critique_no_chapters(self, mock_get_llm):
        """Test critique with no chapters."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm

        critic = CriticAgent(Settings())

        state = create_initial_state(
            input_directory="/tmp/input",
            output_directory="/tmp/output"
        )
        state["chapters"] = []

        result = critic.critique(state)

        assert result["quality_score"] == 0.0
        assert result["status"] == "needs_revision"
        assert not mock_llm.invoke.called  # Should not call LLM if no chapters


class TestExecutorAgent:
    """Test executor agent with focus on content summarization."""

    @patch('ai_book_composer.agents.executor.get_llm')
    @patch('ai_book_composer.mcp_client.get_tools')
    def test_summarize_content_includes_full_content(self, get_tools_mock, mock_llm):
        """Test that _summarize_content includes full file content for chapter planning."""
        get_tools_mock.return_value = []
        mock_llm.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ExecutorAgent(
                Settings(),
                input_directory=tmpdir,
                output_directory=tmpdir
            )

            # Create test content with more than 200 characters to ensure full content is used
            long_content = "A" * 500 + "\n" + "B" * 500 + "\n" + "C" * 200

            gathered_content = {
                "/tmp/file1.txt": {
                    "type": "text",
                    "content": long_content
                },
                "/tmp/file2.txt": {
                    "type": "text",
                    "content": "Short content"
                }
            }

            summary = executor._summarize_content(gathered_content)

            # Verify that full content is included, not just a preview
            assert long_content in summary, "Full content should be included in summary"
            assert "Short content" in summary
            assert len(summary) > 1200, "Summary should contain full content from both files"

            # Verify file information is included
            assert "file1.txt" in summary
            assert "file2.txt" in summary
            assert "text" in summary.lower()

    @patch('ai_book_composer.agents.executor.get_llm')
    @patch('ai_book_composer.mcp_client.get_tools')
    def test_summarize_content_with_multiple_files(self, get_tools_mock, mock_llm):
        """Test that _summarize_content handles multiple files correctly."""
        get_tools_mock.return_value = []
        mock_llm.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ExecutorAgent(
                Settings(),
                input_directory=tmpdir,
                output_directory=tmpdir
            )

            # Create multiple files with different content types
            gathered_content = {
                "/tmp/doc1.txt": {
                    "type": "text",
                    "content": "Document 1 has important content about machine learning algorithms and neural networks."
                },
                "/tmp/audio1.mp3": {
                    "type": "audio_transcription",
                    "content": "This is a transcription of the audio file discussing artificial intelligence topics."
                },
                "/tmp/video1.mp4": {
                    "type": "video_transcription",
                    "content": "Video transcription covering deep learning and computer vision applications."
                }
            }

            summary = executor._summarize_content(gathered_content)

            # Verify all content is included
            assert "machine learning algorithms and neural networks" in summary
            assert "artificial intelligence topics" in summary
            assert "deep learning and computer vision applications" in summary

            # Verify file names and types are included
            assert "doc1.txt" in summary
            assert "audio1.mp3" in summary
            assert "video1.mp4" in summary

    @patch('ai_book_composer.agents.executor.get_llm')
    @patch('ai_book_composer.mcp_client.get_tools')
    def test_summarize_content_truncates_large_files(self, get_tools_mock, mock_llm):
        """Test that _summarize_content truncates very large files to manage token limits."""
        get_tools_mock.return_value = []
        mock_llm.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ExecutorAgent(
                Settings(),
                input_directory=tmpdir,
                output_directory=tmpdir
            )

            # Import the constant to use in test
            from ai_book_composer.agents.executor import MAX_CONTENT_FOR_CHAPTER_PLANNING

            # Create content that exceeds the limit
            very_long_content = "X" * (MAX_CONTENT_FOR_CHAPTER_PLANNING + 5000)

            gathered_content = {
                "/tmp/large_file.txt": {
                    "type": "text",
                    "content": very_long_content
                }
            }

            summary = executor._summarize_content(gathered_content)

            # Verify content is truncated
            assert "Content truncated" in summary
            assert len(summary) < len(very_long_content), "Summary should be truncated"
            # Verify first part is included
            assert "X" * 100 in summary, "Beginning of content should be included"
