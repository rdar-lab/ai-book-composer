"""Unit tests for agents with mocked LLMs."""

import tempfile
from unittest.mock import Mock, patch

# noinspection PyUnresolvedReferences
from ai_book_composer.agents.critic import CriticAgent
# noinspection PyUnresolvedReferences
from ai_book_composer.agents.executor import ExecutorAgent
# noinspection PyUnresolvedReferences
from ai_book_composer.agents.planner import PlannerAgent
# noinspection PyUnresolvedReferences
from ai_book_composer.agents.state import AgentState, create_initial_state
# noinspection PyUnresolvedReferences
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

    def test_plan_generation_static_plan(self):
        """Test plan generation."""
        settings = Settings()
        settings.llm.static_plan = True

        planner = PlannerAgent(settings)

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

    @patch('ai_book_composer.agents.agent_base.get_llm')
    def test_plan_generation_llm_plan(self, mock_get_llm):
        """Test plan generation with LLM (non-static plan)."""
        # Mock LLM response with a valid plan JSON (as string)
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = '''[
            {"task": "gather_content", "description": "Read and transcribe all source files", "status": "pending", "files": ["file1.txt", "file2.txt"]},
            {"task": "plan_chapters", "description": "Determine book structure and chapters", "status": "pending"},
            {"task": "generate_chapters", "description": "Write each chapter based on gathered content", "status": "pending"},
            {"task": "compile_references", "description": "Compile list of references", "status": "pending"},
            {"task": "generate_book", "description": "Generate final book with all components", "status": "pending"}
        ]'''
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        settings = Settings()
        settings.llm.static_plan = False
        planner = PlannerAgent(settings)
        planner.prompts = {
            'planner': {
                'system_prompt': 'SYSTEM {language} {style_instructions_section}',
                'user_prompt': 'USER {file_summary}'
            }
        }
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
        plan = result["plan"]
        assert isinstance(plan, list)
        assert plan[0]["task"] == "gather_content"
        assert plan[1]["task"] == "plan_chapters"
        assert plan[2]["task"] == "generate_chapters"
        assert plan[3]["task"] == "compile_references"
        assert plan[4]["task"] == "generate_book"
        assert plan[0]["files"] == ["file1.txt", "file2.txt"]


class TestCriticAgent:
    """Test critic agent with mocked LLM."""

    @patch('ai_book_composer.agents.agent_base.get_llm')
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

    @patch('ai_book_composer.agents.agent_base.get_llm')
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

    @patch('ai_book_composer.agents.agent_base.get_llm')
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

    @patch('ai_book_composer.agents.agent_base.get_llm')
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

    @patch('ai_book_composer.agents.agent_base.get_llm')
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
            # noinspection PyUnresolvedReferences
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

    @patch('ai_book_composer.agents.agent_base.AgentBase._invoke_agent')
    @patch('ai_book_composer.mcp_client.get_tools')
    def test_llm_agent_else_branch(self, get_tools_mock, mock_invoke_agent):
        """Test ExecutorAgent LLM agent else branch uses prompts and tools correctly."""
        # Mock tools (none needed for this test, just to satisfy init)
        get_tools_mock.return_value = []
        mock_invoke_agent.return_value = '{"result": "Tool executed successfully"}'

        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ExecutorAgent(
                Settings(),
                input_directory=tmpdir,
                output_directory=tmpdir
            )
            # Patch prompts to ensure LLM agent prompt is used
            executor.prompts['executor']['llm_agent_system_prompt'] = 'SYSTEM PROMPT'
            executor.prompts['executor']['llm_agent_user_prompt'] = 'USER PROMPT {state} {current_task}'

            # Create a state and a plan with an unknown task (triggers else branch)
            state = create_initial_state(
                input_directory=tmpdir,
                output_directory=tmpdir
            )
            state['plan'] = [
                {"task": "special task", "description": "Do something special!", "status": "pending"}
            ]
            state['current_task_index'] = 0

            result = executor.execute(state)

            # Check that the LLM was called with the correct prompt
            assert mock_invoke_agent.called
            called_args = mock_invoke_agent.call_args[0]
            assert any('SYSTEM PROMPT' in m for m in called_args)
            assert any('USER PROMPT' in m for m in called_args)
            # Check result structure
            assert 'llm_agent_result' in result
            assert result['status'] == 'executing'
            assert result['current_task_index'] == 1

