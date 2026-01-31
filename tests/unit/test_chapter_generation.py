"""Test chapter-by-chapter generation feature."""

from unittest.mock import Mock, patch, AsyncMock

import pytest
# noinspection PyUnresolvedReferences
from ai_book_composer.agents.executor import ExecutorAgent
# noinspection PyUnresolvedReferences
from ai_book_composer.agents.state import create_initial_state
# noinspection PyUnresolvedReferences
from ai_book_composer.config import Settings


class TestChapterByChapterGeneration:
    """Test that chapters are generated one at a time."""

    @patch('ai_book_composer.agents.agent_base.load_prompts')
    @patch('ai_book_composer.mcp_client.get_tools')
    @patch('ai_book_composer.agents.agent_base.get_llm')
    def test_plan_chapters_creates_individual_tasks(
            self,
            mock_get_llm,
            get_tools_mock,
            mock_load_prompts
    ):
        """Test that plan_chapters creates individual chapter generation tasks."""
        # Mock prompts
        mock_load_prompts.return_value = {
            'executor': {
                'chapter_planning_system_prompt': 'Plan chapters in {language}',
                'chapter_planning_user_prompt': 'Content: {content_summary}',
                'chapter_generation_system_prompt': 'Generate chapter',
                'chapter_generation_user_prompt': 'Generate chapter {chapter_number}'
            }
        }

        # Mock LLM response for chapter planning
        mock_llm = Mock()
        mock_response = Mock()
        # Simulate LLM response with chapter structure
        mock_response.content = """
Chapter 1: Introduction
Chapter 2: Core Concepts
Chapter 3: Applications
"""
        mock_llm.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_llm
        mock_get_llm.return_value = mock_llm

        # Mock ToolRegistry
        write_chapter_list_instance = Mock()
        write_chapter_list_instance.name = "write_chapter_list"
        write_chapter_list_instance.ainvoke = AsyncMock(return_value={"success": True})

        get_tools_mock.return_value = [write_chapter_list_instance]

        settings = Settings()
        settings.parallel.parallel_execution = False  # Disable parallel execution for this test

        # Create executor
        executor = ExecutorAgent(
            settings,
            input_directory="/tmp/test_input",
            output_directory="/tmp/test_output"
        )

        # Create initial state with existing plan
        state = create_initial_state(
            input_directory="/tmp/test_input",
            output_directory="/tmp/test_output"
        )
        state["gathered_content"] = {
            "/tmp/test_input/file1.txt": {
                "type": "text",
                "content": "Sample content for testing"
            }
        }
        state["plan"] = [
            {"task": "gather_content", "description": "Read files", "status": "completed"},
            {"task": "plan_chapters", "description": "Plan chapters", "status": "in_progress"},
            {"task": "generate_chapters", "description": "Generate chapters", "status": "pending"},
            {"task": "compile_references", "description": "Compile references", "status": "pending"},
            {"task": "generate_book", "description": "Generate book", "status": "pending"}
        ]
        state["current_task_index"] = 1  # Currently on plan_chapters

        executor.state = state  # Set current state for tool access
        # Execute plan_chapters
        result = executor._plan_chapters_inner()

        # Check that generate_chapters was replaced with individual tasks
        chapter_list = result['chapter_list']
        assert len(chapter_list) == 3, f"Expected at least 3 individual chapter tasks, got {len(chapter_list)}"

        # Verify each chapter task has required fields
        for chapter in chapter_list:
            assert "number" in chapter
            assert "title" in chapter
            assert "description" in chapter

    @patch('ai_book_composer.agents.agent_base.load_prompts')
    @patch('ai_book_composer.mcp_client.get_tools')
    @patch('ai_book_composer.agents.agent_base.get_llm')
    def test_generate_single_chapter_adds_to_chapters_list(
            self,
            mock_get_llm,
            get_tools_mock,
            mock_load_prompts
    ):
        """Test that generate_single_chapter adds chapter to the list."""
        # Mock prompts
        mock_load_prompts.return_value = {
            'executor': {
                'chapter_generation_system_prompt': 'Generate chapter',
                'chapter_generation_user_prompt': 'Generate chapter {chapter_number}'
            }
        }

        # Mock LLM response for chapter content
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "This is the generated chapter content."
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        mock_llm.bind_tools.return_value = mock_llm

        write_chapter = Mock()
        write_chapter.name = "write_chapter"
        write_chapter.ainvoke = AsyncMock(return_value={"success": True})

        get_tools_mock.return_value = [write_chapter]

        # Create executor
        executor = ExecutorAgent(
            Settings(),
            input_directory="/tmp/test_input",
            output_directory="/tmp/test_output"
        )

        # Create state
        state = create_initial_state(
            input_directory="/tmp/test_input",
            output_directory="/tmp/test_output"
        )
        state["gathered_content"] = {
            "/tmp/test_input/file1.txt": {
                "type": "text",
                "content": "Sample content for testing"
            }
        }
        state["chapters"] = []
        state['chapter_list'] = [
            {
                "number": 1,
                "title": "Introduction",
                "description": "Introduction chapter"
            }
        ]
        state["current_task_index"] = 2

        # Execute generate_single_chapter
        executor.state = state  # Set current state for tool access
        result = executor._generate_chapters_inner()

        # Verify chapter was added
        assert "chapters" in result
        assert len(result["chapters"]) == 1
        assert result["chapters"][0]["number"] == 1
        assert result["chapters"][0]["title"] == "Introduction"
        assert "content" in result["chapters"][0]
        assert result["current_task_index"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
