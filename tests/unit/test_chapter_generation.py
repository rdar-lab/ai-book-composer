"""Test chapter-by-chapter generation feature."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_book_composer.agents.state import create_initial_state
from ai_book_composer.agents.executor import ExecutorAgent


class TestChapterByChapterGeneration:
    """Test that chapters are generated one at a time."""
    
    @patch('ai_book_composer.agents.executor.load_prompts')
    @patch('ai_book_composer.agents.executor.ToolRegistry')
    @patch('ai_book_composer.agents.executor.get_llm')
    def test_plan_chapters_creates_individual_tasks(
        self, 
        mock_get_llm,
        mock_tool_registry,
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
        mock_get_llm.return_value = mock_llm
        
        # Mock ToolRegistry
        mock_registry_instance = Mock()
        mock_registry_instance.get_langchain_tools.return_value = []
        mock_registry_instance.write_chapter_list.return_value = {"success": True}
        mock_tool_registry.return_value = mock_registry_instance
        
        # Create executor
        executor = ExecutorAgent(
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
        
        # Execute plan_chapters
        result = executor._plan_chapters(state)
        
        # Verify that individual chapter tasks were created
        assert "plan" in result
        new_plan = result["plan"]
        
        # Check that generate_chapters was replaced with individual tasks
        chapter_tasks = [task for task in new_plan if task.get("task") == "generate_single_chapter"]
        assert len(chapter_tasks) >= 3, f"Expected at least 3 individual chapter tasks, got {len(chapter_tasks)}"
        
        # Verify each chapter task has required fields
        for chapter_task in chapter_tasks:
            assert "chapter_number" in chapter_task
            assert "chapter_title" in chapter_task
            assert "chapter_description" in chapter_task
            assert chapter_task["status"] == "pending"
        
        # Verify other tasks are still present
        other_tasks = [task for task in new_plan if task.get("task") != "generate_single_chapter"]
        assert any(task.get("task") == "compile_references" for task in other_tasks)
        assert any(task.get("task") == "generate_book" for task in other_tasks)
        
    @patch('ai_book_composer.agents.executor.load_prompts')
    @patch('ai_book_composer.agents.executor.ToolRegistry')
    @patch('ai_book_composer.agents.executor.get_llm')
    def test_generate_single_chapter_adds_to_chapters_list(
        self,
        mock_get_llm,
        mock_tool_registry,
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
        
        # Mock ToolRegistry
        mock_registry_instance = Mock()
        mock_registry_instance.get_langchain_tools.return_value = []
        mock_registry_instance.write_chapter.return_value = {"success": True}
        mock_tool_registry.return_value = mock_registry_instance
        
        # Create executor
        executor = ExecutorAgent(
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
        state["current_task_index"] = 2
        
        # Create a chapter task
        chapter_task = {
            "task": "generate_single_chapter",
            "description": "Generate Chapter 1: Introduction",
            "chapter_number": 1,
            "chapter_title": "Introduction",
            "chapter_description": "Introduction chapter"
        }
        
        # Execute generate_single_chapter
        result = executor._generate_single_chapter(state, chapter_task)
        
        # Verify chapter was added
        assert "chapters" in result
        assert len(result["chapters"]) == 1
        assert result["chapters"][0]["number"] == 1
        assert result["chapters"][0]["title"] == "Introduction"
        assert "content" in result["chapters"][0]
        assert result["current_task_index"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
