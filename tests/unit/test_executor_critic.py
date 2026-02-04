"""Test executor critic step functionality."""

from unittest.mock import Mock, patch, call

import pytest

from src.ai_book_composer.agents.executor import ExecutorAgent
from src.ai_book_composer.agents.state import create_initial_state
from src.ai_book_composer.config import Settings


class TestExecutorCriticStep:
    """Test that executor evaluates chapter list and content quality before caching."""

    @patch('src.ai_book_composer.utils.file_utils.write_cache')
    @patch('src.ai_book_composer.agents.agent_base.load_prompts')
    @patch('src.ai_book_composer.agents.agent_base.AgentBase._invoke_agent')
    def test_chapter_list_approved_and_cached(
            self,
            mock_invoke_agent,
            mock_load_prompts,
            mock_write_cache
    ):
        """Test that approved chapter list is cached."""
        # Mock prompts
        mock_load_prompts.return_value = {
            'executor': {
                'chapter_planning_system_prompt': 'Plan chapters in {language}. {style_instructions_section}',
                'chapter_planning_user_prompt': 'Content: {file_summary}',
                'chapter_list_critic_system_prompt': 'Evaluate chapter list in {language}. {style_instructions_section}',
                'chapter_list_critic_user_prompt': 'Evaluate: {chapter_count} chapters\n{chapter_summary}'
            }
        }

        # First call returns chapter list, second call returns approval
        mock_invoke_agent.side_effect = [
            """
Chapter 1: Introduction
Chapter 2: Core Concepts
Chapter 3: Applications
""",
            "APPROVE - The chapter structure is well-organized and comprehensive."
        ]

        settings = Settings()
        settings.book.use_cached_chapters_list = False

        # Create executor
        executor = ExecutorAgent(
            settings,
            output_directory="/tmp/test_output"
        )

        # Create initial state
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

        executor.state = state
        result = executor._plan_chapters_inner()

        # Verify chapter list was created
        assert "chapter_list" in result
        assert len(result["chapter_list"]) == 3

        # Verify write_cache was called (chapter list was approved and cached)
        assert mock_write_cache.called
        assert mock_write_cache.call_count == 1

    @patch('src.ai_book_composer.utils.file_utils.write_cache')
    @patch('src.ai_book_composer.agents.agent_base.load_prompts')
    @patch('src.ai_book_composer.agents.agent_base.AgentBase._invoke_agent')
    def test_chapter_list_rejected(
            self,
            mock_invoke_agent,
            mock_load_prompts,
            mock_write_cache
    ):
        """Test that rejected chapter list is not cached."""
        # Mock prompts
        mock_load_prompts.return_value = {
            'executor': {
                'chapter_planning_system_prompt': 'Plan chapters in {language}. {style_instructions_section}',
                'chapter_planning_user_prompt': 'Content: {file_summary}',
                'chapter_list_critic_system_prompt': 'Evaluate chapter list in {language}. {style_instructions_section}',
                'chapter_list_critic_user_prompt': 'Evaluate: {chapter_count} chapters\n{chapter_summary}'
            }
        }

        # First call returns chapter list, second call returns rejection
        mock_invoke_agent.side_effect = [
            """
Chapter 1: Introduction
Chapter 2: Core Concepts
Chapter 3: Applications
""",
            "REJECT - The chapter structure needs improvement. Topics are too broad.",
            """
Chapter 1: Introduction
Chapter 2: Core Concepts
Chapter 3: Applications
Chapter 4: Additional
""",

        ]

        settings = Settings()
        settings.book.use_cached_chapters_list = False

        # Create executor
        executor = ExecutorAgent(
            settings,
            output_directory="/tmp/test_output"
        )

        # Create initial state
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

        executor.state = state
        result = executor._plan_chapters_inner()

        # Verify chapter list was created
        assert "chapter_list" in result
        assert len(result["chapter_list"]) == 4


    @patch('src.ai_book_composer.utils.file_utils.write_cache')
    @patch('src.ai_book_composer.agents.agent_base.load_prompts')
    @patch('src.ai_book_composer.agents.agent_base.AgentBase._invoke_agent')
    def test_chapter_content_approved_and_cached(
            self,
            mock_invoke_agent,
            mock_load_prompts,
            mock_write_cache
    ):
        """Test that approved chapter content is cached."""
        # Mock prompts
        mock_load_prompts.return_value = {
            'executor': {
                'chapter_generation_system_prompt': 'Generate chapter in {language}. {style_instructions_section}',
                'chapter_generation_user_prompt': 'Write chapter {chapter_number}: {title}\n{description}',
                'chapter_content_critic_system_prompt': 'Evaluate content in {language}. {style_instructions_section}',
                'chapter_content_critic_user_prompt': 'Evaluate Chapter {chapter_number}: {title}\nWords: {word_count}\n{content_preview}'
            }
        }

        # First call returns chapter content, second call returns approval
        chapter_content = "This is a comprehensive introduction to the topic. " * 50
        mock_invoke_agent.side_effect = [
            chapter_content,
            "APPROVE - The chapter content is well-written and informative."
        ]

        settings = Settings()
        settings.book.use_cached_chapters_content = False

        # Create executor
        executor = ExecutorAgent(
            settings,
            output_directory="/tmp/test_output"
        )

        # Create initial state
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

        executor.state = state
        result = executor._generate_chapter_content(1, "Introduction", "Introduction chapter")

        # Verify content was generated
        assert result == chapter_content

        # Verify write_cache was called (content was approved and cached)
        assert mock_write_cache.called
        assert mock_write_cache.call_count == 1

    @patch('src.ai_book_composer.utils.file_utils.write_cache')
    @patch('src.ai_book_composer.agents.agent_base.load_prompts')
    @patch('src.ai_book_composer.agents.agent_base.AgentBase._invoke_agent')
    def test_chapter_content_rejected(
            self,
            mock_invoke_agent,
            mock_load_prompts,
            mock_write_cache
    ):
        """Test that rejected chapter content is not cached."""
        # Mock prompts
        mock_load_prompts.return_value = {
            'executor': {
                'chapter_generation_system_prompt': 'Generate chapter in {language}. {style_instructions_section}',
                'chapter_generation_user_prompt': 'Write chapter {chapter_number}: {title}\n{description}',
                'chapter_content_critic_system_prompt': 'Evaluate content in {language}. {style_instructions_section}',
                'chapter_content_critic_user_prompt': 'Evaluate Chapter {chapter_number}: {title}\nWords: {word_count}\n{content_preview}'
            }
        }

        # First call returns chapter content, second call returns rejection
        chapter_content = "Short content."
        good_chapter_content = "Very good content."

        mock_invoke_agent.side_effect = [
            chapter_content,
            "REJECT - The chapter content is too brief and lacks substance.",
            good_chapter_content
        ]

        settings = Settings()
        settings.book.use_cached_chapters_content = False

        # Create executor
        executor = ExecutorAgent(
            settings,
            output_directory="/tmp/test_output"
        )

        # Create initial state
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

        executor.state = state
        result = executor._generate_chapter_content(1, "Introduction", "Introduction chapter")

        # Verify content was generated
        assert result == good_chapter_content

    @patch('src.ai_book_composer.agents.agent_base.load_prompts')
    @patch('src.ai_book_composer.agents.agent_base.AgentBase._invoke_agent')
    def test_critic_approves_by_default_on_error(
            self,
            mock_invoke_agent,
            mock_load_prompts
    ):
        """Test that critic approves by default when evaluation fails."""
        # Mock prompts (missing critic prompts)
        mock_load_prompts.return_value = {
            'executor': {}
        }

        settings = Settings()
        executor = ExecutorAgent(settings, output_directory="/tmp/test_output")
        executor.state = create_initial_state(
            input_directory="/tmp/test_input",
            output_directory="/tmp/test_output"
        )

        # Test chapter list evaluation without prompts
        is_approved, reason = executor._evaluate_chapter_list_quality([])
        assert is_approved is True  # Should approve by default

        # Test chapter content evaluation without prompts
        is_approved, reason = executor._evaluate_chapter_content_quality(1, "Test", "test", "Content")
        assert is_approved is True  # Should approve by default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
