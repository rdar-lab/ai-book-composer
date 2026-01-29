"""Unit tests for agents with mocked LLMs."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from ai_book_composer.agents.state import AgentState, create_initial_state
from ai_book_composer.agents.planner import PlannerAgent
from ai_book_composer.agents.critic import CriticAgent


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
        
        planner = PlannerAgent()
        
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
        
        critic = CriticAgent(quality_threshold=0.7)
        
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
        
        critic = CriticAgent()
        
        state = create_initial_state(
            input_directory="/tmp/input",
            output_directory="/tmp/output"
        )
        state["chapters"] = []
        
        result = critic.critique(state)
        
        assert result["quality_score"] == 0.0
        assert result["status"] == "needs_revision"
        assert not mock_llm.invoke.called  # Should not call LLM if no chapters
