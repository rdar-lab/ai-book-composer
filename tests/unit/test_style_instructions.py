"""Unit tests for style instructions feature."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import yaml

from ai_book_composer.config import Settings, BookConfig
from ai_book_composer.agents.state import create_initial_state
from ai_book_composer.workflow import BookComposerWorkflow


class TestStyleInstructionsConfig:
    """Test style instructions configuration."""
    
    def test_default_config_has_empty_style_instructions(self):
        """Test that default configuration has empty style instructions."""
        settings = Settings()
        
        assert hasattr(settings.book, 'style_instructions')
        assert settings.book.style_instructions == ""
    
    def test_custom_style_instructions_in_config_file(self, tmp_path):
        """Test loading custom style instructions from config file."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            'book': {
                'output_language': 'en-US',
                'default_title': 'Test Book',
                'default_author': 'Test Author',
                'quality_threshold': 0.7,
                'max_iterations': 3,
                'style_instructions': 'I want an academic book with formal language'
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        settings = Settings(str(config_file))
        
        assert settings.book.style_instructions == 'I want an academic book with formal language'
    
    def test_book_config_model_accepts_style_instructions(self):
        """Test that BookConfig model accepts style_instructions field."""
        book_config = BookConfig(
            output_language='en-US',
            default_title='Test Book',
            default_author='Test Author',
            quality_threshold=0.7,
            max_iterations=3,
            style_instructions='I want it to be light reading'
        )
        
        assert book_config.style_instructions == 'I want it to be light reading'


class TestStyleInstructionsState:
    """Test style instructions in agent state."""
    
    def test_initial_state_includes_style_instructions(self):
        """Test that initial state includes style_instructions field."""
        state = create_initial_state(
            input_directory='/tmp/input',
            output_directory='/tmp/output',
            language='en-US',
            book_title='Test Book',
            book_author='Test Author',
            style_instructions='I want professional reading material'
        )
        
        assert 'style_instructions' in state
        assert state['style_instructions'] == 'I want professional reading material'
    
    def test_initial_state_default_empty_style_instructions(self):
        """Test that initial state defaults to empty style instructions."""
        state = create_initial_state(
            input_directory='/tmp/input',
            output_directory='/tmp/output'
        )
        
        assert 'style_instructions' in state
        assert state['style_instructions'] == ""


class TestStyleInstructionsWorkflow:
    """Test style instructions in workflow."""
    
    @patch('ai_book_composer.workflow.PlannerAgent')
    @patch('ai_book_composer.workflow.ExecutorAgent')
    @patch('ai_book_composer.workflow.DecoratorAgent')
    @patch('ai_book_composer.workflow.CriticAgent')
    def test_workflow_accepts_style_instructions(
        self,
        mock_critic,
        mock_decorator,
        mock_executor,
        mock_planner
    ):
        """Test that workflow accepts and stores style instructions."""
        workflow = BookComposerWorkflow(
            input_directory='/tmp/input',
            output_directory='/tmp/output',
            language='en-US',
            book_title='Test Book',
            book_author='Test Author',
            max_iterations=3,
            style_instructions='I want kids/fun reading material'
        )
        
        assert workflow.style_instructions == 'I want kids/fun reading material'
    
    @patch('ai_book_composer.workflow.PlannerAgent')
    @patch('ai_book_composer.workflow.ExecutorAgent')
    @patch('ai_book_composer.workflow.DecoratorAgent')
    @patch('ai_book_composer.workflow.CriticAgent')
    def test_workflow_default_empty_style_instructions(
        self,
        mock_critic,
        mock_decorator,
        mock_executor,
        mock_planner
    ):
        """Test that workflow defaults to empty style instructions."""
        workflow = BookComposerWorkflow(
            input_directory='/tmp/input',
            output_directory='/tmp/output'
        )
        
        assert workflow.style_instructions == ""


class TestStyleInstructionsIntegration:
    """Integration tests for style instructions feature."""
    
    @patch('ai_book_composer.cli.BookComposerWorkflow')
    @patch('ai_book_composer.cli.Settings')
    @patch('ai_book_composer.cli.setup_logging')
    def test_cli_passes_style_instructions_to_workflow(
        self,
        mock_setup_logging,
        mock_settings,
        mock_workflow
    ):
        """Test that CLI properly passes style instructions to workflow."""
        from ai_book_composer.cli import main
        from click.testing import CliRunner
        
        # Mock settings
        mock_settings_instance = Mock()
        mock_settings_instance.book.default_title = "Default Title"
        mock_settings_instance.book.default_author = "Default Author"
        mock_settings_instance.book.output_language = "en-US"
        mock_settings_instance.book.max_iterations = 3
        mock_settings_instance.book.style_instructions = ""
        mock_settings_instance.llm.provider = "ollama_embedded"
        mock_settings_instance.llm.model = "llama-3.2-3b-instruct"
        mock_settings.return_value = mock_settings_instance
        
        # Mock workflow
        mock_workflow_instance = Mock()
        mock_workflow_instance.run.return_value = {
            'status': 'completed',
            'chapters': [],
            'references': [],
            'iterations': 0,
            'quality_score': 0.9,
            'final_output_path': '/tmp/output/book.rtf'
        }
        mock_workflow.return_value = mock_workflow_instance
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('input').mkdir()
            Path('output').mkdir()
            
            result = runner.invoke(main, [
                '-i', 'input',
                '-o', 'output',
                '--style-instructions', 'I want an academic book'
            ])
            
            # Verify workflow was called with style_instructions
            mock_workflow.assert_called_once()
            call_kwargs = mock_workflow.call_args[1]
            assert call_kwargs['style_instructions'] == 'I want an academic book'
    
    @patch('ai_book_composer.cli.BookComposerWorkflow')
    @patch('ai_book_composer.cli.Settings')
    @patch('ai_book_composer.cli.setup_logging')
    def test_cli_uses_config_style_instructions_when_not_provided(
        self,
        mock_setup_logging,
        mock_settings,
        mock_workflow
    ):
        """Test that CLI uses config style instructions when not provided via CLI."""
        from ai_book_composer.cli import main
        from click.testing import CliRunner
        
        # Mock settings with style instructions in config
        mock_settings_instance = Mock()
        mock_settings_instance.book.default_title = "Default Title"
        mock_settings_instance.book.default_author = "Default Author"
        mock_settings_instance.book.output_language = "en-US"
        mock_settings_instance.book.max_iterations = 3
        mock_settings_instance.book.style_instructions = "professional reading material"
        mock_settings_instance.llm.provider = "ollama_embedded"
        mock_settings_instance.llm.model = "llama-3.2-3b-instruct"
        mock_settings.return_value = mock_settings_instance
        
        # Mock workflow
        mock_workflow_instance = Mock()
        mock_workflow_instance.run.return_value = {
            'status': 'completed',
            'chapters': [],
            'references': [],
            'iterations': 0,
            'quality_score': 0.9,
            'final_output_path': '/tmp/output/book.rtf'
        }
        mock_workflow.return_value = mock_workflow_instance
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('input').mkdir()
            Path('output').mkdir()
            
            result = runner.invoke(main, [
                '-i', 'input',
                '-o', 'output'
            ])
            
            # Verify workflow was called with style_instructions from config
            mock_workflow.assert_called_once()
            call_kwargs = mock_workflow.call_args[1]
            assert call_kwargs['style_instructions'] == 'professional reading material'
