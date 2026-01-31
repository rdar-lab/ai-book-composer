"""Unit tests for configuration and LLM provider setup."""

from unittest.mock import patch

import pytest
import yaml

from src.ai_book_composer.config import Settings
from src.ai_book_composer.llm import get_llm


class TestConfiguration:
    """Test configuration management."""

    def test_default_config_uses_embedded_ollama(self):
        """Test that default configuration uses embedded ollama."""
        settings = Settings()

        assert settings.llm.provider == "ollama_embedded"
        assert settings.whisper.mode == "local"

    def test_embedded_ollama_config(self):
        """Test embedded ollama configuration."""
        settings = Settings()

        ollama_embedded_config = settings.get_provider_config("ollama_embedded")
        internal = ollama_embedded_config.get("internal", {})
        assert "n_ctx" in internal
        assert "n_threads" in internal
        assert "run_on_gpu" in ollama_embedded_config

    def test_custom_config_file(self, tmp_path):
        """Test loading custom configuration file."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            'llm': {
                'provider': 'openai',
                'model': 'gpt-4'
            },
            'providers': {
                'openai': {
                    'api_key': 'test-key'
                }
            }
        }

        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        settings = Settings(str(config_file))

        assert settings.llm.provider == "openai"
        assert settings.llm.model == "gpt-4"


class TestLLMProvider:
    """Test LLM provider setup."""

    @patch('src.ai_book_composer.llm.ChatOpenAI')
    def test_openai_provider(self, mock_openai):
        """Test OpenAI provider initialization."""
        mock_settings = Settings()
        mock_settings.llm.provider = 'openai'
        mock_settings.llm.model = 'gpt-4'
        mock_settings.providers['openai'] = {'api_key': 'test-key'}

        get_llm(mock_settings, temperature=0.7)

        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs['model'] == 'gpt-4'
        assert call_kwargs['temperature'] == 0.7

    @patch('src.ai_book_composer.llm.ChatOllama')
    def test_ollama_server_provider(self, mock_ollama):
        """Test Ollama server provider initialization."""
        mock_settings = Settings()
        mock_settings.llm.provider = 'ollama'
        mock_settings.llm.model = 'llama2'
        mock_settings.providers['ollama'] = {
            'model': 'llama2',
            'base_url': 'http://localhost:11434'
        }

        get_llm(mock_settings, temperature=0.7, provider='ollama')

        mock_ollama.assert_called_once()
        call_kwargs = mock_ollama.call_args[1]
        assert call_kwargs['model'] == 'llama2'
        assert call_kwargs['temperature'] == 0.7
        assert call_kwargs['base_url'] == 'http://localhost:11434'

    @patch('src.ai_book_composer.llm.hf_hub_download')
    @patch('src.ai_book_composer.llm.ChatLlamaCpp')
    def test_embedded_ollama_provider(self, mock_llamacpp, mock_download):
        """Test embedded ollama provider initialization."""
        mock_download.return_value = '/path/to/downloaded/model.gguf'

        mock_settings = Settings()
        mock_settings.llm.provider = 'ollama_embedded'
        mock_settings.llm.model = 'llama-3.2-3b-instruct'
        mock_settings.providers['ollama_embedded'] = {
            'model_name': 'llama-3.2-3b-instruct',
            'internal':{
                'n_ctx': 2048,
                'n_threads': 4,
                'verbose': False
            },
            'run_on_gpu': False,
        }

        get_llm(mock_settings, temperature=0.7, provider='ollama_embedded')

        # Verify download was called
        mock_download.assert_called_once()

        # Verify ChatLlamaCpp was called with correct parameters
        mock_llamacpp.assert_called_once()
        call_kwargs = mock_llamacpp.call_args[1]
        assert 'model_path' in call_kwargs
        assert call_kwargs['temperature'] == 0.7
        assert call_kwargs['n_ctx'] == 2048
        assert call_kwargs['n_threads'] == 4
        assert call_kwargs['n_gpu_layers'] == 0  # False -> 0

    def test_embedded_ollama_with_gpu(self):
        """Test that run_on_gpu=True sets n_gpu_layers=-1."""
        with patch('src.ai_book_composer.llm.hf_hub_download') as mock_download, \
                patch('src.ai_book_composer.llm.ChatLlamaCpp') as mock_llamacpp:
            mock_download.return_value = '/path/to/model.gguf'
            mock_settings = Settings()
            mock_settings.llm.provider = 'ollama_embedded'
            mock_settings.model = 'llama-3.2-3b-instruct'
            mock_settings.providers['ollama_embedded'] = {
                'internal': {
                    'n_ctx': 2048,
                    'n_threads': 4,
                    'verbose': False
                },
                'run_on_gpu': True
            }

            get_llm(mock_settings, provider='ollama_embedded')

            call_kwargs = mock_llamacpp.call_args[1]
            assert call_kwargs['n_gpu_layers'] == -1  # True -> -1

    def test_embedded_ollama_unknown_model(self):
        """Test that unknown model name raises error."""
        mock_settings = Settings()
        mock_settings.llm.provider = 'ollama_embedded'
        mock_settings.llm.model = 'nonexistent-model'
        mock_settings.providers['ollama_embedded'] = {
            'internal': {
                'n_ctx': 2048,
                'n_threads': 4,
                'verbose': False
            },
            'run_on_gpu': False,
        }

        with pytest.raises(ValueError, match="Unknown embedded model"):
            get_llm(mock_settings, provider='ollama_embedded')

    def test_unsupported_provider(self):
        """Test that unsupported provider raises error."""
        mock_settings = Settings()
        mock_settings.llm.provider = 'unsupported'

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm(mock_settings, provider='unsupported')
