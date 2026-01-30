"""Unit tests for configuration and LLM provider setup."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import yaml

from ai_book_composer.config import Settings
from ai_book_composer.llm import get_llm


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
        assert "model_path" in ollama_embedded_config
        assert "n_ctx" in ollama_embedded_config
        assert "n_threads" in ollama_embedded_config
        assert "n_gpu_layers" in ollama_embedded_config
    
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
    
    @patch('ai_book_composer.llm.ChatOpenAI')
    def test_openai_provider(self, mock_openai):
        """Test OpenAI provider initialization."""
        with patch('ai_book_composer.llm.settings') as mock_settings:
            mock_settings.llm.provider = 'openai'
            mock_settings.llm.model = 'gpt-4'
            mock_settings.get_provider_config.return_value = {'api_key': 'test-key'}
            
            get_llm(temperature=0.7)
            
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs['model'] == 'gpt-4'
            assert call_kwargs['temperature'] == 0.7
    
    @patch('ai_book_composer.llm.ChatOllama')
    def test_ollama_server_provider(self, mock_ollama):
        """Test Ollama server provider initialization."""
        with patch('ai_book_composer.llm.settings') as mock_settings:
            mock_settings.llm.provider = 'ollama'
            mock_settings.llm.model = 'llama2'
            mock_settings.get_provider_config.return_value = {
                'model': 'llama2',
                'base_url': 'http://localhost:11434'
            }
            
            get_llm(temperature=0.7, provider='ollama')
            
            mock_ollama.assert_called_once()
            call_kwargs = mock_ollama.call_args[1]
            assert call_kwargs['model'] == 'llama2'
            assert call_kwargs['temperature'] == 0.7
            assert call_kwargs['base_url'] == 'http://localhost:11434'
    
    @patch('ai_book_composer.llm.ChatLlamaCpp')
    def test_embedded_ollama_provider(self, mock_llamacpp):
        """Test embedded ollama provider initialization."""
        with patch('ai_book_composer.llm.settings') as mock_settings:
            mock_settings.llm.provider = 'ollama_embedded'
            mock_settings.llm.model = 'llama-3.2-1b-instruct'
            mock_settings.get_provider_config.return_value = {
                'model_path': 'models/test-model.gguf',
                'n_ctx': 2048,
                'n_threads': 4,
                'n_gpu_layers': 0,
                'verbose': False
            }
            
            get_llm(temperature=0.7, provider='ollama_embedded')
            
            mock_llamacpp.assert_called_once()
            call_kwargs = mock_llamacpp.call_args[1]
            assert 'model_path' in call_kwargs
            assert call_kwargs['temperature'] == 0.7
            assert call_kwargs['n_ctx'] == 2048
            assert call_kwargs['n_threads'] == 4
            assert call_kwargs['n_gpu_layers'] == 0
    
    def test_unsupported_provider(self):
        """Test that unsupported provider raises error."""
        with patch('ai_book_composer.llm.settings') as mock_settings:
            mock_settings.llm.provider = 'unsupported'
            
            with pytest.raises(ValueError, match="Unsupported LLM provider"):
                get_llm(provider='unsupported')
