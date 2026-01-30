"""Configuration management for AI Book Composer."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = "ollama_embedded"
    model: str = "llama-3.2-1b-instruct"
    temperature: Dict[str, float] = Field(default_factory=lambda: {
        "planning": 0.3,
        "execution": 0.7,
        "critique": 0.2
    })


class WhisperConfig(BaseModel):
    """Whisper configuration."""
    mode: str = "local"
    model_size: str = "base"
    remote: Dict[str, Optional[str]] = Field(default_factory=lambda: {
        "endpoint": "http://localhost:9000",
        "api_key": None
    })
    local: Dict[str, str] = Field(default_factory=lambda: {
        "device": "cpu",
        "compute_type": "int8"
    })


class TextReadingConfig(BaseModel):
    """Text reading configuration."""
    max_lines_per_read: int = 100
    supported_formats: list = Field(default_factory=lambda: ["txt", "md", "rst", "docx", "rtf", "pdf"])


class MediaProcessingConfig(BaseModel):
    """Media processing configuration."""
    audio_formats: list = Field(default_factory=lambda: ["mp3", "wav", "m4a", "flac", "ogg"])
    video_formats: list = Field(default_factory=lambda: ["mp4", "avi", "mov", "mkv"])
    chunk_duration: int = 300
    max_file_duration: int = 3600


class ImageProcessingConfig(BaseModel):
    """Image processing configuration."""
    supported_formats: list = Field(default_factory=lambda: ["jpg", "jpeg", "png", "gif", "bmp"])
    extract_from_pdf: bool = True
    max_image_size_mb: int = 10
    max_images_per_chapter: int = 5


class BookConfig(BaseModel):
    """Book configuration."""
    output_language: str = "en-US"
    default_title: str = "Composed Book"
    default_author: str = "AI Book Composer"
    quality_threshold: float = 0.7
    max_iterations: int = 3
    style_instructions: str = ""  # Optional instructions to guide AI on book style


class ParallelConfig(BaseModel):
    """Parallel execution configuration."""
    parallel_execution: bool = True  # true=enabled, false=disabled
    parallel_workers: int = Field(default=4, ge=1, le=32)  # 1-32 workers


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    file: str = "logs/ai_book_composer.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_output: bool = False


class SecurityConfig(BaseModel):
    """Security configuration."""
    allow_directory_traversal: bool = False
    max_file_size_mb: int = 500


class MCPServerConfig(BaseModel):
    """MCP Server configuration."""
    host: str = "127.0.0.1"
    port: int = 8000
    name: str = "ai-book-composer"
    debug: bool = False
    log_level: str = "INFO"


class Settings:
    """Application settings loaded from YAML."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize settings from YAML file.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.yaml
        """
        if config_path is None:
            # Look for config.yaml in current directory, then in package directory
            if Path("config.yaml").exists():
                config_path = "config.yaml"
            else:
                config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            # Use defaults if config file doesn't exist
            self._config = self._get_defaults()
        else:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                # Replace environment variable placeholders
                config_data = self._replace_env_vars(config_data)
                self._config = config_data
        
        # Initialize config objects
        self.llm = LLMConfig(**self._config.get('llm', {}))
        self.whisper = WhisperConfig(**self._config.get('whisper', {}))
        self.text_reading = TextReadingConfig(**self._config.get('text_reading', {}))
        self.media_processing = MediaProcessingConfig(**self._config.get('media_processing', {}))
        self.image_processing = ImageProcessingConfig(**self._config.get('image_processing', {}))
        self.book = BookConfig(**self._config.get('book', {}))
        self.logging = LoggingConfig(**self._config.get('logging', {}))
        self.security = SecurityConfig(**self._config.get('security', {}))
        self.mcp_server = MCPServerConfig(**self._config.get('mcp_server', {}))
        self.parallel = ParallelConfig(**self._config.get('parallel', {}))
        
        # Store provider configurations
        self.providers = self._config.get('providers', {})
    
    def _replace_env_vars(self, data: Any) -> Any:
        """Recursively replace ${VAR} with environment variables."""
        if isinstance(data, dict):
            return {k: self._replace_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._replace_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith('${') and data.endswith('}'):
            var_name = data[2:-1]
            return os.environ.get(var_name, data)
        return data
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'llm': {
                'provider': 'ollama_embedded',
                'model': 'llama-3.2-3b-instruct',
                'temperature': {'planning': 0.3, 'execution': 0.7, 'critique': 0.2}
            },
            'whisper': {
                'mode': 'local',
                'model_size': 'base',
                'remote': {'endpoint': 'http://localhost:9000', 'api_key': None},
                'local': {'device': 'cpu', 'compute_type': 'int8'}
            },
            'text_reading': {
                'max_lines_per_read': 100,
                'supported_formats': ['txt', 'md', 'rst', 'docx', 'rtf', 'pdf']
            },
            'media_processing': {
                'audio_formats': ['mp3', 'wav', 'm4a', 'flac', 'ogg'],
                'video_formats': ['mp4', 'avi', 'mov', 'mkv'],
                'chunk_duration': 300,
                'max_file_duration': 3600
            },
            'image_processing': {
                'supported_formats': ['jpg', 'jpeg', 'png', 'gif', 'bmp'],
                'extract_from_pdf': True,
                'max_image_size_mb': 10,
                'max_images_per_chapter': 5
            },
            'book': {
                'output_language': 'en-US',
                'default_title': 'Composed Book',
                'default_author': 'AI Book Composer',
                'quality_threshold': 0.7,
                'max_iterations': 3,
                'style_instructions': ''
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/ai_book_composer.log',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'console_output': False
            },
            'security': {
                'allow_directory_traversal': False,
                'max_file_size_mb': 500
            },
            'parallel': {
                'parallel_execution': True,
                'parallel_workers': 4
            },
            'providers': {
                'openai': {'api_key': os.environ.get('OPENAI_API_KEY', '')},
                'gemini': {'api_key': os.environ.get('GOOGLE_API_KEY', '')},
                'azure': {
                    'api_key': os.environ.get('AZURE_OPENAI_API_KEY', ''),
                    'endpoint': os.environ.get('AZURE_OPENAI_ENDPOINT', ''),
                    'deployment': os.environ.get('AZURE_OPENAI_DEPLOYMENT', '')
                },
                'ollama': {
                    'base_url': 'http://localhost:11434',
                    'model': 'llama2'
                },
                'ollama_embedded': {
                    'model_name': 'llama-3.2-3b-instruct',
                    'n_ctx': 2048,
                    'n_threads': 4,
                    'run_on_gpu': False,
                    'verbose': False
                }
            }
        }
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        return self.providers.get(provider, {})


# Global settings instance
settings = Settings()


def load_prompts(prompts_path: Optional[str] = None) -> Dict[str, Any]:
    """Load prompts from YAML file.
    
    Args:
        prompts_path: Path to prompts file. If None, uses default prompts.yaml
        
    Returns:
        Dictionary of prompts
    """
    if prompts_path is None:
        if Path("prompts.yaml").exists():
            prompts_path = "prompts.yaml"
        else:
            prompts_path = Path(__file__).parent.parent.parent / "prompts.yaml"
    
    prompts_path = Path(prompts_path)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    
    with open(prompts_path, 'r') as f:
        return yaml.safe_load(f)

