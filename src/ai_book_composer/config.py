"""Configuration management for AI Book Composer."""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from pydantic import BaseModel, Field


class GeneralConfig(BaseModel):
    cache_dir: str = ".cache"


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = "ollama_embedded"
    model: str = "llama-3.2-1b-instruct"
    temperature: Dict[str, float] = Field(default_factory=lambda: {
        "planning": 0.3,
        "execution": 0.7,
        "critique": 0.2
    })
    static_plan: bool = True


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


class VisionModelConfig(BaseModel):
    """Vision model configuration for image description."""
    provider: str = "openai"  # Options: openai, gemini, azure, ollama
    model: str = "gpt-4o-mini"  # Vision-capable model
    temperature: float = 0.3


class BookConfig(BaseModel):
    """Book configuration."""
    output_language: str = "en-US"
    default_title: str = "Composed Book"
    default_author: str = "AI Book Composer"
    quality_threshold: float = 0.7
    max_iterations: int = 3
    style_instructions: str = ""  # Optional instructions to guide AI on book style
    use_cached_plan: bool = True  # Whether to cache the generated plan
    use_cached_chapters_list: bool = True  # Whether to cache the chapter list
    use_cached_chapters_content: bool = True  # Whether to cache individual chapter content


class ParallelConfig(BaseModel):
    """Parallel execution configuration."""
    parallel_execution: bool = True  # true=enabled, false=disabled
    parallel_workers: int = Field(default=4, ge=1, le=32)  # 1-32 workers


class RAGConfig(BaseModel):
    """RAG (Retrieval Augmented Generation) configuration."""
    enabled: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 5
    max_allowed_distance: float = 0.9


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


class Settings:
    """Application settings loaded from YAML."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize settings from YAML file.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.yaml
        """
        if config_path is not None:
            self.config_path = Path(config_path)
        else:
            self.config_path = None

        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file."""

        config_data = None

        if self.config_path is None:
            # Use defaults
            config_data = self._get_defaults()
        elif not self.config_path.exists():
            raise Exception("Configuration file not found: {}".format(self.config_path))
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
        self.vision_model = VisionModelConfig(**self._config.get('vision_model', {}))
        self.book = BookConfig(**self._config.get('book', {}))
        self.logging = LoggingConfig(**self._config.get('logging', {}))
        self.security = SecurityConfig(**self._config.get('security', {}))
        self.parallel = ParallelConfig(**self._config.get('parallel', {}))
        self.rag = RAGConfig(**self._config.get('rag', {}))
        self.general = GeneralConfig(**self._config.get('general', {}))

        # Store provider configurations
        self.providers = self._config.get('providers', {})

    def save_config(self, path: Optional[str] = None):
        # Save current configuration to YAML file.
        if path is None:
            path = self.config_path

        self._sync_config_state()

        with open(path, 'w') as f:
            yaml.dump(self._config, f)

    def _sync_config_state(self):
        # Reflect the state back to the self._config dictionary
        self._config['llm'] = self.llm.model_dump()
        self._config['whisper'] = self.whisper.model_dump()
        self._config['text_reading'] = self.text_reading.model_dump()
        self._config['media_processing'] = self.media_processing.model_dump()
        self._config['image_processing'] = self.image_processing.model_dump()
        self._config['vision_model'] = self.vision_model.model_dump()
        self._config['book'] = self.book.model_dump()
        self._config['logging'] = self.logging.model_dump()
        self._config['security'] = self.security.model_dump()
        self._config['parallel'] = self.parallel.model_dump()
        self._config['rag'] = self.rag.model_dump()
        self._config['providers'] = self.providers

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

    @staticmethod
    def _get_defaults() -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'llm': {
                'provider': 'ollama_embedded',
                'model': 'llama-3.1-8b-instruct',
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
            'vision_model': {
                'provider': 'openai',
                'model': 'gpt-4o-mini',
                'temperature': 0.3
            },
            'book': {
                'output_language': 'en-US',
                'default_title': 'Composed Book',
                'default_author': 'AI Book Composer',
                'quality_threshold': 0.7,
                'max_iterations': 3,
                'style_instructions': '',
                'use_cached_plan': True,
                'use_cached_chapters_list': True,
                'use_cached_chapters_content': True
            },
            'logging': {
                'level': 'DEBUG',
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
            'rag': {
                'enabled': True,
                'embedding_model': 'all-MiniLM-L6-v2',
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'retrieval_k': 5,
                'max_allowed_distance': 0.9
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
                    'base_url': 'http://localhost:11434'
                },
                'ollama_embedded': {
                    'internal': {
                        'n_ctx': 131072,
                        'n_threads': 4,
                        'n_batch': 64,
                        'verbose': False
                    },
                    'run_on_gpu': False,
                }
            },
            'general': {
                'cache_dir': '.cache'
            }
        }

    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        return self.providers.get(provider, {})


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
