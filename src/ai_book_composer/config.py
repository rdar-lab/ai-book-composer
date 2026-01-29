"""Configuration management for AI Book Composer."""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # LLM Configuration
    llm_provider: Literal["openai", "gemini", "azure", "ollama"] = "openai"
    llm_model: str = "gpt-4"
    
    # OpenAI
    openai_api_key: str = ""
    
    # Google Gemini
    google_api_key: str = ""
    
    # Azure OpenAI
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = ""
    
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    
    # Book Configuration
    output_language: str = "en-US"
    max_lines_per_read: int = 100


# Global settings instance
settings = Settings()
