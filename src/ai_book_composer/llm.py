"""LLM provider abstraction."""

from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama

from .config import settings
from .logging_config import logger


def get_llm(
    temperature: float = 0.7,
    model: Optional[str] = None,
    provider: Optional[str] = None
) -> BaseChatModel:
    """Get LLM instance based on configuration.
    
    Args:
        temperature: Temperature for generation (0.0 to 1.0)
        model: Optional model override
        provider: Optional provider override
        
    Returns:
        Configured LLM instance
    """
    provider = provider or settings.llm.provider
    model_name = model or settings.llm.model
    
    logger.info(f"Initializing LLM: provider={provider}, model={model_name}, temperature={temperature}")
    
    try:
        if provider == "openai":
            provider_config = settings.get_provider_config("openai")
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=provider_config.get("api_key", "")
            )
        
        elif provider == "gemini":
            provider_config = settings.get_provider_config("gemini")
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=provider_config.get("api_key", "")
            )
        
        elif provider == "azure":
            provider_config = settings.get_provider_config("azure")
            return AzureChatOpenAI(
                deployment_name=provider_config.get("deployment", ""),
                temperature=temperature,
                api_key=provider_config.get("api_key", ""),
                azure_endpoint=provider_config.get("endpoint", "")
            )
        
        elif provider == "ollama":
            provider_config = settings.get_provider_config("ollama")
            # Use model from provider config if not specified
            ollama_model = model_name if model else provider_config.get("model", "llama2")
            base_url = provider_config.get("base_url", "http://localhost:11434")
            
            logger.info(f"Using Ollama model: {ollama_model} at {base_url}")
            return ChatOllama(
                model=ollama_model,
                temperature=temperature,
                base_url=base_url
            )
        
        else:
            logger.error(f"Unsupported LLM provider: {provider}")
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

