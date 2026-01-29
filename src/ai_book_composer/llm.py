"""LLM provider abstraction."""

from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama

from .config import settings


def get_llm(
    temperature: float = 0.7,
    model: Optional[str] = None
) -> BaseChatModel:
    """Get LLM instance based on configuration.
    
    Args:
        temperature: Temperature for generation (0.0 to 1.0)
        model: Optional model override
        
    Returns:
        Configured LLM instance
    """
    provider = settings.llm_provider
    model_name = model or settings.llm_model
    
    if provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=settings.openai_api_key
        )
    
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=settings.google_api_key
        )
    
    elif provider == "azure":
        return AzureChatOpenAI(
            deployment_name=settings.azure_openai_deployment,
            temperature=temperature,
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint
        )
    
    elif provider == "ollama":
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=settings.ollama_base_url
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
