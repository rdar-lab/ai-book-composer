"""LLM provider abstraction."""

from typing import Optional
from pathlib import Path
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama, ChatLlamaCpp

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
        
        elif provider == "ollama_embedded":
            provider_config = settings.get_provider_config("ollama_embedded")
            model_path = provider_config.get("model_path", "models/llama-3.2-1b-instruct.gguf")
            n_ctx = provider_config.get("n_ctx", 2048)
            n_threads = provider_config.get("n_threads", 4)
            n_gpu_layers = provider_config.get("n_gpu_layers", 0)
            verbose = provider_config.get("verbose", False)
            
            # Resolve model path
            model_path_obj = Path(model_path)
            if not model_path_obj.is_absolute():
                # Try relative to current directory first
                if not model_path_obj.exists():
                    # Try relative to package directory
                    package_dir = Path(__file__).parent.parent.parent
                    model_path_obj = package_dir / model_path
            
            # Verify model file exists
            if not model_path_obj.exists():
                logger.error(f"Model file not found: {model_path_obj}")
                raise FileNotFoundError(
                    f"Embedded Ollama model file not found: {model_path_obj}\n"
                    f"Please download a GGUF model file and update the model_path in your configuration.\n"
                    f"You can find GGUF models at: https://huggingface.co/models?search=gguf"
                )
            
            model_path_str = str(model_path_obj)
            
            logger.info(f"Using embedded Ollama model: {model_path_str}")
            logger.info(f"  Context size: {n_ctx}, Threads: {n_threads}, GPU layers: {n_gpu_layers}")
            
            return ChatLlamaCpp(
                model_path=model_path_str,
                temperature=temperature,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose
            )
        
        else:
            logger.error(f"Unsupported LLM provider: {provider}")
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

