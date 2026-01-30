"""LLM provider abstraction."""
import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download
from langchain_community.chat_models import ChatOllama, ChatLlamaCpp
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from .config import Settings

logger = logging.getLogger(__name__)


def get_llm(
        settings: Settings,
        temperature: float = 0.7,
        model: Optional[str] = None,
        provider: Optional[str] = None,

) -> BaseChatModel:
    """Get LLM instance based on configuration.
    
    Args:
        settings: The project settings
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
                azure_deployment=provider_config.get("deployment", ""),
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
            model_name = provider_config.get("model_name", "llama-3.2-3b-instruct")
            n_ctx = provider_config.get("n_ctx", 2048)
            n_threads = provider_config.get("n_threads", 4)
            run_on_gpu = provider_config.get("run_on_gpu", False)
            verbose = provider_config.get("verbose", False)

            # Convert run_on_gpu boolean to n_gpu_layers
            # If GPU enabled, use a high number to offload all layers
            # Otherwise use 0 for CPU-only
            n_gpu_layers = -1 if run_on_gpu else 0

            # Map model names to HuggingFace repo IDs and filenames
            model_mappings = {
                "llama-3.2-1b-instruct": {
                    "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
                    "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
                },
                "llama-3.2-3b-instruct": {
                    "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
                    "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
                },
                "llama-3.1-8b-instruct": {
                    "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                    "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
                }
            }

            # Get model info
            if model_name not in model_mappings:
                logger.error(f"Unknown model: {model_name}")
                raise ValueError(
                    f"Unknown embedded model: {model_name}\n"
                    f"Supported models: {', '.join(model_mappings.keys())}\n"
                    f"To add a custom model, use 'model_path' instead of 'model_name' in config."
                )

            model_info = model_mappings[model_name]
            repo_id = model_info["repo_id"]
            filename = model_info["filename"]

            # Download model from HuggingFace Hub
            logger.info(f"Downloading model {model_name} from HuggingFace...")
            logger.info(f"  Repository: {repo_id}")
            logger.info(f"  File: {filename}")

            try:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=Path.home() / ".cache" / "ai-book-composer" / "models"
                )
                logger.info(f"Model downloaded to: {model_path}")
            except Exception as e:
                logger.exception(f"Failed to download model: {e}")
                raise RuntimeError(
                    f"Failed to download model {model_name} from HuggingFace.\n"
                    f"Error: {e}\n"
                    f"Please check your internet connection and try again."
                )

            logger.info(f"Initializing embedded Ollama model: {model_name}")
            logger.info(f"  Context size: {n_ctx}, Threads: {n_threads}, GPU: {run_on_gpu}")

            return ChatLlamaCpp(
                model_path=model_path,
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
        logger.exception(f"Failed to initialize LLM: {e}")
        raise
