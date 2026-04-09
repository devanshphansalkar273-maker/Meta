"""OpenAI-compatible client factory for Hugging Face Router and fallback providers."""

import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)


def create_openai_client(
    api_key: str = None,
    base_url: str = None,
    use_fallback: bool = True
) -> tuple[OpenAI | None, bool]:
    """
    Create an OpenAI-compatible client configured for Hugging Face Router by default.

    Args:
        api_key: API key (default: HF_TOKEN or OPENROUTER_API_KEY env var)
        base_url: API base URL (default: https://router.huggingface.co/v1)
        use_fallback: If True, returns (None, False) on missing credentials instead of raising

    Returns:
        Tuple of (client, is_api_available)
        - client: OpenAI client instance or None
        - is_api_available: True if client is ready, False if fallback needed
    """
    # Get credentials from arguments or environment
    api_key = api_key or os.getenv('HF_TOKEN') or os.getenv('OPENROUTER_API_KEY')
    base_url = base_url or os.getenv('API_BASE_URL') or "https://router.huggingface.co/v1"
    
    # Check if we have API credentials
    if not api_key:
        if use_fallback:
            logger.warning("No API key found. Using fallback mode.")
            return None, False
        else:
            raise EnvironmentError(
                "Missing API credentials. Set HF_TOKEN (preferred for Hugging Face), OPENROUTER_API_KEY, or provide api_key argument."
            )
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        logger.info(f"OpenAI-compatible client initialized with base_url: {base_url}")
        return client, True
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        if use_fallback:
            return None, False
        raise


def get_openai_client() -> OpenAI:
    """
    Get OpenAI client (fails if not available).
    
    Returns:
        OpenAI client instance
        
    Raises:
        EnvironmentError: If API credentials not found or client initialization failed
        
    Examples:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="openai/gpt-4.1",
            messages=[...],
            temperature=0,
            max_tokens=200
        )
    """
    client, is_available = create_openai_client(use_fallback=False)
    return client


def create_nvidia_client(
    api_key: str = None,
    use_fallback: bool = True
) -> tuple[OpenAI | None, bool]:
    """
    Create OpenAI-compatible client for NVIDIA API (gpt-oss-120b reasoning model).
    
    Args:
        api_key: NVIDIA API key (default: NVIDIA_API_KEY env var)
        use_fallback: Return (None, False) on failure
        
    Returns:
        Tuple (client, available). Supports reasoning_content and streaming.
    """
    api_key = api_key or os.getenv('NVIDIA_API_KEY')
    if not api_key:
        if use_fallback:
            logger.warning("No NVIDIA_API_KEY found. Fallback to standard OpenAI client.")
            return None, False
        raise EnvironmentError("Missing NVIDIA_API_KEY")
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )
        logger.info("NVIDIA client initialized")
        return client, True
    except Exception as e:
        logger.error(f"NVIDIA client init failed: {e}")
        if use_fallback:
            return None, False
        raise


if __name__ == "__main__":

    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Try to create client
    client, api_available = create_openai_client()
    
    if api_available:
        print("✓ OpenAI-compatible client ready")
        print("  Base URL: https://router.huggingface.co/v1")
        print("  Model: meta-llama/Llama-3.3-70B-Instruct")
    else:
        print("✗ Using fallback mode (no API key)")
