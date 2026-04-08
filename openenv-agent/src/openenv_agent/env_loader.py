"""Utility for loading OpenEnv environments from YAML config."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from openenv_agent.client import OpenEnvClient

logger = logging.getLogger(__name__)


def load_env_from_yaml(
    yaml_path: str | Path,
    **client_kwargs
) -> Dict[str, Any]:
    """
    Load an OpenEnv environment configuration from a YAML file.

    Args:
        yaml_path: Path to openenv.yaml
        **client_kwargs: Additional arguments for OpenEnvClient

    Returns:
        Dict with keys:
            - client: OpenEnvClient instance
            - config: Parsed environment config
            - schemas: Observation and action schemas
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    env_config = config.get("environment", {})
    api_config = config.get("api", {})
    environment_config = config.get("environment_config", {})
    schemas = config.get("schemas", {})

    # Build base URL
    port = api_config.get("port", 8000)
    base_url = f"http://localhost:{port}"

    client = OpenEnvClient(base_url=base_url, **client_kwargs)

    return {
        "client": client,
        "config": {
            "name": env_config.get("name", "unknown"),
            "version": env_config.get("version", "unknown"),
            "description": env_config.get("description", ""),
        },
        "environment_config": environment_config,
        "schemas": schemas,
        "metadata": config.get("metadata", {}),
    }


def load_env_from_url(
    base_url: str,
    api_key: Optional[str] = None
) -> OpenEnvClient:
    """
    Load an OpenEnv environment from a URL.

    Args:
        base_url: Base URL of the OpenEnv server
        api_key: Optional API key

    Returns:
        OpenEnvClient instance
    """
    return OpenEnvClient(base_url=base_url, api_key=api_key)