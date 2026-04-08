"""OpenEnv client for connecting to OpenEnv servers."""

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class OpenEnvClient:
    """
    Client for connecting to OpenEnv servers.

    Provides a Gymnasium-style API (reset, step, state) for interacting
    with OpenEnv RL environments over HTTP.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        api_key: Optional[str] = None,
        instance_id: Optional[str] = None,
    ):
        """
        Initialize the OpenEnv client.

        Args:
            base_url: Base URL of the OpenEnv server (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication
            instance_id: Optional environment instance ID. Auto-generated if not provided.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        if api_key:
            self._client.headers["Authorization"] = f"Bearer {api_key}"

        # Determine API style by probing the server
        self._api_style = self._detect_api_style()
        logger.info(f"Detected API style: {self._api_style}")
        self._instance_id = instance_id or f"agent-{uuid.uuid4().hex[:8]}"

        if self._api_style == "envmanager":
            self._env_create(self._instance_id)

    def _detect_api_style(self) -> str:
        """Detect whether server uses /reset+step (gym-style) or /inference (stateless)."""
        try:
            resp = self._client.get(f"{self.base_url}/openapi.json")
            if resp.status_code == 200:
                paths = resp.json().get("paths", {})
                # Prefer envmanager detection first (most common for OpenEnv servers)
                if "/environment/create" in paths:
                    return "envmanager"
                if "/reset" in paths:
                    return "gym"
                if "/inference" in paths:
                    return "inference"
        except Exception:
            pass

        # Default to envmanager (most common for OpenEnv servers)
        try:
            resp = self._client.post(f"{self.base_url}/environment/create", json={"instance_id": "probe"})
            if resp.status_code in (200, 422):
                return "envmanager"
        except Exception:
            pass

        return "gym"

    def _env_create(self, instance_id: str):
        """Create an environment instance on the server."""
        try:
            self._client.post(f"{self.base_url}/environment/create", json={"instance_id": instance_id})
        except httpx.HTTPError as e:
            logger.warning(f"Failed to create environment instance: {e}")

    def reset(self) -> dict:
        """
        Reset the environment and return the initial observation.

        Returns:
            dict: Initial observation from the environment
        """
        if self._api_style == "gym":
            try:
                response = self._client.post(f"{self.base_url}/reset")
                response.raise_for_status()
                data = response.json()
                return data.get("observation", data)
            except httpx.HTTPError as e:
                logger.error(f"Failed to reset environment: {e}")
                raise

        elif self._api_style == "envmanager":
            state = self._get_env_state()
            return state

        else:  # inference style - stateless, return empty obs
            return {}

    def _get_env_state(self) -> dict:
        """Get environment state (envmanager style)."""
        try:
            response = self._client.get(f"{self.base_url}/environment/{self._instance_id}/state")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to get environment state: {e}")
            raise

    def step(self, action: Any) -> Tuple[dict, float, bool, dict]:
        """
        Execute an action in the environment.

        Args:
            action: Action to take (dict or object with as_dict method)

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if hasattr(action, "as_dict"):
            action_dict = action.as_dict()
        elif isinstance(action, dict):
            action_dict = action
        else:
            raise ValueError(f"Action must be dict or have as_dict() method, got {type(action)}")

        if self._api_style == "gym":
            try:
                response = self._client.post(
                    f"{self.base_url}/step",
                    json={"action": action_dict},
                )
                response.raise_for_status()
                data = response.json()
                return (
                    data.get("observation", {}),
                    data.get("reward", 0.0),
                    data.get("done", False),
                    data.get("info", {}),
                )
            except httpx.HTTPError as e:
                logger.error(f"Failed to step environment: {e}")
                raise

        elif self._api_style == "envmanager":
            try:
                response = self._client.post(
                    f"{self.base_url}/environment/{self._instance_id}/step",
                    json=action_dict,
                )
                response.raise_for_status()
                data = response.json()
                return (
                    data.get("observation", {}),
                    data.get("reward", 0.0),
                    data.get("done", False),
                    data.get("info", {}),
                )
            except httpx.HTTPError as e:
                logger.error(f"Failed to step environment: {e}")
                raise

        else:  # inference style
            # For inference-only servers, run inference and return
            try:
                response = self._client.post(
                    f"{self.base_url}/inference",
                    json=action_dict,
                )
                response.raise_for_status()
                data = response.json()
                return data, 0.0, True, {}
            except httpx.HTTPError as e:
                logger.error(f"Failed to run inference: {e}")
                raise

    def state(self) -> dict:
        """
        Get the current state of the environment without taking an action.

        Returns:
            dict: Current environment state
        """
        if self._api_style == "envmanager":
            return self._get_env_state()

        try:
            response = self._client.get(f"{self.base_url}/state")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to get state: {e}")
            raise

    def health(self) -> bool:
        """
        Check if the environment server is healthy.

        Returns:
            bool: True if server is healthy, False otherwise
        """
        try:
            response = self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class AsyncOpenEnvClient:
    """
    Async client for connecting to OpenEnv servers.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        if api_key:
            self._client.headers["Authorization"] = f"Bearer {api_key}"

    async def reset(self) -> dict:
        try:
            response = await self._client.post(f"{self.base_url}/reset")
            response.raise_for_status()
            data = response.json()
            return data.get("observation", data)
        except httpx.HTTPError as e:
            logger.error(f"Failed to reset environment: {e}")
            raise

    async def step(self, action: Any) -> Tuple[dict, float, bool, dict]:
        if hasattr(action, "as_dict"):
            action_dict = action.as_dict()
        elif isinstance(action, dict):
            action_dict = action
        else:
            raise ValueError(f"Action must be dict or have as_dict() method, got {type(action)}")

        try:
            response = await self._client.post(
                f"{self.base_url}/step",
                json={"action": action_dict},
            )
            response.raise_for_status()
            data = response.json()
            return (
                data.get("observation", {}),
                data.get("reward", 0.0),
                data.get("done", False),
                data.get("info", {}),
            )
        except httpx.HTTPError as e:
            logger.error(f"Failed to step environment: {e}")
            raise

    async def state(self) -> dict:
        try:
            response = await self._client.get(f"{self.base_url}/state")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to get state: {e}")
            raise

    async def aclose(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()
        return False