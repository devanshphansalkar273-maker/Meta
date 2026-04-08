"""Base agent class for OpenEnv."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str = "BaseAgent"
    temperature: float = 0.0
    max_tokens: int = 200
    timeout: float = 10.0


class BaseAgent(ABC):
    """
    Abstract base class for OpenEnv agents.

    Subclasses must implement the `predict` method.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the agent.

        Args:
            config: Agent configuration
        """
        self.config = config or AgentConfig()

    @abstractmethod
    def predict(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Given an observation, return an action.

        Args:
            observation: Environment observation dict

        Returns:
            Action dict to send to the environment
        """
        pass

    def reset(self):
        """Reset agent state between episodes."""
        pass

    def train(self, episode_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Train the agent on episode data.

        Args:
            episode_data: List of (observation, action, reward, done, info) tuples

        Returns:
            Dict of training metrics
        """
        # Default implementation - override for actual training
        return {"status": "no-op"}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name})"