#!/usr/bin/env python3
"""
Client SDK for Content Moderation OpenEnv API.
Provides Python API for interacting with the moderation environment.
"""

import requests
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModerationResult:
    """Result of a moderation inference."""
    post_id: str
    decision: str
    content_category: str
    reasoning: str
    confidence_score: float


@dataclass
class EnvironmentState:
    """State of an environment instance."""
    instance_id: str
    current_step: int
    cumulative_reward: float
    done: bool
    metrics: Dict[str, Any]


class ContentModerationClient:
    """Python client for Content Moderation OpenEnv API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API server (default: localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """
        Check if the API server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def moderate(
        self,
        post_id: str,
        post_body: str,
        author_trust_score: float = 0.5,
        account_age_days: int = 0,
        reports_count: int = 0,
        virality_score: float = 0.0,
        context: Optional[List[str]] = None
    ) -> ModerationResult:
        """
        Run inference on a single post.
        
        Args:
            post_id: Unique post identifier
            post_body: Post text content
            author_trust_score: Trust score (0.0-1.0)
            account_age_days: Age of author's account in days
            reports_count: Number of user reports
            virality_score: Virality/reach score (0.0-1.0)
            context: Optional historical context
            
        Returns:
            ModerationResult with decision and confidence
            
        Raises:
            requests.RequestException: If API call fails
        """
        payload = {
            "post_id": post_id,
            "post_body": post_body,
            "author_trust_score": author_trust_score,
            "account_age_days": account_age_days,
            "reports_count": reports_count,
            "virality_score": virality_score,
            "context": context or []
        }
        
        response = self.session.post(f"{self.base_url}/inference", json=payload)
        response.raise_for_status()
        
        data = response.json()
        return ModerationResult(
            post_id=data["post_id"],
            decision=data["decision"],
            content_category=data["content_category"],
            reasoning=data["reasoning"],
            confidence_score=data["confidence_score"]
        )
    
    def create_environment(self, instance_id: str) -> EnvironmentState:
        """
        Create a new environment instance.
        
        Args:
            instance_id: Unique identifier for the environment
            
        Returns:
            EnvironmentState with initial state
        """
        response = self.session.post(
            f"{self.base_url}/environment/create",
            json={"instance_id": instance_id}
        )
        response.raise_for_status()
        
        data = response.json()
        return EnvironmentState(
            instance_id=data["instance_id"],
            current_step=data["current_step"],
            cumulative_reward=data["cumulative_reward"],
            done=data["done"],
            metrics=data["metrics"]
        )
    
    def get_environment_state(self, instance_id: str) -> EnvironmentState:
        """
        Get state of an environment instance.
        
        Args:
            instance_id: Environment instance ID
            
        Returns:
            Current EnvironmentState
        """
        response = self.session.get(f"{self.base_url}/environment/{instance_id}/state")
        response.raise_for_status()
        
        data = response.json()
        return EnvironmentState(
            instance_id=data["instance_id"],
            current_step=data["current_step"],
            cumulative_reward=data["cumulative_reward"],
            done=data["done"],
            metrics=data["metrics"]
        )
    
    def list_environments(self) -> Dict[str, EnvironmentState]:
        """
        List all active environment instances.
        
        Returns:
            Dictionary mapping instance IDs to their states
        """
        response = self.session.get(f"{self.base_url}/environment/list")
        response.raise_for_status()
        
        data = response.json()
        result = {}
        for instance_id, state_data in data.items():
            result[instance_id] = EnvironmentState(
                instance_id=state_data["instance_id"],
                current_step=state_data["current_step"],
                cumulative_reward=state_data["cumulative_reward"],
                done=state_data["done"],
                metrics=state_data["metrics"]
            )
        return result
    
    def delete_environment(self, instance_id: str) -> bool:
        """
        Delete an environment instance.
        
        Args:
            instance_id: Environment instance ID
            
        Returns:
            True if successfully deleted
        """
        response = self.session.delete(f"{self.base_url}/environment/{instance_id}")
        return response.status_code == 200
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get detailed API server status.
        
        Returns:
            Status information including configuration and active environments
        """
        response = self.session.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    client = ContentModerationClient()
    
    # Check health
    if client.health_check():
        print("✓ API server is healthy")
        
        # Example moderation call
        result = client.moderate(
            post_id="example_001",
            post_body="This is a test post",
            author_trust_score=0.7
        )
        print(f"Decision: {result.decision} (confidence: {result.confidence_score:.2f})")
    else:
        print("✗ API server is not responding")
