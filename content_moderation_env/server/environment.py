"""Server-side environment wrapper for content moderation."""

import logging
from typing import Dict, Any, Optional
from env import ContentModerationEnv
from models import ModerationObservation, ModerationAction

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Manages environment instances for concurrent requests."""
    
    def __init__(self, max_instances: int = 10):
        self.max_instances = max_instances
        self.instances: Dict[str, ContentModerationEnv] = {}
        self.instance_states: Dict[str, Dict[str, Any]] = {}
    
    def create_instance(self, instance_id: str) -> ContentModerationEnv:
        """Create a new environment instance."""
        if len(self.instances) >= self.max_instances:
            raise RuntimeError(f"Maximum concurrent environments ({self.max_instances}) reached")
        
        env = ContentModerationEnv()
        self.instances[instance_id] = env
        self.instance_states[instance_id] = env.state()
        logger.info(f"Created environment instance: {instance_id}")
        return env
    
    def get_instance(self, instance_id: str) -> Optional[ContentModerationEnv]:
        """Retrieve existing environment instance."""
        return self.instances.get(instance_id)
    
    def delete_instance(self, instance_id: str) -> bool:
        """Delete an environment instance."""
        if instance_id in self.instances:
            del self.instances[instance_id]
            if instance_id in self.instance_states:
                del self.instance_states[instance_id]
            logger.info(f"Deleted environment instance: {instance_id}")
            return True
        return False
    
    def get_state(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get state of an environment instance."""
        if instance_id in self.instances:
            self.instance_states[instance_id] = self.instances[instance_id].state()
            return self.instance_states[instance_id]
        return None
    
    def list_instances(self) -> Dict[str, Dict[str, Any]]:
        """List all active instances and their states."""
        return {
            instance_id: self.get_state(instance_id)
            for instance_id in self.instances.keys()
        }


# Global environment manager
env_manager = EnvironmentManager(max_instances=10)
