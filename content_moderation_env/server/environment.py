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
    
    def create_instance(self, instance_id: str, task: str = "easy") -> ContentModerationEnv:
        """Create a new environment instance."""
        if len(self.instances) >= self.max_instances:
            raise RuntimeError(f"Maximum concurrent environments ({self.max_instances}) reached")

        env = ContentModerationEnv()
        env.reset(task=task)
        self.instances[instance_id] = env
        self.instance_states[instance_id] = env.state()
        logger.info(f"Created environment instance: {instance_id} (task={task})")
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

    def step_instance(self, instance_id: str, action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Step an environment instance with an action."""
        if instance_id not in self.instances:
            return None
        env = self.instances[instance_id]
        # Import here to avoid circular dependency at module level
        from models import ModerationAction, Decision, ContentCategory
        mod_action = ModerationAction(
            decision=Decision(action.get("decision", "ALLOW")),
            content_category=ContentCategory(action.get("content_category", "SAFE")),
            reasoning=action.get("reasoning", ""),
            confidence_score=action.get("confidence_score", 0.5)
        )
        obs, reward, done, info = env.step(mod_action)
        self.instance_states[instance_id] = env.state()
        return {
            "observation": {
                "post_id": obs.post_id,
                "post_body": obs.post_body,
                "metadata": {
                    "user_id": obs.metadata.user_id,
                    "timestamp": obs.metadata.timestamp,
                    "reports_count": obs.metadata.reports_count,
                    "author_trust_score": obs.metadata.author_trust_score,
                    "account_age_days": obs.metadata.account_age_days,
                    "virality_score": obs.metadata.virality_score,
                    "active_global_event": obs.metadata.active_global_event,
                    "temporary_rule": obs.metadata.temporary_rule,
                    "user_appeal_statement": obs.metadata.user_appeal_statement,
                    "media_vision_tags": obs.metadata.media_vision_tags,
                    "visual_text_mismatch_flag": obs.metadata.visual_text_mismatch_flag,
                },
                "context": obs.context,
            },
            "reward": reward,
            "done": done,
            "info": info,
            "state": self.instance_states[instance_id],
        }

    def list_instances(self) -> Dict[str, Dict[str, Any]]:
        """List all active instances and their states."""
        return {
            instance_id: self.get_state(instance_id)
            for instance_id in self.instances.keys()
        }


# Global environment manager
env_manager = EnvironmentManager(max_instances=10)
