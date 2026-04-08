#!/usr/bin/env python3
"""
Production-ready inference script for Content Moderation OpenEnv.
Uses OpenRouter API with GPT-4.1 for moderation decisions.
"""

import os
import sys
import json
import logging
from typing import Optional
from openai import OpenAI, APIError, RateLimitError

try:
    from .env import ContentModerationEnv
    from .models import ModerationAction, Decision, ContentCategory
    from .prompts import get_moderation_system_prompt
    from .openai_client import create_openai_client, create_nvidia_client
except ImportError:
    from env import ContentModerationEnv
    from models import ModerationAction, Decision, ContentCategory
    from prompts import get_moderation_system_prompt
    from openai_client import create_openai_client, create_nvidia_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModerationInferenceEngine:
    """OpenRouter-based content moderation inference engine with fallback logic."""
    
    def __init__(self):
        """Initialize inference engine preferring NVIDIA client, fallback to OpenRouter."""
        
        # Try NVIDIA first
        self.client, self.use_api = create_nvidia_client()
        self.client_type = "nvidia"
        
        if not self.use_api:
            # Fallback to OpenRouter
            self.client, self.use_api = create_openai_client()
            self.client_type = "openrouter"
            if not self.use_api:
                logger.warning("No API available (NVIDIA or OpenRouter). Using fallback only.")
        
        if self.use_api:
            logger.info(f"Using {self.client_type} client")
        
    
    def get_moderation_decision(self, obs) -> ModerationAction:
        """
        Get moderation decision from GPT-4.1 via OpenRouter with fallback logic.
        
        Args:
            obs: ModerationObservation object
            
        Returns:
            ModerationAction with decision, category, reasoning, and confidence
        """
        if self.use_api:
            try:
                return self._call_gpt4_api(obs)
            except (APIError, RateLimitError, Exception) as e:
                logger.warning(f"API call failed: {e}. Using fallback logic.")
                return self._fallback_decision(obs)
        else:
            return self._fallback_decision(obs)
    
    def _call_gpt4_api(self, obs) -> ModerationAction:
        """Call model API with strict moderation prompt."""

        system_prompt = get_moderation_system_prompt("strict")

        
        user_prompt = f"""Post: {obs.post_body}
Author Trust: {obs.metadata.author_trust_score}
Account Age: {obs.metadata.account_age_days} days
Reports: {obs.metadata.reports_count}
Virality: {obs.metadata.virality_score}
Context: {', '.join(obs.context) if obs.context else 'None'}

Decide: allow, flag, remove, or escalate?"""
        
        try:
            from openai import OpenAI
            client = OpenAI(
              base_url="https://integrate.api.nvidia.com/v1",
              api_key="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"
            )

            completion = client.chat.completions.create(
              model="meta/llama-3.3-70b-instruct",
              messages=[
                  {"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}
              ],
              temperature=0.2,
              top_p=0.7,
              max_tokens=1024,
              stream=True
            )

            content = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    chunk_text = chunk.choices[0].delta.content
                    print(chunk_text, end="")
                    content += chunk_text
            print() # Print newline after stream

            if not content:
                content = "ESCALATE"

            content_str = str(content).strip().upper()
            
            decision_found = None
            import re
            
            # First extract JSON decision using regex
            json_match = re.search(r'{"DECISION":\s*"([A-Z]+)"}', content_str)
            if json_match:
                decision_found = json_match.group(1)
            else:
                # Then fallback to keyword detection
                valid_decisions = ["ALLOW", "FLAG", "REMOVE", "ESCALATE"]
                # Exact single word or substring detection
                for word in valid_decisions:
                    if word in content_str:
                        decision_found = word
                        break
            
            # Final fallback -> ESCALATE
            if not decision_found:
                decision_found = "ESCALATE"
            
            # Map to enum values
            decision_map = {
                "ALLOW": ("ALLOW", "SAFE", "API decision: Allow"),
                "FLAG": ("FLAG", "SAFE", "API decision: Flag for review"),
                "REMOVE": ("REMOVE", "SPAM", "API decision: Remove"),
                "ESCALATE": ("ESCALATE", "SAFE", "API decision: Escalate")
            }
            
            # Prevent KeyErrors on unexpected decisions by defaulting to ESCALATE
            decision_tuple = decision_map.get(decision_found, decision_map["ESCALATE"])
            decision_str, category_str, reasoning = decision_tuple
            
            return ModerationAction(
                decision=Decision(decision_str),
                content_category=ContentCategory(category_str),
                reasoning=reasoning,
                confidence_score=0.8
            )
        except Exception as e:
            logger.error(f"API error: {e}")
            raise
    
    def _fallback_decision(self, obs) -> ModerationAction:
        """Fallback: Always ESCALATE per constraints when API unavailable."""
        return ModerationAction(
            decision=Decision.ESCALATE,
            content_category=ContentCategory.SAFE,
            reasoning="Fallback: ESCALATE (API unavailable)",
            confidence_score=0.5
        )


def extract_action(client: OpenAI, obs) -> ModerationAction:
    """Legacy function - kept for compatibility. Use ModerationInferenceEngine directly."""
    engine = ModerationInferenceEngine()
    return engine.get_moderation_decision(obs)


def run_inference(max_steps: Optional[int] = None):
    """
    Execute inference loop over the content moderation environment.
    
    Logs output in strict format:
    [START]
    [STEP] ...
    [STEP] ...
    [END]
    
    Args:
        max_steps: Optional limit on number of steps (default: all)
    """
    print("[START]")
    
    step_count = 0
    success = False
    try:
        # Initialize environment and inference engine
        env = ContentModerationEnv()
        engine = ModerationInferenceEngine()
        
        # Reset environment to hard task
        obs = env.reset(task="hard")
        logger.info(f"Environment reset to HARD tasks. Feed queue size: {len(env.feed_queue)}")
        
        total_reward = 0.0
        step_count = 0
        max_steps_to_run = max_steps or len(env.feed_queue)
        
        # Inference loop
        while not env.done and step_count < max_steps_to_run:
            try:
                # Get moderation decision
                action = engine.get_moderation_decision(obs)
                
                # Execute step in environment
                next_obs, reward, done, info = env.step(action)
                
                # Log step with required format
                print(f"[STEP] step={step_count+1} action={action.decision.value} reward={reward:.2f} done={done} error=None")
                
                total_reward += reward
                step_count += 1
                obs = next_obs
                env.done = done
                
            except Exception as e:
                logger.error(f"Error during step {step_count}: {e}")
                # Continue on step errors
                step_count += 1
                if step_count < max_steps_to_run:
                    try:
                        obs = env._get_current_observation()
                        env.current_idx += 1
                    except:
                        break
        
        avg_reward = total_reward / max(step_count, 1)
        avg_reward = max(0.0, min(1.0, avg_reward))  # Clamp to [0, 1]

        success = avg_reward >= 0.5
        print(f"[END] success={success} steps={step_count} score={avg_reward:.4f}")
        
        # Log final statistics
        logger.info(f"Inference complete:")
        logger.info(f"  Total steps: {step_count}")
        logger.info(f"  Total reward: {total_reward:.2f}")
        logger.info(f"  Average reward (normalized): {avg_reward:.4f}")
        logger.info(f"  Metrics: {json.dumps(env.metrics, indent=2)}")
        
        return {
            "status": "success",
            "steps": step_count,
            "total_reward": total_reward,
            "average_reward": avg_reward,
            "metrics": env.metrics
        }
        
    except Exception as e:
        logger.error(f"Fatal error during inference: {e}", exc_info=True)
        success = False
        print(f"[END] success={success} steps={step_count} score=0.0000")
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    # Support command-line argument for max steps
    max_steps = None
    if len(sys.argv) > 1:
        try:
            max_steps = int(sys.argv[1])
        except ValueError:
            logger.warning(f"Invalid max_steps argument: {sys.argv[1]}")
    
    result = run_inference(max_steps=max_steps)
    
    # Exit with status code based on result
    sys.exit(0 if result["status"] == "success" else 1)
