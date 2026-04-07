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

from env import ContentModerationEnv
from models import ModerationAction, Decision, ContentCategory
from prompts import get_moderation_system_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModerationInferenceEngine:
    """OpenRouter-based content moderation inference engine with fallback logic."""
    
    def __init__(self):
        """Initialize the inference engine with OpenRouter API credentials."""
        self.api_key = os.getenv('HF_TOKEN') or os.getenv('OPENROUTER_API_KEY')
        
        if not self.api_key:
            logger.warning("No API key found (HF_TOKEN or OPENROUTER_API_KEY). Using fallback logic only.")
            self.use_api = False
        else:
            self.use_api = True
        
        if self.use_api:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )
    
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
        """Call OSS model via OpenRouter API with strict moderation prompt."""
        system_prompt = get_moderation_system_prompt("strict")
        
        user_prompt = f"""Post: {obs.post_body}
Author Trust: {obs.metadata.author_trust_score}
Account Age: {obs.metadata.account_age_days} days
Reports: {obs.metadata.reports_count}
Virality: {obs.metadata.virality_score}
Context: {', '.join(obs.context) if obs.context else 'None'}

Decide: allow, flag, remove, or escalate?"""
        
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=5,
                timeout=10
            )
            
            # Validate and clean response
            content = response.choices[0].message.content.strip().lower()
            valid_decisions = ["allow", "flag", "remove", "escalate"]
            
            # Parse simple one-word response
            decision_found = None
            for valid_decision in valid_decisions:
                if valid_decision in content:
                    decision_found = valid_decision
                    break
            
            if not decision_found:
                # Fallback if unexpected response
                logger.warning(f"Unexpected API response: {content}")
                decision_found = "escalate"
            
            # Map to enum values
            decision_map = {
                "allow": ("ALLOW", "SAFE", "API decision: Allow"),
                "flag": ("FLAG", "SAFE", "API decision: Flag for review"),
                "remove": ("REMOVE", "HARASSMENT", "API decision: Remove"),
                "escalate": ("ESCALATE", "SAFE", "API decision: Escalate")
            }
            
            decision_str, category_str, reasoning = decision_map[decision_found]
            
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
        """
        Fallback keyword-based moderation when API is unavailable.
        Uses rule-based heuristics for fast, reliable decisions.
        """
        post_lower = obs.post_body.lower()
        harmful_keywords = {
            'remove': ['kill', 'bomb', 'explode', 'shoot', 'murder'],
            'flag': ['hate', 'stupid', 'dumb', 'idiotic', 'scum'],
            'spam': ['click here', 'earn money', 'buy now', 'http://', 'https://']
        }
        
        # Check for harmful keywords
        for keyword in harmful_keywords['remove']:
            if keyword in post_lower:
                return ModerationAction(
                    decision=Decision.REMOVE,
                    content_category=ContentCategory.HARASSMENT,
                    reasoning=f"Fallback: Detected harmful keyword '{keyword}'",
                    confidence_score=0.6
                )
        
        for keyword in harmful_keywords['flag']:
            if keyword in post_lower:
                return ModerationAction(
                    decision=Decision.FLAG,
                    content_category=ContentCategory.HATE_SPEECH,
                    reasoning=f"Fallback: Detected potentially offensive keyword '{keyword}'",
                    confidence_score=0.5
                )
        
        # Check for spam patterns
        spam_score = 0
        for keyword in harmful_keywords['spam']:
            if keyword in post_lower:
                spam_score += 1
        
        if spam_score >= 2 or len(post_lower) > 200 and post_lower.count('http') > 2:
            return ModerationAction(
                decision=Decision.REMOVE,
                content_category=ContentCategory.SPAM,
                reasoning="Fallback: Detected spam patterns",
                confidence_score=0.5
            )
        
        # Check user trust and reports
        if obs.metadata.author_trust_score < 0.3 and obs.metadata.reports_count > 5:
            return ModerationAction(
                decision=Decision.FLAG,
                content_category=ContentCategory.SAFE,
                reasoning="Fallback: Low-trust user with multiple reports",
                confidence_score=0.4
            )
        
        # Default to allow
        return ModerationAction(
            decision=Decision.ALLOW,
            content_category=ContentCategory.SAFE,
            reasoning="Fallback: No harmful patterns detected",
            confidence_score=0.8
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
    
    try:
        # Initialize environment and inference engine
        env = ContentModerationEnv()
        engine = ModerationInferenceEngine()
        
        # Reset environment
        obs = env.reset()
        logger.info(f"Environment reset. Feed queue size: {len(env.feed_queue)}")
        
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
