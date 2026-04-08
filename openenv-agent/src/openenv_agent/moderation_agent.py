"""Content moderation agent using OpenAI API."""

import os
import logging
from typing import Any, Dict, Optional

from openai import OpenAI, APIError, RateLimitError

from openenv_agent.agent import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)


# Valid decisions and categories from OpenEnv schema
VALID_DECISIONS = ["ALLOW", "FLAG", "REMOVE", "ESCALATE"]
VALID_CATEGORIES = ["SPAM", "HATE_SPEECH", "MISINFORMATION", "HARASSMENT", "SAFE"]

DECISION_MAP = {
    "allow": "ALLOW",
    "flag": "FLAG",
    "remove": "REMOVE",
    "escalate": "ESCALATE",
}

CATEGORY_MAP = {
    "allow": "SAFE",
    "flag": "SAFE",
    "remove": "HARASSMENT",
    "escalate": "SAFE",
}


class ModerationAgent(BaseAgent):
    """
    Content moderation agent using OpenAI API via OpenRouter.

    Connects to OpenEnv content_moderation_env servers and makes
    moderation decisions (ALLOW, FLAG, REMOVE, ESCALATE).
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config or AgentConfig(name="ModerationAgent"))
        self.api_key = os.getenv("HF_TOKEN") or os.getenv("OPENROUTER_API_KEY")
        self.use_api = bool(self.api_key)

        if self.use_api:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            logger.warning("No API key found. Using fallback mode.")
            self.client = None

    def predict(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict a moderation action from an observation.

        Args:
            observation: Observation dict with post_body, metadata, context

        Returns:
            Action dict with decision, content_category, reasoning, confidence_score
        """
        if self.use_api:
            try:
                return self._call_api(observation)
            except (APIError, RateLimitError) as e:
                logger.warning(f"API failed: {e}. Using fallback.")
                return self._fallback_decision(observation)
        else:
            return self._fallback_decision(observation)

    def _call_api(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Call OpenAI API for moderation decision."""
        post_body = obs.get("post_body", "")
        metadata = obs.get("metadata", {})
        context = obs.get("context", [])

        user_prompt = self._build_prompt(post_body, metadata, context)

        response = self.client.chat.completions.create(
            model="openai/gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": self._get_system_prompt()
                },
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout
        )

        content = response.choices[0].message.content.strip().lower()

        # Parse response
        decision_found = None
        for decision in ["allow", "flag", "remove", "escalate"]:
            if decision in content:
                decision_found = decision
                break

        if not decision_found:
            decision_found = "escalate"
            logger.warning(f"Unexpected API response: {content}")

        decision = DECISION_MAP.get(decision_found, "ESCALATE")
        category = CATEGORY_MAP.get(decision_found, "SAFE")

        return {
            "decision": decision,
            "content_category": category,
            "reasoning": f"API decision: {decision_found}",
            "confidence_score": 0.8
        }

    def _get_system_prompt(self) -> str:
        return """You are a content moderation agent. Analyze the post and decide:
- ALLOW: Safe content
- FLAG: Suspicious content needing review
- REMOVE: Harmful content
- ESCALATE: Complex cases requiring human review

Respond with only ONE decision word."""

    def _build_prompt(
        self,
        post_body: str,
        metadata: Dict[str, Any],
        context: list
    ) -> str:
        lines = [
            f"Post: {post_body}",
            f"Author Trust: {metadata.get('author_trust_score', 'N/A')}",
            f"Account Age: {metadata.get('account_age_days', 0)} days",
            f"Reports: {metadata.get('reports_count', 0)}",
            f"Virality: {metadata.get('virality_score', 0)}",
            f"Context: {', '.join(context) if context else 'None'}",
            "",
            "Decide: allow, flag, remove, or escalate?"
        ]
        return "\n".join(lines)

    def _fallback_decision(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keyword-based fallback when API is unavailable.
        """
        post_body = obs.get("post_body", "").lower()
        metadata = obs.get("metadata", {})
        harmful_keywords = {
            "remove": ["kill", "bomb", "explode", "shoot", "murder"],
            "flag": ["hate", "stupid", "dumb", "idiotic", "scum"],
            "spam": ["click here", "earn money", "buy now", "http://", "https://"]
        }

        for keyword in harmful_keywords["remove"]:
            if keyword in post_body:
                return {
                    "decision": "REMOVE",
                    "content_category": "HARASSMENT",
                    "reasoning": f"Fallback: detected '{keyword}'",
                    "confidence_score": 0.6
                }

        for keyword in harmful_keywords["flag"]:
            if keyword in post_body:
                return {
                    "decision": "FLAG",
                    "content_category": "HATE_SPEECH",
                    "reasoning": f"Fallback: detected '{keyword}'",
                    "confidence_score": 0.5
                }

        # Check trust and reports
        trust = metadata.get("author_trust_score", 0.5)
        reports = metadata.get("reports_count", 0)
        if trust < 0.3 and reports > 5:
            return {
                "decision": "FLAG",
                "content_category": "SAFE",
                "reasoning": "Fallback: low-trust user with many reports",
                "confidence_score": 0.4
            }

        return {
            "decision": "ALLOW",
            "content_category": "SAFE",
            "reasoning": "Fallback: no harmful patterns",
            "confidence_score": 0.8
        }