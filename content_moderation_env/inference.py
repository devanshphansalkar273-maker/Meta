#!/usr/bin/env python3
"""Baseline inference script for Content Moderation OpenEnv.
Strictly follows OpenEnv RL Challenge spec.
Root directory: inference.py
Reads: API_BASE_URL (default), MODEL_NAME (default), HF_TOKEN (required)
Uses: Hugging Face router via OpenAI-compatible client
Output: EXACT [START]/[STEP]/[END] format
Runs: All 3 tasks (easy, medium, hard)"""
import os
import sys
import json
import logging
from typing import List, Tuple
from openai import OpenAI

try:
    from .env import ContentModerationEnv
    from .models import ModerationObservation, ModerationAction, Decision, ContentCategory
    from .prompts import STRUCTURED_MODERATION_PROMPT
    from .openai_client import create_openai_client
except ImportError:
    from env import ContentModerationEnv
    from models import ModerationObservation, ModerationAction, Decision, ContentCategory
    from prompts import STRUCTURED_MODERATION_PROMPT
    from openai_client import create_openai_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"


def parse_action_response(content: str) -> ModerationAction:
    """Parse API response to ModerationAction. Robust fallback."""
    try:
        data = json.loads(content)
        decision_str = data.get("decision", "ESCALATE").upper()
        category_str = data.get("category", "SAFE").upper()
        confidence = max(0.0, min(1.0, data.get("confidence", 0.5)))
        reasoning = data.get("reasoning", "API decision")
    except Exception:
        decision_str = "ESCALATE"
        for d in ["REMOVE", "FLAG", "ALLOW", "ESCALATE"]:
            if d.lower() in content.lower():
                decision_str = d
                break
        category_str = "SAFE"
        confidence = 0.5
        reasoning = "Parsed fallback"

    return ModerationAction(
        decision=Decision(decision_str),
        content_category=ContentCategory(category_str),
        reasoning=reasoning,
        confidence_score=confidence,
    )


class ModerationInferenceEngine:
    """Inference engine wrapper used by the API server and training scripts."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("HF_TOKEN")
        self.base_url = base_url or os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
        self.model_name = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)

        if not self.api_key:
            raise EnvironmentError("HF_TOKEN environment variable is required")

        self.client, self.is_available = create_openai_client(
            api_key=self.api_key,
            base_url=self.base_url,
            use_fallback=False,
        )

        if not self.is_available or self.client is None:
            raise RuntimeError("Failed to initialize Hugging Face client")

    def build_prompt(self, obs: ModerationObservation) -> str:
        return (
            f"{STRUCTURED_MODERATION_PROMPT}\n\n"
            f"Post: {obs.post_body}\n"
            f"Metadata: {obs.metadata.model_dump_json()}\n"
            f"Context: {obs.context}"
        )

    def get_moderation_decision(self, obs: ModerationObservation) -> ModerationAction:
        prompt = self.build_prompt(obs)
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": STRUCTURED_MODERATION_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=256,
        )
        content = (resp.choices[0].message.content or "").strip()
        return parse_action_response(content)


def run_single_task(task_name: str, engine: ModerationInferenceEngine, rewards: List[float], step_num: List[int]) -> Tuple[bool, int]:
    """Run inference on single task, append to shared logs."""
    try:
        env = ContentModerationEnv()
        obs = env.reset(task=task_name)

        print(f"[START] task={task_name} env=content_moderation model={engine.model_name}")

        step_count = 0
        total_reward = 0.0
        task_rewards = []

        while not env.done and step_count < 50:
            action = engine.get_moderation_decision(obs)
            next_obs, reward, done, info = env.step(action)

            error = info.get("error", None) or "null"
            task_rewards.append(reward)
            rewards.extend(task_rewards)
            step_num[0] += 1

            print(f"[STEP]  step={step_count+1} action={action.decision.value} reward={reward:.2f} done={done} error={error}")

            total_reward += reward
            step_count += 1
            obs = next_obs
            if done:
                break

        avg_reward = total_reward / max(step_count, 1)
        success = avg_reward >= 0.5
        print(f"[END]   success={success} steps={step_count} rewards={','.join(f'{r:.2f}' for r in task_rewards)}")
        logger.info(f"{task_name}: success={success}, avg_reward={avg_reward:.4f}, steps={step_count}")
        return success, step_count

    except Exception as e:
        error_msg = str(e).replace("\n", " ")[:100]
        print(f"[END]   success=false steps=0 rewards= error={error_msg}")
        logger.error(f"{task_name} failed: {e}")
        return False, 0


if __name__ == "__main__":
    try:
        engine = ModerationInferenceEngine()
    except Exception as exc:
        print(f"[END]   success=false steps=0 rewards= error={str(exc)[:100]}", file=sys.stderr)
        sys.exit(1)

    all_rewards = []
    total_steps = [0]

    for task in ["easy", "medium", "hard"]:
        run_single_task(task, engine, all_rewards, total_steps)

    logger.info("All tasks complete.")
    sys.exit(0)
