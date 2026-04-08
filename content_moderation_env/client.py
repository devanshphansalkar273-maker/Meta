from typing import Any

from openenv import HTTPEnvClient, StepResult
from .models import ModerationAction, ModerationObservation, ModerationState

class ModerationClient(HTTPEnvClient):
    def _step_payload(self, action: ModerationAction) -> dict[str, Any]:
        return {
            "decision": action.action,
            "content_category": "SAFE",  # Default compatible with server
            "reasoning": "",
            "confidence_score": 0.5
        }

    def _parse_result(self, raw_result: dict[str, Any]) -> StepResult[ModerationObservation, ModerationState]:
        return StepResult(
            observation=ModerationObservation(
                post=raw_result["observation"]["post_body"],
                metadata=raw_result["observation"]["metadata"].model_dump() if hasattr(raw_result["observation"]["metadata"], "model_dump") else raw_result["observation"]["metadata"],
                done=raw_result["done"],
                reward=raw_result["reward"]
            ),
            state=ModerationState(
                step_count=raw_result["state"]["step_count"],
                current_index=raw_result["state"]["current_index"],
                task=raw_result["state"]["task"],
                done=raw_result["state"]["done"]
            )
        )

    def _parse_state(self, raw_state: dict[str, Any]) -> ModerationState:
        return ModerationState(
            step_count=raw_state["step_count"],
            current_index=raw_state["current_index"],
            task=raw_state["task"],
            done=raw_state["done"]
        )
