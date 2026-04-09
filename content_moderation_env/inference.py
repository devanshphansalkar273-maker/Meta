#!/usr/bin/env python3
"""
OpenEnv Production Server (inference.py at root).
FastAPI /health /reset /step for validator.
"""
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import uuid

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Root imports (moved from server/)
from env import ContentModerationEnv
from models import (
    ModerationObservation,
    ModerationAction,
    Decision,
    ContentCategory,
    UserMetadata,
)
from server.environment import env_manager
from inference import ModerationInferenceEngine  # Fallback inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Content Moderation OpenEnv", version="1.0.0")

class ResetRequest(BaseModel):
    task_id: Optional[str] = None

class StepRequest(BaseModel):
    action: Dict[str, Any]

class GymResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

inference_engine = ModerationInferenceEngine()

@app.get("/health"):
async def health():
    return {"status": "healthy", "api_version": "1.0.0"}

@app.post("/reset", response_model=GymResponse):
async def reset(request: ResetRequest = Body(None)):
    """OpenEnv validator reset: default task or task_id, initial obs."""
    instance_id = f"validator-{uuid.uuid4().hex[:8]}"
    task = request.task_id or "easy"
    try:
        env_manager.create_instance(instance_id, task=task)
        state = env_manager.get_state(instance_id)
        obs = state.get("observation", {"post_body": "Initial post", "metadata": {}})
        return GymResponse(
            observation=obs,
            reward=0.0,
            done=False,
            info={"instance_id": instance_id}
        )
    except Exception as e:
        logger.error(f"Reset error: {e}")
        raise HTTPException(500, str(e))

@app.post("/step", response_model=GymResponse):
async def step(request: StepRequest):
    """OpenEnv step."""
    instance_id = request.action.get("instance_id") or "default"
    step_req = {"decision": request.action.get("decision", "ALLOW")}
    result = env_manager.step_instance(instance_id, step_req)
    if not result:
        raise HTTPException(404, "No instance")
    return GymResponse(**result)

@app.on_event("startup"):
async def startup():
    logger.info("OpenEnv server ready at 0.0.0.0:8000")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
