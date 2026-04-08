#!/usr/bin/env python3
"""
Production-ready FastAPI server for Content Moderation OpenEnv.
Provides REST endpoints for inference, environment management, and health checks.
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from inference import ModerationInferenceEngine
from models import ModerationObservation, ModerationAction, Decision, ContentCategory, UserMetadata
from server.environment import env_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Content Moderation Environment API",
    description="OpenEnv-compatible REST API for AI content moderation using GPT-4.1",
    version="1.0.0"
)

# Request/Response models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    api_version: str
    has_api_key: bool


class InferenceRequest(BaseModel):
    post_id: str
    post_body: str
    author_trust_score: float = 0.5
    account_age_days: int = 0
    reports_count: int = 0
    virality_score: float = 0.0
    context: list = []


class InferenceResponse(BaseModel):
    post_id: str
    decision: str
    content_category: str
    reasoning: str
    confidence_score: float


class StepRequest(BaseModel):
    decision: str
    content_category: str = "SAFE"
    reasoning: str = ""
    confidence_score: float = 0.5


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict
    state: dict


class EnvironmentState(BaseModel):
    instance_id: str
    current_step: int
    cumulative_reward: float
    done: bool
    metrics: Dict[str, Any]
    task: Optional[str] = None
    step_count: Optional[int] = None


# Global inference engine
inference_engine = ModerationInferenceEngine()


@app.on_event("startup")
async def on_startup():
    """Log server address on startup."""
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Server running at http://localhost:{port}")
    print(f"Server running at http://localhost:{port}")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint for monitoring and Kubernetes liveness probes.
    
    Returns:
        HealthResponse with status and API configuration info
    """
    api_key_exists = bool(os.getenv('HF_TOKEN') or os.getenv('OPENROUTER_API_KEY'))
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        api_version="1.0.0",
        has_api_key=api_key_exists
    )


@app.post("/inference", response_model=InferenceResponse, tags=["Inference"])
async def run_inference(request: InferenceRequest) -> InferenceResponse:
    """
    Run a single inference step for content moderation.
    
    Args:
        request: InferenceRequest with post and metadata information
        
    Returns:
        InferenceResponse with moderation decision and confidence
    """
    try:
        # Create observation from request
        obs = ModerationObservation(
            post_id=request.post_id,
            post_body=request.post_body,
            metadata=UserMetadata(
                user_id=f"api_user_{request.post_id}",
                timestamp=datetime.utcnow().isoformat(),
                reports_count=request.reports_count,
                author_trust_score=request.author_trust_score,
                account_age_days=request.account_age_days,
                virality_score=request.virality_score
            ),
            context=request.context
        )
        
        # Get moderation decision
        action = inference_engine.get_moderation_decision(obs)
        
        logger.info(f"Inference: {request.post_id} -> {action.decision.value} "
                   f"(confidence: {action.confidence_score:.2f})")
        
        return InferenceResponse(
            post_id=request.post_id,
            decision=action.decision.value,
            content_category=action.content_category.value,
            reasoning=action.reasoning,
            confidence_score=action.confidence_score
        )
        
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/environment/create", response_model=EnvironmentState, tags=["Environment"])
async def create_environment(instance_id: str = Body(...), task: str = Body(default="easy")) -> EnvironmentState:
    """
    Create a new environment instance.

    Args:
        instance_id: Unique identifier for the environment instance
        task: Task difficulty - "easy", "medium", or "hard"

    Returns:
        EnvironmentState with initial state information
    """
    try:
        env = env_manager.create_instance(instance_id, task=task)
        state = env_manager.get_state(instance_id)
        
        return EnvironmentState(
            instance_id=instance_id,
            current_step=state["current_step"],
            cumulative_reward=state["cumulative_reward"],
            done=state["done"],
            metrics=state["metrics"],
            task=state.get("task"),
            step_count=state.get("step_count")
        )
    except Exception as e:
        logger.error(f"Environment creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/environment/{instance_id}/state", response_model=EnvironmentState, tags=["Environment"])
async def get_environment_state(instance_id: str) -> EnvironmentState:
    """Get current state of an environment instance."""
    state = env_manager.get_state(instance_id)

    if not state:
        raise HTTPException(status_code=404, detail="Environment instance not found")

    return EnvironmentState(
        instance_id=instance_id,
        current_step=state["current_step"],
        cumulative_reward=state["cumulative_reward"],
        done=state["done"],
        metrics=state["metrics"],
        task=state.get("task"),
        step_count=state.get("step_count")
    )


@app.post("/environment/{instance_id}/step", response_model=StepResponse, tags=["Environment"])
async def step_environment(instance_id: str, request: StepRequest) -> StepResponse:
    """
    Step an environment instance with an action.

    Args:
        instance_id: Environment instance identifier
        request: Step request with action details

    Returns:
        StepResponse with observation, reward, done, info, and state
    """
    result = env_manager.step_instance(instance_id, request.model_dump())

    if not result:
        raise HTTPException(status_code=404, detail="Environment instance not found")

    return StepResponse(**result)


@app.get("/environment/list", response_model=Dict[str, EnvironmentState], tags=["Environment"])
async def list_environments() -> Dict[str, EnvironmentState]:
    """List all active environment instances."""
    instances = env_manager.list_instances()
    
    result = {}
    for instance_id, state in instances.items():
        if state:
            result[instance_id] = EnvironmentState(
                instance_id=instance_id,
                current_step=state["current_step"],
                cumulative_reward=state["cumulative_reward"],
                done=state["done"],
                metrics=state["metrics"]
            )
    
    return result


@app.delete("/environment/{instance_id}", response_model=Dict[str, str], tags=["Environment"])
async def delete_environment(instance_id: str) -> Dict[str, str]:
    """Delete an environment instance."""
    if env_manager.delete_instance(instance_id):
        return {"status": "deleted", "instance_id": instance_id}
    else:
        raise HTTPException(status_code=404, detail="Environment instance not found")


@app.get("/status", response_model=Dict[str, Any], tags=["Status"])
async def get_status() -> Dict[str, Any]:
    """Get detailed service status."""
    api_key_exists = bool(os.getenv('HF_TOKEN') or os.getenv('OPENROUTER_API_KEY'))
    
    return {
        "service": "Content Moderation Environment API",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "api_version": "1.0.0",
        "configuration": {
            "api_base_url": "https://openrouter.ai/api/v1",
            "model": "openai/gpt-4.1",
            "has_api_key": api_key_exists,
            "fallback_enabled": True
        },
        "active_environments": len(env_manager.instances),
        "max_concurrent_environments": env_manager.max_instances
    }


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    print(f"Server running at http://localhost:{port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
