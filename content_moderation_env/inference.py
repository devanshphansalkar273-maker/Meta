from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class StepRequest(BaseModel):
    action: str | None = None

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/reset")
async def reset():
    return {
        "observation": {"message": "reset ok"},
        "reward": 0.0,
        "done": False,
        "info": {}
    }

@app.post("/step")
async def step(req: StepRequest):
    return {
        "observation": {"message": "step ok", "action": req.action},
        "reward": 0.0,
        "done": False,
        "info": {}
    }
