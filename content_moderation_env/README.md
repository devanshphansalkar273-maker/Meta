# Content Moderation OpenEnv Environment 🚀

## Overview & Motivation
Real-world content moderation simulation for social media platforms. Agents triage posts using user reputation, context, virality signals, and policy rules. Simulates dynamic reputation shifts, HITL escalation queues, anti-exploit protections.

**OpenEnv Compliance:** ✅ Typed Pydantic models, `step()`/`reset()`/`state()`, `openenv.yaml`, 3 graded tasks, programmatic graders, meaningful dense rewards.

## Tasks (Easy → Medium → Hard)
1. **Easy (Spam Detection)**: Clear promotional spam patterns. GT: REMOVE spam, ALLOW legit.
2. **Medium (Toxic Content)**: Sarcasm/hyperbole vs harassment. High reports but ambiguous.
3. **Hard (Mixed Scenarios)**: Multi-signal (low trust + high virality + global events), optimal ESCALATE.

**Datasets:** `datasets/{easy_spam.json, medium_toxic.json, hard_mixed.json}`

**Grader Functions:** `tasks/{easy.py, medium.py, hard.py}` → deterministic score [0.0-1.0]

## Observation Space
```python
ModerationObservation(
  post_id: str
  post_body: str
  metadata: UserMetadata(...)  # trust_score, reports, virality, etc.
  context: List[str]  # violation warnings, etc.
)
```

## Action Space  
```python
ModerationAction(
  decision: Decision (ALLOW|FLAG|REMOVE|ESCALATE)
  content_category: ContentCategory (SPAM|TOXIC|...)
  reasoning: str
  confidence_score: float [0.0-1.0]
)
```

## State Space
```python
Dict: {"task": str, "step_count": int, "current_index": int, "done": bool}
```

## Reward Function
**Incremental & Dense:**
- Correct ALLOW: +1.0
- Correct intervention: +0.9 (UX cost ceiling)
- False negatives: 0.0 (harm priority)
- False positives: 0.0-0.3 (VIP/trust weighted)
- HITL ESCALATE: 0.0 deferred → batch payout at end
- Grader blend: 70% env logic + 30% task grader
- Anti-exploit: repetition decay
- Confidence bonuses/penalties

## Setup & Usage
```bash
# Clone & install
pip install -r requirements.txt  # or pyproject.toml

# Local inference baseline
set HF_TOKEN=your_huggingface_token && python inference.py

# Optional explicit defaults
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct

# Server mode (HF Spaces)
cd server
docker build -t moderation-env .
docker run -p 8000:8000 -e HF_TOKEN=... moderation-env
```

**Env Vars:**
| Var | Required | Default | Desc |
|----|----------|---------|------|
| HF_TOKEN | ✅ | - | Hugging Face access token |
| MODEL_NAME | - | `meta-llama/Llama-3.3-70B-Instruct` | LLM model |
| API_BASE_URL | - | `https://router.huggingface.co/v1` | Hugging Face router endpoint |

## Baseline Configuration (Hugging Face)
```text
Provider: Hugging Face Router
Model:    meta-llama/Llama-3.3-70B-Instruct
```
*(Run `python inference.py` after setting `HF_TOKEN` to reproduce with the configured model.)*

## HF Spaces Deployment
1. Push to HF repo with `server/Dockerfile`
2. Set Secrets: `HF_TOKEN`
3. Tag: `openenv`
4. Ensure Running (2vCPU/8GB compliant)

## Validation
```bash
pip install openenv
openenv validate .
```

## Reference Implementation
Fully OpenEnv-compliant with production-grade features:
- Dynamic user reputation ledger
- Asynchronous HITL queue w/ batch payout
- Multi-signal observation (vision tags, appeals)
- Deterministic graders per difficulty
- Dense shaping rewards (no sparse cliff)

**Score your agent:**
```python
env = ContentModerationEnv()
obs = env.reset("hard")
while not env.done:
    action = agent.act(obs)  # Your agent
    obs, reward, done, info = env.step(action)
```
