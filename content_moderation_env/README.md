# 🚀 Content Moderation OpenEnv - Production Ready

**Complete AI Content Moderation Environment using OpenEnv + GPT-4.1 via OpenRouter**

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set API key
export HF_TOKEN="your_openrouter_api_key"

# Run inference
python inference.py

# Start API server
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Features

✅ Production-ready implementation  
✅ OpenEnv compliant (reset, step, state)  
✅ GPT-4.1 via OpenRouter API  
✅ Fallback keyword-based logic  
✅ Dense reward signals  
✅ REST API with /health endpoint  
✅ Docker & HF Spaces ready  
✅ Multi-task (easy, medium, hard)  
✅ Python SDK client  
✅ GRPO training example  

## Project Structure

```
├── inference.py         # Main script [START/STEP/END format]
├── env.py              # OpenEnv environment
├── models.py           # Pydantic schemas
├── client.py           # Python SDK
├── server/             # FastAPI server
├── tasks/              # Task graders
├── datasets/           # Test data
└── Dockerfile          # Container config
```

## API Endpoints

- `GET /health` - Health check
- `POST /inference` - Single moderation
- `POST /environment/create` - Create env instance
- `GET /environment/{id}/state` - Get state
- `GET /status` - Server status

## Reward Logic

**Correct**: +1.0  
**Wrong VIP suppress**: -2.0  
**Wrong safe suppress**: -1.0  
**Missed critical harm**: -5.0 to -6.0  

Scores clamped to [0.0, 1.0].

## Model Configuration

- **API**: https://openrouter.ai/api/v1
- **Model**: openai/gpt-4.1
- **Temperature**: 0
- **Max tokens**: 200

## Docker

```bash
docker build -t moderation-env:latest .
docker run -e HF_TOKEN="key" -p 8000:8000 moderation-env:latest
```

## Version

**1.0.0** - Production Ready ✅
 

In the modern digital ecosystem—and hyper-critical to Meta's sprawling social frameworks (Facebook, Instagram, Threads)—platform safety hinges on **Content Moderation**. Human moderators cannot ingest the sheer volumetric scale of real-time posts, while legacy algorithmic filters are easily circumvented via obfuscation and evolving socio-geopolitical context.

Developing reinforcement learning agents capable of nuanced, human-like Trust & Safety interventions requires equally rigorous simulated testing grounds. **TrustEnv** solves this by strictly simulating the daily dashboard of a real-world social media moderator. Rather than a toy simulation, this environment forces agents to weigh complex metadata, historical contexts, and lethal textual ambiguity to make decisions that emulate the extreme real-word constraints faced across Meta’s infrastructure.

## 🏗️ Environment Design

Built fully natively to the `<OpenEnv>` structural ethos, TrustEnv provides a complete `step()`, `reset()`, and `state()` pipeline encapsulating multi-modal context vectors securely into unified Pydantic schemas. 

* The environment acts as an un-gamable **Deterministic Grader**.
* Each task isolates and heavily penalizes exploits commonly employed by LLM baseline agents (e.g. "always deleting" to secure 100% toxic recall, or "always escalating" to bypass complexity).

### 👁️ Observation Space
Presented to the agent on each iteration natively as a Pydantic `ModerationObservation`:
- `post_id`: Unique tracking identifier.
- `post_body`: The actual unstructured social media content.
- `metadata`: A deeply correlated dictionary supplying crucial context (`timestamp`, `reports_count`, `author_trust_score` [0.0 - 1.0], and `account_age_days`).
- `context`: An array providing historical user warnings or violations.

### 🕹️ Action Space
Agents must supply an un-hallucinated `ModerationAction` decision:
- **`decision`**: Enum enforcing one of:
  - `ALLOW`: The post remains safely on the platform.
  - `FLAG`: Visibility restriction applied (Shadowban).
  - `REMOVE`: Aggressive deletion for policy-violations.
  - `ESCALATE`: Submitted for human geographical/political review.
- **`reasoning`**: String representation driving step-by-step diagnostic chains.
- **`confidence_score`**: Output weighting [0.0, 1.0] representing internal alignment confidence.

## ⚖️ Advanced Dense Reward Function

Unlike legacy environments that only apply binary outcomes at standard episode termination (`done==True`), TrustEnv employs a mathematically constrained **Dense Reward Function** designed to steer partial progress while heavily punishing catastrophic failure:

1. **Extreme Asymmetric Penalization (`-2.5`)**: If the agent yields an `ALLOW` action on a ground-truth `REMOVE` constraint (toxic/hate speech), it eats a substantial numeric deduction representing severe real-world platform health damage.
2. **Censorship Scaling (`-1.0`)**: Deleting inherently safe content aggressively penalizes the model, enforcing high precision and effectively destroying any trivial *"just remove everything"* exploitative loops.
3. **Partial Action Returns (`+0.2 - +0.5`)**: Handing an ambiguous situation by submitting a `FLAG` on content that fundamentally deserved to be removed rewards the agent partially, recognizing their intuitive caution, whilst pushing them to improve decisiveness. 

## 🗺️ Progressive Task Difficulties

Three compartmentalized tasks seamlessly loop via `env.py` mapping specific dataset matrices against tailored validation scripts:

1. 🟢 **Easy (`easy_spam.json`)**: Raw pattern matching against hyper-linked financial scams, testing zero-shot generalization capabilities. *Grader constraint: Any false positive deletion zeros out the score completely to prevent spamming generic removals.*
2. 🟡 **Medium (`medium_toxic.json`)**: Tests profound obfuscated profanity against casual sarcastic expressions. *Grader constraint: Heavy mathematical penalty applied via a false negative/positive precision matrix matrix.*
3. 🔴 **Hard (`hard_mixed.json`)**: Incredibly high-volume, volatile sociopolitical stream inputs demanding expert localized knowledge mappings and extreme situational escalation parsing. *Grader constraint: Severe logarithmic exhaustion penalties applied if the agent delegates >30% of its workload to human ESCALATIONS.*

---

## 🛠️ Setup & Execution Instructions

We guarantee seamless compatibility directly out of the box leveraging pre-configured `pip` schemas and HF spaces `Dockerfiles`.

1. **Clone the Repository**:
   ```bash
   git clone <repository_url> && cd content_moderation_env
   ```
2. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional) Schema Verification**: Validate that the Pydantic schemas map correctly to our generated openenv constraints via the standard OpenEnv specification tool.
   ```bash
   openenv validate
   ```

### 🧠 Running Baseline Inference Evaluation

We provide a robust baseline validator tracking logic through an `openai` endpoint acting purely deterministically (`temperature=0.0`). 

1. Export your API token to the local environment variables wrapper.
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
2. Run the `inference.py` evaluator.
   ```bash
   python inference.py
   ```

### 📊 Expected Baseline Scores (GPT-4o-Mini)

When deploying standard baseline API models against the constraints:
- **Task 1 (Easy/Spam)**: Expected Score `1.0 / 1.0` (Perfect accuracy resolving standard url/scam routing paths).
- **Task 2 (Medium/Toxic)**: Expected Score `0.7 - 0.9` (Occasional misflags penalizing precision due to inability to discern regional sarcasm).
- **Task 3 (Hard/Mixed Feed)**: Expected Score `0.4 - 0.6` (LLMs typically incur brutal exhaustion penalties as they lean heavily on generic `ESCALATE` fallbacks when parsing unverified socio-geopolitical threat mappings).
