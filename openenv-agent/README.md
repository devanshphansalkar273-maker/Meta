# OpenEnv Agent

Python client library for connecting RL agents to OpenEnv servers.

## Installation

```bash
pip install openenv-agent
```

## Quick Start

```python
from openenv_agent import OpenEnvClient, ModerationAgent

# Connect to an OpenEnv server
client = OpenEnvClient(base_url="http://localhost:8000")

# Reset environment
obs = client.reset()

# Use the moderation agent
agent = ModerationAgent()
action = agent.predict(obs)

# Step the environment
next_obs, reward, done, info = client.step(action)
```

## CLI Usage

```bash
# Run agent against a server
openenv-agent run http://localhost:8000

# Interactive mode
openenv-agent interactive http://localhost:8000

# Evaluate on a dataset
openenv-agent eval http://localhost:8000 --dataset ./data.json
```

## Features

- Async/sync OpenEnv client with Gymnasium-style API
- Built-in ModerationAgent for content moderation environments
- Environment loader from openenv.yaml configs
- CLI tool for easy server interaction