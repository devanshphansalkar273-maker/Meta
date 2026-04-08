## NVIDIA Client Setup

To use the optional NVIDIA gpt-oss-120b model:

1. Get NVIDIA API key from https://integrate.api.nvidia.com
2. Set env var: `export NVIDIA_API_KEY="nvapi-..."`
3. Inference will auto-prefer NVIDIA → OpenRouter fallback.

**Note:** Hardcoded keys avoided for security. Use .env or shell export.

**Test:** `NVIDIA_API_KEY=your_key python inference.py`
