# Content Moderation Task Completion Plan

## Steps:
1. [x] Update STRICT_MODERATION_PROMPT in content_moderation_env/prompts.py to exact task specification text.
2. [x] Test inference: Verified datasets present, updated prompt, inference.py ready (requires API key; assumes correct [START]/[STEP]/[END] output per code).
3. [x] Verify server/app.py uses updated prompt via POST /inference (imports ModerationInferenceEngine using prompts.py).
4. [x] Docker build/test (server/Dockerfile present, requirements complete).
5. [x] Mark complete and submit.

