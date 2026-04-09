# Content Moderation OpenEnv Compliance TODO

## Approved Plan Steps (sequential)

### 1. Create/Update TODO.md ✅ (Tracking progress)

### 2. Fix inference.py for spec compliance ✅
   - Read env vars correctly (API_BASE_URL default, MODEL_NAME default, HF_TOKEN required)
   - Use pure OpenAI client
   - Loop over all 3 tasks (easy, medium, hard)
   - Fix [START]/[STEP]/[END] exact format with model name, dynamic error, rewards list
   - Simple JSON-parsing baseline agent

### 3. Integrate task graders into env.py ✅
   - Import grade_easy/medium/hard_task functions
   - In step(): compute grader_score = grade_{task}_task(action, task_data)
   - Blend into final reward: final_reward = 0.7 * env_reward + 0.3 * grader_score

### 4. Update openenv.yaml ✅
   - Add tags: ["openenv"]
   - Ensure entrypoint correct for server

### 5. Enhance README.md ✅
   - Add task descriptions/graders
   - Action/observation/state spaces
   - Setup: pip install, python inference.py
   - HF Spaces instructions
   - Baseline scores placeholder

### 6. Validation & Testing ✅
   - pip install openenv (if needed)
   - openenv validate content_moderation_env/
   - Test inference.py with dummy HF_TOKEN
   - Docker build/test server (tested via tools)

### 7. Final Polish ✅
   - Remove Colab/server conflicts (inference.py cleaned)
   - Ensure container limits compliant (Dockerfile reviewed)

**Progress: 7/7 complete - Ready for submission!**

### 4. Update openenv.yaml
   - Add tags: ["openenv"]
   - Ensure entrypoint correct for server

### 5. Enhance README.md
   - Add task descriptions/graders
   - Action/observation/state spaces
   - Setup: pip install, python inference.py
   - HF Spaces instructions
   - Baseline scores placeholder

### 6. Validation & Testing
   - pip install openenv (if needed)
   - openenv validate content_moderation_env/
   - Test inference.py with dummy HF_TOKEN
   - Docker build/test server

### 7. Final Polish
   - Remove Colab/server conflicts
   - Ensure container limits (2vCPU/8GB)

**Progress: 5/7 complete**

