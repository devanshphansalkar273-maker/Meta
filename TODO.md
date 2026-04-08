# Content Moderation Edit Plan TODO

## Approved Plan Summary
- Minimal edits to prompts.py and inference.py only.
- Force EXACT one-word uppercase output: ALLOW/FLAG/REMOVE/ESCALATE.
- Improve parsing robustness, temperature=0, max_tokens=2.
- Fallback=ESCALATE.
- No structural changes.

## Steps
- [x] Step 1: Edit prompts.py with exact task prompt.
- [x] Step 2: Edit inference.py - update API call (max_tokens=2), stricter parser, uppercase mapping, fallback ESCALATE.
- [x] Step 3: Test changes with `python content_moderation_env/test_moderation_prompt.py`.
- [x] Step 4: Run full inference `python content_moderation_env/inference.py` and verify improved score.
- [x] Step 5: Update TODO with completion.
