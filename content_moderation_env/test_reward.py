import sys
sys.path.insert(0, 'content_moderation_env')

from env import ContentModerationEnv
from models import ModerationAction, Decision, ContentCategory

print("Testing reward structure...")

# Test correct ALLOW on safe content (needs safe first item)
env = ContentModerationEnv()
obs = env.reset(task="easy")  # Assume first item in easy is spam for REMOVE test, check obs
print(f"First obs post_id: {obs.post_id}, body preview: {obs.post_body[:50]}...")

# Assume first is harmful (spam), test correct REMOVE
correct_remove = ModerationAction(
    decision=Decision.REMOVE, 
    content_category=ContentCategory.SPAM, 
    confidence_score=0.95,
    reasoning="Clear spam"
)
next_obs, reward_remove, done, info = env.step(correct_remove)
print(f"Correct REMOVE reward: {reward_remove}")
print(f"Reward details: {info['reward_details']['reason']}")

# Reset and test ALLOW on safe (need safe item; skip if first harmful, or force)
env.reset(task="easy")
allow_action = ModerationAction(
    decision=Decision.ALLOW, 
    content_category=ContentCategory.SAFE, 
    confidence_score=0.95,
    reasoning="Safe content"
)
# But if gt is REMOVE, this is false negative
next_obs, reward_allow_wrong, _, info = env.step(allow_action)
print(f"ALLOW on harmful (false neg): {reward_allow_wrong}")

print("Note: Full test needs dataset inspection for safe gt.")

