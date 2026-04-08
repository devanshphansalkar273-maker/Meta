from content_moderation_env.client import ModerationClient
from content_moderation_env.models import ModerationAction

client = ModerationClient("http://localhost:8000")

obs, state = client.reset()
print("Reset obs:", obs)
print("Initial state:", state)

action = ModerationAction(action="FLAG")
result = client.step(action)
print("Step result:", result)
