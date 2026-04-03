import os
import json
import time
from typing import Optional
from openai import OpenAI
from env import ContentModerationEnv
from models import ModerationAction, Decision

def extract_action(client: OpenAI, obs) -> ModerationAction:
    prompt = f"""
You are an expert AI content moderator.
Evaluate the following post and respond with a JSON object.

Post ID: {obs.post_id}
Post Body: {obs.post_body}
Metadata: {obs.metadata.model_dump_json()}
Context History: {obs.context}

Respond strictly in the following JSON format matching this schema:
{{
  "decision": "ALLOW" | "FLAG" | "REMOVE" | "ESCALATE",
  "reasoning": "Brief concise explanation of why you chose this action.",
  "confidence_score": <float between 0.0 and 1.0>
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a highly capable content moderation API."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0  # Zero temp for high reproducibility
        )
        data = json.loads(response.choices[0].message.content)
        
        # Ensure enum normalization
        data["decision"] = Decision(data.get("decision", "ESCALATE"))
        return ModerationAction(**data)

    except Exception as e:
        print(f"Error during OpenAI inference: {e}")
        return ModerationAction(
            decision=Decision.ESCALATE,
            reasoning=f"Fallback forced due to exception: {str(e)}",
            confidence_score=0.0
        )

def main():
    if "OPENAI_API_KEY" not in os.environ:
        print("CRITICAL: OPENAI_API_KEY environment variable is missing. Validation will fail.")
        return

    print("=============================================")
    print("Meta OpenEnv: Content Moderation Baseline")
    print("=============================================\n")
    
    start_time = time.time()
    client = OpenAI()
    env = ContentModerationEnv()
    
    obs = env.reset()
    done = False
    
    step_count = 0
    total_dense_reward = 0.0

    while not done:
        step_count += 1
        print(f"[Step {step_count}] Processing Post ID: {obs.post_id}")
        print(f"  └─ Content: {obs.post_body}")
        
        action = extract_action(client, obs)
        print(f"  └─ AI Decision: {action.decision.value} (Confidence: {action.confidence_score})")
        print(f"  └─ Reasoning: {action.reasoning}")
        
        obs, reward, done, info = env.step(action)
        total_dense_reward += reward
        
        print(f"  └─ Env Response: Reward = {reward}")
        print(f"  └─ Feedback: {info['reward_details']['reason']}")
        
        if "task_final_score" in info:
            print(f"\n>>> FINISHED TASK: {info['active_task']} | OVERALL MULTIPLIER: {info['task_final_score']}")
            print("----------------------------------------------------------------------------------\n")

    elapsed = time.time() - start_time
    print("=============================================")
    print("EVALUATION COMPLETE")
    print("=============================================")
    print(f"Total Computation Time: {elapsed:.2f} seconds")
    print(f"Total Iterative Dense Reward: {total_dense_reward:.2f}")
    
    final_state = env.state()
    print("\nFINAL DETERMINISTIC TASK SCORES:")
    tasks = ["Task1Easy (Spam)", "Task2Medium (Toxic)", "Task3Hard (Ambiguity)"]
    for idx, t_name in enumerate(tasks):
        score = final_state["task_scores"][idx] if idx < len(final_state["task_scores"]) else 0.0
        print(f"- {t_name}: {score:.2f} / 1.0")

if __name__ == "__main__":
    main()
