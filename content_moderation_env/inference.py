import os
import json
import time
from openai import OpenAI
from env import ContentModerationEnv
from models import ModerationAction, Decision, ContentCategory

def extract_action(client: OpenAI, obs) -> ModerationAction:
    prompt = f"""
    You are an expert AI content moderator running Meta-scale pipeline validation.
    Evaluate the following post across Text, Metadata, and Contextual History.
    
    [CONTENT DATA]
    Post ID: {obs.post_id}
    Post Body: {obs.post_body}
    
    [METADATA & SIGNALS]
    Account Age (Days): {obs.metadata.account_age_days}
    Author Trust Score (0.0-1.0): {obs.metadata.author_trust_score}
    Virality Score: {obs.metadata.virality_score}
    Report Count: {obs.metadata.reports_count}
    
    [HISTORICAL CONTEXT]
    {obs.context if obs.context else "No prior infractions."}
    
    [EVALUATION DIRECTIVES]
    1. VIRALITY RISK: If Virality > 0.8 and you suspect toxicity, you MUST 'REMOVE' or 'ESCALATE'.
    2. RECIDIVISM: If Author Trust < 0.4 and context shows past violations, apply zero-tolerance to ambiguous text.
    3. CIB/SPAM: If Report Count > 20 but the text looks safe, check for hidden bot patterns before allowing.
    
    Respond strictly in the following JSON format matching this schema:
    {{
      "decision": "ALLOW" | "FLAG" | "REMOVE" | "ESCALATE",
      "content_category": "SPAM" | "HATE_SPEECH" | "MISINFORMATION" | "HARASSMENT" | "SAFE",
      "reasoning": "Specify the exact policy violation or contextual justification...",
      "confidence_score": <float 0.0 to 1.0>
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Trust and Safety API."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        data = json.loads(response.choices[0].message.content)
        data["decision"] = Decision(data.get("decision", "ESCALATE"))
        data["content_category"] = ContentCategory(data.get("content_category", "SAFE"))
        return ModerationAction(**data)

    except Exception as e:
        print(f"Error during OpenAI inference: {e}")
        return ModerationAction(
            decision=Decision.ESCALATE,
            content_category=ContentCategory.SAFE,
            reasoning=f"Fallback forced: {str(e)}",
            confidence_score=0.0
        )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Trust and Safety API."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        data = json.loads(response.choices[0].message.content)
        
        # Safe Enum casting
        data["decision"] = Decision(data.get("decision", "ESCALATE"))
        data["content_category"] = ContentCategory(data.get("content_category", "SAFE"))
        return ModerationAction(**data)

    except Exception as e:
        print(f"Error during OpenAI inference: {e}")
        return ModerationAction(
            decision=Decision.ESCALATE,
            content_category=ContentCategory.SAFE,
            reasoning=f"Fallback forced: {str(e)}",
            confidence_score=0.0
        )


def extract_appeal_action(client: OpenAI, obs, original_ai_reason: str) -> ModerationAction:
    appeal_text = obs.metadata.user_appeal_statement or "I did nothing wrong!"

    prompt = f"""
You are the Secondary Review Integrity Agent. This post was previously REMOVED, and the user has submitted an official appeal.

Original Post: {obs.post_body}
Original AI Decision Reason: {original_ai_reason}

🧑‍⚖️ USER APPEAL TEXT: "{appeal_text}"

Instructions:
1. Evaluate if the original removal was a 'False Positive' based on the user's new context.
2. If the user's appeal provides valid context (e.g., reclaiming a slur, obvious satire missed by the first pass), you must 'ALLOW' to reinstate the post.
3. If the user is lying or doubling down on policy violations, you must 'REMOVE' to uphold the ban.
4. Your confidence score must be highly calibrated. Upholding a wrong decision carries a catastrophic penalty.

Respond strictly in the following JSON format matching this schema:
{
  "decision": "ALLOW" | "FLAG" | "REMOVE" | "ESCALATE",
  "content_category": "SPAM" | "HATE_SPEECH" | "MISINFORMATION" | "HARASSMENT" | "SAFE",
  "reasoning": "Brief concise diagnostic...",
  "confidence_score": <float 0.0 to 1.0>
}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Trust and Safety API."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        data = json.loads(response.choices[0].message.content)
        data["decision"] = Decision(data.get("decision", "ESCALATE"))
        data["content_category"] = ContentCategory(data.get("content_category", "SAFE"))
        return ModerationAction(**data)
    except Exception as e:
        print(f"Error during appeal inference: {e}")
        return ModerationAction(
            decision=Decision.ESCALATE,
            content_category=ContentCategory.SAFE,
            reasoning=f"Fallback forced appeal: {str(e)}",
            confidence_score=0.0
        )


def main():
    if "OPENAI_API_KEY" not in os.environ:
        print("CRITICAL: OPENAI_API_KEY environment variable is missing.")
        return

    print("=============================================")
    print("TrustEnv: Continuous Validation Feed")
    print("=============================================\n")
    
    start_time = time.time()
    client = OpenAI()
    env = ContentModerationEnv()
    
    obs = env.reset()
    done = False
    
    step_count = 0
    while not done:
        step_count += 1
        print(f"\n[Feed Step {step_count}] Event: {obs.post_id}")
        
        # Agent decides
        action = extract_action(client, obs)
        print(f"  └─ AI Decision: {action.decision.value} | Category: {action.content_category.value}")
        print(f"  └─ Confidence: {action.confidence_score} | Reason: {action.reasoning}")

        # Secondary appeal review when a remove decision exists and user provided appeal text
        if action.decision == Decision.REMOVE and obs.metadata.user_appeal_statement:
            appeal_action = extract_appeal_action(client, obs, action.reasoning)
            print(f"  └─ Appeal Review: {appeal_action.decision.value} | Confidence: {appeal_action.confidence_score} | Reason: {appeal_action.reasoning}")

        # Step environment
        obs, reward, done, info = env.step(action)

        print(f"  └─ Env Response: Reward {reward:.2f}")
        print(f"  └─ Grader Logic: {info['reward_details']['reason']}")
        print(f"  └─ Active Cumulative Reward: {info['cumulative_reward']:.2f}")

    # Final Metrics Printout
    elapsed = time.time() - start_time
    final_state = env.state()
    metrics = final_state["metrics"]
    
    print("\n=============================================")
    print("EVALUATION COMPLETE")
    print("=============================================")
    print(f"Total Computation Time: {elapsed:.2f} seconds")
    print(f"Total Feed Events: {final_state['current_step']}")
    print(f"Final Cumulative Dense Reward: {final_state['cumulative_reward']}\n")
    
    print("--- 📊 Evaluation Metrics Matrix ---")
    print(f"True Positives (TP):  {metrics['true_positives']}")
    print(f"False Positives (FP): {metrics['false_positives']} (Censorship Cost)")
    print(f"True Negatives (TN):  {metrics['true_negatives']}")
    print(f"False Negatives (FN): {metrics['false_negatives']} (Missed Harm)\n")
    
    print("--- 🎯 Precision & Recall ---")
    print(f"Precision:         {metrics['precision']}")
    print(f"Recall:            {metrics['recall']}")
    print(f"F1 Score:          {metrics['f1_score']}")
    print(f"Category Accuracy: {metrics['category_accuracy']} / {final_state['current_step']} correct")

if __name__ == "__main__":
    main()
