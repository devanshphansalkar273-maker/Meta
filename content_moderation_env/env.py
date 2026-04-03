import json
import logging
from typing import Tuple, Dict, Any, List
from pathlib import Path

from models import (
    Decision, 
    ContentCategory,
    ModerationObservation, 
    ModerationAction, 
    UserMetadata,
    ModerationReward
)

logger = logging.getLogger(__name__)

class ContentModerationEnv:
    """
    OpenEnv-compliant continuous environment featuring Dynamic User Reputation shifting.
    """
    def __init__(self, dataset_dir: str = "datasets"):
        self.dataset_dir = Path(dataset_dir)
        self.feed_queue = []
        self._load_continuous_feed()
        self.reset()

    def _load_continuous_feed(self):
        files = ["easy_spam.json", "medium_toxic.json", "hard_mixed.json"]
        self.feed_queue = []
        for file in files:
            path = self.dataset_dir / file
            if path.exists():
                with open(path, 'r') as f:
                    self.feed_queue.extend(json.load(f))

    def reset(self) -> ModerationObservation:
        self.current_idx = 0
        self.done = False
        self.cumulative_reward = 0.0
        
        # User Reputation Ledger Mapping { user_id : { trust_score, violations } }
        self.reputation_ledger = {}
        
        # Asynchronous HITL Queue
        self.human_review_queue = []
        
        # Anti-exploit tracking state
        self.action_history = []
        self.consecutive_same_action = 0
        
        self.metrics = {
            "true_positives": 0, "false_positives": 0,
            "true_negatives": 0, "false_negatives": 0,
            "category_accuracy": 0
        }
        
        if not self.feed_queue:
            raise RuntimeError("CRITICAL: Failed to load queue")

        return self._get_current_observation()

    def _get_current_observation(self) -> ModerationObservation:
        if self.current_idx >= len(self.feed_queue):
            return self._get_dummy_obs()
            
        data = self.feed_queue[self.current_idx]
        user_id = data.get("user_id", "anonymous")
        
        # Pull or Initialize user in dynamic ledger
        if user_id not in self.reputation_ledger:
            self.reputation_ledger[user_id] = {
                "base_trust": data.get("metadata", {}).get("author_trust_score", 0.5),
                "violations": 0
            }
            
        current_trust = self.reputation_ledger[user_id]["base_trust"]
        
        # Dynamically inject context if they have active violations
        dynamic_context = data.get("context", [])
        if self.reputation_ledger[user_id]["violations"] > 0:
            dynamic_context.append(f"WARNING: User has incurred {self.reputation_ledger[user_id]['violations']} distinct policy violation intervention(s) earlier within this active shift!")
        
        metadata = data.get("metadata", {})
        
        return ModerationObservation(
            post_id=data["post_id"],
            post_body=data["post_body"],
            metadata=UserMetadata(
                user_id=user_id,
                timestamp=metadata.get("timestamp"),
                reports_count=metadata.get("reports_count", 0),
                author_trust_score=current_trust,
                account_age_days=metadata.get("account_age_days", 0),
                virality_score=metadata.get("virality_score", 0.0)
            ),
            context=dynamic_context
        )

    def step(self, action: ModerationAction) -> Tuple[ModerationObservation, float, bool, Dict[str, Any]]:
        if self.done:
             raise RuntimeError("Environment done. Reset required.")

        task_data = self.feed_queue[self.current_idx]
        user_id = task_data.get("user_id", "anonymous")
        gt_decision = Decision(task_data["ground_truth"])
        gt_category = ContentCategory(task_data.get("category", "SAFE"))

        reasons = []
        step_reward = 0.0
        
        user_trust = self.reputation_ledger[user_id]["base_trust"]
        user_violations = self.reputation_ledger[user_id]["violations"]

        # Metric Logic Mapping
        agent_intervened = action.decision != Decision.ALLOW
        gt_intervened = gt_decision != Decision.ALLOW

        if agent_intervened and gt_intervened: self.metrics["true_positives"] += 1
        elif agent_intervened and not gt_intervened: self.metrics["false_positives"] += 1
        elif not agent_intervened and gt_intervened: self.metrics["false_negatives"] += 1
        else: self.metrics["true_negatives"] += 1

        if action.content_category == gt_category: self.metrics["category_accuracy"] += 1

        # ---- DYNAMIC REPUTATION REWARD LOGIC ----
        if action.decision == Decision.ESCALATE:
            # Asynchronous HITL Logic: Reward is 0.0, post awaits review
            step_reward = 0.0
            reasons.append("HITL DEFERRED (0.0): Post pushed to explicit Human Review Queue for asynchronous validation.")
            self.human_review_queue.append({
                "post_id": task_data["post_id"],
                "gt_decision": gt_decision,
                "confidence": action.confidence_score
            })
        elif gt_decision == Decision.ALLOW:
            if action.decision == Decision.ALLOW:
                step_reward += 1.0
                reasons.append("Correctly allowed safe content.")
            else:
                # FALSE POSITIVE PENALTY
                if user_trust > 0.9:
                    step_reward -= 2.0
                    reasons.append("VIP Penalty (-2.0): Falsely suppressed highly trusted active platform user.")
                elif user_violations > 0:
                    step_reward -= 0.5
                    reasons.append("Lenient UX Cost (-0.5): Falsely suppressed safe content, but justified caution given user's active shift violations.")
                else:
                    step_reward -= 1.0
                    reasons.append("UX Cost Penalty (-1.0): False Positive censorship.")
        else: 
            if action.decision == gt_decision:
                step_reward += 1.0
                reasons.append("Correct precise intervention.")
                self.reputation_ledger[user_id]["violations"] += 1
                self.reputation_ledger[user_id]["base_trust"] = max(0.0, user_trust - 0.2)
            elif action.decision == Decision.ALLOW:
                # FALSE NEGATIVE PENALTY
                if user_violations >= 1:
                    step_reward -= 6.0
                    reasons.append(f"NEGLIGENCE (-6.0): Allowed harmful repeated content from an active offender.")
                else:
                    step_reward -= 5.0
                    reasons.append(f"MASSIVE COST (-5.0): Failed to address harmful content ({gt_category.value}).")
            else:
                step_reward += 0.2
                self.reputation_ledger[user_id]["violations"] += 1 
                reasons.append("Partial progress (+0.2): Triggered intervention, but sub-optimal vector.")

        # ---- CATEGORY + CONFIDENCE BONUSES ----
        if action.decision != Decision.ESCALATE:
            if action.content_category == gt_category:
                step_reward += 0.2
            else:
                step_reward -= 0.1

            if action.decision != gt_decision and action.confidence_score < 0.4:
                step_reward += 0.3
            elif action.decision != gt_decision and action.confidence_score > 0.9:
                step_reward -= 0.5

        # ---- ANTI-EXPLOIT TRACKING ----
        if self.action_history and self.action_history[-1] == action.decision:
            self.consecutive_same_action += 1
        else:
            self.consecutive_same_action = 0
            
        self.action_history.append(action.decision)

        if self.consecutive_same_action >= 3:
            penalty = -0.5 * (self.consecutive_same_action - 2)
            step_reward += penalty
            reasons.append(f"EXPLOIT TRACKER ({penalty}): Repetitive action decay active.")

        self.cumulative_reward += step_reward

        self.current_idx += 1
        
        hitl_payout = 0.0
        if self.current_idx >= len(self.feed_queue):
            self.done = True
            # Flush HITL Queue upon episode conclusion
            if self.human_review_queue:
                reasons.append(f"\n--- BATCH PROCESSING HITL QUEUE ({len(self.human_review_queue)} posts) ---")
                for review in self.human_review_queue:
                    if review["gt_decision"] == Decision.ESCALATE:
                        hitl_payout += 1.5
                        reasons.append(f"✅ {review['post_id']}: Optimal Escalation (+1.5)")
                    else:
                        hitl_payout -= 1.0
                        reasons.append(f"❌ {review['post_id']}: Obvious Case - Frivolous Escalation Waste (-1.0)")
                step_reward += hitl_payout
                self.cumulative_reward += hitl_payout
                self.human_review_queue.clear()
        
        # Override payout description into info payload on end state
        info = {
            "reward_details": ModerationReward(score=step_reward, reason=" | ".join(reasons)).model_dump(),
            "hitl_batch_payout": hitl_payout,
            "cumulative_reward": self.cumulative_reward,
            "metrics": self.state()["metrics"]
        }

        return self._get_current_observation(), step_reward, self.done, info

    def state(self) -> Dict[str, Any]:
        tp = self.metrics["true_positives"]
        fp = self.metrics["false_positives"]
        fn = self.metrics["false_negatives"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {
            "current_step": self.current_idx,
            "done": self.done,
            "cumulative_reward": round(self.cumulative_reward, 3),
            "metrics": {
                **self.metrics,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1_score": round(2 * (precision * recall) / (precision + recall), 3) if precision + recall > 0 else 0.0,
            }
        }

    def _get_dummy_obs(self) -> ModerationObservation:
        return ModerationObservation(
            post_id="END", post_body="DONE",
            metadata=UserMetadata(user_id="END", timestamp="1970-01-01T00:00:00Z", author_trust_score=1.0, account_age_days=0),
            context=[]
        )
