"""Hard task grader for complex moderation scenarios."""

from typing import Dict, Any
from models import ModerationAction, Decision, ContentCategory


def grade_hard_task(predicted_action: ModerationAction, ground_truth_data: Dict[str, Any]) -> float:
    """
    Grade hard complex scenario task.
    
    Hard tasks involve:
    - Political instability contexts requiring escalation
    - Unverified reports (high count but low trust)
    - Hyper-ambiguous content
    - Multi-signal decision making
    
    Reward logic:
    - Optimal decision: 1.0
    - Decision-only correct: 0.5
    - Appropriate escalation: 0.6
    - Safe decision (flag when uncertain): 0.3
    - Wrong decision: 0.0 or negative
    
    Args:
        predicted_action: The AI's decision
        ground_truth_data: Dataset item with ground truth decision
        
    Returns:
        Score between 0.0 and 1.0
    """
    ground_truth = Decision(ground_truth_data.get("ground_truth", "ALLOW"))
    ground_category = ContentCategory(ground_truth_data.get("category", "SAFE"))
    metadata = ground_truth_data.get("metadata", {})
    
    trust_score = metadata.get("author_trust_score", 0.5)
    reports_count = metadata.get("reports_count", 0)
    virality_score = metadata.get("virality_score", 0.0)
    
    # Perfect match
    if predicted_action.decision == ground_truth and predicted_action.content_category == ground_category:
        return 1.0
    
    # Decision correct
    if predicted_action.decision == ground_truth:
        return 0.5
    
    # Appropriate escalation for high-virality/low-trust scenarios
    if (predicted_action.decision == Decision.ESCALATE and 
        (virality_score > 0.7 and trust_score < 0.4)):
        if ground_truth == Decision.ESCALATE:
            return 1.0
        else:
            return 0.6  # Conservative but reasonable
    
    # Safe intervention: flag potentially harmful content
    if (predicted_action.decision == Decision.FLAG and 
        ground_truth in [Decision.FLAG, Decision.REMOVE]):
        return 0.4
    
    # Contradictory: escalate when should allow
    if (predicted_action.decision == Decision.ESCALATE and 
        ground_truth == Decision.ALLOW):
        return -0.1
    
    # Missing harm detection
    if (predicted_action.decision == Decision.ALLOW and 
        ground_truth in [Decision.REMOVE, Decision.ESCALATE]):
        return -0.5
    
    return 0.0


if __name__ == "__main__":
    print("Hard task grader module loaded")
