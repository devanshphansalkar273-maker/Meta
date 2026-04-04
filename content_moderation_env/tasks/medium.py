"""Medium task grader for toxic content detection."""

from typing import Dict, Any
from models import ModerationAction, Decision, ContentCategory


def grade_medium_task(predicted_action: ModerationAction, ground_truth_data: Dict[str, Any]) -> float:
    """
    Grade medium toxic detection task.
    
    Medium tasks have ambiguous patterns between sarcasm/hyperbole and actual violations.
    Requires nuanced understanding of context.
    
    Reward logic:
    - Exact match: 1.0
    - Decision correct: 0.6
    - Safe intervention (Flag vs Remove when Remove needed): 0.4
    - Wrong decision: 0.0 (or negative penalties)
    
    Args:
        predicted_action: The AI's decision
        ground_truth_data: Dataset item with ground truth decision
        
    Returns:
        Score between 0.0 and 1.0
    """
    ground_truth = Decision(ground_truth_data.get("ground_truth", "ALLOW"))
    ground_category = ContentCategory(ground_truth_data.get("category", "SAFE"))
    reports_count = ground_truth_data.get("metadata", {}).get("reports_count", 0)
    
    # Exact match: full credit
    if predicted_action.decision == ground_truth and predicted_action.content_category == ground_category:
        return 1.0
    
    # Decision correct: partial credit
    if predicted_action.decision == ground_truth:
        return 0.6
    
    # Conservative intervention on reported content
    if (reports_count > 5 and 
        predicted_action.decision in [Decision.FLAG, Decision.REMOVE] and 
        ground_truth != Decision.ALLOW):
        return 0.4
    
    # Wrong escalation (escalate when clear allow/remove)
    if predicted_action.decision == Decision.ESCALATE:
        if ground_truth == Decision.ALLOW:
            return -0.2  # Unnecessary escalation
        else:
            return 0.2  # Conservative but not ideal
    
    # Completely wrong
    return 0.0


if __name__ == "__main__":
    print("Medium task grader module loaded")
