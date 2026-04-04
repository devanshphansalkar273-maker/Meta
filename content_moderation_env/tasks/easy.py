"""Easy task grader for spam detection."""

from typing import Dict, Any
from models import ModerationAction, Decision, ContentCategory


def grade_easy_task(predicted_action: ModerationAction, ground_truth_data: Dict[str, Any]) -> float:
    """
    Grade easy spam detection task.
    
    Easy tasks have clear spam patterns with high confidence requirements.
    Reward: 1.0 for exact match, 0.5 for decision-only match, 0.0 for mismatch.
    
    Args:
        predicted_action: The AI's decision
        ground_truth_data: Dataset item with ground truth decision
        
    Returns:
        Score between 0.0 and 1.0
    """
    ground_truth = Decision(ground_truth_data.get("ground_truth", "ALLOW"))
    ground_category = ContentCategory(ground_truth_data.get("category", "SAFE"))
    
    # Perfect match: decision and category both correct
    if predicted_action.decision == ground_truth and predicted_action.content_category == ground_category:
        return 1.0
    
    # Partial match: decision correct but category wrong
    if predicted_action.decision == ground_truth:
        return 0.7
    
    # Weak match: intervention attempted but wrong decision
    if (predicted_action.decision != Decision.ALLOW) and (ground_truth != Decision.ALLOW):
        return 0.3
    
    # No match
    return 0.0


if __name__ == "__main__":
    print("Easy task grader module loaded")
