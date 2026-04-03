import json
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple
from models import ModerationObservation, ModerationAction, Decision, UserMetadata

logger = logging.getLogger(__name__)

class BaseTask(ABC):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.posts = []
        self._load_dataset()
        self.current_idx = 0
        self.done = False
        
        # Grading Metrics
        self.total_cases = len(self.posts)
        self.correct_decisions = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.escalations = 0

    def _load_dataset(self):
        try:
            with open(self.dataset_path, 'r') as f:
                self.posts = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_path}: {e}")
            self.posts = []

    def get_remaining_steps(self) -> int:
        return len(self.posts) - self.current_idx

    def get_current_observation(self) -> ModerationObservation:
        if self.current_idx >= len(self.posts):
            # Dummy observation
            return ModerationObservation(
                post_id="DONE", post_body="DONE", 
                metadata=UserMetadata(timestamp="2026-04-03T00:00:00Z", reports_count=0, author_trust_score=1.0, account_age_days=0),
                context=[]
            )
        data = self.posts[self.current_idx]
        return ModerationObservation(
            post_id=data["post_id"],
            post_body=data["post_body"],
            metadata=UserMetadata(**data["metadata"]),
            context=data["context"]
        )

    def process_action(self, action: ModerationAction) -> Tuple[float, str]:
        """ Processes the action and returns a dense step reward with reasoning """
        if self.done:
            return 0.0, "Task already complete."
            
        data = self.posts[self.current_idx]
        ground_truth = Decision(data["ground_truth"])
        act = action.decision
        
        step_reward, reason = self._compute_step_reward(act, ground_truth)
        
        self.current_idx += 1
        if self.current_idx >= len(self.posts):
            self.done = True
            
        return step_reward, reason

    @abstractmethod
    def _compute_step_reward(self, act: Decision, ground_truth: Decision) -> Tuple[float, str]:
        pass

    @abstractmethod
    def get_final_score(self) -> float:
        """ Returns the deterministic [0.0, 1.0] score preventing exploits """
        pass

class Task1Easy(BaseTask):
    def _compute_step_reward(self, act: Decision, ground_truth: Decision) -> Tuple[float, str]:
        if act == ground_truth:
            self.correct_decisions += 1
            return 1.0, "Correct clear decision."
        if ground_truth == Decision.ALLOW and act == Decision.REMOVE:
            self.false_positives += 1
            return -1.0, "Severe Penalty: Removed explicitly safe content."
        return -0.5, "Incorrect spam classification."

    def get_final_score(self) -> float:
        # Prevent "Delete All" exploit
        if self.false_positives > 0:
            return 0.0
        return max(0.0, self.correct_decisions / max(1, self.total_cases))

class Task2Medium(BaseTask):
    def _compute_step_reward(self, act: Decision, ground_truth: Decision) -> Tuple[float, str]:
        if act == ground_truth:
            self.correct_decisions += 1
            return 1.0, "Correct contextual decision."
        
        if ground_truth == Decision.REMOVE and act == Decision.ALLOW:
            self.false_negatives += 1
            return -2.5, "Allowed harmful context."
        elif ground_truth == Decision.ALLOW and act == Decision.REMOVE:
            self.false_positives += 1
            return -1.0, "Censorship penalty."
        elif act == Decision.FLAG:
            return 0.2, "Partial credit: Cautious but failed to definitively act."
        
        return 0.0, "Incorrect decision."

    def get_final_score(self) -> float:
        # Weighted precision penalty
        penalty = (self.false_positives * 0.5) + (self.false_negatives * 0.2)
        score = 1.0 - penalty
        return max(0.0, min(1.0, score))

class Task3Hard(BaseTask):
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        self.optimal_escalations = 0

    def _compute_step_reward(self, act: Decision, ground_truth: Decision) -> Tuple[float, str]:
        if act == Decision.ESCALATE:
            self.escalations += 1
            if ground_truth == Decision.ESCALATE:
                self.optimal_escalations += 1
                return 1.5, "Optimal escalation of extreme ambiguity."
            return -0.5, "Trivial escalation. Missed deterministic context."
            
        if act == ground_truth:
            self.correct_decisions += 1
            return 1.0, "Correct assertive decision."
        
        return -1.0, "Misguided confidence on complex cases."

    def get_final_score(self) -> float:
        raw_score = (self.correct_decisions + (self.optimal_escalations * 0.8)) / max(1, self.total_cases)
        escalation_rate = self.escalations / max(1, self.total_cases)
        
        # Logarithmic exhaustion penalty if escalating >30%
        if escalation_rate > 0.3:
            penalty = (escalation_rate - 0.3) * 2.0
            raw_score -= penalty
            
        return max(0.0, min(1.0, raw_score))
