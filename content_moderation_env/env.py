import logging
from typing import Tuple, Dict, Any
from models import ModerationObservation, ModerationAction, ModerationReward
from tasks import Task1Easy, Task2Medium, Task3Hard

logger = logging.getLogger(__name__)

class ContentModerationEnv:
    """
    OpenEnv-compliant multi-task environment for AI Content Moderation.
    """
    def __init__(self):
        self._initialize_tasks()
        self.current_task_idx = 0
        self.done = False

    def _initialize_tasks(self):
        self.tasks = [
            Task1Easy(dataset_path="datasets/easy_spam.json"),
            Task2Medium(dataset_path="datasets/medium_toxic.json"),
            Task3Hard(dataset_path="datasets/hard_mixed.json")
        ]

    def reset(self) -> ModerationObservation:
        self._initialize_tasks()
        self.current_task_idx = 0
        self.done = False
        return self._get_current_observation()

    def _get_current_observation(self) -> ModerationObservation:
        if self.current_task_idx >= len(self.tasks):
            return self.tasks[-1].get_current_observation() # Dummy
        return self.tasks[self.current_task_idx].get_current_observation()

    def step(self, action: ModerationAction) -> Tuple[ModerationObservation, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Environment is done. Please call reset().")

        current_task = self.tasks[self.current_task_idx]
        
        # Process action via task-specific grader logic
        step_reward, reason = current_task.process_action(action)
        
        # Analyze confidence
        if action.confidence_score > 0.9 and step_reward < 0:
            step_reward -= 0.3
            reason += " | -0.3 Penalty: Dangerously overconfident."
            
        info = {
            "reward_details": ModerationReward(score=step_reward, reason=reason).model_dump(),
            "active_task": current_task.__class__.__name__
        }

        # Check if current task exhausted
        if current_task.done:
            # Inject final score of the finished task into info
            info["task_final_score"] = current_task.get_final_score()
            self.current_task_idx += 1
            
        if self.current_task_idx >= len(self.tasks):
            self.done = True

        return self._get_current_observation(), step_reward, self.done, info

    def state(self) -> Dict[str, Any]:
        return {
            "active_task_idx": self.current_task_idx,
            "done": self.done,
            "task_scores": [t.get_final_score() if t.done else None for t in self.tasks]
        }
