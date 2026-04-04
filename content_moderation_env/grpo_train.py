#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) training script (optional).
Demonstrates how to use the environment for policy training.
"""

import json
import logging
from typing import Dict, List, Tuple
from inference import ModerationInferenceEngine
from env import ContentModerationEnv
from models import ModerationAction, Decision

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer for moderation policies.
    Uses the environment to evaluate and improve moderation decisions.
    """
    
    def __init__(self, num_epochs: int = 10, batch_size: int = 32):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.env = ContentModerationEnv()
        self.engine = ModerationInferenceEngine()
        self.training_history: List[Dict] = []
    
    def collect_trajectory(self, max_steps: int = None) -> Tuple[List[Dict], float]:
        """
        Collect a trajectory of states, actions, and rewards from the environment.
        
        Args:
            max_steps: Maximum number of steps to collect
            
        Returns:
            Tuple of (trajectory list, total reward)
        """
        obs = self.env.reset()
        trajectory = []
        total_reward = 0.0
        step = 0
        
        while not self.env.done:
            if max_steps and step >= max_steps:
                break
            
            # Get action from inference engine
            action = self.engine.get_moderation_decision(obs)
            
            # Step environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Store transition
            trajectory.append({
                "obs": obs.model_dump(),
                "action": action.model_dump(),
                "reward": reward,
                "info": info
            })
            
            total_reward += reward
            step += 1
            obs = next_obs
        
        return trajectory, total_reward
    
    def evaluate_policy(self, num_rollouts: int = 5) -> Dict[str, float]:
        """
        Evaluate current policy across multiple rollouts.
        
        Args:
            num_rollouts: Number of trajectories to collect
            
        Returns:
            Dictionary with evaluation metrics
        """
        rewards = []
        trajectory_lengths = []
        
        for _ in range(num_rollouts):
            trajectory, total_reward = self.collect_trajectory()
            rewards.append(total_reward)
            trajectory_lengths.append(len(trajectory))
        
        avg_reward = sum(rewards) / len(rewards)
        avg_length = sum(trajectory_lengths) / len(trajectory_lengths)
        
        return {
            "avg_reward": avg_reward,
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "avg_trajectory_length": avg_length
        }
    
    def train(self):
        """
        Run GRPO training loop.
        
        This is a demonstration of how the environment can be used for training.
        In practice, you would implement actual policy gradients and optimization.
        """
        logger.info(f"Starting GRPO training for {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Collect batch of trajectories
            batch_rewards = []
            batch_trajectories = []
            
            for _ in range(self.batch_size):
                trajectory, total_reward = self.collect_trajectory()
                batch_rewards.append(total_reward)
                batch_trajectories.append(trajectory)
            
            # Compute statistics
            avg_reward = sum(batch_rewards) / len(batch_rewards)
            max_reward = max(batch_rewards)
            
            # Log progress
            epoch_data = {
                "epoch": epoch + 1,
                "avg_reward": avg_reward,
                "max_reward": max_reward,
                "batch_size": len(batch_rewards)
            }
            self.training_history.append(epoch_data)
            
            logger.info(f"  Avg reward: {avg_reward:.4f}, Max: {max_reward:.4f}")
            
            # In a real implementation, you would:
            # 1. Compute policy gradients from trajectories
            # 2. Compute advantage estimates (rewards - baseline)
            # 3. Perform policy update using GRPO objective
            # 4. Update value function baseline
        
        logger.info("Training complete")
    
    def save_training_history(self, filepath: str):
        """Save training history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved to {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize trainer
    trainer = GRPOTrainer(num_epochs=3, batch_size=5)
    
    # Evaluate initial policy
    print("Evaluating initial policy...")
    metrics = trainer.evaluate_policy(num_rollouts=3)
    print(f"Initial policy metrics: {metrics}")
    
    # Run training (demonstration)
    print("\nRunning training...")
    trainer.train()
    
    # Save results
    trainer.save_training_history("training_history.json")
    
    print("\n✓ Training complete. Results saved to training_history.json")
