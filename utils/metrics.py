"""
MASOS - Metrics tracker for training and evaluation.
Tracks: total_found, coverage, episode_reward, episode_length.
"""
import numpy as np
from typing import Dict, List, Optional
from collections import deque


class MetricsTracker:
    """
    Tracks and aggregates metrics over episodes.

    Maintains rolling window statistics for logging.

    Args:
        window_size: Number of recent episodes for rolling average
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.targets_found = deque(maxlen=window_size)
        self.coverages = deque(maxlen=window_size)
        self.total_episodes = 0

    def add_episode(
        self,
        total_reward: float,
        episode_length: int,
        targets_found: int,
        coverage: float,
    ):
        """Record metrics for a completed episode."""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        self.targets_found.append(targets_found)
        self.coverages.append(coverage)
        self.total_episodes += 1

    def get_stats(self) -> Dict[str, float]:
        """
        Get current rolling average statistics.

        Returns:
            Dictionary of metric name -> value
        """
        if len(self.episode_rewards) == 0:
            return {
                "avg_reward": 0.0,
                "avg_length": 0.0,
                "avg_found": 0.0,
                "avg_coverage": 0.0,
                "max_found": 0.0,
                "total_episodes": 0,
            }

        return {
            "avg_reward": float(np.mean(self.episode_rewards)),
            "avg_length": float(np.mean(self.episode_lengths)),
            "avg_found": float(np.mean(self.targets_found)),
            "avg_coverage": float(np.mean(self.coverages)),
            "max_found": float(np.max(self.targets_found)),
            "total_episodes": self.total_episodes,
        }

    def reset(self):
        """Reset all tracked metrics."""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.targets_found.clear()
        self.coverages.clear()
        self.total_episodes = 0
