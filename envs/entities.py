"""
MASOS - Entity definitions for the grid world environment.
Agents (searchers), Targets, and Obstacles.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Agent:
    """A searcher UAV in the grid world."""
    id: int
    position: np.ndarray       # shape (2,), integer grid coords [row, col]
    alive: bool = True
    total_reward: float = 0.0

    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.int32)


@dataclass
class Target:
    """A target to be found by agents."""
    id: int
    position: np.ndarray       # shape (2,), integer grid coords [row, col]
    found: bool = False
    movement_mode: str = "static"  # "static" or "random"

    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.int32)


@dataclass
class Obstacle:
    """A fixed obstacle occupying a 2x2 block of cells."""
    id: int
    top_left: np.ndarray       # shape (2,), top-left corner [row, col]

    def __post_init__(self):
        self.top_left = np.array(self.top_left, dtype=np.int32)

    @property
    def cells(self) -> List[Tuple[int, int]]:
        """Return all cells occupied by this 2x2 obstacle."""
        r, c = self.top_left
        return [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
