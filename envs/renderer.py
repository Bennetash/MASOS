"""
MASOS - Matplotlib-based renderer for the GridWorld environment.
Provides visualization of agent positions, targets, obstacles, and explored areas.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from typing import Optional


class GridWorldRenderer:
    """Renders the grid world environment using matplotlib."""

    AGENT_COLORS = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
        "#f58231", "#911eb4", "#42d4f4", "#f032e6",
    ]

    def __init__(self, grid_size: int, figsize: tuple = (8, 8)):
        self.grid_size = grid_size
        self.figsize = figsize
        self.fig = None
        self.ax = None

    def render(self, env, title: str = "", save_path: Optional[str] = None):
        """
        Render the current state of the environment.

        Args:
            env: GridWorldEnv instance
            title: Title string for the plot
            save_path: If provided, save the figure to this path
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize)

        self.ax.clear()
        gs = self.grid_size

        # Background: explored vs unexplored
        display = np.zeros((gs, gs, 3), dtype=np.float32)
        # Unexplored = light gray
        display[:, :] = [0.9, 0.9, 0.9]
        # Explored = white
        explored = env.grid_explored
        display[explored] = [1.0, 1.0, 1.0]

        # Obstacles = dark gray
        for obs in env.obstacles:
            for r, c in obs.cells:
                display[r, c] = [0.3, 0.3, 0.3]

        # Targets = green dots
        for target in env.targets:
            if not target.found:
                r, c = target.position
                display[r, c] = [0.0, 0.8, 0.0]

        self.ax.imshow(display, origin="upper", interpolation="nearest")

        # Draw agents as colored circles
        for agent in env.agents:
            r, c = agent.position
            color = self.AGENT_COLORS[agent.id % len(self.AGENT_COLORS)]
            circle = plt.Circle((c, r), 0.4, color=color, zorder=5)
            self.ax.add_patch(circle)
            self.ax.text(c, r, str(agent.id), ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold", zorder=6)

        # Title and info
        coverage = np.sum(explored) / (gs * gs) * 100
        info_str = (f"{title}  Step: {env.step_count}  "
                    f"Found: {env.total_found}/{env.n_targets}  "
                    f"Coverage: {coverage:.1f}%")
        self.ax.set_title(info_str, fontsize=10)
        self.ax.set_xlim(-0.5, gs - 0.5)
        self.ax.set_ylim(gs - 0.5, -0.5)
        self.ax.set_aspect("equal")
        self.ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=[0.9, 0.9, 0.9], label="Unexplored"),
            mpatches.Patch(facecolor="white", edgecolor="black", label="Explored"),
            mpatches.Patch(facecolor=[0.3, 0.3, 0.3], label="Obstacle"),
            mpatches.Patch(facecolor=[0.0, 0.8, 0.0], label="Target"),
        ]
        self.ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

        if save_path:
            self.fig.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.pause(0.01)

    def close(self):
        """Close the matplotlib figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
