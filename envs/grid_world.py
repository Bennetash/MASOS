"""
MASOS - Grid World Environment for Multi-Agent Search.
40x40 or 80x80 grid, 8 UAV agents, targets, and obstacles.
Observation: 4-channel 11x11 FOV + 2 normalized position coords = 486-dim.
Actions: 0=up, 1=down, 2=left, 3=right, 4=stay
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from envs.entities import Agent, Target, Obstacle
from configs.default import TrainingConfig


class GridWorldEnv:
    """Multi-agent search and observation environment on a grid."""

    # Action mapping: action_id -> (row_delta, col_delta)
    ACTION_MAP = {
        0: (-1, 0),   # up
        1: (1, 0),    # down
        2: (0, -1),   # left
        3: (0, 1),    # right
        4: (0, 0),    # stay
    }

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.grid_size = config.grid_size
        self.n_agents = config.n_agents
        self.n_targets = config.n_targets
        self.n_obstacles = config.n_obstacles
        self.obs_radius = config.obs_radius
        self.target_mode = config.target_mode
        self.max_steps = config.max_steps
        self.cooperation_distance = config.cooperation_distance

        # Reward values from Table II
        self.reward_find_target = config.reward_find_target
        self.reward_collide_agent = config.reward_collide_agent
        self.reward_collide_obstacle = config.reward_collide_obstacle
        self.reward_stay = config.reward_stay
        self.reward_move = config.reward_move
        self.reward_cooperation = config.reward_cooperation
        self.reward_boundary_death = config.reward_boundary_death

        # State
        self.agents: List[Agent] = []
        self.targets: List[Target] = []
        self.obstacles: List[Obstacle] = []
        self.obstacle_cells: set = set()
        self.step_count: int = 0
        self.total_found: int = 0

        # Grid layers for fast lookup
        self.grid_agents = None      # grid_size x grid_size, agent id or -1
        self.grid_targets = None     # grid_size x grid_size, bool
        self.grid_obstacles = None   # grid_size x grid_size, bool
        self.grid_explored = None    # grid_size x grid_size, bool

    def reset(self, seed: Optional[int] = None) -> Dict[int, np.ndarray]:
        """Reset the environment and return initial observations."""
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.total_found = 0

        # Initialize grids
        gs = self.grid_size
        self.grid_agents = -np.ones((gs, gs), dtype=np.int32)
        self.grid_targets = np.zeros((gs, gs), dtype=bool)
        self.grid_obstacles = np.zeros((gs, gs), dtype=bool)
        self.grid_explored = np.zeros((gs, gs), dtype=bool)

        # Place obstacles (2x2 blocks) - avoid edges
        self.obstacles = []
        self.obstacle_cells = set()
        placed = 0
        while placed < self.n_obstacles:
            r = np.random.randint(1, gs - 2)
            c = np.random.randint(1, gs - 2)
            cells = [(r, c), (r, c+1), (r+1, c), (r+1, c+1)]
            if any((cr, cc) in self.obstacle_cells for cr, cc in cells):
                continue
            obs = Obstacle(id=placed, top_left=np.array([r, c]))
            self.obstacles.append(obs)
            for cr, cc in cells:
                self.obstacle_cells.add((cr, cc))
                self.grid_obstacles[cr, cc] = True
            placed += 1

        # Place agents - avoid obstacles and each other
        self.agents = []
        occupied = set(self.obstacle_cells)
        for i in range(self.n_agents):
            while True:
                r = np.random.randint(0, gs)
                c = np.random.randint(0, gs)
                if (r, c) not in occupied:
                    break
            agent = Agent(id=i, position=np.array([r, c]))
            self.agents.append(agent)
            occupied.add((r, c))
            self.grid_agents[r, c] = i

        # Place targets - avoid obstacles and agents
        self.targets = []
        for i in range(self.n_targets):
            while True:
                r = np.random.randint(0, gs)
                c = np.random.randint(0, gs)
                if (r, c) not in occupied:
                    break
            target = Target(id=i, position=np.array([r, c]),
                          movement_mode=self.target_mode)
            self.targets.append(target)
            occupied.add((r, c))
            self.grid_targets[r, c] = True

        # Mark initial FOV as explored
        for agent in self.agents:
            self._mark_explored(agent.position)

        return self._get_all_observations()

    def _mark_explored(self, pos: np.ndarray):
        """Mark cells within observation radius as explored."""
        r, c = pos
        rad = self.obs_radius
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    self.grid_explored[nr, nc] = True

    def _get_observation(self, agent: Agent) -> np.ndarray:
        """
        Get local observation for one agent.
        4 channels of 11x11 FOV + 2 normalized position = 486 dim.
        Channels:
            0: other agents (1 where agent present)
            1: targets (1 where target present)
            2: obstacles (1 where obstacle present)
            3: boundary cells (1 where outside grid boundary)
        """
        r, c = agent.position
        rad = self.obs_radius
        fov_size = 2 * rad + 1  # 11

        channels = np.zeros((4, fov_size, fov_size), dtype=np.float32)

        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                nr, nc = r + dr, c + dc
                fr, fc = dr + rad, dc + rad  # FOV coordinates

                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    # Channel 0: other agents
                    aid = self.grid_agents[nr, nc]
                    if aid >= 0 and aid != agent.id:
                        channels[0, fr, fc] = 1.0

                    # Channel 1: targets
                    if self.grid_targets[nr, nc]:
                        channels[1, fr, fc] = 1.0

                    # Channel 2: obstacles
                    if self.grid_obstacles[nr, nc]:
                        channels[2, fr, fc] = 1.0

                    # Channel 3: boundary (always 0 for in-bounds cells)
                    pass
                else:
                    # Out of bounds: mark ONLY as boundary (channel 3)
                    # NOT as obstacle (channel 2) — they are distinct concepts
                    # Obstacle = rebounce + penalty, Boundary = death
                    channels[3, fr, fc] = 1.0

        # Flatten channels and append normalized position
        flat = channels.flatten()  # 4 * 11 * 11 = 484
        norm_pos = np.array([r / self.grid_size, c / self.grid_size],
                           dtype=np.float32)
        obs = np.concatenate([flat, norm_pos])  # 486
        return obs

    def _get_all_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for all agents. Dead agents get zero obs."""
        obs = {}
        for agent in self.agents:
            if agent.alive:
                obs[agent.id] = self._get_observation(agent)
            else:
                obs[agent.id] = np.zeros(self.config.obs_dim, dtype=np.float32)
        return obs

    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],  # observations
        Dict[int, float],       # rewards
        Dict[str, bool],         # dones: {"agent_0": bool, ..., "__all__": bool}
        Dict                     # info
    ]:
        """
        Execute one environment step.

        Args:
            actions: Dict mapping agent_id -> action (0-4)

        Returns:
            observations, rewards, done, info
        """
        self.step_count += 1
        rewards = {agent.id: 0.0 for agent in self.agents}
        agent_dones = {i: False for i in range(self.n_agents)}

        # Compute intended new positions
        intended = {}
        for agent in self.agents:
            if not agent.alive:
                intended[agent.id] = (agent.position[0], agent.position[1], 4)
                continue
            act = actions.get(agent.id, 4)  # default stay
            dr, dc = self.ACTION_MAP[act]
            new_r = agent.position[0] + dr
            new_c = agent.position[1] + dc
            intended[agent.id] = (new_r, new_c, act)

        # Resolve movements
        new_positions = {}
        for agent in self.agents:
            new_r, new_c, act = intended[agent.id]

            # Movement cost
            if act == 4:
                rewards[agent.id] += self.reward_stay
            else:
                rewards[agent.id] += self.reward_move

            # Check boundary -- agent DIES (paper: boundary death mechanic)
            if new_r < 0 or new_r >= self.grid_size or \
               new_c < 0 or new_c >= self.grid_size:
                agent.alive = False
                agent_dones[agent.id] = True
                rewards[agent.id] += self.reward_boundary_death  # -12 penalty
                old_r, old_c = agent.position
                self.grid_agents[old_r, old_c] = -1
                new_positions[agent.id] = tuple(agent.position)
                continue

            # Check obstacle collision
            if (new_r, new_c) in self.obstacle_cells:
                rewards[agent.id] += self.reward_collide_obstacle
                new_positions[agent.id] = tuple(agent.position)
                continue

            new_positions[agent.id] = (new_r, new_c)

        # Check agent-agent collisions (two agents moving to same cell)
        pos_to_agents = {}
        for aid, pos in new_positions.items():
            pos_to_agents.setdefault(pos, []).append(aid)

        for pos, agent_ids in pos_to_agents.items():
            alive_ids = [aid for aid in agent_ids if self.agents[aid].alive]
            if len(alive_ids) > 1:
                # Collision - all agents stay at original position
                for aid in alive_ids:
                    rewards[aid] += self.reward_collide_agent
                    new_positions[aid] = tuple(self.agents[aid].position)

        # Apply movements
        for agent in self.agents:
            if not agent.alive:
                continue
            old_r, old_c = agent.position
            new_r, new_c = new_positions[agent.id]
            if (old_r, old_c) != (new_r, new_c):
                self.grid_agents[old_r, old_c] = -1
                self.grid_agents[new_r, new_c] = agent.id
                agent.position = np.array([new_r, new_c], dtype=np.int32)

            # Mark explored
            self._mark_explored(agent.position)

        # Check target finding + cooperation reward (Table II)
        # Cooperation: when agent finds target, nearby allies (Manhattan <= 3) also get +3
        for agent in self.agents:
            if not agent.alive:
                continue
            r, c = agent.position
            if self.grid_targets[r, c]:
                # Find the target at this position
                for target in self.targets:
                    if (not target.found
                            and target.position[0] == r
                            and target.position[1] == c):
                        target.found = True
                        self.grid_targets[r, c] = False
                        self.total_found += 1
                        rewards[agent.id] += self.reward_find_target

                        # Cooperation reward: flat bonus for finder + each nearby ally
                        nearby_allies = []
                        for other in self.agents:
                            if other.id == agent.id:
                                continue
                            dist = (abs(agent.position[0] - other.position[0])
                                    + abs(agent.position[1] - other.position[1]))
                            if dist <= self.cooperation_distance:
                                nearby_allies.append(other.id)
                        if nearby_allies:
                            # Finder gets flat cooperation bonus (once, not per-ally)
                            rewards[agent.id] += self.reward_cooperation
                            # Each nearby ally also gets one cooperation bonus
                            for ally_id in nearby_allies:
                                rewards[ally_id] += self.reward_cooperation
                        break

        # Move random targets
        if self.target_mode == "random":
            self._move_targets()

        # Accumulate rewards
        for agent in self.agents:
            agent.total_reward += rewards[agent.id]

        # Check done
        all_found = all(t.found for t in self.targets)
        time_up = self.step_count >= self.max_steps
        all_dead = not any(a.alive for a in self.agents)
        global_done = all_found or time_up or all_dead

        # Per-agent dones dict (PettingZoo-style)
        dones = {}
        for i in range(self.n_agents):
            dones[f"agent_{i}"] = agent_dones[i] or global_done
        dones["__all__"] = global_done

        # Compute coverage
        coverage = np.sum(self.grid_explored) / (self.grid_size ** 2)

        info = {
            "step": self.step_count,
            "total_found": self.total_found,
            "coverage": float(coverage),
            "all_found": all_found,
            "n_alive": sum(1 for a in self.agents if a.alive),
        }

        observations = self._get_all_observations()
        return observations, rewards, dones, info

    def _move_targets(self):
        """Move unfound random targets by one step in a random direction."""
        for target in self.targets:
            if target.found or target.movement_mode == "static":
                continue
            # Random direction (including stay)
            act = np.random.randint(0, 5)
            dr, dc = self.ACTION_MAP[act]
            new_r = target.position[0] + dr
            new_c = target.position[1] + dc

            if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
                if (new_r, new_c) not in self.obstacle_cells and \
                   not self.grid_targets[new_r, new_c]:
                    old_r, old_c = target.position
                    self.grid_targets[old_r, old_c] = False
                    target.position = np.array([new_r, new_c], dtype=np.int32)
                    self.grid_targets[new_r, new_c] = True

    def get_state(self) -> Dict:
        """Get full environment state (for centralized critic)."""
        return {
            "agent_positions": np.array([a.position for a in self.agents]),
            "target_positions": np.array([t.position for t in self.targets
                                         if not t.found]),
            "grid_explored": self.grid_explored.copy(),
            "step": self.step_count,
            "total_found": self.total_found,
        }

    @property
    def observation_space_dim(self) -> int:
        return self.config.obs_dim

    @property
    def action_space_dim(self) -> int:
        return self.config.act_dim
