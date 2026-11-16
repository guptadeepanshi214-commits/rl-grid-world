"""
Grid World Navigation Environment
Sequential Decision Making Environment for Reinforcement Learning
"""

import copy
from typing import Tuple, List, Dict, Any, Optional

# Action definitions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTION_NAMES = {
    ACTION_UP: "UP",
    ACTION_DOWN: "DOWN",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT"
}

ACTION_DELTAS = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1)
}

# Reward specifications as requested
REWARD_GOAL = 100.0
REWARD_OBSTACLE = -10.0
REWARD_STEP = -1.0
REWARD_INVALID_MOVE = -5.0


def get_preset_obstacles(rows: int, cols: int, preset_name: str = "default") -> List[List[int]]:
    """Return obstacle coordinate lists for standard presets."""
    if preset_name == "empty":
        return []
    
    if rows == 10 and cols == 10:
        if preset_name == "default":
            # Challenging maze-like structure with multiple routes and narrow corridors
            return [
                [1, 2], [2, 2], [3, 2], [4, 2],
                [1, 6], [2, 6], [3, 6],
                [5, 4], [5, 5], [5, 6], [5, 7],
                [7, 1], [7, 2], [7, 3],
                [7, 6], [7, 7], [7, 8],
                [3, 8], [4, 8],
                [8, 4]
            ]
        elif preset_name == "four_rooms":
            # Classic 4-room reinforcement learning benchmark with doorways
            obstacles = []
            mid_r, mid_c = rows // 2, cols // 2
            # Horizontal wall with doorways at col 2 and col 7
            for c in range(cols):
                if c not in (2, 7):
                    obstacles.append([mid_r, c])
            # Vertical wall with doorways at row 2 and row 7
            for r in range(rows):
                if r not in (2, 7) and [r, mid_c] not in obstacles:
                    obstacles.append([r, mid_c])
            return obstacles
        elif preset_name == "bottleneck":
            # Narrow bottleneck in the middle
            obstacles = []
            for r in range(rows):
                if r != 4:  # single passage at row 4
                    obstacles.append([r, 5])
            return obstacles
        elif preset_name == "cliff":
            # Cliff-walking setup: obstacles along bottom
            obstacles = []
            for c in range(1, cols - 1):
                obstacles.append([rows - 1, c])
            return obstacles

    # Generic generated obstacle layout for other grid dimensions
    obstacles = []
    # Vertical divider with opening
    mid_c = cols // 2
    for r in range(rows):
        if r not in (1, rows - 2):
            obstacles.append([r, mid_c])
    return obstacles


class GridWorldEnv:
    """
    A 2D Grid World navigation environment adhering to standard MDP principles.
    States: (row, col) coordinates or flattened integer indices.
    Actions: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT.
    """

    def __init__(
        self,
        rows: int = 10,
        cols: int = 10,
        start_pos: Tuple[int, int] = (0, 0),
        goal_pos: Optional[Tuple[int, int]] = None,
        obstacles: Optional[List[List[int]]] = None,
        preset: str = "default"
    ):
        self.rows = max(4, min(25, int(rows)))
        self.cols = max(4, min(25, int(cols)))
        self.start_pos = (int(start_pos[0]), int(start_pos[1]))
        
        if goal_pos is None:
            self.goal_pos = (self.rows - 1, self.cols - 1)
        else:
            self.goal_pos = (int(goal_pos[0]), int(goal_pos[1]))

        if obstacles is not None:
            self.obstacles = set((int(o[0]), int(o[1])) for o in obstacles)
        else:
            raw_obs = get_preset_obstacles(self.rows, self.cols, preset)
            self.obstacles = set((int(o[0]), int(o[1])) for o in raw_obs)

        # Ensure start and goal are never obstacles
        self.obstacles.discard(self.start_pos)
        self.obstacles.discard(self.goal_pos)

        # Internal agent state
        self.agent_pos = self.start_pos
        self.steps_taken = 0
        self.total_reward = 0.0
        self.done = False

    def reset(self) -> Tuple[int, int]:
        """Reset environment to the initial state."""
        self.agent_pos = self.start_pos
        self.steps_taken = 0
        self.total_reward = 0.0
        self.done = False
        return self.agent_pos

    def is_valid_coord(self, r: int, c: int) -> bool:
        """Check if coordinates fall within grid boundaries."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_obstacle(self, r: int, c: int) -> bool:
        """Check if coordinates contain an obstacle."""
        return (r, c) in self.obstacles

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.
        
        Returns:
            next_state: (r, c)
            reward: float
            done: bool
            info: dict with step diagnostics
        """
        if self.done:
            return self.agent_pos, 0.0, True, {"status": "already_done"}

        if action not in ACTION_DELTAS:
            raise ValueError(f"Invalid action {action}. Valid actions are {list(ACTION_DELTAS.keys())}")

        dr, dc = ACTION_DELTAS[action]
        curr_r, curr_c = self.agent_pos
        target_r = curr_r + dr
        target_c = curr_c + dc
        self.steps_taken += 1

        info = {
            "action": ACTION_NAMES[action],
            "from_pos": [curr_r, curr_c],
            "target_pos": [target_r, target_c],
            "hit_obstacle": False,
            "hit_boundary": False,
            "reached_goal": False
        }

        # Case 1: Out of bounds (invalid movement)
        if not self.is_valid_coord(target_r, target_c):
            reward = REWARD_INVALID_MOVE
            # Agent remains at current position
            info["hit_boundary"] = True
            info["status"] = "hit_boundary"
            self.total_reward += reward
            return self.agent_pos, reward, False, info

        # Case 2: Hit obstacle
        if self.is_obstacle(target_r, target_c):
            reward = REWARD_OBSTACLE
            # Agent is blocked and stays at current position
            info["hit_obstacle"] = True
            info["status"] = "hit_obstacle"
            self.total_reward += reward
            return self.agent_pos, reward, False, info

        # Case 3: Reached Goal
        if (target_r, target_c) == self.goal_pos:
            self.agent_pos = (target_r, target_c)
            reward = REWARD_GOAL
            self.done = True
            info["reached_goal"] = True
            info["status"] = "reached_goal"
            self.total_reward += reward
            return self.agent_pos, reward, True, info

        # Case 4: Normal valid step
        self.agent_pos = (target_r, target_c)
        reward = REWARD_STEP
        info["status"] = "valid_step"
        self.total_reward += reward
        return self.agent_pos, reward, False, info

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Map 2D state tuple to 1D integer index."""
        return state[0] * self.cols + state[1]

    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Map 1D integer index to 2D state tuple."""
        return (index // self.cols, index % self.cols)

    @property
    def num_states(self) -> int:
        """Total number of discrete states."""
        return self.rows * self.cols

    @property
    def num_actions(self) -> int:
        """Total number of discrete actions."""
        return 4

    def set_layout(
        self,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        start_pos: Optional[List[int]] = None,
        goal_pos: Optional[List[int]] = None,
        obstacles: Optional[List[List[int]]] = None
    ) -> None:
        """Update grid dimensions, positions, and obstacles."""
        if rows is not None:
            self.rows = max(4, min(25, int(rows)))
        if cols is not None:
            self.cols = max(4, min(25, int(cols)))
        if start_pos is not None:
            self.start_pos = (int(start_pos[0]), int(start_pos[1]))
        if goal_pos is not None:
            self.goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
        if obstacles is not None:
            self.obstacles = set((int(o[0]), int(o[1])) for o in obstacles)

        # Boundary checks
        sr, sc = self.start_pos
        self.start_pos = (max(0, min(self.rows - 1, sr)), max(0, min(self.cols - 1, sc)))
        gr, gc = self.goal_pos
        self.goal_pos = (max(0, min(self.rows - 1, gr)), max(0, min(self.cols - 1, gc)))

        self.obstacles.discard(self.start_pos)
        self.obstacles.discard(self.goal_pos)
        self.reset()

    def toggle_obstacle(self, r: int, c: int) -> bool:
        """Toggle an obstacle at (r, c). Cannot be placed at start or goal."""
        pos = (int(r), int(c))
        if pos == self.start_pos or pos == self.goal_pos:
            return False
        if not self.is_valid_coord(pos[0], pos[1]):
            return False
        is_added = False
        if pos in self.obstacles:
            self.obstacles.remove(pos)
            is_added = False
        else:
            self.obstacles.add(pos)
            is_added = True
        self.reset()
        return is_added

    def to_dict(self) -> Dict[str, Any]:
        """Serialize environment configuration for API responses."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "start": list(self.start_pos),
            "goal": list(self.goal_pos),
            "obstacles": [list(o) for o in sorted(self.obstacles)],
            "agent_pos": list(self.agent_pos),
            "steps": self.steps_taken,
            "total_reward": self.total_reward,
            "done": self.done,
            "num_states": self.num_states,
            "num_actions": self.num_actions
        }
