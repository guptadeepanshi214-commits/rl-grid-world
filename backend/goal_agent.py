"""
Goal-Based Agent Implementation (Baseline Pathfinding)
NOTE: This is a classical search / planning agent (BFS), NOT a Reinforcement Learning algorithm.
It serves as an upper-bound baseline with full environmental model knowledge.
"""

from collections import deque
from typing import Tuple, List, Dict, Any, Optional
from backend.environment import GridWorldEnv, ACTION_NAMES, ACTION_DELTAS


class GoalBasedAgent:
    """
    Goal-Based Agent using Breadth-First Search (BFS) pathfinding.
    
    CRITICAL DISTINCTION:
    This is NOT a reinforcement learning agent. It does not learn via trial and error.
    Instead, it assumes full deterministic knowledge of the map (transition model & obstacles)
    to compute an optimal shortest path to the goal state. It is used as a benchmark baseline.
    """

    def __init__(self):
        self.algorithm_name = "Goal-Based Agent (BFS Baseline)"
        self.policy_type = "Model-Based Classical Search (Non-RL Baseline)"
        self.total_updates = 0
        self.episodes_trained = 0

    def solve_path(self, env: GridWorldEnv) -> Dict[str, Any]:
        """
        Compute the shortest valid path from start to goal avoiding obstacles using BFS.
        """
        start = env.start_pos
        goal = env.goal_pos

        if start == goal:
            return {
                "path": [list(start)],
                "actions": [],
                "steps": 0,
                "total_reward": 100.0,
                "success": True,
                "visited_nodes": 1
            }

        queue = deque([start])
        came_from: Dict[Tuple[int, int], Tuple[Optional[Tuple[int, int]], Optional[int]]] = {
            start: (None, None)
        }
        visited_nodes = 0

        found = False
        while queue:
            curr = queue.popleft()
            visited_nodes += 1

            if curr == goal:
                found = True
                break

            # Check all 4 adjacent actions
            for action, (dr, dc) in ACTION_DELTAS.items():
                nxt = (curr[0] + dr, curr[1] + dc)
                if env.is_valid_coord(nxt[0], nxt[1]) and not env.is_obstacle(nxt[0], nxt[1]):
                    if nxt not in came_from:
                        came_from[nxt] = (curr, action)
                        queue.append(nxt)

        if not found:
            return {
                "path": [list(start)],
                "actions": [],
                "steps": 0,
                "total_reward": -200.0,
                "success": False,
                "visited_nodes": visited_nodes,
                "error": "No valid obstacle-free path exists to goal"
            }

        # Reconstruct path and actions
        curr = goal
        rev_path = []
        rev_actions = []

        while curr is not None:
            rev_path.append(list(curr))
            parent, act = came_from[curr]
            if act is not None:
                rev_actions.append(ACTION_NAMES[act])
            curr = parent

        path = list(reversed(rev_path))
        actions = list(reversed(rev_actions))
        steps = len(actions)
        # Reward: 100 for goal + (-1) per step
        total_reward = 100.0 + (steps * -1.0)

        return {
            "path": path,
            "actions": actions,
            "steps": steps,
            "total_reward": round(total_reward, 2),
            "success": True,
            "visited_nodes": visited_nodes
        }

    def extract_optimal_path(self, env: GridWorldEnv, max_steps: int = 150) -> Dict[str, Any]:
        """Interface compatibility with RL agents."""
        return self.solve_path(env)

    def get_policy(self, env: GridWorldEnv) -> List[List[Dict[str, Any]]]:
        """
        Generate policy grid representation based on BFS gradient towards goal.
        """
        # Run reverse BFS from goal to compute distance and direction from every cell
        dist = {env.goal_pos: 0}
        best_action = {}
        queue = deque([env.goal_pos])

        while queue:
            curr = queue.popleft()
            curr_dist = dist[curr]

            for action, (dr, dc) in ACTION_DELTAS.items():
                # Opposite delta
                neighbor = (curr[0] - dr, curr[1] - dc)
                if env.is_valid_coord(neighbor[0], neighbor[1]) and not env.is_obstacle(neighbor[0], neighbor[1]):
                    if neighbor not in dist:
                        dist[neighbor] = curr_dist + 1
                        best_action[neighbor] = action
                        queue.append(neighbor)

        policy_grid = []
        for r in range(env.rows):
            row_data = []
            for c in range(env.cols):
                pos = (r, c)
                is_obstacle = env.is_obstacle(r, c)
                is_goal = pos == env.goal_pos
                is_start = pos == env.start_pos
                act = best_action.get(pos, 0)
                cell_dist = dist.get(pos, 999)
                pseudo_value = round(100.0 - cell_dist, 2) if cell_dist < 999 else 0.0

                row_data.append({
                    "row": r,
                    "col": c,
                    "is_obstacle": is_obstacle,
                    "is_goal": is_goal,
                    "is_start": is_start,
                    "best_action": act,
                    "best_action_name": ACTION_NAMES[act],
                    "max_q": pseudo_value,
                    "q_values": [0.0, 0.0, 0.0, 0.0]
                })
            policy_grid.append(row_data)
        return policy_grid

    def to_dict(self) -> Dict[str, Any]:
        """Serialize baseline metadata."""
        return {
            "algorithm": self.algorithm_name,
            "policy_type": self.policy_type,
            "is_reinforcement_learning": False,
            "search_algorithm": "Breadth-First Search (BFS)",
            "knowledge_requirement": "Full Environment Model (Known Transitions & Obstacles)"
        }
