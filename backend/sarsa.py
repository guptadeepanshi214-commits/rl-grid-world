"""
SARSA Agent Implementation
On-Policy Temporal Difference (TD) Reinforcement Learning (State-Action-Reward-State-Action)
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from backend.environment import GridWorldEnv, ACTION_NAMES


class SarsaAgent:
    """
    SARSA Agent (Rummery & Niranjan, 1994)
    On-Policy TD Control Algorithm.
    
    Update Equation:
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * Q(s', a') - Q(s, a) ]
    where a' is the action chosen from state s' using the current epsilon-greedy policy.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01
    ):
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.alpha = float(learning_rate)
        self.gamma = float(discount_factor)
        self.initial_epsilon = float(epsilon)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.min_epsilon = float(min_epsilon)

        # Initialize Q-table: Shape (num_states, num_actions)
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=np.float64)

        # Diagnostics & Tracking
        self.episodes_trained = 0
        self.total_updates = 0
        self.algorithm_name = "SARSA"
        self.policy_type = "On-Policy TD Control"

    def choose_action(self, state_idx: int, greedy: bool = False) -> int:
        """
        Select action using Epsilon-Greedy strategy.
        If greedy is True, selects argmax Q(s, a) with tie-breaking.
        """
        if not greedy and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.num_actions))

        q_values = self.q_table[state_idx]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return int(np.random.choice(best_actions))

    def update(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        next_action: int,
        done: bool
    ) -> float:
        """
        Apply SARSA on-policy update equation:
        Target = r if done else r + gamma * Q(s', a')
        TD_Error = Target - Q(s, a)
        Q(s, a) <- Q(s, a) + alpha * TD_Error
        """
        current_q = self.q_table[state_idx, action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state_idx, next_action]

        td_error = target - current_q
        self.q_table[state_idx, action] += self.alpha * td_error
        self.total_updates += 1
        return float(td_error)

    def decay_epsilon(self) -> float:
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def get_policy(self, env: GridWorldEnv) -> List[List[Dict[str, Any]]]:
        """Extract the learned greedy policy for every grid cell."""
        policy_grid = []
        for r in range(env.rows):
            row_data = []
            for c in range(env.cols):
                pos = (r, c)
                is_obstacle = env.is_obstacle(r, c)
                is_goal = pos == env.goal_pos
                is_start = pos == env.start_pos
                s_idx = env.state_to_index(pos)
                q_vals = self.q_table[s_idx].tolist()
                max_q = float(np.max(self.q_table[s_idx]))
                best_act = int(np.argmax(self.q_table[s_idx]))

                row_data.append({
                    "row": r,
                    "col": c,
                    "is_obstacle": is_obstacle,
                    "is_goal": is_goal,
                    "is_start": is_start,
                    "best_action": best_act,
                    "best_action_name": ACTION_NAMES[best_act],
                    "max_q": round(max_q, 3),
                    "q_values": [round(v, 3) for v in q_vals]
                })
            policy_grid.append(row_data)
        return policy_grid

    def extract_optimal_path(self, env: GridWorldEnv, max_steps: int = 150) -> Dict[str, Any]:
        """Follow learned greedy policy from start to goal."""
        curr = env.start_pos
        path = [list(curr)]
        actions = []
        rewards = []
        visited = {curr: 0}
        total_reward = 0.0
        success = False

        for step in range(max_steps):
            if curr == env.goal_pos:
                success = True
                break

            s_idx = env.state_to_index(curr)
            act = self.choose_action(s_idx, greedy=True)
            actions.append(ACTION_NAMES[act])

            dr, dc = [( -1, 0 ), ( 1, 0 ), ( 0, -1 ), ( 0, 1 )][act]
            nxt = (curr[0] + dr, curr[1] + dc)

            if not env.is_valid_coord(nxt[0], nxt[1]):
                reward = -5.0
                nxt = curr
            elif env.is_obstacle(nxt[0], nxt[1]):
                reward = -10.0
                nxt = curr
            elif nxt == env.goal_pos:
                reward = 100.0
                success = True
                curr = nxt
                path.append(list(curr))
                rewards.append(reward)
                total_reward += reward
                break
            else:
                reward = -1.0

            rewards.append(reward)
            total_reward += reward
            curr = nxt
            path.append(list(curr))

            if curr in visited and visited[curr] >= 3:
                break
            visited[curr] = visited.get(curr, 0) + 1

        return {
            "path": path,
            "actions": actions,
            "steps": len(actions),
            "total_reward": round(total_reward, 2),
            "success": success
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent hyperparameters and status."""
        return {
            "algorithm": self.algorithm_name,
            "policy_type": self.policy_type,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": round(self.epsilon, 4),
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "episodes_trained": self.episodes_trained,
            "total_updates": self.total_updates
        }
