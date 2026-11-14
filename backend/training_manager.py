"""
Training Manager for Asynchronous Reinforcement Learning Execution
Supports start, pause, resume, stop, real-time status polling, and database persistence.
"""

import time
import threading
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from backend.environment import GridWorldEnv, ACTION_NAMES
from backend.q_learning import QLearningAgent
from backend.sarsa import SarsaAgent
from backend.goal_agent import GoalBasedAgent
from backend.evaluation import evaluate_agent
from backend.database import save_training_session


class TrainingManager:
    """
    Thread-safe training coordinator for interactive web applications.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused by default
        self._stop_event = threading.Event()

        # Training state
        self.status = "idle"  # idle, training, paused, stopped, completed, error
        self.algorithm = "q_learning"
        self.params: Dict[str, Any] = {}
        self.current_episode = 0
        self.total_episodes = 1000
        self.max_steps = 200
        self.current_reward = 0.0
        self.average_reward = 0.0
        self.exploration_rate = 1.0
        self.success_rate = 0.0
        self.progress_percentage = 0.0
        self.error_message: Optional[str] = None
        self.start_time: float = 0.0
        self.duration_sec: float = 0.0

        # Trained agent & environment reference
        self.current_agent: Any = None
        self.current_env: Optional[GridWorldEnv] = None
        self.evaluation_result: Optional[Dict[str, Any]] = None
        self.latest_optimal_path: List[List[int]] = []

        # History buffers for real-time charting
        self.episode_records: List[Dict[str, Any]] = []
        self._recent_rewards: List[float] = []
        self._recent_successes: List[int] = []

    def start_training(self, env: GridWorldEnv, params: Dict[str, Any]) -> Dict[str, Any]:
        """Initiate training in a background thread."""
        with self._lock:
            if self.status == "training":
                return {"status": "error", "message": "Training is already running"}

            self.algorithm = params.get("algorithm", "q_learning").lower()
            self.params = params
            self.total_episodes = int(params.get("episodes", 1000))
            self.max_steps = int(params.get("max_steps", 200))
            alpha = float(params.get("alpha", 0.1))
            gamma = float(params.get("gamma", 0.95))
            epsilon = float(params.get("epsilon", 1.0))
            epsilon_decay = float(params.get("epsilon_decay", 0.995))
            min_epsilon = float(params.get("min_epsilon", 0.01))

            self.current_env = env
            self.current_episode = 0
            self.current_reward = 0.0
            self.average_reward = 0.0
            self.exploration_rate = epsilon
            self.success_rate = 0.0
            self.progress_percentage = 0.0
            self.error_message = None
            self.duration_sec = 0.0
            self.episode_records = []
            self._recent_rewards = []
            self._recent_successes = []
            self.evaluation_result = None
            self.latest_optimal_path = []

            # Initialize Agent
            if self.algorithm == "q_learning":
                self.current_agent = QLearningAgent(
                    num_states=env.num_states,
                    num_actions=env.num_actions,
                    learning_rate=alpha,
                    discount_factor=gamma,
                    epsilon=epsilon,
                    epsilon_decay=epsilon_decay,
                    min_epsilon=min_epsilon
                )
            elif self.algorithm == "sarsa":
                self.current_agent = SarsaAgent(
                    num_states=env.num_states,
                    num_actions=env.num_actions,
                    learning_rate=alpha,
                    discount_factor=gamma,
                    epsilon=epsilon,
                    epsilon_decay=epsilon_decay,
                    min_epsilon=min_epsilon
                )
            elif self.algorithm == "goal_based":
                self.current_agent = GoalBasedAgent()
            else:
                return {"status": "error", "message": f"Unsupported algorithm '{self.algorithm}'"}

            self._pause_event.set()
            self._stop_event.clear()
            self.status = "training"
            self.start_time = time.time()

            # Launch thread
            self._thread = threading.Thread(target=self._run_training_loop, daemon=True)
            self._thread.start()

            return {"status": "success", "message": f"Started {self.current_agent.algorithm_name} training"}

    def _run_training_loop(self) -> None:
        """Internal worker executing the iterative training process."""
        try:
            env = self.current_env
            agent = self.current_agent

            # Handle Goal-Based Agent (instant baseline)
            if isinstance(agent, GoalBasedAgent):
                path_res = agent.solve_path(env)
                self.current_episode = 1
                self.total_episodes = 1
                self.progress_percentage = 100.0
                self.current_reward = path_res["total_reward"]
                self.average_reward = path_res["total_reward"]
                self.success_rate = 100.0 if path_res["success"] else 0.0
                self.exploration_rate = 0.0
                self.latest_optimal_path = path_res["path"]
                self.evaluation_result = {
                    "algorithm": agent.algorithm_name,
                    "policy_type": agent.policy_type,
                    "success_rate": self.success_rate,
                    "avg_reward": self.average_reward,
                    "avg_steps": path_res["steps"],
                    "optimal_path": path_res["path"],
                    "actions": path_res["actions"],
                    "is_baseline": True
                }
                self.episode_records.append({
                    "episode": 1,
                    "reward": self.current_reward,
                    "steps": path_res["steps"],
                    "epsilon": 0.0,
                    "success": path_res["success"]
                })
                self.status = "completed"
                self.duration_sec = time.time() - self.start_time
                self._persist_session()
                return

            # RL Loop for Q-Learning & SARSA
            is_sarsa = isinstance(agent, SarsaAgent)

            for ep in range(1, self.total_episodes + 1):
                # Check for stop signal
                if self._stop_event.is_set():
                    self.status = "stopped"
                    break

                # Check for pause signal
                self._pause_event.wait()
                if self._stop_event.is_set():
                    self.status = "stopped"
                    break

                # Execute Episode
                state = env.reset()
                ep_reward = 0.0
                steps = 0
                done = False
                reached_goal = False

                s_idx = env.state_to_index(state)
                curr_action = agent.choose_action(s_idx)

                while not done and steps < self.max_steps:
                    action_to_take = curr_action if is_sarsa else agent.choose_action(s_idx)
                    next_state, reward, done, info = env.step(action_to_take)
                    ep_reward += reward
                    steps += 1
                    next_s_idx = env.state_to_index(next_state)

                    if is_sarsa:
                        next_action = agent.choose_action(next_s_idx) if not done else 0
                        agent.update(s_idx, action_to_take, reward, next_s_idx, next_action, done)
                        curr_action = next_action
                    else:
                        agent.update(s_idx, action_to_take, reward, next_s_idx, done)

                    s_idx = next_s_idx
                    state = next_state

                    if done and info.get("reached_goal"):
                        reached_goal = True

                # Epsilon decay & episode counter
                agent.decay_epsilon()
                agent.episodes_trained += 1

                # Update metrics
                with self._lock:
                    self.current_episode = ep
                    self.current_reward = round(ep_reward, 2)
                    self.exploration_rate = round(agent.epsilon, 4)
                    self.progress_percentage = round((ep / self.total_episodes) * 100.0, 1)

                    self._recent_rewards.append(ep_reward)
                    self._recent_successes.append(1 if reached_goal else 0)
                    if len(self._recent_rewards) > 50:
                        self._recent_rewards.pop(0)
                        self._recent_successes.pop(0)

                    self.average_reward = round(float(np.mean(self._recent_rewards)), 2)
                    self.success_rate = round(float(np.mean(self._recent_successes)) * 100.0, 1)

                    # Keep records
                    self.episode_records.append({
                        "episode": ep,
                        "reward": round(ep_reward, 2),
                        "steps": steps,
                        "epsilon": round(agent.epsilon, 4),
                        "success": reached_goal
                    })

                # Small throttle every 5 episodes to prevent CPU thrashing and allow GUI updates
                if ep % 10 == 0:
                    time.sleep(0.002)

            # Training finished or stopped
            self.duration_sec = round(time.time() - self.start_time, 2)

            if not self._stop_event.is_set():
                self.status = "completed"

            # Run evaluation on final greedy policy
            eval_data = evaluate_agent(self.current_agent, self.current_env, num_eval_episodes=20, max_steps=self.max_steps)
            with self._lock:
                self.evaluation_result = eval_data
                self.latest_optimal_path = eval_data.get("optimal_path", [])

            # Persist to database
            self._persist_session()

        except Exception as e:
            with self._lock:
                self.status = "error"
                self.error_message = str(e)
            print(f"[TrainingManager] Error during training: {e}")

    def _persist_session(self) -> None:
        """Save completed run to database."""
        try:
            if not self.current_agent or not self.current_env:
                return

            avg_steps = 0.0
            if self.evaluation_result:
                avg_steps = self.evaluation_result.get("avg_steps", 0.0)

            session_data = {
                "session_uuid": f"sess_{int(datetime.now(timezone.utc).timestamp())}_{self.algorithm}",
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "algorithm": self.current_agent.algorithm_name,
                "episodes": self.current_episode,
                "alpha": getattr(self.current_agent, "alpha", 0.0),
                "gamma": getattr(self.current_agent, "gamma", 0.0),
                "initial_epsilon": getattr(self.current_agent, "initial_epsilon", 1.0),
                "epsilon_decay": getattr(self.current_agent, "epsilon_decay", 0.995),
                "min_epsilon": getattr(self.current_agent, "min_epsilon", 0.01),
                "final_epsilon": self.exploration_rate,
                "avg_reward": self.average_reward,
                "success_rate": self.success_rate,
                "avg_steps": avg_steps,
                "duration_sec": self.duration_sec,
                "grid_rows": self.current_env.rows,
                "grid_cols": self.current_env.cols,
                "optimal_path": self.latest_optimal_path,
                "notes": f"Trained {self.algorithm} for {self.current_episode} episodes"
            }

            save_training_session(session_data, self.episode_records)
        except Exception as ex:
            print(f"[TrainingManager] Failed to persist session: {ex}")

    def pause_training(self) -> Dict[str, Any]:
        """Pause active training."""
        with self._lock:
            if self.status != "training":
                return {"status": "error", "message": f"Cannot pause when status is {self.status}"}
            self._pause_event.clear()
            self.status = "paused"
            return {"status": "success", "message": "Training paused"}

    def resume_training(self) -> Dict[str, Any]:
        """Resume paused training."""
        with self._lock:
            if self.status != "paused":
                return {"status": "error", "message": f"Cannot resume when status is {self.status}"}
            self._pause_event.set()
            self.status = "training"
            return {"status": "success", "message": "Training resumed"}

    def stop_training(self) -> Dict[str, Any]:
        """Stop training early."""
        with self._lock:
            if self.status not in ("training", "paused"):
                return {"status": "error", "message": "Training is not currently running"}
            self._stop_event.set()
            self._pause_event.set()
            self.status = "stopped"
            return {"status": "success", "message": "Training stop signal sent"}

    def reset_training(self) -> Dict[str, Any]:
        """Reset training state back to idle."""
        with self._lock:
            if self.status in ("training", "paused"):
                self._stop_event.set()
                self._pause_event.set()
            self.status = "idle"
            self.current_episode = 0
            self.progress_percentage = 0.0
            self.current_reward = 0.0
            self.average_reward = 0.0
            self.exploration_rate = 1.0
            self.success_rate = 0.0
            self.episode_records = []
            self.evaluation_result = None
            self.latest_optimal_path = []
            return {"status": "success", "message": "Training manager reset to idle"}

    def get_status(self) -> Dict[str, Any]:
        """Return real-time diagnostic status payload for web polling."""
        with self._lock:
            # Subsample recent 60 points for streaming charts
            records_count = len(self.episode_records)
            sample_step = max(1, records_count // 60)
            chart_slice = self.episode_records[::sample_step]
            if self.episode_records and (not chart_slice or chart_slice[-1] != self.episode_records[-1]):
                chart_slice.append(self.episode_records[-1])

            return {
                "status": self.status,
                "algorithm": self.algorithm,
                "algorithm_display": getattr(self.current_agent, "algorithm_name", "None"),
                "current_episode": self.current_episode,
                "total_episodes": self.total_episodes,
                "current_reward": self.current_reward,
                "average_reward": self.average_reward,
                "exploration_rate": self.exploration_rate,
                "success_rate": self.success_rate,
                "progress_percentage": self.progress_percentage,
                "duration_sec": round(self.duration_sec if self.status in ("completed", "stopped") else (time.time() - self.start_time if self.start_time else 0.0), 1),
                "error_message": self.error_message,
                "has_agent": self.current_agent is not None,
                "optimal_path": self.latest_optimal_path,
                "evaluation": self.evaluation_result,
                "recent_history": chart_slice
            }
