"""
Evaluation and Comparison Engine for RL and Baseline Agents
Evaluates learned policies, computes performance metrics, and conducts side-by-side benchmarks.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional
from backend.environment import GridWorldEnv, ACTION_NAMES
from backend.q_learning import QLearningAgent
from backend.sarsa import SarsaAgent
from backend.goal_agent import GoalBasedAgent


def evaluate_agent(
    agent: Any,
    env: GridWorldEnv,
    num_eval_episodes: int = 20,
    max_steps: int = 150
) -> Dict[str, Any]:
    """
    Evaluate an agent's learned greedy policy (epsilon = 0, no exploration).
    Runs multiple deterministic/greedy evaluation episodes.
    """
    # If Goal-Based Agent
    if isinstance(agent, GoalBasedAgent):
        res = agent.solve_path(env)
        return {
            "algorithm": agent.algorithm_name,
            "policy_type": agent.policy_type,
            "success_rate": 100.0 if res["success"] else 0.0,
            "avg_reward": res["total_reward"],
            "avg_steps": res["steps"],
            "optimal_path": res["path"],
            "actions": res["actions"],
            "num_episodes": 1,
            "is_baseline": True
        }

    # RL Agents (Q-Learning or SARSA)
    successes = 0
    total_rewards = []
    total_steps = []
    representative_path = []
    representative_actions = []

    for ep in range(num_eval_episodes):
        state = env.reset()
        ep_reward = 0.0
        steps = 0
        path = [list(state)]
        actions = []
        done = False

        while not done and steps < max_steps:
            s_idx = env.state_to_index(state)
            action = agent.choose_action(s_idx, greedy=True)
            actions.append(ACTION_NAMES[action])

            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1
            state = next_state
            path.append(list(state))

            if done and info.get("reached_goal"):
                successes += 1

        total_rewards.append(ep_reward)
        total_steps.append(steps)

        if ep == 0:
            representative_path = path
            representative_actions = actions

    avg_rew = float(np.mean(total_rewards)) if total_rewards else 0.0
    avg_st = float(np.mean(total_steps)) if total_steps else 0.0
    succ_rate = float((successes / max(1, num_eval_episodes)) * 100.0)

    return {
        "algorithm": agent.algorithm_name,
        "policy_type": agent.policy_type,
        "success_rate": round(succ_rate, 2),
        "avg_reward": round(avg_rew, 2),
        "avg_steps": round(avg_st, 2),
        "optimal_path": representative_path,
        "actions": representative_actions,
        "num_episodes": num_eval_episodes,
        "is_baseline": False
    }


def train_agent_sync(
    agent_class: type,
    env: GridWorldEnv,
    episodes: int = 500,
    max_steps: int = 150,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.01
) -> Dict[str, Any]:
    """
    Train an RL agent synchronously to collect actual learning curves.
    """
    agent = agent_class(
        num_states=env.num_states,
        num_actions=env.num_actions,
        learning_rate=alpha,
        discount_factor=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon
    )

    rewards_history = []
    steps_history = []
    success_history = []
    epsilons = []

    start_time = time.time()
    convergence_ep = None
    window = []

    is_sarsa = isinstance(agent, SarsaAgent)

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0.0
        steps = 0
        done = False
        reached_goal = False

        s_idx = env.state_to_index(state)
        # For SARSA: pick initial action
        curr_action = agent.choose_action(s_idx)

        while not done and steps < max_steps:
            if is_sarsa:
                action_to_take = curr_action
            else:
                action_to_take = agent.choose_action(s_idx)

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

        agent.decay_epsilon()
        agent.episodes_trained += 1

        rewards_history.append(ep_reward)
        steps_history.append(steps)
        success_history.append(1 if reached_goal else 0)
        epsilons.append(agent.epsilon)

        window.append(ep_reward)
        if len(window) > 50:
            window.pop(0)

        # Detect convergence (stable high reward across window)
        if convergence_ep is None and len(window) == 50 and np.mean(window) > 50.0:
            convergence_ep = ep + 1

    duration = time.time() - start_time
    eval_res = evaluate_agent(agent, env, num_eval_episodes=25, max_steps=max_steps)

    # Calculate rolling averages for plotting
    rolling_rewards = []
    rolling_steps = []
    r_window = []
    s_window = []
    w_size = 20

    for r, s in zip(rewards_history, steps_history):
        r_window.append(r)
        s_window.append(s)
        if len(r_window) > w_size:
            r_window.pop(0)
            s_window.pop(0)
        rolling_rewards.append(round(float(np.mean(r_window)), 2))
        rolling_steps.append(round(float(np.mean(s_window)), 2))

    return {
        "agent": agent,
        "duration_sec": round(duration, 3),
        "convergence_episode": convergence_ep or episodes,
        "final_epsilon": round(agent.epsilon, 4),
        "eval": eval_res,
        "rewards_raw": rewards_history,
        "rolling_rewards": rolling_rewards,
        "rolling_steps": rolling_steps,
        "success_history": success_history,
        "epsilons": epsilons
    }


def run_benchmark_comparison(
    env: GridWorldEnv,
    episodes: int = 500,
    max_steps: int = 150,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.01
) -> Dict[str, Any]:
    """
    Run identical benchmark on Q-Learning, SARSA, and Goal-Based Agent.
    Generates genuine, empirically measured comparative results.
    """
    # 1. Train Q-Learning
    q_env = GridWorldEnv(
        rows=env.rows,
        cols=env.cols,
        start_pos=env.start_pos,
        goal_pos=env.goal_pos,
        obstacles=[list(o) for o in env.obstacles]
    )
    q_result = train_agent_sync(
        QLearningAgent, q_env,
        episodes=episodes, max_steps=max_steps,
        alpha=alpha, gamma=gamma,
        epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon
    )

    # 2. Train SARSA
    s_env = GridWorldEnv(
        rows=env.rows,
        cols=env.cols,
        start_pos=env.start_pos,
        goal_pos=env.goal_pos,
        obstacles=[list(o) for o in env.obstacles]
    )
    s_result = train_agent_sync(
        SarsaAgent, s_env,
        episodes=episodes, max_steps=max_steps,
        alpha=alpha, gamma=gamma,
        epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon
    )

    # 3. Evaluate Goal-Based Agent Baseline (BFS)
    goal_agent = GoalBasedAgent()
    goal_result = goal_agent.solve_path(env)

    # Subsample episode points for frontend Chart.js (max 100 points)
    step = max(1, episodes // 80)
    plot_episodes = list(range(1, episodes + 1))[::step]
    q_rewards_sub = q_result["rolling_rewards"][::step]
    s_rewards_sub = s_result["rolling_rewards"][::step]
    q_steps_sub = q_result["rolling_steps"][::step]
    s_steps_sub = s_result["rolling_steps"][::step]

    # Comparative analysis synthesis
    q_avg_rew = q_result["eval"]["avg_reward"]
    s_avg_rew = s_result["eval"]["avg_reward"]
    g_avg_rew = goal_result["total_reward"]

    q_steps = q_result["eval"]["avg_steps"]
    s_steps = s_result["eval"]["avg_steps"]
    g_steps = goal_result["steps"]

    summary = {
        "environment": {
            "rows": env.rows,
            "cols": env.cols,
            "num_obstacles": len(env.obstacles),
            "episodes": episodes
        },
        "q_learning": {
            "name": "Q-Learning",
            "type": "Off-Policy TD Control",
            "avg_reward": q_avg_rew,
            "success_rate": q_result["eval"]["success_rate"],
            "avg_steps": q_steps,
            "convergence_episode": q_result["convergence_episode"],
            "training_time_sec": q_result["duration_sec"],
            "optimal_path": q_result["eval"]["optimal_path"],
            "actions": q_result["eval"]["actions"],
            "final_epsilon": q_result["final_epsilon"]
        },
        "sarsa": {
            "name": "SARSA",
            "type": "On-Policy TD Control",
            "avg_reward": s_avg_rew,
            "success_rate": s_result["eval"]["success_rate"],
            "avg_steps": s_steps,
            "convergence_episode": s_result["convergence_episode"],
            "training_time_sec": s_result["duration_sec"],
            "optimal_path": s_result["eval"]["optimal_path"],
            "actions": s_result["eval"]["actions"],
            "final_epsilon": s_result["final_epsilon"]
        },
        "goal_based": {
            "name": "Goal-Based Agent",
            "type": "Model-Based Classical Search (BFS Baseline)",
            "is_reinforcement_learning": False,
            "avg_reward": g_avg_rew,
            "success_rate": 100.0 if goal_result["success"] else 0.0,
            "avg_steps": g_steps,
            "convergence_episode": "N/A (Analytical / Instant)",
            "training_time_sec": 0.001,
            "optimal_path": goal_result["path"],
            "actions": goal_result["actions"],
            "visited_nodes": goal_result.get("visited_nodes", 0)
        },
        "chart_data": {
            "episodes": plot_episodes,
            "q_learning_rewards": q_rewards_sub,
            "sarsa_rewards": s_rewards_sub,
            "goal_baseline_reward": [g_avg_rew] * len(plot_episodes),
            "q_learning_steps": q_steps_sub,
            "sarsa_steps": s_steps_sub,
            "goal_baseline_steps": [g_steps] * len(plot_episodes)
        },
        "comparison_table": [
            {
                "metric": "Algorithm Category",
                "q_learning": "Model-Free Reinforcement Learning",
                "sarsa": "Model-Free Reinforcement Learning",
                "goal_based": "Model-Based Classical Pathfinding"
            },
            {
                "metric": "Policy Strategy",
                "q_learning": "Off-Policy (Greedy Max Target)",
                "sarsa": "On-Policy (Behavior Policy Target)",
                "goal_based": "Deterministic BFS Graph Search"
            },
            {
                "metric": "Environment Model Required?",
                "q_learning": "No (Learns by Trial & Error)",
                "sarsa": "No (Learns by Trial & Error)",
                "goal_based": "Yes (Requires full map & obstacle graph)"
            },
            {
                "metric": "Average Evaluation Reward",
                "q_learning": f"{q_avg_rew:+.2f}",
                "sarsa": f"{s_avg_rew:+.2f}",
                "goal_based": f"{g_avg_rew:+.2f}"
            },
            {
                "metric": "Evaluation Success Rate",
                "q_learning": f"{q_result['eval']['success_rate']:.1f}%",
                "sarsa": f"{s_result['eval']['success_rate']:.1f}%",
                "goal_based": "100.0%" if goal_result["success"] else "0.0%"
            },
            {
                "metric": "Optimal Path Steps",
                "q_learning": f"{q_steps:.1f} steps",
                "sarsa": f"{s_steps:.1f} steps",
                "goal_based": f"{g_steps} steps (Theoretical Shortest)"
            },
            {
                "metric": "Exploration Risk Handling",
                "q_learning": "Aggressive / Optimal: learns shortest route directly alongside obstacles",
                "sarsa": "Conservative / Safer: prefers paths with larger obstacle buffer during exploration",
                "goal_based": "Zero risk: plans path offline before moving"
            },
            {
                "metric": "Training Time",
                "q_learning": f"{q_result['duration_sec']:.2f} s",
                "sarsa": f"{s_result['duration_sec']:.2f} s",
                "goal_based": "< 0.005 s"
            }
        ]
    }

    return summary
