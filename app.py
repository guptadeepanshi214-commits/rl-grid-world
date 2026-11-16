"""
Reinforcement Learning Agent for Sequential Decision Environments
Flask Web Application and REST API Backend
"""

import os
import json
from typing import Dict, Any
from flask import Flask, render_template, request, jsonify

from backend.environment import GridWorldEnv, get_preset_obstacles
from backend.q_learning import QLearningAgent
from backend.sarsa import SarsaAgent
from backend.goal_agent import GoalBasedAgent
from backend.training_manager import TrainingManager
from backend.evaluation import run_benchmark_comparison, evaluate_agent
from backend.database import (
    init_db,
    get_all_training_sessions,
    get_session_by_id,
    save_comparison_benchmark,
    get_latest_comparison,
    save_simulation_run,
    get_simulation_history,
    get_simulation_stats
)

# Initialize Flask App with custom folder paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "frontend", "templates"),
    static_folder=os.path.join(BASE_DIR, "frontend", "static")
)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "rl_agent_secret_key_2026")

# Initialize global singletons
global_env = GridWorldEnv(rows=10, cols=10, preset="default")
training_mgr = TrainingManager()

# Ensure Database is initialized
try:
    init_db()
except Exception as e:
    print(f"[Database] Warning: Could not initialize SQLite DB: {e}")


# ==========================================
# Frontend Page Routes
# ==========================================

@app.route("/")
def index():
    """Render main AI dashboard single-page application."""
    return render_template("index.html")


# ==========================================
# REST API Endpoints
# ==========================================

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "app": "Reinforcement Learning Agent Dashboard",
        "version": "1.0.0",
        "env_size": f"{global_env.rows}x{global_env.cols}"
    })


@app.route("/api/dashboard", methods=["GET"])
def get_dashboard():
    """Summary metrics for the landing dashboard, combining training and simulation telemetry."""
    try:
        past_sessions = get_all_training_sessions(limit=15)
    except Exception:
        past_sessions = []

    try:
        past_simulations = get_simulation_history(limit=15)
        sim_stats = get_simulation_stats()
    except Exception:
        past_simulations = []
        sim_stats = {"total_simulations": 0, "successful_simulations": 0, "success_rate": 0.0, "avg_reward": 0.0}

    total_episodes_all_time = sum(s.get("episodes", 0) for s in past_sessions)
    total_successful_runs = sum(1 for s in past_sessions if (s.get("success_rate") or 0) > 80.0)

    current_status = training_mgr.get_status()

    # Determine current algorithm display
    curr_algo = "Q-Learning"
    if training_mgr.current_agent:
        curr_algo = training_mgr.current_agent.algorithm_name
    elif past_simulations:
        curr_algo = past_simulations[0].get("algorithm", "Q-Learning")
    elif past_sessions:
        curr_algo = past_sessions[0].get("algorithm", "Q-Learning")

    # Aggregate stats
    avg_reward = current_status.get("average_reward", 0.0)
    success_rate = current_status.get("success_rate", 0.0)
    if avg_reward == 0.0 and past_sessions:
        avg_reward = round(sum(s.get("avg_reward", 0.0) for s in past_sessions) / len(past_sessions), 2)
        success_rate = round(sum(s.get("success_rate", 0.0) for s in past_sessions) / len(past_sessions), 1)
    elif avg_reward == 0.0 and past_simulations:
        avg_reward = sim_stats.get("avg_reward", 0.0)
        success_rate = sim_stats.get("success_rate", 0.0)

    return jsonify({
        "title": "Reinforcement Learning Agent for Sequential Decision Environments",
        "domain": "Artificial Intelligence & Reinforcement Learning",
        "total_episodes": total_episodes_all_time or current_status.get("current_episode", 0),
        "total_simulations": sim_stats.get("total_simulations", 0),
        "current_algorithm": curr_algo,
        "average_reward": avg_reward,
        "success_rate": success_rate,
        "successful_episodes": total_successful_runs,
        "successful_simulations": sim_stats.get("successful_simulations", 0),
        "environment": global_env.to_dict(),
        "recent_sessions": past_sessions[:5],
        "recent_simulations": past_simulations[:10],
        "simulation_stats": sim_stats,
        "training_status": current_status.get("status", "idle")
    })


@app.route("/api/environment", methods=["GET"])
def get_environment():
    """Retrieve current Grid World configuration."""
    return jsonify(global_env.to_dict())


@app.route("/api/environment/reset", methods=["POST"])
def reset_environment():
    """Reset Grid World layout to default or requested preset."""
    data = request.get_json(silent=True) or {}
    preset = data.get("preset", "default")
    rows = int(data.get("rows", 10))
    cols = int(data.get("cols", 10))

    raw_obs = get_preset_obstacles(rows, cols, preset)
    global_env.set_layout(
        rows=rows,
        cols=cols,
        start_pos=[0, 0],
        goal_pos=[rows - 1, cols - 1],
        obstacles=raw_obs
    )
    training_mgr.latest_optimal_path = []
    training_mgr.evaluation_result = None
    return jsonify({
        "status": "success",
        "message": f"Environment reset with preset '{preset}'",
        "environment": global_env.to_dict()
    })


@app.route("/api/environment/update", methods=["POST"])
def update_environment():
    """Modify grid layout, start/goal locations, or toggle obstacles."""
    data = request.get_json(silent=True) or {}

    # Check for cell obstacle toggle
    if "toggle_obstacle" in data:
        r, c = data["toggle_obstacle"]
        is_now_obstacle = global_env.toggle_obstacle(r, c)
        training_mgr.latest_optimal_path = []
        training_mgr.evaluation_result = None
        return jsonify({
            "status": "success",
            "toggled": [r, c],
            "is_obstacle": is_now_obstacle,
            "environment": global_env.to_dict()
        })

    rows = data.get("rows")
    cols = data.get("cols")
    start = data.get("start")
    goal = data.get("goal")
    obstacles = data.get("obstacles")

    global_env.set_layout(
        rows=rows,
        cols=cols,
        start_pos=start,
        goal_pos=goal,
        obstacles=obstacles
    )
    training_mgr.latest_optimal_path = []
    training_mgr.evaluation_result = None

    return jsonify({
        "status": "success",
        "message": "Environment layout updated",
        "environment": global_env.to_dict()
    })


@app.route("/api/train", methods=["POST"])
def start_train():
    """Start asynchronous training session."""
    data = request.get_json(silent=True) or {}
    res = training_mgr.start_training(global_env, data)
    return jsonify(res)


@app.route("/api/train/pause", methods=["POST"])
def pause_train():
    """Pause active training."""
    return jsonify(training_mgr.pause_training())


@app.route("/api/train/resume", methods=["POST"])
def resume_train():
    """Resume paused training."""
    return jsonify(training_mgr.resume_training())


@app.route("/api/train/stop", methods=["POST"])
def stop_train():
    """Stop ongoing training."""
    return jsonify(training_mgr.stop_training())


@app.route("/api/train/reset", methods=["POST"])
def reset_train():
    """Reset training manager."""
    return jsonify(training_mgr.reset_training())


@app.route("/api/train/status", methods=["GET"])
def get_train_status():
    """Polling endpoint for real-time training progress."""
    status_data = training_mgr.get_status()
    return jsonify(status_data)


@app.route("/api/results", methods=["GET"])
def get_results():
    """Retrieve learned policy, path, and evaluation for current agent."""
    agent = training_mgr.current_agent
    if agent is None:
        # Fall back to baseline BFS or default evaluation
        agent = GoalBasedAgent()

    if isinstance(agent, GoalBasedAgent):
        policy = agent.get_policy(global_env)
        path_res = agent.solve_path(global_env)
        return jsonify({
            "has_agent": True,
            "algorithm": agent.algorithm_name,
            "policy_type": agent.policy_type,
            "is_baseline": True,
            "policy": policy,
            "optimal_path": path_res["path"],
            "actions": path_res["actions"],
            "steps": path_res["steps"],
            "total_reward": path_res["total_reward"],
            "success": path_res["success"],
            "evaluation": training_mgr.evaluation_result or {
                "avg_reward": path_res["total_reward"],
                "avg_steps": path_res["steps"],
                "success_rate": 100.0 if path_res["success"] else 0.0
            }
        })

    # RL Agent
    policy = agent.get_policy(global_env)
    path_info = agent.extract_optimal_path(global_env, max_steps=150)
    eval_res = training_mgr.evaluation_result or evaluate_agent(agent, global_env, num_eval_episodes=20)

    return jsonify({
        "has_agent": True,
        "algorithm": agent.algorithm_name,
        "policy_type": agent.policy_type,
        "is_baseline": False,
        "policy": policy,
        "optimal_path": path_info["path"],
        "actions": path_info["actions"],
        "steps": path_info["steps"],
        "total_reward": path_info["total_reward"],
        "success": path_info["success"],
        "evaluation": eval_res,
        "hyperparameters": agent.to_dict()
    })


@app.route("/api/comparison", methods=["GET"])
def get_comparison():
    """Retrieve latest algorithm comparison results or generate baseline."""
    cached = get_latest_comparison()
    if cached:
        return jsonify({"status": "success", "comparison": cached, "source": "database"})

    # Auto-run an initial benchmark with 350 episodes for instant real comparison
    benchmark = run_benchmark_comparison(
        global_env,
        episodes=350,
        max_steps=150,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.992,
        min_epsilon=0.01
    )
    try:
        save_comparison_benchmark(benchmark)
    except Exception:
        pass

    return jsonify({"status": "success", "comparison": benchmark, "source": "computed"})


@app.route("/api/comparison/run", methods=["POST"])
def run_comparison():
    """Run full live benchmark comparing Q-Learning, SARSA, and Goal-Based Agent."""
    data = request.get_json(silent=True) or {}
    episodes = int(data.get("episodes", 500))
    alpha = float(data.get("alpha", 0.1))
    gamma = float(data.get("gamma", 0.95))
    epsilon = float(data.get("epsilon", 1.0))
    decay = float(data.get("epsilon_decay", 0.995))
    min_eps = float(data.get("min_epsilon", 0.01))
    max_steps = int(data.get("max_steps", 150))

    benchmark = run_benchmark_comparison(
        global_env,
        episodes=episodes,
        max_steps=max_steps,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=decay,
        min_epsilon=min_eps
    )
    try:
        save_comparison_benchmark(benchmark)
    except Exception as e:
        print(f"[Comparison] Error saving benchmark to DB: {e}")

    return jsonify({"status": "success", "comparison": benchmark})


@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    """Return historical and session data formatted for Chart.js performance curves."""
    session_id = request.args.get("session_id", type=int)

    if session_id:
        sess = get_session_by_id(session_id)
        if sess and sess.get("episode_logs"):
            logs = sess["episode_logs"]
            episodes = [l["episode"] for l in logs]
            rewards = [l["reward"] for l in logs]
            steps = [l["steps"] for l in logs]
            epsilons = [l["epsilon"] for l in logs]

            # Compute moving average
            ma_rewards = []
            w = []
            for r in rewards:
                w.append(r)
                if len(w) > 20:
                    w.pop(0)
                ma_rewards.append(round(sum(w) / len(w), 2))

            return jsonify({
                "source": "session",
                "session_info": sess,
                "episodes": episodes,
                "rewards": rewards,
                "average_rewards": ma_rewards,
                "steps": steps,
                "epsilons": epsilons
            })

    # Return currently active training history if available
    records = training_mgr.episode_records
    if records:
        step = max(1, len(records) // 80)
        sliced = records[::step]
        if sliced[-1] != records[-1]:
            sliced.append(records[-1])

        episodes = [r["episode"] for r in sliced]
        rewards = [r["reward"] for r in sliced]
        steps = [r["steps"] for r in sliced]
        epsilons = [r["epsilon"] for r in sliced]

        # Calculate moving average
        ma_rewards = []
        w = []
        for r in rewards:
            w.append(r)
            if len(w) > 15:
                w.pop(0)
            ma_rewards.append(round(sum(w) / len(w), 2))

        return jsonify({
            "source": "current_training",
            "episodes": episodes,
            "rewards": rewards,
            "average_rewards": ma_rewards,
            "steps": steps,
            "epsilons": epsilons
        })

    # Fallback to comparison benchmark curves if no active training session
    latest_comp = get_latest_comparison()
    if latest_comp and "chart_data" in latest_comp:
        cd = latest_comp["chart_data"]
        return jsonify({
            "source": "comparison",
            "episodes": cd.get("episodes", []),
            "q_learning_rewards": cd.get("q_learning_rewards", []),
            "sarsa_rewards": cd.get("sarsa_rewards", []),
            "q_learning_steps": cd.get("q_learning_steps", []),
            "sarsa_steps": cd.get("sarsa_steps", []),
            "goal_baseline_reward": cd.get("goal_baseline_reward", []),
            "goal_baseline_steps": cd.get("goal_baseline_steps", [])
        })

    return jsonify({
        "source": "empty",
        "episodes": [],
        "rewards": [],
        "average_rewards": [],
        "steps": [],
        "epsilons": []
    })


@app.route("/api/simulate", methods=["POST"])
def simulate_agent():
    """
    Simulate step-by-step navigation of a selected agent on the current grid.
    Accurately classifies outcomes (Goal Reached, Maximum Steps Reached, No Valid Path).
    Persists simulation executions to SQLite simulation_history.
    """
    data = request.get_json(silent=True) or {}
    algo = data.get("algorithm", "current").lower()
    max_steps = int(data.get("max_steps", 150))

    agent = None
    if algo == "goal_based":
        agent = GoalBasedAgent()
    elif algo in ("q_learning", "sarsa") and training_mgr.current_agent and training_mgr.algorithm == algo:
        agent = training_mgr.current_agent
    else:
        agent = training_mgr.current_agent

    if agent is None:
        agent = GoalBasedAgent()

    if isinstance(agent, GoalBasedAgent):
        res = agent.solve_path(global_env)
        success = bool(res.get("success", False))
        steps = int(res.get("steps", 0))
        total_reward = float(res.get("total_reward", 0.0))
        path = res.get("path", [])
        actions = res.get("actions", [])

        if success:
            status_detail = "Goal Reached"
        elif res.get("error") or len(path) <= 1:
            status_detail = "No Valid Path"
        elif steps >= max_steps:
            status_detail = "Maximum Steps Reached"
        else:
            status_detail = "Path Not Found"

        # Persist simulation run to database
        sim_record = {
            "algorithm": agent.algorithm_name,
            "start": list(global_env.start_pos),
            "goal": list(global_env.goal_pos),
            "steps": steps,
            "total_reward": total_reward,
            "success": success,
            "path": path,
            "status": status_detail
        }
        try:
            save_simulation_run(sim_record)
        except Exception as ex:
            print(f"[Simulate] Warning: could not persist simulation: {ex}")

        return jsonify({
            "status": "success",
            "algorithm": agent.algorithm_name,
            "is_baseline": True,
            "path": path,
            "actions": actions,
            "steps": steps,
            "total_reward": total_reward,
            "success": success,
            "status_detail": status_detail,
            "error": res.get("error")
        })

    # RL agent execution
    path_res = agent.extract_optimal_path(global_env, max_steps=max_steps)
    success = bool(path_res.get("success", False))
    steps = int(path_res.get("steps", 0))
    total_reward = float(path_res.get("total_reward", 0.0))
    path = path_res.get("path", [])
    actions = path_res.get("actions", [])

    if success:
        status_detail = "Goal Reached"
    elif steps >= max_steps:
        status_detail = "Maximum Steps Reached"
    elif len(path) <= 1:
        status_detail = "No Valid Path"
    else:
        status_detail = "Maximum Steps Reached"

    # Persist simulation run to database
    sim_record = {
        "algorithm": agent.algorithm_name,
        "start": list(global_env.start_pos),
        "goal": list(global_env.goal_pos),
        "steps": steps,
        "total_reward": total_reward,
        "success": success,
        "path": path,
        "status": status_detail
    }
    try:
        save_simulation_run(sim_record)
    except Exception as ex:
        print(f"[Simulate] Warning: could not persist simulation: {ex}")

    return jsonify({
        "status": "success",
        "algorithm": agent.algorithm_name,
        "is_baseline": False,
        "path": path,
        "actions": actions,
        "steps": steps,
        "total_reward": total_reward,
        "success": success,
        "status_detail": status_detail
    })


@app.route("/api/simulations", methods=["GET"])
def get_simulations():
    """List recent simulation history runs and aggregate metrics."""
    limit = request.args.get("limit", default=25, type=int)
    sims = get_simulation_history(limit=limit)
    stats = get_simulation_stats()
    return jsonify({"status": "success", "simulations": sims, "stats": stats})


@app.route("/api/history", methods=["GET"])
def get_history():
    """List previous training sessions from database."""
    limit = request.args.get("limit", default=25, type=int)
    sessions = get_all_training_sessions(limit=limit)
    return jsonify({"sessions": sessions})


@app.route("/api/history/<int:session_id>", methods=["GET"])
def get_history_detail(session_id: int):
    """Retrieve full details of a specific past session."""
    session_data = get_session_by_id(session_id)
    if not session_data:
        return jsonify({"status": "error", "message": "Session not found"}), 404
    return jsonify({"status": "success", "session": session_data})


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    print(f"Starting Reinforcement Learning Agent Web Application on http://127.0.0.1:{port}")
    print(f"Open in your browser: http://localhost:{port} or http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug_mode, threaded=True)
