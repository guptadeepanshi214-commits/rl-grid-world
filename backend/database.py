"""
SQLite Database Management for RL Training Sessions and Benchmarks
"""

import os
import json
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database")
DB_PATH = os.path.join(DB_DIR, "training.db")


def get_connection() -> sqlite3.Connection:
    """Return a connection with dictionary-like row factory."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database tables with schema."""
    conn = get_connection()
    cursor = conn.cursor()

    # Training sessions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS training_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_uuid TEXT UNIQUE,
        timestamp TEXT,
        algorithm TEXT NOT NULL,
        episodes INTEGER NOT NULL,
        alpha REAL,
        gamma REAL,
        initial_epsilon REAL,
        epsilon_decay REAL,
        min_epsilon REAL,
        final_epsilon REAL,
        avg_reward REAL,
        success_rate REAL,
        avg_steps REAL,
        duration_sec REAL,
        grid_rows INTEGER,
        grid_cols INTEGER,
        optimal_path_json TEXT,
        notes TEXT
    )
    """)

    # Episode metrics history table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS episode_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        episode INTEGER,
        reward REAL,
        steps INTEGER,
        epsilon REAL,
        success INTEGER,
        FOREIGN KEY (session_id) REFERENCES training_sessions(id) ON DELETE CASCADE
    )
    """)

    # Comparisons benchmark table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS comparisons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        summary_json TEXT
    )
    """)

    # Simulation runs history table (stores step-by-step navigation executions)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS simulation_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        algorithm TEXT NOT NULL,
        start_pos TEXT NOT NULL,
        goal_pos TEXT NOT NULL,
        steps INTEGER NOT NULL,
        total_reward REAL NOT NULL,
        success INTEGER NOT NULL,
        path_json TEXT,
        status TEXT
    )
    """)

    conn.commit()
    conn.close()


def save_training_session(session_data: Dict[str, Any], episode_logs: Optional[List[Dict[str, Any]]] = None) -> int:
    """Save completed training session and its episode metrics."""
    conn = get_connection()
    cursor = conn.cursor()

    ts = session_data.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    path_json = json.dumps(session_data.get("optimal_path", {}))

    cursor.execute("""
    INSERT INTO training_sessions (
        session_uuid, timestamp, algorithm, episodes, alpha, gamma,
        initial_epsilon, epsilon_decay, min_epsilon, final_epsilon,
        avg_reward, success_rate, avg_steps, duration_sec,
        grid_rows, grid_cols, optimal_path_json, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_data.get("session_uuid", f"sess_{int(datetime.now(timezone.utc).timestamp())}"),
        ts,
        session_data.get("algorithm", "Unknown"),
        int(session_data.get("episodes", 0)),
        float(session_data.get("alpha", 0.0)),
        float(session_data.get("gamma", 0.0)),
        float(session_data.get("initial_epsilon", 1.0)),
        float(session_data.get("epsilon_decay", 0.995)),
        float(session_data.get("min_epsilon", 0.01)),
        float(session_data.get("final_epsilon", 0.0)),
        float(session_data.get("avg_reward", 0.0)),
        float(session_data.get("success_rate", 0.0)),
        float(session_data.get("avg_steps", 0.0)),
        float(session_data.get("duration_sec", 0.0)),
        int(session_data.get("grid_rows", 10)),
        int(session_data.get("grid_cols", 10)),
        path_json,
        session_data.get("notes", "")
    ))

    session_id = cursor.lastrowid

    if episode_logs:
        # Downsample if too large for fast persistence (up to 200 checkpoints)
        step = max(1, len(episode_logs) // 200)
        downsampled = episode_logs[::step]
        if episode_logs[-1] not in downsampled:
            downsampled.append(episode_logs[-1])

        records = [
            (
                session_id,
                log.get("episode", 0),
                log.get("reward", 0.0),
                log.get("steps", 0),
                log.get("epsilon", 0.0),
                1 if log.get("success", False) else 0
            )
            for log in downsampled
        ]
        cursor.executemany("""
        INSERT INTO episode_logs (session_id, episode, reward, steps, epsilon, success)
        VALUES (?, ?, ?, ?, ?, ?)
        """, records)

    conn.commit()
    conn.close()
    return session_id


def get_all_training_sessions(limit: int = 25) -> List[Dict[str, Any]]:
    """Retrieve list of past training sessions."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM training_sessions ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("optimal_path_json"):
            try:
                d["optimal_path"] = json.loads(d["optimal_path_json"])
            except Exception:
                d["optimal_path"] = None
        results.append(d)
    conn.close()
    return results


def get_session_by_id(session_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve full details and episode logs for a specific session."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM training_sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    session_data = dict(row)
    if session_data.get("optimal_path_json"):
        try:
            session_data["optimal_path"] = json.loads(session_data["optimal_path_json"])
        except Exception:
            session_data["optimal_path"] = None

    cursor.execute("""
    SELECT episode, reward, steps, epsilon, success
    FROM episode_logs WHERE session_id = ? ORDER BY episode ASC
    """, (session_id,))
    logs = [dict(lr) for lr in cursor.fetchall()]
    session_data["episode_logs"] = logs

    conn.close()
    return session_data


def save_comparison_benchmark(summary: Dict[str, Any]) -> int:
    """Save multi-algorithm comparison benchmark run."""
    conn = get_connection()
    cursor = conn.cursor()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO comparisons (timestamp, summary_json) VALUES (?, ?)
    """, (ts, json.dumps(summary)))
    comp_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return comp_id


def get_latest_comparison() -> Optional[Dict[str, Any]]:
    """Retrieve the most recent comparison benchmark."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM comparisons ORDER BY id DESC LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    try:
        return json.loads(row["summary_json"])
    except Exception:
        return None


def save_simulation_run(sim_data: Dict[str, Any]) -> int:
    """Save an individual agent simulation run to SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    ts = sim_data.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    path_json = json.dumps(sim_data.get("path", []))
    start_str = json.dumps(sim_data.get("start", [0, 0]))
    goal_str = json.dumps(sim_data.get("goal", [0, 0]))

    cursor.execute("""
    INSERT INTO simulation_history (
        timestamp, algorithm, start_pos, goal_pos, steps, total_reward, success, path_json, status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ts,
        sim_data.get("algorithm", "Unknown"),
        start_str,
        goal_str,
        int(sim_data.get("steps", 0)),
        float(sim_data.get("total_reward", 0.0)),
        1 if sim_data.get("success", False) else 0,
        path_json,
        sim_data.get("status", "completed")
    ))
    sim_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return sim_id


def get_simulation_history(limit: int = 25) -> List[Dict[str, Any]]:
    """Retrieve recent simulation executions from SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM simulation_history ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("path_json"):
            try:
                d["path"] = json.loads(d["path_json"])
            except Exception:
                d["path"] = []
        if d.get("start_pos"):
            try:
                d["start_pos"] = json.loads(d["start_pos"])
            except Exception:
                pass
        if d.get("goal_pos"):
            try:
                d["goal_pos"] = json.loads(d["goal_pos"])
            except Exception:
                pass
        results.append(d)
    conn.close()
    return results


def get_simulation_stats() -> Dict[str, Any]:
    """Compute summary metrics across recorded simulations."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT 
        COUNT(*) as total_simulations,
        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_simulations,
        AVG(total_reward) as avg_reward,
        AVG(steps) as avg_steps
    FROM simulation_history
    """)
    row = cursor.fetchone()
    conn.close()
    if not row or row["total_simulations"] == 0:
        return {
            "total_simulations": 0,
            "successful_simulations": 0,
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "avg_steps": 0.0
        }
    tot = row["total_simulations"]
    succ = row["successful_simulations"] or 0
    return {
        "total_simulations": tot,
        "successful_simulations": succ,
        "success_rate": round((succ / tot) * 100.0, 1),
        "avg_reward": round(row["avg_reward"] or 0.0, 2),
        "avg_steps": round(row["avg_steps"] or 0.0, 1)
    }
