# Reinforcement Learning Agent for Sequential Decision Environments

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.1-black.svg)](https://flask.palletsprojects.com/)
[![NumPy](https://img.shields.io/badge/Computations-NumPy-013243.svg)](https://numpy.org/)
[![Chart.js](https://img.shields.io/badge/Visualization-Chart.js%204.4-FF6384.svg)](https://www.chartjs.org/)
[![Database](https://img.shields.io/badge/Database-SQLite%203-003B57.svg)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/Academic-AI%20Foundations%20Project-success.svg)]()

> A full-stack, research-grade web application developed for college **Artificial Intelligence Foundations** courses. Demonstrates sequential decision-making in discrete Markov Decision Processes (MDPs) by implementing, training, visualizing, and empirically comparing **Q-Learning**, **SARSA**, and a classical **Goal-Based Agent** (BFS Baseline) in an interactive Grid World.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Technology Stack](#-technology-stack)
4. [Algorithmic Formulation](#-algorithmic-formulation)
   - [Q-Learning (Off-Policy TD)](#1-q-learning-off-policy-temporal-difference-control)
   - [SARSA (On-Policy TD)](#2-sarsa-on-policy-temporal-difference-control)
   - [Goal-Based Agent (BFS Baseline)](#3-goal-based-agent-bfs-baseline)
   - [Algorithmic Comparison & Risk Handling](#4-algorithmic-comparison--risk-handling)
5. [Application Architecture](#-application-architecture)
6. [Folder Structure](#-folder-structure)
7. [Installation & Setup](#-installation--setup)
8. [Running Unit Tests](#-running-unit-tests)
9. [REST API Documentation](#-rest-api-documentation)
10. [College Presentation & Viva Guide](#-college-presentation--viva-guide)

---

## 🌟 Project Overview

Autonomous agents operating in real-world sequential decision-making environments face the fundamental challenge of navigating from an origin to a destination without prior knowledge of transitions or obstacle placements.

This application implements **model-free Reinforcement Learning** algorithms from first principles (pure mathematical update equations using NumPy) without relying on black-box external RL frameworks (like Gymnasium, Stable-Baselines, or RLlib). It enables students and researchers to interactively observe how agents balance **exploration vs. exploitation**, update their internal value estimates ($Q$-tables), extract optimal policies ($\pi$), and avoid hazardous obstacles in real-time.

---

## 🚀 Key Features

### 1. Modern AI Dashboard (Dark Navy & Cyan Theme)
- Clean, responsive UI with sidebar navigation across 7 dedicated views.
- Real-time KPI counters: Total Training Episodes, Active Algorithm, Average Reward, Success Rate, Successful Episodes.
- Interactive recent training sessions table integrated with SQLite history.

### 2. Interactive Grid World Navigation
- Dynamic 2D grid canvas (default 10&times;10, customizable from 4&times;4 up to 20&times;20).
- Visual elements:
  - **Start Position**: Green glowing cell with pulse effect.
  - **Goal Position**: Radiant gold cell with trophy icon ($+100$ reward).
  - **Obstacles**: Dark hazard-striped barrier cells ($-10$ collision penalty).
  - **Agent Avatar**: Blue animated robot character with smooth SVG/CSS movements.
- Interactive capabilities:
  - Add/remove obstacles by clicking on cells.
  - Relocate Start or Goal positions.
  - Pre-configured obstacle presets: *Default Challenge Maze*, *Four Rooms Benchmark*, *Bottleneck Corridor*, *Cliff Walking*, and *Open Field*.
  - **Policy Overlays**: Toggle directional best-action arrows ($\uparrow, \downarrow, \leftarrow, \rightarrow$) and state-value ($V(s) = \max_a Q(s,a)$) color gradients/heatmaps.
  - Step-by-step playback or auto-play with adjustable speed (Slow, Normal, Fast, Ultra).

### 3. Asynchronous Multi-Threaded Training Panel
- Algorithm selection: **Q-Learning**, **SARSA**, or **Goal-Based Agent**.
- Live configurable hyperparameters:
  - Learning Rate ($\alpha$): $0.01 - 1.0$ (Default: $0.1$)
  - Discount Factor ($\gamma$): $0.10 - 0.99$ (Default: $0.95$)
  - Exploration Rate ($\epsilon$): $0.0 - 1.0$ (Default: $1.0$)
  - Epsilon Decay: $0.900 - 0.999$ (Default: $0.995$)
  - Minimum Epsilon ($\epsilon_{\min}$): $0.001 - 0.30$ (Default: $0.01$)
  - Number of Episodes: $50 - 3000$ (Default: $1000$)
  - Max Steps per Episode: $30 - 400$ (Default: $200$)
- Real-time telemetry: Episode counter, current reward, 50-episode rolling average reward, exploration rate, success rate, progress bar, and live streaming Chart.js mini-graph.
- Controls: **Start**, **Pause**, **Resume**, **Stop**, and **Reset**.

### 4. Algorithm Comparison & Live Benchmark Engine
- "Run Live 3-Agent Benchmark" button executes Q-Learning, SARSA, and Goal-Based BFS on identical environments and displays real empirical metrics.
- Side-by-side comparison cards, bar charts, learning convergence curves, and comprehensive structured comparison tables.
- Demonstrates the famous **Cliff Walking Phenomenon**: Q-Learning learns the optimal edge route, while SARSA chooses a safer path with obstacle clearance.

### 5. Dedicated Performance Analytics (6 Interactive Chart.js Graphs)
1. **Reward vs Episodes Graph**: Raw episode rewards displaying trial-and-error volatility.
2. **Average Reward Graph**: Smoothed rolling moving average demonstrating convergence stability.
3. **Success Rate Graph (%)**: Goal completion percentage over time.
4. **Exploration Rate ($\epsilon$) Decay Graph**: Exponential decay curve transitioning from exploration to exploitation.
5. **Steps per Episode Graph**: Step reduction demonstrating navigational path efficiency.
6. **Q-Learning vs SARSA Performance Graph**: Direct head-to-head convergence comparison.
- Historical session loader and JSON data export.

### 6. Comprehensive Learning Theory & KaTeX Formulations
- Complete educational textbook-style documentation explaining MDPs, agent, environment, state space, action space, reward functions, policies, exploration vs. exploitation, and Bellman equations.
- Exact LaTeX mathematical formulas rendered cleanly with KaTeX.

### 7. SQLite Database Persistence
- Automatic schema creation (`training.db`).
- Persists session UUIDs, hyperparameters, durations, final evaluations, optimal coordinate paths, and sampled episode metrics.

---

## 💻 Technology Stack

| Layer | Technologies | Purpose |
|---|---|---|
| **Frontend** | HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3, Bootstrap Icons | Responsive UI, sidebar navigation, grid canvas |
| **Data Viz** | Chart.js 4.4, KaTeX 0.16 | Real-time performance curves, mathematical formulas |
| **Backend** | Python 3.10+, Flask 3.1, Werkzeug | REST API, background threading, session dispatching |
| **AI / ML** | NumPy, Custom Python Algorithms | $Q$-table updates, $\epsilon$-greedy decay, BFS solver |
| **Database** | SQLite 3 | Persistent storage for training logs and benchmarks |
| **Testing** | Python `unittest` | Unit & integration tests for environment, algorithms, and APIs |

---

## 📐 Algorithmic Formulation

### 1. Q-Learning (Off-Policy Temporal Difference Control)
*Watkins, 1989*

Q-Learning is an **off-policy** TD algorithm that estimates the optimal action-value function $Q^*(s, a)$ independently of the agent's behavioral exploration policy.

**Update Equation:**
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

- $\alpha \in (0, 1]$: Learning rate.
- $\gamma \in [0, 1)$: Discount factor.
- $r$: Immediate reward ($+100$ goal, $-10$ obstacle, $-1$ step, $-5$ boundary).
- $\max_{a'} Q(s', a')$: Bellman optimality target (assumes the greedy best action in the next state).
- $\left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$: Temporal Difference (TD) error.

---

### 2. SARSA (On-Policy Temporal Difference Control)
*Rummery & Niranjan, 1994; Sutton, 1996*

SARSA (**State-Action-Reward-State-Action**) is an **on-policy** TD control algorithm. Instead of using the maximum possible next Q-value, it updates using the value of the action $a'$ chosen by the current behavioral ($\epsilon$-greedy) policy.

**Update Equation:**
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

Where $a'$ is sampled from state $s'$ using the same $\epsilon$-greedy policy:
$$a' = \begin{cases} \text{random action} & \text{with probability } \epsilon \\ \arg\max_{a} Q(s', a) & \text{with probability } 1 - \epsilon \end{cases}$$

---

### 3. Goal-Based Agent (BFS Baseline)
*Classical Graph Search*

> **IMPORTANT:** The Goal-Based Agent is **NOT** a reinforcement learning algorithm. It does not learn through trial-and-error experience. Instead, it utilizes complete prior knowledge of the environment grid and obstacle coordinates to execute a **Breadth-First Search (BFS)** graph traversal, finding the deterministic shortest path.

It serves as an analytical **ground-truth upper bound** for evaluating how closely the RL agents approach theoretical optimality.

---

### 4. Algorithmic Comparison & Risk Handling

| Dimension | Q-Learning | SARSA | Goal-Based Agent |
|---|---|---|---|
| **Learning Paradigm** | Model-Free Reinforcement Learning | Model-Free Reinforcement Learning | Model-Based Classical Search |
| **Policy Strategy** | **Off-Policy** (target is greedy) | **On-Policy** (target is behavioral) | Deterministic BFS |
| **Update Target** | $r + \gamma \max_{a'} Q(s', a')$ | $r + \gamma Q(s', a')$ | Analytical Shortest Path |
| **Model Requirement** | Zero model knowledge required | Zero model knowledge required | Complete map & obstacle graph |
| **Risk Around Obstacles** | **Aggressive / Optimal**: Skims directly alongside obstacles to minimize steps | **Conservative / Safe**: Steers clear of obstacles to avoid accidental exploration collisions | **Zero Risk**: Path calculated offline before motion |
| **Convergence Guarantee** | Converges to $Q^*(s,a)$ under standard Robbins-Monro conditions | Converges to optimal policy under GLIE conditions | Guaranteed optimal path if graph connected |

---

## 🏗️ Application Architecture

```
                    +------------------------------------+
                    |       Web Browser (Client)         |
                    |   Bootstrap 5 + Chart.js + KaTeX   |
                    +-----------------+------------------+
                                      |
                     HTTP / REST API  |  JSON Responses
                                      v
                    +------------------------------------+
                    |         Flask Web Server           |
                    |             (app.py)               |
                    +-----------------+------------------+
                                      |
            +-------------------------+-------------------------+
            |                         |                         |
            v                         v                         v
+-----------------------+ +-----------------------+ +-----------------------+
|   GridWorldEnv        | |   TrainingManager     | |   Evaluation Engine   |
| (backend/environment) | | (backend/training_mgr)| | (backend/evaluation)  |
+-----------------------+ +-----------+-----------+ +-----------------------+
                                      |
            +-------------------------+-------------------------+
            |                         |                         |
            v                         v                         v
+-----------------------+ +-----------------------+ +-----------------------+
|    QLearningAgent     | |      SarsaAgent       | |    GoalBasedAgent     |
| (backend/q_learning)  | |    (backend/sarsa)    | |  (backend/goal_agent) |
+-----------------------+ +-----------------------+ +-----------------------+
                                      |
                                      v
                          +-----------------------+
                          |   SQLite Database     |
                          | (database/training.db)|
                          +-----------------------+
```

---

## 📁 Folder Structure

```
reinforcement-learning-project/
│
├── app.py                      # Main Flask application & REST API endpoints
├── requirements.txt            # Python dependencies (Flask, NumPy)
├── README.md                   # Comprehensive project documentation
│
├── backend/                    # Core Python AI / ML modules
│   ├── __init__.py
│   ├── environment.py          # 2D Grid World MDP environment
│   ├── q_learning.py           # Q-Learning agent with epsilon-greedy policy
│   ├── sarsa.py                # SARSA agent with on-policy update
│   ├── goal_agent.py           # Classical Goal-Based BFS baseline solver
│   ├── training_manager.py     # Multi-threaded background training supervisor
│   ├── evaluation.py           # Greedy policy evaluator & benchmark engine
│   └── database.py             # SQLite persistence layer
│
├── frontend/                   # User interface assets
│   ├── templates/
│   │   └── index.html          # Modern Single Page Application dashboard
│   └── static/
│       ├── css/
│       │   └── style.css       # Custom AI dashboard dark navy/cyan theme
│       └── js/
│           ├── app.js          # Main UI controller & API synchronization
│           ├── charts.js       # Chart.js telemetry & comparison charts
│           └── gridworld.js    # Interactive DOM grid visualization & player
│
├── database/                   # SQLite database storage
│   ├── __init__.py
│   └── training.db             # Auto-generated database file
│
└── tests/                      # Automated test suite
    ├── __init__.py
    ├── test_environment.py     # Grid world movement & collision tests
    ├── test_algorithms.py      # Q-learning, SARSA, and BFS solver tests
    └── test_api.py             # Flask REST API endpoints unit tests
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- `pip` package manager
- Any modern web browser (Google Chrome, Microsoft Edge, Mozilla Firefox)

### 1. Clone or Open the Workspace
```bash
cd c:\Users\HP\Downloads\aa
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

*(Dependencies are strictly standard: `flask>=3.0.0` and `numpy>=1.24.0`. No API keys or paid services are required).*

### 3. Launch the Application Server
```bash
python app.py
```

You should see output similar to:
```
Starting Reinforcement Learning Agent Web Application on http://127.0.0.1:5000
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
```

### 4. Open in Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🧪 Running Unit Tests

The test suite covers environment transitions, obstacle collisions, boundary checks, reward formulas, epsilon decay, SARSA and Q-Learning Bellman updates, BFS pathfinding, and REST API routes.

To run the complete test suite:
```bash
python -m unittest discover tests
```

Expected output:
```
...................
----------------------------------------------------------------------
Ran 19 tests in 0.096s

OK
```

---

## 📡 REST API Documentation

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | `GET` | Renders the main SPA dashboard (`index.html`) |
| `GET /api/health` | `GET` | Returns server health status and version |
| `GET /api/dashboard` | `GET` | Fetches overall training statistics, active model, and recent sessions |
| `GET /api/environment` | `GET` | Returns current grid layout (dimensions, start, goal, obstacles) |
| `POST /api/environment/reset` | `POST` | Resets grid to a preset (`default`, `four_rooms`, `bottleneck`, `cliff`, `empty`) |
| `POST /api/environment/update` | `POST` | Updates grid dimensions, sets start/goal, or toggles an obstacle cell |
| `POST /api/train` | `POST` | Starts asynchronous training with specified hyperparameters |
| `POST /api/train/pause` | `POST` | Pauses running training loop |
| `POST /api/train/resume` | `POST` | Resumes paused training |
| `POST /api/train/stop` | `POST` | Stops training and persists session |
| `POST /api/train/reset` | `POST` | Resets training telemetry back to idle |
| `GET /api/train/status` | `GET` | Real-time status polling for progress bar and live telemetry |
| `GET /api/results` | `GET` | Retrieves learned greedy policy grid and optimal path coordinates |
| `GET /api/comparison` | `GET` | Fetches comparative metrics between Q-Learning, SARSA, and BFS |
| `POST /api/comparison/run` | `POST` | Runs live 3-agent benchmark on current grid environment |
| `GET /api/analytics` | `GET` | Returns historical series data for the 6 Chart.js graphs |
| `POST /api/simulate` | `POST` | Returns step-by-step navigational trajectory for frontend animation |
| `GET /api/history` | `GET` | Lists previous training sessions saved in SQLite |
| `GET /api/history/<id>` | `GET` | Retrieves full episode log details for a specific session |

---

## 🎓 College Presentation & Viva Guide

### Recommended 5-Minute Live Demonstration Flow

1. **Introduction (1 min):**
   - Open the **Dashboard** (`Page 1`). Explain that the project explores sequential decision making in Markov Decision Processes using model-free Reinforcement Learning.
   - Point out the KPI counters and explain the distinction between reinforcement learning (trial and error) vs supervised learning.

2. **The Environment & Exploration (1.5 min):**
   - Switch to **Grid World Environment** (`Page 2`).
   - Explain the rules: Green = Start $(0,0)$, Gold = Goal $(9,9)$ with $+100$ reward, Dark = Obstacles with $-10$ penalty, Normal step = $-1$ penalty.
   - Toggle an obstacle cell or select the **Four Rooms** preset to show environment configurability.

3. **Training the Agent (1 min):**
   - Switch to **Train Agent** (`Page 3`).
   - Select **Q-Learning**, keep default parameters ($\alpha=0.1, \gamma=0.95, \text{episodes}=1000$), and click **Start Training**.
   - Watch the live telemetry: show how $\epsilon$ decays from $1.0 \rightarrow 0.01$, and how rolling average reward climbs from negative numbers to $+80$ or higher as the agent converges.

4. **Visualizing the Learned Policy (1 min):**
   - Click **Test in Grid World** or switch to `Page 2`.
   - Click **Show Policy Direction Arrows** to reveal the arrows pointing along the optimal route.
   - Click **Show Q-Value State Heatmap** to show how value propagates backwards from the goal.
   - Click **Start Simulation** to watch the animated blue robot navigate seamlessly to the goal!

5. **Comparison & Analytical Insights (1 min):**
   - Switch to **Algorithm Comparison** (`Page 4`).
   - Click **Run Live 3-Agent Benchmark**.
   - Highlight the comparison table and the difference between Q-Learning (Off-policy, takes the tightest path) vs SARSA (On-policy, takes the safer route with obstacle clearance) vs Goal-Based BFS (ground-truth baseline).
   - Show the 6 graphs in **Performance Analytics** (`Page 5`) and review the mathematical formulations in **Learning Theory** (`Page 6`).

### Typical Viva Questions & Model Answers

**Q1: What is the core difference between Q-Learning and SARSA?**
> *Answer:* Q-Learning is an **off-policy** algorithm. Its update target uses $\max_{a'} Q(s', a')$, assuming the agent will act greedily in the next state, irrespective of the exploration policy. SARSA is an **on-policy** algorithm. Its update target uses $Q(s', a')$ where $a'$ is the action actually chosen by the behavior policy. In hazardous environments with obstacles, SARSA learns a conservative path to avoid accidental exploratory collisions, whereas Q-Learning learns the optimal shortest path.

**Q2: Why is the Goal-Based Agent included, and is it an RL agent?**
> *Answer:* No, the Goal-Based Agent is explicitly **not** a reinforcement learning agent. It is a classical Breadth-First Search (BFS) graph algorithm that requires complete model knowledge (the entire grid and obstacle map). It is included as a theoretical benchmark upper-bound to demonstrate what perfect knowledge achieves compared to what an RL agent learns from scratch.

**Q3: Why does epsilon ($\epsilon$) decay over episodes?**
> *Answer:* This addresses the **exploration vs. exploitation dilemma**. In early episodes, the agent knows nothing about the environment, so high $\epsilon$ (near $1.0$) ensures widespread exploration. As training progresses and the Q-table acquires accurate estimates, $\epsilon$ decays towards $\epsilon_{\min}$ ($0.01$), shifting the agent's behavior towards exploiting its learned knowledge.

---

## 📄 License & Academic Integrity
Developed for educational and academic demonstration purposes in Artificial Intelligence and Machine Learning coursework. Code is cleanly modularized, documented, and fully reproducible.
