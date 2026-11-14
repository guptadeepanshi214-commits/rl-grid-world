/**
 * Grid World Interactive Visualization & Simulation Engine
 * Handles DOM grid rendering, cell interaction, policy overlays, and agent playback.
 */

class GridWorldVisualizer {
    constructor(containerId, hudIds = {}) {
        this.container = document.getElementById(containerId);
        this.hud = {
            pos: document.getElementById(hudIds.pos || 'hud-agent-pos'),
            steps: document.getElementById(hudIds.steps || 'hud-steps'),
            reward: document.getElementById(hudIds.reward || 'hud-reward'),
            action: document.getElementById(hudIds.action || 'hud-action'),
            status: document.getElementById(hudIds.status || 'hud-status')
        };

        // Grid state
        this.rows = 10;
        this.cols = 10;
        this.start = [0, 0];
        this.goal = [9, 9];
        this.obstacles = new Set();
        this.agentPos = [0, 0];

        // Interaction mode: 'toggle_obstacle', 'set_start', 'set_goal'
        this.interactionMode = 'toggle_obstacle';

        // Overlays
        this.showPolicyArrows = false;
        this.showHeatmap = false;
        this.showCoords = false;
        this.policyData = null; // 2D array from /api/results

        // Playback state
        this.animationTimer = null;
        this.currentPath = [];
        this.currentActions = [];
        this.currentStepIndex = 0;
        this.playbackSpeedMs = 200; // Normal speed
        this.accumulatedReward = 0.0;
        this.isPlaying = false;

        this.init();
    }

    init() {
        this.loadEnvironment();
    }

    async loadEnvironment() {
        try {
            const res = await fetch('/api/environment');
            if (!res.ok) throw new Error('Failed to load environment');
            const data = await res.json();
            this.setEnvironmentData(data);
        } catch (err) {
            console.error('[GridWorld] Load error:', err);
        }
    }

    setEnvironmentData(data) {
        this.rows = data.rows || 10;
        this.cols = data.cols || 10;
        this.start = data.start || [0, 0];
        this.goal = data.goal || [this.rows - 1, this.cols - 1];
        this.obstacles = new Set((data.obstacles || []).map(o => `${o[0]},${o[1]}`));
        this.agentPos = [...this.start];
        this.accumulatedReward = 0.0;
        this.currentStepIndex = 0;
        this.render();
        this.updateHUD();
    }

    setInteractionMode(mode) {
        this.interactionMode = mode;
    }

    setPlaybackSpeed(speedStr) {
        const speedMap = {
            'slow': 450,
            'normal': 200,
            'fast': 80,
            'ultra': 30
        };
        this.playbackSpeedMs = speedMap[speedStr] || 200;
        if (this.isPlaying) {
            this.pauseSimulation();
            this.startSimulation();
        }
    }

    togglePolicyArrows(show) {
        this.showPolicyArrows = show !== undefined ? show : !this.showPolicyArrows;
        this.render();
    }

    toggleHeatmap(show) {
        this.showHeatmap = show !== undefined ? show : !this.showHeatmap;
        this.render();
    }

    toggleCoords(show) {
        this.showCoords = show !== undefined ? show : !this.showCoords;
        this.render();
    }

    setPolicyData(policy) {
        this.policyData = policy;
        this.render();
    }

    render() {
        if (!this.container) return;

        this.container.innerHTML = '';
        const board = document.createElement('div');
        board.className = 'grid-board';
        board.style.gridTemplateColumns = `repeat(${this.cols}, var(--grid-cell-size, 44px))`;

        // Calculate min/max Q for heatmap scaling
        let minQ = 0;
        let maxQ = 100;
        if (this.policyData && this.showHeatmap) {
            const allQ = [];
            this.policyData.forEach(row => row.forEach(cell => {
                if (!cell.is_obstacle && !cell.is_goal) allQ.push(cell.max_q);
            }));
            if (allQ.length > 0) {
                minQ = Math.min(...allQ);
                maxQ = Math.max(...allQ);
            }
        }

        // Highlight path cells
        const pathSet = new Set(this.currentPath.slice(0, this.currentStepIndex + 1).map(p => `${p[0]},${p[1]}`));

        for (let r = 0; r < this.rows; r++) {
            for (let c = 0; c < this.cols; c++) {
                const cell = document.createElement('div');
                cell.className = 'grid-cell';
                cell.dataset.row = r;
                cell.dataset.col = c;

                const isStart = (r === this.start[0] && c === this.start[1]);
                const isGoal = (r === this.goal[0] && c === this.goal[1]);
                const isObstacle = this.obstacles.has(`${r},${c}`);
                const isAgent = (r === this.agentPos[0] && c === this.agentPos[1]);
                const inPath = pathSet.has(`${r},${c}`);

                if (isStart) cell.classList.add('cell-start');
                else if (isGoal) cell.classList.add('cell-goal');
                else if (isObstacle) cell.classList.add('cell-obstacle');
                else if (inPath) cell.classList.add('path-highlight');

                // Heatmap background calculation
                if (this.showHeatmap && this.policyData && !isObstacle && !isGoal && !isStart) {
                    const cellData = this.policyData[r] && this.policyData[r][c];
                    if (cellData && maxQ > minQ) {
                        const ratio = Math.max(0, Math.min(1, (cellData.max_q - minQ) / (maxQ - minQ)));
                        // Interpolate dark navy to luminous cyan
                        const rCol = Math.round(18 + ratio * 20);
                        const gCol = Math.round(28 + ratio * 180);
                        const bCol = Math.round(51 + ratio * 200);
                        cell.style.backgroundColor = `rgb(${rCol}, ${gCol}, ${bCol})`;

                        const tag = document.createElement('span');
                        tag.className = 'q-value-tag';
                        tag.innerText = cellData.max_q.toFixed(1);
                        cell.appendChild(tag);
                    }
                }

                // Policy direction arrows
                if (this.showPolicyArrows && this.policyData && !isObstacle && !isGoal) {
                    const cellData = this.policyData[r] && this.policyData[r][c];
                    if (cellData) {
                        const arrowSpan = document.createElement('span');
                        arrowSpan.className = 'policy-arrow';
                        const arrows = ['↑', '↓', '←', '→']; // UP=0, DOWN=1, LEFT=2, RIGHT=3
                        arrowSpan.innerText = arrows[cellData.best_action] || '';
                        cell.appendChild(arrowSpan);
                    }
                }

                // Coordinate label
                if (this.showCoords && !isStart && !isGoal && !isObstacle) {
                    const coordLabel = document.createElement('span');
                    coordLabel.style.fontSize = '0.55rem';
                    coordLabel.style.opacity = '0.5';
                    coordLabel.style.position = 'absolute';
                    coordLabel.style.top = '2px';
                    coordLabel.innerText = `${r},${c}`;
                    cell.appendChild(coordLabel);
                }

                // Agent character avatar
                if (isAgent) {
                    const agentEl = document.createElement('div');
                    agentEl.className = 'agent-character';
                    agentEl.innerHTML = '<i class="bi bi-robot"></i>';
                    cell.appendChild(agentEl);
                }

                // Cell click listener
                cell.addEventListener('click', () => this.handleCellClick(r, c));

                board.appendChild(cell);
            }
        }

        this.container.appendChild(board);
    }

    async handleCellClick(r, c) {
        if (this.interactionMode === 'toggle_obstacle') {
            if ((r === this.start[0] && c === this.start[1]) || (r === this.goal[0] && c === this.goal[1])) {
                return;
            }
            try {
                const res = await fetch('/api/environment/update', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ toggle_obstacle: [r, c] })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    this.setEnvironmentData(data.environment);
                }
            } catch (err) {
                console.error('[GridWorld] Error toggling obstacle:', err);
            }
        } else if (this.interactionMode === 'set_start') {
            if (this.obstacles.has(`${r},${c}`) || (r === this.goal[0] && c === this.goal[1])) return;
            this.start = [r, c];
            this.agentPos = [r, c];
            await this.updateEnvironmentLayout();
        } else if (this.interactionMode === 'set_goal') {
            if (this.obstacles.has(`${r},${c}`) || (r === this.start[0] && c === this.start[1])) return;
            this.goal = [r, c];
            await this.updateEnvironmentLayout();
        }
    }

    async updateEnvironmentLayout() {
        try {
            const obsList = Array.from(this.obstacles).map(str => {
                const parts = str.split(',').map(Number);
                return [parts[0], parts[1]];
            });
            const res = await fetch('/api/environment/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    rows: this.rows,
                    cols: this.cols,
                    start: this.start,
                    goal: this.goal,
                    obstacles: obsList
                })
            });
            const data = await res.json();
            if (data.status === 'success') {
                this.setEnvironmentData(data.environment);
            }
        } catch (err) {
            console.error('[GridWorld] Layout update error:', err);
        }
    }

    async resetToPreset(presetName = 'default') {
        this.pauseSimulation();
        try {
            const res = await fetch('/api/environment/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ preset: presetName, rows: this.rows, cols: this.cols })
            });
            const data = await res.json();
            if (data.status === 'success') {
                this.setEnvironmentData(data.environment);
            }
        } catch (err) {
            console.error('[GridWorld] Reset preset error:', err);
        }
    }

    async changeGridDimensions(newRows, newCols) {
        this.pauseSimulation();
        this.rows = Math.max(4, Math.min(20, newRows));
        this.cols = Math.max(4, Math.min(20, newCols));
        this.start = [0, 0];
        this.goal = [this.rows - 1, this.cols - 1];
        await this.resetToPreset('default');
    }

    // ==========================================
    // Simulation & Path Playback Controls
    // ==========================================

    async loadAgentSimulation(algorithmName = 'current') {
        try {
            const res = await fetch('/api/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ algorithm: algorithmName, max_steps: 150 })
            });
            const data = await res.json();
            if (data.status === 'success' && data.path && data.path.length > 0) {
                this.currentPath = data.path;
                this.currentActions = data.actions || [];
                this.currentStepIndex = 0;
                this.agentPos = [...this.currentPath[0]];
                this.accumulatedReward = 0.0;
                this.render();
                this.updateHUD('Simulation trajectory loaded. Click Start.');
                return data;
            } else {
                this.updateHUD('No valid path found for agent.');
                return null;
            }
        } catch (err) {
            console.error('[GridWorld] Simulation error:', err);
            return null;
        }
    }

    startSimulation() {
        if (!this.currentPath || this.currentPath.length <= 1) {
            this.loadAgentSimulation('current').then(data => {
                if (data) this.startSimulation();
            });
            return;
        }

        if (this.currentStepIndex >= this.currentPath.length - 1) {
            // Restart from beginning
            this.resetSimulation();
        }

        this.isPlaying = true;
        clearInterval(this.animationTimer);
        this.animationTimer = setInterval(() => {
            this.stepForward();
        }, this.playbackSpeedMs);

        this.updateHUD('Navigating environment...');
    }

    pauseSimulation() {
        this.isPlaying = false;
        clearInterval(this.animationTimer);
        this.updateHUD('Simulation paused');
    }

    stepForward() {
        if (this.currentStepIndex < this.currentPath.length - 1) {
            this.currentStepIndex++;
            this.agentPos = [...this.currentPath[this.currentStepIndex]];

            const isGoal = (this.agentPos[0] === this.goal[0] && this.agentPos[1] === this.goal[1]);
            const prevPos = this.currentPath[this.currentStepIndex - 1];
            // Reward calculation
            if (isGoal) {
                this.accumulatedReward += 100.0;
            } else {
                this.accumulatedReward -= 1.0;
            }

            this.render();

            let statusMsg = 'Navigating...';
            if (isGoal) {
                statusMsg = '🎉 Goal Reached! (+100)';
                this.pauseSimulation();
            }
            this.updateHUD(statusMsg);
        } else {
            this.pauseSimulation();
            this.updateHUD('Path completed');
        }
    }

    resetSimulation() {
        this.pauseSimulation();
        this.currentStepIndex = 0;
        this.agentPos = [...this.start];
        this.accumulatedReward = 0.0;
        this.render();
        this.updateHUD('Reset to start position');
    }

    updateHUD(statusOverride = null) {
        if (this.hud.pos) {
            this.hud.pos.innerText = `(${this.agentPos[0]}, ${this.agentPos[1]})`;
        }
        if (this.hud.steps) {
            this.hud.steps.innerText = this.currentStepIndex;
        }
        if (this.hud.reward) {
            this.hud.reward.innerText = this.accumulatedReward.toFixed(1);
            this.hud.reward.style.color = this.accumulatedReward >= 0 ? '#10b981' : '#f87171';
        }
        if (this.hud.action) {
            const currentAct = (this.currentActions && this.currentActions[this.currentStepIndex - 1]) || 'START';
            this.hud.action.innerText = currentAct;
        }
        if (this.hud.status) {
            let msg = statusOverride;
            if (!msg) {
                if (this.agentPos[0] === this.goal[0] && this.agentPos[1] === this.goal[1]) {
                    msg = '🎉 Goal Reached!';
                } else if (this.isPlaying) {
                    msg = 'Navigating...';
                } else {
                    msg = 'Ready';
                }
            }
            this.hud.status.innerText = msg;
        }
    }
}

// Export instance
window.GridWorldVisualizer = GridWorldVisualizer;
