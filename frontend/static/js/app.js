/**
 * Main Application Controller for RL Agent Dashboard
 * Manages SPA navigation, API synchronization, training telemetry, and view rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
    window.App = new RLApp();
});

class RLApp {
    constructor() {
        this.currentPage = 'dashboard';
        this.selectedAlgorithm = 'q_learning';
        this.trainingPollTimer = null;
        this.gridVisualizer = null;

        this.init();
    }

    async init() {
        this.setupNavigation();
        this.setupGridWorld();
        this.setupTrainingControls();
        this.setupComparisonControls();
        this.setupAnalyticsControls();
        this.renderMathFormulas();

        // Initial data fetch
        await this.loadDashboardData();
        await this.loadComparisonData();
    }

    // ==========================================
    // Navigation & SPA Routing
    // ==========================================
    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-item-link');
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetPage = link.getAttribute('data-page');
                if (targetPage) this.navigateTo(targetPage);
            });
        });

        // Mobile sidebar toggle
        const toggleBtn = document.getElementById('sidebar-toggle');
        const sidebar = document.querySelector('.sidebar');
        if (toggleBtn && sidebar) {
            toggleBtn.addEventListener('click', () => {
                sidebar.classList.toggle('show-sidebar');
            });
        }
    }

    navigateTo(pageId) {
        this.currentPage = pageId;

        // Update nav items active state
        document.querySelectorAll('.nav-item-link').forEach(link => {
            if (link.getAttribute('data-page') === pageId) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });

        // Hide all page views and reveal active one
        document.querySelectorAll('.page-view').forEach(view => {
            view.classList.remove('active-page');
        });

        const activeView = document.getElementById(`page-${pageId}`);
        if (activeView) {
            activeView.classList.add('active-page');
        }

        // Close mobile sidebar if open
        const sidebar = document.querySelector('.sidebar');
        if (sidebar) sidebar.classList.remove('show-sidebar');

        // Page-specific initialization triggers
        if (pageId === 'dashboard') {
            this.loadDashboardData();
        } else if (pageId === 'gridworld') {
            if (this.gridVisualizer) {
                this.gridVisualizer.render();
            }
        } else if (pageId === 'train') {
            if (!AppCharts.trainMini) initTrainMiniChart();
            this.checkTrainingStatus();
        } else if (pageId === 'comparison') {
            this.loadComparisonData();
        } else if (pageId === 'analytics') {
            this.loadAnalyticsSessions();
            this.loadAnalyticsData();
        } else if (pageId === 'theory') {
            this.renderMathFormulas();
        }
    }

    // ==========================================
    // Page 1: Dashboard Controller
    // ==========================================
    async loadDashboardData() {
        try {
            const res = await fetch('/api/dashboard');
            if (!res.ok) throw new Error('Failed to fetch dashboard');
            const data = await res.json();

            // Populate KPIs
            const elTotalEp = document.getElementById('dash-total-episodes');
            const elAlgo = document.getElementById('dash-current-algo');
            const elAvgRew = document.getElementById('dash-avg-reward');
            const elSuccRate = document.getElementById('dash-success-rate');
            const elSuccEp = document.getElementById('dash-successful-episodes');

            if (elTotalEp) elTotalEp.innerText = Number(data.total_episodes || 0).toLocaleString();
            if (elAlgo) elAlgo.innerText = data.current_algorithm || 'Q-Learning';
            if (elAvgRew) {
                const rew = Number(data.average_reward || 0);
                elAvgRew.innerText = (rew > 0 ? '+' : '') + rew.toFixed(1);
                elAvgRew.style.color = rew >= 0 ? '#10b981' : '#f87171';
            }
            if (elSuccRate) elSuccRate.innerText = `${(data.success_rate || 0).toFixed(1)}%`;
            if (elSuccEp) elSuccEp.innerText = data.successful_episodes || 0;

            // Populate recent sessions table
            this.renderDashboardSessions(data.recent_sessions || []);
        } catch (err) {
            console.error('[Dashboard] Error:', err);
        }
    }

    renderDashboardSessions(sessions) {
        const tbody = document.getElementById('dash-recent-sessions-body');
        if (!tbody) return;

        if (sessions.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted py-4">No completed training sessions recorded yet. Run a training session to see records here.</td></tr>`;
            return;
        }

        tbody.innerHTML = sessions.map(s => `
            <tr>
                <td><span class="badge ${s.algorithm.includes('SARSA') ? 'badge-rl' : s.algorithm.includes('Q-Learning') ? 'badge-rl' : 'badge-baseline'}">${s.algorithm}</span></td>
                <td><strong>${s.episodes}</strong></td>
                <td style="color: ${s.avg_reward >= 0 ? '#10b981' : '#f87171'}">${s.avg_reward ? (s.avg_reward > 0 ? '+' : '') + s.avg_reward.toFixed(1) : '0.0'}</td>
                <td><span class="text-emerald font-monospace">${(s.success_rate || 0).toFixed(1)}%</span></td>
                <td>${s.avg_steps ? s.avg_steps.toFixed(1) : '-'}</td>
                <td class="text-muted small">${s.timestamp || '-'}</td>
                <td>
                    <button class="btn btn-sm btn-outline-info py-0 px-2" onclick="window.App.loadSessionToVisualizer(${s.id})">
                        <i class="bi bi-play-circle me-1"></i>Test
                    </button>
                </td>
            </tr>
        `).join('');
    }

    // ==========================================
    // Page 2: Grid World Controller
    // ==========================================
    setupGridWorld() {
        this.gridVisualizer = new GridWorldVisualizer('grid-world-canvas', {
            pos: 'hud-agent-pos',
            steps: 'hud-steps',
            reward: 'hud-reward',
            action: 'hud-action',
            status: 'hud-status'
        });

        // Preset Layout Selector
        const presetSelect = document.getElementById('grid-preset-select');
        if (presetSelect) {
            presetSelect.addEventListener('change', (e) => {
                this.gridVisualizer.resetToPreset(e.target.value);
            });
        }

        // Grid Dimensions Form
        const btnApplyDimensions = document.getElementById('btn-apply-dimensions');
        if (btnApplyDimensions) {
            btnApplyDimensions.addEventListener('click', () => {
                const rows = parseInt(document.getElementById('input-grid-rows').value, 10) || 10;
                const cols = parseInt(document.getElementById('input-grid-cols').value, 10) || 10;
                this.gridVisualizer.changeGridDimensions(rows, cols);
            });
        }

        // Clear Obstacles
        const btnClearObs = document.getElementById('btn-clear-obstacles');
        if (btnClearObs) {
            btnClearObs.addEventListener('click', () => {
                this.gridVisualizer.resetToPreset('empty');
            });
        }

        // Interaction Mode Radios
        const modeRadios = document.querySelectorAll('input[name="grid-interaction-mode"]');
        modeRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                if (e.target.checked) this.gridVisualizer.setInteractionMode(e.target.value);
            });
        });

        // Overlay Toggles
        const chkArrows = document.getElementById('toggle-policy-arrows');
        if (chkArrows) {
            chkArrows.addEventListener('change', (e) => this.gridVisualizer.togglePolicyArrows(e.target.checked));
        }

        const chkHeatmap = document.getElementById('toggle-heatmap');
        if (chkHeatmap) {
            chkHeatmap.addEventListener('change', (e) => this.gridVisualizer.toggleHeatmap(e.target.checked));
        }

        const chkCoords = document.getElementById('toggle-coords');
        if (chkCoords) {
            chkCoords.addEventListener('change', (e) => this.gridVisualizer.toggleCoords(e.target.checked));
        }

        // Simulation Buttons
        const btnStartSim = document.getElementById('btn-sim-start');
        if (btnStartSim) {
            btnStartSim.addEventListener('click', () => this.gridVisualizer.startSimulation());
        }

        const btnPauseSim = document.getElementById('btn-sim-pause');
        if (btnPauseSim) {
            btnPauseSim.addEventListener('click', () => this.gridVisualizer.pauseSimulation());
        }

        const btnStepSim = document.getElementById('btn-sim-step');
        if (btnStepSim) {
            btnStepSim.addEventListener('click', () => this.gridVisualizer.stepForward());
        }

        const btnResetSim = document.getElementById('btn-sim-reset');
        if (btnResetSim) {
            btnResetSim.addEventListener('click', () => this.gridVisualizer.resetSimulation());
        }

        // Speed Selector
        const speedSelect = document.getElementById('sim-speed-select');
        if (speedSelect) {
            speedSelect.addEventListener('change', (e) => this.gridVisualizer.setPlaybackSpeed(e.target.value));
        }

        // Load Learned Policy Button
        const btnLoadPolicy = document.getElementById('btn-load-learned-policy');
        if (btnLoadPolicy) {
            btnLoadPolicy.addEventListener('click', async () => {
                await this.loadActivePolicyToVisualizer();
            });
        }
    }

    async loadActivePolicyToVisualizer() {
        try {
            const res = await fetch('/api/results');
            if (!res.ok) throw new Error('Failed to fetch results');
            const data = await res.json();
            if (data.policy) {
                this.gridVisualizer.setPolicyData(data.policy);
                // Also load simulation path
                await this.gridVisualizer.loadAgentSimulation('current');
                // Enable arrows
                const chk = document.getElementById('toggle-policy-arrows');
                if (chk) chk.checked = true;
                this.gridVisualizer.togglePolicyArrows(true);
            }
        } catch (err) {
            console.error('[GridWorld] Load policy error:', err);
        }
    }

    async loadSessionToVisualizer(sessionId) {
        try {
            const res = await fetch(`/api/history/${sessionId}`);
            const data = await res.json();
            if (data.status === 'success' && data.session && data.session.optimal_path) {
                this.navigateTo('gridworld');
                this.gridVisualizer.currentPath = data.session.optimal_path;
                this.gridVisualizer.resetSimulation();
                this.gridVisualizer.startSimulation();
            }
        } catch (err) {
            console.error('[SessionVisualizer] Error:', err);
        }
    }

    // ==========================================
    // Page 3: Training Agent Controller
    // ==========================================
    setupTrainingControls() {
        // Algorithm Selection Cards
        const algoCards = document.querySelectorAll('.algo-card-select');
        algoCards.forEach(card => {
            card.addEventListener('click', () => {
                algoCards.forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                this.selectedAlgorithm = card.getAttribute('data-algo');
            });
        });

        // Hyperparameter Slider Synchronization
        const sliders = [
            { id: 'param-alpha', display: 'val-alpha', decimals: 2 },
            { id: 'param-gamma', display: 'val-gamma', decimals: 2 },
            { id: 'param-epsilon', display: 'val-epsilon', decimals: 2 },
            { id: 'param-decay', display: 'val-decay', decimals: 3 },
            { id: 'param-min-epsilon', display: 'val-min-epsilon', decimals: 3 },
            { id: 'param-episodes', display: 'val-episodes', decimals: 0 },
            { id: 'param-max-steps', display: 'val-max-steps', decimals: 0 }
        ];

        sliders.forEach(s => {
            const input = document.getElementById(s.id);
            const disp = document.getElementById(s.display);
            if (input && disp) {
                input.addEventListener('input', (e) => {
                    const val = parseFloat(e.target.value);
                    disp.innerText = s.decimals > 0 ? val.toFixed(s.decimals) : val;
                });
            }
        });

        // Training Action Buttons
        const btnStart = document.getElementById('btn-train-start');
        if (btnStart) btnStart.addEventListener('click', () => this.startTraining());

        const btnPause = document.getElementById('btn-train-pause');
        if (btnPause) btnPause.addEventListener('click', () => this.pauseTraining());

        const btnResume = document.getElementById('btn-train-resume');
        if (btnResume) btnResume.addEventListener('click', () => this.resumeTraining());

        const btnStop = document.getElementById('btn-train-stop');
        if (btnStop) btnStop.addEventListener('click', () => this.stopTraining());

        const btnReset = document.getElementById('btn-train-reset');
        if (btnReset) btnReset.addEventListener('click', () => this.resetTraining());
    }

    async startTraining() {
        const payload = {
            algorithm: this.selectedAlgorithm,
            alpha: parseFloat(document.getElementById('param-alpha').value),
            gamma: parseFloat(document.getElementById('param-gamma').value),
            epsilon: parseFloat(document.getElementById('param-epsilon').value),
            epsilon_decay: parseFloat(document.getElementById('param-decay').value),
            min_epsilon: parseFloat(document.getElementById('param-min-epsilon').value),
            episodes: parseInt(document.getElementById('param-episodes').value, 10),
            max_steps: parseInt(document.getElementById('param-max-steps').value, 10)
        };

        try {
            const res = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.status === 'success') {
                this.updateTrainingUIState('training');
                this.startTrainingPolling();
            } else {
                alert(`Training Error: ${data.message}`);
            }
        } catch (err) {
            console.error('[Train] Start error:', err);
        }
    }

    async pauseTraining() {
        await fetch('/api/train/pause', { method: 'POST' });
        this.updateTrainingUIState('paused');
    }

    async resumeTraining() {
        await fetch('/api/train/resume', { method: 'POST' });
        this.updateTrainingUIState('training');
    }

    async stopTraining() {
        await fetch('/api/train/stop', { method: 'POST' });
        this.updateTrainingUIState('stopped');
    }

    async resetTraining() {
        clearInterval(this.trainingPollTimer);
        await fetch('/api/train/reset', { method: 'POST' });
        this.updateTrainingUIState('idle');
        this.resetTrainingHUD();
    }

    startTrainingPolling() {
        clearInterval(this.trainingPollTimer);
        this.trainingPollTimer = setInterval(async () => {
            await this.checkTrainingStatus();
        }, 250);
    }

    async checkTrainingStatus() {
        try {
            const res = await fetch('/api/train/status');
            const data = await res.json();
            this.updateTrainingHUD(data);

            if (data.status === 'completed' || data.status === 'stopped' || data.status === 'error') {
                clearInterval(this.trainingPollTimer);
                this.updateTrainingUIState(data.status);
                // Trigger auto update for comparison and dashboard
                this.loadDashboardData();
            }
        } catch (err) {
            console.error('[Train] Polling error:', err);
        }
    }

    updateTrainingHUD(statusData) {
        const epEl = document.getElementById('train-hud-episode');
        const rewEl = document.getElementById('train-hud-reward');
        const avgRewEl = document.getElementById('train-hud-avg-reward');
        const epsEl = document.getElementById('train-hud-epsilon');
        const succEl = document.getElementById('train-hud-success-rate');
        const progBar = document.getElementById('train-progress-bar');
        const progPercent = document.getElementById('train-progress-percentage');
        const statusBadge = document.getElementById('train-status-badge');

        if (epEl) epEl.innerText = `${statusData.current_episode} / ${statusData.total_episodes}`;
        if (rewEl) {
            rewEl.innerText = (statusData.current_reward > 0 ? '+' : '') + statusData.current_reward.toFixed(1);
            rewEl.style.color = statusData.current_reward >= 0 ? '#10b981' : '#f87171';
        }
        if (avgRewEl) {
            avgRewEl.innerText = (statusData.average_reward > 0 ? '+' : '') + statusData.average_reward.toFixed(1);
            avgRewEl.style.color = statusData.average_reward >= 0 ? '#10b981' : '#f87171';
        }
        if (epsEl) epsEl.innerText = statusData.exploration_rate.toFixed(3);
        if (succEl) succEl.innerText = `${statusData.success_rate.toFixed(1)}%`;

        if (progBar) progBar.style.width = `${statusData.progress_percentage}%`;
        if (progPercent) progPercent.innerText = `${statusData.progress_percentage}%`;

        if (statusBadge) {
            statusBadge.innerText = statusData.status.toUpperCase();
            statusBadge.className = `badge ${statusData.status === 'training' ? 'bg-primary' : statusData.status === 'completed' ? 'bg-success' : statusData.status === 'paused' ? 'bg-warning' : 'bg-secondary'}`;
        }

        // Update live mini chart
        if (statusData.recent_history && statusData.recent_history.length > 0) {
            updateTrainMiniChart(statusData.recent_history);
        }
    }

    updateTrainingUIState(status) {
        const btnStart = document.getElementById('btn-train-start');
        const btnPause = document.getElementById('btn-train-pause');
        const btnResume = document.getElementById('btn-train-resume');
        const btnStop = document.getElementById('btn-train-stop');
        const completionAlert = document.getElementById('train-completed-alert');

        if (status === 'training') {
            if (btnStart) btnStart.disabled = true;
            if (btnPause) btnPause.disabled = false;
            if (btnResume) btnResume.disabled = true;
            if (btnStop) btnStop.disabled = false;
            if (completionAlert) completionAlert.classList.add('d-none');
        } else if (status === 'paused') {
            if (btnStart) btnStart.disabled = true;
            if (btnPause) btnPause.disabled = true;
            if (btnResume) btnResume.disabled = false;
            if (btnStop) btnStop.disabled = false;
        } else if (status === 'completed') {
            if (btnStart) btnStart.disabled = false;
            if (btnPause) btnPause.disabled = true;
            if (btnResume) btnResume.disabled = true;
            if (btnStop) btnStop.disabled = true;
            if (completionAlert) completionAlert.classList.remove('d-none');
        } else {
            // Idle or stopped
            if (btnStart) btnStart.disabled = false;
            if (btnPause) btnPause.disabled = true;
            if (btnResume) btnResume.disabled = true;
            if (btnStop) btnStop.disabled = true;
        }
    }

    resetTrainingHUD() {
        const epEl = document.getElementById('train-hud-episode');
        const rewEl = document.getElementById('train-hud-reward');
        const avgRewEl = document.getElementById('train-hud-avg-reward');
        const epsEl = document.getElementById('train-hud-epsilon');
        const succEl = document.getElementById('train-hud-success-rate');
        const progBar = document.getElementById('train-progress-bar');
        const progPercent = document.getElementById('train-progress-percentage');
        const statusBadge = document.getElementById('train-status-badge');

        if (epEl) epEl.innerText = '0 / 1000';
        if (rewEl) rewEl.innerText = '0.0';
        if (avgRewEl) avgRewEl.innerText = '0.0';
        if (epsEl) epsEl.innerText = '1.000';
        if (succEl) succEl.innerText = '0.0%';
        if (progBar) progBar.style.width = '0%';
        if (progPercent) progPercent.innerText = '0%';
        if (statusBadge) {
            statusBadge.innerText = 'IDLE';
            statusBadge.className = 'badge bg-secondary';
        }
        initTrainMiniChart();
    }

    // ==========================================
    // Page 4: Algorithm Comparison Controller
    // ==========================================
    setupComparisonControls() {
        const btnRunBenchmark = document.getElementById('btn-run-benchmark');
        if (btnRunBenchmark) {
            btnRunBenchmark.addEventListener('click', async () => {
                await this.runLiveBenchmark();
            });
        }
    }

    async loadComparisonData() {
        try {
            const res = await fetch('/api/comparison');
            const data = await res.json();
            if (data.status === 'success' && data.comparison) {
                this.renderComparisonView(data.comparison);
            }
        } catch (err) {
            console.error('[Comparison] Load error:', err);
        }
    }

    async runLiveBenchmark() {
        const btn = document.getElementById('btn-run-benchmark');
        const spinner = document.getElementById('benchmark-spinner');
        if (btn) btn.disabled = true;
        if (spinner) spinner.classList.remove('d-none');

        try {
            const res = await fetch('/api/comparison/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ episodes: 500, max_steps: 150 })
            });
            const data = await res.json();
            if (data.status === 'success' && data.comparison) {
                this.renderComparisonView(data.comparison);
            }
        } catch (err) {
            console.error('[Benchmark] Run error:', err);
        } finally {
            if (btn) btn.disabled = false;
            if (spinner) spinner.classList.add('d-none');
        }
    }

    renderComparisonView(compData) {
        const q = compData.q_learning || {};
        const s = compData.sarsa || {};
        const g = compData.goal_based || {};

        // 1. Comparison Cards
        const elQRew = document.getElementById('comp-q-reward');
        const elQSucc = document.getElementById('comp-q-success');
        const elQSteps = document.getElementById('comp-q-steps');
        const elQConv = document.getElementById('comp-q-conv');

        if (elQRew) elQRew.innerText = (q.avg_reward > 0 ? '+' : '') + Number(q.avg_reward || 0).toFixed(1);
        if (elQSucc) elQSucc.innerText = `${(q.success_rate || 0).toFixed(1)}%`;
        if (elQSteps) elQSteps.innerText = `${q.avg_steps || 0} steps`;
        if (elQConv) elQConv.innerText = `Ep ${q.convergence_episode || '-'}`;

        const elSRew = document.getElementById('comp-s-reward');
        const elSSucc = document.getElementById('comp-s-success');
        const elSSteps = document.getElementById('comp-s-steps');
        const elSConv = document.getElementById('comp-s-conv');

        if (elSRew) elSRew.innerText = (s.avg_reward > 0 ? '+' : '') + Number(s.avg_reward || 0).toFixed(1);
        if (elSSucc) elSSucc.innerText = `${(s.success_rate || 0).toFixed(1)}%`;
        if (elSSteps) elSSteps.innerText = `${s.avg_steps || 0} steps`;
        if (elSConv) elSConv.innerText = `Ep ${s.convergence_episode || '-'}`;

        const elGRew = document.getElementById('comp-g-reward');
        const elGSucc = document.getElementById('comp-g-success');
        const elGSteps = document.getElementById('comp-g-steps');
        const elGConv = document.getElementById('comp-g-conv');

        if (elGRew) elGRew.innerText = (g.avg_reward > 0 ? '+' : '') + Number(g.avg_reward || 0).toFixed(1);
        if (elGSucc) elGSucc.innerText = `${(g.success_rate || 100).toFixed(1)}%`;
        if (elGSteps) elGSteps.innerText = `${g.avg_steps || 0} steps`;
        if (elGConv) elGConv.innerText = g.convergence_episode || 'Instant BFS';

        // 2. Comparison Table
        const tbody = document.getElementById('comparison-table-body');
        if (tbody && compData.comparison_table) {
            tbody.innerHTML = compData.comparison_table.map(row => `
                <tr>
                    <td class="fw-bold text-secondary">${row.metric}</td>
                    <td class="text-cyan fw-semibold">${row.q_learning}</td>
                    <td class="text-indigo fw-semibold">${row.sarsa}</td>
                    <td class="text-amber fw-semibold">${row.goal_based}</td>
                </tr>
            `).join('');
        }

        // 3. Render Charts
        updateComparisonCharts(compData);
    }

    // ==========================================
    // Page 5: Performance Analytics Controller
    // ==========================================
    setupAnalyticsControls() {
        const selectSession = document.getElementById('analytics-session-select');
        if (selectSession) {
            selectSession.addEventListener('change', (e) => {
                this.loadAnalyticsData(e.target.value);
            });
        }

        const btnExport = document.getElementById('btn-export-analytics');
        if (btnExport) {
            btnExport.addEventListener('click', () => this.exportAnalyticsData());
        }
    }

    async loadAnalyticsSessions() {
        try {
            const res = await fetch('/api/history');
            const data = await res.json();
            const selectEl = document.getElementById('analytics-session-select');
            if (selectEl && data.sessions) {
                const currentVal = selectEl.value;
                selectEl.innerHTML = '<option value="">Current / Latest Active Run</option>' +
                    data.sessions.map(s => `
                        <option value="${s.id}">#${s.id} - ${s.algorithm} (${s.episodes} ep, Rew: ${s.avg_reward ? s.avg_reward.toFixed(1) : '-'})</option>
                    `).join('');
                if (currentVal) selectEl.value = currentVal;
            }
        } catch (err) {
            console.error('[Analytics] Session list error:', err);
        }
    }

    async loadAnalyticsData(sessionId = null) {
        try {
            let url = '/api/analytics';
            if (sessionId) url += `?session_id=${sessionId}`;
            const res = await fetch(url);
            const data = await res.json();
            renderAnalyticsCharts(data);
        } catch (err) {
            console.error('[Analytics] Load error:', err);
        }
    }

    async exportAnalyticsData() {
        try {
            const res = await fetch('/api/analytics');
            const data = await res.json();
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `rl_agent_analytics_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (err) {
            alert('Failed to export analytics: ' + err.message);
        }
    }

    // ==========================================
    // Page 6: Learning Theory KaTeX Math Rendering
    // ==========================================
    renderMathFormulas() {
        if (window.renderMathInElement) {
            const theoryPage = document.getElementById('page-theory');
            if (theoryPage) {
                window.renderMathInElement(theoryPage, {
                    delimiters: [
                        { left: '$$', right: '$$', display: true },
                        { left: '$', right: '$', display: false },
                        { left: '\\(', right: '\\)', display: false },
                        { left: '\\[', right: '\\]', display: true }
                    ],
                    throwOnError: false
                });
            }
        }
    }
}
