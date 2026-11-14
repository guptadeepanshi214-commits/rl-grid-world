/**
 * Chart.js Manager for Reinforcement Learning Analytics & Benchmark Comparisons
 * Configured with AI Dark Navy & Neon Cyan Palette
 */

const ChartTheme = {
    textColor: '#94a3b8',
    gridColor: 'rgba(255, 255, 255, 0.06)',
    fontFamily: "'Inter', sans-serif",
    cyan: '#00f2fe',
    blue: '#38bdf8',
    indigo: '#6366f1',
    emerald: '#10b981',
    amber: '#f59e0b',
    rose: '#f43f5e',
    cardBg: '#111d35'
};

// Global chart instances
const AppCharts = {
    trainMini: null,
    // Comparison charts
    compReward: null,
    compSteps: null,
    compSuccess: null,
    compConvergence: null,
    // Analytics charts
    analyticsReward: null,
    analyticsAvgReward: null,
    analyticsSuccess: null,
    analyticsEpsilon: null,
    analyticsSteps: null,
    analyticsHeadToHead: null
};

// Common Chart.js Defaults
function getStandardOptions(titleText = '', yLabel = '', xLabel = 'Episode') {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: ChartTheme.textColor,
                    font: { family: ChartTheme.fontFamily, size: 11, weight: '500' },
                    usePointStyle: true,
                    boxWidth: 8
                }
            },
            tooltip: {
                backgroundColor: '#0f172a',
                titleColor: '#f8fafc',
                bodyColor: '#94a3b8',
                borderColor: 'rgba(0, 242, 254, 0.3)',
                borderWidth: 1,
                padding: 10,
                cornerRadius: 8
            }
        },
        scales: {
            x: {
                grid: { color: ChartTheme.gridColor },
                ticks: { color: ChartTheme.textColor, font: { family: ChartTheme.fontFamily, size: 10 } },
                title: { display: !!xLabel, text: xLabel, color: ChartTheme.textColor, font: { size: 11 } }
            },
            y: {
                grid: { color: ChartTheme.gridColor },
                ticks: { color: ChartTheme.textColor, font: { family: ChartTheme.fontFamily, size: 10 } },
                title: { display: !!yLabel, text: yLabel, color: ChartTheme.textColor, font: { size: 11 } }
            }
        }
    };
}

/**
 * Initialize Mini Chart on Training Panel
 */
function initTrainMiniChart() {
    const ctx = document.getElementById('chart-train-mini');
    if (!ctx) return;

    if (AppCharts.trainMini) AppCharts.trainMini.destroy();

    AppCharts.trainMini = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Episode Reward',
                    data: [],
                    borderColor: 'rgba(0, 242, 254, 0.4)',
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Avg Reward (50 ep)',
                    data: [],
                    borderColor: ChartTheme.cyan,
                    backgroundColor: 'rgba(0, 242, 254, 0.08)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.3
                }
            ]
        },
        options: getStandardOptions('Live Training Rewards', 'Reward', 'Episode')
    });
}

function updateTrainMiniChart(records) {
    if (!AppCharts.trainMini || !records || records.length === 0) return;

    const labels = records.map(r => r.episode);
    const rewards = records.map(r => r.reward);

    // Compute rolling window average
    const avgRewards = [];
    const window = [];
    records.forEach(r => {
        window.push(r.reward);
        if (window.length > 30) window.shift();
        avgRewards.push(Number((window.reduce((a, b) => a + b, 0) / window.length).toFixed(2)));
    });

    AppCharts.trainMini.data.labels = labels;
    AppCharts.trainMini.data.datasets[0].data = rewards;
    AppCharts.trainMini.data.datasets[1].data = avgRewards;
    AppCharts.trainMini.update('none'); // Update without full redraw animation for speed
}

/**
 * Initialize and update Algorithm Comparison Charts
 */
function updateComparisonCharts(compData) {
    if (!compData) return;

    const q = compData.q_learning || {};
    const s = compData.sarsa || {};
    const g = compData.goal_based || {};

    const algos = ['Q-Learning', 'SARSA', 'Goal-Based (BFS)'];
    const barColors = [ChartTheme.cyan, ChartTheme.indigo, ChartTheme.amber];

    // 1. Average Reward Bar Chart
    const ctxReward = document.getElementById('chart-comp-reward');
    if (ctxReward) {
        if (AppCharts.compReward) AppCharts.compReward.destroy();
        AppCharts.compReward = new Chart(ctxReward, {
            type: 'bar',
            data: {
                labels: algos,
                datasets: [{
                    label: 'Average Evaluation Reward',
                    data: [q.avg_reward || 0, s.avg_reward || 0, g.avg_reward || 0],
                    backgroundColor: barColors,
                    borderRadius: 8,
                    borderWidth: 0
                }]
            },
            options: {
                ...getStandardOptions('Reward Comparison', 'Average Reward', ''),
                plugins: { legend: { display: false } }
            }
        });
    }

    // 2. Steps to Goal Bar Chart
    const ctxSteps = document.getElementById('chart-comp-steps');
    if (ctxSteps) {
        if (AppCharts.compSteps) AppCharts.compSteps.destroy();
        AppCharts.compSteps = new Chart(ctxSteps, {
            type: 'bar',
            data: {
                labels: algos,
                datasets: [{
                    label: 'Average Steps to Goal',
                    data: [q.avg_steps || 0, s.avg_steps || 0, g.avg_steps || 0],
                    backgroundColor: [ChartTheme.blue, ChartTheme.indigo, ChartTheme.amber],
                    borderRadius: 8
                }]
            },
            options: {
                ...getStandardOptions('Path Length Comparison', 'Steps (lower is faster)', ''),
                plugins: { legend: { display: false } }
            }
        });
    }

    // 3. Success Rate Bar Chart
    const ctxSuccess = document.getElementById('chart-comp-success');
    if (ctxSuccess) {
        if (AppCharts.compSuccess) AppCharts.compSuccess.destroy();
        AppCharts.compSuccess = new Chart(ctxSuccess, {
            type: 'bar',
            data: {
                labels: algos,
                datasets: [{
                    label: 'Success Rate (%)',
                    data: [q.success_rate || 0, s.success_rate || 0, g.success_rate || 100],
                    backgroundColor: [ChartTheme.emerald, ChartTheme.blue, ChartTheme.amber],
                    borderRadius: 8
                }]
            },
            options: {
                ...getStandardOptions('Success Rate', 'Success Rate %', ''),
                plugins: { legend: { display: false } },
                scales: {
                    ...getStandardOptions().scales,
                    y: { min: 0, max: 100, ticks: { callback: v => `${v}%`, color: ChartTheme.textColor } }
                }
            }
        });
    }

    // 4. Convergence Learning Curves (Line Chart)
    const ctxConv = document.getElementById('chart-comp-convergence');
    if (ctxConv && compData.chart_data) {
        if (AppCharts.compConvergence) AppCharts.compConvergence.destroy();
        const cd = compData.chart_data;

        AppCharts.compConvergence = new Chart(ctxConv, {
            type: 'line',
            data: {
                labels: cd.episodes || [],
                datasets: [
                    {
                        label: 'Q-Learning (Off-Policy)',
                        data: cd.q_learning_rewards || [],
                        borderColor: ChartTheme.cyan,
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 0
                    },
                    {
                        label: 'SARSA (On-Policy)',
                        data: cd.sarsa_rewards || [],
                        borderColor: ChartTheme.indigo,
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 0
                    },
                    {
                        label: 'Goal-Based BFS Baseline',
                        data: cd.goal_baseline_reward || [],
                        borderColor: ChartTheme.amber,
                        borderDash: [5, 5],
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        pointRadius: 0
                    }
                ]
            },
            options: getStandardOptions('Learning Convergence Over Training', 'Rolling Average Reward', 'Episodes')
        });
    }
}

/**
 * Initialize & Render the 6 Dedicated Performance Analytics Charts
 */
function renderAnalyticsCharts(analyticsData) {
    if (!analyticsData) return;

    const episodes = analyticsData.episodes || [];
    const rewards = analyticsData.rewards || [];
    const avgRewards = analyticsData.average_rewards || [];
    const steps = analyticsData.steps || [];
    const epsilons = analyticsData.epsilons || [];

    // 1. Reward vs Episodes Graph
    const ctx1 = document.getElementById('chart-analytics-reward');
    if (ctx1) {
        if (AppCharts.analyticsReward) AppCharts.analyticsReward.destroy();
        AppCharts.analyticsReward = new Chart(ctx1, {
            type: 'line',
            data: {
                labels: episodes,
                datasets: [{
                    label: 'Episode Reward',
                    data: rewards,
                    borderColor: 'rgba(0, 242, 254, 0.65)',
                    backgroundColor: 'rgba(0, 242, 254, 0.05)',
                    fill: true,
                    borderWidth: 1.5,
                    pointRadius: episodes.length > 100 ? 0 : 2,
                    tension: 0.2
                }]
            },
            options: getStandardOptions('Raw Reward vs Episodes', 'Reward', 'Episode')
        });
    }

    // 2. Average Reward Graph
    const ctx2 = document.getElementById('chart-analytics-avg-reward');
    if (ctx2) {
        if (AppCharts.analyticsAvgReward) AppCharts.analyticsAvgReward.destroy();
        AppCharts.analyticsAvgReward = new Chart(ctx2, {
            type: 'line',
            data: {
                labels: episodes,
                datasets: [{
                    label: 'Moving Average Reward',
                    data: avgRewards,
                    borderColor: ChartTheme.emerald,
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    borderWidth: 2.5,
                    pointRadius: 0,
                    tension: 0.3
                }]
            },
            options: getStandardOptions('Moving Average Reward Convergence', 'Avg Reward', 'Episode')
        });
    }

    // 3. Success Rate Graph
    const ctx3 = document.getElementById('chart-analytics-success');
    if (ctx3) {
        if (AppCharts.analyticsSuccess) AppCharts.analyticsSuccess.destroy();

        // Calculate rolling success rate percentage
        const succRates = [];
        let win = [];
        rewards.forEach(r => {
            win.push(r > 0 ? 1 : 0);
            if (win.length > 25) win.shift();
            succRates.push(Number(((win.reduce((a, b) => a + b, 0) / win.length) * 100).toFixed(1)));
        });

        AppCharts.analyticsSuccess = new Chart(ctx3, {
            type: 'line',
            data: {
                labels: episodes,
                datasets: [{
                    label: 'Rolling Success Rate (%)',
                    data: succRates,
                    borderColor: ChartTheme.blue,
                    backgroundColor: 'rgba(56, 189, 248, 0.1)',
                    fill: true,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3
                }]
            },
            options: {
                ...getStandardOptions('Success Rate (%) Over Episodes', 'Success Rate %', 'Episode'),
                scales: {
                    ...getStandardOptions().scales,
                    y: { min: 0, max: 100, ticks: { callback: v => `${v}%`, color: ChartTheme.textColor } }
                }
            }
        });
    }

    // 4. Exploration Rate (Epsilon) Decay Graph
    const ctx4 = document.getElementById('chart-analytics-epsilon');
    if (ctx4) {
        if (AppCharts.analyticsEpsilon) AppCharts.analyticsEpsilon.destroy();
        AppCharts.analyticsEpsilon = new Chart(ctx4, {
            type: 'line',
            data: {
                labels: episodes,
                datasets: [{
                    label: 'Exploration Rate (ε)',
                    data: epsilons,
                    borderColor: ChartTheme.amber,
                    backgroundColor: 'rgba(245, 158, 11, 0.08)',
                    fill: true,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.2
                }]
            },
            options: getStandardOptions('Exploration Decay (ε-Greedy Schedule)', 'Epsilon (ε)', 'Episode')
        });
    }

    // 5. Steps per Episode Graph
    const ctx5 = document.getElementById('chart-analytics-steps');
    if (ctx5) {
        if (AppCharts.analyticsSteps) AppCharts.analyticsSteps.destroy();
        AppCharts.analyticsSteps = new Chart(ctx5, {
            type: 'line',
            data: {
                labels: episodes,
                datasets: [{
                    label: 'Steps Taken per Episode',
                    data: steps,
                    borderColor: ChartTheme.rose,
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    pointRadius: episodes.length > 100 ? 0 : 2,
                    tension: 0.1
                }]
            },
            options: getStandardOptions('Steps to Complete / Terminate Episode', 'Steps (Path Efficiency)', 'Episode')
        });
    }

    // 6. Q-Learning vs SARSA Performance Graph
    const ctx6 = document.getElementById('chart-analytics-head-to-head');
    if (ctx6) {
        if (AppCharts.analyticsHeadToHead) AppCharts.analyticsHeadToHead.destroy();

        if (analyticsData.q_learning_rewards && analyticsData.sarsa_rewards) {
            AppCharts.analyticsHeadToHead = new Chart(ctx6, {
                type: 'line',
                data: {
                    labels: analyticsData.episodes,
                    datasets: [
                        {
                            label: 'Q-Learning Reward',
                            data: analyticsData.q_learning_rewards,
                            borderColor: ChartTheme.cyan,
                            borderWidth: 2,
                            pointRadius: 0,
                            tension: 0.3
                        },
                        {
                            label: 'SARSA Reward',
                            data: analyticsData.sarsa_rewards,
                            borderColor: ChartTheme.indigo,
                            borderWidth: 2,
                            pointRadius: 0,
                            tension: 0.3
                        }
                    ]
                },
                options: getStandardOptions('Q-Learning vs SARSA Comparative Performance', 'Rolling Reward', 'Episode')
            });
        } else {
            // Default dual dataset using session rewards
            AppCharts.analyticsHeadToHead = new Chart(ctx6, {
                type: 'line',
                data: {
                    labels: episodes,
                    datasets: [
                        {
                            label: 'Raw Reward',
                            data: rewards,
                            borderColor: 'rgba(0, 242, 254, 0.4)',
                            borderWidth: 1,
                            pointRadius: 0
                        },
                        {
                            label: 'Smoothed Policy Value',
                            data: avgRewards,
                            borderColor: ChartTheme.cyan,
                            borderWidth: 2.5,
                            pointRadius: 0
                        }
                    ]
                },
                options: getStandardOptions('Performance Curve & Smoothed Trend', 'Reward Value', 'Episode')
            });
        }
    }
}

// Global export
window.AppCharts = AppCharts;
window.initTrainMiniChart = initTrainMiniChart;
window.updateTrainMiniChart = updateTrainMiniChart;
window.updateComparisonCharts = updateComparisonCharts;
window.renderAnalyticsCharts = renderAnalyticsCharts;
