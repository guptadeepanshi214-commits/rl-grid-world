"""
Unit tests for Q-Learning, SARSA, and Goal-Based Agents
"""

import unittest
import numpy as np
from backend.environment import GridWorldEnv
from backend.q_learning import QLearningAgent
from backend.sarsa import SarsaAgent
from backend.goal_agent import GoalBasedAgent
from backend.evaluation import train_agent_sync, evaluate_agent


class TestAlgorithms(unittest.TestCase):

    def setUp(self):
        # 4x4 Grid for fast unit testing
        self.env = GridWorldEnv(
            rows=4, cols=4,
            start_pos=(0, 0),
            goal_pos=(3, 3),
            obstacles=[[1, 1], [1, 2]]
        )

    def test_q_learning_initialization_and_update(self):
        agent = QLearningAgent(num_states=16, num_actions=4, learning_rate=0.1, discount_factor=0.9, epsilon=1.0)
        self.assertEqual(agent.q_table.shape, (16, 4))
        self.assertTrue(np.all(agent.q_table == 0))

        # Test single Bellman update
        # s=0, a=3 (RIGHT), r=-1, next_s=1, done=False, max Q(s')=0
        # target = -1 + 0.9 * 0 = -1
        # Q(0, 3) <- 0 + 0.1 * (-1 - 0) = -0.1
        td_err = agent.update(state_idx=0, action=3, reward=-1.0, next_state_idx=1, done=False)
        self.assertAlmostEqual(agent.q_table[0, 3], -0.1)
        self.assertAlmostEqual(td_err, -1.0)

        # Update terminal step: s=14, a=1, r=100, done=True
        # target = 100
        # Q(14, 1) <- 0 + 0.1 * (100 - 0) = 10.0
        agent.update(state_idx=14, action=1, reward=100.0, next_state_idx=15, done=True)
        self.assertAlmostEqual(agent.q_table[14, 1], 10.0)

    def test_sarsa_initialization_and_update(self):
        agent = SarsaAgent(num_states=16, num_actions=4, learning_rate=0.1, discount_factor=0.9, epsilon=1.0)
        self.assertEqual(agent.q_table.shape, (16, 4))

        # Test SARSA on-policy update
        # Set next Q(1, 2) = 5.0
        agent.q_table[1, 2] = 5.0
        # s=0, a=3, r=-1, next_s=1, next_a=2, done=False
        # target = -1 + 0.9 * 5.0 = 3.5
        # Q(0, 3) <- 0 + 0.1 * (3.5 - 0) = 0.35
        td_err = agent.update(state_idx=0, action=3, reward=-1.0, next_state_idx=1, next_action=2, done=False)
        self.assertAlmostEqual(agent.q_table[0, 3], 0.35)

    def test_epsilon_decay(self):
        agent = QLearningAgent(num_states=16, epsilon=1.0, epsilon_decay=0.9, min_epsilon=0.1)
        eps1 = agent.decay_epsilon()
        self.assertAlmostEqual(eps1, 0.9)
        for _ in range(50):
            agent.decay_epsilon()
        self.assertAlmostEqual(agent.epsilon, 0.1)

    def test_goal_based_agent_bfs(self):
        agent = GoalBasedAgent()
        res = agent.solve_path(self.env)
        self.assertTrue(res["success"])
        self.assertGreater(res["steps"], 0)
        # Check path starts at start and ends at goal
        self.assertEqual(res["path"][0], [0, 0])
        self.assertEqual(res["path"][-1], [3, 3])
        # Ensure path does not intersect obstacles
        for coord in res["path"]:
            self.assertNotIn(coord, [[1, 1], [1, 2]])

    def test_q_learning_training_convergence(self):
        # Small open grid for guaranteed fast convergence within 150 episodes
        env = GridWorldEnv(rows=3, cols=3, start_pos=(0, 0), goal_pos=(2, 2), obstacles=[])
        train_res = train_agent_sync(
            QLearningAgent, env,
            episodes=150, max_steps=50,
            alpha=0.2, gamma=0.95,
            epsilon=1.0, epsilon_decay=0.97, min_epsilon=0.01
        )
        self.assertEqual(train_res["eval"]["success_rate"], 100.0)
        self.assertGreater(train_res["eval"]["avg_reward"], 80.0)

    def test_sarsa_training_convergence(self):
        env = GridWorldEnv(rows=3, cols=3, start_pos=(0, 0), goal_pos=(2, 2), obstacles=[])
        train_res = train_agent_sync(
            SarsaAgent, env,
            episodes=150, max_steps=50,
            alpha=0.2, gamma=0.95,
            epsilon=1.0, epsilon_decay=0.97, min_epsilon=0.01
        )
        self.assertEqual(train_res["eval"]["success_rate"], 100.0)
        self.assertGreater(train_res["eval"]["avg_reward"], 80.0)


if __name__ == "__main__":
    unittest.main()
