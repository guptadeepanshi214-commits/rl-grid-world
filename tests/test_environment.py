"""
Unit tests for GridWorldEnv
"""

import unittest
from backend.environment import (
    GridWorldEnv,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT,
    REWARD_GOAL, REWARD_OBSTACLE, REWARD_STEP, REWARD_INVALID_MOVE
)


class TestGridWorldEnv(unittest.TestCase):

    def setUp(self):
        # 5x5 simple grid with start at (0,0), goal at (4,4), obstacle at (1,0)
        self.env = GridWorldEnv(
            rows=5, cols=5,
            start_pos=(0, 0),
            goal_pos=(4, 4),
            obstacles=[[1, 0], [2, 2]]
        )

    def test_initial_state(self):
        state = self.env.reset()
        self.assertEqual(state, (0, 0))
        self.assertEqual(self.env.steps_taken, 0)
        self.assertEqual(self.env.total_reward, 0.0)
        self.assertFalse(self.env.done)

    def test_valid_movement(self):
        self.env.reset()
        # Move RIGHT from (0,0) to (0,1)
        next_state, reward, done, info = self.env.step(ACTION_RIGHT)
        self.assertEqual(next_state, (0, 1))
        self.assertEqual(reward, REWARD_STEP)
        self.assertFalse(done)
        self.assertEqual(info["status"], "valid_step")

    def test_obstacle_collision(self):
        self.env.reset()
        # Move DOWN from (0,0) towards obstacle at (1,0)
        next_state, reward, done, info = self.env.step(ACTION_DOWN)
        # Agent should remain at (0,0)
        self.assertEqual(next_state, (0, 0))
        self.assertEqual(reward, REWARD_OBSTACLE)
        self.assertFalse(done)
        self.assertTrue(info["hit_obstacle"])

    def test_out_of_bounds_boundary(self):
        self.env.reset()
        # Move UP from (0,0) - hits upper boundary
        next_state, reward, done, info = self.env.step(ACTION_UP)
        self.assertEqual(next_state, (0, 0))
        self.assertEqual(reward, REWARD_INVALID_MOVE)
        self.assertFalse(done)
        self.assertTrue(info["hit_boundary"])

        # Move LEFT from (0,0) - hits left boundary
        next_state, reward, done, info = self.env.step(ACTION_LEFT)
        self.assertEqual(next_state, (0, 0))
        self.assertEqual(reward, REWARD_INVALID_MOVE)

    def test_goal_reached(self):
        # Place start directly adjacent to goal
        self.env.set_layout(rows=5, cols=5, start_pos=[4, 3], goal_pos=[4, 4], obstacles=[])
        self.env.reset()
        # Move RIGHT into goal
        next_state, reward, done, info = self.env.step(ACTION_RIGHT)
        self.assertEqual(next_state, (4, 4))
        self.assertEqual(reward, REWARD_GOAL)
        self.assertTrue(done)
        self.assertTrue(info["reached_goal"])

    def test_obstacle_toggle(self):
        self.env.reset()
        self.assertFalse(self.env.is_obstacle(0, 2))
        added = self.env.toggle_obstacle(0, 2)
        self.assertTrue(added)
        self.assertTrue(self.env.is_obstacle(0, 2))
        removed = self.env.toggle_obstacle(0, 2)
        self.assertFalse(removed)
        self.assertFalse(self.env.is_obstacle(0, 2))

        # Start and goal should not be toggled into obstacles
        self.assertFalse(self.env.toggle_obstacle(0, 0))
        self.assertFalse(self.env.toggle_obstacle(4, 4))


if __name__ == "__main__":
    unittest.main()
