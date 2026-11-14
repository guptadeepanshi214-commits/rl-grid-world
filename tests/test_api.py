"""
Unit tests for Flask REST API Endpoints
"""

import unittest
import json
from app import app, global_env, training_mgr


class TestAPIRoutes(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")

    def test_dashboard_endpoint(self):
        res = self.client.get("/api/dashboard")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("title", data)
        self.assertIn("current_algorithm", data)
        self.assertIn("environment", data)

    def test_environment_get_and_reset(self):
        res = self.client.get("/api/environment")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["rows"], 10)
        self.assertEqual(data["cols"], 10)

        # Reset with preset
        res = self.client.post("/api/environment/reset", json={"preset": "empty", "rows": 6, "cols": 6})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["environment"]["rows"], 6)
        self.assertEqual(data["environment"]["obstacles"], [])

        # Reset back to default
        self.client.post("/api/environment/reset", json={"preset": "default", "rows": 10, "cols": 10})

    def test_environment_update_and_toggle(self):
        res = self.client.post("/api/environment/update", json={"toggle_obstacle": [2, 3]})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["toggled"], [2, 3])

    def test_simulation_endpoint(self):
        res = self.client.post("/api/simulate", json={"algorithm": "goal_based"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("path", data)
        self.assertIn("actions", data)

    def test_comparison_endpoint(self):
        res = self.client.get("/api/comparison")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("q_learning", data["comparison"])
        self.assertIn("sarsa", data["comparison"])
        self.assertIn("goal_based", data["comparison"])

    def test_train_and_status_cycle(self):
        # Quick train goal-based agent
        res = self.client.post("/api/train", json={"algorithm": "goal_based"})
        self.assertEqual(res.status_code, 200)

        status_res = self.client.get("/api/train/status")
        self.assertEqual(status_res.status_code, 200)
        status_data = status_res.get_json()
        self.assertIn(status_data["status"], ["training", "completed"])

        # Reset train
        self.client.post("/api/train/reset")


if __name__ == "__main__":
    unittest.main()
