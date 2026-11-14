"""
Gunicorn Production Server Configuration for RL Grid World
"""

import os

# Render dynamically passes the PORT environment variable
port = os.environ.get("PORT", "5000")
bind = f"0.0.0.0:{port}"

# Single worker with 4 threads is essential to preserve the in-memory
# TrainingManager background training thread and global_env across API requests
workers = 1
threads = 4
worker_class = "gthread"
timeout = 120
keepalive = 5

# Logging to standard output for Render dashboard logs
accesslog = "-"
errorlog = "-"
loglevel = "info"
