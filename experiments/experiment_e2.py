"""
Experiment E2 — Effect of obstacle speed on agent performance.
Trains a PPO agent with varying obstacle speeds (v = 0.5, 1.0, 1.5) at fixed number of dynamic obstacles (N = 6).
"""
import os

from training.train import train
from evaluation.evaluate import evaluate

PHASE1_MODEL = "models/ppo_robot_nav_obs0_spd1.0"


def run_e2():
    print("Experiment E2 — varying obstacle speed")

    # Phase 1 — reuse model from E1 if it exists, otherwise train from scratch
    if not os.path.exists(f"{PHASE1_MODEL}.zip"):
        train(n_dynamic_obstacles=0, obstacle_speed=1.0)

    # Phase 2 — training with N=6 obstacles at varying speeds
    for speed in [0.5, 1.0, 1.5]:
        train(
            n_dynamic_obstacles=6,
            obstacle_speed=speed,
            pretrained_model_path=PHASE1_MODEL
        )
        evaluate(
            model_path=f"models/ppo_robot_nav_obs6_spd{speed}",
            n_dynamic_obstacles=6,
            obstacle_speed=speed,
            n_eval_episodes=200
        )