"""
Experiment E2 — Effect of obstacle speed on agent performance.

When train=True: trains a separate model for each speed (original approach).
When train=False: evaluates the curriculum-trained model directly (curriculum approach).
"""
import os
from training.train import train
from evaluation.evaluate import evaluate

PHASE1_MODEL = "models/ppo_robot_nav_obs0_spd1.0_shaping"
CURRICULUM_MODEL = "models/ppo_robot_nav_curriculum_shaping"


def run_e2(train_models = True):
    print("Experiment E2 — varying obstacle speed")

    if train_models:
        # Phase 1 — reuse from E1 if available
        if not os.path.exists(f"{PHASE1_MODEL}.zip"):
            train(n_dynamic_obstacles=0, obstacle_speed=1.0)

        # Phase 2 — train and evaluate for each speed
        for speed in [0.5, 1.0, 1.5]:
            train(
                n_dynamic_obstacles=6,
                obstacle_speed=speed,
                pretrained_model_path=PHASE1_MODEL
            )
            evaluate(
                model_path=f"models/ppo_robot_nav_obs6_spd{speed}_shaping",
                n_dynamic_obstacles=6,
                obstacle_speed=speed,
                n_eval_episodes=200,
                experiment="e2"
            )
    else:
        # Evaluate curriculum model across speeds — no training
        for speed in [0.5, 1.0, 1.5]:
            evaluate(
                model_path=CURRICULUM_MODEL,
                n_dynamic_obstacles=6,
                obstacle_speed=speed,
                n_eval_episodes=200,
                experiment="e2"
            )
