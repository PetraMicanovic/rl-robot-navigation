"""
Experiment E1 — Effect of dynamic obstacle density on agent performance.

When train=True: trains a separate model for each obstacle count (original approach).
When train=False: evaluates the curriculum-trained model directly (curriculum approach).
"""
import os
from training.train import train
from evaluation.evaluate import evaluate

PHASE1_MODEL = "models/ppo_robot_nav_obs0_spd1.0_shaping"
CURRICULUM_MODEL = "models/ppo_robot_nav_curriculum_shaping"

def run_e1(train_models = True):
    print("Experiment E1 — varying number of dynamic obstacles")

    if train_models:
        # Phase 1 — train without obstacles
        if not os.path.exists(f"{PHASE1_MODEL}.zip"):
            train(n_dynamic_obstacles=0, obstacle_speed=1.0)

        # Phase 2 — train and evaluate for each obstacle count
        for n_obstacles in [3, 6, 10]:
            train(
                n_dynamic_obstacles=n_obstacles,
                obstacle_speed=1.0,
                pretrained_model_path=PHASE1_MODEL
            )
            evaluate(
                model_path=f"models/ppo_robot_nav_obs{n_obstacles}_spd1.0_shaping",
                n_dynamic_obstacles=n_obstacles,
                obstacle_speed=1.0,
                n_eval_episodes=200,
                experiment="e1"
            )
    else:
        # Evaluate curriculum model across obstacle counts — no training
        for n_obstacles in [3, 6, 10]:
            evaluate(
                model_path=CURRICULUM_MODEL,
                n_dynamic_obstacles=n_obstacles,
                obstacle_speed=1.0,
                n_eval_episodes=200,
                experiment="e1"
            )