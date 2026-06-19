"""
Experiment E4 — Effect of reward shaping on agent performance.
Compares a model trained with reward shaping against one trained without.

When train=True: trains both variants from scratch (original approach).
When train=False: uses the curriculum-trained shaping model; trains only the no_shaping variant if it doesn't already exist.
"""
import os
from training.train import train, train_curriculum
from evaluation.evaluate import evaluate

PHASE1_MODEL = "models/ppo_robot_nav_obs0_spd1.0_shaping"
TRAINED_MODEL_SHAPING = "models/ppo_robot_nav_obs6_spd1.0_shaping"
CURRICULUM_MODEL_SHAPING = "models/ppo_robot_nav_curriculum_shaping"
CURRICULUM_MODEL_NO_SHAPING = "models/ppo_robot_nav_curriculum_no_shaping"


def run_e4(train_models = True):
    print("Experiment E4 — reward shaping vs no reward shaping")

    if train_models:
        # Original approach — train both variants separately
        if not os.path.exists(f"{PHASE1_MODEL}.zip"):
            train(n_dynamic_obstacles=0, obstacle_speed=1.0)

        train(
            n_dynamic_obstacles=6,
            obstacle_speed=1.0,
            pretrained_model_path=PHASE1_MODEL,
            use_reward_shaping=True
        )
        evaluate(
            model_path=TRAINED_MODEL_SHAPING,
            n_dynamic_obstacles=6,
            obstacle_speed=1.0,
            n_eval_episodes=200,
            experiment="e4",
            label="shaping"
        )

        train(
            n_dynamic_obstacles=6,
            obstacle_speed=1.0,
            pretrained_model_path=PHASE1_MODEL,
            use_reward_shaping=False
        )
        evaluate(
            model_path="models/ppo_robot_nav_obs6_spd1.0_no_shaping",
            n_dynamic_obstacles=6,
            obstacle_speed=1.0,
            n_eval_episodes=200,
            experiment="e4",
            label="no_shaping"
        )
    else:
        # Curriculum approach — shaping model already trained in main
        evaluate(
            model_path=CURRICULUM_MODEL_SHAPING,
            n_dynamic_obstacles=6,
            obstacle_speed=1.0,
            n_eval_episodes=200,
            experiment="e4",
            label="shaping"
        )

        # Train no_shaping curriculum variant only if not already available
        if not os.path.exists(f"{CURRICULUM_MODEL_NO_SHAPING}.zip"):
            train_curriculum(use_reward_shaping=False, log_prefix="no_shaping")
        evaluate(
            model_path=CURRICULUM_MODEL_NO_SHAPING,
            n_dynamic_obstacles=6,
            obstacle_speed=1.0,
            n_eval_episodes=200,
            experiment="e4",
            label="no_shaping"
        )

