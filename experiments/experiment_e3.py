"""
Experiment E3 — Generalization to unseen obstacle configurations.
Evaluates a trained model on configurations not seen during training.

When train=False (default): uses the curriculum-trained model.
When train=True: uses the model trained on N=6, speed=1.0 from E1/E2.
"""
from evaluation.evaluate import evaluate

TRAINED_MODEL = "models/ppo_robot_nav_obs6_spd1.0_shaping"
CURRICULUM_MODEL = "models/ppo_robot_nav_curriculum_shaping"

UNSEEN_CONFIGS = [
    {"n_dynamic_obstacles": 8, "obstacle_speed": 1.2},
    {"n_dynamic_obstacles": 5, "obstacle_speed": 0.8},
    {"n_dynamic_obstacles": 10, "obstacle_speed": 1.5},
    {"n_dynamic_obstacles": 9, "obstacle_speed": 1.3},
    {"n_dynamic_obstacles": 7, "obstacle_speed": 1},
    {"n_dynamic_obstacles": 12, "obstacle_speed": 2},
    {"n_dynamic_obstacles": 15, "obstacle_speed": 0.3},
]

def run_e3(train_models = True):
    print("Experiment E3 — generalization to unseen configurations")

    if train_models:
        model = TRAINED_MODEL  
    else:
        model = CURRICULUM_MODEL

    for config in UNSEEN_CONFIGS:
        evaluate(
            model_path=model,
            n_dynamic_obstacles=config["n_dynamic_obstacles"],
            obstacle_speed=config["obstacle_speed"],
            n_eval_episodes=200,
            experiment="e3"
        )
