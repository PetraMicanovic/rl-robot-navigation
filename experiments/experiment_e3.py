"""
Experiment E3 — Generalization to unseen obstacle configurations.
Evaluates a model trained on N=6, v=1.0 on unseen configurations (N=8, v=1.2 and other combinations not seen during training).
"""
from evaluation.evaluate import evaluate

TRAINED_MODEL = "models/ppo_robot_nav_obs6_spd1.0"

UNSEEN_CONFIGS = [
    {"n_dynamic_obstacles": 8, "obstacle_speed": 1.2},
    {"n_dynamic_obstacles": 5, "obstacle_speed": 0.8},
    {"n_dynamic_obstacles": 10, "obstacle_speed": 1.5},
]

def run_e3():
    print("Experiment E3 — generalization to unseen configurations")

    for config in UNSEEN_CONFIGS:
        evaluate(
            model_path=TRAINED_MODEL,
            n_dynamic_obstacles=config["n_dynamic_obstacles"],
            obstacle_speed=config["obstacle_speed"],
            n_eval_episodes=200,
            experiment="e3"

        )