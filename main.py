from training.train import train
from evaluation.evaluate import evaluate

PHASE1_MODEL = "models/ppo_robot_nav_obs0_spd1.0"

def run_e1():
    print("Experiment E1 — varying number of dynamic obstacles")

    # Phase 1 — training without obstacles
    train(n_dynamic_obstacles=0, obstacle_speed=1.0)
    evaluate(
        model_path=PHASE1_MODEL,
        n_dynamic_obstacles=0,
        obstacle_speed=1.0,
        n_eval_episodes=200
    )

    # Phase 2 — training with dynamic obstacles
    for n_obstacles in [3, 6, 10]:
        train(
            n_dynamic_obstacles=n_obstacles,
            obstacle_speed=1.0,
            pretrained_model_path=PHASE1_MODEL
        )
        evaluate(
            model_path=f"models/ppo_robot_nav_obs{n_obstacles}_spd1.0",
            n_dynamic_obstacles=n_obstacles,
            obstacle_speed=1.0,
            n_eval_episodes=200
        )

if __name__ == "__main__":
    run_e1()
