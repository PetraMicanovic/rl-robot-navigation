from training.train import train
from evaluation.evaluate import evaluate

def main():
    train(n_dynamic_obstacles=0, obstacle_speed=1.0)
    evaluate(
        model_path="models/ppo_robot_nav_obs0_spd1.0",
        n_dynamic_obstacles=0,
        obstacle_speed=1.0
    )

if __name__ == "__main__":
    main()