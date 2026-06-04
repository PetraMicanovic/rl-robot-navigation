"""
Experiment E4 — Effect of reward shaping on agent performance.
Compares two agents: one trained with reward shaping (progress reward) and one without. All other parameters are identical.
"""
from training.train import train
from evaluation.evaluate import evaluate

PHASE1_MODEL = "models/ppo_robot_nav_obs0_spd1.0"


def run_e4():
    print("Experiment E4 — reward shaping vs no reward shaping")

    # With reward shaping (default — defined in config.json)
    train(
        n_dynamic_obstacles=6,
        obstacle_speed=1.0,
        pretrained_model_path=PHASE1_MODEL,
        use_reward_shaping=True
    )
    evaluate(
        model_path="models/ppo_robot_nav_obs6_spd1.0_shaping",
        n_dynamic_obstacles=6,
        obstacle_speed=1.0,
        n_eval_episodes=200,
        experiment="e4"
    )

    # Without reward shaping — requires separate config or env flag
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
        experiment="e4"
    )
