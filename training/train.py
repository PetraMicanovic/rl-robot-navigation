"""
Training module for the PPO agent in RobotNavEnv.
Loads configuration from config.json, creates training and evaluation environments, initializes the PPO agent, runs training and saves trained model.
"""
import os
import json

from robot_env.env_factory import create_parallel_envs, create_eval_env
from training.agent import build_ppo_agent
from training.callbacks import build_callbacks
from stable_baselines3 import PPO

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")


def load_config(config_path):
    """
    Load configuration from a JSON file.

    Parameters
    config_path: str
        Path to the JSON configuration file.

    Returns
    dict
        Full configuration dictionary.
    """
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config

def train(n_dynamic_obstacles=None, obstacle_speed=None, pretrained_model_path=None, use_reward_shaping = True):
    """
    Main training pipeline.

    Loads config, creates environments and agent, runs training,
    and saves the final model.

    Parameters
    n_dynamic_obstacles: int or None
        Number of dynamic obstacles.
    obstacle_speed: float or None
        Speed of dynamic obstacles.
    pretrained_model_path: str
        Path to a pretrained PPO model. If provided, training continues from the loaded model.
    use_reward_shaping: bool
        If True, adds progress reward toward the target.
        If False, only goal, collision and step penalty rewards are used.
    """
    config = load_config(CONFIG_PATH)

    # Apply overrides or use config defaults
    if n_dynamic_obstacles is None:
        n_dynamic_obstacles = config["environment"]["n_dynamic_obstacles"]
    if obstacle_speed is None:
        obstacle_speed = config["environment"]["obstacle_speed"]

    model_dir = config["paths"]["model_dir"]
    log_dir = config["paths"]["log_dir"]
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("PPO Training — RobotNavEnv")
    print(f"Dynamic obstacles: {n_dynamic_obstacles}")
    print(f"Obstacle speed: {obstacle_speed}")
    print(f"Total timesteps: {config['training']['total_timesteps']:,}")
    print(f"Parallel envs: {config['training']['n_envs']}")
    print(f"Model output: {model_dir}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"Reward shaping: {use_reward_shaping}")
    print()

    # Create training and evaluation environments
    training_env = create_parallel_envs(config, CONFIG_PATH, n_dynamic_obstacles, obstacle_speed, use_reward_shaping)
    eval_env = create_eval_env(CONFIG_PATH, n_dynamic_obstacles, obstacle_speed, use_reward_shaping)

    # Create a new PPO agent or load a pretrained one
    if pretrained_model_path is None:
        agent = build_ppo_agent(config, training_env, log_dir)
    else:
        if not os.path.exists(pretrained_model_path):
            raise FileNotFoundError(f"Pretrained model not found: {pretrained_model_path}")
        print(f"Loading pretrained model from: {pretrained_model_path}")
        agent = PPO.load(pretrained_model_path, env=training_env, tensorboard_log=log_dir)

    callbacks = build_callbacks(config, eval_env, n_dynamic_obstacles, obstacle_speed)

    # Run training
    print("Starting training...")
    agent.learn(
        total_timesteps=config["training"]["total_timesteps"],
        callback=callbacks,
        progress_bar=True
    )

    # Save final model
    shaping_suffix = "shaping" if use_reward_shaping else "no_shaping"
    experiment_suffix = f"obs{n_dynamic_obstacles}_spd{obstacle_speed}_{shaping_suffix}"
    final_model_path = os.path.join(model_dir, f"ppo_robot_nav_{experiment_suffix}")
    agent.save(final_model_path)
    meta = {"tb_log_folder": agent.logger.dir}  # SB3 čuva putanju loga ovdje
    meta_path = final_model_path + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print()
    print(f"Training complete. Model saved to: {final_model_path}.zip")

    training_env.close()
    eval_env.close()
