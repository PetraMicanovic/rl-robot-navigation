"""
Module for creating and configurating the PPO agent used in RobotNavEnv training.
"""
from stable_baselines3 import PPO


def build_ppo_agent(config, training_env, log_dir):
    """
    Initialize the PPO agent with parameters from config.json.

    Uses MlpPolicy with a two-layer MLP as defined in config.
    TensorBoard logging is enabled automatically.

    Parameters
    config: dict
        Full configuration dictionary.
    training_env: VecMonitor
        The vectorized training environment.
    log_dir: str
        Directory where TensorBoard logs will be saved.

    Returns
    PPO
        Initialized PPO agent ready for training.
    """
    training_config = config["training"]

    agent = PPO(
        policy=training_config["policy"],
        env=training_env,
        seed=config["seed"],
        learning_rate=training_config["learning_rate"],
        n_steps=training_config["n_steps"],
        batch_size=training_config["batch_size"],
        n_epochs=training_config["n_epochs"],
        gamma=training_config["gamma"],
        ent_coef=training_config["ent_coef"],
        policy_kwargs={"net_arch": training_config["policy_kwargs"]["net_arch"]},
        tensorboard_log=log_dir,
        verbose=1
    )

    return agent