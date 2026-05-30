"""
Training callbacks for periodic evaluation and model checkpointing.
"""
import os

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


def build_callbacks(config, eval_env, n_dynamic_obstacles, obstacle_speed):
    """
    Build training callbacks for periodic evaluation and model checkpointing.

    EvalCallback periodically evaluates the agent and saves the best-performing model. CheckpointCallback saves a snapshot every
    checkpoint_freq steps so training can be resumed if interrupted.

    Parameters
    config: dict
        Full configuration dictionary.
    eval_env: RobotNavEnv
        Environment used for evaluation.
    n_dynamic_obstacles: int
        Number of dynamic obstacles.
    obstacle_speed: float
        Speed of dynamic obstacles.

    Returns
    list
        List of callbacks to pass to model.learn().
    """
    model_dir = config["paths"]["model_dir"]
    log_dir = config["paths"]["log_dir"]
    n_eval_eps = config["evaluation"]["n_eval_episodes"]

    # Unique suffix for this experiment run
    experiment_suffix = f"obs{n_dynamic_obstacles}_spd{obstacle_speed}"

    best_model_dir = os.path.join(model_dir, f"best_{experiment_suffix}")
    checkpoint_dir = os.path.join(model_dir, f"checkpoints_{experiment_suffix}")
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_model_dir,
        log_path=os.path.join(log_dir, f"eval_{experiment_suffix}"),
        eval_freq=10_000,
        n_eval_episodes=n_eval_eps,
        deterministic=config["evaluation"]["deterministic"],
        render=False,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=checkpoint_dir,
        name_prefix="ppo_robot_nav",
        verbose=1
    )

    return [eval_callback, checkpoint_callback]