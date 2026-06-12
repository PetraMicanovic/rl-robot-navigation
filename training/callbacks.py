"""
Training callbacks for periodic evaluation and model checkpointing.
"""
import os

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


def build_callbacks(config, eval_env, n_dynamic_obstacles, obstacle_speed, best_model_save_path=None, log_suffix=None, n_eval_episodes=None,):
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
    best_model_save_path: str or None
        Directory where the best model is saved.
        When None (default) the path is derived from model_dir and the experiment suffix, which is the correct behaviour for single-run train(). 
        train_curriculum() passes a shared per-stage directory so retries all compete for the same best model slot.
    log_suffix: str or None
        Override for the eval log subdirectory name. 
        When None (default) the suffix is ``obs{n}_spd{speed}``. 
        train_curriculum() passes a per-attempt suffix (e.g. ``obs3_spd0.5_attempt1``) so each retry writes a fresh evaluations.npz rather than appending to the same file.
    n_eval_episodes: int or None
        Number of episodes to run during each evaluation. 
        When None (default) the value is taken from config["evaluation"]["n_eval_episodes"].
        Passing an explicit value lets train() and train_curriculum() use different counts without mutating the shared config dict. 

    Returns
    list
        List of callbacks to pass to model.learn().
    """
    model_dir = config["paths"]["model_dir"]
    log_dir = config["paths"]["log_dir"]
    if n_eval_episodes is not None:
        n_eval_eps = n_eval_episodes
    else:
        n_eval_eps = config["evaluation"]["n_eval_episodes"]

    # Unique suffix for this experiment run
    experiment_suffix = log_suffix or f"obs{n_dynamic_obstacles}_spd{obstacle_speed}"

    if best_model_save_path is None:
        best_model_save_path = os.path.join(
            model_dir, f"best_obs{n_dynamic_obstacles}_spd{obstacle_speed}"
        )    
    checkpoint_dir = os.path.join(model_dir, f"checkpoints_{experiment_suffix}")
    os.makedirs(best_model_save_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_model_save_path,
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