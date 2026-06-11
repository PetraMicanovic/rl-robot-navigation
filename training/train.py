"""
Training module for the PPO agent in RobotNavEnv.

Provides two entry points:
  - train(): single-run training for a fixed environment config.
  - train_curriculum(): multi-stage curriculum that progressively increases difficulty.

Both functions load hyperparameters from config.json, build environments,
initialize or resume a PPO agent, run learning, and save the trained model.
"""
import os
import json
import shutil
import numpy as np

from robot_env.env_factory import create_parallel_envs, create_eval_env
from training.agent import build_ppo_agent
from training.callbacks import build_callbacks
from stable_baselines3 import PPO

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

# Default curriculum stages for v3 training.
# Each stage defines obstacle count, speed, timesteps and the success rate threshold the agent must reach before advancing to the next stage.
# The final stage has no threshold — it always runs to completion.
DEFAULT_CURRICULUM_STAGES = [
    {"n_obstacles": 0, "speed": 1.0, "timesteps": 500_000, "threshold": 0.70},
    {"n_obstacles": 2, "speed": 0.8, "timesteps": 500_000, "threshold": 0.60},
    {"n_obstacles": 4, "speed": 1.0, "timesteps": 1_000_000, "threshold": 0.55},
    {"n_obstacles": 6, "speed": 1.0, "timesteps": 1_500_000, "threshold": 0.50},
    {"n_obstacles": 10, "speed": 1.0, "timesteps": 1_500_000, "threshold": None},
]

def load_config(config_path = CONFIG_PATH):
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

def train(n_dynamic_obstacles = None, obstacle_speed = None, pretrained_model_path = None, use_reward_shaping = True):
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
    shaping_tag = _shaping_tag(use_reward_shaping)
    suffix = _model_suffix(n_dynamic_obstacles, obstacle_speed, shaping_tag)

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
        if not os.path.exists(pretrained_model_path + ".zip"):
            raise FileNotFoundError(f"Pretrained model not found: {pretrained_model_path}.zip")

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
    final_model_path = os.path.join(model_dir, f"ppo_robot_nav_{suffix}")
    agent.save(final_model_path)
    _save_meta(final_model_path, {"tb_log_folder": agent.logger.dir})
    print(f"Training complete. Model saved to: {final_model_path}.zip")

    training_env.close()
    eval_env.close()

def train_curriculum(stages=None, use_reward_shaping=True):
    """
    Multi-stage curriculum training pipeline (v3).

    Trains the agent progressively through stages of increasing difficulty.
    Each stage starts from the model saved in the previous stage, so the agent builds on what it already learned rather than starting from scratch.

    After all stages complete, the final model is copied to a canonical alias so that experiment files can reference a stable, predictable path:
        models/ppo_robot_nav_curriculum_shaping.zip      (use_reward_shaping=True)
        models/ppo_robot_nav_curriculum_no_shaping.zip   (use_reward_shaping=False)

    A stage is skipped to the next one early if the best model saved by EvalCallback already exceeds the stage threshold — this avoids wasting timesteps when the agent has already mastered the current difficulty.

    Parameters
    stages: list of dict or None
        List of stage definitions. Each dict must contain:
            n_obstacles (int): number of dynamic obstacles for this stage
            speed (float): obstacle speed for this stage
            timesteps (int): maximum timesteps to train for this stage
            threshold (float or None): success rate (0–1) required to advance;
                                       None means always run to completion
        If None, DEFAULT_CURRICULUM_STAGES is used.
    use_reward_shaping: bool
        Passed through to each train() call. If True, includes progress reward and proximity penalty. Should be True for curriculum training.

    Returns
    str
        Path to the final saved model.
    """
    if stages is None:
        stages = DEFAULT_CURRICULUM_STAGES

    config = load_config(CONFIG_PATH)
    model_dir = config["paths"]["model_dir"]
    shaping_tag = _shaping_tag(use_reward_shaping)

    os.makedirs(model_dir, exist_ok=True)

    # Will be set to the best available model path after each stage
    pretrained_path = None

    for stage_index, stage in enumerate(stages):
        n_obstacles = stage["n_obstacles"]
        speed = stage["speed"]
        timesteps = stage["timesteps"]
        threshold = stage["threshold"]

        print(f"\n{'-'*60}")
        print(f"Curriculum stage {stage_index + 1}/{len(stages)}")
        print(f"N obstacles: {n_obstacles}")
        print(f"Speed: {speed}")
        print(f"Timesteps: {timesteps:,}")
        if threshold is not None:
            print(f"Threshold: {f'{threshold:.0%}'}")
        else:
            print(f"Threshold: {'none (final stage)'}")        
        if pretrained_path is not None:
            print(f"Starting from: {pretrained_path}.zip")

        # Override total_timesteps for this stage
        config["training"]["total_timesteps"] = timesteps

        training_env = create_parallel_envs(config, CONFIG_PATH, n_obstacles, speed, use_reward_shaping)
        eval_env = create_eval_env(CONFIG_PATH, n_obstacles, speed, use_reward_shaping)

        if pretrained_path is None:
            agent = build_ppo_agent(config, training_env, config["paths"]["log_dir"])
        else:
            if not os.path.exists(pretrained_path + ".zip"):
                raise FileNotFoundError(f"Previous stage model not found: {pretrained_path}.zip")
            agent = PPO.load(pretrained_path, env=training_env, tensorboard_log=config["paths"]["log_dir"])

        stage_suffix = f"curriculum_stage{stage_index + 1}_obs{n_obstacles}_spd{speed}_{shaping_tag}"
        callbacks = build_callbacks(config, eval_env, n_obstacles, speed)

        agent.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True,
            # Keep the global step counter continuous across stages
            reset_num_timesteps=(pretrained_path is None)
        )

        # Save end-of-stage model
        stage_model_path = os.path.join(model_dir, f"ppo_robot_nav_{stage_suffix}")
        agent.save(stage_model_path)
        _save_meta(stage_model_path, {
            "stage": stage_index + 1,
            "n_obstacles": n_obstacles,
            "speed": speed,
            "tb_log_folder": agent.logger.dir,
        })
        print(f"\nStage {stage_index + 1} complete. Model saved to: {stage_model_path}.zip")

        training_env.close()
        eval_env.close()

        # Check threshold: use best model from EvalCallback if available, otherwise fall back to the end-of-stage model for the next stage
        best_model_dir = os.path.join(model_dir, f"best_obs{n_obstacles}_spd{speed}", "best_model")
        if os.path.exists(best_model_dir + ".zip"):
            pretrained_path = best_model_dir
            print(f"Using best model for next stage: {best_model_dir}.zip")
        else:
            pretrained_path = stage_model_path

        # If a threshold is set and the best eval reward already exceeds it, skip remaining timesteps in this stage.
        # SB3 EvalCallback saves evaluations to a numpy file we can read.
        if threshold is not None and stage_index < len(stages) - 1:
            _check_threshold(config, n_obstacles, speed, threshold)

    # Save a canonical alias at a stable path so experiment files do not need to know which stage number was the last one
    alias = os.path.join(model_dir, f"ppo_robot_nav_curriculum_{shaping_tag}")
    shutil.copy2(pretrained_path + ".zip", alias + ".zip")

    print(f"\n{'-'*60}")
    print(f"Curriculum training complete.")
    print(f"Final model: {pretrained_path}.zip")

    return alias

def _model_suffix(n_obstacles, speed, shaping_tag):
    """
    Build the canonical model filename suffix.

    Returns a string of the form ``obs{n}_spd{speed}_{shaping_tag}`` used consistently across train(), train_curriculum() and experiment files.

    Parameters
    n_obstacles: int
        Number of dynamic obstacles.
    speed: float
        Obstacle movement speed.
    shaping_tag: str
        Either ``"shaping"`` or ``"no_shaping"``.

    Returns
    str
        Canonical suffix string.
    """
    return f"obs{n_obstacles}_spd{speed}_{shaping_tag}"

def _save_meta(model_path, data):
    """
    Write a JSON sidecar file alongside a saved model.

    The sidecar records metadata such as the TensorBoard log folder and curriculum stage, which are useful for debugging and result analysis.

    Parameters
    model_path: str
        Path to the model file, without the .zip extension.
        The sidecar is saved as ``{model_path}_meta.json``.
    data: dict
        Metadata to serialise.
    """
    with open(model_path + "_meta.json", "w") as f:
        json.dump(data, f, indent=2)


def _check_threshold(config, n_obstacles, speed, threshold):
    """
    Read EvalCallback's evaluations.npz and log whether the success-rate
    threshold was reached for the current curriculum stage.

    This function is informational only — the curriculum always advances
    to the next stage regardless of the result.

    The success rate is approximated from the mean eval reward using:
        success_rate ≈ max(0, best_mean_reward) / reward["goal"]
    This is a heuristic: it works best when most reward comes from the
    goal signal (10.0 by default) rather than shaping terms.

    Parameters
    config : dict
        Full configuration dictionary (used for log_dir and reward.goal).
    n_obstacles : int
        Number of dynamic obstacles in the current stage.
    speed : float
        Obstacle speed in the current stage.
    threshold : float
        Target success rate (0–1) to reach before advancing.
    """
    eval_log_path = os.path.join(
        config["paths"]["log_dir"],
        f"eval_obs{n_obstacles}_spd{speed}",
        "evaluations.npz",
    )
    if not os.path.exists(eval_log_path):
        return

    evals = np.load(eval_log_path)
    best_mean_reward   = float(np.max(evals["results"].mean(axis=1)))
    approx_success_rate = max(0.0, best_mean_reward) / config["reward"]["goal"]
    print(f"Best eval success rate estimate : {approx_success_rate:.1%}  "
          f"(threshold: {threshold:.0%})")
    if approx_success_rate >= threshold:
        print("Threshold reached — advancing to next stage.")
    else:
        print("Threshold not yet reached — continuing curriculum.")

def _shaping_tag(use_reward_shaping):
    """
    Return the canonical shaping label used in model filenames.

    Parameters
    use_reward_shaping : bool
        Whether reward shaping is enabled.

    Returns
    str
        ``"shaping"`` or ``"no_shaping"``.

    """
    if use_reward_shaping:
        return "shaping" 
    
    return "no_shaping"