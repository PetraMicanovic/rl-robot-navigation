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
    # Stage 1: pure navigation — no obstacles, agent learns to reach the target
    {"n_obstacles": 0, "speed": 1.0, "timesteps": 500_000, "threshold": 0.70, "max_retries": 2},
    # Stage 2: introduce slow obstacles at low density to begin collision avoidance
    {"n_obstacles": 3, "speed": 0.5, "timesteps": 600_000, "threshold": 0.55, "max_retries": 1},
    # Stage 3: increase obstacle speed to standard level
    {"n_obstacles": 3, "speed": 1.0, "timesteps": 600_000, "threshold": 0.50, "max_retries": 1},
    # Stage 4: increase obstacle density while keeping standard speed
    {"n_obstacles": 6, "speed": 1.0, "timesteps": 800_000, "threshold": 0.45, "max_retries": 1},
    # Stage 5: final density — no threshold, always runs to completion
    {"n_obstacles": 10, "speed": 1.0, "timesteps": 1_000_000, "threshold": None, "max_retries": 0},
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
    training_env = create_parallel_envs(config, CONFIG_PATH, n_dynamic_obstacles, obstacle_speed, use_reward_shaping = use_reward_shaping)
    eval_env = create_eval_env(CONFIG_PATH, n_dynamic_obstacles, obstacle_speed, use_reward_shaping = use_reward_shaping)

    # Create a new PPO agent or load a pretrained one
    if pretrained_model_path is None:
        agent = build_ppo_agent(config, training_env, log_dir)
    else:
        if not os.path.exists(pretrained_model_path + ".zip"):
            raise FileNotFoundError(f"Pretrained model not found: {pretrained_model_path}.zip")

        print(f"Loading pretrained model from: {pretrained_model_path}")
        agent = PPO.load(
            pretrained_model_path,
            env=training_env,
            tensorboard_log=log_dir,
            custom_objects={"ent_coef": config["training"]["ent_coef"]},
        )
    # Pass n_eval_episodes explicitly so train() always uses the value from
    # config["evaluation"] and is never affected by curriculum overrides.
    callbacks = build_callbacks(
        config,
        eval_env,
        n_dynamic_obstacles,
        obstacle_speed,
        n_eval_episodes=config["evaluation"]["n_eval_episodes"],
    )


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
            speed_range  (list/tuple): [min, max] speed range
            timesteps (int): maximum timesteps to train for this stage
            threshold (float or None): success rate (0–1) required to advance;
                                       None means always run to completion
            max_retries (int): number of extra training passes if threshold missed 
    use_reward_shaping: bool
        Passed through to each train() call. If True, includes progress reward and proximity penalty. Should be True for curriculum training.

    Returns
    str
        Path to the final saved model.
    """
    if stages is None:
        stages = DEFAULT_CURRICULUM_STAGES

    config = load_config(CONFIG_PATH)
    curriculum_config = config.get("curriculum", {})
    curriculum_ent_coef = curriculum_config.get("ent_coef", config["training"]["ent_coef"])
    curriculum_n_eval = curriculum_config.get("n_eval_episodes", config["evaluation"]["n_eval_episodes"])

    model_dir = config["paths"]["model_dir"]
    log_dir = config["paths"]["log_dir"]
    shaping_tag = _shaping_tag(use_reward_shaping)

    os.makedirs(model_dir, exist_ok=True)

    # Path to the model that the next stage will start from.
    # None means build a fresh agent for stage 1.
    pretrained_path = None

    for stage_index, stage in enumerate(stages):
        n_obstacles = stage["n_obstacles"]
        speed = stage.get("speed", None)
        speed_range = stage.get("speed_range", None)
        timesteps = stage["timesteps"]
        threshold = stage["threshold"]
        max_retries = stage.get("max_retries", 1)

        if speed is None and speed_range is None:
            raise ValueError(f"Stage {stage_index + 1} must define either 'speed' or 'speed_range'.")

        if speed_range is not None:
            speed_str = f"range={speed_range}"  
        else:
            speed_str = f"fixed={speed}"

        print(f"\n{'-'*60}")
        print(f"Curriculum stage {stage_index + 1}/{len(stages)}")
        print(f"N obstacles: {n_obstacles}")
        print(f"Speed: {speed_str}")
        print(f"Timesteps: {timesteps:,}")
        if threshold is not None:
            print(f"Threshold: {f'{threshold:.0%}'}")
        else:
            print(f"Threshold: {'none (final stage)'}")   
        print(f"max_retries: {max_retries}")     
        if pretrained_path is not None:
            print(f"Starting from: {pretrained_path}.zip")

        # Override total_timesteps for this stage
        config["training"]["total_timesteps"] = timesteps

        training_env = create_parallel_envs(config, CONFIG_PATH, n_obstacles, speed, speed_range, use_reward_shaping = use_reward_shaping)
        if speed_range is not None:
            eval_speed = float(np.mean(speed_range))
        else:
            eval_speed = speed
        eval_env = create_eval_env(CONFIG_PATH, n_obstacles, eval_speed, use_reward_shaping = use_reward_shaping)

        # Build or load agent
        if pretrained_path is None:
            config["training"]["ent_coef"] = curriculum_ent_coef
            agent = build_ppo_agent(config, training_env, log_dir)    
        else:
            if not os.path.exists(pretrained_path + ".zip"):
                raise FileNotFoundError(f"Previous stage model not found: {pretrained_path}.zip")
            agent = PPO.load(
                pretrained_path,
                env=training_env,
                tensorboard_log=log_dir,
                custom_objects={"ent_coef": curriculum_ent_coef}
            )
        stage_suffix = f"curriculum_stage{stage_index + 1}_obs{n_obstacles}_spd{speed or speed_range}_{shaping_tag}"
        # Canonical best-model directory for this stage (shared across retries
        # so EvalCallback always updates the same "global best" for the stage).
        best_model_dir = os.path.join(model_dir, f"best_obs{n_obstacles}_spd{eval_speed}")

        # Retry loop 
        threshold_met = False
        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"Retry {attempt}/{max_retries} — threshold not met, reloading best model.")
                # Reload the best model saved during the previous attempt rather than continuing from degraded weights.
                best_model_path = os.path.join(best_model_dir, "best_model")
                if os.path.exists(best_model_path + ".zip"):
                    print(f"Reloading best model: {best_model_path}.zip")
                    agent = PPO.load(
                        best_model_path,
                        env=training_env,
                        tensorboard_log=log_dir,
                        # Use local curriculum_ent_coef, not the unmodified config value
                        custom_objects={"ent_coef": curriculum_ent_coef},
                    )
                else:
                    print("No best model found — continuing from end-of-previous-attempt weights.")

            # Rebuild callbacks with a per-attempt log path so that evaluations.npz is fresh and _read_success_rate() is accurate.
            attempt_log_suffix = (f"obs{n_obstacles}_spd{eval_speed}_attempt{attempt}")
            callbacks = build_callbacks(
                config,
                eval_env,
                n_obstacles,
                eval_speed,
                best_model_save_path=best_model_dir,
                log_suffix=attempt_log_suffix,
                n_eval_episodes=curriculum_n_eval,
            )


            agent.learn(
                total_timesteps=timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=False,  # keep global timestep counter
            )

            if threshold is None:
                threshold_met = True
                break

            # Read success rate from the per-attempt evaluations.npz
            rate = _read_success_rate(config, n_obstacles, eval_speed, attempt)
            print(f"Eval success rate: {rate:.1%}  (threshold: {threshold:.0%})")

            if rate >= threshold:
                print("Threshold met — advancing to next stage.")
                threshold_met = True
                break
            elif attempt < max_retries:
                print(f"Threshold not met — will retry ({attempt + 1}/{max_retries}).")

        if not threshold_met:
            print(f"Threshold not met after {max_retries + 1} attempt(s) — advancing anyway.")

        # Save end-of-stage model
        stage_model_path = os.path.join(model_dir, f"ppo_robot_nav_{stage_suffix}")
        agent.save(stage_model_path)
        _save_meta(stage_model_path, {
            "stage": stage_index + 1,
            "n_obstacles": n_obstacles,
            "speed": speed,
            "speed_range": speed_range,
            "threshold_met": threshold_met,
            "tb_log_folder": agent.logger.dir,
        })
        print(f"\nStage {stage_index + 1} complete. Model saved to: {stage_model_path}.zip")

        training_env.close()
        eval_env.close()

        # Prefer the best model (highest eval reward) for the next stage; fall back to the end-of-stage model if none was saved.
        best_model_path = os.path.join(best_model_dir, "best_model")
        if os.path.exists(best_model_path + ".zip"):
            pretrained_path = best_model_path
            print(f"Using best model for next stage: {pretrained_path}.zip")
        else:
            pretrained_path = stage_model_path
            print(f"Using end-of-stage model for next stage: {pretrained_path}.zip")

    # Canonical alias 
    alias = os.path.join(model_dir, f"ppo_robot_nav_curriculum_{shaping_tag}")
    shutil.copy2(pretrained_path + ".zip", alias + ".zip")

    print(f"\n{'-'*60}")
    print(f"Curriculum training complete.")
    print(f"Final model: {pretrained_path}.zip")
    print(f"Alias: {alias}.zip")

    return alias

def _read_success_rate(config, n_obstacles, speed, attempt = 0):
    """
    Read the most recent success rate from EvalCallback's evaluations.npz.

    Uses the per-attempt log path written by build_callbacks so retries do not contaminate each other's measurements.

    Parameters
    config: dict
    n_obstacles: int
    speed: float
    attempt : int
        Attempt index (0-based). Must match the suffix used in build_callbacks.

    Returns
    float
        Success rate in [0, 1], or 0.0 if the file does not exist yet.
    """
    attempt_log_suffix = f"obs{n_obstacles}_spd{speed}_attempt{attempt}"

    eval_log_path = os.path.join(
        config["paths"]["log_dir"],
        f"eval_{attempt_log_suffix}",
        "evaluations.npz",
    )
    if not os.path.exists(eval_log_path):
        return 0.0

    evals = np.load(eval_log_path)

    if "successes" in evals and evals["successes"].size > 0:
        # Shape: (n_evals, n_eval_episodes) — take the most recent eval row
        return float(evals["successes"][-1].mean())

    # Fallback: reward heuristic (less accurate, kept for backward compat)
    best_mean_reward = float(np.max(evals["results"].mean(axis=1)))
    return max(0.0, best_mean_reward) / config["reward"]["goal"]

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