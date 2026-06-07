"""
Evaluation module for a trained PPO agent in RobotNavEnv.
Loads a trained model, evaluates it over a fixed number of episodes, collects aggregate and per-episode metrics and saves evaluation results to a JSON file.
"""

import os
import json
from stable_baselines3 import PPO
from robot_env.robot_nav_env import RobotNavEnv

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


def evaluate(model_path, n_dynamic_obstacles=None, obstacle_speed=None, n_eval_episodes=None, experiment = None, label = None):
    """
    Evaluate a trained PPO agent over a fixed number of episodes.

    Runs the agent deterministically (no exploration) and collects per-episode metrics. Prints a summary at the end.

    Parameters
    model_path: str
        Path to the saved model (.zip file).
    n_dynamic_obstacles: int or None
        Number of dynamic obstacles. Overrides config if provided.
    obstacle_speed: float or None
        Speed of dynamic obstacles. Overrides config if provided.
    n_eval_episodes: int or None
        Number of evaluation episodes. Overrides config if provided.
    experiment: str
        Experiment name used for saving results.
    label: str or None
        Optional label appended to the results filename to distinguish between variants evaluated on the same configuration.

    Returns
    dict
        Dictionary with aggregated evaluation results:
        - 'success_rate'(float): fraction of successful episodes
        - 'avg_episode_length'(float): average number of steps per episode
        - 'avg_collision_count'(float): average collisions per episode
        - 'n_success'(int): number of successful episodes
        - 'n_collision'(int): number of episodes ending in collision
        - 'n_truncated'(int): number of truncated episodes
        - 'n_episodes'(int): total episodes evaluated
        - 'episodes'(list): detailed per-episode evaluation results
    """
    config = load_config(CONFIG_PATH)

    # Apply overrides or use config defaults
    if n_dynamic_obstacles is None:
        n_dynamic_obstacles = config["environment"]["n_dynamic_obstacles"]
    if obstacle_speed is None:
        obstacle_speed = config["environment"]["obstacle_speed"]
    if n_eval_episodes is None:
        n_eval_episodes = config["evaluation"]["n_eval_episodes"]

    deterministic = config["evaluation"]["deterministic"]

    print("PPO Evaluation — RobotNavEnv")
    print(f"Model: {model_path}")
    print(f"Dynamic obstacles: {n_dynamic_obstacles}")
    print(f"Obstacle speed: {obstacle_speed}")
    print(f"Episodes: {n_eval_episodes}")
    print(f"Deterministic: {deterministic}")
    print()

    # Load trained model
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {model_path}")

    eval_env = RobotNavEnv(
        config_path=CONFIG_PATH,
        n_dynamic_obstacles=n_dynamic_obstacles,
        obstacle_speed=obstacle_speed,
        render_mode=None
    )

    model = PPO.load(model_path, env=eval_env)
    print("Model loaded successfully.")
    print()

    # Metrics accumulators 
    number_of_successes = 0
    number_of_collisions = 0
    number_of_truncations = 0
    total_steps = 0
    total_collisions = 0

    # Per-episode metrics for JSON export and analysis
    episode_results = []

    for episode_index in range(n_eval_episodes):
        observation, info = eval_env.reset(seed=episode_index)
        episode_step_count = 0
        episode_collision_count = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            # Run agent deterministically — no random exploration
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, info = eval_env.step(action)
            episode_step_count += 1

            # Count collisions: episode ended due to collision
            if terminated and not info["success"]:
                episode_collision_count += 1

        # Classify episode outcome
        if info["success"]:
            number_of_successes += 1
            outcome = "SUCCESS"
        elif truncated:
            number_of_truncations += 1
            outcome = "TRUNCATED"
        else:
            number_of_collisions  += 1
            outcome = "COLLISION"

        total_steps += episode_step_count
        total_collisions += episode_collision_count

        episode_results.append({
            "episode": episode_index + 1,
            "outcome": outcome,
            "steps": episode_step_count,
            "collisions": episode_collision_count,
            "final_dist": info["distance_to_target"]
        })

        # Print progress every 50 episodes
        if (episode_index + 1) % 50 == 0:
            current_success_rate = number_of_successes / (episode_index + 1) * 100
            print(f"Episode {episode_index + 1:>4} / {n_eval_episodes}")
            print(f"Success rate so far: {current_success_rate:.1f}%")

    eval_env.close()

    # Compute summary metrics 
    success_rate = number_of_successes / n_eval_episodes
    avg_episode_length = total_steps / n_eval_episodes
    avg_collision_count = total_collisions / n_eval_episodes

    results = {
        "success_rate": success_rate,
        "avg_episode_length": avg_episode_length,
        "avg_collision_count": avg_collision_count,
        "n_success": number_of_successes,
        "n_collision": number_of_collisions,
        "n_truncated": number_of_truncations,
        "n_episodes": n_eval_episodes,
        "episodes": episode_results
    }

    # Print summary 
    print()
    print("Evaluation Results")
    print(f"Episodes evaluated: {n_eval_episodes}")
    print(f"Successful: {number_of_successes} ({success_rate * 100:.1f}%)")
    print(f"Collisions: {number_of_collisions} ({number_of_collisions / n_eval_episodes * 100:.1f}%)")
    print(f"Truncated: {number_of_truncations} ({number_of_truncations / n_eval_episodes * 100:.1f}%)")
    print(f"Average episode length: {avg_episode_length:.1f} steps")
    print(f"Average collisions/episode: {avg_collision_count:.3f}")

    # Save evaluation results to JSON file
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    if experiment is not None:
        results_dir = os.path.join(results_dir, experiment)
    os.makedirs(results_dir, exist_ok=True)
    
    if label:
        results_filename = f"eval_obs{n_dynamic_obstacles}_spd{obstacle_speed}_{label}.json"
    else:
        results_filename = f"eval_obs{n_dynamic_obstacles}_spd{obstacle_speed}.json"    
    results_path = os.path.join(results_dir, results_filename)
    with open(results_path, "w", encoding="utf-8") as results_file:        
        json.dump(results, results_file, indent=4, ensure_ascii=False)    
    print(f"Results saved to: {results_path}")
    
    return results