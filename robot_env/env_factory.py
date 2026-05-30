"""
Environment creation module for RobotNavEnv training.
This module contains factory functions used to create vectorized training environments and evaluation environments for RL experiments.
"""
import os

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from robot_env.robot_nav_env import RobotNavEnv


def make_env(config_path, n_dynamic_obstacles, obstacle_speed, rank, seed=0):
    """
    Factory function used by SubprocVecEnv to create parallel RobotNavEnv instances.
    Each environment receives a different random seed.

    Parameters
    config_path: str
        Path to config.json.
    n_dynamic_obstacles: int
        Number of dynamic obstacles in the environment.
    obstacle_speed: float
        Speed of dynamic obstacles.
    rank: int
        Index of this environment in the parallel pool (used for seeding).
    seed: int
        Base random seed. Each environment gets seed + rank.

    Returns
    callable
        A no-argument function that returns an initialized RobotNavEnv.
    """
    # Function returned to SubprocVecEnv
    def environment_factory():
        env = RobotNavEnv(
            config_path=config_path,
            n_dynamic_obstacles=n_dynamic_obstacles,
            obstacle_speed=obstacle_speed,
            render_mode=None
        )
        env.reset(seed=seed + rank)
        return env

    return environment_factory


def create_parallel_envs(config, config_path, n_dynamic_obstacles, obstacle_speed):
    """
    Create a vectorized environment with multiple parallel instances.
    Wraps SubprocVecEnv with VecMonitor to record episode rewards and lengths for TensorBoard monitoring.

    Parameters
    config: dict
        Full configuration dictionary.
    config_path: str
        Path to config.json, passed to each environment instance.
    n_dynamic_obstacles: int
        Number of dynamic obstacles in each environment.
    obstacle_speed: float
        Speed of dynamic obstacles in each environment.

    Returns
    VecMonitor
        Vectorized and monitored parallel environment.
    """
    number_of_envs = config["training"]["n_envs"]
    log_dir = config["paths"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    env_factory_list = []
    for i in range(number_of_envs):
        env_factory_list.append(make_env(config_path, n_dynamic_obstacles, obstacle_speed, rank=i))

    parallel_env = SubprocVecEnv(env_factory_list)
    monitored_env = VecMonitor(parallel_env, filename=os.path.join(log_dir, "monitor"))

    return monitored_env


def create_eval_env(config_path, n_dynamic_obstacles, obstacle_speed):
    """
    Create a single environment used for periodic evaluation during training.

    Parameters
    config_path: str
        Path to config.json.
    n_dynamic_obstacles: int
        Number of dynamic obstacles.
    obstacle_speed: float
        Speed of dynamic obstacles.

    Returns
    VecMonitor
        Vectorized and monitored single environment instance for evaluation.
    """
    env_factory = make_env(
        config_path=config_path,
        n_dynamic_obstacles=n_dynamic_obstacles,
        obstacle_speed=obstacle_speed,
        rank=0
    )
    eval_env = DummyVecEnv([env_factory])
    
    return VecMonitor(eval_env)