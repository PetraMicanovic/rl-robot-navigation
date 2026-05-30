import time
import numpy as np
import sys
import os

from robot_env.robot_nav_env import RobotNavEnv

def main():
    # Create environment with visualization enabled
    env = RobotNavEnv(
        config_path=os.path.join(os.path.dirname(__file__), "config.json"),
        n_dynamic_obstacles=3,
        render_mode="human",
    )

    print("Environment created successfully.")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"World size: {env.WORLD_SIZE}")
    print(f"Dynamic obstacles: {env.n_dynamic_obstacles}")
    print(f"Obstacle speed: {env.obstacle_speed}")
    print()

    number_of_episodes = 3

    for episode_index in range(number_of_episodes):
        observation, info = env.reset(seed=episode_index)
        episode_reward = 0
        step_count = 0
        terminated = False
        truncated = False

        print(f"Episode {episode_index + 1} started.")

        while not terminated and not truncated:
            # Random agent — samples a random action from the action space
            action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            # Slow down rendering so the window is visible
            time.sleep(0.03)

        if info["success"]:
            outcome = "SUCCESS — reached target"
        elif truncated:
            outcome = "TRUNCATED — max steps reached"
        else:
            outcome = "COLLISION"

        print(f"Outcome: {outcome}")
        print(f"Steps taken: {step_count}")
        print(f"Total reward: {episode_reward:.3f}")
        print(f"Final distance: {info['distance_to_target']:.3f}")
        print()

    env.close()
    print("All episodes finished. Environment closed.")


if __name__ == "__main__":
    main()
