# rl-robot-navigation

Autonomous navigation of a mobile robot in a dynamic 2D environment using **Proximal Policy Optimization (PPO)** — a deep reinforcement learning algorithm. The agent learns to reach a target position while avoiding both static and moving obstacles, without any predefined rules or map of the environment.

The robot learns a reactive navigation policy entirely through trial and error in a custom simulation, with no prior knowledge of the environment dynamics.

---

## Environment

The environment is built from scratch as a custom Gymnasium environment — a 2D space where the robot has to find its way to a target while dealing with both walls and moving obstacles. To sense its surroundings, the agent uses a simulated LIDAR with 8 rays that measure how far the nearest obstacle is in each direction. Training runs with PPO through Stable-Baselines3, with 8 environments running in parallel to speed things up. The experiments cover a few different scenarios — changing how many obstacles there are, how fast they move, and testing whether the agent can handle layouts it has never seen before. At the end, the results are visualized through learning curves and trajectory plots to get a clearer picture of what the agent actually learned.

de mi Each episode starts with a randomized agent position and target, preventing the agent from memorizing specific scenarios.

| Element | Description |
|---|---|
| Agent | Continuous 2D velocity control |
| Target | Randomly placed each episode |
| Static obstacles | Fixed walls and objects |
| Dynamic obstacles | N entities moving on random trajectories |

**Observation space:**

```
s = [dx, dy, vx, vy, l1, l2, l3, l4, l5, l6, l7, l8]
```

- `dx, dy` — relative position of the target
- `vx, vy` — current agent velocity
- `l1...l8` — LIDAR readings (distance to nearest obstacle in 8 directions)

**Action space:** continuous vector `[vx, vy] ∈ [−1, 1]²`

**Reward function:**

```
+10.0   reaching the target
−10.0   collision with any obstacle (episode ends)
+0.1·Δd progress toward the target (reward shaping)
−0.01   penalty per timestep (encourages efficiency)
```
## Installation

```bash
git clone https://github.com/your-username/rl-robot-navigation.git
cd rl-robot-navigation
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, see `requirements.txt` for full list.
---