# rl-robot-navigation

Autonomous navigation of a mobile robot in a dynamic 2D environment using **Proximal Policy Optimization (PPO)** — a deep reinforcement learning algorithm. The agent learns to reach a target position while avoiding both static and moving obstacles, without any predefined rules or map of the environment.

The robot learns a reactive navigation policy entirely through trial and error in a custom simulation, with no prior knowledge of the environment dynamics.

---

## Environment

The environment is built from scratch as a custom Gymnasium environment — a 2D space where the robot has to find its way to a target while dealing with both walls and moving obstacles. To sense its surroundings, the agent uses a simulated LIDAR with 24 rays that measure how far the nearest obstacle is in each direction. Training runs with PPO through Stable-Baselines3, with 8 environments running in parallel to speed things up. The experiments cover a few different scenarios — changing how many obstacles there are, how fast they move, and testing whether the agent can handle layouts it has never seen before. At the end, the results are visualized through learning curves and trajectory plots to get a clearer picture of what the agent actually learned.

Each episode starts with a randomized agent position and target, preventing the agent from memorizing specific scenarios.

| Element | Description |
|---|---|
| Agent | Continuous 2D velocity control |
| Target | Randomly placed each episode |
| Static obstacles | Fixed walls and objects |
| Dynamic obstacles | N entities moving on random trajectories |

**Observation space:**

```
s = [dx, dy, vx, vy, l1...l24, ovx1, ovy1, ovx2, ovy2, ovx3, ovy3]
```

- `dx, dy` — relative position of the target (normalized by world size)
- `vx, vy` — current agent velocity (normalized by max speed)
- `l1...l24` — LIDAR readings at 15° intervals (normalized by lidar range)
- `ovx1, ovy1...ovx3, ovy3` — velocity of the 3 nearest dynamic obstacles (normalized; zero-padded if fewer than 3 present)

Total observation size: **34** (up from 12 in v1)

**Action space:** continuous vector `[vx, vy] ∈ [−1, 1]²`

**Reward function:**

```
+10.0        reaching the target
−10.0        collision with any obstacle (episode ends)
+0.1·Δd      progress toward the target (reward shaping)
−0.01        penalty per timestep (encourages efficiency)
−f(d)        proximity penalty, active when closer than danger_zone_radius
```
The proximity penalty `f(d)` scales smoothly from `0` at `danger_zone_radius` to `-proximity_penalty_scale` at contact distance. The idea is to discourage the agent from getting too close to obstacles even when it doesn't actually collide. Both shaping terms are only active when `use_reward_shaping=True`.

## Installation

```bash
git clone https://github.com/PetraMicanovic/rl-robot-navigation.git
cd rl-robot-navigation
pip install -r requirements.txt
```

**Requirements:** 
- Python 3.10+, see `requirements.txt` for full list.
---
## Releases

### [v1.0-baseline](https://github.com/PetraMicanovic/rl-robot-navigation/releases/tag/v1.0-baseline)

Initial working version of the environment and training pipeline.

- 8-ray LIDAR observation space: `[dx, dy, vx, vy, l1...l8]` (12 dimensions)
- Basic reward function: goal, collision, progress shaping, step penalty
- Experiments E1–E4 covering obstacle density, speed, generalization, and reward shaping

### v2.0-obs-reward 

Major upgrade to observation space and reward function.

**Observation space** expanded from 12 to 34 dimensions:
- LIDAR resolution increased from 8 to **24 rays** (45° → 15° angular resolution)
- Added velocities of the **3 nearest dynamic obstacles** `[vx1, vy1, vx2, vy2, vx3, vy3]` — gives the agent awareness of obstacle movement direction, not just proximity

**Reward function** extended with a proximity penalty:
- Smooth penalty that scales from `0` at `danger_zone_radius` down to `-proximity_penalty_scale` at contact distance — discourages the agent from lingering near obstacles even without colliding

---