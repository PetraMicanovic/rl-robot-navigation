# Evaluation Results — v4.0-action-smoothing

This document summarizes the results of four experiments evaluating a PPO-based robot navigation agent trained with the v4.0 configuration. All experiments use 200 evaluation episodes with a deterministic policy.

Two changes were introduced compared to v3.0-curriculum:

```json
"action_smoothing_scale": 0.16
```

```python
lr = linear_schedule(3e-4)
```

Action smoothing adds a penalty for large action changes between consecutive steps, discouraging erratic movement. Several values were tested; 0.16 gave the best results. The linear LR schedule decays the learning rate from 3e-4 to 0 over the course of training, allowing larger updates early on and smaller, more conservative updates as the policy matures. Both changes contributed to consistent improvements over v3.

---

## Training configuration

| Parameter | Value |
|---|---|
| Algorithm | PPO |
| Network | MLP `[256, 256]` |
| Observation space | 24 lidar rays, agent velocity, target direction, 3 nearest obstacle velocities |
| Action space | Continuous velocity control |
| Curriculum stages | 10 (N=0 → N=2 → N=3 → ... → N=13, with variable speed from stage 4) |
| Action smoothing scale | 0.16 |
| Learning rate | linear_schedule(3e-4) |
| Total timesteps | ~ 6 300 000 |

---

## E1 — Effect of obstacle density

Evaluated at fixed speed=1.0 across three obstacle counts.

| Obstacles | Success | Collisions | Truncated | Avg steps | Avg col/ep | vs v3 |
|---|---|---|---|---|---|---|
| N=3 | 86.5% (173/200) | 13.5% | 0.0% | 5.5 | 0.135 | +27.0pp |
| N=6 (training) | 71.5% (143/200) | 28.5% | 0.0% | 4.9 | 0.285 | +20.0pp |
| N=10 | 62.0% (124/200) | 38.0% | 0.0% | 4.7 | 0.380 | +19.5pp |

![E1 metrics](figures/e1_metrics.png)

Success rate drops with obstacle count, as expected. The decline is gradual — about 24pp between N=3 and N=10 — which suggests the policy scales reasonably with density rather than breaking down outside the training condition. Gains over v3 are uniform across all three conditions (+20–27pp), so neither change appears to be density-specific. N=3 at 86.5% is the best single result across all versions so far.

Episode lengths are similar across conditions (~4.7–5.5 steps), meaning the agent is not compensating for higher density by slowing down or taking longer routes.

![E1 training curves](figures/e1_training_curves.png)

---

## E2 — Effect of obstacle speed

Evaluated at fixed N=6 across three speeds.

| Speed | Success | Collisions | Truncated | Avg steps | Avg col/ep | vs v3 |
|---|---|---|---|---|---|---|
| 0.5 (slower) | 77.5% (155/200) | 22.5% | 0.0% | 5.3 | 0.225 | +25.5pp |
| 1.0 (training) | 71.5% (143/200) | 28.5% | 0.0% | 4.9 | 0.285 | +20.0pp |
| 1.5 (faster) | 75.5% (151/200) | 24.5% | 0.0% | 5.2 | 0.245 | +24.5pp |

![E2 metrics](figures/e2_metrics.png)

Performance is fairly stable across speeds — only about 6pp separates the best and worst conditions. Notably, speed=1.5 (75.5%) outperforms the training condition speed=1.0 (71.5%). This did not occur in v3. The margin is small enough that it could be noise, but it is worth monitoring in future runs. One possible explanation is that faster obstacles move far enough between steps to be more predictable to avoid, while speed=1.0 obstacles are in a regime where they are neither slow enough to wait out nor fast enough to simply sidestep.

![E2 training curves](figures/e2_training_curves.png)

---

## E3 — Generalization to unseen configurations

Evaluated on seven configurations not seen during training.

| Obstacles | Speed | Success | Collisions | Truncated | Avg steps | Avg col/ep | vs v3 |
|---|---|---|---|---|---|---|---|
| 5 | 0.8 | 78.5% (157/200) | 21.5% | 0.0% | 5.2 | 0.215 | +19.0pp |
| 7 | 1.0 | 68.5% (137/200) | 31.5% | 0.0% | 5.1 | 0.315 | +20.0pp |
| 8 | 1.2 | 70.0% (140/200) | 30.0% | 0.0% | 4.9 | 0.300 | +23.0pp |
| 9 | 1.3 | 64.5% (129/200) | 35.5% | 0.0% | 4.7 | 0.355 | +14.0pp |
| 10 | 1.5 | 60.5% (121/200) | 39.5% | 0.0% | 4.7 | 0.395 | +21.5pp |
| 12 | 2.0 | 56.5% (113/200) | 43.5% | 0.0% | 4.7 | 0.435 | +16.0pp |
| 15 | 0.3 | 57.0% (114/200) | 43.0% | 0.0% | 4.5 | 0.430 | +18.5pp |

![E3 metrics](figures/e3_metrics.png)

All seven unseen configurations improved over v3, with gains between +14.0pp and +23.0pp. The hardest cases from earlier versions — N=15/speed=0.3 (6.5% in v2, 38.5% in v3) and N=12/speed=2.0 — now reach 57.0% and 56.5% respectively.

The trend across configurations is clear: performance degrades with obstacle count more than with speed. The (5, 0.8) and (7, 1.0) configurations sit around 68–78%, while anything with N≥10 falls below 65% regardless of speed. This is consistent with E1 and suggests that density, not speed, is the harder dimension for the agent to handle.

One outlier is N=9/speed=1.3, which gained only +14.0pp — the smallest improvement in the experiment. There is no obvious structural reason for this; it sits in the middle of the density range with a moderate speed. It may be noise, but it warrants attention if it appears again in future runs.

---

## E4 — Effect of reward shaping

Both variants trained with action smoothing and linear LR scheduler enabled, evaluated at N=6, speed=1.0.

| Variant | Success | Collisions | Truncated | Avg steps | Avg col/ep | vs v3 |
|---|---|---|---|---|---|---|
| With shaping | 71.5% (143/200) | 28.5% | 0.0% | 4.9 | 0.285 | +20.0pp |
| Without shaping | 28.5% (57/200) | 71.5% | 0.0% | 6.2 | 0.715 | +22.5pp |

![E4 metrics](figures/e4_metrics.png)

The 43pp gap between variants confirms that reward shaping remains the dominant factor in performance. The more notable result is the no-shaping variant: in both v2 and v3, training without shaping produced near-random behavior (~6%). Here it reaches 28.5%, which means the agent is learning something despite the absence of explicit shaping terms.

The most likely cause is the action smoothing penalty, which implicitly encourages the agent to move in consistent directions — a weak but nonzero learning signal on its own. The LR schedule may contribute as well by preventing early instability before any useful behavior emerges. Either way, the result suggests that action smoothing and reward shaping are complementary rather than redundant.

---

## Overview — all training curves

![All eval curves](figures/all_eval_curves.png)

---

## Discussion

### Effect of obstacle density

Obstacle density is the main limiting factor for this agent. Each increase in N reduces the navigable space and increases the likelihood of unavoidable collisions. The E1 results show a consistent ~8–10pp drop per density step, and the E3 results reinforce this — configurations with N≥10 all fall below 65% regardless of obstacle speed. At high densities, the agent's reactive, step-by-step navigation strategy starts to break down because avoiding one obstacle often puts it on a collision course with another. The current architecture and reward structure seem to approach a ceiling somewhere around N=10–12.

### Effect of obstacle speed

Speed has a smaller and less consistent effect than density. Across E1 and E2, varying speed between 0.5 and 1.5 costs at most 6pp, and the ordering is not monotonic — speed=1.5 outperforms speed=1.0 in E2, and in E3 the slowest configuration (N=15/speed=0.3) is among the worst results despite having slow obstacles. This suggests that once the agent has enough lidar information to detect an obstacle, the exact speed matters less than the number of obstacles it has to track simultaneously.

### Generalization

Generalization improved substantially from v3 to v4. In v3, out-of-distribution configurations dropped off sharply; here, the policy transfers reasonably to all seven tested configurations. The uniform size of the gains (+14–23pp) suggests this is not specific to any one condition. A likely contributor is the action smoothing, which discourages policies that rely on narrow, configuration-specific maneuvers — smoother trajectories tend to generalize better because they depend less on precise timing relative to obstacle positions. The LR schedule may help as well by reducing late-training overfitting to the curriculum's final stage distribution.

### Effect of reward shaping

Reward shaping remains essential for good performance — the 43pp gap in E4 makes that clear. What changed in v4 is that the no-shaping baseline is no longer broken. Going from ~6% (v2, v3) to 28.5% (v4) means the agent can learn basic goal-directed behavior from the smoothing penalty alone, without any explicit reward for reaching the target. This is a useful property: it means future experiments that modify the shaping terms will be starting from a less degenerate baseline, making ablations easier to interpret.

### Evolution from v2 to v4

v2 introduced the curriculum and the basic reward structure. Training was unstable and generalization was poor — hard configurations like N=15 sat at 6.5% and the no-shaping baseline was essentially random.

v3 refined the curriculum and improved generalization considerably (N=15 reached 38.5%), but collision rates remained high and training without shaping was still broken. The policy showed signs of learning reactive avoidance but not coherent navigation.

v4 addressed two issues: jittery movement (via action smoothing) and late-training instability (via the LR schedule). The combination pushed success rates up by ~20pp on average and, for the first time, produced a no-shaping baseline that learns something useful. The remaining gap — particularly at N≥10 — probably reflects a limitation of reactive policies on short rollout horizons rather than a reward engineering problem. Longer rollouts, richer observations, or more time spent at high densities during training are the natural next steps.

---

## Overall summary

| Experiment | Best result | Configuration | vs v3 |
|---|---|---|---|
| E1 | 86.5% | N=3, speed=1.0 | +27.0pp |
| E2 | 77.5% | N=6, speed=0.5 | +25.5pp |
| E3 | 78.5% | N=5, speed=0.8 | +19.0pp |
| E4 | 71.5% | shaping, N=6, speed=1.0 | +20.0pp |

v4 is the best version so far. Action smoothing (`scale=0.16`) and a linear LR schedule (3e-4 → 0) together gave ~+20pp over v3 across all experiments. Collision rates dropped from 40–60% in v3 to 15–45% depending on the configuration. The main open problem is performance at high obstacle densities (N≥10), where failure rates of 35–45% suggest the current setup is near its ceiling.

---