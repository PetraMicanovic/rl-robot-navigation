# Evaluation Results — v4.0-action-smoothing

This document summarizes the results of four experiments evaluating a PPO-based robot navigation agent trained with the v4.0 configuration. All experiments use 200 evaluation episodes with a deterministic policy.

The only change compared to v3.0-curriculum is the addition of action smoothing:

```json
"action_smoothing_scale": 0.16
```

This adds a penalty to the reward for large changes in action between consecutive steps. The idea is to discourage erratic movement and encourage the agent to commit to smoother trajectories. Several values were tested for this parameter; 0.16 gave the best results and is used here. Overall, the results are noticeably better than v3 across all experiments.

---

## Training configuration

| Parameter | Value |
|---|---|
| Algorithm | PPO |
| Network | MLP `[256, 256]` |
| Observation space | 24 lidar rays, agent velocity, target direction, 3 nearest obstacle velocities |
| Action space | Continuous velocity control |
| Curriculum stages | 10 (N=0 -> N=2 -> N=3 -> ... -> N=13, with variable speed from stage 4) |
| Action smoothing scale | 0.16 |
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

All three configurations improved by roughly 20–27pp over v3. N=3 reaches 86.5%, which is the best single result across all versions so far. The gains are fairly uniform across density levels, which suggests the smoothing is helping in a general way rather than being specific to any particular obstacle count.

Episode length stayed in the same range as v3 (~4–5 steps), so the agent isn't just taking slower, more cautious paths — it's still reaching the goal at roughly the same speed.

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

Performance is fairly consistent across speeds, which was already the case in v3. The improvement over v3 is large across the board, especially at speed=0.5 (+25.5pp) and speed=1.5 (+24.5pp).

One slightly unexpected result is that speed=1.5 (75.5%) outperforms speed=1.0 (71.5%), which is the training condition. This didn't happen in v3. The difference is small enough that it might just be noise, but it's worth keeping an eye on in future runs.

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

Generalization improved across all seven configurations, with gains between +14.0pp and +23.0pp. N=15/speed=0.3, which was the hardest configuration in previous versions (6.5% in v2, 38.5% in v3), now reaches 57.0%. Similarly, N=12/speed=2.0 crossed 50% for the first time (56.5%).

The smallest gain is N=9/speed=1.3 at +14.0pp. It's not obvious why this configuration improved less than the others — it falls in the middle of the density range and has a moderate speed, so there's no clear reason it should be harder. It still improved (50.5% → 64.5%), just less than expected.

---

## E4 — Effect of reward shaping

Both variants trained with action smoothing enabled, evaluated at N=6, speed=1.0.

| Variant | Success | Collisions | Truncated | Avg steps | Avg col/ep | vs v3 |
|---|---|---|---|---|---|---|
| With shaping | 71.5% (143/200) | 28.5% | 0.0% | 4.9 | 0.285 | +20.0pp |
| Without shaping | 28.5% (57/200) | 71.5% | 0.0% | 6.2 | 0.715 | +22.5pp |

![E4 metrics](figures/e4_metrics.png)

The with-shaping variant improves by 20pp over v3, consistent with the other experiments. The more interesting result is the no-shaping variant: it goes from 6.0% in v3 to 28.5% here. In v3 (and v2 before it), training without shaping produced essentially random behavior. Here it actually learns something, which suggests the smoothing penalty is providing some useful training signal on its own.

That said, 28.5% vs 71.5% is still a large gap, so the reward shaping is clearly doing most of the work. The two seem to work well together rather than one replacing the other.

---

## Overview — all training curves

![All eval curves](figures/all_eval_curves.png)

---

## Overall summary

| Experiment | Best result | Configuration | vs v3 |
|---|---|---|---|
| E1 | 86.5% | N=3, speed=1.0 | +27.0pp |
| E2 | 77.5% | N=6, speed=0.5 | +25.5pp |
| E3 | 78.5% | N=5, speed=0.8 | +19.0pp |
| E4 | 71.5% | shaping, N=6, speed=1.0 | +20.0pp |

v4 is the best version so far. Adding `action_smoothing_scale=0.16` gave consistent gains of around +20pp over v3 across all experiments, and collision rates dropped from the 40–60% range in v3 down to 15–45% depending on the configuration.

The E4 no-shaping result (6% → 28.5%) is worth investigating further — it's not clear whether this is a direct effect of the smoother actions making the task easier to learn, or whether the smoothing penalty is acting as an implicit regularizer that helps even in the absence of explicit reward shaping.

The main remaining issue is that harder configurations (N≥10 or high speed) still fail 35–45% of the time. Several other values of `action_smoothing_scale` were tested prior to this, but 0.16 gave the best results — both lower and higher values led to noticeably worse performance, so this appears to be close to optimal for the current setup.

---