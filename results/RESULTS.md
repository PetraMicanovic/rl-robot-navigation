# Evaluation Results — v3.0-curriculum (Curriculum Training)

This document summarizes the results of four experiments run to evaluate a PPO-based robot navigation agent trained with the v3.0 curriculum. All experiments use 200 evaluation episodes with a deterministic policy.

Compared to v2.0-obs-reward, this version replaces the fixed two-phase training with a multi-stage curriculum that gradually increases the number and speed of obstacles across 5 stages.

---

## Training configuration

| Parameter | Value |
|---|---|
| Algorithm | PPO |
| Network | MLP `[256, 256]` |
| Observation space | 24 lidar rays, agent velocity, target direction, 3 nearest obstacle velocities |
| Action space | Continuous velocity control |
| Curriculum stages | 5 (N=0 -> N=3 -> N=3 -> N=6 -> N=10, speed increases from stage 2) |
| Total timesteps | ~ 3 500 000 |
---

## E1 — Effect of obstacle density

The curriculum-trained model was evaluated across three obstacle counts at fixed speed=1.0.

| Obstacles | Success | Collisions | Truncated | Avg steps | Avg col/ep | vs v2 |
|---|---|---|---|---|---|---|
| N=3 | 59.5% (119/200) | 40.5% | 0.0% | 5.1 | 0.405 | +38.5pp |
| N=6 (training) | 51.5% (103/200) | 48.5% | 0.0% | 4.6 | 0.485 | +37.0pp |
| N=10 | 42.5% (85/200) | 57.5% | 0.0% | 4.2 | 0.575 | +34.0pp |

![E1 metrics](figures/e1_metrics.png)

Results improved substantially compared to v2 across all three configurations. The gradual increase in difficulty during curriculum training likely contributed to this — the agent was first exposed to simpler scenarios before being evaluated on denser ones.

Average episode length dropped significantly compared to v2 (from ~20–40 steps down to ~4–5 steps), which means the agent is reaching the goal faster when it succeeds rather than wandering around.

![E1 training curves](figures/e1_training_curves.png)

---

## E2 — Effect of obstacle speed

The curriculum-trained model was evaluated across different speeds at fixed N=6.

| Speed | Success | Collisions | Truncated | Avg steps | Avg col/ep | vs v2 |
|---|---|---|---|---|---|---|
| 0.5 (slower) | 52.0% (104/200) | 48.0% | 0.0% | 4.8 | 0.480 | +51.0pp |
| 1.0 (training) | 51.5% (103/200) | 48.5% | 0.0% | 4.6 | 0.485 | +41.5pp |
| 1.5 (faster) | 51.0% (102/200) | 49.0% | 0.0% | 4.5 | 0.490 | +36.0pp |

![E2 metrics](figures/e2_metrics.png)

The most notable change compared to v2 is at speed=0.5, where success rate increased from 1.0% to 52.0%. In v2 this was attributed to the proximity penalty causing the agent to hesitate near slow obstacles. Including variable speed ranges in later curriculum stages appears to have addressed this. Performance is fairly consistent across all three speeds, which was not the case in previous versions.

![E2 training curves](figures/e2_training_curves.png)

---

## E3 — Generalization to unseen configurations

The curriculum-trained model was evaluated on seven configurations not seen during training.

| Obstacles | Speed | Success | Collisions | Truncated | Avg steps | Avg col/ep | vs v2 |
|---|---|---|---|---|---|---|---|
| 5 | 0.8 | 59.5% (119/200) | 40.5% | 0.0% | 4.7 | 0.405 | +45.0pp |
| 7 | 1.0 | 48.5% (97/200) | 51.5% | 0.0% | 4.9 | 0.515 | +34.5pp |
| 8 | 1.2 | 47.0% (94/200) | 53.0% | 0.0% | 4.2 | 0.530 | +37.0pp |
| 9 | 1.3 | 50.5% (101/200) | 49.5% | 0.0% | 4.3 | 0.495 | +43.5pp |
| 10 | 1.5 | 39.0% (78/200) | 61.0% | 0.0% | 4.0 | 0.610 | +33.0pp |
| 12 | 2.0 | 40.5% (81/200) | 59.5% | 0.0% | 3.9 | 0.595 | +35.0pp |
| 15 | 0.3 | 38.5% (77/200) | 61.5% | 0.0% | 3.9 | 0.615 | +32.0pp |

![E3 metrics](figures/e3_metrics.png)

Generalization improved across all seven configurations compared to v2, with gains ranging from +32pp to +45pp. The agent now handles high-density unseen configs (N=12, N=15) much better than before, likely because the curriculum included stages up to N=13 with variable speed. Even the previously problematic N=15/speed=0.3 config reaches 38.5%, up from 6.5% in v2.

---

## E4 — Effect of reward shaping

Both models trained via curriculum, evaluated at N=6, speed=1.0.

| Variant | Success | Collisions | Truncated | Avg steps | Avg col/ep | vs v2 |
|---|---|---|---|---|---|---|
| With shaping | 51.5% (103/200) | 48.5% | 0.0% | 4.6 | 0.485 | +44.0pp |
| Without shaping | 6.0% (12/200) | 94.0% | 0.0% | 3.1 | 0.940 | -1.5pp |

![E4 metrics](figures/e4_metrics.png)

The difference between the two variants is much larger here than in v2, where both ended at 7.5%. With shaping the agent reaches 51.5%, while without shaping it drops to 6.0%. One possible explanation is that the progress reward and proximity penalty play a more important role during curriculum training — without them, the agent may not receive enough signal to advance through the harder stages effectively.

---

## Overview — all training curves

![All eval curves](figures/all_eval_curves.png)

---

## Overall summary

| Experiment | Best result | Configuration | vs v2 |
|---|---|---|---|
| E1 | 59.5% | N=3, speed=1.0 | +38.5pp |
| E2 | 52.0% | N=6, speed=0.5 | +51.0pp |
| E3 | 59.5% | N=5, speed=0.8 | +45.0pp |
| E4 | 51.5% | shaping, N=6, speed=1.0 | +44.0pp |

Overall, v3 shows the largest improvement compared to previous versions. The slow-obstacle regression from v2 is no longer present, and generalization results are consistently better. The E4 result is worth noting — reward shaping had no measurable effect in v2 but shows a large effect here, which suggests that the curriculum training changes how the agent uses the shaping signal, though the exact reason is not entirely clear.

The main remaining challenge is that collision rates are still relatively high (40–60%) even for the best configurations, so there is room for further improvement. 

---