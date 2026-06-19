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

| Obstacles | Success | Collisions | Truncated | Avg steps | Avg col/ep | vs v4 |
|---|---|---|---|---|---|---|
| N=3 | 86.5% (173/200) | 13.5% | 0.0% | 5.7 | 0.135 | +0.0pp |
| N=6 (training) | 72.0% (144/200) | 28.0% | 0.0% | 5.0 | 0.280 | +0.5pp |
| N=10 | 67.0% (134/200) | 33.0% | 0.0% | 5.1 | 0.330 | +5.0pp |

![E1 metrics](figures/e1_metrics.png)

Results are largely in line with v4. N=3 is essentially identical (86.5% both times), which is reassuring. N=6 went from 71.5% to 72.0% — basically the same. The more notable change is N=10, which improved from 62.0% to 67.0% (+5.0pp). Whether this is a meaningful gain or just run-to-run variance is hard to say with 200 episodes; it would be worth re-evaluating with more episodes to see if it holds.

Episode lengths are slightly longer at N=3 (5.7 vs 5.5 in v4), which could mean the agent is being a bit more cautious in low-density environments. Not a concern, just something to keep an eye on.

![E1 training curves](figures/e1_training_curves.png)

---

## E2 — Effect of obstacle speed

Evaluated at fixed N=6 across three speeds.

| Speed | Success | Collisions | Truncated | Avg steps | Avg col/ep | vs v4 |
|---|---|---|---|---|---|---|
| 0.5 (slower) | 80.5% (161/200) | 19.5% | 0.0% | 5.4 | 0.195 | +3.0pp |
| 1.0 (training) | 72.0% (144/200) | 28.0% | 0.0% | 5.0 | 0.280 | +0.5pp |
| 1.5 (faster) | 74.5% (149/200) | 25.5% | 0.0% | 5.2 | 0.255 | -1.0pp |

![E2 metrics](figures/e2_metrics.png)

The pattern from v4 holds: performance is fairly consistent across speeds, and the agent doesn't seem to struggle notably more at speed=1.5 than at the training condition. The training speed (1.0) and fast speed (1.5) are within ~2pp of each other, which is consistent with what we saw before.

Speed=0.5 gained +3.0pp over v4 and is now at 80.5%, which is the best result in this experiment. Speed=1.5 dropped by 1.0pp, which is negligible. The observation from v4 that speed=1.5 slightly outperforms the training speed is still present here, which makes it slightly less likely to be just noise — though the margin (74.5% vs 72.0%) is still small enough that we shouldn't read too much into it.

![E2 training curves](figures/e2_training_curves.png)

---

## E3 — Generalization to unseen configurations

Evaluated on seven configurations not seen during training.

| Obstacles | Speed | Success | Collisions | Truncated | Avg steps | Avg col/ep | vs v4 |
|---|---|---|---|---|---|---|---|
| 5 | 0.8 | 81.5% (163/200) | 18.5% | 0.0% | 5.5 | 0.185 | +3.0pp |
| 7 | 1.0 | 73.0% (146/200) | 27.0% | 0.0% | 5.0 | 0.270 | +4.5pp |
| 8 | 1.2 | 66.0% (132/200) | 34.0% | 0.0% | 5.0 | 0.340 | -4.0pp |
| 9 | 1.3 | 67.0% (134/200) | 33.0% | 0.0% | 4.9 | 0.330 | +2.5pp |
| 10 | 1.5 | 63.5% (127/200) | 36.5% | 0.0% | 4.8 | 0.365 | +3.0pp |
| 12 | 2.0 | 55.5% (111/200) | 44.5% | 0.0% | 4.4 | 0.445 | -1.0pp |
| 15 | 0.3 | 60.5% (121/200) | 39.5% | 0.0% | 4.7 | 0.395 | +3.5pp |

![E3 metrics](figures/e3_metrics.png)

Most configurations are within a few pp of their v4 values, which is what we'd expect from a rerun. The clearest outlier is N=8/speed=1.2, which dropped by 4.0pp (from 70.0% to 66.0%). This is on the larger end of what you'd expect from noise with 200 episodes, so it's at least worth flagging, but not necessarily something to worry about yet.

On the positive side, N=5/speed=0.8 (+3.0pp), N=7/speed=1.0 (+4.5pp), and N=15/speed=0.3 (+3.5pp) all improved slightly. The hard configurations (N≥10) remain in the 55–67% range, consistent with v4.

N=12/speed=2.0 dropped marginally from 56.5% to 55.5%. It's still above 50%, which was already a milestone in v4.

---

## E4 — Effect of reward shaping

Both variants trained with action smoothing enabled, evaluated at N=6, speed=1.0.

| Variant | Success | Collisions | Truncated | Avg steps | Avg col/ep | vs v4 |
|---|---|---|---|---|---|---|
| With shaping | 72.0% (144/200) | 28.0% | 0.0% | 5.0 | 0.280 | +0.5pp |
| Without shaping | 29.5% (59/200) | 70.5% | 0.0% | 4.6 | 0.705 | +1.0pp |

![E4 metrics](figures/e4_metrics.png)

Both variants reproduced closely. The shaping variant is at 72.0% (vs 71.5% in v4) and the no-shaping variant at 29.5% (vs 28.5%). The gap between them is essentially the same as before (~42pp), which confirms that reward shaping is doing most of the heavy lifting and the smoothing penalty alone isn't sufficient.

The no-shaping result being reproducibly around 29% (as opposed to the ~6% seen in v3) does strengthen the case that action smoothing is providing some meaningful training signal on its own. Whether that's because smoother trajectories are inherently easier to learn from, or because the penalty is acting as an implicit regularizer, is still an open question.

---

## Overview — all training curves

![All eval curves](figures/all_eval_curves.png)

---

## Overall summary

| Experiment | Best result | Configuration | vs v4 |
|---|---|---|---|
| E1 | 86.5% | N=3, speed=1.0 | +0.0pp |
| E2 | 80.5% | N=6, speed=0.5 | +3.0pp |
| E3 | 81.5% | N=5, speed=0.8 | +3.0pp |
| E4 | 72.0% | shaping, N=6, speed=1.0 | +0.5pp |

Overall, the results reproduce well. The gains over v4 are small (0–5pp in most places), which is expected given that the configuration is unchanged — this was always meant to be a consistency check rather than an improvement.

The one configuration that went in the other direction (N=8/speed=1.2 in E3, -4.0pp) is worth monitoring but isn't alarming on its own. Run-to-run variance with 200 episodes can easily produce differences of this size.

The next step would probably be to try a different value for `action_smoothing_scale`, or to look at whether some form of curriculum adjustment could help with the harder configurations where the agent is still failing 35–45% of the time.

---