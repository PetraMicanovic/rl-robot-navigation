# Evaluation Results — Baseline (v1.0)

This document summarizes the results of four experiments conducted to evaluate the performance of a PPO-based robot navigation agent. The goal was to understand how the agent behaves under different environment conditions and whether reward shaping has a measurable effect on performance. All experiments were run with 200 evaluation episodes using a deterministic policy.

---

## Training configuration

The base model was trained with the following settings:

| Parameter | Value |
|---|---|
| Algorithm | PPO |
| Network | MLP `[256, 256]` |
| Observation space | 8 lidar rays, agent velocity, target direction |
| Action space | Continuous velocity control |
| Base training config | N=6 obstacles, speed=1.0 |
| Total timesteps | 3,000,000 |

---

## E1 — Effect of obstacle density

The model trained on N=6, speed=1.0 was evaluated across three obstacle counts while keeping speed fixed at 1.0. The goal was to see how well the agent handles environments with more or fewer obstacles than it was trained on.

| Obstacles | Success | Collisions | Truncated | Avg steps | Avg col/ep |
|---|---|---|---|---|---|
| N=3 | 13.0% (26/200) | 87.0% | 0.0% | 52.4 | 0.870 |
| N=6 (training) | 8.0% (16/200) | 92.0% | 0.0% | 27.5 | 0.920 |
| N=10 | 6.0% (12/200) | 94.0% | 0.0% | 13.3 | 0.940 |

![E1 metrics](figures/e1_metrics.png)

The results show a clear drop in performance as obstacle density increases. What is perhaps more telling is the average episode length — it falls from 52.4 steps at N=3 down to just 13.3 at N=10. This suggests the agent is not actively navigating around obstacles but rather colliding with them sooner as the environment becomes more crowded.

It is also worth noting that even at N=6, which is the exact training configuration, success rate is only 8%. This suggests that the agent may not have learned a general navigation strategy but rather something much more specific to the conditions it was trained under.

![E1 training curves](figures/e1_training_curves.png)

Looking at the training curves, the N=0 model learns reasonably well — it converges and stays around −5.2 for most of training. The other three models, trained with dynamic obstacles, basically do not improve at all over 3M timesteps. They all stay between −9 and −10 with no clear trend. This suggests that simply extending training might not be sufficient and that some aspect of the setup may need to change.

---

## E2 — Effect of obstacle speed

Model trained on N=6, speed=1.0. Evaluated on different speeds at fixed N=6.

| Speed | Success | Collisions | Truncated | Avg steps | Avg col/ep |
|---|---|---|---|---|---|
| 0.5 (slower) | 8.0% (16/200) | 92.0% | 0.0% | 51.4 | 0.920 |
| 1.0 (training) | 9.5% (19/200) | 90.5% | 0.0% | 25.8 | 0.905 |
| 1.5 (faster) | 17.5% (35/200) | 82.5% | 0.0% | 11.2 | 0.825 |

![E2 metrics](figures/e2_metrics.png)

The most interesting finding here is that the agent actually performs better with faster obstacles. At speed=1.5, success rate reaches 17.5%, which is more than double the 8.0% achieved with speed=0.5. This seems counterintuitive at first, but it makes sense when you consider what strategy the agent has likely learned — it appears to rush toward the goal in a fairly direct path rather than actively avoiding obstacles. When obstacles move faster, they clear out of the way more quickly, which happens to work in the agent's favor. Slower obstacles stay in the path longer and the agent has no strategy for waiting or re-routing.

The drop in average episode length from 51.4 to 11.2 steps further supports this interpretation — at speed=1.5 the agent either reaches the goal or collides very quickly.

![E2 training curves](figures/e2_training_curves.png)

In the training curves, speed=1.5 stands out as the only one that actually improves over time, ending around −6.6. The other two are mostly flat. This lines up with the evaluation results and again suggests the agent is benefiting from the obstacle dynamics rather than learning to avoid them.

---

## E3 — Generalization to unseen configurations

To test how well the trained model generalizes, it was evaluated on seven configurations that were not seen during training. These vary in both obstacle count and speed.

| Obstacles | Speed | Success | Collisions | Truncated | Avg steps | Avg col/ep |
|---|---|---|---|---|---|---|
| 5 | 0.8 | 14.0% (28/200) | 85.5% | 0.5% | 44.1 | 0.855 |
| 7 | 1.0 | 10.0% (20/200) | 90.0% | 0.0% | 20.8 | 0.900 |
| 8 | 1.2 | 8.0% (16/200) | 92.0% | 0.0% | 15.6 | 0.920 |
| 9 | 1.3 | 9.0% (18/200) | 91.0% | 0.0% | 15.4 | 0.910 |
| 10 | 1.5 | 4.0% (8/200) | 96.0% | 0.0% | 10.6 | 0.960 |
| 12 | 2.0 | 5.5% (11/200) | 94.5% | 0.0% | 8.1 | 0.945 |
| 15 | 0.3 | 6.5% (13/200) | 93.5% | 0.0% | 25.2 | 0.935 |

![E3 metrics](figures/e3_metrics.png)

The dashed line marks the training-configuration baseline (8%). Bar colors indicate whether a configuration performs above (green), near (grey), or below (coral) that baseline. Generalization is reasonable close to the training distribution but degrades quickly as configurations become more extreme. The best results are on N=5/speed=0.8 (14%) and N=7/speed=1.0 (10%), both of which are close to the N=6/speed=1.0 training setup. At N=10/speed=1.5 and N=12/speed=2.0, success rate drops to 4–5.5%.

One slightly unexpected result is N=15/speed=0.3, which achieves 6.5% despite having the most obstacles. The slower speed means episodes last longer on average (25.2 steps vs 8.1 for N=12/speed=2.0), which gives the agent more time to incidentally reach the goal. This again points to the agent relying on something closer to luck than a learned avoidance strategy.

---

## E4 — Effect of reward shaping

Both models start from a phase-1 pretrained model (N=0, speed=1.0) and are fine-tuned on N=6, speed=1.0 for 3,000,000 timesteps. The only difference is whether a progress reward toward the target is included.

| Variant | Success | Collisions | Truncated | Avg steps | Avg col/ep |
|---|---|---|---|---|---|
| With shaping | 9.5% (19/200) | 90.5% | 0.0% | 22.1 | 0.905 |
| Without shaping | 7.5% (15/200) | 92.5% | 0.0% | 26.9 | 0.925 |

![E4 metrics](figures/e4_metrics.png)

Adding the progress reward does help — success rate goes from 7.5% to 9.5% and collisions per episode drop slightly as well. The shaping model also has shorter average episodes (22.1 vs 26.9 steps), which likely means it moves more directly toward the goal instead of wandering. That said, the overall numbers are still quite low, suggesting that reward shaping alone is not sufficient to achieve reliable navigation behavior.

---

## Overview — all training curves

![All eval curves](figures/all_eval_curves.png)

This plot shows all training runs together. The N=0 model is clearly different from the rest — it shows clear convergence, while all obstacle configurations remain largely flat throughout training. Among those, speed=1.5 is the only one that improves noticeably in the second half, which matches what we saw in the evaluation.

---

## Overall summary

| Experiment | Best result | Configuration |
|---|---|---|
| E1 | 13.0% | N=3, speed=1.0 |
| E2 | 17.5% | N=6, speed=1.5 |
| E3 | 14.0% | N=5, speed=0.8 |
| E4 | 9.5% | N=6, speed=1.0 + shaping |

Across all experiments, the agent never gets above 17.5% success rate. The main issue seems to be that the agent never fully learns to avoid obstacles — it just tries to reach the goal as fast as possible and occasionally makes it through.

---

**Identified limitations**

With only 8 lidar rays, the agent has 45° blind spots where obstacles can approach without being detected. The observation space does not include obstacle velocity, so the agent has no way to predict where an obstacle is heading. The reward signal only distinguishes between success and collision, with no intermediate feedback for near-miss situations, which makes it hard for the agent to learn that staying far from obstacles is generally beneficial. Finally, training on a single fixed configuration (N=6, speed=1.0) makes the policy brittle to distribution shift, as the E3 results clearly show.

---