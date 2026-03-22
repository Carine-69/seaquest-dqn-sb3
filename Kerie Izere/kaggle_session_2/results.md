# Kaggle Session 2 — Results (Mid-Range Hyperparameters)

Training DQN with CnnPolicy on Seaquest (ALE/Seaquest-v5).
Used `optimize_memory_usage=True` + `handle_timeout_termination=False` to avoid OOM.
Buffer size: 50,000 | Eval every 25k steps | Early stopping after 3 no-improvement evals.

## What improved vs Session 1 (200k, incomplete)

Session 1 only completed 8 out of 10 experiments (exp9 and exp10 crashed from OOM).
Session 2 completed **all 10 experiments** thanks to memory optimizations.

Two experiments trained beyond 200k via resume:
- **Exp 9** reached 400k steps (best reward: 20.0 at 300k)
- **Exp 10** reached 325k steps (best reward: 6.0 at 225k)

Exps 1-8 ran 200k steps (same as session 1, early stopping kicked in).

## Results Summary

| Exp | Name | lr | gamma | eps_frac | Steps | Best Reward | Final Reward | vs Session 1 |
|-----|------|-----|-------|----------|-------|-------------|--------------|--------------|
| 1 | baseline_mid | 0.0001 | 0.94 | 0.10 | 200k | 18.8 | 15.6 | same |
| 2 | lr_bump | 0.0002 | 0.94 | 0.10 | 200k | 15.4 | 15.4 | same |
| 3 | higher_gamma | 0.0001 | 0.95 | 0.10 | 200k | 18.6 | 18.6 | same |
| 4 | lower_eps_start | 0.0003 | 0.94 | 0.10 | 200k | 8.0 | 8.0 | same |
| 5 | lower_eps_end | 0.0001 | 0.96 | 0.10 | 200k | 13.0 | 13.0 | same |
| 6 | faster_decay | 0.0002 | 0.95 | 0.05 | 200k | 16.8 | 16.8 | same |
| 7 | slower_decay | 0.0001 | 0.94 | 0.25 | 200k | 13.8 | 11.8 | same |
| 8 | mixed_mid | 0.0004 | 0.95 | 0.10 | 200k | 9.2 | 9.2 | same |
| 9 | moderate_decay_high_gamma | 0.0003 | 0.96 | 0.15 | **400k** | **20.0** | 14.0 | NEW (was incomplete) |
| 10 | best_of_mid | 0.0005 | 0.95 | 0.20 | **325k** | 6.0 | 5.8 | NEW (was incomplete) |

## Key Observations

1. **Best model: Exp 9** (reward 20.0 at 300k steps) — moderate decay + high gamma (0.96) performed best
2. **Exp 1 and Exp 3** are close behind at 18.8 and 18.6 — conservative lr (0.0001) works well
3. **Exp 10 underperformed** despite the most steps (325k) — lr=0.0005 + low eps_start=0.9 + low eps_end=0.02 was too aggressive
4. **Early stopping** kicked in for exps 1-8 around 200k — they plateaued and need more total timesteps (500k+) to break through
5. All rewards are still low (max 20.0) — agent barely shoots enemies, hasn't learned oxygen surfacing or diver rescuing yet
6. **Next step:** resume all experiments to 500k-1M steps to push rewards higher

## Ranking (by best reward)

1. Exp 9 — 20.0 (moderate_decay_high_gamma)
2. Exp 1 — 18.8 (baseline_mid)
3. Exp 3 — 18.6 (higher_gamma)
4. Exp 6 — 16.8 (faster_decay)
5. Exp 2 — 15.4 (lr_bump)
6. Exp 7 — 13.8 (slower_decay)
7. Exp 5 — 13.0 (lower_eps_end)
8. Exp 8 — 9.2 (mixed_mid)
9. Exp 4 — 8.0 (lower_eps_start)
10. Exp 10 — 6.0 (best_of_mid)

## Plots

All plots are in the `plots/` folder:
- Individual reward curves: `plots/exp{N}_{name}_reward_plot.png`
- Summary comparison: `plots/summary_all_experiments.png`
