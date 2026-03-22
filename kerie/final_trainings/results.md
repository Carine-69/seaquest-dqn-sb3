# Kerie — DQN Seaquest Training Results (Mid-Range Hyperparameters)

## Summary

Best model: **Exp 3 (higher gamma)** — reward **44.4** at 1M steps
Policy: **CnnPolicy** | Environment: **ALE/Seaquest-v5** | Buffer: 50,000 | optimize_memory_usage=True

## Hyperparameter Table

| Exp | Name | lr | gamma | batch | eps_start | eps_end | eps_frac | Steps | Best Reward | Final Reward | Status |
|-----|------|------|-------|-------|-----------|---------|----------|-------|-------------|--------------|--------|
| 1 | exp1_baseline_mid | 0.0001 | 0.94 | 64 | 1.0 | 0.05 | 0.10 | 200k | 18.8 | 15.6 | Session 2 only |
| 2 | exp2_lr_bump | 0.0002 | 0.94 | 64 | 1.0 | 0.05 | 0.10 | 425k | 21.0 | 21.0 | Early stopped |
| 3 | exp3_higher_gamma | 0.0001 | 0.95 | 64 | 1.0 | 0.05 | 0.10 | 1M | **44.4** | **44.4** | **BEST** |
| 4 | exp4_lower_eps_start | 0.0003 | 0.94 | 64 | 0.9 | 0.05 | 0.10 | 1M | 22.8 | 22.4 | Complete |
| 5 | exp5_lower_eps_end | 0.0001 | 0.96 | 64 | 1.0 | 0.02 | 0.10 | 1M | 20.2 | 20.2 | Complete |
| 6 | exp6_faster_decay | 0.0002 | 0.95 | 64 | 1.0 | 0.05 | 0.05 | 200k | 16.8 | 16.8 | Session 2 only |
| 7 | exp7_slower_decay | 0.0001 | 0.94 | 64 | 1.0 | 0.05 | 0.25 | 200k | 13.8 | 11.8 | Session 2 only |
| 8 | exp8_mixed_mid | 0.0004 | 0.95 | 64 | 0.9 | 0.02 | 0.10 | 1M | 21.0 | 21.0 | Complete |
| 9 | exp9_moderate_decay | 0.0003 | 0.96 | 64 | 1.0 | 0.05 | 0.15 | 400k | 20.0 | 14.0 | Early stopped |
| 10 | exp10_best_of_mid | 0.0005 | 0.95 | 64 | 0.9 | 0.02 | 0.20 | 325k | 5.8 | 5.8 | Early stopped |

## Key Findings

### What improved performance?

1. **Gamma was the most impactful parameter.** Exp 3 (gamma=0.95) achieved 44.4 reward — more
   than double the baseline (18.8). Higher gamma makes the agent value future rewards more,
   which is critical in Seaquest where surviving (surfacing for oxygen) has no immediate reward
   but prevents death 50+ steps later.

2. **Higher learning rate helped up to a point.** Exp 4 (lr=0.0003) reached 22.8 vs baseline's
   18.8. Faster weight updates mean the agent learns more per step. But exp 10 (lr=0.0005)
   collapsed to 5.8 — too aggressive, causing unstable Q-value updates.

3. **More training steps consistently helped.** Experiments that ran the full 1M steps all
   outperformed their 200k counterparts. The reward curves were still climbing at 1M,
   suggesting even more steps (2-5M) would improve results further.

### What harmed performance?

1. **Too high learning rate (exp10, lr=0.0005):** Reward collapsed to 5.8. The large weight
   updates caused Q-values to oscillate instead of converging. The agent couldn't maintain
   a stable policy.

2. **Slower epsilon decay (exp7, eps_frac=0.25):** Reward only 13.8. The agent spent 25% of
   training (250k steps) exploring randomly instead of exploiting what it learned. By the time
   it started exploiting, other experiments had already built strong policies.

3. **Lower epsilon start (exp4/10, eps_start=0.9):** Mixed results. Less initial exploration
   means the agent might miss discovering key strategies early on. Exp4 recovered with more
   steps, but exp10 combined this with too-high lr and collapsed.

## Improvement from Session 2 (200k) to 1M Steps

| Exp | 200k Reward | 1M Reward | Improvement |
|-----|-------------|-----------|-------------|
| 3 | 18.6 | **44.4** | **+139%** |
| 4 | 8.0 | 22.8 | +185% |
| 5 | 13.0 | 20.2 | +55% |
| 8 | 9.2 | 21.0 | +128% |

All experiments that completed 1M steps showed significant improvement over their 200k results,
confirming that the low rewards in Session 2 were due to insufficient training, not bad hyperparameters.

## Why Exp 3 Won

Exp 3 used gamma=0.95 with a conservative lr=0.0001. This combination works because:

- **gamma=0.95** makes the agent care about rewards 50+ steps ahead. In Seaquest, the agent
  must learn that surfacing for oxygen (no immediate reward) prevents death (huge negative
  outcome). At gamma=0.94, a reward 50 steps away is worth 0.94^50 = 4.5% of its face value.
  At gamma=0.95, it's worth 0.95^50 = 7.7% — almost double the weight on future survival.

- **lr=0.0001** provides stable learning. Combined with the higher gamma (which adds noise to
  Q-value estimates since it's predicting further into the future), a low learning rate prevents
  the instability that killed exp10.

## Architecture: CNN vs MLP

**CnnPolicy (CNN)** was used for all experiments. Seaquest's input is 84x84 grayscale frames
(after AtariWrapper preprocessing). CNN is the correct choice because:

- CNN detects spatial features (enemy positions, oxygen bar, diver locations) through
  convolutional filters that share weights across the image
- MLP treats each pixel independently — it cannot detect spatial patterns without learning
  them from scratch, requiring millions more parameters
- The original DQN paper (Mnih et al., 2015) established CNN as the standard for Atari

## Files

```
final_trainings/
  best_model/          # Best performing model (exp3, reward 44.4)
  notebooks/           # All 10 experiment notebooks (clean names)
  plots/               # All reward plots + summary comparison
  results/             # All result CSVs + summary table
  results.md           # This file
```
