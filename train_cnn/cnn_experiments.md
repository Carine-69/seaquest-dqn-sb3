# CNN Policy Hyperparameter Experiments

Environment: `ALE/Seaquest-v5`  
Policy: `CnnPolicy`  
Total Timesteps per Experiment: `1,000,000`  
Fixed Parameters: `buffer_size=100000`, `learning_starts=10000`, `target_update_interval=1000`

## Hyperparameter Table

| # | lr | gamma | batch | eps_start | eps_end | eps_frac | mean_reward | best_reward | final_reward | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.00001 | 0.90 | 16 | 1.0 | 0.05 | 0.10 | 5.02 | 19.0 | 9.4 | Baseline low — very slow learning, stable |
| 2 | 0.00002 | 0.91 | 16 | 1.0 | 0.05 | 0.10 | 5.87 | 24.0 | 8.2 | Slight lr bump — marginally faster convergence |
| 3 | 0.00003 | 0.90 | 32 | 1.0 | 0.05 | 0.10 | 10.99 | 29.0 | 23.6 | Bigger batch — more stable gradient updates |
| 4 | 0.00001 | 0.92 | 16 | 0.8 | 0.05 | 0.10 | 6.97 | 21.0 | 10.6 | Lower eps_start — less early exploration |
| 5 | 0.00005 | 0.91 | 32 | 1.0 | 0.01 | 0.10 | 15.44 | 30.0 | 26.6 | Lower eps_end — more exploitation late game |
| 6 | 0.00001 | 0.93 | 16 | 1.0 | 0.05 | 0.05 | 5.73 | 20.0 | 8.8 | Faster decay — exploitation kicks in early |
| 7 | 0.00002 | 0.92 | 32 | 1.0 | 0.05 | 0.20 | 10.77 | 21.0 | 17.2 | Slower decay — longer exploration phase |
| 8 | 0.00003 | 0.93 | 16 | 0.9 | 0.01 | 0.10 | 11.12 | 22.0 | 19.8 | Mixed low — observe combined effect |
| 9 | 0.00004 | 0.90 | 32 | 1.0 | 0.05 | 0.15 | 11.68 | 33.0 | 16.0 | Moderate decay — balanced explore/exploit |
| 10 | 0.00005 | 0.93 | 32 | 0.8 | 0.01 | 0.10 | 14.57 | 22.0 | 20.6 | Best-of-low — final low-range candidate |

## Key Insights

### What improved performance
- **Larger batch size (32 vs 16):** Experiments 3, 5, 7, 9, 10 all used batch=32 and consistently scored higher than batch=16 experiments. More stable gradient updates help the CNN learn better representations.
- **Lower eps_end:** Experiments 5 and 8 used eps_end=0.01, meaning the agent exploits more aggressively in later training. Both showed strong final rewards (26.6 and 19.8).
- **Higher learning rate:** Experiment 5 (lr=0.00005) combined with batch=32 and low eps_end produced the best final reward of 26.6.

### What harmed performance
- **Very low lr + small batch:** Experiment 1 (lr=0.00001, batch=16) was the worst performer with mean=5.02. The network learned too slowly to make progress in 1M steps.
- **Faster epsilon decay (eps_frac=0.05):** Experiment 6 shifted to exploitation too early before the agent had learned enough, resulting in mean=5.73.
- **Lower eps_start (0.8):** Experiment 4 reduced early exploration which hurt initial learning, giving mean=6.97.

### Best configuration
**Experiment 9** achieved the highest single reward of **33.0**:
- `lr=0.00004`, `gamma=0.90`, `batch=32`
- `eps_start=1.0`, `eps_end=0.05`, `eps_frac=0.15`
- The moderate epsilon decay (0.15) allowed a longer exploration phase before exploitation, helping the agent discover better strategies.

## Notes on Overall Performance
All experiments used 1,000,000 timesteps which is conservative for Seaquest with CNNPolicy. Scores are low range by design and increasing timesteps to 5-10 M may increase learning,the purpose of this experiment set is to establish a performance baseline at conservative hyperparameters. Higher learning rates and larger batchesare expected to yield significantly better scores.
